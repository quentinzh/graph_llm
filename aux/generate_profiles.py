#!/usr/bin/env python
"""Fast heuristic structured user-profile generator (no LLM)."""

from __future__ import annotations

import argparse
import math
import pickle
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_llm.config.datasets import resolve_dataset_paths
from graph_llm.dataload.legacy_data import read_split_indices
from graph_llm.metrics.metrics import DEFAULT_STOP_TOKENS

BRAND_SPARSE_SUFFIX = (
    " Brand evidence is sparse, so these should be treated as supporting "
    "rather than dominant signals."
)
PROFILE_PREFIX = "User interests:"
TOKEN_RE = re.compile(r"[a-z0-9']+")

# Review/template keywords: keep contractions like "hes"/"wasnt" by using a
# smaller stop set and dropping generic movie-domain terms.
MINIMAL_STOP_TOKENS = {
    "a", "an", "and", "the", "of", "to", "in", "on", "at", "by", "for", "from", "as",
    "or", "is", "it", "its", "with", "that", "this", "are", "be", "was", "were",
}
REVIEW_DOMAIN_STOP_TOKENS = {
    "movie", "movies", "film", "films", "great", "good", "dvd", "video", "show", "shows",
}
REVIEW_STOP_TOKENS = MINIMAL_STOP_TOKENS | REVIEW_DOMAIN_STOP_TOKENS

# Description keywords keep pronouns but drop generic plot/filler terms seen in
# Amazon item descriptions. Legacy profiles also only scan the opening text.
DESCRIPTION_DOMAIN_STOP_TOKENS = {
    "film", "films", "movie", "movies", "video", "dvd", "show", "shows",
    "high", "into", "about", "all", "one", "more", "when", "what", "which",
    "so", "also", "well", "very", "most", "many", "much", "even", "just",
    "only", "over", "out", "up", "down", "off", "back", "through", "between",
    "story", "stories", "series", "season", "episode", "episodes",
    "he", "she", "comedy", "girl", "play", "finds", "after", "along",
    "than", "their", "has", "michael",
}
DESCRIPTION_STOP_TOKENS = (DEFAULT_STOP_TOKENS - {
    "his", "her", "but", "he", "she", "him", "was", "were", "we", "they", "them",
    "their", "there", "it's",
}) | DESCRIPTION_DOMAIN_STOP_TOKENS
DESCRIPTION_MAX_WORDS = 80

TITLE_STOP_TOKENS = DEFAULT_STOP_TOKENS | {
    "movie", "movies", "film", "films", "new",
}


def _tokenize_text(text, stop_tokens=None, *, drop_digits: bool = False):
    if stop_tokens is None:
        stop_tokens = DEFAULT_STOP_TOKENS
    text = "" if text is None else str(text).lower()
    tokens = []
    for token in TOKEN_RE.findall(text):
        if not token or token in stop_tokens:
            continue
        if drop_digits and token.isdigit():
            continue
        tokens.append(token)
    return tokens


def _count_list(counter: Counter, limit: int = 50):
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [{"name": name, "count": count} for name, count in ranked[:limit]]


def _leaf_category(meta: dict) -> str | None:
    categories = meta.get("categories")
    if categories is None:
        return None
    if isinstance(categories, str):
        parts = [part.strip() for part in categories.split(",") if part.strip()]
        return parts[-1] if parts else None
    if isinstance(categories, list) and categories:
        leaf = categories[-1]
        if isinstance(leaf, list) and leaf:
            return str(leaf[-1]).strip() or None
        return str(leaf).strip() or None
    return None


def _profile_category(meta: dict) -> str | None:
    category = _leaf_category(meta)
    if category in {"TV", "Movies & TV"}:
        return "Movies"
    return category


def _item_title(meta: dict, raw_item: str) -> str:
    for key in ("title", "name"):
        value = meta.get(key)
        if value:
            return str(value).strip()
    return str(raw_item)


def _item_price(meta: dict):
    price = meta.get("price")
    if price is None:
        return None
    try:
        value = float(price)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        return None
    return None


def _join_names(items, max_names=3):
    names = [item["name"] for item in items[:max_names] if item.get("name")]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _join_keywords(items, max_names=8):
    names = [item["name"] for item in items[:max_names] if item.get("name")]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _brand_section(brands: list[dict], brand_coverage: float) -> str:
    if not brands:
        return (
            "Preferred brands/studios: Brand/studio metadata is too sparse to "
            "identify a reliable brand preference."
        )
    if len(brands) == 1:
        return (
            "Preferred brands/studios: Brand/studio metadata is sparse, but "
            f"{brands[0]['name']} appears among the available records."
        )
    max_brand_count = max(brand["count"] for brand in brands)
    if max_brand_count <= 1 and brand_coverage < 0.1:
        return (
            "Preferred brands/studios: Brand/studio metadata is sparse, but "
            f"{_join_names(brands, max_names=3)} appears among the available records."
        )
    line = (
        "Preferred brands/studios: The strongest available brand/studio signals are "
        f"{_join_names(brands)}."
    )
    line += BRAND_SPARSE_SUFFIX
    return line


def _category_section(categories: list[dict]) -> str:
    if not categories:
        return "Preferred genres/categories: Category metadata is sparse for this user."
    line = (
        "Preferred genres/categories: The user shows recurring interest in "
        f"{_join_names(categories)}."
    )
    if len(categories) > 3:
        line += (
            " Secondary category signals include "
            f"{_join_names(categories[3:], max_names=10)}."
        )
    return line


def _compose_llama_profile(evidence: dict) -> str:
    sections = []
    categories = evidence.get("category_counts", [])
    brands = evidence.get("brand_counts", [])
    desc_kw = evidence.get("description_keyword_counts", [])
    title_kw = evidence.get("title_keyword_counts", [])
    review_kw = evidence.get("review_keyword_counts", [])
    brand_coverage = evidence.get("brand_coverage", 0.0)

    sections.append(_category_section(categories))
    sections.append(_brand_section(brands, brand_coverage))

    theme_parts = []
    if desc_kw:
        theme_parts.append("description themes such as " + _join_keywords(desc_kw, max_names=8))
    if title_kw:
        theme_parts.append("title signals such as " + _join_keywords(title_kw, max_names=5))
    if review_kw:
        theme_parts.append(
            "review/template signals such as " + _join_keywords(review_kw, max_names=5)
        )
    if theme_parts:
        sections.append(
            "Recurring themes: The recurring signals include "
            + "; ".join(theme_parts)
            + "."
        )
    else:
        sections.append(
            "Recurring themes: No strong recurring lexical signals are available."
        )

    summary_bits = []
    if categories:
        summary_bits.append(_join_names(categories))
    if desc_kw:
        summary_bits.append(_join_keywords(desc_kw, max_names=8))
    if len(brands) >= 2:
        summary_bits.append(_join_names(brands))
    summary_text = "; ".join(bit for bit in summary_bits if bit)
    sections.append(
        "Overall preference summary: Overall, the user appears to prefer "
        f"{summary_text or 'the available interaction signals'}."
    )
    return "\n".join(sections)


def _compose_profile_text(llama_profile: str) -> str:
    return f"{PROFILE_PREFIX}\n{llama_profile}"


def _description_keyword_counter(description: str) -> Counter:
    counter = Counter()
    seen = set()
    text = "" if description is None else str(description)
    if DESCRIPTION_MAX_WORDS > 0:
        text = " ".join(text.split()[:DESCRIPTION_MAX_WORDS])
    for token in _tokenize_text(text, DESCRIPTION_STOP_TOKENS):
        if len(token) < 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        counter[token] += 1
    return counter


def _review_text_parts(row: dict) -> list[str]:
    parts = [str(row.get("predicted") or "")]
    template = row.get("template")
    if isinstance(template, (list, tuple)):
        if len(template) >= 1:
            parts.append(str(template[0]))
        if len(template) >= 2:
            parts.append(str(template[1]))
    return parts


def _aggregate_user_profile(
    interactions: list[dict],
    item_meta: dict,
    *,
    fold: str,
    scope: str,
    top_price_items_n: int = 5,
    evidence_limit: int = 50,
):
    category_counter = Counter()
    brand_counter = Counter()
    desc_counter = Counter()
    title_counter = Counter()
    review_counter = Counter()
    rating_counter = Counter()
    prices = []
    priced_items = []

    for row in interactions:
        raw_item = str(row["raw_item"])
        meta = item_meta.get(raw_item, {})
        category = _profile_category(meta)
        if category:
            category_counter[category] += 1
        brand = meta.get("brand")
        if brand:
            brand_counter[str(brand).strip()] += 1
        title = _item_title(meta, raw_item)
        description = meta.get("description") or ""
        desc_counter.update(_description_keyword_counter(description))
        title_counter.update(
            _tokenize_text(title, TITLE_STOP_TOKENS, drop_digits=True)
        )
        for part in _review_text_parts(row):
            review_counter.update(_tokenize_text(part, REVIEW_STOP_TOKENS))
        rating = row.get("rating")
        if rating is not None:
            rating_counter[str(int(rating))] += 1
        price = _item_price(meta)
        if price is not None:
            prices.append(price)
            priced_items.append({"item": raw_item, "title": title, "price": price})

    num_interactions = len(interactions)
    brand_coverage = (
        sum(1 for row in interactions if item_meta.get(str(row["raw_item"]), {}).get("brand"))
        / num_interactions
        if num_interactions
        else 0.0
    )
    price_summary = {"count": 0}
    if prices:
        price_summary = {
            "count": len(prices),
            "min": float(min(prices)),
            "median": round(float(statistics.median(prices)), 2),
            "max": float(max(prices)),
        }
    top_price_items = sorted(
        priced_items,
        key=lambda item: (-item["price"], item["item"]),
    )[:top_price_items_n]

    evidence = {
        "num_interactions": num_interactions,
        "category_counts": _count_list(category_counter, evidence_limit),
        "brand_counts": _count_list(brand_counter, evidence_limit),
        "description_keyword_counts": _count_list(desc_counter, evidence_limit),
        "title_keyword_counts": _count_list(title_counter, evidence_limit),
        "review_keyword_counts": _count_list(review_counter, evidence_limit),
        "rating_counts": dict(rating_counter),
        "price_summary": price_summary,
        "brand_coverage": round(brand_coverage, 4),
    }
    llama_profile = _compose_llama_profile(evidence)
    profile_text = _compose_profile_text(llama_profile)
    return {
        "profile_mode": "structured",
        "num_interactions": num_interactions,
        "llama_profile": llama_profile,
        "top_price_items": top_price_items,
        "profile_evidence": evidence,
        "profile_text": profile_text,
        "config": {
            "profile_mode": "structured",
            "top_price_items": top_price_items_n,
            "evidence_limit": evidence_limit,
        },
    }


def generate_profiles(
    *,
    data_dir: Path,
    dataset_name: str,
    profile_dir: Path,
    fold: str,
    scopes: list[str],
    top_price_items_n: int = 5,
    evidence_limit: int = 50,
):
    reviews = pd.DataFrame(pd.read_pickle(data_dir / dataset_name / "reviews.pickle"))
    reviews["raw_user"] = reviews["user"].astype(str)
    reviews["raw_item"] = reviews["item"].astype(str)
    item_path = data_dir / dataset_name / "item.json"
    if item_path.is_file():
        import json

        item_rows = json.load(item_path.open("r", encoding="utf-8"))
        item_meta = {
            str(row.get("item")): row for row in item_rows if row.get("item") is not None
        }
    else:
        print(f"WARNING: no item.json at {item_path}; profiles will use sparse metadata.")
        item_meta = {}

    split_indices = read_split_indices(data_dir, dataset_name, fold)
    out_dir = profile_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for scope in scopes:
        if scope == "train":
            row_indices = split_indices["train"]
        elif scope == "train_valid":
            row_indices = split_indices["train"] + split_indices["validation"]
        else:
            raise ValueError(f"Unsupported scope: {scope}")

        scoped = reviews.iloc[row_indices].reset_index(drop=True)
        profiles = {}
        for raw_user, group in scoped.groupby("raw_user", sort=False):
            interactions = group.to_dict("records")
            profile = _aggregate_user_profile(
                interactions,
                item_meta,
                fold=fold,
                scope=scope,
                top_price_items_n=top_price_items_n,
                evidence_limit=evidence_limit,
            )
            profiles[str(raw_user)] = {
                "raw_user": str(raw_user),
                "fold": str(fold),
                "scope": scope,
                **profile,
            }

        out_path = out_dir / f"fold_{fold}_{scope}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(profiles, f)
        print(f"Wrote {len(profiles)} profiles to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate structured user profiles")
    parser.add_argument("--dataset_name", "--dataset", dest="dataset_name", required=True)
    parser.add_argument("--data_dir", default=str(PACKAGE_ROOT / "data"), type=str)
    parser.add_argument(
        "--profile_dir",
        default=str(PACKAGE_ROOT / "data" / "profiles"),
        type=str,
    )
    parser.add_argument("--fold", default="1", type=str)
    parser.add_argument("--scopes", default="train,train_valid", type=str)
    parser.add_argument("--top_price_items", default=5, type=int)
    parser.add_argument("--evidence_limit", default=50, type=int)
    args = parser.parse_args()

    class _ResolveArgs:
        dataset_name = args.dataset_name
        data_dir = args.data_dir

    resolve_args = _ResolveArgs()
    resolve_dataset_paths(resolve_args)

    scopes = [scope.strip() for scope in args.scopes.split(",") if scope.strip()]
    generate_profiles(
        data_dir=Path(resolve_args.data_dir),
        dataset_name=resolve_args.dataset_name,
        profile_dir=Path(args.profile_dir),
        fold=str(args.fold),
        scopes=scopes,
        top_price_items_n=args.top_price_items,
        evidence_limit=args.evidence_limit,
    )


if __name__ == "__main__":
    main()
