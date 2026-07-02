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

TOP_PRICE_MARKER = (
    "Highest-price interacted items, treated as strongest preference signals:"
)
PROFILE_HEADER = (
    "User profile extracted from all items interacted with by this user.\n"
    "The interacted items are treated as an unordered set; their order does "
    "not imply interaction order."
)
TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize_text(text, stop_tokens=None):
    if stop_tokens is None:
        stop_tokens = DEFAULT_STOP_TOKENS
    text = "" if text is None else str(text).lower()
    return [tok for tok in TOKEN_RE.findall(text) if tok and tok not in stop_tokens]


def _count_list(counter: Counter, limit: int = 50):
    return [{"name": name, "count": count} for name, count in counter.most_common(limit)]


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


def _format_price(value: float) -> str:
    return f"${value:.2f}"


def _compose_llama_profile(evidence: dict, top_price_items: list) -> str:
    sections = []
    categories = evidence.get("category_counts", [])
    brands = evidence.get("brand_counts", [])
    desc_kw = evidence.get("description_keyword_counts", [])
    title_kw = evidence.get("title_keyword_counts", [])
    review_kw = evidence.get("review_keyword_counts", [])
    price_summary = evidence.get("price_summary", {})
    brand_coverage = evidence.get("brand_coverage", 0.0)

    if categories:
        cat_names = _join_names(categories)
        sections.append(
            f"Preferred genres/categories: The user shows recurring interest in {cat_names}."
        )
    else:
        sections.append(
            "Preferred genres/categories: Category metadata is sparse for this user."
        )

    if len(brands) >= 2:
        brand_names = _join_names(brands)
        sections.append(
            "Preferred brands/studios: The strongest available brand/studio signals are "
            f"{brand_names}."
        )
    elif len(brands) == 1:
        sections.append(
            "Preferred brands/studios: Brand/studio metadata is sparse, but "
            f"{brands[0]['name']} appears among the available records."
        )
    else:
        sections.append(
            "Preferred brands/studios: Brand/studio metadata is sparse for this user."
        )

    theme_parts = []
    if desc_kw:
        theme_parts.append(
            "description themes such as " + _join_keywords(desc_kw)
        )
    if title_kw:
        theme_parts.append("title signals such as " + _join_keywords(title_kw))
    if review_kw:
        theme_parts.append(
            "review/template signals such as " + _join_keywords(review_kw)
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

    if price_summary.get("count", 0) > 0:
        sections.append(
            "Price preference: Available prices range from "
            f"{_format_price(price_summary['min'])} to "
            f"{_format_price(price_summary['max'])}, with a median around "
            f"{_format_price(price_summary['median'])}."
        )

    if brand_coverage < 0.2:
        sections.append(
            "Less relevant signals: No clearly disliked signals are evident. "
            "brand/studio metadata provide weaker or less reliable evidence."
        )
    else:
        sections.append(
            "Less relevant signals: No clearly disliked signals are evident from "
            "implicit interactions; lower-frequency interests should be treated "
            "as weaker signals."
        )

    summary_bits = []
    if categories:
        summary_bits.append(_join_names(categories))
    if desc_kw:
        summary_bits.append(_join_keywords(desc_kw))
    if brands:
        summary_bits.append(_join_names(brands))
    summary_text = "; ".join(bit for bit in summary_bits if bit)
    if top_price_items:
        sections.append(
            "Overall preference summary: Overall, the user appears to prefer "
            f"{summary_text}. The highest-price interacted items provide "
            "additional strong preference signals."
        )
    else:
        sections.append(
            "Overall preference summary: Overall, the user appears to prefer "
            f"{summary_text or 'the available interaction signals'}."
        )
    return "\n".join(sections)


def _compose_profile_text(
    num_interactions: int,
    top_price_items: list,
    llama_profile: str,
) -> str:
    lines = [
        PROFILE_HEADER,
        "",
        f"Number of interacted items used: {num_interactions}.",
        "",
    ]
    if top_price_items:
        lines.append(TOP_PRICE_MARKER)
        for idx, item in enumerate(top_price_items, start=1):
            price = item.get("price")
            price_text = _format_price(price) if price is not None else "N/A"
            lines.append(f"{idx}. {item.get('title', item.get('item'))} ({price_text})")
        lines.append("")
    lines.append(llama_profile)
    return "\n".join(lines)


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
        category = _leaf_category(meta)
        if category:
            category_counter[category] += 1
        brand = meta.get("brand")
        if brand:
            brand_counter[str(brand).strip()] += 1
        title = _item_title(meta, raw_item)
        description = meta.get("description") or ""
        review_text = row.get("review_text") or ""
        if isinstance(row.get("template"), (list, tuple)) and len(row["template"]) >= 3:
            review_text = row["template"][2]
        desc_counter.update(_tokenize_text(description))
        title_counter.update(_tokenize_text(title))
        review_counter.update(_tokenize_text(review_text))
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
            "median": float(statistics.median(prices)),
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
    llama_profile = _compose_llama_profile(evidence, top_price_items)
    profile_text = _compose_profile_text(num_interactions, top_price_items, llama_profile)
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
