#!/usr/bin/env python
"""Validate LLM-generated user profile pickles."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_llm.config.args import default_model_path
from graph_llm.config.datasets import resolve_dataset_paths
from graph_llm.dataload.legacy_data import read_split_indices
from graph_llm.train.trainer import ensure_tokenizer_ready

REQUIRED_LINES = ("User:", "Strictness:", "Frequent nouns:", "Interests:", "Review style:")


def scope_users(data_dir: Path, dataset_name: str, fold: str, scope: str) -> set[str]:
    reviews = pd.DataFrame(pd.read_pickle(data_dir / dataset_name / "reviews.pickle"))
    split_indices = read_split_indices(data_dir, dataset_name, fold)
    if scope == "train":
        row_indices = split_indices["train"]
    elif scope == "train_valid":
        row_indices = split_indices["train"] + split_indices["validation"]
    else:
        raise ValueError(scope)
    scoped = reviews.iloc[row_indices]
    return set(scoped["user"].astype(str).unique())


def validate_profile(path: Path, expected_users: set[str], tokenizer, max_tokens: int) -> dict:
    profiles = pickle.load(path.open("rb"))
    missing = expected_users - set(profiles.keys())
    over_tokens = []
    bad_format = []
    thinking = []
    not_llm = []

    for uid, record in profiles.items():
        if not isinstance(record, dict):
            bad_format.append(uid)
            continue
        if record.get("profile_mode") != "llm":
            not_llm.append(uid)
        text = str(record.get("profile_text") or "")
        if "redacted_thinking" in text.lower():
            thinking.append(uid)
        token_len = len(tokenizer.encode(text, add_special_tokens=False))
        if token_len > max_tokens:
            over_tokens.append((uid, token_len))
        if not all(marker in text for marker in REQUIRED_LINES):
            bad_format.append(uid)

    return {
        "path": str(path),
        "count": len(profiles),
        "missing": len(missing),
        "missing_sample": sorted(missing)[:5],
        "over_tokens": len(over_tokens),
        "over_tokens_sample": over_tokens[:5],
        "bad_format": len(bad_format),
        "bad_format_sample": bad_format[:5],
        "thinking": len(thinking),
        "not_llm": len(not_llm),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate LLM profile pickles")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--profile_dir", default=str(PACKAGE_ROOT / "data" / "profiles"))
    parser.add_argument("--fold", default="1")
    parser.add_argument("--scopes", default="train,train_valid")
    parser.add_argument("--max_tokens", default=128, type=int)
    args = parser.parse_args()

    class _ResolveArgs:
        dataset_name = args.dataset_name
        data_dir = args.data_dir

    resolve_dataset_paths(_ResolveArgs())
    tokenizer = ensure_tokenizer_ready(
        AutoTokenizer.from_pretrained(
            default_model_path(),
            local_files_only=True,
            trust_remote_code=True,
        )
    )

    data_dir = Path(_ResolveArgs.data_dir)
    dataset_name = _ResolveArgs.dataset_name
    profile_dir = Path(args.profile_dir) / dataset_name.strip("/")

    ok = True
    for scope in [s.strip() for s in args.scopes.split(",") if s.strip()]:
        path = profile_dir / f"fold_{args.fold}_{scope}.pkl"
        expected = scope_users(data_dir, dataset_name, args.fold, scope)
        result = validate_profile(path, expected, tokenizer, args.max_tokens)
        print(result)
        if result["missing"] or result["over_tokens"] or result["bad_format"] or result["thinking"]:
            ok = False

    if not ok:
        raise SystemExit(1)
    print("Validation passed.")


if __name__ == "__main__":
    main()
