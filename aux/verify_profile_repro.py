"""Verify regenerated profiles against the legacy MoviesAndTV cache."""

from __future__ import annotations

import hashlib
import json
import pickle
import statistics
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_llm.aux.generate_profiles import _aggregate_user_profile
from graph_llm.dataload.legacy_data import read_split_indices

DATASET = "Amazon/MoviesAndTV"
REF_PKL = ROOT / "graph_llm/data/profiles" / DATASET / "fold_1_train.pkl"
SIMILARITY_TARGET = 0.95


def _similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def main() -> None:
    with REF_PKL.open("rb") as f:
        reference = pickle.load(f)

    reviews = pd.DataFrame(pd.read_pickle(ROOT / "data" / DATASET / "reviews.pickle"))
    reviews["raw_user"] = reviews["user"].astype(str)
    reviews["raw_item"] = reviews["item"].astype(str)
    item_meta = {
        str(row.get("item")): row
        for row in json.load((ROOT / "data" / DATASET / "item.json").open(encoding="utf-8"))
        if row.get("item") is not None
    }
    scoped = reviews.iloc[read_split_indices(ROOT / "data", DATASET, "1")["train"]]

    same_text = 0
    same_md5 = 0
    similarities: list[float] = []
    for uid, rec in reference.items():
        interactions = scoped[scoped["raw_user"] == uid].to_dict("records")
        regenerated = _aggregate_user_profile(interactions, item_meta, fold="1", scope="train")
        old_text = rec["profile_text"]
        new_text = regenerated["profile_text"]
        if new_text == old_text:
            same_text += 1
        old_md5 = hashlib.md5(old_text.encode("utf-8")).hexdigest()
        new_md5 = hashlib.md5(new_text.encode("utf-8")).hexdigest()
        if old_md5 == new_md5:
            same_md5 += 1
        similarities.append(_similarity(old_text, new_text))

    total = len(reference)
    mean_sim = statistics.fmean(similarities)
    median_sim = statistics.median(similarities)
    ge_target = sum(similarity >= SIMILARITY_TARGET for similarity in similarities)
    print(f"profile_text exact match: {same_text}/{total}", flush=True)
    print(f"profile_text md5 match: {same_md5}/{total}", flush=True)
    print(f"profile_text mean similarity: {mean_sim:.4f}", flush=True)
    print(f"profile_text median similarity: {median_sim:.4f}", flush=True)
    print(
        f"profile_text similarity >= {SIMILARITY_TARGET:.0%}: {ge_target}/{total}",
        flush=True,
    )
    if ge_target < total:
        worst = sorted(
            ((uid, sim) for uid, sim in zip(reference.keys(), similarities)),
            key=lambda item: item[1],
        )[:5]
        print("lowest-similarity users:", flush=True)
        for uid, sim in worst:
            print(f"  {uid}: {sim:.4f}", flush=True)


if __name__ == "__main__":
    main()
