"""推荐排序指标 HR@K 与 NDCG@K。"""

from __future__ import annotations

import math
from typing import Iterable


def dcg_at_k(relevances: list[int], k: int) -> float:
    score = 0.0
    for idx, rel in enumerate(relevances[:k], start=1):
        if rel:
            score += (2**rel - 1) / math.log2(idx + 1)
    return score


def ndcg_at_k(relevances: list[int], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    denom = dcg_at_k(ideal, k)
    if denom <= 0:
        return 0.0
    return dcg_at_k(relevances, k) / denom


def hr_at_k(ranked: list[int], target: int, k: int) -> float:
    return 1.0 if target in ranked[:k] else 0.0


def evaluate_ranking(
    targets: list[int],
    ranked_lists: list[list[int]],
    ks: Iterable[int] = (5, 10, 20),
) -> dict[str, float]:
    """每个样本一个目标商品（leave-one-out）。"""
    ks = tuple(ks)
    hr_sums = {k: 0.0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    n = len(targets)
    if n == 0:
        return {f"HR@{k}": 0.0 for k in ks} | {f"NDCG@{k}": 0.0 for k in ks}
    for target, ranked in zip(targets, ranked_lists):
        rel = [1 if item == target else 0 for item in ranked]
        for k in ks:
            hr_sums[k] += hr_at_k(ranked, target, k)
            ndcg_sums[k] += ndcg_at_k(rel, k)
    metrics = {}
    for k in ks:
        metrics[f"HR@{k}"] = hr_sums[k] / n
        metrics[f"NDCG@{k}"] = ndcg_sums[k] / n
    return metrics
