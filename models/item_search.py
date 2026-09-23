"""RPG 式商品图搜索与全量精确打分。"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch


class ItemNeighborIndex:
    """按 RoBERTa 文本向量建稀疏近邻（CPU）。"""

    def __init__(self, item_vectors: np.ndarray, neighbors: int = 32, seed: int = 5254):
        self.neighbors = neighbors
        self.rng = random.Random(seed)
        n = item_vectors.shape[0]
        # 归一化后内积 = 余弦
        norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        self.vectors = item_vectors / norms
        self.adj: dict[int, list[int]] = {}
        block = 512
        for start in range(0, n, block):
            end = min(n, start + block)
            sims = self.vectors[start:end] @ self.vectors.T
            for local_i, global_i in enumerate(range(start, end)):
                row = sims[local_i]
                row[global_i] = -1.0
                top = np.argpartition(-row, min(neighbors, n - 1))[:neighbors]
                top = sorted(top.tolist(), key=lambda j: (-row[j], j))
                self.adj[global_i] = top

    def neighbors_of(self, item_index: int) -> list[int]:
        return self.adj.get(item_index, [])


def exact_top_k(
    score_fn: Callable[[], torch.Tensor],
    exclude: set[int],
    k: int,
) -> tuple[list[int], torch.Tensor]:
    scores = score_fn()
    if exclude:
        for idx in exclude:
            if 0 <= idx < scores.numel():
                scores[idx] = float("-inf")
    topk = torch.topk(scores, k=min(k, scores.numel()))
    return topk.indices.tolist(), topk.values


def graph_search_top_k(
    score_fn: Callable[[], torch.Tensor],
    neighbor_index: ItemNeighborIndex,
    seed_items: list[int],
    *,
    exclude: set[int],
    k: int,
    rounds: int,
    max_candidates: int,
    rng: random.Random,
) -> tuple[list[int], torch.Tensor]:
    visited: set[int] = set()
    candidates: dict[int, float] = {}
    frontier = list(seed_items)
    for _ in range(rounds):
        if not frontier:
            break
        next_frontier = []
        scores = score_fn()
        for item in frontier:
            if item in exclude:
                continue
            s = float(scores[item].item())
            prev = candidates.get(item)
            if prev is None or s > prev:
                candidates[item] = s
            visited.add(item)
            for nb in neighbor_index.neighbors_of(item):
                if nb not in visited and nb not in exclude:
                    next_frontier.append(nb)
        frontier = next_frontier[:max_candidates]
    if not candidates:
        return exact_top_k(score_fn, exclude, k)
    ranked = sorted(candidates.items(), key=lambda x: (-x[1], x[0]))
    items = [i for i, _ in ranked[:k]]
    vals = torch.tensor([candidates[i] for i in items])
    return items, vals


def recovery_rate(exact_items: list[int], approx_items: list[int], k: int) -> float:
    es = set(exact_items[:k])
    ap = set(approx_items[:k])
    if not es:
        return 1.0
    return len(es & ap) / len(es)
