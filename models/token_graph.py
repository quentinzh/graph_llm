"""Leakage-safe per-user personalized token graph construction."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class UserTokenGraph:
    """Directed token graph for one user-item sample view."""

    node_token_ids: np.ndarray  # [N] LM vocab ids
    node_surfaces: list[str]
    node_counts: np.ndarray  # [N] total occurrences in allowed history
    node_doc_freq: np.ndarray  # [N] number of reviews containing token
    edge_index: np.ndarray  # [2, E] directed src -> dst
    edge_weight: np.ndarray  # [E] co-occurrence counts
    in_degree: np.ndarray  # [N]
    out_degree: np.ndarray  # [N]

    @property
    def num_nodes(self) -> int:
        return int(self.node_token_ids.shape[0])

    @classmethod
    def empty(cls) -> UserTokenGraph:
        return cls(
            node_token_ids=np.empty((0,), dtype=np.int64),
            node_surfaces=[],
            node_counts=np.empty((0,), dtype=np.float32),
            node_doc_freq=np.empty((0,), dtype=np.float32),
            edge_index=np.empty((2, 0), dtype=np.int64),
            edge_weight=np.empty((0,), dtype=np.float32),
            in_degree=np.empty((0,), dtype=np.float32),
            out_degree=np.empty((0,), dtype=np.float32),
        )


@dataclass
class ReviewRecord:
    row_key: int
    raw_user: str
    raw_item: str
    token_ids: tuple[int, ...]


def _filter_token_ids(
    token_ids: Iterable[int],
    skip_ids: set[int],
) -> list[int]:
    return [int(t) for t in token_ids if int(t) not in skip_ids]


def extract_explanation_tokens(tokenizer, explanation: str, skip_ids: set[int]) -> list[int]:
    ids = tokenizer(explanation, add_special_tokens=False)["input_ids"]
    return _filter_token_ids(ids, skip_ids)


def _add_sequence_edges(
    tokens: list[int],
    token_to_node: dict[int, int],
    edge_counter: Counter[tuple[int, int]],
) -> None:
    node_seq = [token_to_node[t] for t in tokens if t in token_to_node]
    for i in range(len(node_seq)):
        src = node_seq[i]
        for j in range(i + 1, len(node_seq)):
            dst = node_seq[j]
            if src != dst:
                edge_counter[(src, dst)] += 1


def build_sample_token_graph(
    history_records: list[ReviewRecord],
    *,
    exclude_row_key: int,
    target_raw_item: str,
    skip_token_ids: set[int],
    max_nodes: int = 512,
    min_token_count: int = 1,
) -> UserTokenGraph:
    """Build a directed token graph from leakage-safe user history.

    Excludes the current sample and every review for the target item.
    """
    eligible: list[ReviewRecord] = []
    for record in history_records:
        if record.row_key == exclude_row_key:
            continue
        if str(record.raw_item) == str(target_raw_item):
            continue
        tokens = _filter_token_ids(record.token_ids, skip_token_ids)
        if tokens:
            eligible.append(ReviewRecord(record.row_key, record.raw_user, record.raw_item, tuple(tokens)))

    if not eligible:
        return UserTokenGraph.empty()

    token_count: Counter[int] = Counter()
    token_doc_freq: Counter[int] = Counter()
    for record in eligible:
        seen = set(record.token_ids)
        token_count.update(record.token_ids)
        token_doc_freq.update(seen)

    ranked_tokens = [
        tid for tid, _ in token_count.most_common(max_nodes)
        if token_count[tid] >= min_token_count
    ]
    if not ranked_tokens:
        return UserTokenGraph.empty()

    token_to_node = {tid: idx for idx, tid in enumerate(ranked_tokens)}
    edge_counter: Counter[tuple[int, int]] = Counter()

    by_item: dict[str, list[ReviewRecord]] = defaultdict(list)
    for record in eligible:
        _add_sequence_edges(list(record.token_ids), token_to_node, edge_counter)
        by_item[str(record.raw_item)].append(record)

    for item_records in by_item.values():
        if len(item_records) < 2:
            continue
        merged: list[int] = []
        for rec in item_records:
            merged.extend([t for t in rec.token_ids if t in token_to_node])
        _add_sequence_edges(merged, token_to_node, edge_counter)

    if not edge_counter:
        # Keep isolated nodes when no co-occurrence edges exist.
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_weight = np.empty((0,), dtype=np.float32)
    else:
        edges = list(edge_counter.items())
        edge_index = np.array([[e[0][0], e[0][1]] for e in edges], dtype=np.int64).T
        edge_weight = np.array([float(e[1]) for e in edges], dtype=np.float32)

    in_degree = np.zeros(len(ranked_tokens), dtype=np.float32)
    out_degree = np.zeros(len(ranked_tokens), dtype=np.float32)
    if edge_index.size > 0:
        src = edge_index[0]
        dst = edge_index[1]
        for s in src:
            out_degree[s] += 1.0
        for d in dst:
            in_degree[d] += 1.0

    surfaces = []
    for tid in ranked_tokens:
        try:
            surfaces.append(str(tokenizer_decode_stub(tid)))
        except Exception:
            surfaces.append(str(tid))

    return UserTokenGraph(
        node_token_ids=np.array(ranked_tokens, dtype=np.int64),
        node_surfaces=surfaces,
        node_counts=np.array([float(token_count[t]) for t in ranked_tokens], dtype=np.float32),
        node_doc_freq=np.array([float(token_doc_freq[t]) for t in ranked_tokens], dtype=np.float32),
        edge_index=edge_index,
        edge_weight=edge_weight,
        in_degree=in_degree,
        out_degree=out_degree,
    )


def tokenizer_decode_stub(token_id: int) -> str:
    return f"<tok:{token_id}>"


def attach_tokenizer_decode(tokenizer) -> None:
    global tokenizer_decode_stub

    def _decode(token_id: int) -> str:
        return tokenizer.decode([int(token_id)], skip_special_tokens=True)

    tokenizer_decode_stub = _decode


def log_frequency_features(counts: np.ndarray, doc_freq: np.ndarray, num_reviews: int) -> np.ndarray:
    """Return [N, 3] numeric features: log count, log doc freq, idf-like score."""
    if counts.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    log_count = np.log1p(counts)
    log_df = np.log1p(doc_freq)
    idf = np.log((max(num_reviews, 1) + 1.0) / (doc_freq + 1.0))
    return np.stack([log_count, log_df, idf], axis=-1).astype(np.float32)


def select_high_frequency_negatives(
    graph: UserTokenGraph,
    evidence_node_indices: np.ndarray,
    *,
    top_k: int,
    protected_token_ids: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Pick unselected high-frequency nodes as UL negative candidates."""
    if graph.num_nodes == 0 or top_k <= 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

    evidence_set = set(int(x) for x in evidence_node_indices.tolist())
    ranked = np.argsort(-graph.node_counts)
    neg_nodes: list[int] = []
    neg_weights: list[float] = []
    max_count = float(graph.node_counts.max()) if graph.node_counts.size else 1.0
    for node_idx in ranked:
        node_idx = int(node_idx)
        if node_idx in evidence_set:
            continue
        token_id = int(graph.node_token_ids[node_idx])
        if token_id in protected_token_ids:
            continue
        neg_nodes.append(node_idx)
        neg_weights.append(float(graph.node_counts[node_idx]) / max(max_count, 1.0))
        if len(neg_nodes) >= top_k:
            break

    if not neg_nodes:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
    return (
        np.array(neg_nodes, dtype=np.int64),
        np.array(neg_weights, dtype=np.float32),
    )


def batch_graphs(graphs: list[UserTokenGraph]) -> dict[str, np.ndarray | list[UserTokenGraph]]:
    """Batch variable-size graphs for GNN processing."""
    if not graphs:
        return {
            "graphs": graphs,
            "node_token_ids": np.empty((0,), dtype=np.int64),
            "node_counts": np.empty((0,), dtype=np.float32),
            "node_doc_freq": np.empty((0,), dtype=np.float32),
            "node_in_degree": np.empty((0,), dtype=np.float32),
            "node_out_degree": np.empty((0,), dtype=np.float32),
            "edge_index": np.empty((2, 0), dtype=np.int64),
            "edge_weight": np.empty((0,), dtype=np.float32),
            "batch_index": np.empty((0,), dtype=np.int64),
            "num_nodes_per_graph": np.empty((0,), dtype=np.int64),
        }

    node_token_ids = []
    node_counts = []
    node_doc_freq = []
    node_in_degree = []
    node_out_degree = []
    edge_src = []
    edge_dst = []
    edge_weight = []
    batch_index = []
    num_nodes_per_graph = []
    offset = 0

    for batch_idx, graph in enumerate(graphs):
        n = graph.num_nodes
        num_nodes_per_graph.append(n)
        if n == 0:
            continue
        node_token_ids.append(graph.node_token_ids)
        node_counts.append(graph.node_counts)
        node_doc_freq.append(graph.node_doc_freq)
        node_in_degree.append(graph.in_degree)
        node_out_degree.append(graph.out_degree)
        batch_index.append(np.full((n,), batch_idx, dtype=np.int64))
        if graph.edge_index.size > 0:
            edge_src.append(graph.edge_index[0] + offset)
            edge_dst.append(graph.edge_index[1] + offset)
            edge_weight.append(graph.edge_weight)
        offset += n

    if offset == 0:
        return {
            "graphs": graphs,
            "node_token_ids": np.empty((0,), dtype=np.int64),
            "node_counts": np.empty((0,), dtype=np.float32),
            "node_doc_freq": np.empty((0,), dtype=np.float32),
            "node_in_degree": np.empty((0,), dtype=np.float32),
            "node_out_degree": np.empty((0,), dtype=np.float32),
            "edge_index": np.empty((2, 0), dtype=np.int64),
            "edge_weight": np.empty((0,), dtype=np.float32),
            "batch_index": np.empty((0,), dtype=np.int64),
            "num_nodes_per_graph": np.array(num_nodes_per_graph, dtype=np.int64),
        }

    edge_index = np.stack([
        np.concatenate(edge_src) if edge_src else np.empty((0,), dtype=np.int64),
        np.concatenate(edge_dst) if edge_dst else np.empty((0,), dtype=np.int64),
    ], axis=0)

    return {
        "graphs": graphs,
        "node_token_ids": np.concatenate(node_token_ids),
        "node_counts": np.concatenate(node_counts),
        "node_doc_freq": np.concatenate(node_doc_freq),
        "node_in_degree": np.concatenate(node_in_degree),
        "node_out_degree": np.concatenate(node_out_degree),
        "edge_index": edge_index,
        "edge_weight": np.concatenate(edge_weight) if edge_weight else np.empty((0,), dtype=np.float32),
        "batch_index": np.concatenate(batch_index),
        "num_nodes_per_graph": np.array(num_nodes_per_graph, dtype=np.int64),
    }
