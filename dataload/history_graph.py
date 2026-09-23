"""推荐模式用户历史图：稳定文本片段 + 历史商品节点。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from graph_llm.dataload.sequential_data import InteractionRecord


@dataclass
class HistoryGraphNode:
    node_id: int
    node_type: str  # fragment | item
    text: str
    raw_item: str
    raw_user: str
    timestamp: int
    rating_raw: float | None
    item_index: int | None


@dataclass
class HistoryUserGraph:
    nodes: list[HistoryGraphNode]
    edge_index: np.ndarray
    edge_weight: np.ndarray
    node_types: np.ndarray
    node_item_indices: np.ndarray  # -1 for fragments

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @classmethod
    def empty(cls) -> HistoryUserGraph:
        return cls(
            nodes=[],
            edge_index=np.empty((2, 0), dtype=np.int64),
            edge_weight=np.empty((0,), dtype=np.float32),
            node_types=np.empty((0,), dtype=np.int64),
            node_item_indices=np.empty((0,), dtype=np.int64),
        )


def _split_fragments(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _stable_fragment_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_history_user_graph(
    history: list[InteractionRecord],
    *,
    max_nodes: int = 512,
) -> HistoryUserGraph:
    if not history:
        return HistoryUserGraph.empty()

    nodes: list[HistoryGraphNode] = []
    frag_key_to_idx: dict[str, int] = {}
    item_key_to_idx: dict[int, int] = {}

    def add_node(**kwargs) -> int:
        nodes.append(HistoryGraphNode(node_id=len(nodes), **kwargs))
        return nodes[-1].node_id

    for rec in history:
        if rec.item_index not in item_key_to_idx:
            item_key_to_idx[rec.item_index] = add_node(
                node_type="item",
                text=rec.raw_item,
                raw_item=rec.raw_item,
                raw_user=rec.raw_user,
                timestamp=rec.timestamp,
                rating_raw=rec.rating_raw,
                item_index=rec.item_index,
            )
        item_node = item_key_to_idx[rec.item_index]
        frags = _split_fragments(rec.review_text) or _split_fragments(rec.summary)
        frag_indices = []
        for frag in frags:
            key = _stable_fragment_id(frag)
            if key not in frag_key_to_idx:
                frag_key_to_idx[key] = add_node(
                    node_type="fragment",
                    text=frag,
                    raw_item=rec.raw_item,
                    raw_user=rec.raw_user,
                    timestamp=rec.timestamp,
                    rating_raw=None,
                    item_index=None,
                )
            frag_indices.append(frag_key_to_idx[key])
        for f_idx in frag_indices:
            # 片段连到商品
            pass  # edges below

    if len(nodes) > max_nodes:
        # 按时间保留较新的节点
        nodes.sort(key=lambda n: n.timestamp)
        keep = set(range(max(0, len(nodes) - max_nodes), len(nodes)))
        old_to_new = {}
        new_nodes = []
        for i, node in enumerate(nodes):
            if i in keep:
                old_to_new[i] = len(new_nodes)
                node.node_id = len(new_nodes)
                new_nodes.append(node)
        nodes = new_nodes
        frag_key_to_idx = {
            k: old_to_new[v] for k, v in frag_key_to_idx.items() if v in old_to_new
        }
        item_key_to_idx = {
            k: old_to_new[v] for k, v in item_key_to_idx.items() if v in old_to_new
        }

    edges: list[tuple[int, int]] = []
    for rec in history:
        if rec.item_index not in item_key_to_idx:
            continue
        item_node = item_key_to_idx[rec.item_index]
        frags = _split_fragments(rec.review_text) or _split_fragments(rec.summary)
        for frag in frags:
            key = _stable_fragment_id(frag)
            if key in frag_key_to_idx:
                f_idx = frag_key_to_idx[key]
                edges.append((f_idx, item_node))
                edges.append((item_node, f_idx))

    if not edges:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_weight = np.empty((0,), dtype=np.float32)
    else:
        edge_index = np.array(edges, dtype=np.int64).T
        edge_weight = np.ones((edge_index.shape[1],), dtype=np.float32)

    node_types = np.array([1 if n.node_type == "item" else 0 for n in nodes], dtype=np.int64)
    node_item_indices = np.array(
        [n.item_index if n.item_index is not None else -1 for n in nodes],
        dtype=np.int64,
    )
    return HistoryUserGraph(
        nodes=nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_types=node_types,
        node_item_indices=node_item_indices,
    )


def batch_history_graphs(graphs: Iterable[HistoryUserGraph]) -> dict:
    """拼成 selector 可用的 batch 张量。"""
    graphs = list(graphs)
    if not graphs:
        return {
            "num_nodes_per_graph": [],
            "batch_index": np.empty((0,), dtype=np.int64),
            "edge_index": np.empty((2, 0), dtype=np.int64),
            "edge_weight": np.empty((0,), dtype=np.float32),
            "node_types": np.empty((0,), dtype=np.int64),
            "node_item_indices": np.empty((0,), dtype=np.int64),
        }
    node_types_list = []
    node_item_list = []
    batch_index = []
    edges = []
    weights = []
    offset = 0
    num_nodes_per_graph = []
    for b, g in enumerate(graphs):
        n = g.num_nodes
        num_nodes_per_graph.append(n)
        if n == 0:
            continue
        node_types_list.append(g.node_types)
        node_item_list.append(g.node_item_indices)
        batch_index.extend([b] * n)
        if g.edge_index.size > 0:
            ei = g.edge_index + offset
            edges.append(ei)
            weights.append(g.edge_weight)
        offset += n
    return {
        "num_nodes_per_graph": num_nodes_per_graph,
        "batch_index": np.array(batch_index, dtype=np.int64),
        "edge_index": np.concatenate(edges, axis=1) if edges else np.empty((2, 0), dtype=np.int64),
        "edge_weight": np.concatenate(weights, axis=0) if weights else np.empty((0,), dtype=np.float32),
        "node_types": np.concatenate(node_types_list, axis=0) if node_types_list else np.empty((0,), dtype=np.int64),
        "node_item_indices": np.concatenate(node_item_list, axis=0) if node_item_list else np.empty((0,), dtype=np.int64),
        "graphs": graphs,
    }
