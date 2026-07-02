"""Lightweight GNN-based neural evidence selector."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_llm.models.token_graph import (
    UserTokenGraph,
    batch_graphs,
    log_frequency_features,
    select_high_frequency_negatives,
)


class GraphSAGEConv(nn.Module):
    """Single GraphSAGE layer with mean aggregation."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        num_nodes = x.shape[0]
        if edge_index.numel() == 0:
            neigh = torch.zeros_like(x)
        else:
            src, dst = edge_index
            agg = torch.zeros(num_nodes, x.shape[1], device=x.device, dtype=x.dtype)
            count = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)
            msg = x[src]
            agg.index_add_(0, dst, msg)
            ones = torch.ones(msg.shape[0], 1, device=x.device, dtype=x.dtype)
            count.index_add_(0, dst, ones)
            neigh = agg / count.clamp_min(1.0)
        return F.relu(self.lin_self(x) + self.lin_neigh(neigh))


class EvidenceSelector(nn.Module):
    """Score token-graph nodes for current user-item explanation utility."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        gnn_layers: int = 2,
        stat_dim: int = 5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim + stat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs = nn.ModuleList([
            GraphSAGEConv(hidden_dim, hidden_dim) for _ in range(gnn_layers)
        ])
        self.item_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _node_stats(
        self,
        counts: torch.Tensor,
        doc_freq: torch.Tensor,
        in_degree: torch.Tensor,
        out_degree: torch.Tensor,
        num_reviews: float,
    ) -> torch.Tensor:
        log_count = torch.log1p(counts)
        log_df = torch.log1p(doc_freq)
        idf = torch.log(torch.tensor(num_reviews + 1.0, device=counts.device) / (doc_freq + 1.0))
        norm_in = in_degree / in_degree.max().clamp_min(1.0) if in_degree.numel() else in_degree
        norm_out = out_degree / out_degree.max().clamp_min(1.0) if out_degree.numel() else out_degree
        return torch.stack([log_count, log_df, idf, norm_in, norm_out], dim=-1)

    def encode_nodes(
        self,
        node_token_emb: torch.Tensor,
        node_counts: torch.Tensor,
        node_doc_freq: torch.Tensor,
        node_in_degree: torch.Tensor,
        node_out_degree: torch.Tensor,
        edge_index: torch.Tensor,
        num_reviews: float = 1.0,
    ) -> torch.Tensor:
        stats = self._node_stats(
            node_counts, node_doc_freq, node_in_degree, node_out_degree, num_reviews,
        )
        x = self.input_proj(torch.cat([node_token_emb, stats], dim=-1))
        for conv in self.convs:
            x = conv(x, edge_index)
        return x

    def forward_single(
        self,
        graph: UserTokenGraph,
        node_token_emb: torch.Tensor,
        item_emb: torch.Tensor,
        num_reviews: float = 1.0,
    ) -> torch.Tensor:
        if graph.num_nodes == 0 or node_token_emb.numel() == 0:
            return torch.empty((0,), device=item_emb.device)

        device = item_emb.device
        counts = torch.tensor(graph.node_counts, device=device, dtype=torch.float32)
        doc_freq = torch.tensor(graph.node_doc_freq, device=device, dtype=torch.float32)
        in_degree = torch.tensor(graph.in_degree, device=device, dtype=torch.float32)
        out_degree = torch.tensor(graph.out_degree, device=device, dtype=torch.float32)
        edge_index = torch.tensor(graph.edge_index, device=device, dtype=torch.long)

        node_repr = self.encode_nodes(
            node_token_emb,
            counts,
            doc_freq,
            in_degree,
            out_degree,
            edge_index,
            num_reviews=num_reviews,
        )
        item_repr = self.item_proj(item_emb.unsqueeze(0)).expand(node_repr.shape[0], -1)
        freq_repr = torch.stack([
            torch.log1p(counts),
            torch.log1p(doc_freq),
            counts / counts.max().clamp_min(1.0),
        ], dim=-1)
        freq_proj = F.pad(freq_repr, (0, self.hidden_dim - 3))
        logits = self.scorer(torch.cat([node_repr, item_repr, freq_proj], dim=-1)).squeeze(-1)
        return logits

    def select_evidence_and_negatives(
        self,
        utility_scores: torch.Tensor,
        graph: UserTokenGraph,
        *,
        top_m: int,
        ul_candidate_k: int,
        protected_token_ids: set[int],
    ) -> dict[str, Any]:
        if utility_scores.numel() == 0:
            return {
                "evidence_node_indices": np.empty((0,), dtype=np.int64),
                "evidence_token_ids": np.empty((0,), dtype=np.int64),
                "neg_node_indices": np.empty((0,), dtype=np.int64),
                "neg_token_ids": np.empty((0,), dtype=np.int64),
                "neg_weights": np.empty((0,), dtype=np.float32),
            }

        k = min(top_m, utility_scores.numel())
        top_values, top_indices = torch.topk(utility_scores, k=k)
        evidence_nodes = top_indices.detach().cpu().numpy().astype(np.int64)
        evidence_token_ids = graph.node_token_ids[evidence_nodes]

        neg_nodes, neg_weights = select_high_frequency_negatives(
            graph,
            evidence_nodes,
            top_k=ul_candidate_k,
            protected_token_ids=protected_token_ids,
        )
        neg_token_ids = graph.node_token_ids[neg_nodes] if neg_nodes.size else np.empty((0,), dtype=np.int64)
        return {
            "evidence_node_indices": evidence_nodes,
            "evidence_token_ids": evidence_token_ids,
            "neg_node_indices": neg_nodes,
            "neg_token_ids": neg_token_ids,
            "neg_weights": neg_weights,
            "utility_scores": utility_scores.detach().cpu().numpy(),
            "top_values": top_values.detach().cpu().numpy(),
        }


def pad_token_matrix(
    values: list,
    pad_value: int | float = -1,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not values:
        empty = torch.empty((0, 0), dtype=dtype or torch.long)
        return empty, torch.empty((0, 0), dtype=torch.bool)
    max_len = max(len(v) for v in values)
    batch = len(values)
    out_dtype = dtype or (torch.float32 if isinstance(pad_value, float) else torch.long)
    out = torch.full((batch, max_len), pad_value, dtype=out_dtype)
    mask = torch.zeros((batch, max_len), dtype=torch.bool)
    for i, arr in enumerate(values):
        if len(arr) == 0:
            continue
        out[i, :len(arr)] = torch.tensor(arr, dtype=out_dtype)
        mask[i, :len(arr)] = True
    return out, mask
