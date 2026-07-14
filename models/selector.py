"""Lightweight GNN-based neural evidence selector."""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_llm.models.token_graph import (
    UserTokenGraph,
)
from graph_llm.dataload.tail_stats import TailTokenStats


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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Sequential(
            # 节点编码只使用冻结语义向量；不再拼接频率或度数统计量。
            nn.Linear(embed_dim, hidden_dim),
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

    def encode_nodes(
        self,
        node_token_emb: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_proj(node_token_emb)
        for conv in self.convs:
            x = conv(x, edge_index)
        return x

    def forward_single(
        self,
        graph: UserTokenGraph,
        node_token_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        if graph.num_nodes == 0 or node_token_emb.numel() == 0:
            return torch.empty((0,), device=item_emb.device)

        device = item_emb.device
        counts = torch.tensor(graph.node_counts, device=device, dtype=torch.float32)
        doc_freq = torch.tensor(graph.node_doc_freq, device=device, dtype=torch.float32)
        edge_index = torch.tensor(graph.edge_index, device=device, dtype=torch.long)

        node_repr = self.encode_nodes(
            node_token_emb,
            edge_index,
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

    def sampled_bce_loss(
        self,
        utility_scores: torch.Tensor,
        graph: UserTokenGraph,
        positive_token_ids: set[int],
        tail_stats: TailTokenStats,
    ) -> torch.Tensor:
        """Supervise graph evidence with one positive set and sampled negatives.

        The loss retains the standard weighted BCE form.  For every positive
        node it samples four current hard negatives, two popularity-matched
        negatives, and two globally popular negatives.  The positive weight is
        scaled by |N| / |P| so eight negatives do not drown out one positive.
        """
        if utility_scores.numel() == 0 or graph.num_nodes == 0:
            return utility_scores.new_tensor(0.0)

        node_ids = [int(token_id) for token_id in graph.node_token_ids.tolist()]
        positive_indices = [
            index for index, token_id in enumerate(node_ids)
            if token_id in positive_token_ids
        ]
        if not positive_indices:
            # No graph-grounded label is available; avoid treating every node as negative.
            return utility_scores.new_tensor(0.0)

        positive_set = set(positive_indices)
        candidate_indices = [
            index for index in range(graph.num_nodes)
            if index not in positive_set
        ]
        if not candidate_indices:
            return utility_scores.new_tensor(0.0)

        # Hard negatives are selected on detached scores only; BCE still sends
        # gradients through every selected negative logit.
        hard_sorted = sorted(
            candidate_indices,
            key=lambda index: (-float(utility_scores[index].detach().item()), index),
        )
        popular_sorted = sorted(
            candidate_indices,
            key=lambda index: (-tail_stats.document_frequency(node_ids[index]), index),
        )

        negative_indices: list[int] = []
        for positive_index in positive_indices:
            selected_for_positive: list[int] = []

            def take(candidates: list[int], limit: int) -> None:
                for candidate in candidates:
                    if candidate in selected_for_positive:
                        continue
                    selected_for_positive.append(candidate)
                    if len(selected_for_positive) >= limit:
                        return

            take(hard_sorted, 4)

            positive_bucket = tail_stats.frequency_bucket(node_ids[positive_index])
            same_bucket = [
                index for index in candidate_indices
                if tail_stats.frequency_bucket(node_ids[index]) == positive_bucket
            ]
            if len(same_bucket) < 2:
                same_bucket = sorted(
                    candidate_indices,
                    key=lambda index: (
                        abs(tail_stats.frequency_bucket(node_ids[index]) - positive_bucket),
                        index,
                    ),
                )
            if same_bucket:
                same_bucket = random.sample(same_bucket, k=len(same_bucket))
            take(same_bucket, 6)
            take(popular_sorted, 8)
            take(hard_sorted + popular_sorted + candidate_indices, 8)
            negative_indices.extend(selected_for_positive[:8])

        if not negative_indices:
            return utility_scores.new_tensor(0.0)

        positive_tensor = torch.tensor(
            positive_indices, device=utility_scores.device, dtype=torch.long,
        )
        negative_tensor = torch.tensor(
            negative_indices, device=utility_scores.device, dtype=torch.long,
        )
        balance = float(len(negative_indices)) / float(len(positive_indices))
        positive_weights = torch.tensor(
            [balance * tail_stats.tail_weight(node_ids[index]) for index in positive_indices],
            device=utility_scores.device,
            dtype=utility_scores.dtype,
        )
        positive_loss = F.softplus(-utility_scores[positive_tensor]) * positive_weights
        negative_loss = F.softplus(utility_scores[negative_tensor])
        normalizer = positive_weights.sum() + negative_loss.new_tensor(float(len(negative_indices)))
        return (positive_loss.sum() + negative_loss.sum()) / normalizer.clamp_min(1.0)

    def select_evidence(
        self,
        utility_scores: torch.Tensor,
        graph: UserTokenGraph,
        *,
        top_m: int,
    ) -> torch.Tensor:
        """Return hard top-M evidence ids for the legacy non-differentiable bonus."""
        if utility_scores.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=utility_scores.device)

        k = min(top_m, utility_scores.numel())
        top_indices = torch.topk(utility_scores, k=k).indices
        ids = torch.tensor(graph.node_token_ids, device=utility_scores.device, dtype=torch.long)
        return ids[top_indices].detach()


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
