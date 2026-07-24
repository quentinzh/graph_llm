"""Lightweight GNN-based neural evidence selector."""

from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_llm.models.token_graph import (
    UserTokenGraph,
)
from graph_llm.dataload.tail_stats import TailTokenStats


def _complex_relu(real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MagNet complex ReLU：仅保留幅角落在 [-pi/2, pi/2) 的复数分量。"""
    mask = real >= 0
    return real * mask, imag * mask


def _build_magnetic_adjacency(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    *,
    q: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从有向边构造 MagNet 的复 Hermitian 邻接（实部/虚部）与行归一化因子。

    连通强度写入幅值，边方向写入相位；自环保证孤立节点也能更新。
    """
    if num_nodes <= 0:
        empty = torch.empty((0, 0), device=device, dtype=dtype)
        return empty, empty, torch.empty((0,), device=device, dtype=dtype)

    directed = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
    if edge_index.numel() > 0:
        src, dst = edge_index
        weights = edge_weight
        if weights is None:
            weights = torch.ones(src.shape[0], device=device, dtype=dtype)
        else:
            weights = weights.to(device=device, dtype=dtype)
        flat_idx = src * num_nodes + dst
        directed_flat = torch.zeros(num_nodes * num_nodes, device=device, dtype=dtype)
        directed_flat.index_add_(0, flat_idx, weights)
        directed = directed_flat.view(num_nodes, num_nodes)

    # 对称部分编码“是否相连”，反对称部分编码“朝哪边”。
    sym = 0.5 * (directed + directed.transpose(0, 1))
    antisym = directed - directed.transpose(0, 1)
    phase = (2.0 * math.pi * float(q)) * antisym
    h_real = sym * torch.cos(phase)
    h_imag = sym * torch.sin(phase)

    eye = torch.eye(num_nodes, device=device, dtype=dtype)
    h_real = h_real + eye
    sym_with_loop = sym + eye
    deg = sym_with_loop.sum(dim=1).clamp_min(1.0)
    inv_sqrt_deg = deg.pow(-0.5)
    return h_real, h_imag, inv_sqrt_deg


class MagNetConv(nn.Module):
    """单层 MagNet-GCN（K=1）：在有向 token 图上做复值谱式消息传递。"""

    def __init__(self, in_dim: int, out_dim: int, q: float = 0.15):
        super().__init__()
        self.q = float(q)
        # 复卷积输出先展开为 [real|imag]，再投影回实特征，便于堆叠多层。
        self.unwind = nn.Linear(in_dim * 2, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.numel() == 0:
            return x

        num_nodes = x.shape[0]
        device, dtype = x.device, x.dtype
        h_real, h_imag, inv_sqrt_deg = _build_magnetic_adjacency(
            num_nodes,
            edge_index,
            edge_weight,
            q=self.q,
            device=device,
            dtype=dtype,
        )
        if h_real.numel() == 0:
            return self.unwind(torch.cat([x, torch.zeros_like(x)], dim=-1))

        # D^{-1/2} H D^{-1/2}，对稀疏小图直接做稠密乘法即可。
        scale = inv_sqrt_deg.unsqueeze(1) * inv_sqrt_deg.unsqueeze(0)
        h_real_norm = h_real * scale
        h_imag_norm = h_imag * scale

        out_real = h_real_norm @ x
        out_imag = h_imag_norm @ x
        out_real, out_imag = _complex_relu(out_real, out_imag)
        return self.unwind(torch.cat([out_real, out_imag], dim=-1))


class EvidenceSelector(nn.Module):
    """Score token-graph nodes for current user-item explanation utility."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        gnn_layers: int = 2,
        magnet_q: float = 0.15,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.magnet_q = float(magnet_q)
        self.input_proj = nn.Sequential(
            # 节点编码只使用冻结语义向量；不再拼接频率或度数统计量。
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs = nn.ModuleList([
            MagNetConv(hidden_dim, hidden_dim, q=self.magnet_q)
            for _ in range(gnn_layers)
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
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_proj(node_token_emb)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
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
        edge_weight = None
        if graph.edge_weight.size > 0:
            edge_weight = torch.tensor(graph.edge_weight, device=device, dtype=torch.float32)

        node_repr = self.encode_nodes(
            node_token_emb,
            edge_index,
            edge_weight,
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
        feature_token_ids: set[int] | None = None,
        feature_positive_weight: float = 1.0,
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
        feature_token_ids = feature_token_ids or set()
        positive_weights = torch.tensor(
            [
                # 仅对目标解释中的核心 feature 加权，避免把邻近词误当作 feature。
                balance
                * tail_stats.tail_weight(node_ids[index])
                * (
                    feature_positive_weight
                    if node_ids[index] in feature_token_ids
                    else 1.0
                )
                for index in positive_indices
            ],
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
