"""MagNet 图卷积层（推荐路径 GNN 主干）。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


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
    """从有向边构造 MagNet 的复 Hermitian 邻接（实部/虚部）与行归一化因子。"""
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
    """单层 MagNet-GCN（K=1）：在有向图上做复值谱式消息传递。"""

    def __init__(self, in_dim: int, out_dim: int, q: float = 0.15):
        super().__init__()
        self.q = float(q)
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

        scale = inv_sqrt_deg.unsqueeze(1) * inv_sqrt_deg.unsqueeze(0)
        h_real_norm = h_real * scale
        h_imag_norm = h_imag * scale

        out_real = h_real_norm @ x
        out_imag = h_imag_norm @ x
        out_real, out_imag = _complex_relu(out_real, out_imag)
        return self.unwind(torch.cat([out_real, out_imag], dim=-1))
