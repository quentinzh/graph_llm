"""GNN 图读出 + 并行 Semantic ID 打分（推荐路径，不依赖解释 LLM）。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_llm.dataload.semantic_id import SemanticIDBundle
from graph_llm.models.magnet import MagNetConv


class SIDParallelHeads(nn.Module):
    """对同一用户表示 g_u 并行预测各 SID 位置。"""

    def __init__(
        self,
        hidden_dim: int,
        text_positions: int,
        text_classes: int,
        use_rating: bool,
        use_popularity: bool,
        rating_classes: int = 9,
        pop_classes: int = 9,
    ):
        super().__init__()
        self.text_positions = text_positions
        self.text_classes = text_classes
        self.use_rating = use_rating
        self.use_popularity = use_popularity
        self.text_code_emb = nn.Parameter(torch.randn(text_positions, text_classes, hidden_dim) * 0.02)
        self.text_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(text_positions)])
        if use_rating:
            self.rating_code_emb = nn.Parameter(torch.randn(rating_classes, hidden_dim) * 0.02)
            self.rating_proj = nn.Linear(hidden_dim, hidden_dim)
        if use_popularity:
            self.pop_code_emb = nn.Parameter(torch.randn(pop_classes, hidden_dim) * 0.02)
            self.pop_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, user_repr: torch.Tensor) -> dict[str, torch.Tensor]:
        """user_repr: [B, H]"""
        b = user_repr.shape[0]
        text_logits = []
        for j in range(self.text_positions):
            q = self.text_proj[j](user_repr)
            logits = torch.einsum("bh,ch->bc", q, self.text_code_emb[j])
            text_logits.append(logits)
        out = {"text_logits": torch.stack(text_logits, dim=1)}
        if self.use_rating:
            q = self.rating_proj(user_repr)
            out["rating_logits"] = torch.einsum("bh,ch->bc", q, self.rating_code_emb)
        if self.use_popularity:
            q = self.pop_proj(user_repr)
            out["pop_logits"] = torch.einsum("bh,ch->bc", q, self.pop_code_emb)
        return out


class GraphReadout(nn.Module):
    """带 mask 的注意力图读出。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        self.default_user = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, node_repr: torch.Tensor, batch_index: torch.Tensor, batch_size: int) -> torch.Tensor:
        if node_repr.numel() == 0:
            return self.default_user.unsqueeze(0).expand(batch_size, -1)
        scores = self.attn(node_repr).squeeze(-1)
        # 分段 softmax 注意力池化（与逐 batch 循环等价）
        max_score = torch.full((batch_size,), float("-inf"), device=node_repr.device, dtype=scores.dtype)
        max_score.scatter_reduce_(0, batch_index, scores, reduce="amax", include_self=True)
        exp_scores = torch.exp(scores - max_score[batch_index])
        denom = torch.zeros(batch_size, device=node_repr.device, dtype=scores.dtype)
        denom.scatter_add_(0, batch_index, exp_scores)
        weights = exp_scores / denom[batch_index].clamp_min(1e-12)
        weighted = node_repr * weights.unsqueeze(-1)
        out = torch.zeros(batch_size, node_repr.shape[-1], device=node_repr.device, dtype=node_repr.dtype)
        out.scatter_add_(0, batch_index.unsqueeze(-1).expand_as(weighted), weighted)
        empty = denom <= 0
        if empty.any():
            out[empty] = self.default_user
        return out


class SIDRecommender(nn.Module):
    """推荐主干：历史图 GNN -> 读出 -> SID 并行分类。"""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        gnn_layers: int = 2,
        magnet_q: float = 0.15,
        text_positions: int = 4,
        text_classes: int = 256,
        use_rating: bool = True,
        use_popularity: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.type_emb = nn.Embedding(2, embed_dim)
        self.rating_emb = nn.Embedding(6, embed_dim)  # 0=缺失, 1-5 分桶
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs = nn.ModuleList(
            [MagNetConv(hidden_dim, hidden_dim, q=magnet_q) for _ in range(gnn_layers)]
        )
        self.readout = GraphReadout(hidden_dim)
        self.sid_heads = SIDParallelHeads(
            hidden_dim,
            text_positions=text_positions,
            text_classes=text_classes,
            use_rating=use_rating,
            use_popularity=use_popularity,
        )
        self.use_rating_sid = use_rating
        self.use_popularity_sid = use_popularity
        self.text_positions = text_positions
        # 证据节点打分（解释阶段辅助损失）
        self.evidence_scorer = nn.Linear(hidden_dim, 1)

    def _rating_bucket(self, rating_raw: float | None) -> int:
        if rating_raw is None:
            return 0
        r = int(round(rating_raw))
        return max(1, min(5, r))

    def _gnn_forward(
        self,
        node_emb: torch.Tensor,
        node_types: torch.Tensor,
        node_ratings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        type_vec = self.type_emb(node_types.clamp(0, 1))
        rating_vec = self.rating_emb(node_ratings.clamp(0, 5))
        x = self.input_proj(torch.cat([node_emb + type_vec, rating_vec], dim=-1))
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
        return x

    def encode_nodes_batch(
        self,
        node_emb: torch.Tensor,
        node_types: torch.Tensor,
        node_ratings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """返回 GNN 节点表示（不做图读出）。"""
        return self._gnn_forward(node_emb, node_types, node_ratings, edge_index, edge_weight)

    def encode_history_batch(
        self,
        node_emb: torch.Tensor,
        node_types: torch.Tensor,
        node_ratings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        batch_index: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        x = self._gnn_forward(node_emb, node_types, node_ratings, edge_index, edge_weight)
        return self.readout(x, batch_index, batch_size)

    def forward_sid_logits(self, user_repr: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.sid_heads(user_repr)

    def sid_loss(
        self,
        logits: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        *,
        lambda_rating: float,
        lambda_pop: float,
    ) -> torch.Tensor:
        text_logits = logits["text_logits"]
        text_tgt = targets["text"]
        per_pos = [F.cross_entropy(text_logits[:, j], text_tgt[:, j]) for j in range(text_logits.size(1))]
        loss = torch.stack(per_pos).mean()
        if "rating_logits" in logits and "rating" in targets:
            loss = loss + lambda_rating * F.cross_entropy(logits["rating_logits"], targets["rating"])
        if "pop_logits" in logits and "pop" in targets:
            loss = loss + lambda_pop * F.cross_entropy(logits["pop_logits"], targets["pop"])
        return loss

    def item_rec_loss(
        self,
        user_repr: torch.Tensor,
        target_item_indices: torch.Tensor,
        item_codes: torch.Tensor,
        *,
        lambda_rating: float,
        lambda_pop: float,
        text_sid_length: int,
        use_rating_sid: bool,
        use_popularity_sid: bool,
        temperature: float,
        code_log_priors: list[torch.Tensor] | None = None,
        pmi_lambda: float = 0.0,
    ) -> torch.Tensor:
        """商品级 full-softmax，与推理打分 S(u,i) 一致。"""
        logits = self.forward_sid_logits(user_repr)
        log_probs = logits_to_log_probs(logits)
        scores = vectorized_item_scores(
            log_probs,
            item_codes,
            text_sid_length=text_sid_length,
            use_rating_sid=use_rating_sid,
            use_popularity_sid=use_popularity_sid,
            lambda_rating=lambda_rating,
            lambda_pop=lambda_pop,
            code_log_priors=code_log_priors,
            pmi_lambda=pmi_lambda,
        )
        scores = scores / max(float(temperature), 1e-6)
        return F.cross_entropy(scores, target_item_indices)

    def code_contributions_for_items(
        self,
        user_repr: torch.Tensor,
        item_codes: torch.Tensor,
        target_item_indices: torch.Tensor,
        *,
        lambda_rating: float,
        lambda_pop: float,
        text_sid_length: int,
        use_rating_sid: bool,
        use_popularity_sid: bool,
    ) -> torch.Tensor:
        """目标商品各 SID 位的 log-prob 贡献，用于 rationale 向量 [B, num_positions]。"""
        logits = self.forward_sid_logits(user_repr)
        log_probs = logits_to_log_probs(logits)
        b = user_repr.shape[0]
        contribs: list[torch.Tensor] = []
        for j in range(text_sid_length):
            codes_j = item_codes[target_item_indices, j]
            contribs.append(log_probs["text"][j].gather(1, codes_j.unsqueeze(1)).squeeze(1))
        offset = text_sid_length
        if use_rating_sid and "rating" in log_probs:
            codes_j = item_codes[target_item_indices, offset]
            contribs.append(
                lambda_rating
                * log_probs["rating"].gather(1, codes_j.unsqueeze(1)).squeeze(1)
            )
            offset += 1
        if use_popularity_sid and "pop" in log_probs:
            codes_j = item_codes[target_item_indices, offset]
            contribs.append(
                lambda_pop * log_probs["pop"].gather(1, codes_j.unsqueeze(1)).squeeze(1)
            )
        return torch.stack(contribs, dim=1)

    def score_items_from_logprobs(
        self,
        log_probs: dict[str, torch.Tensor],
        item_codes: torch.Tensor,
        *,
        lambda_rating: float,
        lambda_pop: float,
        text_len: int,
        use_rating: bool,
        use_pop: bool,
    ) -> torch.Tensor:
        """item_codes: [N, L] 完整 SID"""
        scores = []
        for j in range(text_len):
            lp = log_probs["text"][j]
            scores.append(lp.gather(1, item_codes[:, j].unsqueeze(1)).squeeze(1))
        total = torch.stack(scores, dim=0).mean(dim=0)
        offset = text_len
        if use_rating:
            lp = log_probs["rating"]
            total = total + lambda_rating * lp.gather(1, item_codes[:, offset].unsqueeze(1)).squeeze(1)
            offset += 1
        if use_pop:
            lp = log_probs["pop"]
            total = total + lambda_pop * lp.gather(1, item_codes[:, offset].unsqueeze(1)).squeeze(1)
        return total


def build_item_code_matrix(sid_bundle: SemanticIDBundle, device: torch.device) -> torch.Tensor:
    """全商品 SID 矩阵 [num_items, L]。"""
    n = len(sid_bundle.item_index_to_raw)
    length = sid_bundle.sid_length()
    mat = torch.zeros((n, length), dtype=torch.long)
    for item_index, raw in sid_bundle.item_index_to_raw.items():
        mat[item_index] = torch.tensor(sid_bundle.codes_for_item(raw), dtype=torch.long)
    return mat.to(device)


def logits_to_log_probs(logits: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {"text": [F.log_softmax(logits["text_logits"][:, j], dim=-1) for j in range(logits["text_logits"].size(1))]}
    if "rating_logits" in logits:
        out["rating"] = F.log_softmax(logits["rating_logits"], dim=-1)
    if "pop_logits" in logits:
        out["pop"] = F.log_softmax(logits["pop_logits"], dim=-1)
    return out


def build_code_log_prior_tensors(
    sid_bundle: SemanticIDBundle,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """将 bundle 中的 log 先验转为 GPU 张量列表。"""
    if not sid_bundle.code_log_priors:
        return []
    return [
        torch.tensor(p, device=device, dtype=dtype) for p in sid_bundle.code_log_priors
    ]


def vectorized_item_scores(
    log_probs: dict[str, torch.Tensor | list[torch.Tensor]],
    item_codes: torch.Tensor,
    *,
    text_sid_length: int,
    use_rating_sid: bool,
    use_popularity_sid: bool,
    lambda_rating: float,
    lambda_pop: float,
    code_log_priors: list[torch.Tensor] | None = None,
    pmi_lambda: float = 0.0,
) -> torch.Tensor:
    """对 batch 内每个用户计算全商品分数 [B, N]，与 brief 中 S(u,i) 一致（文本位取平均）。"""
    text_lps = log_probs["text"]
    batch_size = text_lps[0].shape[0]
    num_items = item_codes.shape[0]
    scores = torch.zeros(batch_size, num_items, device=item_codes.device, dtype=text_lps[0].dtype)
    for j in range(text_sid_length):
        codes_j = item_codes[:, j]
        part = text_lps[j].gather(1, codes_j.unsqueeze(0).expand(batch_size, -1))
        if pmi_lambda > 0 and code_log_priors and j < len(code_log_priors):
            prior = code_log_priors[j][codes_j].unsqueeze(0).expand(batch_size, -1)
            part = part - float(pmi_lambda) * prior
        scores += part
    scores = scores / max(text_sid_length, 1)
    offset = text_sid_length
    pos_idx = text_sid_length
    if use_rating_sid and "rating" in log_probs:
        codes_j = item_codes[:, offset]
        part = log_probs["rating"].gather(1, codes_j.unsqueeze(0).expand(batch_size, -1))
        if pmi_lambda > 0 and code_log_priors and pos_idx < len(code_log_priors):
            prior = code_log_priors[pos_idx][codes_j].unsqueeze(0).expand(batch_size, -1)
            part = part - float(pmi_lambda) * prior
        scores = scores + lambda_rating * part
        offset += 1
        pos_idx += 1
    if use_popularity_sid and "pop" in log_probs:
        codes_j = item_codes[:, offset]
        part = log_probs["pop"].gather(1, codes_j.unsqueeze(0).expand(batch_size, -1))
        if pmi_lambda > 0 and code_log_priors and pos_idx < len(code_log_priors):
            prior = code_log_priors[pos_idx][codes_j].unsqueeze(0).expand(batch_size, -1)
            part = part - float(pmi_lambda) * prior
        scores = scores + lambda_pop * part
    return scores
