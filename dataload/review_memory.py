"""冻结评论向量的泄漏安全检索与批处理。"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def _history_fingerprint(history) -> str:
    """为评论历史构造稳定指纹，防止复用到不匹配的数据切分。"""
    digest = hashlib.sha256()
    columns = [
        "raw_user",
        "raw_item",
        "review_text",
        "_review_source_split",
        "_review_local_idx",
    ]
    for values in history[columns].itertuples(index=False, name=None):
        digest.update("\x1f".join(str(value) for value in values).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class ReviewMemoryBank:
    """在 CPU 保存冻结评论向量，并按当前 user-item 对执行 Top-K 检索。

    用户分支使用当前物品向量检索该用户历史；物品分支使用检索到的用户
    历史均值作为动态用户查询，再检索当前物品的历史评论。
    """

    def __init__(
        self,
        *,
        embeddings: torch.Tensor,
        raw_users: list[str],
        raw_items: list[str],
        source_splits: list[str],
        local_indices: list[int],
        top_k_user: int,
        top_k_item: int,
    ):
        if embeddings.ndim != 2:
            raise ValueError("review embeddings must have shape [N, D]")
        if len(raw_users) != embeddings.shape[0]:
            raise ValueError("review metadata and embedding rows must have equal length")
        self.embeddings = F.normalize(
            embeddings.float().cpu(),
            dim=-1,
        ).to(torch.float16)
        self.embedding_dim = int(self.embeddings.shape[1])
        self.raw_users = [str(value) for value in raw_users]
        self.raw_items = [str(value) for value in raw_items]
        self.source_splits = [str(value) for value in source_splits]
        self.local_indices = [int(value) for value in local_indices]
        self.top_k_user = int(top_k_user)
        self.top_k_item = int(top_k_item)
        if self.top_k_user < 1 or self.top_k_item < 1:
            raise ValueError("review Top-K values must be positive")

        self.user_to_rows = defaultdict(list)
        self.item_to_rows = defaultdict(list)
        for row_idx, (raw_user, raw_item) in enumerate(zip(self.raw_users, self.raw_items)):
            self.user_to_rows[raw_user].append(row_idx)
            self.item_to_rows[raw_item].append(row_idx)

    @classmethod
    def build_or_load(
        cls,
        history,
        embedding_encoder,
        *,
        cache_path: Path,
        top_k_user: int,
        top_k_item: int,
        encode_batch_size: int = 64,
        rebuild: bool = False,
    ):
        required = {
            "raw_user",
            "raw_item",
            "review_text",
            "_review_source_split",
            "_review_local_idx",
        }
        missing = required - set(history.columns)
        if missing:
            raise ValueError(f"review history is missing columns: {sorted(missing)}")

        history = history.reset_index(drop=True)
        fingerprint = _history_fingerprint(history)
        cache_path = Path(cache_path)
        payload = None
        if cache_path.exists() and not rebuild:
            try:
                candidate = torch.load(cache_path, map_location="cpu", weights_only=False)
                if (
                    candidate.get("fingerprint") == fingerprint
                    and int(candidate.get("embedding_dim", -1)) == embedding_encoder.hidden_size
                ):
                    payload = candidate
            except Exception as exc:
                print(f"WARNING: review embedding bank cache is invalid ({exc}); rebuilding.")

        if payload is None:
            print(f"Encoding {len(history)} historical reviews for {cache_path.name} ...")
            chunks = []
            texts = history["review_text"].fillna("").astype(str).tolist()
            for start in range(0, len(texts), encode_batch_size):
                vectors = embedding_encoder.encode_texts(
                    texts[start:start + encode_batch_size],
                    batch_size=encode_batch_size,
                    # 评论库本身已有整库缓存，不再生成海量逐文本缓存文件。
                    use_cache=False,
                )
                chunks.append(vectors.detach().float().cpu())
            embeddings = (
                torch.cat(chunks, dim=0)
                if chunks
                else torch.empty((0, embedding_encoder.hidden_size), dtype=torch.float32)
            )
            payload = {
                "fingerprint": fingerprint,
                "embedding_dim": int(embedding_encoder.hidden_size),
                # float16 足以承担余弦检索，并将常驻 CPU 内存和缓存减半。
                "embeddings": embeddings.to(torch.float16),
                "raw_users": history["raw_user"].astype(str).tolist(),
                "raw_items": history["raw_item"].astype(str).tolist(),
                "source_splits": history["_review_source_split"].astype(str).tolist(),
                "local_indices": history["_review_local_idx"].astype(int).tolist(),
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".part")
            try:
                torch.save(payload, tmp_path)
                os.replace(tmp_path, cache_path)
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            print(f"Loaded review embedding bank from {cache_path}")

        return cls(
            embeddings=payload["embeddings"],
            raw_users=payload["raw_users"],
            raw_items=payload["raw_items"],
            source_splits=payload["source_splits"],
            local_indices=payload["local_indices"],
            top_k_user=top_k_user,
            top_k_item=top_k_item,
        )

    def subset(self, source_splits):
        """从总历史库派生共享配置的切分视图，不重新运行 embedding 模型。"""
        source_splits = {str(value) for value in source_splits}
        rows = [
            row_idx
            for row_idx, split_name in enumerate(self.source_splits)
            if split_name in source_splits
        ]
        row_tensor = torch.tensor(rows, dtype=torch.long)
        return ReviewMemoryBank(
            embeddings=self.embeddings.index_select(0, row_tensor),
            raw_users=[self.raw_users[row_idx] for row_idx in rows],
            raw_items=[self.raw_items[row_idx] for row_idx in rows],
            source_splits=[self.source_splits[row_idx] for row_idx in rows],
            local_indices=[self.local_indices[row_idx] for row_idx in rows],
            top_k_user=self.top_k_user,
            top_k_item=self.top_k_item,
        )

    def _eligible_rows(self, rows, context):
        """训练时排除当前行；验证/测试的目标本来就不在对应历史库中。"""
        split_name = str(context["split_name"])
        local_idx = int(context["local_idx"])
        return [
            row_idx
            for row_idx in rows
            if not (
                self.source_splits[row_idx] == split_name
                and self.local_indices[row_idx] == local_idx
            )
        ]

    def _topk(self, rows, query, k):
        if not rows:
            return []
        row_tensor = torch.tensor(rows, dtype=torch.long)
        candidates = self.embeddings.index_select(0, row_tensor).float()
        query = F.normalize(query.float().cpu(), dim=-1)
        scores = candidates @ query
        keep = min(int(k), len(rows))
        selected = torch.topk(scores, k=keep, largest=True, sorted=True).indices
        return row_tensor.index_select(0, selected).tolist()

    def prepare_batch(self, review_contexts, item_queries, device):
        """返回 prefix projector 所需的固定形状评论张量与 mask。"""
        if item_queries.ndim != 2 or item_queries.shape[0] != len(review_contexts):
            raise ValueError("item_queries must have shape [batch, embedding_dim]")
        if item_queries.shape[1] != self.embedding_dim:
            raise ValueError(
                f"item query dim {item_queries.shape[1]} != review dim {self.embedding_dim}"
            )

        batch_size = len(review_contexts)
        user_reviews = torch.zeros(
            (batch_size, self.top_k_user, self.embedding_dim), dtype=torch.float32
        )
        item_reviews = torch.zeros(
            (batch_size, self.top_k_item, self.embedding_dim), dtype=torch.float32
        )
        user_mask = torch.zeros((batch_size, self.top_k_user), dtype=torch.bool)
        item_mask = torch.zeros((batch_size, self.top_k_item), dtype=torch.bool)
        user_queries = torch.zeros((batch_size, self.embedding_dim), dtype=torch.float32)

        item_queries_cpu = item_queries.detach().float().cpu()
        for batch_idx, context in enumerate(review_contexts):
            user_rows = self._eligible_rows(
                self.user_to_rows.get(str(context["raw_user"]), []),
                context,
            )
            selected_user_rows = self._topk(
                user_rows,
                item_queries_cpu[batch_idx],
                self.top_k_user,
            )
            if selected_user_rows:
                count = len(selected_user_rows)
                selected = self.embeddings[selected_user_rows].float()
                user_reviews[batch_idx, :count] = selected
                user_mask[batch_idx, :count] = True
                user_queries[batch_idx] = F.normalize(selected.mean(dim=0), dim=-1)

            item_rows = self._eligible_rows(
                self.item_to_rows.get(str(context["raw_item"]), []),
                context,
            )
            # 冷启动用户没有历史时，退化为当前物品语义查询，仍能选出代表性物品评论。
            item_query = (
                user_queries[batch_idx]
                if selected_user_rows
                else item_queries_cpu[batch_idx]
            )
            selected_item_rows = self._topk(item_rows, item_query, self.top_k_item)
            if selected_item_rows:
                count = len(selected_item_rows)
                item_reviews[batch_idx, :count] = self.embeddings[selected_item_rows].float()
                item_mask[batch_idx, :count] = True

        return {
            "user_review_embeddings": user_reviews.to(device),
            "user_review_mask": user_mask.to(device),
            "item_review_embeddings": item_reviews.to(device),
            "item_review_mask": item_mask.to(device),
            "item_review_query": F.normalize(item_queries_cpu, dim=-1).to(device),
            "user_review_query": user_queries.to(device),
        }


def mark_review_history(dataframe, split_name: str):
    """给历史行写入可用于防泄漏排除的稳定身份。"""
    history = dataframe.reset_index(drop=True).copy()
    history["_review_source_split"] = str(split_name)
    history["_review_local_idx"] = range(len(history))
    return history


__all__ = ["ReviewMemoryBank", "mark_review_history"]
