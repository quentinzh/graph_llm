"""Profile-aware dataloader and collate utilities."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from graph_llm.dataload.legacy_data import (
    MyDataset,
    assert_profile_coverage,
    dataset_split,
    load_profile_cache,
    profile_text_from_record,
    read_split_indices,
    tokenize_profile_text,
    tokenize_target_item_text,
    tokenizer_eos_id,
    tokenizer_pad_id,
    tokenizer_special_ids,
)

from graph_llm.aux.prompt_utils import item_meta_from_row
from graph_llm.dataload.cache import GraphCacheManager
from graph_llm.models.token_graph import UserTokenGraph, batch_graphs


FEATURE_STOPWORDS = {
    "", ".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "'", "\"",
    "'s", "'m", "'ve", "n't", "'re", "'d", "'ll",
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "he", "her", "his", "i", "in", "is", "it", "its", "me", "my", "of", "on",
    "or", "our", "she", "that", "the", "their", "them", "there", "they",
    "this", "to", "was", "we", "were", "with", "you", "your",
    "user", "profile", "current", "item", "information", "title",
    "description", "explanation", "useful", "token", "evidence", "none",
}

# 保持 feature 为辅助损失的主要监督信号，同时为其附近的内容词提供少量短语上下文。
FEATURE_CORE_WEIGHT = 2.0
FEATURE_NEIGHBOR_WEIGHTS = {
    1: 0.2,
    2: 0.1,
}


class GraphDataset(Dataset):
    """Dataset wrapper with split metadata."""

    def __init__(self, dataframe, split_name: str):
        self.split_name = split_name
        df = dataframe.reset_index(drop=False).rename(columns={"index": "row_key"})
        # 初始化时一次性物化行，避免 __getitem__ 反复走 pandas iloc。
        self.rows: list[dict] = []
        for local_idx, (_, row) in enumerate(df.iterrows()):
            item = row.to_dict()
            item["local_idx"] = local_idx
            item["split_name"] = split_name
            self.rows.append(item)
        self.features = dataframe["keyword_words"].tolist() if "keyword_words" in dataframe else []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class GraphCollater:
    """Collate explanations with profile, item text, and cached token graphs."""

    def __init__(
        self,
        *,
        max_step=1,
        word=40,
        tokenizer=None,
        profile_records=None,
        max_profile_tokens=512,
        item_meta=None,
        max_target_item_tokens=64,
        item_description_mode="keywords",
        graph_manager: GraphCacheManager | None = None,
        split_name: str = "train",
        materialize_graph_batch: bool = False,
    ):
        self.max_step = max_step
        self.cur_step = 1
        self.word = word
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer_pad_id(tokenizer)
        self.eos_token_id = tokenizer_eos_id(tokenizer)
        self.feature_ignored_token_ids = set(tokenizer_special_ids(tokenizer))
        self.feature_ignored_token_ids.add(self.pad_token_id)
        self.feature_ignored_token_ids.add(self.eos_token_id)
        self.profile_records = profile_records or {}
        self.max_profile_tokens = max_profile_tokens
        self.item_meta = item_meta or {}
        self.max_target_item_tokens = max_target_item_tokens
        self.item_description_mode = item_description_mode
        self.graph_manager = graph_manager
        self.split_name = split_name
        # True：在 collate 内直接 batch 图（兼容 SimDPO/单测）；False：只返回 local_idx，主进程取图。
        self.materialize_graph_batch = bool(materialize_graph_batch)
        self._keep_feature_token_cache: dict[int, bool] = {}
        # local_idx -> 预计算的 profile/item/feature 字段
        self._sample_cache: dict[int, dict] = {}

    def bind_dataset_cache(self, dataset: GraphDataset) -> None:
        """在折初始化时预计算 collate 热路径上的分词与 feature 权重。"""
        profile_by_user: dict[str, list[int]] = {}
        target_by_item: dict[str, list[int]] = {}
        self._sample_cache = {}
        for idx in range(len(dataset)):
            row = dataset[idx]
            raw_user = str(row["raw_user"]) if "raw_user" in row else str(row["user"])
            raw_item = str(row["raw_item"]) if "raw_item" in row else str(row["item"])
            if raw_user not in profile_by_user:
                profile_by_user[raw_user] = self._profile_ids(row)
            if raw_item not in target_by_item:
                target_by_item[raw_item] = self._target_item_ids(row)
            text_ids = list(row["text"][: self.word])
            if len(text_ids) == 0:
                text_ids = [self.eos_token_id]
            feature_weights = self._feature_position_weights(
                text_ids,
                row.get("keyword_words", ""),
            )
            self._sample_cache[idx] = {
                "text_ids": text_ids,
                "feature_weights": feature_weights,
                "profile_ids": profile_by_user[raw_user],
                "target_item_ids": target_by_item[raw_item],
            }

    def _graph_for_row(self, row) -> UserTokenGraph:
        if self.graph_manager is None:
            return UserTokenGraph.empty()
        local_idx = int(row["local_idx"])
        split_name = str(row.get("split_name", self.split_name))
        return self.graph_manager.get_graph(split_name, local_idx)

    def _profile_ids(self, row):
        if self.tokenizer is None:
            return []
        raw_user = str(row["raw_user"]) if "raw_user" in row else str(row["user"])
        text = profile_text_from_record(self.profile_records.get(raw_user))
        return tokenize_profile_text(self.tokenizer, text, self.max_profile_tokens)

    def _target_item_ids(self, row):
        if self.tokenizer is None:
            return []
        raw_item = str(row["raw_item"]) if "raw_item" in row else str(row["item"])
        return tokenize_target_item_text(
            self.tokenizer,
            raw_item,
            self.item_meta,
            self.max_target_item_tokens,
            description_mode=self.item_description_mode,
        )

    def _keep_feature_token(self, token_id: int) -> bool:
        token_id = int(token_id)
        cached = self._keep_feature_token_cache.get(token_id)
        if cached is not None:
            return cached
        if token_id < 0 or token_id in self.feature_ignored_token_ids:
            self._keep_feature_token_cache[token_id] = False
            return False
        if self.tokenizer is None:
            self._keep_feature_token_cache[token_id] = True
            return True
        surface = self.tokenizer.decode([token_id], skip_special_tokens=True).strip().lower()
        normalized = surface.strip(" \t\r\n.,!?;:'\"()[]{}")
        if not normalized or normalized in FEATURE_STOPWORDS:
            self._keep_feature_token_cache[token_id] = False
            return False
        if normalized.isdigit() or len(normalized) <= 2:
            self._keep_feature_token_cache[token_id] = False
            return False
        result = any(ch.isalpha() for ch in normalized)
        self._keep_feature_token_cache[token_id] = result
        return result

    def _keyword_token_variants(self, keyword):
        if self.tokenizer is None or keyword is None:
            return []
        keyword = str(keyword).strip()
        if not keyword:
            return []
        texts = [keyword, f" {keyword}"]
        variants = []
        seen = set()
        for text in texts:
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            ids = tuple(int(x) for x in ids)
            if ids and ids not in seen:
                variants.append(list(ids))
                seen.add(ids)
        return variants

    def _feature_position_weights(self, ids, keyword):
        """构造 feature 及其局部上下文的位置权重。

        feature 本身使用较大权重以优先保障 FMR；仅对相邻的有效内容词
        赋予较小权重，帮助模型学习 feature 所在的短语表达。
        """
        weights = [0.0] * len(ids)
        feature_positions = set()
        for variant in self._keyword_token_variants(keyword):
            width = len(variant)
            if width == 0 or width > len(ids):
                continue
            for start in range(0, len(ids) - width + 1):
                if ids[start:start + width] != variant:
                    continue
                for offset, token_id in enumerate(variant):
                    if self._keep_feature_token(token_id):
                        feature_positions.add(start + offset)

        for position in feature_positions:
            weights[position] = FEATURE_CORE_WEIGHT

        # 多 token feature 的内部位置保持 core 权重；只扩展到 feature span 外的内容词。
        for position in feature_positions:
            for distance, neighbor_weight in FEATURE_NEIGHBOR_WEIGHTS.items():
                for neighbor in (position - distance, position + distance):
                    if (
                        neighbor < 0
                        or neighbor >= len(ids)
                        or neighbor in feature_positions
                        or not self._keep_feature_token(ids[neighbor])
                    ):
                        continue
                    weights[neighbor] = max(weights[neighbor], neighbor_weight)
        return weights

    def __call__(self, data):
        input_ids, rating = [], []
        profile_ids, target_item_ids = [], []
        feature_weight_rows = []
        graphs = []
        local_idxs = []
        item_texts = []
        item_titles = []
        raw_users = []
        review_contexts = []
        max_length = max([
            min(self.word, max(len(x["text"]), 1))
            for x in data
        ])

        for x in data:
            local_idx = int(x["local_idx"])
            cached = self._sample_cache.get(local_idx)
            if cached is not None:
                ids = list(cached["text_ids"][:max_length])
                feature_weights = list(cached["feature_weights"][:max_length])
                profile_row_ids = cached["profile_ids"]
                target_row_ids = cached["target_item_ids"]
            else:
                ids = list(x["text"][:max_length])
                feature_weights = self._feature_position_weights(
                    ids,
                    x.get("keyword_words", ""),
                )
                profile_row_ids = self._profile_ids(x)
                target_row_ids = self._target_item_ids(x)
            if len(ids) == 0:
                ids = [self.eos_token_id]
                feature_weights = [0.0]

            pad_len = max_length - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            feature_weight_rows.append(feature_weights + [0.0] * pad_len)
            profile_ids.append(profile_row_ids)
            target_item_ids.append(target_row_ids)
            rating.append(x["rating"])
            local_idxs.append(local_idx)
            if self.materialize_graph_batch:
                graphs.append(self._graph_for_row(x))
            raw_item = str(x["raw_item"]) if "raw_item" in x else str(x["item"])
            title, _description, item_text = item_meta_from_row(raw_item, self.item_meta)
            item_titles.append(title)
            item_texts.append(item_text)
            raw_user = str(x["raw_user"]) if "raw_user" in x else str(x["user"])
            raw_users.append(raw_user)
            # 评论检索只依赖样本身份，不把真实解释文本传给模型。
            # split_name + local_idx 用于训练时精确排除当前评论，避免标签泄漏。
            review_contexts.append({
                "raw_user": raw_user,
                "raw_item": raw_item,
                "split_name": str(x.get("split_name", self.split_name)),
                "local_idx": int(x["local_idx"]),
            })

        self.cur_step += 1

        max_profile_len = max([len(ids) for ids in profile_ids], default=0)
        if max_profile_len == 0:
            profile_tensor = torch.empty((len(profile_ids), 0), dtype=torch.long)
            profile_mask = torch.empty((len(profile_ids), 0), dtype=torch.long)
        else:
            padded_profiles, masks = [], []
            for ids in profile_ids:
                pad_len = max_profile_len - len(ids)
                padded_profiles.append(ids + [self.pad_token_id] * pad_len)
                masks.append([1] * len(ids) + [0] * pad_len)
            profile_tensor = torch.tensor(padded_profiles, dtype=torch.long)
            profile_mask = torch.tensor(masks, dtype=torch.long)

        max_target_item_len = max([len(ids) for ids in target_item_ids], default=0)
        if max_target_item_len == 0:
            target_item_tensor = torch.empty((len(target_item_ids), 0), dtype=torch.long)
            target_item_mask = torch.empty((len(target_item_ids), 0), dtype=torch.long)
        else:
            padded_target_items, target_masks = [], []
            for ids in target_item_ids:
                pad_len = max_target_item_len - len(ids)
                padded_target_items.append(ids + [self.pad_token_id] * pad_len)
                target_masks.append([1] * len(ids) + [0] * pad_len)
            target_item_tensor = torch.tensor(padded_target_items, dtype=torch.long)
            target_item_mask = torch.tensor(target_masks, dtype=torch.long)

        feature_position_weights = torch.tensor(feature_weight_rows, dtype=torch.float32)
        feature_position_mask = feature_position_weights > 0

        common_tail = (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(rating, dtype=torch.long),
            profile_tensor,
            profile_mask,
            target_item_tensor,
            target_item_mask,
        )
        if self.materialize_graph_batch:
            batched_graph = batch_graphs(graphs)
            graph_tensors = {
                "node_token_ids": torch.tensor(batched_graph["node_token_ids"], dtype=torch.long),
                "node_counts": torch.tensor(batched_graph["node_counts"], dtype=torch.float32),
                "node_doc_freq": torch.tensor(batched_graph["node_doc_freq"], dtype=torch.float32),
                "node_in_degree": torch.tensor(batched_graph["node_in_degree"], dtype=torch.float32),
                "node_out_degree": torch.tensor(batched_graph["node_out_degree"], dtype=torch.float32),
                "edge_index": torch.tensor(batched_graph["edge_index"], dtype=torch.long),
                "batch_index": torch.tensor(batched_graph["batch_index"], dtype=torch.long),
                "num_nodes_per_graph": torch.tensor(
                    batched_graph["num_nodes_per_graph"], dtype=torch.long
                ),
            }
            return (
                *common_tail,
                graph_tensors,
                graphs,
                item_texts,
                item_titles,
                raw_users,
                feature_position_mask,
                feature_position_weights,
                review_contexts,
            )
        return (
            *common_tail,
            local_idxs,
            item_texts,
            item_titles,
            raw_users,
            feature_position_mask,
            feature_position_weights,
            review_contexts,
        )


def resolve_batch_graphs(graph_manager, split_name: str, local_idxs: list[int]):
    """主进程按 local_idx 取图并 batch，避免 worker 进程 pickle 图对象。"""
    graphs = [graph_manager.get_graph(split_name, int(idx)) for idx in local_idxs]
    batched_graph = batch_graphs(graphs)
    graph_tensors = {
        "node_token_ids": torch.tensor(batched_graph["node_token_ids"], dtype=torch.long),
        "node_counts": torch.tensor(batched_graph["node_counts"], dtype=torch.float32),
        "node_doc_freq": torch.tensor(batched_graph["node_doc_freq"], dtype=torch.float32),
        "node_in_degree": torch.tensor(batched_graph["node_in_degree"], dtype=torch.float32),
        "node_out_degree": torch.tensor(batched_graph["node_out_degree"], dtype=torch.float32),
        "edge_index": torch.tensor(batched_graph["edge_index"], dtype=torch.long),
        "batch_index": torch.tensor(batched_graph["batch_index"], dtype=torch.long),
        "num_nodes_per_graph": torch.tensor(
            batched_graph["num_nodes_per_graph"], dtype=torch.long
        ),
    }
    return graphs, graph_tensors


def compute_profile_lengths(
    dataset,
    profile_records,
    tokenizer,
    max_profile_tokens,
    collater: GraphCollater | None = None,
):
    """Return per-sample profile token lengths for length-bucket sampling."""
    if collater is not None and collater._sample_cache:
        return [
            len(collater._sample_cache[idx]["profile_ids"])
            for idx in range(len(dataset))
        ]
    lengths = []
    for idx in range(len(dataset)):
        row = dataset[idx]
        raw_user = str(row["raw_user"]) if "raw_user" in row else str(row["user"])
        text = profile_text_from_record(profile_records.get(raw_user))
        ids = tokenize_profile_text(tokenizer, text, max_profile_tokens)
        lengths.append(len(ids))
    return lengths


__all__ = [
    "GraphDataset",
    "GraphCollater",
    "GraphCacheManager",
    "MyDataset",
    "assert_profile_coverage",
    "compute_profile_lengths",
    "resolve_batch_graphs",
    "dataset_split",
    "load_profile_cache",
    "read_split_indices",
    "tokenizer_special_ids",
    "tokenizer_pad_id",
    "tokenizer_eos_id",
]
