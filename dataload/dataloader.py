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


class GraphDataset(Dataset):
    """Dataset wrapper with split metadata."""

    def __init__(self, dataframe, split_name: str):
        self.df = dataframe.reset_index(drop=False).rename(columns={"index": "row_key"})
        self.split_name = split_name
        self.features = dataframe["keyword_words"].tolist() if "keyword_words" in dataframe else []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = row.to_dict()
        item["local_idx"] = idx
        item["split_name"] = self.split_name
        return item


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
        if token_id < 0 or token_id in self.feature_ignored_token_ids:
            return False
        if self.tokenizer is None:
            return True
        surface = self.tokenizer.decode([token_id], skip_special_tokens=True).strip().lower()
        normalized = surface.strip(" \t\r\n.,!?;:'\"()[]{}")
        if not normalized or normalized in FEATURE_STOPWORDS:
            return False
        if normalized.isdigit() or len(normalized) <= 2:
            return False
        return any(ch.isalpha() for ch in normalized)

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
        weights = [0.0] * len(ids)
        for variant in self._keyword_token_variants(keyword):
            width = len(variant)
            if width == 0 or width > len(ids):
                continue
            for start in range(0, len(ids) - width + 1):
                if ids[start:start + width] != variant:
                    continue
                for offset, token_id in enumerate(variant):
                    if self._keep_feature_token(token_id):
                        weights[start + offset] = 1.0
        return weights

    def __call__(self, data):
        input_ids, rating = [], []
        profile_ids, target_item_ids = [], []
        feature_weight_rows = []
        graphs = []
        item_texts = []
        item_titles = []
        raw_users = []
        max_length = max([
            min(self.word, max(len(x["text"]), 1))
            for x in data
        ])

        for x in data:
            ids = list(x["text"][:max_length])
            if len(ids) == 0:
                ids = [self.eos_token_id]

            feature_weights = self._feature_position_weights(ids, x.get("keyword_words", ""))
            pad_len = max_length - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            feature_weight_rows.append(feature_weights + [0.0] * pad_len)
            profile_ids.append(self._profile_ids(x))
            target_item_ids.append(self._target_item_ids(x))
            rating.append(x["rating"])
            graphs.append(self._graph_for_row(x))
            raw_item = str(x["raw_item"]) if "raw_item" in x else str(x["item"])
            title, _description, item_text = item_meta_from_row(raw_item, self.item_meta)
            item_titles.append(title)
            item_texts.append(item_text)
            raw_users.append(str(x["raw_user"]) if "raw_user" in x else str(x["user"]))

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

        batched_graph = batch_graphs(graphs)
        graph_tensors = {
            "node_token_ids": torch.tensor(batched_graph["node_token_ids"], dtype=torch.long),
            "node_counts": torch.tensor(batched_graph["node_counts"], dtype=torch.float32),
            "node_doc_freq": torch.tensor(batched_graph["node_doc_freq"], dtype=torch.float32),
            "node_in_degree": torch.tensor(batched_graph["node_in_degree"], dtype=torch.float32),
            "node_out_degree": torch.tensor(batched_graph["node_out_degree"], dtype=torch.float32),
            "edge_index": torch.tensor(batched_graph["edge_index"], dtype=torch.long),
            "batch_index": torch.tensor(batched_graph["batch_index"], dtype=torch.long),
            "num_nodes_per_graph": torch.tensor(batched_graph["num_nodes_per_graph"], dtype=torch.long),
        }

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(rating, dtype=torch.long),
            profile_tensor,
            profile_mask,
            target_item_tensor,
            target_item_mask,
            graph_tensors,
            graphs,
            item_texts,
            item_titles,
            raw_users,
            feature_position_mask,
            feature_position_weights,
        )


def compute_profile_lengths(dataset, profile_records, tokenizer, max_profile_tokens):
    """Return per-sample profile token lengths for length-bucket sampling."""
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
    "dataset_split",
    "load_profile_cache",
    "read_split_indices",
    "tokenizer_special_ids",
    "tokenizer_pad_id",
    "tokenizer_eos_id",
]
