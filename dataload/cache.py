"""Graph cache management for per-sample user token graphs."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from graph_llm.models.token_graph import (
    ReviewRecord,
    UserTokenGraph,
    attach_tokenizer_decode,
    build_sample_token_graph,
    extract_explanation_tokens,
)
from graph_llm.dataload.tail_stats import TailTokenStats, is_content_token
from graph_llm.aux.prompt_utils import item_meta_from_row


def _cache_version(
    max_nodes: int,
    min_token_count: int,
    *,
    tail_stats_fingerprint: str | None = None,
    tail_node_quota: int = 0,
    relevance_node_quota: int = 0,
    preference_node_quota: int = 0,
) -> str:
    payload = (
        "v2"
        f"|max_nodes={max_nodes}|min_token_count={min_token_count}"
        f"|tail_stats={tail_stats_fingerprint or 'legacy'}"
        f"|tail_quota={tail_node_quota}"
        f"|relevance_quota={relevance_node_quota}"
        f"|preference_quota={preference_node_quota}"
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


class GraphCacheManager:
    """Build or load leakage-safe token graphs for dataset samples."""

    def __init__(
        self,
        *,
        dataset_name: str,
        fold: int,
        cache_root: Path,
        user_histories: dict[str, list[ReviewRecord]],
        allowed_history_keys: dict[str, set[int]],
        graphs: dict[tuple[str, int], UserTokenGraph],
        meta: dict[str, Any],
    ):
        self.dataset_name = dataset_name
        self.fold = fold
        self.cache_root = cache_root
        self.user_histories = user_histories
        self.allowed_history_keys = allowed_history_keys
        self.graphs = graphs
        self.meta = meta

    @classmethod
    def cache_path(
        cls,
        cache_root: Path,
        dataset_name: str,
        fold: int,
        split_name: str,
        max_nodes: int,
        min_token_count: int,
        *,
        tail_stats_fingerprint: str | None = None,
        tail_node_quota: int = 0,
        relevance_node_quota: int = 0,
        preference_node_quota: int = 0,
    ) -> Path:
        version = _cache_version(
            max_nodes,
            min_token_count,
            tail_stats_fingerprint=tail_stats_fingerprint,
            tail_node_quota=tail_node_quota,
            relevance_node_quota=relevance_node_quota,
            preference_node_quota=preference_node_quota,
        )
        safe_name = dataset_name.replace("/", "__")
        return cache_root / safe_name / f"fold_{fold}" / split_name / f"graphs_{version}.pkl"

    @classmethod
    def build_or_load(
        cls,
        *,
        full_dataset: pd.DataFrame,
        split_dataset: pd.DataFrame,
        split_name: str,
        history_dataset: pd.DataFrame,
        dataset_name: str,
        fold: int,
        tokenizer,
        skip_token_ids: set[int],
        cache_root: Path,
        max_nodes: int = 512,
        min_token_count: int = 1,
        rebuild: bool = False,
        tail_stats: TailTokenStats | None = None,
        item_meta: dict | None = None,
        tail_node_quota: int = 256,
        relevance_node_quota: int = 128,
        preference_node_quota: int = 128,
    ) -> GraphCacheManager:
        tail_fingerprint = tail_stats.fingerprint if tail_stats is not None else None
        cache_path = cls.cache_path(
            cache_root,
            dataset_name,
            fold,
            split_name,
            max_nodes,
            min_token_count,
            tail_stats_fingerprint=tail_fingerprint,
            tail_node_quota=tail_node_quota,
            relevance_node_quota=relevance_node_quota,
            preference_node_quota=preference_node_quota,
        )
        if cache_path.exists() and not rebuild:
            print(f"Loading graph cache: {cache_path}")
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            print(
                f"Loaded graph cache split={split_name} fold={fold} "
                f"graphs={payload['meta'].get('num_graphs', len(payload['graphs']))}"
            )
            return cls(
                dataset_name=dataset_name,
                fold=fold,
                cache_root=cache_root,
                user_histories=payload["user_histories"],
                allowed_history_keys=payload["allowed_history_keys"],
                graphs=payload["graphs"],
                meta=payload["meta"],
            )

        attach_tokenizer_decode(tokenizer)
        content_filter = lambda token_id: is_content_token(tokenizer, token_id, skip_token_ids)

        user_histories: dict[str, list[ReviewRecord]] = defaultdict(list)
        for row_key, row in history_dataset.iterrows():
            raw_user = str(row["raw_user"])
            raw_item = str(row["raw_item"])
            explanation = row["review_text"] if "review_text" in row else row["template"][2]
            tokens = tuple(extract_explanation_tokens(tokenizer, explanation, skip_token_ids))
            user_histories[raw_user].append(
                ReviewRecord(int(row_key), raw_user, raw_item, tokens),
            )

        allowed_history_keys: dict[str, set[int]] = defaultdict(set)
        for row_key, row in history_dataset.iterrows():
            allowed_history_keys[str(row["raw_user"])].add(int(row_key))

        graphs: dict[tuple[str, int], UserTokenGraph] = {}
        for local_idx, (_, row) in enumerate(tqdm(
            split_dataset.iterrows(),
            total=len(split_dataset),
            desc=f"build graphs fold={fold} split={split_name}",
        )):
            raw_user = str(row["raw_user"])
            raw_item = str(row["raw_item"])
            row_key = int(row.name)
            history = [
                rec for rec in user_histories.get(raw_user, [])
                if rec.row_key in allowed_history_keys[raw_user]
            ]
            _title, _description, item_text = item_meta_from_row(raw_item, item_meta)
            target_item_token_ids = {
                int(token_id)
                for token_id in tokenizer(item_text, add_special_tokens=False)["input_ids"]
                if int(token_id) not in skip_token_ids
            }
            graph = build_sample_token_graph(
                history,
                exclude_row_key=row_key,
                target_raw_item=raw_item,
                skip_token_ids=skip_token_ids,
                max_nodes=max_nodes,
                min_token_count=min_token_count,
                tail_stats=tail_stats,
                target_item_token_ids=target_item_token_ids,
                content_token_filter=content_filter if tail_stats is not None else None,
                tail_node_quota=tail_node_quota,
                relevance_node_quota=relevance_node_quota,
                preference_node_quota=preference_node_quota,
            )
            graphs[(split_name, local_idx)] = graph

        meta = {
            "dataset_name": dataset_name,
            "fold": fold,
            "split_name": split_name,
            "max_nodes": max_nodes,
            "min_token_count": min_token_count,
            "tail_stats_fingerprint": tail_fingerprint,
            "tail_node_quota": tail_node_quota,
            "relevance_node_quota": relevance_node_quota,
            "preference_node_quota": preference_node_quota,
            "num_graphs": len(graphs),
            "version": _cache_version(
                max_nodes,
                min_token_count,
                tail_stats_fingerprint=tail_fingerprint,
                tail_node_quota=tail_node_quota,
                relevance_node_quota=relevance_node_quota,
                preference_node_quota=preference_node_quota,
            ),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(
                {
                    "user_histories": dict(user_histories),
                    "allowed_history_keys": {k: set(v) for k, v in allowed_history_keys.items()},
                    "graphs": graphs,
                    "meta": meta,
                },
                f,
            )
        meta_path = cache_path.with_suffix(".json")
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return cls(
            dataset_name=dataset_name,
            fold=fold,
            cache_root=cache_root,
            user_histories=dict(user_histories),
            allowed_history_keys={k: set(v) for k, v in allowed_history_keys.items()},
            graphs=graphs,
            meta=meta,
        )

    def get_graph(self, split_name: str, local_idx: int) -> UserTokenGraph:
        return self.graphs.get((split_name, local_idx), UserTokenGraph.empty())
