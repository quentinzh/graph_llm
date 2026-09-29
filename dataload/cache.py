"""Graph cache management for per-sample user token graphs."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from graph_llm.models.token_graph import (
    ReviewRecord,
    UserTokenGraph,
)
from graph_llm.dataload.graph_build_fast import HistoryTokenCache, build_split_graphs_fast
from graph_llm.dataload.tail_stats import TailTokenStats


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
        history_token_cache: HistoryTokenCache | None = None,
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

        user_histories, allowed_history_keys, graphs = build_split_graphs_fast(
            split_dataset=split_dataset,
            split_name=split_name,
            fold=fold,
            history_dataset=history_dataset,
            tokenizer=tokenizer,
            skip_token_ids=skip_token_ids,
            max_nodes=max_nodes,
            min_token_count=min_token_count,
            tail_stats=tail_stats,
            item_meta=item_meta,
            tail_node_quota=tail_node_quota,
            relevance_node_quota=relevance_node_quota,
            preference_node_quota=preference_node_quota,
            history_token_cache=history_token_cache,
        )

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
