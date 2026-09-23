"""PLEASER 序列数据集加载：交互、评论、leave-one-out 划分与 D_calib/D_rec。"""

from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class InteractionRecord:
    """单条用户-商品交互。"""

    raw_user: str
    raw_item: str
    user_index: int
    item_index: int
    rating_raw: float | None
    timestamp: int
    review_text: str
    summary: str
    feature: str
    interaction_id: int


@dataclass
class SequentialDatasetBundle:
    """序列推荐数据集的全部结构化视图。"""

    name: str
    data_dir: Path
    interactions: list[InteractionRecord]
    item_meta: dict[str, dict[str, Any]]
    index2item: dict[str, str]
    item2index: dict[str, int]
    user_train_items: dict[int, list[int]]
    user_val_item: dict[int, int]
    user_test_item: dict[int, int]
    calib_ids: set[int]
    rec_train_ids: set[int]
    val_sample_ids: list[int]
    test_sample_ids: list[int]
    interaction_by_id: dict[int, InteractionRecord] = field(default_factory=dict)
    user_interactions: dict[int, list[InteractionRecord]] = field(default_factory=dict)

    @property
    def num_items(self) -> int:
        return len(self.item2index)

    @property
    def num_users(self) -> int:
        users = {r.user_index for r in self.interactions}
        return len(users)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_rating(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_sequential_dataset(data_dir: Path, dataset_name: str) -> SequentialDatasetBundle:
    root = data_dir / dataset_name
    split_payload = json.loads((root / "split.json").read_text(encoding="utf-8"))
    index2item = {str(k): str(v) for k, v in split_payload["index2item"].items()}
    item2index = {str(k): int(v) for k, v in split_payload["item2index"].items()}

    item_meta: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(root / "items.jsonl"):
        asin = str(row.get("asin") or row.get("item"))
        item_meta[asin] = row

    interactions: list[InteractionRecord] = []
    interaction_id = 0
    for row in _read_jsonl(root / "reviews.jsonl"):
        raw_user = str(row["user"])
        raw_item = str(row["item"])
        if raw_item not in item2index:
            continue
        user_index = int(row.get("user_index", -1))
        item_index = int(item2index[raw_item])
        interactions.append(
            InteractionRecord(
                raw_user=raw_user,
                raw_item=raw_item,
                user_index=user_index,
                item_index=item_index,
                rating_raw=_parse_rating(row.get("rating")),
                timestamp=int(row["timestamp"]),
                review_text=str(row.get("review_text") or ""),
                summary=str(row.get("summary") or ""),
                feature=str(row.get("feature") or ""),
                interaction_id=interaction_id,
            )
        )
        interaction_id += 1

    # 若 reviews 未带 user_index，从 sequences 补全
    if any(r.user_index < 0 for r in interactions):
        user_map: dict[str, int] = {}
        for row in _read_jsonl(root / "sequences.jsonl"):
            user_map[str(row["user"])] = int(row["user_index"])
        for rec in interactions:
            if rec.user_index < 0:
                rec.user_index = user_map.get(rec.raw_user, -1)
        interactions = [r for r in interactions if r.user_index >= 0]

    interactions.sort(key=lambda r: (r.user_index, r.timestamp, r.interaction_id))
    by_id = {r.interaction_id: r for r in interactions}

    split = split_payload["split"]
    user_train_items: dict[int, list[int]] = {}
    user_val_item: dict[int, int] = {}
    user_test_item: dict[int, int] = {}
    for u_str, item_ids in split["train"].items():
        user_train_items[int(u_str)] = [int(x) for x in item_ids]
    for u_str, item_ids in split["val"].items():
        user_val_item[int(u_str)] = int(item_ids[0])
    for u_str, item_ids in split["test"].items():
        user_test_item[int(u_str)] = int(item_ids[0])

    # 官方 train 交互：用户序列中属于 split.train 的 item_index 集合
    train_item_sets = {u: set(ids) for u, ids in user_train_items.items()}
    train_records: list[InteractionRecord] = []
    for rec in interactions:
        allowed = train_item_sets.get(rec.user_index)
        if allowed and rec.item_index in allowed:
            train_records.append(rec)

    val_sample_ids: list[int] = []
    test_sample_ids: list[int] = []
    for rec in interactions:
        if rec.user_index in user_val_item and rec.item_index == user_val_item[rec.user_index]:
            # 同一用户可能重复购买：取时间最晚且属于 val 的那条
            val_sample_ids.append(rec.interaction_id)
        if rec.user_index in user_test_item and rec.item_index == user_test_item[rec.user_index]:
            test_sample_ids.append(rec.interaction_id)

    # 每用户只保留一条 val/test（时间最晚）
    def _latest_per_user(ids: list[int]) -> list[int]:
        best: dict[int, int] = {}
        for iid in ids:
            rec = by_id[iid]
            prev = best.get(rec.user_index)
            if prev is None or by_id[prev].timestamp < rec.timestamp:
                best[rec.user_index] = iid
        return sorted(best.values())

    val_sample_ids = _latest_per_user(val_sample_ids)
    test_sample_ids = _latest_per_user(test_sample_ids)

    user_interactions: dict[int, list[InteractionRecord]] = defaultdict(list)
    for rec in interactions:
        user_interactions[rec.user_index].append(rec)
    for u in user_interactions:
        user_interactions[u].sort(key=lambda r: (r.timestamp, r.interaction_id))

    return SequentialDatasetBundle(
        name=dataset_name,
        data_dir=root,
        interactions=interactions,
        item_meta=item_meta,
        index2item=index2item,
        item2index=item2index,
        user_train_items=user_train_items,
        user_val_item=user_val_item,
        user_test_item=user_test_item,
        calib_ids=set(),
        rec_train_ids=set(),
        val_sample_ids=val_sample_ids,
        test_sample_ids=test_sample_ids,
        interaction_by_id=by_id,
        user_interactions=dict(user_interactions),
    )


def assign_calib_split(bundle: SequentialDatasetBundle, calib_ratio: float) -> None:
    """按时间重算 D_calib / D_rec（仅官方 train 交互）。"""
    train_item_sets = {u: set(ids) for u, ids in bundle.user_train_items.items()}
    train_records: list[InteractionRecord] = []
    for recs in bundle.user_interactions.values():
        for rec in recs:
            allowed = train_item_sets.get(rec.user_index)
            if allowed and rec.item_index in allowed:
                train_records.append(rec)
    train_records.sort(key=lambda r: r.timestamp)
    if not train_records:
        bundle.calib_ids = set()
        bundle.rec_train_ids = set()
        return
    n_calib = max(1, int(len(train_records) * calib_ratio))
    if n_calib >= len(train_records):
        n_calib = max(1, len(train_records) // 5)
    bundle.calib_ids = {r.interaction_id for r in train_records[:n_calib]}
    bundle.rec_train_ids = {r.interaction_id for r in train_records[n_calib:]}


def history_before(
    bundle: SequentialDatasetBundle,
    user_index: int,
    cutoff_timestamp: int,
    *,
    exclude_interaction_id: int | None = None,
    include_val: bool = False,
) -> list[InteractionRecord]:
    """返回用户在 cutoff 之前允许的交互历史（不含目标自身）。"""
    train_items = set(bundle.user_train_items.get(user_index, []))
    val_item = bundle.user_val_item.get(user_index)
    seq = bundle.user_interactions.get(user_index, [])
    if not seq:
        return []
    timestamps = [r.timestamp for r in seq]
    end = bisect.bisect_left(timestamps, cutoff_timestamp)
    out: list[InteractionRecord] = []
    for rec in seq[:end]:
        if rec.interaction_id == exclude_interaction_id:
            continue
        if rec.item_index in train_items:
            out.append(rec)
            continue
        if include_val and val_item is not None and rec.item_index == val_item:
            out.append(rec)
    return out


def history_item_indices(history: list[InteractionRecord]) -> set[int]:
    return {rec.item_index for rec in history}


def item_catalog_text(meta: dict[str, Any]) -> str:
    title = str(meta.get("title") or "").strip()
    desc = str(meta.get("description_str") or "").strip()
    if not desc and meta.get("description"):
        parts = meta.get("description")
        if isinstance(parts, list):
            desc = " ".join(str(x) for x in parts)
        else:
            desc = str(parts)
    if desc:
        return f"{title}\n{desc}".strip()
    return title or "Unknown"


def bundle_to_legacy_frame(bundle: SequentialDatasetBundle) -> pd.DataFrame:
    """将交互表转为 DataFrame，供推荐训练索引。"""
    rows = []
    for rec in bundle.interactions:
        rows.append(
            {
                "user": rec.user_index,
                "item": rec.item_index,
                "raw_user": rec.raw_user,
                "raw_item": rec.raw_item,
                "rating_raw": rec.rating_raw,
                "timestamp": rec.timestamp,
                "review_text": rec.review_text,
                "summary": rec.summary,
                "feature": rec.feature,
                "keyword_words": rec.feature,
                "interaction_id": rec.interaction_id,
            }
        )
    return pd.DataFrame(rows)
