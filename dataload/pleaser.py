"""Load PLEASER category datasets from sequences/reviews/items JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# 仓库 data/ 下固定的五个 PLEASER 数据集目录名
PLEASER_DATASET_NAMES: tuple[str, ...] = (
    "Software",
    "Instruments",
    "Arts",
    "Office",
    "Tools",
    "Instruments_small",
    "Arts_small",
    "Office_small",
    "Tools_small",
)


def canonical_pleaser_name(name: str) -> str | None:
    """大小写不敏感地将用户输入解析为合法 PLEASER 数据集名。"""
    clean = str(name).strip().strip("/")
    if not clean:
        return None
    for canonical in PLEASER_DATASET_NAMES:
        if canonical.lower() == clean.lower():
            return canonical
    return None


def pleaser_dataset_dir(data_dir: Path, name: str) -> Path:
    """返回 data_dir 下对应 PLEASER 数据集目录；不存在时抛出明确错误。"""
    canonical = canonical_pleaser_name(name)
    if canonical is None:
        raise ValueError(
            f"Unknown PLEASER dataset {name!r}. "
            f"Valid names: {', '.join(PLEASER_DATASET_NAMES)}"
        )
    root = Path(data_dir)
    dataset_dir = root / canonical
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"PLEASER dataset directory not found: {dataset_dir}. "
            f"Expected one of: {', '.join(PLEASER_DATASET_NAMES)} under {root}"
        )
    for filename in ("sequences.jsonl", "reviews.jsonl", "items.jsonl"):
        path = dataset_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {filename} for dataset {canonical!r} at {path}"
            )
    return dataset_dir


def _load_review_map(reviews_path: Path) -> dict[tuple[str, str, int], dict]:
    """按 (user, item, timestamp) 索引 reviews.jsonl 行。"""
    review_map: dict[tuple[str, str, int], dict] = {}
    with reviews_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (str(row["user"]), str(row["item"]), int(row["timestamp"]))
            review_map[key] = row
    return review_map


def _interaction_split(position: int, num_interactions: int) -> str:
    """与 PLEASER 留一划分一致：倒数第二条 validation，最后一条 test，其余 train。"""
    if position < num_interactions - 2:
        return "train"
    if position == num_interactions - 2:
        return "validation"
    return "test"


def load_pleaser_frame(data_dir: Path, name: str) -> pd.DataFrame:
    """从 sequences + reviews 构造摊平交互表。

    列：user, item, rating, review_text, feature, split, timestamp。
    行顺序与 sequences.jsonl 文件顺序及每条序列内交互顺序一致。
    """
    dataset_dir = pleaser_dataset_dir(data_dir, name)
    review_map = _load_review_map(dataset_dir / "reviews.jsonl")

    rows: list[dict] = []
    with (dataset_dir / "sequences.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            seq = json.loads(line)
            user = str(seq["user"])
            interactions = seq["interactions"]
            n = len(interactions)
            for pos, inter in enumerate(interactions):
                item = str(inter["item"])
                timestamp = int(inter["timestamp"])
                key = (user, item, timestamp)
                rev = review_map.get(key, {})
                review_text = str(rev.get("review_text") or rev.get("summary") or "")
                feature = str(rev.get("feature") or "")
                rows.append(
                    {
                        "user": user,
                        "item": item,
                        "rating": float(inter["rating"]),
                        "review_text": review_text,
                        "feature": feature,
                        "split": _interaction_split(pos, n),
                        "timestamp": timestamp,
                    }
                )

    return pd.DataFrame(rows)


def load_pleaser_item_meta(data_dir: Path, name: str) -> dict[str, dict]:
    """从 items.jsonl 读取物品元数据，键为 asin。"""
    dataset_dir = pleaser_dataset_dir(data_dir, name)
    meta: dict[str, dict] = {}
    with (dataset_dir / "items.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            asin = str(row.get("asin") or row.get("item") or "")
            if not asin:
                continue
            title = row.get("title")
            description = row.get("description_str")
            if description is None:
                description = ""
            # 官方解释 encoder 只用 description 列表的第一段，不用拼起来的全文
            raw_desc = row.get("description")
            description_first = ""
            if isinstance(raw_desc, list):
                for part in raw_desc:
                    text = str(part or "").strip()
                    if text:
                        description_first = text
                        break
            elif raw_desc:
                description_first = str(raw_desc).strip()
            if not description_first:
                description_first = str(description)
            meta[asin] = {
                "title": "" if title is None else str(title),
                "description": str(description),
                "description_first": description_first,
            }
    return meta
