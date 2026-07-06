#!/usr/bin/env python
"""从父数据集的 profile 缓存快速派生子数据集 profile。

子数据集（如 MoviesAndTV_corsa_filtered_small_15pct）通常是大数据集的用户/交互子集。
若父数据集已用 LLM 或启发式方法生成过 profile，本脚本只需按子数据集各 split
中的用户 ID 过滤复制，无需重新跑 Qwen3-4B。

语义说明：
  - 继承的是 profile_text / llama_profile 等画像文本，不按子数据集 15% 交互重算。
  - num_interactions / num_sampled 等计数保持源 profile 原值（表示生成时的全量历史）。
"""

from __future__ import annotations

import argparse
import copy
import pickle
import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_llm.config.datasets import resolve_dataset_paths
from graph_llm.dataload.legacy_data import assert_profile_coverage, read_split_indices
from graph_llm.train.trainer import profile_dataset_name_candidates


def scope_row_indices(split_indices: dict, scope: str) -> list[int]:
    """将 scope 名称映射为需要纳入的评论行索引。"""
    if scope == "train":
        return list(split_indices["train"])
    if scope == "train_valid":
        return list(split_indices["train"]) + list(split_indices["validation"])
    raise ValueError(f"Unsupported scope: {scope}")


def scope_users(reviews: pd.DataFrame, split_indices: dict, scope: str) -> set[str]:
    """从子数据集 reviews 与 split 索引中提取某 scope 内的用户集合。"""
    row_indices = scope_row_indices(split_indices, scope)
    scoped = reviews.iloc[row_indices]
    return set(scoped["raw_user"].astype(str).tolist())


def resolve_source_dataset_name(
    *,
    profile_dir: Path,
    target_dataset_name: str,
    fold: str,
    scopes: list[str],
    source_dataset_name: str | None,
) -> str:
    """解析父数据集名称：显式指定或从候选链中自动选取首个存在 pkl 的父集。"""
    if source_dataset_name:
        return source_dataset_name.strip().strip("/")

    candidates = profile_dataset_name_candidates(target_dataset_name)
    if len(candidates) <= 1:
        raise FileNotFoundError(
            f"No parent dataset candidate for {target_dataset_name!r}. "
            "Pass --source_dataset_name explicitly."
        )

    for candidate in candidates[1:]:
        for scope in scopes:
            source_path = profile_dir / candidate / f"fold_{fold}_{scope}.pkl"
            if source_path.is_file():
                print(f"Auto-selected source dataset: {candidate!r} (found {source_path})")
                return candidate

    tried = [
        str(profile_dir / candidate / f"fold_{fold}_{scope}.pkl")
        for candidate in candidates[1:]
        for scope in scopes
    ]
    raise FileNotFoundError(
        f"Could not auto-resolve parent profiles for {target_dataset_name!r}. Tried:\n  "
        + "\n  ".join(tried)
    )


def derive_profiles(
    *,
    data_dir: Path,
    dataset_name: str,
    profile_dir: Path,
    fold: str,
    scopes: list[str],
    source_dataset_name: str | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """按子数据集 split 用户过滤父 profile 并写出 pkl。"""
    target_name = dataset_name.strip("/")
    source_name = resolve_source_dataset_name(
        profile_dir=profile_dir,
        target_dataset_name=target_name,
        fold=fold,
        scopes=scopes,
        source_dataset_name=source_dataset_name,
    )

    reviews = pd.DataFrame(pd.read_pickle(data_dir / dataset_name / "reviews.pickle"))
    reviews["raw_user"] = reviews["user"].astype(str)
    split_indices = read_split_indices(data_dir, dataset_name, fold)

    out_dir = profile_dir / target_name
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for scope in scopes:
        out_path = out_dir / f"fold_{fold}_{scope}.pkl"
        if out_path.exists() and not overwrite:
            print(f"Skip existing {out_path} (pass --overwrite to replace)")
            written[scope] = out_path
            continue

        source_path = profile_dir / source_name / f"fold_{fold}_{scope}.pkl"
        if not source_path.is_file():
            raise FileNotFoundError(f"Source profile not found: {source_path}")

        with source_path.open("rb") as f:
            source_profiles = pickle.load(f)

        target_users = scope_users(reviews, split_indices, scope)
        derived: dict[str, dict] = {}
        missing_users: list[str] = []
        for user_id in sorted(target_users):
            record = source_profiles.get(user_id)
            if record is None:
                missing_users.append(user_id)
                continue
            new_record = copy.deepcopy(record)
            if isinstance(new_record, dict):
                new_record["scope"] = scope
                new_record["derived_from"] = str(source_path.resolve())
            derived[user_id] = new_record

        if missing_users:
            preview = ", ".join(missing_users[:10])
            raise ValueError(
                f"Source profile {source_path} misses {len(missing_users)} users "
                f"for scope={scope!r}. Examples: {preview}"
            )

        row_indices = scope_row_indices(split_indices, scope)
        scoped_df = reviews.iloc[row_indices].reset_index(drop=True)
        assert_profile_coverage(
            f"derived fold {fold} {scope}",
            [scoped_df],
            derived,
            allow_missing=False,
        )

        with out_path.open("wb") as f:
            pickle.dump(derived, f)

        print(
            f"Wrote scope={scope}: {len(derived)} users "
            f"(from {len(source_profiles)} source users) -> {out_path}"
        )
        written[scope] = out_path

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Derive child-dataset profiles by filtering parent profile caches"
    )
    parser.add_argument("--dataset_name", "--dataset", dest="dataset_name", required=True)
    parser.add_argument("--source_dataset_name", default=None, type=str)
    parser.add_argument("--data_dir", default=str(PACKAGE_ROOT / "data"), type=str)
    parser.add_argument(
        "--profile_dir",
        default=str(PACKAGE_ROOT / "data" / "profiles"),
        type=str,
    )
    parser.add_argument("--fold", default="1", type=str)
    parser.add_argument("--scopes", default="train,train_valid", type=str)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    class _ResolveArgs:
        dataset_name = args.dataset_name
        data_dir = args.data_dir

    resolve_args = _ResolveArgs()
    resolve_dataset_paths(resolve_args)

    scopes = [scope.strip() for scope in args.scopes.split(",") if scope.strip()]
    derive_profiles(
        data_dir=Path(resolve_args.data_dir),
        dataset_name=resolve_args.dataset_name,
        profile_dir=Path(args.profile_dir),
        fold=str(args.fold),
        scopes=scopes,
        source_dataset_name=args.source_dataset_name,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
