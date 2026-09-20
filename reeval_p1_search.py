#!/usr/bin/env python
"""一键重评 P1 已有 checkpoint 的 test 指标（不重新训练）。

用途：修复 ``load_best_checkpoint`` 后，把原先错误的 test 指标用正确
LoRA 加载路径重跑一遍，并回写 ``p1_search_results.md``。

示例：
  # 全量重评（默认 GPU=cuda:1；Mac 上默认 CPU）
  conda run -n fair python graph_llm/reeval_p1_search.py \\
      --dataset_name Amazon/MoviesAndTV_corsa_filtered_small_15pct/

  # smoke：只跑 2 个 batch，并显式选设备
  conda run -n fair python graph_llm/reeval_p1_search.py \\
      --dataset_name Amazon/MoviesAndTV_corsa_filtered_small_15pct/ \\
      --smoke --device gpu

  # 只重评某几个 tag
  conda run -n fair python graph_llm/reeval_p1_search.py \\
      --dataset_name Amazon/MoviesAndTV_corsa_filtered_small_15pct/ \\
      --tags lsel0.3_sfpw3_lpfx0.1,lsel0.3_sfpw4_lpfx0.1
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", os.environ.get("GRAPH_HF_ENDPOINT", "https://hf-mirror.com"))

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_llm.config import build_arg_parser
from graph_llm.p1_search import (
    TrialResult,
    default_results_file,
    extract_primary_fold_metrics,
    mark_stage_best,
    refresh_results_file,
)

# 已知的 P1 三阶段目录名
STAGE_DIRS = {
    "stage1_lambda_selector",
    "stage2_selector_feature_positive_weight",
    "stage3_lambda_prefix_feature",
}

# 写入 args 时跳过这些键，由本脚本显式控制
_CONFIG_SKIP_KEYS = {
    "ckpt_dir",
    "log_dir",
    "output_dir",
    "only_eval",
    "force",
    "max_eval_batches",
    "max_train_batches",
    "devices",
    "active_oom_plan",
    "active_oom_plan_desc",
}


@dataclass
class DiscoveredTrial:
    """磁盘上已发现的一次 P1 trial。"""

    stage: str
    tag: str
    trial_ckpt_dir: Path
    dataset_dir: Path
    split_index: str
    graph_config: dict[str, Any]

    @property
    def adapter_dir(self) -> Path:
        return self.dataset_dir / f"{self.split_index}model"

    @property
    def selector_path(self) -> Path:
        return self.dataset_dir / f"{self.split_index}selector.bin"

    @property
    def review_prefix_path(self) -> Path:
        return self.dataset_dir / f"{self.split_index}review_prefix.bin"


def default_device_choice() -> str:
    """MacBook 默认 CPU；其他机器默认 GPU。"""
    return "cpu" if platform.system() == "Darwin" else "gpu"


def resolve_devices(device_choice: str, devices_cli: str) -> str:
    """把 --device/--devices 解析成 trainer 认识的 devices 字符串。"""
    if devices_cli.strip():
        return devices_cli.strip()
    choice = (device_choice or default_device_choice()).lower()
    if choice == "cpu":
        return "cpu"
    if choice == "gpu":
        # 服务器约定默认 cuda:1
        return "1"
    raise ValueError(f"Unsupported --device={device_choice!r}; use cpu|gpu")


def discover_trials(
    checkpoint_root: Path,
    dataset_name: str,
    split_index: str,
) -> list[DiscoveredTrial]:
    """扫描 ``checkpoints/p1_search/<stage>/<tag>/<dataset>/<split>model``。

    ``dataset_name`` 可能含多级路径（如 ``Amazon/MoviesAndTV_...``），
    因此 trial 根目录要用 ``model_dir.parents[len(parts)]`` 回退。
    """
    safe_dataset = dataset_name.strip("/")
    dataset_parts = Path(safe_dataset).parts
    if not dataset_parts:
        raise ValueError(f"invalid dataset_name: {dataset_name!r}")
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"checkpoint root not found: {checkpoint_root}")

    found: list[DiscoveredTrial] = []
    for model_dir in sorted(checkpoint_root.rglob(f"{split_index}model")):
        if not model_dir.is_dir():
            continue
        # .../<tag>/<dataset_parts...>/<split>model
        if len(model_dir.parents) <= len(dataset_parts):
            continue
        trial_ckpt_dir = model_dir.parents[len(dataset_parts)]
        expected_model = trial_ckpt_dir.joinpath(*dataset_parts) / f"{split_index}model"
        if model_dir.resolve() != expected_model.resolve():
            continue

        stage = trial_ckpt_dir.parent.name
        tag = trial_ckpt_dir.name
        if stage not in STAGE_DIRS:
            print(f"[skip] unrecognized stage dir: {model_dir}")
            continue

        dataset_dir = model_dir.parent
        config_path = dataset_dir / f"{split_index}graph_config.json"
        if not config_path.is_file():
            print(f"[skip] missing graph_config: {config_path}")
            continue
        if not (model_dir / "adapter_model.safetensors").is_file() and not (
            model_dir / "adapter_model.bin"
        ).is_file():
            print(f"[skip] missing adapter weights: {model_dir}")
            continue

        with open(config_path, encoding="utf-8") as handle:
            graph_config = json.load(handle)

        found.append(
            DiscoveredTrial(
                stage=stage,
                tag=tag,
                trial_ckpt_dir=trial_ckpt_dir,
                dataset_dir=dataset_dir,
                split_index=str(split_index),
                graph_config=graph_config,
            )
        )
    return found


def filter_trials(
    trials: list[DiscoveredTrial],
    *,
    stages: set[str] | None,
    tags: set[str] | None,
) -> list[DiscoveredTrial]:
    selected = []
    for trial in trials:
        if stages and trial.stage not in stages:
            continue
        if tags and trial.tag not in tags:
            continue
        selected.append(trial)
    return selected


def build_eval_args(
    base_args,
    trial: DiscoveredTrial,
    *,
    devices: str,
    max_eval_batches: int,
    reeval_log_root: Path,
    reeval_output_root: Path,
):
    """从 graph_config 恢复超参，并强制 only_eval + 指向原 checkpoint。"""
    args = copy.copy(base_args)
    for key, value in trial.graph_config.items():
        if key in _CONFIG_SKIP_KEYS:
            continue
        if hasattr(args, key):
            setattr(args, key, value)

    # 必须指向该 trial 原始 ckpt 根目录（其下才是 <dataset>/<split>model）
    args.ckpt_dir = str(trial.trial_ckpt_dir)
    args.log_dir = str(reeval_log_root / trial.stage / trial.tag)
    args.output_dir = str(reeval_output_root / trial.stage / trial.tag)
    args.log_name = "graph_profile_reeval.log"
    args.only_eval = True
    args.force = False
    args.devices = devices
    args.max_eval_batches = int(max_eval_batches)
    args.max_train_batches = 0
    args.split_indices = str(trial.split_index)
    # 保证 dataset_name 与目录一致
    if getattr(args, "dataset_name", None):
        args.dataset_name = str(args.dataset_name)
    return args


def trial_result_from_discovery(
    trial: DiscoveredTrial,
    metrics: dict[str, float],
) -> TrialResult:
    cfg = trial.graph_config
    return TrialResult(
        stage=trial.stage,
        lambda_feat=float(cfg.get("lambda_feat", 0.0)),
        evidence_bonus=float(cfg.get("evidence_bonus", 0.0)),
        top_m_evidence=int(cfg.get("top_m_evidence", 0)),
        lambda_selector=float(cfg.get("lambda_selector", 0.0)),
        selector_feature_positive_weight=float(
            cfg.get("selector_feature_positive_weight", 0.0)
        ),
        lambda_prefix_feature=float(cfg.get("lambda_prefix_feature", 0.0)),
        metrics=metrics,
        tag=trial.tag,
    )


def group_by_stage(
    results: list[TrialResult],
) -> tuple[list[TrialResult], list[TrialResult], list[TrialResult]]:
    stage1 = [item for item in results if item.stage == "stage1_lambda_selector"]
    stage2 = [
        item for item in results if item.stage == "stage2_selector_feature_positive_weight"
    ]
    stage3 = [item for item in results if item.stage == "stage3_lambda_prefix_feature"]
    return stage1, stage2, stage3


def choose_final_hyperparams(
    stage1: list[TrialResult],
    stage2: list[TrialResult],
    stage3: list[TrialResult],
) -> tuple[float, float, float]:
    """按阶段最优连锁选择最终 P1 超参；某阶段为空则回退上一阶段 best。"""
    if not stage1 and not stage2 and not stage3:
        raise ValueError("No reevaluated trials to select hyper-parameters from")

    if stage1:
        best1 = mark_stage_best(stage1)
        best_lsel = best1.lambda_selector
        best_sfpw = best1.selector_feature_positive_weight
        best_lpfx = best1.lambda_prefix_feature
    else:
        seed = (stage2 or stage3)[0]
        best_lsel = seed.lambda_selector
        best_sfpw = seed.selector_feature_positive_weight
        best_lpfx = seed.lambda_prefix_feature

    if stage2:
        # stage2 应在 stage1 最优 lsel 上比较；若混有其他 lsel，只取匹配子集
        pool2 = [item for item in stage2 if item.lambda_selector == best_lsel] or stage2
        best2 = mark_stage_best(pool2)
        for item in stage2:
            item.is_best = item.tag == best2.tag
        best_sfpw = best2.selector_feature_positive_weight
        best_lpfx = best2.lambda_prefix_feature

    if stage3:
        pool3 = [
            item
            for item in stage3
            if item.lambda_selector == best_lsel
            and item.selector_feature_positive_weight == best_sfpw
        ] or stage3
        best3 = mark_stage_best(pool3)
        for item in stage3:
            item.is_best = item.tag == best3.tag
        best_lpfx = best3.lambda_prefix_feature

    return best_lsel, best_sfpw, best_lpfx


def backup_results_file(path: Path) -> Path | None:
    """重写前备份旧汇总，避免误覆盖。"""
    if not path.is_file():
        return None
    backup = path.with_suffix(path.suffix + ".bak_before_reeval")
    shutil.copy2(path, backup)
    return backup


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Re-evaluate existing P1 checkpoints with fixed LoRA reload."
    )
    parser.add_argument(
        "--dataset_name",
        default="Amazon/MoviesAndTV_corsa_filtered_small_15pct/",
        type=str,
    )
    parser.add_argument("--split_indices", default="1", type=str)
    parser.add_argument(
        "--checkpoint_root",
        default=str(PACKAGE_ROOT / "checkpoints" / "p1_search"),
        type=str,
        help="P1 checkpoint 根目录",
    )
    parser.add_argument(
        "--results_file",
        default="",
        type=str,
        help="汇总 Markdown；默认 log/p1_search/<dataset>/p1_search_results.md",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default=default_device_choice(),
        help="cpu 或 gpu；Mac 默认 cpu，其他默认 gpu(=cuda:1)",
    )
    parser.add_argument(
        "--devices",
        default="",
        type=str,
        help="直接传给 trainer 的 devices（优先于 --device），例如 1 / 0 / cpu",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="smoke test：只跑 2 个 eval batch（不写正式汇总，除非同时给 --force_write_results）",
    )
    parser.add_argument(
        "--max_eval_batches",
        default=0,
        type=int,
        help=">0 时限制 test batch 数；--smoke 时默认 2",
    )
    parser.add_argument(
        "--stages",
        default="",
        type=str,
        help="逗号分隔阶段过滤，例如 stage1_lambda_selector,stage2_selector_feature_positive_weight",
    )
    parser.add_argument(
        "--tags",
        default="",
        type=str,
        help="逗号分隔 tag 过滤，例如 lsel0.3_sfpw4_lpfx0.1",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印将要重评的 trial，不加载模型",
    )
    parser.add_argument(
        "--force_write_results",
        action="store_true",
        help="即便 --smoke 也写回 results markdown（一般不要开）",
    )
    parser.add_argument(
        "--limit",
        default=0,
        type=int,
        help="最多重评前 N 个 trial（调试用）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    cli = parse_args(argv)
    dataset_name = cli.dataset_name
    split_index = str(cli.split_indices).split(",")[0].strip()
    if not split_index:
        raise ValueError("--split_indices is empty")

    devices = resolve_devices(cli.device, cli.devices)
    max_eval_batches = int(cli.max_eval_batches)
    if cli.smoke and max_eval_batches <= 0:
        max_eval_batches = 2

    checkpoint_root = Path(cli.checkpoint_root).expanduser().resolve()
    results_file = (
        Path(cli.results_file).expanduser()
        if cli.results_file
        else default_results_file(dataset_name)
    )

    stages = {item.strip() for item in cli.stages.split(",") if item.strip()} or None
    tags = {item.strip() for item in cli.tags.split(",") if item.strip()} or None

    trials = discover_trials(checkpoint_root, dataset_name, split_index)
    trials = filter_trials(trials, stages=stages, tags=tags)
    if cli.limit > 0:
        trials = trials[: cli.limit]

    print("=" * 88)
    print("P1 checkpoint re-evaluation")
    print(f"dataset_name     : {dataset_name}")
    print(f"split_index      : {split_index}")
    print(f"checkpoint_root  : {checkpoint_root}")
    print(f"results_file     : {results_file}")
    print(f"devices          : {devices}  (device_choice={cli.device})")
    print(f"max_eval_batches : {max_eval_batches or 'all'}")
    print(f"smoke            : {cli.smoke}")
    print(f"discovered       : {len(trials)} trial(s)")
    print("=" * 88)

    if not trials:
        raise SystemExit("No matching P1 checkpoints found.")

    for idx, trial in enumerate(trials, start=1):
        has_selector = trial.selector_path.is_file()
        has_prefix = trial.review_prefix_path.is_file()
        print(
            f"[{idx}/{len(trials)}] {trial.stage}/{trial.tag} "
            f"selector={'yes' if has_selector else 'NO'} "
            f"review_prefix={'yes' if has_prefix else 'NO'} "
            f"cfg_lsel={trial.graph_config.get('lambda_selector')} "
            f"sfpw={trial.graph_config.get('selector_feature_positive_weight')} "
            f"lpfx={trial.graph_config.get('lambda_prefix_feature')}"
        )

    if cli.dry_run:
        print("dry_run=True: exit before model load.")
        return

    # 用项目标准 parser 拿一份完整默认 args，再按 trial 覆盖
    base_parser = build_arg_parser()
    base_parser.set_defaults(
        dataset_name=dataset_name,
        split_indices=split_index,
        devices=devices,
        only_eval=True,
        model_path=str(PACKAGE_ROOT / "pretrain_llm" / "qwen3-4b"),
        embedding_model_path=str(PACKAGE_ROOT / "pretrain_llm" / "qwen3-embedding-0.6b"),
        profile_dir=str(PACKAGE_ROOT / "data" / "profiles"),
        data_dir=str(PACKAGE_ROOT / "data"),
    )
    base_args = base_parser.parse_args([])

    reeval_log_root = PACKAGE_ROOT / "log" / "p1_search_reeval"
    reeval_output_root = PACKAGE_ROOT / "log" / "output" / "p1_search_reeval"

    from graph_llm.train import run

    results: list[TrialResult] = []
    for idx, trial in enumerate(trials, start=1):
        print("\n" + "#" * 88)
        print(f"Re-eval [{idx}/{len(trials)}] {trial.stage}/{trial.tag}")
        print("#" * 88)
        if not trial.selector_path.is_file():
            raise FileNotFoundError(f"Missing selector checkpoint: {trial.selector_path}")
        if not trial.review_prefix_path.is_file():
            raise FileNotFoundError(
                f"Missing review_prefix checkpoint: {trial.review_prefix_path}"
            )

        args = build_eval_args(
            base_args,
            trial,
            devices=devices,
            max_eval_batches=max_eval_batches,
            reeval_log_root=reeval_log_root,
            reeval_output_root=reeval_output_root,
        )
        fold_metrics = run(args)
        metrics = extract_primary_fold_metrics(fold_metrics)
        result = trial_result_from_discovery(trial, metrics)
        results.append(result)
        print(
            f"[done] {trial.tag} "
            f"FMR={metrics.get('FMR', float('nan')):.4f} "
            f"BLEU-1={metrics.get('BLEU-1', float('nan')):.4f} "
            f"USR={metrics.get('USR', float('nan')):.4f} "
            f"rouge_l={metrics.get('rouge_l', float('nan')):.4f}"
        )

    # smoke 默认不覆盖正式汇总，避免用 2-batch 指标污染选参
    write_results = (not cli.smoke) or cli.force_write_results
    if not write_results:
        print(
            "\nsmoke 模式：已完成重评，但不回写 p1_search_results.md。"
            "去掉 --smoke 再跑一次以更新正式汇总。"
        )
        return

    stage1, stage2, stage3 = group_by_stage(results)
    best_lsel, best_sfpw, best_lpfx = choose_final_hyperparams(stage1, stage2, stage3)

    # 冻结 P0 参数：优先用任一 trial 的 config
    seed_cfg = results[0]
    frozen_lambda_feat = seed_cfg.lambda_feat
    frozen_evidence_bonus = seed_cfg.evidence_bonus
    frozen_top_m = seed_cfg.top_m_evidence

    backup = backup_results_file(results_file)
    if backup is not None:
        print(f"Backed up old results to: {backup}")

    refresh_results_file(
        results_file,
        dataset_name=dataset_name,
        split_indices=split_index,
        frozen_lambda_feat=frozen_lambda_feat,
        frozen_evidence_bonus=frozen_evidence_bonus,
        frozen_top_m_evidence=frozen_top_m,
        stage1_trials=stage1,
        stage2_trials=stage2,
        stage3_trials=stage3,
        best_lambda_selector=best_lsel,
        best_selector_feature_positive_weight=best_sfpw,
        best_lambda_prefix_feature=best_lpfx,
    )
    print(f"\nUpdated results: {results_file}")
    print(
        "Selected P1 hyper-params: "
        f"lambda_selector={best_lsel} "
        f"selector_feature_positive_weight={best_sfpw} "
        f"lambda_prefix_feature={best_lpfx}"
    )


if __name__ == "__main__":
    main()
