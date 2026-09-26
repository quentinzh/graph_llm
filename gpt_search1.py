#!/usr/bin/env python
"""在现有 graph_llm 方法内执行 grounding 约束下的序列质量搜索。

搜索阶段只访问 validation split。依次校准：

1. ``lambda_feat × tail_weight_max``；
2. ``lambda_prefix_feature``；
3. LoRA rank。

选出最终配置后，脚本才会对 test split 评估一次，避免沿用 P0/P1 的
test-driven 超参数选择方式。
"""

from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", os.environ.get("GRAPH_HF_ENDPOINT", "https://hf-mirror.com"))

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_llm.config import build_arg_parser
from graph_llm.train.trainer import (
    load_saved_validation_metrics,
    validation_selection_key,
)


STAGE1_NAME = "stage1_feature_tail"
STAGE2_NAME = "stage2_prefix_alignment"
STAGE3_NAME = "stage3_lora_capacity"

METRIC_COLUMNS = [
    "BLEU-1",
    "BLEU-2",
    "BLEU-4",
    "USR",
    "Distinct-1",
    "Distinct-2",
    "ENTR",
    "DIV",
    "FCR",
    "FMR",
    "rouge_1",
    "rouge_2",
    "rouge_4",
    "rouge_l",
    "BERTScore Precision",
    "BERTScore Recall",
    "BERTScore F1",
]


def _format_float(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _parse_float_grid(value: str) -> list[float]:
    values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise ValueError("float search grid must not be empty")
    return list(dict.fromkeys(values))


def _parse_int_grid(value: str) -> list[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("integer search grid must contain positive values")
    return list(dict.fromkeys(values))


def experiment_tag(
    lambda_feat: float,
    tail_weight_max: float,
    lambda_prefix_feature: float,
    lora_r: int,
    lora_alpha: int,
) -> str:
    return (
        f"lfeat{_format_float(lambda_feat)}_"
        f"tailmax{_format_float(tail_weight_max)}_"
        f"lpfx{_format_float(lambda_prefix_feature)}_"
        f"r{int(lora_r)}_a{int(lora_alpha)}"
    )


@dataclass
class TrialResult:
    """单次搜索试验及其 validation/test 指标。"""

    stage: str
    lambda_feat: float
    tail_weight_max: float
    lambda_prefix_feature: float
    lora_r: int
    lora_alpha: int
    ckpt_dir: str
    log_dir: str
    output_dir: str
    metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    tag: str = ""
    is_best: bool = False
    reused: bool = False

    def __post_init__(self) -> None:
        if not self.tag:
            self.tag = experiment_tag(
                self.lambda_feat,
                self.tail_weight_max,
                self.lambda_prefix_feature,
                self.lora_r,
                self.lora_alpha,
            )


def default_results_file(dataset_name: str) -> Path:
    safe_name = str(dataset_name).strip("/")
    return PACKAGE_ROOT / "log" / "gpt_search1" / safe_name / "gpt_search1_results.md"


def _primary_split_index(split_indices: str) -> str:
    parts = [item.strip() for item in str(split_indices).split(",") if item.strip()]
    if not parts:
        raise ValueError("split_indices must not be empty")
    return parts[0]


def _checkpoint_prefix(args, split_index: str) -> str:
    return str(Path(args.ckpt_dir) / str(args.dataset_name).strip("/") / split_index)


def _checkpoint_ready(args, split_index: str) -> bool:
    prefix = _checkpoint_prefix(args, split_index)
    return Path(prefix + "model").is_dir() and Path(prefix + "selector.bin").is_file()


def build_trial_args(
    base_args,
    *,
    stage: str,
    lambda_feat: float,
    tail_weight_max: float,
    lambda_prefix_feature: float,
    lora_r: int,
    lora_alpha: int,
):
    """为 trial 构造隔离目录，并固定本搜索的 grounding 配置。"""
    args = copy.copy(base_args)
    tag = experiment_tag(
        lambda_feat,
        tail_weight_max,
        lambda_prefix_feature,
        lora_r,
        lora_alpha,
    )
    args.lambda_feat = float(lambda_feat)
    args.tail_weight_max = float(tail_weight_max)
    args.lambda_prefix_feature = float(lambda_prefix_feature)
    args.lora_r = int(lora_r)
    args.lora_alpha = int(lora_alpha)

    args.ckpt_dir = str(Path(base_args.ckpt_dir) / "gpt_search1" / stage / tag)
    args.log_dir = str(Path(base_args.log_dir) / "gpt_search1" / stage / tag)
    args.output_dir = str(Path(base_args.output_dir) / "gpt_search1" / stage / tag)
    args.log_name = "graph_profile.log"
    args.only_eval = False
    args.skip_test_evaluation = True
    args.force = False
    return args


def trial_from_args(args, stage: str) -> TrialResult:
    return TrialResult(
        stage=stage,
        lambda_feat=float(args.lambda_feat),
        tail_weight_max=float(args.tail_weight_max),
        lambda_prefix_feature=float(args.lambda_prefix_feature),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        ckpt_dir=str(args.ckpt_dir),
        log_dir=str(args.log_dir),
        output_dir=str(args.output_dir),
    )


def extract_primary_fold_metrics(fold_metrics: dict[str, dict]) -> dict[str, float]:
    """从 trainer.run 的多 fold 返回值中读取第一个 fold。"""
    if not fold_metrics:
        return {}
    first_key = sorted(fold_metrics, key=lambda item: (len(item), item))[0]
    metrics = fold_metrics[first_key] or {}
    return {str(key): float(value) for key, value in metrics.items()}


def trial_selection_key(trial: TrialResult, fmr_threshold: float) -> tuple[float, ...]:
    return validation_selection_key(
        trial.metrics,
        mode="grounded_sequence",
        fmr_threshold=fmr_threshold,
    )


def pick_best_trial(trials: list[TrialResult], fmr_threshold: float) -> TrialResult:
    if not trials:
        raise ValueError("Cannot pick best trial from an empty list")
    return max(trials, key=lambda trial: trial_selection_key(trial, fmr_threshold))


def mark_stage_best(trials: list[TrialResult], fmr_threshold: float) -> TrialResult:
    for trial in trials:
        trial.is_best = False
    best = pick_best_trial(trials, fmr_threshold)
    best.is_best = True
    return best


def print_trial_summary(args, trial: TrialResult) -> None:
    print("=" * 96)
    print(f"[{trial.stage}] {trial.tag}")
    print(
        f"lambda_feat={trial.lambda_feat} tail_weight_max={trial.tail_weight_max} "
        f"lambda_prefix_feature={trial.lambda_prefix_feature} "
        f"lora={trial.lora_r}/{trial.lora_alpha}"
    )
    print(
        f"grounding: evidence_bonus={args.evidence_bonus} top_m={args.top_m_evidence} "
        f"lambda_selector={args.lambda_selector} "
        f"selector_feature_positive_weight={args.selector_feature_positive_weight}"
    )
    print(
        f"review: top_k={args.review_top_k_user}/{args.review_top_k_item} "
        f"prefix_len={args.user_review_prefix_len}/{args.item_review_prefix_len}"
    )
    print(f"ckpt_dir={trial.ckpt_dir}")
    print("=" * 96)


def run_trial(
    args,
    *,
    stage: str,
    dry_run: bool,
    resume: bool,
    split_index: str,
) -> TrialResult:
    trial = trial_from_args(args, stage)
    print_trial_summary(args, trial)
    if dry_run:
        return trial

    if resume and _checkpoint_ready(args, split_index):
        saved = load_saved_validation_metrics(_checkpoint_prefix(args, split_index))
        if saved:
            trial.metrics = saved
            trial.reused = True
            print(f"[{stage}] resume validation metrics: {saved}")
            return trial

    from graph_llm.train import run

    fold_metrics = run(args)
    trial.metrics = extract_primary_fold_metrics(fold_metrics)
    print(
        f"[{stage}] {trial.tag} validation "
        f"BLEU-1={trial.metrics.get('BLEU-1', float('nan')):.4f} "
        f"rouge_l={trial.metrics.get('rouge_l', float('nan')):.4f} "
        f"FMR={trial.metrics.get('FMR', float('nan')):.4f}"
    )
    return trial


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def render_results_markdown(
    *,
    dataset_name: str,
    split_indices: str,
    fmr_threshold: float,
    stage1_trials: list[TrialResult],
    stage2_trials: list[TrialResult],
    stage3_trials: list[TrialResult],
    final_trial: TrialResult | None,
) -> str:
    lines = [
        "# graph_llm GPT Search 1",
        "",
        f"- dataset: `{dataset_name}`",
        f"- split_indices: `{split_indices}`",
        f"- updated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- search data: `validation only`",
        (
            "- selection: maximize `min(BLEU-1, rouge_l)`, then their mean, under "
            f"validation `FMR >= {fmr_threshold:g}`"
        ),
        "- test policy: evaluate once after all hyperparameters are fixed",
        "",
    ]

    header = (
        "| stage | tag | lambda_feat | tail_weight_max | lambda_prefix_feature | "
        "lora_r | lora_alpha | "
        + " | ".join(METRIC_COLUMNS)
        + " | reused | best |"
    )
    separator = (
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | "
        + " | ".join(["---:"] * len(METRIC_COLUMNS))
        + " | --- | --- |"
    )
    sections = [
        ("Stage 1: feature-tail interaction", stage1_trials),
        ("Stage 2: prefix alignment", stage2_trials),
        ("Stage 3: LoRA capacity", stage3_trials),
    ]
    for title, trials in sections:
        lines.extend([f"## {title}", "", header, separator])
        for trial in trials:
            metric_cells = " | ".join(
                _format_metric(trial.metrics.get(name)) for name in METRIC_COLUMNS
            )
            lines.append(
                "| {stage} | {tag} | {lfeat} | {tailmax} | {lpfx} | {r} | {alpha} | "
                "{metrics} | {reused} | {best} |".format(
                    stage=trial.stage,
                    tag=trial.tag,
                    lfeat=_format_metric(trial.lambda_feat),
                    tailmax=_format_metric(trial.tail_weight_max),
                    lpfx=_format_metric(trial.lambda_prefix_feature),
                    r=trial.lora_r,
                    alpha=trial.lora_alpha,
                    metrics=metric_cells,
                    reused="yes" if trial.reused else "",
                    best="yes" if trial.is_best else "",
                )
            )
        lines.append("")

    lines.extend(["## Final fixed configuration", ""])
    if final_trial is None:
        lines.append("Not selected yet.")
    else:
        lines.extend(
            [
                f"- tag: `{final_trial.tag}`",
                f"- lambda_feat: `{final_trial.lambda_feat}`",
                f"- tail_weight_max: `{final_trial.tail_weight_max}`",
                f"- lambda_prefix_feature: `{final_trial.lambda_prefix_feature}`",
                f"- lora_r / lora_alpha: `{final_trial.lora_r} / {final_trial.lora_alpha}`",
                "",
                "### Final test metrics",
                "",
            ]
        )
        if final_trial.test_metrics:
            for name in METRIC_COLUMNS:
                lines.append(f"- {name}: `{_format_metric(final_trial.test_metrics.get(name))}`")
        else:
            lines.append("Test has not been evaluated.")
    return "\n".join(lines).rstrip() + "\n"


def write_results_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def refresh_results(
    path: Path,
    base_args,
    stage1_trials: list[TrialResult],
    stage2_trials: list[TrialResult],
    stage3_trials: list[TrialResult],
    final_trial: TrialResult | None,
) -> None:
    write_results_file(
        path,
        render_results_markdown(
            dataset_name=base_args.dataset_name,
            split_indices=base_args.split_indices,
            fmr_threshold=base_args.selection_fmr_threshold,
            stage1_trials=stage1_trials,
            stage2_trials=stage2_trials,
            stage3_trials=stage3_trials,
            final_trial=final_trial,
        ),
    )


def evaluate_final_test(base_args, winner: TrialResult) -> dict[str, float]:
    """复用获胜 checkpoint，并且只在这里访问一次 test split。"""
    args = copy.copy(base_args)
    args.lambda_feat = winner.lambda_feat
    args.tail_weight_max = winner.tail_weight_max
    args.lambda_prefix_feature = winner.lambda_prefix_feature
    args.lora_r = winner.lora_r
    args.lora_alpha = winner.lora_alpha
    args.ckpt_dir = winner.ckpt_dir
    args.log_dir = str(Path(base_args.log_dir) / "gpt_search1" / "final_test" / winner.tag)
    args.output_dir = str(Path(base_args.output_dir) / "gpt_search1" / "final_test" / winner.tag)
    args.log_name = "graph_profile.log"
    args.only_eval = True
    args.skip_test_evaluation = False
    args.force = False
    args.rebuild_graph_cache = False
    args.rebuild_dataset_cache = False

    from graph_llm.train import run

    return extract_primary_fold_metrics(run(args))


def main() -> None:
    parser = build_arg_parser()
    parser.set_defaults(
        # 由已有 P0/P1 结果负责 grounding，搜索重点转向序列实现质量。
        evidence_bonus=1.0,
        top_m_evidence=5,
        lambda_selector=0.2,
        selector_feature_positive_weight=2.0,
        lambda_prefix_feature=0.05,
        tail_weight_min=0.5,
        tail_weight_max=1.5,
        tail_alpha=0.5,
        review_top_k_user=4,
        review_top_k_item=8,
        user_review_prefix_len=4,
        item_review_prefix_len=4,
        max_generation_prompt_tokens=40,
        word=40,
        lora_r=16,
        lora_alpha=32,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj",
        checkpoint_selection="grounded_sequence",
        selection_fmr_threshold=0.17,
        skip_test_evaluation=True,
        devices="1",
        model_path=str(PACKAGE_ROOT / "pretrain_llm" / "qwen3-4b"),
        embedding_model_path=str(PACKAGE_ROOT / "pretrain_llm" / "qwen3-embedding-0.6b"),
        profile_dir=str(PACKAGE_ROOT / "data" / "profiles"),
        data_dir=str(PACKAGE_ROOT / "data"),
    )
    parser.add_argument(
        "--lambda_feat_grid",
        default="0.01,0.03,0.05",
        help="Stage1 lambda_feat 网格，逗号分隔",
    )
    parser.add_argument(
        "--tail_weight_max_grid",
        default="1.5,2.0",
        help="Stage1 tail_weight_max 网格，逗号分隔",
    )
    parser.add_argument(
        "--lambda_prefix_feature_grid",
        default="0,0.05,0.1",
        help="Stage2 prefix-feature 对齐权重网格，逗号分隔",
    )
    parser.add_argument(
        "--lora_rank_grid",
        default="16,32",
        help="Stage3 LoRA rank 网格；alpha 自动设为 2r",
    )
    parser.add_argument(
        "--results_file",
        default="",
        help="结果 Markdown；默认 graph_llm/log/gpt_search1/<dataset>/gpt_search1_results.md",
    )
    parser.add_argument("--dry_run", action="store_true", help="只打印配置，不训练")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="checkpoint 与 validation_metrics 均存在时复用已完成 trial",
    )
    parser.add_argument(
        "--skip_final_test",
        action="store_true",
        help="完成 validation 搜索后暂不运行最终 test",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="只跑一个配置，每个 train/validation/test 最多两个 batch",
    )
    base_args = parser.parse_args()

    if base_args.only_eval:
        raise ValueError("gpt_search1 controls only_eval internally; do not pass --only_eval")
    if base_args.force:
        raise ValueError(
            "gpt_search1 uses isolated fresh trial directories; --force would repeatedly "
            "delete shared caches and is therefore not supported"
        )
    if not 0.0 <= base_args.selection_fmr_threshold <= 1.0:
        raise ValueError("selection_fmr_threshold must be in [0, 1]")

    lambda_feat_grid = _parse_float_grid(base_args.lambda_feat_grid)
    tail_max_grid = _parse_float_grid(base_args.tail_weight_max_grid)
    prefix_grid = _parse_float_grid(base_args.lambda_prefix_feature_grid)
    lora_rank_grid = _parse_int_grid(base_args.lora_rank_grid)
    if base_args.smoke_test:
        # 遵循仓库约定：smoke 只检查两个 batch，并保留 --devices cpu/1 接口。
        base_args.max_train_batches = 2
        base_args.max_eval_batches = 2
        base_args.epochs = 1
        base_args.early_stop_patience = 1
        lambda_feat_grid = [0.03]
        tail_max_grid = [1.5]
        prefix_grid = [0.05]
        lora_rank_grid = [16]

    split_index = _primary_split_index(base_args.split_indices)
    results_path = (
        Path(base_args.results_file).expanduser()
        if base_args.results_file
        else default_results_file(base_args.dataset_name)
    )

    stage1_trials: list[TrialResult] = []
    stage2_trials: list[TrialResult] = []
    stage3_trials: list[TrialResult] = []
    refresh_results(
        results_path, base_args, stage1_trials, stage2_trials, stage3_trials, None
    )

    for lambda_feat, tail_max in product(lambda_feat_grid, tail_max_grid):
        args = build_trial_args(
            base_args,
            stage=STAGE1_NAME,
            lambda_feat=lambda_feat,
            tail_weight_max=tail_max,
            lambda_prefix_feature=base_args.lambda_prefix_feature,
            lora_r=base_args.lora_r,
            lora_alpha=base_args.lora_alpha,
        )
        stage1_trials.append(
            run_trial(
                args,
                stage=STAGE1_NAME,
                dry_run=base_args.dry_run,
                resume=base_args.resume,
                split_index=split_index,
            )
        )
        refresh_results(
            results_path, base_args, stage1_trials, stage2_trials, stage3_trials, None
        )
    best_stage1 = mark_stage_best(stage1_trials, base_args.selection_fmr_threshold)

    # Stage2 始终保留 Stage1 获胜配置作为对照，再训练不同的对齐权重。
    stage2_trials.append(
        replace(
            best_stage1,
            stage=STAGE2_NAME,
            is_best=False,
            reused=True,
        )
    )
    for prefix_weight in prefix_grid:
        if abs(prefix_weight - best_stage1.lambda_prefix_feature) < 1e-12:
            continue
        args = build_trial_args(
            base_args,
            stage=STAGE2_NAME,
            lambda_feat=best_stage1.lambda_feat,
            tail_weight_max=best_stage1.tail_weight_max,
            lambda_prefix_feature=prefix_weight,
            lora_r=best_stage1.lora_r,
            lora_alpha=best_stage1.lora_alpha,
        )
        stage2_trials.append(
            run_trial(
                args,
                stage=STAGE2_NAME,
                dry_run=base_args.dry_run,
                resume=base_args.resume,
                split_index=split_index,
            )
        )
        refresh_results(
            results_path, base_args, stage1_trials, stage2_trials, stage3_trials, None
        )
    best_stage2 = mark_stage_best(stage2_trials, base_args.selection_fmr_threshold)

    # Stage3 同样保留当前容量作为对照，只训练尚未出现的 rank。
    stage3_trials.append(
        replace(
            best_stage2,
            stage=STAGE3_NAME,
            is_best=False,
            reused=True,
        )
    )
    for rank in lora_rank_grid:
        alpha = int(2 * rank)
        if rank == best_stage2.lora_r and alpha == best_stage2.lora_alpha:
            continue
        args = build_trial_args(
            base_args,
            stage=STAGE3_NAME,
            lambda_feat=best_stage2.lambda_feat,
            tail_weight_max=best_stage2.tail_weight_max,
            lambda_prefix_feature=best_stage2.lambda_prefix_feature,
            lora_r=rank,
            lora_alpha=alpha,
        )
        stage3_trials.append(
            run_trial(
                args,
                stage=STAGE3_NAME,
                dry_run=base_args.dry_run,
                resume=base_args.resume,
                split_index=split_index,
            )
        )
        refresh_results(
            results_path, base_args, stage1_trials, stage2_trials, stage3_trials, None
        )
    winner = mark_stage_best(stage3_trials, base_args.selection_fmr_threshold)

    if not base_args.dry_run and not base_args.skip_final_test:
        winner.test_metrics = evaluate_final_test(base_args, winner)
        print(
            "Final test: "
            f"BLEU-1={winner.test_metrics.get('BLEU-1', float('nan')):.4f} "
            f"rouge_l={winner.test_metrics.get('rouge_l', float('nan')):.4f} "
            f"FMR={winner.test_metrics.get('FMR', float('nan')):.4f}"
        )

    refresh_results(
        results_path,
        base_args,
        stage1_trials,
        stage2_trials,
        stage3_trials,
        winner,
    )
    print(f"gpt_search1 complete. Results written to: {results_path}")


if __name__ == "__main__":
    main()
