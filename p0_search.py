#!/usr/bin/env python
"""按 P0 优先级顺序搜索 lambda_feat -> evidence_bonus -> top_m_evidence。"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", os.environ.get("GRAPH_HF_ENDPOINT", "https://hf-mirror.com"))

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_llm.config import build_arg_parser

# 三阶段固定搜索网格
STAGE1_LAMBDA_FEAT = [1e-3, 1e-2, 1e-1]
STAGE2_EVIDENCE_BONUS = [0.0, 0.1, 0.5, 1.0, 2.0]
STAGE3_TOP_M_EVIDENCE = [5, 10, 15, 20]

# 阶段 1/2 冻结的默认值
DEFAULT_EVIDENCE_BONUS = 0.1
DEFAULT_TOP_M_EVIDENCE = 5

# 汇总表展示的 test 指标列（仅写 run() 实际返回的键）
METRIC_COLUMNS = [
    "BLEU-1",
    "BLEU-4",
    "Distinct-1",
    "Distinct-2",
    "ENTR",
    "DIV",
    "FCR",
    "FMR",
    "rouge_1",
    "rouge_2",
    "rouge_l",
]


def _format_float(value: float) -> str:
    """把浮点超参格式化成目录 tag 友好的字符串。"""
    if 0 < abs(value) <= 1e-2:
        text = f"{value:.0e}"
        mantissa, exponent = text.split("e", 1)
        sign = "-" if exponent.startswith("-") else ""
        digits = exponent.lstrip("+-").lstrip("0") or "0"
        return f"{mantissa}e{sign}{digits}"
    return f"{value:g}"


def experiment_tag(
    lambda_feat: float,
    evidence_bonus: float,
    top_m_evidence: int,
) -> str:
    return (
        f"lfeat{_format_float(lambda_feat)}_"
        f"eb{_format_float(evidence_bonus)}_"
        f"top{int(top_m_evidence)}"
    )


def default_results_file(dataset_name: str) -> Path:
    safe_name = dataset_name.strip("/")
    return PACKAGE_ROOT / "log" / "p0_search" / safe_name / "p0_search_results.md"


@dataclass
class TrialResult:
    """单次试验记录。"""

    stage: str
    lambda_feat: float
    evidence_bonus: float
    top_m_evidence: int
    metrics: dict[str, float] = field(default_factory=dict)
    tag: str = ""
    is_best: bool = False

    def __post_init__(self) -> None:
        if not self.tag:
            self.tag = experiment_tag(self.lambda_feat, self.evidence_bonus, self.top_m_evidence)


def build_experiment_args(
    base_args,
    *,
    stage: str,
    lambda_feat: float,
    evidence_bonus: float,
    top_m_evidence: int,
):
    """为单次试验构造隔离目录的 args。"""
    args = copy.copy(base_args)
    tag = experiment_tag(lambda_feat, evidence_bonus, top_m_evidence)

    args.lambda_feat = float(lambda_feat)
    args.evidence_bonus = float(evidence_bonus)
    args.top_m_evidence = int(top_m_evidence)

    args.ckpt_dir = str(Path(base_args.ckpt_dir) / "p0_search" / stage / tag)
    args.log_dir = str(Path(base_args.log_dir) / "p0_search" / stage / tag)
    args.output_dir = str(Path(base_args.output_dir) / "p0_search" / stage / tag)
    args.log_name = "graph_profile.log"
    return args


def extract_primary_fold_metrics(fold_metrics: dict[str, dict]) -> dict[str, float]:
    """从 run() 返回值中取第一个 fold 的 test all 指标。"""
    if not fold_metrics:
        return {}
    first_key = sorted(fold_metrics.keys(), key=lambda item: (len(item), item))[0]
    metrics = fold_metrics[first_key] or {}
    return {str(key): float(value) for key, value in metrics.items()}


def compare_trials(left: TrialResult, right: TrialResult) -> TrialResult:
    """按 test FMR 优先、rouge_l 次优选择更优试验。"""
    left_fmr = float(left.metrics.get("FMR", float("-inf")))
    right_fmr = float(right.metrics.get("FMR", float("-inf")))
    if left_fmr != right_fmr:
        return left if left_fmr > right_fmr else right

    left_rouge = float(left.metrics.get("rouge_l", float("-inf")))
    right_rouge = float(right.metrics.get("rouge_l", float("-inf")))
    if left_rouge != right_rouge:
        return left if left_rouge > right_rouge else right

    return left


def pick_best_trial(trials: list[TrialResult]) -> TrialResult:
    if not trials:
        raise ValueError("Cannot pick best trial from an empty list")
    best = trials[0]
    for trial in trials[1:]:
        best = compare_trials(trial, best)
    return best


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
    stage1_trials: list[TrialResult],
    stage2_trials: list[TrialResult],
    stage3_trials: list[TrialResult],
    best_lambda_feat: float,
    best_evidence_bonus: float,
    best_top_m_evidence: int,
) -> str:
    """把所有阶段的 test 指标渲染为 Markdown 汇总。"""
    lines = [
        "# graph_llm P0 顺序调参结果",
        "",
        f"- dataset: `{dataset_name}`",
        f"- split_indices: `{split_indices}`",
        f"- updated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- selection: test `FMR`（并列时 `rouge_l`）",
        "",
        "## 最终选定超参",
        "",
        f"- `lambda_feat`: `{best_lambda_feat}`",
        f"- `evidence_bonus`: `{best_evidence_bonus}`",
        f"- `top_m_evidence`: `{best_top_m_evidence}`",
        "",
    ]

    stage_sections = [
        (
            "Stage 1: lambda_feat",
            stage1_trials,
            f"固定 evidence_bonus={DEFAULT_EVIDENCE_BONUS}, top_m_evidence={DEFAULT_TOP_M_EVIDENCE}",
        ),
        (
            "Stage 2: evidence_bonus",
            stage2_trials,
            f"固定 lambda_feat={best_lambda_feat}, top_m_evidence={DEFAULT_TOP_M_EVIDENCE}",
        ),
        (
            "Stage 3: top_m_evidence",
            stage3_trials,
            f"固定 lambda_feat={best_lambda_feat}, evidence_bonus={best_evidence_bonus}",
        ),
    ]

    header = (
        "| stage | tag | lambda_feat | evidence_bonus | top_m_evidence | "
        + " | ".join(METRIC_COLUMNS)
        + " | best |"
    )
    separator = (
        "| --- | --- | ---: | ---: | ---: | "
        + " | ".join(["---:"] * len(METRIC_COLUMNS))
        + " | --- |"
    )

    for title, trials, frozen_note in stage_sections:
        lines.extend([f"## {title}", "", frozen_note, "", header, separator])
        for trial in trials:
            metric_cells = " | ".join(
                _format_metric(trial.metrics.get(name)) for name in METRIC_COLUMNS
            )
            lines.append(
                "| {stage} | {tag} | {lambda_feat} | {evidence_bonus} | {top_m_evidence} | {metrics} | {best} |".format(
                    stage=trial.stage,
                    tag=trial.tag,
                    lambda_feat=_format_float(trial.lambda_feat),
                    evidence_bonus=_format_float(trial.evidence_bonus),
                    top_m_evidence=trial.top_m_evidence,
                    metrics=metric_cells,
                    best="yes" if trial.is_best else "",
                )
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_results_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def print_experiment_summary(args, stage: str, trial: TrialResult) -> None:
    print("=" * 88)
    print(f"[{stage}] {trial.tag}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"split_indices: {args.split_indices}")
    print(f"lambda_feat: {args.lambda_feat}")
    print(f"evidence_bonus: {args.evidence_bonus}")
    print(f"top_m_evidence: {args.top_m_evidence}")
    print(f"ckpt_dir: {args.ckpt_dir}")
    print(f"log_dir: {args.log_dir}")
    print(f"output_dir: {args.output_dir}")
    print("=" * 88)


def run_trial(
    base_args,
    *,
    stage: str,
    lambda_feat: float,
    evidence_bonus: float,
    top_m_evidence: int,
    dry_run: bool,
) -> TrialResult:
    """执行单次试验并返回 test metrics。"""
    args = build_experiment_args(
        base_args,
        stage=stage,
        lambda_feat=lambda_feat,
        evidence_bonus=evidence_bonus,
        top_m_evidence=top_m_evidence,
    )
    trial = TrialResult(
        stage=stage,
        lambda_feat=lambda_feat,
        evidence_bonus=evidence_bonus,
        top_m_evidence=top_m_evidence,
    )
    print_experiment_summary(args, stage, trial)
    if dry_run:
        return trial

    from graph_llm.train import run

    fold_metrics = run(args)
    trial.metrics = extract_primary_fold_metrics(fold_metrics)
    print(
        f"[{stage}] {trial.tag} test FMR={trial.metrics.get('FMR', float('nan')):.4f} "
        f"rouge_l={trial.metrics.get('rouge_l', float('nan')):.4f}"
    )
    return trial


def mark_stage_best(trials: list[TrialResult]) -> TrialResult:
    """标记某一阶段的最优试验。"""
    for trial in trials:
        trial.is_best = False
    best = pick_best_trial(trials)
    for trial in trials:
        if trial.tag == best.tag:
            trial.is_best = True
    return best


def refresh_results_file(
    results_file: Path,
    *,
    dataset_name: str,
    split_indices: str,
    stage1_trials: list[TrialResult],
    stage2_trials: list[TrialResult],
    stage3_trials: list[TrialResult],
    best_lambda_feat: float,
    best_evidence_bonus: float,
    best_top_m_evidence: int,
) -> None:
    """每次试验结束后立即重写汇总文件，避免中断丢结果。"""
    content = render_results_markdown(
        dataset_name=dataset_name,
        split_indices=split_indices,
        stage1_trials=stage1_trials,
        stage2_trials=stage2_trials,
        stage3_trials=stage3_trials,
        best_lambda_feat=best_lambda_feat,
        best_evidence_bonus=best_evidence_bonus,
        best_top_m_evidence=best_top_m_evidence,
    )
    write_results_file(results_file, content)


def main() -> None:
    parser = build_arg_parser()
    parser.set_defaults(
        lambda_feat=1e-2,
        evidence_bonus=DEFAULT_EVIDENCE_BONUS,
        top_m_evidence=DEFAULT_TOP_M_EVIDENCE,
        review_top_k_user=16,
        review_top_k_item=32,
        user_review_prefix_len=4,
        item_review_prefix_len=4,
        lambda_prefix_feature=0.1,
        devices="1",
        model_path=str(PACKAGE_ROOT / "pretrain_llm" / "qwen3-4b"),
        embedding_model_path=str(PACKAGE_ROOT / "pretrain_llm" / "qwen3-embedding-0.6b"),
        profile_dir=str(PACKAGE_ROOT / "data" / "profiles"),
        data_dir=str(PACKAGE_ROOT / "data"),
    )
    parser.add_argument(
        "--results_file",
        default="",
        help="汇总 Markdown 路径；默认 graph_llm/log/p0_search/<dataset>/p0_search_results.md",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印试验配置，不启动训练。",
    )
    base_args = parser.parse_args()

    dry_run = bool(getattr(base_args, "dry_run", False))
    if hasattr(base_args, "dry_run"):
        delattr(base_args, "dry_run")

    results_file = (
        Path(base_args.results_file).expanduser()
        if getattr(base_args, "results_file", "")
        else default_results_file(base_args.dataset_name)
    )
    if hasattr(base_args, "results_file"):
        delattr(base_args, "results_file")

    stage1_trials: list[TrialResult] = []
    stage2_trials: list[TrialResult] = []
    stage3_trials: list[TrialResult] = []

    best_lambda_feat = float(base_args.lambda_feat)
    best_evidence_bonus = float(base_args.evidence_bonus)
    best_top_m_evidence = int(base_args.top_m_evidence)

    def _refresh() -> None:
        refresh_results_file(
            results_file,
            dataset_name=base_args.dataset_name,
            split_indices=base_args.split_indices,
            stage1_trials=stage1_trials,
            stage2_trials=stage2_trials,
            stage3_trials=stage3_trials,
            best_lambda_feat=best_lambda_feat,
            best_evidence_bonus=best_evidence_bonus,
            best_top_m_evidence=best_top_m_evidence,
        )

    # Stage 1: 搜索 lambda_feat
    for lambda_feat in STAGE1_LAMBDA_FEAT:
        trial = run_trial(
            base_args,
            stage="stage1_lambda_feat",
            lambda_feat=lambda_feat,
            evidence_bonus=DEFAULT_EVIDENCE_BONUS,
            top_m_evidence=DEFAULT_TOP_M_EVIDENCE,
            dry_run=dry_run,
        )
        stage1_trials.append(trial)
        _refresh()

    best_stage1 = mark_stage_best(stage1_trials)
    best_lambda_feat = best_stage1.lambda_feat
    print(f"Stage 1 best: lambda_feat={best_lambda_feat} (tag={best_stage1.tag})")
    _refresh()

    # Stage 2: 固定最优 lambda_feat，搜索 evidence_bonus
    for evidence_bonus in STAGE2_EVIDENCE_BONUS:
        trial = run_trial(
            base_args,
            stage="stage2_evidence_bonus",
            lambda_feat=best_lambda_feat,
            evidence_bonus=evidence_bonus,
            top_m_evidence=DEFAULT_TOP_M_EVIDENCE,
            dry_run=dry_run,
        )
        stage2_trials.append(trial)
        _refresh()

    best_stage2 = mark_stage_best(stage2_trials)
    best_evidence_bonus = best_stage2.evidence_bonus
    print(f"Stage 2 best: evidence_bonus={best_evidence_bonus} (tag={best_stage2.tag})")
    _refresh()

    # Stage 3: 固定前两阶段最优值，搜索 top_m_evidence
    for top_m_evidence in STAGE3_TOP_M_EVIDENCE:
        trial = run_trial(
            base_args,
            stage="stage3_top_m_evidence",
            lambda_feat=best_lambda_feat,
            evidence_bonus=best_evidence_bonus,
            top_m_evidence=top_m_evidence,
            dry_run=dry_run,
        )
        stage3_trials.append(trial)
        _refresh()

    best_stage3 = mark_stage_best(stage3_trials)
    best_top_m_evidence = best_stage3.top_m_evidence
    print(f"Stage 3 best: top_m_evidence={best_top_m_evidence} (tag={best_stage3.tag})")
    _refresh()

    print(f"P0 search complete. Results written to: {results_file}")


if __name__ == "__main__":
    main()
