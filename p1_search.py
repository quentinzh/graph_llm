#!/usr/bin/env python
"""在冻结 P0 最优超参基础上，顺序搜索 lambda_selector -> selector_feature_positive_weight -> lambda_prefix_feature。"""

from __future__ import annotations

import copy
import os
import re
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
STAGE1_LAMBDA_SELECTOR = [0.1, 0.2, 0.3]
STAGE2_SELECTOR_FEATURE_POSITIVE_WEIGHT = [2.0, 3.0, 4.0]
STAGE3_LAMBDA_PREFIX_FEATURE = [0.1, 0.2, 0.3]

# 阶段 1/2 冻结的 P1 默认值
DEFAULT_SELECTOR_FEATURE_POSITIVE_WEIGHT = 3.0
DEFAULT_LAMBDA_PREFIX_FEATURE = 0.1

# 汇总表展示的 test 指标列（仅写 run() 实际返回的键）
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
    """把浮点超参格式化成目录 tag 友好的字符串。"""
    if 0 < abs(value) <= 1e-2:
        text = f"{value:.0e}"
        mantissa, exponent = text.split("e", 1)
        sign = "-" if exponent.startswith("-") else ""
        digits = exponent.lstrip("+-").lstrip("0") or "0"
        return f"{mantissa}e{sign}{digits}"
    return f"{value:g}"


def experiment_tag(
    lambda_selector: float,
    selector_feature_positive_weight: float,
    lambda_prefix_feature: float,
) -> str:
    return (
        f"lsel{_format_float(lambda_selector)}_"
        f"sfpw{_format_float(selector_feature_positive_weight)}_"
        f"lpfx{_format_float(lambda_prefix_feature)}"
    )


def default_p0_results_file(dataset_name: str) -> Path:
    safe_name = dataset_name.strip("/")
    return PACKAGE_ROOT / "log" / "p0_search" / safe_name / "p0_search_results.md"


def default_results_file(dataset_name: str) -> Path:
    safe_name = dataset_name.strip("/")
    return PACKAGE_ROOT / "log" / "p1_search" / safe_name / "p1_search_results.md"


def _cli_explicit(flag_name: str) -> bool:
    """判断某个 CLI 参数是否在命令行中显式传入。"""
    prefix = f"--{flag_name}"
    return any(arg == prefix or arg.startswith(prefix + "=") for arg in sys.argv)


def parse_p0_results(path: Path) -> dict[str, float | int]:
    """从 p0_search_results.md 的「最终选定超参」段落解析 P0 三参。"""
    if not path.is_file():
        raise FileNotFoundError(f"P0 results file not found: {path}")

    text = path.read_text(encoding="utf-8")
    patterns = {
        "lambda_feat": r"`lambda_feat`:\s*`([^`]+)`",
        "evidence_bonus": r"`evidence_bonus`:\s*`([^`]+)`",
        "top_m_evidence": r"`top_m_evidence`:\s*`([^`]+)`",
    }
    parsed: dict[str, float | int] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Cannot parse `{key}` from P0 results file: {path}")
        raw = match.group(1).strip()
        if key == "top_m_evidence":
            parsed[key] = int(float(raw))
        else:
            parsed[key] = float(raw)
    return parsed


@dataclass
class TrialResult:
    """单次试验记录。"""

    stage: str
    lambda_feat: float
    evidence_bonus: float
    top_m_evidence: int
    lambda_selector: float
    selector_feature_positive_weight: float
    lambda_prefix_feature: float
    metrics: dict[str, float] = field(default_factory=dict)
    tag: str = ""
    is_best: bool = False

    def __post_init__(self) -> None:
        if not self.tag:
            self.tag = experiment_tag(
                self.lambda_selector,
                self.selector_feature_positive_weight,
                self.lambda_prefix_feature,
            )


def build_experiment_args(
    base_args,
    *,
    stage: str,
    lambda_feat: float,
    evidence_bonus: float,
    top_m_evidence: int,
    lambda_selector: float,
    selector_feature_positive_weight: float,
    lambda_prefix_feature: float,
):
    """为单次试验构造隔离目录的 args。"""
    args = copy.copy(base_args)
    tag = experiment_tag(
        lambda_selector,
        selector_feature_positive_weight,
        lambda_prefix_feature,
    )

    # 冻结 P0 最优超参
    args.lambda_feat = float(lambda_feat)
    args.evidence_bonus = float(evidence_bonus)
    args.top_m_evidence = int(top_m_evidence)

    # 当前 P1 搜索超参
    args.lambda_selector = float(lambda_selector)
    args.selector_feature_positive_weight = float(selector_feature_positive_weight)
    args.lambda_prefix_feature = float(lambda_prefix_feature)

    args.ckpt_dir = str(Path(base_args.ckpt_dir) / "p1_search" / stage / tag)
    args.log_dir = str(Path(base_args.log_dir) / "p1_search" / stage / tag)
    args.output_dir = str(Path(base_args.output_dir) / "p1_search" / stage / tag)
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
    frozen_lambda_feat: float,
    frozen_evidence_bonus: float,
    frozen_top_m_evidence: int,
    stage1_trials: list[TrialResult],
    stage2_trials: list[TrialResult],
    stage3_trials: list[TrialResult],
    best_lambda_selector: float,
    best_selector_feature_positive_weight: float,
    best_lambda_prefix_feature: float,
) -> str:
    """把所有阶段的 test 指标渲染为 Markdown 汇总。"""
    lines = [
        "# graph_llm P1 顺序调参结果",
        "",
        f"- dataset: `{dataset_name}`",
        f"- split_indices: `{split_indices}`",
        f"- updated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- selection: test `FMR`（并列时 `rouge_l`）",
        "",
        "## 冻结的 P0 最优超参",
        "",
        f"- `lambda_feat`: `{frozen_lambda_feat}`",
        f"- `evidence_bonus`: `{frozen_evidence_bonus}`",
        f"- `top_m_evidence`: `{frozen_top_m_evidence}`",
        "",
        "## 最终选定 P1 超参",
        "",
        f"- `lambda_selector`: `{best_lambda_selector}`",
        f"- `selector_feature_positive_weight`: `{best_selector_feature_positive_weight}`",
        f"- `lambda_prefix_feature`: `{best_lambda_prefix_feature}`",
        "",
    ]

    stage_sections = [
        (
            "Stage 1: lambda_selector",
            stage1_trials,
            (
                f"固定 lambda_feat={frozen_lambda_feat}, evidence_bonus={frozen_evidence_bonus}, "
                f"top_m_evidence={frozen_top_m_evidence}, "
                f"selector_feature_positive_weight={DEFAULT_SELECTOR_FEATURE_POSITIVE_WEIGHT}, "
                f"lambda_prefix_feature={DEFAULT_LAMBDA_PREFIX_FEATURE}"
            ),
        ),
        (
            "Stage 2: selector_feature_positive_weight",
            stage2_trials,
            (
                f"固定 lambda_feat={frozen_lambda_feat}, evidence_bonus={frozen_evidence_bonus}, "
                f"top_m_evidence={frozen_top_m_evidence}, lambda_selector={best_lambda_selector}, "
                f"lambda_prefix_feature={DEFAULT_LAMBDA_PREFIX_FEATURE}"
            ),
        ),
        (
            "Stage 3: lambda_prefix_feature",
            stage3_trials,
            (
                f"固定 lambda_feat={frozen_lambda_feat}, evidence_bonus={frozen_evidence_bonus}, "
                f"top_m_evidence={frozen_top_m_evidence}, lambda_selector={best_lambda_selector}, "
                f"selector_feature_positive_weight={best_selector_feature_positive_weight}"
            ),
        ),
    ]

    header = (
        "| stage | tag | lambda_selector | selector_feature_positive_weight | lambda_prefix_feature | "
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
                "| {stage} | {tag} | {lambda_selector} | {selector_feature_positive_weight} | "
                "{lambda_prefix_feature} | {metrics} | {best} |".format(
                    stage=trial.stage,
                    tag=trial.tag,
                    lambda_selector=_format_float(trial.lambda_selector),
                    selector_feature_positive_weight=_format_float(
                        trial.selector_feature_positive_weight
                    ),
                    lambda_prefix_feature=_format_float(trial.lambda_prefix_feature),
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
    print(
        "frozen P0: "
        f"lambda_feat={args.lambda_feat} "
        f"evidence_bonus={args.evidence_bonus} "
        f"top_m_evidence={args.top_m_evidence}"
    )
    print(f"lambda_selector: {args.lambda_selector}")
    print(f"selector_feature_positive_weight: {args.selector_feature_positive_weight}")
    print(f"lambda_prefix_feature: {args.lambda_prefix_feature}")
    print(f"ckpt_dir: {args.ckpt_dir}")
    print(f"log_dir: {args.log_dir}")
    print(f"output_dir: {args.output_dir}")
    print("=" * 88)


def run_trial(
    base_args,
    *,
    stage: str,
    frozen_lambda_feat: float,
    frozen_evidence_bonus: float,
    frozen_top_m_evidence: int,
    lambda_selector: float,
    selector_feature_positive_weight: float,
    lambda_prefix_feature: float,
    dry_run: bool,
) -> TrialResult:
    """执行单次试验并返回 test metrics。"""
    args = build_experiment_args(
        base_args,
        stage=stage,
        lambda_feat=frozen_lambda_feat,
        evidence_bonus=frozen_evidence_bonus,
        top_m_evidence=frozen_top_m_evidence,
        lambda_selector=lambda_selector,
        selector_feature_positive_weight=selector_feature_positive_weight,
        lambda_prefix_feature=lambda_prefix_feature,
    )
    trial = TrialResult(
        stage=stage,
        lambda_feat=frozen_lambda_feat,
        evidence_bonus=frozen_evidence_bonus,
        top_m_evidence=frozen_top_m_evidence,
        lambda_selector=lambda_selector,
        selector_feature_positive_weight=selector_feature_positive_weight,
        lambda_prefix_feature=lambda_prefix_feature,
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
    frozen_lambda_feat: float,
    frozen_evidence_bonus: float,
    frozen_top_m_evidence: int,
    stage1_trials: list[TrialResult],
    stage2_trials: list[TrialResult],
    stage3_trials: list[TrialResult],
    best_lambda_selector: float,
    best_selector_feature_positive_weight: float,
    best_lambda_prefix_feature: float,
) -> None:
    """每次试验结束后立即重写汇总文件，避免中断丢结果。"""
    content = render_results_markdown(
        dataset_name=dataset_name,
        split_indices=split_indices,
        frozen_lambda_feat=frozen_lambda_feat,
        frozen_evidence_bonus=frozen_evidence_bonus,
        frozen_top_m_evidence=frozen_top_m_evidence,
        stage1_trials=stage1_trials,
        stage2_trials=stage2_trials,
        stage3_trials=stage3_trials,
        best_lambda_selector=best_lambda_selector,
        best_selector_feature_positive_weight=best_selector_feature_positive_weight,
        best_lambda_prefix_feature=best_lambda_prefix_feature,
    )
    write_results_file(results_file, content)


def resolve_frozen_p0_params(base_args, p0_results_file: Path) -> tuple[float, float, int]:
    """优先使用 CLI 显式传入的 P0 参数，否则从 p0_search_results.md 读取。"""
    p0_from_file = parse_p0_results(p0_results_file)
    lambda_feat = (
        float(base_args.lambda_feat)
        if _cli_explicit("lambda_feat")
        else float(p0_from_file["lambda_feat"])
    )
    evidence_bonus = (
        float(base_args.evidence_bonus)
        if _cli_explicit("evidence_bonus")
        else float(p0_from_file["evidence_bonus"])
    )
    top_m_evidence = (
        int(base_args.top_m_evidence)
        if _cli_explicit("top_m_evidence")
        else int(p0_from_file["top_m_evidence"])
    )
    return lambda_feat, evidence_bonus, top_m_evidence


def main() -> None:
    parser = build_arg_parser()
    parser.set_defaults(
        lambda_selector=0.1,
        selector_feature_positive_weight=DEFAULT_SELECTOR_FEATURE_POSITIVE_WEIGHT,
        lambda_prefix_feature=DEFAULT_LAMBDA_PREFIX_FEATURE,
        review_top_k_user=16,
        review_top_k_item=32,
        user_review_prefix_len=4,
        item_review_prefix_len=4,
        devices="1",
        model_path=str(PACKAGE_ROOT / "pretrain_llm" / "qwen3-4b"),
        embedding_model_path=str(PACKAGE_ROOT / "pretrain_llm" / "qwen3-embedding-0.6b"),
        profile_dir=str(PACKAGE_ROOT / "data" / "profiles"),
        data_dir=str(PACKAGE_ROOT / "data"),
    )
    parser.add_argument(
        "--p0_results_file",
        default="",
        help="P0 汇总 Markdown 路径；默认 graph_llm/log/p0_search/<dataset>/p0_search_results.md",
    )
    parser.add_argument(
        "--results_file",
        default="",
        help="P1 汇总 Markdown 路径；默认 graph_llm/log/p1_search/<dataset>/p1_search_results.md",
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

    p0_results_file = (
        Path(base_args.p0_results_file).expanduser()
        if getattr(base_args, "p0_results_file", "")
        else default_p0_results_file(base_args.dataset_name)
    )
    if hasattr(base_args, "p0_results_file"):
        delattr(base_args, "p0_results_file")

    results_file = (
        Path(base_args.results_file).expanduser()
        if getattr(base_args, "results_file", "")
        else default_results_file(base_args.dataset_name)
    )
    if hasattr(base_args, "results_file"):
        delattr(base_args, "results_file")

    frozen_lambda_feat, frozen_evidence_bonus, frozen_top_m_evidence = resolve_frozen_p0_params(
        base_args,
        p0_results_file,
    )
    base_args.lambda_feat = frozen_lambda_feat
    base_args.evidence_bonus = frozen_evidence_bonus
    base_args.top_m_evidence = frozen_top_m_evidence

    print(
        "Frozen P0 params: "
        f"lambda_feat={frozen_lambda_feat} "
        f"evidence_bonus={frozen_evidence_bonus} "
        f"top_m_evidence={frozen_top_m_evidence} "
        f"(from {p0_results_file})"
    )

    stage1_trials: list[TrialResult] = []
    stage2_trials: list[TrialResult] = []
    stage3_trials: list[TrialResult] = []

    best_lambda_selector = float(base_args.lambda_selector)
    best_selector_feature_positive_weight = float(base_args.selector_feature_positive_weight)
    best_lambda_prefix_feature = float(base_args.lambda_prefix_feature)

    def _refresh() -> None:
        refresh_results_file(
            results_file,
            dataset_name=base_args.dataset_name,
            split_indices=base_args.split_indices,
            frozen_lambda_feat=frozen_lambda_feat,
            frozen_evidence_bonus=frozen_evidence_bonus,
            frozen_top_m_evidence=frozen_top_m_evidence,
            stage1_trials=stage1_trials,
            stage2_trials=stage2_trials,
            stage3_trials=stage3_trials,
            best_lambda_selector=best_lambda_selector,
            best_selector_feature_positive_weight=best_selector_feature_positive_weight,
            best_lambda_prefix_feature=best_lambda_prefix_feature,
        )

    # Stage 1: 搜索 lambda_selector
    for lambda_selector in STAGE1_LAMBDA_SELECTOR:
        trial = run_trial(
            base_args,
            stage="stage1_lambda_selector",
            frozen_lambda_feat=frozen_lambda_feat,
            frozen_evidence_bonus=frozen_evidence_bonus,
            frozen_top_m_evidence=frozen_top_m_evidence,
            lambda_selector=lambda_selector,
            selector_feature_positive_weight=DEFAULT_SELECTOR_FEATURE_POSITIVE_WEIGHT,
            lambda_prefix_feature=DEFAULT_LAMBDA_PREFIX_FEATURE,
            dry_run=dry_run,
        )
        stage1_trials.append(trial)
        _refresh()

    best_stage1 = mark_stage_best(stage1_trials)
    best_lambda_selector = best_stage1.lambda_selector
    print(f"Stage 1 best: lambda_selector={best_lambda_selector} (tag={best_stage1.tag})")
    _refresh()

    # Stage 2: 固定最优 lambda_selector，搜索 selector_feature_positive_weight
    for selector_feature_positive_weight in STAGE2_SELECTOR_FEATURE_POSITIVE_WEIGHT:
        trial = run_trial(
            base_args,
            stage="stage2_selector_feature_positive_weight",
            frozen_lambda_feat=frozen_lambda_feat,
            frozen_evidence_bonus=frozen_evidence_bonus,
            frozen_top_m_evidence=frozen_top_m_evidence,
            lambda_selector=best_lambda_selector,
            selector_feature_positive_weight=selector_feature_positive_weight,
            lambda_prefix_feature=DEFAULT_LAMBDA_PREFIX_FEATURE,
            dry_run=dry_run,
        )
        stage2_trials.append(trial)
        _refresh()

    best_stage2 = mark_stage_best(stage2_trials)
    best_selector_feature_positive_weight = best_stage2.selector_feature_positive_weight
    print(
        "Stage 2 best: "
        f"selector_feature_positive_weight={best_selector_feature_positive_weight} "
        f"(tag={best_stage2.tag})"
    )
    _refresh()

    # Stage 3: 固定前两阶段最优值，搜索 lambda_prefix_feature
    for lambda_prefix_feature in STAGE3_LAMBDA_PREFIX_FEATURE:
        trial = run_trial(
            base_args,
            stage="stage3_lambda_prefix_feature",
            frozen_lambda_feat=frozen_lambda_feat,
            frozen_evidence_bonus=frozen_evidence_bonus,
            frozen_top_m_evidence=frozen_top_m_evidence,
            lambda_selector=best_lambda_selector,
            selector_feature_positive_weight=best_selector_feature_positive_weight,
            lambda_prefix_feature=lambda_prefix_feature,
            dry_run=dry_run,
        )
        stage3_trials.append(trial)
        _refresh()

    best_stage3 = mark_stage_best(stage3_trials)
    best_lambda_prefix_feature = best_stage3.lambda_prefix_feature
    print(
        f"Stage 3 best: lambda_prefix_feature={best_lambda_prefix_feature} "
        f"(tag={best_stage3.tag})"
    )
    _refresh()

    print(f"P1 search complete. Results written to: {results_file}")


if __name__ == "__main__":
    main()
