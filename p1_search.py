#!/usr/bin/env python
"""在冻结 P0 最优超参基础上，顺序搜索 lambda_selector -> selector_feature_positive_weight -> lambda_prefix_feature。

支持 ``--start_stage 2/3`` 续跑：保留汇总里已有 Stage1（或 Stage1+2）结果，跳过对应训练，
在正确的 ``lambda_selector`` 上重跑后续阶段。适用于 test reload 修复后纠正 Stage2/3 汇总，
而非 Stage1 选参逻辑本身有误（reload 修复后 Stage1 按 test FMR 选参已可信）。
"""

from __future__ import annotations

import copy
import os
import re
import shutil
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

STAGE1_NAME = "stage1_lambda_selector"
STAGE2_NAME = "stage2_selector_feature_positive_weight"
STAGE3_NAME = "stage3_lambda_prefix_feature"

STAGE_SECTION_TITLE = {
    STAGE1_NAME: "Stage 1: lambda_selector",
    STAGE2_NAME: "Stage 2: selector_feature_positive_weight",
    STAGE3_NAME: "Stage 3: lambda_prefix_feature",
}

# 续跑时可从 Stage1 复制 checkpoint 到 Stage2 的来源阶段
CHECKPOINT_REUSE_SOURCE_STAGE = {
    STAGE2_NAME: STAGE1_NAME,
}

# 汇总表展示的 test 指标列（仅写 run() 实际返回的键）
METRIC_COLUMNS = [
    "BLEU-1",
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


def parse_frozen_p0_from_p1_results(path: Path) -> tuple[float, float, int]:
    """从 p1_search_results.md 的「冻结的 P0 最优超参」段落解析三参。"""
    if not path.is_file():
        raise FileNotFoundError(f"P1 results file not found: {path}")
    parsed = parse_p0_results(path)  # 复用同一套反引号字段格式
    return float(parsed["lambda_feat"]), float(parsed["evidence_bonus"]), int(parsed["top_m_evidence"])


def _section_table_lines(text: str, section_title: str) -> list[str]:
    """提取某个 ``## Stage ...`` 小节里 Markdown 表格的数据行。"""
    lines = text.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == f"## {section_title}":
            start_idx = idx
            break
    if start_idx is None:
        return []

    table_rows: list[str] = []
    in_table = False
    for line in lines[start_idx + 1 :]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if not stripped:
            if in_table:
                break
            continue
        if (
            stripped.startswith("|")
            and stripped.startswith("| stage | tag |")
        ):
            in_table = True
            continue
        if stripped.startswith("| ---"):
            continue
        if in_table and stripped.startswith("|"):
            table_rows.append(stripped)
    return table_rows


def parse_stage_trials_from_results(
    path: Path,
    stage_name: str,
    *,
    frozen_lambda_feat: float,
    frozen_evidence_bonus: float,
    frozen_top_m_evidence: int,
) -> list[TrialResult]:
    """从 p1_search_results.md 解析某一阶段的试验行。"""
    section_title = STAGE_SECTION_TITLE.get(stage_name)
    if section_title is None:
        raise ValueError(f"Unknown stage name: {stage_name}")

    if not path.is_file():
        raise FileNotFoundError(f"P1 results file not found: {path}")

    text = path.read_text(encoding="utf-8")
    rows = _section_table_lines(text, section_title)
    if not rows:
        raise ValueError(
            f"No table rows found for stage {stage_name!r} in results file: {path}"
        )

    trials: list[TrialResult] = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if len(cells) < 6:
            continue
        row_stage, tag = cells[0], cells[1]
        if row_stage != stage_name:
            continue
        lambda_selector = float(cells[2])
        selector_feature_positive_weight = float(cells[3])
        lambda_prefix_feature = float(cells[4])
        metric_cells = cells[5 : 5 + len(METRIC_COLUMNS)]
        best_cell = cells[5 + len(METRIC_COLUMNS)] if len(cells) > 5 + len(METRIC_COLUMNS) else ""
        metrics: dict[str, float] = {}
        for name, raw in zip(METRIC_COLUMNS, metric_cells):
            raw = raw.strip()
            if raw and raw != "-":
                try:
                    metrics[name] = float(raw)
                except ValueError:
                    pass
        trials.append(
            TrialResult(
                stage=stage_name,
                lambda_feat=frozen_lambda_feat,
                evidence_bonus=frozen_evidence_bonus,
                top_m_evidence=frozen_top_m_evidence,
                lambda_selector=lambda_selector,
                selector_feature_positive_weight=selector_feature_positive_weight,
                lambda_prefix_feature=lambda_prefix_feature,
                metrics=metrics,
                tag=tag,
                is_best=best_cell.lower() == "yes",
            )
        )
    if not trials:
        raise ValueError(
            f"Parsed zero trials for stage {stage_name!r} from results file: {path}"
        )
    return trials


def _primary_split_index(split_indices: str) -> str:
    parts = [part.strip() for part in str(split_indices).split(",") if part.strip()]
    if not parts:
        raise ValueError("split_indices is empty")
    return parts[0]


def _trial_checkpoint_prefix(ckpt_root: Path, dataset_name: str, split_index: str) -> Path:
    """与 trainer 一致：``ckpt_root / dataset_name / {split}`` 前缀（无后缀）。"""
    return Path(ckpt_root) / dataset_name / split_index


def checkpoint_is_ready(ckpt_root: Path, dataset_name: str, split_index: str) -> bool:
    """判断某 trial 根目录下是否已有可 only_eval 的 best checkpoint。"""
    prefix = _trial_checkpoint_prefix(ckpt_root, dataset_name, split_index)
    model_dir = Path(f"{prefix}model")
    selector_path = Path(f"{prefix}selector.bin")
    return model_dir.is_dir() and selector_path.is_file()


def copy_trial_checkpoint(
    src_ckpt_root: Path,
    dst_ckpt_root: Path,
    dataset_name: str,
    split_index: str,
) -> None:
    """把 Stage1 等同 tag 的 checkpoint 复制到 Stage2 目录，供 only_eval 使用。"""
    src_prefix = _trial_checkpoint_prefix(src_ckpt_root, dataset_name, split_index)
    dst_prefix = _trial_checkpoint_prefix(dst_ckpt_root, dataset_name, split_index)
    dst_prefix.parent.mkdir(parents=True, exist_ok=True)

    for suffix in ("model", "selector.bin", "review_prefix.bin", "graph_config.json"):
        src = Path(f"{src_prefix}{suffix}")
        dst = Path(f"{dst_prefix}{suffix}")
        if not src.exists():
            if suffix in ("model", "selector.bin"):
                raise FileNotFoundError(f"Missing checkpoint artifact: {src}")
            continue
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    print(f"Reused checkpoint: {src_ckpt_root} -> {dst_ckpt_root}")


def maybe_reuse_prior_stage_checkpoint(
    base_args,
    *,
    stage: str,
    tag: str,
    split_index: str,
) -> bool:
    """若当前 stage 尚无 checkpoint，尝试从上一阶段同 tag 复制。"""
    source_stage = CHECKPOINT_REUSE_SOURCE_STAGE.get(stage)
    if source_stage is None:
        return False

    dst_root = Path(base_args.ckpt_dir) / "p1_search" / stage / tag
    if checkpoint_is_ready(dst_root, base_args.dataset_name, split_index):
        return False

    src_root = Path(base_args.ckpt_dir) / "p1_search" / source_stage / tag
    if not checkpoint_is_ready(src_root, base_args.dataset_name, split_index):
        return False

    copy_trial_checkpoint(
        src_root,
        dst_root,
        base_args.dataset_name,
        split_index,
    )
    return True


def resolve_best_from_loaded_trials(
    trials: list[TrialResult],
    *,
    param_name: str,
    cli_value: float | None,
) -> float:
    """续跑时从已加载 trial 或 CLI 显式值确定上一阶段最优超参。"""
    if cli_value is not None:
        return float(cli_value)
    marked = [trial for trial in trials if trial.is_best]
    if len(marked) == 1:
        return float(getattr(marked[0], param_name))
    if marked:
        return float(getattr(pick_best_trial(marked), param_name))
    return float(getattr(pick_best_trial(trials), param_name))


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
    split_index: str,
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

    reused = maybe_reuse_prior_stage_checkpoint(
        base_args,
        stage=stage,
        tag=trial.tag,
        split_index=split_index,
    )
    if reused:
        args.only_eval = True
        print(f"[{stage}] {trial.tag}: reused prior-stage checkpoint, running only_eval")

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
    parser.add_argument(
        "--start_stage",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="从第几阶段开始搜索：1=全流程；2=跳过 Stage1（从 results 读 Stage1）；3=再跳过 Stage2。",
    )
    base_args = parser.parse_args()

    dry_run = bool(getattr(base_args, "dry_run", False))
    if hasattr(base_args, "dry_run"):
        delattr(base_args, "dry_run")

    start_stage = int(getattr(base_args, "start_stage", 1))
    if hasattr(base_args, "start_stage"):
        delattr(base_args, "start_stage")

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

    split_index = _primary_split_index(base_args.split_indices)
    print(f"start_stage={start_stage} (resume skips training for earlier stages)")

    stage1_trials: list[TrialResult] = []
    stage2_trials: list[TrialResult] = []
    stage3_trials: list[TrialResult] = []

    best_lambda_selector = float(base_args.lambda_selector)
    best_selector_feature_positive_weight = float(base_args.selector_feature_positive_weight)
    best_lambda_prefix_feature = float(base_args.lambda_prefix_feature)

    if start_stage >= 2:
        stage1_trials = parse_stage_trials_from_results(
            results_file,
            STAGE1_NAME,
            frozen_lambda_feat=frozen_lambda_feat,
            frozen_evidence_bonus=frozen_evidence_bonus,
            frozen_top_m_evidence=frozen_top_m_evidence,
        )
        cli_lsel = float(base_args.lambda_selector) if _cli_explicit("lambda_selector") else None
        best_lambda_selector = resolve_best_from_loaded_trials(
            stage1_trials,
            param_name="lambda_selector",
            cli_value=cli_lsel,
        )
        for trial in stage1_trials:
            trial.is_best = trial.lambda_selector == best_lambda_selector
        print(
            f"Resume: loaded {len(stage1_trials)} Stage1 trial(s); "
            f"fixed lambda_selector={best_lambda_selector}"
        )

    if start_stage >= 3:
        stage2_trials = parse_stage_trials_from_results(
            results_file,
            STAGE2_NAME,
            frozen_lambda_feat=frozen_lambda_feat,
            frozen_evidence_bonus=frozen_evidence_bonus,
            frozen_top_m_evidence=frozen_top_m_evidence,
        )
        cli_sfpw = (
            float(base_args.selector_feature_positive_weight)
            if _cli_explicit("selector_feature_positive_weight")
            else None
        )
        best_selector_feature_positive_weight = resolve_best_from_loaded_trials(
            stage2_trials,
            param_name="selector_feature_positive_weight",
            cli_value=cli_sfpw,
        )
        for trial in stage2_trials:
            trial.is_best = (
                trial.selector_feature_positive_weight == best_selector_feature_positive_weight
            )
        print(
            f"Resume: loaded {len(stage2_trials)} Stage2 trial(s); "
            f"fixed selector_feature_positive_weight={best_selector_feature_positive_weight}"
        )

    def _refresh() -> None:
        if dry_run:
            return
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
    if start_stage <= 1:
        for lambda_selector in STAGE1_LAMBDA_SELECTOR:
            trial = run_trial(
                base_args,
                stage=STAGE1_NAME,
                frozen_lambda_feat=frozen_lambda_feat,
                frozen_evidence_bonus=frozen_evidence_bonus,
                frozen_top_m_evidence=frozen_top_m_evidence,
                lambda_selector=lambda_selector,
                selector_feature_positive_weight=DEFAULT_SELECTOR_FEATURE_POSITIVE_WEIGHT,
                lambda_prefix_feature=DEFAULT_LAMBDA_PREFIX_FEATURE,
                dry_run=dry_run,
                split_index=split_index,
            )
            stage1_trials.append(trial)
            _refresh()

        best_stage1 = mark_stage_best(stage1_trials)
        best_lambda_selector = best_stage1.lambda_selector
        print(f"Stage 1 best: lambda_selector={best_lambda_selector} (tag={best_stage1.tag})")
        _refresh()

    # Stage 2: 固定最优 lambda_selector，搜索 selector_feature_positive_weight
    if start_stage <= 2:
        stage2_trials = []
        for selector_feature_positive_weight in STAGE2_SELECTOR_FEATURE_POSITIVE_WEIGHT:
            trial = run_trial(
                base_args,
                stage=STAGE2_NAME,
                frozen_lambda_feat=frozen_lambda_feat,
                frozen_evidence_bonus=frozen_evidence_bonus,
                frozen_top_m_evidence=frozen_top_m_evidence,
                lambda_selector=best_lambda_selector,
                selector_feature_positive_weight=selector_feature_positive_weight,
                lambda_prefix_feature=DEFAULT_LAMBDA_PREFIX_FEATURE,
                dry_run=dry_run,
                split_index=split_index,
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
    if start_stage <= 3:
        stage3_trials = []
        for lambda_prefix_feature in STAGE3_LAMBDA_PREFIX_FEATURE:
            trial = run_trial(
                base_args,
                stage=STAGE3_NAME,
                frozen_lambda_feat=frozen_lambda_feat,
                frozen_evidence_bonus=frozen_evidence_bonus,
                frozen_top_m_evidence=frozen_top_m_evidence,
                lambda_selector=best_lambda_selector,
                selector_feature_positive_weight=best_selector_feature_positive_weight,
                lambda_prefix_feature=lambda_prefix_feature,
                dry_run=dry_run,
                split_index=split_index,
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
