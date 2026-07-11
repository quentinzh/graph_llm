#!/usr/bin/env python
"""Run low-cost evidence_bonus ablations for graph_llm."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", os.environ.get("GRAPH_HF_ENDPOINT", "https://hf-mirror.com"))

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_llm.config import build_arg_parser


EXPERIMENTS = [
    {
        "name": "B0",
        "lambda_feat": 1e-2,
        "evidence_bonus": 0.0,
        "top_m_evidence": 5,
    },
    {
        "name": "B1",
        "lambda_feat": 1e-2,
        "evidence_bonus": 0.5,
        "top_m_evidence": 5,
    },
    {
        "name": "B2",
        "lambda_feat": 1e-2,
        "evidence_bonus": 1.0,
        "top_m_evidence": 5,
    },
    {
        "name": "B3",
        "lambda_feat": 1e-2,
        "evidence_bonus": 2.0,
        "top_m_evidence": 5,
    },
]


def _format_float(value: float) -> str:
    if 0 < abs(value) <= 1e-2:
        text = f"{value:.0e}"
        mantissa, exponent = text.split("e", 1)
        sign = "-" if exponent.startswith("-") else ""
        digits = exponent.lstrip("+-").lstrip("0") or "0"
        return f"{mantissa}e{sign}{digits}"
    return f"{value:g}"


def experiment_tag(exp: dict[str, float | int | str]) -> str:
    return (
        f"{exp['name']}_"
        f"lfeat{_format_float(float(exp['lambda_feat']))}_"
        f"eb{_format_float(float(exp['evidence_bonus']))}_"
        f"top{int(exp['top_m_evidence'])}"
    )


def parse_experiment_names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("--experiments must contain at least one experiment name")
    if len(names) == 1 and names[0].lower() == "all":
        return [str(exp["name"]) for exp in EXPERIMENTS]
    known = {str(exp["name"]) for exp in EXPERIMENTS}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(
            f"Unknown experiment name(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(known))}, all"
        )
    return names


def selected_experiments(names: list[str]) -> list[dict[str, float | int | str]]:
    selected = set(names)
    return [exp for exp in EXPERIMENTS if str(exp["name"]) in selected]


def build_experiment_args(base_args, exp: dict[str, float | int | str]):
    args = copy.copy(base_args)
    tag = experiment_tag(exp)

    args.lambda_feat = float(exp["lambda_feat"])
    args.evidence_bonus = float(exp["evidence_bonus"])
    args.top_m_evidence = int(exp["top_m_evidence"])

    args.ckpt_dir = str(Path(base_args.ckpt_dir) / "bonus_search" / tag)
    args.log_dir = str(Path(base_args.log_dir) / "bonus_search" / tag)
    args.output_dir = str(Path(base_args.output_dir) / "bonus_search" / tag)
    args.log_name = "graph_profile.log"
    return args


def print_experiment_summary(args, exp: dict[str, float | int | str]) -> None:
    tag = experiment_tag(exp)
    print("=" * 88)
    print(f"Experiment {exp['name']} ({tag})")
    print(f"dataset_name: {args.dataset_name}")
    print(f"split_indices: {args.split_indices}")
    print(f"lambda_feat: {args.lambda_feat}")
    print(f"evidence_bonus: {args.evidence_bonus}")
    print(f"top_m_evidence: {args.top_m_evidence}")
    print(f"lambda_ul: {args.lambda_ul}")
    print(f"devices: {args.devices}")
    print(f"ckpt_dir: {args.ckpt_dir}")
    print(f"log_dir: {args.log_dir}")
    print(f"output_dir: {args.output_dir}")
    print("=" * 88)


def main() -> None:
    parser = build_arg_parser()
    parser.set_defaults(
        lambda_ul=0.1,
        devices="1",
        model_path=str(PACKAGE_ROOT / "pretrain_llm" / "qwen3-4b"),
        embedding_model_path=str(PACKAGE_ROOT / "pretrain_llm" / "qwen3-embedding-0.6b"),
        profile_dir=str(PACKAGE_ROOT / "data" / "profiles"),
        data_dir=str(PACKAGE_ROOT / "data"),
    )
    parser.add_argument(
        "--experiments",
        default="B0,B1,B2,B3",
        help="Comma-separated experiment names to run: B0,B1,B2,B3, or all.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print experiment settings without launching training.",
    )
    base_args = parser.parse_args()
    try:
        names = parse_experiment_names(base_args.experiments)
    except ValueError as exc:
        parser.error(str(exc))

    if hasattr(base_args, "experiments"):
        delattr(base_args, "experiments")
    dry_run = bool(getattr(base_args, "dry_run", False))
    if hasattr(base_args, "dry_run"):
        delattr(base_args, "dry_run")

    experiments = selected_experiments(names)
    for exp in experiments:
        args = build_experiment_args(base_args, exp)
        print_experiment_summary(args, exp)
        if dry_run:
            continue
        from graph_llm.train import run

        run(args)


if __name__ == "__main__":
    main()
