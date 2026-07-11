#!/usr/bin/env python
"""Run beam + rerank decoding ablations on a fixed training checkpoint."""

from __future__ import annotations

import copy
import json
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


# 本轮固定训练侧超参，不在此脚本中搜索。
FIXED_LAMBDA_FEAT = 1e-2
FIXED_EVIDENCE_BONUS = 0.0
FIXED_TOP_M_EVIDENCE = 5
DEFAULT_SOURCE_SEARCH = "bonus_search"
DEFAULT_SOURCE_TAG = "B0_lfeat1e-2_eb0_top5"

# 只扫解码策略：greedy / beam / beam+rerank。
DECODE_EXPERIMENTS = [
    {
        "name": "G0",
        "decode_strategy": "greedy",
        "num_beams": 4,
        "num_return_sequences": 4,
        "use_rerank": False,
    },
    {
        "name": "B4",
        "decode_strategy": "beam",
        "num_beams": 4,
        "num_return_sequences": 4,
        "use_rerank": False,
    },
    {
        "name": "B4R",
        "decode_strategy": "beam",
        "num_beams": 4,
        "num_return_sequences": 4,
        "use_rerank": True,
    },
    {
        "name": "B8R",
        "decode_strategy": "beam",
        "num_beams": 8,
        "num_return_sequences": 8,
        "use_rerank": True,
    },
]


def parse_experiment_names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("--experiments must contain at least one experiment name")
    if len(names) == 1 and names[0].lower() == "all":
        return [str(exp["name"]) for exp in DECODE_EXPERIMENTS]
    known = {str(exp["name"]) for exp in DECODE_EXPERIMENTS}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(
            f"Unknown experiment name(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(known))}, all"
        )
    return names


def parse_source_tags(value: str) -> list[str]:
    tags = [part.strip() for part in value.split(",") if part.strip()]
    if not tags:
        raise ValueError("--source_tags must contain at least one checkpoint tag")
    return tags


def selected_experiments(names: list[str]) -> list[dict[str, str | int | bool]]:
    selected = set(names)
    return [exp for exp in DECODE_EXPERIMENTS if str(exp["name"]) in selected]


def _float_equal(lhs: float, rhs: float, tol: float = 1e-12) -> bool:
    return abs(float(lhs) - float(rhs)) <= tol


def validate_graph_config(ckpt_prefix: Path) -> None:
    """确认 checkpoint 的训练超参与本轮固定设置一致。"""
    config_path = ckpt_prefix.with_name(ckpt_prefix.name + "graph_config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"graph_config.json not found: {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    checks = {
        "lambda_feat": FIXED_LAMBDA_FEAT,
        "evidence_bonus": FIXED_EVIDENCE_BONUS,
        "top_m_evidence": FIXED_TOP_M_EVIDENCE,
    }
    mismatches = []
    for key, expected in checks.items():
        actual = config.get(key)
        if key == "top_m_evidence":
            matched = int(actual) == int(expected)
        else:
            matched = _float_equal(actual, expected)
        if not matched:
            mismatches.append(f"{key}: expected={expected}, actual={actual}")
    if mismatches:
        raise ValueError(
            "Checkpoint hyperparameters do not match fixed beam_rerank settings: "
            + "; ".join(mismatches)
        )


def build_experiment_args(base_args, source_search: str, source_tag: str, decode_exp: dict):
    args = copy.copy(base_args)
    combo_tag = f"{source_tag}__{decode_exp['name']}"

    args.only_eval = True
    args.lambda_feat = FIXED_LAMBDA_FEAT
    args.evidence_bonus = FIXED_EVIDENCE_BONUS
    args.top_m_evidence = FIXED_TOP_M_EVIDENCE
    args.decode_strategy = str(decode_exp["decode_strategy"])
    args.num_beams = int(decode_exp["num_beams"])
    args.num_return_sequences = int(decode_exp["num_return_sequences"])
    args.use_rerank = bool(decode_exp["use_rerank"])

    # 权重从 source_search 读取，日志写到 beam_rerank_search 下，避免覆盖原实验。
    args.ckpt_dir = str(Path(base_args.ckpt_dir) / source_search / source_tag)
    args.log_dir = str(Path(base_args.log_dir) / "beam_rerank_search" / combo_tag)
    args.output_dir = str(Path(base_args.output_dir) / "beam_rerank_search" / combo_tag)
    args.log_name = "graph_profile.log"
    return args


def print_experiment_summary(args, source_search: str, source_tag: str, decode_exp: dict) -> None:
    combo_tag = f"{source_tag}__{decode_exp['name']}"
    print("=" * 88)
    print(f"Decode experiment {decode_exp['name']} ({combo_tag})")
    print(f"source_search: {source_search}")
    print(f"source_tag: {source_tag}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"split_indices: {args.split_indices}")
    print(f"lambda_feat: {args.lambda_feat}")
    print(f"evidence_bonus: {args.evidence_bonus}")
    print(f"top_m_evidence: {args.top_m_evidence}")
    print(f"decode_strategy: {args.decode_strategy}")
    print(f"num_beams: {args.num_beams}")
    print(f"num_return_sequences: {args.num_return_sequences}")
    print(f"use_rerank: {args.use_rerank}")
    print(f"only_eval: {args.only_eval}")
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
        "--source_search",
        default=DEFAULT_SOURCE_SEARCH,
        help="Checkpoint search directory under ckpt_dir, e.g. bonus_search.",
    )
    parser.add_argument(
        "--source_tags",
        default=DEFAULT_SOURCE_TAG,
        help="Comma-separated source checkpoint tags under source_search.",
    )
    parser.add_argument(
        "--experiments",
        default="G0,B4,B4R,B8R",
        help="Comma-separated decode experiments: G0,B4,B4R,B8R, or all.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print experiment settings without launching evaluation.",
    )
    base_args = parser.parse_args()

    try:
        experiment_names = parse_experiment_names(base_args.experiments)
        source_tags = parse_source_tags(base_args.source_tags)
    except ValueError as exc:
        parser.error(str(exc))

    source_search = str(base_args.source_search)
    dry_run = bool(getattr(base_args, "dry_run", False))
    for attr in ("experiments", "dry_run", "source_search", "source_tags"):
        if hasattr(base_args, attr):
            delattr(base_args, attr)

    decode_experiments = selected_experiments(experiment_names)
    for source_tag in source_tags:
        for split_index in str(base_args.split_indices).split(","):
            split_index = split_index.strip()
            ckpt_prefix = Path(base_args.ckpt_dir) / source_search / source_tag / base_args.dataset_name / split_index
            validate_graph_config(ckpt_prefix)

        for decode_exp in decode_experiments:
            args = build_experiment_args(base_args, source_search, source_tag, decode_exp)
            print_experiment_summary(args, source_search, source_tag, decode_exp)
            if dry_run:
                continue
            from graph_llm.train import run

            run(args)


if __name__ == "__main__":
    main()
