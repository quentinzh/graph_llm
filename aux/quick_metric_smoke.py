#!/usr/bin/env python
"""Quick CPU smoke test for CIER-aligned eval metrics and tail-demand groups."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_llm.config import build_arg_parser, resolve_dataset_paths
from graph_llm.config.args import resolve_local_model_path, qwen3_4b_model_candidates
from graph_llm.dataload.dataloader import GraphDataset
from graph_llm.dataload.legacy_data import dataset_split
from graph_llm.train.trainer import (
    append_eval_metrics,
    build_dataset,
    get_tail_demand_eval_groups,
    output_path_with_group,
)


def run_quick_metric_smoke(args) -> Path:
    resolve_dataset_paths(args)
    args.model_path = resolve_local_model_path(
        args.model_path,
        candidates=qwen3_4b_model_candidates(),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    dataset = build_dataset(args, tokenizer)
    _train, _valid, test_df = dataset_split(dataset, args.fold, args)
    test_set = GraphDataset(test_df, "test")

    n = min(args.max_eval_samples, len(test_set))
    label = [row for row in test_df["text"].tolist()[:n]]
    predict = [ids[:] for ids in label]

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "quick_metric_smoke.log"
        output_base = str(Path(tmpdir) / "generate.dataset")
        eval_groups = get_tail_demand_eval_groups(test_set, tokenizer, args)
        all_indices = list(range(n))
        group_info = (
            f"quick_metric_smoke samples={n} | " + eval_groups.get("info", "")
        )
        append_eval_metrics(
            str(log_path),
            test_set,
            tokenizer,
            predict,
            label,
            output_path_with_group(output_base, "all"),
            indices=all_indices,
            group_name="all",
            group_info=group_info,
        )
        for group_name, group_indices in eval_groups["indices"].items():
            subset = [idx for idx in group_indices if idx < n]
            append_eval_metrics(
                str(log_path),
                test_set,
                tokenizer,
                predict,
                label,
                output_path_with_group(output_base, group_name),
                indices=subset,
                group_name=group_name,
                group_info=group_info,
            )

        dest = Path(args.log_dir) / args.dataset_name
        dest.mkdir(parents=True, exist_ok=True)
        out_log = dest / "quick_metric_smoke.log"
        out_log.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Wrote {out_log}")
        print(out_log.read_text(encoding="utf-8"))
        return out_log


def main():
    parser = build_arg_parser()
    parser.add_argument("--max_eval_samples", default=64, type=int)
    args = parser.parse_args()
    args.fold = str(getattr(args, "fold", args.split_indices)).split(",")[0]
    run_quick_metric_smoke(args)


if __name__ == "__main__":
    main()
