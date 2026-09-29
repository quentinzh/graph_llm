#!/usr/bin/env python
"""从已有 generate.all.dataset 重算 test 指标（含 R-4），无需 GPU 再跑生成。

用法示例（CPU，fold 1 全量 MoviesAndTV_corsa_filtered）::

  conda run --no-capture-output -n fair python aux/recompute_metrics_from_generate.py \\
      --dataset_name Amazon/MoviesAndTV_corsa_filtered \\
      --split_indices 1 \\
      --recompute_device cpu
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_llm.config import build_arg_parser, resolve_dataset_paths
from graph_llm.config.args import qwen3_4b_model_candidates, resolve_local_model_path
from graph_llm.dataload.dataloader import GraphDataset
from graph_llm.dataload.legacy_data import dataset_split, tokenizer_pad_id
from graph_llm.train.trainer import (
    append_eval_metrics,
    build_dataset,
    tokenizer_eos_ids,
)


def collect_test_labels_like_test_step(
    test_set,
    *,
    eval_batch_size: int,
    word: int,
    pad_token_id: int,
    eos_token_ids: tuple[int, ...],
) -> list[list[int]]:
    """按 test DataLoader（shuffle=False）的 batch 截断规则重建 gold label。"""
    eos_id = eos_token_ids[0] if eos_token_ids else pad_token_id
    labels: list[list[int]] = []
    n = len(test_set)
    batch_size = max(1, int(eval_batch_size))
    for start in range(0, n, batch_size):
        batch_rows = [test_set[i] for i in range(start, min(start + batch_size, n))]
        max_length = max(
            min(word, max(len(row["text"]), 1))
            for row in batch_rows
        )
        for row in batch_rows:
            ids = list(row["text"][:max_length])
            if len(ids) == 0:
                ids = [eos_id]
            pad_len = max_length - len(ids)
            labels.append(ids + [pad_token_id] * pad_len)
    return labels


def default_device_choice() -> str:
    """MacBook 默认 CPU；其他机器默认 GPU（本脚本重算指标仍可用 CPU）。"""
    return "cpu" if platform.system() == "Darwin" else "gpu"


def resolve_device_choice(device_choice: str) -> None:
    """本脚本只做指标重算，强制 CPU 即可；保留 gpu 选项仅为接口一致。"""
    choice = (device_choice or default_device_choice()).lower()
    if choice not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported --device={device_choice!r}; use cpu|gpu")
    # 指标计算不加载 LLM，无需设置 CUDA；gpu 选项表示「与训练环境一致」的占位，仍走 CPU。
    if choice == "gpu":
        print("Note: metric recompute runs on CPU only (no model forward).")


def recompute_metrics_from_generate(args) -> Path:
    resolve_dataset_paths(args)
    args.model_path = resolve_local_model_path(
        args.model_path,
        candidates=qwen3_4b_model_candidates(),
    )
    split_index = str(args.split_indices).split(",")[0].strip()
    if not split_index:
        raise ValueError("--split_indices must not be empty")

    output_dir = Path(args.output_dir) / args.dataset_name
    generate_path = (
        Path(args.generate_path).expanduser()
        if args.generate_path
        else output_dir / f"{split_index}generate.all.dataset"
    )
    if not generate_path.is_file():
        raise FileNotFoundError(f"generate cache not found: {generate_path}")

    pred_df = pd.read_pickle(generate_path)
    if "text" not in pred_df.columns:
        raise ValueError(f"unexpected generate format (missing 'text'): {generate_path}")
    predict = [list(row) for row in pred_df["text"].tolist()]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    dataset = build_dataset(args, tokenizer)
    _train, _valid, test_df = dataset_split(dataset, split_index, args)
    test_set = GraphDataset(test_df, "test")
    pad_id = tokenizer_pad_id(tokenizer)
    eos_ids = tokenizer_eos_ids(tokenizer)
    eval_batch_size = args.eval_batch_size or args.batch_size
    label = collect_test_labels_like_test_step(
        test_set,
        eval_batch_size=eval_batch_size,
        word=args.word,
        pad_token_id=pad_id,
        eos_token_ids=eos_ids,
    )

    if len(predict) != len(label):
        raise ValueError(
            f"predict/label length mismatch: {len(predict)} vs {len(label)} "
            f"(generate={generate_path})"
        )

    log_path = Path(args.log_dir) / args.dataset_name / args.log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"recompute_from_generate: {generate_path}\n")
        handle.write(f"split_index:{split_index}\n")
        handle.write(f"samples:{len(predict)}\n")

    all_indices = list(range(len(predict)))

    append_eval_metrics(
        str(log_path),
        test_set,
        tokenizer,
        predict,
        label,
        output_dir=None,
        indices=all_indices,
        group_name="all",
        group_info=None,
    )

    print(f"Wrote {log_path}")
    print(log_path.read_text(encoding="utf-8"))
    return log_path


def main():
    parser = build_arg_parser()
    parser.add_argument(
        "--recompute_device",
        choices=["cpu", "gpu"],
        default=default_device_choice(),
        help="接口保留（与训练 --devices 区分）；重算本身只在 CPU 上跑 tokenizer+指标",
    )
    parser.add_argument(
        "--generate_path",
        default="",
        type=str,
        help="预测 pickle 路径；默认 log/output/<dataset>/{fold}generate.all.dataset",
    )
    # 复用 build_arg_parser 的 --log_name，此处覆盖默认文件名
    parser.set_defaults(log_name="only_eval_fold1_rouge4.log")
    args = parser.parse_args()
    resolve_device_choice(args.recompute_device)
    recompute_metrics_from_generate(args)


if __name__ == "__main__":
    main()
