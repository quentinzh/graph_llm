"""端到端 recommend smoke：2 batch 训练 + 少量验证（真实 RoBERTa）。"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from graph_llm.config import build_arg_parser
from graph_llm.train.recommend_trainer import run_recommend


def test_recommend_smoke_two_batches():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--dataset_name",
            "Instruments",
            "--data_dir",
            str(REPO / "data"),
            "--epochs",
            "1",
            "--batch_size",
            "2",
            "--eval_batch_size",
            "2",
            "--max_train_batches",
            "2",
            "--max_eval_batches",
            "1",
            "--devices",
            "1",
            "--smoke_codebook_size",
            "16",
        ]
    )
    out = run_recommend(args)
    assert "HR@10" in out["val"]
    assert "NDCG@10" in out["val"]
