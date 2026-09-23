#!/usr/bin/env python
"""graph_llm Semantic ID 推荐入口。"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", os.environ.get("GRAPH_HF_ENDPOINT", "https://hf-mirror.com"))

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from graph_llm.config import build_arg_parser, resolve_dataset_paths
from graph_llm.train.explain_trainer import run_explain
from graph_llm.train.recommend_trainer import run_recommend


def seed_everything(seed: int = 5254) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    resolve_dataset_paths(args)
    if getattr(args, "mode", "recommend") == "explain":
        run_explain(args)
    else:
        run_recommend(args)
