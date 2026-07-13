#!/usr/bin/env python
"""Entry point for graph_llm."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# 与 bonus_search.py 保持一致：需要下载模型时优先使用服务器配置的镜像。
os.environ.setdefault("HF_ENDPOINT", os.environ.get("GRAPH_HF_ENDPOINT", "https://hf-mirror.com"))

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_llm.config import build_arg_parser
from graph_llm.train import run

if __name__ == "__main__":
    parser = build_arg_parser()
    # 默认配置与 bonus_search.py 中 B1 完全对齐，可通过命令行参数覆盖（如 --devices 0）。
    parser.set_defaults(
        lambda_ul=0.1,
        lambda_feat=1e-2,
        evidence_bonus=0.5,
        top_m_evidence=5,
        devices="1",
        model_path=str(PACKAGE_ROOT / "pretrain_llm" / "qwen3-4b"),
        embedding_model_path=str(PACKAGE_ROOT / "pretrain_llm" / "qwen3-embedding-0.6b"),
        profile_dir=str(PACKAGE_ROOT / "data" / "profiles"),
        data_dir=str(PACKAGE_ROOT / "data"),
    )
    run(parser.parse_args())
