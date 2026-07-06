#!/usr/bin/env python
"""Entry point for graph_llm."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    parser.set_defaults(lambda_ul=0.1)
    run(parser.parse_args())
