"""Semantic ID 推荐路径单元测试（不跑完整训练）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from graph_llm.config import build_arg_parser
from graph_llm.config.datasets import resolve_dataset_paths
from graph_llm.dataload.sequential_data import assign_calib_split, load_sequential_dataset
from graph_llm.dataload.semantic_id import _build_rating_pop
from graph_llm.dataload.sequential_data import InteractionRecord
from graph_llm.metrics.rec_ranking import evaluate_ranking
from graph_llm.models.item_search import recovery_rate
from graph_llm.models.sid_recommender import SIDRecommender


def test_sequential_load_and_calib():
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset_name", "Instruments", "--data_dir", str(REPO / "data")])
    resolve_dataset_paths(args)
    bundle = load_sequential_dataset(Path(args.data_dir), args.dataset_name)
    assign_calib_split(bundle, 0.2)
    assert len(bundle.calib_ids) > 0
    assert len(bundle.rec_train_ids) > 0
    assert len(bundle.val_sample_ids) == bundle.num_users or len(bundle.val_sample_ids) > 0


def test_rating_bucket_isolation():
    calib = [
        InteractionRecord("u1", "i1", 0, 0, 5.0, 1, "", "", "", 0),
        InteractionRecord("u2", "i2", 1, 1, 1.0, 2, "", "", "", 1),
    ]
    items = ["i1", "i2", "i3"]
    smooth, rating_code, pop_code, edges, gmean, pop_edges, distinct = _build_rating_pop(
        calib, items, rating_min=1.0, rating_max=5.0, alpha=10.0
    )
    assert rating_code["i3"] == 0
    assert pop_code["i3"] == 0


def test_hr_ndcg_metrics():
    metrics = evaluate_ranking([1, 2], [[1, 3, 4], [5, 2, 1]], ks=(5, 10))
    assert metrics["HR@5"] == 1.0
    assert metrics["NDCG@5"] > 0


def test_item_rec_loss_backward():
    model = SIDRecommender(embed_dim=32, hidden_dim=16, gnn_layers=1, text_classes=8, text_positions=4)
    user_repr = torch.randn(2, 16, requires_grad=True)
    item_codes = torch.randint(0, 8, (20, 6))
    item_codes[:, 4] = torch.randint(0, 9, (20,))
    item_codes[:, 5] = torch.randint(0, 9, (20,))
    target = torch.tensor([1, 3], dtype=torch.long)
    loss = model.item_rec_loss(
        user_repr,
        target,
        item_codes,
        lambda_rating=0.2,
        lambda_pop=0.1,
        text_sid_length=4,
        use_rating_sid=True,
        use_popularity_sid=True,
        temperature=4.0,
    )
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_sid_loss_backward():
    model = SIDRecommender(embed_dim=32, hidden_dim=16, gnn_layers=1, text_classes=8)
    user_repr = torch.randn(2, 16, requires_grad=True)
    logits = model.forward_sid_logits(user_repr)
    targets = {
        "text": torch.randint(0, 8, (2, 4)),
        "rating": torch.randint(0, 9, (2,)),
        "pop": torch.randint(0, 9, (2,)),
    }
    loss = model.sid_loss(logits, targets, lambda_rating=0.2, lambda_pop=0.1)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_search_recovery_identity():
    items = [1, 2, 3, 4, 5]
    assert recovery_rate(items, items, 5) == 1.0
