"""优化路径与旧逻辑的结果一致性 smoke test（1-2 batch 规模）。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from graph_llm.dataload.dataloader import GraphCollater, GraphDataset
from graph_llm.metrics.metrics import ids_clear
from graph_llm.models.model import GraphEvidenceCIER
from graph_llm.models.selector import EvidenceSelector, MagNetConv


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=True):
        ids = [min(10 + (ord(ch) % 5), 31) for ch in str(text)[:8]]
        if add_special_tokens:
            ids = [1] + ids
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        if not ids:
            return ""
        return "word"


def test_collate_precompute_matches_dynamic():
    import pandas as pd

    df = pd.DataFrame(
        {
            "text": [[11, 12, 13], [14, 15]],
            "rating": [5, 4],
            "raw_user": ["u1", "u2"],
            "raw_item": ["i1", "i2"],
            "keyword_words": ["good", "nice"],
        }
    )
    dataset = GraphDataset(df, "train")
    tokenizer = _DummyTokenizer()
    collate = GraphCollater(
        word=40,
        tokenizer=tokenizer,
        profile_records={"u1": {"text": "profile one"}, "u2": {"text": "profile two"}},
        item_meta={"i1": {"title": "T1", "description": "D1"}, "i2": {"title": "T2", "description": "D2"}},
        item_description_mode="none",
        split_name="train",
        materialize_graph_batch=True,
    )
    rows = [dataset[i] for i in range(len(dataset))]
    dynamic_batch = collate(rows)

    collate.bind_dataset_cache(dataset)
    cached_batch = collate(rows)

    tensor_fields = (0, 1, 2, 3, 4, 5, 11, 12)
    for field_idx in tensor_fields:
        assert torch.equal(dynamic_batch[field_idx], cached_batch[field_idx])
    for key in dynamic_batch[6]:
        assert torch.equal(dynamic_batch[6][key], cached_batch[6][key])


def test_magnet_cache_matches_uncached():
    conv = MagNetConv(4, 4, q=0.15)
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_weight = torch.ones(3)
    out1 = conv(x, edge_index, edge_weight)
    out2 = conv(x, edge_index, edge_weight)
    assert torch.allclose(out1, out2, atol=1e-6, rtol=1e-5)


def test_evidence_bonus_plan_matches_loop():
    model = GraphEvidenceCIER(
        tokenizer=_DummyTokenizer(),
        vocab_size=32,
        evidence_selector=EvidenceSelector(embed_dim=4, hidden_dim=8, gnn_layers=1),
        evidence_bonus=0.25,
        eos_token_ids=(2,),
        pad_token_id=0,
    )
    logits = torch.zeros(2, 32)
    evidence_ids = torch.tensor([[5, 6, 7], [8, 9, 5]])
    evidence_mask = torch.tensor([[True, True, True], [True, True, True]])
    plan = model._prepare_evidence_bonus_plan(evidence_ids, evidence_mask)
    loop_logits = logits.clone()
    for batch_idx in range(2):
        for token_id in evidence_ids[batch_idx].tolist():
            if model._evidence_control_allowed(int(token_id)):
                loop_logits[batch_idx, int(token_id)] += model.evidence_bonus
    plan_logits = model._apply_evidence_bonus_plan(logits.clone(), plan)
    assert torch.allclose(loop_logits, plan_logits)


def test_greedy_early_stop_ids_clear_stable():
    """全 batch 结束后提前停止时，ids_clear 结果应与逐步填满 pad 一致。"""
    full = [11, 12, 2, 0, 0]
    early = [11, 12, 2, 2, 2]
    assert ids_clear(full, pad_token_id=0, eos_token_ids=(2,)) == ids_clear(
        early, pad_token_id=0, eos_token_ids=(2,)
    )


if __name__ == "__main__":
    test_collate_precompute_matches_dynamic()
    test_magnet_cache_matches_uncached()
    test_evidence_bonus_plan_matches_loop()
    test_greedy_early_stop_ids_clear_stable()
    print("speedup equivalence smoke: ok")
