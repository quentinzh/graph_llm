"""第二轮工程优化 smoke test。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from graph_llm.dataload.dataloader import GraphCollater, GraphDataset, resolve_batch_graphs
from graph_llm.models.selector import EvidenceSelector, MagNetConv, get_graph_device_tensors
from graph_llm.models.token_graph import UserTokenGraph


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=True):
        ids = [min(10 + (ord(ch) % 5), 31) for ch in str(text)[:8]]
        if add_special_tokens:
            ids = [1] + ids
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return "word"


def test_lite_collate_and_resolve_batch_graphs():
    import pandas as pd

    df = pd.DataFrame(
        {
            "text": [[11, 12, 13]],
            "rating": [5],
            "raw_user": ["u1"],
            "raw_item": ["i1"],
            "keyword_words": ["good"],
        }
    )
    dataset = GraphDataset(df, "train")
    collate = GraphCollater(
        word=40,
        tokenizer=_DummyTokenizer(),
        profile_records={"u1": {"text": "profile"}},
        item_meta={"i1": {"title": "T", "description": "D"}},
        item_description_mode="none",
        split_name="train",
        materialize_graph_batch=False,
    )
    collate.bind_dataset_cache(dataset)
    batch = collate([dataset[0]])
    assert len(batch) == 13
    assert batch[6] == [0]

    class _GraphLookup:
        def get_graph(self, split_name, local_idx):
            return UserTokenGraph.empty()

    graphs, tensors = resolve_batch_graphs(_GraphLookup(), "train", batch[6])
    assert len(graphs) == 1
    assert "node_token_ids" in tensors


def test_graph_device_tensors_cached_once():
    graph = UserTokenGraph.empty()
    device = torch.device("cpu")
    t1 = get_graph_device_tensors(graph, device)
    t2 = get_graph_device_tensors(graph, device)
    assert t1["edge_index"] is t2["edge_index"]


def test_magnet_graph_id_cache():
    conv = MagNetConv(4, 4, q=0.15)
    import numpy as np

    graph = UserTokenGraph(
        node_token_ids=np.array([1, 2], dtype=np.int64),
        node_surfaces=["a", "b"],
        node_counts=np.array([1.0, 1.0], dtype=np.float32),
        node_doc_freq=np.array([1.0, 1.0], dtype=np.float32),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        edge_weight=np.array([1.0], dtype=np.float32),
        in_degree=np.array([0.0, 1.0], dtype=np.float32),
        out_degree=np.array([1.0, 0.0], dtype=np.float32),
    )
    x = torch.randn(2, 4)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_weight = torch.tensor([1.0])
    out1 = conv(x, edge_index, edge_weight, graph=graph)
    out2 = conv(x, edge_index, edge_weight, graph=graph)
    assert torch.allclose(out1, out2)


if __name__ == "__main__":
    test_lite_collate_and_resolve_batch_graphs()
    test_graph_device_tensors_cached_once()
    test_magnet_graph_id_cache()
    print("speedup round2 smoke: ok")
