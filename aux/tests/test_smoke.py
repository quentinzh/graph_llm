"""CPU-safe smoke tests for graph evidence selector."""

from __future__ import annotations

import argparse
import platform
import sys
import tempfile
from importlib.util import find_spec
from types import ModuleType, SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

if find_spec("peft") is None:
    peft_stub = ModuleType("peft")
    peft_stub.LoraConfig = object
    peft_stub.TaskType = SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
    peft_stub.get_peft_model = lambda model, _config: model
    sys.modules["peft"] = peft_stub

from graph_llm.aux.derive_profiles_from_parent import derive_profiles
from graph_llm.aux.prompt_utils import (
    build_generation_prompt_text,
    build_generation_prompt_batch,
    item_meta_from_row,
    tokenize_text_list,
)
from graph_llm.dataload.dataloader import GraphCollater
from graph_llm.dataload.review_memory import ReviewMemoryBank, mark_review_history
from graph_llm.dataload.tail_stats import TailTokenStats, build_tail_token_stats
from graph_llm.metrics.metrics import bleu_score, rouge_score
from graph_llm.models.selector import EvidenceSelector, pad_token_matrix
from graph_llm.models.token_graph import (
    ReviewRecord,
    UserTokenGraph,
    batch_graphs,
    build_sample_token_graph,
)
from graph_llm.models.model import GraphEvidenceCIER
from graph_llm.config import (
    build_arg_parser,
    qwen3_4b_model_candidates,
    resolve_local_model_path,
    snapshot_training_args,
)
from graph_llm.train.trainer import (
    build_llm_max_memory,
    build_oom_plans,
    build_run_oom_plans,
    compute_batch_selector_tensors,
    default_preferred_device_id,
    flash_attn_available,
    is_explicit_single_device,
    parse_device_ids,
    profile_cache_path,
    profile_dataset_name_candidates,
    preflight_profile_cache_files,
    resolve_attn_implementation,
    resolve_devices_string,
    resolve_embedding_device,
    resolve_llm_device_map_mode,
    resolve_training_devices,
)




def _make_model(tokenizer, **kwargs):
    defaults = dict(
        tokenizer=tokenizer,
        vocab_size=32,
        evidence_selector=EvidenceSelector(embed_dim=4, hidden_dim=8, gnn_layers=1),
        lambda_feat=0.0,
        evidence_bonus=0.0,
        pad_token_id=0,
        eos_token_ids=(2,),
        special_token_ids=(0, 1, 2),
    )
    defaults.update(kwargs)
    if "evidence_selector" not in kwargs and kwargs.get("_selector") is not None:
        defaults["evidence_selector"] = kwargs["_selector"]
    defaults.pop("_selector", None)
    return GraphEvidenceCIER(**defaults)


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        mapping = {
            "great movie": [10, 11, 12],
            " great": [18],
            "great": [10],
            "good story": [11, 13, 14],
            "target item leak": [99, 100],
            "Relevant keywords: great, movie\n": [20, 21, 22],
            'The explanation of Inception for user_a is "': [30, 31],
            "Current item information:\nTitle: Inception\nDescription: A dream heist film\n\n": [40, 41],
            " television": [21],
            "television": [19, 20],
        }
        if text in mapping:
            return {"input_ids": mapping[text]}
        for key, ids in mapping.items():
            if key in text:
                return {"input_ids": ids}
        return {"input_ids": [len(text) % 17 + 3]}

    def decode(self, ids, skip_special_tokens=True):
        inv = {
            10: "great", 11: "movie", 12: ".", 13: "story", 14: ".",
            15: "the", 16: "1", 17: "j", 18: " great",
            19: "te", 20: "levision", 21: " television",
            99: "target", 100: "leak",
        }
        return " ".join(inv.get(i, str(i)) for i in ids)


class FixedLogitLM(torch.nn.Module):
    def __init__(self, vocab_size=32, hidden_size=8, logits=None):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.register_buffer("fixed_logits", logits.float())

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds=None, attention_mask=None, use_cache=False, logits_to_keep=None, **_kwargs):
        batch_size = inputs_embeds.shape[0]
        keep = logits_to_keep or inputs_embeds.shape[1]
        logits = self.fixed_logits[:keep].unsqueeze(0).expand(batch_size, -1, -1).clone()
        return {"logits": logits}


class FixedFullLogitLM(torch.nn.Module):
    def __init__(self, vocab_size=32, hidden_size=8, logits=None):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.register_buffer("fixed_logits", logits.float())

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds=None, attention_mask=None, use_cache=False, logits_to_keep=None, **_kwargs):
        if logits_to_keep is not None:
            raise TypeError("logits_to_keep is not supported by this test model")
        batch_size, seq_len = inputs_embeds.shape[:2]
        logits = self.fixed_logits[:seq_len].unsqueeze(0).expand(batch_size, -1, -1).clone()
        return {"logits": logits}


class TrainableTinyLM(torch.nn.Module):
    """Small causal-LM stand-in used by the two-batch CPU training smoke test."""

    def __init__(self, vocab_size=32, hidden_size=8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        use_cache=False,
        logits_to_keep=None,
        output_hidden_states=False,
        **_kwargs,
    ):
        # 补一个末位置，使 GraphEvidenceCIER 能对每个 target token 计算 NLL。
        last_state = inputs_embeds[:, -1:, :]
        hidden = torch.cat([inputs_embeds, last_state], dim=1)
        output = {"logits": self.lm_head(hidden)}
        if output_hidden_states:
            output["hidden_states"] = (hidden,)
        return output


def test_directed_edges_and_leakage():
    skip = {0, 1, 2}
    history = [
        ReviewRecord(1, "u1", "item_a", (10, 11, 12)),
        ReviewRecord(2, "u1", "item_b", (11, 13, 14)),
        ReviewRecord(3, "u1", "target_item", (99, 100)),
    ]
    graph = build_sample_token_graph(
        history,
        exclude_row_key=2,
        target_raw_item="target_item",
        skip_token_ids=skip,
        max_nodes=32,
    )
    assert graph.num_nodes > 0
    assert 99 not in graph.node_token_ids.tolist()
    assert 100 not in graph.node_token_ids.tolist()
    if graph.edge_index.size > 0:
        src, dst = graph.edge_index
        assert np.all(src != dst)
        node_map = {int(t): i for i, t in enumerate(graph.node_token_ids.tolist())}
        if 10 in node_map and 11 in node_map:
            pairs = set(zip(src.tolist(), dst.tolist()))
            assert (node_map[10], node_map[11]) in pairs


def test_selector_shapes():
    graph = UserTokenGraph(
        node_token_ids=np.array([10, 11, 13], dtype=np.int64),
        node_surfaces=["great", "movie", "story"],
        node_counts=np.array([5.0, 4.0, 2.0], dtype=np.float32),
        node_doc_freq=np.array([3.0, 3.0, 1.0], dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        edge_weight=np.array([2.0, 1.0], dtype=np.float32),
        in_degree=np.array([0.0, 1.0, 1.0], dtype=np.float32),
        out_degree=np.array([1.0, 1.0, 0.0], dtype=np.float32),
    )
    selector = EvidenceSelector(embed_dim=8, hidden_dim=16, gnn_layers=2)
    assert selector.input_proj[0].in_features == 8
    node_emb = torch.randn(graph.num_nodes, 8)
    item_emb = torch.randn(8)
    scores = selector.forward_single(graph, node_emb, item_emb)
    assert scores.shape == (graph.num_nodes,)
    selected = selector.select_evidence(scores, graph, top_m=2)
    assert selected.numel() == 2


def test_tail_stats_and_selector_bce():
    tokenizer = DummyTokenizer()
    stats = build_tail_token_stats(
        [[10, 11, 2], [10, 13, 2], [10, 11, 2]],
        tokenizer=tokenizer,
        ignored_token_ids={0, 1, 2},
        weight_min=0.5,
        weight_max=2.0,
    )
    assert stats.document_frequency(10) == 3
    assert stats.tail_weight(13) >= stats.tail_weight(10)

    graph = UserTokenGraph(
        node_token_ids=np.array([10, 11, 13], dtype=np.int64),
        node_surfaces=["great", "movie", "story"],
        node_counts=np.array([3.0, 2.0, 1.0], dtype=np.float32),
        node_doc_freq=np.array([3.0, 2.0, 1.0], dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        edge_weight=np.array([1.0, 1.0], dtype=np.float32),
        in_degree=np.array([0.0, 1.0, 1.0], dtype=np.float32),
        out_degree=np.array([1.0, 1.0, 0.0], dtype=np.float32),
    )
    selector = EvidenceSelector(embed_dim=4, hidden_dim=8, gnn_layers=1)
    scores = selector.forward_single(graph, torch.randn(3, 4), torch.randn(4))
    loss = selector.sampled_bce_loss(scores, graph, {13}, stats)
    feature_weighted_loss = selector.sampled_bce_loss(
        scores,
        graph,
        {13},
        stats,
        feature_token_ids={13},
        feature_positive_weight=3.0,
    )
    loss.backward()
    assert loss.item() > 0.0
    assert not torch.isclose(loss, feature_weighted_loss)
    assert any(param.grad is not None for param in selector.parameters())


def test_prompt_text_includes_title_and_user_id():
    tokenizer = DummyTokenizer()
    generation_text = build_generation_prompt_text("Inception", "user_a")
    assert generation_text == 'The explanation of Inception for user_a is "'

    title, description, item_text = item_meta_from_row(
        "item123",
        {"item123": {"title": "Inception", "description": "A dream heist film"}},
    )
    assert title == "Inception"
    assert "Title: Inception" in item_text

    gen_ids, gen_mask = build_generation_prompt_batch(
        ["Inception"], ["user_a"], tokenizer, pad_token_id=0, max_tokens=32,
    )
    assert gen_ids.shape[0] == 1
    assert gen_mask.sum().item() > 0



def test_prompt_length_accounts_for_new_segments():
    tokenizer = DummyTokenizer()
    model = _make_model(tokenizer)
    profile_ids = torch.tensor([[1, 2, 3]])
    target_item_ids = torch.tensor([[4, 5]])
    generation_prompt_ids = torch.tensor([[8, 9, 10]])
    review_prefixes = torch.randn(1, 4, 8)
    prompt_len = model._prompt_length(
        profile_ids=profile_ids,
        review_prefixes=review_prefixes,
        target_item_ids=target_item_ids,
        generation_prompt_ids=generation_prompt_ids,
    )
    assert prompt_len == 3 + 4 + 2 + 3


def test_generation_controls_filter_evidence_and_block_repeats():
    tokenizer = DummyTokenizer()
    model = _make_model(tokenizer, evidence_bonus=0.5, max_consecutive_token_repeat=2)
    logits = torch.zeros(1, 32)
    evidence_token_ids, evidence_token_mask = pad_token_matrix([[10, 15, 16, 17]], pad_value=-1)
    adjusted = model._apply_evidence_bonus(logits.clone(), evidence_token_ids, evidence_token_mask)
    assert adjusted[0, 10].item() > 0.0
    assert adjusted[0, 15].item() == 0.0
    assert adjusted[0, 16].item() == 0.0
    assert adjusted[0, 17].item() == 0.0

    repeat_logits = torch.zeros(1, 32)
    repeat_logits[0, 10] = 3.0
    blocked = model._apply_repetition_controls(repeat_logits, torch.tensor([[10, 10]]))
    assert blocked[0, 10].item() < -1000.0


def test_train_step_uses_adjusted_logits_nll():
    tokenizer = DummyTokenizer()
    selector = EvidenceSelector(embed_dim=4, hidden_dim=8, gnn_layers=1)
    fixed_logits = torch.zeros(3, 32)
    fixed_logits[0, 10] = 1.5
    fixed_logits[1, 11] = 0.5

    model = _make_model(
        tokenizer,
        evidence_selector=selector,
        evidence_bonus=0.0,
    )
    model.model = FixedLogitLM(logits=fixed_logits)

    input_ids = torch.tensor([[10, 11]])
    evidence_token_ids, evidence_token_mask = pad_token_matrix([[10]], pad_value=-1)
    loss, nll, feat, prefix_feat = model.train_step(
        input_ids,
        evidence_token_ids=evidence_token_ids,
        evidence_token_mask=evidence_token_mask,
    )

    manual_log_probs = torch.log_softmax(fixed_logits[:2], dim=-1)
    manual_nll = -torch.stack([manual_log_probs[0, 10], manual_log_probs[1, 11]]).mean()
    assert torch.allclose(nll, manual_nll)
    assert torch.allclose(loss, manual_nll)
    assert feat.item() == 0.0
    assert prefix_feat.item() == 0.0


def test_train_step_uses_tail_weighted_sft():
    tokenizer = DummyTokenizer()
    fixed_logits = torch.zeros(3, 32)
    fixed_logits[0, 10] = 1.5
    fixed_logits[1, 11] = 0.5
    model = _make_model(tokenizer, evidence_bonus=0.0)
    model.model = FixedLogitLM(logits=fixed_logits)

    input_ids = torch.tensor([[10, 11]])
    tail_weights = torch.tensor([[2.0, 0.5]])
    loss, nll, feat, prefix_feat = model.train_step(
        input_ids,
        tail_position_weights=tail_weights,
    )

    log_probs = torch.log_softmax(fixed_logits[:2], dim=-1)
    token_nll = torch.stack([-log_probs[0, 10], -log_probs[1, 11]])
    expected = (token_nll * torch.tensor([2.0, 0.5])).sum() / 2.5
    assert torch.allclose(nll, expected)
    assert torch.allclose(loss, expected)
    assert feat.item() == 0.0
    assert prefix_feat.item() == 0.0


def test_two_batch_tail_sft_training_smoke():
    """Run exactly two CPU batches to check the tail-weighted optimization path."""
    tokenizer = DummyTokenizer()
    model = _make_model(tokenizer, evidence_bonus=0.0)
    model.model = TrainableTinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batches = [
        (torch.tensor([[10, 11]]), torch.tensor([[2.0, 0.5]])),
        (torch.tensor([[11, 13]]), torch.tensor([[0.5, 2.0]])),
    ]
    for input_ids, tail_weights in batches:
        optimizer.zero_grad()
        loss, _nll, _feat, _prefix_feat = model.train_step(
            input_ids,
            tail_position_weights=tail_weights,
        )
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()


def resolve_review_prefix_smoke_device(requested="auto"):
    """为两批 prefix smoke test 选择设备，并保留显式 CPU/GPU 接口。"""
    requested = str(requested).strip().lower()
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda":
            if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
                raise RuntimeError(f"Requested smoke device is unavailable: {device}")
        return device
    # MacBook 默认 CPU；服务器优先 cuda:1，同时保留 cuda:0 回退。
    if platform.system() == "Darwin":
        return torch.device("cpu")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return torch.device("cuda:1")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def test_review_memory_excludes_current_training_target():
    history = mark_review_history(
        pd.DataFrame({
            "raw_user": ["u1", "u1", "u2"],
            "raw_item": ["i1", "i2", "i1"],
            "review_text": ["target", "user history", "item history"],
        }),
        "train",
    )
    bank = ReviewMemoryBank(
        embeddings=torch.eye(3, 4),
        raw_users=history["raw_user"].tolist(),
        raw_items=history["raw_item"].tolist(),
        source_splits=history["_review_source_split"].tolist(),
        local_indices=history["_review_local_idx"].tolist(),
        top_k_user=2,
        top_k_item=2,
    )
    review_tensors = bank.prepare_batch(
        [{
            "raw_user": "u1",
            "raw_item": "i1",
            "split_name": "train",
            "local_idx": 0,
        }],
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.device("cpu"),
    )
    # 当前 target 的 one-hot 向量为第 0 维；两条检索结果都不能包含它。
    assert review_tensors["user_review_embeddings"][0, :, 0].sum().item() == 0.0
    assert review_tensors["item_review_embeddings"][0, :, 0].sum().item() == 0.0
    assert review_tensors["user_review_mask"].sum().item() == 1
    assert review_tensors["item_review_mask"].sum().item() == 1


def run_two_batch_review_prefix_training_smoke(device="auto"):
    """以 batch_size=8 恰好跑两个 batch，检查 prefix 前向、反向和参数更新。"""
    device = resolve_review_prefix_smoke_device(device)
    tokenizer = DummyTokenizer()
    model = _make_model(
        tokenizer,
        hidden_size=8,
        review_embedding_dim=4,
        user_review_prefix_len=2,
        item_review_prefix_len=2,
        review_attention_heads=2,
        review_prefix_dropout=0.0,
        lambda_prefix_feature=0.1,
    )
    model.model = TrainableTinyLM()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch_size = 8
    # 列表长度固定为 2，确保 smoke test 不会意外跑完整 epoch。
    batches = [
        torch.randint(3, 20, (batch_size, 4), device=device),
        torch.randint(3, 20, (batch_size, 4), device=device),
    ]
    for input_ids in batches:
        input_ids[:, -1] = 2
        user_mask = torch.ones((batch_size, 3), dtype=torch.bool, device=device)
        item_mask = torch.ones((batch_size, 5), dtype=torch.bool, device=device)
        review_tensors = {
            "user_review_embeddings": torch.randn(batch_size, 3, 4, device=device),
            "user_review_mask": user_mask,
            "item_review_embeddings": torch.randn(batch_size, 5, 4, device=device),
            "item_review_mask": item_mask,
            "item_review_query": torch.randn(batch_size, 4, device=device),
            "user_review_query": torch.randn(batch_size, 4, device=device),
        }
        feature_weights = torch.zeros_like(input_ids, dtype=torch.float32)
        feature_weights[:, 1] = 2.0
        optimizer.zero_grad(set_to_none=True)
        loss, nll, feat, prefix_feat = model.train_step(
            input_ids,
            feature_position_weights=feature_weights,
            **review_tensors,
        )
        assert torch.isfinite(loss)
        assert torch.isfinite(nll)
        assert feat.item() == 0.0
        assert prefix_feat.item() > 0.0
        loss.backward()
        assert any(
            parameter.grad is not None
            for parameter in model.user_review_projector.parameters()
        )
        optimizer.step()


def test_two_batch_review_prefix_training_smoke_cpu():
    run_two_batch_review_prefix_training_smoke("cpu")


def test_train_step_masks_prompt_tokens_from_nll_targets():
    tokenizer = DummyTokenizer()
    selector = EvidenceSelector(embed_dim=4, hidden_dim=8, gnn_layers=1)
    profile_ids = torch.tensor([[3, 4, 5]])
    target_item_ids = torch.tensor([[6]])
    generation_prompt_ids = torch.tensor([[9]])
    input_ids = torch.tensor([[10, 11]])

    model = _make_model(
        tokenizer,
        evidence_selector=selector,
        evidence_bonus=0.0,
    )

    prompt_len = model._prompt_length(
        profile_ids=profile_ids,
        target_item_ids=target_item_ids,
        generation_prompt_ids=generation_prompt_ids,
    )
    total_len = prompt_len + input_ids.shape[1]
    fixed_logits = torch.zeros(total_len, 32)
    fixed_logits[:prompt_len - 1, 20] = 8.0
    fixed_logits[prompt_len - 1, 10] = 1.5
    fixed_logits[prompt_len, 11] = 0.5
    model.model = FixedFullLogitLM(logits=fixed_logits)

    loss, nll, feat, prefix_feat = model.train_step(
        input_ids,
        profile_ids=profile_ids,
        profile_mask=torch.ones_like(profile_ids),
        target_item_ids=target_item_ids,
        target_item_mask=torch.ones_like(target_item_ids),
        generation_prompt_ids=generation_prompt_ids,
        generation_prompt_mask=torch.ones_like(generation_prompt_ids),
    )

    response_logits = fixed_logits[prompt_len - 1:prompt_len + 1]
    manual_log_probs = torch.log_softmax(response_logits, dim=-1)
    manual_nll = -torch.stack([manual_log_probs[0, 10], manual_log_probs[1, 11]]).mean()
    assert torch.allclose(nll, manual_nll)
    assert torch.allclose(loss, manual_nll)
    assert feat.item() == 0.0
    assert prefix_feat.item() == 0.0


def test_evidence_bonus_reduces_nll_for_gold_evidence_token():
    tokenizer = DummyTokenizer()
    selector = EvidenceSelector(embed_dim=4, hidden_dim=8, gnn_layers=1)
    fixed_logits = torch.zeros(3, 32)
    fixed_logits[0, 10] = 1.5
    fixed_logits[1, 10] = 0.5

    input_ids = torch.tensor([[10, 10]])
    userid = torch.tensor([0])
    itemid = torch.tensor([1])
    evidence_token_ids, evidence_token_mask = pad_token_matrix([[10]], pad_value=-1)

    no_bonus_model = _make_model(tokenizer, evidence_bonus=0.0)
    no_bonus_model.model = FixedLogitLM(logits=fixed_logits)

    bonus_model = _make_model(tokenizer, evidence_bonus=0.5)
    bonus_model.model = FixedLogitLM(logits=fixed_logits)

    _loss0, nll0, _feat0, _prefix_feat0 = no_bonus_model.train_step(
        input_ids,
        evidence_token_ids=evidence_token_ids,
        evidence_token_mask=evidence_token_mask,
    )
    _loss_bonus, nll_bonus, _feat_bonus, _prefix_feat_bonus = bonus_model.train_step(
        input_ids,
        evidence_token_ids=evidence_token_ids,
        evidence_token_mask=evidence_token_mask,
    )

    adjusted = fixed_logits[:2].clone()
    adjusted[:, 10] += 0.5
    manual_log_probs = torch.log_softmax(adjusted, dim=-1)
    manual_bonus_nll = -torch.stack([manual_log_probs[0, 10], manual_log_probs[1, 10]]).mean()
    assert torch.allclose(nll_bonus, manual_bonus_nll)
    assert nll_bonus.item() < nll0.item()


def test_feature_learning_loss_uses_matched_target_positions():
    tokenizer = DummyTokenizer()
    selector = EvidenceSelector(embed_dim=4, hidden_dim=8, gnn_layers=1)
    fixed_logits = torch.zeros(3, 32)
    fixed_logits[0, 11] = 2.0
    fixed_logits[1, 12] = 1.0

    model = _make_model(
        tokenizer,
        evidence_selector=selector,
        lambda_feat=0.5,
        evidence_bonus=0.0,
    )
    model.model = FixedLogitLM(logits=fixed_logits)

    input_ids = torch.tensor([[11, 12]])
    feature_position_mask = torch.tensor([[False, True]])
    feature_position_weights = torch.tensor([[0.0, 1.0]])

    loss, nll, feat, prefix_feat = model.train_step(
        input_ids,
        feature_position_mask=feature_position_mask,
        feature_position_weights=feature_position_weights,
    )

    manual_log_probs = torch.log_softmax(fixed_logits[:2], dim=-1)
    manual_nll = -torch.stack([manual_log_probs[0, 11], manual_log_probs[1, 12]]).mean()
    manual_feat = -manual_log_probs[1, 12]
    assert torch.allclose(nll, manual_nll)
    assert torch.allclose(feat, manual_feat)
    assert torch.allclose(loss, manual_nll + 0.5 * feat)
    assert prefix_feat.item() == 0.0


def test_batch_graphs():
    g1 = build_sample_token_graph(
        [ReviewRecord(1, "u", "a", (10, 11))],
        exclude_row_key=-1,
        target_raw_item="x",
        skip_token_ids={0, 1, 2},
    )
    g2 = build_sample_token_graph(
        [ReviewRecord(2, "u", "b", (13, 14))],
        exclude_row_key=-1,
        target_raw_item="x",
        skip_token_ids={0, 1, 2},
    )
    batched = batch_graphs([g1, g2])
    assert batched["batch_index"].shape[0] == batched["node_token_ids"].shape[0]
    assert batched["num_nodes_per_graph"].tolist() == [g1.num_nodes, g2.num_nodes]


def test_stratified_tail_node_selection():
    stats = TailTokenStats(
        doc_freq={10: 20, 11: 15, 13: 1, 14: 1},
        num_documents=100,
        reference_df=15.0,
        tail_threshold=5,
    )
    graph = build_sample_token_graph(
        [ReviewRecord(1, "u", "a", (10, 10, 10, 11, 13, 14))],
        exclude_row_key=-1,
        target_raw_item="x",
        skip_token_ids={0, 1, 2},
        max_nodes=4,
        tail_stats=stats,
        target_item_token_ids={13},
        content_token_filter=lambda token_id: token_id in {10, 11, 13, 14},
        tail_node_quota=2,
        relevance_node_quota=1,
        preference_node_quota=1,
    )
    assert 13 in graph.node_token_ids.tolist()
    assert 14 in graph.node_token_ids.tolist()


def test_tokenize_text_list_padding():
    tokenizer = DummyTokenizer()
    ids, mask = tokenize_text_list(
        tokenizer,
        ['The explanation of Inception is "', "short"],
        pad_token_id=0,
        max_tokens=8,
    )
    assert ids.shape[0] == 2
    assert mask.shape == ids.shape
    assert mask.sum().item() >= 2


def test_graph_collater_batch_layout_without_auxiliary_fields():
    tokenizer = DummyTokenizer()
    collater = GraphCollater(
        max_step=1000,
        word=20,
        tokenizer=tokenizer,
        profile_records={},
        item_meta={},
        graph_manager=None,
    )
    batch = collater([
        {
            "text": [10, 11],
            "keyword": [13],
            "keyword_words": "great",
            "user": 0,
            "item": 1,
            "raw_user": "user_a",
            "raw_item": "item_a",
            "rating": 4,
            "local_idx": 0,
            "split_name": "test",
        }
    ])
    assert len(batch) == 14
    assert batch[0].tolist() == [[10, 11]]
    assert "node_token_ids" in batch[6]
    assert isinstance(batch[7], list)
    assert batch[10] == ["user_a"]
    assert batch[11].shape[0] == 1
    assert batch[11].dtype == torch.bool
    assert batch[12].dtype == torch.float32
    assert batch[13] == [{
        "raw_user": "user_a",
        "raw_item": "item_a",
        "split_name": "test",
        "local_idx": 0,
    }]


def test_graph_collater_feature_mask_matches_spaced_keyword_variant():
    tokenizer = DummyTokenizer()
    collater = GraphCollater(
        max_step=1000,
        word=20,
        tokenizer=tokenizer,
        profile_records={},
        item_meta={},
        graph_manager=None,
    )
    batch = collater([
        {
            "text": [30, 18, 31],
            "keyword": [10],
            "keyword_words": "great",
            "user": 0,
            "item": 1,
            "raw_user": "user_a",
            "raw_item": "item_a",
            "rating": 4,
            "local_idx": 0,
            "split_name": "test",
        }
    ])
    assert batch[11].tolist() == [[False, True, False]]
    assert batch[12].tolist() == [[0.0, 2.0, 0.0]]


def test_graph_collater_feature_mask_filters_short_bpe_pieces():
    tokenizer = DummyTokenizer()
    collater = GraphCollater(
        max_step=1000,
        word=20,
        tokenizer=tokenizer,
        profile_records={},
        item_meta={},
        graph_manager=None,
    )
    batch = collater([
        {
            "text": [19, 20],
            "keyword": [19, 20],
            "keyword_words": "television",
            "user": 0,
            "item": 1,
            "raw_user": "user_a",
            "raw_item": "item_a",
            "rating": 4,
            "local_idx": 0,
            "split_name": "test",
        }
    ])
    assert batch[11].tolist() == [[False, True]]
    assert batch[12].tolist() == [[0.0, 2.0]]


def test_graph_collater_feature_weights_include_local_content_context():
    tokenizer = DummyTokenizer()
    collater = GraphCollater(
        max_step=1000,
        word=20,
        tokenizer=tokenizer,
        profile_records={},
        item_meta={},
        graph_manager=None,
    )
    batch = collater([
        {
            "text": [10, 11, 13],
            "keyword": [10],
            "keyword_words": "great",
            "user": 0,
            "item": 1,
            "raw_user": "user_a",
            "raw_item": "item_a",
            "rating": 4,
            "local_idx": 0,
            "split_name": "test",
        }
    ])
    assert batch[11].tolist() == [[True, True, True]]
    torch.testing.assert_close(
        batch[12],
        torch.tensor([[2.0, 0.2, 0.1]], dtype=torch.float32),
    )


def test_lora_defaults_target_qkvo_r16():
    args = build_arg_parser().parse_args([])
    assert args.lora_r == 16
    assert args.lora_alpha == 32
    assert args.lora_dropout == 0.05
    assert args.lora_target_modules == "q_proj,k_proj,v_proj,o_proj"


def test_standalone_default_paths_are_graph_local():
    args = build_arg_parser().parse_args([])
    graph_root = ROOT.resolve()
    assert Path(args.data_dir).resolve() == graph_root / "data"
    assert Path(args.profile_dir).resolve() == graph_root / "data" / "profiles"
    model_candidates = {path.resolve() for path in qwen3_4b_model_candidates()}
    assert Path(args.model_path).resolve() in model_candidates
    assert (
        Path(args.embedding_model_path).resolve()
        == graph_root / "pretrain_llm" / "qwen3-embedding-0.6b"
    )
    assert args.dataset_name == "Amazon/MoviesAndTV_corsa_filtered_small_15pct/"


def test_resolve_llm_device_map_mode_auto():
    args = SimpleNamespace(llm_device_map="auto")
    assert resolve_llm_device_map_mode(args, [0, 1]) == "balanced"
    assert resolve_llm_device_map_mode(args, [1]) == "single"


def test_build_llm_max_memory_reserves_embedding_space():
    from unittest.mock import patch

    args = SimpleNamespace(
        primary_gpu_max_gib=0.0,
        secondary_gpu_max_gib=0.0,
        gpu_memory_fraction=0.92,
        primary_foreign_reserve_gib=12.0,
        primary_gpu_balance_ratio=0.92,
    )
    with patch("graph_llm.train.trainer.gpu_total_gib", return_value=44.0):
        max_memory = build_llm_max_memory([0, 1], torch.device("cuda:1"), 10.0, args)
    assert max_memory[0] == "28GiB"
    assert max_memory[1] == "30GiB"


def test_resolve_local_model_path_prefers_graph_copy():
    graph_root = ROOT.resolve()
    graph_copy = graph_root / "models" / "qwen3-4b"
    gpt1_copy = REPO / "gpt1" / "models" / "qwen3-4b"
    if not (graph_copy / "config.json").exists() and not (gpt1_copy / "config.json").exists():
        return
    resolved = resolve_local_model_path(
        graph_copy,
        candidates=qwen3_4b_model_candidates(),
    )
    expected = graph_copy if (graph_copy / "config.json").exists() else gpt1_copy
    assert Path(resolved).resolve() == expected.resolve()


def test_standalone_metrics_available():
    references = [["great", "movie"]]
    generated = [["great", "story"]]
    assert bleu_score(references, generated, n_gram=1) > 0.0
    scores = rouge_score(["great movie"], ["great story"])
    assert {"rouge_1", "rouge_2", "rouge_l"}.issubset(scores)


def test_derive_profiles_from_parent_subset():
    """子数据集 profile 应从父 profile 按 split 用户过滤派生。"""
    import pickle

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        data_dir = base / "data"
        profile_dir = base / "profiles"
        parent_name = "Amazon/MoviesAndTV_corsa_filtered"
        child_name = "Amazon/MoviesAndTV_corsa_filtered_small_15pct"

        reviews = pd.DataFrame(
            {
                "user": ["u1", "u1", "u2", "u3"],
                "item": ["i1", "i2", "i3", "i4"],
                "rating": [5, 4, 3, 2],
                "template": [("k", "w", "t")] * 4,
            }
        )
        child_ds = data_dir / child_name
        child_ds.mkdir(parents=True)
        reviews.to_pickle(child_ds / "reviews.pickle")
        fold_dir = child_ds / "1"
        fold_dir.mkdir(parents=True)
        (fold_dir / "train.index").write_text("0 1 2\n", encoding="utf-8")
        (fold_dir / "validation.index").write_text("2\n", encoding="utf-8")
        (fold_dir / "test.index").write_text("3\n", encoding="utf-8")

        parent_profiles = {
            "u1": {
                "raw_user": "u1",
                "scope": "train",
                "profile_text": "profile u1",
                "llama_profile": "profile u1",
            },
            "u2": {
                "raw_user": "u2",
                "scope": "train",
                "profile_text": "profile u2",
                "llama_profile": "profile u2",
            },
            "u3": {
                "raw_user": "u3",
                "scope": "train",
                "profile_text": "profile u3",
                "llama_profile": "profile u3",
            },
        }
        parent_dir = profile_dir / parent_name
        parent_dir.mkdir(parents=True)
        with (parent_dir / "fold_1_train.pkl").open("wb") as f:
            pickle.dump(parent_profiles, f)
        with (parent_dir / "fold_1_train_valid.pkl").open("wb") as f:
            pickle.dump(parent_profiles, f)

        written = derive_profiles(
            data_dir=data_dir,
            dataset_name=child_name,
            profile_dir=profile_dir,
            fold="1",
            scopes=["train", "train_valid"],
            source_dataset_name=parent_name,
            overwrite=True,
        )

        train_path = written["train"]
        with train_path.open("rb") as f:
            train_profiles = pickle.load(f)
        assert set(train_profiles.keys()) == {"u1", "u2"}
        assert train_profiles["u1"]["scope"] == "train"
        assert train_profiles["u1"]["derived_from"].endswith("fold_1_train.pkl")
        assert train_profiles["u1"]["profile_text"] == "profile u1"

        with written["train_valid"].open("rb") as f:
            tv_profiles = pickle.load(f)
        assert set(tv_profiles.keys()) == {"u1", "u2"}
        assert tv_profiles["u2"]["scope"] == "train_valid"

        args = SimpleNamespace(
            profile_dir=str(profile_dir),
            dataset_name=f"{child_name}/",
            allow_missing_profiles=False,
        )
        assert profile_cache_path(args, "1", "train") == train_path


def test_profile_cache_falls_back_to_base_dataset():
    candidates = profile_dataset_name_candidates("Amazon/MoviesAndTV_corsa_filtered_small_15pct/")
    assert candidates[:2] == [
        "Amazon/MoviesAndTV_corsa_filtered_small_15pct",
        "Amazon/MoviesAndTV_corsa_filtered_small",
    ]
    assert "Amazon/MoviesAndTV_corsa_filtered" in candidates
    assert "Amazon/MoviesAndTV" in candidates

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        fallback_dir = base / "Amazon" / "MoviesAndTV"
        fallback_dir.mkdir(parents=True)
        (fallback_dir / "fold_1_train.pkl").write_bytes(b"cache")
        (fallback_dir / "fold_1_train_valid.pkl").write_bytes(b"cache")
        args = SimpleNamespace(
            profile_dir=str(base),
            dataset_name="Amazon/MoviesAndTV_corsa_filtered_small_15pct/",
            allow_missing_profiles=False,
        )
        assert profile_cache_path(args, "1", "train") == fallback_dir / "fold_1_train.pkl"
        preflight_profile_cache_files(args, ["1"])


def test_devices_parsing_dual_gpu():
    assert parse_device_ids("0,1") == [0, 1]
    assert parse_device_ids("1") == [1]
    assert parse_device_ids("cuda:0,cuda:1") == [0, 1]

    primary = torch.device("cuda:0")
    assert resolve_embedding_device("auto", primary, [0, 1]) == torch.device("cuda:1")
    assert resolve_embedding_device("auto", torch.device("cuda:1"), [1]) == torch.device("cuda:1")
    assert resolve_embedding_device("cuda:1", primary, [0]) == torch.device("cuda:1")
    assert resolve_embedding_device("1", primary, [0]) == torch.device("cuda:1")


def test_devices_arg_defaults():
    args = build_arg_parser().parse_args([])
    assert args.devices == "default"
    assert args.llm_device_map == "single"
    assert args.oom_fallback == "auto"
    assert args.embedding_device == "auto"
    assert args.memory_warn_gib == 24.0
    assert args.review_top_k_user == 16
    assert args.review_top_k_item == 32
    assert args.user_review_prefix_len == 4
    assert args.item_review_prefix_len == 4
    assert args.lambda_prefix_feature == 0.1


def test_resolve_devices_string_prefers_cuda1():
    from unittest.mock import patch

    with patch("graph_llm.train.trainer.available_cuda_device_ids", return_value=[0, 1]):
        assert resolve_devices_string("default") == "1"
        assert parse_device_ids("default") == [1]
        assert default_preferred_device_id() == 1
    with patch("graph_llm.train.trainer.available_cuda_device_ids", return_value=[0]):
        assert resolve_devices_string("default") == "0"
        assert default_preferred_device_id() == 0


def test_build_oom_plans_progressively_reduce_memory():
    from unittest.mock import patch

    baseline = snapshot_training_args(build_arg_parser().parse_args([]))
    baseline.batch_size = 8
    baseline.eval_batch_size = 8
    with patch("graph_llm.train.trainer.available_cuda_device_ids", return_value=[0, 1]):
        plans = build_oom_plans(baseline)
    names = [plan.name for plan in plans]
    assert names[0] == "single_fast"
    assert "single_checkpointing" in names
    assert "dual_balanced" in names
    assert "single_batch4" in names
    assert "single_batch2" in names
    assert plans[0].devices == "1"
    assert plans[0].gradient_checkpointing is False


def test_resolve_training_devices_single_gpu():
    from unittest.mock import patch

    args = SimpleNamespace(devices="1", embedding_device="auto")
    with patch("graph_llm.train.trainer.torch.cuda.is_available", return_value=True):
        with patch("graph_llm.train.trainer.available_cuda_device_ids", return_value=[0, 1]):
            with patch("graph_llm.train.trainer.torch.cuda.set_device") as set_device:
                primary, embedding, device_ids = resolve_training_devices(args)
    assert device_ids == [1]
    assert primary == torch.device("cuda:1")
    assert embedding == torch.device("cuda:1")
    set_device.assert_called_once_with(1)


def test_resolve_training_devices_dual_gpu():
    from unittest.mock import patch

    args = SimpleNamespace(devices="0,1", embedding_device="auto")
    with patch("graph_llm.train.trainer.torch.cuda.is_available", return_value=True):
        with patch("graph_llm.train.trainer.available_cuda_device_ids", return_value=[0, 1]):
            with patch("graph_llm.train.trainer.torch.cuda.set_device") as set_device:
                primary, embedding, device_ids = resolve_training_devices(args)
    assert device_ids == [0, 1]
    assert primary == torch.device("cuda:0")
    assert embedding == torch.device("cuda:1")
    set_device.assert_called_once_with(0)


class _StubEmbeddingEncoder:
    backend = "qwen_embedding"

    def __init__(self, embed_device):
        self.embed_device = embed_device

    def encode_token_ids(self, token_ids, decode_fn):
        count = len(list(token_ids))
        return torch.randn(count, 4, device=self.embed_device)

    def encode_texts(self, texts):
        return torch.randn(len(texts), 4, device=self.embed_device)


def test_compute_batch_selector_tensors_moves_embeddings_to_primary():
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        embed_device = torch.device("cuda:1")
        primary = torch.device("cuda:0")
    else:
        embed_device = torch.device("cpu")
        primary = torch.device("cpu")

    tokenizer = DummyTokenizer()
    model = _make_model(tokenizer)
    model.evidence_selector.to(primary)
    graph = UserTokenGraph(
        node_token_ids=np.array([10, 11, 13], dtype=np.int64),
        node_surfaces=["great", "movie", "story"],
        node_counts=np.array([2.0, 1.0, 1.0], dtype=np.float32),
        node_doc_freq=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        edge_weight=np.array([1.0, 1.0], dtype=np.float32),
        in_degree=np.array([0.0, 1.0, 1.0], dtype=np.float32),
        out_degree=np.array([1.0, 1.0, 0.0], dtype=np.float32),
    )
    graph_tensors = {
        "node_token_ids": torch.tensor([10, 11, 13], dtype=torch.long),
    }
    args = SimpleNamespace(
        special_token_ids="auto",
        top_m_evidence=1,
    )
    tail_stats = TailTokenStats(
        doc_freq={10: 2, 11: 5, 13: 1},
        num_documents=5,
        reference_df=5.0,
        tail_threshold=5,
    )
    evidence_token_ids, evidence_token_mask, selector_loss = (
        compute_batch_selector_tensors(
            model,
            _StubEmbeddingEncoder(embed_device),
            [graph],
            graph_tensors,
            ["great movie"],
            tokenizer,
            args,
            primary,
            input_ids=torch.tensor([[10, 11]], dtype=torch.long, device=primary),
            tail_stats=tail_stats,
            compute_selector_loss=True,
        )
    )
    assert evidence_token_ids.device == primary
    assert selector_loss.device == primary
    assert selector_loss.requires_grad


def test_embedding_cache_corruption_recovery():
    import json
    import tempfile
    from graph_llm.dataload.embeddings import EmbeddingCache

    cache_dir = Path(tempfile.mkdtemp())
    vec_path = cache_dir / "abc123.pt"
    torch.save(torch.tensor([1.0, 2.0]), vec_path)
    (cache_dir / "index.json").write_text('{"abc123": "abc123.pt"', encoding="utf-8")

    cache = EmbeddingCache(cache_dir)
    assert cache.index.get("abc123") == "abc123.pt"
    loaded = cache.get("ignored")
    assert loaded is None
    loaded = torch.load(vec_path, weights_only=True)
    assert loaded.tolist() == [1.0, 2.0]
    with cache.index_path.open("r", encoding="utf-8") as f:
        json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run graph_llm smoke tests")
    parser.add_argument(
        "--smoke_device",
        default="auto",
        help="prefix 两批 smoke test 设备：auto、cpu、cuda:0 或 cuda:1",
    )
    smoke_args = parser.parse_args()
    test_directed_edges_and_leakage()
    test_selector_shapes()
    test_tail_stats_and_selector_bce()
    test_prompt_text_includes_title_and_user_id()
    test_prompt_length_accounts_for_new_segments()
    test_generation_controls_filter_evidence_and_block_repeats()
    test_train_step_uses_tail_weighted_sft()
    test_two_batch_tail_sft_training_smoke()
    test_review_memory_excludes_current_training_target()
    run_two_batch_review_prefix_training_smoke(smoke_args.smoke_device)
    test_batch_graphs()
    test_stratified_tail_node_selection()
    test_tokenize_text_list_padding()
    test_devices_parsing_dual_gpu()
    test_devices_arg_defaults()
    test_resolve_devices_string_prefers_cuda1()
    test_build_oom_plans_progressively_reduce_memory()
    test_resolve_training_devices_single_gpu()
    test_resolve_training_devices_dual_gpu()
    test_resolve_llm_device_map_mode_auto()
    test_build_llm_max_memory_reserves_embedding_space()
    test_compute_batch_selector_tensors_moves_embeddings_to_primary()
    test_resolve_local_model_path_prefers_graph_copy()
    test_embedding_cache_corruption_recovery()
    print("graph smoke tests passed")
