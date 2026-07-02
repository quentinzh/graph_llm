"""CPU-safe smoke tests for graph evidence selector."""

from __future__ import annotations

import sys
import tempfile
from importlib.util import find_spec
from types import ModuleType, SimpleNamespace
from pathlib import Path

import numpy as np
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

from graph_llm.aux.prompt_utils import (
    build_evidence_prompt_text,
    build_generation_prompt_text,
    build_generation_prompt_batch,
    item_meta_from_row,
    tokenize_text_list,
)
from graph_llm.dataload.dataloader import GraphCollater
from graph_llm.metrics.metrics import bleu_score, rouge_score
from graph_llm.models.selector import EvidenceSelector, pad_token_matrix
from graph_llm.models.token_graph import (
    ReviewRecord,
    UserTokenGraph,
    batch_graphs,
    build_sample_token_graph,
    select_high_frequency_negatives,
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
    compute_batch_selector_tensors,
    default_preferred_device_id,
    parse_device_ids,
    profile_cache_path,
    profile_dataset_name_candidates,
    preflight_profile_cache_files,
    resolve_devices_string,
    resolve_embedding_device,
    resolve_llm_device_map_mode,
    resolve_training_devices,
)


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
            'The explanation of Inception is "': [30, 31],
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
    node_emb = torch.randn(graph.num_nodes, 8)
    item_emb = torch.randn(8)
    scores = selector.forward_single(graph, node_emb, item_emb, num_reviews=3.0)
    assert scores.shape == (graph.num_nodes,)
    selected = selector.select_evidence_and_negatives(
        scores, graph, top_m=2, ul_candidate_k=2, protected_token_ids={0, 1, 2},
    )
    assert selected["evidence_token_ids"].size == 2
    assert selected["neg_token_ids"].size <= 2
    evidence_set = set(selected["evidence_token_ids"].tolist())
    neg_set = set(selected["neg_token_ids"].tolist())
    assert evidence_set.isdisjoint(neg_set)


def test_ul_masking():
    tokenizer = DummyTokenizer()
    selector = EvidenceSelector(embed_dim=4, hidden_dim=8, gnn_layers=1)
    model = GraphEvidenceCIER(
        user_num=2,
        item_num=2,
        hidden=8,
        llm_hidden=16,
        tokenizer=tokenizer,
        vocab_size=32,
        evidence_selector=selector,
        lambda_ul=0.1,
        pad_token_id=0,
        eos_token_ids=(2,),
        special_token_ids=(0, 1, 2),
    )

    gen_logits = torch.zeros(1, 2, 32)
    gen_logits[0, 0, 10] = 3.0
    gen_logits[0, 0, 11] = 2.0
    gen_logits[0, 1, 13] = 3.0
    targets = torch.tensor([[10, 13]], dtype=torch.long)
    valid_mask = torch.ones(1, 2, dtype=torch.long)
    evidence_token_ids, evidence_token_mask = pad_token_matrix([[10]], pad_value=-1)
    neg_token_ids, neg_token_mask = pad_token_matrix([[11]], pad_value=-1)
    neg_weights, _ = pad_token_matrix([[1.0]], pad_value=0.0, dtype=torch.float32)

    ul = model._graph_unlikelihood_loss(
        gen_logits,
        targets,
        valid_mask,
        neg_token_ids,
        neg_token_mask,
        evidence_token_ids,
        evidence_token_mask,
        neg_weights,
    )
    assert ul.item() > 0.0


def test_prompt_text_includes_title_and_evidence():
    tokenizer = DummyTokenizer()
    evidence_text = build_evidence_prompt_text([10, 11, 15, 16, 17], tokenizer)
    assert "great" in evidence_text
    assert "movie" in evidence_text
    assert "the" not in evidence_text
    assert "1" not in evidence_text
    assert "j" not in evidence_text
    generation_text = build_generation_prompt_text("Inception")
    assert generation_text == 'The explanation of Inception is "'

    title, description, item_text = item_meta_from_row(
        "item123",
        {"item123": {"title": "Inception", "description": "A dream heist film"}},
    )
    assert title == "Inception"
    assert "Title: Inception" in item_text
    assert "Description: A dream heist film" in item_text

    gen_ids, gen_mask = build_generation_prompt_batch(
        ["Inception"], tokenizer, pad_token_id=0, max_tokens=32,
    )
    assert gen_ids.shape[0] == 1
    assert gen_mask.sum().item() > 0


def test_prompt_length_accounts_for_new_segments():
    tokenizer = DummyTokenizer()
    selector = EvidenceSelector(embed_dim=4, hidden_dim=8, gnn_layers=1)
    model = GraphEvidenceCIER(
        user_num=2,
        item_num=2,
        hidden=8,
        llm_hidden=16,
        tokenizer=tokenizer,
        vocab_size=32,
        evidence_selector=selector,
        lambda_ul=0.1,
        pad_token_id=0,
        eos_token_ids=(2,),
        special_token_ids=(0, 1, 2),
    )
    profile_ids = torch.tensor([[1, 2, 3]])
    target_item_ids = torch.tensor([[4, 5]])
    evidence_prompt_ids = torch.tensor([[6, 7]])
    generation_prompt_ids = torch.tensor([[8, 9, 10]])
    prompt_len = model._prompt_length(
        profile_ids,
        target_item_ids,
        evidence_prompt_ids,
        generation_prompt_ids,
    )
    assert prompt_len == 3 + 2 + 2 + 2 + 3


def test_generation_controls_filter_evidence_and_block_repeats():
    tokenizer = DummyTokenizer()
    selector = EvidenceSelector(embed_dim=4, hidden_dim=8, gnn_layers=1)
    model = GraphEvidenceCIER(
        user_num=2,
        item_num=2,
        hidden=8,
        llm_hidden=16,
        tokenizer=tokenizer,
        vocab_size=32,
        evidence_selector=selector,
        lambda_ul=0.1,
        evidence_bonus=0.5,
        max_consecutive_token_repeat=2,
        pad_token_id=0,
        eos_token_ids=(2,),
        special_token_ids=(0, 1, 2),
    )
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

    model = GraphEvidenceCIER(
        user_num=2,
        item_num=2,
        hidden=8,
        llm_hidden=8,
        tokenizer=tokenizer,
        vocab_size=32,
        evidence_selector=selector,
        lambda_ul=0.0,
        evidence_bonus=0.0,
        pad_token_id=0,
        eos_token_ids=(2,),
        special_token_ids=(0, 1, 2),
    )
    model.model = FixedLogitLM(logits=fixed_logits)

    input_ids = torch.tensor([[10, 11]])
    userid = torch.tensor([0])
    itemid = torch.tensor([1])
    evidence_token_ids, evidence_token_mask = pad_token_matrix([[10]], pad_value=-1)
    loss, nll, ul, feat = model.train_step(
        input_ids,
        userid,
        itemid,
        evidence_token_ids=evidence_token_ids,
        evidence_token_mask=evidence_token_mask,
        apply_unlikelihood=False,
    )

    manual_log_probs = torch.log_softmax(fixed_logits[:2], dim=-1)
    manual_nll = -torch.stack([manual_log_probs[0, 10], manual_log_probs[1, 11]]).mean()
    assert torch.allclose(nll, manual_nll)
    assert torch.allclose(loss, manual_nll)
    assert ul.item() == 0.0
    assert feat.item() == 0.0


def test_train_step_masks_prompt_tokens_from_nll_targets():
    tokenizer = DummyTokenizer()
    selector = EvidenceSelector(embed_dim=4, hidden_dim=8, gnn_layers=1)
    profile_ids = torch.tensor([[3, 4, 5]])
    target_item_ids = torch.tensor([[6]])
    evidence_prompt_ids = torch.tensor([[7, 8]])
    generation_prompt_ids = torch.tensor([[9]])
    input_ids = torch.tensor([[10, 11]])
    userid = torch.tensor([0])
    itemid = torch.tensor([1])

    model = GraphEvidenceCIER(
        user_num=2,
        item_num=2,
        hidden=8,
        llm_hidden=8,
        tokenizer=tokenizer,
        vocab_size=32,
        evidence_selector=selector,
        lambda_ul=0.0,
        evidence_bonus=0.0,
        pad_token_id=0,
        eos_token_ids=(2,),
        special_token_ids=(0, 1, 2),
    )

    prompt_len = model._prompt_length(
        profile_ids,
        target_item_ids,
        evidence_prompt_ids,
        generation_prompt_ids,
    )
    total_len = prompt_len + input_ids.shape[1]
    fixed_logits = torch.zeros(total_len, 32)
    fixed_logits[:prompt_len - 1, 20] = 8.0
    fixed_logits[prompt_len - 1, 10] = 1.5
    fixed_logits[prompt_len, 11] = 0.5
    model.model = FixedFullLogitLM(logits=fixed_logits)

    loss, nll, ul, feat = model.train_step(
        input_ids,
        userid,
        itemid,
        profile_ids=profile_ids,
        profile_mask=torch.ones_like(profile_ids),
        target_item_ids=target_item_ids,
        target_item_mask=torch.ones_like(target_item_ids),
        evidence_prompt_ids=evidence_prompt_ids,
        evidence_prompt_mask=torch.ones_like(evidence_prompt_ids),
        generation_prompt_ids=generation_prompt_ids,
        generation_prompt_mask=torch.ones_like(generation_prompt_ids),
        apply_unlikelihood=False,
    )

    response_logits = fixed_logits[prompt_len - 1:prompt_len + 1]
    manual_log_probs = torch.log_softmax(response_logits, dim=-1)
    manual_nll = -torch.stack([manual_log_probs[0, 10], manual_log_probs[1, 11]]).mean()
    assert torch.allclose(nll, manual_nll)
    assert torch.allclose(loss, manual_nll)
    assert ul.item() == 0.0
    assert feat.item() == 0.0


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

    no_bonus_model = GraphEvidenceCIER(
        user_num=2,
        item_num=2,
        hidden=8,
        llm_hidden=8,
        tokenizer=tokenizer,
        vocab_size=32,
        evidence_selector=selector,
        lambda_ul=0.0,
        evidence_bonus=0.0,
        pad_token_id=0,
        eos_token_ids=(2,),
        special_token_ids=(0, 1, 2),
    )
    no_bonus_model.model = FixedLogitLM(logits=fixed_logits)

    bonus_model = GraphEvidenceCIER(
        user_num=2,
        item_num=2,
        hidden=8,
        llm_hidden=8,
        tokenizer=tokenizer,
        vocab_size=32,
        evidence_selector=selector,
        lambda_ul=0.0,
        evidence_bonus=0.5,
        pad_token_id=0,
        eos_token_ids=(2,),
        special_token_ids=(0, 1, 2),
    )
    bonus_model.model = FixedLogitLM(logits=fixed_logits)

    _loss0, nll0, _ul0, _feat0 = no_bonus_model.train_step(
        input_ids,
        userid,
        itemid,
        evidence_token_ids=evidence_token_ids,
        evidence_token_mask=evidence_token_mask,
        apply_unlikelihood=False,
    )
    _loss_bonus, nll_bonus, _ul_bonus, _feat_bonus = bonus_model.train_step(
        input_ids,
        userid,
        itemid,
        evidence_token_ids=evidence_token_ids,
        evidence_token_mask=evidence_token_mask,
        apply_unlikelihood=False,
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

    model = GraphEvidenceCIER(
        user_num=2,
        item_num=2,
        hidden=8,
        llm_hidden=8,
        tokenizer=tokenizer,
        vocab_size=32,
        evidence_selector=selector,
        lambda_ul=0.1,
        lambda_feat=0.5,
        evidence_bonus=0.0,
        pad_token_id=0,
        eos_token_ids=(2,),
        special_token_ids=(0, 1, 2),
    )
    model.model = FixedLogitLM(logits=fixed_logits)

    input_ids = torch.tensor([[11, 12]])
    userid = torch.tensor([0])
    itemid = torch.tensor([1])
    feature_position_mask = torch.tensor([[False, True]])
    feature_position_weights = torch.tensor([[0.0, 1.0]])

    loss, nll, ul, feat = model.train_step(
        input_ids,
        userid,
        itemid,
        feature_position_mask=feature_position_mask,
        feature_position_weights=feature_position_weights,
        apply_unlikelihood=True,
    )

    manual_log_probs = torch.log_softmax(fixed_logits[:2], dim=-1)
    manual_nll = -torch.stack([manual_log_probs[0, 11], manual_log_probs[1, 12]]).mean()
    manual_feat = -manual_log_probs[1, 12]
    assert torch.allclose(nll, manual_nll)
    assert torch.allclose(feat, manual_feat)
    assert torch.allclose(loss, manual_nll + 0.1 * ul + 0.5 * feat)
    assert ul.item() == 0.0


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


def test_high_frequency_negative_selection():
    graph = UserTokenGraph(
        node_token_ids=np.array([10, 11, 13], dtype=np.int64),
        node_surfaces=["a", "b", "c"],
        node_counts=np.array([5.0, 4.0, 1.0], dtype=np.float32),
        node_doc_freq=np.array([2.0, 2.0, 1.0], dtype=np.float32),
        edge_index=np.empty((2, 0), dtype=np.int64),
        edge_weight=np.empty((0,), dtype=np.float32),
        in_degree=np.zeros(3, dtype=np.float32),
        out_degree=np.zeros(3, dtype=np.float32),
    )
    neg_nodes, neg_weights = select_high_frequency_negatives(
        graph,
        np.array([2], dtype=np.int64),
        top_k=2,
        protected_token_ids={0},
    )
    assert 2 not in neg_nodes.tolist()
    assert len(neg_nodes) <= 2


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
    assert "node_token_ids" in batch[8]
    assert isinstance(batch[9], list)
    assert batch[12].shape[0] == 1
    assert batch[12].dtype == torch.bool
    assert batch[13].dtype == torch.float32


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
    assert batch[12].tolist() == [[False, True, False]]
    assert batch[13].tolist() == [[0.0, 1.0, 0.0]]


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
    assert batch[12].tolist() == [[False, True]]
    assert batch[13].tolist() == [[0.0, 1.0]]


def test_lora_defaults_target_qkvo_r16():
    args = build_arg_parser().parse_args([])
    assert args.lora_r == 16
    assert args.lora_alpha == 32
    assert args.lora_dropout == 0.05
    assert args.lora_target_modules == "q_proj,k_proj,v_proj,o_proj"


def test_standalone_default_paths_are_graph_llm_local():
    args = build_arg_parser().parse_args([])
    graph_root = ROOT.resolve()
    assert Path(args.data_dir).resolve() == graph_root / "data"
    assert Path(args.profile_dir).resolve() == graph_root / "data" / "profiles"
    model_candidates = {path.resolve() for path in qwen3_4b_model_candidates()}
    assert Path(args.model_path).resolve() in model_candidates
    assert Path(args.embedding_model_path).resolve() == graph_root / "pretrain_llm" / "qwen3-embedding-0.6b"
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


def test_resolve_local_model_path_prefers_graph_llm_copy():
    graph_root = ROOT.resolve()
    graph_copy = graph_root / "pretrain_llm" / "qwen3-4b"
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


def test_profile_cache_uses_exact_dataset_name():
    candidates = profile_dataset_name_candidates("Amazon/MoviesAndTV_corsa_filtered_small_15pct/")
    assert candidates == ["Amazon/MoviesAndTV_corsa_filtered_small_15pct"]

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        dataset_dir = base / "Amazon" / "MoviesAndTV_corsa_filtered_small_15pct"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "fold_1_train.pkl").write_bytes(b"cache")
        (dataset_dir / "fold_1_train_valid.pkl").write_bytes(b"cache")
        args = SimpleNamespace(
            profile_dir=str(base),
            dataset_name="Amazon/MoviesAndTV_corsa_filtered_small_15pct/",
            allow_missing_profiles=False,
        )
        assert profile_cache_path(args, "1", "train") == dataset_dir / "fold_1_train.pkl"
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
    selector = EvidenceSelector(embed_dim=4, hidden_dim=8, gnn_layers=1)
    model = GraphEvidenceCIER(
        user_num=2,
        item_num=2,
        hidden=8,
        llm_hidden=8,
        tokenizer=tokenizer,
        vocab_size=32,
        evidence_selector=selector,
        lambda_ul=0.0,
        pad_token_id=0,
        eos_token_ids=(2,),
        special_token_ids=(0, 1, 2),
    )
    graph = UserTokenGraph(
        node_token_ids=np.array([10, 11], dtype=np.int64),
        node_surfaces=["great", "movie"],
        node_counts=np.array([2.0, 1.0], dtype=np.float32),
        node_doc_freq=np.array([1.0, 1.0], dtype=np.float32),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        edge_weight=np.array([1.0], dtype=np.float32),
        in_degree=np.array([0.0, 1.0], dtype=np.float32),
        out_degree=np.array([1.0, 0.0], dtype=np.float32),
    )
    graph_tensors = {
        "node_token_ids": torch.tensor([10, 11], dtype=torch.long),
    }
    args = SimpleNamespace(
        special_token_ids="auto",
        top_m_evidence=1,
        ul_candidate_k=1,
    )
    evidence_token_ids, evidence_token_mask, neg_token_ids, neg_token_mask, neg_weights = (
        compute_batch_selector_tensors(
            model,
            _StubEmbeddingEncoder(embed_device),
            [graph],
            graph_tensors,
            ["great movie"],
            tokenizer,
            args,
            primary,
        )
    )
    assert evidence_token_ids.device == primary
    assert neg_token_ids.device == primary
    assert neg_weights.device == primary


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
    test_directed_edges_and_leakage()
    test_selector_shapes()
    test_ul_masking()
    test_prompt_text_includes_title_and_evidence()
    test_prompt_length_accounts_for_new_segments()
    test_batch_graphs()
    test_high_frequency_negative_selection()
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
    test_resolve_local_model_path_prefers_graph_llm_copy()
    test_embedding_cache_corruption_recovery()
    print("graph_llm smoke tests passed")
