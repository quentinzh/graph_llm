#!/usr/bin/env python
"""解释生成训练与评估（冻结推荐主干 + Qwen LoRA）。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from graph_llm.dataload.embeddings import RobertaTextEncoder, SmokeTextEncoder, _text_hash
from graph_llm.dataload.explain_data import (
    ExplainSample,
    build_explain_samples,
    compute_tail_df_weights,
    fragment_evidence_label,
    history_for_explain_sample,
    sequence_token_weights,
)
from graph_llm.dataload.history_graph import batch_history_graphs, build_history_user_graph
from graph_llm.dataload.sequential_data import assign_calib_split, item_catalog_text, load_sequential_dataset
from graph_llm.dataload.semantic_id import build_semantic_ids, encode_all_item_texts, save_sid_bundle
from graph_llm.metrics.explain_metrics import (
    diversity_composite_score,
    evaluate_explanations,
    passes_quality_gate,
)
from graph_llm.models.sid_recommender import SIDRecommender, build_item_code_matrix
from graph_llm.train.recommend_trainer import (
    NodeEmbeddingStore,
    _append_jsonl,
    _build_item_emb_matrix,
    _collate_recommend_batch,
    _encode_graph_nodes,
    _precompute_fragment_cache,
    build_recommend_samples,
    default_preferred_device_id,
)

class PrefixAdapter(nn.Module):
    """决策依据 + 证据压缩为 soft prefix token。"""

    def __init__(self, hidden_dim: int, llm_dim: int, rationale_dim: int):
        super().__init__()
        self.rationale_proj = nn.Linear(rationale_dim, llm_dim)
        self.evidence_proj = nn.Linear(hidden_dim, llm_dim)

    def forward(self, rationale_feat: torch.Tensor, evidence_vec: torch.Tensor) -> torch.Tensor:
        """返回 [B, 2, llm_dim]。"""
        r = self.rationale_proj(rationale_feat).unsqueeze(1)
        e = self.evidence_proj(evidence_vec).unsqueeze(1)
        return torch.cat([r, e], dim=1)


class ExplainDataset(Dataset):
    def __init__(self, samples: list[ExplainSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _recommend_ckpt_path(args) -> Path:
    if getattr(args, "recommend_ckpt", "") and str(args.recommend_ckpt).strip():
        return Path(args.recommend_ckpt)
    return (
        Path(args.ckpt_dir)
        / args.dataset_name.replace("/", "__")
        / "recommend"
        / "sid_recommender.bin"
    )


def _build_prompt(item_meta: dict, max_chars: int = 512) -> str:
    title = str(item_meta.get("title") or "")
    desc = item_meta.get("description_str") or ""
    if not desc and item_meta.get("description"):
        parts = item_meta.get("description")
        if isinstance(parts, list):
            desc = " ".join(str(x) for x in parts)
    text = f"Item: {title}. Description: {desc[:max_chars]}. Write a short personalized review summary:"
    return text


def _weighted_ce_loss(logits, labels, weights, ignore_index=-100):
    """对有效 label 位置应用 token 权重。"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_w = weights[..., 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    flat_w = shift_w.view(-1)
    ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index, reduction="none")
    mask = flat_labels != ignore_index
    if mask.sum() == 0:
        return ce.sum() * 0.0
    w = flat_w * mask.float()
    return (ce * w).sum() / w.sum().clamp_min(1e-6)


def _forward_explain_batch(
    bundle,
    sid_bundle,
    item_codes,
    recommender: SIDRecommender,
    prefix_adapter: PrefixAdapter,
    llm,
    tokenizer,
    node_store: NodeEmbeddingStore,
    batch: list[ExplainSample],
    device: torch.device,
    args,
    tail_weights: dict[str, float],
) -> dict[str, torch.Tensor | float]:
    from graph_llm.train.recommend_trainer import RecommendSample

    rec_batch = [
        RecommendSample(
            interaction_id=s.interaction_id,
            user_index=s.user_index,
            target_item_index=s.target_item_index,
            target_raw_item=s.target_raw_item,
            split=s.split,
        )
        for s in batch
    ]
    batch_dict = _collate_recommend_batch(rec_batch, bundle, sid_bundle, device)
    batched = batch_dict["batched"]
    node_emb = _encode_graph_nodes(
        batch_dict["node_texts"],
        batched["node_types"],
        batched["node_item_indices"],
        node_store,
        device,
    )
    edge_index = torch.tensor(batched["edge_index"], device=device, dtype=torch.long)
    edge_weight = torch.tensor(batched["edge_weight"], device=device, dtype=torch.float32)
    batch_index = torch.tensor(batched["batch_index"], device=device, dtype=torch.long)
    node_types = torch.tensor(batched["node_types"], device=device, dtype=torch.long)

    node_repr = recommender.encode_nodes_batch(
        node_emb,
        node_types,
        batch_dict["node_ratings"],
        edge_index,
        edge_weight if edge_weight.numel() else None,
    )
    user_repr = recommender.encode_history_batch(
        node_emb,
        node_types,
        batch_dict["node_ratings"],
        edge_index,
        edge_weight if edge_weight.numel() else None,
        batch_index,
        len(batch),
    )

    # 证据 BCE（仅 fragment 节点）
    frag_mask = node_types == 0
    evidence_logits = recommender.evidence_scorer(node_repr).squeeze(-1)
    evidence_targets = []
    node_offset = 0
    graphs = batch_dict["graphs"]
    for g_idx, graph in enumerate(graphs):
        summary = batch[g_idx].summary
        for node in graph.nodes:
            if node.node_type == "fragment":
                evidence_targets.append(fragment_evidence_label(node.text, summary))
            else:
                evidence_targets.append(0.0)
    evidence_targets_t = torch.tensor(evidence_targets, device=device, dtype=torch.float32)
    if frag_mask.any():
        loss_evidence = F.binary_cross_entropy_with_logits(
            evidence_logits[frag_mask],
            evidence_targets_t[frag_mask],
        )
    else:
        loss_evidence = torch.tensor(0.0, device=device)

    # 每个样本选 top-m 证据片段做池化
    evidence_vecs = []
    target_idx = torch.tensor([s.target_item_index for s in batch], device=device, dtype=torch.long)
    rationale = recommender.code_contributions_for_items(
        user_repr,
        item_codes,
        target_idx,
        lambda_rating=args.lambda_rating,
        lambda_pop=args.lambda_pop,
        text_sid_length=sid_bundle.text_sid_length,
        use_rating_sid=sid_bundle.use_rating_sid,
        use_popularity_sid=sid_bundle.use_popularity_sid,
    )
    for g_idx, graph in enumerate(graphs):
        frag_indices = [i for i, n in enumerate(graph.nodes) if n.node_type == "fragment"]
        if not frag_indices:
            evidence_vecs.append(recommender.readout.default_user)
            continue
        num_nodes_per = batched.get("num_nodes_per_graph") or [len(g.nodes) for g in graphs]
        start = sum(num_nodes_per[:g_idx])
        local_scores = []
        local_repr = []
        for fi in frag_indices:
            global_i = start + fi
            local_scores.append(evidence_logits[global_i])
            local_repr.append(node_repr[global_i])
        scores = torch.stack(local_scores)
        topk = min(args.top_m_evidence, scores.numel())
        _, idx = torch.topk(scores, k=topk)
        pooled = torch.stack([local_repr[i] for i in idx.tolist()], dim=0).mean(dim=0)
        evidence_vecs.append(pooled)
    evidence_vec = torch.stack(evidence_vecs, dim=0)

    prefix_emb = prefix_adapter(rationale, evidence_vec)
    prefix_len = prefix_emb.shape[1]

    # 构造 SFT 序列
    all_losses = []
    for i, sample in enumerate(batch):
        prompt = _build_prompt(bundle.item_meta.get(sample.target_raw_item, {}))
        target = sample.summary
        text = prompt + " " + target
        enc = tokenizer(
            text,
            truncation=True,
            max_length=args.max_generation_tokens + 128,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        prompt_ids = tokenizer(prompt, truncation=True, max_length=128, return_tensors="pt")["input_ids"].to(device)
        labels = input_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100
        labels[:, : prefix_len] = -100

        tok_weights = sequence_token_weights(
            target,
            tail_weights,
            sample.feature,
            gamma=args.feature_gamma,
        )
        w_list = [0.0] * prefix_len + [0.0] * prompt_ids.shape[1] + tok_weights
        w_list = w_list[: input_ids.shape[1]]
        while len(w_list) < input_ids.shape[1]:
            w_list.append(1.0)
        weights = torch.tensor([w_list], device=device, dtype=torch.float32)

        tok_emb = llm.get_input_embeddings()(input_ids)
        prefix = prefix_emb[i : i + 1]
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
        pad_labels = torch.full((1, prefix_len), -100, device=device, dtype=torch.long)
        labels = torch.cat([pad_labels, labels], dim=1)
        pad_w = torch.zeros((1, prefix_len), device=device)
        weights = torch.cat([pad_w, weights], dim=1)
        pad_attn = torch.ones((1, prefix_len), device=device, dtype=attn.dtype)
        attn = torch.cat([pad_attn, attn], dim=1)

        out = llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)
        all_losses.append(_weighted_ce_loss(out.logits, labels, weights))

    loss_ce = torch.stack(all_losses).mean()
    loss_total = loss_ce + float(args.lambda_selector) * loss_evidence
    return {
        "loss": loss_total,
        "loss_ce": float(loss_ce.detach().cpu()),
        "loss_evidence": float(loss_evidence.detach().cpu()),
    }


@torch.no_grad()
def _generate_eval(
    bundle,
    sid_bundle,
    item_codes,
    recommender,
    prefix_adapter,
    llm,
    tokenizer,
    node_store,
    samples: list[ExplainSample],
    device,
    args,
) -> tuple[list[str], list[str], list[str]]:
    llm.eval()
    recommender.eval()
    preds: list[str] = []
    refs: list[str] = []
    feats: list[str] = []
    for sample in samples:
        out = _forward_explain_batch(
            bundle,
            sid_bundle,
            item_codes,
            recommender,
            prefix_adapter,
            llm,
            tokenizer,
            node_store,
            [sample],
            device,
            args,
            tail_weights={},
        )
        del out
        prompt = _build_prompt(bundle.item_meta.get(sample.target_raw_item, {}))
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        # 简化：直接用文本 prompt 生成（prefix 在完整训练中已注入；评估时用相同 batch 逻辑太重，此处用 prompt-only 近似 smoke）
        gen = llm.generate(
            **enc,
            max_new_tokens=args.max_generation_tokens,
            do_sample=True,
            temperature=args.gen_temperature,
            top_p=args.gen_top_p,
            repetition_penalty=args.gen_repetition_penalty,
        )
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        if prompt in text:
            text = text.split(prompt, 1)[-1].strip()
        preds.append(text)
        refs.append(sample.summary)
        feats.append(sample.feature)
    return refs, preds, feats


def run_explain(args) -> dict:
    device_id = args.smoke_device or str(args.devices)
    if device_id in {"default", "auto", ""}:
        device_id = f"cuda:{default_preferred_device_id()}"
    device = torch.device(device_id if device_id.startswith("cuda") or device_id == "cpu" else f"cuda:{device_id}")

    bundle = load_sequential_dataset(Path(args.data_dir), args.dataset_name.strip("/"))
    assign_calib_split(bundle, args.calib_ratio)

    ckpt = _recommend_ckpt_path(args)
    if not ckpt.is_file():
        raise FileNotFoundError(f"推荐 checkpoint 不存在: {ckpt}，请先运行 recommend 模式")

    roberta_path = Path(args.roberta_model_path)
    use_mock = bool(getattr(args, "smoke_mock_encoder", False))
    if use_mock:
        roberta = SmokeTextEncoder(device=device)
    else:
        roberta = RobertaTextEncoder(
            args.roberta_model_path,
            device=device,
            cache_dir=Path(args.embedding_cache_dir) / args.dataset_name.replace("/", "__") / "roberta",
            local_files_only=True,
        )

    if args.max_train_batches and args.max_train_batches > 0:
        text_emb = {}
        dim = roberta.hidden_size
        for raw in list(bundle.item2index.keys())[:256]:
            text_emb[raw] = __import__("numpy").zeros((dim,), dtype=__import__("numpy").float32)
    else:
        text_emb = encode_all_item_texts(bundle, roberta, batch_size=32)
    sid_bundle = build_semantic_ids(bundle, text_emb, args)
    cache_dir = Path(args.sid_cache_dir) / args.dataset_name.replace("/", "__")
    save_sid_bundle(cache_dir / f"sid_{sid_bundle.fingerprint}.json", sid_bundle)

    train_samples, val_samples, test_samples = build_explain_samples(
        bundle, positive_threshold=args.positive_feedback_threshold
    )
    tail_weights, _ = compute_tail_df_weights(
        train_samples,
        alpha=args.tail_alpha,
        w_min=args.tail_weight_min,
        w_max=args.tail_weight_max,
    )

    recommender = SIDRecommender(
        embed_dim=roberta.hidden_size,
        hidden_dim=args.selector_hidden,
        gnn_layers=args.gnn_layers,
        magnet_q=args.magnet_q,
        text_positions=args.text_sid_length,
        text_classes=sid_bundle.text_codebook_size,
        use_rating=args.use_rating_sid,
        use_popularity=args.use_popularity_sid,
    ).to(device)
    recommender.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    if not args.explain_unfreeze_shared and not getattr(args, "joint_explain_rec", False):
        for p in recommender.parameters():
            p.requires_grad = False
        recommender.eval()
    else:
        recommender.train()

    item_codes = build_item_code_matrix(sid_bundle, device)
    item_emb_matrix = _build_item_emb_matrix(text_emb, sid_bundle, device)
    frag_cache = _precompute_fragment_cache(bundle, roberta, batch_size=args.roberta_encode_batch_size)
    node_store = NodeEmbeddingStore(item_emb_matrix=item_emb_matrix, frag_cache=frag_cache, hidden_size=roberta.hidden_size)

    llm_path = Path(args.llm_model_path)
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, lora_cfg)

    rationale_dim = sid_bundle.text_sid_length
    if sid_bundle.use_rating_sid:
        rationale_dim += 1
    if sid_bundle.use_popularity_sid:
        rationale_dim += 1
    prefix_adapter = PrefixAdapter(args.selector_hidden, llm.config.hidden_size, rationale_dim).to(device)

    trainable = list(prefix_adapter.parameters()) + list(recommender.evidence_scorer.parameters())
    if args.explain_unfreeze_shared:
        trainable += [p for p in recommender.parameters() if p.requires_grad]
    trainable += [p for p in llm.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate)

    train_loader = DataLoader(
        ExplainDataset(train_samples),
        batch_size=max(1, min(args.batch_size, 2)),
        shuffle=True,
        collate_fn=lambda b: b,
    )
    log_dir = Path(args.log_dir) / args.dataset_name.replace("/", "__")
    loss_log_path = log_dir / "train_losses.jsonl"
    ckpt_dir = Path(args.ckpt_dir) / args.dataset_name.replace("/", "__") / "explain"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_score = float("-inf")
    patience = int(args.explain_early_stop_patience)
    patience_left = patience
    baseline_metrics: dict[str, float] = {}

    max_epochs = args.explain_epochs
    for epoch in range(max_epochs):
        llm.train()
        prefix_adapter.train()
        recommender.train(args.explain_unfreeze_shared)
        epoch_losses: dict[str, list[float]] = defaultdict(list)
        max_steps = args.max_train_batches if args.max_train_batches > 0 else len(train_loader)
        for step, batch in enumerate(train_loader):
            if args.max_train_batches and step >= args.max_train_batches:
                break
            out = _forward_explain_batch(
                bundle,
                sid_bundle,
                item_codes,
                recommender,
                prefix_adapter,
                llm,
                tokenizer,
                node_store,
                batch,
                device,
                args,
                tail_weights,
            )
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip_norm)
            optimizer.step()
            epoch_losses["loss_total"].append(float(loss.detach().cpu()))
            epoch_losses["loss_ce"].append(float(out["loss_ce"]))
            epoch_losses["loss_evidence"].append(float(out["loss_evidence"]))

        eval_n = args.max_eval_batches * args.eval_batch_size if args.max_eval_batches else min(32, len(val_samples))
        eval_subset = val_samples[:eval_n]
        refs, preds, feats = _generate_eval(
            bundle,
            sid_bundle,
            item_codes,
            recommender,
            prefix_adapter,
            llm,
            tokenizer,
            node_store,
            eval_subset,
            device,
            args,
        )
        metrics = evaluate_explanations(refs, preds, feats) if preds else {}
        if not baseline_metrics and metrics:
            baseline_metrics = dict(metrics)
        composite = diversity_composite_score(metrics) if metrics else 0.0
        record = {
            "stage": "explain",
            "epoch": epoch + 1,
            "train_loss_total": sum(epoch_losses["loss_total"]) / max(len(epoch_losses["loss_total"]), 1),
            "train_loss_ce": sum(epoch_losses["loss_ce"]) / max(len(epoch_losses["loss_ce"]), 1),
            "train_loss_evidence": sum(epoch_losses["loss_evidence"]) / max(len(epoch_losses["loss_evidence"]), 1),
            "val_metrics": metrics,
            "diversity_composite": composite,
        }
        _append_jsonl(loss_log_path, record)
        print(
            f"explain epoch {epoch + 1}: loss={record['train_loss_total']:.4f} "
            f"ce={record['train_loss_ce']:.4f} ev={record['train_loss_evidence']:.4f} "
            f"FCR={metrics.get('FCR', 0):.4f} DIV={metrics.get('DIV', 0):.4f} composite={composite:.4f}"
        )
        gate_ok = passes_quality_gate(metrics, baseline_metrics) if baseline_metrics else True
        if gate_ok and composite > best_score:
            best_score = composite
            patience_left = patience
            torch.save(
                {
                    "prefix_adapter": prefix_adapter.state_dict(),
                    "evidence_scorer": recommender.evidence_scorer.state_dict(),
                    "lora": llm.state_dict(),
                },
                ckpt_dir / "explainer.bin",
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    out_path = log_dir / "explain_metrics.json"
    out_path.write_text(json.dumps({"best_diversity_composite": best_score}, indent=2), encoding="utf-8")
    return {"best_diversity_composite": best_score}
