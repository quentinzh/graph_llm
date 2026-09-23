#!/usr/bin/env python
"""Semantic ID 推荐训练与评估（GNN -> selector，不加载解释 LLM）。"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from graph_llm.dataload.embeddings import RobertaTextEncoder, SmokeTextEncoder
from graph_llm.dataload.history_graph import batch_history_graphs, build_history_user_graph
from graph_llm.dataload.sequential_data import (
    InteractionRecord,
    SequentialDatasetBundle,
    assign_calib_split,
    history_before,
    history_item_indices,
    load_sequential_dataset,
)
from graph_llm.dataload.semantic_id import (
    SemanticIDBundle,
    build_semantic_ids,
    encode_all_item_texts,
    save_sid_bundle,
)
from graph_llm.metrics.rec_ranking import evaluate_ranking
from graph_llm.models.item_search import (
    ItemNeighborIndex,
    exact_top_k,
    graph_search_top_k,
    recovery_rate,
)
from graph_llm.models.sid_recommender import (
    SIDRecommender,
    build_item_code_matrix,
    logits_to_log_probs,
    vectorized_item_scores,
)


def default_preferred_device_id() -> int:
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            return 1
        return 0
    raise RuntimeError("No CUDA devices are available.")


@dataclass
class RecommendSample:
    interaction_id: int
    user_index: int
    target_item_index: int
    target_raw_item: str
    split: str  # train_rec | val | test


class RecommendDataset(Dataset):
    def __init__(self, samples: list[RecommendSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _passes_positive_filter(rec: InteractionRecord, threshold: float) -> bool:
    if threshold <= 0:
        return True
    if rec.rating_raw is None:
        return False
    return rec.rating_raw >= threshold


def build_recommend_samples(
    bundle: SequentialDatasetBundle,
    *,
    positive_threshold: float,
) -> tuple[list[RecommendSample], list[RecommendSample], list[RecommendSample]]:
    train_samples: list[RecommendSample] = []
    for iid in sorted(bundle.rec_train_ids):
        rec = bundle.interaction_by_id[iid]
        if not _passes_positive_filter(rec, positive_threshold):
            continue
        train_samples.append(
            RecommendSample(
                interaction_id=iid,
                user_index=rec.user_index,
                target_item_index=rec.item_index,
                target_raw_item=rec.raw_item,
                split="train_rec",
            )
        )
    val_samples = []
    for iid in bundle.val_sample_ids:
        rec = bundle.interaction_by_id[iid]
        if not _passes_positive_filter(rec, positive_threshold):
            continue
        val_samples.append(
            RecommendSample(
                interaction_id=iid,
                user_index=rec.user_index,
                target_item_index=rec.item_index,
                target_raw_item=rec.raw_item,
                split="val",
            )
        )
    test_samples = []
    for iid in bundle.test_sample_ids:
        rec = bundle.interaction_by_id[iid]
        if not _passes_positive_filter(rec, positive_threshold):
            continue
        test_samples.append(
            RecommendSample(
                interaction_id=iid,
                user_index=rec.user_index,
                target_item_index=rec.item_index,
                target_raw_item=rec.raw_item,
                split="test",
            )
        )
    return train_samples, val_samples, test_samples


def _history_for_sample(bundle: SequentialDatasetBundle, sample: RecommendSample) -> list[InteractionRecord]:
    rec = bundle.interaction_by_id[sample.interaction_id]
    include_val = sample.split == "test"
    return history_before(
        bundle,
        sample.user_index,
        rec.timestamp,
        exclude_interaction_id=sample.interaction_id,
        include_val=include_val,
    )


def _collate_recommend_batch(
    batch: list[RecommendSample],
    bundle: SequentialDatasetBundle,
    sid_bundle: SemanticIDBundle,
    device: torch.device,
) -> dict:
    graphs = [build_history_user_graph(_history_for_sample(bundle, s), max_nodes=512) for s in batch]
    batched = batch_history_graphs(graphs)
    texts = []
    node_ratings = []
    for g in graphs:
        for node in g.nodes:
            texts.append(node.text if node.node_type == "fragment" else node.text)
            if node.node_type == "item":
                r = node.rating_raw
                bucket = 0 if r is None else max(1, min(5, int(round(r))))
                node_ratings.append(bucket)
            else:
                node_ratings.append(0)
    text_targets = []
    rating_targets = []
    pop_targets = []
    for s in batch:
        rec = sid_bundle.items[s.target_raw_item]
        text_targets.append(list(rec.text_codes))
        rating_targets.append(rec.rating_code)
        pop_targets.append(rec.pop_code)
    histories = [_history_for_sample(bundle, s) for s in batch]
    return {
        "samples": batch,
        "histories": histories,
        "graphs": graphs,
        "batched": batched,
        "node_texts": texts,
        "node_ratings": torch.tensor(node_ratings, dtype=torch.long, device=device),
        "text_targets": torch.tensor(text_targets, dtype=torch.long, device=device),
        "rating_targets": torch.tensor(rating_targets, dtype=torch.long, device=device),
        "pop_targets": torch.tensor(pop_targets, dtype=torch.long, device=device),
    }


def _encode_graph_nodes(roberta: RobertaTextEncoder, texts: list[str], device: torch.device) -> torch.Tensor:
    if not texts:
        return torch.empty((0, roberta.hidden_size), device=device)
    return roberta.encode_texts(texts, batch_size=32)


def _forward_batch(
    model: SIDRecommender,
    roberta: RobertaTextEncoder,
    batch_dict: dict,
    device: torch.device,
    args,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    node_emb = _encode_graph_nodes(roberta, batch_dict["node_texts"], device)
    batched = batch_dict["batched"]
    if node_emb.numel() == 0:
        user_repr = model.readout.default_user.unsqueeze(0).expand(len(batch_dict["samples"]), -1)
    else:
        edge_index = torch.tensor(batched["edge_index"], device=device, dtype=torch.long)
        edge_weight = torch.tensor(batched["edge_weight"], device=device, dtype=torch.float32)
        batch_index = torch.tensor(batched["batch_index"], device=device, dtype=torch.long)
        node_types = torch.tensor(batched["node_types"], device=device, dtype=torch.long)
        user_repr = model.encode_history_batch(
            node_emb,
            node_types,
            batch_dict["node_ratings"],
            edge_index,
            edge_weight if edge_weight.numel() else None,
            batch_index,
            len(batch_dict["samples"]),
        )
    logits = model.forward_sid_logits(user_repr)
    targets = {
        "text": batch_dict["text_targets"],
        "rating": batch_dict["rating_targets"],
        "pop": batch_dict["pop_targets"],
    }
    loss = model.sid_loss(
        logits,
        targets,
        lambda_rating=args.lambda_rating,
        lambda_pop=args.lambda_pop,
    )
    return logits, loss


def _rank_users(
    model: SIDRecommender,
    roberta: RobertaTextEncoder,
    samples: list[RecommendSample],
    bundle: SequentialDatasetBundle,
    sid_bundle: SemanticIDBundle,
    item_codes: torch.Tensor,
    neighbor_index: ItemNeighborIndex,
    args,
    device: torch.device,
) -> tuple[list[list[int]], dict[str, float]]:
    ks = (5, 10, 20)
    targets = []
    ranked_lists = []
    recoveries = []
    allow_repeat = args.allow_repeat_recommend
    rng = random.Random(args.seed)
    lambda_rating = args.lambda_rating
    lambda_pop = args.lambda_pop

    loader = DataLoader(
        RecommendDataset(samples),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )
    for batch in loader:
        if isinstance(batch, RecommendSample):
            batch = [batch]
        batch_dict = _collate_recommend_batch(list(batch), bundle, sid_bundle, device)
        with torch.no_grad():
            node_emb = _encode_graph_nodes(roberta, batch_dict["node_texts"], device)
            batched = batch_dict["batched"]
            edge_index = torch.tensor(batched["edge_index"], device=device, dtype=torch.long)
            edge_weight = torch.tensor(batched["edge_weight"], device=device, dtype=torch.float32)
            batch_index = torch.tensor(batched["batch_index"], device=device, dtype=torch.long)
            node_types = torch.tensor(batched["node_types"], device=device, dtype=torch.long)
            if node_emb.numel() == 0:
                user_repr = model.readout.default_user.unsqueeze(0).expand(len(batch), -1)
            else:
                user_repr = model.encode_history_batch(
                    node_emb,
                    node_types,
                    batch_dict["node_ratings"],
                    edge_index,
                    edge_weight if edge_weight.numel() else None,
                    batch_index,
                    len(batch),
                )
            logits = model.forward_sid_logits(user_repr)
            log_probs = logits_to_log_probs(logits)

        batch_scores = vectorized_item_scores(
            log_probs,
            item_codes,
            text_sid_length=sid_bundle.text_sid_length,
            use_rating_sid=sid_bundle.use_rating_sid,
            use_popularity_sid=sid_bundle.use_popularity_sid,
            lambda_rating=lambda_rating,
            lambda_pop=lambda_pop,
        )
        for local_i, sample in enumerate(batch):
            hist = batch_dict["histories"][local_i]
            exclude = set() if allow_repeat else history_item_indices(hist)

            def sf(scores=batch_scores[local_i].clone()):
                if exclude:
                    scores = scores.clone()
                    ex = torch.tensor(sorted(exclude), device=scores.device, dtype=torch.long)
                    scores[ex] = float("-inf")
                return scores

            exact_items, _ = exact_top_k(sf, set(), max(ks))
            if args.search_mode == "graph":
                seeds = list(history_item_indices(hist))[:8]
                if not seeds:
                    seeds = [sample.target_item_index]
                approx_items, _ = graph_search_top_k(
                    sf,
                    neighbor_index,
                    seeds,
                    exclude=exclude,
                    k=max(ks),
                    rounds=args.search_rounds,
                    max_candidates=args.search_candidates,
                    rng=rng,
                )
                recoveries.append(recovery_rate(exact_items, approx_items, max(ks)))
                ranked = approx_items
            else:
                ranked = exact_items
            targets.append(sample.target_item_index)
            ranked_lists.append(ranked)

    metrics = evaluate_ranking(targets, ranked_lists, ks=ks)
    if recoveries:
        metrics["search_recovery@maxk"] = float(sum(recoveries) / len(recoveries))
    return ranked_lists, metrics


def run_recommend(args) -> dict:
    device_id = args.smoke_device or str(args.devices)
    if device_id in {"default", "auto", ""}:
        device_id = f"cuda:{default_preferred_device_id()}"
    device = torch.device(device_id if device_id.startswith("cuda") or device_id == "cpu" else f"cuda:{device_id}")

    bundle = load_sequential_dataset(Path(args.data_dir), args.dataset_name.strip("/"))
    assign_calib_split(bundle, args.calib_ratio)

    cache_dir = Path(args.sid_cache_dir) / args.dataset_name.replace("/", "__")
    cache_dir.mkdir(parents=True, exist_ok=True)

    roberta_path = Path(args.roberta_model_path)
    use_mock = bool(getattr(args, "smoke_mock_encoder", False))
    if use_mock:
        print("WARNING: 使用 smoke_mock_encoder 确定性向量，正式实验请加载 RoBERTa-base")
        roberta = SmokeTextEncoder(device=device)
    else:
        if not roberta_path.is_dir() or not (roberta_path / "config.json").is_file():
            raise FileNotFoundError(
                f"RoBERTa 未找到: {roberta_path}。请运行: "
                f"bash {Path(__file__).resolve().parent.parent / 'aux' / 'download_roberta_base.sh'}"
            )
        roberta = RobertaTextEncoder(
            args.roberta_model_path,
            device=device,
            cache_dir=Path(args.embedding_cache_dir) / args.dataset_name.replace("/", "__") / "roberta",
            local_files_only=True,
        )

    if args.max_train_batches and args.max_train_batches > 0:
        needed_raw = {bundle.interaction_by_id[i].raw_item for i in bundle.calib_ids}
        for iid in list(bundle.rec_train_ids)[: args.max_train_batches * args.batch_size * 4]:
            needed_raw.add(bundle.interaction_by_id[iid].raw_item)
        for iid in bundle.val_sample_ids[: args.max_eval_batches * args.eval_batch_size if args.max_eval_batches else 32]:
            needed_raw.add(bundle.interaction_by_id[iid].raw_item)
        subset = {k: bundle.item_meta[k] for k in needed_raw if k in bundle.item_meta}
        mini_bundle = bundle
        text_emb = {}
        from graph_llm.dataload.sequential_data import item_catalog_text

        texts = [item_catalog_text(subset.get(r, {})) for r in sorted(needed_raw)]
        vecs = roberta.encode_texts(texts, batch_size=32)
        for i, raw in enumerate(sorted(needed_raw)):
            text_emb[raw] = vecs[i].detach().cpu().numpy()
        # 未编码商品用零向量占位（仅 smoke）
        dim = roberta.hidden_size
        for raw in bundle.item2index:
            if raw not in text_emb:
                text_emb[raw] = np.zeros((dim,), dtype=np.float32)
    else:
        text_emb = encode_all_item_texts(bundle, roberta, batch_size=32)
    sid_bundle = build_semantic_ids(bundle, text_emb, args)
    save_sid_bundle(cache_dir / f"sid_{sid_bundle.fingerprint}.json", sid_bundle)

    train_samples, val_samples, test_samples = build_recommend_samples(
        bundle,
        positive_threshold=args.positive_feedback_threshold,
    )

    codebook_size = sid_bundle.text_codebook_size
    model = SIDRecommender(
        embed_dim=roberta.hidden_size,
        hidden_dim=args.selector_hidden,
        gnn_layers=args.gnn_layers,
        magnet_q=args.magnet_q,
        text_positions=args.text_sid_length,
        text_classes=codebook_size,
        use_rating=args.use_rating_sid,
        use_popularity=args.use_popularity_sid,
    ).to(device)

    item_codes = build_item_code_matrix(sid_bundle, device)
    item_vectors = np.stack(
        [text_emb[sid_bundle.item_index_to_raw[i]] for i in range(len(sid_bundle.item_index_to_raw))]
    )
    if args.max_train_batches and args.max_train_batches > 0:
        args.search_mode = "exact"
    neighbor_index = ItemNeighborIndex(item_vectors, neighbors=args.search_neighbors, seed=args.seed)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    def _collate_samples(batch: list[RecommendSample]) -> list[RecommendSample]:
        return batch

    train_loader = DataLoader(
        RecommendDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_samples,
    )

    ckpt_dir = Path(args.ckpt_dir) / args.dataset_name.replace("/", "__") / "recommend"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ndcg = float("-inf")

    max_batches = args.max_train_batches if args.max_train_batches > 0 else None
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            if max_batches and step >= max_batches:
                break
            if isinstance(batch, RecommendSample):
                batch = [batch]
            batch_dict = _collate_recommend_batch(list(batch), bundle, sid_bundle, device)
            _, loss = _forward_batch(model, roberta, batch_dict, device, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        _, val_metrics = _rank_users(
            model,
            roberta,
            val_samples[: args.max_eval_batches * args.eval_batch_size]
            if args.max_eval_batches
            else val_samples,
            bundle,
            sid_bundle,
            item_codes,
            neighbor_index,
            args,
            device,
        )
        ndcg10 = float(val_metrics.get("NDCG@10", 0.0))
        if ndcg10 > best_ndcg:
            best_ndcg = ndcg10
            torch.save(model.state_dict(), ckpt_dir / "sid_recommender.bin")
            print(f"save recommend checkpoint (valid NDCG@10={ndcg10:.6f})")

    model.eval()
    if (ckpt_dir / "sid_recommender.bin").is_file():
        model.load_state_dict(torch.load(ckpt_dir / "sid_recommender.bin", map_location=device, weights_only=True))
    eval_limit = args.max_eval_batches
    eval_val = val_samples[: eval_limit * args.eval_batch_size] if eval_limit else val_samples
    eval_test = test_samples[: eval_limit * args.eval_batch_size] if eval_limit else test_samples
    _, val_metrics = _rank_users(
        model, roberta, eval_val, bundle, sid_bundle, item_codes, neighbor_index, args, device
    )
    _, test_metrics = _rank_users(
        model, roberta, eval_test, bundle, sid_bundle, item_codes, neighbor_index, args, device
    )

    out = {"val": val_metrics, "test": test_metrics, "sid_fingerprint": sid_bundle.fingerprint}
    log_path = Path(args.log_dir) / args.dataset_name / "recommend_metrics.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out
