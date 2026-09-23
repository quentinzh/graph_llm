#!/usr/bin/env python
"""Semantic ID 推荐训练与评估（GNN -> selector，不加载解释 LLM）。"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from graph_llm.dataload.embeddings import RobertaTextEncoder, SmokeTextEncoder, _text_hash
from graph_llm.dataload.history_graph import (
    _split_fragments,
    batch_history_graphs,
    build_history_user_graph,
)
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
    graph_search_seed_items,
    graph_search_top_k,
    recovery_rate,
)
from graph_llm.models.sid_recommender import (
    SIDRecommender,
    build_code_log_prior_tensors,
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


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _popularity_scores(sid_bundle: SemanticIDBundle, num_items: int, device: torch.device) -> torch.Tensor:
    """按校准快照流行度（distinct_users）的全局排序分。"""
    scores = torch.zeros(num_items, device=device, dtype=torch.float32)
    for rec in sid_bundle.items.values():
        if 0 <= rec.item_index < num_items:
            scores[rec.item_index] = math.log1p(float(rec.distinct_users))
    return scores


def _random_user_scores(num_items: int, device: torch.device, rng: random.Random) -> torch.Tensor:
    perm = torch.tensor(rng.sample(range(num_items), num_items), device=device, dtype=torch.float32)
    return perm


@dataclass
class RecommendSample:
    interaction_id: int
    user_index: int
    target_item_index: int
    target_raw_item: str
    split: str  # train_rec | val | test


@dataclass
class NodeEmbeddingStore:
    """训练/评估热循环用：商品矩阵 + 片段内存缓存，避免每 batch RoBERTa/磁盘读。"""

    item_emb_matrix: torch.Tensor
    frag_cache: dict[str, torch.Tensor]
    hidden_size: int


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


def _collect_unique_fragments(bundle: SequentialDatasetBundle) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for rec in bundle.interactions:
        frags = _split_fragments(rec.review_text) or _split_fragments(rec.summary)
        for frag in frags:
            key = _text_hash(frag)
            if key not in seen:
                seen.add(key)
                ordered.append(frag)
    return ordered


def _precompute_fragment_cache(
    bundle: SequentialDatasetBundle,
    roberta: RobertaTextEncoder | SmokeTextEncoder,
    *,
    batch_size: int = 64,
) -> dict[str, torch.Tensor]:
    frags = _collect_unique_fragments(bundle)
    cache: dict[str, torch.Tensor] = {}
    if not frags:
        return cache
    pbar = tqdm(total=len(frags), desc="RoBERTa encode fragments", unit="frag")
    for start in range(0, len(frags), batch_size):
        chunk = frags[start : start + batch_size]
        vecs = roberta.encode_texts(chunk, batch_size=batch_size, use_cache=False, show_progress=False)
        for i, frag in enumerate(chunk):
            cache[_text_hash(frag)] = vecs[i].detach().cpu().float()
        pbar.update(len(chunk))
    pbar.close()
    return cache


def _build_item_emb_matrix(
    text_emb: dict[str, np.ndarray],
    sid_bundle: SemanticIDBundle,
    device: torch.device,
) -> torch.Tensor:
    rows = [
        text_emb[sid_bundle.item_index_to_raw[i]] for i in range(len(sid_bundle.item_index_to_raw))
    ]
    return torch.tensor(np.stack(rows), dtype=torch.float32, device=device)


def _encode_graph_nodes(
    node_texts: list[str],
    node_types: np.ndarray,
    node_item_indices: np.ndarray,
    store: NodeEmbeddingStore,
    device: torch.device,
) -> torch.Tensor:
    n = len(node_texts)
    if n == 0:
        return torch.empty((0, store.hidden_size), device=device)
    out = torch.empty((n, store.hidden_size), dtype=torch.float32, device=device)
    item_mask = node_types == 1
    frag_mask = ~item_mask
    if item_mask.any():
        idx = torch.tensor(node_item_indices[item_mask], device=device, dtype=torch.long)
        out[item_mask] = store.item_emb_matrix[idx]
    if frag_mask.any():
        frag_rows = []
        for i in np.where(frag_mask)[0]:
            key = _text_hash(node_texts[i])
            vec = store.frag_cache.get(key)
            if vec is None:
                vec = torch.zeros(store.hidden_size, dtype=torch.float32)
            frag_rows.append(vec)
        out[frag_mask] = torch.stack(frag_rows).to(device)
    return out


def _forward_batch(
    model: SIDRecommender,
    node_store: NodeEmbeddingStore,
    batch_dict: dict,
    device: torch.device,
    args,
    item_codes: torch.Tensor,
    code_log_priors: list[torch.Tensor],
) -> dict[str, torch.Tensor | float]:
    batched = batch_dict["batched"]
    node_emb = _encode_graph_nodes(
        batch_dict["node_texts"],
        batched["node_types"],
        batched["node_item_indices"],
        node_store,
        device,
    )
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
    targets = {
        "text": batch_dict["text_targets"],
        "rating": batch_dict["rating_targets"],
        "pop": batch_dict["pop_targets"],
    }
    logits = model.forward_sid_logits(user_repr)
    rec_mode = getattr(args, "rec_loss", "item")
    loss_sid = torch.tensor(0.0, device=device)
    loss_item = torch.tensor(0.0, device=device)
    if rec_mode in {"sid", "both"}:
        loss_sid = model.sid_loss(
            logits,
            targets,
            lambda_rating=args.lambda_rating,
            lambda_pop=args.lambda_pop,
        )
    if rec_mode in {"item", "both"}:
        target_idx = torch.tensor(
            [s.target_item_index for s in batch_dict["samples"]],
            device=device,
            dtype=torch.long,
        )
        loss_item = model.item_rec_loss(
            user_repr,
            target_idx,
            item_codes,
            lambda_rating=args.lambda_rating,
            lambda_pop=args.lambda_pop,
            text_sid_length=args.text_sid_length,
            use_rating_sid=args.use_rating_sid,
            use_popularity_sid=args.use_popularity_sid,
            temperature=args.rec_temperature,
            code_log_priors=code_log_priors if args.score_pmi_lambda > 0 else None,
            pmi_lambda=args.score_pmi_lambda,
        )
    if rec_mode == "sid":
        total = loss_sid
    elif rec_mode == "item":
        total = loss_item
    else:
        total = loss_sid + loss_item
    return {
        "loss": total,
        "loss_sid": float(loss_sid.detach().cpu()),
        "loss_item": float(loss_item.detach().cpu()),
    }


def _rank_users(
    model: SIDRecommender | None,
    node_store: NodeEmbeddingStore | None,
    samples: list[RecommendSample],
    bundle: SequentialDatasetBundle,
    sid_bundle: SemanticIDBundle,
    item_codes: torch.Tensor,
    neighbor_index: ItemNeighborIndex,
    args,
    device: torch.device,
    popularity_scores: torch.Tensor | None = None,
    code_log_priors: list[torch.Tensor] | None = None,
    desc: str = "eval",
    leave: bool = True,
    search_mode: str | None = None,
) -> tuple[list[list[int]], dict[str, float]]:
    ks = (5, 10, 20)
    targets = []
    ranked_lists = []
    recoveries = []
    allow_repeat = args.allow_repeat_recommend
    rng = random.Random(args.seed)
    lambda_rating = args.lambda_rating
    lambda_pop = args.lambda_pop
    baseline = getattr(args, "eval_baseline", "model")
    mode = search_mode or args.search_mode
    num_items = item_codes.shape[0]
    rank_positions: list[float] = []
    score_stds: list[float] = []

    loader = DataLoader(
        RecommendDataset(samples),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )
    for batch in tqdm(loader, desc=desc, leave=leave):
        if isinstance(batch, RecommendSample):
            batch = [batch]
        batch_dict = _collate_recommend_batch(list(batch), bundle, sid_bundle, device)
        if baseline == "model":
            assert model is not None and node_store is not None
            with torch.no_grad():
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
                code_log_priors=code_log_priors,
                pmi_lambda=args.score_pmi_lambda,
            )
        elif baseline == "popularity":
            assert popularity_scores is not None
            batch_scores = popularity_scores.unsqueeze(0).expand(len(batch), -1).clone()
        else:
            batch_scores = torch.stack(
                [_random_user_scores(num_items, device, rng) for _ in range(len(batch))],
                dim=0,
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

            scored = sf()
            if scored.numel() > 0:
                finite = scored[torch.isfinite(scored)]
                if finite.numel() > 0:
                    score_stds.append(float(finite.std(unbiased=False).item()))
                tgt = sample.target_item_index
                if 0 <= tgt < scored.numel() and torch.isfinite(scored[tgt]):
                    rank_positions.append(float((scored > scored[tgt]).sum().item() + 1))

            exact_items, _ = exact_top_k(sf, set(), max(ks))
            if mode == "graph" and baseline == "model":
                hist_items = list(history_item_indices(hist))
                seeds = graph_search_seed_items(
                    neighbor_index,
                    hist_items,
                    fallback_item=sample.target_item_index,
                )
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
    metrics["eval_samples"] = float(len(targets))
    if rank_positions:
        metrics["target_rank_mean"] = float(sum(rank_positions) / len(rank_positions))
        metrics["target_rank_median"] = float(sorted(rank_positions)[len(rank_positions) // 2])
    if score_stds:
        metrics["score_std_mean"] = float(sum(score_stds) / len(score_stds))
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
    code_log_priors = build_code_log_prior_tensors(sid_bundle, device)
    popularity_scores = _popularity_scores(sid_bundle, len(sid_bundle.item_index_to_raw), device)
    item_vectors = np.stack(
        [text_emb[sid_bundle.item_index_to_raw[i]] for i in range(len(sid_bundle.item_index_to_raw))]
    )
    if args.max_train_batches and args.max_train_batches > 0:
        args.search_mode = "exact"
    neighbor_index = ItemNeighborIndex(item_vectors, neighbors=args.search_neighbors, seed=args.seed)

    item_emb_matrix = _build_item_emb_matrix(text_emb, sid_bundle, device)
    frag_cache = _precompute_fragment_cache(bundle, roberta, batch_size=args.roberta_encode_batch_size)
    node_store = NodeEmbeddingStore(
        item_emb_matrix=item_emb_matrix,
        frag_cache=frag_cache,
        hidden_size=roberta.hidden_size,
    )

    log_dir = Path(args.log_dir) / args.dataset_name.replace("/", "__")
    loss_log_path = log_dir / "train_losses.jsonl"
    ckpt_dir = Path(args.ckpt_dir) / args.dataset_name.replace("/", "__") / "recommend"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    baseline = getattr(args, "eval_baseline", "model")
    if baseline != "model":
        eval_limit = args.max_eval_batches
        eval_val = val_samples[: eval_limit * args.eval_batch_size] if eval_limit else val_samples
        eval_test = test_samples[: eval_limit * args.eval_batch_size] if eval_limit else test_samples
        _, val_metrics = _rank_users(
            None,
            None,
            eval_val,
            bundle,
            sid_bundle,
            item_codes,
            neighbor_index,
            args,
            device,
            popularity_scores=popularity_scores,
            code_log_priors=code_log_priors,
            desc=f"val[{baseline}]",
            search_mode="exact",
        )
        _, test_metrics = _rank_users(
            None,
            None,
            eval_test,
            bundle,
            sid_bundle,
            item_codes,
            neighbor_index,
            args,
            device,
            popularity_scores=popularity_scores,
            code_log_priors=code_log_priors,
            desc=f"test[{baseline}]",
            search_mode="exact",
        )
        out = {
            "baseline": baseline,
            "val": val_metrics,
            "test": test_metrics,
            "sid_fingerprint": sid_bundle.fingerprint,
        }
        log_path = log_dir / "recommend_metrics.json"
        log_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return out

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
    patience_left = int(args.early_stop_patience)

    max_batches = args.max_train_batches if args.max_train_batches > 0 else None
    train_steps_per_epoch = len(train_loader)
    if max_batches:
        train_steps_per_epoch = min(train_steps_per_epoch, max_batches)
    epoch_pbar = tqdm(range(args.epochs), desc="epochs", unit="ep")
    for epoch in epoch_pbar:
        model.train()
        epoch_losses: dict[str, list[float]] = defaultdict(list)
        for step, batch in enumerate(
            tqdm(
                train_loader,
                total=train_steps_per_epoch,
                desc=f"train e{epoch + 1}",
                leave=False,
            )
        ):
            if max_batches and step >= max_batches:
                break
            if isinstance(batch, RecommendSample):
                batch = [batch]
            batch_dict = _collate_recommend_batch(list(batch), bundle, sid_bundle, device)
            out_loss = _forward_batch(
                model, node_store, batch_dict, device, args, item_codes, code_log_priors
            )
            loss = out_loss["loss"]
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            epoch_losses["loss_total"].append(float(loss.detach().cpu()))
            epoch_losses["loss_sid"].append(float(out_loss["loss_sid"]))
            epoch_losses["loss_item"].append(float(out_loss["loss_item"]))

        model.eval()
        eval_limit_batches = args.max_eval_batches
        eval_val = val_samples[: eval_limit_batches * args.eval_batch_size] if eval_limit_batches else val_samples
        _, val_metrics = _rank_users(
            model,
            node_store,
            eval_val,
            bundle,
            sid_bundle,
            item_codes,
            neighbor_index,
            args,
            device,
            popularity_scores=popularity_scores,
            code_log_priors=code_log_priors,
            desc=f"val[e{epoch + 1}]",
            leave=False,
            search_mode="exact",
        )
        ndcg10 = float(val_metrics.get("NDCG@10", 0.0))
        record = {
            "stage": "recommend",
            "epoch": epoch + 1,
            "rec_loss_mode": getattr(args, "rec_loss", "item"),
            "train_loss_total": sum(epoch_losses["loss_total"]) / max(len(epoch_losses["loss_total"]), 1),
            "train_loss_sid": sum(epoch_losses["loss_sid"]) / max(len(epoch_losses["loss_sid"]), 1),
            "train_loss_item": sum(epoch_losses["loss_item"]) / max(len(epoch_losses["loss_item"]), 1),
            "val": val_metrics,
        }
        _append_jsonl(loss_log_path, record)
        print(
            f"epoch {epoch + 1} losses: total={record['train_loss_total']:.4f} "
            f"sid={record['train_loss_sid']:.4f} item={record['train_loss_item']:.4f} "
            f"val NDCG@10={ndcg10:.6f} HR@10={val_metrics.get('HR@10', 0):.6f}"
        )
        if ndcg10 > best_ndcg + 1e-9:
            best_ndcg = ndcg10
            patience_left = int(args.early_stop_patience)
            torch.save(model.state_dict(), ckpt_dir / "sid_recommender.bin")
            print(f"save recommend checkpoint (valid NDCG@10={ndcg10:.6f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stop at epoch {epoch + 1} (patience={args.early_stop_patience})")
                break

    epoch_pbar.close()
    model.eval()
    if (ckpt_dir / "sid_recommender.bin").is_file():
        model.load_state_dict(torch.load(ckpt_dir / "sid_recommender.bin", map_location=device, weights_only=True))

    if getattr(args, "auto_pmi_lambda", False):
        eval_limit = args.max_eval_batches
        eval_val = val_samples[: eval_limit * args.eval_batch_size] if eval_limit else val_samples
        saved_lambda = float(args.score_pmi_lambda)
        args.score_pmi_lambda = 0.0
        _, base_val = _rank_users(
            model,
            node_store,
            eval_val,
            bundle,
            sid_bundle,
            item_codes,
            neighbor_index,
            args,
            device,
            popularity_scores=popularity_scores,
            code_log_priors=code_log_priors,
            desc="val[pmi_base]",
            search_mode="exact",
        )
        base_ndcg = float(base_val.get("NDCG@10", 0.0))
        best_lam = 0.0
        best_ndcg = base_ndcg
        for lam in (0.25, 0.5, 1.0):
            args.score_pmi_lambda = lam
            _, vm = _rank_users(
                model,
                node_store,
                eval_val,
                bundle,
                sid_bundle,
                item_codes,
                neighbor_index,
                args,
                device,
                popularity_scores=popularity_scores,
                code_log_priors=code_log_priors,
                desc=f"val[pmi_{lam}]",
                search_mode="exact",
            )
            nd = float(vm.get("NDCG@10", 0.0))
            if nd >= base_ndcg - 1e-9 and nd >= best_ndcg:
                best_ndcg = nd
                best_lam = lam
        args.score_pmi_lambda = best_lam if best_lam > 0 else saved_lambda
        print(f"auto PMI: base NDCG@10={base_ndcg:.6f} chosen lambda={args.score_pmi_lambda}")

    model.eval()
    eval_limit = args.max_eval_batches
    eval_val = val_samples[: eval_limit * args.eval_batch_size] if eval_limit else val_samples
    eval_test = test_samples[: eval_limit * args.eval_batch_size] if eval_limit else test_samples
    _, val_metrics = _rank_users(
        model,
        node_store,
        eval_val,
        bundle,
        sid_bundle,
        item_codes,
        neighbor_index,
        args,
        device,
        popularity_scores=popularity_scores,
        code_log_priors=code_log_priors,
        desc="val[final]",
        search_mode="exact",
    )
    _, test_metrics = _rank_users(
        model,
        node_store,
        eval_test,
        bundle,
        sid_bundle,
        item_codes,
        neighbor_index,
        args,
        device,
        popularity_scores=popularity_scores,
        code_log_priors=code_log_priors,
        desc="test[final]",
        search_mode="exact",
    )

    out = {
        "val": val_metrics,
        "test": test_metrics,
        "sid_fingerprint": sid_bundle.fingerprint,
        "score_pmi_lambda": args.score_pmi_lambda,
        "rec_loss": getattr(args, "rec_loss", "item"),
    }
    log_path = log_dir / "recommend_metrics.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out
