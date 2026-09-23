"""商品 Semantic ID：RoBERTa 文本 OPQ+PQ、评分码、流行度码与缓存。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from graph_llm.dataload.sequential_data import (
    InteractionRecord,
    SequentialDatasetBundle,
    item_catalog_text,
)


@dataclass
class ItemSIDRecord:
    raw_item: str
    item_index: int
    text_codes: tuple[int, ...]
    rating_code: int
    pop_code: int
    smooth_rating: float | None
    raw_rating_count: int
    distinct_users: int


@dataclass
class SemanticIDBundle:
    fingerprint: str
    text_sid_length: int
    text_codebook_size: int
    use_rating_sid: bool
    use_popularity_sid: bool
    items: dict[str, ItemSIDRecord]
    item_index_to_raw: dict[int, str]
    # 可训练码向量初始化用：每段 text 码本中心（numpy）
    text_codebooks: list[np.ndarray]
    rating_bucket_edges: list[float]
    pop_bucket_edges: list[float]
    global_mean_rating: float | None
    collision_rate: float

    def sid_length(self) -> int:
        n = self.text_sid_length
        if self.use_rating_sid:
            n += 1
        if self.use_popularity_sid:
            n += 1
        return n

    def codes_for_item(self, raw_item: str) -> tuple[int, ...]:
        rec = self.items[raw_item]
        codes = list(rec.text_codes)
        if self.use_rating_sid:
            codes.append(rec.rating_code)
        if self.use_popularity_sid:
            codes.append(rec.pop_code)
        return tuple(codes)

    def codes_for_index(self, item_index: int) -> tuple[int, ...]:
        raw = self.item_index_to_raw[item_index]
        return self.codes_for_item(raw)

    def target_tensor(self, raw_item: str, device: torch.device) -> torch.Tensor:
        codes = self.codes_for_item(raw_item)
        return torch.tensor(codes, device=device, dtype=torch.long)


def _fingerprint_payload(args, n_calib: int) -> str:
    payload = {
        "dataset": args.dataset_name,
        "text_len": args.text_sid_length,
        "codebook": args.text_codebook_size,
        "quant_seed": args.quant_seed,
        "rating": args.use_rating_sid,
        "pop": args.use_popularity_sid,
        "calib_ratio": args.calib_ratio,
        "rating_min": args.rating_min,
        "rating_max": args.rating_max,
        "alpha": args.rating_smooth_alpha,
        "n_calib": n_calib,
        "roberta": args.roberta_model_path,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _bucket_uniform(value: float, edges: list[float]) -> int:
    """value in [0,1] -> 1..8；上端点进第 8 桶。"""
    if value >= 1.0:
        return 8
    for idx, edge in enumerate(edges):
        if value < edge:
            return idx + 1
    return 8


def _build_rating_pop(
    calib_records: list[InteractionRecord],
    all_items: list[str],
    *,
    rating_min: float,
    rating_max: float,
    alpha: float,
) -> tuple[
    dict[str, float],
    dict[str, int],
    dict[str, int],
    list[float],
    float | None,
    list[float],
    dict[str, int],
]:
    # 用户-商品去重：取最后一次
    pair_last: dict[tuple[str, str], InteractionRecord] = {}
    for rec in calib_records:
        pair_last[(rec.raw_user, rec.raw_item)] = rec

    item_sum: dict[str, float] = {}
    item_cnt: dict[str, int] = {}
    for (_u, item), rec in pair_last.items():
        if rec.rating_raw is None:
            continue
        item_sum[item] = item_sum.get(item, 0.0) + rec.rating_raw
        item_cnt[item] = item_cnt.get(item, 0) + 1

    valid_ratings = [r.rating_raw for r in pair_last.values() if r.rating_raw is not None]
    global_mean = float(np.mean(valid_ratings)) if valid_ratings else None

    smooth: dict[str, float] = {}
    rating_code: dict[str, int] = {}
    span = max(rating_max - rating_min, 1e-6)
    smooth_values: list[float] = []
    for item in all_items:
        cnt = item_cnt.get(item, 0)
        if cnt == 0 or global_mean is None:
            rating_code[item] = 0
            smooth[item] = float("nan")
            continue
        s = (item_sum[item] + alpha * global_mean) / (cnt + alpha)
        smooth[item] = s
        norm = (s - rating_min) / span
        norm = min(max(norm, 0.0), 1.0)
        smooth_values.append(norm)

    if smooth_values:
        edges = [float(x) for x in np.linspace(1.0 / 8, 1.0, 8)]
    else:
        edges = [float(x) for x in np.linspace(1.0 / 8, 1.0, 8)]
    for item in all_items:
        if rating_code.get(item, -1) == 0:
            continue
        norm = (smooth[item] - rating_min) / span
        norm = min(max(norm, 0.0), 1.0)
        rating_code[item] = _bucket_uniform(norm, edges)

    user_sets: dict[str, set[str]] = {}
    for rec in calib_records:
        user_sets.setdefault(rec.raw_item, set()).add(rec.raw_user)
    distinct_users = {item: len(user_sets.get(item, set())) for item in all_items}
    pop_values = sorted({math.log1p(distinct_users[item]) for item in all_items if distinct_users[item] > 0})
    pop_code: dict[str, int] = {}
    if pop_values:
        qs = np.linspace(0, 1, 9)[1:8]
        pop_edges = [float(np.quantile(pop_values, q)) for q in qs]
        # 去重分位点：相同值共享桶
        for item in all_items:
            n = distinct_users.get(item, 0)
            if n == 0:
                pop_code[item] = 0
                continue
            p = math.log1p(n)
            assigned = 8
            for idx, edge in enumerate(pop_edges):
                if p <= edge:
                    assigned = idx + 1
                    break
            pop_code[item] = assigned
    else:
        pop_edges = []
        for item in all_items:
            pop_code[item] = 0

    return smooth, rating_code, pop_code, edges, global_mean, pop_edges, distinct_users


def _train_text_codes(
    vectors: np.ndarray,
    *,
    text_sid_length: int,
    codebook_size: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """OPQ+PQ；向量不足时缩小码本。返回码本与训练好的 OPQ 旋转矩阵（若可用）。"""
    import faiss

    faiss.omp_set_num_threads(min(8, os.cpu_count() or 8))
    n, dim = vectors.shape
    sub_dim = dim // text_sid_length
    if sub_dim * text_sid_length != dim:
        raise ValueError("hidden dim must be divisible by text_sid_length")
    ncentroids = min(codebook_size, max(n // 5, 2))
    if n < ncentroids:
        raise ValueError(
            f"校准商品向量数 {n} 少于聚类中心 {ncentroids}，请减小 codebook 或增大 calib 集"
        )
    vectors = np.ascontiguousarray(vectors.astype("float32"))
    opq_matrix = None
    try:
        opq = faiss.OPQMatrix(dim, text_sid_length)
        opq.train(vectors)
        opq.apply_py(vectors)
        opq_matrix = faiss.vector_to_array(opq.A).reshape(dim, dim).copy()
    except Exception:
        # smoke 或小样本时允许退化为无 OPQ 的分段 PQ
        pass
    codebooks: list[np.ndarray] = []
    for seg in range(text_sid_length):
        start = seg * sub_dim
        end = start + sub_dim
        seg_x = vectors[:, start:end]
        kmeans = faiss.Kmeans(sub_dim, ncentroids, niter=20, seed=seed, verbose=False)
        kmeans.train(seg_x)
        codebooks.append(kmeans.centroids.copy())
    return codebooks, opq_matrix


def _encode_text_codes(
    vectors: np.ndarray,
    codebooks: list[np.ndarray],
    text_sid_length: int,
    opq_matrix: np.ndarray | None,
) -> np.ndarray:
    n, dim = vectors.shape
    sub_dim = dim // text_sid_length
    vectors = np.ascontiguousarray(vectors.astype("float32"))
    if opq_matrix is not None:
        vectors = vectors @ opq_matrix.T
    codes = np.zeros((n, text_sid_length), dtype=np.int64)
    for seg in range(text_sid_length):
        start = seg * sub_dim
        end = start + sub_dim
        seg_x = vectors[:, start:end]
        centroids = codebooks[seg]
        dists = ((seg_x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        codes[:, seg] = dists.argmin(axis=1)
    return codes


def build_semantic_ids(
    bundle: SequentialDatasetBundle,
    text_embeddings: dict[str, np.ndarray],
    args,
) -> SemanticIDBundle:
    calib_records = [bundle.interaction_by_id[i] for i in sorted(bundle.calib_ids)]
    all_raw_items = sorted(bundle.item2index.keys())
    item_index_to_raw = {idx: raw for raw, idx in bundle.item2index.items()}

    smooth, rating_code, pop_code, rating_edges, global_mean, pop_edges, distinct_users = _build_rating_pop(
        calib_records,
        all_raw_items,
        rating_min=args.rating_min,
        rating_max=args.rating_max,
        alpha=args.rating_smooth_alpha,
    )

    codebook_size = args.text_codebook_size
    if args.max_train_batches and args.max_train_batches > 0:
        codebook_size = int(args.smoke_codebook_size)

    calib_items = sorted({r.raw_item for r in calib_records})
    if args.max_train_batches and args.max_train_batches > 0:
        calib_items = calib_items[: max(codebook_size * 10, 64)]
    calib_vecs = np.stack([text_embeddings[item] for item in calib_items if item in text_embeddings])
    if calib_vecs.shape[0] == 0:
        raise ValueError("校准集商品无文本向量")

    codebooks, opq_matrix = _train_text_codes(
        calib_vecs,
        text_sid_length=args.text_sid_length,
        codebook_size=codebook_size,
        seed=args.quant_seed,
    )

    all_vecs = np.stack([text_embeddings[item] for item in all_raw_items])
    text_codes_all = _encode_text_codes(
        all_vecs, codebooks, args.text_sid_length, opq_matrix
    )

    items: dict[str, ItemSIDRecord] = {}
    for idx, raw_item in enumerate(all_raw_items):
        item_index = bundle.item2index[raw_item]
        tc = tuple(int(x) for x in text_codes_all[idx])
        items[raw_item] = ItemSIDRecord(
            raw_item=raw_item,
            item_index=item_index,
            text_codes=tc,
            rating_code=int(rating_code.get(raw_item, 0)),
            pop_code=int(pop_code.get(raw_item, 0)),
            smooth_rating=smooth.get(raw_item),
            raw_rating_count=0,
            distinct_users=int(distinct_users.get(raw_item, 0)),
        )

    def _full_codes(rec: ItemSIDRecord) -> tuple[int, ...]:
        c = list(rec.text_codes)
        if args.use_rating_sid:
            c.append(rec.rating_code)
        if args.use_popularity_sid:
            c.append(rec.pop_code)
        return tuple(c)

    sid_strings = ["-".join(map(str, _full_codes(items[r]))) for r in all_raw_items]

    unique = len(set(sid_strings))
    collision_rate = 1.0 - unique / max(len(sid_strings), 1)

    fp = _fingerprint_payload(args, len(calib_records))
    return SemanticIDBundle(
        fingerprint=fp,
        text_sid_length=args.text_sid_length,
        text_codebook_size=codebook_size,
        use_rating_sid=args.use_rating_sid,
        use_popularity_sid=args.use_popularity_sid,
        items=items,
        item_index_to_raw=item_index_to_raw,
        text_codebooks=codebooks,
        rating_bucket_edges=rating_edges,
        pop_bucket_edges=pop_edges if isinstance(pop_edges, list) else [],
        global_mean_rating=global_mean,
        collision_rate=collision_rate,
    )


def save_sid_bundle(path: Path, bundle: SemanticIDBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": bundle.fingerprint,
        "text_sid_length": bundle.text_sid_length,
        "text_codebook_size": bundle.text_codebook_size,
        "use_rating_sid": bundle.use_rating_sid,
        "use_popularity_sid": bundle.use_popularity_sid,
        "rating_bucket_edges": bundle.rating_bucket_edges,
        "pop_bucket_edges": bundle.pop_bucket_edges,
        "global_mean_rating": bundle.global_mean_rating,
        "collision_rate": bundle.collision_rate,
        "items": {
            k: {
                "item_index": v.item_index,
                "text_codes": v.text_codes,
                "rating_code": v.rating_code,
                "pop_code": v.pop_code,
                "smooth_rating": v.smooth_rating,
                "distinct_users": v.distinct_users,
            }
            for k, v in bundle.items.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        path.with_suffix(".npz"),
        *[bundle.text_codebooks[i] for i in range(len(bundle.text_codebooks))],
    )


def encode_all_item_texts(
    bundle: SequentialDatasetBundle,
    encoder,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    raw_items = sorted(bundle.item2index.keys())
    texts = [item_catalog_text(bundle.item_meta.get(r, {})) for r in raw_items]
    vecs = encoder.encode_texts(texts, batch_size=batch_size, use_cache=True)
    vecs = vecs.detach().cpu().numpy()
    # L2 归一化（encoder 内部应已做；再保证一次）
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    vecs = vecs / norms
    return {raw: vecs[i] for i, raw in enumerate(raw_items)}
