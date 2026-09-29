"""加速版 token 图构建：与 build_sample_token_graph 输出一致。"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from graph_llm.aux.prompt_utils import item_meta_from_row
from graph_llm.dataload.tail_stats import TailTokenStats, is_content_token
from graph_llm.models.token_graph import (
    ReviewRecord,
    UserTokenGraph,
    _select_stratified_nodes,
    attach_tokenizer_decode,
    build_sample_token_graph,
    extract_explanation_tokens,
    tokenizer_decode_stub,
)

try:
    import numba
except ImportError:  # pragma: no cover
    numba = None

# 建图并行线程上限（约 800% CPU）
_MAX_GRAPH_BUILD_WORKERS = 8

# row_key -> 预分词 token 元组，跨 train/valid/test 复用
HistoryTokenCache = dict[int, tuple[int, ...]]


def _accumulate_segment_py(
    mapped: np.ndarray,
    start: int,
    end: int,
    buffer: np.ndarray,
    count_mat: np.ndarray,
    first_seen_mat: np.ndarray,
    clock: int,
) -> int:
    n = 0
    for i in range(start, end):
        v = mapped[i]
        if v >= 0:
            buffer[n] = v
            n += 1
    if n < 2:
        return clock
    for i in range(n):
        src = buffer[i]
        for j in range(i + 1, n):
            dst = buffer[j]
            if src == dst:
                continue
            if count_mat[src, dst] == 0:
                first_seen_mat[src, dst] = clock
                clock += 1
            count_mat[src, dst] += 1
    return clock


def _accumulate_item_merge_py(
    mapped: np.ndarray,
    review_starts: np.ndarray,
    review_ends: np.ndarray,
    review_indices: np.ndarray,
    buffer: np.ndarray,
    count_mat: np.ndarray,
    first_seen_mat: np.ndarray,
    clock: int,
) -> int:
    n = 0
    for ridx in review_indices:
        start = int(review_starts[ridx])
        end = int(review_ends[ridx])
        for i in range(start, end):
            v = mapped[i]
            if v >= 0:
                buffer[n] = v
                n += 1
    if n < 2:
        return clock
    for i in range(n):
        src = buffer[i]
        for j in range(i + 1, n):
            dst = buffer[j]
            if src == dst:
                continue
            if count_mat[src, dst] == 0:
                first_seen_mat[src, dst] = clock
                clock += 1
            count_mat[src, dst] += 1
    return clock


if numba is not None:
    _accumulate_segment = numba.njit(nogil=True)(_accumulate_segment_py)
    _accumulate_item_merge = numba.njit(nogil=True)(_accumulate_item_merge_py)
else:
    _accumulate_segment = _accumulate_segment_py
    _accumulate_item_merge = _accumulate_item_merge_py


def _export_edges_numpy(
    count_mat: np.ndarray,
    first_seen_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    src, dst = np.nonzero(count_mat)
    if src.size == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    order = np.argsort(first_seen_mat[src, dst], kind="mergesort")
    src = src[order]
    dst = dst[order]
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_weight = count_mat[src, dst].astype(np.float32)
    return edge_index, edge_weight


def _degrees_bincount(num_nodes: int, edge_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out_degree = np.zeros(num_nodes, dtype=np.float32)
    in_degree = np.zeros(num_nodes, dtype=np.float32)
    if edge_index.size == 0:
        return in_degree, out_degree
    src = edge_index[0]
    dst = edge_index[1]
    out_degree += np.bincount(src, minlength=num_nodes).astype(np.float32)
    in_degree += np.bincount(dst, minlength=num_nodes).astype(np.float32)
    return in_degree, out_degree


def select_nodes_fast(
    eligible_compact: np.ndarray,
    cnt: np.ndarray,
    df: np.ndarray,
    token_vocab_ids: np.ndarray,
    *,
    tail_stats: TailTokenStats | None,
    target_item_token_ids: set[int],
    max_nodes: int,
    tail_node_quota: int,
    relevance_node_quota: int,
    preference_node_quota: int,
    gdf: np.ndarray | None = None,
    is_tail: np.ndarray | None = None,
) -> np.ndarray:
    """用 lexsort 逻辑复刻 _select_stratified_nodes / legacy 排序选点。"""
    if max_nodes <= 0 or eligible_compact.size == 0:
        return np.empty((0,), dtype=np.int64)

    budget = min(int(max_nodes), int(eligible_compact.size))

    if tail_stats is None:
        vocab = token_vocab_ids[eligible_compact]
        cnt_e = cnt[eligible_compact].astype(np.int64)
        order = np.lexsort((vocab, -cnt_e))
        return eligible_compact[order[:budget]].astype(np.int64)

    if gdf is None or is_tail is None:
        raise ValueError("gdf and is_tail required when tail_stats is set")

    selected: list[int] = []
    selected_set: set[int] = set()

    def append_ranked(candidates: np.ndarray, quota: int, key_fn) -> None:
        if quota <= 0 or len(selected) >= budget or candidates.size == 0:
            return
        limit = min(int(quota), budget - len(selected))
        available = [int(c) for c in candidates if int(c) not in selected_set]
        if not available:
            return
        available.sort(key=lambda compact_id: key_fn(compact_id, int(token_vocab_ids[compact_id])))
        added = 0
        for compact_id in available:
            if compact_id in selected_set:
                continue
            selected.append(compact_id)
            selected_set.add(compact_id)
            added += 1
            if added >= limit or len(selected) >= budget:
                break

    tail_candidates = eligible_compact[is_tail[eligible_compact]]
    append_ranked(
        tail_candidates,
        tail_node_quota,
        lambda c, tid: (
            -int(tid in target_item_token_ids),
            int(gdf[c]),
            -int(df[c]),
            -int(cnt[c]),
            tid,
        ),
    )
    append_ranked(
        eligible_compact,
        relevance_node_quota,
        lambda c, tid: (
            -int(tid in target_item_token_ids),
            -int(df[c]),
            -int(cnt[c]),
            tid,
        ),
    )
    append_ranked(
        eligible_compact,
        preference_node_quota,
        lambda c, tid: (
            -int(df[c]),
            -int(cnt[c]),
            tid,
        ),
    )
    append_ranked(
        eligible_compact,
        budget - len(selected),
        lambda c, tid: (
            -int(tid in target_item_token_ids),
            -int(df[c]),
            -int(cnt[c]),
            tid,
        ),
    )
    return np.array(selected[:budget], dtype=np.int64)


@dataclass
class _UserHistory:
    """单用户历史的紧凑表示，供多样本增量建图复用。"""

    row_keys: np.ndarray
    item_codes: np.ndarray
    item_strings: list[str]
    starts: np.ndarray
    ends: np.ndarray
    tokens: np.ndarray
    token_vocab_ids: np.ndarray
    total_cnt: np.ndarray
    total_df: np.ndarray
    gdf: np.ndarray
    is_tail: np.ndarray
    content_mask: np.ndarray
    item_to_review_indices: list[np.ndarray]
    max_segment_len: int


def _build_user_history(
    records: list[ReviewRecord],
    *,
    tail_stats: TailTokenStats | None,
    skip_token_ids: set[int],
) -> _UserHistory | None:
    if not records:
        return None

    item_strings: list[str] = []
    item_to_code: dict[str, int] = {}
    row_keys_list: list[int] = []
    item_codes_list: list[int] = []
    starts_list: list[int] = []
    ends_list: list[int] = []
    token_chunks: list[np.ndarray] = []

    vocab_to_compact: dict[int, int] = {}
    compact_vocab: list[int] = []

    offset = 0
    for record in records:
        row_keys_list.append(int(record.row_key))
        item = str(record.raw_item)
        if item not in item_to_code:
            item_to_code[item] = len(item_strings)
            item_strings.append(item)
        item_codes_list.append(item_to_code[item])

        compact_ids: list[int] = []
        for tid in record.token_ids:
            tid = int(tid)
            if tid in skip_token_ids:
                continue
            if tid not in vocab_to_compact:
                vocab_to_compact[tid] = len(compact_vocab)
                compact_vocab.append(tid)
            compact_ids.append(vocab_to_compact[tid])

        starts_list.append(offset)
        if compact_ids:
            chunk = np.array(compact_ids, dtype=np.int32)
            token_chunks.append(chunk)
            offset += int(chunk.size)
        ends_list.append(offset)

    if offset == 0:
        tokens = np.empty((0,), dtype=np.int32)
    else:
        tokens = np.concatenate(token_chunks).astype(np.int32)

    num_unique = len(compact_vocab)
    token_vocab_ids = np.array(compact_vocab, dtype=np.int64)
    total_cnt = np.zeros(num_unique, dtype=np.int32)
    total_df = np.zeros(num_unique, dtype=np.int32)

    num_reviews = len(row_keys_list)
    for ridx in range(num_reviews):
        s, e = starts_list[ridx], ends_list[ridx]
        if e <= s:
            continue
        seg = tokens[s:e]
        np.add.at(total_cnt, seg, 1)
        uniq = np.unique(seg)
        np.add.at(total_df, uniq, 1)

    item_to_review_indices: list[list[int]] = [[] for _ in item_strings]
    for ridx, code in enumerate(item_codes_list):
        item_to_review_indices[code].append(ridx)
    item_to_review_arrays = [
        np.array(indices, dtype=np.int32) for indices in item_to_review_indices
    ]

    gdf = np.zeros(num_unique, dtype=np.int32)
    is_tail = np.zeros(num_unique, dtype=bool)
    content_mask = np.ones(num_unique, dtype=bool)
    if tail_stats is not None:
        for compact_id, vocab_id in enumerate(compact_vocab):
            gdf[compact_id] = int(tail_stats.document_frequency(int(vocab_id)))
            is_tail[compact_id] = bool(tail_stats.is_tail(int(vocab_id)))

    seg_lens = [ends_list[i] - starts_list[i] for i in range(num_reviews)]
    max_segment_len = max(seg_lens) if seg_lens else 0

    return _UserHistory(
        row_keys=np.array(row_keys_list, dtype=np.int64),
        item_codes=np.array(item_codes_list, dtype=np.int32),
        item_strings=item_strings,
        starts=np.array(starts_list, dtype=np.int32),
        ends=np.array(ends_list, dtype=np.int32),
        tokens=tokens,
        token_vocab_ids=token_vocab_ids,
        total_cnt=total_cnt,
        total_df=total_df,
        gdf=gdf,
        is_tail=is_tail,
        content_mask=content_mask,
        item_to_review_indices=item_to_review_arrays,
        max_segment_len=max_segment_len,
    )


def build_graph_from_user_history(
    user_hist: _UserHistory,
    *,
    exclude_row_key: int,
    target_raw_item: str,
    min_token_count: int,
    max_nodes: int,
    tail_stats: TailTokenStats | None,
    target_item_token_ids: set[int],
    use_content_filter: bool,
    tail_node_quota: int,
    relevance_node_quota: int,
    preference_node_quota: int,
    surface_by_token: dict[int, str],
    edge_buffer: np.ndarray,
) -> UserTokenGraph:
    num_reviews = int(user_hist.row_keys.shape[0])
    excluded = np.zeros(num_reviews, dtype=bool)
    target_code = -1
    for code, item in enumerate(user_hist.item_strings):
        if item == str(target_raw_item):
            target_code = code
            break

    for ridx in range(num_reviews):
        if int(user_hist.row_keys[ridx]) == int(exclude_row_key):
            excluded[ridx] = True
        elif target_code >= 0 and int(user_hist.item_codes[ridx]) == target_code:
            excluded[ridx] = True

    cnt = user_hist.total_cnt.copy()
    df = user_hist.total_df.copy()
    for ridx in range(num_reviews):
        if not excluded[ridx]:
            continue
        s, e = int(user_hist.starts[ridx]), int(user_hist.ends[ridx])
        if e <= s:
            continue
        seg = user_hist.tokens[s:e]
        np.add.at(cnt, seg, -1)
        uniq = np.unique(seg)
        np.add.at(df, uniq, -1)

    min_cnt = max(int(min_token_count), 1)
    eligible_compact = np.where((cnt >= min_cnt) & user_hist.content_mask)[0]

    if eligible_compact.size == 0:
        return UserTokenGraph.empty()

    ranked_compact = select_nodes_fast(
        eligible_compact,
        cnt,
        df,
        user_hist.token_vocab_ids,
        tail_stats=tail_stats,
        target_item_token_ids=target_item_token_ids,
        max_nodes=max_nodes,
        tail_node_quota=tail_node_quota,
        relevance_node_quota=relevance_node_quota,
        preference_node_quota=preference_node_quota,
        gdf=user_hist.gdf,
        is_tail=user_hist.is_tail,
    )
    if ranked_compact.size == 0:
        return UserTokenGraph.empty()

    n_nodes = int(ranked_compact.size)
    ranked_vocab = user_hist.token_vocab_ids[ranked_compact]

    map_arr = np.full(user_hist.token_vocab_ids.shape[0], -1, dtype=np.int32)
    map_arr[ranked_compact] = np.arange(n_nodes, dtype=np.int32)
    mapped = map_arr[user_hist.tokens]

    eligible_review_indices: list[int] = []
    for ridx in range(num_reviews):
        if excluded[ridx]:
            continue
        if int(user_hist.ends[ridx]) <= int(user_hist.starts[ridx]):
            continue
        eligible_review_indices.append(ridx)

    eligible_arr = np.array(eligible_review_indices, dtype=np.int32)

    # 同物品合并：eligible 中首次出现顺序，且 eligible 评论数 >= 2
    item_first_pos: dict[int, int] = {}
    item_eligible_reviews: dict[int, list[int]] = defaultdict(list)
    for pos, ridx in enumerate(eligible_review_indices):
        code = int(user_hist.item_codes[ridx])
        if code not in item_first_pos:
            item_first_pos[code] = pos
        item_eligible_reviews[code].append(ridx)
    merge_lists: list[np.ndarray] = []
    for code in sorted(item_first_pos, key=lambda c: item_first_pos[c]):
        reviews = item_eligible_reviews[code]
        if len(reviews) >= 2:
            merge_lists.append(np.array(reviews, dtype=np.int32))

    count_mat = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    first_seen_mat = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    clock = 0
    for ridx in eligible_arr:
        clock = _accumulate_segment(
            mapped,
            int(user_hist.starts[ridx]),
            int(user_hist.ends[ridx]),
            edge_buffer,
            count_mat,
            first_seen_mat,
            clock,
        )
    for review_indices in merge_lists:
        clock = _accumulate_item_merge(
            mapped,
            user_hist.starts,
            user_hist.ends,
            review_indices,
            edge_buffer,
            count_mat,
            first_seen_mat,
            clock,
        )

    edge_index, edge_weight = _export_edges_numpy(count_mat, first_seen_mat)
    in_degree, out_degree = _degrees_bincount(n_nodes, edge_index)

    surfaces: list[str] = []
    for tid in ranked_vocab.tolist():
        tid = int(tid)
        if tid in surface_by_token:
            surfaces.append(surface_by_token[tid])
        else:
            try:
                surfaces.append(str(tokenizer_decode_stub(tid)))
            except Exception:
                surfaces.append(str(tid))

    return UserTokenGraph(
        node_token_ids=ranked_vocab.astype(np.int64),
        node_surfaces=surfaces,
        node_counts=cnt[ranked_compact].astype(np.float32),
        node_doc_freq=df[ranked_compact].astype(np.float32),
        edge_index=edge_index,
        edge_weight=edge_weight,
        in_degree=in_degree,
        out_degree=out_degree,
    )


def build_sample_token_graph_fast(
    history_records: list[ReviewRecord],
    *,
    exclude_row_key: int,
    target_raw_item: str,
    skip_token_ids: set[int],
    max_nodes: int,
    min_token_count: int,
    tail_stats: TailTokenStats | None,
    target_item_token_ids: set[int],
    content_token_filter: Callable[[int], bool] | None,
    tail_node_quota: int,
    relevance_node_quota: int,
    preference_node_quota: int,
    surface_by_token: dict[int, str],
    user_hist: _UserHistory | None = None,
    edge_buffer: np.ndarray | None = None,
) -> UserTokenGraph:
    if numba is None:
        raise RuntimeError("numba is required for fast graph build; install with: conda install -n fair numba")

    use_content_filter = content_token_filter is not None
    if user_hist is None:
        user_hist = _build_user_history(
            history_records,
            tail_stats=tail_stats,
            skip_token_ids=skip_token_ids,
        )
        if user_hist is None:
            return UserTokenGraph.empty()
        if use_content_filter and content_token_filter is not None:
            for compact_id, vocab_id in enumerate(user_hist.token_vocab_ids.tolist()):
                user_hist.content_mask[compact_id] = bool(
                    content_token_filter(int(vocab_id)),
                )

    if user_hist is None:
        return UserTokenGraph.empty()

    if edge_buffer is None:
        buf_len = max(int(user_hist.max_segment_len), int(user_hist.tokens.shape[0]), 1)
        edge_buffer = np.empty(buf_len, dtype=np.int32)

    return build_graph_from_user_history(
        user_hist,
        exclude_row_key=exclude_row_key,
        target_raw_item=target_raw_item,
        min_token_count=min_token_count,
        max_nodes=max_nodes,
        tail_stats=tail_stats,
        target_item_token_ids=target_item_token_ids,
        use_content_filter=use_content_filter,
        tail_node_quota=tail_node_quota,
        relevance_node_quota=relevance_node_quota,
        preference_node_quota=preference_node_quota,
        surface_by_token=surface_by_token,
        edge_buffer=edge_buffer,
    )


@dataclass(frozen=True)
class _SampleTask:
    local_idx: int
    row_key: int
    raw_item: str


def _graphs_equal(a: UserTokenGraph, b: UserTokenGraph) -> bool:
    if a.num_nodes != b.num_nodes:
        return False
    if not np.array_equal(a.node_token_ids, b.node_token_ids):
        return False
    if a.node_surfaces != b.node_surfaces:
        return False
    if not np.allclose(a.node_counts, b.node_counts):
        return False
    if not np.allclose(a.node_doc_freq, b.node_doc_freq):
        return False
    if not np.array_equal(a.edge_index, b.edge_index):
        return False
    if not np.allclose(a.edge_weight, b.edge_weight):
        return False
    if not np.allclose(a.in_degree, b.in_degree):
        return False
    if not np.allclose(a.out_degree, b.out_degree):
        return False
    return True


def assert_graphs_equal(a: UserTokenGraph, b: UserTokenGraph) -> None:
    if not _graphs_equal(a, b):
        raise AssertionError("Graph mismatch between reference and fast build")


def extend_history_token_cache(
    cache: HistoryTokenCache,
    history_dataset: pd.DataFrame,
    tokenizer,
    skip_token_ids: set[int],
) -> None:
    """批量分词并写入 cache，已存在的 row_key 跳过。"""
    pending_keys: list[int] = []
    pending_texts: list[str] = []
    for row_key, row in history_dataset.iterrows():
        rk = int(row_key)
        if rk in cache:
            continue
        explanation = row["review_text"] if "review_text" in row else row["template"][2]
        pending_keys.append(rk)
        pending_texts.append("" if explanation is None else str(explanation))

    if not pending_texts:
        return

    encoded = tokenizer(pending_texts, add_special_tokens=False)
    for rk, ids in zip(pending_keys, encoded["input_ids"]):
        cache[rk] = tuple(
            int(t) for t in ids if int(t) not in skip_token_ids
        )


def _records_from_cache(
    history_dataset: pd.DataFrame,
    cache: HistoryTokenCache,
    allowed_history_keys: dict[str, set[int]],
) -> dict[str, list[ReviewRecord]]:
    user_histories: dict[str, list[ReviewRecord]] = defaultdict(list)
    for row_key, row in history_dataset.iterrows():
        raw_user = str(row["raw_user"])
        rk = int(row_key)
        if rk not in allowed_history_keys.get(raw_user, set()):
            continue
        tokens = cache.get(rk)
        if tokens is None:
            raise KeyError(f"Missing token cache for row_key={rk}")
        user_histories[raw_user].append(
            ReviewRecord(
                rk,
                raw_user,
                str(row["raw_item"]),
                tokens,
            ),
        )
    return dict(user_histories)


def _precompute_item_token_sets(
    raw_items: set[str],
    item_meta: dict | None,
    tokenizer,
    skip_token_ids: set[int],
) -> dict[str, set[int]]:
    cache: dict[str, set[int]] = {}
    items = sorted(raw_items)
    if not items:
        return cache
    texts = []
    for raw_item in items:
        _title, _description, item_text = item_meta_from_row(raw_item, item_meta)
        texts.append(item_text)
    encoded = tokenizer(texts, add_special_tokens=False)
    for raw_item, ids in zip(items, encoded["input_ids"]):
        cache[raw_item] = {
            int(token_id)
            for token_id in ids
            if int(token_id) not in skip_token_ids
        }
    return cache


def _precompute_surfaces(
    token_ids: set[int],
    tokenizer,
) -> dict[int, str]:
    surfaces: dict[int, str] = {}
    for token_id in token_ids:
        try:
            surfaces[int(token_id)] = str(
                tokenizer.decode([int(token_id)], skip_special_tokens=True),
            )
        except Exception:
            surfaces[int(token_id)] = str(token_id)
    return surfaces


def build_split_graphs_fast(
    *,
    split_dataset: pd.DataFrame,
    split_name: str,
    fold: int,
    history_dataset: pd.DataFrame,
    tokenizer,
    skip_token_ids: set[int],
    max_nodes: int,
    min_token_count: int,
    tail_stats: TailTokenStats | None,
    item_meta: dict | None,
    tail_node_quota: int,
    relevance_node_quota: int,
    preference_node_quota: int,
    max_workers: int = _MAX_GRAPH_BUILD_WORKERS,
    history_token_cache: HistoryTokenCache | None = None,
) -> tuple[
    dict[str, list[ReviewRecord]],
    dict[str, set[int]],
    dict[tuple[str, int], UserTokenGraph],
]:
    if numba is None:
        raise RuntimeError("numba is required for fast graph build; install with: conda install -n fair numba")

    attach_tokenizer_decode(tokenizer)
    use_content_filter = tail_stats is not None

    token_cache: HistoryTokenCache = history_token_cache if history_token_cache is not None else {}
    extend_history_token_cache(token_cache, history_dataset, tokenizer, skip_token_ids)

    allowed_history_keys: dict[str, set[int]] = defaultdict(set)
    for row_key, row in history_dataset.iterrows():
        allowed_history_keys[str(row["raw_user"])].add(int(row_key))

    user_histories = _records_from_cache(history_dataset, token_cache, allowed_history_keys)

    user_tasks: dict[str, list[_SampleTask]] = defaultdict(list)
    for local_idx, (_, row) in enumerate(split_dataset.iterrows()):
        raw_user = str(row["raw_user"])
        user_tasks[raw_user].append(
            _SampleTask(
                local_idx=local_idx,
                row_key=int(row.name),
                raw_item=str(row["raw_item"]),
            ),
        )

    raw_items: set[str] = set()
    all_history_tokens: set[int] = set()
    for records in user_histories.values():
        for rec in records:
            raw_items.add(str(rec.raw_item))
            all_history_tokens.update(int(t) for t in rec.token_ids)
    for tasks in user_tasks.values():
        for task in tasks:
            raw_items.add(task.raw_item)

    item_token_cache = _precompute_item_token_sets(raw_items, item_meta, tokenizer, skip_token_ids)
    surface_by_token = _precompute_surfaces(all_history_tokens, tokenizer)

    user_history_structs: dict[str, _UserHistory] = {}
    for raw_user, records in user_histories.items():
        uh = _build_user_history(
            records,
            tail_stats=tail_stats,
            skip_token_ids=skip_token_ids,
        )
        if uh is not None and use_content_filter:
            for compact_id, vocab_id in enumerate(uh.token_vocab_ids.tolist()):
                uh.content_mask[compact_id] = is_content_token(
                    tokenizer,
                    int(vocab_id),
                    skip_token_ids,
                )
        if uh is not None:
            user_history_structs[raw_user] = uh

    graphs: dict[tuple[str, int], UserTokenGraph] = {}

    def _build_user(raw_user: str, tasks: list[_SampleTask]) -> list[tuple[int, UserTokenGraph]]:
        uh = user_history_structs.get(raw_user)
        if uh is None:
            return [(task.local_idx, UserTokenGraph.empty()) for task in tasks]
        buf_len = max(int(uh.max_segment_len), int(uh.tokens.shape[0]), 1)
        buffer = np.empty(buf_len, dtype=np.int32)
        built: list[tuple[int, UserTokenGraph]] = []
        for task in tasks:
            graph = build_sample_token_graph_fast(
                [],
                exclude_row_key=task.row_key,
                target_raw_item=task.raw_item,
                skip_token_ids=skip_token_ids,
                max_nodes=max_nodes,
                min_token_count=min_token_count,
                tail_stats=tail_stats,
                target_item_token_ids=item_token_cache.get(task.raw_item, set()),
                content_token_filter=(lambda _: True) if use_content_filter else None,
                tail_node_quota=tail_node_quota,
                relevance_node_quota=relevance_node_quota,
                preference_node_quota=preference_node_quota,
                surface_by_token=surface_by_token,
                user_hist=uh,
                edge_buffer=buffer,
            )
            built.append((task.local_idx, graph))
        return built

    user_list = list(user_tasks.items())
    workers = min(max_workers, max(1, len(user_list)))

    if workers <= 1:
        for raw_user, tasks in tqdm(
            user_list,
            desc=f"build graphs fold={fold} split={split_name}",
        ):
            for local_idx, graph in _build_user(raw_user, tasks):
                graphs[(split_name, local_idx)] = graph
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_build_user, raw_user, tasks)
                for raw_user, tasks in user_list
            ]
            for future in tqdm(
                futures,
                total=len(futures),
                desc=f"build graphs fold={fold} split={split_name}",
            ):
                for local_idx, graph in future.result():
                    graphs[(split_name, local_idx)] = graph

    return (
        user_histories,
        {k: set(v) for k, v in allowed_history_keys.items()},
        graphs,
    )


def build_sample_token_graph_reference(
    history_records: list[ReviewRecord],
    *,
    exclude_row_key: int,
    target_raw_item: str,
    skip_token_ids: set[int],
    max_nodes: int,
    min_token_count: int,
    tail_stats: TailTokenStats | None,
    target_item_token_ids: set[int],
    content_token_filter: Callable[[int], bool] | None,
    tail_node_quota: int,
    relevance_node_quota: int,
    preference_node_quota: int,
) -> UserTokenGraph:
    return build_sample_token_graph(
        history_records,
        exclude_row_key=exclude_row_key,
        target_raw_item=target_raw_item,
        skip_token_ids=skip_token_ids,
        max_nodes=max_nodes,
        min_token_count=min_token_count,
        tail_stats=tail_stats,
        target_item_token_ids=target_item_token_ids,
        content_token_filter=content_token_filter,
        tail_node_quota=tail_node_quota,
        relevance_node_quota=relevance_node_quota,
        preference_node_quota=preference_node_quota,
    )
