"""fast 建图与 build_sample_token_graph 输出一致性测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import time

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from collections import Counter

from graph_llm.dataload.graph_build_fast import (
    assert_graphs_equal,
    build_sample_token_graph_fast,
    build_sample_token_graph_reference,
    select_nodes_fast,
)
from graph_llm.models.token_graph import _select_stratified_nodes
from graph_llm.dataload.pleaser import load_pleaser_frame, load_pleaser_item_meta
from graph_llm.dataload.tail_stats import build_tail_token_stats, is_content_token
from graph_llm.models.token_graph import ReviewRecord, attach_tokenizer_decode


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    unk_token_id = 3

    def __call__(self, text, add_special_tokens=True):
        ids = [10 + (ord(ch) % 7) for ch in str(text)[:12]]
        if add_special_tokens:
            ids = [1] + ids
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        if not ids:
            return ""
        return f"t{ids[0]}"


def _surface_map(tokenizer, token_ids: set[int]) -> dict[int, str]:
    return {
        int(tid): str(tokenizer.decode([int(tid)], skip_special_tokens=True))
        for tid in token_ids
    }


def _compare(
    history,
    *,
    exclude_row_key: int,
    target_raw_item: str,
    skip_token_ids: set[int],
    tail_stats=None,
    target_item_token_ids=None,
    content_filter=None,
    surfaces: dict[int, str],
    max_nodes: int = 8,
):
    kwargs = {
        "exclude_row_key": exclude_row_key,
        "target_raw_item": target_raw_item,
        "skip_token_ids": skip_token_ids,
        "max_nodes": max_nodes,
        "min_token_count": 1,
        "tail_stats": tail_stats,
        "target_item_token_ids": target_item_token_ids or set(),
        "content_token_filter": content_filter,
        "tail_node_quota": 4,
        "relevance_node_quota": 2,
        "preference_node_quota": 2,
    }
    ref = build_sample_token_graph_reference(history, **kwargs)
    fast = build_sample_token_graph_fast(
        history,
        surface_by_token=surfaces,
        **kwargs,
    )
    assert_graphs_equal(ref, fast)


def test_manual_subsequence_and_item_merge():
    tokenizer = _TinyTokenizer()
    attach_tokenizer_decode(tokenizer)
    skip = {0, 1, 2}
    history = [
        ReviewRecord(0, "u", "i1", (10, 11, 12, 13)),
        ReviewRecord(1, "u", "i1", (10, 14, 15)),
        ReviewRecord(2, "u", "i2", (11, 16, 17)),
    ]
    tokens = {10, 11, 12, 13, 14, 15, 16, 17}
    surfaces = _surface_map(tokenizer, tokens)
    _compare(
        history,
        exclude_row_key=2,
        target_raw_item="i2",
        skip_token_ids=skip,
        surfaces=surfaces,
    )


def test_exclude_target_item_reviews():
    tokenizer = _TinyTokenizer()
    attach_tokenizer_decode(tokenizer)
    skip = {0, 1, 2}
    history = [
        ReviewRecord(0, "u", "target", (10, 11)),
        ReviewRecord(1, "u", "other", (12, 13, 14)),
    ]
    tokens = {10, 11, 12, 13, 14}
    surfaces = _surface_map(tokenizer, tokens)
    _compare(
        history,
        exclude_row_key=1,
        target_raw_item="target",
        skip_token_ids=skip,
        surfaces=surfaces,
    )


def test_tail_stats_stratified_nodes():
    tokenizer = _TinyTokenizer()
    attach_tokenizer_decode(tokenizer)
    skip = {0, 1, 2}
    history = [
        ReviewRecord(0, "u", "i1", (10, 11, 12)),
        ReviewRecord(1, "u", "i2", (13, 14, 15)),
        ReviewRecord(2, "u", "i3", (16, 17, 18)),
    ]
    tail_stats = build_tail_token_stats(
        [[10, 11, 12], [13, 14, 15]],
        tokenizer=tokenizer,
        ignored_token_ids=skip,
    )
    content_filter = lambda token_id: is_content_token(tokenizer, token_id, skip)
    tokens = set(range(10, 19))
    surfaces = _surface_map(tokenizer, tokens)
    _compare(
        history,
        exclude_row_key=2,
        target_raw_item="i3",
        skip_token_ids=skip,
        tail_stats=tail_stats,
        target_item_token_ids={16, 17},
        content_filter=content_filter,
        surfaces=surfaces,
    )


def test_heavy_user_long_reviews_and_item_merge():
    tokenizer = _TinyTokenizer()
    attach_tokenizer_decode(tokenizer)
    skip = {0, 1, 2}
    long_tokens = tuple(range(10, 10 + 40))
    history = [
        ReviewRecord(0, "u", "shared", long_tokens),
        ReviewRecord(1, "u", "shared", tuple(range(10, 30))),
        ReviewRecord(2, "u", "other", tuple(range(20, 60))),
        ReviewRecord(3, "u", "target", tuple(range(30, 70))),
    ]
    tokens = set(range(10, 70))
    surfaces = _surface_map(tokenizer, tokens)
    _compare(
        history,
        exclude_row_key=3,
        target_raw_item="target",
        skip_token_ids=skip,
        surfaces=surfaces,
        max_nodes=16,
    )


def test_select_nodes_fast_matches_stratified():
    tokenizer = _TinyTokenizer()
    skip = {0, 1, 2}
    token_count = Counter({10: 5, 11: 4, 12: 3, 13: 2, 14: 1, 15: 6, 16: 2})
    token_doc_freq = Counter({10: 3, 11: 2, 12: 2, 13: 1, 14: 1, 15: 4, 16: 1})
    eligible = sorted(token_count.keys())
    token_vocab_ids = np.array(eligible, dtype=np.int64)
    cnt = np.array([token_count[t] for t in eligible], dtype=np.int32)
    df = np.array([token_doc_freq[t] for t in eligible], dtype=np.int32)
    eligible_compact = np.arange(len(eligible), dtype=np.int64)
    tail_stats = build_tail_token_stats(
        [[10, 11, 12], [13, 14, 15]],
        tokenizer=tokenizer,
        ignored_token_ids=skip,
    )

    ref = _select_stratified_nodes(
        eligible,
        token_count=token_count,
        token_doc_freq=token_doc_freq,
        tail_stats=tail_stats,
        target_item_token_ids={15},
        max_nodes=5,
        tail_node_quota=2,
        relevance_node_quota=2,
        preference_node_quota=2,
    )
    gdf = np.array([tail_stats.document_frequency(int(t)) for t in eligible], dtype=np.int32)
    is_tail = np.array([tail_stats.is_tail(int(t)) for t in eligible], dtype=bool)
    fast = select_nodes_fast(
        eligible_compact,
        cnt,
        df,
        token_vocab_ids,
        tail_stats=tail_stats,
        target_item_token_ids={15},
        max_nodes=5,
        tail_node_quota=2,
        relevance_node_quota=2,
        preference_node_quota=2,
        gdf=gdf,
        is_tail=is_tail,
    )
    assert ref == token_vocab_ids[fast].tolist()


def test_instruments_user_slice_equivalence():
    pytest.importorskip("numba")
    from graph_llm.config.args import default_model_path, resolve_local_model_path, qwen3_4b_model_candidates
    from transformers import AutoTokenizer
    from graph_llm.aux.prompt_utils import item_meta_from_row

    frame = load_pleaser_frame(REPO / "data", "Instruments")
    item_meta = load_pleaser_item_meta(REPO / "data", "Instruments")
    model_path = resolve_local_model_path(
        default_model_path(),
        candidates=qwen3_4b_model_candidates(),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    attach_tokenizer_decode(tokenizer)
    skip = {tokenizer.pad_token_id, tokenizer.eos_token_id}
    if tokenizer.unk_token_id is not None:
        skip.add(int(tokenizer.unk_token_id))

    user_lens = frame.groupby("user").size()
    heavy_users = user_lens.nlargest(3).index.astype(str).tolist()
    typical_users = user_lens[(user_lens >= 8) & (user_lens <= 10)].index.astype(str).tolist()[:5]
    users = list(dict.fromkeys(heavy_users + typical_users))
    subset = frame[frame["user"].astype(str).isin(users)]

    histories: dict[str, list[ReviewRecord]] = {}
    for row_key, row in subset.iterrows():
        raw_user = str(row["user"])
        ids = tokenizer(str(row["review_text"]), add_special_tokens=False)["input_ids"]
        tokens = tuple(int(t) for t in ids if int(t) not in skip)
        histories.setdefault(raw_user, []).append(
            ReviewRecord(int(row_key), raw_user, str(row["item"]), tokens),
        )

    all_tokens = {t for recs in histories.values() for rec in recs for t in rec.token_ids}
    surfaces = _surface_map(tokenizer, all_tokens)
    tail_stats = build_tail_token_stats(
        [list(rec.token_ids) for recs in histories.values() for rec in recs[:5]],
        tokenizer=tokenizer,
        ignored_token_ids=skip,
    )
    content_filter = lambda token_id: is_content_token(tokenizer, token_id, skip)

    checked = 0
    for row_key, row in subset.iterrows():
        if checked >= 40:
            break
        raw_user = str(row["user"])
        raw_item = str(row["item"])
        _title, _description, item_text = item_meta_from_row(raw_item, item_meta)
        target_ids = {
            int(t)
            for t in tokenizer(item_text, add_special_tokens=False)["input_ids"]
            if int(t) not in skip
        }
        _compare(
            histories[raw_user],
            exclude_row_key=int(row_key),
            target_raw_item=raw_item,
            skip_token_ids=skip,
            tail_stats=tail_stats,
            target_item_token_ids=target_ids,
            content_filter=content_filter,
            surfaces=surfaces,
        )
        checked += 1


def test_instruments_build_timing_smoke():
    pytest.importorskip("numba")
    from graph_llm.config.args import default_model_path, resolve_local_model_path, qwen3_4b_model_candidates
    from transformers import AutoTokenizer

    frame = load_pleaser_frame(REPO / "data", "Instruments")
    model_path = resolve_local_model_path(
        default_model_path(),
        candidates=qwen3_4b_model_candidates(),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    skip = {tokenizer.pad_token_id, tokenizer.eos_token_id}
    user_lens = frame.groupby("user").size()
    users = user_lens.nlargest(20).index.astype(str).tolist()
    subset = frame[frame["user"].astype(str).isin(users)]

    histories: dict[str, list[ReviewRecord]] = {}
    for row_key, row in subset.iterrows():
        raw_user = str(row["user"])
        ids = tokenizer(str(row["review_text"]), add_special_tokens=False)["input_ids"]
        tokens = tuple(int(t) for t in ids if int(t) not in skip)
        histories.setdefault(raw_user, []).append(
            ReviewRecord(int(row_key), raw_user, str(row["item"]), tokens),
        )

    surfaces = {t: "x" for t in range(100000)}
    t0 = time.time()
    samples = 0
    for row_key, row in subset.head(200).iterrows():
        build_sample_token_graph_fast(
            histories[str(row["user"])],
            exclude_row_key=int(row_key),
            target_raw_item=str(row["item"]),
            skip_token_ids=skip,
            max_nodes=512,
            min_token_count=1,
            tail_stats=None,
            target_item_token_ids=set(),
            content_token_filter=None,
            tail_node_quota=256,
            relevance_node_quota=128,
            preference_node_quota=128,
            surface_by_token=surfaces,
        )
        samples += 1
    elapsed = time.time() - t0
    per_sample = elapsed / max(samples, 1)
    assert per_sample < 0.5, f"too slow: {per_sample:.3f}s/sample on heavy slice"


@pytest.mark.slow
def test_software_first_users_real_tokenizer():
    numba = pytest.importorskip("numba")
    del numba  # 仅检查依赖存在

    from graph_llm.config.args import default_model_path, resolve_local_model_path, qwen3_4b_model_candidates
    from transformers import AutoTokenizer

    data_dir = REPO / "data"
    frame = load_pleaser_frame(data_dir, "Software")
    item_meta = load_pleaser_item_meta(data_dir, "Software")
    model_path = resolve_local_model_path(
        default_model_path(),
        candidates=qwen3_4b_model_candidates(),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    attach_tokenizer_decode(tokenizer)
    skip = {tokenizer.pad_token_id, tokenizer.eos_token_id}
    if tokenizer.unk_token_id is not None:
        skip.add(int(tokenizer.unk_token_id))

    users = frame["user"].astype(str).unique()[:20]
    subset = frame[frame["user"].astype(str).isin(users)].head(200)
    history_records: dict[str, list[ReviewRecord]] = {}
    for row_key, row in subset.iterrows():
        raw_user = str(row["user"])
        raw_item = str(row["item"])
        text = str(row["review_text"])
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        tokens = tuple(int(t) for t in ids if int(t) not in skip)
        history_records.setdefault(raw_user, []).append(
            ReviewRecord(int(row_key), raw_user, raw_item, tokens),
        )

    all_tokens = {t for recs in history_records.values() for rec in recs for t in rec.token_ids}
    surfaces = _surface_map(tokenizer, all_tokens)
    tail_stats = build_tail_token_stats(
        [list(rec.token_ids) for recs in history_records.values() for rec in recs[:3]],
        tokenizer=tokenizer,
        ignored_token_ids=skip,
    )
    content_filter = lambda token_id: is_content_token(tokenizer, token_id, skip)

    from graph_llm.aux.prompt_utils import item_meta_from_row

    for row_key, row in subset.head(30).iterrows():
        raw_user = str(row["user"])
        raw_item = str(row["item"])
        history = history_records[raw_user]
        _title, _description, item_text = item_meta_from_row(raw_item, item_meta)
        target_ids = {
            int(t)
            for t in tokenizer(item_text, add_special_tokens=False)["input_ids"]
            if int(t) not in skip
        }
        _compare(
            history,
            exclude_row_key=int(row_key),
            target_raw_item=raw_item,
            skip_token_ids=skip,
            tail_stats=tail_stats,
            target_item_token_ids=target_ids,
            content_filter=content_filter,
            surfaces=surfaces,
        )
