"""解释阶段样本与 tail/feature 加权工具。"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from graph_llm.dataload.sequential_data import InteractionRecord, SequentialDatasetBundle, history_before


def _postprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r" +", " ", text).strip()
    return text


def _tokens(text: str) -> list[str]:
    return _postprocess(text).split()


@dataclass
class ExplainSample:
    interaction_id: int
    user_index: int
    target_item_index: int
    target_raw_item: str
    summary: str
    feature: str
    split: str  # train_rec | val | test


def build_explain_samples(
    bundle: SequentialDatasetBundle,
    *,
    positive_threshold: float = 0.0,
) -> tuple[list[ExplainSample], list[ExplainSample], list[ExplainSample]]:
    def ok(rec: InteractionRecord) -> bool:
        if positive_threshold > 0 and (rec.rating_raw is None or rec.rating_raw < positive_threshold):
            return False
        return bool((rec.summary or "").strip())

    train: list[ExplainSample] = []
    for iid in sorted(bundle.rec_train_ids):
        rec = bundle.interaction_by_id[iid]
        if not ok(rec):
            continue
        train.append(
            ExplainSample(
                interaction_id=iid,
                user_index=rec.user_index,
                target_item_index=rec.item_index,
                target_raw_item=rec.raw_item,
                summary=rec.summary.strip(),
                feature=str(rec.feature or "").strip(),
                split="train_rec",
            )
        )
    val: list[ExplainSample] = []
    for iid in bundle.val_sample_ids:
        rec = bundle.interaction_by_id[iid]
        if not ok(rec):
            continue
        val.append(
            ExplainSample(
                interaction_id=iid,
                user_index=rec.user_index,
                target_item_index=rec.item_index,
                target_raw_item=rec.raw_item,
                summary=rec.summary.strip(),
                feature=str(rec.feature or "").strip(),
                split="val",
            )
        )
    test: list[ExplainSample] = []
    for iid in bundle.test_sample_ids:
        rec = bundle.interaction_by_id[iid]
        if not ok(rec):
            continue
        test.append(
            ExplainSample(
                interaction_id=iid,
                user_index=rec.user_index,
                target_item_index=rec.item_index,
                target_raw_item=rec.raw_item,
                summary=rec.summary.strip(),
                feature=str(rec.feature or "").strip(),
                split="test",
            )
        )
    return train, val, test


def compute_tail_df_weights(
    train_samples: list[ExplainSample],
    *,
    alpha: float,
    w_min: float,
    w_max: float,
) -> tuple[dict[str, float], float]:
    """按训练 summary 的 document frequency 计算 token 权重。"""
    df: Counter[str] = Counter()
    for s in train_samples:
        toks = set(_tokens(s.summary))
        for t in toks:
            if t:
                df[t] += 1
    if not df:
        return {}, 1.0
    ref = sorted(df.values())[int(0.75 * (len(df) - 1))]
    d_ref = float(ref) if ref > 0 else 1.0
    weights: dict[str, float] = {}
    for t, c in df.items():
        r = ((c + 1.0) / (d_ref + 1.0)) ** (-alpha)
        weights[t] = max(w_min, min(w_max, r))
    return weights, d_ref


def sequence_token_weights(
    summary: str,
    tail_weights: dict[str, float],
    feature_word: str,
    *,
    gamma: float,
) -> list[float]:
    """单条 summary 的 token 权重 w(t)=w_tail(t)·(1+γ·1[feature])。"""
    feat = _postprocess(feature_word)
    out: list[float] = []
    for tok in _tokens(summary):
        w = tail_weights.get(tok, 1.0)
        if feat and tok == feat:
            w *= 1.0 + gamma
        out.append(w)
    return out


def fragment_evidence_label(fragment_text: str, summary: str) -> float:
    """弱监督：片段与目标 summary 有词重叠则为正例。"""
    frag = set(_tokens(fragment_text))
    summ = set(_tokens(summary))
    if not frag or not summ:
        return 0.0
    return 1.0 if frag & summ else 0.0


def history_for_explain_sample(bundle: SequentialDatasetBundle, sample: ExplainSample) -> list[InteractionRecord]:
    rec = bundle.interaction_by_id[sample.interaction_id]
    include_val = sample.split == "test"
    return history_before(
        bundle,
        sample.user_index,
        rec.timestamp,
        exclude_interaction_id=sample.interaction_id,
        include_val=include_val,
    )
