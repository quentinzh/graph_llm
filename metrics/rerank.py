"""Generation-time reranking for multi-candidate decoding."""

from __future__ import annotations

from dataclasses import dataclass

from graph_llm.metrics.metrics import feature_detect, ids2words, ids_clear


# 高频泛化短句模板，命中则扣分。
GENERIC_PHRASES = (
    "good movie",
    "great movie",
    "great acting",
    "very nice",
    "really good",
    "well done",
    "highly recommend",
)


@dataclass
class RerankWeights:
    """rerank 各项权重，默认与文档 10.6 一致。"""

    logprob: float = 1.0
    feature_match: float = 0.8
    evidence_coverage: float = 0.5
    repetition: float = 0.7
    generic: float = 0.5


def _normalize_logprob(logprob: float) -> float:
    # 将 logprob 压到 [0, 1]，避免量纲过大主导总分。
    return max(0.0, min(1.0, (float(logprob) + 5.0) / 5.0))


def _feature_match_score(candidate_tokens: list[str], keyword_words: str) -> float:
    if not keyword_words:
        return 0.0
    feature_batch = feature_detect([candidate_tokens], {keyword_words})
    return 1.0 if keyword_words in feature_batch[0] else 0.0


def _evidence_token_coverage(candidate_ids: list[int], evidence_token_ids, evidence_token_mask) -> float:
    if evidence_token_ids is None or evidence_token_mask is None:
        return 0.0
    evidence_ids = [
        int(token_id)
        for token_id, valid in zip(evidence_token_ids.tolist(), evidence_token_mask.tolist())
        if valid
    ]
    if not evidence_ids:
        return 0.0
    candidate_set = set(int(token_id) for token_id in candidate_ids)
    hit = sum(1 for token_id in evidence_ids if token_id in candidate_set)
    return hit / len(evidence_ids)


def _repetition_penalty(candidate_tokens: list[str]) -> float:
    if not candidate_tokens:
        return 0.0
    max_run = 1
    current_run = 1
    for idx in range(1, len(candidate_tokens)):
        if candidate_tokens[idx] == candidate_tokens[idx - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    bigram_total = max(len(candidate_tokens) - 1, 1)
    bigrams = [tuple(candidate_tokens[idx : idx + 2]) for idx in range(bigram_total)]
    repeated_bigrams = len(bigrams) - len(set(bigrams))
    run_penalty = min(1.0, max(0, max_run - 1) / 3.0)
    bigram_penalty = min(1.0, repeated_bigrams / bigram_total)
    return 0.5 * run_penalty + 0.5 * bigram_penalty


def _generic_sentence_penalty(candidate_tokens: list[str]) -> float:
    text = " ".join(candidate_tokens).lower()
    hits = sum(1 for phrase in GENERIC_PHRASES if phrase in text)
    return min(1.0, hits / max(len(GENERIC_PHRASES), 1))


def score_candidate(
    candidate_ids: list[int],
    keyword_words: str,
    normalized_logprob: float,
    tokenizer,
    evidence_token_ids=None,
    evidence_token_mask=None,
    pad_token_id: int = 0,
    eos_token_ids=(2,),
    skip_token_ids=(1,),
    weights: RerankWeights | None = None,
) -> float:
    """对单条候选计算 rerank 分数。"""
    weights = weights or RerankWeights()
    cleaned_ids = ids_clear(
        candidate_ids,
        pad_token_id=pad_token_id,
        eos_token_ids=eos_token_ids,
        skip_token_ids=skip_token_ids,
    )
    candidate_tokens = ids2words(cleaned_ids, tokenizer)
    feature_score = _feature_match_score(candidate_tokens, keyword_words)
    evidence_score = _evidence_token_coverage(cleaned_ids, evidence_token_ids, evidence_token_mask)
    repetition_score = _repetition_penalty(candidate_tokens)
    generic_score = _generic_sentence_penalty(candidate_tokens)
    return (
        weights.logprob * _normalize_logprob(normalized_logprob)
        + weights.feature_match * feature_score
        + weights.evidence_coverage * evidence_score
        - weights.repetition * repetition_score
        - weights.generic * generic_score
    )


def select_best_by_logprob(candidates: list[list[int]], logprobs: list[float]) -> list[int]:
    """不做 rerank 时，直接选 normalized logprob 最高的候选。"""
    if not candidates:
        return []
    best_idx = max(range(len(candidates)), key=lambda idx: float(logprobs[idx]))
    return candidates[best_idx]


def rerank_candidates(
    candidates: list[list[int]],
    logprobs: list[float],
    keyword_words: str,
    tokenizer,
    evidence_token_ids=None,
    evidence_token_mask=None,
    pad_token_id: int = 0,
    eos_token_ids=(2,),
    skip_token_ids=(1,),
    weights: RerankWeights | None = None,
) -> list[int]:
    """从多条候选中选出 rerank 分数最高的一条。"""
    if not candidates:
        return []
    if len(candidates) == 1:
        return candidates[0]

    best_idx = 0
    best_score = float("-inf")
    for idx, (candidate_ids, logprob) in enumerate(zip(candidates, logprobs)):
        score = score_candidate(
            candidate_ids,
            keyword_words,
            logprob,
            tokenizer,
            evidence_token_ids=evidence_token_ids,
            evidence_token_mask=evidence_token_mask,
            pad_token_id=pad_token_id,
            eos_token_ids=eos_token_ids,
            skip_token_ids=skip_token_ids,
            weights=weights,
        )
        if score > best_score:
            best_score = score
            best_idx = idx
    return candidates[best_idx]


def rerank_batch(
    batch_candidates: list[list[list[int]]],
    batch_logprobs: list[list[float]],
    keyword_words_batch: list[str],
    tokenizer,
    evidence_token_ids=None,
    evidence_token_mask=None,
    use_rerank: bool = True,
    pad_token_id: int = 0,
    eos_token_ids=(2,),
    skip_token_ids=(1,),
    weights: RerankWeights | None = None,
) -> list[list[int]]:
    """对一个 batch 的多候选结果做逐样本选优。"""
    selected = []
    for sample_idx, (candidates, logprobs) in enumerate(zip(batch_candidates, batch_logprobs)):
        sample_evidence_ids = None
        sample_evidence_mask = None
        if evidence_token_ids is not None:
            sample_evidence_ids = evidence_token_ids[sample_idx]
        if evidence_token_mask is not None:
            sample_evidence_mask = evidence_token_mask[sample_idx]

        if use_rerank:
            chosen = rerank_candidates(
                candidates,
                logprobs,
                keyword_words_batch[sample_idx],
                tokenizer,
                evidence_token_ids=sample_evidence_ids,
                evidence_token_mask=sample_evidence_mask,
                pad_token_id=pad_token_id,
                eos_token_ids=eos_token_ids,
                skip_token_ids=skip_token_ids,
                weights=weights,
            )
        else:
            chosen = select_best_by_logprob(candidates, logprobs)
        selected.append(chosen)
    return selected
