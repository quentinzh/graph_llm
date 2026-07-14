"""Training-fold token popularity statistics for tail-aware graph learning."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from graph_llm.aux.prompt_utils import useful_evidence_surface


TAIL_STATS_VERSION = "tail_stats_v1"


def is_content_token(tokenizer, token_id: int, ignored_token_ids: set[int]) -> bool:
    """Return whether a vocabulary token can serve as lexical evidence."""
    token_id = int(token_id)
    if token_id < 0 or token_id in ignored_token_ids:
        return False
    surface = tokenizer.decode([token_id], skip_special_tokens=True).strip()
    return bool(useful_evidence_surface(surface))


@dataclass(frozen=True)
class TailTokenStats:
    """Fold-local document frequencies and bounded tail weights.

    All statistics are constructed from the training split only.  This keeps
    validation/test labels out of both the loss reweighting and graph cache.
    """

    doc_freq: dict[int, int]
    num_documents: int
    reference_df: float
    tail_threshold: int
    alpha: float = 0.5
    weight_min: float = 0.5
    weight_max: float = 2.0
    version: str = TAIL_STATS_VERSION

    def document_frequency(self, token_id: int) -> int:
        return int(self.doc_freq.get(int(token_id), 0))

    def tail_weight(self, token_id: int) -> float:
        """Return 1.0 for train-unseen tokens to avoid speculative boosting."""
        df = self.document_frequency(token_id)
        if df <= 0:
            return 1.0
        raw = ((df + 1.0) / (self.reference_df + 1.0)) ** (-self.alpha)
        return float(np.clip(raw, self.weight_min, self.weight_max))

    def is_tail(self, token_id: int) -> bool:
        df = self.document_frequency(token_id)
        return 0 < df <= self.tail_threshold

    def frequency_bucket(self, token_id: int) -> int:
        """Log-frequency bucket used for popularity-matched negative sampling."""
        return int(math.floor(math.log2(self.document_frequency(token_id) + 1)))

    def weight_table(self, vocab_size: int) -> np.ndarray:
        """Build a dense lookup table; default 1.0 preserves non-content tokens."""
        table = np.ones(int(vocab_size), dtype=np.float32)
        for token_id in self.doc_freq:
            if 0 <= int(token_id) < int(vocab_size):
                table[int(token_id)] = self.tail_weight(int(token_id))
        return table

    @property
    def fingerprint(self) -> str:
        payload = {
            "version": self.version,
            "num_documents": self.num_documents,
            "reference_df": round(float(self.reference_df), 8),
            "tail_threshold": self.tail_threshold,
            "alpha": self.alpha,
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "doc_freq": sorted((int(k), int(v)) for k, v in self.doc_freq.items()),
        }
        text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_tail_token_stats(
    token_sequences: Iterable[Iterable[int]],
    *,
    tokenizer,
    ignored_token_ids: set[int],
    alpha: float = 0.5,
    weight_min: float = 0.5,
    weight_max: float = 2.0,
    tail_df_fraction: float = 0.001,
    tail_df_minimum: int = 5,
) -> TailTokenStats:
    """Construct leakage-safe popularity statistics from training targets."""
    if alpha != 0.5:
        raise ValueError("tail alpha is fixed to 0.5 by the current training design")
    if not 0 < weight_min <= 1.0:
        raise ValueError("tail_weight_min must be in (0, 1]")
    if weight_max < 1.0:
        raise ValueError("tail_weight_max must be at least 1")
    if tail_df_fraction <= 0:
        raise ValueError("tail_df_fraction must be positive")
    if tail_df_minimum < 1:
        raise ValueError("tail_df_minimum must be positive")

    content_sequences: list[list[int]] = []
    doc_freq: Counter[int] = Counter()
    for sequence in token_sequences:
        content_ids = [
            int(token_id)
            for token_id in sequence
            if is_content_token(tokenizer, int(token_id), ignored_token_ids)
        ]
        content_sequences.append(content_ids)
        doc_freq.update(set(content_ids))

    num_documents = len(content_sequences)
    position_df = [
        doc_freq[token_id]
        for content_ids in content_sequences
        for token_id in content_ids
    ]
    reference_df = float(np.quantile(position_df, 0.75)) if position_df else 1.0
    tail_threshold = max(
        int(tail_df_minimum),
        int(math.ceil(float(tail_df_fraction) * max(num_documents, 1))),
    )
    return TailTokenStats(
        doc_freq={int(token_id): int(count) for token_id, count in doc_freq.items()},
        num_documents=num_documents,
        reference_df=reference_df,
        tail_threshold=tail_threshold,
        alpha=float(alpha),
        weight_min=float(weight_min),
        weight_max=float(weight_max),
    )


__all__ = [
    "TAIL_STATS_VERSION",
    "TailTokenStats",
    "build_tail_token_stats",
    "is_content_token",
]
