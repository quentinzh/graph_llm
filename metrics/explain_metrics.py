"""解释生成指标封装（复用 rober CIER 评测）。"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rober.metrics.cier_eval import compute_all_metrics, postprocess, tokenize_text


def evaluate_explanations(
    references: list[str],
    predictions: list[str],
    feature_words: list[str],
) -> dict[str, float]:
    ref_toks = [tokenize_text(r) for r in references]
    pred_toks = [tokenize_text(p) for p in predictions]
    feats = [postprocess(f) for f in feature_words]
    return compute_all_metrics(ref_toks, pred_toks, feats)


def diversity_composite_score(metrics: dict[str, float]) -> float:
    """FCR、1-DIV、D-1、ENTR 等权复合（越高越好）。"""
    fcr = float(metrics.get("FCR", 0.0))
    div = float(metrics.get("DIV", 0.0))
    d1 = float(metrics.get("Distinct-1", 0.0))
    entr = float(metrics.get("ENTR", 0.0))
    return 0.25 * (fcr + (1.0 - div) + d1 + entr / 10.0)


def passes_quality_gate(
    metrics: dict[str, float],
    baseline: dict[str, float],
    *,
    min_ratio: float = 0.95,
) -> bool:
    fmr = metrics.get("FMR", 0.0)
    b4 = metrics.get("B-4", 0.0)
    base_fmr = baseline.get("FMR", 0.0)
    base_b4 = baseline.get("B-4", 0.0)
    if base_fmr > 0 and fmr < base_fmr * min_ratio:
        return False
    if base_b4 > 0 and b4 < base_b4 * min_ratio:
        return False
    return True
