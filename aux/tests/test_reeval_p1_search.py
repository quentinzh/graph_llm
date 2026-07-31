"""reeval_p1_search 路径发现与选参逻辑的轻量测试。"""

from __future__ import annotations

import json
from pathlib import Path

from graph_llm.p1_search import TrialResult
from graph_llm.reeval_p1_search import (
    choose_final_hyperparams,
    default_device_choice,
    discover_trials,
    resolve_devices,
)


def test_resolve_devices_priority(tmp_path: Path):
    assert resolve_devices("gpu", "cpu") == "cpu"
    assert resolve_devices("cpu", "") == "cpu"
    assert resolve_devices("gpu", "") == "1"


def test_discover_trials_nested_dataset(tmp_path: Path):
    root = tmp_path / "p1_search"
    trial = (
        root
        / "stage1_lambda_selector"
        / "lsel0.3_sfpw3_lpfx0.1"
        / "Amazon"
        / "MoviesAndTV_corsa_filtered_small_15pct"
    )
    model_dir = trial / "1model"
    model_dir.mkdir(parents=True)
    (model_dir / "adapter_model.safetensors").write_bytes(b"x")
    cfg = {
        "lambda_feat": 0.1,
        "evidence_bonus": 1.0,
        "top_m_evidence": 5,
        "lambda_selector": 0.3,
        "selector_feature_positive_weight": 3.0,
        "lambda_prefix_feature": 0.1,
    }
    (trial / "1graph_config.json").write_text(json.dumps(cfg), encoding="utf-8")

    found = discover_trials(
        root,
        "Amazon/MoviesAndTV_corsa_filtered_small_15pct/",
        "1",
    )
    assert len(found) == 1
    assert found[0].stage == "stage1_lambda_selector"
    assert found[0].tag == "lsel0.3_sfpw3_lpfx0.1"
    assert found[0].trial_ckpt_dir.name == "lsel0.3_sfpw3_lpfx0.1"


def test_choose_final_hyperparams_chain():
    stage1 = [
        TrialResult(
            stage="stage1_lambda_selector",
            lambda_feat=0.1,
            evidence_bonus=1.0,
            top_m_evidence=5,
            lambda_selector=0.1,
            selector_feature_positive_weight=3.0,
            lambda_prefix_feature=0.1,
            metrics={"FMR": 0.10, "rouge_l": 5.0},
            tag="a",
        ),
        TrialResult(
            stage="stage1_lambda_selector",
            lambda_feat=0.1,
            evidence_bonus=1.0,
            top_m_evidence=5,
            lambda_selector=0.3,
            selector_feature_positive_weight=3.0,
            lambda_prefix_feature=0.1,
            metrics={"FMR": 0.20, "rouge_l": 5.0},
            tag="b",
        ),
    ]
    stage2 = [
        TrialResult(
            stage="stage2_selector_feature_positive_weight",
            lambda_feat=0.1,
            evidence_bonus=1.0,
            top_m_evidence=5,
            lambda_selector=0.3,
            selector_feature_positive_weight=2.0,
            lambda_prefix_feature=0.1,
            metrics={"FMR": 0.21, "rouge_l": 5.0},
            tag="c",
        ),
        TrialResult(
            stage="stage2_selector_feature_positive_weight",
            lambda_feat=0.1,
            evidence_bonus=1.0,
            top_m_evidence=5,
            lambda_selector=0.3,
            selector_feature_positive_weight=4.0,
            lambda_prefix_feature=0.1,
            metrics={"FMR": 0.25, "rouge_l": 5.0},
            tag="d",
        ),
    ]
    stage3 = [
        TrialResult(
            stage="stage3_lambda_prefix_feature",
            lambda_feat=0.1,
            evidence_bonus=1.0,
            top_m_evidence=5,
            lambda_selector=0.3,
            selector_feature_positive_weight=4.0,
            lambda_prefix_feature=0.2,
            metrics={"FMR": 0.30, "rouge_l": 5.0},
            tag="e",
        ),
    ]
    lsel, sfpw, lpfx = choose_final_hyperparams(stage1, stage2, stage3)
    assert (lsel, sfpw, lpfx) == (0.3, 4.0, 0.2)
    assert default_device_choice() in {"cpu", "gpu"}
