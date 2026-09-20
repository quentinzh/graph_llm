"""P1 搜索 --start_stage 续跑与 md 解析测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from graph_llm.p1_search import (
    STAGE1_NAME,
    STAGE2_NAME,
    _section_table_lines,
    experiment_tag,
    parse_stage_trials_from_results,
    resolve_best_from_loaded_trials,
)


FIXTURE_MD = Path(__file__).resolve().parents[2] / (
    "log/p1_search/Amazon/MoviesAndTV_corsa_filtered_small_15pct/p1_search_results.md"
)


@pytest.fixture
def frozen_p0() -> tuple[float, float, int]:
    return 0.1, 1.0, 5


def test_section_table_lines_reads_stage1(frozen_p0):
    text = FIXTURE_MD.read_text(encoding="utf-8")
    rows = _section_table_lines(text, "Stage 1: lambda_selector")
    assert len(rows) == 3
    assert all("stage1_lambda_selector" in row for row in rows)


def test_parse_stage1_trials_from_fixture(frozen_p0):
    if not FIXTURE_MD.is_file():
        pytest.skip("fixture results md missing")
    trials = parse_stage_trials_from_results(
        FIXTURE_MD,
        STAGE1_NAME,
        frozen_lambda_feat=frozen_p0[0],
        frozen_evidence_bonus=frozen_p0[1],
        frozen_top_m_evidence=frozen_p0[2],
    )
    assert len(trials) == 3
    tags = {trial.tag for trial in trials}
    assert experiment_tag(0.2, 3.0, 0.1) in tags
    best = [trial for trial in trials if trial.is_best]
    assert len(best) == 1
    assert best[0].lambda_selector == 0.2
    assert best[0].metrics.get("FMR", 0) > 0.1


def test_resolve_best_from_loaded_trials_cli_override(frozen_p0):
    if not FIXTURE_MD.is_file():
        pytest.skip("fixture results md missing")
    trials = parse_stage_trials_from_results(
        FIXTURE_MD,
        STAGE1_NAME,
        frozen_lambda_feat=frozen_p0[0],
        frozen_evidence_bonus=frozen_p0[1],
        frozen_top_m_evidence=frozen_p0[2],
    )
    assert resolve_best_from_loaded_trials(
        trials,
        param_name="lambda_selector",
        cli_value=0.2,
    ) == 0.2
    assert resolve_best_from_loaded_trials(
        trials,
        param_name="lambda_selector",
        cli_value=None,
    ) == 0.2


def test_start_stage2_dry_run_tags_use_best_lsel(monkeypatch, frozen_p0):
    """dry_run + start_stage=2 时 Stage2 tag 应以 lsel=0.2 为前缀。"""
    if not FIXTURE_MD.is_file():
        pytest.skip("fixture results md missing")

    captured: list[str] = []

    def fake_run_trial(*_args, **kwargs):
        trial = kwargs
        captured.append(
            experiment_tag(
                trial["lambda_selector"],
                trial["selector_feature_positive_weight"],
                trial["lambda_prefix_feature"],
            )
        )
        from graph_llm.p1_search import TrialResult

        return TrialResult(
            stage=kwargs["stage"],
            lambda_feat=kwargs["frozen_lambda_feat"],
            evidence_bonus=kwargs["frozen_evidence_bonus"],
            top_m_evidence=kwargs["frozen_top_m_evidence"],
            lambda_selector=kwargs["lambda_selector"],
            selector_feature_positive_weight=kwargs["selector_feature_positive_weight"],
            lambda_prefix_feature=kwargs["lambda_prefix_feature"],
        )

    import graph_llm.p1_search as p1_search

    monkeypatch.setattr(p1_search, "run_trial", fake_run_trial)
    monkeypatch.setattr(
        p1_search,
        "resolve_frozen_p0_params",
        lambda *_args, **_kwargs: frozen_p0,
    )
    monkeypatch.setattr(
        p1_search,
        "refresh_results_file",
        lambda *_args, **_kwargs: None,
    )

    argv = [
        "p1_search.py",
        "--dataset_name",
        "Amazon/MoviesAndTV_corsa_filtered_small_15pct/",
        "--split_indices",
        "1",
        "--devices",
        "1",
        "--start_stage",
        "2",
        "--lambda_selector",
        "0.2",
        "--dry_run",
        "--results_file",
        str(FIXTURE_MD),
        "--p0_results_file",
        str(
            Path(__file__).resolve().parents[2]
            / "log/p0_search/Amazon/MoviesAndTV_corsa_filtered_small_15pct/p0_search_results.md"
        ),
    ]
    monkeypatch.setattr("graph_llm.p1_search.sys.argv", argv)
    p1_search.main()

    assert len(captured) == 6  # 3 stage2 + 3 stage3 dry_run tags
    stage2_tags = captured[:3]
    assert all(tag.startswith("lsel0.2_") for tag in stage2_tags)
    assert experiment_tag(0.2, 3.0, 0.1) in stage2_tags


def test_parse_stage2_from_fixture(frozen_p0):
    if not FIXTURE_MD.is_file():
        pytest.skip("fixture results md missing")
    trials = parse_stage_trials_from_results(
        FIXTURE_MD,
        STAGE2_NAME,
        frozen_lambda_feat=frozen_p0[0],
        frozen_evidence_bonus=frozen_p0[1],
        frozen_top_m_evidence=frozen_p0[2],
    )
    assert len(trials) == 3
    # 续跑前旧 Stage2 行仍为 lsel=0.3（待 --start_stage 2 重跑后替换）
    assert all(trial.lambda_selector == 0.3 for trial in trials)
