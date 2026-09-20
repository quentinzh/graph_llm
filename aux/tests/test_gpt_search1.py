"""gpt_search1 validation-only 搜索与 prompt 截断的轻量测试。"""

from __future__ import annotations

from types import SimpleNamespace

from graph_llm.aux.prompt_utils import build_generation_prompt_batch
from graph_llm.gpt_search1 import (
    TrialResult,
    build_trial_args,
    pick_best_trial,
)
from graph_llm.train.trainer import validation_selection_key


class CharacterTokenizer:
    """用字符编码精确检查截断后缀，不依赖真实 tokenizer。"""

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(char) for char in str(text)]}


def _trial(tag: str, metrics: dict[str, float]) -> TrialResult:
    return TrialResult(
        stage="test",
        lambda_feat=0.03,
        tail_weight_max=1.5,
        lambda_prefix_feature=0.05,
        lora_r=16,
        lora_alpha=32,
        ckpt_dir="ckpt",
        log_dir="log",
        output_dir="output",
        metrics=metrics,
        tag=tag,
    )


def test_grounded_sequence_requires_fmr_before_sequence_score():
    below = {"BLEU-1": 20.0, "rouge_l": 20.0, "FMR": 0.169}
    feasible = {"BLEU-1": 12.0, "rouge_l": 12.0, "FMR": 0.170}
    assert validation_selection_key(
        feasible,
        mode="grounded_sequence",
        fmr_threshold=0.17,
    ) > validation_selection_key(
        below,
        mode="grounded_sequence",
        fmr_threshold=0.17,
    )


def test_pick_best_trial_balances_bleu_and_rouge_after_fmr_gate():
    trials = [
        # 即使不均衡配置的均值更高，也应优先提高两项中的短板。
        _trial("imbalanced", {"BLEU-1": 14.0, "rouge_l": 11.0, "FMR": 0.18}),
        _trial("balanced", {"BLEU-1": 12.2, "rouge_l": 12.1, "FMR": 0.17}),
    ]
    assert pick_best_trial(trials, 0.17).tag == "balanced"


def test_generation_prompt_truncation_preserves_user_suffix():
    tokenizer = CharacterTokenizer()
    prompt_ids, prompt_mask = build_generation_prompt_batch(
        ["A" * 100],
        ["user_a"],
        tokenizer,
        pad_token_id=0,
        max_tokens=32,
    )
    text = "".join(chr(value) for value in prompt_ids[0][prompt_mask[0].bool()].tolist())
    assert len(text) == 32
    assert text.endswith(' for user_a is "')


def test_build_trial_args_isolates_search_directories():
    base = SimpleNamespace(
        ckpt_dir="/tmp/ckpt",
        log_dir="/tmp/log",
        output_dir="/tmp/output",
        only_eval=False,
        skip_test_evaluation=False,
        force=False,
    )
    args = build_trial_args(
        base,
        stage="stage1_feature_tail",
        lambda_feat=0.03,
        tail_weight_max=1.5,
        lambda_prefix_feature=0.05,
        lora_r=32,
        lora_alpha=64,
    )
    assert args.lambda_feat == 0.03
    assert args.tail_weight_max == 1.5
    assert args.lora_r == 32
    assert args.skip_test_evaluation is True
    assert "gpt_search1/stage1_feature_tail" in args.ckpt_dir
