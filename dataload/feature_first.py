"""Feature-first SFT 的目标拼接与生成结果解析工具。"""

from __future__ import annotations


FEATURE_PROMPT_SUFFIX = "\n<feature> "
FEATURE_EXPLANATION_SEPARATOR = " </feature>\n<explanation> "


def encode_without_special_tokens(tokenizer, text: str) -> list[int]:
    """统一把控制文本编码为普通 token，避免隐式插入 BOS/EOS。"""
    return [
        int(token_id)
        for token_id in tokenizer(text, add_special_tokens=False)["input_ids"]
    ]


def feature_separator_token_ids(tokenizer) -> list[int]:
    return encode_without_special_tokens(tokenizer, FEATURE_EXPLANATION_SEPARATOR)


def find_subsequence(sequence, pattern) -> int:
    """返回 pattern 第一次出现的位置；未找到时返回 -1。"""
    sequence = [int(token_id) for token_id in sequence]
    pattern = [int(token_id) for token_id in pattern]
    if not pattern or len(pattern) > len(sequence):
        return -1
    width = len(pattern)
    for start in range(len(sequence) - width + 1):
        if sequence[start:start + width] == pattern:
            return start
    return -1


def split_feature_first_ids(sequence, separator_ids, *, pad_token_id=None, eos_token_ids=()):
    """把 ``feature + separator + explanation`` 拆成两个 token 序列。

    对未生成分隔符的样本，解释返回空序列。这样格式失败会真实反映在
    BLEU/ROUGE/FMR 中，而不会误把 feature 当成解释参与评估。
    """
    eos_ids = {int(token_id) for token_id in eos_token_ids}
    clean = []
    for token_id in sequence:
        token_id = int(token_id)
        if pad_token_id is not None and token_id == int(pad_token_id):
            break
        if token_id in eos_ids:
            break
        clean.append(token_id)

    split_at = find_subsequence(clean, separator_ids)
    if split_at < 0:
        return clean, []
    explanation_start = split_at + len(separator_ids)
    return clean[:split_at], clean[explanation_start:]


def contains_suffix(sequence, suffix) -> bool:
    """判断已生成序列是否刚好以控制分隔符结尾。"""
    if not suffix or len(sequence) < len(suffix):
        return False
    return [int(x) for x in sequence[-len(suffix):]] == [int(x) for x in suffix]


__all__ = [
    "FEATURE_PROMPT_SUFFIX",
    "FEATURE_EXPLANATION_SEPARATOR",
    "contains_suffix",
    "encode_without_special_tokens",
    "feature_separator_token_ids",
    "find_subsequence",
    "split_feature_first_ids",
]
