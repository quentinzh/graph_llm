"""泄漏安全的 prototype 检索与 overlap 打分（仅用于 rerank，不进 prompt）。"""

from __future__ import annotations

from collections import defaultdict

from graph_llm.metrics.metrics import postprocessing


def _tokenize_text(text: str) -> list[str]:
    """将文本规范化为 token 列表。"""
    return [token for token in postprocessing(str(text).lower()).split() if token]


def _shorten_text(text: str, max_words: int = 14) -> str:
    """截断 prototype 到固定词数，避免 rerank 被长句主导。"""
    words = _tokenize_text(text)
    return " ".join(words[:max_words])


class PrototypeIndex:
    """基于 train/valid 历史构建 item/user 侧 prototype 索引。"""

    def __init__(self, history_df):
        # item 侧：raw_item -> [(raw_user, review_text), ...]
        self._item_entries: dict[str, list[tuple[str, str]]] = defaultdict(list)
        # user 侧：raw_user -> [(raw_item, review_text), ...]
        self._user_entries: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for _, row in history_df.iterrows():
            raw_user = str(row["raw_user"])
            raw_item = str(row["raw_item"])
            review_text = row.get("review_text", "")
            if (not review_text or str(review_text).strip() == "") and "template" in row:
                review_text = row["template"][2]
            review_text = str(review_text).strip()
            if not review_text:
                continue
            self._item_entries[raw_item].append((raw_user, review_text))
            self._user_entries[raw_user].append((raw_item, review_text))

    @classmethod
    def from_history(cls, history_df) -> "PrototypeIndex":
        return cls(history_df)

    def get_prototypes(self, raw_user: str, raw_item: str, k: int = 3) -> list[str]:
        """优先 item-side（同 item、异 user），不足再 user-side（同 user、异 item）。"""
        raw_user = str(raw_user)
        raw_item = str(raw_item)
        seen: set[str] = set()
        prototypes: list[str] = []

        for user, text in self._item_entries.get(raw_item, []):
            if user == raw_user:
                continue
            short_text = _shorten_text(text)
            if not short_text or short_text in seen:
                continue
            seen.add(short_text)
            prototypes.append(short_text)
            if len(prototypes) >= k:
                return prototypes[:k]

        for item, text in self._user_entries.get(raw_user, []):
            if item == raw_item:
                continue
            short_text = _shorten_text(text)
            if not short_text or short_text in seen:
                continue
            seen.add(short_text)
            prototypes.append(short_text)
            if len(prototypes) >= k:
                break
        return prototypes[:k]


def token_f1(candidate_tokens: list[str], reference_tokens: list[str]) -> float:
    """候选与 prototype 的 token 级 F1。"""
    if not candidate_tokens or not reference_tokens:
        return 0.0
    cand_set = set(candidate_tokens)
    ref_set = set(reference_tokens)
    overlap = len(cand_set & ref_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand_set)
    recall = overlap / len(ref_set)
    return 2.0 * precision * recall / (precision + recall + 1e-12)


def prototype_overlap(candidate_tokens: list[str], prototypes: list[str]) -> float:
    """对多条 prototype 取最大 token F1，作为 overlap 分数。"""
    if not candidate_tokens or not prototypes:
        return 0.0
    scores = []
    for proto in prototypes:
        ref_tokens = _tokenize_text(proto)
        if not ref_tokens:
            continue
        scores.append(token_f1(candidate_tokens, ref_tokens))
    return max(scores) if scores else 0.0
