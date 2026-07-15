"""仅从训练集检索 item/user prototype 的工具。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from graph_llm.aux.prompt_utils import clean_ws
from graph_llm.dataload.legacy_data import profile_text_from_record


@dataclass(frozen=True)
class PrototypeCandidate:
    """一条可被检索的训练集评论及其必要元数据。"""

    row_key: int
    raw_user: str
    raw_item: str
    review_text: str


@dataclass(frozen=True)
class PrototypeBatch:
    """一个 batch 的双 prototype 结果与可用性统计。"""

    item_prototypes: list[str]
    user_prototypes: list[str]
    item_available: int
    user_available: int


class TrainingOnlyPrototypeRetriever:
    """用冻结 embedding 从训练评论中选择两个无泄漏 prototype。

    item prototype 的候选限定为同 item、其他用户的训练评论，按用户画像余弦
    相似度选取；user prototype 的候选限定为同 user、其他 item 的训练评论，按
    当前 item 文本余弦相似度选取。两种选择都不混入图分数、质量分或人工权重。
    """

    def __init__(
        self,
        train_dataframe,
        profile_records,
        embedding_encoder,
        *,
        embedding_batch_size: int = 16,
    ):
        self.profile_records = profile_records or {}
        self.embedding_encoder = embedding_encoder
        self.embedding_batch_size = max(int(embedding_batch_size), 1)
        self.candidates: list[PrototypeCandidate] = []
        self.item_to_candidate_ids: dict[str, list[int]] = defaultdict(list)
        self.user_to_candidate_ids: dict[str, list[int]] = defaultdict(list)
        # 内存缓存避免同一条训练评论在不同测试样本中被反复编码；底层 encoder
        # 仍会把向量持久化到它已有的 embedding cache。
        self._review_embeddings: dict[int, torch.Tensor] = {}

        required = {"raw_user", "raw_item", "review_text"}
        missing = required - set(train_dataframe.columns)
        if missing:
            raise ValueError(f"Prototype retrieval misses required columns: {sorted(missing)}")

        for row_key, row in train_dataframe.iterrows():
            review_text = clean_ws(row["review_text"])
            if not review_text:
                continue
            candidate_id = len(self.candidates)
            candidate = PrototypeCandidate(
                row_key=int(row_key),
                raw_user=str(row["raw_user"]),
                raw_item=str(row["raw_item"]),
                review_text=review_text,
            )
            self.candidates.append(candidate)
            self.item_to_candidate_ids[candidate.raw_item].append(candidate_id)
            self.user_to_candidate_ids[candidate.raw_user].append(candidate_id)

    def __len__(self) -> int:
        return len(self.candidates)

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        """兼容真实 encoder 与轻量测试 stub 的文本编码入口。"""
        try:
            return self.embedding_encoder.encode_texts(
                texts,
                batch_size=self.embedding_batch_size,
            )
        except TypeError:
            return self.embedding_encoder.encode_texts(texts)

    def _candidate_embeddings(self, candidate_ids: list[int]) -> torch.Tensor:
        missing_ids = [idx for idx in candidate_ids if idx not in self._review_embeddings]
        if missing_ids:
            texts = [self.candidates[idx].review_text for idx in missing_ids]
            vectors = self._encode_texts(texts).detach().cpu().float()
            for idx, vector in zip(missing_ids, vectors):
                self._review_embeddings[idx] = vector
        return torch.stack([self._review_embeddings[idx] for idx in candidate_ids], dim=0)

    def _select_by_cosine(
        self,
        query_embedding: torch.Tensor,
        candidate_ids: list[int],
    ) -> str:
        if not candidate_ids:
            return ""
        candidate_vectors = self._candidate_embeddings(candidate_ids).to(query_embedding.device)
        query = F.normalize(query_embedding.float(), p=2, dim=0)
        candidates = F.normalize(candidate_vectors.float(), p=2, dim=1)
        scores = torch.mv(candidates, query)
        # candidate_ids 保持训练数据的确定性顺序，argmax 的首个最大值可复现。
        selected_idx = candidate_ids[int(torch.argmax(scores).item())]
        return self.candidates[selected_idx].review_text

    def retrieve_batch(
        self,
        raw_users: list[str],
        raw_items: list[str],
        item_texts: list[str],
        *,
        excluded_row_keys: list[int | None] | None = None,
    ) -> PrototypeBatch:
        """为 batch 选择训练集内的 item/user prototype。

        ``excluded_row_keys`` 使训练期也能安全复用本类；当前第一阶段仅在验证/测试
        调用，因此所有候选天然来自不同 split 的训练集。
        """
        if not (len(raw_users) == len(raw_items) == len(item_texts)):
            raise ValueError("raw_users, raw_items and item_texts must have the same length")
        if excluded_row_keys is None:
            excluded_row_keys = [None] * len(raw_users)
        if len(excluded_row_keys) != len(raw_users):
            raise ValueError("excluded_row_keys must align with the batch")

        profile_texts = [
            profile_text_from_record(self.profile_records.get(str(raw_user)))
            for raw_user in raw_users
        ]
        profile_embeddings = self._encode_texts(profile_texts)
        item_embeddings = self._encode_texts([clean_ws(text) for text in item_texts])

        item_prototypes, user_prototypes = [], []
        item_available = user_available = 0
        for batch_idx, (raw_user, raw_item, excluded_row_key) in enumerate(
            zip(raw_users, raw_items, excluded_row_keys)
        ):
            user = str(raw_user)
            item = str(raw_item)
            # item prototype 只来自同 item 的其他用户训练评论。
            item_candidates = [
                candidate_id
                for candidate_id in self.item_to_candidate_ids.get(item, [])
                if self.candidates[candidate_id].raw_user != user
                and self.candidates[candidate_id].row_key != excluded_row_key
            ]
            # user prototype 只来自同 user 的其他 item 训练评论。
            user_candidates = [
                candidate_id
                for candidate_id in self.user_to_candidate_ids.get(user, [])
                if self.candidates[candidate_id].raw_item != item
                and self.candidates[candidate_id].row_key != excluded_row_key
            ]
            item_prototype = self._select_by_cosine(
                profile_embeddings[batch_idx], item_candidates
            )
            user_prototype = self._select_by_cosine(
                item_embeddings[batch_idx], user_candidates
            )
            item_prototypes.append(item_prototype)
            user_prototypes.append(user_prototype)
            item_available += int(bool(item_prototype))
            user_available += int(bool(user_prototype))

        return PrototypeBatch(
            item_prototypes=item_prototypes,
            user_prototypes=user_prototypes,
            item_available=item_available,
            user_available=user_available,
        )


def _ngram_f1(candidate_words: list[str], prototype_words: list[str], n: int) -> float:
    if n <= 0 or len(candidate_words) < n or len(prototype_words) < n:
        return 0.0
    candidate_ngrams = {
        tuple(candidate_words[idx:idx + n])
        for idx in range(len(candidate_words) - n + 1)
    }
    prototype_ngrams = {
        tuple(prototype_words[idx:idx + n])
        for idx in range(len(prototype_words) - n + 1)
    }
    if not candidate_ngrams or not prototype_ngrams:
        return 0.0
    overlap = len(candidate_ngrams & prototype_ngrams)
    precision = overlap / len(candidate_ngrams)
    recall = overlap / len(prototype_ngrams)
    return 2.0 * precision * recall / (precision + recall + 1e-12)


def prototype_phrase_score(candidate_words: list[str], prototype_words: list[str]) -> float:
    """同时奖励词汇和连续二元短语重合，供候选重排而非 prototype 检索使用。"""
    return 0.5 * _ngram_f1(candidate_words, prototype_words, 1) + 0.5 * _ngram_f1(
        candidate_words, prototype_words, 2
    )


def graph_evidence_coverage(candidate_words: list[str], evidence_words: list[str]) -> float:
    candidate = set(candidate_words)
    evidence = {word for word in evidence_words if word}
    if not evidence:
        return 0.0
    return len(candidate & evidence) / len(evidence)


def repetition_penalty(candidate_words: list[str]) -> float:
    if len(candidate_words) < 2:
        return 0.0
    bigrams = [tuple(candidate_words[idx:idx + 2]) for idx in range(len(candidate_words) - 1)]
    return 1.0 - len(set(bigrams)) / len(bigrams)


def generic_template_penalty(candidate_words: list[str]) -> float:
    text = " ".join(candidate_words).lower()
    generic_templates = (
        "the movie is very good",
        "the film is very good",
        "the plot is very good",
        "the acting is good and the story is interesting",
    )
    return float(any(template in text for template in generic_templates))


def select_reranked_candidate(
    candidate_words: list[list[str]],
    average_logprobs: list[float],
    item_prototype_words: list[str],
    user_prototype_words: list[str],
    evidence_words: list[str],
) -> int:
    """在不访问标签的前提下，从多候选中选出最贴合双原型的解释。"""
    if not candidate_words:
        raise ValueError("candidate_words must not be empty")
    if len(candidate_words) != len(average_logprobs):
        raise ValueError("candidate_words and average_logprobs must align")

    logprobs = torch.tensor(average_logprobs, dtype=torch.float32)
    if len(logprobs) == 1 or float(logprobs.max() - logprobs.min()) < 1e-8:
        normalized_logprobs = torch.ones_like(logprobs)
    else:
        normalized_logprobs = (logprobs - logprobs.min()) / (logprobs.max() - logprobs.min())

    best_idx, best_score = 0, float("-inf")
    for idx, words in enumerate(candidate_words):
        score = (
            0.40 * float(normalized_logprobs[idx])
            + 0.25 * prototype_phrase_score(words, item_prototype_words)
            + 0.20 * prototype_phrase_score(words, user_prototype_words)
            + 0.15 * graph_evidence_coverage(words, evidence_words)
            - 0.20 * repetition_penalty(words)
            - 0.20 * generic_template_penalty(words)
        )
        if score > best_score:
            best_idx, best_score = idx, score
    return best_idx
