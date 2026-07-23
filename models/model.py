"""Graph-guided explanation model with tail-aware SFT (no evidence text in prompt)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_llm.models.selector import EvidenceSelector, pad_token_matrix


CONTROL_STOPWORDS = {
    "", ".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "'", "\"",
    "'s", "'m", "'ve", "n't", "'re", "'d", "'ll",
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "he", "her", "his", "i", "in", "is", "it", "its", "me", "my", "of", "on",
    "or", "our", "she", "that", "the", "their", "them", "there", "they",
    "this", "to", "was", "we", "were", "with", "you", "your",
    "user", "profile", "current", "item", "information", "title",
    "description", "explanation", "useful", "token", "evidence", "none",
}


class TargetAwareReviewProjector(nn.Module):
    """用目标条件化的多查询注意力把 Top-K 评论压缩为 soft prefix。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        prefix_len: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError("review embedding dim must be divisible by attention heads")
        self.prefix_len = int(prefix_len)
        self.review_norm = nn.LayerNorm(input_dim)
        self.query_norm = nn.LayerNorm(input_dim)
        self.queries = nn.Parameter(torch.empty(self.prefix_len, input_dim))
        self.condition_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(input_dim)
        self.output_proj = nn.Linear(input_dim, output_dim)
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.dropout = nn.Dropout(dropout)
        # 初始只给 LLM 很弱的扰动，训练中再自动提高 prefix 强度。
        self.gate = nn.Parameter(torch.tensor(-2.0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.queries, std=0.02)
        nn.init.xavier_uniform_(self.condition_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.zeros_(self.type_embedding)

    def forward(self, review_embeddings, review_mask, target_query):
        if review_embeddings.ndim != 3:
            raise ValueError("review_embeddings must have shape [B, K, D]")
        if review_mask.shape != review_embeddings.shape[:2]:
            raise ValueError("review_mask must match review_embeddings[:2]")
        if target_query.shape != (
            review_embeddings.shape[0],
            review_embeddings.shape[2],
        ):
            raise ValueError("target_query must have shape [B, D]")

        valid_mask = review_mask.bool()
        has_history = valid_mask.any(dim=1)
        safe_mask = valid_mask.clone()
        safe_reviews = review_embeddings.float().clone()
        # MultiheadAttention 不接受整行都被 mask；冷启动样本放入零占位，输出随后清零。
        if (~has_history).any():
            safe_mask[~has_history, 0] = True
            safe_reviews[~has_history, 0] = 0

        reviews = self.review_norm(safe_reviews)
        conditioned = self.query_norm(target_query.float())
        queries = self.queries.unsqueeze(0).expand(reviews.shape[0], -1, -1)
        queries = queries + self.condition_proj(conditioned).unsqueeze(1)
        attended, _ = self.cross_attention(
            query=queries,
            key=reviews,
            value=reviews,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        prefix = self.output_proj(self.output_norm(attended)) + self.type_embedding
        prefix = torch.sigmoid(self.gate) * self.dropout(prefix)
        prefix = prefix * has_history[:, None, None].to(prefix.dtype)
        prefix_mask = has_history[:, None].expand(-1, self.prefix_len)
        return prefix, prefix_mask


class GraphEvidenceCIER(nn.Module):
    """Profile-aware Qwen explainer with graph evidence and tail-aware SFT."""

    def __init__(
        self,
        tokenizer,
        vocab_size,
        evidence_selector: EvidenceSelector,
        lambda_feat=0.0001,
        evidence_bonus=0.1,
        hidden_size=None,
        review_embedding_dim=None,
        user_review_prefix_len=0,
        item_review_prefix_len=0,
        review_attention_heads=8,
        review_prefix_dropout=0.1,
        lambda_prefix_feature=0.0,
        max_consecutive_token_repeat=3,
        pad_token_id=0,
        eos_token_ids=None,
        special_token_ids=(0, 1, 2),
    ):
        super().__init__()
        self.evidence_selector = evidence_selector
        self.model = None
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.lambda_feat = float(lambda_feat)
        self.evidence_bonus = float(evidence_bonus)
        self.hidden_size = int(hidden_size) if hidden_size is not None else None
        self.review_embedding_dim = (
            int(review_embedding_dim) if review_embedding_dim is not None else None
        )
        self.user_review_prefix_len = int(user_review_prefix_len)
        self.item_review_prefix_len = int(item_review_prefix_len)
        self.lambda_prefix_feature = float(lambda_prefix_feature)
        self.max_consecutive_token_repeat = int(max_consecutive_token_repeat)
        self.pad_token_id = int(pad_token_id)
        self.eos_token_ids = tuple(int(x) for x in (eos_token_ids or ()))
        self.special_token_ids = tuple(int(x) for x in special_token_ids)

        if self.user_review_prefix_len < 0 or self.item_review_prefix_len < 0:
            raise ValueError("review prefix lengths must be non-negative")
        if self.lambda_prefix_feature < 0:
            raise ValueError("lambda_prefix_feature must be non-negative")
        prefix_enabled = self.user_review_prefix_len > 0 or self.item_review_prefix_len > 0
        if prefix_enabled and (self.hidden_size is None or self.review_embedding_dim is None):
            raise ValueError(
                "hidden_size and review_embedding_dim are required when review prefix is enabled"
            )
        self.user_review_projector = (
            TargetAwareReviewProjector(
                self.review_embedding_dim,
                self.hidden_size,
                self.user_review_prefix_len,
                review_attention_heads,
                review_prefix_dropout,
            )
            if self.user_review_prefix_len > 0
            else None
        )
        self.item_review_projector = (
            TargetAwareReviewProjector(
                self.review_embedding_dim,
                self.hidden_size,
                self.item_review_prefix_len,
                review_attention_heads,
                review_prefix_dropout,
            )
            if self.item_review_prefix_len > 0
            else None
        )

    def _embed_tokens(self):
        return self.model.get_input_embeddings()

    def _llm_input_device(self):
        llm = self.model
        if llm is None:
            raise RuntimeError("LLM backbone is not attached.")
        device_map = getattr(llm, "hf_device_map", None)
        if not device_map and hasattr(llm, "base_model"):
            device_map = getattr(llm.base_model, "hf_device_map", None)
        if device_map:
            for key in (
                "base_model.model.model.embed_tokens",
                "base_model.model.embed_tokens",
                "model.model.embed_tokens",
                "model.embed_tokens",
                "transformer.wte",
            ):
                if key in device_map:
                    device = device_map[key]
                    return torch.device(device) if isinstance(device, str) else device
            first = next(iter(device_map.values()))
            return torch.device(first) if isinstance(first, str) else first
        return self._embed_tokens().weight.device

    def _ignored_token_ids(self):
        ids = set(self.special_token_ids)
        ids.add(self.pad_token_id)
        ids.update(self.eos_token_ids)
        return ids

    def _is_evidence_control_token(self, token_id: int) -> bool:
        token_id = int(token_id)
        if token_id < 0 or token_id in self._ignored_token_ids():
            return False
        if self.tokenizer is None:
            return True
        surface = self.tokenizer.decode([token_id], skip_special_tokens=True).strip().lower()
        if not surface:
            return False
        normalized = surface.strip(" \t\r\n.,!?;:'\"()[]{}")
        if not normalized or normalized in CONTROL_STOPWORDS:
            return False
        if normalized.isdigit() or len(normalized) <= 1:
            return False
        return any(ch.isalpha() for ch in normalized)

    def get_embedding(
        self,
        input_ids=None,
        profile_ids=None,
        review_prefixes=None,
        target_item_ids=None,
        generation_prompt_ids=None,
    ):
        embeddings = self._embed_tokens()
        llm_dtype = embeddings.weight.dtype
        llm_device = self._llm_input_device()
        parts = []
        if profile_ids is not None and profile_ids.numel() > 0:
            parts.append(embeddings(profile_ids.to(llm_device)).to(dtype=llm_dtype))
        if review_prefixes is not None and review_prefixes.numel() > 0:
            parts.append(review_prefixes.to(device=llm_device, dtype=llm_dtype))
        if target_item_ids is not None and target_item_ids.numel() > 0:
            parts.append(embeddings(target_item_ids.to(llm_device)).to(dtype=llm_dtype))
        if generation_prompt_ids is not None and generation_prompt_ids.numel() > 0:
            parts.append(embeddings(generation_prompt_ids.to(llm_device)).to(dtype=llm_dtype))
        if input_ids is not None and input_ids.shape[1] > 0:
            parts.append(embeddings(input_ids.to(llm_device)).to(dtype=llm_dtype))
        if not parts:
            raise ValueError("get_embedding received no non-empty prompt segments.")
        return torch.cat(parts, dim=1)

    def _attention_mask(
        self,
        input_ids=None,
        profile_mask=None,
        review_prefix_mask=None,
        target_item_mask=None,
        generation_prompt_mask=None,
        batch_size=None,
        device=None,
    ):
        masks = []
        if profile_mask is not None and profile_mask.numel() > 0:
            masks.append(profile_mask.to(device))
        if review_prefix_mask is not None and review_prefix_mask.numel() > 0:
            masks.append(review_prefix_mask.long().to(device))
        if target_item_mask is not None and target_item_mask.numel() > 0:
            masks.append(target_item_mask.to(device))
        if generation_prompt_mask is not None and generation_prompt_mask.numel() > 0:
            masks.append(generation_prompt_mask.to(device))
        if input_ids is not None and input_ids.shape[1] > 0:
            masks.append((input_ids != self.pad_token_id).long().to(device))
        return torch.cat(masks, dim=1)

    def _prompt_length(
        self,
        profile_ids=None,
        review_prefixes=None,
        target_item_ids=None,
        generation_prompt_ids=None,
    ):
        total = 0
        if profile_ids is not None:
            total += profile_ids.shape[1]
        if review_prefixes is not None:
            total += review_prefixes.shape[1]
        if target_item_ids is not None:
            total += target_item_ids.shape[1]
        if generation_prompt_ids is not None:
            total += generation_prompt_ids.shape[1]
        return total

    def build_review_prefixes(
        self,
        *,
        user_review_embeddings=None,
        user_review_mask=None,
        item_review_embeddings=None,
        item_review_mask=None,
        item_review_query=None,
        user_review_query=None,
    ):
        """分别构造用户偏好和物品属性 prefix，保持两套参数不共享。"""
        supplied = (
            user_review_embeddings,
            user_review_mask,
            item_review_embeddings,
            item_review_mask,
            item_review_query,
            user_review_query,
        )
        # 兼容不使用评论前缀的旧辅助阶段；主训练/验证/测试始终会传入完整张量。
        if all(value is None for value in supplied):
            return None, None
        prefixes, masks = [], []
        if self.user_review_projector is not None:
            if any(
                value is None
                for value in (
                    user_review_embeddings,
                    user_review_mask,
                    item_review_query,
                )
            ):
                raise ValueError("user review prefix inputs are incomplete")
            prefix, mask = self.user_review_projector(
                user_review_embeddings,
                user_review_mask,
                item_review_query,
            )
            prefixes.append(prefix)
            masks.append(mask)
        if self.item_review_projector is not None:
            if any(
                value is None
                for value in (
                    item_review_embeddings,
                    item_review_mask,
                    user_review_query,
                )
            ):
                raise ValueError("item review prefix inputs are incomplete")
            prefix, mask = self.item_review_projector(
                item_review_embeddings,
                item_review_mask,
                user_review_query,
            )
            prefixes.append(prefix)
            masks.append(mask)
        if not prefixes:
            return None, None
        return torch.cat(prefixes, dim=1), torch.cat(masks, dim=1)

    def _prefix_feature_alignment_loss(
        self,
        review_prefixes,
        review_prefix_mask,
        input_ids,
        feature_position_weights,
    ):
        """让聚合 prefix 靠近当前真实 feature 的 LLM 词向量语义。"""
        if (
            review_prefixes is None
            or review_prefix_mask is None
            or feature_position_weights is None
        ):
            device = input_ids.device if input_ids is not None else self._llm_input_device()
            return torch.zeros((), dtype=torch.float32, device=device)

        prefix_mask = review_prefix_mask.to(review_prefixes.device).float()
        prefix_mean = (
            review_prefixes.float() * prefix_mask.unsqueeze(-1)
        ).sum(dim=1) / prefix_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        embedding_layer = self._embed_tokens()
        embedding_device = self._llm_input_device()
        target_embeddings = embedding_layer(input_ids.to(embedding_device)).detach().float()
        feature_mask = (
            feature_position_weights.to(embedding_device) >= 2.0
        ) & (input_ids.to(embedding_device) != self.pad_token_id)
        feature_mean = (
            target_embeddings * feature_mask.unsqueeze(-1).float()
        ).sum(dim=1) / feature_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        prefix_mean = prefix_mean.to(embedding_device)
        valid = (prefix_mask.sum(dim=1) > 0).to(embedding_device) & feature_mask.any(dim=1)
        if not valid.any():
            return target_embeddings.new_tensor(0.0)
        similarity = F.cosine_similarity(prefix_mean[valid], feature_mean[valid], dim=-1)
        return (1.0 - similarity).mean()

    def _feature_learning_loss(
        self,
        nll,
        valid_mask,
        feature_position_mask,
        feature_position_weights=None,
    ):
        if feature_position_mask is None or feature_position_mask.numel() == 0:
            return nll.new_tensor(0.0)
        if feature_position_mask.shape != nll.shape:
            raise ValueError(
                "feature_position_mask must match nll shape, "
                f"got {tuple(feature_position_mask.shape)} vs {tuple(nll.shape)}"
            )

        mask = feature_position_mask.to(device=nll.device).bool() & (valid_mask.to(nll.device) > 0)
        if feature_position_weights is None:
            weights = mask.float()
        else:
            if feature_position_weights.shape != nll.shape:
                raise ValueError(
                    "feature_position_weights must match nll shape, "
                    f"got {tuple(feature_position_weights.shape)} vs {tuple(nll.shape)}"
                )
            weights = feature_position_weights.to(device=nll.device, dtype=nll.dtype) * mask.to(dtype=nll.dtype)

        denom = weights.sum()
        if denom <= 0:
            return nll.new_tensor(0.0)
        return (nll * weights).sum() / denom

    def train_step(
        self,
        input_ids,
        profile_ids=None,
        profile_mask=None,
        target_item_ids=None,
        target_item_mask=None,
        generation_prompt_ids=None,
        generation_prompt_mask=None,
        evidence_token_ids=None,
        evidence_token_mask=None,
        feature_position_mask=None,
        feature_position_weights=None,
        tail_position_weights=None,
        user_review_embeddings=None,
        user_review_mask=None,
        item_review_embeddings=None,
        item_review_mask=None,
        item_review_query=None,
        user_review_query=None,
    ):
        review_prefixes, review_prefix_mask = self.build_review_prefixes(
            user_review_embeddings=user_review_embeddings,
            user_review_mask=user_review_mask,
            item_review_embeddings=item_review_embeddings,
            item_review_mask=item_review_mask,
            item_review_query=item_review_query,
            user_review_query=user_review_query,
        )
        inputs_embeds = self.get_embedding(
            input_ids=input_ids,
            profile_ids=profile_ids,
            review_prefixes=review_prefixes,
            target_item_ids=target_item_ids,
            generation_prompt_ids=generation_prompt_ids,
        )
        batch_size = input_ids.shape[0]
        device = input_ids.device
        attention_mask = self._attention_mask(
            input_ids=input_ids,
            profile_mask=profile_mask,
            review_prefix_mask=review_prefix_mask,
            target_item_mask=target_item_mask,
            generation_prompt_mask=generation_prompt_mask,
            batch_size=batch_size,
            device=device,
        )
        logits_to_keep = input_ids.shape[1] + 1
        try:
            output = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=logits_to_keep,
                return_dict=True,
            )
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            output = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        logits = output["logits"]

        if logits.shape[1] == logits_to_keep:
            gen_logits = logits[:, :-1, :]
        else:
            prompt_len = self._prompt_length(
                profile_ids,
                review_prefixes,
                target_item_ids,
                generation_prompt_ids,
            )
            gen_logits = logits[:, prompt_len - 1:-1, :]

        targets = input_ids
        valid_mask = (targets != self.pad_token_id).long()
        adjusted_logits = self._apply_evidence_bonus(
            gen_logits.clone(),
            evidence_token_ids,
            evidence_token_mask,
        )
        log_probs = torch.log_softmax(adjusted_logits.float(), dim=-1)
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        if tail_position_weights is None:
            sft_weights = valid_mask.to(dtype=nll.dtype)
        else:
            if tail_position_weights.shape != nll.shape:
                raise ValueError(
                    "tail_position_weights must match NLL shape, "
                    f"got {tuple(tail_position_weights.shape)} vs {tuple(nll.shape)}"
                )
            sft_weights = (
                tail_position_weights.to(device=nll.device, dtype=nll.dtype)
                * valid_mask.to(dtype=nll.dtype)
            )
        nll_loss = (nll * sft_weights).sum() / sft_weights.sum().clamp_min(1.0)

        if self.lambda_feat > 0:
            feat_loss = self._feature_learning_loss(
                nll,
                valid_mask,
                feature_position_mask,
                feature_position_weights,
            )
        else:
            feat_loss = gen_logits.new_tensor(0.0)

        if self.lambda_prefix_feature > 0:
            prefix_feature_loss = self._prefix_feature_alignment_loss(
                review_prefixes,
                review_prefix_mask,
                input_ids,
                feature_position_weights,
            )
        else:
            prefix_feature_loss = gen_logits.new_tensor(0.0)

        total = (
            nll_loss
            + self.lambda_feat * feat_loss
            + self.lambda_prefix_feature * prefix_feature_loss
        )
        return total, nll_loss.detach(), feat_loss.detach(), prefix_feature_loss.detach()

    def forward(
        self,
        input_ids=None,
        profile_ids=None,
        profile_mask=None,
        target_item_ids=None,
        target_item_mask=None,
        generation_prompt_ids=None,
        generation_prompt_mask=None,
        user_review_embeddings=None,
        user_review_mask=None,
        item_review_embeddings=None,
        item_review_mask=None,
        item_review_query=None,
        user_review_query=None,
        attention_mask=None,
        kv_cache=None,
    ):
        if kv_cache is None:
            review_prefixes, review_prefix_mask = self.build_review_prefixes(
                user_review_embeddings=user_review_embeddings,
                user_review_mask=user_review_mask,
                item_review_embeddings=item_review_embeddings,
                item_review_mask=item_review_mask,
                item_review_query=item_review_query,
                user_review_query=user_review_query,
            )
            inputs_embeds = self.get_embedding(
                input_ids=input_ids,
                profile_ids=profile_ids,
                review_prefixes=review_prefixes,
                target_item_ids=target_item_ids,
                generation_prompt_ids=generation_prompt_ids,
            )
            attention_mask = self._attention_mask(
                input_ids=input_ids,
                profile_mask=profile_mask,
                review_prefix_mask=review_prefix_mask,
                target_item_mask=target_item_mask,
                generation_prompt_mask=generation_prompt_mask,
                batch_size=input_ids.shape[0] if input_ids is not None else profile_ids.shape[0],
                device=inputs_embeds.device,
            )
            output = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
            )
        else:
            if attention_mask is not None and input_ids is not None and input_ids.shape[1] > 0:
                new_mask = torch.ones(
                    attention_mask.shape[0],
                    input_ids.shape[1],
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, new_mask], dim=1)
            output = self.model(
                input_ids=input_ids,
                past_key_values=kv_cache,
                attention_mask=attention_mask,
                use_cache=True,
            )
        logits = output["logits"][:, -1, :]
        return logits, output["past_key_values"], attention_mask

    def _apply_evidence_bonus(self, logits, evidence_token_ids, evidence_token_mask):
        if self.evidence_bonus == 0 or evidence_token_ids is None or evidence_token_mask is None:
            return logits
        if logits.dim() not in (2, 3):
            raise ValueError(f"Expected logits with shape [B,V] or [B,T,V], got {tuple(logits.shape)}")
        for batch_idx in range(logits.shape[0]):
            ids = evidence_token_ids[batch_idx][evidence_token_mask[batch_idx]]
            for token_id in ids.tolist():
                token_id = int(token_id)
                if self._is_evidence_control_token(token_id):
                    if logits.dim() == 2:
                        logits[batch_idx, token_id] += self.evidence_bonus
                    else:
                        logits[batch_idx, :, token_id] += self.evidence_bonus
        return logits

    def _repetition_run_length(self, generated_ids):
        if generated_ids is None or generated_ids.shape[1] == 0:
            batch = generated_ids.shape[0] if generated_ids is not None else 0
            return (
                torch.full((batch,), -1, dtype=torch.long, device=generated_ids.device),
                torch.zeros(batch, dtype=torch.long, device=generated_ids.device),
            )
        last_ids = generated_ids[:, -1]
        run_lens = torch.ones(generated_ids.shape[0], dtype=torch.long, device=generated_ids.device)
        for batch_idx in range(generated_ids.shape[0]):
            last_id = int(last_ids[batch_idx].item())
            run = 0
            for token_id in reversed(generated_ids[batch_idx].tolist()):
                if int(token_id) != last_id:
                    break
                run += 1
            run_lens[batch_idx] = run
        return last_ids, run_lens

    def _apply_repetition_controls(self, logits, generated_ids=None, last_ids=None, run_lens=None):
        max_repeat = self.max_consecutive_token_repeat
        if max_repeat <= 0:
            return logits
        floor = -1e4
        if generated_ids is not None and generated_ids.shape[1] > 0:
            if last_ids is None or run_lens is None:
                last_ids, run_lens = self._repetition_run_length(generated_ids)
            for batch_idx in range(logits.shape[0]):
                last_id = int(last_ids[batch_idx].item())
                if int(run_lens[batch_idx].item()) >= max_repeat and last_id >= 0:
                    logits[batch_idx, last_id] = floor
        return logits

    def _apply_generation_controls(
        self,
        logits,
        evidence_token_ids=None,
        evidence_token_mask=None,
        generated_ids=None,
        last_ids=None,
        run_lens=None,
    ):
        logits = self._apply_evidence_bonus(logits, evidence_token_ids, evidence_token_mask)
        logits = self._apply_repetition_controls(
            logits,
            generated_ids=generated_ids,
            last_ids=last_ids,
            run_lens=run_lens,
        )
        return logits

    @torch.no_grad()
    def greedy_generate(
        self,
        profile_ids,
        profile_mask,
        target_item_ids,
        target_item_mask,
        generation_prompt_ids,
        generation_prompt_mask,
        word,
        device,
        evidence_token_ids=None,
        evidence_token_mask=None,
        user_review_embeddings=None,
        user_review_mask=None,
        item_review_embeddings=None,
        item_review_mask=None,
        item_review_query=None,
        user_review_query=None,
    ):
        text = torch.tensor([[]], dtype=torch.long, device=device)
        last_words = torch.tensor([[]], dtype=torch.long, device=device)
        kv_cache = None
        attention_mask = None
        for _ in range(word):
            logits, kv_cache, attention_mask = self.forward(
                input_ids=last_words,
                profile_ids=profile_ids if kv_cache is None else None,
                profile_mask=profile_mask if kv_cache is None else None,
                target_item_ids=target_item_ids if kv_cache is None else None,
                target_item_mask=target_item_mask if kv_cache is None else None,
                generation_prompt_ids=generation_prompt_ids if kv_cache is None else None,
                generation_prompt_mask=generation_prompt_mask if kv_cache is None else None,
                user_review_embeddings=user_review_embeddings if kv_cache is None else None,
                user_review_mask=user_review_mask if kv_cache is None else None,
                item_review_embeddings=item_review_embeddings if kv_cache is None else None,
                item_review_mask=item_review_mask if kv_cache is None else None,
                item_review_query=item_review_query if kv_cache is None else None,
                user_review_query=user_review_query if kv_cache is None else None,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
            logits = self._apply_generation_controls(
                logits,
                evidence_token_ids=evidence_token_ids,
                evidence_token_mask=evidence_token_mask,
                generated_ids=text if text.shape[1] > 0 else None,
            )
            last_words = torch.argmax(logits, dim=1).unsqueeze(1)
            text = last_words if text.shape[1] == 0 else torch.cat([text, last_words], 1)
        return text.cpu().tolist()


def build_selector_outputs(
    selector: EvidenceSelector,
    graphs,
    node_token_emb: torch.Tensor,
    item_embs: torch.Tensor,
    top_m: int,
):
    evidence_lists = []
    utility_scores = []
    for batch_idx, graph in enumerate(graphs):
        if graph.num_nodes == 0:
            evidence_lists.append([])
            utility_scores.append(torch.empty((0,), device=item_embs.device))
            continue
        start = sum(g.num_nodes for g in graphs[:batch_idx])
        end = start + graph.num_nodes
        node_emb = node_token_emb[start:end]
        item_emb = item_embs[batch_idx]
        utility = selector.forward_single(graph, node_emb, item_emb)
        evidence_lists.append(selector.select_evidence(utility, graph, top_m=top_m).tolist())
        utility_scores.append(utility)

    evidence_token_ids, evidence_token_mask = pad_token_matrix(evidence_lists)
    return evidence_token_ids, evidence_token_mask, utility_scores
