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


class FutureTokenAdapter(nn.Module):
    """用低秩残差映射把共享隐藏状态变换到指定未来步。"""

    def __init__(self, hidden_size: int, rank: int, eps: float = 1e-6):
        super().__init__()
        self.input_norm = nn.RMSNorm(hidden_size, eps=eps)
        self.down_proj = nn.Linear(hidden_size, rank, bias=False)
        self.up_proj = nn.Linear(rank, hidden_size, bias=False)
        self.output_norm = nn.RMSNorm(hidden_size, eps=eps)
        self.activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.input_norm.reset_parameters()
        self.output_norm.reset_parameters()
        nn.init.xavier_uniform_(self.down_proj.weight)
        # 从近似恒等映射开始，避免随机辅助头在训练初期强烈扰动共享主干。
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, hidden_states):
        residual = hidden_states
        update = self.up_proj(self.activation(self.down_proj(self.input_norm(hidden_states))))
        return self.output_norm(residual + update)


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
        future_steps=1,
        lambda_future=0.0,
        future_decay=0.5,
        future_head_rank=64,
        future_num_candidates=1024,
        future_random_negatives=256,
        future_tail_negatives=128,
        future_graph_candidates=128,
        future_temperature=1.0,
        future_tail_token_ids=None,
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
        self.future_steps = int(future_steps)
        self.lambda_future = float(lambda_future)
        self.future_decay = float(future_decay)
        self.future_head_rank = int(future_head_rank)
        self.future_num_candidates = int(future_num_candidates)
        self.future_random_negatives = int(future_random_negatives)
        self.future_tail_negatives = int(future_tail_negatives)
        self.future_graph_candidates = int(future_graph_candidates)
        self.future_temperature = float(future_temperature)
        self.max_consecutive_token_repeat = int(max_consecutive_token_repeat)
        self.pad_token_id = int(pad_token_id)
        self.eos_token_ids = tuple(int(x) for x in (eos_token_ids or ()))
        self.special_token_ids = tuple(int(x) for x in special_token_ids)

        if self.future_steps < 1:
            raise ValueError("future_steps must be at least 1")
        if self.future_steps > 1 and self.hidden_size is None:
            raise ValueError("hidden_size is required when future_steps > 1")
        if self.lambda_future < 0:
            raise ValueError("lambda_future must be non-negative")
        if not 0 < self.future_decay <= 1:
            raise ValueError("future_decay must be in (0, 1]")
        if self.future_head_rank < 1:
            raise ValueError("future_head_rank must be positive")
        if self.future_num_candidates < 1:
            raise ValueError("future_num_candidates must be positive")
        if min(
            self.future_random_negatives,
            self.future_tail_negatives,
            self.future_graph_candidates,
        ) < 0:
            raise ValueError("future candidate counts must be non-negative")
        if self.future_temperature <= 0:
            raise ValueError("future_temperature must be positive")

        self.future_adapters = nn.ModuleList()
        if self.future_steps > 1:
            self.future_adapters.extend(
                FutureTokenAdapter(self.hidden_size, self.future_head_rank)
                for _ in range(self.future_steps - 1)
            )

        tail_ids = sorted({
            int(token_id)
            for token_id in (future_tail_token_ids or ())
            if 0 <= int(token_id) < self.vocab_size
        })
        tail_tensor = torch.tensor(tail_ids, dtype=torch.long)
        tail_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        if tail_ids:
            tail_mask[tail_tensor] = True
        # 采样池属于训练期临时状态，不需要写入推理 checkpoint。
        self.register_buffer("future_tail_token_ids", tail_tensor, persistent=False)
        self.register_buffer("future_tail_token_mask", tail_mask, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for adapter in self.future_adapters:
            adapter.reset_parameters()

    def _future_loss_enabled(self):
        return self.future_steps > 1 and self.lambda_future > 0

    def _future_head_device(self):
        if not self.future_adapters:
            return self._llm_input_device()
        return next(self.future_adapters.parameters()).device

    def _output_embedding_weight(self):
        output_embeddings = self.model.get_output_embeddings()
        if output_embeddings is not None and hasattr(output_embeddings, "weight"):
            return output_embeddings.weight
        # 部分轻量测试模型没有显式 output head，此时退化为 tied input embedding。
        return self._embed_tokens().weight

    def _future_hidden_capture_module(self):
        """优先定位最终归一化层，避免返回所有 Transformer 层隐藏状态。"""
        if self.model is None:
            return None
        base_model = self.model
        if hasattr(base_model, "get_base_model"):
            base_model = base_model.get_base_model()
        if hasattr(base_model, "get_decoder"):
            decoder = base_model.get_decoder()
            norm = getattr(decoder, "norm", None)
            if norm is not None:
                return norm
        return None

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
        target_item_ids=None,
        generation_prompt_ids=None,
    ):
        embeddings = self._embed_tokens()
        llm_dtype = embeddings.weight.dtype
        llm_device = self._llm_input_device()
        parts = []
        if profile_ids is not None and profile_ids.numel() > 0:
            parts.append(embeddings(profile_ids.to(llm_device)).to(dtype=llm_dtype))
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
        target_item_mask=None,
        generation_prompt_mask=None,
        batch_size=None,
        device=None,
    ):
        masks = []
        if profile_mask is not None and profile_mask.numel() > 0:
            masks.append(profile_mask.to(device))
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
        target_item_ids=None,
        generation_prompt_ids=None,
    ):
        total = 0
        if profile_ids is not None:
            total += profile_ids.shape[1]
        if target_item_ids is not None:
            total += target_item_ids.shape[1]
        if generation_prompt_ids is not None:
            total += generation_prompt_ids.shape[1]
        return total

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

    @torch.no_grad()
    def _build_future_candidates(
        self,
        input_ids,
        evidence_token_ids=None,
        evidence_token_mask=None,
        device=None,
    ):
        """构造 batch 共享候选集及其采样概率修正项。

        所有 t+2/t+3 正样本都会被强制保留；随机、tail 和 graph token
        仅用于补充负样本。候选上限不足时宁可临时扩容，也不会丢失正样本。
        """
        device = device or input_ids.device
        input_ids = input_ids.to(device)
        positive_parts = []
        for offset in range(1, self.future_steps):
            if input_ids.shape[1] <= offset:
                continue
            targets = input_ids[:, offset:]
            positive_parts.append(targets[targets != self.pad_token_id])
        if not positive_parts:
            empty_ids = torch.empty(0, dtype=torch.long, device=device)
            empty_log_q = torch.empty(0, dtype=torch.float32, device=device)
            return empty_ids, empty_log_q
        positive_ids = torch.unique(torch.cat(positive_parts), sorted=True)

        extra_parts = []
        if (
            evidence_token_ids is not None
            and evidence_token_mask is not None
            and self.future_graph_candidates > 0
        ):
            graph_ids = evidence_token_ids.to(device)[evidence_token_mask.to(device).bool()]
            graph_ids = graph_ids[(graph_ids >= 0) & (graph_ids < self.vocab_size)]
            if graph_ids.numel() > 0:
                extra_parts.append(torch.unique(graph_ids)[:self.future_graph_candidates])

        if self.future_random_negatives > 0:
            # 多抽一倍后过滤特殊 token，以尽量得到足量且有效的随机负样本。
            draw_count = max(self.future_random_negatives * 2, self.future_random_negatives)
            random_ids = torch.randint(0, self.vocab_size, (draw_count,), device=device)
            ignored = torch.tensor(
                sorted(self._ignored_token_ids()),
                dtype=torch.long,
                device=device,
            )
            if ignored.numel() > 0:
                random_ids = random_ids[~torch.isin(random_ids, ignored)]
            random_ids = torch.unique(random_ids)[:self.future_random_negatives]
            if random_ids.numel() > 0:
                extra_parts.append(random_ids)

        tail_pool = self.future_tail_token_ids.to(device)
        if self.future_tail_negatives > 0 and tail_pool.numel() > 0:
            tail_indices = torch.randint(
                0,
                tail_pool.numel(),
                (self.future_tail_negatives,),
                device=device,
            )
            extra_parts.append(torch.unique(tail_pool[tail_indices]))

        if extra_parts:
            extra_ids = torch.unique(torch.cat(extra_parts), sorted=True)
            extra_ids = extra_ids[~torch.isin(extra_ids, positive_ids)]
            remaining = max(self.future_num_candidates - positive_ids.numel(), 0)
            extra_ids = extra_ids[:remaining]
            candidate_ids = torch.unique(
                torch.cat([positive_ids, extra_ids]),
                sorted=True,
            )
        else:
            candidate_ids = positive_ids

        # q(v) 使用 uniform 与 tail-uniform 的混合分布。统一对所有候选修正，
        # 使过采样的 tail token 不会仅因为出现频率高而成为过强负样本。
        random_count = float(self.future_random_negatives)
        tail_count = float(self.future_tail_negatives if tail_pool.numel() > 0 else 0)
        sample_count = random_count + tail_count
        if sample_count <= 0:
            uniform_mix, tail_mix = 1.0, 0.0
        else:
            # 保留至少 10% 的均匀参考概率，让任意强制正样本都有非零 q(v)。
            uniform_mix = max(random_count / sample_count, 0.1)
            tail_mix = 1.0 - uniform_mix
        q = torch.full(
            (candidate_ids.numel(),),
            uniform_mix / max(float(self.vocab_size), 1.0),
            dtype=torch.float32,
            device=device,
        )
        if tail_mix > 0:
            tail_mask = self.future_tail_token_mask.to(device).index_select(0, candidate_ids)
            q = q + tail_mask.to(q.dtype) * (tail_mix / float(tail_pool.numel()))
        return candidate_ids, torch.log(q.clamp_min(1e-12))

    def _response_hidden_states(
        self,
        output,
        logits,
        logits_to_keep,
        captured_final_hidden=None,
        profile_ids=None,
        target_item_ids=None,
        generation_prompt_ids=None,
    ):
        final_hidden = captured_final_hidden
        if final_hidden is None:
            hidden_states = output.get("hidden_states")
            if not hidden_states:
                raise RuntimeError(
                    "Sampled MTP requires the LLM to expose its final hidden state. "
                    "Neither a final-norm hook nor output_hidden_states is available."
                )
            final_hidden = hidden_states[-1]
        if logits.shape[1] == logits_to_keep:
            response_hidden = final_hidden[:, -logits_to_keep:-1, :]
        else:
            prompt_len = self._prompt_length(
                profile_ids,
                target_item_ids,
                generation_prompt_ids,
            )
            response_hidden = final_hidden[:, prompt_len - 1:-1, :]
        expected_length = logits_to_keep - 1
        if response_hidden.shape[1] != expected_length:
            raise ValueError(
                "MTP response hidden length mismatch: "
                f"expected {expected_length}, got {response_hidden.shape[1]}"
            )
        return response_hidden

    def _future_prediction_loss(
        self,
        response_hidden,
        input_ids,
        evidence_token_ids=None,
        evidence_token_mask=None,
        tail_position_weights=None,
    ):
        if not self._future_loss_enabled() or input_ids.shape[1] < 2:
            return response_hidden.new_tensor(0.0)

        future_device = self._future_head_device()
        candidate_ids, candidate_log_q = self._build_future_candidates(
            input_ids,
            evidence_token_ids=evidence_token_ids,
            evidence_token_mask=evidence_token_mask,
            device=future_device,
        )
        if candidate_ids.numel() == 0:
            return response_hidden.new_tensor(0.0)

        output_weight = self._output_embedding_weight()
        candidate_embeddings = output_weight.index_select(
            0,
            candidate_ids.to(output_weight.device),
        ).to(future_device)
        candidate_log_q = candidate_log_q.to(future_device)
        response_hidden = response_hidden.to(future_device)
        targets_all = input_ids.to(future_device)
        tail_weights_all = (
            tail_position_weights.to(future_device)
            if tail_position_weights is not None
            else None
        )

        weighted_losses = []
        horizon_weights = []
        for future_step in range(2, self.future_steps + 1):
            offset = future_step - 1
            if targets_all.shape[1] <= offset:
                continue
            targets = targets_all[:, offset:]
            valid_mask = targets != self.pad_token_id
            if not valid_mask.any():
                continue

            # 同一个 h_t 分别对齐 y_{t+2}、y_{t+3}，不读取中间未来 token。
            hidden = response_hidden[:, :targets.shape[1], :][valid_mask]
            target_ids = targets[valid_mask]
            future_repr = self.future_adapters[future_step - 2](hidden)
            logits = F.linear(future_repr, candidate_embeddings)
            logits = logits.float() / self.future_temperature
            logits = logits - candidate_log_q.float().unsqueeze(0)

            target_columns = torch.searchsorted(candidate_ids, target_ids)
            if not torch.equal(candidate_ids[target_columns], target_ids):
                raise RuntimeError("Future candidate set is missing a positive target token")
            token_loss = F.cross_entropy(logits, target_columns, reduction="none")
            if tail_weights_all is None:
                token_weights = torch.ones_like(token_loss)
            else:
                token_weights = tail_weights_all[:, offset:][valid_mask].to(token_loss.dtype)
            horizon_loss = (
                (token_loss * token_weights).sum()
                / token_weights.sum().clamp_min(1.0)
            )
            gamma = self.future_decay ** (future_step - 2)
            weighted_losses.append(gamma * horizon_loss)
            horizon_weights.append(gamma)

        if not weighted_losses:
            return response_hidden.new_tensor(0.0)
        return torch.stack(weighted_losses).sum() / sum(horizon_weights)

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
    ):
        inputs_embeds = self.get_embedding(
            input_ids=input_ids,
            profile_ids=profile_ids,
            target_item_ids=target_item_ids,
            generation_prompt_ids=generation_prompt_ids,
        )
        batch_size = input_ids.shape[0]
        device = input_ids.device
        attention_mask = self._attention_mask(
            input_ids=input_ids,
            profile_mask=profile_mask,
            target_item_mask=target_item_mask,
            generation_prompt_mask=generation_prompt_mask,
            batch_size=batch_size,
            device=device,
        )
        logits_to_keep = input_ids.shape[1] + 1
        need_future_hidden = self._future_loss_enabled()
        captured_hidden = {}
        capture_module = (
            self._future_hidden_capture_module()
            if need_future_hidden
            else None
        )
        capture_handle = None
        if capture_module is not None:
            def _capture_final_hidden(_module, _inputs, module_output):
                # Qwen3 最终 RMSNorm 输出为 Tensor；兼容少数返回 tuple 的实现。
                captured_hidden["value"] = (
                    module_output[0]
                    if isinstance(module_output, (tuple, list))
                    else module_output
                )

            capture_handle = capture_module.register_forward_hook(_capture_final_hidden)
        request_all_hidden_states = need_future_hidden and capture_module is None
        try:
            try:
                output = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=logits_to_keep,
                    output_hidden_states=request_all_hidden_states,
                    return_dict=True,
                )
            except TypeError as exc:
                if "logits_to_keep" not in str(exc):
                    raise
                output = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=request_all_hidden_states,
                    return_dict=True,
                )
        finally:
            if capture_handle is not None:
                capture_handle.remove()
        logits = output["logits"]

        if logits.shape[1] == logits_to_keep:
            gen_logits = logits[:, :-1, :]
        else:
            prompt_len = self._prompt_length(
                profile_ids,
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

        if need_future_hidden:
            response_hidden = self._response_hidden_states(
                output,
                logits,
                logits_to_keep,
                captured_final_hidden=captured_hidden.get("value"),
                profile_ids=profile_ids,
                target_item_ids=target_item_ids,
                generation_prompt_ids=generation_prompt_ids,
            )
            future_loss = self._future_prediction_loss(
                response_hidden,
                input_ids,
                evidence_token_ids=evidence_token_ids,
                evidence_token_mask=evidence_token_mask,
                tail_position_weights=tail_position_weights,
            )
        else:
            future_loss = gen_logits.new_tensor(0.0)

        total = (
            nll_loss
            + self.lambda_feat * feat_loss
            + self.lambda_future * future_loss
        )
        return total, nll_loss.detach(), feat_loss.detach(), future_loss.detach()

    def forward(
        self,
        input_ids=None,
        profile_ids=None,
        profile_mask=None,
        target_item_ids=None,
        target_item_mask=None,
        generation_prompt_ids=None,
        generation_prompt_mask=None,
        attention_mask=None,
        kv_cache=None,
    ):
        if kv_cache is None:
            inputs_embeds = self.get_embedding(
                input_ids=input_ids,
                profile_ids=profile_ids,
                target_item_ids=target_item_ids,
                generation_prompt_ids=generation_prompt_ids,
            )
            attention_mask = self._attention_mask(
                input_ids=input_ids,
                profile_mask=profile_mask,
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
