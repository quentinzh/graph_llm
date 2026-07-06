"""Graph-guided explanation model with selector UL loss (no evidence text in prompt)."""

from __future__ import annotations

import torch
import torch.nn as nn

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


class GraphEvidenceCIER(nn.Module):
    """Profile-aware Qwen explainer with graph selector UL (no evidence prompt text)."""

    def __init__(
        self,
        tokenizer,
        vocab_size,
        evidence_selector: EvidenceSelector,
        lambda_ul=0.1,
        lambda_feat=0.0001,
        evidence_bonus=0.1,
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
        self.lambda_ul = float(lambda_ul)
        self.lambda_feat = float(lambda_feat)
        self.evidence_bonus = float(evidence_bonus)
        self.max_consecutive_token_repeat = int(max_consecutive_token_repeat)
        self.pad_token_id = int(pad_token_id)
        self.eos_token_ids = tuple(int(x) for x in (eos_token_ids or ()))
        self.special_token_ids = tuple(int(x) for x in special_token_ids)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

    def _graph_unlikelihood_loss(
        self,
        gen_logits,
        targets,
        valid_mask,
        neg_token_ids,
        neg_token_mask,
        evidence_token_ids,
        evidence_token_mask,
        neg_token_weights,
    ):
        if neg_token_ids is None or neg_token_ids.numel() == 0:
            return gen_logits.new_tensor(0.0)

        probs = torch.softmax(gen_logits, dim=-1)
        ignored = self._ignored_token_ids()
        total = gen_logits.new_tensor(0.0)
        count = 0

        batch_size, seq_len, _ = gen_logits.shape
        for b in range(batch_size):
            neg_ids = neg_token_ids[b][neg_token_mask[b]]
            if neg_ids.numel() == 0:
                continue
            evidence_ids = set(
                evidence_token_ids[b][evidence_token_mask[b]].detach().cpu().tolist()
            )
            weights = neg_token_weights[b][neg_token_mask[b]].float()
            for t in range(seq_len):
                if valid_mask[b, t] <= 0:
                    continue
                target_id = int(targets[b, t].item())
                step_probs = probs[b, t]
                for j, neg_id in enumerate(neg_ids.tolist()):
                    neg_id = int(neg_id)
                    if (
                        neg_id < 0
                        or neg_id == target_id
                        or neg_id in evidence_ids
                        or neg_id in ignored
                    ):
                        continue
                    p = step_probs[neg_id]
                    w = weights[j] if j < weights.numel() else 1.0
                    total = total + w * (-torch.log(torch.clamp(1.0 - p, min=1e-8)))
                    count += 1

        if count == 0:
            return gen_logits.new_tensor(0.0)
        return total / count

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
        neg_token_ids=None,
        neg_token_mask=None,
        evidence_token_ids=None,
        evidence_token_mask=None,
        neg_token_weights=None,
        feature_position_mask=None,
        feature_position_weights=None,
        apply_unlikelihood=True,
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
        try:
            output = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=logits_to_keep,
            )
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            output = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
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
        nll_loss = (nll * valid_mask).sum() / valid_mask.sum().clamp_min(1)

        if apply_unlikelihood:
            if self.lambda_ul > 0:
                ul_loss = self._graph_unlikelihood_loss(
                    adjusted_logits,
                    targets,
                    valid_mask,
                    neg_token_ids,
                    neg_token_mask,
                    evidence_token_ids,
                    evidence_token_mask,
                    neg_token_weights,
                )
            else:
                ul_loss = gen_logits.new_tensor(0.0)

            if self.lambda_feat > 0:
                feat_loss = self._feature_learning_loss(
                    nll,
                    valid_mask,
                    feature_position_mask,
                    feature_position_weights,
                )
            else:
                feat_loss = gen_logits.new_tensor(0.0)
        else:
            ul_loss = gen_logits.new_tensor(0.0)
            feat_loss = gen_logits.new_tensor(0.0)

        total = nll_loss + self.lambda_ul * ul_loss + self.lambda_feat * feat_loss
        return total, nll_loss.detach(), ul_loss.detach(), feat_loss.detach()

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
    ul_candidate_k: int,
    protected_token_ids: set[int],
    num_reviews_per_graph: list[float] | None = None,
):
    evidence_lists = []
    neg_lists = []
    weight_lists = []
    for batch_idx, graph in enumerate(graphs):
        if graph.num_nodes == 0:
            evidence_lists.append([])
            neg_lists.append([])
            weight_lists.append([])
            continue
        start = sum(g.num_nodes for g in graphs[:batch_idx])
        end = start + graph.num_nodes
        node_emb = node_token_emb[start:end]
        item_emb = item_embs[batch_idx]
        num_reviews = 1.0
        if num_reviews_per_graph is not None:
            num_reviews = float(num_reviews_per_graph[batch_idx])
        utility = selector.forward_single(graph, node_emb, item_emb, num_reviews=num_reviews)
        selected = selector.select_evidence_and_negatives(
            utility,
            graph,
            top_m=top_m,
            ul_candidate_k=ul_candidate_k,
            protected_token_ids=protected_token_ids,
        )
        evidence_lists.append(selected["evidence_token_ids"])
        neg_lists.append(selected["neg_token_ids"])
        weight_lists.append(selected["neg_weights"])

    evidence_token_ids, evidence_token_mask = pad_token_matrix(evidence_lists)
    neg_token_ids, neg_token_mask = pad_token_matrix(neg_lists)
    neg_weights, neg_weight_mask = pad_token_matrix(weight_lists, pad_value=0.0, dtype=torch.float32)
    neg_weights = neg_weights * neg_weight_mask.float()
    return evidence_token_ids, evidence_token_mask, neg_token_ids, neg_token_mask, neg_weights
