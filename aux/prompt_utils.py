"""Prompt text helpers for evidence and title-aware generation."""

from __future__ import annotations

import torch


EVIDENCE_PREFIX = "Relevant keywords: "
EVIDENCE_EMPTY = "Relevant keywords: none\n"
EVIDENCE_STOPWORDS = {
    "", ".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}",
    "'", "\"", "'s", "'m", "'ve", "n't", "'re", "'d", "'ll",
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "he", "her", "his", "i", "in", "is", "it", "its", "me", "my", "of", "on",
    "or", "our", "she", "that", "the", "their", "them", "there", "they",
    "this", "to", "was", "we", "were", "with", "you", "your",
    "user", "profile", "current", "item", "information", "title",
    "description", "explanation", "useful", "token", "evidence", "none",
}


def clean_ws(text) -> str:
    text = "" if text is None else str(text)
    return " ".join(text.split())


def useful_evidence_surface(surface: str) -> str:
    surface = clean_ws(surface).strip()
    normalized = surface.lower().strip(" \t\r\n.,!?;:'\"()[]{}")
    if not normalized or normalized in EVIDENCE_STOPWORDS:
        return ""
    if normalized.isdigit() or len(normalized) <= 1:
        return ""
    if not any(ch.isalpha() for ch in normalized):
        return ""
    return surface


def item_meta_from_row(raw_item, item_meta: dict | None) -> tuple[str, str, str]:
    """Return title, description, and selector-facing item text."""
    meta = item_meta.get(str(raw_item), {}) if item_meta else {}
    title = clean_ws(meta.get("title")) or str(raw_item)
    description = clean_ws(meta.get("description")) or "Unknown"
    item_text = f"Title: {title}\nDescription: {description}"
    return title, description, item_text


def build_evidence_prompt_text(token_ids, tokenizer) -> str:
    ids = [int(t) for t in token_ids if int(t) >= 0]
    if not ids:
        return EVIDENCE_EMPTY
    surfaces = []
    for token_id in ids:
        surface = useful_evidence_surface(tokenizer.decode([token_id], skip_special_tokens=True))
        if surface:
            surfaces.append(surface)
    if not surfaces:
        return EVIDENCE_EMPTY
    return EVIDENCE_PREFIX + ", ".join(surfaces) + "\n"


def build_generation_prompt_text(title: str) -> str:
    title = clean_ws(title) or "this item"
    return f'The explanation of {title} is "'


def tokenize_text_list(
    tokenizer,
    texts: list[str],
    pad_token_id: int,
    max_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not texts:
        return (
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.long),
        )
    encoded = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if max_tokens > 0 and len(ids) > max_tokens:
            ids = ids[:max_tokens]
        encoded.append(ids)

    max_len = max(len(ids) for ids in encoded)
    batch = len(encoded)
    out = torch.full((batch, max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros((batch, max_len), dtype=torch.long)
    for idx, ids in enumerate(encoded):
        if not ids:
            continue
        out[idx, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[idx, :len(ids)] = 1
    return out, mask


def build_evidence_prompt_batch(
    evidence_token_ids: torch.Tensor,
    evidence_token_mask: torch.Tensor,
    tokenizer,
    pad_token_id: int,
    max_tokens: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    texts = []
    batch_size = evidence_token_ids.shape[0] if evidence_token_ids.numel() else 0
    for batch_idx in range(batch_size):
        if evidence_token_mask is not None and evidence_token_mask.numel() > 0:
            ids = evidence_token_ids[batch_idx][evidence_token_mask[batch_idx]].tolist()
        else:
            ids = []
        texts.append(build_evidence_prompt_text(ids, tokenizer))
    return tokenize_text_list(tokenizer, texts, pad_token_id, max_tokens=max_tokens)


def build_generation_prompt_batch(
    item_titles: list[str],
    tokenizer,
    pad_token_id: int,
    max_tokens: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    texts = [build_generation_prompt_text(title) for title in item_titles]
    return tokenize_text_list(tokenizer, texts, pad_token_id, max_tokens=max_tokens)
