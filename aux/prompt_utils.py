"""Prompt text helpers for title/user-aware generation."""

from __future__ import annotations

import torch


def clean_ws(text) -> str:
    text = "" if text is None else str(text)
    return " ".join(text.split())


def useful_evidence_surface(surface: str) -> str:
    surface = clean_ws(surface).strip()
    normalized = surface.lower().strip(" \t\r\n.,!?;:'\"()[]{}")
    stopwords = {
        "", ".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}",
        "'", "\"", "'s", "'m", "'ve", "n't", "'re", "'d", "'ll",
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "he", "her", "his", "i", "in", "is", "it", "its", "me", "my", "of", "on",
        "or", "our", "she", "that", "the", "their", "them", "there", "they",
        "this", "to", "was", "we", "were", "with", "you", "your",
        "user", "profile", "current", "item", "information", "title",
        "description", "explanation", "useful", "token", "evidence", "none",
    }
    if not normalized or normalized in stopwords:
        return ""
    if normalized.isdigit() or len(normalized) <= 1:
        return ""
    if not any(ch.isalpha() for ch in normalized):
        return ""
    return surface


def item_meta_from_row(raw_item, item_meta: dict | None) -> tuple[str, str, str]:
    """Return title, description, and selector-facing item text."""
    meta = item_meta.get(str(raw_item), {}) if item_meta else {}
    title = clean_ws(meta.get("title") or meta.get("name")) or str(raw_item)
    description = clean_ws(meta.get("description")) or "Unknown"
    item_text = f"Title: {title}\nDescription: {description}"
    return title, description, item_text


def build_generation_prompt_text(title: str, user_id: str) -> str:
    title = clean_ws(title) or "this item"
    user_id = clean_ws(user_id) or "this user"
    return f'The explanation of {title} for {user_id} is "'


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


def build_generation_prompt_batch(
    item_titles: list[str],
    user_ids: list[str],
    tokenizer,
    pad_token_id: int,
    max_tokens: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    texts = [
        build_generation_prompt_text(title, user_id)
        for title, user_id in zip(item_titles, user_ids)
    ]
    return tokenize_text_list(tokenizer, texts, pad_token_id, max_tokens=max_tokens)
