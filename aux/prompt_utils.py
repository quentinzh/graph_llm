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


def truncate_text_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    """按模型 tokenizer 截断 prototype，避免长证据抢占主输入上下文。"""
    text = clean_ws(text)
    if not text or max_tokens <= 0:
        return ""
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_tokens]
    return clean_ws(tokenizer.decode(token_ids, skip_special_tokens=True))


def build_generation_prompt_text(
    title: str,
    user_id: str,
    *,
    item_prototype: str = "",
    user_prototype: str = "",
) -> str:
    """构造生成段；profile 与 item 信息由调用方作为前置 embedding 传入。"""
    title = clean_ws(title) or "this item"
    user_id = clean_ws(user_id) or "this user"
    legacy_prompt = f'The explanation of {title} for {user_id} is "'
    # 未启用 prototype 时严格保留原 prompt，保证旧 checkpoint 的评测可复现。
    if not item_prototype and not user_prototype:
        return legacy_prompt
    parts = []
    if item_prototype:
        parts.append(f"Relevant review of this item:\n{item_prototype}")
    if user_prototype:
        parts.append(f"Similar review previously written by this user:\n{user_prototype}")
    parts.append(
        "Write a concise personalized explanation:\n"
        + legacy_prompt
    )
    return "\n\n".join(parts)


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
    item_prototypes: list[str] | None = None,
    user_prototypes: list[str] | None = None,
    prototype_max_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if item_prototypes is None:
        item_prototypes = [""] * len(item_titles)
    if user_prototypes is None:
        user_prototypes = [""] * len(item_titles)
    if not (len(item_titles) == len(user_ids) == len(item_prototypes) == len(user_prototypes)):
        raise ValueError("Prompt fields must have the same batch size")
    texts = [
        build_generation_prompt_text(
            title,
            user_id,
            item_prototype=truncate_text_to_tokens(tokenizer, item_prototype, prototype_max_tokens),
            user_prototype=truncate_text_to_tokens(tokenizer, user_prototype, prototype_max_tokens),
        )
        for title, user_id, item_prototype, user_prototype in zip(
            item_titles, user_ids, item_prototypes, user_prototypes
        )
    ]
    return tokenize_text_list(tokenizer, texts, pad_token_id, max_tokens=max_tokens)
