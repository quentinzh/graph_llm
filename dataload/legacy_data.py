"""Standalone data/profile helpers copied from the GPT1 pipeline."""

from __future__ import annotations

import pickle
from pathlib import Path

from torch.utils.data import Dataset


DEFAULT_EMPTY_PROFILE = "User interests:\nNo profile available yet."
TOP_PRICE_MARKER = "Highest-price interacted items, treated as strongest preference signals:"
SPECIAL_TOKEN_IDS = (0, 1, 2)


def clean_ws(text):
    text = "" if text is None else str(text)
    return " ".join(text.split())


def tokenizer_special_ids(tokenizer):
    if tokenizer is None:
        return set(SPECIAL_TOKEN_IDS)
    ids = {
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "unk_token_id", None),
    }
    return {int(x) for x in ids if x is not None}


def tokenizer_pad_id(tokenizer):
    if tokenizer is None:
        return 0
    for attr in ["pad_token_id", "eos_token_id", "bos_token_id"]:
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return 0


def tokenizer_eos_id(tokenizer):
    if tokenizer is None:
        return 2
    value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else tokenizer_pad_id(tokenizer)
    if value is not None:
        return int(value)
    return tokenizer_pad_id(tokenizer)


def read_split_indices(data_dir, dataset_name, split_index):
    base = Path(data_dir) / dataset_name / str(split_index)
    result = {}
    for split in ["train", "validation", "test"]:
        with (base / f"{split}.index").open("r", encoding="utf-8") as f:
            result[split] = [int(x) for x in f.read().split() if x.strip()]
    return result


def dataset_split(dataset, split_index, args):
    indices = read_split_indices(args.data_dir, args.dataset_name, split_index)
    train_dataset = dataset.iloc[indices["train"]].reset_index(drop=True)
    valid_dataset = dataset.iloc[indices["validation"]].reset_index(drop=True)
    test_dataset = dataset.iloc[indices["test"]].reset_index(drop=True)
    return train_dataset, valid_dataset, test_dataset


def load_profile_cache(path, allow_missing=False):
    path = Path(path)
    if not path.exists():
        if allow_missing:
            return {}
        raise FileNotFoundError(
            f"Profile cache not found: {path}. Run generate_llama_user_profiles.py first."
        )
    with path.open("rb") as f:
        return pickle.load(f)


def profile_text_from_record(record):
    if record is None:
        return DEFAULT_EMPTY_PROFILE
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for key in ["profile_text", "final_profile", "llama_profile"]:
            value = record.get(key)
            if value:
                return str(value)
    return DEFAULT_EMPTY_PROFILE


def split_profile_sections(text):
    lines = str(text).splitlines()
    marker_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith(TOP_PRICE_MARKER):
            marker_idx = idx
            break
    if marker_idx is None:
        prefix = "\n".join(lines[:3]).strip()
        body = "\n".join(lines[3:]).strip()
        return prefix, body

    end = marker_idx + 1
    while end < len(lines):
        stripped = lines[end].strip()
        if not stripped:
            end += 1
            break
        if stripped[0].isdigit() or stripped.startswith("None"):
            end += 1
            continue
        break
    prefix = "\n".join(lines[:end]).strip()
    body = "\n".join(lines[end:]).strip()
    return prefix, body


def tokenize_profile_text(tokenizer, text, max_profile_tokens):
    special_ids = tokenizer_special_ids(tokenizer)
    ids = tokenizer(text)["input_ids"]
    if max_profile_tokens <= 0 or len(ids) <= max_profile_tokens:
        return ids

    prefix_text, body_text = split_profile_sections(text)
    prefix_ids = tokenizer(prefix_text)["input_ids"] if prefix_text else []
    if len(prefix_ids) >= max_profile_tokens:
        lines = str(text).splitlines()
        marker_idx = None
        for idx, line in enumerate(lines):
            if line.strip().startswith(TOP_PRICE_MARKER):
                marker_idx = idx
                break
        if marker_idx is None:
            return prefix_ids[:max_profile_tokens]

        top_end = marker_idx + 1
        while top_end < len(lines):
            stripped = lines[top_end].strip()
            if not stripped:
                break
            if stripped[0].isdigit() or stripped.startswith("None"):
                top_end += 1
                continue
            break
        header_text = "\n".join(lines[:marker_idx]).strip()
        top_text = "\n".join(lines[marker_idx:top_end]).strip()
        header_budget = max(1, max_profile_tokens // 2)
        top_budget = max_profile_tokens - header_budget
        header_ids = tokenizer(header_text)["input_ids"][:header_budget]
        top_ids = tokenizer(top_text)["input_ids"]
        if top_ids and top_ids[0] in special_ids:
            top_ids = top_ids[1:]
        return header_ids + top_ids[:top_budget]

    remaining = max_profile_tokens - len(prefix_ids)
    body_ids = tokenizer(body_text)["input_ids"] if body_text else []
    if body_ids and body_ids[0] in special_ids:
        body_ids = body_ids[1:]
    if len(body_ids) > remaining:
        head_len = max(0, remaining // 2)
        tail_len = remaining - head_len
        body_ids = body_ids[:head_len] + body_ids[-tail_len:]
    return prefix_ids + body_ids[:remaining]


def item_text_from_meta(raw_item, item_meta):
    meta = item_meta.get(str(raw_item), {}) if item_meta else {}
    title = clean_ws(meta.get("title")) or str(raw_item)
    description = clean_ws(meta.get("description")) or "Unknown"
    return title, description


def strip_leading_special(ids, tokenizer=None):
    if ids and ids[0] in tokenizer_special_ids(tokenizer):
        return ids[1:]
    return ids


def _extract_description_keywords(tokenizer, description: str) -> str:
    """Extract keywords from description text, deduplicate, join with commas."""
    if not description or description == "Unknown":
        return ""
    from graph_llm.aux.prompt_utils import useful_evidence_surface
    ids = tokenizer(description, add_special_tokens=False)["input_ids"]
    keywords = []
    seen = set()
    for tid in ids:
        surface = tokenizer.decode([tid], skip_special_tokens=True).strip()
        filtered = useful_evidence_surface(surface)
        if filtered and filtered.lower() not in seen:
            seen.add(filtered.lower())
            keywords.append(filtered)
    return ", ".join(keywords) if keywords else ""


def tokenize_target_item_text(tokenizer, raw_item, item_meta,
                               max_target_item_tokens,
                               description_mode="keywords"):
    if tokenizer is None:
        return []
    if max_target_item_tokens <= 0:
        return []

    title, description = item_text_from_meta(raw_item, item_meta)

    # ---- mode "none": title only ----
    if description_mode == "none":
        text = f"Current item information:\nTitle: {title}\n\n"
        ids = tokenizer(text)["input_ids"]
        if len(ids) > max_target_item_tokens:
            ids = ids[:max_target_item_tokens]
        return ids

    # ---- mode "keywords": title + description keywords ----
    if description_mode == "keywords":
        desc_text = _extract_description_keywords(tokenizer, description)
        if not desc_text:
            text = f"Current item information:\nTitle: {title}\n\n"
            ids = tokenizer(text)["input_ids"]
            return ids[:max_target_item_tokens] if len(ids) > max_target_item_tokens else ids
        prefix = f"Current item information:\nTitle: {title}\nDescription: "
        suffix = "\n\n"
        prefix_ids = tokenizer(prefix)["input_ids"]
        suffix_ids = strip_leading_special(tokenizer(suffix)["input_ids"], tokenizer)
        desc_ids = strip_leading_special(tokenizer(desc_text)["input_ids"], tokenizer)
        remaining = max_target_item_tokens - len(prefix_ids) - len(suffix_ids)
        if remaining <= 0:
            return (prefix_ids + suffix_ids)[:max_target_item_tokens]
        return prefix_ids + desc_ids[:remaining] + suffix_ids

    # ---- mode "full": title + full description (original logic) ----
    prefix = f"Current item information:\nTitle: {title}\nDescription: "
    suffix = "\n\n"
    prefix_ids = tokenizer(prefix)["input_ids"]
    suffix_ids = strip_leading_special(tokenizer(suffix)["input_ids"], tokenizer)

    if len(prefix_ids) + len(suffix_ids) >= max_target_item_tokens:
        header_ids = tokenizer("Current item information:\nTitle: ")["input_ids"]
        title_ids = strip_leading_special(tokenizer(title)["input_ids"], tokenizer)
        desc_label_ids = strip_leading_special(tokenizer("\nDescription: ")["input_ids"], tokenizer)
        budget = max_target_item_tokens - len(header_ids) - len(desc_label_ids) - len(suffix_ids)
        if budget <= 0:
            return (header_ids + title_ids + desc_label_ids + suffix_ids)[:max_target_item_tokens]
        return (header_ids + title_ids[:budget] + desc_label_ids + suffix_ids)[:max_target_item_tokens]

    description_ids = strip_leading_special(tokenizer(description)["input_ids"], tokenizer)
    remaining = max_target_item_tokens - len(prefix_ids) - len(suffix_ids)
    return prefix_ids + description_ids[:remaining] + suffix_ids


def missing_profile_users(dataframes, profile_records):
    if not isinstance(dataframes, (list, tuple)):
        dataframes = [dataframes]
    available = {str(key) for key in profile_records.keys()}
    required = set()
    for dataframe in dataframes:
        if "raw_user" in dataframe:
            required.update(str(x) for x in dataframe["raw_user"].tolist())
        else:
            required.update(str(x) for x in dataframe["user"].tolist())
    return sorted(required - available)


def assert_profile_coverage(name, dataframes, profile_records, allow_missing=False):
    if allow_missing:
        return
    missing = missing_profile_users(dataframes, profile_records)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"Profile cache for {name} misses {len(missing)} users. "
            f"Examples: {preview}. Regenerate the cache without --limit-users, "
            "or pass --allow_missing_profiles to use empty fallback profiles."
        )


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        self.feature_set = set(dataframe["keyword_words"])
        self.features = dataframe["keyword_words"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]
