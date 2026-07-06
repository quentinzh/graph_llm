#!/usr/bin/env python
"""使用 Qwen3-4B 从交互历史生成用户画像。

流程：
  1. 加载数据集的 reviews.pickle 与 item.json。
  2. 对每个用户，按评分分层抽样至多 cap 条交互记录。（cap默认50）
  3. 将交互渲染为 chat-template 提示词，让 Qwen3-4B 输出 5 行结构化画像
     （严格程度 / 常用名词 / 兴趣 / 评论风格）。
  4. 以 bf16 批量生成；按 token 预算组 batch，避免长 prompt 撑爆 KV cache；
     OOM 时自动减半 batch 重试。
  5. 去除 <think> 块，按 (fold, scope) pickle 保存，支持断点续跑。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

# 降低长时间、变长 batch 循环中的 CUDA 显存碎片。
# 必须在首次 CUDA 分配前设置；setdefault 会保留用户已有的环境变量覆盖。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_llm.config.args import (  # noqa: E402
    default_model_path,
    qwen3_4b_model_candidates,
    resolve_local_model_path,
    resolve_torch_dtype,
)
from graph_llm.config.datasets import resolve_dataset_paths  # noqa: E402
from graph_llm.dataload.legacy_data import (  # noqa: E402
    DEFAULT_EMPTY_PROFILE,
    read_split_indices,
)
from graph_llm.train.trainer import ensure_tokenizer_ready  # noqa: E402

# Qwen3 在关闭 thinking 时仍可能输出推理块；保存前需剥离。
THINKING_RE = re.compile(
    r"<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)
SYSTEM_PROMPT = (
    "You are a concise user-profile summarizer. Output only the 5 labeled lines."
)
# 模型必须遵循的固定 5 行格式；保持简短以限制输出 token 数。
PROFILE_INSTRUCTION = """Write a profile with EXACTLY these 5 lines:
User: <user id>
Strictness: <strict|lenient|moderate> — <one short clause from rating tendency & review tone>
Frequent nouns: <up to 8 nouns often used in the reviews, comma-separated>
Interests: <the types/categories of items the user prefers, e.g. genres/cuisines/cities, one short phrase>
Review style: <factual|emotional|mixed> — <one short clause>
Keep under 128 tokens. No extra text."""


def _leaf_category(meta: dict) -> str | None:
    """从物品元数据中提取最细粒度（叶子）类别。"""
    categories = meta.get("categories")
    if categories is None:
        return None
    if isinstance(categories, str):
        parts = [part.strip() for part in categories.split(",") if part.strip()]
        return parts[-1] if parts else None
    if isinstance(categories, list) and categories:
        leaf = categories[-1]
        if isinstance(leaf, list) and leaf:
            return str(leaf[-1]).strip() or None
        return str(leaf).strip() or None
    return None


def _profile_category(meta: dict) -> str | None:
    """将叶子类别映射为画像中使用的类别名（如 TV -> Movies）。"""
    category = _leaf_category(meta)
    if category in {"TV", "Movies & TV"}:
        return "Movies"
    return category


def _item_title(meta: dict, raw_item: str) -> str:
    """从元数据取物品标题，缺失时回退到 raw_item。"""
    for key in ("title", "name"):
        value = meta.get(key)
        if value:
            return str(value).strip()
    return str(raw_item)


def truncate_words(text: str, max_words: int = 40) -> str:
    """将文本截断至最多 max_words 个词（用于控制 prompt 的 token 预算）。"""
    words = str(text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def item_description(meta: dict, raw_item: str) -> str:
    """从元数据构建简短物品描述，若无则回退到标题/ID。"""
    desc = meta.get("description")
    if desc:
        return truncate_words(str(desc).strip())
    name = meta.get("name") or meta.get("title")
    city = meta.get("city")
    if name and city:
        return truncate_words(f"{name} in {city}")
    if name:
        return truncate_words(str(name))
    return truncate_words(_item_title(meta, raw_item))


def review_explanation(row: dict) -> str:
    """从 template 元组（第 3 个元素）提取评论说明文本。"""
    template = row.get("template")
    if isinstance(template, (list, tuple)) and len(template) >= 3:
        return truncate_words(str(template[2]))
    return ""


def stratified_sample_interactions(
    group: pd.DataFrame,
    *,
    cap: int,
    seed: int,
) -> pd.DataFrame:
    """在保持评分分布的前提下，抽样至多 cap 条交互记录。

    按各评分的占比分配名额，在可能时每个评分至少保留 1 条。
    随后用两个 while 循环将总数精确调整到 cap。
    """
    if len(group) <= cap:
        return group.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    ratings = group["rating"].astype(float)
    unique_ratings = sorted(ratings.unique())
    counts = {rating: int((ratings == rating).sum()) for rating in unique_ratings}
    total = len(group)

    # 初始按比例分配（每个评分桶至少 1 条）。
    alloc = {
        rating: max(1, int(round(cap * counts[rating] / total)))
        for rating in unique_ratings
    }
    # 削减超额分配的桶，直到总和等于 cap。
    while sum(alloc.values()) > cap:
        candidates = [r for r in unique_ratings if alloc[r] > 1]
        if not candidates:
            break
        drop_rating = max(candidates, key=lambda r: alloc[r])
        alloc[drop_rating] -= 1
    # 在仍有剩余行可抽时，补足分配不足的桶。
    while sum(alloc.values()) < cap:
        added = False
        for rating in unique_ratings:
            if alloc[rating] < counts[rating] and sum(alloc.values()) < cap:
                alloc[rating] += 1
                added = True
        if not added:
            break

    parts = []
    for rating in unique_ratings:
        sub = group[group["rating"].astype(float) == rating]
        n = min(alloc[rating], len(sub))
        if n <= 0:
            continue
        if len(sub) <= n:
            parts.append(sub)
        else:
            parts.append(sub.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))

    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > cap:
        sampled = sampled.sample(n=cap, random_state=seed).reset_index(drop=True)
    return sampled.reset_index(drop=True)


def format_interaction_line(idx: int, row: dict, item_meta: dict) -> str:
    """将一条交互渲染为紧凑的一行，供 LLM prompt 使用。

    描述与评论均按词数截断，以限制单用户的 token 总量。
    """
    raw_item = str(row["raw_item"])
    meta = item_meta.get(raw_item, {})
    category = _profile_category(meta) or _leaf_category(meta) or "Unknown"
    rating = row.get("rating")
    try:
        rating_text = f"{float(rating):g}/5"
    except (TypeError, ValueError):
        rating_text = str(rating)
    desc = item_description(meta, raw_item)
    review = review_explanation(row)
    return (
        f"[{idx}] item={raw_item} rating={rating_text} cat={category} "
        f"desc={desc} review={review}"
    )


def build_user_messages(
    raw_user: str,
    interactions: list[dict],
    item_meta: dict,
    *,
    total_interactions: int,
) -> list[dict]:
    """构建 chat 消息：system 角色 + 含交互历史的 user prompt。"""
    lines = [
        format_interaction_line(i + 1, row, item_meta)
        for i, row in enumerate(interactions)
    ]
    user_prompt = (
        f"Below are a user's N={len(interactions)} interactions "
        f"(sampled from {total_interactions} total):\n"
        + "\n".join(lines)
        + "\n\n"
        + PROFILE_INSTRUCTION
        + f"\nUse this exact user id in line 1: {raw_user}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def apply_chat_prompt(tokenizer, messages: list[dict]) -> str:
    """应用模型的 chat 模板，并追加生成提示后缀。"""
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        # Qwen3：关闭思维链，避免冗长的 <think> 块。
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        # 非 Qwen 的 tokenizer 不接受 enable_thinking 参数。
        return tokenizer.apply_chat_template(messages, **kwargs)


def clean_generated_text(text: str) -> str:
    """去除 Qwen3 思维链块及首尾空白。"""
    return THINKING_RE.sub("", text or "").strip()


def parse_device(device: str) -> torch.device:
    """解析设备字符串；多 GPU 时默认使用 cuda:1。"""
    text = str(device).strip()
    if not text or text.lower() in {"default", "auto"}:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            return torch.device("cuda:1")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    return torch.device(text)


def load_model_and_tokenizer(model_path: str, device: torch.device, torch_dtype: str):
    """加载 Qwen3-4B 与 tokenizer，左填充以支持批量生成。"""
    tokenizer = ensure_tokenizer_ready(
        AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
    )
    # 左填充使 decoder-only 批量推理时生成起始位置对齐。
    tokenizer.padding_side = "left"
    # 左截断保留长 prompt 末尾的画像指令部分。
    tokenizer.truncation_side = "left"
    dtype = resolve_torch_dtype(torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=True,
        trust_remote_code=True,
        device_map={"": str(device)},
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def _generate_batch_once(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    max_input_tokens: int,
) -> list[str]:
    """对已组好的 batch 执行一次 generate()（不处理 OOM）。"""
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    input_width = input_ids.shape[1]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    results = []
    for idx in range(len(prompts)):
        # 切掉左侧填充的 prompt 前缀，只解码新生成的 token。
        new_tokens = outputs[idx, input_width:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append(clean_generated_text(text))
    return results


@torch.inference_mode()
def generate_profiles_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    max_input_tokens: int = 28000,
) -> list[str]:
    """批量生成画像，OOM 时自动重试。

    CUDA OOM 时：清空缓存，多 prompt 则减半 batch，
    单 prompt 则减半 max_input_tokens（最后手段）。
    确保个别超长 prompt 不会导致整次运行失败。
    """
    try:
        # 正常路径：整批一次性送进 _generate_batch_once。
        return _generate_batch_once(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
        )
    except torch.cuda.OutOfMemoryError:
        # OOM 后先释放碎片化的显存缓存，再决定降级策略。
        torch.cuda.empty_cache()
        if len(prompts) <= 1:
            # 单条仍 OOM：只能缩短输入（左截断），下限 2048 token。
            smaller = max(2048, max_input_tokens // 2)
            if smaller >= max_input_tokens:
                raise  # 已无法继续缩小，重新抛出以暴露真实问题
            print(f"OOM on single prompt; retrying with max_input_tokens={smaller}")
            return generate_profiles_batch(
                model,
                tokenizer,
                prompts,
                max_new_tokens=max_new_tokens,
                max_input_tokens=smaller,
            )
        # 多条 OOM：把 batch 一分为二递归重试，直到能放下或降到单条。
        mid = len(prompts) // 2
        print(f"OOM on batch of {len(prompts)}; splitting into {mid} + {len(prompts) - mid}")
        left = generate_profiles_batch(
            model,
            tokenizer,
            prompts[:mid],
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
        )
        right = generate_profiles_batch(
            model,
            tokenizer,
            prompts[mid:],
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
        )
        # 保持与原始 prompts 相同的顺序。
        return left + right


def build_profile_record(
    raw_user: str,
    *,
    fold: str,
    scope: str,
    generated: str,
    total_interactions: int,
    num_sampled: int,
    model_path: str,
    cap: int,
    seed: int,
) -> dict:
    """将单个用户的生成画像打包为下游训练所需的 schema。"""
    # 模型返回空字符串时回退到占位画像。
    profile_body = generated if generated else DEFAULT_EMPTY_PROFILE
    return {
        "raw_user": str(raw_user),
        "fold": str(fold),
        "scope": scope,
        "profile_mode": "llm",
        "profile_text": profile_body,
        "llama_profile": profile_body,
        "num_interactions": int(total_interactions),
        "num_sampled": int(num_sampled),
        "config": {
            "profile_mode": "llm",
            "model": "qwen3-4b",
            "model_path": model_path,
            "cap": cap,
            "seed": seed,
            "scope": scope,
        },
    }


def save_profiles(path: Path, profiles: dict) -> None:
    """将完整 profiles 字典持久化到 pickle 文件（崩溃安全检查点）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(profiles, f)


def load_existing_profiles(path: Path) -> dict:
    """加载已生成的画像，用于断点续跑。"""
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        data = pickle.load(f)
    return data if isinstance(data, dict) else {}


def load_item_meta(data_dir: Path, dataset_name: str) -> dict:
    """加载 item.json，返回 raw_item -> 元数据 的字典。"""
    item_path = data_dir / dataset_name / "item.json"
    if not item_path.is_file():
        print(f"WARNING: no item.json at {item_path}; profiles will use sparse metadata.")
        return {}
    rows = json.load(item_path.open("r", encoding="utf-8"))
    return {str(row.get("item")): row for row in rows if row.get("item") is not None}


def scope_row_indices(split_indices: dict, scope: str) -> list[int]:
    """将 scope 名称映射为需要纳入的评论行索引。"""
    if scope == "train":
        return list(split_indices["train"])
    if scope == "train_valid":
        return list(split_indices["train"]) + list(split_indices["validation"])
    raise ValueError(f"Unsupported scope: {scope}")


def plan_batches(
    token_lengths: list[int],
    *,
    batch_size: int,
    max_batch_tokens: int,
) -> list[list[int]]:
    """按数量与 max_len*count（KV cache 代理）双重上限将索引分组为 batch。

    先按 token 长度排序，短 prompt 可组成大 batch（更快），
    长 prompt 自动组成小 batch（更安全）。
    """
    order = sorted(range(len(token_lengths)), key=lambda i: token_lengths[i])
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for idx in order:
        length = token_lengths[idx]
        new_max = max(current_max, length)
        new_count = len(current) + 1
        if current and (
            new_count > batch_size or new_max * new_count > max_batch_tokens
        ):
            batches.append(current)
            current = [idx]
            current_max = length
        else:
            current.append(idx)
            current_max = new_max
    if current:
        batches.append(current)
    return batches


def generate_llm_profiles(
    *,
    data_dir: Path,
    dataset_name: str,
    profile_dir: Path,
    fold: str,
    scopes: list[str],  # train, train_valid，字符串列表，用来指定「要为哪些数据划分（split）里的用户生成画像」
    model_path: str,
    device: torch.device,
    cap: int = 50,
    seed: int = 42,
    batch_size: int = 16,
    max_new_tokens: int = 128,
    max_batch_tokens: int = 24576,
    max_input_tokens: int = 28000,
    torch_dtype: str = "bfloat16",
    checkpoint_every: int = 200,  # 保留以兼容 CLI；实际每个 batch 后都会保存
    max_users: int | None = None,
    overwrite: bool = False,
):
    """为指定 scope 内的所有用户生成 LLM 画像。"""
    reviews = pd.DataFrame(pd.read_pickle(data_dir / dataset_name / "reviews.pickle"))
    reviews["raw_user"] = reviews["user"].astype(str)
    reviews["raw_item"] = reviews["item"].astype(str)
    item_meta = load_item_meta(data_dir, dataset_name)
    split_indices = read_split_indices(data_dir, dataset_name, fold)

    # 解析本地 Qwen3-4B 权重路径：优先用 --model_path，否则在候选目录中查找。
    resolved_model_path = resolve_local_model_path(
        model_path,
        candidates=qwen3_4b_model_candidates(),
    )
    # 加载模型与 tokenizer 到指定 GPU（默认 cuda:1），bf16 推理，左 padding 便于批量生成。
    model, tokenizer = load_model_and_tokenizer(
        resolved_model_path,
        device,
        torch_dtype,
    )

    out_dir = profile_dir / dataset_name.strip("/")
    out_dir.mkdir(parents=True, exist_ok=True)

    for scope in scopes:
        # scope 决定用哪些 split 行：train 仅训练集，train_valid = train + validation。
        row_indices = scope_row_indices(split_indices, scope)
        scoped = reviews.iloc[row_indices].reset_index(drop=True)
        # 每个 scope 单独落盘，便于下游按需加载（如只要 train 画像）。
        out_path = out_dir / f"fold_{fold}_{scope}.pkl"
        # 断点续跑：--overwrite 时清空重来，否则从已有 pickle 恢复已生成用户。
        profiles = {} if overwrite else load_existing_profiles(out_path)

        # 按用户分组；每个 group 是该用户在本 scope 内的全部交互记录。
        user_groups = list(scoped.groupby("raw_user", sort=False))
        if max_users is not None:
            user_groups = user_groups[:max_users]  # 调试用：只处理前 N 个用户

        # 过滤掉 profiles 里已有的用户，只对待生成用户跑 LLM。
        pending = [
            (raw_user, group)
            for raw_user, group in user_groups
            if str(raw_user) not in profiles
        ]
        print(
            f"Scope={scope}: {len(user_groups)} users total, "
            f"{len(profiles)} cached, {len(pending)} pending -> {out_path}"
        )

        # 为所有待处理用户准备 prompt 与 token 长度（CPU 侧，较快）。
        pending_items: list[tuple[str, int, int, str, int]] = []
        for raw_user, group in tqdm(pending, desc=f"prepare:{scope}"):
            total_count = len(group)
            # 按评分分层抽样至多 cap 条交互记录。
            sampled = stratified_sample_interactions(group, cap=cap, seed=seed)
            interactions = sampled.to_dict("records")
            # 将交互渲染为 chat-template 提示词，让 Qwen3-4B 输出 5 行结构化画像
            messages = build_user_messages(
                str(raw_user),
                interactions,
                item_meta,
                total_interactions=total_count,
            )
            # 应用模型的 chat 模板，并追加生成提示后缀
            prompt = apply_chat_prompt(tokenizer, messages)
            # 仅用于分桶的长度估计；add_special_tokens=False 因为
            # apply_chat_prompt 已嵌入 chat 模板的特殊 token。
            token_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            pending_items.append(
                (str(raw_user), total_count, len(interactions), prompt, token_len)
            )

        # 按 token 长度分组为 batch，避免长 prompt 撑爆 KV cache。
        batches = plan_batches(
            [item[4] for item in pending_items],
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
        )

        for batch_idxs in tqdm(batches, desc=f"profiles:{scope}"):
            # 从 pending_items 中提取 batch 内每个 prompt 与用户元信息。
            batch_prompts = [pending_items[i][3] for i in batch_idxs]
            # 提取 batch 内每个用户的原始用户 ID、总交互数、抽样数。
            batch_meta = [
                (pending_items[i][0], pending_items[i][1], pending_items[i][2])
                for i in batch_idxs
            ]
            # 批量生成画像，OOM 时自动重试。
            generated_list = generate_profiles_batch(
                model,
                tokenizer,
                batch_prompts,
                max_new_tokens=max_new_tokens,
                max_input_tokens=max_input_tokens,
            )
            # 将生成结果打包为 profile 记录，并保存到 profiles 字典。
            for (raw_user, total_count, sampled_count), generated in zip(
                batch_meta,
                generated_list,
            ):
                # 构建 profile 记录，包含用户 ID、fold、scope、生成画像等。
                profiles[str(raw_user)] = build_profile_record(
                    str(raw_user),
                    fold=fold,
                    scope=scope,
                    generated=generated,
                    total_interactions=total_count,
                    num_sampled=sampled_count,
                    model_path=resolved_model_path,
                    cap=cap,
                    seed=seed,
                )
            # 崩溃安全：每个 batch 后持久化，中途失败可续跑。
            save_profiles(out_path, profiles)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Wrote {len(profiles)} profiles to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="使用 Qwen3-4B 生成 LLM 用户画像")
    parser.add_argument("--dataset_name", "--dataset", dest="dataset_name", required=True)
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"), type=str)
    parser.add_argument(
        "--profile_dir",
        default=str(PACKAGE_ROOT / "data" / "profiles"),
        type=str,
    )
    parser.add_argument("--fold", default="1", type=str)
    parser.add_argument("--scopes", default="train,train_valid", type=str)
    parser.add_argument("--model_path", default=default_model_path(), type=str)
    parser.add_argument("--device", default="cuda:1", type=str)
    parser.add_argument("--cap", default=50, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_new_tokens", default=128, type=int)
    parser.add_argument(
        "--max_batch_tokens",
        default=24576,
        type=int,
        help="batch 内最长 prompt * batch_size 的上限；若仍 OOM 可适当调低。",
    )
    parser.add_argument(
        "--max_input_tokens",
        default=28000,
        type=int,
        help="传给 tokenizer 的单条 prompt 截断上限。",
    )
    parser.add_argument("--torch_dtype", default="bfloat16", type=str)
    parser.add_argument("--checkpoint_every", default=200, type=int)
    parser.add_argument("--max_users", default=None, type=int)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="忽略已有 profile pickle，重新生成全部用户。",
    )
    args = parser.parse_args()

    class _ResolveArgs:
        dataset_name = args.dataset_name
        data_dir = args.data_dir

    resolve_args = _ResolveArgs()
    resolve_dataset_paths(resolve_args)

    scopes = [scope.strip() for scope in args.scopes.split(",") if scope.strip()]
    generate_llm_profiles(
        data_dir=Path(resolve_args.data_dir),
        dataset_name=resolve_args.dataset_name,
        profile_dir=Path(args.profile_dir),
        fold=str(args.fold),
        scopes=scopes,
        model_path=args.model_path,
        device=parse_device(args.device),
        cap=args.cap,
        seed=args.seed,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_batch_tokens=args.max_batch_tokens,
        max_input_tokens=args.max_input_tokens,
        torch_dtype=args.torch_dtype,
        checkpoint_every=args.checkpoint_every,
        max_users=args.max_users,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
