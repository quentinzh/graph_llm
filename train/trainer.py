#!/usr/bin/env python
"""Train profile-conditioned Qwen3-4B LoRA explainer."""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_llm.aux.prompt_utils import build_generation_prompt_batch
from graph_llm.config.args import _is_local_model_dir
from graph_llm.config import (
    dataset_cache_path,
    parse_csv,
    parse_special_token_ids,
    qwen3_4b_model_candidates,
    resolve_dataset_paths,
    resolve_local_model_path,
    resolve_torch_dtype,
    snapshot_training_args,
)
from graph_llm.dataload.cache import GraphCacheManager
from graph_llm.dataload.dataloader import (
    GraphCollater,
    GraphDataset,
    assert_profile_coverage,
    compute_profile_lengths,
    dataset_split,
    load_profile_cache,
    read_split_indices,
    tokenizer_eos_id,
    tokenizer_pad_id,
    tokenizer_special_ids,
)
from graph_llm.dataload.embeddings import QwenEmbeddingEncoder
from graph_llm.dataload.sampler import LengthBucketSampler
from graph_llm.metrics.metrics import (
    assign_tail_demand_groups,
    bleu_score,
    compute_tail_demand_from_tokens,
    corpus_diversity,
    feature_coverage_ratio,
    feature_detect,
    feature_diversity,
    feature_matching_ratio,
    ids2words,
    ids_clear,
    rouge_score,
    unique_sentence_percent,
)
from graph_llm.metrics.rerank import RerankWeights, rerank_batch
from graph_llm.models.model import GraphEvidenceCIER, build_selector_outputs
from graph_llm.models.selector import EvidenceSelector


def seed_everything(seed=5254):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tokenizer_eos_ids(tokenizer):
    value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value if x is not None)
    if value is not None:
        return (int(value),)
    return (tokenizer_eos_id(tokenizer),)


def tokenizer_skip_ids(tokenizer):
    skip_ids = tokenizer_special_ids(tokenizer)
    skip_ids.discard(tokenizer_pad_id(tokenizer))
    for eos_id in tokenizer_eos_ids(tokenizer):
        skip_ids.discard(eos_id)
    return tuple(sorted(skip_ids))


def ensure_tokenizer_ready(tokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.bos_token is not None:
            tokenizer.pad_token = tokenizer.bos_token
    return tokenizer


def load_item_meta(args):
    item_path = Path(args.data_dir) / args.dataset_name / "item.json"
    if not item_path.is_file():
        print(f"WARNING: item metadata not found at {item_path}; using empty item_meta.")
        return {}
    print(f"Loading item metadata from {item_path}")
    with item_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    meta = {str(row.get("item")): row for row in rows if row.get("item") is not None}
    print(f"Loaded metadata for {len(meta)} items")
    return meta


def parse_device_ids(devices_value):
    if isinstance(devices_value, int):
        devices_value = str(devices_value)
    parts = [part.strip() for part in str(devices_value).split(",") if part.strip()]
    if parts and parts[0].lower() in {"auto", "default"} and len(parts) == 1:
        return [default_preferred_device_id()]
    if not parts:
        raise ValueError("At least one CUDA device must be specified via --devices")
    device_ids = []
    for part in parts:
        if part.lower() in {"auto", "default"}:
            continue
        if part.startswith("cuda:"):
            part = part.split(":", 1)[1]
        device_ids.append(int(part))
    if not device_ids:
        device_ids = [default_preferred_device_id()]
    return device_ids


def available_cuda_device_ids():
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def default_preferred_device_id():
    available = available_cuda_device_ids()
    if 1 in available:
        return 1
    if 0 in available:
        return 0
    if available:
        return available[0]
    raise RuntimeError("No CUDA devices are available.")


def resolve_devices_string(devices_value):
    text = str(devices_value).strip()
    if text.lower() in {"auto", "default", ""}:
        return str(default_preferred_device_id())
    return text


def is_explicit_single_device(raw_devices):
    text = str(raw_devices).strip().lower()
    return text not in {"auto", "default", ""} and "," not in text


def flash_attn_available():
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_attn_implementation(requested):
    if requested != "flash_attention_2":
        return requested
    if flash_attn_available():
        return "flash_attention_2"
    raise ImportError(
        "--attn_implementation flash_attention_2 requires the flash_attn package in "
        "the active Python environment. Use graph_llm_fa2 (see "
        "graph_llm/aux/setup_graph_fa2_env.sh) or install flash-attn manually."
    )


def dual_device_string(preferred_id):
    available = set(available_cuda_device_ids())
    if preferred_id not in available:
        preferred_id = default_preferred_device_id()
    others = sorted(device_id for device_id in available if device_id != preferred_id)
    if not others:
        return str(preferred_id)
    return f"{preferred_id},{others[0]}"


def resolve_embedding_device(embedding_device_arg, primary_device, device_ids):
    if embedding_device_arg == "auto":
        if len(device_ids) > 1:
            return torch.device(f"cuda:{device_ids[-1]}")
        return primary_device
    if embedding_device_arg.startswith("cuda:"):
        return torch.device(embedding_device_arg)
    return torch.device(f"cuda:{int(embedding_device_arg)}")


def resolve_training_devices(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Training this backbone model requires a CUDA GPU.")
    available = set(available_cuda_device_ids())
    device_ids = parse_device_ids(args.devices)
    for device_id in device_ids:
        if device_id not in available:
            raise RuntimeError(
                f"Requested cuda:{device_id} is unavailable. "
                f"Available devices: {sorted(available)}"
            )
    primary_id = device_ids[0]
    torch.cuda.set_device(primary_id)
    primary_device = torch.device(f"cuda:{primary_id}")
    embedding_device = resolve_embedding_device(
        args.embedding_device,
        primary_device,
        device_ids,
    )
    return primary_device, embedding_device, device_ids


def training_cuda_devices(primary_device, embedding_device, embedding_backend):
    devices = [primary_device]
    if embedding_backend == "qwen_embedding" and embedding_device != primary_device:
        devices.append(embedding_device)
    return devices


def devices_layout_string(device_ids, primary_device, embedding_device):
    devices_str = ",".join(str(device_id) for device_id in device_ids)
    return (
        f"devices:{devices_str} "
        f"primary_device:{primary_device} "
        f"embedding_device:{embedding_device}"
    )


def resolve_llm_device_map_mode(args, device_ids):
    mode = args.llm_device_map
    if mode == "auto":
        return "balanced" if len(device_ids) >= 2 else "single"
    return mode


def gpu_total_gib(device_id):
    props = torch.cuda.get_device_properties(device_id)
    return props.total_memory / (1024 ** 3)


def build_llm_max_memory(device_ids, embedding_device, reserve_embedding_gib, args):
    """Build per-GPU LLM weight budgets for a fast balanced layer split."""
    primary_id = device_ids[0]
    embed_id = embedding_device.index
    fraction = float(args.gpu_memory_fraction)

    secondary_total = gpu_total_gib(embed_id)
    primary_total = gpu_total_gib(primary_id)

    secondary_llm = secondary_total * fraction - float(reserve_embedding_gib)
    if args.secondary_gpu_max_gib > 0:
        secondary_llm = min(secondary_llm, float(args.secondary_gpu_max_gib))

    primary_available = primary_total * fraction
    if primary_id == 0:
        primary_available -= float(args.primary_foreign_reserve_gib)
    primary_llm = secondary_llm * float(args.primary_gpu_balance_ratio)
    primary_llm = min(primary_llm, primary_available)
    if args.primary_gpu_max_gib > 0:
        primary_llm = min(primary_llm, float(args.primary_gpu_max_gib))

    primary_llm = max(4, int(primary_llm))
    secondary_llm = max(4, int(secondary_llm))
    return {
        primary_id: f"{primary_llm}GiB",
        embed_id: f"{secondary_llm}GiB",
    }


def describe_module_device_map(module):
    device_map = getattr(module, "hf_device_map", None)
    if not device_map:
        return "single_device"
    counts = {}
    for device in device_map.values():
        counts[str(device)] = counts.get(str(device), 0) + 1
    return " ".join(f"{device}:{count}" for device, count in sorted(counts.items()))


@dataclass(frozen=True)
class OOMPlan:
    name: str
    devices: str
    batch_size: int | None
    eval_batch_size: int | None
    gradient_checkpointing: bool
    llm_device_map: str

    def describe(self):
        batch = self.batch_size if self.batch_size is not None else "baseline"
        eval_batch = self.eval_batch_size if self.eval_batch_size is not None else "baseline"
        return (
            f"{self.name} devices={self.devices} batch_size={batch} "
            f"eval_batch_size={eval_batch} gradient_checkpointing={self.gradient_checkpointing} "
            f"llm_device_map={self.llm_device_map}"
        )


def build_oom_plans(baseline):
    preferred = default_preferred_device_id()
    single_devices = str(preferred)
    dual_devices = dual_device_string(preferred)
    has_dual = dual_devices != single_devices
    batch_size = int(baseline.batch_size)
    eval_batch_size = int(baseline.eval_batch_size or batch_size)

    plans = [
        OOMPlan("single_fast", single_devices, batch_size, eval_batch_size, False, "single"),
        OOMPlan("single_checkpointing", single_devices, batch_size, eval_batch_size, True, "single"),
    ]
    if has_dual:
        plans.extend([
            OOMPlan("dual_balanced", dual_devices, batch_size, eval_batch_size, False, "balanced"),
            OOMPlan("dual_checkpointing", dual_devices, batch_size, eval_batch_size, True, "balanced"),
        ])

    reduced_batches = sorted(
        {
            max(2, batch_size // 2),
            max(2, batch_size // 4),
            2,
        }
        - {batch_size},
        reverse=True,
    )
    for reduced in reduced_batches:
        plans.append(
            OOMPlan(
                f"single_batch{reduced}",
                single_devices,
                reduced,
                reduced,
                False,
                "single",
            )
        )
        plans.append(
            OOMPlan(
                f"single_batch{reduced}_checkpointing",
                single_devices,
                reduced,
                reduced,
                True,
                "single",
            )
        )
        if has_dual:
            plans.append(
                OOMPlan(
                    f"dual_batch{reduced}_checkpointing",
                    dual_devices,
                    reduced,
                    reduced,
                    True,
                    "balanced",
                )
            )
    return plans


def build_run_oom_plans(args, baseline, user_pinned_single):
    if args.oom_fallback == "auto" and not user_pinned_single:
        return build_oom_plans(baseline)
    if user_pinned_single:
        print(
            f"Device pinned to {args.devices}: OOM fallback disabled. "
            f"First OOM will raise (no batch reduction, no multi-GPU spread)."
        )
    return [
        OOMPlan(
            "manual",
            args.devices,
            None,
            None,
            args.gradient_checkpointing,
            args.llm_device_map,
        )
    ]


def apply_oom_plan(args, plan, baseline):
    args.devices = plan.devices
    args.batch_size = plan.batch_size if plan.batch_size is not None else baseline.batch_size
    args.eval_batch_size = (
        plan.eval_batch_size
        if plan.eval_batch_size is not None
        else baseline.eval_batch_size
    )
    args.gradient_checkpointing = plan.gradient_checkpointing
    args.llm_device_map = plan.llm_device_map
    args.active_oom_plan = plan.name
    args.active_oom_plan_desc = plan.describe()


def release_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for device_id in available_cuda_device_ids():
            torch.cuda.synchronize(device_id)


def build_dataset(args, tokenizer):
    data_path = Path(args.data_dir) / args.dataset_name
    cache_path = dataset_cache_path(args)
    required = {
        "user", "item", "raw_user", "raw_item", "text", "keyword",
        "keyword_words", "review_text", "rating",
    }
    if cache_path.exists() and not args.rebuild_dataset_cache:
        print(f"Loading dataset cache: {cache_path}")
        dataset = pd.read_pickle(cache_path)
        if required.issubset(dataset.columns):
            if dataset["rating"].min() >= 1:
                dataset["rating"] = [int(x - 1) for x in dataset["rating"].tolist()]
            print(f"Loaded cached dataset with {len(dataset)} rows")
            return dataset

    reviews_path = data_path / "reviews.pickle"
    print(f"Reading reviews from {reviews_path}")
    raw_reviews = pd.read_pickle(reviews_path)
    dataset = pd.DataFrame(raw_reviews)
    dataset["raw_user"] = dataset["user"].astype(str)
    dataset["raw_item"] = dataset["item"].astype(str)

    encoder = LabelEncoder()
    dataset["user"] = encoder.fit_transform(dataset["raw_user"]).tolist()
    dataset["item"] = encoder.fit_transform(dataset["raw_item"]).tolist()

    keywords, keyword_words, text, review_text = [], [], [], []
    eos_id = tokenizer_eos_id(tokenizer)
    for row in tqdm(dataset["template"], desc="Tokenizing explanations"):
        keywords.append(tokenizer(row[0], add_special_tokens=False)["input_ids"])
        keyword_words.append(row[0])
        review_text.append(row[2])
        text.append(tokenizer(row[2], add_special_tokens=False)["input_ids"] + [eos_id])
    dataset["text"] = text
    dataset["keyword"] = keywords
    dataset["keyword_words"] = keyword_words
    dataset["review_text"] = review_text
    dataset = dataset[
        [
            "user", "item", "raw_user", "raw_item", "text", "keyword",
            "keyword_words", "review_text", "rating",
        ]
    ]
    dataset.to_pickle(cache_path)
    dataset["rating"] = [int(x - 1) for x in dataset["rating"].tolist()]
    return dataset


def profile_dataset_name_candidates(dataset_name):
    clean_name = str(dataset_name).strip("/")
    if not clean_name:
        return []
    parts = [part for part in clean_name.split("/") if part]
    candidates = ["/".join(parts)]
    if len(parts) >= 2:
        prefix = parts[:-1]
        leaf = parts[-1]
        while "_" in leaf:
            leaf = leaf.rsplit("_", 1)[0]
            candidates.append("/".join(prefix + [leaf]))
    return list(dict.fromkeys(candidates))


def profile_cache_candidate_paths(args, split_index, scope):
    base = Path(args.profile_dir)
    filename = f"fold_{split_index}_{scope}.pkl"
    candidates = [
        base / dataset_name / filename
        for dataset_name in profile_dataset_name_candidates(args.dataset_name)
    ]
    candidates.append(base / filename)
    return list(dict.fromkeys(candidates))


def profile_cache_path(args, split_index, scope):
    candidates = profile_cache_candidate_paths(args, split_index, scope)
    for path in candidates:
        if path.exists():
            return path
    return candidates[1] if len(candidates) > 1 else candidates[0]


def preflight_profile_cache_files(args, split_indices):
    if args.allow_missing_profiles:
        print("WARNING: --allow_missing_profiles is set; missing profile caches will use empty profiles.")
        return
    missing = []
    for split_index in split_indices:
        for scope in ["train", "train_valid"]:
            candidates = profile_cache_candidate_paths(args, split_index, scope)
            if not any(path.exists() for path in candidates):
                missing.append((split_index, scope, candidates))
    if missing:
        lines = []
        for split_index, scope, candidates in missing:
            lines.append(f"  - fold {split_index} {scope}; tried:")
            lines.extend(f"      {path}" for path in candidates)
        print(
            "WARNING: Profile cache preflight found missing files; "
            "falling back to empty profiles:\n"
            f"{chr(10).join(lines)}"
        )
        args.allow_missing_profiles = True


def dataset_safe_name_variants(dataset_name):
    """Return filesystem-safe dataset name variants used by caches."""
    text = str(dataset_name)
    clean = text.strip("/")
    variants = []
    for value in (clean.replace("/", "__"), text.replace("/", "__")):
        if value and value not in variants:
            variants.append(value)
    return variants


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"Removed {path}")


def apply_force(args, split_indices):
    if args.force and args.only_eval:
        raise ValueError(
            "--force regenerates checkpoints via fresh training; "
            "it conflicts with --only_eval. Drop --only_eval to use --force."
        )
    if not args.force:
        return

    from graph_llm.aux.generate_profiles import generate_profiles

    dataset_name = str(args.dataset_name).strip("/")
    data_dir = Path(args.data_dir)
    graph_cache_dir = Path(args.graph_cache_dir)
    embedding_cache_dir = Path(args.embedding_cache_dir)
    ckpt_dir = Path(args.ckpt_dir)
    profile_dir = Path(args.profile_dir)
    safe_variants = dataset_safe_name_variants(args.dataset_name)

    print(f"--force: regenerating artifacts for dataset={dataset_name!r}, folds={split_indices}")

    dataset_dir = data_dir / dataset_name
    for cache_file in dataset_dir.glob("dataset_keywords_cache_*.pickle"):
        _remove_path(cache_file)

    for split_index in split_indices:
        fold = str(split_index)
        for safe_name in safe_variants:
            graph_fold = graph_cache_dir / safe_name / f"fold_{fold}"
            _remove_path(graph_fold)
            embedding_fold = embedding_cache_dir / safe_name / fold
            _remove_path(embedding_fold)

        ckpt_prefix = ckpt_dir / dataset_name / fold
        for suffix in ("model", "selector.bin"):
            _remove_path(Path(f"{ckpt_prefix}{suffix}"))
        for suffix in ("graph_config.json",):
            _remove_path(Path(f"{ckpt_prefix}{suffix}"))

        print(f"--force: regenerating profiles for fold {fold}")
        generate_profiles(
            data_dir=data_dir,
            dataset_name=dataset_name,
            profile_dir=profile_dir,
            fold=fold,
            scopes=["train", "train_valid"],
        )

    args.rebuild_dataset_cache = True
    args.rebuild_graph_cache = True
    print("--force: artifact cleanup complete; starting fresh training run.")


def unpack_batch(batch, device):
    (
        input_ids,
        rating,
        profile_ids,
        profile_mask,
        target_item_ids,
        target_item_mask,
        graph_tensors,
        graphs,
        item_texts,
        item_titles,
        raw_users,
        feature_position_mask,
        feature_position_weights,
    ) = batch
    graph_tensors = {k: v.to(device) for k, v in graph_tensors.items()}
    return (
        input_ids.to(device),
        rating.to(device),
        profile_ids.to(device),
        profile_mask.to(device),
        target_item_ids.to(device),
        target_item_mask.to(device),
        graph_tensors,
        graphs,
        item_texts,
        item_titles,
        raw_users,
        feature_position_mask.to(device),
        feature_position_weights.to(device),
    )


def build_batch_prompt_tensors(
    item_titles,
    raw_users,
    tokenizer,
    args,
    device,
):
    pad_id = tokenizer_pad_id(tokenizer)
    generation_prompt_ids, generation_prompt_mask = build_generation_prompt_batch(
        item_titles,
        raw_users,
        tokenizer,
        pad_id,
        max_tokens=args.max_generation_prompt_tokens,
    )
    return (
        generation_prompt_ids.to(device),
        generation_prompt_mask.to(device),
    )


def compute_batch_selector_tensors(
    model,
    embedding_encoder,
    graphs,
    graph_tensors,
    item_texts,
    tokenizer,
    args,
    device,
):
    protected = set(parse_special_token_ids(args.special_token_ids, tokenizer))
    protected.add(tokenizer_pad_id(tokenizer))
    protected.update(tokenizer_eos_ids(tokenizer))

    node_token_ids = graph_tensors["node_token_ids"]
    if node_token_ids.numel() == 0:
        batch_size = len(graphs)
        empty_long = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        empty_mask = torch.empty((batch_size, 0), dtype=torch.bool, device=device)
        empty_float = torch.empty((batch_size, 0), dtype=torch.float32, device=device)
        return empty_long, empty_mask, empty_long, empty_mask, empty_float

    unique_ids = sorted(set(node_token_ids.detach().cpu().tolist()))
    decode_fn = lambda tid: tokenizer.decode([int(tid)], skip_special_tokens=True)
    unique_emb = embedding_encoder.encode_token_ids(unique_ids, decode_fn)
    if unique_emb.device != device:
        unique_emb = unique_emb.to(device)

    item_embs = embedding_encoder.encode_texts(item_texts)
    if item_embs.device != device:
        item_embs = item_embs.to(device)
    id_to_idx = {tid: idx for idx, tid in enumerate(unique_ids)}
    node_indices = torch.tensor(
        [id_to_idx[int(t)] for t in node_token_ids.detach().cpu().tolist()],
        device=device,
        dtype=torch.long,
    )
    node_token_emb = unique_emb[node_indices]
    evidence_token_ids, evidence_token_mask, neg_token_ids, neg_token_mask, neg_weights = (
        build_selector_outputs(
            model.evidence_selector,
            graphs,
            node_token_emb,
            item_embs,
            top_m=args.top_m_evidence,
            ul_candidate_k=args.ul_candidate_k,
            protected_token_ids=protected,
        )
    )
    return (
        evidence_token_ids.to(device),
        evidence_token_mask.to(device),
        neg_token_ids.to(device),
        neg_token_mask.to(device),
        neg_weights.to(device),
    )


def cuda_memory_status(devices):
    if not torch.cuda.is_available():
        return "CUDA memory: n/a"
    if isinstance(devices, torch.device):
        devices = [devices]
    lines = []
    for device in devices:
        if device.type != "cuda":
            continue
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        lines.append(
            f"{device} GiB current allocated/reserved={allocated:.2f}/{reserved:.2f}, "
            f"peak allocated/reserved={peak_allocated:.2f}/{peak_reserved:.2f}"
        )
    if not lines:
        return "CUDA memory: n/a"
    return "CUDA memory " + "; ".join(lines)


def maybe_warn_cuda_memory(devices, warn_gib):
    if warn_gib <= 0 or not torch.cuda.is_available():
        return
    if isinstance(devices, torch.device):
        devices = [devices]
    for device in devices:
        if device.type != "cuda":
            continue
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        if peak_reserved > warn_gib:
            msg = (
                f"WARNING: {device} peak reserved memory {peak_reserved:.2f} GiB "
                f"exceeds --memory_warn_gib={warn_gib:g}"
            )
            print(msg)


def write_log(log_name, message):
    with open(log_name, "a+", encoding="utf-8") as f:
        f.write(message + "\n")


def train_epoch(
    model,
    embedding_encoder,
    dataloader,
    optimizer,
    device,
    cuda_devices,
    epoch,
    args,
    scaler,
    log_name,
    tokenizer,
):
    model.train()
    loss_log, nll_log, ul_log, feat_log = [], [], [], []
    apply_ul = epoch >= args.ul_start_epoch
    last_batch_idx = -1
    for cuda_device in cuda_devices:
        if cuda_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(cuda_device)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Train epoch {epoch}")):
        last_batch_idx = batch_idx
        (
            input_ids,
            _rating,
            profile_ids,
            profile_mask,
            target_item_ids,
            target_item_mask,
            graph_tensors,
            graphs,
            item_texts,
            item_titles,
            raw_users,
            feature_position_mask,
            feature_position_weights,
        ) = unpack_batch(batch, device)

        evidence_token_ids, evidence_token_mask, neg_token_ids, neg_token_mask, neg_weights = (
            compute_batch_selector_tensors(
                model, embedding_encoder, graphs, graph_tensors, item_texts, tokenizer, args, device,
            )
        )
        generation_prompt_ids, generation_prompt_mask = build_batch_prompt_tensors(
            item_titles,
            raw_users,
            tokenizer,
            args,
            device,
        )

        try:
            with autocast():
                loss, nll_loss, ul_loss, feat_loss = model.train_step(
                    input_ids,
                    profile_ids=profile_ids,
                    profile_mask=profile_mask,
                    target_item_ids=target_item_ids,
                    target_item_mask=target_item_mask,
                    generation_prompt_ids=generation_prompt_ids,
                    generation_prompt_mask=generation_prompt_mask,
                    neg_token_ids=neg_token_ids,
                    neg_token_mask=neg_token_mask,
                    evidence_token_ids=evidence_token_ids,
                    evidence_token_mask=evidence_token_mask,
                    neg_token_weights=neg_weights,
                    feature_position_mask=feature_position_mask,
                    feature_position_weights=feature_position_weights,
                    apply_unlikelihood=apply_ul,
                )
            loss_log.append(loss.item())
            nll_log.append(nll_loss.item())
            ul_log.append(ul_loss.item())
            feat_log.append(feat_loss.item())
            scaler.scale(loss / args.accumulation_steps).backward()
        except torch.cuda.OutOfMemoryError:
            msg = (
                "CUDA OOM during training. "
                f"epoch={epoch} batch_idx={batch_idx} batch_size={args.batch_size}. "
                f"{cuda_memory_status(cuda_devices)}"
            )
            print(msg)
            write_log(log_name, msg)
            raise

        if (batch_idx + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if (batch_idx + 1) % args.show_train_loss_steps == 0:
            msg = (
                f"Train Epoch: {epoch} "
                f"Loss: {np.mean(loss_log):.6f}\tNLL: {np.mean(nll_log):.6f}\t"
                f"GraphUL: {np.mean(ul_log):.6f}\tFeat: {np.mean(feat_log):.6f}\t"
                f"{cuda_memory_status(cuda_devices)}"
            )
            print(msg)
            write_log(log_name, msg)
            maybe_warn_cuda_memory(cuda_devices, args.memory_warn_gib)
            loss_log, nll_log, ul_log, feat_log = [], [], [], []

        if args.max_train_batches and (batch_idx + 1) >= args.max_train_batches:
            print(f"Stopping train epoch early after {args.max_train_batches} batches (--max_train_batches).")
            break

    if last_batch_idx >= 0 and (last_batch_idx + 1) % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    maybe_warn_cuda_memory(cuda_devices, args.memory_warn_gib)


def valid_step(model, embedding_encoder, dataloader, device, log_name, args, tokenizer, epoch=0):
    model.eval()
    loss_log = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Valid epoch {epoch}"):
            (
                input_ids,
                _rating,
                profile_ids,
                profile_mask,
                target_item_ids,
                target_item_mask,
                graph_tensors,
                graphs,
                item_texts,
                item_titles,
                raw_users,
                feature_position_mask,
                feature_position_weights,
            ) = unpack_batch(batch, device)
            evidence_token_ids, evidence_token_mask, neg_token_ids, neg_token_mask, neg_weights = (
                compute_batch_selector_tensors(
                    model, embedding_encoder, graphs, graph_tensors, item_texts, tokenizer, args, device,
                )
            )
            generation_prompt_ids, generation_prompt_mask = build_batch_prompt_tensors(
                item_titles,
                raw_users,
                tokenizer,
                args,
                device,
            )
            with autocast():
                loss, _nll, _ul, _feat = model.train_step(
                    input_ids,
                    profile_ids=profile_ids,
                    profile_mask=profile_mask,
                    target_item_ids=target_item_ids,
                    target_item_mask=target_item_mask,
                    generation_prompt_ids=generation_prompt_ids,
                    generation_prompt_mask=generation_prompt_mask,
                    neg_token_ids=neg_token_ids,
                    neg_token_mask=neg_token_mask,
                    evidence_token_ids=evidence_token_ids,
                    evidence_token_mask=evidence_token_mask,
                    neg_token_weights=neg_weights,
                    feature_position_mask=feature_position_mask,
                    feature_position_weights=feature_position_weights,
                    apply_unlikelihood=epoch >= args.ul_start_epoch,
                )
            loss_log.append(loss.item())
    avg_loss = float(np.mean(loss_log)) if loss_log else float("inf")
    print(f"valid Loss: {avg_loss}")
    write_log(log_name, f"valid Loss: {avg_loss}\n")
    return avg_loss


def save_generation_output(output_dir, predict, pad_token_id=0, eos_token_ids=(2,)):
    eos_ids = set(eos_token_ids or ())
    predict_text = []
    for row in predict:
        temp = []
        for item in row:
            item = int(item)
            if item == pad_token_id or item in eos_ids:
                break
            temp.append(item)
        predict_text.append(temp)
    pd.DataFrame({"text": predict_text}).to_pickle(output_dir)


def output_path_with_group(output_dir, group_name):
    root, ext = os.path.splitext(output_dir)
    if ext == "":
        return output_dir + "." + group_name
    return root + "." + group_name + ext


def get_tail_demand_eval_groups(test_dataset, tokenizer, args):
    pad_id = tokenizer_pad_id(tokenizer)
    eos_ids = tokenizer_eos_ids(tokenizer)
    skip_ids = tokenizer_skip_ids(tokenizer)
    reference_tokens = [
        ids2words(
            ids_clear(ids, pad_token_id=pad_id, eos_token_ids=eos_ids, skip_token_ids=skip_ids),
            tokenizer,
        )
        for ids in test_dataset.df["text"].tolist()
    ]
    demands = compute_tail_demand_from_tokens(reference_tokens)
    labels = assign_tail_demand_groups(
        demands,
        low_frac=args.tail_low_percent,
        high_frac=args.tail_high_percent,
    )
    indices = {
        "low": [idx for idx, label in enumerate(labels) if label == "low"],
        "high": [idx for idx, label in enumerate(labels) if label == "high"],
    }
    excluded_indices = [idx for idx, label in enumerate(labels) if label == "excluded"]

    def demand_stats(group_indices):
        group_demands = [demands[idx] for idx in group_indices]
        if len(group_demands) == 0:
            return "n=0"
        return "n={} | mean={:.6f} | min={:.6f} | max={:.6f}".format(
            len(group_demands),
            float(np.mean(group_demands)),
            float(np.min(group_demands)),
            float(np.max(group_demands)),
        )

    info = (
        "tail_low_percent: {} | tail_high_percent: {} | "
        "excluded_samples: {} | low_tail_demand: {} | high_tail_demand: {}"
    ).format(
        args.tail_low_percent,
        args.tail_high_percent,
        len(excluded_indices),
        demand_stats(indices["low"]),
        demand_stats(indices["high"]),
    )
    return {
        "info": info,
        "indices": indices,
        "excluded_indices": excluded_indices,
        "labels": labels,
        "tail_demand": demands,
    }


def append_eval_metrics(
    log_name,
    dataset,
    tokenizer,
    predict,
    label,
    output_dir,
    indices=None,
    group_name=None,
    group_info=None,
):
    if indices is None:
        indices = list(range(len(predict)))
    else:
        indices = list(indices)

    pad_id = tokenizer_pad_id(tokenizer)
    eos_ids = tokenizer_eos_ids(tokenizer)
    skip_ids = tokenizer_skip_ids(tokenizer)

    with open(log_name, "a+", encoding="utf-8") as f:
        if group_name is not None:
            info = "" if group_info is None else " | " + group_info
            f.write("eval_group: {} | samples: {}{}\n".format(group_name, len(indices), info))
        if len(indices) == 0:
            f.write("no samples\n")
            return

        group_predict = [predict[i] for i in indices]
        group_label = [label[i] for i in indices]
        group_features = [dataset.features[i] for i in indices]

        if output_dir is not None:
            save_generation_output(
                output_dir,
                group_predict,
                pad_token_id=pad_id,
                eos_token_ids=eos_ids,
            )

        tokens_test = [
            ids2words(
                ids_clear(ids, pad_token_id=pad_id, eos_token_ids=eos_ids, skip_token_ids=skip_ids),
                tokenizer,
            )
            for ids in group_label
        ]
        tokens_predict = [
            ids2words(
                ids_clear(ids, pad_token_id=pad_id, eos_token_ids=eos_ids, skip_token_ids=skip_ids),
                tokenizer,
            )
            for ids in group_predict
        ]
        f.write("BLEU-1 {:7.4f}\n".format(bleu_score(tokens_test, tokens_predict, n_gram=1)))
        f.write("BLEU-4 {:7.4f}\n".format(bleu_score(tokens_test, tokens_predict, n_gram=4)))
        usr, usn = unique_sentence_percent(tokens_predict)
        f.write("USR {:7.4f} | USN {:7}\n".format(usr, usn))
        d1, d2, entr = corpus_diversity(tokens_predict)
        f.write("Distinct-1 {:7.4f}\n".format(d1))
        f.write("Distinct-2 {:7.4f}\n".format(d2))
        f.write("ENTR {:7.4f}\n".format(entr))
        feature_set = set(group_features)
        feature_batch = feature_detect(tokens_predict, feature_set)
        div = 0.0 if len(feature_batch) < 2 else feature_diversity(feature_batch)
        f.write("DIV {:7.4f}\n".format(div))
        fcr = feature_coverage_ratio(feature_batch, feature_set) if len(feature_set) > 0 else 0.0
        f.write("FCR {:7.4f}\n".format(fcr))
        f.write("FMR {:7.4f}\n".format(feature_matching_ratio(feature_batch, group_features)))
        text_test = [" ".join(tokens) for tokens in tokens_test]
        text_predict = [" ".join(tokens) for tokens in tokens_predict]
        for key, value in rouge_score(text_test, text_predict).items():
            f.write("{} {:7.4f}\n".format(key, value))


def test_step(model, embedding_encoder, dataloader, device, log_name, dataset, output_dir, word, tokenizer, args):
    model.eval()
    predict, label = [], []
    max_batches = int(getattr(args, "max_eval_batches", 0) or 0)
    decode_strategy = getattr(args, "decode_strategy", "greedy")
    use_rerank = bool(getattr(args, "use_rerank", False))
    write_log(
        log_name,
        "decode_config: "
        f"strategy={decode_strategy} "
        f"num_beams={getattr(args, 'num_beams', 4)} "
        f"num_return_sequences={getattr(args, 'num_return_sequences', 4)} "
        f"use_rerank={use_rerank}",
    )
    rerank_weights = RerankWeights(
        logprob=float(getattr(args, "rerank_w_logprob", 1.0)),
        feature_match=float(getattr(args, "rerank_w_feature", 0.8)),
        evidence_coverage=float(getattr(args, "rerank_w_evidence", 0.5)),
        repetition=float(getattr(args, "rerank_w_repetition", 0.7)),
        generic=float(getattr(args, "rerank_w_generic", 0.5)),
    )
    pad_id = tokenizer_pad_id(tokenizer)
    eos_ids = tokenizer_eos_ids(tokenizer)
    skip_ids = tokenizer_skip_ids(tokenizer)
    sample_offset = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Test generate")):
            (
                input_ids,
                _rating,
                profile_ids,
                profile_mask,
                target_item_ids,
                target_item_mask,
                graph_tensors,
                graphs,
                item_texts,
                item_titles,
                raw_users,
                _feature_position_mask,
                _feature_position_weights,
            ) = unpack_batch(batch, device)
            evidence_token_ids, evidence_token_mask, _, _, _ = compute_batch_selector_tensors(
                model, embedding_encoder, graphs, graph_tensors, item_texts, tokenizer, args, device,
            )
            generation_prompt_ids, generation_prompt_mask = build_batch_prompt_tensors(
                item_titles,
                raw_users,
                tokenizer,
                args,
                device,
            )
            batch_size = profile_ids.shape[0]
            keyword_words_batch = [
                dataset.features[sample_offset + sample_idx]
                for sample_idx in range(batch_size)
            ]

            if decode_strategy == "beam":
                batch_candidates, batch_logprobs = model.beam_generate(
                    profile_ids,
                    profile_mask,
                    target_item_ids,
                    target_item_mask,
                    generation_prompt_ids,
                    generation_prompt_mask,
                    word,
                    device,
                    num_beams=getattr(args, "num_beams", 4),
                    num_return_sequences=getattr(args, "num_return_sequences", 4),
                    evidence_token_ids=evidence_token_ids,
                    evidence_token_mask=evidence_token_mask,
                )
                generated = rerank_batch(
                    batch_candidates,
                    batch_logprobs,
                    keyword_words_batch,
                    tokenizer,
                    evidence_token_ids=evidence_token_ids,
                    evidence_token_mask=evidence_token_mask,
                    use_rerank=use_rerank,
                    pad_token_id=pad_id,
                    eos_token_ids=eos_ids,
                    skip_token_ids=skip_ids,
                    weights=rerank_weights,
                )
            else:
                generated = model.greedy_generate(
                    profile_ids,
                    profile_mask,
                    target_item_ids,
                    target_item_mask,
                    generation_prompt_ids,
                    generation_prompt_mask,
                    word,
                    device,
                    evidence_token_ids=evidence_token_ids,
                    evidence_token_mask=evidence_token_mask,
                )
            predict.extend(generated)
            label.extend(input_ids.tolist())
            sample_offset += batch_size
            if max_batches and (batch_idx + 1) >= max_batches:
                print(f"Stopping test early after {max_batches} batches (--max_eval_batches).")
                break

    def _subset_group_indices(group_indices):
        n = len(predict)
        return [idx for idx in group_indices if idx < n]

    if getattr(args, "eval_tail_demand_groups", False):
        eval_groups = get_tail_demand_eval_groups(dataset, tokenizer, args)
        all_indices = list(range(len(predict)))
        group_info = eval_groups.get("info")
        if max_batches:
            group_info = (
                f"max_eval_batches: {max_batches} | evaluated_samples: {len(predict)} | "
                + (group_info or "")
            )
        append_eval_metrics(
            log_name,
            dataset,
            tokenizer,
            predict,
            label,
            output_path_with_group(output_dir, "all"),
            indices=all_indices,
            group_name="all",
            group_info=group_info,
        )
        for group_name, group_indices in eval_groups["indices"].items():
            subset = _subset_group_indices(group_indices)
            append_eval_metrics(
                log_name,
                dataset,
                tokenizer,
                predict,
                label,
                output_path_with_group(output_dir, group_name),
                indices=subset,
                group_name=group_name,
                group_info=group_info,
            )
    else:
        append_eval_metrics(log_name, dataset, tokenizer, predict, label, output_dir)


def load_best_checkpoint(model, ckpt_prefix, device):
    selector_path = ckpt_prefix + "selector.bin"
    adapter_path = ckpt_prefix + "model"
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_prefix}")
    model.model.load_adapter(adapter_path, "best_lora")
    model.model.set_adapter("best_lora")
    if os.path.exists(selector_path):
        state = torch.load(selector_path, map_location=device, weights_only=False)
        model.evidence_selector.load_state_dict(state)


def _run_split(
    args,
    split_index,
    tokenizer,
    dataset,
    item_meta,
    user_num,
    item_num,
    skip_token_ids,
):
    primary_device, embedding_device, device_ids = resolve_training_devices(args)
    device = primary_device
    device_layout = devices_layout_string(device_ids, primary_device, embedding_device)
    print(device_layout)
    if getattr(args, "active_oom_plan", None):
        print(f"Active OOM plan: {args.active_oom_plan} ({args.active_oom_plan_desc})")

    train_dataset, valid_dataset, test_dataset = dataset_split(dataset, split_index, args)
    train_history = train_dataset.copy()
    train_valid_history = pd.concat([train_dataset, valid_dataset], axis=0)

    train_profiles = load_profile_cache(
        profile_cache_path(args, split_index, "train"),
        allow_missing=args.allow_missing_profiles,
    )
    print(
        f"Loaded train profile cache: "
        f"{profile_cache_path(args, split_index, 'train')} ({len(train_profiles)} users)"
    )
    train_valid_profiles = load_profile_cache(
        profile_cache_path(args, split_index, "train_valid"),
        allow_missing=args.allow_missing_profiles,
    )
    print(
        f"Loaded train_valid profile cache: "
        f"{profile_cache_path(args, split_index, 'train_valid')} ({len(train_valid_profiles)} users)"
    )
    assert_profile_coverage(
        f"fold {split_index} train profile cache for train/validation",
        [train_dataset, valid_dataset],
        train_profiles,
        allow_missing=args.allow_missing_profiles,
    )
    assert_profile_coverage(
        f"fold {split_index} train_valid profile cache for test",
        test_dataset,
        train_valid_profiles,
        allow_missing=args.allow_missing_profiles,
    )
    cache_root = Path(args.graph_cache_dir)
    graph_train = GraphCacheManager.build_or_load(
        full_dataset=dataset,
        split_dataset=train_dataset,
        split_name="train",
        history_dataset=train_history,
        dataset_name=args.dataset_name,
        fold=int(split_index),
        tokenizer=tokenizer,
        skip_token_ids=skip_token_ids,
        cache_root=cache_root,
        max_nodes=args.max_graph_nodes,
        min_token_count=args.min_token_count,
        rebuild=args.rebuild_graph_cache,
    )
    graph_valid = GraphCacheManager.build_or_load(
        full_dataset=dataset,
        split_dataset=valid_dataset,
        split_name="validation",
        history_dataset=train_history,
        dataset_name=args.dataset_name,
        fold=int(split_index),
        tokenizer=tokenizer,
        skip_token_ids=skip_token_ids,
        cache_root=cache_root,
        max_nodes=args.max_graph_nodes,
        min_token_count=args.min_token_count,
        rebuild=args.rebuild_graph_cache,
    )
    graph_test = GraphCacheManager.build_or_load(
        full_dataset=dataset,
        split_dataset=test_dataset,
        split_name="test",
        history_dataset=train_valid_history,
        dataset_name=args.dataset_name,
        fold=int(split_index),
        tokenizer=tokenizer,
        skip_token_ids=skip_token_ids,
        cache_root=cache_root,
        max_nodes=args.max_graph_nodes,
        min_token_count=args.min_token_count,
        rebuild=args.rebuild_graph_cache,
    )

    train_set = GraphDataset(train_dataset, "train")
    valid_set = GraphDataset(valid_dataset, "validation")
    test_set = GraphDataset(test_dataset, "test")

    collate_train = GraphCollater(
        max_step=args.epochs * max(len(train_set), 1) // max(args.batch_size, 1),
        word=args.word,
        tokenizer=tokenizer,
        profile_records=train_profiles,
        max_profile_tokens=args.max_profile_tokens,
        item_meta=item_meta,
        max_target_item_tokens=args.max_target_item_tokens,
        item_description_mode=args.item_description_mode,
        graph_manager=graph_train,
        split_name="train",
    )
    collate_valid = GraphCollater(
        max_step=1,
        word=args.word,
        tokenizer=tokenizer,
        profile_records=train_profiles,
        max_profile_tokens=args.max_profile_tokens,
        item_meta=item_meta,
        max_target_item_tokens=args.max_target_item_tokens,
        item_description_mode=args.item_description_mode,
        graph_manager=graph_valid,
        split_name="validation",
    )
    collate_test = GraphCollater(
        max_step=1,
        word=args.word,
        tokenizer=tokenizer,
        profile_records=train_valid_profiles,
        max_profile_tokens=args.max_profile_tokens,
        item_meta=item_meta,
        max_target_item_tokens=args.max_target_item_tokens,
        item_description_mode=args.item_description_mode,
        graph_manager=graph_test,
        split_name="test",
    )

    eval_batch_size = args.eval_batch_size or args.batch_size
    print(
        f"Creating dataloaders: train={len(train_set)}, valid={len(valid_set)}, "
        f"test={len(test_set)}, batch_size={args.batch_size}"
    )
    profile_lengths = compute_profile_lengths(
        train_set,
        train_profiles,
        tokenizer,
        args.max_profile_tokens,
    )
    train_sampler = LengthBucketSampler(
        profile_lengths,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, collate_fn=collate_train,
        sampler=train_sampler, pin_memory=True, num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=eval_batch_size, collate_fn=collate_valid,
        shuffle=False, pin_memory=True, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set, batch_size=eval_batch_size, collate_fn=collate_test,
        shuffle=False, pin_memory=True, num_workers=args.num_workers,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=parse_csv(args.lora_target_modules),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    llm_device_map_mode = resolve_llm_device_map_mode(args, device_ids)
    embedding_path = Path(args.embedding_model_path)
    reserve_embedding_gib = (
        args.embedding_device_reserve_gib
        if _is_local_model_dir(embedding_path)
        else 0.0
    )
    llm_load_kwargs = {
        "dtype": resolve_torch_dtype(args.torch_dtype),
        "local_files_only": True,
        "trust_remote_code": True,
    }
    attn_impl = resolve_attn_implementation(args.attn_implementation)
    if attn_impl != "auto":
        llm_load_kwargs["attn_implementation"] = attn_impl
    if llm_device_map_mode == "balanced" and len(device_ids) >= 2:
        max_memory = build_llm_max_memory(
            device_ids,
            embedding_device,
            reserve_embedding_gib,
            args,
        )
        llm_load_kwargs["device_map"] = "auto"
        llm_load_kwargs["max_memory"] = max_memory
        print(
            f"Loading Qwen backbone from {args.model_path} with balanced layer split "
            f"(no checkpointing by default for speed) max_memory={max_memory} ..."
        )
    else:
        llm_load_kwargs["device_map"] = {"": str(primary_device)}
        print(f"Loading Qwen backbone from {args.model_path} on {primary_device} ...")
    model_llm = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **llm_load_kwargs,
    )
    if hasattr(model_llm, "hf_device_map") and model_llm.hf_device_map:
        print(f"LLM layer placement: {describe_module_device_map(model_llm)}")
    model_llm.config.use_cache = False
    if args.gradient_checkpointing:
        model_llm.gradient_checkpointing_enable()
        model_llm.enable_input_require_grads()
        print("Enabled gradient checkpointing (slower, lower memory).")
    model_llm = get_peft_model(model_llm, lora_config)
    model_llm.print_trainable_parameters()
    if hasattr(model_llm, "hf_device_map") and model_llm.hf_device_map:
        print(f"LoRA layer placement: {describe_module_device_map(model_llm)}")

    print(f"Initializing embedding encoder from {args.embedding_model_path} on {embedding_device} ...")
    embedding_encoder = QwenEmbeddingEncoder(
        args.embedding_model_path,
        device=embedding_device,
        cache_dir=Path(args.embedding_cache_dir) / args.dataset_name.replace("/", "__") / str(split_index),
        fallback_lm=model_llm,
        local_files_only=not args.download_embedding_model,
    )
    if embedding_encoder.backend != "qwen_embedding" and len(device_ids) > 1:
        print(
            "WARNING: embedding model unavailable; using LM embed_tokens fallback. "
            "Embedding encoder is not resident on the embedding device; "
            "dual-GPU memory benefit is reduced."
        )
    cuda_devices = training_cuda_devices(
        primary_device,
        embedding_device,
        embedding_encoder.backend,
    )
    selector = EvidenceSelector(
        embed_dim=embedding_encoder.hidden_size,
        hidden_dim=args.selector_hidden,
        gnn_layers=args.gnn_layers,
    ).to(device)

    model = GraphEvidenceCIER(
        tokenizer=tokenizer,
        vocab_size=model_llm.config.vocab_size,
        evidence_selector=selector,
        lambda_ul=args.lambda_ul,
        lambda_feat=args.lambda_feat,
        evidence_bonus=args.evidence_bonus,
        max_consecutive_token_repeat=args.max_consecutive_token_repeat,
        pad_token_id=tokenizer_pad_id(tokenizer),
        eos_token_ids=tokenizer_eos_ids(tokenizer),
        special_token_ids=parse_special_token_ids(args.special_token_ids, tokenizer),
    ).to(device)
    model.model = model_llm

    optimizer = AdamW([
        {
            "params": [p for p in model.evidence_selector.parameters() if p.requires_grad],
            "lr": args.learning_rate,
        },
        {
            "params": [p for p in model.model.parameters() if p.requires_grad],
            "lr": args.learning_rate / 10,
        },
    ])
    scaler = GradScaler()

    ckpt_dir = Path(args.ckpt_dir) / args.dataset_name
    log_dir = Path(args.log_dir) / args.dataset_name
    output_dir = Path(args.output_dir) / args.dataset_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_prefix = str(ckpt_dir / split_index)
    log_name = str(log_dir / args.log_name)
    generation_path = str(output_dir / f"{split_index}generate.dataset")

    with open(log_name, "a+", encoding="utf-8") as f:
        f.write(args.model_path + "\n")
        f.write(f"split_index:{split_index}\n")
        f.write(
            f"batch_size:{args.batch_size} eval_batch_size:{args.eval_batch_size} "
            f"accumulation_steps:{args.accumulation_steps} word:{args.word} "
            f"gradient_checkpointing:{args.gradient_checkpointing}\n"
        )
        f.write(
            f"lora_r:{args.lora_r} lora_alpha:{args.lora_alpha} "
            f"lora_dropout:{args.lora_dropout} "
            f"lora_target_modules:{args.lora_target_modules}\n"
        )
        f.write(
            f"max_profile_tokens:{args.max_profile_tokens} "
            f"max_target_item_tokens:{args.max_target_item_tokens} "
            f"max_generation_prompt_tokens:{args.max_generation_prompt_tokens} "
            f"max_graph_nodes:{args.max_graph_nodes}\n"
        )
        f.write(
            f"lambda_ul:{args.lambda_ul} lambda_feat:{args.lambda_feat} "
            f"top_m_evidence:{args.top_m_evidence} "
            f"ul_candidate_k:{args.ul_candidate_k}\n"
        )
        f.write(
            f"ul_start_epoch:{args.ul_start_epoch} evidence_bonus:{args.evidence_bonus} "
            f"max_consecutive_token_repeat:{args.max_consecutive_token_repeat}\n"
        )
        f.write(
            "feature_loss: enabled via lambda_feat * feat on matched target positions, "
            f"gated by ul_start_epoch={args.ul_start_epoch}\n"
        )
        f.write(f"{device_layout}\n")
        f.write(f"embedding_backend:{embedding_encoder.backend}\n")
        f.write(f"active_oom_plan:{getattr(args, 'active_oom_plan', 'manual')}\n")
        f.write(f"active_oom_plan_desc:{getattr(args, 'active_oom_plan_desc', '')}\n")
        f.write(
            f"llm_device_map:{llm_device_map_mode} "
            f"gradient_checkpointing:{args.gradient_checkpointing} "
            f"attn_implementation:{args.attn_implementation} "
            f"primary_foreign_reserve_gib:{args.primary_foreign_reserve_gib} "
            f"primary_gpu_balance_ratio:{args.primary_gpu_balance_ratio} "
            f"embedding_device_reserve_gib:{args.embedding_device_reserve_gib}\n"
        )
        if hasattr(model_llm, "hf_device_map") and model_llm.hf_device_map:
            f.write(f"llm_layer_placement:{describe_module_device_map(model_llm)}\n")

    best_loss = float("inf")
    early_stop = args.early_stop_patience
    if not args.only_eval:
        for epoch in range(args.epochs):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            train_epoch(
                model, embedding_encoder, train_loader, optimizer, device,
                cuda_devices, epoch, args, scaler, log_name, tokenizer,
            )
            collate_train.cur_step = len(train_loader) * (epoch + 1)
            valid_loss = valid_step(
                model, embedding_encoder, valid_loader, device, log_name, args, tokenizer, epoch=epoch,
            )
            with open(log_name, "a+", encoding="utf-8") as f:
                if best_loss < valid_loss:
                    early_stop -= 1
                else:
                    print("save model")
                    f.write("save model\n")
                    best_loss = valid_loss
                    model.model.save_pretrained(ckpt_prefix + "model")
                    torch.save(model.evidence_selector.state_dict(), ckpt_prefix + "selector.bin")
                    with open(ckpt_prefix + "graph_config.json", "w", encoding="utf-8") as cf:
                        json.dump(vars(args), cf, indent=2, default=str)
            if early_stop == 0:
                break

    load_best_checkpoint(model, ckpt_prefix, device)
    if args.gradient_checkpointing and hasattr(model.model, "gradient_checkpointing_disable"):
        model.model.gradient_checkpointing_disable()
    test_step(
        model, embedding_encoder, test_loader, device, log_name,
        test_set, generation_path, args.word, tokenizer, args,
    )
    torch.cuda.empty_cache()

def run(args):
    seed_everything(args.seed)
    resolve_dataset_paths(args)
    _prompt_budget = (
        args.max_profile_tokens
        + args.max_target_item_tokens
        + args.max_generation_prompt_tokens
    )
    if _prompt_budget > 512:
        print(
            f"WARNING: prompt token budget ({_prompt_budget}) > 512. "
            f"profile={args.max_profile_tokens} item={args.max_target_item_tokens} "
            f"generation={args.max_generation_prompt_tokens}"
        )
    args.model_path = resolve_local_model_path(
        args.model_path,
        candidates=qwen3_4b_model_candidates(),
    )
    split_indices = parse_csv(args.split_indices)
    if getattr(args, "force", False):
        apply_force(args, split_indices)
    preflight_profile_cache_files(args, split_indices)
    raw_devices = str(args.devices).strip()
    user_pinned_single = is_explicit_single_device(raw_devices)
    baseline = snapshot_training_args(args)
    args.devices = resolve_devices_string(args.devices)
    baseline.devices = args.devices
    oom_plans = build_run_oom_plans(args, baseline, user_pinned_single)
    print(
        f"Preferred GPU: cuda:{default_preferred_device_id()} "
        f"initial devices={args.devices} oom_fallback={args.oom_fallback} "
        f"plans={len(oom_plans)}"
    )
    tokenizer = ensure_tokenizer_ready(AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    ))
    dataset = build_dataset(args, tokenizer)
    item_meta = load_item_meta(args)
    user_num = int(dataset["user"].max()) + 1
    item_num = int(dataset["item"].max()) + 1
    skip_token_ids = set(parse_special_token_ids(args.special_token_ids, tokenizer))
    skip_token_ids.add(tokenizer_pad_id(tokenizer))
    skip_token_ids.update(tokenizer_eos_ids(tokenizer))
    print(f"Dataset {args.dataset_name}: users={user_num}, items={item_num}, rows={len(dataset)}")

    log_name = str(Path(args.log_dir) / args.dataset_name / args.log_name)
    for split_index in split_indices:
        last_oom = None
        for plan_idx, plan in enumerate(oom_plans):
            apply_oom_plan(args, plan, baseline)
            args.devices = resolve_devices_string(args.devices)
            if plan_idx == 0:
                print(f"OOM plan 1/{len(oom_plans)}: {plan.describe()}")
            else:
                msg = (
                    f"Switching OOM fallback plan {plan_idx + 1}/{len(oom_plans)}: "
                    f"{plan.describe()}"
                )
                print(msg)
                write_log(log_name, msg)
            try:
                _run_split(
                    args,
                    split_index,
                    tokenizer,
                    dataset,
                    item_meta,
                    user_num,
                    item_num,
                    skip_token_ids,
                )
                if plan_idx > 0:
                    msg = f"Training succeeded with OOM plan: {plan.name}"
                    print(msg)
                    write_log(log_name, msg)
                break
            except torch.cuda.OutOfMemoryError as exc:
                last_oom = exc
                msg = f"OOM on plan {plan.name}: {exc}"
                print(msg)
                write_log(log_name, msg)
                release_cuda_memory()
        else:
            raise RuntimeError(
                f"All OOM fallback plans exhausted for fold {split_index}. "
                f"Last error: {last_oom}"
            ) from last_oom


