"""Argument parsing and default configuration for graph_llm."""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import torch

from graph_llm.dataload.embeddings import default_embedding_model_path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent


def qwen3_4b_model_candidates():
    return [
        PACKAGE_ROOT / "pretrain_llm" / "qwen3-4b",
        REPO_ROOT / "gpt1" / "models" / "qwen3-4b",
    ]


def _is_local_model_dir(path):
    path = Path(path)
    return path.is_dir() and (path / "config.json").is_file()


def resolve_local_model_path(model_path, *, candidates=None, model_label="Qwen3-4B"):
    path = Path(model_path).expanduser()
    if _is_local_model_dir(path):
        return str(path.resolve())

    search_paths = [path]
    if candidates is not None:
        search_paths.extend(candidates)
    elif path.name == "qwen3-4b":
        search_paths.extend(qwen3_4b_model_candidates())

    seen = set()
    for candidate in search_paths:
        candidate = Path(candidate).expanduser()
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _is_local_model_dir(candidate):
            if candidate.resolve() != path.resolve() and path.exists() is False:
                print(
                    f"WARNING: local model path {path} not found; "
                    f"using {candidate.resolve()}"
                )
            return str(candidate.resolve())

    download_hint = (
        f"bash {PACKAGE_ROOT / 'aux' / 'download_qwen3_4b.sh'} "
        f"or bash {PACKAGE_ROOT / 'aux' / 'setup_graph_env.sh'}"
    )
    tried = ", ".join(str(item) for item in search_paths)
    raise FileNotFoundError(
        f"{model_label} model not found at {path}. Tried: {tried}. "
        f"Download with: {download_hint}"
    )


def default_model_path():
    for local in qwen3_4b_model_candidates():
        if _is_local_model_dir(local):
            return str(local.resolve())
    return str(qwen3_4b_model_candidates()[0])


def model_cache_namespace(args):
    name = Path(args.model_path).name or "model"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def dataset_cache_path(args):
    data_path = Path(args.data_dir) / args.dataset_name
    return data_path / f"dataset_keywords_cache_{model_cache_namespace(args)}.pickle"


def resolve_torch_dtype(value):
    if value == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return mapping[value]


def parse_special_token_ids(value, tokenizer):
    from graph_llm.dataload.dataloader import tokenizer_special_ids

    if value == "auto":
        return tuple(sorted(tokenizer_special_ids(tokenizer)))
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def parse_csv(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def snapshot_training_args(args):
    return copy.copy(args)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train profile-conditioned Qwen3-4B LoRA explainer."
    )
    parser.add_argument("--device", "--devices", dest="devices", default="default", type=str)
    parser.add_argument("--embedding_device", default="auto", type=str)
    parser.add_argument("--memory_warn_gib", default=24.0, type=float)
    parser.add_argument(
        "--oom_fallback",
        choices=["auto", "off"],
        default="auto",
        help="auto: on CUDA OOM, retry with progressively lower-memory plans",
    )
    parser.add_argument(
        "--llm_device_map",
        choices=["auto", "single", "balanced"],
        default="single",
        help="single: keep Qwen on primary GPU; balanced: split Qwen layers across --devices",
    )
    parser.add_argument(
        "--primary_gpu_max_gib",
        default=0.0,
        type=float,
        help="Optional hard cap on cuda:0 LLM budget; 0 means auto-balance",
    )
    parser.add_argument(
        "--secondary_gpu_max_gib",
        default=0.0,
        type=float,
        help="Optional hard cap on secondary GPU LLM budget; 0 means auto",
    )
    parser.add_argument(
        "--primary_foreign_reserve_gib",
        default=12.0,
        type=float,
        help="Memory reserved on cuda:0 for other processes when auto-balancing LLM split",
    )
    parser.add_argument(
        "--primary_gpu_balance_ratio",
        default=0.92,
        type=float,
        help="cuda:0 LLM budget as a fraction of cuda:1 LLM budget; keep near 1.0 for balance",
    )
    parser.add_argument(
        "--embedding_device_reserve_gib",
        default=10.0,
        type=float,
        help="Memory reserved on embedding GPU for Qwen3-Embedding before LLM layer placement",
    )
    parser.add_argument(
        "--gpu_memory_fraction",
        default=0.92,
        type=float,
        help="Fraction of total GPU memory usable when building balanced LLM max_memory",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="sdpa",
        help="Attention backend; sdpa is faster than eager and usually needs less memory than checkpointing",
    )
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--seed", default=5254, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--accumulation_steps", default=4, type=int)
    parser.add_argument("--early_stop_patience", default=2, type=int)
    parser.add_argument("--word", default=40, type=int)
    parser.add_argument("--show_train_loss_steps", default=500, type=int)
    parser.add_argument("--max_profile_tokens", default=128, type=int)
    parser.add_argument("--max_target_item_tokens", default=64, type=int)
    parser.add_argument("--max_generation_prompt_tokens", default=20, type=int)
    parser.add_argument(
        "--item_description_mode",
        default="keywords",
        choices=["keywords", "full", "none"],
        help="How to include item description: keywords (default), full, or none.",
    )
    parser.add_argument("--lambda_feat", default=0.0001, type=float)
    parser.add_argument("--lambda_ul", default=0.05, type=float)
    parser.add_argument("--top_m_evidence", default=5, type=int)
    parser.add_argument("--ul_candidate_k", default=20, type=int)
    parser.add_argument("--ul_start_epoch", default=1, type=int)
    parser.add_argument("--evidence_bonus", default=0.1, type=float)
    parser.add_argument("--max_consecutive_token_repeat", default=3, type=int)
    parser.add_argument("--selector_hidden", default=256, type=int)
    parser.add_argument("--gnn_layers", default=2, type=int)
    parser.add_argument("--max_graph_nodes", default=512, type=int)
    parser.add_argument("--min_token_count", default=1, type=int)
    parser.add_argument(
        "--graph_cache_dir",
        default=str(PACKAGE_ROOT / "data" / "graph_cache"),
        type=str,
    )
    parser.add_argument(
        "--embedding_cache_dir",
        default=str(PACKAGE_ROOT / "checkpoints" / "embedding_cache"),
        type=str,
    )
    parser.add_argument(
        "--embedding_model_path",
        default=default_embedding_model_path(PACKAGE_ROOT),
        type=str,
    )
    parser.add_argument("--download_embedding_model", action="store_true")
    parser.add_argument("--rebuild_graph_cache", action="store_true")
    parser.add_argument("--special_token_ids", default="auto")
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj",
        type=str,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate per-dataset artifacts (dataset cache, graph cache, "
        "embedding cache, checkpoints, profiles) for the given fold before running. "
        "Implies fresh training; conflicts with --only_eval.",
    )
    parser.add_argument(
        "--max_eval_batches",
        default=0,
        type=int,
        help="If >0, run at most this many test batches during evaluation (quick smoke).",
    )
    parser.add_argument(
        "--max_train_batches",
        default=0,
        type=int,
        help="If >0, run at most this many train batches per epoch (quick smoke).",
    )
    parser.add_argument("--fold", "--split_indices", dest="split_indices", default="1", type=str)
    parser.add_argument(
        "--eval_tail_demand_groups",
        action="store_true",
        default=True,
        help="Evaluate metrics on all/low/high tail-demand groups (default: on)",
    )
    parser.add_argument(
        "--no_eval_tail_demand_groups",
        dest="eval_tail_demand_groups",
        action="store_false",
        help="Disable all/low/high tail-demand group evaluation",
    )
    parser.add_argument("--tail_low_percent", default=0.20, type=float)
    parser.add_argument("--tail_high_percent", default=0.20, type=float)
    parser.add_argument(
        "--dataset_name",
        "--dataset",
        default="Amazon/MoviesAndTV_corsa_filtered_small_15pct/",
        type=str,
    )
    parser.add_argument("--data_dir", default=str(PACKAGE_ROOT / "data"), type=str)
    parser.add_argument("--model_path", default=default_model_path(), type=str)
    parser.add_argument(
        "--profile_dir",
        default=str(PACKAGE_ROOT / "data" / "profiles"),
        type=str,
    )
    parser.add_argument("--allow_missing_profiles", action="store_true")
    parser.add_argument("--rebuild_dataset_cache", action="store_true")
    parser.add_argument(
        "--ckpt_dir",
        default=str(PACKAGE_ROOT / "checkpoints"),
        type=str,
    )
    parser.add_argument("--log_dir", default=str(PACKAGE_ROOT / "log"), type=str)
    parser.add_argument(
        "--output_dir",
        default=str(PACKAGE_ROOT / "log" / "output"),
        type=str,
    )
    parser.add_argument("--log_name", default="graph_profile.log", type=str)
    parser.add_argument(
        "--decode_strategy",
        default="greedy",
        choices=["greedy", "beam"],
        help="Test-time decoding strategy.",
    )
    parser.add_argument("--num_beams", default=4, type=int)
    parser.add_argument("--num_return_sequences", default=4, type=int)
    parser.add_argument(
        "--use_rerank",
        dest="use_rerank",
        action="store_true",
        default=False,
        help="Rerank beam candidates with metric-aware score.",
    )
    parser.add_argument(
        "--no_rerank",
        dest="use_rerank",
        action="store_false",
        help="Disable metric-aware rerank for beam decoding.",
    )
    parser.add_argument("--rerank_w_logprob", default=1.0, type=float)
    parser.add_argument("--rerank_w_feature", default=0.8, type=float)
    parser.add_argument("--rerank_w_evidence", default=0.5, type=float)
    parser.add_argument("--rerank_w_repetition", default=0.7, type=float)
    parser.add_argument("--rerank_w_generic", default=0.5, type=float)
    return parser
