"""graph_llm Semantic ID 推荐参数。"""

from __future__ import annotations

import argparse
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent


def _is_local_model_dir(path: Path | str) -> bool:
    path = Path(path)
    return path.is_dir() and (path / "config.json").is_file()


def default_roberta_model_path() -> str:
    local = PACKAGE_ROOT / "pretrain_llm" / "roberta-base"
    if _is_local_model_dir(local):
        return str(local.resolve())
    return str(local)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="graph_llm Semantic ID 推荐（GNN + SID）")
    parser.add_argument("--device", "--devices", dest="devices", default="1", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--seed", default=5254, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--early_stop_patience", default=5, type=int)
    parser.add_argument("--grad_clip_norm", default=1.0, type=float)
    parser.add_argument(
        "--rec_loss",
        choices=["item", "sid", "both"],
        default="item",
        help="推荐损失：item=商品级 full-softmax（默认）；sid=逐位置 CE；both=消融用",
    )
    parser.add_argument("--rec_temperature", default=4.0, type=float, help="L_rec 全商品 softmax 温度 τ")
    parser.add_argument(
        "--score_pmi_lambda",
        default=0.0,
        type=float,
        help="推理时 PMI 校正强度 λ；0 表示不校正",
    )
    parser.add_argument(
        "--auto_pmi_lambda",
        action="store_true",
        help="验证集上搜索 PMI λ，仅当 NDCG@10 不低于 λ=0 时采用更优 λ",
    )
    parser.add_argument(
        "--eval_baseline",
        choices=["model", "popularity", "random"],
        default="model",
        help="评估打分来源：model=训练模型；popularity/random=基线校验",
    )
    parser.add_argument(
        "--mode",
        choices=["recommend", "explain"],
        default="recommend",
        help="recommend=SID 推荐；explain=解释生成（需已有推荐 checkpoint）",
    )
    parser.add_argument("--selector_hidden", default=256, type=int)
    parser.add_argument("--gnn_layers", default=2, type=int)
    parser.add_argument("--magnet_q", default=0.15, type=float)
    parser.add_argument("--max_graph_nodes", default=512, type=int)
    parser.add_argument(
        "--embedding_cache_dir",
        default=str(PACKAGE_ROOT / "checkpoints" / "embedding_cache"),
        type=str,
    )
    parser.add_argument("--roberta_encode_batch_size", default=64, type=int)
    parser.add_argument(
        "--max_eval_batches",
        default=0,
        type=int,
        help=">0 时限制验证/测试 batch 数（smoke）",
    )
    parser.add_argument(
        "--max_train_batches",
        default=0,
        type=int,
        help=">0 时限制每 epoch 训练 batch 数（smoke）",
    )
    parser.add_argument(
        "--dataset_name",
        "--dataset",
        default="Instruments",
        type=str,
    )
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"), type=str)
    parser.add_argument("--dataset_format", default="", type=str)
    parser.add_argument(
        "--roberta_model_path",
        default=default_roberta_model_path(),
        type=str,
    )
    parser.add_argument("--text_sid_length", default=4, type=int)
    parser.add_argument("--text_codebook_size", default=256, type=int)
    parser.add_argument("--smoke_codebook_size", default=16, type=int)
    parser.add_argument("--quant_seed", default=5254, type=int)
    parser.add_argument("--use_rating_sid", action="store_true", default=True)
    parser.add_argument("--no_rating_sid", dest="use_rating_sid", action="store_false")
    parser.add_argument("--use_popularity_sid", action="store_true", default=True)
    parser.add_argument("--no_popularity_sid", dest="use_popularity_sid", action="store_false")
    parser.add_argument("--rating_min", default=1.0, type=float)
    parser.add_argument("--rating_max", default=5.0, type=float)
    parser.add_argument("--rating_smooth_alpha", default=10.0, type=float)
    parser.add_argument("--calib_ratio", default=0.2, type=float)
    parser.add_argument("--lambda_rating", default=0.2, type=float)
    parser.add_argument("--lambda_pop", default=0.1, type=float)
    parser.add_argument(
        "--search_mode",
        choices=["exact", "graph"],
        default="graph",
    )
    parser.add_argument("--search_neighbors", default=32, type=int)
    parser.add_argument("--search_rounds", default=3, type=int)
    parser.add_argument("--search_candidates", default=256, type=int)
    parser.add_argument("--allow_repeat_recommend", action="store_true", default=False)
    parser.add_argument("--positive_feedback_threshold", default=0.0, type=float)
    parser.add_argument(
        "--sid_cache_dir",
        default=str(PACKAGE_ROOT / "data" / "sid_cache"),
        type=str,
    )
    parser.add_argument("--rebuild_sid_cache", action="store_true")
    parser.add_argument("--smoke_device", default="", type=str)
    parser.add_argument(
        "--smoke_mock_encoder",
        action="store_true",
        help="仅调试：用哈希向量代替 RoBERTa",
    )
    parser.add_argument(
        "--ckpt_dir",
        default=str(PACKAGE_ROOT / "checkpoints"),
        type=str,
    )
    parser.add_argument("--log_dir", default=str(PACKAGE_ROOT / "log"), type=str)
    # --- 解释阶段 ---
    parser.add_argument(
        "--llm_model_path",
        default=str(PACKAGE_ROOT / "pretrain_llm" / "qwen3-4b"),
        type=str,
    )
    parser.add_argument("--explain_epochs", default=5, type=int)
    parser.add_argument("--explain_early_stop_patience", default=2, type=int)
    parser.add_argument("--lambda_selector", default=0.1, type=float)
    parser.add_argument("--feature_gamma", default=2.0, type=float, help="feature 加权 CE 系数 γ")
    parser.add_argument("--tail_alpha", default=0.5, type=float)
    parser.add_argument("--tail_weight_min", default=0.5, type=float)
    parser.add_argument("--tail_weight_max", default=2.0, type=float)
    parser.add_argument("--top_m_evidence", default=5, type=int)
    parser.add_argument("--max_generation_tokens", default=40, type=int)
    parser.add_argument("--gen_temperature", default=0.9, type=float)
    parser.add_argument("--gen_top_p", default=0.92, type=float)
    parser.add_argument("--gen_repetition_penalty", default=1.15, type=float)
    parser.add_argument("--explain_unfreeze_shared", action="store_true", default=False)
    parser.add_argument(
        "--joint_explain_rec",
        action="store_true",
        help="联合微调：解释损失回传共享 GNN/selector（默认分阶段冻结推荐）",
    )
    parser.add_argument(
        "--recommend_ckpt",
        default="",
        type=str,
        help="解释阶段加载的推荐 checkpoint 路径；空则使用 ckpt_dir/.../recommend/sid_recommender.bin",
    )
    return parser


def default_qwen_model_path() -> str:
    local = PACKAGE_ROOT / "pretrain_llm" / "qwen3-4b"
    if _is_local_model_dir(local):
        return str(local.resolve())
    return str(local)
