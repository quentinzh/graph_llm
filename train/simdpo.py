"""训练集离线负采样与单 epoch SimDPO。"""

from __future__ import annotations

import math
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from graph_llm.dataload.dataloader import tokenizer_eos_id, tokenizer_pad_id
from graph_llm.metrics.metrics import ids2words, ids_clear


def _simdpo_precompute_batch_size(args) -> int:
    """SimDPO 离线阶段 batch；仅前向、无反向，可比训练 epoch 用更大 batch。"""
    return max(1, int(args.simdpo_precompute_batch_size))


def _feature_variants(tokenizer, feature: str) -> list[list[int]]:
    """兼容 feature 位于句首或普通词间时的两种 tokenizer 切分。"""
    variants, seen = [], set()
    for text in (str(feature).strip(), " " + str(feature).strip()):
        ids = tuple(int(x) for x in tokenizer(text, add_special_tokens=False)["input_ids"])
        if ids and ids not in seen:
            variants.append(list(ids))
            seen.add(ids)
    return variants


def feature_span_mask(token_rows, features, tokenizer, pad_id: int) -> torch.Tensor:
    """为一批解释构造 feature token mask；多个 self feature 会取并集。"""
    mask = torch.zeros((len(token_rows), max((len(x) for x in token_rows), default=0)), dtype=torch.bool)
    for row_idx, (ids, row_features) in enumerate(zip(token_rows, features)):
        if isinstance(row_features, str):
            row_features = [row_features]
        for feature in row_features:
            for variant in _feature_variants(tokenizer, feature):
                width = len(variant)
                for start in range(0, max(0, len(ids) - width + 1)):
                    if ids[start:start + width] == variant:
                        mask[row_idx, start:start + width] = True
        if pad_id in ids:
            first_pad = ids.index(pad_id)
            mask[row_idx, first_pad:] = False
    return mask


def _tokens_for_fmr(ids, tokenizer, pad_id: int, eos_id: int) -> list[str]:
    return ids2words(ids_clear(ids, pad_token_id=pad_id, eos_token_ids=(eos_id,)), tokenizer)


def _feature_sentence_records(train_dataset, tokenizer, pad_id: int, eos_id: int):
    """仅用训练集建立 feature→真实解释句索引与流行度。"""
    records = defaultdict(list)
    frequency = Counter()
    for local_idx, row in train_dataset.reset_index(drop=True).iterrows():
        feature = str(row["keyword_words"])
        ids = [int(x) for x in row["text"]]
        # 负句必须能按同一 FMR 规则检测到自己的 feature。
        if feature not in _tokens_for_fmr(ids, tokenizer, pad_id, eos_id):
            continue
        records[feature].append({
            "local_idx": int(local_idx),
            "ids": ids,
            "tokens": _tokens_for_fmr(ids, tokenizer, pad_id, eos_id),
        })
        frequency[feature] += 1
    return records, frequency


def _high_pop_features(frequency: Counter, fraction: float) -> list[str]:
    if not frequency:
        return []
    ordered = sorted(frequency, key=lambda feature: (-frequency[feature], feature))
    keep = max(1, math.ceil(len(ordered) * float(fraction)))
    return ordered[:keep]


def _choose_sentence(
    records,
    feature: str,
    current_idx: int,
    rng: random.Random,
    forbidden_feature: str | None = None,
):
    candidates = [
        row for row in records[feature]
        if row["local_idx"] != current_idx
        and (forbidden_feature is None or forbidden_feature not in row["tokens"])
    ]
    if not candidates:
        candidates = [
            row for row in records[feature]
            if forbidden_feature is None or forbidden_feature not in row["tokens"]
        ]
    return rng.choice(candidates) if candidates else None


@torch.no_grad()
def _generate_train_outputs(
    model,
    embedding_encoder,
    train_set,
    train_collater,
    tokenizer,
    args,
    device,
):
    """以与评估一致的 greedy decoding 生成训练集 self negative。"""
    # 延迟导入避免 trainer 与本模块在加载阶段循环依赖。
    from graph_llm.train.trainer import build_batch_prompt_tensors, compute_batch_selector_tensors, unpack_batch

    loader = DataLoader(
        train_set,
        batch_size=_simdpo_precompute_batch_size(args),
        collate_fn=train_collater,
        shuffle=False,
        num_workers=args.num_workers,
    )
    outputs = []
    model.eval()
    for batch in tqdm(loader, desc="SimDPO mine train outputs"):
        (
            _input_ids,
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
        evidence_ids, evidence_mask, _ = compute_batch_selector_tensors(
            model, embedding_encoder, graphs, graph_tensors, item_texts, tokenizer, args, device
        )
        prompt_ids, prompt_mask = build_batch_prompt_tensors(item_titles, raw_users, tokenizer, args, device)
        outputs.extend(model.greedy_generate(
            profile_ids, profile_mask, target_item_ids, target_item_mask,
            prompt_ids, prompt_mask, args.word, device,
            evidence_token_ids=evidence_ids, evidence_token_mask=evidence_mask,
        ))
    return outputs


def build_preference_groups(
    model,
    embedding_encoder,
    train_dataset,
    train_set,
    train_collater,
    tokenizer,
    args,
    device,
):
    """根据 SFT 是否严格命中训练集 gold feature 构建四负例 group。"""
    pad_id, eos_id = tokenizer_pad_id(tokenizer), tokenizer_eos_id(tokenizer)
    records, frequency = _feature_sentence_records(train_dataset, tokenizer, pad_id, eos_id)
    if len(records) < args.simdpo_num_negatives + 1:
        raise ValueError("SimDPO 训练 feature 数不足，无法为每个样本构造四个不同负例。")

    # Qwen3-0.6B 只在离线构造阶段编码训练 feature vocabulary；其参数始终冻结。
    features = sorted(records)
    vectors = embedding_encoder.encode_texts(features, batch_size=64).float().cpu()
    vectors = F.normalize(vectors, dim=1)
    feature_index = {feature: index for index, feature in enumerate(features)}
    similarity = vectors @ vectors.T
    high_pop = _high_pop_features(frequency, args.simdpo_high_pop_fraction)
    generated = _generate_train_outputs(
        model, embedding_encoder, train_set, train_collater, tokenizer, args, device
    )
    if len(generated) != len(train_dataset):
        raise RuntimeError("SimDPO 训练输出数与训练样本数不一致。")

    groups = []
    rng = random.Random(args.seed)
    train_frame = train_dataset.reset_index(drop=True)
    feature_vocab = set(features)
    for local_idx, row in tqdm(train_frame.iterrows(), total=len(train_frame), desc="Build SimDPO groups"):
        gold = str(row["keyword_words"])
        if gold not in records:
            continue
        generated_ids = [int(x) for x in generated[local_idx]]
        generated_tokens = _tokens_for_fmr(generated_ids, tokenizer, pad_id, eos_id)
        predicted_features = sorted(set(generated_tokens) & feature_vocab)
        hit = gold in predicted_features
        negatives = []

        if hit:
            candidates = [feature for feature in high_pop if feature != gold]
            if len(candidates) < args.simdpo_num_negatives:
                candidates = [feature for feature in features if feature != gold]
            chosen_features = rng.sample(candidates, k=args.simdpo_num_negatives)
            for feature in chosen_features:
                source = _choose_sentence(
                    records, feature, int(local_idx), rng, forbidden_feature=gold
                )
                if source is not None:
                    negatives.append({
                        "type": "popularity",
                        "features": [feature],
                        "ids": source["ids"],
                    })
        else:
            # self negative 必须保留模型完整输出，即使当前输出没有被词表检测到 feature。
            clean_generated = ids_clear(generated_ids, pad_token_id=pad_id, eos_token_ids=(eos_id,))
            negatives.append({
                "type": "self",
                "features": [feature for feature in predicted_features if feature != gold],
                "ids": clean_generated + [eos_id],
            })
            gold_idx = feature_index[gold]
            ranked = torch.argsort(similarity[gold_idx], descending=True).tolist()
            for candidate_idx in ranked:
                feature = features[candidate_idx]
                if feature == gold or any(feature in neg["features"] for neg in negatives):
                    continue
                source = _choose_sentence(
                    records, feature, int(local_idx), rng, forbidden_feature=gold
                )
                if source is None:
                    continue
                negatives.append({
                    "type": "semantic",
                    "features": [feature],
                    "ids": source["ids"],
                })
                if len(negatives) >= args.simdpo_num_negatives:
                    break

        if len(negatives) != args.simdpo_num_negatives:
            continue
        groups.append({
            "local_idx": int(local_idx),
            "gold_feature": gold,
            "negatives": negatives,
            "hit": bool(hit),
        })
    if not groups:
        raise RuntimeError("没有成功构造任何 SimDPO 偏好 group。")
    return groups


def _checkpoint_stamp(reference_adapter_path: str) -> int:
    adapter_dir = Path(reference_adapter_path)
    checkpoint_files = [path for path in adapter_dir.rglob("*") if path.is_file()] if adapter_dir.exists() else []
    return max((int(path.stat().st_mtime_ns) for path in checkpoint_files), default=0)


def _cache_dir(args, split_index: str, reference_adapter_path: str) -> Path:
    """将缓存绑定到产生 self negative 的 SFT adapter，避免重训后误用旧输出。"""
    dataset_name = str(args.dataset_name).strip("/").replace("/", "__")
    stamp = _checkpoint_stamp(reference_adapter_path)
    return Path(args.simdpo_cache_dir) / dataset_name / f"fold_{split_index}_sft_{stamp}"


def _groups_cache_path(args, split_index: str, reference_adapter_path: str) -> Path:
    base = _cache_dir(args, split_index, reference_adapter_path)
    return base.with_name(base.name + "_groups.pkl")


def _sidecar_cache_path(args, split_index: str, reference_adapter_path: str) -> Path:
    """evidence + reference log-prob 旁路缓存，与 preference groups 共用同一 SFT stamp。"""
    base = _cache_dir(args, split_index, reference_adapter_path)
    return base.with_name(base.name + "_ref_evidence.pkl")


def load_or_build_preference_groups(
    model,
    embedding_encoder,
    train_dataset,
    train_set,
    train_collater,
    tokenizer,
    args,
    device,
    split_index: str,
    reference_adapter_path: str,
):
    cache_path = _groups_cache_path(args, split_index, reference_adapter_path)
    if cache_path.exists() and not args.rebuild_simdpo_cache:
        with cache_path.open("rb") as handle:
            groups = pickle.load(handle)
        print(
            f"Reusing SimDPO preference groups cache (skip mining): "
            f"{cache_path} ({len(groups)} groups)"
        )
        return groups
    groups = build_preference_groups(
        model, embedding_encoder, train_dataset, train_set, train_collater, tokenizer, args, device
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(groups, handle)
    print(f"Saved SimDPO preference groups: {cache_path} ({len(groups)})")
    return groups


def _pad_rows(rows, pad_id: int, device: torch.device):
    width = max((len(row) for row in rows), default=1)
    out = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        out[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    return out


def _batch_evidence_from_cache(group_batch, device: torch.device):
    """从 group 上挂载的 CPU evidence 拼成 padded batch；不触发 selector。"""
    ids_rows = [group["_simdpo_cache"]["evidence_ids"] for group in group_batch]
    mask_rows = [group["_simdpo_cache"]["evidence_mask"] for group in group_batch]
    batch_size = len(group_batch)
    width = max((int(row.numel()) for row in ids_rows), default=0)
    if width == 0:
        return (
            torch.empty((batch_size, 0), dtype=torch.long, device=device),
            torch.empty((batch_size, 0), dtype=torch.bool, device=device),
        )
    evidence_ids = torch.zeros((batch_size, width), dtype=torch.long, device=device)
    evidence_mask = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
    for row_idx, (ids_row, mask_row) in enumerate(zip(ids_rows, mask_rows)):
        length = int(ids_row.numel())
        evidence_ids[row_idx, :length] = ids_row.to(device=device, dtype=torch.long)
        evidence_mask[row_idx, :length] = mask_row.to(device=device, dtype=torch.bool)
    return evidence_ids, evidence_mask


def _batch_evidence_from_map(local_indices, evidence_map, device: torch.device):
    """预计算阶段：按 local_idx 列表从 evidence_map 拼 batch。"""
    ids_rows = [evidence_map[idx]["ids"] for idx in local_indices]
    mask_rows = [evidence_map[idx]["mask"] for idx in local_indices]
    batch_size = len(local_indices)
    width = max((int(row.numel()) for row in ids_rows), default=0)
    if width == 0:
        return (
            torch.empty((batch_size, 0), dtype=torch.long, device=device),
            torch.empty((batch_size, 0), dtype=torch.bool, device=device),
        )
    evidence_ids = torch.zeros((batch_size, width), dtype=torch.long, device=device)
    evidence_mask = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
    for row_idx, (ids_row, mask_row) in enumerate(zip(ids_rows, mask_rows)):
        length = int(ids_row.numel())
        evidence_ids[row_idx, :length] = ids_row.to(device=device, dtype=torch.long)
        evidence_mask[row_idx, :length] = mask_row.to(device=device, dtype=torch.bool)
    return evidence_ids, evidence_mask


def _dpo_term(policy_pos, policy_neg, ref_pos, ref_neg, beta: float):
    margin = (policy_pos - ref_pos) - (policy_neg - ref_neg)
    return -F.logsigmoid(float(beta) * margin)


def _freeze_graph_modules(model):
    """纯 SimDPO 下 hard top-M selector 没有可用的生成梯度，因此显式冻结。"""
    for parameter in model.evidence_selector.parameters():
        parameter.requires_grad = False
    model.evidence_selector.eval()


def _offload_embedding_encoder(embedding_encoder):
    """SimDPO 训练 epoch 不再编码文本，暂把 embedding 挪到 CPU 腾显存。"""
    if embedding_encoder is None or not hasattr(embedding_encoder, "model"):
        return None
    model = embedding_encoder.model
    if model is None:
        return None
    try:
        embed_device = next(model.parameters()).device
    except StopIteration:
        return None
    if embed_device.type == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()
    return embed_device


def _restore_embedding_encoder(embedding_encoder, embed_device):
    if (
        embedding_encoder is None
        or embed_device is None
        or embed_device.type != "cuda"
        or not hasattr(embedding_encoder, "model")
        or embedding_encoder.model is None
    ):
        return
    embedding_encoder.model.to(embed_device)
    torch.cuda.empty_cache()


def _load_policy_adapter(model, adapter_path: str):
    if "simdpo" not in getattr(model.model, "peft_config", {}):
        model.model.load_adapter(adapter_path, adapter_name="simdpo", is_trainable=True)
    for name, parameter in model.model.named_parameters():
        parameter.requires_grad = "simdpo" in name
    model.model.set_adapter("simdpo")


def _sidecar_is_valid(sidecar, groups) -> bool:
    """校验旁路缓存是否与当前 preference groups 对齐。"""
    if not isinstance(sidecar, dict):
        return False
    evidence = sidecar.get("evidence")
    reference = sidecar.get("reference")
    if not isinstance(evidence, dict) or not isinstance(reference, list):
        return False
    if len(reference) != len(groups):
        return False
    for group, ref in zip(groups, reference):
        local_idx = int(group["local_idx"])
        if local_idx not in evidence:
            return False
        neg = ref.get("neg")
        if not isinstance(neg, list) or len(neg) != len(group["negatives"]):
            return False
    return True


@torch.no_grad()
def precompute_evidence_and_reference(
    model,
    embedding_encoder,
    groups,
    train_set,
    train_collater,
    tokenizer,
    args,
    device,
):
    """离线缓存 frozen selector evidence 与 best_lora reference log-prob（float32）。

    预计算使用 simdpo_precompute_batch_size（默认 16，仅前向）；训练仍按 simdpo_batch_size。
    """
    from graph_llm.train.trainer import build_batch_prompt_tensors, compute_batch_selector_tensors, unpack_batch

    pad_id = tokenizer_pad_id(tokenizer)
    model.eval()
    model.evidence_selector.eval()
    model.model.set_adapter("best_lora")

    # --- 方案 3：按唯一 local_idx 缓存 evidence ---
    unique_indices = sorted({int(group["local_idx"]) for group in groups})
    evidence_map = {}
    precompute_bs = _simdpo_precompute_batch_size(args)
    for start in tqdm(range(0, len(unique_indices), precompute_bs), desc="SimDPO cache evidence"):
        chunk = unique_indices[start:start + precompute_bs]
        batch = train_collater([train_set[idx] for idx in chunk])
        (
            _positive_ids,
            _rating,
            _profile_ids,
            _profile_mask,
            _target_item_ids,
            _target_item_mask,
            graph_tensors,
            graphs,
            item_texts,
            _item_titles,
            _raw_users,
            _feature_position_mask,
            _feature_position_weights,
        ) = unpack_batch(batch, device)
        evidence_ids, evidence_mask, _ = compute_batch_selector_tensors(
            model, embedding_encoder, graphs, graph_tensors, item_texts, tokenizer, args, device
        )
        for row_idx, local_idx in enumerate(chunk):
            evidence_map[int(local_idx)] = {
                "ids": evidence_ids[row_idx].detach().cpu().contiguous().clone(),
                "mask": evidence_mask[row_idx].detach().cpu().contiguous().clone(),
            }

    # --- 方案 2：best_lora 上预计算 pos/neg reference 分数 ---
    references = []
    num_negatives = int(args.simdpo_num_negatives)
    for start in tqdm(range(0, len(groups), precompute_bs), desc="SimDPO cache reference"):
        group_batch = groups[start:start + precompute_bs]
        batch = train_collater([train_set[group["local_idx"]] for group in group_batch])
        (
            positive_ids,
            _rating,
            profile_ids,
            profile_mask,
            target_item_ids,
            target_item_mask,
            _graph_tensors,
            _graphs,
            _item_texts,
            item_titles,
            raw_users,
            _feature_position_mask,
            _feature_position_weights,
        ) = unpack_batch(batch, device)
        evidence_ids, evidence_mask = _batch_evidence_from_map(
            [int(group["local_idx"]) for group in group_batch], evidence_map, device
        )
        prompt_ids, prompt_mask = build_batch_prompt_tensors(
            item_titles, raw_users, tokenizer, args, device
        )
        positive_rows_ids = positive_ids.detach().cpu().tolist()
        positive_feature_mask = feature_span_mask(
            positive_rows_ids, [group["gold_feature"] for group in group_batch], tokenizer, pad_id
        ).to(device)
        ref_pos_sent, ref_pos_feat, ref_pos_valid = model.sequence_log_probs(
            positive_ids, profile_ids, profile_mask, target_item_ids, target_item_mask,
            prompt_ids, prompt_mask, evidence_ids, evidence_mask, positive_feature_mask,
        )
        # 每个 group 收集四个负例的 ref；仍按负例下标串行，避免一次性堆五条 response。
        neg_refs = [[] for _ in group_batch]
        for negative_index in range(num_negatives):
            negative_rows = [group["negatives"][negative_index]["ids"] for group in group_batch]
            negative_features = [group["negatives"][negative_index]["features"] for group in group_batch]
            negative_ids = _pad_rows(negative_rows, pad_id, device)
            negative_feature_mask = feature_span_mask(
                negative_rows, negative_features, tokenizer, pad_id
            ).to(device)
            ref_neg_sent, ref_neg_feat, ref_neg_valid = model.sequence_log_probs(
                negative_ids, profile_ids, profile_mask, target_item_ids, target_item_mask,
                prompt_ids, prompt_mask, evidence_ids, evidence_mask, negative_feature_mask,
            )
            for row_idx in range(len(group_batch)):
                neg_refs[row_idx].append({
                    "sent": float(ref_neg_sent[row_idx].item()),
                    "feat": float(ref_neg_feat[row_idx].item()),
                    "valid": bool(ref_neg_valid[row_idx].item()),
                })
        for row_idx, _group in enumerate(group_batch):
            references.append({
                "pos_sent": float(ref_pos_sent[row_idx].item()),
                "pos_feat": float(ref_pos_feat[row_idx].item()),
                "pos_valid": bool(ref_pos_valid[row_idx].item()),
                "neg": neg_refs[row_idx],
            })

    return {"evidence": evidence_map, "reference": references}


def load_or_build_ref_evidence_cache(
    model,
    embedding_encoder,
    groups,
    train_set,
    train_collater,
    tokenizer,
    args,
    device,
    split_index: str,
    reference_adapter_path: str,
):
    """加载或构建 evidence + reference 旁路缓存。"""
    cache_path = _sidecar_cache_path(args, split_index, reference_adapter_path)
    if cache_path.exists() and not args.rebuild_simdpo_cache:
        with cache_path.open("rb") as handle:
            sidecar = pickle.load(handle)
        if _sidecar_is_valid(sidecar, groups):
            print(
                f"Reusing SimDPO ref/evidence cache (skip precompute): {cache_path} "
                f"(groups={len(groups)}, evidence={len(sidecar['evidence'])})"
            )
            return sidecar
        print(f"SimDPO ref/evidence cache mismatch, rebuilding: {cache_path}")

    sidecar = precompute_evidence_and_reference(
        model, embedding_encoder, groups, train_set, train_collater, tokenizer, args, device
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(sidecar, handle)
    print(
        f"Saved SimDPO ref/evidence cache: {cache_path} "
        f"(groups={len(groups)}, evidence={len(sidecar['evidence'])})"
    )
    return sidecar


def _attach_sidecar_to_groups(groups, sidecar):
    """把旁路缓存挂到每个 group 上，shuffle 后仍可直接取用。"""
    for group, ref in zip(groups, sidecar["reference"]):
        local_idx = int(group["local_idx"])
        evidence = sidecar["evidence"][local_idx]
        group["_simdpo_cache"] = {
            "evidence_ids": evidence["ids"],
            "evidence_mask": evidence["mask"],
            "ref_pos_sent": float(ref["pos_sent"]),
            "ref_pos_feat": float(ref["pos_feat"]),
            "ref_pos_valid": bool(ref["pos_valid"]),
            "ref_neg": list(ref["neg"]),
        }


def run_simdpo_stage(
    model,
    embedding_encoder,
    train_dataset,
    train_set,
    train_collater,
    tokenizer,
    args,
    device,
    cuda_devices,
    ckpt_prefix: str,
    log_name: str,
    split_index: str,
):
    """执行一次纯 SimDPO epoch；返回 active 的 SimDPO policy。"""
    from graph_llm.train.trainer import build_batch_prompt_tensors, unpack_batch

    reference_adapter_path = ckpt_prefix + "model"
    _freeze_graph_modules(model)
    model.model.set_adapter("best_lora")
    groups = load_or_build_preference_groups(
        model, embedding_encoder, train_dataset, train_set, train_collater,
        tokenizer, args, device, split_index, reference_adapter_path,
    )
    # 方案 2+3：离线 reference / evidence；训练阶段不再跑 best_lora 与 selector。
    sidecar = load_or_build_ref_evidence_cache(
        model, embedding_encoder, groups, train_set, train_collater,
        tokenizer, args, device, split_index, reference_adapter_path,
    )
    _attach_sidecar_to_groups(groups, sidecar)

    _load_policy_adapter(model, reference_adapter_path)
    optimizer = AdamW(
        [parameter for parameter in model.model.parameters() if parameter.requires_grad],
        lr=args.learning_rate / 10,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    # 训练阶段 evidence/reference 均已缓存，embedding 可暂离 GPU。
    embed_restore_device = _offload_embedding_encoder(embedding_encoder)
    group_loader = DataLoader(
        groups,
        batch_size=args.simdpo_batch_size,
        shuffle=True,
        collate_fn=lambda items: items,
        num_workers=0,
    )
    pad_id = tokenizer_pad_id(tokenizer)
    num_negatives = int(args.simdpo_num_negatives)
    model.train()
    model.evidence_selector.eval()
    optimizer.zero_grad(set_to_none=True)
    losses, feature_losses, sentence_losses = [], [], []

    for batch_idx, group_batch in enumerate(tqdm(group_loader, desc="SimDPO epoch 0")):
        positive_rows = [train_set[group["local_idx"]] for group in group_batch]
        batch = train_collater(positive_rows)
        (
            positive_ids,
            _rating,
            profile_ids,
            profile_mask,
            target_item_ids,
            target_item_mask,
            _graph_tensors,
            _graphs,
            _item_texts,
            item_titles,
            raw_users,
            _feature_position_mask,
            _feature_position_weights,
        ) = unpack_batch(batch, device)
        # 方案 3：直接读缓存 evidence，跳过 compute_batch_selector_tensors。
        evidence_ids, evidence_mask = _batch_evidence_from_cache(group_batch, device)
        prompt_ids, prompt_mask = build_batch_prompt_tensors(
            item_titles, raw_users, tokenizer, args, device
        )
        positive_rows_ids = positive_ids.detach().cpu().tolist()
        positive_feature_mask = feature_span_mask(
            positive_rows_ids, [group["gold_feature"] for group in group_batch], tokenizer, pad_id
        ).to(device)

        # 方案 2：reference 分数来自旁路缓存（float32），训练只跑 policy。
        ref_pos_sent = torch.tensor(
            [group["_simdpo_cache"]["ref_pos_sent"] for group in group_batch],
            dtype=torch.float32, device=device,
        )
        ref_pos_feat = torch.tensor(
            [group["_simdpo_cache"]["ref_pos_feat"] for group in group_batch],
            dtype=torch.float32, device=device,
        )
        ref_pos_valid = torch.tensor(
            [group["_simdpo_cache"]["ref_pos_valid"] for group in group_batch],
            dtype=torch.bool, device=device,
        )

        model.train()
        model.evidence_selector.eval()
        model.model.set_adapter("simdpo")
        # 方案 1：正例只前向一次；负例串行 forward+backward，并用 retain_graph 复用 pos 图，
        # 峰值仍约为「1×pos + 1×neg」，避免一次堆叠四条 neg 计算图抬高显存。
        with autocast(enabled=device.type == "cuda"):
            policy_pos_sent, policy_pos_feat, policy_pos_valid = model.sequence_log_probs(
                positive_ids, profile_ids, profile_mask, target_item_ids, target_item_mask,
                prompt_ids, prompt_mask, evidence_ids, evidence_mask, positive_feature_mask,
            )

        for negative_index in range(num_negatives):
            negative_rows = [group["negatives"][negative_index]["ids"] for group in group_batch]
            negative_features = [
                group["negatives"][negative_index]["features"] for group in group_batch
            ]
            negative_ids = _pad_rows(negative_rows, pad_id, device)
            negative_feature_mask = feature_span_mask(
                negative_rows, negative_features, tokenizer, pad_id
            ).to(device)
            ref_neg_sent = torch.tensor(
                [group["_simdpo_cache"]["ref_neg"][negative_index]["sent"] for group in group_batch],
                dtype=torch.float32, device=device,
            )
            ref_neg_feat = torch.tensor(
                [group["_simdpo_cache"]["ref_neg"][negative_index]["feat"] for group in group_batch],
                dtype=torch.float32, device=device,
            )
            ref_neg_valid = torch.tensor(
                [group["_simdpo_cache"]["ref_neg"][negative_index]["valid"] for group in group_batch],
                dtype=torch.bool, device=device,
            )
            with autocast(enabled=device.type == "cuda"):
                policy_neg_sent, policy_neg_feat, policy_neg_valid = model.sequence_log_probs(
                    negative_ids, profile_ids, profile_mask, target_item_ids, target_item_mask,
                    prompt_ids, prompt_mask, evidence_ids, evidence_mask, negative_feature_mask,
                )
                sentence_loss = _dpo_term(
                    policy_pos_sent, policy_neg_sent, ref_pos_sent, ref_neg_sent,
                    args.simdpo_beta_sentence,
                ).mean()
                feature_valid = (
                    policy_pos_valid & policy_neg_valid & ref_pos_valid & ref_neg_valid
                )
                if feature_valid.any():
                    feature_loss = _dpo_term(
                        policy_pos_feat[feature_valid], policy_neg_feat[feature_valid],
                        ref_pos_feat[feature_valid], ref_neg_feat[feature_valid],
                        args.simdpo_beta_feature,
                    ).mean()
                else:
                    feature_loss = sentence_loss.new_zeros(())
                loss = (
                    float(args.simdpo_lambda_sentence) * sentence_loss
                    + float(args.simdpo_lambda_feature) * feature_loss
                ) / float(num_negatives)
            # 前 N-1 个负例保留 pos 计算图；最后一个释放。
            scaler.scale(loss).backward(retain_graph=(negative_index + 1) < num_negatives)
            losses.append(float(loss.detach().item()) * num_negatives)
            feature_losses.append(float(feature_loss.detach().item()))
            sentence_losses.append(float(sentence_loss.detach().item()))

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.model.parameters() if parameter.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if args.max_train_batches and (batch_idx + 1) >= args.max_train_batches:
            print(
                "Stopping SimDPO early after "
                f"{args.max_train_batches} batches (--max_train_batches)."
            )
            break

    _restore_embedding_encoder(embedding_encoder, embed_restore_device)
    model.model.set_adapter("simdpo")
    output_path = ckpt_prefix + "simdpo_model"
    model.model.save_pretrained(output_path, selected_adapters=["simdpo"])
    message = (
        "SimDPO epoch complete | groups={} | loss={:.6f} | feature={:.6f} | sentence={:.6f} | "
        "selector_graph=frozen | ref_evidence_cache=on | adapter={}"
    ).format(
        len(groups),
        sum(losses) / max(len(losses), 1),
        sum(feature_losses) / max(len(feature_losses), 1),
        sum(sentence_losses) / max(len(sentence_losses), 1),
        output_path,
    )
    print(message)
    with open(log_name, "a+", encoding="utf-8") as handle:
        handle.write(message + "\n")
    return output_path
