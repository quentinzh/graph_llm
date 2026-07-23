"""SFT 后同一 LoRA 上的语境–解释表征 InfoNCE（干净实现，峰值接近单次 LM 前向）。

设计要点：
1. 正例一次前向同时得到 z_x（context 段）与 z_p（response 段），无全词表 CE。
2. 硬负例在 no_grad 下编码，不进入计算图，避免 SimDPO 式双图 / retain_graph。
3. 默认 batch=8，与 SFT 同量级；可选 in-batch 其它 gold 作额外负例。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from graph_llm.dataload.dataloader import tokenizer_pad_id
from graph_llm.train.simdpo import (
    _freeze_graph_modules,
    _load_policy_adapter,
    _offload_embedding_encoder,
    _pad_rows,
    _restore_embedding_encoder,
    load_or_build_preference_groups,
)


def _infonce_logits(z_x, z_p, z_hard_neg, *, tau: float, in_batch: bool):
    """构造 InfoNCE logits：列 0 为正例，其后为硬负例与可选 in-batch 负例。"""
    # z_x, z_p: [B,H]；z_hard_neg: [B,K,H] 或空
    pos = (z_x * z_p).sum(dim=-1, keepdim=True) / float(tau)
    parts = [pos]
    if z_hard_neg is not None and z_hard_neg.numel() > 0:
        hard = torch.einsum("bh,bkh->bk", z_x, z_hard_neg) / float(tau)
        parts.append(hard)
    if in_batch and z_p.shape[0] > 1:
        # in-batch：其它样本的 z_p；对角为正例已在列 0，这里去掉对角避免重复
        sim = (z_x @ z_p.T) / float(tau)  # [B,B]
        eye = torch.eye(z_p.shape[0], dtype=torch.bool, device=z_p.device)
        # 将对角置为极小，使 CE 不会选到「自己当负例」之外的重复正列
        sim = sim.masked_fill(eye, torch.finfo(sim.dtype).min)
        parts.append(sim)
    return torch.cat(parts, dim=1)


def run_repr_contrast_stage(
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
    """执行一次表征 InfoNCE epoch；返回保存的 adapter 路径。"""
    from graph_llm.train.trainer import build_batch_prompt_tensors, unpack_batch

    reference_adapter_path = ckpt_prefix + "model"
    _freeze_graph_modules(model)
    model.model.set_adapter("best_lora")
    # 复用 SimDPO 的 preference groups（正/负句挖掘与缓存）
    groups = load_or_build_preference_groups(
        model, embedding_encoder, train_dataset, train_set, train_collater,
        tokenizer, args, device, split_index, reference_adapter_path,
    )

    _load_policy_adapter(model, reference_adapter_path)
    # 训练阶段不需要 embedding / selector，腾出显存
    embed_restore_device = _offload_embedding_encoder(embedding_encoder)

    optimizer = AdamW(
        [parameter for parameter in model.model.parameters() if parameter.requires_grad],
        lr=args.learning_rate / 10,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    batch_size = max(1, int(args.repr_contrast_batch_size))
    group_loader = DataLoader(
        groups,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda items: items,
        num_workers=0,
    )
    pad_id = tokenizer_pad_id(tokenizer)
    tau = float(args.repr_contrast_temperature)
    in_batch = bool(args.repr_contrast_in_batch_negatives)
    num_negatives = int(args.simdpo_num_negatives)

    model.train()
    model.evidence_selector.eval()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    for batch_idx, group_batch in enumerate(
        tqdm(group_loader, desc="ReprContrast InfoNCE epoch 0")
    ):
        if device.type == "cuda" and batch_idx < 2:
            torch.cuda.reset_peak_memory_stats(device)
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
        prompt_ids, prompt_mask = build_batch_prompt_tensors(
            item_titles, raw_users, tokenizer, args, device
        )

        model.train()
        model.evidence_selector.eval()
        model.model.set_adapter("simdpo")

        # --- 正例：一次前向 → z_x + z_p（唯一需要反传的大图）---
        with autocast(enabled=device.type == "cuda"):
            z_x, z_p = model.pooled_context_and_response_repr(
                positive_ids,
                profile_ids=profile_ids,
                profile_mask=profile_mask,
                target_item_ids=target_item_ids,
                target_item_mask=target_item_mask,
                generation_prompt_ids=prompt_ids,
                generation_prompt_mask=prompt_mask,
            )

        # --- 硬负例：no_grad 编码，不抬峰值 ---
        hard_neg_list = []
        with torch.no_grad():
            for negative_index in range(num_negatives):
                negative_rows = [
                    group["negatives"][negative_index]["ids"] for group in group_batch
                ]
                negative_ids = _pad_rows(negative_rows, pad_id, device)
                with autocast(enabled=device.type == "cuda"):
                    _z_nx, z_nk = model.pooled_context_and_response_repr(
                        negative_ids,
                        profile_ids=profile_ids,
                        profile_mask=profile_mask,
                        target_item_ids=target_item_ids,
                        target_item_mask=target_item_mask,
                        generation_prompt_ids=prompt_ids,
                        generation_prompt_mask=prompt_mask,
                    )
                hard_neg_list.append(z_nk.float())
        z_hard = (
            torch.stack(hard_neg_list, dim=1)
            if hard_neg_list
            else torch.empty(
                (len(group_batch), 0, z_p.shape[-1]),
                device=device,
                dtype=torch.float32,
            )
        )

        with autocast(enabled=device.type == "cuda"):
            logits = _infonce_logits(
                z_x, z_p, z_hard, tau=tau, in_batch=in_batch
            )
            targets = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits.float(), targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.model.parameters() if parameter.requires_grad],
            1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().item()))

        if device.type == "cuda" and (batch_idx + 1) <= 2:
            # smoke / 排查用：打印前两个 batch 的显存峰值
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            print(
                f"ReprContrast CUDA peak after batch {batch_idx + 1}: "
                f"allocated={peak:.2f} GiB reserved={reserved:.2f} GiB"
            )
            torch.cuda.reset_peak_memory_stats(device)

        if args.max_train_batches and (batch_idx + 1) >= args.max_train_batches:
            print(
                "Stopping ReprContrast early after "
                f"{args.max_train_batches} batches (--max_train_batches)."
            )
            break

    _restore_embedding_encoder(embedding_encoder, embed_restore_device)
    model.model.set_adapter("simdpo")
    output_path = ckpt_prefix + "repr_contrast_model"
    model.model.save_pretrained(output_path, selected_adapters=["simdpo"])
    message = (
        "ReprContrast epoch complete | groups={} | loss={:.6f} | "
        "batch_size={} | tau={} | in_batch_neg={} | hard_neg={} | "
        "selector_graph=frozen | adapter={}"
    ).format(
        len(groups),
        sum(losses) / max(len(losses), 1),
        batch_size,
        tau,
        in_batch,
        num_negatives,
        output_path,
    )
    print(message)
    with open(log_name, "a+", encoding="utf-8") as handle:
        handle.write(message + "\n")
    return output_path
