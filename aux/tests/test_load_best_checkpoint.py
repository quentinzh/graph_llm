"""回归：load_best_checkpoint 必须把磁盘 LoRA 真正写回，且层内挂载完整。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
from transformers import GPT2Config, GPT2LMHeadModel

from graph_llm.train.trainer import (
    _adapter_attached_to_all_lora_layers,
    load_best_checkpoint,
)


def _tiny_peft_model():
    cfg = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_inner=64,
    )
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(GPT2LMHeadModel(cfg), lora_config)


def test_load_best_checkpoint_restores_default_and_best_lora():
    """模拟：训练后权重变化 → save → 继续训练污染 default → reload 应回到 save 点。"""
    peft_model = _tiny_peft_model()
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_A.default" in name:
                param.add_(0.25)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_prefix = str(Path(tmp) / "1")
        adapter_dir = ckpt_prefix + "model"
        peft_model.save_pretrained(adapter_dir)
        saved = get_peft_model_state_dict(peft_model, adapter_name="default")
        saved_key = next(iter(saved))
        saved_norm = saved[saved_key].float().norm().item()

        # 继续“训练”，污染 default
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if "lora_A.default" in name:
                    param.add_(3.0)
        polluted = get_peft_model_state_dict(peft_model, adapter_name="default")
        assert polluted[saved_key].float().norm().item() != pytest.approx(saved_norm, abs=1e-4)

        wrapper = SimpleNamespace(
            model=peft_model,
            evidence_selector=None,
            user_review_projector=None,
            item_review_projector=None,
            eval=lambda: None,
        )
        # selector / review_prefix 不存在时应跳过（projector 均为 None）
        load_best_checkpoint(wrapper, ckpt_prefix, device=torch.device("cpu"))

        assert "best_lora" in peft_model.peft_config
        assert _adapter_attached_to_all_lora_layers(peft_model, "best_lora")
        assert _adapter_attached_to_all_lora_layers(peft_model, "default")

        restored_default = get_peft_model_state_dict(peft_model, adapter_name="default")
        restored_best = get_peft_model_state_dict(peft_model, adapter_name="best_lora")
        assert restored_default[saved_key].float().norm().item() == pytest.approx(
            saved_norm, abs=1e-4
        )
        assert restored_best[saved_key].float().norm().item() == pytest.approx(
            saved_norm, abs=1e-4
        )
        assert peft_model.active_adapter in {"best_lora", "default"}
