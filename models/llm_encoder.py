"""Qwen3 + LoRA 用户历史文本编码（推荐 rec / 解释 exp 双 adapter）。"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


REC_ADAPTER = "rec"
EXP_ADAPTER = "exp"


def resolve_local_adapter_dir(adapter_dir: Path, adapter_name: str) -> Path:
    """定位含 adapter_config.json 的本地目录。

    PEFT 保存非 default adapter 时会写到 ``<save_dir>/<adapter_name>/``，
    直接把父目录传给 load_adapter 会被当成 Hugging Face repo id。
    """
    adapter_dir = Path(adapter_dir)
    if (adapter_dir / "adapter_config.json").is_file():
        return adapter_dir
    nested = adapter_dir / adapter_name
    if (nested / "adapter_config.json").is_file():
        return nested
    if adapter_dir.is_dir():
        for child in adapter_dir.iterdir():
            if child.is_dir() and (child / "adapter_config.json").is_file():
                return child
    raise FileNotFoundError(f"未找到本地 LoRA adapter: {adapter_dir} (adapter={adapter_name})")


class QwenLoRAEncoder(nn.Module):
    """左 padding 序列，取最后有效 token 的 hidden 作为用户文本表示。"""

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        *,
        adapter_name: str = REC_ADAPTER,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = False,
        local_files_only: bool = True,
        load_in_4bit: bool = False,
    ):
        super().__init__()
        self.device = device
        self.adapter_name = adapter_name
        path = Path(model_path)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=local_files_only,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModel.from_pretrained(
            path,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        # 先挂 rec adapter；exp 由 load_dual_adapters 追加
        if adapter_name == REC_ADAPTER:
            self.model = get_peft_model(base, lora_cfg, adapter_name=REC_ADAPTER)
        else:
            self.model = get_peft_model(base, lora_cfg, adapter_name=adapter_name)

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(device)
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 2560))

    @property
    def peft_model(self) -> PeftModel:
        return self.model

    def set_active_adapter(self, name: str) -> None:
        self.model.set_adapter(name)
        self.adapter_name = name

    def freeze_adapter(self, name: str) -> None:
        for n, p in self.model.named_parameters():
            if f".{name}." in n or n.endswith(f"{name}.default"):
                p.requires_grad = False

    def trainable_adapter_parameters(self, name: str):
        for n, p in self.model.named_parameters():
            if name in n and "lora" in n.lower() and p.requires_grad:
                yield p

    def encode(
        self,
        texts: list[str],
        *,
        max_length: int = 256,
    ) -> torch.Tensor:
        """返回 [B, hidden_size] 用户序列表示。"""
        if not texts:
            return torch.empty((0, self.hidden_size), device=self.device, dtype=torch.float32)
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc, output_hidden_states=False)
        last_hidden = out.last_hidden_state  # [B, T, H]
        attn = enc["attention_mask"]
        # 左 padding：每条序列最后一个有效 token 在 attention_mask 求和 - 1 位置
        lengths = attn.sum(dim=1).clamp_min(1) - 1
        batch_idx = torch.arange(last_hidden.size(0), device=self.device)
        pooled = last_hidden[batch_idx, lengths, :]
        return pooled.float()

    def save_adapter(self, save_dir: Path, adapter_name: str | None = None) -> None:
        name = adapter_name or self.adapter_name
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir, selected_adapters=[name])

    def load_adapter(self, adapter_dir: Path, adapter_name: str, *, is_trainable: bool = True) -> None:
        resolved = resolve_local_adapter_dir(adapter_dir, adapter_name)
        # 同名 adapter 已挂上时，PEFT 只按该目录重载权重
        self.model.load_adapter(str(resolved), adapter_name=adapter_name, is_trainable=is_trainable)


def build_qwen_rec_encoder(args, device: torch.device) -> QwenLoRAEncoder:
    return QwenLoRAEncoder(
        args.llm_model_path,
        device,
        adapter_name=REC_ADAPTER,
        lora_r=args.rec_lora_r,
        lora_alpha=args.rec_lora_alpha,
        lora_dropout=args.rec_lora_dropout,
        gradient_checkpointing=bool(getattr(args, "gradient_checkpointing", False)),
    )


def attach_exp_adapter(encoder: QwenLoRAEncoder, args) -> None:
    """在同一 PeftModel 上追加 exp adapter（解释训练用）。"""
    if EXP_ADAPTER in encoder.model.peft_config:
        return
    exp_cfg = LoraConfig(
        r=getattr(args, "exp_lora_r", args.rec_lora_r),
        lora_alpha=getattr(args, "exp_lora_alpha", args.rec_lora_alpha),
        lora_dropout=getattr(args, "exp_lora_dropout", args.rec_lora_dropout),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    encoder.model.add_adapter(EXP_ADAPTER, exp_cfg)


def load_explain_generation_llm(
    args,
    device: torch.device,
    rec_lora_dir: Path | None,
):
    """解释生成用 CausalLM：加载 rec adapter（冻结）+ 新建/训练 exp adapter。"""
    from peft import PeftModel

    llm_path = Path(args.llm_model_path)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    rec_dir = Path(rec_lora_dir) if rec_lora_dir else None
    if rec_dir and rec_dir.is_dir():
        resolved = resolve_local_adapter_dir(rec_dir, REC_ADAPTER)
        llm = PeftModel.from_pretrained(
            base,
            str(resolved),
            adapter_name=REC_ADAPTER,
            is_trainable=False,
        )
    else:
        llm = base
    exp_cfg = LoraConfig(
        r=args.exp_lora_r,
        lora_alpha=args.exp_lora_alpha,
        lora_dropout=args.exp_lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    if isinstance(llm, PeftModel):
        if EXP_ADAPTER not in llm.peft_config:
            llm.add_adapter(EXP_ADAPTER, exp_cfg)
    else:
        llm = get_peft_model(llm, exp_cfg, adapter_name=EXP_ADAPTER)
    llm.set_adapter(EXP_ADAPTER)
    for n, p in llm.named_parameters():
        if REC_ADAPTER in n and "lora" in n.lower():
            p.requires_grad = False
    if getattr(args, "gradient_checkpointing", False):
        llm.gradient_checkpointing_enable()
    llm.to(device)
    return llm, tokenizer


def save_exp_adapter(llm, save_dir: Path) -> None:
    if isinstance(llm, PeftModel):
        save_dir.mkdir(parents=True, exist_ok=True)
        llm.save_pretrained(save_dir, selected_adapters=[EXP_ADAPTER])


class MockLLMEncoder(nn.Module):
    """smoke：确定性小向量，不加载 Qwen。"""

    def __init__(self, device: torch.device, hidden_size: int = 64):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.proj = nn.Linear(hidden_size, hidden_size)

    def encode(self, texts: list[str], *, max_length: int = 256) -> torch.Tensor:
        del max_length
        rows = []
        for t in texts:
            h = hash(t) % 9973
            vec = torch.randn(self.hidden_size) * 0.01 + (h % 100) * 1e-4
            rows.append(vec)
        if not rows:
            return torch.empty((0, self.hidden_size), device=self.device)
        x = torch.stack(rows).to(self.device)
        return self.proj(x).float()

    def set_active_adapter(self, name: str) -> None:
        del name

    def save_adapter(self, save_dir: Path, adapter_name: str | None = None) -> None:
        del adapter_name
        save_dir.mkdir(parents=True, exist_ok=True)

    def load_adapter(self, adapter_dir: Path, adapter_name: str) -> None:
        del adapter_dir, adapter_name

    def trainable_adapter_parameters(self, name: str):
        del name
        return iter(())
