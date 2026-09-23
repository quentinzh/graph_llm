"""冻结 RoBERTa 文本编码与磁盘缓存。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Disk cache for text -> embedding vectors."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "index.json"
        self.index = self._load_index()

    def _rebuild_index_from_files(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for path in sorted(self.cache_dir.glob("*.pt")):
            if path.name.endswith(".part"):
                continue
            index[path.stem] = path.name
        return index

    def _load_index(self) -> dict[str, str]:
        if not self.index_path.exists():
            return self._rebuild_index_from_files()
        try:
            with self.index_path.open("r", encoding="utf-8") as f:
                index = json.load(f)
            if not isinstance(index, dict):
                raise ValueError("embedding cache index must be a JSON object")
            return index
        except (json.JSONDecodeError, ValueError, OSError):
            rebuilt = self._rebuild_index_from_files()
            if rebuilt:
                self.index = rebuilt
                self._save_index()
            return rebuilt

    def _save_index(self) -> None:
        payload = json.dumps(self.index, indent=2, ensure_ascii=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json.part",
            prefix="index.",
            dir=self.cache_dir,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.index_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def get(self, text: str) -> torch.Tensor | None:
        key = _text_hash(text)
        rel = self.index.get(key)
        if rel is None:
            path = self.cache_dir / f"{key}.pt"
            if not path.exists():
                return None
            rel = path.name
            self.index[key] = rel
        path = self.cache_dir / rel
        if not path.exists():
            self.index.pop(key, None)
            return None
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            self.index.pop(key, None)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def set(self, text: str, vector: torch.Tensor) -> None:
        key = _text_hash(text)
        rel = f"{key}.pt"
        path = self.cache_dir / rel
        tmp_path = path.with_suffix(".pt.part")
        try:
            torch.save(vector.detach().cpu(), tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        self.index[key] = rel
        self._save_index()


class RobertaTextEncoder(nn.Module):
    """冻结 RoBERTa-base，attention-mask mean pooling + L2 归一化。"""

    EMPTY_TEXT_MARKER = "[EMPTY_TEXT]"

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        *,
        cache_dir: Path | None = None,
        local_files_only: bool = False,
        max_length: int = 512,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.backend = "roberta"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            dtype=torch.float32,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self.model.eval().to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = int(self.model.config.hidden_size)
        self.cache = EmbeddingCache(cache_dir / f"dim{self.hidden_size}") if cache_dir else None
        self._empty_vec: torch.Tensor | None = None

    @torch.no_grad()
    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 16,
        *,
        use_cache: bool = True,
        show_progress: bool = False,
    ) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.hidden_size), device=self.device)
        normalized = [t if str(t).strip() else self.EMPTY_TEXT_MARKER for t in texts]
        if not use_cache or self.cache is None:
            return self._encode_texts_no_cache(
                normalized, batch_size=batch_size, show_progress=show_progress
            )

        out = torch.zeros((len(normalized), self.hidden_size), dtype=torch.float32)
        missing_texts: list[str] = []
        missing_indices: list[int] = []
        for idx, text in enumerate(normalized):
            vec = self.cache.get(text)
            if vec is not None and vec.shape[0] == self.hidden_size:
                out[idx] = vec.float()
            else:
                missing_texts.append(text)
                missing_indices.append(idx)
        if missing_texts:
            encoded = self._encode_texts_no_cache(
                missing_texts, batch_size=batch_size, show_progress=show_progress
            )
            for local_i, global_i in enumerate(missing_indices):
                out[global_i] = encoded[local_i].float()
                self.cache.set(normalized[global_i], encoded[local_i].cpu())
        return out.to(self.device)

    @torch.no_grad()
    def _encode_texts_no_cache(
        self, texts: list[str], batch_size: int = 16, *, show_progress: bool = False
    ) -> torch.Tensor:
        outputs = []
        batch_starts = range(0, len(texts), batch_size)
        if show_progress:
            batch_starts = tqdm(batch_starts, desc="RoBERTa encode")
        for start in batch_starts:
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            model_out = self.model(**encoded)
            hidden = model_out.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            for i, text in enumerate(batch):
                if text == self.EMPTY_TEXT_MARKER:
                    if self._empty_vec is None:
                        self._empty_vec = pooled[i].detach().clone()
                    outputs.append(self._empty_vec.clone())
                else:
                    outputs.append(pooled[i].float())
        return torch.stack(outputs, dim=0)


class SmokeTextEncoder(nn.Module):
    """调试：确定性哈希向量，非正式实验。"""

    def __init__(self, device: torch.device, hidden_size: int = 768):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.backend = "smoke_hash"

    @torch.no_grad()
    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 16,
        *,
        use_cache: bool = True,
        show_progress: bool = False,
    ) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.hidden_size), device=self.device)
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = torch.tensor(
                [digest[i % len(digest)] / 255.0 for i in range(self.hidden_size)],
                dtype=torch.float32,
            )
            vec = torch.nn.functional.normalize(vec, p=2, dim=0)
            out.append(vec)
        return torch.stack(out, dim=0).to(self.device)
