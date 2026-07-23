"""Qwen embedding backends for token and item-description encoding."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_embedding_model_path(repo_root: Path | None = None) -> str:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    candidates = [
        root / "pretrain_llm" / "qwen3-embedding-0.6b",
    ]
    for local in candidates:
        if local.exists() and (local / "config.json").exists():
            return str(local)
    return str(candidates[0])


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
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            backup = self.index_path.with_suffix(".json.corrupt")
            try:
                self.index_path.replace(backup)
            except OSError:
                pass
            rebuilt = self._rebuild_index_from_files()
            print(
                f"WARNING: corrupted embedding cache index at {self.index_path} ({exc}); "
                f"rebuilt index from {len(rebuilt)} cached vectors."
            )
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


class QwenEmbeddingEncoder(nn.Module):
    """Frozen Qwen3-Embedding encoder with optional causal-LM fallback."""

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        *,
        cache_dir: Path | None = None,
        fallback_lm=None,
        local_files_only: bool = True,
        max_length: int = 512,
    ):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self._cache_dir = cache_dir
        self.cache = None  # created after hidden_size is known, with dim suffix
        self.fallback_lm = fallback_lm
        self.backend = "fallback"
        self.hidden_size = 2560
        self.tokenizer = None
        self.model = None

        if fallback_lm is not None and not Path(model_path).exists() and "/" not in model_path:
            self._init_fallback(fallback_lm)
            if self._cache_dir is not None:
                self.cache = EmbeddingCache(self._cache_dir / f"dim{self.hidden_size}")
            return

        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(
                model_path,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                device_map={"": str(device)} if device.type == "cuda" else None,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            self.hidden_size = int(getattr(self.model.config, "hidden_size", self.hidden_size))
            self.backend = "qwen_embedding"
            if self._cache_dir is not None:
                self.cache = EmbeddingCache(self._cache_dir / f"dim{self.hidden_size}")
        except Exception as exc:
            if fallback_lm is None:
                raise RuntimeError(
                    f"Failed to load embedding model at {model_path}: {exc}. "
                    "Download with aux/download_qwen3_embedding_0.6b.sh "
                    "or pass a local path; LM embed_tokens fallback requires fallback_lm."
                ) from exc
            print(
                f"WARNING: embedding model unavailable ({exc}); using LM embed_tokens fallback. "
                "Embedding encoder is not resident on a separate GPU; dual-GPU memory benefit is reduced."
            )
            self._init_fallback(fallback_lm)

    def _init_fallback(self, fallback_lm) -> None:
        self.fallback_lm = fallback_lm
        self.tokenizer = None
        self.model = None
        self.hidden_size = int(fallback_lm.config.hidden_size)
        self.backend = "lm_embed_tokens"
        if self._cache_dir is not None:
            self.cache = EmbeddingCache(self._cache_dir / f"dim{self.hidden_size}")

    @torch.no_grad()
    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 16,
        *,
        use_cache: bool = True,
    ) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.hidden_size), device=self.device)
        if not use_cache:
            return self._encode_texts_no_cache(texts, batch_size=batch_size)

        cached_vectors = []
        missing_texts: list[str] = []
        missing_indices: list[int] = []
        for idx, text in enumerate(texts):
            if self.cache is not None:
                vec = self.cache.get(text)
                if vec is not None and vec.shape[0] == self.hidden_size:
                    cached_vectors.append((idx, vec))
                    continue
            missing_texts.append(text)
            missing_indices.append(idx)

        out = torch.zeros((len(texts), self.hidden_size), dtype=torch.float32)
        for idx, vec in cached_vectors:
            out[idx] = vec.float()

        if missing_texts:
            encoded = self._encode_texts_no_cache(missing_texts, batch_size=batch_size)
            for local_idx, global_idx in enumerate(missing_indices):
                out[global_idx] = encoded[local_idx].float()
                if self.cache is not None:
                    self.cache.set(texts[global_idx], encoded[local_idx].cpu())

        return out.to(self.device)

    @torch.no_grad()
    def _encode_texts_no_cache(self, texts: list[str], batch_size: int = 16) -> torch.Tensor:
        if self.backend == "lm_embed_tokens":
            assert self.fallback_lm is not None
            embed = self.fallback_lm.get_input_embeddings()
            outputs = []
            for text in texts:
                token_ids = self._tokenize_with_fallback(text)
                if not token_ids:
                    outputs.append(torch.zeros(self.hidden_size, device=self.device))
                    continue
                ids = torch.tensor([token_ids], device=self.device, dtype=torch.long)
                vec = embed(ids).mean(dim=1).squeeze(0).float()
                outputs.append(vec)
            return torch.stack(outputs, dim=0)

        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
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
            outputs.append(pooled.float().cpu())
        return torch.cat(outputs, dim=0).to(self.device)

    def _tokenize_with_fallback(self, text: str) -> list[int]:
        if self.tokenizer is not None:
            return self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if self.fallback_lm is not None:
            tok = getattr(self.fallback_lm, "tokenizer", None)
            if tok is not None:
                return tok(text, add_special_tokens=False)["input_ids"]
        return []

    @torch.no_grad()
    def encode_token_ids(
        self,
        token_ids: Iterable[int],
        decode_fn,
        batch_size: int = 64,
    ) -> torch.Tensor:
        token_ids = [int(t) for t in token_ids]
        if not token_ids:
            return torch.empty((0, self.hidden_size), device=self.device)
        surfaces = [decode_fn(t) for t in token_ids]
        return self.encode_texts(surfaces, batch_size=batch_size)


class NodeFeatureProjector(nn.Module):
    """Project cached/frozen embeddings + numeric graph stats to selector dim."""

    def __init__(self, input_dim: int, hidden_dim: int, stat_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + stat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, token_emb: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([token_emb, stats], dim=-1))
