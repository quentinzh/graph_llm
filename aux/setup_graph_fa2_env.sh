#!/usr/bin/env bash
# Conda env with flash-attn for --attn_implementation flash_attention_2.
# Does not modify the fair environment.
set -euo pipefail

GRAPH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${GRAPH_FA2_ENV_NAME:-graph_llm_fa2}"
HF_ENDPOINT_VALUE="${GRAPH_HF_ENDPOINT:-${HF_ENDPOINT:-https://hf-mirror.com}}"
SKIP_MODEL_DOWNLOAD="${GRAPH_SKIP_MODEL_DOWNLOAD:-1}"
MAX_JOBS="${GRAPH_FA2_MAX_JOBS:-8}"

export HF_ENDPOINT="$HF_ENDPOINT_VALUE"

if ! command -v conda >/dev/null 2>&1; then
  for conda_sh in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/opt/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$conda_sh" ]]; then
      # shellcheck disable=SC1090
      source "$conda_sh"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install conda or load it before running this script." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda environment: $ENV_NAME (python=3.11 for flash-attn wheels)"
  conda create -y -n "$ENV_NAME" python=3.11
fi

echo "Installing graph_llm dependencies into conda env: $ENV_NAME"
conda run --no-capture-output -n "$ENV_NAME" env PIP_USER=0 PYTHONNOUSERSITE=1 python -m pip install -U pip
conda run --no-capture-output -n "$ENV_NAME" env PIP_USER=0 PYTHONNOUSERSITE=1 \
  python -m pip install -r "$GRAPH_ROOT/requirements.txt"

echo "Installing flash-attn (MAX_JOBS=$MAX_JOBS; may take several minutes)..."
conda run --no-capture-output -n "$ENV_NAME" env PIP_USER=0 PYTHONNOUSERSITE=1 MAX_JOBS="$MAX_JOBS" \
  python -m pip install flash-attn --no-build-isolation

echo "Verifying flash_attn import..."
conda run --no-capture-output -n "$ENV_NAME" env PYTHONNOUSERSITE=1 python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"

if [[ "$SKIP_MODEL_DOWNLOAD" == "1" ]]; then
  echo "GRAPH_SKIP_MODEL_DOWNLOAD=1; skipping model downloads."
  echo "Activate with: conda activate $ENV_NAME"
  echo "Run training: cd $GRAPH_ROOT && python main.py --device 1 --attn_implementation flash_attention_2 ..."
  exit 0
fi

echo "Downloading Qwen models via HF_ENDPOINT=$HF_ENDPOINT"
conda run --no-capture-output -n "$ENV_NAME" python - "$GRAPH_ROOT" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(sys.argv[1])
models = [
    ("Qwen/Qwen3-4B", root / "pretrain_llm" / "qwen3-4b"),
    ("Qwen/Qwen3-Embedding-0.6B", root / "pretrain_llm" / "qwen3-embedding-0.6b"),
]

for repo_id, target_dir in models:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

print("graph_llm_fa2 environment setup complete.")
PY
