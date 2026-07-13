#!/usr/bin/env bash
# 创建 graph_llm 运行所需的 conda 虚拟环境并安装 Python 依赖。
# 不下载预训练模型；请确保 graph_llm/pretrain_llm/ 下已有本地权重。
set -euo pipefail

GRAPH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$GRAPH_ROOT/.." && pwd)"
ENV_NAME="${GRAPH_ENV_NAME:-fair}"
PYTHON_VERSION="${GRAPH_PYTHON_VERSION:-3.11}"
HF_ENDPOINT_VALUE="${GRAPH_HF_ENDPOINT:-${HF_ENDPOINT:-https://hf-mirror.com}}"

export HF_ENDPOINT="$HF_ENDPOINT_VALUE"

# 尝试加载 conda（非交互 shell 中 conda 可能未初始化）
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
  echo "错误: 未找到 conda。请先安装 conda 或在当前 shell 中加载 conda。" >&2
  exit 1
fi

echo "graph_llm 根目录: $GRAPH_ROOT"
echo "仓库根目录:     $REPO_ROOT"
echo "目标 conda 环境: $ENV_NAME (python=$PYTHON_VERSION)"
echo "HF_ENDPOINT:    $HF_ENDPOINT"

# 创建 conda 环境（若已存在则跳过）
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "正在创建 conda 环境: $ENV_NAME"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
else
  echo "conda 环境已存在: $ENV_NAME"
fi

echo "正在安装 Python 依赖..."
conda run --no-capture-output -n "$ENV_NAME" env PIP_USER=0 PYTHONNOUSERSITE=1 \
  python -m pip install -U pip
conda run --no-capture-output -n "$ENV_NAME" env PIP_USER=0 PYTHONNOUSERSITE=1 \
  python -m pip install -r "$GRAPH_ROOT/requirements.txt"

# 检查本地预训练模型是否存在（仅提示，不下载）
check_local_model() {
  local model_dir="$1"
  local model_name="$2"
  if [[ -f "$model_dir/config.json" ]]; then
    echo "  [OK] $model_name -> $model_dir"
  else
    echo "  [WARN] 未找到 $model_name: $model_dir/config.json" >&2
    echo "         请手动放置权重，或运行 aux/download_*.sh 下载。" >&2
  fi
}

echo "检查本地预训练模型..."
check_local_model "$GRAPH_ROOT/pretrain_llm/qwen3-4b" "Qwen3-4B"
check_local_model "$GRAPH_ROOT/pretrain_llm/qwen3-embedding-0.6b" "Qwen3-Embedding-0.6B"

echo "验证关键依赖导入..."
conda run --no-capture-output -n "$ENV_NAME" env PYTHONNOUSERSITE=1 python - <<'PY'
import importlib

packages = [
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "pandas",
    "numpy",
    "sklearn",
    "tqdm",
    "sentencepiece",
    "safetensors",
]

for name in packages:
    mod = importlib.import_module(name)
    version = getattr(mod, "__version__", "unknown")
    print(f"  [OK] {name} {version}")

import torch
if torch.cuda.is_available():
    print(f"  [OK] CUDA available, device_count={torch.cuda.device_count()}")
else:
    print("  [WARN] CUDA 不可用，训练/推理将回退到 CPU")
PY

cat <<EOF

graph_llm 环境创建完成。

激活环境:
  conda activate $ENV_NAME

运行 smoke test（2 个 batch，无需完整 epoch）:
  cd $GRAPH_ROOT
  conda run -n $ENV_NAME python aux/tests/test_smoke.py

训练示例:
  cd $GRAPH_ROOT
  bash aux/run.sh --dataset_name Amazon/MoviesAndTV_corsa_filtered_small_15pct --split_indices 1

如需 flash-attn（--attn_implementation flash_attention_2）:
  bash aux/setup_graph_fa2_env.sh

EOF
