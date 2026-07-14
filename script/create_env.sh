#!/usr/bin/env bash
# 创建 graph_llm 运行所需的 conda 虚拟环境并安装 Python 依赖。
# 不下载预训练模型；请确保 graph_llm/pretrain_llm/ 下已有本地权重。
set -euo pipefail

GRAPH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$GRAPH_ROOT/.." && pwd)"
ENV_NAME="${GRAPH_ENV_NAME:-fair}"
PYTHON_VERSION="${GRAPH_PYTHON_VERSION:-3.11}"
HF_ENDPOINT_VALUE="${GRAPH_HF_ENDPOINT:-${HF_ENDPOINT:-https://hf-mirror.com}}"
# AutoDL 默认优先使用阿里云 PyPI 镜像。仅使用一个常规 PyPI 源，避免 pip 对每个包
# 同时请求多个索引而拖慢解析；特殊包源可通过 GRAPH_PIP_EXTRA_INDEX_URL 补充。
PIP_INDEX_URL="${GRAPH_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
PIP_EXTRA_INDEX_URL="${GRAPH_PIP_EXTRA_INDEX_URL:-}"
# CUDA 版 PyTorch 不在普通 PyPI 中。阿里云地址是平铺的 wheel 下载目录，
# 因此必须通过 --find-links 解析，而不能作为 --extra-index-url 包索引使用。
# 该地址是阿里云镜像，而非 download.pytorch.org 官方源；仅在 Linux CUDA 依赖中启用。
PYTORCH_WHEEL_INDEX_URL="${GRAPH_PYTORCH_WHEEL_INDEX_URL:-https://mirrors.aliyun.com/pytorch-wheels/cu128}"
# 默认复用下载过的 wheel，AutoDL 断点重连或重复建环境时无需再次下载大体积 PyTorch。
PIP_CACHE_DIR="${GRAPH_PIP_CACHE_DIR:-$HOME/.cache/pip}"
PIP_TIMEOUT="${GRAPH_PIP_TIMEOUT:-30}"
PIP_RETRIES="${GRAPH_PIP_RETRIES:-3}"
# conda 使用清华 Anaconda 镜像，且不修改用户全局的 .condarc 配置。
CONDA_MAIN_CHANNEL="${GRAPH_CONDA_MAIN_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}"
CONDA_R_CHANNEL="${GRAPH_CONDA_R_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r}"
CONDA_MSYS2_CHANNEL="${GRAPH_CONDA_MSYS2_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2}"
# Linux 默认安装 CUDA 12.8 版本；macOS 没有 CUDA wheel，自动改用标准 PyTorch wheel。
OS_NAME="$(uname -s)"
MACHINE_ARCH="$(uname -m)"
if [[ -n "${GRAPH_REQUIREMENTS_FILE:-}" ]]; then
  REQUIREMENTS_FILE="$GRAPH_REQUIREMENTS_FILE"
elif [[ "$OS_NAME" == "Darwin" ]]; then
  REQUIREMENTS_FILE="$GRAPH_ROOT/requirements-macos.txt"
else
  REQUIREMENTS_FILE="$GRAPH_ROOT/requirements.txt"
fi

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  echo "错误: 未找到依赖文件: $REQUIREMENTS_FILE" >&2
  echo "      可通过 GRAPH_REQUIREMENTS_FILE 指定其他依赖文件。" >&2
  exit 1
fi

export HF_ENDPOINT="$HF_ENDPOINT_VALUE"

# 使用数组传递参数，避免 URL 或环境变量中包含特殊字符时被 shell 错误拆分。
CONDA_CHANNEL_ARGS=(
  --override-channels
  -c "$CONDA_MAIN_CHANNEL"
  -c "$CONDA_R_CHANNEL"
  -c "$CONDA_MSYS2_CHANNEL"
)
PIP_MIRROR_ARGS=(
  --index-url "$PIP_INDEX_URL"
  --prefer-binary
  --no-input
  --disable-pip-version-check
  --timeout "$PIP_TIMEOUT"
  --retries "$PIP_RETRIES"
)

# 空值表示不添加普通 PyPI 的备用源；需要备用源时由用户显式指定。
if [[ -n "$PIP_EXTRA_INDEX_URL" ]]; then
  PIP_MIRROR_ARGS+=(--extra-index-url "$PIP_EXTRA_INDEX_URL")
fi

# 仅当 Linux 依赖文件确实声明了 `torch==...+cu...` 时才加入 CUDA wheel 目录。
# 这样通过 GRAPH_REQUIREMENTS_FILE 指定 CPU 依赖文件时不会访问多余的专用镜像。
PYTORCH_INDEX_ARGS=()
if [[ "$OS_NAME" == "Linux" \
  && "${GRAPH_DISABLE_CUDA_TORCH_INDEX:-0}" != "1" \
  ]] && grep -Eq '^[[:space:]]*torch==[^[:space:]]*\+cu[[:digit:]]+' "$REQUIREMENTS_FILE"; then
  PYTORCH_INDEX_ARGS=(--find-links "$PYTORCH_WHEEL_INDEX_URL")
fi

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
echo "运行平台:       $OS_NAME ($MACHINE_ARCH)"
echo "目标 conda 环境: $ENV_NAME (python=$PYTHON_VERSION)"
echo "HF_ENDPOINT:    $HF_ENDPOINT"
echo "conda 镜像:      $CONDA_MAIN_CHANNEL"
echo "pip 主镜像:      $PIP_INDEX_URL"
echo "pip wheel 缓存:  $PIP_CACHE_DIR"
if [[ -n "$PIP_EXTRA_INDEX_URL" ]]; then
  echo "pip 补充镜像:    $PIP_EXTRA_INDEX_URL"
fi
if (( ${#PYTORCH_INDEX_ARGS[@]} > 0 )); then
  echo "PyTorch CUDA wheel 目录: $PYTORCH_WHEEL_INDEX_URL"
fi
echo "依赖文件:       $REQUIREMENTS_FILE"

# 创建 conda 环境（若已存在则跳过）
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "正在创建 conda 环境: $ENV_NAME"
  conda create -y "${CONDA_CHANNEL_ARGS[@]}" -n "$ENV_NAME" "python=$PYTHON_VERSION"
else
  echo "conda 环境已存在: $ENV_NAME"
fi

echo "正在安装 Python 依赖..."
mkdir -p "$PIP_CACHE_DIR"
conda run --no-capture-output -n "$ENV_NAME" env PIP_USER=0 PYTHONNOUSERSITE=1 PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  python -m pip install "${PIP_MIRROR_ARGS[@]}" -U pip

# Bash 在 set -u 下不能直接展开空数组。先构造始终非空的基础参数，
# 再按需附加 Linux CUDA 的 PyTorch 专用镜像，兼容 macOS 与 CPU 自定义 requirements。
PIP_INSTALL_ARGS=("${PIP_MIRROR_ARGS[@]}")
if (( ${#PYTORCH_INDEX_ARGS[@]} > 0 )); then
  PIP_INSTALL_ARGS+=("${PYTORCH_INDEX_ARGS[@]}")
fi
conda run --no-capture-output -n "$ENV_NAME" env PIP_USER=0 PYTHONNOUSERSITE=1 PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  python -m pip install "${PIP_INSTALL_ARGS[@]}" -r "$REQUIREMENTS_FILE"

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
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("  [OK] Apple MPS available")
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

镜像覆盖示例（临时改用其他源）:
  GRAPH_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash script/create_env.sh

替换 CUDA PyTorch wheel 镜像（或禁用 CUDA 专用索引）:
  GRAPH_PYTORCH_WHEEL_INDEX_URL=https://mirrors.aliyun.com/pytorch-wheels/cu128 bash script/create_env.sh
  GRAPH_DISABLE_CUDA_TORCH_INDEX=1 bash script/create_env.sh

在 macOS 上强制使用其他依赖文件:
  GRAPH_REQUIREMENTS_FILE=requirements.txt bash script/create_env.sh

EOF
