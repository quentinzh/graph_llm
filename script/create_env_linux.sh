#!/usr/bin/env bash
# 在 Linux 上创建 graph_llm 环境，并固定安装 CUDA 12.8 版 PyTorch。
# 具体的 conda、pip 与 Hugging Face 镜像逻辑复用 create_env.sh，避免两份脚本配置漂移。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRAPH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OS_NAME="$(uname -s)"
MACHINE_ARCH="$(uname -m)"

if [[ "$OS_NAME" != "Linux" ]]; then
  echo "错误: create_env_linux.sh 只能在 Linux 上运行，当前平台为 $OS_NAME ($MACHINE_ARCH)。" >&2
  echo "      macOS 请运行: bash $SCRIPT_DIR/create_env.sh" >&2
  exit 1
fi

if [[ ! -f "$GRAPH_ROOT/requirements.txt" ]]; then
  echo "错误: 未找到 Linux CUDA 依赖文件: $GRAPH_ROOT/requirements.txt" >&2
  exit 1
fi

# requirements.txt 固定使用阿里云的 PyTorch CUDA 12.8 wheel 镜像。
export GRAPH_REQUIREMENTS_FILE="$GRAPH_ROOT/requirements.txt"

echo "检测到 Linux ($MACHINE_ARCH)，将安装 CUDA 12.8 版 PyTorch。"
exec bash "$SCRIPT_DIR/create_env.sh" "$@"
