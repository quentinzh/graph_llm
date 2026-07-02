#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO="Qwen/Qwen3-Embedding-0.6B"
REV="main"
DIR="$(cd "$(dirname "$0")/.." && pwd)/pretrain_llm/qwen3-embedding-0.6b"

# 0.6B uses a single model.safetensors (not sharded), no index file needed.
files=(
  config.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
  model.safetensors
)

mkdir -p "$DIR"

for f in "${files[@]}"; do
  out="$DIR/$f"
  if [[ -f "$out" && -s "$out" ]]; then
    echo "Skip existing: $f"
    continue
  fi
  url="${ENDPOINT%/}/${REPO}/resolve/${REV}/${f}"
  echo "Downloading: $f"
  curl -L -C - --retry 5 --retry-delay 2 -f -o "${out}.part" "$url"
  mv "${out}.part" "$out"
done

echo "Download complete: $DIR"
