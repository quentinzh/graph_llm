#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO="Qwen/Qwen3-4B"
REV="main"
DIR="$(cd "$(dirname "$0")/.." && pwd)/pretrain_llm/qwen3-4b"

files=(
  config.json
  generation_config.json
  merges.txt
  vocab.json
  tokenizer.json
  tokenizer_config.json
  model.safetensors.index.json
  model-00001-of-00003.safetensors
  model-00002-of-00003.safetensors
  model-00003-of-00003.safetensors
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
