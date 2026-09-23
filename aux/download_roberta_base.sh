#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO="FacebookAI/roberta-base"
REV="main"
DIR="$(cd "$(dirname "$0")/.." && pwd)/pretrain_llm/roberta-base"

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
