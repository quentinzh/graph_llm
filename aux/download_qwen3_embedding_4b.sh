#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO="Qwen/Qwen3-Embedding-4B"
REV="main"
DIR="$(cd "$(dirname "$0")/.." && pwd)/pretrain_llm/qwen3-embedding-4b"

files=(
  config.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
  model.safetensors.index.json
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

# Download all safetensor shards listed in index if present.
index="$DIR/model.safetensors.index.json"
if [[ -f "$index" ]]; then
  python - <<'PY' "$index" "$DIR" "$ENDPOINT" "$REPO" "$REV"
import json, subprocess, sys
from pathlib import Path
index_path, out_dir, endpoint, repo, rev = sys.argv[1:6]
data = json.loads(Path(index_path).read_text())
files = sorted(set(data.get("weight_map", {}).values()))
for name in files:
    out = Path(out_dir) / name
    if out.exists() and out.stat().st_size > 0:
        print(f"Skip existing: {name}")
        continue
    url = f"{endpoint.rstrip('/')}/{repo}/resolve/{rev}/{name}"
    print(f"Downloading: {name}")
    subprocess.check_call([
        "curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "2", "-f",
        "-o", str(out) + ".part", url,
    ])
    Path(str(out) + ".part").replace(out)
PY
fi

echo "Download complete: $DIR"
