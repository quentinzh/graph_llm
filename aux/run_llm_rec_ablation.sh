#!/usr/bin/env bash
# Instruments 推荐消融：融合 / w/o LLM / w/o GNN（需 GPU 与 Qwen 权重；smoke 可加 --smoke_mock_encoder）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
COMMON=(
  -m graph_llm.main
  --dataset_name Instruments
  --data_dir "$ROOT/data"
  --search_mode exact
  --batch_size 8
  --devices 1
)

echo "=== 融合 (GNN+LLM) ==="
$PY "${COMMON[@]}" --epochs 20 --early_stop_patience 3 "$@"

echo "=== w/o LLM (GNN-only) ==="
$PY "${COMMON[@]}" --no_llm_rec --epochs 50 --early_stop_patience 5 "$@"

echo "=== w/o GNN (LLM-only) ==="
$PY "${COMMON[@]}" --no_gnn_rec --epochs 20 --early_stop_patience 3 "$@"
