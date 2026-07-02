#!/usr/bin/env bash
set -euo pipefail

GRAPH_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$GRAPH_ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-${GRAPH_HF_ENDPOINT:-https://hf-mirror.com}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
ENV_NAME="${GRAPH_ENV_NAME:-fair}"

LOG_DIR="$GRAPH_ROOT/log"
DATASET_NAME="Amazon/MoviesAndTV_corsa_filtered_small_15pct/"
LOG_NAME="graph_evidence.log"
ARGS=("$@")

for ((idx = 0; idx < ${#ARGS[@]}; idx++)); do
  arg="${ARGS[$idx]}"
  case "$arg" in
    --log_dir=*)
      LOG_DIR="${arg#*=}"
      ;;
    --log_dir)
      if ((idx + 1 < ${#ARGS[@]})); then
        LOG_DIR="${ARGS[$((idx + 1))]}"
      fi
      ;;
    --dataset_name=*|--dataset=*)
      DATASET_NAME="${arg#*=}"
      ;;
    --dataset_name|--dataset)
      if ((idx + 1 < ${#ARGS[@]})); then
        DATASET_NAME="${ARGS[$((idx + 1))]}"
      fi
      ;;
    --log_name=*)
      LOG_NAME="${arg#*=}"
      ;;
    --log_name)
      if ((idx + 1 < ${#ARGS[@]})); then
        LOG_NAME="${ARGS[$((idx + 1))]}"
      fi
      ;;
  esac
done

if [[ "$LOG_DIR" != /* ]]; then
  LOG_DIR="$GRAPH_ROOT/$LOG_DIR"
fi
LOG_PATH="$LOG_DIR/$DATASET_NAME/$LOG_NAME"
mkdir -p "$(dirname "$LOG_PATH")"

STDERR_FILE="$(mktemp "${TMPDIR:-/tmp}/graph-run-stderr.XXXXXX")"
trap 'rm -f "$STDERR_FILE"' EXIT

CMD=(
  conda run --no-capture-output -n "$ENV_NAME" python -u main.py
  --batch_size 8
  --eval_batch_size 8
  --accumulation_steps 4
  --epochs 3
  --lambda_ul 0.1
  --top_m_evidence 10
  --ul_candidate_k 20
  --model_path "$GRAPH_ROOT/models/qwen3-4b"
  --embedding_model_path "$GRAPH_ROOT/models/qwen3-embedding-0.6b"
  --profile_dir "$GRAPH_ROOT/user_profiles_structured"
  --data_dir "$GRAPH_ROOT/data"
  "$@"
)

set +e
"${CMD[@]}" 2> >(tee "$STDERR_FILE" >&2)
status=$?
set -e

if ((status != 0)); then
  {
    printf '\n===== run.sh failed =====\n'
    printf 'time: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"
    printf 'exit_code: %s\n' "$status"
    printf 'command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    printf '%s\n' '----- stderr -----'
    cat "$STDERR_FILE"
    printf '%s\n' '===== end run.sh failure ====='
  } >> "$LOG_PATH"
  exit "$status"
fi
