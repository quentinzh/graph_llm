#!/usr/bin/env bash
set -euo pipefail

GRAPH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$GRAPH_ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-${GRAPH_HF_ENDPOINT:-https://hf-mirror.com}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
ENV_NAME="${GRAPH_ENV_NAME:-fair}"

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

LOG_DIR="$GRAPH_ROOT/log"
DATASET_NAME="Amazon/MoviesAndTV_corsa_filtered_small_15pct/"
LOG_NAME="graph_profile.log"
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
  --lambda_selector 0.1
  --top_m_evidence 10
  --tail_weight_min 0.5
  --tail_weight_max 2.0
  --model_path "$GRAPH_ROOT/pretrain_llm/qwen3-4b"
  --embedding_model_path "$GRAPH_ROOT/pretrain_llm/qwen3-embedding-0.6b"
  --profile_dir "$GRAPH_ROOT/data/profiles"
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
