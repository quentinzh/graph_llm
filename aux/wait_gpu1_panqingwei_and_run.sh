#!/usr/bin/env bash
set -euo pipefail

GRAPH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$GRAPH_ROOT"

WAIT_GPU_ID="${WAIT_GPU_ID:-1}"
WAIT_USER="${WAIT_USER:-panqingwei}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-60}"
WAIT_LOG="${WAIT_LOG:-$GRAPH_ROOT/log/wait_gpu1_panqingwei.log}"

mkdir -p "$(dirname "$WAIT_LOG")"

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S %z')] $*"
  echo "$msg"
  echo "$msg" >> "$WAIT_LOG"
}

list_target_pids() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi not found"
    exit 1
  fi

  local pids=()
  local pid owner
  while IFS= read -r pid; do
    pid="${pid// /}"
    [[ -z "$pid" ]] && continue
    if ! owner="$(ps -o user= -p "$pid" 2>/dev/null)"; then
      continue
    fi
    owner="${owner// /}"
    if [[ "$owner" == "$WAIT_USER" ]]; then
      pids+=("$pid")
    fi
  done < <(nvidia-smi -i "$WAIT_GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)

  if ((${#pids[@]} > 0)); then
    printf '%s\n' "${pids[@]}"
  fi
}

describe_pid() {
  local pid="$1"
  local owner cmd
  owner="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ' || echo unknown)"
  cmd="$(ps -o args= -p "$pid" 2>/dev/null | head -c 200 || echo unknown)"
  echo "pid=$pid user=$owner cmd=$cmd"
}

log "Waiting for $WAIT_USER compute processes on cuda:$WAIT_GPU_ID to exit"
log "Poll interval: ${POLL_INTERVAL_SEC}s"
log "Training will start via aux/run.sh after wait completes"

while true; do
  mapfile -t target_pids < <(list_target_pids)
  if ((${#target_pids[@]} == 0)); then
    log "No $WAIT_USER compute processes on cuda:$WAIT_GPU_ID; starting training"
    break
  fi

  log "Still waiting: ${#target_pids[@]} process(es) on cuda:$WAIT_GPU_ID"
  for pid in "${target_pids[@]}"; do
    log "  - $(describe_pid "$pid")"
  done

  sleep "$POLL_INTERVAL_SEC"
done

log "Launching: bash aux/run.sh --devices 1 --fold 1 --dataset MoviesAndTV_corsa_filtered_small_15pct"
set +e
bash "$GRAPH_ROOT/aux/run.sh" \
  --devices 1 \
  --fold 1 \
  --dataset MoviesAndTV_corsa_filtered_small_15pct
status=$?
set -e

if ((status == 0)); then
  log "Training finished successfully (exit_code=0)"
else
  log "Training failed (exit_code=$status)"
fi

exit "$status"
