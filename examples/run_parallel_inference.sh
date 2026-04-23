#!/usr/bin/env bash
set -euo pipefail

# Run Stein-guided SD examples in parallel across GPUs (without DAS/accelerate).
#
# Notes:
# - This launcher starts one Python process per prompt and pins each to a GPU.
# - This launcher starts the SD 1.5 Stein example by default.
# - Set `RUN_SCRIPT=examples/sdxl.py` if you want the SDXL Stein path instead.
#
# Usage:
#   bash examples/run_parallel_inference.sh prompts/hps_v2_all_eval.txt
#   bash examples/run_parallel_inference.sh prompts/hps_v2_all_eval.txt "pick" "0,1,2,3"
#   bash examples/run_parallel_inference.sh prompts/hps_v2_all_eval.txt "pick" "0,1" "logs/sd_parallel"
#
# Args:
#   1) prompts file    (required)
#   2) config preset   (default: pick)
#   3) gpu ids csv     (default: 0)
#   4) output root     (default: logs/sd_parallel)
#
# Env overrides:
#   PYTHON_BIN         (default: python)
#   RUN_SCRIPT         (default: examples/sd.py)
#   STEER_REWARD       (default: use script/config default)
#   EVAL_REWARD        (default: image_reward)
#   COMMON_ARGS        (default: "")
#   STREAM_LOGS=1      (mirror child output to terminal with tee)
#   DRY_RUN=1          (print commands only)

PROMPTS_FILE="${1:-}"
CONFIG_PRESET="${2:-pick}"
GPU_IDS_CSV="${3:-0}"
OUTPUT_ROOT="${4:-logs/sd_parallel}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_SCRIPT="${RUN_SCRIPT:-examples/sd.py}"
STEER_REWARD="${STEER_REWARD:-}"
EVAL_REWARD="${EVAL_REWARD:-image_reward}"
COMMON_ARGS="${COMMON_ARGS:-}"
STREAM_LOGS="${STREAM_LOGS:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [ -z "$PROMPTS_FILE" ]; then
  echo "Usage: bash examples/run_parallel_inference.sh <prompts_file> [config] [gpu_ids_csv] [output_root]" >&2
  exit 2
fi

if [ ! -f "$PROMPTS_FILE" ]; then
  echo "Error: prompts file not found: $PROMPTS_FILE" >&2
  exit 2
fi

if [ ! -f "$RUN_SCRIPT" ]; then
  echo "Error: run script not found: $RUN_SCRIPT" >&2
  exit 2
fi


IFS=',' read -r -a GPUS <<< "$GPU_IDS_CSV"
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "Error: no GPU ids provided." >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/_logs"

declare -a PIDS=()
declare -a PID_GPU=()
declare -a PID_RUN=()
FAILURES=0

slot_wait() {
  local max_jobs="$1"
  while [ "${#PIDS[@]}" -ge "$max_jobs" ]; do
    local i pid
    local -a next_pids=()
    local -a next_pid_gpu=()
    local -a next_pid_run=()

    for i in "${!PIDS[@]}"; do
      pid="${PIDS[$i]:-}"
      if [ -z "$pid" ]; then
        continue
      fi

      if kill -0 "$pid" 2>/dev/null; then
        next_pids+=("$pid")
        next_pid_gpu+=("${PID_GPU[$i]:-unknown_gpu}")
        next_pid_run+=("${PID_RUN[$i]:-unknown_run}")
      else
        local run_name="${PID_RUN[$i]:-unknown_run}"
        local gpu="${PID_GPU[$i]:-unknown_gpu}"
        if wait "$pid"; then
          echo "[$run_name] done on gpu=$gpu"
        else
          echo "[$run_name] failed on gpu=$gpu" >&2
          FAILURES=$((FAILURES + 1))
        fi
      fi
    done

    PIDS=("${next_pids[@]}")
    PID_GPU=("${next_pid_gpu[@]}")
    PID_RUN=("${next_pid_run[@]}")

    sleep 1
  done
}

run_idx=0
while IFS= read -r raw_line || [ -n "$raw_line" ]; do
  prompt="${raw_line#"${raw_line%%[![:space:]]*}"}"
  prompt="${prompt%"${prompt##*[![:space:]]}"}"

  if [ -z "$prompt" ]; then
    continue
  fi
  case "$prompt" in
    \#*) continue ;;
  esac

  gpu="${GPUS[$((run_idx % ${#GPUS[@]}))]}"
  run_name=$(printf "run_%04d" "$run_idx")
  run_dir="$OUTPUT_ROOT/$run_name"
  log_file="$OUTPUT_ROOT/_logs/${run_name}.log"

  slot_wait "${#GPUS[@]}"

  cmd=(
    "$PYTHON_BIN" "$RUN_SCRIPT"
    --config "$CONFIG_PRESET"
    --prompt "$prompt"
    --eval-reward "$EVAL_REWARD"
    --output-dir "$run_dir"
    --device cuda
  )

  if [ -n "$STEER_REWARD" ]; then
    cmd+=(--steer-reward "$STEER_REWARD")
  fi

  if [ -n "$COMMON_ARGS" ]; then
    # shellcheck disable=SC2206
    extra_args=( $COMMON_ARGS )
    cmd+=("${extra_args[@]}")
  fi

  echo "[$run_name] gpu=$gpu"
  echo "[$run_name] prompt=$prompt"
  echo "[$run_name] cmd: CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO_ROOT ${cmd[*]}"

  if [ "$DRY_RUN" = "1" ]; then
    run_idx=$((run_idx + 1))
    continue
  fi

  if [ "$STREAM_LOGS" = "1" ]; then
    (
      export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
      CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" 2>&1 | tee "$log_file"
    ) &
  else
    (
      export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
      CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"
    ) >"$log_file" 2>&1 &
  fi

  PIDS+=("$!")
  PID_GPU+=("$gpu")
  PID_RUN+=("$run_name")

  run_idx=$((run_idx + 1))
done < "$PROMPTS_FILE"

failures="$FAILURES"
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]:-}"
  if [ -z "$pid" ]; then
    continue
  fi
  run_name="${PID_RUN[$i]:-unknown_run}"
  gpu="${PID_GPU[$i]:-unknown_gpu}"
  if wait "$pid"; then
    echo "[$run_name] done on gpu=$gpu"
  else
    echo "[$run_name] failed on gpu=$gpu" >&2
    failures=$((failures + 1))
  fi
done

echo "Finished. Total runs: $run_idx, failures: $failures"
echo "Logs: $OUTPUT_ROOT/_logs"

if [ "$failures" -gt 0 ]; then
  exit 1
fi
