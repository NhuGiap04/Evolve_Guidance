#!/usr/bin/env bash
set -euo pipefail

# Launch DAS.py with accelerate for multi-GPU data-parallel inference.
#
# Usage:
#   bash examples/run_parallel_inference.sh
#   bash examples/run_parallel_inference.sh config/sd.py:pick 4 0,1,2,3
#   bash examples/run_parallel_inference.sh config/sdxl.py:clip 2
#
# Args:
#   1) config target     (default: config/sd.py:pick)
#   2) num_processes     (default: 2)
#   3) gpu_ids CSV       (optional, e.g. 0,1,2,3)
#
# Env overrides:
#   ACCELERATE_BIN       (default: accelerate)
#   DRY_RUN=1            (print command only)

CONFIG_TARGET="${1:-config/sd.py:pick}"
NUM_PROCESSES="${2:-2}"
GPU_IDS="${3:-}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
DRY_RUN="${DRY_RUN:-0}"

if ! [[ "$NUM_PROCESSES" =~ ^[0-9]+$ ]] || [ "$NUM_PROCESSES" -lt 1 ]; then
  echo "Error: num_processes must be an integer >= 1" >&2
  exit 2
fi

if [ -n "$GPU_IDS" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi

CMD=(
  "$ACCELERATE_BIN"
  launch
  --num_processes
  "$NUM_PROCESSES"
  DAS.py
  --config
  "$CONFIG_TARGET"
)

echo "Command: ${CMD[*]}"
if [ -n "$GPU_IDS" ]; then
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

"${CMD[@]}"
