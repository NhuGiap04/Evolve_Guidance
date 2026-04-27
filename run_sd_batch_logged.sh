#!/usr/bin/env bash
set -euo pipefail

# Runs SD batch generation and writes combined stdout/stderr to a timestamped log file.
# Override defaults via environment variables, e.g.:
#   PYTHON_BIN=/venv/main/bin/python DEVICE=cuda:1 ./run_sd_batch_logged.sh
#   PYTHON_BIN=/venv/main/bin/python DEVICES="cuda:0 cuda:1" ./run_sd_batch_logged.sh

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PROMPTS_FILE="${PROMPTS_FILE:-prompts/hps_v2_all_eval.txt}"
OUTPUT_ROOT_DIR="${OUTPUT_DIR:-logs/sd_batch}"
CONFIG="${CONFIG:-pick}"
EVAL_REWARD="${EVAL_REWARD:-image_reward}"
DEVICE="${DEVICE:-cuda:1}"
DEVICES="${DEVICES:-}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

NUM_STEPS="${NUM_STEPS:-100}"
NUM_PARTICLES="${NUM_PARTICLES:-4}"
BATCH_P="${BATCH_P:-1}"
STEIN_STEP="${STEIN_STEP:-0.005}"
STEIN_LOOP="${STEIN_LOOP:-1}"
STEER_START="${STEER_START:-0}"
STEER_END="${STEER_END:-20}"
SAVE_INTERMEDIATE_REWARDS="${SAVE_INTERMEDIATE_REWARDS:-0}"

SAVE_INTERMEDIATE_REWARDS_ARG=""
if [[ "$SAVE_INTERMEDIATE_REWARDS" == "1" || "$SAVE_INTERMEDIATE_REWARDS" == "true" ]]; then
  SAVE_INTERMEDIATE_REWARDS_ARG="--save-intermediate-rewards"
fi

mkdir -p "$OUTPUT_ROOT_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR_NAME="${RUN_DIR_NAME:-batch_${TIMESTAMP}_$$}"
RUN_OUTPUT_DIR="$OUTPUT_ROOT_DIR/$RUN_DIR_NAME"
mkdir -p "$RUN_OUTPUT_DIR"
LOG_FILE="$RUN_OUTPUT_DIR/batch.log"

echo "[INFO] Starting batch run"
echo "[INFO] Output root: $OUTPUT_ROOT_DIR"
echo "[INFO] Run output: $RUN_OUTPUT_DIR"
echo "[INFO] Log file: $LOG_FILE"

echo "[INFO] Command:"
if [[ -n "$DEVICES" ]]; then
  echo "  $PYTHON_BIN runs/gradient_sd_batch.py --prompts-file $PROMPTS_FILE --config $CONFIG --negative-prompt \"$NEGATIVE_PROMPT\" --output-dir $RUN_OUTPUT_DIR --eval-reward $EVAL_REWARD --devices $DEVICES --num-steps $NUM_STEPS --num-particles $NUM_PARTICLES --batch-p $BATCH_P --stein-step $STEIN_STEP --stein-loop $STEIN_LOOP --steer-start $STEER_START --steer-end $STEER_END --verbose ${SAVE_INTERMEDIATE_REWARDS_ARG}"
else
  echo "  $PYTHON_BIN runs/gradient_sd_batch.py --prompts-file $PROMPTS_FILE --config $CONFIG --negative-prompt \"$NEGATIVE_PROMPT\" --output-dir $RUN_OUTPUT_DIR --eval-reward $EVAL_REWARD --device $DEVICE --num-steps $NUM_STEPS --num-particles $NUM_PARTICLES --batch-p $BATCH_P --stein-step $STEIN_STEP --stein-loop $STEIN_LOOP --steer-start $STEER_START --steer-end $STEER_END --verbose ${SAVE_INTERMEDIATE_REWARDS_ARG}"
fi

device_args=(--device "$DEVICE")
if [[ -n "$DEVICES" ]]; then
  # shellcheck disable=SC2206
  device_list=($DEVICES)
  device_args=(--devices "${device_list[@]}")
fi

"$PYTHON_BIN" runs/gradient_sd_batch.py \
  --prompts-file "$PROMPTS_FILE" \
  --config "$CONFIG" \
  --negative-prompt "$NEGATIVE_PROMPT" \
  --output-dir "$RUN_OUTPUT_DIR" \
  --eval-reward "$EVAL_REWARD" \
  "${device_args[@]}" \
  --num-steps "$NUM_STEPS" \
  --num-particles "$NUM_PARTICLES" \
  --batch-p "$BATCH_P" \
  --stein-step "$STEIN_STEP" \
  --stein-loop "$STEIN_LOOP" \
  --steer-start "$STEER_START" \
  --steer-end "$STEER_END" \
  --verbose \
  ${SAVE_INTERMEDIATE_REWARDS_ARG:+$SAVE_INTERMEDIATE_REWARDS_ARG} \
  2>&1 | tee "$LOG_FILE"

echo "[INFO] Finished. Full log: $LOG_FILE"
