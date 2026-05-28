#!/usr/bin/env sh
set -eu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-12}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-12}"
export TASK2_THRESHOLD="${TASK2_THRESHOLD:-0.35}"
export TASK2_AUTOZOOM_ENABLE="${TASK2_AUTOZOOM_ENABLE:-1}"
export TASK2_AUTOZOOM_MAX_PASSES="${TASK2_AUTOZOOM_MAX_PASSES:-2}"
export TASK2_AUTOZOOM_MIN_PASSES="${TASK2_AUTOZOOM_MIN_PASSES:-1}"
export TASK2_AUTOZOOM_GROWTH_ZYX="${TASK2_AUTOZOOM_GROWTH_ZYX:-1.25,1.15,1.15}"
export TASK2_AUTOZOOM_SCALE_MODE="${TASK2_AUTOZOOM_SCALE_MODE:-adaptive}"
export TASK2_AUTOZOOM_REFINE_DISABLE="${TASK2_AUTOZOOM_REFINE_DISABLE:-1}"
export PYTHONUNBUFFERED=1

extra_args=""
case "${TASK2_ALLOW_UNTRAINED_SMOKE:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    extra_args="--allow-untrained-smoke"
    ;;
esac

"${PYTHON:-python3}" -m lett_next.submission_cpu \
  --inputs "${TASK2_INPUT_DIR:-/workspace/inputs}" \
  --outputs "${TASK2_OUTPUT_DIR:-/workspace/outputs}" \
  --checkpoint "${TASK2_CHECKPOINT:-/workspace/model/checkpoint.pt}" \
  --model-name "${TASK2_MODEL_NAME:-mednextv2_f32}" \
  --crop-size-zyx "${TASK2_CROP_SIZE_ZYX:-128,160,160}" \
  --prompt-sigma "${TASK2_PROMPT_SIGMA:-2.0}" \
  --threshold "${TASK2_THRESHOLD}" \
  --threads "${TASK2_CPU_THREADS:-${OMP_NUM_THREADS}}" \
  --time-limit-seconds "${TASK2_TIME_LIMIT_SECONDS:-60}" \
  --output-format "${TASK2_OUTPUT_FORMAT:-nii}" \
  --autozoom-min-passes "${TASK2_AUTOZOOM_MIN_PASSES}" \
  --autozoom-scale-mode "${TASK2_AUTOZOOM_SCALE_MODE}" \
  ${extra_args}

case "${TASK2_KEEP_RUN_METADATA:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    ;;
  *)
    rm -f \
      "${TASK2_OUTPUT_DIR:-/workspace/outputs}/prediction_log.jsonl" \
      "${TASK2_OUTPUT_DIR:-/workspace/outputs}/prediction_summary.json"
    ;;
esac
