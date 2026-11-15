#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Supply verified original 2-channel G/H weights, OR a new complete resume prefix.
# Example: PRETRAINED=/path/I100000_E4 GPU=0 bash run_train_refine_v1.sh
# Resume:  RESUME=/path/I1000_E1 GPU=0 bash run_train_refine_v1.sh
if [[ -n "${PRETRAINED:-}" && -n "${RESUME:-}" ]]; then
  echo 'PRETRAINED 和 RESUME 只能设置一个' >&2
  exit 1
fi
if [[ -z "${PRETRAINED:-}${RESUME:-}" ]]; then
  echo '请提供已核实的 PRETRAINED 原模型前缀或 RESUME 新实验前缀' >&2
  exit 1
fi
mkdir -p logs_launch
stamp=$(date +%Y%m%d_%H%M%S)
launch_args=()
if [[ -n "${PRETRAINED:-}" ]]; then
  launch_args+=(--pretrained "$PRETRAINED")
else
  launch_args+=(--resume "$RESUME")
fi
python -u main.py --config config/train_kaggle_refine_v1.json --phase train \
  --gpu_ids "${GPU:-0}" "${launch_args[@]}" \
  2>&1 | tee "logs_launch/refine_v1_${stamp}.log"
