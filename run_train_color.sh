#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p logs_launch
PYTHON=/home/WVK/anaconda3/envs/medsam2/bin/python
if [ ! -x "$PYTHON" ]; then
    echo "Python missing: $PYTHON" >&2
    exit 1
fi
echo 'Color head is off. The generator is not frozen. Use run_train_jump.sh.' >&2
exit 1
if [ "${1:-}" = "--worker" ]; then
    exec 9>logs_launch/train_color.lock
    flock -n 9 || { echo 'Color head already running'; exit 1; }
    echo "$$" > logs_launch/train_color.pid
    trap 'rm -f logs_launch/train_color.pid' EXIT
    "$PYTHON" -u train_color.py \
        -c config/train_kaggle.json \
        --gpu 1 \
        --weight experiments/CDARL_kaggle_quality_261110_223239/checkpoint/I110000_E5 \
        --steps 5000 \
        --lr 1e-5 \
        --save_every 1000 \
        --preview_every 500
    echo 'Color head training completed'
    exit 0
fi
if [ -f logs_launch/train_color.pid ] && kill -0 "$(cat logs_launch/train_color.pid)" 2>/dev/null; then
    echo "Already running: PID $(cat logs_launch/train_color.pid)"
    exit 0
fi
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs_launch/train_color_$STAMP.log"
ln -sfn "$(basename "$LOG")" logs_launch/train_color.log
setsid nohup sh ./run_train_color.sh --worker > "$LOG" 2>&1 < /dev/null &
echo "Launched PID $!; log: $LOG"
sleep 2
if [ ! -f logs_launch/train_color.pid ]; then
    echo "Worker failed to start; see $LOG" >&2
    exit 1
fi
