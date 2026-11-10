#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p logs_launch
PYTHON=/home/WVK/anaconda3/envs/medsam2/bin/python
if [ ! -x "$PYTHON" ]; then
    echo "Python missing: $PYTHON" >&2
    exit 1
fi
if [ "${1:-}" = "--worker" ]; then
    exec 9>logs_launch/train_kaggle.lock
    flock -n 9 || { echo 'Kaggle pipeline already running'; exit 1; }
    echo "$$" > logs_launch/train_kaggle.pid
    trap 'rm -f logs_launch/train_kaggle.pid' EXIT
    "$PYTHON" -u prepare_kaggle.py
    "$PYTHON" -u main.py -p train -c config/train_kaggle.json
    echo 'Kaggle training completed'
    exit 0
fi
if [ -f logs_launch/train_kaggle.pid ] && kill -0 "$(cat logs_launch/train_kaggle.pid)" 2>/dev/null; then
    echo "Already running: PID $(cat logs_launch/train_kaggle.pid)"
    exit 0
fi
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs_launch/train_kaggle_$STAMP.log"
ln -sfn "$(basename "$LOG")" logs_launch/train_kaggle.log
setsid nohup sh ./run_train_kaggle.sh --worker > "$LOG" 2>&1 < /dev/null &
echo "Launched PID $!; log: $LOG"
# Allow the detached worker to acquire its lock before the launcher exits.
sleep 2
if [ ! -f logs_launch/train_kaggle.pid ]; then
    echo "Worker failed to start; see $LOG" >&2
    exit 1
fi
