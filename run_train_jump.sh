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
    exec 9>logs_launch/train_jump.lock
    flock -n 9 || { echo 'Jump training already running'; exit 1; }
    echo "$$" > logs_launch/train_jump.pid
    trap 'rm -f logs_launch/train_jump.pid' EXIT
    # Physical GPU 1. ReTree stays on GPU 0. Do not inherit another card.
    export CUDA_VISIBLE_DEVICES=1
    # Conditions are offline and unchanged. Do not rewrite 31612 maps on every launch.
    if ! "$PYTHON" -c "
import json, os, sys
report = 'data/Dataset_Kaggle/conditions/report.json'
if not os.path.isfile(report):
    sys.exit(1)
data = json.load(open(report))
ok = data.get('written') == 31612 and data.get('size') == 384
sys.exit(0 if ok else 1)
"; then
        "$PYTHON" -u prepare_conditions.py --size 384
    else
        echo 'conditions already complete, skip prepare'
    fi
    "$PYTHON" -u main.py -p train -c config/train_kaggle_jump.json
    echo 'Jump training completed'
    exit 0
fi
if [ -f logs_launch/train_jump.pid ] && kill -0 "$(cat logs_launch/train_jump.pid)" 2>/dev/null; then
    echo "Already running: PID $(cat logs_launch/train_jump.pid)"
    exit 0
fi
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs_launch/train_jump_$STAMP.log"
ln -sfn "$(basename "$LOG")" logs_launch/train_jump.log
setsid nohup sh ./run_train_jump.sh --worker > "$LOG" 2>&1 < /dev/null &
echo "Launched PID $!; log: $LOG"
sleep 2
if [ ! -f logs_launch/train_jump.pid ]; then
    echo "Worker failed to start; see $LOG" >&2
    exit 1
fi
