#!/bin/sh
# Third MEDIA_CDARL run: Kaggle images with EyeQ quality == 0 only.
# Does not touch Dataset_Kaggle, train_kaggle.json, or any existing experiment dir.
set -eu
cd "$(dirname "$0")"
mkdir -p logs_launch
PYTHON=/home/WVK/anaconda3/envs/medsam2/bin/python
LOCK=logs_launch/train_kaggle_eyeq_q0.lock
PIDFILE=logs_launch/train_kaggle_eyeq_q0.pid

if [ ! -x "$PYTHON" ]; then
    echo "Python missing: $PYTHON" >&2
    exit 1
fi

pick_gpu() {
    "$PYTHON" - <<'PY'
import subprocess
rows = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
], text=True)
free = []
for line in rows.splitlines():
    index, used, util = [part.strip() for part in line.split(",")]
    if int(used) < 2000 and int(util) < 10:
        free.append(index)
if not free:
    raise SystemExit("no free GPU; not starting")
print(free[0])
PY
}

if [ "${1:-}" = "--worker" ]; then
    exec 9>"$LOCK"
    flock -n 9 || { echo "EyeQ q0 pipeline already running"; exit 1; }
    echo "$$" > "$PIDFILE"
    trap 'rm -f "$PIDFILE"' EXIT
    GPU=$(pick_gpu)
    echo "EyeQ q0 using physical GPU $GPU"
    "$PYTHON" -u prepare_kaggle_eyeq_q0.py
    "$PYTHON" -u main.py -p train -c config/train_kaggle_eyeq_q0.json -gpu "$GPU"
    echo "EyeQ q0 training completed"
    exit 0
fi

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Already running: PID $(cat "$PIDFILE")"
    exit 0
fi
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs_launch/train_kaggle_eyeq_q0_$STAMP.log"
ln -sfn "$(basename "$LOG")" logs_launch/train_kaggle_eyeq_q0.log
setsid nohup sh ./run_train_kaggle_eyeq_q0.sh --worker > "$LOG" 2>&1 < /dev/null &
echo "Launched PID $!; log: $LOG"
sleep 2
if [ ! -f "$PIDFILE" ]; then
    echo "Worker failed to start; see $LOG" >&2
    exit 1
fi
