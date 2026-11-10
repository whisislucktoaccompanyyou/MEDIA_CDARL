#!/bin/sh
# Detach C-DARL fundus training from the SSH session.
# nohup ignores hangup; setsid starts a new session so closing the terminal does not stop it.
cd "$(dirname "$0")" || exit 1
mkdir -p logs_launch
LOG="logs_launch/train_fundus.log"
PIDFILE="logs_launch/train_fundus.pid"

if [ -f "$PIDFILE" ]; then
    OLD=$(cat "$PIDFILE")
    if kill -0 "$OLD" 2>/dev/null; then
        echo "training already running, pid=$OLD"
        echo "log: $LOG"
        exit 1
    fi
fi

PYTHON="/home/WVK/anaconda3/envs/medsam2/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "medsam2 python not found: $PYTHON"
    exit 1
fi

# Truncate the previous failed log so a new launch is not mistaken for the old traceback.
: > "$LOG"
setsid nohup "$PYTHON" -u main.py -p train -c config/train_fundus.json > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "started pid=$(cat "$PIDFILE")"
echo "log: $(pwd)/$LOG"
echo "disconnecting SSH will not stop this process"
