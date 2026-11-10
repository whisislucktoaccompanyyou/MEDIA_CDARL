#!/usr/bin/env bash
# Mix kaggle / MESSIDOR / REFUGE / FIVES / DomainB_FP and train MEDIA_CDARL.
# The training process is detached from this SSH session.
# USER_LAUNCH=1 is required so a previously queued automatic start cannot fire.
set -u

if [[ "${USER_LAUNCH:-}" != "1" ]]; then
  echo "Refusing to start. Run:"
  echo "  USER_LAUNCH=1 bash \"$0\""
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="$(cd "$ROOT/../data" && pwd)"
DATASETS=(kaggle MESSIDOR REFUGE FIVES DomainB_FP)
MANIFEST="$ROOT/data/Dataset_Fundus/split_manifest.json"
CONFIG="$ROOT/config/train_mixed_fundus.json"
LOG_DIR="$ROOT/logs/mixed_launch"
PID_FILE="$LOG_DIR/train.pid"
LAUNCH_LOG="$LOG_DIR/launch.log"
GPU_ID="${GPU_ID:-0}"
PYTHON="${PYTHON:-python3}"

mkdir -p "$LOG_DIR"
# Keep the launcher runnable even if the filesystem refused chmod earlier.
chmod +x "$ROOT/scripts/start_mixed_train.sh" "$ROOT/scripts/build_mixed_manifest.py" 2>/dev/null || true

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Training is already running, pid ${old_pid}."
    echo "Log: $LAUNCH_LOG"
    echo "Stop it with: kill ${old_pid}"
    exit 1
  fi
fi

echo "[$(date '+%F %T')] building mixed manifest" | tee -a "$LAUNCH_LOG"
"$PYTHON" -I "$ROOT/scripts/build_mixed_manifest.py" \
  --data-root "$DATA_ROOT" \
  --out "$MANIFEST" \
  --datasets "${DATASETS[@]}" | tee -a "$LAUNCH_LOG"

label_n="$("$PYTHON" -I -c 'import glob,os,sys; print(len(glob.glob(os.path.join(sys.argv[1], "*.png"))))' "$ROOT/data/Dataset_Fundus/train/trainA")"
if [[ "${label_n}" -lt 1 ]]; then
  echo "No fractal labels in $ROOT/data/Dataset_Fundus/train/trainA" | tee -a "$LAUNCH_LOG"
  exit 1
fi
echo "fractal labels: ${label_n}" | tee -a "$LAUNCH_LOG"

cd "$ROOT"
# setsid + nohup: closing SSH does not deliver SIGHUP to the training process.
setsid nohup "$PYTHON" -u main.py -p train -c "$CONFIG" -gpu "$GPU_ID" \
  >> "$LAUNCH_LOG" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Training exited immediately. Last log lines:" | tee -a "$LAUNCH_LOG"
  tail -n 40 "$LAUNCH_LOG"
  exit 1
fi

echo "Started mixed training, pid ${pid}."
echo "SSH can be closed. The process is in its own session."
echo "Follow log:  tail -f $LAUNCH_LOG"
echo "Check pid:   kill -0 ${pid} && echo running"
echo "Stop:        kill ${pid}"
