#!/usr/bin/env bash
# Runner script for manual evals: serves vLLM, polls /health, runs evaluate.py, copies results, cleans up.
# Usage:
#   pipeline/eval_runner.sh <slug> <port> <cuda_devices> <raw_dir> <deploy_cmd...>

set -eo pipefail

SLUG="$1"
PORT="$2"
CUDA_DEVICES="$3"
RAW_DIR="$4"
shift 4
DEPLOY_CMD=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep caller-provided relative paths rooted at the dashboard repo. The eval
# changes into SPECULATORS_DIR below, so leaving RAW_DIR relative would copy
# results into the wrong checkout.
case "$RAW_DIR" in
    /*) ;;
    *) RAW_DIR="$REPO_ROOT/$RAW_DIR" ;;
esac
RAW_PARENT="$(dirname "$RAW_DIR")"
RAW_NAME="$(basename "$RAW_DIR")"
source "$REPO_ROOT/local/eval-manual.env"
export PATH="$VLLM_VENV/bin:$PATH"
export HF_HOME="${HF_HOME:-/data/fast/huggingface}"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR" "$RAW_PARENT"
LOG_FILE="$LOG_DIR/eval-manual-$SLUG.log"
# Keep live artifacts on the same filesystem as RAW_DIR so they survive runner
# failures and can be promoted without copying after the sweep finishes.
TMP_OUT=$(mktemp -d "$RAW_PARENT/.$RAW_NAME.in-progress-XXXXXX")

echo "=== Eval Runner for $SLUG ===" | tee -a "$LOG_FILE"
echo "Port: $PORT" | tee -a "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_DEVICES" | tee -a "$LOG_FILE"
echo "Deploy Command: ${DEPLOY_CMD[*]}" | tee -a "$LOG_FILE"
echo "Raw Dir: $RAW_DIR" | tee -a "$LOG_FILE"
echo "TMP_OUT: $TMP_OUT" | tee -a "$LOG_FILE"

# Start vllm serve
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
echo "Starting vLLM server..." | tee -a "$LOG_FILE"

"${DEPLOY_CMD[@]}" >> "$LOG_FILE" 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID" | tee -a "$LOG_FILE"

cleanup() {
    echo "Cleaning up vLLM process $VLLM_PID..." | tee -a "$LOG_FILE"
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    sleep 5
    kill -9 "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    if [ -n "$TMP_OUT" ] && [ -d "$TMP_OUT" ]; then
        echo "Preserving evaluation output at $TMP_OUT" | tee -a "$LOG_FILE"
    fi
}
trap cleanup EXIT

# Poll health
echo "Waiting for vLLM server on port $PORT to become ready (up to 40 min)..." | tee -a "$LOG_FILE"
READY=0
for i in $(seq 1 480); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server died unexpectedly!" | tee -a "$LOG_FILE"
        tail -n 40 "$LOG_FILE"
        exit 1
    fi
    sleep 5
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: vLLM server timed out waiting for health check" | tee -a "$LOG_FILE"
    tail -n 40 "$LOG_FILE"
    exit 1
fi

echo "vLLM server is healthy and ready!" | tee -a "$LOG_FILE"

# Run evaluate.py
echo "Running evaluation sweep..." | tee -a "$LOG_FILE"
cd "$SPECULATORS_DIR/scripts/evaluate"
PATH="$GUIDELLM_VENV/bin:$PATH" "$GUIDELLM_VENV/bin/python" evaluate.py \
    --target "http://localhost:${PORT}/v1" \
    --output-dir "$TMP_OUT" \
    --max-requests 80 \
    sweep >> "$LOG_FILE" 2>&1 || {
        echo "ERROR: evaluate.py sweep failed!" | tee -a "$LOG_FILE"
        tail -n 40 "$LOG_FILE"
        exit 1
    }

echo "Evaluation sweep finished successfully. Promoting results to $RAW_DIR..." | tee -a "$LOG_FILE"
BACKUP_DIR=""
if [ -e "$RAW_DIR" ]; then
    BACKUP_DIR=$(mktemp -d "$RAW_PARENT/.$RAW_NAME.previous-XXXXXX")
    rmdir "$BACKUP_DIR"
    mv "$RAW_DIR" "$BACKUP_DIR"
fi
if ! mv "$TMP_OUT" "$RAW_DIR"; then
    echo "ERROR: failed promoting evaluation results to $RAW_DIR" | tee -a "$LOG_FILE"
    if [ -n "$BACKUP_DIR" ] && [ -e "$BACKUP_DIR" ]; then
        mv "$BACKUP_DIR" "$RAW_DIR" || true
    fi
    exit 1
fi
TMP_OUT=""
if [ -n "$BACKUP_DIR" ]; then
    rm -rf "$BACKUP_DIR" || echo "WARNING: could not remove previous raw results at $BACKUP_DIR" | tee -a "$LOG_FILE"
fi
echo "Files copied to $RAW_DIR:" | tee -a "$LOG_FILE"
ls -la "$RAW_DIR" | tee -a "$LOG_FILE"

echo "=== SUCCESS: Eval for $SLUG completed ===" | tee -a "$LOG_FILE"
