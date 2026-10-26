#!/usr/bin/env bash
# Run one discovered eval directly on a WDC self-hosted GPU runner.
set -Eeuo pipefail

ENTRY=${1:?usage: wdc_eval.sh ENTRY_JSON OUTPUT_DIR}
OUT=${2:?usage: wdc_eval.sh ENTRY_JSON OUTPUT_DIR}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"
RAW="$OUT/raw"
SPECULATORS_DIR=${SPECULATORS_DIR:?SPECULATORS_DIR must point to the Actions checkout}
VENV="$OUT/venv"
mkdir -p "$RAW"
test -f "$SPECULATORS_DIR/scripts/evaluate/evaluate.py"

readarray -t CONFIG < <(python3 - "$ENTRY" <<'PY'
import json
import sys

entry = json.load(open(sys.argv[1]))
for key in ("model", "gpus", "algorithm", "hf_revision"):
    print(entry[key])
PY
)
MODEL=${CONFIG[0]}
GPUS=${CONFIG[1]}
ALGORITHM=${CONFIG[2]}
HF_REVISION=${CONFIG[3]}
VLLM_VERSION=${VLLM_VERSION:-nightly}
SPECULATORS_REF=${SPECULATORS_REF:-main}

export HF_HOME=${HF_HOME:-/model-cache}
export UV_TORCH_BACKEND=${UV_TORCH_BACKEND:-cu130}
export UV_LINK_MODE=copy
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

nvidia-smi
python3 - "$GPUS" <<'PY'
import socket
import subprocess
import sys

expected = int(sys.argv[1])
visible = len(subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines())
if visible != expected:
    raise SystemExit(f"runner exposes {visible} GPUs; expected {expected}")
with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", 8000))
    except OSError as exc:
        raise SystemExit("port 8000 is already in use") from exc
PY

uv venv --seed --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" -e "$SPECULATORS_DIR" guidellm
if [ "$VLLM_VERSION" = nightly ]; then
    uv pip install --python "$VENV/bin/python" vllm \
        --extra-index-url https://wheels.vllm.ai/nightly/cu130
else
    uv pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION"
fi

SERVE_ARGS=(serve "$MODEL" --revision "$HF_REVISION" --port 8000)
if [ "$GPUS" -gt 1 ]; then
    SERVE_ARGS+=(--tensor-parallel-size "$GPUS")
fi
if [ "$ALGORITHM" = mtp ]; then
    SERVE_ARGS+=(--no-enable-chunked-prefill)
fi
EVAL_ARGS=(
    "$SPECULATORS_DIR/scripts/evaluate/evaluate.py"
    --target http://localhost:8000/v1
    --output-dir "$RAW"
    --max-requests 80
    sweep
)
"$VENV/bin/python" - "$OUT/provenance.json" "$SPECULATORS_DIR" \
    "$SPECULATORS_REF" "$UV_TORCH_BACKEND" "$VLLM_VERSION" \
    "${SERVE_ARGS[@]}" -- "${EVAL_ARGS[@]}" <<'PY'
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys

output, checkout, requested_ref, backend, requested_vllm = sys.argv[1:6]
serve_args = sys.argv[6 : sys.argv.index("--", 6)]
eval_args = sys.argv[sys.argv.index("--", 6) + 1 :]
serve_argv = ["vllm", *serve_args]
eval_argv = ["python", *eval_args]
provenance = {
    "checkpoint_sha": serve_args[serve_args.index("--revision") + 1],
    "speculators_ref": requested_ref,
    "speculators_sha": subprocess.check_output(
        ["git", "-C", checkout, "rev-parse", "HEAD"], text=True
    ).strip(),
    "speculators_version": importlib.metadata.version("speculators"),
    "vllm_requested": requested_vllm,
    "vllm_version": importlib.metadata.version("vllm"),
    "guidellm_version": importlib.metadata.version("guidellm"),
    "cuda_backend": backend,
    "serve_argv": serve_argv,
    "serve_command": shlex.join(serve_argv),
    "eval_argv": eval_argv,
    "eval_command": shlex.join(eval_argv),
    "num_speculative_tokens_source": "drafter config.json: speculators_config.proposal_methods",
    "benchmark": {"mode": "sweep", "max_requests": 80},
    "packages": subprocess.check_output(
        [os.path.join(os.path.dirname(sys.executable), "python"), "-m", "pip", "freeze"],
        text=True,
    ).splitlines(),
}
with open(output, "w") as handle:
    json.dump(provenance, handle, indent=2)
PY

VLLM_PID=""
cleanup() {
    if [ -n "$VLLM_PID" ]; then
        kill -TERM -- "-$VLLM_PID" 2>/dev/null || true
        sleep 5
        kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
record_failure() {
    local status=$?
    trap - ERR EXIT
    set +e
    cleanup
    python3 "$REPO_ROOT/pipeline/normalize.py" eval \
        --entry "$ENTRY" --status failed \
        --error "WDC eval failed at line $1 (exit $status); see job artifacts" \
        --provenance "$OUT/provenance.json" --output "$OUT/record.json"
    exit "$status"
}
trap 'record_failure $LINENO' ERR
trap cleanup EXIT

setsid "$VENV/bin/vllm" "${SERVE_ARGS[@]}" > "$OUT/vllm.log" 2>&1 &
VLLM_PID=$!

READY=0
for _ in $(seq 1 480); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        tail -50 "$OUT/vllm.log"
        false
    fi
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 5
done
if [ "$READY" != 1 ]; then
    tail -50 "$OUT/vllm.log"
    false
fi

timeout --signal=TERM --kill-after=30 "${EVAL_TIMEOUT:-5400}" \
    "$VENV/bin/python" "${EVAL_ARGS[@]}" > "$OUT/evaluate.log" 2>&1 || {
        tail -50 "$OUT/evaluate.log"
        false
    }

DEPLOY_COMMAND=$(python3 - "$OUT/provenance.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["serve_command"])
PY
)
python3 "$REPO_ROOT/pipeline/normalize.py" eval --entry "$ENTRY" --status ok \
    --raw-dir "$RAW" --deploy-command "$DEPLOY_COMMAND" \
    --provenance "$OUT/provenance.json" --output "$OUT/record.json"
