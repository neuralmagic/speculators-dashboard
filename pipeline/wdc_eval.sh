#!/usr/bin/env bash
# Run one discovered eval directly on a WDC self-hosted GPU runner.
set -Eeuo pipefail

ENTRY=${1:?usage: wdc_eval.sh ENTRY_JSON OUTPUT_DIR}
OUT=${2:?usage: wdc_eval.sh ENTRY_JSON OUTPUT_DIR}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"
RAW="$OUT/raw"
SPECULATORS_DIR="$OUT/speculators"
VENV="$OUT/venv"
mkdir -p "$RAW"

readarray -t CONFIG < <(python3 - "$ENTRY" <<'PY'
import json
import sys

entry = json.load(open(sys.argv[1]))
for key in ("model", "gpus", "algorithm"):
    print(entry[key])
PY
)
MODEL=${CONFIG[0]}
GPUS=${CONFIG[1]}
ALGORITHM=${CONFIG[2]}
SPECULATORS_REF=${SPECULATORS_REF:-main}
VLLM_VERSION=${VLLM_VERSION:-nightly}

export HF_HOME=${HF_HOME:-/model-cache}
export UV_TORCH_BACKEND=${UV_TORCH_BACKEND:-cu130}
export UV_LINK_MODE=copy
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

nvidia-smi
python3 - "$GPUS" <<'PY'
import subprocess
import sys

expected = int(sys.argv[1])
visible = len(subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines())
if visible != expected:
    raise SystemExit(f"runner exposes {visible} GPUs; expected {expected}")
PY

git clone --filter=blob:none --no-checkout \
    https://github.com/vllm-project/speculators.git "$SPECULATORS_DIR"
git -C "$SPECULATORS_DIR" fetch --depth 1 origin "$SPECULATORS_REF"
git -C "$SPECULATORS_DIR" checkout --detach FETCH_HEAD
uv venv --seed --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" -e "$SPECULATORS_DIR" guidellm
if [ "$VLLM_VERSION" = nightly ]; then
    uv pip install --python "$VENV/bin/python" vllm \
        --extra-index-url https://wheels.vllm.ai/nightly/cu130
else
    uv pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION"
fi

"$VENV/bin/python" - "$ENTRY" "$OUT/provenance.json" "$SPECULATORS_DIR" <<'PY'
import importlib.metadata
import json
import subprocess
import sys
import urllib.request

entry = json.load(open(sys.argv[1]))
with urllib.request.urlopen(
    f"https://huggingface.co/api/models/{entry['model']}", timeout=30
) as response:
    checkpoint = json.load(response).get("sha")
provenance = {
    "checkpoint_sha": checkpoint,
    "speculators_ref": sys.argv[3],
    "speculators_sha": subprocess.check_output(
        ["git", "-C", sys.argv[3], "rev-parse", "HEAD"], text=True
    ).strip(),
    "speculators_version": importlib.metadata.version("speculators"),
    "vllm_version": importlib.metadata.version("vllm"),
    "num_speculative_tokens_source": "drafter config.json: speculators_config.proposal_methods",
    "benchmark": {"mode": "sweep", "max_requests": 80},
}
json.dump(provenance, open(sys.argv[2], "w"), indent=2)
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

SERVE_ARGS=(--port 8000)
if [ "$GPUS" -gt 1 ]; then
    SERVE_ARGS+=(--tensor-parallel-size "$GPUS")
fi
if [ "$ALGORITHM" = mtp ]; then
    SERVE_ARGS+=(--no-enable-chunked-prefill)
fi
setsid "$VENV/bin/vllm" serve "$MODEL" "${SERVE_ARGS[@]}" > "$OUT/vllm.log" 2>&1 &
VLLM_PID=$!

READY=0
for _ in $(seq 1 480); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        READY=1
        break
    fi
    kill -0 "$VLLM_PID" 2>/dev/null || {
        tail -50 "$OUT/vllm.log"
        exit 1
    }
    sleep 5
done
if [ "$READY" != 1 ]; then
    tail -50 "$OUT/vllm.log"
    exit 1
fi

timeout --signal=TERM "${EVAL_TIMEOUT:-5400}" \
    "$VENV/bin/python" "$SPECULATORS_DIR/scripts/evaluate/evaluate.py" \
    --target http://localhost:8000/v1 --output-dir "$RAW" \
    --max-requests 80 sweep > "$OUT/evaluate.log" 2>&1 || {
        tail -50 "$OUT/evaluate.log"
        exit 1
    }

python3 "$REPO_ROOT/pipeline/deploy_agent.py" --model "$MODEL" \
    --target "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("target") or "")' "$ENTRY")" \
    --algorithm "$ALGORITHM" --gpus "$GPUS" > "$OUT/deploy.json"
readarray -t DEPLOY < <(python3 - "$OUT/deploy.json" <<'PY'
import json
import sys

deploy = json.load(open(sys.argv[1]))
for key in ("command", "recipe_source", "recipe_model"):
    print(deploy.get(key) or "")
PY
)
python3 "$REPO_ROOT/pipeline/normalize.py" eval --entry "$ENTRY" --status ok \
    --raw-dir "$RAW" --deploy-command "${DEPLOY[0]}" \
    --deploy-recipe-source "${DEPLOY[1]}" --deploy-recipe-model "${DEPLOY[2]}" \
    --provenance "$OUT/provenance.json" --output "$OUT/record.json"
