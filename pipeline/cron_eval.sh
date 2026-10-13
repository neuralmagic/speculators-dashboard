#!/usr/bin/env bash
# pipeline/cron_eval.sh - WDC cron job: pick up eval:pending GitHub issues
# and run evaluations via orchestrate.py.
#
# Prerequisites:
#   - gh CLI authenticated (PAT in env)
#   - Run from the repo root (or set REPO_ROOT)
#   - Python 3 with pipeline dependencies on PATH
#
# Safe to interrupt: a crash mid-eval leaves the issue labeled eval:running,
# which is never auto-picked-up (requires manual label reset).
set -euo pipefail

REPO="neuralmagic/speculators-dashboard"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PENDING_DIR="$REPO_ROOT/results/pending"
LOG_DIR="$REPO_ROOT/logs"

mkdir -p "$PENDING_DIR" "$LOG_DIR"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

slugify() {
    # Match discover.py: take text after last /, lowercase, replace non-alnum
    # runs with single dash, strip leading/trailing dashes.
    echo "$1" \
        | awk -F/ '{print $NF}' \
        | tr '[:upper:]' '[:lower:]' \
        | sed 's/[^a-z0-9]\+/-/g; s/^-\+//; s/-\+$//'
}

parse_field() {
    # Extract a value from a markdown table row.
    #   | Field | `value` |   ->   value
    # Handles both backtick-wrapped and plain values.
    local body="$1" field="$2"
    echo "$body" \
        | grep -i "| *${field} *|" \
        | head -1 \
        | awk -F'|' '{print $3}' \
        | sed 's/`//g; s/^ *//; s/ *$//'
}

# ── Phase 1: Retry handling ──────────────────────────────────────────────────
# On each cron tick, eval:failed issues WITHOUT retry:1 get flipped back to
# eval:pending (with retry:1 added) for one automatic retry.  Issues that
# already carry retry:1 stay eval:failed for manual investigation.
log "Phase 1: checking for retryable failed evals"

failed_json=$(gh issue list --repo "$REPO" --label "eval:failed" \
    --json number,labels --limit 50 2>/dev/null || echo "[]")
failed_count=$(echo "$failed_json" | jq 'length')

for i in $(seq 0 $((failed_count - 1))); do
    number=$(echo "$failed_json" | jq -r ".[$i].number")
    has_retry=$(echo "$failed_json" | jq \
        ".[$i].labels | [.[].name] | any(. == \"retry:1\")")

    if [ "$has_retry" = "false" ]; then
        log "Queuing retry for issue #$number"
        gh issue edit "$number" --repo "$REPO" \
            --remove-label "eval:failed" \
            --add-label "eval:pending" \
            --add-label "retry:1"
    else
        log "Issue #$number already retried (retry:1 present), skipping"
    fi
done

# ── Phase 2: Process pending evals (one at a time) ───────────────────────────
log "Phase 2: discovering eval:pending issues"

pending_json=$(gh issue list --repo "$REPO" --label "eval:pending" \
    --json number,title,body --limit 50 2>/dev/null || echo "[]")
pending_count=$(echo "$pending_json" | jq 'length')
log "Found $pending_count pending issue(s)"

for i in $(seq 0 $((pending_count - 1))); do
    number=$(echo "$pending_json" | jq -r ".[$i].number")
    title=$(echo "$pending_json"  | jq -r ".[$i].title")
    body=$(echo "$pending_json"   | jq -r ".[$i].body")

    log "Processing issue #$number: $title"

    # ── Parse the structured markdown table ───────────────────────────────
    MODEL=$(parse_field "$body" "Model")
    TARGET=$(parse_field "$body" "Target")
    ALGORITHM=$(parse_field "$body" "Algorithm")
    SPEC_TOKENS=$(parse_field "$body" "Speculative Tokens")
    GPUS_RAW=$(parse_field "$body" "GPUs")
    HF_MODIFIED=$(parse_field "$body" "HF Last Modified")

    # "4xh100" -> GPU_COUNT=4, GPU_TYPE=h100
    GPU_COUNT=$(echo "$GPUS_RAW" | sed 's/[xX].*//')
    GPU_TYPE=$(echo "$GPUS_RAW" | sed 's/.*[xX]//')

    SLUG=$(slugify "$MODEL")

    # Validate required fields
    if [ -z "$MODEL" ] || [ -z "$ALGORITHM" ] || [ -z "$GPU_COUNT" ] || [ -z "$GPU_TYPE" ]; then
        log "ERROR: could not parse issue #$number body, skipping"
        gh issue comment "$number" --repo "$REPO" \
            --body "Cron: failed to parse required fields from issue body. Please check the markdown table format." \
            || true
        continue
    fi

    # num_speculative_tokens may be absent or non-numeric -> JSON null
    if echo "$SPEC_TOKENS" | grep -qE '^[0-9]+$'; then
        SPEC_TOKENS_JSON="$SPEC_TOKENS"
    else
        SPEC_TOKENS_JSON="null"
    fi

    # ── Step a: Claim the issue (eval:pending -> eval:running) ────────────
    log "Claiming issue #$number"
    gh issue edit "$number" --repo "$REPO" \
        --remove-label "eval:pending" \
        --add-label "eval:running"

    # ── Step b: Write pending entry JSON for orchestrate.py ───────────────
    ENTRY_PATH="$PENDING_DIR/${SLUG}.json"
    cat > "$ENTRY_PATH" <<ENTRY_JSON
{
  "model": "${MODEL}",
  "slug": "${SLUG}",
  "target": "${TARGET}",
  "algorithm": "${ALGORITHM}",
  "num_speculative_tokens": ${SPEC_TOKENS_JSON},
  "gpus": ${GPU_COUNT},
  "gpu_type": "${GPU_TYPE}",
  "hf_last_modified": "${HF_MODIFIED}"
}
ENTRY_JSON

    # ── Step c: Run the eval ──────────────────────────────────────────────
    log "Running eval for $MODEL (${GPU_COUNT}x${GPU_TYPE})"
    eval_exit=0
    python3 "$SCRIPT_DIR/orchestrate.py" single --entry "$ENTRY_PATH" 2>&1 \
        | tee "$LOG_DIR/eval-${SLUG}.log" \
        || eval_exit=$?

    if [ "$eval_exit" -eq 0 ]; then
        # ── Success ───────────────────────────────────────────────────────
        log "Eval succeeded for $MODEL"

        # Commit and push results
        git_exit=0
        {
            cd "$REPO_ROOT"
            git add results.json
            git pull --rebase origin main
            git commit -m "Eval: $MODEL"
            git push
        } || git_exit=$?

        if [ "$git_exit" -ne 0 ]; then
            log "WARNING: git push failed (exit $git_exit), results committed locally"
        fi

        # Build a metrics summary from results.json
        SUMMARY=$(python3 -c "
import json, sys
data = json.loads(open('${REPO_ROOT}/results.json').read())
for m in data.get('models', []):
    if m['model'] == '${MODEL}' and m['status'] == 'ok':
        met = m['metrics']
        print(f'Acceptance length: {met[\"acceptance_length\"]}')
        print(f'Throughput: {met[\"throughput_tps\"]} tok/s')
        print(f'Speedup: {met[\"speedup\"]}x')
        sys.exit(0)
print('Eval completed successfully.')
" 2>/dev/null || echo "Eval completed successfully.")

        gh issue comment "$number" --repo "$REPO" --body "$(cat <<COMMENT_EOF
Eval completed successfully.

\`\`\`
$SUMMARY
\`\`\`
COMMENT_EOF
)" || true

        gh issue edit "$number" --repo "$REPO" \
            --remove-label "eval:running" \
            --add-label "eval:done" || true
        gh issue close "$number" --repo "$REPO" || true

        log "Issue #$number closed with eval:done"
    else
        # ── Failure ───────────────────────────────────────────────────────
        log "Eval FAILED for $MODEL (exit code $eval_exit)"

        has_retry=$(gh issue view "$number" --repo "$REPO" --json labels \
            | jq '[.labels[].name] | any(. == "retry:1")')

        gh issue edit "$number" --repo "$REPO" \
            --remove-label "eval:running" \
            --add-label "eval:failed"

        LAST_LINES=$(tail -20 "$LOG_DIR/eval-${SLUG}.log" 2>/dev/null \
            || echo "(no log available)")

        if [ "$has_retry" = "true" ]; then
            gh issue comment "$number" --repo "$REPO" --body "$(cat <<COMMENT_EOF
Eval failed on retry (exit code $eval_exit). Manual investigation needed.

<details><summary>Last 20 log lines</summary>

\`\`\`
$LAST_LINES
\`\`\`

</details>
COMMENT_EOF
)" || true
        else
            gh issue comment "$number" --repo "$REPO" --body "$(cat <<COMMENT_EOF
Eval failed (exit code $eval_exit). Will retry on next cron tick.

<details><summary>Last 20 log lines</summary>

\`\`\`
$LAST_LINES
\`\`\`

</details>
COMMENT_EOF
)" || true
        fi

        log "Issue #$number marked eval:failed"
    fi

    # Clean up the temporary pending entry
    rm -f "$ENTRY_PATH"
done

log "Cron eval run complete."
