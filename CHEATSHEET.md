# Cheatsheet — running the dashboard pipeline

## WDC GitHub Actions

Preview the exact H100 execution plan without using GPUs:

```bash
python3 pipeline/discover.py --gha
cat results/pending/gha-matrix.json
python3 pipeline/discover.py --gha --model RedHatAI/<drafter>  # force one model
```

In GitHub, open **Actions → Evaluate speculators on WDC → Run workflow**.
Enter the full drafter ID to force one model and leave `publish` disabled for a
branch smoke test. The plan, per-model raw metrics/logs, and merged dashboard
JSON are separate run artifacts. Set `publish` only when intentionally updating
`results.json` on the selected branch.

## Legacy OpenShift commands

These commands are meant for the machine that has `oc` access to the OpenShift
cluster and the `devenv` repo checked out (i.e. where `devenv` normally works).
Nothing here requires a GPU locally.

## 0. Prerequisites

```bash
make setup    # verifies all of the below AND installs the eval scheduler
```

or by hand:

```bash
oc whoami                                    # must be logged in (else: oc login <cluster-url>)
gh auth status                               # cron_eval.sh files/updates eval issues
command -v flock python3 curl                # required tools
export DEVENV_DIR=~/projects/devenv          # path to the devenv checkout (adjust)
ls "$DEVENV_DIR/launch.sh"
```

Clone this repo next to it:

```bash
cd ~/projects
git clone <your-github>/speculators-dashboard.git
cd speculators-dashboard
```

## 1. Preview what would be evaluated (no cluster needed)

```bash
python3 pipeline/discover.py
cat results/pending/index.json               # slugs pending eval
ls results/pending/                          # per-model entries incl. GPU sizing
```

## 2. Smoke test — evaluate ONE small model end-to-end

```bash
python3 pipeline/discover.py                 # regenerate pending entries
./pipeline/refresh.sh single --entry results/pending/qwen3-8b-speculator-eagle3.json
```

Watch for: pod `devenv-$USER-eval` becomes Ready → vLLM serves the speculator →
acceptance metrics printed → `results.json` gets the entry → **pod is deleted**.
Verify teardown:

```bash
oc get pods -n machine-learning | grep eval  # should be empty
python3 -m json.tool results.json | less
```

## 3. View the website

```bash
cd ~/projects/speculators-dashboard
python3 -m http.server 8000
# open http://localhost:8000/site/index.html
# (or from your laptop: ssh -L 8000:localhost:8000 <this-machine>)
```

## 4. Full refresh (all pending models, sequential)

```bash
./pipeline/refresh.sh                        # logs to stdout; flock-guarded
tail -f logs/cron.log                        # when running via the scheduler
```

Each model takes ~15–45 min (model download + server start + eval). The shared
HF cache PVC makes re-runs much faster.

## 5. Schedule it (automated evals)

```bash
make setup    # preflight checks + installs the scheduler (systemd user timer,
              # or a flock-guarded crontab entry where systemd is unavailable)
```

The scheduler runs `pipeline/cron_eval.sh` every 30 minutes: it picks up
GitHub issues labeled `eval:pending` and evaluates them one at a time. See the
README's **Automation** section for verify/watch commands and edge-case notes.

## 6. Useful knobs

```bash
EVAL_TIMEOUT=7200 ./pipeline/refresh.sh single --entry <entry.json>   # longer cap for huge models
DEVENV_NAMESPACE=machine-learning                                      # default namespace
```

## 7. Troubleshooting

| Symptom | Check |
|---|---|
| `Not logged into OpenShift` | `oc login <cluster-url>` |
| Pod stays Pending / Unschedulable | `oc describe pod devenv-$USER-eval -n machine-learning` — likely no free GPUs of that type; check `devenv-status` |
| vLLM dies at startup | results dir has `raw/vllm.log`; often means the sizing table under-allocated GPUs — bump it in `pipeline/discover.py` (`SIZING_OVERRIDES`) |
| Eval failed entry on dashboard | `results/<slug>/raw/evaluate.log` and `vllm.log` |
| Leftover eval pod after crash | `printf 'y\n' \| $DEVENV_DIR/launch.sh --name eval --down` (refresh.sh also sweeps pods older than 4h) |
| Re-run a model that already has results | `python3 -c "import json; d=json.load(open('results.json')); d['models']=[m for m in d['models'] if m['model']!='<MODEL_ID>']; json.dump(d, open('results.json','w'), indent=2)"` then `python3 pipeline/discover.py` |

## 8. Publish updated results

```bash
cd ~/projects/speculators-dashboard
git add results.json results/
git commit -m "eval results $(date -u +%F)"
git push
```
