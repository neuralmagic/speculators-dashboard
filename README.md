# Speculators Dashboard

A SpecBundle-style dashboard for speculative-decoding draft models ("speculators")
from the HuggingFace collection
[RedHatAI/speculator-models](https://huggingface.co/collections/RedHatAI/speculator-models).

An automated pipeline watches the collection, evaluates new/updated speculators on
our OpenShift cluster (GPU pods provisioned with
[devenv](https://github.com/neuralmagic/devenv), models served with
[vLLM](https://github.com/vllm-project/vllm), acceptance metrics measured with the
[speculators](https://github.com/vllm-project/speculators) eval harness), stores
results in `results.json`, and tears the GPU pod down. The site in `site/` renders
`results.json` as a static page.

## Quick start

Requires Python 3.11+ (pinned in `.python-version`).

```bash
make install   # create .venv and install the pipeline + dev tooling
make dev       # serve the dashboard at http://localhost:8000
```

That's the whole local setup — the pipeline is stdlib-only, so there are no
runtime dependencies to resolve.

On the cluster-connected machine (where `oc` and the `devenv` checkout live),
starting automated evals is two commands:

```bash
git clone https://github.com/neuralmagic/speculators-dashboard.git && cd speculators-dashboard
make setup   # preflight checks (oc/gh/devenv) + installs the eval scheduler
```

`make setup` is idempotent — re-run it any time to re-check or repair the setup.
See **CHEATSHEET.md** for manual/smoke-test commands.

All common tasks are Makefile targets:

| Target | What it does |
|---|---|
| `make install` | Create `.venv`, `pip install -e ".[dev]"` |
| `make dev` | Serve `site/` locally with real `results.json` data |
| `make setup` | Cluster machine: preflight checks + install the eval scheduler |
| `make eval` | Full eval cycle for all pending models (needs cluster access) |
| `make deploy` | Commit and push updated `results.json` + `results/` |
| `make quality` | Ruff lint + verify all pipeline modules import cleanly |
| `make style` | Ruff auto-format the pipeline code |

## Architecture

```
results.json            the "database" — one entry per evaluated model
pipeline/
  discover.py           list collection, diff vs results.json -> results/pending/*.json
  orchestrate.py        full eval cycle: discover -> eval -> normalize
  deploy_agent.py       vllm serve command lookup (recipes.vllm.ai, then fallback)
  normalize.py          eval output -> results.json entry (idempotent upsert)
  in_pod_eval.sh        runs inside the pod: vllm serve + evaluate.py
  cron_eval.sh          cron entrypoint: picks up eval:pending GitHub issues
  setup.sh              cluster-machine setup: preflight checks + install scheduler
  speculators-eval.{service,timer}  systemd user units that schedule cron_eval.sh
site/                   static dashboard (no build step, vanilla JS)
prototype/              marketing-grade design prototypes
schema/                 results.json schema
logs/                   refresh logs + flock
```

For each pending model, `orchestrate.py`:

1. Brings up pod `devenv-$USER-eval` via devenv (`--gpus N --gpu-type ...` chosen
   from a sizing table in `discover.py`).
2. Streams `in_pod_eval.sh` into the pod: `vllm serve <model>`, then
   `speculators/scripts/evaluate/evaluate.py ... sweep`
   (acceptance length + per-position acceptance from vLLM Prometheus counters).
3. Copies `/tmp/eval-out` back and upserts the metrics into `results.json`.
4. Deletes the pod — guaranteed teardown even on failure/timeout.

Failures are recorded in `results.json` with `status: "failed"` and shown on the
dashboard; a model is re-evaluated when its HF `lastModified` changes.

## Contributing

- **Issues first.** Work is tracked as GitHub issues on this repo via the `gh`
  CLI; eval requests use the `eval:pending` label and are picked up by cron.
- **Checks before pushing.** Run `make quality` (and `make style` if the
  linter complains about formatting). CI-equivalent checks are local for now.
- **Pipeline code is stdlib-only.** Don't add third-party runtime dependencies
  without a discussion issue; dev tooling (ruff) goes in the `dev` extra in
  `pyproject.toml`.
- **Site is vanilla JS, no build step.** Edit `site/index.html` directly;
  `site/sample-results.json` is the offline fallback dataset.
- **Results changes** ship via `make deploy` (commits `results.json` +
  `results/` and pushes).

## Automation

On the cluster-connected machine, `pipeline/cron_eval.sh` runs unattended:
every 30 minutes it picks up `eval:pending` GitHub issues and evaluates them
one at a time. `make setup` (→ `pipeline/setup.sh`) installs the scheduler —
a systemd **user** timer where available, a flock-guarded crontab entry
otherwise — after verifying `oc`, `gh`, `flock`, and the `devenv` checkout.

Verify and watch:

```bash
systemctl --user list-timers speculators-eval.timer
journalctl --user -u speculators-eval.service -f
tail -f logs/cron.log            # wrapper log; per-eval logs are logs/eval-<slug>.log
```

Edge cases are handled by construction: no pending issues → the script logs
`Found 0 pending issue(s)` and exits; overlapping ticks → systemd never starts a
second instance of the oneshot service while one is active (crontab fallback
uses `flock`); reboot → `enable` + linger + `Persistent=true` (catches up a
missed run). Note `oc login` sessions expire — if evals start failing with
auth errors, re-login on the machine and re-run `make setup`.

The units themselves live in `pipeline/speculators-eval.{service,timer}` for
reference; `setup.sh` copies them into `~/.config/systemd/user/` with
`WorkingDirectory` pointed at the actual clone location.

## Deployment

The site is live at <https://neuralmagic.github.io/speculators-dashboard/>.
On every push to `main` that touches `results.json` or `site/`, the
`Deploy to GitHub Pages` workflow (`.github/workflows/deploy.yml`) copies the
canonical `results.json` into `site/` and publishes `site/` as the Pages
artifact. The dashboard loads that file at runtime, so newly published results
appear without rebuilding the page. The copy is never committed (it's
gitignored, same as the local `make dev` copy).

## Follow-ups (not yet done)

- Full SPEED-Bench second pass
- Parallel evals, historical trend charts
