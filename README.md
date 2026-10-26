# Speculators Dashboard

A SpecBundle-style dashboard for speculative-decoding draft models ("speculators")
from the HuggingFace collection
[RedHatAI/speculator-models](https://huggingface.co/collections/RedHatAI/speculator-models).

An automated pipeline watches the collection, evaluates new/updated speculators on
WDC H100 runners through GitHub Actions, serves them with
[vLLM](https://github.com/vllm-project/vllm), measures them with the
[speculators](https://github.com/vllm-project/speculators) eval harness, and stores
results in `results.json`. The site in `site/` renders `results.json` as a static
page. A legacy OpenShift/devenv execution path remains available for manual use.

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
  discover.py           diff collection -> pending entries + H100 GHA matrix
  wdc_eval.sh           direct WDC runner: vllm serve + evaluate.py + record artifact
  merge_records.py      validate and merge per-job records into results.json
  orchestrate.py        legacy OpenShift eval cycle: discover -> eval -> normalize
  deploy_agent.py       vllm serve command lookup (recipes.vllm.ai, then fallback)
  normalize.py          eval output -> results.json entry (idempotent upsert)
  in_pod_eval.sh        legacy in-pod adapter: vllm serve + evaluate.py
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

## WDC GitHub Actions evals

`.github/workflows/evals.yml` is the end-to-end WDC pipeline. Its hosted
discovery job diffs the HF collection against `results.json` and emits one H100
matrix entry per pending model. GPU jobs use the WDC `solo`/`duo`/`quad`/`octo`
runner labels, run the existing vLLM + Speculators sweep, and upload independent
records, raw CSVs, provenance, and logs. A final job merges those records without
concurrent writes to `results.json`.

The drafter's HF `config.json` is authoritative for `num_speculative_tokens`;
missing or invalid values block automatic execution instead of selecting a
default. Discovery pins the drafter commit SHA and vLLM serves that exact
revision. GitHub Actions checks out `vllm-project/speculators` at the selected
ref before the runner script starts. The result's deploy command and provenance
record the exact serve/eval arguments and resolved software packages.

Branch/manual runs default to artifacts only and do not update the website. To
test one model from the GitHub Actions UI, choose **Evaluate speculators on WDC**,
enter its full HF model ID, and leave **publish** off. Download
`dashboard-results-<run-id>-<attempt>` to inspect the merged site data and each
`eval-result-<run-id>-<attempt>-<slug>` artifact for diagnostics. Use **Re-run
all jobs**, not a partial failed-job rerun, because publication requires one
record for every model in the plan. If model downloads need authentication,
configure the read-only `HF_READ_TOKEN` repository secret.

Scheduled runs on `main` publish `results.json`, which triggers Pages. This
requires the repository's `GITHUB_TOKEN` to have permission to push `main`.
Publication re-merges records onto the latest branch with bounded non-force
retries. A failed reevaluation keeps its last successful metrics in
`last_success` while remaining eligible for retry. The legacy cluster timer/cron
must be stopped before scheduled GHA publication is enabled; two schedulers must
not write results concurrently.

## Legacy OpenShift automation

On the cluster-connected machine, `pipeline/cron_eval.sh` can still process
`eval:pending` GitHub issues for manual/legacy operation. `make setup` installs
that scheduler. The HF poll-only workflow remains available for manual issue
creation, but its schedule is replaced by the unified WDC eval workflow.

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
