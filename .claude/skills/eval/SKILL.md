---
name: eval
description: Evaluate a speculator checkpoint on this GPU box and open a PR adding the raw eval results to its HuggingFace repo. Clones speculators, builds the eval venv, serves with vLLM, runs the 9-subset sweep, opens the PR. Use when asked to eval, evaluate, or benchmark a speculator/drafter model on local GPUs.
---

Run one checkpoint's full eval on the machine you are sitting on: `/eval <model>` (e.g. `/eval RedHatAI/Qwen3-8B-speculator.eagle3`), optional `--gpus N` and `--speedbench`. The deliverable is a PR on the model's HuggingFace repo adding `eval/<hardware>/` (raw harness output only) — once merged, the site's ingest picks it up. Redirect all server/eval output to log files; logs never enter this context.

## Steps

1. **Preflight** — all must pass:
   ```bash
   nvidia-smi --query-gpu=name --format=csv,noheader   # GPU box
   python3 --version && git --version
   ```
   HF auth with **write** access to RedHatAI: `HF_TOKEN` in the environment, or `huggingface-cli whoami` succeeding. Gated targets (e.g. Llama) also need the token for download. Missing token → ask the user for one.

2. **speculators checkout + eval venv** (idempotent). Default checkout `~/projects/speculators`; if the user has one elsewhere, use it. Otherwise:
   ```bash
   git clone https://github.com/vllm-project/speculators.git ~/projects/speculators
   ```
   One venv at `<checkout>/.venv` providing speculators (editable), guidellm, **and vllm** — one venv because `launch_vllm.py` execs vLLM from its own interpreter and reads provenance from the installed speculators:
   ```bash
   python3 -m venv <checkout>/.venv
   <checkout>/.venv/bin/pip install -e <checkout> guidellm vllm
   ```
   Reuse an existing venv; install only what is missing. **Done when**: `<checkout>/.venv/bin/python -c "import speculators, vllm, guidellm"` passes.

3. **Read the checkpoint's spec.** Fetch `https://huggingface.co/<model>/raw/main/config.json`: target = `speculators_config.verifier.name_or_path`, algorithm = `speculators_config.algorithm`. **Done when**: you know the target model to serve.

4. **Hardware tag** — `<count>x<type>`, lowercase. Count = GPUs used: `--gpus N` if given, else all visible GPUs. Type = the model token from the nvidia-smi name, lowercased: `NVIDIA H100 80GB HBM3` → `h100`, `NVIDIA B300` → `b300`, `NVIDIA A100-SXM4-80GB` → `a100`. This tag becomes the `eval/<hardware>/` folder name and the site's hardware badge (existing convention: `1xh100`, `4xh100`). **Done when**: the tag is decided (e.g. `1xb300`).

5. **Serve.** Output dir `~/eval-out/<slug>/<tag>/` (slug = text after the model's last `/`, lowercased, non-alnum runs → single dash). Sanity-check the command with `--dry-run` first, then launch for real as a background task:
   ```bash
   cd <checkout>/scripts
   <checkout>/.venv/bin/python launch_vllm.py eval <target> --spec-model <model> \
       --provenance-dir ~/eval-out/<slug>/<tag> \
       -- --port 8000 --tensor-parallel-size <count> > ~/eval-out/<slug>/vllm.log 2>&1
   ```
   Omit `--tensor-parallel-size` for 1 GPU. `--provenance-dir` writes `vllm_command.txt`, `checkpoint_sha256.txt`, `drafter_checkpoint_sha256.txt`, `vllm.patch` into the output dir. Do not pass `--spec-method` — vLLM infers the method from the checkpoint config. Poll `http://localhost:8000/health` up to 40 min (model downloads are slow); if the process dies, report the last ~30 log lines and stop. **Done when**: health returns 200.

6. **Run the 9-subset sweep.**
   ```bash
   <checkout>/.venv/bin/python <checkout>/scripts/evaluate/evaluate.py \
       --target http://localhost:8000/v1 \
       --output-dir ~/eval-out/<slug>/<tag> \
       --max-requests 80 sweep > ~/eval-out/<slug>/evaluate.log 2>&1
   ```
   Default subsets are the 9 standard ones. On failure: report the last ~30 lines of `evaluate.log`, kill the server, stop — no PR. **Done when**: `acceptance.csv`, `perf_results.csv`, `max_tokens.json`, and `eval_command.txt` exist in the output dir. Then kill the vLLM server.

7. **SPEED-Bench — only with `--speedbench`.** Run `prepare_speedbench.py` once, then a throughput-mode run with `--dataset speedbench/qualitative --speedbench-data-dir <dir>`, output into `~/eval-out/<slug>/<tag>/speedbench/`. This layout is provisional — it settles on the first real SPEED-Bench run; update this step and the map's fog note when it does.

8. **Open the PR on the checkpoint repo.** Upload only the top-level raw harness files present in the output dir — `acceptance.csv`, `perf_results.csv`, `max_tokens.json`, `eval_command.txt`, `vllm_command.txt`, `checkpoint_sha256.txt`, `drafter_checkpoint_sha256.txt`, `vllm.patch` — under `eval/<tag>/`. Artifact subdirectories (`gen_len/`, logs) stay local: the site's ingest reads exactly `eval/<hardware>/<file>` and ignores anything deeper.
   ```python
   from huggingface_hub import HfApi, CommitOperationAdd
   HfApi().create_commit(
       repo_id="<model>", repo_type="model", create_pr=True,
       commit_message="Eval: <tag> results",
       operations=[CommitOperationAdd(path_in_repo=f"eval/<tag>/{f}", path_or_fileobj=f"~/eval-out/<slug>/<tag>/{f}") for f in files],
   )
   ```
   Report the PR URL. **Done when**: the PR is open and its URL reported.

## Notes

- The site renders one hardware eval per model (newest wins), so a re-eval on new hardware just opens a fresh PR with a different tag — no cleanup needed.
- Everything is idempotent: re-running reuses the checkout and venv, overwrites the output dir, and opens a new PR.
- If the box's vLLM cannot serve the checkpoint (unsupported algorithm, OOM at the given count), say so and stop — a failed eval never produces a PR.
