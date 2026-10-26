# WDC eval infrastructure questions

These questions do not block artifact-only branch development. They should be answered before enabling unattended production evals.

## GPU runners and capacity

1. Do `ibm-wdc-os-h100-solo`, `duo`, `quad`, and `octo` guarantee exclusive access to exactly 1, 2, 4, and 8 H100 GPUs? Does each job see only its allocated devices?
2. How many runners are available for each label, and what is the expected queueing behavior when a matrix requests several configurations at once? What maximum matrix parallelism should we set?
3. Are runner workspaces ephemeral? How much local disk is available for the checkout, Python environment, and eval artifacts?
4. Is direct execution on these runners approved, or must evals use the Kubernetes pattern from `llm-compressor-testing` (`h100-util` controller plus a GPU-requesting pod) for isolation?

## Model cache and authentication

5. Is `/model-cache` present and writable on every H100 runner? What are its quota, eviction, file-locking, and concurrent-download semantics?
6. Can every current drafter and verifier be downloaded anonymously? If not, should the `HF_READ_WRITE_AUTOMATION` repository secret be configured, or is there a read-only token preferred for eval jobs?
7. Are outbound access to Hugging Face, PyPI, the nightly vLLM wheel index, and GitHub available from all runners?

## Software and benchmark policy

8. Should scheduled evals use nightly vLLM and `vllm-project/speculators@main`, as the initial workflow does, or pinned releases/SHAs? Which CUDA backend is supported on the runners: 13.0 (the existing test convention) or 12.9?
9. Is a 180-minute job limit, 40-minute server readiness limit, and 90-minute eval limit appropriate for the largest models?
10. The legacy sizing sends small models to A100s. The GHA path uses H100s throughout: 1 GPU through 8B, 2 through 14B, 4 through 70B, and 8 above 70B, preserving explicit overrides. Is this memory/capacity policy acceptable?
11. When vLLM reports OOM, should the workflow retry with more GPUs, mark the model for triage, or leave it failed? Unknown target sizes currently do not launch GPU jobs.
12. `num_speculative_tokens` now comes only from the drafter's Hugging Face `config.json` (`speculators_config.proposal_methods`). Missing or non-positive values block automatic execution. Is that the authoritative field for every algorithm?

## Publication and cutover

13. May `GITHUB_TOKEN` push eval commits directly to `main`, or does branch protection require a PR/bot? What is the desired production schedule and raw-artifact retention period?
14. Should a failed model replace the previous successful record (current normalization behavior), or should the website retain the last success and show the new failure separately?
15. Who will disable the external `pipeline/cron_eval.sh` systemd timer/crontab when GHA becomes authoritative? Both schedulers must not publish results concurrently.
16. Should eval issue creation/commenting remain part of the new workflow, or are the run, artifacts, and dashboard failure status sufficient?
