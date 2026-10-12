---
name: deploy-commands
description: Research and post vLLM deployment commands for drafter models on the open eval issues of this repo. Use when asked to fill in, figure out, or refresh deployment/deploy/serve commands on the eval issue backlog.
---

Every open eval issue needs a deployment command for its drafter, posted as a comment. The command must come from polling real sources — the drafter's HuggingFace model card and the verifier's recipe on https://recipes.vllm.ai/ — never from memory.

## Steps

1. **List the open eval issues.**
   ```bash
   gh issue list --repo rahul-tuli/speculators-dashboard --label eval:pending --state open --json number,title,body --limit 100
   ```
   If the user asked for all eval issues regardless of label, drop `--label` and match titles starting with `Eval:`.

2. **Skip issues already answered.** Fetch each issue's comments and skip any that already contain a `### Deployment command` heading — unless the user asked to refresh, in which case post a fresh comment anyway.
   ```bash
   gh issue view <number> --repo rahul-tuli/speculators-dashboard --json comments --jq '.comments[].body'
   ```

3. **Parse the issue body table.** Each eval issue carries a markdown table with: `Model` (the drafter), `Target` (the verifier), `Algorithm`, `Speculative Tokens`, `GPUs` (form `4xh100`). Values may be backtick-wrapped. Use the table as the initial metadata and GPU count; verify the drafter metadata against its `config.json` in the next step.

4. **Poll the sources, in this order.**
   - **Drafter's HF config**: `https://huggingface.co/<model>/raw/main/config.json` — read the verifier from `speculators_config.verifier.name_or_path`, the algorithm from `speculators_config.algorithm` (fall back to top-level `speculators_model_type`), and `num_speculative_tokens` from `speculators_config.proposal_methods`. Use the `greedy` proposal's `speculative_tokens`; if there is no `greedy` proposal, use the first proposal's `speculative_tokens`, matching `pipeline/discover.py`. These config values take precedence over conflicting issue-table values.
   - **Drafter's HF model card**: `https://huggingface.co/<model>/raw/main/README.md` — look for a `vllm serve` snippet or a documented speculative config.
   - **Verifier's recipe**: fetch `https://recipes.vllm.ai/models.json`, find the entry matching the verifier from the drafter config (fall back to the issue's `Target` if the config omits it), fetch that recipe, and extract its serve command.
   - If a config field is missing, use the corresponding issue-table value. If neither source provides it, report the gap instead of guessing. If issue metadata conflicts with the config, use the config and mention the discrepancy in the comment.

5. **Compose the command.** Base it on the verifier's recipe command, then add the speculative config for the drafter using the verified drafter ID, algorithm, and `num_speculative_tokens`. If no recipe exists for the verifier, fall back to the model card's command. If neither source yields a command, post a comment saying which sources were checked and came up empty — do not invent flags.

6. **Post the comment.**
   ```bash
   gh issue comment <number> --repo rahul-tuli/speculators-dashboard --body "..."
   ```
   Format: a `### Deployment command` heading, the command in a `bash` code block, then a Sources line naming the drafter config URL, HF card URL, and recipe URL (or "no recipe found for `<target>`").

7. **Report.** End with a per-issue list: issue title → posted / skipped (already had one) / no command found. Every open eval issue must appear in the report exactly once.

## Command rules

- **Long-form flags only**: `--tensor-parallel-size 4`, never `-tp 4`. Expand any shorthand a source uses.
- **Use the checkpoint's token count**: set `"num_speculative_tokens"` to the selected proposal's `speculative_tokens` from the drafter's `config.json`. Do not impose a minimum or replace it with a generic recommendation. Use the issue table only when the config has no usable token count.
- **Tensor parallelism follows the issue's GPU count** (the number before `x` in the GPUs field); omit the flag entirely for 1 GPU.
- The speculative config points at the **drafter** (the issue's Model), with the method matching the verified algorithm from `config.json` (or the issue's Algorithm when the config omits it).
