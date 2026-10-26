#!/usr/bin/env python3
"""Discover speculator models in the RedHatAI/speculator-models HF collection.

Stdlib-only. Fetches the collection listing + each model's config.json,
extracts speculator metadata, decides GPU sizing, and diffs against
results.json to find models that need (re-)evaluation.

Outputs:
  results/pending/<slug>.json   one file per model needing eval
  results/pending/index.json    list of all pending slugs (for humans)

A model needs eval when (model, hf_last_modified) is not present in
results.json with status "ok".
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

COLLECTION_URL = "https://huggingface.co/api/collections/RedHatAI/speculator-models"
CONFIG_URL = "https://huggingface.co/{model}/raw/main/config.json"
MODEL_API_URL = "https://huggingface.co/api/models/{model}"

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = REPO_ROOT / "results.json"
PENDING_DIR = REPO_ROOT / "results" / "pending"

# Explicit overrides for targets whose naive size parse is misleading
# (mostly MoE, where the "17B" in the name is the active size).
SIZING_OVERRIDES = {
    "Llama-4-Maverick": (8, "h100"),
    "Llama-4-Scout": (4, "h100"),
    "gpt-oss-120b": (4, "h100"),
    "gpt-oss-20b": (1, "a100"),
}


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "speculators-dashboard"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def parse_target_size_b(target: str) -> float | None:
    """Extract total parameter count in billions from a model name.

    'Qwen3-235B-A22B' -> 235, 'Llama-3.1-8B' -> 8, 'gemma-4-31B-it' -> 31.
    Takes the largest NxB match (for MoE names the total comes first and is
    larger than the active count).
    """
    sizes = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)[bB](?![a-zA-Z])", target)]
    return max(sizes) if sizes else None


def gpu_sizing(target: str) -> tuple[int, str, str]:
    """Return (gpus, gpu_type, note) for serving the target verifier."""
    for key, (gpus, gtype) in SIZING_OVERRIDES.items():
        if key.lower() in target.lower():
            return gpus, gtype, f"override:{key}"
    size = parse_target_size_b(target)
    if size is None:
        return 1, "a100", "unknown-size-default"
    if size <= 8:
        return 1, "a100", ""
    if size <= 14:
        return 2, "a100", ""
    if size <= 32:
        return 4, "a100", ""
    if size <= 70:
        return 4, "h100", ""
    return 8, "h100", ""


def slugify(model_id: str) -> str:
    name = model_id.split("/", 1)[-1].lower()
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def fetch_hf_metadata(model_id: str) -> dict:
    """Pull params/architecture/base_model from the HF model API.

    Returns a dict with any of: params (int, from safetensors total),
    architecture (str, first entry of config.architectures), base_model
    (str, from card data). Fields that are unavailable are omitted; an
    empty dict means the API call itself failed.
    """
    try:
        info = fetch_json(MODEL_API_URL.format(model=model_id))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  WARN: cannot fetch HF metadata for {model_id}: {e}", file=sys.stderr)
        return {}
    meta = {}
    total = (info.get("safetensors") or {}).get("total")
    if isinstance(total, int) and total > 0:
        meta["params"] = total
    architectures = (info.get("config") or {}).get("architectures") or []
    if architectures:
        meta["architecture"] = architectures[0]
    base_model = (info.get("cardData") or {}).get("base_model")
    if isinstance(base_model, list):
        base_model = base_model[0] if base_model else None
    if base_model:
        meta["base_model"] = base_model
    return meta


def extract_metadata(model_id: str) -> dict | None:
    """Fetch config.json and pull speculator metadata. None if not a speculator."""
    try:
        cfg = fetch_json(CONFIG_URL.format(model=model_id))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  WARN: cannot fetch config for {model_id}: {e}", file=sys.stderr)
        return None
    if "speculators_model_type" not in cfg:
        return None
    sc = cfg.get("speculators_config", {})
    proposals = sc.get("proposal_methods", [])
    greedy = next(
        (p for p in proposals if p.get("proposal_type") == "greedy"),
        proposals[0] if proposals else {},
    )
    verifier = sc.get("verifier", {})
    return {
        "algorithm": sc.get("algorithm", cfg["speculators_model_type"]),
        "target": verifier.get("name_or_path"),
        "num_speculative_tokens": greedy.get("speculative_tokens"),
        **fetch_hf_metadata(model_id),
    }


RUNNER_LABELS = {
    1: "ibm-wdc-os-h100-solo",
    2: "ibm-wdc-os-h100-duo",
    4: "ibm-wdc-os-h100-quad",
    8: "ibm-wdc-os-h100-octo",
}


def h100_sizing(target: str) -> tuple[int | None, str]:
    """Return an H100 count for GHA without changing legacy cluster sizing."""
    for key, (gpus, _gtype) in SIZING_OVERRIDES.items():
        if key.lower() in target.lower():
            return gpus, f"override:{key}"
    size = parse_target_size_b(target)
    if size is None:
        return None, "unknown-size-needs-triage"
    if size <= 8:
        return 1, ""
    if size <= 14:
        return 2, ""
    if size <= 70:
        return 4, ""
    return 8, ""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gha", action="store_true", help="emit an H100 GHA matrix")
    parser.add_argument("--model", help="evaluate this model even if results are current")
    parser.add_argument(
        "--output",
        type=Path,
        default=PENDING_DIR / "gha-matrix.json",
        help="matrix output path used with --gha",
    )
    args = parser.parse_args()

    collection = fetch_json(COLLECTION_URL)
    items = [it for it in collection.get("items", []) if it.get("type") == "model"]
    print(f"Collection has {len(items)} models")
    if args.model and not any(it["id"] == args.model for it in items):
        parser.error(f"model is not in the collection: {args.model}")

    results = (
        json.loads(RESULTS_JSON.read_text())
        if RESULTS_JSON.exists()
        else {"models": []}
    )
    done = {
        (m["model"], m.get("hf_last_modified"))
        for m in results.get("models", [])
        if m.get("status") == "ok"
    }

    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    # Clear stale pending entries from a previous (possibly interrupted) run.
    for old in PENDING_DIR.glob("*.json"):
        old.unlink()

    pending = []
    matrix = []
    for item in items:
        model_id = item["id"]
        if args.model and model_id != args.model:
            continue
        last_modified = item.get("lastModified")
        if item.get("gated") or item.get("private"):
            print(f"  SKIP (gated/private): {model_id}")
            continue
        if not args.model and (model_id, last_modified) in done:
            print(f"  up-to-date: {model_id}")
            continue
        meta = extract_metadata(model_id)
        if meta is None:
            print(f"  SKIP (no speculators config): {model_id}")
            continue
        tokens = meta["num_speculative_tokens"]
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
            print(f"  NEEDS TRIAGE (invalid speculative_tokens in config): {model_id}")
            continue
        if args.gha:
            gpus, note = h100_sizing(meta["target"] or "")
            gpu_type = "h100"
        else:
            gpus, gpu_type, note = gpu_sizing(meta["target"] or "")
        if gpus is None:
            print(f"  NEEDS TRIAGE (unknown target size): {model_id}")
            continue
        slug = slugify(model_id)
        entry = {
            "model": model_id,
            "slug": slug,
            "hf_last_modified": last_modified,
            "algorithm": meta["algorithm"],
            "target": meta["target"],
            "num_speculative_tokens": meta["num_speculative_tokens"],
            "gpus": gpus,
            "gpu_type": gpu_type,
        }
        if note:
            entry["sizing_note"] = note
        for key in ("params", "architecture", "base_model"):
            if key in meta:
                entry[key] = meta[key]
        entry_path = PENDING_DIR / f"{slug}.json"
        entry_path.write_text(json.dumps(entry, indent=2) + "\n")
        pending.append(slug)
        if args.gha:
            matrix.append(
                {
                    "model": model_id,
                    "slug": slug,
                    "gpus": gpus,
                    "runner": RUNNER_LABELS[gpus],
                    "entry": str(entry_path.relative_to(REPO_ROOT)),
                }
            )
        print(f"  PENDING: {model_id} -> {gpus}x{gpu_type} target={meta['target']}")

    (PENDING_DIR / "index.json").write_text(json.dumps(pending, indent=2) + "\n")
    if args.gha:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"include": matrix}, indent=2) + "\n")
        print(f"GHA matrix: {args.output}")
    print(f"\n{len(pending)} model(s) pending evaluation")


if __name__ == "__main__":
    main()
