#!/usr/bin/env python3
"""Build-time ingest: regenerate the site's data from checkpoint eval folders.

Stdlib-only. Scans the RedHatAI/speculator-models HF collection, finds
`eval/<hardware>/` folders in each checkpoint repo, parses the raw harness
output (`acceptance.csv`, `perf_results.csv`, provenance from
`eval_command.txt` / `vllm_command.txt`), and emits the trimmed JSON the
static site serves (site/data.json). No duplicate summary schema: the raw
harness output is the contract.

Models without an eval folder fall back to the seed (the existing
results.json) so evals captured under the old pipeline carry over.

Usage:
  ingest.py                      full run -> site/data.json
  ingest.py --collection FILE    read a saved collection API response
  ingest.py --seed PATH          seed results.json (default: ./results.json)
  ingest.py --out PATH           output JSON (default: ./site/data.json)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

COLLECTION_URL = "https://huggingface.co/api/collections/RedHatAI/speculator-models"
TREE_URL = "https://huggingface.co/api/models/{model}/tree/main/eval?recursive=true&expand=true"
RAW_URL = "https://huggingface.co/{model}/resolve/main/{path}"
CONFIG_URL = "https://huggingface.co/{model}/raw/main/config.json"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEED = REPO_ROOT / "results.json"
DEFAULT_OUT = REPO_ROOT / "site" / "data.json"

# Fields the site's trimmed shape keeps (sweep arrays, acceptance_at_pos,
# baselines, and speedup are dropped).
METRIC_FIELDS = (
    "acceptance_length",
    "throughput_tps",
    "ttft_ms",
    "itl_ms",
    "num_drafts",
)
SUBSET_FIELDS = ("acceptance_length", "throughput_tps", "itl_ms", "ttft_ms")


def fetch_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode()


def _headers() -> dict:
    import os

    headers = {"User-Agent": "speculators-dashboard"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# ---------------------------------------------------------------------------
# Raw harness output parsing
# ---------------------------------------------------------------------------


def parse_acceptance(text: str) -> tuple[dict, dict]:
    """Parse acceptance.csv into (top_level, per-subset) acceptance data."""
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError("acceptance.csv is empty")

    def f(row: dict, key: str) -> float:
        return float(row.get(key) or 0)

    subsets = {}
    total_drafts = total_accepted = 0.0
    for row in rows:
        drafts = f(row, "num_drafts")
        accepted = f(row, "num_accepted_tokens")
        total_drafts += drafts
        total_accepted += accepted
        subsets[row["subset"]] = {
            "acceptance_length": round(f(row, "acceptance_length"), 4),
            "num_drafts": int(drafts),
            "num_accepted_tokens": int(accepted),
        }

    if total_drafts <= 0:
        raise ValueError("acceptance.csv has zero drafts")

    top_level = {
        "acceptance_length": round(1 + total_accepted / total_drafts, 4),
        "num_drafts": int(total_drafts),
    }
    return top_level, subsets


def parse_perf(text: str) -> dict[str, dict]:
    """Parse sweep mode's perf_results.csv into per-subset throughput/latency.

    Takes the row with the highest output_tps_median (peak throughput) per
    subset.
    """
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError("perf_results.csv is empty")

    best: dict[str, dict] = {}
    for row in rows:
        subset = row["subset"]
        tps = float(row.get("output_tps_median") or 0)
        if not math.isfinite(tps):
            continue
        if subset not in best or tps > best[subset]["throughput_tps"]:
            best[subset] = {
                "throughput_tps": round(tps, 2),
                "ttft_ms": round(float(row.get("ttft_median_ms") or 0), 2),
                "itl_ms": round(float(row.get("itl_median_ms") or 0), 2),
            }
    return best


def parse_provenance(text: str) -> dict:
    """Parse the `# Key: value` header lines of eval_command.txt / vllm_command.txt."""
    prov: dict = {"versions": {}}
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        key, _, value = line[1:].partition(":")
        key, value = key.strip().lower(), value.strip()
        if key == "timestamp":
            prov["timestamp"] = value
        elif key == "git sha":
            prov["git_sha"] = value
        elif key and value and value != "not installed":
            prov["versions"][key] = value
    return prov


# ---------------------------------------------------------------------------
# Checkpoint eval-folder scan
# ---------------------------------------------------------------------------


def list_eval_files(model: str) -> dict[str, set[str]]:
    """List eval/<hardware>/ files in a checkpoint repo. {} if no eval folder."""
    try:
        entries = fetch_json(TREE_URL.format(model=model))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {}
        raise
    folders: dict[str, set[str]] = {}
    for entry in entries:
        if entry.get("type") != "file":
            continue
        parts = entry["path"].split("/")
        # eval/<hardware>/<file>; anything deeper belongs to harness artifacts
        if len(parts) == 3:
            folders.setdefault(parts[1], set()).add(parts[2])
    return folders


def extract_speculator_meta(model: str) -> dict:
    """Pull target + algorithm from the checkpoint's config.json."""
    try:
        cfg = fetch_json(CONFIG_URL.format(model=model))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  WARN: cannot fetch config for {model}: {e}", file=sys.stderr)
        return {}
    sc = cfg.get("speculators_config", {})
    verifier = sc.get("verifier", {})
    return {
        "algorithm": sc.get("algorithm", cfg.get("speculators_model_type")),
        "target": verifier.get("name_or_path"),
    }


def build_eval_record(model: str, hardware: str, files: set[str]) -> dict | None:
    """Build one site record from a checkpoint's eval/<hardware>/ folder."""
    missing = {"acceptance.csv", "perf_results.csv"} - files
    if missing:
        print(
            f"  SKIP {model} eval/{hardware}: missing {sorted(missing)}",
            file=sys.stderr,
        )
        return None

    def raw(path: str) -> str:
        return fetch_text(RAW_URL.format(model=model, path=f"eval/{hardware}/{path}"))

    try:
        top_acceptance, subsets = parse_acceptance(raw("acceptance.csv"))
        perf_by_subset = parse_perf(raw("perf_results.csv"))
    except (ValueError, KeyError, urllib.error.URLError) as e:
        print(f"  SKIP {model} eval/{hardware}: {e}", file=sys.stderr)
        return None

    prov = {}
    vllm_prov = {}
    if "eval_command.txt" in files:
        prov = parse_provenance(raw("eval_command.txt"))
    if "vllm_command.txt" in files:
        vllm_prov = parse_provenance(raw("vllm_command.txt"))

    total_tokens = 0
    weighted_tps = weighted_ttft = weighted_itl = 0.0
    trimmed_subsets = {}
    for name, sub in subsets.items():
        perf = perf_by_subset.get(name, {})
        merged = {**sub, **perf}
        tokens = sub["num_accepted_tokens"]
        total_tokens += tokens
        weighted_tps += perf.get("throughput_tps", 0.0) * tokens
        weighted_ttft += perf.get("ttft_ms", 0.0) * tokens
        weighted_itl += perf.get("itl_ms", 0.0) * tokens
        trimmed_subsets[name] = {k: merged[k] for k in SUBSET_FIELDS if k in merged}

    if total_tokens > 0:
        top_tps = round(weighted_tps / total_tokens, 2)
        top_ttft = round(weighted_ttft / total_tokens, 2)
        top_itl = round(weighted_itl / total_tokens, 2)
    else:
        top_tps = top_ttft = top_itl = 0.0

    evaluated_at = prov.get("timestamp")
    if not evaluated_at:
        print(
            f"  WARN {model} eval/{hardware}: no timestamp in eval_command.txt",
            file=sys.stderr,
        )

    record = {
        "model": model,
        "gpus": hardware.lower(),
        "evaluated_at": evaluated_at,
        "status": "ok",
        "metrics": {
            "acceptance_length": top_acceptance["acceptance_length"],
            "throughput_tps": top_tps,
            "ttft_ms": top_ttft,
            "itl_ms": top_itl,
            "num_drafts": top_acceptance["num_drafts"],
        },
        "subsets": trimmed_subsets,
        **extract_speculator_meta(model),
    }
    versions = prov.get("versions", {})
    vllm_version = versions.get("vllm") or vllm_prov.get("versions", {}).get("vllm")
    if vllm_version:
        record["vllm"] = vllm_version
    if versions.get("speculators"):
        record["speculators"] = versions["speculators"]
    return record


def scan_eval_folders(collection: list[dict]) -> dict[str, dict]:
    """Scan every collection model for eval folders. Returns {model: record}.

    A model with multiple hardware folders keeps the newest eval; the rest
    are dropped with a warning (the site renders one row per model).
    """
    records: dict[str, dict] = {}
    for item in collection:
        model = item["model"]
        try:
            folders = list_eval_files(model)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            print(f"  WARN: cannot list eval files for {model}: {e}", file=sys.stderr)
            continue
        if not folders:
            continue
        candidates = [
            r
            for hw, files in sorted(folders.items())
            if (r := build_eval_record(model, hw, files))
        ]
        if not candidates:
            continue
        if len(candidates) > 1:
            dropped = sorted(candidates, key=lambda r: r["evaluated_at"] or "")[:-1]
            for d in dropped:
                print(
                    f"  WARN {model}: dropping older {d['gpus']} eval "
                    f"(site shows one hardware per model)",
                    file=sys.stderr,
                )
        best = max(candidates, key=lambda r: r["evaluated_at"] or "")
        records[model] = best
        print(f"  EVAL: {model} ({best['gpus']}, evaluated {best['evaluated_at']})")
    return records


# ---------------------------------------------------------------------------
# Seed carry-over
# ---------------------------------------------------------------------------


def trim_seed_record(rec: dict) -> dict:
    """Trim an old-pipeline results.json record to the site's shape."""
    metrics = rec.get("metrics") or {}
    subsets = metrics.get("subsets") or {}
    return {
        "model": rec["model"],
        "target": rec.get("target"),
        "algorithm": rec.get("algorithm"),
        "gpus": rec.get("gpus"),
        "evaluated_at": rec.get("evaluated_at"),
        "status": "ok",
        "metrics": {k: metrics[k] for k in METRIC_FIELDS if k in metrics},
        "subsets": {
            name: {k: sub[k] for k in SUBSET_FIELDS if k in sub}
            for name, sub in subsets.items()
        },
    }


def load_seed(path: Path) -> dict[str, dict]:
    """Load ok-status records from the old results.json, keyed by model."""
    if not path.exists():
        print(f"WARN: seed {path} not found; no carry-over", file=sys.stderr)
        return {}
    results = json.loads(path.read_text())
    seeded = {}
    for rec in results.get("models", []):
        if rec.get("status") != "ok":
            continue
        seeded[rec["model"]] = trim_seed_record(rec)
    return seeded


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--collection",
        type=Path,
        default=None,
        help="Read the collection API response from a file instead of fetching",
    )
    args = ap.parse_args()

    if args.collection:
        raw = json.loads(args.collection.read_text())
    else:
        raw = fetch_json(COLLECTION_URL)
    items = [it for it in raw.get("items", []) if it.get("type") == "model"]
    collection = [
        {"model": it["id"], "hf_last_modified": it.get("lastModified")} for it in items
    ]
    print(f"Collection has {len(collection)} models")

    records = scan_eval_folders(collection)
    seeded = load_seed(args.seed)

    models = []
    carried = 0
    for item in collection:
        model = item["model"]
        if model in records:
            models.append(records[model])
        elif model in seeded:
            carried += 1
            models.append(seeded[model])
    orphaned = set(seeded) - {item["model"] for item in collection}
    for model in sorted(orphaned):
        print(f"  WARN: seeded {model} is no longer in the collection", file=sys.stderr)
        models.append(seeded[model])

    print(f"{len(records)} from eval folders, {carried} carried over from seed")

    out = {
        "updated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "collection": collection,
        "models": models,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {args.out} ({len(models)} evaluated models)")


if __name__ == "__main__":
    main()
