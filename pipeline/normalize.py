#!/usr/bin/env python3
"""Normalize one eval run into results.json (idempotent upsert).

Usage:
  normalize.py --entry pending.json --status ok --raw-dir results/<slug>/raw \
               --deploy-command "vllm serve ..." --deploy-recipe-source "https://..." \
               --deploy-recipe-model "org/model"
  normalize.py --entry pending.json --status failed --error "pod unschedulable"
  normalize.py --baseline --target "org/model" --gpus "4xh100" --throughput 150.2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = REPO_ROOT / "results.json"
SCHEMA_JSON = REPO_ROOT / "schema" / "results.json"

REQUIRED_RECORD_KEYS = {
    "model",
    "target",
    "algorithm",
    "num_speculative_tokens",
    "hf_last_modified",
    "evaluated_at",
    "gpus",
    "status",
    "metrics",
    "error",
    "deploy",
}
REQUIRED_METRICS_KEYS = {
    "acceptance_length",
    "acceptance_at_pos",
    "num_drafts",
    "throughput_tps",
    "speedup",
    "ttft_ms",
    "itl_ms",
    "subsets",
}
REQUIRED_SUBSET_KEYS = {
    "acceptance_length",
    "acceptance_at_pos",
    "num_drafts",
    "num_draft_tokens",
    "num_accepted_tokens",
    "throughput_tps",
    "speedup",
    "ttft_ms",
    "itl_ms",
}


def validate_record(record: dict) -> None:
    missing = REQUIRED_RECORD_KEYS - set(record)
    if missing:
        raise ValueError(f"record missing required keys: {missing}")

    if record["status"] not in ("ok", "failed"):
        raise ValueError(f"status must be 'ok' or 'failed', got {record['status']!r}")

    metrics = record["metrics"]
    if record["status"] == "failed" and metrics is not None:
        raise ValueError("metrics must be null when status is 'failed'")

    if record["status"] == "ok":
        if metrics is None:
            raise ValueError("metrics must not be null when status is 'ok'")

        missing_m = REQUIRED_METRICS_KEYS - set(metrics)
        if missing_m:
            raise ValueError(f"metrics missing required keys: {missing_m}")

        if not isinstance(metrics["acceptance_at_pos"], list):
            raise ValueError("metrics.acceptance_at_pos must be a list")
        if not isinstance(metrics["subsets"], dict) or len(metrics["subsets"]) == 0:
            raise ValueError("metrics.subsets must be a non-empty dict")

        for name, subset in metrics["subsets"].items():
            missing_s = REQUIRED_SUBSET_KEYS - set(subset)
            if missing_s:
                raise ValueError(f"subset {name!r} missing required keys: {missing_s}")

    deploy = record.get("deploy")
    if not isinstance(deploy, dict):
        raise ValueError("deploy must be an object")
    for key in ("command", "recipe_source", "recipe_model"):
        if key not in deploy:
            raise ValueError(f"deploy missing required key: {key!r}")

    try:
        import jsonschema
    except ImportError:
        return

    schema = json.loads(SCHEMA_JSON.read_text())
    model_schema = json.loads(json.dumps(schema["$defs"]["model"]))
    model_schema["$defs"] = schema["$defs"]
    try:
        jsonschema.validate(record, model_schema)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"schema validation failed: {exc.message}") from exc


def parse_acceptance(raw_dir: Path) -> dict:
    """Parse the eval harness's acceptance.csv into per-subset acceptance data."""
    csvs = sorted(raw_dir.rglob("acceptance.csv"))
    if not csvs:
        raise FileNotFoundError(f"no acceptance.csv found under {raw_dir}")
    rows = list(csv.DictReader(csvs[0].open()))
    if not rows:
        raise ValueError(f"{csvs[0]} is empty")

    def f(row: dict, key: str) -> float:
        return float(row.get(key) or 0)

    pos_keys = sorted(
        (k for k in rows[0] if k.startswith("acceptance_at_pos_")),
        key=lambda k: int(k.rsplit("_", 1)[1]),
    )

    subsets = {}
    total_drafts = total_accepted = 0.0
    weighted_pos = dict.fromkeys(pos_keys, 0.0)
    for row in rows:
        drafts = f(row, "num_drafts")
        accepted = f(row, "num_accepted_tokens")
        total_drafts += drafts
        total_accepted += accepted
        for k in pos_keys:
            weighted_pos[k] += f(row, k) * drafts
        subsets[row["subset"]] = {
            "acceptance_length": round(f(row, "acceptance_length"), 4),
            "acceptance_at_pos": [round(f(row, k), 4) for k in pos_keys],
            "num_drafts": int(drafts),
            "num_draft_tokens": int(f(row, "num_draft_tokens")),
            "num_accepted_tokens": int(accepted),
        }

    if total_drafts <= 0:
        raise ValueError("acceptance.csv has zero drafts")

    top_level = {
        "acceptance_length": round(1 + total_accepted / total_drafts, 4),
        "acceptance_at_pos": [
            round(weighted_pos[k] / total_drafts, 4) for k in pos_keys
        ],
        "num_drafts": int(total_drafts),
    }
    return top_level, subsets


def _num(value: str | None) -> float | None:
    """Parse a CSV number; empty or non-finite values (e.g. inf rate) become None."""
    if not value:
        return None
    v = float(value)
    return round(v, 2) if math.isfinite(v) else None


def parse_perf(raw_dir: Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Parse sweep mode's perf_results.csv into per-subset throughput/latency data.

    Returns (best, sweeps):
    - best: {subset_name: {throughput_tps, ttft_ms, itl_ms}}, taking the row
      with the highest output_tps_median (peak throughput).
    - sweeps: {subset_name: [rate points sorted by target_rate]}, preserving
      all rows for latency-vs-throughput curves.
    """
    csvs = sorted(raw_dir.rglob("perf_results.csv"))
    if not csvs:
        raise FileNotFoundError(f"no perf_results.csv found under {raw_dir}")
    rows = list(csv.DictReader(csvs[0].open()))
    if not rows:
        raise ValueError(f"{csvs[0]} is empty")

    best: dict[str, dict] = {}
    sweeps: dict[str, list[dict]] = {}
    for row in rows:
        subset = row["subset"]
        tps = float(row.get("output_tps_median") or 0)
        sweeps.setdefault(subset, []).append(
            {
                "target_rate": _num(row.get("target_rate")),
                "rps_median": _num(row.get("rps_median")),
                "latency_median_s": _num(row.get("latency_median_s")),
                "ttft_median_ms": _num(row.get("ttft_median_ms")),
                "itl_median_ms": _num(row.get("itl_median_ms")),
                "output_tps_median": _num(row.get("output_tps_median")),
            }
        )
        if subset not in best or tps > best[subset]["throughput_tps"]:
            best[subset] = {
                "throughput_tps": round(tps, 2),
                "ttft_ms": round(float(row.get("ttft_median_ms") or 0), 2),
                "itl_ms": round(float(row.get("itl_median_ms") or 0), 2),
            }
    for points in sweeps.values():
        points.sort(key=lambda p: (p["target_rate"] is None, p["target_rate"] or 0))
    return best, sweeps


def build_metrics(raw_dir: Path, target: str, gpus: str, baselines: dict) -> dict:
    """Combine acceptance + perf data into the full metrics object."""
    top_acceptance, subsets = parse_acceptance(raw_dir)
    perf_by_subset, sweeps_by_subset = parse_perf(raw_dir)

    baseline_tps = None
    target_baselines = baselines.get(target)
    if target_baselines:
        baseline_tps = target_baselines.get(gpus)

    total_tokens = 0
    weighted_tps = 0.0
    weighted_ttft = 0.0
    weighted_itl = 0.0

    for name, sub in subsets.items():
        perf = perf_by_subset.get(name, {})
        sub["throughput_tps"] = perf.get("throughput_tps", 0.0)
        sub["ttft_ms"] = perf.get("ttft_ms", 0.0)
        sub["itl_ms"] = perf.get("itl_ms", 0.0)
        sub["sweep"] = sweeps_by_subset.get(name, [])
        sub["speedup"] = (
            round(sub["throughput_tps"] / baseline_tps, 4) if baseline_tps else 0.0
        )

        tokens = sub["num_accepted_tokens"]
        total_tokens += tokens
        weighted_tps += sub["throughput_tps"] * tokens
        weighted_ttft += sub["ttft_ms"] * tokens
        weighted_itl += sub["itl_ms"] * tokens

    if total_tokens > 0:
        top_tps = round(weighted_tps / total_tokens, 2)
        top_ttft = round(weighted_ttft / total_tokens, 2)
        top_itl = round(weighted_itl / total_tokens, 2)
    else:
        top_tps = top_ttft = top_itl = 0.0

    top_speedup = round(top_tps / baseline_tps, 4) if baseline_tps else 0.0

    return {
        **top_acceptance,
        "throughput_tps": top_tps,
        "speedup": top_speedup,
        "ttft_ms": top_ttft,
        "itl_ms": top_itl,
        "subsets": subsets,
    }


def _load_results() -> dict:
    if RESULTS_JSON.exists():
        results = json.loads(RESULTS_JSON.read_text())
        if "baselines" not in results:
            results["baselines"] = {}
        return results
    return {"baselines": {}, "models": []}


def _save_results(results: dict) -> None:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    results["updated_at"] = now
    RESULTS_JSON.write_text(json.dumps(results, indent=2) + "\n")


def upsert_baseline(target: str, gpus: str, throughput_tps: float) -> None:
    """Record an autoregressive baseline throughput for a target model + GPU config."""
    results = _load_results()
    if target not in results["baselines"]:
        results["baselines"][target] = {}
    results["baselines"][target][gpus] = round(throughput_tps, 2)
    _save_results(results)
    print(f"results.json: baseline {target} @ {gpus} = {throughput_tps:.2f} tok/s")


def upsert_result(
    entry_path,
    status,
    raw_dir=None,
    error=None,
    deploy_command=None,
    deploy_recipe_source=None,
    deploy_recipe_model=None,
    provenance=None,
    output_path=None,
):
    """Normalize one eval run into results.json (idempotent upsert)."""
    entry_path = Path(entry_path)
    if raw_dir is not None:
        raw_dir = Path(raw_dir)

    entry = json.loads(entry_path.read_text())
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    gpus_str = f"{entry['gpus']}x{entry['gpu_type']}"
    record = {
        "model": entry["model"],
        "target": entry.get("target"),
        "algorithm": entry.get("algorithm"),
        "num_speculative_tokens": entry.get("num_speculative_tokens"),
        "hf_last_modified": entry.get("hf_last_modified"),
        "hf_revision": entry.get("hf_revision"),
        "evaluated_at": now,
        "gpus": gpus_str,
        "status": status,
        "metrics": None,
        "error": error,
        "deploy": {
            "command": deploy_command or "",
            "recipe_source": deploy_recipe_source or "",
            "recipe_model": deploy_recipe_model or "",
        },
    }
    for key in ("params", "architecture", "base_model"):
        if entry.get(key) is not None:
            record[key] = entry[key]
    if provenance is not None:
        record["provenance"] = provenance

    if status == "ok":
        if not raw_dir:
            raise ValueError("raw_dir is required when status is 'ok'")
        results = _load_results()
        record["metrics"] = build_metrics(
            raw_dir,
            entry.get("target", ""),
            gpus_str,
            results.get("baselines", {}),
        )

    validate_record(record)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{output_path}: wrote {record['model']} (status={status})")
        return

    results = _load_results()
    models = [m for m in results.get("models", []) if m["model"] != record["model"]]
    models.append(record)
    models.sort(key=lambda m: m["model"])
    results["models"] = models
    _save_results(results)
    print(f"results.json: upserted {record['model']} (status={status})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    eval_p = sub.add_parser("eval", help="Upsert a drafter eval result")
    eval_p.add_argument(
        "--entry", required=True, help="pending entry JSON from discover.py"
    )
    eval_p.add_argument("--status", choices=["ok", "failed"], required=True)
    eval_p.add_argument("--raw-dir", type=Path, default=None)
    eval_p.add_argument("--error", default=None)
    eval_p.add_argument("--deploy-command", default=None)
    eval_p.add_argument("--deploy-recipe-source", default=None)
    eval_p.add_argument("--deploy-recipe-model", default=None)
    eval_p.add_argument("--provenance", type=Path, default=None)
    eval_p.add_argument("--output", type=Path, default=None)

    base_p = sub.add_parser("baseline", help="Upsert an autoregressive baseline")
    base_p.add_argument("--target", required=True)
    base_p.add_argument("--gpus", required=True)
    base_p.add_argument("--throughput", type=float, required=True)

    # Backward compat: if no subcommand, treat as eval with flat args
    args, remaining = ap.parse_known_args()
    if args.cmd is None:
        eval_p2 = argparse.ArgumentParser()
        eval_p2.add_argument("--entry", required=True)
        eval_p2.add_argument("--status", choices=["ok", "failed"], required=True)
        eval_p2.add_argument("--raw-dir", type=Path, default=None)
        eval_p2.add_argument("--error", default=None)
        eval_p2.add_argument("--deploy-command", default=None)
        eval_p2.add_argument("--deploy-recipe-source", default=None)
        eval_p2.add_argument("--deploy-recipe-model", default=None)
        eval_p2.add_argument("--provenance", type=Path, default=None)
        eval_p2.add_argument("--output", type=Path, default=None)
        args = eval_p2.parse_args()
        args.cmd = "eval"

    if args.cmd == "baseline":
        upsert_baseline(args.target, args.gpus, args.throughput)
    else:
        provenance = (
            json.loads(args.provenance.read_text()) if args.provenance else None
        )
        upsert_result(
            args.entry,
            args.status,
            raw_dir=args.raw_dir,
            error=args.error,
            deploy_command=args.deploy_command,
            deploy_recipe_source=args.deploy_recipe_source,
            deploy_recipe_model=args.deploy_recipe_model,
            provenance=provenance,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
