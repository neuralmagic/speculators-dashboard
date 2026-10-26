#!/usr/bin/env python3
"""Merge per-job eval records into the canonical dashboard results file."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from normalize import REPO_ROOT, SCHEMA_JSON, validate_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results.json")
    parser.add_argument("--expected-matrix", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results_path = args.results
    results = (
        json.loads(results_path.read_text())
        if results_path.exists()
        else {"baselines": {}, "models": []}
    )
    models = {model["model"]: model for model in results.get("models", [])}
    seen = set()
    for path in args.records:
        record = json.loads(path.read_text())
        validate_record(record)
        model = record["model"]
        if model in seen:
            raise ValueError(f"duplicate eval record for {model}")
        seen.add(model)
        models[model] = record

    if args.expected_matrix:
        matrix = json.loads(args.expected_matrix.read_text())
        expected = {job["model"] for job in matrix["include"]}
        if seen != expected:
            raise ValueError(
                f"eval records do not match plan; missing={sorted(expected - seen)}, "
                f"unexpected={sorted(seen - expected)}"
            )

    results["models"] = [models[name] for name in sorted(models)]
    results["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        import jsonschema
    except ImportError:
        pass
    else:
        schema = json.loads(SCHEMA_JSON.read_text())
        jsonschema.validate(results, schema)
    output = args.output or results_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"{output}: merged {len(seen)} record(s); {len(models)} total model(s)")


if __name__ == "__main__":
    main()
