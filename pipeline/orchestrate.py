#!/usr/bin/env python3
"""Orchestrate the full speculator eval cycle: discover -> eval -> normalize.

Replaces refresh.sh + run_eval.sh. Keeps in_pod_eval.sh as the in-pod
adapter (it runs inside the cluster pod, a genuine boundary).
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from deploy_agent import lookup_deploy
from normalize import upsert_baseline, upsert_result

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEVENV_DIR = Path(os.environ.get("DEVENV_DIR", Path.home() / "projects" / "devenv"))
NAMESPACE = os.environ.get("DEVENV_NAMESPACE", "machine-learning")
EVAL_TIMEOUT = int(os.environ.get("EVAL_TIMEOUT", "5400"))

_teardown_registered = False


def pod_name() -> str:
    user = os.environ.get("USER", os.environ.get("LOGNAME", "unknown"))
    return f"devenv-{user.lower()}-eval"


def run_cmd(cmd, **kwargs):
    return subprocess.run(cmd, check=True, **kwargs)


def teardown_pod():
    pod = pod_name()
    print(f"=== Tearing down {pod} ===")
    try:
        subprocess.run(
            [str(DEVENV_DIR / "launch.sh"), "--name", "eval", "--down"],
            input=b"y\n",
            timeout=120,
        )
    except Exception:
        pass


def _register_teardown_signals():
    """Wire SIGTERM/SIGINT to tear down the pod before exiting."""
    global _teardown_registered
    if _teardown_registered:
        return
    _teardown_registered = True

    def _handler(signum, _frame):
        teardown_pod()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def provision_pod(gpus: int, gpu_type: str):
    # launch.sh tries to attach an interactive tmux after creating the pod;
    # that step fails without a TTY, which is expected — the pod stays up.
    subprocess.run(
        [
            str(DEVENV_DIR / "launch.sh"),
            "--name",
            "eval",
            "--gpus",
            str(gpus),
            "--gpu-type",
            gpu_type,
            "--cluster",
        ],
        stdin=subprocess.DEVNULL,
    )
    run_cmd(
        [
            "oc",
            "wait",
            "--for=condition=Ready",
            f"pod/{pod_name()}",
            "-n",
            NAMESPACE,
            "--timeout=1800s",
        ]
    )


def run_eval_in_pod(model: str, gpus: int, algorithm: str):
    with open(SCRIPT_DIR / "in_pod_eval.sh") as f:
        run_cmd(
            [
                "oc",
                "exec",
                "-i",
                pod_name(),
                "-n",
                NAMESPACE,
                "--",
                "bash",
                "-s",
                "--",
                model,
                str(gpus),
                algorithm,
            ],
            stdin=f,
            timeout=EVAL_TIMEOUT,
        )


def copy_results_out(slug: str) -> Path:
    outdir = REPO_ROOT / "results" / slug / "raw"
    if outdir.exists():
        shutil.rmtree(outdir)
    run_cmd(
        [
            "oc",
            "cp",
            f"{NAMESPACE}/{pod_name()}:/tmp/eval-out",
            str(outdir),
        ]
    )
    return outdir


def evaluate_model(entry: dict):
    model = entry["model"]
    slug = entry["slug"]
    gpus = int(entry["gpus"])
    gpu_type = entry["gpu_type"]
    algorithm = entry["algorithm"]

    entry_path = REPO_ROOT / "results" / "pending" / f"{slug}.json"

    print(f"\n=== Eval: {model} on {gpus}x{gpu_type} (pod {pod_name()}) ===")

    _register_teardown_signals()
    try:
        provision_pod(gpus, gpu_type)
        run_eval_in_pod(model, gpus, algorithm)
        raw_dir = copy_results_out(slug)

        deploy = lookup_deploy(model, entry.get("target", ""), algorithm, gpus)
        upsert_result(
            entry_path,
            "ok",
            raw_dir=raw_dir,
            deploy_command=deploy["command"],
            deploy_recipe_source=deploy["recipe_source"],
            deploy_recipe_model=deploy["recipe_model"],
        )
        print(f"=== Done: {model} ===")
    except Exception as e:
        print(f"ERROR: {e}")
        try:
            upsert_result(
                entry_path,
                "failed",
                error=str(e),
                deploy_command="",
                deploy_recipe_source="",
                deploy_recipe_model="",
            )
        except Exception:
            print(f"ERROR: normalize.py also failed for {model}")
        raise
    finally:
        teardown_pod()


def run_baseline_in_pod(target: str, gpus: int):
    with open(SCRIPT_DIR / "in_pod_baseline.sh") as f:
        run_cmd(
            [
                "oc",
                "exec",
                "-i",
                pod_name(),
                "-n",
                NAMESPACE,
                "--",
                "bash",
                "-s",
                "--",
                target,
                str(gpus),
            ],
            stdin=f,
            timeout=EVAL_TIMEOUT,
        )


def copy_baseline_out(slug: str) -> Path:
    outdir = REPO_ROOT / "results" / slug / "baseline"
    if outdir.exists():
        shutil.rmtree(outdir)
    run_cmd(
        [
            "oc",
            "cp",
            f"{NAMESPACE}/{pod_name()}:/tmp/baseline-out",
            str(outdir),
        ]
    )
    return outdir


def parse_baseline_throughput(baseline_dir: Path) -> float:
    """Extract peak throughput from guidellm's baseline output."""
    baseline_json = baseline_dir / "baseline.json"
    if not baseline_json.exists():
        raise FileNotFoundError(f"no baseline.json in {baseline_dir}")
    data = json.loads(baseline_json.read_text())
    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        raise ValueError("no benchmarks in baseline.json")
    tps = max(b.get("output_tokens_per_second", 0) for b in benchmarks)
    if tps <= 0:
        raise ValueError("baseline throughput is zero")
    return tps


def evaluate_baseline(target: str, gpus: int, gpu_type: str):
    slug = target.replace("/", "--") + f"__{gpus}x{gpu_type}"
    gpus_str = f"{gpus}x{gpu_type}"

    print(f"\n=== Baseline: {target} on {gpus_str} (pod {pod_name()}) ===")

    _register_teardown_signals()
    try:
        provision_pod(gpus, gpu_type)
        run_baseline_in_pod(target, gpus)
        baseline_dir = copy_baseline_out(slug)
        tps = parse_baseline_throughput(baseline_dir)
        upsert_baseline(target, gpus_str, tps)
        print(f"=== Baseline done: {target} @ {gpus_str} = {tps:.2f} tok/s ===")
    except Exception as e:
        print(f"ERROR baseline {target}: {e}")
    finally:
        teardown_pod()


def cleanup_stale_pod():
    pod = pod_name()
    try:
        result = subprocess.run(
            [
                "oc",
                "get",
                "pod",
                pod,
                "-n",
                NAMESPACE,
                "-o",
                "jsonpath={.metadata.creationTimestamp}",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return
        created = datetime.fromisoformat(result.stdout.strip().replace("Z", "+00:00"))
        age = (datetime.now(UTC) - created).total_seconds()
        if age > 14400:
            print(f"Deleting stale eval pod {pod} (age {int(age)}s)")
            subprocess.run(
                ["oc", "delete", "pod", pod, "-n", NAMESPACE, "--wait=false"]
            )
    except Exception:
        pass


def refresh():
    lock_path = REPO_ROOT / "logs" / "refresh.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another refresh is already running")
        return

    print(f"=== refresh started {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} ===")

    cleanup_stale_pod()

    run_cmd([sys.executable, str(SCRIPT_DIR / "discover.py")])

    pending_dir = REPO_ROOT / "results" / "pending"
    pending = json.loads((pending_dir / "index.json").read_text())
    failures = []
    for slug in pending:
        entry_path = pending_dir / f"{slug}.json"
        entry = json.loads(entry_path.read_text())
        try:
            evaluate_model(entry)
        except Exception as e:
            failures.append((entry["model"], e))
        entry_path.unlink(missing_ok=True)

    print(
        f"=== refresh finished {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} ==="
    )
    if failures:
        raise RuntimeError(f"{len(failures)} eval(s) failed: {failures}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("refresh", help="Discover + eval pending models (default)")

    bp = sub.add_parser(
        "baseline", help="Run autoregressive baseline for a target model"
    )
    bp.add_argument("--target", required=True, help="Target model HF ID")
    bp.add_argument("--gpus", type=int, required=True)
    bp.add_argument("--gpu-type", required=True)

    sp = sub.add_parser("single", help="Eval a single model from a pending entry JSON")
    sp.add_argument(
        "--entry",
        required=True,
        type=Path,
        help="Path to pending entry JSON (same schema as discover.py output)",
    )

    args = ap.parse_args()
    if args.cmd == "single":
        cleanup_stale_pod()
        entry = json.loads(args.entry.read_text())
        evaluate_model(entry)
    elif args.cmd == "baseline":
        evaluate_baseline(args.target, args.gpus, args.gpu_type)
    else:
        refresh()


if __name__ == "__main__":
    main()
