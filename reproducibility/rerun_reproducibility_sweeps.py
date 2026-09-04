#!/usr/bin/env python3
"""Run the 12 reported injection/chunking configurations under pool-size.

The driver is resumable: successful source run IDs recorded in the state file
are skipped. Historical result directories are read only and never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "retained"
OUTPUT = ROOT / "results" / "reproducibility_reruns"
LOGS = ROOT / "reproducibility_run_logs"
STATE = ROOT / "reproducibility" / "run_state.json"

# One SciFact `all` execution covers both injection and chunking, hence 12
# distinct source configurations rather than 13 experiment families.
RUNS = [
    ("2026-07-28_115804_redundancy_trec-covid", "redundancy"),
    ("2026-07-28_154431_redundancy_musique", "redundancy"),
    ("2026-07-23_101022_redundancy_musique", "chunking"),
    ("2026-07-16_095435_redundancy_trec-covid", "chunking"),
    ("2026-07-23_085711_redundancy_2wikimultihopqa", "chunking"),
    ("2026-07-28_170313_redundancy_2wikimultihopqa", "redundancy"),
    ("2026-07-28_095742_redundancy_fiqa", "chunking"),
    ("2026-06-25_082550_redundancy_scifact", "all"),
    ("2026-06-24_083656_redundancy_hotpotqa_fullwiki", "chunking"),
    ("2026-07-23_104202_redundancy_fiqa", "both"),
    ("2026-06-24_230733_redundancy_hotpotqa_fullwiki", "redundancy"),
    ("2026-06-16_094433_redundancy_nq", "chunking"),
]
LIST_KEYS = [
    "alpha_grid", "lambda_grid", "dedup_grid", "rho_grid", "overlap_grid",
]
SCALAR_KEYS = [
    "dataset", "split", "encoder_model", "device", "batch_size", "top_m",
    "top_k", "metric", "objective", "chunk_window", "dup_noise",
    "dup_target", "val_fraction",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state() -> dict:
    if not STATE.is_file():
        return {"format_version": 1, "created_at": now(), "runs": {}}
    return json.loads(STATE.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(STATE)


def command(source_run: str, experiment: str) -> tuple[list[str], Path]:
    params_path = RESULTS / source_run / "run_params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    cmd = [
        sys.executable, "-u", "-m", "experiments.evaluate_redundancy",
        "--experiment", experiment,
    ]
    for key in SCALAR_KEYS:
        value = params.get(key)
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    max_samples = params.get("max_samples")
    cmd.extend(["--max_samples", "all" if max_samples is None else str(max_samples)])
    for key in LIST_KEYS:
        values = params.get(key)
        if values:
            cmd.append(f"--{key}")
            cmd.extend(str(value) for value in values)
    seeds = params.get("seeds")
    if seeds:
        cmd.append("--seeds")
        cmd.extend(str(seed) for seed in seeds)
    elif params.get("seed") is not None:
        cmd.extend(["--seed", str(params["seed"])])
    if params.get("single_subtopic"):
        cmd.append("--single_subtopic")
    if params.get("encode_cache") is False:
        cmd.append("--no_encode_cache")
    if params.get("run_generation"):
        raise ValueError(
            f"{source_run} unexpectedly requests generation; pool-size reruns are retrieval sweeps"
        )
    cmd.extend(["--output_dir", str(OUTPUT)])
    return cmd, params_path


def output_directories() -> set[Path]:
    if not OUTPUT.is_dir():
        return set()
    return {path.resolve() for path in OUTPUT.iterdir() if path.is_dir()}


def run_one(source_run: str, experiment: str, state: dict) -> None:
    cmd, params_path = command(source_run, experiment)
    LOGS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{source_run}.log"
    before = output_directories()
    record = {
        "source_run": source_run,
        "source_run_params": params_path.relative_to(ROOT).as_posix(),
        "source_run_params_sha256": sha256_file(params_path),
        "experiment": experiment,
        "command": cmd,
        "command_shell": shlex.join(cmd),
        "status": "running",
        "started_at": now(),
        "log": log_path.relative_to(ROOT).as_posix(),
    }
    state["runs"][source_run] = record
    save_state(state)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] SOURCE {source_run}\n")
        log.write(f"[{now()}] COMMAND {shlex.join(cmd)}\n")
        log.flush()
        result = subprocess.run(
            cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            check=False,
        )
    after = output_directories()
    created = sorted(str(path) for path in after.difference(before))
    record.update({
        "finished_at": now(),
        "returncode": result.returncode,
        "status": "complete" if result.returncode == 0 else "failed",
        "created_run_directories": created,
    })
    save_state(state)
    if result.returncode != 0:
        raise RuntimeError(f"{source_run} failed; inspect {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="*", help="Optional source run IDs")
    args = parser.parse_args()
    selected = [item for item in RUNS if not args.only or item[0] in set(args.only)]
    state = load_state()
    for source_run, experiment in selected:
        cmd, _ = command(source_run, experiment)
        if args.dry_run:
            print(source_run)
            print("  " + shlex.join(cmd))
            continue
        existing = state["runs"].get(source_run, {})
        if existing.get("status") == "complete":
            print(f"SKIP complete: {source_run}")
            continue
        print(f"RUN {source_run} ({experiment})", flush=True)
        run_one(source_run, experiment, state)
        print(f"DONE {source_run}", flush=True)


if __name__ == "__main__":
    main()
