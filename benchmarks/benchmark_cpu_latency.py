#!/usr/bin/env python3
"""Reproducible single-query CPU latency benchmark for paper claims."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Optional, Set

# Set thread controls before importing NumPy/BLAS.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np

from evaluation.metrics import vendi_score

# Load the standalone reranker module without importing optional encoder
# dependencies exposed by retrieval/__init__.py.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RERANKERS_PATH = _PROJECT_ROOT / "retrieval" / "rerankers.py"
_RERANKERS_SPEC = importlib.util.spec_from_file_location("paper_rerankers", _RERANKERS_PATH)
if _RERANKERS_SPEC is None or _RERANKERS_SPEC.loader is None:
    raise RuntimeError(f"Could not load {_RERANKERS_PATH}")
_RERANKERS = importlib.util.module_from_spec(_RERANKERS_SPEC)
_RERANKERS_SPEC.loader.exec_module(_RERANKERS)
rerank_rng_score = _RERANKERS.rerank_rng_score


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10_000)
    parser.add_argument("--inputs", type=int, default=128)
    parser.add_argument(
        "--cpu", type=int, default=None,
        help="Logical CPU to pin. Defaults to the first CPU allowed for this process.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/cpu_latency")
    return parser.parse_args()


def _summarize(values_ns: list[int]) -> dict[str, float | int]:
    values_us = np.asarray(values_ns, dtype=np.float64) / 1_000.0
    return {
        "n": len(values_ns),
        "mean_us": float(values_us.mean()),
        "std_us": float(values_us.std(ddof=1)),
        "p50_us": float(np.percentile(values_us, 50)),
        "p90_us": float(np.percentile(values_us, 90)),
        "p95_us": float(np.percentile(values_us, 95)),
        "p99_us": float(np.percentile(values_us, 99)),
        "min_us": float(values_us.min()),
        "max_us": float(values_us.max()),
    }


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _current_affinity() -> Optional[Set[int]]:
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return set(os.sched_getaffinity(0))
    except OSError:
        return None


def _pin_cpu(requested_cpu: Optional[int]) -> Optional[int]:
    """Pin to an allowed CPU when supported, falling back without crashing."""
    if not hasattr(os, "sched_setaffinity"):
        return None

    allowed = _current_affinity()
    if not allowed:
        print("Warning: CPU affinity is unavailable; continuing without pinning.")
        return None

    selected_cpu = requested_cpu if requested_cpu in allowed else min(allowed)
    if requested_cpu is not None and requested_cpu not in allowed:
        print(
            f"Warning: CPU {requested_cpu} is not available to this process; "
            f"using CPU {selected_cpu}."
        )
    try:
        os.sched_setaffinity(0, {selected_cpu})
    except OSError as exc:
        print(f"Warning: could not pin CPU affinity ({exc}); continuing unpinned.")
        return None
    return selected_cpu


def main() -> None:
    args = _args()
    pinned_cpu = _pin_cpu(args.cpu)

    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError as exc:
        raise SystemExit("Install threadpoolctl to enforce and record one BLAS thread") from exc

    rng = np.random.default_rng(args.seed)
    pools = rng.standard_normal(
        (args.inputs, args.pool_size, args.dim), dtype=np.float32
    )
    queries = rng.standard_normal((args.inputs, args.dim), dtype=np.float32)
    pools /= np.linalg.norm(pools, axis=2, keepdims=True) + 1e-12
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12

    rng_times: list[int] = []
    vendi_times: list[int] = []
    with threadpool_limits(limits=1):
        for index in range(args.warmup):
            pool = pools[index % args.inputs]
            query = queries[index % args.inputs]
            rerank_rng_score(pool, query, args.top_k, 0.0, "cosine")
            vendi_score(pool[: args.top_k])

        # Alternate operations per input to reduce temporal bias.
        for index in range(args.repeats):
            pool = pools[index % args.inputs]
            query = queries[index % args.inputs]
            if index % 2 == 0:
                start = time.perf_counter_ns()
                rerank_rng_score(pool, query, args.top_k, 0.0, "cosine")
                rng_times.append(time.perf_counter_ns() - start)
                start = time.perf_counter_ns()
                vendi_score(pool[: args.top_k])
                vendi_times.append(time.perf_counter_ns() - start)
            else:
                start = time.perf_counter_ns()
                vendi_score(pool[: args.top_k])
                vendi_times.append(time.perf_counter_ns() - start)
                start = time.perf_counter_ns()
                rerank_rng_score(pool, query, args.top_k, 0.0, "cosine")
                rng_times.append(time.perf_counter_ns() - start)

        pools_info = threadpool_info()

    summaries = [
        {"operation": "RNG-Score (full)", **_summarize(rng_times)},
        {"operation": "Vendi trigger (full)", **_summarize(vendi_times)},
    ]
    metadata = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "requested_cpu": args.cpu,
        "cpu": pinned_cpu,
        "affinity": sorted(_current_affinity() or []),
        "cpu_details": _command_output(["lscpu"]),
        "threadpools": pools_info,
        "timer": "time.perf_counter_ns",
        "synchronization": "NumPy CPU calls are synchronous",
        "batch_size": 1,
        "dtype": "float32",
        "metric": "cosine",
        "pool_size": args.pool_size,
        "top_k": args.top_k,
        "dimension": args.dim,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "pre_generated_inputs": args.inputs,
        "seed": args.seed,
        "scope": "selection only; inputs precomputed and resident in host memory",
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "latencies_us.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "rng_score_us", "vendi_trigger_us"])
        for index, (rng_ns, vendi_ns) in enumerate(zip(rng_times, vendi_times)):
            writer.writerow([index, rng_ns / 1_000.0, vendi_ns / 1_000.0])
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    for row in summaries:
        print(
            f"{row['operation']}: mean={row['mean_us']:.1f} us, "
            f"p50={row['p50_us']:.1f} us, p95={row['p95_us']:.1f} us, "
            f"p99={row['p99_us']:.1f} us"
        )
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
