#!/usr/bin/env python3
"""Benchmark CPU reranking latency on synthetic embedding pools."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from tabulate import tabulate

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from retrieval.rerankers import (
    rerank_greedy_dpp,
    rerank_knn,
    rerank_maxmin,
    rerank_mmr,
    rerank_rng_score,
    rerank_rng_score2,
)

_DEFAULT_POOL_SIZES = [25, 50, 100, 200, 500]
_DEFAULT_DIM = 1024
_DEFAULT_REPEATS = 200
_DEFAULT_K = 5
_DEFAULT_METRIC = "cosine"


# ---------------------------------------------------------------------------
# Build methods to time
# ---------------------------------------------------------------------------

def _build_methods(k: int, metric: str) -> List[Tuple[str, object]]:
    """Return (name, fn) pairs.  fn(embs, q, scores, k) → list[int]."""
    return [
        ("kNN",
         lambda e, q, s, _k=k: rerank_knn(s, _k)),
        ("MMR(λ=0.5)",
         lambda e, q, s, _k=k: rerank_mmr(e, q, s, _k, 0.5)),
        ("Maxmin",
         lambda e, q, s, _k=k: rerank_maxmin(e, s, _k)),
        ("Greedy-DPP",
         lambda e, q, s, _k=k: rerank_greedy_dpp(e, s, _k)),
        ("RNG-Score(α=0)",
         lambda e, q, s, _k=k, _m=metric: rerank_rng_score(e, q, _k, 0.0, _m)),
        ("RNG-Score(α=0.1)",
         lambda e, q, s, _k=k, _m=metric: rerank_rng_score(e, q, _k, 0.1, _m)),
        ("Seg-Score(α=0)",
         lambda e, q, s, _k=k, _m=metric: rerank_rng_score2(e, q, _k, 0.0, _m)),
        ("Seg-Score(α=0.1)",
         lambda e, q, s, _k=k, _m=metric: rerank_rng_score2(e, q, _k, 0.1, _m)),
    ]


# ---------------------------------------------------------------------------
# Timing loop
# ---------------------------------------------------------------------------

def time_methods(
    pool_sizes: List[int],
    dim: int,
    repeats: int,
    k: int,
    metric: str,
    seed: int = 42,
) -> Dict[str, Dict[int, float]]:
    """Time each method over each pool size.

    Returns
    -------
    results : dict of {method_name: {pool_size: mean_latency_ms}}
    """
    methods = _build_methods(k, metric)
    rng = np.random.default_rng(seed)
    results: Dict[str, Dict[int, float]] = {name: {} for name, _ in methods}

    for m in pool_sizes:
        print(f"   Pool size m={m} …")
        latencies: Dict[str, List[float]] = {name: [] for name, _ in methods}

        for _ in range(repeats):
            embs = rng.standard_normal((m, dim)).astype(np.float32)
            q = rng.standard_normal(dim).astype(np.float32)
            # Normalise so cosine dot-product is well-defined
            embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            q /= np.linalg.norm(q) + 1e-12
            scores = (embs @ q).astype(np.float32)

            for name, fn in methods:
                t0 = time.perf_counter()
                fn(embs, q, scores, k)
                latencies[name].append((time.perf_counter() - t0) * 1000.0)

        for name in latencies:
            results[name][m] = float(np.mean(latencies[name]))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"   Saved: {path}")


def _build_table(results: Dict[str, Dict[int, float]], pool_sizes: List[int]) -> List[Dict]:
    """Convert results to a flat list of row dicts."""
    rows = []
    for method_name, size_map in results.items():
        row: Dict = {"Method": method_name}
        for m in pool_sizes:
            row[f"m={m}"] = f"{size_map.get(m, float('nan')):.2f}"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Scaling exponent estimation
# ---------------------------------------------------------------------------

def _estimate_scaling(results: Dict[str, Dict[int, float]], pool_sizes: List[int]) -> List[Dict]:
    """Fit a power law t ~ C * m^p to estimate empirical scaling exponent."""
    rows = []
    log_m = np.log(pool_sizes)
    for method_name, size_map in results.items():
        lats = [size_map.get(m, float("nan")) for m in pool_sizes]
        valid = [(lm, lt) for lm, lt in zip(log_m, lats) if not np.isnan(lt) and lt > 0]
        if len(valid) < 2:
            rows.append({"Method": method_name, "Slope (p)": "n/a", "R²": "n/a"})
            continue
        xs, ys = np.array([v[0] for v in valid]), np.log([v[1] for v in valid])
        p = np.polyfit(xs, ys, 1)
        y_pred = np.polyval(p, xs)
        ss_res = np.sum((ys - y_pred) ** 2)
        ss_tot = np.sum((ys - ys.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rows.append({"Method": method_name, "Slope (p)": f"{p[0]:.3f}", "R²": f"{r2:.3f}"})
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ5: efficiency benchmarks")
    p.add_argument("--pool_sizes", type=int, nargs="+", default=_DEFAULT_POOL_SIZES)
    p.add_argument("--dim", type=int, default=_DEFAULT_DIM,
                   help="Embedding dimension (default 1024 for bge-m3).")
    p.add_argument("--repeats", type=int, default=_DEFAULT_REPEATS,
                   help="Number of random queries per (method, pool_size) cell.")
    p.add_argument("--top_k", type=int, default=_DEFAULT_K)
    p.add_argument("--metric", choices=["cosine", "euclidean", "angular"],
                   default=_DEFAULT_METRIC)
    p.add_argument("--output_dir", default="results")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print(
        f"\n── Efficiency benchmark ──\n"
        f"   D={args.dim}, k={args.top_k}, repeats={args.repeats}, "
        f"pool_sizes={args.pool_sizes}, metric={args.metric}\n"
    )

    t0 = time.time()
    results = time_methods(
        pool_sizes=args.pool_sizes,
        dim=args.dim,
        repeats=args.repeats,
        k=args.top_k,
        metric=args.metric,
    )
    elapsed = time.time() - t0

    # Print table
    table = _build_table(results, args.pool_sizes)
    header = f"\n══ Mean latency (ms / query), D={args.dim} ══"
    print(header)
    print(tabulate(table, headers="keys", tablefmt="github"))

    # Print scaling exponents
    scaling = _estimate_scaling(results, args.pool_sizes)
    print("\n══ Empirical scaling exponent (t ~ m^p) ══")
    print(tabulate(scaling, headers="keys", tablefmt="github"))
    print(f"\n   Expected for RNG-Score / Seg-Score: p ≈ 2  (O(m²D))")
    print(f"   Total elapsed: {elapsed:.1f}s")

    # Save
    _save_csv(table, os.path.join(args.output_dir, "results_efficiency.csv"))
    _save_csv(scaling, os.path.join(args.output_dir, "results_efficiency_scaling.csv"))


if __name__ == "__main__":
    main()
