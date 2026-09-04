"""Recompute sweep summaries from retained per-query metrics.

The command selects S-Recall on validation data and does not rerun retrieval,
encoding, or generation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from experiments.evaluate_redundancy import (  # noqa: E402
    _aggregate_sweep_summary,
    _pad_rows,
    _seed_summary_rows,
)
from ftrb.run_utils import attach_run_params, save_csv  # noqa: E402

OBJECTIVE = "S-Recall@k"
METRIC_KEYS = ["Recall@k", "NDCG@k", "MRR", "alpha-NDCG@k", "S-Recall@k",
               "ERR-IA@k", "APD", "Vendi"]

SWEEP_KINDS = [
    ("results_redundancy_per_query.csv",
     "results_redundancy_per_seed_summary.csv", "rho", "redundancy"),
    ("results_chunking_per_query.csv",
     "results_chunking_per_seed_summary.csv", "overlap", "chunking"),
]

RESULT_ROOTS = [
    os.path.abspath(os.path.join(HERE, "..", "results")),
    os.path.abspath(os.path.join(HERE, "results")),
]


# ---------------------------------------------------------------------------

def find_redundancy_runs(roots: List[str]) -> List[Tuple[str, Dict]]:
    """Every run dir written by evaluate_redundancy.py, across both roots."""
    runs: List[Tuple[str, Dict]] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            rp = os.path.join(d, "run_params.json")
            if not os.path.isfile(rp):
                continue
            try:
                params = json.load(open(rp, encoding="utf-8"))
            except Exception:
                continue
            if params.get("script") == "evaluate_redundancy.py":
                runs.append((d, params))
    return runs


def _unique_levels(path: str, col: str) -> List[float]:
    return sorted(pd.read_csv(path, usecols=[col])[col].dropna().unique().tolist())


def completeness(pq_path: str, ps_path: str, level_col: str) -> Tuple[bool, str]:
    """Is the per-query file complete? Compare its levels to the per_seed_summary
    (which always carries every level of the run)."""
    pq_levels = set(_unique_levels(pq_path, level_col))
    if not os.path.isfile(ps_path):
        return True, f"levels={sorted(pq_levels)} (no per_seed_summary to cross-check)"
    ps_levels = set(pd.read_csv(ps_path, usecols=[level_col])[level_col].dropna().unique())
    if pq_levels >= ps_levels:
        return True, f"levels={sorted(pq_levels)}"
    missing = sorted(ps_levels - pq_levels)
    return False, f"TRUNCATED: per_query has {sorted(pq_levels)}, missing {missing}"


def run_analyze_regimes(pq_path: str, level_col: str, top_k: int,
                        out_prefix: str) -> Tuple[int, str]:
    cmd = [sys.executable, os.path.join(HERE, "analyze_regimes.py"),
           "--per_query", pq_path, "--objective", OBJECTIVE, "--top_k", str(top_k),
           "--out_prefix", out_prefix]
    if level_col != "rho":
        cmd += ["--level_col", level_col]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    tail = (proc.stdout or "")[-400:] + (proc.stderr or "")[-400:]
    return proc.returncode, tail


def regenerate_sweep_summary(pq_path: str, level_col: str, file_tag: str,
                             params: Dict, out_dir: str) -> None:
    """Rewrite results_<tag>_summary.csv / _per_seed_summary.csv on S-Recall@k
    from the per-query rows, reusing evaluate_redundancy's own aggregation so the
    format is identical to a native run."""
    df = pd.read_csv(pq_path)
    if "seed" not in df.columns:
        df["seed"] = int(params.get("seed", 0))
    cfg = {"alpha_grid": params["alpha_grid"],
           "lambda_grid": params["lambda_grid"],
           "val_fraction": float(params.get("val_fraction", 0.2))}

    per_seed_rows: List[Dict] = []
    for (seed, level), block in df.groupby(["seed", level_col], sort=True):
        qids = list(dict.fromkeys(block["qid"]))          # example order
        pos = {q: i for i, q in enumerate(qids)}
        n = len(qids)
        pivots = {m: block.pivot_table(index="qid", columns="Method",
                                       values=m, sort=False).reindex(qids)
                  for m in METRIC_KEYS}
        methods = list(pivots[METRIC_KEYS[0]].columns)
        per_method: Dict[str, List[Dict[str, float]]] = {}
        cols = {m: {name: pivots[m][name].to_numpy() for name in methods}
                for m in METRIC_KEYS}
        for name in methods:
            per_method[name] = [{m: float(cols[m][name][i]) for m in METRIC_KEYS}
                                for i in range(n)]
        if "PoolRedundancy" in block.columns:
            knn_block = block[block.Method == "kNN"]["PoolRedundancy"]
            mean_red = float(knn_block.mean()) if len(knn_block) else 0.0
        else:
            mean_red = 0.0  # older per-query files predate PoolRedundancy logging
        per_seed_rows.extend(_seed_summary_rows(
            per_method, n, cfg, float(level), level_col, mean_red,
            int(seed), OBJECTIVE))

    summary = _aggregate_sweep_summary(per_seed_rows, level_col, OBJECTIVE)
    stamp = {**params, "objective": OBJECTIVE, "reanalyzed_objective": OBJECTIVE}
    save_csv(attach_run_params(_pad_rows(summary), stamp),
             os.path.join(out_dir, f"results_{file_tag}_summary.csv"))
    save_csv(attach_run_params(_pad_rows(per_seed_rows), stamp),
             os.path.join(out_dir, f"results_{file_tag}_per_seed_summary.csv"))


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry_run", action="store_true",
                    help="Report what would be done without running anything.")
    ap.add_argument("--force", action="store_true",
                    help="Re-analyze even runs already on S-Recall@k with a "
                         "current-format analysis (default skips those).")
    ap.add_argument("--roots", nargs="+", default=RESULT_ROOTS)
    args = ap.parse_args()

    runs = find_redundancy_runs(args.roots)
    report: List[str] = []
    n_ok = n_skip = n_regen = n_current = 0

    def emit(line: str) -> None:
        report.append(line)
        print(line, flush=True)

    for run_dir, params in runs:
        top_k = int(params.get("top_k", 5))
        obj = params.get("objective")
        obj_is_andcg = (obj == "alpha-NDCG@k") or (obj == ["alpha_ndcg"])
        label = os.path.relpath(run_dir, HERE)
        has_redundancy = os.path.isfile(os.path.join(run_dir, SWEEP_KINDS[0][0]))
        for pq_name, ps_name, level_col, tag in SWEEP_KINDS:
            pq = os.path.join(run_dir, pq_name)
            if not os.path.isfile(pq):
                continue
            # An experiment=all dir holds BOTH redundancy and chunking per-query;
            # give the chunking analysis a distinct prefix so it does not clobber
            # the redundancy analysis_*.csv in the same dir.
            out_prefix = ("analysis_chunking"
                          if tag == "chunking" and has_redundancy else "analysis")
            marker = os.path.join(run_dir, f"{out_prefix}_crossover.csv")
            ps = os.path.join(run_dir, ps_name)
            ok, detail = completeness(pq, ps, level_col)
            if not ok:
                emit(f"[SKIP ] {label} [{tag}] {detail}")
                n_skip += 1
                continue
            if args.dry_run:
                extra = "  (+regen sweep summary)" if obj_is_andcg else ""
                emit(f"[WOULD] {label} [{tag}] obj={obj} -> {OBJECTIVE}{extra}")
                continue
            try:
                # Skip re-analysis only when already correct AND already current
                # (a current-format run has the crossover output). alpha-NDCG runs
                # are always re-analyzed + their sweep summary rewritten.
                already_current = (not obj_is_andcg) and os.path.isfile(marker)
                if not (already_current and not args.force):
                    rc, tail = run_analyze_regimes(pq, level_col, top_k, out_prefix)
                    if rc != 0:
                        emit(f"[FAIL ] {label} [{tag}] analyze_regimes rc={rc}\n{tail}")
                        n_skip += 1
                        continue
                    n_ok += 1
                    msg = f"[OK   ] {label} [{tag}] {out_prefix}_* on {OBJECTIVE}"
                else:
                    n_current += 1
                    msg = f"[CURR ] {label} [{tag}] already S-Recall + current; skipped"
                if obj_is_andcg:
                    regenerate_sweep_summary(pq, level_col, tag, params, run_dir)
                    n_regen += 1
                    msg += " + sweep summary rewritten (was alpha-NDCG@k)"
                emit(msg)
            except Exception as exc:  # keep going; one bad run must not abort all
                import traceback
                emit(f"[ERROR] {label} [{tag}] {exc.__class__.__name__}: {exc}\n"
                     f"{traceback.format_exc()[-600:]}")
                n_skip += 1

    # runs that cannot be fixed locally (no per-query file at all)
    for run_dir, params in runs:
        has_pq = any(os.path.isfile(os.path.join(run_dir, k[0])) for k in SWEEP_KINDS)
        if not has_pq:
            emit(f"[NOPQ ] {os.path.relpath(run_dir, HERE)} "
                 f"obj={params.get('objective')} — no per-query file, "
                 f"cannot re-analyze locally")

    print(f"\nSummary: {n_ok} analyzed, {n_current} already-current (skipped), "
          f"{n_regen} sweep-summaries rewritten, {n_skip} skipped (truncated/failed).")


if __name__ == "__main__":
    main()
