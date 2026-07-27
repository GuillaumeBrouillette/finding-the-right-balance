"""Re-tune the cross-encoder (RQ4) table on s_recall, offline, no GPU re-run.

`evaluate_cross_encoder.py` selects each S1/S2 strategy's alpha*/beta* by
maximising an objective on the validation split. The RQ4 run was tuned on
`alpha_ndcg`; the QA tables tune on S-Recall, so for consistency we re-select on
S-Recall. This needs no re-run: the retrieval per-query CSV already logs every
grid member's per-query metrics on both splits (with a `Split` column), so the
only thing that changes is which member each family selects.

For each seed: on `Split==val`, pick the S1/S2 member maximising mean
S-Recall@k; on `Split==test`, report each method's metric means. Aggregate across
seeds to mean +/- 95% CI with a median Wilcoxon p-value vs CE-topk (retrieval
objectives + EM/F1 when generation is present). Rows are paired by position
within (seed, Method) — the per-query file has no qid column, but rows are in
example order per method, so CE-topk and any method align by row.

Usage::

    python reanalyze_ce.py --run_dir ../results/2026-06-24_064052_ce_hotpotqa_fullwiki
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from evaluate_cross_encoder import (  # noqa: E402
    _CE_BASELINE,
    _CE_GEN_METRICS,
    _CE_RET_METRICS,
    _ce_method_key,
)
from run_utils import attach_run_params, save_csv  # noqa: E402
from stats import aggregate_seed_rows, paired_wilcoxon_p  # noqa: E402

OBJ = "S-Recall@k"
SIG_RET = ["alpha-NDCG@k", "S-Recall@k"]
FAM_ORDER = ["S1-V1-Blend", "S1-V2-Blend",
             "S2-V1", "S2-V2", "S2-V3", "S2-V4", "S2-V5"]


def _display(raw: str) -> str:
    """S1-V1-Blend(α=X,β=Y) -> S1-V1-Blend(α*=X,β*=Y); S2-Vn(α=X) -> (α*=X)."""
    return raw.replace("α=", "α*=").replace("β=", "β*=")


def build_per_seed(ret: pd.DataFrame, gen: pd.DataFrame,
                   run_gen: bool) -> List[Dict]:
    """One test-split summary row per (seed, method), with S1/S2 re-selected on
    validation S-Recall@k. Fixed methods (CE-topk/CE-MMR/CE-DPP) are untuned."""
    seeds = sorted(int(s) for s in ret["seed"].unique())
    methods = list(dict.fromkeys(ret["Method"].astype(str)))
    fixed = [m for m in methods if not (m.startswith("S1-") or m.startswith("S2-"))]

    def fkey(m: str):
        rank = 0 if m == "CE-topk" else 1 if m.startswith("CE-MMR") else 2 if m == "CE-DPP" else 3
        return (rank, m)
    fixed.sort(key=fkey)
    tuned = [f for f in FAM_ORDER if any(_ce_method_key(m) == f for m in methods)]

    ret_cols = [c for c in _CE_RET_METRICS if c in ret.columns]
    test = ret[ret["Split"] == "test"]
    val = ret[ret["Split"] == "val"]
    test_mean = test.groupby(["seed", "Method"], observed=True)[ret_cols].mean()
    val_sr = val.groupby(["seed", "Method"], observed=True)[OBJ].mean()
    gtest_mean = None
    gmet: List[str] = []
    if run_gen and gen is not None:
        gmet = [c for c in _CE_GEN_METRICS if c in gen.columns]
        gtest_mean = (gen[gen["Split"] == "test"]
                      .groupby(["seed", "Method"], observed=True)[gmet].mean())

    rows: List[Dict] = []
    for seed in seeds:
        def add(method_key: str, raw: str, display: str) -> None:
            row: Dict[str, object] = {"Method": display, "MethodKey": method_key,
                                      "RawMethod": raw, "seed": seed}
            if (seed, raw) in test_mean.index:
                for c in ret_cols:
                    row[c] = round(float(test_mean.loc[(seed, raw), c]), 4)
            if gtest_mean is not None and (seed, raw) in gtest_mean.index:
                for c in gmet:
                    row[c] = round(float(gtest_mean.loc[(seed, raw), c]), 4)
            rows.append(row)

        for m in fixed:
            add(m, m, m)
        for fam in tuned:
            members = [m for m in methods if _ce_method_key(m) == fam]
            scored = [(val_sr.get((seed, m), float("nan")), m) for m in members]
            scored = [(s, m) for s, m in scored if s == s]  # drop NaN val scores
            if not scored:
                continue
            best = max(scored)[1]
            add(fam, best, _display(best))
    return rows


def aggregate(per_seed: List[Dict], ret: pd.DataFrame, gen: pd.DataFrame,
              run_gen: bool) -> List[Dict]:
    """Across-seed mean +/- 95% CI per MethodKey, with median Wilcoxon p vs
    CE-topk on the intent-aware objectives (+ EM/F1 when generation is on)."""
    metric_cols = list(_CE_RET_METRICS) + (list(_CE_GEN_METRICS) if run_gen else [])
    agg = aggregate_seed_rows(per_seed, key_cols=["MethodKey"],
                              metric_cols=metric_cols, seed_col="seed",
                              keep_cols=["Method"])
    # Position-aligned test groups (rows stay in example order within a group).
    tg = {k: v for k, v in ret[ret["Split"] == "test"]
          .groupby(["seed", "Method"], observed=True)}
    gg = {}
    if run_gen and gen is not None:
        gg = {k: v for k, v in gen[gen["Split"] == "test"]
              .groupby(["seed", "Method"], observed=True)}

    by_key: Dict[str, List[Dict]] = {}
    for r in per_seed:
        by_key.setdefault(r["MethodKey"], []).append(r)

    for row in agg:
        grp = by_key.get(row["MethodKey"], [])
        raws = list(dict.fromkeys(g["RawMethod"] for g in grp))
        row["alpha*(per seed)"] = ";".join(raws) if len(raws) != 1 else raws[0]
        if row["MethodKey"] == _CE_BASELINE:
            continue
        specs = [(m, tg) for m in SIG_RET]
        if run_gen:
            specs += [(m, gg) for m in _CE_GEN_METRICS]
        for metric, groups in specs:
            ps: List[float] = []
            for g in grp:
                a = groups.get((g["seed"], g["RawMethod"]))
                b = groups.get((g["seed"], _CE_BASELINE))
                if a is None or b is None:
                    continue
                av = pd.to_numeric(a[metric], errors="coerce").to_numpy()
                bv = pd.to_numeric(b[metric], errors="coerce").to_numpy()
                if len(av) == len(bv) and len(av):
                    p = paired_wilcoxon_p(av, bv)
                    if p is not None:
                        ps.append(p)
            if ps:
                row[f"p({metric} vs {_CE_BASELINE})"] = round(float(np.median(ps)), 6)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run_dir", required=True,
                    help="Cross-encoder run directory to re-tune in place.")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    params = json.load(open(os.path.join(run_dir, "run_params.json"), encoding="utf-8"))
    ds = params["dataset"]
    ret_path = os.path.join(run_dir, f"results_ce_{ds}_retrieval_per_query.csv")
    gen_path = os.path.join(run_dir, f"results_ce_{ds}_generation_per_query.csv")
    if not os.path.isfile(ret_path):
        raise SystemExit(f"No retrieval per-query CSV in {run_dir}; re-run with "
                         f"--save_per_query, or re-run evaluate_cross_encoder on GPU.")

    print(f"Loading {os.path.basename(ret_path)} …")
    ret = pd.read_csv(ret_path, dtype={"Method": "category", "Split": "category"})
    run_gen = bool(params.get("run_generation")) and os.path.isfile(gen_path)
    gen = (pd.read_csv(gen_path, dtype={"Method": "category", "Split": "category"})
           if run_gen else None)
    print(f"   {len(ret)} retrieval rows, seeds={sorted(ret['seed'].unique())}, "
          f"generation={run_gen}")

    per_seed = build_per_seed(ret, gen, run_gen)
    summary = aggregate(per_seed, ret, gen, run_gen)
    stamp = {**params, "objective": ["s_recall"], "reanalyzed_objective": OBJ}
    save_csv(attach_run_params(summary, stamp),
             os.path.join(run_dir, f"results_ce_{ds}_summary.csv"))
    save_csv(attach_run_params(per_seed, stamp),
             os.path.join(run_dir, f"results_ce_{ds}_per_seed_summary.csv"))
    print(f"Re-tuned CE table on {OBJ} → results_ce_{ds}_summary.csv "
          f"({len(summary)} methods).")


if __name__ == "__main__":
    main()
