"""
Frozen-rule transfer summary (Table `tab:rq6-transfer` of the paper).

Evaluates the decision rule frozen at (tau, D) = (h, MMR(0.7)) -- the
evidence-anchored threshold: the rule diversifies when the Vendi score of
the k-NN selection falls below the query's evidence requirement h. On the
bge-m3 HotpotQA fullwiki injection sweep (both k=5 and k=10), validation
tuning of the threshold recovers h (tuned across-seed means 1.83 and 1.9,
results/2026-06-24_230733 and 2026-07-20_121145) -- on the transfer sweeps
listed in TARGETS, with no retuning, and writes one consolidated CSV:

    results/frozen_rule_transfer_summary.csv

Per (target, level): mean kNN / always-D / rule objective over seeds, the
trigger rate, and rule-minus-kNN deltas; per target: pooled values and the
clean/heavy/pooled deltas reported in the paper (cleanDelta = worst per-level
delta among near-clean levels, rho<=0.05 or overlap<=0.25; heavyDelta = delta
at the most redundant level).

The h specification per target (H_SPEC below):
    ("const", v)  -- fixed evidence requirement (2 on the two-hop HotpotQA
                     and 2WikiMultiHopQA sweeps, 1 on single-hop NQ-Open,
                     where the trigger T(q) >= 1 never fires and the rule
                     reduces to k-NN by construction);
    "qid_hops"    -- per-query hop count parsed from the qid prefix
                     (MuSiQue, 2--4 hops);
    "relset"      -- per-query relevant-set size (graded-relevance BEIR
                     tasks; uses labels in evaluation, estimators in
                     deployment).

The rule logic otherwise replicates analyze_regimes.threshold_rule exactly
(same validation/test split reconstruction, same Vendi trigger):

    python analyze_transfer_summary.py
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

FALLBACK = "MMR(0.7)"
T95 = 4.302652  # two-sided Student-t, 2 df (three seeds)

# (label, run_dir, per_query file, level column, h spec)
TARGETS = [
    ("HotpotQA bge-m3 (tuning run)", "2026-06-24_230733_redundancy_hotpotqa_fullwiki",
     "results_redundancy_per_query.csv", "rho", ("const", 2.0)),
    ("HotpotQA Qwen3-Emb (inj.)", "2026-06-25_172848_redundancy_hotpotqa_fullwiki",
     "results_redundancy_per_query.csv", "rho", ("const", 2.0)),
    ("HotpotQA MiniLM (inj.)", "2026-06-28_094117_redundancy_hotpotqa_fullwiki",
     "results_redundancy_per_query.csv", "rho", ("const", 2.0)),
    ("HotpotQA bge-m3 k=10 (inj.)", "2026-07-20_121145_redundancy_hotpotqa_fullwiki",
     "results_redundancy_per_query.csv", "rho", ("const", 2.0)),
    ("HotpotQA bge-m3 (chunking)", "2026-06-24_083656_redundancy_hotpotqa_fullwiki",
     "results_chunking_per_query.csv", "overlap", ("const", 2.0)),
    ("2WikiMultiHopQA (chunking)", "2026-07-23_085711_redundancy_2wikimultihopqa",
     "results_chunking_per_query.csv", "overlap", ("const", 2.0)),
    ("MuSiQue (chunking)", "2026-07-23_101022_redundancy_musique",
     "results_chunking_per_query.csv", "overlap", "qid_hops"),
    ("NQ-Open (chunking)", "2026-06-16_094433_redundancy_nq",
     "results_chunking_per_query.csv", "overlap", ("const", 1.0)),
    ("TREC-COVID (chunking)", "2026-07-16_095435_redundancy_trec-covid",
     "results_chunking_per_query.csv", "overlap", "relset"),
    ("SciFact bge-m3 (inj.)", "2026-06-25_082550_redundancy_scifact",
     "results_redundancy_per_query.csv", "rho", "relset"),
    ("FiQA-2018 (inj.)", "2026-07-23_104202_redundancy_fiqa",
     "results_redundancy_per_query.csv", "rho", "relset"),
    ("SciFact bge-m3 (chunking)", "2026-06-25_082550_redundancy_scifact",
     "results_chunking_per_query.csv", "overlap", "relset"),
]

KEEP = {"kNN", FALLBACK}


def load_filtered(path: str, lvl: str) -> pd.DataFrame:
    """Read only kNN / fallback rows (chunked: per-query files can be large)."""
    chunks = []
    for c in pd.read_csv(path, chunksize=2_000_000):
        chunks.append(c[c["Method"].isin(KEEP)])
    df = pd.concat(chunks, ignore_index=True)
    return df.dropna(subset=["Method"])


def val_split(qids_in_order, seed: int, val_fraction: float) -> set:
    """Reproduce the validation split of evaluate_redundancy.py."""
    n = len(qids_in_order)
    n_val = max(1, int(val_fraction * n))
    order = np.random.default_rng(seed).permutation(n)
    return {qids_in_order[int(i)] for i in order[:n_val]}


def per_seed(df: pd.DataFrame, lvl: str, hspec, seed: int,
             val_fraction: float) -> dict:
    d = df[df["seed"] == seed] if "seed" in df.columns else df
    first = sorted(d[lvl].unique())[0]
    qids = list(dict.fromkeys(d.loc[d[lvl] == first, "qid"]))
    val = val_split(qids, int(seed), val_fraction)
    obj = d.pivot_table(index=[lvl, "qid"], columns="Method",
                        values="S-Recall@k")
    knn = d[d.Method == "kNN"].set_index([lvl, "qid"])
    wide = obj.join(knn["Vendi"].rename("trigger"))
    if hspec == "qid_hops":
        wide["h"] = [float(re.match(r"(\d+)hop", q).group(1))
                     for q in wide.index.get_level_values("qid")]
    elif hspec == "relset":
        wide = wide.join(knn["RelSetSize"].rename("h"))
    else:
        wide["h"] = float(hspec[1])
    wt = wide[~wide.index.get_level_values("qid").isin(val)]
    fire = wt["trigger"] < wt["h"]
    out = {"levels": {}}
    for lev in sorted(d[lvl].unique()):
        wl, fl = wt.xs(lev, level=lvl), fire.xs(lev, level=lvl)
        out["levels"][lev] = {"kNN": wl["kNN"].mean(), "D": wl[FALLBACK].mean(),
                              "rule": wl[FALLBACK].where(fl, wl["kNN"]).mean(),
                              "trigger_rate": fl.mean()}
    out["pooled"] = {"kNN": wt["kNN"].mean(), "D": wt[FALLBACK].mean(),
                     "rule": wt[FALLBACK].where(fire, wt["kNN"]).mean(),
                     "trigger_rate": fire.mean()}
    return out


def mci(vals):
    m = float(np.mean(vals))
    hw = float(T95 * np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return m, hw


def main() -> None:
    rows = []
    for label, run, fname, lvl, hspec in TARGETS:
        run_dir = os.path.join(RESULTS, run)
        params = json.load(open(os.path.join(run_dir, "run_params.json")))
        vf = params.get("val_fraction", 0.2)
        df = load_filtered(os.path.join(run_dir, fname), lvl)
        seeds = (sorted(df["seed"].unique()) if "seed" in df.columns
                 else [params.get("seed", 0)])
        res = {s: per_seed(df, lvl, hspec, s, vf) for s in seeds}
        levels = sorted(res[seeds[0]]["levels"])
        keys = list(res[seeds[0]]["levels"][levels[0]])
        hlabel = ("h(q)" if hspec == "qid_hops"
                  else "RelSetSize" if hspec == "relset" else hspec[1])
        for lev in levels:
            row = {"target": label, "run": run, "level_col": lvl,
                   "level": lev, "n_seeds": len(seeds), "tau": hlabel}
            for key in keys:
                m, hw = mci([res[s]["levels"][lev][key] for s in seeds])
                row[key], row[key + "_ci95"] = round(m, 4), round(hw, 4)
            rows.append(row)
        row = {"target": label, "run": run, "level_col": lvl,
               "level": "pooled", "n_seeds": len(seeds), "tau": hlabel}
        for key in keys:
            m, hw = mci([res[s]["pooled"][key] for s in seeds])
            row[key], row[key + "_ci95"] = round(m, 4), round(hw, 4)
        clean = [lev for lev in levels if lev <= (0.05 if lvl == "rho" else 0.25)]
        deltas = {lev: np.mean([res[s]["levels"][lev]["rule"]
                                - res[s]["levels"][lev]["kNN"] for s in seeds])
                  for lev in levels}
        row["cleanDelta"] = round(float(min(deltas[lev] for lev in clean)), 4)
        row["heavyDelta"] = round(float(deltas[max(levels)]), 4)
        row["pooledDelta"] = round(row["rule"] - row["kNN"], 4)
        rows.append(row)
        print(f"{label}: cleanDelta {row['cleanDelta']:+.4f}  "
              f"heavyDelta {row['heavyDelta']:+.4f}  pooledDelta {row['pooledDelta']:+.4f}")
    out = os.path.join(RESULTS, "frozen_rule_transfer_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
