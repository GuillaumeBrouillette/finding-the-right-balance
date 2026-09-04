"""Derive tuned, regret, oracle, threshold, and generation analyses.

The input is a per-query redundancy or chunking sweep produced by
``evaluate_redundancy.py``. Outputs are written beside the input file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ftrb.stats import as_float, seed_interval

try:
    from scipy.stats import wilcoxon as _wilcoxon
except ImportError:  # pragma: no cover
    _wilcoxon = None


# Seg-Score is outside the reported analysis families.
FAMILIES = ("Dedup", "MMR", "VendiG", "RNG")
SINGLETONS = ("Maxmin", "Greedy-DPP")


# ---------------------------------------------------------------------------
# Across-seed aggregation
# ---------------------------------------------------------------------------

def seeds_in(df: pd.DataFrame, fallback: int) -> List[int]:
    """Seeds present in the per-query frame (multi-seed runs tag every row
    with a ``seed`` column); fall back to the single run seed otherwise."""
    if "seed" in df.columns:
        return sorted(int(s) for s in df["seed"].dropna().unique())
    return [int(fallback)]


def aggregate_frames(frames: List[pd.DataFrame], key_cols: List[str]) -> pd.DataFrame:
    """Combine per-seed analysis frames into one with across-seed mean and CI.

    For every group keyed by ``key_cols`` and every numeric column, the
    output carries the across-seed mean and a ``<col>_ci95`` half-width
    (Student-t).  p-value columns (``p(...)``) are summarised by their
    across-seed median instead of a mean+CI.  Non-numeric columns are carried
    by first value.  Group order follows first appearance across the frames,
    so the canonical filenames keep their familiar row order and the plotting
    code keeps working (mean lives in the original column name).
    """
    big = pd.concat(frames, ignore_index=True)
    order: List[tuple] = []
    for f in frames:
        for key in (tuple(r[c] for c in key_cols) for _, r in f.iterrows()):
            if key not in order:
                order.append(key)
    value_cols = [c for c in big.columns if c not in key_cols]
    rows: List[Dict] = []
    for key in order:
        mask = np.ones(len(big), dtype=bool)
        for c, v in zip(key_cols, key):
            mask &= (big[c] == v).to_numpy()
        grp = big[mask]
        row: Dict[str, object] = dict(zip(key_cols, key))
        row["n_seeds"] = int(len(grp))
        for c in value_cols:
            numeric = pd.to_numeric(grp[c], errors="coerce")
            if numeric.notna().any():
                vals = numeric.dropna().to_numpy()
                if c.startswith("p("):
                    row[c] = round(float(np.median(vals)), 6)
                else:
                    mean, half, _ = seed_interval(vals)
                    row[c] = round(mean, 4)
                    row[c + "_ci95"] = round(half, 4)
            else:
                non_null = grp[c].dropna()
                row[c] = non_null.iloc[0] if len(non_null) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def crossover_estimate(summary: pd.DataFrame, level_col: str,
                       objective: str) -> float:
    """Redundancy level at which diversification overtakes kNN.

    From the validation-tuned summary, ``delta(level)`` is the best tuned
    diversifier's test objective minus kNN's.  The crossover is the level of
    the first sign change from ``delta <= 0`` to ``delta > 0``, linearly
    interpolated for a continuous estimate.  Returns the smallest level if
    diversification already helps there, or NaN if it never overtakes kNN in
    the swept range (so the caller can report ">max").
    """
    s = summary.copy()
    s[objective] = pd.to_numeric(s[objective], errors="coerce")
    levels = sorted(pd.to_numeric(s[level_col], errors="coerce").dropna().unique())
    deltas: List[Tuple[float, float]] = []
    for lv in levels:
        g = s[pd.to_numeric(s[level_col], errors="coerce") == lv]
        knn = g[g.Method == "kNN"][objective]
        div = g[g.Method != "kNN"][objective]
        if knn.empty or div.dropna().empty:
            continue
        deltas.append((float(lv), float(div.max()) - float(knn.iloc[0])))
    for i in range(1, len(deltas)):
        (x0, d0), (x1, d1) = deltas[i - 1], deltas[i]
        if d0 <= 0 < d1:
            t = -d0 / (d1 - d0) if d1 != d0 else 0.0
            return round(x0 + t * (x1 - x0), 5)
    if deltas and deltas[0][1] > 0:
        return deltas[0][0]
    return float("nan")


def _aggregate_info(infos: List[Dict]) -> Dict[str, object]:
    """Combine per-seed threshold-rule ``info`` dicts.

    Numeric fields (tau, pooled means, headroom captured) become an
    across-seed mean with a ``<field>_ci95`` half-width; categorical fields
    (the fallback diversifier ``D``, the trigger metric) report the modal
    value, flagged when it varied across seeds.
    """
    out: Dict[str, object] = {"n_seeds": len(infos)}
    keys = list(infos[0].keys())
    for key in keys:
        vals = [info.get(key) for info in infos]
        numeric = [as_float(v) for v in vals]
        if all(v is not None for v in numeric):
            mean, half, _ = seed_interval(numeric)
            out[key] = round(mean, 5)
            out[key + "_ci95"] = round(half, 5)
        else:
            uniq = list(dict.fromkeys(map(str, vals)))
            out[key] = uniq[0] if len(uniq) == 1 else ";".join(uniq)
    return out


# ---------------------------------------------------------------------------
# Loading and split reconstruction
# ---------------------------------------------------------------------------

# Apply the reported method scope consistently to summaries and oracles.
EXCLUDED_METHOD_PATTERN = r"^Seg\("


def load_per_query(path: str, level_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if level_col not in df.columns:
        raise SystemExit(
            f"Column '{level_col}' not found in {path}; "
            f"available: {list(df.columns)}"
        )
    n0 = len(df)
    df = df[~df["Method"].str.match(EXCLUDED_METHOD_PATTERN)]
    if len(df) < n0:
        print(f"   [exclude] dropped {n0 - len(df)} rows matching "
              f"{EXCLUDED_METHOD_PATTERN!r}")
    return df


def example_order(df: pd.DataFrame, level_col: str) -> List[str]:
    """Recover the example order of the run: order of first appearance of
    qids at the first sweep level (rows were written in example order)."""
    first_level = sorted(df[level_col].unique())[0]
    qids = df.loc[df[level_col] == first_level, "qid"]
    return list(dict.fromkeys(qids))


def val_test_qids(
    qids_in_order: Sequence[str], seed: int, val_fraction: float
) -> Tuple[set, set]:
    """Reproduce the validation/test split of evaluate_redundancy.py."""
    n = len(qids_in_order)
    n_val = max(1, int(val_fraction * n))
    order = np.random.default_rng(seed).permutation(n)
    val_ids, test_ids = order[:n_val], order[n_val:]
    if len(test_ids) == 0:
        test_ids = val_ids
    val = {qids_in_order[int(i)] for i in val_ids}
    test = {qids_in_order[int(i)] for i in test_ids}
    return val, test


def grid_members(methods: Sequence[str], prefix: str) -> List[str]:
    pat = re.compile(rf"^{prefix}\((-?[\d.]+)\)$")
    members = [(float(pat.match(m).group(1)), m)
               for m in methods if pat.match(m)]
    return [m for _, m in sorted(members)]


# ---------------------------------------------------------------------------
# Analysis 1 — validation-tuned summary at each level
# ---------------------------------------------------------------------------

def level_redundancy(df: pd.DataFrame, level_col: str,
                     run_dir: str) -> Dict[float, float]:
    """Mean measured PoolRedundancy per sweep level.

    Newer runs log it per query; for older per-query files, fall back to the
    run's own summary CSV, which always carried the per-level mean."""
    if "PoolRedundancy" in df.columns:
        knn = df[df.Method == "kNN"]
        return knn.groupby(level_col)["PoolRedundancy"].mean().to_dict()
    for name in os.listdir(run_dir):
        if name.startswith("results_") and name.endswith("_summary.csv"):
            s = pd.read_csv(os.path.join(run_dir, name))
            if level_col in s.columns and "PoolRedundancy" in s.columns:
                return s.groupby(level_col)["PoolRedundancy"].first().to_dict()
    return {}


def tuned_summary(
    df: pd.DataFrame, level_col: str, objective: str,
    val: set, test: set, metrics: List[str],
    red_map: Optional[Dict[float, float]] = None,
) -> pd.DataFrame:
    rows = []
    methods = df["Method"].unique()
    for level, g in df.groupby(level_col, sort=True):
        gval = g[g.qid.isin(val)]
        gtest = g[g.qid.isin(test)]
        # Stable family labels (MMR*, RNG*, Seg*) so rows aggregate across
        # seeds even when the validation-tuned member differs; the chosen
        # member is recorded separately in "Chosen".
        report: List[Tuple[str, str]] = [("kNN", "kNN")]
        for s in SINGLETONS:
            report.append((s, s))
        for fam in FAMILIES:
            members = grid_members(methods, fam)
            if not members:
                continue
            means = gval[gval.Method.isin(members)].groupby("Method")[objective].mean()
            report.append((f"{fam}*", means.idxmax()))
        knn_t = gtest[gtest.Method == "kNN"].set_index("qid")[objective]
        for label, name in report:
            sub = gtest[gtest.Method == name]
            row: Dict[str, object] = {level_col: level, "Method": label,
                                      "Chosen": "" if name == label else name,
                                      "n_test": len(sub)}
            if red_map and level in red_map:
                row["PoolRedundancy"] = round(float(red_map[level]), 4)
            for met in metrics:
                row[met] = round(float(sub[met].mean()), 4)
            if name != "kNN" and _wilcoxon is not None:
                vals = sub.set_index("qid")[objective].reindex(knn_t.index)
                diffs = vals.to_numpy() - knn_t.to_numpy()
                row[f"p({objective} vs kNN)"] = (
                    round(float(_wilcoxon(diffs).pvalue), 6)
                    if np.any(diffs != 0.0) else 1.0
                )
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 2 — mis-specification regret
# ---------------------------------------------------------------------------

def regret_table(
    df: pd.DataFrame, level_col: str, objective: str, test: set,
) -> pd.DataFrame:
    rows = []
    methods = df["Method"].unique()
    for level, g in df.groupby(level_col, sort=True):
        gtest = g[g.qid.isin(test)]
        knn = float(gtest[gtest.Method == "kNN"][objective].mean())
        fam_members = {fam: grid_members(methods, fam) for fam in FAMILIES}
        fam_members.update({s: [s] for s in SINGLETONS})
        for fam, members in fam_members.items():
            if not members:
                continue
            means = gtest[gtest.Method.isin(members)].groupby("Method")[objective].mean()
            best_m, worst_m = means.idxmax(), means.idxmin()
            rows.append({
                level_col: level, "Family": fam, "kNN": round(knn, 4),
                "best member": best_m, "best": round(float(means.max()), 4),
                "worst member": worst_m, "worst": round(float(means.min()), 4),
                "regret(best)": round(knn - float(means.max()), 4),
                "regret(worst)": round(knn - float(means.min()), 4),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 3 — per-query oracle headroom at each level
# ---------------------------------------------------------------------------

def oracle_table(
    df: pd.DataFrame, level_col: str, objective: str,
    val: set, test: set,
) -> pd.DataFrame:
    rows = []
    methods = df["Method"].unique()
    div_methods = [m for m in methods if m != "kNN"]
    for level, g in df.groupby(level_col, sort=True):
        piv = g.pivot_table(index="qid", columns="Method", values=objective)
        pval, ptest = piv[piv.index.isin(val)], piv[piv.index.isin(test)]
        knn_t = ptest["kNN"]
        rows.append({level_col: level, "Selector": "kNN",
                     objective: round(float(knn_t.mean()), 4)})
        for fam in FAMILIES:
            members = grid_members(methods, fam)
            if not members:
                continue
            best = pval[members].mean().idxmax()
            fixed_t = ptest[best]
            oracle_t = ptest[members].max(axis=1)
            rows.append({level_col: level, "Selector": f"{fam} fixed [{best}]",
                         objective: round(float(fixed_t.mean()), 4)})
            rows.append({
                level_col: level, "Selector": f"{fam} oracle",
                objective: round(float(oracle_t.mean()), 4),
                "oracle - fixed": round(float((oracle_t - fixed_t).mean()), 4),
                "oracle - kNN": round(float((oracle_t - knn_t).mean()), 4),
                "% beats kNN": round(float((oracle_t > knn_t).mean()) * 100, 1),
            })
        all_oracle_t = ptest[list(methods)].max(axis=1)
        rows.append({
            level_col: level, "Selector": "All-methods oracle",
            objective: round(float(all_oracle_t.mean()), 4),
            "oracle - kNN": round(float((all_oracle_t - knn_t).mean()), 4),
            "% beats kNN": round(float((all_oracle_t > knn_t).mean()) * 100, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 4 — redundancy-threshold decision rule
# ---------------------------------------------------------------------------

def threshold_rule(
    df: pd.DataFrame, level_col: str, objective: str,
    val: set, test: set, trigger_metric: str = "Vendi",
    top_k: int = 5,
    freeze_tau: Optional[float] = None,
    freeze_d: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Tune and evaluate the rule:  diversify iff trigger(kNN top-k) < tau.

    The trigger is the per-query *Vendi score of the kNN selection* (the
    effective number of distinct documents among the k selected), which is
    label-free and computed from embeddings the system already holds.  Both
    tau and the fallback diversifier D are selected on the pooled validation
    split across levels; the rule is then frozen and evaluated per level.

    Pass ``freeze_tau``/``freeze_d`` to skip tuning and evaluate a rule
    frozen elsewhere — the transfer experiment: tune on one sweep (e.g.
    injection), evaluate frozen on another (e.g. chunking, another dataset).
    """
    methods = [m for m in df["Method"].unique() if m != "kNN"]
    levels = sorted(df[level_col].unique())

    # Wide tables: one row per (level, qid).
    obj = df.pivot_table(index=[level_col, "qid"], columns="Method",
                         values=objective)
    trig = (
        df[df.Method == "kNN"]
        .set_index([level_col, "qid"])[trigger_metric]
        .rename("trigger")
    )
    wide = obj.join(trig)

    is_val = wide.index.get_level_values("qid").isin(val)
    wval, wtest = wide[is_val], wide[~is_val]

    # Threshold grid: distinct trigger values on validation (plus extremes).
    qs = np.unique(np.round(wval["trigger"].to_numpy(), 2))
    taus = np.concatenate(([qs[0] - 0.01], qs, [qs[-1] + 0.01]))

    def rule_scores(w: pd.DataFrame, tau: float, d: str) -> pd.Series:
        fire = w["trigger"] < tau
        return w[d].where(fire, w["kNN"])

    # Joint tuning on pooled validation (skipped when a frozen rule is given).
    best = {"tau": None, "D": None, "val": -np.inf}
    curve_rows = []
    for d in methods:
        for tau in taus:
            v = float(rule_scores(wval, tau, d).mean())
            curve_rows.append({"D": d, "tau": round(float(tau), 2),
                               "val mean": round(v, 4)})
            if v > best["val"]:
                best = {"tau": float(tau), "D": d, "val": v}

    if freeze_tau is not None and freeze_d is not None:
        if freeze_d not in methods:
            raise SystemExit(f"--freeze_d '{freeze_d}' not in {methods}")
        tau_s, d_s = float(freeze_tau), freeze_d
    else:
        tau_s, d_s = best["tau"], best["D"]

    # Frozen-rule evaluation.
    rows = []
    pooled = {
        "kNN": wtest["kNN"], f"always {d_s}": wtest[d_s],
        f"rule (tau={tau_s:g}, {d_s})": rule_scores(wtest, tau_s, d_s),
        "all-methods oracle": wtest[methods + ["kNN"]].max(axis=1),
    }
    for level in levels:
        wl_val = wval.xs(level, level=level_col)
        wl = wtest.xs(level, level=level_col)
        level_best = wl_val[methods + ["kNN"]].mean().idxmax()
        sel = {
            "kNN": wl["kNN"],
            f"always {d_s}": wl[d_s],
            f"rule (tau={tau_s:g}, {d_s})":
                wl[d_s].where(wl["trigger"] < tau_s, wl["kNN"]),
            "per-level tuned": wl[level_best],
            "all-methods oracle": wl[methods + ["kNN"]].max(axis=1),
        }
        for name, series in sel.items():
            rows.append({
                level_col: level, "Selector": name,
                objective: round(float(series.mean()), 4),
                "trigger rate":
                    round(float((wl["trigger"] < tau_s).mean()), 3)
                    if name.startswith("rule") else "",
                "note": f"[{level_best}]" if name == "per-level tuned" else "",
            })
    for name, series in pooled.items():
        rows.append({level_col: "pooled", "Selector": name,
                     objective: round(float(series.mean()), 4),
                     "trigger rate": "", "note": ""})

    knn_p = float(pooled["kNN"].mean())
    rule_p = float(pooled[f"rule (tau={tau_s:g}, {d_s})"].mean())
    oracle_p = float(pooled["all-methods oracle"].mean())
    info = {
        "tau": tau_s, "D": d_s,
        "trigger_metric": trigger_metric,
        "tau_normalized": round(tau_s / top_k, 4),
        "pooled kNN": round(knn_p, 4),
        "pooled rule": round(rule_p, 4),
        "pooled oracle": round(oracle_p, 4),
        "headroom captured (%)":
            round(100 * (rule_p - knn_p) / max(oracle_p - knn_p, 1e-12), 1),
    }
    return pd.DataFrame(rows), pd.DataFrame(curve_rows), info


# ---------------------------------------------------------------------------
# Analysis 4b — relevant-set-size gate on the decision rule
# ---------------------------------------------------------------------------

def gate_evaluation(
    df: pd.DataFrame, level_col: str, objective: str,
    test: set, tau: float, d: str, min_rel: int = 2,
    trigger_metric: str = "Vendi",
) -> Optional[Tuple[pd.DataFrame, Dict[str, object]]]:
    """Evaluate the relevant-set-size gate on the tuned decision rule.

    The rule diversifies iff ``trigger(kNN top-k) < tau``.  The *gate* adds a
    second condition: fire only when the query has at least ``min_rel``
    distinct relevant subtopics, so the rule never diversifies a single-hop
    query (one relevant item), where reranking for coverage can only displace
    the answer.  This is exactly the mis-fire the rule shows on single-hop
    synthetic injection; the gate is meant to remove it without giving back
    the multi-hop gains.

    For each level and pooled, on the test split, reports kNN, always-D, the
    ungated rule, and the gated rule, with the trigger/gate firing rates and
    the recovery of the gate over the ungated rule and over kNN.  Returns
    ``None`` if the per-query file predates RelSetSize logging or if the tuned
    fallback ``d`` is absent.

    The gate uses the gold relevant-set size, so this validates the gate's
    *design*; a deployment would estimate the hop count from the query (a
    cheap query-side classifier), not from labels.
    """
    if "RelSetSize" not in df.columns:
        return None
    if d not in set(df["Method"].unique()):
        return None

    levels = sorted(df[level_col].unique())
    obj = df.pivot_table(index=[level_col, "qid"], columns="Method",
                         values=objective)
    aux = (df[df.Method == "kNN"]
           .set_index([level_col, "qid"])[[trigger_metric, "RelSetSize"]]
           .rename(columns={trigger_metric: "trigger", "RelSetSize": "relsize"}))
    wide = obj.join(aux)
    wtest = wide[wide.index.get_level_values("qid").isin(test)]

    def _scores(w: pd.DataFrame, gated: bool) -> pd.Series:
        fire = w["trigger"] < tau
        if gated:
            fire = fire & (w["relsize"] >= min_rel)
        return w[d].where(fire, w["kNN"])

    def _row(level_label, w: pd.DataFrame) -> Dict[str, object]:
        fire = w["trigger"] < tau
        gate_on = w["relsize"] >= min_rel
        ungated, gated = _scores(w, False), _scores(w, True)
        knn = w["kNN"]
        return {
            level_col: level_label, "D": d, "tau": round(float(tau), 4),
            "min_rel": min_rel, "n_test": int(len(w)),
            "kNN": round(float(knn.mean()), 4),
            "always_D": round(float(w[d].mean()), 4),
            "ungated_rule": round(float(ungated.mean()), 4),
            "gated_rule": round(float(gated.mean()), 4),
            "trigger_rate": round(float(fire.mean()), 3),
            "gate_off_rate": round(float((~gate_on).mean()), 3),
            "rule_fires_gated": round(float((fire & gate_on).mean()), 3),
            "mean_RelSetSize": round(float(w["relsize"].mean()), 2),
            "ungated_minus_kNN": round(float((ungated - knn).mean()), 4),
            "gated_minus_kNN": round(float((gated - knn).mean()), 4),
            "gated_minus_ungated": round(float((gated - ungated).mean()), 4),
        }

    rows = [_row(level, wtest.xs(level, level=level_col)) for level in levels]
    rows.append(_row("pooled", wtest))

    ungated_p = float(_scores(wtest, False).mean())
    gated_p = float(_scores(wtest, True).mean())
    knn_p = float(wtest["kNN"].mean())
    info = {
        "min_rel": min_rel, "tau": round(float(tau), 4), "D": d,
        "pooled kNN": round(knn_p, 4),
        "pooled ungated rule": round(ungated_p, 4),
        "pooled gated rule": round(gated_p, 4),
        "pooled gate-off rate":
            round(float((wtest["relsize"] < min_rel).mean()), 3),
        "ungated - kNN": round(ungated_p - knn_p, 4),
        "gated - kNN": round(gated_p - knn_p, 4),
        "gate recovers (gated - ungated)": round(gated_p - ungated_p, 4),
    }
    return pd.DataFrame(rows), info


# ---------------------------------------------------------------------------
# Analysis 5 — answer quality (generation) under redundancy
# ---------------------------------------------------------------------------

def generation_summary(
    df: pd.DataFrame, gen_df: pd.DataFrame, level_col: str,
    test: set, tau: float, d: str, trigger_metric: str = "Vendi",
) -> pd.DataFrame:
    """Per-level mean EM/F1 for kNN, the fallback diversifier ``d``, and the
    reconstructed decision rule (rule = d-answer when the kNN trigger fires,
    else kNN-answer), on the test split, with Wilcoxon markers vs kNN.

    The trigger is the per-query ``trigger_metric`` of the kNN selection,
    taken from the retrieval per-query table ``df`` (joined on level+qid);
    the generation EM/F1 come from ``gen_df``.  This closes the loop from
    S-Recall to answer quality and validates the rule on answers without any
    extra generation (the rule reuses the kNN and d answers already produced).
    """
    gen_methods = set(gen_df["Method"].unique())
    if "kNN" not in gen_methods:
        raise SystemExit(
            f"Generation file lacks kNN; available: {sorted(gen_methods)}.")
    if d not in gen_methods:
        # The rule's tuned fallback was not generated for; substitute a present
        # diversifier (prefer an MMR variant) so the rule can still be
        # reconstructed, and report the substitution.
        alt = sorted(m for m in gen_methods if m != "kNN")
        if not alt:
            raise SystemExit(
                f"Generation file has only kNN; cannot reconstruct the rule. "
                f"Re-run with --gen_methods kNN and the fallback diversifier.")
        d_sub = next((m for m in alt if m.startswith("MMR")), alt[0])
        print(f"   [generation] tuned fallback '{d}' not generated for; "
              f"reconstructing rule with present diversifier '{d_sub}'.")
        d = d_sub

    trig = (df[df.Method == "kNN"]
            .set_index([level_col, "qid"])[trigger_metric].rename("trigger"))
    rows = []
    for met in ("EM", "F1"):
        piv = gen_df.pivot_table(index=[level_col, "qid"], columns="Method",
                                 values=met)
        piv = piv.join(trig, how="left")
        fire = piv["trigger"] < tau
        piv["rule"] = piv[d].where(fire, piv["kNN"])
        is_test = piv.index.get_level_values("qid").isin(test)
        pt = piv[is_test]
        for level, g in pt.groupby(level_col):
            row = {level_col: level, "metric": met, "n": len(g),
                   "kNN": round(float(g["kNN"].mean()), 4),
                   d: round(float(g[d].mean()), 4),
                   "rule": round(float(g["rule"].mean()), 4),
                   "trigger rate": round(float((g["trigger"] < tau).mean()), 3)}
            for col in (d, "rule"):
                if _wilcoxon is not None:
                    diff = g[col].to_numpy() - g["kNN"].to_numpy()
                    row[f"p({col} vs kNN)"] = (
                        round(float(_wilcoxon(diff).pvalue), 5)
                        if np.any(diff != 0.0) else 1.0)
            # carry any extra generated methods (e.g. RNG) for the table
            for extra in sorted(gen_methods - {"kNN", d}):
                if extra in g.columns:
                    row[extra] = round(float(g[extra].mean()), 4)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--per_query", required=True,
                   help="results_*_per_query.csv from evaluate_redundancy.py")
    p.add_argument("--level_col", default="rho",
                   help="Sweep column: 'rho' (injection) or 'overlap' (chunking).")
    p.add_argument("--objective", default="S-Recall@k")
    p.add_argument("--trigger_metric", default="Vendi",
                   choices=["Vendi", "APD"],
                   help="Label-free statistic of the kNN selection used as "
                        "the decision-rule trigger.")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--seed", type=int, default=None,
                   help="Override the split seed (default: from run_params.json).")
    p.add_argument("--val_fraction", type=float, default=None)
    p.add_argument("--freeze_tau", type=float, default=None,
                   help="Evaluate the decision rule with this frozen threshold "
                        "instead of tuning (transfer experiment).")
    p.add_argument("--freeze_d", default=None,
                   help="Frozen fallback diversifier, e.g. 'MMR(0.7)'.")
    p.add_argument("--gate_min_rel", type=int, default=2,
                   help="Relevant-set-size gate: the decision rule may fire "
                        "only on queries with at least this many distinct "
                        "relevant subtopics (default 2, i.e. gate off "
                        "single-hop queries). Needs RelSetSize in the "
                        "per-query file.")
    p.add_argument("--out_prefix", default="analysis")
    args = p.parse_args()

    run_dir = os.path.dirname(os.path.abspath(args.per_query))
    params_path = os.path.join(run_dir, "run_params.json")
    seed, val_fraction = args.seed, args.val_fraction
    if os.path.exists(params_path):
        with open(params_path, encoding="utf-8") as f:
            params = json.load(f)
        seed = params["seed"] if seed is None else seed
        val_fraction = (params["val_fraction"]
                        if val_fraction is None else val_fraction)
    if seed is None or val_fraction is None:
        raise SystemExit("No run_params.json found; pass --seed and --val_fraction.")

    print(f"Loading {args.per_query} …")
    df = load_per_query(args.per_query, args.level_col)

    # Seed handling: a multi-seed run tags every per-query row with a seed.
    # Run each analysis once per seed on its own validation/test split, then
    # aggregate with a 95% across-seed CI. An explicit
    # --seed forces single-seed mode on that seed.
    if args.seed is not None:
        seeds_list = [int(args.seed)]
    else:
        seeds_list = seeds_in(df, seed)
    metrics = [c for c in df.columns
               if c not in (args.level_col, "qid", "Method", "PoolRedundancy",
                            "RelSetSize", "seed")]
    out = lambda tag: os.path.join(run_dir, f"{args.out_prefix}_{tag}.csv")
    red_map = level_redundancy(df, args.level_col, run_dir)

    gen_path = args.per_query.replace("_per_query.csv", "_gen_per_query.csv")
    gen_df_all = pd.read_csv(gen_path) if os.path.exists(gen_path) else None

    def analyze_one(seed_s: int) -> Dict[str, object]:
        df_s = df[df["seed"] == seed_s] if "seed" in df.columns else df
        qids = example_order(df_s, args.level_col)
        val, test = val_test_qids(qids, seed_s, val_fraction)
        res: Dict[str, object] = {
            "summary": tuned_summary(df_s, args.level_col, args.objective,
                                     val, test, metrics, red_map=red_map),
            "regret": regret_table(df_s, args.level_col, args.objective, test),
            "oracle": oracle_table(df_s, args.level_col, args.objective,
                                   val, test),
            "n_val": len(val), "n_test": len(test), "n": len(qids),
        }
        rule, curve, info = threshold_rule(
            df_s, args.level_col, args.objective, val, test,
            trigger_metric=args.trigger_metric, top_k=args.top_k,
            freeze_tau=args.freeze_tau, freeze_d=args.freeze_d,
        )
        res.update(rule=rule, curve=curve, info=info)
        res["crossover"] = crossover_estimate(res["summary"], args.level_col,
                                               args.objective)
        gate = gate_evaluation(
            df_s, args.level_col, args.objective, test, info["tau"], info["D"],
            min_rel=args.gate_min_rel, trigger_metric=args.trigger_metric)
        if gate is not None:
            res["gate"], res["gate_info"] = gate
        if gen_df_all is not None:
            gen_s = (gen_df_all[gen_df_all["seed"] == seed_s]
                     if "seed" in gen_df_all.columns else gen_df_all)
            res["gen"] = generation_summary(
                df_s, gen_s, args.level_col, test, info["tau"], info["D"],
                trigger_metric=args.trigger_metric)
        return res

    print(f"Seeds: {seeds_list} (val_fraction={val_fraction}).")
    runs = [analyze_one(s) for s in seeds_list]
    multi = len(seeds_list) > 1

    # --- Tuned summary -----------------------------------------------------
    if multi:
        summary = aggregate_frames([r["summary"] for r in runs],
                                   [args.level_col, "Method"])
    else:
        summary = runs[0]["summary"]
    summary.to_csv(out("summary"), index=False)
    if multi:
        pd.concat([r["summary"].assign(seed=s)
                   for s, r in zip(seeds_list, runs)], ignore_index=True
                  ).to_csv(out("summary_per_seed"), index=False)
    ci_note = f"; mean ± 95% CI over {len(seeds_list)} seeds" if multi else ""
    print(f"\n── Tuned summary (objective {args.objective}{ci_note}) ──")
    print(summary.to_string(index=False))

    # --- Mis-specification regret -----------------------------------------
    regret = (aggregate_frames([r["regret"] for r in runs],
                               [args.level_col, "Family"])
              if multi else runs[0]["regret"])
    regret.to_csv(out("regret"), index=False)
    print("\n── Mis-specification regret ──")
    print(regret.to_string(index=False))

    # --- Oracle headroom ---------------------------------------------------
    oracle = (aggregate_frames([r["oracle"] for r in runs],
                              [args.level_col, "Selector"])
              if multi else runs[0]["oracle"])
    oracle.to_csv(out("oracle"), index=False)
    print("\n── Oracle headroom ──")
    print(oracle.to_string(index=False))

    # --- Decision rule -----------------------------------------------------
    rule = (aggregate_frames([r["rule"] for r in runs],
                            [args.level_col, "Selector"])
            if multi else runs[0]["rule"])
    curve = (aggregate_frames([r["curve"] for r in runs], ["D", "tau"])
             if multi else runs[0]["curve"])
    rule.to_csv(out("threshold"), index=False)
    curve.to_csv(out("threshold_curve"), index=False)
    info = _aggregate_info([r["info"] for r in runs]) if multi else runs[0]["info"]
    print("\n── Redundancy-threshold decision rule ──")
    print(rule.to_string(index=False))
    print("\nSelected rule:", json.dumps(info, indent=2))
    with open(out("threshold_info").replace(".csv", ".json"), "w",
              encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    # --- Crossover location (with across-seed CI) -------------------------
    xs = [float(r["crossover"]) for r in runs]
    x_mean, x_half, _ = seed_interval([x for x in xs if x == x])  # drop NaN
    n_cross = sum(1 for x in xs if x == x)
    cross_info = {
        "objective": args.objective,
        "per_seed": [None if x != x else round(x, 5) for x in xs],
        "mean": None if n_cross == 0 else round(x_mean, 5),
        "ci95_half_width": None if n_cross < 2 else round(x_half, 5),
        "n_seeds_with_crossover": n_cross,
        "note": ("diversification never overtakes kNN in the swept range for "
                 "some seed(s)" if n_cross < len(xs) else ""),
    }
    pd.DataFrame([cross_info]).to_csv(out("crossover"), index=False)
    print("\n── Crossover location (rho where diversification overtakes kNN) ──")
    print(json.dumps(cross_info, indent=2))

    # --- Relevant-set-size gate (if RelSetSize was logged) -----------------
    if all("gate" in r for r in runs):
        gate = (aggregate_frames([r["gate"] for r in runs], [args.level_col])
                if multi else runs[0]["gate"])
        gate.to_csv(out("gate"), index=False)
        if multi:
            pd.concat([r["gate"].assign(seed=s)
                       for s, r in zip(seeds_list, runs)], ignore_index=True
                      ).to_csv(out("gate_per_seed"), index=False)
        gate_info = (_aggregate_info([r["gate_info"] for r in runs])
                     if multi else runs[0]["gate_info"])
        with open(out("gate_info").replace(".csv", ".json"), "w",
                  encoding="utf-8") as f:
            json.dump(gate_info, f, indent=2)
        print(f"\n── Relevant-set-size gate (min_rel={args.gate_min_rel}; "
              f"rule fires only on multi-hop queries) ──")
        print(gate.to_string(index=False))
        print("\nGate summary:", json.dumps(gate_info, indent=2))
    elif args.seed is None and "RelSetSize" not in df.columns:
        print("\n[gate] RelSetSize not in per-query file; skipping gate "
              "evaluation (re-run evaluate_redundancy.py to log it).")

    # --- Answer quality (generation), if the sweep produced it -------------
    if gen_df_all is not None:
        gen = (aggregate_frames([r["gen"] for r in runs],
                               [args.level_col, "metric"])
               if multi else runs[0]["gen"])
        gen.to_csv(out("generation"), index=False)
        print("\n── Answer quality under redundancy (EM/F1; rule reconstructed) ──")
        print(gen.to_string(index=False))


if __name__ == "__main__":
    main()
