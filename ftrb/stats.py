"""Confidence intervals and paired tests used by evaluation scripts."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # scipy is a hard dependency of the suite, but degrade gracefully.
    from scipy.stats import t as _t_dist
    from scipy.stats import wilcoxon as _wilcoxon
except ImportError:  # pragma: no cover
    _t_dist = None
    _wilcoxon = None


# Standard repeated-run seeds used by the experiment configurations.
DEFAULT_SEEDS: List[int] = [0, 1, 2]
N_BOOTSTRAP: int = 1000
CONFIDENCE: float = 0.95


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def add_seed_arg(parser, default: Optional[Sequence[int]] = None) -> None:
    """Register the standard ``--seeds`` option on an argparse parser.

    Kept identical across scripts so a single command shape drives every
    experiment.  Scripts retain their own ``--seed`` for the single-run
    default; ``--seeds`` overrides it when given.
    """
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=default,
        help="One or more integer seeds. The whole experiment is repeated "
             "once per seed and the tables are aggregated to a mean with a "
             "95%% confidence interval across seeds (Student-t). Defaults to "
             "a single run on --seed. Use '--seeds 0 1 2' for the 3-seed "
             "protocol.",
    )


def resolve_seeds(args, fallback_attr: str = "seed",
                  default_single: int = 0) -> List[int]:
    """Return the de-duplicated seed list for a run.

    ``--seeds`` wins when provided; otherwise fall back to the script's
    single ``--seed`` (or ``default_single`` if absent), so existing
    single-run invocations behave exactly as before.
    """
    seeds = getattr(args, "seeds", None)
    if seeds:
        return list(dict.fromkeys(int(s) for s in seeds))
    one = getattr(args, fallback_attr, None)
    return [int(one) if one is not None else default_single]


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def t_critical(n: int, confidence: float = CONFIDENCE) -> float:
    """Two-sided Student-t critical value for ``n`` observations (n-1 df)."""
    if n < 2:
        return 0.0
    if _t_dist is not None:
        return float(_t_dist.ppf(0.5 + confidence / 2.0, n - 1))
    return 1.96  # large-sample normal fallback when scipy is unavailable


def seed_interval(values: Sequence[float],
                  confidence: float = CONFIDENCE) -> Tuple[float, float, float]:
    """``(mean, half_width, std)`` of a Student-t CI over per-seed estimates.

    ``half_width`` is the +/- term, so the interval is ``mean +/- half_width``.
    A single seed yields a zero-width interval (the honest statement that no
    seed variance was measured); the bootstrap CI then carries the
    uncertainty.
    """
    a = np.asarray(list(values), dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(a.mean())
    if n < 2:
        return mean, 0.0, 0.0
    sd = float(a.std(ddof=1))
    half = t_critical(n, confidence) * sd / np.sqrt(n)
    return mean, half, sd


def bootstrap_interval(values: Sequence[float], confidence: float = CONFIDENCE,
                       n_resamples: int = N_BOOTSTRAP,
                       seed: int = 0) -> Tuple[float, float, float]:
    """``(mean, lo, hi)`` percentile bootstrap CI over a sample (e.g. queries)."""
    a = np.asarray(list(values), dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1:
        return float(a[0]), float(a[0]), float(a[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = a[idx].mean(axis=1)
    lo = float(np.percentile(means, (1.0 - confidence) / 2.0 * 100.0))
    hi = float(np.percentile(means, (1.0 + confidence) / 2.0 * 100.0))
    return float(a.mean()), lo, hi


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------

def paired_wilcoxon_p(method_vals: Sequence[float],
                      baseline_vals: Sequence[float]) -> Optional[float]:
    """Two-sided Wilcoxon signed-rank p-value of *method* vs *baseline*.

    Returns ``1.0`` when every paired difference is zero (no detectable
    effect) and ``None`` when scipy is unavailable or the inputs are
    unusable, so callers can render an explicit blank.
    """
    if _wilcoxon is None:
        return None
    a = np.asarray(list(method_vals), dtype=float)
    b = np.asarray(list(baseline_vals), dtype=float)
    if len(a) == 0 or len(a) != len(b):
        return None
    diffs = a - b
    if not np.any(diffs != 0.0):
        return 1.0
    return float(_wilcoxon(a, b).pvalue)


def holm_bonferroni(pvalues: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Holm step-down adjusted p-values, preserving input order.

    ``None`` entries (missing tests) are passed through unchanged and do not
    count toward the family size.
    """
    indexed = [(p, i) for i, p in enumerate(pvalues) if p is not None]
    m = len(indexed)
    adjusted: List[Optional[float]] = [None] * len(pvalues)
    running = 0.0
    for rank, (p, i) in enumerate(sorted(indexed, key=lambda t: t[0])):
        running = min(1.0, max(running, (m - rank) * p))
        adjusted[i] = round(running, 8)
    return adjusted


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def as_float(x) -> Optional[float]:
    """Best-effort float parse; ``None`` for blanks and non-numeric cells."""
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def aggregate_seed_rows(
    rows: Sequence[Dict],
    key_cols: Sequence[str],
    metric_cols: Sequence[str],
    seed_col: str = "seed",
    confidence: float = CONFIDENCE,
    ci_suffix: str = "_ci95",
    keep_cols: Sequence[str] = (),
) -> List[Dict]:
    """Collapse per-seed rows to one row per key, with across-seed mean and CI.

    For each group (defined by ``key_cols``) and each metric in
    ``metric_cols`` the output row carries ``<metric>`` (across-seed mean)
    and ``<metric><ci_suffix>`` (the 95% half-width).  ``n_seeds`` records how
    many seeds contributed.  ``keep_cols`` are carried through by first value
    (use for static descriptors that do not vary across seeds).

    Rows whose metric is absent contribute nothing to that metric, so tables
    that mix row types with different column sets (method rows, headroom
    rows) aggregate cleanly.
    """
    groups: "OrderedDict[tuple, List[Dict]]" = OrderedDict()
    for r in rows:
        key = tuple(r.get(c) for c in key_cols)
        groups.setdefault(key, []).append(r)

    out: List[Dict] = []
    for key, grp in groups.items():
        row: Dict[str, object] = {c: v for c, v in zip(key_cols, key)}
        seeds = {g.get(seed_col) for g in grp if g.get(seed_col) is not None}
        row["n_seeds"] = len(seeds)
        for c in keep_cols:
            row[c] = grp[0].get(c, "")
        for metric in metric_cols:
            vals = [as_float(g.get(metric)) for g in grp]
            vals = [v for v in vals if v is not None]
            if not vals:
                row[metric] = ""
                row[metric + ci_suffix] = ""
                continue
            mean, half, _ = seed_interval(vals, confidence)
            row[metric] = round(mean, 4)
            row[metric + ci_suffix] = round(half, 4)
        out.append(row)
    return out


def significance_table(
    acc: Dict[str, Dict[str, List[float]]],
    metric_order: Sequence[Tuple[str, str]],
    baseline: str = "kNN",
    bootstrap_seed: int = 0,
) -> List[Dict]:
    """Long-format mean / bootstrap-CI / Wilcoxon table from per-query lists.

    ``acc`` maps method -> metric_key -> per-query values (exactly the
    accumulators the evaluation scripts already build).  For each method and
    each ``(metric_key, label)`` the row carries the mean, a 95% across-query
    percentile-bootstrap interval, the sample size, and — for non-baseline
    methods evaluated on the same queries — the paired Wilcoxon p-value
    against ``baseline``.  This is the right uncertainty for a deterministic
    method on a finite query set, where re-seeding changes nothing.
    """
    rows: List[Dict] = []
    base = acc.get(baseline, {})
    for method, metrics in acc.items():
        for metric_key, label in metric_order:
            vals = metrics.get(metric_key, [])
            if not vals:
                continue
            mean, lo, hi = bootstrap_interval(vals, seed=bootstrap_seed)
            row: Dict[str, object] = {
                "Method": method, "Metric": label, "n": len(vals),
                "mean": round(mean, 4), "ci95_lo": round(lo, 4),
                "ci95_hi": round(hi, 4),
            }
            if method != baseline:
                bvals = base.get(metric_key, [])
                if bvals and len(bvals) == len(vals):
                    p = paired_wilcoxon_p(vals, bvals)
                    if p is not None:
                        row[f"p_vs_{baseline}"] = round(p, 6)
            rows.append(row)
    return rows


def fmt_mean_ci(mean: float, half: float, precision: int = 3) -> str:
    """Render ``mean +/- half`` as a compact string for printed tables."""
    if mean != mean:  # NaN
        return ""
    if half is None or half != half or half == 0.0:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} +/- {half:.{precision}f}"
