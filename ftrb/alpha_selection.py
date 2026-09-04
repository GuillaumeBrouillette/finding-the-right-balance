"""Select, persist, and retrieve validation-tuned reranking parameters."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np


OBJECTIVES = ["recall", "ndcg", "alpha_ndcg", "vendi", "recall_vendi", "apd", "em", "f1"]


# ---------------------------------------------------------------------------
# Alpha grid construction
# ---------------------------------------------------------------------------

def build_coarse_to_fine_alpha_grid(
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    fine_step: Optional[float] = None,
    dead_zone_min: float = -1.0,
) -> List[float]:
    """Build a fine-resolution alpha grid used by all evaluation scripts.

    Constructs a sorted grid over [alpha_min, alpha_max] at fine resolution,
    extending one coarse step beyond each boundary (left extension disabled
    when alpha_min <= dead_zone_min).

    Parameters
    ----------
    alpha_min, alpha_max : float
        Nominal range boundaries.
    alpha_step : float
        Coarse step; sets the boundary extension margin and, by default,
        derives fine_step = alpha_step / 5.
    fine_step : float, optional
        Override for the fine grid resolution.
    dead_zone_min : float, optional
        Left extension is skipped when alpha_min <= dead_zone_min.

    Returns
    -------
    list of float
        Sorted, deduplicated grid values rounded to 8 decimal places.
    """
    coarse = float(alpha_step)
    fine = float(fine_step) if fine_step is not None else coarse / 5.0
    if fine <= 0.0:
        raise ValueError("fine_step must be > 0")
    left = float(alpha_min - coarse) if float(alpha_min) > float(dead_zone_min) else float(alpha_min)
    right = float(alpha_max + coarse)
    grid = np.round(np.arange(left, right + fine / 2.0, fine), 8)
    return sorted(set(float(a) for a in grid.tolist()))

_HIGHER_IS_BETTER = {obj: True for obj in OBJECTIVES}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class AlphaResult:
    """Stores the outcome of a single alpha (or alpha+beta) sweep."""
    setting: str          # e.g. "hotpotqa/all-MiniLM-L6-v2"
    score_type: str       # "RNG-Score" | "Seg-Score" | "S1-Blend" | "S2-V1" | …
    objective: str        # which metric was maximised
    alpha_star: float     # best alpha found
    beta_star: Optional[float]  # best beta (S1-Blend only), else None
    val_score: float      # objective value at alpha_star on the val split
    sweep: Dict[str, float]  # key = "α=<val>" or "α=<a>,β=<b>"; value = objective
    generator_model: Optional[str] = None  # generator used for em/f1 objectives


# ---------------------------------------------------------------------------
# Val-accumulator helpers (used inside evaluate scripts)
# ---------------------------------------------------------------------------

def _key_alpha(alpha: float) -> str:
    return f"α={alpha:.4g}"


def _key_blend(alpha: float, beta: float) -> str:
    return f"α={alpha:.4g},β={beta:.4g}"


# ---------------------------------------------------------------------------
# Finding optimal hyper-parameters
# ---------------------------------------------------------------------------

def _warn_if_boundary(
    alpha_star: float,
    alpha_grid: List[float],
    score_type: str,
    objective: str,
) -> None:
    """Print a warning when alpha_star lands on the grid boundary.

    The fallback sentinel (alpha <= -2, which deactivates every penalty under
    cosine distance) is exempt: selecting it means "do not diversify", not
    that the optimum lies outside the grid.
    """
    tol = 1e-9
    if alpha_star <= -2.0 + tol:
        return
    if abs(alpha_star - alpha_grid[0]) < tol or abs(alpha_star - alpha_grid[-1]) < tol:
        print(
            f"   [WARNING] {score_type} [{objective}]: α*={alpha_star:.4g} is at the "
            f"grid boundary [{alpha_grid[0]:.4g}, {alpha_grid[-1]:.4g}]. "
            "The true optimum may lie outside — consider widening "
            "--alpha_min / --alpha_max."
        )


def find_optimal_alpha(
    val_acc: Dict[str, float],
    alpha_grid: List[float],
    score_type: str,
    objective: str,
    setting: str = "",
    generator_model: Optional[str] = None,
) -> AlphaResult:
    """Pick the best alpha from per-alpha validation averages.

    Parameters
    ----------
    val_acc : dict  {_key_alpha(a): mean_objective_on_val_split}
    alpha_grid : list of floats (same values used to build val_acc keys)
    score_type : "RNG-Score" | "Seg-Score" | "S2-V1" | "S2-V2" | "S2-V3"
    objective : one of OBJECTIVES
    setting : free-form label for the (dataset, encoder) pair

    Tie-breaking
    ------------
    When multiple alpha values achieve the same best score (e.g. when all are
    in the cosine-distance dead zone where every hinge term is zero), the alpha
    closest to 0 is preferred.  This keeps the model as conservative as possible
    while still selecting a valid value.
    """
    if not val_acc:
        raise ValueError("val_acc is empty – cannot select alpha")

    best_val = max(val_acc.values())
    # Among tied keys prefer the alpha closest to 0 (least bias from kNN).
    best_key = min(
        (k for k in val_acc if val_acc[k] == best_val),
        key=lambda k: abs(float(k.split("α=")[1].split(",")[0])),
    )

    # Recover alpha_star from key
    alpha_star = float(best_key.split("α=")[1].split(",")[0])

    _warn_if_boundary(alpha_star, alpha_grid, score_type, objective)

    return AlphaResult(
        setting=setting,
        score_type=score_type,
        objective=objective,
        alpha_star=alpha_star,
        beta_star=None,
        val_score=best_val,
        sweep=dict(val_acc),
        generator_model=generator_model,
    )


def find_optimal_blend(
    val_acc: Dict[str, float],
    alpha_grid: List[float],
    beta_grid: List[float],
    score_type: str,
    objective: str,
    setting: str = "",
    generator_model: Optional[str] = None,
) -> AlphaResult:
    """Pick best (alpha, beta) for S1-Blend from a 2-D val grid.

    Parameters
    ----------
    val_acc : dict  {_key_blend(a, b): mean_objective_on_val_split}
    """
    if not val_acc:
        raise ValueError("val_acc is empty – cannot select (alpha, beta)")

    best_val = max(val_acc.values())
    # Among ties prefer (α, β) closest to (0, 0) in L1 norm.
    best_key = min(
        (k for k in val_acc if val_acc[k] == best_val),
        key=lambda k: (
            abs(float(k.split(",")[0].split("α=")[1]))
            + abs(float(k.split(",")[1].split("β=")[1]))
        ),
    )

    parts = best_key.split(",")
    alpha_star = float(parts[0].split("α=")[1])
    beta_star = float(parts[1].split("β=")[1])

    _warn_if_boundary(alpha_star, alpha_grid, score_type, objective)

    return AlphaResult(
        setting=setting,
        score_type=score_type,
        objective=objective,
        alpha_star=alpha_star,
        beta_star=beta_star,
        val_score=best_val,
        sweep=dict(val_acc),
        generator_model=generator_model,
    )


# ---------------------------------------------------------------------------
# Accumulator helpers (call per query in the val loop)
# ---------------------------------------------------------------------------

def accumulate_alpha(
    accumulators: Dict[str, List[float]],
    alpha: float,
    value: float,
) -> None:
    """Append `value` to the running list for key _key_alpha(alpha)."""
    k = _key_alpha(alpha)
    accumulators.setdefault(k, []).append(value)


def accumulate_blend(
    accumulators: Dict[str, List[float]],
    alpha: float,
    beta: float,
    value: float,
) -> None:
    """Append `value` to the running list for key _key_blend(alpha, beta)."""
    k = _key_blend(alpha, beta)
    accumulators.setdefault(k, []).append(value)


def average_accumulator(accumulators: Dict[str, List[float]]) -> Dict[str, float]:
    """Return mean of each accumulator list."""
    import numpy as np
    return {k: float(np.mean(v)) for k, v in accumulators.items() if v}


# ---------------------------------------------------------------------------
# Objective extraction from a per-method metrics dict
# ---------------------------------------------------------------------------

def get_objective_value(
    metrics: Dict[str, float],
    objective: str,
    vendi_max: float = 1.0,
) -> float:
    """Extract the scalar objective from a metrics dict.

    `metrics` should contain keys like 'recall', 'ndcg', 'alpha_ndcg',
    'vendi', 'em' (same as those produced by the evaluate scripts).

    For 'recall_vendi' the combined score is  0.5*recall + 0.5*(vendi/vendi_max).
    vendi_max should be the maximum vendi across all methods on the current
    query (pass the kNN value, or 1.0 as a safe default).
    """
    if objective == "recall_vendi":
        r = metrics.get("recall", 0.0)
        v = metrics.get("vendi", 0.0) / max(vendi_max, 1e-9)
        return 0.5 * r + 0.5 * v
    return metrics.get(objective, 0.0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_alphas(results: List[AlphaResult], path: str) -> None:
    """Append (or create) `path` with the given results.

    Existing entries with the same
    (setting, score_type, objective, generator_model) key
    are overwritten; all others are preserved.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    existing: List[Dict] = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    key_fn = lambda d: (
        d["setting"],
        d["score_type"],
        d["objective"],
        d.get("generator_model"),
    )
    new_keys = {
        (r.setting, r.score_type, r.objective, r.generator_model)
        for r in results
    }
    kept = [e for e in existing if key_fn(e) not in new_keys]
    merged = kept + [asdict(r) for r in results]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"   Saved alpha results → {path}")


def load_alphas(path: str) -> List[AlphaResult]:
    """Load all AlphaResult objects from a JSON file."""
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [AlphaResult(**r) for r in raw]


def lookup_alpha(
    results: List[AlphaResult],
    score_type: str,
    setting: str = "",
    objective: str = "apd",
    generator_model: Optional[str] = None,
) -> Optional[AlphaResult]:
    """Find the best-matching AlphaResult.

    Matches on score_type + objective first; setting and generator_model are
    used to narrow when multiple entries share the same
    (score_type, objective).

    If a generator_model is requested but no exact match exists, this falls
    back to an untagged (legacy) entry for backward compatibility.
    Returns None if no entry is found.
    """
    candidates = [
        r for r in results
        if r.score_type == score_type and r.objective == objective
    ]
    if not candidates:
        return None
    if setting:
        exact = [r for r in candidates if r.setting == setting]
        if exact:
            candidates = exact

    if generator_model is not None:
        exact_gen = [r for r in candidates if r.generator_model == generator_model]
        if exact_gen:
            return exact_gen[0]
        legacy = [r for r in candidates if r.generator_model in (None, "")]
        if legacy:
            return legacy[0]

    return candidates[0]


# ---------------------------------------------------------------------------
# Dead-zone diagnostics
# ---------------------------------------------------------------------------

def print_sweep_diagnostics(result: AlphaResult, tol: float = 1e-6) -> None:
    """Print the full per-alpha sweep and warn if many values are tied.

    A "dead zone" warning is emitted when ≥50 % of sweep entries share the
    same score.  This typically means the alpha range extends into the region
    where all hinge terms are inactive (equivalent to plain kNN).
    """
    sweep = result.sweep
    if not sweep:
        return

    sorted_items = sorted(sweep.items(),
                          key=lambda kv: float(kv[0].split("α=")[1].split(",")[0]))

    try:
        from tabulate import tabulate
        rows = [[k, f"{v:.6f}"] for k, v in sorted_items]
        print(tabulate(rows, headers=["params", result.objective], tablefmt="simple"))
    except ImportError:
        for k, v in sorted_items:
            print(f"  {k}: {v:.6f}")

    # Dead-zone check: how many entries share the most common score?
    from collections import Counter
    rounded = [round(v / tol) * tol for v in sweep.values()]
    most_common_count = Counter(rounded).most_common(1)[0][1]
    fraction_tied = most_common_count / len(sweep)
    if fraction_tied >= 0.5:
        print(
            f"   WARNING ({result.score_type}): {most_common_count}/{len(sweep)} alpha values "
            f"({fraction_tied:.0%}) produce the same {result.objective} score.\n"
            f"   This often means the sweep range extends into the kNN-equivalent dead zone\n"
            f"   where all RNG hinge terms are zero.  Consider narrowing the alpha range\n"
            f"   to [0, +0.5] for cosine distance, or widening top_m."
        )


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_alpha_table(results: List[AlphaResult]) -> None:
    """Print a compact summary of all stored results."""
    try:
        from tabulate import tabulate
        rows = [
            [r.setting, r.score_type, r.objective,
             f"α={r.alpha_star:.4g}" + (f", β={r.beta_star:.4g}" if r.beta_star is not None else ""),
             f"{r.val_score:.4f}"]
            for r in results
        ]
        print(tabulate(rows,
                       headers=["Setting", "Score type", "Objective", "Best params", "Val score"],
                       tablefmt="github"))
    except ImportError:
        for r in results:
            print(r)
