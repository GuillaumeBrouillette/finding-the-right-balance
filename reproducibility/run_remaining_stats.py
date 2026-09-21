#!/usr/bin/env python3
"""Run every remaining CPU-side statistical computation for the revision.

One launcher for the revision analyses that complement
``create_statistical_archive.py``. Everything runs from retained per-query
result files; no encoder, reranker, generator, or GPU is invoked. Run it from
the repository root:

    python reproducibility/run_remaining_stats.py

Two source modes, detected automatically:

  * corrected mode — ``results/reproducibility_reruns/`` is present (the 13
    fixed-pool re-executions from the reproducibility release): every step
    runs, including the table refresh.
  * historical mode — selected explicitly, or used when no reruns are present:
    the script runs on the original run directories under ``results/retained/``
    (the same runs the paper's tables
    display), which is the consistent source for the rule-transfer, gate,
    generation, oracle and regret quantities; only the table refresh (which
    exists precisely to replace historical values by corrected ones) is
    skipped, and can be run later with ``--download-reruns`` or on a machine
    that has the release archive.

Steps (each writes one folder under ``results/revision_stats/``):

  table_refresh        (A) test-split level means of the paper's method
                           families recomputed from the fixed-pool
                           re-executions, to refresh the displayed sweep
                           tables.
  rule_transfer        (B) frozen decision rule (tau = min(h, k),
                           D* = MMR(0.7)) versus kNN with query-clustered
                           bootstrap intervals, Holm-adjusted, per level and
                           pooled, for the retained sweeps.
  gate_budgeted        (F) the BEIR subset of (B) under the budgeted
                           threshold, with per-level trigger rates.
  generation_inference (C) paired EM/F1 differences versus kNN from the two
                           retained per-query generation files (EM is a
                           paired risk difference; binary outcome, so no
                           Wilcoxon).
  oracle_intervals     (D) bootstrap intervals for the all-methods oracle
                           headroom, the frozen rule's capture ratios and
                           each family's misspecification downside on the
                           headline HotpotQA injection sweep.
  minimax_regret       (E) the common-oracle minimax-regret companion of the
                           misspecification-downside table (same per-regime
                           oracle for every row) on the headline sweep.

Inputs required:

  * ``manifests/splits/*.csv`` at the repository root (the frozen
    validation/test manifests);
  * per-query result files under results/ (either mode above).

Reuses the statistical conventions of
``reproducibility/create_statistical_archive.py`` (query-cluster bootstrap
with one weight vector per run, centered bootstrap p-values with the
(extreme+1)/(B+1) correction, Holm step-down within declared families).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"
RUN_ROOT = RESULTS_ROOT / "reproducibility_reruns"
HISTORICAL_ROOT = RESULTS_ROOT / "retained"
SPLIT_ROOT = ROOT / "manifests" / "splits"
OUT_ROOT = RESULTS_ROOT / "revision_stats"

BASELINE = "kNN"
RULE_FALLBACK = "MMR(0.7)"  # frozen D* (paper, Section RQ4)
PAPER_FAMILIES = ("MMR", "RNG")  # tuned families displayed in the paper
PAPER_FIXED = ("Maxmin", "Greedy-DPP")  # parameter-free methods displayed
ORACLE_EXCLUDED_PREFIXES = ("Seg(",)  # Seg-Score is excluded from analyses
# Headline HotpotQA injection sweep for the oracle/regret quantities, in
# order of preference (corrected rerun, then the paper's historical run).
HEADLINE_PREFERENCE = ("2026-08-07_230907", "2026-06-24_230733")
DEFAULT_BOOTSTRAP_SAMPLES = 1999
DEFAULT_BOOTSTRAP_SEED = 20260809
GENERATION_SOURCE_RUNS = (
    "2026-06-18_121117_redundancy_2wikimultihopqa",
    "2026-06-18_173800_redundancy_hotpotqa_fullwiki",
)

# Evidence requirement h per dataset (paper, Section "Rule and oracle").
# "relset" reads the per-query RelSetSize column; "hops" parses the MuSiQue
# qid prefix ("2hop__...").
EVIDENCE_REQUIREMENT = {
    "hotpotqa": 2,
    "hotpotqa_fullwiki": 2,
    "2wikimultihopqa": 2,
    "musique": "hops",
    "nq": 1,
    "scifact": "relset",
    "fiqa": "relset",
    "trec-covid": "relset",
}


# ---------------------------------------------------------------------------
# Shared machinery (conventions of create_statistical_archive.py)
# ---------------------------------------------------------------------------

def canonical_number(value: str) -> str:
    number = Decimal(value)
    if number == 0:
        return "0"
    return format(number.normalize(), "f")


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def bootstrap_columns(matrix: np.ndarray, samples: int, seed: int, block_size: int = 32) -> np.ndarray:
    """Ordinary query-cluster bootstrap means, sharing weights across columns."""
    n_queries, n_columns = matrix.shape
    clean = np.nan_to_num(matrix, nan=0.0)
    present = np.isfinite(matrix).astype(np.float64)
    probabilities = np.full(n_queries, 1.0 / n_queries)
    rng = np.random.default_rng(seed)
    output = np.empty((samples, n_columns), dtype=np.float64)
    for start in range(0, samples, block_size):
        stop = min(start + block_size, samples)
        weights = rng.multinomial(n_queries, probabilities, size=stop - start).astype(np.float64)
        numerators = weights @ clean
        denominators = weights @ present
        output[start:stop] = np.divide(
            numerators, denominators,
            out=np.full_like(numerators, np.nan), where=denominators != 0,
        )
    return output


def run_seed_for(name: str, global_seed: int) -> int:
    import hashlib

    return int(hashlib.sha256(f"{global_seed}:{name}".encode()).hexdigest()[:16], 16)


def centered_p(boots: np.ndarray, observed: float, samples: int, one_sided: bool = False) -> float:
    centered = boots - observed
    if one_sided:
        extreme = np.count_nonzero(centered >= observed)
    else:
        extreme = np.count_nonzero(np.abs(centered) >= abs(observed))
    return (1.0 + extreme) / (samples + 1.0)


def load_membership(dataset: str) -> Dict[Tuple[int, str], str]:
    path = SPLIT_ROOT / f"{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen split manifest: {path}")
    membership: Dict[Tuple[int, str], str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            membership[(int(row["seed"]), row["query_id"])] = row["partition"]
    return membership


def discover_runs(source_mode: str = "auto") -> Tuple[str, List[dict]]:
    """Return ``(mode, runs)`` from corrected or retained historical inputs."""
    corrected_available = any(RUN_ROOT.glob("*/run_params.json"))
    if source_mode == "corrected" and not corrected_available:
        raise FileNotFoundError(
            f"corrected mode requested but no run metadata was found under {RUN_ROOT}"
        )
    if source_mode == "corrected" or (source_mode == "auto" and corrected_available):
        mode, roots = "corrected", sorted(RUN_ROOT.glob("*/run_params.json"))
    else:
        retained = sorted(
            path for path in HISTORICAL_ROOT.glob("*/run_params.json")
            if path.parent.name < "2026-08-01"
        )
        roots = retained or sorted(
            p for p in RESULTS_ROOT.glob("*/run_params.json")
            if p.parent.name != "revision_stats"
        )
        mode = "historical"
    runs: List[dict] = []
    for params_path in roots:
        run_dir = params_path.parent
        with params_path.open(encoding="utf-8") as handle:
            params = json.load(handle)
        for experiment, stem in (("redundancy", "results_redundancy"), ("chunking", "results_chunking")):
            per_query = run_dir / f"{stem}_per_query.csv"
            if not per_query.exists():
                continue
            dataset = params.get("dataset")
            if dataset is None:
                continue
            if not (SPLIT_ROOT / f"{dataset}.csv").exists():
                print(f"[discover] {run_dir.name}__{experiment}: no frozen split manifest for dataset "
                      f"'{dataset}'; skipped")
                continue
            runs.append({
                "run": f"{run_dir.name}__{experiment}",
                "dataset": dataset,
                "experiment": experiment,
                "objective": "S-Recall@k",
                "top_k": int(params.get("top_k", 5)),
                "default_seed": int(params.get("seed", 0)),
                "condition_name": "rho" if experiment == "redundancy" else "overlap",
                "per_query": per_query,
            })
    return mode, runs


def family_of(method: str) -> Optional[str]:
    """Map a per-query Method label to the reported series name."""
    if method == BASELINE:
        return BASELINE
    if method in PAPER_FIXED:
        return method
    for prefix in PAPER_FAMILIES:
        if method.startswith(f"{prefix}("):
            return f"{prefix}*"
    return None  # Seg/Dedup/VendiG members are not displayed series


def is_oracle_member(method: str) -> bool:
    return not any(method.startswith(prefix) for prefix in ORACLE_EXCLUDED_PREFIXES)


def stream_groups(per_query: Path, condition_name: str, default_seed: int = 0):
    """Yield ((condition, seed, qid), {method: row-dict}) groups.

    Relies on the contiguous (condition, seed, qid) grouping of the sweep
    outputs, the same assumption made and validated by the archive builder.
    Single-run files without a seed column use the run's recorded seed.
    """
    with per_query.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or ()
        has_seed = "seed" in fields
        current_key = None
        group: Dict[str, dict] = {}
        for row in reader:
            seed_value = int(row["seed"]) if has_seed else default_seed
            key = (canonical_number(row[condition_name]), seed_value, row["qid"])
            if current_key is not None and key != current_key:
                yield current_key, group
                group = {}
            current_key = key
            group[row["Method"]] = row
        if current_key is not None:
            yield current_key, group


# ---------------------------------------------------------------------------
# Archive download
# ---------------------------------------------------------------------------

def download_reruns() -> None:
    """Fetch the fixed-pool re-executions from the reproducibility release."""
    tarball = "finding-the-right-balance-retained-results-2026-08-10.tar.zst"
    artifact_dir = ROOT / "artifacts"
    archive = artifact_dir / tarball
    url = (
        "https://github.com/GuillaumeBrouillette/finding-the-right-balance/"
        f"releases/download/reproducibility-v1/{tarball}"
    )
    print("Downloading the reproducibility-v1 retained-results archive ...")
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["curl", "-L", "-o", str(archive), url],
            cwd=ROOT, check=True,
        )
        subprocess.run(
            ["tar", "--use-compress-program=unzstd", "-xf", str(archive)],
            cwd=ROOT, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(
            "Automatic download failed ({}).\n"
            "Download the public release asset manually, then extract it at the\n"
            "repository root so that results/reproducibility_reruns appears:\n"
            "  tar --use-compress-program=unzstd -xf {}".format(error, archive)
        )
    if not any(RUN_ROOT.glob("*/run_params.json")):
        raise SystemExit("extraction finished but results/reproducibility_reruns is still missing")


# ---------------------------------------------------------------------------
# Step A: refreshed level means for the displayed sweep tables
# ---------------------------------------------------------------------------

def step_table_refresh(runs: List[dict], samples: int, seed: int) -> None:
    out_dir = OUT_ROOT / "table_refresh"
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        condition_name = run["condition_name"]
        objective = run["objective"]
        membership = load_membership(run["dataset"])

        # Pass 1: validation means per (condition, seed, method) for tuning.
        val_sums: Dict[Tuple[str, int, str], List[float]] = {}
        method_order: List[str] = []
        metric_fields: List[str] = []
        with run["per_query"].open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            metric_fields = [
                f for f in (reader.fieldnames or [])
                if f not in (condition_name, "seed", "qid", "Method") and not f.startswith("Pool")
                and f not in ("OriginalPoolSize", "TransformedPoolSize", "CandidatePoolTarget",
                              "CandidatePoolSize", "PoolSizeAssertion", "RelSetSize")
            ]
            has_seed = "seed" in (reader.fieldnames or ())
            for row in reader:
                if row["Method"] not in method_order:
                    method_order.append(row["Method"])
                seed_value = int(row["seed"]) if has_seed else run["default_seed"]
                if membership.get((seed_value, row["qid"])) != "validation":
                    continue
                key = (canonical_number(row[condition_name]), seed_value, row["Method"])
                slot = val_sums.setdefault(key, [0.0, 0.0])
                slot[0] += float(row[objective])
                slot[1] += 1.0

        def chosen(condition: str, seed_value: int, prefix: str) -> str:
            # first maximum in frozen grid (file encounter) order, the tie
            # rule of the archive builder and the original implementation
            members = [
                m for m in method_order
                if m.startswith(f"{prefix}(") and (condition, seed_value, m) in val_sums
            ]
            if not members:
                raise RuntimeError(f"no {prefix} members on validation split of {run['run']}")
            return max(members, key=lambda m: val_sums[(condition, seed_value, m)][0] / val_sums[(condition, seed_value, m)][1])

        # Pass 2: accumulate test means per (condition, series, metric),
        # seed observations averaged within query first.
        acc: Dict[Tuple[str, str, str], Dict[str, List[float]]] = defaultdict(dict)
        chosen_labels: Dict[Tuple[str, int, str], str] = {}
        for (condition, seed_value, qid), group in stream_groups(run["per_query"], condition_name, run["default_seed"]):
            if membership.get((seed_value, qid)) != "test":
                continue
            series_rows = {BASELINE: group.get(BASELINE)}
            for fixed in PAPER_FIXED:
                series_rows[fixed] = group.get(fixed)
            for prefix in PAPER_FAMILIES:
                label = chosen_labels.get((condition, seed_value, prefix))
                if label is None:
                    label = chosen(condition, seed_value, prefix)
                    chosen_labels[(condition, seed_value, prefix)] = label
                series_rows[f"{prefix}*"] = group.get(label)
            for series, row in series_rows.items():
                if row is None:
                    continue
                for metric in metric_fields:
                    value = row.get(metric, "")
                    if value in ("", None):
                        continue
                    slot = acc[(condition, series, metric)].setdefault(qid, [0.0, 0.0])
                    slot[0] += float(value)
                    slot[1] += 1.0

        rows = []
        for (condition, series, metric), per_query in sorted(acc.items()):
            means = [total / count for total, count in per_query.values()]
            rows.append({
                "run": run["run"], "dataset": run["dataset"], "experiment": run["experiment"],
                "condition_name": condition_name, "condition": condition, "series": series,
                "metric": metric, "n_test_queries": len(means), "mean": float(np.mean(means)),
                "chosen_members": ";".join(
                    f"{s}:{chosen_labels[(condition, s, p)]}"
                    for (c, s, p) in sorted(chosen_labels) if c == condition
                ) if series.endswith("*") else "",
            })
        path = out_dir / f"{run['run']}_level_means.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        try:
            display_path = path.relative_to(ROOT)
        except ValueError:
            display_path = path
        print(f"[table_refresh] {run['run']}: {len(rows)} rows -> {display_path}")


# ---------------------------------------------------------------------------
# Steps B + F: frozen decision rule versus kNN, budgeted threshold
# ---------------------------------------------------------------------------

def evidence_requirement(dataset: str, qid: str, row: Mapping[str, str]) -> Optional[float]:
    rule = EVIDENCE_REQUIREMENT.get(dataset)
    if rule is None:
        return None
    if rule == "hops":
        head = qid.split("hop")[0]
        try:
            return float(int(head))
        except ValueError:
            return None
    if rule == "relset":
        value = row.get("RelSetSize", "")
        if value in ("", None):
            return None
        return float(value)
    return float(rule)


def step_rule_transfer(runs: List[dict], samples: int, seed: int) -> None:
    out_dir = OUT_ROOT / "rule_transfer"
    gate_dir = OUT_ROOT / "gate_budgeted"
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[dict] = []
    skipped: List[str] = []
    for run in runs:
        condition_name = run["condition_name"]
        objective = run["objective"]
        dataset = run["dataset"]
        top_k = run["top_k"]
        membership = load_membership(dataset)

        # per condition: qid -> [sum_diff, n]; plus trigger counters
        diffs: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
        triggers: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
        missing_h = 0
        conditions: List[str] = []
        for (condition, seed_value, qid), group in stream_groups(run["per_query"], condition_name, run["default_seed"]):
            if condition not in conditions:
                conditions.append(condition)
            if membership.get((seed_value, qid)) != "test":
                continue
            knn = group.get(BASELINE)
            fallback = group.get(RULE_FALLBACK)
            if knn is None or fallback is None:
                raise RuntimeError(f"{run['run']}: missing {BASELINE} or {RULE_FALLBACK} row for {qid}")
            h = evidence_requirement(dataset, qid, knn)
            if h is None:
                missing_h += 1
                continue
            tau = min(h, float(top_k))
            trigger_stat = float(knn["Vendi"])
            fired = trigger_stat < tau
            value = float(fallback[objective]) if fired else float(knn[objective])
            diff = value - float(knn[objective])
            slot = diffs[condition].setdefault(qid, [0.0, 0.0])
            slot[0] += diff
            slot[1] += 1.0
            counter = triggers[condition]
            counter[0] += 1.0 if fired else 0.0
            counter[1] += 1.0

        if missing_h and not diffs:
            skipped.append(f"{run['run']}: no evidence requirement available (missing RelSetSize?)")
            continue

        qids = sorted({qid for condition in diffs for qid in diffs[condition]})
        columns = conditions + ["pooled"]
        matrix = np.full((len(qids), len(columns)), np.nan)
        index = {qid: i for i, qid in enumerate(qids)}
        for c_index, condition in enumerate(conditions):
            for qid, (total, count) in diffs[condition].items():
                matrix[index[qid], c_index] = 100.0 * total / count
        with np.errstate(invalid="ignore"):
            matrix[:, -1] = np.nanmean(matrix[:, :-1], axis=1)

        boots = bootstrap_columns(matrix, samples, run_seed_for(f"rule:{run['run']}", seed))
        observed = np.nanmean(matrix, axis=0)
        run_rows = []
        for c_index, condition in enumerate(columns):
            values = matrix[:, c_index]
            values = values[np.isfinite(values)]
            fired, total = triggers.get(condition, [float("nan"), float("nan")])
            run_rows.append({
                "run": run["run"], "dataset": dataset, "experiment": run["experiment"],
                "condition": condition, "n_test_queries": len(values),
                "trigger_rate": (fired / total) if condition != "pooled" and total else "",
                "mean_rule_minus_knn_pp": observed[c_index],
                "ci95_lo_pp": np.nanpercentile(boots[:, c_index], 2.5),
                "ci95_hi_pp": np.nanpercentile(boots[:, c_index], 97.5),
                "raw_p": centered_p(boots[:, c_index], observed[c_index], samples),
                "adjusted_p": "",
                "fraction_improved": float(np.mean(values > 0)),
                "fraction_harmed": float(np.mean(values < 0)),
                "skipped_queries_missing_h": missing_h,
            })
        adjusted = holm_adjust([row["raw_p"] for row in run_rows])
        for row, value in zip(run_rows, adjusted):
            row["adjusted_p"] = value
        all_rows.extend(run_rows)
        print(f"[rule_transfer] {run['run']}: {len(qids)} queries, pooled effect {observed[-1]:+.2f} pp")

    fields = list(all_rows[0].keys()) if all_rows else []
    with (out_dir / "rule_inference.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    beir = [row for row in all_rows if row["dataset"] in ("scifact", "fiqa", "trec-covid")]
    with (gate_dir / "gate_budgeted.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(beir)
    if skipped:
        (out_dir / "SKIPPED.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8")
        print("[rule_transfer] skipped:", "; ".join(skipped))


# ---------------------------------------------------------------------------
# Step C: paired EM/F1 inference from the retained generation files
# ---------------------------------------------------------------------------

def step_generation_inference(samples: int, seed: int) -> None:
    out_dir = OUT_ROOT / "generation_inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[dict] = []
    gen_paths = []
    for run_name in GENERATION_SOURCE_RUNS:
        candidates = (
            HISTORICAL_ROOT / run_name / "results_redundancy_gen_per_query.csv",
            RESULTS_ROOT / run_name / "results_redundancy_gen_per_query.csv",
        )
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is not None:
            gen_paths.append(path)
    if not gen_paths:
        print("[generation_inference] no results_redundancy_gen_per_query.csv found under results/; skipped")
        return
    for path in gen_paths:
        relative = str(path.relative_to(ROOT))
        run_dir = path.parent
        params_path = run_dir / "run_params.json"
        dataset = None
        if params_path.exists():
            with params_path.open(encoding="utf-8") as handle:
                dataset = json.load(handle).get("dataset")
        if dataset is None:
            dataset = "hotpotqa_fullwiki" if "hotpotqa" in run_dir.name else "2wikimultihopqa"
        membership = load_membership(dataset)

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            condition_name = "rho" if "rho" in fields else ("overlap" if "overlap" in fields else None)
            if condition_name is None or "qid" not in fields or "Method" not in fields:
                print(f"[generation_inference] unexpected schema in {relative}: {fields}; skipped")
                continue
            has_seed = "seed" in fields
            metric_fields = [f for f in fields if f.lower() in ("em", "f1", "halluc", "hallucination", "hallucinationrate")]
            data: Dict[Tuple[str, int, str], Dict[str, dict]] = defaultdict(dict)
            for row in reader:
                seed_value = int(row["seed"]) if has_seed else 0
                key = (canonical_number(row[condition_name]), seed_value, row["qid"])
                data[key][row["Method"]] = row

        methods = sorted({m for group in data.values() for m in group if m != BASELINE})
        # per (condition, method, metric): qid -> [sum_diff, n]
        acc: Dict[Tuple[str, str, str], Dict[str, List[float]]] = defaultdict(dict)
        conditions: List[str] = []
        for (condition, seed_value, qid), group in data.items():
            if condition not in conditions:
                conditions.append(condition)
            if membership.get((seed_value, qid)) != "test":
                continue
            knn = group.get(BASELINE)
            if knn is None:
                continue
            for method in methods:
                row = group.get(method)
                if row is None:
                    continue
                for metric in metric_fields:
                    if row.get(metric, "") in ("", None) or knn.get(metric, "") in ("", None):
                        continue
                    diff = float(row[metric]) - float(knn[metric])
                    slot = acc[(condition, method, metric)].setdefault(qid, [0.0, 0.0])
                    slot[0] += diff
                    slot[1] += 1.0

        conditions = sorted(conditions, key=lambda c: float(c))
        columns = sorted(acc)
        qids = sorted({qid for column in columns for qid in acc[column]})
        index = {qid: i for i, qid in enumerate(qids)}
        matrix = np.full((len(qids), len(columns)), np.nan)
        for c_index, column in enumerate(columns):
            for qid, (total, count) in acc[column].items():
                matrix[index[qid], c_index] = 100.0 * total / count
        boots = bootstrap_columns(matrix, samples, run_seed_for(f"generation:{run_dir.name}", seed))
        observed = np.nanmean(matrix, axis=0)
        run_rows = []
        for c_index, (condition, method, metric) in enumerate(columns):
            values = matrix[:, c_index]
            values = values[np.isfinite(values)]
            run_rows.append({
                "source": relative, "dataset": dataset, "condition": condition,
                "method": method, "metric": metric,
                "contrast": "paired risk difference (pp)" if metric.lower() == "em" else "paired mean difference (pp)",
                "n_test_queries": len(values),
                "mean_diff_pp": observed[c_index],
                "ci95_lo_pp": np.nanpercentile(boots[:, c_index], 2.5),
                "ci95_hi_pp": np.nanpercentile(boots[:, c_index], 97.5),
                "raw_p": centered_p(boots[:, c_index], observed[c_index], samples),
                "adjusted_p": "",
            })
        adjusted = holm_adjust([row["raw_p"] for row in run_rows])
        for row, value in zip(run_rows, adjusted):
            row["adjusted_p"] = value
        all_rows.extend(run_rows)
        print(f"[generation_inference] {run_dir.name}: {len(run_rows)} contrasts over {len(qids)} queries")

    if all_rows:
        with (out_dir / "generation_inference.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(all_rows)


# ---------------------------------------------------------------------------
# Steps D + E: oracle headroom, capture ratios, downside, minimax regret
# (headline HotpotQA injection sweep)
# ---------------------------------------------------------------------------

def step_oracle_and_regret(runs: List[dict], samples: int, seed: int) -> None:
    oracle_dir = OUT_ROOT / "oracle_intervals"
    regret_dir = OUT_ROOT / "minimax_regret"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    regret_dir.mkdir(parents=True, exist_ok=True)
    headline = [
        run for run in runs
        if run["experiment"] == "redundancy" and "hotpotqa_fullwiki" in run["run"]
    ]
    if not headline:
        print("[oracle/minimax] headline HotpotQA injection run not found; skipped")
        return
    headline.sort(key=lambda run: next(
        (i for i, prefix in enumerate(HEADLINE_PREFERENCE) if run["run"].startswith(prefix)),
        len(HEADLINE_PREFERENCE),
    ))
    run = headline[0]
    print(f"[oracle/minimax] headline run: {run['run']}")
    condition_name = run["condition_name"]
    objective = run["objective"]
    top_k = run["top_k"]
    membership = load_membership(run["dataset"])

    conditions: List[str] = []
    member_names: List[str] = []
    # per condition: qid -> {series: [sum, n]}; series = members, oracle, rule
    acc: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(dict))
    for (condition, seed_value, qid), group in stream_groups(run["per_query"], condition_name, run["default_seed"]):
        if condition not in conditions:
            conditions.append(condition)
        if membership.get((seed_value, qid)) != "test":
            continue
        knn_row = group.get(BASELINE)
        if knn_row is None:
            continue
        knn = float(knn_row[objective])
        oracle_best = knn
        for method, row in group.items():
            if not is_oracle_member(method):
                continue
            value = float(row[objective])
            if method not in member_names:
                member_names.append(method)
            slot = acc[condition][method].setdefault(qid, [0.0, 0.0])
            slot[0] += value
            slot[1] += 1.0
            oracle_best = max(oracle_best, value)
        slot = acc[condition]["__oracle__"].setdefault(qid, [0.0, 0.0])
        slot[0] += oracle_best
        slot[1] += 1.0
        # frozen rule (tau = min(h,k) with h = 2 on HotpotQA, D* = MMR(0.7))
        h = evidence_requirement(run["dataset"], qid, knn_row)
        fallback = group.get(RULE_FALLBACK)
        if h is not None and fallback is not None:
            fired = float(knn_row["Vendi"]) < min(h, float(top_k))
            value = float(fallback[objective]) if fired else knn
            slot = acc[condition]["__rule__"].setdefault(qid, [0.0, 0.0])
            slot[0] += value
            slot[1] += 1.0

    series = member_names + ["__oracle__", "__rule__"]
    qids = sorted({qid for condition in acc for s in acc[condition] for qid in acc[condition][s]})
    index = {qid: i for i, qid in enumerate(qids)}
    columns = [(condition, s) for condition in conditions for s in series]
    matrix = np.full((len(qids), len(columns)), np.nan)
    for c_index, (condition, s) in enumerate(columns):
        for qid, (total, count) in acc[condition].get(s, {}).items():
            matrix[index[qid], c_index] = total / count
    boots = bootstrap_columns(matrix, samples, run_seed_for(f"oracle:{run['run']}", seed))
    observed = np.nanmean(matrix, axis=0)

    def col(condition: str, s: str) -> int:
        return columns.index((condition, s))

    def series_matrix(source: np.ndarray, s: str) -> np.ndarray:
        return np.stack([source[..., col(condition, s)] for condition in conditions], axis=-1)

    # --- oracle headroom, rule gain, capture ratios (pooled over levels) ---
    knn_levels = series_matrix(observed[None, :], BASELINE)[0]
    oracle_levels = series_matrix(observed[None, :], "__oracle__")[0]
    rule_levels = series_matrix(observed[None, :], "__rule__")[0]
    boots_knn = series_matrix(boots, BASELINE)
    boots_oracle = series_matrix(boots, "__oracle__")
    boots_rule = series_matrix(boots, "__rule__")

    # per-level tuned best among displayed families: use the best member mean
    # per level over the oracle library (matches the per-level tuned selector
    # up to validation noise; recorded as such in the output).
    member_matrix = np.stack([series_matrix(observed[None, :], m)[0] for m in member_names], axis=0)
    tuned_levels = member_matrix.max(axis=0)
    boots_members = np.stack([series_matrix(boots, m) for m in member_names], axis=0)
    boots_tuned = boots_members.max(axis=0)

    rows = []

    def add(name, observed_value, boot_values):
        rows.append({
            "run": run["run"], "quantity": name,
            "estimate": float(observed_value),
            "ci95_lo": float(np.nanpercentile(boot_values, 2.5)),
            "ci95_hi": float(np.nanpercentile(boot_values, 97.5)),
        })

    add("oracle_headroom_pooled_pp", 100.0 * float(np.mean(oracle_levels - knn_levels)),
        100.0 * np.mean(boots_oracle - boots_knn, axis=1))
    add("rule_gain_pooled_pp", 100.0 * float(np.mean(rule_levels - knn_levels)),
        100.0 * np.mean(boots_rule - boots_knn, axis=1))
    add("tuned_gain_pooled_pp", 100.0 * float(np.mean(tuned_levels - knn_levels)),
        100.0 * np.mean(boots_tuned - boots_knn, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        add("rule_capture_of_oracle", float(np.mean(rule_levels - knn_levels) / np.mean(oracle_levels - knn_levels)),
            np.mean(boots_rule - boots_knn, axis=1) / np.mean(boots_oracle - boots_knn, axis=1))
        add("rule_capture_of_tuned", float(np.mean(rule_levels - knn_levels) / np.mean(tuned_levels - knn_levels)),
            np.mean(boots_rule - boots_knn, axis=1) / np.mean(boots_tuned - boots_knn, axis=1))

    # --- misspecification downside per displayed family, with intervals ---
    def display_family(method: str) -> Optional[str]:
        return family_of(method)

    families = sorted({display_family(m) for m in member_names if display_family(m) and m != BASELINE})
    for family in families:
        members = [m for m in member_names if display_family(m) == family]
        member_stack = np.stack([series_matrix(observed[None, :], m)[0] for m in members], axis=0)
        downside = float(np.maximum(0.0, (knn_levels[None, :] - member_stack)).max())
        boot_stack = np.stack([series_matrix(boots, m) for m in members], axis=0)
        boot_downside = np.maximum(0.0, boots_knn[None, ...] - boot_stack).max(axis=(0, 2))
        add(f"downside_{family}_pp", 100.0 * downside, 100.0 * boot_downside)

    with (oracle_dir / "oracle_intervals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[oracle_intervals] {run['run']}: {len(rows)} quantities")

    # --- common-oracle minimax regret over the clean/heavy regimes ---
    regimes = [conditions[0], conditions[-1]]
    action_names = [BASELINE] + [m for m in member_names if display_family(m) and m != BASELINE]
    action_values = np.array([[observed[col(r, a)] for r in regimes] for a in action_names])
    oracle_by_regime = action_values.max(axis=0)
    regret_by_action = (oracle_by_regime[None, :] - action_values).max(axis=1)
    boot_actions = np.stack([np.stack([boots[:, col(r, a)] for r in regimes], axis=-1) for a in action_names], axis=0)
    boot_oracle = boot_actions.max(axis=0)
    boot_regret = (boot_oracle[None, ...] - boot_actions).max(axis=-1)

    regret_rows = []
    for family in [BASELINE] + families:
        if family == BASELINE:
            indices = [action_names.index(BASELINE)]
        else:
            indices = [i for i, a in enumerate(action_names) if display_family(a) == family]
        best = int(np.argmin(regret_by_action[indices]))
        chosen_index = indices[best]
        regret_rows.append({
            "run": run["run"], "family": family,
            "minimax_regret_pp": 100.0 * float(regret_by_action[chosen_index]),
            "argmin_member": action_names[chosen_index],
            "ci95_lo_pp": 100.0 * float(np.nanpercentile(boot_regret[indices].min(axis=0), 2.5)),
            "ci95_hi_pp": 100.0 * float(np.nanpercentile(boot_regret[indices].min(axis=0), 97.5)),
            "regimes": f"{condition_name}={regimes[0]} vs {condition_name}={regimes[1]}",
            "action_library": "kNN + all displayed-family grid members (Seg excluded)",
        })
    with (regret_dir / "minimax_regret.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(regret_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(regret_rows)
    print(f"[minimax_regret] {run['run']}: {len(regret_rows)} rows")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--download-reruns", action="store_true",
                        help="fetch the fixed-pool re-executions from the public reproducibility release first")
    parser.add_argument(
        "--source-mode", choices=("auto", "corrected", "historical"), default="auto",
        help="input family to analyze; auto prefers corrected reruns when present",
    )
    parser.add_argument("--only", type=str, default="", help="comma-separated subset of steps: table_refresh,rule_transfer,generation_inference,oracle_minimax")
    args = parser.parse_args()

    if args.download_reruns:
        download_reruns()
    if not SPLIT_ROOT.exists():
        raise SystemExit(f"missing frozen split manifests under {SPLIT_ROOT}")
    mode, runs = discover_runs(args.source_mode)
    if not runs:
        raise SystemExit("no sweep run directories with per-query files found under results/")
    print(f"Source mode: {mode}. Found {len(runs)} sweep artifacts.")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected = {step.strip() for step in args.only.split(",") if step.strip()} or {
        "table_refresh", "rule_transfer", "generation_inference", "oracle_minimax",
    }
    complementary_outputs_exist = any(
        (OUT_ROOT / folder).exists()
        for folder in ("rule_transfer", "gate_budgeted", "generation_inference",
                       "oracle_intervals", "minimax_regret")
    )
    if mode == "corrected" and selected == {"table_refresh"} and complementary_outputs_exist:
        source_description = (
            "mixed\n"
            "table_refresh/ was computed from the fixed-pool re-executions under "
            "results/reproducibility_reruns/.\n"
            "rule_transfer/, gate_budgeted/, generation_inference/, oracle_intervals/ "
            "and minimax_regret/ retain historical-mode outputs from results/retained/.\n"
        )
    elif mode == "corrected":
        source_description = "corrected\nValues computed from the fixed-pool re-executions.\n"
    else:
        source_description = (
            "historical\n"
            "Values computed from the originally executed runs displayed by the paper;\n"
            "the fixed-pool table refresh requires corrected mode.\n"
        )
    (OUT_ROOT / "SOURCE_MODE.txt").write_text(source_description, encoding="utf-8")

    if "table_refresh" in selected:
        if mode == "corrected":
            step_table_refresh(runs, args.bootstrap_samples, args.bootstrap_seed)
        else:
            print("[table_refresh] skipped: needs the fixed-pool re-executions (historical mode); "
                  "run with --download-reruns when convenient")
    if "rule_transfer" in selected:
        step_rule_transfer(runs, args.bootstrap_samples, args.bootstrap_seed)
    if "generation_inference" in selected:
        step_generation_inference(args.bootstrap_samples, args.bootstrap_seed)
    if "oracle_minimax" in selected:
        step_oracle_and_regret(runs, args.bootstrap_samples, args.bootstrap_seed)
    print(f"Done. Outputs under {OUT_ROOT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
