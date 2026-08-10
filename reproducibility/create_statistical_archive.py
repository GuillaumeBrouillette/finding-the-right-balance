#!/usr/bin/env python3
"""Build the paired-inference archive from corrected sweep results.

Repeated seeds are not treated as independent studies.
For each corrected sweep this script therefore:

1. reproduces the validation-selected method member recorded in the per-seed
   summary;
2. forms method-minus-kNN contrasts on held-out test queries;
3. averages repeated seed observations within query;
4. resamples whole query trajectories with an ordinary cluster bootstrap; and
5. applies Holm's step-down correction within declared level and interaction
   families.

No encoder, reranker, generator, or GPU is invoked.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results" / "reproducibility_reruns"
SPLIT_ROOT = ROOT / "manifests" / "splits"
OUTPUT = ROOT / "manifests" / "statistics"
BASELINE = "kNN"
DEFAULT_BOOTSTRAP_SAMPLES = 1999
DEFAULT_BOOTSTRAP_SEED = 20260809
METHOD_ORDER = ("Maxmin", "Greedy-DPP", "Dedup*", "MMR*", "VendiG*", "RNG*")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_number(value: str) -> str:
    number = Decimal(value)
    if number == 0:
        return "0"
    return format(number.normalize(), "f")


def deterministic_gzip_text(path: Path):
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="")


def load_test_membership(dataset: str) -> Dict[Tuple[int, str], str]:
    path = SPLIT_ROOT / f"{dataset}.csv"
    membership: Dict[Tuple[int, str], str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            membership[(int(row["seed"]), row["query_id"])] = row["partition"]
    return membership


def discover_runs() -> List[dict]:
    runs: List[dict] = []
    for params_path in sorted(RUN_ROOT.glob("*/run_params.json")):
        run_dir = params_path.parent
        with params_path.open(encoding="utf-8") as handle:
            params = json.load(handle)
        for experiment, stem in (
            ("redundancy", "results_redundancy"),
            ("chunking", "results_chunking"),
        ):
            per_query = run_dir / f"{stem}_per_query.csv"
            per_seed = run_dir / f"{stem}_per_seed_summary.csv"
            if not per_query.exists() or not per_seed.exists():
                continue
            runs.append(
                {
                    "run": f"{run_dir.name}__{experiment}",
                    "source_run": run_dir.name,
                    "run_dir": str(run_dir),
                    "dataset": params["dataset"],
                    "experiment": experiment,
                    # The manuscript defines S-Recall as the primary
                    # endpoint for the redundancy/chunking sweeps. Several old
                    # run_params files retained an alpha-NDCG objective, so the
                    # archive deliberately corrects and re-tunes this here.
                    "objective": "S-Recall@k",
                    "historical_objective": params["objective"],
                    "condition_name": "rho" if experiment == "redundancy" else "overlap",
                    "per_query": str(per_query),
                    "per_seed": str(per_seed),
                    "run_params": str(params_path),
                }
            )
    if len(runs) != 13:
        raise RuntimeError(f"expected 13 corrected sweep artifacts, found {len(runs)}")
    return runs


def recompute_choices(
    run: Mapping[str, str], membership: Mapping[Tuple[int, str], str]
) -> Tuple[Dict[Tuple[str, int], Dict[str, str]], List[str], List[dict]]:
    """Tune each parameterized family on frozen validation S-Recall.

    Candidate order is the stable order in the per-query file, which matches
    the run's frozen parameter grids.  Python's ``max`` returns the first
    maximizer, reproducing the original tie rule.
    """
    condition_name = run["condition_name"]
    objective = run["objective"]
    sums: Dict[Tuple[str, int, str], list] = {}
    conditions: List[str] = []
    method_order: List[str] = []
    condition_seeds: List[Tuple[str, int]] = []
    with Path(run["per_query"]).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            condition = canonical_number(row[condition_name])
            if condition not in conditions:
                conditions.append(condition)
            seed = int(row["seed"])
            condition_seed = (condition, seed)
            if condition_seed not in condition_seeds:
                condition_seeds.append(condition_seed)
            method = row["Method"]
            if method not in method_order:
                method_order.append(method)
            if membership.get((seed, row["qid"])) != "validation":
                continue
            slot = sums.setdefault((condition, seed, method), [0.0, 0])
            slot[0] += float(row[objective])
            slot[1] += 1

    choices: Dict[Tuple[str, int], Dict[str, str]] = {}
    selection_rows: List[dict] = []
    for condition, seed in condition_seeds:
        selected = {"Maxmin": "Maxmin", "Greedy-DPP": "Greedy-DPP"}
        for prefix in ("Dedup", "MMR", "VendiG", "RNG"):
            members = [method for method in method_order if method.startswith(f"{prefix}(")]
            available = [method for method in members if (condition, seed, method) in sums]
            if not available:
                raise RuntimeError(f"no validation members for {prefix} in {run['run']} {(condition, seed)}")
            selected[f"{prefix}*"] = max(
                available,
                key=lambda method: sums[(condition, seed, method)][0] / sums[(condition, seed, method)][1],
            )
        choices[(condition, seed)] = selected
        for method in METHOD_ORDER:
            chosen = selected[method]
            total, count = sums[(condition, seed, chosen)]
            selection_rows.append(
                {
                    "run": run["run"],
                    "dataset": run["dataset"],
                    "experiment": run["experiment"],
                    "condition_name": condition_name,
                    "condition": condition,
                    "seed": seed,
                    "reported_method": method,
                    "chosen_method": chosen,
                    "selection_objective": objective,
                    "validation_query_count": count,
                    "validation_mean": total / count,
                    "tie_rule": "first maximum in frozen grid order",
                }
            )
    return choices, conditions, selection_rows


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def bootstrap_columns(
    matrix: np.ndarray,
    samples: int,
    seed: int,
    block_size: int = 32,
) -> np.ndarray:
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
            numerators,
            denominators,
            out=np.full_like(numerators, np.nan),
            where=denominators != 0,
        )
    return output


def infer_run(run: Mapping[str, str], output_root: str, samples: int, global_seed: int) -> dict:
    out = Path(output_root)
    run_name = run["run"]
    dataset = run["dataset"]
    objective = run["objective"]
    condition_name = run["condition_name"]
    membership = load_test_membership(dataset)
    choices, condition_order, selection_rows = recompute_choices(run, membership)

    raw_path = out / "raw_seed_contrasts" / f"{run_name}.csv.gz"
    averaged_path = out / "query_averaged_contrasts" / f"{run_name}.csv.gz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    averaged_path.parent.mkdir(parents=True, exist_ok=True)

    raw_fields = [
        "run", "dataset", "experiment", "condition_name", "condition", "seed",
        "qid", "method", "chosen_method", "objective", "method_value",
        "baseline_value", "paired_difference", "paired_difference_pp",
    ]
    averaged_fields = [
        "run", "dataset", "experiment", "condition_name", "condition", "qid",
        "method", "objective", "n_seed_observations", "seed_chosen_methods",
        "mean_method_value", "mean_baseline_value", "mean_paired_difference",
        "mean_paired_difference_pp",
    ]

    # condition -> (qid, reported method) -> sums/count/chosen labels
    accumulated: Dict[str, MutableMapping[Tuple[str, str], list]] = {
        condition: {} for condition in condition_order
    }
    raw_count = 0
    skipped_validation_groups = 0
    group_count = 0

    with deterministic_gzip_text(raw_path) as raw_handle:
        raw_writer = csv.DictWriter(raw_handle, fieldnames=raw_fields, lineterminator="\n")
        raw_writer.writeheader()

        with Path(run["per_query"]).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {condition_name, "seed", "qid", "Method", objective}
            if not required.issubset(reader.fieldnames or ()):
                raise RuntimeError(f"missing columns {sorted(required - set(reader.fieldnames or ())) } in {run['per_query']}")

            current_key = None
            group_rows: Dict[str, float] = {}

            def flush_group(key, values):
                nonlocal raw_count, skipped_validation_groups, group_count
                if key is None:
                    return
                condition, seed, qid = key
                group_count += 1
                if membership.get((seed, qid)) != "test":
                    skipped_validation_groups += 1
                    return
                if BASELINE not in values:
                    raise RuntimeError(f"missing kNN row in {run_name}: {key}")
                selected = choices[(condition, seed)]
                baseline_value = values[BASELINE]
                for method in METHOD_ORDER:
                    chosen = selected[method]
                    if chosen not in values:
                        raise RuntimeError(f"missing chosen row {chosen} in {run_name}: {key}")
                    method_value = values[chosen]
                    difference = method_value - baseline_value
                    raw_writer.writerow(
                        {
                            "run": run_name,
                            "dataset": dataset,
                            "experiment": run["experiment"],
                            "condition_name": condition_name,
                            "condition": condition,
                            "seed": seed,
                            "qid": qid,
                            "method": method,
                            "chosen_method": chosen,
                            "objective": objective,
                            "method_value": repr(method_value),
                            "baseline_value": repr(baseline_value),
                            "paired_difference": repr(difference),
                            "paired_difference_pp": repr(100.0 * difference),
                        }
                    )
                    raw_count += 1
                    slot = accumulated[condition].setdefault(
                        (qid, method), [0.0, 0.0, 0, []]
                    )
                    slot[0] += method_value
                    slot[1] += baseline_value
                    slot[2] += 1
                    slot[3].append(f"{seed}:{chosen}")

            for row in reader:
                key = (
                    canonical_number(row[condition_name]),
                    int(row["seed"]),
                    row["qid"],
                )
                if current_key is not None and key != current_key:
                    flush_group(current_key, group_rows)
                    group_rows = {}
                current_key = key
                group_rows[row["Method"]] = float(row[objective])
            flush_group(current_key, group_rows)

    trajectory: Dict[str, Dict[Tuple[str, str], float]] = {}
    averaged_count = 0
    with deterministic_gzip_text(averaged_path) as averaged_handle:
        writer = csv.DictWriter(averaged_handle, fieldnames=averaged_fields, lineterminator="\n")
        writer.writeheader()
        for condition in condition_order:
            for (qid, method), (method_sum, baseline_sum, count, chosen) in sorted(
                accumulated[condition].items(), key=lambda item: (item[0][0], METHOD_ORDER.index(item[0][1]))
            ):
                method_mean = method_sum / count
                baseline_mean = baseline_sum / count
                difference = method_mean - baseline_mean
                writer.writerow(
                    {
                        "run": run_name,
                        "dataset": dataset,
                        "experiment": run["experiment"],
                        "condition_name": condition_name,
                        "condition": condition,
                        "qid": qid,
                        "method": method,
                        "objective": objective,
                        "n_seed_observations": count,
                        "seed_chosen_methods": ";".join(chosen),
                        "mean_method_value": repr(method_mean),
                        "mean_baseline_value": repr(baseline_mean),
                        "mean_paired_difference": repr(difference),
                        "mean_paired_difference_pp": repr(100.0 * difference),
                    }
                )
                trajectory.setdefault(qid, {})[(condition, method)] = 100.0 * difference
                averaged_count += 1

    qids = sorted(trajectory)
    level_columns = [(condition, method) for condition in condition_order for method in METHOD_ORDER]
    matrix = np.full((len(qids), len(level_columns)), np.nan, dtype=np.float64)
    for q_index, qid in enumerate(qids):
        values = trajectory[qid]
        for c_index, column in enumerate(level_columns):
            value = values.get(column)
            if value is not None:
                matrix[q_index, c_index] = value

    clean_condition = condition_order[0]
    heavy_condition = condition_order[-1]
    interaction = np.full((len(qids), len(METHOD_ORDER)), np.nan, dtype=np.float64)
    for method_index, method in enumerate(METHOD_ORDER):
        clean_index = level_columns.index((clean_condition, method))
        heavy_index = level_columns.index((heavy_condition, method))
        valid = np.isfinite(matrix[:, clean_index]) & np.isfinite(matrix[:, heavy_index])
        interaction[valid, method_index] = matrix[valid, heavy_index] - matrix[valid, clean_index]

    combined = np.concatenate([matrix, interaction], axis=1)
    run_seed = int(hashlib.sha256(f"{global_seed}:{run_name}".encode()).hexdigest()[:16], 16)
    bootstrap = bootstrap_columns(combined, samples, run_seed)
    observed = np.nanmean(combined, axis=0)
    summary_rows: List[dict] = []

    for index, (condition, method) in enumerate(level_columns):
        values = matrix[:, index]
        values = values[np.isfinite(values)]
        boots = bootstrap[:, index]
        centered = boots - observed[index]
        raw_p = (1.0 + np.count_nonzero(np.abs(centered) >= abs(observed[index]))) / (samples + 1.0)
        summary_rows.append(
            {
                "run": run_name,
                "dataset": dataset,
                "experiment": run["experiment"],
                "contrast_type": "level_effect",
                "condition_name": condition_name,
                "condition": condition,
                "clean_condition": "",
                "heavy_condition": "",
                "method": method,
                "objective": objective,
                "n_independent_queries": len(values),
                "mean_paired_difference_pp": observed[index],
                "ci95_lo_pp": np.nanpercentile(boots, 2.5),
                "ci95_hi_pp": np.nanpercentile(boots, 97.5),
                "one_sided_95_lower_pp": "",
                "fraction_improved": np.mean(values > 0),
                "fraction_tied": np.mean(values == 0),
                "fraction_harmed": np.mean(values < 0),
                "alternative": "two-sided",
                "raw_p": raw_p,
                "adjusted_p": "",
                "multiplicity_family": f"{run_name}:all_method_by_level_effects",
            }
        )

    interaction_offset = len(level_columns)
    for method_index, method in enumerate(METHOD_ORDER):
        index = interaction_offset + method_index
        values = interaction[:, method_index]
        values = values[np.isfinite(values)]
        boots = bootstrap[:, index]
        centered = boots - observed[index]
        raw_p = (1.0 + np.count_nonzero(centered >= observed[index])) / (samples + 1.0)
        summary_rows.append(
            {
                "run": run_name,
                "dataset": dataset,
                "experiment": run["experiment"],
                "contrast_type": "heavy_minus_clean_interaction",
                "condition_name": condition_name,
                "condition": "",
                "clean_condition": clean_condition,
                "heavy_condition": heavy_condition,
                "method": method,
                "objective": objective,
                "n_independent_queries": len(values),
                "mean_paired_difference_pp": observed[index],
                "ci95_lo_pp": np.nanpercentile(boots, 2.5),
                "ci95_hi_pp": np.nanpercentile(boots, 97.5),
                "one_sided_95_lower_pp": np.nanpercentile(boots, 5.0),
                "fraction_improved": np.mean(values > 0),
                "fraction_tied": np.mean(values == 0),
                "fraction_harmed": np.mean(values < 0),
                "alternative": "greater",
                "raw_p": raw_p,
                "adjusted_p": "",
                "multiplicity_family": f"{run_name}:primary_interactions",
            }
        )

    families: Dict[str, List[int]] = {}
    for index, row in enumerate(summary_rows):
        families.setdefault(row["multiplicity_family"], []).append(index)
    for indices in families.values():
        adjusted = holm_adjust([float(summary_rows[index]["raw_p"]) for index in indices])
        for index, value in zip(indices, adjusted):
            summary_rows[index]["adjusted_p"] = value

    return {
        "run": run_name,
        "dataset": dataset,
        "experiment": run["experiment"],
        "objective": objective,
        "conditions": condition_order,
        "raw_seed_contrast_rows": raw_count,
        "query_averaged_contrast_rows": averaged_count,
        "unique_query_ids": len(qids),
        "source_group_count": group_count,
        "validation_groups_excluded": skipped_validation_groups,
        "summary_rows": summary_rows,
        "selection_rows": selection_rows,
        "historical_objective": run["historical_objective"],
        "inputs": {
            "per_query": {"path": os.path.relpath(run["per_query"], ROOT), "sha256": sha256_file(Path(run["per_query"]))},
            "per_seed_summary": {"path": os.path.relpath(run["per_seed"], ROOT), "sha256": sha256_file(Path(run["per_seed"]))},
            "run_params": {"path": os.path.relpath(run["run_params"], ROOT), "sha256": sha256_file(Path(run["run_params"]))},
            "split_manifest": {"path": f"manifests/splits/{dataset}.csv", "sha256": sha256_file(SPLIT_ROOT / f"{dataset}.csv")},
        },
        "outputs": {
            "raw_seed_contrasts": os.path.relpath(raw_path, out),
            "query_averaged_contrasts": os.path.relpath(averaged_path, out),
        },
    }


def write_manifest(output: Path, run_results: List[dict], samples: int, seed: int, workers: int) -> None:
    summary_fields = [
        "run", "dataset", "experiment", "contrast_type", "condition_name", "condition",
        "clean_condition", "heavy_condition", "method", "objective", "n_independent_queries",
        "mean_paired_difference_pp", "ci95_lo_pp", "ci95_hi_pp", "one_sided_95_lower_pp",
        "fraction_improved", "fraction_tied", "fraction_harmed", "alternative", "raw_p",
        "adjusted_p", "multiplicity_family",
    ]
    summary_path = output / "paired_inference.csv"
    all_rows: List[dict] = []
    all_selection_rows: List[dict] = []
    for result in sorted(run_results, key=lambda item: item["run"]):
        all_rows.extend(result.pop("summary_rows"))
        all_selection_rows.extend(result.pop("selection_rows"))
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)

    selection_fields = [
        "run", "dataset", "experiment", "condition_name", "condition", "seed",
        "reported_method", "chosen_method", "selection_objective",
        "validation_query_count", "validation_mean", "tie_rule",
    ]
    with (output / "validation_selections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selection_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_selection_rows)

    totals = {
        "corrected_sweep_artifacts": len(run_results),
        "raw_seed_contrast_rows": sum(item["raw_seed_contrast_rows"] for item in run_results),
        "query_averaged_contrast_rows": sum(item["query_averaged_contrast_rows"] for item in run_results),
        "level_effect_rows": sum(row["contrast_type"] == "level_effect" for row in all_rows),
        "primary_interaction_rows": sum(row["contrast_type"] == "heavy_minus_clean_interaction" for row in all_rows),
        "inference_rows": len(all_rows),
        "validation_selection_rows": len(all_selection_rows),
    }
    metadata = {
        "format_version": 1,
        "status": "addressed_with_archival_limitation",
        "scope": "primary objective contrasts for all 13 corrected injection/chunking sweep artifacts",
        "baseline": BASELINE,
        "reported_method_families": list(METHOD_ORDER),
        "sampling_unit": "frozen query_id; repeated test-seed observations averaged within query",
        "primary_objective": "S-Recall@k for every corrected sweep artifact",
        "selection": "each parameterized family is re-tuned per seed and level on frozen validation S-Recall; first maximum in frozen grid order breaks exact ties",
        "confidence_interval": {
            "method": "ordinary nonparametric query-cluster percentile bootstrap",
            "confidence": 0.95,
            "resamples": samples,
            "seed": seed,
            "trajectory_preservation": "one multinomial query-weight vector is shared across every method and redundancy level in a run",
        },
        "hypothesis_tests": {
            "level_effects": "two-sided centered cluster-bootstrap test of mean(method-kNN)=0",
            "primary_interactions": "one-sided centered cluster-bootstrap test that (heavy-clean method effect)>0",
            "finite_monte_carlo_correction": "(extreme+1)/(B+1)",
        },
        "multiplicity": {
            "method": "Holm step-down familywise correction",
            "level_family": "all six method-by-level primary-objective contrasts within one corrected run",
            "interaction_family": "six prespecified heavy-minus-clean method interactions within one corrected run",
        },
        "units": "paired effects and confidence intervals are percentage points",
        "totals": totals,
        "runs": sorted(run_results, key=lambda item: item["run"]),
        "archival_limitations": [
            "No frozen query-family identifier links questions that may share source documents; query_id is therefore the finest available sampling unit.",
            "Historical Table 2 generation has aggregate-only outputs, so raw EM/F1 paired contrasts cannot be recovered.",
            "Historical Table 3 ArguAna and Touché outputs are aggregate-only, so raw paired contrasts cannot be recovered.",
            "Historical Table 4 non-Hotpot cross-encoder outputs are aggregate-only, so raw paired contrasts cannot be recovered.",
        ],
        "execution_workers": workers,
    }

    # Add checksums after all run files and the summary exist.
    artifacts = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"metadata.json", "SHA256SUMS"}:
            artifacts.append(
                {
                    "path": os.path.relpath(path, output),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    metadata["artifacts"] = artifacts
    metadata_path = output / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    checksum_paths = [path for path in sorted(output.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"]
    with (output / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in checksum_paths:
            handle.write(f"{sha256_file(path)}  {os.path.relpath(path, output)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.workers < 1 or args.bootstrap_samples < 99:
        parser.error("--workers must be >=1 and --bootstrap-samples must be >=99")

    runs = discover_runs()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="statistics-build-", dir=OUTPUT.parent))
    try:
        results: List[dict] = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(infer_run, run, str(temp_dir), args.bootstrap_samples, args.bootstrap_seed): run
                for run in runs
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"[{len(results):02d}/{len(runs):02d}] {result['run']}: "
                    f"{result['raw_seed_contrast_rows']} raw, "
                    f"{result['query_averaged_contrast_rows']} averaged",
                    flush=True,
                )
        write_manifest(temp_dir, results, args.bootstrap_samples, args.bootstrap_seed, args.workers)
        if OUTPUT.exists():
            shutil.rmtree(OUTPUT)
        temp_dir.replace(OUTPUT)
        with (OUTPUT / "metadata.json").open(encoding="utf-8") as handle:
            totals = json.load(handle)["totals"]
        print(
            "Statistical archive: "
            f"{totals['corrected_sweep_artifacts']} sweeps, "
            f"{totals['raw_seed_contrast_rows']} raw contrasts, "
            f"{totals['inference_rows']} inference rows"
        )
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
