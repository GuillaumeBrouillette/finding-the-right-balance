#!/usr/bin/env python3
"""Independently validate the generated Statistical archive paired-inference archive."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from create_statistical_archive import (
    METHOD_ORDER,
    bootstrap_columns,
    canonical_number,
    holm_adjust,
    recompute_choices,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "manifests" / "statistics"
TOLERANCE = 1e-10


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=TOLERANCE, abs_tol=TOLERANCE):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def validate_checksums(metadata: dict) -> None:
    checksum_path = ARCHIVE / "SHA256SUMS"
    listed = {}
    with checksum_path.open(encoding="utf-8") as handle:
        for line in handle:
            digest, relative = line.rstrip("\n").split("  ", 1)
            path = ARCHIVE / relative
            if not path.is_file():
                raise AssertionError(f"missing checksum target: {relative}")
            actual = sha256_file(path)
            if actual != digest:
                raise AssertionError(f"SHA-256 mismatch: {relative}")
            listed[relative] = digest
    expected_files = {
        str(path.relative_to(ARCHIVE))
        for path in ARCHIVE.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(listed) != expected_files:
        raise AssertionError("SHA256SUMS file set does not match archive")
    for artifact in metadata["artifacts"]:
        path = ARCHIVE / artifact["path"]
        if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise AssertionError(f"metadata artifact mismatch: {artifact['path']}")


def load_summary() -> dict:
    rows = {}
    with (ARCHIVE / "paired_inference.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["run"], row["contrast_type"], row["condition"],
                row["clean_condition"], row["heavy_condition"], row["method"],
            )
            if key in rows:
                raise AssertionError(f"duplicate inference key: {key}")
            rows[key] = row
    return rows


def load_selections() -> dict:
    rows = {}
    with (ARCHIVE / "validation_selections.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["run"], row["condition"], int(row["seed"]), row["reported_method"]
            )
            if key in rows:
                raise AssertionError(f"duplicate validation selection: {key}")
            rows[key] = row
    return rows


def validate_run(run: dict, metadata: dict, summary: dict, selections: dict) -> tuple[int, int, int, int]:
    raw_path = ARCHIVE / run["outputs"]["raw_seed_contrasts"]
    averaged_path = ARCHIVE / run["outputs"]["query_averaged_contrasts"]
    aggregate = defaultdict(lambda: [0.0, 0.0, 0, []])
    raw_count = 0
    raw_keys = set()
    split_rows = {}
    split_path = ROOT / run["inputs"]["split_manifest"]["path"]
    with split_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split_rows[(int(row["seed"]), row["query_id"])] = row["partition"]

    # Recompute validation tuning directly from the hashed per-query source.
    selection_run = {
        "run": run["run"],
        "dataset": run["dataset"],
        "experiment": run["experiment"],
        "condition_name": "rho" if run["experiment"] == "redundancy" else "overlap",
        "objective": run["objective"],
        "per_query": str(ROOT / run["inputs"]["per_query"]["path"]),
    }
    _choices, _conditions, expected_selections = recompute_choices(selection_run, split_rows)
    for expected in expected_selections:
        key = (
            expected["run"], expected["condition"], int(expected["seed"]),
            expected["reported_method"],
        )
        row = selections.get(key)
        if row is None:
            raise AssertionError(f"missing validation selection: {key}")
        if row["chosen_method"] != expected["chosen_method"]:
            raise AssertionError(f"validation choice mismatch: {key}")
        if int(row["validation_query_count"]) != expected["validation_query_count"]:
            raise AssertionError(f"validation count mismatch: {key}")
        close(row["validation_mean"], expected["validation_mean"], "validation mean")

    with gzip.open(raw_path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_count += 1
            if row["run"] != run["run"] or row["objective"] != run["objective"]:
                raise AssertionError("raw row identity mismatch")
            if split_rows.get((int(row["seed"]), row["qid"])) != "test":
                raise AssertionError("raw row is not in frozen test partition")
            key = (row["condition"], int(row["seed"]), row["qid"], row["method"])
            if key in raw_keys:
                raise AssertionError(f"duplicate raw contrast key: {key}")
            raw_keys.add(key)
            method_value = float(row["method_value"])
            baseline_value = float(row["baseline_value"])
            difference = method_value - baseline_value
            close(row["paired_difference"], difference, "raw difference")
            close(row["paired_difference_pp"], 100.0 * difference, "raw difference pp")
            slot = aggregate[(row["condition"], row["qid"], row["method"])]
            slot[0] += method_value
            slot[1] += baseline_value
            slot[2] += 1
            slot[3].append(f"{row['seed']}:{row['chosen_method']}")
    if raw_count != run["raw_seed_contrast_rows"]:
        raise AssertionError(f"raw row count mismatch for {run['run']}")

    trajectory = defaultdict(dict)
    averaged_count = 0
    seen_averaged = set()
    with gzip.open(averaged_path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            averaged_count += 1
            key = (row["condition"], row["qid"], row["method"])
            if key in seen_averaged:
                raise AssertionError(f"duplicate averaged key: {key}")
            seen_averaged.add(key)
            if key not in aggregate:
                raise AssertionError(f"averaged row lacks raw rows: {key}")
            method_sum, baseline_sum, count, chosen = aggregate.pop(key)
            method_mean = method_sum / count
            baseline_mean = baseline_sum / count
            difference = method_mean - baseline_mean
            if int(row["n_seed_observations"]) != count:
                raise AssertionError(f"seed observation count mismatch: {key}")
            if row["seed_chosen_methods"] != ";".join(chosen):
                raise AssertionError(f"selected method provenance mismatch: {key}")
            close(row["mean_method_value"], method_mean, "averaged method")
            close(row["mean_baseline_value"], baseline_mean, "averaged baseline")
            close(row["mean_paired_difference"], difference, "averaged difference")
            close(row["mean_paired_difference_pp"], 100.0 * difference, "averaged difference pp")
            trajectory[row["qid"]][(row["condition"], row["method"])] = 100.0 * difference
    if aggregate:
        raise AssertionError(f"raw contrasts lack averaged rows in {run['run']}")
    if averaged_count != run["query_averaged_contrast_rows"]:
        raise AssertionError(f"averaged row count mismatch for {run['run']}")

    qids = sorted(trajectory)
    conditions = run["conditions"]
    level_columns = [(condition, method) for condition in conditions for method in METHOD_ORDER]
    matrix = np.full((len(qids), len(level_columns)), np.nan, dtype=np.float64)
    for q_index, qid in enumerate(qids):
        for c_index, column in enumerate(level_columns):
            if column in trajectory[qid]:
                matrix[q_index, c_index] = trajectory[qid][column]
    interactions = np.full((len(qids), len(METHOD_ORDER)), np.nan, dtype=np.float64)
    for method_index, method in enumerate(METHOD_ORDER):
        clean = matrix[:, level_columns.index((conditions[0], method))]
        heavy = matrix[:, level_columns.index((conditions[-1], method))]
        valid = np.isfinite(clean) & np.isfinite(heavy)
        interactions[valid, method_index] = heavy[valid] - clean[valid]
    combined = np.concatenate([matrix, interactions], axis=1)
    observed = np.nanmean(combined, axis=0)
    seed = int(
        hashlib.sha256(
            f"{metadata['confidence_interval']['seed']}:{run['run']}".encode()
        ).hexdigest()[:16],
        16,
    )
    boots = bootstrap_columns(combined, metadata["confidence_interval"]["resamples"], seed)
    samples = metadata["confidence_interval"]["resamples"]
    expected_rows = []
    for index, (condition, method) in enumerate(level_columns):
        values = matrix[:, index]
        values = values[np.isfinite(values)]
        centered = boots[:, index] - observed[index]
        raw_p = (1 + np.count_nonzero(np.abs(centered) >= abs(observed[index]))) / (samples + 1)
        expected_rows.append(
            (
                (run["run"], "level_effect", condition, "", "", method),
                values, observed[index], np.percentile(boots[:, index], 2.5),
                np.percentile(boots[:, index], 97.5), "", raw_p,
            )
        )
    offset = len(level_columns)
    for method_index, method in enumerate(METHOD_ORDER):
        index = offset + method_index
        values = interactions[:, method_index]
        values = values[np.isfinite(values)]
        centered = boots[:, index] - observed[index]
        raw_p = (1 + np.count_nonzero(centered >= observed[index])) / (samples + 1)
        expected_rows.append(
            (
                (run["run"], "heavy_minus_clean_interaction", "", conditions[0], conditions[-1], method),
                values, observed[index], np.percentile(boots[:, index], 2.5),
                np.percentile(boots[:, index], 97.5), np.percentile(boots[:, index], 5.0), raw_p,
            )
        )

    family_indices = defaultdict(list)
    for index, (key, *_rest) in enumerate(expected_rows):
        family_indices[key[1]].append(index)
    adjusted = {}
    for indices in family_indices.values():
        values = [expected_rows[index][-1] for index in indices]
        for index, value in zip(indices, holm_adjust(values)):
            adjusted[index] = value

    for index, (key, values, mean, lo, hi, lower, raw_p) in enumerate(expected_rows):
        row = summary.get(key)
        if row is None:
            raise AssertionError(f"missing inference row: {key}")
        if int(row["n_independent_queries"]) != len(values):
            raise AssertionError(f"independent-query count mismatch: {key}")
        close(row["mean_paired_difference_pp"], mean, "inference mean")
        close(row["ci95_lo_pp"], lo, "CI lower")
        close(row["ci95_hi_pp"], hi, "CI upper")
        if lower != "":
            close(row["one_sided_95_lower_pp"], lower, "one-sided lower")
        close(row["fraction_improved"], np.mean(values > 0), "fraction improved")
        close(row["fraction_tied"], np.mean(values == 0), "fraction tied")
        close(row["fraction_harmed"], np.mean(values < 0), "fraction harmed")
        close(row["raw_p"], raw_p, "raw p")
        close(row["adjusted_p"], adjusted[index], "Holm p")
    return raw_count, averaged_count, len(expected_rows), len(expected_selections)


def main() -> None:
    with (ARCHIVE / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata["status"] != "addressed_with_archival_limitation":
        raise AssertionError("Statistical archive status must disclose archival limitations")
    if len(metadata["runs"]) != 13:
        raise AssertionError("expected 13 corrected sweep artifacts")
    if metadata["primary_objective"] != "S-Recall@k for every corrected sweep artifact":
        raise AssertionError("incorrect primary objective declaration")
    validate_checksums(metadata)
    for run in metadata["runs"]:
        for source in run["inputs"].values():
            path = ROOT / source["path"]
            if sha256_file(path) != source["sha256"]:
                raise AssertionError(f"source input changed: {source['path']}")

    summary = load_summary()
    selections = load_selections()
    totals = [0, 0, 0, 0]
    for run in metadata["runs"]:
        counts = validate_run(run, metadata, summary, selections)
        totals = [a + b for a, b in zip(totals, counts)]
    expected = metadata["totals"]
    if totals[0] != expected["raw_seed_contrast_rows"]:
        raise AssertionError("total raw contrast count mismatch")
    if totals[1] != expected["query_averaged_contrast_rows"]:
        raise AssertionError("total averaged contrast count mismatch")
    if totals[2] != expected["inference_rows"] or len(summary) != totals[2]:
        raise AssertionError("total inference row count mismatch")
    if totals[3] != expected["validation_selection_rows"] or len(selections) != totals[3]:
        raise AssertionError("total validation selection count mismatch")
    print(
        "Statistical archive validation OK: "
        f"13 sweeps, {totals[0]} raw contrasts, {totals[1]} query-averaged "
        f"contrasts, {totals[2]} inference rows"
    )


if __name__ == "__main__":
    main()
