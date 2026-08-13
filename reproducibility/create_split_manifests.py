#!/usr/bin/env python3
"""Recover and freeze the exact validation/test query assignments.

The experiment scripts created the split by permuting the loader's query order
with ``numpy.random.default_rng(seed)``.  The large per-query result files retain
that order at every sweep level.  This script recovers the order from the clean
level's kNN rows, applies the original permutation, and writes small, committed
CSV manifests plus provenance checksums.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "reproducibility" / "split_sources.json"
DEFAULT_OUTPUT = ROOT / "manifests" / "splits"
DEFAULT_EXPERIMENT_DATA = ROOT / "data"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _clean_level(rows: csv.DictReader, level_column: str) -> str:
    values = []
    for row in rows:
        raw = row[level_column]
        try:
            value = float(raw)
        except ValueError:
            continue
        values.append((value, raw))
    if not values:
        raise ValueError(f"No numeric values found in {level_column!r}")
    return min(values)[1]


def query_order_by_seed(path: Path, level_column: str) -> Dict[int, List[str]]:
    """Return query order from the clean-level kNN rows of a result CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        required = {level_column, "qid", "Method"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        # All recovered sweeps use 0 as their clean injection/overlap level.
        clean = "0.0"
        orders: Dict[int, List[str]] = {}
        seen: Dict[int, set[str]] = {}
        has_seed = "seed" in reader.fieldnames
        for row in reader:
            try:
                is_clean = float(row[level_column]) == float(clean)
            except ValueError:
                is_clean = False
            if not is_clean or row["Method"] != "kNN":
                continue
            seed = int(row["seed"]) if has_seed and row["seed"] != "" else 0
            qid = row["qid"]
            seed_seen = seen.setdefault(seed, set())
            if qid in seed_seen:
                raise ValueError(
                    f"Duplicate clean-level kNN qid {qid!r} for seed {seed} in {path}"
                )
            seed_seen.add(qid)
            orders.setdefault(seed, []).append(qid)
    if not orders:
        raise ValueError(f"No clean-level kNN rows found in {path}")
    return orders


def validate_orders(orders: Mapping[int, Sequence[str]], path: Path) -> List[str]:
    """Require every seed to contain the same unique queries in the same order."""
    first_seed = min(orders)
    canonical = list(orders[first_seed])
    if len(canonical) != len(set(canonical)):
        raise ValueError(f"Non-unique query IDs in {path}, seed {first_seed}")
    for seed, order in sorted(orders.items()):
        if list(order) != canonical:
            raise ValueError(
                f"Query order differs between seeds {first_seed} and {seed} in {path}"
            )
    return canonical


def partition(order: Sequence[str], seed: int, val_fraction: float) -> Dict[str, str]:
    n_val = max(1, int(val_fraction * len(order)))
    permutation = np.random.default_rng(seed).permutation(len(order))
    val_indices = {int(index) for index in permutation[:n_val]}
    return {
        qid: ("validation" if index in val_indices else "test")
        for index, qid in enumerate(order)
    }


def write_manifest(
    output: Path,
    dataset: str,
    dataset_split: str,
    order: Sequence[str],
    seeds: Sequence[int],
    val_fraction: float,
) -> Dict[str, Dict[str, int]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, Dict[str, int]] = {}
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "dataset_split", "seed", "query_id", "partition"],
            lineterminator="\n",
        )
        writer.writeheader()
        for seed in seeds:
            assignments = partition(order, seed, val_fraction)
            seed_counts = {"validation": 0, "test": 0}
            for qid in order:
                split = assignments[qid]
                seed_counts[split] += 1
                writer.writerow(
                    {
                        "dataset": dataset,
                        "dataset_split": dataset_split,
                        "seed": seed,
                        "query_id": qid,
                        "partition": split,
                    }
                )
            counts[str(seed)] = seed_counts
    return counts


def verify_test_summary(
    source: Path,
    level_column: str,
    objective: str,
    assignments_by_seed: Mapping[int, Mapping[str, str]],
) -> Dict[str, object]:
    """Confirm reconstructed test membership reproduces the saved kNN mean."""
    per_seed_summary = source.with_name(source.name.replace(
        "_per_query.csv", "_per_seed_summary.csv"))
    summary = per_seed_summary if per_seed_summary.is_file() else source.with_name(
        source.name.replace("_per_query.csv", "_summary.csv"))
    if not summary.is_file():
        raise FileNotFoundError(f"No summary corresponding to {source}")

    values: Dict[int, List[float]] = {seed: [] for seed in assignments_by_seed}
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        has_seed = "seed" in (reader.fieldnames or [])
        for row in reader:
            try:
                is_clean = float(row[level_column]) == 0.0
            except ValueError:
                is_clean = False
            if not is_clean or row["Method"] != "kNN":
                continue
            seed = int(row["seed"]) if has_seed and row["seed"] else 0
            if assignments_by_seed[seed][row["qid"]] != "test":
                continue
            value = float(row[objective])
            if math.isfinite(value):
                values[seed].append(value)

    expected: Dict[int, float] = {}
    with summary.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        has_seed = "seed" in (reader.fieldnames or [])
        for row in reader:
            try:
                is_clean = float(row[level_column]) == 0.0
            except (KeyError, ValueError):
                is_clean = False
            if not is_clean or row.get("Method") != "kNN":
                continue
            seed = int(row["seed"]) if has_seed and row.get("seed") else 0
            expected[seed] = float(row[objective])

    evidence: Dict[str, object] = {
        "summary": summary.relative_to(ROOT).as_posix(),
        "objective": objective,
        "by_seed": {},
    }
    for seed, seed_values in values.items():
        if seed not in expected or not seed_values:
            raise ValueError(f"Cannot validate seed {seed} against {summary}")
        recovered = float(np.mean(seed_values))
        # Saved summaries are rounded to four decimal places.
        if abs(recovered - expected[seed]) > 5.1e-5:
            raise ValueError(
                f"Split mismatch for {source}, seed {seed}: recovered "
                f"{recovered:.8f}, saved {expected[seed]:.8f}"
            )
        evidence["by_seed"][str(seed)] = {
            "n_test": len(seed_values),
            "recovered_mean": recovered,
            "saved_rounded_mean": expected[seed],
        }
    return evidence


def _read_jsonl_ids(path: Path) -> set[str]:
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(str(json.loads(line)["_id"]))
    return ids


def beir_query_order(task_dir: Path) -> List[str]:
    """Historical loader order: first positive qrel occurrence per query."""
    queries = _read_jsonl_ids(task_dir / "queries.jsonl")
    order = []
    seen = set()
    with (task_dir / "qrels" / "test.tsv").open(encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or int(float(parts[2])) <= 0:
                continue
            qid = parts[0]
            if qid in queries and qid not in seen:
                seen.add(qid)
                order.append(qid)
    return order


def create(config_path: Path, output_dir: Path, experiment_data: Path) -> None:
    entries = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = {
        "format_version": 1,
        "split_algorithm": "numpy.random.default_rng(seed).permutation(n)",
        "numpy_version_used_for_recovery": np.__version__,
        "datasets": [],
        "source_group_manifests": "manifests/source_groups/metadata.json",
        "known_limitations": [
            "The split and source-group manifests do not claim to contain the "
            "historical post-encoding per-query candidate rankings; those are "
            "reported separately as execution-artifact artifacts."
        ],
    }
    checksums: List[tuple[str, str]] = []

    for entry in entries:
        dataset = entry["dataset"]
        params_path = ROOT / entry["run_params"]
        if not params_path.is_file():
            raise FileNotFoundError(f"Missing recovered source for {dataset}")
        params = json.loads(params_path.read_text(encoding="utf-8"))
        if entry.get("source_type") == "beir_qrels":
            task_dir = experiment_data / "beir" / entry["beir_task"]
            source = task_dir / "qrels" / "test.tsv"
            queries_path = task_dir / "queries.jsonl"
            if not source.is_file() or not queries_path.is_file():
                raise FileNotFoundError(f"Missing BEIR source for {dataset}")
            order = beir_query_order(task_dir)
            seeds = [0]
            val_fraction = 0.2
            manifest_path = output_dir / f"{dataset}.csv"
            counts = write_manifest(
                manifest_path, dataset, entry["dataset_split"], order,
                seeds, val_fraction,
            )
            manifest_hash = sha256_file(manifest_path)
            checksums.append((manifest_hash, manifest_path.name))
            metadata["datasets"].append({
                "dataset": dataset,
                "dataset_split": entry["dataset_split"],
                "query_count": len(order),
                "seeds": seeds,
                "validation_fraction": val_fraction,
                "counts_by_seed": counts,
                "query_order_sha256": sha256_lines(order),
                "manifest": manifest_path.relative_to(ROOT).as_posix(),
                "manifest_sha256": manifest_hash,
                "source_qrels": source.name,
                "source_qrels_sha256": sha256_file(source),
                "source_queries": queries_path.name,
                "source_queries_sha256": sha256_file(queries_path),
                "source_run_params": params_path.relative_to(ROOT).as_posix(),
                "source_run_params_sha256": sha256_file(params_path),
                "historical_git_commit": params.get("git_commit", "unknown"),
                "saved_summary_validation": {
                    "status": "not_available",
                    "reason": "The early BEIR run retained summary rows but no per-query rows."
                },
                "verified_same_query_order": [],
            })
            print(f"{dataset}: {len(order)} queries, seeds={seeds}, counts={counts}")
            continue

        source = ROOT / entry["per_query"]
        if not source.is_file():
            raise FileNotFoundError(f"Missing recovered source for {dataset}")
        orders = query_order_by_seed(source, entry["level_column"])
        order = validate_orders(orders, source)
        configured_seeds = params.get("seeds") or [params.get("seed", 0)]
        seeds = [int(seed) for seed in configured_seeds]
        if set(orders) not in ({0}, set(seeds)):
            raise ValueError(
                f"Recovered seeds {sorted(orders)} do not match configured seeds {seeds}"
            )
        if set(orders) == {0} and len(seeds) > 1:
            raise ValueError(f"{source} has no per-seed rows for configured seeds {seeds}")

        verified = []
        for raw_path in entry.get("verification_files", []):
            other = ROOT / raw_path
            other_level = "overlap" if "chunking" in other.name else "rho"
            other_order = validate_orders(query_order_by_seed(other, other_level), other)
            if other_order != order:
                raise ValueError(f"Query order mismatch: {source} vs {other}")
            verified.append(
                {
                    "path": other.relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(other),
                }
            )

        manifest_path = output_dir / f"{dataset}.csv"
        val_fraction = float(params.get("val_fraction", 0.2))
        assignments_by_seed = {
            seed: partition(order, seed, val_fraction)
            for seed in seeds
        }
        counts = write_manifest(
            manifest_path,
            dataset,
            entry["dataset_split"],
            order,
            seeds,
            val_fraction,
        )
        summary_validation = verify_test_summary(
            source,
            entry["level_column"],
            params["objective"],
            assignments_by_seed,
        )
        manifest_hash = sha256_file(manifest_path)
        checksums.append((manifest_hash, manifest_path.name))
        metadata["datasets"].append(
            {
                "dataset": dataset,
                "dataset_split": entry["dataset_split"],
                "query_count": len(order),
                "seeds": seeds,
                "validation_fraction": val_fraction,
                "counts_by_seed": counts,
                "query_order_sha256": sha256_lines(order),
                "manifest": manifest_path.relative_to(ROOT).as_posix(),
                "manifest_sha256": manifest_hash,
                "source_per_query": source.relative_to(ROOT).as_posix(),
                "source_per_query_sha256": sha256_file(source),
                "source_run_params": params_path.relative_to(ROOT).as_posix(),
                "source_run_params_sha256": sha256_file(params_path),
                "historical_git_commit": params.get("git_commit", "unknown"),
                "saved_summary_validation": summary_validation,
                "verified_same_query_order": verified,
            }
        )
        print(f"{dataset}: {len(order)} queries, seeds={seeds}, counts={counts}")

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums.append((sha256_file(metadata_path), metadata_path.name))
    checksum_path = output_dir / "SHA256SUMS"
    checksum_path.write_text(
        "".join(f"{digest}  {name}\n" for digest, name in sorted(checksums, key=lambda x: x[1])),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--experiment-data-root", type=Path, default=DEFAULT_EXPERIMENT_DATA
    )
    args = parser.parse_args()
    create(
        args.config.resolve(), args.output_dir.resolve(),
        args.experiment_data_root.resolve(),
    )


if __name__ == "__main__":
    main()
