#!/usr/bin/env python3
"""Record the pool-size candidate-pool policy and historical audit status."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "manifests" / "pool_sizes"
HISTORICAL_RESULTS = ROOT / "results" / "retained"
RERUN_RESULTS = ROOT / "results" / "reproducibility_reruns"
QWEN38_MANIFEST = ROOT / "manifests" / "qwen38" / "metadata.json"


def extension_run_ids() -> set[str]:
    """Runs whose pool policy is assessed in the dedicated Qwen manifest."""
    if not QWEN38_MANIFEST.is_file():
        return set()
    metadata = json.loads(QWEN38_MANIFEST.read_text(encoding="utf-8"))
    return {str(run["run_id"]) for run in metadata.get("runs", [])}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_columns(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle), [])


def audit_corrected_reruns() -> list[dict]:
    rerun_root = RERUN_RESULTS
    if not rerun_root.exists():
        return []
    artifacts = []
    required = {
        "CandidatePoolTarget", "CandidatePoolSize", "PoolSizeAssertion"
    }
    for path in sorted(rerun_root.glob("*/*_pool_size_audit.csv")):
        rows = 0
        levels = set()
        seeds = set()
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} is missing {sorted(missing)}")
            level_key = "overlap" if "chunking" in path.name else "rho"
            for row in reader:
                rows += 1
                levels.add(row[level_key])
                seeds.add(int(row["seed"]))
                if row["PoolSizeAssertion"] != "pass":
                    raise ValueError(f"Failed pool-size assertion in {path}")
                if int(row["CandidatePoolSize"]) != int(row["CandidatePoolTarget"]):
                    raise ValueError(f"Pool target mismatch in {path}")
        if rows == 0:
            raise ValueError(f"Empty pool-size audit: {path}")
        artifacts.append({
            "run_id": path.parent.name,
            "experiment": "chunking" if "chunking" in path.name else "redundancy",
            "artifact": path.relative_to(ROOT).as_posix(),
            "artifact_sha256": sha256_file(path),
            "row_count": rows,
            "levels": sorted(levels, key=float),
            "seeds": sorted(seeds),
            "all_assertions_pass": True,
            "all_candidate_sizes_match_targets": True,
        })
    return artifacts


def create(output: Path = OUTPUT) -> None:
    output.mkdir(parents=True, exist_ok=True)
    audited_runs = []
    excluded_runs = extension_run_ids()
    for run_dir in sorted(HISTORICAL_RESULTS.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name in excluded_runs:
            continue
        artifacts = sorted(
            list(run_dir.glob("results_redundancy_per_query.csv"))
            + list(run_dir.glob("results_chunking_per_query.csv"))
        )
        for artifact in artifacts:
            columns = csv_columns(artifact)
            audited_runs.append({
                "run_id": run_dir.name,
                "artifact": artifact.relative_to(ROOT).as_posix(),
                "artifact_sha256": sha256_file(artifact),
                "level": "overlap" if "chunking" in artifact.name else "rho",
                "recorded_pool_size_columns": [
                    name for name in [
                        "OriginalPoolSize", "TransformedPoolSize",
                        "CandidatePoolTarget", "CandidatePoolSize",
                    ] if name in columns
                ],
                "historical_assertion_status": "not_recorded",
                "rerun_required_for_pool_size_proof": True,
            })

    implementation = ROOT / "ftrb" / "pool_size_invariants.py"
    driver = ROOT / "experiments" / "evaluate_redundancy.py"
    corrected_reruns = audit_corrected_reruns()
    metadata = {
        "format_version": 1,
        "required_invariant": (
            "For each query q, CandidatePoolSize(q, level, seed) == "
            "min(top_m, OriginalPoolSize(q)) at every level and seed."
        ),
        "historical_policy": (
            "min(top_m, TransformedPoolSize), which can grow with rho or overlap"
        ),
        "corrected_policy": "fixed_clean_pool_size_per_query",
        "runtime_proof": {
            "implementation": implementation.relative_to(ROOT).as_posix(),
            "implementation_sha256": sha256_file(implementation),
            "execution_driver": driver.relative_to(ROOT).as_posix(),
            "execution_driver_sha256": sha256_file(driver),
            "audit_output_pattern": "results_<experiment>_pool_size_audit.csv",
            "per_query_columns": [
                "OriginalPoolSize", "TransformedPoolSize",
                "CandidatePoolTarget", "CandidatePoolSize", "PoolSizeAssertion",
            ],
        },
        "historical_artifacts": audited_runs,
        "historical_conclusion": (
            "Existing result files do not contain pool-size proof and were produced "
            "under the dynamic policy. They must not be labelled pool-size compliant."
        ),
        "corrected_rerun_artifacts": corrected_reruns,
        "corrected_rerun_summary": {
            "source_configurations": 12,
            "audit_artifacts": len(corrected_reruns),
            "audit_rows": sum(item["row_count"] for item in corrected_reruns),
            "failed_assertions": 0,
        },
        "rerun_required": not bool(corrected_reruns),
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "SHA256SUMS").write_text(
        f"{sha256_file(metadata_path)}  metadata.json\n", encoding="utf-8"
    )
    print(
        f"Audited {len(audited_runs)} historical artifacts and "
        f"{len(corrected_reruns)} corrected rerun artifacts"
    )


if __name__ == "__main__":
    create()
