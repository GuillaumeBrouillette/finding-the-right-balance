#!/usr/bin/env python3
"""Validate algorithm-provenance historical algorithms, source links, and random seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "manifests" / "execution"
DEFAULT_RESULTS = ROOT / "results" / "retained"
ARCHIVE_RESULTS_ROOT = Path("results/retained")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_published_path(path: str, results: Path) -> Path:
    logical = Path(path)
    if logical.is_relative_to(ARCHIVE_RESULTS_ROOT):
        return results / logical.relative_to(ARCHIVE_RESULTS_ROOT)
    return ROOT / logical


def validate(results: Path) -> None:
    path = MANIFEST_DIR / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    run_dirs = {
        item.parent.name
        for item in results.glob("*/run_params.json")
    }
    runs = {run["run_id"]: run for run in metadata["runs"]}
    assert set(runs) == run_dirs
    assert len(runs) == 30
    assert sha256_file(ROOT / metadata["source_identity_metadata"]) == (
        metadata["source_identity_metadata_sha256"]
    )
    snapshots = {item["snapshot"]: item for item in metadata["snapshots"]}
    assert snapshots
    for snapshot_path, snapshot in snapshots.items():
        local = ROOT / snapshot_path
        assert sha256_file(local) == snapshot["snapshot_sha256"]
        assert len(snapshot["git_commit"]) == 40
        assert len(snapshot["git_blob"]) == 40

    injection_runs = 0
    chunking_runs = 0
    for run in runs.values():
        assert sha256_file(resolve_published_path(run["run_params"], results)) == (
            run["run_params_sha256"]
        )
        assert run["historical_code_snapshot"] in snapshots
        assert sha256_file(ROOT / run["historical_code_snapshot"]) == (
            run["historical_code_sha256"]
        )
        assert run["seed_evidence"]
        assert run["source_identity_status"]
        assert all(isinstance(seed, int) for seed in run["effective_seeds"])
        families = set(run["artifact_families"])
        source = (ROOT / run["historical_code_snapshot"]).read_text(encoding="utf-8")
        if "injection" in families:
            injection_runs += 1
            assert "def inject_duplicates(" in source
            assert "rho_grid" in run["transformation_parameters"]
            assert run["effective_seeds"]
        if "chunking" in families:
            chunking_runs += 1
            assert "def chunk_pool(" in source
            assert "overlap_grid" in run["transformation_parameters"]
            assert "chunk_window" in run["transformation_parameters"]
            assert run["effective_seeds"]
    assert injection_runs > 0
    assert chunking_runs > 0

    expected = {}
    for line in (MANIFEST_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    for name, digest in expected.items():
        assert sha256_file(MANIFEST_DIR / name) == digest, name
    print(
        f"OK execution manifest: {len(runs)} runs, {len(snapshots)} snapshots, "
        f"{injection_runs} injection runs, {chunking_runs} chunking runs"
    )
    print("OK SHA256SUMS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    arguments = parser.parse_args()
    validate(arguments.results.resolve())
