#!/usr/bin/env python3
"""Validate frozen query split manifests and their recorded checksums."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "manifests" / "splits"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate() -> None:
    metadata = json.loads((MANIFEST_DIR / "metadata.json").read_text(encoding="utf-8"))
    for dataset in metadata["datasets"]:
        path = ROOT / dataset["manifest"]
        actual_hash = sha256_file(path)
        assert actual_hash == dataset["manifest_sha256"], path
        by_seed: dict[int, dict[str, set[str]]] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == [
                "dataset", "dataset_split", "seed", "query_id", "partition"
            ]
            for row in reader:
                assert row["dataset"] == dataset["dataset"]
                assert row["dataset_split"] == dataset["dataset_split"]
                assert row["partition"] in {"validation", "test"}
                seed = int(row["seed"])
                parts = by_seed.setdefault(seed, {"validation": set(), "test": set()})
                assert row["query_id"] not in parts["validation"] | parts["test"]
                parts[row["partition"]].add(row["query_id"])
        assert sorted(by_seed) == dataset["seeds"]
        for seed, parts in by_seed.items():
            assert parts["validation"].isdisjoint(parts["test"])
            expected = dataset["counts_by_seed"][str(seed)]
            assert len(parts["validation"]) == expected["validation"]
            assert len(parts["test"]) == expected["test"]
            assert len(parts["validation"] | parts["test"]) == dataset["query_count"]
        print(f"OK {dataset['dataset']}: {dataset['query_count']} queries")

    expected_sums = {}
    for line in (MANIFEST_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        expected_sums[name] = digest
    for name, expected in expected_sums.items():
        assert sha256_file(MANIFEST_DIR / name) == expected, name
    print("OK SHA256SUMS")


if __name__ == "__main__":
    validate()
