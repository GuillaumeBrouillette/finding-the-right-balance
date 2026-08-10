#!/usr/bin/env python3
"""Validate frozen source-document group manifests and checksums."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "manifests" / "source_groups"
REQUIRED_DATASETS = {
    "hotpotqa_fullwiki", "nq", "2wikimultihopqa", "musique",
    "scifact", "fiqa", "trec-covid",
    "arguana", "webis-touche2020",
}
FIELDS = [
    "dataset", "dataset_split", "source_document_id", "source_group_id",
    "source_variant_id", "title_sha256", "text_sha256", "content_sha256",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate() -> None:
    metadata = json.loads((MANIFEST_DIR / "metadata.json").read_text(encoding="utf-8"))
    entries = {entry["dataset"]: entry for entry in metadata["datasets"]}
    assert set(entries) == REQUIRED_DATASETS
    for dataset, entry in sorted(entries.items()):
        path = ROOT / entry["manifest"]
        assert sha256_file(path) == entry["manifest_sha256"]
        rows = 0
        groups: set[str] = set()
        variants: set[str] = set()
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == FIELDS
            for row in reader:
                assert row["dataset"] == dataset
                assert len(row["title_sha256"]) == 64
                assert len(row["text_sha256"]) == 64
                assert len(row["content_sha256"]) == 64
                assert row["source_variant_id"].endswith(row["content_sha256"])
                assert row["source_variant_id"] not in variants
                variants.add(row["source_variant_id"])
                groups.add(row["source_group_id"])
                rows += 1
        assert rows == entry["source_variant_count"]
        assert len(groups) == entry["source_group_count"]
        print(f"OK {dataset}: {len(groups)} groups, {rows} variants")

    expected = {}
    for line in (MANIFEST_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    for name, digest in expected.items():
        assert sha256_file(MANIFEST_DIR / name) == digest, name
    print("OK SHA256SUMS")


if __name__ == "__main__":
    validate()
