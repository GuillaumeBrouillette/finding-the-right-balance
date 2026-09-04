#!/usr/bin/env python3
"""Validate the Qwen3.8 extension manifest and retained outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "qwen38"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate() -> None:
    metadata_path = MANIFEST / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["format_version"] == 1
    assert len(metadata["model_revision"]) == 40
    assert len(metadata["runs"]) == 3
    assert metadata["runs"][0]["generation"]["rows"] == 59_240
    assert set(metadata["runs"][0]["generation"]["method_rows"].values()) == {7_405}
    assert metadata["runs"][0]["status"] == "accepted_clean_only_policy_equivalent"
    assert all(
        run["status"] == "legacy_unfixed_candidate_pool_disclosed"
        for run in metadata["runs"][1:]
    )
    for item in metadata["files"]:
        path = ROOT / item["path"]
        assert path.is_file(), path
        assert path.stat().st_size == item["bytes"], path
        assert sha256(path) == item["sha256"], path
    expected = {}
    for line in (MANIFEST / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    for relative, digest in expected.items():
        assert sha256(ROOT / relative) == digest, relative
    print("Qwen manifest validation OK: 3 runs, 59,240 clean RQ1 rows, all hashes")


if __name__ == "__main__":
    validate()
