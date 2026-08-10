#!/usr/bin/env python3
"""Validate pool-size candidate-pool assertions and policy metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "manifests" / "pool_sizes"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate() -> None:
    metadata = json.loads((MANIFEST_DIR / "metadata.json").read_text(encoding="utf-8"))
    proof = metadata["runtime_proof"]
    assert sha256_file(ROOT / proof["implementation"]) == proof["implementation_sha256"]
    assert sha256_file(ROOT / proof["execution_driver"]) == proof["execution_driver_sha256"]
    driver = (ROOT / proof["execution_driver"]).read_text(encoding="utf-8")
    for call in [
        "assert_pool_collection(", "assert_encoded_pool(", "assert_candidate_pool("
    ]:
        assert call in driver
    assert metadata["historical_artifacts"]
    assert metadata["rerun_required"] is False
    for item in metadata["historical_artifacts"]:
        assert sha256_file(ROOT / item["artifact"]) == item["artifact_sha256"]
        assert item["historical_assertion_status"] == "not_recorded"
        assert item["rerun_required_for_pool_size_proof"] is True
    corrected = metadata["corrected_rerun_artifacts"]
    assert len(corrected) == 13
    assert metadata["corrected_rerun_summary"]["source_configurations"] == 12
    assert metadata["corrected_rerun_summary"]["failed_assertions"] == 0
    assert metadata["corrected_rerun_summary"]["audit_rows"] == sum(
        item["row_count"] for item in corrected
    )
    for item in corrected:
        assert sha256_file(ROOT / item["artifact"]) == item["artifact_sha256"]
        assert item["row_count"] > 0
        assert item["levels"]
        assert item["seeds"]
        assert item["all_assertions_pass"] is True
        assert item["all_candidate_sizes_match_targets"] is True
    line = (MANIFEST_DIR / "SHA256SUMS").read_text(encoding="utf-8").strip()
    digest, name = line.split("  ", 1)
    assert sha256_file(MANIFEST_DIR / name) == digest
    print(
        f"OK pool-size policy: {len(metadata['historical_artifacts'])} historical "
        f"and {len(corrected)} corrected artifacts audited"
    )


if __name__ == "__main__":
    validate()
