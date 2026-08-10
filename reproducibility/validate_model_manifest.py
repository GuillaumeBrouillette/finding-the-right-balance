#!/usr/bin/env python3
"""Validate the model-provenance model and inference manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "models"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate() -> None:
    metadata = json.loads((MANIFEST / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["format_version"] == 1
    assert len(metadata["model_registry"]) == 5
    run_keys = set()
    for run in metadata["runs"]:
        key = (run["result_collection"], run["run_id"])
        assert key not in run_keys
        run_keys.add(key)
        assert sha256_file(ROOT / run["run_params"]) == run["run_params_sha256"]
        if run["models"]:
            assert run["inference_settings"]["model_inference"] is True
            assert run["inference_settings"]["batch_size"] > 0
        else:
            assert run["inference_settings"] == {
                "model_inference": False,
                "status": "analysis-only run",
            }
        for model in run["models"]:
            assert len(model["model_revision"]) == 40
            assert len(model["tokenizer_revision"]) == 40
            assert model["model_revision"] == model["tokenizer_revision"]
            assert model["weight_precision"] in {"float32", "bfloat16"}
    expected = {}
    for line in (MANIFEST / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    assert sha256_file(MANIFEST / "metadata.json") == expected["metadata.json"]
    assert sha256_file(ROOT / "reproducibility" / "model_registry.json") == (
        expected["../../reproducibility/model_registry.json"]
    )
    print(
        f"OK model manifest: {len(run_keys)} runs, "
        f"{len(metadata['model_registry'])} immutable model/tokenizer revisions"
    )


if __name__ == "__main__":
    validate()
