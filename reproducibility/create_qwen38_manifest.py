#!/usr/bin/env python3
"""Index the Qwen3.8 extension without rerunning model inference."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manifests" / "qwen38"
RQ1 = ROOT / "results" / "retained" / "2026-09-01_191110_redundancy_hotpotqa_fullwiki"
LEGACY = ROOT / "results" / "retained" / "qwen38_legacy_unfixed"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_generation(path: Path) -> dict:
    counts = Counter()
    levels = set()
    seeds = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            counts[row["Method"]] += 1
            levels.add(float(row["rho"]))
            seeds.add(int(row["seed"]))
    return {
        "rows": sum(counts.values()),
        "method_rows": dict(sorted(counts.items())),
        "rho": sorted(levels),
        "seeds": sorted(seeds),
    }


def file_record(path: Path) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def create() -> None:
    rq1_csv = RQ1 / "results_redundancy_gen_per_query.csv"
    legacy_runs = [
        LEGACY / "2026-08-31_163327_redundancy_hotpotqa_fullwiki",
        LEGACY / "2026-08-31_232318_redundancy_2wikimultihopqa",
    ]
    required = [rq1_csv, RQ1 / "run_params.json"]
    for run in legacy_runs:
        required.extend([
            run / "results_redundancy_gen_per_query.csv",
            run / "results_redundancy_per_query.csv",
            run / "analysis_generation.csv",
            run / "run_params.json",
        ])
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Qwen artifacts: " + ", ".join(map(str, missing)))

    rq1_params = json.loads((RQ1 / "run_params.json").read_text(encoding="utf-8"))
    if rq1_params.get("generator_revision") != MODEL_REVISION:
        raise ValueError("Unexpected Qwen model revision")
    rq1 = inspect_generation(rq1_csv)
    if rq1["rows"] != 59_240 or len(rq1["method_rows"]) != 8:
        raise ValueError(f"Incomplete RQ1 generation file: {rq1}")

    all_files = sorted({
        *[path for path in RQ1.rglob("*") if path.is_file()],
        *[path for path in LEGACY.rglob("*") if path.is_file()],
        *[path for path in (OUT / "executed_code").glob("*.py") if path.is_file()],
    })
    metadata = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen/Qwen3.8-27B",
        "model_revision": MODEL_REVISION,
        "reader_protocol": {
            "prompt_version": "short_direct_v1",
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "max_new_tokens": 64,
            "thinking": False,
            "server": "NVIDIA vLLM 26.08",
            "gpu": "NVIDIA GB10",
        },
        "runs": [
            {
                "run_id": RQ1.name,
                "status": "accepted_clean_only_policy_equivalent",
                "reason": "rho=0; old and fixed candidate-pool truncation coincide",
                "generation": rq1,
                "run_params": file_record(RQ1 / "run_params.json"),
                "generation_csv": file_record(rq1_csv),
                "exact_executed_code_hashes_available": True,
            },
            *[
                {
                    "run_id": run.name,
                    "status": "legacy_unfixed_candidate_pool_disclosed",
                    "reason": (
                        "Executed before the official fixed-clean-pool-size correction; "
                        "retained because these are the supplied clean/heavy Qwen results"
                    ),
                    "generation": inspect_generation(
                        run / "results_redundancy_gen_per_query.csv"),
                    "run_params": file_record(run / "run_params.json"),
                    "generation_csv": file_record(
                        run / "results_redundancy_gen_per_query.csv"),
                    "exact_executed_code_hashes_available": False,
                }
                for run in legacy_runs
            ],
        ],
        "paper_reconstruction": {
            "table_02": "accepted clean-pool RQ1 run",
            "table_11": "legacy clean/heavy runs, with protocol limitation disclosed",
            "model_execution": False,
        },
        "files": [file_record(path) for path in all_files],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    metadata_path = OUT / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    indexed = all_files + [metadata_path]
    (OUT / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in indexed),
        encoding="utf-8",
    )
    print(f"Indexed {len(all_files)} Qwen files across three completed runs")


if __name__ == "__main__":
    create()
