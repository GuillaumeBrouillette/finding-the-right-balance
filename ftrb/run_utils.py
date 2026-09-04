"""Shared utilities for run tracking, argument parsing, and CSV output."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime
from importlib import metadata
from typing import Dict, List, Optional


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _runtime_environment() -> Dict:
    """Collect compact software/hardware provenance without requiring CUDA."""
    packages = {}
    for name in ("numpy", "torch", "transformers", "sentence-transformers", "vllm"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    runtime = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }
    try:
        import torch
        runtime["cuda_runtime"] = torch.version.cuda
        runtime["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            runtime["gpu"] = torch.cuda.get_device_name(0)
            runtime["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        runtime["cuda_available"] = False
    return runtime


def make_run_dir(base_dir: str, tag: str, params: Dict) -> str:
    """Create a timestamped run directory and write run_params.json into it.

    The directory name is  <base_dir>/<YYYY-MM-DD_HHMMSS>_<tag>/
    run_params.json contains every entry from *params* plus a timestamp
    and the current git commit hash for full reproducibility.

    Returns the path to the created directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_tag = tag.replace("/", "_").replace(" ", "_")
    run_dir = os.path.join(base_dir, f"{timestamp}_{safe_tag}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "run_params.json"), "w", encoding="utf-8") as f:
        json.dump(
            {**params, "timestamp": timestamp, "git_commit": _git_commit(),
             "runtime_environment": _runtime_environment()},
            f,
            indent=2,
            sort_keys=True,
        )

    print(f"   Run directory: {run_dir}")
    return run_dir


def int_or_all(value: str) -> Optional[int]:
    """Argparse type that accepts a positive integer or the literal 'all' (→ None)."""
    if value.lower() == "all":
        return None
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer or 'all', got {value!r}")
    return n


def save_csv(rows: List[Dict], path: str) -> None:
    """Write a list of row-dicts to a CSV file, creating parent dirs as needed.

    The header is the *union* of keys across all rows (first-seen order), and
    rows missing a column are written blank (``restval=""``).  This makes the
    writer robust to ragged row dicts — e.g. summary tables where the baseline
    row carries no ``p(... vs kNN)`` column but the method rows do — instead of
    raising ``ValueError: dict contains fields not in fieldnames`` after the
    expensive computation has already run.
    """
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Write beside the destination and replace it atomically. Long generation
    # Runs checkpoint this file repeatedly; interruption must not leave a
    # partial CSV that cannot be resumed.
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp",
        dir=os.path.dirname(path) or ".",
    )
    try:
        f = os.fdopen(fd, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
        f.close()
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    print(f"   Saved: {path}")


def attach_run_params(rows: List[Dict], run_params: Dict) -> List[Dict]:
    """Return a copy of rows with a JSON-encoded run-params snapshot appended."""
    if not rows:
        return rows
    params_json = json.dumps(run_params, sort_keys=True, ensure_ascii=True)
    return [{**row, "RunParams": params_json} for row in rows]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

ENCODER_ALIASES: Dict[str, str] = {
    "minilm":         "sentence-transformers/all-MiniLM-L6-v2",
    "bge-m3":         "BAAI/bge-m3",
    "qwen3-embed-4b": "Qwen/Qwen3-Embedding-4B",
}

CE_ALIASES: Dict[str, str] = {
    "minilm-ce":          "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "qwen3-reranker-4b":  "Qwen/Qwen3-Reranker-4B",
}

GENERATOR_ALIASES: Dict[str, str] = {
    "flan-t5-small": "google/flan-t5-small",
    "flan-t5-base":  "google/flan-t5-base",
    "flan-t5-large": "google/flan-t5-large",
    "llama-3.2-3b":  "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.1-8b":  "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3-4b":      "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-8b":      "Qwen/Qwen3-8B",
    "qwen3.8-27b":   "Qwen/Qwen3.8-27B",
}


def resolve_model(name: str, registry: Dict[str, str]) -> str:
    """Resolve a short alias to a full HuggingFace model ID.

    If *name* is not in *registry* it is returned unchanged, so callers can
    freely mix short aliases and explicit HuggingFace paths.

    Examples
    --------
    >>> resolve_model("bge-m3", ENCODER_ALIASES)
    'BAAI/bge-m3'
    >>> resolve_model("BAAI/bge-m3", ENCODER_ALIASES)
    'BAAI/bge-m3'
    """
    return registry.get(name, name)


def default_encoder(device: str) -> str:
    """Return the recommended encoder alias for *device*."""
    return "bge-m3" if device.startswith("cuda") else "minilm"


def default_ce_model(device: str) -> str:
    """Return the recommended cross-encoder alias for *device*."""
    return "bge-reranker-v2-m3" if device.startswith("cuda") else "minilm-ce"


def default_generator(device: str) -> str:
    """Return the recommended generator alias for *device*."""
    return "flan-t5-base" if device.startswith("cuda") else "flan-t5-small"


# ---------------------------------------------------------------------------
# Device normalisation
# ---------------------------------------------------------------------------

def normalize_device(device: str) -> str:
    """Normalise a device string to a torch-compatible form.

    Accepts ``"cpu"``, ``"cuda"``, ``"cuda:N"``, and aliases ``"gpu"`` /
    ``"gpu:N"``.  Warns and falls back to CPU when CUDA is requested but
    unavailable.
    """
    d = str(device).strip().lower()
    if d == "gpu":
        d = "cuda"
    elif d.startswith("gpu:"):
        d = "cuda:" + d.split(":", 1)[1]
    if d.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                print("   Warning: CUDA requested but not available; falling back to cpu.")
                return "cpu"
        except ImportError:
            print("   Warning: torch not found; falling back to cpu.")
            return "cpu"
    return d
