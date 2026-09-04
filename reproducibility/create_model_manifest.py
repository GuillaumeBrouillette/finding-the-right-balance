#!/usr/bin/env python3
"""Create the model-provenance model, tokenizer, precision, and inference manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "reproducibility" / "model_registry.json"
OUTPUT = ROOT / "manifests" / "models"
RESULT_ROOTS = (
    ROOT / "results" / "retained",
    ROOT / "results" / "reproducibility_reruns",
)
ALIASES = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}
ROLE_KEYS = {
    "encoder": "encoder_model",
    "cross_encoder": "ce_model",
    "generator": "generator_model",
}
PACKAGE_VERSIONS = {
    "torch": "2.12.0",
    "transformers": "5.9.0",
    "sentence-transformers": "5.5.1",
    "tokenizers": "0.22.2",
    "numpy": "2.4.6",
    "faiss-cpu": "1.14.2",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_timestamp(run_id: str) -> datetime:
    return datetime.strptime(run_id[:15], "%Y-%m-%d_%H%M%S").replace(
        tzinfo=timezone.utc
    )


def model_record(role: str, raw_name: str, registry: dict) -> dict:
    name = ALIASES.get(raw_name, raw_name)
    if name not in registry:
        raise ValueError(f"No immutable model revision registered for {raw_name!r}")
    spec = registry[name]
    return {
        "role": role,
        "name_as_recorded": raw_name,
        "canonical_name": name,
        **spec,
    }


def inference_settings(params: dict, models: list[dict]) -> dict:
    if not models:
        return {"model_inference": False, "status": "analysis-only run"}
    roles = {model["role"] for model in models}
    settings = {
        "model_inference": True,
        "device": params.get("device", "cpu"),
        "batch_size": params.get("batch_size", 64),
        "evaluation_mode": True,
        "autograd": "disabled for embedding/generation inference",
        "automatic_mixed_precision": False,
    }
    if "encoder" in roles:
        settings["encoder"] = {
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "stored_embedding_precision": "float32",
            "similarity_metric": params.get("metric", "cosine"),
            "candidate_depth": params.get("top_m"),
            "output_depth": params.get("top_k"),
        }
    if "cross_encoder" in roles:
        settings["cross_encoder"] = {
            "max_length_tokens": 512,
            "raw_score_storage_precision": "float32",
            "score_transform": "sigmoid",
        }
    if "generator" in roles:
        settings["generator"] = {
            "max_input_tokens": 512,
            "max_new_tokens": params.get(
                "generator_max_new_tokens", params.get("max_new_tokens", 64)),
            "num_beams": params.get(
                "generator_num_beams", params.get("num_beams", 4)),
            "do_sample": False,
            "early_stopping": True,
            "tokenizer_truncation": True,
            "skip_special_tokens": True,
        }
        if params.get("generator_backend") == "openai-compatible":
            settings["generator"].update({
                "backend": "openai-compatible",
                "model_revision": params.get("generator_revision"),
                "temperature": params.get("generator_temperature"),
                "top_p": params.get("generator_top_p"),
                "seed": params.get("generator_seed"),
                "thinking": params.get("generator_thinking"),
                "prompt_version": params.get("generator_prompt_version"),
                "concurrent_requests": params.get("generator_batch_size"),
            })
    return settings


def create() -> None:
    registry_document = json.loads(REGISTRY.read_text(encoding="utf-8"))
    registry = registry_document["models"]
    runs = []
    for result_root in RESULT_ROOTS:
        if not result_root.exists():
            continue
        for params_path in sorted(result_root.glob("*/run_params.json")):
            params = json.loads(params_path.read_text(encoding="utf-8"))
            models = []
            for role, key in ROLE_KEYS.items():
                value = params.get(key)
                if value:
                    models.append(model_record(role, str(value), registry))
            timestamp = run_timestamp(params_path.parent.name)
            for model in models:
                modified_raw = model.get("upstream_last_modified")
                modified = datetime.fromisoformat(modified_raw) if modified_raw else None
                if modified is not None and modified > timestamp:
                    raise ValueError(
                        f"{model['canonical_name']} revision postdates "
                        f"{params_path.parent.name}"
                    )
            protocol_status = "historical"
            if params.get("generator_model") == "Qwen/Qwen3.8-27B":
                if params.get("candidate_pool_policy") == "fixed_clean_pool_size_per_query":
                    protocol_status = "fixed_clean_pool_size"
                elif set(params.get("rho_grid", [])) == {0.0}:
                    protocol_status = "clean_only_policy_equivalent"
                else:
                    protocol_status = "legacy_unfixed_candidate_pool_excluded"
            runs.append({
                "run_id": params_path.parent.name,
                "result_collection": result_root.name,
                "run_params": params_path.relative_to(ROOT).as_posix(),
                "run_params_sha256": sha256_file(params_path),
                "models": models,
                "inference_settings": inference_settings(params, models),
                "protocol_status": protocol_status,
            })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "All archived run parameter files and all pool-size reruns present "
            "when this manifest was generated"
        ),
        "revision_rule": (
            "Model and tokenizer were loaded from the same Hugging Face model "
            "identifier without separate revision arguments. The registered "
            "immutable repository SHA therefore identifies both. Every upstream "
            "revision predates its associated run."
        ),
        "runtime_packages": PACKAGE_VERSIONS,
        "model_registry": registry,
        "runs": runs,
    }
    metadata_path = OUTPUT / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums = [
        f"{sha256_file(metadata_path)}  metadata.json\n",
        f"{sha256_file(REGISTRY)}  ../../reproducibility/model_registry.json\n",
    ]
    (OUTPUT / "SHA256SUMS").write_text("".join(checksums), encoding="utf-8")
    print(f"Frozen model metadata for {len(runs)} runs and {len(registry)} models")


if __name__ == "__main__":
    create()
