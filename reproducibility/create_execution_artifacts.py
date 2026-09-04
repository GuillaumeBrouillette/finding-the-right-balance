#!/usr/bin/env python3
"""Build the maximum recoverable execution-artifact archive without model inference."""

from __future__ import annotations

import csv
import gc
import gzip
import hashlib
import io
import json
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "manifests" / "execution_artifacts"
STATE = ROOT / "reproducibility" / "run_state.json"
SOURCE_GROUPS = ROOT / "manifests" / "source_groups"
SPLITS = ROOT / "manifests" / "splits"
RESULT_ROOTS = {
    "historical": ROOT / "results" / "retained",
    "corrected_reruns": ROOT / "results" / "reproducibility_reruns",
}
METRIC_COLUMNS = {
    "Recall@k", "NDCG@k", "MRR", "alpha-NDCG@k", "S-Recall@k",
    "ERR-IA@k", "APD", "Vendi", "EM", "F1", "Halluc",
    "Hallucination Rate",
}
LEVEL_COLUMNS = ("rho", "overlap")
ARTIFACT_OUTPUT_NAMES = (
    "clean_pool_members.csv.gz",
    "clean_pool_checksums.csv.gz",
    "trigger_decisions.csv.gz",
)
QWEN38_MANIFEST = ROOT / "manifests" / "qwen38" / "metadata.json"


def extension_run_ids() -> set[str]:
    """Runs documented separately from the historical execution archive."""
    if not QWEN38_MANIFEST.is_file():
        return set()
    metadata = json.loads(QWEN38_MANIFEST.read_text(encoding="utf-8"))
    return {str(run["run_id"]) for run in metadata.get("runs", [])}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def deterministic_gzip_text(path: Path) -> Iterator[TextIO]:
    """Write deterministic gzip (mtime=0, no host filename)."""
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                yield text


def classify_artifact(path: Path, columns: list[str]) -> str:
    name = path.name
    if name == "run_params.json":
        return "hyperparameters"
    if "pool_size_audit" in name:
        return "pool_size_audit"
    if "gen_per_query" in name or "generation_per_query" in name:
        return "generation_scores_per_query"
    if "retrieval_per_query" in name:
        return "retrieval_metrics_per_query"
    if name.endswith("_per_query.csv"):
        return "metrics_per_query"
    if "gate" in name or "threshold" in name:
        return "trigger_analysis"
    if "summary" in name or "significance" in name:
        return "summary_or_statistics"
    if name.endswith(".csv"):
        return "other_csv"
    if name.endswith(".json"):
        return "other_json"
    if name.endswith(".gz"):
        return "quarantined_or_compressed"
    return "other"


def file_statistics(path: Path) -> dict:
    """Hash a file and count line-oriented CSV rows in one binary pass."""
    digest = hashlib.sha256()
    size = 0
    newline_count = 0
    last_byte = b""
    first_line = b""
    with path.open("rb") as handle:
        first_line = handle.readline()
        if first_line:
            digest.update(first_line)
            size += len(first_line)
            newline_count += first_line.count(b"\n")
            last_byte = first_line[-1:]
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
            size += len(block)
            newline_count += block.count(b"\n")
            last_byte = block[-1:]
    columns: list[str] = []
    row_count = None
    if path.suffix == ".csv":
        header = first_line.decode("utf-8-sig", errors="replace").rstrip("\r\n")
        columns = next(csv.reader([header]), []) if header else []
        total_lines = newline_count + (1 if size and last_byte != b"\n" else 0)
        row_count = max(0, total_lines - 1)
    result = {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
        "logical_type": classify_artifact(path, columns),
    }
    if columns:
        result["columns"] = columns
        result["row_count"] = row_count
        result["field_presence"] = {
            "query_id": next((c for c in ("qid", "Query ID", "query_id") if c in columns), None),
            "seed": "seed" if "seed" in columns else None,
            "level": next((c for c in LEVEL_COLUMNS if c in columns), None),
            "method": "Method" if "Method" in columns else None,
            "metrics": sorted(METRIC_COLUMNS.intersection(columns)),
            "candidate_identity": next((c for c in ("candidate_id", "document_id", "source_variant_id") if c in columns), None),
            "embedding_checksum": next((c for c in ("embedding_sha256", "embedding_checksum") if c in columns), None),
            "rank": next((c for c in ("rank", "Rank", "selected_rank") if c in columns), None),
            "trigger_decision": next((c for c in ("trigger_fired", "gated_fired", "selected_method") if c in columns), None),
            "prediction": next((c for c in ("Prediction", "prediction", "generated_answer") if c in columns), None),
        }
    return result


def inventory_results() -> tuple[list[dict], dict]:
    runs: list[dict] = []
    totals = defaultdict(int)
    excluded_runs = extension_run_ids()
    for collection, result_root in RESULT_ROOTS.items():
        if not result_root.exists():
            continue
        for params_path in sorted(result_root.glob("*/run_params.json")):
            run_dir = params_path.parent
            if collection == "historical" and run_dir.name in excluded_runs:
                continue
            params = json.loads(params_path.read_text(encoding="utf-8"))
            artifacts = []
            for path in sorted(p for p in run_dir.iterdir() if p.is_file()):
                stats = file_statistics(path)
                artifacts.append(stats)
                totals["artifact_count"] += 1
                totals["artifact_bytes"] += stats["size_bytes"]
                if stats.get("row_count") is not None:
                    totals["csv_rows"] += int(stats["row_count"])
            field_records = [a.get("field_presence", {}) for a in artifacts]
            run_generation = bool(params.get("run_generation"))
            has_generation_scores = any(
                a["logical_type"] == "generation_scores_per_query" for a in artifacts
            )
            coverage = {
                "hyperparameters": "retained",
                "per_query_metrics": (
                    "retained" if any(a["logical_type"] in {
                        "metrics_per_query", "retrieval_metrics_per_query"
                    } for a in artifacts) else "not_present_for_this_run"
                ),
                "pool_size_records": (
                    "retained" if any(a["logical_type"] == "pool_size_audit" for a in artifacts)
                    else "not_present_for_this_run"
                ),
                "candidate_identities": (
                    "retained" if any(f.get("candidate_identity") for f in field_records)
                    else "not_serialized"
                ),
                "embedding_checksums": (
                    "retained" if any(f.get("embedding_checksum") for f in field_records)
                    else "not_serialized"
                ),
                "ordered_rankings": (
                    "retained" if any(f.get("rank") and f.get("candidate_identity") for f in field_records)
                    else "not_serialized"
                ),
                "trigger_decisions": "not_derived_for_this_run",
                "generated_answers": (
                    "not_applicable" if not run_generation and not has_generation_scores
                    else "retained" if any(f.get("prediction") for f in field_records)
                    else "score_only_prediction_not_serialized"
                ),
            }
            runs.append({
                "run_id": run_dir.name,
                "collection": collection,
                "dataset": params.get("dataset"),
                "experiment": params.get("experiment"),
                "run_params": params_path.relative_to(ROOT).as_posix(),
                "run_params_sha256": sha256_file(params_path),
                "hyperparameters": params,
                "artifacts": artifacts,
                "coverage": coverage,
            })
            totals["run_count"] += 1
    return runs, dict(totals)


def load_source_registry(dataset: str) -> tuple[dict, dict]:
    path = SOURCE_GROUPS / f"{dataset}.csv.gz"
    by_content: dict[str, list[dict]] = defaultdict(list)
    by_native: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_content[row["content_sha256"]].append(row)
            by_native[row["source_document_id"]].append(row)
    return dict(by_content), dict(by_native)


def clean_pool_configs(state: dict) -> list[dict]:
    configs: dict[tuple, dict] = {}
    for source_run, record in sorted(state["runs"].items()):
        created = record.get("created_run_directories") or []
        if not created:
            continue
        run_dir = Path(created[0])
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        params_path = run_dir / "run_params.json"
        params = json.loads(params_path.read_text(encoding="utf-8"))
        key = (params["dataset"], params["split"], params.get("max_samples"))
        config_id = "__".join([
            str(params["dataset"]), str(params["split"]),
            "all" if params.get("max_samples") is None else str(params["max_samples"]),
        ])
        item = configs.setdefault(key, {
            "pool_config_id": config_id,
            "dataset": params["dataset"],
            "requested_split": params["split"],
            "max_samples": params.get("max_samples"),
            "source_runs": [],
            "corrected_runs": [],
        })
        item["source_runs"].append(source_run)
        item["corrected_runs"].append(run_dir.name)
    return list(configs.values())


def effective_source_split(dataset: str, requested_split: str) -> tuple[str, str]:
    """Return the split actually read, including BEIR's documented fallback."""
    if dataset in {"scifact", "fiqa", "trec-covid"}:
        requested_qrels = ROOT / "data" / "beir" / dataset / "qrels" / f"{requested_split}.tsv"
        if not requested_qrels.exists():
            return "test", f"missing_{requested_split}_qrels_fallback_to_test"
    return requested_split, "requested_split_available"


def resolve_source_member(
    dataset: str,
    title: str,
    text: str,
    by_content: dict,
    by_native: dict,
    native_id_hint: str | None = None,
) -> tuple[dict, str, str]:
    loader_content = sha256_text(f"{title}\0{text}")
    candidates = by_native.get(native_id_hint, []) if native_id_hint else []
    rule = "native_id_from_frozen_query_pool" if candidates else "loader_title_and_text"
    if not candidates:
        candidates = by_content.get(loader_content, [])
    if not candidates and dataset in {"scifact", "fiqa", "trec-covid"}:
        candidates = by_native.get(title, [])
        rule = "beir_native_document_id"
    if not candidates:
        candidates = by_native.get(title, [])
        rule = "native_title_fallback"
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one source identity for {dataset}/{title!r}; got {len(candidates)}"
        )
    return candidates[0], rule, loader_content


def nq_native_id_hints(split: str) -> dict[str, list[str]]:
    """Recover DPR passage IDs that the public loader intentionally drops."""
    import json as json_module
    from data.loaders import _ensure_dpr_file

    path = _ensure_dpr_file("nq", split)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = json_module.load(handle)
    return {
        str(index): [str(context["id"]) for context in row["ctxs"]]
        for index, row in enumerate(rows)
    }


def write_clean_pool_archive(state: dict) -> tuple[list[dict], dict]:
    # Delayed imports keep inventory/validation usable without the ML environment.
    from experiments.evaluate_redundancy import LOADERS

    configs = clean_pool_configs(state)
    members_path = OUTPUT / "clean_pool_members.csv.gz"
    checksums_path = OUTPUT / "clean_pool_checksums.csv.gz"
    member_fields = [
        "pool_config_id", "dataset", "requested_split", "qid",
        "preencoder_position", "source_document_id", "source_group_id",
        "source_variant_id", "source_title_sha256", "source_text_sha256",
        "source_content_sha256", "loader_title_sha256", "loader_text_sha256",
        "loader_content_sha256", "source_match_rule",
    ]
    checksum_fields = [
        "pool_config_id", "dataset", "requested_split", "qid", "member_count",
        "query_key_sha256", "ordered_pool_sha256", "unordered_pool_sha256",
    ]
    stats = {
        "pool_config_count": 0,
        "query_pool_record_count": 0,
        "membership_rows": 0,
    }
    distinct_dataset_queries: set[tuple[str, str]] = set()
    with deterministic_gzip_text(members_path) as member_handle, \
            deterministic_gzip_text(checksums_path) as checksum_handle:
        member_writer = csv.DictWriter(member_handle, fieldnames=member_fields)
        checksum_writer = csv.DictWriter(checksum_handle, fieldnames=checksum_fields)
        member_writer.writeheader()
        checksum_writer.writeheader()
        for config in configs:
            dataset = config["dataset"]
            by_content, by_native = load_source_registry(dataset)
            examples = LOADERS[dataset](
                split=config["requested_split"], max_samples=config["max_samples"]
            )
            examples = [ex for ex in examples if ex["passages"] and ex["gold_titles"]]
            native_hints = (
                nq_native_id_hints(config["requested_split"])
                if dataset == "nq" else {}
            )
            config["query_count"] = len(examples)
            effective_split, split_rule = effective_source_split(
                dataset, config["requested_split"]
            )
            config["effective_source_split"] = effective_split
            config["split_resolution"] = split_rule
            config["source_group_manifest"] = (
                SOURCE_GROUPS / f"{dataset}.csv.gz"
            ).relative_to(ROOT).as_posix()
            config["source_group_manifest_sha256"] = sha256_file(
                SOURCE_GROUPS / f"{dataset}.csv.gz"
            )
            stats["pool_config_count"] += 1
            for example in examples:
                qid = str(example["id"])
                ordered_lines: list[str] = []
                unordered_lines: list[str] = []
                for position, passage in enumerate(example["passages"]):
                    title = str(passage.get("title") or "")
                    text = str(passage.get("text") or "")
                    source, match_rule, loader_content = resolve_source_member(
                        dataset, title, text, by_content, by_native,
                        native_id_hint=(
                            native_hints.get(qid, [])[position]
                            if dataset == "nq" else None
                        ),
                    )
                    loader_title = sha256_text(title)
                    loader_text = sha256_text(text)
                    member_writer.writerow({
                        "pool_config_id": config["pool_config_id"],
                        "dataset": dataset,
                        "requested_split": config["requested_split"],
                        "qid": qid,
                        "preencoder_position": position,
                        "source_document_id": source["source_document_id"],
                        "source_group_id": source["source_group_id"],
                        "source_variant_id": source["source_variant_id"],
                        "source_title_sha256": source["title_sha256"],
                        "source_text_sha256": source["text_sha256"],
                        "source_content_sha256": source["content_sha256"],
                        "loader_title_sha256": loader_title,
                        "loader_text_sha256": loader_text,
                        "loader_content_sha256": loader_content,
                        "source_match_rule": match_rule,
                    })
                    ordered_lines.append(
                        f"{position}\t{source['source_variant_id']}\t{loader_content}\n"
                    )
                    unordered_lines.append(
                        f"{source['source_variant_id']}\t{loader_content}\n"
                    )
                    stats["membership_rows"] += 1
                checksum_writer.writerow({
                    "pool_config_id": config["pool_config_id"],
                    "dataset": dataset,
                    "requested_split": config["requested_split"],
                    "qid": qid,
                    "member_count": len(ordered_lines),
                    "query_key_sha256": sha256_text(
                        f"{dataset}\t{config['requested_split']}\t{qid}\n"
                    ),
                    "ordered_pool_sha256": sha256_text("".join(ordered_lines)),
                    "unordered_pool_sha256": sha256_text(
                        "".join(sorted(unordered_lines))
                    ),
                })
                stats["query_pool_record_count"] += 1
                distinct_dataset_queries.add((dataset, qid))
            del examples, by_content, by_native, native_hints
            gc.collect()
    stats["distinct_dataset_query_keys"] = len(distinct_dataset_queries)
    return configs, stats


def split_partitions(dataset: str) -> dict[tuple[int, str], str]:
    mapping = {}
    with (SPLITS / f"{dataset}.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mapping[(int(row["seed"]), str(row["query_id"]))] = row["partition"]
    return mapping


def corrected_artifacts(state: dict) -> list[dict]:
    result = []
    for source_run, record in sorted(state["runs"].items()):
        created = record.get("created_run_directories") or []
        if not created:
            continue
        run_dir = Path(created[0])
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        params = json.loads((run_dir / "run_params.json").read_text(encoding="utf-8"))
        for path in sorted(run_dir.glob("results_*_per_query.csv")):
            if "oracle" in path.name or "gen_per_query" in path.name:
                continue
            if "redundancy" in path.name:
                experiment, level_col = "redundancy", "rho"
            elif "chunking" in path.name:
                experiment, level_col = "chunking", "overlap"
            else:
                continue
            result.append({
                "source_run": source_run,
                "corrected_run": run_dir.name,
                "dataset": params["dataset"],
                "experiment": experiment,
                "level_col": level_col,
                "path": path,
                "params": params,
            })
    return result


def write_trigger_decisions(state: dict) -> tuple[list[dict], dict]:
    import numpy as np
    import pandas as pd
    from analysis.analyze_regimes import threshold_rule

    output_path = OUTPUT / "trigger_decisions.csv.gz"
    fields = [
        "source_run", "corrected_run", "dataset", "experiment", "level_name",
        "level", "seed", "qid", "partition", "objective", "trigger_metric",
        "trigger_value", "tau", "fallback_method", "trigger_fired",
        "relevant_set_size", "gate_min_rel", "gate_pass", "gated_fired",
        "ungated_selected_method", "gated_selected_method", "knn_objective",
        "fallback_objective", "gated_selected_objective",
    ]
    summaries = []
    totals = {"decision_rows": 0, "validation_rows": 0, "test_rows": 0}
    with deterministic_gzip_text(output_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for artifact in corrected_artifacts(state):
            params = artifact["params"]
            level_col = artifact["level_col"]
            objective = params["objective"]
            df = pd.read_csv(artifact["path"], dtype={"qid": "string"})
            if "seed" not in df.columns:
                df["seed"] = int(params.get("seed", 0))
            df = df[~df["Method"].astype(str).str.match(r"^Seg\(")]
            partitions = split_partitions(artifact["dataset"])
            artifact_rows = 0
            selected_rules = []
            for seed in sorted(int(value) for value in df["seed"].unique()):
                ds = df[df["seed"] == seed]
                val = {
                    qid for (seed_s, qid), partition in partitions.items()
                    if seed_s == seed and partition == "validation"
                }
                test = {
                    qid for (seed_s, qid), partition in partitions.items()
                    if seed_s == seed and partition == "test"
                }
                _, _, info = threshold_rule(
                    ds, level_col, objective, val, test,
                    trigger_metric="Vendi", top_k=int(params["top_k"]),
                )
                tau, fallback = float(info["tau"]), str(info["D"])
                selected_rules.append({"seed": seed, "tau": tau, "fallback_method": fallback})
                knn = (
                    ds[ds["Method"] == "kNN"]
                    .set_index([level_col, "qid"], drop=False)
                )
                divert = (
                    ds[ds["Method"] == fallback]
                    .set_index([level_col, "qid"])[objective]
                )
                if not knn.index.is_unique or not divert.index.is_unique:
                    raise ValueError(f"Duplicate decision keys in {artifact['path']}")
                fallback_values = divert.reindex(knn.index)
                if fallback_values.isna().any():
                    raise ValueError(f"Missing fallback rows in {artifact['path']}")
                for (_, qid), (_, row), fallback_value in zip(
                    knn.index, knn.iterrows(), fallback_values.to_numpy()
                ):
                    qid_s = str(qid)
                    partition = partitions.get((seed, qid_s))
                    if partition not in {"validation", "test"}:
                        raise ValueError(
                            f"No frozen split for {artifact['dataset']} seed={seed} qid={qid_s}"
                        )
                    trigger_value = float(row["Vendi"])
                    rel_size = int(row["RelSetSize"])
                    fired = trigger_value < tau
                    gate_pass = rel_size >= 2
                    gated_fired = fired and gate_pass
                    knn_value = float(row[objective])
                    fallback_float = float(fallback_value)
                    writer.writerow({
                        "source_run": artifact["source_run"],
                        "corrected_run": artifact["corrected_run"],
                        "dataset": artifact["dataset"],
                        "experiment": artifact["experiment"],
                        "level_name": level_col,
                        "level": row[level_col],
                        "seed": seed,
                        "qid": qid_s,
                        "partition": partition,
                        "objective": objective,
                        "trigger_metric": "Vendi",
                        "trigger_value": repr(trigger_value),
                        "tau": repr(tau),
                        "fallback_method": fallback,
                        "trigger_fired": str(fired).lower(),
                        "relevant_set_size": rel_size,
                        "gate_min_rel": 2,
                        "gate_pass": str(gate_pass).lower(),
                        "gated_fired": str(gated_fired).lower(),
                        "ungated_selected_method": fallback if fired else "kNN",
                        "gated_selected_method": fallback if gated_fired else "kNN",
                        "knn_objective": repr(knn_value),
                        "fallback_objective": repr(fallback_float),
                        "gated_selected_objective": repr(
                            fallback_float if gated_fired else knn_value
                        ),
                    })
                    artifact_rows += 1
                    totals["decision_rows"] += 1
                    totals[f"{partition}_rows"] += 1
            summaries.append({
                "source_run": artifact["source_run"],
                "corrected_run": artifact["corrected_run"],
                "dataset": artifact["dataset"],
                "experiment": artifact["experiment"],
                "source_per_query": artifact["path"].relative_to(ROOT).as_posix(),
                "source_per_query_sha256": sha256_file(artifact["path"]),
                "decision_rows": artifact_rows,
                "selected_rules": selected_rules,
            })
            del df
            gc.collect()
    return summaries, totals


def referenced_manifests() -> list[dict]:
    paths = [
        ROOT / "manifests" / "splits" / "metadata.json",
        ROOT / "manifests" / "source_groups" / "metadata.json",
        ROOT / "manifests" / "execution" / "metadata.json",
        ROOT / "manifests" / "models" / "metadata.json",
        ROOT / "manifests" / "pool_sizes" / "metadata.json",
        ROOT / "reproducibility" / "environment-lock.txt",
    ]
    return [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
        for path in paths
    ]


def reconstruction_dependencies() -> dict:
    paths = [
        ROOT / "reproducibility" / "create_execution_artifacts.py",
        ROOT / "experiments" / "evaluate_redundancy.py",
        ROOT / "data" / "loaders.py",
        ROOT / "data" / "beir.py",
        ROOT / "analysis" / "analyze_regimes.py",
        STATE,
    ]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        commit, status = None, "unavailable"
    return {
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
        "repository_commit": commit,
        "worktree_dirty": bool(status.strip()),
        "git_status_sha256": sha256_text(status),
        "provenance_note": (
            "Exact file hashes govern this post-hoc reconstruction because the "
            "working tree contains documented local reproducibility changes."
        ),
    }


def generation_coverage(runs: list[dict]) -> dict:
    artifacts = [
        artifact
        for run in runs
        for artifact in run["artifacts"]
        if "gen" in Path(artifact["path"]).name.lower()
        or "generation" in Path(artifact["path"]).name.lower()
    ]
    keyed = [
        artifact for artifact in artifacts
        if artifact["logical_type"] == "generation_scores_per_query"
        and artifact.get("field_presence", {}).get("query_id")
    ]
    unkeyed_per_query = [
        artifact for artifact in artifacts
        if artifact["logical_type"] == "generation_scores_per_query"
        and not artifact.get("field_presence", {}).get("query_id")
    ]
    return {
        "generation_related_artifact_count": len(artifacts),
        "qid_keyed_score_artifact_count": len(keyed),
        "qid_keyed_score_rows": sum(int(a.get("row_count") or 0) for a in keyed),
        "qid_keyed_score_paths": [a["path"] for a in keyed],
        "unkeyed_per_query_score_artifact_count": len(unkeyed_per_query),
        "unkeyed_per_query_score_paths": [a["path"] for a in unkeyed_per_query],
        "raw_prediction_text_artifact_count": sum(
            1 for artifact in artifacts
            if artifact.get("field_presence", {}).get("prediction")
        ),
        "scope_note": (
            "Only the listed qid-keyed files preserve per-query EM/F1/"
            "hallucination scores; other generation artifacts are summaries or "
            "scores without query identifiers. No raw prediction text survives."
        ),
    }
def create() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    print("[1/3] Inventorying and hashing retained artifacts …", flush=True)
    runs, inventory_totals = inventory_results()
    print("[2/3] Reconstructing clean pre-encoding pool membership …", flush=True)
    pool_configs, pool_totals = write_clean_pool_archive(state)
    print("[3/3] Deriving per-query trigger decisions …", flush=True)
    trigger_summaries, trigger_totals = write_trigger_decisions(state)
    trigger_run_ids = {item["corrected_run"] for item in trigger_summaries}
    source_lineage_ids = {item["source_run"] for item in trigger_summaries}
    for run in runs:
        if run["run_id"] in trigger_run_ids:
            run["coverage"]["trigger_decisions"] = "derived_in_execution_artifacts_archive"
        elif run["run_id"] in source_lineage_ids:
            run["coverage"]["trigger_decisions"] = (
                "not_derived_for_historical_run_corrected_rerun_available"
            )
    output_artifacts = []
    for name in ARTIFACT_OUTPUT_NAMES:
        path = OUTPUT / name
        output_artifacts.append({
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    metadata = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "addressed_with_archival_limitation",
        "scope": "maximum recoverable execution-artifact evidence without model inference",
        "evidence_classes": {
            "retained": "files serialized by the historical or corrected runners",
            "derived": "exact deterministic calculations from retained rows",
            "reconstructed_input": "clean pre-encoding pools rebuilt from frozen sources/loaders",
            "unavailable": "historical runtime state that was never serialized",
        },
        "inventory_totals": inventory_totals,
        "clean_pool_digest_schema": {
            "version": "execution_artifacts-clean-pool-v1",
            "encoding": "UTF-8, LF, no BOM, no additional Unicode normalization",
            "ordered": "SHA256 of position<TAB>source_variant_id<TAB>loader_content_sha256<LF> in loader order",
            "unordered": "SHA256 of repeated source_variant_id<TAB>loader_content_sha256<LF> lines sorted lexicographically",
            "position_semantics": "zero-based loader position; not a dense-model rank",
        },
        "clean_pool_configs": pool_configs,
        "clean_pool_totals": pool_totals,
        "trigger_rule": {
            "trigger": "raw per-query kNN Vendi < validation-tuned tau",
            "threshold_candidates": "rounded validation Vendi values plus lower/upper extremes",
            "fallback": "jointly tuned with tau on each seed's pooled validation levels",
            "gate": "relevant_set_size >= 2, applied after ungated tuning",
            "gate_semantics": (
                "analysis-only oracle/design gate derived from gold titles/qrels; "
                "not a deployable label-free trigger"
            ),
            "split_source": "manifests/splits/<dataset>.csv",
        },
        "trigger_artifacts": trigger_summaries,
        "trigger_totals": trigger_totals,
        "output_artifacts": output_artifacts,
        "referenced_manifests": referenced_manifests(),
        "reconstruction_dependencies": reconstruction_dependencies(),
        "generation_coverage": generation_coverage(runs),
        "runs": runs,
        "field_status": {
            "clean_preencoding_candidate_pools": "reconstructed_and_checksummed",
            "transformed_preencoding_candidate_pools": "reconstructable_from_frozen_code_sources_levels_seeds_and_query_order",
            "hyperparameters": "retained",
            "per_query_metrics": "retained",
            "pool_size_assertions": "retained",
            "trigger_decisions": "derived_exactly",
            "generation_scores": "partially_retained_with_two_qid_keyed_artifacts",
            "embedding_tensors_or_checksums": "unavailable_not_serialized",
            "post_encoder_top_m_membership": "unavailable_not_serialized",
            "ordered_method_rankings": "unavailable_not_serialized",
            "raw_generated_answers": "unavailable_not_serialized",
        },
        "noninvertibility": {
            "rankings": "saved counts and scalar metrics are many-to-one and cannot identify or order selected passages",
            "answers": "EM, F1, and hallucination scores are many-to-one and cannot recover prediction strings",
        },
    }
    metadata_path = OUTPUT / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_paths = [metadata_path] + [OUTPUT / name for name in ARTIFACT_OUTPUT_NAMES]
    (OUTPUT / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths
        ),
        encoding="utf-8",
    )
    print(
        f"Execution artifact archive: {inventory_totals['run_count']} runs, "
        f"{pool_totals['membership_rows']} clean-pool memberships, "
        f"{trigger_totals['decision_rows']} trigger decisions"
    )


if __name__ == "__main__":
    create()
