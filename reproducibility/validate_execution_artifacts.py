#!/usr/bin/env python3
"""Validate the no-inference Execution-artifact reproducibility archive."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "manifests" / "execution_artifacts"
HEX = frozenset("0123456789abcdef")
EXPECTED_ARCHIVE_FILES = {
    "metadata.json",
    "clean_pool_members.csv.gz",
    "clean_pool_checksums.csv.gz",
    "trigger_decisions.csv.gz",
}
MEMBER_FIELDS = [
    "pool_config_id", "dataset", "requested_split", "qid",
    "preencoder_position", "source_document_id", "source_group_id",
    "source_variant_id", "source_title_sha256", "source_text_sha256",
    "source_content_sha256", "loader_title_sha256", "loader_text_sha256",
    "loader_content_sha256", "source_match_rule",
]
CHECKSUM_FIELDS = [
    "pool_config_id", "dataset", "requested_split", "qid", "member_count",
    "query_key_sha256", "ordered_pool_sha256", "unordered_pool_sha256",
]
TRIGGER_FIELDS = [
    "source_run", "corrected_run", "dataset", "experiment", "level_name",
    "level", "seed", "qid", "partition", "objective", "trigger_metric",
    "trigger_value", "tau", "fallback_method", "trigger_fired",
    "relevant_set_size", "gate_min_rel", "gate_pass", "gated_fired",
    "ungated_selected_method", "gated_selected_method", "knn_objective",
    "fallback_objective", "gated_selected_objective",
]


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def valid_sha256(value: str) -> bool:
    return len(value) == 64 and set(value) <= HEX


def parse_bool(value: str, context: str) -> bool:
    require(value in {"true", "false"}, f"Invalid Boolean {value!r} in {context}")
    return value == "true"


def parse_finite(value: str, context: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValidationError(f"Invalid float {value!r} in {context}") from exc
    require(math.isfinite(result), f"Nonfinite float in {context}")
    return result


def validate_sha256sums() -> None:
    entries: dict[str, str] = {}
    checksum_file = ARCHIVE / "SHA256SUMS"
    require(checksum_file.is_file(), "Missing manifests/execution_artifacts/SHA256SUMS")
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        parts = line.split("  ", 1)
        require(len(parts) == 2, f"Malformed SHA256SUMS line: {line!r}")
        digest, name = parts
        require(valid_sha256(digest), f"Invalid digest for {name}")
        require(name not in entries, f"Duplicate SHA256SUMS entry: {name}")
        entries[name] = digest
    require(set(entries) == EXPECTED_ARCHIVE_FILES, "Unexpected SHA256SUMS file set")
    for name, expected in entries.items():
        path = ARCHIVE / name
        require(path.is_file(), f"Missing archive file: {name}")
        require(sha256_file(path) == expected, f"Archive checksum mismatch: {name}")


def file_stats(path: Path) -> tuple[int, str, list[str] | None, int | None]:
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
    columns = None
    row_count = None
    if path.suffix == ".csv":
        header = first_line.decode("utf-8-sig", errors="strict").rstrip("\r\n")
        columns = next(csv.reader([header]), []) if header else []
        total_lines = newline_count + (1 if size and last_byte != b"\n" else 0)
        row_count = max(0, total_lines - 1)
    return size, digest.hexdigest(), columns, row_count


def validate_reference_hashes(metadata: dict) -> None:
    for section in ("referenced_manifests",):
        for record in metadata[section]:
            path = ROOT / record["path"]
            require(path.is_file(), f"Missing referenced file: {record['path']}")
            require(sha256_file(path) == record["sha256"],
                    f"Referenced hash mismatch: {record['path']}")
    dependencies = metadata["reconstruction_dependencies"]["files"]
    required_dependencies = {
        "reproducibility/create_execution_artifacts.py",
        "evaluate_redundancy.py",
        "data/loaders.py",
        "data/beir.py",
        "analyze_regimes.py",
        "reproducibility/run_state.json",
    }
    require({item["path"] for item in dependencies} == required_dependencies,
            "Reconstruction dependency set is incomplete")
    for record in dependencies:
        path = ROOT / record["path"]
        require(path.is_file(), f"Missing reconstruction dependency: {record['path']}")
        require(path.stat().st_size == record["size_bytes"],
                f"Dependency size mismatch: {record['path']}")
        require(sha256_file(path) == record["sha256"],
                f"Dependency hash mismatch: {record['path']}")


def validate_inventory(metadata: dict) -> dict[str, dict]:
    runs = metadata["runs"]
    require(len(runs) == 42, f"Expected 42 runs, found {len(runs)}")
    require(len({(r["collection"], r["run_id"]) for r in runs}) == len(runs),
            "Duplicate run inventory key")
    totals = Counter()
    run_by_id: dict[str, dict] = {}
    for run in runs:
        run_id = run["run_id"]
        require(run_id not in run_by_id, f"Run ID occurs in multiple collections: {run_id}")
        run_by_id[run_id] = run
        params_path = ROOT / run["run_params"]
        require(params_path.is_file(), f"Missing run parameters: {run['run_params']}")
        require(sha256_file(params_path) == run["run_params_sha256"],
                f"Run-parameter hash mismatch: {run_id}")
        require(json.loads(params_path.read_text(encoding="utf-8")) == run["hyperparameters"],
                f"Run-parameter content mismatch: {run_id}")
        actual_files = {
            path.relative_to(ROOT).as_posix()
            for path in params_path.parent.iterdir() if path.is_file()
        }
        declared_files = {artifact["path"] for artifact in run["artifacts"]}
        require(actual_files == declared_files, f"Artifact set mismatch: {run_id}")
        for artifact in run["artifacts"]:
            path = ROOT / artifact["path"]
            size, digest, columns, row_count = file_stats(path)
            require(size == artifact["size_bytes"], f"Artifact size mismatch: {artifact['path']}")
            require(digest == artifact["sha256"], f"Artifact hash mismatch: {artifact['path']}")
            if columns is not None:
                require(columns == artifact.get("columns"),
                        f"CSV schema mismatch: {artifact['path']}")
                require(row_count == artifact.get("row_count"),
                        f"CSV row-count mismatch: {artifact['path']}")
                totals["csv_rows"] += int(row_count or 0)
            totals["artifact_count"] += 1
            totals["artifact_bytes"] += size
        totals["run_count"] += 1
    require(dict(totals) == metadata["inventory_totals"], "Inventory totals mismatch")

    trigger_runs = {item["corrected_run"] for item in metadata["trigger_artifacts"]}
    lineage_runs = {item["source_run"] for item in metadata["trigger_artifacts"]}
    for run in runs:
        observed = run["coverage"]["trigger_decisions"]
        if run["run_id"] in trigger_runs:
            expected = "derived_in_execution_artifacts_archive"
        elif run["run_id"] in lineage_runs:
            expected = "not_derived_for_historical_run_corrected_rerun_available"
        else:
            expected = "not_derived_for_this_run"
        require(observed == expected, f"Incorrect trigger coverage: {run['run_id']}")
    return run_by_id


def validate_output_artifacts(metadata: dict) -> None:
    expected_paths = {
        "manifests/execution_artifacts/clean_pool_members.csv.gz",
        "manifests/execution_artifacts/clean_pool_checksums.csv.gz",
        "manifests/execution_artifacts/trigger_decisions.csv.gz",
    }
    records = metadata["output_artifacts"]
    require({r["path"] for r in records} == expected_paths,
            "Unexpected Execution-artifact output artifact set")
    for record in records:
        path = ROOT / record["path"]
        require(path.stat().st_size == record["size_bytes"],
                f"Output size mismatch: {record['path']}")
        require(sha256_file(path) == record["sha256"],
                f"Output hash mismatch: {record['path']}")


def load_pool_checksums(metadata: dict) -> dict[tuple[str, str], dict]:
    configs = {item["pool_config_id"]: item for item in metadata["clean_pool_configs"]}
    require(len(configs) == metadata["clean_pool_totals"]["pool_config_count"],
            "Pool-config count mismatch")
    for config in configs.values():
        require(config["effective_source_split"], "Missing effective source split")
        source_path = ROOT / config["source_group_manifest"]
        require(sha256_file(source_path) == config["source_group_manifest_sha256"],
                f"Source-group hash mismatch: {config['pool_config_id']}")

    result: dict[tuple[str, str], dict] = {}
    per_config = Counter()
    path = ARCHIVE / "clean_pool_checksums.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == CHECKSUM_FIELDS, "Clean-pool checksum schema mismatch")
        for line_number, row in enumerate(reader, 2):
            key = (row["pool_config_id"], row["qid"])
            require(key not in result, f"Duplicate pool checksum key at line {line_number}")
            config = configs.get(row["pool_config_id"])
            require(config is not None, f"Unknown pool config at line {line_number}")
            require(row["dataset"] == config["dataset"], f"Pool dataset mismatch at line {line_number}")
            require(row["requested_split"] == config["requested_split"],
                    f"Pool split mismatch at line {line_number}")
            count = int(row["member_count"])
            require(count > 0, f"Empty pool at line {line_number}")
            expected_query_hash = sha256_text(
                f"{row['dataset']}\t{row['requested_split']}\t{row['qid']}\n"
            )
            require(row["query_key_sha256"] == expected_query_hash,
                    f"Query-key hash mismatch at line {line_number}")
            require(valid_sha256(row["ordered_pool_sha256"]),
                    f"Invalid ordered pool hash at line {line_number}")
            require(valid_sha256(row["unordered_pool_sha256"]),
                    f"Invalid unordered pool hash at line {line_number}")
            result[key] = row
            per_config[row["pool_config_id"]] += 1
    expected_records = metadata["clean_pool_totals"]["query_pool_record_count"]
    require(len(result) == expected_records, "Pool-checksum record count mismatch")
    for config_id, config in configs.items():
        require(per_config[config_id] == config["query_count"],
                f"Pool query count mismatch: {config_id}")
    return result


def validate_pool_members(metadata: dict, checksums: dict[tuple[str, str], dict]) -> None:
    seen: set[tuple[str, str]] = set()
    distinct_queries: set[tuple[str, str]] = set()
    member_rows = 0
    current_key: tuple[str, str] | None = None
    expected_position = 0
    ordered = hashlib.sha256()
    unordered_lines: list[str] = []

    def finish_group() -> None:
        nonlocal current_key, expected_position, ordered, unordered_lines
        if current_key is None:
            return
        expected = checksums[current_key]
        require(expected_position == int(expected["member_count"]),
                f"Pool member count mismatch: {current_key}")
        require(ordered.hexdigest() == expected["ordered_pool_sha256"],
                f"Ordered pool digest mismatch: {current_key}")
        unordered_digest = sha256_text("".join(sorted(unordered_lines)))
        require(unordered_digest == expected["unordered_pool_sha256"],
                f"Unordered pool digest mismatch: {current_key}")
        seen.add(current_key)

    with gzip.open(ARCHIVE / "clean_pool_members.csv.gz", "rt",
                   encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == MEMBER_FIELDS, "Clean-pool member schema mismatch")
        for line_number, row in enumerate(reader, 2):
            key = (row["pool_config_id"], row["qid"])
            require(key in checksums, f"Member has no checksum record at line {line_number}")
            if key != current_key:
                finish_group()
                require(key not in seen, f"Non-contiguous pool group at line {line_number}")
                current_key = key
                expected_position = 0
                ordered = hashlib.sha256()
                unordered_lines = []
            position = int(row["preencoder_position"])
            require(position == expected_position,
                    f"Non-consecutive pool position at line {line_number}")
            expected = checksums[key]
            require(row["dataset"] == expected["dataset"],
                    f"Member dataset mismatch at line {line_number}")
            require(row["requested_split"] == expected["requested_split"],
                    f"Member split mismatch at line {line_number}")
            for field in (
                "source_title_sha256", "source_text_sha256", "source_content_sha256",
                "loader_title_sha256", "loader_text_sha256", "loader_content_sha256",
            ):
                require(valid_sha256(row[field]), f"Invalid {field} at line {line_number}")
            ordered.update(
                f"{position}\t{row['source_variant_id']}\t{row['loader_content_sha256']}\n".encode("utf-8")
            )
            unordered_lines.append(
                f"{row['source_variant_id']}\t{row['loader_content_sha256']}\n"
            )
            expected_position += 1
            member_rows += 1
            distinct_queries.add((row["dataset"], row["qid"]))
    finish_group()
    require(seen == set(checksums), "Pool member/checksum key coverage mismatch")
    totals = metadata["clean_pool_totals"]
    require(member_rows == totals["membership_rows"], "Pool membership total mismatch")
    require(len(seen) == totals["query_pool_record_count"], "Pool record total mismatch")
    require(len(distinct_queries) == totals["distinct_dataset_query_keys"],
            "Distinct dataset/query count mismatch")


def load_splits(dataset: str) -> dict[tuple[int, str], str]:
    result = {}
    path = ROOT / "manifests" / "splits" / f"{dataset}.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["seed"]), row["query_id"])
            require(key not in result, f"Duplicate frozen split key: {dataset}/{key}")
            result[key] = row["partition"]
    return result


def validate_triggers(metadata: dict, run_by_id: dict[str, dict]) -> None:
    summaries = {}
    rules = {}
    expected_artifact_rows = {}
    split_maps = {}
    for item in metadata["trigger_artifacts"]:
        key = (item["corrected_run"], item["experiment"])
        require(key not in summaries, f"Duplicate trigger artifact: {key}")
        summaries[key] = item
        expected_artifact_rows[key] = int(item["decision_rows"])
        source_path = ROOT / item["source_per_query"]
        require(sha256_file(source_path) == item["source_per_query_sha256"],
                f"Trigger source hash mismatch: {item['source_per_query']}")
        split_maps.setdefault(item["dataset"], load_splits(item["dataset"]))
        for rule in item["selected_rules"]:
            rule_key = (item["corrected_run"], item["experiment"], int(rule["seed"]))
            require(rule_key not in rules, f"Duplicate selected trigger rule: {rule_key}")
            rules[rule_key] = (float(rule["tau"]), rule["fallback_method"])

    require(len(summaries) == 13, f"Expected 13 trigger artifacts, found {len(summaries)}")
    seen: set[tuple[str, str, str, int, str]] = set()
    artifact_counts = Counter()
    partition_counts = Counter()
    levels_by_artifact_seed: dict[tuple[str, str, int], set[float]] = defaultdict(set)
    path = ARCHIVE / "trigger_decisions.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == TRIGGER_FIELDS, "Trigger-decision schema mismatch")
        for line_number, row in enumerate(reader, 2):
            context = f"trigger line {line_number}"
            artifact_key = (row["corrected_run"], row["experiment"])
            summary = summaries.get(artifact_key)
            require(summary is not None, f"Unknown artifact in {context}")
            require(row["source_run"] == summary["source_run"], f"Source run mismatch in {context}")
            require(row["dataset"] == summary["dataset"], f"Dataset mismatch in {context}")
            seed = int(row["seed"])
            key = (row["corrected_run"], row["experiment"], row["level"], seed, row["qid"])
            require(key not in seen, f"Duplicate trigger key in {context}")
            seen.add(key)

            run = run_by_id[row["corrected_run"]]
            params = run["hyperparameters"]
            require(row["objective"] == params["objective"], f"Objective mismatch in {context}")
            expected_level_name = "rho" if row["experiment"] == "redundancy" else "overlap"
            require(row["level_name"] == expected_level_name, f"Level name mismatch in {context}")
            level = parse_finite(row["level"], context)
            grid = params["rho_grid"] if expected_level_name == "rho" else params["overlap_grid"]
            require(any(math.isclose(level, float(value), rel_tol=0, abs_tol=1e-12) for value in grid),
                    f"Level outside run grid in {context}")
            levels_by_artifact_seed[(row["corrected_run"], row["experiment"], seed)].add(level)

            split = split_maps[row["dataset"]].get((seed, row["qid"]))
            require(split == row["partition"], f"Frozen split mismatch in {context}")
            require(split in {"validation", "test"}, f"Invalid partition in {context}")
            require(row["trigger_metric"] == "Vendi", f"Trigger metric mismatch in {context}")
            tau_expected, fallback_expected = rules[(row["corrected_run"], row["experiment"], seed)]
            tau = parse_finite(row["tau"], context)
            require(tau == tau_expected, f"Threshold mismatch in {context}")
            require(row["fallback_method"] == fallback_expected, f"Fallback mismatch in {context}")

            trigger_value = parse_finite(row["trigger_value"], context)
            fired = parse_bool(row["trigger_fired"], context)
            require(fired == (trigger_value < tau), f"Strict trigger rule mismatch in {context}")
            relevant_size = int(row["relevant_set_size"])
            require(int(row["gate_min_rel"]) == 2, f"Gate threshold mismatch in {context}")
            gate_pass = parse_bool(row["gate_pass"], context)
            require(gate_pass == (relevant_size >= 2), f"Gate rule mismatch in {context}")
            gated_fired = parse_bool(row["gated_fired"], context)
            require(gated_fired == (fired and gate_pass), f"Gated trigger mismatch in {context}")
            require(row["ungated_selected_method"] == (fallback_expected if fired else "kNN"),
                    f"Ungated method mismatch in {context}")
            require(row["gated_selected_method"] == (fallback_expected if gated_fired else "kNN"),
                    f"Gated method mismatch in {context}")
            knn = parse_finite(row["knn_objective"], context)
            fallback = parse_finite(row["fallback_objective"], context)
            selected = parse_finite(row["gated_selected_objective"], context)
            expected_selected = fallback if gated_fired else knn
            require(math.isclose(selected, expected_selected, rel_tol=0, abs_tol=1e-15),
                    f"Selected objective mismatch in {context}")
            artifact_counts[artifact_key] += 1
            partition_counts[row["partition"]] += 1

    totals = metadata["trigger_totals"]
    require(len(seen) == totals["decision_rows"], "Trigger decision total mismatch")
    require(partition_counts["validation"] == totals["validation_rows"],
            "Validation trigger-row total mismatch")
    require(partition_counts["test"] == totals["test_rows"],
            "Test trigger-row total mismatch")
    require(dict(artifact_counts) == expected_artifact_rows, "Per-artifact trigger totals mismatch")

    for artifact_key, summary in summaries.items():
        params = run_by_id[summary["corrected_run"]]["hyperparameters"]
        grid = params["rho_grid"] if summary["experiment"] == "redundancy" else params["overlap_grid"]
        for rule in summary["selected_rules"]:
            seed = int(rule["seed"])
            observed_levels = levels_by_artifact_seed[(summary["corrected_run"], summary["experiment"], seed)]
            require(len(observed_levels) == len(grid),
                    f"Incomplete level coverage: {artifact_key}/seed={seed}")
            split_query_count = sum(
                1 for (split_seed, _), partition in split_maps[summary["dataset"]].items()
                if split_seed == seed and partition in {"validation", "test"}
            )
            expected = split_query_count * len(grid)
            actual = sum(
                1 for key in seen
                if key[0] == summary["corrected_run"] and key[1] == summary["experiment"]
                and key[3] == seed
            )
            require(actual == expected, f"Incomplete query/level coverage: {artifact_key}/seed={seed}")


def validate_generation_coverage(metadata: dict) -> None:
    coverage = metadata["generation_coverage"]
    require(coverage["qid_keyed_score_artifact_count"] == 2,
            "Expected exactly two qid-keyed generation-score artifacts")
    require(coverage["qid_keyed_score_rows"] == 199810,
            "Unexpected qid-keyed generation-score row total")
    require(coverage["raw_prediction_text_artifact_count"] == 0,
            "Raw prediction text is incorrectly claimed as retained")


def validate() -> None:
    metadata_path = ARCHIVE / "metadata.json"
    require(metadata_path.is_file(), "Missing Execution-artifact metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(metadata["format_version"] == 1, "Unsupported Execution-artifact format version")
    require(metadata["status"] == "addressed_with_archival_limitation",
            "Execution-artifact status must disclose its archival limitation")
    unavailable = metadata["field_status"]
    for field in (
        "embedding_tensors_or_checksums", "post_encoder_top_m_membership",
        "ordered_method_rankings", "raw_generated_answers",
    ):
        require(unavailable[field] == "unavailable_not_serialized",
                f"Historical limitation is not disclosed for {field}")

    validate_sha256sums()
    validate_output_artifacts(metadata)
    validate_reference_hashes(metadata)
    run_by_id = validate_inventory(metadata)
    checksums = load_pool_checksums(metadata)
    validate_pool_members(metadata, checksums)
    validate_triggers(metadata, run_by_id)
    validate_generation_coverage(metadata)
    print(
        "Execution artifact validation OK: "
        f"{metadata['inventory_totals']['run_count']} runs, "
        f"{metadata['inventory_totals']['artifact_count']} retained artifacts, "
        f"{metadata['clean_pool_totals']['membership_rows']} pool memberships, "
        f"{metadata['trigger_totals']['decision_rows']} trigger decisions."
    )


if __name__ == "__main__":
    validate()
