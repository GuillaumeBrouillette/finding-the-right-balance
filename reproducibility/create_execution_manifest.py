#!/usr/bin/env python3
"""Freeze historical execution code, transformation parameters, and RNG seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "retained"
ARCHIVE_RESULTS_ROOT = Path("results/retained")
OUTPUT = ROOT / "manifests" / "execution"
RNG_PATTERN = re.compile(
    r"default_rng\(([^)]*)\)|manual_seed\(([^)]*)\)|random\.seed\(([^)]*)\)"
)
TRANSFORMATION_KEYS = [
    "experiment", "rho_grid", "overlap_grid", "chunk_window",
    "dup_noise", "dup_target",
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def artifact_families(run_dir: Path) -> list[str]:
    families = []
    names = {path.name for path in run_dir.iterdir() if path.is_file()}
    prefixes = {
        "injection": "results_redundancy",
        "chunking": "results_chunking",
        "oracle": "results_oracle",
        "cross_encoder": "results_ce_",
        "beir": "results_beir",
        "generation": "results_generation",
    }
    for family, prefix in prefixes.items():
        if any(name.startswith(prefix) for name in names):
            families.append(family)
    if not families:
        families.append("summary_only")
    return families


def effective_seeds(params: dict, source_text: str) -> tuple[list[int], str]:
    if params.get("seeds") is not None:
        return [int(seed) for seed in params["seeds"]], "run_params.seeds"
    if params.get("seed") is not None:
        return [int(params["seed"])], "run_params.seed"
    if "default_rng(0)" in source_text:
        return [0], "hard-coded numpy.random.default_rng(0) in historical source"
    return [], "no random-number generator invoked by the primary execution script"


def rng_sites(source_text: str) -> list[dict]:
    sites = []
    for number, line in enumerate(source_text.splitlines(), start=1):
        match = RNG_PATTERN.search(line)
        if not match:
            continue
        expression = next(group for group in match.groups() if group is not None).strip()
        sites.append({
            "line": number,
            "call": match.group(0),
            "seed_expression": expression,
            "source_line": line.strip(),
        })
    return sites


def source_datasets(params: dict) -> list[str]:
    if params.get("dataset"):
        dataset = str(params["dataset"])
        return ["trec-covid" if dataset == "trec_covid" else dataset]
    return [str(task) for task in (params.get("tasks") or [])]


def create(historical_repo: Path | None, output: Path, results: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    code_dir = output / "historical_code"
    code_dir.mkdir(parents=True, exist_ok=True)
    existing_metadata_path = output / "metadata.json"
    existing_metadata = (
        json.loads(existing_metadata_path.read_text(encoding="utf-8"))
        if existing_metadata_path.is_file()
        else {}
    )
    archived_snapshots = {
        (item["git_commit"], item["repository_path"]): item
        for item in existing_metadata.get("snapshots", [])
    }
    source_groups = ROOT / "manifests" / "source_groups" / "metadata.json"
    source_group_metadata = json.loads(source_groups.read_text(encoding="utf-8"))
    available_sources = {entry["dataset"] for entry in source_group_metadata["datasets"]}

    runs = []
    snapshots: dict[tuple[str, str], dict] = {}
    for params_path in sorted(results.glob("*/run_params.json")):
        run_dir = params_path.parent
        params = json.loads(params_path.read_text(encoding="utf-8"))
        short_commit = str(params["git_commit"])
        script = str(params["script"])
        repo_path = f"implementation/{script}"
        if historical_repo is not None:
            full_commit = git(historical_repo, "rev-parse", short_commit).decode().strip()
        else:
            matches = [
                item for (commit, path), item in archived_snapshots.items()
                if commit.startswith(short_commit) and path == repo_path
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"No unique archived snapshot for {short_commit}:{repo_path}; "
                    "provide --historical-repo"
                )
            full_commit = matches[0]["git_commit"]
        key = (full_commit, script)
        if key not in snapshots:
            destination = code_dir / full_commit / script
            if historical_repo is not None:
                source = git(historical_repo, "show", f"{full_commit}:{repo_path}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source)
                blob = git(historical_repo, "rev-parse", f"{full_commit}:{repo_path}").decode().strip()
            else:
                archived = archived_snapshots[(full_commit, repo_path)]
                source = (ROOT / archived["snapshot"]).read_bytes()
                blob = archived["git_blob"]
            snapshots[key] = {
                "git_commit": full_commit,
                "git_blob": blob,
                "repository_path": repo_path,
                "snapshot": destination.relative_to(ROOT).as_posix(),
                "snapshot_sha256": sha256_bytes(source),
                "rng_sites": rng_sites(source.decode("utf-8")),
            }
        snapshot = snapshots[key]
        source_text = (ROOT / snapshot["snapshot"]).read_text(encoding="utf-8")
        seeds, seed_evidence = effective_seeds(params, source_text)
        datasets = source_datasets(params)
        missing_sources = set(datasets).difference(available_sources)
        # The exploratory distractor HotpotQA run is not part of the seven
        # headline datasets and consequently has no source-identity source registry.
        if missing_sources.difference({"hotpotqa"}):
            raise ValueError(f"Missing source identities for {sorted(missing_sources)}")
        source_identity_status = (
            "complete"
            if not missing_sources
            else "not_in_reported_dataset_scope: exploratory HotpotQA distractor run"
        )
        families = artifact_families(run_dir)
        transformations = {
            key: params[key] for key in TRANSFORMATION_KEYS if key in params
        }
        random_streams = []
        for site in snapshot["rng_sites"]:
            expression = site["seed_expression"]
            values: list[int] | str
            if expression in {"seed", "cfg[\"seed\"]", "cfg['seed']"}:
                values = seeds
            elif expression == "seed + 1":
                values = [seed + 1 for seed in seeds]
            elif expression.isdigit():
                values = [int(expression)]
            else:
                values = f"derived at runtime from: {expression}"
            random_streams.append({**site, "resolved_seed_values": values})
        runs.append({
            "run_id": run_dir.name,
            "run_params": (
                ARCHIVE_RESULTS_ROOT / params_path.relative_to(results)
            ).as_posix(),
            "run_params_sha256": sha256_file(params_path),
            "script": script,
            "historical_git_commit_short": short_commit,
            "historical_git_commit": full_commit,
            "historical_code_snapshot": snapshot["snapshot"],
            "historical_code_sha256": snapshot["snapshot_sha256"],
            "artifact_families": families,
            "source_datasets": datasets,
            "source_identity_manifest": "manifests/source_groups/metadata.json",
            "source_identity_status": source_identity_status,
            "effective_seeds": seeds,
            "seed_evidence": seed_evidence,
            "random_streams": random_streams,
            "transformation_parameters": transformations,
            "generation_is_deterministic": (
                True if params.get("run_generation") or "generation" in families else None
            ),
            "generation_decoding": (
                "beam/greedy decoding with do_sample=False; no sampling seed"
                if params.get("run_generation") or "generation" in families else None
            ),
        })

    metadata = {
        "format_version": 1,
        "historical_repository_head_at_recovery": (
            git(historical_repo, "rev-parse", "HEAD").decode().strip()
            if historical_repo is not None
            else existing_metadata["historical_repository_head_at_recovery"]
        ),
        "source_identity_metadata": source_groups.relative_to(ROOT).as_posix(),
        "source_identity_metadata_sha256": sha256_file(source_groups),
        "seed_rules": {
            "injection_pool": (
                "numpy.random.default_rng(seed), re-created for each rho; source "
                "selection and text perturbation share that stream"
            ),
            "validation_test_split": (
                "an independent numpy.random.default_rng(seed) permutation"
            ),
            "generation_query_subset": (
                "numpy.random.default_rng(seed + 1) where that feature exists"
            ),
            "chunking": "deterministic; seed affects only validation/test assignment",
            "generation_decoding": "deterministic do_sample=False; no sampling seed",
        },
        "algorithm_rules": {
            "injection": (
                "n_dup=round(rho*original_pool_size); append exact/light/heavy "
                "copies selected by gold/random/mixed targeting; copied passages "
                "inherit their source title"
            ),
            "chunking": (
                "whitespace word windows; stride=max(1,round(window*(1-overlap))); "
                "drop a trailing fragment shorter than half a window; chunks inherit "
                "their source title"
            ),
        },
        "snapshots": sorted(
            snapshots.values(), key=lambda item: (item["git_commit"], item["repository_path"])
        ),
        "runs": runs,
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    checksums = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(
                f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
            )
    (output / "SHA256SUMS").write_text("".join(checksums), encoding="utf-8")
    print(f"Frozen {len(runs)} runs and {len(snapshots)} historical script snapshots")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-repo",
        type=Path,
        help="Optional checkout containing the historical commits; published snapshots are used by default",
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    create(
        args.historical_repo.resolve() if args.historical_repo else None,
        args.output_dir.resolve(),
        args.results.resolve(),
    )


if __name__ == "__main__":
    main()
