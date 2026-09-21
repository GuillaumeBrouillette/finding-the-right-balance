#!/usr/bin/env python3
"""Validate and summarize the Qwen3.8 RQ1 generation run."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


METHODS = [
    ("kNN", r"$k$-NN"),
    ("MMR(0.3)", r"\MMR ($\lambda=0.3$)"),
    ("MMR(0.5)", r"\MMR ($\lambda=0.5$)"),
    ("MMR(0.7)", r"\MMR ($\lambda=0.7$)"),
    ("MMR(0.9)", r"\MMR ($\lambda=0.9$)"),
    ("Maxmin", "Maxmin"),
    ("Greedy-DPP", r"Greedy \DPP"),
    ("RNG(0.2)", r"\RNGSCORE ($\gamma^{\*}=0.2$)"),
]


def filter_partition(rows, split_manifest, partition, split_seed):
    """Select one frozen split partition and verify complete query coverage."""
    with split_manifest.open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    required = {"dataset", "seed", "query_id", "partition"}
    if not split_rows or not required.issubset(split_rows[0]):
        present = set(split_rows[0]) if split_rows else set()
        raise SystemExit(
            f"Split manifest missing required columns: {sorted(required - present)}"
        )
    selected_list = [
        row["query_id"] for row in split_rows
        if int(row["seed"]) == split_seed and row["partition"] == partition
        and row["dataset"] == "hotpotqa_fullwiki"
    ]
    selected = set(selected_list)
    if not selected or len(selected) != len(selected_list):
        raise SystemExit(
            f"Expected unique HotpotQA query IDs for seed={split_seed}, "
            f"partition={partition}; found {len(selected_list)} rows and "
            f"{len(selected)} unique IDs"
        )
    filtered = [
        row for row in rows
        if int(row["seed"]) == split_seed and row["qid"] in selected
    ]
    present = {row["qid"] for row in filtered}
    if present != selected:
        raise SystemExit(
            f"Generation rows do not cover the frozen partition: "
            f"{len(selected - present)} missing, {len(present - selected)} unexpected"
        )
    return filtered, len(selected)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mark(values):
    """Rank values at the one-decimal percentage precision displayed."""
    shown = {name: round(100 * value, 1) for name, value in values.items()}
    levels = sorted(set(shown.values()), reverse=True)
    best = levels[0]
    second = levels[1] if len(levels) > 1 else None
    return {
        name: "best" if value == best else "second" if value == second else ""
        for name, value in shown.items()
    }


def render(value, decoration):
    value = f"{100 * value:.1f}"
    if decoration == "best":
        return rf"\textbf{{{value}}}"
    if decoration == "second":
        return rf"\underline{{{value}}}"
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-query", required=True, type=Path)
    parser.add_argument("--run-params", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-tex", required=True, type=Path)
    parser.add_argument(
        "--partition", choices=("all", "validation", "test"), default="all",
        help="query scope to summarize; validation/test require --split-manifest",
    )
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()
    if args.partition != "all" and args.split_manifest is None:
        raise SystemExit("--split-manifest is required when --partition is not 'all'")
    if args.partition == "all" and args.split_manifest is not None:
        raise SystemExit("--split-manifest requires --partition validation or test")

    with args.run_params.open(encoding="utf-8") as handle:
        params = json.load(handle)
    expected_params = {
        "dataset": "hotpotqa_fullwiki",
        "rho_grid": [0.0],
        "seeds": [0],
        "generator_model": "Qwen/Qwen3.8-27B",
        "generator_temperature": 0,
        "generator_top_p": 1,
        "generator_seed": 0,
        "generator_num_beams": 1,
        "generator_thinking": False,
        "generator_prompt_version": "short_direct_v1",
    }
    for key, expected in expected_params.items():
        if params.get(key) != expected:
            raise SystemExit(f"Unexpected {key}: {params.get(key)!r}; expected {expected!r}")

    with args.per_query.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"rho", "seed", "qid", "Method", "EM", "F1"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"Missing required columns: {sorted(required - set(rows[0] if rows else []))}")

    expected_methods = [name for name, _ in METHODS]
    raw_counts = Counter(row["Method"] for row in rows)
    unexpected = sorted(set(raw_counts) - set(expected_methods))
    if unexpected:
        raise SystemExit(f"Unexpected methods: {unexpected}")
    if (len(set(raw_counts.values())) != 1
            or set(raw_counts) != set(expected_methods)):
        raise SystemExit(f"Unbalanced or missing method rows: {dict(raw_counts)}")
    raw_n_queries = next(iter(raw_counts.values()))
    if raw_n_queries != 7405:
        raise SystemExit(
            f"Expected 7405 source queries per method, found {raw_n_queries}"
        )

    keys = [(row["rho"], row["seed"], row["qid"], row["Method"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("Duplicate (rho, seed, qid, Method) rows detected")
    if {float(row["rho"]) for row in rows} != {0.0} or {int(row["seed"]) for row in rows} != {0}:
        raise SystemExit("Expected only rho=0 and seed=0")

    split_provenance = None
    if args.partition != "all":
        rows, expected_queries = filter_partition(
            rows, args.split_manifest, args.partition, args.split_seed
        )
        split_provenance = (
            f"% [split] {args.split_manifest}; SHA-256 "
            f"{sha256(args.split_manifest)}; seed={args.split_seed}; "
            f"partition={args.partition}."
        )
        scope = f"the {expected_queries:,} seed-{args.split_seed} {args.partition} queries"
    else:
        expected_queries = raw_n_queries
        scope = f"all {raw_n_queries:,} queries"

    counts = Counter(row["Method"] for row in rows)
    if set(counts) != set(expected_methods) or set(counts.values()) != {expected_queries}:
        raise SystemExit(f"Unbalanced partition method rows: {dict(counts)}")
    n_queries = expected_queries

    values = defaultdict(dict)
    for method in expected_methods:
        selected = [row for row in rows if row["Method"] == method]
        for metric in ("EM", "F1"):
            values[metric][method] = sum(float(row[metric]) for row in selected) / n_queries

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Method", "n_queries", "EM", "F1"])
        for method in expected_methods:
            writer.writerow([method, n_queries, f"{values['EM'][method]:.9f}", f"{values['F1'][method]:.9f}"])

    decorations = {metric: mark(values[metric]) for metric in ("EM", "F1")}
    body = []
    for method, label in METHODS:
        em = render(values["EM"][method], decorations["EM"][method])
        f1 = render(values["F1"][method], decorations["F1"][method])
        body.append(f"    {label} & {em} & {f1} \\\\")

    provenance = [
        rf"% [provenance] {args.per_query}; SHA-256 {sha256(args.per_query)}",
        rf"% [parameters] {args.run_params}; SHA-256 {sha256(args.run_params)}",
    ]
    if split_provenance:
        provenance.append(split_provenance)
    table = "\n".join([
        *provenance,
        rf"% Qwen revision {params['generator_revision']}; prompt {params['generator_prompt_version']}; temperature=0; top_p=1; seed=0; max_new_tokens={params['generator_max_new_tokens']}; thinking disabled.",
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \small",
        rf"  \caption{{Answer quality on the clean HotpotQA pools with the modern open reader \texttt{{Qwen3.8-27B}} (one deterministic run over {scope}). Best in bold, second best underlined.}}",
        r"  \label{tab:rq1-qwen}",
        r"  \begin{tabular}{lcc}",
        r"    \hline",
        r"    \textbf{Method} & \textbf{EM} & \textbf{F1} \\",
        r"    \hline",
        *body,
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ])
    args.output_tex.write_text(table, encoding="utf-8")
    print(f"Validated {len(rows)} rows ({n_queries} queries x {len(METHODS)} methods; {args.partition})")
    print(f"Saved: {args.output_csv}")
    print(f"Saved: {args.output_tex}")


if __name__ == "__main__":
    main()
