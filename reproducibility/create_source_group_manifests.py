#!/usr/bin/env python3
"""Freeze source-document identities used by the reported experiments.

The split manifests freeze query identity and membership.  This companion
script assigns a stable source-group ID to every original document available
to the historical loaders and records content digests without copying the
document text into the repository.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable, Iterator, TextIO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "manifests" / "source_groups"
DEFAULT_LOCAL_DATA = ROOT / "data"
DEFAULT_EXPERIMENT_DATA = ROOT / "data"
FIELDS = [
    "dataset",
    "dataset_split",
    "source_document_id",
    "source_group_id",
    "source_variant_id",
    "title_sha256",
    "text_sha256",
    "content_sha256",
]


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


def source_group_id(namespace: str, native_id: str) -> str:
    digest = sha256_text(f"{namespace}\0{native_id}")
    return f"{namespace}:{digest}"


def source_row(
    dataset: str,
    dataset_split: str,
    namespace: str,
    native_id: str,
    title: str,
    text: str,
) -> dict[str, str]:
    group_id = source_group_id(namespace, native_id)
    title_hash = sha256_text(title)
    text_hash = sha256_text(text)
    content_hash = sha256_text(f"{title}\0{text}")
    return {
        "dataset": dataset,
        "dataset_split": dataset_split,
        "source_document_id": native_id,
        "source_group_id": group_id,
        "source_variant_id": f"{group_id}:{content_hash}",
        "title_sha256": title_hash,
        "text_sha256": text_hash,
        "content_sha256": content_hash,
    }


def iter_concatenated_json(handle: TextIO) -> Iterator[dict]:
    """Stream whitespace-separated JSON values (the MDR file format)."""
    decoder = json.JSONDecoder()
    buffer = ""
    eof = False
    while True:
        buffer = buffer.lstrip()
        if not buffer and eof:
            return
        try:
            value, end = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            if eof:
                raise
            block = handle.read(1024 * 1024)
            eof = not block
            buffer += block
            continue
        yield value
        buffer = buffer[end:]


def iter_json_array(handle: TextIO) -> Iterator[dict]:
    """Stream objects from one top-level JSON array."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    eof = False
    while True:
        if not eof and len(buffer) < 1024 * 1024:
            block = handle.read(1024 * 1024)
            eof = not block
            buffer += block
        buffer = buffer.lstrip()
        if not started:
            if not buffer and eof:
                raise ValueError("Empty JSON input")
            if not buffer:
                continue
            if buffer[0] != "[":
                raise ValueError("Expected a top-level JSON array")
            buffer = buffer[1:]
            started = True
        buffer = buffer.lstrip()
        if buffer.startswith("]"):
            return
        if buffer.startswith(","):
            buffer = buffer[1:].lstrip()
        try:
            value, end = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            if eof:
                raise
            continue
        yield value
        buffer = buffer[end:]


def manifest_query_order(dataset: str) -> list[str]:
    path = ROOT / "manifests" / "splits" / f"{dataset}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            row["query_id"]
            for row in csv.DictReader(handle)
            if row["seed"] == "0"
        ]


class ManifestWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.raw = path.open("wb")
        self.compressed = gzip.GzipFile(
            filename="", mode="wb", fileobj=self.raw, mtime=0
        )
        self.text = io.TextIOWrapper(self.compressed, encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.text, fieldnames=FIELDS, lineterminator="\n")
        self.writer.writeheader()
        self.seen_variants: set[str] = set()
        self.groups: set[str] = set()
        self.rows = 0

    def add(self, row: dict[str, str]) -> None:
        variant = row["source_variant_id"]
        if variant in self.seen_variants:
            return
        self.seen_variants.add(variant)
        self.groups.add(row["source_group_id"])
        self.writer.writerow(row)
        self.rows += 1

    def close(self) -> tuple[int, int]:
        self.text.flush()
        self.text.detach()
        self.compressed.close()
        self.raw.close()
        return self.rows, len(self.groups)


def verify_query_order(dataset: str, actual: list[str]) -> str:
    expected = manifest_query_order(dataset)
    if actual != expected:
        raise ValueError(
            f"{dataset}: source query order does not match the frozen manifest "
            f"({len(actual)} source IDs versus {len(expected)} expected IDs)"
        )
    return sha256_text("".join(f"{qid}\n" for qid in actual))


def create_hotpot(path: Path, output: Path) -> dict:
    writer = ManifestWriter(output)
    query_ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for example in iter_concatenated_json(handle):
            query_ids.append(str(example["_id"]))
            seen_titles: set[str] = set()
            for chain in example["candidate_chains"]:
                for passage in chain:
                    title = str(passage["title"])
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    writer.add(source_row(
                        "hotpotqa_fullwiki", "validation", "wikipedia-title",
                        title, title, " ".join(passage["sents"]),
                    ))
    rows, groups = writer.close()
    return result_metadata(path, output, rows, groups, query_ids, "hotpotqa_fullwiki")


def create_nq(path: Path, output: Path) -> dict:
    writer = ManifestWriter(output)
    query_ids: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, example in enumerate(iter_json_array(handle)):
            contexts = example["ctxs"]
            if not any(bool(context.get("has_answer")) for context in contexts):
                continue
            query_ids.append(str(index))
            for context in contexts:
                native_id = str(context["id"])
                writer.add(source_row(
                    "nq", "validation", "dpr-wikipedia-passage", native_id,
                    str(context.get("title") or ""), str(context.get("text") or ""),
                ))
    rows, groups = writer.close()
    return result_metadata(path, output, rows, groups, query_ids, "nq")


def create_two_wiki(path: Path, output: Path) -> dict:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is required to read the frozen 2Wiki parquet") from error
    writer = ManifestWriter(output)
    query_ids: list[str] = []
    table = parquet.read_table(path, columns=["_id", "context"])
    for example in table.to_pylist():
        query_ids.append(str(example["_id"]))
        for title, sentences in json.loads(example["context"]):
            title = str(title)
            writer.add(source_row(
                "2wikimultihopqa", "validation", "wikipedia-title", title,
                title, " ".join(sentences),
            ))
    rows, groups = writer.close()
    return result_metadata(path, output, rows, groups, query_ids, "2wikimultihopqa")


def create_musique(path: Path, output: Path) -> dict:
    writer = ManifestWriter(output)
    query_ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            example = json.loads(line)
            if not example.get("answerable", True):
                continue
            query_ids.append(str(example["id"]))
            for passage in example["paragraphs"]:
                title = str(passage["title"])
                writer.add(source_row(
                    "musique", "validation", "wikipedia-title", title, title,
                    str(passage["paragraph_text"]),
                ))
    rows, groups = writer.close()
    return result_metadata(path, output, rows, groups, query_ids, "musique")


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def beir_query_order(task_dir: Path) -> list[str]:
    queries = {str(row["_id"]) for row in read_jsonl(task_dir / "queries.jsonl")}
    result: list[str] = []
    seen: set[str] = set()
    with (task_dir / "qrels" / "test.tsv").open(encoding="utf-8") as handle:
        header = next(handle, "")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or int(float(parts[2])) <= 0:
                continue
            query_id = parts[0]
            if query_id in queries and query_id not in seen:
                seen.add(query_id)
                result.append(query_id)
    return result


def create_beir(dataset: str, path: Path, output: Path) -> dict:
    writer = ManifestWriter(output)
    for document in read_jsonl(path / "corpus.jsonl"):
        native_id = str(document["_id"])
        writer.add(source_row(
            dataset, "test", f"beir-{dataset}", native_id,
            str(document.get("title") or ""), str(document.get("text") or ""),
        ))
    rows, groups = writer.close()
    metadata = result_metadata(
        path / "corpus.jsonl", output, rows, groups,
        beir_query_order(path), dataset,
    )
    metadata["input_files"] = {
        relative_label(path / name): sha256_file(path / name)
        for name in ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]
    }
    return metadata


def relative_label(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def result_metadata(
    source: Path,
    output: Path,
    rows: int,
    groups: int,
    query_ids: list[str],
    dataset: str,
) -> dict:
    return {
        "dataset": dataset,
        "source_input": relative_label(source),
        "source_input_sha256": sha256_file(source),
        "query_count": len(query_ids),
        "query_order_sha256": verify_query_order(dataset, query_ids),
        "source_group_count": groups,
        "source_variant_count": rows,
        "manifest": output.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(output),
    }


def require(path: Path) -> Path:
    if not path.is_file() and not path.is_dir():
        raise FileNotFoundError(path)
    return path


def create(local_data: Path, experiment_data: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        create_hotpot(
            require(experiment_data / "mdr" / "hotpotqa_dev_top100.json"),
            output_dir / "hotpotqa_fullwiki.csv.gz",
        ),
        create_nq(
            require(experiment_data / "dpr" / "nq-dev.json.gz"),
            output_dir / "nq.csv.gz",
        ),
        create_two_wiki(
            require(local_data / "2wikimultihopqa" / "dev.parquet"),
            output_dir / "2wikimultihopqa.csv.gz",
        ),
        create_musique(
            require(local_data / "musique" / "musique_ans_v1.0_dev.jsonl"),
            output_dir / "musique.csv.gz",
        ),
    ]
    for dataset in [
        "scifact", "fiqa", "trec-covid", "arguana", "webis-touche2020"
    ]:
        entries.append(create_beir(
            dataset,
            require(experiment_data / "beir" / dataset),
            output_dir / f"{dataset}.csv.gz",
        ))

    identities = {
        "2wikimultihopqa": {
            "source": "Hugging Face xanhho/2WikiMultihopQA dev.parquet",
            "immutable_revision": "612bc5039a457880d9e7d84c3b0a4cf154b70e4f",
            "source_url": "https://huggingface.co/datasets/xanhho/2WikiMultihopQA",
        },
        "musique": {
            "source": "MuSiQue answerable v1.0 development JSONL",
            "immutable_revision": "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321",
            "source_url": "https://huggingface.co/datasets/dgslibisey/musique",
        },
        "hotpotqa_fullwiki": {
            "source": "MDR HotpotQA dev top-100 chains",
            "source_url": "https://dl.fbaipublicfiles.com/mdpr/data/hotpot/dev_retrieval_top100_sp.json",
        },
        "nq": {
            "source": "DPR NQ development top-100 retrieval results",
            "source_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-dev.json.gz",
        },
        "scifact": {
            "source": "BEIR SciFact corpus and test qrels",
            "source_url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        },
        "fiqa": {
            "source": "BEIR FiQA corpus and test qrels",
            "source_url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
        },
        "trec-covid": {
            "source": "BEIR TREC-COVID corpus and test qrels",
            "source_url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip",
        },
        "arguana": {
            "source": "BEIR ArguAna corpus and test qrels",
            "source_url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip",
        },
        "webis-touche2020": {
            "source": "BEIR Webis-Touche2020 corpus and test qrels",
            "source_url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip",
        },
    }
    for entry in entries:
        entry.update(identities[entry["dataset"]])

    metadata = {
        "format_version": 1,
        "group_rule": (
            "SHA256(namespace + NUL + native document ID); title is the native "
            "ID when the dataset exposes no passage ID"
        ),
        "variant_rule": "source_group_id + ':' + SHA256(title + NUL + text)",
        "scope": (
            "Original source-document identities. Per-query post-encoding candidate "
            "pools and transformed duplicate/chunk instances are separate execution-artifact artifacts."
        ),
        "datasets": entries,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_lines = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.name == "SHA256SUMS" or not path.is_file():
            continue
        checksum_lines.append(f"{sha256_file(path)}  {path.name}\n")
    (output_dir / "SHA256SUMS").write_text("".join(checksum_lines), encoding="utf-8")
    for entry in entries:
        print(
            f"{entry['dataset']}: {entry['query_count']} queries, "
            f"{entry['source_group_count']} source groups"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-data-root", type=Path, default=DEFAULT_LOCAL_DATA)
    parser.add_argument(
        "--experiment-data-root", type=Path, default=DEFAULT_EXPERIMENT_DATA
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    create(
        args.local_data_root.resolve(),
        args.experiment_data_root.resolve(),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
