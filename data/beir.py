"""BEIR dataset loader with BM25 candidate-pool construction.

Loads a BEIR task (Thakur et al. 2021) and returns examples in the same
schema as data.loaders, with candidate pools attached so that the
redundancy / oracle / chunking experiments of evaluate_redundancy.py can run
unchanged on heterogeneous retrieval tasks:

    {
        "id"         : str,
        "question"   : str,
        "answers"    : [],
        "passages"   : List[{"title": str, "text": str}],
        "gold_titles": List[str],
    }

Candidate pools are built with a self-contained Okapi BM25 first stage
(k1 = 0.9, b = 0.4, the standard BEIR configuration), so no extra
dependencies are required.  The "title" field of each passage is the BEIR
corpus id, which is unique by construction; the human-readable title is
prepended to the passage text instead.  gold_titles are therefore the
relevant corpus ids of the query (qrels with score > 0), and all title-based
coverage metrics work unchanged.

Notes
-----
* SciFact (5.2K docs) is the laptop-friendly default; FiQA-2018 (57K) is
  moderate; TREC-COVID (171K) takes a few minutes to index and ~2 GB RAM.
* BM25 pools cap attainable Recall: a gold document missing from the
  pool cannot be retrieved by any reranker.  The loader prints the mean
  fraction of golds present in the pool so this ceiling is explicit.
* qrels splits differ by task (scifact: train/test; trec-covid: test only).
  If the requested split has no qrels file, the loader falls back to "test"
  and says so.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
import zipfile
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

BEIR_URL = ("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/"
            "datasets/{name}.zip")

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class _BM25:
    """Minimal Okapi BM25 with an inverted index (NumPy posting lists)."""

    def __init__(self, docs_tokens: List[List[str]], k1: float = 0.9,
                 b: float = 0.4):
        self.k1, self.b = k1, b
        self.n_docs = len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype=float)
        self.avgdl = max(1e-9, float(self.doc_len.mean()))
        postings: Dict[str, List] = {}
        for di, toks in enumerate(docs_tokens):
            for t, c in Counter(toks).items():
                postings.setdefault(t, []).append((di, c))
        self.idf = {
            t: math.log(1.0 + (self.n_docs - len(pl) + 0.5) / (len(pl) + 0.5))
            for t, pl in postings.items()
        }
        self.postings = {
            t: (np.array([d for d, _ in pl], dtype=np.int64),
                np.array([c for _, c in pl], dtype=float))
            for t, pl in postings.items()
        }

    def top(self, query_tokens: List[str], n: int) -> List[int]:
        scores = np.zeros(self.n_docs)
        for t in set(query_tokens):
            if t not in self.postings:
                continue
            ids, tf = self.postings[t]
            denom = tf + self.k1 * (
                1.0 - self.b + self.b * self.doc_len[ids] / self.avgdl)
            scores[ids] += self.idf[t] * tf * (self.k1 + 1.0) / denom
        n = min(n, self.n_docs)
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return [int(i) for i in top if scores[i] > 0.0]


def _download_and_extract(name: str, root: str) -> str:
    task_dir = os.path.join(root, name)
    if os.path.exists(os.path.join(task_dir, "corpus.jsonl")):
        return task_dir
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, f"{name}.zip")
    if not os.path.exists(zip_path):
        url = BEIR_URL.format(name=name)
        print(f"Downloading BEIR/{name} from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    return task_dir


def _read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_qrels(task_dir: str, split: str) -> Optional[Dict[str, Dict[str, int]]]:
    path = os.path.join(task_dir, "qrels", f"{split}.tsv")
    if not os.path.exists(path):
        return None
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline()  # "query-id\tcorpus-id\tscore"
        if not header.lower().startswith("query"):
            f.seek(0)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(float(parts[2]))
            qrels.setdefault(qid, {})[did] = score
    return qrels


def load_beir(
    name: str = "scifact",
    split: str = "test",
    max_samples: Optional[int] = None,
    pool_size: int = 200,
    data_dir: Optional[str] = None,
) -> List[Dict]:
    """Load a BEIR task with BM25-built candidate pools.

    Parameters
    ----------
    name : str
        BEIR task name as used in the official download URLs
        (e.g. "scifact", "fiqa", "trec-covid").
    split : str
        qrels split. Falls back to "test" (with a printed note) when the
        requested split has no qrels file — evaluate_redundancy.py passes
        "validation" by default, which most BEIR tasks do not have.
    max_samples : int, optional
        Cap on the number of queries (after filtering to queries with
        relevant documents).
    pool_size : int
        BM25 candidate-pool size per query. The experiment scripts truncate
        to their own top-m with the dense encoder afterwards.
    """
    root = data_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "beir")
    task_dir = _download_and_extract(name, root)

    corpus_rows = _read_jsonl(os.path.join(task_dir, "corpus.jsonl"))
    doc_ids = [str(r["_id"]) for r in corpus_rows]
    doc_texts = []
    for r in corpus_rows:
        title = (r.get("title") or "").strip()
        body = (r.get("text") or "").strip()
        doc_texts.append(f"{title}. {body}" if title else body)

    queries = {str(r["_id"]): r.get("text") or "" for r in
               _read_jsonl(os.path.join(task_dir, "queries.jsonl"))}

    qrels = _read_qrels(task_dir, split)
    if qrels is None:
        print(f"   [beir/{name}] no qrels for split '{split}'; using 'test'.")
        qrels = _read_qrels(task_dir, "test")
    if qrels is None:
        raise FileNotFoundError(
            f"No usable qrels found for BEIR task '{name}'.")

    print(f"   [beir/{name}] indexing {len(doc_ids)} documents with BM25 ...")
    bm25 = _BM25([_tokenize(t) for t in doc_texts])

    examples: List[Dict] = []
    gold_in_pool: List[float] = []
    for qid, rels in qrels.items():
        golds = [did for did, score in rels.items() if score > 0]
        if not golds or qid not in queries:
            continue
        question = queries[qid]
        pool_idx = bm25.top(_tokenize(question), pool_size)
        passages = [{"title": doc_ids[i], "text": doc_texts[i]}
                    for i in pool_idx]
        pool_ids = {doc_ids[i] for i in pool_idx}
        gold_in_pool.append(
            sum(1 for g in golds if g in pool_ids) / len(golds))
        examples.append({
            "id": qid,
            "question": question,
            "answers": [],
            "passages": passages,
            "gold_titles": golds,
        })
        if max_samples is not None and len(examples) >= max_samples:
            break

    if gold_in_pool:
        print(f"   [beir/{name}] {len(examples)} queries; mean fraction of "
              f"golds inside the BM25 pool: {np.mean(gold_in_pool):.3f} "
              f"(ceiling on Recall@k).")
    return examples


def load_scifact(split: str = "test", max_samples: Optional[int] = None,
                 **kwargs) -> List[Dict]:
    return load_beir("scifact", split=split, max_samples=max_samples, **kwargs)


def load_fiqa(split: str = "test", max_samples: Optional[int] = None,
              **kwargs) -> List[Dict]:
    return load_beir("fiqa", split=split, max_samples=max_samples, **kwargs)


def load_trec_covid(split: str = "test", max_samples: Optional[int] = None,
                    **kwargs) -> List[Dict]:
    return load_beir("trec-covid", split=split, max_samples=max_samples,
                     **kwargs)
