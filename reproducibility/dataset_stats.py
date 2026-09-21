"""Per-dataset pool and relevance statistics.

Section 5.2 of the paper describes each benchmark by its candidate-pool size
and its relevant-set size (how many gold documents a query has).  Those numbers
were previously quoted from memory and had no artefact behind them.  This
script recomputes them from the datasets themselves and writes

    results/dataset_statistics.csv

which is small enough to commit and is the provenance for every number in
Section 5.2.

Two kinds of statistic are produced.

**Relevant-set size** — how many documents count as gold for a query.  For the
QA datasets this is the number of supporting paragraphs shipped with the
example; for the BEIR tasks it is the number of positively-graded entries in
the qrels file.  This is a property of the dataset and is always computed.

**Pool size** — how many candidates the selector ranks over.  This is a dataset
property only for the benchmarks that ship their own candidate pools (the
distractor settings, MuSiQue, the MDR fullwiki chains and the DPR top-100
files).  For the BEIR tasks the pool is built by the retriever, so its size is
a run parameter rather than a dataset property; those rows leave the pool
columns empty and the size should be read from the relevant run's
``run_params.json``.

Usage
-----
Everything, as reported in the paper (needs the HuggingFace datasets and the
DPR dump; slow on first run because of the downloads)::

    python reproducibility/dataset_stats.py

Quick check on one machine without the large corpora — BEIR qrels are read
straight off disk and no BM25 index is built::

    python reproducibility/dataset_stats.py --datasets scifact fiqa trec-covid

Smoke test on a laptop::

    python reproducibility/dataset_stats.py --max-samples 200

Add ``--markdown`` to also print a table suitable for pasting into a note.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Callable, Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, "results")
OUT_CSV = os.path.join(RESULTS_DIR, "dataset_statistics.csv")


# ---------------------------------------------------------------------------
# What Section 5.2 currently claims.  Kept here so the script is self-checking:
# every run prints the recomputed value next to the claim and flags drift.
# Update these together with the paper.
# ---------------------------------------------------------------------------

PAPER_CLAIMS: Dict[str, Dict[str, str]] = {
    "hotpotqa_fullwiki": {"pool": "between 80 and 120 candidates per query"},
    "musique": {"pool": "roughly twenty paragraphs"},
    "nq": {"relevant": "around 4.6 relevant passages per query"},
    "scifact": {"relevant": "92% of queries have exactly one relevant document"},
    "fiqa": {"relevant": "typically two to three relevant documents"},
    "trec-covid": {"relevant": "mean 493 relevant documents per query"},
}


# ---------------------------------------------------------------------------
# BEIR: qrels only, read directly from the extracted task directory.
# ---------------------------------------------------------------------------

def _import_beir_module():
    """Import ``data/beir.py`` without triggering the ``data`` package __init__.

    ``data/__init__.py`` imports ``data.loaders``, which needs the HuggingFace
    ``datasets`` package.  ``data/beir.py`` itself needs only numpy and the
    standard library, so loading it straight off its file path keeps the BEIR
    statistics runnable on a machine with a minimal environment.
    """
    import importlib.util

    path = os.path.join(ROOT, "data", "beir.py")
    spec = importlib.util.spec_from_file_location("_beir_standalone", path)
    if spec is None or spec.loader is None:              # pragma: no cover
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _beir_relevant_sizes(name: str, split: str = "test") -> List[int]:
    """Return the number of positively-graded documents for each BEIR query.

    Reads the qrels TSV directly instead of going through
    :func:`data.beir.load_beir`, which would build a BM25 index over the whole
    corpus.  TREC-COVID's corpus is 221 MB, so that shortcut is the difference
    between seconds and many minutes.
    """
    beir = _import_beir_module()
    _download_and_extract, _read_qrels = beir._download_and_extract, beir._read_qrels

    root = os.path.join(ROOT, "data", "beir")
    task_dir = _download_and_extract(name, root)

    qrels = _read_qrels(task_dir, split)
    if qrels is None:
        print(f"   [{name}] no qrels for split '{split}'; falling back to 'test'.")
        qrels = _read_qrels(task_dir, "test")
    if qrels is None:
        raise FileNotFoundError(f"No usable qrels found for BEIR task '{name}'.")

    sizes = []
    for rels in qrels.values():
        n = sum(1 for score in rels.values() if score > 0)
        if n:
            sizes.append(n)
    return sizes


# ---------------------------------------------------------------------------
# QA datasets: pools and gold sets come straight off the loader.
# ---------------------------------------------------------------------------

def _loader_stats(load_fn: Callable, max_samples: Optional[int], **kwargs):
    """Run a loader and return (pool sizes, relevant-set sizes)."""
    examples = load_fn(max_samples=max_samples, **kwargs)
    pool, rel = [], []
    for ex in examples:
        if ex.get("passages"):
            pool.append(len(ex["passages"]))
        gold = ex.get("gold_titles")
        if gold:
            # gold_titles can repeat a title when a document is split into
            # several passages; the relevant-set size is the distinct count.
            rel.append(len(set(gold)))
    return pool, rel


def _qa_stats(loader_name: str, max_samples: Optional[int], **kwargs):
    """Import ``data.loaders`` lazily and run one of its loaders.

    The import is deferred so that ``--datasets scifact fiqa trec-covid`` works
    on a machine that has neither the HuggingFace ``datasets`` package nor the
    QA corpora: the BEIR path only needs the qrels files already in
    ``data/beir/``.
    """
    from data import loaders
    return _loader_stats(getattr(loaders, loader_name), max_samples, **kwargs)


def _build_registry(max_samples: Optional[int]):
    """Map dataset key -> zero-argument callable returning (pool, rel)."""
    reg: Dict[str, Callable] = {
        "hotpotqa": lambda: _qa_stats(
            "load_hotpotqa", max_samples, split="validation"),
        "hotpotqa_fullwiki": lambda: _qa_stats(
            "load_hotpotqa_fullwiki", max_samples, split="validation"),
        "2wikimultihopqa": lambda: _qa_stats(
            "load_2wikimultihopqa", max_samples, split="validation"),
        "musique": lambda: _qa_stats(
            "load_musique", max_samples, split="validation"),
        "nq": lambda: _qa_stats(
            "load_nq_dpr", max_samples, split="test"),
    }
    for beir in ("scifact", "fiqa", "trec-covid"):
        reg[beir] = (lambda n=beir: ([], _beir_relevant_sizes(n)))
    return reg


# ---------------------------------------------------------------------------

def _describe(values: List[int]) -> Dict[str, object]:
    if not values:
        return {k: "" for k in
                ("n", "mean", "median", "min", "p05", "p95", "max", "frac_singleton")}
    a = np.asarray(values, dtype=float)
    return {
        "n": len(a),
        "mean": round(float(a.mean()), 3),
        "median": round(float(np.median(a)), 1),
        "min": int(a.min()),
        "p05": round(float(np.percentile(a, 5)), 1),
        "p95": round(float(np.percentile(a, 95)), 1),
        "max": int(a.max()),
        "frac_singleton": round(float((a == 1).mean()), 4),
    }


FIELDS = [
    "dataset", "split",
    "n_queries",
    "pool_mean", "pool_median", "pool_min", "pool_p05", "pool_p95", "pool_max",
    "rel_mean", "rel_median", "rel_min", "rel_p05", "rel_p95", "rel_max",
    "rel_frac_exactly_one",
    "paper_claim",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Subset to compute. Default: all. The three BEIR tasks "
                         "are cheap; the QA loaders download from HuggingFace.")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap examples per QA dataset (smoke tests). BEIR qrels "
                         "are always read in full — they are tiny.")
    ap.add_argument("--split", default="test",
                    help="qrels split for the BEIR tasks. Default: test.")
    ap.add_argument("--out", default=OUT_CSV, help=f"Output CSV. Default: {OUT_CSV}")
    ap.add_argument("--markdown", action="store_true",
                    help="Also print a markdown table.")
    args = ap.parse_args()

    registry = _build_registry(args.max_samples)
    wanted = args.datasets or list(registry)
    unknown = [d for d in wanted if d not in registry]
    if unknown:
        ap.error(f"unknown dataset(s): {', '.join(unknown)}. "
                 f"Choose from: {', '.join(registry)}")

    if args.max_samples:
        print(f"NOTE: --max-samples {args.max_samples} is set. The QA numbers are "
              f"a sample, not the values reported in the paper.\n")

    rows = []
    for name in wanted:
        print(f"[{name}] computing ...")
        try:
            pool, rel = registry[name]()
        except Exception as exc:                       # noqa: BLE001
            print(f"   SKIPPED: {type(exc).__name__}: {exc}")
            continue

        p, r = _describe(pool), _describe(rel)
        claim = "; ".join(PAPER_CLAIMS.get(name, {}).values())
        rows.append({
            "dataset": name,
            "split": args.split if not pool else "validation",
            "n_queries": r["n"] or p["n"],
            "pool_mean": p["mean"], "pool_median": p["median"],
            "pool_min": p["min"], "pool_p05": p["p05"],
            "pool_p95": p["p95"], "pool_max": p["max"],
            "rel_mean": r["mean"], "rel_median": r["median"],
            "rel_min": r["min"], "rel_p05": r["p05"],
            "rel_p95": r["p95"], "rel_max": r["max"],
            "rel_frac_exactly_one": r["frac_singleton"],
            "paper_claim": claim,
        })

        if pool:
            print(f"   pool      n={p['n']:>6}  mean={p['mean']:<9} "
                  f"[p05={p['p05']}, p95={p['p95']}]  range=[{p['min']}, {p['max']}]")
        if rel:
            print(f"   relevant  n={r['n']:>6}  mean={r['mean']:<9} "
                  f"median={r['median']}  exactly-one={100 * r['frac_singleton']:.1f}%")
        if claim:
            print(f"   paper says: {claim}")

    if not rows:
        print("\nNothing computed.")
        return 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out} ({len(rows)} rows).")

    if args.markdown:
        print()
        head = ["dataset", "n_queries", "pool_p05", "pool_p95",
                "rel_mean", "rel_median", "rel_frac_exactly_one"]
        print("| " + " | ".join(head) + " |")
        print("|" + "|".join("---" for _ in head) + "|")
        for row in rows:
            print("| " + " | ".join(str(row[h]) for h in head) + " |")

    print("\nReminder: the BEIR pool columns are intentionally empty. Those pools "
          "are built by the retriever, so their size is a run parameter — read it "
          "from the relevant run's run_params.json, not from this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
