"""
Redundancy-injection and oracle-headroom experiments.

This module complements evaluate.py with two controlled experiments that test
*when* geometric diversification pays off, rather than whether it pays off on
a fixed benchmark:

Experiment 1 — redundancy injection (``--experiment redundancy``)
    Candidate pools drawn from deduplicated corpora (HotpotQA paragraphs,
    Wikipedia articles) contain almost no near-duplicates, which is the
    regime where diversification has nothing to remove.  Production RAG
    pipelines, in contrast, chunk documents into overlapping windows and
    index multiple near-identical versions of the same content.  This
    experiment interpolates between the two regimes: it injects a controlled
    fraction rho of near-duplicate passages into each candidate pool and
    measures every reranker at each redundancy level.  Duplicates inherit
    the title of their source passage, so coverage-style metrics
    (Recall over distinct golds, S-Recall, alpha-NDCG) are unaffected by
    retrieving extra copies — exactly the cost model in which redundancy
    wastes context slots.

Experiment 1b — natural redundancy via chunk overlap (``--experiment chunking``)
    The injection experiment controls redundancy synthetically.  This
    experiment creates it the way production pipelines actually do: every
    passage is re-chunked into sliding word windows (``--chunk_window``)
    whose overlap is swept over ``--overlap_grid``.  No text is perturbed or
    copied; redundancy arises from genuinely overlapping content.  Realized
    redundancy is reported through the same PoolRedundancy measurement as
    Experiment 1, so the natural and synthetic sweeps share one x-axis.

Experiment 2 — per-query oracle headroom (``--experiment oracle``)
    DF-RAG (Khan et al. 2026) reports a large gap between a fixed
    diversity level and a per-query oracle that selects the optimal level
    using ground truth, in a chunked long-context setting.  This experiment
    measures the same headroom in our setting: for each query the margin
    alpha (RNG-Score, Seg-Score) and the trade-off lambda (MMR) are swept,
    and the per-query best value of the objective metric is recorded.
    Comparing  kNN  vs  validation-tuned fixed parameter  vs  per-query
    oracle quantifies how much any query-adaptive diversification policy
    could possibly gain on the dataset.

Both experiments are retrieval-only (no generation) and run on CPU with the
default settings.  Results are written to a timestamped subdirectory of
``results/`` together with run_params.json, following the conventions of the
other evaluation scripts.

Usage
-----
::

    # Quick smoke test (CPU, ~100 examples)
    python evaluate_redundancy.py --max_samples 100 --experiment both

    # Redundancy sweep on the MDR fullwiki pools with a stronger encoder
    python evaluate_redundancy.py --dataset hotpotqa_fullwiki \
        --encoder_model bge-m3 --device cuda --max_samples all \
        --experiment redundancy

    # Oracle headroom only, finer alpha grid
    python evaluate_redundancy.py --experiment oracle \
        --alpha_grid -0.3 -0.2 -0.1 -0.05 0.0 0.05 0.1 0.2 0.3

    # Natural redundancy through overlapping chunking (no synthetic copies)
    python evaluate_redundancy.py --experiment chunking \
        --dataset hotpotqa_fullwiki --encoder_model bge-m3 --device cuda \
        --max_samples all --overlap_grid 0.0 0.25 0.5 0.75

    # BEIR replication of the injection sweep (SciFact, laptop-friendly)
    python evaluate_redundancy.py --experiment both --dataset scifact \
        --split test --max_samples all --top_k 10
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.beir import load_fiqa, load_scifact, load_trec_covid
from data.loaders import (
    load_2wikimultihopqa,
    load_hotpotqa,
    load_hotpotqa_fullwiki,
    load_musique,
    load_nq_dpr,
    load_squad,
)
from evaluation.metrics import (
    alpha_ndcg_at_k,
    avg_pairwise_distance,
    err_ia_at_k,
    exact_match,
    f1_score_single,
    gold_recall,
    hallucination_rate,
    mrr,
    ndcg_at_k,
    subtopic_coverage_sets,
    subtopic_recall_at_k,
    vendi_score,
)
from generation.generator import load_generator
from retrieval.precompute import _format_passage, encode_queries_and_passages
from retrieval.rerankers import (
    rerank_greedy_dpp,
    rerank_knn,
    rerank_maxmin,
    rerank_mmr,
    rerank_rng_score,
    rerank_rng_score2,
)
from retrieval.retriever import DenseRetriever
from run_utils import (
    ENCODER_ALIASES,
    GENERATOR_ALIASES,
    attach_run_params,
    default_encoder,
    default_generator,
    int_or_all,
    make_run_dir,
    normalize_device,
    resolve_model,
    save_csv,
)

try:
    from scipy.stats import wilcoxon as _wilcoxon
except ImportError:  # pragma: no cover
    _wilcoxon = None


LOADERS: Dict[str, Callable] = {
    "hotpotqa": load_hotpotqa,
    "hotpotqa_fullwiki": load_hotpotqa_fullwiki,
    "2wikimultihopqa": load_2wikimultihopqa,
    "musique": load_musique,
    "squad": load_squad,
    # NQ-Open with DPR pre-retrieved top-100 candidate pools (single-hop
    # control; passages attached, no FAISS build needed). See load_dpr.
    "nq": load_nq_dpr,
    # BEIR tasks with BM25-built candidate pools (see data/beir.py).
    "scifact": load_scifact,
    "fiqa": load_fiqa,
    "trec-covid": load_trec_covid,
}

# Alpha grid extended past 0.3 because the fullwiki run tuned to the grid
# edge; rho grid refined at the low end to localize the crossover rho*:
# the coarse sweep showed the entire regime flip happening inside (0, 0.25),
# with the curves flat from 0.25 onward, so the resolution goes where the
# action is and the stable tail keeps only {0.25, 0.5, 1.0}.
DEFAULT_ALPHA_GRID = [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3,
                      0.5, 1.0]
DEFAULT_LAMBDA_GRID = [0.3, 0.5, 0.7, 0.9]
DEFAULT_RHO_GRID = [0.0, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0]
DEFAULT_OVERLAP_GRID = [0.0, 0.25, 0.5, 0.75]

# Cosine-similarity threshold above which a pair of passages is counted as a
# near-duplicate when measuring pool redundancy.
NEAR_DUP_SIM = 0.95


# ---------------------------------------------------------------------------
# Near-duplicate construction
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    """Split *text* into sentences with a lightweight punctuation heuristic."""
    return [s for s in _SENT_SPLIT.split(text.strip()) if s]


def _perturb_text(text: str, rng: np.random.Generator, noise: str) -> str:
    """Return a near-duplicate of *text*.

    Parameters
    ----------
    text : str
        Source passage text.
    rng : np.random.Generator
        Seeded generator for reproducibility.
    noise : str
        ``"exact"``  — verbatim copy (identical embedding; degenerate ties).
        ``"light"``  — sentence-order shuffle (default; mimics re-chunked or
        re-crawled versions of the same content).
        ``"heavy"``  — sentence shuffle plus one dropped sentence when the
        passage has at least three (mimics partially overlapping chunks).
    """
    if noise == "exact":
        return text
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return text
    sentences = list(rng.permutation(sentences))
    if noise == "heavy" and len(sentences) >= 3:
        drop = int(rng.integers(len(sentences)))
        sentences = sentences[:drop] + sentences[drop + 1:]
    return " ".join(sentences)


def inject_duplicates(
    passages: List[Dict],
    gold_titles: List[str],
    rate: float,
    rng: np.random.Generator,
    noise: str = "light",
    target: str = "mixed",
) -> List[Dict]:
    """Return a new passage pool with near-duplicates appended.

    The number of injected duplicates is ``round(rate * len(passages))``.
    Each duplicate keeps the *title of its source passage*, so that distinct-
    coverage metrics treat extra copies as redundant rather than as new
    relevant evidence.

    Parameters
    ----------
    passages : list of dict
        Original candidate pool ({"title", "text"} dicts).
    gold_titles : list of str
        Titles of the gold supporting passages; used by the ``gold`` and
        ``mixed`` targeting modes.
    rate : float
        Redundancy level rho >= 0 (duplicates as a fraction of pool size).
    rng : np.random.Generator
        Seeded generator.
    noise : str
        Perturbation level passed to :func:`_perturb_text`.
    target : str
        ``"gold"``   — duplicate gold passages only (cycling);
        ``"random"`` — duplicate uniformly sampled passages;
        ``"mixed"``  — half gold-sourced, half random (default).  Redundancy
        in production pools concentrates around highly relevant content, so
        purely random duplication understates the effect.
    """
    n_dup = int(round(rate * len(passages)))
    if n_dup == 0:
        return list(passages)

    gold_set = {t.lower().strip() for t in (gold_titles or [])}
    gold_idx = [
        i for i, p in enumerate(passages)
        if (p.get("title") or "").lower().strip() in gold_set
    ]
    all_idx = list(range(len(passages)))

    if target == "gold" and gold_idx:
        sources = [gold_idx[i % len(gold_idx)] for i in range(n_dup)]
    elif target == "random" or not gold_idx:
        sources = list(rng.choice(all_idx, size=n_dup, replace=True))
    elif target == "mixed":
        n_gold = n_dup // 2
        sources = [gold_idx[i % len(gold_idx)] for i in range(n_gold)]
        sources += list(rng.choice(all_idx, size=n_dup - n_gold, replace=True))
    else:
        raise ValueError(f"Unknown target '{target}'. Choose gold, random, or mixed.")

    pool = list(passages)
    for src in sources:
        source = passages[int(src)]
        pool.append(
            {
                "title": source["title"],            # same title: a copy, not new evidence
                "text": _perturb_text(source["text"], rng, noise),
                "is_duplicate": True,
            }
        )
    return pool


def pool_redundancy(p_embs: np.ndarray) -> float:
    """Fraction of passage pairs whose cosine similarity exceeds NEAR_DUP_SIM.

    Quantifies the controlling variable of Experiment 1 directly on the
    embedded pool, so that natural and injected redundancy are measured on
    the same scale.
    """
    n = len(p_embs)
    if n < 2:
        return 0.0
    sims = p_embs @ p_embs.T
    upper = np.triu_indices(n, k=1)
    return float(np.mean(sims[upper] > NEAR_DUP_SIM))


# ---------------------------------------------------------------------------
# Selection and evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_selection(
    selected: Sequence[int],
    passages: List[Dict],
    p_embs: np.ndarray,
    gold_titles: List[str],
    k: int,
    single_subtopic: bool = False,
) -> Dict[str, float]:
    """Compute the standard metric family for one selection on one pool.

    Because injected duplicates share the title of their source passage,
    rank-based metrics that credit every relevant occurrence (NDCG, MRR)
    would double-count copies and can exceed 1.  Repeated titles are
    therefore masked after their first occurrence for NDCG and MRR, so that
    only the first copy of a gold passage earns rank credit.  Set-based
    metrics (Recall, S-Recall) and the redundancy-aware alpha-NDCG are
    unaffected by construction.

    When ``single_subtopic`` is True, all gold passages collapse to one
    subtopic (see :func:`subtopic_coverage_sets`), so the intent-aware
    metrics (S-Recall, alpha-NDCG, ERR-IA) treat the query as single-hop:
    use this for NQ-Open, whose DPR gold set is many copies of the one
    answer rather than several distinct pieces of evidence.
    """
    titles = [passages[i].get("title") or "" for i in selected]
    seen: set = set()
    titles_first = []
    for j, t in enumerate(titles):
        key = t.lower().strip()
        titles_first.append(t if key not in seen else f"__dup_{j}__")
        seen.add(key)
    coverage = subtopic_coverage_sets(passages, gold_titles=gold_titles,
                                      single_subtopic=single_subtopic)
    sel_embs = p_embs[list(selected)]
    return {
        "Recall@k": gold_recall(titles, gold_titles),
        "NDCG@k": ndcg_at_k(titles_first, gold_titles, k),
        "MRR": mrr(titles_first, gold_titles),
        "alpha-NDCG@k": alpha_ndcg_at_k(selected, coverage, k),
        "S-Recall@k": subtopic_recall_at_k(selected, coverage, k),
        "ERR-IA@k": err_ia_at_k(selected, coverage, k),
        "APD": avg_pairwise_distance(sel_embs),
        "Vendi": vendi_score(sel_embs),
    }


def method_selections(
    p_embs: np.ndarray,
    q_emb: np.ndarray,
    k: int,
    metric: str,
    alpha_grid: Sequence[float],
    lambda_grid: Sequence[float],
) -> Dict[str, List[int]]:
    """Run every reranker on one pool and return {method_name: selection}.

    Parameterised methods appear once per grid value, e.g. ``MMR(0.5)`` and
    ``RNG(0.05)``; the grids are resolved into fixed and oracle operating
    points later, at the aggregation stage.
    """
    sims = p_embs @ q_emb
    out: Dict[str, List[int]] = {"kNN": rerank_knn(sims, k)}
    for lam in lambda_grid:
        out[f"MMR({lam:g})"] = rerank_mmr(p_embs, q_emb, sims, k, lambda_=lam)
    out["Maxmin"] = rerank_maxmin(p_embs, sims, k)
    out["Greedy-DPP"] = rerank_greedy_dpp(p_embs, sims, k)
    for a in alpha_grid:
        out[f"RNG({a:g})"] = rerank_rng_score(p_embs, q_emb, k, alpha=a, metric=metric)
        out[f"Seg({a:g})"] = rerank_rng_score2(p_embs, q_emb, k, alpha=a, metric=metric)
    return out


def _grid_members(prefix: str, grid: Sequence[float]) -> List[str]:
    return [f"{prefix}({v:g})" for v in grid]


def _pad_rows(rows: List[Dict]) -> List[Dict]:
    """Give every row the union of all columns (missing values become "").

    ``run_utils.save_csv`` derives the CSV header from the first row and
    raises on rows containing additional keys; summary tables built here mix
    method rows, significance columns, and headroom rows, so the key sets
    differ across rows.
    """
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return [{c: row.get(c, "") for c in columns} for row in rows]


# ---------------------------------------------------------------------------
# Experiment 1 — redundancy injection
# ---------------------------------------------------------------------------

def _run_pool_sweep(
    examples: List[Dict],
    encoder: DenseRetriever,
    cfg: Dict,
    run_dir: str,
    levels: Sequence[float],
    make_pools: Callable[[float], List[List[Dict]]],
    level_col: str,
    file_tag: str,
    generator=None,
) -> None:
    """Shared sweep driver: for each level, build pools with *make_pools*,
    encode, truncate to top-m, run every reranker, evaluate, tune
    parameterised methods on the validation fraction and report test means
    with Wilcoxon significance against kNN. Used by both the synthetic
    injection sweep (level = rho) and the natural chunk-overlap sweep
    (level = overlap).

    Encoding is cached across levels: queries are encoded once per sweep,
    and passage embeddings are kept in a text-keyed cache so that each level
    only encodes texts not seen before (the original passages recur at every
    level; with exact-copy noise even the duplicates are cache hits).
    Non-original entries are evicted after each level, so the steady-state
    cache holds exactly one embedding per original passage. Disable with
    --no_encode_cache if memory is tighter than encoder throughput."""
    k = cfg["top_k"]
    m = cfg["top_m"]
    metric = cfg["metric"]
    alpha_grid = cfg["alpha_grid"]
    lambda_grid = cfg["lambda_grid"]
    objective = cfg["objective"]
    use_cache = bool(cfg.get("encode_cache", True))

    summary_rows: List[Dict] = []
    per_query_rows: List[Dict] = []
    gen_rows: List[Dict] = []

    # Generation subset: a seeded sample of examples on which answers are
    # generated for the selected gen_methods (decorrelated from the split
    # permutation by offsetting the seed). The decision rule's answer quality
    # is reconstructed offline in analyze_regimes.py from the per-query EM/F1
    # of kNN and the fallback diversifier plus the kNN Vendi trigger.
    gen_methods = list(cfg.get("gen_methods") or [])
    if generator is not None and gen_methods:
        gn = cfg.get("gen_max_samples") or len(examples)
        gn = len(examples) if gn == "all" else min(int(gn), len(examples))
        gen_idx = set(
            np.random.default_rng(cfg["seed"] + 1)
            .permutation(len(examples))[:gn].tolist()
        )
        print(f"   Generation on for {gen_methods} over {len(gen_idx)} queries "
              f"per level (reader {cfg.get('generator_model')}).")
    else:
        gen_idx = set()

    questions = [ex["question"] for ex in examples]
    q_embs: Optional[np.ndarray] = None
    cache: Dict[str, np.ndarray] = {}
    protected = {_format_passage(p) for ex in examples for p in ex["passages"]}

    for level in levels:
        print(f"\n── {level_col} = {level:g} ──")
        pools = make_pools(level)
        gen_jobs: Dict[str, List] = {}

        if not use_cache:
            q_embs, p_flat, p_off = encode_queries_and_passages(
                encoder, questions, pools)
        else:
            if q_embs is None:
                print(f"   Encoding {len(questions)} queries …")
                q_embs = encoder.encode(questions, normalize=True,
                                        show_progress=True)
            p_off = np.zeros(len(questions) + 1, dtype=np.int64)
            for i, pool in enumerate(pools):
                p_off[i + 1] = p_off[i] + len(pool)
            flat_texts = [_format_passage(p) for pool in pools for p in pool]
            unique_texts = list(dict.fromkeys(flat_texts))
            miss = [t for t in unique_texts if t not in cache]
            n_cached = len(unique_texts) - len(miss)
            n_repeats = len(flat_texts) - len(unique_texts)
            print(f"   {len(flat_texts)} passage instances: "
                  f"{len(miss)} unique to encode, "
                  f"{n_repeats} repeated within this level, "
                  f"{n_cached} cached from previous levels …")
            if miss:
                new_embs = encoder.encode(miss, normalize=True,
                                          show_progress=True)
                for t, emb in zip(miss, new_embs):
                    cache[t] = emb
            p_flat = np.stack([cache[t] for t in flat_texts])

        per_method: Dict[str, List[Dict[str, float]]] = {}
        redundancies: List[float] = []

        for i, ex in enumerate(examples):
            s, e = int(p_off[i]), int(p_off[i + 1])
            q_emb = q_embs[i]
            p_embs = p_flat[s:e]
            pool = pools[i]

            # First-stage truncation: top-m by query similarity, as in evaluate.py.
            sims = p_embs @ q_emb
            keep = np.argsort(-sims)[: min(m, len(pool))]
            pool_m = [pool[j] for j in keep]
            p_embs_m = p_embs[keep]

            redundancies.append(pool_redundancy(p_embs_m))

            selections = method_selections(
                p_embs_m, q_emb, k, metric, alpha_grid, lambda_grid
            )
            for name, sel in selections.items():
                res = evaluate_selection(sel, pool_m, p_embs_m, ex["gold_titles"] or [], k,
                                         single_subtopic=cfg.get("single_subtopic", False))
                per_method.setdefault(name, []).append(res)
                per_query_rows.append(
                    {level_col: level, "qid": ex["id"], "Method": name,
                     "PoolRedundancy": round(redundancies[-1], 4), **res}
                )

            if i in gen_idx:
                answers = ex.get("answers") or []
                for gm in gen_methods:
                    if gm in selections:
                        sel_passages = [pool_m[j] for j in selections[gm]]
                        gen_jobs.setdefault(gm, []).append(
                            (ex["id"], ex["question"], answers, sel_passages))

        # Answer generation for the sampled queries at this level.
        if generator is not None and gen_jobs:
            for gm, jobs in gen_jobs.items():
                preds = generator.generate_batch(
                    [j[1] for j in jobs], [j[3] for j in jobs],
                    batch_size=cfg.get("batch_size", 32))
                for (qid, _q, answers, sel_passages), pred in zip(jobs, preds):
                    em = max((exact_match(pred, a) for a in answers), default=0.0)
                    f1 = max((f1_score_single(pred, a) for a in answers), default=0.0)
                    hall = hallucination_rate(pred, [p["text"] for p in sel_passages])
                    gen_rows.append(
                        {level_col: level, "qid": qid, "Method": gm,
                         "EM": round(em, 4), "F1": round(f1, 4),
                         "Halluc": round(hall, 4)})

        mean_red = float(np.mean(redundancies))

        # Resolve parameterised methods into a fixed validation-tuned point.
        n_val = max(1, int(cfg["val_fraction"] * len(examples)))
        order = np.random.default_rng(cfg["seed"]).permutation(len(examples))
        val_ids, test_ids = order[:n_val], order[n_val:]
        if len(test_ids) == 0:
            test_ids = val_ids

        def _mean(name: str, ids: np.ndarray, key: str) -> float:
            vals = [per_method[name][int(i)][key] for i in ids]
            return float(np.mean(vals))

        report: Dict[str, str] = {"kNN": "kNN", "Maxmin": "Maxmin",
                                  "Greedy-DPP": "Greedy-DPP"}
        for prefix, grid in [("MMR", lambda_grid), ("RNG", alpha_grid), ("Seg", alpha_grid)]:
            members = _grid_members(prefix, grid)
            best = max(members, key=lambda nm: _mean(nm, val_ids, objective))
            report[f"{prefix}* [{best}]"] = best

        knn_test = [per_method["kNN"][int(i)][objective] for i in test_ids]
        for label, name in report.items():
            row: Dict[str, object] = {level_col: level,
                                      "PoolRedundancy": round(mean_red, 4),
                                      "Method": label}
            for key in per_method[name][0]:
                row[key] = round(_mean(name, test_ids, key), 4)
            if _wilcoxon is not None and name != "kNN":
                vals = [per_method[name][int(i)][objective] for i in test_ids]
                diffs = np.array(vals) - np.array(knn_test)
                if np.any(diffs != 0.0):
                    row[f"p({objective} vs kNN)"] = round(
                        float(_wilcoxon(vals, knn_test).pvalue), 5
                    )
                else:
                    row[f"p({objective} vs kNN)"] = 1.0
            summary_rows.append(row)

        if use_cache:
            # Evict per-level texts (perturbed duplicates, level-specific
            # chunks) so the cache never grows beyond the original passages.
            for t in [tt for tt in cache if tt not in protected]:
                del cache[t]

    summary_rows = attach_run_params(_pad_rows(summary_rows), cfg)
    save_csv(summary_rows,
             os.path.join(run_dir, f"results_{file_tag}_summary.csv"))
    save_csv(per_query_rows,
             os.path.join(run_dir, f"results_{file_tag}_per_query.csv"))
    if gen_rows:
        save_csv(attach_run_params(_pad_rows(gen_rows), cfg),
                 os.path.join(run_dir, f"results_{file_tag}_gen_per_query.csv"))
        print(f"   Saved generation EM/F1 for {len(gen_rows)} "
              f"(level, query, method) rows.")


def run_redundancy_experiment(
    examples: List[Dict],
    encoder: DenseRetriever,
    cfg: Dict,
    run_dir: str,
    generator=None,
) -> None:
    """Sweep the injected-redundancy level rho and evaluate every reranker.

    For each rho the full pipeline is repeated from the text level: duplicate
    injection, encoding, top-m pool truncation, reranking, and evaluation.
    Per-query rows are saved so paired significance tests against kNN can be
    computed (Wilcoxon signed-rank, reported in the summary when scipy is
    available).
    """
    def make_pools(rho: float) -> List[List[Dict]]:
        rng = np.random.default_rng(cfg["seed"])
        return [
            inject_duplicates(
                ex["passages"], ex["gold_titles"] or [], rho, rng,
                noise=cfg["dup_noise"], target=cfg["dup_target"],
            )
            for ex in examples
        ]

    _run_pool_sweep(examples, encoder, cfg, run_dir, cfg["rho_grid"],
                    make_pools, level_col="rho", file_tag="redundancy",
                    generator=generator)


# ---------------------------------------------------------------------------
# Experiment 1b — natural redundancy through overlapping chunking
# ---------------------------------------------------------------------------

def _chunk_words(text: str, window: int, stride: int) -> List[str]:
    """Split *text* into sliding windows of *window* words with the given
    *stride*. Trailing fragments shorter than half a window are dropped
    (unless the text fits in a single window)."""
    words = text.split()
    if len(words) <= window:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(words):
        piece = words[start:start + window]
        if len(piece) < max(1, window // 2) and chunks:
            break
        chunks.append(" ".join(piece))
        if start + window >= len(words):
            break
        start += stride
    return chunks


def chunk_pool(passages: List[Dict], window: int, stride: int) -> List[Dict]:
    """Re-chunk every passage of a pool into sliding word windows.

    Every chunk inherits the *title* of its source passage, so title-based
    coverage metrics treat multiple chunks of the same passage as redundant
    copies of one piece of evidence rather than as new evidence — the same
    cost model as the injection experiment. Note that the metrics are
    title-based: a gold-titled chunk counts as gold coverage even if the
    specific window does not contain the answer span, which matches the
    retrieval-only scope of these experiments.
    """
    out: List[Dict] = []
    for p in passages:
        for piece in _chunk_words(p["text"], window, stride):
            out.append({"title": p["title"], "text": piece,
                        "is_chunk": True})
    return out


def run_chunking_experiment(
    examples: List[Dict],
    encoder: DenseRetriever,
    cfg: Dict,
    run_dir: str,
    generator=None,
) -> None:
    """Natural-redundancy counterpart of the injection experiment.

    Instead of appending perturbed copies, every pool is re-chunked into
    sliding word windows whose overlap is swept over ``--overlap_grid``
    (stride = window * (1 - overlap)). This reproduces exactly how
    production RAG pipelines create redundancy — overlapping chunking of
    source documents — so no synthetic text perturbation is involved.
    The realized redundancy is reported through the same PoolRedundancy
    column (near-duplicate pair fraction on the embedded pool), placing the
    natural and injected sweeps on one measured x-axis.
    """
    window = cfg["chunk_window"]

    def make_pools(overlap: float) -> List[List[Dict]]:
        stride = max(1, int(round(window * (1.0 - overlap))))
        return [chunk_pool(ex["passages"], window, stride) for ex in examples]

    _run_pool_sweep(examples, encoder, cfg, run_dir, cfg["overlap_grid"],
                    make_pools, level_col="overlap", file_tag="chunking",
                    generator=generator)


# ---------------------------------------------------------------------------
# Experiment 2 — per-query oracle headroom
# ---------------------------------------------------------------------------

def run_oracle_experiment(
    examples: List[Dict],
    encoder: DenseRetriever,
    cfg: Dict,
    run_dir: str,
) -> None:
    """Quantify the gap between fixed and per-query-optimal diversification.

    Following the oracle construction of DF-RAG, the per-query oracle selects,
    for each query, the grid value whose selection maximises the objective
    metric computed against the ground truth.  The reported quantities are::

        headroom(method) = mean_q max_param objective  -  mean_q objective(fixed param*)

    where param* is tuned on a held-out validation fraction.  A small headroom
    over kNN bounds the achievable gain of *any* query-adaptive policy
    (including LLM-based planner/evaluator pipelines) on the dataset, because
    the oracle has access to the ground truth.
    """
    k = cfg["top_k"]
    m = cfg["top_m"]
    metric = cfg["metric"]
    alpha_grid = cfg["alpha_grid"]
    lambda_grid = cfg["lambda_grid"]
    objective = cfg["objective"]

    questions = [ex["question"] for ex in examples]
    pools = [ex["passages"] for ex in examples]
    q_embs, p_flat, p_off = encode_queries_and_passages(encoder, questions, pools)

    per_method: Dict[str, List[Dict[str, float]]] = {}
    per_query_rows: List[Dict] = []

    for i, ex in enumerate(examples):
        s, e = int(p_off[i]), int(p_off[i + 1])
        q_emb = q_embs[i]
        p_embs = p_flat[s:e]
        pool = pools[i]

        sims = p_embs @ q_emb
        keep = np.argsort(-sims)[: min(m, len(pool))]
        pool_m = [pool[j] for j in keep]
        p_embs_m = p_embs[keep]

        selections = method_selections(p_embs_m, q_emb, k, metric, alpha_grid, lambda_grid)
        row: Dict[str, object] = {"qid": ex["id"]}
        for name, sel in selections.items():
            res = evaluate_selection(sel, pool_m, p_embs_m, ex["gold_titles"] or [], k,
                                     single_subtopic=cfg.get("single_subtopic", False))
            per_method.setdefault(name, []).append(res)
            row[f"{name}:{objective}"] = round(res[objective], 4)
        per_query_rows.append(row)

    n_val = max(1, int(cfg["val_fraction"] * len(examples)))
    order = np.random.default_rng(cfg["seed"]).permutation(len(examples))
    val_ids, test_ids = order[:n_val], order[n_val:]
    if len(test_ids) == 0:
        test_ids = val_ids

    def _mean(name: str, ids: np.ndarray, key: str) -> float:
        return float(np.mean([per_method[name][int(i)][key] for i in ids]))

    summary_rows: List[Dict] = []

    def _report(label: str, values_per_query: List[float],
                extra: Optional[Dict[str, float]] = None) -> None:
        test_vals = [values_per_query[int(i)] for i in test_ids]
        row = {"Method": label, objective: round(float(np.mean(test_vals)), 4)}
        if extra:
            row.update({kk: round(vv, 4) for kk, vv in extra.items()})
        summary_rows.append(row)

    # Baseline.
    knn_obj = [r[objective] for r in per_method["kNN"]]
    _report("kNN", knn_obj,
            {key: _mean("kNN", test_ids, key) for key in per_method["kNN"][0]
             if key != objective})

    for prefix, grid in [("MMR", lambda_grid), ("RNG", alpha_grid), ("Seg", alpha_grid)]:
        members = _grid_members(prefix, grid)

        # Fixed operating point tuned on the validation fraction.
        best = max(members, key=lambda nm: _mean(nm, val_ids, objective))
        fixed_obj = [r[objective] for r in per_method[best]]
        _report(f"{prefix} fixed [{best}]", fixed_obj,
                {key: _mean(best, test_ids, key) for key in per_method[best][0]
                 if key != objective})

        # Per-query oracle over the same grid.
        oracle_obj: List[float] = []
        oracle_choice: List[str] = []
        for qi in range(len(examples)):
            vals = {nm: per_method[nm][qi][objective] for nm in members}
            nm_best = max(vals, key=vals.get)
            oracle_obj.append(vals[nm_best])
            oracle_choice.append(nm_best)
        _report(f"{prefix} oracle (per-query)", oracle_obj)

        # Headroom rows: what adaptivity could add, and what diversification
        # adds over plain kNN even with oracle knowledge.
        test_o = np.array([oracle_obj[int(i)] for i in test_ids])
        test_f = np.array([fixed_obj[int(i)] for i in test_ids])
        test_k = np.array([knn_obj[int(i)] for i in test_ids])
        summary_rows.append({
            "Method": f"{prefix} headroom",
            objective: "",
            "oracle - fixed": round(float(np.mean(test_o - test_f)), 4),
            "oracle - kNN": round(float(np.mean(test_o - test_k)), 4),
            "% queries where oracle beats kNN":
                round(float(np.mean(test_o > test_k)) * 100.0, 1),
        })

        for qi, row in enumerate(per_query_rows):
            row[f"{prefix} oracle choice"] = oracle_choice[qi]

    summary_rows = attach_run_params(_pad_rows(summary_rows), cfg)
    save_csv(summary_rows, os.path.join(run_dir, "results_oracle_summary.csv"))
    save_csv(per_query_rows, os.path.join(run_dir, "results_oracle_per_query.csv"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Redundancy-injection and oracle-headroom experiments."
    )
    p.add_argument("--experiment",
                   choices=["redundancy", "oracle", "chunking", "both", "all"],
                   default="both",
                   help="'both' = redundancy + oracle (historical default); "
                        "'chunking' = natural redundancy via overlapping "
                        "windows; 'all' = all three.")
    p.add_argument("--dataset", choices=sorted(LOADERS), default="hotpotqa",
                   help="Dataset with attached candidate pools and gold titles. "
                        "hotpotqa_fullwiki uses the MDR top-100 pools (~652 MB "
                        "download on first use). nq is NQ-Open with DPR "
                        "pre-retrieved top-100 pools (auto-downloaded to "
                        "data/dpr/ on first use; use --split test). "
                        "scifact/fiqa/trec-covid are BEIR tasks with "
                        "BM25-built pools (use --split test).")
    p.add_argument("--split", default="validation")
    p.add_argument("--max_samples", type=int_or_all, default=100,
                   help="Number of examples, or 'all'.")
    p.add_argument("--encoder_model", default=None,
                   help="Encoder alias or HuggingFace ID (default: device-based).")
    p.add_argument(
        "--device",
        default="cpu",
        help='Device for model inference, e.g. "cpu", "cuda", "cuda:0" (default: cpu). '
             "When cuda is selected, bge-m3 is used as the default encoder instead of minilm.",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--top_m", type=int, default=100)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--metric", choices=["euclidean", "angular", "cosine"],
                   default="cosine")
    p.add_argument("--objective", default="alpha-NDCG@k",
                   choices=["alpha-NDCG@k", "Recall@k", "NDCG@k", "S-Recall@k"],
                   help="Metric used for validation tuning, oracle selection, "
                        "and significance tests.")
    p.add_argument("--alpha_grid", type=float, nargs="+", default=DEFAULT_ALPHA_GRID)
    p.add_argument("--lambda_grid", type=float, nargs="+", default=DEFAULT_LAMBDA_GRID)
    p.add_argument("--rho_grid", type=float, nargs="+", default=DEFAULT_RHO_GRID,
                   help="Injected-redundancy levels (duplicates / pool size).")
    p.add_argument("--overlap_grid", type=float, nargs="+",
                   default=DEFAULT_OVERLAP_GRID,
                   help="Chunk-overlap levels for --experiment chunking "
                        "(fraction of window shared by consecutive chunks).")
    p.add_argument("--chunk_window", type=int, default=60,
                   help="Chunk window size in words for --experiment chunking.")
    p.add_argument("--dup_noise", choices=["exact", "light", "heavy"],
                   default="light")
    p.add_argument("--dup_target", choices=["gold", "random", "mixed"],
                   default="mixed")
    p.add_argument("--single_subtopic", action="store_true",
                   help="Collapse all gold passages of a query to one "
                        "subtopic, so intent-aware metrics (S-Recall, "
                        "alpha-NDCG, ERR-IA) treat the query as single-hop. "
                        "Recommended for nq, whose DPR gold set is many "
                        "copies of the one answer (mean ~4.6 gold "
                        "titles/query) rather than distinct evidence.")
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--run_generation", action="store_true",
                   help="Also generate answers from each gen-method's top-k at "
                        "every sweep level and record per-query EM/F1/Halluc "
                        "(retrieval-only by default). Closes the loop from "
                        "S-Recall to answer quality; the decision rule's answer "
                        "quality is reconstructed offline by analyze_regimes.py.")
    p.add_argument("--generator_model", default=None,
                   help="Reader alias or HuggingFace ID (default: flan-t5-base "
                        "on cuda, flan-t5-small on cpu).")
    p.add_argument("--gen_methods", nargs="+", default=["kNN", "MMR(0.7)"],
                   help="Method names (as produced in the sweep) to generate "
                        "for; e.g. kNN MMR(0.7) RNG(-0.2). kNN and the fallback "
                        "diversifier are enough to reconstruct the rule.")
    p.add_argument("--gen_max_samples", type=int_or_all, default=1000,
                   help="Number of queries to generate for per level (or 'all').")
    p.add_argument("--no_encode_cache", action="store_true",
                   help="Disable cross-level embedding caching (re-encode "
                        "everything at every sweep level, as before). Use "
                        "when RAM is tighter than encoder throughput.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="results")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = normalize_device(args.device)
    encoder_name = resolve_model(
        args.encoder_model or default_encoder(device), ENCODER_ALIASES
    )
    generator_name = (
        resolve_model(args.generator_model or default_generator(device),
                      GENERATOR_ALIASES)
        if args.run_generation else None
    )

    cfg: Dict = {
        "script": "evaluate_redundancy.py",
        "experiment": args.experiment,
        "dataset": args.dataset,
        "split": args.split,
        "max_samples": args.max_samples,
        "encoder_model": encoder_name,
        "device": device,
        "top_m": args.top_m,
        "top_k": args.top_k,
        "metric": args.metric,
        "objective": args.objective,
        "alpha_grid": list(args.alpha_grid),
        "lambda_grid": list(args.lambda_grid),
        "rho_grid": list(args.rho_grid),
        "overlap_grid": list(args.overlap_grid),
        "chunk_window": args.chunk_window,
        "dup_noise": args.dup_noise,
        "dup_target": args.dup_target,
        "single_subtopic": args.single_subtopic,
        "val_fraction": args.val_fraction,
        "encode_cache": not args.no_encode_cache,
        "run_generation": args.run_generation,
        "generator_model": generator_name,
        "gen_methods": list(args.gen_methods) if args.run_generation else [],
        "gen_max_samples": args.gen_max_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }

    print(f"── Loading {args.dataset} ({args.split}) ──")
    examples = LOADERS[args.dataset](split=args.split, max_samples=args.max_samples)
    examples = [ex for ex in examples if ex["passages"] and ex["gold_titles"]]
    print(f"   {len(examples)} examples with pools and gold labels.")

    print(f"\n── Loading encoder: {encoder_name} ──")
    encoder = DenseRetriever(
        model_name=encoder_name, device=device, batch_size=args.batch_size
    )

    generator = None
    if generator_name is not None:
        print(f"\n── Loading reader: {generator_name} ──")
        generator = load_generator(generator_name, device=device)

    run_dir = make_run_dir(
        args.output_dir, f"redundancy_{args.dataset}", cfg
    )

    if args.experiment in ("redundancy", "both", "all"):
        run_redundancy_experiment(examples, encoder, cfg, run_dir,
                                  generator=generator)
    if args.experiment in ("chunking", "all"):
        run_chunking_experiment(examples, encoder, cfg, run_dir,
                                generator=generator)
    if args.experiment in ("oracle", "both", "all"):
        run_oracle_experiment(examples, encoder, cfg, run_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
