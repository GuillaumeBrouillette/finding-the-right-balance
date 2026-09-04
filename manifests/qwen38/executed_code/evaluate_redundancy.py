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
import csv
import hashlib
import json
import os
import re
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
    rerank_dedup,
    rerank_greedy_dpp,
    rerank_knn,
    rerank_maxmin,
    rerank_mmr,
    rerank_vendi_greedy,
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
from stats import (
    add_seed_arg,
    aggregate_seed_rows,
    as_float,
    paired_wilcoxon_p,
    resolve_seeds,
)


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
# -2.0 is the fallback sentinel: under cosine distance it deactivates every
# obstruction penalty (Prop. knnlimit), so the sweep grid contains the exact
# k-NN operating point by construction (-0.3 deviates from k-NN on ~1% of
# clean fullwiki queries and is an active operating point under injection).
DEFAULT_ALPHA_GRID = [-2.0, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3,
                      0.5, 1.0]
DEFAULT_LAMBDA_GRID = [0.3, 0.5, 0.7, 0.9]
DEFAULT_DEDUP_GRID = [0.85, 0.9, 0.95]
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


def relevant_set_size(gold_titles: Sequence[str], single_subtopic: bool) -> int:
    """Number of distinct relevant subtopics for a query.

    This is the gate variable for the decision rule: a single-hop query has
    one relevant subtopic (one piece of evidence), so diversification can only
    displace the answer; a multi-hop query has two or more distinct gold
    documents that genuinely need covering.  Counted from the query's gold
    set (pool-independent), matching the subtopic convention of
    :func:`evaluation.metrics.subtopic_coverage_sets`: distinct normalised
    gold titles, or 1 under ``single_subtopic`` (all golds collapse to one).
    """
    if not gold_titles:
        return 0
    if single_subtopic:
        return 1
    return len({t.lower().strip() for t in gold_titles})


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
    dedup_grid: Sequence[float] = DEFAULT_DEDUP_GRID,
) -> Dict[str, List[int]]:
    """Run every reranker on one pool and return {method_name: selection}.

    Parameterised methods appear once per grid value, e.g. ``MMR(0.5)`` and
    ``RNG(0.05)``; the grids are resolved into fixed and oracle operating
    points later, at the aggregation stage.
    """
    sims = p_embs @ q_emb
    out: Dict[str, List[int]] = {"kNN": rerank_knn(sims, k)}
    for t in dedup_grid:
        out[f"Dedup({t:g})"] = rerank_dedup(p_embs, sims, k, threshold=t)
    for lam in lambda_grid:
        out[f"MMR({lam:g})"] = rerank_mmr(p_embs, q_emb, sims, k, lambda_=lam)
    out["Maxmin"] = rerank_maxmin(p_embs, sims, k)
    out["Greedy-DPP"] = rerank_greedy_dpp(p_embs, sims, k)
    for lam in lambda_grid:
        out[f"VendiG({lam:g})"] = rerank_vendi_greedy(p_embs, sims, k, lambda_=lam)
    for a in alpha_grid:
        out[f"RNG({a:g})"] = rerank_rng_score(p_embs, q_emb, k, alpha=a, metric=metric)
        # Seg-Score excluded (2026-07-16)
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


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_reuse_source(path: str, cfg: Dict) -> Dict:
    """Validate an earlier generation CSV before reusing compatible rows."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise SystemExit(f"Generation reuse source not found: {path}")
    params_path = os.path.join(os.path.dirname(path), "run_params.json")
    if not os.path.isfile(params_path):
        raise SystemExit(f"Generation reuse source has no run_params.json: {params_path}")
    with open(params_path, encoding="utf-8") as handle:
        source = json.load(handle)
    critical = (
        "dataset", "split", "max_samples", "encoder_model", "top_m", "top_k",
        "metric", "dup_noise", "dup_target", "single_subtopic", "generator_model",
        "generator_backend", "generator_revision", "generator_max_new_tokens",
        "generator_num_beams", "generator_temperature", "generator_top_p",
        "generator_seed", "generator_thinking", "generator_prompt_version",
        "gen_max_samples", "seed", "seeds",
    )
    mismatches = [key for key in critical if source.get(key) != cfg.get(key)]
    requested_levels = {float(value) for value in cfg.get("rho_grid", [])}
    source_levels = {float(value) for value in source.get("rho_grid", [])}
    if not requested_levels.issubset(source_levels):
        mismatches.append("rho_grid")
    if mismatches:
        raise SystemExit(
            "Cannot reuse generations with incompatible settings: "
            + ", ".join(mismatches)
        )
    return {
        "path": path,
        "sha256": _sha256_file(path),
        "run_params_path": params_path,
        "source_git_commit": source.get("git_commit", "unknown"),
    }


# ---------------------------------------------------------------------------
# Experiment 1 — redundancy injection
# ---------------------------------------------------------------------------

_SWEEP_METRIC_COLS = ["PoolRedundancy", "Recall@k", "NDCG@k", "MRR",
                      "alpha-NDCG@k", "S-Recall@k", "ERR-IA@k", "APD", "Vendi"]


def _seed_summary_rows(
    per_method: Dict[str, List[Dict[str, float]]],
    n_examples: int,
    cfg: Dict,
    level: float,
    level_col: str,
    mean_red: float,
    seed: int,
    objective: str,
) -> List[Dict]:
    """Validation-tuned test-split summary for one (level, seed).

    Resolves each parameterised family to the grid member that maximises the
    objective on this seed's validation split, then reports test-split means
    for every metric plus a Wilcoxon p-value against kNN.  The reported
    ``Method`` label is the *stable* family name (``MMR*``, ``RNG*``,
    ``Seg*``); the selected member is recorded separately in ``Chosen`` so
    rows aggregate across seeds even when the tuned member differs.
    """
    alpha_grid = cfg["alpha_grid"]
    lambda_grid = cfg["lambda_grid"]
    dedup_grid = cfg["dedup_grid"]

    n_val = max(1, int(cfg["val_fraction"] * n_examples))
    order = np.random.default_rng(seed).permutation(n_examples)
    val_ids, test_ids = order[:n_val], order[n_val:]
    if len(test_ids) == 0:
        test_ids = val_ids

    def _mean(name: str, ids: np.ndarray, key: str) -> float:
        return float(np.mean([per_method[name][int(i)][key] for i in ids]))

    report: List[Tuple[str, str]] = [("kNN", "kNN"), ("Maxmin", "Maxmin"),
                                     ("Greedy-DPP", "Greedy-DPP")]
    for prefix, grid in [("Dedup", dedup_grid), ("MMR", lambda_grid),
                         ("VendiG", lambda_grid),
                         ("RNG", alpha_grid)]:
        members = [m for m in _grid_members(prefix, grid) if m in per_method]
        if not members:
            continue
        best = max(members, key=lambda nm: _mean(nm, val_ids, objective))
        report.append((f"{prefix}*", best))

    knn_test = [per_method["kNN"][int(i)][objective] for i in test_ids]
    rows: List[Dict] = []
    for label, name in report:
        row: Dict[str, object] = {level_col: level, "seed": seed,
                                  "Method": label, "Chosen": name,
                                  "PoolRedundancy": round(mean_red, 4)}
        for key in per_method[name][0]:
            row[key] = round(_mean(name, test_ids, key), 4)
        if name != "kNN":
            vals = [per_method[name][int(i)][objective] for i in test_ids]
            p = paired_wilcoxon_p(vals, knn_test)
            row[f"p({objective} vs kNN)"] = "" if p is None else round(p, 6)
        rows.append(row)
    return rows


def _aggregate_sweep_summary(
    per_seed_rows: List[Dict], level_col: str, objective: str,
) -> List[Dict]:
    """Collapse per-(level, seed) summary rows to per-(level, method) rows
    with across-seed mean +/- 95% CI on every metric.

    The tuned member (``Chosen``) is carried through, collapsed to the single
    value when stable across seeds and to a ``;``-joined list when it varied
    (an honest signal that the tuning is seed-sensitive).  The Wilcoxon
    p-values are summarised by their across-seed median and max (the
    conservative bound), rather than by a CI.
    """
    agg = aggregate_seed_rows(
        per_seed_rows, key_cols=[level_col, "Method"],
        metric_cols=_SWEEP_METRIC_COLS, seed_col="seed",
    )
    pcol = f"p({objective} vs kNN)"
    groups: Dict[Tuple, List[Dict]] = {}
    for r in per_seed_rows:
        groups.setdefault((r[level_col], r["Method"]), []).append(r)
    for row in agg:
        grp = groups.get((row[level_col], row["Method"]), [])
        chosen = list(dict.fromkeys(g.get("Chosen") for g in grp))
        row["Chosen"] = chosen[0] if len(chosen) == 1 else ";".join(map(str, chosen))
        ps = [as_float(g.get(pcol)) for g in grp]
        ps = [p for p in ps if p is not None]
        if ps:
            row["p_median(vs kNN)"] = round(float(np.median(ps)), 6)
            row["p_max(vs kNN)"] = round(float(np.max(ps)), 6)
    return agg


def _run_pool_sweep(
    examples: List[Dict],
    encoder: DenseRetriever,
    cfg: Dict,
    run_dir: str,
    levels: Sequence[float],
    make_pools: Callable[[float, int], List[List[Dict]]],
    level_col: str,
    file_tag: str,
    generator=None,
) -> None:
    """Shared sweep driver: for each level and each seed, build pools with
    *make_pools(level, seed)*, encode, truncate to top-m, run every reranker,
    evaluate, tune parameterised methods on the validation fraction and report
    test means.  Used by both the synthetic injection sweep (level = rho) and
    the natural chunk-overlap sweep (level = overlap).

    Seeds and confidence intervals
    ------------------------------
    The whole sweep is repeated once per seed in ``cfg["seeds"]``.  For the
    injection sweep the seed controls *which* near-duplicates land in each
    pool (and the validation/test split); for the chunking sweep the pools are
    deterministic and only the split moves.  Per-(level, seed) test summaries
    are aggregated to a mean with a 95% across-seed CI on every metric, written
    to ``results_<tag>_summary.csv``; the raw per-seed rows are kept alongside
    in ``results_<tag>_per_seed_summary.csv``, and ``results_<tag>_per_query.csv``
    carries a ``seed`` column so the downstream analyses can recompute their
    own across-seed intervals.

    Encoding is cached across seeds and levels: queries are encoded once per
    sweep, and passage embeddings are kept in a text-keyed cache so that each
    (level, seed) only encodes texts not seen before (the original passages
    recur everywhere; the chunked pools are seed-independent, so after the
    first seed of a chunking level the chunks are cache hits).  Non-original
    entries are evicted after each level, so the steady-state cache holds
    roughly one embedding per original passage plus the current level's
    duplicates.  Disable with --no_encode_cache if memory is tighter than
    encoder throughput."""
    k = cfg["top_k"]
    m = cfg["top_m"]
    metric = cfg["metric"]
    alpha_grid = cfg["alpha_grid"]
    lambda_grid = cfg["lambda_grid"]
    dedup_grid = cfg["dedup_grid"]
    objective = cfg["objective"]
    use_cache = bool(cfg.get("encode_cache", True))
    seeds = cfg["seeds"]

    per_seed_summary: List[Dict] = []
    per_query_rows: List[Dict] = []
    gen_rows: List[Dict] = []

    gen_methods = list(cfg.get("gen_methods") or [])
    do_gen = generator is not None and bool(gen_methods)
    gen_path = os.path.join(run_dir, f"results_{file_tag}_gen_per_query.csv")
    completed_gen = set()
    if do_gen and os.path.isfile(gen_path):
        with open(gen_path, newline="", encoding="utf-8") as handle:
            gen_rows = list(csv.DictReader(handle))
        completed_gen = {
            (float(row[level_col]), int(row["seed"]), row["qid"], row["Method"])
            for row in gen_rows
        }
        print(f"   Resuming from {len(completed_gen)} checkpointed generations.")
    elif do_gen and cfg.get("reuse_generation_csv"):
        source_path = cfg["reuse_generation_csv"]
        requested_levels = {float(value) for value in levels}
        requested_seeds = {int(value) for value in seeds}
        with open(source_path, newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
        for source_row in source_rows:
            if (source_row.get("Method") not in gen_methods
                    or float(source_row[level_col]) not in requested_levels
                    or int(source_row["seed"]) not in requested_seeds):
                continue
            row = {key: value for key, value in source_row.items()
                   if key != "RunParams"}
            row["GenerationOrigin"] = "reused"
            row["GenerationSourceSHA256"] = cfg["reuse_generation_sha256"]
            gen_rows.append(row)
        completed_gen = {
            (float(row[level_col]), int(row["seed"]), row["qid"], row["Method"])
            for row in gen_rows
        }
        if len(completed_gen) != len(gen_rows):
            raise SystemExit("Generation reuse source contains duplicate compatible keys.")
        print(f"   Reused {len(gen_rows)} provenance-validated generations "
              f"from SHA256 {cfg['reuse_generation_sha256'][:12]}…")
    if do_gen:
        print(f"   Generation on for {gen_methods} per (level, seed) "
              f"(reader {cfg.get('generator_model')}).")

    questions = [ex["question"] for ex in examples]
    q_embs: Optional[np.ndarray] = None
    cache: Dict[str, np.ndarray] = {}
    protected = {_format_passage(p) for ex in examples for p in ex["passages"]}

    for level in levels:
        print(f"\n── {level_col} = {level:g} ──")
        for seed in seeds:
            print(f"   · seed {seed}")
            pools = make_pools(level, seed)
            gen_jobs: Dict[str, List] = {}

            if not use_cache:
                q_embs_l, p_flat, p_off = encode_queries_and_passages(
                    encoder, questions, pools)
            else:
                if q_embs is None:
                    print(f"   Encoding {len(questions)} queries …")
                    q_embs = encoder.encode(questions, normalize=True,
                                            show_progress=True)
                q_embs_l = q_embs
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
                      f"{n_cached} cached from previous (level, seed)s …")
                if miss:
                    new_embs = encoder.encode(miss, normalize=True,
                                              show_progress=True)
                    for t, emb in zip(miss, new_embs):
                        cache[t] = emb
                p_flat = np.stack([cache[t] for t in flat_texts])

            # Generation subset for this seed (decorrelated from the split
            # permutation by offsetting the seed). The decision rule's answer
            # quality is reconstructed offline in analyze_regimes.py.
            if do_gen:
                gn = cfg.get("gen_max_samples") or len(examples)
                gn = len(examples) if gn == "all" else min(int(gn), len(examples))
                gen_idx = set(
                    np.random.default_rng(seed + 1)
                    .permutation(len(examples))[:gn].tolist()
                )
            else:
                gen_idx = set()

            per_method: Dict[str, List[Dict[str, float]]] = {}
            redundancies: List[float] = []

            for i, ex in enumerate(examples):
                s, e = int(p_off[i]), int(p_off[i + 1])
                q_emb = q_embs_l[i]
                p_embs = p_flat[s:e]
                pool = pools[i]

                # First-stage truncation: top-m by query similarity.
                sims = p_embs @ q_emb
                keep = np.argsort(-sims)[: min(m, len(pool))]
                pool_m = [pool[j] for j in keep]
                p_embs_m = p_embs[keep]

                redundancies.append(pool_redundancy(p_embs_m))
                rel_size = relevant_set_size(
                    ex["gold_titles"] or [], cfg.get("single_subtopic", False))

                selections = method_selections(
                    p_embs_m, q_emb, k, metric, alpha_grid, lambda_grid,
                    dedup_grid,
                )
                for name, sel in selections.items():
                    res = evaluate_selection(
                        sel, pool_m, p_embs_m, ex["gold_titles"] or [], k,
                        single_subtopic=cfg.get("single_subtopic", False))
                    per_method.setdefault(name, []).append(res)
                    per_query_rows.append(
                        {level_col: level, "seed": seed, "qid": ex["id"],
                         "Method": name,
                         "PoolRedundancy": round(redundancies[-1], 4),
                         "RelSetSize": rel_size, **res}
                    )

                if i in gen_idx:
                    answers = ex.get("answers") or []
                    for gm in gen_methods:
                        if gm in selections:
                            sel_passages = [pool_m[j] for j in selections[gm]]
                            gen_jobs.setdefault(gm, []).append(
                                (ex["id"], ex["question"], answers, sel_passages))

            # Answer generation for the sampled queries at this (level, seed).
            if generator is not None and gen_jobs:
                for gm, jobs in gen_jobs.items():
                    pending = [j for j in jobs if
                               (float(level), int(seed), j[0], gm) not in completed_gen]
                    checkpoint_every = int(cfg.get("generation_checkpoint_every", 256))
                    for start in range(0, len(pending), checkpoint_every):
                        chunk = pending[start:start + checkpoint_every]
                        preds = generator.generate_batch(
                            [j[1] for j in chunk], [j[3] for j in chunk],
                            batch_size=cfg.get("generator_batch_size", 8),
                            show_progress=True,
                            desc=f"   generating {gm} ({level_col}={level:g}, seed={seed})")
                        for (qid, _q, answers, sel_passages), pred in zip(chunk, preds):
                            em = max((exact_match(pred, a) for a in answers), default=0.0)
                            f1 = max((f1_score_single(pred, a) for a in answers), default=0.0)
                            hall = hallucination_rate(pred, [p["text"] for p in sel_passages])
                            gen_rows.append(
                                {level_col: level, "seed": seed, "qid": qid,
                                 "Method": gm, "EM": round(em, 4),
                                 "F1": round(f1, 4), "Halluc": round(hall, 4),
                                 "Prediction": pred[:2000],
                                 "GenerationOrigin": "generated",
                                 "GenerationSourceSHA256": ""})
                            completed_gen.add((float(level), int(seed), qid, gm))
                        save_csv(attach_run_params(_pad_rows(gen_rows), cfg), gen_path)

            mean_red = float(np.mean(redundancies))
            per_seed_summary.extend(_seed_summary_rows(
                per_method, len(examples), cfg, level, level_col,
                mean_red, seed, objective))

        if use_cache:
            # Evict per-level texts (perturbed duplicates, level-specific
            # chunks) so the cache never grows beyond the original passages.
            for t in [tt for tt in cache if tt not in protected]:
                del cache[t]

    summary_rows = _aggregate_sweep_summary(per_seed_summary, level_col, objective)
    save_csv(attach_run_params(_pad_rows(summary_rows), cfg),
             os.path.join(run_dir, f"results_{file_tag}_summary.csv"))
    save_csv(attach_run_params(_pad_rows(per_seed_summary), cfg),
             os.path.join(run_dir, f"results_{file_tag}_per_seed_summary.csv"))
    save_csv(per_query_rows,
             os.path.join(run_dir, f"results_{file_tag}_per_query.csv"))
    print(f"   {len(seeds)} seed(s); summary carries 95% across-seed CIs.")
    if gen_rows:
        save_csv(attach_run_params(_pad_rows(gen_rows), cfg), gen_path)
        print(f"   Saved generation EM/F1 for {len(gen_rows)} "
              f"(level, seed, query, method) rows.")


def run_redundancy_experiment(
    examples: List[Dict],
    encoder: DenseRetriever,
    cfg: Dict,
    run_dir: str,
    generator=None,
) -> None:
    """Sweep the injected-redundancy level rho and evaluate every reranker.

    For each (rho, seed) the full pipeline is repeated from the text level:
    duplicate injection, encoding, top-m pool truncation, reranking, and
    evaluation.  The injection RNG is seeded per seed, so the across-seed CIs
    in the summary capture the run-to-run variance of *which* near-duplicates
    land in each pool.  Per-query rows (with a ``seed`` column) are saved so
    paired significance tests against kNN can be computed.
    """
    def make_pools(rho: float, seed: int) -> List[List[Dict]]:
        rng = np.random.default_rng(seed)
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

    def make_pools(overlap: float, seed: int) -> List[List[Dict]]:
        # Chunking is deterministic: the pools do not depend on the seed, so
        # the across-seed CIs here reflect only the validation/test split.
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
    dedup_grid = cfg["dedup_grid"]
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

        selections = method_selections(p_embs_m, q_emb, k, metric, alpha_grid,
                                       lambda_grid, dedup_grid)
        row: Dict[str, object] = {"qid": ex["id"]}
        for name, sel in selections.items():
            res = evaluate_selection(sel, pool_m, p_embs_m, ex["gold_titles"] or [], k,
                                     single_subtopic=cfg.get("single_subtopic", False))
            per_method.setdefault(name, []).append(res)
            row[f"{name}:{objective}"] = round(res[objective], 4)
        per_query_rows.append(row)

    # The per-query metrics above are deterministic (no injection in the
    # oracle experiment); only the validation/test split depends on the seed.
    # We therefore reuse one per_method computation and loop seeds over the
    # split alone, then aggregate the headroom across seeds with a 95% CI.
    seeds = cfg["seeds"]
    obj_metrics = [key for key in per_method["kNN"][0] if key != objective]

    def _mean(name: str, ids: np.ndarray, key: str) -> float:
        return float(np.mean([per_method[name][int(i)][key] for i in ids]))

    per_seed_rows: List[Dict] = []
    for seed in seeds:
        n_val = max(1, int(cfg["val_fraction"] * len(examples)))
        order = np.random.default_rng(seed).permutation(len(examples))
        val_ids, test_ids = order[:n_val], order[n_val:]
        if len(test_ids) == 0:
            test_ids = val_ids

        def _report(label: str, chosen: str, values_per_query: List[float],
                    extra: Optional[Dict[str, float]] = None) -> None:
            test_vals = [values_per_query[int(i)] for i in test_ids]
            row = {"Method": label, "Chosen": chosen, "seed": seed,
                   objective: round(float(np.mean(test_vals)), 4)}
            if extra:
                row.update({kk: round(vv, 4) for kk, vv in extra.items()})
            per_seed_rows.append(row)

        knn_obj = [r[objective] for r in per_method["kNN"]]
        _report("kNN", "kNN", knn_obj,
                {key: _mean("kNN", test_ids, key) for key in obj_metrics})

        for prefix, grid in [("Dedup", dedup_grid), ("MMR", lambda_grid),
                             ("VendiG", lambda_grid),
                             ("RNG", alpha_grid)]:
            members = [m for m in _grid_members(prefix, grid)
                       if m in per_method]
            if not members:
                continue

            # Fixed operating point tuned on this seed's validation fraction.
            best = max(members, key=lambda nm: _mean(nm, val_ids, objective))
            fixed_obj = [r[objective] for r in per_method[best]]
            _report(f"{prefix} fixed", best, fixed_obj,
                    {key: _mean(best, test_ids, key) for key in obj_metrics})

            # Per-query oracle over the same grid (seed-independent, but the
            # reported mean is over this seed's test split).
            oracle_obj: List[float] = []
            oracle_choice: List[str] = []
            for qi in range(len(examples)):
                vals = {nm: per_method[nm][qi][objective] for nm in members}
                nm_best = max(vals, key=vals.get)
                oracle_obj.append(vals[nm_best])
                oracle_choice.append(nm_best)
            _report(f"{prefix} oracle (per-query)", f"{prefix}-oracle", oracle_obj)

            # Headroom rows: what adaptivity could add, and what
            # diversification adds over plain kNN even with oracle knowledge.
            test_o = np.array([oracle_obj[int(i)] for i in test_ids])
            test_f = np.array([fixed_obj[int(i)] for i in test_ids])
            test_k = np.array([knn_obj[int(i)] for i in test_ids])
            per_seed_rows.append({
                "Method": f"{prefix} headroom", "Chosen": "", "seed": seed,
                objective: "",
                "oracle - fixed": round(float(np.mean(test_o - test_f)), 4),
                "oracle - kNN": round(float(np.mean(test_o - test_k)), 4),
                "% queries where oracle beats kNN":
                    round(float(np.mean(test_o > test_k)) * 100.0, 1),
            })

            if seed == seeds[0]:
                for qi, row in enumerate(per_query_rows):
                    row[f"{prefix} oracle choice"] = oracle_choice[qi]

    # Aggregate across seeds: mean +/- 95% CI on the objective and the headroom
    # quantities; the tuned member (Chosen) is collapsed when stable.
    oracle_metric_cols = [objective, "oracle - fixed", "oracle - kNN",
                          "% queries where oracle beats kNN"] + obj_metrics
    summary_rows = aggregate_seed_rows(
        per_seed_rows, key_cols=["Method"], metric_cols=oracle_metric_cols,
        seed_col="seed")
    groups: Dict[str, List[Dict]] = {}
    for r in per_seed_rows:
        groups.setdefault(r["Method"], []).append(r)
    for row in summary_rows:
        chosen = list(dict.fromkeys(g.get("Chosen") for g in groups.get(row["Method"], [])))
        row["Chosen"] = chosen[0] if len(chosen) == 1 else ";".join(map(str, chosen))

    save_csv(attach_run_params(_pad_rows(summary_rows), cfg),
             os.path.join(run_dir, "results_oracle_summary.csv"))
    save_csv(attach_run_params(_pad_rows(per_seed_rows), cfg),
             os.path.join(run_dir, "results_oracle_per_seed_summary.csv"))
    save_csv(per_query_rows, os.path.join(run_dir, "results_oracle_per_query.csv"))
    print(f"   {len(seeds)} seed(s); oracle summary carries 95% across-seed CIs.")


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
    p.add_argument("--dedup_grid", type=float, nargs="+", default=DEFAULT_DEDUP_GRID,
                   help="Cosine thresholds t for the Dedup(t) baseline "
                        "(greedy near-duplicate removal, then top-k by "
                        "relevance).")
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
    p.add_argument("--generator_backend", choices=["transformers", "openai-compatible"],
                   default="transformers",
                   help="Use openai-compatible for a local vLLM server.")
    p.add_argument("--generator_api_base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--generator_api_key", default="EMPTY")
    p.add_argument("--generator_revision", default=None,
                   help="Pinned model revision recorded for reproducibility; the server must use it.")
    p.add_argument("--generator_max_new_tokens", type=int, default=128)
    p.add_argument("--generator_num_beams", type=int, default=4)
    p.add_argument("--generator_batch_size", type=int, default=None,
                   help="Local batch size or concurrent requests for an API reader.")
    p.add_argument("--generator_timeout", type=float, default=180.0)
    p.add_argument("--generation_checkpoint_every", type=int, default=256)
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
    p.add_argument("--seed", type=int, default=0,
                   help="Single-run seed (used when --seeds is not given).")
    add_seed_arg(p)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--resume_run_dir", default=None,
                   help="Existing compatible run directory whose generation CSV should be resumed.")
    p.add_argument("--reuse_generation_csv", default=None,
                   help="Earlier generation CSV whose compatible rows should be reused in a new run. "
                        "Its run_params.json and SHA256 are validated and recorded.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = resolve_seeds(args)
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
        "evaluation_script_sha256": _sha256_file(os.path.abspath(__file__)),
        "generator_script_sha256": _sha256_file(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "generation", "generator.py")),
        "run_utils_sha256": _sha256_file(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "run_utils.py")),
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
        "dedup_grid": list(args.dedup_grid),
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
        "generator_backend": args.generator_backend,
        "generator_api_base": args.generator_api_base,
        "generator_revision": args.generator_revision,
        "generator_max_new_tokens": args.generator_max_new_tokens,
        "generator_num_beams": args.generator_num_beams,
        "generator_batch_size": args.generator_batch_size or args.batch_size,
        "generator_timeout": args.generator_timeout,
        "generation_checkpoint_every": args.generation_checkpoint_every,
        "generator_temperature": 0,
        "generator_top_p": 1,
        "generator_seed": 0,
        "generator_thinking": False,
        "generator_prompt_version": "short_direct_v1",
        "gen_methods": list(args.gen_methods) if args.run_generation else [],
        "gen_max_samples": args.gen_max_samples,
        "batch_size": args.batch_size,
        "seed": seeds[0],
        "seeds": seeds,
    }

    if args.reuse_generation_csv:
        reuse = _validate_reuse_source(args.reuse_generation_csv, cfg)
        cfg.update(
            reuse_generation_csv=reuse["path"],
            reuse_generation_sha256=reuse["sha256"],
            reuse_generation_run_params=reuse["run_params_path"],
            reuse_generation_source_git_commit=reuse["source_git_commit"],
        )
        print(f"   Validated generation reuse source: {reuse['sha256']}")

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
        generator = load_generator(
            generator_name, device=device,
            max_new_tokens=args.generator_max_new_tokens,
            num_beams=args.generator_num_beams,
            backend=args.generator_backend,
            api_base=args.generator_api_base,
            api_key=args.generator_api_key,
            timeout=args.generator_timeout,
            seed=0,
        )

    if args.resume_run_dir:
        run_dir = os.path.abspath(args.resume_run_dir)
        params_path = os.path.join(run_dir, "run_params.json")
        if not os.path.isfile(params_path):
            raise SystemExit(f"Cannot resume: missing {params_path}")
        with open(params_path, encoding="utf-8") as handle:
            previous = json.load(handle)
        critical = ("experiment", "dataset", "split", "max_samples", "encoder_model",
                    "rho_grid", "dup_noise", "dup_target", "top_m", "top_k",
                    "generator_model", "generator_backend", "generator_revision",
                    "gen_methods", "seeds")
        mismatches = [key for key in critical if previous.get(key) != cfg.get(key)]
        if mismatches:
            raise SystemExit("Cannot resume with changed settings: " + ", ".join(mismatches))
        print(f"   Resuming run directory: {run_dir}")
    else:
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
