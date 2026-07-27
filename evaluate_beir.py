#!/usr/bin/env python3
"""
evaluate_beir.py  –  RQ3: generalisation to heterogeneous BEIR retrieval tasks
===============================================================================

Evaluates kNN, MMR, Maxmin, Greedy-DPP, RNG-Score, and Seg-Score on BEIR
tasks: SciFact, FiQA-2018, TREC-COVID, Arguana, and Webis-Touche2020.

When --ce_model is supplied, also evaluates CE-topk, CE-MMR, CE-DPP, and the
S1 / S2 cross-encoder integration strategies from RQ4, applied to BEIR.

Usage
-----
    python evaluate_beir.py                            # all tasks, dense only
    python evaluate_beir.py --tasks scifact fiqa
    python evaluate_beir.py --tasks scifact --max_queries 50 --no_index_rebuild
    python evaluate_beir.py --ce_model minilm-ce       # add CE methods

For each task the script:
  1. Downloads the BEIR corpus and queries from HuggingFace Hub.
  2. Builds (or loads) a FAISS flat inner-product index over the corpus.
  3. For each query retrieves the top-m passages.
  4. Re-ranks with every method and evaluates using graded qrels.
  5. Saves CSV results to --output_dir.

The validation-optimal margin alpha* is selected on a 20% held-out validation
split of the available queries (random seed 0), and test metrics are reported
at that alpha*.

Relevance metrics : NDCG@k (graded), Recall@k (binary, relevant = score>=1).
Intent-aware      : alpha-NDCG@k (alpha_r=0.5), S-Recall@k.
Annotation-free   : APD (avg pairwise distance), Vendi Score.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from alpha_selection import (
    AlphaResult,
    accumulate_alpha,
    accumulate_blend,
    average_accumulator,
    build_coarse_to_fine_alpha_grid,
    find_optimal_alpha,
    find_optimal_blend,
    load_alphas,
    lookup_alpha,
    save_alphas,
)
from data.loaders import load_beir_dataset
from evaluation.metrics import (
    alpha_ndcg_at_k,
    avg_pairwise_distance,
    ndcg_graded,
    recall_from_qrels,
    subtopic_coverage_sets,
    subtopic_recall_at_k,
    vendi_score,
)
from retrieval.cross_encoder import CrossEncoderReranker
from retrieval.precompute import score_cross_encoder
from retrieval.rerankers import (
    rerank_ce_rng_blended,
    rerank_ce_semimetric,
    rerank_ce_topk,
    rerank_greedy_dpp,
    rerank_knn,
    rerank_maxmin,
    rerank_mmr,
    rerank_mmr_ce,
    rerank_rng_score,
    rerank_rng_score2,
)
from retrieval.retriever import DenseRetriever
from run_utils import (
    CE_ALIASES,
    ENCODER_ALIASES,
    attach_run_params,
    default_encoder,
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

_TASKS = ["scifact", "fiqa", "trec-covid", "arguana", "webis-touche2020"]
# -2.0 is the fallback sentinel: under cosine distance it deactivates every
# obstruction penalty (Prop. knnlimit), guaranteeing the exact k-NN fallback.
_ALPHA_GRID = [-2.0, -0.5, -0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3, 0.5]
_BETA_GRID = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
_MMR_LAMBDAS = [0.3, 0.5, 0.7]

# Metric columns aggregated to mean +/- 95% CI across seeds in the BEIR table.
_BEIR_METRIC_COLS = ["NDCG@k", "Recall@k", "alpha-NDCG@k", "S-Recall@k",
                     "APD", "Vendi", "Latency (ms)"]
# Objectives on which significance against k-NN is reported.
_BEIR_SIG_OBJECTIVES = ["alpha-NDCG@k", "S-Recall@k"]
# Families whose reported operating point is validation-tuned, so the alpha*/
# beta* in the method label moves with the seed; collapse them to a stable key
# (the prefix before the first parenthesis) for across-seed aggregation.
_TUNED_PREFIXES = ("RNG-Score", "Seg-Score", "S1-", "S2-")


def _method_key(method_name: str) -> str:
    """Seed-stable grouping key for a (possibly tuned) method label.

    Tuned families drop their ``(alpha*=...)`` suffix so the same family
    aggregates across seeds even when the tuned value differs; fixed methods
    (kNN, MMR(lambda=...), CE-MMR(lambda=...), Maxmin, ...) keep their label.
    """
    if any(method_name.startswith(p) for p in _TUNED_PREFIXES):
        return method_name.split("(", 1)[0]
    return method_name


# ---------------------------------------------------------------------------
# Dense methods
# ---------------------------------------------------------------------------

def _build_methods(k: int, metric: str, alpha_grid: List[float]) -> List[Tuple[str, Callable[..., List[int]]]]:
    methods = []
    methods.append(("kNN", lambda embs, q, scores, _k=k: rerank_knn(scores, _k)))
    for lam in _MMR_LAMBDAS:
        lam_ = lam
        methods.append((
            f"MMR(λ={lam_})",
            lambda embs, q, scores, _k=k, _l=lam_: rerank_mmr(embs, q, scores, _k, _l),
        ))
    methods.append(("Maxmin", lambda embs, q, scores, _k=k: rerank_maxmin(embs, scores, _k)))
    methods.append(("Greedy-DPP", lambda embs, q, scores, _k=k: rerank_greedy_dpp(embs, scores, _k)))
    for alpha in alpha_grid:
        a_ = alpha
        methods.append((
            f"RNG-Score(α={a_})",
            lambda embs, q, scores, _k=k, _a=a_, _m=metric: rerank_rng_score(embs, q, _k, _a, _m),
        ))
        methods.append((
            f"Seg-Score(α={a_})",
            lambda embs, q, scores, _k=k, _a=a_, _m=metric: rerank_rng_score2(embs, q, _k, _a, _m),
        ))
    return methods


# ---------------------------------------------------------------------------
# Cross-encoder methods
# ---------------------------------------------------------------------------

def _build_ce_methods(
    k: int,
    metric: str,
    alpha_grid: List[float],
    beta_grid: List[float],
    mmr_lambdas: List[float],
) -> List[Tuple[str, Callable[..., List[int]]]]:
    """Return CE-based (name, callable) pairs.

    Each callable has the same signature as dense methods:
        fn(p_embs, q_emb, ce_scores, k) -> List[int]
    """
    methods: List[Tuple[str, Callable[..., List[int]]]] = []

    methods.append(("CE-topk", lambda embs, q, ce, _k=k: rerank_ce_topk(ce, _k)))

    for lam in mmr_lambdas:
        lam_ = lam
        methods.append((
            f"CE-MMR(λ={lam_})",
            lambda embs, q, ce, _k=k, _l=lam_: rerank_mmr_ce(embs, ce, _k, _l),
        ))

    methods.append(("CE-DPP", lambda embs, q, ce, _k=k: rerank_greedy_dpp(embs, ce, _k)))

    for s1_variant, transform in [(1, "sigmoidal"), (2, "reciprocal")]:
        sv_, tr_ = s1_variant, transform
        for alpha in alpha_grid:
            for beta in beta_grid:
                a_, b_ = alpha, beta
                methods.append((
                    f"S1-V{sv_}-Blend(α={a_:.4g},β={b_:.4g})",
                    lambda embs, q, ce, _k=k, _a=a_, _b=b_, _m=metric, _t=tr_: rerank_ce_rng_blended(
                        embs, q, ce, _k, alpha=_a, beta=_b, metric=_m, transform=_t
                    ),
                ))

    for variant in [1, 2, 3, 4, 5]:
        for alpha in alpha_grid:
            a_, v_ = alpha, variant
            methods.append((
                f"S2-V{v_}(α={a_:.4g})",
                lambda embs, q, ce, _k=k, _a=a_, _v=v_, _m=metric: rerank_ce_semimetric(
                    embs, q, ce, _k, alpha=_a, variant=_v, metric=_m
                ),
            ))

    return methods


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------

def evaluate_task(
    task_name: str,
    cfg: Dict,
) -> Tuple[List[Dict], List[Dict], List[AlphaResult]]:
    """Run full evaluation for one BEIR task.

    Returns
    -------
    summary_rows : list of dict  (method × mean metrics)
    per_query_rows : list of dict
    alpha_results : list of AlphaResult  (val-optimal α* per score type)
    """
    print(f"\n══ Task: {task_name} ══")
    max_queries = cfg.get("max_queries")
    top_m = cfg.get("top_m", 100)
    top_k = cfg.get("top_k", 10)
    metric = cfg.get("metric", "cosine")
    alpha_r = float(cfg.get("alpha_r", 0.5))
    val_frac = float(cfg.get("val_frac", 0.2))
    index_dir = cfg.get("index_dir")
    encoder_name = cfg["encoder_model"]
    alpha_grid = cfg.get("alpha_grid", _ALPHA_GRID)
    beta_grid = cfg.get("beta_grid", _BETA_GRID)
    mmr_lambdas = cfg.get("mmr_lambdas", _MMR_LAMBDAS)
    cross_encoder: Optional[CrossEncoderReranker] = cfg.get("_cross_encoder")
    ce_model_name: Optional[str] = cfg.get("ce_model")

    # 1. Load dataset
    print(f"   Loading {task_name} …")
    examples, corpus = load_beir_dataset(task_name, max_queries=max_queries)
    print(f"   {len(examples)} queries, {len(corpus)} corpus documents.")

    # 2. Build or load index
    encoder = DenseRetriever(
        model_name=encoder_name,
        device=normalize_device(cfg.get("device", "cpu")),
        batch_size=cfg.get("batch_size", 64),
    )
    corpus_passages = list(corpus.values())

    if index_dir:
        cache = os.path.join(index_dir, task_name.replace("-", "_"))
        if os.path.exists(os.path.join(cache, "index.faiss")):
            print(f"   Loading cached index from {cache}")
            encoder.load(cache)
        else:
            encoder.build_index(corpus_passages, text_field="text", title_field="title")
            os.makedirs(cache, exist_ok=True)
            encoder.save(cache)
    else:
        encoder.build_index(corpus_passages, text_field="text", title_field="title")

    # 3. Val / test split for alpha selection (seed-controlled so the across-
    # seed CIs capture the variance from which queries tune alpha*).
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(examples))
    rng.shuffle(idx_all)
    n_val = max(1, int(len(examples) * val_frac))
    val_idx = set(idx_all[:n_val].tolist())

    # 4. Build method lists
    methods = _build_methods(top_k, metric, alpha_grid)
    ce_methods = (
        _build_ce_methods(top_k, metric, alpha_grid, beta_grid, mmr_lambdas)
        if cross_encoder is not None else []
    )

    # Accumulators: split → method → metric list
    all_method_names = [name for name, _ in methods] + [name for name, _ in ce_methods]
    acc: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        "val": {name: defaultdict(list) for name in all_method_names},
        "test": {name: defaultdict(list) for name in all_method_names},
    }
    per_query_rows: List[Dict] = []

    _raw_obj = cfg.get("objective", "apd")
    objectives: List[str] = [_raw_obj] if isinstance(_raw_obj, str) else list(_raw_obj)

    # 5. Pre-compute: encode all queries and batch-search the index in one sweep.
    print(f"\n── Pre-computing search for {len(examples)} queries ──")
    _all_questions = [ex["question"] for ex in examples]
    _q_embs_all, _scores_all, _idxs_all = encoder.search_batch(_all_questions, top_m)

    # 6. Optional CE pre-computation: score all (query, candidate) pairs in one GPU sweep.
    ce_scores_flat: Optional[np.ndarray] = None
    _p_offsets: Optional[np.ndarray] = None
    if cross_encoder is not None:
        print(f"\n── CE pre-computing {len(examples)} queries × {top_m} candidates ──")
        _all_pools = [
            [corpus_passages[i] for i in _idxs_all[qi].tolist()]
            for qi in range(len(examples))
        ]
        ce_scores_flat = score_cross_encoder(cross_encoder, _all_questions, _all_pools)
        _p_offsets = np.zeros(len(examples) + 1, dtype=np.int64)
        for i, pool in enumerate(_all_pools):
            _p_offsets[i + 1] = _p_offsets[i] + len(pool)

    # 7. Per-query loop (pure CPU: slice pre-computed arrays + numpy reranking)
    for qi, ex in enumerate(tqdm(examples, desc=f"  {task_name}")):
        split_name = "val" if qi in val_idx else "test"
        qid = ex["id"]
        question = _all_questions[qi]
        qrels = ex["qrels"]

        _ridxs = _idxs_all[qi].tolist()
        pool = [corpus_passages[i] for i in _ridxs]
        q_emb = _q_embs_all[qi]
        scores = _scores_all[qi]
        p_embs = encoder.passage_embeddings[_ridxs]

        rel_ids = list(qrels.keys())
        subtopic_sets = subtopic_coverage_sets(pool, qrel_ids=rel_ids)

        ce_scores: Optional[np.ndarray] = None
        if ce_scores_flat is not None:
            _s, _e = int(_p_offsets[qi]), int(_p_offsets[qi + 1])
            ce_scores = ce_scores_flat[_s:_e]

        def _eval_and_record(method_name: str, sel: List[int], lat_ms: float) -> None:
            if not sel:
                return
            sel_embs = p_embs[sel]
            ndcg_g = ndcg_graded(sel, pool, qrels, top_k)
            rec = recall_from_qrels(sel, pool, qrels, top_k)
            an = alpha_ndcg_at_k(sel, subtopic_sets, top_k, alpha_r=alpha_r)
            sr = subtopic_recall_at_k(sel, subtopic_sets, top_k)
            apd = avg_pairwise_distance(sel_embs)
            vs = vendi_score(sel_embs)

            acc[split_name][method_name]["ndcg"].append(ndcg_g)
            acc[split_name][method_name]["recall"].append(rec)
            acc[split_name][method_name]["alpha_ndcg"].append(an)
            acc[split_name][method_name]["s_recall"].append(sr)
            acc[split_name][method_name]["apd"].append(apd)
            acc[split_name][method_name]["vendi"].append(vs)
            acc[split_name][method_name]["latency_ms"].append(lat_ms)

            per_query_rows.append({
                "Task": task_name,
                "seed": seed,
                "Split": split_name,
                "Query ID": qid,
                "Method": method_name,
                "NDCG@k": f"{ndcg_g:.6f}",
                "Recall@k": f"{rec:.6f}",
                "alpha-NDCG@k": f"{an:.6f}",
                "S-Recall@k": f"{sr:.6f}",
                "APD": f"{apd:.6f}",
                "Vendi": f"{vs:.6f}",
                "Latency (ms)": f"{lat_ms:.6f}",
            })

        for method_name, rerank_fn in methods:
            t0 = time.perf_counter()
            sel = rerank_fn(p_embs, q_emb, scores, top_k)
            _eval_and_record(method_name, sel, (time.perf_counter() - t0) * 1000.0)

        if ce_scores is not None:
            for method_name, rerank_fn in ce_methods:
                t0 = time.perf_counter()
                sel = rerank_fn(p_embs, q_emb, ce_scores, top_k)
                _eval_and_record(method_name, sel, (time.perf_counter() - t0) * 1000.0)

    # 8. Select validation-optimal alpha for dense methods
    setting = f"beir/{task_name}/{encoder_name}"
    pre_alphas: List[AlphaResult] = cfg.get("_pre_alphas", [])
    multi_obj = len(objectives) > 1

    def _get_alpha_results(score_type: str) -> List[AlphaResult]:
        results = []
        for obj in objectives:
            pre = lookup_alpha(pre_alphas, score_type, setting=setting,
                               objective=obj) if pre_alphas else None
            if pre is not None:
                results.append(pre)
                continue
            val_acc: Dict[str, List[float]] = {}
            for alpha in alpha_grid:
                key = f"{score_type}(α={alpha})"
                vals = acc["val"].get(key, {}).get(obj, [])
                for v in vals:
                    accumulate_alpha(val_acc, alpha, v)
            avg = average_accumulator(val_acc)
            results.append(find_optimal_alpha(avg, alpha_grid, score_type, obj, setting))
        return results

    rng_results = _get_alpha_results("RNG-Score")
    seg_results = _get_alpha_results("Seg-Score")

    # 9. Select validation-optimal alpha/beta for CE methods
    ce_setting = (
        f"beir/{task_name}/{encoder_name}/{ce_model_name}"
        if ce_model_name else setting
    )
    ce_alpha_results: List[AlphaResult] = []
    ce_alpha_results_by_obj: Dict[str, Dict[str, Optional[AlphaResult]]] = {}

    if cross_encoder is not None:
        def _find_ce_blend(score_type: str, obj: str) -> Optional[AlphaResult]:
            pre = lookup_alpha(pre_alphas, score_type, setting=ce_setting,
                               objective=obj) if pre_alphas else None
            if pre is not None:
                return pre
            blend_acc: Dict = {}
            for alpha in alpha_grid:
                for beta in beta_grid:
                    key = f"{score_type}(α={alpha:.4g},β={beta:.4g})"
                    vals = acc["val"].get(key, {}).get(obj, [])
                    for v in vals:
                        accumulate_blend(blend_acc, alpha, beta, v)
            avg = average_accumulator(blend_acc)
            if not avg:
                return None
            return find_optimal_blend(avg, alpha_grid, beta_grid, score_type, obj, ce_setting)

        def _find_ce_alpha(score_type: str, obj: str) -> Optional[AlphaResult]:
            pre = lookup_alpha(pre_alphas, score_type, setting=ce_setting,
                               objective=obj) if pre_alphas else None
            if pre is not None:
                return pre
            val_acc: Dict = {}
            for alpha in alpha_grid:
                key = f"{score_type}(α={alpha:.4g})"
                vals = acc["val"].get(key, {}).get(obj, [])
                for v in vals:
                    accumulate_alpha(val_acc, alpha, v)
            avg = average_accumulator(val_acc)
            if not avg:
                return None
            return find_optimal_alpha(avg, alpha_grid, score_type, obj, ce_setting)

        ce_alpha_results_by_obj = {
            obj: {
                "S1-V1-Blend": _find_ce_blend("S1-V1-Blend", obj),
                "S1-V2-Blend": _find_ce_blend("S1-V2-Blend", obj),
                **{f"S2-V{v}": _find_ce_alpha(f"S2-V{v}", obj) for v in range(1, 6)},
            }
            for obj in objectives
        }
        ce_alpha_results = [
            r
            for obj_results in ce_alpha_results_by_obj.values()
            for r in obj_results.values()
            if r is not None
        ]

    # 10. Build summary table (test split only)
    summary_rows: List[Dict] = []

    def _add_row(method_name: str, tag: str = ""):
        m_acc = acc["test"].get(method_name, {})
        if not m_acc.get("ndcg"):
            return
        row = {
            "Task": task_name,
            "Method": tag or method_name,
            "MethodKey": _method_key(method_name),
            "RawMethod": method_name,
            "seed": seed,
            "NDCG@k": f"{np.mean(m_acc['ndcg']):.4f}",
            "Recall@k": f"{np.mean(m_acc['recall']):.4f}",
            "alpha-NDCG@k": f"{np.mean(m_acc['alpha_ndcg']):.4f}",
            "S-Recall@k": f"{np.mean(m_acc['s_recall']):.4f}",
            "APD": f"{np.mean(m_acc['apd']):.4f}",
            "Vendi": f"{np.mean(m_acc['vendi']):.4f}",
            "Latency (ms)": f"{np.mean(m_acc['latency_ms']):.2f}",
        }
        summary_rows.append(row)

    _add_row("kNN")
    for lam in _MMR_LAMBDAS:
        _add_row(f"MMR(λ={lam})")
    _add_row("Maxmin")
    _add_row("Greedy-DPP")
    for r in rng_results:
        if r.alpha_star is not None:
            tag = (f"RNG-Score(α*={r.alpha_star},{r.objective})" if multi_obj
                   else f"RNG-Score(α*={r.alpha_star})")
            _add_row(f"RNG-Score(α={r.alpha_star})", tag)
    for r in seg_results:
        if r.alpha_star is not None:
            tag = (f"Seg-Score(α*={r.alpha_star},{r.objective})" if multi_obj
                   else f"Seg-Score(α*={r.alpha_star})")
            _add_row(f"Seg-Score(α={r.alpha_star})", tag)

    if cross_encoder is not None:
        _add_row("CE-topk")
        for lam in mmr_lambdas:
            _add_row(f"CE-MMR(λ={lam})")
        _add_row("CE-DPP")
        for obj, obj_results in ce_alpha_results_by_obj.items():
            obj_suffix = f" [{obj}]" if multi_obj else ""
            for st in ["S1-V1-Blend", "S1-V2-Blend"]:
                res = obj_results[st]
                if res is not None and res.alpha_star is not None:
                    _add_row(
                        f"{st}(α={res.alpha_star:.4g},β={res.beta_star:.4g})",
                        f"{st}(α*={res.alpha_star:.4g},β*={res.beta_star:.4g}){obj_suffix}",
                    )
            for v in range(1, 6):
                st = f"S2-V{v}"
                res = obj_results[st]
                if res is not None and res.alpha_star is not None:
                    _add_row(
                        f"{st}(α={res.alpha_star:.4g})",
                        f"{st}(α*={res.alpha_star:.4g}){obj_suffix}",
                    )

    print(f"\n── {task_name} summary (test, k={top_k}) ──")
    print(tabulate(summary_rows, headers="keys", tablefmt="github"))
    for r in rng_results:
        print(f"   Val-optimal RNG-Score α* = {r.alpha_star:.4g}  "
              f"(val {r.objective}={r.val_score:.4f})")
    for r in seg_results:
        print(f"   Val-optimal Seg-Score  α* = {r.alpha_star:.4g}  "
              f"(val {r.objective}={r.val_score:.4f})")
    for r in ce_alpha_results:
        beta_str = f", β*={r.beta_star:.4g}" if r.beta_star is not None else ""
        print(f"   [{r.score_type}] α*={r.alpha_star:.4g}{beta_str}  "
              f"(val {r.objective}={r.val_score:.4f})")

    return summary_rows, per_query_rows, rng_results + seg_results + ce_alpha_results


# ---------------------------------------------------------------------------
# Across-seed aggregation
# ---------------------------------------------------------------------------

def _beir_test_values(per_query_rows: List[Dict], task: str, seed: int,
                      raw_method: str, metric: str) -> List[float]:
    """Test-split per-query values of *metric* for one (task, seed, method)."""
    return [as_float(r[metric]) for r in per_query_rows
            if r["Task"] == task and r["seed"] == seed
            and r["Split"] == "test" and r["Method"] == raw_method]


def _aggregate_beir(per_seed_summary: List[Dict],
                    per_query_rows: List[Dict]) -> List[Dict]:
    """One row per (Task, MethodKey) with across-seed mean +/- 95% CI on each
    metric and across-seed median Wilcoxon p-values vs kNN on the intent-aware
    objectives.  The tuned operating point per seed is recorded in
    ``alpha*(per seed)`` so a seed-sensitive selection is visible."""
    agg = aggregate_seed_rows(
        per_seed_summary, key_cols=["Task", "MethodKey"],
        metric_cols=_BEIR_METRIC_COLS, seed_col="seed", keep_cols=["Method"])
    by_key: Dict[Tuple[str, str], List[Dict]] = {}
    for r in per_seed_summary:
        by_key.setdefault((r["Task"], r["MethodKey"]), []).append(r)
    for row in agg:
        grp = by_key.get((row["Task"], row["MethodKey"]), [])
        raws = list(dict.fromkeys(g["RawMethod"] for g in grp))
        row["alpha*(per seed)"] = ";".join(raws) if len(raws) != 1 else raws[0]
        if row["MethodKey"] == "kNN":
            continue
        for obj in _BEIR_SIG_OBJECTIVES:
            ps: List[float] = []
            for g in grp:
                a = [v for v in _beir_test_values(per_query_rows, row["Task"],
                                                  g["seed"], g["RawMethod"], obj)
                     if v is not None]
                b = [v for v in _beir_test_values(per_query_rows, row["Task"],
                                                  g["seed"], "kNN", obj)
                     if v is not None]
                if a and b and len(a) == len(b):
                    p = paired_wilcoxon_p(a, b)
                    if p is not None:
                        ps.append(p)
            if ps:
                row[f"p({obj} vs kNN)"] = round(float(np.median(ps)), 6)
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BEIR evaluation (RQ3)")
    p.add_argument("--tasks", nargs="+", default=_TASKS, choices=_TASKS + ["all"],
                   help="BEIR tasks to evaluate.")
    p.add_argument("--encoder_model", default=None,
                   help="Encoder alias or HuggingFace ID. "
                        f"Aliases: {', '.join(ENCODER_ALIASES)}. "
                        "Default: bge-m3 on CUDA, minilm on CPU.")
    p.add_argument("--ce_model", default=None,
                   help="Cross-encoder alias or HuggingFace ID. When provided, adds "
                        "CE-topk, CE-MMR, CE-DPP, S1, and S2 methods to the evaluation. "
                        f"Aliases: {', '.join(CE_ALIASES)}. "
                        "Default: None (dense-only evaluation).")
    p.add_argument("--top_m", type=int, default=100,
                   help="Candidate pool size per query.")
    p.add_argument("--top_k", type=int, default=10,
                   help="Final selection size.")
    p.add_argument("--metric", choices=["cosine", "euclidean", "angular"],
                   default="cosine")
    p.add_argument("--max_queries", type=int_or_all, default=None,
                   help="Cap on total queries per task (for quick tests); pass 'all' for no cap.")
    p.add_argument("--max_samples", dest="max_queries", type=int_or_all, default=None,
                   help="Alias for --max_queries (kept for CLI consistency across evaluate scripts).")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--index_dir", default=None,
                   help="Directory to cache / reload FAISS indices.")
    p.add_argument("--no_index_rebuild", action="store_true",
                   help="Alias for passing --index_dir to the default cache location.")
    p.add_argument("--alpha_json", default=None,
                   help="Path to alpha_results.json (from learn_alpha.py). When provided, "
                        "skips val-split selection and uses pre-found α*.")
    p.add_argument("--save_alpha_json", default=None,
                   help="Path to save val-optimal α* results as JSON (e.g. results/alpha_results.json).")
    p.add_argument("--objective", nargs="+", default=["alpha_ndcg"],
                   choices=["recall", "ndcg", "alpha_ndcg", "s_recall", "vendi", "apd"],
                   help="Metric(s) used to select val-optimal α*. Default "
                        "alpha_ndcg (intent-aware coverage, the metric the BEIR "
                        "table reports); pass s_recall for parity with the QA "
                        "tuning, or ndcg for pure relevance. apd/vendi tune for "
                        "raw spread (intrinsic credit) and are not appropriate "
                        "for this instrumental-credit table. Multiple values "
                        "produce one α* per objective.")
    p.add_argument("--alpha_objective", dest="objective", nargs="+", default=None,
                   choices=["recall", "ndcg", "alpha_ndcg", "s_recall", "vendi", "apd"],
                   help="Alias for --objective.")
    p.add_argument("--coarse_to_fine", action="store_true",
                   help="Use an expanded fine-resolution alpha sweep for RNG/Seg scoring.")
    p.add_argument("--alpha_min", type=float, default=-1.0)
    p.add_argument("--alpha_max", type=float, default=1.0)
    p.add_argument("--alpha_step", type=float, default=0.25)
    p.add_argument("--fine_step", type=float, default=None,
                   help="Fine sweep step (default: alpha_step / 5).")
    p.add_argument("--dead_zone_min", type=float, default=-1.0,
                   help="Left extension disabled when alpha_min <= dead_zone_min.")
    p.add_argument(
        "--save_per_query",
        action="store_true",
        help="Also save per-query diagnostics CSV.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help='Device for model inference, e.g. "cpu", "cuda", "cuda:0" (default: cpu).',
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoder and cross-encoder inference (default: 64). "
             "Increase on GPU for better utilisation; reduce if you run out of memory.",
    )
    p.add_argument("--seed", type=int, default=0,
                   help="Single-run seed for the val/test split "
                        "(used when --seeds is not given).")
    add_seed_arg(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    tasks = args.tasks if "all" not in args.tasks else _TASKS

    if args.no_index_rebuild and args.index_dir is None:
        args.index_dir = os.path.join(args.output_dir, "indices")

    # Resolve device and model defaults.
    device = normalize_device(args.device)
    encoder_model = resolve_model(args.encoder_model or default_encoder(device), ENCODER_ALIASES)
    ce_model = resolve_model(args.ce_model, CE_ALIASES) if args.ce_model else None

    # Load pre-computed alphas if provided
    pre_alphas: List[AlphaResult] = []
    if args.alpha_json:
        pre_alphas = load_alphas(args.alpha_json)
        print(f"   Loaded {len(pre_alphas)} alpha result(s) from {args.alpha_json}")

    # Load cross-encoder once (shared across all tasks to avoid repeated downloads).
    cross_encoder: Optional[CrossEncoderReranker] = None
    if ce_model:
        print(f"\n── Loading cross-encoder: {ce_model} ──")
        cross_encoder = CrossEncoderReranker(
            model_name=ce_model, device=device, batch_size=args.batch_size
        )

    cfg = {
        "encoder_model": encoder_model,
        "ce_model": ce_model,
        "_cross_encoder": cross_encoder,
        "device": device,
        "batch_size": args.batch_size,
        "top_m": args.top_m,
        "top_k": args.top_k,
        "metric": args.metric,
        "max_queries": args.max_queries,
        "index_dir": args.index_dir,
        "alpha_r": 0.5,
        "val_frac": 0.2,
        "objective": args.objective,
        "_pre_alphas": pre_alphas,
        "beta_grid": _BETA_GRID,
        "mmr_lambdas": _MMR_LAMBDAS,
        "alpha_grid": (
            build_coarse_to_fine_alpha_grid(
                alpha_min=args.alpha_min,
                alpha_max=args.alpha_max,
                alpha_step=args.alpha_step,
                fine_step=args.fine_step,
                dead_zone_min=args.dead_zone_min,
            )
            if args.coarse_to_fine
            else _ALPHA_GRID
        ),
    }

    if args.coarse_to_fine:
        ag = cfg["alpha_grid"]
        print(
            "   [coarse_to_fine] using expanded alpha grid "
            f"[{ag[0]:.4g}, {ag[-1]:.4g}] with {len(ag)} values"
        )

    seeds = resolve_seeds(args)

    all_summary: List[Dict] = []
    all_per_query: List[Dict] = []
    all_alpha_results: List[AlphaResult] = []

    t0 = time.time()
    for seed in seeds:
        if len(seeds) > 1:
            print(f"\n========== seed {seed} ==========")
        for task in tasks:
            summary, per_query, alpha_res = evaluate_task(task, {**cfg, "seed": seed})
            all_summary.extend(summary)
            all_per_query.extend(per_query)
            all_alpha_results.extend(alpha_res)

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")
    run_params = {
        "script": "evaluate_beir.py",
        "tasks": tasks,
        "encoder_model": encoder_model,
        "ce_model": ce_model,
        "device": device,
        "top_m": args.top_m,
        "top_k": args.top_k,
        "metric": args.metric,
        "max_queries": args.max_queries,
        "index_dir": args.index_dir,
        "alpha_json": args.alpha_json,
        "objective": args.objective,
        "coarse_to_fine": args.coarse_to_fine,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_step": args.alpha_step,
        "fine_step": args.fine_step,
        "dead_zone_min": args.dead_zone_min,
        "alpha_grid": cfg.get("alpha_grid"),
        "seeds": seeds,
    }
    run_dir = make_run_dir(args.output_dir, "beir_" + "_".join(tasks), run_params)

    # Persist the raw data FIRST — it is the source of truth and the summary is
    # re-derivable from it, so an error in the (cheap) aggregation can never
    # destroy a long run's results.
    if args.save_per_query:
        save_csv(all_per_query, os.path.join(run_dir, "results_beir_per_query.csv"))
    if len(seeds) > 1:
        save_csv(attach_run_params(all_summary, run_params),
                 os.path.join(run_dir, "results_beir_per_seed_summary.csv"))
        # Headline table: across-seed mean +/- 95% CI with Wilcoxon vs kNN.
        summary_out = _aggregate_beir(all_summary, all_per_query)
        save_csv(attach_run_params(summary_out, run_params),
                 os.path.join(run_dir, "results_beir_summary.csv"))
        print(f"   {len(seeds)} seeds; summary carries 95% across-seed CIs "
              f"and median Wilcoxon p vs kNN.")
    else:
        # Single run: preserve the legacy column layout exactly.
        legacy = [{k: v for k, v in r.items()
                   if k not in ("MethodKey", "RawMethod", "seed")}
                  for r in all_summary]
        save_csv(attach_run_params(legacy, run_params),
                 os.path.join(run_dir, "results_beir_summary.csv"))

    # Save alpha* results to JSON
    save_path = args.save_alpha_json or os.path.join(run_dir, "alpha_results_beir.json")
    if all_alpha_results:
        save_alphas(all_alpha_results, save_path)


if __name__ == "__main__":
    main()
