#!/usr/bin/env python3
"""Evaluate retrieval, reranking, and optional answer generation."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml
from tabulate import tabulate
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ftrb.alpha_selection import OBJECTIVES, build_coarse_to_fine_alpha_grid, load_alphas, lookup_alpha
from data.loaders import (
    load_2wikimultihopqa,
    load_hotpotqa,
    load_hotpotqa_fullwiki,
    load_musique,
    load_nq_dpr,
    load_nq_open,
    load_squad,
    load_trivia_dpr,
)
from evaluation.metrics import (
    alpha_ndcg_at_k,
    avg_pairwise_distance,
    err_ia_at_k,
    exact_match,
    f1_score_single,
    hallucination_rate,
    mrr_from_relevance,
    ndcg_from_relevance,
    recall_at_k,
    relevance_labels,
    subtopic_coverage_sets,
    subtopic_recall_at_k,
    vendi_score,
)
from generation.generator import load_generator
from retrieval.precompute import encode_queries_and_passages
from retrieval.rerankers import (
    rerank_greedy_dpp,
    rerank_knn,
    rerank_maxmin,
    rerank_mmr,
    rerank_rds,
    rerank_rads,
    rerank_rng_score,
    rerank_rng_score2,
)
from retrieval.retriever import DenseRetriever
from ftrb.run_utils import (
    DEFAULT_DEVICE,
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
from ftrb.stats import significance_table

# -2.0 is the fallback sentinel: under cosine distance it deactivates every
# obstruction penalty (Prop. knnlimit), so the grid contains the exact k-NN
# operating point by construction rather than empirically.
_ALPHA_GRID = [-2.0, -0.5, -0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3, 0.5]
_MMR_LAMBDAS = [0.3, 0.5, 0.7]

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config.yaml")


def _load_config(path: str) -> Dict:
    """Load a YAML configuration file and return its contents as a dict.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    cfg : dict
        Parsed configuration as a Python dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _apply_cli_overrides(cfg: Dict, args: argparse.Namespace) -> Dict:
    """Override config values with any non-None CLI arguments.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary loaded from a YAML file.  Modified in place.
    args : argparse.Namespace
        Parsed command-line arguments.  Only non-``None`` attributes
        override the corresponding config keys.

    Returns
    -------
    cfg : dict
        The updated configuration dictionary (same object as input).
    """
    if cfg is None:
        cfg = {}
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.split:
        cfg["split"] = args.split
    # Preserve explicit --max_samples all (parsed as None) so it can override
    # config defaults like max_samples: 100.
    max_samples_was_provided = any(
        token == "--max_samples" or token.startswith("--max_samples=")
        for token in sys.argv[1:]
    )
    if max_samples_was_provided:
        cfg["max_samples"] = args.max_samples
    if args.top_k is not None:
        cfg["top_k"] = args.top_k
    if args.top_m is not None:
        cfg["top_m"] = args.top_m
    if getattr(args, "encoder_model", None):
        cfg["encoder_model"] = args.encoder_model
    if getattr(args, "generator_model", None):
        cfg["generator_model"] = args.generator_model
    if args.metric:
        cfg["metric"] = args.metric
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.retriever_index_dir:
        cfg["retriever_index_dir"] = args.retriever_index_dir
    if args.no_generation:
        cfg["run_generation"] = False
    if args.max_new_tokens is not None:
        cfg["max_new_tokens"] = args.max_new_tokens
    if args.num_beams is not None:
        cfg["num_beams"] = args.num_beams
    if args.include_rds:
        cfg["include_rds"] = True
    if getattr(args, "objective", None):
        cfg["alpha_objective"] = args.objective
    if getattr(args, "alpha_objective", None):
        cfg["alpha_objective"] = args.alpha_objective

    device_was_provided = any(
        token == "--device" or token.startswith("--device=")
        for token in sys.argv[1:]
    )
    if device_was_provided:
        cfg["device"] = args.device

    if getattr(args, "batch_size", None) is not None:
        cfg["batch_size"] = args.batch_size

    if getattr(args, "coarse_to_fine", False):
        expanded = build_coarse_to_fine_alpha_grid(
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            alpha_step=args.alpha_step,
            fine_step=args.fine_step,
            dead_zone_min=args.dead_zone_min,
        )
        cfg["rng_score_alphas"] = expanded  # Seg-Score excluded (2026-07-16)
        print(
            "   [coarse_to_fine] using expanded alpha grid "
            f"[{expanded[0]:.4g}, {expanded[-1]:.4g}] with {len(expanded)} values"
        )

    # --alpha_json: restrict alpha grids to the val-optimal values
    if getattr(args, "alpha_json", None):
        alpha_results = load_alphas(args.alpha_json)
        dataset = cfg.get("dataset", "hotpotqa")
        encoder = resolve_model(
            cfg.get("encoder_model") or default_encoder(cfg.get("device", DEFAULT_DEVICE)),
            ENCODER_ALIASES,
        )
        setting = f"{dataset}/{encoder}"
        alpha_objective = cfg.get("alpha_objective", cfg.get("objective", "apd"))
        generator_for_objective = None
        if alpha_objective in {"em", "f1"}:
            device = normalize_device(cfg.get("device", DEFAULT_DEVICE))
            generator_for_objective = resolve_model(
                cfg.get("generator_model") or default_generator(device),
                GENERATOR_ALIASES,
            )
        for score_type, cfg_key in [("RNG-Score", "rng_score_alphas"),
                                     ("Seg-Score", "seg_score_alphas")]:
            r = lookup_alpha(
                alpha_results,
                score_type,
                setting=setting,
                objective=alpha_objective,
                generator_model=generator_for_objective,
            )
            if r is not None:
                cfg[cfg_key] = [r.alpha_star]
                print(f"   [{score_type}] using val-optimal α*={r.alpha_star:.4g} "
                      f"(objective={r.objective}, val={r.val_score:.4f})")
            else:
                print(
                    f"   [{score_type}] no alpha found for objective={alpha_objective!r} "
                    f"and setting={setting!r}; keeping configured alpha grid."
                )

    return cfg


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Build method list from config
# ---------------------------------------------------------------------------

def _build_methods(cfg: Dict) -> List[Tuple[str, Callable[..., List[int]]]]:
    """Build the list of (method_name, rerank_fn) pairs from a config dict.

    Each callable has the unified signature::

        rerank_fn(passage_embs, query_emb, scores, k) -> List[int]

    where ``passage_embs`` is (m, D) float32, ``query_emb`` is (D,) float32,
    ``scores`` is (m,) float32, and the return value is a list of at most k
    integer indices into the passage pool.

    Parameters
    ----------
    cfg : dict
        Evaluation configuration containing at minimum the key ``"top_k"``
        (int) and optionally ``"metric"`` (str), ``"mmr_lambdas"`` (list),
        ``"rng_score_alphas"`` (list), ``"include_rds"`` (bool),
        ``"rds_alphas"`` (list), and ``"rads_alphas"`` (list).

    Returns
    -------
    methods : list of (str, callable)
        Ordered list of ``(method_name, rerank_fn)`` pairs.  The following
        methods are always included: kNN, Maxmin, Greedy-DPP, and RNG-Score.
        MMR is added once per element of ``cfg["mmr_lambdas"]``; RDS and RADS
        are added only when ``cfg["include_rds"]`` is True.  Seg-Score is
        added once per element of ``cfg["seg_score_alphas"]``.
    """
    k = cfg["top_k"]
    metric = cfg.get("metric", "cosine")
    methods: List[Tuple[str, Callable[..., List[int]]]] = []

    # --- kNN ---
    methods.append(
        ("kNN", lambda embs, qemb, scores, _k=k: rerank_knn(scores, _k))
    )

    # --- MMR ---
    for lam in cfg.get("mmr_lambdas", _MMR_LAMBDAS):
        lam_ = lam
        methods.append(
            (
                f"MMR(λ={lam_})",
                lambda embs, qemb, scores, _k=k, _l=lam_: rerank_mmr(
                    embs, qemb, scores, _k, _l
                ),
            )
        )

    # --- Maxmin ---
    methods.append(
        ("Maxmin", lambda embs, qemb, scores, _k=k: rerank_maxmin(embs, scores, _k))
    )

    # --- Greedy DPP ---
    methods.append(
        (
            "Greedy-DPP",
            lambda embs, qemb, scores, _k=k: rerank_greedy_dpp(embs, scores, _k),
        )
    )

    # --- RDS / RADS (opt-in only) ---
    if cfg.get("include_rds", False):
        for alpha in cfg.get("rds_alphas", [0.0, 0.1, 0.3]):
            a_ = alpha
            methods.append(
                (
                    f"RDS(α={a_})",
                    lambda embs, qemb, scores, _k=k, _a=a_, _m=metric: rerank_rds(
                        embs, qemb, _k, _a, _m
                    ),
                )
            )
        for alpha in cfg.get("rads_alphas", [0.0, 0.1, 0.3]):
            a_ = alpha
            methods.append(
                (
                    f"RADS(α={a_})",
                    lambda embs, qemb, scores, _k=k, _a=a_, _m=metric: rerank_rads(
                        embs, qemb, _k, _a, _m
                    ),
                )
            )

    # --- RNG-Score ---
    for alpha in cfg.get("rng_score_alphas", _ALPHA_GRID):
        a_ = alpha
        methods.append(
            (
                f"RNG-Score(α={a_})",
                lambda embs, qemb, scores, _k=k, _a=a_, _m=metric: rerank_rng_score(
                    embs, qemb, _k, _a, _m
                ),
            )
        )

    # --- Seg-Score (RNG-Score 2 with alpha sweep) ---
    for alpha in cfg.get("seg_score_alphas", _ALPHA_GRID):
        a_ = alpha
        methods.append(
            (
                f"Seg-Score(α={a_})",
                lambda embs, qemb, scores, _k=k, _a=a_, _m=metric: rerank_rng_score2(
                    embs, qemb, _k, _a, _m
                ),
            )
        )

    return methods


# ---------------------------------------------------------------------------
# Statistical summaries
# ---------------------------------------------------------------------------

def _relevance_source(gold_titles: List[str], answers: List[str]) -> str:
    if gold_titles:
        return "gold_titles"
    if answers:
        return "answer_string"
    return "none"


def _build_summary_rows(
    acc: Dict[str, Dict[str, List[float]]],
    metric_order: List[Tuple[str, str]],
) -> List[Dict]:
    """Convert raw per-query metric lists into mean summary rows."""
    rows: List[Dict] = []
    for method_name, metrics in acc.items():
        row: Dict[str, str] = {"Method": method_name}
        has_values = False
        for metric_key, label in metric_order:
            values = metrics.get(metric_key, [])
            if not values:
                row[label] = ""
                continue
            has_values = True
            row[label] = f"{np.mean(values):.3f}"
        if has_values:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    cfg: Dict,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict],
           List[Dict], List[Dict], List[Dict]]:
    """Run the full evaluation pipeline and aggregate per-method metrics.

    The pipeline:

    1. Load the dataset specified by ``cfg["dataset"]``.
    2. Initialise the SentenceTransformer encoder.
    3. (Optionally) load the Flan-T5 generator.
    4. Build the method list via :func:`_build_methods`.
    5. For each example, embed passages + query, compute cosine scores, then
       apply each re-ranking method to select top-k passages.
    6. Compute retrieval metrics (gold recall, avg pairwise distance, Vendi
       Score) and, if generation is enabled, EM and F1.
    7. Aggregate per-example values and return summary rows.

    Parameters
    ----------
    cfg : dict
        Evaluation configuration.  Required keys:

        * ``"dataset"``        : str — ``"hotpotqa"`` or ``"nq_open"``.
        * ``"encoder_model"``  : str — SentenceTransformer model identifier.
        * ``"top_k"``          : int — number of passages to select per method.

        Optional keys (with defaults):

        * ``"split"``          : str — dataset split (default ``"validation"``).
        * ``"max_samples"``    : int or None — cap on examples (default None).
        * ``"top_m"``          : int — candidate pool size (default 100).
        * ``"metric"``         : str — distance metric for RDS/RADS (default ``"cosine"``).
        * ``"run_generation"`` : bool — whether to run Flan-T5 (default True).
        * ``"generator_model"``: str — Flan-T5 model identifier.
        * ``"max_new_tokens"`` : int — generation length (default 64).
        * ``"num_beams"``      : int — beam width (default 4).
        * ``"mmr_lambdas"``    : list of float — MMR λ values.
        * ``"rng_score_alphas"``: list of float — RNG-Score α values (may be negative).
        * ``"seg_score_alphas"``: list of float — Seg-Score α values (may be negative).
        * ``"include_rds"``    : bool — include RDS/RADS methods (default False).
        * ``"rds_alphas"``     : list of float — RDS α values (used when include_rds is True).
        * ``"rads_alphas"``    : list of float — RADS α values (used when include_rds is True).
    """

    dataset_name = cfg["dataset"]
    split = cfg.get("split", "validation")
    max_samples = cfg.get("max_samples", None)
    retriever_index_dir = cfg.get("retriever_index_dir")
    print(f"\n── Loading dataset: {dataset_name} / {split} ──")
    if dataset_name == "hotpotqa":
        examples = load_hotpotqa(split=split, max_samples=max_samples)
    elif dataset_name == "hotpotqa_fullwiki":
        examples = load_hotpotqa_fullwiki(split=split, max_samples=max_samples)
    elif dataset_name == "nq_open":
        examples = load_nq_open(split=split, max_samples=max_samples)
    elif dataset_name == "squad":
        examples = load_squad(split=split, max_samples=max_samples)
    elif dataset_name == "2wikimultihopqa":
        examples = load_2wikimultihopqa(split=split, max_samples=max_samples)
    elif dataset_name == "musique":
        examples = load_musique(split=split, max_samples=max_samples)
    elif dataset_name == "nq":
        examples = load_nq_dpr(split=split, max_samples=max_samples)
    elif dataset_name == "trivia":
        examples = load_trivia_dpr(split=split, max_samples=max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")
    print(f"   {len(examples)} examples loaded.")

    print(f"\n── Loading encoder: {cfg['encoder_model']} ──")
    device = normalize_device(cfg.get("device", DEFAULT_DEVICE))
    batch_size = cfg.get("batch_size", 64)
    retriever = DenseRetriever(
        model_name=cfg["encoder_model"],
        device=device,
        batch_size=batch_size,
    )
    corpus_retriever = None
    if retriever_index_dir:
        print(f"\n── Loading retrieval index: {retriever_index_dir} ──")
        corpus_retriever = DenseRetriever(
            model_name=cfg["encoder_model"],
            device=device,
            batch_size=batch_size,
        )
        corpus_retriever.load(retriever_index_dir)

    run_gen = cfg.get("run_generation", True)
    generator = None
    if run_gen:
        print(f"\n── Loading generator: {cfg['generator_model']} ──")
        generator = load_generator(
            model_name=cfg["generator_model"],
            device=device,
            max_new_tokens=cfg.get("max_new_tokens", 64),
            num_beams=cfg.get("num_beams", 4),
        )

    methods = _build_methods(cfg)
    top_m = cfg.get("top_m", 100)
    top_k = cfg["top_k"]

    # Accumulators: method_name → list of metric values
    ret_acc: Dict[str, Dict[str, List[float]]] = {
        name: defaultdict(list) for name, _ in methods
    }
    gen_acc: Dict[str, Dict[str, List[float]]] = {
        name: defaultdict(list) for name, _ in methods
    }
    ret_per_query: List[Dict] = []
    gen_per_query: List[Dict] = []

    # Batch encoding avoids one GPU dispatch per query.
    _all_questions = [ex["question"] for ex in examples]
    _is_pre_attached = all(ex["passages"] is not None for ex in examples)

    if _is_pre_attached:
        _all_pools = [ex["passages"][:top_m] for ex in examples]
        print("\n── Pre-computing embeddings ──")
        _q_embs_all, _p_embs_flat, _p_offsets = encode_queries_and_passages(
            retriever, _all_questions, _all_pools
        )
    else:
        if corpus_retriever is None:
            raise ValueError(
                "Dataset examples do not include passages. Set retriever_index_dir "
                "to a FAISS index built with DenseRetriever.save()."
            )
        print("\n── Pre-computing corpus search ──")
        print(f"   Encoding and searching {len(_all_questions)} queries …")
        _q_embs_all, _scores_all, _idxs_all = corpus_retriever.search_batch(
            _all_questions, top_m
        )

    for ei, ex in enumerate(tqdm(examples, desc="Evaluating")):
        example_id = ex.get("id", "")
        question = _all_questions[ei]
        answers = ex["answers"]
        gold_titles = ex.get("gold_titles") or []

        if _is_pre_attached:
            _s, _e = int(_p_offsets[ei]), int(_p_offsets[ei + 1])
            passages_pool = _all_pools[ei]
            q_emb = _q_embs_all[ei]
            p_embs = _p_embs_flat[_s:_e]
            scores = p_embs @ q_emb
        else:
            _ridxs = _idxs_all[ei].tolist()
            passages_pool = [corpus_retriever.passages[i] for i in _ridxs]
            q_emb = _q_embs_all[ei]
            p_embs = corpus_retriever.passage_embeddings[_ridxs]
            scores = _scores_all[ei]

        relevance = relevance_labels(passages_pool, gold_titles=gold_titles, answers=answers)
        relevance_source = _relevance_source(gold_titles, answers)
        subtopic_sets = subtopic_coverage_sets(passages_pool, gold_titles=gold_titles)

        # Per-method evaluation
        # Phase A – reranking + retrieval metrics; collect unique selections.
        _results: Dict[str, tuple] = {}
        _unique_sels: Dict[tuple, List[Dict]] = {}
        for method_name, rerank_fn in methods:
            t_start = time.perf_counter()
            selected_idx = rerank_fn(p_embs, q_emb, scores, top_k)
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            if not selected_idx:
                continue
            selected_embs = p_embs[selected_idx]
            gr = recall_at_k(selected_idx, relevance, k=top_k)
            ndcg = ndcg_from_relevance(selected_idx, relevance, k=top_k)
            rr = mrr_from_relevance(selected_idx, relevance)
            apd = avg_pairwise_distance(selected_embs)
            vs = vendi_score(selected_embs)
            an = alpha_ndcg_at_k(selected_idx, subtopic_sets, k=top_k, alpha_r=0.5)
            sr = subtopic_recall_at_k(selected_idx, subtopic_sets, k=top_k)
            ei = err_ia_at_k(selected_idx, subtopic_sets, k=top_k)
            _results[method_name] = (selected_idx, latency_ms, gr, ndcg, rr, apd, vs, an, sr, ei)
            sel_key = tuple(selected_idx)
            if sel_key not in _unique_sels:
                _unique_sels[sel_key] = [passages_pool[i] for i in selected_idx]

        # Phase B – batch-generate for all unique passage selections.
        _gen_cache: Dict[tuple, Optional[Tuple[float, float, float]]] = {}
        if run_gen and generator is not None and _unique_sels:
            _keys = list(_unique_sels.keys())
            _preds = generator.generate_batch(
                [question] * len(_keys),
                [_unique_sels[k] for k in _keys],
            )
            for _key, _pred in zip(_keys, _preds):
                _em = max(exact_match(_pred, a) for a in answers) if answers else 0.0
                _f1 = max(f1_score_single(_pred, a) for a in answers) if answers else 0.0
                _ctx = [p.get("text", "") for p in _unique_sels[_key]]
                _hall = hallucination_rate(_pred, _ctx)
                _gen_cache[_key] = (_em, _f1, _hall)

        # Phase C – accumulate metrics.
        for method_name, (selected_idx, latency_ms, gr, ndcg, rr, apd, vs, an, sr, ei) in _results.items():
            ret_acc[method_name]["recall"].append(gr)
            ret_acc[method_name]["ndcg"].append(ndcg)
            ret_acc[method_name]["mrr"].append(rr)
            ret_acc[method_name]["avg_pairwise_dist"].append(apd)
            ret_acc[method_name]["vendi_score"].append(vs)
            ret_acc[method_name]["alpha_ndcg"].append(an)
            ret_acc[method_name]["subtopic_recall"].append(sr)
            ret_acc[method_name]["err_ia"].append(ei)
            ret_acc[method_name]["latency_ms"].append(latency_ms)
            ret_per_query.append(
                {
                    "Query ID": example_id,
                    "Method": method_name,
                    "Relevance Source": relevance_source,
                    "Num Relevant": str(int(np.sum(relevance))),
                    "Recall@k": f"{gr:.6f}",
                    "NDCG@k": f"{ndcg:.6f}",
                    "MRR": f"{rr:.6f}",
                    "Avg Pair Dist": f"{apd:.6f}",
                    "Vendi Score": f"{vs:.6f}",
                    "alpha-NDCG@k": f"{an:.6f}",
                    "S-Recall@k": f"{sr:.6f}",
                    "ERR-IA@k": f"{ei:.6f}",
                    "Latency (ms)": f"{latency_ms:.6f}",
                }
            )
            gen_triple = _gen_cache.get(tuple(selected_idx))
            if gen_triple is not None:
                best_em, best_f1, hall = gen_triple
                gen_acc[method_name]["em"].append(best_em)
                gen_acc[method_name]["f1"].append(best_f1)
                gen_acc[method_name]["hallucination_rate"].append(hall)
                gen_per_query.append(
                    {
                        "Query ID": example_id,
                        "Method": method_name,
                        "EM": f"{best_em:.6f}",
                        "F1": f"{best_f1:.6f}",
                        "Hallucination Rate": f"{hall:.6f}",
                    }
                )

    # 6. Aggregate
    retrieval_metric_order = [
        ("recall", "Recall@k"),
        ("ndcg", "NDCG@k"),
        ("mrr", "MRR"),
        ("alpha_ndcg", "alpha-NDCG@k"),
        ("subtopic_recall", "S-Recall@k"),
        ("err_ia", "ERR-IA@k"),
        ("avg_pairwise_dist", "Avg Pair Dist"),
        ("vendi_score", "Vendi Score"),
        ("latency_ms", "Latency (ms)"),
    ]
    generation_metric_order = [
        ("em", "EM"),
        ("f1", "F1"),
        ("hallucination_rate", "Hallucination Rate"),
    ]

    ret_rows = _build_summary_rows(
        ret_acc,
        retrieval_metric_order,
    )
    gen_rows = _build_summary_rows(
        gen_acc,
        generation_metric_order,
    )

    ret_table_rows = [
        {
            "Method": row["Method"],
            "Recall@k": row["Recall@k"],
            "NDCG@k": row["NDCG@k"],
            "MRR": row["MRR"],
            "alpha-NDCG@k": row.get("alpha-NDCG@k", ""),
            "S-Recall@k": row.get("S-Recall@k", ""),
            "ERR-IA@k": row.get("ERR-IA@k", ""),
            "Avg Pair Dist": row["Avg Pair Dist"],
            "Vendi Score": row["Vendi Score"],
            "Latency (ms)": row["Latency (ms)"],
        }
        for row in ret_rows
    ]
    gen_table_rows = [
        {
            "Method": row["Method"],
            "EM": row["EM"],
            "F1": row["F1"],
            "Hallucination Rate": row["Hallucination Rate"],
        }
        for row in gen_rows
        if row.get("EM")
    ]

    # Deterministic retrieval and beam-generation metrics use query-bootstrap
    # intervals and paired Wilcoxon tests against kNN.
    ret_significance = significance_table(ret_acc, retrieval_metric_order,
                                          baseline="kNN")
    gen_significance = significance_table(gen_acc, generation_metric_order,
                                          baseline="kNN")

    return (
        ret_table_rows,
        gen_table_rows,
        ret_rows,
        gen_rows,
        ret_per_query,
        gen_per_query,
        ret_significance,
        gen_significance,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate RDS vs. baselines on HotpotQA / NQ-Open."
    )
    p.add_argument("--config", default=_DEFAULT_CONFIG, help="YAML config file")
    p.add_argument("--dataset", choices=["hotpotqa", "hotpotqa_fullwiki", "nq_open", "squad",
                                          "2wikimultihopqa", "musique", "nq", "trivia"],
                   default=None)
    p.add_argument("--split", choices=["train", "validation", "test"], default=None)
    p.add_argument("--max_samples", type=int_or_all, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_m", type=int, default=None,
                   help="Candidate pool size fed to every re-ranker (default 100).")
    p.add_argument("--encoder_model", default=None,
                   help="Encoder alias or HuggingFace ID. "
                        f"Aliases: {', '.join(ENCODER_ALIASES)}. "
                        "Default: bge-m3 on CUDA, minilm on CPU.")
    p.add_argument(
        "--metric",
        choices=["euclidean", "angular", "cosine"],
        default=None,
    )
    p.add_argument("--output_dir", default=None)
    p.add_argument(
        "--retriever_index_dir",
        default=None,
        help="Path to a DenseRetriever FAISS index for datasets without attached passages.",
    )
    p.add_argument(
        "--no_generation",
        action="store_true",
        help="Skip generation; only evaluate retrieval metrics.",
    )
    p.add_argument("--max_new_tokens", type=int, default=None,
                   help="Maximum number of tokens generated per answer (overrides config).")
    p.add_argument("--num_beams", type=int, default=None,
                   help="Beam width for generation (overrides config).")
    p.add_argument("--generator_model", default=None,
                   help="Generator alias or HuggingFace ID. "
                        f"Aliases: {', '.join(GENERATOR_ALIASES)}. "
                        "Default: flan-t5-base on CUDA, flan-t5-small on CPU.")
    p.add_argument(
        "--include_rds",
        action="store_true",
        help="Include RDS and RADS methods in the evaluation (off by default).",
    )
    p.add_argument(
        "--objective",
        choices=OBJECTIVES,
        default=None,
        help="Alias for --alpha_objective (single value).",
    )
    p.add_argument(
        "--alpha_json",
        default=None,
        help="Path to alpha_results.json (from learn_alpha.py). When provided, "
             "restricts RNG-Score and Seg-Score to the val-optimal α* only.",
    )
    p.add_argument(
        "--alpha_objective",
        choices=OBJECTIVES,
        default=None,
        help="Objective key used to lookup α* from --alpha_json (e.g. em, f1, alpha_ndcg).",
    )
    p.add_argument(
        "--coarse_to_fine",
        action="store_true",
        help="Use an expanded fine-resolution alpha sweep for RNG-Score and Seg-Score.",
    )
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
        help="Also save per-query diagnostics CSVs.",
    )
    p.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help='Device for model inference, e.g. "cpu", "cuda", "cuda:0" '
             '(default: cuda; use cpu explicitly for CPU execution).',
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoder inference (default: 64). "
             "Increase on GPU for better utilisation; reduce if you run out of memory.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)

    # Resolve device-aware model defaults and expand any short aliases.
    device = normalize_device(cfg.get("device", DEFAULT_DEVICE))
    using_default_config = os.path.abspath(args.config) == os.path.abspath(_DEFAULT_CONFIG)

    # Keep default-config behaviour device-aware while preserving explicit
    # overrides from CLI/custom config files.
    if using_default_config and args.encoder_model is None and cfg.get("encoder_model") in {None, "minilm"}:
        cfg["encoder_model"] = default_encoder(device)
    if (
        using_default_config
        and args.generator_model is None
        and cfg.get("run_generation", True)
        and cfg.get("generator_model") in {None, "flan-t5-small"}
    ):
        cfg["generator_model"] = default_generator(device)

    cfg["encoder_model"] = resolve_model(
        cfg.get("encoder_model") or default_encoder(device), ENCODER_ALIASES
    )
    if cfg.get("run_generation", True):
        cfg["generator_model"] = resolve_model(
            cfg.get("generator_model") or default_generator(device), GENERATOR_ALIASES
        )

    t0 = time.time()
    (
        ret_rows,
        gen_rows,
        ret_summary_rows,
        gen_summary_rows,
        ret_per_query,
        gen_per_query,
        ret_significance,
        gen_significance,
    ) = run_evaluation(cfg)
    elapsed = time.time() - t0

    output_dir = cfg.get("output_dir", "results")

    if cfg.get("print_table", True):
        print("\n══ Retrieval metrics ══")
        print(tabulate(ret_rows, headers="keys", tablefmt="github"))
        if gen_rows:
            print("\n══ Generation metrics ══")
            print(tabulate(gen_rows, headers="keys", tablefmt="github"))

    print(f"\nTotal elapsed: {elapsed:.1f}s")

    run_params = {
        "script": "evaluate.py",
        "config": args.config,
        "dataset": cfg.get("dataset"),
        "split": cfg.get("split"),
        "max_samples": cfg.get("max_samples"),
        "top_m": cfg.get("top_m"),
        "top_k": cfg.get("top_k"),
        "metric": cfg.get("metric"),
        "device": device,
        "run_generation": cfg.get("run_generation"),
        "include_rds": cfg.get("include_rds"),
        "alpha_json": cfg.get("alpha_json"),
        "alpha_objective": cfg.get("alpha_objective"),
        "coarse_to_fine": cfg.get("coarse_to_fine"),
        "alpha_grid": cfg.get("alpha_grid"),
    }
    run_dir = make_run_dir(output_dir, cfg.get("dataset", "run"), run_params)

    save_csv(
        attach_run_params(ret_rows, run_params),
        os.path.join(run_dir, "results_retrieval.csv"),
    )
    save_csv(
        attach_run_params(gen_rows, run_params),
        os.path.join(run_dir, "results_generation.csv"),
    )
    # Mean + 95% across-query bootstrap CI + Wilcoxon vs kNN, one row per
    # (method, metric). The error bars and significance for the main tables.
    save_csv(attach_run_params(ret_significance, run_params),
             os.path.join(run_dir, "results_retrieval_significance.csv"))
    if gen_significance:
        save_csv(attach_run_params(gen_significance, run_params),
                 os.path.join(run_dir, "results_generation_significance.csv"))
    if args.save_per_query:
        save_csv(ret_per_query, os.path.join(run_dir, "results_retrieval_per_query.csv"))
        save_csv(gen_per_query, os.path.join(run_dir, "results_generation_per_query.csv"))


if __name__ == "__main__":
    main()
