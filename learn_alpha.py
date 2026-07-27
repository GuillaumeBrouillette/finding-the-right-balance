#!/usr/bin/env python3
"""
learn_alpha.py  –  Find the optimal α for RNG-Score / Seg-Score (and CE variants)
==================================================================================

Performs a fine-grained grid search over α (and β for S1-Blend) on a held-out
validation split.  Saves the best hyper-parameters to a JSON file so that
evaluate.py, evaluate_beir.py, and evaluate_cross_encoder.py can load them
directly rather than repeating the sweep.

Supported settings
------------------
  Embedding-based retrieval (no cross-encoder):
    RNG-Score, Seg-Score
  Cross-encoder pipeline (requires --ce_model):
    S1-Blend(α, β), S2-V1(α), S2-V2(α), S2-V3(α)

Supported datasets
------------------
  hotpotqa, squad  (passages already attached)
  beir/<task>      e.g. beir/scifact, beir/fiqa, beir/trec-covid

Usage
-----
    # Standard α sweep on HotpotQA val set (default objective: apd)
    python learn_alpha.py --dataset hotpotqa --objective apd

    # Fine grid
    python learn_alpha.py --alpha_min -0.5 --alpha_max 0.5 --alpha_step 0.05

    # With cross-encoder
    python learn_alpha.py --ce_model cross-encoder/ms-marco-MiniLM-L-6-v2

    # On a BEIR task
    python learn_alpha.py --dataset beir/scifact --objective ndcg

Output
------
    results/alpha_results.json   (default --output_json)
    results/alpha_results_run_params.json   (same run parameters)
    console table of all (α, objective) pairs
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

from alpha_selection import (
    AlphaResult,
    OBJECTIVES,
    accumulate_alpha,
    accumulate_blend,
    average_accumulator,
    find_optimal_alpha,
    find_optimal_blend,
    get_objective_value,
    print_alpha_table,
    print_sweep_diagnostics,
    save_alphas,
)
from data.loaders import load_hotpotqa, load_squad
from evaluation.metrics import (
    alpha_ndcg_at_k,
    avg_pairwise_distance,
    exact_match,
    f1_score_single,
    ndcg_from_relevance,
    recall_at_k,
    subtopic_coverage_sets,
    vendi_score,
)
from generation.generator import load_generator
from retrieval.rerankers import (
    rerank_ce_rng_blended,
    rerank_ce_semimetric,
    rerank_rng_score,
    rerank_rng_score2,
)
from retrieval.retriever import DenseRetriever
from run_utils import (
    CE_ALIASES,
    ENCODER_ALIASES,
    GENERATOR_ALIASES,
    default_ce_model,
    default_encoder,
    default_generator,
    int_or_all,
    make_run_dir,
    normalize_device,
    resolve_model,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    # Add dataset validation
    def _dataset_arg(value: str) -> str:
        dataset = value.strip().lower()
        if dataset in {"hotpotqa", "hotpotqa_fullwiki", "squad"} or dataset.startswith("beir/"):
            return dataset
        raise argparse.ArgumentTypeError(
            "invalid dataset. Use hotpotqa, hotpotqa_fullwiki, squad, or beir/<task> (e.g. beir/scifact)."
        )
    p = argparse.ArgumentParser(
        description="Grid search for optimal α (and β) in RNG-Score / CE variants."
    )
    p.add_argument("--dataset", default="hotpotqa", type=_dataset_arg,
                   help="hotpotqa | squad | beir/<task>  (e.g. beir/scifact)")
    p.add_argument("--split", default="validation")
    p.add_argument("--max_samples", type=int_or_all, default=200,
                   help="Max examples to use (val+test combined); pass 'all' for no cap.")
    p.add_argument("--val_fraction", type=float, default=0.2,
                   help="Fraction used for validation (alpha selection).")
    p.add_argument("--val_seed", type=int, default=0)

    p.add_argument("--encoder_model", default=None,
                   help="Encoder alias or HuggingFace ID. "
                        f"Aliases: {', '.join(ENCODER_ALIASES)}. "
                        "Default: bge-m3 on CUDA, minilm on CPU.")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--top_m", type=int, default=50)
    p.add_argument("--metric", default="cosine",
                   choices=["cosine", "euclidean", "angular"])

    p.add_argument("--alpha_min", type=float, default=-1.0)
    p.add_argument("--alpha_max", type=float, default=1.0)
    p.add_argument("--alpha_step", type=float, default=0.25)

    p.add_argument("--coarse_to_fine", action="store_true",
                   help="Two-stage search: coarse sweep over [alpha_min, alpha_max], "
                        "then fine sweep of width ±alpha_step around the best value. "
                        "Extends by one coarse step if the optimum is at a boundary.")
    p.add_argument("--fine_step", type=float, default=None,
                   help="Step size for the fine stage (default: alpha_step / 5).")
    p.add_argument("--dead_zone_min", type=float, default=-1,
                   help="Hard lower bound for boundary extension (cosine dead zone). "
                        "Extension left is skipped when alpha_min <= dead_zone_min. "
                        "Default: -1.0.")

    p.add_argument("--objective", choices=OBJECTIVES, default="apd")
    p.add_argument("--lambda_div", type=float, default=0.5,
                   help="Weight on diversity in recall_vendi objective.")

    # Cross-encoder
    p.add_argument("--ce_model", default=None,
                   help="Cross-encoder alias or HuggingFace ID; if given, also runs CE strategies. "
                        f"Aliases: {', '.join(CE_ALIASES)}.")
    p.add_argument("--beta_min", type=float, default=0.0)
    p.add_argument("--beta_max", type=float, default=1.0)
    p.add_argument("--beta_step", type=float, default=0.2)

    # Generation (for generative objectives)
    p.add_argument("--generator_model", default=None,
                   help="Generator alias or HuggingFace ID (used when objective is em/f1). "
                        f"Aliases: {', '.join(GENERATOR_ALIASES)}. "
                        "Default: llama-3.2-3b on CUDA, flan-t5-small on CPU.")

    p.add_argument("--output_json", default="results/alpha_results.json")
    p.add_argument("--index_dir", default=None,
                   help="FAISS index cache dir (for BEIR datasets).")
    p.add_argument(
        "--device",
        default="cpu",
        help='Device for model inference, e.g. "cpu", "cuda", "cuda:0" (default: cpu).',
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_examples(args: argparse.Namespace):
    """Load examples, splitting into val and test portions."""
    dataset = args.dataset.lower()

    if dataset == "hotpotqa":
        all_ex = load_hotpotqa(split=args.split, max_samples=args.max_samples)
    elif dataset == "hotpotqa_fullwiki":
        all_ex = load_hotpotqa(split=args.split, max_samples=args.max_samples, config="fullwiki")
    elif dataset == "squad":
        all_ex = load_squad(split=args.split, max_samples=args.max_samples)
    elif dataset.startswith("beir/"):
        task = dataset.split("/", 1)[1]
        from data.loaders import load_beir_dataset
        all_ex, corpus = load_beir_dataset(task, split="test",
                                           max_queries=args.max_samples)
        return all_ex, corpus
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. "
                         "Use hotpotqa, squad, or beir/<task>.")
    return all_ex, None


def _val_test_split(examples, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(examples)).tolist()
    n_val = max(1, int(len(examples) * val_fraction))
    val_idx = set(idx[:n_val])
    val_ex = [e for i, e in enumerate(examples) if i in val_idx]
    test_ex = [e for i, e in enumerate(examples) if i not in val_idx]
    return val_ex, test_ex


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _encode_passages(retriever: DenseRetriever, passages: List[Dict]) -> np.ndarray:
    texts = [
        ((p.get("title") or "") + ": " + (p.get("text") or "")).strip()
        for p in passages
    ]
    return retriever.encode(texts, normalize=True)


def _encode_query(retriever: DenseRetriever, question: str) -> np.ndarray:
    return retriever.encode([question], normalize=True)[0]


# ---------------------------------------------------------------------------
# Per-example relevance helpers
# ---------------------------------------------------------------------------

def _relevance_vector(passages: List[Dict], gold_titles: Optional[List[str]],
                      qrels: Optional[Dict] = None) -> np.ndarray:
    """Binary relevance label for each passage in the pool."""
    if gold_titles:
        gt_set = {t.lower().strip() for t in gold_titles}
        return np.array([
            1 if (p.get("title") or "").lower().strip() in gt_set else 0
            for p in passages
        ], dtype=np.int32)
    if qrels:
        return np.array([
            1 if (p.get("id") or p.get("pid") or "") in qrels else 0
            for p in passages
        ], dtype=np.int32)
    return np.zeros(len(passages), dtype=np.int32)


def _subtopic_sets_for(passages: List[Dict], gold_titles: Optional[List[str]],
                       qrels: Optional[Dict] = None):
    if gold_titles:
        return subtopic_coverage_sets(passages, gold_titles=gold_titles)
    if qrels:
        return subtopic_coverage_sets(passages, qrel_ids=list(qrels.keys()))
    return subtopic_coverage_sets(passages)


# ---------------------------------------------------------------------------
# BEIR: build FAISS index, retrieve top-m, attach passages to examples
# ---------------------------------------------------------------------------

def _attach_beir_passages(examples, corpus: Dict, retriever: DenseRetriever,
                          top_m: int, index_dir: Optional[str]) -> List[Dict]:
    """Retrieve top-m passages for each BEIR query and attach them."""
    import faiss

    doc_ids = list(corpus.keys())
    doc_texts = [
        ((corpus[d].get("title") or "") + ": " + (corpus[d].get("text") or "")).strip()
        for d in doc_ids
    ]

    index_path = None
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)
        index_path = os.path.join(index_dir, f"beir_learn_alpha.faiss")

    if index_path and os.path.isfile(index_path):
        print("   Loading cached FAISS index …")
        index = faiss.read_index(index_path)
        doc_embs = None
    else:
        print("   Encoding corpus …")
        doc_embs = retriever.encode(doc_texts, normalize=True, batch_size=128,
                                    show_progress=True)
        index = faiss.IndexFlatIP(doc_embs.shape[1])
        index.add(doc_embs)
        if index_path:
            faiss.write_index(index, index_path)

    enriched = []
    for ex in tqdm(examples, desc="Retrieving BEIR passages"):
        q_emb = _encode_query(retriever, ex["question"])
        sims, nn_idx = index.search(q_emb[np.newaxis, :].astype(np.float32), top_m)
        passages = []
        for rank, (idx, sim) in enumerate(zip(nn_idx[0], sims[0])):
            did = doc_ids[idx]
            passages.append({
                "id": did,
                "title": corpus[did].get("title", ""),
                "text": corpus[did].get("text", ""),
                "score": float(sim),
            })
        enriched.append({**ex, "passages": passages})
    return enriched


# ---------------------------------------------------------------------------
# Main sweep logic
# ---------------------------------------------------------------------------

def _sweep_standard(
    val_examples,
    retriever: DenseRetriever,
    alphas: np.ndarray,
    args: argparse.Namespace,
    generator: Optional[object],
    setting: str,
    label: str = "Val sweep (standard)",
    print_diagnostics: bool = True,
) -> Tuple[List[AlphaResult], Dict[str, float], Dict[str, float]]:
    """Sweep RNG-Score and Seg-Score over *alphas* on the validation split.

    Returns
    -------
    results : List[AlphaResult]
        Best-alpha results for RNG-Score and Seg-Score.
    avg_rng : dict  {key: mean_objective}
    avg_seg : dict  {key: mean_objective}
        Per-alpha averages, exposed so callers can merge multiple sweeps.
    """
    acc_rng: Dict[str, List[float]] = {}
    acc_seg: Dict[str, List[float]] = {}

    for ex in tqdm(val_examples, desc=label):
        passages = (ex.get("passages") or [])[:args.top_m]
        if not passages:
            continue
        q_emb = _encode_query(retriever, ex["question"])
        p_embs = _encode_passages(retriever, passages)
        relevance = _relevance_vector(passages,
                                      ex.get("gold_titles"),
                                      ex.get("qrels"))
        stopic = _subtopic_sets_for(passages,
                                    ex.get("gold_titles"),
                                    ex.get("qrels"))

        for alpha in alphas:
            a = float(alpha)
            for score_fn, acc in [(rerank_rng_score, acc_rng),
                                   (rerank_rng_score2, acc_seg)]:
                sel = score_fn(p_embs, q_emb, args.top_k, a, args.metric)
                m: Dict[str, float] = {
                    "recall": recall_at_k(sel, relevance, args.top_k),
                    "ndcg": ndcg_from_relevance(sel, relevance, args.top_k),
                    "alpha_ndcg": alpha_ndcg_at_k(sel, stopic, args.top_k),
                    "vendi": vendi_score(p_embs[sel]) if len(sel) > 1 else 0.0,
                    "apd": avg_pairwise_distance(p_embs[sel]) if len(sel) > 1 else 0.0,
                }
                if args.objective in {"em", "f1"} and generator is not None:
                    ps = [passages[i] for i in sel]
                    pred = generator.generate(ex["question"], ps)
                    answers = ex.get("answers")
                    if answers is None:
                        singular = ex.get("answer")
                        answers = [singular] if singular else []
                    elif isinstance(answers, str):
                        answers = [answers]
                    m["em"] = max(exact_match(pred, a) for a in answers) if answers else 0.0
                    m["f1"] = max(f1_score_single(pred, a) for a in answers) if answers else 0.0
                obj_val = get_objective_value(m, args.objective)
                accumulate_alpha(acc, a, obj_val)

    avg_rng = average_accumulator(acc_rng)
    avg_seg = average_accumulator(acc_seg)

    res_rng = find_optimal_alpha(avg_rng, list(alphas), "RNG-Score",
                                 args.objective, setting,
                                 generator_model=(
                                     args.generator_model if args.objective in {"em", "f1"} else None
                                 ))
    res_seg = find_optimal_alpha(avg_seg, list(alphas), "Seg-Score",
                                 args.objective, setting,
                                 generator_model=(
                                     args.generator_model if args.objective in {"em", "f1"} else None
                                 ))

    if print_diagnostics:
        print("\n── RNG-Score sweep (val) ──")
        print_sweep_diagnostics(res_rng)
        print("\n── Seg-Score sweep (val) ──")
        print_sweep_diagnostics(res_seg)

    return [res_rng, res_seg], avg_rng, avg_seg


# ---------------------------------------------------------------------------
# Coarse-to-fine sweep
# ---------------------------------------------------------------------------

def _coarse_to_fine_sweep(
    val_examples,
    retriever: DenseRetriever,
    alphas: np.ndarray,       # initial coarse grid (alpha_min … alpha_max)
    args: argparse.Namespace,
    generator: Optional[object],
    setting: str,
) -> List[AlphaResult]:
    """Three-stage alpha search: coarse sweep → boundary extension → fine sweep.

    Stage 1 – Coarse sweep over *alphas* (user-supplied grid).
    Stage 2 – If the best alpha is at a grid boundary, extend by one coarse step
               in that direction (subject to dead-zone guard for negative alpha).
    Stage 3 – Fine sweep of width ±alpha_step around the best alpha, at step
               fine_step (default alpha_step / 5).

    The three sets of per-alpha means are merged (fine values override coarse for
    overlapping keys) and the global optimum is selected from the merged dict.
    """
    alpha_step = args.alpha_step
    fine_step   = round(getattr(args, "fine_step", alpha_step / 5.0), 8)
    dead_zone_min = getattr(args, "dead_zone_min", -1.0)

    alpha_min = float(alphas[0])
    alpha_max = float(alphas[-1])

    # ── Stage 1: coarse ──────────────────────────────────────────────────────
    print("\n── Coarse sweep [{:.4g}, {:.4g}] step={:.4g} ──".format(
        alpha_min, alpha_max, alpha_step))
    _, avg_rng, avg_seg = _sweep_standard(
        val_examples, retriever, alphas, args, generator, setting,
        label="Coarse sweep", print_diagnostics=False,
    )

    def _best(avg: Dict[str, float]) -> float:
        """Alpha with highest mean objective (tie-breaks toward 0)."""
        res = find_optimal_alpha(avg, [], "tmp", args.objective, "")
        return res.alpha_star

    a_rng = _best(avg_rng)
    a_seg = _best(avg_seg)

    # ── Stage 2: boundary extension ──────────────────────────────────────────
    _EPS = 1e-9

    def _ext_alphas(a_best: float) -> List[float]:
        """Return new alpha values needed to extend past a boundary, if any."""
        if abs(a_best - alpha_min) < _EPS and alpha_min > dead_zone_min + _EPS:
            candidate = round(alpha_min - alpha_step, 8)
            return [candidate] if candidate >= dead_zone_min else []
        if abs(a_best - alpha_max) < _EPS:
            return [round(alpha_max + alpha_step, 8)]
        return []

    ext_needed = sorted(set(_ext_alphas(a_rng) + _ext_alphas(a_seg)))

    if ext_needed:
        print("\n── Boundary extension: {} ──".format(ext_needed))
        _, avg_ext_rng, avg_ext_seg = _sweep_standard(
            val_examples, retriever, np.array(ext_needed), args, generator, setting,
            label="Boundary extension", print_diagnostics=False,
        )
        avg_rng.update(avg_ext_rng)
        avg_seg.update(avg_ext_seg)
        a_rng = _best(avg_rng)
        a_seg = _best(avg_seg)
        print("   Best after extension: RNG α*={:.4g}  Seg α*={:.4g}".format(a_rng, a_seg))

    # ── Stage 3: fine sweep ───────────────────────────────────────────────────
    def _fine_grid(a_center: float) -> np.ndarray:
        lo = round(a_center - alpha_step, 8)
        hi = round(a_center + alpha_step, 8)
        return np.round(np.arange(lo, hi + fine_step / 2, fine_step), 8)

    fine_union = sorted(set(
        _fine_grid(a_rng).tolist() + _fine_grid(a_seg).tolist()
    ))
    print("\n── Fine sweep around α*={:.4g} (RNG) / α*={:.4g} (Seg), step={:.4g} ──".format(
        a_rng, a_seg, fine_step))
    _, avg_fine_rng, avg_fine_seg = _sweep_standard(
        val_examples, retriever, np.array(fine_union), args, generator, setting,
        label="Fine sweep", print_diagnostics=False,
    )

    # Fine values override coarse for any overlapping keys
    avg_rng.update(avg_fine_rng)
    avg_seg.update(avg_fine_seg)

    # ── Final selection ────────────────────────────────────────────────────────
    all_rng_alphas = sorted(float(k.split("=")[1]) for k in avg_rng)
    all_seg_alphas = sorted(float(k.split("=")[1]) for k in avg_seg)

    res_rng = find_optimal_alpha(avg_rng, all_rng_alphas, "RNG-Score",
                                 args.objective, setting,
                                 generator_model=(
                                     args.generator_model if args.objective in {"em", "f1"} else None
                                 ))
    res_seg = find_optimal_alpha(avg_seg, all_seg_alphas, "Seg-Score",
                                 args.objective, setting,
                                 generator_model=(
                                     args.generator_model if args.objective in {"em", "f1"} else None
                                 ))

    print("\n── RNG-Score merged sweep ──")
    print_sweep_diagnostics(res_rng)
    print("\n── Seg-Score merged sweep ──")
    print_sweep_diagnostics(res_seg)

    return [res_rng, res_seg]


def _sweep_ce(
    val_examples,
    retriever: DenseRetriever,
    ce_reranker,
    alphas: np.ndarray,
    betas: np.ndarray,
    args: argparse.Namespace,
    generator: Optional[object],
    setting: str,
) -> List[AlphaResult]:
    """Sweep S1-Blend(α,β) and S2-V1/V2/V3(α) on the validation split."""
    acc_s1: Dict[str, List[float]] = {}
    acc_s2v1: Dict[str, List[float]] = {}
    acc_s2v2: Dict[str, List[float]] = {}
    acc_s2v3: Dict[str, List[float]] = {}

    for ex in tqdm(val_examples, desc="Val sweep (CE)"):
        passages = (ex.get("passages") or [])[:args.top_m]
        if not passages:
            continue
        q_emb = _encode_query(retriever, ex["question"])
        p_embs = _encode_passages(retriever, passages)
        ce_scores = ce_reranker.score(ex["question"], passages)
        relevance = _relevance_vector(passages,
                                      ex.get("gold_titles"),
                                      ex.get("qrels"))
        stopic = _subtopic_sets_for(passages,
                                    ex.get("gold_titles"),
                                    ex.get("qrels"))

        # S1-Blend: joint (α, β) grid
        for alpha in alphas:
            for beta in betas:
                a, b = float(alpha), float(beta)
                sel = rerank_ce_rng_blended(p_embs, q_emb, ce_scores,
                                             args.top_k, a, b, metric=args.metric)
                m = {
                    "recall": recall_at_k(sel, relevance, args.top_k),
                    "ndcg": ndcg_from_relevance(sel, relevance, args.top_k),
                    "alpha_ndcg": alpha_ndcg_at_k(sel, stopic, args.top_k),
                    "vendi": vendi_score(p_embs[sel]) if len(sel) > 1 else 0.0,
                    "apd": avg_pairwise_distance(p_embs[sel]) if len(sel) > 1 else 0.0,
                }
                obj_val = get_objective_value(m, args.objective)
                accumulate_blend(acc_s1, a, b, obj_val)

        # S2 variants: only sweep α
        for alpha in alphas:
            a = float(alpha)
            for variant, acc in [(1, acc_s2v1), (2, acc_s2v2), (3, acc_s2v3)]:
                sel = rerank_ce_semimetric(p_embs, q_emb, ce_scores,
                                           args.top_k, a, variant, metric=args.metric)
                m = {
                    "recall": recall_at_k(sel, relevance, args.top_k),
                    "ndcg": ndcg_from_relevance(sel, relevance, args.top_k),
                    "alpha_ndcg": alpha_ndcg_at_k(sel, stopic, args.top_k),
                    "vendi": vendi_score(p_embs[sel]) if len(sel) > 1 else 0.0,
                    "apd": avg_pairwise_distance(p_embs[sel]) if len(sel) > 1 else 0.0,
                }
                obj_val = get_objective_value(m, args.objective)
                accumulate_alpha(acc, a, obj_val)

    results: List[AlphaResult] = []

    if acc_s1:
        avg_s1 = average_accumulator(acc_s1)
        r = find_optimal_blend(avg_s1, list(alphas), list(betas),
                               "S1-Blend", args.objective, setting,
                               generator_model=(
                                   args.generator_model if args.objective in {"em", "f1"} else None
                               ))
        results.append(r)
        print("\n── S1-Blend sweep (val) ──")
        print_sweep_diagnostics(r)
    for st, acc in [("S2-V1", acc_s2v1), ("S2-V2", acc_s2v2), ("S2-V3", acc_s2v3)]:
        if acc:
            avg = average_accumulator(acc)
            r = find_optimal_alpha(
                avg,
                list(alphas),
                st,
                args.objective,
                setting,
                generator_model=(
                    args.generator_model if args.objective in {"em", "f1"} else None
                ),
            )
            results.append(r)
            print(f"\n── {st} sweep (val) ──")
            print_sweep_diagnostics(r)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Resolve device-aware model defaults and short aliases.
    device = normalize_device(args.device)
    args.encoder_model   = resolve_model(args.encoder_model   or default_encoder(device),   ENCODER_ALIASES)
    args.generator_model = resolve_model(args.generator_model or default_generator(device), GENERATOR_ALIASES)
    if args.ce_model:
        args.ce_model = resolve_model(args.ce_model, CE_ALIASES)

    print(f"\n── learn_alpha ──")
    print(f"   dataset={args.dataset}  encoder={args.encoder_model}")
    print(f"   objective={args.objective}  top_k={args.top_k}  top_m={args.top_m}")
    print(f"   α ∈ [{args.alpha_min}, {args.alpha_max}] step={args.alpha_step}")
    if args.ce_model:
        print(f"   CE model={args.ce_model}  β ∈ [{args.beta_min}, {args.beta_max}]")

    # Build grids
    alphas = np.round(
        np.arange(args.alpha_min, args.alpha_max + args.alpha_step / 2, args.alpha_step),
        decimals=6,
    )
    betas = np.round(
        np.arange(args.beta_min, args.beta_max + args.beta_step / 2, args.beta_step),
        decimals=6,
    )

    # Load data
    print("\n── Loading data ──")
    all_examples, corpus = _load_examples(args)
    print(f"   {len(all_examples)} examples loaded.")

    val_examples, test_examples = _val_test_split(
        all_examples, args.val_fraction, args.val_seed
    )
    print(f"   val={len(val_examples)}  test={len(test_examples)}")

    # Load encoder
    print(f"\n── Loading encoder: {args.encoder_model} ──")
    retriever = DenseRetriever(
        model_name=args.encoder_model, device=device, batch_size=32
    )

    # BEIR: retrieve and attach passages
    if corpus is not None:
        print("\n── Attaching BEIR passages ──")
        val_examples = _attach_beir_passages(
            val_examples, corpus, retriever, args.top_m, args.index_dir
        )

    # Load generator if needed
    generator = None
    if args.objective in {"em", "f1"}:
        print(f"\n── Loading generator: {args.generator_model} ──")
        generator = load_generator(
            model_name=args.generator_model, device=args.device,
            max_new_tokens=128, num_beams=4,
        )

    # Resolve fine_step default here so it's available as an attribute
    if args.fine_step is None:
        args.fine_step = round(args.alpha_step / 5.0, 8)

    setting = f"{args.dataset}/{args.encoder_model}"

    # ----- Standard sweep -----
    print("\n── Sweeping standard methods (RNG-Score / Seg-Score) ──")
    if args.coarse_to_fine:
        print("   Mode: coarse-to-fine  fine_step={:.4g}  dead_zone_min={:.4g}".format(
            args.fine_step, args.dead_zone_min))
        results = _coarse_to_fine_sweep(val_examples, retriever, alphas, args,
                                        generator, setting)
    else:
        results, _, _ = _sweep_standard(val_examples, retriever, alphas, args,
                                        generator, setting)

    # ----- CE sweep -----
    if args.ce_model:
        print(f"\n── Loading cross-encoder: {args.ce_model} ──")
        from retrieval.cross_encoder import CrossEncoderReranker
        ce_reranker = CrossEncoderReranker(model_name=args.ce_model, device=args.device)

        print("\n── Sweeping CE strategies (S1-Blend / S2-V1/V2/V3) ──")
        ce_results = _sweep_ce(val_examples, retriever, ce_reranker,
                               alphas, betas, args, generator, setting)
        results.extend(ce_results)

    # ----- Report -----
    print("\n══ Optimal hyper-parameters found ══")
    print_alpha_table(results)

    # ----- Save -----
    run_params = {
        "script": "learn_alpha.py",
        "setting": setting,
        "dataset": args.dataset,
        "split": args.split,
        "max_samples": args.max_samples,
        "val_fraction": args.val_fraction,
        "val_seed": args.val_seed,
        "encoder_model": args.encoder_model,
        "ce_model": args.ce_model,
        "device": args.device,
        "objective": args.objective,
        "lambda_div": args.lambda_div,
        "top_k": args.top_k,
        "top_m": args.top_m,
        "metric": args.metric,
        "coarse_to_fine": args.coarse_to_fine,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_step": args.alpha_step,
        "fine_step": args.fine_step,
        "dead_zone_min": args.dead_zone_min,
        "beta_min": args.beta_min,
        "beta_max": args.beta_max,
        "beta_step": args.beta_step,
        "alpha_grid": [float(a) for a in alphas.tolist()],
        "beta_grid": [float(b) for b in betas.tolist()],
        "generator_model": args.generator_model,
        "index_dir": args.index_dir,
    }
    base_dir = os.path.dirname(args.output_json) or "results"
    json_filename = os.path.basename(args.output_json)
    run_dir = make_run_dir(base_dir, f"alpha_{args.dataset.replace('/', '_')}", run_params)
    output_json = os.path.join(run_dir, json_filename)
    save_alphas(results, output_json)

    print(f"\nDone. Load with:")
    print(f"  from alpha_selection import load_alphas, lookup_alpha")
    print(f"  alphas = load_alphas({output_json!r})")
    print(f"  r = lookup_alpha(alphas, 'RNG-Score', setting={setting!r})")
    print(f"  print(r.alpha_star)")


if __name__ == "__main__":
    main()
