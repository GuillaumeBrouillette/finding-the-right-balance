#!/usr/bin/env python3
"""
evaluate_cross_encoder.py  –  RQ4: diversification on top of a cross-encoder
=============================================================================

Tests whether the RNG-Score can add diversity value after a cross-encoder
reranker in a multi-stage pipeline, using two integration strategies:

  S1 – score blending     (Appendix §A.2):
       sim_blend(q,w) = (1-β)·sim_CE(q,w) + β·(1 - f_c(score_α(w;q)))

  S2 – CE-induced semimetric  (Appendix §A.3), three d_CE(v,w) variants:
       V1: d_CE(v,w) = d_emb(v,w) / R
       V2: d_CE(v,w) = 2·f_c(d_emb(v,w) / R)
       V3: d_CE(v,w) = d_emb(v,w)·(2 - sim_CE(q,v) - sim_CE(q,w))
                       / (d_emb(q,v) + d_emb(q,w))

No training is performed; both strategies are purely inference-time.

Usage
-----
    python evaluate_cross_encoder.py
    python evaluate_cross_encoder.py --dataset hotpotqa --max_samples 100
    python evaluate_cross_encoder.py --no_generation --top_m 50 --top_k 5

Pipeline
--------
  dense retrieval (top-m)  →  cross-encoder rescoring  →  diversification  →  generation (optional)
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

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

from alpha_selection import (
    AlphaResult,
    accumulate_alpha,
    accumulate_blend,
    average_accumulator,
    build_coarse_to_fine_alpha_grid,
    find_optimal_alpha,
    find_optimal_blend,
    get_objective_value,
    load_alphas,
    lookup_alpha,
    save_alphas,
)
from data.loaders import (
    load_2wikimultihopqa,
    load_hotpotqa,
    load_hotpotqa_fullwiki,
    load_musique,
    load_nq_dpr,
    load_squad,
    load_trivia_dpr,
)
from evaluation.metrics import (
    alpha_ndcg_at_k,
    avg_pairwise_distance,
    exact_match,
    f1_score_single,
    mrr_from_relevance,
    ndcg_from_relevance,
    recall_at_k,
    relevance_labels,
    subtopic_coverage_sets,
    subtopic_recall_at_k,
    vendi_score,
)
from generation.generator import load_generator
from retrieval.cross_encoder import CrossEncoderReranker
from retrieval.precompute import encode_queries_and_passages, score_cross_encoder
from retrieval.rerankers import (
    rerank_ce_rng_blended,
    rerank_ce_semimetric,
    rerank_ce_topk,
    rerank_greedy_dpp,
    rerank_mmr_ce,
)
from retrieval.retriever import DenseRetriever
from run_utils import (
    CE_ALIASES,
    ENCODER_ALIASES,
    GENERATOR_ALIASES,
    attach_run_params,
    default_ce_model,
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

_ALPHA_GRID = [-0.5, -0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3, 0.5]
_BETA_GRID = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
_MMR_LAMBDAS = [0.3, 0.5, 0.7]

# Across-seed aggregation: the no-diversification cross-encoder top-k is the
# baseline for significance (Table 4 compares integration strategies to it).
_CE_BASELINE = "CE-topk"
_CE_RET_METRICS = ["Recall@k", "NDCG@k", "MRR", "alpha-NDCG@k", "S-Recall@k",
                   "APD", "Vendi", "Latency (ms)"]
_CE_GEN_METRICS = ["EM", "F1"]
_CE_SIG_RET = ["alpha-NDCG@k", "S-Recall@k"]
_CE_TUNED_PREFIXES = ("S1-", "S2-")


def _ce_method_key(method_name: str) -> str:
    """Seed-stable grouping key: tuned S1/S2 strategies drop their tuned
    (alpha*,beta*) suffix; fixed strategies (CE-topk, CE-MMR, CE-DPP) keep it."""
    if any(method_name.startswith(p) for p in _CE_TUNED_PREFIXES):
        return method_name.split("(", 1)[0]
    return method_name





# ---------------------------------------------------------------------------
# Build method list
# ---------------------------------------------------------------------------

def _build_methods(
    k: int,
    metric: str,
    alpha_grid: List[float],
    beta_grid: List[float],
    mmr_lambdas: List[float],
) -> List[Tuple[str, Callable[..., List[int]]]]:
    """Return (name, callable) pairs.

    Each callable has signature:
        fn(embs, q_emb, ce_scores, k) -> List[int]
    """
    methods: List[Tuple[str, Callable[..., List[int]]]] = []

    # --- CE baseline (no diversification) ---
    methods.append(("CE-topk", lambda embs, q, ce, _k=k: rerank_ce_topk(ce, _k)))

    # --- CE + MMR ---
    for lam in mmr_lambdas:
        lam_ = lam
        methods.append((
            f"CE-MMR(λ={lam_})",
            lambda embs, q, ce, _k=k, _l=lam_: rerank_mmr_ce(embs, ce, _k, _l),
        ))

    # --- CE + Greedy-DPP (uses embedding similarity for diversity) ---
    methods.append((
        "CE-DPP",
        lambda embs, q, ce, _k=k: rerank_greedy_dpp(embs, ce, _k),
    ))

    # --- S1: score blending (two transform variants) ---
    for s1_variant, transform in [(1, "sigmoidal"), (2, "reciprocal")]:
        sv_ = s1_variant
        tr_ = transform
        for alpha in alpha_grid:
            for beta in beta_grid:
                a_, b_ = alpha, beta
                methods.append((
                    f"S1-V{sv_}-Blend(α={a_:.4g},β={b_:.4g})",
                    lambda embs, q, ce, _k=k, _a=a_, _b=b_, _m=metric, _t=tr_: rerank_ce_rng_blended(
                        embs, q, ce, _k, alpha=_a, beta=_b, metric=_m, transform=_t
                    ),
                ))

    # --- S2: CE-induced semimetric, five variants ---
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
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(cfg: Dict) -> Tuple[List[Dict], List[Dict], List[Dict], List[AlphaResult]]:
    """Run RQ4 evaluation and return table rows."""

    dataset_name = cfg["dataset"]
    split = cfg.get("split", "validation")
    max_samples = cfg.get("max_samples")
    top_m = cfg.get("top_m", 100)
    top_k = cfg.get("top_k", 5)
    metric = cfg.get("metric", "cosine")
    run_gen = cfg.get("run_generation", True)
    val_frac = float(cfg.get("val_frac", 0.2))
    _raw_obj = cfg.get("objective", "apd")
    objectives: List[str] = [_raw_obj] if isinstance(_raw_obj, str) else list(_raw_obj)
    if any(obj in {"em", "f1"} for obj in objectives):
        run_gen = True

    alpha_grid = cfg.get("alpha_grid", _ALPHA_GRID)
    beta_grid = cfg.get("beta_grid", _BETA_GRID)
    mmr_lambdas = cfg.get("mmr_lambdas", _MMR_LAMBDAS)

    # 1. Load dataset
    print(f"\n── Loading dataset: {dataset_name} / {split} ──")
    if dataset_name == "hotpotqa":
        examples = load_hotpotqa(split=split, max_samples=max_samples)
    elif dataset_name == "hotpotqa_fullwiki":
        examples = load_hotpotqa_fullwiki(split=split, max_samples=max_samples)
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
        raise ValueError(f"Unknown dataset: {dataset_name!r}.")
    print(f"   {len(examples)} examples.")

    # 2. Load dense encoder
    print(f"\n── Loading encoder: {cfg['encoder_model']} ──")
    device = normalize_device(cfg.get("device", "cpu"))
    batch_size = cfg.get("batch_size", 64)
    encoder = DenseRetriever(model_name=cfg["encoder_model"], device=device, batch_size=batch_size)

    # 3. Load cross-encoder
    print(f"\n── Loading cross-encoder: {cfg['ce_model']} ──")
    cross_encoder = CrossEncoderReranker(model_name=cfg["ce_model"], device=device, batch_size=batch_size)

    # 4. (Optional) generator
    generator = None
    if run_gen:
        print(f"\n── Loading generator: {cfg['generator_model']} ──")
        generator = load_generator(
            model_name=cfg["generator_model"],
            device=device,
            max_new_tokens=cfg.get("max_new_tokens", 64),
            num_beams=cfg.get("num_beams", 4),
        )

    # 5. Build methods
    methods = _build_methods(top_k, metric, alpha_grid, beta_grid, mmr_lambdas)

    # 6. Val/test split for hyper-parameter selection (seed-controlled so the
    # across-seed CIs capture which queries tune alpha*/beta*).
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(examples))
    rng.shuffle(idx_all)
    n_val = max(1, int(len(examples) * val_frac))
    val_set = set(idx_all[:n_val].tolist())

    # Accumulators
    ret_acc: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        "val":  {name: defaultdict(list) for name, _ in methods},
        "test": {name: defaultdict(list) for name, _ in methods},
    }
    gen_acc: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        "val":  {name: defaultdict(list) for name, _ in methods},
        "test": {name: defaultdict(list) for name, _ in methods},
    }
    per_query_ret: List[Dict] = []
    per_query_gen: List[Dict] = []

    setting = f"{dataset_name}/{cfg['encoder_model']}/{cfg['ce_model']}"
    pre_alphas: List[AlphaResult] = cfg.get("_pre_alphas", [])

    # Alpha-selection accumulators (val split only) — one set per objective
    _SCORE_TYPES = ["S1-V1-Blend", "S1-V2-Blend", "S2-V1", "S2-V2", "S2-V3", "S2-V4", "S2-V5"]
    _alpha_accs: Dict[str, Dict[str, Dict]] = {
        obj: {st: {} for st in _SCORE_TYPES}
        for obj in objectives
    }

    # 7. Pre-compute all embeddings and CE scores in one GPU sweep.
    for ex in examples:
        if ex["passages"] is None:
            raise ValueError(
                "Cross-encoder evaluation requires datasets with attached passages. "
                "Use hotpotqa or squad."
            )

    all_questions: List[str] = [ex["question"] for ex in examples]
    all_passages_pools: List[List[Dict]] = [ex["passages"][:top_m] for ex in examples]

    print("\n── Pre-computing embeddings and CE scores ──")
    q_embs_all, p_embs_flat, _p_offsets = encode_queries_and_passages(
        encoder, all_questions, all_passages_pools
    )
    ce_scores_flat = score_cross_encoder(
        cross_encoder, all_questions, all_passages_pools
    )

    # 8. Per-example loop (pure CPU: slice pre-computed arrays + numpy diversification)
    for qi, ex in enumerate(tqdm(examples, desc="Evaluating")):
        split_name = "val" if qi in val_set else "test"
        question = all_questions[qi]
        answers = ex["answers"]
        passages_pool = all_passages_pools[qi]
        gold_titles = ex.get("gold_titles") or []

        _s, _e = int(_p_offsets[qi]), int(_p_offsets[qi + 1])
        q_emb = q_embs_all[qi]
        p_embs = p_embs_flat[_s:_e]
        ce_scores = ce_scores_flat[_s:_e]

        relevance = relevance_labels(passages_pool, gold_titles=gold_titles, answers=answers)
        subtopic_sets = subtopic_coverage_sets(passages_pool, gold_titles=gold_titles)

        # Phase A – reranking + retrieval metrics; collect unique selections.
        _results: Dict[str, tuple] = {}          # method → (sel, lat_ms, metrics…)
        _unique_sels: Dict[tuple, List[Dict]] = {}  # sel_key → passage list (dedup)
        for method_name, rerank_fn in methods:
            t0 = time.perf_counter()
            sel = rerank_fn(p_embs, q_emb, ce_scores, top_k)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            if not sel:
                continue
            sel_embs = p_embs[sel]
            rec = recall_at_k(sel, relevance, k=top_k)
            ndcg = ndcg_from_relevance(sel, relevance, k=top_k)
            rr = mrr_from_relevance(sel, relevance)
            apd = avg_pairwise_distance(sel_embs)
            vs = vendi_score(sel_embs)
            an = alpha_ndcg_at_k(sel, subtopic_sets, k=top_k, alpha_r=0.5)
            sr = subtopic_recall_at_k(sel, subtopic_sets, k=top_k)
            _results[method_name] = (sel, lat_ms, rec, ndcg, rr, apd, vs, an, sr)
            sel_key = tuple(sel)
            if sel_key not in _unique_sels:
                _unique_sels[sel_key] = [passages_pool[i] for i in sel]

        # Phase B – batch-generate for all unique passage selections.
        # 176 methods often produce far fewer unique selections; padding batches
        # them into a single forward pass instead of 176 sequential ones.
        _gen_cache: Dict[tuple, Optional[Tuple[float, float]]] = {}
        if run_gen and generator is not None and _unique_sels:
            _keys = list(_unique_sels.keys())
            _preds = generator.generate_batch(
                [question] * len(_keys),
                [_unique_sels[k] for k in _keys],
            )
            for _key, _pred in zip(_keys, _preds):
                _em = max(exact_match(_pred, a) for a in answers) if answers else 0.0
                _f1 = max(f1_score_single(_pred, a) for a in answers) if answers else 0.0
                _gen_cache[_key] = (_em, _f1)

        # Phase C – accumulate metrics and alpha-selection values.
        for method_name, (sel, lat_ms, rec, ndcg, rr, apd, vs, an, sr) in _results.items():
            ret_acc[split_name][method_name]["recall"].append(rec)
            ret_acc[split_name][method_name]["ndcg"].append(ndcg)
            ret_acc[split_name][method_name]["mrr"].append(rr)
            ret_acc[split_name][method_name]["alpha_ndcg"].append(an)
            ret_acc[split_name][method_name]["s_recall"].append(sr)
            ret_acc[split_name][method_name]["apd"].append(apd)
            ret_acc[split_name][method_name]["vendi"].append(vs)
            ret_acc[split_name][method_name]["latency_ms"].append(lat_ms)

            per_query_ret.append({
                "seed": seed,
                "Split": split_name,
                "Method": method_name,
                "Recall@k": f"{rec:.6f}",
                "NDCG@k": f"{ndcg:.6f}",
                "MRR": f"{rr:.6f}",
                "alpha-NDCG@k": f"{an:.6f}",
                "S-Recall@k": f"{sr:.6f}",
                "APD": f"{apd:.6f}",
                "Vendi": f"{vs:.6f}",
                "Latency (ms)": f"{lat_ms:.6f}",
            })

            em_val = f1_val = None
            gen_pair = _gen_cache.get(tuple(sel))
            if gen_pair is not None:
                em_val, f1_val = gen_pair
                gen_acc[split_name][method_name]["em"].append(em_val)
                gen_acc[split_name][method_name]["f1"].append(f1_val)
                per_query_gen.append({
                    "seed": seed,
                    "Split": split_name,
                    "Method": method_name,
                    "EM": f"{em_val:.6f}",
                    "F1": f"{f1_val:.6f}",
                })

            if split_name == "val":
                _obj_metrics = {
                    "recall": rec, "ndcg": ndcg, "alpha_ndcg": an,
                    "s_recall": sr,
                    "vendi": vs, "recall_vendi": 0.5 * rec + 0.5 * vs,
                    "apd": apd,
                }
                if em_val is not None:
                    _obj_metrics["em"] = em_val
                    _obj_metrics["f1"] = f1_val

                if method_name.startswith("S1-V1-Blend("):
                    inner = method_name[len("S1-V1-Blend("):-1]
                    _a = float(inner.split("α=")[1].split(",")[0])
                    _b = float(inner.split("β=")[1])
                    for obj in objectives:
                        accumulate_blend(_alpha_accs[obj]["S1-V1-Blend"], _a, _b,
                                         get_objective_value(_obj_metrics, obj))
                elif method_name.startswith("S1-V2-Blend("):
                    inner = method_name[len("S1-V2-Blend("):-1]
                    _a = float(inner.split("α=")[1].split(",")[0])
                    _b = float(inner.split("β=")[1])
                    for obj in objectives:
                        accumulate_blend(_alpha_accs[obj]["S1-V2-Blend"], _a, _b,
                                         get_objective_value(_obj_metrics, obj))
                elif method_name.startswith("S2-V1("):
                    _a = float(method_name[len("S2-V1(α="):-1])
                    for obj in objectives:
                        accumulate_alpha(_alpha_accs[obj]["S2-V1"], _a,
                                         get_objective_value(_obj_metrics, obj))
                elif method_name.startswith("S2-V2("):
                    _a = float(method_name[len("S2-V2(α="):-1])
                    for obj in objectives:
                        accumulate_alpha(_alpha_accs[obj]["S2-V2"], _a,
                                         get_objective_value(_obj_metrics, obj))
                elif method_name.startswith("S2-V3("):
                    _a = float(method_name[len("S2-V3(α="):-1])
                    for obj in objectives:
                        accumulate_alpha(_alpha_accs[obj]["S2-V3"], _a,
                                         get_objective_value(_obj_metrics, obj))
                elif method_name.startswith("S2-V4("):
                    _a = float(method_name[len("S2-V4(α="):-1])
                    for obj in objectives:
                        accumulate_alpha(_alpha_accs[obj]["S2-V4"], _a,
                                         get_objective_value(_obj_metrics, obj))
                elif method_name.startswith("S2-V5("):
                    _a = float(method_name[len("S2-V5(α="):-1])
                    for obj in objectives:
                        accumulate_alpha(_alpha_accs[obj]["S2-V5"], _a,
                                         get_objective_value(_obj_metrics, obj))

    # 8. Select best hyper-parameters via shared alpha_selection module
    def _find_for_obj(score_type: str, acc: Dict, obj: str) -> Optional[AlphaResult]:
        gen_for_obj = cfg.get("generator_model") if obj in {"em", "f1"} else None
        pre = lookup_alpha(pre_alphas, score_type, setting=setting,
                           objective=obj,
                           generator_model=gen_for_obj) if pre_alphas else None
        if pre is not None:
            return pre
        avg = average_accumulator(acc)
        if not avg:
            return None
        if "Blend" in score_type:
            return find_optimal_blend(avg, alpha_grid, beta_grid,
                                      score_type, obj, setting,
                                      generator_model=gen_for_obj)
        return find_optimal_alpha(avg, alpha_grid, score_type, obj, setting,
                                  generator_model=gen_for_obj)

    alpha_results_by_obj: Dict[str, Dict[str, Optional[AlphaResult]]] = {
        obj: {st: _find_for_obj(st, _alpha_accs[obj][st], obj) for st in _SCORE_TYPES}
        for obj in objectives
    }
    alpha_results: List[AlphaResult] = [
        r
        for obj_results in alpha_results_by_obj.values()
        for r in obj_results.values()
        if r is not None
    ]

    # 9. Build test summary rows
    summary_rows: List[Dict] = []

    def _add_test_row(method_name: str, tag: Optional[str] = None):
        m = ret_acc["test"].get(method_name, {})
        if not m.get("recall"):
            return
        row: Dict = {"Method": tag or method_name,
                     "MethodKey": _ce_method_key(method_name),
                     "RawMethod": method_name, "seed": seed}
        for col, k_acc in [
            ("Recall@k", "recall"),
            ("NDCG@k", "ndcg"),
            ("MRR", "mrr"),
            ("alpha-NDCG@k", "alpha_ndcg"),
            ("S-Recall@k", "s_recall"),
            ("APD", "apd"),
            ("Vendi", "vendi"),
            ("Latency (ms)", "latency_ms"),
        ]:
            vals = m.get(k_acc, [])
            row[col] = f"{np.mean(vals):.4f}" if vals else ""
        if run_gen:
            gen_m = gen_acc["test"].get(method_name, {})
            row["EM"] = f"{np.mean(gen_m['em']):.4f}" if gen_m.get("em") else ""
            row["F1"] = f"{np.mean(gen_m['f1']):.4f}" if gen_m.get("f1") else ""
        summary_rows.append(row)

    _add_test_row("CE-topk")
    for lam in mmr_lambdas:
        _add_test_row(f"CE-MMR(λ={lam})")
    _add_test_row("CE-DPP")

    multi_obj = len(objectives) > 1
    for obj, obj_results in alpha_results_by_obj.items():
        obj_suffix = f" [{obj}]" if multi_obj else ""
        for st in ["S1-V1-Blend", "S1-V2-Blend"]:
            res = obj_results[st]
            if res is not None:
                a, b = res.alpha_star, res.beta_star
                _add_test_row(
                    f"{st}(α={a:.4g},β={b:.4g})",
                    f"{st}(α*={a:.4g},β*={b:.4g}){obj_suffix}",
                )
        for st in ["S2-V1", "S2-V2", "S2-V3", "S2-V4", "S2-V5"]:
            res = obj_results[st]
            if res is not None:
                _add_test_row(
                    f"{st}(α={res.alpha_star:.4g})",
                    f"{st}(α*={res.alpha_star:.4g}){obj_suffix}",
                )

    print(f"\n══ Cross-encoder pipeline – {dataset_name} ══")
    print(tabulate(summary_rows, headers="keys", tablefmt="github"))
    for r in alpha_results:
        beta_str = f", β*={r.beta_star:.4g}" if r.beta_star is not None else ""
        print(f"   [{r.score_type}] α*={r.alpha_star:.4g}{beta_str}  "
              f"(val {r.objective}={r.val_score:.4f})")

    return summary_rows, per_query_ret, per_query_gen, alpha_results


# ---------------------------------------------------------------------------
# Across-seed aggregation
# ---------------------------------------------------------------------------

def _ce_test_values(rows: List[Dict], seed: int, raw_method: str,
                    metric: str) -> List[float]:
    """Test-split per-query values of *metric* for one (seed, method)."""
    return [as_float(r[metric]) for r in rows
            if r.get("seed") == seed and r.get("Split") == "test"
            and r.get("Method") == raw_method]


def _aggregate_ce(per_seed_summary: List[Dict], per_query_ret: List[Dict],
                  per_query_gen: List[Dict], run_gen: bool) -> List[Dict]:
    """One row per MethodKey with across-seed mean +/- 95% CI on each metric
    and across-seed median Wilcoxon p-values vs CE-topk (retrieval objectives
    plus EM/F1 when generation is on)."""
    metric_cols = list(_CE_RET_METRICS) + (list(_CE_GEN_METRICS) if run_gen else [])
    agg = aggregate_seed_rows(
        per_seed_summary, key_cols=["MethodKey"], metric_cols=metric_cols,
        seed_col="seed", keep_cols=["Method"])
    by_key: Dict[str, List[Dict]] = {}
    for r in per_seed_summary:
        by_key.setdefault(r["MethodKey"], []).append(r)
    sig_specs = [(m, per_query_ret) for m in _CE_SIG_RET]
    if run_gen:
        sig_specs += [(m, per_query_gen) for m in _CE_GEN_METRICS]
    for row in agg:
        grp = by_key.get(row["MethodKey"], [])
        raws = list(dict.fromkeys(g["RawMethod"] for g in grp))
        row["alpha*(per seed)"] = ";".join(raws) if len(raws) != 1 else raws[0]
        if row["MethodKey"] == _CE_BASELINE:
            continue
        for metric, rows in sig_specs:
            ps: List[float] = []
            for g in grp:
                a = [v for v in _ce_test_values(rows, g["seed"], g["RawMethod"], metric)
                     if v is not None]
                b = [v for v in _ce_test_values(rows, g["seed"], _CE_BASELINE, metric)
                     if v is not None]
                if a and b and len(a) == len(b):
                    p = paired_wilcoxon_p(a, b)
                    if p is not None:
                        ps.append(p)
            if ps:
                row[f"p({metric} vs {_CE_BASELINE})"] = round(float(np.median(ps)), 6)
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ4: cross-encoder + RNG-Score evaluation")
    p.add_argument("--dataset", choices=["hotpotqa", "hotpotqa_fullwiki", "squad",
                                          "2wikimultihopqa", "musique", "nq", "trivia"],
                   default="hotpotqa")
    p.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    p.add_argument("--max_samples", type=int_or_all, default=100,
                   help="Cap on examples per run (default 100 for quick pre-tests); pass 'all' for no cap.")
    p.add_argument("--encoder_model", default=None,
                   help="Encoder alias or HuggingFace ID. "
                        f"Aliases: {', '.join(ENCODER_ALIASES)}. "
                        "Default: bge-m3 on CUDA, minilm on CPU.")
    p.add_argument("--ce_model", default=None,
                   help="Cross-encoder alias or HuggingFace ID. "
                        f"Aliases: {', '.join(CE_ALIASES)}. "
                        "Default: bge-reranker-v2-m3 on CUDA, minilm-ce on CPU.")
    p.add_argument("--top_m", type=int, default=100)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--metric", choices=["cosine", "euclidean", "angular"], default="cosine")
    p.add_argument("--no_generation", action="store_true",
                   help="Skip Flan-T5 generation (retrieval metrics only).")
    p.add_argument("--generator_model", default=None,
                   help="Generator alias or HuggingFace ID. "
                        f"Aliases: {', '.join(GENERATOR_ALIASES)}. "
                        "Default: flan-t5-base on CUDA, flan-t5-small on CPU.")
    p.add_argument("--max_new_tokens", type=int, default=64,
                   help="Maximum number of tokens generated per answer (default 64).")
    p.add_argument("--num_beams", type=int, default=4,
                   help="Beam width for generation (default 4).")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--alpha_json", default=None,
                   help="Path to alpha_results.json (from learn_alpha.py). When provided, "
                        "uses pre-found α* and β* for S1/S2 strategies.")
    p.add_argument("--save_alpha_json", default=None,
                   help="Path to save val-optimal α* results (e.g. results/alpha_results_ce.json).")
    p.add_argument("--objective", nargs="+", default=["alpha_ndcg"],
                   choices=["recall", "ndcg", "alpha_ndcg", "s_recall", "vendi",
                            "recall_vendi", "apd", "em", "f1"],
                   help="Objective(s) used to select val-optimal α* and β*. "
                        "Default alpha_ndcg (intent-aware coverage); pass "
                        "s_recall for parity with the QA tuning, or em/f1 to tune "
                        "on answer quality (requires generation). apd/vendi tune "
                        "for raw spread (intrinsic credit). Multiple values "
                        "produce one α* per objective.")
    p.add_argument("--coarse_to_fine", action="store_true",
                   help="Use an expanded fine-resolution alpha sweep for S1/S2 strategies.")
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

    # Resolve device first — model defaults depend on it.
    device = normalize_device(args.device)
    encoder_model  = resolve_model(args.encoder_model  or default_encoder(device),  ENCODER_ALIASES)
    ce_model       = resolve_model(args.ce_model       or default_ce_model(device),  CE_ALIASES)
    generator_model = resolve_model(args.generator_model or default_generator(device), GENERATOR_ALIASES)

    # Load pre-computed alphas if provided
    pre_alphas: List[AlphaResult] = []
    if args.alpha_json:
        pre_alphas = load_alphas(args.alpha_json)
        print(f"   Loaded {len(pre_alphas)} alpha result(s) from {args.alpha_json}")

    cfg = {
        "dataset": args.dataset,
        "split": args.split,
        "max_samples": args.max_samples,
        "encoder_model": encoder_model,
        "ce_model": ce_model,
        "device": device,
        "top_m": args.top_m,
        "top_k": args.top_k,
        "metric": args.metric,
        "run_generation": not args.no_generation,
        "generator_model": generator_model,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
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
        "beta_grid": _BETA_GRID,
        "mmr_lambdas": _MMR_LAMBDAS,
        "objective": args.objective,
        "batch_size": args.batch_size,
        "_pre_alphas": pre_alphas,
    }

    if args.coarse_to_fine:
        ag = cfg["alpha_grid"]
        print(
            "   [coarse_to_fine] using expanded alpha grid "
            f"[{ag[0]:.4g}, {ag[-1]:.4g}] with {len(ag)} values"
        )

    seeds = resolve_seeds(args)

    t0 = time.time()
    all_summary: List[Dict] = []
    per_query_ret: List[Dict] = []
    per_query_gen: List[Dict] = []
    alpha_results: List[AlphaResult] = []
    for seed in seeds:
        if len(seeds) > 1:
            print(f"\n========== seed {seed} ==========")
        s_summary, s_ret, s_gen, s_alpha = run_evaluation({**cfg, "seed": seed})
        all_summary.extend(s_summary)
        per_query_ret.extend(s_ret)
        per_query_gen.extend(s_gen)
        alpha_results.extend(s_alpha)
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")

    run_params = {
        "script": "evaluate_cross_encoder.py",
        "dataset": args.dataset,
        "split": args.split,
        "max_samples": args.max_samples,
        "encoder_model": encoder_model,
        "ce_model": ce_model,
        "device": device,
        "top_m": args.top_m,
        "top_k": args.top_k,
        "metric": args.metric,
        "run_generation": not args.no_generation,
        "generator_model": generator_model,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "alpha_json": args.alpha_json,
        "objective": args.objective,
        "coarse_to_fine": args.coarse_to_fine,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_step": args.alpha_step,
        "fine_step": args.fine_step,
        "dead_zone_min": args.dead_zone_min,
        "alpha_grid": cfg.get("alpha_grid"),
        "beta_grid": cfg.get("beta_grid"),
        "mmr_lambdas": cfg.get("mmr_lambdas"),
        "seeds": seeds,
    }

    tag = args.dataset
    run_dir = make_run_dir(args.output_dir, f"ce_{tag}", run_params)

    # Persist the raw data FIRST — the summary is re-derivable from it, so an
    # error in the (cheap) aggregation can never destroy a long run's results.
    if args.save_per_query:
        save_csv(per_query_ret, os.path.join(run_dir, f"results_ce_{tag}_retrieval_per_query.csv"))
        if per_query_gen:
            save_csv(per_query_gen, os.path.join(run_dir, f"results_ce_{tag}_generation_per_query.csv"))

    if len(seeds) > 1:
        save_csv(attach_run_params(all_summary, run_params),
                 os.path.join(run_dir, f"results_ce_{tag}_per_seed_summary.csv"))
        # Headline table: across-seed mean +/- 95% CI with Wilcoxon vs CE-topk.
        summary_out = _aggregate_ce(all_summary, per_query_ret, per_query_gen,
                                    cfg["run_generation"])
        save_csv(attach_run_params(summary_out, run_params),
                 os.path.join(run_dir, f"results_ce_{tag}_summary.csv"))
        print(f"   {len(seeds)} seeds; summary carries 95% across-seed CIs "
              f"and median Wilcoxon p vs {_CE_BASELINE}.")
    else:
        # Single run: preserve the legacy column layout exactly.
        legacy = [{k: v for k, v in r.items()
                   if k not in ("MethodKey", "RawMethod", "seed")}
                  for r in all_summary]
        save_csv(attach_run_params(legacy, run_params),
                 os.path.join(run_dir, f"results_ce_{tag}_summary.csv"))

    # Save alpha* results to JSON
    save_path = args.save_alpha_json or os.path.join(run_dir, f"alpha_results_ce_{tag}.json")
    if alpha_results:
        save_alphas(alpha_results, save_path)


if __name__ == "__main__":
    main()
