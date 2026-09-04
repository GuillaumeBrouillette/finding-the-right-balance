#!/usr/bin/env python3
"""Export ranked passages for each configured reranking method."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from data.loaders import (
    load_2wikimultihopqa,
    load_hotpotqa,
    load_hotpotqa_fullwiki,
    load_musique,
    load_nq_dpr,
    load_squad,
    load_trivia_dpr,
)
from evaluation.metrics import relevance_labels
from retrieval.retriever import DenseRetriever
from retrieval.rerankers import (
    rerank_greedy_dpp,
    rerank_knn,
    rerank_mmr,
    rerank_rng_score,
)
from ftrb.run_utils import (
    ENCODER_ALIASES,
    default_encoder,
    int_or_all,
    normalize_device,
    resolve_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def _build_methods(
    top_k: int,
    metric: str,
    rng_alphas: List[float],
    mmr_lambdas: List[float],
) -> List[Tuple[str, Callable]]:
    """Return (name, fn) pairs. fn(embs, q_emb, scores, k) -> List[int]."""
    methods: List[Tuple[str, Callable]] = []

    methods.append((
        "kNN",
        lambda embs, q, sc, _k=top_k: rerank_knn(sc, _k),
    ))

    for lam in mmr_lambdas:
        lam_ = lam
        methods.append((
            f"MMR(λ={lam_})",
            lambda embs, q, sc, _k=top_k, _l=lam_: rerank_mmr(embs, q, sc, _k, _l),
        ))

    methods.append((
        "DPP",
        lambda embs, q, sc, _k=top_k: rerank_greedy_dpp(embs, sc, _k),
    ))

    for alpha in rng_alphas:
        a_ = alpha
        methods.append((
            f"RNG(α={a_:.4g})",
            lambda embs, q, sc, _k=top_k, _a=a_, _m=metric: rerank_rng_score(
                embs, q, _k, alpha=_a, metric=_m
            ),
        ))

    return methods


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    device = normalize_device(args.device)
    encoder_model = resolve_model(
        args.encoder_model or default_encoder(device), ENCODER_ALIASES
    )

    # Load dataset
    print(f"\n── Loading dataset: {args.dataset} / {args.split} ──")
    split = args.split
    max_s = args.max_samples
    if args.dataset == "hotpotqa":
        examples = load_hotpotqa(split=split, max_samples=max_s)
    elif args.dataset == "hotpotqa_fullwiki":
        examples = load_hotpotqa_fullwiki(split=split, max_samples=max_s)
    elif args.dataset == "squad":
        examples = load_squad(split=split, max_samples=max_s)
    elif args.dataset == "2wikimultihopqa":
        examples = load_2wikimultihopqa(split=split, max_samples=max_s)
    elif args.dataset == "musique":
        examples = load_musique(split=split, max_samples=max_s)
    elif args.dataset == "nq":
        examples = load_nq_dpr(split=split, max_samples=max_s)
    elif args.dataset == "trivia":
        examples = load_trivia_dpr(split=split, max_samples=max_s)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset!r}")

    examples = [ex for ex in examples if ex["passages"]]
    print(f"   {len(examples)} examples with passages.")

    # Load encoder
    print(f"\n── Loading encoder: {encoder_model} ──")
    encoder = DenseRetriever(
        model_name=encoder_model, device=device, batch_size=args.batch_size
    )

    # Build method list
    methods = _build_methods(
        args.top_k, args.metric, args.rng_alphas, args.mmr_lambdas
    )
    method_names = [name for name, _ in methods]
    print(f"\n── Methods: {method_names} ──")

    # Pre-compute embeddings
    print("\n── Encoding queries and passages ──")
    all_questions = [ex["question"] for ex in examples]
    all_pools = [ex["passages"] for ex in examples]

    from retrieval.precompute import encode_queries_and_passages
    q_embs, p_embs_flat, offsets = encode_queries_and_passages(
        encoder, all_questions, all_pools
    )

    # Per-query retrieval
    records: List[Dict] = []

    for qi, ex in enumerate(tqdm(examples, desc="Retrieving")):
        s, e = int(offsets[qi]), int(offsets[qi + 1])
        q_emb = q_embs[qi]
        p_embs = p_embs_flat[s:e]
        passages = all_pools[qi]
        gold_titles = ex.get("gold_titles") or []

        # Cosine similarity between query and each passage
        nq = np.linalg.norm(q_emb)
        np_ = np.linalg.norm(p_embs, axis=1)
        emb_scores = (p_embs @ q_emb) / (np_ * nq + 1e-12)

        relevance = relevance_labels(passages, gold_titles=gold_titles, answers=ex["answers"])

        # Run all methods and collect selected index sets
        selections: Dict[str, List[int]] = {}
        for method_name, fn in methods:
            sel = fn(p_embs, q_emb, emb_scores, args.top_k)
            selections[method_name] = sel

        # Number of distinct passage sets across all methods for this query
        unique_sets = len({frozenset(sel) for sel in selections.values()})

        for method_name, sel in selections.items():
            for rank, idx in enumerate(sel, start=1):
                p = passages[idx]
                records.append({
                    "query_id":       ex["id"],
                    "question":       ex["question"],
                    "num_unique_sets": unique_sets,
                    "method":         method_name,
                    "rank":           rank,
                    "title":          p["title"],
                    "text":           _truncate(p["text"], args.text_max_chars),
                    "is_gold":        bool(relevance[idx] > 0),
                    "emb_score":      round(float(emb_scores[idx]), 4),
                })

    # Build DataFrame and sort: most disagreement first, then stable order
    df = pd.DataFrame(records)
    method_order = {name: i for i, name in enumerate(method_names)}
    df["_method_order"] = df["method"].map(method_order)
    df = df.sort_values(
        ["num_unique_sets", "query_id", "_method_order", "rank"],
        ascending=[False, True, True, True],
    ).drop(columns=["_method_order"])
    df = df.reset_index(drop=True)

    # Save
    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.join(PROJECT_ROOT, "..", "retrieved_passages")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"retrieve_passages_{args.dataset}_{args.split}_k{args.top_k}.csv"
        )
    df.to_csv(out_path, index=False)
    print(f"\n   Saved {len(df)} rows → {out_path}")
    print(f"   Queries with num_unique_sets > 1: "
          f"{df.groupby('query_id')['num_unique_sets'].first().gt(1).sum()} "
          f"/ {df['query_id'].nunique()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrieve top-k passages per query with multiple methods and save to CSV."
    )
    p.add_argument(
        "--dataset",
        choices=["hotpotqa", "hotpotqa_fullwiki", "squad", "2wikimultihopqa",
                 "musique", "nq", "trivia"],
        default="hotpotqa",
    )
    p.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    p.add_argument(
        "--max_samples", type=int_or_all, default=100,
        help="Number of queries to process (default 100); pass 'all' for the full split.",
    )
    p.add_argument("--top_k", type=int, default=5, help="Number of passages to retrieve per query.")
    p.add_argument(
        "--rng_alphas", type=float, nargs="+", default=[0.0],
        help="Alpha values for RNG-Score (default: 0.0).",
    )
    p.add_argument(
        "--mmr_lambdas", type=float, nargs="+", default=[0.3, 0.5, 0.7],
        help="Lambda values for MMR (default: 0.3 0.5 0.7).",
    )
    p.add_argument(
        "--metric", choices=["cosine", "euclidean", "angular"], default="cosine",
    )
    p.add_argument(
        "--encoder_model", default=None,
        help=f"Encoder alias or HuggingFace ID. Aliases: {', '.join(ENCODER_ALIASES)}. "
             "Default: bge-m3 on CUDA, minilm on CPU.",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--text_max_chars", type=int, default=300,
        help="Truncate passage text to this many characters in the CSV (0 = no truncation).",
    )
    p.add_argument("--output", default=None, help="Output CSV path (default: auto-named).")
    return p.parse_args()


if __name__ == "__main__":
    main()
