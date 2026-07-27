"""
retrieval/precompute.py
=======================
Batch pre-computation helpers for evaluation pipelines.

Encoding queries and passages one at a time creates many tiny GPU dispatches
with Python overhead between each.  The functions here collect all texts and
dispatch a single large encode / score call *before* the per-example loop,
keeping the GPU fed continuously.

Typical usage
-------------
::

    from retrieval.precompute import encode_queries_and_passages, score_cross_encoder

    # One-time sweep before the evaluation loop
    q_embs, p_embs_flat, p_offsets = encode_queries_and_passages(
        encoder, questions, passages_pools
    )
    ce_scores_flat = score_cross_encoder(
        cross_encoder, questions, passages_pools
    )

    # Inside the per-example loop — pure numpy, no GPU calls
    s, e = int(p_offsets[i]), int(p_offsets[i + 1])
    q_emb      = q_embs[i]
    p_embs     = p_embs_flat[s:e]
    ce_scores  = ce_scores_flat[s:e]
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Internal text formatting
# ---------------------------------------------------------------------------

def _format_passage(p: Dict) -> str:
    """Format a passage dict as 'title: text' for dense encoding."""
    return ((p.get("title") or "") + ": " + (p.get("text") or "")).strip()


def _format_passage_ce(p: Dict) -> str:
    """Format a passage for cross-encoder scoring.

    Falls back to plain text when the 'title: text' combination would be
    empty (e.g. passages that have no title field).
    """
    combined = ((p.get("title") or "") + ": " + (p.get("text") or "")).strip()
    return combined or (p.get("text") or "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_queries_and_passages(
    encoder,
    questions: List[str],
    passages_pools: List[List[Dict]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode all queries and passages in one large batch sweep.

    Parameters
    ----------
    encoder : DenseRetriever
        Encoder used to produce L2-normalised embeddings.
    questions : list of str
        One question per example (length N).
    passages_pools : list of list of dict
        Candidate passage dicts for each example.

    Returns
    -------
    q_embs : np.ndarray
        (N, D) float32 query embeddings.
    p_embs_flat : np.ndarray
        (total_passages, D) float32 passage embeddings, laid out
        contiguously.  Passages for example *i* occupy
        ``p_embs_flat[p_offsets[i]:p_offsets[i+1]]``.
    p_offsets : np.ndarray
        (N+1,) int64 offsets for slicing ``p_embs_flat`` per example.
    """
    p_offsets = np.zeros(len(questions) + 1, dtype=np.int64)
    for i, pool in enumerate(passages_pools):
        p_offsets[i + 1] = p_offsets[i] + len(pool)

    print(f"   Encoding {len(questions)} queries …")
    q_embs = encoder.encode(questions, normalize=True, show_progress=True)

    flat_texts = [_format_passage(p) for pool in passages_pools for p in pool]
    print(f"   Encoding {len(flat_texts)} passages …")
    p_embs_flat = encoder.encode(flat_texts, normalize=True, show_progress=True)

    return q_embs, p_embs_flat, p_offsets


def score_cross_encoder(
    cross_encoder,
    questions: List[str],
    passages_pools: List[List[Dict]],
) -> np.ndarray:
    """Batch-score all (query, passage) pairs with a cross-encoder.

    Uses the cross-encoder text convention (``"title: text"``, falling back
    to text only) and calls ``cross_encoder.score_pairs`` in one large batch.
    The result is a flat array in the same order as ``p_embs_flat`` returned
    by :func:`encode_queries_and_passages` when called with the same
    ``passages_pools``.

    Parameters
    ----------
    cross_encoder : CrossEncoderReranker
    questions : list of str
    passages_pools : list of list of dict

    Returns
    -------
    ce_scores_flat : np.ndarray
        (total_passages,) float32 relevance scores in [0, 1].
    """
    pairs = [
        (q, _format_passage_ce(p))
        for q, pool in zip(questions, passages_pools)
        for p in pool
    ]
    print(f"   CE-scoring {len(pairs)} pairs …")
    return cross_encoder.score_pairs(pairs, show_progress=True)
