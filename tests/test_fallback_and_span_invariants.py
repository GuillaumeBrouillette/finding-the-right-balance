"""Invariants the paper states about fallbacks and redundancy construction.

* RNG-Score reproduces the k-NN ranking exactly at a provably sufficient
  negative margin (gamma <= -2 under cosine distance, Proposition A.4) and
  at a sufficiently large positive margin (Proposition A.5).
* MMR reproduces the k-NN ranking exactly at lambda = 1.
* The "light" injection perturbation is a pure sentence permutation, so
  every answer-bearing sentence survives verbatim in each copy.
* Sliding-window chunks are verbatim contiguous spans that keep their
  source identity, and consecutive windows overlap enough that a short
  span inside the covered text is never split across all windows.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from experiments.evaluate_redundancy import (
    _chunk_words,
    _perturb_text,
    _split_sentences,
    chunk_pool,
)
from retrieval.rerankers import rerank_knn, rerank_mmr, rerank_rng_score


def _pool(seed: int, n: int = 60, dim: int = 32, near_duplicates: int = 12):
    """Random unit-norm pool with clusters of near-duplicates, tie-free."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n - near_duplicates, dim))
    copies = base[:near_duplicates] + 0.02 * rng.standard_normal((near_duplicates, dim))
    embs = np.vstack([base, copies])
    query = base[0] + 0.5 * rng.standard_normal(dim)
    unit = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    scores = unit @ (query / np.linalg.norm(query))
    assert len(np.unique(scores)) == len(scores)  # general position
    return embs, query, scores


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("margin", [-2.0, -5.0])
def test_rng_score_equals_knn_at_sufficient_negative_margin(seed, margin):
    embs, query, scores = _pool(seed)
    full = len(scores)
    assert rerank_rng_score(embs, query, k=full, alpha=margin) == rerank_knn(scores, full)


@pytest.mark.parametrize("seed", range(5))
def test_rng_score_diversifies_at_interior_margin_but_not_at_fallback(seed):
    """The equality above is not vacuous: an interior margin reorders the pool."""
    embs, query, scores = _pool(seed)
    full = len(scores)
    assert rerank_rng_score(embs, query, k=full, alpha=0.0) != rerank_knn(scores, full)


@pytest.mark.parametrize("seed", range(5))
def test_rng_score_equals_knn_at_large_positive_margin(seed):
    embs, query, scores = _pool(seed)
    full = len(scores)
    assert rerank_rng_score(embs, query, k=full, alpha=1e4) == rerank_knn(scores, full)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k", [5, 10, 60])
def test_mmr_equals_knn_at_lambda_one(seed, k):
    embs, query, scores = _pool(seed)
    assert rerank_mmr(embs, query, scores, k=k, lambda_=1.0) == rerank_knn(scores, k)


@pytest.mark.parametrize("seed", range(5))
def test_mmr_diversifies_below_lambda_one(seed):
    embs, query, scores = _pool(seed)
    assert rerank_mmr(embs, query, scores, k=10, lambda_=0.5) != rerank_knn(scores, 10)


PASSAGE = (
    "The 1925 Birthday Honours were appointments by King George V. "
    "They were announced on 3 June 1925. "
    "The recipients are displayed here as they were styled before their new honour. "
    "Several colonial governors were included!"
)
ANSWER_SENTENCE = "They were announced on 3 June 1925."


@pytest.mark.parametrize("seed", range(20))
def test_light_perturbation_preserves_every_sentence_verbatim(seed):
    copy = _perturb_text(PASSAGE, np.random.default_rng(seed), "light")
    assert sorted(_split_sentences(copy)) == sorted(_split_sentences(PASSAGE))
    assert ANSWER_SENTENCE in copy
    assert len(copy) == len(PASSAGE)  # nothing deleted, nothing added


def test_single_sentence_passage_is_copied_verbatim():
    text = "Only one sentence here."
    assert _perturb_text(text, np.random.default_rng(0), "light") == text


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5, 0.75])
def test_chunks_are_verbatim_contiguous_spans_with_source_identity(overlap):
    window = 60
    stride = max(1, round(window * (1 - overlap)))
    words = [f"w{i}" for i in range(257)]
    text = " ".join(words)
    chunks = _chunk_words(text, window, stride)
    for j, chunk in enumerate(chunks):
        piece = chunk.split()
        assert piece == words[j * stride:j * stride + len(piece)]
    pooled = chunk_pool([{"title": "Source", "text": text}], window, stride)
    assert {p["title"] for p in pooled} == {"Source"}
    assert [p["text"] for p in pooled] == chunks


@pytest.mark.parametrize("overlap", [0.25, 0.5, 0.75])
def test_short_span_inside_covered_text_is_never_split_by_every_window(overlap):
    window = 60
    stride = max(1, round(window * (1 - overlap)))
    words = [f"w{i}" for i in range(257)]
    chunks = [c.split() for c in _chunk_words(" ".join(words), window, stride)]
    covered = (len(chunks) - 1) * stride + len(chunks[-1])
    span = window - stride + 1  # longest span guaranteed to fit in one window
    for start in range(0, covered - span + 1):
        target = words[start:start + span]
        assert any(
            chunk[i:i + span] == target
            for chunk in chunks
            for i in range(len(chunk) - span + 1)
        )
