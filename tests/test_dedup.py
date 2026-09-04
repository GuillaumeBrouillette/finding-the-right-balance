"""Behavioral tests for the greedy near-duplicate baseline."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from retrieval.rerankers import rerank_dedup, rerank_knn


def test_dedup_equals_knn_on_clean_pool():
    # Random pool: with high probability no cosine pair reaches 0.95, so
    # Dedup must return the exact kNN ranking.
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((30, 16))
    q = rng.standard_normal(16)
    ce = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    scores = ce @ (q / np.linalg.norm(q))
    sims = ce @ ce.T
    np.fill_diagonal(sims, -1.0)
    assert sims.max() < 0.95, "test setup: pool accidentally contains near-dups"
    for k in (1, 5, 10, 30):
        assert rerank_dedup(embs, scores, k, threshold=0.95) == rerank_knn(scores, k)


def test_dedup_skips_near_duplicates():
    # Two exact copies of the top passage; a distinct third one.
    embs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    scores = np.array([0.9, 0.8, 0.1])
    assert rerank_dedup(embs, scores, k=2, threshold=0.95) == [0, 2]
    # kNN would take the copy instead.
    assert rerank_knn(scores, k=2) == [0, 1]


def test_dedup_backfills_when_pool_exhausted():
    # Only two distinct directions but k=4: the two skipped copies must
    # backfill in relevance order so exactly k indices come back.
    embs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    assert rerank_dedup(embs, scores, k=4, threshold=0.95) == [0, 3, 1, 2]


def test_dedup_threshold_one_is_knn_even_with_copies():
    # cosine == 1.0 is not >= threshold only when strictly below; with exact
    # copies and t=1.0 the copies are still blocked (sim == 1.0 >= 1.0), so
    # use t slightly above 1 to disable dedup entirely.
    embs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    scores = np.array([0.9, 0.8, 0.1])
    assert rerank_dedup(embs, scores, k=3, threshold=1.01) == rerank_knn(scores, k=3)


if __name__ == "__main__":
    failures = 0
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    sys.exit(1 if failures else 0)
