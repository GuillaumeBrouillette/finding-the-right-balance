"""Regression test: NDCG@k and MRR must stay in [0, 1] when the retrieved
list contains multiple copies of the same gold title (redundancy injection,
evaluate_redundancy.py).

Background
----------
inject_duplicates() appends near-duplicate passages that inherit the title of
their source passage.  A reranker that fills the top-k with copies of one
gold passage then presents a title list like ["Paris", "Paris", "Paris",
"France", "Other"].  ndcg_at_k() used to add a DCG term for every occurrence
while the ideal DCG was computed from the deduplicated gold set, so NDCG
could exceed 1 (e.g. 1.59 for kNN at rho=0.25 in the 2026-06-10 fullwiki
run, produced before evaluate_selection() masked repeated titles).  The
metric itself must credit each distinct gold title at most once so that the
bound holds for every caller, masked or not.

Run with pytest, or directly:  python tests/test_ndcg_duplicates.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evaluation.metrics import mrr, ndcg_at_k

# evaluate_redundancy pulls in the retrieval stack (torch, datasets, …); the
# end-to-end test is skipped on machines that only have numpy installed.
try:
    from evaluate_redundancy import evaluate_selection, inject_duplicates
    _HAVE_PIPELINE = True
except ImportError:
    _HAVE_PIPELINE = False


GOLD = ["Paris", "France"]


def test_ndcg_bounded_under_duplicate_titles():
    # Three copies of one gold title in the top-k (unmasked, as the
    # pre-2026-06-11 evaluate_selection passed them).
    retrieved = ["Paris", "Paris", "Paris", "France", "Other"]
    score = ndcg_at_k(retrieved, GOLD, k=5)
    assert 0.0 <= score <= 1.0, f"NDCG@5 = {score} out of [0, 1]"
    # Both golds first, copies behind them: a perfect ranking, not >1.
    assert abs(ndcg_at_k(["Paris", "France", "Paris", "Paris"], GOLD, k=5) - 1.0) < 1e-12


def test_ndcg_duplicate_copies_earn_no_extra_credit():
    # Extra copies of an already-seen gold title must not change the score.
    base = ndcg_at_k(["Paris", "Other", "France"], GOLD, k=5)
    with_dup = ndcg_at_k(["Paris", "Paris", "Other", "France", "Paris"], GOLD, k=5)
    # The duplicate at rank 2 pushes "France" one rank down, so the score may
    # drop, but a copy itself earns nothing: replacing the copy with junk
    # gives the identical score.
    with_junk = ndcg_at_k(["Paris", "Junk", "Other", "France", "Junk"], GOLD, k=5)
    assert abs(with_dup - with_junk) < 1e-12
    assert with_dup <= base + 1e-12


def test_ndcg_matches_masked_convention():
    # The internal first-occurrence dedup must agree with the explicit
    # masking convention used by evaluate_selection since 33dc187.
    retrieved = ["Paris", "Paris", "France", "Paris", "Other"]
    masked = ["Paris", "__dup_1__", "France", "__dup_3__", "Other"]
    assert abs(ndcg_at_k(retrieved, GOLD, k=5) - ndcg_at_k(masked, GOLD, k=5)) < 1e-12


def test_mrr_bounded_and_unaffected_by_duplicates():
    # MRR only uses the first match, so duplicates can never push it past 1,
    # and masking repeats does not change it.
    retrieved = ["Other", "Paris", "Paris", "Paris"]
    masked = ["Other", "Paris", "__dup_2__", "__dup_3__"]
    assert mrr(retrieved, GOLD) == mrr(masked, GOLD) == 0.5
    assert mrr(["Paris"] * 10, GOLD) == 1.0


def test_evaluate_selection_bounded_on_injected_pool():
    if not _HAVE_PIPELINE:
        print("      (skipped: retrieval stack not installed)")
        return
    # End-to-end: gold-targeted injection at rho=1 makes the pool mostly
    # copies of the two gold passages; selecting the five most query-similar
    # passages (kNN-style) then picks several copies.  Every rank metric
    # must stay in [0, 1].
    rng = np.random.default_rng(0)
    passages = [
        {"title": "Paris", "text": "Paris is the capital of France. It is large."},
        {"title": "France", "text": "France is a country in Europe. It is old."},
        {"title": "Other", "text": "Something unrelated entirely. Filler text."},
        {"title": "Misc", "text": "More filler content here. Nothing relevant."},
    ]
    pool = inject_duplicates(passages, GOLD, rate=1.0, rng=rng,
                             noise="exact", target="gold")
    assert sum(p.get("is_duplicate", False) for p in pool) == 4

    # Deterministic embeddings: copies share their source's direction, so a
    # similarity ranking puts gold copies on top.
    base = {"Paris": [1.0, 0.0, 0.0], "France": [0.9, 0.1, 0.0],
            "Other": [0.0, 1.0, 0.0], "Misc": [0.0, 0.0, 1.0]}
    p_embs = np.array([base[p["title"]] for p in pool])
    p_embs /= np.linalg.norm(p_embs, axis=1, keepdims=True)
    q_emb = np.array([1.0, 0.05, 0.0])
    q_emb /= np.linalg.norm(q_emb)

    selected = list(np.argsort(-(p_embs @ q_emb))[:5])
    titles = [pool[i]["title"] for i in selected]
    assert titles.count("Paris") >= 2, f"setup failed to select copies: {titles}"

    res = evaluate_selection(selected, pool, p_embs, GOLD, k=5)
    for name in ("NDCG@k", "MRR", "Recall@k", "alpha-NDCG@k",
                 "S-Recall@k", "ERR-IA@k"):
        assert 0.0 <= res[name] <= 1.0 + 1e-12, f"{name} = {res[name]}"


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
