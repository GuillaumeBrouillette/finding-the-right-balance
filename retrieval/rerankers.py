"""
Retrieval re-ranking methods.

All functions receive pre-computed embeddings and return a list of *indices*
into the candidate list, ordered by selection priority.

References
----------
- kNN        : standard nearest-neighbour, no diversity
- Dedup      : greedy near-duplicate removal, then top-k by relevance
- MMR        : Carbonell & Goldstein (1998)
- Maxmin     : greedy max-min diversity
- Greedy DPP : Chen et al. (2018), NeurIPS
- RDS        : this paper
- RADS       : this paper
- REDS       : this paper

Examples
--------
>>> import numpy as np
>>> from retrieval.rerankers import rerank_knn, rerank_rds
>>> scores = np.array([0.9, 0.4, 0.7, 0.3])
>>> rerank_knn(scores, k=2)
[0, 2]
>>> rng = np.random.default_rng(0)
>>> embs = rng.standard_normal((4, 8))
>>> q = rng.standard_normal(8)
>>> selected = rerank_rds(embs, q, k=2, alpha=0.1, metric='euclidean')
>>> len(selected) <= 2
True
"""

from __future__ import annotations

import numpy as np
from typing import List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (..., D).  Each row is treated as an independent
        vector to be normalised.  A small epsilon (1e-12) is added to
        norms to avoid division by zero.

    Returns
    -------
    x_norm : np.ndarray
        Array of the same shape as ``x`` with unit-L2 rows.
    """
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + 1e-12)


# ---------------------------------------------------------------------------
# k-Nearest Neighbours (baseline)
# ---------------------------------------------------------------------------

def rerank_knn(scores: np.ndarray, k: int) -> List[int]:
    """Return indices of the top-k documents by relevance score.

    Parameters
    ----------
    scores : np.ndarray
        (n,) float array of pre-computed relevance scores (higher is more
        relevant, e.g. cosine similarity to the query).
    k : int
        Number of documents to select.  If k > n, all n indices are returned.

    Returns
    -------
    selected : list of int
        Indices of the top-k documents, sorted in descending score order.

    Examples
    --------
    >>> import numpy as np
    >>> rerank_knn(np.array([0.1, 0.9, 0.4, 0.7]), k=2)
    [1, 3]
    """
    k = min(k, len(scores))
    return np.argsort(-scores)[:k].tolist()


# ---------------------------------------------------------------------------
# Near-duplicate removal (Dedup baseline)
# ---------------------------------------------------------------------------

def rerank_dedup(
    candidate_embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    threshold: float = 0.9,
) -> List[int]:
    """
    Greedy near-duplicate removal followed by top-k by relevance.

    Candidates are visited in descending relevance order; a candidate is
    kept iff its maximum cosine similarity to every already-kept candidate
    is below ``threshold``.  Selection stops once k candidates are kept.
    If the pool is exhausted with fewer than k kept (everything left is a
    near-duplicate), the remaining slots are filled with the skipped
    candidates in relevance order, so exactly min(k, n) indices are always
    returned.

    On a pool with no pair at or above ``threshold`` this reproduces the
    kNN ranking exactly, which makes it the natural "dedup-only" control
    between kNN and the diversity-pressure methods (MMR, Maxmin, DPP).

    Parameters
    ----------
    candidate_embeddings : np.ndarray
        (n, D) float array of candidate document embeddings.
    scores : np.ndarray
        (n,) float array of pre-computed relevance scores (higher is more
        relevant, e.g. cosine similarity to the query).
    k : int
        Number of documents to select.
    threshold : float, optional
        Cosine-similarity threshold t above which a candidate counts as a
        near-duplicate of an already-kept one.  Default 0.9.

    Returns
    -------
    selected : list of int
        Indices of the selected documents, in selection order.

    Examples
    --------
    >>> import numpy as np
    >>> embs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> scores = np.array([0.9, 0.8, 0.1])
    >>> rerank_dedup(embs, scores, k=2, threshold=0.95)  # skips the copy
    [0, 2]
    >>> rerank_dedup(embs, scores, k=3, threshold=0.95)  # backfills the copy
    [0, 2, 1]
    """
    n = len(scores)
    k = min(k, n)
    ce = _normalize(candidate_embeddings)

    kept: List[int] = []
    skipped: List[int] = []
    for idx in np.argsort(-scores):
        idx = int(idx)
        if kept and float(np.max(ce[idx] @ ce[kept].T)) >= threshold:
            skipped.append(idx)
            continue
        kept.append(idx)
        if len(kept) >= k:
            break

    if len(kept) < k:
        kept.extend(skipped[: k - len(kept)])
    return kept


# ---------------------------------------------------------------------------
# Maximal Marginal Relevance
# ---------------------------------------------------------------------------

def rerank_mmr(
    candidate_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    scores: np.ndarray,
    k: int,
    lambda_: float = 0.5,
) -> List[int]:
    """
    Maximal Marginal Relevance (Carbonell & Goldstein 1998).

    Parameters
    ----------
    candidate_embeddings : np.ndarray
        (n, D) float array of candidate document embeddings.
    query_embedding : np.ndarray
        (D,) float array representing the query embedding.
    scores : np.ndarray
        (n,) float array of pre-computed relevance scores (e.g. cosine
        similarities to the query).  Used as Sim1 in the original MMR
        formula.
    k : int
        Number of documents to select.
    lambda_ : float, optional
        Trade-off weight in [0, 1].  ``1`` gives pure relevance ranking
        (equivalent to kNN); ``0`` gives pure diversity.  Default 0.5.

    Returns
    -------
    selected : list of int
        Indices of the selected documents, in selection order.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> embs = rng.standard_normal((5, 4))
    >>> q = rng.standard_normal(4)
    >>> scores = embs @ q / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q))
    >>> selected = rerank_mmr(embs, q, scores, k=3, lambda_=0.5)
    >>> len(selected)
    3
    """
    n = len(scores)
    k = min(k, n)
    ce = _normalize(candidate_embeddings)

    selected: List[int] = []
    remaining = list(range(n))

    while len(selected) < k and remaining:
        if not selected:
            best = max(remaining, key=lambda i: scores[i])
        else:
            sel_embs = ce[selected]  # (|S|, D)
            best = max(
                remaining,
                key=lambda i: (
                    lambda_ * scores[i]
                    - (1.0 - lambda_) * float(np.max(ce[i] @ sel_embs.T))
                ),
            )
        selected.append(best)
        remaining.remove(best)

    return selected


# ---------------------------------------------------------------------------
# Maxmin diversity
# ---------------------------------------------------------------------------

def rerank_maxmin(
    candidate_embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> List[int]:
    """
    Greedy maxmin: seed with the most relevant document, then iteratively
    pick the candidate furthest (minimum cosine similarity) from the
    already-selected set.

    Parameters
    ----------
    candidate_embeddings : np.ndarray
        (n, D) float array of candidate document embeddings.
    scores : np.ndarray
        (n,) float array of pre-computed relevance scores.  The document
        with the highest score is used as the seed of the selection.
    k : int
        Number of documents to select.

    Returns
    -------
    selected : list of int
        Indices of the selected documents, in selection order.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> embs = rng.standard_normal((6, 4))
    >>> scores = np.array([0.8, 0.2, 0.6, 0.4, 0.9, 0.1])
    >>> selected = rerank_maxmin(embs, scores, k=3)
    >>> len(selected)
    3
    """
    n = len(scores)
    k = min(k, n)
    ce = _normalize(candidate_embeddings)

    seed = int(np.argmax(scores))
    selected = [seed]
    remaining = [i for i in range(n) if i != seed]

    while len(selected) < k and remaining:
        sel_embs = ce[selected]
        # min cosine similarity to any selected doc, for each remaining
        min_sims = (ce[remaining] @ sel_embs.T).min(axis=1)
        best_pos = int(np.argmin(min_sims))
        best = remaining[best_pos]
        selected.append(best)
        remaining.pop(best_pos)

    return selected


# ---------------------------------------------------------------------------
# Greedy MAP Determinantal Point Process
# ---------------------------------------------------------------------------

def rerank_greedy_dpp(
    candidate_embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> List[int]:
    """
    Greedy MAP-DPP via Gram-Schmidt orthogonalisation (Chen et al. 2018).

    Kernel: K[i,j] = score_i * cosine(e_i, e_j) * score_j.
    At each step the document with the largest residual squared-norm is
    chosen; this maximises the incremental log-determinant.

    Parameters
    ----------
    candidate_embeddings : np.ndarray
        (n, D) float array of candidate document embeddings.
    scores : np.ndarray
        (n,) float array of non-negative relevance scores used to weight
        the kernel (negative values are clipped to 0).
    k : int
        Number of documents to select.

    Returns
    -------
    selected : list of int
        Indices of the selected documents, in selection order.  May contain
        fewer than k items if the residual norms collapse to zero.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(2)
    >>> embs = rng.standard_normal((6, 4))
    >>> scores = np.array([0.8, 0.2, 0.6, 0.4, 0.9, 0.1])
    >>> selected = rerank_greedy_dpp(embs, scores, k=3)
    >>> len(selected) <= 3
    True
    """
    n = len(scores)
    k = min(k, n)
    ce = _normalize(candidate_embeddings)

    # Weighted embeddings: u_i = score_i * e_i
    # Then <u_i, u_j> = score_i * cosine(e_i, e_j) * score_j = K[i,j].
    u = ce * np.maximum(scores, 0.0)[:, np.newaxis]  # (n, D)

    selected: List[int] = []
    excluded = set()

    for _ in range(k):
        norms_sq = np.sum(u ** 2, axis=1)
        for s in excluded:
            norms_sq[s] = -np.inf
        if norms_sq.max() <= 1e-14:
            break

        best = int(np.argmax(norms_sq))
        selected.append(best)
        excluded.add(best)

        # Orthogonalise all remaining vectors against u[best]
        u_best_norm = float(norms_sq[best]) ** 0.5
        u_best_unit = u[best] / u_best_norm
        projections = u @ u_best_unit        # (n,)
        u -= np.outer(projections, u_best_unit)

    return selected


# ---------------------------------------------------------------------------
# Relative Distance Selection  (this paper)
# ---------------------------------------------------------------------------

def rerank_rds(
    candidate_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    k: int,
    alpha: float = 0.0,
    metric: str = "angular",
) -> List[int]:
    """
    Relative Distance Selection.

    Parameters
    ----------
    candidate_embeddings : (n, D) array, unit-normalised or raw.
    query_embedding      : (D,) array.
    k                    : maximum number of documents to select.
    alpha                : diversity margin (>= 0).
    metric               : one of
        - "euclidean"  : d(u,v) = ||u - v||_2
        - "angular"    : d(u,v) = arccos(<u/||u||, v/||v||>)  [Sec. 6.2]
        - "cosine"     : d(u,v) = 1 - <u/||u||, v/||v||>      [Sec. 6.3,
                          not a true metric; simplified threshold]

    Returns
    -------
    List of selected indices (may be fewer than k if alpha is large).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(3)
    >>> embs = rng.standard_normal((8, 4))
    >>> q = rng.standard_normal(4)
    >>> selected = rerank_rds(embs, q, k=3, alpha=0.1, metric='angular')
    >>> len(selected) <= 3
    True
    """
    n = len(candidate_embeddings)
    k = min(k, n)

    ce = _normalize(candidate_embeddings)
    qe = _normalize(query_embedding)

    if metric == "euclidean":
        dists_to_q = np.linalg.norm(candidate_embeddings - query_embedding, axis=1)
        order = np.argsort(dists_to_q)

        sigma: List[int] = []
        for idx in order:
            d_jq = float(dists_to_q[idx])
            admit = all(
                np.linalg.norm(candidate_embeddings[idx] - candidate_embeddings[i])
                > d_jq + alpha
                for i in sigma
            )
            if admit:
                sigma.append(int(idx))
            if len(sigma) >= k:
                break
        return sigma

    elif metric == "angular":
        # Sort by ascending angular distance to query
        sims_to_q = np.clip(ce @ qe, -1.0, 1.0)
        theta_jq = np.arccos(sims_to_q)          # in [0, π]
        order = np.argsort(theta_jq)

        # Per-candidate admission threshold (Section 6.2):
        #   sim(v_i, v_j) < cos(theta_jq + alpha)
        #                 = sim(v_j,q)*cos(alpha) - sqrt(1-sim(v_j,q)^2)*sin(alpha)
        cos_a, sin_a = float(np.cos(alpha)), float(np.sin(alpha))

        sigma: List[int] = []
        for idx in order:
            s_jq = float(sims_to_q[idx])
            threshold = s_jq * cos_a - np.sqrt(max(0.0, 1.0 - s_jq ** 2)) * sin_a
            admit = all(
                float(ce[idx] @ ce[i]) < threshold
                for i in sigma
            )
            if admit:
                sigma.append(int(idx))
            if len(sigma) >= k:
                break
        return sigma

    elif metric == "cosine":
        # Cosine distance: d_c(u,v) = 1 - <u_norm, v_norm>
        # Admission condition (Section 6.3):
        #   sim(v_j, v_i) < sim(v_j, q) - alpha   for all v_i in sigma
        sims_to_q = ce @ qe
        dists_to_q = 1.0 - sims_to_q
        order = np.argsort(dists_to_q)

        sigma: List[int] = []
        for idx in order:
            threshold = float(sims_to_q[idx]) - alpha
            admit = all(
                float(ce[idx] @ ce[i]) < threshold
                for i in sigma
            )
            if admit:
                sigma.append(int(idx))
            if len(sigma) >= k:
                break
        return sigma

    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: euclidean, angular, cosine.")


# ---------------------------------------------------------------------------
# Relative Exhaustive Distance Selection  (this paper)
# ---------------------------------------------------------------------------

def rerank_reds(
    candidate_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    k: int,
    alpha: float = 0.0,
    metric: str = "angular",
) -> List[int]:
    """
    Relative Exhaustive Distance Selection (REDS).

    Identical to RDS, but the admission condition is checked against *all*
    previously examined candidates (both admitted and rejected), not only
    those in σ.  A candidate v_j is admitted iff

        d(v_j, q) + α  <  d(v_i, v_j)   for every previously examined v_i.

    This guarantees the same full-corpus RNG adjacency as RADS, but is less
    restrictive (admits at least as many candidates) because it does not rely
    on the adaptive margin update.  The trade-off is a higher worst-case
    complexity: O(n²D) instead of O(nkD), though in practice the inner loop
    terminates early and the typical cost is close to O(nkD).

    Parameters
    ----------
    candidate_embeddings : (n, D) array, unit-normalised or raw.
    query_embedding      : (D,) array.
    k                    : maximum number of documents to select.
    alpha                : diversity margin (>= 0).
    metric               : one of
        - "euclidean"  : d(u,v) = ||u - v||_2
        - "angular"    : d(u,v) = arccos(<u/||u||, v/||v||>)
        - "cosine"     : d(u,v) = 1 - <u/||u||, v/||v||>

    Returns
    -------
    List of selected indices (may be fewer than k if alpha is large).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(4)
    >>> embs = rng.standard_normal((8, 4))
    >>> q = rng.standard_normal(4)
    >>> selected = rerank_reds(embs, q, k=3, alpha=0.0, metric='angular')
    >>> len(selected) <= 3
    True
    """
    n = len(candidate_embeddings)
    k = min(k, n)

    ce = _normalize(candidate_embeddings)
    qe = _normalize(query_embedding)

    if metric == "euclidean":
        dists_to_q = np.linalg.norm(candidate_embeddings - query_embedding, axis=1)
        order = np.argsort(dists_to_q)

        sigma: List[int] = []
        examined: List[int] = []   # all candidates processed so far (σ ∪ rejected)

        for idx in order:
            d_jq = float(dists_to_q[idx])
            admit = all(
                np.linalg.norm(candidate_embeddings[idx] - candidate_embeddings[i])
                > d_jq + alpha
                for i in examined
            )
            examined.append(int(idx))
            if admit:
                sigma.append(int(idx))
            if len(sigma) >= k:
                break
        return sigma

    elif metric == "angular":
        sims_to_q = np.clip(ce @ qe, -1.0, 1.0)
        theta_jq = np.arccos(sims_to_q)          # in [0, π]
        order = np.argsort(theta_jq)

        cos_a, sin_a = float(np.cos(alpha)), float(np.sin(alpha))

        sigma: List[int] = []
        examined: List[int] = []

        for idx in order:
            s_jq = float(sims_to_q[idx])
            threshold = s_jq * cos_a - np.sqrt(max(0.0, 1.0 - s_jq ** 2)) * sin_a
            admit = all(
                float(ce[idx] @ ce[i]) < threshold
                for i in examined
            )
            examined.append(int(idx))
            if admit:
                sigma.append(int(idx))
            if len(sigma) >= k:
                break
        return sigma

    elif metric == "cosine":
        sims_to_q = ce @ qe
        dists_to_q = 1.0 - sims_to_q
        order = np.argsort(dists_to_q)

        sigma: List[int] = []
        examined: List[int] = []

        for idx in order:
            threshold = float(sims_to_q[idx]) - alpha
            admit = all(
                float(ce[idx] @ ce[i]) < threshold
                for i in examined
            )
            examined.append(int(idx))
            if admit:
                sigma.append(int(idx))
            if len(sigma) >= k:
                break
        return sigma

    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: euclidean, angular, cosine.")


# ---------------------------------------------------------------------------
# Adaptive Relative Distance Selection  (this paper)
# ---------------------------------------------------------------------------

def rerank_rads(
    candidate_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    k: int,
    alpha: float = 0.0,
    metric: str = "angular",
) -> List[int]:
    """
    Relative Adaptive Distance Selection (RADS).

    A variant of RDS where each selected point v_i ∈ σ maintains its own
    diversity margin α_i, initialised to ``alpha``.  When a candidate v_j is
    examined, the per-point admission criterion

        d(v_j, q) + α_i  <  d(v_i, v_j)   for all v_i ∈ σ

    is evaluated.  For every v_i that *blocks* v_j (i.e. the criterion fails),
    α_i is updated:

        α_i  ←  max(α_i,  d(v_i, v_j))

    so that α_i can only grow, reflecting the maximum distance at which v_i
    has blocked a candidate.

    Parameters
    ----------
    candidate_embeddings : (n, D) array, unit-normalised or raw.
    query_embedding      : (D,) array.
    k                    : maximum number of documents to select.
    alpha                : initial diversity margin for every selected point.
    metric               : one of
        - "euclidean"  : d(u,v) = ||u - v||_2
        - "angular"    : d(u,v) = arccos(<u/||u||, v/||v||>)
        - "cosine"     : d(u,v) = 1 - <u/||u||, v/||v||>

    Returns
    -------
    List of selected indices (may be fewer than k if alpha is large).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(5)
    >>> embs = rng.standard_normal((8, 4))
    >>> q = rng.standard_normal(4)
    >>> selected = rerank_rads(embs, q, k=3, alpha=0.1, metric='angular')
    >>> len(selected) <= 3
    True
    """
    n = len(candidate_embeddings)
    k = min(k, n)

    ce = _normalize(candidate_embeddings)
    qe = _normalize(query_embedding)

    if metric == "euclidean":
        dists_to_q = np.linalg.norm(candidate_embeddings - query_embedding, axis=1)
        order = np.argsort(dists_to_q)

        sigma: List[int] = []
        per_point_alphas: List[float] = []   # α_i for each v_i ∈ σ

        for idx in order:
            d_jq = float(dists_to_q[idx])
            admit = True
            for pos, i in enumerate(sigma):
                d_ij = float(np.linalg.norm(candidate_embeddings[idx] - candidate_embeddings[i]))
                if d_jq + per_point_alphas[pos] >= d_ij:
                    per_point_alphas[pos] = max(per_point_alphas[pos], d_ij)
                    admit = False
                    # continue: update every blocking sigma member, not just the first
            if admit:
                sigma.append(int(idx))
                per_point_alphas.append(alpha)
            if len(sigma) >= k:
                break
        return sigma

    elif metric == "angular":
        sims_to_q = np.clip(ce @ qe, -1.0, 1.0)
        theta_jq = np.arccos(sims_to_q)          # in [0, π]
        order = np.argsort(theta_jq)

        sigma: List[int] = []
        per_point_alphas: List[float] = []   # α_i for each v_i ∈ σ

        for idx in order:
            s_jq = float(sims_to_q[idx])
            admit = True
            for pos, i in enumerate(sigma):
                a_i = per_point_alphas[pos]
                threshold = s_jq * np.cos(a_i) - np.sqrt(max(0.0, 1.0 - s_jq ** 2)) * np.sin(a_i)
                if float(ce[idx] @ ce[i]) >= threshold:
                    theta_ij = float(np.arccos(np.clip(ce[idx] @ ce[i], -1.0, 1.0)))
                    per_point_alphas[pos] = max(a_i, theta_ij)
                    admit = False
            if admit:
                sigma.append(int(idx))
                per_point_alphas.append(alpha)
            if len(sigma) >= k:
                break
        return sigma

    elif metric == "cosine":
        sims_to_q = ce @ qe
        dists_to_q = 1.0 - sims_to_q
        order = np.argsort(dists_to_q)

        sigma: List[int] = []
        per_point_alphas: List[float] = []   # α_i for each v_i ∈ σ

        for idx in order:
            sim_jq = float(sims_to_q[idx])
            admit = True
            for pos, i in enumerate(sigma):
                sim_ij = float(ce[idx] @ ce[i])
                threshold = sim_jq - per_point_alphas[pos]
                if sim_ij >= threshold:
                    d_ij = 1.0 - sim_ij
                    per_point_alphas[pos] = max(per_point_alphas[pos], d_ij)
                    admit = False
            if admit:
                sigma.append(int(idx))
                per_point_alphas.append(alpha)
            if len(sigma) >= k:
                break
        return sigma

    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: euclidean, angular, cosine.")


# ---------------------------------------------------------------------------
# Shared distance helper for soft rerankers
# ---------------------------------------------------------------------------

def _pairwise_dists(
    candidate_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute distances-to-query and the n×n pairwise distance matrix.

    Parameters
    ----------
    candidate_embeddings : np.ndarray
        (n, D) float array of candidate document embeddings.
    query_embedding : np.ndarray
        (D,) float array representing the query embedding.
    metric : str
        One of ``"euclidean"``, ``"angular"``, ``"cosine"``.

    Returns
    -------
    dq : np.ndarray
        (n,) distances from each candidate to the query.
    D_mat : np.ndarray
        (n, n) pairwise distance matrix, where ``D_mat[i, j] = d(v_i, v_j)``.
    """
    if metric == "euclidean":
        dq = np.linalg.norm(candidate_embeddings - query_embedding, axis=1)
        diff = candidate_embeddings[:, np.newaxis, :] - candidate_embeddings[np.newaxis, :, :]
        D_mat = np.linalg.norm(diff, axis=2)
    elif metric == "angular":
        ce = _normalize(candidate_embeddings)
        qe = _normalize(query_embedding)
        sims_q = np.clip(ce @ qe, -1.0, 1.0)
        dq = np.arccos(sims_q)
        sims_pp = np.clip(ce @ ce.T, -1.0, 1.0)
        D_mat = np.arccos(sims_pp)
    elif metric == "cosine":
        ce = _normalize(candidate_embeddings)
        qe = _normalize(query_embedding)
        dq = 1.0 - (ce @ qe)
        D_mat = 1.0 - (ce @ ce.T)
    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: euclidean, angular, cosine.")
    return dq, D_mat


# ---------------------------------------------------------------------------
# RNG-Score soft reranker
# ---------------------------------------------------------------------------

def rerank_rng_score(
    candidate_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    k: int,
    alpha: float = 0.0,
    metric: str = "cosine",
) -> List[int]:
    """RNG-Score soft reranker.

    Scores every candidate $w$ by

    .. math::

        C(w, q) = d(w, q)
                  + \\sum_{\\substack{v \\in V \\\\ d(v,q) < d(w,q)}}
                    \\max\\!\\bigl(d(w,q) - d(v,w) + \\alpha,\\; 0\\bigr)

    and returns the top-k documents with the *smallest* scores.

    Each hinge term is strictly positive iff $v$ would block $w$ from
    admission in RDS.  Summing over all closer documents yields a
    continuous measure of geometric obstruction.  For an RNG-neighbor of
    $q$ (no obstructor), all cross-document terms vanish and
    $C(w, q) = d(w, q)$.

    Unlike the hard RDS algorithm, this reranker always returns exactly
    $k$ documents.

    Parameters
    ----------
    candidate_embeddings : np.ndarray
        (n, D) float array of candidate document embeddings.
    query_embedding : np.ndarray
        (D,) float array representing the query embedding.
    k : int
        Number of documents to return.
    alpha : float, optional
        Diversity margin (same role as in RDS).  Default 0.0.
    metric : str, optional
        One of ``"euclidean"``, ``"angular"``, ``"cosine"``.  Default
        ``"cosine"``.

    Returns
    -------
    selected : list of int
        Indices of the top-k documents sorted by ascending RNG-Score.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(6)
    >>> embs = rng.standard_normal((8, 4))
    >>> q = rng.standard_normal(4)
    >>> selected = rerank_rng_score(embs, q, k=3, alpha=0.0)
    >>> len(selected)
    3
    """
    n = len(candidate_embeddings)
    k = min(k, n)
    dq, D_mat = _pairwise_dists(candidate_embeddings, query_embedding, metric)

    # mask[i, j] = True iff d(v_i, q) < d(v_j, q)  (i is a potential obstructor of j)
    mask = dq[:, np.newaxis] < dq[np.newaxis, :]          # (n, n)
    # hinge[i, j] = max(d(v_j, q) - d(v_i, v_j) + alpha, 0)
    hinge = np.maximum(dq[np.newaxis, :] - D_mat + alpha, 0.0)  # (n, n)
    # Sum hinge contributions over obstructors i for each candidate j
    scores = dq + (hinge * mask).sum(axis=0)               # (n,)

    return np.argsort(scores)[:k].tolist()   # rng_score


# ---------------------------------------------------------------------------
# RNG-Score 2 soft reranker
# ---------------------------------------------------------------------------

def rerank_rng_score2(
    candidate_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    k: int,
    alpha: float = 0.0,
    metric: str = "cosine",
) -> List[int]:
    """Segment-Obstruction Score (RNG-Score 2) soft reranker.

    Implements the SEGOBSCORE from Definition 3.5 of the paper.  Scores
    every candidate $w$ by

    .. math::

        \\operatorname{segscore}_{\\alpha}(w;q,V)
        = d(w, q)
          + \\sum_{\\substack{v \\in P(w;q,V)\\\\
                    d(v,w) < d(w,q) + \\alpha}}
            \\bigl(2\\,d(w,q) + \\alpha - d(v,w) - d(v,q)\\bigr)

    where $P(w;q,V) = \\{v \\in V : d(v,q) < d(w,q)\\}$ is the set of closer
    competitors.  Returns the top-k documents with the *smallest* scores.

    When $\\alpha = 0$ the obstruction threshold coincides with $d(w,q)$,
    giving the parameter-free variant described in Remark 3.5 of the paper.
    Positive $\\alpha$ widens the set of active obstructors; negative $\\alpha$
    narrows it (same monotonicity as RNG-Score, Proposition 4.2).

    The penalty for an obstructor $v$ lying exactly on the geodesic segment
    from $q$ to $w$ (i.e. $d(w,q) = d(v,q) + d(v,w)$) equals $d(w,q)+\\alpha$
    regardless of how far $v$ is from $q$, giving uniform weight to all such
    in-between points (see Remark 3.4 for the metric-excess interpretation).

    Parameters
    ----------
    candidate_embeddings : np.ndarray
        (n, D) float array of candidate document embeddings.
    query_embedding : np.ndarray
        (D,) float array representing the query embedding.
    k : int
        Number of documents to return.
    alpha : float, optional
        Diversity margin.  Default 0.0.
    metric : str, optional
        One of ``"euclidean"``, ``"angular"``, ``"cosine"``.  Default
        ``"cosine"``.

    Returns
    -------
    selected : list of int
        Indices of the top-k documents sorted by ascending Seg-Score.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(7)
    >>> embs = rng.standard_normal((8, 4))
    >>> q = rng.standard_normal(4)
    >>> selected = rerank_rng_score2(embs, q, k=3, alpha=0.0)
    >>> len(selected)
    3
    """
    n = len(candidate_embeddings)
    k = min(k, n)
    dq, D_mat = _pairwise_dists(candidate_embeddings, query_embedding, metric)

    # mask1[i, j] = True iff d(v_i, q) < d(v_j, q)  (i in P(v_j; q, V))
    mask1 = dq[:, np.newaxis] < dq[np.newaxis, :]          # (n, n)
    # mask2[i, j] = True iff d(v_i, v_j) < d(v_j, q) + alpha
    mask2 = D_mat < dq[np.newaxis, :] + alpha               # (n, n)
    combined = mask1 & mask2

    # penalty[i, j] = 2*d(v_j, q) + alpha - d(v_i, v_j) - d(v_i, q)
    penalty = 2.0 * dq[np.newaxis, :] + alpha - D_mat - dq[:, np.newaxis]  # (n, n)
    scores = dq + (penalty * combined).sum(axis=0)          # (n,)

    return np.argsort(scores)[:k].tolist()   # seg_score


# ---------------------------------------------------------------------------
# Cross-encoder integration (RQ4)
# ---------------------------------------------------------------------------

def _squash(x: np.ndarray, c: float = 2.0) -> np.ndarray:
    """Squashing function f_c(x) = 2/(1 + exp(-cx)) - 1, range (-1, 1).

    At x = 0 → 0; x → +∞ → 1; x → -∞ → -1.
    """
    return 2.0 / (1.0 + np.exp(-c * np.asarray(x, dtype=np.float64))) - 1.0


def rerank_ce_topk(ce_scores: np.ndarray, k: int) -> List[int]:
    """Select top-k candidates by cross-encoder relevance (no diversification).

    Parameters
    ----------
    ce_scores : np.ndarray
        (n,) float array of cross-encoder relevance scores ∈ [0, 1].
    k : int
        Number of documents to select.

    Returns
    -------
    selected : list of int

    Examples
    --------
    >>> import numpy as np
    >>> rerank_ce_topk(np.array([0.1, 0.9, 0.4, 0.7]), k=2)
    [1, 3]
    """
    k = min(k, len(ce_scores))
    return np.argsort(-ce_scores)[:k].tolist()


def rerank_vendi_greedy(
    candidate_embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    lambda_: float = 0.5,
) -> List[int]:
    """
    Greedy Vendi-score selector: a modern diversity baseline in the spirit of
    Vendi-RAG's retrieval objective (Rezaei et al. 2025), isolated to the
    selection step on a fixed pool.

    At each step the candidate maximising

        lambda_ * relevance(c)  +  (1 - lambda_) * [Vendi(S + c) - Vendi(S)]

    is appended, where Vendi(S) = exp(entropy of the eigenvalues of K/|S|)
    on the cosine-similarity kernel K of the L2-normalised selected
    embeddings (Friedman & Dieng 2023): the effective number of distinct
    documents in the selection.  lambda_ = 1 recovers kNN ranking.

    Parameters
    ----------
    candidate_embeddings : np.ndarray
        (n, D) float array of candidate document embeddings.
    scores : np.ndarray
        (n,) relevance scores (query similarities).
    k : int
        Number of documents to select.
    lambda_ : float
        Relevance/diversity trade-off in [0, 1], as in MMR.

    Returns
    -------
    selected : list of int
        Indices of the selected documents, in selection order.
    """
    n = candidate_embeddings.shape[0]
    if n == 0 or k <= 0:
        return []
    norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    embs = candidate_embeddings / np.clip(norms, 1e-12, None)

    def vendi(idx: List[int]) -> float:
        if not idx:
            return 0.0
        K = embs[idx] @ embs[idx].T
        w = np.linalg.eigvalsh(K / len(idx))
        w = np.clip(w, 0.0, None)
        w = w[w > 1e-12]
        return float(np.exp(-(w * np.log(w)).sum()))

    selected: List[int] = [int(np.argmax(scores))]
    v_cur = vendi(selected)
    while len(selected) < min(k, n):
        best_i, best_val, best_v = -1, -np.inf, v_cur
        for i in range(n):
            if i in selected:
                continue
            v_new = vendi(selected + [i])
            val = lambda_ * float(scores[i]) + (1.0 - lambda_) * (v_new - v_cur)
            if val > best_val:
                best_i, best_val, best_v = i, val, v_new
        selected.append(best_i)
        v_cur = best_v
    return selected


def rerank_mmr_ce(
    candidate_embeddings: np.ndarray,
    ce_scores: np.ndarray,
    k: int,
    lambda_: float = 0.5,
) -> List[int]:
    """MMR using cross-encoder scores as relevance, embeddings for diversity.

    Parameters
    ----------
    candidate_embeddings : np.ndarray
        (n, D) float array of candidate document embeddings.
    ce_scores : np.ndarray
        (n,) float array of cross-encoder relevance scores ∈ [0, 1].
    k : int
        Number of documents to select.
    lambda_ : float, optional
        MMR trade-off weight.  Default 0.5.

    Returns
    -------
    selected : list of int

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(8)
    >>> embs = rng.standard_normal((6, 4))
    >>> ce = np.array([0.9, 0.4, 0.7, 0.3, 0.8, 0.5])
    >>> selected = rerank_mmr_ce(embs, ce, k=3)
    >>> len(selected)
    3
    """
    n = len(ce_scores)
    k = min(k, n)
    ce_norm = _normalize(candidate_embeddings)

    selected: List[int] = []
    remaining = list(range(n))

    while len(selected) < k and remaining:
        if not selected:
            best = max(remaining, key=lambda i: ce_scores[i])
        else:
            sel_embs = ce_norm[selected]
            best = max(
                remaining,
                key=lambda i: (
                    lambda_ * float(ce_scores[i])
                    - (1.0 - lambda_) * float(np.max(ce_norm[i] @ sel_embs.T))
                ),
            )
        selected.append(best)
        remaining.remove(best)

    return selected


def rerank_ce_rng_blended(
    candidate_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    ce_scores: np.ndarray,
    k: int,
    alpha: float = 0.0,
    beta: float = 0.5,
    c: float = 2.0,
    metric: str = "cosine",
    transform: str = "sigmoidal",
) -> List[int]:
    """Strategy S1: score blending between CE relevance and RNG-Score.

    Two variants are supported (Appendix §A.2):

    V1 (sigmoidal)::

        sim_blend(q, w) = (1-β)·sim_CE(q,w) + β·(1 - f_c(score_α(w;q)))

    V2 (reciprocal)::

        sim_blend(q, w) = (1-β)·sim_CE(q,w) + β·1/(1 + score_α(w;q))

    At β=0 both recover the CE ranking; at β=1 both recover the RNG-Score
    ranking.  Top-k by *largest* blended similarity.

    Parameters
    ----------
    candidate_embeddings : np.ndarray
        (n, D) float array.
    query_embedding : np.ndarray
        (D,) float array.
    ce_scores : np.ndarray
        (n,) float array of CE relevance probabilities ∈ [0, 1].
    k : int
    alpha : float
        RNG-Score diversity margin.
    beta : float
        Blend weight ∈ [0, 1].
    c : float
        Squashing parameter for f_c (V1 only).
    metric : str
        Embedding distance metric.
    transform : str
        ``"sigmoidal"`` for V1 or ``"reciprocal"`` for V2.

    Returns
    -------
    selected : list of int

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(9)
    >>> embs = rng.standard_normal((8, 4))
    >>> q = rng.standard_normal(4)
    >>> ce = np.ones(8) * 0.5
    >>> selected = rerank_ce_rng_blended(embs, q, ce, k=3)
    >>> len(selected)
    3
    """
    n = len(candidate_embeddings)
    k = min(k, n)

    dq, D_mat = _pairwise_dists(candidate_embeddings, query_embedding, metric)
    mask = dq[:, np.newaxis] < dq[np.newaxis, :]
    hinge = np.maximum(dq[np.newaxis, :] - D_mat + alpha, 0.0)
    rng_scores = dq + (hinge * mask).sum(axis=0)     # (n,), lower = better

    if transform == "sigmoidal":
        rng_sim = 1.0 - _squash(rng_scores, c=c)     # large score → near 0
    elif transform == "reciprocal":
        rng_sim = 1.0 / (1.0 + rng_scores)           # large score → near 0
    else:
        raise ValueError(f"Unknown transform '{transform}'. Choose 'sigmoidal' or 'reciprocal'.")

    blended = (1.0 - beta) * ce_scores + beta * rng_sim
    return np.argsort(-blended)[:k].tolist()


def rerank_ce_semimetric(
    candidate_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    ce_scores: np.ndarray,
    k: int,
    alpha: float = 0.0,
    variant: int = 1,
    c: float = 2.0,
    metric: str = "cosine",
) -> List[int]:
    """Strategy S2: CE-induced semimetric applied as RNG-Score.

    Defines a semimetric on the candidate pool using cross-encoder scores for
    query-document distances and embedding distances for document-document
    distances.  The RNG-Score is then computed in this semimetric space.

    Five variants for the inter-document distance ``d_CE(v, w)``
    (Appendix §A.3)::

        V1: d_CE(v,w) = d_emb(v,w) / R
        V2: d_CE(v,w) = 2·f_c(d_emb(v,w)/R)
        V3: d_CE(v,w) = d_emb(v,w)·(2 - sim_CE(q,v) - sim_CE(q,w))
                        / (d_emb(q,v) + d_emb(q,w))
        V4: d_CE(v,w) = 2·d_emb(v,w) / (1 + d_emb(v,w))
        V5: d_CE(v,w) = d_emb(v,w)   [direct; requires d∈[0,2], e.g. cosine]

    where ``R = max_v d_emb(q, v)`` is the pool radius.  The query-document
    distance is always ``d_CE(q,w) = 1 - sim_CE(q,w)``.  Top-k by *smallest*
    CE-semimetric RNG-Score.

    Parameters
    ----------
    candidate_embeddings : np.ndarray
        (n, D) float array.
    query_embedding : np.ndarray
        (D,) float array.
    ce_scores : np.ndarray
        (n,) float array of CE relevance probabilities ∈ [0, 1].
    k : int
    alpha : float
        RNG-Score diversity margin.
    variant : int
        1, 2, 3, 4, or 5.
    c : float
        Squashing parameter for f_c (used in variant 2).
    metric : str
        Embedding distance metric for d_emb.

    Returns
    -------
    selected : list of int

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(10)
    >>> embs = rng.standard_normal((8, 4))
    >>> q = rng.standard_normal(4)
    >>> ce = np.linspace(0.2, 0.9, 8)
    >>> selected = rerank_ce_semimetric(embs, q, ce, k=3, variant=1)
    >>> len(selected)
    3
    """
    n = len(candidate_embeddings)
    k = min(k, n)

    dq_ce = (1.0 - ce_scores).astype(np.float64)          # (n,)

    dq_emb, D_emb = _pairwise_dists(candidate_embeddings, query_embedding, metric)
    dq_emb = dq_emb.astype(np.float64)
    D_emb = D_emb.astype(np.float64)
    R = float(dq_emb.max()) if dq_emb.max() > 1e-12 else 1.0

    if variant == 1:
        D_ce = D_emb / R
    elif variant == 2:
        D_ce = 2.0 * _squash(D_emb / R, c=c)
    elif variant == 3:
        ce_i = ce_scores[:, np.newaxis].astype(np.float64)
        ce_j = ce_scores[np.newaxis, :].astype(np.float64)
        dq_i = dq_emb[:, np.newaxis]
        dq_j = dq_emb[np.newaxis, :]
        denom = np.where(dq_i + dq_j < 1e-12, 1e-12, dq_i + dq_j)
        D_ce = D_emb * (2.0 - ce_i - ce_j) / denom
    elif variant == 4:
        D_ce = 2.0 * D_emb / (1.0 + D_emb)
    elif variant == 5:
        D_ce = D_emb
    else:
        raise ValueError(f"Unknown variant {variant}. Choose 1, 2, 3, 4, or 5.")

    ce_f = ce_scores.astype(np.float64)
    mask = ce_f[:, np.newaxis] > ce_f[np.newaxis, :]       # i obstructs j
    hinge = np.maximum(dq_ce[np.newaxis, :] - D_ce + alpha, 0.0)
    scores = dq_ce + (hinge * mask).sum(axis=0)

    return np.argsort(scores)[:k].tolist()
