"""Retrieval, diversity, intent-aware, and answer-quality metrics."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set

import numpy as np


# ---------------------------------------------------------------------------
# Text normalisation (SQuAD official script)
# ---------------------------------------------------------------------------

def _normalise(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

def exact_match(prediction: str, ground_truth: str) -> float:
    """Return 1.0 if the normalised strings are identical, else 0.0.

    Normalisation removes articles (a, an, the), punctuation, and extra
    whitespace, and lower-cases the text (SQuAD / HotpotQA convention).

    Parameters
    ----------
    prediction : str
        Model-generated answer string.
    ground_truth : str
        Reference answer string.

    Returns
    -------
    score : float
        1.0 if the normalised prediction equals the normalised ground truth,
        0.0 otherwise.

    Examples
    --------
    >>> exact_match("The Eiffel Tower", "eiffel tower")
    1.0
    >>> exact_match("Paris", "London")
    0.0
    """
    return float(_normalise(prediction) == _normalise(ground_truth))


def f1_score_single(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth.

    Both strings are normalised (lower-case, articles and punctuation
    removed) before tokenisation.  This follows the SQuAD evaluation
    script convention.

    Parameters
    ----------
    prediction : str
        Model-generated answer string.
    ground_truth : str
        Reference answer string.

    Returns
    -------
    f1 : float
        Token-level F1 score in [0, 1].  Returns 1.0 when both strings
        normalise to the empty string, and 0.0 when exactly one of them
        does.

    Examples
    --------
    >>> round(f1_score_single("Eiffel Tower in Paris", "Eiffel Tower"), 4)
    0.6667
    >>> f1_score_single("yes", "no")
    0.0
    """
    pred_toks = _normalise(prediction).split()
    gold_toks = _normalise(ground_truth).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------

def gold_recall(
    retrieved_titles: List[str],
    gold_titles: List[str],
) -> float:
    """Return the fraction of gold paragraph titles found in the retrieved set.

    Title comparison is case-insensitive and strips leading/trailing
    whitespace.  Returns 1.0 if ``gold_titles`` is empty (vacuously true).

    Parameters
    ----------
    retrieved_titles : list of str
        Titles of the passages returned by the retrieval / re-ranking step.
    gold_titles : list of str
        Titles of the gold supporting passages for the current question.

    Returns
    -------
    recall : float
        Fraction of gold titles covered by ``retrieved_titles``, in [0, 1].

    Examples
    --------
    >>> gold_recall(["Paris", "France", "History"], ["Paris", "France"])
    1.0
    >>> gold_recall(["Paris"], ["Paris", "France"])
    0.5
    >>> gold_recall([], [])
    1.0
    """
    if not gold_titles:
        return 1.0
    gold_set = {t.lower().strip() for t in gold_titles}
    ret_set = {t.lower().strip() for t in retrieved_titles}
    return len(gold_set & ret_set) / len(gold_set)


def ndcg_at_k(
    retrieved_titles: List[str],
    gold_titles: List[str],
    k: int,
) -> float:
    """Compute Normalised Discounted Cumulative Gain at rank k.

    Binary relevance: a retrieved document is relevant iff its title matches
    a gold title (case-insensitive, stripped).  The ideal ranking places all
    gold documents in the top positions.

    Each distinct gold title earns gain at most once, at its first
    occurrence; later copies of an already-credited title contribute nothing.
    The ideal DCG is computed over the deduplicated gold set, so without this
    first-occurrence rule a retrieved list containing several copies of one
    gold passage (e.g. under redundancy injection, where duplicates inherit
    their source title) would push NDCG above 1.

    Parameters
    ----------
    retrieved_titles : list of str
        Ordered list of titles returned by the retrieval / re-ranking step.
        Only the first ``k`` elements are considered.
    gold_titles : list of str
        Titles of the gold supporting passages.
    k : int
        Rank cutoff.

    Returns
    -------
    ndcg : float
        NDCG@k in [0, 1].  Returns 1.0 if ``gold_titles`` is empty.

    Examples
    --------
    >>> ndcg_at_k(["Paris", "France", "History"], ["Paris", "France"], k=3)
    1.0
    >>> round(ndcg_at_k(["History", "France", "Paris"], ["Paris", "France"], k=3), 4)
    0.6934
    >>> ndcg_at_k(["Paris", "France", "Paris"], ["Paris", "France"], k=3)
    1.0
    """
    if not gold_titles:
        return 1.0
    gold_set = {t.lower().strip() for t in gold_titles}
    retrieved_k = retrieved_titles[:k]
    uncredited = set(gold_set)
    dcg = 0.0
    for i, t in enumerate(retrieved_k):
        key = t.lower().strip()
        if key in uncredited:
            dcg += 1.0 / np.log2(i + 2)
            uncredited.discard(key)
    ideal_k = min(len(gold_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))
    return float(dcg / idcg) if idcg > 0 else 0.0


def mrr(
    retrieved_titles: List[str],
    gold_titles: List[str],
) -> float:
    """Compute Mean Reciprocal Rank (single-query reciprocal rank).

    Returns the reciprocal of the rank of the first retrieved document whose
    title matches a gold title.  When averaged over a set of queries the
    result is the MRR metric.

    Parameters
    ----------
    retrieved_titles : list of str
        Ordered list of titles returned by the retrieval / re-ranking step.
    gold_titles : list of str
        Titles of the gold supporting passages.

    Returns
    -------
    rr : float
        Reciprocal rank in (0, 1].  Returns 1.0 if ``gold_titles`` is empty,
        and 0.0 if no gold title appears in ``retrieved_titles``.

    Examples
    --------
    >>> mrr(["Paris", "France", "History"], ["France"])
    0.5
    >>> mrr(["Paris", "History"], ["France"])
    0.0
    """
    if not gold_titles:
        return 1.0
    gold_set = {t.lower().strip() for t in gold_titles}
    for i, t in enumerate(retrieved_titles):
        if t.lower().strip() in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def hallucination_rate(
    prediction: str,
    context_passages: List[str],
) -> float:
    """Estimate hallucination rate as the fraction of prediction tokens absent
    from the concatenated context (n-gram faithfulness proxy).

    Both the prediction and the context are normalised (lower-case, articles
    and punctuation removed) before tokenisation.  A prediction token is
    considered *faithful* if it appears at least once in any context token
    set.  Hallucination rate = 1 - faithfulness.

    This is a lightweight proxy; it does not use an NLI model.  It is most
    useful as a relative indicator when comparing rerankers on the same
    dataset.

    Parameters
    ----------
    prediction : str
        Model-generated answer string.
    context_passages : list of str
        Texts of the retrieved context passages used to generate the answer.

    Returns
    -------
    rate : float
        Fraction of unique normalised prediction tokens not found in any
        context passage, in [0, 1].  Returns 0.0 if the prediction normalises
        to the empty string.

    Examples
    --------
    >>> hallucination_rate("Paris is the capital", ["Paris is the capital of France"])
    0.0
    >>> round(hallucination_rate("Jupiter is the capital", ["Paris is the capital of France"]), 4)
    0.3333
    """
    pred_toks = set(_normalise(prediction).split())
    if not pred_toks:
        return 0.0
    context_toks: set = set()
    for passage in context_passages:
        context_toks.update(_normalise(passage).split())
    unsupported = pred_toks - context_toks
    return len(unsupported) / len(pred_toks)


# ---------------------------------------------------------------------------
# Dataset-aware retrieval relevance helpers
# ---------------------------------------------------------------------------

def passage_contains_answer(passage_text: str, answers: Sequence[str]) -> bool:
    """Return True when the passage text contains any normalised answer string.

    This lightweight heuristic is useful for open-domain QA datasets that do
    not ship gold passage identifiers.  It mirrors the string-normalisation
    used by the EM/F1 metrics and therefore provides a consistent answer-support
    signal for retrieval evaluation.
    """
    norm_passage = _normalise(passage_text)
    if not norm_passage:
        return False
    return any(
        (norm_answer := _normalise(answer)) and norm_answer in norm_passage
        for answer in answers
    )


def relevance_labels(
    passages: Sequence[dict],
    gold_titles: Sequence[str] | None = None,
    answers: Sequence[str] | None = None,
) -> List[int]:
    """Build binary relevance labels for a candidate pool.

    Title supervision is preferred when gold passage identifiers are available.
    Otherwise, the function falls back to answer-support labels obtained by
    string matching against the passage text.
    """
    if gold_titles:
        gold_set = {t.lower().strip() for t in gold_titles}
        return [
            int((p.get("title") or "").lower().strip() in gold_set)
            for p in passages
        ]
    if answers:
        return [
            int(passage_contains_answer(p.get("text", ""), answers))
            for p in passages
        ]
    return [0 for _ in passages]


def recall_at_k(
    selected_indices: Sequence[int],
    relevance: Sequence[int],
    k: int | None = None,
) -> float:
    """Compute recall over binary relevance labels."""
    total_relevant = int(np.sum(relevance))
    if total_relevant == 0:
        return 0.0
    cutoff = len(selected_indices) if k is None else min(k, len(selected_indices))
    retrieved_relevant = sum(int(relevance[i] > 0) for i in selected_indices[:cutoff])
    return retrieved_relevant / total_relevant


def ndcg_from_relevance(
    selected_indices: Sequence[int],
    relevance: Sequence[int],
    k: int,
) -> float:
    """Compute NDCG@k from binary relevance labels."""
    gains = [int(relevance[i] > 0) for i in selected_indices[:k]]
    dcg = sum(g / np.log2(rank + 2) for rank, g in enumerate(gains))
    ideal_k = min(int(np.sum(relevance)), k)
    if ideal_k == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_k))
    return dcg / idcg


def mrr_from_relevance(
    selected_indices: Sequence[int],
    relevance: Sequence[int],
) -> float:
    """Compute reciprocal rank from binary relevance labels."""
    for rank, idx in enumerate(selected_indices):
        if relevance[idx] > 0:
            return 1.0 / (rank + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

def avg_pairwise_distance(embeddings: np.ndarray) -> float:
    """Compute the average pairwise cosine distance among selected embeddings.

    Cosine distance = 1 − cosine_similarity ∈ [0, 2].  Higher values
    indicate greater diversity.

    Parameters
    ----------
    embeddings : np.ndarray
        (k, D) float array of document embeddings.  Rows are L2-normalised
        internally, so raw or pre-normalised embeddings are both accepted.

    Returns
    -------
    avg_dist : float
        Mean cosine distance over all C(k, 2) unique pairs.  Returns 0.0
        when fewer than 2 documents are provided.

    Examples
    --------
    >>> import numpy as np
    >>> embs = np.eye(3)   # three orthogonal unit vectors
    >>> round(avg_pairwise_distance(embs), 4)
    1.0
    """
    n = len(embeddings)
    if n < 2:
        return 0.0
    embs = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    sims = embs @ embs.T
    upper = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - sims[upper]))


def vendi_score(embeddings: np.ndarray) -> float:
    """Compute the Vendi Score — an effective count of distinct embeddings.

    VS = exp(−Σ λ_i log λ_i)  where {λ_i} are eigenvalues of K/n,
    K[i,j] = cosine_similarity(e_i, e_j).

    Range: [1, k].  VS = k  ↔  all embeddings orthogonal (maximally diverse).
                    VS = 1  ↔  all embeddings identical.

    Reference: Friedman & Dieng (2023) "The Vendi Score: A Diversity
    Evaluation Metric for Machine Learning", TMLR.

    Parameters
    ----------
    embeddings : np.ndarray
        (k, D) float array of document embeddings.  Rows are L2-normalised
        internally before building the kernel matrix.

    Returns
    -------
    score : float
        Vendi Score in [1, k].  Returns 0.0 for empty input and 1.0 for a
        single embedding.

    Examples
    --------
    >>> import numpy as np
    >>> embs = np.eye(3)   # three orthogonal unit vectors (maximally diverse)
    >>> round(vendi_score(embs), 4)
    3.0
    >>> vendi_score(np.ones((3, 4)))  # all identical → VS ≈ 1
    1.0
    """
    n = len(embeddings)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    embs = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    K = (embs @ embs.T) / n          # trace = 1 for unit-normalised embeddings
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = np.maximum(eigenvalues, 0.0)   # numerical safety
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 1.0
    eigenvalues /= eigenvalues.sum()             # re-normalise
    return float(np.exp(-np.sum(eigenvalues * np.log(eigenvalues + 1e-30))))


# ---------------------------------------------------------------------------
# Intent-aware diversity metrics
# ---------------------------------------------------------------------------

def subtopic_coverage_sets(
    passages: Sequence[dict],
    gold_titles: Optional[Sequence[str]] = None,
    qrel_ids: Optional[Sequence[str]] = None,
    single_subtopic: bool = False,
) -> List[Set[str]]:
    """Return per-passage subtopic coverage sets.

    Each passage covers at most one subtopic (its normalised title when
    gold_titles are used, or its document ID when qrel_ids are used).
    This matches the convention in Section 5.4 of the paper: for HotpotQA
    the two distinct gold supporting documents are the subtopics; for BEIR
    each relevant document is its own subtopic.

    Parameters
    ----------
    passages : list of dict
        Passage dicts, each with at least a ``"title"`` or ``"id"`` key.
    gold_titles : list of str, optional
        Gold supporting passage titles (HotpotQA-style).
    qrel_ids : list of str, optional
        Relevant document IDs from qrels (BEIR-style).
    single_subtopic : bool, optional
        When True, every gold passage maps to one shared subtopic
        (``"__gold__"``) rather than to its own title/id.  This models a
        genuinely single-hop query, where all answer-containing passages
        encode the *same* piece of evidence: S-Recall then becomes binary
        (the answer was retrieved or not), which is the correct semantics
        for NQ-Open, whose DPR loader marks every answer-bearing passage as
        gold (mean ~4.6 gold titles/query).  Default False (one subtopic per
        distinct gold, the multi-hop convention).

    Returns
    -------
    coverage : list of set of str
        For each passage, the set of subtopic IDs it covers (0 or 1 element).

    Examples
    --------
    >>> passages = [{"title": "Paris"}, {"title": "France"}, {"title": "Rome"}]
    >>> subtopic_coverage_sets(passages, gold_titles=["Paris", "France"])
    [{'paris'}, {'france'}, set()]
    >>> subtopic_coverage_sets(passages, gold_titles=["Paris", "France"],
    ...                        single_subtopic=True)
    [{'__gold__'}, {'__gold__'}, set()]
    """
    if gold_titles:
        gold_set = {t.lower().strip() for t in gold_titles}
        result = []
        for p in passages:
            t = (p.get("title") or "").lower().strip()
            if t and t in gold_set:
                result.append({"__gold__"} if single_subtopic else {t})
            else:
                result.append(set())
        return result
    if qrel_ids:
        qrel_set = set(qrel_ids)
        result = []
        for p in passages:
            doc_id = p.get("id") or p.get("title") or ""
            result.append({doc_id} if doc_id and doc_id in qrel_set else set())
        return result
    return [set() for _ in passages]


def _alpha_ndcg_idcg(
    pool_coverage: List[Set[str]],
    all_subtopics: Set[str],
    k: int,
    alpha_r: float,
) -> float:
    """Greedy oracle ideal DCG for alpha-NDCG."""
    counts: Dict[str, int] = {t: 0 for t in all_subtopics}
    selected: Set[int] = set()
    idcg = 0.0
    for r in range(min(k, len(pool_coverage))):
        best_gain = -1.0
        best_i = -1
        for i, cov in enumerate(pool_coverage):
            if i in selected:
                continue
            g = sum(
                (1.0 - alpha_r) ** counts[t]
                for t in cov
                if t in all_subtopics
            )
            if g > best_gain:
                best_gain = g
                best_i = i
        if best_i < 0 or best_gain <= 0.0:
            break
        idcg += best_gain / np.log2(r + 2)
        selected.add(best_i)
        for t in pool_coverage[best_i]:
            if t in all_subtopics:
                counts[t] += 1
    return idcg


def alpha_ndcg_at_k(
    selected_indices: Sequence[int],
    pool_coverage: Sequence[Set[str]],
    k: int,
    alpha_r: float = 0.5,
) -> float:
    """alpha-NDCG at k (Clarke et al. 2008).

    Relevance gain at rank r is
    ``sum_t rel(d_r, t) * (1 - alpha_r)^{c_{r-1}(t)}``
    where ``c_{r-1}(t)`` counts how many times subtopic t was covered by
    documents ranked above r.  The parameter is written ``alpha_r`` to avoid
    clashing with the diversity margin ``alpha`` used throughout the paper.

    Parameters
    ----------
    selected_indices : list of int
        Ranking order, indexing into ``pool_coverage``.
    pool_coverage : list of set of str
        Per-document subtopic coverage for the entire candidate pool.
    k : int
        Rank cutoff.
    alpha_r : float
        Redundancy parameter (``alpha_r=0`` → standard NDCG;
        ``alpha_r=1`` → binary subtopic coverage).  Default 0.5.

    Returns
    -------
    score : float in [0, 1].  Returns 0.0 when no subtopics exist.

    Examples
    --------
    >>> pool = [{'a'}, {'b'}, {'a', 'b'}, set()]
    >>> round(alpha_ndcg_at_k([0, 1, 2, 3], pool, k=2, alpha_r=0.5), 4)
    0.7044
    """
    all_subtopics: Set[str] = set()
    for cov in pool_coverage:
        all_subtopics.update(cov)
    if not all_subtopics:
        return 0.0

    counts: Dict[str, int] = {t: 0 for t in all_subtopics}
    dcg = 0.0
    for r, idx in enumerate(list(selected_indices)[:k]):
        gain = sum(
            (1.0 - alpha_r) ** counts[t]
            for t in pool_coverage[idx]
            if t in all_subtopics
        )
        dcg += gain / np.log2(r + 2)
        for t in pool_coverage[idx]:
            if t in all_subtopics:
                counts[t] += 1

    idcg = _alpha_ndcg_idcg(list(pool_coverage), all_subtopics, k, alpha_r)
    return float(dcg / idcg) if idcg > 0.0 else 0.0


def subtopic_recall_at_k(
    selected_indices: Sequence[int],
    pool_coverage: Sequence[Set[str]],
    k: int,
) -> float:
    """Subtopic Recall at k: fraction of subtopics covered by top-k.

    Parameters
    ----------
    selected_indices : list of int
        Ranking order, indexing into ``pool_coverage``.
    pool_coverage : list of set of str
        Per-document subtopic coverage for the entire candidate pool.
    k : int
        Rank cutoff.

    Returns
    -------
    score : float in [0, 1].  Returns 0.0 when no subtopics exist.

    Examples
    --------
    >>> pool = [{'a'}, {'b'}, set(), set()]
    >>> subtopic_recall_at_k([0, 2, 1, 3], pool, k=2)
    0.5
    >>> subtopic_recall_at_k([0, 1, 2, 3], pool, k=2)
    1.0
    """
    all_subtopics: Set[str] = set()
    for cov in pool_coverage:
        all_subtopics.update(cov)
    if not all_subtopics:
        return 0.0
    covered: Set[str] = set()
    for idx in list(selected_indices)[:k]:
        covered.update(pool_coverage[idx] & all_subtopics)
    return len(covered) / len(all_subtopics)


def err_ia_at_k(
    selected_indices: Sequence[int],
    pool_coverage: Sequence[Set[str]],
    k: int,
) -> float:
    """ERR-IA at k (Chapelle et al. 2011): intent-aware ERR.

    With binary per-subtopic relevance this equals the average reciprocal
    rank of first coverage over all subtopics present in the pool.

    Parameters
    ----------
    selected_indices : list of int
        Ranking order, indexing into ``pool_coverage``.
    pool_coverage : list of set of str
        Per-document subtopic coverage for the entire candidate pool.
    k : int
        Rank cutoff.

    Returns
    -------
    score : float in [0, 1].  Returns 0.0 when no subtopics exist.

    Examples
    --------
    >>> pool = [{'a'}, {'b'}, set()]
    >>> round(err_ia_at_k([0, 1, 2], pool, k=2), 4)
    0.75
    """
    all_subtopics: Set[str] = set()
    for cov in pool_coverage:
        all_subtopics.update(cov)
    if not all_subtopics:
        return 0.0

    first_rank: Dict[str, Optional[int]] = {t: None for t in all_subtopics}
    remaining = set(all_subtopics)
    for r, idx in enumerate(list(selected_indices)[:k]):
        newly = pool_coverage[idx] & remaining
        for t in newly:
            first_rank[t] = r + 1   # 1-indexed
        remaining -= newly
        if not remaining:
            break

    total_rr = sum(
        1.0 / first_rank[t]
        for t in all_subtopics
        if first_rank[t] is not None
    )
    return total_rr / len(all_subtopics)


# ---------------------------------------------------------------------------
# BEIR / qrel-based metrics
# ---------------------------------------------------------------------------

def ndcg_graded(
    selected_indices: Sequence[int],
    passages_pool: List[Dict],
    qrels: Dict[str, int],
    k: int,
) -> float:
    """NDCG@k with graded relevance from a qrel judgement dict.

    Each retrieved document's gain equals its qrel score (0 if absent).
    The ideal DCG is computed over all qrel entries, not just retrieved ones.

    Parameters
    ----------
    selected_indices : sequence of int
        Indices into *passages_pool* in ranked order.
    passages_pool : list of dict
        Pool of passages; each must have an ``"id"`` or ``"title"`` key used
        as the document identifier for qrel lookup.
    qrels : dict
        Mapping from document id to integer relevance grade.
    k : int
        Evaluation cutoff.
    """
    gains = []
    for idx in list(selected_indices)[:k]:
        doc_id = passages_pool[idx].get("id") or passages_pool[idx].get("title") or ""
        gains.append(float(qrels.get(doc_id, 0)))
    dcg = sum(g / np.log2(r + 2) for r, g in enumerate(gains))
    ideal_gains = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0.0 else 0.0


def recall_from_qrels(
    selected_indices: Sequence[int],
    passages_pool: List[Dict],
    qrels: Dict[str, int],
    k: int,
) -> float:
    """Recall@k against a qrel relevance judgement dict.

    A document is considered relevant if it appears in *qrels* (any grade > 0).

    Parameters
    ----------
    selected_indices : sequence of int
    passages_pool : list of dict
    qrels : dict
        Mapping from document id to relevance grade.
    k : int
    """
    relevant_ids = set(qrels.keys())
    total = len(relevant_ids)
    if total == 0:
        return 0.0
    retrieved = set()
    for idx in list(selected_indices)[:k]:
        doc_id = passages_pool[idx].get("id") or passages_pool[idx].get("title") or ""
        retrieved.add(doc_id)
    return len(retrieved & relevant_ids) / total
