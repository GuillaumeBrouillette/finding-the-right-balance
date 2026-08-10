"""
Dataset loaders.

Supported datasets
------------------
hotpotqa          : HotpotQA distractor setting (Yang et al. 2018).
                    Each example ships with 10 candidate paragraphs
                    (2 gold + 8 distractors).
                    HuggingFace: hotpotqa/hotpot_qa / distractor.

nq_open           : Natural Questions Open (Lee et al. 2019).
                    No passages attached; requires external retriever.
                    HuggingFace: google-research-datasets/nq_open.

nq                : Natural Questions Open with DPR pre-retrieved top-100
                    candidate passages (Karpukhin et al. 2020).
                    Data: https://dl.fbaipublicfiles.com/dpr/data/retriever/
                    Downloaded automatically to data/dpr/ on first use.

trivia            : TriviaQA with DPR pre-retrieved top-100 candidate passages.
                    Same format and source as nq.

squad             : SQuAD v1.1 (Rajpurkar et al. 2016).
                    Passages are all paragraphs of the same Wikipedia article.
                    HuggingFace: rajpurkar/squad.

2wikimultihopqa   : 2WikiMultiHopQA distractor setting (Ho et al. 2020).
                    Each example ships with 10 candidate paragraphs,
                    same format as HotpotQA distractor.
                    HuggingFace: xanhho/2WikiMultiHopQA.

musique           : MuSiQue answerable split (Trivedi et al. 2022).
                    Each example ships with ~20 candidate paragraphs
                    (supporting + distractors).
                    HuggingFace: dgslibisey/musique.

All loaders return a list of dicts:
    {
        "id"         : str,
        "question"   : str,
        "answers"    : List[str],
        "passages"   : List[{"title": str, "text": str}] | None,
        "gold_titles": List[str] | None,
    }

When passages is None the caller must use DenseRetriever over a pre-built
corpus index (NQ-Open, BEIR datasets).
"""

from __future__ import annotations

import gzip
import json
import os
import urllib.request
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset


_LOCAL_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# HotpotQA – distractor setting
# ---------------------------------------------------------------------------

def load_hotpotqa(
    split: str = "validation",
    max_samples: Optional[int] = None,
    config: str = "distractor",
) -> List[Dict]:
    """
    Load HotpotQA in the distractor setting.

    Each example contains exactly 10 paragraphs (2 gold + 8 distractors).

    Parameters
    ----------
    split : str, optional
        Dataset split to load.  One of ``"train"`` or ``"validation"``.
        Test labels are hidden.  Default ``"validation"``.
    max_samples : int, optional
        If given, only the first ``max_samples`` examples are returned.
        Useful for quick smoke-tests.

    Returns
    -------
    examples : list of dict
        Each dict has the keys:

        * ``"id"``          : str — unique example identifier.
        * ``"question"``    : str — natural-language question.
        * ``"answers"``     : list of str — list of acceptable answers (single element).
        * ``"passages"``    : list of {``"title"``: str, ``"text"``: str} — exactly 10
          candidate paragraphs (2 gold + 8 distractors).
        * ``"gold_titles"`` : list of str — titles of the gold supporting paragraphs.

    Examples
    --------
    >>> examples = load_hotpotqa(split="validation", max_samples=3)  # doctest: +SKIP
    >>> len(examples)  # doctest: +SKIP
    3
    >>> set(examples[0].keys())  # doctest: +SKIP
    {'id', 'question', 'answers', 'passages', 'gold_titles'}
    """
    ds = load_dataset("hotpotqa/hotpot_qa", config, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    examples = []
    for row in ds:
        # Build passage list: context is a dict with "title" and "sentences" lists
        passages = []
        for title, sentences in zip(
            row["context"]["title"], row["context"]["sentences"]
        ):
            text = " ".join(sentences)
            passages.append({"title": title, "text": text})

        # Gold titles come from the "supporting_facts" field
        gold_titles = list(set(row["supporting_facts"]["title"]))

        examples.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answers": [row["answer"]],
                "passages": passages,
                "gold_titles": gold_titles,
            }
        )

    return examples


# ---------------------------------------------------------------------------
# NQ-Open – requires external retriever
# ---------------------------------------------------------------------------

def load_nq_open(
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Load Natural Questions Open.

    No passages are attached; the caller must retrieve them via DenseRetriever.

    Parameters
    ----------
    split : str, optional
        Dataset split to load.  One of ``"train"`` or ``"validation"``.
        Default ``"validation"``.
    max_samples : int, optional
        If given, only the first ``max_samples`` examples are returned.

    Returns
    -------
    examples : list of dict
        Each dict has the keys:

        * ``"id"``          : str — sequential index string.
        * ``"question"``    : str — natural-language question.
        * ``"answers"``     : list of str — acceptable short answers.
        * ``"passages"``    : None — no passages are attached; the caller must
          use :class:`~retrieval.retriever.DenseRetriever` to fill these in.
        * ``"gold_titles"`` : None — not available in NQ-Open.

    Examples
    --------
    >>> examples = load_nq_open(split="validation", max_samples=2)  # doctest: +SKIP
    >>> len(examples)  # doctest: +SKIP
    2
    >>> examples[0]["passages"] is None  # doctest: +SKIP
    True
    """
    ds = load_dataset("google-research-datasets/nq_open", split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    examples = []
    for i, row in enumerate(ds):
        examples.append(
            {
                "id": str(i),
                "question": row["question"],
                "answers": row["answer"],
                "passages": None,
                "gold_titles": None,
            }
        )

    return examples


# ---------------------------------------------------------------------------
# SQuAD v1.1 – passages come from Wikipedia article paragraphs
# ---------------------------------------------------------------------------

def load_squad(
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Load SQuAD v1.1.

    Each Wikipedia article is split into multiple paragraphs; all paragraphs
    from the same article form the candidate pool for every question drawn
    from that article.  The gold passage is the paragraph that contains the
    answer; it is identified by a stable unique label ``"{title}|{index}"``
    used as its ``"title"`` key throughout the pipeline.

    Parameters
    ----------
    split : str, optional
        Dataset split to load.  One of ``"train"`` or ``"validation"``.
        Default ``"validation"``.
    max_samples : int, optional
        If given, only the first ``max_samples`` examples (questions) are
        returned.

    Returns
    -------
    examples : list of dict
        Each dict has the keys:

        * ``"id"``          : str — unique question identifier.
        * ``"question"``    : str — natural-language question.
        * ``"answers"``     : list of str — acceptable answer strings.
        * ``"passages"``    : list of {``"title"``: str, ``"text"``: str} —
          all paragraphs from the same Wikipedia article, each with a stable
          unique ``"title"`` of the form ``"{article_title}|{index}"``.
        * ``"gold_titles"`` : list of str — single-element list containing
          the unique ``"title"`` of the gold paragraph.

    Examples
    --------
    >>> examples = load_squad(split="validation", max_samples=3)  # doctest: +SKIP
    >>> len(examples)  # doctest: +SKIP
    3
    >>> examples[0]["passages"] is not None  # doctest: +SKIP
    True
    """
    ds = load_dataset("rajpurkar/squad", split=split)

    # Collect unique paragraphs per article, preserving first-seen order.
    title_to_contexts: Dict[str, List[str]] = {}
    for row in ds:
        t = row["title"]
        if t not in title_to_contexts:
            title_to_contexts[t] = []
        if row["context"] not in title_to_contexts[t]:
            title_to_contexts[t].append(row["context"])

    # Build passage lists with stable per-paragraph unique IDs.
    # Each passage "title" is "{article_title}|{paragraph_index}".
    title_to_passages: Dict[str, List[Dict]] = {
        t: [{"title": f"{t}|{i}", "text": ctx} for i, ctx in enumerate(ctxs)]
        for t, ctxs in title_to_contexts.items()
    }
    # Reverse map: (article_title, context_text) → unique passage title
    context_to_pid: Dict[tuple, str] = {
        (t, ctx): f"{t}|{i}"
        for t, ctxs in title_to_contexts.items()
        for i, ctx in enumerate(ctxs)
    }

    examples: List[Dict] = []
    seen_ids: set = set()
    for row in ds:
        qid = row["id"]
        if qid in seen_ids:
            continue
        seen_ids.add(qid)

        t = row["title"]
        gold_pid = context_to_pid[(t, row["context"])]

        examples.append(
            {
                "id": qid,
                "question": row["question"],
                "answers": row["answers"]["text"],
                "passages": title_to_passages[t],
                "gold_titles": [gold_pid],
            }
        )

        if max_samples is not None and len(examples) >= max_samples:
            break

    return examples


# ---------------------------------------------------------------------------
# 2WikiMultiHopQA – distractor setting
# ---------------------------------------------------------------------------

def load_2wikimultihopqa(
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Load 2WikiMultiHopQA in the distractor setting (Ho et al. 2020).

    The context format is identical to HotpotQA distractor: 10 candidate
    paragraphs per question (supporting + distractors).

    HuggingFace dataset: ``xanhho/2WikiMultiHopQA``.

    Parameters
    ----------
    split : str, optional
        One of ``"train"``, ``"validation"``, or ``"test"``.
        Default ``"validation"``.
    max_samples : int, optional
        Cap on the number of examples returned.
    """
    local_dev = os.path.join(
        _LOCAL_DATA_DIR, "2wikimultihopqa", "dev.parquet"
    )
    if split == "validation" and os.path.isfile(local_dev):
        # Immutable snapshot recorded by the reproducibility manifest.
        ds = load_dataset("parquet", data_files={split: local_dev}, split=split)
    else:
        ds = load_dataset(
            "xanhho/2WikiMultiHopQA",
            revision="612bc5039a457880d9e7d84c3b0a4cf154b70e4f",
            split=split,
        )
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    examples = []
    for row in ds:
        # context and supporting_facts are stored as JSON strings in this HF upload
        context = json.loads(row["context"])        # [[title, [sent, ...]], ...]
        supporting_facts = json.loads(row["supporting_facts"])  # [[title, sent_id], ...]

        passages = [
            {"title": title, "text": " ".join(sentences)}
            for title, sentences in context
        ]
        gold_titles = list({title for title, _ in supporting_facts})

        examples.append(
            {
                "id": row["_id"],
                "question": row["question"],
                "answers": [row["answer"]],
                "passages": passages,
                "gold_titles": gold_titles,
            }
        )

    return examples


# ---------------------------------------------------------------------------
# MuSiQue – answerable split, distractor setting
# ---------------------------------------------------------------------------

def load_musique(
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Load MuSiQue answerable split (Trivedi et al. 2022).

    Each example contains ~20 candidate paragraphs (supporting + distractors).
    Only answerable examples are included (``answerable == True``).

    HuggingFace dataset: ``dgslibisey/musique``.

    Parameters
    ----------
    split : str, optional
        One of ``"train"``, ``"validation"``, or ``"test"``.
        Default ``"validation"``.
    max_samples : int, optional
        Cap on the number of examples returned.
    """
    local_dev = os.path.join(
        _LOCAL_DATA_DIR, "musique", "musique_ans_v1.0_dev.jsonl"
    )
    if split == "validation" and os.path.isfile(local_dev):
        # Immutable snapshot recorded by the reproducibility manifest.
        ds = load_dataset("json", data_files={split: local_dev}, split=split)
    else:
        ds = load_dataset(
            "dgslibisey/musique",
            revision="c8f4f8c9465fb69d31a8eae894c3fd509c4ca321",
            split=split,
        )
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    examples = []
    for row in ds:
        if not row.get("answerable", True):
            continue

        passages = [
            {"title": p["title"], "text": p["paragraph_text"]}
            for p in row["paragraphs"]
        ]
        gold_titles = [
            p["title"] for p in row["paragraphs"] if p.get("is_supporting", False)
        ]
        answers = [row["answer"]] + [a for a in row.get("answer_aliases", []) if a]

        examples.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answers": answers,
                "passages": passages,
                "gold_titles": gold_titles,
            }
        )

    return examples


# ---------------------------------------------------------------------------
# MDR pre-retrieved HotpotQA fullwiki (top-100 chains)
# ---------------------------------------------------------------------------

_MDR_HOTPOTQA_URL = "https://dl.fbaipublicfiles.com/mdpr/data/hotpot/dev_retrieval_top100_sp.json"
_MDR_DATA_DIR = os.path.join(os.path.dirname(__file__), "mdr")


def load_hotpotqa_fullwiki(
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Load HotpotQA fullwiki with MDR top-100 pre-retrieved passage chains.

    Uses the Multi-hop Dense Retrieval (MDR, Xiong et al. 2021) retrieval
    results for the HotpotQA dev set.  Each question comes with 100 two-hop
    candidate chains; flattening their unique passages yields a realistic pool
    of ~80–120 candidates with exactly 2 gold supporting passages from
    different Wikipedia articles.

    The file (~652 MB) is downloaded automatically from the Facebook AI
    Research CDN on first use and cached under ``data/mdr/``.

    Parameters
    ----------
    split : str
        Only ``"validation"`` is available (MDR only released dev results).
    max_samples : int, optional
        Cap on the number of examples returned.

    Returns
    -------
    examples : list of dict
        Standard format with ``passages`` (unique passages across all chains,
        ordered by first appearance) and ``gold_titles`` (the two supporting
        Wikipedia article titles).
    """
    if split != "validation":
        raise ValueError(
            "MDR HotpotQA fullwiki only has a 'validation' split. "
            "Train/test pre-retrieved pools are not publicly available."
        )

    os.makedirs(_MDR_DATA_DIR, exist_ok=True)
    local_path = os.path.join(_MDR_DATA_DIR, "hotpotqa_dev_top100.json")
    if not os.path.exists(local_path):
        print(f"   Downloading MDR HotpotQA top-100 retrieval results (~652 MB) …")
        req = urllib.request.Request(_MDR_HOTPOTQA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(local_path, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1 << 20
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                f.write(block)
                downloaded += len(block)
                if total:
                    print(f"\r   {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB",
                          end="", flush=True)
        print(f"\r   Saved to {local_path}          ")

    examples: List[Dict] = []
    decoder = json.JSONDecoder()
    with open(local_path, "r", encoding="utf-8") as f:
        raw = f.read()

    pos = 0
    while pos < len(raw):
        # skip whitespace between objects
        while pos < len(raw) and raw[pos] in " \n\r\t":
            pos += 1
        if pos >= len(raw):
            break
        row, end = decoder.raw_decode(raw, pos)
        pos = end

        # Flatten unique passages from all candidate chains (order = first seen)
        seen: Dict[str, Dict] = {}
        for chain in row["candidate_chains"]:
            for p in chain:
                title = p["title"]
                if title not in seen:
                    seen[title] = {
                        "title": title,
                        "text": " ".join(p["sents"]),
                    }

        gold_titles = [sp["title"] for sp in row["sp"]]
        answers = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]

        examples.append({
            "id": row["_id"],
            "question": row["question"],
            "answers": answers,
            "passages": list(seen.values()),
            "gold_titles": gold_titles,
        })

        if max_samples is not None and len(examples) >= max_samples:
            break

    return examples


# ---------------------------------------------------------------------------
# DPR pre-retrieved datasets (NQ and TriviaQA)
# ---------------------------------------------------------------------------

# NQ retriever-results (top-100 passages with has_answer flags) are hosted under
# retriever_results/single/.  TriviaQA retriever-result files are not publicly
# accessible on the DPR CDN; only the biencoder training data is (different format).
_DPR_URLS: Dict[str, Dict[str, str]] = {
    "nq": {
        "validation": "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-dev.json.gz",
        "test":       "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-test.json.gz",
    },
}

_DPR_DATA_DIR = os.path.join(os.path.dirname(__file__), "dpr")


def _ensure_dpr_file(dataset: str, split: str) -> str:
    """Return path to the local DPR file, downloading it if necessary."""
    if dataset not in _DPR_URLS:
        raise NotImplementedError(
            f"DPR pre-retrieved data for {dataset!r} is not publicly available "
            "in the required format on the DPR CDN. Only 'nq' is supported."
        )
    if split not in _DPR_URLS[dataset]:
        raise ValueError(
            f"DPR dataset {dataset!r} has no {split!r} split. "
            f"Available: {list(_DPR_URLS[dataset])}"
        )
    url = _DPR_URLS[dataset][split]
    filename = url.split("/")[-1]
    os.makedirs(_DPR_DATA_DIR, exist_ok=True)
    local_path = os.path.join(_DPR_DATA_DIR, filename)
    if not os.path.exists(local_path):
        print(f"   Downloading {filename} …")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(local_path, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1 << 20  # 1 MB
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                f.write(block)
                downloaded += len(block)
                if total:
                    print(f"\r   {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print(f"\r   Saved to {local_path}          ")
    return local_path


def load_dpr(
    dataset: str,
    split: str = "validation",
    max_samples: Optional[int] = None,
    top_k: Optional[int] = None,
) -> List[Dict]:
    """Load NQ or TriviaQA from DPR pre-retrieved top-100 candidate passages.

    Each example ships with up to 100 candidate paragraphs retrieved by the
    DPR bi-encoder (Karpukhin et al. 2020) from the Wikipedia 100-word passage
    split.  Gold passages are those marked ``has_answer=True`` by the DPR
    pipeline (i.e. passages whose text contains at least one answer string).

    Files are downloaded automatically from the Facebook AI Research CDN on
    first use and cached under ``data/dpr/``.

    Parameters
    ----------
    dataset : str
        ``"nq"`` or ``"trivia"``.
    split : str, optional
        ``"validation"`` (dev set) or ``"test"``.  Default ``"validation"``.
    max_samples : int, optional
        Cap on the number of examples returned.
    top_k : int, optional
        Retain only the top-k candidates per question (default: all 100).

    Returns
    -------
    examples : list of dict
        Each dict has the keys:

        * ``"id"``          : str — sequential index string.
        * ``"question"``    : str — natural-language question.
        * ``"answers"``     : list of str — acceptable short answers.
        * ``"passages"``    : list of {``"title"``: str, ``"text"``: str} —
          up to 100 candidate paragraphs, ordered by DPR retrieval score.
        * ``"gold_titles"`` : list of str — titles of passages that contain
          at least one answer string (``has_answer=True``).
    """
    path = _ensure_dpr_file(dataset, split)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    if max_samples is not None:
        data = data[:max_samples]

    examples = []
    for i, row in enumerate(data):
        ctxs = row["ctxs"]
        if top_k is not None:
            ctxs = ctxs[:top_k]
        passages = [{"title": c["title"], "text": c["text"]} for c in ctxs]
        gold_titles = [c["title"] for c in ctxs if c.get("has_answer", False)]
        examples.append(
            {
                "id": str(i),
                "question": row["question"],
                "answers": row["answers"],
                "passages": passages,
                "gold_titles": gold_titles if gold_titles else None,
            }
        )

    return examples


def load_nq_dpr(
    split: str = "validation",
    max_samples: Optional[int] = None,
    top_k: Optional[int] = None,
) -> List[Dict]:
    """Load Natural Questions with DPR pre-retrieved top-100 passages.

    Convenience wrapper around :func:`load_dpr` with ``dataset="nq"``.
    """
    return load_dpr("nq", split=split, max_samples=max_samples, top_k=top_k)


def load_trivia_dpr(
    split: str = "validation",
    max_samples: Optional[int] = None,
    top_k: Optional[int] = None,
) -> List[Dict]:
    """Load TriviaQA with DPR pre-retrieved top-100 passages.

    Convenience wrapper around :func:`load_dpr` with ``dataset="trivia"``.
    """
    return load_dpr("trivia", split=split, max_samples=max_samples, top_k=top_k)


# ---------------------------------------------------------------------------
# BEIR – corpus-level retrieval benchmarks (RQ3)
# ---------------------------------------------------------------------------

def load_beir_dataset(
    name: str,
    split: str = "test",
    max_queries: Optional[int] = None,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Load a BEIR dataset for retrieval evaluation.

    Retrieves queries, builds a corpus dict, and collects qrels.  Passages
    are set to ``None`` because the caller must fill them via a FAISS index
    built from the returned corpus dict (same pattern as NQ-Open).

    Parameters
    ----------
    name : str
        BEIR dataset short name accepted by the ``BeIR/`` HuggingFace
        organisation: ``"scifact"``, ``"fiqa"``, or ``"trec-covid"``.
    split : str, optional
        Query split.  Typical values: ``"test"``, ``"validation"``.
        Default ``"test"``.
    max_queries : int, optional
        Cap on the number of queries (useful for quick smoke-tests).

    Returns
    -------
    examples : list of dict
        Each dict has:

        * ``"id"``       : str — query ID.
        * ``"question"`` : str — query text.
        * ``"answers"``  : list (empty; BEIR uses qrels, not answer strings).
        * ``"passages"`` : None — filled in by a ``DenseRetriever``.
        * ``"gold_titles"`` : None.
        * ``"qrels"``    : dict of {doc_id: relevance_score} for this query.

    corpus : dict of {doc_id: {"id": str, "title": str, "text": str}}
        Mapping used to build the FAISS index.

    Examples
    --------
    >>> examples, corpus = load_beir_dataset("scifact", max_queries=5)  # doctest: +SKIP
    >>> len(corpus) > 0  # doctest: +SKIP
    True
    """
    hf_name = f"BeIR/{name}"

    # --- corpus -------------------------------------------------------
    corpus_ds = load_dataset(hf_name, "corpus", split="corpus")
    corpus: Dict[str, Dict] = {}
    for row in corpus_ds:
        doc_id = row["_id"]
        corpus[doc_id] = {
            "id": doc_id,
            "title": row.get("title") or "",
            "text": row.get("text") or "",
        }

    # --- queries ------------------------------------------------------
    queries_ds = load_dataset(hf_name, "queries", split="queries")
    queries: Dict[str, str] = {row["_id"]: row["text"] for row in queries_ds}

    # --- qrels --------------------------------------------------------
    qrels_hf_name = f"BeIR/{name}-qrels"
    try:
        qrels_ds = load_dataset(qrels_hf_name, split=split)
    except Exception:
        # fall back to "test" if the requested split is missing
        qrels_ds = load_dataset(qrels_hf_name, split="test")

    qrels: Dict[str, Dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        score = int(row["score"])
        if score > 0:
            qrels.setdefault(qid, {})[did] = score

    # --- build examples -----------------------------------------------
    query_ids = [qid for qid in qrels if qid in queries]
    if max_queries is not None:
        query_ids = query_ids[:max_queries]

    examples: List[Dict] = []
    for qid in query_ids:
        examples.append(
            {
                "id": qid,
                "question": queries[qid],
                "answers": [],
                "passages": None,
                "gold_titles": None,
                "qrels": qrels[qid],
            }
        )

    return examples, corpus


def load_scifact(
    split: str = "test",
    max_queries: Optional[int] = None,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Load the SciFact BEIR task (scientific claim verification).

    SciFact has 300 queries and ~5 000 documents; it fits in memory without
    difficulty.  One query typically has 1–2 relevant documents.

    Returns the same (examples, corpus) tuple as :func:`load_beir_dataset`.

    Examples
    --------
    >>> examples, corpus = load_scifact(max_queries=10)  # doctest: +SKIP
    >>> len(corpus)  # doctest: +SKIP
    5183
    """
    return load_beir_dataset("scifact", split=split, max_queries=max_queries)


def load_fiqa(
    split: str = "test",
    max_queries: Optional[int] = None,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Load the FiQA-2018 BEIR task (financial opinion QA).

    FiQA has ~648 queries and ~57 000 documents.

    Returns the same (examples, corpus) tuple as :func:`load_beir_dataset`.

    Examples
    --------
    >>> examples, corpus = load_fiqa(max_queries=10)  # doctest: +SKIP
    >>> len(corpus) > 50000  # doctest: +SKIP
    True
    """
    return load_beir_dataset("fiqa", split=split, max_queries=max_queries)


def load_trec_covid(
    split: str = "test",
    max_queries: Optional[int] = None,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Load the TREC-COVID BEIR task (biomedical literature retrieval).

    TREC-COVID has 50 queries and ~171 000 documents; indexing requires
    significant memory (≈3 GB) and time on CPU.  Use ``max_queries`` to
    restrict evaluation for quick tests.

    Returns the same (examples, corpus) tuple as :func:`load_beir_dataset`.

    Examples
    --------
    >>> examples, corpus = load_trec_covid(max_queries=5)  # doctest: +SKIP
    >>> examples[0]["qrels"]  # doctest: +SKIP  # noqa: E501
    {'doc_id_1': 2, 'doc_id_2': 1, ...}
    """
    return load_beir_dataset("trec-covid", split=split, max_queries=max_queries)
