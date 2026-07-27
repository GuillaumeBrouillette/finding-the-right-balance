"""
Dense retriever: SentenceTransformers encoder + FAISS flat inner-product index.

Used in open-domain (corpus-scale) mode.  For the reading-comprehension mode
(HotpotQA distractor) the retriever is not needed — passages are provided
per-question and embedded on the fly in evaluate.py.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """
    Thin wrapper around SentenceTransformer + FAISS IndexFlatIP.

    All embeddings are L2-normalised so inner product == cosine similarity.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier for the SentenceTransformer encoder.
        Default ``"sentence-transformers/all-MiniLM-L6-v2"``.
    device : str, optional
        PyTorch device string (``"cpu"`` or ``"cuda"``).  Default ``"cpu"``.
    batch_size : int, optional
        Batch size used by :meth:`encode`.  Default 64.

    Attributes
    ----------
    model : SentenceTransformer
        The underlying encoder model.
    batch_size : int
        Encoding batch size.
    index : faiss.Index or None
        FAISS flat inner-product index.  ``None`` until :meth:`build_index`
        or :meth:`load` is called.
    passages : list of dict or None
        Passage dictionaries associated with the index.  ``None`` until
        the index is built or loaded.
    passage_embeddings : np.ndarray or None
        (n, D) float32 array of L2-normalised passage embeddings.  ``None``
        until the index is built or loaded.

    Examples
    --------
    >>> from retrieval.retriever import DenseRetriever
    >>> retriever = DenseRetriever(
    ...     model_name="sentence-transformers/all-MiniLM-L6-v2",
    ...     device="cpu",
    ...     batch_size=32,
    ... )
    >>> emb = retriever.encode(["Hello world"])
    >>> emb.shape[1]  # embedding dimension depends on model
    384
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        """Initialise the retriever with a SentenceTransformer encoder.

        Parameters
        ----------
        model_name : str, optional
            HuggingFace model identifier for the SentenceTransformer encoder.
            Default ``"sentence-transformers/all-MiniLM-L6-v2"``.
        device : str, optional
            PyTorch device string.  Default ``"cpu"``.
        batch_size : int, optional
            Number of texts per encoding batch.  Default 64.
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.index: Optional[faiss.Index] = None
        self.passages: Optional[List[Dict]] = None
        self.passage_embeddings: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of texts into dense embeddings.

        Parameters
        ----------
        texts : list of str
            Texts to encode.
        normalize : bool, optional
            If ``True`` (default), L2-normalise each embedding so that
            inner products equal cosine similarities.
        show_progress : bool, optional
            Display a tqdm progress bar during encoding.  Default ``False``.

        Returns
        -------
        embeddings : np.ndarray
            (len(texts), D) float32 array of embeddings, where D is the
            model's output dimension.

        Examples
        --------
        >>> retriever = DenseRetriever()  # doctest: +SKIP
        >>> embs = retriever.encode(["hello", "world"], normalize=True)  # doctest: +SKIP
        >>> embs.shape  # doctest: +SKIP
        (2, 384)
        """
        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embs.astype(np.float32)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(
        self,
        passages: List[Dict],
        text_field: str = "text",
        title_field: str = "title",
        show_progress: bool = True,
    ) -> None:
        """Encode all passages and build a FAISS flat inner-product index.

        Parameters
        ----------
        passages : list of dict
            Passage dictionaries, each containing at least the fields named
            by ``text_field`` and (optionally) ``title_field``.
        text_field : str, optional
            Key in each passage dict holding the passage text.  Default
            ``"text"``.
        title_field : str, optional
            Key in each passage dict holding the passage title.  Default
            ``"title"``.  If present, the title is prepended to the text
            before encoding: ``"<title>: <text>"``.
        show_progress : bool, optional
            Display a tqdm progress bar during encoding.  Default ``True``.

        Examples
        --------
        >>> passages = [{"title": "A", "text": "first passage"},
        ...             {"title": "B", "text": "second passage"}]
        >>> retriever = DenseRetriever()  # doctest: +SKIP
        >>> retriever.build_index(passages)  # doctest: +SKIP
        Encoding 2 passages …
        Index built: 2 vectors of dim 384.
        """
        self.passages = passages
        texts = [
            (p.get(title_field, "") + ": " + p.get(text_field, "")).strip()
            for p in passages
        ]
        print(f"Encoding {len(texts)} passages …")
        embs = self.encode(texts, normalize=True, show_progress=show_progress)
        d = embs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embs)
        self.passage_embeddings = embs
        print(f"Index built: {self.index.ntotal} vectors of dim {d}.")

    def save(self, directory: str) -> None:
        """Persist the FAISS index, embeddings, and passages to disk.

        Three files are written under ``directory``:

        * ``index.faiss``   – FAISS flat index.
        * ``embeddings.npy`` – (n, D) float32 passage-embedding array.
        * ``passages.pkl``  – pickled list of passage dicts.

        Parameters
        ----------
        directory : str
            Target directory path.  Created automatically if it does not
            exist yet.

        Examples
        --------
        >>> retriever.save("corpus_index")  # doctest: +SKIP
        Index saved to corpus_index.
        """
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        np.save(os.path.join(directory, "embeddings.npy"), self.passage_embeddings)
        with open(os.path.join(directory, "passages.pkl"), "wb") as f:
            pickle.dump(self.passages, f)
        print(f"Index saved to {directory}.")

    def load(self, directory: str) -> None:
        """Load a previously saved FAISS index, embeddings, and passages.

        Reads the three files written by :meth:`save` from ``directory`` and
        populates ``self.index``, ``self.passage_embeddings``, and
        ``self.passages``.

        Parameters
        ----------
        directory : str
            Directory containing ``index.faiss``, ``embeddings.npy``, and
            ``passages.pkl``.

        Examples
        --------
        >>> retriever = DenseRetriever()  # doctest: +SKIP
        >>> retriever.load("corpus_index")  # doctest: +SKIP
        Index loaded: 10000 passages.
        """
        self.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        self.passage_embeddings = np.load(os.path.join(directory, "embeddings.npy"))
        with open(os.path.join(directory, "passages.pkl"), "rb") as f:
            self.passages = pickle.load(f)
        print(f"Index loaded: {self.index.ntotal} passages.")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 100,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode all queries in one batch and run FAISS search for all at once.

        Preferred over calling :meth:`search` in a loop: one large encode call
        keeps the GPU fed continuously instead of launching a new kernel for
        every single query.

        Parameters
        ----------
        queries : list of str
            Query strings.
        top_k : int, optional
            Number of top passages to retrieve per query.  Default 100.
        show_progress : bool, optional
            Show a tqdm progress bar while encoding.  Default True.

        Returns
        -------
        q_embs : np.ndarray
            (len(queries), D) float32 query embeddings (L2-normalised).
        scores : np.ndarray
            (len(queries), top_k) float32 similarity scores.
        indices : np.ndarray
            (len(queries), top_k) int64 passage indices into
            ``self.passages`` / ``self.passage_embeddings``.

        Examples
        --------
        >>> retriever = DenseRetriever()  # doctest: +SKIP
        >>> retriever.load("corpus_index")  # doctest: +SKIP
        >>> q_embs, scores, idxs = retriever.search_batch(  # doctest: +SKIP
        ...     ["What is the capital of France?", "Who wrote Hamlet?"], top_k=5
        ... )
        >>> q_embs.shape[0]  # doctest: +SKIP
        2
        """
        assert self.index is not None, "Call build_index() or load() first."
        q_embs = self.encode(queries, normalize=True, show_progress=show_progress)
        scores, indices = self.index.search(q_embs, top_k)
        return q_embs, scores, indices

    def search(
        self,
        query: str,
        top_k: int = 100,
    ) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
        """Encode a query and retrieve the top-k passages by cosine similarity.

        Parameters
        ----------
        query : str
            Natural-language query string.
        top_k : int, optional
            Number of top-scoring passages to return.  Default 100.

        Returns
        -------
        indices : list of int
            Indices (into ``self.passages``) of the top-k passages.
        scores : np.ndarray
            (top_k,) float32 array of cosine-similarity scores.
        embeddings : np.ndarray
            (top_k, D) float32 array of the retrieved passage embeddings.
        q_emb : np.ndarray
            (D,) float32 query embedding used for the search.

        Examples
        --------
        >>> retriever = DenseRetriever()  # doctest: +SKIP
        >>> retriever.load("corpus_index")  # doctest: +SKIP
        >>> idxs, scores, embs, q_emb = retriever.search(
        ...     "What is the capital of France?", top_k=5
        ... )  # doctest: +SKIP
        >>> len(idxs)  # doctest: +SKIP
        5
        """
        assert self.index is not None, "Call build_index() or load() first."
        q_emb = self.encode([query])[0]
        scores, idxs = self.index.search(q_emb.reshape(1, -1), top_k)
        idxs = idxs[0].tolist()
        scores = scores[0]
        embs = self.passage_embeddings[idxs]
        return idxs, scores, embs, q_emb
