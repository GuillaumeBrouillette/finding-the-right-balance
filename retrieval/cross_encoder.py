"""Cross-encoder relevance scoring for query-passage pairs."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


_DEFAULT_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Lightweight cross-encoder wrapper based on ``sentence-transformers``.

    Scores query-passage pairs and squashes raw logits to probabilities
    ``∈ [0, 1]`` via a sigmoid.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier for a cross-encoder checkpoint.
        Default ``"cross-encoder/ms-marco-MiniLM-L-6-v2"`` (~85 MB).
    device : str, optional
        PyTorch device string (``"cpu"`` or ``"cuda"``).  Default ``"cpu"``.
    batch_size : int, optional
        Number of query-passage pairs scored in one forward pass.
        Default 64.

    Examples
    --------
    >>> ce = CrossEncoderReranker()  # doctest: +SKIP
    >>> passages = [{"title": "Paris", "text": "Paris is the capital of France."}]
    >>> scores = ce.score("What is the capital of France?", passages)  # doctest: +SKIP
    >>> scores.shape  # doctest: +SKIP
    (1,)
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_CE_MODEL,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        from sentence_transformers import CrossEncoder

        print(f"Loading cross-encoder: {model_name} …")
        self.model = CrossEncoder(model_name, device=device, max_length=512)
        self.batch_size = batch_size

    # ------------------------------------------------------------------

    def score(
        self,
        query: str,
        passages: List[dict],
        text_field: str = "text",
        title_field: str = "title",
    ) -> np.ndarray:
        """Score all query-passage pairs and return probabilities in [0, 1].

        Parameters
        ----------
        query : str
            Natural-language query.
        passages : list of dict
            Passage dicts.  Each must have a ``text_field`` entry and
            optionally a ``title_field`` entry.  If a title is present it
            is prepended as ``"<title>: <text>"``.
        text_field : str, optional
            Key holding the passage text.  Default ``"text"``.
        title_field : str, optional
            Key holding the passage title.  Default ``"title"``.

        Returns
        -------
        probs : np.ndarray
            Float32 array of shape ``(len(passages),)`` with values in
            ``[0, 1]`` (sigmoid of the raw cross-encoder logit).

        Examples
        --------
        >>> ce = CrossEncoderReranker()  # doctest: +SKIP
        >>> ps = [{"text": "Paris is the capital of France."},
        ...       {"text": "Berlin is the capital of Germany."}]
        >>> s = ce.score("Capital of France?", ps)  # doctest: +SKIP
        >>> s[0] > s[1]  # doctest: +SKIP
        True
        """
        texts = [
            (
                ((p.get(title_field) or "") + ": " + (p.get(text_field) or "")).strip()
                or (p.get(text_field) or "")
            )
            for p in passages
        ]
        pairs = [(query, t) for t in texts]
        raw = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        raw = np.asarray(raw, dtype=np.float32)
        # Sigmoid: map R → [0, 1]
        probs = 1.0 / (1.0 + np.exp(-raw))
        return probs

    def score_pairs(
        self,
        pairs: List[Tuple[str, str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Score pre-built (query, text) pairs in a single batched call.

        Use this for bulk scoring across many queries at once; it avoids the
        per-query overhead of :meth:`score` and lets the GPU process one large
        continuous batch instead of many small ones.

        Parameters
        ----------
        pairs : list of (str, str)
            Query-text pairs to score.
        show_progress : bool, optional
            Display a tqdm progress bar.  Default ``False``.

        Returns
        -------
        probs : np.ndarray
            Float32 array of shape ``(len(pairs),)`` with values in ``[0, 1]``.
        """
        raw = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
        )
        raw = np.asarray(raw, dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-raw))
