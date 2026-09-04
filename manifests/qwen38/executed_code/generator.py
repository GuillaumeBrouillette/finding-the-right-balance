"""
Answer generators for retrieval-augmented question answering.

Supports:
  - Flan-T5 (seq2seq): google/flan-t5-small, google/flan-t5-base, …
  - Causal LMs (decoder-only): Llama, Qwen, and other AutoModelForCausalLM models
  - OpenAI-compatible servers (including vLLM), for large modern readers

Use ``load_generator(model_name, ...)`` to get the right class automatically.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Union
from urllib import error, request

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):  # type: ignore
        return iterable


_DEFAULT_MODEL = "google/flan-t5-small"
_MAX_INPUT_TOKENS = 512
_MAX_OUTPUT_TOKENS = 128


class FlanT5Generator:
    """Flan-T5 answer generator for retrieval-augmented question answering.

    Wraps HuggingFace ``T5ForConditionalGeneration`` with a simple prompt
    format that concatenates the question and the provided context passages.

    Default: ``google/flan-t5-small``  (~80 MB, ≈0.3 s/sample on modern CPU).
    Upgrade:  ``google/flan-t5-base``  (~250 MB, ≈1 s/sample) for better quality.

    Prompt format (seq2seq):
        Answer the question based on the context.
        Question: <question>
        Context: <doc1_title>: <doc1_text> | <doc2_title>: <doc2_text> | …
        Answer:

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier.
        Default ``"google/flan-t5-small"``.
    device : str, optional
        PyTorch device string (``"cpu"`` or ``"cuda"``).  Default ``"cpu"``.
    max_new_tokens : int, optional
        Maximum number of tokens to generate.  Default 128.
    num_beams : int, optional
        Number of beams for beam-search decoding.  Default 4.

    Attributes
    ----------
    tokenizer : T5Tokenizer
        Tokenizer loaded from ``model_name``.
    model : T5ForConditionalGeneration
        Flan-T5 model set to eval mode.
    device : torch.device
        Device the model lives on.
    max_new_tokens : int
        Maximum generation length.
    num_beams : int
        Beam-search width.

    Examples
    --------
    >>> gen = FlanT5Generator()  # doctest: +SKIP
    >>> passages = [{"title": "Paris", "text": "Paris is the capital of France."}]
    >>> gen.generate("What is the capital of France?", passages)  # doctest: +SKIP
    'Paris'
    """
    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
        max_new_tokens: int = _MAX_OUTPUT_TOKENS,
        num_beams: int = 4,
    ) -> None:
        """Load the Flan-T5 model and tokenizer.

        Parameters
        ----------
        model_name : str, optional
            HuggingFace model identifier for the Flan-T5 checkpoint.
            Default ``"google/flan-t5-small"``.
        device : str, optional
            PyTorch device string.  Default ``"cpu"``.
        max_new_tokens : int, optional
            Maximum number of new tokens to generate per call.  Default 128.
        num_beams : int, optional
            Number of beams for beam-search decoding.  Default 4.
        """
        print(f"Loading generator: {model_name} …")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(question: str, passages: List[Dict]) -> str:
        """Construct the Flan-T5 prompt string from a question and passages.

        Parameters
        ----------
        question : str
            Natural-language question.
        passages : list of dict
            Context passages, each containing at least a ``"text"`` key and
            optionally a ``"title"`` key.  Passages are concatenated with
            ``" | "`` as separator.

        Returns
        -------
        prompt : str
            Formatted prompt ready for tokenisation:
            ``"Answer the question based on the context.\\nQuestion: ...\\n
            Context: ...\\nAnswer:"``
        """
        ctx_parts = []
        for p in passages:
            title = p.get("title", "")
            text = p.get("text", "")
            ctx_parts.append(f"{title}: {text}" if title else text)
        context = " | ".join(ctx_parts)
        return (
            "Answer the question based on the context.\n"
            f"Question: {question}\n"
            f"Context: {context}\n"
            "Answer:"
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, question: str, passages: List[Dict]) -> str:
        """Generate an answer for a single question given context passages.

        Parameters
        ----------
        question : str
            Natural-language question.
        passages : list of dict
            Context passages, each with at least ``"text"`` (and optionally
            ``"title"``).  The prompt is truncated to ``_MAX_INPUT_TOKENS``
            tokens if necessary.

        Returns
        -------
        answer : str
            Decoded answer string with special tokens stripped and
            leading/trailing whitespace removed.

        Examples
        --------
        >>> gen = FlanT5Generator()  # doctest: +SKIP
        >>> passages = [{"title": "Eiffel Tower",
        ...              "text": "The Eiffel Tower is in Paris, France."}]
        >>> gen.generate("Where is the Eiffel Tower?", passages)  # doctest: +SKIP
        'Paris, France'
        """
        prompt = self._build_prompt(question, passages)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=_MAX_INPUT_TOKENS,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
            )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

    def generate_batch(
        self,
        questions: List[str],
        passages_list: List[List[Dict]],
        batch_size: int = 32,
        show_progress: bool = False,
        desc: str = "generating",
    ) -> List[str]:
        """Generate answers for a list of questions using padded batch inference.

        All prompts in each mini-batch are padded to the same length and sent
        to the model in a single forward pass, which keeps the GPU fed
        continuously instead of dispatching one kernel per question.

        Parameters
        ----------
        questions : list of str
            Natural-language questions, one per example.
        passages_list : list of list of dict
            For each question, a list of context passage dicts containing
            at least ``"text"`` (and optionally ``"title"``).
        batch_size : int, optional
            Number of prompts processed in each GPU call.  Default 32.
        show_progress : bool, optional
            Display a tqdm progress bar over mini-batches.  Default False.
        desc : str, optional
            Progress-bar label.

        Returns
        -------
        answers : list of str
            Generated answer strings, in the same order as ``questions``.

        Examples
        --------
        >>> gen = FlanT5Generator()  # doctest: +SKIP
        >>> qs = ["Capital of France?", "Capital of Germany?"]
        >>> ps = [
        ...     [{"title": "France", "text": "Paris is the capital."}],
        ...     [{"title": "Germany", "text": "Berlin is the capital."}],
        ... ]
        >>> gen.generate_batch(qs, ps)  # doctest: +SKIP
        ['Paris', 'Berlin']
        """
        prompts = [self._build_prompt(q, ps) for q, ps in zip(questions, passages_list)]
        results: List[str] = []
        starts = range(0, len(prompts), batch_size)
        if show_progress:
            starts = tqdm(starts, total=(len(prompts) + batch_size - 1) // batch_size,
                          desc=desc, unit="batch")
        for i in starts:
            batch = prompts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=_MAX_INPUT_TOKENS,
                truncation=True,
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    early_stopping=True,
                )
            decoded = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            results.extend(s.strip() for s in decoded)
        return results


class CausalLMGenerator:
    """Decoder-only (causal LM) answer generator for Llama, Qwen, and similar models.

    Uses ``AutoModelForCausalLM`` / ``AutoTokenizer`` so any HuggingFace
    causal LM checkpoint works out of the box.  The tokenizer is configured
    with left-padding so batched generation produces aligned outputs.

    Input length is controlled by truncating the *context passages* before
    the prompt is assembled (``max_input_tokens`` budget, default 4096),
    never by truncating the rendered prompt string.  Truncating the rendered
    prompt is unsafe for chat models: right-side truncation removes the end
    of the chat template (the assistant-turn generation prompt), so the
    model continues the user turn instead of answering — which silently
    destroys answer quality.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_new_tokens: int = _MAX_OUTPUT_TOKENS,
        num_beams: int = 1,
        max_input_tokens: int = 4096,
    ) -> None:
        print(f"Loading generator: {model_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        # Use the model's native chat format when it ships one (modern instruct
        # models such as Qwen3 / Llama-3), so the "modern instruction-tuned
        # reader" is prompted faithfully; fall back to a plain completion prompt
        # for base/seq2seq-style checkpoints without a chat template.
        self.use_chat_template = getattr(self.tokenizer, "chat_template", None) is not None
        print(f"   prompt format: {'chat template' if self.use_chat_template else 'plain completion'}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.max_input_tokens = max_input_tokens
        # Tokens reserved for the chat-template wrapper (role markers,
        # generation prompt) on top of the instruction/question/context.
        self._template_margin = 64

    def _n_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _build_prompt(self, question: str, passages: List[Dict]) -> str:
        """Assemble the prompt, budgeting the context to ``max_input_tokens``.

        Passages are added in rank order until the token budget is reached;
        the first passage that does not fit is truncated to the remaining
        budget (if at least 16 tokens remain) and the rest are dropped, so
        the instruction, the question, and the chat template always survive
        intact.
        """
        header = (
            "Answer the question based on the context. "
            "Give only a short, direct answer.\n"
            f"Question: {question}\n"
            "Context: "
        )
        budget = (self.max_input_tokens - self._n_tokens(header)
                  - self._template_margin)
        ctx_parts, used = [], 0
        for p in passages:
            title = p.get("title", "")
            text = p.get("text", "")
            part = f"{title}: {text}" if title else text
            ids = self.tokenizer(part, add_special_tokens=False)["input_ids"]
            sep = 3 if ctx_parts else 0  # " | " joiner
            if used + len(ids) + sep > budget:
                remaining = budget - used - sep
                if remaining >= 16:
                    ctx_parts.append(self.tokenizer.decode(
                        ids[:remaining], skip_special_tokens=True))
                break
            ctx_parts.append(part)
            used += len(ids) + sep
        context = " | ".join(ctx_parts)
        return f"{header}{context}\nAnswer:"

    def _render(self, question: str, passages: List[Dict]) -> str:
        """Final prompt string: chat-templated for instruct models, else plain.

        For chat models the trailing ``Answer:`` cue is dropped — the chat
        template's generation prompt opens the assistant turn instead.
        """
        prompt = self._build_prompt(question, passages)
        if not self.use_chat_template:
            return prompt
        user_msg = prompt.rsplit("\nAnswer:", 1)[0]
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, question: str, passages: List[Dict]) -> str:
        prompt = self._render(question, passages)
        # No truncation here: the context is already budgeted in
        # _build_prompt, and truncating a chat-templated string would cut
        # off the generation prompt (see class docstring).
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = out_ids[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_batch(
        self,
        questions: List[str],
        passages_list: List[List[Dict]],
        batch_size: int = 32,
        show_progress: bool = False,
        desc: str = "generating",
    ) -> List[str]:
        prompts = [self._render(q, ps) for q, ps in zip(questions, passages_list)]
        results: List[str] = []
        starts = range(0, len(prompts), batch_size)
        if show_progress:
            starts = tqdm(starts, total=(len(prompts) + batch_size - 1) // batch_size,
                          desc=desc, unit="batch")
        for i in starts:
            batch = prompts[i : i + batch_size]
            # Context is budgeted in _build_prompt; do not truncate the
            # rendered (chat-templated) prompts — see class docstring.
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            # With left-padding every sequence in the batch has the same total
            # length, so slicing at input_len recovers the generated tokens for
            # all sequences uniformly.
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            for seq in out_ids:
                new_tokens = seq[input_len:]
                results.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return results


class OpenAICompatibleGenerator:
    """Deterministic chat-completion client for a local vLLM-style server."""

    PROMPT_VERSION = "short_direct_v1"

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        max_new_tokens: int = 64,
        timeout: float = 180.0,
        seed: int = 0,
        disable_thinking: bool = True,
    ) -> None:
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.seed = seed
        self.disable_thinking = disable_thinking
        print(f"Using OpenAI-compatible generator: {model_name} at {self.api_base}")
        self._check_server()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _check_server(self) -> None:
        req = request.Request(f"{self.api_base}/models", headers=self._headers())
        try:
            with request.urlopen(req, timeout=min(self.timeout, 30.0)) as response:
                models = json.loads(response.read().decode("utf-8")).get("data", [])
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot reach generator server at {self.api_base}: {exc}") from exc
        names = {item.get("id") for item in models}
        if names and self.model_name not in names:
            raise RuntimeError(
                f"Generator server exposes {sorted(names)}, not {self.model_name!r}. "
                "Set --served-model-name to the requested model ID."
            )

    @staticmethod
    def _prompt(question: str, passages: List[Dict]) -> str:
        parts = []
        for passage in passages:
            title, body = passage.get("title", ""), passage.get("text", "")
            parts.append(f"{title}: {body}" if title else body)
        return (
            "Answer the question based on the context. "
            "Give only a short, direct answer.\n"
            f"Question: {question}\n"
            f"Context: {' | '.join(parts)}"
        )

    def _complete(self, question: str, passages: List[Dict]) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": self._prompt(question, passages)}],
            "max_tokens": self.max_new_tokens,
            "temperature": 0,
            "top_p": 1,
            "seed": self.seed,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        result = None
        for attempt in range(3):
            req = request.Request(
                f"{self.api_base}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=self._headers(), method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                break
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                # Invalid requests are deterministic; retry only server faults.
                if exc.code < 500 or attempt == 2:
                    raise RuntimeError(
                        f"Generator server returned HTTP {exc.code}: {detail}") from exc
            except (error.URLError, TimeoutError) as exc:
                if attempt == 2:
                    raise RuntimeError(f"Generator request failed after 3 attempts: {exc}") from exc
            time.sleep(2 ** attempt)
        assert result is not None
        return result["choices"][0]["message"]["content"].strip()

    def generate(self, question: str, passages: List[Dict]) -> str:
        return self._complete(question, passages)

    def generate_batch(
        self,
        questions: List[str],
        passages_list: List[List[Dict]],
        batch_size: int = 8,
        show_progress: bool = False,
        desc: str = "generating",
    ) -> List[str]:
        pairs = list(zip(questions, passages_list))
        with ThreadPoolExecutor(max_workers=max(1, batch_size)) as executor:
            outputs = executor.map(lambda pair: self._complete(*pair), pairs)
            if show_progress:
                outputs = tqdm(outputs, total=len(pairs), desc=desc, unit="answer")
            return list(outputs)


def load_generator(
    model_name: str,
    device: str = "cpu",
    max_new_tokens: int = _MAX_OUTPUT_TOKENS,
    num_beams: int = 4,
    backend: str = "transformers",
    api_base: str = "http://127.0.0.1:8000/v1",
    api_key: str = "EMPTY",
    timeout: float = 180.0,
    seed: int = 0,
) -> Union[FlanT5Generator, CausalLMGenerator, OpenAICompatibleGenerator]:
    """Return a ``FlanT5Generator`` for T5 checkpoints or a ``CausalLMGenerator`` for all others."""
    if backend == "openai-compatible":
        return OpenAICompatibleGenerator(
            model_name=model_name, api_base=api_base, api_key=api_key,
            max_new_tokens=max_new_tokens, timeout=timeout, seed=seed,
        )
    if "t5" in model_name.lower():
        return FlanT5Generator(model_name=model_name, device=device, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return CausalLMGenerator(model_name=model_name, device=device, max_new_tokens=max_new_tokens, num_beams=num_beams)
