"""Focused tests for the reproducible OpenAI-compatible reader path."""

import csv
import json

import pytest

from experiments.evaluate_redundancy import _validate_reuse_source
from generation.generator import OpenAICompatibleGenerator
from ftrb.run_utils import save_csv
from reproducibility.build_qwen38_rq1_table import filter_partition


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_qwen_prompt_is_fixed_and_short() -> None:
    prompt = OpenAICompatibleGenerator._prompt(
        "Where?", [{"title": "Place", "text": "The answer is here."}]
    )
    assert prompt == (
        "Answer the question based on the context. Give only a short, direct answer.\n"
        "Question: Where?\nContext: Place: The answer is here."
    )


def test_openai_payload_fixes_decoding_and_disables_thinking(monkeypatch) -> None:
    requests = []

    def fake_urlopen(req, timeout):
        requests.append((req, timeout))
        if req.full_url.endswith("/models"):
            return _Response({"data": [{"id": "Qwen/Qwen3.8-27B"}]})
        return _Response({"choices": [{"message": {"content": " Montreal "}}]})

    monkeypatch.setattr("generation.generator.request.urlopen", fake_urlopen)
    generator = OpenAICompatibleGenerator(
        "Qwen/Qwen3.8-27B", max_new_tokens=64, timeout=300, seed=0
    )
    assert generator.generate("Where?", [{"text": "In Montreal."}]) == "Montreal"
    payload = json.loads(requests[-1][0].data.decode("utf-8"))
    assert payload["temperature"] == 0
    assert payload["top_p"] == 1
    assert payload["seed"] == 0
    assert payload["max_tokens"] == 64
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def _compatible_params() -> dict:
    return {
        "dataset": "hotpotqa_fullwiki", "split": "validation",
        "max_samples": None, "encoder_model": "BAAI/bge-m3",
        "top_m": 100, "top_k": 5,
        "candidate_pool_policy": "fixed_clean_pool_size_per_query",
        "metric": "cosine", "dup_noise": "light", "dup_target": "mixed",
        "single_subtopic": False, "generator_model": "Qwen/Qwen3.8-27B",
        "generator_backend": "openai-compatible", "generator_revision": "rev",
        "generator_max_new_tokens": 64, "generator_num_beams": 1,
        "generator_temperature": 0, "generator_top_p": 1,
        "generator_seed": 0, "generator_thinking": False,
        "generator_prompt_version": "short_direct_v1", "gen_max_samples": None,
        "seed": 0, "seeds": [0], "rho_grid": [0.0, 1.0],
    }


def test_reuse_requires_fixed_pool_policy_and_records_hash(tmp_path) -> None:
    source = tmp_path / "results_redundancy_gen_per_query.csv"
    source.write_text("rho,seed,qid,Method\n0,0,q,kNN\n", encoding="utf-8")
    params = _compatible_params()
    (tmp_path / "run_params.json").write_text(json.dumps(params), encoding="utf-8")
    result = _validate_reuse_source(str(source), params)
    assert len(result["sha256"]) == 64

    params["candidate_pool_policy"] = None
    (tmp_path / "run_params.json").write_text(json.dumps(params), encoding="utf-8")
    with pytest.raises(SystemExit, match="candidate_pool_policy"):
        _validate_reuse_source(str(source), _compatible_params())


def test_save_csv_replaces_atomically_without_temp_files(tmp_path) -> None:
    path = tmp_path / "checkpoint.csv"
    save_csv([{"qid": "q1", "value": 1}], str(path))
    save_csv([{"qid": "q2", "value": 2}], str(path))
    with path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [{"qid": "q2", "value": "2"}]
    assert list(tmp_path.glob(".checkpoint.csv.*.tmp")) == []


def test_qwen_summary_can_filter_a_frozen_partition(tmp_path) -> None:
    manifest = tmp_path / "split.csv"
    manifest.write_text(
        "dataset,seed,query_id,partition\n"
        "hotpotqa_fullwiki,0,q1,test\n"
        "hotpotqa_fullwiki,0,q2,validation\n"
        "hotpotqa_fullwiki,1,q1,validation\n",
        encoding="utf-8",
    )
    rows = [
        {"seed": "0", "qid": qid, "Method": method}
        for qid in ("q1", "q2") for method in ("kNN", "RNG(0.2)")
    ]
    selected, n_queries = filter_partition(rows, manifest, "test", 0)
    assert n_queries == 1
    assert {row["qid"] for row in selected} == {"q1"}
    assert {row["Method"] for row in selected} == {"kNN", "RNG(0.2)"}
