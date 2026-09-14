"""Regression checks for package paths used by installed commands."""

from pathlib import Path
import sys

from experiments import evaluate_beir
from experiments import evaluate_cross_encoder
from experiments import evaluate as evaluate_impl
from experiments import evaluate_redundancy as redundancy_impl
from experiments import learn_alpha
from experiments import retrieve_passages
from ftrb.run_utils import DEFAULT_DEVICE


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_paths_resolve_from_project_root() -> None:
    assert Path(evaluate_impl.PROJECT_ROOT) == ROOT
    assert Path(redundancy_impl.PROJECT_ROOT) == ROOT
    assert Path(evaluate_impl._DEFAULT_CONFIG) == ROOT / "config.yaml"


def test_experiment_commands_default_to_gpu(monkeypatch) -> None:
    parsers = (
        evaluate_impl._parse_args,
        evaluate_beir._parse_args,
        evaluate_cross_encoder._parse_args,
        redundancy_impl._parse_args,
        learn_alpha._parse_args,
        retrieve_passages._parse_args,
    )
    monkeypatch.setattr(sys, "argv", ["command"])
    assert DEFAULT_DEVICE == "cuda"
    assert all(parser().device == DEFAULT_DEVICE for parser in parsers)


def test_evaluate_preserves_config_device_without_cli_override(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["ftrb-evaluate"])
    args = evaluate_impl._parse_args()

    cfg = evaluate_impl._apply_cli_overrides({"device": "cpu"}, args)

    assert cfg["device"] == "cpu"


def test_evaluate_explicit_cli_device_overrides_config(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["ftrb-evaluate", "--device", "cuda:1"])
    args = evaluate_impl._parse_args()

    cfg = evaluate_impl._apply_cli_overrides({"device": "cpu"}, args)

    assert cfg["device"] == "cuda:1"
