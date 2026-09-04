"""Regression checks for package paths used by installed commands."""

from pathlib import Path

from experiments import evaluate as evaluate_impl
from experiments import evaluate_redundancy as redundancy_impl


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_paths_resolve_from_project_root() -> None:
    assert Path(evaluate_impl.PROJECT_ROOT) == ROOT
    assert Path(redundancy_impl.PROJECT_ROOT) == ROOT
    assert Path(evaluate_impl._DEFAULT_CONFIG) == ROOT / "config.yaml"
