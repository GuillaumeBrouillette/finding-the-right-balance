"""Checks for the committed frozen validation/test query manifests."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reproducibility.validate_split_manifests import validate
from reproducibility.validate_source_group_manifests import validate as validate_sources
from reproducibility.validate_execution_manifest import validate as validate_execution
from reproducibility.validate_pool_size_manifest import validate as validate_pool_sizes


def test_frozen_split_manifests() -> None:
    validate()


def test_frozen_source_group_manifests() -> None:
    validate_sources()


def test_frozen_execution_manifest() -> None:
    validate_execution()


def test_pool_size_manifest() -> None:
    validate_pool_sizes()
