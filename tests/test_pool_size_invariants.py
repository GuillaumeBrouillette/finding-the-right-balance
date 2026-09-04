"""Tests for fixed candidate-pool sizes across redundancy levels."""

import pytest

from ftrb.pool_size_invariants import (
    assert_candidate_pool,
    assert_encoded_pool,
    assert_pool_collection,
    clean_pool_target,
)


def test_clean_pool_target_uses_top_m_as_upper_bound() -> None:
    assert clean_pool_target(200, top_m=100, top_k=10) == 100
    assert clean_pool_target(10, top_m=100, top_k=5) == 10


def test_injection_levels_keep_the_clean_candidate_target() -> None:
    target = clean_pool_target(10, top_m=100, top_k=5)
    for rho, transformed_size in [(0.0, 10), (0.5, 15), (1.0, 20)]:
        assert_pool_collection(
            query_ids=["q"], pools=[[None] * transformed_size],
            original_sizes=[10], targets=[target],
            level_name="rho", level=rho, seed=0,
        )
        assert_candidate_pool(
            query_id="q", candidate_size=10, target=target,
            level_name="rho", level=rho, seed=0,
        )


def test_chunk_levels_keep_the_clean_candidate_target() -> None:
    target = clean_pool_target(20, top_m=100, top_k=5)
    for overlap, transformed_size in [(0.0, 35), (0.5, 60), (0.75, 110)]:
        assert_pool_collection(
            query_ids=["q"], pools=[[None] * transformed_size],
            original_sizes=[20], targets=[target],
            level_name="overlap", level=overlap, seed=2,
        )
        assert_candidate_pool(
            query_id="q", candidate_size=20, target=target,
            level_name="overlap", level=overlap, seed=2,
        )


def test_assertions_reject_size_drift() -> None:
    with pytest.raises(AssertionError, match="below fixed target"):
        assert_pool_collection(
            query_ids=["q"], pools=[[None] * 9], original_sizes=[10],
            targets=[10], level_name="rho", level=1.0, seed=0,
        )
    with pytest.raises(AssertionError, match="expected fixed target"):
        assert_candidate_pool(
            query_id="q", candidate_size=20, target=10,
            level_name="rho", level=1.0, seed=0,
        )
    with pytest.raises(AssertionError, match="encoded 9"):
        assert_encoded_pool(
            query_id="q", transformed_size=10, encoded_size=9,
            level_name="rho", level=0.5, seed=1,
        )
