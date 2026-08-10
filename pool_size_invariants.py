"""Candidate-pool size invariants for redundancy and chunking sweeps."""

from __future__ import annotations

from typing import Sequence, Sized


def clean_pool_target(original_size: int, top_m: int, top_k: int) -> int:
    """Return the per-query candidate count frozen across sweep levels.

    ``top_m`` is an upper bound because attached benchmark pools can contain
    fewer than ``top_m`` documents. The clean pool therefore defines the only
    feasible fixed target for that query.
    """
    if original_size <= 0:
        raise AssertionError("original candidate pool must be non-empty")
    if top_m <= 0 or top_k <= 0:
        raise AssertionError("top_m and top_k must be positive")
    target = min(original_size, top_m)
    if target < top_k:
        raise AssertionError(
            f"candidate target {target} is smaller than requested top_k {top_k}"
        )
    return target


def assert_transformed_pool(
    *, query_id: str, original_size: int, transformed_size: int, target: int,
    level_name: str, level: float, seed: int,
) -> None:
    """Require a transformed pool to contain the frozen clean-pool target."""
    if transformed_size < target:
        raise AssertionError(
            f"{query_id}: transformed pool has {transformed_size} candidates, "
            f"below fixed target {target} at {level_name}={level}, seed={seed} "
            f"(original={original_size})"
        )


def assert_encoded_pool(
    *, query_id: str, transformed_size: int, encoded_size: int,
    level_name: str, level: float, seed: int,
) -> None:
    """Require one embedding row for every transformed candidate instance."""
    if encoded_size != transformed_size:
        raise AssertionError(
            f"{query_id}: encoded {encoded_size} candidates but transformed pool "
            f"contains {transformed_size} at {level_name}={level}, seed={seed}"
        )


def assert_candidate_pool(
    *, query_id: str, candidate_size: int, target: int,
    level_name: str, level: float, seed: int,
) -> None:
    """Require the post-truncation candidate pool to equal its frozen target."""
    if candidate_size != target:
        raise AssertionError(
            f"{query_id}: post-truncation pool has {candidate_size} candidates, "
            f"expected fixed target {target} at {level_name}={level}, seed={seed}"
        )


def assert_pool_collection(
    *, query_ids: Sequence[str], pools: Sequence[Sized],
    original_sizes: Sequence[int], targets: Sequence[int],
    level_name: str, level: float, seed: int,
) -> None:
    """Validate pool count and every transformed pool before encoding."""
    expected = len(query_ids)
    if not (len(pools) == len(original_sizes) == len(targets) == expected):
        raise AssertionError(
            f"pool builder returned {len(pools)} pools for {expected} queries at "
            f"{level_name}={level}, seed={seed}"
        )
    for query_id, pool, original_size, target in zip(
        query_ids, pools, original_sizes, targets
    ):
        assert_transformed_pool(
            query_id=query_id,
            original_size=original_size,
            transformed_size=len(pool),
            target=target,
            level_name=level_name,
            level=level,
            seed=seed,
        )
