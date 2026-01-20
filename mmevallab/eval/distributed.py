"""Distributed/sharded execution utilities."""

from __future__ import annotations

from typing import Iterator, TypeVar

T = TypeVar("T")


def shard_examples(
    examples: list[T],
    shard_id: int,
    num_shards: int,
) -> Iterator[T]:
    """Yield examples for this shard."""
    for i, ex in enumerate(examples):
        if i % num_shards == shard_id:
            yield ex


def get_shard_info() -> tuple[int, int]:
    """Get shard info from environment (RANK, WORLD_SIZE)."""
    import os

    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    return rank, world
