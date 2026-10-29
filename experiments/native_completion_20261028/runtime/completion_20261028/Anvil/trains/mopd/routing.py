"""Deterministic, distributed-safe family batches for routed distillation."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable, Iterator, Mapping, Sequence

import torch.distributed as dist

GSA_FAMILIES = ("G", "S", "A")


def dataset_families(dataset: Sequence[Mapping[str, object]]) -> list[str]:
    """Read declared families without inferring them from prompts or answers."""
    try:
        values = dataset["gsa_family"]
    except (KeyError, TypeError):
        values = [row.get("gsa_family") for row in dataset]
    families = [str(value) if value is not None else "" for value in values]
    invalid = sorted(set(families) - set(GSA_FAMILIES))
    if invalid:
        raise ValueError(f"invalid or missing gsa_family values: {invalid}")
    return families


class FamilyBalancedBatchSampler:
    """Yield full local batches from one G/S/A family at a time.

    Each global batch has ``world_size * batch_size`` examples from one family.
    The three family streams are oversampled to the largest stream, producing
    matched exposure without changing the dataset on disk. All ranks use the
    same shuffled global blocks and take disjoint contiguous local slices.
    """

    def __init__(
        self,
        families: Iterable[str],
        batch_size: int,
        *,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
        required_families: Sequence[str] = GSA_FAMILIES,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self._rank = rank
        self._world_size = world_size
        self.families = tuple(required_families)
        if not self.families or len(set(self.families)) != len(self.families) or not set(self.families) <= set(GSA_FAMILIES):
            raise ValueError('invalid declared sampler families')
        self.buckets: dict[str, list[int]] = defaultdict(list)
        for index, family in enumerate(families):
            if family not in self.families:
                raise ValueError(f"sample {index} has invalid gsa_family={family!r}")
            self.buckets[family].append(index)
        missing = [family for family in self.families if not self.buckets[family]]
        if missing:
            raise ValueError(f"routed MOPD requires all G/S/A families; missing {missing}")

    @property
    def world_size(self) -> int:
        if self._world_size is not None:
            return self._world_size
        return dist.get_world_size() if dist.is_initialized() else 1

    @property
    def rank(self) -> int:
        if self._rank is not None:
            return self._rank
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def global_batch_size(self) -> int:
        return self.batch_size * self.world_size

    def _blocks_per_family(self) -> int:
        return math.ceil(max(map(len, self.buckets.values())) / self.global_batch_size)

    def __len__(self) -> int:
        return len(self.families) * self._blocks_per_family()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _take_cyclic(indices: list[int], start: int, count: int) -> list[int]:
        size = len(indices)
        return [indices[(start + offset) % size] for offset in range(count)]

    def __iter__(self) -> Iterator[list[int]]:
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} outside world_size {self.world_size}")
        global_batch = self.global_batch_size
        too_small = [family for family, rows in self.buckets.items() if len(rows) < global_batch]
        if too_small:
            raise ValueError(
                "each family needs at least one global batch "
                f"({global_batch} rows); too small: {too_small}"
            )

        rng = random.Random(self.seed + self.epoch)
        streams = {family: list(self.buckets[family]) for family in self.families}
        for rows in streams.values():
            rng.shuffle(rows)

        starts = {family: 0 for family in self.families}
        family_order = list(self.families)
        family_order = family_order[self.epoch % len(family_order):] + family_order[:self.epoch % len(family_order)]
        for _ in range(self._blocks_per_family()):
            for family in family_order:
                rows = streams[family]
                block = self._take_cyclic(rows, starts[family], global_batch)
                starts[family] = (starts[family] + global_batch) % len(rows)
                begin = self.rank * self.batch_size
                yield block[begin:begin + self.batch_size]
