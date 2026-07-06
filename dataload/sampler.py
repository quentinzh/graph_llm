"""Samplers that reduce padding waste in variable-length batches."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class LengthBucketSampler(Sampler[int]):
    """Yield indices grouped by similar lengths to reduce batch padding."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(lengths) == 0:
            raise ValueError("lengths must be non-empty")
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.lengths)))
        if not self.shuffle:
            yield from indices
            return

        rng = random.Random(self.seed + self.epoch)
        indices.sort(key=lambda idx: (self.lengths[idx], idx))
        batches = [
            indices[start:start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        rng.shuffle(batches)
        for batch in batches:
            batch_indices = list(batch)
            rng.shuffle(batch_indices)
            yield from batch_indices

    def __len__(self) -> int:
        return len(self.lengths)
