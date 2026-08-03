from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from .models import Sample


@dataclass(slots=True)
class SplitUnit:
    group_id: str
    samples: list[Sample]
    signature: str

    @property
    def size(self) -> int:
        return len(self.samples)


def _unit_signature(samples: list[Sample]) -> str:
    signatures = {sample.class_signature for sample in samples}
    if len(signatures) == 1:
        return next(iter(signatures))
    # A near-duplicate cluster should normally share labels. If it does not,
    # keep the whole cluster together and classify it by union of classes.
    has_fire = any(sample.fire_boxes > 0 for sample in samples)
    has_smoke = any(sample.smoke_boxes > 0 for sample in samples)
    if has_fire and has_smoke:
        return "fire_and_smoke"
    if has_fire:
        return "fire_only"
    if has_smoke:
        return "smoke_only"
    return "negative"


def assign_splits(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, int]:
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.duplicate_group].append(sample)

    by_signature: dict[str, list[SplitUnit]] = defaultdict(list)
    for group_id, group_samples in grouped.items():
        by_signature[_unit_signature(group_samples)].append(
            SplitUnit(group_id, group_samples, _unit_signature(group_samples))
        )

    rng = random.Random(seed)
    split_names = ("train", "val", "test")
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    counts = Counter()

    for signature, units in sorted(by_signature.items()):
        rng.shuffle(units)
        units.sort(key=lambda unit: unit.size, reverse=True)
        total_images = sum(unit.size for unit in units)
        targets = {name: total_images * ratios[name] for name in split_names}
        signature_counts = Counter()

        for unit in units:
            # Choose the split furthest below its target. This keeps duplicate
            # groups intact while closely approximating each requested ratio.
            def priority(split_name: str) -> tuple[float, float, float]:
                absolute_deficit = targets[split_name] - signature_counts[split_name]
                relative_deficit = absolute_deficit / max(targets[split_name], 1e-9)
                return absolute_deficit, relative_deficit, rng.random()

            chosen = max(split_names, key=priority)
            for sample in unit.samples:
                sample.split = chosen
            signature_counts[chosen] += unit.size
            counts[chosen] += unit.size

    return {name: counts[name] for name in split_names}
