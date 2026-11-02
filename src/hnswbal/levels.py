"""HNSW level-assignment strategies used in the paper.

Each strategy returns ``(levels, balanced_layer)`` where ``levels[i]`` is the
maximum layer of node ``i``; pass it to ``Index.add_items_with_level``.

- ``original``: Eq. (1) of HNSW, l = floor(-ln(U) * mL)
- ``one_layer_random``: one-layer balancing (our previous work)
- ``alloc_equal_keep``: all-layer rebalancing (the proposed method)
- ``alloc_{equal,sqrt,prop}_{keep,drop}``: allocation rules compared in the paper
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

LevelStrategy = Callable[[int, np.ndarray, int, int, np.random.Generator], "tuple[np.ndarray, int]"]


def level_mult(m: int) -> float:
    """mL = 1 / ln(M), the level-generation scale of Eq. (1)."""
    return 1.0 / np.log(m)


def sample_levels(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Eq. (1): l = floor(-ln(U) * mL)."""
    return (-np.log(rng.uniform(1e-10, 1.0, size=n)) * level_mult(m)).astype(np.int32)


def balanced_layer(num_elements: int, num_groups: int, m: int) -> int:
    """Highest layer whose expected node count N * M^-l still reaches num_groups."""
    return int(np.log(num_elements / num_groups) * level_mult(m))


def original(num_elements: int, labels: np.ndarray, num_groups: int, m: int,
             rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Plain Eq. (1) levels, for building an Original-equivalent index via add_items_with_level."""
    return sample_levels(num_elements, m, rng), 0


def one_layer_random(num_elements: int, labels: np.ndarray, num_groups: int, m: int,
                     rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """One-layer balancing (previous work): one representative per group in the balanced layer.

    Missing groups get a random node promoted; the same number of surplus nodes
    (from groups that keep >=1 representative) are re-drawn from Eq. (1).
    """
    target_level = balanced_layer(num_elements, num_groups, m)
    levels = sample_levels(num_elements, m, rng)
    frame = pd.DataFrame({"idx": np.arange(num_elements), "label": labels, "level": levels})
    present_groups = set(frame.loc[frame["level"] >= target_level, "label"].unique().tolist())
    missing_groups = set(range(num_groups)) - present_groups

    promoted_count = 0
    for group in sorted(missing_groups):
        group_nodes = frame.loc[frame["label"] == group, "idx"].to_numpy()
        if len(group_nodes) == 0:
            continue
        representative = int(rng.choice(group_nodes))
        levels[representative] = target_level
        frame.loc[representative, "level"] = target_level
        promoted_count += 1

    candidates: dict[int, list[int]] = {}
    for group in sorted(present_groups):
        nodes = frame[(frame["label"] == group) & (frame["level"] >= target_level)].sort_values("level", ascending=False)
        if len(nodes) > 1:
            candidates[group] = nodes["idx"].tolist()[1:]

    selected: list[int] = []
    for _ in range(promoted_count):
        available = [node for nodes in candidates.values() for node in nodes]
        if not available:
            break
        node = int(rng.choice(available))
        selected.append(node)
        for nodes in candidates.values():
            if node in nodes:
                nodes.remove(node)
                break

    levels[selected] = (-np.log(rng.uniform(1e-10, 1.0, size=len(selected))) * level_mult(m)).astype(np.int32)
    return levels, target_level


def _largest_remainder(total: int, weights: np.ndarray, caps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Split ``total`` into integers proportional to ``weights`` without exceeding ``caps``."""
    share = total * weights / weights.sum()
    out = np.minimum(np.floor(share).astype(int), caps)
    rest = total - out.sum()
    order = np.lexsort((rng.random(len(share)), -(share - np.floor(share))))   # largest remainder, random ties
    while rest > 0:
        moved = False
        for g in order:
            if rest == 0:
                break
            if out[g] < caps[g]:
                out[g] += 1; rest -= 1; moved = True
        if not moved:
            break
    return out


def all_layer_weighted(num_elements: int, labels: np.ndarray, num_groups: int, m: int,
                       rng: np.random.Generator, power: float = 0.0, keep_layers: bool = True) -> tuple[np.ndarray, int]:
    """General all-layer rebalancing for comparing allocation rules.

    Layer l >= 1 holds round(N * M^-l) nodes in total, split among groups with weights
    size^power (0 = equal, 0.5 = square root, 1 = proportional; largest remainder, a group
    never has more nodes on a layer than on the layer below). keep_layers=False drops every
    layer whose total is smaller than the number of groups (fewer than one node per group).
    Within a group the promoted nodes are random.
    """
    groups, sizes = np.unique(labels, return_counts=True)
    weights = sizes.astype(float) ** power
    counts = [sizes.copy()]
    level = 1
    while True:
        total = int(round(num_elements * m ** -level))
        if total < 1 or (not keep_layers and total < len(groups)):
            break
        counts.append(_largest_remainder(total, weights, counts[-1], rng))
        level += 1
    counts = np.stack(counts)
    levels = np.zeros(num_elements, dtype=np.int32)
    for j, g in enumerate(groups):
        members = rng.permutation(np.flatnonzero(labels == g))
        rank = np.arange(len(members))
        levels[members] = (rank[:, None] < counts[1:, j][None, :]).sum(axis=1)
    return levels, 0


def _weighted(power: float, keep_layers: bool):
    def strategy(num_elements, labels, num_groups, m, rng):
        return all_layer_weighted(num_elements, labels, num_groups, m, rng, power, keep_layers)
    return strategy


STRATEGIES: dict[str, LevelStrategy] = {
    "original": original,
    "one_layer_random": one_layer_random,
    "alloc_equal_drop": _weighted(0.0, False),
    "alloc_equal_keep": _weighted(0.0, True),
    "alloc_sqrt_drop": _weighted(0.5, False),
    "alloc_sqrt_keep": _weighted(0.5, True),
    "alloc_prop_drop": _weighted(1.0, False),
    "alloc_prop_keep": _weighted(1.0, True),
}
