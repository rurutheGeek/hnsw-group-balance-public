import numpy as np
import pytest

from hnswbal import levels

N, G, M = 5000, 20, 8


@pytest.fixture
def labels():
    return np.random.default_rng(0).integers(0, G, size=N).astype(np.int32)


def test_one_layer_every_group_in_balanced_layer(labels):
    for seed in range(10):
        lv, target = levels.one_layer_random(N, labels, G, M, np.random.default_rng(seed))
        assert target == levels.balanced_layer(N, G, M)
        assert set(labels[lv >= target]) == set(range(G))


def test_all_layer_keeps_layer_sizes_and_divides_equally(labels):
    lv, _ = levels.STRATEGIES["alloc_equal_keep"](N, labels, G, M, np.random.default_rng(0))
    for layer in range(1, lv.max() + 1):
        assert (lv >= layer).sum() == round(N * M ** -layer)
        per_group = np.bincount(labels[lv >= layer], minlength=G)
        assert per_group.max() - per_group.min() <= 1


def test_strategies_are_deterministic(labels):
    for name, fn in levels.STRATEGIES.items():
        a, _ = fn(N, labels, G, M, np.random.default_rng(1))
        b, _ = fn(N, labels, G, M, np.random.default_rng(1))
        assert np.array_equal(a, b), name
