"""Smoke tests for the modified hnswlib bindings."""
import hnswlib
import numpy as np


def _index(n=2000, dim=16, m=8):
    data = np.random.default_rng(0).normal(size=(n, dim)).astype(np.float32)
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=n, ef_construction=100, M=m, random_seed=1)
    return idx, data


def test_add_items_with_level_sets_levels():
    idx, data = _index()
    lv = np.zeros(len(data), dtype=np.int32)
    lv[:5] = 3
    idx.add_items_with_level(data, np.arange(len(data)), levels=lv)
    assert idx.get_max_level() == 3


def test_distance_counter():
    idx, data = _index()
    idx.add_items(data)
    idx.set_ef(50)
    idx.reset_distance_computations()
    labels, _ = idx.knn_query(data[:10], k=5, num_threads=1)
    assert (labels[:, 0] == np.arange(10)).mean() > 0.8
    assert idx.get_distance_computations() > 0
