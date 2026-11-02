"""Index building, search measurement, and summary statistics."""

from __future__ import annotations

import time
from dataclasses import dataclass

import hnswlib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from . import levels


@dataclass(frozen=True)
class Condition:
    """One compared method.

    level_strategy=None builds with plain ``add_items`` (hnswlib's own RNG).
    Otherwise levels come from ``levels.STRATEGIES[level_strategy]`` seeded
    with ``index_seed + seed_offset``.
    """

    name: str
    level_strategy: str | None = None
    seed_offset: int = 0


# Seed offsets are the ones used for the paper, so the same seeds give the same levels.
CONDITIONS = [
    Condition("Original"),
    Condition("One-layer balancing", "one_layer_random", 1),
    Condition("All-layer rebalancing", "alloc_equal_keep", 10),
    # Allocation rules: weight size^p (equal p=0, sqrt p=0.5, proportional p=1) x drop / keep layers.
    Condition("Alloc equal, drop layers", "alloc_equal_drop", 9),
    Condition("Alloc sqrt, drop layers", "alloc_sqrt_drop", 11),
    Condition("Alloc sqrt, keep layers", "alloc_sqrt_keep", 12),
    Condition("Alloc proportional, drop layers", "alloc_prop_drop", 13),
    Condition("Alloc proportional, keep layers", "alloc_prop_keep", 14),
]
DEFAULT_CONDITIONS = CONDITIONS[:3]


def ground_truth(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    neighbours = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="cosine", n_jobs=-1)
    neighbours.fit(data)
    return neighbours.kneighbors(queries)[1]


def build_index(data: np.ndarray, labels: np.ndarray, m: int, ef_construction: int, index_seed: int,
                level_strategy: str | None, seed_offset: int):
    """Returns (index, levels or None, timings) where timings has level_assign_s and insert_s."""
    index = hnswlib.Index(space="cosine", dim=data.shape[1])
    index.init_index(max_elements=len(data), ef_construction=ef_construction, M=m, random_seed=index_seed)
    if level_strategy is None:
        t0 = time.perf_counter()
        index.add_items(data)
        return index, None, {"level_assign_s": 0.0, "insert_s": time.perf_counter() - t0}
    t0 = time.perf_counter()
    rng = np.random.default_rng(index_seed + seed_offset)
    node_levels, _ = levels.STRATEGIES[level_strategy](len(data), labels, len(np.unique(labels)), m, rng)
    t1 = time.perf_counter()
    index.add_items_with_level(data, np.arange(len(data)), levels=node_levels)
    return index, node_levels, {"level_assign_s": t1 - t0, "insert_s": time.perf_counter() - t1}


def measure(index, queries: np.ndarray, gt: np.ndarray, k: int, ef_list) -> list[dict]:
    """Single-threaded search per ef; returns recall, mean distance computations and time."""
    rows = []
    for ef in ef_list:
        index.set_ef(ef)
        index.reset_distance_computations()
        started = time.perf_counter()
        result, _ = index.knn_query(queries, k=k, num_threads=1)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        per_query = np.array([len(set(gt[i]).intersection(set(result[i]))) / k for i in range(len(queries))])
        rows.append({
            "ef_search": ef,
            "recall": float(per_query.mean()),
            "avg_dist_comps": float(index.get_distance_computations() / len(queries)),
            "query_time_ms": elapsed_ms,
            "query_time_ms_per_query": elapsed_ms / len(queries),
        })
    return rows


def run_trial(data, labels, queries, gt, *, m: int, ef_construction: int, k: int, ef_list,
              index_seed: int, conditions=DEFAULT_CONDITIONS) -> list[dict]:
    """Evaluate every condition on one index seed."""
    rows = []
    for cond in conditions:
        index, _, timings = build_index(data, labels, m, ef_construction, index_seed, cond.level_strategy, cond.seed_offset)
        for row in measure(index, queries, gt, k, ef_list):
            row.update({"condition": cond.name,
                        "build_level_assign_s": timings["level_assign_s"], "build_insert_s": timings["insert_s"],
                        "build_total_s": timings["level_assign_s"] + timings["insert_s"]})
            rows.append(row)
    return rows


def ci95(values: pd.Series) -> float:
    """1.96 * standard error over trials."""
    values = values.dropna()
    if len(values) < 2:
        return float("nan")
    return float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def interpolate_log_distance(frame: pd.DataFrame, target: float) -> float:
    """Distance computations at a target recall (log-linear interpolation, no extrapolation)."""
    frame = frame.sort_values("recall").drop_duplicates(subset=["recall"], keep="last")
    recalls = frame["recall"].to_numpy(dtype=float)
    distances = frame["avg_dist_comps"].to_numpy(dtype=float)
    if len(recalls) == 0 or target < recalls.min() or target > recalls.max():
        return float("nan")
    return float(10 ** np.interp(target, recalls, np.log10(distances)))


def summarize_targets(result: pd.DataFrame, targets, baseline: str = "Original",
                      keys=("dataset", "M")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-run and aggregated reduction (%) of distance computations vs baseline at target recalls."""
    keys = [k for k in keys if k in result.columns]
    rows = []
    for group_key, run_frame in result.groupby([*keys, "run_id"]):
        base = run_frame[run_frame["condition"] == baseline]
        for target in targets:
            base_d = interpolate_log_distance(base, target)
            for condition, frame in run_frame.groupby("condition"):
                d = interpolate_log_distance(frame, target)
                reduction = (base_d - d) / base_d * 100.0 if np.isfinite(base_d) and np.isfinite(d) else np.nan
                rows.append({**dict(zip([*keys, "run_id"], group_key)), "target_recall": target,
                             "condition": condition, "distance_at_target": d,
                             "reduction_vs_original_percent": reduction, "target_reached": bool(np.isfinite(d))})
    per_run = pd.DataFrame(rows)
    summary = per_run.groupby([*keys, "target_recall", "condition"]).agg(
        total_runs=("run_id", "size"),
        reached_runs=("target_reached", "sum"),
        reduction_mean_percent=("reduction_vs_original_percent", "mean"),
        reduction_ci95_percent=("reduction_vs_original_percent", ci95),
    ).reset_index()
    return per_run, summary
