"""Compare level-assignment strategies against the original HNSW.

Example (20 Newsgroups, M=8, original HNSW / one-layer balancing / all-layer rebalancing):
    python experiments/run_compare.py --out out/20news_m8 --datasets "20 Newsgroups (full)" --m 8

Writes <out>/{config.json, results.csv, target_results_per_run.csv, target_summary.csv}.
Seeds: query_seed = seed_base + dataset_pos*1000 + 17, index_seed = seed_base + dataset_pos*100000 + run
(dataset_pos is the position in --datasets).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hnswbal import datasets
from hnswbal.evaluation import CONDITIONS, DEFAULT_CONDITIONS, ground_truth, run_trial, summarize_targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="output folder")
    p.add_argument("--datasets", nargs="+", default=["20 Newsgroups (full)"], choices=datasets.DATASETS)
    p.add_argument("--m", type=int, nargs="+", default=[16])
    p.add_argument("--ef", type=int, nargs="+", default=[10, 20, 50, 100, 200, 500, 1000])
    p.add_argument("--ef-construction", type=int, default=100)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--num-runs", type=int, default=30)
    p.add_argument("--query-num", type=int, default=500)
    p.add_argument("--seed-base", type=int, default=20260726)
    p.add_argument("--conditions", nargs="+", default=[c.name for c in DEFAULT_CONDITIONS],
                   choices=[c.name for c in CONDITIONS])
    p.add_argument("--targets", type=float, nargs="+", default=[0.90, 0.95])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True,
                        handlers=[logging.FileHandler(out / "run.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    conditions = [c for c in CONDITIONS if c.name in args.conditions]
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["conditions_resolved"] = [c.__dict__ for c in conditions]
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows = []
    for pos, name in enumerate(args.datasets):
        query_seed = args.seed_base + pos * 1000 + 17
        logging.info("dataset %s", name)
        data, labels, queries, query_labels, meta = datasets.load_dataset(name, args.query_num, query_seed)
        gt = ground_truth(data, queries, args.k)
        for m in args.m:
            for run_id in range(args.num_runs):
                index_seed = args.seed_base + pos * 100_000 + run_id
                logging.info("%s M=%d run %d/%d", name, m, run_id + 1, args.num_runs)
                for row in run_trial(data, labels, queries, gt, m=m, ef_construction=args.ef_construction,
                                     k=args.k, ef_list=args.ef, index_seed=index_seed, conditions=conditions):
                    row.update({"dataset": name, "M": m, "run_id": run_id, "index_seed": index_seed,
                                "query_seed": query_seed, "num_elements": len(data), "num_queries": len(queries)})
                    rows.append(row)
        pd.DataFrame(rows).to_csv(out / "results.csv", index=False)  # checkpoint per dataset

    result = pd.DataFrame(rows)
    per_run, summary = summarize_targets(result, args.targets)
    per_run.to_csv(out / "target_results_per_run.csv", index=False)
    summary.to_csv(out / "target_summary.csv", index=False)
    logging.info("done: %d rows -> %s", len(result), out)
    with pd.option_context("display.width", 200):
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}" if np.isfinite(v) else "nan"))


if __name__ == "__main__":
    main()
