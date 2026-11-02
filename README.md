# hnsw-group-balance

Code for the paper *Improving HNSW Search Efficiency via Group-Aware All-Layer Rebalancing*
(NLPIR 2026).

All-layer rebalancing keeps the number of nodes on each upper layer of HNSW at its expected value
and divides these nodes equally among the groups of the data. Only the layer of each node is
changed; the construction and search algorithms of HNSW are not modified.

## Contents

```
hnswlib/                 hnswlib (nmslib/hnswlib@c1b9b79) with a few additions, see below
src/hnswbal/levels.py    level assignment: original HNSW, one-layer balancing, all-layer rebalancing
src/hnswbal/datasets.py  synthetic data (Separated, Zipfian) and real datasets
src/hnswbal/evaluation.py  index building, measurement, reduction at equal recall
experiments/run_compare.py  runs the comparison
tests/                   pytest
```

Additions to hnswlib (marked with `MODIFICATION START/END` in the C++ code):

- `add_items_with_level(data, ids, levels)`: inserts nodes with given maximum layers
- `get_distance_computations()`, `reset_distance_computations()`: distance computations during search
- `get_max_level()`, `get_level_counts()`: layer information

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install pybind11 numpy
pip install ./hnswlib
pip install -e '.[dev,text]'
pytest -q
```

## Running

```bash
python experiments/run_compare.py --out out/20news --datasets "20 Newsgroups (full)" --m 4 8 16 32
```

This compares the original HNSW, one-layer balancing, and all-layer rebalancing over 30 runs and
writes the reduction in distance computations at equal recall to `out/20news/target_summary.csv`.
Other conditions are listed in `src/hnswbal/evaluation.py` and datasets in `datasets.DATASETS`.

The datasets used in the paper are:

| Paper | `--datasets` |
|---|---|
| 20 Newsgroups, Languages, AG News, DBpedia | `"20 Newsgroups (full)"`, `"Languages (full)"`, `"AG News (full)"`, `"DBpedia (full)"` |
| Separated, 1,000 / 5,000 / 20,000 vectors per group | `"Separated s=2.0 n=1000 order=random"` etc. |
| Separated, standard deviation s | `"Separated s=0.25"` ... `"Separated s=5.0"` |
| Zipfian | `"Zipfian s=1.0"`, `"Zipfian s=2.0"` |

The real datasets are downloaded from scikit-learn and Hugging Face and embedded with
multilingual-e5-small on the first run. The embeddings are cached in `data/`
(or `$HNSWBAL_CACHE`).

## License

hnswlib is distributed under the Apache License 2.0 (`hnswlib/LICENSE`).
The rest of this repository is also distributed under the Apache License 2.0 (`LICENSE`).
