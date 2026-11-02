"""Datasets: synthetic (Normal / Zipfian / Concentrated / Hollowed) and 20 Newsgroups.

Ported verbatim from experiments/legacy/20260727_independent_query_4dataset_30runs.
Each loader returns (train, train_labels, queries, query_labels, metadata);
vectors are L2-normalized float32.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups, make_blobs

DIM = 384
N_CLUSTERS = 20
BASE_COUNT = 100
STD = 0.01
LOW = -0.1
HIGH = 0.1
NUM_ELEMENTS_SYNTH = 50_000

# "20 Newsgroups" is the paper setting (multilingual-e5-small). The MiniLM variant
# only exists to reproduce the 2026-07 legacy experiments.
DATASETS = ["Normal", "Zipfian s=1.0", "Zipfian s=2.0", "Concentrated", "Hollowed", "20 Newsgroups",
            "20 Newsgroups (train/test split)", "20 Newsgroups (MiniLM, legacy)"]
DATASETS += ["AG News", "DBpedia", "Languages"]
# Natural sizes (group sizes not equalised), from 2026-09-28.
DATASETS += ["20 Newsgroups (full)", "AG News (full)", "DBpedia (full)", "Languages (full)"]
DATASETS += [f"Separated s={v}" for v in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]]
DATASETS += [f"Separated s=2.0 n={n}" for n in [1000, 5000, 20000]]   # indexed vectors per group
DATASETS += [f"Separated s=2.0 g={g}" for g in [4, 10, 20, 50, 100]]    # number of groups
DATASETS += ["Separated s=2.0 order=random"]                            # insertion order check
DATASETS += [f"Separated s=2.0 n={n} order=random" for n in [1000, 5000, 20000]]   # sizes, random order
DATASETS += [f"Separated s={v} data={d}" for v in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0] for d in (1, 2, 3)]   # other random data


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-10, None)


def zipf_counts(total: int, exponent: float, base: int = BASE_COUNT) -> np.ndarray:
    ranks = np.arange(1, N_CLUSTERS + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, exponent)
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(42)
    return rng.multinomial(total - N_CLUSTERS * base, probabilities) + base


def sample_labels_from_counts(rng: np.random.Generator, counts: np.ndarray, size: int) -> np.ndarray:
    probabilities = counts.astype(np.float64) / counts.sum()
    return rng.choice(len(counts), size=size, replace=True, p=probabilities).astype(np.int32)


def generate_normal_data(query_num: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    train_rng = np.random.default_rng(seed)
    query_rng = np.random.default_rng(seed + 1)
    train = normalize_rows(train_rng.normal(0.0, 1.0, size=(NUM_ELEMENTS_SYNTH, DIM)).astype(np.float32))
    queries = normalize_rows(query_rng.normal(0.0, 1.0, size=(query_num, DIM)).astype(np.float32))
    train_labels = train_rng.integers(0, N_CLUSTERS, size=len(train), dtype=np.int32)
    counts = np.bincount(train_labels, minlength=N_CLUSTERS)
    query_labels = sample_labels_from_counts(query_rng, counts, query_num)
    metadata = {
        "dataset_description": "L2-normalized standard normal vectors; labels are random negative-control groups.",
        "query_description": "Independent standard normal vectors; group labels are sampled proportionally for EP negative control.",
        "query_label_mode": "synthetic_random_negative_control",
    }
    return train, train_labels, queries, query_labels, metadata


def generate_zipfian_data(exponent: float, query_num: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    counts = zipf_counts(NUM_ELEMENTS_SYNTH, exponent)
    center_rng = np.random.default_rng(10_000 + int(exponent * 100))
    centers = center_rng.uniform(low=LOW, high=HIGH, size=(N_CLUSTERS, DIM))
    train, labels = make_blobs(
        n_samples=counts,
        n_features=DIM,
        centers=centers,
        cluster_std=STD,
        shuffle=True,
        random_state=42,
    )
    query_rng = np.random.default_rng(seed)
    query_labels = sample_labels_from_counts(query_rng, counts, query_num)
    queries = centers[query_labels] + query_rng.normal(0.0, STD, size=(query_num, DIM))
    metadata = {
        "dataset_description": f"20-cluster Zipfian synthetic data, exponent s={exponent}.",
        "query_description": "Independent Gaussian samples around the same cluster centers with population-proportional labels.",
        "query_label_mode": "known_synthetic_cluster_label",
        "zipf_exponent": exponent,
    }
    return normalize_rows(train.astype(np.float32)), np.asarray(labels, dtype=np.int32), normalize_rows(queries.astype(np.float32)), query_labels, metadata


def generate_skewed_data(kind: str, query_num: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    counts = zipf_counts(NUM_ELEMENTS_SYNTH, 1.0)
    center_rng = np.random.default_rng(20_000)
    centers = center_rng.uniform(low=LOW, high=HIGH, size=(N_CLUSTERS, DIM))
    base_train, labels = make_blobs(
        n_samples=counts,
        n_features=DIM,
        centers=centers,
        cluster_std=STD,
        shuffle=True,
        random_state=42,
    )
    base_train = base_train.astype(np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    query_rng = np.random.default_rng(seed)
    query_labels = sample_labels_from_counts(query_rng, counts, query_num)
    base_queries = centers[query_labels] + query_rng.normal(0.0, STD, size=(query_num, DIM))

    if kind == "concentrated":
        train = np.empty_like(base_train)
        for idx, label in enumerate(labels):
            center = centers[label]
            train[idx] = center + 0.1 * (base_train[idx] - center)
        queries = np.empty_like(base_queries)
        for idx, label in enumerate(query_labels):
            center = centers[label]
            queries[idx] = center + 0.1 * (base_queries[idx] - center)
        description = "Cluster-centered concentration with alpha=0.1."
    elif kind == "hollowed":
        train = _hollow_vectors(base_train, labels, centers, np.random.default_rng(30_000))
        queries = _hollow_vectors(base_queries, query_labels, centers, query_rng)
        description = "Cluster hollowing onto a radius STD*2 shell with small Gaussian noise."
    else:
        raise ValueError(f"Unknown skewed dataset kind: {kind}")

    metadata = {
        "dataset_description": description,
        "query_description": "Independent transformed samples generated from the same cluster distribution as the indexed data.",
        "query_label_mode": "known_synthetic_cluster_label",
        "base_distribution": "Zipfian s=1.0",
    }
    return normalize_rows(train), labels, normalize_rows(queries), query_labels, metadata


def _hollow_vectors(values: np.ndarray, labels: np.ndarray, centers: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = np.empty_like(values)
    for idx, label in enumerate(labels):
        center = centers[label]
        diff = values[idx] - center
        distance = np.linalg.norm(diff)
        if distance > 0:
            result[idx] = center + (diff / distance) * (STD * 2.0) + rng.normal(0.0, STD * 0.1, values.shape[1])
        else:
            result[idx] = values[idx]
    return result.astype(np.float32)


def load_20news_data(query_num: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Legacy (2026-07) setting: all-MiniLM-L6-v2 on whole documents. Not the paper setting."""
    logging.info("20 Newsgroupsのtrain/testを読み込み、独立testクエリを生成します。")
    train_set = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    test_set = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))

    train_indices = [i for i, text in enumerate(train_set.data) if text.strip()]
    test_indices = [i for i, text in enumerate(test_set.data) if text.strip()]
    train_texts = [train_set.data[i] for i in train_indices]
    test_texts = [test_set.data[i] for i in test_indices]
    train_labels = np.asarray(train_set.target[train_indices], dtype=np.int32)
    test_labels = np.asarray(test_set.target[test_indices], dtype=np.int32)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    train_vectors = model.encode(train_texts, show_progress_bar=False).astype(np.float32)
    test_vectors = model.encode(test_texts, show_progress_bar=False).astype(np.float32)
    query_rng = np.random.default_rng(seed)
    query_num = min(query_num, len(test_vectors))
    query_indices = query_rng.choice(len(test_vectors), size=query_num, replace=False)
    metadata = {
        "dataset_description": "20 Newsgroups train documents indexed and independent test documents queried.",
        "query_description": "Independent test split queries; category labels are used only for the EP oracle condition.",
        "query_label_mode": "known_20news_category_oracle",
        "train_count": len(train_vectors),
        "test_count": len(test_vectors),
    }
    return normalize_rows(train_vectors), train_labels, normalize_rows(test_vectors[query_indices]), test_labels[query_indices], metadata


E5_MODEL = "intfloat/multilingual-e5-small"
E5_DOCS_PER_GROUP = 1100
E5_QUERIES_PER_GROUP = 50
E5_CHUNK_SIZE = 500
E5_CHUNK_OVERLAP = 50
E5_DATA_SEED = 0
CACHE_DIR = Path(os.environ.get("HNSWBAL_CACHE", Path(__file__).resolve().parents[2] / "data"))


def load_20news_e5(query_num: int, seed: int, split: str = "paper") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Paper setting (Table 1): 1100 chunks per group, 1050 indexed and 50 used as queries.

    multilingual-e5-small, 500-char chunks as in experiments/legacy/20260330_*.
    Queries and indexed chunks come from disjoint documents, so a query never
    overlaps an indexed chunk. The sample is fixed by E5_DATA_SEED, and all
    20 x 50 queries are used, so ``query_num``/``seed`` are ignored.
    Embeddings are cached under CACHE_DIR (override with $HNSWBAL_CACHE).

    split="train_test" reproduces the paper-era code instead: 1100 chunks per
    group from the train subset and 50 from the test subset.
    """
    prefix = {"paper": "20news_e5small_split", "train_test": "20news_e5small_traintest"}[split]
    cache = CACHE_DIR / f"{prefix}_c{E5_CHUNK_SIZE}_n{E5_DOCS_PER_GROUP}_q{E5_QUERIES_PER_GROUP}_s{E5_DATA_SEED}.npz"
    if cache.exists():
        z = np.load(cache)
        train, train_labels, queries, query_labels = z["train"], z["train_labels"], z["queries"], z["query_labels"]
    else:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from sentence_transformers import SentenceTransformer

        logging.info("20 Newsgroups を %s で埋め込みます（初回のみ、結果は %s にキャッシュ）", E5_MODEL, cache)
        splitter = RecursiveCharacterTextSplitter(chunk_size=E5_CHUNK_SIZE, chunk_overlap=E5_CHUNK_OVERLAP)
        rng = random.Random(E5_DATA_SEED)
        categories = fetch_20newsgroups(subset="train").target_names
        texts = {"train": [], "test": []}
        labels = {"train": [], "test": []}
        for group, category in enumerate(categories):
            if split == "train_test":
                for subset, per_group in (("train", E5_DOCS_PER_GROUP), ("test", E5_QUERIES_PER_GROUP)):
                    docs = fetch_20newsgroups(subset=subset, categories=[category], remove=("headers", "footers", "quotes")).data
                    chunks = [c for t in docs if len(t.strip()) > 50 for c in splitter.split_text(t)]
                    sampled = rng.sample(chunks, per_group) if len(chunks) >= per_group else chunks
                    texts[subset].extend(sampled)
                    labels[subset].extend([group] * len(sampled))
                continue
            docs = fetch_20newsgroups(subset="all", categories=[category], remove=("headers", "footers", "quotes")).data
            docs = [t for t in docs if len(t.strip()) > 50]
            rng.shuffle(docs)
            # Queries: one chunk from each of the first E5_QUERIES_PER_GROUP documents.
            # Index: chunks from the remaining documents only (no query/index overlap).
            query_docs, index_docs = docs[:E5_QUERIES_PER_GROUP], docs[E5_QUERIES_PER_GROUP:]
            query_chunks = [rng.choice(splitter.split_text(t)) for t in query_docs]
            index_chunks = [c for t in index_docs for c in splitter.split_text(t)]
            index_chunks = rng.sample(index_chunks, E5_DOCS_PER_GROUP - E5_QUERIES_PER_GROUP)
            texts["test"].extend(query_chunks)
            labels["test"].extend([group] * len(query_chunks))
            texts["train"].extend(index_chunks)
            labels["train"].extend([group] * len(index_chunks))
        model = SentenceTransformer(E5_MODEL)
        train = model.encode(texts["train"], batch_size=64, show_progress_bar=True).astype(np.float32)
        queries = model.encode(texts["test"], batch_size=64, show_progress_bar=True).astype(np.float32)
        train_labels = np.asarray(labels["train"], dtype=np.int32)
        query_labels = np.asarray(labels["test"], dtype=np.int32)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, train=train, train_labels=train_labels, queries=queries, query_labels=query_labels)
    metadata = {
        "dataset_description": f"20 Newsgroups, {E5_MODEL}, {E5_CHUNK_SIZE}-char chunks, "
                               f"{E5_DOCS_PER_GROUP} chunks per group: {E5_DOCS_PER_GROUP - E5_QUERIES_PER_GROUP} indexed / {E5_QUERIES_PER_GROUP} queries (paper Table 1).",
        "query_description": "Chunks from documents not used for indexing; category labels are used only for the EP oracle condition.",
        "query_label_mode": "known_20news_category_oracle",
        "train_count": len(train),
        "test_count": len(queries),
    }
    return normalize_rows(train), train_labels, normalize_rows(queries), query_labels, metadata


LABELED_TEXT = {
    # name: (HF dataset, text field(s), label field)
    "AG News": ("fancyzhx/ag_news", ("text",), "label"),
    "DBpedia": ("fancyzhx/dbpedia_14", ("title", "content"), "label"),
    "Languages": ("papluca/language-identification", ("text",), "labels"),
}
TEXT_INDEX_TOTAL = 21_000
TEXT_QUERY_TOTAL = 1_000


def load_labeled_text(name: str, query_num: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Labeled text collections embedded with multilingual-e5-small (same as 20 Newsgroups).

    Every group gets TEXT_INDEX_TOTAL / nG indexed and TEXT_QUERY_TOTAL / nG query
    documents (sampled without overlap from train+test, fixed seed); whole documents,
    no chunking. ``query_num``/``seed`` are ignored. Cached under CACHE_DIR.
    """
    repo, fields, label_field = LABELED_TEXT[name]
    cache = CACHE_DIR / f"{name.replace(' ', '_').lower()}_e5small_{TEXT_INDEX_TOTAL}_{TEXT_QUERY_TOTAL}_s{E5_DATA_SEED}.npz"
    if cache.exists():
        z = np.load(cache)
        train, train_labels, queries, query_labels = z["train"], z["train_labels"], z["queries"], z["query_labels"]
    else:
        from datasets import concatenate_datasets, load_dataset as hf_load
        from sentence_transformers import SentenceTransformer

        logging.info("%s を %s で埋め込みます（初回のみ、%s にキャッシュ）", name, E5_MODEL, cache)
        ds = hf_load(repo)
        ds = concatenate_datasets([ds[k] for k in ds.keys()])
        texts = [" ".join(str(row[f]) for f in fields) for row in ds]
        raw = np.asarray(ds[label_field])
        classes = sorted(set(raw.tolist()))
        labels = np.asarray([classes.index(v) for v in raw], dtype=np.int32)
        n_groups = len(classes)
        per_index, per_query = TEXT_INDEX_TOTAL // n_groups, TEXT_QUERY_TOTAL // n_groups
        rng = np.random.default_rng(E5_DATA_SEED)
        idx_i, idx_q = [], []
        for g in range(n_groups):
            members = rng.permutation(np.flatnonzero((labels == g) & np.asarray([len(t.strip()) > 20 for t in texts])))
            idx_q.extend(members[:per_query])
            idx_i.extend(members[per_query: per_query + per_index])
        model = SentenceTransformer(E5_MODEL)
        train = model.encode([texts[i] for i in idx_i], batch_size=64, show_progress_bar=True).astype(np.float32)
        queries = model.encode([texts[i] for i in idx_q], batch_size=64, show_progress_bar=True).astype(np.float32)
        train_labels, query_labels = labels[idx_i], labels[idx_q]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, train=train, train_labels=train_labels, queries=queries, query_labels=query_labels)
    metadata = {
        "dataset_description": f"{name} ({repo}), {E5_MODEL}, {len(train)} indexed / {len(queries)} queries, "
                               f"{len(np.unique(train_labels))} groups.",
        "query_description": "Held-out documents; category labels are used only for the EP oracle condition.",
        "query_label_mode": "known_text_category_oracle",
    }
    return normalize_rows(train), train_labels, normalize_rows(queries), query_labels, metadata


FULL_QUERY_TOTAL = 1_000


def _encode_cached(cache: Path, index_texts: list[str], index_labels, query_texts: list[str], query_labels):
    """Embed with multilingual-e5-small once and cache (vectors are not normalised here)."""
    if cache.exists():
        z = np.load(cache)
        return z["train"], z["train_labels"], z["queries"], z["query_labels"]
    from sentence_transformers import SentenceTransformer

    logging.info("%s: %d + %d 件を %s で埋め込みます（初回のみ）", cache.name, len(index_texts), len(query_texts), E5_MODEL)
    model = SentenceTransformer(E5_MODEL)
    train = model.encode(index_texts, batch_size=128, show_progress_bar=True).astype(np.float32)
    queries = model.encode(query_texts, batch_size=128, show_progress_bar=True).astype(np.float32)
    train_labels, query_labels = np.asarray(index_labels, dtype=np.int32), np.asarray(query_labels, dtype=np.int32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, train=train, train_labels=train_labels, queries=queries, query_labels=query_labels)
    return train, train_labels, queries, query_labels


def load_full_dataset(name: str, query_num: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Datasets at their natural size (group sizes are not equalised).

    FULL_QUERY_TOTAL documents are drawn at random (fixed seed) as queries, so the query
    groups follow the natural group proportions; everything else is indexed. Queries and
    indexed items never come from the same document.
      - "20 Newsgroups (full)": all documents (subset="all"), 500-char chunks as in the paper
        setting; one random chunk per query document, all chunks of the other documents indexed.
      - "<name> (full)" for AG News / DBpedia / Languages: whole documents, train+test.
    ``query_num``/``seed`` are ignored.
    """
    rng = random.Random(E5_DATA_SEED)
    base = name.removesuffix(" (full)")
    cache = CACHE_DIR / f"full_{base.replace(' ', '_').lower()}_e5small_q{FULL_QUERY_TOTAL}_s{E5_DATA_SEED}.npz"
    if cache.exists():
        train, train_labels, queries, query_labels = _encode_cached(cache, [], [], [], [])
    elif base == "20 Newsgroups":
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=E5_CHUNK_SIZE, chunk_overlap=E5_CHUNK_OVERLAP)
        ds = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        docs = [(t, int(l)) for t, l in zip(ds.data, ds.target) if len(t.strip()) > 50]
        order = list(range(len(docs)))
        rng.shuffle(order)
        q_idx, i_idx = order[:FULL_QUERY_TOTAL], order[FULL_QUERY_TOTAL:]
        q_texts = [rng.choice(splitter.split_text(docs[i][0])) for i in q_idx]
        q_labels = [docs[i][1] for i in q_idx]
        i_texts, i_labels = [], []
        for i in i_idx:
            chunks = splitter.split_text(docs[i][0])
            i_texts.extend(chunks)
            i_labels.extend([docs[i][1]] * len(chunks))
        train, train_labels, queries, query_labels = _encode_cached(cache, i_texts, i_labels, q_texts, q_labels)
    else:
        from datasets import concatenate_datasets, load_dataset as hf_load

        repo, fields, label_field = LABELED_TEXT[base]
        ds = hf_load(repo)
        ds = concatenate_datasets([ds[k] for k in ds.keys()])
        texts = [" ".join(str(row[f]) for f in fields) for row in ds]
        raw = np.asarray(ds[label_field])
        classes = sorted(set(raw.tolist()))
        labels = [classes.index(v) for v in raw]
        keep = [i for i, t in enumerate(texts) if len(t.strip()) > 20]
        rng.shuffle(keep)
        q_idx, i_idx = keep[:FULL_QUERY_TOTAL], keep[FULL_QUERY_TOTAL:]
        train, train_labels, queries, query_labels = _encode_cached(
            cache, [texts[i] for i in i_idx], [labels[i] for i in i_idx],
            [texts[i] for i in q_idx], [labels[i] for i in q_idx])
    metadata = {
        "dataset_description": f"{name}: natural size ({len(train)} indexed / {len(queries)} queries), {E5_MODEL}.",
        "query_description": "Random held-out documents (natural group proportions).",
        "query_label_mode": "known_text_category_oracle",
    }
    return normalize_rows(train), train_labels, normalize_rows(queries), query_labels, metadata


SEPARATION_LEVELS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


def generate_separated_data(spread: float, query_num: int, seed: int, per_group: int | None = None,
                            n_groups: int | None = None, random_order: bool = False,
                            data_seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """20 equal clusters with controllable separation (20 Newsgroups-sized: 1050 indexed / 50 queries per group).

    Centers are random unit vectors (nearly orthogonal in 384 dims). A point is
    normalize(center + spread * N(0, I/DIM)); expected intra/inter cosine-distance
    ratio is about spread^2 / (1 + spread^2): 0.06 (spread 0.25) ... 0.90 (spread 3,
    close to 20 Newsgroups) ... 0.96 (spread 5). Data are fixed by the spread;
    ``query_num``/``seed`` are ignored. ``data_seed`` > 0 ("Separated s=2.0 data=1") draws
    other centres, points and queries from the same distribution (0 = the original data).
    """
    groups = N_CLUSTERS if n_groups is None else n_groups
    rng = np.random.default_rng(40_000 + int(spread * 100) + (0 if n_groups is None else 1_000 * n_groups)
                                + 7_000_000 * data_seed)
    centers = normalize_rows(rng.normal(size=(groups, DIM)))
    per_index, per_query = E5_DOCS_PER_GROUP - E5_QUERIES_PER_GROUP, E5_QUERIES_PER_GROUP
    if per_group is not None:              # larger groups: "Separated s=2.0 n=5000"
        per_index = per_group
    if n_groups is not None:               # other group counts: "Separated s=2.0 g=50" (21000 / 1000 in total)
        per_index, per_query = TEXT_INDEX_TOTAL // n_groups, TEXT_QUERY_TOTAL // n_groups

    def sample(per_group: int) -> tuple[np.ndarray, np.ndarray]:
        labels = np.repeat(np.arange(groups, dtype=np.int32), per_group)
        points = centers[labels] + spread * rng.normal(size=(len(labels), DIM)) / np.sqrt(DIM)
        return normalize_rows(points).astype(np.float32), labels

    train, train_labels = sample(per_index)
    queries, query_labels = sample(per_query)
    if random_order:                        # "order=random": insert in random order instead of group by group
        perm = np.random.default_rng(50_000).permutation(len(train))
        train, train_labels = train[perm], train_labels[perm]
    metadata = {
        "dataset_description": f"20 separated Gaussian clusters, spread={spread}, "
                               f"expected intra/inter distance ratio {spread**2 / (1 + spread**2):.2f}.",
        "query_description": "Independent samples from the same clusters; labels are the generating cluster.",
        "query_label_mode": "known_synthetic_cluster_label",
        "spread": spread,
    }
    return train, train_labels, queries, query_labels, metadata


def load_dataset(dataset_name: str, query_num: int, seed: int):
    if dataset_name.endswith(" (full)"):
        return load_full_dataset(dataset_name, query_num, seed)
    if dataset_name in LABELED_TEXT:
        return load_labeled_text(dataset_name, query_num, seed)
    if dataset_name.startswith("Separated s="):
        parts = dict(p.split("=") for p in dataset_name.split()[1:])
        return generate_separated_data(float(parts["s"]), query_num, seed,
                                       int(parts["n"]) if "n" in parts else None,
                                       int(parts["g"]) if "g" in parts else None,
                                       parts.get("order") == "random", int(parts.get("data", 0)))
    if dataset_name == "Normal":
        return generate_normal_data(query_num, seed)
    if dataset_name == "Zipfian s=1.0":
        return generate_zipfian_data(1.0, query_num, seed)
    if dataset_name == "Zipfian s=2.0":
        return generate_zipfian_data(2.0, query_num, seed)
    if dataset_name == "Concentrated":
        return generate_skewed_data("concentrated", query_num, seed)
    if dataset_name == "Hollowed":
        return generate_skewed_data("hollowed", query_num, seed)
    if dataset_name == "20 Newsgroups":
        return load_20news_e5(query_num, seed)
    if dataset_name == "20 Newsgroups (train/test split)":
        return load_20news_e5(query_num, seed, split="train_test")
    if dataset_name == "20 Newsgroups (MiniLM, legacy)":
        return load_20news_data(query_num, seed)
    raise ValueError(dataset_name)
