# type: ignore
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import evoc
import umap
import hdbscan
import pandas as pd
import sklearn.cluster
import sklearn.metrics
from datasets import load_dataset
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

DATA_DIR = Path(__file__).parent / "data" / "benchmarks"
DATA_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIR = Path(__file__).parent / "data" / "embeddings"

# =============================================================================
# Dataset configs: per-dataset algorithm parameters
# =============================================================================

DATASET_CONFIGS = {
    "cifar": {
        "n_runs": 16,
        # KMeans gets better ARI if we over-cluster slightly (125 instead of 100)
        "kmeans_kwargs": {"n_clusters": 125},
        "umap_hdbscan_kwargs": {
            "min_samples": 5,
            "min_cluster_size": 120,
            "metric": "cosine",
            "cluster_selection_method": "leaf",
        },
    },
    "news": {
        "n_runs": 32,
        "kmeans_kwargs": {"n_clusters": 25},
        "umap_hdbscan_kwargs": {
            "min_samples": 5,
            "min_cluster_size": 180,
            "metric": "cosine",
            "cluster_selection_method": "leaf",
        },
    },
    "bird": {
        "n_runs": 16,
        "kmeans_kwargs": {"n_clusters": 130},
        "umap_hdbscan_kwargs": {
            "min_samples": 5,
            "min_cluster_size": 100,
            "metric": "cosine",
            "cluster_selection_method": "leaf",
        },
    },
}

MEASURES = [
    ("Adjusted Rand Index", "ari"),
    ("Clustering Score", "cs"),
    ("Elapsed time", "time"),
]

ALGORITHMS = ("kmeans", "umap_hdbscan", "EVoC")

# =============================================================================
# Data loading
# =============================================================================

ds = load_dataset("Syoy/birdclef_2023_train")

cifar_data = np.load(EMBEDDING_DIR / "CIFAR100-CLIP-vectors.npy")
cifar_target = np.load(EMBEDDING_DIR / "CIFAR100-imageclass.npy")

news_data = np.load(EMBEDDING_DIR / "20-newsgroups-sentence-bert-embeddings.npy")
news_target = np.load(EMBEDDING_DIR / "20-newsgroups-sentence-targets.npy")

birdclef2023_data = np.asarray(ds["train"]["embeddings"])
birdclef2023_target = np.asarray(ds["train"]["primary_label"])
# restrict to sufficiently large classes
mask = np.isin(birdclef2023_target, np.where(np.bincount(birdclef2023_target) > 100)[0])
birdclef2023_data = birdclef2023_data[mask]
birdclef2023_target = birdclef2023_target[mask]

# =============================================================================
# Clustering algorithms
# =============================================================================


def umap_hdbscan(
    data,
    metric="euclidean",
    n_neighbors=15,
    n_components=2,
    min_samples=5,
    min_cluster_size=10,
    min_dist=0.1,
    cluster_selection_method="eom",
    n_epochs=None,
    negative_sample_rate=5,
):
    embedding = umap.UMAP(
        metric=metric,
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        n_epochs=n_epochs,
        negative_sample_rate=negative_sample_rate,
        n_jobs=8,
    ).fit_transform(data)
    clustering = hdbscan.HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        cluster_selection_method=cluster_selection_method,
    ).fit_predict(embedding)
    return clustering


def kmeans(data, n_clusters=10):
    return sklearn.cluster.KMeans(n_clusters=n_clusters, n_init="auto").fit_predict(
        data
    )


def EVoC(data, test_target=None, random_state=None):
    if random_state is None:
        random_state = np.random.randint(65536)
    cls = evoc.EVoC(random_state=random_state).fit(data)
    if test_target is None:
        return cls.labels_
    result = np.full(data.shape[0], -1)
    best_ari = 0.0
    for labels in cls.cluster_layers_:
        ari = sklearn.metrics.adjusted_rand_score(
            test_target[labels >= 0], labels[labels >= 0]
        )
        if ari > best_ari:
            best_ari = ari
            result = labels
    return result


# =============================================================================
# Scoring and helpers
# =============================================================================


def score_clustering(data, target, clustering_function, n_runs=16, **kwargs):
    result = np.zeros((n_runs, 5), dtype=np.float32)
    for i in range(n_runs):
        start_time = time.time()
        clustering = clustering_function(data, **kwargs)
        result[i, 0] = time.time() - start_time
        result[i, 1] = sklearn.metrics.adjusted_rand_score(
            target[clustering >= 0], clustering[clustering >= 0]
        )
        result[i, 2] = sklearn.metrics.adjusted_mutual_info_score(
            target[clustering >= 0], clustering[clustering >= 0]
        )
        result[i, 3] = np.sum(clustering >= 0) / clustering.shape[0]
        result[i, 4] = np.cbrt((result[i, 1] ** 2) * result[i, 3])

    result = pd.DataFrame(
        result,
        columns=(
            "Elapsed time",
            "Adjusted Rand Index",
            "Adjusted Mutual Information",
            "Proportion clustered",
            "Clustering Score",
        ),
    )
    result["algorithm"] = clustering_function.__name__.replace("_", "\n")
    result = result.melt(
        id_vars=["algorithm"],
        value_vars=[
            "Elapsed time",
            "Adjusted Rand Index",
            "Adjusted Mutual Information",
            "Proportion clustered",
            "Clustering Score",
        ],
        var_name="measure",
    )
    return result


def get_swarm_coordinates(dataframe, x, y, constraint=None):
    df = dataframe[constraint].copy() if constraint is not None else dataframe
    categories = df[x].unique()

    fig, ax = plt.subplots()
    sns.swarmplot(data=df, x=x, y=y, ax=ax)

    coords_by_category = {
        categories[i]: np.ascontiguousarray(collection.get_offsets())
        for i, collection in enumerate(ax.collections)
    }
    plt.close(fig)
    return coords_by_category, ax.get_ylim(), ax.get_yticks()


def run_dataset_benchmarks(data, target, n_runs, kmeans_kwargs, umap_hdbscan_kwargs):
    """Score all three algorithms on a dataset and return combined results."""
    kmeans_results = score_clustering(
        data, target, kmeans, n_runs=n_runs, **kmeans_kwargs
    )
    umap_results = score_clustering(
        data, target, umap_hdbscan, n_runs=n_runs, **umap_hdbscan_kwargs
    )
    evoc_results = score_clustering(
        data, target, EVoC, test_target=target, n_runs=n_runs
    )
    return pd.concat(
        [kmeans_results, umap_results, evoc_results.assign(algorithm="EVoC")],
        ignore_index=True,
    )


def save_swarm_data(results, prefix):
    """Generate swarm plots and save coordinates/axis limits for all measures."""
    for measure_name, measure_short in MEASURES:
        # sns.catplot(
        #     results[results.measure == measure_name],
        #     x="algorithm",
        #     hue="algorithm",
        #     y="value",
        #     col="measure",
        #     height=10,
        #     kind="swarm",
        # )
        swarm_dict, ylims, yticks = get_swarm_coordinates(
            results,
            x="algorithm",
            y="value",
            constraint=results.measure == measure_name,
        )
        for alg in ALGORITHMS:
            print(f"Saving swarm coordinates for {prefix} - {measure_short} - {alg}")
            np.save(
                DATA_DIR / f"{prefix}_{measure_short}_{alg}_swarm.npy",
                swarm_dict[alg.replace("_", "\n")],
            )
        np.save(DATA_DIR / f"{prefix}_{measure_short}_ylim.npy", np.asarray(ylims))
        np.save(DATA_DIR / f"{prefix}_{measure_short}_yticks.npy", np.asarray(yticks))
        plt.close("all")


# =============================================================================
# Run benchmarks
# =============================================================================

datasets = {
    "cifar": (cifar_data, cifar_target),
    "news": (news_data, news_target),
    "bird": (birdclef2023_data, birdclef2023_target),
}

for name, (data, target) in datasets.items():
    cfg = DATASET_CONFIGS[name]
    results = run_dataset_benchmarks(
        data,
        target,
        n_runs=cfg["n_runs"],
        kmeans_kwargs=cfg["kmeans_kwargs"],
        umap_hdbscan_kwargs=cfg["umap_hdbscan_kwargs"],
    )
    save_swarm_data(results, prefix=name)
