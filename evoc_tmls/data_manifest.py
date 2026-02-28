"""Central registry of all data file paths used by the EVoC presentation.

Every module that reads or writes data files should import paths from here
rather than constructing its own.  This ensures filenames stay in sync
between generator scripts and consumer code.
"""

from pathlib import Path

# ── Root directories ─────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

EXTRADATA_DIR = DATA_DIR / "extradata"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDING_CARTOON_DIR = DATA_DIR / "embedding_cartoon"
LOGO_DIR = DATA_DIR / "logo"
SIMULATION_DIR = DATA_DIR / "simulation"

# ── extradata (density_and_barcodes.py) ──────────────────────────────────────

BASE_DATA = EXTRADATA_DIR / "base_data.npy"
DATA_COLORMAP = EXTRADATA_DIR / "data_colormap.npy"

CLUSTER_DENSITY_PROFILES = EXTRADATA_DIR / "cluster_density_profiles.npy"
CLUSTER_SIZES = EXTRADATA_DIR / "cluster_sizes.npy"
CLUSTER_BINARY_TREE = EXTRADATA_DIR / "cluster_binary_tree.npy"
BARCODE_BARS = EXTRADATA_DIR / "barcode_bars.npy"
PERSISTENCE_SCORES_TRACE = EXTRADATA_DIR / "persistence_scores_trace.npy"

# Precomputed HDBSCAN on SCALED_BASE_DATA (density_and_barcodes.py)
SCALED_CTREE = EXTRADATA_DIR / "scaled_ctree.pkl"
SCALED_POINTS_IN_PDF_ORDER = EXTRADATA_DIR / "scaled_points_in_pdf_order.npy"
SCALED_DENSITY_VALUES = EXTRADATA_DIR / "scaled_density_values.npy"

DENSITY_AND_BARCODES_FILES = [
    CLUSTER_DENSITY_PROFILES,
    CLUSTER_SIZES,
    CLUSTER_BINARY_TREE,
    BARCODE_BARS,
    PERSISTENCE_SCORES_TRACE,
    SCALED_CTREE,
    SCALED_POINTS_IN_PDF_ORDER,
    SCALED_DENSITY_VALUES,
]

# ── embedding_cartoon (static data, not generated) ──────────────────────────

EMBEDDING_CARTOON_RAW = EMBEDDING_CARTOON_DIR / "embedding_cartoon_raw_data.npy"
EMBEDDING_CARTOON_2D = EMBEDDING_CARTOON_DIR / "embedding_cartoon_2d_data.npy"
EMBEDDING_CARTOON_EDGES = EMBEDDING_CARTOON_DIR / "embedding_cartoon_edges.csv"

# ── logo (static data) ──────────────────────────────────────────────────────

EVOC_LOGO_DATA = LOGO_DIR / "evoc_logo_data.npy"
EVOC_LOGO_COLORS = LOGO_DIR / "evoc_logo_colors.npy"

# ── simulation (falling_icon_simulation.py) ──────────────────────────────────

ICON_DELUGE_SIMULATION = SIMULATION_DIR / "icon_deluge_simulation.json"

SIMULATION_FILES = [
    ICON_DELUGE_SIMULATION,
]

# ── benchmarks (benchmarks.py) ──────────────────────────────────────────────

BENCHMARK_DATASETS = ["cifar", "news", "bird"]
BENCHMARK_METRICS = ["ari", "cs", "time"]
BENCHMARK_ALGORITHMS = ["kmeans", "umap_hdbscan", "EVoC"]


def benchmark_file(dataset: str, metric: str, algo: str) -> Path:
    """Return the path for a benchmark swarm data file."""
    return BENCHMARKS_DIR / f"{dataset}_{metric}_{algo}_swarm.npy"


def benchmark_ylim_file(dataset: str, metric: str) -> Path:
    return BENCHMARKS_DIR / f"{dataset}_{metric}_ylim.npy"


def benchmark_yticks_file(dataset: str, metric: str) -> Path:
    return BENCHMARKS_DIR / f"{dataset}_{metric}_yticks.npy"


def all_benchmark_files() -> list[Path]:
    """Return the full list of expected benchmark output files."""
    files = []
    for ds in BENCHMARK_DATASETS:
        for metric in BENCHMARK_METRICS:
            for algo in BENCHMARK_ALGORITHMS:
                files.append(benchmark_file(ds, metric, algo))
            files.append(benchmark_ylim_file(ds, metric))
            files.append(benchmark_yticks_file(ds, metric))
    return files
