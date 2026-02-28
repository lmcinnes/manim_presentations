# EVoC — TMLS 2026 Presentation

Animated slide deck for the [EVoC](https://github.com/TutteInstitute/evoc) talk
at TMLS 2026 (Toronto Machine Learning Summit), built with
[Manim](https://www.manim.community/) and
[manim-slides](https://github.com/jeertmans/manim-slides).

## Prerequisites

* [uv](https://docs.astral.sh/uv/) for project environment

## Installation

From the `evoc_tmls/` directory:

```bash
cd evoc_tmls
uv sync          # creates .venv and installs all dependencies
```

## Data pipeline

Presentation data lives under `data/` in several subdirectories:

| Directory | Contents | Generator |
|-----------|----------|-----------|
| `data/extradata/` | Base dataset, density profiles, barcodes, cluster trees, precomputed HDBSCAN | `density_and_barcodes.py` |
| `data/benchmarks/` | Swarm-plot arrays for ARI / CS / time benchmarks | `benchmarks.py` |
| `data/simulation/` | Pymunk physics simulation for the icon-deluge animation | `falling_icon_simulation.py` |
| `data/embeddings/` | Raw embedding vectors (CIFAR-100, 20 Newsgroups, BirdCLEF) | Manually placed / downloaded |
| `data/embedding_cartoon/` | Toy kNN-graph embedding data | Static (checked in) |
| `data/logo/` | EVoC logo point-cloud data | Static (checked in) |

All file paths are centralised in **`data_manifest.py`** — generator scripts
and `slides.py` both import from there.

### Automatic generation on import

When `slides.py` is imported (i.e. during rendering), it calls `ensure_data()`
for each generator module. This checks whether all expected output files exist
and regenerates any that are missing. **You do not need to run the generators
manually for a normal build.**

### Manual regeneration

To force-regenerate a specific dataset (e.g. after changing parameters):

```bash
uv run python density_and_barcodes.py      # regenerate density/barcode/HDBSCAN data
uv run python falling_icon_simulation.py    # regenerate pymunk simulation
uv run python benchmarks.py                 # regenerate benchmark swarm data (slow)
```

To generate only missing files (same as the automatic import-time behaviour):

```bash
uv run python density_and_barcodes.py --only-missing
uv run python falling_icon_simulation.py --only-missing
uv run python benchmarks.py --only-missing
```

> **Note:** `benchmarks.py` downloads datasets and runs clustering benchmarks,
> which can take several minutes.

## Building the presentation

### Render all scenes

```bash
uv run manim-slides render slides.py
```

This renders every `Slide` subclass in `slides.py` into the `media/` directory
and writes per-scene JSON configs into `slides/`.

To render a single scene:

```bash
uv run manim-slides render slides.py SortingDensity
```

### Present interactively

```bash
uv run manim-slides present \
  EmbeddingUseCase EVoCLogo HighDClusteringOverview \
  PhaseManifold KnnEmbeddingStep PhaseDensity \
  SortingDensity PhaseClusters DensityToBarcode \
  PersistenceBarcodeAnimation Benefits EVoCPerformance Summary
```

Navigate with arrow keys; press `q` to quit.

### Export to HTML

```bash
uv run manim-slides convert --to html \
  EmbeddingUseCase EVoCLogo HighDClusteringOverview \
  PhaseManifold KnnEmbeddingStep PhaseDensity \
  SortingDensity PhaseClusters DensityToBarcode \
  PersistenceBarcodeAnimation Benefits EVoCPerformance Summary \
  evoc_tmls.html
```

### Export to PowerPoint

```bash
uv run manim-slides convert --to pptx \
  EmbeddingUseCase EVoCLogo HighDClusteringOverview \
  PhaseManifold KnnEmbeddingStep PhaseDensity \
  SortingDensity PhaseClusters DensityToBarcode \
  PersistenceBarcodeAnimation Benefits EVoCPerformance Summary \
  evoc_tmls.pptx
```

### Scene order

The canonical scene order is listed in `scene_order.txt` (one class name per
line). You can use it to build command lines programmatically:

```bash
uv run manim-slides present $(cat scene_order.txt | tr '\n' ' ')
```

## Project structure

```
evoc_tmls/
├── slides.py                  # Main presentation (all Slide subclasses)
├── extra_scenes.py            # Bonus / unused scene classes
├── config.py                  # Shared Manim styling (imported as ../config.py)
├── manim.cfg                  # Manim rendering settings (2560×1440, 30 fps)
├── pyproject.toml             # Project dependencies
├── data_manifest.py           # Central registry of all data file paths
├── density_and_barcodes.py    # Generator: density profiles, barcodes, HDBSCAN
├── falling_icon_simulation.py # Generator: pymunk icon-deluge simulation
├── benchmarks.py              # Generator: clustering benchmark swarm data
├── scene_order.txt            # Canonical scene ordering
├── icons/                     # Icon PNG assets for the deluge animation
├── data/
│   ├── extradata/             # Density, barcode, cluster-tree arrays
│   ├── benchmarks/            # Benchmark swarm-plot arrays
│   ├── simulation/            # Pymunk simulation JSON
│   ├── embeddings/            # Raw embedding vectors
│   ├── embedding_cartoon/     # Toy kNN embedding data
│   └── logo/                  # EVoC logo point-cloud data
├── media/                     # Rendered video output (git-ignored)
└── slides/                    # manim-slides JSON configs (git-ignored)
```

## Rendering settings

Configured in `manim.cfg`:

- **Resolution:** 2560 × 1440 (QHD)
- **Frame rate:** 30 fps
- **Background:** white
- **Text colour:** black
