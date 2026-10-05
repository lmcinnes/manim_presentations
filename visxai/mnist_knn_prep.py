"""
Precompute data for mnist_knn_slides.py: MNIST digits, nearest neighbours
in raw pixel space, 2D UMAP, and ranked showcase edges (see knn_edge_prep).

    python mnist_knn_prep.py                  # all 70,000 digits (a few min)
    python mnist_knn_prep.py --n 20000        # subsample (faster)

Writes mnist_knn.npz and mnist_knn.candidates.png (for hand-picking).
"""

import argparse
import gzip
import urllib.request
from pathlib import Path

import numpy as np

from knn_edge_prep import add_analysis_args, analyse_and_save

HERE = Path(__file__).resolve().parent
MIRROR = "https://raw.githubusercontent.com/fgnt/mnist/master/"
FILES = dict(
    xtr="train-images-idx3-ubyte.gz",
    ytr="train-labels-idx1-ubyte.gz",
    xte="t10k-images-idx3-ubyte.gz",
    yte="t10k-labels-idx1-ubyte.gz",
)


def load_mnist(cache_dir):
    """All 70k MNIST images (uint8, N x 784) and labels. Tries a GitHub
    mirror of the original IDX files first, then OpenML via sklearn."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    try:
        arrs = {}
        for key, name in FILES.items():
            path = cache_dir / name
            if not path.exists():
                print("downloading", name)
                urllib.request.urlretrieve(MIRROR + name, path)
            with gzip.open(path, "rb") as f:
                data = f.read()
            if key.startswith("x"):
                arrs[key] = np.frombuffer(data, np.uint8, offset=16).reshape(-1, 784)
            else:
                arrs[key] = np.frombuffer(data, np.uint8, offset=8)
        X = np.vstack([arrs["xtr"], arrs["xte"]])
        y = np.concatenate([arrs["ytr"], arrs["yte"]]).astype(int)
    except Exception as e:  # network dependent
        print("mirror failed:", e, "- trying OpenML")
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False,
            data_home=str(cache_dir),
        )
        X, y = X.astype(np.uint8), y.astype(int)
    return X, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--n", type=int, default=0, help="subsample size (0 = all 70,000 digits)"
    )
    p.add_argument("--cache", default=str(HERE / "mnist_cache"))
    p.add_argument("--out", default=str(HERE / "mnist_knn.npz"))
    add_analysis_args(p)
    a = p.parse_args()

    X, y = load_mnist(a.cache)
    if a.n and a.n < len(X):
        sel = np.sort(np.random.default_rng(a.seed).choice(len(X), a.n, replace=False))
    else:
        sel = np.arange(len(X))
    X, y = X[sel], y[sel]
    analyse_and_save(
        X.astype(np.float32) / 255.0,
        y,
        [str(d) for d in range(10)],
        X,
        a.out,
        a,
        feature_space="pixel space",
        source_index=sel,
    )


if __name__ == "__main__":
    main()
