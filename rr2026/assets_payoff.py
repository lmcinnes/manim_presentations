"""Precompute data for the Payoff slides (MNIST).

    python assets_payoff.py                         # MNIST via OpenML
    python assets_payoff.py --mnist-pickle mnist.pkl.gz   # Nielsen's pickle instead
    python assets_payoff.py --time-budget 20        # stop starting new fits after 20 min

Every fit is cached in assets/payoff_cache/, so the script can be rerun to
continue where it stopped; the outputs are written once all fits exist.

Fits (all sharing one precomputed kNN graph):

* the repulsion sweep, gamma in {1, 2, 4, 8, 16, 32}, for three setups:
  - new:        compatibility_layout=False, Adam, recursive init,
                negative_selection_range=20000 (hard negatives on)
  - new_no_hn:  the same with the window covering every point (hard negatives off)
  - old:        compatibility_layout=True, compatibility optimizer, spectral init
* longer runs at gamma = 1, n_epochs in {200, 500, 1000, 2000}, old and new.

Metrics per sweep fit: trustworthiness, kNN classifier accuracy, neighbour
preservation (share of each point's data-space nearest neighbours kept in the
embedding), digit-centroid distance correlation, and silhouette.

Writes assets/payoff.npz, assets/payoff.json and old-stack thumbnails in
assets/payoff/.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score

import umap
from umap.umap_ import nearest_neighbors

from assets_recursive import normalise, quality, rotation_to
from assets_render import COLOR_CYCLE, render

OUT = Path(__file__).parent / "assets"
CACHE = OUT / "payoff_cache"
THUMBS = OUT / "payoff"
GAMMAS = [1, 2, 4, 8, 16, 32]
EPOCHS = [200, 500, 1000, 2000]
WINDOW = 20_000


def setups(n):
    new = dict(
        compatibility_layout=False,
        optimizer="adam",
        init="recursive",
        negative_selection_range=WINDOW,
    )
    new_no_hn = dict(new, negative_selection_range=n)
    old = dict(compatibility_layout=True, optimizer="compatibility", init="spectral")
    fits = {}
    for g in GAMMAS:
        fits[f"new_g{g}"] = dict(new, repulsion_strength=float(g))
        fits[f"new_no_hn_g{g}"] = dict(new_no_hn, repulsion_strength=float(g))
        fits[f"old_g{g}"] = dict(old, repulsion_strength=float(g))
    for e in EPOCHS:
        fits[f"new_e{e}"] = dict(new, n_epochs=e)
        fits[f"old_e{e}"] = dict(old, n_epochs=e)
    return fits


def load_mnist(pickle_path=None):
    if pickle_path:
        import gzip
        import pickle

        with gzip.open(pickle_path, "rb") as f:
            parts = pickle.load(f, encoding="latin1")
        X = np.concatenate([p[0] for p in parts]).astype(np.float32)
        y = np.concatenate([p[1] for p in parts]).astype(int)
        return X, y
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    return X.astype(np.float32), y.astype(int)


def separation(Y, labels):
    """Mean distance between class centroids over mean within-class RMS radius."""
    centroids = np.array([Y[labels == c].mean(0) for c in np.unique(labels)])
    radius = np.mean(
        [
            np.sqrt(((Y[labels == c] - centroids[k]) ** 2).sum(1).mean())
            for k, c in enumerate(np.unique(labels))
        ]
    )
    return float(pdist(centroids).mean() / radius)


def neighbour_preservation(knn_indices, Y):
    """Share of each point's nearest neighbours in the data (the kNN graph UMAP
    was given, self excluded) that are also among its nearest in the embedding."""
    from sklearn.neighbors import NearestNeighbors

    true = knn_indices[:, 1:]
    k = true.shape[1]
    emb = (
        NearestNeighbors(n_neighbors=k + 1)
        .fit(Y)
        .kneighbors(Y, return_distance=False)[:, 1:]
    )
    return float((emb[:, :, None] == true[:, None, :]).any(axis=2).mean())


def global_metrics(X, Y, labels, seed=0):
    classes = np.unique(labels)
    cx = np.array([X[labels == c].mean(0) for c in classes])
    cy = np.array([Y[labels == c].mean(0) for c in classes])
    rho = spearmanr(pdist(cx), pdist(cy)).correlation
    sil = silhouette_score(Y, labels, sample_size=10_000, random_state=seed)
    return {"centroid_correlation": float(rho), "silhouette": float(sil)}


def aligned_sequence(frames):
    out = [normalise(frames[0])]
    for frame in frames[1:]:
        current = normalise(frame)
        out.append(current @ rotation_to(current, out[-1]))
    return np.stack(out).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-pickle", help="path to mnist.pkl.gz (Nielsen format)")
    parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="minutes; stop starting new fits after this long",
    )
    args = parser.parse_args()
    for folder in (OUT, CACHE, THUMBS):
        folder.mkdir(exist_ok=True)
    start = time.time()

    X, labels = load_mnist(args.mnist_pickle)
    knn_file = CACHE / "knn.npz"
    if knn_file.exists():
        knn = np.load(knn_file)
        knn_i, knn_d = knn["i"], knn["d"]
    else:
        knn_i, knn_d, _ = nearest_neighbors(
            X, 15, "euclidean", {}, False, np.random.RandomState(42)
        )
        np.savez(knn_file, i=knn_i, d=knn_d)

    fits = setups(len(X))
    for name, params in fits.items():
        path = CACHE / f"{name}.npy"
        if path.exists():
            continue
        if args.time_budget and (time.time() - start) / 60 > args.time_budget:
            print("time budget reached; rerun to continue", flush=True)
            return
        t0 = time.time()
        embedding = umap.UMAP(
            random_state=0, precomputed_knn=(knn_i, knn_d, None), **params
        ).fit_transform(X)
        np.save(path, embedding.astype(np.float32))
        print(f"{name}: {time.time() - t0:.0f}s", flush=True)

    E = {name: np.load(CACHE / f"{name}.npy") for name in fits}
    sweep_names = [n for n in fits if "_g" in n]
    local = quality(X, labels, {n: E[n] for n in sweep_names})
    metrics = {
        n: dict(
            local[n],
            neighbour_preservation=neighbour_preservation(knn_i, E[n]),
            **global_metrics(X, E[n], labels),
        )
        for n in sweep_names
    }
    spread = {n: separation(E[n], labels) for n in fits if "_e" in n}

    # Old-stack thumbnails, rendered like every other point cloud in the talk.
    for g in GAMMAS:
        render(
            E[f"old_g{g}"],
            labels,
            COLOR_CYCLE,
            size=640,
            spread_px=2,
            fixed_spread=False,
        ).save(THUMBS / f"old_g{g}.png")

    np.savez_compressed(
        OUT / "payoff.npz",
        labels=labels,
        new_sweep=aligned_sequence([E[f"new_g{g}"] for g in GAMMAS]),
        new_epochs=aligned_sequence([E[f"new_e{e}"] for e in EPOCHS]),
        old_epochs=aligned_sequence([E[f"old_e{e}"] for e in EPOCHS]),
    )
    meta = {
        "gammas": GAMMAS,
        "epochs": EPOCHS,
        "window": WINDOW,
        "n": int(len(X)),
        "metrics": metrics,
        "separation": spread,
    }
    (OUT / "payoff.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
