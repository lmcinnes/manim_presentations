"""Precompute data for the HardNegatives slides.

    python assets_negatives.py              # MNIST via OpenML + UMAP 0.6dev
    python assets_negatives.py --synthetic  # quick stand-in, no downloads

Writes assets/negatives.npz and assets/negatives.json.

Two parts:

* a toy 2D embedding (a few thousand points in UMAP-like clusters) used for
  the mechanism slides, with the source point, projection direction and
  sampled negatives fixed here so the slides are reproducible;
* an MNIST embedding plus validation statistics for the random-projection
  window with negative_selection_range=20000: the measured chance that a
  point at distance r falls in the window, against the arc formula applied
  to each window's actual extent, and the distances, labels and forces of
  negatives drawn uniformly vs from the window.
"""

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "assets"
A, B = 1.5769434602697652, 0.8950608778515733
GAMMA = 1.0
WINDOW = 20_000  # negative_selection_range for MNIST
TOY_WINDOW_FRACTION = 0.1
TOY_ANGLE = np.radians(28.0)
N_SHOWN_NEGATIVES = 14


def repulsion(d):
    """Soft-clipped repulsive force magnitude at distance d (gamma = 1)."""
    d = np.asarray(d, dtype=np.float64)
    raw = 2 * GAMMA * B * d / ((0.001 + d * d) * (1 + A * d ** (2 * B)))
    return GAMMA * np.tanh(raw / GAMMA)


def window_indices(Y, source, angle, size):
    """Indices of the negative-selection window for ``source`` along ``angle``."""
    u = np.array([np.cos(angle), np.sin(angle)])
    t = (Y - Y[source]) @ u
    order = np.argsort(t, kind="stable")
    rank = np.empty(len(Y), dtype=np.int64)
    rank[order] = np.arange(len(Y))
    start = int(np.clip(rank[source] - size // 2, 0, len(Y) - size))
    return order[start : start + size], t


# ---------------------------------------------------------------------------
# Toy embedding
# ---------------------------------------------------------------------------
def toy_embedding(seed=11):
    rng = np.random.default_rng(seed)
    centres = np.array(
        [
            [-10.5, 4.5], [-5.5, 6.0], [0.5, 5.0], [6.5, 5.5], [11.0, 3.0],
            [-11.0, -2.0], [-4.5, 0.5], [2.0, -0.5], [8.5, -1.5],
            [-7.5, -6.0], [-1.0, -6.5], [5.5, -6.0],
        ]
    )
    points, labels = [], []
    for k, c in enumerate(centres):
        n = int(rng.integers(220, 480))
        sx, sy = rng.uniform(0.45, 1.25), rng.uniform(0.35, 0.8)
        theta = rng.uniform(0, np.pi)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        z = rng.normal(size=(n, 2)) * [sx, sy]
        if k % 3 == 0:  # a few curved clusters
            z[:, 1] += 0.25 * z[:, 0] ** 2 - 0.2
        points.append(c + z @ rot.T)
        labels.append(np.full(n, k))
    return np.concatenate(points), np.concatenate(labels)


def toy_part():
    Y, labels = toy_embedding()
    rng = np.random.default_rng(5)
    # Source: the most central point of the cluster nearest the middle.
    middle = np.argmin(np.linalg.norm(Y - Y.mean(0), axis=1))
    members = np.flatnonzero(labels == labels[middle])
    source = members[np.argmin(np.linalg.norm(Y[members] - Y[members].mean(0), axis=1))]

    size = int(TOY_WINDOW_FRACTION * len(Y))
    window, t = window_indices(Y, source, TOY_ANGLE, size)
    others = np.setdiff1d(np.arange(len(Y)), [source])
    pool_window = np.setdiff1d(window, [source])

    def typical_draw(pool):
        """Of 500 random draws, the one whose mean force is closest to the
        pool's mean force, so the handful shown on screen is representative."""
        target = repulsion(np.linalg.norm(Y[pool] - Y[source], axis=1)).mean()
        best, best_gap = None, np.inf
        for _ in range(500):
            draw = rng.choice(pool, N_SHOWN_NEGATIVES, replace=False)
            gap = abs(repulsion(np.linalg.norm(Y[draw] - Y[source], axis=1)).mean() - target)
            if gap < best_gap:
                best, best_gap = draw, gap
        return best

    uniform_neg = typical_draw(others)
    window_neg = typical_draw(pool_window)
    return {
        "toy_Y": Y.astype(np.float32),
        "toy_labels": labels,
        "toy_source": source,
        "toy_angle": TOY_ANGLE,
        "toy_window": window,
        "toy_uniform_neg": uniform_neg,
        "toy_window_neg": window_neg,
    }


# ---------------------------------------------------------------------------
# MNIST (or a stand-in) and validation statistics
# ---------------------------------------------------------------------------
def mnist_embedding():
    from sklearn.datasets import fetch_openml
    import umap

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    reducer = umap.UMAP(
        random_state=42,
        compatibility_layout=False,
        optimizer="adam",
        init="recursive",
        negative_selection_range=WINDOW,
    )
    return reducer.fit_transform(X.astype(np.float32)), y.astype(int)


def synthetic_stand_in(n=70_000, seed=3):
    """Ten elongated clusters on a UMAP-like scale; for testing only."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 10, n)
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False) + 0.4
    radii = rng.uniform(6.0, 11.0, 10)
    centres = np.c_[np.cos(angles), np.sin(angles)] * radii[:, None]
    stretch = rng.normal(size=(n, 2)) * [1.4, 0.6]
    rot = angles[labels] * 1.3
    local = np.c_[
        stretch[:, 0] * np.cos(rot) - stretch[:, 1] * np.sin(rot),
        stretch[:, 0] * np.sin(rot) + stretch[:, 1] * np.cos(rot),
    ]
    return centres[labels] + local, labels


def validation(Y, labels, size=WINDOW, n_sources=300, n_neg=50, seed=0):
    """Window statistics for ``n_sources`` random (source, direction) pairs.

    The hit rate is the measured chance that a point at distance r lies in
    the window.  The prediction is the arc formula applied to each window's
    actual extent on either side of the source (the kernel centres windows
    on rank, not position, so the two sides differ):
        P(r) = [arcsin(min(1, t_hi / r)) + arcsin(min(1, -t_lo / r))] / pi
    """
    rng = np.random.default_rng(seed)
    n = len(Y)
    sample = rng.choice(n, (2000, 2))
    r_max = float(np.quantile(np.linalg.norm(Y[sample[:, 0]] - Y[sample[:, 1]], axis=1), 0.95))
    edges = np.linspace(0.0, r_max, 41)
    hits, pred, counts = (np.zeros(len(edges) - 1) for _ in range(3))
    d_uni, d_win, same_uni, same_win, f_uni, f_win = [], [], [], [], [], []
    used = 0
    while used < n_sources:
        s = int(rng.integers(n))
        angle = rng.uniform(0, np.pi)
        window, t = window_indices(Y, s, angle, size)
        rank_pos = np.flatnonzero(window == s)
        if len(rank_pos) == 0 or not (size // 4 < rank_pos[0] < 3 * size // 4):
            continue  # skip sources whose window is clipped at an end
        used += 1
        t_lo, t_hi = t[window].min(), t[window].max()
        dist = np.linalg.norm(Y - Y[s], axis=1)
        inside = np.zeros(n, dtype=bool)
        inside[window] = True
        r = np.maximum(dist, 1e-12)
        predicted = (
            np.arcsin(np.minimum(1.0, t_hi / r)) + np.arcsin(np.minimum(1.0, -t_lo / r))
        ) / np.pi
        bins = np.digitize(dist, edges) - 1
        ok = (bins >= 0) & (bins < len(hits)) & (np.arange(n) != s)
        np.add.at(hits, bins[ok], inside[ok])
        np.add.at(pred, bins[ok], predicted[ok])
        np.add.at(counts, bins[ok], 1)

        wn = rng.choice(window[window != s], n_neg)
        un = rng.integers(0, n, n_neg)
        un = un[un != s]
        d_win.append(dist[wn])
        d_uni.append(dist[un])
        same_win.append(labels[wn] == labels[s])
        same_uni.append(labels[un] == labels[s])
        f_win.append(repulsion(dist[wn]))
        f_uni.append(repulsion(dist[un]))

    centres = 0.5 * (edges[:-1] + edges[1:])
    keep = counts > 50
    d_uni, d_win = np.concatenate(d_uni), np.concatenate(d_win)
    stats = {
        "window": size,
        "n": n,
        "r_max": r_max,
        "same_label_uniform": float(np.concatenate(same_uni).mean()),
        "same_label_window": float(np.concatenate(same_win).mean()),
        "mean_force_uniform": float(np.concatenate(f_uni).mean()),
        "mean_force_window": float(np.concatenate(f_win).mean()),
        "median_distance_uniform": float(np.median(d_uni)),
        "median_distance_window": float(np.median(d_win)),
    }
    arrays = {
        "hit_r": centres[keep],
        "hit_rate": hits[keep] / counts[keep],
        "hit_pred": pred[keep] / counts[keep],
        "dist_uniform": d_uni.astype(np.float32),
        "dist_window": d_win.astype(np.float32),
    }
    return arrays, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="use a stand-in for MNIST")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)

    toy = toy_part()
    if args.synthetic:
        Y, labels = synthetic_stand_in()
    else:
        Y, labels = mnist_embedding()
    arrays, stats = validation(Y, labels)

    # Example slab for the MNIST overview slide.
    rng = np.random.default_rng(1)
    centre = np.median(Y, axis=0)
    near = np.argsort(np.linalg.norm(Y - centre, axis=1))[:2000]
    example_source = int(rng.choice(near))

    np.savez_compressed(
        OUT / "negatives.npz",
        **toy,
        mnist_Y=Y.astype(np.float32),
        mnist_labels=labels,
        **arrays,
    )
    meta = {
        "synthetic": bool(args.synthetic),
        "example_source": example_source,
        "example_angle": float(np.radians(62.0)),
        **stats,
    }
    (OUT / "negatives.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
