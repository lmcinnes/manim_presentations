"""Precompute data for the WhereNext slides (the pivot to ongoing research,
and "noise in high dimensions").

    python assets_where_next.py                          # MNIST via OpenML
    python assets_where_next.py --mnist-pickle mnist.pkl.gz

Writes assets/where_next.npz and assets/where_next.json.

Parts:
* mnist: nearest neighbour of every digit in pixel space, a UMAP layout, and
  which 1-NN edges are long in the layout and join different map regions.
  Candidate showcase edges (long, between typical members of different
  digits) go on a contact sheet, assets/where_next_candidates.png, labelled by
  edge number: pick the pairs that look best and list them in
  WhereNext.SHOWCASE_EDGES. Both digit images of every candidate are stored,
  so any candidate can be shown without rerunning this script.
* circle: 48 points on a circle, with isotropic noise in 400 extra
  dimensions; the 1-NN graph at each noise level (a genuine random draw:
  CIRCLE_SEED was chosen for a clean start, from seeds that all behave alike).
* distances: from one point on the circle, the distances to every other point
  as the number of noise dimensions grows, with contrast and rank agreement.
* spiral: a two-turn spiral lifted into 256 dimensions with noise; its kNN
  graph at 400 and 1,600 points (same noise, same k), edges classified by how
  far apart their endpoints really are along the curve.
* strands: two parallel strands in 1,024 dimensions; the kNN graph bridges
  them heavily while a PCA split of the raw data separates them.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

OUT = Path(__file__).parent / "assets"

CIRCLE_N, CIRCLE_NOISE_DIMS, CIRCLE_SEED, CIRCLE_SIGMA_MAX = 48, 400, 3, 0.30
DIST_SIGMA, DIST_DIMS_MAX = 0.15, 400
SPIRAL = dict(turns=2.0, gap=3.0, r0=1.5, D=256, sigma=0.08, k=10, sizes=(400, 1600))
STRANDS = dict(n_per=300, length=10.0, gap=0.5, D=1024, sigma=0.12, k=10)


# ---------------------------------------------------------------------------
# MNIST: pixel-space nearest neighbours on a UMAP layout
# ---------------------------------------------------------------------------
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
    return (X / 255.0).astype(np.float32), y.astype(int)


N_CANDIDATES = 60


def contact_sheet(cand, pairs, labels, length, images, chosen, path, cols=6):
    """Every candidate edge as its two digits side by side, labelled
    '#edge: a-b' (and the edge's length as a fraction of the layout's span);
    * marks the default picks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(cand) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 1.45))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for k, e in enumerate(cand):
        a, b = pairs[e]
        pair = np.hstack([images[k, 0].reshape(28, 28), np.zeros((28, 4)), images[k, 1].reshape(28, 28)])
        ax = np.atleast_1d(axes).ravel()[k]
        ax.imshow(255 - pair, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"#{e}: {labels[a]}-{labels[b]}  ({length[e]:.2f})" + (" *" if e in chosen else ""), fontsize=8)
    fig.suptitle("Showcase candidates: edge number, digits, length (* = default pick).\n"
                 "List your picks in WhereNext.SHOWCASE_EDGES.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("contact sheet:", path, flush=True)


def mnist_part(pickle_path, seed=42, long_fraction=0.12):
    import umap
    from umap.umap_ import nearest_neighbors

    X, y = load_mnist(pickle_path)
    n = len(X)
    knn_i, knn_d, _ = nearest_neighbors(X, 15, "euclidean", {}, False, np.random.RandomState(seed))
    # first neighbour other than the point itself
    nn1 = np.array([row[row != i][0] for i, row in enumerate(knn_i)])
    layout = umap.UMAP(random_state=seed, precomputed_knn=(knn_i, knn_d, None)).fit_transform(X)

    pairs = np.unique(np.sort(np.c_[np.arange(n), nn1], axis=1), axis=0)
    u, v = pairs[:, 0], pairs[:, 1]
    span = float(np.linalg.norm(layout.max(0) - layout.min(0)))
    length = np.linalg.norm(layout[u] - layout[v], axis=1) / span
    lay_nb = NearestNeighbors(n_neighbors=16).fit(layout).kneighbors(layout, return_distance=False)[:, 1:]
    region = np.array([np.bincount(y[r], minlength=10).argmax() for r in lay_nb])
    is_long = (length > long_fraction) & (region[u] != region[v])

    # Showcase: long edges between different digits whose endpoints are both
    # typical of their own digit, in the data and in the layout.
    data_pure = (y[knn_i[:, 1:11]] == y[:, None]).mean(1)
    lay_pure = (y[lay_nb] == y[:, None]).mean(1)
    # Loose filters: the final choice is made by eye from the contact sheet.
    ok = is_long & (y[u] != y[v]) & (data_pure[u] >= 0.7) & (data_pure[v] >= 0.7) \
        & (lay_pure[u] >= 0.8) & (lay_pure[v] >= 0.8)
    cand = np.flatnonzero(ok)
    cand = cand[np.argsort(-length[cand])][:N_CANDIDATES]
    # default picks, used until SHOWCASE_EDGES is set: two pairs, four different digits
    chosen, used = [], set()
    for e in cand:
        a, b = int(y[u[e]]), int(y[v[e]])
        if a in used or b in used:
            continue
        chosen.append(int(e))
        used |= {a, b}
        if len(chosen) == 2:
            break
    cand_images = np.stack([[np.round(X[pairs[e, 0]] * 255), np.round(X[pairs[e, 1]] * 255)] for e in cand])
    contact_sheet(cand, pairs, y, length, cand_images, chosen, OUT / "where_next_candidates.png")
    arrays = dict(mnist_layout=layout.astype(np.float32), mnist_labels=y, mnist_pairs=pairs.astype(np.int32),
                  mnist_long=is_long, mnist_candidates=cand.astype(np.int64),
                  mnist_candidate_images=cand_images.astype(np.uint8))
    stats = dict(n=int(n), edges=int(len(pairs)), cross=int((y[u] != y[v]).sum()), long=int(is_long.sum()),
                 candidates=int(len(cand)),
                 showcase=[dict(edge=e, digits=[int(y[u[e]]), int(y[v[e]])]) for e in chosen])
    return arrays, stats


# ---------------------------------------------------------------------------
# Circle with noise in many hidden directions
# ---------------------------------------------------------------------------
def angular_gap(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def circle_part():
    rng = np.random.default_rng(CIRCLE_SEED)
    n = CIRCLE_N
    theta = 2 * np.pi * (np.arange(n) + 0.35 * rng.uniform(-1, 1, n)) / n
    eps = rng.standard_normal((n, CIRCLE_NOISE_DIMS))
    ring = np.c_[np.cos(theta), np.sin(theta)]
    sigmas = np.linspace(0, CIRCLE_SIGMA_MAX, 61)
    nn = np.empty((len(sigmas), n), dtype=np.int32)
    across = np.empty(len(sigmas), dtype=np.int32)
    for s, sigma in enumerate(sigmas):
        X = np.c_[ring, sigma * eps]
        d = ((X[:, None] - X[None]) ** 2).sum(-1)
        np.fill_diagonal(d, np.inf)
        nn[s] = d.argmin(1)
        edges = {(min(i, int(j)), max(i, int(j))) for i, j in enumerate(nn[s])}
        across[s] = sum(angular_gap(theta[a], theta[b]) > np.pi / 2 for a, b in edges)

    # Distances from one point as the number of noise dimensions grows.
    query = 0
    others = np.setdiff1d(np.arange(n), [query])
    dims = np.unique(np.round(np.geomspace(1, DIST_DIMS_MAX, 60)).astype(int))
    true_gap = angular_gap(theta[query], theta[others])
    chord2 = ((ring[others] - ring[query]) ** 2).sum(1)
    dist = np.empty((len(dims), len(others)))
    for s, D in enumerate(dims):
        noise = DIST_SIGMA * (eps[others, :D] - eps[query, :D])
        dist[s] = np.sqrt(chord2 + (noise ** 2).sum(1))
    from scipy.stats import spearmanr

    contrast = dist.max(1) / dist.min(1)
    agreement = np.array([spearmanr(true_gap, row).correlation for row in dist])
    return dict(circle_theta=theta, circle_sigmas=sigmas, circle_nn=nn, circle_across=across,
                dist_dims=dims, dist_values=dist.astype(np.float32), dist_true_gap=true_gap,
                dist_contrast=contrast, dist_agreement=agreement, dist_query=np.array(query))


# ---------------------------------------------------------------------------
# Spiral and strands
# ---------------------------------------------------------------------------
def spiral_points(n, turns, gap, r0):
    b = gap / (2 * np.pi)
    th = np.linspace(0, 2 * np.pi * turns, 20000)
    r = r0 + b * th
    xy = np.c_[r * np.cos(th), r * np.sin(th)]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    s = np.linspace(0, arc[-1], n)
    return np.c_[np.interp(s, arc, xy[:, 0]), np.interp(s, arc, xy[:, 1])], s


def lift(P, D, sigma, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(D, P.shape[1])))
    return P @ Q.T + sigma * rng.normal(size=(len(P), D))


def spiral_part():
    c = SPIRAL
    out, stats = {}, {}
    for n in c["sizes"]:
        P, s = spiral_points(n, c["turns"], c["gap"], c["r0"])
        X = lift(P, c["D"], c["sigma"], seed=0)
        k = c["k"]
        obs = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
        gap_arc = np.abs(s[:, None] - s[None, :])
        np.fill_diagonal(gap_arc, np.inf)
        true = np.argsort(gap_arc, axis=1)[:, :k]
        overlap = float(np.mean([len(set(o) & set(t)) / k for o, t in zip(obs, true)]))
        r_k = k * (s[1] - s[0]) / 2  # latent distance to the k-th neighbour (both sides)
        src = np.repeat(np.arange(n), k)
        dst = obs.ravel()
        sep = np.abs(s[src] - s[dst]) / r_k
        genuine = np.array([dst[m] in set(true[src[m]]) for m in range(len(src))])
        kind = np.where(genuine, 0, np.where(sep <= 2, 1, 2))  # 0 genuine, 1 near miss, 2 wrong
        indeg = np.bincount(dst, minlength=n)
        # Hubs: estimate each point's own noise effect locally (its mean squared
        # distance to its 50 nearest points, minus the same quantity averaged over
        # those points), subtract it from every distance to that point, rebuild.
        sq = (X ** 2).sum(1)
        D2 = sq[:, None] + sq[None] - 2 * X @ X.T
        np.fill_diagonal(D2, np.inf)
        near = np.argsort(D2, axis=1)[:, :50]
        local = np.take_along_axis(D2, near, axis=1).mean(1)
        effect = local - local[near].mean(1)
        fixed = np.argsort(D2 - effect[None, :], axis=1)[:, :k]
        indeg_fixed = np.bincount(fixed.ravel(), minlength=n)
        overlap_fixed = float(np.mean([len(set(o) & set(t)) / k for o, t in zip(fixed, true)]))
        out[f"spiral_{n}_indegree_corrected"] = indeg_fixed
        out[f"spiral_{n}_points"] = P
        out[f"spiral_{n}_edges"] = np.c_[src, dst].astype(np.int32)
        out[f"spiral_{n}_kind"] = kind.astype(np.int8)
        out[f"spiral_{n}_indegree"] = indeg
        stats[str(n)] = dict(overlap=overlap, near_miss=float(np.mean(kind == 1)), wrong=float(np.mean(kind == 2)),
                             indegree_sd=float(indeg.std()), indegree_max=int(indeg.max()), hub=int(indeg.argmax()),
                             corrected=dict(indegree_sd=float(indeg_fixed.std()), indegree_max=int(indeg_fixed.max()),
                                            overlap=overlap_fixed))
    return out, stats


def strands_part():
    c = STRANDS
    x = np.linspace(0, c["length"], c["n_per"])
    P = np.r_[np.c_[x, np.zeros(c["n_per"])], np.c_[x, np.full(c["n_per"], c["gap"])]]
    lab = np.r_[np.zeros(c["n_per"], int), np.ones(c["n_per"], int)]
    X = lift(P, c["D"], c["sigma"], seed=1)
    obs = NearestNeighbors(n_neighbors=c["k"] + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    src = np.repeat(np.arange(len(P)), c["k"])
    dst = obs.ravel()
    bridging = float(np.mean(lab[src] != lab[dst]))
    pcs = PCA(4, random_state=0).fit_transform(X)
    accs = []
    for col in range(4):
        side = pcs[:, col] > np.median(pcs[:, col])
        accs.append(max(np.mean(side == lab), np.mean(side != lab)))
    best = int(np.argmax(accs))
    return (dict(strands_points=P, strands_labels=lab, strands_edges=np.c_[src, dst].astype(np.int32),
                 strands_pc=pcs[:, best]),
            dict(bridging=bridging, pca_accuracy=float(accs[best]), pc=best + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-pickle", help="mnist.pkl.gz (Nielsen format) instead of OpenML")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    mnist, mnist_stats = mnist_part(args.mnist_pickle)
    circle = circle_part()
    spiral, spiral_stats = spiral_part()
    strands, strands_stats = strands_part()
    np.savez_compressed(OUT / "where_next.npz", **mnist, **circle, **spiral, **strands)
    meta = dict(mnist=mnist_stats, spiral=spiral_stats, strands=strands_stats,
                circle=dict(n=CIRCLE_N, noise_dims=CIRCLE_NOISE_DIMS, seed=CIRCLE_SEED,
                            sigma_max=CIRCLE_SIGMA_MAX, across_max=int(circle["circle_across"][-1])),
                distances=dict(sigma=DIST_SIGMA, dims_max=DIST_DIMS_MAX,
                               contrast_start=float(circle["dist_contrast"][0]),
                               contrast_end=float(circle["dist_contrast"][-1]),
                               agreement_start=float(circle["dist_agreement"][0]),
                               agreement_end=float(circle["dist_agreement"][-1])),
                spiral_params=SPIRAL, strands_params=STRANDS)
    (OUT / "where_next.json").write_text(json.dumps(meta, indent=2, default=int))
    print(json.dumps(meta, indent=2, default=int))


if __name__ == "__main__":
    main()
