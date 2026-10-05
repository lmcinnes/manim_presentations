"""
Layouts made with the attraction-kernel family, for kernel_knobs.py.

    python kernel_layouts_prep.py        # a few minutes; writes kernel_layouts.npz

Kernel (from the point-process derivation):

    q(d) = I_x(alpha, k),   x = 1 / (1 + (d / sigma0)^m / alpha)

(k = 1: q = (1 + (d/sigma0)^m / alpha)^-alpha, the Burr form.)
Objective: edges pull with -log q, non-edges push with -gamma log(1 - q).

Runs:
  * circle: the rewired circle graph (as in long_tail_repair.py), exact
    all-pairs repulsion, Adam, one run per alpha from the same spectral start
  * MNIST (6,000 digits): 15-NN graph, UMAP-style negative sampling, one run
    per m and per k, all from the same spectral start; with measurements of
    output dimension, neighbour recall and cluster separation
"""
import gzip
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.special import beta as beta_fn, betainc

HERE = Path(__file__).resolve().parent
OUT = HERE / "kernel_layouts.npz"

CIRCLE_ALPHAS = (8.0, 2.0, 1.0, 0.5)
MNIST_M = (1.0, 1.5, 2.0)
MNIST_K = (1, 4, 8)
BASE = dict(sigma0=1.0, m=1.5, alpha=1.0, k=1)
MNIST_N, MNIST_KNN, EPOCHS, NEG = 6000, 15, 300, 5
SNAP_EVERY = 10


# ---------------------------------------------------------------------------
# The kernel and its forces
# ---------------------------------------------------------------------------
def kernel_terms(d, sigma0, m, alpha, k):
    """q(d) and dq/dd for the family (vectorised over d)."""
    d = np.maximum(d, 1e-4)
    u = (d / sigma0) ** m
    x = 1.0 / (1.0 + u / alpha)
    q = betainc(alpha, k, x)
    dI_dx = x ** (alpha - 1) * (1 - x) ** (k - 1) / beta_fn(alpha, k)
    dx_du = -(x ** 2) / alpha
    du_dd = m * d ** (m - 1) / sigma0 ** m
    return q, dI_dx * dx_du * du_dd


def forces(d, sigma0, m, alpha, k, eps=1e-3):
    """Attraction (from -log q) and repulsion (from -log(1-q)) magnitudes
    along the separation; both >= 0."""
    q, dq = kernel_terms(d, sigma0, m, alpha, k)
    attract = -dq / np.maximum(q, 1e-12)
    repel = -dq / np.maximum(1 - q, eps)
    return attract, repel


# ---------------------------------------------------------------------------
# Circle: exact repulsion, Adam (as in the repair scene)
# ---------------------------------------------------------------------------
def gap(a, b):
    x = np.abs(a - b) % (2 * np.pi)
    return np.minimum(x, 2 * np.pi - x)


def corrupted_circle(n=120, half=5, n_short=10, n_miss=30, seed=1):
    rng = np.random.default_rng(seed)
    theta = 2 * np.pi * (np.arange(n) + 0.3 * rng.uniform(-1, 1, n)) / n
    true = sorted({(min(i, (i + s) % n), max(i, (i + s) % n))
                   for i in range(n) for s in range(1, half + 1)})
    short = set()
    while len(short) < n_short:
        a, b = rng.choice(n, 2, replace=False)
        if gap(theta[a], theta[b]) > np.pi / 2:
            short.add((min(a, b), max(a, b)))
    drop = set(rng.choice(len(true), n_miss, replace=False).tolist())
    kept = [e for j, e in enumerate(true) if j not in drop]
    return theta, np.array(kept), np.array(sorted(short))


def spectral(edges, n, dim=2):
    W = np.zeros((n, n))
    W[edges[:, 0], edges[:, 1]] = W[edges[:, 1], edges[:, 0]] = 1
    inv = 1 / np.sqrt(W.sum(1))
    _, vecs = np.linalg.eigh(np.eye(n) - inv[:, None] * W * inv[None])
    Y = vecs[:, 1:dim + 1] * inv[:, None]
    return 2.0 * Y / np.abs(Y).max()


def circle_layout(Y0, edges, kern, gamma, steps=1500, lr=0.02, every=15):
    Y, n = Y0.copy(), len(Y0)
    m_, v_ = np.zeros_like(Y), np.zeros_like(Y)
    a, b = edges[:, 0], edges[:, 1]
    snaps = [Y.copy()]
    for t in range(1, steps + 1):
        diff = Y[:, None] - Y[None]
        d = np.sqrt((diff ** 2).sum(-1)) + np.eye(n)
        _, rep = forces(d, **kern)
        np.fill_diagonal(rep, 0)
        g = -gamma * ((rep / d)[:, :, None] * diff).sum(1)
        e = Y[a] - Y[b]
        de = np.sqrt((e ** 2).sum(1))
        att, _ = forces(de, **kern)
        pull = (att / np.maximum(de, 1e-4))[:, None] * e
        np.add.at(g, a, pull)
        np.add.at(g, b, -pull)
        m_ = 0.9 * m_ + 0.1 * g
        v_ = 0.99 * v_ + 0.01 * g ** 2
        Y -= lr * (m_ / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.99 ** t)) + 1e-8)
        if t % every == 0:
            snaps.append(Y.copy())
    return np.array(snaps, dtype=np.float32)


def roundness(Y):
    r = np.linalg.norm(Y - Y.mean(0), axis=1)
    return float(r.std() / r.mean())


# ---------------------------------------------------------------------------
# MNIST: negative-sampling layout
# ---------------------------------------------------------------------------
def load_mnist(n, seed=0, cache=HERE / "mnist_cache"):
    cache.mkdir(exist_ok=True)
    url = "https://raw.githubusercontent.com/fgnt/mnist/master/"
    out = {}
    for key, name, off in (("x", "train-images-idx3-ubyte.gz", 16),
                           ("y", "train-labels-idx1-ubyte.gz", 8)):
        path = cache / name
        if not path.exists():
            urllib.request.urlretrieve(url + name, path)
        with gzip.open(path, "rb") as f:
            out[key] = np.frombuffer(f.read(), np.uint8, offset=off)
    X = out["x"].reshape(-1, 784).astype(np.float32) / 255.0
    sel = np.random.default_rng(seed).choice(len(X), n, replace=False)
    return X[sel], out["y"][sel].astype(int)


def knn_graph(X, k):
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(
        X, return_distance=False)[:, 1:]
    rows = np.repeat(np.arange(len(X)), k)
    E = np.unique(np.sort(np.c_[rows, idx.ravel()], axis=1), axis=0)
    return E, idx


def graph_spectral_init(E, n, scale=10.0):
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import eigsh
    W = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    W = (W + W.T).tocsr()
    deg = np.asarray(W.sum(1)).ravel()
    Dm = diags(1 / np.sqrt(deg))
    L = diags(np.ones(n)) - Dm @ W @ Dm
    vals, vecs = eigsh(L, k=3, sigma=-1e-3, which="LM",
                       v0=np.random.default_rng(0).uniform(size=n))
    Y = vecs[:, np.argsort(vals)[1:3]]
    return (scale * (Y - Y.mean(0)) / np.abs(Y).max()).astype(np.float64)


def negsample_layout(Y0, E, kern, epochs=EPOCHS, neg=NEG, seed=0,
                     every=SNAP_EVERY, clip=4.0):
    """Batched UMAP-style SGD: each epoch, every edge pulls its ends
    together and each end is pushed from `neg` random points."""
    rng = np.random.default_rng(seed)
    Y, n = Y0.copy(), len(Y0)
    a, b = E[:, 0], E[:, 1]
    snaps = [Y.astype(np.float32)]
    for ep in range(epochs):
        lr = 1.0 * (1 - ep / epochs)
        e = Y[a] - Y[b]
        d = np.sqrt((e ** 2).sum(1))
        att, _ = forces(d, **kern)
        g = np.clip((att / np.maximum(d, 1e-4))[:, None] * e, -clip, clip)
        step = np.zeros_like(Y)
        np.add.at(step, a, -g)
        np.add.at(step, b, g)
        src = np.repeat(np.r_[a, b], neg)
        dst = rng.integers(0, n, len(src))
        e = Y[src] - Y[dst]
        d = np.sqrt((e ** 2).sum(1))
        _, rep = forces(d, **kern)
        g = np.clip((rep / np.maximum(d, 1e-4))[:, None] * e, -clip, clip)
        g[src == dst] = 0
        np.add.at(step, src, g)
        Y += lr * step / np.maximum(np.bincount(np.r_[a, b], minlength=n),
                                    1)[:, None]
        if (ep + 1) % every == 0:
            snaps.append(Y.astype(np.float32))
    return np.array(snaps)


def measures(Y, labels, in_idx, k=15):
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    D, idx = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y)
    D, idx = D[:, 1:], idx[:, 1:]
    recall = np.mean([len(set(a) & set(b)) / k for a, b in zip(idx, in_idx)])
    T = D[:, :10]
    dim = np.median(8 / np.log(T[:, -1:] / T[:, :-1]).sum(1))
    sil = silhouette_score(Y, labels, sample_size=3000, random_state=0)
    return dict(recall=float(recall), dimension=float(dim),
                silhouette=float(sil))


# ---------------------------------------------------------------------------
def main():
    arrays = {}
    # ---- circle: the denoising dial (alpha) ----
    theta, kept, short = corrupted_circle()
    obs = np.r_[kept, short]
    start = spectral(obs, len(theta))
    gamma = 0.1 * 96 / len(theta) * 5 / 3
    arrays.update(circle_theta=theta, circle_kept=kept, circle_short=short,
                  circle_alphas=np.array(CIRCLE_ALPHAS))
    for al in CIRCLE_ALPHAS:
        t0 = time.time()
        kern = dict(sigma0=1.0, m=2.0, alpha=al, k=1)
        snaps = circle_layout(start, obs, kern, gamma)
        arrays[f"circle_alpha_{al:g}"] = snaps
        print(f"circle alpha {al:g}: roundness {roundness(snaps[-1]):.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)

    # ---- MNIST: m and k ----
    X, y = load_mnist(MNIST_N)
    from sklearn.decomposition import PCA
    Xp = PCA(50, random_state=0).fit_transform(X)
    E, in_idx = knn_graph(Xp, MNIST_KNN)
    init = graph_spectral_init(E, len(X))
    arrays.update(mnist_labels=y, mnist_m=np.array(MNIST_M),
                  mnist_k=np.array(MNIST_K))
    runs = [("m", v, dict(BASE, m=v)) for v in MNIST_M] + \
           [("k", v, dict(BASE, k=v)) for v in MNIST_K]
    for name, v, kern in runs:
        t0 = time.time()
        snaps = negsample_layout(init, E, kern)
        arrays[f"mnist_{name}_{v:g}"] = snaps
        mm = measures(snaps[-1], y, in_idx)
        for key, val in mm.items():
            arrays[f"mnist_{name}_{v:g}_{key}"] = val
        print(f"mnist {name}={v:g}: " + ", ".join(
            f"{k_} {val:.3f}" for k_, val in mm.items())
              + f" ({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(OUT, **arrays)
    print("saved", OUT)


if __name__ == "__main__":
    main()
