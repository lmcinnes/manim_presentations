"""Precompute data for the KernelDials slides.

    python assets_kernel_dials.py                        # MNIST via OpenML
    python assets_kernel_dials.py --mnist-pickle mnist.pkl.gz

Reads assets/graph_repair.npz (the test circle), writes assets/kernel_dials.npz
and assets/kernel_dials.json. The MNIST part lays out 1,500 digits four times
with exact all-pairs repulsion: a few minutes each on one core.

The kernel family (Burr XII):  q(d) = (1 + (d/sigma)^p)^(-nu).
Everything here is computed directly, so every figure on the slides is ours:

* mixture: for p = 2 the kernel is E[exp(-tau d^2)], tau ~ Gamma(nu, rate sigma^2);
  checked by numerical integration, and sample Gaussians for the animation.
* target: a single edge minimising w phi + gamma rho settles at q* = w/(w+gamma),
  checked by direct minimisation for several kernels.
* denoise: the test circle from the repair section, laid out at several values
  of nu*p (p = 2, onset scale fixed).
* compact: 1,500 MNIST digits laid out at several p with nu*p = 2 fixed.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import integrate, optimize

OUT = Path(__file__).parent / "assets"
DENOISE_NUP = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
COMPACT_P = (1.0, 2.0, 4.0)
COMPACT_NUP = 2.0


def burr(d, sigma, p, nu):
    return (1 + (d / sigma) ** p) ** (-nu)


def burr_layout(Y0, edges, p, nu, sigma, gamma, steps=2500, lr=0.02):
    """Adam on F = sum_edges phi + gamma * sum_pairs rho, exact repulsion."""
    Y = Y0.copy()
    n = len(Y)
    m, v = np.zeros_like(Y), np.zeros_like(Y)
    a, b = edges[:, 0], edges[:, 1]
    h = p / 2.0
    for t in range(1, steps + 1):
        diff = Y[:, None] - Y[None]
        s = (diff ** 2).sum(-1) + np.eye(n)
        u = (s / sigma ** 2) ** h
        du = h * (s / sigma ** 2) ** (h - 1) / sigma ** 2
        q = (1 + u) ** (-nu)
        drho = -nu * (1 + u) ** (-nu - 1) * du / np.maximum(1 - q, 1e-12)
        np.fill_diagonal(drho, 0)
        g = gamma * 2 * (drho[:, :, None] * diff).sum(1)
        e = Y[a] - Y[b]
        se = (e ** 2).sum(1) + 1e-12
        dphi = nu * h * (se / sigma ** 2) ** (h - 1) / sigma ** 2 / (1 + (se / sigma ** 2) ** h)
        pull = dphi[:, None] * 2 * e
        np.add.at(g, a, pull)
        np.add.at(g, b, -pull)
        m = 0.9 * m + 0.1 * g
        v = 0.99 * v + 0.01 * g ** 2
        Y -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.99 ** t)) + 1e-8)
    return Y


def procrustes_onto(Y, ref):
    """Centre, scale to unit RMS radius, and rotate/reflect Y onto ref."""
    A = Y - Y.mean(0)
    A = A / np.sqrt((A ** 2).sum(1).mean())
    B = ref - ref.mean(0)
    B = B / np.sqrt((B ** 2).sum(1).mean())
    u, _, vt = np.linalg.svd(A.T @ B)
    return A @ (u @ vt)


def mixture_part():
    d = np.linspace(0, 10, 201)
    worst = 0.0
    for nu in (0.25, 1.0, 3.0):
        for sigma in (0.5, 1.0, 2.0):
            rate = sigma ** 2
            for dv in d[::10]:
                def density_times_gaussian(t, dv=dv, nu=nu, rate=rate):
                    # exp(-t d^2) times the Gamma(shape nu, rate) density
                    return math.exp(-t * dv ** 2 + nu * math.log(rate) + (nu - 1) * math.log(max(t, 1e-300))
                                    - rate * t - math.lgamma(nu))

                val, _ = integrate.quad(density_times_gaussian, 0, np.inf, limit=200)
                worst = max(worst, abs(val - burr(dv, sigma, 2, nu)))
    rng = np.random.default_rng(3)
    taus = rng.gamma(shape=1.0, scale=1.0, size=40)  # nu = 1, sigma = 1: the t-SNE kernel
    grid = np.linspace(0, 4, 161)
    samples = np.exp(-taus[:, None] * grid[None] ** 2)
    return dict(mix_grid=grid, mix_samples=samples, mix_taus=taus), dict(max_error=float(worst))


def target_part():
    kernels = {"t-SNE": (1.0, 2.0, 1.0), "UMAP default": (1.5769 ** (-1 / (2 * 0.8951)), 2 * 0.8951, 1.0),
               "p=1, nu=2": (1.0, 1.0, 2.0), "p=4, nu=0.5": (1.0, 4.0, 0.5)}
    worst = 0.0
    for sigma, p, nu in kernels.values():
        for w in (1.0, 0.5):
            for gamma in (0.1, 0.5, 1.0):
                def energy(logd):
                    q = burr(np.exp(logd), sigma, p, nu)
                    return -w * np.log(q) - gamma * np.log(1 - q)
                res = optimize.minimize_scalar(energy, bounds=(-8, 8), method="bounded",
                                               options=dict(xatol=1e-12))
                worst = max(worst, abs(burr(np.exp(res.x), sigma, p, nu) - w / (w + gamma)))
    return dict(max_error=float(worst), kernels={k: list(v) for k, v in kernels.items()})


def denoise_part():
    from assets_graph_repair import gap, layout_neighbours, roundness

    rep = np.load(OUT / "graph_repair.npz")
    meta = json.loads((OUT / "graph_repair.json").read_text())
    theta, kept, shortcuts, missing, start = rep["theta"], rep["kept"], rep["shortcuts"], rep["missing"], rep["start"]
    observed = np.r_[kept, shortcuts]
    k = 2 * meta["half"]
    g = gap(theta[:, None], theta[None])
    np.fill_diagonal(g, np.inf)
    true_nb = np.argsort(g, axis=1)[:, :k]
    layouts, stats, ref = [], {}, None
    for nup in DENOISE_NUP:
        p = 2.0
        nu = nup / p
        sigma = nu ** (1 / p)  # onset scale sigma * nu^(-1/p) = 1
        Y = burr_layout(start, observed, p, nu, sigma, meta["gamma"])
        nb, lay = layout_neighbours(Y, k)
        lengths = np.linalg.norm(Y[observed[:, 0]] - Y[observed[:, 1]], axis=1)
        stats[str(nup)] = dict(
            roundness=roundness(Y),
            overlap=float(np.mean([len(set(x) & set(y)) / k for x, y in zip(nb, true_nb)])),
            restored=int(sum(tuple(e) in lay for e in missing)),
            stretch=float(lengths[len(kept):].mean() / lengths[:len(kept)].mean()),
        )
        ref = Y if ref is None else ref
        layouts.append(procrustes_onto(Y, layouts[-1] if layouts else Y))
        print(f"nu*p {nup}: {stats[str(nup)]}", flush=True)
    return dict(denoise_layouts=np.array(layouts, dtype=np.float32)), stats


def compact_part(pickle_path, n=1500, k=10, steps=1000):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from assets_graph_repair import spectral

    if pickle_path:
        import gzip
        import pickle

        with gzip.open(pickle_path, "rb") as f:
            parts = pickle.load(f, encoding="latin1")
        X = np.concatenate([q[0] for q in parts])
        y = np.concatenate([q[1] for q in parts]).astype(int)
    else:
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        X, y = X / 255.0, y.astype(int)
    sel = np.sort(np.random.default_rng(0).choice(len(X), n, replace=False))
    X, y = X[sel], y[sel]
    nb = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    edges = np.unique(np.sort(np.c_[np.repeat(np.arange(n), k), nb.ravel()], axis=1), axis=0)
    start = spectral(edges, n) * 5
    gamma = len(edges) / (n * (n - 1) / 2)
    layouts, stats = [], {}
    for p in COMPACT_P:
        nu = COMPACT_NUP / p
        sigma = nu ** (1 / p)
        Y = burr_layout(start, edges, p, nu, sigma, gamma, steps=steps)
        ari = np.mean([adjusted_rand_score(y, KMeans(10, n_init=4, random_state=r).fit_predict(Y)) for r in range(3)])
        stats[str(p)] = dict(silhouette=float(silhouette_score(Y, y)), ari=float(ari))
        layouts.append(procrustes_onto(Y, layouts[-1] if layouts else Y))
        print(f"p {p}: {stats[str(p)]}", flush=True)
    return dict(compact_layouts=np.array(layouts, dtype=np.float32), compact_labels=y), stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-pickle")
    parser.add_argument("--mnist-steps", type=int, default=1000)
    args = parser.parse_args()
    mix, mix_stats = mixture_part()
    target_stats = target_part()
    print("mixture check:", mix_stats, "| target check:", target_stats["max_error"], flush=True)
    denoise, denoise_stats = denoise_part()
    compact, compact_stats = compact_part(args.mnist_pickle, steps=args.mnist_steps)
    np.savez_compressed(OUT / "kernel_dials.npz", **mix, **denoise, **compact)
    meta = dict(mixture=mix_stats, target=target_stats, denoise_nup=list(DENOISE_NUP), denoise=denoise_stats,
                compact_p=list(COMPACT_P), compact_nup=COMPACT_NUP, compact=compact_stats,
                umap_p=2 * 0.8951, umap_sigma=1.5769 ** (-1 / (2 * 0.8951)))
    (OUT / "kernel_dials.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
