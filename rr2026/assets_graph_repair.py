"""Precompute data for the GraphRepair slides.

    python assets_graph_repair.py

Writes assets/graph_repair.npz and assets/graph_repair.json (about a minute).
The MNIST callback slide also reads assets/where_next.npz.

All layouts minimise the likelihood objective from the research,

    F(Y) = sum_edges w phi(d^2) + gamma * sum_all_pairs rho(d^2),
    phi = -log q,  rho = -log(1 - q),

with full (exact) repulsion over every pair, and Adam steps. Two links q:

    spring (Gaussian)   q = exp(-d^2)      attraction grows with distance
    heavy tail (Cauchy) q = 1 / (1 + d^2)  attraction fades like 1/d

Parts:
* repair: 200 points on a circle, each joined to its 12 nearest along the
  circle; then 12 random shortcuts added and 48 true edges removed. Both
  kernels lay out this graph from the same spectral start; snapshots are kept
  for the animation, with shape and repair measures at the end.
* chord: one diametric chord added to the clean circle; how much each kernel
  lets it pull the pair together.
* bundles: m aligned chords between opposite arcs (the heavy-tailed kernel);
  how far the circle gives way as m grows.
"""

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "assets"
N, HALF, SEED = 200, 6, 0
N_SHORTCUTS, N_MISSING = 12, 48
STEPS, SNAPSHOT_EVERY = 2500, 25
BUNDLES = (1, 4, 8, 12, 20)
GAMMA = 0.1 * 96 / N * HALF / 3


def gap(a, b):
    x = np.abs(a - b) % (2 * np.pi)
    return np.minimum(x, 2 * np.pi - x)


# attraction phi and its derivative, repulsion rho and its derivative, in s = d^2
KERNELS = {
    "spring": (lambda s: 1.0 + 0 * s, lambda s: -np.exp(-s) / (1 - np.exp(-s) + 1e-12)),
    "cauchy": (lambda s: 1 / (1 + s), lambda s: -1 / (s * (1 + s) + 1e-12)),
}


def layout(Y0, edges, kernel, gamma=GAMMA, steps=STEPS, every=None, lr=0.02):
    """Adam on F(Y) with exact all-pairs repulsion; optional snapshots."""
    dphi, drho = KERNELS[kernel]
    Y = Y0.copy()
    n = len(Y)
    m, v = np.zeros_like(Y), np.zeros_like(Y)
    snaps = [Y.copy()] if every else None
    a, b = edges[:, 0], edges[:, 1]
    for t in range(1, steps + 1):
        diff = Y[:, None] - Y[None]
        s = (diff ** 2).sum(-1)
        R = drho(s + np.eye(n))
        np.fill_diagonal(R, 0)
        g = gamma * 2 * (R[:, :, None] * diff).sum(1)
        e = Y[a] - Y[b]
        pull = dphi((e ** 2).sum(1))[:, None] * 2 * e
        np.add.at(g, a, pull)
        np.add.at(g, b, -pull)
        m = 0.9 * m + 0.1 * g
        v = 0.99 * v + 0.01 * g ** 2
        Y -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.99 ** t)) + 1e-8)
        if every and t % every == 0:
            snaps.append(Y.copy())
    return (Y, np.array(snaps)) if every else Y


def spectral(edges, n):
    W = np.zeros((n, n))
    W[edges[:, 0], edges[:, 1]] = W[edges[:, 1], edges[:, 0]] = 1
    inv = 1 / np.sqrt(W.sum(1))
    _, vecs = np.linalg.eigh(np.eye(n) - inv[:, None] * W * inv[None])
    Y = vecs[:, 1:3] * inv[:, None]
    return 2.0 * Y / np.abs(Y).max()


def roundness(Y):
    """Spread of the radius about the centre, as a fraction of the mean."""
    r = np.linalg.norm(Y - Y.mean(0), axis=1)
    return float(r.std() / r.mean())


def layout_neighbours(Y, k):
    d = ((Y[:, None] - Y[None]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    nb = np.argsort(d, axis=1)[:, :k]
    return nb, {(min(i, int(j)), max(i, int(j))) for i in range(len(Y)) for j in nb[i]}


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    theta = 2 * np.pi * (np.arange(N) + 0.3 * rng.uniform(-1, 1, N)) / N
    true = sorted({(min(i, (i + s) % N), max(i, (i + s) % N)) for i in range(N) for s in range(1, HALF + 1)})
    shortcuts = set()
    while len(shortcuts) < N_SHORTCUTS:
        a, b = rng.choice(N, 2, replace=False)
        if gap(theta[a], theta[b]) > np.pi / 2:
            shortcuts.add((min(a, b), max(a, b)))
    shortcuts = sorted(shortcuts)
    drop = set(rng.choice(len(true), N_MISSING, replace=False).tolist())
    missing = [e for m, e in enumerate(true) if m in drop]
    kept = [e for m, e in enumerate(true) if m not in drop]
    observed = np.array(kept + shortcuts)
    start = spectral(observed, N)

    k = 2 * HALF
    g = gap(theta[:, None], theta[None])
    np.fill_diagonal(g, np.inf)
    true_nb = np.argsort(g, axis=1)[:, :k]
    arrays = dict(theta=theta, kept=np.array(kept), shortcuts=np.array(shortcuts), missing=np.array(missing),
                  start=start)
    stats = {"start": dict(roundness=roundness(start))}
    for kernel in ("spring", "cauchy"):
        final, snaps = layout(start, observed, kernel, every=SNAPSHOT_EVERY)
        nb, lay_edges = layout_neighbours(final, k)
        lengths = np.linalg.norm(final[observed[:, 0]] - final[observed[:, 1]], axis=1)
        stats[kernel] = dict(
            roundness=roundness(final),
            stretch=float(lengths[len(kept):].mean() / lengths[:len(kept)].mean()),
            shortcuts_kept=int(sum(e in lay_edges for e in shortcuts)),
            missing_restored=int(sum(e in lay_edges for e in missing)),
            overlap=float(np.mean([len(set(a) & set(b)) / k for a, b in zip(nb, true_nb)])),
        )
        arrays[f"{kernel}_snapshots"] = snaps.astype(np.float32)
        print(kernel, stats[kernel], flush=True)

    # One diametric chord on the clean circle.
    clean = np.array(true)
    i, j = 0, N // 2
    stats["chord"] = {}
    for kernel in ("spring", "cauchy"):
        base = layout(spectral(clean, N), clean, kernel, steps=1500)
        after = layout(base, np.r_[clean, [[i, j]]], kernel, steps=1500)
        stats["chord"][kernel] = float(1 - np.linalg.norm(after[i] - after[j]) / np.linalg.norm(base[i] - base[j]))

    # Aligned bundles of chords between opposite arcs (heavy-tailed kernel).
    base = layout(spectral(clean, N), clean, "cauchy", steps=1500)
    stats["bundles"] = {}
    for m in BUNDLES:
        offsets = np.arange(m) - m // 2
        chords = np.sort(np.c_[offsets % N, (N // 2 - offsets) % N], axis=1)
        final = layout(base, np.r_[clean, chords], "cauchy", steps=1500)
        contraction = 1 - np.linalg.norm(final[0] - final[N // 2]) / np.linalg.norm(base[0] - base[N // 2])
        stats["bundles"][str(m)] = float(contraction)
        arrays[f"bundle_{m}"] = final.astype(np.float32)
        arrays[f"bundle_{m}_chords"] = chords
        print(f"{m} aligned chords: contraction {contraction:.0%}", flush=True)

    np.savez_compressed(OUT / "graph_repair.npz", **arrays)
    meta = dict(n=N, half=HALF, shortcuts=N_SHORTCUTS, missing=N_MISSING, gamma=GAMMA, steps=STEPS,
                snapshot_every=SNAPSHOT_EVERY, bundles=list(BUNDLES), stats=stats)
    (OUT / "graph_repair.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
