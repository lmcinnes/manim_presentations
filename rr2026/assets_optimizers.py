"""Precompute data for the NewOptimizers slides.

    python assets_optimizers.py          # writes assets/optimizers.npz and .json

Needs numpy, scikit-learn and umap-learn (0.6dev branch).  No manim required.

The optimizers here are small numpy re-implementations of the kernels in
umap/layouts.py on the 0.6dev branch, so the slides show real behaviour:

* ``old_epoch``: the compatibility kernel, looping over edges, moving both
  endpoints immediately, clamping each coordinate to +/-4.
* ``adam_small_epoch``: the new small-graph Adam kernel
  (optimize_small_layout_euclidean_single_epoch_adam): per-node force
  accumulation against frozen positions, half-strength attraction,
  hash-selected negatives, tanh-soft-clipped repulsion, then one Adam step.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs
from umap.umap_ import fuzzy_simplicial_set

OUT = Path(__file__).parent / "assets"
A, B = 1.5769434602697652, 0.8950608778515733  # min_dist=0.1, spread=1
GAMMA = 1.0
NEGATIVE_SAMPLE_RATE = 5.0

# ---------------------------------------------------------------------------
# Tiny graph for the mechanism diagrams (scene coordinates, hand placed)
# ---------------------------------------------------------------------------
TINY_POS = np.array(
    [
        [-5.0, 1.2], [-4.0, 1.7], [-4.8, -0.1], [-3.7, 0.4],  # group A
        [-1.7, -1.1], [-0.6, -0.7], [-1.0, -2.1],  # group B
        [0.9, 1.5], [2.0, 1.2], [1.3, 0.2],  # group C
    ]
)
TINY_EMB_SCALE = 0.4  # embedding units per scene unit: realistic UMAP spacings
TINY_SOURCE = 3  # node used for the single-node close-ups


def fuzzy_graph(X, n_neighbors, seed):
    graph, _, _ = fuzzy_simplicial_set(
        X, n_neighbors, np.random.RandomState(seed), "euclidean"
    )
    graph = graph.tocsr()
    graph.sort_indices()
    return graph


def prune(graph, n_epochs):
    graph = graph.copy()
    graph.data[graph.data < graph.data.max() / float(n_epochs)] = 0.0
    graph.eliminate_zeros()
    return graph


def schedules(weights, n_epochs, new_kernel):
    eps = weights.max() / weights  # make_epochs_per_sample
    epns = eps / NEGATIVE_SAMPLE_RATE
    if new_kernel:
        epns = epns * 1.5  # modern optimizers use 1/1.5 of the negative rate
    return eps, epns, eps.copy(), epns.copy()


def adam_alpha_schedule(n_epochs, initial_alpha=1.0):
    """layouts._create_alpha_schedule for optimizer='adam', not a good init."""
    warm = int(min(n_epochs / 2, 100))
    head = [(2.0 * initial_alpha - 0.1) * (1 - n / warm) ** 2 + 0.1 for n in range(warm)]
    tail = [
        0.15 * (1 - (n - warm) / (n_epochs - warm)) + 0.05
        for n in range(warm, n_epochs)
    ]
    return np.array(head + tail)


# ---------------------------------------------------------------------------
# Old (compatibility) kernel: edge loop, immediate updates, per-coordinate clip
# ---------------------------------------------------------------------------
def old_epoch(Y, heads, tails, sched, n, alpha, edge_order, node_rngs):
    eps, epns, eons, eonns = sched
    n_vertices = Y.shape[0]
    for idx in edge_order:
        if eons[idx] > n:
            continue
        j, k = heads[idx], tails[idx]
        diff = Y[j] - Y[k]
        d2 = diff @ diff
        coeff = -2 * A * B * d2 ** (B - 1) / (A * d2**B + 1) if d2 > 0 else 0.0
        grad = np.clip(coeff * diff, -4.0, 4.0)
        Y[j] += grad * alpha
        Y[k] -= grad * alpha  # move_other=True
        eons[idx] += eps[idx]
        n_neg = int((n - eonns[idx]) / epns[idx])
        for _ in range(max(n_neg, 0)):
            kk = node_rngs[j].integers(n_vertices)
            diff = Y[j] - Y[kk]
            d2 = diff @ diff
            if d2 > 0:
                coeff = 2 * GAMMA * B / ((0.001 + d2) * (A * d2**B + 1))
                Y[j] += np.clip(coeff * diff, -4.0, 4.0) * alpha
        eonns[idx] += n_neg * epns[idx]


def run_old_interleaved(Y0, graph, n_epochs, interleave_seed, rng_seed=0):
    """Two 'threads' own contiguous halves of the edge list (static prange
    scheduling); the interleaving of their updates varies run to run, while
    the random seed stays fixed."""
    coo = graph.tocoo()
    heads, tails = coo.row, coo.col
    sched = schedules(coo.data, n_epochs, new_kernel=False)
    n_edges = len(heads)
    halves = [np.arange(n_edges // 2), np.arange(n_edges // 2, n_edges)]
    mix = np.random.default_rng(interleave_seed)
    node_rngs = [np.random.default_rng(rng_seed + i) for i in range(Y0.shape[0])]
    Y = Y0.copy()
    for n in range(n_epochs):
        which = mix.permutation(np.r_[np.zeros(len(halves[0])), np.ones(len(halves[1]))])
        pointers = [0, 0]
        order = []
        for w in which.astype(int):
            order.append(halves[w][pointers[w]])
            pointers[w] += 1
        old_epoch(Y, heads, tails, sched, n, 1.0 - n / n_epochs, order, node_rngs)
    return Y


# ---------------------------------------------------------------------------
# New small-graph Adam kernel, vectorised per epoch
# ---------------------------------------------------------------------------
def adam_small_epoch(Y, graph, rows, sched, n, alpha, m, v, to_node_order,
                     beta1=0.9, beta2=0.99):
    eps, epns, eons, eonns = sched
    n_vertices = Y.shape[0]
    indices = graph.indices
    updates = np.zeros_like(Y)

    e_idx = np.flatnonzero(eons <= n)
    i, j = rows[e_idx], indices[e_idx]
    diff = Y[i] - Y[j]
    d2 = np.einsum("ij,ij->i", diff, diff)
    safe = np.where(d2 > 0, d2, 1.0)
    coeff = np.where(d2 > 0, -2 * A * B * safe ** (B - 1) / (A * safe**B + 1), 0.0)
    np.add.at(updates, i, 0.5 * coeff[:, None] * diff)  # SMALL_LAYOUT_ATTRACTION_SCALE
    eons[e_idx] += eps[e_idx]

    n_neg = np.trunc((n - eonns[e_idx]) / epns[e_idx]).astype(np.int64)
    counts = np.maximum(n_neg, 0)
    src = np.repeat(i, counts)
    e_rep = np.repeat(e_idx, counts)
    p = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    k = to_node_order[(e_rep * (n + p + 1)) % n_vertices]
    diff = Y[src] - Y[k]
    d2 = np.einsum("ij,ij->i", diff, diff)
    ok = d2 > 0
    d2 = np.where(ok, d2, 1.0)
    coeff = 2 * GAMMA * B / ((0.001 + d2) * (A * d2**B + 1))
    norm = np.sqrt(coeff**2 * d2)
    scale = GAMMA * np.tanh(norm / GAMMA) / norm
    np.add.at(updates, src[ok], (coeff * scale)[ok, None] * diff[ok])
    eonns[e_idx] += n_neg * epns[e_idx]

    nz = updates != 0.0
    m[nz] = beta1 * m[nz] + (1 - beta1) * updates[nz]
    v[nz] = beta2 * v[nz] + (1 - beta2) * updates[nz] ** 2
    m_hat = m / (1 - beta1 ** (n + 1))
    v_hat = v / (1 - beta2 ** (n + 1))
    step = np.where(nz, alpha * m_hat / (np.sqrt(v_hat) + 1e-4), 0.0)
    Y += step
    return updates, step


def run_adam(Y0, graph, n_epochs, seed=0):
    rows = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
    sched = schedules(graph.data, n_epochs, new_kernel=True)
    alphas = adam_alpha_schedule(n_epochs)
    Y = Y0.copy()
    m, v = np.zeros_like(Y), np.zeros_like(Y)
    random_state = np.random.RandomState(seed)
    to_node_order = np.arange(Y.shape[0])
    positions, grads, steps = [Y.copy()], [], []
    for n in range(n_epochs):
        g, s = adam_small_epoch(Y, graph, rows, sched, n, alphas[n], m, v, to_node_order)
        positions.append(Y.copy())
        grads.append(g)
        steps.append(s)
        random_state.shuffle(to_node_order)
    return np.array(positions), np.array(grads), np.array(steps), alphas


def main():
    OUT.mkdir(exist_ok=True)

    # Tiny graph: structure from the hand-placed positions themselves.
    tiny_graph = fuzzy_graph(TINY_POS, n_neighbors=4, seed=0)
    tiny_coo = tiny_graph.tocoo()
    upper = tiny_coo.row < tiny_coo.col
    tiny_edges = np.c_[tiny_coo.row[upper], tiny_coo.col[upper]]
    tiny_weights = tiny_coo.data[upper]

    # Old kernel, same seed, two thread interleavings.
    start = TINY_POS * TINY_EMB_SCALE + np.random.default_rng(1).normal(
        scale=0.6, size=TINY_POS.shape
    )
    tiny_run_1 = run_old_interleaved(start, prune(tiny_graph, 60), 60, interleave_seed=1)
    tiny_run_2 = run_old_interleaved(start, prune(tiny_graph, 60), 60, interleave_seed=2)

    # Toy world for the Adam trajectory.
    toy_x, toy_labels = make_blobs(
        n_samples=300,
        n_features=10,
        centers=4,
        cluster_std=[1.0, 1.6, 1.2, 2.0],
        random_state=3,
    )
    toy_graph = prune(fuzzy_graph(toy_x, n_neighbors=15, seed=0), 200)
    toy_init = np.random.RandomState(0).uniform(-10, 10, size=(300, 2))
    positions, grads, steps, alphas = run_adam(toy_init, toy_graph, 200, seed=0)

    # Follow a point that travels far but not pathologically.
    travel = np.linalg.norm(positions[-1] - positions[0], axis=1)
    candidates = np.argsort(travel)[-40:-10]
    path_length = np.linalg.norm(np.diff(positions[:, candidates], axis=0), axis=2).sum(0)
    toy_node = int(candidates[np.argmin(path_length / travel[candidates])])

    np.savez_compressed(
        OUT / "optimizers.npz",
        tiny_pos=TINY_POS,
        tiny_edges=tiny_edges,
        tiny_weights=tiny_weights,
        tiny_run_1=tiny_run_1,
        tiny_run_2=tiny_run_2,
        toy_positions=positions.astype(np.float32),
        toy_labels=toy_labels,
        toy_node_grad=grads[:, toy_node],
        toy_node_step=steps[:, toy_node],
        toy_alpha=alphas,
    )
    meta = {
        "tiny_source": TINY_SOURCE,
        "tiny_emb_scale": TINY_EMB_SCALE,
        "toy_node": toy_node,
        "toy_n_epochs": 200,
        # Warm full-pipeline MNIST timings from doc/optimizers.rst on the
        # 0.6dev branch (author's machine).  Replace with timings measured on
        # the presentation machine before the talk.
        "timings": {
            "compatibility_seconds": 41.9,
            "adam_seconds": 5.63,
            "source": "0.6dev doc/optimizers.rst, MNIST, warm",
        },
    }
    (OUT / "optimizers.json").write_text(json.dumps(meta, indent=2))

    runs_differ = np.abs(tiny_run_1 - tiny_run_2).max()
    print(f"tiny graph: {len(tiny_edges)} edges; old-kernel runs differ by up to {runs_differ:.3f}")
    print(f"toy node {toy_node}: travel {travel[toy_node]:.2f}, "
          f"|grad| range {np.linalg.norm(grads[:, toy_node], axis=1).min():.3f}"
          f"-{np.linalg.norm(grads[:, toy_node], axis=1).max():.3f}")


if __name__ == "__main__":
    main()
