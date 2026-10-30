"""Precompute data for the RecursiveInit slides.

    python assets_recursive.py              # MNIST via OpenML + UMAP 0.6dev
    python assets_recursive.py --synthetic  # quick stand-in, no downloads
    python assets_recursive.py --coarsening transition   # transition coarsening

Writes assets/recursive.npz and assets/recursive.json, or with
``--coarsening transition`` assets/recursive_transition.npz / .json (for
recursive_init_transition.py). Transition coarsening needs the label_prop
patch that adds coarse_layout_weights / coarsen_on_layout_weights.

Three parts, all using the real code on the 0.6dev branch:

* toy: one coarsening level on a ~120-node graph, calling umap.label_prop's
  own helpers step by step (hub seeding, each label-propagation pass,
  outlier labelling, fuzzy-union coarsening, a coarse Adam layout,
  expansion) and then the fine-graph optimization that follows;
* trace: a full UMAP fit with init="recursive", instrumented to record the
  part labels, coarse inputs and optimized coarse layouts at every level,
  turned into per-point keyframes for a coarse-to-fine morph of all points;
* comparison: the same fit with init="spectral", with warm timings of each
  initializer, their starting layouts, epoch snapshots, and quality metrics.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

import umap
import umap.label_prop as lp
import umap.umap_ as umap_module
from umap.layouts import optimize_layout_euclidean
from umap.umap_ import fuzzy_simplicial_set
from umap.utils import make_epochs_per_sample

OUT = Path(__file__).parent / "assets"
A, B = 1.5769434602697652, 0.8950608778515733
INT32_MIN, INT32_MAX = np.iinfo(np.int32).min + 1, np.iinfo(np.int32).max - 1
SNAPSHOT_EPOCHS = [0, 1, 2, 4, 7, 11, 17, 25, 36, 50, 70, 95, 125, 160, 199]


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------
def normalise(Y, q=0.9):
    """Centre on the median and scale so the q-quantile radius is 1."""
    Y = np.asarray(Y, dtype=np.float64)
    Y = Y - np.median(Y, axis=0)
    radius = np.quantile(np.linalg.norm(Y, axis=1), q)
    return Y / max(radius, 1e-12)


def rotation_to(source, target, allow_reflection=False):
    """Orthogonal matrix R minimising ||source @ R - target||."""
    u, _, vt = np.linalg.svd(source.T @ target)
    R = u @ vt
    if not allow_reflection and np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return R


def similarity_to(source, target):
    """Scale, rotation and shift mapping ``source`` onto ``target``."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    S, T = source - mu_s, target - mu_t
    R = rotation_to(S, T)
    scale = np.trace((S @ R).T @ T) / np.trace(S.T @ S)
    return lambda X: (X - mu_s) @ R * scale + mu_t


# ---------------------------------------------------------------------------
# Part 1: toy coarsening, one level, with the library's own helpers
# ---------------------------------------------------------------------------
def toy_points(seed=4):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.15, 0.85, 40) * np.pi
    arc = np.c_[3.2 * np.cos(t) - 3.5, 2.0 * np.sin(t) - 0.6] + rng.normal(
        scale=0.12, size=(40, 2)
    )
    blob = rng.normal(size=(32, 2)) * [0.65, 0.5] + [1.6, 1.3]
    bar = rng.normal(size=(30, 2)) * [1.1, 0.3] + [1.9, -1.6]
    small = rng.normal(size=(18, 2)) * 0.35 + [-2.4, -2.2]
    return np.concatenate([arc, blob, bar, small])


def transition_coarse_graph(graph, labels):
    """Transition-coarsened graph over parts: W = R^T W_fine R (total weight
    between parts), each part's share f_PQ = W_PQ / sum_Q' W_PQ' of its
    connections to other parts, and w_PQ = f_PQ + f_QP - f_PQ f_QP."""
    from scipy.sparse import csr_matrix, diags

    n, k = len(labels), int(labels.max()) + 1
    R = csr_matrix((np.ones(n), labels, np.arange(n + 1)), shape=(n, k))
    W = (R.T @ graph @ R).tocsr().astype(np.float64)
    W = (W - diags(W.diagonal())).tocsr()
    W.eliminate_zeros()
    out = np.asarray(W.sum(axis=1)).ravel()
    f = diags(1.0 / np.where(out > 0, out, 1.0)) @ W
    coarse = (f + f.T - f.multiply(f.T)).tocsr()
    coarse.sort_indices()
    coarse.data = np.clip(coarse.data, 0.0, 1.0).astype(np.float32)
    return coarse


def toy_part(coarsening="union"):
    P = toy_points()
    graph, _, _ = fuzzy_simplicial_set(P, 8, np.random.RandomState(0), "euclidean")
    graph = graph.tocsr()
    graph.sort_indices()
    n = graph.shape[0]
    n_parts = n // 4  # the ratio-4 rule used below the top level

    random_state = np.random.RandomState(0)
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    labels = np.full(n, -1, dtype=np.int32)
    degrees = np.squeeze(np.asarray(graph.sum(axis=1)))
    labels = lp.initialize_labels_from_hubs(labels, n_parts, degrees)
    seeds = np.flatnonzero(labels >= 0)

    # Label propagation, pass by pass, with label_propagation_init's stopping rule.
    passes = [labels.copy()]
    prev_unlabeled = np.sum(labels < 0)
    for i in range(100):
        labels = lp.label_prop_iteration(
            graph.indptr, graph.indices, graph.data, labels, rng_state
        )
        passes.append(labels.copy())
        if i % 5 == 0:
            unlabeled = np.sum(labels < 0)
            if unlabeled == 0 or unlabeled == prev_unlabeled:
                break
        prev_unlabeled = unlabeled
    # Drop trailing passes that changed nothing.
    while len(passes) > 1 and np.array_equal(passes[-1], passes[-2]):
        passes.pop()
    labels = lp.label_outliers(graph.indptr, graph.indices, labels.copy(), rng_state)
    labels = lp.remap_labels(labels.copy())
    if coarsening == "transition":
        coarse = transition_coarse_graph(graph, labels)
    else:
        _, coarse = lp._coarsen_graph(graph, labels)
        coarse = coarse.tocsr()
    n_coarse = coarse.shape[0]

    # Base case: parts start at the mean of their members, rescaled as in
    # label_propagation_init's base branch (one level down, depth 2).
    centroids = np.array([P[labels == k].mean(0) for k in range(n_coarse)])
    base = centroids - centroids.mean(0)
    spread = np.quantile(base, 0.95, 0) - np.quantile(base, 0.05, 0)
    base = (base * (np.log10(n_coarse) * 3 * np.log2(3)) / spread).astype(np.float32)

    # Coarse layout as in label_propagation_init (gamma 4, one negative per edge).
    a1, b1 = lp._initial_curve_parameters(A, B, "strong_to_one")
    coarse_epochs = 64
    coarse_layout = optimize_layout_euclidean(
        base.copy(),
        base.copy(),
        None,
        None,
        coarse_epochs,
        n_coarse,
        make_epochs_per_sample(coarse.data, coarse_epochs),
        a1,
        b1,
        random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64),
        4.0,
        0.5,
        1,
        parallel=False,
        verbose=False,
        densmap_kwds={},
        move_other=False,
        csr_indptr=coarse.indptr,
        csr_indices=coarse.indices,
        csr_data=coarse.data,
        random_state=random_state,
        optimizer="adam",
        good_initialization=False,
        negative_selection_range=lp._resolve_negative_selection_range(
            n_coarse, "scaled_coarse", 0.5
        ),
    )
    expanded = lp._expand_layout(
        graph.indptr, graph.indices, graph.data, labels, coarse_layout
    )
    expanded = (expanded - expanded.mean(0)).astype(np.float32)

    # The fine graph then refines from the expanded layout.
    fine_epochs = 120
    settled = optimize_layout_euclidean(
        expanded.copy(),
        expanded.copy(),
        None,
        None,
        fine_epochs,
        n,
        make_epochs_per_sample(graph.data, fine_epochs),
        A,
        B,
        random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64),
        1.0,
        1.0,
        5,
        parallel=False,
        verbose=False,
        densmap_kwds={},
        move_other=False,
        csr_indptr=graph.indptr,
        csr_indices=graph.indices,
        csr_data=graph.data,
        random_state=random_state,
        optimizer="adam",
        good_initialization=True,
        negative_selection_range=n,
    )

    # Display coordinates: everything expressed in the toy's own frame.
    to_display = similarity_to(base, centroids)  # coarse coords -> data frame
    coarse_disp = to_display(coarse_layout)
    expanded_disp = to_display(expanded)
    settled_disp = similarity_to(settled, expanded_disp)(settled)

    coo = graph.tocoo()
    upper = coo.row < coo.col
    ccoo = coarse.tocoo()
    cupper = ccoo.row < ccoo.col
    return {
        "toy_pos": P,
        "toy_edges": np.c_[coo.row[upper], coo.col[upper]],
        "toy_weights": coo.data[upper],
        "toy_seeds": seeds,
        "toy_passes": np.array(passes),
        "toy_labels": labels,
        "toy_coarse_edges": np.c_[ccoo.row[cupper], ccoo.col[cupper]],
        "toy_coarse_weights": ccoo.data[cupper],
        "toy_centroids": centroids,
        "toy_coarse_layout": coarse_disp,
        "toy_expanded": expanded_disp,
        "toy_settled": settled_disp,
    }


# ---------------------------------------------------------------------------
# Part 2 and 3: instrumented fits
# ---------------------------------------------------------------------------
class Recorder:
    """Wraps library functions to record what the recursion does."""

    def __init__(self):
        self.labels, self.coarse_calls, self.main_init = [], [], None
        self.timings = {}
        self._saved = []

    def patch(self, module, name, wrapper):
        original = getattr(module, name)
        self._saved.append((module, name, original))
        setattr(module, name, wrapper(original))

    def restore(self):
        for module, name, original in reversed(self._saved):
            setattr(module, name, original)
        self._saved = []

    def install(self, trace):
        def timed(key):
            def wrap(fn):
                def inner(*args, **kwargs):
                    start = time.perf_counter()
                    out = fn(*args, **kwargs)
                    self.timings[key] = time.perf_counter() - start
                    return out

                return inner

            return wrap

        def capture_main(fn):
            def inner(*args, **kwargs):
                self.main_init = np.array(args[0], copy=True)
                return fn(*args, **kwargs)

            return inner

        self.patch(umap_module, "recursive_init", timed("recursive"))
        self.patch(umap_module, "spectral_layout", timed("spectral"))
        self.patch(umap_module, "optimize_layout_euclidean", capture_main)
        if not trace:
            return

        def capture_labels(fn):
            def inner(labels):
                out = fn(labels)
                self.labels.append(np.array(out, copy=True))
                return out

            return inner

        def capture_coarse(fn):
            def inner(*args, **kwargs):
                init = np.array(args[0], copy=True)
                out = fn(*args, **kwargs)
                self.coarse_calls.append(
                    dict(
                        init=init,
                        out=np.array(out, copy=True),
                        n=int(args[5]),
                        a=float(args[7]),
                        b=float(args[8]),
                        epochs=int(args[4]),
                    )
                )
                return out

            return inner

        self.patch(lp, "remap_labels", capture_labels)
        self.patch(lp, "optimize_layout_euclidean", capture_coarse)


TRANSITION_OPTIONS = dict(
    coarse_layout_weights="transition", coarsen_on_layout_weights=True
)
UNION_OPTIONS = dict(
    coarse_layout_weights="fuzzy_union", coarsen_on_layout_weights=False
)


def fit(X, init, trace=False, snapshots=True, seed=0, recursive_options=None):
    import functools

    original = umap_module.recursive_init
    if recursive_options:
        umap_module.recursive_init = functools.partial(
            lp.recursive_init, **recursive_options
        )
    rec = Recorder()
    rec.install(trace)
    try:
        reducer = umap.UMAP(
            random_state=seed,
            compatibility_layout=False,
            optimizer="adam",
            init=init,
            n_epochs=list(SNAPSHOT_EPOCHS) if snapshots else None,
        )
        embedding = reducer.fit_transform(X)
    finally:
        rec.restore()
        umap_module.recursive_init = original
    snaps = getattr(reducer, "embedding_list_", None)
    return reducer, embedding, rec, snaps


def morph_keyframes(rec, final_init, final_embedding, seed=0):
    """Per-point keyframes, coarsest level first, for the coarse-to-fine morph."""
    n = len(final_init)
    ancestors = [np.arange(n)]
    for labels in rec.labels:  # depth 1, 2, ... (coarsening order)
        ancestors.append(labels[ancestors[-1]])
    by_size = {call["n"]: call for call in rec.coarse_calls}
    level_sizes = [n] + [int(lab.max()) + 1 for lab in rec.labels]

    frames, jitter_scale, names = [], [], []
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    for depth in range(len(rec.labels), 0, -1):  # coarsest first
        call = by_size[level_sizes[depth]]
        anc = ancestors[depth]
        counts = np.bincount(anc, minlength=level_sizes[depth])
        sigma = 0.35 * np.sqrt(counts[anc] / n)
        for key in ("init", "out"):
            frames.append(call[key][anc])
            jitter_scale.append(sigma)
            names.append(f"{level_sizes[depth]}:{key}")
    frames += [final_init, final_embedding]
    jitter_scale += [np.zeros(n), np.zeros(n)]
    names += [f"{n}:init", f"{n}:final"]

    aligned = [normalise(frames[0])]
    for frame in frames[1:]:
        current = normalise(frame)
        aligned.append(current @ rotation_to(current, aligned[-1]))
    keyframes = np.stack([f + s[:, None] * z for f, s in zip(aligned, jitter_scale)])
    curves = [
        (c["n"], c["a"], c["b"], c["epochs"])
        for c in sorted(rec.coarse_calls, key=lambda c: c["n"])
    ]
    return keyframes.astype(np.float32), names, level_sizes, curves


def run_keyframes(init, snaps):
    """Normalised, rotation-aligned keyframes: init followed by snapshots."""
    frames = [init] + list(snaps)
    aligned = [normalise(frames[0])]
    for frame in frames[1:]:
        current = normalise(frame)
        aligned.append(current @ rotation_to(current, aligned[-1]))
    return np.stack(aligned).astype(np.float32)


def travel(init, final):
    """Mean distance points move from start to finish, in layout radii."""
    a, b = normalise(init), normalise(final)
    a = a @ rotation_to(a, b, allow_reflection=True)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def quality(X, labels, layouts, k=10, n_queries=2000, chunk=200, seed=0):
    """Trustworthiness and kNN accuracy for several layouts of the same data.

    Trustworthiness is computed exactly for a fixed random set of query points
    against all points (ranks in the original space), normalised per query.
    kNN accuracy is the leave-one-out k-nearest-neighbour classifier accuracy
    of the labels in the layout, on the same queries.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    queries = rng.choice(n, n_queries, replace=False)
    X = np.asarray(X, dtype=np.float32)
    sq = np.einsum("ij,ij->i", X, X)
    neighbours = {}
    for name, Y in layouts.items():
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Y)
        neighbours[name] = nn.kneighbors(Y[queries], return_distance=False)[:, 1:]
    penalty = {name: 0.0 for name in layouts}
    for start in range(0, n_queries, chunk):
        q = queries[start : start + chunk]
        D = sq[q, None] + sq[None, :] - 2.0 * X[q] @ X.T
        D[np.arange(len(q)), q] = -np.inf  # the query itself ranks first
        for name in layouts:
            emb = neighbours[name][start : start + chunk]
            d_emb = np.take_along_axis(D, emb, axis=1)
            ranks = (D[:, None, :] < d_emb[:, :, None]).sum(
                axis=2
            )  # self counted: 1-based rank
            penalty[name] += np.maximum(ranks - k, 0).sum()
    scale = 2.0 / (n_queries * k * (2 * n - 3 * k - 1))
    results = {}
    for name in layouts:
        votes = labels[neighbours[name]]
        predicted = np.array([np.bincount(row).argmax() for row in votes])
        results[name] = {
            "trustworthiness": float(1.0 - scale * penalty[name]),
            "knn_accuracy": float(np.mean(predicted == labels[queries])),
        }
    return results


def load_data(synthetic, mnist_pickle=None):
    if mnist_pickle:
        import gzip
        import pickle

        with gzip.open(mnist_pickle, "rb") as f:
            parts = pickle.load(f, encoding="latin1")
        return (
            np.concatenate([p[0] for p in parts]).astype(np.float32),
            np.concatenate([p[1] for p in parts]).astype(int),
        )
    if synthetic:
        from sklearn.datasets import make_blobs

        X, y = make_blobs(
            n_samples=70_000, n_features=40, centers=10, cluster_std=4.0, random_state=0
        )
        return X.astype(np.float32), y
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    return X.astype(np.float32), y.astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic", action="store_true", help="use a stand-in for MNIST"
    )
    parser.add_argument(
        "--mnist-pickle", help="path to mnist.pkl.gz (Nielsen format) instead of OpenML"
    )
    parser.add_argument(
        "--coarsening", choices=("union", "transition"), default="union"
    )
    args = parser.parse_args()
    import inspect

    patched = "coarse_layout_weights" in inspect.signature(lp.recursive_init).parameters
    if args.coarsening == "transition":
        if not patched:
            raise SystemExit(
                "--coarsening transition needs the label_prop transition-coarsening patch"
            )
        options = TRANSITION_OPTIONS
    else:
        # With the patch installed, ask for the released behaviour explicitly.
        options = UNION_OPTIONS if patched else None
    stem = "recursive_transition" if args.coarsening == "transition" else "recursive"
    OUT.mkdir(exist_ok=True)

    print("toy coarsening ...", flush=True)
    toy = toy_part(args.coarsening)

    X, labels = load_data(args.synthetic, args.mnist_pickle)
    print("warm-up fits (numba compilation) ...", flush=True)
    small = np.random.default_rng(0).choice(len(X), 6000, replace=False)
    for init in ("recursive", "spectral"):
        fit(X[small], init, snapshots=False, recursive_options=options)

    print("recursive fit ...", flush=True)
    _, rec_final, rec, rec_snaps = fit(
        X, "recursive", trace=True, recursive_options=options
    )
    rec_init = rec.main_init
    print("spectral fit ...", flush=True)
    _, spec_final, spec, spec_snaps = fit(X, "spectral")
    spec_init = spec.main_init

    print("keyframes and metrics ...", flush=True)
    morph, names, level_sizes, curves = morph_keyframes(rec, rec_init, rec_final)
    metrics = quality(
        X,
        labels,
        {
            "recursive_init": rec_init,
            "spectral_init": spec_init,
            "recursive_final": rec_final,
            "spectral_final": spec_final,
        },
    )

    np.savez_compressed(
        OUT / f"{stem}.npz",
        **toy,
        labels=labels,
        morph=morph,
        rec_run=run_keyframes(rec_init, rec_snaps),
        spec_run=run_keyframes(spec_init, spec_snaps),
    )
    meta = {
        "synthetic": bool(args.synthetic),
        "coarsening": args.coarsening,
        "n": int(len(X)),
        "level_sizes": level_sizes,
        "morph_frames": names,
        "coarse_curves": [
            {"n": c[0], "a": c[1], "b": c[2], "epochs": c[3]} for c in curves
        ],
        "final_curve": {"a": A, "b": B},
        "init_seconds": {
            "recursive": rec.timings.get("recursive"),
            "spectral": spec.timings.get("spectral"),
        },
        "travel": {
            "recursive": travel(rec_init, rec_final),
            "spectral": travel(spec_init, spec_final),
        },
        "metrics": metrics,
        "snapshot_epochs": SNAPSHOT_EPOCHS,
    }
    (OUT / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
