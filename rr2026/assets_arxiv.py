"""ArXiv renders for the Payoff slides: optimization over wall-clock time,
and zooms into the final embeddings, for two rounds of comparison.

Three fits, each with the same fixed random seed and one shared kNN graph:

    old     compatibility mode: spectral init, the 0.5 optimizer, repulsion 1
    new     new stack: recursive init, Adam, repulsion 1, hard negatives off
    new_hn  new stack: recursive init, Adam, repulsion 8, hard negatives on

Rounds: old vs new, then new vs new_hn.

    python assets_arxiv.py --vectors arxiv_vectors.npy --labels arxiv_categories.npy
    python assets_arxiv.py --demo-mnist          # the whole pipeline on MNIST

Timing and frames. The clock starts when UMAP enters its embedding stage
(after the shared kNN graph and fuzzy graph are built), so it covers
initialization and optimization. After each epoch the hook may copy the full
embedding (float16) into a bounded buffer: when the buffer passes
--max-frames it keeps every other frame and doubles the capture interval, so
any run length ends up with evenly spaced snapshots. Time spent copying is
excluded from the clock. With a fixed seed the old stack runs single-threaded
(as UMAP 0.5 did); the new stack runs on all threads and stays reproducible.
Numba is warmed up on a small subset first, so compilation is not timed.

Every frame, in the timelapses and the zooms, is drawn from all the points
by the same render call; the last timelapse frame is the final layout, and
it is also the first zoom frame.

Other inputs
------------
--labels        .npy (or .txt) with one label per point; strings such as
                "cs.LG" are grouped by archive; nine largest groups coloured.
--window        negative_selection_range for new_hn (default 200000).
--epochs        epochs for every fit (default 200).
--zoom-label / --zoom-indices / --zoom-fraction / --max-zoom
                the zoom target (a set of papers, found in every embedding)
                and how far to zoom (default: 2% of points, at most 8x).
--time-frames   frames on the time grid per round (default 90).
--max-frames    snapshots kept per run (default 96; each is n x 2 float16).
--zoom-spread   pixels to grow each point by at the deepest zoom (default 1);
                it grows with the zoom factor from 0 at the full view.
--density       shade by density (a 256-step colormap) even when labels are
                given.
--keep-snapshots  also cache each run's full-resolution snapshots, so frames
                can be re-rendered with new settings without refitting.

Writes assets/arxiv/*.png and assets/arxiv.json; each run's frames and final
layout are cached in assets/arxiv_cache/ (delete it to refit).
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from assets_render import COLOR_CYCLE, OTHER_COLOR, render, square_extent

OUT = Path(__file__).parent / "assets"
CACHE = OUT / "arxiv_cache"
FRAMES = OUT / "arxiv"

CONFIGS = {
    "old": dict(
        label="old stack",
        detail="spectral init, 0.5 optimizer, repulsion 1",
        params=dict(
            compatibility_layout=True,
            init="spectral",
            optimizer="compatibility",
            repulsion_strength=1.0,
        ),
    ),
    "new": dict(
        label="new stack",
        detail="recursive init, Adam, repulsion 1",
        params=dict(
            compatibility_layout=False,
            init="recursive",
            optimizer="adam",
            repulsion_strength=1.0,
        ),
    ),  # window set to n: hard negatives off
    "new_hn": dict(
        label="new stack, repulsion 8",
        detail="hard negatives on",
        params=dict(
            compatibility_layout=False,
            init="recursive",
            optimizer="adam",
            repulsion_strength=8.0,
        ),
    ),  # window set from --window
}
ROUNDS = [
    dict(left="old", right="new", title="old stack vs new stack"),
    dict(left="new", right="new_hn", title="turning up repulsion"),
]


# ---------------------------------------------------------------------------
# Timed fits
# ---------------------------------------------------------------------------
class EpochClock(list):
    """Stands in for n_epochs inside optimize_layout_euclidean.

    The layout loop asks ``n in epochs_list`` after every epoch. This copies
    the live embedding into a bounded buffer whenever ``interval`` seconds have
    passed, then answers no, so UMAP never stores copies itself. When the
    buffer passes ``max_frames`` it keeps every other frame and doubles the
    interval. Copying time is excluded from the clock. ``max()`` of the list is
    the epoch count, so the optimization itself is unchanged.
    """

    def __init__(self, n_epochs, embedding, t0, max_frames, interval=0.02):
        super().__init__([n_epochs])
        self.embedding, self.t0 = embedding, t0
        self.max_frames, self.interval = max_frames, interval
        self.paused = 0.0
        self.times, self.frames = [], []
        self.last_capture = -np.inf
        self.last_epoch = 0.0

    def now(self):
        return time.perf_counter() - self.t0 - self.paused

    def __contains__(self, n):
        t = self.now()
        self.last_epoch = t
        if t - self.last_capture >= self.interval:
            start = time.perf_counter()
            self.frames.append(self.embedding.astype(np.float16))
            self.times.append(t)
            self.last_capture = t
            if len(self.frames) > self.max_frames:
                self.frames, self.times = self.frames[::2], self.times[::2]
                self.interval *= 2
                self.last_capture = self.times[-1]
            self.paused += time.perf_counter() - start
        return False


def timed_fit(X, knn, params, epochs, seed, max_frames):
    import umap
    import umap.umap_ as um

    record = {}
    original_embed, original_opt = (
        um.simplicial_set_embedding,
        um.optimize_layout_euclidean,
    )

    def embed(*args, **kwargs):
        record["t0"] = time.perf_counter()
        return original_embed(*args, **kwargs)

    def optimize(*args, **kwargs):
        record["init_time"] = time.perf_counter() - record["t0"]
        start = time.perf_counter()
        record["init"] = args[0].astype(np.float16)
        clock = EpochClock(
            args[4], args[0], record["t0"] + (time.perf_counter() - start), max_frames
        )
        record["clock"] = clock
        out = original_opt(*args[:4], clock, *args[5:], **kwargs)
        return out[-1] if isinstance(out, list) else out

    um.simplicial_set_embedding, um.optimize_layout_euclidean = embed, optimize
    try:
        final = umap.UMAP(
            random_state=seed, n_epochs=epochs, precomputed_knn=knn, **params
        ).fit_transform(X)
    finally:
        um.simplicial_set_embedding, um.optimize_layout_euclidean = (
            original_embed,
            original_opt,
        )
    clock = record["clock"]
    return dict(
        final=final.astype(np.float32),
        init=record["init"],
        init_time=record["init_time"],
        times=clock.times,
        frames=clock.frames,
        done=clock.last_epoch,
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def normaliser(Y):
    """Centre and scale (median, 90th-percentile radius) as a function."""
    centre = np.median(Y, axis=0)
    radius = np.quantile(np.linalg.norm(Y - centre, axis=1), 0.9)
    return lambda P: (P - centre) / max(radius, 1e-12)


def rotation(source, target, reflect=False):
    u, _, vt = np.linalg.svd(source.T @ target)
    R = u @ vt
    if not reflect and np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return R


def load_labels(path, n):
    if path is None:
        return None
    path = Path(path)
    labels = (
        np.load(path, allow_pickle=True)
        if path.suffix == ".npy"
        else np.array(path.read_text().splitlines())
    )
    if len(labels) != n:
        raise ValueError(f"{len(labels)} labels for {n} points")
    return labels


def group_labels(labels, n_colors=9):
    if labels.dtype.kind in "US" or labels.dtype == object:
        groups = np.array([str(l).split(".")[0] for l in labels])
    else:
        groups = labels.astype(str)
    names, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    keep = np.argsort(-counts)[:n_colors]
    code_of = np.full(len(names), len(keep), dtype=np.int64)
    code_of[keep] = np.arange(len(keep))
    shown, colors = [str(names[k]) for k in keep], list(COLOR_CYCLE[: len(keep)])
    if len(names) > len(keep):
        shown.append("other")
        colors.append(OTHER_COLOR)
    return code_of[inverse], shown, colors


def zoom_members(labels, reference, zoom_label, zoom_indices, fraction, seed=0):
    if zoom_indices is not None:
        return np.asarray(zoom_indices)
    if labels is not None:
        values, counts = np.unique(labels.astype(str), return_counts=True)
        if zoom_label is None:
            zoom_label = values[np.argmin(np.abs(counts - fraction * len(labels)))]
        members = np.flatnonzero(labels.astype(str) == str(zoom_label))
        if len(members):
            return members
    rng = np.random.default_rng(seed)
    centre = np.median(reference, axis=0)
    near = np.argsort(np.linalg.norm(reference - centre, axis=1))[
        : max(1, len(reference) // 20)
    ]
    seed_point = reference[rng.choice(near)]
    k = max(50, int(fraction * len(reference)))
    return np.argsort(np.linalg.norm(reference - seed_point, axis=1))[:k]


def zoom_plan(Y, members, n_frames, max_zoom, pad=1.4):
    """Square extents from the full view down to the target (geometric), and
    each frame's zoom factor."""
    x0, x1, y0, y1 = square_extent(Y)
    c0, h0 = np.array([(x0 + x1) / 2, (y0 + y1) / 2]), (x1 - x0) / 2
    c1 = np.median(Y[members], axis=0)
    h1 = max(pad * np.quantile(np.abs(Y[members] - c1).max(axis=1), 0.9), h0 / max_zoom)
    plan = []
    for t in np.linspace(0, 1, n_frames):
        h = h0 * (h1 / h0) ** t
        progress = (h0 - h) / (h0 - h1) if h0 != h1 else t
        c = c0 + (c1 - c0) * progress
        plan.append(((c[0] - h, c[0] + h, c[1] - h, c[1] + h), h0 / h))
    box = [
        (c1[0] - h1 - x0) / (2 * h0),
        (c1[0] + h1 - x0) / (2 * h0),
        (c1[1] - h1 - y0) / (2 * h0),
        (c1[1] + h1 - y0) / (2 * h0),
    ]
    return plan, box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors")
    parser.add_argument("--labels")
    parser.add_argument("--demo-mnist", action="store_true")
    parser.add_argument(
        "--mnist-pickle", default=None, help="for --demo-mnist without OpenML"
    )
    parser.add_argument("--window", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zoom-label")
    parser.add_argument("--zoom-indices")
    parser.add_argument("--zoom-fraction", type=float, default=0.02)
    parser.add_argument("--max-zoom", type=float, default=8.0)
    parser.add_argument("--time-frames", type=int, default=90)
    parser.add_argument("--zoom-frames", type=int, default=120)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--zoom-spread", type=int, default=1)
    parser.add_argument("--keep-snapshots", action="store_true")
    parser.add_argument("--density", action="store_true")
    parser.add_argument("--size", type=int, default=1152)
    args = parser.parse_args()
    for folder in (CACHE, FRAMES):
        folder.mkdir(parents=True, exist_ok=True)

    title = "ArXiv"
    if args.demo_mnist:
        if args.mnist_pickle:
            import gzip
            import pickle

            with gzip.open(args.mnist_pickle, "rb") as f:
                parts = pickle.load(f, encoding="latin1")
            X = np.concatenate([p[0] for p in parts]).astype(np.float32)
            labels = np.concatenate([p[1] for p in parts]).astype(int)
        else:
            from sklearn.datasets import fetch_openml

            X, labels = fetch_openml(
                "mnist_784", version=1, return_X_y=True, as_frame=False
            )
            X, labels = X.astype(np.float32), labels.astype(int)
        title, window = "MNIST (demo of the ArXiv slides)", 20_000
    elif args.vectors:
        X = np.asarray(np.load(args.vectors), dtype=np.float32)
        labels = load_labels(args.labels, len(X))
        window = args.window
    else:
        parser.error("give --vectors (with optional --labels) or --demo-mnist")
    n = len(X)
    CONFIGS["new"]["params"]["negative_selection_range"] = n
    CONFIGS["new_hn"]["params"]["negative_selection_range"] = window

    # Shared kNN graph, then warm up numba on a small subset (not timed).
    from umap.umap_ import nearest_neighbors

    knn_file = CACHE / "knn.npz"
    if knn_file.exists():
        k = np.load(knn_file)
        knn_i, knn_d = k["i"], k["d"]
    else:
        knn_i, knn_d, _ = nearest_neighbors(
            X, 15, "euclidean", {}, False, np.random.RandomState(args.seed)
        )
        np.savez(knn_file, i=knn_i, d=knn_d)
    rng = np.random.default_rng(0)

    codes, names, colors = (
        group_labels(labels)
        if labels is not None and not args.density
        else (None, [], [])
    )
    rng_align = np.random.default_rng(1)
    align = np.sort(
        rng_align.choice(n, min(50_000, n), replace=False)
    )  # points used to fit rotations

    def draw(Y, extent, path, zoom=1.0):
        """The one render call used for every frame. ``zoom`` is how far this
        frame is magnified relative to the full view: points grow from 0 px at
        the full view to --zoom-spread px at the deepest zoom."""
        depth = np.log(max(zoom, 1.0)) / np.log(max(args.max_zoom, 1.0 + 1e-9))
        spread = int(round(args.zoom_spread * min(depth, 1.0)))
        render(Y, codes, colors, extent, size=args.size, spread_px=spread).save(path)

    def render_run(name, snapshots, final):
        """Timelapse frames for one run: each snapshot normalised and rotated
        onto the final layout, then the final layout itself as the last frame."""
        extent = square_extent(final)
        count = 0
        for k, Y in enumerate(snapshots):
            Y = Y.astype(np.float32)
            Y = normaliser(Y)(Y)
            draw(
                Y @ rotation(Y[align], final[align]),
                extent,
                FRAMES / f"{name}_run_{k:03d}.png",
            )
            count = k + 1
        draw(final, extent, FRAMES / f"{name}_run_{count:03d}.png")

    # 'new' runs first: every other run is rotated onto its final layout.
    runs, reference = {}, None
    for name in ("new", "old", "new_hn"):
        cfg = CONFIGS[name]
        info_path, final_path = CACHE / f"{name}.json", CACHE / f"{name}_final.npy"
        if not info_path.exists():
            small = np.sort(rng.choice(n, min(20_000, n), replace=False))
            import umap

            warm = dict(
                cfg["params"],
                negative_selection_range=min(
                    len(small),
                    cfg["params"].get("negative_selection_range", len(small)),
                ),
            )
            umap.UMAP(random_state=args.seed, n_epochs=10, **warm).fit(X[small])
            run = timed_fit(
                X,
                (knn_i, knn_d, None),
                cfg["params"],
                args.epochs,
                args.seed,
                args.max_frames,
            )
            final = normaliser(run["final"])(run["final"])
            if reference is not None:
                final = final @ rotation(final[align], reference[align], reflect=True)
            np.save(final_path, final.astype(np.float32))
            snapshots = [run["init"]] + list(run["frames"])
            times = (
                [float(run["init_time"])]
                + [float(t) for t in run["times"]]
                + [float(run["done"])]
            )
            if args.keep_snapshots:
                np.save(
                    CACHE / f"{name}_snapshots.npy",
                    np.stack(snapshots).astype(np.float16),
                )
            render_run(name, snapshots, final)
            info_path.write_text(
                json.dumps(
                    dict(
                        times=times,
                        init_time=float(run["init_time"]),
                        done=float(run["done"]),
                    )
                )
            )
            print(
                f"{name}: init {run['init_time']:.1f}s, done {run['done']:.1f}s, {len(times)} frames",
                flush=True,
            )
        elif (CACHE / f"{name}_snapshots.npy").exists():
            # cached run with snapshots: re-render its timelapse with the current settings
            render_run(
                name,
                np.load(CACHE / f"{name}_snapshots.npy", mmap_mode="r"),
                np.load(final_path),
            )
        runs[name] = json.loads(info_path.read_text())
        runs[name]["final"] = np.load(final_path)
        if reference is None:
            reference = runs[name]["final"]

    # Timelapses: one uniform time grid per round; equal steps are equal seconds.
    meta_rounds = []
    blank = FRAMES / "blank.png"
    from PIL import Image

    Image.new("RGB", (args.size, args.size), "white").save(
        blank
    )  # a run still initializing
    for r, rnd in enumerate(ROUNDS):
        t_max = max(runs[rnd["left"]]["done"], runs[rnd["right"]]["done"])
        grid = np.linspace(0, t_max, args.time_frames)
        for side in ("left", "right"):
            name = rnd[side]
            times = np.array(runs[name]["times"])
            for k, t in enumerate(grid):
                source = (
                    blank
                    if t < times[0]
                    else FRAMES
                    / f"{name}_run_{int(np.searchsorted(times, t, side='right') - 1):03d}.png"
                )
                if k == len(grid) - 1 and t >= times[-1] - 1e-9:
                    source = FRAMES / f"{name}_run_{len(times) - 1:03d}.png"
                shutil.copyfile(source, FRAMES / f"round{r}_{side}_time_{k:03d}.png")
        meta_rounds.append(dict(rnd, grid=grid.tolist()))
        print(f"round {r}: {args.time_frames} time frames", flush=True)

    # Zooms into the final layouts (a set of papers as the target), same render call.
    members = zoom_members(
        labels,
        runs["new"]["final"],
        args.zoom_label,
        np.load(args.zoom_indices) if args.zoom_indices else None,
        args.zoom_fraction,
    )
    boxes = {}
    for name in CONFIGS:
        plan, box = zoom_plan(
            runs[name]["final"], members, args.zoom_frames, args.max_zoom
        )
        boxes[name] = [float(v) for v in box]
        for k, (extent, factor) in enumerate(plan):
            draw(
                runs[name]["final"],
                extent,
                FRAMES / f"{name}_zoom_{k:03d}.png",
                zoom=factor,
            )
        print(f"{name}: {args.zoom_frames} zoom frames", flush=True)

    meta = {
        "title": title,
        "n": int(n),
        "window": int(window),
        "epochs": args.epochs,
        "seed": args.seed,
        "configs": {
            name: dict(
                label=c["label"],
                detail=c["detail"],
                init_time=runs[name]["init_time"],
                done=runs[name]["done"],
            )
            for name, c in CONFIGS.items()
        },
        "rounds": meta_rounds,
        "time_frames": args.time_frames,
        "zoom_frames": args.zoom_frames,
        "zoom_box": boxes,
        "zoom_members": int(len(members)),
        "groups": names,
        "colors": colors,
        "size": args.size,
    }
    (OUT / "arxiv.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("title", "n", "configs")}, indent=2))


if __name__ == "__main__":
    main()
