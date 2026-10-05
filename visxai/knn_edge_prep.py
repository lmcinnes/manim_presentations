"""
Shared analysis for the nearest-neighbour edge showcase slides
(mnist_knn_prep.py, quickdraw_knn_prep.py).

Given features (the space in which "nearest neighbour" is defined), labels,
class names and 28x28 images, this computes:

  * each item's first nearest neighbour (1-NN) in feature space
  * a 2D UMAP of the features
  * the undirected 1-NN edge list, its length in the UMAP layout, and which
    edges are "long" (long in the layout AND joining different map regions)
  * ranked showcase edges: long, between visually distinct classes, with both
    endpoints typical members of their own class
  * a contact sheet PNG of the top candidates for hand-picking

and writes everything to one .npz read by knn_edge_slides.py.
"""

import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def add_analysis_args(p):
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument(
        "--long-frac",
        type=float,
        default=0.12,
        help="an edge is 'long' if its UMAP length exceeds this "
        "fraction of the layout's diagonal (and it joins "
        "different map regions)",
    )
    p.add_argument(
        "--feature-purity",
        type=float,
        default=0.8,
        help="min fraction of an endpoint's 2nd..11th feature-space "
        "neighbours sharing its label",
    )
    p.add_argument(
        "--embed-purity",
        type=float,
        default=0.9,
        help="min fraction of an endpoint's 15 UMAP neighbours "
        "sharing its label (i.e. it sits inside its cluster)",
    )
    p.add_argument(
        "--n-show", type=int, default=5, help="number of showcase edges to pre-select"
    )
    p.add_argument(
        "--n-sheet", type=int, default=40, help="candidates on the contact sheet"
    )
    return p


def label_purity(neigh_idx, labels):
    """Fraction of each row's neighbours sharing that point's label."""
    return (labels[neigh_idx] == labels[:, None]).mean(1)


def neighbours_excluding_self(features, k):
    """Exact k nearest neighbours of every row, excluding the row itself
    (robust to exact duplicates, where self may not come first)."""
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=-1).fit(features)
    _, idx = nn.kneighbors(features)
    out = np.empty((len(features), k), dtype=np.int64)
    for i in range(len(features)):
        r = idx[i][idx[i] != i]
        out[i] = r[:k]
    return out


def analyse_and_save(
    features, labels, class_names, images, out, a, feature_space="", source_index=None
):
    """features: (N, F) float; labels: (N,) int in [0, len(class_names));
    images: (N, 784) uint8, ink = high values."""
    labels = np.asarray(labels, int)
    N, n_cls = len(labels), len(class_names)
    features = np.asarray(features, np.float32)

    # ---- feature-space neighbours (1-NN plus 10 more for purity) ----
    print(f"feature-space kNN on {N} x {features.shape[1]} ...")
    t0 = time.time()
    idx = neighbours_excluding_self(features, 11)
    print(f"  done in {time.time() - t0:.0f} s")
    nn1 = idx[:, 0]
    feat_purity = label_purity(idx[:, 1:11], labels)

    # ---- UMAP ----
    print("UMAP ...")
    import umap

    t0 = time.time()
    emb = umap.UMAP(
        n_neighbors=a.umap_neighbors, min_dist=a.umap_min_dist, random_state=a.seed
    ).fit_transform(features)
    print(f"  done in {time.time() - t0:.0f} s")
    emb = emb - emb.mean(0)
    _, eidx = NearestNeighbors(n_neighbors=16).fit(emb).kneighbors(emb)
    emb_purity = label_purity(eidx[:, 1:], labels)
    # the map region a point sits in: majority label of its UMAP neighbours
    region = np.array(
        [np.bincount(labels[r], minlength=n_cls).argmax() for r in eidx[:, 1:]]
    )

    # ---- undirected 1-NN edges ----
    pairs = np.unique(np.sort(np.c_[np.arange(N), nn1], axis=1), axis=0)
    u, v = pairs[:, 0], pairs[:, 1]
    diag = float(np.linalg.norm(emb.max(0) - emb.min(0)))
    length = np.linalg.norm(emb[u] - emb[v], axis=1)
    # "long" = long in the layout AND joining two different map regions
    # (UMAP stretches some clusters, so long edges can also occur inside a
    # single cluster; those are not what we want to show)
    is_long = (length > a.long_frac * diag) & (region[u] != region[v])
    cross = labels[u] != labels[v]
    print(
        f"{len(pairs)} edges; {cross.sum()} join different labels; "
        f"{is_long.sum()} long (> {a.long_frac:.2f} x diagonal, "
        f"between map regions)"
    )

    # ---- class-pair distinctness from the data itself ----
    # how often 1-NN edges join classes c and d, relative to chance given
    # their sizes: rarely joined pairs = visually distinct classes
    C = np.zeros((n_cls, n_cls))
    for s, t in pairs[cross]:
        C[labels[s], labels[t]] += 1
        C[labels[t], labels[s]] += 1
    counts = np.bincount(labels, minlength=n_cls).astype(float)
    rate = C / np.maximum(np.outer(counts, counts), 1) * N
    distinct = -np.log(rate + 1e-3)
    np.fill_diagonal(distinct, -np.inf)

    # ---- showcase candidates ----
    ok = (
        cross
        & is_long
        & (feat_purity[u] >= a.feature_purity)
        & (feat_purity[v] >= a.feature_purity)
        & (emb_purity[u] >= a.embed_purity)
        & (emb_purity[v] >= a.embed_purity)
    )
    cand = np.flatnonzero(ok)
    if len(cand):
        d_score = distinct[labels[u[cand]], labels[v[cand]]]
        l_score = length[cand] / diag
        score = (d_score - d_score.mean()) / (d_score.std() + 1e-9) + (
            l_score - l_score.mean()
        ) / (l_score.std() + 1e-9)
        cand = cand[np.argsort(-score)]
    print(f"{len(cand)} showcase candidates")
    if len(cand) < a.n_show:
        print(
            "  few candidates: try lowering --feature-purity / "
            "--embed-purity / --long-frac"
        )

    # greedy pick: no class used twice, edges spread over the map
    chosen, used = [], set()
    mids = (emb[u] + emb[v]) / 2
    for e in cand:
        cu, cv = int(labels[u[e]]), int(labels[v[e]])
        if cu in used or cv in used:
            continue
        if any(np.linalg.norm(mids[e] - mids[c]) < 0.08 * diag for c in chosen):
            continue
        chosen.append(int(e))
        used |= {cu, cv}
        if len(chosen) == a.n_show:
            break
    for e in cand:  # relax the rules if we still need more
        if len(chosen) >= a.n_show:
            break
        if int(e) not in chosen:
            chosen.append(int(e))

    for e in chosen:
        print(
            f"  edge {e}: {class_names[labels[u[e]]]} <-> "
            f"{class_names[labels[v[e]]]}  length {length[e]/diag:.2f} diag"
        )

    out = Path(out)
    np.savez_compressed(
        out,
        images=np.asarray(images, np.uint8),
        labels=labels,
        class_names=np.array(class_names),
        feature_space=feature_space,
        feature_dim=features.shape[1],
        emb=emb.astype(np.float32),
        pairs=pairs,
        length=length.astype(np.float32),
        is_long=is_long,
        region=region,
        diag=diag,
        candidates=cand,
        showcase=np.array(chosen, int),
        source_index=(
            np.arange(N) if source_index is None else np.asarray(source_index)
        ),
    )
    print("saved", out)
    contact_sheet(
        images,
        labels,
        class_names,
        pairs,
        cand,
        chosen,
        out.with_suffix(".candidates.png"),
        a.n_sheet,
    )


def contact_sheet(images, labels, class_names, pairs, cand, chosen, path, n_sheet):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = min(n_sheet, len(cand))
    if k == 0:
        return
    cols = 5
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 1.5))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for n, e in enumerate(cand[:k]):
        s, t = pairs[e]
        img = np.hstack(
            [
                images[s].reshape(28, 28),
                np.zeros((28, 4), np.uint8),  # white after inversion
                images[t].reshape(28, 28),
            ]
        )
        axes[n].imshow(255 - img, cmap="gray", vmin=0, vmax=255)
        star = " *" if e in chosen else ""
        axes[n].set_title(
            f"{e}: {class_names[labels[s]]}-" f"{class_names[labels[t]]}{star}",
            fontsize=7,
        )
    fig.suptitle(
        "showcase candidates (edge index: classes; " "* = pre-selected)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("contact sheet:", path)
