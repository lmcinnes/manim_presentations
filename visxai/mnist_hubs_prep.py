"""
Precompute data for mnist_hubs_slides.py: good and bad hubs in noisy MNIST.

  * all 70,000 MNIST digits plus isotropic Gaussian pixel noise (sigma below)
  * k nearest neighbours (k = 10, raw pixel distances); for each point:
      N_k   = how many points have it among their k nearest neighbours
      BN_k  = how many of those are a different digit ("bad" occurrences,
              Radovanovic, Nanopoulos & Ivanovic, JMLR 2010)
    plus local statistics of the symmetrised kNN graph (triangles,
    clustering coefficient)
  * a 2D UMAP of the noisy data
  * candidate pairs: a bad hub (many pointers, mostly other digits, from 3+
    classes) and a good hub with matching LOCAL statistics whose pointers are
    all its own digit; ranked by how much further the bad hub's pointers
    reach on the map
  * contact sheets (PNG pages) of the candidate pairs, for hand-picking

    python mnist_hubs_prep.py                  # sigma = NOISE_SIGMA below
    python mnist_hubs_prep.py --sigma 0.3

Writes mnist_hubs.npz and mnist_hubs.candidates_NN.png.
"""
import argparse
import gzip
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

NOISE_SIGMA = 0.2      # pixel noise (pixels scaled to [0, 1]); set once chosen

HERE = Path(__file__).resolve().parent
MIRROR = "https://raw.githubusercontent.com/fgnt/mnist/master/"
FILES = dict(xtr="train-images-idx3-ubyte.gz", ytr="train-labels-idx1-ubyte.gz",
             xte="t10k-images-idx3-ubyte.gz", yte="t10k-labels-idx1-ubyte.gz")


def load_mnist(cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    arrs = {}
    for key, name in FILES.items():
        path = cache_dir / name
        if not path.exists():
            print("downloading", name)
            urllib.request.urlretrieve(MIRROR + name, path)
        with gzip.open(path, "rb") as f:
            data = f.read()
        arrs[key] = (np.frombuffer(data, np.uint8, offset=16).reshape(-1, 784)
                     if key.startswith("x")
                     else np.frombuffer(data, np.uint8, offset=8))
    return (np.vstack([arrs["xtr"], arrs["xte"]]),
            np.concatenate([arrs["ytr"], arrs["yte"]]).astype(int))


def noise_matrix(n, seed):
    """The fixed noise draw (the slide regenerates it to show noisy digits)."""
    return np.random.default_rng(seed).standard_normal((n, 784),
                                                       dtype=np.float32)


def knn(X, k):
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute",
                          n_jobs=-1).fit(X)
    _, idx = nn.kneighbors(X)
    return np.array([r[r != i][:k] for i, r in enumerate(idx)])


def local_stats(idx):
    """Triangles and clustering coefficient in the symmetrised kNN graph."""
    N, k = idx.shape
    A = csr_matrix((np.ones(N * k), (np.repeat(np.arange(N), k), idx.ravel())),
                   shape=(N, N))
    A = ((A + A.T) > 0).astype(np.float32)
    deg = np.asarray(A.sum(1)).ravel()
    tri = np.asarray((A @ A).multiply(A).sum(1)).ravel() / 2
    return tri, np.where(deg > 1, 2 * tri / (deg * (deg - 1)), 0.0)


def reverse_neighbours(idx):
    """For each point, the points that have it among their k neighbours."""
    order = np.argsort(idx.ravel(), kind="stable")
    owners = np.repeat(np.arange(len(idx)), idx.shape[1])[order]
    bounds = np.searchsorted(idx.ravel()[order], np.arange(len(idx) + 1))
    return [owners[bounds[i]:bounds[i + 1]] for i in range(len(idx))]


# ---------------------------------------------------------------------------
def find_pairs(y, idx, emb, hub_quantile, same_class_slack):
    Nk = np.bincount(idx.ravel(), minlength=len(y))
    bad_occ = np.bincount(idx.ravel(), weights=(y[idx] != y[:, None]).ravel(),
                          minlength=len(y)).astype(int)
    rev = reverse_neighbours(idx)
    n_classes = np.array([len(np.unique(y[r])) for r in rev])
    tri, clust = local_stats(idx)
    bad_frac = bad_occ / np.maximum(Nk, 1)
    thr = np.quantile(Nk, hub_quantile)
    bad = np.flatnonzero((Nk >= thr) & (bad_frac >= 0.5) & (n_classes >= 3))
    good = np.flatnonzero((Nk >= thr) & (bad_frac <= 0.05))
    print(f"hubs: N_k >= {thr:.0f}; {len(bad)} bad hubs, {len(good)} good hubs")

    F = np.c_[Nk, tri, clust].astype(float)
    F = (F - F.mean(0)) / F.std(0)

    def reach(h):
        return float(np.median(np.linalg.norm(emb[rev[h]] - emb[h], axis=1)))

    rows = []
    for b in bad:
        dist = np.linalg.norm(F[good] - F[b], axis=1)
        g = good[np.argmin(dist)]
        same = good[y[good] == y[b]]
        if len(same):                    # prefer a twin of the same digit
            ds = np.linalg.norm(F[same] - F[b], axis=1)
            if ds.min() <= same_class_slack * max(dist.min(), 0.05):
                g = same[np.argmin(ds)]
        rows.append((b, g, reach(b) / max(reach(g), 1e-9)))
    rows.sort(key=lambda r: -r[2])
    pairs = np.array([(b, g) for b, g, _ in rows], int).reshape(-1, 2)
    return pairs, dict(Nk=Nk, bad_occ=bad_occ, n_rev_classes=n_classes,
                       tri=tri, clust=clust), rev


def contact_sheets(pairs, y, images, emb, rev, stats, out, per_page=6,
                   n_pages=5, n_show=12):
    """Pages in the style used for hand-picking: map with each hub's
    incoming edges, plus the hub and its pointers as images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    Nk, bo, tri = stats["Nk"], stats["bad_occ"], stats["tri"]
    bg = np.random.default_rng(0).choice(len(y), min(len(y), 25000), False)
    files = []
    for page in range(n_pages):
        chunk = pairs[page * per_page:(page + 1) * per_page]
        if not len(chunk):
            break
        fig = plt.figure(figsize=(16, 4 * len(chunk)))
        gs = GridSpec(len(chunk), 2, figure=fig, width_ratios=[1.0, 1.35])
        for r, (b, g) in enumerate(chunk):
            ax = fig.add_subplot(gs[r, 0])
            ax.scatter(*emb[bg].T, c=y[bg], cmap="tab10", s=0.4, alpha=0.35)
            for h, col in ((g, "green"), (b, "red")):
                for j in rev[h]:
                    ax.plot(*emb[[h, j]].T, color=col, lw=0.9)
                ax.scatter(*emb[h], s=80, c=col, edgecolor="k", zorder=5)
            ax.set_title(f"pair ({b}, {g}): bad '{y[b]}' (red) vs good "
                         f"'{y[g]}' (green)", fontsize=9)
            ax.axis("off")
            sub = gs[r, 1].subgridspec(2, n_show + 1)
            for rr, (h, col) in enumerate(((g, "green"), (b, "red"))):
                a0 = fig.add_subplot(sub[rr, 0])
                a0.imshow(255 - images[h].reshape(28, 28), cmap="gray")
                a0.axis("off")
                a0.set_title(f"hub '{y[h]}'\nN_k {Nk[h]} bad {bo[h]}\n"
                             f"tri {tri[h]:.0f}", fontsize=7, color=col)
                for c, j in enumerate(rev[h][:n_show]):
                    a = fig.add_subplot(sub[rr, c + 1])
                    a.imshow(255 - images[j].reshape(28, 28), cmap="gray")
                    a.axis("off")
                    a.set_title(str(y[j]), fontsize=8,
                                color="black" if y[j] == y[h] else "red")
        plt.tight_layout()
        path = Path(out).with_name(f"{Path(out).stem}.candidates_"
                                   f"{page + 1:02d}.png")
        plt.savefig(path, dpi=55)
        plt.close(fig)
        files.append(path)
    print("contact sheets:", ", ".join(str(f.name) for f in files))


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, default=NOISE_SIGMA)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n", type=int, default=0, help="subsample (0 = all)")
    p.add_argument("--seed", type=int, default=0, help="subsample / UMAP seed")
    p.add_argument("--noise-seed", type=int, default=1)
    p.add_argument("--hub-quantile", type=float, default=0.85,
                   help="hubs are points with N_k above this quantile")
    p.add_argument("--same-class-slack", type=float, default=1.5,
                   help="prefer a same-digit good twin if its local match is "
                        "within this factor of the best match")
    p.add_argument("--pages", type=int, default=5)
    p.add_argument("--cache", default=str(HERE / "mnist_cache"))
    p.add_argument("--out", default=str(HERE / "mnist_hubs.npz"))
    a = p.parse_args()

    X8, y = load_mnist(a.cache)
    sel = np.arange(len(X8))
    if a.n and a.n < len(X8):
        sel = np.sort(np.random.default_rng(a.seed).choice(len(X8), a.n,
                                                           replace=False))
    X8, y = X8[sel], y[sel]
    X = X8.astype(np.float32) / 255.0
    X += a.sigma * noise_matrix(len(X), a.noise_seed)
    print(f"{len(X)} digits, noise sigma {a.sigma}")

    t0 = time.time()
    idx = knn(X, a.k)
    print(f"kNN done in {time.time() - t0:.0f} s")
    t0 = time.time()
    import umap
    emb = umap.UMAP(random_state=a.seed).fit_transform(X).astype(np.float32)
    print(f"UMAP done in {time.time() - t0:.0f} s")

    pairs, stats, rev = find_pairs(y, idx, emb, a.hub_quantile,
                                   a.same_class_slack)
    for b, g in pairs[:10]:
        print(f"  pair ({b:5d}, {g:5d}): bad '{y[b]}' N_k {stats['Nk'][b]} "
              f"({stats['bad_occ'][b]} bad, {stats['n_rev_classes'][b]} "
              f"classes) | good '{y[g]}' N_k {stats['Nk'][g]} "
              f"({stats['bad_occ'][g]} bad)")

    np.savez_compressed(
        a.out, images=X8, labels=y, emb=emb, idx=idx.astype(np.int32),
        pairs=pairs, sigma=a.sigma, noise_seed=a.noise_seed,
        source_index=sel, k=a.k, **stats)
    print("saved", a.out)
    contact_sheets(pairs, y, X8, emb, rev, stats, a.out, n_pages=a.pages)


if __name__ == "__main__":
    main()
