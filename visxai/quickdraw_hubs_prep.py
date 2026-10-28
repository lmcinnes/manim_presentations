"""
Precompute data for quickdraw_hubs_slides.py: good and bad hubs among clean
(no added noise) Quick, Draw! doodles embedded with DINOv2.

  * doodles from a few visually distinct classes (as in quickdraw_knn_prep),
    embedded with DINOv2 (L2-normalised, so Euclidean = cosine neighbours)
  * k nearest neighbours (k = 10) in the DINOv2 embedding space; for each
    point:
      N_k   = how many points have it among their k nearest neighbours
      BN_k  = how many of those are a different class ("bad" occurrences,
              Radovanovic, Nanopoulos & Ivanovic, JMLR 2010)
    plus local statistics of the symmetrised kNN graph (triangles,
    clustering coefficient)
  * a 2D UMAP of the embeddings
  * candidate pairs: a bad hub (many pointers, mostly other classes, from 3+
    classes) and a good hub with matching LOCAL statistics whose pointers are
    all its own class; ranked by how much further the bad hub's pointers
    reach on the map
  * contact sheets (PNG pages) of the candidate pairs, for hand-picking

    pip install torch transformers umap-learn
    python quickdraw_hubs_prep.py
    python quickdraw_hubs_prep.py --per-class 3000 --model facebook/dinov2-small

Writes quickdraw_hubs.npz and quickdraw_hubs.candidates_NN.png. Doodles and
DINOv2 features are cached in quickdraw_cache/ (shared with
quickdraw_knn_prep.py), so re-running with other selection settings is quick.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from mnist_hubs_prep import contact_sheets, find_pairs, knn
from quickdraw_knn_prep import CATEGORIES, embed_dinov2, load_quickdraw

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--categories", nargs="+", default=CATEGORIES,
                   help="at most 10 (one palette colour each)")
    p.add_argument("--per-class", type=int, default=1000)
    p.add_argument("--offset", type=int, default=0,
                   help="first row taken from each class file")
    p.add_argument("--model", default="facebook/dinov2-base")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0, help="UMAP seed")
    p.add_argument("--hub-quantile", type=float, default=0.85,
                   help="hubs are points with N_k above this quantile")
    p.add_argument("--bad-frac", type=float, default=0.5,
                   help="bad hubs: at least this fraction of pointers are "
                        "another class")
    p.add_argument("--min-classes", type=int, default=3,
                   help="bad hubs: pointers from at least this many classes")
    p.add_argument("--same-class-slack", type=float, default=1.5,
                   help="prefer a same-class good twin if its local match is "
                        "within this factor of the best match")
    p.add_argument("--pages", type=int, default=5)
    p.add_argument("--cache", default=str(HERE / "quickdraw_cache"))
    p.add_argument("--out", default=str(HERE / "quickdraw_hubs.npz"))
    a = p.parse_args()
    if len(a.categories) > 10:
        p.error("at most 10 categories (one palette colour each)")

    X8, y = load_quickdraw(a.categories, a.per_class, a.offset, a.cache)
    tag = a.model.replace("/", "_")
    F = embed_dinov2(
        X8, a.model, a.batch_size,
        Path(a.cache) / f"features_{tag}_{'-'.join(a.categories)}_"
        f"{a.offset}_{a.per_class}.npz")
    print(f"{len(F)} doodles, {F.shape[1]}-dimensional DINOv2 features")

    t0 = time.time()
    idx = knn(F, a.k)
    print(f"kNN done in {time.time() - t0:.0f} s")
    t0 = time.time()
    import umap
    emb = umap.UMAP(random_state=a.seed).fit_transform(F).astype(np.float32)
    print(f"UMAP done in {time.time() - t0:.0f} s")

    pairs, stats, rev = find_pairs(y, idx, emb, a.hub_quantile,
                                   a.same_class_slack, a.bad_frac,
                                   a.min_classes)
    names = list(a.categories)
    for b, g in pairs[:10]:
        print(f"  pair ({b:5d}, {g:5d}): bad '{names[y[b]]}' N_k "
              f"{stats['Nk'][b]} ({stats['bad_occ'][b]} bad, "
              f"{stats['n_rev_classes'][b]} classes) | good '{names[y[g]]}' "
              f"N_k {stats['Nk'][g]} ({stats['bad_occ'][g]} bad)")

    np.savez_compressed(
        a.out, images=X8, labels=y, class_names=np.array(names), emb=emb,
        idx=idx.astype(np.int32), pairs=pairs, k=a.k, model=a.model,
        feature_dim=F.shape[1], **stats)
    print("saved", a.out)
    contact_sheets(pairs, y, X8, emb, rev, stats, a.out, n_pages=a.pages,
                   names=names)


if __name__ == "__main__":
    main()
