"""
Precompute the frames for the noise-sweep animation.

Pipeline:
  1. Build a clean multi-plane curve embedded in R^D (multiplane_bench).
  2. Draw ONE fixed standard-normal noise matrix Z (N x D).
  3. For each sigma in a sweep:  X(sigma) = X_clean + sigma * Z
     -> symmetric kNN graph -> 3D spectral embedding.
  4. Normalise each frame (centre, unit RMS radius) and align it to the
     previous frame. Spectral embeddings are only defined up to sign and
     rotation within near-degenerate eigenspaces (for a loop the modes
     come in cos/sin-like pairs, and a 3D embedding splits the second
     pair), so without this the cloud would flip and spin between frames
     for reasons unrelated to the noise. See embed_aligned.
  5. Optionally (--umap) also run 3D UMAP on the same X(sigma), with the
     same neighbourhood size, for the side-by-side comparison scene.
  6. Save everything to an .npz that the manim scenes read.

Also records, per frame, the fraction of kNN edges that are "shortcuts":
edges joining points more than 5% of the loop apart in the parameter t.
"""

import argparse
import numpy as np
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph

from multiplane_bench import generate_multiplane_benchmark, multiplane_curve


def knn_adjacency(X, k):
    A = kneighbors_graph(X, k, mode="connectivity", include_self=False)
    return A.maximum(A.T)  # symmetrise (union)


def spectral_modes(A, n_modes, seed):
    """Smallest nontrivial eigenpairs of the symmetric normalised Laplacian,
    mapped back by D^{-1/2} (the same convention as sklearn's
    spectral_embedding with norm_laplacian=True, drop_first=True)."""
    L, dd = laplacian(A.astype(float), normed=True, return_diag=True)
    v0 = np.random.default_rng(seed).uniform(-1, 1, A.shape[0])
    vals, vecs = eigsh(L, k=n_modes + 1, sigma=-1e-3, which="LM", v0=v0)
    order = np.argsort(vals)
    vals, vecs = vals[order][1:], vecs[:, order][:, 1:]
    return vals, vecs / dd[:, None]


def embed_aligned(A, Y_prev, seed, n_extra=3, degen_tol=0.3, min_modes=4):
    """3D spectral embedding, gauge-fixed against the previous frame.

    Takes the first min_modes nontrivial modes (4 = the two lowest
    cos/sin-like pairs of a loop), plus any further modes whose eigenvalue
    is within degen_tol (relative) of the last one kept. Those are
    near-degenerate partners; the 3rd coordinate is not well defined
    without them. A fixed floor avoids jumps when the degeneracy test
    toggles between frames. A rectangular orthogonal Procrustes then picks the
    isometric 3D slice of that span that best matches Y_prev.
    """
    vals, V = spectral_modes(A, 3 + n_extra, seed)
    keep = min_modes + int(
        np.sum(vals[min_modes:] <= vals[min_modes - 1] * (1 + degen_tol))
    )
    V = V[:, :keep] - V[:, :keep].mean(0)
    V /= np.sqrt((V[:, :3] ** 2).sum(1).mean())
    if Y_prev is None:
        return V[:, :3], keep
    U, _, Wt = np.linalg.svd(V.T @ Y_prev, full_matrices=False)
    Y = V @ (U @ Wt)
    Y /= np.sqrt((Y**2).sum(1).mean())  # unit RMS radius
    return Y, keep


def umap_frame(X, k, Y_prev_raw, seed, init_mode, n_epochs_warm, lr_warm):
    """One 3D UMAP layout. n_neighbors = k + 1 because UMAP counts each
    point as its own neighbour, so this matches the kNN graph above.

    init_mode="warm": start from the previous frame's layout (with fewer
    epochs). This gives temporal coherence, the same role Procrustes plays
    for the spectral frames, but it means each frame is "UMAP refined from
    the previous layout" rather than a fresh run.
    Warm starts use a reduced learning rate (lr_warm): at the default
    rate, repeated refits of near-identical data still move points
    noticeably (SGD jitter), which shows up as shimmer in the animation.
    init_mode="fresh": default spectral init every frame (a truer picture
    of what a user would get at that noise level, but can jump between
    frames, e.g. breaking the loop in a different place).
    """
    import umap  # optional dependency

    kw = dict(n_components=3, n_neighbors=k + 1, random_state=seed)
    if init_mode == "warm" and Y_prev_raw is not None:
        kw.update(init=Y_prev_raw, n_epochs=n_epochs_warm, learning_rate=lr_warm)
    elif n_epochs_warm:  # fresh fit with an explicit epoch count
        kw.update(n_epochs=n_epochs_warm)
    return umap.UMAP(**kw).fit_transform(X)


def normalise_align(Y, Y_prev):
    """Centre, unit RMS radius, and (if given) orthogonal Procrustes onto
    Y_prev. For UMAP only rotations/reflections are removed: the layout
    itself is not otherwise changed."""
    Y = Y - Y.mean(0)
    Y = Y / np.sqrt((Y**2).sum(1).mean())
    if Y_prev is not None:
        U, _, Vt = np.linalg.svd(Y.T @ Y_prev)
        Y = Y @ (U @ Vt)
    return Y


def circ_dt(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-planes", type=int, default=9)
    p.add_argument("--ambient-dim", type=int, default=1024)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--freq-max", type=int, default=3)
    p.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="RMS radius of the clean curve (benchmark convention)",
    )
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-frames", type=int, default=150)
    p.add_argument("--sigma-start", type=float, default=0.10)
    p.add_argument("--sigma-end", type=float, default=0.22)
    p.add_argument(
        "--scale-units",
        action="store_true",
        help="read --sigma-start/--sigma-end as multiples of "
        "d_typical / D**0.25 instead of absolute sigma",
    )
    p.add_argument(
        "--iid-t",
        action="store_true",
        help="keep the benchmark's i.i.d. uniform t. By default t "
        "is resampled on a jittered grid: i.i.d. sampling can "
        "leave a hole no kNN edge crosses, so the 'clean' "
        "graph is a path rather than a loop",
    )
    p.add_argument(
        "--umap",
        action="store_true",
        help="also compute UMAP frames (for SpectralVsUMAP)",
    )
    p.add_argument("--umap-init", choices=["warm", "fresh"], default="warm")
    p.add_argument("--umap-warm-epochs", type=int, default=100)
    p.add_argument("--umap-warm-lr", type=float, default=0.25)
    p.add_argument(
        "--umap-first-epochs",
        type=int,
        default=3000,
        help="epochs for the initial (fresh) UMAP fit at frame 0; "
        "well above UMAP's default (500 for this N) so the "
        "starting layout is converged",
    )
    p.add_argument(
        "--umap-burn-in",
        type=int,
        default=5,
        help="warm refits of the frame-0 data, with the sweep's "
        "own settings, before the sweep starts, so later "
        "motion is due to the noise and not to the change "
        "from fresh-fit to warm-refit dynamics",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-seed", type=int, default=12345)
    p.add_argument("--out", default="embedding_frames.npz")
    a = p.parse_args()

    bench = generate_multiplane_benchmark(
        a.n_planes,
        a.ambient_dim,
        a.n_samples,
        noise_std=0.0,
        freq_max=a.freq_max,
        radius=a.radius,
        seed=a.seed,
    )
    X_clean, t = bench["X_clean"], bench["t"]
    diag = bench["diagnostics"]
    N, D = X_clean.shape
    if not a.iid_t:
        curve = lambda th: multiplane_curve(
            th, bench["freq_pairs"], bench["phases_x"], bench["phases_y"]
        )
        # recover the radius rescaling applied inside the generator
        scale = np.linalg.norm(X_clean[0]) / np.linalg.norm(curve(t[:1])[0])
        rng_t = np.random.default_rng(a.seed + 1)
        t = (np.arange(N) + rng_t.uniform(size=N)) * (2 * np.pi / N)
        t = rng_t.permutation(t)  # no ordering info in the point indices
        X_clean = scale * curve(t) @ bench["embedding"].T
    print("freq pairs:", bench["freq_pairs"])
    print({k: round(diag[k], 4) for k in ("reach", "curve_length", "d_typical")})

    # fixed noise directions and relative magnitudes for the whole sweep
    Z = np.random.default_rng(a.noise_seed).standard_normal((N, D))

    A0 = knn_adjacency(X_clean, a.k)
    r, c = A0.nonzero()
    d_nn = float(np.median(np.linalg.norm(X_clean[r] - X_clean[c], axis=1)))
    dt_thresh = 0.05 * 2 * np.pi

    # Isotropic noise adds ~2 sigma^2 D to EVERY squared distance, which
    # doesn't change neighbour ranks; what breaks them is the fluctuation,
    # ~ sigma^2 sqrt(D). So the relevant scale is d_typical / D^(1/4).
    # The sweep is focused on the breakdown window (geometric spacing, so
    # the scenes can interpolate sigma in log space).
    sigma_scale = diag["d_typical"] / D**0.25
    unit = sigma_scale if a.scale_units else 1.0
    sigmas = np.geomspace(a.sigma_start * unit, a.sigma_end * unit, a.n_frames)
    print(
        f"d_nn={d_nn:.4g}  d_typical/D^(1/4)={sigma_scale:.4g}  "
        f"sigma: {sigmas[0]:.4g} -> {sigmas[-1]:.4g}  "
        f"(= {sigmas[0]/sigma_scale:.3f} -> {sigmas[-1]/sigma_scale:.3f} "
        f"x d_typical/D^(1/4))"
    )

    frames, shortcut_frac, n_comp = [], [], []
    Y_prev = None
    umap_frames, U_prev, U_prev_raw = [], None, None
    for i, s in enumerate(sigmas):
        A = knn_adjacency(X_clean + s * Z, a.k)
        nc, _ = connected_components(A, directed=False)
        Y, keep = embed_aligned(A, Y_prev, a.seed)
        frames.append(Y)
        Y_prev = Y
        r, c = A.nonzero()
        shortcut_frac.append(float((circ_dt(t[r], t[c]) > dt_thresh).mean()))
        n_comp.append(nc)
        if a.umap:
            Xs = X_clean + s * Z
            if i == 0 and a.umap_init == "warm":
                # long fresh fit, then burn-in with the sweep's own settings
                U_prev_raw = umap_frame(
                    Xs, a.k, None, a.seed, "fresh", a.umap_first_epochs, None
                )
                for b_ in range(a.umap_burn_in):
                    U_prev_raw = umap_frame(
                        Xs,
                        a.k,
                        U_prev_raw,
                        a.seed + b_,
                        "warm",
                        a.umap_warm_epochs,
                        a.umap_warm_lr,
                    )
            U_raw = umap_frame(
                Xs,
                a.k,
                U_prev_raw,
                a.seed,
                a.umap_init,
                a.umap_warm_epochs,
                a.umap_warm_lr,
            )
            U_prev = normalise_align(U_raw, U_prev)
            umap_frames.append(U_prev)
            U_prev_raw = U_raw
        if i % 10 == 0 or i == len(sigmas) - 1:
            print(
                f"[{i:3d}] sigma={s:.3e}  noise offset/d_nn="
                f"{s*np.sqrt(D)/d_nn:7.2f}  sigma/scale={s/sigma_scale:.3f}  shortcuts={shortcut_frac[-1]:.3f}"
                f"  modes used={keep}  components={nc}"
            )

    extra = {}
    if a.umap:
        extra["umap_embeddings"] = np.array(umap_frames, dtype=np.float32)
    np.savez_compressed(
        a.out,
        embeddings=np.array(frames, dtype=np.float32),
        sigma_over_scale=sigmas / sigma_scale,
        sigma_scale=sigma_scale,
        sigmas=sigmas,
        offset_over_dnn=sigmas * np.sqrt(D) / d_nn,
        offset_over_reach=sigmas * np.sqrt(D) / max(diag["reach"], 1e-12),
        shortcut_frac=np.array(shortcut_frac),
        n_components=np.array(n_comp),
        t=t,
        ambient_dim=D,
        n_planes=a.n_planes,
        k=a.k,
        **extra,
    )
    print("saved", a.out)


if __name__ == "__main__":
    main()
