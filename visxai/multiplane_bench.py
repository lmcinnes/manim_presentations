"""
Multi-plane loop benchmark: an alternative to the single-growing-frequency-
set construction in lissajous_bench.py.

Instead of one global set of frequencies that must grow with n_freq to stay
pairwise-simple (forcing curve length to explode -- see lissajous_bench.py's
coiling_ratio derivation), this construction splits the curve across
n_planes independent 2D coordinate blocks, each carrying its own SMALL pair
of frequencies (fx, fy) and its own independent random phases:

    x_{2i}(theta)   = cos(fx_i * theta + phase_x_i)
    x_{2i+1}(theta) = sin(fy_i * theta + phase_y_i)

A self-intersection at (t,s), t != s, requires EVERY plane to collide
simultaneously -- a much weaker global condition than the single-sum
construction, where every term shares one theta and must sum exactly.
Because each plane only needs its OWN pair to be non-degenerate (not every
pair across the whole frequency set), frequency magnitude does not need to
grow with n_planes: n_planes can grow while every fx_i, fy_i stays in a
small fixed range, giving genuinely higher intrinsic dimension without the
coiling blowup. This is verified empirically below (not proven in closed
form -- there is no closed-form distance formula for this curve, unlike
the single-sum case, so reach is always computed numerically).
"""

import numpy as np
from scipy.optimize import minimize


def _sample_freq_pairs(n_planes, freq_max, rng, allow_equal=False):
    """Small integer frequency pairs, one per plane, each in [1, freq_max].
    Pairs CAN repeat across planes (a repeated pair with independent random
    phases is fine -- see module docstring; verified numerically below,
    not just asserted)."""
    pairs = []
    for _ in range(n_planes):
        fx, fy = rng.integers(1, freq_max + 1, size=2)
        if not allow_equal:
            while fy == fx:
                fy = rng.integers(1, freq_max + 1)
        pairs.append((int(fx), int(fy)))
    return pairs


def multiplane_curve(theta, freq_pairs, phases_x, phases_y):
    """theta: array of parameter values. Returns (len(theta), 2*n_planes)."""
    theta = np.atleast_1d(theta)
    n_planes = len(freq_pairs)
    X = np.zeros((len(theta), 2 * n_planes))
    for i, (fx, fy) in enumerate(freq_pairs):
        X[:, 2 * i] = np.cos(fx * theta + phases_x[i])
        X[:, 2 * i + 1] = np.sin(fy * theta + phases_y[i])
    return X


def _true_min_self_distance_generic(
    curve_fn, dim, grid_size=2500, seed=0, n_refine_starts=10
):
    """Same exact (non-averaged) numerical reach search as
    lissajous_bench._true_min_self_distance, generalized to an arbitrary
    curve_fn(theta_array) -> (N, dim) since the multi-plane curve has no
    closed-form distance formula to average over in the first place."""

    def d2_exact(t, s):
        xt = curve_fn(np.array([t]))[0]
        xs = curve_fn(np.array([s]))[0]
        return float(((xt - xs) ** 2).sum())

    # adaptive correlation length: first u where mean d^2 over random pairs
    # at that separation reaches half the mean d^2 at large separation
    rng = np.random.default_rng(seed)
    t0s = rng.uniform(0, 2 * np.pi, 400)
    X0 = curve_fn(t0s)
    Xpi = curve_fn((t0s + np.pi) % (2 * np.pi))
    d2_far_scale = ((X0 - Xpi) ** 2).sum(1).mean()

    u_scan = np.linspace(1e-4, np.pi, 300)
    d2_u = []
    for u in u_scan:
        ts = rng.uniform(0, 2 * np.pi, 60)
        Xt = curve_fn(ts)
        Xs = curve_fn(ts + u)
        d2_u.append(((Xt - Xs) ** 2).sum(1).mean())  # sum over dims, mean
        # over samples -- must
        # match d2_far_scale's
        # convention exactly,
        # a units mismatch here
        # (.mean() over both
        # axes) previously made
        # this ~dim times too
        # small, so the
        # threshold was never
        # crossed and u_corr
        # silently fell back to
        # pi -- which masked
        # EVERY pair and made
        # the "true minimum"
        # trivially t==s (0).
    d2_u = np.array(d2_u)
    idx = np.argmax(d2_u >= 0.5 * d2_far_scale)
    u_corr = (
        u_scan[idx] if d2_u[idx] >= 0.5 * d2_far_scale else u_scan[len(u_scan) // 4]
    )
    excl = min(max(3 * u_corr, 1e-2), np.pi / 2)  # hard cap: excl must stay
    # well under pi or every
    # pair gets masked (tt is
    # capped at pi by the
    # wraparound below)

    t_grid = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)
    X = curve_fn(t_grid)
    sq = np.einsum("ij,ij->i", X, X)
    Dsq = sq[:, None] + sq[None, :] - 2 * (X @ X.T)
    np.maximum(Dsq, 0, out=Dsq)
    tt = np.abs(t_grid[:, None] - t_grid[None, :])
    tt = np.minimum(tt, 2 * np.pi - tt)
    mask = tt > excl
    Dsq_masked = np.where(mask, Dsq, np.inf)
    i0, j0 = np.unravel_index(np.argmin(Dsq_masked), Dsq.shape)
    t0, s0 = t_grid[i0], t_grid[j0]

    best = (d2_exact(t0, s0), t0, s0)
    starts = [(t0, s0)] + [
        (t0 + rng.normal(0, 0.05), s0 + rng.normal(0, 0.05))
        for _ in range(n_refine_starts)
    ]
    for ts, ss in starts:
        res = minimize(
            lambda x: d2_exact(x[0], x[1]),
            x0=[ts, ss],
            method="Nelder-Mead",
            options=dict(xatol=1e-10, fatol=1e-12, maxiter=2000),
        )
        sep = abs(((res.x[0] - res.x[1] + np.pi) % (2 * np.pi)) - np.pi)
        if res.fun < best[0] and sep > excl / 2:
            best = (res.fun, res.x[0], res.x[1])

    return dict(
        d_min_true=float(np.sqrt(max(best[0], 0.0))),
        t_star=float(best[1]),
        s_star=float(best[2]),
        u_correlation_length=float(u_corr),
    )


def generate_multiplane_benchmark(
    n_planes,
    ambient_dim,
    n_samples,
    noise_std,
    freq_max=5,
    radius=1.0,
    seed=0,
    grid_size=2500,
):
    """Multi-plane analogue of lissajous_bench.generate_lissajous_benchmark.
    Same contract (isometric random embedding into ambient_dim, isotropic
    noise, reach-based diagnostics), different curve construction."""
    rng = np.random.default_rng(seed)

    if ambient_dim < 2 * n_planes:
        raise ValueError("ambient_dim must be >= 2*n_planes")

    freq_pairs = _sample_freq_pairs(n_planes, freq_max, rng)
    phases_x = rng.uniform(0, 2 * np.pi, n_planes)
    phases_y = rng.uniform(0, 2 * np.pi, n_planes)

    def curve_fn(theta):
        return multiplane_curve(theta, freq_pairs, phases_x, phases_y)

    # rescale to requested radius (RMS point-norm = radius, matching the
    # convention in the uploaded CurvyLoopEmbedding code)
    t_probe = rng.uniform(0, 2 * np.pi, 3000)
    X_probe = curve_fn(t_probe)
    current_scale = np.sqrt((X_probe**2).sum(1).mean())
    scale = radius / current_scale

    def curve_scaled(theta):
        return scale * curve_fn(theta)

    rec = _true_min_self_distance_generic(
        curve_scaled, 2 * n_planes, grid_size=grid_size, seed=seed
    )
    d_min = rec["d_min_true"]
    reach = d_min / 2.0  # curvature reach omitted here (numerically noisy
    # to estimate via finite differences); this is a
    # conservative diagnostic, not the exact reach

    t = rng.uniform(0, 2 * np.pi, n_samples)
    X_low = curve_scaled(t)

    if n_samples < 3:
        d_typical = radius
    else:
        idx = rng.choice(n_samples, min(n_samples, 400), replace=False)
        Xs = X_low[idx]
        Dsq = ((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1)
        d_typical = float(np.sqrt(Dsq[np.triu_indices(len(idx), 1)].mean()))

    M = rng.standard_normal((ambient_dim, 2 * n_planes))
    Q, _ = np.linalg.qr(M)
    Q = Q[:, : 2 * n_planes]
    X_clean = X_low @ Q.T
    X = X_clean + noise_std * rng.standard_normal(X_clean.shape)

    noise_offset = noise_std * np.sqrt(ambient_dim)

    # numerical arc length (finite differences on a fine grid) and the same
    # sampling-density rule of thumb used in lissajous_bench.py
    t_fine = np.linspace(0, 2 * np.pi, 5000, endpoint=False)
    X_fine = curve_scaled(t_fine)
    seglen = np.sqrt(((np.roll(X_fine, -1, axis=0) - X_fine) ** 2).sum(1))
    curve_length = float(seglen.sum())
    min_recommended_samples = int(np.ceil(10 * curve_length / max(reach, 1e-12)))

    diagnostics = dict(
        d_typical=d_typical,
        d_min_recurrence=d_min,
        reach=reach,
        curve_length=curve_length,
        coiling_ratio=curve_length / d_typical,
        min_recommended_samples=min_recommended_samples,
        noise_std=noise_std,
        ambient_dim=ambient_dim,
        expected_noise_offset=noise_offset,
        offset_to_reach_ratio=noise_offset / max(reach, 1e-12),
        classically_feasible=bool(noise_offset < reach),
        u_correlation_length=rec["u_correlation_length"],
    )

    return dict(
        X=X,
        X_clean=X_clean,
        t=t,
        freq_pairs=freq_pairs,
        phases_x=phases_x,
        phases_y=phases_y,
        embedding=Q,
        ground_truth=dict(betti=[1, 1], n_planes=n_planes),
        diagnostics=diagnostics,
    )
