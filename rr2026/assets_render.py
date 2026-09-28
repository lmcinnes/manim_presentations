"""Point-cloud rendering for slide assets (datashader), shared by the
Payoff and ArXiv asset scripts.  No manim required."""

import numpy as np
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf

# Keep in step with config.COLOR_CYCLE (digits / categories) and the talk palette.
COLOR_CYCLE = ["#597ec9", "#fcaf3e", "#7ec959", "#c9597e", "#7e59c9",
               "#00d7e3", "#6d7561", "#c2badf", "#b66100", "#fba2a6"]
OTHER_COLOR = "#b9bfc9"
# Density shading: a full 256-step colormap through the talk's anchor colours
# (a three-entry list gives datashader only three levels to work with).
DENSITY_ANCHORS = ["#dfe4ee", "#7088b8", "#2c3e63"]


def _density_cmap(anchors=DENSITY_ANCHORS, n=256):
    from matplotlib.colors import LinearSegmentedColormap, to_hex

    cmap = LinearSegmentedColormap.from_list("talk_density", anchors, N=n)
    return [to_hex(cmap(x)) for x in np.linspace(0, 1, n)]


DENSITY_CMAP = _density_cmap()


def square_extent(xy, pad=0.04, quantile=0.001):
    """Square (x0, x1, y0, y1) around the bulk of the points."""
    lo = np.quantile(xy, quantile, axis=0)
    hi = np.quantile(xy, 1 - quantile, axis=0)
    centre = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo).max() * (1 + 2 * pad)
    return (centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half)


def render(xy, categories=None, colors=None, extent=None, size=1024, spread_px=0, min_alpha=150,
           fixed_spread=True):
    """Render points to a PIL image on a white background.

    ``categories``: integer codes per point (or None for density shading);
    ``colors``: one hex colour per code. Shading is histogram-equalised per
    image, with rescale_discrete_levels so sparse images (few distinct counts,
    as in deep zooms) use the visible top of the colour range instead of
    washing out. ``spread_px`` grows every point.
    """
    x0, x1, y0, y1 = extent if extent is not None else square_extent(xy)
    canvas = ds.Canvas(plot_width=size, plot_height=size, x_range=(x0, x1), y_range=(y0, y1))
    df = pd.DataFrame({"x": xy[:, 0].astype(np.float32), "y": xy[:, 1].astype(np.float32)})
    if categories is not None:
        codes = np.asarray(categories)
        df["cat"] = pd.Categorical(codes, categories=list(range(len(colors))))
        agg = canvas.points(df, "x", "y", ds.count_cat("cat"))
        img = tf.shade(agg, color_key=list(colors), how="eq_hist", min_alpha=min_alpha,
                       rescale_discrete_levels=True)
    else:
        agg = canvas.points(df, "x", "y")
        img = tf.shade(agg, cmap=DENSITY_CMAP, how="eq_hist", min_alpha=min_alpha,
                       rescale_discrete_levels=True)
    if spread_px:
        # fixed_spread: grow every point by spread_px pixels (tf.spread), which
        # keeps sparse, zoomed-in views legible; otherwise spread adaptively.
        if fixed_spread:
            img = tf.spread(img, px=int(spread_px), shape="circle")
        else:
            img = tf.dynspread(img, threshold=0.6, max_px=spread_px)
    img = tf.set_background(img, "white")
    return img.to_pil().convert("RGB")
