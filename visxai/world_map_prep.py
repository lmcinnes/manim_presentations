"""
Precompute the world map data for world_projections.py.

  * Natural Earth 110m coastlines and a 30-degree graticule (GeoJSON, from
    the natural-earth-vector repository on GitHub)
  * flattened to one (lon, lat) vertex array plus a list of segments, with
    segments split at the antimeridian so tears are real rather than drawn
    straight across the map
  * a grid of cells for colour fills: land / ocean from the land polygons,
    and (if an Equal Earth map image is available, by default the Wikimedia
    "Equal_Earth_projection_SW.jpg") each cell's colour read off that image
    and quantised to a small palette
  * the Dymaxion net: Fuller's icosahedron (see dymaxion.py) with the 11 cut
    edges chosen to run through ocean, so the continents stay whole

    python world_map_prep.py          # writes world_map.npz
"""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

IMAGE_URL = ("https://commons.wikimedia.org/wiki/Special:FilePath/"
             "Equal_Earth_projection_SW.jpg")

from dymaxion import GRAY_VERTICES, edge_faces, faces_of, ocean_cut_hinges

HERE = Path(__file__).resolve().parent
REPO = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
        "master/geojson/")
LAYERS = {"coastline": "ne_110m_coastline.geojson",
          "graticule": "ne_110m_graticules_30.geojson",
          "land": "ne_110m_land.geojson"}


def load_geojson(name, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    path = cache_dir / LAYERS[name]
    if not path.exists():
        print("downloading", LAYERS[name])
        urllib.request.urlretrieve(REPO + LAYERS[name], path)
    with open(path) as f:
        return json.load(f)


def geometry_lines(geojson):
    """Every LineString / ring in the file, as (M, 2) lon-lat arrays."""
    out = []
    for feat in geojson["features"]:
        g = feat["geometry"]
        if g is None:
            continue
        t, c = g["type"], g["coordinates"]
        if t == "LineString":
            out.append(np.asarray(c, float))
        elif t == "MultiLineString":
            out += [np.asarray(p, float) for p in c]
        elif t == "Polygon":
            out += [np.asarray(r, float) for r in c]
        elif t == "MultiPolygon":
            out += [np.asarray(r, float) for poly in c for r in poly]
    return [a[:, :2] for a in out if len(a) > 1]


def split_at_antimeridian(line):
    """Break a line wherever it jumps across +-180 degrees of longitude."""
    jumps = np.flatnonzero(np.abs(np.diff(line[:, 0])) > 180) + 1
    return [p for p in np.split(line, jumps) if len(p) > 1]


def land_mask(geojson, step=1.0):
    """Boolean land grid on a (lon, lat) raster, for scoring orientations."""
    from matplotlib.path import Path as MplPath
    lons = np.arange(-180, 180, step) + step / 2
    lats = np.arange(-90, 90, step) + step / 2
    LON, LAT = np.meshgrid(lons, lats)
    pts = np.c_[LON.ravel(), LAT.ravel()]
    mask = np.zeros(len(pts), bool)
    for ring in geometry_lines(geojson):
        mask |= MplPath(ring).contains_points(pts)
    return lons, lats, mask.reshape(LAT.shape)


def edge_land(V, a, b, lons, lats, mask, n=200):
    """Land samples along the great-circle arc between two vertices."""
    t = np.linspace(0, 1, n)[:, None]
    p = (1 - t) * V[a] + t * V[b]
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    lon = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(p[:, 2], -1, 1)))
    i = np.clip(((lat - lats[0]) / (lats[1] - lats[0])).astype(int), 0,
                len(lats) - 1)
    j = np.clip(((lon - lons[0]) / (lons[1] - lons[0])).astype(int), 0,
                len(lons) - 1)
    return int(mask[i, j].sum())


def land_cells(lons, lats, mask, cell):
    """Corners, centres and land flags of a lon-lat grid of `cell`-degree
    cells (land if at least 40% of the fine mask inside is land)."""
    step = lons[1] - lons[0]
    k = int(round(cell / step))
    ny, nx = mask.shape[0] // k, mask.shape[1] // k
    frac = mask[:ny * k, :nx * k].reshape(ny, k, nx, k).mean((1, 3))
    lon0 = -180 + cell * np.arange(nx)
    lat0 = -90 + cell * np.arange(ny)
    LON, LAT = np.meshgrid(lon0, lat0)          # (ny, nx), lower-left corners
    corners = np.stack([np.c_[LON.ravel(), LAT.ravel()],
                        np.c_[LON.ravel() + cell, LAT.ravel()],
                        np.c_[LON.ravel() + cell, LAT.ravel() + cell],
                        np.c_[LON.ravel(), LAT.ravel() + cell]], axis=1)
    return corners, (frac >= 0.4).ravel(), nx


# ---------------------------------------------------------------------------
# Colours from an Equal Earth map image
# ---------------------------------------------------------------------------
def load_image(src, cache_dir):
    """RGB array of a local image file, or of a URL (cached)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None              # the Wikimedia file is large
    path = Path(src)
    if not path.exists():
        path = Path(cache_dir) / "map_image.jpg"
        if not path.exists():
            print("downloading", src)
            req = urllib.request.Request(src, headers={
                "User-Agent": "slides-map-prep/1.0 (presentation figure)"})
            with urllib.request.urlopen(req) as r, open(path, "wb") as f:
                f.write(r.read())
    img = Image.open(path).convert("RGB")
    if max(img.size) > 4000:                   # plenty for 2-3 degree cells
        img.thumbnail((4000, 4000))
    return np.asarray(img)


def map_extent(img, tol=40):
    """Bounding box (x0, y0, x1, y1) of the map: everything differing from
    the background colour, taken from the image corners."""
    corners = np.array([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]], int)
    bg = np.median(corners, axis=0)
    ys, xs = np.nonzero(np.abs(img.astype(int) - bg).sum(-1) > tol)
    inset = 3                       # step inside any outline drawn round it
    return xs.min() + inset, ys.min() + inset, xs.max() - inset, ys.max() - inset


def equal_earth_unit(lam, phi):
    A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796
    th = np.arcsin(np.sqrt(3) / 2 * np.sin(phi))
    x = (2 * np.sqrt(3) * lam * np.cos(th)
         / (3 * (9 * A4 * th ** 8 + 7 * A3 * th ** 6 + 3 * A2 * th ** 2 + A1)))
    return x, th * (A1 + A2 * th ** 2 + A3 * th ** 6 + A4 * th ** 8)


def cell_colours(img, extent, cells, sub=5):
    """Median image colour over each cell (sub x sub samples per cell), with
    each sample placed in the image through the Equal Earth projection. The
    median ignores thin lines drawn on the map (graticule, equator)."""
    x0, y0, x1, y1 = extent
    xmax = equal_earth_unit(np.pi, 0.0)[0]
    ymax = equal_earth_unit(0.0, np.pi / 2)[1]
    lo, hi = cells[:, 0], cells[:, 2]               # lower-left, upper-right
    f = (np.arange(sub) + 0.5) / sub
    lon = lo[:, None, None, 0] + (hi - lo)[:, None, None, 0] * f[None, :, None]
    lat = lo[:, None, None, 1] + (hi - lo)[:, None, None, 1] * f[None, None, :]
    lon, lat = np.broadcast_arrays(lon, lat)
    x, y = equal_earth_unit(np.radians(lon), np.radians(lat))
    px = np.clip(np.round(x0 + (x + xmax) / (2 * xmax) * (x1 - x0)), 0,
                 img.shape[1] - 1).astype(int)
    py = np.clip(np.round(y0 + (ymax - y) / (2 * ymax) * (y1 - y0)), 0,
                 img.shape[0] - 1).astype(int)
    return np.median(img[py, px].reshape(len(cells), -1, 3), axis=1)


def quantise(colours, k, seed=0):
    """Cluster cell colours into a k-colour palette."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(colours)
    return km.labels_, np.clip(km.cluster_centers_, 0, 255).astype(np.uint8)


def preview_cells(cells, palette, index, path, nx):
    """The quantised cells as a plate carree image, to check the result."""
    from PIL import Image
    ny = len(cells) // nx
    im = palette[index].reshape(ny, nx, 3)[::-1]
    Image.fromarray(im).resize((nx * 4, ny * 4), Image.NEAREST).save(path)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(HERE / "world_cache"))
    p.add_argument("--out", default=str(HERE / "world_map.npz"))
    p.add_argument("--tries", type=int, default=3000,
                   help="spanning trees tried for the Dymaxion net")
    p.add_argument("--cell", type=float, default=2.0,
                   help="size (degrees) of the fill cells")
    p.add_argument("--image", default=IMAGE_URL,
                   help="Equal Earth map image (URL or file) to colour the "
                        "cells from; 'none' for plain land/ocean colours")
    p.add_argument("--extent", type=int, nargs=4, default=None,
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="pixel bounding box of the map in the image "
                        "(detected automatically if omitted)")
    p.add_argument("--colours", type=int, default=16,
                   help="palette size after colour quantisation")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    verts, segs, group = [], [], []
    for gi, name in enumerate(("coastline", "graticule")):
        lines = geometry_lines(load_geojson(name, a.cache))
        n_pts = 0
        for line in lines:
            for piece in split_at_antimeridian(line):
                base = len(verts)
                verts += [tuple(v) for v in piece]
                segs += [(base + k, base + k + 1) for k in range(len(piece) - 1)]
                group += [gi] * (len(piece) - 1)
                n_pts += len(piece)
        print(f"{name}: {len(lines)} lines, {n_pts} vertices")

    # ---- land / ocean cells for the colour fills ----
    lons, lats, mask = land_mask(load_geojson("land", a.cache), step=0.5)
    cells, cell_land, nx = land_cells(lons, lats, mask, a.cell)
    print(f"{len(cells)} cells of {a.cell} degrees, {cell_land.sum()} land")
    vv = np.array(verts, float)
    vertex_cell = (np.clip(((vv[:, 1] + 90) // a.cell).astype(int), 0,
                           len(cells) // nx - 1) * nx
                   + np.clip(((vv[:, 0] + 180) // a.cell).astype(int), 0,
                             nx - 1))

    # ---- colours from the map image ----
    extra = {}
    if a.image.lower() != "none":
        try:
            img = load_image(a.image, a.cache)
            extent = a.extent or map_extent(img)
            print(f"image {img.shape[1]}x{img.shape[0]}, map extent {extent}")
            index, palette = quantise(cell_colours(img, extent, cells),
                                      a.colours, a.seed)
            extra = dict(cell_colour=index, palette=palette)
            preview = Path(a.out).with_suffix(".cells.png")
            preview_cells(cells, palette, index, preview, nx)
            print(f"{a.colours}-colour palette; check {preview}")
        except Exception as e:                      # network or file issues
            print(f"could not colour from the image ({e}); "
                  "using plain land/ocean colours")

    # ---- Dymaxion: Fuller's icosahedron, cuts through the oceans ----
    V = GRAY_VERTICES
    faces = faces_of(V)
    weights = {e: edge_land(V, *e, lons, lats, mask)
               for e in edge_faces(faces)}
    cut_land, hinges, root = ocean_cut_hinges(V, faces, weights,
                                              tries=a.tries, seed=a.seed)
    print(f"Dymaxion net: {sum(w > 0 for w in weights.values())} of 30 "
          f"edges cross land; land samples on the 11 cut edges: {cut_land} "
          f"(of {11 * 200})")

    np.savez_compressed(a.out, lonlat=np.array(verts, float),
                        segments=np.array(segs, int),
                        group=np.array(group, int),
                        vertex_cell=vertex_cell,
                        cells=cells, cell_land=cell_land,
                        ico_vertices=V, ico_faces=faces,
                        ico_hinges=np.array(hinges, int), ico_root=root,
                        **extra)
    print(f"saved {a.out}: {len(verts)} vertices, {len(segs)} segments")


if __name__ == "__main__":
    main()
