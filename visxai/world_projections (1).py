from manim import *

import sys

sys.path.append("..")  # Add parent directory to path to import config
sys.path.append(".")

from config import (
    apply_defaults,
    COLOR_CYCLE,
    DEFAULT_COLOR,
    ACCENT_COLOR,
    HIGHLIGHT_COLOR,
    BACKGROUND_COLOR,
    add_logo_to_background,
    create_styled_axes,
    TIMCSlide,
    ThreeDTIMCSlide,
    PhaseSlide,
    colormap_color,
    create_logo,
)

import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))  # dymaxion.py lives next to this file
from dymaxion import Icosahedron, rot

apply_defaults()

DATA_FILE = Path(os.environ.get("WORLD_MAP", _HERE / "world_map.npz"))

OCEAN_COLOR = interpolate_color(COLOR_CYCLE[0], WHITE, 0.6)
LAND_COLOR = interpolate_color(COLOR_CYCLE[2], WHITE, 0.45)
COAST_COLOR = DEFAULT_COLOR
GRATICULE_COLOR = WHITE   # reads on both the photographic land and sea
FACE_SHADING = 0.35       # Dymaxion: darkest face shading (0 = none)
EDGE_COLOR, EDGE_WIDTH, EDGE_OPACITY = DEFAULT_COLOR, 1.2, 0.55
INSIDE_DARKEN = 0.4       # the inside of the surface: same map, darker
                          # (0 = same as outside; 1 = black)

# The icosahedron (dymaxion.py) uses x = cos(lat) cos(lon), y = cos(lat)
# sin(lon); the screen layout here has the prime meridian facing the camera
# (from -y). These convert between the two frames.
GEO_FROM_SCREEN = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
SCREEN_FROM_GEO = GEO_FROM_SCREEN.T


# ---------------------------------------------------------------------------
# Projections. Screen layout: x to the right, z up, camera looking from -y.
# Flat maps lie in the plane y = -radius, facing the camera.
# ---------------------------------------------------------------------------
def sphere_xyz(lon, lat, radius):
    lam, phi = np.radians(lon), np.radians(lat)
    return radius * np.c_[np.cos(phi) * np.sin(lam), -np.cos(phi) * np.cos(lam),
                          np.sin(phi)]


def lambert(lon, lat, rc):
    """Equal-area cylindrical: the globe pushed straight out to a cylinder."""
    return rc * np.radians(lon), rc * np.sin(np.radians(lat))


def mercator(lon, lat, rc, max_lat=80.0):
    phi = np.radians(np.clip(lat, -max_lat, max_lat))
    return rc * np.radians(lon), rc * np.log(np.tan(PI / 4 + phi / 2))


def equal_earth(lon, lat, rc):
    """Equal Earth, scaled so the equator keeps its length (2 pi rc)."""
    A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796

    def raw(lam, phi):
        th = np.arcsin(np.sqrt(3) / 2 * np.sin(phi))
        x = (2 * np.sqrt(3) * lam * np.cos(th)
             / (3 * (9 * A4 * th ** 8 + 7 * A3 * th ** 6 + 3 * A2 * th ** 2
                     + A1)))
        return x, th * (A1 + A2 * th ** 2 + A3 * th ** 6 + A4 * th ** 8)

    k = rc * PI / raw(PI, 0.0)[0]
    x, y = raw(np.radians(lon), np.radians(lat))
    return k * x, k * y


def cylinder_stage(lon, lat, rc, target, w, b, m, globe_radius, drop=0.0):
    """The staged unwrapping of a cylindrical-style map.

    w: globe -> wrapped cylinder (each point pushed straight out sideways)
    b: the cylinder, cut along the antimeridian, unbends into a flat sheet
       (arc length kept, curvature (1 - b) / rc)
    m: the flat sheet stretches / reshapes into the target projection
    """
    X0, Z0 = lambert(lon, lat, rc)
    X1, Z1 = target(lon, lat, rc)
    X, Z = (1 - m) * X0 + m * X1, (1 - m) * Z0 + m * Z1
    k = (1 - b) / rc
    if k > 1e-6:
        x, y = np.sin(k * X) / k, -rc + (1 - np.cos(k * X)) / k
    else:
        x, y = X, np.full_like(X, -rc)
    sheet = np.c_[x, y, Z - drop]
    return (1 - w) * sphere_xyz(lon, lat, globe_radius) + w * sheet


def axis_angle(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if np.isclose(ang, 0):
        return np.array([0.0, 0.0, 1.0]), 0.0
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return ax / np.linalg.norm(ax), float(ang)


# ---------------------------------------------------------------------------
class WorldProjections(ThreeDTIMCSlide):
    """Globe -> Mercator -> globe -> Equal Earth -> globe -> Dymaxion."""

    GLOBE_RADIUS = 2.2
    MERCATOR_RC = 1.2      # cylinder radius for each map (sets its size)
    EQUAL_EARTH_RC = 1.9
    NET_WIDTH, NET_HEIGHT = 11.5, 5.8
    DROP = 0.35            # flat maps sit a little low, clear of the title
    FACE_SUBDIVISIONS = 8  # Dymaxion ocean: small triangles per face side

    COAST_WIDTH = 1.2
    COAST_OPACITY = 0.8
    GRATICULE_WIDTH = 0.7
    GRATICULE_OPACITY = 0.5

    GLOBE_CAMERA = dict(phi=72 * DEGREES, theta=-90 * DEGREES)
    FLAT_CAMERA = dict(phi=90 * DEGREES, theta=-90 * DEGREES)
    SPIN_RATE = 0.45
    STEP_SECONDS = 2.5     # each stage of an unwrapping

    # ---- data ------------------------------------------------------------
    def setup_data(self):
        with np.load(DATA_FILE) as z:
            d = {k: z[k] for k in z.files}
        self.line_ll = d["lonlat"]                     # (N, 2) lon, lat
        self.segments, self.group = d["segments"], d["group"]
        self.vertex_cell = d["vertex_cell"]
        cells = d["cells"]                             # (C, 4, 2) corners
        self.n_cells = len(cells)
        self.cell_ll = cells.reshape(-1, 2)            # corners, flattened
        self.cell_land = d["cell_land"].astype(bool)
        # colours read off a map image (if the prep had one), else two tones
        if "palette" in d:
            self.palette = [rgb_to_color(c / 255.0) for c in d["palette"]]
            self.cell_colour = d["cell_colour"]
            ocean_idx = np.bincount(self.cell_colour[~self.cell_land],
                                    minlength=len(self.palette)).argmax()
            self.ocean_colour = self.palette[ocean_idx]
        else:
            self.palette = [OCEAN_COLOR, LAND_COLOR]
            self.cell_colour = self.cell_land.astype(int)
            self.ocean_colour = OCEAN_COLOR
        # fill colours: a palette read off a map image if the prep made one,
        # otherwise plain land / ocean
        if "palette" in d:
            self.cell_group = d["cell_colour"].astype(int)
            self.group_colors = [ManimColor(c / 255.0) for c in d["palette"]]
            ocean_groups = self.cell_group[~self.cell_land]
            self.net_ocean_color = self.group_colors[
                np.bincount(ocean_groups).argmax()]
        else:
            self.cell_group = self.cell_land.astype(int)
            self.group_colors = [OCEAN_COLOR, LAND_COLOR]
            self.net_ocean_color = OCEAN_COLOR

        # Dymaxion: every point goes onto one icosahedron face (mesh cells
        # onto the face of their centre, so each cell stays in one piece)
        self.ico = Icosahedron(d["ico_vertices"], d["ico_faces"],
                               d["ico_hinges"], int(d["ico_root"]))
        geo = lambda ll: sphere_xyz(ll[:, 0], ll[:, 1], 1.0) @ GEO_FROM_SCREEN.T
        self.line_face = self.ico.face_of(geo(self.line_ll))
        self.cell_face = np.repeat(self.ico.face_of(geo(cells.mean(1))), 4)
        self.line_unit, self.cell_unit = geo(self.line_ll), geo(self.cell_ll)
        self.line_gno = self.ico.gnomonic(self.line_unit, self.line_face)
        self.cell_gno = self.ico.gnomonic(self.cell_unit, self.cell_face)
        # Dymaxion ocean: each face cut into small triangles, so the net has
        # exact straight edges (the lon-lat cells would make them jagged)
        n, tris, tri_face = self.FACE_SUBDIVISIONS, [], []
        for fi, (a, b, c) in enumerate(self.ico.faces):
            A, B, C = self.ico.V[a], self.ico.V[b], self.ico.V[c]
            P = lambda i, j: A + (B - A) * i / n + (C - A) * j / n
            for i in range(n):
                for j in range(n - i):
                    tris.append([P(i, j), P(i + 1, j), P(i, j + 1)])
                    tri_face.append(fi)
                    if i + j < n - 1:
                        tris.append([P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)])
                        tri_face.append(fi)
        self.n_tris = len(tris)
        self.tri_face_of = np.array(tri_face)       # face of each triangle
        # each face's three edges as short polylines (they bend with the
        # surface while it morphs between globe and icosahedron)
        t = np.linspace(0, 1, 12)[:, None]
        edge_pts, edge_face = [], []
        for fi, f in enumerate(self.ico.faces):
            for a, b in ((0, 1), (1, 2), (2, 0)):
                edge_pts.append((1 - t) * self.ico.V[f[a]] + t * self.ico.V[f[b]])
                edge_face.append(fi)
        self.edge_gno = np.vstack(edge_pts)
        self.edge_face_of = np.array(edge_face)
        self.edge_face = np.repeat(edge_face, len(t))
        self.edge_unit = self.edge_gno / np.linalg.norm(
            self.edge_gno, axis=1, keepdims=True)
        self.edge_len = len(t)
        self.tri_gno = np.array(tris).reshape(-1, 3)
        self.tri_face = np.repeat(tri_face, 3)
        self.tri_unit = self.tri_gno / np.linalg.norm(self.tri_gno, axis=1,
                                                      keepdims=True)
        # coastline segments that span a cut edge are dropped (the tears)
        f0 = self.line_face[self.segments[:, 0]]
        f1 = self.line_face[self.segments[:, 1]]
        self.seg_not_torn = np.array(
            [a == b or (min(a, b), max(a, b)) in self.ico.hinge_pairs
             for a, b in zip(f0, f1)])
        self._setup_net_layout()

    def _setup_net_layout(self):
        """Rotation that turns the flat net to face the camera and lie
        horizontally, plus the scale/shift that fits it on the slide."""
        T = self.ico.transforms(1.0)
        corners = np.vstack([self.ico.V[self.ico.faces[fi]] @ T[fi][:3, :3].T
                             + T[fi][:3, 3] for fi in self.ico.order])
        corners = corners @ SCREEN_FROM_GEO.T
        n0 = SCREEN_FROM_GEO @ self.ico.normals[self.ico.order[0]]
        target = np.array([0.0, -1.0, 0.0])           # towards the camera
        ax = np.cross(n0, target)
        A1 = (np.eye(3) if np.linalg.norm(ax) < 1e-9
              else rot(ax, np.arccos(np.clip(n0 @ target, -1, 1))))
        flat = corners @ A1.T                         # now in an x-z plane
        xz = flat[:, [0, 2]] - flat[:, [0, 2]].mean(0)
        _, evecs = np.linalg.eigh(xz.T @ xz)
        major = evecs[:, -1]                          # long axis -> screen x
        A2 = rot([0, 1, 0], np.arctan2(major[1], major[0]))
        self.net_axis, self.net_angle = axis_angle(A2 @ A1)
        flat = corners @ (A2 @ A1).T
        lo, hi = flat[:, [0, 2]].min(0), flat[:, [0, 2]].max(0)
        self.net_scale = min(self.NET_WIDTH / (hi[0] - lo[0]),
                             self.NET_HEIGHT / (hi[1] - lo[1]))
        mid = (lo + hi) / 2
        self.net_shift = np.array([-mid[0], 0.0, -mid[1]]) * self.net_scale

    # ---- positions ---------------------------------------------------------
    def positions(self, ll, gno, face, unit):
        if self.mode == "dymaxion":
            return self.dymaxion_positions(unit, gno, face)
        rc, target = ((self.MERCATOR_RC, mercator) if self.mode == "mercator"
                      else (self.EQUAL_EARTH_RC, equal_earth))
        return cylinder_stage(ll[:, 0], ll[:, 1], rc, target,
                              self.wrap.get_value(), self.unbend.get_value(),
                              self.stretch.get_value(), self.GLOBE_RADIUS,
                              self.DROP)

    def dymaxion_positions(self, unit, gno, face):
        """wrap: globe -> icosahedron; unfold: icosahedron -> flat net
        (turned to face the camera and scaled to fit as it unfolds)."""
        t, u = self.wrap.get_value(), self.unfold.get_value()
        T = self.ico.transforms(u)
        P = np.empty_like(gno)
        for fi, M in T.items():
            sel = face == fi
            if sel.any():
                P[sel] = gno[sel] @ M[:3, :3].T + M[:3, 3]
        P = P @ SCREEN_FROM_GEO.T @ rot(self.net_axis, self.net_angle * u).T
        P = P * ((1 - u) * self.GLOBE_RADIUS + u * self.net_scale) \
            + u * (self.net_shift + np.array([0.0, 0.0, -self.DROP]))
        S = unit @ SCREEN_FROM_GEO.T * self.GLOBE_RADIUS
        return (1 - t) * S + t * P

    def cell_visibility(self):
        """Cell corners now, and which cells face the camera (back faces are
        hidden, since filled shapes aren't depth-sorted)."""
        C = self.positions(self.cell_ll, self.cell_gno, self.cell_face,
                           self.cell_unit).reshape(self.n_cells, 4, 3)
        n = np.cross(C[:, 2] - C[:, 0], C[:, 3] - C[:, 1])
        return C, n @ self.camera_direction()

    def camera_direction(self):
        phi, theta = self.camera.get_phi(), self.camera.get_theta()
        return np.array([np.sin(phi) * np.cos(theta),
                         np.sin(phi) * np.sin(theta), np.cos(phi)])

    def ocean_triangles(self):
        """Dymaxion ocean triangles, and how much each faces the camera
        (negative = we see its inside)."""
        T = self.dymaxion_positions(self.tri_unit, self.tri_gno,
                                    self.tri_face).reshape(self.n_tris, 3, 3)
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        return T, n @ self.camera_direction()

    # ---- mobjects ------------------------------------------------------------
    def build_fills(self):
        """One VMobject of small filled cells per palette colour, plus the
        Dymaxion ocean (exact icosahedron faces, so the net's edges are
        straight)."""
        def quads(C):
            """Closed polygons (quads or triangles) as one VMobject."""
            if len(C) == 0:
                return np.zeros((0, 3))
            A, B = C, np.roll(C, -1, axis=1)          # each side A -> B
            k = C.shape[1]
            pts = np.empty((len(C), k, 4, 3))
            pts[:, :, 0], pts[:, :, 3] = A, B
            pts[:, :, 1] = A + (B - A) / 3
            pts[:, :, 2] = A + 2 * (B - A) / 3
            return pts.reshape(-1, 3)

        def fill(colour):
            return VMobject(fill_color=colour, fill_opacity=1,
                            stroke_color=colour, stroke_width=0.6)

        def darker(c):
            return interpolate_color(ManimColor(c), BLACK, INSIDE_DARKEN)

        # Draw order (no depth sorting): first the inside of the surface,
        # darker, then the outside on top. On the closed globe the inside is
        # entirely hidden; as the map unrolls or unfolds it shows through.
        n_col = len(self.palette)
        inside = [fill(darker(self.ocean_colour))] + \
            [fill(darker(c)) for c in self.palette]
        outside = [fill(self.ocean_colour)] + [fill(c) for c in self.palette]
        eps = 1e-9

        def update(group):
            C, facing = self.cell_visibility()
            front, back = facing > eps, facing < -eps
            self._visible = front
            if self.wrap.get_value() <= 0:        # closed globe: no inside
                back = np.zeros_like(back)
            if self.mode == "dymaxion":
                T, tf = self.ocean_triangles()
                group[0].set_points(quads(T[tf < -eps]) if back.any()
                                    else np.zeros((0, 3)))
                group[n_col + 1].set_points(quads(T[tf > eps]))
                front = front & self.cell_land    # ocean comes from faces
                back = back & self.cell_land
            else:
                group[0].set_points(np.zeros((0, 3)))
                group[n_col + 1].set_points(np.zeros((0, 3)))
            for k in range(n_col):
                this = self.cell_colour == k
                group[1 + k].set_points(quads(C[back & this]))
                group[n_col + 2 + k].set_points(quads(C[front & this]))

        fills = VGroup(*inside, *outside)
        update(fills)
        fills.add_updater(update)
        return fills

    def build_facets(self):
        """Dymaxion only: per-face shading (a translucent dark overlay set
        by a light just above-left of the camera) and the icosahedron's
        edges, so the folded map reads as a solid."""
        shades = VGroup(*[VMobject(fill_color=BLACK, fill_opacity=0,
                                   stroke_width=0) for _ in self.ico.faces])
        edges = VMobject(stroke_color=EDGE_COLOR, stroke_width=EDGE_WIDTH,
                         stroke_opacity=EDGE_OPACITY)

        def tris_points(T):
            if len(T) == 0:
                return np.zeros((0, 3))
            A, B = T, np.roll(T, -1, axis=1)
            pts = np.empty((len(T), 3, 4, 3))
            pts[:, :, 0], pts[:, :, 3] = A, B
            pts[:, :, 1] = A + (B - A) / 3
            pts[:, :, 2] = A + 2 * (B - A) / 3
            return pts.reshape(-1, 3)

        def update(group):
            sh, ed = group
            if self.mode != "dymaxion" or self.wrap.get_value() <= 0:
                for m in sh:
                    m.set_points(np.zeros((0, 3)))
                ed.set_points(np.zeros((0, 3)))
                return
            t = self.wrap.get_value()
            cam = self.camera_direction()
            phi, theta = self.camera.get_phi(), self.camera.get_theta()
            right = np.array([-np.sin(theta), np.cos(theta), 0.0])
            up = np.cross(cam, right)
            light = cam + 0.6 * up - 0.45 * right
            light /= np.linalg.norm(light)
            T, facing = self.ocean_triangles()
            n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            for fi, m in enumerate(sh):
                sel = (self.tri_face_of == fi) & (facing > 1e-9)
                if not sel.any():
                    m.set_points(np.zeros((0, 3)))
                    continue
                lit = np.clip(n[sel].mean(0) @ light, 0, 1)
                m.set_points(tris_points(T[sel]))
                m.set_fill(BLACK, opacity=t * FACE_SHADING * (1 - lit))
            P = self.dymaxion_positions(self.edge_unit, self.edge_gno,
                                        self.edge_face)
            P = P.reshape(-1, self.edge_len, 3)
            # keep the edges of faces that face the camera
            face_front = np.array([
                (facing[self.tri_face_of == fi] > 1e-9).mean() > 0.5
                for fi in range(len(self.ico.faces))])
            P = P[face_front[self.edge_face_of]]
            A, B = P[:, :-1].reshape(-1, 3), P[:, 1:].reshape(-1, 3)
            pts = np.empty((4 * len(A), 3))
            pts[0::4], pts[3::4] = A, B
            pts[1::4] = A + (B - A) / 3
            pts[2::4] = A + 2 * (B - A) / 3
            ed.set_points(pts)
            ed.set_stroke(opacity=t * EDGE_OPACITY)

        facets = VGroup(shades, edges)
        update(facets)
        facets.add_updater(update)
        return facets

    def build_lines(self, group, color, width, opacity=1.0):
        vm = VMobject(stroke_color=color, stroke_width=width,
                      stroke_opacity=opacity)
        sel = self.group == group

        def update(m):
            seg = self.segments[sel]
            ok = self._visible[self.vertex_cell[seg[:, 0]]]
            if self.mode == "dymaxion":
                ok &= self.seg_not_torn[sel]
            seg = seg[ok]
            P = self.positions(self.line_ll, self.line_gno, self.line_face,
                               self.line_unit)
            A, B = P[seg[:, 0]], P[seg[:, 1]]
            pts = np.empty((4 * len(A), 3))
            pts[0::4], pts[3::4] = A, B
            pts[1::4] = A + (B - A) / 3
            pts[2::4] = A + 2 * (B - A) / 3
            m.set_points(pts)

        update(vm)
        vm.add_updater(update)
        return vm

    # ---- labels ---------------------------------------------------------------
    def label(self, name, note):
        t = Text(name, font_size=36, color=DEFAULT_COLOR).to_edge(UP, buff=0.3)
        s = Text(note, font_size=22, color=ACCENT_COLOR).next_to(t, DOWN,
                                                                  buff=0.12)
        return VGroup(t, s)

    def swap_label(self, old, name, note, run_time=0.8):
        """Cross-fade the heading (FadeTransform leaves a copy behind that
        then gets drawn in 3D, so use a plain fade)."""
        new = self.label(name, note)
        self.add_fixed_in_frame_mobjects(new)
        self.play(FadeOut(old), FadeIn(new), run_time=run_time)
        self.remove_fixed_in_frame_mobjects(old)
        return new

    # ---- animation pieces ---------------------------------------------------
    def unwrap(self):
        """Globe -> cylinder -> cut and unrolled -> stretched."""
        st = self.STEP_SECONDS
        self.play(self.wrap.animate.set_value(1.0), run_time=st)
        self.move_camera(**self.FLAT_CAMERA, run_time=st + 0.5,
                         added_anims=[self.unbend.animate.set_value(1.0)])
        self.play(self.stretch.animate.set_value(1.0), run_time=st)

    def rewrap(self):
        """The same three stages in reverse."""
        st = self.STEP_SECONDS
        self.play(self.stretch.animate.set_value(0.0), run_time=st)
        self.move_camera(**self.GLOBE_CAMERA, run_time=st + 0.5,
                         added_anims=[self.unbend.animate.set_value(0.0)])
        self.play(self.wrap.animate.set_value(0.0), run_time=st)

    # ---- scene -----------------------------------------------------------------
    def construct(self):
        self.setup_data()
        self.mode = "mercator"
        self.wrap = ValueTracker(0.0)
        self.unbend = ValueTracker(0.0)
        self.stretch = ValueTracker(0.0)
        self.unfold = ValueTracker(0.0)
        self.set_camera_orientation(**self.GLOBE_CAMERA)

        fills = self.build_fills()
        facets = self.build_facets()
        grat = self.build_lines(1, GRATICULE_COLOR, self.GRATICULE_WIDTH,
                                self.GRATICULE_OPACITY)
        coast = self.build_lines(0, COAST_COLOR, self.COAST_WIDTH,
                                 self.COAST_OPACITY)

        head = self.label("The globe",
                          "every flat map has to tear or distort it somewhere")
        self.add_fixed_in_frame_mobjects(head)
        self.play(FadeIn(head))
        self.play(FadeIn(fills), FadeIn(grat), FadeIn(coast), run_time=1.5)
        self.add(fills, facets, grat, coast)

        # --- a turn around the globe (loops seamlessly) ---
        self.marked_next_slide(loop=True)
        self.begin_ambient_camera_rotation(rate=self.SPIN_RATE)
        self.wait(TAU / self.SPIN_RATE)
        self.stop_ambient_camera_rotation()

        # --- Mercator ---
        self.marked_next_slide()
        head = self.swap_label(head, "Mercator",
                               "cut, unroll, then stretch towards the poles "
                               "to keep angles true")
        self.unwrap()
        self.marked_next_slide()
        self.rewrap()

        # --- Equal Earth ---
        self.marked_next_slide()
        self.mode = "equal_earth"
        head = self.swap_label(head, "Equal Earth",
                               "cut, unroll, then reshape to keep areas true")
        self.unwrap()
        self.marked_next_slide()
        self.rewrap()

        # --- Dymaxion: onto an icosahedron, then unfold ---
        self.marked_next_slide()
        self.mode = "dymaxion"
        head = self.swap_label(head, "Dymaxion",
                               "the globe projected onto an icosahedron")
        self.play(self.wrap.animate.set_value(1.0),
                  run_time=2 * self.STEP_SECONDS)
        self.marked_next_slide()
        head = self.swap_label(head, "Dymaxion",
                               "unfolded with every cut in the ocean: "
                               "the continents stay whole")
        self.move_camera(**self.FLAT_CAMERA, run_time=3 * self.STEP_SECONDS,
                         added_anims=[self.unfold.animate.set_value(1.0)])
        self.marked_next_slide()

        # --- fold back up to the globe ---
        self.move_camera(**self.GLOBE_CAMERA, run_time=2 * self.STEP_SECONDS,
                         added_anims=[self.unfold.animate.set_value(0.0)])
        head = self.swap_label(head, "The globe",
                               "every flat map has to tear or distort it "
                               "somewhere")
        self.play(self.wrap.animate.set_value(0.0),
                  run_time=2 * self.STEP_SECONDS)
        self.marked_next_slide()
