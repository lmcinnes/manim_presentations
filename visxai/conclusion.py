"""
Conclusion: the shape of the argument, what it means for reading a layout,
and questions to take away.

  1. The ladder: a story of giving up trust, rung by rung -- everything,
     global distances, exact local distances, their rank order, even the
     neighbours -- with what survives at each step, down to a corrupted kNN
     graph read as topology and repaired using global information.
  2. The thesis: every layout's kernel is a set of assumptions about how
     much to trust the graph (the kernel formula returns, with its tags).
  3. Reading a layout: what it repairs, and what it declines to preserve.
  4. Questions for the audience.

    manim-slides render conclusion.py Conclusion
"""

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
    TIMCSlide,
    make_heading,
    SmallText,
    mixbox_interpolate_color,
    cyclic_color,
)

import numpy as np

apply_defaults()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
LOST_COLOR = HIGHLIGHT_COLOR  # what we stop trusting / give up
KEPT_COLOR = COLOR_CYCLE[2]  # what survives / is repaired
ICON_COLOR = DEFAULT_COLOR
SHORTCUT_COLOR = HIGHLIGHT_COLOR
STRESS_COLOR = ManimColor("#9e1b32")  # as in the swiss roll stress map
SIGMA_COLOR, M_COLOR, ALPHA_COLOR = COLOR_CYCLE[0], COLOR_CYCLE[3], COLOR_CYCLE[2]
TAIL_COLOR = HIGHLIGHT_COLOR

RUNG_Y0, RUNG_STEP = 2.0, 0.9  # ladder rows

# Contact details, faded into the bottom corners of the last slide.
# By default each gets a simple drawn icon (an envelope, a speech bubble);
# to use logo files instead, give their paths here (e.g. the official SVGs
# from the services' brand / press kits) and they are loaded and sized to fit.
CONTACT_EMAIL = "leland.mcinnes@gmail.com"
CONTACT_SOCIAL = "@leland_mcinnes"
EMAIL_ICON_SVG = "gmail.svg"  # e.g. "assets/email_logo.svg"
SOCIAL_ICON_SVG = "x_logo.svg"  # e.g. "assets/x_logo.svg"
CONTACT_ICON_HEIGHT = 0.3
X_LOST, X_ICON, X_KEPT = -3.9, -0.55, 3.2
TEXT_SIZE = 22


# ---------------------------------------------------------------------------
# Icons: filled illustrations on tinted tiles. Each icon_* returns a tile
# (a VGroup) with an .intro() method giving its entrance animation; build the
# tile, place it, then play tile.intro() instead of FadeIn(tile).
# ---------------------------------------------------------------------------
LADDER_TILE = (1.15, 0.8)  # rows are RUNG_STEP = 0.9 apart
QUESTION_TILE = (1.4, 1.4)


def tint(color, t):
    return interpolate_color(ManimColor(color), WHITE, t)


def shade(color, t):
    return interpolate_color(ManimColor(color), BLACK, t)


def shaded_disc(r, color, center=ORIGIN, n=12, highlight=True):
    """A disc lit from the upper left: stacked circles from dark rim to light
    centre, plus a soft specular highlight."""
    color = ManimColor(color)
    dark, light = shade(color, 0.25), tint(color, 0.45)
    g = VGroup(
        Circle(radius=r)
        .set_fill(dark, 1)
        .set_stroke(shade(color, 0.45), max(0.6, 8 * r))
    )
    for k in range(1, n):
        t = k / n
        g.add(
            Circle(radius=r * (1 - 0.85 * t))
            .set_fill(interpolate_color(dark, light, t), 1)
            .set_stroke(width=0)
            .shift(0.3 * r * t * normalize(UL))
        )
    if highlight:
        g.add(
            Ellipse(width=0.5 * r, height=0.3 * r)
            .rotate(PI / 4)
            .set_fill(WHITE, 0.55)
            .set_stroke(width=0)
            .shift(0.45 * r * normalize(UL))
        )
    return g.shift(center)


def glow(mob, color, layers=4, width=3):
    """mob over a halo of wider, fainter strokes."""
    halo = VGroup(
        *[
            mob.copy().set_stroke(color, width + 3 * k, opacity=0.1)
            for k in range(layers, 0, -1)
        ]
    )
    return VGroup(halo, mob)


def zigzag(a, b, coils=7, amp=0.05, lead=0.12):
    """Points of a zig-zag spring from a to b, straight for `lead` at each end."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    length = np.linalg.norm(d)
    u = d / length
    n = np.array([-u[1], u[0], 0])
    s0, s1 = lead * length, (1 - lead) * length
    pts = [a, a + s0 * u]
    for k in range(1, 2 * coils):
        s = s0 + (s1 - s0) * k / (2 * coils)
        pts.append(a + s * u + (1 if k % 2 else -1) * amp * n)
    return pts + [a + s1 * u, b]


def icon_tile(content, color=ACCENT_COLOR, size=QUESTION_TILE, pad=0.13, intro=None):
    """content fitted onto a rounded, gradient-filled tile with a soft shadow.
    intro(content) -> animation for the content (default: FadeIn)."""
    w, h = size
    m = min(w, h)
    bg = RoundedRectangle(width=w, height=h, corner_radius=0.22 * m)
    bg.set_fill([tint(color, 0.95), tint(color, 0.8)], opacity=1)
    bg.set_sheen_direction(DR)
    bg.set_stroke(tint(color, 0.35), 1.5)
    shadow = (
        bg.copy()
        .set_fill(DEFAULT_COLOR, 0.14)
        .set_sheen_direction(RIGHT)
        .set_stroke(width=0)
        .shift(0.04 * m * DR)
    )
    content.scale(min((w - 2 * pad) / content.width, (h - 2 * pad) / content.height))
    content.move_to(bg)
    # intros can add content to the scene before the tile; keep it beneath
    shadow.set_z_index(-2)
    bg.set_z_index(-1)
    tile = VGroup(shadow, bg, content)
    intro = intro or (lambda c: FadeIn(c))

    def tile_intro():
        return LaggedStart(
            FadeIn(VGroup(shadow, bg), scale=0.85), intro(content), lag_ratio=0.35
        )

    tile.intro = tile_intro
    return tile


# ---- the ladder (LADDER_TILE) ---------------------------------------------
def icon_globe():
    """Everything: a globe beside its peeled-off gores -- no flat map keeps it
    all."""
    sea = COLOR_CYCLE[0]
    r = 0.3
    sphere = shaded_disc(r, sea)
    grid = VGroup()
    for w in (0.4, 0.8):
        grid.add(Ellipse(width=2 * r * w, height=2 * r))
    grid.add(Line(LEFT * r, RIGHT * r), Line(UP * r, DOWN * r))
    for y in (0.5, -0.5):
        hw = r * np.sqrt(1 - y**2)
        grid.add(Line([-hw, y * r, 0], [hw, y * r, 0]))
    grid.set_stroke(WHITE, 1.2, opacity=0.55)
    globe = VGroup(sphere, grid).shift(0.33 * LEFT)

    gores = VGroup()
    gh, gw = 0.62, 0.11
    ys = np.linspace(-gh / 2, gh / 2, 25)
    for k in range(5):
        xs = gw / 2 * np.cos(PI * ys / gh)
        gore = Polygon(
            *[[x, y, 0] for x, y in zip(xs, ys)],
            *[[-x, y, 0] for x, y in zip(xs[::-1], ys[::-1])],
        )
        gore.set_fill([tint(sea, 0.35), shade(sea, 0.1)], 1).set_sheen_direction(DOWN)
        gore.set_stroke(shade(sea, 0.4), 0.8)
        gores.add(gore.move_to([0.18 + k * gw * 1.02, 0, 0]))
    content = VGroup(globe, gores)

    def intro(c):
        return LaggedStart(
            GrowFromCenter(globe),
            LaggedStart(
                *[FadeIn(g, shift=0.25 * RIGHT, scale=0.6) for g in gores],
                lag_ratio=0.15,
            ),
            lag_ratio=0.5,
        )

    return icon_tile(content, sea, LADDER_TILE, intro=intro)


def icon_roll():
    """Global distances: a rolled strip, coloured along its length, and a
    straight chord across the gap that unrolling stretches."""
    a = 0.026
    p = lambda s: np.array([a * s * np.cos(s), a * s * np.sin(s), 0])
    t = np.linspace(1.5 * np.pi, 4.3 * np.pi, 200)
    spiral = VMobject().set_points_smoothly([p(s) for s in t])
    spiral.set_stroke([cyclic_color(u) for u in np.linspace(0, 0.85, 12)], 7)
    spiral.set_cap_style(CapStyleType.ROUND)
    s1, s2 = 2.0 * np.pi, 4.0 * np.pi  # adjacent layers
    chord = glow(Line(p(s1), p(s2)).set_stroke(STRESS_COLOR, 3), STRESS_COLOR)
    ends = VGroup(
        *[shaded_disc(0.045, STRESS_COLOR, p(s), n=4, highlight=False) for s in (s1, s2)]
    )
    content = VGroup(spiral, chord, ends)

    def intro(c):
        return Succession(
            Create(spiral, run_time=0.8),
            AnimationGroup(Create(chord), FadeIn(ends, scale=0.3), run_time=0.5),
        )

    return icon_tile(content, STRESS_COLOR, LADDER_TILE, intro=intro)


def icon_concentration():
    """Exact local distances: every neighbour at nearly the same distance --
    a thin shell."""
    R = 0.32
    shell = Annulus(inner_radius=R - 0.035, outer_radius=R + 0.035)
    shell.set_fill(ACCENT_COLOR, 0.35).set_stroke(ACCENT_COLOR, 0.8, opacity=0.6)
    rng = np.random.default_rng(1)
    angles = np.linspace(0, TAU, 8)[:-1] + 0.3
    finals = [
        R * (1 + 0.04 * rng.standard_normal()) * np.array([np.cos(a), np.sin(a), 0])
        for a in angles
    ]
    nbrs = VGroup(
        *[
            shaded_disc(0.05, cyclic_color(k / len(finals)), p, n=4)
            for k, p in enumerate(finals)
        ]
    )
    spokes = VGroup(*[Line(ORIGIN, p) for p in finals]).set_stroke(
        ACCENT_COLOR, 1.2, opacity=0.8
    )
    centre = shaded_disc(0.07, DEFAULT_COLOR)
    content = VGroup(shell, spokes, nbrs, centre)
    starts = rng.uniform(0.25, 1.6, len(finals))

    def intro(c):
        moves = []
        for n_, s in zip(nbrs, starts):
            n_.generate_target()
            n_.move_to(
                centre.get_center() + s * (n_.get_center() - centre.get_center())
            )
            moves.append(MoveToTarget(n_))
        return LaggedStart(
            GrowFromCenter(centre),
            AnimationGroup(FadeIn(shell), *moves, run_time=0.9),
            Create(spokes),
            lag_ratio=0.4,
        )

    return icon_tile(content, ACCENT_COLOR, LADDER_TILE, intro=intro)


def badge(label, color, r=0.1):
    disc = shaded_disc(r, color, n=4, highlight=False)
    txt = SmallText(label, font_size=20, color=WHITE, weight=BOLD)
    txt.set(height=1.1 * r).move_to(disc)
    return VGroup(disc, txt)


def icon_rank():
    """Their rank order: nearer neighbours carry the wrong ranks -- the badges
    shuffle out of order."""
    rings = VGroup(
        *[DashedVMobject(Circle(radius=rad), num_dashes=22) for rad in (0.14, 0.24, 0.34)]
    ).set_stroke(ACCENT_COLOR, 1.2, opacity=0.7)
    window = Circle(radius=0.44).set_fill(ACCENT_COLOR, 0.12).set_stroke(width=0)
    slots = [
        rad * np.array([np.cos(ang), np.sin(ang), 0])
        for ang, rad in ((0.5, 0.14), (2.4, 0.24), (4.4, 0.34))
    ]
    # final: nearest shows 2, middle 3, farthest 1
    labels = ("2", "3", "1")
    badges = VGroup(
        *[badge(l, LOST_COLOR).move_to(p) for l, p in zip(labels, slots)]
    )
    centre = shaded_disc(0.06, DEFAULT_COLOR)
    content = VGroup(window, rings, badges, centre)

    def intro(c):
        # start in the honest order (1, 2, 3 outward), then cycle
        pos = [b.get_center() for b in badges]
        for b, j in zip(badges, (1, 2, 0)):
            b.generate_target()
            b.move_to(pos[j])
        return Succession(
            AnimationGroup(FadeIn(window), Create(rings), GrowFromCenter(centre)),
            LaggedStart(*[FadeIn(b, scale=0.4) for b in badges], lag_ratio=0.25),
            AnimationGroup(*[MoveToTarget(b, path_arc=PI / 2) for b in badges]),
        )

    return icon_tile(content, LOST_COLOR, LADDER_TILE, intro=intro)


def icon_shortcut():
    """Even the neighbours: a ring of points with a shortcut across it."""
    n, r = 12, 0.34
    P = [
        r * np.array([np.cos(a), np.sin(a), 0]) for a in np.linspace(0, TAU, n + 1)[:-1]
    ]
    edges = VGroup(
        *[
            Line(P[i], P[(i + 1) % n]).set_stroke(
                [cyclic_color(i / n), cyclic_color((i + 1) / n)], 2.5
            )
            for i in range(n)
        ]
    )
    cut = glow(Line(P[1], P[7]).set_stroke(SHORTCUT_COLOR, 3.5), SHORTCUT_COLOR)
    pts = VGroup(
        *[shaded_disc(0.05, cyclic_color(i / n), p, n=4) for i, p in enumerate(P)]
    )
    content = VGroup(edges, cut, pts)

    def intro(c):
        return Succession(
            AnimationGroup(
                LaggedStart(*[FadeIn(p, scale=0.3) for p in pts], lag_ratio=0.08),
                Create(edges),
            ),
            Create(cut, run_time=0.6),
        )

    return icon_tile(content, SHORTCUT_COLOR, LADDER_TILE, intro=intro)


def icon_tail():
    """The repair: a pull that rises, peaks and fades slowly (the heavy tail),
    against a light tail that dies quickly."""
    w, h = 1.0, 0.55
    ax = VGroup(
        Line([-w / 2, -h / 2, 0], [w / 2, -h / 2, 0]),
        Line([-w / 2, -h / 2, 0], [-w / 2, h / 2, 0]),
    ).set_stroke(DEFAULT_COLOR, 1.5)
    xs = np.linspace(0, 4, 80)
    to_pt = lambda x, y: [-w / 2 + w * x / 4, -h / 2 + h * y / 1.05, 0]
    heavy = 2 * xs / (1 + xs**2)
    light = np.sqrt(2 * np.e) * xs * np.exp(-(xs**2))  # same peak height
    curve = VMobject().set_points_smoothly([to_pt(x, y) for x, y in zip(xs, heavy)])
    curve.set_stroke(shade(KEPT_COLOR, 0.15), 3.5)
    area = Polygon(
        to_pt(0, 0), *[to_pt(x, y) for x, y in zip(xs, heavy)], to_pt(4, 0)
    )
    area.set_fill([tint(KEPT_COLOR, 0.1), tint(KEPT_COLOR, 0.8)], 0.85)
    area.set_sheen_direction(DOWN).set_stroke(width=0)
    ref = DashedVMobject(
        VMobject().set_points_smoothly([to_pt(x, y) for x, y in zip(xs, light)]),
        num_dashes=18,
    ).set_stroke(ACCENT_COLOR, 2)
    content = VGroup(area, ax, ref, curve)

    def intro(c):
        return Succession(
            AnimationGroup(Create(ax), Create(ref)),
            AnimationGroup(Create(curve), FadeIn(area), run_time=0.8),
        )

    return icon_tile(content, KEPT_COLOR, (1.2, 0.85), intro=intro)


# ---- the questions (QUESTION_TILE) ----------------------------------------
def icon_noise():
    """Noise in your representation: an isotropic cloud against a structured,
    stretched one."""
    rng = np.random.default_rng(3)

    def blob(A, color, centre):
        contours = VGroup(
            *[
                Circle(radius=s)
                .apply_matrix(A)
                .set_fill(color, 0.16 + 0.1 * k)
                .set_stroke(color, 1, opacity=0.5)
                for k, s in enumerate((2.3, 1.6, 0.9))
            ]
        )
        pts = VGroup(
            *[
                Dot(
                    (A @ np.clip(rng.standard_normal(2), -2, 2)).tolist() + [0],
                    radius=0.018,
                    color=shade(color, 0.35),
                )
                for _ in range(45)
            ]
        )
        return VGroup(contours, pts).shift(centre)

    iso = blob(0.11 * np.eye(2), COLOR_CYCLE[0], [-0.24, 0.2, 0])
    rot = np.array([[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]])
    aniso = blob(rot @ np.diag([0.2, 0.045]), COLOR_CYCLE[3], [0.16, -0.16, 0])
    content = VGroup(iso, aniso)

    def intro(c):
        return LaggedStart(
            *[
                AnimationGroup(
                    LaggedStart(
                        *[GrowFromCenter(e) for e in b[0][::-1]], lag_ratio=0.25
                    ),
                    LaggedStart(*[FadeIn(d, scale=0.2) for d in b[1]], lag_ratio=0.02),
                )
                for b in (iso, aniso)
            ],
            lag_ratio=0.5,
        )

    return icon_tile(content, ACCENT_COLOR, intro=intro)


def icon_dial():
    """Setting the knobs from the data: a histogram feeding three knobs, one
    per parameter, in the colours of the kernel formula."""
    rng = np.random.default_rng(7)
    counts = np.histogram(rng.standard_normal(400), bins=9, range=(-2.6, 2.6))[0]
    bw, bh = 0.09, 0.32
    bars = VGroup(
        *[
            Rectangle(width=bw * 0.8, height=max(bh * c / counts.max(), 0.01))
            .set_fill([tint(ACCENT_COLOR, 0.3), shade(ACCENT_COLOR, 0.15)], 1)
            .set_sheen_direction(DOWN)
            .set_stroke(width=0)
            .move_to([(k - 4) * bw, 0.17, 0], aligned_edge=DOWN)
            for k, c in enumerate(counts)
        ]
    )
    base = Line([-0.45, 0.17, 0], [0.45, 0.17, 0]).set_stroke(DEFAULT_COLOR, 1.2)

    knobs, pointers, sweeps = VGroup(), VGroup(), []
    settings = ((SIGMA_COLOR, r"\sigma_0", 0.6), (M_COLOR, "m", -0.5), (ALPHA_COLOR, r"\alpha", 1.3))
    for k, (color, sym, ang) in enumerate(settings):
        c = np.array([(k - 1) * 0.4, -0.22, 0])
        ticks = VGroup(
            *[
                Dot(c + 0.175 * np.array([np.cos(a), np.sin(a), 0]), radius=0.011)
                for a in np.linspace(PI / 6, 5 * PI / 6, 5)
            ]
        ).set_color(ACCENT_COLOR)
        body = shaded_disc(0.13, color, c)
        d = np.array([np.cos(PI / 2 + ang), np.sin(PI / 2 + ang), 0])
        pointer = Line(c + 0.03 * d, c + 0.11 * d).set_stroke(WHITE, 3)
        pointer.set_cap_style(CapStyleType.ROUND)
        lab = MathTex(sym, color=shade(color, 0.2), font_size=22).next_to(
            body, DOWN, buff=0.06
        )
        knobs.add(VGroup(ticks, body, pointer, lab))
        pointers.add(pointer)
        sweeps.append(ang)
    content = VGroup(bars, base, knobs)

    def intro(c):
        turns = []
        for knob, pointer, ang in zip(knobs, pointers, sweeps):
            centre = knob[1].get_center()
            pointer.rotate(-ang - 0.8, about_point=centre)
            turns.append(Rotate(pointer, ang + 0.8, about_point=centre))
        return Succession(
            AnimationGroup(
                Create(base),
                LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.1),
                LaggedStart(*[FadeIn(k, scale=0.6) for k in knobs], lag_ratio=0.2),
            ),
            AnimationGroup(*turns, run_time=0.9),
        )

    return icon_tile(content, ACCENT_COLOR, intro=intro)


def icon_given_up():
    """Showing what was given up: two clusters keep their short edges while
    the long edge between them is stretched taut."""
    rng = np.random.default_rng(5)

    def cluster(centre, color):
        ang = np.linspace(0, TAU, 7)[:-1] + rng.uniform(0, TAU) + 0.25 * rng.standard_normal(6)
        rad = 0.12 * (1 + 0.15 * rng.standard_normal(6))
        P = np.vstack([[0, 0, 0], np.c_[rad * np.cos(ang), rad * np.sin(ang), np.zeros(6)]])
        P = P + centre
        pairs = {
            tuple(sorted((i, int(j))))
            for i in range(len(P))
            for j in np.argsort(np.linalg.norm(P - P[i], axis=1))[1:3]
        }
        edges = VGroup(*[Line(P[i], P[j]) for i, j in sorted(pairs)])
        edges.set_stroke(KEPT_COLOR, 2.5)
        pts = VGroup(*[shaded_disc(0.045, color, p, n=4) for p in P])
        return VGroup(edges, pts), P

    left, PL = cluster(np.array([-0.34, -0.16, 0]), COLOR_CYCLE[0])
    right, PR = cluster(np.array([0.34, 0.16, 0]), COLOR_CYCLE[4])
    ia = np.argmax(PL[:, 0] + PL[:, 1])  # the facing points
    ib = np.argmin(PR[:, 0] + PR[:, 1])
    pa, pb = left[1][ia], right[1][ib]
    rest = 0.3  # natural length of the spring

    def make_spring(a, b, unit):
        length = np.linalg.norm(b - a) / unit
        amp = 0.045 * unit * np.clip(rest / length, 0.6, 1.2)
        z = VMobject().set_points_as_corners(zigzag(a, b, coils=6, amp=amp))
        return glow(z.set_stroke(STRESS_COLOR, 2.5), STRESS_COLOR, layers=3)

    spring = make_spring(pa.get_center(), pb.get_center(), 1.0)
    content = VGroup(spring, left, right)

    def intro(c):
        unit = c.width / content_width
        L0, R0 = left.get_center(), right.get_center()
        squeeze = 0.2 * (R0 - L0)

        def stretch(_, a):
            left.move_to(L0 + (1 - a) * squeeze)
            right.move_to(R0 - (1 - a) * squeeze)
            spring.become(make_spring(pa.get_center(), pb.get_center(), unit))

        stretch(None, 0)  # start squeezed together
        return Succession(
            AnimationGroup(FadeIn(left), FadeIn(right), FadeIn(spring)),
            UpdateFromAlphaFunc(Group(), stretch, run_time=1.2, rate_func=smooth),
        )

    content_width = content.width
    return icon_tile(content, ACCENT_COLOR, intro=intro)


def icon_envelope():
    body = Rectangle(width=0.42, height=0.28, color=ICON_COLOR, stroke_width=2)
    flap = VMobject(color=ICON_COLOR, stroke_width=2).set_points_as_corners(
        [body.get_corner(UL), body.get_center() + 0.02 * DOWN, body.get_corner(UR)]
    )
    return VGroup(body, flap)


def icon_speech_bubble():
    bubble = RoundedRectangle(
        width=0.42, height=0.28, corner_radius=0.08, color=ICON_COLOR, stroke_width=2
    )
    tail = VMobject(color=ICON_COLOR, stroke_width=2).set_points_as_corners(
        [
            bubble.get_bottom() + 0.12 * LEFT,
            bubble.get_bottom() + 0.16 * LEFT + 0.1 * DOWN,
            bubble.get_bottom() + 0.02 * LEFT,
        ]
    )
    dots = VGroup(
        *[
            Dot(bubble.get_center() + x * RIGHT, radius=0.022, color=ICON_COLOR)
            for x in (-0.09, 0, 0.09)
        ]
    )
    return VGroup(bubble, tail, dots)


def contact_icon(svg_path, fallback):
    """A logo file if one is given (and exists), else a drawn icon."""
    from pathlib import Path

    if svg_path and Path(svg_path).exists():
        return SVGMobject(svg_path).set(height=CONTACT_ICON_HEIGHT)
    return fallback().set(height=CONTACT_ICON_HEIGHT)


# ---------------------------------------------------------------------------
class Conclusion(TIMCSlide):

    def heading(self, title, subtitle):
        return make_heading(title, subtitle)

    def swap(self, old, title, subtitle, *anims):
        new = self.heading(title, subtitle)
        self.play(FadeTransform(old, new), *anims)
        return new

    def rung(self, k, lost, icon, kept):
        y = RUNG_Y0 - k * RUNG_STEP
        lost_t = SmallText(lost, font_size=TEXT_SIZE, color=LOST_COLOR).move_to(
            [X_LOST, y, 0]
        )
        icon.move_to([X_ICON, y, 0])
        kept_t = SmallText(kept, font_size=TEXT_SIZE, color=KEPT_COLOR).move_to(
            [X_KEPT, y, 0]
        )
        arrow = Arrow(
            [X_ICON + 0.75, y, 0],
            [X_KEPT - kept_t.width / 2 - 0.15, y, 0],
            buff=0,
            color=ACCENT_COLOR,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.12,
        )
        return VGroup(lost_t, icon, arrow, kept_t)

    def construct(self):
        # ================= 1. the ladder =================================
        head = self.heading(
            "What we gave up, and what survived",
            "the talk as a ladder: each step trusts less of " "the data",
        )
        cols = VGroup(
            SmallText(
                "we stop trusting",
                font="Marcellus SC",
                font_size=28,
                color=LOST_COLOR,
            ).move_to([X_LOST, RUNG_Y0 + 0.65, 0]),
            SmallText(
                "what survives",
                font="Marcellus SC",
                font_size=28,
                color=KEPT_COLOR,
            ).move_to([X_KEPT, RUNG_Y0 + 0.65, 0]),
        )
        rungs = [
            ("everything", icon_globe(), "a choice of what to keep"),
            ("global distances", icon_roll(), "local neighbourhoods"),
            ("exact local distances", icon_concentration(), "which points are near"),
            ("their rank order", icon_rank(), "the neighbourhood as a set"),
            ("even the neighbours", icon_shortcut(), "a graph that is mostly right"),
        ]
        trust = Arrow(
            [X_LOST - 2.55, RUNG_Y0 + 0.2, 0],
            [X_LOST - 2.55, RUNG_Y0 - 4 * RUNG_STEP - 0.2, 0],
            buff=0,
            color=ACCENT_COLOR,
            stroke_width=2.5,
            max_tip_length_to_length_ratio=0.06,
        )
        trust_lab = (
            SmallText("trust", font_size=18, color=ACCENT_COLOR)
            .rotate(PI / 2)
            .next_to(trust, LEFT, buff=0.1)
        )
        self.play(FadeIn(head), FadeIn(cols), GrowArrow(trust), FadeIn(trust_lab))
        ladder = VGroup()
        for k, (lost, icon, kept) in enumerate(rungs):
            row = self.rung(k, lost, icon, kept)
            self.play(FadeIn(row[0], shift=0.1 * RIGHT), row[1].intro())
            self.play(GrowArrow(row[2]), FadeIn(row[3], shift=0.1 * RIGHT))
            ladder.add(row)
            self.marked_next_slide()

        bottom_text = SmallText(
            "a corrupted kNN graph — read as topology, not geometry —\n"
            "repaired by a layout that uses global information",
            font_size=TEXT_SIZE,
            color=DEFAULT_COLOR,
        )
        bottom_icon = icon_tail()
        bottom = VGroup(bottom_icon, bottom_text).arrange(RIGHT, buff=0.45)
        box = SurroundingRectangle(
            bottom, color=KEPT_COLOR, buff=0.2, corner_radius=0.1, stroke_width=2
        )
        VGroup(box, bottom).move_to([0, RUNG_Y0 - 5 * RUNG_STEP - 0.25, 0])
        self.play(Create(box), bottom_icon.intro(), FadeIn(bottom_text))
        self.marked_next_slide()

        # ================= 2. the thesis =================================
        head = self.swap(
            head,
            "Every layout encodes assumptions",
            "whether you chose them or not",
            FadeOut(cols, trust, trust_lab, ladder, box, bottom),
        )
        thesis = SmallText(
            "a layout's kernel is a set of assumptions\n"
            "about how much to trust the graph",
            font_size=40,
            color=DEFAULT_COLOR,
        ).move_to(1.2 * UP)
        formula = MathTex(
            r"q(d) \;=\; \Big(1 + ",
            r"\tfrac{1}{\alpha}",
            r"\,\Big(d \,/",
            r"\sigma_0",
            r"\Big)",
            r"^{m}",
            r"\Big)",
            r"^{-\alpha}",
            font_size=60,
        ).move_to(0.9 * DOWN)
        formula[1].set_color(ALPHA_COLOR)
        formula[3].set_color(SIGMA_COLOR)
        formula[5].set_color(M_COLOR)
        formula[7].set_color(ALPHA_COLOR)

        def tag(part, text, color, side):
            label = SmallText(text, font_size=24, color=color)
            label.next_to(part, side, buff=0.45)
            arrow = Arrow(
                label.get_edge_center(-side),
                part.get_edge_center(side),
                buff=0.08,
                color=color,
                stroke_width=2.5,
                max_tip_length_to_length_ratio=0.3,
            )
            return VGroup(label, arrow)

        def tag_at(part, text, color, label_pos):
            """A label at label_pos with a straight arrow to the part."""
            label = SmallText(text, font_size=24, color=color).move_to(label_pos)
            start = (
                label.get_bottom() if label_pos[1] > part.get_y() else label.get_top()
            )
            end = part.get_top() if label_pos[1] > part.get_y() else part.get_bottom()
            return VGroup(
                label,
                Arrow(
                    start,
                    end,
                    buff=0.08,
                    color=color,
                    stroke_width=2.5,
                    tip_length=0.16,
                    max_tip_length_to_length_ratio=0.5,
                ),
            )

        fb = formula.get_bottom()[1]
        tags = VGroup(
            tag_at(
                formula[5],
                "what dimension neighbourhoods fill",
                M_COLOR,
                [min(formula[5].get_x(), 3.6), formula.get_top()[1] + 0.55, 0],
            ),
            tag_at(
                formula[1],
                "how certain the local density is",
                ALPHA_COLOR,
                [formula[1].get_x() - 0.8, fb - 0.45, 0],
            ),
            tag_at(
                formula[7],
                "how hard a long edge may pull",
                TAIL_COLOR,
                [min(formula[7].get_x(), 4.9), fb - 1.25, 0],
            ),
        )
        self.play(FadeIn(thesis))
        self.marked_next_slide()
        self.play(Write(formula))
        for t in tags:
            self.play(FadeIn(t), run_time=0.8)
        self.marked_next_slide()

        # ================= 3. reading a layout ===========================
        head = self.swap(
            head,
            "Reading a layout",
            "what it works to repair, and what it has declared " "it will not preserve",
            FadeOut(thesis, formula, tags),
        )

        def column(title, items, color, x):
            """A title and items on fixed rows, so the two columns line up."""
            # align by the top: "repaired" has a descender, "declined" none
            t = SmallText(title, font_size=32, color=color).move_to(
                [x, 1.45, 0], aligned_edge=UP
            )
            rows = VGroup(
                *[
                    SmallText(a, font_size=22, color=DEFAULT_COLOR).move_to(
                        [x, 0.15 - 1.15 * k, 0]
                    )
                    for k, a in enumerate(items)
                ]
            )
            return VGroup(t, rows)

        repaired = column(
            "repaired",
            [
                "neighbourhoods\n(short distances survive)",
                "connectivity\n(shortcuts rejected, true edges kept)",
                "grouping",
            ],
            KEPT_COLOR,
            -3.4,
        )
        declined = column(
            "declined",
            [
                "long distances\n(the tail lets long edges go)",
                "gaps between clusters\n(the same)",
                "density and cluster size\n(averaged out by the prior)",
            ],
            LOST_COLOR,
            3.4,
        )
        divider = DashedLine(1.6 * UP, 2.4 * DOWN, color=ACCENT_COLOR, stroke_width=1.5)
        self.play(FadeIn(repaired[0]), FadeIn(declined[0]), Create(divider))
        for a, b in zip(repaired[1], declined[1]):
            self.play(
                FadeIn(a, shift=0.1 * UP), FadeIn(b, shift=0.1 * UP), run_time=0.8
            )
        self.marked_next_slide()

        # ================= 4. questions ==================================
        head = self.swap(
            head,
            "Questions to take away",
            "three places this community is well placed to " "push further",
            FadeOut(repaired, declined, divider),
        )
        questions = [
            (
                icon_noise(),
                "What is noise in your representation?",
                "our model was isotropic Gaussian noise —\n"
                "activations and embeddings almost certainly are not",
            ),
            (
                icon_dial(),
                "Can the knobs be set from the data?",
                "a layout's measured dimension tracks the m you ask for —\n"
                "but any estimate comes from the same corrupted graph",
            ),
            (
                icon_given_up(),
                "How do we show what a layout gave up?",
                "it cannot be judged point by point:\n"
                "good and bad hubs look the same locally",
            ),
        ]
        qs = VGroup()
        for k, (icon, q, why) in enumerate(questions):
            y = 1.55 - 1.75 * k
            icon.move_to([-5.3, y, 0])
            q_t = SmallText(q, font_size=30, color=DEFAULT_COLOR)
            why_t = SmallText(why, font_size=20, color=ACCENT_COLOR)
            text = VGroup(q_t, why_t).arrange(DOWN, buff=0.15)
            text.move_to([1.0, y, 0])
            qs.add(VGroup(icon, text))
            self.play(icon.intro(), FadeIn(q_t, shift=0.1 * RIGHT))
            self.play(FadeIn(why_t))
            self.marked_next_slide()

        # ---- contact details, in the bottom corners ----
        contacts = VGroup()
        for text, svg, fallback, corner in (
            (CONTACT_EMAIL, EMAIL_ICON_SVG, icon_envelope, DL),
            (CONTACT_SOCIAL, SOCIAL_ICON_SVG, icon_speech_bubble, DR),
        ):
            if not text:
                continue
            item = VGroup(
                contact_icon(svg, fallback),
                SmallText(text, font_size=20, color=ACCENT_COLOR),
            ).arrange(RIGHT, buff=0.18)
            contacts.add(item.to_corner(corner, buff=0.35))
        if len(contacts):
            self.play(FadeIn(contacts), run_time=1.5)
            self.marked_next_slide()
