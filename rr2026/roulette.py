"""Opening: "Powerpoint roulette". A prize wheel of possible presenters spins
and lands on the person actually giving the talk.

    manim-slides render roulette.py PowerpointRoulette

Put this class first in the presentation, before the title slide.
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
    add_logo_to_background,
    create_styled_axes,
    TIMCSlide,
    ThreeDTIMCSlide,
    PhaseSlide,
    colormap_color,
    create_logo,
)

apply_defaults()

from umap_talk_common import *

apply_umap_defaults()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
NAMES = ["Dave", "Jasmine", "John", "Francois", "Benoit", "Gael"]
WINNER = "John"
SEGMENT_COLORS = [COLOR_CYCLE[i] for i in range(len(NAMES))]

RADIUS = 2.45
CENTRE = np.array([0.0, -0.8, 0.0])
PAUSE_SECONDS = 1.5      # the wheel sits still before it spins
WIND_UP = 0.15           # radians backwards before the spin
MIN_TURNS = 4.0          # at least this many turns before it stops
LANDING_TARGET = 0.4     # where to stop in the winner's segment: 0 = centre,
                         # 0.5 = right at the edge with the next name
PREFER_ROCK_BACK = True  # prefer endings where the next peg pushes the flapper,
                         # fails to get past, and the wheel rocks back

# Physics (scene units; the wheel's inertia is 1). The spin is simulated, not
# scripted: a flywheel slowed by friction, and a spring-loaded flapper that
# the pegs cannot pass through, so each peg pushes it aside until the tip
# slips over, takes energy from the wheel, and the flapper snaps back. The
# flapper's travel is limited by stops, as on a real wheel.
PEG_RADIUS = RADIUS + 0.07           # pegs sit on the rim
PIVOT_HEIGHT = RADIUS + 0.85         # flapper hinge, above the top of the wheel
FLAP_LENGTH = 0.95                   # hinge to tip: the tip reaches just inside the pegs
FLAP_HALF_WIDTH = 0.22               # at the hinge; it tapers to the tip
CONTACT = 0.11                       # peg radius plus the flapper's half-thickness where pegs meet it
FLAP_LIMIT = 1.0                     # stops either side (radians); pegs need about 0.8 to pass
FLAP_INERTIA, FLAP_SPRING = 0.0012, 0.7
FLAP_DAMPING = 2 * 0.25 * np.sqrt(FLAP_SPRING * FLAP_INERTIA)
FRICTION, DRAG = 0.25, 0.25          # Coulomb and speed-proportional friction on the wheel
DT = 2e-4


def simulate(launch, theta0, n, t_max=20.0, keep_every=None):
    """Wheel angle theta (anticlockwise +; the spin is clockwise) and flapper
    angle phi (+ = tip pushed right), for one or many launch speeds."""
    seg = TAU / n
    launch = np.atleast_1d(np.asarray(launch, float))
    m = len(launch)
    peg_base = PI / 2 - seg / 2 + seg * np.arange(n)
    theta, omega = np.full(m, theta0), -launch.copy()
    phi, phid = np.zeros(m), np.zeros(m)
    trace = []
    last_push, rocked, stop_time = np.zeros(m), np.zeros(m, bool), np.zeros(m)
    for k in range(int(t_max / DT)):
        new = omega - DT * (FRICTION * np.sign(omega) + DRAG * omega)
        omega = np.where(np.sign(new) != np.sign(omega), 0.0, new)   # friction stops, never reverses
        theta = theta + DT * omega
        phid = phid + DT * (-FLAP_SPRING * phi - FLAP_DAMPING * phid) / FLAP_INERTIA
        phi = phi + DT * phid
        # the stops: the flapper bounces off them, losing most of its speed
        stopped = np.abs(phi) > FLAP_LIMIT
        phid = np.where(stopped & (np.sign(phid) == np.sign(phi)), -0.3 * phid, phid)
        phi = np.clip(phi, -FLAP_LIMIT, FLAP_LIMIT)
        # the peg nearest the top, measured clockwise from the top
        beta = (PI / 2 - (peg_base[None, :] + theta[:, None]) + PI) % TAU - PI
        b = beta[np.arange(m), np.abs(beta).argmin(1)]
        px, py = PEG_RADIUS * np.sin(b), PEG_RADIUS * np.cos(b)
        d = np.hypot(px, PIVOT_HEIGHT - py)
        toward = np.arctan2(px, PIVOT_HEIGHT - py)               # hinge-to-peg direction
        side = np.where(toward < phi, 1.0, -1.0)                 # a peg on the left pushes the flapper right
        limit = toward + side * np.arcsin(np.clip(CONTACT / d, 0, 1))
        hit = (d < FLAP_LENGTH + CONTACT) & (side * (phi - limit) < 0)
        if hit.any():
            gear = -(PEG_RADIUS * PIVOT_HEIGHT * np.cos(b) - PEG_RADIUS ** 2) / d ** 2   # d(limit)/d(theta)
            closing = side * (phid - gear * omega)
            impulse = np.where(hit & (closing < 0), -closing / (gear ** 2 + 1 / FLAP_INERTIA), 0.0)
            omega = omega - side * gear * impulse
            phid = phid + side * impulse / FLAP_INERTIA
            phi = np.where(hit, limit, phi)
        if k % 25 == 0:
            last_push = np.where(phi > 0.17, k * DT, last_push)
            rocked |= omega > 1e-3
            stop_time = np.where(np.abs(omega) > 1e-5, k * DT, stop_time)
        if keep_every and k % keep_every == 0:
            trace.append((k * DT, theta[0], phi[0], omega[0]))
    ending = dict(last_push=last_push, rocked=rocked, stop_time=stop_time)
    return theta, phi, omega, (np.array(trace) if keep_every else ending)


def choose_launch(n, winner, theta0):
    """A launch speed that stops cleanly on the winner, after MIN_TURNS turns,
    as near LANDING_TARGET as the physics allows; with PREFER_ROCK_BACK, one
    whose last moments include a failed push past the next peg, if any does."""
    seg = TAU / n
    for lo, hi in ((9.0, 16.0), (16.0, 24.0)):
        speeds = np.linspace(lo, hi, 281)
        theta, phi, omega, ending = simulate(speeds, theta0, n)
        centre = PI / 2 + winner * seg
        offset = ((PI / 2 - (centre + theta) + PI) % TAU - PI) / seg      # + = past the centre
        turns = (theta0 - theta) / TAU
        ok = (np.abs(offset) < 0.45) & (turns >= MIN_TURNS) & (np.abs(omega) < 1e-6) & (np.abs(phi) < 0.03)
        drama = ok & ending["rocked"] & (ending["stop_time"] - ending["last_push"] < 1.5)
        if PREFER_ROCK_BACK and drama.any():
            ok = drama
        if ok.any():
            best = np.flatnonzero(ok)[np.abs(offset[ok] - LANDING_TARGET).argmin()]
            return speeds[best]
    raise RuntimeError("no launch speed lands on the winner; adjust MIN_TURNS or the physics settings")


def readable_on(colour):
    """Navy text on light segments, white on dark ones."""
    r, g, b = ManimColor(colour).to_rgb()
    return STRUCTURE_COLOR if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else WHITE


class PowerpointRoulette(UMAPSlide):
    def construct(self):
        n = len(NAMES)
        seg = TAU / n
        winner = NAMES.index(WINNER)

        # -- the wheel: segment i is centred at angle PI/2 + i * seg (segment 0 under the pointer)
        segments = VGroup()
        for i, (name, colour) in enumerate(zip(NAMES, SEGMENT_COLORS)):
            mid = PI / 2 + i * seg
            wedge = AnnularSector(inner_radius=0, outer_radius=RADIUS, angle=seg, start_angle=mid - seg / 2,
                                  fill_color=colour, fill_opacity=1, stroke_color=WHITE, stroke_width=5)
            wedge.shift(CENTRE)
            label = crisp_text(name, 34, color=readable_on(colour))
            room = 2 * 0.64 * RADIUS * np.sin(seg / 2) * 0.8  # chord at the label's radius, with margin
            if label.width > room:
                label.scale_to_fit_width(room)
            label.rotate(mid - PI / 2)  # baseline tangent to the rim: upright at the top
            label.move_to(CENTRE + 0.64 * RADIUS * np.array([np.cos(mid), np.sin(mid), 0]))
            segments.add(VGroup(wedge, label))
        rim = Circle(radius=RADIUS + 0.07, color=STRUCTURE_COLOR, stroke_width=12).move_to(CENTRE)
        pegs = VGroup(*[Dot(CENTRE + (RADIUS + 0.07) * np.array([np.cos(a), np.sin(a), 0]), radius=0.07,
                            color=WHITE, stroke_color=STRUCTURE_COLOR, stroke_width=2)
                        for a in PI / 2 - seg / 2 + seg * np.arange(n)])
        hub = VGroup(Circle(radius=0.42, fill_color=WHITE, fill_opacity=1, stroke_color=STRUCTURE_COLOR,
                            stroke_width=6),
                     Dot(radius=0.1, color=STRUCTURE_COLOR)).move_to(CENTRE)
        wheel = VGroup(segments, rim, pegs, hub)

        # -- the pointer: a flag at the top, pivoting on its upper edge
        pivot = CENTRE + np.array([0, PIVOT_HEIGHT, 0])
        tip = pivot + DOWN * FLAP_LENGTH
        flag = Polygon(tip, pivot + LEFT * FLAP_HALF_WIDTH, pivot + RIGHT * FLAP_HALF_WIDTH,
                       fill_color=HIGHLIGHT_COLOR, fill_opacity=1, stroke_color=STRUCTURE_COLOR, stroke_width=4)
        pin = Dot(pivot, radius=0.07, color=STRUCTURE_COLOR)
        pointer = VGroup(flag, pin)

        # -- simulate the spin, then play it back in real time
        launch = choose_launch(n, winner, WIND_UP)
        keep = int(round(1 / 240 / DT))
        _, _, _, trace = simulate([launch], WIND_UP, n, keep_every=keep)
        t_sim, theta_sim, phi_sim, omega_sim = trace.T
        moving = np.flatnonzero((np.abs(omega_sim) > 1e-5) | (np.abs(phi_sim) > 0.004))
        duration = t_sim[moving[-1]] + 0.3 if len(moving) else 1.0   # until the flapper has settled too

        angle = ValueTracker(0.0)   # the wheel's rotation; the wind-up is scripted, the spin simulated
        clock = ValueTracker(0.0)
        state = {"angle": 0.0}

        def turn(m):
            a = angle.get_value()
            m.rotate(a - state["angle"], about_point=CENTRE)
            state["angle"] = a

        rest = pointer.copy()

        def flap(m):
            m.become(rest.copy().rotate(float(np.interp(clock.get_value(), t_sim, phi_sim)), about_point=pivot))

        # -- slide 1: appear, pause, wind up, spin, settle on the winner
        self.set_title("Powerpoint roulette")
        self.play(GrowFromCenter(wheel), FadeIn(pointer, shift=DOWN * 0.3), run_time=1.2)
        self.wait(PAUSE_SECONDS)
        wheel.add_updater(turn)
        self.play(angle.animate.set_value(WIND_UP), run_time=0.6, rate_func=smooth)
        wheel.remove_updater(turn)

        def spin(m):  # the wheel and the flapper both read the same simulation clock
            a = float(np.interp(clock.get_value(), t_sim, theta_sim))
            m.rotate(a - state["angle"], about_point=CENTRE)
            state["angle"] = a

        wheel.add_updater(spin)
        pointer.add_updater(flap)
        self.play(clock.animate.set_value(duration), run_time=duration, rate_func=linear)
        wheel.remove_updater(spin)
        pointer.remove_updater(flap)
        others = [s for i, s in enumerate(segments) if i != winner]
        self.play(*[s[0].animate.set_fill(opacity=0.35) for s in others],
                  *[s[1].animate.set_opacity(0.45) for s in others], run_time=0.8)
        self.play(segments[winner].animate(rate_func=there_and_back).scale(1.08, about_point=CENTRE),
                  Flash(tip + UP * 0.05, color=HIGHLIGHT_COLOR, line_length=0.35, num_lines=12, flash_radius=0.45),
                  run_time=0.9)
        self.marked_next_slide(notes=NOTES["r2_card"])

        # -- slide 2: the card
        board = VGroup(wheel, pointer)
        self.play(board.animate.scale(0.72).move_to(to3((-3.6, -0.55))), run_time=1.0)
        col = VGroup(
            crisp_text("Today's presenter", 26, color=SECONDARY_COLOR),
            crisp_text(WINNER, 72, color=STRUCTURE_COLOR),
            crisp_lines(["The scheduled speaker is ill, so " + WINNER, "is presenting at very short notice."],
                        24, buff=0.1),
            crisp_lines(["Please adjust your expectations accordingly.",
                         "Hard questions will be forwarded to the author."], 22, buff=0.1, color=SECONDARY_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        col.move_to(to3((2.6, -0.4)))
        self.play(FadeIn(col[0], shift=UP * 0.1))
        self.play(Write(col[1]), run_time=1.0)
        self.play(FadeIn(col[2]))
        self.play(FadeIn(col[3]))
        self.marked_next_slide(notes=NOTES["r3_title"])
        self.clear_slide()  # fade out, so the title slide starts clean


def to3(p):
    return np.array([p[0], p[1], 0.0])


NOTES = {
    "r1_spin": "Let the wheel spin; it plays on its own.",
    "r2_card": (
        "Well, it landed on me. I'm John, standing in for the author, who is ill. I've had "
        "these slides for a very short time, so bear with me; anything hard, I'll pass on."
    ),
    "r3_title": "On to the talk.",
}
