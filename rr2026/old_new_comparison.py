from manim import *

import json
import sys
from pathlib import Path

from PIL import Image

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

ASSETS = Path(__file__).parent / "assets"

# Shown on the final slide; edit to match the release plan.
DEFAULTS_NOTE = "The new stack is opt-in for 0.6 and becomes the default after it."

# The longer-runs comparison.  Off by default: on MNIST at repulsion 1 the
# distance between digit centres grows about 8% from 200 to 1000 epochs for
# both optimizers, so it does not show a difference.  See assets/payoff.json
# ("separation") before switching it on.
SHOW_EPOCHS = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to3(p):
    return np.array([p[0], p[1], 0.0])


def caption(text, font_size=22, color=DEFAULT_COLOR, **kwargs):
    return crisp_text(text, font_size=font_size, color=color, **kwargs)


def caption_stack(lines, font_size=22, buff=0.1, color=DEFAULT_COLOR):
    group = VGroup(*[caption(line, font_size, color=color) for line in lines])
    return group.arrange(DOWN, aligned_edge=LEFT, buff=buff)


def gradient_bar(width, height, left_color, right_color, pieces=60):
    """A horizontal colour ramp built from thin slices."""
    slices = VGroup(
        *[
            Rectangle(
                width=width / pieces,
                height=height,
                stroke_width=0,
                fill_opacity=1,
                fill_color=interpolate_color(
                    ManimColor(left_color), ManimColor(right_color), k / (pieces - 1)
                ),
            )
            for k in range(pieces)
        ]
    ).arrange(RIGHT, buff=0)
    return slices


class GammaScale(VGroup):
    """A log2 scale of repulsion strengths with a movable knob."""

    def __init__(self, gammas, left, right, y, bar=None, tick_size=22):
        self.gammas = list(gammas)
        self.left, self.right, self.y = left, right, y
        span = np.log2(self.gammas[-1]) - np.log2(self.gammas[0])
        self._x = (
            lambda g: left
            + (right - left) * (np.log2(g) - np.log2(self.gammas[0])) / span
        )
        parts = []
        if bar is None:
            bar = Line(
                to3((left, y)), to3((right, y)), color=SECONDARY_COLOR, stroke_width=4
            )
        parts.append(bar)
        ticks = VGroup()
        for g in self.gammas:
            x = self._x(g)
            ticks.add(
                VGroup(
                    Line(
                        to3((x, y - 0.12)),
                        to3((x, y - 0.26)),
                        color=STRUCTURE_COLOR,
                        stroke_width=2,
                    ),
                    caption(str(g), tick_size).move_to(to3((x, y - 0.5))),
                )
            )
        parts.append(ticks)
        self.knob = Dot(
            to3((self._x(self.gammas[0]), y)),
            radius=0.14,
            color=SOURCE_COLOR,
            stroke_color=STRUCTURE_COLOR,
            stroke_width=2.5,
        )
        parts.append(self.knob)
        super().__init__(*parts)

    def point(self, g):
        return to3((self._x(g), self.y))


class SeriesChart(VGroup):
    """Styled axes over the six gamma stops, with series revealed stop by stop."""

    def __init__(
        self, gammas, y_range, title, series, y_label, x_length=2.2, y_length=1.9
    ):
        self.chart = styled_axes(
            x_range=[0, len(gammas) + 1, 1],
            y_range=y_range,
            x_label="repulsion",
            y_label=y_label,
            x_length=x_length,
            y_length=y_length,
            y_decimal_places=2,
            y_tick_font_size=14,
            x_tick_labels=[str(g) for g in gammas],
        )
        self.axes = self.chart[0]
        self.title = caption(title, 22, color=STRUCTURE_COLOR).next_to(
            self.chart, UP, buff=0.12
        )
        self.series = series  # list of (values, color, dashed)
        super().__init__(self.chart, self.title)

    def marks(self, k):
        """Dots at stop k and segments from stop k-1."""
        group = VGroup()
        for values, color, dashed in self.series:
            p = self.axes.c2p(k + 1, values[k])
            if k > 0:
                q = self.axes.c2p(k, values[k - 1])
                seg = (
                    DashedLine(q, p, color=color, stroke_width=2.5, dash_length=0.06)
                    if dashed
                    else Line(q, p, color=color, stroke_width=3)
                )
                group.add(seg)
            group.add(Dot(p, radius=0.045 if dashed else 0.06, color=color))
        return group


def nice_range(values, step, pad=0.0):
    lo = np.floor((min(values) - pad) / step) * step
    hi = np.ceil((max(values) + pad) / step) * step
    if hi - lo < 2 * step:
        hi = lo + 2 * step
    return [round(lo, 4), round(hi + step, 4), step]


class FrameSequence(ImageMobject):
    """An image whose pixels come from a numbered frame on disk."""

    def __init__(self, pattern, n_frames, height, floor=False):
        self.pattern, self.n_frames, self.floor = pattern, n_frames, floor
        super().__init__(np.array(Image.open(pattern.format(0)).convert("RGBA")))
        self.set_height(height)
        self.current = 0

    def show(self, k):
        # floor=True never shows a frame from later than k (for timed frames).
        k = int(np.clip(np.floor(k) if self.floor else round(k), 0, self.n_frames - 1))
        if k != self.current:
            self.pixel_array = np.array(
                Image.open(self.pattern.format(k)).convert("RGBA")
            )
            self.current = k
        return self


# ---------------------------------------------------------------------------
# The slide class
# ---------------------------------------------------------------------------
class OldVNew(UMAPSlide):
    def construct(self):
        self.data = np.load(ASSETS / "payoff.npz")
        self.meta = json.loads((ASSETS / "payoff.json").read_text())
        self.gammas = self.meta["gammas"]
        self.digit_colors = labels_to_rgb(self.data["labels"], digit_rgb())

        # self.end_section_wipe(
        #     SECTION_TITLES["payoff"],
        #     next_slide_prep=lambda: None,
        #     notes=NOTES["p01_dial"],
        # )
        # self.stage_dial()
        # self.stage_old()
        # self.stage_new()
        # if SHOW_EPOCHS:
        #     self.stage_epochs()
        self.stage_arxiv()
        # self.stage_try_it()

    # -- stage 1: the dial ---------------------------------------------------------------
    def stage_dial(self):
        self.set_title("Repulsion is a dial")
        bar = gradient_bar(9.0, 0.32, STRUCTURE_COLOR, REPEL_COLOR).move_to(
            to3((0, 0.9))
        )
        scale = GammaScale(self.gammas, -4.5, 4.5, 0.9, bar=bar, tick_size=24)
        left = caption_stack(
            ["cluster and", "global structure"], 24, color=STRUCTURE_COLOR
        )
        left.next_to(bar, UP, buff=0.3).align_to(bar, LEFT)
        right = caption_stack(["local", "fidelity"], 24, color=REPEL_COLOR)
        right.next_to(bar, UP, buff=0.3).align_to(bar, RIGHT)
        name = crisp_text("repulsion_strength", font=MONO_FONT, font_size=22).next_to(
            scale, DOWN, buff=0.25
        )
        note = caption_stack(
            [
                "Turning repulsion up trades cluster separation for keeping more of each",
                "point's true neighbours. The new stack makes the whole range usable.",
            ],
            24,
        ).move_to(to3((0, -2.3)))
        self.play(FadeIn(bar), FadeIn(scale[1]), FadeIn(name))
        self.play(FadeIn(left), FadeIn(right))
        self.play(FadeIn(scale.knob, scale=1.5))
        self.play(scale.knob.animate.move_to(scale.point(self.gammas[-1])), run_time=2)
        self.play(scale.knob.animate.move_to(scale.point(self.gammas[0])), run_time=1.2)
        self.play(FadeIn(note))
        self.marked_next_slide(notes=NOTES["p02_old"])

    # -- stage 2: the old optimizer at high repulsion ------------------------------------------
    def stage_old(self):
        self.clear_slide()
        self.set_title("The old optimizer, turned up")
        m = self.meta["metrics"]
        tiles = Group()
        for g in self.gammas:
            img = ImageMobject(str(ASSETS / "payoff" / f"old_g{g}.png")).set_height(
                1.95
            )
            frame = SurroundingRectangle(
                img, buff=0.02, color=SECONDARY_COLOR, stroke_width=1
            )
            label = MathTex(rf"\gamma = {g}", font_size=28).next_to(img, UP, buff=0.15)
            stats = (
                VGroup(
                    caption(f"trust {m[f'old_g{g}']['trustworthiness']:.3f}", 16),
                    caption(
                        f"neighbours {m[f'old_g{g}']['neighbour_preservation']:.3f}", 16
                    ),
                )
                .arrange(DOWN, aligned_edge=LEFT, buff=0.06)
                .next_to(img, DOWN, buff=0.15)
            )
            tiles.add(Group(img, frame, label, stats))
        tiles.arrange(RIGHT, buff=0.22).move_to(to3((0, 0.3)))
        sub = caption(
            "compatibility mode: the 0.5 optimizer and spectral initialization",
            20,
            color=SECONDARY_COLOR,
        ).to_edge(DOWN, buff=0.75)
        self.play(
            LaggedStart(*[FadeIn(t, shift=UP * 0.15) for t in tiles], lag_ratio=0.2),
            run_time=2.4,
        )
        self.play(FadeIn(sub))
        self.marked_next_slide(notes=NOTES["p03_new"])

    # -- stage 3: the new stack, one slide per stop -----------------------------------------
    def stage_new(self):
        self.clear_slide()
        self.set_title("The new stack: a usable dial")
        m, gs = self.meta["metrics"], self.gammas
        cloud = EmbeddingCloud(
            self.data["new_sweep"],
            self.digit_colors,
            width=6.0,
            height=5.7,
            center=(-3.4, -0.45),
            fit="union",
            point_px=3.0,
        )
        stop = ValueTracker(0.0)
        cloud.track(stop)
        scale = GammaScale(gs, 1.0, 6.3, 2.5, tick_size=20)
        scale_name = crisp_text(
            "repulsion_strength", font=MONO_FONT, font_size=16
        ).next_to(scale, UP, buff=0.12)

        def series(key, setup="new"):
            return [m[f"{setup}_g{g}"][key] for g in gs]

        trust = series("trustworthiness")
        keep, keep_no = series("neighbour_preservation"), series(
            "neighbour_preservation", "new_no_hn"
        )
        sil = series("silhouette")
        local = SeriesChart(
            gs,
            nice_range(keep + keep_no, 0.05),
            "neighbours kept",
            [(keep, STRUCTURE_COLOR, False), (keep_no, SECONDARY_COLOR, True)],
            "share",
        )
        glob = SeriesChart(
            gs,
            nice_range(sil, 0.1),
            "cluster separation",
            [(sil, REPEL_COLOR, False)],
            "silhouette",
        )
        # Side by side between the cloud and the frame edge, with a clear gap.
        glob.next_to(local, RIGHT, buff=0.35)
        VGroup(local, glob).move_to(to3((3.35, -0.3)))

        def key(color, text, dashed=False):
            sample = (
                DashedLine(
                    ORIGIN,
                    RIGHT * 0.35,
                    color=color,
                    stroke_width=2.5,
                    dash_length=0.06,
                )
                if dashed
                else Line(ORIGIN, RIGHT * 0.35, color=color, stroke_width=3)
            )
            return VGroup(sample, caption(text, 15)).arrange(RIGHT, buff=0.1)

        readouts = VGroup(
            VGroup(
                MetricReadout(
                    "neighbours kept", keep[0], font_size=20, color=STRUCTURE_COLOR
                ),
                MetricReadout(
                    "trustworthiness", trust[0], font_size=20, color=STRUCTURE_COLOR
                ),
                key(SECONDARY_COLOR, "without hard negatives", dashed=True),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1),
            VGroup(
                MetricReadout("silhouette", sil[0], font_size=20, color=REPEL_COLOR),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1),
        )
        readouts[0].next_to(local, DOWN, buff=0.3).align_to(local, LEFT)
        readouts[1].next_to(glob, DOWN, buff=0.3).align_to(glob, LEFT)
        values = [keep, trust, sil]
        numbers = [readouts[0][0].number, readouts[0][1].number, readouts[1][0].number]

        self.add(cloud)
        self.play(PMFadeIn(cloud), FadeIn(scale), FadeIn(scale_name), run_time=1.5)
        self.play(FadeIn(local), FadeIn(glob))
        marks = [local.marks(0), glob.marks(0)]
        self.play(*[FadeIn(mk) for mk in marks], FadeIn(readouts))
        for k in range(1, len(gs)):
            last = k == len(gs) - 1
            self.marked_next_slide(
                notes=NOTES["p04_last_stop" if last else "p04_stop"].format(g=gs[k])
            )
            new_marks = [local.marks(k), glob.marks(k)]
            self.play(
                stop.animate.set_value(k),
                scale.knob.animate.move_to(scale.point(gs[k])),
                *[Create(mk) for mk in new_marks],
                run_time=2,
            )
            for number, series_values in zip(numbers, values):
                number.set_value(series_values[k])
        cloud.clear_updaters()
        self.marked_next_slide(
            notes=NOTES["p05_epochs" if SHOW_EPOCHS else "p07_arxiv"]
        )

    # -- stage 4: longer runs ----------------------------------------------------------------
    def stage_epochs(self):
        self.clear_slide()
        sep, epochs = self.meta["separation"], self.meta["epochs"]
        self.set_title("More epochs, same picture")
        tracker = ValueTracker(0.0)
        clouds, readouts, heads = {}, {}, VGroup()
        for key, name, x in (
            ("old", "old optimizer", -3.45),
            ("new", "new optimizer", 3.45),
        ):
            cloud = EmbeddingCloud(
                self.data[f"{key}_epochs"],
                self.digit_colors,
                width=5.2,
                height=4.3,
                center=(x, -0.2),
                fit="each",
                point_px=2.6,
            )
            cloud.track(tracker)
            clouds[key] = cloud
            heads.add(caption(name, 28).move_to(to3((x, 2.45))))
            readouts[key] = MetricReadout(
                "cluster separation",
                sep[f"{key}_e{epochs[0]}"],
                num_decimal_places=2,
                font_size=22,
            )
            readouts[key].move_to(to3((x, -2.85)))
        counter = MetricReadout("epochs", epochs[0], num_decimal_places=0, font_size=26)
        counter.move_to(to3((0, 2.45)))
        note = caption(
            "separation: distance between digit centres over the digits' own spread",
            18,
            color=SECONDARY_COLOR,
        ).to_edge(DOWN, buff=0.35)
        self.add(clouds["old"], clouds["new"])
        self.play(
            PMFadeIn(clouds["old"]),
            PMFadeIn(clouds["new"]),
            FadeIn(heads),
            FadeIn(counter),
            FadeIn(readouts["old"]),
            FadeIn(readouts["new"]),
            FadeIn(note),
            run_time=1.5,
        )
        self.marked_next_slide(notes=NOTES["p06_epochs_run"])
        for k in range(1, len(epochs)):
            self.play(tracker.animate.set_value(k), run_time=1.8)
            counter.number.set_value(epochs[k])
            for key in ("old", "new"):
                readouts[key].number.set_value(sep[f"{key}_e{epochs[k]}"])
            self.wait(0.6)
        for cloud in clouds.values():
            cloud.clear_updaters()
        self.marked_next_slide(notes=NOTES["p07_arxiv"])

    # -- stage 5: ArXiv at scale, two rounds ---------------------------------------------
    def stage_arxiv(self):
        self.clear_slide()
        meta_file = ASSETS / "arxiv.json"
        if not meta_file.exists():
            self.set_title("At scale: ArXiv")
            self.play(
                FadeIn(caption("Run assets_arxiv.py to render this section.", 28))
            )
            self.marked_next_slide(notes=NOTES["p09_try"])
            return
        meta = json.loads(meta_file.read_text())
        rounds = meta["rounds"]
        for r, rnd in enumerate(rounds):
            if r > 0:
                self.clear_slide()
            after = (
                NOTES["p09_try"]
                if r == len(rounds) - 1
                else NOTES[f"arxiv_time_{r + 1}"]
            )
            self.arxiv_round(meta, r, rnd, after)

    def arxiv_round(self, meta, r, rnd, notes_after):
        demo = "demo" in meta["title"].lower()
        self.set_title(
            ("At scale (MNIST demo): " if demo else "At scale: ") + rnd["title"]
        )
        size, n_frames = 5.5, meta["time_frames"]
        grid = np.array(rnd["grid"])
        t_max = float(grid[-1])
        xs = {"left": -3.45, "right": 3.45}
        names = {side: rnd[side] for side in xs}
        configs = meta["configs"]

        panels, frames, heads, pending = Group(), {}, VGroup(), {}
        for side, x in xs.items():
            seq = FrameSequence(
                str(ASSETS / "arxiv" / f"round{r}_{side}_time_{{:03d}}.png"),
                n_frames,
                size,
                floor=True,
            )
            seq.move_to(to3((x, -0.45)))
            frame = SurroundingRectangle(
                seq, buff=0.0, color=SECONDARY_COLOR, stroke_width=1
            )
            frames[side] = seq
            panels.add(Group(seq, frame))
            cfg = configs[names[side]]
            heads.add(
                VGroup(
                    caption(cfg["label"], 24),
                    caption(cfg["detail"], 17, color=SECONDARY_COLOR),
                )
                .arrange(DOWN, buff=0.06)
                .next_to(seq, UP, buff=0.12)
            )
            pending[side] = caption(
                "initializing\u2026", 22, color=SECONDARY_COLOR
            ).move_to(seq)

        clock = MetricReadout("time (s)", 0.0, num_decimal_places=1, font_size=24)
        clock_group = VGroup(clock).move_to(to3((0, 2.62)))
        tracker = ValueTracker(0.0)
        for seq in frames.values():
            seq.add_updater(lambda m: m.show(tracker.get_value()))
        clock.number.add_updater(
            lambda m: m.set_value(tracker.get_value() / (n_frames - 1) * t_max)
        )

        self.play(
            FadeIn(panels),
            FadeIn(heads),
            FadeIn(clock_group),
            *[FadeIn(p) for p in pending.values()],
            run_time=1.2,
        )

        # Play the clock through each run's milestones; equal animation time
        # is equal optimization time for both panels.
        duration = 9.0
        events = []
        for side in xs:
            cfg = configs[names[side]]
            events.append((cfg["init_time"], "started", side))
            events.append((cfg["done"], "done", side))
        events.sort()
        done_labels = VGroup()
        for when, kind, side in events:
            k = float(np.clip(when / t_max * (n_frames - 1), 0, n_frames - 1))
            span = k - tracker.get_value()
            if span > 1e-6:
                self.play(
                    tracker.animate.set_value(k),
                    run_time=max(0.2, duration * span / (n_frames - 1)),
                    rate_func=linear,
                )
            if kind == "started":
                self.play(FadeOut(pending[side]), run_time=0.3)
            else:
                label = caption(f"done in {when:.1f} s", 22, color=STRUCTURE_COLOR)
                label.next_to(frames[side], DOWN, buff=0.15)
                done_labels.add(label)
                self.play(FadeIn(label), run_time=0.3)
        for seq in frames.values():
            seq.clear_updaters()
        clock.number.clear_updaters()
        self.marked_next_slide(notes=NOTES[f"arxiv_final_{r}"])

        # Full-resolution final layouts, with the zoom target boxed.
        zooms, boxes = Group(), VGroup()
        for side, x in xs.items():
            name = names[side]
            seq = FrameSequence(
                str(ASSETS / "arxiv" / f"{name}_zoom_{{:03d}}.png"),
                meta["zoom_frames"],
                size,
            )
            seq.move_to(frames[side])
            zooms.add(seq)
            fx0, fx1, fy0, fy1 = meta["zoom_box"][name]
            corner = seq.get_corner(DL)
            box = Rectangle(
                width=(fx1 - fx0) * size,
                height=(fy1 - fy0) * size,
                color=SOURCE_COLOR,
                stroke_width=3,
            )
            box.move_to(
                corner + np.array([(fx0 + fx1) / 2 * size, (fy0 + fy1) / 2 * size, 0])
            )
            boxes.add(box)
        self.play(FadeOut(clock_group), FadeIn(zooms), *[FadeOut(p[0]) for p in panels])
        self.play(Create(boxes))
        self.marked_next_slide(notes=NOTES[f"arxiv_zoom_{r}"])

        self.play(FadeOut(boxes), run_time=0.5)
        zoom = ValueTracker(0.0)
        for seq in zooms:
            seq.add_updater(lambda m: m.show(zoom.get_value()))
        self.play(
            zoom.animate.set_value(meta["zoom_frames"] - 1),
            run_time=7,
            rate_func=smooth,
        )
        for seq in zooms:
            seq.clear_updaters()
        self.marked_next_slide(notes=notes_after)

    # -- stage 6: try it --------------------------------------------------------------------
    def stage_try_it(self):
        self.clear_slide()
        self.set_title("Try it")
        lines = (
            "import umap",
            None,  # blank line
            "reducer = umap.UMAP(",
            "    compatibility_layout=False,",
            '    optimizer="adam",',
            '    init="recursive",',
            "    repulsion_strength=8.0,",
            "    negative_selection_range=20_000,",
            ")",
        )
        char_width = crisp_text("M", font=MONO_FONT, font_size=26).width
        code = VGroup()
        for line in lines:
            if line is None:
                code.add(
                    Rectangle(width=0.01, height=0.12, stroke_width=0, fill_opacity=0)
                )
                continue
            indent = len(line) - len(line.lstrip(" "))
            text = crisp_text(line.lstrip(" "), font=MONO_FONT, font_size=26)
            text.indent = indent
            code.add(text)
        code.arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        for mob in code:  # Text drops leading spaces, so indent by hand
            mob.shift(RIGHT * getattr(mob, "indent", 0) * char_width)
        code.move_to(to3((0, 0.35)))
        box = SurroundingRectangle(
            code, buff=0.35, color=SECONDARY_COLOR, stroke_width=1.5
        )
        note = caption(DEFAULTS_NOTE, 22, color=STRUCTURE_COLOR).next_to(
            box, DOWN, buff=0.4
        )
        self.play(Create(box), FadeIn(code, lag_ratio=0.1), run_time=2)
        self.play(FadeIn(note))
        self.marked_next_slide(notes=NOTES["p10_end"])


# ---------------------------------------------------------------------------
# Speaker notes
# ---------------------------------------------------------------------------
NOTES = {
    "p01_dial": (
        "What does all this buy you? Repulsion strength is a dial. Low repulsion "
        "favours cluster separation and global structure; high repulsion favours "
        "faithful local neighbourhoods. The new stack makes the whole range usable."
    ),
    "p02_old": (
        "Here is the old optimizer on MNIST as repulsion goes from one to thirty-two. "
        "The clusters shrink into islands and then tear: from about eight, the ones "
        "break into pieces. Trustworthiness still rises, because tearing does not "
        "create false neighbours, but the share of true neighbours kept rises much "
        "less than with the new stack."
    ),
    "p03_new": (
        "Now the new stack: Adam, soft clipping, recursive initialization, and hard "
        "negatives with a window of twenty thousand. Left, the embedding. Right, "
        "local scores and global scores; the dashed line is the same runs without "
        "hard negatives. Repulsion strength one."
    ),
    "p04_stop": "Repulsion strength {g}.",
    "p04_last_stop": (
        "Repulsion strength {g}. The share of true neighbours kept roughly doubles "
        "from one to thirty-two, and it is higher with hard negatives than without at "
        "every setting. Silhouette falls: the clusters give up some separation. That "
        "is the trade, and now the whole dial is usable."
    ),
    "p05_epochs": (
        "The same question for training length: both optimizers from 200 to 2000 "
        "epochs, with cluster separation measured underneath."
    ),
    "p06_epochs_run": "Watch the cluster separation readouts as training runs longer.",
    "p07_arxiv": (
        "Now at scale. First, the whole old stack against the whole new stack, both at "
        "repulsion one with no hard negatives. The clock covers initialization and "
        "optimization, with the same fixed seed for both: the old stack runs "
        "single-threaded, as UMAP 0.5 did with a seed; the new stack uses every core "
        "and stays reproducible. Equal steps of animation are equal seconds."
    ),
    "arxiv_final_0": (
        "The final layouts, at full resolution. The box marks the region we zoom into; "
        "it is chosen by its papers, so it is the same region in both."
    ),
    "arxiv_zoom_0": "Zooming in on that region in both.",
    "arxiv_time_1": (
        "Second round: the new stack at repulsion one, against the new stack with "
        "repulsion turned up to eight and hard negatives switched on. Same seed, same clock."
    ),
    "arxiv_final_1": "The final layouts, with the same region boxed.",
    "arxiv_zoom_1": "And zooming in: this is where the extra local detail shows.",
    "p09_try": (
        "To try it: switch off compatibility layout, choose Adam and recursive "
        "initialization, and set repulsion strength and the negative selection "
        "window to taste."
    ),
    "p10_end": "",
}
