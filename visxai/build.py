#!/usr/bin/env python3
"""
Build the visxai deck: (optionally) prepare data, render every scene with
manim-slides, convert to a PowerPoint deck, and (optionally) shrink it.

    python build.py                      # render everything, convert to pptx
    python build.py --prep               # first run prep steps with missing output
    python build.py --only KernelKnobs   # re-render one scene, then convert
    python build.py --quality l -j 8     # quick low-quality draft, 8 in parallel
    python build.py --shrink             # also write visxai-small.pptx
    python build.py --no-auto-next       # every slide waits for a click
    python build.py --dry-run            # just print the commands

The slide order lives in slide_sequence.txt; the prep steps and output names
are in the config block below. Both are meant to be edited by hand.

Slides made with next_slide(auto_next=True) -- e.g. a scene's closing fade,
so it runs straight into the next scene -- are given a timed advance in the
pptx (manim-slides' own pptx converter ignores auto_next).
"""

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ===========================================================================
# CONFIG -- edit freely
# ===========================================================================

# Prefix for every command (scjobsripts, manim-slides). Use [] to run whatever
# python / manim-slides is on PATH (e.g. an activated venv) instead of uv.
RUN = ["uv", "run", "--active"]

# Data preparation, in the order it should run: (name, command, outputs).
# Commands run in this directory as `RUN python <command...>`. A step counts
# as "missing" if any of its outputs does not exist.
#
# Notes:
#  * mnist_knn / quickdraw_knn both use knn_edge_prep.py. Re-running them (or
#    mnist_hubs) can change the edge / pair indices hand-picked in
#    SHOWCASE_EDGES (mnist_knn_slides.py, quickdraw_knn_slides.py) and PAIRS
#    (mnist_hubs_slides.py, quickdraw_hubs_slides.py): check the
#    *.candidates*.png sheets afterwards.
#  * quickdraw_knn / quickdraw_hubs download doodles and a DINOv2 model (cached in
#    quickdraw_cache/). The existing quickdraw_knn.npz has 25 pre-selected
#    showcase edges (default --n-show is 5); that only affects pre-selection.
#  * world_map: --cell 0.5 --colours 256 match the existing world_map.npz
#    (the defaults, 2.0 and 16, give a much coarser map).
#  * curve_frames: the existing embedding_frames.npz matches the defaults
#    (9 planes, D = 1024, 2000 points, k = 10, 150 frames); --umap is needed
#    for SpectralVsUMAPSlide.
PREP_STEPS = [
    ("mnist_knn", ["mnist_knn_prep.py"], ["mnist_knn.npz"]),
    ("quickdraw_knn", ["quickdraw_knn_prep.py"], ["quickdraw_knn.npz"]),
    ("mnist_hubs", ["mnist_hubs_prep.py"], ["mnist_hubs.npz"]),
    ("quickdraw_hubs", ["quickdraw_hubs_prep.py"], ["quickdraw_hubs.npz"]),
    ("kernel_layouts", ["kernel_layouts_prep.py"], ["kernel_layouts.npz"]),
    (
        "world_map",
        ["world_map_prep.py", "--cell", "0.5", "--colours", "256"],
        ["world_map.npz"],
    ),
    (
        "curve_frames",
        ["curve_embedding_creation.py", "--umap"],
        ["embedding_frames.npz"],
    ),
]

SEQUENCE_FILE = "slide_sequence.txt"
SLIDES_DIR = "slides"  # where manim-slides render writes (fixed by manim-slides)
OUTPUT_PPTX = "visxai.pptx"
SHRINK_SCRIPT = "../shrink_pptx.py"
AUTO_NEXT_SCRIPT = "../pptx_auto_next.py"
LOG_DIR = "build_logs"
JOB_MEDIA_DIR = "media/jobs"  # per-scene media dirs for parallel renders

# ===========================================================================

HERE = Path(__file__).resolve().parent


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, dry_run, **kw):
    """Run a command (list), echoing it. Returns the exit code."""
    print("  $ " + shlex.join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, **kw).returncode


# ---------------------------------------------------------------------------
# Slide sequence
# ---------------------------------------------------------------------------
def read_sequence(path):
    """[(module, scene)] from the sequence file; exits on any problem."""
    entries, errors, seen = [], [], {}
    for n, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            errors.append(f"line {n}: expected 'module.py SceneName', got {raw!r}")
            continue
        module, scene = parts
        if not Path(module).is_file():
            errors.append(f"line {n}: no such file {module}")
        elif not re.search(
            rf"^class\s+{re.escape(scene)}\s*\(", Path(module).read_text(), re.M
        ):
            errors.append(f"line {n}: {module} does not define class {scene}")
        if scene in seen:
            errors.append(f"line {n}: {scene} already listed on line {seen[scene]}")
        seen[scene] = n
        entries.append((module, scene))
    if errors:
        sys.exit(f"{path}:\n  " + "\n  ".join(errors))
    if not entries:
        sys.exit(f"{path}: no scenes listed")
    return entries


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def prep(args):
    names = [s[0] for s in PREP_STEPS]
    for name in args.prep_only or []:
        if name not in names:
            sys.exit(f"unknown prep step {name!r}; choose from {', '.join(names)}")
    for name, cmd, outputs in PREP_STEPS:
        missing = [o for o in outputs if not Path(o).exists()]
        if args.prep_only:
            todo = name in args.prep_only
        else:
            todo = args.prep_force or bool(missing)
        if not todo:
            log(f"prep {name}: outputs present, skipping")
            continue
        log(f"prep {name}")
        if run(RUN + ["python", *cmd], args.dry_run) != 0:
            sys.exit(f"prep step {name} failed")
        gone = [o for o in outputs if not Path(o).exists()]
        if gone and not args.dry_run:
            sys.exit(f"prep step {name} did not produce {', '.join(gone)}")


def render_cmd(args, module, scenes, media_dir=None):
    # "--quality h" rather than "-qh": manim-slides render has its own -h
    # (help), and click would read "-qh" as "-q -h".
    cmd = RUN + ["manim-slides", "render", "--quality", args.quality]
    if not args.cache:
        cmd.append("--disable_caching")
    if media_dir:
        cmd += ["--media_dir", media_dir]
    return cmd + [module, *scenes]


def json_mtime(scene):
    p = Path(SLIDES_DIR) / f"{scene}.json"
    return p.stat().st_mtime_ns if p.exists() else None


def render(args, entries):
    if args.only:
        known = {s for _, s in entries}
        bad = [s for s in args.only if s not in known]
        if bad:
            sys.exit(f"--only: not in {SEQUENCE_FILE}: {', '.join(bad)}")
        entries = [(m, s) for m, s in entries if s in args.only]
    if args.skip_existing:
        done = [s for _, s in entries if (Path(SLIDES_DIR) / f"{s}.json").exists()]
        if done:
            log(f"skipping already rendered: {', '.join(done)}")
        entries = [(m, s) for m, s in entries if s not in done]
    if not entries:
        log("nothing to render")
        return

    # A scene succeeded if its slides/<Scene>.json was (re)written this run
    # and its process did not report an error.
    before = {s: json_mtime(s) for _, s in entries}
    errored = set()
    if args.jobs > 1:
        # One process per scene, each with its own media dir: scenes sharing
        # a media dir race on the TeX cache and fail at random.
        Path(LOG_DIR).mkdir(exist_ok=True)
        log(
            f"rendering {len(entries)} scenes, {args.jobs} at a time "
            f"(logs in {LOG_DIR}/)"
        )

        def job(entry):
            module, scene = entry
            cmd = render_cmd(args, module, [scene], f"{JOB_MEDIA_DIR}/{scene}")
            if args.dry_run:
                print("  $ " + shlex.join(cmd), flush=True)
                return
            with open(Path(LOG_DIR) / f"{scene}.log", "w") as f:
                code = subprocess.run(
                    cmd, stdout=f, stderr=subprocess.STDOUT
                ).returncode
            if code != 0:
                errored.add(scene)
            log(f"  {scene}: {'ok' if code == 0 else f'FAILED (exit {code})'}")

        with ThreadPoolExecutor(args.jobs) as pool:
            list(pool.map(job, entries))
    else:
        groups = {}  # module -> scenes, in order of first appearance
        for module, scene in entries:
            groups.setdefault(module, []).append(scene)
        for module, scenes in groups.items():
            log(f"rendering {module}: {' '.join(scenes)}")
            run(render_cmd(args, module, scenes), args.dry_run)

    if args.dry_run:
        return
    failed = [
        s for _, s in entries if s in errored or json_mtime(s) in (None, before[s])
    ]
    log(f"rendered {len(entries) - len(failed)}/{len(entries)} scenes")
    if failed:
        hint = f" (see {LOG_DIR}/<Scene>.log)" if args.jobs > 1 else ""
        sys.exit(f"failed: {', '.join(failed)}{hint}")


def convert(args, entries):
    scenes = [s for _, s in entries]
    missing = [s for s in scenes if not (Path(SLIDES_DIR) / f"{s}.json").exists()]
    if missing and not args.dry_run:
        sys.exit(f"not rendered yet: {', '.join(missing)}")
    log(f"converting {len(scenes)} scenes to {args.output}")
    cmd = RUN + [
        "manim-slides",
        "convert",
        "--to",
        "pptx",
        "--folder",
        SLIDES_DIR,
        *scenes,
        args.output,
    ]
    if run(cmd, args.dry_run) != 0:
        sys.exit("conversion failed")


def auto_next(args, entries):
    log("applying auto_next slide advances")
    slides = [str(Path(SLIDES_DIR) / f"{s}.json") for _, s in entries]
    cmd = RUN + ["python", AUTO_NEXT_SCRIPT, args.output, *slides]
    if run(cmd, args.dry_run) != 0:
        sys.exit("auto_next post-processing failed")


def shrink(args):
    if not shutil.which("ffmpeg"):
        log("ffmpeg not found: skipping shrink")
        return
    extra = shlex.split(args.shrink_args)
    if not shutil.which("pngquant") and "--skip-png" not in extra:
        log("pngquant not found: shrinking videos only (--skip-png)")
        extra.append("--skip-png")
    log(f"shrinking {args.output}")
    if run(RUN + ["python", SHRINK_SCRIPT, args.output, *extra], args.dry_run) != 0:
        sys.exit("shrink failed")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_argument_group("prep (off by default)")
    g.add_argument(
        "--prep", action="store_true", help="run prep steps whose outputs are missing"
    )
    g.add_argument("--prep-force", action="store_true", help="run every prep step")
    g.add_argument(
        "--prep-only",
        action="append",
        metavar="NAME",
        help="run this prep step (repeatable): " + ", ".join(s[0] for s in PREP_STEPS),
    )
    g = p.add_argument_group("render")
    g.add_argument("--no-render", action="store_true", help="skip rendering")
    g.add_argument(
        "--only",
        nargs="+",
        metavar="SCENE",
        help="render only these scenes (the whole sequence is " "still converted)",
    )
    g.add_argument(
        "--skip-existing",
        action="store_true",
        help=f"skip scenes that already have {SLIDES_DIR}/<Scene>.json",
    )
    g.add_argument(
        "--quality",
        choices="lmhpk",
        default="h",
        help="manim quality: l=480p15 m=720p30 h=1080p60 "
        "p=1440p60 k=2160p60 (default: h)",
    )
    g.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="render this many scenes in parallel (default: 1)",
    )
    g.add_argument(
        "--cache",
        action="store_true",
        help="allow manim's partial-movie cache (default: disabled)",
    )
    g = p.add_argument_group("output")
    g.add_argument("--no-convert", action="store_true", help="stop after rendering")
    g.add_argument(
        "-o",
        "--output",
        default=OUTPUT_PPTX,
        help=f"pptx to write (default: {OUTPUT_PPTX})",
    )
    g.add_argument(
        "--no-auto-next",
        action="store_true",
        help="leave auto_next slides waiting for a click in the pptx",
    )
    g.add_argument(
        "--shrink",
        action="store_true",
        help=f"run {SHRINK_SCRIPT} on the pptx (writes <name>-small.pptx)",
    )
    g.add_argument(
        "--shrink-args",
        default="",
        metavar="ARGS",
        help="extra arguments for the shrink script, quoted, "
        "e.g. --shrink-args='--heavy-crf 28'",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="print the commands without running them"
    )
    args = p.parse_args()

    os.chdir(HERE)
    entries = read_sequence(SEQUENCE_FILE)
    log(f"{len(entries)} scenes in {SEQUENCE_FILE}")

    if args.prep or args.prep_force or args.prep_only:
        prep(args)
    if not args.no_render:
        render(args, entries)
    if not args.no_convert:
        convert(args, entries)
        if not args.no_auto_next:
            auto_next(args, entries)
        if args.shrink:
            shrink(args)
    log("done")


if __name__ == "__main__":
    main()
