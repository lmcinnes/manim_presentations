#!/usr/bin/env python3
"""
shrink_pptx.py - reduce the size of PowerPoint decks exported by manim-slides.

What it does, working on a copy of the deck (the original is never modified):

  * Re-encodes every video in ppt/media with H.264 tuned for animation.
    Videos larger than --fps-threshold MB are additionally reduced to
    --fps frames per second and encoded with the (stronger) --heavy-crf.
  * Quantises every PNG in ppt/media with pngquant (poster frames etc.).
  * Keeps a re-encoded file only if it is actually smaller.
  * Keeps all filenames, so no pptx internals need to be edited.

Requirements: ffmpeg + ffprobe (with libx264), and pngquant for PNGs.

Example:
  shrink_pptx.py talk.pptx                    # writes talk-small.pptx
  shrink_pptx.py talk.pptx --fps-threshold 1 --heavy-crf 28 -o share.pptx
"""

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

MEDIA_PREFIX = "ppt/media/"
VIDEO_EXTS = {".mp4", ".m4v", ".mov"}
PNG_EXTS = {".png"}
ALREADY_COMPRESSED = VIDEO_EXTS | PNG_EXTS | {".jpg", ".jpeg", ".gif", ".webm"}
MB = 1024 * 1024


def human(n: float) -> str:
    return f"{n / MB:.1f} MB" if n >= MB / 10 else f"{n / 1024:.0f} KB"


def probe_fps(path: Path) -> float:
    """Return the video's frame rate, or 0.0 if it cannot be determined."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        )
        num, den = json.loads(out.stdout)["streams"][0]["r_frame_rate"].split("/")
        return float(num) / float(den) if float(den) else 0.0
    except Exception:
        return 0.0


def shrink_video(src: Path, args):
    old = src.stat().st_size
    heavy = old > args.fps_threshold * MB
    crf = args.heavy_crf if heavy else args.crf
    filters, notes = [], [f"crf {crf}"]

    if heavy:
        fps = probe_fps(src)
        if fps == 0.0 or fps > args.fps + 0.01:
            filters.append(f"fps={args.fps:g}")
            notes.append(f"{args.fps:g} fps")
        if args.heavy_max_height:
            # Only ever scale down; the comma inside min() must be escaped for ffmpeg.
            filters.append(f"scale=-2:min({args.heavy_max_height}\\,ih)")
            notes.append(f"<= {args.heavy_max_height}p")

    dst = src.with_name(src.stem + ".shrunk" + src.suffix)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src)]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += [
        "-c:v", "libx264", "-preset", args.preset, "-tune", "animation",
        "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-c:a", "copy", str(dst),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    new = dst.stat().st_size
    if new >= old:
        dst.unlink()
        return None, old, old, "kept original (re-encode not smaller)"
    return dst, old, new, ", ".join(notes)


def shrink_png(src: Path, args):
    old = src.stat().st_size
    dst = src.with_name(src.stem + ".shrunk" + src.suffix)
    res = subprocess.run(
        ["pngquant", "--force", "--skip-if-larger", "--quality", args.png_quality,
         "--output", str(dst), str(src)],
        capture_output=True, text=True,
    )
    # 98 = result would be larger, 99 = quality target not reachable: keep original.
    if res.returncode in (98, 99):
        return None, old, old, "kept original"
    if res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, res.args, res.stdout, res.stderr)
    new = dst.stat().st_size
    if new >= old:
        dst.unlink()
        return None, old, old, "kept original"
    return dst, old, new, f"pngquant {args.png_quality}"


def parse_args():
    p = argparse.ArgumentParser(
        description="Shrink a manim-slides PowerPoint export by re-encoding its media.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=Path, help="the .pptx to shrink")
    p.add_argument("-o", "--output", type=Path,
                   help="output file (default: <input>-small.pptx)")
    p.add_argument("--crf", type=int, default=26,
                   help="x264 CRF for ordinary videos (lower = better quality)")
    p.add_argument("--heavy-crf", type=int, default=30,
                   help="x264 CRF for videos above --fps-threshold")
    p.add_argument("--fps-threshold", type=float, default=2.0,
                   help="videos larger than this many MB get frame-rate reduction "
                        "and --heavy-crf (0 = apply to all, a huge value = none)")
    p.add_argument("--fps", type=float, default=30,
                   help="target frame rate for videos above the threshold")
    p.add_argument("--heavy-max-height", type=int, default=None,
                   help="optionally scale videos above the threshold down to this height, e.g. 720")
    p.add_argument("--preset", default="veryslow",
                   help="x264 preset; 'medium' is much faster for quick trials")
    p.add_argument("--png-quality", default="70-95",
                   help="pngquant quality range (min-max)")
    p.add_argument("--skip-video", action="store_true", help="leave videos untouched")
    p.add_argument("--skip-png", action="store_true", help="leave PNGs untouched")
    p.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="number of files to process in parallel")
    return p.parse_args()


def main():
    args = parse_args()
    src = args.input
    if not src.is_file():
        sys.exit(f"error: {src} not found")
    out = args.output or src.with_name(src.stem + "-small" + src.suffix)
    if out.resolve() == src.resolve():
        sys.exit("error: output must differ from input")

    needed = []
    if not args.skip_video:
        needed += ["ffmpeg", "ffprobe"]
    if not args.skip_png:
        needed += ["pngquant"]
    missing = [t for t in needed if shutil.which(t) is None]
    if missing:
        sys.exit(f"error: missing required tool(s): {', '.join(missing)}")

    with zipfile.ZipFile(src) as zin, tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        infos = zin.infolist()

        # Extract only the media we intend to process.
        work = []
        for info in infos:
            name = info.filename
            if not name.startswith(MEDIA_PREFIX):
                continue
            ext = Path(name).suffix.lower()
            if ext in VIDEO_EXTS and not args.skip_video:
                fn = shrink_video
            elif ext in PNG_EXTS and not args.skip_png:
                fn = shrink_png
            else:
                continue
            local = tmp / Path(name).name
            local.write_bytes(zin.read(name))
            work.append((name, fn, local))

        print(f"Processing {len(work)} media files from {src} with {args.jobs} job(s)...")
        replacements = {}
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(fn, local, args): name for name, fn, local in work}
            for fut in cf.as_completed(futures):
                name = futures[fut]
                short = Path(name).name
                try:
                    new_path, old, new, note = fut.result()
                except subprocess.CalledProcessError as e:
                    msg = (e.stderr or "").strip().splitlines()
                    print(f"  WARNING {short}: failed, keeping original "
                          f"({msg[-1] if msg else e})")
                    continue
                if new_path:
                    replacements[name] = new_path
                print(f"  {short:<20} {human(old):>9} -> {human(new):>9}  ({note})")

        # Rebuild the zip in the original entry order ([Content_Types].xml stays first).
        partial = out.with_name(out.name + ".partial")
        with zipfile.ZipFile(partial, "w") as zout:
            for info in infos:
                name = info.filename
                data = (replacements[name].read_bytes() if name in replacements
                        else zin.read(name))
                zi = zipfile.ZipInfo(name, date_time=info.date_time)
                zi.external_attr = info.external_attr
                zi.compress_type = (zipfile.ZIP_STORED
                                    if Path(name).suffix.lower() in ALREADY_COMPRESSED
                                    else zipfile.ZIP_DEFLATED)
                zout.writestr(zi, data)
        partial.replace(out)

    before, after = src.stat().st_size, out.stat().st_size
    print(f"\n{src.name}: {human(before)} -> {out.name}: {human(after)} "
          f"({100 * (1 - after / before):.0f}% smaller)")


if __name__ == "__main__":
    main()
