# visxai

## Building the deck

`build.py` renders every scene listed in `slide_sequence.txt` with
manim-slides, converts them to `visxai.pptx` in that order, and can
optionally run the data prep scripts first and shrink the deck afterwards.
Commands run through `uv run`, from this directory.

```bash
python build.py                       # render everything (1080p60), write visxai.pptx
python build.py --quality l -j 8      # quick 480p draft, 8 scenes at a time
python build.py --only KernelKnobs    # re-render one scene, then rebuild the pptx
python build.py --skip-existing       # render only scenes not yet in slides/
python build.py --no-render           # just rebuild the pptx from slides/
python build.py --shrink              # also write visxai-small.pptx (../shrink_pptx.py)
python build.py --no-auto-next        # make every slide wait for a click
python build.py --dry-run             # print the commands without running them
```

Scenes flow into each other without an extra click: each scene ends with a
closing slide (its fade-out, or a section title wiping away) made with
`next_slide(auto_next=True)` -- `self.marked_next_slide(auto_next=True)`
then `self.clear_slide()`, or `self.new_section(..., auto_next=True)`.
manim-slides' pptx converter ignores `auto_next`, so after converting,
`build.py` runs `../pptx_auto_next.py` to give those slides a timed advance
(the length of their video). A click still advances early. When adding or
reordering scenes, give each one that is followed by another such an ending.

Data prep is off by default (the `.npz` files only need making once):

```bash
python build.py --prep                # run prep steps whose outputs are missing
python build.py --prep-only world_map # run one step (repeatable)
python build.py --prep-force          # rerun every step
```

Re-running the kNN or hubs prep can change the edges and pairs hand-picked
in `SHOWCASE_EDGES` / `PAIRS`; check the `*.candidates*.png` sheets.

To change the deck:

- **Slide order / which scenes:** edit `slide_sequence.txt`
  (`module.py SceneName` per line, `#` comments).
- **Prep steps and their arguments, output names:** edit the config block at
  the top of `build.py`.

Parallel renders (`-j`) log to `build_logs/<Scene>.log` and use one media
directory per scene (`media/jobs/`), since scenes sharing a media directory
can fail at random on the shared TeX cache.
