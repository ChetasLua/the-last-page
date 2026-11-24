# The Last Page

A 65-second hand-drawn-style anime short, painted, animated and scored in pure Python. A small red fox wakes a library that has been asleep for a very long time.

2048×1152, 24 fps, 1,560 frames. No image models, no image files, no samples, no fonts: every stroke, cel, book, cloud and note comes from the code here, and every random seed is fixed.

This is an unofficial concept film. It is not made or endorsed by Anthropic, and it announces nothing.

The prompt is in [PROMPT.md](PROMPT.md). The working brief for the shots is in [docs/CONTRACT.md](docs/CONTRACT.md).

## Build

pycairo has no macOS wheel, so the environment comes from conda-forge ([micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)). Run from the repository root:

```bash
micromamba create -y -p .mm/env -c conda-forge python=3.12 numpy opencv pycairo scipy pillow numba ffmpeg pysoundfile
```

```bash
./.mm/env/bin/python audio/build.py
```

```bash
./.mm/env/bin/python tools/master.py --jobs 8
```

That writes `dist/the-last-page-16x9.mp4`, the full-quality 16:9 master. On an 18-core M5 Pro the render takes about 7 minutes once the background plates are cached; the first run also paints the plates.

For the single self-contained web page (`dist/the-last-page.html`), render the nine shots and assemble:

```bash
for s in s1 s2 s3 s4 s5 s6 s7 s8 s9; do ./.mm/env/bin/python tools/render.py $s --jobs 6; done
```

```bash
./.mm/env/bin/python tools/assemble.py --rate 7400k --preset slower --html
```

One frame at a time, with 1:1 crops:

```bash
./.mm/env/bin/python tools/still.py s3 40 --crop 620,330
```

## What is where

| Path | What it does |
|---|---|
| `lp/paint.py` | gouache engine: bristled brush strokes rasterised in numba, painterly repaint with masked paint edges |
| `lp/ink.py` | 2x supersampled pycairo cels, splines, pressure-varied ink lines |
| `lp/cam.py` | metric pinhole camera, plane homographies, near-plane clipping, Voronoi masonry |
| `lp/comp.py` | compositing, light shafts, disc bokeh, 35mm finish (halation, shoulder, grain) |
| `lp/fox*.py` | the fox: three-quarter head, profile, sitting body, paw, tail, and its acting in every shot |
| `lp/clouds.py` `plants.py` `books.py` `pages.py` | painters for clouds, pines, ivy, pampas, shelves and paper birds |
| `lp/brush.py` `ink_letters.py` | brush strokes that draw themselves, the title lettering and the seal |
| `lp/greatbird.py` `gb_art.py` `gb_sky.py` | the great bird of pages and its sky |
| `shots/` | the nine shots |
| `audio/` | the score and sound effects, all synthesised |
| `tools/` | stills, shot renders, the master, the web page, a playback test |
