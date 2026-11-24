# THE LAST PAGE: crew contract

A 65 s hand-drawn 2D anime short (launch film for **Opus 5.6**), 2048x1152, 24 fps, 1560 frames, painted in pure Python
(numpy, opencv, scipy, pycairo, numba, Pillow). No image models, no external images, no fonts on the picture.
It must look like a finished Studio Ghibli-era theatrical film: gouache backgrounds, inked cels, real anime timing.
No people. The hero is one small red fox. Nothing from earlier films may be opened, read or reused.

Python: `./.mm/env/bin/python` (run from the root). ffmpeg: `./.mm/env/bin/ffmpeg`.

## Engine (read these files first; they are short)
- `lp/paint.py` `rgb mix smooth ramp fbm warp_by`, `Canvas.strokes` (real bristled brush marks: shape flat|round|leaf|taper, label clipping),
  `arcs` (build curved stroke polylines), `repaint` (gouache repaint of a guide picture; `lab=` keeps paint edges crisp), `mottle flat_dabs tooth jit_color scatter_mask`.
- `lp/ink.py` `Cel` (2x PyCairo cel: fill, shade(within=), clip, line), `spl` (Catmull-Rom, `corners=`), `ribbon` pressure lines, `xf boil resample noise1 over_roi paste`.
- `lp/cam.py` `Camera3D` (project, H_plane, warp_plane, plane_t, uv, clip_seg, rays), `voronoi_crevice`, rotations. `+X` right `+Y` forward `+Z` up, metres.
- `lp/comp.py` `over screen glow shaft bokeh disc_kernel cam2d radial blur finish to8 save`, timing helpers `ease keyed lerp_keys`.
- `lp/clouds.py` `lobes grow cumulus sea`. `lp/props.py` `lantern`. `lp/foxact.py` fox hook (lead owns; do not edit). `lp/fox.py` fox model (lead owns).
Add helpers in YOUR files. Do not edit engine files you do not own; if one has a bug, work around it and report it.

## Shot module interface: `shots/<id>.py`
```python
class Shot:
    id = 's1'; n = 168                      # frames, local f = 0..n-1
    finish = dict(hal_amt=.22, vig=.10)     # kwargs for comp.finish (grain, optics are applied by the driver, never by you)
    def setup(self): ...                    # paint plates once; cache as float16 .npy under cache/<id>_*.npy with a version key; must be fork-safe and fast when cached
    def frame(self, f): -> float32 (1152, 2048, 3) display-space RGB, may exceed 1 where light burns
```
`tools/still.py s1 0 40 80 --crop 900,400 --sheet` renders finished stills to `qa/s1/`. `tools/render.py s1 --jobs 4` renders `out/s1.mkv`.
A frame must cost < 2.5 s CPU (target < 1 s). Everything is seeded; a frame must be a pure function of f (no state carried between frames, frames render in parallel).

## Timeline (24 fps)
| shot | film frames | local n | owner |
|---|---|---|---|
| s1 Before Dawn | 0-168 | 168 | sky lane |
| s2 Reading Hall | 168-336 | 168 | library lane |
| s3 The Fox (close-up) | 336-432 | 96 | lead |
| s4 Blank Book | 432-600 | 168 | ink lane |
| s5 Library Stirs | 600-768 | 168 | library lane (fox insert: lead) |
| s6 Pages Burst | 768-936 | 168 | flock lane |
| s7 Sunrise | 936-1056 | 120 | balcony lane |
| s8 Great Bird | 1056-1248 | 192 | bird lane |
| s9 Last Page | 1248-1560 | 312 | balcony lane (9a, f 0-150), ink lane (9b title insert, f 150-312) |
Picture is shown at 1.85:1 on the page (22 px cropped top and bottom): keep important things off those bands.

## Colour script (hold to it; one film, one painter)
- s1 night, last hour: sky `#1b2348 -> #2c3666 -> #4f5a92 -> #8484b4 -> #a79ac0` with faint rose at the horizon. Moonlit cloud `#c3cbe8 / #8d95cc / #5d69ab / #2f3a78`. Stone `#8d93bd / #565d8c / #343a66`. Verdigris `#78a8a4 / #3f6f7d / #2b4a63`. The one warm light `#ffc46b`.
- s2-s5 library: never pure black. Dark `#1c1a30`, ambient `#2c2a48`, wood `#6a4b3c / #463239 / #2b2233`, moon shaft `#9fb4e0`, lantern `#ffb458` with short falloff. Books faded: oxblood `#6b3436`, moss `#4c5540`, indigo `#333a5e`, ochre `#8a6a3a`, plum `#54405c`, grey-green `#5d6a66`, bone `#b9a98a`.
- s6 violet dark -> gold. Pages lit `#f1e4c6`, half `#cbb9a8`, shadow `#8f86a6`, burning `#fff3cf`. Light `#ffd98a`.
- s7-s9 sunrise: sky `#5c97c4 -> #a8c9cf -> #f3dcae -> #fbd590`, sun `#fff3cf`. Cloud `#fff1d4 / #f3c9a6 / #b998ad / #8a7ba6`. Stone `#e6c9a0 / #b89a8c / #8d7690`. Grass `#d2c067 / #94904f / #5f6a4a`. Shadows are always coloured (violet, rose, blue), never grey or black.
Soft, natural, slightly low-contrast poster colour. No neon, no teal-orange, no HDR.

## The look
**Backgrounds**: poster-colour gouache on paper. Big simple value shapes first (squint test), then individually considered things: every book, stone, leaf clump, cloud lobe is its own mark with its own colour. Crisp masked edges on silhouettes, lost edges in shadow and mist. Visible flat-brush strokes that FOLLOW the form (along planks, around lobes, along leaves), never one global texture filter. Quiet areas matter: do not cover the frame in equal-size marks. Layered depth with air between layers. Off-axis composition, vanishing point near 55-65 % width.
**Cels** (fox, paper birds, pages, props that move): flat colour + ONE hard-edged shadow + optional rim, inked with `Cel.line` (warm dark line, 1.6-3.2 px, pressure, tapered ends, broken where a pencil would lift). No soft gradients on cels. Redraw wobble: jitter control points by 0.3-0.8 px per drawing (`boil`, seeded by drawing index).
**Light**: transmitted-light glow, shafts, dust, mist, coloured shadows, unite cel and background (a warm lantern must warm the floor, the books and the fox).
**Motion**: drawings on twos and threes (`d = f // 2` or `f // 3`), key poses with holds, anticipation, smears (stretched drawings with dry-brush speed lines) on the fastest frames, overlapping follow-through. Camera moves on ones, slow. Never float, never bob, never ease things like a web page, never slide rigid layers. Grass, ivy, paper bend as drawn curves.

**It must not look like**: 3D/CGI shading, vector art, uniform outlines, perfect ellipses and straight rulers, particle systems, copies (no two books, birds, stones or trees alike), muddy colour, empty areas that look unfinished, synthetic noise striping, HDR glow soup.

## Fox hook
`img = foxact.draw(img, key, f, M)` composites the fox at that depth. `M` = your marks (screen px), stored as a class constant so the lead can read it. The lead animates later; a grey-box stand-in is drawn now so you can judge scale. Leave room for the fox's whole silhouette and do not paint important things behind its head.
Contact shadow, lantern light on your background and glow are YOURS (paint them from the marks); the cel itself is the lead's.

## QA protocol (mandatory, per lane)
1. `python -m py_compile`, then look at your frames with the Read tool: full frame AND 1:1 crops (`--crop x,y` = 600x450 actual pixels) of at least three areas, on at least 6 different frames.
2. Write down, honestly, the three things that least look like a theatrical Ghibli still. Fix them. Repeat at least 4 rounds. Squint test every round: are there 3-5 clear value masses?
3. Look at motion as a strip of CONSECUTIVE frames (`--sheet` on e.g. 60 62 64 66 68 70) as well as stills.
4. Stop when a still would pass at arm's length as a frame from a film, or after about 8 rounds.
QA images are JPEG, at most ~60 files per lane; delete old ones. Do not leave processes running. Use at most 4 jobs (`--jobs 4`) and `cv2.setNumThreads(4)`; eight lanes share 18 cores and 24 GB (float16 caches, no giant arrays held).

## Report back (short, under 350 words)
Files written; fox marks; frame cost; what is strong; the honest weak spots left; engine problems found.
