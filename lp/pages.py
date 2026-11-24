"""Paper for The Last Page: loose handwritten pages and PAPER BIRDS folded from pages.  Owner: flock lane.

Every bird is a 3D fold model (13 fold designs x 8 wing drawings) that is projected, cut out as flat paper facets with one
hard lit / half / shadow split, redrawn with a little wobble per drawing (pass drawing = f // 2) and inked.

USAGE
    from lp import pages as PG
    NAMES: crane swallow gull dove pheasant torn engraved heron swift owl dart codex wren   (index or name)
    PHASES: 0 top  1 breaking down  2 down-stroke smear  3 bottom  4 folding up (edge-on)  5 wrist leads  6 nearly top   7 glide
    8 blade (wings shut upright)  9 half sprung open
            PG.CYCLE = (0, 1, 2, 3, 4, 5, 6);  phase = PG.CYCLE[(drawing + offset) % 7]

  near cels (ink, creases, curled corners, handwriting, illustrations):
    R = PG.basis(fwd, up)                       # camera-space unit vectors: x right, y DOWN, z away from the viewer
    c = Cel(...)                                # lp.ink.Cel
    PG.bird(c, 'crane', phase, R, (x, y), span_px, pal='cream', seed=7, drawing=d, light=(-.4, -.8, -.4))
    PG.fold(c, 'kite', (x, y), size_px, ...)     # stages of a sheet folding itself: taut crease half kite petal
    PG.page(c, R, (x, y), size_px, pal='cream', seed=3, drawing=d, bend=.9, light=...)      # one curled sheet
    or straight into a frame:  img = PG.put_bird(img, ...same args...)   img = PG.put_page(img, ...)

  crowds (2..200 px, hundreds to thousands, painter-sorted, any Camera3D):
    PG.flock(img, cam, pos, fwd, up, span_m, design, phase, seed, drawing, light_w, pal, tint=None, smear=None)
    PG.sheets(img, cam, pos, nrm, axis, size_m, bend, seed, drawing, light_w, pal)
    img is float32 RGB, painted in place.  pal: one palette name, or an int array into a list of names (pals=[...]).

  palettes PG.PAL: 'cream' (contract s6), 'violet' (moonlit dark), 'deep' (silhouette violet), 'gold' (burning), 'dawn' (s7-s8)
  low level: PG.fill_polys(img, V, N, C, op) antialiased painter-order polygon batch; PG.scribble() handwriting polylines.
"""
import numpy as np, cv2, math
from numba import njit
from . import W, H
from .paint import rgb, f32, mix, Canvas
from .ink import Cel, spl, over_roi, ribbon, noise1

# ------------------------------------------------------------------ palettes
PAL = {
    'cream':  dict(lit='#f1e4c6', half='#cbb9a8', shade='#8f86a6', deep='#655d8c', ink='#2c2338', text='#5c4a5e', rim='#fff8e4'),
    'violet': dict(lit='#c9c4e0', half='#9d97c0', shade='#716a9e', deep='#4b4677', ink='#1d1936', text='#4a4470', rim='#dcdcf6'),
    'deep':   dict(lit='#7d78a6', half='#5f5a8a', shade='#474270', deep='#332f58', ink='#171430', text='#3a3560', rim='#b3b4dc'),
    'gold':   dict(lit='#fff5d6', half='#ffe3a2', shade='#f2c687', deep='#cf9a80', ink='#8f5540', text='#b67a45', rim='#fffbe9'),
    'ember':  dict(lit='#ffe3a6', half='#e9b274', shade='#a8798a', deep='#6f5a86', ink='#4a2c3a', text='#8a5a4c', rim='#fff3cf'),
    'dawn':   dict(lit='#fff1d4', half='#f0cfa6', shade='#b79bb2', deep='#8d7ba6', ink='#59405a', text='#9a7a80', rim='#fffaf0'),
}
TONES = ('lit', 'half', 'shade', 'deep', 'ink', 'rim')


def pal_array(p):
    d = PAL[p] if isinstance(p, str) else p
    return np.stack([rgb(d[k]) if isinstance(d[k], str) else np.asarray(d[k], f32) for k in TONES])


# page tints: aged paper, some foxed, some cool, a few coloured endpapers
TINTS = np.array([[1, 1, 1], [1, .985, .94], [1.01, .965, .90], [.965, .98, 1.0], [1.0, .95, .925], [.985, .955, .96],
                  [1.02, 1.0, .965], [.96, .95, .91], [1.0, .99, .95], [.93, .90, .86], [1.0, .86, .80], [.84, .90, .95]], f32)
TINT_P = np.array([.16, .16, .12, .09, .09, .08, .10, .07, .07, .03, .015, .015]); TINT_P = TINT_P / TINT_P.sum()


def tints(seed):
    s = np.atleast_1d(np.asarray(seed)).astype(np.int64)
    u = ((s * 2654435761) % 4294967296) / 4294967296.
    return TINTS[np.searchsorted(np.cumsum(TINT_P), u, side='right').clip(0, len(TINTS) - 1)]


def hash01(a, b=0):
    n = (np.asarray(a).astype(np.int64) * 374761393 + np.asarray(b).astype(np.int64) * 668265263 + 1274126177) & 0xffffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0xffffffff
    n = (((n ^ (n >> 16)) & 0x7fffffff) * 1103515245) & 0xffffffff
    return ((n ^ (n >> 15)) & 0xffffff) / 16777216.


# ------------------------------------------------------------------ batch polygon rasteriser
@njit(cache=True, fastmath=True)
def _fill_polys(img, A, V, N, C, OP, hasA):
    h, w = img.shape[0], img.shape[1]
    cov = np.zeros(w + 4, np.float32)
    xs = np.zeros(32, np.float32)
    for i in range(V.shape[0]):
        nv = N[i]
        if nv < 3 or OP[i] <= 0: continue
        x0 = 1e9; y0 = 1e9; x1 = -1e9; y1 = -1e9
        for k in range(nv):
            px = V[i, k, 0]; py = V[i, k, 1]
            if not (px > -1e6 and px < 1e6 and py > -1e6 and py < 1e6):
                x0 = 1e9; break
            if px < x0: x0 = px
            if px > x1: x1 = px
            if py < y0: y0 = py
            if py > y1: y1 = py
        if x0 > 1e8 or x1 < 0 or y1 < 0 or x0 >= w or y0 >= h: continue
        ya = max(0, int(math.floor(y0))); yb = min(h - 1, int(math.floor(y1)))
        xa = max(0, int(math.floor(x0))); xb = min(w - 1, int(math.floor(x1)) + 1)
        ns = 8 if (y1 - y0) < 60 else 5
        inv = 1.0 / ns
        for y in range(ya, yb + 1):
            for x in range(xa, xb + 2): cov[x] = 0
            hit = False
            for s in range(ns):
                sy = y + (s + .5) * inv
                cnt = 0
                for k in range(nv):
                    k2 = k + 1 if k + 1 < nv else 0
                    ay = V[i, k, 1]; by = V[i, k2, 1]
                    if (ay <= sy and by > sy) or (by <= sy and ay > sy):
                        if cnt < 32:
                            xs[cnt] = V[i, k, 0] + (sy - ay) / (by - ay) * (V[i, k2, 0] - V[i, k, 0]); cnt += 1
                if cnt < 2: continue
                for p in range(1, cnt):
                    v = xs[p]; q = p - 1
                    while q >= 0 and xs[q] > v:
                        xs[q + 1] = xs[q]; q -= 1
                    xs[q + 1] = v
                for p in range(0, cnt - 1, 2):
                    xl = xs[p]; xr = xs[p + 1]
                    if xl < 0: xl = 0.0
                    if xr > w: xr = float(w)
                    if xr <= xl: continue
                    hit = True
                    il = int(xl); ir = int(xr)
                    if il == ir: cov[il] += (xr - xl) * inv
                    else:
                        cov[il] += (il + 1 - xl) * inv
                        for x in range(il + 1, ir): cov[x] += inv
                        cov[ir] += (xr - ir) * inv
            if not hit: continue
            op = OP[i]
            for x in range(xa, xb + 1):
                a = cov[x]
                if a <= .002: continue
                if a > 1: a = 1.0
                a *= op
                img[y, x, 0] += (C[i, 0] - img[y, x, 0]) * a
                img[y, x, 1] += (C[i, 1] - img[y, x, 1]) * a
                img[y, x, 2] += (C[i, 2] - img[y, x, 2]) * a
                if hasA: A[y, x] += (1 - A[y, x]) * a


_NOA = np.zeros((1, 1), f32)


def fill_polys(img, V, N, C, op=1.0, A=None):
    """painter-order batch.  V (n,M,2) px, N (n,) vertex counts, C (n,3), op scalar or (n,).  In place."""
    V = np.ascontiguousarray(V, f32); n = len(V)
    if n == 0: return img
    N = np.broadcast_to(np.asarray(N, np.int32), (n,)).copy()
    C = np.broadcast_to(np.asarray(C, f32), (n, 3)).copy()
    OP = np.broadcast_to(np.asarray(op, f32), (n,)).copy()
    _fill_polys(img, _NOA if A is None else A, V, N, C, OP, A is not None)
    return img


def edge_quads(P, Q, w, taper=.35, mid=None):
    """thin tapered ink strokes from P to Q (n,2) as 6-gons, optionally bowed through mid.  returns (n,6,2)"""
    P = np.asarray(P, f32); Q = np.asarray(Q, f32)
    d = Q - P; L = np.sqrt((d ** 2).sum(-1, keepdims=True)) + 1e-6
    nrm = np.concatenate([-d[..., 1:2], d[..., 0:1]], -1) / L
    w = np.asarray(w, f32)
    w = np.broadcast_to(w if w.ndim == 0 else w[..., None], nrm.shape[:-1] + (1,)) * .5
    M = (P + Q) * .5 if mid is None else np.asarray(mid, f32)
    return np.stack([P + nrm * w * taper, M + nrm * w, Q + nrm * w * taper, Q - nrm * w * taper, M - nrm * w, P - nrm * w * taper], -2)


# ------------------------------------------------------------------ fold designs
def _wing(L0, L1, L2, T2, T1, T0, C0, C1, pl=.05, notch=None):
    return dict(L0=L0, L1=L1, L2=L2, T2=T2, T1=T1, T0=T0, C0=C0, C1=C1, pl=pl, notch=notch)


DES = {
    # name: wing, body(bw, y0, y1, zt, zb), neck [(y, z)...top ridge: base, (mid), head, beak] widths, tail(len, half width root, half width tip, notch, rise)
    'crane':    dict(w=_wing((.05, .25), (.56, .34), (1.06, -.02), (1.0, -.13), (.50, -.27), (.05, -.26), (.05, .0), (.54, .05), .06),
                     b=(.10, .26, -.30, .09, .11), n=[(.22, .06), (.60, .34), (.82, .25)], nw=(.07, .045), t=(.64, .055, .02, -.0, .20)),
    'swallow':  dict(w=_wing((.05, .18), (.48, .27), (1.18, -.36), (1.12, -.40), (.50, -.03), (.05, -.12), (.05, .04), (.5, .10), .04),
                     b=(.085, .24, -.24, .07, .07), n=[(.2, .03), (.33, .06), (.43, .02)], nw=(.08, .06), t=(.66, .05, .26, .34, .02)),
    'gull':     dict(w=_wing((.05, .20), (.56, .25), (1.30, -.06), (1.26, -.12), (.54, -.09), (.05, -.13), (.05, .04), (.55, .09), .04),
                     b=(.095, .28, -.28, .08, .08), n=[(.24, .04), (.38, .06), (.52, .02)], nw=(.085, .06), t=(.42, .06, .16, .05, .02)),
    'dove':     dict(w=_wing((.06, .24), (.46, .32), (.90, -.02), (.80, -.22), (.42, -.32), (.06, -.26), (.06, .0), (.45, .03), .05),
                     b=(.16, .32, -.30, .10, .17), n=[(.26, .08), (.40, .15), (.50, .07)], nw=(.12, .08), t=(.56, .08, .25, -.03, .05)),
    'pheasant': dict(w=_wing((.05, .22), (.46, .31), (.86, .10), (.82, -.14), (.42, -.28), (.05, -.22), (.05, .0), (.45, .04), .05),
                     b=(.11, .28, -.30, .09, .12), n=[(.23, .05), (.46, .26), (.60, .18)], nw=(.075, .05), t=(1.42, .07, .06, .16, .16)),
    'torn':     dict(w=_wing((.05, .22), (.52, .29), (1.12, .02), (1.02, -.10), (.52, -.22), (.05, -.20), (.05, .02), (.53, .06), .05, notch=((.93, -.02), (.78, -.21))),
                     b=(.10, .28, -.28, .08, .09), n=[(.24, .05), (.42, .14), (.55, .06)], nw=(.085, .055), t=(.50, .06, .12, .16, .05)),
    'engraved': dict(w=_wing((.05, .27), (.60, .38), (1.10, .10), (1.04, -.18), (.56, -.34), (.05, -.28), (.05, .0), (.58, .03), .035),
                     b=(.11, .28, -.30, .09, .11), n=[(.24, .05), (.45, .18), (.60, .10)], nw=(.085, .055), t=(.58, .06, .18, -.06, .06)),
    'heron':    dict(w=_wing((.05, .26), (.58, .30), (1.16, .04), (1.10, -.20), (.54, -.36), (.05, -.30), (.05, -.0), (.56, -.01), .07),
                     b=(.095, .26, -.28, .08, .10), n=[(.22, .05), (.40, .24), (.52, .22), (.82, .12)], nw=(.06, .04), t=(.94, .04, .03, .0, -.02)),
    'swift':    dict(w=_wing((.04, .15), (.40, .24), (1.28, -.30), (1.24, -.34), (.46, .06), (.04, -.07), (.04, .04), (.44, .14), .03),
                     b=(.07, .25, -.26, .06, .06), n=[(.2, .02), (.30, .03), (.37, .0)], nw=(.075, .06), t=(.46, .04, .10, .10, .0)),
    'owl':      dict(w=_wing((.06, .30), (.52, .40), (.98, .22), (1.0, -.16), (.50, -.36), (.06, -.30), (.06, .0), (.51, .03), .05),
                     b=(.17, .30, -.28, .12, .14), n=[(.22, .10), (.36, .17), (.42, .06)], nw=(.16, .13), t=(.42, .10, .20, -.02, .02)),
    'dart':     dict(w=_wing((.02, .62), (.34, .06), (.78, -.58), (.70, -.64), (.30, -.60), (.02, -.56), (.02, .05), (.30, -.22), .02),
                     b=(.035, .64, -.56, .02, .17), n=[(.6, .0), (.66, .01), (.74, .0)], nw=(.02, .015), t=(.58, .02, .03, .02, .0)),
    'codex':    dict(w=_wing((.02, .40), (.42, .43), (.84, .37), (.86, -.37), (.42, -.43), (.02, -.40), (.02, .0), (.42, .0), .0),
                     b=(.035, .38, -.38, .03, .04), n=[(.3, .02), (.42, .04), (.48, .02)], nw=(.03, .02), t=(.42, .03, .05, .0, .02)),
    'wren':     dict(w=_wing((.06, .20), (.38, .29), (.70, .08), (.66, -.12), (.34, -.26), (.06, -.22), (.06, .0), (.37, .02), .04),
                     b=(.15, .30, -.26, .11, .14), n=[(.24, .09), (.36, .15), (.46, .10)], nw=(.11, .07), t=(.48, .05, .10, .02, .22)),
}
NAMES = list(DES)
IDX = {n: i for i, n in enumerate(NAMES)}
CYCLE = (0, 1, 2, 3, 4, 5, 6)
# phase: inner angle, outer relative angle (deg), outer sweep back, twist, body bob
FLAP = [(63, 12, .00, -.06, -.03), (36, 27, .04, .12, .00), (2, 27, .02, .20, .05), (-34, -3, .00, .10, .08),
        (-17, -50, .26, -.16, .03), (20, -54, .34, -.20, -.02), (50, -24, .16, -.10, -.04), (13, -5, .03, .02, .0), (84, 3, .02, .0, .0), (70, -6, .05, .02, .0)]
F_, V_ = 20, 6


def _rot(p, a, ax, ang):
    ax = ax / np.linalg.norm(ax); v = p - a
    c, s = math.cos(ang), math.sin(ang)
    return a + v * c + np.cross(ax, v) * s + ax * (v @ ax)[..., None] * (1 - c)


def _build(D, ph):
    w = D['w']; ai, ao, sw, tw, bob = FLAP[ph]
    ai, ao = math.radians(ai), math.radians(ao)
    if D is DES['codex']: ao *= .55; sw = 0
    if D is DES['dart']: ai, ao, sw = ai * .35 + .1, ao * .25, 0
    facets = []

    def add(name, pts, ew, text=0, curl=-1):
        facets.append((name, np.asarray(pts, float), ew, text, curl))
    bw, y0, y1, zt, zb = D['b']
    zh = .03
    P = {k: np.array([w[k][0], w[k][1], zh]) for k in ('L0', 'L1', 'L2', 'T2', 'T1', 'T0', 'C0', 'C1')}
    if w['notch']: P['N1'] = np.array([*w['notch'][0], zh]); P['N2'] = np.array([*w['notch'][1], zh])
    pl = w['pl'] * 1.75
    P['C0'][2] += pl * .6; P['C1'][2] += pl
    P['L1'][2] -= pl * .25; P['T1'][2] -= pl * .8; P['T0'][2] -= pl * .3
    P['T2'][2] -= pl * .9
    # chord twist (leading edge down on the power stroke)
    for k in P:
        s = P[k][0]
        P[k][2] -= tw * (P[k][1] - .02) * (.4 + s * .6)
    outer = [k for k in ('L2', 'T2', 'N1', 'N2') if k in P]
    a, ax = P['T1'].copy(), P['L1'] - P['T1']
    # positive relative angle lifts the tip
    for k in outer:
        P[k] = _rot(P[k], a, ax, -ao)
        P[k][1] -= sw * (P[k][0] - P['T1'][0] + .25)
        P[k][0] -= sw * .30
    for k in P:
        P[k] = _rot(P[k], np.array([P['L0'][0] * .8, 0, zh]), np.array([0, 1., 0]), -ai)
        P[k][2] += bob
    for sx in (1, -1):
        Q = {k: v * np.array([sx, 1, 1]) for k, v in P.items()}
        q = 'R' if sx > 0 else 'L'
        add('wiL' + q, [Q['L0'], Q['L1'], Q['C1'], Q['C0']], (1, .55, .55, .0), 1)
        fg = D.get('fing', .0 if (w['notch'] or D is DES['dart'] or D is DES['codex']) else .34)
        if fg > 0:
            a_, b_, c_ = Q['T1'], Q['T0'], Q['C0'] * .5 + Q['C1'] * .5
            n3 = a_ * .62 + b_ * .38; n3 = n3 + (c_ - n3) * fg * .75; n4 = a_ * .44 + b_ * .56 + (b_ - a_) * .0 + np.array([0, -.03, 0]); n4 = n4 + (n4 - c_) * .10
            add('wiT' + q, [Q['C0'], Q['C1'], Q['T1'], n3, n4, Q['T0']], (.0, .55, 1, 1, 1, .3), 1, 5)
        else: add('wiT' + q, [Q['C0'], Q['C1'], Q['T1'], Q['T0']], (.0, .55, 1, .3), 1, 3)
        add('woL' + q, [Q['L1'], Q['L2'], Q['C1']], (1, .5, .0), 2)
        if 'N1' in Q: add('woT' + q, [Q['C1'], Q['L2'], Q['T2'], Q['N1'], Q['N2'], Q['T1']], (.0, 1, 1, 1, 1, .0), 1)
        elif fg > 0:
            a_, b_, c_ = Q['T2'], Q['T1'], Q['C1']
            n1 = a_ * .58 + b_ * .42; n1 = n1 + (c_ - n1) * fg; n2 = a_ * .42 + b_ * .58; n2 = n2 + (n2 - c_) * .12
            add('woT' + q, [Q['C1'], Q['L2'], Q['T2'], n1, n2, Q['T1']], (.0, 1, 1, 1, 1, .0), 1, 2)
        else: add('woT' + q, [Q['C1'], Q['L2'], Q['T2'], Q['T1']], (.0, 1, 1, .0), 1, 2)
        z = np.array([0, 0, bob])
        rf, rb = np.array([0, y0, zt * .7]) + z, np.array([0, y1, zt * .5]) + z
        sf, sb = np.array([sx * bw, y0 * .55, -.005]) + z, np.array([sx * bw * .62, y1 * .72, .0]) + z
        kf, kb = np.array([0, y0 * .72, -zb * .8]) + z, np.array([0, y1 * .55, -zb * .75]) + z
        rm = np.array([0, (y0 + y1) * .5, zt]) + z
        add('bu' + q, [rf, rm, rb, sb, sf], (.0, .0, .4, .5, .5), 0)
        add('bl' + q, [rf, sf, sb, rb, kb, kf], (.0, .4, .0, .8, 1, .8), 0)
        n = D['n']; nw, hw = D['nw']
        top = [np.array([0, y, zz]) + z for y, zz in n]
        base_b = np.array([sx * nw, n[0][0] - .03, n[0][1] - .10]) + z
        hd = n[-2]
        head_b = np.array([sx * hw, hd[0] - .015, hd[1] - .055 - hw * .3]) + z
        if len(n) == 4:
            mid_b = np.array([sx * (nw + hw) * .45, n[1][0] + .06, n[1][1] - .10]) + z
            add('nk' + q, [top[0], top[1], top[2], mid_b, base_b], (.9, .9, .0, 1, .0), 0)
            add('hd' + q, [top[2], top[3], head_b, mid_b], (1, 1, 1, .0), 0)
        else:
            mid = top[0] * .45 + top[1] * .55; mid[2] += .012
            add('nk' + q, [top[0], mid, top[1], head_b, base_b], (.9, .9, .4, 1, .0), 0)
            add('hd' + q, [top[1], top[2], head_b], (1, 1, .0), 0)
        tl, tw0, tw1, nt, rise = D['t']
        r0, r1 = np.array([0, y1 * .80, .03]) + z, np.array([sx * tw0, y1 * .72, .005]) + z
        te, tc = np.array([sx * tw1, -tl, rise - .015]) + z, np.array([0, -tl + nt, rise + .015]) + z
        if tl > 1.2:
            m1 = np.array([sx * tw0 * 1.5, -tl * .55, rise * .55]) + z; m0 = np.array([0, -tl * .55, rise * .62]) + z
            add('t0' + q, [r0, r1, m1, m0], (.0, 1, .0, .5), 0)
            add('t1' + q, [m0, m1, te, tc], (.0, 1, 1, .5), 0)
        else:
            add('t0' + q, [r0, r1, te, tc], (.0, 1, 1, .55), 1 if tw1 > .12 else 0)
    return facets


def _tables():
    D, P = len(NAMES), len(FLAP)
    G = np.zeros((D, P, F_, V_, 3), f32); NV = np.zeros((D, F_), np.int32); EW = np.zeros((D, F_, V_), f32)
    TX = np.zeros((D, F_), np.int32); CU = -np.ones((D, F_), np.int32); NM = []
    VID = -np.ones((D, F_, V_), np.int32)
    for d, nme in enumerate(NAMES):
        for p in range(P):
            fs = _build(DES[nme], p)
            for i, (nm, v, ew, tx, cu) in enumerate(fs):
                G[d, p, i, :len(v)] = v; G[d, p, i, len(v):] = v[-1]
                if p == 0:
                    NV[d, i] = len(v); EW[d, i, :len(ew)] = ew; TX[d, i] = tx; CU[d, i] = cu
        NM.append([f[0] for f in _build(DES[nme], 7)])
        ids = {}
        fs = _build(DES[nme], 7)
        for i, (nm, v, *_r) in enumerate(fs):
            for k in range(len(v)):
                key = tuple(np.round(v[k] + 1e-9, 4)); VID[d, i, k] = ids.setdefault(key, len(ids))
    # far models: 5 marks (near wing halves + body)
    GF = np.zeros((D, P, 5, 6, 3), f32)
    for d, nme in enumerate(NAMES):
        for p in range(P):
            g = G[d, p]
            nm = NM[d]
            ix = {n: i for i, n in enumerate(nm)}
            for s, q in enumerate('RL'):
                iL, iT, oL, oT = g[ix['wiL' + q]], g[ix['wiT' + q]], g[ix['woL' + q]], g[ix['woT' + q]]
                GF[d, p, s * 2] = np.stack([iL[0], iL[1], oL[1], iT[1], iT[0], iT[0]])       # lead strip
                GF[d, p, s * 2 + 1] = np.stack([iT[0], iT[1], oL[1], oT[2], iT[2], iT[NV[d, ix['wiT' + q]] - 1]])   # trail strip
            hd = g[ix['hdR']]; t = g[ix['t1R']] if 't1R' in ix else g[ix['t0R']]; tl = g[ix['t1L']] if 't1L' in ix else g[ix['t0L']]
            bl = g[ix['blR']]; bu = g[ix['buR']]; buL = g[ix['buL']]
            GF[d, p, 4] = np.stack([hd[1], bu[4], bu[3], t[2], tl[2], buL[4]])
    return G, NV, EW, TX, CU, NM, VID, GF


GEO, NVT, EWT, TXT, CUT, NMT, VIDT, GEOF = _tables()


def basis(fwd, up):
    """camera-space basis matrix (columns right, forward, up) from camera-space heading and up hint.  (…,3) arrays ok"""
    y = np.asarray(fwd, float); u = np.asarray(up, float)
    y = y / (np.linalg.norm(y, axis=-1, keepdims=True) + 1e-9)
    x = np.cross(y, u); x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
    z = np.cross(x, y)
    return np.stack([x, y, z], -1)


# ------------------------------------------------------------------ handwriting and old engravings
# pen skeletons in x-height units (x right, y up): points, extra marks
_LET = {
    'n': ([(0, .1), (.10, .92), (.15, .02), (.26, .86), (.40, .84), (.43, .06), (.56, .12)], ()),
    'e': ([(0, .10), (.26, .50), (.31, .92), (.15, .92), (.08, .36), (.24, .0), (.46, .16)], ()),
    'l': ([(0, .10), (.22, 1.1), (.28, 2.0), (.15, 1.9), (.10, .8), (.18, .0), (.38, .16)], ()),
    'o': ([(0, .3), (.14, .90), (.02, .5), (.14, .02), (.36, .22), (.36, .78), (.22, .92), (.52, .72)], ()),
    'u': ([(0, .55), (.06, .92), (.10, .12), (.28, .02), (.40, .88), (.43, .08), (.57, .14)], ()),
    'g': ([(0, .25), (.30, .92), (.09, .74), (.12, .10), (.35, .5), (.39, .9), (.36, -.8), (.17, -1.15), (.12, -.7), (.48, .10)], ()),
    't': ([(0, .1), (.20, 1.5), (.21, .3), (.29, .0), (.46, .18)], (((.04, 1.02), (.40, 1.08)),)),
    'i': ([(0, .1), (.15, .88), (.17, .15), (.27, .0), (.38, .18)], (((.20, 1.34), (.24, 1.40)),)),
    's': ([(0, .1), (.24, .98), (.31, .52), (.18, .02), (.02, .14), (.40, .12)], ()),
    'f': ([(0, .1), (.27, 1.4), (.34, 1.95), (.22, 1.85), (.18, -.6), (.10, -1.1), (.20, -.25), (.43, .2)], ()),
    'a': ([(.32, .80), (.10, .82), (.05, .2), (.22, .06), (.34, .84), (.35, .10), (.50, .14)], ()),
    'd': ([(.30, .74), (.08, .78), (.05, .18), (.24, .08), (.36, 1.9), (.35, .10), (.50, .16)], ()),
    'h': ([(0, .1), (.22, 1.3), (.26, 1.95), (.12, 1.7), (.10, .0), (.24, .82), (.38, .78), (.40, .04), (.54, .14)], ()),
    'r': ([(0, .1), (.12, .95), (.18, 1.02), (.30, .80), (.28, .10), (.44, .10)], ()),
    'y': ([(0, .6), (.06, .92), (.12, .2), (.30, .12), (.40, .9), (.36, -.7), (.16, -1.15), (.12, -.65), (.48, .12)], ()),
    'v': ([(0, .7), (.08, .94), (.18, .02), (.36, .84), (.44, .92), (.56, .74)], ()),
}
_LN = list(_LET); _LP = np.array([9, 12, 5, 8, 6, 3, 8, 7, 6, 2.5, 8, 4, 5, 6, 2.5, 3.]); _LP = _LP / _LP.sum()


def scribble(r, length, size, pts=6, slant=.30):
    """one row of handwriting as a list of polylines in local units, baseline y=0 (y down).  pts: 4 coarse .. 11 smooth"""
    out = []; x = 0.0
    while x < length:
        nl = int(np.clip(r.geometric(.22) + (r.random() < .8), 1, 10)); xh = size * .44 * r.uniform(.9, 1.1)
        P = []; ex = []; cx = 0.0
        for k in range(nl):
            L, extra = _LET[_LN[int(r.choice(len(_LN), p=_LP))]]
            q = np.array(L) * [r.uniform(.85, 1.25) * 1.18, r.uniform(.86, 1.12)] + r.normal(0, .025, (len(L), 2))
            for e in extra: ex.append(np.array(e) * [1.18, 1] + [cx, 0] + r.normal(0, .03, (2, 2)))
            P.append(q + [cx, 0]); cx += q[-1, 0] + .02
        w = cx * xh
        if x + w > length:
            if not out or length - x < size: break
            k = max(1, int(nl * (length - x) / w)); P = P[:k]; ex = [e for e in ex if e[0, 0] * xh < length - x]
        P = np.vstack(P)
        if pts >= 7: P = spl(P, n=3)
        elif pts <= 4: P = P[::2] if len(P) > 6 else P
        f = lambda q: np.c_[x + (q[:, 0] + slant * q[:, 1]) * xh, -q[:, 1] * xh]
        out.append(f(P))
        if pts >= 5: out += [f(e) for e in ex]
        x += min(w, length - x) + size * r.uniform(.30, .62) * (2.2 if r.random() < .07 else 1)
    return out


def motif(kind, r):
    """tiny engraving in a -1..1 box: list of (polyline, weight, closed)"""
    L = []
    a = np.linspace(0, 2 * math.pi, 26)
    if kind == 0:      # sun with a face of rays
        L.append((np.c_[np.cos(a), np.sin(a)] * .38, 1.0))
        for i in range(14):
            t = i / 14 * 6.283; q = (.46, .9 if i % 2 else .7)
            L.append((np.array([[math.cos(t + .05) * q[0], math.sin(t + .05) * q[0]], [math.cos(t + .02) * q[1], math.sin(t + .02) * q[1]], [math.cos(t - .05) * q[0], math.sin(t - .05) * q[0]]]), .6))
        L.append((np.array([[-.14, -.06], [-.08, -.11], [-.02, -.06]]), .7)); L.append((np.array([[.04, -.06], [.1, -.11], [.16, -.06]]), .7))
        L.append((np.array([[-.12, .13], [0, .2], [.13, .12]]), .7))
    elif kind == 1:    # little ship
        L.append((np.array([[-.75, .25], [-.55, .55], [.5, .55], [.8, .18], [.4, .3], [-.5, .3], [-.75, .25]]), 1.0))
        L.append((np.array([[-.15, .3], [-.12, -.85]]), .9)); L.append((np.array([[.35, .3], [.36, -.6]]), .8))
        L.append((np.array([[-.12, -.75], [.18, -.5], [.22, -.1], [-.12, .1]]), .8)); L.append((np.array([[-.12, -.75], [-.5, -.35], [-.55, .0], [-.12, .1]]), .8))
        L.append((np.array([[.36, -.55], [.62, -.3], [.62, .02], [.36, .12]]), .8))
        for k in range(4): L.append((np.c_[np.linspace(-.95, .95, 12), .68 + .07 * k + .04 * np.sin(np.linspace(0, 14, 12) + k)], .5))
        for k in range(5): L.append((np.array([[-.45 + k * .2, .35], [-.48 + k * .2, .5]]), .4))
    elif kind == 2:    # sprig
        s = np.c_[np.linspace(-.1, .12, 10) + .08 * np.sin(np.linspace(0, 3, 10)), np.linspace(.95, -.85, 10)]
        L.append((s, 1.0))
        for k in range(7):
            y = .65 - k * .22; sd = 1 if k % 2 else -1; x = .0
            L.append((np.array([[x, y], [x + sd * .25, y - .22], [x + sd * .52, y - .22], [x + sd * .34, y - .02], [x, y]]), .75))
            L.append((np.array([[x, y], [x + sd * .42, y - .18]]), .45))
    elif kind == 3:    # compass rose
        L.append((np.c_[np.cos(a), np.sin(a)] * .62, .7)); L.append((np.c_[np.cos(a), np.sin(a)] * .55, .4))
        for i in range(8):
            t = i * math.pi / 4; q = .92 if i % 2 == 0 else .5
            p = np.array([math.cos(t), math.sin(t)]); n = np.array([-p[1], p[0]])
            L.append((np.array([n * .09 + p * .1, p * q, -n * .09 + p * .1]), .8))
            L.append((np.array([p * .1, p * q]), .4))
    else:              # tower on a hill
        L.append((np.array([[-.95, .8], [-.5, .5], [0, .42], [.5, .52], [.95, .82]]), .9))
        L.append((np.array([[-.2, .44], [-.17, -.5], [.17, -.5], [.2, .44]]), .9)); L.append((np.array([[-.25, -.5], [0, -.95], [.25, -.5], [-.25, -.5]]), .9))
        for k in range(4): L.append((np.array([[-.15, -.3 + k * .18], [-.02, -.3 + k * .18]]), .45))
        L.append((np.array([[-.05, .42], [-.05, .22], [0, .16], [.05, .22], [.05, .42]]), .7))
        for k in range(6): L.append((np.array([[-.8 + k * .3, .72], [-.74 + k * .3, .62]]), .4))
    return L


def _affine_rows(ctx, M, o, ext, r, size, col, alpha, lw, art=None):
    """handwriting rows in a local frame: M 2x2 maps local px->screen, o origin, ext half extents (local px)"""
    ctx.set_line_cap(1); ctx.set_line_join(1)
    y = -ext[1] + size * r.uniform(.3, 1)
    box = None
    if art is not None:
        bs = min(ext) * .8; bx, by = r.uniform(-.25, .25) * ext[0], r.uniform(-.2, .2) * ext[1]; box = (bx, by, bs)
    fx, fy = r.uniform(-1, 1) * ext[0], r.uniform(-1, 1) * ext[1]          # where the ink has gone
    para = 0
    while y < ext[1]:
        slope = r.normal(0, .016)
        x0 = -ext[0] + size * (r.uniform(1.5, 4) if r.random() < .22 else 0); ln = 2 * ext[0] * (r.uniform(.22, .75) if r.random() < .34 else 1)
        if r.random() < .09: y += size * r.uniform(1.2, 2.2); continue
        fade = 1 - .72 * math.exp(-((y - fy) / (ext[1] * .55 + 1e-6)) ** 2)
        ctx.set_source_rgba(float(col[0]), float(col[1]), float(col[2]), alpha * r.uniform(.5, 1.15) * fade); ctx.set_line_width(lw * r.uniform(.72, 1.25))
        ctx.new_path()
        for pl in scribble(r, ln, size, int(np.clip(size * .9, 4, 11))):
            if r.random() < .07: continue                # a word rubbed away
            q = pl + [x0, y]; q[:, 1] += slope * q[:, 0] + size * .05 * np.sin(q[:, 0] / (size * 9) + y)
            if box is not None and np.abs(q[:, 0].mean() - box[0]) < box[2] * 1.25 and abs(y - box[1]) < box[2] * 1.15: continue
            s = q @ M.T + o
            ctx.move_to(*s[0])
            for p in s[1:]: ctx.line_to(p[0], p[1])
        ctx.stroke()
        y += size * r.uniform(1.65, 2.15)
    if box is not None:
        ctx.set_source_rgba(float(col[0]), float(col[1]), float(col[2]), alpha)
        for pl, wgt in motif(art, r):
            if len(pl) >= 4: pl = spl(pl, n=6)
            s = (pl * box[2] + [box[0], box[1]]) @ M.T + o
            ctx.new_path(); ctx.set_line_width(lw * 1.35 * (.4 + .8 * wgt)); ctx.move_to(*s[0])
            for p in s[1:]: ctx.line_to(p[0], p[1])
            ctx.stroke()


# ------------------------------------------------------------------ near bird cel
def _norm(v): return v / (np.linalg.norm(v) + 1e-9)


def _facet_tone(n, L, trans):
    """index into TONES.  n faces the viewer (n.z<0), L points to the light (camera space)"""
    d = float(n @ L)
    if d > .30: return 0
    if d > -.02: return 1
    return 1 if trans and d < -.55 else 2 if d > -.62 else 3


def bird(cel, design, phase, R, at, span, pal='cream', seed=0, drawing=0, light=(-.45, -.78, -.42), persp=.16, lw=None,
         text=True, smear=None, rim=0.0, trans=False, ink_alpha=1.0, tone_shift=0, bounce=None):
    """near paper bird.  span = px per model unit (half wing span is about 1 unit).  smear=(dx,dy) px stretch vector"""
    d = IDX[design] if isinstance(design, str) else int(design)
    P = PAL[pal] if isinstance(pal, str) else pal
    r = np.random.default_rng(int(seed) * 131 + 7)
    rb = np.random.default_rng(int(seed) * 977 + int(drawing) * 31 + 5)
    tint = tints(seed)[0]
    col = lambda k: np.clip(rgb(P[k]) * (tint if k in ('lit', 'half') else 1 + (tint - 1) * .5), 0, 1.2)
    L = _norm(np.asarray(light, float))
    g = GEO[d, phase].astype(float)
    # each bird is folded a little differently
    asym = r.normal(0, .035, 3); side = 1 + np.sign(g[..., :1]) * asym[0]
    g = g * np.array([1 + asym[1], 1 + asym[2], 1]) * np.concatenate([side, np.ones_like(g[..., :2])], -1)
    c3 = g @ np.asarray(R).T * span
    zz = 1 + c3[..., 2] / span * persp
    v2 = c3[..., :2] / np.maximum(zz, .25)[..., None]
    nid = VIDT[d].max() + 1
    jit_s = r.normal(0, span * .012, (nid, 2)); jit_b = rb.normal(0, min(.8, span * .0035 + .22), (nid, 2))
    v2 = v2 + (jit_s + jit_b)[np.maximum(VIDT[d], 0)]
    if smear is not None:
        s = np.asarray(smear, float); m = np.hypot(*s) + 1e-6; u = s / m
        al = v2 @ u
        k = m / span
        back = np.clip(-al / span, 0, 2)
        v2 = v2 + u * (al * .25 * min(1.2, k) - back * span * .8 * min(1.5, k))[..., None] * (1 + .25 * r.normal(0, 1, (nid,))[np.maximum(VIDT[d], 0)])[..., None]
    v2 = v2 + np.asarray(at, float)
    nf = int((NVT[d] > 0).sum())
    lw = lw if lw is not None else float(np.clip(span * .0115, .7, 3.0))
    order = sorted(range(nf), key=lambda i: -c3[i, :NVT[d, i], 2].mean() + (-.02 * span if NMT[d][i][:2] in ('bu',) else 0))
    ink = col('ink')
    edges = {}

    def edge(i, a, b):
        ia, ib = int(VIDT[d, i, a]), int(VIDT[d, i, b])
        key = (min(ia, ib), max(ia, ib))
        if key not in edges:
            rr = np.random.default_rng(int(seed) * 7919 + key[0] * 101 + key[1] * 13)
            pa, pb = (v2[i, a], v2[i, b]) if ia < ib else (v2[i, b], v2[i, a])
            dv = pb - pa; ln = np.hypot(*dv) + 1e-6; nn = np.array([-dv[1], dv[0]]) / ln
            bow = (rr.normal(0, .026) + (.030 if EWT[d, i, a] >= 1 and NMT[d][i][:2] in ('wi', 'wo') and ia < ib else -.030 if EWT[d, i, a] >= 1 and NMT[d][i][:2] in ('wi', 'wo') else 0)) * ln
            deck = EWT[d, i, a] >= 1 and NMT[d][i][:3] in ('wiT', 'woT') and ln > 40
            m = int(np.clip(ln / (9 if deck else 24), 2, 22))
            t = np.linspace(0, 1, m + 1)
            off = bow * 4 * t * (1 - t)
            if deck: off = off + np.r_[0, rr.normal(0, min(1.6, span * .006), m - 1), 0]
            off = off + np.r_[0, rb.normal(0, .25, m - 1), 0]
            edges[key] = pa + np.outer(t, dv) + off[:, None] * nn
        e = edges[key]
        return e if ia < ib else e[::-1]
    nme = NMT[d]
    for i in order:
        nv = NVT[d, i]; q3 = c3[i, :nv]
        nrm = np.zeros(3)
        for k in range(nv): nrm += np.cross(q3[k] - q3[0], q3[(k + 1) % nv] - q3[0])
        nn = np.linalg.norm(nrm)
        if nn < 1e-6: continue
        nrm /= nn
        if nrm[2] > 0: nrm = -nrm
        edge_on = abs(nrm[2]) < .06
        ti = min(3, _facet_tone(nrm, L, trans) + tone_shift)
        key = TONES[ti]
        base = col(key)
        if nme[i][:2] in ('bl',) and ti < 2: base = col(TONES[ti + 1])
        pts = []
        for k in range(nv): pts.append(edge(i, k, (k + 1) % nv)[:-1])
        cu = CUT[d, i]
        curl = None
        if cu >= 0 and span > 70 and hash01(seed * 7 + i, 3) < .62 and not edge_on:
            V = v2[i, cu]; A_, B_ = v2[i, (cu - 1) % nv], v2[i, (cu + 1) % nv]
            fa, fb = .16 + .16 * hash01(seed + i, 4), .16 + .18 * hash01(seed + i, 5)
            p1, p2 = V + (A_ - V) * fa, V + (B_ - V) * fb
            mid = (p1 + p2) / 2; tip = mid + (mid - V) * (.55 + .4 * hash01(seed + i, 6)) + rb.normal(0, .4, 2)
            curl = (p1, p2, tip)
            poly = []
            for k in range(nv):
                e = edge(i, k, (k + 1) % nv)
                if k == (cu - 1) % nv: e = e[:max(2, int(len(e) * (1 - fa)) + 1)]; poly += [e[:-1], p1[None]]
                elif k == cu: poly += [p2[None], e[max(1, int(len(e) * fb) + 1):-1]]
                else: poly.append(e[:-1])
            poly = np.vstack([p for p in poly if len(p)])
        else:
            poly = np.vstack(pts)
        cel.fill(poly, base)
        area = abs(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])) / 2
        if bounce is not None and ti >= 2 and area > 350 and not edge_on:      # bounce light in the shadow facet: a third hard tone
            lo_, hi_ = poly.min(0), poly.max(0); xs = np.linspace(lo_[0] - 6, hi_[0] + 6, 7); yy = lo_[1] + (hi_[1] - lo_[1]) * .64 + (xs - xs.mean()) * .28 + (hi_[1] - lo_[1]) * .02 * np.sin(xs * .07 + i)
            cel.shade(np.vstack([np.c_[xs, yy], [[xs[-1], hi_[1] + 30], [xs[0], hi_[1] + 30]]]), mix(base, rgb(bounce), .46), within=poly)
        # cast shadow wedge of the body on the wing root (hard edge)
        if nme[i][:2] == 'wi' and span > 40 and ti < 2:
            a, b = v2[i, 0], v2[i, nv - 1]; o1, o2 = v2[i, 1], v2[i, 2]
            f1, f2 = (.26, .10) if nme[i][2] == 'T' else (.10, .24)
            if nme[i][2] == 'L': w_ = np.array([a, a + (o1 - a) * .07, b + (o2 - b) * .22, b])
            else: w_ = np.array([a, a + (o1 - a) * .22, b + (o2 - b) * .36, b])
            cel.shade(w_, col(TONES[min(3, ti + 1)]), within=poly)
        on = poly[:, 0].max() > cel.ox and poly[:, 0].min() < cel.ox + cel.w and poly[:, 1].max() > cel.oy and poly[:, 1].min() < cel.oy + cel.h
        if text and TXT[d, i] and area > 900 and not edge_on and on:
            rr = np.random.default_rng(int(seed) * 53 + i)
            e0 = _norm(q3[1] - q3[0]); e1 = _norm(np.cross(nrm, e0))
            an = (hash01(seed, 77) * 6.28 + (0 if nme[i][:2] == 'wi' else .9) + (1.2 if nme[i][2] == 'T' else 0))
            ux = e0 * math.cos(an) + e1 * math.sin(an); uy = -e0 * math.sin(an) + e1 * math.cos(an)
            M = np.array([[ux[0], uy[0]], [ux[1], uy[1]]]) / max(.3, zz[i, :nv].mean())
            if abs(np.linalg.det(M)) > .12:
                o = poly.mean(0); ext = np.abs((poly - o) @ np.linalg.inv(M).T).max(0)
                size = span * (.034 + .010 * hash01(seed, 78))
                cel.clip(poly)
                art = None
                if (nme[i][:3] == 'wiL' or nme[i][:3] == 'wiT') and design in ('engraved', IDX['engraved']): art = int(hash01(seed, 9) * 5) if nme[i].endswith('R') == (hash01(seed, 10) < .5) and nme[i][:3] == 'wiL' else None
                tc = col('text') * .72 + base * .28
                _affine_rows(cel.ctx, M, o, np.minimum(ext, span * 1.3), rr, size, tc, .62, max(.55, span * .0042), art)
                cel.unclip()
        # secondary crease folded into the paper, drawn as a tone step
        if nme[i][:3] in ('wiL', 'woT', 'wiT') and area > 2500 and hash01(seed + i, 21) < .6:
            a = edge(i, 0, 1); b = edge(i, 2, 3 % nv)
            p, q = a[int(len(a) * (.3 + .4 * hash01(seed + i, 22)))], b[int(len(b) * (.3 + .4 * hash01(seed + i, 23)))]
            cel.line([p, (p + q) / 2 + rb.normal(0, .5, 2), q], lw * .42, mix(base, ink, .42), (.3, .5), seed=seed + i, alpha=.8)
        if rim > 0:
            for k in range(nv):
                if EWT[d, i, k] < 1: continue
                e = edge(i, k, (k + 1) % nv); dv = e[-1] - e[0]; ln = np.hypot(*dv)
                if ln < 6: continue
                nr = np.array([dv[1], -dv[0]]) / ln; cen = poly.mean(0)
                if (e.mean(0) - cen) @ nr < 0: nr = -nr
                if nr @ L[:2] > .25 * np.hypot(*L[:2]):
                    cel.line(e - nr * lw * .9, lw * 1.9 * rim, col('rim'), (.15, .3), seed=seed + i * 7 + k, press=.35)
        for k in range(nv):
            wgt = EWT[d, i, k]
            if wgt <= 0: continue
            e = edge(i, k, (k + 1) % nv)
            if np.hypot(*(e[-1] - e[0])) < 2.5: continue
            if curl is not None and k in (cu, (cu - 1) % nv):
                fa, fb = .16 + .16 * hash01(seed + i, 4), .16 + .18 * hash01(seed + i, 5)
                e = e[:max(2, int(len(e) * (1 - fa)) + 1)] if k == (cu - 1) % nv else e[max(0, int(len(e) * fb)):]
            if len(e) < 2: continue
            c_ = ink if wgt > .6 else mix(base, ink, .55)
            # pencil lifts: long outlines are sometimes broken
            if wgt >= 1 and hash01(seed * 3 + i * 17 + k, drawing // 3) < .18 and len(e) > 5:
                h_ = int(len(e) * (.45 + .25 * hash01(seed + k, 8)))
                cel.line(e[:h_ + 1], lw * wgt, c_, (.12, .35), seed=seed + i * 11 + k, alpha=ink_alpha); e = e[min(len(e) - 2, h_ + 1):]
            cel.line(e, lw * (wgt if wgt >= .8 else wgt * .9), c_, (.14, .22), seed=seed + i * 13 + k + drawing, alpha=ink_alpha)
        if curl is not None:
            p1, p2, tip = curl
            ct = TONES[2 if ti < 2 else max(0, ti - 2)]
            cp = spl([p1, (p1 + tip) / 2 + (tip - p2) * .08, tip, (p2 + tip) / 2 + (tip - p1) * .08, p2], n=5)
            cel.fill(cp, col(ct)); cel.line(cp, lw * .8, ink, (.1, .1), seed=seed + i, alpha=ink_alpha)
            cel.line([p1, p2], lw * .5, mix(base, ink, .5), seed=seed + i + 3)
    return cel


# ------------------------------------------------------------------ near page cel
def _clip_s(poly, ax, smin, smax):
    def half(P, sgn, lim):
        out = []
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            da, db = sgn * (a @ ax - lim), sgn * (b @ ax - lim)
            if da >= 0: out.append(a)
            if (da >= 0) != (db >= 0): out.append(a + (b - a) * da / (da - db))
        return np.array(out) if out else np.zeros((0, 2))
    P = half(poly, 1, smin)
    return half(P, -1, smax) if len(P) else P


def _bender(beta, k0, k1, k2, L):
    s = np.linspace(-L, L, 81); u = s / L
    ph = k0 + k1 * u + k2 * u * u
    X = np.cumsum(np.cos(ph)) * (s[1] - s[0]); Z = np.cumsum(np.sin(ph)) * (s[1] - s[0])
    X -= np.interp(0, s, X); Z -= np.interp(0, s, Z)
    ax = np.array([math.cos(beta), math.sin(beta)]); pp = np.array([-ax[1], ax[0]])

    def f(P):
        P = np.asarray(P, float); a = P @ ax; b = P @ pp
        x = np.interp(a, s, X); z = np.interp(a, s, Z)
        return np.c_[np.outer(x, ax) + np.outer(b, pp), z]
    return f, ax, lambda a: np.interp(a, s, ph)


def page(cel, R, at, size, pal='cream', seed=0, drawing=0, bend=.8, light=(-.45, -.78, -.42), persp=.22, lw=None, text=True,
         smear=None, beta=None, scurve=None, rim=0.0, trans=False, torn=None, aspect=.74, ink_alpha=1.0, art=None):
    """one curled, deckled, handwritten sheet.  size = height px.  bend ~0 flat .. 1.6 rolled"""
    P = PAL[pal] if isinstance(pal, str) else pal
    r = np.random.default_rng(int(seed) * 17 + 3); rb = np.random.default_rng(int(seed) * 331 + int(drawing) * 7)
    tint = tints(seed)[0]
    col = lambda k: np.clip(rgb(P[k]) * (tint if k in ('lit', 'half') else 1 + (tint - 1) * .5), 0, 1.2)
    L = _norm(np.asarray(light, float))
    beta = r.choice([r.normal(1.57, .25), r.normal(.0, .25), r.normal(.75, .2), r.normal(2.4, .2)]) if beta is None else beta
    sc = r.normal(0, .6) if scurve is None else scurve
    hw, hh = aspect / 2, .5
    f, ax, ph = _bender(beta, rb.normal(0, .06), bend * 1.35, sc * bend, .64)
    # border with deckle, a torn edge or a missing corner
    c = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    if torn is None: torn = r.random() < .45
    dog = r.integers(0, 4) if r.random() < .5 else -1
    bd = []
    for k in range(4):
        a, b = c[k], c[(k + 1) % 4]
        m = 12
        t = np.linspace(0, 1, m, endpoint=False)
        e = a + np.outer(t, b - a); nn = np.array([(b - a)[1], -(b - a)[0]]); nn = nn / np.hypot(*nn)
        j = r.normal(0, .004, m); j[0] = 0
        if torn and k == 3: j = -np.abs(r.normal(0, .016, m)) - .01 * np.sin(t * 9 + r.uniform(0, 6))
        bd.append(e + nn * j[:, None])
    if dog >= 0:
        cut = .10 + .10 * r.random()
        bd[dog][0] = c[dog] + (c[(dog + 1) % 4] - c[dog]) * cut
        bd[(dog - 1) % 4] = np.vstack([bd[(dog - 1) % 4], c[dog] + (c[(dog - 1) % 4] - c[dog]) * cut])
    border = np.vstack(bd)
    Rm = np.asarray(R)

    def proj(P2, jj=0.0):
        p3 = f(P2) @ Rm.T
        z = 1 + p3[:, 2] * persp * 1.6
        q = p3[:, :2] * size / np.maximum(z, .25)[:, None]
        if smear is not None:
            s = np.asarray(smear, float); m = np.hypot(*s) + 1e-6; u = s / m
            al = q @ u; kk = m / size
            q = q + u * (al * .2 * min(1.2, kk) - np.clip(-al, 0, None) * .9 * min(1.6, kk))[:, None]
        return q + np.asarray(at, float), p3
    S = np.linspace(-.66, .66, 15)
    tones = []
    for a, b in zip(S[:-1], S[1:]):
        an = ph((a + b) / 2)
        n2 = np.array([-math.sin(an) * ax[0], -math.sin(an) * ax[1], math.cos(an)])
        n = Rm @ n2
        front = n[2] < 0
        if not front: n = -n
        tones.append((_facet_tone(n, L, trans), front))
    runs = []; i0 = 0
    for i in range(1, len(tones) + 1):
        if i == len(tones) or tones[i] != tones[i0]:
            runs.append((S[i0], S[i], *tones[i0])); i0 = i
    lw = lw if lw is not None else float(np.clip(size * .010, .7, 3.0))
    draws = []
    for a, b, ti, fr in runs:
        poly = _clip_s(border, ax, a - 1e-4, b + .006)
        if len(poly) < 3: continue
        # densify across the bend
        dens = []
        for k in range(len(poly)):
            p, q = poly[k], poly[(k + 1) % len(poly)]
            m = max(1, int(abs((q - p) @ ax) / .05))
            dens.append(p + np.outer(np.linspace(0, 1, m, endpoint=False), q - p))
        poly = np.vstack(dens)
        s2, p3 = proj(poly)
        draws.append((p3[:, 2].mean(), s2, ti, fr, a, b))
    draws.sort(key=lambda q: -q[0])
    ink = col('ink')
    rows = []
    if text and size > 60:
        rr = np.random.default_rng(int(seed) * 71 + 1)
        sz = .026 + .010 * rr.random(); y = -hh + .08
        artk = art if art is not None else (int(rr.integers(0, 5)) if rr.random() < .22 else None)
        bx = None
        if artk is not None:
            bs = .16 + .06 * rr.random(); bx = (rr.uniform(-.08, .08), rr.uniform(-.2, .1), bs)
        while y < hh - .06:
            L_ = aspect - .14; rwt = int(rr.choice(3, p=[.36, .40, .24]))
            for pl in scribble(rr, L_ * (rr.uniform(.25, .75) if rr.random() < .30 else 1), sz, int(np.clip(size * sz * .9, 4, 11))):
                q = pl + [-hw + .07, y]
                if bx and abs(q[:, 0].mean() - bx[0]) < bx[2] * 1.3 and abs(y - bx[1]) < bx[2] * 1.1: continue
                if rr.random() < .06: continue
                rows.append((q, 1.0, rwt))
            y += sz * rr.uniform(1.65, 2.1) + (sz * 1.4 if rr.random() < .10 else 0)
        if bx:
            for pl, wgt in motif(artk, rr): rows.append(((spl(pl, n=4) if len(pl) >= 4 else pl) * bx[2] + [bx[0], bx[1]], 1.4 * wgt, 3))
    sides = []
    for k in range(4):
        e = np.vstack([bd[k], bd[(k + 1) % 4][:1]])
        m = np.vstack([e[i] + np.outer(np.linspace(0, 1, 4, endpoint=False), e[i + 1] - e[i]) for i in range(len(e) - 1)] + [e[-1:]])
        q, _ = proj(m); q = q + rb.normal(0, .16, q.shape)
        cen = proj(border)[0].mean(0)
        sides.append((m @ ax, q, cen))
    stains = []
    if size > 110:
        rs = np.random.default_rng(int(seed) * 19 + 8)
        for _k in range(int(rs.integers(2, 9))):
            c0 = rs.uniform(-.42, .42, 2) * [aspect, 1]; rad = rs.uniform(.004, .016) if rs.random() < .75 else rs.uniform(.03, .08)
            th = np.linspace(0, 6.283, 9, endpoint=False)
            stains.append(c0 + np.c_[np.cos(th), np.sin(th)] * (rad * rs.uniform(.6, 1.3, 9))[:, None])
    for _, s2, ti, fr, a, b in draws:
        base = col(TONES[ti])
        if not fr: base = np.clip(base * np.array([.985, .965, .95], f32), 0, 1.2)
        cel.fill(s2, base)
        if stains and fr:
            cel.clip(s2)
            for st in stains:
                sa = st @ ax
                if sa.max() > a - .02 and sa.min() < b + .02: cel.fill(spl(proj(st)[0], closed=True, n=4), base * np.array([.955, .915, .86], f32), .8)
            cel.unclip()
        if rows:
            cel.clip(s2)
            cx = cel.ctx; tc = col('text') * .7 + base * .3 if fr else col('text') * .25 + base * .75
            cx.set_source_rgba(float(tc[0]), float(tc[1]), float(tc[2]), .66); cx.set_line_cap(1); cx.set_line_join(1)
            for lwk in (0, 1, 2, 3):
                cx.new_path(); any_ = False
                for q, wgt, cl in rows:
                    if cl != lwk: continue
                    sa = q @ ax
                    if sa.max() < a or sa.min() > b: continue
                    if lwk == 3 and len(q) < 8:
                        q = np.vstack([q[k] + np.outer(np.linspace(0, 1, 5, endpoint=False), q[k + 1] - q[k]) for k in range(len(q) - 1)] + [q[-1:]])
                    s_, _p = proj(q); any_ = True
                    cx.move_to(*s_[0])
                    for p in s_[1:]: cx.line_to(p[0], p[1])
                if any_:
                    cx.set_source_rgba(float(tc[0]), float(tc[1]), float(tc[2]), (.72, .46, .26, .62)[lwk]); cx.set_line_width(max(.5, size * (.0036, .0031, .0028, .0046)[lwk])); cx.stroke()
            cel.unclip()
        for k, (sv, q, cen) in enumerate(sides):
            if size < 34: continue
            inr = (sv >= a - 1e-3) & (sv <= b + 8e-3)
            if inr.sum() < 2: continue
            idx = np.nonzero(inr)[0]
            for grp in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
                g0, g1 = grp[0], min(len(q) - 1, grp[-1] + 1)
                e = q[g0:g1 + 1]
                if len(e) < 3: continue
                if rim > 0:
                    dv = e[-1] - e[0]; nr = np.array([dv[1], -dv[0]]); nr = nr / (np.hypot(*nr) + 1e-6)
                    if (e.mean(0) - cen) @ nr < 0: nr = -nr
                    if nr @ L[:2] > .2 * np.hypot(*L[:2]): cel.line(e - nr * lw * .8, lw * 1.8 * rim, col('rim'), (.2, .2), seed=seed + k)
                cel.line(e, lw * (1.0 if k != 3 or not torn else .75), ink, (.16 if g0 == 0 else 0, .2 if g1 == len(q) - 1 else 0), seed=seed + k * 3 + drawing, press=.3, alpha=ink_alpha)
        # silhouette ruling where the sheet turns over
    if dog >= 0 and size > 70:
        a, b = bd[dog][0], bd[(dog - 1) % 4][-1]; m = (a + b) / 2; tip = m + (m - c[dog]) * .8
        s2, _ = proj(np.array([a, tip * .96, b])); t2 = 2
        cel.fill(s2, col(TONES[t2])); cel.line(s2, lw * .8, ink, (.1, .1), seed=seed + 41, alpha=ink_alpha)
        cel.line(s2[[0, 2]], lw * .5, mix(col('half'), ink, .5), seed=seed + 43)
    return cel


# ------------------------------------------------------------------ a sheet folding itself (hand drawn keys)
# unit = page height, x right, y down.  each facet: points, tone, handwriting angle (None = blank), flags ('c' cast shadow: no ink)
_RAG = [(-.350, .43), (-.366, .34), (-.345, .26), (-.364, .15), (-.342, .06), (-.362, -.05), (-.348, -.15), (-.368, -.26), (-.350, -.37), (-.364, -.45)]     # the torn gutter edge
FOLD = {
    'taut': [([(-.37, -.50), (-.02, -.485), (.37, -.50), (.365, .0), (.37, .50), (.0, .49), (-.37, .50)] + _RAG, 0, 0.0, ''),
             ([(.17, -.50), (.37, -.50), (.365, .0), (.37, .50), (.27, .495), (.23, .05)], 1, None, 'c'),
             ([(-.37, .33), (-.22, .50), (-.37, .50)], 1, None, 'c')],
    'crease': [([(-.37, -.50), (.0, -.15), (.37, .24), (.37, .50), (.0, .49), (-.37, .50)] + _RAG, 0, 0.0, ''),
               ([(-.34, -.44), (.35, .23), (.10, .30), (-.13, .07)], 2, None, 'c'),
               ([(-.37, -.50), (-.12, -.36), (.14, -.30), (.28, -.05), (.37, .24), (.0, -.15)], 1, .8, 'b'),
               ([(-.12, -.36), (.14, -.30), (.28, -.05), (.06, -.17)], 2, None, 'c')],
    'half': [([(-.37, .22), (.37, .25), (.33, .37), (-.34, .35)], 2, None, ''),
             ([(-.375, -.50), (.0, -.13), (.385, .255), (.0, .25), (-.375, .25), (-.37, -.1)], 0, .0, ''),
             ([(-.36, -.46), (-.37, .24), (-.08, .24), (-.17, .0)], 1, None, 'c'),
             ([(-.38, -.50), (-.36, -.50), (.385, .25), (.36, .262)], 3, None, 'c')],
    'kite': [([(0, -.62), (-.22, .02), (-.01, .06)], 0, 1.25, ''), ([(0, -.62), (.23, -.02), (-.01, .06)], 1, 1.9, ''),
             ([(-.22, .02), (.03, .62), (-.01, .06)], 1, .5, ''), ([(.23, -.02), (.03, .62), (-.01, .06)], 2, None, ''),
             ([(0, -.62), (-.075, -.40), (.0, -.31), (.085, -.385)], 3, None, ''), ([(-.22, .02), (-.10, .02), (.0, .33)], 0, None, 'c')],
    'petal': [([(.16, -.12), (.46, -.26), (.30, .05), (.08, .24)], 0, .2, ''), ([(.16, -.12), (.36, -.13), (.08, .24)], 1, None, 'c'),
              ([(.04, -.76), (-.16, -.04), (.0, .06)], 0, 1.3, ''), ([(.04, -.76), (.15, -.10), (.0, .06)], 2, None, ''),
              ([(-.16, -.04), (-.02, .58), (.0, .06)], 1, .6, ''), ([(.15, -.10), (-.02, .58), (.0, .06)], 3, None, ''),
              ([(.04, -.76), (-.05, -.40), (.02, -.26), (.09, -.40)], 1, None, '')],
}
FOLDS = list(FOLD)


def fold(cel, key, at, size, rot=0.0, pal='cream', seed=0, drawing=0, flip=False, shear=0.0, lw=None, text=True, warm=None):
    """one hand drawn folding stage: 'taut' 'crease' 'half' 'kite' 'petal'.  size = page height px"""
    P = PAL[pal] if isinstance(pal, str) else pal
    rb = np.random.default_rng(int(seed) * 71 + int(drawing) * 13 + 2)
    tint = tints(seed)[0]
    col = lambda k: np.clip(rgb(P[k]) * (tint if k in ('lit', 'half') else 1 + (tint - 1) * .5), 0, 1.2)
    lw = lw if lw is not None else float(np.clip(size * .0062, .6, 3.2))
    c, s_ = math.cos(rot), math.sin(rot)
    M = np.array([[c, -s_], [s_, c]]) @ np.array([[(-1 if flip else 1), shear], [0, 1]]) * size
    ink = col('ink'); at = np.asarray(at, float); last = None
    for i, (pts, ti, ta, fl) in enumerate(FOLD[key]):
        q = np.array(pts, float); q = q + rb.normal(0, min(.004, .8 / size), q.shape)
        big = len(q) > 4
        # every paper edge bows a hair
        out = []
        for k in range(len(q)):
            a, b = q[k], q[(k + 1) % len(q)]; d = b - a; ln = np.hypot(*d); n = np.array([-d[1], d[0]]) / (ln + 1e-9)
            h = (hash01(seed * 5 + i * 11 + k, 3) - .5) * .05 * ln
            t = np.linspace(0, 1, max(3, int(ln * size / 16)), endpoint=False)
            out.append(a + np.outer(t, d) + np.outer(h * 4 * t * (1 - t) + (rb.normal(0, .0012, len(t)) if ln > .2 else 0), n))
        poly = np.vstack(out) @ M.T + at
        base = col(TONES[ti])
        if 'c' in fl and last is not None:       # a tone step on the facet below: stay inside it, then re-ink it
            cel.shade(poly, base, within=last[0])
            for ln_, w_, sd_ in last[1]: cel.line(ln_, w_, ink, (.10, .14), seed=sd_, press=.28)
            continue
        if 'b' in fl: base = np.clip(base * np.array([.99, .965, .94], f32), 0, 1.2)
        cel.fill(poly, base)
        if warm is not None and ti >= 2: cel.fill(poly, rgb(warm[0]), warm[1])
        if i == 0 and size > 200 and key in ('taut', 'crease', 'half'):       # foxing and a tea ring
            rs = np.random.default_rng(int(seed) * 19 + 8); cel.clip(poly)
            for _k in range(9):
                c0 = rs.uniform(-.42, .42, 2) * [.74, 1]; rad = rs.uniform(.004, .013) if _k > 1 else rs.uniform(.035, .07); th = np.linspace(0, 6.283, 9, endpoint=False)
                cel.fill(spl((c0 + np.c_[np.cos(th), np.sin(th)] * (rad * rs.uniform(.7, 1.25, 9))[:, None]) @ M.T + at, closed=True, n=5), base * np.array([.965, .925, .86], f32), .8)
            cel.unclip()
        if text and ta is not None and size > 90 and 'c' not in fl:
            rr = np.random.default_rng(int(seed) * 31 + i)
            ca, sa = math.cos(ta), math.sin(ta); A = M @ np.array([[ca, -sa], [sa, ca]]) / size
            o = poly.mean(0); ext = np.abs((poly - o) @ np.linalg.inv(A).T).max(0)
            cel.clip(poly)
            tc = col('text') * .74 + base * .26
            _affine_rows(cel.ctx, A, o, ext * .94, rr, size * .030, tc, .62 if 'b' not in fl else .25, max(.55, size * .0030), 2 if key == 'taut' else None)
            cel.unclip()
        k0 = 0; inked = []
        for e in out:
            m = len(e); nxt = out[(k0 + 1) % len(out)][:1]
            ln = np.vstack([e, nxt]) @ M.T + at
            inked.append((ln, lw * (1.0 + .25 * (hash01(seed + i * 3 + k0, 8) - .5)), seed + i * 13 + k0 + drawing))
            cel.line(ln, inked[-1][1], ink, (.10, .14), seed=inked[-1][2], press=.28)
            k0 += 1
        last = (poly, inked)
    return cel


def put_fold(img, key, at, size, opacity=1.0, **kw):
    return _put(img, at, size * .95 + 8, lambda c: fold(c, key, at, size, **kw), opacity)


# ------------------------------------------------------------------ origami: the page really folded
class Sheet:
    """flat folded page.  facets = [uv polygon (k,2), edge flags (1 page border, 0 crease), A 3x3 with xyz = A @ (u, v, 1)].
    page units: height 1, x right, y down, z away from the viewer."""
    def __init__(s, seed=0, aspect=.74):
        r = np.random.default_rng(seed * 13 + 5); hw, hh = aspect / 2, .5
        c = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]]); P = []
        for k in range(4):
            a, b = c[k], c[(k + 1) % 4]; m = 23; t = np.linspace(0, 1, m, endpoint=False); n = np.array([(b - a)[1], -(b - a)[0]]); n = n / np.hypot(*n)
            j = np.convolve(r.normal(0, .0115, m), [.25, .5, .25], 'same') + r.normal(0, .0026, m)
            if k == 3: j = -np.abs(r.normal(0, .017, m)) - .007 * np.sin(t * 11 + 1)      # torn gutter
            j[0] = 0; P.append(a + np.outer(t, b - a) + n * j[:, None])
        P = np.vstack(P); A = np.array([[1., 0, 0], [0, 1, 0], [0, 0, 0]])
        s.F = [(P, np.ones(len(P), int), A)]

    def xyz(s, uv, A): return np.c_[uv, np.ones(len(uv))] @ A.T

    @staticmethod
    def _cut(uv, fl, d, sgn):
        o, g = [], []
        n = len(uv)
        for k in range(n):
            a, b, da, db = uv[k], uv[(k + 1) % n], d[k] * sgn, d[(k + 1) % n] * sgn
            if da >= 0: o.append(a); g.append(fl[k])
            if (da >= 0) != (db >= 0):
                o.append(a + (b - a) * da / (da - db)); g.append(fl[k] if da < 0 else 0)
        return np.array(o).reshape(-1, 2), np.array(g, int)

    def fold(s, p, q, pt, ang, lift=-1):
        """crease along p-q (current xy), move the side that contains pt by ang degrees.  lift -1 toward the viewer, +1 behind"""
        p, q, pt = np.array(p, float), np.array(q, float), np.array(pt, float)
        dr = q - p; n = np.array([-dr[1], dr[0]]); sg = np.sign((pt - p) @ n)
        ax = np.r_[dr / np.hypot(*dr), 0]; c = math.radians(ang)
        # choose the rotation sense that lifts the moving side the asked way
        test = np.cross(ax, np.r_[pt - p, 0])[2]; c *= lift * np.sign(test)
        co, si = math.cos(c), math.sin(c); K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Rm = np.eye(3) * co + si * K + (1 - co) * np.outer(ax, ax); o = np.r_[p, 0]
        out = []
        for uv, fl, A in s.F:
            X = s.xyz(uv, A); d = (X[:, :2] - p) @ n * sg
            if (d <= 1e-9).all(): out.append((uv, fl, A)); continue
            if (d >= -1e-9).all(): mv = [(uv, fl)]; st = None
            else: mv = [s._cut(uv, fl, d, 1)]; st = s._cut(uv, fl, d, -1)
            if st is not None and len(st[0]) >= 3: out.append((st[0], st[1], A))
            T = np.c_[np.eye(3), -o]
            for u2, f2 in mv:
                if len(u2) < 3: continue
                A2 = Rm @ A; A2[:, 2] = Rm @ (A[:, 2] - o) + o
                out.append((u2, f2, A2))
        s.F = out
        return s


def origami_stage(stage, seed=0, t=1.0):
    """'crease' 'half' 'kite' 'petal' 'blade' -> Sheet.  t trims the last motion (0..1) for in-betweens"""
    S = Sheet(seed)
    A, B_, C, R_, M_ = (-.37, -.5), (.37, -.5), (.37, .24), (-.37, .24), (0., -.13)
    if stage == 'crease': return S.fold(A, C, B_, 112 * t)
    S.fold(A, C, B_, 176)
    if stage == 'half': return S.fold((-.5, .24), (.5, .24), (0, .5), 140 * t, +1)
    S.fold((-.5, .24), (.5, .24), (0, .5), 177, +1)
    def ray(deg): a = math.radians(deg); return (R_[0] + math.cos(a), R_[1] + math.sin(a))
    fl = dict(kite=(166, 158), petal=(160, 62), blade=(58, 66))[stage]
    S.fold(R_, ray(-22.5), (.37, .24), fl[0] * t if stage == 'kite' else fl[0]); S.fold(R_, ray(-67.5), (-.37, -.5), fl[1])
    if stage == 'kite': return S
    S.fold(R_, M_, (-.3, -.3), 52 * t if stage == 'petal' else 158, +1)
    if stage == 'petal': return S
    # neck: a reverse fold near the narrow tip
    u = np.array([-.7071, .7071]); w = np.array([.7071, .7071]); p = np.array(M_) + u * .24
    d = u * math.cos(.62) - w * math.sin(.62)
    return S.fold(p, p + d, np.array(R_), 172 * t)


def page_ink(seed, art=2):
    """the writing of one page in page units: [(polyline, weight, alpha, closed engraving?)].  broken, fading, uneven rows and one small engraving"""
    r = np.random.default_rng(seed * 71 + 9); W_ = []
    bx = np.array([r.uniform(.06, .14), r.uniform(-.17, -.05)]); bs = .105
    y = -.425; fx, fy = r.uniform(-.3, .3), r.uniform(.05, .42); row = 0
    while y < .45:
        sz = .030 if row else .046
        x0 = -.295 + (r.uniform(.03, .10) if r.random() < .25 or row == 0 else r.normal(0, .004)); ln = r.uniform(.18, .34) if row == 0 else .60 * (r.uniform(.25, .7) if r.random() < .30 else r.uniform(.9, 1.0))
        ra = r.uniform(.45, 1.0); sl = r.normal(0, .012)
        for pl in scribble(r, ln, sz, 9):
            q = pl + [x0, y]; q[:, 1] += sl * q[:, 0] + .0035 * np.sin(q[:, 0] * 14 + y * 9); c = q.mean(0)
            if abs(c[0] - bx[0]) < bs * 1.15 and abs(c[1] - bx[1]) < bs * 1.15: continue
            fade = 1 - .82 * math.exp(-(((c[0] - fx) / .30) ** 2 + ((c[1] - fy) / .20) ** 2))
            if fade * ra < .20 or r.random() < .05: continue
            W_.append((q, r.uniform(.75, 1.25) * (1.25 if row == 0 else 1), ra * fade, 0))
            if r.random() < .025: W_.append((np.array([[q[0, 0], y - sz * .18], [c[0], y - sz * .24], [q[-1, 0], y - sz * .16]]), .7, ra * .8, 0))        # struck out
        row += 1; y += sz * r.uniform(1.45, 1.95) + (sz * r.uniform(.8, 1.6) if r.random() < .13 else 0)
    for k in range(3):                                   # a note squeezed into the margin
        for pl in scribble(r, .16, .019, 7): q = pl @ np.array([[.05, -1], [1, .05]]) + [.335 + k * .016, .30]; W_.append((q, .55, .45, 0))
    for pl, wgt in motif(art, r): W_.append(((spl(pl, n=5) if len(pl) >= 4 else pl) * bs + bx, wgt * .95, .8, 1))
    for k in range(9):                                   # engraver's hatching under the motif
        x = bx[0] - bs * .6 + k * bs * .15; W_.append((np.array([[x, bx[1] + bs * .95], [x + .008, bx[1] + bs * (.98 + .05 * math.sin(k))], [x + .03, bx[1] + bs * 1.02]]), .4, .6, 1))
    return W_


def origami(cel, S, at, size, R, pal='cream', seed=0, drawing=0, light=(-.5, -.75, -.42), persp=.30, lw=None, ink_rows=None, curl=(.10, -.07, .05),
            bounce=None, rim=0.8, ink_alpha=1.0):
    """draw a folded Sheet: hard lit / half / shade facets, cast shadows between layers, bounce light in the shadow facets, writing that follows the folds.
    R: 3x3 camera-space rotation (columns right, down, away)"""
    P = PAL[pal] if isinstance(pal, str) else pal
    rb = np.random.default_rng(int(seed) * 977 + int(drawing) * 31 + 5)
    tint = tints(seed)[0]; col = lambda k: np.clip(rgb(P[k]) * (tint if k in ('lit', 'half') else 1 + (tint - 1) * .5), 0, 1.2)
    L = _norm(np.asarray(light, float)); ink = col('ink'); at = np.asarray(at, float); R = np.asarray(R, float)
    lw = lw if lw is not None else float(np.clip(size * .0068, .6, 3.0))
    cen = np.mean([S.xyz(uv, A).mean(0) for uv, _, A in S.F], 0)

    def warp(X):
        X = X - cen
        z = X[:, 2] + curl[0] * X[:, 0] ** 2 * 2 + curl[1] * X[:, 1] ** 2 * 2 + curl[2] * X[:, 0] * X[:, 1] * 2
        return np.c_[X[:, :2], z] @ R.T

    def scr(V):
        k = 1 + V[:, 2] * persp * 2
        return V[:, :2] * size / np.maximum(k, .3)[:, None] + at
    fac = []
    for i, (uv, fl, A) in enumerate(S.F):
        seg = []
        for k in range(len(uv)):
            a, b = uv[k], uv[(k + 1) % len(uv)]; d = b - a; ln = np.hypot(*d); m = max(1, int(ln * size / 11)); t = np.linspace(0, 1, m, endpoint=False)
            off = 0 if fl[k] else (hash01(seed * 5 + i * 11 + k, 3) - .5) * .022 * ln * 4 * t * (1 - t)
            seg.append((a + np.outer(t, d) + np.outer(off * np.ones(m), [-d[1], d[0]]) / (ln + 1e-9), fl[k]))
        dense = np.vstack([q for q, _ in seg]); V = warp(S.xyz(dense, A))
        n = np.zeros(3); q0 = V.mean(0)
        for k in range(len(V)): n += np.cross(V[k] - q0, V[(k + 1) % len(V)] - q0)
        n = _norm(n); uvn = R @ np.cross(A[:, 0], A[:, 1]); front = uvn[2] > 0
        nv = n if n[2] < 0 else -n
        fac.append(dict(i=i, uv=dense, A=A, V=V, s=scr(V) + rb.normal(0, .22, (len(V), 2)), n=nv, front=front, seg=[(len(q), f_) for q, f_ in seg], z=V[:, 2].mean()))
    # a hair of layer order so stacked facets sort the way they were folded
    fac.sort(key=lambda f: -(f['z'] - .004 * f['i']))
    for j, f in enumerate(fac):
        ti = _facet_tone(f['n'], L, False); base = col(TONES[ti])
        if not f['front']: base = np.clip(base * np.array([.985, .955, .925], f32), 0, 1.2)
        poly = f['s']; cel.fill(poly, base)
        ar = abs(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])) / 2
        ext = poly.max(0) - poly.min(0)
        if ar > 500 and abs(f['n'][2]) > .08:
            c = poly.mean(0); lo, hi = poly.min(0), poly.max(0)
            def band(y0, tilt, sgn, cc, al=1.0):
                xs = np.linspace(lo[0] - 8, hi[0] + 8, 8); yy = y0 + (xs - c[0]) * tilt + ext[1] * .018 * np.sin(xs / max(ext[0], 1) * 9 + seed + f['i'])
                far = hi[1] + 30 if sgn > 0 else lo[1] - 30
                cel.shade(np.vstack([np.c_[xs, yy], [[xs[-1], far], [xs[0], far]]]), cc, within=poly, alpha=al)
            if ti == 0: band(lo[1] + ext[1] * (.72 + .12 * hash01(seed + f['i'], 5)), .34 * (hash01(seed + f['i'], 6) - .35) * 2, 1, col('half'))
            elif ti == 1: band(lo[1] + ext[1] * (.70 + .14 * hash01(seed + f['i'], 5)), -.30, 1, mix(col('half'), col('shade'), .55))
            else:
                bc = rgb('#b99a9a') if bounce is None else rgb(bounce)
                band(lo[1] + ext[1] * (.50 + .14 * hash01(seed + f['i'], 5)), .30, 1, mix(base, bc, .46))
        # cast shadows of the layers in front
        for g in fac[j + 1:]:
            if g['z'] > f['z'] - .012 or ti > 1: continue
            den = f['n'] @ L
            if abs(den) < .15: continue
            tt = ((f['V'].mean(0) - g['V']) @ f['n']) / den
            if tt.max() < .01: continue
            Q = scr(g['V'] - 0 + L * np.clip(tt, None, .9)[:, None] * 1.0)
            cel.shade(Q, col('shade') if ti == 0 else mix(col('shade'), col('half'), .25), within=poly, alpha=1.0)
        # foxing
        if f['front'] and size > 200:
            rs = np.random.default_rng(int(seed) * 19 + 8); cel.clip(poly)
            for _k in range(10):
                c0 = rs.uniform(-.44, .44, 2) * [.74, 1]; rad = rs.uniform(.004, .012) if _k > 1 else rs.uniform(.03, .06); th = np.linspace(0, 6.283, 9, endpoint=False)
                uvq = c0 + np.c_[np.cos(th), np.sin(th)] * (rad * rs.uniform(.7, 1.25, 9))[:, None]
                cel.fill(spl(scr(warp(S.xyz(uvq, f['A']))), closed=True, n=5), base * np.array([.965, .92, .85], f32), .55)
            cel.unclip()
        if ink_rows is not None and size > 130 and abs(f['n'][2]) > .12:
            lo, hi = f['uv'].min(0), f['uv'].max(0); cel.clip(poly); cx = cel.ctx
            tc = col('text') * .80 + base * .20 if f['front'] else col('text') * .30 + base * .70
            fine = size > 330
            for q, wgt, al, art in ink_rows:
                if q[:, 0].max() < lo[0] or q[:, 0].min() > hi[0] or q[:, 1].max() < lo[1] or q[:, 1].min() > hi[1]: continue
                sq = scr(warp(S.xyz(q if not art or len(q) > 5 else np.vstack([q[k] + np.outer(np.linspace(0, 1, 4, endpoint=False), q[k + 1] - q[k]) for k in range(len(q) - 1)] + [q[-1:]]), f['A'])))
                a_ = float(np.clip(al * (.95 if f['front'] else .58), 0, 1)); w_ = max(.5, size * .0040 * wgt)
                if fine and not art: cel.line(sq, w_ * 1.0, tc, (.22, .45), seed=int(abs(q[0, 0]) * 9000), press=.55, tremor=.15, alpha=a_, tmin=.18)
                else:
                    cx.set_source_rgba(float(tc[0]), float(tc[1]), float(tc[2]), a_); cx.set_line_width(w_ * .85); cx.new_path(); cx.move_to(*sq[0])
                    for p in sq[1:]: cx.line_to(p[0], p[1])
                    cx.stroke()
            cel.unclip()
        # ink: deckled borders in full, creases thinner and lifted
        k0 = 0; runs = []
        for m, fl in f['seg']:
            pts = np.vstack([poly[k0:k0 + m], poly[(k0 + m) % len(poly)][None]]); k0 += m
            if runs and runs[-1][1] == fl and fl == 1: runs[-1][0] = np.vstack([runs[-1][0], pts[1:]])
            else: runs.append([pts, fl])
        for q, (pts, fl) in enumerate(runs):
            ln = np.hypot(*np.diff(pts, axis=0).T).sum()
            if ln < 4: continue
            if rim > 0 and fl == 1:
                dv = pts[-1] - pts[0]; nr = np.array([dv[1], -dv[0]]) / (np.hypot(*dv) + 1e-6)
                if (pts.mean(0) - poly.mean(0)) @ nr < 0: nr = -nr
                if nr @ L[:2] > .3 * np.hypot(*L[:2]) and ti < 2: cel.line(pts - nr * lw * .9, lw * 1.5 * rim, col('rim'), (.2, .3), seed=seed + q, press=.4)
            if fl == 1:
                pieces = [pts]
                if len(pts) > 24:          # the pencil lifts now and then
                    cut = int(len(pts) * (.35 + .3 * hash01(seed + q + f['i'], drawing // 4))); pieces = [pts[:cut], pts[cut + 2:]]
                for e in pieces:
                    if len(e) > 2: cel.line(e, lw * (.85 + .3 * hash01(seed + q, 8)), ink, (.12, .25), seed=seed + q * 7 + drawing, press=.40, alpha=ink_alpha)
            else: cel.line(pts, lw * .72, mix(base, ink, .78), (.12, .35), seed=seed + q * 5 + drawing, press=.35, alpha=ink_alpha * .95)
    return cel


def put_origami(img, stage, at, size, R, t=1.0, opacity=1.0, ss=2, **kw):
    S = stage if isinstance(stage, Sheet) else origami_stage(stage, kw.get('seed', 0), t)
    return _put(img, at, size * .78 + 8, lambda c: origami(c, S, at, size, R, **kw), opacity, ss)


def rot3(ax=0., ay=0., az=0.):
    """camera-space rotation for origami(): degrees about x (nod), y (turn), z (spin)"""
    a, b, c = (math.radians(v) for v in (ax, ay, az))
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]); Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def _put(img, at, ext, fn, opacity=1.0, ss=2):
    h, w = img.shape[:2]
    x0, y0 = int(max(-40, at[0] - ext)), int(max(-40, at[1] - ext)); x1, y1 = int(min(w + 40, at[0] + ext)), int(min(h + 40, at[1] + ext))
    if x1 - x0 < 4 or y1 - y0 < 4: return img
    c = Cel(x1 - x0, y1 - y0, x0, y0, ss=ss)
    fn(c)
    return over_roi(img, c.rgba(), x0, y0, opacity)


def put_bird(img, design, phase, R, at, span, opacity=1.0, ss=2, **kw):
    sm = np.hypot(*kw['smear']) if kw.get('smear') is not None else 0
    return _put(img, at, span * 1.5 + sm * 1.2 + 8, lambda c: bird(c, design, phase, R, at, span, **kw), opacity, ss)


def put_page(img, R, at, size, opacity=1.0, ss=2, **kw):
    sm = np.hypot(*kw['smear']) if kw.get('smear') is not None else 0
    return _put(img, at, size * .72 + sm * .8 + 8, lambda c: page(c, R, at, size, **kw), opacity, ss)


# ------------------------------------------------------------------ crowds
def _tone_vec(n, L, trans):
    d = (n * L).sum(-1)
    t = np.where(d > .30, 0, np.where(d > -.02, 1, np.where(d > -.62, 2, 3)))
    if trans: t = np.where((t >= 2) & (d < -.55), 1, t)
    return t


def _smooth_polys(v, corner, sub=3):
    """(m,F,6,2) control polygons -> (m,F,6*sub,2) closed curves with breaks at the corner vertices.  corner (F,6) 0/1"""
    n1, p1 = np.roll(v, -1, 2), np.roll(v, 1, 2)
    m0 = (n1 - p1) * .5 * (1 - corner[None, :, :, None])
    m1 = np.roll(m0, -1, 2)
    out = []
    for q in range(sub):
        t = q / sub; t2, t3 = t * t, t * t * t
        out.append((2 * t3 - 3 * t2 + 1) * v + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * n1 + (t3 - t2) * m1)
    return np.stack(out, 3).reshape(v.shape[0], v.shape[1], -1, 2)


_CORN = np.array([[1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1], [1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1], [1, 0, 0, 1, 1, 0]], f32)
_SUB = np.array([[0, 1], [2, 3]])


def _tier(img, cam, k, c, pos, B, span, design, phase, seed, drawing, L, PA, palids, tn, px, z, s2, smear, trans, fog, A):
    """far and mid tiers: confident flat shapes with curved edges"""
    G = GEOF[design[k], phase[k]].astype(f32); m = len(k)
    ha = np.stack([hash01(seed[k], 11), hash01(seed[k], 12), hash01(seed[k], 13)], -1).astype(f32) - .5
    G = G * np.stack([1 + ha[:, 0] * .22, 1 + ha[:, 1] * .20, np.ones(m, f32)], -1)[:, None, None]
    G[..., 0] *= 1 + np.sign(G[..., 0]) * ha[:, 2, None, None] * .14
    c3 = np.einsum('mij,mfvj->mfvi', B[k].astype(f32), G) * np.asarray(span, f32)[k][:, None, None, None]
    w3 = c3 + ((pos[k] - cam.eye) @ cam.R.T).astype(f32)[:, None, None]
    zz = np.maximum(w3[..., 2], .08)
    v2 = np.stack([w3[..., 0] / zz * cam.f + cam.cx, w3[..., 1] / zz * cam.f + cam.cy], -1)
    vid = np.arange(30).reshape(5, 6)
    ws = px[k][:, None, None, None]
    j1 = np.stack([hash01(seed[k][:, None, None] * 61 + vid, 31), hash01(seed[k][:, None, None] * 61 + vid, 32)], -1) - .5
    j2 = np.stack([hash01(seed[k][:, None, None] * 61 + vid, 41 + int(drawing)), hash01(seed[k][:, None, None] * 61 + vid, 5041 + int(drawing))], -1) - .5
    v2 = v2 + (j1 * np.minimum(ws * .08, 6) + j2 * np.minimum(ws * .025 + .2, 1.0)).astype(f32)
    # shared vertices of the two strips stay together
    for q in (0, 2):
        v2[:, q + 1, 0] = v2[:, q, 4]; v2[:, q + 1, 1] = v2[:, q, 3]; v2[:, q + 1, 2] = v2[:, q, 2]; v2[:, q, 5] = v2[:, q, 4]
    if smear is not None:
        sm = np.asarray(smear, f32)[k]; mg = np.sqrt((sm ** 2).sum(-1)) + 1e-6; u = sm / mg[:, None]
        o = v2 - s2[k].astype(f32)[:, None, None]; al = (o * u[:, None, None]).sum(-1)
        kk = np.minimum(mg / np.maximum(px[k], 1), 1.8)[:, None, None]
        v2 = v2 + (u[:, None, None] * (al * .22 * kk - np.maximum(-al, 0) * .8 * kk * (1 + .5 * j1[..., 0]))[..., None]).astype(f32)
    e = c3 - c3[:, :, :1]
    nrm = np.cross(e, np.roll(e, -1, 2)).sum(2); nrm = nrm / (np.sqrt((nrm ** 2).sum(-1, keepdims=True)) + 1e-9); nrm = np.where(nrm[..., 2:3] > 0, -nrm, nrm)
    d = (nrm * (L if L.ndim == 1 else L[k][:, None])).sum(-1)
    t = np.where(d > .12, 0, np.where(d > -.45, 1, 2))
    if trans: t = np.where(d < -.60, 1, t)
    if c == 0:      # whole wings, two or three values
        lead = np.where(t[:, [0, 2]] == 0, 0, 2 if not trans else 1)
        tone = np.stack([lead[:, 0], lead[:, 0], lead[:, 1], lead[:, 1], np.maximum(2, lead.max(1))], 1)
    else:
        tone = np.stack([t[:, 0], np.minimum(t[:, 0] + 1 + (hash01(seed[k], 14) < .25), 3), t[:, 2], np.minimum(t[:, 2] + 1, 3), np.minimum(t[:, [0, 2]].min(1) + 1, 3)], 1)
    pid = palids[k]
    col = PA[pid[:, None], tone]
    tv = tn[k][:, None]; col = col * np.where((tone < 2)[..., None], tv, 1 + (tv - 1) * .5)
    fk = None
    if fog is not None:
        fk = 1 - np.exp(-np.maximum(0, z[k] - fog[1]) * fog[2]); fc = np.asarray(fog[0], f32)
        col = col * (1 - fk[:, None, None]) + fc * fk[:, None, None]
    fo = np.argsort(-w3[..., 2].mean(2), 1); mi = np.arange(m)[:, None]
    if c == 0: P = v2; nv = 6
    else: P = _smooth_polys(v2, _CORN, 3).astype(f32); nv = 18
    P = P[mi, fo]; col = col[mi, fo]
    if c == 0:
        fill_polys(img, P.reshape(-1, nv, 2), nv, col.reshape(-1, 3), 1.0, A); return
    # one broken accent along the leading edge of one wing, drawn last
    w = np.where(hash01(seed[k], 15) < .5, 0, 2)
    edge = _smooth_polys(v2, _CORN, 6)[np.arange(m), w]            # 36 samples: L0..L1..L2 are samples 0..12
    i0 = (1 + 2 * hash01(seed[k], 16)).astype(int); i1 = (5 + 2 * hash01(seed[k], 17 + int(drawing) // 2)).astype(int); i2 = i1 + 1 + (hash01(seed[k], 18) < .5); i3 = 11 - (hash01(seed[k], 19) * 2).astype(int)
    ic = PA[pid, 4]
    if fog is not None: ic = ic * (1 - fk[:, None]) + fc * fk[:, None]
    lw = np.clip(px[k] * .020 + .2, .5, 1.7)
    ar = np.arange(m); Q = []
    for a_, b_ in ((i0, i1), (i2, i3)):
        mid_ = ((a_ + b_) // 2)
        Q.append(edge_quads(edge[ar, a_], edge[ar, b_], lw.astype(f32), .35, edge[ar, mid_]))
    AL = np.stack(Q, 1)                                             # m,2,6,2
    V18 = np.concatenate([AL, np.repeat(AL[:, :, -1:], 12, 2)], 2)
    allV = np.concatenate([P, V18], 1).reshape(-1, 18, 2)
    allC = np.concatenate([col, np.repeat(ic[:, None], 2, 1)], 1).reshape(-1, 3)
    allN = np.tile(np.r_[np.full(5, 18), 6, 6], m)
    aop = np.clip((px[k] - 16) / 16, 0, 1); allO = np.concatenate([np.ones((m, 5)), np.repeat((aop * .9)[:, None], 2, 1)], 1).reshape(-1)
    fill_polys(img, allV, allN, allC, allO, A)


def flock(img, cam, pos, fwd, up, span, design, phase, seed, drawing, light, pal='cream', pals=None, smear=None, trans=False,
          lod=(13., 150.), fog=None, ink=1.0, rim=0.0, near_fn=None, clipz=.12, A=None, tier=False):
    """painter-sorted crowd of paper birds.  tier=True: far = flat silhouettes in 2-3 values, mid = two facets and one broken accent line, near = full ink.  arrays over n birds; span in metres (wing half span), light = world vector to the light.
    smear: (n,2) px vectors or None.  fog=(rgb, start, density).  Birds bigger than lod[1] px are drawn as full cels."""
    pos = np.asarray(pos, float); n = len(pos)
    if n == 0: return img
    s2, z = cam.project(pos)
    px = np.asarray(span, float) * cam.f / np.maximum(z, .05)
    vis = (z > clipz) & (s2[:, 0] > -px * 1.6) & (s2[:, 0] < cam.w + px * 1.6) & (s2[:, 1] > -px * 1.6) & (s2[:, 1] < cam.h + px * 1.6)
    ii = np.nonzero(vis)[0]
    if len(ii) == 0: return img
    ii = ii[np.argsort(-z[ii])]
    palids = np.zeros(n, int) if pals is None else np.asarray(pal)
    PA = np.stack([pal_array(p) for p in (pals if pals is not None else [pal])])
    F = np.asarray(fwd, float) @ cam.R.T; U = np.asarray(up, float) @ cam.R.T
    B = basis(F, U)
    L = _norm(np.asarray(light, float) @ cam.R.T) if np.ndim(light) == 1 else np.asarray(light, float) @ cam.R.T
    design = np.asarray(design); phase = np.asarray(phase); seed = np.asarray(seed).astype(np.int64)
    tn = tints(seed)
    # split into consecutive painter batches by LOD so order is kept
    cls = np.where(px[ii] < lod[0], 0, np.where(px[ii] < lod[1], 1, 2))
    brk = np.nonzero(np.diff(cls))[0] + 1
    for seg in np.split(np.arange(len(ii)), brk):
        k = ii[seg]; c = cls[seg[0]]
        if c == 2:
            for b in k:
                lc = L if L.ndim == 1 else _norm(L[b])
                kw = dict(pal=(pals[palids[b]] if pals is not None else pal), seed=int(seed[b]), drawing=int(drawing), light=lc, trans=trans, rim=rim,
                          persp=min(.5, float(span[b] / max(z[b], .3))), smear=None if smear is None or np.hypot(*smear[b]) < 2 else smear[b])
                if near_fn is not None: near_fn(img, b, design[b], phase[b], B[b], s2[b], px[b], kw)
                else: put_bird(img, int(design[b]), int(phase[b]), B[b], s2[b], float(px[b]), ss=1, **kw)
            continue
        if tier:
            _tier(img, cam, k, c, pos, B, span, design, phase, seed, drawing, L, PA, palids, tn, px, z, s2, smear, trans, fog, A)
            continue
        G = (GEOF if c == 0 else GEO)[design[k], phase[k]].astype(f32)           # m,F,V,3
        m, Fn, Vn = G.shape[:3]
        ha = np.stack([hash01(seed[k], 11), hash01(seed[k], 12), hash01(seed[k], 13)], -1).astype(f32) - .5
        G = G * np.stack([1 + ha[:, 0] * .16 + 0, 1 + ha[:, 1] * .16, np.ones(m, f32)], -1)[:, None, None]
        G[..., 0] *= 1 + np.sign(G[..., 0]) * ha[:, 2, None, None] * .12
        c3 = np.einsum('mij,mfvj->mfvi', B[k].astype(f32), G) * np.asarray(span, f32)[k][:, None, None, None]
        base = ((pos[k] - cam.eye) @ cam.R.T).astype(f32)
        w3 = c3 + base[:, None, None]
        zz = np.maximum(w3[..., 2], .08)
        v2 = np.stack([w3[..., 0] / zz * cam.f + cam.cx, w3[..., 1] / zz * cam.f + cam.cy], -1)
        # wobble of the hand: static per bird + redraw per drawing (per vertex id)
        vid = np.maximum(VIDT[design[k]], 0) if c == 1 else np.broadcast_to(np.arange(Fn * Vn).reshape(Fn, Vn), (m, Fn, Vn))
        ws = px[k][:, None, None, None]
        j1 = np.stack([hash01(seed[k][:, None, None] * 61 + vid, 31), hash01(seed[k][:, None, None] * 61 + vid, 32)], -1) - .5
        j2 = np.stack([hash01(seed[k][:, None, None] * 61 + vid, 41 + int(drawing)), hash01(seed[k][:, None, None] * 61 + vid, 5041 + int(drawing))], -1) - .5
        amp = .05 if c == 1 else .10
        v2 = v2 + (j1 * np.minimum(ws * amp, 7) + j2 * np.minimum(ws * .02 + .2, .9)).astype(f32)
        if smear is not None:
            sm = np.asarray(smear, f32)[k]; mg = np.sqrt((sm ** 2).sum(-1)) + 1e-6; u = sm / mg[:, None]
            cen = np.stack([s2[k, 0], s2[k, 1]], -1).astype(f32)[:, None, None]
            o = v2 - cen; al = (o * u[:, None, None]).sum(-1)
            kk = np.minimum(mg / np.maximum(px[k], 1), 1.8)[:, None, None]
            rag = 1 + .6 * (j1[..., 0])
            v2 = v2 + (u[:, None, None] * (al * .22 * kk - np.maximum(-al, 0) * .9 * kk * rag)[..., None]).astype(f32)
        # tones
        e = c3 - c3[:, :, :1]
        nrm = np.cross(e, np.roll(e, -1, 2)).sum(2)
        nn = np.sqrt((nrm ** 2).sum(-1, keepdims=True)) + 1e-9
        nrm = nrm / nn; nrm = np.where(nrm[..., 2:3] > 0, -nrm, nrm)
        Lk = L if L.ndim == 1 else L[k][:, None]
        tone = _tone_vec(nrm, Lk, trans)
        pid = palids[k]
        col = PA[pid[:, None], tone]
        tv = tn[k][:, None]
        col = col * np.where((tone < 2)[..., None], tv, 1 + (tv - 1) * .5)
        col = col * (1 + (hash01(seed[k][:, None] * 5 + np.arange(Fn), 51) - .5)[..., None] * .05)
        if fog is not None:
            fk = 1 - np.exp(-np.maximum(0, z[k] - fog[1]) * fog[2]); fc = np.asarray(fog[0], f32)
            col = col * (1 - fk[:, None, None]) + fc * fk[:, None, None]
        dep = w3[..., 2].mean(2)
        fo = np.argsort(-dep, 1)
        mi = np.arange(m)[:, None]
        v2s = v2[mi, fo].reshape(-1, Vn, 2).astype(f32); cs = col[mi, fo].reshape(-1, 3).astype(f32)
        if c == 0:
            Nv = np.full(m * Fn, Vn, np.int32)
            fill_polys(img, v2s, Nv, cs, 1.0, A)
            continue
        Nv = NVT[design[k]][mi, fo].reshape(-1)
        # facets and their ink interleaved: [fill, edge x V] per facet
        EW = EWT[design[k]][mi, fo].reshape(-1, Vn)
        nf = len(v2s); pxf = np.repeat(px[k], Fn)
        lw = np.clip(pxf * .016 + .25, .45, 1.5) * ink
        ic = PA[pid, 4].repeat(Fn, 0)
        if fog is not None: ic = ic * (1 - fk.repeat(Fn)[:, None]) + fc * fk.repeat(Fn)[:, None]
        Pq = v2s
        Qq = np.stack([np.where(((q + 1) < Nv)[:, None], v2s[:, min(q + 1, Vn - 1)], v2s[:, 0]) for q in range(Vn)], 1)
        vs = VIDT[design[k]][mi, fo].reshape(-1, Vn)
        vq = np.stack([np.where((q + 1) < Nv, vs[:, min(q + 1, Vn - 1)], vs[:, 0]) for q in range(Vn)], 1)
        kk_ = np.minimum(vs, vq) * 64 + np.maximum(vs, vq)
        dq = Qq - Pq
        bw = (hash01(np.repeat(seed[k], Fn)[:, None] * 13 + kk_, 81) - .5) * .20 * np.where(vs < vq, 1, -1)
        Mq = (Pq + Qq) * .5 + np.stack([-dq[..., 1], dq[..., 0]], -1) * bw[..., None].astype(f32)
        quads = edge_quads(Pq.reshape(-1, 2), Qq.reshape(-1, 2), (lw[:, None] * np.where(EW > 0, .35 + .65 * EW, 0)).reshape(-1).astype(f32), .45, Mq.reshape(-1, 2)).reshape(nf, Vn, 6, 2)
        poly12 = np.stack([Pq, Mq], 2).reshape(nf, Vn * 2, 2)
        quads = np.concatenate([quads, quads[:, :, -1:].repeat(6, 2)], 2)
        allV = np.concatenate([poly12[:, None], quads], 1).reshape(-1, 12, 2)
        ic2 = ic[:, None] * np.ones((1, Vn, 1)); soft = (cs * .45 + ic * .55)[:, None]
        ic2 = np.where((EW > .6)[..., None], ic2, soft)
        allC = np.concatenate([cs[:, None], ic2], 1).reshape(-1, 3)
        allN = np.concatenate([Nv[:, None] * 2, np.where((EW > 0) & (np.arange(Vn) < Nv[:, None]), 6, 0)], 1).reshape(-1)
        fadein = np.clip((pxf - lod[0]) / (lod[0] * 1.2) + .25, 0, 1)
        allO = np.concatenate([np.ones((nf, 1)), np.repeat(fadein[:, None], Vn, 1)], 1).reshape(-1)
        fill_polys(img, allV, allN, allC, allO, A)
    return img


def sheets(img, cam, pos, nrm, axis, size, bend, seed, drawing, light, pal='cream', pals=None, smear=None, trans=False, fog=None,
           ink=1.0, lod=(9., 120.), rim=0.0, clipz=.12, A=None, near=True):
    """painter-sorted crowd of loose curled pages (three bent strips each).  nrm, axis: world vectors (normal, bend axis)"""
    pos = np.asarray(pos, float); n = len(pos)
    if n == 0: return img
    s2, z = cam.project(pos); size = np.asarray(size, float)
    px = size * cam.f / np.maximum(z, .05)
    vis = (z > clipz) & (s2[:, 0] > -px * 1.2) & (s2[:, 0] < cam.w + px * 1.2) & (s2[:, 1] > -px * 1.2) & (s2[:, 1] < cam.h + px * 1.2)
    ii = np.nonzero(vis)[0]
    if len(ii) == 0: return img
    ii = ii[np.argsort(-z[ii])]
    palids = np.zeros(n, int) if pals is None else np.asarray(pal)
    PA = np.stack([pal_array(p) for p in (pals if pals is not None else [pal])])
    Z = np.asarray(nrm, float) @ cam.R.T; Yv = np.asarray(axis, float) @ cam.R.T
    B = basis(Yv, Z)
    L = _norm(np.asarray(light, float) @ cam.R.T) if np.ndim(light) == 1 else np.asarray(light, float) @ cam.R.T
    seed = np.asarray(seed).astype(np.int64); bend = np.broadcast_to(np.asarray(bend, float), (n,))
    tn = tints(seed)
    cls = np.where(px[ii] < lod[1], 0, 1) if near else np.zeros(len(ii), int)
    brk = np.nonzero(np.diff(cls))[0] + 1
    for seg in np.split(np.arange(len(ii)), brk):
        k = ii[seg]
        if cls[seg[0]] == 1:
            for b in k:
                lc = L if L.ndim == 1 else _norm(L[b])
                put_page(img, B[b], s2[b], float(px[b]), ss=1, pal=(pals[palids[b]] if pals is not None else pal), seed=int(seed[b]), drawing=int(drawing),
                         bend=float(bend[b]), light=lc, trans=trans, rim=rim, persp=min(.5, float(size[b] / max(z[b], .3))),
                         smear=None if smear is None or np.hypot(*smear[b]) < 2 else smear[b])
            continue
        m = len(k)
        NS = 5
        a0 = bend[k] * (hash01(seed[k], 3) - .5) * 1.2; a1 = bend[k] * (.55 + .75 * hash01(seed[k], 4)) * np.where(hash01(seed[k], 6) < .5, 1, -1); a2 = bend[k] * (hash01(seed[k], 7) - .5) * 2.0
        hh = .5
        u = (np.arange(NS) + .5) / NS * 2 - 1
        an = a0[:, None] + a1[:, None] * u[None] + a2[:, None] * u[None] ** 2
        wd = .74 / NS
        xr = np.concatenate([np.zeros((m, 1)), np.cumsum(np.cos(an) * wd, 1)], 1); zr = np.concatenate([np.zeros((m, 1)), np.cumsum(np.sin(an) * wd, 1)], 1)
        xr -= xr.mean(1, keepdims=True); zr -= zr.mean(1, keepdims=True)
        sk = (hash01(seed[k], 5) - .5) * .34
        cr = bend[k][:, None] * .16 * (hash01(seed[k][:, None], 60 + np.arange(4)) - .2)          # four corners lift by different amounts
        ew_ = np.array([1., .45, .08, 0, .0, .0]); G = np.zeros((m, NS, 4, 3))
        for q in range(NS):
            for v, (rail, yy) in enumerate(((q, -hh), (q + 1, -hh), (q + 1, hh), (q, hh))):
                lf, rt = (1 - rail / NS) ** 3, (rail / NS) ** 3
                G[:, q, v, 0] = xr[:, rail] + sk * yy; G[:, q, v, 1] = yy + (hash01(seed[k], 70 + rail * 2 + (yy > 0)) - .5) * .045
                G[:, q, v, 2] = zr[:, rail] + (cr[:, 0] * lf + cr[:, 1] * rt if yy < 0 else cr[:, 2] * lf + cr[:, 3] * rt)
        c3 = np.einsum('mij,mfvj->mfvi', B[k], G) * size[k][:, None, None, None]
        w = c3 + ((pos[k] - cam.eye) @ cam.R.T)[:, None, None]
        zz = np.maximum(w[..., 2], .08)
        v2 = np.stack([w[..., 0] / zz * cam.f + cam.cx, w[..., 1] / zz * cam.f + cam.cy], -1)
        vid = np.stack([np.array([q, q + 1, q + 1 + NS + 1, q + NS + 1]) for q in range(NS)])[None]
        j1 = np.stack([hash01(seed[k][:, None, None] * 61 + vid, 31), hash01(seed[k][:, None, None] * 61 + vid, 32)], -1) - .5
        j2 = np.stack([hash01(seed[k][:, None, None] * 61 + vid, 41 + int(drawing)), hash01(seed[k][:, None, None] * 61 + vid, 5041 + int(drawing))], -1) - .5
        ws = px[k][:, None, None, None]
        v2 = v2 + j1 * np.minimum(ws * .055, 8) + j2 * np.minimum(ws * .02 + .2, .9)
        if smear is not None:
            sm = np.asarray(smear, float)[k]; mg = np.sqrt((sm ** 2).sum(-1)) + 1e-6; u = sm / mg[:, None]
            o = v2 - s2[k][:, None, None]; al = (o * u[:, None, None]).sum(-1)
            kk = np.minimum(mg / np.maximum(px[k], 1), 2.2)[:, None, None]
            v2 = v2 + u[:, None, None] * (al * .22 * kk - np.maximum(-al, 0) * .9 * kk * (1 + .6 * j1[..., 0]))[..., None]
        e = c3 - c3[:, :, :1]
        nr = np.cross(e, np.roll(e, -1, 2)).sum(2); nr = nr / (np.sqrt((nr ** 2).sum(-1, keepdims=True)) + 1e-9)
        nr = np.where(nr[..., 2:3] > 0, -nr, nr)
        tone = _tone_vec(nr, L if L.ndim == 1 else L[k][:, None], trans)
        pid = palids[k]; col = PA[pid[:, None], tone]; tv = tn[k][:, None]
        col = col * np.where((tone < 2)[..., None], tv, 1 + (tv - 1) * .5)
        if fog is not None:
            fk = 1 - np.exp(-np.maximum(0, z[k] - fog[1]) * fog[2]); fc = np.asarray(fog[0], f32)
            col = col * (1 - fk[:, None, None]) + fc * fk[:, None, None]
        fo = np.argsort(-w[..., 2].mean(2), 1); mi = np.arange(m)[:, None]
        v2s = v2[mi, fo].reshape(-1, 4, 2); cs = col[mi, fo].reshape(-1, 3)
        pxf = np.repeat(px[k], NS)
        lw = np.clip(pxf * .011 + .25, .4, 1.5) * ink
        ic = PA[pid, 4].repeat(NS, 0)
        if fog is not None: ic = ic * (1 - fk.repeat(NS)[:, None]) + fc * fk.repeat(NS)[:, None]
        # outline edges: top and bottom always, side rails only on the outer strips
        strip = np.broadcast_to(fo, (m, NS)).reshape(-1)
        ew = np.stack([np.ones(m * NS), (strip == NS - 1) * 1., np.ones(m * NS), (strip == 0) * 1.], -1)
        quads = edge_quads(v2s.reshape(-1, 2), np.roll(v2s, -1, 1).reshape(-1, 2), (lw[:, None] * ew).reshape(-1).astype(f32), .5).reshape(-1, 4, 6, 2)
        ov = v2s[:, [0, 3]] - v2s[:, [1, 2]]; ov = ov / (np.sqrt((ov ** 2).sum(-1, keepdims=True)) + 1e-6) * .55
        vx = v2s.copy(); vx[:, [0, 3]] += ov; vx[:, [1, 2]] -= ov          # overlap neighbours so no seam shows
        v6 = np.concatenate([vx, vx[:, -1:], vx[:, -1:]], 1)[:, None]
        on = (pxf > lod[0]) & (ink > 0)
        # a few rows of faded writing on mid pages
        allV = np.concatenate([v6, quads], 1).reshape(-1, 6, 2)
        allC = np.concatenate([cs[:, None], np.repeat(ic[:, None], 4, 1)], 1).reshape(-1, 3)
        allN = np.concatenate([np.full((m * NS, 1), 4), np.where((ew > 0) & on[:, None], 6, 0)], 1).reshape(-1)
        if (px[k] > max(34, lod[0] * .75)).any():
            rows = 6
            sdr = np.repeat(seed[k], NS)[:, None]; rid = np.arange(rows)
            t = (rid[None] + 1.0 + (hash01(sdr, 200 + rid * 5) - .5) * .5 + (hash01(sdr * 3 + strip[:, None], 230 + rid) - .5) * .05) / (rows + 1.2)
            a = v2s[:, 0, None] * (1 - t[..., None]) + v2s[:, 3, None] * t[..., None]
            b = v2s[:, 1, None] * (1 - t[..., None]) + v2s[:, 2, None] * t[..., None]
            h_ = hash01(sdr * 7 + strip[:, None] * 13, 300 + rid)
            l0 = np.where(strip[:, None] == 0, .35, 0) + .3 * h_ * (h_ < .4); l1 = np.where(strip[:, None] == NS - 1, .55, 1) - .3 * (1 - h_) * (h_ > .6)
            a_, b_ = a + (b - a) * l0[..., None], a + (b - a) * l1[..., None]
            tq = edge_quads(a_.reshape(-1, 2), b_.reshape(-1, 2), (np.repeat(pxf, rows) * .010).clip(.3, .9).astype(f32), .5).reshape(-1, rows, 6, 2)
            tc = cs[:, None] * .76 + ic[:, None] * .24
            tn_ = np.where((pxf > max(34, lod[0] * .75))[:, None] & (np.abs(nr[mi, fo].reshape(-1, 3)[:, 2]) > .3)[:, None] & (hash01(sdr + strip[:, None] * 9, 400 + rid * 3) < .84), 6, 0)
            blk = np.concatenate([v6, tq, quads], 1); Cb = np.concatenate([cs[:, None], np.repeat(tc, rows, 1), np.repeat(ic[:, None], 4, 1)], 1)
            Nb = np.concatenate([np.full((m * NS, 1), 4), tn_, np.where((ew > 0) & on[:, None], 6, 0)], 1)
            allV, allC, allN = blk.reshape(-1, 6, 2), Cb.reshape(-1, 3), Nb.reshape(-1)
        fill_polys(img, allV, allN, allC, 1.0, A)
    return img
