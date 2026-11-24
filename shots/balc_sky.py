"""Balcony lane: backlit dawn skies and cloud masses (s7, s9a).

A cloud is a tree of lobes.  Each lobe carries its own normal AND its ancestors' normals, so the light is decided by
the big mass first and the small crowns only modulate it.  Three buffers come out of the lobe painter:
  S sun (backlit, crisp)   U up-ness (soft skylight modelling of the shade side)   R rim (thin burning edge)
which are coloured as flat poster tones and then gone over with arc-shaped brush marks that follow the lobes."""
import numpy as np, math, cv2
from numba import njit
from lp import W, H
from lp.paint import f32, rgb, mix, smooth, ramp, fbm, Canvas, arcs, jit_color, _vn, lum, grid


@njit(cache=True, fastmath=True)
def _cell(x, y, s):
    """nearest jittered cell seed: returns (dx, dy, d) from the seed in cell units (domes with creases between)"""
    ix = int(math.floor(x)); iy = int(math.floor(y))
    best = 9.0; bx = 0.0; by = 0.0
    for j in range(-1, 2):
        for i in range(-1, 2):
            a = ix + i; b = iy + j
            sx = a + .5 + (_vn(a + .5, b * 1.37 + .5, s) - .5) * 1.3
            sy = b + .5 + (_vn(a * 1.31 + 9.5, b + .5, s + 7) - .5) * 1.3
            w = .75 + .5 * _vn(a * .77 + 3.5, b * .91 + 4.5, s + 11)      # unequal cells
            dx = x - sx; dy = y - sy
            d = (dx * dx + dy * dy) / w
            if d < best:
                best = d; bx = dx; by = dy
    return bx, by, math.sqrt(best)


@njit(cache=True, fastmath=True)
def _mass(S, U, R, A, AN, X, Y, RX, RY, P1, P2, SOFT, SCAL, NS, SUNX, SUNY, lz, SD, rimw, fade, halo, w1, w2, w3, b1, b2, cellk, floor_y, floor_s):
    h, w = S.shape
    TWO = 6.2831853
    for i in range(X.shape[0]):
        cx = X[i]; cy = Y[i]; rx = RX[i]; ry = RY[i]; s = SD[i]
        x0 = max(0, int(cx - rx * 1.45)); x1 = min(w - 1, int(cx + rx * 1.45) + 1)
        y0 = max(0, int(cy - ry * 1.45)); y1 = min(h - 1, int(cy + ry * 1.3) + 1)
        if x0 > x1 or y0 > y1: continue
        rp = math.sqrt(rx * ry)
        p = P1[i]; g = P2[i]
        c1 = rp * cellk; c2 = rp * cellk * .42
        for y in range(y0, y1 + 1):
            dy = (y - cy) / ry
            for x in range(x0, x1 + 1):
                dx = (x - cx) / rx
                d = math.sqrt(dx * dx + dy * dy)
                if d > 1.45: continue
                th = math.atan2(dy, dx)
                if th < 0: th += TWO
                # scalloped silhouette: two octaves of unequal outward bumps, cusps inward
                Rr = 1.0
                for o in range(2):
                    m = NS[i, o]
                    if m <= 0: continue
                    k = 0
                    while k < m - 1 and th > SCAL[i, o, k]: k += 1
                    hi = SCAL[i, o, k]; lo = SCAL[i, o, k - 1] if k > 0 else 0.0
                    gp = hi - lo
                    u = (th - lo) / gp * 2 - 1
                    bump = math.sqrt(max(0.0, 1 - u * u)) - .62
                    Rr += bump * gp * (.20 if o == 0 else .17)
                n = _vn(math.cos(th) * 2.0 + 10, math.sin(th) * 2.0 + 10, s)
                Rr += .07 * (n - .5)
                q = d / Rr
                lxp = SUNX - x; lyp = SUNY - y
                ll = math.sqrt(lxp * lxp + lyp * lyp) + 1e-6
                lxp /= ll; lyp /= ll
                side = -dy / (d + 1e-6)
                if q >= 1.0:
                    if halo > 0 and q < 1.4 and side > -.1 and A[y, x] > .2:
                        k2 = (1.4 - q) / .4
                        k2 = halo * k2 * k2 * (.3 + .7 * max(0.0, side)) / (1 + SOFT[i] * 10)
                        S[y, x] -= k2 * .9
                        U[y, x] -= k2 * .6
                    continue
                e = (1 - q) * rp
                ea = e / (1.2 + SOFT[i] * rp)
                if ea > 1: ea = 1.0
                if SOFT[i] > .02: ea = ea * ea * (3 - 2 * ea)
                if side < .3:
                    t = (.3 - side) / 1.3
                    fw = fade * t ** .8 + 1e-4
                    eb = (1 - q) / fw
                    if eb > 1: eb = 1.0
                    eb = eb * eb * (3 - 2 * eb)
                    if eb < ea: ea = eb
                if y > floor_y:
                    ea *= max(0.0, 1 - (y - floor_y) / floor_s)
                if ea <= 0: continue
                nz = math.sqrt(max(0.0, 1 - q * q))
                nx = dx / Rr * w1; ny = dy / Rr * w1; nnz = nz * w1
                if p >= 0:
                    ax = (x - X[p]) / RX[p]; ay = (y - Y[p]) / RY[p]
                    dd = math.sqrt(ax * ax + ay * ay)
                    if dd > 1: ax /= dd; ay /= dd; dd = 1.0
                    nx += ax * w2; ny += ay * w2; nnz += math.sqrt(1 - dd * dd) * w2
                if g >= 0:
                    ax = (x - X[g]) / RX[g]; ay = (y - Y[g]) / RY[g]
                    dd = math.sqrt(ax * ax + ay * ay)
                    if dd > 1: ax /= dd; ay /= dd; dd = 1.0
                    nx += ax * w3; ny += ay * w3; nnz += math.sqrt(1 - dd * dd) * w3
                # billows: cauliflower domes packed on the face
                wq = 1.1 / max(c1, 6.0)
                bx, by, bd = _cell(x / c1 + .55 * (_vn(x * wq, y * wq, s + 40) - .5) + .22 * (_vn(x * wq * 3.1, y * wq * 3.1, s + 43) - .5) + s % 97, y / c1 + .55 * (_vn(x * wq + 50, y * wq, s + 41) - .5) + .22 * (_vn(x * wq * 3.1 + 9, y * wq * 3.1, s + 44) - .5), s + 60)
                ex, ey, ed = _cell(x / c2 + s % 89, y / c2 + 3.1, s + 80)
                nx += bx * b1 + ex * b2; ny += by * b1 + ey * b2
                nl = math.sqrt(nx * nx + ny * ny + nnz * nnz) + 1e-6
                nx /= nl; ny /= nl; nnz /= nl
                cb = math.sqrt(max(0.0, 1 - lz * lz))
                sv = nx * lxp * cb + ny * lyp * cb + nnz * lz - .16 * max(0.0, bd - .45) * (1 if b1 > 0 else 0)
                uv = -ny * .8 + nnz * .12 + nx * lxp * .15 - .12 * max(0.0, bd - .45)
                rim = math.exp(-e / rimw) * max(0.0, (dx * lxp + dy * lyp) / (d + 1e-6) * .75 + .35 * max(0.0, side))
                S[y, x] += (sv - S[y, x]) * ea
                U[y, x] += (uv - U[y, x]) * ea
                R[y, x] += (rim - R[y, x]) * ea
                A[y, x] += (1 - A[y, x]) * ea
                if ea > .3: AN[y, x] = th + 1.5708


def _scallops(n, r, rp):
    SC = np.full((n, 2, 40), 6.2831853, f32); NS = np.zeros((n, 2), np.int64)
    for i in range(n):
        for o in range(2):
            want = (8, 22)[o]
            if o == 1 and rp[i] < 14: continue
            if o == 0 and rp[i] < 5: want = 4
            g = r.uniform(.35, 1.7, want); c = np.cumsum(g); c = c / c[-1] * 6.2831853
            SC[i, o, :want] = c; NS[i, o] = want
    return SC, NS


class Tree:
    """lobe tree in painting order (back to front)"""
    def __init__(self): self.L = []
    def add(self, x, y, rx, ry, p=-1, g=-1, lv=0, soft=0.): self.L.append((x, y, rx, ry, p, g, lv, soft)); return len(self.L) - 1

    def grow(self, x, y, rx, ry, r, kids=(6, 4), shrink=(.56, .5), face=(.3, .2), spread=1.0, lean=0., p=-1, g=-1, lv=0, rmin=3., soft=0., asp=(.78, 1.0)):
        """a lobe, crowns round its upper rim, some soft bulges on its face.  lean: rotate the crowns (radians)"""
        i = self.add(x, y, rx, ry, p, g, lv, soft)
        if lv >= len(kids): return
        rr = math.sqrt(rx * ry)
        if rr * shrink[lv] < rmin: return
        C = []
        K = kids[lv]
        for j in range(K):
            fc = r.random() < face[lv]
            a = -math.pi / 2 + lean + ((j + r.uniform(.15, .85)) / K - .5) * 2 * 1.75 * spread if not fc else r.uniform(-math.pi, 0)
            q = r.uniform(.70, .98) if not fc else r.uniform(.1, .62)
            cx = x + math.cos(a) * rx * q; cy = y + math.sin(a) * ry * q + (.18 * ry if fc else 0)
            k = shrink[lv] * r.uniform(.5, 1.25) * (1.0 - .25 * abs((j + .5) / K - .5) * 2)
            cr = rr * k
            C.append((cy + cr * .3, cx, cy, cr * r.uniform(1.0, 1.15), cr * r.uniform(*asp), r.uniform(.12, .5) if fc else (0. if r.random() < .72 else r.uniform(.05, .2))))
        C.sort()
        for _, cx, cy, crx, cry, sf in C:
            self.grow(cx, cy, crx, cry, r, kids, shrink, face, spread, lean * .7, i, p, lv + 1, rmin, max(sf, soft * .7), asp)


class Mass:
    """buffers of one cloud mass"""
    def __init__(self, tree, h=H, w=W, sun=(1310, 800), lz=-.45, rimw=5., fade=.95, halo=.25, weights=(.42, .36, .22), bill=(.30, .12), cell=.42, floor=(1e6, 1.), seed=0, ids=None):
        L = np.array([t[:4] for t in tree.L], f32); n = len(L)
        self.h, self.w = h, w
        self.S = np.zeros((h, w), f32); self.U = np.zeros((h, w), f32); self.R = np.zeros((h, w), f32); self.A = np.zeros((h, w), f32); self.AN = np.zeros((h, w), f32)
        p1 = np.array([t[4] for t in tree.L], np.int64); p2 = np.array([t[5] for t in tree.L], np.int64); lv = np.array([t[6] for t in tree.L], np.int64); sf = np.array([t[7] for t in tree.L], f32)
        sd = ((np.arange(n) * 613 + seed * 977) & 0xffffff).astype(np.int64)
        if n:
            SC, NS = _scallops(n, np.random.default_rng(seed + 5), np.sqrt(L[:, 2] * L[:, 3]))
            if ids is not None:                                            # each lobe keeps its own outline from drawing to drawing
                sd = ((np.asarray(ids, np.int64) * 613 + seed * 977) & 0xffffff)
                for i in range(n):
                    a, b = _scallops(1, np.random.default_rng(int(ids[i]) + 9), np.sqrt(L[i:i + 1, 2] * L[i:i + 1, 3])); SC[i], NS[i] = a[0], b[0]
            _mass(self.S, self.U, self.R, self.A, self.AN, L[:, 0], L[:, 1], L[:, 2], L[:, 3], p1, p2, sf, SC, NS, float(sun[0]), float(sun[1]), float(lz), sd,
              float(rimw), float(fade), float(halo), *[float(v) for v in weights], float(bill[0]), float(bill[1]), float(cell), float(floor[0]), float(floor[1]))
        self.L = L; self.lv = lv; self.p1 = p1; self.sun = sun


def tone_steps(t, n, seed, h, w, crisp=.16, lost=.95, scale=90):
    """poster-tone quantise t in 0..n, boundaries crisp in places and lost in others"""
    k = np.clip(np.floor(t), 0, n - 1)
    fr = t - k
    m = smooth(-.9, .9, fbm(h, w, (scale, scale * .4), (1, .5), seed))
    wd = crisp + (lost - crisp) * m
    return k + np.clip((fr - .5) / wd + .5, 0, 1)


def colour(m, shade, light, rimc, sun_lo=-.05, sun_hi=.75, rim=.8, seed=1, up=(-.75, .75), noise=.16, lightamt=1.0, crisp=.16):
    """shade: 4 colours deep->pale by up-ness.  light: 3 colours warm half -> light -> burning."""
    h, w = m.h, m.w
    n1 = fbm(h, w, (64, 23, 9), (1, .6, .35), seed); n2 = fbm(h, w, (120, 40, 14), (1, .6, .35), seed + 3)
    sh = np.stack([rgb(c) if isinstance(c, str) else c for c in shade]); li = np.stack([rgb(c) if isinstance(c, str) else c for c in light])
    u = (m.U + noise * n1 - up[0]) / (up[1] - up[0]) * (len(sh) - 1)
    u = tone_steps(np.clip(u, 0, len(sh) - 1.001), len(sh) - 1, seed + 5, h, w, crisp * 1.6, 1.0)
    i = np.clip(np.floor(u).astype(int), 0, len(sh) - 2); fr = (u - i)[..., None]
    c = sh[i] * (1 - fr) + sh[i + 1] * fr
    s = (m.S + noise * .75 * n2 - sun_lo) / (sun_hi - sun_lo) * len(li)
    s = tone_steps(np.clip(s, 0, len(li) - .001), len(li), seed + 9, h, w, crisp, .85)
    for k in range(len(li)):
        a = np.clip(s - k, 0, 1)[..., None] * lightamt
        c = c + (li[k] - c) * a
    ra = np.clip(m.R * rim * (1 + .35 * n1), 0, 1)[..., None]
    rc = rgb(rimc) if isinstance(rimc, str) else rimc
    return c + (rc - c) * ra


def marks(m, img, r, n, size=(.10, .22), lo=0, op=(.35, .8), val=.02, dry=.35, cols=None, pick=None, lenk=(.5, 1.3), where=None):
    """arc brush marks that follow individual lobes, coloured from the picture (or cols(S)->rgb)"""
    L = m.L; N = len(L)
    if N == 0: return img
    area = np.sqrt(L[:, 2] * L[:, 3]); pr = area.astype(float); pr[m.lv < lo] = 0
    if pr.sum() <= 0: return img
    k = r.choice(N, n, p=pr / pr.sum())
    rx, ry = L[k, 2], L[k, 3]
    th = -np.pi / 2 + r.normal(0, .85, n); q = r.uniform(.25, .97, n)
    x = L[k, 0] + np.cos(th) * rx * q; y = L[k, 1] + np.sin(th) * ry * q
    ix = np.clip(x.astype(int), 0, m.w - 1); iy = np.clip(y.astype(int), 0, m.h - 1)
    ok = (m.A[iy, ix] > .6) & (x >= 0) & (x < m.w) & (y >= 0) & (y < m.h)
    if where is not None: ok &= where(m.S[iy, ix], m.U[iy, ix])
    k, rx, ry, th, q, x, y, ix, iy = [v[ok] for v in (k, rx, ry, th, q, x, y, ix, iy)]
    n = len(k)
    if n == 0: return img
    rp = np.sqrt(rx * ry) * q
    ln = np.clip(rp * r.uniform(*lenk, n), 5, 90)
    an = np.arctan2(np.cos(th) * ry, -np.sin(th) * rx)
    P = arcs(x, y, an, ln, np.clip(ln / np.maximum(rp, 3), 0, 1.6))
    sm = cv2.GaussianBlur(img, (0, 0), 2.2)
    c = sm[iy, ix] if cols is None else cols(m.S[iy, ix], m.U[iy, ix], sm[iy, ix])
    c = jit_color(c, r, .006, .015, val)
    cv_ = Canvas(m.h, m.w, img=img)
    wd = np.clip(np.sqrt(rx * ry) * r.uniform(*size, n), 1.2, 11)
    cv_.strokes(P, wd, c, r.uniform(*op, n), r.uniform(0, 1, n) * dry, seed=int(r.integers(1 << 20)), shape='round', bristle=.03)
    return cv_.img


def lay(img, m, col, a=None, op=1.0):
    a = m.A if a is None else a
    return img + (col - img) * (a * op)[..., None]


# --------------------------------------------------------------- sky wash
def sky_wash(img, r, n=420, hw=(18, 60), ln=(260, 760), centre=None, val=.012, hue=.006, op=(.08, .26), y0=-40, y1=900, dry=.25):
    """long flat-brush washes that follow the sky (level, bowed a little around the sun)"""
    h, w = img.shape[:2]
    y = r.uniform(y0, y1, n); x = r.uniform(-100, w + 100, n)
    cx = w / 2 if centre is None else centre[0]
    an = (x - cx) / w * -.10 * (1 if centre is None else np.clip((centre[1] - y) / 700, -.3, 1)) + r.normal(0, .03, n)
    ref = cv2.GaussianBlur(img, (0, 0), 14)
    ix = np.clip(x, 0, w - 1).astype(int); iy = np.clip(y, 0, h - 1).astype(int)
    c = jit_color(ref[iy, ix], r, hue, .01, val)
    cv_ = Canvas(h, w, img=img)
    cv_.strokes(arcs(x, y, an, r.uniform(*ln, n), r.normal(0, .06, n)), r.uniform(*hw, n), c, r.uniform(*op, n), r.uniform(0, 1, n) * dry, seed=int(r.integers(1 << 20)), shape='round', bristle=.014)
    return cv_.img
