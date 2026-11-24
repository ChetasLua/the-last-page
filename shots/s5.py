"""s5 The Library Stirs (168 f), hard cuts 5a..5i.  Library lane."""
import os, math, numpy as np, cv2
from lp import W, H, books as bk, comp, foxact, props
from lp.paint import rgb, mix, smooth, ramp, fbm, warp_by, repaint, Canvas, arcs, mottle, f32, grid, lum
from lp.ink import Cel, spl, over_roi, paste, noise1, ribbon
from lp.cam import Camera3D
from . import hall, s2
from .s2 import _save, CACHE

KEY = 'a7'
UP = np.array([0, 0, 1.])


def _blob(h, w, pts, soft=2., wob=3., seed=1, scale=40):
    m = np.zeros((h, w), f32); cv2.fillPoly(m, [np.round(np.asarray(pts, float) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, 4)
    if wob > 0: m = warp_by(m, wob, scale, seed, cv2.BORDER_REPLICATE)
    return cv2.GaussianBlur(m, (0, 0), soft) if soft > 0 else m


def _brush(h, w, ang, seed, amt=.05, scales=(46, 14, 5), aspect=5.):
    """streaky painted variation running at angle ang (radians)"""
    d = int(math.hypot(h, w)) + 8
    n = fbm(d, d, scales, (1, .7, .4), seed, aspect=aspect)
    M = cv2.getRotationMatrix2D((d / 2, d / 2), -math.degrees(ang), 1.0); M[:, 2] += (w / 2 - d / 2, h / 2 - d / 2)
    return 1 + amt * cv2.warpAffine(n, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def speck(img, x, y, s, col, a, ang=0., L=0.):
    """one hand-placed grain / short dash"""
    h, w = img.shape[:2]; R = int(s + L / 2 + 3); ix, iy = int(x), int(y)
    if not (R < ix < w - R - 1 and R < iy < h - R - 1): return
    yy, xx = np.mgrid[iy - R:iy + R + 1, ix - R:ix + R + 1].astype(f32)
    c, s_ = math.cos(ang), math.sin(ang); u = (xx - x) * c + (yy - y) * s_; v = -(xx - x) * s_ + (yy - y) * c
    q = np.hypot(np.maximum(np.abs(u) - L / 2, 0), v); g = np.clip(s + .6 - q, 0, 1) * a
    t = img[iy - R:iy + R + 1, ix - R:ix + R + 1]; t += (rgb(col) - t) * g[..., None]


def stream(img, p0, dirv, t, speed, seed, col='#b7c2e4', w=1.2, n=70, amt=.6, reach=900., spread=.012, grow=0., bright=None):
    """a thin thread of falling dust that started t frames ago (drawn on twos): loose grains and short dashes in hand-made
    clumps, thinning and fanning as they fall.  p0 px, dirv unit px direction, speed px/frame, bright(x, y) -> 0..1 light"""
    if t < 0: return
    r = np.random.default_rng(seed)
    head = min(reach, speed * t + .5 * grow * t * t)                           # a thread of haze first, then the grains
    for j in range(4):
        d0, d1 = head * j / 4., head * (j + 1.06) / 4.; dm = (d0 + d1) / 2
        x, y = p0[0] + dirv[0] * dm, p0[1] + dirv[1] * dm
        if d1 - d0 > 4: speck(img, x, y, 1.1 + .7 * j * (w + .2), col, .13 * amt * (1 - .20 * j) * (bright(x, y) if bright is not None else 1.), math.atan2(dirv[1], dirv[0]), d1 - d0)
    n = int(n * 4.2 / max(8., reach / speed * .8) * 48)                       # grains emitted over 48 frames, in clumps that fall with the thread
    tb = np.sort(r.uniform(0, 48, n)); cl = np.interp(tb, np.linspace(0, 48, 34), r.uniform(.05, 1.25, 34) ** 1.4)
    J, G, LA, LN = r.uniform(.80, 1.06, n), r.normal(0, 1, n), r.uniform(.2, 1., n), r.uniform(0, 1, n); WD = r.choice([.35, .5, .8, 1.1], n, p=[.4, .35, .18, .07])
    nx, ny = -dirv[1], dirv[0]; ang = math.atan2(dirv[1], dirv[0]); bow = r.normal(0, 5)
    for k in range(int(np.searchsorted(tb, t, 'right'))):
        age = t - tb[k]; jit, g, la, wd, ln = J[k], G[k], LA[k], WD[k], LN[k]
        d = (speed * age + .5 * grow * age * age) * jit
        if d > reach: continue
        q = d / reach
        off = g * (.7 + spread * 1.9 * d) + bow * math.sin(q * 2.2 + seed) + 2.0 * math.sin(d * .010 + seed)
        a = amt * cl[k] * (1 - q) ** 1.1 * (.35 + .65 * min(1, d / 50.)) * la * (.85 if ln < .6 else .6)
        x = p0[0] + dirv[0] * d + nx * off; y = p0[1] + dirv[1] * d + ny * off
        if bright is not None: a *= bright(x, y)
        if a < .03: continue
        L = (0. if ln < .55 else min(16, speed * .42 * ln)) * (1 - .5 * q)
        speck(img, x, y, w * wd * (1 + .7 * q), col, a, ang, L)


def puff(img, x, y, t, seed, s=1.0, col='#8f97bd', up=1.0, n=9):
    """a small jump of dust: a few drawn flecks thrown up and settling, t in drawings (twos)"""
    r = np.random.default_rng(seed)
    for k in range(n):
        vx, vy = r.normal(0, 6.0) * s, -abs(r.normal(8.5, 4.0)) * s * up
        tt = t + r.uniform(.4, 1.5)
        X = x + r.normal(0, 20 * s) + vx * tt * (1 - .06 * tt); Y = y + r.uniform(-2, 3) + vy * tt + 1.9 * s * tt * tt * .5
        a = r.uniform(.15, .5) * max(0, 1 - t / r.uniform(4, 9))
        if a <= .02: continue
        speck(img, X, Y, r.choice([.5, .8, 1.2, 2.0]) * s, col, a, math.atan2(vy + 1.9 * t, vx), r.uniform(0, 6) * s * max(0, 1 - t / 4.))


def lift(img, c='#1c1a30', k=.42):
    """shadows end in violet, never in black"""
    return 1 - (1 - np.clip(img, 0, 1.2)) * (1 - rgb(c) * k)


def shade(img, Fm):
    """light field (rgb) + moon mask; moonlight drains colour"""
    m = Fm[..., 3:4] * .42
    img = img * (1 - m) + lum(img)[..., None] * np.array([.97, 1.0, 1.08], f32) * m
    return lift(img * Fm[..., :3])


# ------------------------------------------------------------------ generic close shelf
class Shelf:
    """painted shelf + cels + one light field shared by both"""

    def __init__(self, T, items, moving, L, **kw):
        self.T, self.items, self.moving, self.L = T, items, moving, L
        self.kw = kw

    def cels(self, pose, seed, lw=2.2, roi=None, order=1):
        """pose(i) -> dict of offsets.  premultiplied layer"""
        x0, y0, x1, y1 = roi or (0, 0, W, H)
        c = Cel(x1 - x0, y1 - y0, x0, y0)
        ids = sorted(self.moving, key=lambda i: self.items[i].u * order)
        for i in ids:
            p = pose(i)
            if p is None: continue
            bk.book_cel(c, self.T, self.items[i], self.L, seed=seed + i * 7, lw=p.pop('lw', lw), **{**self.kw, **p})
        return c.rgba(), x0, y0


def board(P, T, u0, u1, v, L, th=.036, d0=-.34, d1=.0, wood='#54403c', seed=1, sagf=None, top=True, n=5, dust=True):
    """one plank of shelf: upper (or under) face and front edge"""
    wd = rgb(wood); N_ = 11 if sagf is None else 49; us = np.linspace(u0, u1, N_); sg = np.zeros(N_) if sagf is None else sagf(us)
    a, b = T(us, v + sg, d1), T(us, v + sg, d0)
    if top: P.strip(b, a, wd * 1.05, L, n=n + 2, val=.06, seed=seed, hi=0, lo=.25)
    else:
        a, b = T(us, v - th + sg, d1), T(us, v - th + sg, d0)
        P.strip(a, b, wd * .42, L, n=n, val=.08, seed=seed, hi=0, lo=0)
    if top and dust:
        r = np.random.default_rng(seed)
        for _ in range(40):
            u = r.uniform(u0, u1); dd = r.uniform(d0 * .25, d1); p = T(np.array([u, u + r.uniform(.02, .10)]), np.full(2, v) + np.interp(u, us, sg), np.array([dd, dd + r.normal(0, .01)]))
            P.q(np.stack([p[0], p.mean(0) + r.normal(0, .5, 2), p[1]]), r.uniform(1.0, 3.2), L(rgb('#8f88a0') * r.uniform(.5, .85), *p[0])[0], .45, .5)
    f0, f1 = T(us, v + sg, d1), T(us, v - th + sg, d1)
    P.strip(f0, f1, wd * 1.1, L, n=n, val=.08, seed=seed + 9, hi=.45, lo=-.45, dry=.25)
    P.flush()


def stiff(items, ids=None):
    for i in (range(len(items)) if ids is None else ids): items[i].swell = 0.
    return items


# ------------------------------------------------------------------ 5a
HERO_A = 6
MOVE_A = {3: (-1, 3), 4: (-1, 2), 5: (-1, 1), 7: (1, 1), 8: (1, 2), 9: (1, 3)}      # neighbour -> (side, how far out)


def scene_a():
    T = bk.flat(262, 892, 1720, vp=(1235, 590), k=.62)
    L = bk.Light(np.array([.50, .475, .62], f32), '#12101f', -1)
    items = bk.make_row(77, 1.36, .40, fill=1.0, messy=.0, unit=1.18, hmin=.62)
    h = items[HERO_A]; h.label = (.665, .775, 0); h.col2 = rgb('#8f7448'); h.torn = .35; stiff(items, [HERO_A] + list(MOVE_A))                # the one that wakes: an old ribbed folio
    h.bands = ((.055, .010, 1), (.15, .017, 0), (.335, .018, 0), (.52, .017, 0), (.62, .008, 1), (.82, .018, 0), (.945, .010, 1))
    return T, L, items


def plate_a():
    T, L, items = scene_a(); r = np.random.default_rng(3)
    P = bk.Painter(H, W, 2, base='#191629')
    for i in range(13):
        u0 = -.28 + i * .125; P.strip(T(np.full(5, u0 + .004), np.linspace(-.02, .42, 5), -.33), T(np.full(5, u0 + .121), np.linspace(-.02, .42, 5), -.33), rgb('#3a2f3c') * r.uniform(.40, .62), L, n=4, val=.1, seed=i)
    P.flush()
    Ld = bk.Light(np.array([.25, .235, .34], f32), '#12101f', -1)
    P.poly([[0, 940], [W, 940], [W, H], [0, H]], Ld(rgb('#241d30'), 1000, 1050)[0])
    low = bk.make_row(33, 1.6, .40, fill=1.0, messy=.0, unit=1.25, hmin=.80)
    bk.paint_row(P, T, low, Ld, v0=-.448, width=1.6, d0=-.07, u0=-.28)
    board(P, T, -.3, 1.4, 0., L, th=.036, seed=4, d1=.012)
    bk.paint_row(P, T, bk.make_row(5, .208, .40, fill=1.0, messy=0, unit=1.1, hmin=.6), L, width=.21, d0=-.055, u0=-.212)
    bk.paint_row(P, T, items, L, width=1.36, skip=(HERO_A,) + tuple(MOVE_A), d0=-.055)
    # contact shadows at the feet
    r = np.random.default_rng(8)
    for u in np.arange(-.24, 1.3, .05):
        p = T(np.array([u, u + .06]), np.zeros(2), -.045); P.q(np.stack([p[0], p.mean(0) + [0, 1.5], p[1]]), 4.5, L(rgb('#2b2233') * .6, *p[0])[0], .5, .25)
    P.flush()
    board(P, T, -.3, 1.4, .425, L, th=.045, seed=6, d1=.02, top=False)
    P.poly([[0, -5], [W, -5], [W, T(0, .425, -.02)[1]], [0, T(0, .425, -.02)[1]]], Ld(rgb('#241d30'), 1000, 50)[0])
    Lu = bk.Light(np.array([.34, .32, .45], f32), '#12101f', -1)
    bk.paint_row(P, T, bk.make_row(19, 1.6, .40, fill=1.0, messy=.0, unit=1.2, hmin=.7), Lu, v0=.426, width=1.6, d0=-.05, u0=-.3)
    a_, b_ = T(np.linspace(-.3, 1.4, 5), np.full(5, .425), .02), T(np.linspace(-.3, 1.4, 5), np.full(5, .380), .02)
    P.strip(a_, b_, rgb('#54403c') * 1.1, L, n=5, val=.08, seed=15, hi=.4, lo=-.45, dry=.25); P.flush()
    im = P.image()
    _save(f's5_{KEY}_a.npy', im.astype(np.float16))
    # light both the painting and the cel share: moon through a pane, head shadow of the upper board, dim corners
    x, y = grid()
    F = np.ones((H, W, 3), f32) * .90
    F *= (1 - .30 * smooth(500, 1500, np.hypot(x - 1060, (y - 560) * 1.2)))[..., None]
    rs = np.random.default_rng(6); edges = [-40.] + sorted(float(T(i.u, 0.)[0]) for i in items if 0 < T(i.u, 0.)[0] < W) + [W + 40.]
    lowe = []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        yy = 338 + 34 * (1 - a_ / W) + rs.normal(0, 9); lowe += [[a_ + 1, yy], [b_ - 1, yy + rs.normal(0, 4)]]
    hs = _blob(H, W, [[-40, 238], [W + 40, 238]] + lowe[::-1], 1.6, 0, 3, 60)
    F *= 1 - hs[..., None] * np.array([.50, .52, .42], f32)
    m = np.zeros((H, W), f32)
    # two panes of moonlight, leaning with the same sun, crossing the hero's left half
    for q in ([[250, 1152], [700, 1152], [1120, 300], [840, 300]], [[820, 1152], [1010, 1152], [1290, 420], [1205, 420]]): m = np.maximum(m, _blob(H, W, q, 2.6, 1.6, 11, 50))
    m *= np.clip(1.1 - hs * 1.2, 0, 1) * (.75 + .25 * smooth(1152, 300, y))
    F += m[..., None] * np.array([1.22, 1.30, 1.42], f32) * .95
    F *= _brush(H, W, -1.1, 5, .045)[..., None]
    _save(f's5_{KEY}_aF.npy', np.dstack([F, m]).astype(np.float16))


# frame: (lean, slid out of the row m, lifted mm, shifted mm).  still / twitch / hold / double twitch / shiver
HERO_KEYS = [(0, (0., 0., 0., 0.)), (8, (.062, .034, 5., 1.)), (10, (-.034, .026, 0., -1.)), (12, (.008, .022, 0., 0.)),
             (19, (-.072, .050, 6., -2.)), (21, (.034, .046, 0., 0.)), (22, (.108, .070, 11., 3.)), (24, (-.046, .062, 2., -1.)), (26, (.014, .058, 0., 0.)),
             (28, (.092, .078, 9., 2.)), (29, (-.070, .074, 3., -2.)), (30, (.104, .088, 11., 3.)), (31, (-.082, .084, 4., -2.)), (32, (.112, .100, 13., 3.)), (33, (-.090, .096, 5., -2.))]


def hero_key(f):
    k = [i for i, (a, _) in enumerate(HERO_KEYS) if f >= a][-1]; return k, HERO_KEYS[k][1]


def pose_a(i, f):
    """-> drawing number, rigid pose of book i"""
    if i == HERO_A:
        k, (rot, dd, dv, du) = hero_key(f); return k, dict(drot=rot, dd=dd, dv=dv * .001, du=du * .001)
    side, n = MOVE_A[i]
    start = {1: 8, 2: 19, 3: 28}[n]
    if f < start + n - 1: return 0, dict()
    k, (rot, dd, dv, du) = hero_key(max(0, f - (n - 1)))
    hit = rot * side > 0                                                        # the hero has come down on this side
    q = .74 * .66 ** (n - 1)
    r = rot * q if hit else -.30 * rot * q * (-1 if n % 2 else 1)
    sh = 0.
    if f >= 27 + n:                                                             # the shiver spreads outward, one book a frame
        sh = (.046, .038, .032)[n - 1] * (1 if (f + i) % 2 else -1)
    return k + 100 * n + (f if f >= 27 + n else 0), dict(drot=r * .8 + sh, dd=.010 * n ** -.5 + .16 * dd * q, dv=(.0035 if hit else 0.) + (.003 if sh > 0 else 0.), du=side * .0015 * (1 if hit else 0))


PUFF_A = [(8, 735, 886, .8), (8, 950, 886, .8), (19, 738, 886, 1.0), (22, 955, 886, 1.3), (22, 730, 886, 1.0), (28, 742, 886, 1.2), (29, 960, 886, 1.3), (31, 1085, 886, 1.0), (32, 640, 886, 1.0)]


def frame_a(S, f):
    T, L, items = S.sa
    c = Cel(1060, 900, 420, 110)
    ids = [HERO_A] + list(MOVE_A)
    od = [i for i in bk.draw_order(T, items, 0., 0., -.055) if i in ids]
    P_ = {i: pose_a(i, f) for i in ids}
    for i in sorted(od, key=lambda j: (round(P_[j][1].get('dd', 0.) * 60), od.index(j))):
        k, p = P_[i]; bk.book_cel(c, T, items[i], L, seed=k + i * 3, d0=-.055, lw=3.0 if i == HERO_A else 2.4, rigid=True, **p)
    img = S.get('a'); img = over_roi(img, c.rgba(), c.ox, c.oy)
    img = shade(img, S.get('aF'))
    F = S.c['aF']; br = lambda x, y: .62 + .38 * float(F[min(H - 1, max(0, int(y))), min(W - 1, max(0, int(x))), 3])
    for j, (fs, x, y, s) in enumerate(PUFF_A): hall.dust_puff(img, x, y, f - fs, 40 + j, s * 1.15, 1.0)
    for j, (fs, x, rc) in enumerate(((10, 905, 260), (20, 1180, 330), (23, 780, 420), (27, 1012, 420), (30, 640, 300))):
        hall.veil(img, (x, 226.), (0.02, 1.), f - fs, 60 + j, speed=11., reach=rc, w0=1.8, spread=.080, amt=1.45, bright=br, grow=.2)
    return comp.cam2d(img, W / 2 + 60 * .02 * f / 34., H / 2, 1 + .022 * f / 34.)


# ------------------------------------------------------------------ 5b  the tremble runs along the row
def scene_b():
    cam = Camera3D((0., -1.62, .47), (.28, 0., .205), 1500, W, H, cx=W * .50, cy=H * .50)
    an = .27; eu = np.array([math.cos(an), math.sin(an), 0]); n = np.array([math.sin(an), -math.cos(an), 0.])
    p0 = np.array([-1.22, -.30, 0.])
    T = bk.plane(cam, p0, eu, UP, n)
    L = bk.Light(np.array([.50, .475, .62], f32), '#12101f', -1)
    r = np.random.default_rng(3)
    items = [i for i in stiff(bk.make_row(41, 4.6, .365, fill=1.0, messy=.10, unit=1.02, hmin=.60))]
    return cam, T, L, items


def _edges_shadow(T, items, width, hv, amt, seed, u0=0.):
    r = np.random.default_rng(seed); pts = []
    us = [-.1] + sorted(i.u for i in items) + [width]
    for a_, b_ in zip(us[:-1], us[1:]):
        v = hv * (1 - amt) + r.normal(0, .006); pts += [(a_ + .001, v), (b_ - .001, v + r.normal(0, .003))]
    pts = np.array(pts)
    return np.vstack([T(pts[:, 0] + u0, pts[:, 1], -.02), T(pts[::-1, 0] + u0, np.full(len(pts), hv * 1.04), -.02)])


def plate_b():
    cam, T, L, items = scene_b()
    P = _shelfplate(T, L, 51, -.25, 5.2, hv=.405, nlow=2, nup=2, pitch=.425, kl=(.30, .285, .40), ku=(.40, .38, .52))
    _save(f's5_{KEY}_b.npy', P.image().astype(np.float16))
    x, y = grid()
    F = np.ones((H, W, 3), f32) * .92
    F *= (1 - .36 * smooth(800, 2000, x))[..., None]
    hs = _blob(H, W, _edges_shadow(T, items, 4.7, .365, .22, 2), 1.3, 0)
    F *= 1 - hs[..., None] * np.array([.42, .45, .36], f32)
    # what is left of the moon: a low slab raking in from the left, cut by the post of the next bay
    m = np.maximum(_blob(H, W, [[-20, 1180], [-20, 300], [300, -20], [420, -20], [560, 1180]], 3, 1.5, 5), _blob(H, W, [[640, 1180], [500, -20], [610, -20], [880, 1180]], 3, 1.5, 6))
    m *= np.clip(1.1 - hs * 1.2, 0, 1) * .8
    F += m[..., None] * np.array([1.10, 1.20, 1.36], f32) * .75
    F *= _brush(H, W, -.1, 5, .04)[..., None]
    _save(f's5_{KEY}_bF.npy', np.dstack([F, m]).astype(np.float16))


STREAM_B = [(2.25, 1), (1.98, 0), (1.66, 1), (1.40, 0), (1.18, 1), (.95, 0), (.74, 1), (.55, 0), (.30, 1), (.13, 0)]


def frame_b(S, f):
    cam, T, L, items = S.sb
    d = f // 2; t = d * 2.
    U0, VU = 2.30, .080
    uc = U0 - VU * t                                                             # crest of the wave, far right to near left
    def pose(i):
        it = items[i]; r = np.random.default_rng(i * 31 + 5); rd = np.random.default_rng(i * 131 + d * 17)
        ppm = np.hypot(*(T(it.u, .1) - T(it.u, 0.))) / .1                        # px per metre where this book stands
        s_ = (it.u - uc) / .20
        g = math.exp(-s_ * s_) if s_ > -2.2 else 0.
        drop = math.exp(-((s_ - 1.75) / .55) ** 2)                               # it comes down, knocks, hops once more
        wake = .55 * math.exp(-s_ * .085) if s_ > .9 else 0.
        sg = 1 if r.random() < .5 else -1
        q = g * r.uniform(.8, 1.25); w = wake * r.uniform(.4, 1.3)
        k = sg * ((-1) ** d)
        lift = (19. * q + 6. * drop * r.uniform(.5, 1.2)) / ppm
        return dict(dv=lift + .002 * w * rd.random(), dd=.065 * q + .012 * drop + .006 * w, du=.003 * k * w, drot=-.125 * q * (.6 + .4 * sg) + .060 * drop * sg + k * .020 * w + .010 * rd.normal() * w)
    img = S.get('b')
    c = Cel(W, 780, 0, 170)
    for i in bk.draw_order(T, items, 0., 0., -.045):
        x = T(items[i].u, .1)[0]
        if not (-260 < x < W + 200): continue
        bk.book_cel(c, T, items[i], L, seed=d + i * 5, lw=2.1 if x < 900 else 1.7, d0=-.045, rigid=True, **pose(i))
    img = over_roi(img, c.rgba(), 0, 170)
    img = shade(img, S.get('bF'))
    F = S.c['bF']; br = lambda x, y: .70 + .30 * float(F[min(H - 1, max(0, int(y))), min(W - 1, max(0, int(x))), 3])
    for k, (u, lo) in enumerate(STREAM_B):
        t0 = (U0 + .10 - u) / VU
        p = T(u, .366 if not lo else -.038, .018); p2 = T(u, .166 if not lo else -.238, .018); dv = (p2 - p) / np.hypot(*(p2 - p)); sc = np.hypot(*(p2 - p)) / 200.
        hall.veil(img, p, dv, f - t0, 70 + k, speed=11.5, reach=(430 if not lo else 340) * sc * 1.25, w0=1.7, spread=.075, amt=1.25, bright=br, grow=.3)
    for k, u in enumerate((1.9, 1.5, 1.12, .78, .45, .2)):
        p = T(u, .004, -.0); hall.dust_puff(img, p[0], p[1], f - (U0 + .05 - u) / VU, 300 + k, 1.0, .95)
    return comp.cam2d(img, W / 2, H / 2, 1.012 + .010 * f / 32.)


# ------------------------------------------------------------------ 5c  the creak runs up the tower
def cam_c(): return Camera3D((-3.4, 7.2, .5), (1.9, 17, 8.6), 1060, W, H, cx=W * .50, cy=H * .52)


ZC = [hall.HS * k for k in range(1, 8)]


def plate_c():
    cam = cam_c(); hh = hall.Hall(cam, H, W, 2, 'up')
    E, N = hall.EX, hall.EY
    L = hh.use((0, hall.YB + .3, 0), -N); hh.quad((hall.XL, hall.YB + .3, 0), (hall.XR, hall.YB + .3, 0), (hall.XR, hall.YB + .3, 26), (hall.XL, hall.YB + .3, 26), rgb('#3a3550'), L, 12)
    L = hh.use((0, 0, 20.4), -UP, .5)
    for y in np.arange(6., 17., 1.5):                                       # roof timbers, almost lost
        hh.quad((hall.XL, y, 19.5), (hall.XR, y, 19.5), (hall.XR, y + .5, 19.5), (hall.XL, y + .5, 19.5), rgb('#463239'), L, 3, val=.1)
    p0 = np.array([hall.XL, hall.YB, 0.])
    bays = [4.25, 5.85, 7.35, 8.95, 10.5, 12.1, 13.8]
    hh.wall(p0, E, -N, [0, 1.15, 4.25] + bays[1:], 8, 1, hole=lambda i, k: i == 1 and k in (2, 3), hazeoff=.02); hall.window(hh)
    for z in ZC[::-1]:
        hh.gallery(p0, E, -N, 4.25 if z == 3 * hall.HS else 0., 13.8, z, posts=bays[1:-1] if z < 18 else (), broken=[(9.1, 10.4)] if z == 5. else [(6.0, 7.0)] if z == 12.5 else [])
    pr = np.array([hall.XR, 17., 0.]); rb = [0, 1.7, 3.4, 5.1, 6.8, 8.5, 10.2]
    hh.wall(pr, -N, -E, rb, 8, 2)
    for z in ZC[::-1]: hh.gallery(pr, -N, -E, 0., 10.2, z, posts=rb[1:-1], broken=[(3.2, 4.4)] if z == 7.5 else [])
    pl = np.array([hall.XL, 8., 0.]); lb = [0, 1.75, 3.5, 5.2, 6.9, 9.0]
    hh.wall(pl, N, E, lb, 8, 3, gain=.92)
    for z in ZC[::-1]: hh.gallery(pl, N, E, 0., 9.0, z, posts=lb[1:-1], gain=.92)
    hall.banner(hh, np.array([1.0, hall.YB - .97, 0.]), E, -N, 1.15, 8.4, 4.6, '#6b3a44', 5, .95)
    hh.ladder((2.2, hall.YB - .95, 2.5), (2.3, hall.YB - .2, 4.8), .42, E, 9, -N)
    for k in range(10):
        r = np.random.default_rng(40 + k); x = r.uniform(-6, -2.6); z0 = 4.9; ln = r.uniform(.5, 2)
        hh.vine([(x, 16.05, z0)] + [(x + r.normal(0, .06), 16.0, z0 - ln * q) for q in np.linspace(.2, 1, 5)], .03, .012, '#3f4a42', -N, 1.2, leaves=int(14 + ln * 14), seed=200 + k, lsize=.08, lcol='#66785a')
    img, pid, planes, boxes = hh.finish()
    out, alb, pos, n, b, m = hall.light_plate(img, pid, planes, boxes, cam)
    # keep the tower readable: more moon on the edges that face the window, air between the storeys
    out = comp.screen(np.clip(out * 1.12, 0, 1.2), (np.clip(b, 0, 1.6)[..., None] * .36) * hall.MOON)
    out = mottle(np.clip(out, 0, 1), .012, 3, .004)
    _save(f's5_{KEY}_c.npy', out.astype(np.float16)); _save(f's5_{KEY}_cb.npy', b.astype(np.float16))


STREAM_C = [(-2.6, 16.0, 10.0, 8), (-.35, 16.02, 12.5, 12), (.62, 16.02, 7.5, 5), (1.95, 16.02, 15.0, 15), (2.9, 16.02, 10.0, 9), (4.2, 16.02, 5.0, 2), (5.25, 14.2, 7.5, 6), (5.25, 12.6, 12.5, 13), (5.25, 11.2, 5.0, 3), (-4.6, 16.02, 12.5, 14), (3.6, 16.02, 17.5, 17)]


def frame_c(S, f):
    img = S.get('c'); cam = S.cc
    # two held poses of flex, storey after storey from the floor up
    dx = np.zeros((H // 8, W // 8), f32); dy = np.zeros_like(dx)
    yy, xx = np.mgrid[0:H // 8, 0:W // 8].astype(f32) * 8
    any_ = False
    for k, z in enumerate(ZC[:6]):
        t0 = 1 + 3 * k
        if f < t0: continue
        any_ = True
        p = 1. if f < t0 + 6 else -.42                                       # bowed, then sprung back and held
        for (a, b, A) in (((hall.XL, 16.06, z), (hall.XR, 16.06, z), 7.0), ((5.26, 16., z), (5.26, 7.6, z), 8.0)):
            ts = np.linspace(0, 1, 13)[:, None]; P3 = np.array(a) + (np.array(b) - np.array(a)) * ts
            q, zc = cam.project(P3); ok = zc > .5
            if ok.sum() < 3: continue
            q = q[ok]; u = ts[ok, 0]
            xs = np.clip(xx, q[:, 0].min(), q[:, 0].max()); o = np.argsort(q[:, 0])
            yl = np.interp(xs, q[o, 0], q[o, 1]); ul = np.interp(xs, q[o, 0], u[o])
            hgt = 70. if A == 7. else 120.
            w = np.exp(-((yy - yl) / hgt) ** 2) * np.sin(np.pi * np.clip((ul - .12) / .8, 0, 1)) * (np.abs(xx - xs) < 4)
            if A == 8.: w *= smooth(1500, 1640, xx)
            else: w *= 1 - smooth(1500, 1640, xx)
            dy += A * p * w; dx += 1.6 * p * w * (1 if k % 2 else -1)
    if any_:
        DX = cv2.resize(cv2.GaussianBlur(dx, (0, 0), 1.2), (W, H), interpolation=cv2.INTER_LINEAR); DY = cv2.resize(cv2.GaussianBlur(dy, (0, 0), 1.2), (W, H), interpolation=cv2.INTER_LINEAR)
        gx, gy = grid()
        img = cv2.remap(img, gx - DX, gy - DY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    d2 = f // 2 * 2
    B = S.c['cb']; br = lambda x, y: .36 + .64 * min(1., float(B[min(H - 1, max(0, int(y))), min(W - 1, max(0, int(x)))]) * .9)
    for k, (X, Y, Z, t0) in enumerate(STREAM_C):
        p = cam.project(np.array([X, Y, Z]))[0]; q = cam.project(np.array([X, Y - .25, Z - 5.5]))[0]; v = q - p; Lg = np.hypot(*v)
        hall.veil(img, p, v / Lg, f - t0, 500 + k, speed=15. + 4 * (k % 3), reach=Lg * 1.05, w0=1.5 + .4 * (k % 3), spread=.052 + .012 * (k % 2), amt=1.05, cols=('#d6dcf2', '#b0badc', '#8791bd'), bright=br, grow=.8)
    return comp.cam2d(img, W / 2, H / 2 - 5 * f / 26., 1.012)


# ------------------------------------------------------------------ 5d  the flame leans with no wind
LEAN = [0, .25, .25, .6, .95, 1.]
LX = np.array([-2.42, 5.30, 0.])


def warmalb(al):
    """albedo as the lantern sees it: a little less chroma, paper does not burn out"""
    a = al * .60 + lum(al)[..., None] * .28
    return a / (1 + np.maximum(a - .30, 0) * 1.5) + .035



def cam_d(): return Camera3D((-2.28, 3.86, .50), (-2.14, 6.4, .17), 1500, W, H, cx=W * .52, cy=H * .50)


def plate_d():
    cam = cam_d(); hh = hall.Hall(cam, H, W, 2, 'd'); E, N = hall.EX, hall.EY
    P = hh.P; P.poly([[-5, -5], [W + 5, -5], [W + 5, H + 5], [-5, H + 5]], rgb('#22203a'))
    hh.use((0, 0, 0), UP); fq = hh.pj((-9, 4.2, 0), (-9, 17.2, 0), (8.4, 17.2, 0), (8.4, 4.2, 0)); P.poly(fq, rgb('#16131f'))
    hh.floor(-12., 8.0, 4.15, 14.2, gap=.0042, wob=.9)
    r = np.random.default_rng(4)
    Lb = hh.use((0, 14.2, 0), -N, .8)
    T = bk.plane(cam, (-8, 14.2, 0), E, UP, -N)
    T = bk.plane(cam, (-14, 14.2, 0), E, UP, -N)
    for i in range(13): bk.paint_bay(P, T, i * 1.8, i * 1.8 + 1.8, 0, [.5, .46, .44, .44, .42, .42, .4, .4, .4], Lb, 70 + i, post=.12, board=.05)
    for (x, y, n, sd, sc, yw) in ((-3.9, 9.5, 9, 11, 1.1, None), (-3.1, 9.0, 5, 12, 1.1, None), (-1.0, 9.6, 8, 13, 1.1, None), (-.35, 8.8, 4, 14, 1.1, None), (-2.1, 10.2, 3, 15, 1.1, None),
                                  (-2.86, 6.75, 3, 16, 1.0, .3), (-1.62, 6.62, 7, 17, 1.05, -.2), (-1.30, 6.25, 3, 18, 1.0, .5), (-3.20, 5.95, 6, 19, 1.0, .15), (-1.78, 5.62, 2, 20, 1.0, -.35), (-2.95, 5.28, 1, 21, 1., .5)):
        hh.stack(x, y, n, 4000 + sd, sc, yaw=yw)
    L = hh.use((0, 0, 0), UP, 1.)
    for (x, y, a) in ((-2.05, 5.02, .4), (-2.75, 4.92, -.5), (-2.25, 6.2, .2)):
        c = np.array([x, y, .004]); ex = np.array([math.cos(a), math.sin(a), 0]) * .12; ey = np.array([-math.sin(a), math.cos(a), 0]) * .085
        q = hh.pj(c - ex - ey, c + ex - ey, c + ex + ey + UP * .02, c - ex + ey); lb = P.poly(q, L(rgb('#b9a98a') * .85, *q.mean(0))[0])
        for k in range(5):
            t = .18 + .15 * k; e0 = q[0] * (1 - t) + q[3] * t; e1 = q[1] * (1 - t) + q[2] * t
            P.q(np.stack([e0 * .85 + e1 * .15, (e0 + e1) / 2, e0 * (.25 + .1 * r.random()) + e1 * .7]), 1.2, L(rgb('#7d7068'), *e0)[0], .4, .5, lb)
    img, pid, planes, boxes = hh.finish()
    pos, n, t = hall.positions(cam, pid, planes)
    alb = hall.albedo_of(img, pos, n, cam.eye)
    # background out of focus
    far = smooth(3.6, 5.2, t) * (t < 39)
    far = np.maximum(far, (pid == 0) * 1.)
    bl = lambda a: comp.bokeh(np.dstack([a, np.ones((H, W, 1), f32)]), 9, 0)[..., :3]
    k = cv2.GaussianBlur(far.astype(f32), (0, 0), 3)[..., None]
    img = img * (1 - k) + bl(img) * k; alb = alb * (1 - k) + bl(alb.astype(f32)) * k
    _save(f's5_{KEY}_d.npy', np.clip(img * .86, 0, 1).astype(np.float16)); _save(f's5_{KEY}_dA.npy', alb.astype(np.float16))
    tx = fbm(H, W, (70, 24, 9), (1, .8, .5), 71, aspect=2.6)
    Es = []
    for j, le in enumerate([0, .25, .6, .95, 1.]):
        W3 = LX + np.array([.0 + .34 * le, -.02, .30 - .07 * le])
        v = W3 + np.array([0, 0, .12], f32) * (n[..., 2:3] > .6) - pos; dist = np.sqrt((v * v).sum(-1)) + 1e-6
        ndl = np.clip((v * n).sum(-1) / dist, 0, 1)
        E = (.16 + .84 * ndl ** .8) * (ndl > 0) / (1 + (dist / (.74 - .06 * le)) ** 2.1)
        pp = (pos + n * .012).astype(f32); pp[E < .006] = 1e9
        vis = np.ones(E.shape, f32); hall._boxvis(pp, 0., 0., 0., W3.astype(np.float64), boxes, vis)
        vis = cv2.GaussianBlur(cv2.medianBlur((vis * 255).astype(np.uint8), 5).astype(f32) / 255., (0, 0), 2.6)
        E = E * (.08 + .92 * vis)
        # the lantern's own foot shadow falls opposite the flame
        e = np.clip(E * 1.9, 0, 1.25) + tx * .03 * np.sqrt(np.clip(E, 0, 1))
        st = np.array([0, .05, .13, .27, .48, .78, 1.15]); lv = np.array([0, .035, .10, .23, .44, .76, 1.08]); q = np.interp(e, st, lv); hd = np.zeros_like(e)
        for a, l0, l1 in zip(st[1:], lv[:-1], lv[1:]): hd += (l1 - l0) * smooth(a * .86, a * 1.02, e)
        Es.append(cv2.GaussianBlur((.5 * q + .5 * hd).astype(f32), (0, 0), 1.0))
    _save(f's5_{KEY}_dE.npy', np.stack(Es).astype(np.float16))


def frame_d(S, f):
    k = min(5, f // 4); le = LEAN[k]; cam = S.cd
    E = S.c['dE'][[0, 1, 1, 2, 3, 4][k]].astype(f32)[..., None]
    img = S.get('d'); al = S.get('dA')
    r = np.random.default_rng(60 + f // 3); fl = 1 + r.normal(0, .02)
    img = img + warmalb(al) * rgb('#ffb458') * np.array([1.5, 1.28, 1.0], f32) * E * 1.8 * fl * (1 - .5 * np.clip(img, 0, 1))
    p = cam.project(LX)[0]; sc = float(cam.scale(LX)) * .43 / 100.
    # contact shadow of the lantern foot, sliding away from the flame
    sh = np.zeros((H, W), f32); cv2.ellipse(sh, (int(p[0] - 46 * le), int(p[1] - 4)), (int(30 * sc + 20 * le), int(6.0 * sc)), 0, 0, 360, 1., -1, cv2.LINE_AA)
    img = img * (1 - .6 * cv2.GaussianBlur(sh, (0, 0), 7)[..., None])
    c = Cel(620, 720, int(p[0]) - 310, int(p[1]) - 640)
    props.lantern(c, p[0], p[1], sc, 0.015 * le, le, 1.0, seed=k, lw=3.4)
    fx, fy = p[0] + (10 + 46 * le) * sc * .42, p[1] - 44 * sc + 5 * le * sc
    # flame that has left the paper: a long drawn tongue pulled sideways
    img = s2.Shot.halo(img.astype(f32).copy(), fx, fy, 620, '#ffab52', .26 * fl, 2.2)
    rg = c.rgba(); rg[int(p[1]) + 1 - c.oy:] = 0
    img = over_roi(img, rg, c.ox, c.oy)
    img = s2.Shot.halo(img, fx, fy, 300, '#ffd48c', .34 * fl, 2.0)
    return lift(img, k=.25)


# ------------------------------------------------------------------ 5e  behind the fox: the page it stands on
def plate_e():
    im = s2.hall_bokeh().astype(f32) * .84
    x, y = grid()
    lf = spl([[-60, 1064], [160, 1040], [430, 1036], [760, 1050], [1080, 1054], [1270, 1050], [1352, 1072], [1392, 1120], [1404, 1180]], n=14)
    rt = spl([[1412, 1180], [1424, 1112], [1470, 1058], [1580, 1026], [1800, 1014], [2110, 1022]], n=14)
    ml = _blob(H, W, np.vstack([lf, [[-60, 1180]]]), 0, 1.5, 3, 60); mr = _blob(H, W, np.vstack([rt, [[2110, 1180]]]), 0, 1.5, 4, 60)
    yl = np.interp(x[0], lf[:, 0], lf[:, 1])[None]; yr = np.interp(x[0], rt[:, 0], rt[:, 1])[None]
    warm = np.clip(1 - np.hypot((x - 1990) / 900., (y - 1090) / 420.), 0, 1)
    cl = ramp(np.clip((y - yl) / 110., 0, 1), [(0, '#8e8ea8'), (.25, '#7d7c97'), (1, '#656480')]) * (1 - .34 * smooth(1150, 1400, x))[..., None] + (warm * .18)[..., None] * np.array([1., .75, .4], f32)
    cr = ramp(np.clip((x - 1420) / 620., 0, 1), [(0, '#6b5560'), (.16, '#957c70'), (.5, '#c3a37a'), (1, '#e6c487')]) * (1 - .14 * np.clip((y - yr) / 120., 0, 1))[..., None]
    k = np.zeros((H, W, 3), f32); a = np.maximum(ml, mr)
    k = cl * ml[..., None] + cr * (mr * (1 - ml))[..., None]
    edge = np.exp(-((y - yl - 3) / 3.2) ** 2) * (x < 1370) * .22 + np.exp(-((y - yr - 3) / 3.2) ** 2) * (x > 1440) * .26
    k = k + edge[..., None] * np.array([.9, .9, 1.], f32) * (a > .5)[..., None]
    k = k * _brush(H, W, .03, 12, .035)[..., None]
    lay = np.dstack([k * 1.0, a]); lay = comp.bokeh(lay.astype(f32), 5.0, 0); lay = cv2.GaussianBlur(lay, (0, 0), 1.6)
    # the page throws a little of its light up into the dark behind it
    im = comp.screen(im, (cv2.GaussianBlur(a, (0, 0), 60)[..., None] * rgb('#8a86a6') * .10))
    im = im * (1 - lay[..., 3:4]) + lay[..., :3]
    im = comp.glow(im, 1985, 985, 560, '#ffb458', .30, 2.1); im = comp.glow(im, 2010, 1000, 190, '#ffdfa6', .30, 1.8)
    _save(f's5_{KEY}_e.npy', mottle(np.clip(im, 0, 1), .008, 4, .003).astype(np.float16))


# ------------------------------------------------------------------ 5f 5g 5h  three shouts
def _rowcels(c, T, items, L, pose, seed, lw, d0=-.04, order=1, v0=0.):
    od = bk.draw_order(T, items, 0., v0, d0)
    for i in sorted(od, key=lambda j: (round((pose(j) or {}).get('dd', 0.) * 4), od.index(j))):
        p = pose(i)
        if p is None: continue
        bk.book_cel(c, T, items[i], L, seed=seed + i * 5, lw=lw, d0=d0, v0=v0, rigid=True, **p)


def scene_f():
    T = bk.flat(-60, 840, 1320, vp=(1170, 545), k=1.05); L = bk.Light(np.array([.50, .475, .62], f32), '#12101f', -1)
    return T, L, stiff(bk.make_row(93, 1.7, .37, fill=1.0, messy=0., unit=1.12, hmin=.62))


def scene_g():
    cam = Camera3D((.25, -1.15, -.42), (.12, 0, .20), 1250, W, H, cx=W * .55, cy=H * .50, roll=.07)
    T = bk.plane(cam, (-1.15, 0, 0), (1, 0, 0), UP, (0, -1, 0)); L = bk.Light(np.array([.50, .475, .62], f32), '#12101f', 1)
    return T, L, stiff(bk.make_row(133, 2.5, .37, fill=1.0, messy=0., unit=1.15, hmin=.62))


def scene_h():
    cam = Camera3D((.10, -.92, .80), (.22, 0, .19), 1480, W, H, cx=W * .50, cy=H * .50, roll=-.06)
    T = bk.plane(cam, (-1.0, 0, 0), (1, 0, 0), UP, (0, -1, 0)); L = bk.Light(np.array([.50, .475, .62], f32), '#12101f', -1)
    return T, L, stiff(bk.make_row(177, 2.3, .40, fill=1.0, messy=0., unit=1.3, hmin=.62))


def _shelfplate(T, L, seed, u0, u1, hv=.40, under=False, sag=None, nlow=2, nup=2, pitch=.425, kl=(.28, .265, .38), ku=(.38, .36, .50), th=.036):
    """painted shelves: the hero row (empty, its books are cels) between as many rows as the frame needs"""
    r = np.random.default_rng(seed); P = bk.Painter(H, W, 2, base='#1b1830')
    for i in range(int((u1 - u0) / .125) + 1):
        u = u0 + i * .125; P.strip(T(np.full(5, u + .003), np.linspace(-.02, hv, 5), -.30), T(np.full(5, u + .122), np.linspace(-.02, hv, 5), -.30), rgb('#3a2f3c') * r.uniform(.40, .62), L, n=3, val=.1, seed=i)
    P.flush()
    q = np.array([u0, u1, u1, u0])
    def back(v0, v1, Lx): P.poly(T(q, np.array([v0, v0, v1, v1]), -.06), Lx(rgb('#241d30'), W / 2, H / 2)[0])
    rows = [(hv + pitch * k, k) for k in range(nup)] + [(-pitch * (k + 1), -k - 1) for k in range(nlow)]
    for (v, k) in sorted(rows, key=lambda z: -abs(z[1])):
        f = .86 ** (abs(k) - 1 if k < 0 else abs(k)); Lx = bk.Light(np.array(ku if k >= 0 else kl, f32) * f, '#12101f', L.side)
        back(v, v + pitch, Lx)
        Lb = bk.Light(np.array([.50, .475, .62], f32) * (.8 if k < 0 else .95) * f, '#12101f', L.side)
        if not under and not (k >= 0): board(P, T, u0, u1, v, Lb, th=th, seed=seed + 11 + k, d1=.016, top=True)
        bk.paint_row(P, T, bk.make_row(seed + 3 + k * 5, u1 - u0, pitch - .055, fill=.98, messy=.25, unit=1.05, hmin=.70), Lx, v0=v + .001, width=u1 - u0, d0=-.05, u0=u0, clear_h=pitch - .05, top_shadow=.16, seed=seed + k)
        board(P, T, u0, u1, v, Lb, th=th, seed=seed + 31 + k, d1=.016, top=False if (under or k >= 0) else True, dust=False) if (under or k >= 0) else None
    if under: board(P, T, u0, u1, 0., L, th=.04, seed=4, d1=.016, sagf=sag, top=False)
    else: board(P, T, u0, u1, 0., L, th=th, seed=4, d1=.016, sagf=sag)
    board(P, T, u0, u1, hv, L, th=.040, seed=6, d1=.016, top=False)
    P.flush()
    return P


def _fieldF(T, items, width, hv, quads, seed, base=.90, dark=None, k=.8, u0=0., hs_amt=.2):
    x, y = grid(); F = np.ones((H, W, 3), f32) * base
    if dark is not None: F *= dark(x, y)[..., None]
    hs = _blob(H, W, _edges_shadow(T, items, width, hv, hs_amt, seed, u0), 1.4, 0)
    F *= 1 - hs[..., None] * np.array([.42, .45, .36], f32)
    m = np.zeros((H, W), f32)
    for j, q in enumerate(quads): m = np.maximum(m, _blob(H, W, q, 2.6, 1.5, 11 + j, 50))
    m *= np.clip(1.1 - hs * 1.2, 0, 1)
    F += m[..., None] * np.array([1.18, 1.28, 1.40], f32) * k
    F *= _brush(H, W, -.9, seed, .045)[..., None]
    return np.dstack([F, m]).astype(np.float16)


def plate_f():
    T, L, items = scene_f(); P = _shelfplate(T, L, 210, -.5, 2.2, nlow=1, nup=1)
    _save(f's5_{KEY}_f.npy', P.image().astype(np.float16))
    _save(f's5_{KEY}_fF.npy', _fieldF(T, items, 1.7, .37, ([[1180, -20], [1640, -20], [1010, 1180], [430, 1180]], [[1740, -20], [1900, -20], [1330, 1180], [1120, 1180]]), 3, .90, lambda x, y: 1 - .35 * smooth(600, 0, x) - .25 * smooth(1500, 2048, x)))


SAG_G = lambda u: .085 * np.exp(-((np.asarray(u) - 1.30) / .36) ** 2) - .016


def plate_g():
    T, L, items = scene_g(); P = _shelfplate(T, L, 240, -.6, 3.2, under=True, sag=SAG_G, nlow=1, nup=3)
    _save(f's5_{KEY}_g.npy', P.image().astype(np.float16))
    _save(f's5_{KEY}_gF.npy', _fieldF(T, items, 2.5, .37, ([[-20, 300], [-20, -20], [420, -20], [1180, 1180], [560, 1180]],), 5, .90, lambda x, y: 1 - .40 * smooth(1000, 2000, x)))


def plate_h():
    T, L, items = scene_h(); P = _shelfplate(T, L, 270, -.5, 2.9, hv=.43, nlow=3, nup=1)
    _save(f's5_{KEY}_h.npy', P.image().astype(np.float16))
    _save(f's5_{KEY}_hF.npy', _fieldF(T, items, 2.3, .46, ([[1060, -20], [1460, -20], [1860, 1180], [1180, 1180]], [[700, -20], [880, -20], [1010, 1180], [760, 1180]]), 7, .90, lambda x, y: 1 - .35 * smooth(700, 100, x), hs_amt=.10))


def speed(c, p0, p1, n, spread, cols, seed, w=3.0):
    """dry-brush speed lines of a smear"""
    r = np.random.default_rng(seed); p0, p1 = np.asarray(p0, float), np.asarray(p1, float); v = p1 - p0; L = np.hypot(*v); nv = np.array([-v[1], v[0]]) / L
    for k in range(n):
        o = nv * r.uniform(-spread, spread); a, b = sorted(r.uniform(0, 1, 2)); b = min(1, b + .25)
        P = np.stack([p0 + v * a + o, p0 + v * (a + b) / 2 + o * 1.03 + r.normal(0, 1, 2), p0 + v * b + o * 1.05])
        c.line(P, w * r.uniform(.4, 1.5), cols[int(r.integers(len(cols)))], (.5, .2), alpha=r.uniform(.35, .85), seed=seed + k)


def frame_f(S, f):
    """books slide straight out of the row or tip forward on the board edge.  Nothing bends."""
    T, L, items = S.sf; jolt = f < 2; q = 1.12 if jolt else 1.0
    rr = np.random.default_rng(8); ps = {}
    for i, it in enumerate(items):
        a = rr.random()
        if a < .22: p = dict(dd=rr.uniform(0., .03), drot=rr.normal(0, .02))
        elif a < .62: p = dict(dd=rr.uniform(.08, .30), drot=rr.normal(0, .045))          # slid out
        else: p = dict(dd=rr.uniform(.05, .10), pitch=rr.uniform(.28, .62), drot=rr.normal(0, .06))   # tipping over the edge
        ps[i] = {k: v * q for k, v in p.items()}
    c = Cel(W, H)
    reach = lambda j: ps[j].get('dd', 0) + .22 * ps[j].get('pitch', 0)
    for i in sorted(ps, key=lambda j: reach(j) - .10 * abs(T(items[j].u, .1)[0] - 1170) / 1000.):
        bk.book_cel(c, T, items[i], L, seed=(1 if jolt else 2) + i * 5, lw=2.2 + 3. * reach(i), d0=-.04, rigid=True, **ps[i])
        o = ps[i].get('dd', 0)
        if jolt and o > .10:
            it = items[i]
            for t_ in (.10, .42, .78): speed(c, T(it.u + it.w * .5, it.h * t_, -.04 + o * 1.1), T(it.u + it.w * .5, it.h * t_, -.04 + o * .2), 2, 10, ['#241a26', '#3c3558'], 60 + i, 2.4)
    img = over_roi(S.get('f'), c.rgba(), 0, 0); img = shade(img, S.get('fF'))
    for k in range(6): p = T(.14 + .27 * k, .004, 0.02); hall.dust_puff(img, p[0], p[1], 5 + f, 700 + k, 1.5, .85)
    for k, u in enumerate((.32, .86, 1.22)): p = T(u, .352, .02); hall.veil(img, p, (0., 1.), 14 + f, 740 + k, speed=15., reach=430., spread=.08, amt=1.1)
    off = [(-16, 9), (10, -6)][f] if jolt else (0, 0)
    return comp.cam2d(img, W / 2 + off[0], H / 2 + off[1], 1.03 if jolt else 1.02)


def frame_g(S, f):
    """the board bows up; rigid books ride it, part like teeth and fall against each other"""
    T, L, items = S.sg; jolt = f < 2; q = 1.15 if jolt else 1.0
    rr = np.random.default_rng(14); c = Cel(W, H); ps = {}
    for i, it in enumerate(items):
        g = math.exp(-((it.u - 1.30) / .36) ** 2); sl = -(it.u - 1.30) / .36 * g
        ps[i] = dict(dv=float(SAG_G(it.u)) + .055 * g * rr.uniform(.1, 1.2) * q, drot=-.34 * sl * q * rr.uniform(.75, 1.2) + rr.normal(0, .035), dd=.12 * g * rr.uniform(0, 1.2) * q, pitch=(.34 * g * rr.random() if rr.random() < .4 else 0.) * q)
    fall = min(range(len(items)), key=lambda j: abs(items[j].u - 1.05))
    ps[fall] = None
    _rowcels(c, T, items, L, lambda i: ps[i], 1 if jolt else 2, 2.4)
    it = items[fall]; p0 = T(it.u, .15, .05); p1 = np.array([560., 830.]) if jolt else np.array([520., 890.])
    sp = bk.Light(np.array([.50, .475, .62], f32), '#12101f', 1)
    lit = lambda col, k=1.: sp(np.clip(col, 0, 1), 0, 0, k)[0]
    speed(c, p0 + (p1 - p0) * .10, p1, 26, 135, [tuple(lit(it.col)), tuple(lit(it.col * .62)), tuple(lit(it.col * 1.2)), '#241a26'], 77, 13.0)
    bk.book_free(c, p1[0], p1[1], 270, 380, .62 if jolt else .70, it, sp, open_=.5, bend=0., seed=3, lw=3.2, thick=74, sx=1.0, sy=1.18 if jolt else 1.0)
    img = over_roi(S.get('g'), c.rgba(), 0, 0); img = shade(img, S.get('gF'))
    for k, u in enumerate((1.16, 1.3, 1.46, 1.62, 1.8, 2.0)):
        p = T(u, float(SAG_G(u)) - .04, .02); v = np.array([.05, 1.]); hall.veil(img, p, v / np.hypot(*v), 8 + 2 * (k % 3) + f, 800 + k, speed=17., reach=420., spread=.085, amt=.95)
    off = [(8, -18), (-6, 11)][f] if jolt else (0, 0)
    return comp.cam2d(img, W / 2 + off[0], H / 2 + off[1], 1.03 if jolt else 1.02)


def frame_h(S, f):
    """runs of rigid books have gone over like dominoes and caught on each other; two leap clear"""
    T, L, items = S.sh; jolt = f < 2; q = 1.12 if jolt else 1.0
    rr = np.random.default_rng(23); c = Cel(W, H); ps = {}
    jump = {min(range(len(items)), key=lambda j: abs(items[j].u - u)): a for u, a in ((.72, .085), (1.06, .15), (1.62, .07))}
    ang, left, sg = 0., 0, 1
    for i, it in enumerate(items):
        if left <= 0: left = int(rr.integers(2, 5)); sg = -sg; ang = sg * rr.uniform(.12, .26)      # a new run leans the other way
        left -= 1; ang *= rr.uniform(.80, 1.02)
        ps[i] = dict(drot=ang * q, dd=rr.uniform(0, .07) * q, dv=0., pitch=(rr.uniform(.12, .38) if rr.random() < .22 else 0.) * q)
        if i in jump: ps[i].update(dv=jump[i] * q, drot=sg * .16 * q, pitch=.10, dd=.06)
    _rowcels(c, T, items, L, lambda i: ps[i], 1 if jolt else 2, 2.6)
    if jolt:
        for i, a in jump.items():
            it = items[i]; speed(c, T(it.u + it.w / 2, .02, -.02), T(it.u + it.w / 2 + .02, a * .9 + .06, -.02), 7, it.w * 420, ['#241a26', '#3c3558'], 90 + i, 3.)
    img = over_roi(S.get('h'), c.rgba(), 0, 0); img = shade(img, S.get('hF'))
    for k in range(6): p = T(.5 + .24 * k, .004, .0); hall.dust_puff(img, p[0], p[1], 5 + f, 900 + k, 1.6, .85)
    off = [(14, 12), (-9, -7)][f] if jolt else (0, 0)
    return comp.cam2d(img, W / 2 + off[0], H / 2 + off[1], 1.03 if jolt else 1.02)


# ------------------------------------------------------------------ 5i  stillness
def cam_i(): return Camera3D((2.2, -1.6, 7.8), (-2.2, 7.6, .3), 1240, W, H, cx=W * .52, cy=H * .50)


FOX_I = hall.onb(hall.LEC['x'] - .62, .40, hall.pageh(.62) + .045)            # middle of the left page
LANT_I = np.array([hall.LEC['x'] + 1.92, hall.LEC['y'] - .56, 0.])              # on the floor, right of the desk


def plate_i():
    cam = cam_i(); hh = hall.Hall(cam, H, W, 2, 'high'); hall.assemble(hh, 'high'); hall.lectern(hh)
    img, pid, planes, boxes = hh.finish()
    page = cv2.GaussianBlur(np.isin(pid, hh.page_ids).astype(f32), (0, 0), .8)
    out, alb, pos, n, b, m = hall.light_plate(img, pid, planes, boxes, cam)
    W3 = LANT_I + np.array([.0, -.02, .20])
    v = W3 + np.array([0, 0, .13], f32) * ((pos[..., 2:3] < .16) & (n[..., 2:3] > .6)) + np.array([0, 0, .75], f32) * ((pos[..., 2:3] > .4) & (n[..., 2:3] > .6)) - pos; dist = np.sqrt((v * v).sum(-1)) + 1e-6
    ndl = np.clip((v * n).sum(-1) / dist, 0, 1); E = (.22 + .78 * ndl ** .7) * (ndl > 0) / (1 + (dist / .98) ** 2.1) * (1 - page)
    pp = (pos + n * .014).astype(f32); pp[E < .008] = 1e9
    vis = np.ones(E.shape, f32); hall._boxvis(pp, 0., 0., 0., W3.astype(np.float64), boxes, vis)
    vis = np.maximum(vis, .55 * (pos[..., 2] > .5) * (n[..., 2] > .6))
    E *= .10 + .90 * cv2.GaussianBlur(vis, (0, 0), 1.3)
    e = np.clip(E * 1.9, 0, 1.25); st = np.array([0, .05, .13, .27, .48, .78, 1.15]); lv = np.array([0, .035, .10, .23, .44, .76, 1.08]); hd = np.zeros_like(e)
    for a, l0, l1 in zip(st[1:], lv[:-1], lv[1:]): hd += (l1 - l0) * smooth(a * .86, a * 1.02, e)
    q = cv2.GaussianBlur((.5 * np.interp(e, st, lv) + .5 * hd).astype(f32), (0, 0), .9)
    out = out + warmalb(alb) * rgb('#ffb458') * np.array([1.5, 1.28, 1.0], f32) * q[..., None] * 1.85 * (1 - .5 * np.clip(out, 0, 1))
    out = comp.screen(np.clip(out, 0, 1), (np.minimum(b, 1.9)[..., None] * .33) * hall.MOON)
    # the pages breathe a little pale light; the fox stands in its own soft shadow
    q8 = cv2.resize((out * page[..., None]).astype(f32), (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    out = comp.screen(out, cv2.resize(cv2.GaussianBlur(q8, (0, 0), 3.5) * .22 + cv2.GaussianBlur(q8, (0, 0), 10) * .42, (W, H), interpolation=cv2.INTER_LINEAR))
    p, _ = cam.project(FOX_I); hf = float(cam.scale(FOX_I)) * .50
    sh = np.zeros((H, W), f32); cv2.ellipse(sh, (int(p[0] - .30 * hf), int(p[1] + 1)), (int(.72 * hf), int(.10 * hf)), 0, 0, 360, 1, -1, cv2.LINE_AA)
    out = out * (1 - .46 * cv2.GaussianBlur(sh, (0, 0), 3.2)[..., None] * np.array([1., .98, .86], f32))
    _save(f's5_{KEY}_i.npy', mottle(np.clip(out, 0, 1.1), .012, 3, .004).astype(np.float16))


class Shot:
    id = 's5'; n = 168
    finish = dict(hal_amt=.22, vig=.16, grain=.010)
    NEED = ['a', 'aF', 'b', 'bF', 'c', 'cb', 'd', 'dA', 'dE', 'e', 'f', 'fF', 'g', 'gF', 'h', 'hF', 'i']
    M5e = dict(x=1000, y=640, h=900)
    CUTS = [(0, 'a'), (34, 'b'), (66, 'c'), (92, 'd'), (116, 'e'), (138, 'f'), (143, 'g'), (148, 'h'), (153, 'i')]
    M5i = dict(x=0, y=0, h=0)                                                # filled below from cam_i

    def setup(self):
        for k in self.NEED:
            if not os.path.exists(os.path.join(CACHE, f's5_{KEY}_{k}.npy')): globals()['plate_' + k[0]]()
        self.c = {k: np.load(os.path.join(CACHE, f's5_{KEY}_{k}.npy')) for k in self.NEED}
        self.sa, self.sb, self.sf, self.sg, self.sh = scene_a(), scene_b(), scene_f(), scene_g(), scene_h()
        self.cc, self.cd = cam_c(), cam_d()

    def get(self, k): return self.c[k].astype(f32)

    def frame(self, f):
        a, k = [q for q in self.CUTS if f >= q[0]][-1]; g = f - a
        if k == 'e':
            if not hasattr(self, 'e0'):                                          # lead: plain soft hall, the page humps read as furniture
                self.e0 = comp.glow(comp.glow(s2.hall_bokeh().astype(f32) * .84, 1985, 1040, 620, '#ffb458', .34, 2.1), 2030, 1070, 210, '#ffdfa6', .30, 1.8)
            img = self.e0 * (1. - .10 * comp.ease(g / 22.))
            return foxact.draw(img.astype(f32), 's5e', g, self.M5e).astype(f32)
        if k == 'i':
            img = self.get('i'); fx_, fy_ = self.M5i['lantern_foot']; hl = self.M5i['lantern_h']
            c = Cel(80, 80, int(fx_) - 40, int(fy_) - 66); props.lantern(c, fx_, fy_, hl * .0038, lw=1.1); rg = c.rgba(); rg[int(fy_) + 1 - c.oy:] = 0
            img = over_roi(img, rg, c.ox, c.oy)
            img = foxact.draw(img, 's5i', g, self.M5i)
            lx, ly = self.M5i['lantern']
            img = s2.Shot.halo(img, lx, ly, 170, '#ffab52', .34, 2.2); img = s2.Shot.halo(img, lx, ly, 46, '#ffd48c', .50, 1.8)
            return img.astype(f32)
        return globals()['frame_' + k](self, g).astype(f32)


def _mi():
    c = cam_i(); p, z = c.project(FOX_I); h = float(c.scale(FOX_I)) * .50
    l = c.project(LANT_I)[0]; hl = float(c.scale(LANT_I)) * .50
    return dict(x=float(round(p[0])), y=float(round(p[1])), h=round(h, 1), draw_lantern=False, lantern_foot=(float(round(l[0])), float(round(l[1]))), lantern_h=round(hl, 1), lantern=(float(round(l[0])), float(round(l[1] - 43 * hl * .0038))))


Shot.M5i = _mi()


if __name__ == '__main__':
    import sys
    os.makedirs('qa/s5', exist_ok=True)
    import time
    for k in sys.argv[1:]: t = time.time(); globals()['plate_' + k](); print(k, 'plate %.1fs' % (time.time() - t))
