"""Cliff, rock planes, pines and grass of shot 1."""
import numpy as np, cv2, math
from scipy.spatial import cKDTree
from lp import W, H
from lp.paint import f32, rgb, mix, Canvas, arcs, jit_color, smooth, fbm, repaint
from lp.ink import spl, Cel
from lp import plants

ROCK = ['#1d2046', '#272b57', '#363c6a', '#50578a', '#7a80b2', '#a5abd3', '#c6cde8']


def rough(P, r, amp=1.1, step=9.):
    """hand-cut edge: subdivide a polygon and knock the points about, corners stay sharp"""
    P = np.asarray(P, float); out = []
    for a, b in zip(P, np.roll(P, -1, 0)):
        L = np.hypot(*(b - a)); n = max(1, int(L / step))
        t = np.arange(n) / n
        q = a + (b - a) * t[:, None]
        if n > 1: q[1:] += r.normal(0, amp, (n - 1, 2)) * np.array([1., .6])
        out.append(q)
    return np.vstack(out)


def rock(poly, seed=1, fv=((1060, .50), (1300, .2)), big=(), mids=46, chips=150, mist=None, pal=ROCK):
    """cliff painted as stacked split slabs: every slab has a lit left plane, a shadow plane and a worn top, big ones
    first, chips last, so the planes read before the detail.  fv: x -> base value.  big: (x, y_top, w, h, dv)."""
    r = np.random.default_rng(seed)
    P = spl(np.asarray(poly, float), closed=True, corners=range(len(poly)), n=4)
    x0, y0 = np.floor(P.min(0)).astype(int) - 8; x1, y1 = np.ceil(P.max(0)).astype(int) + 8
    h, w = y1 - y0, x1 - x0
    c = Cel(w, h, x0, y0, ss=2); c.fill(P, '#ffffff'); A = c.rgba()[..., 3]
    gx_, gy_ = np.meshgrid(np.arange(w), np.arange(h))
    A = np.clip(cv2.remap(A, (gx_ + fbm(h, w, (16, 5), (1, .7), seed) * 1.3).astype(f32), (gy_ + fbm(h, w, (16, 5), (1, .7), seed + 1) * 1.1).astype(f32), cv2.INTER_LINEAR), 0, 1)
    c = Cel(w, h, x0, y0, ss=2)
    lab = np.zeros((h, w), np.int32); ang = np.full((h, w), np.pi / 2, f32); k = [0]
    xs, vs = zip(*fv)
    FV = lambda x: float(np.interp(x, xs, vs))
    def col(v, j=.02):
        cc = plants.pick(pal, np.array([np.clip(v + r.normal(0, j), .03, 1)]))
        return jit_color(cc, r, .004, .012, .012)[0]
    def put(Q, v, an=np.pi / 2, amp=1.0):
        Q = rough(Q, r, amp); k[0] += 1
        c.fill(Q, col(v))
        q = np.round((Q - (x0, y0)) * 8).astype(np.int32)
        cv2.fillPoly(lab, [q], k[0], cv2.LINE_8, 3); cv2.fillPoly(ang, [q], float(an), cv2.LINE_8, 3)
        return Q
    # underpaint
    for xx in range(x0, x1, 12):
        c.fill([[xx - 1, y0], [xx + 13, y0], [xx + 13, y1], [xx - 1, y1]], col(FV(xx) * .8 + .02, .008))
    def slab(x, y, sw, sh, dv=0., amp=1.0, top=True):
        f = FV(x) + dv; con = .35 + 1.3 * min(.5, f)
        lean = r.normal(0, .05) * sh
        sp = r.uniform(.24, .52)                                                   # split between lit and shadow plane
        a = np.array([x - sw / 2, y + min(16, r.uniform(0, .10) * sh)]); b = np.array([x - sw / 2 + sw * sp, y - min(7, r.uniform(.01, .05) * sh)]); d = np.array([x + sw / 2, y + min(26, r.uniform(.04, .20) * sh)])
        e = d + [lean + r.normal(0, .06) * sw, sh]; g = b + [lean + r.normal(0, .12) * sw, sh * r.uniform(.96, 1.08)]; q = a + [lean - r.uniform(0, .12) * sw, sh]
        tp = min(7., sw * .09)
        put([b, d, e, g], f - .055 * con + r.normal(0, .014), np.pi / 2 + .05, amp)
        put([a, b, g, q], f + .20 * con + r.normal(0, .03) * con, np.pi / 2 - .08, amp)
        if top and sw > 13:
            m = b + [r.uniform(.1, .3) * sw, tp * 1.1]
            put([a, b, d, m], f + (.34 if sw < 60 else .20) * con, .1, amp * .6)
        # chipped arris, dark seam down the split, undercut
        n = max(3, int(sh / 26))
        for _ in range(n):
            t0 = r.uniform(0, .85); t1 = min(1., t0 + r.uniform(.05, .22))
            L = np.array([b + (g - b) * t for t in np.linspace(t0, t1, 4)]) + r.normal(0, .5, (4, 2))
            if r.random() < .5: c.line(L - [.8, 0], r.uniform(.8, 1.6), col(f + .36 * con), (.2, .5), .5, seed=k[0] + _, alpha=r.uniform(.35, .85))
        for _ in range(r.integers(1, 3)):
            t0 = r.uniform(0, .6); t1 = min(1., t0 + r.uniform(.2, .5))
            off = r.uniform(.2, .8)
            L = np.array([(b + (g - b) * t) * (1 - off) + (d + (e - d) * t) * off for t in np.linspace(t0, t1, 6)]) + np.cumsum(r.normal(0, .7, (6, 2)), 0)
            c.line(L, r.uniform(.9, 2.2), col(max(.03, f - .20 * con)), (.15, .7), .5, seed=k[0] + 40 + _, alpha=r.uniform(.4, .85))
        c.line(np.array([d, d + (e - d) * .5, e]) + [1.0, 0], r.uniform(1.4, 2.6), col(max(.03, f - .20)), (.05, .6), .4, seed=k[0], alpha=.75)
    for (x, y, sw, sh, dv) in big: slab(x, y, sw, sh, dv, 1.5)
    ys, xs_ = np.nonzero(A > .5)
    def spots(n, ylo, yhi):
        sel = (ys + y0 > ylo) & (ys + y0 < yhi); idx = np.flatnonzero(sel); q = r.choice(idx, n)
        S = np.c_[xs_[q] + x0, ys[q] + y0].astype(float); return S[np.argsort(S[:, 1])]
    yt = P[:, 1].min(); yb = P[:, 1].max()
    for (x, y) in spots(mids, yt - 6, yt + (yb - yt) * .72): slab(x, y, r.uniform(22, 52), r.uniform(48, 120), r.normal(0, .02), 1.1)
    for (x, y) in spots(chips, yt, yt + (yb - yt) * .8):
        # detail gathers on the lit side, the shadow side stays quiet
        if r.random() > .25 + 1.5 * FV(x): continue
        slab(x, y, r.uniform(7, 20), r.uniform(14, 46), r.normal(0, .025), .6)
    f = c.rgba()
    colr = f[..., :3] / np.maximum(f[..., 3:4], 1e-3)
    yy, xx = np.mgrid[0:h, 0:w].astype(f32); X = xx + x0; Y = yy + y0
    hue = fbm(h, w, (120, 50), (1, .5), seed + 12)[..., None]
    colr = np.clip(colr * (1 + hue * np.array([.030, -.006, -.016], f32)), 0, 1).astype(f32)
    out = repaint(colr, (6.0, 3.0), seed=seed, angle=ang + fbm(h, w, (40,), None, seed + 2) * .1, lab=lab, mask=(A > .4).astype(f32), thr=(0, .02), density=1.15, aspect=4.0, jit=(.003, .010, .018), op=.75, dry=.5, curve=.2, bristle=.08)
    colr = colr * .3 + out * .7
    a = A
    if mist is not None:
        my0, my1, mc = mist
        n = fbm(h, w, (90, 34), (1, .5), seed + 20, aspect=3.4)
        t = smooth(my0, my1, Y + n * (my1 - my0) * .07)
        colr = colr + (rgb(mc) - colr) * (t[..., None] ** 1.1) * .97
        a = A * (1 - smooth(my1 - (my1 - my0) * .25, my1 + 4, Y + n * 8))
    L = np.zeros((H, W, 4), f32)
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    sl = (slice(Y0 - y0, Y1 - y0), slice(X0 - x0, X1 - x0))
    L[Y0:Y1, X0:X1] = np.dstack([colr * a[..., None], a])[sl]
    return L


def ledge(cv, pts, seed=0, w=3.0, v=.8, tufts=True):
    """a grassy ledge: pale worn top, dark underside, hanging grass"""
    r = np.random.default_rng(seed); c = spl(np.asarray(pts, float), n=8)
    for k in range(0, len(c) - 5, 3):
        q = c[k:k + 5] + r.normal(0, .7, (5, 2))
        cv.strokes(q[None] + [1.5, w * 1.3], w * r.uniform(.7, 1.2), plants.pick(ROCK, np.array([.14 + r.normal(0, .03)])), op=.7, seed=seed + k, shape='round')
        cv.strokes(q[None], w * r.uniform(.45, .9), plants.pick(ROCK, np.array([v + r.normal(0, .07)])), op=.9, seed=seed + k + 400, shape='round', dry=.15)
        if tufts and r.random() < .7:
            plants.tuft(cv, q[2, 0], q[2, 1] + 1, int(r.integers(4, 9)), r.uniform(5, 12), seed + k, spread=9, v=.30 + .3 * r.random())


CLIFF = [(1102, 712), (1118, 704), (1150, 702), (1250, 704), (1350, 702), (1404, 702), (1436, 708), (1462, 706), (1492, 718), (1516, 742), (1548, 748), (1574, 772), (1584, 806), (1622, 820), (1650, 846), (1694, 856), (1738, 896), (1790, 930), (1810, 1000), (1000, 1000), (1022, 962), (1010, 930), (1042, 900), (1050, 872), (1036, 846), (1060, 812), (1058, 786), (1080, 764), (1078, 736)]


def cliff():
    fv = [(1020, .66), (1080, .60), (1112, .54), (1140, .42), (1172, .32), (1230, .22), (1330, .17), (1362, .26), (1392, .28), (1420, .17), (1468, .26), (1502, .24), (1535, .14), (1600, .24), (1680, .16)]
    big = [(1480, 704, 96, 240, 0.), (1300, 698, 150, 290, 0.), (1216, 700, 92, 280, .01), (1390, 700, 70, 260, .02), (1140, 700, 84, 280, .03), (1096, 716, 60, 270, .03), (1556, 752, 74, 220, 0.), (1620, 822, 80, 160, 0.),
           (1064, 792, 52, 190, .02), (1178, 772, 78, 210, .02), (1262, 790, 96, 190, 0.), (1352, 780, 88, 190, .02), (1438, 786, 76, 180, 0.), (1500, 800, 70, 170, .02), (1110, 842, 70, 140, .02), (1230, 856, 80, 120, .01), (1400, 862, 90, 110, .01)]
    return rock(CLIFF, seed=14, fv=fv, big=big, mist=(806, 985, '#6b73b4'))


def dressing():
    """everything that grows on the cliff (layer Canvas)"""
    cv = Canvas(H, W, layer=True)
    r = np.random.default_rng(31)
    # small growth on the tops of the slabs
    for k in range(46):
        x = r.uniform(1060, 1640); y = r.uniform(736, 870)
        if x > 1500: y = max(y, 720 + (x - 1490) * .95)
        v = .60 - .38 * float(smooth(1080, 1180, x)) + r.normal(0, .04)
        if r.random() < .55:
            plants.tuft(cv, x, y, int(r.integers(3, 7)), r.uniform(5, 10), 2000 + k, spread=5, v=v, lean=.35)
        else:
            n = int(r.integers(8, 26)); a = r.uniform(0, 2 * np.pi, n); q = np.sqrt(r.random(n)) * r.uniform(4, 9)
            plants.ivy(cv, x + np.cos(a) * q * 1.3, y + np.sin(a) * q * .8 + q * .4, np.clip(v * .55 - .03 * np.sin(a) * q + r.normal(0, .06, n), 0, 1), seed=k, size=2.1)
    # boulders and rubble at the foot of the wall
    for _ in range(64):
        x = r.uniform(1112, 1500); y = 708 + r.uniform(-2, 8) + max(0, (x - 1440) * .16)
        s = r.uniform(2.5, 8)
        v = .62 - .42 * smooth(1130, 1260, x) + r.normal(0, .05)
        P = arcs(np.array([x]), np.array([y]), np.array([r.uniform(-.3, .3)]), np.array([s * 2.0]), np.array([0.]), K=4)
        cv.strokes(P, s * .62, plants.pick(ROCK, np.array([max(.1, v - .2)])), seed=_, shape='round')
        cv.strokes(P + np.array([-s * .22, -s * .26]), s * .34, plants.pick(ROCK, np.array([v + .12])), seed=_ + 99, shape='round', op=.9)
    # turf along the brink
    xs = np.r_[np.arange(1104, 1500, 5.5), np.arange(1500, 1660, 9)]
    for i, x in enumerate(xs):
        y = np.interp(x, [1100, 1120, 1440, 1492, 1516, 1548, 1574, 1584, 1622, 1650], [714, 709, 711, 722, 746, 752, 776, 810, 824, 850])
        v = .58 - .34 * smooth(1120, 1230, x)
        plants.tuft(cv, x + r.normal(0, 3), y + r.uniform(0, 5), int(r.integers(6, 13)), r.uniform(7, 19) * (1.35 if r.random() < .12 else 1), 1000 + i, spread=11, v=v, lean=.35)
    # low shrubs against the wall
    for (x, y, rx, ry, v) in ((1176, 704, 22, 12, .5), (1226, 706, 17, 9, .34), (1292, 706, 24, 11, .26), (1352, 704, 26, 15, .24), (1398, 704, 22, 17, .28), (1432, 708, 20, 11, .26)):
        n = int(rx * ry * .9); a = r.uniform(0, np.pi, n); q = np.sqrt(r.random(n))
        X = x + np.cos(a) * rx * q; Y = y - np.sin(a) * ry * q
        V = np.clip(v * .4 + .5 * v * (Y < y - ry * .45) - .25 * (X - x) / rx * v + r.normal(0, .07, n), 0, 1)
        plants.ivy(cv, X, Y, V, seed=int(x), size=2.5)
    return cv


def pines():
    cv = Canvas(H, W, layer=True)
    # far small ones behind the shoulder
    plants.pine(cv, [(1452, 716, 7), (1456, 694, 5.5), (1466, 676, 4), (1480, 664, 2.5)], [(1462, 690, 17, 7, .45), (1486, 676, 22, 8, .8), (1476, 660, 18, 8, 1.0), (1494, 662, 14, 6, 1.0)], seed=12, needle=6, flag=.55)
    # right shoulder, flung by the wind
    plants.pine(cv, [(1500, 730, 12), (1508, 702, 10), (1524, 682, 8), (1550, 670, 6.5), (1572, 650, 4.5), (1590, 640, 3)],
                [(1520, 694, 20, 8, .35), (1546, 682, 30, 10, .55), (1590, 672, 34, 11, .7), (1562, 652, 28, 11, .85), (1604, 648, 30, 10, .95), (1586, 632, 24, 11, 1.0), (1622, 662, 18, 7, .9)], seed=3, needle=7.5, flag=.6)
    # on the lower ledge, half in the mist
    plants.pine(cv, [(1610, 822, 8), (1616, 800, 6.5), (1630, 786, 5), (1646, 776, 3)], [(1622, 798, 17, 7, .4), (1642, 786, 24, 8, .7), (1658, 774, 22, 8, 1.0), (1640, 768, 15, 7, 1.)], seed=8, needle=6, flag=.6)
    # hero pine on the moon-side brink: leans out over the void, crown swept back by the wind
    plants.pine(cv, [(1128, 722, 19), (1104, 704, 15), (1076, 694, 13), (1050, 676, 11), (1040, 650, 9.5), (1050, 624, 8), (1072, 604, 6), (1090, 586, 4), (1104, 574, 2.4)],
                [(1010, 676, 30, 10, .40), (1040, 660, 24, 9, .46), (1022, 642, 34, 11, .55), (1078, 652, 30, 10, .58), (1096, 632, 36, 12, .68), (1048, 620, 32, 12, .70), (1082, 606, 30, 12, .82), (1120, 612, 28, 10, .86), (1100, 586, 30, 13, .95), (1124, 586, 24, 9, 1.0), (1108, 568, 22, 11, 1.0)],
                seed=5, needle=9, flag=.45)
    # a stubborn one out of the cliff face
    plants.pine(cv, [(1070, 798, 9), (1052, 792, 7.5), (1034, 780, 6), (1026, 762, 4.5), (1032, 748, 3)], [(1014, 786, 20, 7, .45), (1030, 768, 24, 8, .65), (1048, 758, 20, 8, .85), (1034, 744, 22, 9, 1.0)], seed=9, needle=7, flag=.4)
    return cv
