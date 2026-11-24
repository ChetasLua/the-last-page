"""s4a set: the enormous open book on its carved lectern, dark hall behind.  Owner: ink lane."""
import math, numpy as np, cv2
from lp import W, H, comp, books as bk, sheet3d as S3
from lp.paint import rgb, mix, smooth, fbm, ramp, Canvas, arcs, repaint, jit_color, f32, lum, grid, tooth, mottle
from lp.ink import Cel, spl, ribbon, xf, resample, noise1
from lp.cam import Camera3D
from lp.brush import fibre_tile, tile_to

MOON, LAMP = rgb('#9fb4e0'), rgb('#ffb458')
EYE = np.array([0., 0., 1.5]); F = 2400.; CX, CY = 1230, 290
Y0, Z0, SL = 4.75, .70, math.radians(17)
XG, XLE, XRE = .25, -.89, 1.39
DEP = .92
CS, SN = math.cos(SL), math.sin(SL)
WOOD = rgb('#7c5a3c'); WOOD_D = rgb('#5c4334'); STONE = rgb('#95928f')


_over = {}


def cam():
    e = np.asarray(_over.get('eye', EYE), float)
    return Camera3D(e, e + (0, 1, 0), F, cx=_over.get('cx', CX), cy=_over.get('cy', CY))


def pj(*p): return cam().project(np.asarray(p, float))[0]


def onb(X, t, h=0.):
    """point on the sloping book board"""
    X, t, h = np.broadcast_arrays(*[np.asarray(v, float) for v in (X, t, h)])
    return np.stack([X, Y0 + t * CS - h * SN, Z0 + t * SN + h * CS], -1)


def prof(d, side=-1):
    top = .118 if side < 0 else .132
    return .030 + (top - .030) * (1 - np.exp(-(d / .050) ** 1.35)) - .030 * smooth(.14, 1.14, d)


# ------------------------------------------------------------------- light (screen-space, painted)
def lampf(x, y): return np.clip(1 - np.hypot((np.asarray(x, float) - 2150) / 1330., (np.asarray(y, float) - 800) / 800.), 0, 1) ** 1.35
def light(kind):
    def fn(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        b = lampf(x, y)[..., None]
        lf = np.clip(1.1 - x / 1500., 0, 1)[..., None]
        A = np.array([.265, .25, .365]) * (1 - .45 * b)
        if kind == 'front': return A * .92 + lf * np.array([.07, .09, .145]) + b * np.array([1.26, .80, .25])
        if kind == 'top': return A * 1.10 + lf * np.array([.10, .125, .19]) + b * np.array([1.32, .86, .29])
        if kind == 'right': return A * .85 + b * np.array([1.62, 1.04, .34])
        if kind == 'left': return A * .80 + lf * np.array([.24, .30, .44])
        return A * .6 + b * np.array([.3, .18, .08])
    return bk.Light(fn, '#15122a', 1 if kind != 'left' else -1)


# ------------------------------------------------------------------- dark hall
def backdrop():
    C = cam()
    P = bk.Painter(H, W, ss=1, base='#1f1d36')
    def fn(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        a = np.clip(1.15 - np.abs(x - 500) / 900., 0, 1) * np.clip(1.1 - np.abs(y - 420) / 700., .2, 1)
        b = np.clip(1 - np.hypot((x - 2150) / 900., (y - 700) / 560.), 0, 1) ** 1.4
        c = np.zeros(np.shape(x) + (3,)) + np.array([.34, .325, .46])
        return c + a[..., None] * np.array([.30, .36, .55]) + b[..., None] * np.array([.95, .55, .22])
    air = lambda x, y: (np.array([.150, .155, .290]), np.clip(.46 + 0 * np.asarray(x, float), 0, 1))
    air2 = lambda x, y: (np.array([.150, .155, .290]), np.clip(.26 + 0 * np.asarray(x, float), 0, 1))
    L = bk.Light(fn, '#131126', -1, air); L2 = bk.Light(fn, '#131126', 1, air2); L3 = bk.Light(fn, '#131126', -1, air2)
    T = bk.plane(C, (-8., 13., 0), (1, 0, 0), (0, 0, 1), (0, -1, 0))
    rr = [.40, .37, .36, .34, .34, .33]
    for i in range(9):
        for k in range(2):
            bk.paint_bay(P, T, i * 1.6, i * 1.6 + 1.6, k * 2.5 + .10, rr, L, 400 + i * 3 + k, post=.14, board=.045, book_kw=dict(messy=.45), empty=.08)
    # gallery: beam, rail, posts
    def quad(a, b, c, d, col, LL, n=3, **kw):
        t = np.linspace(0, 1, 7)[:, None]; a, b, c, d = [np.asarray(q, float) for q in (a, b, c, d)]
        P.strip(C.project(a + (b - a) * t)[0], C.project(d + (c - d) * t)[0], col, LL, n=n, seed=int(abs(a[0] * 97 + a[2] * 31 + b[0] * 7)) + 3, **kw)
    for X in np.arange(-8, 6.4, 1.6):
        quad((X, 12.1, 2.42), (X + 1.6, 12.1, 2.42), (X + 1.6, 12.1, 2.16), (X, 12.1, 2.16), WOOD * .8, L3, 4, hi=.35, lo=-.4)
        quad((X, 12.0, 2.50), (X + 1.6, 12.0, 2.50), (X + 1.6, 12.0, 2.42), (X, 12.0, 2.42), WOOD * 1.1, L3, 2, hi=.4, lo=-.3)
        quad((X - .09, 12.05, 2.2), (X + .09, 12.05, 2.2), (X + .07, 12.05, 0), (X - .07, 12.05, 0), WOOD * .75, L3, 3, hi=.3, lo=-.4)
        quad((X - .16, 12.0, 2.2), (X + .16, 12.0, 2.2), (X + .10, 12.0, 1.95), (X - .10, 12.0, 1.95), WOOD * .9, L3, 2, hi=.3)
        quad((X, 12.0, 3.2), (X + 1.6, 12.0, 3.2), (X + 1.6, 12.0, 3.1), (X, 12.0, 3.1), WOOD * .9, L3, 2, hi=.4)
        for j in range(8):
            u = X + .1 + j * .2; quad((u - .03, 12.0, 3.1), (u + .03, 12.0, 3.1), (u + .035, 12.0, 2.5), (u - .035, 12.0, 2.5), WOOD * .8, L3, 1, lo=-.3)
    P.flush()
    # right-hand case coming toward us
    Tr = bk.plane(C, (2.95, 12.2, 0), (0, -1, 0), (0, 0, 1), (-1, 0, 0))
    for i in range(4): bk.paint_bay(P, Tr, i * 1.7, i * 1.7 + 1.7, .10, [.42, .38, .37, .35, .34, .33, .33], L2, 470 + i, post=.16, board=.05, book_kw=dict(messy=.6), empty=.05)
    Tl = bk.plane(C, (-4.6, 6.4, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0))
    for i in range(4): bk.paint_bay(P, Tl, i * 1.6, i * 1.6 + 1.6, .10, [.42, .38, .37, .35, .34, .33, .33], L3, 490 + i, post=.16, board=.05, book_kw=dict(messy=.6), empty=.05)
    im = P.image()
    return im


def floor(P):
    C = cam(); r = np.random.default_rng(31)
    L = light('top')
    X = -7.
    while X < 5.2:
        w = r.uniform(.17, .24); Y = r.uniform(2.6, 4.0)
        while Y < 13.:
            ln = r.uniform(1.3, 3.0); ya, yb = Y, min(13., Y + ln)
            pa, _ = C.project(np.array([[X + .006, ya + .004, 0], [X + .006, yb - .004, 0]])); pb, _ = C.project(np.array([[X + w - .006, ya + .004, 0], [X + w - .006, yb - .004, 0]]))
            if max(pa[:, 0].max(), pb[:, 0].max()) > -40 and min(pa[:, 0].min(), pb[:, 0].min()) < W + 40 and pa[:, 1].max() > 560:
                col = mix(WOOD, rgb('#675a5e'), r.uniform(.1, .65)) * r.uniform(.78, 1.10)
                t = np.linspace(0, 1, 7)[:, None]
                # sample evenly on screen, not in depth
                A = pa[0] + (pa[1] - pa[0]) * t; B = pb[0] + (pb[1] - pb[0]) * t
                P.strip(A, B, col, L, n=max(4, int(np.hypot(*(pa[0] - pb[0])) / 9)), val=.055, hi=.16, lo=-.28, seed=int(r.integers(1 << 30)), op=.6, seg=(.25, .9), cover=1.1)
            Y = yb
        X += w
    P.flush()


def set_layer():
    """step, lectern, stacks, candles, leaves (book pages come after).  premult RGBA + ink accents"""
    C = cam(); r = np.random.default_rng(77)
    P = bk.Painter(H, W, ss=2, base='#000000', layer=True)
    Lf, Lt, Lr, Ll = light('front'), light('top'), light('right'), light('left')

    def quad(a, b, c, d, col, L, n=4, **kw):
        t = np.linspace(0, 1, 7)[:, None]; a, b, c, d = [np.asarray(q, float) for q in (a, b, c, d)]
        kw.setdefault('seed', int(r.integers(1 << 30)))
        return P.strip(C.project(a + (b - a) * t)[0], C.project(d + (c - d) * t)[0], col, L, n=n, **kw)

    # ---- stone step: two courses of worn blocks
    sx0, sx1, sy0, sy1, sz = -1.10, 2.6, 4.52, 6.4, .115
    for k, (ins, z0, z1) in enumerate(((0, 0, sz * .52), (.06, sz * .52, sz))):
        quad((sx0 + ins, sy1, z0), (sx0 + ins, sy0 + ins, z0), (sx0 + ins, sy0 + ins, z1), (sx0 + ins, sy1, z1), STONE * .7, Ll, 3)
        X = sx0 + ins
        while X < sx1:
            w = r.uniform(.42, .72); a = X + .004; b = min(X + w - .004, sx1)
            quad((a, sy0 + ins, z1), (b, sy0 + ins, z1 + r.normal(0, .002)), (b, sy0 + ins, z0), (a, sy0 + ins, z0), STONE * r.uniform(.78, 1.0), Lf, 4, val=.06, hi=.55, lo=-.35, wob=.7)
            X += w
        yy = [sy0 + ins, sy0 + ins + .34, sy0 + ins + .95, sy1] if k else [sy0, sy0 + .09]
        for ya, yb in zip(yy[:-1], yy[1:]):
            X = sx0 + ins - r.uniform(0, .2)
            while X < sx1:
                w = r.uniform(.5, .8); a = max(X + .004, sx0 + ins); b = min(X + w - .004, sx1)
                if b > a: quad((a, yb, z1), (b, yb, z1), (b, ya, z1), (a, ya, z1), STONE * r.uniform(.62, .78), Lt, 5, val=.055, lo=.30, wob=.5)
                X += w
    P.flush()
    # ---- lectern body
    by = 4.98; x0, x1 = -.74, 1.26; zt, zb = .60, sz
    # side wall facing left, seen a little
    P.glaze(C.project(np.array([(x0 - .62, by - .16, sz), (x1 + .1, by - .22, sz), (x1 + .12, by + .6, sz), (x0 - .75, by + .6, sz)]))[0], np.array([.46, .42, .55]), 1., soft=9)
    P.glaze(C.project(np.array([(x0 - .16, by - .19, sz), (x1 + .12, by - .19, sz), (x1 + .12, by + .6, sz), (x0 - .2, by + .6, sz)]))[0], np.array([.45, .40, .50]), 1., soft=4)
    P.poly(C.project(np.array([(x0, by, zb), (x1, by, zb), (x1, by, zt), (x0, by, zt)]))[0], Lf(WOOD_D * .7, 1230., 800.)[0])
    # boards behind the arcade
    for X in np.arange(x0, x1, .125): quad((X, by, zt), (X + .12, by, zt), (X + .12, by, zb), (X, by, zb), WOOD_D * r.uniform(.6, .78), Lf, 4, val=.08, hi=.1, lo=-.3)
    # plinth, three mouldings
    for (z0, z1, y, e, k) in ((sz, sz + .050, by - .13, .10, .95), (sz + .050, sz + .072, by - .115, .085, 1.35), (sz + .072, sz + .098, by - .09, .06, .85), (sz + .098, sz + .112, by - .065, .04, 1.25)):
        quad((x0 - e, y, z1), (x1 + e, y, z1), (x1 + e, y, z0), (x0 - e, y, z0), WOOD * k, Lf, 4, val=.07, hi=.40, lo=-.40)
    # pilasters (turned, with rings) and pointed arches with carved spandrels
    cols = np.linspace(x0 + .05, x1 - .05, 5)
    bz, tz = sz + .112, zt - .02
    for i in range(4):
        a, b = cols[i] + .055, cols[i + 1] - .055; m = (a + b) / 2
        # recessed panel with its own shadowed reveal
        arch = [(a, bz), (a, bz + .20)] + [(m + (a - m) * math.cos(q) ** .85, bz + .20 + .155 * math.sin(q) ** 1.15) for q in np.linspace(.1, math.pi / 2 - .02, 7)]
        arch += [(m + (b - m) * math.cos(q) ** .85, bz + .20 + .155 * math.sin(q) ** 1.15) for q in np.linspace(math.pi / 2 - .02, .1, 7)] + [(b, bz + .20), (b, bz)]
        ar = np.array(arch); q3 = np.c_[ar[:, 0], np.full(len(ar), by - .05), ar[:, 1]]
        # spandrel wall in front of the recess
        quad((a - .06, by - .05, tz), (b + .06, by - .05, tz), (b + .06, by - .05, bz), (a - .06, by - .05, bz), WOOD * .80, Lf, 8, val=.08)
        pq = C.project(q3)[0]
        lab = P.poly(pq, Lf(WOOD_D * .46, *pq.mean(0))[0])
        for X in np.linspace(a + .02, b - .02, 5):
            p0, p1 = C.project(np.array([(X, by, bz), (X + r.normal(0, .004), by, bz + .34)]))[0]
            P.q(np.linspace(p0, p1, 5), r.uniform(6, 11), Lf(WOOD_D * r.uniform(.5, .72), *((p0 + p1) / 2))[0], .85, .3, lab)
        P.flush()
        # a carved sprig in the panel, different in each
        mc = C.project(np.array([m, by - .01, bz + .045]))[0]
        for (dy, k_, w_) in ((1.8, .40, 2.4), (0, 1.0, 1.8)):
            P.q(np.array([mc + [0, dy], mc + [3 * (-1) ** i, -34 + dy], mc + [-2, -66 + dy], mc + [0, -92 + dy]]), w_, Lf(WOOD_D * k_, *mc)[0], .8, .2, lab)
            for j in range(6):
                sd = 1 if j % 2 else -1; b0 = mc + [0, -18 - 13 * j + dy]; an = -1.1 - .1 * (i % 2); ln = (30 - 2.6 * j) * (1 + .12 * ((i * 7 + j * 3) % 5 - 2))
                P.q(np.stack([b0, b0 + [sd * ln * .55, -ln * .30], b0 + [sd * ln, -ln * (.62 + .1 * (i % 3))]]), 5.6 - .5 * j, Lf(WOOD_D * k_ * r.uniform(.9, 1.15), *b0)[0], .85, .15, lab, 'leaf')
        P.flush()
        # light catches the lamp-side jamb of the recess
        P.q(C.project(np.linspace(q3[0], q3[1], 5))[0], 3.2, Lr(WOOD * .75, *pq[0])[0], .9, .2, lab)
        # moulded archivolt: two passes
        o1 = spl(pq[1:-1], n=5); t = resample(o1, 9.); seg = [t[j:j + 5] for j in range(0, len(t) - 4, 4)]
        for s in seg: P.q(s, 5.2, Lf(WOOD * r.uniform(1.0, 1.2), *s.mean(0))[0], .95, .1)
        for s in seg: P.q(s + [0, -3.2], 1.6, Lf(WOOD * 1.75, *s.mean(0))[0], .8, .2)
        for s in seg: P.q(s + [0, 4.2], 1.4, Lf(WOOD_D * .5, *s.mean(0))[0], .8, .2)
        P.flush()
    for X in cols:
        zs = [(bz, .060), (bz + .022, .062), (bz + .030, .045), (bz + .06, .036), (bz + .075, .047), (bz + .09, .034), (bz + .20, .030), (bz + .28, .037), (bz + .295, .050), (bz + .305, .038), (tz - .05, .046), (tz - .03, .066), (tz, .066)]
        for (za, wa), (zc, wc) in zip(zs[:-1], zs[1:]):
            y = by - .10
            L3 = C.project(np.array([(X - wa, y, za), (X - wc, y, zc)]))[0]; R3 = C.project(np.array([(X + wa, y, za), (X + wc, y, zc)]))[0]
            P.strip(L3, R3, WOOD * r.uniform(.92, 1.1), Lf, n=4, val=.06, hi=-.38, lo=.55, seed=int(r.integers(1 << 30)), seg=(.8, .98))
        # spiral fluting on the shaft
        for z in np.arange(bz + .098, bz + .275, .020):
            p = C.project(np.array([(X - .031, by - .101, z), (X, by - .104, z + .008), (X + .031, by - .101, z + .022)]))[0]
            P.q(spl(p, n=2)[:5], 1.7, Lf(WOOD_D * .55, *p[1])[0], .85, .15); P.q(spl(p, n=2)[:5] + [0, -2.6], 1.1, Lf(WOOD * 1.5, *p[1])[0], .7, .2)
    P.flush()
    # frieze under the desk: a carved vine
    quad((x0 - .05, by - .12, zt + .055), (x1 + .05, by - .12, zt + .055), (x1 + .05, by - .12, zt - .03), (x0 - .05, by - .12, zt - .03), WOOD * .92, Lf, 5, val=.06, hi=.3, lo=-.45)
    # ---- sloping desk: underside lip, front moulding, top board
    e0, e1 = XLE - .17, XRE + .17
    f = lambda X, t, h: onb(X, t, h)
    for (t, h0, h1, k) in ((-.12, -.055, -.018, .85), (-.135, -.018, .004, 1.35), (-.12, .004, .040, 1.0), (-.112, .040, .050, 1.5)):
        quad(f(e0, t, h1), f(e1, t, h1), f(e1, t, h0), f(e0, t, h0), WOOD * k, Lf, 4, val=.07, hi=.45, lo=-.4)
    quad(f(e0, -.08, .050), f(e1, -.08, .050), f(e1, -.112, .050), f(e0, -.112, .050), WOOD * 1.25, Lt, 2)
    for a, b in zip(np.linspace(e0, e1, 5)[:-1], np.linspace(e0, e1, 5)[1:]):
        for (t0, t1) in ((-.08, .30), (.30, .66), (.66, 1.04)): quad(f(a, t1, 0), f(b, t1, 0), f(b, t0, 0), f(a, t0, 0), WOOD * r.uniform(.85, 1.05), Lt, 6, val=.07)
    P.flush()
    # egg-and-dart along the front moulding, each carved by hand
    for X in np.arange(e0 + .03, e1 - .02, .052):
        p = C.project(f(X, -.136, -.036))[0]; s = F / 4.7
        o = np.c_[np.cos(np.linspace(0, 2 * np.pi, 9)[:-1]) * .017 * s, np.sin(np.linspace(0, 2 * np.pi, 9)[:-1]) * .014 * s] * r.uniform(.9, 1.1)
        P.poly(p + o, Lf(WOOD * r.uniform(.95, 1.15), *p)[0]); P.q(np.array([p + [2, -4], p + [5, -1], p + [5, 3]]), 1.5, Lf(WOOD * 1.8, *p)[0], .8); P.q(np.array([p + [-5, -3], p + [-5, 2], p + [-2, 5]]), 1.6, Lf(WOOD_D * .45, *p)[0], .8)
        q = C.project(f(X + .026, -.136, -.036))[0]; P.q(np.array([q + [0, -6], q + [0, 6]]), 1.5, Lf(WOOD * 1.3, *q)[0], .8, 0, -1, 'taper')
    P.flush()
    # vine on the frieze
    vz = zt + .012; t = np.linspace(x0, x1, 90); wv = .022 * np.sin(t * 19)
    vp = C.project(np.c_[t, np.full(90, by - .125), vz + wv])[0]
    for j in range(0, 86, 4):
        s = vp[j:j + 5]; P.q(s + [0, 2.2], 2.6, Lf(WOOD_D * .42, *s.mean(0))[0], .85, .1); P.q(s, 2.1, Lf(WOOD * 1.35, *s.mean(0))[0], .9, .1)
    for j in range(3, 88, 5):
        p = vp[j]; sd = 1 if (j // 5) % 2 else -1; an = sd * r.uniform(.6, 1.2) + r.uniform(-.2, .2)
        for (dd, cl, w_) in ((2.0, WOOD_D * .42, 5.5), (0, WOOD * r.uniform(1.1, 1.5), 4.6)):
            P.q(np.stack([p + [0, dd], p + [math.cos(an) * 9, math.sin(an) * 9 + dd], p + [math.cos(an) * 20, math.sin(an) * 17 + dd]]), w_, Lf(cl, *p)[0], .92, 0, -1, 'leaf')
    P.flush()
    for _ in range(110):
        X = r.uniform(x0 - .1, x1 + .1); z = r.uniform(tz - .02, zt + .05) if r.random() < .5 else r.uniform(sz, bz); p = C.project(np.array([X, by - .12, z]))[0]
        P.q(arcs([p[0]], [p[1]], [r.normal(0, .35)], [r.uniform(8, 46)], [r.normal(0, .3)])[0], r.uniform(.5, 1.5), Lf(WOOD * (r.uniform(1.4, 1.9) if r.random() < .5 else r.uniform(.35, .55)), *p)[0], r.uniform(.2, .55), .4)
    for _ in range(60):
        X = r.uniform(e0, e1); p = C.project(f(X, -.136, r.choice([-.018, .004, .045])))[0]
        P.q(arcs([p[0]], [p[1]], [0.], [r.uniform(14, 70)], [0.])[0], r.uniform(.8, 1.8), Lf(WOOD * r.uniform(1.6, 2.1), *p)[0], r.uniform(.3, .7), .5)
    P.flush()
    # end brackets: carved, hollow underneath, with a scroll cut in the face
    for (X, sd) in ((x0 + .01, -1), (x1 - .01, 1)):
        pts = np.array([(-.02, .05), (.20, .05), (.205, .02), (.17, -.015), (.115, -.04), (.07, -.09), (.05, -.15), (.03, -.175), (-.02, -.18)])
        q = C.project(np.c_[X + sd * pts[:, 0], np.full(len(pts), by - .09), zt - .02 + pts[:, 1]])[0]
        o = spl(q, closed=True, corners=(0, 1, 2, 7, 8), n=6)
        L_ = Lf
        lab = P.poly(o, L_(WOOD * .88, *q.mean(0))[0])
        for _ in range(16):
            p = q.mean(0) + r.normal(0, 22, 2); P.q(arcs([p[0]], [p[1]], [sd * -.8 + r.normal(0, .2)], [r.uniform(20, 50)], [r.normal(0, .5)])[0], r.uniform(3, 7), L_(WOOD * r.uniform(.8, 1.0), *p)[0], .7, .25, lab)
        ed = spl(q[2:8], n=6)
        for j in range(0, len(ed) - 4, 4): P.q(ed[j:j + 5], 2.2, Lf(WOOD * .45, *ed[j])[0], .9, .1, lab)
        t = np.linspace(0, 4.4 * math.pi / 2, 40); rr_ = 25 * np.exp(-t * .26); cc = C.project(np.array([X + sd * .085, by - .09, zt - .012]))[0]
        sp_ = np.c_[cc[0] + sd * rr_ * np.cos(t), cc[1] - rr_ * np.sin(t) * .86]
        for j in range(0, 36, 4): P.q(sp_[j:j + 5] + [0, 2], 2.2, Lf(WOOD_D * .42, *sp_[j])[0], .9, .1, lab)
        for j in range(0, 36, 4): P.q(sp_[j:j + 5] + [1, -1.5], 1.5, Lf(WOOD * 1.6, *sp_[j])[0], .85, .1, lab)
    P.flush()
    return P, quad


def stacks(P, items, kindlight):
    """piles of closed books: (cx, base_y, scale, n, seed)"""
    for (cx, by, s, n, sd) in items:
        r = np.random.default_rng(sd); v = 0.
        for k in range(n):
            wd = r.uniform(190, 300) * s; th = r.uniform(26, 52) * s
            it = bk.make_item(r, 'flat', wd, th); dx = r.normal(0, 18 * s)
            T = lambda u, vv, d=0., cx=cx, by=by: np.stack(np.broadcast_arrays(cx + np.asarray(u, float) + np.asarray(d, float) * 0, by - np.asarray(vv, float) + np.asarray(d, float) * 60 * s), -1)
            it.u = -wd / 2 + dx; it.v = v; it.fore = r.random() < .45; v += th * .97
            bk.paint_row(P, T, [it], kindlight, tops=False)
    P.flush()


def leaf(P, c, size, rot, L, seed, curl=.2, col='#e9d8ae', sq=.42):
    """a fallen page on the floor (screen px)"""
    r = np.random.default_rng(seed)
    a, b = size, size * .72
    q = np.array([[-a, -b], [0, -b * (1 + .15 * r.normal())], [a, -b], [a * 1.02, 0], [a, b], [0, b * (1 + .12 * r.normal())], [-a, b], [-a * 1.02, 0]]) / 2
    q = xf(q, 1, rot=rot) * [1, sq]
    lift = np.zeros(8); lift[r.integers(0, 4) * 2] = curl * size
    q[:, 1] -= lift
    q = spl(q + c, closed=True, corners=(0, 2, 4, 6), n=6)
    P.glaze(q + [-7, 6], np.array([.45, .42, .55]), .85, soft=2.4)
    cc = rgb(col) * r.uniform(.9, 1.02)
    lab = P.poly(q, L(cc, *c)[0])
    for _ in range(7):
        p = np.asarray(c) + r.normal(0, size * .22, 2) * [1, sq]
        P.q(arcs([p[0]], [p[1]], [rot * .6 + r.normal(0, .1)], [size * r.uniform(.4, .9)], [r.normal(0, .2)])[0], size * .07, L(cc * r.uniform(.93, 1.06), *p)[0], .6, .3, lab)
    P.flush()
    e = np.vstack([q, q[:1]]); k = len(q) // 4
    P.q(resample(e[:k + 1], max(2, np.hypot(*(e[k] - e[0])) / 5))[:5], 1.0, L(cc * 1.18, *c)[0], .7)
    P.q(resample(e[k * 2:k * 3 + 1], max(2, np.hypot(*(e[k * 2] - e[k * 3])) / 5))[:5], 1.2, L(cc * .55, *c)[0], .6)
    P.flush()


def candle(P, x, y, h, w, L, seed):
    """spent candle stub with frozen drips (screen px, foot at x, y)"""
    r = np.random.default_rng(seed); wax = rgb('#d8c6a4')
    P.glaze(np.array([[x - w * 2.4, y + 2], [x - w * .2, y - w * .5], [x + w * 1.1, y + 3], [x - w * .6, y + w * .7]]), np.array([.42, .38, .5]), .9, soft=2)
    pool = spl(np.array([[x - w * 1.5, y], [x - w * .6, y - w * .34], [x + w * .7, y - w * .30], [x + w * 1.45, y + 1], [x + w * .6, y + w * .42], [x - w * .7, y + w * .46]]) + r.normal(0, w * .08, (6, 2)), closed=True)
    P.poly(pool, L(wax * .8, x, y)[0])
    b = np.array([[x - w, y], [x - w * 1.02, y - h * .55], [x - w * .95, y - h], [x - w * .2, y - h * 1.06], [x + w * .5, y - h * .95], [x + w * .98, y - h * 1.04], [x + w * 1.03, y - h * .5], [x + w, y]]) + r.normal(0, w * .05, (8, 2))
    o = spl(b, closed=True, corners=(0, 7, 2, 5), n=5); lab = P.poly(o, L(wax, x, y - h / 2)[0])
    P.q(np.linspace([x - w * .55, y - h], [x - w * .6, y], 5), w * .42, L(wax * .62, x, y - h / 2)[0], .9, .1, lab)
    P.q(np.linspace([x + w * .55, y - h * .9], [x + w * .6, y - h * .1], 5), w * .25, L(np.minimum(wax * 1.35, 1.2), x, y - h / 2)[0], .9, .15, lab)
    P.flush()
    for dx, ln in ((-.7, .7), (.2, .45), (.8, .85)):
        if r.random() < .8:
            p = np.array([[x + dx * w, y - h * .97], [x + dx * w * 1.04, y - h * (1 - ln * .5)], [x + dx * w * 1.02, y - h * (1 - ln)]])
            P.q(spl(p, n=2)[:5], w * .20, L(wax * r.uniform(1.0, 1.2), x, y - h / 2)[0], .95)
    P.q(np.array([[x, y - h * .98], [x + 1, y - h - w * .35], [x + 3, y - h - w * .6]]), 1.3, L(rgb('#2a2024') * 1.0, x, y)[0], .95)
    P.flush()


# ------------------------------------------------------------------- the book itself
def page_mesh(side, nu=121, nv=41):
    u, v = np.meshgrid(np.linspace(0, 1, nu) ** 1.5, np.linspace(0, 1, nv))
    wid = (XG - XLE) if side < 0 else (XRE - XG)
    d = u * wid
    X = XG + side * d
    t = v * DEP
    h = prof(d, side) + (.006 * np.sin(v * 5 + d * 6) + .004 * np.sin(d * 13 + v * 3)) * smooth(.1, .4, d) + .022 * smooth(.80, 1.0, u) * smooth(.55, 1, v) * (1 if side < 0 else .4)
    return onb(X, t, h), d, t


def paint_book(img):
    """covers, page block, the two blank pages.  returns img, page alpha, light"""
    C = cam(); r = np.random.default_rng(9)
    Lf, Lt, Lr, Ll = light('front'), light('top'), light('right'), light('left')
    P = bk.Painter(H, W, ss=2, base='#000000', layer=True)
    lea = rgb('#6b3436')
    q4 = lambda X0, X1, t0, t1, h: C.project(onb([X0, X1, X1, X0], [t1, t1, t0, t0], h))[0]
    # shadow of the book on the desk
    P.glaze(q4(XLE - .10, XRE + .04, -.07, DEP + .10, 0), np.array([.35, .32, .45]), .9, soft=6)
    # cover boards: top surface and front thickness
    for (a, b) in ((XLE - .045, XG), (XG, XRE + .045)):
        P.strip(C.project(onb([a, b], DEP + .03, .022))[0], C.project(onb([a, b], -.035, .022))[0], lea * .9, Lt, n=4, seed=4)
        A = C.project(onb(np.linspace(a, b, 7), -.037, .024))[0]; B = C.project(onb(np.linspace(a, b, 7), -.037, -.002))[0]
        P.strip(A, B, lea * 1.0, Lf, n=4, val=.10, hi=.65, lo=-.45, seed=int(a * 100) + 300, wob=.5)
    # spine roll at the gutter
    sp = C.project(np.array([onb(XG + .045 * math.cos(q), -.037, .004 + .028 * math.sin(q)) for q in np.linspace(math.pi, 0, 9)]))[0]
    P.poly(np.vstack([sp, sp[::-1][-1:] + [0, 14], sp[:1] + [0, 14]]), Lf(lea * .6, *sp[4])[0])
    # brass corners
    for X, sd in ((XLE - .045, 1), (XRE + .045, -1)):
        q = C.project(onb([X, X + sd * .10, X + sd * .085, X], [-.038] * 4, [.026, .026, -.003, -.003]))[0]
        P.poly(q, Lf(rgb('#a07a3e'), *q[0])[0]); P.q(np.linspace(q[0], q[1], 5) + [0, 2], 1.4, Lf(rgb('#e8c67a') * 1.1, *q[0])[0], .9)
    P.flush()
    # page block, front edge: the sheaf drawn sheet by sheet
    for side in (-1, 1):
        wid = (XG - XLE) if side < 0 else (XRE - XG)
        d = np.linspace(.004, wid - .006, 60) ; X = XG + side * d
        top = C.project(onb(X, 0., prof(d, side)))[0]; bot = C.project(onb(X, -.004, .024))[0]
        pc = rgb('#d9c395')
        mid = (top + bot) / 2
        ring = np.vstack([top, bot[::-1]])
        lab = P.poly(ring, Lf(pc * .8, *mid.mean(0))[0])
        for j in range(0, 56, 4):
            m = mid[j:j + 5]; P.q(m, np.abs(top[j:j + 5, 1] - bot[j:j + 5, 1]).mean() * .62, Lf(pc * r.uniform(.78, .92), *m.mean(0))[0], 1., 0., lab)
        for k in range(64):
            f_ = r.uniform(.02, .99); s0 = r.uniform(0, .75); ln = r.uniform(.12, .5)
            jj = np.linspace(s0, min(1, s0 + ln), 5) * 59; i0 = np.minimum(jj.astype(int), 58); fr = (jj - i0)[:, None]
            c0 = top[i0] * (1 - fr) + top[i0 + 1] * fr; c1 = bot[i0] * (1 - fr) + bot[i0 + 1] * fr
            p = c0 * (1 - f_) + c1 * f_
            dk = r.random() < .62
            P.q(p, r.uniform(.5, 1.0) if dk else r.uniform(.6, 1.4), Lf(pc * (r.uniform(.45, .64) if dk else r.uniform(1.0, 1.2)), *p.mean(0))[0], r.uniform(.5, .9), .1, lab)
        # a few leaves slipped forward
        for k in range(5):
            f_ = r.uniform(.12, .9); s0 = r.uniform(.02, .7); jj = np.linspace(s0, s0 + r.uniform(.1, .28), 5) * 59; i0 = np.minimum(jj.astype(int), 58)
            p = top[i0] * (1 - f_) + bot[i0] * f_ + [0, 2.5]
            P.q(p, 1.8, Lf(pc * 1.15, *p.mean(0))[0], .9, .1); P.q(p + [0, 2.4], 1.0, Lf(pc * .40, *p.mean(0))[0], .8, .1)
        P.flush()
    # ribbon marker falling over the edge
    c0 = C.project(onb(XG + .10, .02, prof(.10, 1)))[0]
    for j, (pts, w_) in enumerate(((np.array([c0 + [-6, -70], c0 + [-2, -20], c0 + [1, 4], c0 + [0, 60], c0 + [10, 128], c0 + [4, 190]]), 6.5),)):
        s = spl(pts, n=8)
        for a in range(0, len(s) - 4, 3): P.q(s[a:a + 5] + [-4, 2], w_, np.array([.10, .06, .08]), .35)
        for a in range(0, len(s) - 4, 3): P.q(s[a:a + 5], w_, Lf(rgb('#8e2f33') * r.uniform(.9, 1.1), *s[a])[0], .97)
        for a in range(0, len(s) - 4, 3): P.q(s[a:a + 5] + [2.4, 0], 1.6, Lf(rgb('#d0605a'), *s[a])[0], .7)
    P.flush()
    layB = P.layer()
    img = comp.over(img, layB)
    # ---- the pages
    out, zb = None, None
    for sd in (-1, 1):
        S, d, t = page_mesh(sd); N = S3.normals(S)
        out, zb = S3.raster(C, S, np.dstack([np.full(d.shape, sd, float), d, t, N, S]), ss=2, out=out, zb=zb)
    msk = S3.down((zb > 0).astype(f32), 2)
    o1 = out[1::2, 1::2]; right = o1[..., 0] > 0; d = o1[..., 1]; t = o1[..., 2]; N = o1[..., 3:6]; X = o1[..., 6:9]
    on = zb[1::2, 1::2] > 0
    x, y = grid()
    w1 = fbm(H, W, (300, 110, 44), (1, .5, .25), 7, aspect=2.2)
    lamp = np.array([2.0, 4.6, 1.05]); Lv = lamp - X; dl = np.linalg.norm(Lv, axis=-1) + 1e-6
    nl = np.clip((N * Lv).sum(-1) / dl, 0, 1)
    am = 1 / (1 + (dl / 1.15) ** 2.3)
    s_ = np.clip(1.30 * am * (.42 + 1.15 * nl) + .030 * w1, 0, 1)
    cols = [(0, '#676682'), (.10, '#807d94'), (.22, '#a09a9c'), (.34, '#bdaa95'), (.48, '#d8b68a'), (.68, '#f0c886'), (1, '#ffe2a2')]
    k = s_ * 7; kf = np.floor(k); s_p = (kf + smooth(.2, .8, k - kf)) / 7
    g = ramp(.55 * s_p + .45 * s_, cols)
    # the moon shaft lands on the left page
    edge = fbm(H, W, (160, 60), (1, .5), 19) * 5
    sh = smooth(-9, 9, (x - 806 + (y - 560) * .60) + edge) * smooth(26, -26, (x - 1150 + (y - 560) * .60) + edge)
    bar = np.minimum(np.abs(x - 922 + (y - 560) * .60 + edge), np.abs(x - 1040 + (y - 560) * .60 + edge)); sh = sh * (1 - .75 * smooth(11, 4, bar)) * (1 - .6 * smooth(6, 2, np.abs(t - .40 + .03 * d)))
    g = mix(g, np.minimum(g * np.array([1.18, 1.20, 1.24], f32) + np.array([.14, .145, .15], f32), 1), (sh * smooth(.62, .30, s_) * .9)[..., None])
    gd = np.where(right, d, d)
    turn = smooth(.085, .010, gd) * (~right)
    g = mix(g, g * np.array([.52, .42, .50], f32), (turn * .85)[..., None])
    roll = smooth(.26, .15, gd) * smooth(.06, .15, gd) * (~right)
    g = mix(g, np.minimum(g * np.array([1.12, 1.07, .98], f32) + .015, 1), (roll * .8)[..., None])
    rp = ramp(np.clip(gd / .40 + .05 * w1, 0, 1), [(0, '#38263a'), (.10, '#6a4a4c'), (.26, '#b27e56'), (.55, '#e8ba78'), (1, '#fbd796')])
    g = np.where(right[..., None], mix(rp, g, (smooth(.25, .6, gd) * .5)[..., None]), g)
    g = mix(g, np.ones_like(g) * rgb('#2a1a28'), (smooth(.012, .002, gd) * on * .9)[..., None])
    # deckled, toned border; foxing
    wl = (XG - XLE); ed = np.minimum(np.minimum(t, DEP - t), np.where(right, (XRE - XG), wl) - gd)
    old = smooth(.11, 0, ed + .03 * fbm(H, W, (80, 30), (1, .5), 23))
    g = mix(g, g * np.array([.93, .86, .78], f32), (old * .8)[..., None])
    lab = np.where(on, np.where(right, 2, 1), 0).astype(np.int32)
    gl = C.project(onb([XG, XG], [0, DEP], [.06, .06]))[0]; ga = math.atan2(*(gl[0] - gl[1])[::-1])
    ang = .03 + .07 * fbm(H, W, (200, 80), (1, .5), 15) + (ga - .03) * smooth(.22, .08, gd) * on
    g = repaint(np.clip(g, 0, 1), sizes=(18, 9, 4.5), seed=5, angle=ang, lab=lab, mask=on.astype(f32), thr=(0, .010, .016), density=1.05, aspect=8., jit=(.003, .010, .012), op=.82, dry=.15, bristle=.028)
    g = S3.soften_edges(g, np.where(on, lab, 1))
    Uv = np.dstack([(X[..., 0] + 1.) * 900 % 2046, t * 1900 % 2046])
    f1 = tile_to(fibre_tile(), 2048, 2048); s1 = S3.tex(f1.astype(f32), Uv)
    g = g * (1 - .045 * s1[..., None] * np.array([.75, .95, 1.2], f32))
    fx = np.zeros((H, W), f32)
    for _ in range(150):
        side = -1 if r.random() < .5 else 1; dd = r.uniform(.05, 1.1); tt = r.uniform(.02, DEP - .02)
        p, z = C.project(onb(XG + side * dd, tt, prof(dd, side))); rad = r.gamma(1.3, 1.0) + .5
        cv2.ellipse(fx, (int(p[0]), int(p[1])), (max(1, int(rad * 1.5)), max(1, int(rad * .6))), 0, 0, 360, float(r.uniform(.3, .8)), -1, cv2.LINE_AA)
    g = mix(g, g * np.array([.80, .66, .55], f32), (cv2.GaussianBlur(fx, (0, 0), .9) * .7)[..., None])
    c = Cel(ss=1)
    for (side, pts, w_, al) in ((-1, [(.95, .25), (.6, .30), (.3, .24)], 2.6, .30), (-1, [(.8, .62), (.5, .68), (.22, .64)], 2.4, .28), (1, [(.2, .35), (.6, .42), (1.0, .36)], 2.6, .26), (1, [(.4, .75), (.8, .70), (1.1, .76)], 2.2, .26), (-1, [(1.0, .80), (.8, .86), (.6, .84)], 2.2, .3)):
        p = C.project(np.array([onb(XG + side * a, b, prof(a, side) + .002) for a, b in pts]))[0]; q = spl(p, n=14)
        c.line(q + [0, 2], w_, '#3c2b46', (.45, .45), seed=int(abs(pts[0][0]) * 90), alpha=al, press=.5); c.line(q - [0, 1.4], w_ * .8, '#ffe6b4', (.45, .45), seed=int(abs(pts[0][1]) * 90), alpha=al * .8, press=.5)
    l = cv2.GaussianBlur(c.rgba(), (0, 0), 1.0); g = g * (1 - l[..., 3:]) + l[..., :3] * (.35 + .65 * lum(g)[..., None])
    # edges: hand cut, near edge catches light, far edge rimmed by the moon
    a = msk * np.clip(ed / .0022 * .5 + fbm(H, W, (16, 6, 3), (1, .6, .4), 61) * .45 + .4, 0, 1)
    near = smooth(.010, .001, t) * on; far = smooth(DEP - .016, DEP - .002, t) * on
    g = g + near[..., None] * (.10 + .16 * s_[..., None]) * rgb('#ffe9c0') + far[..., None] * .16 * MOON
    img = img * (1 - a[..., None]) + g * a[..., None]
    return img, a.astype(f32), (s_ * a).astype(f32)


def moon_pool():
    """window light on floor and step (mullions and all), in world metres, screen mask"""
    C = cam()
    u, v, t = C.uv((0, 0, 0), (1, 0, 0), (0, 1, 0))
    n = fbm(H, W, (60, 22), (1, .5), 5) * .006
    a = u + .55 * (v - 5.2) + n; b = v + n
    m = smooth(-2.18, -2.15, a) * smooth(-1.02, -1.05, a) * smooth(4.32, 4.36, b) * smooth(6.75, 6.70, b)
    for c in (-1.89, -1.60, -1.31): m = m * (1 - .9 * smooth(.030, .020, np.abs(a - c)))
    for c in (4.9, 5.5, 6.1): m = m * (1 - .9 * smooth(.034, .022, np.abs(b - c)))
    return cv2.GaussianBlur(m.astype(f32), (0, 0), 1.4)


def build():
    C = cam(); x, y = grid(); r = np.random.default_rng(3)
    bg = backdrop()
    bg = comp.bokeh(bg.astype(f32), 3.0, 0)
    bg = bg * (.62 + .38 * np.exp(-(((x - 520) / 620) ** 2 + ((y - 380) / 420) ** 2)))[..., None]
    bg = mix(bg, np.ones_like(bg) * rgb('#1d2040'), (.30 + .30 * smooth(260, 0, y))[..., None])
    P = bk.Painter(H, W, ss=2, base='#000000', layer=True)
    floor(P)
    img = comp.over(bg, P.layer())
    hz = np.exp(-((y - 566) / 60.) ** 2) * .60
    img = mix(img, np.ones_like(img) * rgb('#262544'), hz[..., None])
    # moonlight on the planks
    mp = moon_pool()
    def moonlit(a, m):
        Y = lum(a)[..., None]
        return mix(a, a * np.array([.95, 1.15, 1.10], f32) + rgb('#adc0e6') * (.12 + 1.35 * Y), m[..., None])
    img = moonlit(img, mp)
    # ---- things standing in the hall behind the lectern
    Lf, Lt, Lr, Ll = light('front'), light('top'), light('right'), light('left')
    P = bk.Painter(H, W, ss=2, base='#000000', layer=True)
    stacks(P, [(655, 720, .34, 6, 11), (600, 735, .34, 3, 12), (320, 700, .30, 5, 13), (455, 668, .24, 4, 14), (1985, 690, .42, 5, 15)], Lf)
    for (cx, cy, sz_, ro, sd) in ((300, 830, 70, .3, 1), (560, 800, 60, -.5, 2), (150, 930, 90, .1, 3)): leaf(P, (cx, cy), sz_, ro, Lt, sd)
    lay = P.layer(); lay = cv2.GaussianBlur(lay, (0, 0), .9)
    img = comp.over(img, lay)
    # ---- shaft of moonlight in the air, behind the lectern
    for (quad, sft, am_) in (([[-60, -60], [120, -60], [760, 720], [520, 760]], 22, .22), ([[150, -60], [300, -60], [980, 640], [800, 690]], 24, .28), ([[330, -60], [420, -60], [1130, 600], [1010, 620]], 20, .18)):
        sh = comp.shaft(H, W, quad, sft, seed=int(quad[1][0]), streak=.20) * smooth(1000, 60, y)
        img = comp.screen(img, sh[..., None] * MOON * am_)
    # ---- step and lectern
    P, quad = set_layer()
    mst = mp
    lay = P.layer()
    al = lay[..., 3:]
    # the pool climbs on to the step
    sm = lum(lay[..., :3])
    top = ((y < 1032) & (y > 800) & (x < 900)).astype(f32)
    lay[..., :3] = mix(lay[..., :3], moonlit(lay[..., :3], mp), (top * al[..., 0])[..., None])
    img = comp.over(img, lay)
    img, pa, s_ = paint_book(img)
    # ---- small things on the step and in front
    P = bk.Painter(H, W, ss=2, base='#000000', layer=True)
    stacks(P, [(812, 1000, .40, 3, 21), (1965, 1012, .74, 5, 22), (1800, 1016, .50, 2, 23)], Lf)
    for (cx, cy, sz_, ro, sd, cu) in ((1010, 1004, 96, .15, 5, .10), (1330, 1010, 84, -.4, 6, .22), (1610, 1006, 90, .5, 7, .12), (330, 1092, 130, -.2, 8, .16), (690, 1112, 120, .35, 9, .26), (1140, 1116, 128, -.15, 10, .14), (1500, 1108, 110, .2, 14, .2), (70, 1010, 110, .5, 12, .1)):
        leaf(P, (cx, cy), sz_, ro, Lt, sd, cu)
    for (cx, cy, h_, w_, sd) in ((716, 1022, 46, 10, 1), (744, 1028, 24, 11, 2), (690, 1029, 31, 8.5, 3), (1900, 1024, 52, 11, 4), (1872, 1030, 26, 12, 5), (1218, 1026, 20, 9, 6)): candle(P, cx, cy, h_, w_, Lf, sd)
    img = comp.over(img, P.layer())
    # a little of the beam also hangs in front
    sh = comp.shaft(H, W, [[150, -60], [420, -60], [1130, 600], [800, 690]], 30, seed=9, streak=.2) * smooth(800, 100, y)
    img = comp.screen(img, sh[..., None] * MOON * .075)
    # lantern off frame right: transmitted light
    img = comp.glow(img, 2160, 800, 1000, '#ff9a42', .30, 2.2)
    img = comp.glow(img, 2160, 800, 420, '#ffcf86', .28, 2.0)
    # corners and foreground go down into the dark
    v = smooth(.55, 1.2, np.hypot((x - 1180) / 1300., (y - 560) / 760.)) * .42 * (1 - .7 * lampf(x, y))
    img = img * (1 - v[..., None] * np.array([.9, 1.0, .8], f32))
    img = mottle(np.clip(img, 0, 1.2), .012, 4, .004) if False else img
    # ---- foreground pile, out of focus, over the fox as it runs in
    P = bk.Painter(H, W, ss=1, base='#000000', layer=True)
    dk = bk.Light(lambda x, y: np.zeros(np.shape(x) + (3,)) + np.array([.20, .19, .30]) + np.clip(1 - np.abs(np.asarray(x, float) - 260) / 300., 0, 1)[..., None] * np.array([.10, .12, .18]), '#0f0d1c', -1)
    stacks(P, [(40, 1230, 1.25, 5, 31), (300, 1260, 1.0, 3, 32)], dk)
    fg = comp.bokeh(P.layer(), 6.5, 0)
    return dict(plate=img.astype(f32), page=pa, am=s_, fg=fg.astype(f32), step=np.clip(al[..., 0], 0, 1))


def hall_close():
    """the same hall for 4c, camera gone in close over the page: painted, then thrown out of focus"""
    x, y = grid()
    _over.update(eye=(0., 2.6, 1.42), cx=1230, cy=333)
    try:
        bg = backdrop()
        bg = bg * (.70 + .42 * np.exp(-(((x - 560) / 560) ** 2 + ((y - 420) / 420) ** 2)))[..., None]
        bg = mix(bg, np.ones_like(bg) * rgb('#1d2040'), (.22 + .25 * smooth(260, 0, y))[..., None])
        P = bk.Painter(H, W, ss=1, base='#000000', layer=True); floor(P)
        img = comp.over(bg, P.layer())
        hz = np.exp(-((y - 660) / 60.) ** 2) * .5
        img = mix(img, np.ones_like(img) * rgb('#262544'), hz[..., None])
        pool = smooth(-1, 1, (x - 690) / 60. + (y - 780) / 90.) * smooth(1, -1, (x - 1120) / 60. + (y - 780) / 90.) * smooth(700, 716, y) * smooth(852, 836, y)
        bar = np.abs(((x + (y - 780) * .66 - 690) / 108.) % 1 - .5); pool = pool * (1 - .8 * smooth(.42, .48, bar))
        img = mix(img, img * 1.35 + rgb('#b2c3ea') * (.08 + .9 * lum(img)[..., None]), pool[..., None])
        P = bk.Painter(H, W, ss=1, base='#000000', layer=True)
        stacks(P, [(1690, 872, .78, 5, 41), (1440, 800, .52, 4, 42), (1960, 880, .9, 6, 43), (1290, 742, .36, 3, 44), (330, 790, .5, 6, 45), (150, 830, .6, 4, 46)], light('front'))
        img = comp.over(img, P.layer())
    finally: _over.clear()
    for (quad, sft, am_) in (([[180, -60], [380, -60], [1010, 940], [780, 960]], 24, .22), ([[420, -60], [600, -60], [1300, 900], [1090, 930]], 26, .30), ([[640, -60], [730, -60], [1440, 860], [1330, 880]], 22, .16)):
        sh = comp.shaft(H, W, quad, sft, seed=int(quad[1][0]), streak=.22) * smooth(1080, 60, y)
        img = comp.screen(img, sh[..., None] * MOON * am_)
    img = comp.glow(img, 2160, 800, 900, '#ff9a42', .34, 2.2)
    return comp.bokeh(img.astype(f32), 7.0, 0)
