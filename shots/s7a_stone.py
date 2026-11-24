"""s7a masonry: the tower's flank, the broken balcony ledge on its great bracket, what is left of the balustrade."""
import numpy as np, math, cv2
from lp import W, H
from lp.paint import f32, rgb, mix, smooth, fbm, Canvas, arcs, jit_color
from shots.balc_stone import *

FC, CX, CY, EZ = 1667., 1230., 805., .62
RW, RH = 1040, 1152


def pj(X, Y, Z): return np.array([CX + FC * X / Y, CY + FC * (EZ - Z) / Y])


AX, AY, AR = -10.75, 10.2, 4.9


def cyl(a, z): return pj(AX + AR * math.cos(a), AY + AR * math.sin(a), z)


SH = ['#8d7690', '#97809a', '#84708f', '#a08a9b', '#8f7c9a', '#9a8494']
LIT = ['#e6c9a0', '#ecd0a2', '#e2c29b', '#efd6aa', '#e7c698']


def build():
    r = np.random.default_rng(77)
    S = Stone(0, 0, RW, RH); G = Stone(0, 0, RW, RH, ground='#dcbc98')            # shade paint, light paint
    T = np.full((RH, RW), 40., f32)
    tag = {}
    pick = lambda L: rgb(L[int(r.integers(0, len(L)))])
    def tmap(pts, t):
        cv2.fillPoly(T, [np.round(np.asarray(pts) * 16).astype(np.int32)], float(t), cv2.LINE_8, 4)

    def clear(pts):
        m = np.zeros((RH * 2, RW * 2), np.uint8); cv2.fillPoly(m, [np.round(S.P(pts) * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
        x, y, w, h = cv2.boundingRect(np.round(S.P(pts)).astype(np.int32)); x0, y0 = max(0, x - 2), max(0, y - 2)
        G.cv.a[y0:y + h + 2, x0:x + w + 2] *= 1 - m[y0:y + h + 2, x0:x + w + 2].astype(f32) / 255.
    _blk = S.block; _fil = S.fill
    def blk(pts, *a, **k): clear(pts); return _blk(pts, *a, **k)
    S.block = blk
    def base(pts, col): clear(pts); _fil(pts, col)

    def lit(i, pts, amt, col=None, ang=0., dry=.35, dens=1.0, hw=None):
        """sun on one block: broad strokes inside its own outline.  amt blends shade paint -> sun paint; a raking terminator is dry brushed"""
        if amt <= .04: return
        q = S.P(pts); mn = q.min(0); mx = q.max(0); sz = mx - mn
        lc = pick(LIT) if col is None else rgb(col) if isinstance(col, str) else col
        bc = S.blocks[-1][1]
        half = mix(mix(bc, rgb('#c7a196'), .75), lc, .0)
        c0 = mix(mix(bc, half, min(1, amt * 2.2)), lc, np.clip(amt * 1.5 - .5, 0, 1))
        G.lab = S.lab
        h_ = max(2., min(sz) * r.uniform(.13, .22)) if hw is None else hw * 2
        G.fill(pts, c0, min(1, amt * 3) * .86)
        n = int(np.clip(sz[0] * sz[1] / (h_ * h_ * 3.2) * dens, 8, 160))
        x = r.uniform(mn[0], mx[0], n); y = r.uniform(mn[1], mx[1], n)
        c = jit_color(np.repeat(c0[None], n, 0), r, .006, .012, .022)
        G.cv.strokes(arcs(x, y, ang + r.normal(0, .09, n), h_ * r.uniform(3.5, 9, n), r.normal(0, .1, n)), h_ * r.uniform(.6, 1.25, n), c, r.uniform(.6, .95, n) * min(1, amt * 3),
                     np.clip(r.normal(.10, .1, n), 0, 1), seed=int(r.integers(1 << 20)), lab=S.lab, LAB=i, bristle=.035)
        # lighter accents high and sunward on the stone
        n = int(3 + 7 * amt); x = mn[0] + sz[0] * r.beta(2, 1.2, n); y = mn[1] + sz[1] * r.beta(1.1, 2.4, n)
        G.cv.strokes(arcs(x, y, ang + r.normal(0, .1, n), np.minimum(sz[0] * .5, h_ * r.uniform(3, 9, n)), r.normal(0, .1, n)), h_ * r.uniform(.25, .6, n), jit_color(np.repeat(np.clip(lc * 1.03 + .012, 0, 1)[None], n, 0), r, .004, .01, .02),
                     r.uniform(.2, .6, n) * amt, r.uniform(.2, .7, n), seed=int(r.integers(1 << 20)), lab=S.lab, LAB=i, bristle=.05)

    def rim(a, b, col='#fff0c8', w=1.6, op=.9, seg=1, brk=.15):
        """burning edge: broken stroke"""
        a = np.asarray(a, float); b = np.asarray(b, float)
        L = np.hypot(*(b - a)); m = max(1, int(L / 34)) if seg else 1
        for k in range(m):
            if r.random() < brk: continue
            p0 = a + (b - a) * (k + r.uniform(0, .06)) / m; p1 = a + (b - a) * (k + 1 - r.uniform(0, .08)) / m
            P = np.stack([p0 + (p1 - p0) * t + r.normal(0, .3, 2) for t in (0, .33, .66, 1)])
            G.cv.strokes(G.P(P)[None], w * 2 * r.uniform(.35, .65), jit_color(rgb(col)[None], r, .004, .01, .02), op * r.uniform(.6, 1), r.uniform(0, .25), seed=int(r.integers(1 << 20)), shape='round', bristle=.03)

    # ------------------------------------------------------------ tower flank
    a0, a1 = math.radians(-34), math.radians(27.6)
    ol = [cyl(a0, -4)] + [cyl(a0, 6.5)] + [cyl(a, 6.5) for a in np.linspace(a0, a1, 12)] + [cyl(a1, -4)] + [cyl(a, -4) for a in np.linspace(a1, a0, 12)]
    S.fill([(-5, -5), (cyl(a1, 6.5)[0], -5), tuple(cyl(a1, 0)), (cyl(a1, -4)[0], RH + 5), (-5, RH + 5)], '#75648a')
    n = 260; x = r.uniform(0, 640, n); y = r.uniform(0, RH * 2, n)
    wall = Canvas(RH * 2, RW * 2, layer=True)
    v = -3.6
    row = 0
    while v < 6.2:
        hgt = r.uniform(.25, .40) * (1.35 if row % 5 == 2 else 1)
        a = a0 - r.uniform(0, .14)
        while a < a1 - .004:
            dw = r.uniform(.075, .19) * (1 if r.random() > .2 else .55)
            b = min(a + dw, a1)
            if a1 - b < .03: b = a1
            poly = cut(a, b, v, v + hgt, r, gap=.0, chip=.0, wob=0, n=2)
            # cut in metric space then wrap onto the cylinder
            m = cut(a * AR, b * AR, v, v + hgt, r, gap=r.uniform(.003, .009), chip=.05, pchip=.2, wob=.006, n=4)
            pts = np.array([cyl(u / AR, z) for u, z in m])
            if pts[:, 0].max() > -6 and pts[:, 1].max() > -10 and pts[:, 1].min() < RH + 10:
                am = (a + b) / 2; zc = v + hgt / 2
                base_ = mix(pick(SH), rgb('#8f7a94'), .4) * mix(.90, 1.04, (am - a0) / (a1 - a0)) * (1 + .03 * math.sin(zc * 1.3 + am * 3))
                base_ = mix(base_, rgb('#7f78a8'), .16 * max(0, math.sin(zc * .9 + 1)))             # cool sky
                if zc < 1.2 and zc > 0: base_ = mix(base_, rgb('#b08a8c'), .18)                     # bounce off the floor
                d = cyl(am + .02, zc) - cyl(am - .02, zc); ang = math.atan2(d[1], d[0])
                i = S.block(pts, base_, r, ang=ang, hw=r.uniform(2.6, 5.), jit=(.006, .014, .024), grad=(.4, -.6, .05), light=(.75, -.65), dens=1.3)
                ndl = math.cos(am) * .34 + math.sin(am) * .94
                tt = 14 + (6.2 - zc) / 6.2 * 26 + r.normal(0, 2.6)
                tmap(pts, tt)
                amt = float(np.clip((ndl - .03 + r.normal(0, .07)) / .40, 0, 1))
                if zc < 0: amt *= max(0, 1 + zc / 2.4)
                lit(i, pts, amt, ang=ang)
                if amt > .25:
                    # lit arris of each stone: right edge and top
                    tr = pts[np.argmax(pts[:, 0] - pts[:, 1] * .5)]; tl = pts[np.argmin(pts[:, 0] + pts[:, 1])]
                    rim(tl, tr, '#fbe3b4', 1.1, .55 * amt, 1, .35)
                if r.random() < .18: S.lichen(pts.mean(0) + r.normal(0, 8, 2), r.uniform(5, 13), r, n=18, lab=i)
                if r.random() < .10: S.crack(pts[int(r.integers(len(pts)))], pts.mean(0) + r.normal(0, 8, 2), r, lab=i)
            a = b
        v += hgt; row += 1
    # sunlit mortar keeps a warm rose, and the door the fox came out of
    jm = (S.lab == 0)
    xx = np.arange(RW * 2)[None] / 2.
    zone = smooth(110, 250, xx) * .85
    wp = np.zeros((RH * 2, RW * 2), np.uint8); cv2.fillPoly(wp, [np.round(S.P([(-5, -5), (cyl(a1, 6.5)[0], -5), tuple(cyl(a1, 0)), (-5, cyl(a1, 0)[1] + 30)]) * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
    k = jm * wp * zone
    G.cv.img += (rgb('#a98c8f') - G.cv.img) * (k > 0)[..., None]; G.cv.a = np.maximum(G.cv.a, k.astype(f32))
    ac, aw, zs = math.radians(-11.5), .082, 1.30                       # centre angle, half width (rad), spring line
    arch = lambda kk, inset=0.: [cyl(ac - aw * kk, -.02)] + [cyl(ac - aw * kk * math.cos(t), zs + aw * AR * kk * math.sin(t)) for t in np.linspace(0, math.pi, 15)] + [cyl(ac + aw * kk, -.02)]
    base(arch(1.0), '#5a4c74')
    # inside: cool dark, a lit left jamb, a warm patch of floor
    q_ = [cyl(ac - aw * math.cos(t) * 1.0, zs + aw * AR * math.sin(t)) for t in np.linspace(0, math.pi * .62, 10)]
    inner = [p_ + np.array([26, 3 + 5 * i_ / 9]) for i_, p_ in enumerate(q_)]
    S.fill([cyl(ac - aw, -.02)] + q_ + inner[::-1] + [cyl(ac - aw, -.02) + np.array([26, 2])], '#857092')          # the wall's thickness, left jamb
    G.fill([cyl(ac - aw, .0) + np.array([2, 0])] + [p_ + np.array([2, 1]) for p_ in q_[:6]] + [p_ + np.array([-3, 2]) for p_ in inner[:6]][::-1] + [cyl(ac - aw, .0) + np.array([23, 2])], '#dcb592', .9)
    for zz in np.arange(.28, zs + .2, .3):
        a_ = cyl(ac - aw, zz); S.cv.strokes(S.P(np.stack([a_ + np.array([26 * t, 1.5 * t]) for t in (0, .33, .66, 1)]))[None], 1.2, rgb('#63547e'), .8, 0, seed=int(zz * 99), shape='round'); G.cv.strokes(G.P(np.stack([a_ + np.array([26 * t, 1.5 * t]) for t in (0, .33, .66, 1)]))[None], 1.2, rgb('#b18f8e'), .9, 0, seed=int(zz * 99), shape='round')
    S.fill([cyl(ac - aw, .0) + np.array([26, 1]), cyl(ac + aw, 0), cyl(ac + aw, .09), cyl(ac - aw, .12) + np.array([26, 1])], '#86718f'); G.fill([cyl(ac - aw, .0) + np.array([26, 1]), cyl(ac + aw * .4, 0), cyl(ac + aw * .3, .07), cyl(ac - aw, .10) + np.array([26, 1])], '#d9b594', .8)
    tmap([cyl(ac - aw, 0), cyl(ac - aw, zs + .4), cyl(ac + aw, zs + .4), cyl(ac + aw, 0)], 40)
    for k2 in range(60):
        u = r.uniform(-.9, .9); z0 = r.uniform(0, zs + .3); p = cyl(ac + aw * u, z0); L_ = r.uniform(20, 70)
        if math.hypot(u, max(0, z0 - zs) / (aw * AR)) > .9: continue
        S.cv.strokes(S.P(np.stack([p + np.array([r.normal(0, .6), -L_ * t]) for t in (0, .33, .66, 1)]))[None], r.uniform(3, 8), jit_color(rgb('#4f4370')[None], r, .006, .01, .03), r.uniform(.12, .34) * min(1, z0 / .8), .3, seed=k2, shape='round')
    vs = [('j', -1, z) for z in np.arange(0, zs, .33)] + [('a', t, 0) for t in np.linspace(0, math.pi, 10)[:-1]] + [('j', 1, z) for z in np.arange(0, zs, .33)]
    for (kd, q, z) in vs:
        if kd == 'j': m = [(ac + q * aw * 1.0, z + .006), (ac + q * aw * 1.34, z + .006), (ac + q * aw * 1.32, min(zs, z + .32)), (ac + q * aw * 1.0, min(zs, z + .32))]
        else:
            t0_, t1_ = q + .02, q + math.pi / 9 - .02; k0, k1 = 1.0, r.uniform(1.30, 1.42)
            m = [(ac - aw * k0 * math.cos(t0_), zs + aw * AR * k0 * math.sin(t0_)), (ac - aw * k1 * math.cos(t0_), zs + aw * AR * k1 * math.sin(t0_)), (ac - aw * k1 * math.cos(t1_), zs + aw * AR * k1 * math.sin(t1_)), (ac - aw * k0 * math.cos(t1_), zs + aw * AR * k0 * math.sin(t1_))]
        pts = np.array([cyl(u + r.normal(0, .0012), zz + r.normal(0, .006)) for u, zz in m])
        i = S.block(pts, mix(pick(SH), rgb('#a38c9b'), .5) * r.uniform(.95, 1.03), r, ang=np.pi / 2 if kd == 'j' and False else 0, hw=2.4, light=(.75, -.65)); tmap(pts, 34 + r.normal(0, 2))
        lit(i, pts, float(np.clip(r.uniform(.12, .34) + (.18 if pts[:, 0].mean() > 90 else 0), 0, 1)))
    # weather streaks down the flank
    for k in range(26):
        xs = r.uniform(0, 290); ys = r.uniform(-40, 820); L = r.uniform(60, 260)
        P = np.stack([np.array([xs + r.normal(0, 3) * t, ys + L * t]) for t in np.linspace(0, 1, 5)])
        S.cv.strokes(S.P(P)[None], r.uniform(3, 9), rgb('#665a86') if r.random() < .7 else rgb('#a08f9f'), r.uniform(.08, .22), r.uniform(.3, .7), seed=k, shape='taper')
    rim(cyl(a1, 6.2) + (-1.5, 0), cyl(a1, .02) + (-1.5, 0), '#fff2cc', 3.0, .95, 1, .08)
    rim(cyl(a1, -.7) + (-1.5, 0), cyl(a1, -2.4) + (-1.5, 0), '#f3cfa0', 2.0, .6, 1, .3)
    for k in range(30):                                        # nicks break the silhouette
        z = r.uniform(-.2, 6); p = cyl(a1, z)
        S.fill(p + r.normal(0, 1, (5, 2)) * (2.2, 3.5) + (-1, 0), '#8d7690')

    # ------------------------------------------------------------ great bracket under the ledge
    Yb = 8.9
    br_out = [(-5.9, -.62), (-3.0, -.62), (-3.02, -.78), (-3.25, -.86), (-3.6, -.95), (-3.95, -1.12), (-4.25, -1.36), (-4.5, -1.66), (-4.7, -2.02), (-4.85, -2.5), (-5.9, -2.5)]
    base([pj(x, Yb, z) for x, z in br_out], '#5c5080')
    def in_br(x, z):
        P = np.array(br_out[1:10]); xx = np.interp(-z, -P[:, 1], P[:, 0]); return x < xx
    zt = -.62
    for row, hgt in enumerate((.36, .40, .34, .42, .40)):
        x = -5.9 - r.uniform(0, .3)
        while x < -2.9:
            w = r.uniform(.42, .85); b = x + w
            zm = zt - hgt / 2
            P = np.array(br_out[1:10])
            xe_t = np.interp(-(zt - .01), -P[:, 1], P[:, 0]); xe_b = np.interp(-(zt - hgt), -P[:, 1], P[:, 0])
            if x < min(xe_t, xe_b) - .06:
                if b > xe_b - .16: b = 99
                if b == 99:
                    m = np.array([(x + .012, zt - .012), (xe_t - .03, zt - .012), (np.interp(-(zt - hgt * .5), -P[:, 1], P[:, 0]), zt - hgt * .5), (xe_b - .02, zt - hgt + .01), (x + .012, zt - hgt + .01)]) + r.normal(0, .006, (5, 2))
                else: m = cut(x, b, zt - hgt, zt, r, .012, .05, .16, .007)
                pts = np.array([pj(u, Yb, z) for u, z in m])
                bb = pick(SH) * mix(.84, .72, row / 4.) ; bb = mix(bb, rgb('#6a608f'), .25)
                if b == 99: bb = mix(bb, rgb('#b58d86'), .28)                              # bounce from the clouds
                i = S.block(pts, bb, r, ang=0, hw=r.uniform(2.6, 5.), grad=(.6, .2, .06), light=(.7, .5))
                tmap(pts, 52 + r.normal(0, 3))
                if b == 99:
                    lit(i, pts[[1, 2, 3]] if False else pts, .0)
                if r.random() < .3: S.lichen(pts.mean(0) + r.normal(0, 10, 2), r.uniform(6, 14), r, lab=i)
            x = x + w
        zt -= hgt
    prof = [pj(x, Yb, z) for x, z in br_out[2:10]]
    for p, q in zip(prof[:-1], prof[1:]): rim(p, q, '#f7d3a2', 2.2, .75, 1, .2)
    # dark soffit between slab and bracket, cool
    base([pj(-6.6, 8.75, -.5), pj(-2.5, 8.75, -.5), pj(-2.3, 9.6, -.5), pj(-2.3, 12., -.5), pj(-3.4, 12., -.62), pj(-6.6, 9.2, -.66)], '#5f5384')

    # ------------------------------------------------------------ far balustrade (Y = 12)
    YB = 11.9
    def rail_piece(x0, x1, z0, z1, Y, light=.9, top=True, rough=0, base=None, t0=38.):
        m = cut(x0, x1, z0, z1, r, .006, .035, .12 + rough * .5, .006)
        pts = np.array([pj(u, Y, z) for u, z in m])
        c = pick(SH) * r.uniform(.94, 1.04) if base is None else base
        i = S.block(pts, c, r, ang=0, hw=r.uniform(1.4, 2.4), light=(.7, -.7))
        tmap(pts, t0 + r.normal(0, 2))
        if top: rim(pj(x0 + .01, Y, z1 - .012), pj(x1 - .01, Y, z1 - .012), '#fff0c6', 2.2, light, 1, .12)
        return i, pts
    def baluster(xc, Y, h, z0=.14, broken=None, w=.105, t0=40.):
        zs = np.array([0, .05, .07, .12, .20, .30, .40, .52, .64, .74, .80, .86, .93, .95, 1.0]); ws = np.array([1.05, 1.05, .72, .62, .92, 1.16, 1.2, 1.02, .7, .5, .46, .62, .7, 1.02, 1.02]) * w * r.uniform(.94, 1.06)
        k = 1.0 if broken is None else broken
        z = zs * h; use = zs <= k + 1e-6
        L = [(xc - a * r.uniform(.94, 1.04), z0 + b) for a, b in zip(ws[use], z[use])]; R_ = [(xc + a * r.uniform(.96, 1.05), z0 + b) for a, b in zip(ws[use], z[use])]
        pts = L + ([(xc - .05, z0 + k * h + .035), (xc + .0, z0 + k * h + .02), (xc + .04, z0 + k * h - .02)] if broken else []) + R_[::-1]
        pts = np.array([pj(u, Y, zz) for u, zz in pts])
        i = S.block(pts, pick(SH) * r.uniform(.9, 1.0), r, ang=np.pi / 2, hw=1.6, light=(.8, -.5))
        tmap(pts, t0 + r.normal(0, 3))
        rp = np.array([pj(u, Y, zz) for u, zz in R_])
        for a, b in zip(rp[:-1], rp[1:]):
            if r.random() > .12: rim(a + (-1.6, 0), b + (-1.6, 0), '#ffe6b4', 2.4, .85, 0, 0)
        for a, b in zip(rp[:-1], rp[1:]):
            if r.random() > .25: rim(a + (-5, 0), b + (-5, 0), '#e9bf94', 3.0, .5, 0, 0)
        return pts
    def post(xc, Y, h, w=.36, cap=True, top_pts=None, t0=38.):
        zs = np.cumsum([0] + [h / 3 * k for k in (1.1, .95, .95)])
        for k in range(3):
            if top_pts is not None and zs[k] > h * .6: break
            m = cut(xc - w / 2, xc + w / 2, zs[k], zs[k + 1], r, .005, .05, .3, .007)
            pts = np.array([pj(u, Y, z) for u, z in m]); i = S.block(pts, pick(SH) * r.uniform(.92, 1.04), r, ang=0, hw=2., light=(.8, -.6)); tmap(pts, t0 + r.normal(0, 2.5))
            # sun side return face (thin sliver seen on the right)
            s = np.array([pj(xc + w / 2, Y, zs[k] + .01), pj(xc + w / 2, Y, zs[k + 1] - .01), pj(xc + w / 2, Y + w, zs[k + 1] - .01), pj(xc + w / 2, Y + w, zs[k] + .01)])
            j = S.block(s, mix(pick(SH), rgb('#b99a98'), .4), r, ang=np.pi / 2, hw=1.6, edge=False); tmap(s, t0 + r.normal(0, 2.5)); lit(j, s, .95, '#f3d7a6', np.pi / 2, .12)
            rim(s[0] + (0, 0), s[1], '#ffefc4', 1.8, .9, 1, .1)
            if r.random() < .6: S.lichen(pts.mean(0) + r.normal(0, 6, 2), r.uniform(4, 9), r, n=14, lab=i)
        if top_pts is not None:
            pts = np.array([pj(u, Y, z) for u, z in top_pts]); S.block(pts, pick(SH), r, ang=.4, hw=2.); tmap(pts, t0)
            for a, b in zip(pts[:-1], pts[1:]):
                if (b - a)[0] > 0: rim(a, b, '#fff0c8', 2., .9, 1, .1)
        if cap:
            for (dw, z0, z1) in ((.05, h, h + .07), (.085, h + .07, h + .15), (.03, h + .15, h + .20)):
                m = cut(xc - w / 2 - dw, xc + w / 2 + dw, z0, z1, r, .003, .03, .2, .004)
                pts = np.array([pj(u, Y, z) for u, z in m]); S.block(pts, pick(SH) * 1.02, r, ang=0, hw=1.6); tmap(pts, t0)
                rim(pj(xc - w / 2 - dw, Y, z1 - .01), pj(xc + w / 2 + dw, Y, z1 - .01), '#fff0c8', 2., .9, 1, .1)
    # plinth course along the far edge
    x = -6.3
    while x < -2.35:
        w = r.uniform(.5, .95)
        if not (-4.1 < x < -3.0): rail_piece(x, min(x + w, -2.3), 0., .13 + r.normal(0, .006), YB, .8, t0=40)
        x += w
    for xc, bk in ((-5.92, None), (-5.58, None), (-4.88, None), (-4.52, .52), (-4.2, .2)):
        baluster(xc, YB, .66, .13, bk)
    post(-5.24, YB, .86)
    rail_piece(-6.3, -5.75, .79, .93, YB); rail_piece(-5.75, -5.42, .79, .935, YB)
    i, pts = rail_piece(-5.04, -4.62, .785, .925, YB, rough=1)                          # broken end of the rail
    S.fill(pts[np.argsort(pts[:, 0])[-3:]] + r.normal(0, 2, (3, 2)), '#7a668a')
    # newel stub at the outer corner
    post(-2.52, 11.6, .62, .40, cap=False, top_pts=[(-2.72, .36), (-2.725, .58), (-2.66, .605), (-2.60, .575), (-2.53, .585), (-2.47, .52), (-2.42, .515), (-2.37, .44), (-2.325, .425), (-2.32, .36)])

    # ------------------------------------------------------------ floor (grazing)
    y0, y1 = 8.55, 12.15
    base([pj(-6.9, y0, 0), pj(-2.55, y0, 0), pj(-2.15, 9.6, 0), pj(-2.2, 10.6, 0), pj(-2.3, y1, 0), pj(-6.9, y1, 0)], '#86748f')
    yy = y1
    ex = lambda Y: float(np.interp(Y, [8.5, 8.9, 9.3, 9.6, 10.2, 10.6, 11.2, 12.2], [-2.62, -2.55, -2.36, -2.12, -2.16, -2.28, -2.22, -2.26]))
    ys = [12.15]
    while ys[-1] > y0 + .15: ys.append(max(y0, ys[-1] - r.uniform(.38, .62)))
    for a, b in zip(ys[:-1], ys[1:]):
        x = -6.4 + r.uniform(0, .4)
        while x < ex((a + b) / 2) - .1:
            w = r.uniform(.5, 1.0); e = min(x + w, ex((a + b) / 2) + r.uniform(-.06, .05))
            m = cut(x, e, b, a, r, .014, .05, .2, .008)
            pts = np.array([pj(u, Y, 0) for u, Y in m])
            i = S.block(pts, mix(pick(SH), rgb('#a794ab'), .35), r, ang=0, hw=1.5, jit=(.006, .015, .025), edge=False)
            tt = 46 - (12 - a) * -1.2 + r.normal(0, 2); tmap(pts, tt)
            lit(i, pts, r.uniform(.8, 1), None, 0, .14, 1.2, 1.8)
            x += w
    # joints of the floor, thin
    # long shadows of balusters and post reach toward camera-left over the lit floor
    for xc, wd in ((-5.92, .1), (-5.58, .1), (-5.24, .2), (-4.88, .1), (-4.52, .08), (-2.52, .22)):
        a = np.array([pj(xc - wd, YB, 0), pj(xc + wd, YB, 0), pj(xc - 1.7 + wd * 1.2, 8.6, 0), pj(xc - 1.7 - wd * 1.2, 8.6, 0)])
        m = np.zeros((RH * 2, RW * 2), np.uint8); cv2.fillPoly(m, [np.round(G.P(a) * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
        k = cv2.GaussianBlur(m, (0, 0), 1.6).astype(f32) / 255. * .82
        G.cv.a *= 1 - k * (np.arange(RH * 2)[:, None] > pj(0, 12.1, 0)[1] * 2)

    # ------------------------------------------------------------ outer end face (sunlit, broken)
    base([pj(-2.6, 8.55, 0), pj(-2.2, 9.3, 0), pj(-2.16, 10., 0), pj(-2.25, 12.1, 0), pj(-2.25, 12.1, -.5), pj(-2.16, 10., -.56), pj(-2.2, 9.3, -.6), pj(-2.6, 8.6, -.62)], '#8a7390')
    for (Ya, Yb2, xa) in ((8.6, 9.25, -2.58), (9.25, 9.9, -2.2), (9.9, 10.75, -2.16), (10.75, 12.1, -2.25)):
        zt = 0.
        for hgt in (.16, .30, .14):
            xx = xa + r.normal(0, .03)
            m = cut(Ya, Yb2, zt - hgt, zt, r, .008, .05, .35, .01)
            pts = np.array([pj(xx, Y, z) for Y, z in m])
            i = S.block(pts, mix(pick(SH), rgb('#b08f92'), .5), r, ang=-.2, hw=2.2); tmap(pts, 44 + r.normal(0, 2)); lit(i, pts, .95, None, -.2, .12, 1.0, 3.4)
            zt -= hgt
    for (xa_, xb_, Yn, z1_, z0_) in ((-2.62, -2.2, 9.27, 0., -.17), (-2.62, -2.2, 9.27, -.17, -.46), (-2.6, -2.22, 9.3, -.46, -.62)):
        m = cut(xa_, xb_, z0_, z1_, r, .006, .06, .6, .01); pts = np.array([pj(u, Yn, z) for u, z in m])
        i = S.block(pts, mix(pick(SH), rgb('#b28e8c'), .22), r, ang=0, hw=2.6); tmap(pts, 46)
        rim(pj(xa_ + .1, Yn, z1_ - .01), pj(xb_ - .01, Yn, z1_ - .01), '#ffe6b4', 1.8, .7, 1, .2)
    # ------------------------------------------------------------ front face of the ledge
    base([pj(-6.9, 8.62, 0), pj(-2.6, 8.62, 0), pj(-2.62, 8.62, -.5), pj(-2.8, 8.74, -.66), pj(-6.9, 8.74, -.66)], '#665681')
    def face(z0, z1, Y, x1, ch, tone=1.0, t0=47., lip=False):
        x = -7.1 + r.uniform(0, .4)
        while x < x1 - .05:
            w = r.uniform(*ch); e = min(x + w, x1 + r.uniform(-.02, .02))
            if x1 - e < .2: e = x1
            m = cut(x, e, z0, z1, r, .009, .05, .25 if not lip else .35, .008, 5)
            pts = np.array([pj(u, Y, z) for u, z in m])
            bb = pick(SH) * r.uniform(.95, 1.05) * tone
            u = (x + 6.4) / 4.
            bb = mix(bb, rgb('#b28e8c'), .24 * u)                                        # rosy bounce toward the open end
            i = S.block(pts, bb, r, ang=0, hw=r.uniform(2.6, 4.6), grad=(.3, -.5, .06), light=(.5, -.85), dens=1.1)
            tmap(pts, t0 + r.normal(0, 2.5))
            if r.random() < .35: S.lichen(pts.mean(0) + r.normal(0, 1, 2) * (24, 7), r.uniform(6, 15), r, lab=i)
            if r.random() < .4: S.crack(pts[int(r.integers(len(pts)))], pts.mean(0) + r.normal(0, 12, 2), r, lab=i)
            if not lip:
                mn, mx = pts.min(0), pts.max(0)
                for kk in range(3):                                                         # broad weathered patches on the stone
                    cc = mix(bb, rgb(['#7a6a94', '#b39a98', '#a7927f', '#7f7596'][int(r.integers(4))]), r.uniform(.3, .6))
                    nn_ = int(r.integers(4, 9)); px = r.uniform(mn[0], mx[0]); py = r.uniform(mn[1], mx[1])
                    S.dabs(np.c_[px + r.normal(0, 22, nn_), py + r.normal(0, 9, nn_)], r.uniform(5, 11, nn_), cc, r, (.15, .45), .02, lab=i, asp=3., ang=0., dry=.8)
            if lip: rim(pj(x + .02, Y, z1 - .008), pj(e - .02, Y, z1 - .008), '#fff0c6', 2.6, .95, 1, .10)
            x = e if e < x1 else x1
            if e >= x1: break
    face(-.66, -.50, 8.74, -2.75, (.6, 1.1), .86)
    face(-.50, -.15, 8.62, -2.60, (.7, 1.35))
    face(-.15, .0, 8.5, -2.58, (.55, 1.2), 1.06, 45., True)
    # cast shadow of the lip on the fascia, and stain runs
    for k in range(70):
        xx = r.uniform(-6.3, -2.7); p = pj(xx, 8.6, -.155); L = r.uniform(8, 46)
        P = np.stack([p + np.array([r.normal(0, 1.2) * t, L * t]) for t in np.linspace(0, 1, 5)])
        S.cv.strokes(S.P(P)[None], r.uniform(2, 7), rgb('#5c4f7c'), r.uniform(.10, .32), r.uniform(.2, .6), seed=900 + k, shape='taper')
    for k in range(60):
        xx = r.uniform(-6.4, -2.65); a = pj(xx, 8.6, -.165); b = pj(xx + r.uniform(.1, .5), 8.6, -.165)
        S.cv.strokes(S.P(np.stack([a + (b - a) * t for t in (0, .33, .66, 1)]))[None], r.uniform(2.5, 5), rgb('#594c79'), r.uniform(.25, .5), .2, seed=1900 + k, shape='round')

    # ------------------------------------------------------------ the years on the fascia: spalls, chips, lichen, a moss run
    r2 = np.random.default_rng(515)
    F = lambda X, z, Y=8.6: pj(X, Y, z)
    def spall(X, z, w, h, seed):
        """a face that has flaked away: paler fresh stone in a hollow, dark lip above, lit lip below"""
        q = np.random.default_rng(seed); n = 13; th = np.linspace(0, 2 * np.pi, n, endpoint=False); rad = q.uniform(.72, 1.12, n) * np.where(q.random(n) < .25, .72, 1)
        c = F(X, z); e = np.c_[c[0] + np.cos(th) * w * rad, c[1] + np.sin(th) * h * rad]
        e = e + q.normal(0, 1.6, e.shape); _fil(e, '#6b5c85'); e2 = e + (1.2, 2.4); e2 = c + (e2 - c) * .95; _fil(e2, mix(rgb('#917c99'), rgb('#9c8699'), q.uniform(0, 1)))
        m = 16; S.dabs(np.c_[c[0] + q.normal(0, w * .38, m), c[1] + 3 + q.normal(0, h * .32, m)], q.uniform(2, 5, m), ['#8a7594', '#a28c9d', '#a9929c'][int(q.integers(3))], q, (.25, .6), .03, asp=2.6, ang=.15, dry=.6)
        lo = e2[np.argsort(e2[:, 0])]; lo = lo[lo[:, 1] > c[1]]
        for p0, p1 in zip(lo[:-1], lo[1:]): S.cv.strokes(S.P(np.stack([p0 + (p1 - p0) * t for t in (0, .33, .66, 1)]))[None], 1.2, rgb('#b79fa4'), .55, .3, seed=seed, shape='round')
    def chip(p, dx, dy, seed):
        """a corner knocked off: a facet that turns to the light, its shadow under it"""
        q = np.random.default_rng(seed); p = np.asarray(p, float); sx, sy = np.sign(dx), np.sign(dy)
        tri = np.array([p, p + (dx, sy * 2), p + (dx * .62, dy * .55), p + (dx * .25, dy * .9), p + (sx * 1, dy)]) + q.normal(0, 1.0, (5, 2))
        _fil(tri, '#675881'); _fil(p + (tri - p) * .82 + (sx * .5, 0), '#a28ca0'); _fil(p + (tri - p) * .40, '#85718f')
        S.cv.strokes(S.P(np.stack([tri[1] + (tri[4] - tri[1]) * t for t in (0, .33, .66, 1)]))[None], 1.1, rgb('#bba3a6'), .5, .3, seed=seed, shape='round')
    spall(-3.95, -.345, 34, 15, 1); spall(-3.16, -.27, 21, 11, 2); spall(-4.62, -.40, 16, 8, 3); spall(-2.92, -.57, 18, 6, 4)
    for (X, z, dx, dy, sd) in ((-4.42, -.155, 26, 20, 11), (-3.48, -.155, -30, 17, 12), (-2.66, -.16, -24, 26, 13), (-3.48, -.495, 20, -15, 14), (-4.42, -.495, -24, -14, 15), (-3.05, -.49, -17, -20, 16), (-5.06, -.155, -18, 14, 17), (-2.66, -.49, -16, -13, 18)): chip(F(X, z), dx, dy, sd)
    for (X, dx) in ((-3.98, 22), (-3.22, -20), (-4.75, 16), (-2.85, 14)):                        # bites out of the lip
        p_ = pj(X, 8.5, 0); _fil(np.array([p_ + (-abs(dx), -1), p_ + (-abs(dx) * .5, 5), p_ + (0, 9), p_ + (abs(dx) * .6, 6), p_ + (abs(dx), -1)]) + r2.normal(0, .8, (5, 2)), '#75658c')
        for u0, u1 in ((-1, -.45), (-.45, .05), (.05, .6)): rim(p_ + (u0 * abs(dx), 3 + 5 * (1 - abs(u0))), p_ + (u1 * abs(dx), 3 + 5 * (1 - abs(u1))), '#f2d6ae', 1.6, .75, 0, 0)
    for (X, z, rad) in ((-4.18, -.23, 17), (-4.10, -.29, 10), (-4.30, -.20, 8), (-3.34, -.40, 14), (-3.28, -.34, 8), (-2.84, -.30, 12), (-4.9, -.36, 13), (-3.72, -.08, 9), (-2.98, -.07, 8), (-3.66, -.585, 10)):
        S.lichen(F(X, z), rad, r2, cols=('#c0ab72', '#a9a287', '#d3bd92', '#8f9478', '#b7a98f'), n=int(22 + rad * 2))
        S.lichen(F(X, z), rad * .55, r2, cols=('#d9c596', '#c9bd9a'), n=12)
    # moss: cushions along the lip joint, then a run where the water comes down
    for (X0_, X1_, z, n_) in ((-4.86, -4.50, -.152, 60), (-3.86, -3.56, -.152, 46), (-3.02, -2.80, -.152, 30), (-4.8, -4.3, -.50, 44), (-3.7, -3.3, -.50, 34)):
        u = r2.beta(1.6, 1.6, n_); p_ = np.array([F(X0_ + (X1_ - X0_) * t, z) for t in u]); h_ = r2.gamma(1.6, 2.2, n_); p_[:, 1] += h_ - 1
        S.dabs(p_, np.clip(3.6 - h_ * .18, 1.1, 3.6), '#5b6353', r2, (.5, .9), .05, asp=1.5); S.dabs(p_[::2] + (.6, -1.4), 1.5, '#8b8c5a', r2, (.5, .9), .06, asp=1.4); S.dabs(p_[::5] + (.8, -2.2), 1.0, '#b9ad66', r2, (.6, .9), .05, asp=1.3)
    for (X, L_, wd, n_, sd) in ((-4.68, 118, 13, 150, 5), (-4.60, 74, 7, 60, 6), (-3.70, 64, 8, 62, 7), (-2.93, 40, 5, 32, 8)):
        q = np.random.default_rng(sd); c = F(X, -.16); t = q.beta(1.0, 1.7, n_); wob = 4 * np.sin(t * 7 + sd)
        p_ = np.c_[c[0] + wob + q.normal(0, 1, n_) * wd * (1 - .72 * t), c[1] + t * L_]
        S.dabs(p_, q.uniform(1.6, 3.8, n_) * (1 - .45 * t), '#566052', q, (.35, .8), .05, asp=2.2, ang=1.55); k_ = q.random(n_) < .42
        S.dabs(p_[k_] + (.9, -1.0), q.uniform(1., 2., k_.sum()), '#7f8357', q, (.4, .85), .06, asp=1.8, ang=1.55); k_ = q.random(n_) < .13
        S.dabs(p_[k_] + (1.2, -1.6), 1.1, '#b1a763', q, (.5, .9), .05, asp=1.4)
        for j in range(7):
            x_ = c[0] + q.normal(0, wd); S.cv.strokes(S.P(np.stack([np.array([x_ + q.normal(0, 1.2) * t_, c[1] + L_ * q.uniform(.8, 1.15) * t_]) for t_ in np.linspace(0, 1, 5)]))[None], q.uniform(2, 5), rgb('#5a5570'), q.uniform(.12, .3), .5, seed=sd * 9 + j, shape='taper')
    # ------------------------------------------------------------ near stub post and rubble
    post(-4.32, 8.75, .42, .40, cap=False, top_pts=[(-4.52, .26), (-4.525, .44), (-4.47, .455), (-4.41, .52), (-4.34, .51), (-4.29, .455), (-4.22, .44), (-4.18, .37), (-4.125, .36), (-4.12, .26)], t0=42.)
    for (xc, Y, w, h, rot) in ((-3.86, 9.0, .30, .17, .18), (-3.62, 8.9, .16, .10, -.3), (-3.1, 11.3, .34, .16, -.12), (-2.95, 11.0, .18, .12, .4), (-3.5, 11.2, .2, .1, .1), (-4.75, 8.9, .22, .13, .2), (-3.74, 8.75, .10, .07, .5)):
        m = cut(-w / 2, w / 2, 0, h, r, 0, .05, .5, .01, 3); c, s = math.cos(rot), math.sin(rot)
        pts = np.array([pj(xc + u * c - v * s, Y, max(0, u * s + v * c + .02)) for u, v in m])
        i = S.block(pts, pick(SH) * r.uniform(.92, 1.06), r, ang=rot, hw=1.8); tmap(pts, 46)
        ce = pts.mean(0)
        for a, b in zip(pts, np.roll(pts, -1, 0)):                                        # only edges that really face the sun, just inside the stone
            e = b - a; L_ = np.hypot(*e)
            if L_ < 2.5: continue
            nn = np.array([e[1], -e[0]]) / L_
            if nn @ ((a + b) / 2 - ce) < 0: nn = -nn
            sv = nn @ np.array([.55, -.83])
            if sv > .3: rim(a - nn * 1.0, b - nn * 1.0, '#fff0c8', 1.8 * sv, .9, 1, .05)
    # moss in the joints of the ledge
    for k in range(240):
        xx = r.uniform(-6.3, -2.7); z = r.choice([-.15, -.5, 0.0]) + r.normal(0, .012)
        p = pj(xx, 8.6, z); S.dabs(p[None] + r.normal(0, 2, (3, 2)), r.uniform(1., 2.6), r.choice(['#6d6f62', '#7c7a5c', '#5c6266']), r, (.3, .75), .05)
    L = S.layer(); Lg = G.layer()
    # light may only fall on stone
    Lg = Lg * np.minimum(1, L[..., 3:4])
    T = cv2.GaussianBlur(T, (0, 0), 1.2)
    return L, Lg, T


if __name__ == '__main__':
    import time
    from lp import comp
    t = time.time(); L, G, T = build(); print('stone', time.time() - t)
    bg = np.ones((RH, RW, 3), f32) * rgb('#f2d6a4')
    a = comp.over(bg, L); b = comp.over(a, G)
    comp.save('qa/s7/st_shade.jpg', a); comp.save('qa/s7/st_lit.jpg', b); comp.save('qa/s7/st_T.jpg', np.repeat((T / 60)[..., None], 3, 2))
