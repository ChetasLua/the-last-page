"""s9a The Last Page (f 0-150): the quiet balcony floor, the low sun, and the one blank page."""
import numpy as np, math, cv2, os
from lp import W, H, comp
from lp.paint import f32, rgb, mix, smooth, ramp, fbm, Canvas, arcs, jit_color, grid, mottle
from lp.ink import Cel, spl, over_roi
from shots.balc_sky import *
from shots.balc_stone import *
from shots.s7a_sky import pal, bank, PAL
from shots import balc_life as life

FC, CX, CY, EZ = 1658., 1240., 681., .36
YB = 7.5                                  # balustrade plane
SUN = (1640., 416.)
MF = dict(x=985, y=932, h=470, page=(1180, 985))
Y0 = 530                                  # top of the stone ROI
BZ = .92


def pj(X, Y, Z):
    X, Y, Z = np.asarray(X, float), np.asarray(Y, float), np.asarray(Z, float)
    return np.stack([CX + FC * X / Y, CY + FC * (EZ - Z) / Y], -1)


def proj3(P):
    P = np.asarray(P, float); y = np.maximum(P[:, 1], .05)
    return np.c_[CX + FC * P[:, 0] / y, CY + FC * (EZ - P[:, 2]) / y], P[:, 1]


def floor_xy(sx, sy):
    Y = FC * EZ / (sy - CY); return np.array([(sx - CX) / FC * Y, Y])


# ------------------------------------------------------------- balustrade pieces in (X, Z) metres
def parts():
    """hand-placed: posts, balusters (some broken or gone), rails.  Gap between the two posts right of centre lets the sun in."""
    r = np.random.default_rng(31)
    posts = [(-4.75, 1.0, None), (-2.62, 1.0, None), (-.42, 1.0, None), (.86, .62, 'l'), (1.92, .72, 'r'), (4.0, 1.0, None)]
    bal = []
    def run(a, b, n, sk=()):
        for k in range(n):
            x = a + (b - a) * (k + .5) / n + r.normal(0, .008)
            bal.append((x, sk[k] if k in sk and isinstance(sk, dict) else 1.0))
    run(-4.55, -2.82, 5, {}); run(-2.42, -.62, 5, {2: .42}); run(-.22, .68, 3, {2: .55}); run(2.12, 3.8, 5, {0: .5, 3: .0})
    rails = [(-5.6, -4.95), (-4.55, -3.6), (-3.6, -2.82), (-2.42, -1.35), (-1.35, -.62), (-.22, .30), (2.55, 3.1), (3.1, 3.8), (4.2, 5.2)]
    return posts, bal, rails


PZ = np.array([0, .05, .07, .12, .20, .30, .40, .52, .64, .74, .80, .86, .93, .95, 1.0]); PW = np.array([1.05, 1.05, .72, .62, .92, 1.16, 1.2, 1.02, .7, .5, .46, .62, .7, 1.02, 1.02]) * .105


def bal_poly(x, k, r):
    h = .62; z0 = .14; use = PZ <= k + 1e-6; w = PW * r.uniform(.94, 1.06)
    L = [(x - a * r.uniform(.95, 1.04), z0 + b * h) for a, b in zip(w[use], PZ[use])]; R = [(x + a * r.uniform(.95, 1.05), z0 + b * h) for a, b in zip(w[use], PZ[use])]
    top = [] if k >= .99 else [(x - .05, z0 + k * h + .03), (x + .01, z0 + k * h + .012), (x + .05, z0 + k * h - .02)]
    return np.array(L + top + R[::-1]), np.array(R)


def light_mask():
    """how much sun reaches each floor pixel (computed from the real silhouettes, softer the farther the shadow falls)"""
    s = 180; w, h = int(14 * s), int(3.2 * s)                     # X -7..7, Z 0..3.2
    B = np.zeros((h, w), np.uint8)
    posts, bal, rails = parts(); r = np.random.default_rng(1)
    f = lambda P: np.round((np.c_[(np.asarray(P)[:, 0] + 7) * s, np.asarray(P)[:, 1] * s]) * 16).astype(np.int32)
    rect = lambda a, b, c, d: cv2.fillPoly(B, [f([(a, c), (b, c), (b, d), (a, d)])], 255, cv2.LINE_AA, 4)
    for (x, ht, bk) in posts:
        rect(x - .2, x + .2, 0, ht * .92)
        if bk is None: rect(x - .27, x + .27, .86, 1.06)
    for (x, k) in bal:
        if k > .05: cv2.fillPoly(B, [f(bal_poly(x, k, r)[0])], 255, cv2.LINE_AA, 4)
    for (a, b) in rails: rect(a, b, .77, .93)
    for (a, b) in ((-7, -.2), (2.0, 7)): rect(a, b, 0, .14)
    rect(-.2, .62, 0, .13); rect(1.5, 2.1, 0, .10)
    B = B.astype(f32) / 255.
    P = [B] + [cv2.GaussianBlur(B, (0, 0), q) for q in (3, 7, 14, 26)]
    x, y = grid()
    Y = FC * EZ / np.maximum(y - CY, 1.); X = (x - CX) / FC * Y
    dx, dz = (SUN[0] - CX) / FC, (CY - SUN[1]) / FC               # sun ray slope per metre forward
    dist = np.clip(YB - Y, 0, None)
    U = (X + dx * dist + 7) * s; V = dist * dz * s
    q = np.clip(dist / 7.5, 0, 1) * 3.4 + .25
    out = np.zeros((H, W), f32)
    for i in range(5):
        wgt = np.clip(1 - np.abs(q - i), 0, 1)
        out += wgt * cv2.remap(P[i], U.astype(f32), V.astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return (1 - out) * (y > CY + 4)


# ----------------------------------------------------------------- sky
def sky9():
    r = np.random.default_rng(90)
    x, y = grid()
    d = np.hypot((x - SUN[0]) / 1500, (y - SUN[1]) / 1050)
    img = ramp(np.clip(d + .035 * fbm(H, W, (420, 160), (1, .4), 4), 0, 1.3), [(0, '#ffe2ac'), (.16, '#fccd98'), (.36, '#f6c8a4'), (.58, '#ead0bf'), (.82, '#c0ccca'), (1.05, '#98bccc'), (1.3, '#7eacc8')])
    g = np.exp(-((y - 700) / 130) ** 2)[..., None]; img = img + (rgb('#fbcfa0') - img) * g * .6
    img = sky_wash(img, r, 260, hw=(20, 70), ln=(260, 800), centre=SUN, op=(.06, .2))
    cv_ = Canvas(img=img)
    for (a, b, th, ln, n) in (((-80, 170), (900, 60), 30, 100, 170), ((300, 300), (1180, 235), 20, 80, 150), ((900, 120), (2100, 40), 26, 110, 170), ((-60, 400), (420, 372), 12, 70, 90)):
        a = np.array(a, float); b = np.array(b, float); D = b - a; L = np.hypot(*D); e = D / L; nn = np.array([-e[1], e[0]])
        t = r.beta(1.3, 1.3, n); v = r.normal(0, .5, n) + .4 * np.sin(t * 8 + th); env = np.sin(np.pi * t) ** .6
        p = a + np.outer(t, D) + nn * (v * th * env)[:, None]
        near = np.exp(-np.hypot((p[:, 0] - SUN[0]) / 800, (p[:, 1] - SUN[1]) / 400) ** 1.5)
        c = np.stack([mix(rgb('#f3e0c8'), rgb('#fff3cf'), q) for q in near])
        cv_.strokes(arcs(p[:, 0], p[:, 1], math.atan2(e[1], e[0]) + r.normal(0, .08, n), ln * r.uniform(.3, 1.2, n), r.normal(0, .2, n)), th * r.uniform(.06, .16, n), jit_color(c, r, .004, .008, .012), env * r.uniform(.12, .45, n), r.uniform(.2, .8, n), seed=int(r.integers(1 << 20)), shape='round', bristle=.03)
    img = cv_.img
    def tower(bl, seed, d, vs, lean=0., floor=(690, 60), lz=-.22, nm=400, sl=(.12, .86)):
        nonlocal img
        rr = np.random.default_rng(seed); T = Tree()
        for (bx, by, br) in bl: T.grow(bx, by, br * 1.06, br, rr, kids=(6,), lean=lean, rmin=2.5)
        m = Mass(T, sun=vs, seed=seed, lz=lz, rimw=8, floor=floor); sh, li, rc, _ = pal('B', d)
        c = np.ascontiguousarray(colour(m, sh, li, rc, sun_lo=sl[0], sun_hi=sl[1], rim=.85, seed=seed), f32)
        c = marks(m, c, rr, nm, size=(.04, .1), op=(.2, .6), val=.012); img = lay(img, m, c)
    tower([(170, 420, 80), (250, 470, 100), (80, 490, 110), (340, 540, 90), (190, 560, 130), (20, 600, 130), (420, 610, 70), (300, 630, 120), (500, 660, 60), (120, 680, 150)], 41, .50, (1400, 160), .3, floor=(690, 60))
    tower([(1960, 470, 90), (2050, 520, 120), (1880, 560, 90), (1980, 610, 120), (1800, 640, 70), (1900, 670, 100), (2060, 680, 140)], 43, .42, (1560, 180), -.3, floor=(690, 60))
    for (x0, x1, yc, R, s, dd) in ((380, 900, 672, 24, 51, .16), (860, 1240, 682, 18, 52, .10), (1180, 1640, 690, 14, 53, .06), (1600, 2100, 676, 24, 54, .12)):
        m = Mass(bank(x0, x1, yc, R, r, .8, .8, (5,), .7, big=.35), sun=(SUN[0], 100), seed=s, lz=-.3, rimw=3, floor=(700, 16)); sh, li, rc, _ = pal('B', dd)
        img = lay(img, m, colour(m, sh, li, rc, sun_lo=.04, sun_hi=.75, rim=.75, seed=s))
    a = smooth(696, 712, y + 4 * fbm(H, W, (60,), None, 5)); dd = np.clip((y - 700) / 130, 0, 1)
    c0 = mix(rgb('#fbd9ac'), rgb('#e0b9ac'), .0) * (1 - dd[..., None]) + rgb('#c9a3b0') * dd[..., None]
    img = img * (1 - a[..., None]) + c0 * a[..., None]
    for k, (yc, R, xs) in enumerate([(704, 8, [(-40, 2100)]), (712, 12, [(-60, 900), (960, 2100)]), (724, 17, [(-60, 700), (760, 1500), (1560, 2100)]), (742, 24, [(-60, 420), (560, 1260), (1400, 2100)]), (766, 32, [(-80, 800), (900, 1660), (1800, 2160)])]):
        for j, (a0, b0) in enumerate(xs):
            m = Mass(bank(a0, b0, yc, R, r, .55, .56, (5,), .78), sun=(SUN[0], 420), seed=200 + k * 7 + j, lz=-.35, rimw=3, floor=(yc + R * .7, R)); sh, li, rc, _ = pal('B', .10 + .12 * k)
            img = lay(img, m, colour(m, sh, li, rc, sun_lo=.1, sun_hi=.8, rim=.9, seed=200 + k * 7 + j))
    # the sun, behind thin gold air
    rd = np.hypot(x - SUN[0], y - SUN[1])
    img = img + (rgb('#ffd294') - img) * (np.exp(-(rd / 230) ** 1.5) * .8)[..., None]
    img = img + (rgb('#ffe9ba') - img) * (np.exp(-(np.maximum(0, rd - 40) / 56) ** 1.15) * .9)[..., None]
    a = np.clip((50 - rd) / 1.6 + .5, 0, 1)[..., None]
    img = img + (np.array([1.16, 1.07, .86], f32) - img) * a
    # two thin bars of cloud lie across the light
    cv_ = Canvas(img=img)
    for k in range(260):
        q = r.random() < .55; t = r.beta(1.6, 1.6); px = (1395 + 470 * t) if q else (1540 + 560 * t); py = (446 - 12 * t if q else 388 - 16 * t) + r.normal(0, 6 if q else 4) * math.sin(math.pi * t)
        nr = math.exp(-((px - SUN[0]) / 110) ** 2)
        cv_.strokes(arcs([px], [py], [-.03 + r.normal(0, .04)], [r.uniform(24, 110)], [r.normal(0, .15)]), [r.uniform(1.6, 5.)], jit_color(mix(rgb('#f1c79c'), rgb('#ffe3a8'), nr)[None], r, .004, .01, .015), [r.uniform(.25, .7) * math.sin(math.pi * t) ** .5], [r.uniform(.1, .6)], seed=k, shape='round', bristle=.03)
    img = cv_.img
    return img.astype(f32)


SH = ['#8d7690', '#97809a', '#84708f', '#a08a9b', '#8f7c9a', '#9a8494', '#8a7a9c']
LIT = ['#e9cda2', '#f0d5a6', '#e4c49c', '#f3dcb0', '#ebcb9c']


def stone9():
    """floor, balustrade: shade paint, sun paint (full), separately"""
    r = np.random.default_rng(404)
    hh = H - Y0
    S = Stone(0, Y0, W, hh); G = Stone(0, Y0, W, hh, ground='#e2c29c')
    pick = lambda L: rgb(L[int(r.integers(0, len(L)))])
    full = np.zeros((hh, W), np.uint8)

    def lit(i, pts, col=None, ang=0., hw=None, dens=1., op=1., val=.022):
        q = S.P(pts); mn = q.min(0); mx = q.max(0); sz = mx - mn
        lc = pick(LIT) if col is None else rgb(col) if isinstance(col, str) else col
        h_ = max(2., min(sz) * r.uniform(.10, .18)) if hw is None else hw * 2
        G.lab = S.lab; G.fill(pts, lc, .94 * op)
        n = int(np.clip(sz[0] * sz[1] / (h_ * h_ * 5) * dens, 6, 200)); x = r.uniform(mn[0], mx[0], n); y = r.uniform(mn[1], mx[1], n)
        G.cv.strokes(arcs(x, y, ang + r.normal(0, .07, n), np.minimum(h_ * r.uniform(4, 11, n), sz[0]), r.normal(0, .08, n)), h_ * r.uniform(.5, 1.2, n), jit_color(np.repeat(lc[None], n, 0), r, .006, .014, val), r.uniform(.3, .8, n) * op, r.uniform(0, .4, n),
                     seed=int(r.integers(1 << 20)), lab=S.lab, LAB=i, bristle=.05)

    def stroke(T, a, b, col, w, op=.9, seg=34, brk=.15, dry=.2, shape='round'):
        a = np.asarray(a, float); b = np.asarray(b, float); L = np.hypot(*(b - a)); m = max(1, int(L / seg))
        for k in range(m):
            if r.random() < brk: continue
            p0 = a + (b - a) * (k + r.uniform(0, .06)) / m; p1 = a + (b - a) * (k + 1 - r.uniform(0, .08)) / m
            P = np.stack([p0 + (p1 - p0) * t + r.normal(0, .25, 2) for t in (0, .33, .66, 1)])
            T.cv.strokes(T.P(P)[None], w * 2 * r.uniform(.35, .65), jit_color(rgb(col)[None], r, .004, .01, .02), op * r.uniform(.6, 1), r.uniform(0, dry), seed=int(r.integers(1 << 20)), shape=shape, bristle=.03)

    # ------------------------------------------------ floor: worn flags, each its own stone
    S.fill([(-5, 757), (W + 5, 757), (W + 5, H + 5), (-5, H + 5)], '#635478')
    G.fill([(-5, 757), (W + 5, 757), (W + 5, H + 5), (-5, H + 5)], '#b08f84', .85)
    th0 = math.radians(27); ct, st_ = math.cos(th0), math.sin(th0)
    toXY = lambda uv: np.c_[uv[:, 0] * ct - uv[:, 1] * st_ + .35, uv[:, 0] * st_ + uv[:, 1] * ct]
    def arris(T, pts, hi, lo, w, oh=.8, ol=.75, light=(.5, -.87)):
        ctr = pts.mean(0); n_ = len(pts)
        for k in range(n_):
            p, q = pts[k], pts[(k + 1) % n_]; e = q - p; L = np.hypot(*e)
            if L < 3: continue
            nn = np.array([e[1], -e[0]]) / L
            if nn @ ((p + q) / 2 - ctr) < 0: nn = -nn
            sv = nn @ np.array(light)
            if sv > .2 and hi is not None and r.random() < .85: stroke(T, p, q, hi, w * (.5 + .5 * sv), oh * min(1, sv * 1.3), 70, .12)
            elif sv < -.2 and lo is not None and r.random() < .9: stroke(T, p, q, lo, w * 1.3 * (.5 - .5 * sv), ol * min(1, -sv * 1.3), 70, .10)
    flags = []; v = -3.2
    while v < 8.5:
        dv_ = r.uniform(.42, .78); u = -8. + r.uniform(0, .6)
        while u < 9.:
            w = r.uniform(.42, .92) * (1 if r.random() > .16 else .58)
            m = cut(u, u + w, v, v + dv_, r, gap=r.uniform(.004, .013), chip=.06, pchip=.28, wob=.007, n=4); m += r.normal(0, .003, m.shape)
            XY = toXY(m)
            q = life._clip_half(XY, np.array([0., 1.]), 1.16)
            if q is not None and q[:, 1].min() < 7.9:
                q = life._clip_half(q, np.array([0., -1.]), -7.75)
                if q is not None:
                    pts = pj(q[:, 0], q[:, 1], 0 * q[:, 0])
                    if pts[:, 0].max() > -20 and pts[:, 0].min() < W + 20: flags.append((pts, q.mean(0)))
            u += w
        v += dv_
    flags.sort(key=lambda t: -t[1][1])
    for pts, (fx, fy) in flags:
        near = np.clip((3.2 - fy) / 2, 0, 1)
        bc = mix(pick(SH), rgb('#9583a0'), .30) * r.uniform(.91, 1.06)
        bc = mix(bc, rgb('#b59698'), .10 + .12 * np.clip(fx, -1, 1))
        d = pj(fx + .1 * ct, fy + .1 * st_, 0) - pj(fx - .1 * ct, fy - .1 * st_, 0); an = math.atan2(d[1], d[0])
        hw = float(np.clip(1.5 + 3.6 * near, 1.3, 5.4))
        i = S.block(pts, bc, r, ang=an, hw=hw, jit=(.007, .02, .026), dens=1.1, light=(.45, -.9), grad=(.3, -.4, .05), edge=False)
        lc = mix(pick(LIT), rgb(['#dcb793', '#e6cfae', '#d9b99c'][int(r.integers(3))]), r.uniform(0, .5))
        lit(i, pts, lc, an, hw * 1.15)
        mn, mx = pts.min(0), pts.max(0); sz = mx - mn
        if sz[1] > 10:
            for kk in range(int(r.integers(1, 4))):                  # worn and stained patches, both in shade and in sun
                nn_ = int(r.integers(5, 11)); px = r.uniform(mn[0], mx[0]); py = r.uniform(mn[1], mx[1]); q = np.c_[px + r.normal(0, sz[0] * .12, nn_), py + r.normal(0, sz[1] * .14, nn_)]
                k_ = int(r.integers(0, 4)); s_ = np.clip(sz[1] * r.uniform(.06, .15, nn_), 1.5, 10)
                S.dabs(q, s_, mix(bc, rgb(['#75658f', '#b59c9c', '#8c8573', '#6f6a92'][k_]), .45), r, (.15, .45), .02, lab=i, asp=3.4, ang=an, dry=.8)
                G.lab = S.lab; G.dabs(q, s_, mix(lc, rgb(['#c99f86', '#fff0cc', '#c4ad84', '#cfa794'][k_]), .6), r, (.15, .5), .02, lab=i, asp=3.4, ang=an, dry=.8)
            if r.random() < .42:
                a_, b_ = pts[int(r.integers(len(pts)))], pts.mean(0) + r.normal(0, 1, 2) * sz * .25
                st = r.bit_generator.state; S.crack(a_, b_, r, '#5d4d72', 1.1, lab=i); r.bit_generator.state = st; G.crack(a_, b_, r, '#a07c74', 1.1, lab=i)
            if r.random() < .34:
                c_ = pts.mean(0) + r.normal(0, .2, 2) * sz; rad = np.clip(sz[1] * r.uniform(.12, .28), 4, 18)
                st = r.bit_generator.state; S.lichen(c_, rad, r, cols=('#9d967c', '#8b8577', '#a79888'), n=20, lab=i); r.bit_generator.state = st; G.lichen(c_, rad, r, cols=('#d9c489', '#c9bb8e', '#e6d3a0'), n=20, lab=i)
        w = float(np.clip(.7 + near * 1.7, .7, 2.5))
        arris(G, pts, '#fff2d0', '#b0887e', w); arris(S, pts, '#b3a0b4', '#5f5076', w, .5, .6)
    # moss and dirt in the joints
    nj = 900; ys, xs = np.nonzero((S.lab == 0)); sel = r.choice(len(xs), min(nj, len(xs)), replace=False)
    for k in sel:
        p = np.array([xs[k] / 2, ys[k] / 2 + Y0])
        if p[1] < 770: continue
        z = np.clip((p[1] - 760) / 390, 0, 1); S.dabs(p[None] + r.normal(0, 1, (2, 2)) * (3 + 5 * z, 1 + z), 1 + 2.6 * z * r.uniform(.4, 1), ['#63685a', '#6d6d62', '#585f5c'][int(r.integers(3))], r, (.3, .8), .05)
        G.dabs(p[None] + r.normal(0, 1, (2, 2)) * (3 + 5 * z, 1 + z), 1 + 2.6 * z * r.uniform(.4, 1), ['#a5985c', '#b7a662', '#94895c'][int(r.integers(3))], r, (.3, .8), .05)
    floor_lab = (S.lab > 0).copy()
    FG = G.layer(); FS = S.layer()

    # ------------------------------------------------ balustrade, backlit
    B = Stone(0, 400, W, 420, ground='#8d7690')
    R2 = B
    posts, bal, rails = parts(); rb = np.random.default_rng(1)
    def piece(m, tone=1.0, ang=0., hw=1.8, base=None, Y=YB, solid=False):
        pts = pj(m[:, 0], np.full(len(m), Y), m[:, 1]); bc = (mix(pick(SH), rgb('#94809c'), .3) if base is None else base) * tone
        B.fill(pj(np.r_[m[:, 0].min(), m[:, 0].max(), m[:, 0].max(), m[:, 0].min()] + np.array([.006, -.006, -.006, .006]), np.full(4, Y), np.r_[m[:, 1].min() - .01, m[:, 1].min() - .01, m[:, 1].max() - .012, m[:, 1].max() - .012]), '#675680') if solid else None
        i = B.block(pts, bc, r, ang=ang, hw=hw, light=(.6, -.8), grad=(.5, .0, .05)); return i, pts, bc
    def rim(a, b, w=2.0, col='#fff0c8', op=.95, brk=.08): stroke(B, a, b, col, w, op, 30, brk)
    def sunlit(pts, w=2.0, light=(.55, -.83), lo=.28, ymax=None):
        """burning edge on the arrises that truly face the sun, laid just inside the stone's own outline"""
        pts = np.asarray(pts, float); ce = pts.mean(0)
        for a, b in zip(pts, np.roll(pts, -1, 0)):
            e = b - a; L = np.hypot(*e)
            if L < 2.5 or (ymax is not None and (a[1] + b[1]) / 2 > ymax): continue
            nn = np.array([e[1], -e[0]]) / L
            if nn @ ((a + b) / 2 - ce) < 0: nn = -nn
            sv = float(nn @ np.array(light))
            if sv > lo: rim(a - nn * 1.1, b - nn * 1.1, w * (.55 + .45 * sv), '#fff0c8' if nn[1] < -.5 else '#ffe6b6')
    # plinth course
    x = -6.6
    while x < 6:
        w = r.uniform(.6, 1.1); e = x + w
        if e < .66 or x > 2.0 or (1.45 < x < 2.0):
            z1 = .14 + r.normal(0, .004); i, pts, bc = piece(cut(x, e, 0, z1, r, .004, .025, .2, .004), .92, solid=True); rim(pj(x + .02, YB, z1 - .008), pj(e - .02, YB, z1 - .008), 1.8)
        x = e
    for (x, k) in bal:
        if k < .05: continue
        m, Rr = bal_poly(x, k, rb); i, pts, bc = piece(m, r.uniform(.90, 1.0), np.pi / 2, 1.6)
        rp = pj(Rr[:, 0], np.full(len(Rr), YB), Rr[:, 1]); lp_ = pj(2 * x - Rr[:, 0], np.full(len(Rr), YB), Rr[:, 1])
        for a, b in zip(rp[:-1], rp[1:]):
            stroke(B, a + (-4.2, 0), b + (-4.2, 0), '#cf9f90', 3.4, .55, 30, .15)
            if r.random() > .08: rim(a + (-1.2, 0), b + (-1.2, 0), 2.6, '#ffe9bc')
        for a, b in zip(lp_[:-1], lp_[1:]):
            if r.random() > .5: stroke(B, a + (1, 0), b + (1, 0), '#b7a0ae', 1.4, .5, 30, .2)
        # collars
        for zz in (.14 + .62 * .06, .14 + .62 * .82, .14 + .62 * .94):
            if zz < .14 + k * .62: stroke(B, pj(x - .1, YB, zz), pj(x + .1, YB, zz), '#6f5d82', 1.4, .6, 40, 0)
    for (x, ht, bk) in posts:
        zs = [0, .34, .62, .92] if bk is None else [0, .34, ht * .92 if ht > .65 else .56]
        for z0, z1 in zip(zs[:-1], zs[1:]):
            m = cut(x - .2, x + .2, z0, z1, r, .003, .03, .22, .004); i, pts, bc = piece(m, r.uniform(.94, 1.03), 0., 2.2, solid=True)
            s = np.array([pj(x + .2, YB, z0 + .005), pj(x + .2, YB, z1 - .005), pj(x + .2, YB + .4, z1 - .005), pj(x + .2, YB + .4, z0 + .005)])
            if x < 1.5 and False: pass
            rim(pj(x + .193, YB, z0 + .02), pj(x + .193, YB, z1 - .02), 3.0, '#ffe9bc')
            stroke(B, pj(x + .17, YB, z0 + .02), pj(x + .17, YB, z1 - .02), '#d3a592', 5., .55, 30, .1)
            if r.random() < .6: B.lichen(pts.mean(0) + r.normal(0, 8, 2), r.uniform(6, 11), r, cols=('#a89d74', '#9a947c', '#b8a987'), n=18, lab=i)
            if r.random() < .5: B.crack(pts[int(r.integers(len(pts)))], pts.mean(0) + r.normal(0, 8, 2), r, '#5d4d72', .9, lab=i)
        if bk is None:
            for (dw, z0, z1) in ((.04, .90, .95), (.075, .95, 1.03), (.02, 1.03, 1.07)):
                i, pts, bc = piece(cut(x - .2 - dw, x + .2 + dw, z0, z1, r, .001, .016, .2, .003), 1.02, solid=True); rim(pj(x - .2 - dw, YB, z1 - .006), pj(x + .2 + dw, YB, z1 - .006), 2.4)
        else:
            t = zs[-1]
            m = np.array([(x - .2, t - .015), (x - .2, t + .085), (x - .15, t + .10), (x - .08, t + .092), (x - .04, t + .055), (x + .03, t + .06), (x + .08, t + .03), (x + .15, t + .025), (x + .2, t - .005), (x + .2, t - .015)])
            if bk == 'r': m = np.array([(2 * x - u, z) for u, z in m])[::-1]
            i, pts, bc = piece(m, 1.0, .3)
            pp = pj(m[:, 0], np.full(len(m), YB), m[:, 1]); sunlit(pp, 2.4, ymax=pj(x, YB, t)[1] - 1.5)
    for (a, b) in rails:
        for (z0, z1, t) in ((.765, .83, .93), (.83, .925, 1.0)):
            m = cut(a, b, z0, z1 + r.normal(0, .003), r, .002, .022, .25, .004); i, pts, bc = piece(m, t * r.uniform(.96, 1.04), 0., 2., solid=True); rim(pj(a + .01, YB, z1 - .008), pj(b - .01, YB, z1 - .008), 2.2 if t == 1. else 1.2, '#fff0c8', .95 if t == 1. else .5)
        stroke(B, pj(b - .01, YB, .78), pj(b - .01, YB, .92), '#ffe6b4', 2.2, .9, 30, 0)
    # fallen stones by the gap
    for (xc, Y, w, h, rot) in ((.95, 7.0, .42, .22, .14), (1.36, 6.8, .24, .15, -.3), (1.62, 7.2, .30, .17, .22), (1.16, 6.5, .14, .09, .5), (2.25, 6.6, .34, .2, -.12), (.62, 6.4, .12, .08, .3), (2.6, 6.2, .16, .1, .2)):
        m = cut(-w / 2, w / 2, 0, h, r, 0, .05, .5, .006, 2); c, s = math.cos(rot), math.sin(rot)
        mm = np.array([(xc + u * c - v * s, max(0, u * s + v * c + .02)) for u, v in m]); i, pts, bc = piece(mm, r.uniform(.92, 1.04), rot, 2., Y=Y)
        sunlit(pts, 2.2)
    BL = B.layer()
    return FS, FG, BL


def quiet_things(bg):
    """a few small true things on the nearest flags"""
    r = np.random.default_rng(88)
    sm = cv2.GaussianBlur(bg, (0, 0), 5)
    at = lambda p: sm[int(np.clip(p[1], 0, H - 1)), int(np.clip(p[0], 0, W - 1))]
    cv_ = Canvas(img=bg)
    def path(pts, w, dark, op, seed, taper=True, hi=True):
        P = life.resample(spl(np.asarray(pts, float), corners=range(1, len(pts) - 1)), 5.); n = len(P); P = P + r.normal(0, .45, P.shape)
        for i in range(0, n - 3, 3):
            q = P[i:i + 4]
            if len(q) < 4: break
            k = 1 - .75 * i / n if taper else 1.; c = at(q[1])
            if hi: cv_.strokes((q + (0, w * 1.3))[None], w * .5 * k + .25, np.clip(c * 1.07 + .012, 0, 1), op * .55, .2, seed=seed + i, shape='round')
            cv_.strokes(q[None], w * .55 * k + .25, c * np.array(dark, f32), op * r.uniform(.7, 1), 0, seed=seed + i + 900, shape='round')
    # a hairline crack wandering across the big near stone, with one short branch
    path([(1012, 1124), (1060, 1106), (1092, 1109), (1138, 1090), (1171, 1092), (1206, 1076), (1262, 1070)], 1.3, (.66, .60, .70), .8, 10)
    path([(1138, 1090), (1156, 1101), (1184, 1104)], .9, (.70, .64, .73), .7, 40)
    path([(404, 1046), (436, 1040), (462, 1044), (500, 1032)], .9, (.70, .65, .74), .6, 70)
    # a chipped corner where three stones meet, and the flake that came off
    c = Cel(260, 130, 1250, 1010); g = at((1335, 1030)); dk = at((1395, 1050)) * .80
    c.fill(spl([(1318, 1041), (1336, 1030), (1354, 1034), (1372, 1043), (1369, 1052), (1346, 1047)], closed=True, corners=(0, 3, 4)), g * np.array([.70, .63, .70]))
    c.fill(spl([(1321, 1040), (1337, 1031.5), (1353, 1035), (1366, 1042), (1344, 1041)], closed=True, corners=(0, 3)), np.clip(g * 1.05 + .02, 0, 1))
    c.fill(spl([(1296, 1090), (1322, 1084), (1338, 1090), (1318, 1098), (1300, 1096)], closed=True), g * np.array([.64, .57, .68]), .8)
    fk = spl([(1304, 1087), (1318, 1078), (1337, 1081), (1341, 1088), (1322, 1093), (1307, 1092)], closed=True, corners=(0, 1, 2, 3))
    c.fill(fk, np.clip(g * .93, 0, 1)); c.shade(spl([(1300, 1088), (1322, 1086), (1344, 1086), (1340, 1098), (1300, 1098)], closed=True, corners=(0, 2, 3, 4)), g * np.array([.76, .68, .73]), within=fk)
    c.line(np.array([(1319, 1079.5), (1328, 1080.5), (1336, 1082)]), 1.3, '#fff0cc', (.2, .3), seed=3)
    for x, y, s_ in ((1352, 1094, 2.6), (1286, 1082, 1.9), (1362, 1080, 1.5)): c.fill(spl([(x - s_ * 1.5, y), (x, y - s_ * .7), (x + s_ * 1.5, y), (x, y + s_ * .6)], closed=True), g * np.array([.74, .67, .72]))
    cv_.img = np.ascontiguousarray(over_roi(cv_.img, c.rgba(), c.ox, c.oy), f32)
    # moss in two joints
    def moss(a, b, n, seed):
        q = np.random.default_rng(seed); a = np.array(a, float); b = np.array(b, float); t = q.beta(1.5, 1.5, n); env = np.sin(np.pi * t) ** .7
        p = a + np.outer(t, b - a) + np.c_[q.normal(0, 5, n) * env, q.normal(0, 2.6, n) * env]
        L = np.array([at(v).mean() for v in p]) / .70
        for (f_, off, sz, cols, o) in ((1.0, (0, 0), (2.4, 5.0), ['#5f6550', '#6c6d52', '#596055'], (.55, .9)), (.55, (.8, -1.6), (1.4, 2.8), ['#8c8b55', '#9c965a'], (.5, .9)), (.22, (1.2, -2.6), (.9, 1.6), ['#c6b76c', '#d6c67c'], (.6, .95))):
            k = q.random(n) < f_; m = int(k.sum()); col = np.stack([rgb(cols[int(j)]) for j in q.integers(0, len(cols), m)]) * np.clip(L[k], .62, 1.08)[:, None].astype(f32)
            s_ = q.uniform(*sz, m) * (.5 + .5 * env[k])
            cv_.strokes(arcs(p[k, 0] + off[0], p[k, 1] + off[1], q.normal(0, .5, m), s_ * 2.0, q.normal(0, .8, m)), s_ * .62, jit_color(col, q, .008, .02, .04), q.uniform(*o, m), q.uniform(0, .4, m), seed=seed, shape='round')
    moss((790, 1014), (886, 1094), 120, 5); moss((1452, 1048), (1606, 1027), 130, 6); moss((1606, 1027), (1660, 1021), 26, 7)
    # two leaves that came down from the ivy
    def leaf(x, y, s_, rot_, cols, seed):
        u = np.array([(-1, 0), (-.72, -.42), (-.18, -.60), (.42, -.44), (1.18, .06), (.50, .40), (-.12, .56), (-.70, .42)]) * s_
        u = u + np.random.default_rng(seed).normal(0, .03 * s_, u.shape); ca, sa = math.cos(rot_), math.sin(rot_)
        T = lambda q: np.asarray(q, float) @ np.array([[ca, sa * .6], [-sa, ca * .6]]) + (x, y)                 # lying flat, foreshortened
        c = Cel(int(s_ * 4 + 30), int(s_ * 3 + 24), int(x - s_ * 2 - 15), int(y - s_ * 1.5 - 12)); g = at((x, y))
        b = spl(T(u), closed=True, corners=(0, 4))
        c.fill(b + (-5, 3.0), g * np.array([.66, .58, .70]), .75)
        c.fill(b, cols[0]); half = np.vstack([T([(-1, 0), (0, .02), (1.12, .02)]), T([(1.2, .8), (-1.1, .8)])]); c.shade(half, cols[1], within=b)
        c.line(T([(-1.5, .10), (-1, 0), (0, .02), (1.0, .02)]), 1.2, '#6a5640', (.3, .4), seed=seed); c.line(np.vstack([b, b[:1]]), 1.1, '#5f4b3c', (.05, .05), press=.5, seed=seed + 1, alpha=.85)
        for (p0, p1) in (((-.4, 0), (-.06, -.40)), ((-.4, .02), (-.08, .38)), ((.1, .0), (.5, -.2)), ((.1, .03), (.46, .22))): c.line(T([p0, ((p0[0] + p1[0]) / 2 + .03, (p0[1] + p1[1]) / 2), p1]), .7, '#7b6546', (.2, .5), seed=seed + 2, alpha=.7)
        c.line(T([(-.1, -.55), (.42, -.40), (1.08, .02)]), 1.3, '#fff0c4', (.3, .3), seed=seed + 5, alpha=.9)
        cv_.img = np.ascontiguousarray(over_roi(cv_.img, c.rgba(), c.ox, c.oy), f32)
    leaf(600, 1072, 27, .42, ('#c2a85c', '#967e50'), 3); leaf(1540, 957, 20, -.55, ('#b5ac62', '#8a8654'), 4)
    return cv_.img


def plate9():
    sk = sky9(); FS, FG, BL = stone9(); lm = light_mask()
    x, y = grid()
    fl = np.zeros((H, W), f32); fl[Y0:] = FS[..., 3]
    shade = np.zeros((H, W, 3), f32); shade[Y0:] = FS[..., :3]
    sun = np.zeros((H, W, 4), f32); sun[Y0:] = FG
    # the floor drops into a deeper, cooler value toward the camera-left corner
    dv = 1 - .30 * smooth(.2, 1.1, np.hypot((x - 1500) / 1700, (y - 880) / 620)); shade *= dv[..., None] * np.array([1, 1, 1], f32)
    # sun, broken by the balustrade; painted edge, not a computed one
    lmw = np.clip(lm + .10 * fbm(H, W, (26, 9), (1, .6), 12, aspect=3.) * np.clip(4 * lm * (1 - lm), 0, 1), 0, 1)
    k = smooth(.22, .80, lmw)
    fall = .40 + .60 * np.exp(-np.hypot((x - 1330) / 860, (y - 900) / 270) ** 1.8)
    fall = np.clip(fall + .06 * fbm(H, W, (220, 70), (1, .5), 14, aspect=2.5), 0, 1)
    sa = sun[..., 3] * k * np.clip(fall, 0, 1)
    warm = sun[..., :3] / np.maximum(sun[..., 3:4], 1e-3)
    floor = shade * (1 - sa[..., None]) + warm * sa[..., None]
    # half-tone edge to the light: a rose transition between sun and shade
    e = np.clip(4 * k * (1 - k), 0, 1) * fl * .30; floor = floor + (rgb('#d09b8d') * fl[..., None] - floor * 1.0) * e[..., None]
    bal = np.zeros((H, W, 4), f32); bal[400:820] = BL
    bg = sk * (1 - fl[..., None]) + floor
    bg = comp.over(bg, bal)
    # glowing haze where floor meets the light
    hz = np.exp(-((y - 770) / 46) ** 2) * np.exp(-((x - 1600) / 620) ** 2) * .20
    bg = bg + (rgb('#ffdca0') - bg) * hz[..., None]
    bg = quiet_things(np.ascontiguousarray(bg, f32))
    # sunbeams: the sky's brightness dragged out from the sun past whatever stands in its way
    occ = np.clip(1 - bal[..., 3] - fl, 0, 1) * np.exp(-np.hypot((x - SUN[0]) / 620, (y - SUN[1]) / 420) ** 2)
    q = cv2.resize(occ, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(f32); sx, sy = SUN[0] / 4, SUN[1] / 4
    acc = np.zeros_like(q)
    N_ = 110
    for i in range(N_):
        t = i / N_ * .93
        acc += cv2.remap(q, xx + (sx - xx) * t, yy + (sy - yy) * t, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * (1 - t) ** .25
    acc /= N_
    acc = acc / np.percentile(acc, 99.5)
    beam = cv2.resize(acc, (W, H), interpolation=cv2.INTER_CUBIC)
    # a few broader hand-placed shafts falling to the left
    sh = np.zeros((H, W), f32)
    for (a0, wd, st) in ((2.50, .060, 1.0), (2.31, .030, .75), (2.72, .038, .9), (2.10, .026, .6), (2.93, .03, .6), (2.40, .016, .6), (2.83, .014, .5), (1.86, .03, .35)):
        an = np.arctan2(y - SUN[1], x - SUN[0]); rd = np.hypot(x - SUN[0], y - SUN[1])
        sh += st * np.exp(-((an - a0) / wd) ** 2) * smooth(90, 420, rd) * np.exp(-(rd / 1500) ** 2) * (1 + .35 * np.sin(rd / 170. + a0 * 9))
    return dict(bg=bg, lm=lmw * fl, fl=fl, bal=bal[..., 3], beam=np.clip(beam, 0, 1.2), sh=np.clip(sh, 0, 1.2))


if __name__ == '__main__':
    import time
    t = time.time(); P = plate9(); print(time.time() - t)
    comp.save('qa/s9/bg.jpg', comp.finish(P['bg'], 3)); comp.save('qa/s9/lm.jpg', np.repeat(P['beam'][..., None], 3, 2))
