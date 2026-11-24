"""s9b  The title insert.  f 0..161.  Owner: ink lane.  setup() / frame(f) are called by shots/s9.py"""
import os, math, numpy as np, cv2
from lp import W, H, comp, foxact
from lp.paint import (rgb, mix, smooth, fbm, Canvas, arcs, repaint, jit_color, scatter_mask, mottle, tooth, f32, lum)
from lp.ink import Cel, spl, ribbon, xf, boil, over_roi, resample, noise1
from lp.cam import Camera3D
from lp.brush import Sheet, Stroke, blob, SparseInk, ink_at, transmit, sheen, fibre_tile, tile_to
from lp import ink_letters as IL, sheet3d as S3

VER = 'a6'
TEXT = 'Every story begins with a blank page.'
M9 = dict(px=430, py=820, scale=1.0, tail=(1830, 1010))
SUN = np.array([-.66, -.46])                    # ground direction the shadows run (world x, y): toward lower left on screen
PW, PD = 1.0, .62
TW, TH = 1840, 1141
PITCH = math.radians(63.5); DIST = 2.2; FOC = 1475 * DIST
ROT = math.radians(-2.3)
P = {}


def camera():
    fwd = np.array([.03, math.cos(PITCH), -math.sin(PITCH)]); fwd /= np.linalg.norm(fwd)
    return Camera3D(-fwd * DIST, (0, 0, 0), FOC, cx=1030, cy=528)


def page_surface(nu=161, nv=101):
    u, v = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv))
    x = (u - .5) * PW; y = (.5 - v) * PD
    def corner(cu, cv, r, a, p=2.0):
        d = np.hypot((u - cu) * PW, (v - cv) * PD)
        return a * np.clip(1 - d / r, 0, 1) ** p
    z = .0022 + corner(0, 0, .30, .085, 1.8) + corner(1, 0, .20, .022) + corner(1, 1, .25, .050, 1.9)
    z += .007 * np.sin(u * 6.5 + 1.1) * np.abs(v - .52) ** 2 * 4 * (.4 + u) + .004 * np.sin(v * 7 + 2) * smooth(.8, 1, u)
    z = np.maximum(z, .002)
    # lifted corners pull in a little
    x = x + corner(0, 0, .30, .030, 2.4) - corner(1, 1, .25, .016, 2.4); y = y - corner(0, 0, .30, .022, 2.4) + corner(1, 1, .25, .012, 2.4)
    c, s = math.cos(ROT), math.sin(ROT)
    return np.stack([x * c - y * s, x * s + y * c, z], -1), u, v


# ---------------------------------------------------------------- paintings
def paint_sheet():
    """the blank page, flat: gouache washes over laid, fibrous paper.  RGBA (deckle edge in alpha)"""
    r = np.random.default_rng(42)
    g = np.ones((TH, TW, 3), f32) * rgb('#f5e6c6')
    n1 = fbm(TH, TW, (420, 160, 60), (1, .6, .35), 1)
    g = g + .022 * n1[..., None] * np.array([.8, 1, 1.35], f32)
    x, y = np.meshgrid(np.linspace(0, 1, TW), np.linspace(0, 1, TH))
    ed = np.minimum(np.minimum(x, 1 - x) * TW, np.minimum(y, 1 - y) * TH)
    old = smooth(150, 0, ed + 40 * fbm(TH, TW, (120, 44), (1, .6), 2))
    g = g * (1 - old[..., None] * np.array([.045, .085, .16], f32))
    g = cv2.GaussianBlur(g, (0, 0), 2.2)
    ang = np.full((TH, TW), .04, f32) + .22 * fbm(TH, TW, (300, 120), (1, .5), 5)
    g = repaint(g, sizes=(34, 15), seed=7, angle=ang, thr=(0, .010), density=.9, aspect=6.5, jit=(.003, .008, .009), op=.75, dry=.10, bristle=.018)
    fib = tile_to(fibre_tile(), TH, TW); fib2 = tile_to(fibre_tile(1024, 23, 2600, 20, 90), TH, TW, 300, 140)
    g = g * (1 - .040 * fib[..., None] * np.array([.7, .95, 1.3], f32)) + .020 * fib2[..., None] * np.array([1, .95, .8], f32)
    g = g * tooth(TH, TW, 15, .011)
    fx = np.zeros((TH, TW), f32)
    for (ccx, ccy, sp, n_) in ((180, 980, 90, 22), (1640, 180, 90, 20), (1560, 1010, 120, 22), (260, 150, 80, 14), (1020, 1080, 160, 14), (900, 60, 160, 10), (1770, 620, 50, 10), (60, 520, 40, 8)):
        for _ in range(n_):
            cx, cy = ccx + r.normal(0, sp), ccy + r.normal(0, sp * .5); rad = min(9., r.gamma(1.3, 1.6) + .7)
            th_ = np.linspace(0, 2 * np.pi, 10, endpoint=False)
            q = np.c_[cx + np.cos(th_) * rad * (1 + r.normal(0, .18, 10)), cy + np.sin(th_) * rad * (1 + r.normal(0, .18, 10))]
            cv2.fillPoly(fx, [np.round(spl(q, closed=True, n=4) * 8).astype(np.int32)], float(r.uniform(.25, .7)), cv2.LINE_AA, 3)
    fx = .7 * cv2.GaussianBlur(fx, (0, 0), 1.2) + .4 * cv2.GaussianBlur(fx, (0, 0), 4)
    g = g * (1 - np.clip(fx, 0, .9)[..., None] * np.array([.10, .22, .36], f32))
    # faint creases where the page was bound and folded by the wind
    c = Cel(TW, TH, ss=1)
    for (pts, a) in (([(40, 70), (60, 420), (44, 800), (58, 1100)], .30), ([(1540, 0), (1640, 130), (1790, 260), (1840, 300)], .26), ([(1600, 1141), (1690, 1010), (1840, 930)], .22),
                     ([(0, 240), (120, 170), (230, 0)], .18)):
        q = spl(np.array(pts, float) + r.normal(0, 3, (len(pts), 2)), n=14)
        c.line(q, 2.6, '#b98f6c', (.3, .3), seed=int(pts[0][0]) + 3, alpha=a); c.line(q + [2.2, 2.0], 2.2, '#fff6dd', (.3, .3), seed=int(pts[0][0]) + 5, alpha=a * 1.6)
    l = c.rgba(); g = g * (1 - l[..., 3:]) + l[..., :3]
    # deckle
    nz = fbm(TH, TW, (46, 15, 5), (1, .55, .35), 9)
    nz = .45 * nz + .75 * fbm(TH, TW, (19, 7, 3), (1, .6, .4), 10)
    d = ed + 2.0 * nz - 4.5
    a = np.clip(d + .5, 0, 1)
    fr = smooth(7, 0, d) * a
    g = g * (1 - fr[..., None] * .05) + fr[..., None] * .06 * rgb('#fff3d8')
    return np.dstack([g, a]).astype(f32)


def stones(cam):
    """flagstones: label map, edge distance, local axes"""
    u, v, t = cam.uv((0, 0, 0), (1, 0, 0), (0, 1, 0))
    th = math.radians(13)
    c, s = math.cos(th), math.sin(th)
    wx = fbm(H, W, (260, 90), (1, .4), 21) * .006; wy = fbm(H, W, (260, 90), (1, .4), 22) * .006
    a = (u * c + v * s) + wx + 3.0; b = (-u * s + v * c) + wy + 3.0
    r = np.random.default_rng(12)
    rows = [0.]
    while rows[-1] < 6: rows.append(rows[-1] + r.choice([.30, .36, .42, .50]) * r.uniform(.92, 1.08))
    rows = np.array(rows) - .07
    ri = np.clip(np.searchsorted(rows, b.ravel()).reshape(b.shape) - 1, 0, len(rows) - 2)
    d = np.minimum(b - rows[ri], rows[ri + 1] - b)
    lab = np.zeros((H, W), np.int32); dcol = np.full((H, W), 9., f32); side = np.zeros((H, W), f32)
    for k in range(len(rows) - 1):
        m = ri == k
        if not m.any(): continue
        cs = [r.uniform(0, .4)]
        while cs[-1] < 6: cs.append(cs[-1] + r.uniform(.36, .78))
        cs = np.array(cs)
        tilt = r.normal(0, .03)
        aa = a + (b - rows[k]) * tilt
        ci = np.clip(np.searchsorted(cs, aa[m]) - 1, 0, len(cs) - 2)
        lab[m] = k * 40 + ci + 1
        dcol[m] = np.minimum(np.abs(aa[m] - cs[ci]), np.abs(cs[ci + 1] - aa[m]))
    # rounded worn corner: smooth-min of the two distances
    kk = .03
    h_ = np.clip(.5 + .5 * (d - dcol) / kk, 0, 1)
    e = d * (1 - h_) + dcol * h_ - kk * h_ * (1 - h_)
    chip = fbm(H, W, (120, 46, 17), (1, .62, .30), 31)
    e = e + .0040 * chip - .0070
    return lab, e.astype(f32), u, v, th


def leaf_shadow_poses(cam, n=4):
    """soft shadow of leafy twigs, four drawings of a small sway (screen masks, half res)"""
    out = []
    TX, TY = 2600, 2000
    Hm = cam.H_plane((-1.3, 1.1, 0), (1, 0, 0), (0, -1, 0), TX, TY, 2.6, 2.0)
    for k in range(n):
        sw = [-1, -.25, .7, .2][k]
        sf = np.zeros((TY, TX), np.uint8)
        r = np.random.default_rng(77)
        for (bx, by, ang, L, nl, sz) in ((250, 330, .62, 560, 17, 1.0), (520, 250, 1.05, 420, 13, .9), (380, 700, .18, 360, 11, 1.0), (330, 1250, -.30, 420, 12, 1.1), (640, 180, .30, 520, 14, .8),
                                         (2380, 1840, -2.2, 560, 15, 1.1), (2480, 1560, -2.75, 420, 12, 1.0), (2050, 1960, -1.9, 360, 10, .95), (1250, 1960, -1.2, 260, 8, .9)):
            phase = r.uniform(0, 6)
            t = np.linspace(0, 1, 30); bend0 = r.normal(0, .25)
            th = ang + (.35 + bend0) * t - .5 * t ** 2 + sw * .030 * t * (1 + .4 * math.sin(phase))
            px = bx + np.cumsum(np.cos(th)) * L / 30; py = by + np.cumsum(np.sin(th)) * L / 30
            cv2.polylines(sf, [np.round(np.c_[px, py] * 8).astype(np.int32)], False, 255, 6, cv2.LINE_AA, 3)
            for j in range(nl):
                q = (j + .5 + r.uniform(-.25, .25)) / nl; i = int(np.clip(q, 0, 1) * 29)
                sd = 1 if j % 2 else -1
                an = th[i] + sd * r.uniform(.35, 1.15) * (1 - .5 * q) + sw * .10 * (q + .25) * (1 + .6 * math.sin(phase + j * 1.7))
                ll = r.uniform(62, 104) * sz * (1 - .30 * q); ww = ll * r.uniform(.20, .30)
                tt = np.linspace(0, 1, 14)
                prof = np.sin(np.pi * tt ** .75) ** .9
                bend = sd * r.uniform(-.1, .35) * tt ** 2 * ll
                ex = px[i] + np.cos(an) * (tt * ll + 8) - np.sin(an) * bend; ey = py[i] + np.sin(an) * (tt * ll + 8) + np.cos(an) * bend
                up = np.c_[ex - np.sin(an) * prof * ww, ey + np.cos(an) * prof * ww]; dn = np.c_[ex + np.sin(an) * prof * ww, ey - np.cos(an) * prof * ww]
                cv2.fillPoly(sf, [np.round(np.vstack([up, dn[::-1]]) * 8).astype(np.int32)], 255, cv2.LINE_AA, 3)
        a = sf.astype(f32) / 255.
        soft = cv2.GaussianBlur(a, (0, 0), 3.0); soft2 = cv2.GaussianBlur(a, (0, 0), 7.5)
        nz = cv2.resize(fbm(TY // 4, TX // 4, (90, 40), (1, .5), 303), (TX, TY))
        mxx = smooth(-.8, .9, nz)
        a = np.clip((soft * (1 - mxx) + soft2 * mxx) * 1.15, 0, 1)
        m = cv2.warpPerspective(a, Hm, (W, H), flags=cv2.INTER_LINEAR)
        out.append(cv2.resize(m, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
    return out


def balustrade_shadow(u, v):
    """long shadow of a stone balustrade standing off frame to the right: balusters, then the rail, in ground coordinates along the sun"""
    L = SUN / np.hypot(*SUN); Q = np.array([-L[1], L[0]])
    p = u * L[0] + v * L[1]; q = u * Q[0] + v * Q[1]
    w1 = fbm(H, W, (120, 44), (1, .5), 51) * .0042; w2 = fbm(H, W, (120, 44), (1, .5), 52) * .0042
    q = q + w1; p = p + w2
    sp = .172
    k = np.round(q / sp + .35); dq = np.abs(q + .35 * sp - k * sp)
    s = (p + 1.45) / .78                                          # 0 foot .. 1 underside of the rail
    G = lambda c, w: np.exp(-((s - c) / w) ** 2)
    prof = .022 + .036 * G(.40, .17) + .018 * G(.90, .05) + .022 * G(.02, .07) + .013 * G(.72, .035)
    prof = np.where(s > .95, .040, prof)
    pen = .006 + .006 * np.clip(s, 0, 1)
    bar = smooth(prof + pen, prof - pen, dq)
    rail = smooth(.985, 1.005, s) * smooth(1.165, 1.138, s)
    return np.clip(np.maximum(bar * smooth(1.02, 1.0, s), rail), 0, 1)


def shadowed(c, amt=1.0):
    """gouache shadow mix: rose violet, never grey"""
    s = c * np.array([.665, .590, .735], f32)
    s = mix(s, np.array(rgb('#8d7690')), .16)
    return mix(c, s, amt)


def build():
    cam = camera(); r = np.random.default_rng(5)
    lab, e, u, v, th = stones(cam)
    L = SUN / np.hypot(*SUN)
    # ------- stone guide
    ids = np.unique(lab); pal = np.array([rgb(h) for h in ('#efd3a6', '#e6c698', '#ecce9f', '#dfbd93', '#e9caa4', '#dbb790')])
    lut = np.zeros((lab.max() + 1, 3), f32)
    for i in ids: lut[i] = np.clip(pal[r.integers(0, len(pal))] * r.uniform(.94, 1.05) + r.normal(0, .010, 3), 0, 1)
    g = lut[lab]
    n1 = fbm(H, W, (240, 90, 34), (1, .7, .4), 61); n2 = fbm(H, W, (60, 22, 9), (1, .7, .5), 62)
    g = g * (1 + .040 * n1[..., None]) + .018 * n1[..., None] * np.array([1, .2, -.8], f32)
    g = mix(g, np.ones_like(g) * rgb('#c6a596'), (smooth(.6, 1.6, n2) * .42)[..., None])
    worn = smooth(.0, .07, e)[..., None]
    g = g * (.90 + .10 * worn)
    ex = cv2.GaussianBlur(e, (0, 0), 1.2); gx = cv2.Sobel(ex, cv2.CV_32F, 1, 0); gy = cv2.Sobel(ex, cv2.CV_32F, 0, 1)
    gn = np.hypot(gx, gy) + 1e-6
    fs, _ = cam.project(np.array([[0, 0, 0], [L[0] * .1, L[1] * .1, 0]])); sd = fs[1] - fs[0]; sd /= np.hypot(*sd)
    face = (gx * sd[0] + gy * sd[1]) / gn                       # + : edge faces the sun
    bev = smooth(.017, .002, e) * smooth(-.004, .002, e)
    g = mix(g, np.ones_like(g) * rgb('#fff0cc'), (bev * smooth(.15, .8, face) * .75)[..., None])
    g = mix(g, np.ones_like(g) * rgb('#a37f83'), (bev * smooth(-.1, -.8, face) * .70)[..., None])
    joint = smooth(.0012, -.0018, e)
    jc = mix(np.ones_like(g) * rgb('#75585f'), np.ones_like(g) * rgb('#9a7c74'), smooth(-.5, 1.2, n2)[..., None])
    moss = smooth(.2, 1.1, n1 + .5 * n2) * smooth(.013, -.004, e)
    jc = mix(jc, np.ones_like(g) * rgb('#7d7a4b'), (moss * .8)[..., None])
    g = mix(g, jc, joint[..., None]); g = mix(g, np.ones_like(g) * rgb('#93885a'), (moss * (1 - joint) * .45)[..., None])
    lab2 = np.where(joint > .5, 0, lab)
    fan = math.atan2(sd[1], sd[0])
    ax, _ = cam.project(np.array([[0, 0, 0], [math.cos(th), math.sin(th), 0]])); ad = ax[1] - ax[0]
    ang = np.full((H, W), math.atan2(ad[1], ad[0]), f32) + .35 * fbm(H, W, (200, 80), (1, .5), 71)
    per = np.zeros(lab.max() + 1, f32); per[ids] = r.normal(0, .28, len(ids)); ang = ang + per[lab]
    lit = repaint(np.clip(g, 0, 1), sizes=(30, 14, 6.5), seed=3, angle=ang, lab=lab2, thr=(0, .016, .020), density=1.0, aspect=5.0, jit=(.008, .02, .024), op=.88, dry=.22, bristle=.045)
    lit = S3.soften_edges(lit, lab2)
    # pits, chips, lichen: individually placed
    cv = Canvas(img=lit)
    area = (joint < .5).astype(f32)
    px, py = scatter_mask(area * smooth(.2, 1.4, n2 + .3), 420, r)
    col = jit_color(np.tile(rgb('#b08d83'), (len(px), 1)), r, .01, .03, .05)
    cv.strokes(arcs(px, py, r.normal(math.atan2(ad[1], ad[0]), .5, len(px)), r.uniform(5, 22, len(px)), r.normal(0, .6, len(px))), r.uniform(1.0, 3.4, len(px)), col, op=r.uniform(.35, .8, len(px)), dry=.2, seed=8, shape='round')
    px, py = scatter_mask(area, 300, r)
    col = jit_color(np.tile(rgb('#f7e2b8'), (len(px), 1)), r, .006, .02, .02)
    cv.strokes(arcs(px, py, r.normal(math.atan2(ad[1], ad[0]), .35, len(px)), r.uniform(14, 70, len(px)), r.normal(0, .4, len(px))), r.uniform(1.6, 6.5, len(px)), col, op=r.uniform(.2, .55, len(px)), dry=.55, seed=9)
    lit = cv.img
    c = Cel()
    for (pts) in ([(40, 600), (92, 640), (120, 712), (168, 740)], [(1930, 360), (1962, 420), (1958, 480), (2010, 540)], [(620, 1152), (668, 1090), (742, 1062), (800, 1010)], [(1500, 20), (1560, 52), (1592, 110)], [(120, 118), (190, 92), (260, 100)]):
        q = np.array(pts, float); q = spl(q + r.normal(0, 4, q.shape), n=9); q = q + np.c_[noise1(len(q), 4, r), noise1(len(q), 4, r)] * 2.0
        c.line(q, 2.6, '#8a6a6c', (.2, .6), seed=int(pts[0][0]), alpha=.75); c.line(q + [1.8, 1.8], 1.5, '#f6ddb2', (.3, .6), seed=int(pts[0][0]) + 1, alpha=.6)
    l = c.rgba(); lit = lit * (1 - l[..., 3:]) + l[..., :3]
    # ------- page
    S, pu, pv = page_surface()
    N = S3.normals(S)
    att = np.dstack([pu * (TW - 1), pv * (TH - 1), N, S[..., 2]])
    buf, zb = S3.raster(cam, S, att, ss=2)
    sheet = paint_sheet()
    pg = S3.tex(sheet * np.dstack([np.ones((TH, TW, 3), f32) * 1, np.ones((TH, TW), f32)]), buf[..., :2], cv2.INTER_AREA if False else cv2.INTER_LINEAR)
    pa = pg[..., 3] * (zb > 0)
    # sun from upper left, low: curled corners turn away from it
    l3 = np.array([-L[0] * .80, -L[1] * .80, .60]); l3 /= np.linalg.norm(l3)
    nl = np.clip((buf[..., 2:5] * l3.astype(f32)).sum(-1), 0, 1)
    base = l3[2]
    rel = nl - base
    wob = cv2.resize(fbm(H, W, (70, 26), (1, .5), 91), (W * 2, H * 2)) * .012
    half = smooth(-.022, -.075, rel + wob); dark = smooth(-.16, -.20, rel + wob); hi = smooth(.03, .075, rel + wob)
    pc = pg[..., :3]
    pc = mix(pc, pc * np.array([.935, .870, .845], f32), half[..., None]); pc = mix(pc, pc * np.array([.84, .765, .79], f32), dark[..., None])
    pc = mix(pc, np.minimum(pc * np.array([1.045, 1.035, 1.0], f32) + .012, 1.0), hi[..., None])
    zz = buf[..., 5]
    pgc = S3.down(pc * pa[..., None], 2); pal_ = S3.down(pa, 2); zzs = S3.down(zz * (zb > 0), 2)
    # thickness: lit rim on edges that face the sun, warm dark line on the far side
    ea = cv2.GaussianBlur(pal_, (0, 0), 1.3); egx = cv2.Sobel(ea, cv2.CV_32F, 1, 0); egy = cv2.Sobel(ea, cv2.CV_32F, 0, 1)
    ef = egx * sd[0] + egy * sd[1]
    rim = np.clip(ef * 2.2, 0, 1) * pal_; und = np.clip(-ef * 2.2, 0, 1) * pal_
    pgc = pgc * (1 - und[..., None] * np.array([.10, .16, .18], f32)) + rim[..., None] * .06 * rgb('#fff6de')
    # page cast shadow on the stones (displaced by its own height)
    zsh = S[..., 2:3] * np.array([SUN[0], SUN[1]]) * 1.9
    Ssh = np.concatenate([S[..., :2] + zsh, np.zeros_like(S[..., 2:])], -1)
    sbuf, szb = S3.raster(cam, Ssh, np.dstack([S[..., 2]]), ss=1)
    sm = (szb > 0).astype(f32); sz = sbuf[..., 0]
    sh_hard = cv2.GaussianBlur(sm, (0, 0), 1.6); sh_soft = cv2.GaussianBlur(sm, (0, 0), 7.0)
    k = smooth(.004, .05, cv2.GaussianBlur(sz, (0, 0), 6))
    pshadow = sh_hard * (1 - k) + sh_soft * k
    contact = np.clip(cv2.GaussianBlur(sm, (0, 0), 5) * 1.1, 0, 1)
    bal = balustrade_shadow(u, v)
    leaves = leaf_shadow_poses(cam)
    # stone without moving shadows (page shadow + baluster are static)
    stat = np.clip(np.maximum(bal, pshadow * .96), 0, 1)
    P.update(cam=cam)
    litS = lit * (1 - .10 * (contact * (1 - pshadow))[..., None])
    shdS = shadowed(lit)
    # grass tufts growing in the joint at the frame edge (drawn blades, lit and shaded versions)
    gl, gs = grass_layers(r)
    # assemble lit / shadow plates with page on top
    def asm(st, pgk, gr):
        out = st * (1 - pal_[..., None]) + pgk
        return out * (1 - gr[..., 3:]) + gr[..., :3]
    lit_p = asm(mix(litS, shdS, stat[..., None]), mix(pgc, shadowed(pgc / np.maximum(pal_, 1e-3)[..., None]) * pal_[..., None], (bal * .0)[..., None]), gl)
    bal_pg = np.clip(bal, 0, 1) * pal_
    pgs = np.clip(shadowed(pgc / np.maximum(pal_, 1e-3)[..., None]), 0, 1) * pal_[..., None]
    lit_p = lit_p * (1 - pal_[..., None]) + (mix(pgc, pgs, bal_pg[..., None]) if True else 0) * (1 - gl[..., 3:]) + gl[..., :3] * (pal_[..., None] > 0) * 0 + lit_p * pal_[..., None] * 0 if False else lit_p
    shd_p = asm(shdS, pgs, gs)
    page_bal = mix(pgc, pgs, bal_pg[..., None])
    lit_p = np.where(pal_[..., None] > .001, (mix(litS, shdS, stat[..., None]) * (1 - pal_[..., None]) + page_bal) * (1 - gl[..., 3:]) + gl[..., :3], lit_p)
    return lit_p.astype(f32), shd_p.astype(f32), pal_.astype(f32), np.stack(leaves).astype(f32), np.stack([stat, pshadow]).astype(f32)


def grass_layers(r):
    """a few drawn blades at the frame edges"""
    out = []
    for pal in (('#c4b164', '#94904f', '#e2cf86'), ('#7f7f58', '#5f6a4a', '#93905f')):
        r = np.random.default_rng(313)
        cv = Canvas(layer=True)
        for (bx, by, n, lean, ln) in ((24, 300, 11, .5, 150), (70, 1140, 15, .35, 190), (1500, 1160, 8, -.2, 120), (2030, 250, 8, -.6, 120), (800, -8, 5, 2.6, 60)):
            for j in range(n):
                x0, y0 = bx + r.normal(0, 26), by + r.normal(0, 8)
                a0 = -math.pi / 2 + r.normal(lean * .6, .34); L = ln * r.uniform(.5, 1.15); cur = r.normal(lean, .5)
                t = np.linspace(0, 1, 9)
                th = a0 + cur * t ** 1.4
                q = np.c_[x0 + np.cumsum(np.cos(th)) * L / 9, y0 + np.cumsum(np.sin(th)) * L / 9]
                c = rgb(pal[r.integers(0, 3)]) * r.uniform(.92, 1.06)
                cv.strokes(q[None], r.uniform(2.6, 5.0), c[None], op=.97, seed=int(r.integers(1, 9999)), shape='taper', bristle=.04)
                if r.random() < .4: cv.strokes((q + [1.2, .6])[None], 1.0, (rgb(pal[2]) * 1.02)[None], op=.7, seed=int(r.integers(1, 9999)), shape='taper')
        out.append(cv.layer())
    return out


def write_ink(cam):
    """title, small line, satellite drops and the bloom on the flat sheet -> warped sparse ink on screen"""
    S, pu, pv = page_surface()
    att = np.dstack([pu * (TW - 1), pv * (TH - 1)])
    buf, zb = S3.raster(cam, S, att, ss=1)
    px, py = M9['px'], M9['py']
    pu_, pv_ = buf[py, px, :2]
    sc, oy = 1.17, 512.; ox = 958 - 527 * sc
    # run the lead-in out from under the paw (control points given on screen, carried on to the sheet)
    lead = []
    for (qx, qy, qr, *kd) in ((px + 66, py - 34, 9, 'w'), (px + 84, py - 78, 6.0), (px + 82, py - 150, 3.4), (px + 78, py - 250, 2.2), (px + 100, py - 350, 1.8), (px + 152, py - 448, 1.6)):
        su, sv = buf[int(qy), int(qx), :2]
        uy = (sv - oy) / sc; ux = (su - ox) / sc - IL.SH * uy
        lead.append((ux, uy, qr) + tuple(kd))
    title = lead + IL.LEAD[7:] + IL.TITLE[len(IL.LEAD):]
    st = Stroke(IL.place(title, sc, ox, oy), entry=-.5, load=(1.0, .14), depl=1.1, step=.7)
    sh = Sheet(TW, TH, K=5, margin=12)
    sh.add(st, 14, 70)
    fr = 14 + 56 * st.t
    for (x, y, rad, (ax, ay), an) in IL.title_drops(sc, ox, oy):
        j = int(np.argmin(np.hypot(st.X[:, 0] - ax, st.X[:, 1] - ay)))
        sh.add(blob(x, y, rad, seed=int(x * 7 + y), stretch=.5, ang=an), fr[j] + .4, fr[j] + 1.2)
    lw = IL.line_width(TEXT); s2 = .362
    for s, a, b in IL.line_strokes(TEXT, s2, 958 - lw * s2 / 2 + 8, 922, seed=21): sh.add(s, 88 + a * 24, 88 + b * 24)
    sh.bake()
    maps = sh.maps()
    # screen warp
    ys, xs = np.nonzero(zb > 0)
    rx0, rx1, ry0, ry1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    rw, rh = int(rx1 - rx0), int(ry1 - ry0)
    b2, z2 = S3.raster(cam, S, att, w=rw, h=rh, ss=2, ox=int(rx0), oy=int(ry0))
    wm = []
    for D, F in maps:
        d2 = S3.tex(D, b2, cv2.INTER_LINEAR) * (z2 > 0); f2 = S3.tex(np.minimum(F, 1e4), b2, cv2.INTER_NEAREST, val=1e4)
        wm.append((d2, f2))
    wm.append(toe_bleed(b2, z2, int(rx0), int(ry0), rw, rh))
    ink = SparseInk(wm, rw, rh, int(rx0), int(ry0), 2)
    # seal quad
    sw, shh = 104, 186
    sx, sy = 952, 634
    sm = IL.seal_mask(sw * 2, shh * 2)
    T = np.zeros((TH, TW), f32); 
    Mx = np.float32([[.5, 0, sx - sw / 2], [0, .5, sy]])
    c, s = math.cos(.018), math.sin(.018)
    Rm = np.float32([[c * .5, -s * .5, sx - sw / 2], [s * .5, c * .5, sy]])
    T = cv2.warpAffine(sm, Rm, (TW, TH), flags=cv2.INTER_AREA)
    seal = S3.down(S3.tex(T, b2) * (z2 > 0), 2)
    return ink, seal, (int(rx0), int(ry0), rw, rh), st, fr


def toe_bleed(b2, z2, ox, oy, rw, rh):
    """ink let out between the toes: a faint wet halo close to the paw and three darker toe-shaped wets, feathered along the fibres (2x ROI maps)"""
    from lp import foxbody
    px, py, s = M9['px'], M9['py'], M9['scale']; ang = M9.get('ang', -.55)
    c = Cel(); foxbody.paw_top(c, px, py, s, ang, 1., 'gold', 1, 2.8, key=1)
    m = (c.rgba()[..., 3] > .4).astype(np.uint8)
    dist = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    up = lambda a: cv2.resize(a[oy:oy + rh, ox:ox + rw].astype(f32), (rw * 2, rh * 2), interpolation=cv2.INTER_LINEAR)
    x, y = np.meshgrid(np.arange(W, dtype=f32), np.arange(H, dtype=f32))
    X, Y, Dp = up(x), up(y), up(dist)
    fib = S3.tex(tile_to(fibre_tile(), TH, TW), b2)
    fl = cv2.GaussianBlur(fib, (0, 0), 1.6)
    n1 = up(fbm(H, W, (46, 17), (1, .5), 131)); n2 = up(fbm(H, W, (9, 4), (1, .6), 132))
    ca, sa = math.cos(ang), math.sin(ang)
    fwd = (X - px) * ca + (Y - py) * sa
    front = smooth(-46, 34, fwd)
    lim = np.clip(21 + 5 * n1 + 20 * fl + 1.5 * n2, 6, 44)
    halo = smooth(1., .0, Dp / lim) ** 1.35 * front
    D = .165 * halo
    F = 10 + 24 * np.clip(Dp / 45., 0, 1) ** 1.4
    for (gx, gy, dx, dy, ln, wd, dk) in ((14, -76, .30, -.95, 46, 17, .60), (62, -40, .86, -.50, 50, 18, .66), (70, 18, .99, .12, 44, 16.5, .58)):
        q = math.hypot(dx, dy); dx, dy = dx / q, dy / q
        u = (X - (px + gx * s)) * dx + (Y - (py + gy * s)) * dy; v = -(X - (px + gx * s)) * dy + (Y - (py + gy * s)) * dx
        t = (u + 12) / (ln + 12)
        prof = wd * np.sqrt(np.clip(t, 0, 1)) * (1 - np.clip(t, 0, 1) ** 2.6) ** .7
        e = prof - np.abs(v + 2.2 * n1) + 5.0 * fl * smooth(-.1, .5, t) + 1.0 * n2
        w_ = smooth(-3.5, 4.5, e) * smooth(1.12, .92, t) * smooth(-.15, .05, t) * np.clip(1 - Dp / 46., 0, 1) ** .35
        D = np.maximum(D, (dk - .24 * np.clip(t, 0, 1)) * w_)
        F = np.where(w_ > .25, np.minimum(F, 10 + 13 * np.clip(t, 0, 1.2) ** 1.3), F)
    D = np.where((z2 > 0) & (Dp < 46), D * (.93 + .14 * fib), 0)
    return D.astype(f32), F.astype(f32)


def setup():
    if P.get('ready'): return
    os.makedirs('cache', exist_ok=True)
    fn = f'cache/s9b_{VER}.npz'
    cam = camera()
    if not os.path.exists(fn):
        lit, shd, pal, lv, st = build()
        np.savez(fn[:-4] + '.tmp.npz', lit=lit.astype(np.float16), shd=shd.astype(np.float16), pal=pal.astype(np.float16), lv=lv.astype(np.float16), st=st.astype(np.float16))
        os.replace(fn[:-4] + '.tmp.npz', fn)
    z = np.load(fn)
    P.update({k: z[k].astype(f32) for k in ('lit', 'shd', 'pal', 'lv', 'st')})
    P['ink'], P['seal'], P['roi'], P['stroke'], P['fr'] = write_ink(cam)
    P['cam'] = cam; P['ready'] = True


def leaf_pose(f):
    d = (f + 1) // 3
    seq = [0, 0, 1, 1, 2, 2, 2, 3, 3, 1, 0, 0, 0, 1, 2, 2, 3, 3]
    return seq[d % len(seq)]


def standin_paw(img, f, M, pal=('#d9793f', '#b25c42', '#f8ecd8', '#4b2c25')):
    """stand-in until the lead animates the key"""
    px, py, s = M['px'], M['py'], M['scale']
    up = np.interp(f, [0, 2, 5, 8], [1, .7, .15, 0])
    if up > .95: return img
    x, y = px - 60 * up * s, py - 10 + 150 * up * s
    c = Cel(ox=0, oy=0, ss=1)
    leg = spl(xf(np.array([[-40, -60], [-120, 60], [-330, 420], [-170, 520], [-10, 160], [40, 40]], float), s, tx=x, ty=y), closed=True)
    paw = spl(xf(np.array([[-70, 20], [-78, -40], [-40, -78], [0, -86], [44, -76], [78, -38], [70, 20], [10, 60]], float), s, tx=x, ty=y), closed=True)
    c.fill(leg, pal[0]); c.line(leg, 3, pal[3], closed=False); c.fill(paw, pal[2]); c.line(np.vstack([paw, paw[:1]]), 3, pal[3])
    for dx in (-30, 8, 44): c.line(xf(np.array([[dx, -70], [dx - 4, -44], [dx - 2, -24]], float), s, tx=x, ty=y), 2.4, pal[3])
    return comp.over(img, c.layer())


def frame(f):
    if not P.get('ready'): setup()
    m = cv2.resize(P['lv'][leaf_pose(f)], (W, H), interpolation=cv2.INTER_LINEAR) * .92
    img = P['lit'] * (1 - m[..., None]) + P['shd'] * m[..., None]
    stat = P['st'][0]
    # ------ ink
    x0, y0, w, h = P['roi']
    cov, wet = P['ink'].at(f)
    if f >= 10:
        sub = img[y0:y0 + h, x0:x0 + w]
        lit = 1 - np.clip(m[y0:y0 + h, x0:x0 + w] + P['st'][0][y0:y0 + h, x0:x0 + w] * .6, 0, 1)
        sub = sub * transmit(cov, wet) + (cov * (.004 + .012 * lit))[..., None] * rgb('#ffd9a0')
        sub = sub + (sheen(cov, wet) * (.30 + .45 * lit))[..., None] * rgb('#fff0d2')
        img[y0:y0 + h, x0:x0 + w] = sub
    # ------ seal
    if f >= 78:
        d = min(f - 78, 4)
        a = P['seal']
        k = [.0, .55, .86, .95, 1.0][d]                     # ink transfer while pressed and rocked
        if k > 0:
            sub = img[y0:y0 + h, x0:x0 + w]
            ys, xs = np.linspace(0, 1, h)[:, None], np.linspace(0, 1, w)[None]
            if d < 4:                                       # pressed, then rocked from the head of the seal to its foot
                ry, rx = np.nonzero(a > .1)
                t = (np.arange(h)[:, None] - ry.min()) / float(ry.max() - ry.min()) + .10 * (np.arange(w)[None] - rx.min()) / float(rx.max() - rx.min())
                t = t + .035 * P.setdefault('sealn', fbm(h, w, (22, 8), (1, .6), 88))
                e_ = [.0, .56, 1.3, 1.3][d]; lo = [0, 0, .20, 0][d]
                a = a * smooth(e_ + .025, e_ - .025, t) * (1 - .30 * smooth(lo + .12, lo - .12, t) * (d == 2)) * [0, .90, .95, 1.0][d]
            if d <= 4:                                      # the paper gives a little around the stone
                if 'sealf' not in P: P['sealf'] = cv2.morphologyEx(np.clip(P['seal'] * 2, 0, 1), cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
                dent = np.clip(cv2.GaussianBlur(P['sealf'], (0, 0), 6) * 1.5, 0, 1) * (1 - P['sealf']) * [0, .5, .9, .6, .25][d]
                sub = sub * (1 - dent[..., None] * np.array([.16, .22, .20], f32))
            ink = np.array([1.0, .0, .0], f32)
            vm = np.array(rgb('#c43a28')); trans = mix(np.ones(3, f32), np.array([.830, .205, .150], f32), a[..., None])
            wets = np.exp(-max(0, f - 82) / 16.) if f >= 80 else 1
            sub = sub * trans + a[..., None] * np.array([.02, .035, .028], f32)
            img[y0:y0 + h, x0:x0 + w] = sub * (1 - .06 * wets * a[..., None])
    # ------ paw
    img = paw_shadow(img, f)
    if 's9b' in foxact.ACT: img = foxact.draw(img, 's9b', f, M9)
    else: img = standin_paw(img, f, M9)
    z = 1.0 + .020 * comp.ease(f / 170.) 
    img = comp.cam2d(img, W / 2 + 4 * (z - 1) * 10, H / 2 - 6 * (z - 1) * 10, zoom=z, interp=cv2.INTER_LANCZOS4)
    # ------ fades
    if f >= 136:
        t = comp.ease(np.clip((f - 135) / 15., 0, 1))
        img = mix(img, paper_frame(), float(t))
    if f >= 150:
        t = comp.ease(np.clip((f - 149) / 12., 0, 1))
        img = img * (1 - float(t))
    return img.astype(f32)


def tail_flick(f):
    fd = f - f % 2; fl = 0.
    for a, v in ((116, -.18), (119, .55), (121, 1.15), (124, .8), (127, .25), (130, -.10), (133, 0.)):
        if fd >= a: fl = v
    return fl


def paw_shadow(img, f):
    """drawn shadows the fox throws on page and stone (sun upper right), from the marks"""
    px, py, s = M9['px'], M9['py'], M9['scale']
    up = float(np.interp(f, [0, 2, 5, 8], [1, .7, .15, 0]))
    dx, dy = -150 * .55 * up, 90 * .55 * up
    ox, oy = (-22 - 60 * up) * s + dx, (22 + 46 * up) * s + dy
    q = np.array([[92, -20], [86, -62], [44, -96], [-10, -84], [-70, -58], [-130, 0], [-330, 150], [-470, 250], [-420, 340], [-240, 250], [-100, 150], [0, 92], [70, 60]], float)
    q = spl(xf(q, s, tx=px + ox, ty=py + oy), closed=True)
    tx, ty = M9['tail']
    base = np.array([[W + 60, ty + 110], [tx + 150, ty + 104], [tx + 60, ty + 86], [tx - 18, ty + 34], [tx - 34, ty - 44], [tx + 6, ty - 116]], float)
    sw = np.array([[0, 0], [0, -4], [0, -14], [8, -34], [34, -50], [78, -46]], float)
    t = spl(base + sw * tail_flick(f) * 1.3 + [-46, 40], n=10)
    R = ribbon(t, 118, (.02, .45), press=.1, seed=3, tremor=2.)
    c = Cel(ss=1); c.fill(R, '#ffffff'); tl = cv2.GaussianBlur(c.rgba()[..., 3], (0, 0), 4) * .85
    c = Cel(ss=1); c.fill(q, '#ffffff'); pw = cv2.GaussianBlur(c.rgba()[..., 3], (0, 0), 2.6 + 9 * up) * (.9 - .5 * up)
    a = np.maximum(tl, pw) * (1 - .75 * P['st'][0] * .9)
    return img * (1 - a[..., None] * (1 - np.array([.70, .615, .745], f32)))


def paper_frame():
    if 'paperf' not in P:
        g = np.ones((H, W, 3), f32) * rgb('#f0dfba')
        n1 = fbm(H, W, (420, 150, 50), (1, .6, .4), 201)
        g = g + .016 * n1[..., None] * np.array([.8, 1, 1.4], f32)
        fib = tile_to(fibre_tile(), H, W)
        g = g * (1 - .045 * fib[..., None] * np.array([.7, .95, 1.3], f32)) * tooth(H, W, 205, .012)
        P['paperf'] = g.astype(f32)
    return P['paperf']


class Shot:
    id = 's9b'; n = 162
    finish = dict(hal_amt=.22, vig=.10)
    def setup(self): setup()
    def frame(self, f): return frame(f)
