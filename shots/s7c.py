"""s7c aerial: the tower small in a sea of cloud, a ring of wind rolling outward under the camera."""
import numpy as np, math, cv2
from lp import W, H, comp
from lp.paint import f32, rgb, mix, smooth, ramp, fbm, Canvas, arcs, jit_color, grid, mottle
from lp.ink import Cel, spl, over_roi
from shots.balc_sky import *
from shots.s7a_sky import pal, bank, PAL

HZ, TX, TY = 214., 668., 478.           # horizon, tower foot
K1, FX = (TY - HZ), 800.                # y = HZ + K1 / Z,   x = TX + FX * X / Z   (tower at X=0, Z=1)
SUN = (1600., 152.)


def scr(X, Z): return TX + FX * np.asarray(X) / Z, HZ + K1 / np.asarray(Z)


def plane():
    x, y = grid()
    Z = K1 / np.maximum(y - HZ, .5); X = (x - TX) * Z / FX
    return X, Z


def sky_c(r):
    x, y = grid()
    d = np.hypot((x - SUN[0]) / 1250, (y - SUN[1]) / 640)
    yy = np.clip((y + 10 * fbm(H, W, (400, 120), (1, .4), 2)) / HZ, 0, 1)
    img = ramp(yy, [(0, '#74a6c6'), (.4, '#b4c6c8'), (.72, '#eecbb4'), (1, '#f8bd98')])
    g = np.exp(-(d / .62) ** 1.5)[..., None]; img = img + (rgb('#ffc488') - img) * g * .8
    g = np.exp(-(d / .2) ** 1.3)[..., None]; img = img + (rgb('#ffdca2') - img) * g * .85
    img = sky_wash(img, r, 160, hw=(8, 28), centre=SUN, y1=300)
    cv_ = Canvas(img=img)
    for k in range(300):
        t = r.random(); s = int(r.integers(0, 3)); px = [-100, 700, 200][s] + [1000, 1500, 900][s] * t; py = [86, 70, 170][s] - [34, 26, 14][s] * t + r.normal(0, [10, 8, 5][s])
        nr = math.exp(-((px - SUN[0]) / 700) ** 2)
        cv_.strokes(arcs([px], [py], [-.03 + r.normal(0, .05)], [r.uniform(30, 120)], [r.normal(0, .2)]), [r.uniform(1.5, 5.5)], jit_color(mix(rgb('#f0d8c0'), rgb('#fff0c4'), nr)[None], r, .004, .01, .02), [r.uniform(.15, .5) * math.sin(math.pi * t)], [r.uniform(.2, .7)], seed=k, shape='round')
    return cv_.img


def crowns_only(T):
    """keep the opaque crowns of a tree, drop the soft bulges on its face (close up they read as see-through discs)"""
    keep = [i for i, t in enumerate(T.L) if t[7] < .11]; mp = {o: n for n, o in enumerate(keep)}
    T.L = [T.L[i][:4] + (mp.get(T.L[i][4], -1), mp.get(T.L[i][5], -1)) + (T.L[i][6], 0.) for i in keep]


def depth_pal(Z):
    d = float(np.clip((1.9 / Z - .35) / 6., 0, 1)) ** .6
    return d


def sea_c(seed=5):
    """the untouched sea of cloud seen from high up"""
    r = np.random.default_rng(seed)
    img = sky_c(r)
    x, y = grid()
    X, Z = plane()
    P = PAL['B']
    dd = np.clip((np.clip(y - HZ, 0, 900) / 884.), 0, 1) ** .7
    base = ramp(dd, [(0, '#fbdcae'), (.12, '#f2cbac'), (.35, '#d9b0ae'), (.7, '#b797b1'), (1, '#9f86ab')])
    a = smooth(HZ - 5, HZ + 7, y + 2.5 * fbm(H, W, (40,), None, 3))
    img = img * (1 - a[..., None]) + base * a[..., None]
    # far standing heads on the horizon
    for (x0, x1, yc, R, s) in ((-40, 420, HZ - 8, 18, 1), (380, 900, HZ - 2, 10, 2), (1000, 1440, HZ - 1, 8, 3), (1760, 2100, HZ - 8, 20, 4)):
        m = Mass(bank(x0, x1, yc, R, r, .8, .8, (5,), .7, big=.35), sun=(SUN[0], 60), seed=s, lz=-.2, rimw=2, floor=(HZ + 2, 8))
        sh, li, rc, _ = pal('B', .10); c = colour(m, sh, li, rc, sun_lo=.05, sun_hi=.75, rim=.7, seed=s); img = lay(img, m, c)
    # banks: each lies on a depth row with its own gaps; big soft valleys wind between them
    zs = []
    z = 4.2
    while z > .235:
        zs.append(z); z *= .878
    for k, z in enumerate(zs):
        yc = HZ + K1 / z; R = 54. / z * r.uniform(.8, 1.1)
        xs = []
        xx = -200 + r.uniform(0, 300)
        while xx < W + 150:
            L = r.uniform(3.2, 9.0) * R
            xs.append((xx, xx + L)); xx += L + r.uniform(.2, 3.0) * R
        d = depth_pal(z)
        sh, li, rc, mi = pal('B', min(1, .08 + .85 * d))
        for j, (a0, b0) in enumerate(xs):
            T = bank(a0, b0, yc + r.normal(0, R * .14), R, r, .55, .5 + .06 * d, (5,), .8, big=.3)
            if z < .8: crowns_only(T)
            m = Mass(T, sun=(SUN[0], HZ - 420), seed=300 + k * 13 + j, lz=-.14, rimw=1.5 + 3.5 * d, floor=(yc + R * .6, R * .9))
            c = np.ascontiguousarray(colour(m, sh, li, rc, sun_lo=.16, sun_hi=.86, rim=.8, seed=300 + k * 13 + j, up=(-.6, .7)), f32)
            c = marks(m, c, np.random.default_rng(k * 50 + j), int(60 + 90 * d), size=(.04, .11) if z > .8 else (.03, .07), op=(.2, .55) if z > .8 else (.10, .28), val=.012 if z > .8 else .008)
            img = lay(img, m, c)
        n = fbm(H, W, (170, 60), (1, .5), 60 + k, aspect=3.2)
        t = (y - (yc + R * .2) + n * R * .22) / (R * 1.15); am = np.clip(np.minimum(t / .5, (1 - t) / .5), 0, 1); am = am * am * (3 - 2 * am) * .6
        img = img + (mi - img) * am[..., None]
    # one hand-placed clump so that the big near crown reads as cloud and nothing else
    rr = np.random.default_rng(8); T = Tree()
    for (bx, by, br) in ((330, 1004, 60), (540, 1012, 58), (420, 1020, 84), (500, 1040, 78), (352, 1050, 88), (592, 1048, 60), (446, 1066, 96), (560, 1078, 74)): T.grow(bx, by, br * 1.15, br * .62, rr, kids=(5,), shrink=(.55,), face=(0.,), rmin=4., asp=(.7, .92))
    m = Mass(T, sun=(SUN[0], HZ - 420), seed=77, lz=-.14, rimw=4.5, floor=(1085, 60)); sh, li, rc, mi = pal('B', .92)
    c = np.ascontiguousarray(colour(m, sh, li, rc, sun_lo=.16, sun_hi=.86, rim=.8, seed=77, up=(-.6, .7)), f32)
    return lay(img, m, marks(m, c, rr, 120, size=(.03, .07), op=(.1, .28), val=.008))


def flat_bank(x0, x1, yc, R, r, asp=.17):
    """low, long, combed-flat lobes"""
    T = Tree(); L = []; x = x0; ph = r.uniform(0, 9)
    while x < x1:
        u = (x - x0) / (x1 - x0 + 1e-6); env = min(1., min(u, 1 - u) * 4 + .2); rr = R * r.uniform(.6, 1.25) * (.5 + .5 * env)
        L.append((yc + R * .12 * math.sin(x / (R * 3.3) + ph) + r.normal(0, R * .05), x, rr)); x += R * r.uniform(.55, 1.3)
    L.sort()
    for y, x, rr in L: T.grow(x, y, rr * 1.25, rr * asp, r, kids=(3,), shrink=(.62,), face=(.5,), spread=.8, rmin=3., soft=.10, asp=(.16, .26))
    return T


def silk(seed=9):
    """the sea after the ring has gone by: the same cloud, laid smooth and pale"""
    r = np.random.default_rng(seed)
    x, y = grid()
    dd = np.clip((np.clip(y - HZ, 0, 900) / 884.), 0, 1) ** .7
    img = ramp(dd, [(0, '#f9dfba'), (.14, '#f4d6bd'), (.36, '#e8c6b6'), (.68, '#d6b2b6'), (1, '#c6a5b7')]).astype(f32)
    sh0, sh1 = ['#d6b0ab', '#e2bfb0', '#edcfb8', '#f5dcc0'], ['#a98bac', '#b999b2', '#c8a7b5', '#d8b6b7']
    li0, li1 = ['#f7e0bf', '#fbe7c8', '#fff0d6'], ['#e2c1b7', '#eed1bd', '#f8e2ca']
    z, k = 4.4, 0
    while z > .21:
        yc = HZ + K1 / z; R = 78. / z; d = float(depth_pal(z)); xs = []; xx = -260 + r.uniform(0, 300)
        while xx < W + 200:
            L = r.uniform(3.5, 8.) * R; xs.append((xx, xx + L)); xx += L + r.uniform(.4, 2.6) * R
        shd = [mix(rgb(p), rgb(q), d) for p, q in zip(sh0, sh1)]; lig = [mix(rgb(p), rgb(q), d) for p, q in zip(li0, li1)]
        for j, (a0, b0) in enumerate(xs):
            m = Mass(flat_bank(a0, b0, yc, R, r), sun=(SUN[0], HZ - 420), seed=900 + k * 13 + j, lz=.05, rimw=1.5 + 3 * d, fade=1.0, halo=.05, bill=(.10, .03), cell=.7, floor=(yc + R * .10, R * .26))
            c = colour(m, shd, lig, '#fff3da', sun_lo=.05, sun_hi=.95, rim=.35, seed=900 + k * 13 + j, up=(-.7, .8), noise=.10, crisp=.5)
            img = lay(img, m, c, op=.92)
        z *= .80; k += 1
    img = np.ascontiguousarray(img, f32)
    # silk: softened along the level, more toward the camera
    so = cv2.GaussianBlur(img, (0, 0), sigmaX=9, sigmaY=2.6); w = smooth(HZ + 40, HZ + 520, y)[..., None] * .40
    img = img + (so - img) * w
    n = 900; py = HZ + 30 + (H - HZ) * r.random(n) ** 1.5; px = r.uniform(-80, W + 80, n); s_ = (py - HZ) / 900.
    ix, iy = np.clip(px, 0, W - 1).astype(int), np.clip(py, 0, H - 1).astype(int)
    cv_ = Canvas(img=img)
    cv_.strokes(arcs(px, py, r.normal(0, .03, n), 60 + 420 * s_ * r.uniform(.4, 1.2, n), r.normal(0, .08, n)), 1.5 + 13 * s_ * r.uniform(.4, 1.1, n), jit_color(so[iy, ix], r, .004, .012, .016), r.uniform(.06, .22, n), r.uniform(.1, .6, n), seed=4, shape='round', bristle=.03)
    img = cv_.img
    # a wash of sun from the right
    g = np.exp(-(np.hypot((x - 1750) / 1250, (y - 330) / 560)) ** 1.6) * .42
    return img + (rgb('#ffe9c4') - img) * g[..., None]


def trough():
    """deep rose-violet hollow, with torn strands lying along the ring"""
    x, y = grid(); X, Z = plane()
    dd = np.clip((np.clip(y - HZ, 0, 900) / 884.), 0, 1) ** .7
    g = ramp(dd, [(0, '#dcae9f'), (.14, '#c597a0'), (.36, '#a9809d'), (.68, '#906f99'), (1, '#80648f')]).astype(f32)
    th = np.arctan2(X, 1 - Z); rr = np.hypot(X, Z - 1); r = np.random.default_rng(31); q = np.zeros((H, W), f32)
    for nt, kr, w in ((14, 34., .55), (30, 80., .3), (60, 170., .15)):                     # strands: long round the ring, short across it
        gx = r.standard_normal((nt, 400)).astype(f32); u = ((th + np.pi) / (2 * np.pi) * nt); v = np.clip(rr * kr + .6 * np.sin(th * 4 + kr), 0, 398.9)
        u0 = np.floor(u).astype(int) % nt; v0 = np.floor(v).astype(int); fu = u - np.floor(u); fv = v - v0; fu = fu * fu * (3 - 2 * fu); fv = fv * fv * (3 - 2 * fv)
        q += w * ((gx[u0, v0] * (1 - fu) + gx[(u0 + 1) % nt, v0] * fu) * (1 - fv) + (gx[u0, v0 + 1] * (1 - fu) + gx[(u0 + 1) % nt, v0 + 1] * fu) * fv)
    hi = ramp(dd, [(0, '#f1cfb4'), (.2, '#e0b7ae'), (.6, '#c7a0b0'), (1, '#b592ae')]); k = smooth(.15, 1.0, q)[..., None] * .55
    return (g + (hi - g) * k) * (1 - smooth(.2, 1.1, -q)[..., None] * .07)


def tower_cel():
    """tiny tower, drawn twice (lit / shade) at its true size, with its own little ragged feet of mist"""
    c = Cel(150, 300, int(TX - 75), int(TY - 232), ss=3)
    X0, Y0 = TX, TY + 18
    sh, hf, lt = '#8d7690', '#b99a94', '#efd2a2'
    body = spl([(X0 - 23, Y0 + 8), (X0 - 20, Y0 - 60), (X0 - 18.5, Y0 - 122), (X0 + 17.5, Y0 - 122), (X0 + 19.5, Y0 - 60), (X0 + 23, Y0 + 8)], closed=True, corners=(0, 2, 3, 5))
    c.fill(body, sh)
    c.shade([(X0 + 1, Y0 + 10), (X0 + 2.5, Y0 - 125), (X0 + 40, Y0 - 125), (X0 + 40, Y0 + 10)], hf, within=body)
    c.shade([(X0 + 9, Y0 + 10), (X0 + 9.5, Y0 - 60), (X0 + 8.5, Y0 - 125), (X0 + 40, Y0 - 125), (X0 + 40, Y0 + 10)], lt, within=body)
    for (yy, w, h) in ((-124, 23, 6), (-92, 21, 3), (-52, 22, 3)):
        b = [(X0 - w, Y0 + yy), (X0 + w, Y0 + yy), (X0 + w - 1, Y0 + yy + h), (X0 - w + 1, Y0 + yy + h)]; c.fill(b, '#9c8497'); c.shade([(X0 + 5, Y0 + yy - 2), (X0 + w + 3, Y0 + yy - 2), (X0 + w + 3, Y0 + yy + h + 2), (X0 + 5, Y0 + yy + h + 2)], '#f4dcae', within=b)
    # balcony on the sun side
    b = [(X0 + 17, Y0 - 78), (X0 + 33, Y0 - 78), (X0 + 32, Y0 - 73), (X0 + 24, Y0 - 68), (X0 + 18, Y0 - 62)]; c.fill(b, '#a88c98'); c.fill([(X0 + 17, Y0 - 78.5), (X0 + 33.5, Y0 - 78.5), (X0 + 33.5, Y0 - 76.5), (X0 + 17, Y0 - 76.5)], '#fbe3b4')
    # windows and drum
    r = np.random.default_rng(2)
    for (dx, yy) in ((-12, -108), (-4, -107), (5, -108), (12, -108), (-9, -74), (2, -73), (-13, -40), (4, -36), (-6, -20)): c.fill([(X0 + dx - 1.6, Y0 + yy + 5), (X0 + dx - 1.6, Y0 + yy), (X0 + dx, Y0 + yy - 2), (X0 + dx + 1.6, Y0 + yy), (X0 + dx + 1.6, Y0 + yy + 5)], '#5c4c74' if dx < 4 else '#9a7c86')
    for k in range(44):
        yy = r.uniform(-120, 0); xx = r.uniform(-20, 20); c.line([(X0 + xx, Y0 + yy), (X0 + xx + r.uniform(2, 6), Y0 + yy + .3)], .6, '#75638a' if xx < 4 else '#b8958e', seed=k, alpha=.7)
    drum = [(X0 - 16, Y0 - 124), (X0 - 15.5, Y0 - 138), (X0 + 15.5, Y0 - 138), (X0 + 16, Y0 - 124)]; c.fill(drum, '#8a7492'); c.shade([(X0 + 4, Y0 - 140), (X0 + 20, Y0 - 140), (X0 + 20, Y0 - 120), (X0 + 4, Y0 - 120)], '#ecd0a2', within=drum)
    for dx in (-11, -5.5, 0, 5.5, 11): c.fill([(X0 + dx - 1.2, Y0 - 127), (X0 + dx - 1.2, Y0 - 134), (X0 + dx + 1.2, Y0 - 134), (X0 + dx + 1.2, Y0 - 127)], '#5c4c74' if dx < 4 else '#a38389')
    dome = spl([(X0 - 18, Y0 - 138), (X0 - 17, Y0 - 150), (X0 - 10, Y0 - 161), (X0, Y0 - 166), (X0 + 10, Y0 - 161), (X0 + 17, Y0 - 150), (X0 + 18, Y0 - 138)], closed=True, corners=(0, 6))
    c.fill(dome, '#5f8791'); c.shade(spl([(X0 - 2, Y0 - 136), (X0 + 1, Y0 - 152), (X0 - 1, Y0 - 168), (X0 + 24, Y0 - 168), (X0 + 24, Y0 - 136)], closed=True, corners=(0, 2, 3, 4)), '#8fb4a4', within=dome); c.shade(spl([(X0 + 8, Y0 - 136), (X0 + 9, Y0 - 150), (X0 + 3, Y0 - 163), (X0 + 24, Y0 - 168), (X0 + 24, Y0 - 136)], closed=True, corners=(0, 2, 3, 4)), '#d6d9ae', within=dome)
    for u in (-.75, -.4, 0, .4, .75): c.line(spl([(X0 + 18 * u, Y0 - 138), (X0 + 16.5 * u, Y0 - 151), (X0 + 8 * u, Y0 - 161), (X0, Y0 - 166)]), .8, '#4b7080' if u < .3 else '#9cae8e', seed=int(u * 10) + 20, alpha=.8)
    lan = [(X0 - 3.4, Y0 - 165), (X0 - 3, Y0 - 173), (X0, Y0 - 178), (X0 + 3, Y0 - 173), (X0 + 3.4, Y0 - 165)]; c.fill(lan, '#b7a08f'); c.fill([(X0 + .5, Y0 - 165), (X0 + .5, Y0 - 177), (X0 + 3, Y0 - 173), (X0 + 3.4, Y0 - 165)], '#ffe9b8')
    c.line([(X0, Y0 - 178), (X0, Y0 - 190)], 1.1, '#6d5a70'); 
    ol = np.vstack([body[:1], dome, body[-1:]])
    c.line(spl([(X0 - 23, Y0 + 2), (X0 - 20, Y0 - 60), (X0 - 18.5, Y0 - 122)]), 1.1, '#54466c', seed=1); c.line(dome, 1.1, '#3f5868', seed=3); c.line(spl([(X0 + 18.5, Y0 - 122), (X0 + 20.5, Y0 - 60), (X0 + 23.5, Y0 + 2)]), 1.6, '#fff0c8', seed=2)
    return c.rgba(), c.ox, c.oy


def tower_sprite(scale=.315):
    """the sky lane's library tower (s1), seen from its other side in morning light: same painter, same stones.
    Falls back to the little cel above if that module ever moves."""
    from shots import s1_tower as tw, s1_cliff as cl
    from lp.paint import lum
    L = [(cl.cliff(), 'rock')]
    t = tw.paint_tower(); L.append((tw.gouache(t), 'stone'))
    for l in (tw.ivy_layer(t), tw.balcony_ivy(), cl.dressing().layer(), cl.pines().layer()): L.append((l, 'leaf'))
    RP = {'stone': [(0, '#5f4f78'), (.16, '#78668c'), (.30, '#937a94'), (.42, '#b3938f'), (.54, '#dcb996'), (.70, '#f6dfb2'), (.88, '#fff3d2')],
          'rock': [(0, '#5b4d76'), (.16, '#74638a'), (.30, '#8f7893'), (.45, '#ae9092'), (.60, '#d6b596'), (.80, '#f3dcb0')],
          'leaf': [(0, '#4b5460'), (.14, '#5a6255'), (.28, '#757750'), (.45, '#a8a158'), (.65, '#dccb7a'), (.85, '#f5e7a6')],
          'dome': [(0, '#476873'), (.20, '#5d858a'), (.34, '#7fa698'), (.48, '#a9bf9d'), (.64, '#d8d9aa'), (.85, '#f6eec2')]}
    out = np.zeros((H, W, 4), f32)
    yy = np.arange(H)[:, None]
    for l, k in L:
        l = l.astype(f32); a = l[..., 3]; c = l[..., :3] / np.maximum(a, 1e-3)[..., None]; v = np.clip(lum(c), 0, 1)
        n = ramp(v, RP[k])
        if k == 'stone':
            d = smooth(219, 211, yy) * np.ones((1, W), f32)
            n = n + (ramp(v, RP['dome']) - n) * d[..., None]
        out = out * (1 - a[..., None]) + np.dstack([n * a[..., None], a])
    out = out[40:912, 930:1760][:, ::-1]
    sp = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hh = sp.shape[0]; sp = sp * smooth(0, 46, hh - 1 - np.arange(hh, dtype=f32) + 5 * np.sin(np.arange(hh) * 0))[:, None, None]
    ax = (1760 - 1262) * scale                                                # the tower's axis in the sprite
    return sp.astype(f32), float(ax)


class Ring:
    """the crest: a rolling wall of cloud on a ragged circle, redrawn every second frame"""
    def __init__(self, seed=14):
        r = np.random.default_rng(seed); n = 520
        self.th = np.sort(r.uniform(0, 2 * np.pi, n)); self.sz = r.uniform(.45, 1.15, n) * np.where(r.random(n) < .14, 1.5, 1.); self.o = r.normal(0, 1, n); self.h = r.uniform(.2, 1., n)
        self.ph = r.uniform(0, 6.28, (8,))
        self.row = r.integers(0, 3, n)
        self.curl = [(r.uniform(0, 2 * np.pi), int(r.integers(-8, 24)), int(r.integers(5, 9)), r.uniform(.6, 1.25), r.uniform(-.5, .5), r.uniform(0, 1)) for _ in range(150)]

    def rag(self, th, R):
        p = self.ph
        big = .5 + .5 * np.sin(2 * th + p[4])                                    # some stretches of the wall stand taller
        return 1 + .030 * np.sin(3 * th + p[0]) + .024 * np.sin(7 * th + p[1] + R * 3) + .016 * np.sin(13 * th + p[2] - R * 5) + .012 * np.sin(29 * th + p[3] + R * 9) + .008 * np.sin(53 * th + p[5] - R * 14)

    def tall(self, th):
        """the wall's height round the ring: rears up toward the camera, thins to a line behind the tower"""
        p = self.ph; n = .5 + .5 * np.cos(th)
        return (.36 + .70 * n ** 1.25) * (.86 + .22 * np.sin(2 * th + p[4]) + .16 * np.sin(5 * th + p[6]))

    def whole(self, th):
        """1 where the crest stands, 0 where it has torn open"""
        p = self.ph; v = .42 + .34 * np.sin(4 * th + p[0] + 1) + .30 * np.sin(7 * th + p[3]) + .22 * np.sin(11 * th + p[5])
        return smooth(-.02, .34, v)

    def at(self, f):
        t = f / 26.
        return .22 + .43 * (t ** .95)

    def pts(self, th, R): return R * np.sin(th), 1 - R * np.cos(th)          # th=0 nearest the camera

    def drawing(self, f):
        """every lobe is painted again, opaque, at its new place; its shape is its own, only the hand wobbles"""
        d = f // 2; R = self.at(d * 2)
        r = np.random.default_rng(700 + d)
        th = self.th; rad = R * self.rag(th, R) * (1 + np.array([-.030, 0., .022])[self.row] * .30 / max(R, .2) + .012 * self.o)
        X, Z = self.pts(th, rad); ok = Z > .13
        x, y = scr(X, np.maximum(Z, .13)); grow = min(1., .55 + R)
        s = 50. / np.maximum(Z, .13) * self.sz * grow * self.tall(th) * np.array([.85, 1.1, .7])[self.row]
        ok &= (x > -300) & (x < W + 300) & (y < H + 300)
        jit = r.normal(0, .016, (len(th), 2))
        T = Tree(); ids = []
        for i in np.argsort(y + np.array([-1., 0, 1.])[self.row]):
            if not ok[i]: continue
            wh = float(self.whole(np.array(th[i])))
            if wh < .25 and self.row[i] == 1 and i % 3: continue
            q = np.random.default_rng(i * 17 + 3); si = s[i] * (.30 + .70 * wh); up = [.34, .62, .10][self.row[i]] * self.h[i]
            cx, cy = x[i] + jit[i, 0] * si, y[i] - si * up + jit[i, 1] * si
            n0 = len(T.L)
            if wh > .4: T.add(cx, cy + si * .34, si * 1.45, si * .70)              # solid body under the crowns: nothing shows through the wall
            T.grow(cx, cy, si * 1.05, si * .76, q, kids=(5,), shrink=(.55,), face=(.10,), lean=.35 * float(np.sin(th[i])), rmin=3., asp=(.72, .95))
            ids += [int(i) * 12 + k for k in range(len(T.L) - n0)]
        T.L = [t[:7] + (0.,) for t in T.L]
        return T, R, np.array(ids)

    def tufts(self, f, R):
        """torn rags of mist thrown up off the lip"""
        d = f // 2; r = np.random.default_rng(55); n = 210
        th0 = r.uniform(0, 2 * np.pi, n); per = r.integers(4, 8, n); ph = r.integers(0, 8, n); hh = r.uniform(.4, 1.2, n); sz = r.uniform(.08, .30, n)
        T = Tree(); rows = []
        for i in range(n):
            t = ((d + ph[i]) % per[i]) / per[i]; q = np.random.default_rng(i * 31 + (d + ph[i]) // per[i])
            th = th0[i] + q.normal(0, .06); rad = R * float(self.rag(np.array(th), R)) * (1 + .03 * t + q.normal(0, .01)); X, Z = self.pts(th, rad)
            if Z < .52: continue
            x, y = scr(X, Z); s = 46. / Z * min(1., .5 + R) * float(self.tall(np.array(th))) * (1.25 - .45 * float(self.whole(np.array(th))))
            rows.append((y, x + s * q.normal(0, .3) + s * .5 * t * math.sin(th), y - s * (.75 + hh[i] * 1.1 * t ** .7), s * sz[i] * (1 - .55 * t), q))
        rows.sort(key=lambda v: v[0])
        for _, x, y, si, q in rows: T.grow(x, y, si * 1.5, si * .8, q, kids=(4,), shrink=(.6,), face=(.1,), rmin=2., asp=(.6, .9))
        T.L = [t[:7] + (0.,) for t in T.L]
        return T


def crest_layer(ring, f):
    T, R, ids = ring.drawing(f)
    m = Mass(T, sun=(SUN[0], HZ - 260), seed=90, lz=-.14, rimw=5, fade=.62, halo=.18, cell=.46, bill=(.24, .10), ids=ids)
    x, y = grid()
    sh0, li0, rc, _ = pal('B', .25); sh1, li1, _, _ = pal('B', .88)
    dn = np.clip((np.clip(y - HZ, 0, 900) / 884.), 0, 1)[..., None] ** .7
    kw = dict(sun_lo=.08, sun_hi=.74, rim=1.0, seed=5, crisp=.42, noise=.10)
    c = colour(m, sh0, li0, rc, **kw); c = np.ascontiguousarray(c + (colour(m, sh1, li1, rc, **kw) - c) * dn, f32)
    c = marks(m, c, np.random.default_rng(40), 700, size=(.03, .08), op=(.15, .40), val=.010)
    # on the camera's side the wall's feet are lost in mist, painted as one soft edge along the ring
    X, Z = plane(); th = np.arctan2(X, 1 - Z); s = (np.hypot(X, Z - 1) - R * ring.rag(th, R)) / (R * ring.rag(th, R))
    n = fbm(H, W, (120, 44), (1, .5), 11, aspect=2.0)
    near = smooth(-.05, .45, np.cos(th)) * (y > TY)
    a = m.A * (1 - near * smooth(.02, .15, s + .022 * n))
    out = np.dstack([c * a[..., None], a]).astype(f32)
    tf = Mass(ring.tufts(f, R), sun=(SUN[0], HZ - 260), seed=191, lz=-.05, rimw=3, fade=.7, halo=0., cell=.6)
    tf.U = tf.U * .5 + .45; tf.S = tf.S * .7 + .25
    ct = colour(tf, sh0, li0, rc, sun_lo=.0, sun_hi=.8, rim=.9, seed=6); at = tf.A * .94
    out[..., :3] = out[..., :3] * (1 - at[..., None]) + ct * at[..., None]; out[..., 3] = out[..., 3] * (1 - at) + at
    return out


ARCS = [(-1.25, 0, 12, .50, .0), (1.05, 2, 13, .46, .25), (1.65, 5, 12, .50, .1), (-1.85, 8, 12, .46, .3), (2.25, 10, 13, .50, .15), (1.20, 14, 12, .50, .0), (-1.40, 15, 12, .50, .2)]


def arcs_of_mist(ring, f, R):
    """a few broad dry-brush arcs that lie along the ring (never more than six), drawn again on every second frame"""
    d = f // 2; cv_ = Canvas(layer=True); cv_.img[:] = rgb('#fff0d6'); K = 18; n_ = 0
    for i, (th, f0, lf, span, off) in enumerate(ARCS):
        t = (d * 2 - f0) / float(lf)
        if t < 0 or t >= 1 or n_ >= 6: continue
        n_ += 1; r = np.random.default_rng(i * 40 + d); op = math.sin(math.pi * np.clip(t, .08, .94)) ** .8
        for j in range(5):
            q = np.linspace(0, 1, K); tt = th + .16 * t + span * (.75 + .35 * t) * (q - .5) * r.uniform(.8, 1.05) + r.normal(0, .02)
            rad = R * ring.rag(tt * .6 + th * .4, R) * (1 - .045 + .024 * (j - 2) / 2 + off * .03)
            X, Z = ring.pts(tt, rad)
            if Z.min() < .50: continue
            x, y = scr(X, Z); s = 50. / Z * min(1., .5 + R) * (.4 + .6 * (.5 + .5 * np.cos(tt)))
            y = y - s * (.62 + .34 * np.sin(q * math.pi) + .12 * j)
            P = np.c_[x, y]; P[1:-1] += r.normal(0, .6, (K - 2, 2)); sm = float(s.mean())
            cv_.strokes(P[None], np.clip(sm * r.uniform(.07, .15), 1.5, 26), rgb(['#fff4de', '#fbe3c4', '#fff0d4', '#f2cfb8', '#fff6e4'][j]), op * r.uniform(.34, .58), r.uniform(.58, .88), seed=i * 9 + j + d * 3, shape='round', bristle=.06)
    L = cv_.layer()
    w = smooth(HZ + 100, HZ + 620, np.arange(H)[:, None, None].astype(f32))
    return cv2.GaussianBlur(L, (0, 0), .9) * (1 - w) + cv2.GaussianBlur(L, (0, 0), 2.1) * w
