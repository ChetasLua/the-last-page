"""s7a sky, sun path and sea of clouds: two lighting states (A before the sun clears, B sunlit) + arrival map."""
import numpy as np, math, cv2
from lp import W, H
from lp.paint import f32, rgb, mix, smooth, ramp, fbm, Canvas, arcs, jit_color, grid, mottle
from shots.balc_sky import *

SUN0 = np.array([1310., 948.]); SUN1 = np.array([1310., 786.]); SUNR = 70
HZ = 884


def flecks(img, r, state):
    """high mackerel cloud: long ragged sheets of flat-brush flecks, lit from below"""
    x, y = grid()
    cols = {'A': [('#b9a9c4', '#d7b7bd', '#eec4b0'), ('#9c9cc2', '#c3adc2', '#e2b9b6')],
            'B': [('#d9cfc8', '#f2d9b8', '#ffe9bd'), ('#b9c6cc', '#e6d3c0', '#fbe0b4')]}[state]
    cv_ = Canvas(img=img)
    # sheets: centre line from (x0,y0) to (x1,y1), half thickness, fleck length, count
    sheets = [((-60, 250), (900, 118), 46, 90, 300, 0), ((520, 330), (1500, 250), 34, 70, 260, 0), ((1180, 120), (2150, 40), 40, 100, 220, 1),
              ((60, 470), (760, 415), 22, 60, 170, 0), ((1420, 360), (1800, 330), 16, 46, 80, 0), ((760, 560), (1260, 538), 14, 46, 110, 0), ((-80, 590), (420, 566), 12, 50, 90, 1)]
    for (a, b, th, ln, n, pk) in sheets:
        a = np.array(a, float); b = np.array(b, float); d = b - a; L = np.hypot(*d); e = d / L; nn = np.array([-e[1], e[0]])
        t = r.beta(1.25, 1.25, n); v = r.normal(0, .55, n) + .35 * np.sin(t * 9 + th); env = np.sin(np.pi * t) ** .6
        p = a + np.outer(t, d) + nn * (v * th * env)[:, None] + np.c_[np.zeros(n), 18 * np.sin(t * 3.1)]
        ang = math.atan2(e[1], e[0]) + r.normal(0, .10, n)
        near = np.exp(-((p[:, 0] - SUN1[0]) / 760) ** 2 - ((p[:, 1] - SUN1[1]) / 560) ** 2)
        c3 = np.stack([rgb(c) for c in cols[pk]])
        u = np.clip(.35 + .5 * near + .28 * v + r.normal(0, .12, n), 0, 1) * 2
        i = np.clip(u.astype(int), 0, 1); c = c3[i] * (1 - (u - i))[:, None] + c3[i + 1] * (u - i)[:, None]
        big = r.random(n) < .18
        l = ln * r.uniform(.25, .8, n) * np.where(big, 2.2, 1); hw = th * r.uniform(.05, .13, n) * np.where(big, 1.5, 1)
        m_ = 46; tt = r.beta(1.4, 1.4, m_); en = np.sin(np.pi * tt) ** .7
        pb = a + np.outer(tt, d) + nn * (r.normal(0, .5, m_) * th * en)[:, None] + np.c_[np.zeros(m_), 18 * np.sin(tt * 3.1)]
        cv_.strokes(arcs(pb[:, 0], pb[:, 1], math.atan2(e[1], e[0]) + r.normal(0, .06, m_), ln * r.uniform(1.2, 2.8, m_), r.normal(0, .2, m_)), th * r.uniform(.16, .42, m_), jit_color(np.repeat(c3[1][None], m_, 0), r, .004, .01, .012), en * r.uniform(.06, .17, m_), r.uniform(.1, .5, m_), seed=int(r.integers(1 << 20)), shape='round', bristle=.03)
        cv_.strokes(arcs(p[:, 0], p[:, 1], ang, l, r.normal(0, .25, n)), hw, jit_color(c, r, .004, .008, .012), env * r.uniform(.14, .5, n), r.uniform(.15, .75, n), seed=int(r.integers(1 << 20)), shape='round', bristle=.03)
    return cv_.img


def sky(state, seed=4):
    r = np.random.default_rng(seed)
    x, y = grid()
    yy = np.clip((y + 26 * fbm(H, W, (420, 160), (1, .4), 8)) / HZ, 0, 1)
    if state == 'B':
        img = ramp(yy, [(0, '#5a94c2'), (.22, '#7fafca'), (.44, '#b2cccb'), (.64, '#ecd1bc'), (.82, '#f7c6a4'), (1, '#f6b690')])
        for sx, sy, c, k in ((900, 480, '#fdcb98', .60), (430, 240, '#ffd092', .55)):
            g = np.exp(-(((x - SUN1[0]) / sx) ** 2 + ((y - SUN1[1]) / sy) ** 2))[..., None]; img = img + (rgb(c) - img) * g * k
    else:
        img = ramp(yy, [(0, '#41619b'), (.22, '#5d7fb0'), (.44, '#8f9fc2'), (.62, '#c3b0c0'), (.8, '#e8b9ab'), (1, '#f2b592')])
        for sx, sy, c, k in ((820, 330, '#f9cfa0', .55), (300, 130, '#ffe0a6', .6)):
            g = np.exp(-(((x - SUN1[0]) / sx) ** 2 + ((y - HZ) / sy) ** 2))[..., None]; img = img + (rgb(c) - img) * g * k
    img = sky_wash(img, r, 360, centre=SUN1)
    img = flecks(img, r, state)
    return np.clip(mottle(img, .006, 6, .003, (90, 30, 8)), 0, 1).astype(f32)


PAL = {   # depth 0 far .. 1 near
    'B': dict(shf=['#dfb9ac', '#e9c6ae', '#f2d3b2', '#f8dfb9'], shn=['#8a7ba6', '#a28cab', '#bd9fae', '#d6b3b0'], lif=['#f8d9ac', '#fee8c0', '#fff4d8'], lin=['#f0c29e', '#fbd9ac', '#fff0cf'], rim='#fff3cf', mf='#fbdcae', mn='#b99db2'),
    'A': dict(shf=['#b99cae', '#c9a8b0', '#dab4af', '#e6c0b0'], shn=['#5d5c94', '#736b9d', '#8d7da6', '#a78fae'], lif=['#e2b9ad', '#efc6ac', '#f8d4ae'], lin=['#a58dad', '#bd9fb0', '#d9b3b0'], rim='#ffd9a6', mf='#e8bfac', mn='#8a7fa8')}


def pal(state, d):
    p = PAL[state]; T = lambda a, b: [mix(rgb(i), rgb(j), d) for i, j in zip(a, b)]
    return T(p['shf'], p['shn']), T(p['lif'], p['lin']), rgb(p['rim']), mix(rgb(p['mf']), rgb(p['mn']), d)


class Sheet:
    """accumulates premultiplied cloud layers for both lighting states"""
    def __init__(self): self.c = {s: np.zeros((H, W, 3), f32) for s in 'AB'}; self.a = np.zeros((H, W), f32); self.T = np.full((H, W), 30., f32)

    def put(self, m, d, seed, rows=None, nm=200, lo=0, sl=(.1, .8), rim=.8, crisp=.16, op=1.0, mk=(.04, .12), Tb=0., noise=.16, up=(-.75, .75)):
        a = m.A * op
        for s in 'AB':
            sh, li, rc, _ = pal(s, d)
            c = colour(m, sh, li, rc, sun_lo=sl[0] + (.10 if s == 'A' else 0), sun_hi=sl[1] + (.2 if s == 'A' else 0), rim=rim * (.7 if s == 'A' else 1), seed=seed, crisp=crisp, noise=noise, up=up)
            c = np.ascontiguousarray(c, f32)
            rr = np.random.default_rng(seed * 7 + 1)
            c = marks(m, c, rr, nm, size=mk, lo=lo, op=(.2, .6), val=.012)
            self.c[s] = self.c[s] * (1 - a[..., None]) + c * a[..., None]
        crown = np.clip((m.S - sl[0]) / (sl[1] - sl[0]), 0, 1) * .7 + np.clip(m.R, 0, 1) * .5
        x, y = grid()
        T = 4 + np.hypot((x - SUN1[0]) / 1000, (y - HZ) / 620) * 26 + 18 * (1 - crown) + Tb + 4 * fbm(H, W, (140, 40), (1, .5), seed + 2)
        self.T += (T - self.T) * a
        self.a = self.a * (1 - a) + a

    def mist(self, y0, y1, d, op, seed=0, xr=None, peak=.5):
        """pooled mist band"""
        x, y = grid()
        n = fbm(H, W, (160, 60), (1, .5), seed, aspect=3.)
        t = (y - y0 + n * (y1 - y0) * .16) / (y1 - y0)
        a = np.clip(np.minimum(t / peak, (1 - t) / (1 - peak)), 0, 1); a = a * a * (3 - 2 * a) * op
        if xr is not None: a = a * smooth(xr[0] - 160, xr[0] + 160, x + n * 60) * (1 - smooth(xr[1] - 160, xr[1] + 160, x + n * 60))
        for s in 'AB':
            self.c[s] += (pal(s, d)[3] * a[..., None] - self.c[s] * a[..., None])
        T = 8 + np.hypot((x - SUN1[0]) / 1000, (y - HZ) / 620) * 26 + 8
        self.T += (T - self.T) * a
        self.a = self.a * (1 - a) + a


def bank(x0, x1, yc, R, r, amp=.5, asp=.6, kids=(5,), dens=.8, lean=0., big=.25):
    T = Tree(); xs = []
    x = x0
    while x < x1:
        xs.append(x); x += R * dens * r.uniform(.55, 1.5)
    ph = r.uniform(0, 9)
    L = []
    for x in xs:
        u = (x - x0) / (x1 - x0 + 1e-6); env = min(1., min(u, 1 - u) * 5 + .25)
        rr = R * r.uniform(.5, 1.0) * (1.55 if r.random() < big else 1) * (.55 + .45 * env)
        yy = yc + amp * R * (math.sin(x / (R * 4.1) + ph) * .6 + math.sin(x / (R * 1.7) + 2 * ph) * .25) + (1 - env) * R * .4 + r.normal(0, R * .08)
        L.append((yy + rr * asp * .8, x, yy + rr * asp * .75, rr))
    L.sort()
    for _, x, y, rr in L: T.grow(x, y, rr, rr * asp, r, kids=kids, shrink=(.55, .5), face=(.2, .15), lean=lean, rmin=2.2, asp=(.62, .9), spread=.95)
    return T


def clouds():
    r = np.random.default_rng(21)
    S = Sheet()
    def tower(bl, seed, d, vs, lean=0., kids=(6,), floor=(845, 90), lz=-.12, nm=700, rim=.8, sl=(.1, .82), rimw=8, Tb=0., rr=None, mk=(.05, .14)):
        rr = np.random.default_rng(seed)
        T = Tree()
        for (bx, by, br) in bl: T.grow(bx, by, br * 1.06, br, rr, kids=kids, lean=lean, rmin=2.5)
        S.put(Mass(T, sun=vs, seed=seed, lz=lz, rimw=rimw, floor=floor), d, seed, nm=nm, sl=sl, rim=rim, Tb=Tb, mk=mk)
    # far little heads along the horizon, hazed into gold
    for (x0, x1, yc, R, s, d) in ((-60, 520, 848, 26, 31, .10), (480, 820, 862, 15, 32, .04), (860, 1180, 858, 17, 33, .04), (1430, 1640, 862, 14, 34, .03), (1600, 2120, 842, 26, 35, .10)):
        S.put(Mass(bank(x0, x1, yc, R, r, .9, .85, (5,), .7, big=.35), sun=(1240, 560), seed=s, lz=-.25, rimw=3, floor=(868, 18)), d, s, nm=160, sl=(.02, .7), rim=.6)
    # left mass, mid distance, behind the balustrade
    tower([(330, 640, 62), (410, 690, 78), (255, 700, 88), (150, 730, 80), (500, 755, 70), (360, 780, 110), (210, 800, 110), (580, 815, 56), (470, 835, 90), (660, 850, 40)], 41, .50, (900, 330), .25, floor=(850, 40), lz=-.24, nm=500, mk=(.04, .1))
    # right tower: the great cumulus
    tower([(1905, 432, 92), (1985, 505, 128), (1830, 520, 112), (1945, 600, 150), (1735, 628, 92), (1840, 690, 140), (2030, 700, 170), (1660, 742, 80), (1770, 780, 130), (1940, 800, 160), (1600, 826, 62), (1700, 850, 100), (1860, 868, 140), (1530, 866, 40), (2040, 870, 130)], 3, .72, (1090, 400), -.3, floor=(858, 60), lz=-.24, nm=900, sl=(.12, .86), mk=(.04, .10))
    S.mist(838, 930, .02, .62, 3)
    # sea fill
    x, y = grid()
    a = smooth(HZ - 10, HZ + 14, y + 5 * fbm(H, W, (60,), None, 5))
    for s in 'AB':
        d = np.clip((y - HZ) / 300, 0, 1)
        c0 = np.stack([pal(s, 0.)[3], pal(s, 1.)[3]])
        c = c0[0] * (1 - d[..., None]) + c0[1] * d[..., None]
        S.c[s] = S.c[s] * (1 - a[..., None]) + c * a[..., None]
    S.T += (22 - S.T) * a; S.a = S.a * (1 - a) + a
    # banks of the sea, far to near: (y, R, x ranges)
    rows = [(884, 9, [(-40, 1180), (1400, 2100)]), (890, 12, [(300, 1300), (1330, 2100)]), (898, 16, [(700, 1240), (1290, 1720), (1800, 2120)]), (908, 22, [(560, 1120), (1180, 1560), (1640, 2120)]),
            (922, 30, [(520, 1010), (1120, 1390), (1500, 2120)]), (942, 42, [(500, 1270), (1440, 1760), (1860, 2160)]), (970, 58, [(440, 1040), (1250, 1640), (1720, 2180)]),
            (1010, 78, [(400, 760), (860, 1430), (1560, 2060)]), (1062, 104, [(420, 1150), (1290, 1760), (1900, 2200)]), (1126, 135, [(380, 700), (820, 1480), (1600, 2220)])]
    for k, (yc, R, xs) in enumerate(rows):
        d = (k / (len(rows) - 1)) ** .8
        for j, (a, b) in enumerate(xs):
            m = Mass(bank(a, b, yc + r.normal(0, R * .1), R, r, .55, .56 + .1 * d, (5,), .78), sun=(1310, HZ - 260), seed=100 + k * 7 + j, lz=-.30 - .1 * d, rimw=2 + 5 * d, floor=(yc + R * .75, R * 1.0))
            S.put(m, .06 + .86 * d, 100 + k * 7 + j, nm=int(160 + 160 * d), sl=(.12, .8), rim=.9, up=(-.6, .7))
        S.mist(yc + R * .35, yc + R * 1.75, min(1, .12 + .75 * d), .55, 50 + k)
    return S


if __name__ == '__main__':
    import time
    from lp import comp
    t = time.time(); S = clouds(); print('clouds', time.time() - t)
    for s in 'AB':
        im = sky(s); im = im * (1 - S.a[..., None]) + S.c[s]
        o = comp.finish(im, 3); comp.save('qa/s7/sky_%s.jpg' % s, o); comp.save('qa/s7/sky_%s_c.jpg' % s, o[700:1150, 1300:1900])
    print(time.time() - t)
