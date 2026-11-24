"""s9 The Last Page.  9a f 0-150 (balcony lane), 9b f 150-312 in shots/s9b.py (ink lane)."""
import numpy as np, math, cv2, os, importlib
from lp import W, H, comp, foxact
from lp.paint import f32, rgb, mix, smooth, fbm, Canvas, arcs, jit_color, grid
from lp.ink import Cel, spl, over_roi
from shots import s9a, balc_life as life
from shots.balc_life import Grass, Ivy, Sheet, paper, rot
from shots.s9a import pj, proj3, floor_xy, SUN, FC, CX, CY, EZ

VER = 'v5'
MF = s9a.MF


def cached(name, fn):
    """dict of arrays, one float16 .npy each under cache/s9_<name>_<VER>_*.npy, memory-mapped on load"""
    os.makedirs('cache', exist_ok=True)
    stem = f'cache/s9_{name}_{VER}'; idx = stem + '.txt'
    if not os.path.exists(idx):
        d = fn()
        for k, v in d.items():
            tmp = f'{stem}_{k}.{os.getpid()}.npy'; np.save(tmp, v.astype(np.float16)); os.replace(tmp, f'{stem}_{k}.npy')
        open(idx + '.tmp%d' % os.getpid(), 'w').write('\n'.join(d)); os.replace(idx + '.tmp%d' % os.getpid(), idx)
    return {k: np.load(f'{stem}_{k}.npy', mmap_mode='r') for k in open(idx).read().split()}


LIGHT = np.array([(SUN[0] - CX) / FC, 1., (CY - SUN[1]) / FC]); LIGHT /= np.linalg.norm(LIGHT)


def still_air(x, f, amp=1.):
    x = np.asarray(x, float)
    return amp * (.10 + .07 * np.sin(2 * np.pi * (f / 47. - x / 1700.)) + .04 * np.sin(2 * np.pi * (f / 17. - x / 500.) + 1.))


class Page:
    """one falling sheet with its own arc.  Positions in metres; lands at `land` (X, Y) on frame fl, or passes out of sight."""
    def __init__(self, sheet, f0, f1, X, Y, Z, swing=.35, turns=1.6, ph=0., yaw=0., spin=.5, bend=1.0, beta=.5, land=None, drift=0., seed=0, yaw1=None, blur=0.):
        self.__dict__.update(locals())

    def state(self, f):
        f0, f1 = self.f0, self.f1
        t = (f - f0) / float(f1 - f0)
        if t < 0: return None
        if t > 1 and self.land is None: return None
        tt = min(1., t)
        w = 2 * math.pi * self.turns
        a = w * tt + self.ph
        X1, Y1 = self.land if self.land else (self.X + self.drift, self.Y)
        Z1 = 0.004 if self.land else -.25
        k = tt
        damp = 1 - .55 * tt if not self.land else (1 - tt ** 3)
        X = self.X + (X1 - self.X) * k + self.swing * (math.sin(a) - math.sin(self.ph) * (1 - k) - math.sin(w + self.ph) * k) * damp
        Y = self.Y + (Y1 - self.Y) * k + .30 * self.swing * (math.cos(a) - math.cos(self.ph)) * damp * (1 - k)
        # falls fast through the middle of each swing, hangs at its ends
        zz = tt - .55 * math.sin(2 * a) / (2 * w) + .55 * math.sin(2 * self.ph) / (2 * w) * (1 - tt) + .55 * math.sin(2 * (w + self.ph)) / (2 * w) * tt
        zz = min(1., max(0., zz))
        if self.land: zz = 1 - (1 - zz) ** 1.15
        Z = self.Z + (Z1 - self.Z) * zz
        flat = smooth(.80, 1.0, tt) if self.land else 0.
        yw = self.yaw + self.spin * 2 * math.pi * tt
        if self.yaw1 is not None: yw = yw + (self.yaw1 - yw) * float(smooth(.55, 1., tt))
        pitch = (.95 * math.cos(a) * .75 + .35 * math.sin(a * .5 + 1)) * (1 - flat)
        roll = (-1.05 * math.cos(a)) * (1 - flat) * damp
        bend = self.bend * 1.45 * math.sin(a * 1.0 + .6) * (1 - flat) + .10 * flat
        return (X, Y, max(Z, .004)), rot(yw, pitch, roll), bend, .95 * math.sin(a * .5 + self.seed) * (1 - flat) + .18 * flat * (self.seed % 3), float(flat), tt


class A9:
    def setup(self):
        P = cached('a', s9a.plate9)
        self.P = {k: np.array(P[k], f32) for k in ('bg', 'lm', 'beam', 'sh')}
        x, y = grid(); self.x, self.y = x, y
        r = np.random.default_rng(12)
        # ---------------------------------------------------------- pages
        mk = lambda k, w, h, kind, torn=None: Sheet(w, h, 60 + k, kind, torn)
        L = floor_xy
        pg = []
        pg.append(Page(mk(0, .25, .34, 'text'), -14, 30, -1.10, 1.8, 1.10, .22, 1.3, .4, .5, .35, 1.2, .6, None, -.35, 1))
        pg.append(Page(mk(1, .22, .30, 'pict'), -4, 44, .74, 2.0, 1.00, .20, 1.7, 2.2, -.6, -.4, 1.0, 1.1, None, .28, 2))
        pg.append(Page(mk(2, .30, .40, 'letter', 2), -10, 40, -1.35, 3.5, 1.65, .45, 1.8, 1.0, .9, .45, 1.2, .4, tuple(L(418, 872)), 0, 3, yaw1=.7))
        pg.append(Page(mk(3, .26, .34, 'text', 0), -18, 34, .95, 4.4, 1.9, .40, 1.7, 3.1, .2, -.5, 1.1, 2.1, tuple(L(1580, 868)), 0, 4, yaw1=-.5))
        pg.append(Page(mk(4, .22, .30, 'pict'), 0, 52, -.10, 4.0, 1.75, .38, 2.1, .2, -.4, .6, 1.3, .9, tuple(L(1052, 842)), 0, 5, yaw1=1.1))
        pg.append(Page(mk(5, .32, .44, 'text'), -12, 26, -2.1, 5.2, 2.2, .5, 1.5, 4.0, .3, .4, 1.0, 1.6, tuple(L(225, 838)), 0, 6, yaw1=.2))
        pg.append(Page(mk(6, .24, .32, 'text'), -6, 48, 2.15, 3.1, 1.30, .34, 1.9, 5.0, -.8, .5, 1.1, .3, tuple(L(1835, 908)), 0, 7, yaw1=.9))
        pg.append(Page(mk(7, .28, .38, 'letter'), -22, 40, -2.9, 8.8, 2.3, .6, 1.8, 2.6, .1, .5, 1.2, 1.2, None, .5, 8))
        pg.append(Page(mk(8, .28, .36, 'pict'), 4, 92, .35, 9.6, 2.9, .6, 2.7, .9, .7, .7, 1.2, .5, None, -.3, 9))
        pg.append(Page(mk(9, .26, .36, 'text'), 12, 98, 1.9, 9.0, 2.8, .55, 2.5, 3.6, -.3, -.6, 1.1, 1.9, None, .2, 10))
        pg.append(Page(mk(10, .28, .38, 'text', 1), 20, 96, -1.6, 10.2, 3.1, .65, 2.4, 1.9, .5, .6, 1.2, .8, None, .4, 11))
        pg.append(Page(mk(11, .20, .28, 'letter'), -8, 22, .12, 2.9, 1.1, .3, 1.2, .0, .2, .5, 1.2, 2.4, tuple(L(1370, 818)), 0, 12, yaw1=-.2))
        self.pages = pg
        self.blank = Sheet(.275, .40, 5, 'blank'); self.blank.seed = 0; self.blank.creases = [(np.array([-.135, .03]), np.array([.135, .015]))]
        tx, ty = floor_xy(*MF['page'])
        self.bl = Page(self.blank, 6, 46, .80, 2.75, 1.16, .33, 1.75, .9, .9, -.35, 1.15, .7, (tx + .05, ty), 0, 3, yaw1=-.30)
        # ---------------------------------------------------------- grass and ivy
        def tuft(sx, sy, n, h, sp): return (sx, sy, n, h, sp, sy)
        T = [tuft(90, 1150, 26, 210, 60), tuft(250, 1172, 16, 150, 40), tuft(-10, 1080, 18, 150, 40), tuft(395, 1128, 9, 80, 24), tuft(330, 1030, 7, 54, 16), tuft(620, 1158, 10, 84, 22), tuft(905, 1112, 6, 46, 14), tuft(1440, 1060, 8, 60, 16), tuft(1725, 1010, 12, 84, 22),
             tuft(1960, 1150, 26, 220, 60), tuft(1830, 1165, 16, 140, 40), tuft(2060, 1040, 16, 130, 34), tuft(1650, 1140, 9, 80, 22), tuft(1250, 1150, 6, 52, 14), tuft(1100, 930, 4, 26, 8), tuft(520, 960, 6, 40, 12), tuft(660, 905, 5, 30, 9), tuft(30, 960, 10, 66, 18)]
        self.gf = Grass(T, 5, wk=.050, wmin=1.0, lean=.5)
        B = []
        for k in range(46):
            sx = r.uniform(-20, 2060); B.append((sx, 762 + r.uniform(0, 5), int(r.integers(4, 10)), r.uniform(12, 34), r.uniform(5, 13), 1.))
        for k in range(16):
            sy = r.uniform(790, 880); sx = r.uniform(0, 2048); B.append((sx, sy, int(r.integers(3, 7)), r.uniform(12, 28), r.uniform(4, 9), 2.))
        self.gb = Grass(B, 6, wk=.06, lean=.5)
        tp = 560.
        st = []
        for x0 in (6, 40, 92, 130, 188, 236, 262, 300, 352, 392, 440, 505, 560, 622, 660, 700, 1108, 1140, 1176, 1200, 1815, 1870, 1940, 2000):
            L_ = r.uniform(40, 170) * (1.15 if x0 < 420 or x0 > 1800 else .75)
            n = 5; st.append([(x0 + r.normal(0, 5) * k, tp - 4 + L_ * k / (n - 1)) for k in range(n)])
        st += [[(-20, 552), (60, 556), (150, 548), (235, 556), (330, 552), (430, 560)], [(120, 530), (190, 526), (250, 545)], [(600, 552), (660, 528), (720, 552), (770, 562)], [(1085, 540), (1140, 526), (1205, 545)], [(1800, 556), (1890, 552), (1980, 558), (2070, 552)],
               [(150, 760), (152, 700), (146, 640), (158, 590), (150, 540)], [(1110, 760), (1106, 690), (1114, 620), (1108, 560)], [(1706, 760), (1700, 700), (1708, 640)]]
        self.iv = Ivy(st, 21, 14, 7., free=.22, dens=1.1, pal=[('#4f5763', '#58605a', '#5f5c66'), ('#6b7256', '#7a7d58', '#666c5c'), ('#97945a', '#aea45e', '#8a8a56'), ('#e3d387', '#f3e39c', '#d6c673')])
        # dust
        n = 110
        self.du = np.c_[r.uniform(0, W, n), r.uniform(250, 1120, n), r.uniform(.6, 1.9, n), r.uniform(0, 6.28, n), r.uniform(-.25, -.05, n), r.uniform(-.06, .16, n)]
        return self

    # --------------------------------------------------------------- cels
    def sheet(self, img, p, f, lw=1.6, shadow=True, light=None, pal=None):
        d = f // 2 * 2
        s = p.state(d)
        if s is None: return img
        pos, R, bend, curl, flat, tt = s
        sc = FC / pos[1]; c0 = pj(*pos); rad = max(p.sheet.w, p.sheet.h) * sc * .85 + 14
        if c0[1] < -rad or c0[1] > H + rad or c0[0] < -rad or c0[0] > W + rad: return img
        if shadow and p.land is not None:
            # floor shadow: sun is low, so it lies far to the camera-left and closes in as the sheet comes down
            # every corner throws its own shadow point; a tilted sheet makes a long one, an edge-on sheet a sliver
            sl_ = .22
            Q = []
            for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                e = R @ np.array([u * p.sheet.w / 2, v * p.sheet.h / 2, 0.]); zc = max(0., pos[2] + e[2])
                Q.append((pos[0] + e[0] - LIGHT[0] / LIGHT[1] * zc / sl_, pos[1] + e[1] - zc / sl_))
            Q = np.array(Q)
            if pos[2] < .55 and Q[:, 1].min() > 1.2:
                P2 = pj(Q[:, 0], Q[:, 1], 0 * Q[:, 0]); mn, mx = P2.min(0) - 14, P2.max(0) + 14
                if mx[0] > 0 and mn[0] < W and mx[0] - mn[0] < 1400:
                    c = Cel(int(mx[0] - mn[0]), int(mx[1] - mn[1]), int(mn[0]), int(mn[1]))
                    n4 = [P2[k] + (P2[(k + 1) % 4] - P2[k]) * t for k in range(4) for t in (0, .35, .7)]
                    c.fill(spl(np.array(n4) + np.random.default_rng(p.seed * 13 + d // 2).normal(0, .8, (12, 2)), closed=True, corners=(0, 3, 6, 9)), '#7d6384', .50 * (1 - pos[2] / .55) ** 1.5)
                    img = over_roi(img, cv2.GaussianBlur(c.rgba(), (0, 0), 1.2 + 7 * pos[2]), c.ox, c.oy)
        light = LIGHT + np.array([0, 0, .55 * flat + (.18 if p is self.bl else 0)]) if light is None else light
        b = float(self.P['sh'][int(np.clip(c0[1], 0, H - 1)), int(np.clip(c0[0], 0, W - 1))]) + .6 * float(self.P['beam'][int(np.clip(c0[1], 0, H - 1)), int(np.clip(c0[0], 0, W - 1))])
        pl = dict(life.PAPER)
        g = float(np.clip(b, 0, 1))
        pl = {k: (mix(rgb(v), rgb('#fff0cc'), .24 * g) if k in ('trans', 'lit', 'burn') else mix(rgb(v), rgb('#d7b39c'), .25 * g) if k == 'sh' else rgb(v)) for k, v in pl.items()}
        if pal: pl.update(pal)
        c = Cel(int(rad * 2), int(rad * 2), int(c0[0] - rad), int(c0[1] - rad))
        beta = p.beta
        paper(c, p.sheet, pos, R, proj3, bend=bend, beta=beta, curl=curl, light=LIGHT if light is None else light, eye=(0, 0, EZ), pal=pl, lw=float(np.clip(lw * sc / 520, .7, 2.2)), seed=p.seed * 100 + d // 2, N=13, boil=float(np.clip(sc / 900, .1, .5)), detail=sc > 150)
        L = c.rgba()
        if p.blur: L = cv2.GaussianBlur(L, (0, 0), p.blur)
        return over_roi(img, L, c.ox, c.oy)

    def blankpage(self, img, f):
        d = f // 2 * 2
        if d < 46: return self.sheet(img, self.bl, f, 1.8, pal=dict(lit=rgb('#fdeccb'), burn=rgb('#fff6e0'), trans=rgb('#fdeac4'), blank=rgb('#fff0d0'), sh=rgb('#cfa796')))
        # landed: a short slide, a corner that lifts and falls.  Bespoke drawing: the sheet, the bent corner, its shadow.
        tx, ty = floor_xy(*MF['page'])
        sl = {46: .055, 48: .030, 50: .012, 52: .004}.get(d, 0.)
        al = math.radians({46: 22, 48: 50, 50: 68, 52: 56, 54: 34, 56: 18, 58: 27, 60: 23}.get(d, 21))
        dr = min(d, 60); rr = np.random.default_rng(900 + dr)
        yaw = -.30 - sl * .8; hw, hh = .1375, .20
        X1 = np.array([math.cos(yaw), math.sin(yaw), 0]); Y1 = np.array([-math.sin(yaw), math.cos(yaw), 0]); C0 = np.array([tx + sl, ty, .003])
        P3 = lambda u, v, z=0.: C0 + X1 * u * hw + Y1 * v * hh + np.array([0, 0, z])
        pr = lambda q: proj3(np.asarray(q)[None])[0][0]
        def wavy(p, q, n, amp, sd):
            r2 = np.random.default_rng(sd); t = np.linspace(0, 1, n); e = q - p; nn = np.array([-e[1], e[0]]) / (np.hypot(*e) + 1e-6)
            return p + np.outer(t, e) + np.outer(np.r_[0, r2.normal(0, amp, n - 2), 0], nn) + np.random.default_rng(sd * 7 + dr).normal(0, .22, (n, 2))
        FL, FR, NL, NR = P3(-1, 1), P3(1, 1.01), P3(-1.01, -1), P3(1, -1)
        fa, fb = .60, .62                                                  # where the bend meets the right edge and the near edge
        Ap, Bp = NR + (FR - NR) * fa, NR + (NL - NR) * fb
        ab = Bp - Ap; ft = Ap + ab * ((NR - Ap) @ ab) / (ab @ ab); arm = NR - ft; L = np.linalg.norm(arm); arm = arm / L
        # the corner is a curl, not a hinge: march along the arm with growing slope
        def lift(q):
            f2 = Ap + ab * ((q - Ap) @ ab) / (ab @ ab); dd = np.linalg.norm(q - f2); n_ = 10; h = dd / n_; o = np.zeros(3) + f2; 
            for k in range(n_):
                sl_ = al * min(1., (k + .5) / n_ * dd / L * 1.5); o = o + (arm * math.cos(sl_) + np.array([0, 0, 1.]) * math.sin(sl_)) * h
            return o
        eA = np.array([pr(lift(Ap + (NR - Ap) * t)) for t in np.linspace(0, 1, 8)]); eB = np.array([pr(lift(Bp + (NR - Bp) * t)) for t in np.linspace(0, 1, 8)])
        pFL, pFR, pNL, pA, pB, tip = pr(FL), pr(FR), pr(NL), pr(Ap), pr(Bp), eA[-1]
        q = pr(C0); c = Cel(620, 280, int(q[0] - 320), int(q[1] - 150))
        # cast shadow: toward camera-left, plus the corner's own
        sv = np.array([-16., 10.])
        c.fill(spl(np.array([pFL + sv * .5, pFR + sv * .25, pA + sv * .4, (tip + pr(NR)) / 2 + sv * 1.0 + (-26 * math.sin(al), 8 * math.sin(al)), pB + sv * 1.2, pNL + sv]), closed=True, corners=(0, 1, 2, 4, 5)), '#8a6a78', .58)
        e1, e2, e3, e4 = wavy(pNL, pFL, 8, .9, 1), wavy(pFL, pFR, 12, .7, 2), wavy(pFR, pA, 7, .6, 3), wavy(pB, pNL, 9, 1.0, 4)
        sheet = np.vstack([e1, e2, e3, pB[None], e4])
        pap, pap2, und, inkc = '#f6dfb3', '#fcebc9', '#c39a90', '#7a5a4b'
        c.fill(sheet, pap)
        c.shade(np.array([pFL + (0, 0), pFR, pFR + (pA - pFR) * .55, pFL + (pNL - pFL) * .30]), pap2, within=sheet)         # the far half takes more of the low sun
        for (pc, d1, d2, aa) in ((pFL, pFR - pFL, pNL - pFL, .55), (pNL, pFL - pNL, pr(P3(1, -1)) - pNL, .42), (pFR, pFL - pFR, pA - pFR, .38)):         # corners browned by age
            c.shade(spl(np.array([pc - (d1 + d2) * .1, pc + d1 * .17, pc + d1 * .10 + d2 * .10, pc + d2 * .24]), closed=True, corners=(0,)), '#e5c191', within=sheet, alpha=aa)
        # old paper: a fold once pressed flat, a few fox-marks near the edge.  Nothing written.
        m0, m1 = pr(P3(-1, .12)), pr(P3(1, .06)); c.line(np.stack([m0 + (m1 - m0) * t for t in np.linspace(0, 1, 7)]), 1.1, '#e0c698', (.1, .1), seed=5, alpha=.8)
        r3 = np.random.default_rng(4)
        for k in range(9):
            u, v = r3.choice([-1, 1]) * r3.uniform(.72, .95), r3.uniform(-.8, .9); p_ = pr(P3(u, v))
            if u > .3 and v < .2: continue
            s_ = r3.uniform(.8, 2.0); c.fill(spl(np.array([p_ + (-s_ * 1.6, 0), p_ + (0, -s_ * .5), p_ + (s_ * 1.6, 0), p_ + (0, s_ * .5)]), closed=True), '#dfc08f', .55)
        c.shade(np.vstack([pA[None] + (-5, 0), pB[None] + (-7, -1), pB[None], pA[None]]), '#ecd0a6', within=sheet, alpha=.9)       # the rise of the bend
        for e_, sd, w_ in ((e1, 11, 1.7), (e2, 12, 1.3), (e3, 13, 1.5), (e4, 14, 2.0)): c.line(spl(e_), w_, inkc, (.05, .05), press=.35, seed=sd + dr, tremor=.3, tmin=.4)
        flap = np.vstack([eA, eB[::-1]])
        # does the corner show us its underside?
        nrm = np.cross(np.array([*(eB[4] - eA[4]), 0.]), np.array([*(tip - (pA + pB) / 2), 0.]))[2]
        back = nrm > 0
        mid = (eA + eB) / 2
        c.fill(flap, und if not back and al > .5 else pap2 if al < .5 else und)
        if al > .5: c.shade(np.vstack([eA[5:], eB[5:][::-1]]), '#e9c3a0', within=flap)                                          # light coming through the thin tip
        for e_, sd, w_ in ((eA, 15, 1.7), (eB, 16, 1.9)): c.line(spl(e_), w_, inkc, (.05, .05), press=.35, seed=sd + dr, tremor=.3, tmin=.4)
        return over_roi(img, c.rgba(), c.ox, c.oy)

    def foxshadow(self, img):
        x, y, h = MF['x'], MF['y'], MF['h']
        c = Cel(W, 330, 0, y - 90, ss=1)
        e = np.array([x - SUN[0], y - CY]); e = e / np.hypot(*e)
        P = [(x + .36 * h, y - 6), (x + .1 * h, y - 22), (x - .3 * h, y - 14), (x - .75 * h, y + 22), (x - 1.35 * h, y + 92), (x - 1.75 * h, y + 150), (x - 1.5 * h, y + 186), (x - .9 * h, y + 120), (x - .35 * h, y + 52), (x + .1 * h, y + 26), (x + .38 * h, y + 8)]
        c.fill(spl(P, closed=True), '#6f5a80', 1.)
        L = c.rgba(); L = cv2.GaussianBlur(L, (0, 0), 5)
        fade = np.clip(1 - (x - self.x[y - 90:y + 240] - 0) / (1.9 * h), .15, 1) * .62
        yy = slice(y - 90, min(H, y + 240)); L = L[:yy.stop - yy.start]
        img[yy] = img[yy] * (1 - L[..., 3:4] * fade[:L.shape[0], :, None]) + L[..., :3] * fade[:L.shape[0], :, None]
        return img

    def frame(self, f):
        P = self.P; d = f // 2
        img = P['bg'].copy()
        calm = float(np.clip(1 - f / 70., 0, 1))
        gfun = lambda x, t: still_air(x, t, 1 + 2.6 * calm)
        hold = f // 3 * 3
        # beams breathe very slowly, held on threes
        bm = P['beam'] * (.20 + .02 * math.sin(hold * .11)) + P['sh'] * (.46 + .05 * math.sin(hold * .07 + 1))
        img = img + (rgb('#ffdb9e') - img) * np.clip(bm, 0, .8)[..., None]
        cv_ = Canvas(img=np.ascontiguousarray(img, f32))
        self.gb.draw(cv_, f, 1., np.array([.94, .9, .88]), gfun=gfun)
        self.iv.draw(cv_, f, .8, np.array([.97, .90, .84]), gfun=gfun)
        img = cv_.img
        for p in self.pages:
            if p.Y > 2.5: img = self.sheet(img, p, f)
        img = self.foxshadow(img)
        late = f >= 30                                                   # the blank page passes behind the fox's shoulder, then lands in front
        if f >= 118: img = self.blankpage(img, f)
        img = foxact.draw(img, 's9a', f, MF)
        if f < 118: img = self.blankpage(img, f)
        cv_ = Canvas(img=np.ascontiguousarray(img, f32))
        self.gf.draw(cv_, f, 1., np.array([.90, .86, .84]), gfun=gfun, K=8)
        img = cv_.img
        for p in self.pages:
            if p.Y <= 2.5: img = self.sheet(img, p, f, 1.9)
        img = self.dust(img, f)
        g = np.exp(-(np.hypot((self.x - SUN[0]) / 1.5, self.y - SUN[1]) / 520) ** 1.6) * .16
        img = img + (rgb('#ffd292') - img) * g[..., None]
        z = 1 + .02 * comp.ease(f / 150.)
        return comp.cam2d(img, 1010 + (W / 2 - 1010) / z, 720 + (H / 2 - 720) / z, z)

    def dust(self, img, f):
        h3 = f // 3 * 3
        for (x, y, s, ph, vx, vy) in self.du:
            px = (x + vx * f * 2 + 10 * math.sin(f * .03 + ph)) % W; py = y + vy * f + 6 * math.sin(f * .04 + ph * 2)
            ix, iy = int(px), int(py)
            if not (4 <= ix < W - 5 and 4 <= iy < H - 5): continue
            b = min(1., float(self.P['beam'][iy, ix]) * .9 + float(self.P['sh'][iy, ix]) * .9) * max(0., math.sin(h3 * .12 + ph * 3)) ** 1.5
            if b < .06: continue
            yy, xx = np.mgrid[iy - 4:iy + 5, ix - 4:ix + 5]
            k = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * s * s)) * b * .85
            img[iy - 4:iy + 5, ix - 4:ix + 5] += (np.array([1.02, .95, .78], f32) - img[iy - 4:iy + 5, ix - 4:ix + 5]) * k[..., None].astype(f32)
        return img


class Shot:
    id = 's9'; n = 312
    finish = dict(hal_amt=.24, vig=.10)
    FOX = {'s9a': MF}

    def setup(self):
        self.a = A9(); self.a.setup(); self.close_setup()
        self.b = None
        if os.path.exists(os.path.join(os.path.dirname(__file__), 's9b.py')):
            import shots.s9b as b
            b.setup(); self.b = b

    # lead: the look to us, f 76-120, a closer drawing of the same moment
    def close_setup(self):
        bg = self.a.P['bg']
        z = 1.95
        pl = comp.cam2d(bg, 1120, 640, z)
        pl = comp.blur(pl, 3.0); pl = cv2.filter2D(pl, -1, comp.disc_kernel(17.5), borderType=cv2.BORDER_REPLICATE)
        g = np.exp(-(np.hypot((self.a.x - 1760) / 1.5, self.a.y - 250) / 640) ** 1.7) * .30
        pl = pl + (rgb('#ffd899') - pl) * g[..., None]
        self.cb = np.clip(pl, 0, 1.05).astype(f32)
        r = np.random.default_rng(77)
        self.mt = np.c_[r.uniform(0, W, 26), r.uniform(0, H, 26), r.uniform(1.5, 5, 26), r.uniform(0, 6.28, 26)]

    def close(self, f):
        g = f - 76; gd = g - g % 2
        img = comp.cam2d(self.cb, W / 2 + 6, H / 2 + 4, 1.03 + .012 * g / 44.)
        E = lambda a, b: float(comp.ease((gd - a) / float(b - a)))
        up = E(8, 14) if gd >= 8 else 0.
        tl = E(24, 32) if gd >= 24 else 0.
        bl = 0.
        for a, v in ((10, .55), (12, 1.), (16, .5), (18, 0.), (34, .5), (36, 1.), (40, .5), (42, 0.)):
            if gd >= a: bl = v
        hd = dict(yaw=.34 * (1 - up) + .03 * up, pitch=-.30 * (1 - up) + .04 * up, roll=.02 * (1 - up) - .075 * tl, look=(.5 * (1 - up), .85 * (1 - up)), blink=bl,
                  ear_l=(.10 * tl, .04), ear_r=(-.30 * tl, .04), wide=.35 * up, pupil=1.04, lantern=False, brow=.10, mouth=0.)
        br = .5 + .5 * math.sin(gd * .16)
        seed = foxact.pseed(up, tl)
        s = 2.02 * (1 + .012 * g / 44.)
        img = foxact.closeup(img, 905 - 10 * tl, 600 + 12 * (1 - up) - 2 * br, s, f, hd, 'gold', key=1, lw=3.0, seed=seed, body_seed=13, breath=2 * br, under=False, rim_k=1, turn=.55 - .35 * up,
                             tint=((.88, .80, .88), (1.06, 1.02, .94), (400, 1100), (1500, 200)))
        c = Cel(W, H); h3 = g // 3
        for (x, y, r_, ph) in self.mt:
            xx = (x + h3 * .8 + 10 * math.sin(ph + h3 * .12)) % W; yy = (y + h3 * .6 + 8 * math.sin(ph * 2 + h3 * .1)) % H
            c.fill(spl([[xx - r_, yy], [xx, yy - r_ * .8], [xx + r_, yy], [xx, yy + r_]], closed=True), '#fff0c4', .12 + .3 * (.5 + .5 * math.sin(ph + h3 * .4)))
        L = c.rgba()
        return over_roi(img, L, 0, 0)

    def frame(self, f):
        if 76 <= f < 120: return self.close(f)
        if 120 <= f < 150:                       # cut in on the action: paw to page
            z = 1.30 + .02 * comp.ease((f - 120) / 30.)
            return comp.cam2d(self.a.frame(f), 1040, 700, z, interp=cv2.INTER_CUBIC)
        if f < 150: return self.a.frame(f)
        if self.b is not None: return self.b.frame(f - 150)
        return np.ones((H, W, 3), f32) * rgb('#f1e2bf')
