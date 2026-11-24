"""s7 Sunrise.  7a wide 0-72, 7b close profile 72-94, 7c aerial ring of wind 94-120.   Owner: balcony lane."""
import numpy as np, math, cv2, os
from lp import W, H, comp, foxact
from lp.paint import f32, rgb, mix, smooth, fbm, Canvas, arcs, jit_color, grid
from lp.ink import Cel, spl, over_roi
from shots import s7a_sky, s7a_stone, balc_life as life
from shots.balc_life import Grass, Ivy, Sheet, paper, rot, gust

VER = 'v15'


def cached(name, fn, keys_only=False):
    """dict of arrays, one float16 .npy each under cache/s7_<name>_<VER>_*.npy, memory-mapped on load"""
    os.makedirs('cache', exist_ok=True)
    stem = f'cache/s7_{name}_{VER}'
    idx = stem + '.txt'
    if not os.path.exists(idx):
        d = fn()
        for k, v in d.items():
            tmp = f'{stem}_{k}.{os.getpid()}.npy'; np.save(tmp, v.astype(np.float16)); os.replace(tmp, f'{stem}_{k}.npy')
        open(idx + '.tmp%d' % os.getpid(), 'w').write('\n'.join(d)); os.replace(idx + '.tmp%d' % os.getpid(), idx)
    if keys_only: return None
    return {k: np.load(f'{stem}_{k}.npy', mmap_mode='r') for k in open(idx).read().split()}


def zoomto(img, px, py, z, dx=0., dy=0.):
    return comp.cam2d(img, px + dx + (W / 2 - px) / z, py + dy + (H / 2 - py) / z, z)


# ========================================================================= 7a
MA = dict(x=700, y=905, h=100)


def plate_a():
    S = s7a_sky.clouds()
    SA, SB = s7a_sky.sky('A'), s7a_sky.sky('B')
    L, G, T = s7a_stone.build()
    h, w = L.shape[:2]
    A, B, al, TM = S.c['A'], S.c['B'], S.a, S.T
    st = np.zeros((H, W), f32)
    a = L[..., 3:4]
    tintA = np.array([.74, .76, .90], f32)
    A[:h, :w] = A[:h, :w] * (1 - a) + L[..., :3] * tintA
    Bs = L[..., :3] * np.array([1.03, 1.0, .99], f32)
    B[:h, :w] = B[:h, :w] * (1 - a) + Bs * (1 - G[..., 3:4]) + G[..., :3]
    TM[:h, :w] = TM[:h, :w] * (1 - a[..., 0]) + T * a[..., 0]
    al[:h, :w] = al[:h, :w] * (1 - a[..., 0]) + a[..., 0]
    st[:h, :w] = a[..., 0]
    # painted rays fanning from the sun
    x, y = grid()
    sx, sy = 1310., 800.
    an = np.arctan2(y - sy, x - sx); rd = np.hypot(x - sx, y - sy)
    r = np.random.default_rng(8); g = np.convolve(r.standard_normal(720), np.hanning(11), 'same'); g = (g - g.min()) / np.ptp(g)
    ray = np.interp((an + np.pi) / (2 * np.pi) * 719, np.arange(720), g).astype(f32)
    ray = smooth(.42, .8, ray) * np.exp(-(rd / 820) ** 1.3) * smooth(60, 190, rd)
    return dict(SA=SA, SB=SB, A=A, B=B, al=al, T=TM, st=st, ray=cv2.GaussianBlur(ray, (0, 0), 2))


class A7:
    def setup(self):
        self.P = {k: np.array(v, f32) for k, v in cached("a", plate_a).items()}
        x, y = grid(); self.x, self.y = x, y
        r = np.random.default_rng(5)
        pj = s7a_stone.pj
        # grass: back tufts (behind fox) and lip tufts (in front)
        back, front = [], []
        for k in range(15):
            X = r.uniform(-6.0, -2.5); Y = r.uniform(11.2, 12.1)
            if -4.15 < X < -3.2 and Y > 10: Y = 12.0; h = 14
            else: h = r.uniform(18, 40)
            p = pj(X, Y, 0); back.append((p[0], p[1] + 1, int(r.integers(5, 12)), h, r.uniform(6, 13), 1 / Y))
        for (X, n, h, sp) in ((-6.1, 16, 62, 20), (-5.55, 10, 40, 12), (-5.1, 18, 58, 17), (-4.72, 8, 34, 9), (-4.05, 14, 50, 14), (-3.72, 7, 30, 8), (-3.62, 20, 82, 19), (-3.42, 9, 48, 10), (-3.0, 5, 17, 7), (-2.78, 6, 20, 7), (-2.62, 11, 44, 9), (-6.5, 14, 66, 22)):
            p = pj(X, 8.52 + r.uniform(0, .12), 0); front.append((p[0], p[1] + 2, n, h, sp, 1.))
        for (X, Y, n, h, sp) in ((-4.4, 9.6, 9, 34, 10), (-3.15, 9.4, 7, 26, 8), (-5.2, 9.7, 10, 36, 11), (-2.3, 10.2, 13, 48, 11), (-2.62, 9.5, 15, 62, 13), (-2.4, 11.3, 9, 34, 9)):
            p = pj(X, Y, 0); back.append((p[0], p[1] + 1, n, h, sp, 1 / Y))
        # tufts growing out of the fascia joints, hanging
        side = [(pj(-5.7, 8.6, -.52)[0], pj(0, 8.6, -.5)[1], 7, 26, 9, 2.), (pj(-4.5, 8.6, -.16)[0], pj(0, 8.6, -.15)[1], 8, 24, 8, 2.), (pj(-3.2, 8.6, -.5)[0], pj(0, 8.6, -.5)[1], 9, 30, 8, 2.)]
        self.gb, self.gf = Grass(back, 3), Grass(front + side, 4, wk=.06)
        self.gw = Grass([(390 - 2, 130, 5, 30, 7, 1), (378, 372, 8, 40, 9, 1), (374, 610, 6, 30, 8, 1), (300, 898, 14, 44, 16, 1), (352, 894, 10, 36, 10, 1)], 8)
        # ivy: wall curtains, silhouette sprays, drape over the lip
        st = [[(20, -30), (46, 60), (34, 150), (60, 260), (52, 340)], [(90, -30), (104, 70), (126, 170), (118, 250)], [(150, -30), (172, 40), (166, 120), (188, 190), (182, 262)],
              [(-20, 120), (14, 200), (8, 300), (30, 410), (24, 500), (44, 590)], [(60, -30), (72, 100), (88, 210), (80, 300), (100, 420), (96, 500)],
              [(236, -30), (250, 30), (244, 90), (262, 140)], [(330, -30), (352, 60), (364, 150), (372, 210), (380, 260)], [(340, -30), (366, 40), (380, 110), (392, 160)], [(-20, 560), (10, 620), (6, 690), (26, 760)]]
        self.iw = Ivy(st, 11, 13, 6.5, free=[.3, .35, .4, .2, .25, .5, .4, .55, .3], dens=1.15)
        st2 = [[(60, 918), (66, 940), (60, 980), (74, 1030), (68, 1080)], [(120, 920), (132, 950), (126, 1000), (140, 1046)], [(188, 920), (190, 960), (204, 1010), (198, 1060), (210, 1120)], [(160, 918), (152, 948), (166, 990)],
               [(262, 921), (270, 950), (262, 985)], [(20, 918), (12, 960), (24, 1020), (18, 1090), (30, 1150)], [(90, 918), (98, 970), (92, 1020), (108, 1070), (104, 1110)], [(236, 920), (244, 950), (236, 1000), (250, 1050)],
               [(430, 764), (438, 790), (432, 820), (446, 850)], [(402, 766), (398, 790), (410, 816)], [(470, 758), (480, 780), (474, 806), (484, 836)], [(392, 760), (440, 758), (498, 748)]]
        self.il = Ivy(st2, 12, 11, 6., free=.35, dens=1.2)
        self.sheets = [Sheet(34, 46, 40 + k, kind) for k, kind in enumerate(('text', 'pict', 'text', 'letter', 'text', 'pict', 'text'))]
        # flights: start frame, dur, path control points, size
        self.fl = [(2, 18, [(-70, 700), (250, 520), (620, 250), (980, -80)], 1.0, .5), (7, 20, [(-60, 520), (260, 480), (700, 420), (1150, -60)], .8, 2.1), (13, 24, [(60, 1210), (420, 1010), (780, 600), (1010, -60)], 1.15, 4.0),
                   (20, 26, [(-60, 840), (300, 640), (760, 520), (1420, -60)], .62, 1.0), (30, 26, [(140, 1200), (360, 960), (560, 560), (640, -70)], .95, 3.0), (39, 28, [(-60, 420), (320, 430), (900, 360), (1640, -50)], .5, 5.0), (47, 26, [(260, 1210), (620, 1060), (960, 720), (1260, -60)], .8, .2)]
        return self

    def sun_y(self, f):
        t = np.clip(f / 52., 0, 1); t = 1 - (1 - t) ** 1.35
        return 938 + (788 - 938) * t - max(0, f - 52) * .35

    def light(self, f):
        """global daylight 0..1 in held steps on twos"""
        d = f // 2 * 2
        return float(comp.ease(np.clip((d + 2) / 54., 0, 1)))

    def frame(self, f):
        P = self.P; d = f // 2 * 2
        k = self.light(f)
        sky = P['SA'] + (P['SB'] - P['SA']) * k
        cx, cy = 1310., self.sun_y(f)
        # gold pooling round the disc: the sky moves toward gold, it does not blow out
        dist = np.hypot(self.x - cx, (self.y - min(cy, 884)) * 1.12)
        to = lambda im, col, w: im + (rgb(col) - im) * w[..., None]
        sky = to(sky, '#ffc888', np.exp(-(dist / 520) ** 1.4) * (.22 + .42 * k))
        sky = to(sky, '#ffdba2', np.exp(-(dist / 190) ** 1.5) * (.40 + .34 * k))
        sky = to(sky, '#fff3cc', np.exp(-(np.maximum(0, dist - 60) / 42) ** 1.2) * (.55 + .35 * k))
        # the disc, redrawn on threes
        rr = np.random.default_rng(300 + f // 3)
        th = np.arctan2(self.y[700:1060, 1180:1440] - cy, self.x[700:1060, 1180:1440] - cx)
        rd = np.hypot(self.x[700:1060, 1180:1440] - cx, self.y[700:1060, 1180:1440] - cy)
        wob = np.interp(th, np.linspace(-np.pi, np.pi, 25), np.r_[(q := rr.normal(0, .5, 24)), q[0]])
        a = np.clip(s7a_sky.SUNR + wob - rd + .5, 0, 1)[..., None]
        sky[700:1060, 1180:1440] += (np.array([1.20, 1.11, .88], f32) - sky[700:1060, 1180:1440]) * a
        # clouds and stone, lit in held steps
        t = d + 1.0
        soft = 1.2 + 4.5 * (1 - P['st'])
        m = smooth(P['T'] - soft, P['T'] + soft, t)[..., None]
        F = P['A'] + (P['B'] - P['A']) * m
        img = sky * (1 - P['al'][..., None]) + F
        # burning rim where the light spills over the far crests
        al = P['al']
        edge = np.clip(al - np.roll(al, 4, 0), 0, 1) * (1 - P['st'])
        if not hasattr(self, 'crest'):
            col = al[846:930, :] > .55; self.crest = cv2.GaussianBlur((846 + np.argmax(col, 0)).astype(f32)[None], (9, 1), 0)[0]
        lw = np.exp(-((self.x - cx) / (70 + 5.5 * d)) ** 2) * np.exp(-((self.y - self.crest[None] - 1.0) / (1.6 + 1.2 * np.exp(-((self.x - cx) / 60) ** 2))) ** 2)
        img = img + (np.array([1.16, 1.02, .74], f32) - img) * np.clip(lw * 1.15 * max(0., 1 - d / 34.), 0, 1)[..., None]
        rimw = np.exp(-((self.x - cx) / (190 + 260 * k)) ** 2) * np.exp(-((self.y - 884) / 26) ** 2)
        img = comp.screen(img, (edge * rimw * (1.0 - .35 * k))[..., None] * rgb('#fff2c8'))
        spill = np.exp(-(np.hypot((self.x - cx) / 2.2, self.y - 894) / 130) ** 1.5) * (.34 + .22 * k) * (1 - .9 * P['st'])
        img = to(img, '#ffd89a', spill * (1 - np.clip((s7a_sky.SUNR + 6 - np.hypot(self.x - cx, self.y - cy)) / 12, 0, 1) * (1 - P['al'])))
        img = comp.screen(img, (P['ray'] * (.05 + .13 * k) * (1 - .75 * P['st']))[..., None] * rgb('#ffe9b8'))
        # ------------------------------------------------ life
        tw = np.array([.74, .72, 1.0], f32) * (1 - k) + np.array([1.04, 1.0, .93], f32) * k
        cv_ = Canvas(img=np.ascontiguousarray(img, f32))
        wg = lambda x, t: gust(x, t, 1500., 22., .16) * 1.15
        self.iw.draw(cv_, f, 1.1, tw * (.82 + .12 * k), gfun=wg); self.gw.draw(cv_, f, 1.3, tw * (.9 + .1 * k), gfun=wg)
        self.gb.draw(cv_, f, 1.45, tw, gfun=wg)
        img = cv_.img
        img = self.foxshadow(img, f, k)
        img = foxact.draw(img, 's7a', f, MA)
        cv_ = Canvas(img=np.ascontiguousarray(img, f32))
        self.il.draw(cv_, f, 1.1, tw * .86, gfun=wg); self.gf.draw(cv_, f, 1.5, tw, gfun=wg)
        img = cv_.img
        img = self.pages(img, f, k)
        # warm air over everything near the light
        air = np.exp(-(np.hypot((self.x - cx) / 1.5, self.y - cy) / 600) ** 1.7) * (.05 + .13 * k) * (1 - np.clip((s7a_sky.SUNR + 6 - np.hypot(self.x - cx, self.y - cy)) / 12, 0, 1))
        img = to(img, '#ffd090', air)
        return zoomto(img, 1180, 800, 1 + .022 * f / 72.)

    def foxshadow(self, img, f, k):
        x, y, h = MA['x'], MA['y'], MA['h']
        a = float(np.clip((f // 2 * 2 - 34) / 10., 0, 1)) * .55
        c = Cel(int(h * 2.6), int(h * .5), int(x - h * 2.0), int(y - h * .12))
        c.fill(spl([(x + .36 * h, y - 3), (x - .1 * h, y - 6), (x - .5 * h, y - 2), (x - 1.2 * h, y + 5), (x - 1.75 * h, y + 12), (x - 1.2 * h, y + 12), (x - .5 * h, y + 7), (x, y + 5), (x + .3 * h, y + 3)], closed=True, corners=(4,)), '#7c6488', .30 + a)
        return over_roi(img, c.rgba(), c.ox, c.oy)

    def pages(self, img, f, k):
        d = f // 2
        c = Cel(W, H, ss=2)
        pal = dict(life.PAPER, lit='#fff6e0', trans='#fff0d2', sh='#c0948f', burn='#fffaea'); any_ = False
        if k < .99:
            tint = lambda h_: rgb(h_) * (np.array([.80, .78, .92], f32) * (1 - k) + k)
            pal = {n: tint(v) for n, v in pal.items()}
        proj = lambda P: (np.c_[P[:, 0], P[:, 2]], P[:, 1])
        for i, (f0, n, ctl, sz, ph) in enumerate(self.fl):
            t = (d * 2 - f0) / n
            if t < 0 or t > 1: continue
            any_ = True
            tt = t ** .92
            C = np.array(ctl, float); b = np.array([(1 - tt) ** 3, 3 * tt * (1 - tt) ** 2, 3 * tt * tt * (1 - tt), tt ** 3]); pos = b @ C
            v = (3 * (1 - tt) ** 2 * (C[1] - C[0]) + 6 * tt * (1 - tt) * (C[2] - C[1]) + 3 * tt ** 2 * (C[3] - C[2])) / n
            sp = np.hypot(*v); ang = math.atan2(v[1], v[0])
            q = d * 1.15 + ph
            R = rot(ang + .4 * math.sin(q * .7), 1.0 + .8 * math.sin(q), .5 * math.cos(q * 1.3 + ph))
            st = 1 + np.clip((sp - 18) / 34, 0, .7)                                           # smear along the flight
            ca, sa = math.cos(ang), math.sin(ang)
            SM = np.array([[ca, -sa], [sa, ca]]) @ np.diag([st, 1 / st ** .5]) @ np.array([[ca, sa], [-sa, ca]])
            R = R * sz
            R = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]) @ R if False else R
            Rw = np.stack([SM[0, 0] * R[0] + SM[0, 1] * R[2], R[1], SM[1, 0] * R[0] + SM[1, 1] * R[2]])
            paper(c, self.sheets[i], (pos[0], 0, pos[1]), Rw, proj, bend=1.5 * math.sin(q * .9 + 1), beta=.6 + .5 * math.sin(q * .35), curl=.8, seed=i * 31 + d, lw=1.5 * sz + .5, eye=(pos[0], -900, pos[1]), light=(.6, .75, -.3), boil=.4, detail=sz > .7, N=9)
            if sp > 34:
                nn = np.array([-sa, ca]); rr = np.random.default_rng(i * 97 + d)
                for j in range(5):
                    o = nn * rr.uniform(-16, 16) * sz; L = sp * rr.uniform(.9, 2.0)
                    p1 = pos - np.array([ca, sa]) * (22 * sz * st) + o; p0 = p1 - np.array([ca, sa]) * L
                    c.line(np.stack([p0 + (p1 - p0) * u for u in np.linspace(0, 1, 6)]) + rr.normal(0, .8, (6, 2)), rr.uniform(.8, 2.0) * sz, pal['lit'], (.7, .15), seed=j, alpha=rr.uniform(.25, .6))
        if any_: img = comp.over(img, c.layer())
        return img


# ========================================================================= 7b
MB = dict(x=930, y=1500, h=1150)


def plate_b():
    from shots.balc_sky import Tree, Mass, colour, marks, lay, sky_wash
    from shots.s7a_sky import pal
    r = np.random.default_rng(61)
    x, y = grid()
    VS = (1900, 250)
    d = np.hypot((x - 2000) / 1500, (y - 640) / 900)
    img = ramp_(np.clip(d + .05 * fbm(H, W, (420, 160), (1, .4), 4), 0, 1.25), [(0, '#ffdfa6'), (.28, '#fbcb96'), (.55, '#f2ceb2'), (.8, '#d3cfc9'), (1.0, '#a6c2cd'), (1.25, '#86b0c9')])
    img = sky_wash(img, r, 260, hw=(30, 90), ln=(300, 900), centre=(1700, 560), op=(.06, .2))
    cv_ = Canvas(img=img)
    # a few broad high streaks, upper left
    for k in range(120):
        t = r.random(); px = -100 + 1500 * t; py = 260 - 190 * t + r.normal(0, 30)
        cv_.strokes(arcs([px], [py], [-.13 + r.normal(0, .06)], [r.uniform(90, 320)], [r.normal(0, .2)]), [r.uniform(6, 20)], jit_color(rgb('#f6e4c4')[None], r, .004, .01, .02), [r.uniform(.12, .4)], [r.uniform(.2, .8)], seed=k, shape='round')
    img = cv_.img
    def tower(bl, seed, d, lean=0., floor=(1000, 200), lz=-.2, nm=500, sl=(.1, .85)):
        nonlocal img
        rr = np.random.default_rng(seed); T = Tree()
        for (bx, by, br) in bl: T.grow(bx, by, br * 1.08, br, rr, kids=(6,), lean=lean, rmin=4)
        m = Mass(T, sun=VS, seed=seed, lz=lz, rimw=12, floor=floor)
        sh, li, rc, _ = pal('B', d)
        c = np.ascontiguousarray(colour(m, sh, li, rc, sun_lo=sl[0], sun_hi=sl[1], rim=.8, seed=seed), f32)
        c = marks(m, c, rr, nm, size=(.04, .1), op=(.2, .6), val=.012); img = lay(img, m, c)
    tower([(1860, 600, 150), (2010, 520, 170), (1720, 760, 150), (1960, 760, 220), (1600, 880, 130), (1800, 920, 220), (2060, 960, 240)], 71, .40, -.3, (960, 220), -.25)
    tower([(80, 690, 170), (-60, 560, 190), (270, 800, 160), (430, 900, 130), (90, 900, 240), (330, 960, 200), (560, 1000, 120)], 72, .62, .35, (1010, 200), -.2)
    tower([(700, 1060, 140), (520, 1080, 160), (900, 1090, 130), (1120, 1070, 120), (1320, 1080, 150), (1540, 1060, 140), (250, 1100, 220), (1750, 1100, 200)], 73, .72, .0, (1150, 160), -.3)
    return dict(bg=img)


def ramp_(t, st):
    from lp.paint import ramp
    return ramp(t, st)


class B7:
    def setup(self):
        bg = cached('b', plate_b)['bg'].astype(f32)
        L = np.dstack([bg, np.ones((H, W), f32)])
        self.bg = comp.bokeh(L, 9.5, 1.0)[..., :3]
        r = np.random.default_rng(17)
        # three sizes, in clumps with gaps; nothing stands in front of the chest or the face
        self.bl = {0: [], 1: [], 2: []}; self.heads = []
        def clump(cl, x, n, h, w, spread, fan, seed_heads=0):
            for k in range(n):
                u = np.clip(r.normal(0, .5), -1.1, 1.1)
                self.bl[cl].append(dict(x=x + u * spread + r.normal(0, spread * .1), y=1190 + r.uniform(0, 40) + (60 if cl == 2 else 0), h=h * r.uniform(.42, 1.0) * (1 - .22 * abs(u)), a=-math.pi / 2 + u * fan + r.normal(0, .06), w=w * r.uniform(.6, 1.15),
                                        stiff=r.uniform(.85, 1.2), kink=(r.uniform(.5, .72), r.uniform(.5, 1.0)) if r.random() < .13 else None, c=int(r.integers(0, 4)), ph=r.uniform(0, 9), seed=len(self.bl[cl]) + cl * 400))
            for k in range(seed_heads):
                u = r.uniform(-.8, .8); self.heads.append(dict(x=x + u * spread, y=1200, h=h * r.uniform(1.0, 1.22), a=-math.pi / 2 + u * fan * .6, w=3.2, stiff=r.uniform(.95, 1.25), kink=None, c=0, ph=r.uniform(0, 9), seed=900 + len(self.heads)))
        # fine, far: thin tufts
        for (x, n, h, sp) in ((40, 13, 300, 46), (178, 8, 230, 26), (372, 15, 330, 44), (520, 8, 210, 24), (610, 5, 130, 16), (1285, 7, 170, 20), (1420, 14, 320, 40), (1566, 8, 230, 24), (1740, 17, 350, 52), (1905, 9, 250, 28), (2030, 12, 320, 40)): clump(0, x, n, h, 8.5, sp, .34, 1 if n > 12 else 0)
        # middle
        for (x, n, h, sp) in ((118, 8, 470, 44), (300, 6, 400, 30), (470, 5, 300, 26), (1375, 5, 340, 26), (1500, 8, 470, 42), (1690, 6, 420, 30), (1860, 9, 540, 46)): clump(1, x, n, h, 16, sp, .30, 1 if n > 6 else 0)
        # nearest, out of focus, only in the corners
        for (x, n, h, sp) in ((-40, 6, 640, 60), (2075, 7, 760, 70), (1960, 2, 420, 30)): clump(2, x, n, h, 40, sp, .26)
        self.pal = {0: [('#b9a95a', '#8d8450'), ('#c8b665', '#9a8c56'), ('#a69c55', '#7f7b4e'), ('#d1bf6c', '#a1935b')],
                    1: [('#b4a552', '#807a49'), ('#c5b25c', '#8f8450'), ('#a39950', '#737148'), ('#d0bb66', '#998b55')],
                    2: [('#8f8850', '#66684a'), ('#9c9152', '#72704b'), ('#827f4d', '#5d624a'), ('#a59957', '#7b764c')]}
        # torn paper streaks: start frame, dur, y0, y1, size, design.  None crosses the face.
        self.scr = [(0, 8, -80, 1010, 2150, 760, 92, 1.), (2, 9, -120, 330, 2160, -10, 62, 2.), (6, 8, -60, 1180, 2150, 820, 128, 3.), (10, 9, -100, 210, 2160, -70, 48, 4.), (13, 8, -60, 1100, 2140, 800, 100, 5.), (17, 7, -100, 300, 2150, 30, 74, 6.), (4, 10, -100, 120, 2200, -60, 40, 7.), (15, 9, -100, 1180, 2200, 900, 60, 8.)]
        rr = np.random.default_rng(3)
        self.shapes = []
        for k in range(len(self.scr)):
            sh = Sheet(1., rr.uniform(.6, .95), 500 + k, 'text' if k % 2 else 'letter', None)
            n = 34; a = np.linspace(0, 2 * np.pi, n, endpoint=False); pw = rr.uniform(2.5, 5)
            q = (np.abs(np.cos(a)) ** pw + np.abs(np.sin(a)) ** pw) ** (-1 / pw)                              # rough sheet, two torn sides
            jag = np.where(np.sin(a + rr.uniform(0, 6)) > .1, rr.uniform(-.16, .06, n), rr.normal(0, .012, n))
            sh.E = np.c_[np.cos(a), np.sin(a)] * (q * .5 * (1 + jag))[:, None] * (1., sh.h); sh.creases = sh.creases[:1]
            self.shapes.append(sh)
        return self

    def g(self, x, f):
        """one gust after another, crossing the frame left to right; every blade answers the same wave"""
        w = .5 + .5 * np.sin(2 * np.pi * (f / 11.0 - np.asarray(x, float) / 2400.))
        return .20 + .92 * w ** 1.5

    def curve(self, b, fd, lag, K=11):
        """base answers first, the tip follows through late; heavier blades move less and later"""
        s = np.linspace(0, 1, K)
        ga = self.g(b['x'], fd - lag * s)
        th = b['a'] + (ga * .40 * s ** .5 + ga * 1.12 * s ** 1.9) / b['stiff'] + .05 * np.sin(fd * 1.5 + b['ph'] * 5 - s * 6) * s * s
        if b['kink']: th = th + b['kink'][1] * smooth(b['kink'][0] - .03, b['kink'][0] + .05, s)
        st = b['h'] / (K - 1)
        P = np.c_[b['x'] + np.r_[0, np.cumsum(np.cos(th[:-1]) * st)], b['y'] + np.r_[0, np.cumsum(np.sin(th[:-1]) * st)]]
        return P + np.random.default_rng(b['seed'] * 100 + fd).normal(0, .5, P.shape)

    def grass(self, f, cl):
        fd = f // 2 * 2
        c = Cel(W, 1000, 0, H - 1000, ss=2 if cl < 2 else 1)
        for b in sorted(self.bl[cl], key=lambda q: -q['h']):
            P = life.resample(spl(self.curve(b, fd, (3.0, 4.0, 5.5)[cl])), 7.); n = len(P)
            if n < 5: continue
            T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]; t = np.linspace(0, 1, n)
            r = np.random.default_rng(b['seed'] + 5)
            hw = b['w'] * .5 * np.clip((1 - t) / .82, 0, 1) ** .8 * (.55 + .45 * np.clip(t / .3, 0, 1)) * (1 + .06 * life.noise1(n, 14, r)) + .2
            L, R = P + N * hw[:, None], P - N * hw[:, None]                     # L lies to the right of a rising blade: the sunny side
            fl, sh = self.pal[cl][b['c']]
            c.fill(np.vstack([L, R[::-1]]), fl)
            k0, k1 = int(n * r.uniform(0, .04)), int(n * r.uniform(.66, .86)); e = P + N * (hw * r.uniform(-.12, .14) * np.linspace(1, .2, n))[:, None]
            if k1 - k0 > 3: c.fill(np.vstack([e[k0:k1], R[k0:k1][::-1]]), sh)
            kt = int(n * r.uniform(.74, .86))                                    # the tip catches the sun
            c.fill(np.vstack([L[kt:], R[kt:][::-1]]), '#fbe8a6' if cl < 2 else '#e9d68f')
            if cl < 2:                                                           # thin ink, lifted here and there
                lw = (.9, 1.25)[cl]; q = sorted(r.uniform(.05, .95, 4))
                for a_, b_ in ((.02, q[0] * .8 + .1), (q[1], q[2] + .08), (q[3], .99)):
                    i0, i1 = int(n * a_), int(n * min(1, b_))
                    if i1 - i0 > 4: c.line(R[i0:i1], lw, '#655138', (.25, .4), press=.4, seed=b['seed'] + i0, tremor=.2, alpha=.9)
                i0, i1 = int(n * r.uniform(0, .3)), int(n * r.uniform(.4, .7))
                if i1 - i0 > 4 and r.random() < .6: c.line(L[i0:i1], lw * .7, '#7a6442', (.3, .5), press=.4, seed=b['seed'] + 7, tremor=.2, alpha=.7)
        if cl == 1:
            for b in self.heads:
                P = life.resample(spl(self.curve(b, fd, 5.0)), 6.); n = len(P)
                c.line(P, 2.6, '#a6955a', (.02, .2), press=.15, seed=b['seed'])
                T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; r = np.random.default_rng(b['seed'])
                for j in range(13):
                    i = int(n * (.80 + .197 * j / 12.)); e = T[i]; nn = np.array([-e[1], e[0]]) * (1 if j % 2 else -1); g_ = 8.5 * (1 - .45 * j / 12.)
                    p0 = P[i]; p1 = p0 + (e * 1.5 + nn * .62) * g_; m0 = (p0 + p1) / 2; w_ = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / (np.hypot(*(p1 - p0)) + 1e-6) * g_ * .36
                    gr = spl([p0, m0 + w_, p1, m0 - w_], closed=True, corners=(0, 2)); c.fill(gr, '#e9cd84' if j % 2 else '#c9a862'); c.line(np.vstack([gr, gr[:1]]), .8, '#7a5f3c', (.1, .1), seed=j)
        L = c.layer()
        if cl == 2: L = comp.bokeh(L, 13.5, 1.2)
        return L

    def scraps(self, f):
        d = f // 2; fd = d * 2
        c = Cel(W, H, ss=2)
        for k, (f0, n, x0, y0, x1, y1, sz, sd) in enumerate(self.scr):
            t = (fd - f0) / n
            if t < -.1 or t > 1.1: continue
            p = np.array([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t ** .8 + 40 * math.sin(t * 3 + sd)])
            v = np.array([(x1 - x0) / n, (y1 - y0) / n * .9]); sp = np.hypot(*v); e = v / sp; nn = np.array([-e[1], e[0]])
            a = d * 1.25 + sd; ang = math.atan2(v[1], v[0]); sq = .5
            R = rot(a * .6, .9 + .9 * math.sin(a), .8 * math.cos(a * .8 + sd)) * sz
            ca, sa = math.cos(ang), math.sin(ang); SM = np.array([[ca, -sa], [sa, ca]]) @ np.diag([1.75, .86]) @ np.array([[ca, sa], [-sa, ca]])
            Rw = np.stack([SM[0, 0] * R[0] + SM[0, 1] * R[2], R[1], SM[1, 0] * R[0] + SM[1, 1] * R[2]])
            paper(c, self.shapes[k], (p[0], 0, p[1]), Rw, lambda P: (np.c_[P[:, 0], P[:, 2]], P[:, 1]), bend=2.0 * math.sin(a + 1), beta=.5 + sd, curl=1.2, seed=k * 40 + d, lw=2.0, eye=(p[0], -900, p[1]), light=(.7, .6, .1), boil=.6, N=11, pal=dict(life.PAPER, sh='#d0a492', trans='#f0d3a4', lit='#fbefd0'))
            rr = np.random.default_rng(k * 77 + d)
            for j in range(6):
                o = nn * rr.uniform(-.7, .7) * sz * sq; L = sp * rr.uniform(.8, 1.9); q1 = p - e * sz * .8 + o; q0 = q1 - e * L
                c.line(np.stack([q0 + (q1 - q0) * u for u in np.linspace(0, 1, 6)]) + rr.normal(0, 1.2, (6, 2)), rr.uniform(1.4, 4.2), '#fdeecb', (.75, .1), seed=j, alpha=rr.uniform(.25, .6))
        return c.layer()

    def frame(self, f):
        z = 1.03 + .02 * f / 22.
        img = zoomto(self.bg, 1500, 600, z, 2.4 * f * .5, 0)
        x, y = grid()
        img = img + (rgb('#ffd69a') - img) * (np.exp(-(np.hypot((x - 2080) / 900, (y - 600) / 620)) ** 1.6) * .45)[..., None]
        img = foxact.draw(img, 's7b', 72 + f, MB)
        img = comp.over(img, self.scraps(f))
        img = comp.over(img, self.grass(f, 0))
        img = comp.over(img, self.grass(f, 1))
        img = comp.over(img, self.grass(f, 2))
        # transmitted light from the right
        g = np.exp(-(np.hypot((x - 2120) / 760, (y - 620) / 700)) ** 1.5) * .30
        return img + (rgb('#ffd494') - img) * g[..., None]


# ========================================================================= 7c
def plate_c():
    from shots import s7c
    ring = s7c.Ring()
    out = dict(sea=s7c.sea_c(), sw=s7c.silk(), tr=s7c.trough())
    try:
        sp, ax = s7c.tower_sprite(); out['tw'] = sp; out['tx'] = np.array([ax])
    except Exception as e:
        print('s7c: own tower cel used:', e)
    for d in range(13):
        L = s7c.crest_layer(ring, d * 2)
        out['cr%d' % d] = L
    return out


class C7:
    def setup(self):
        from shots import s7c
        self.m = s7c
        P = cached('c', plate_c)
        self.sea, self.sw, self.tro = np.array(P['sea'], f32), np.array(P['sw'], f32), np.array(P['tr'], f32)
        self.cr = [P['cr%d' % d] for d in range(13)]
        self.ring = s7c.Ring()
        X, Z = s7c.plane()
        self.X, self.Z = X, Z
        self.th = np.arctan2(X, 1 - Z); self.rr = np.hypot(X, Z - 1)
        self.n = fbm(H, W, (60, 22, 8), (1, .6, .4), 77, aspect=1.6)
        self.n2 = fbm(H, W, (150, 60, 26), (1, .55, .3), 78, aspect=2.4)
        if 'tw' in P:
            sp = P['tw'].astype(f32); self.tw = (sp, int(round(s7c.TX - float(P['tx'][0]))), int(s7c.TY + 34 - sp.shape[0]))
        else: self.tw = s7c.tower_cel()
        self.x, self.y = grid()
        return self

    def frame(self, f):
        m = self.m; d = f // 2; R = self.ring.at(d * 2)
        rag = R * self.ring.rag(self.th, R)
        s = (self.rr - rag) / rag                                                # 0 at the crest, <0 inside
        below = self.y > m.HZ + 3
        sea = self.sea                                                           # never warped: the wall is new paint over an untouched plate
        k = smooth(.015, -.03, s + self.n * .012) * below
        k = k[..., None].astype(f32)
        # behind the crest a narrow hollow (about a sixth of the radius), then the sea lying smooth again
        wd = .165 * (1 + .22 * np.sin(3 * self.th + 1.1) + .12 * np.sin(8 * self.th + 2))
        tm = smooth(-1.12, -.42, (s + .020 * self.n2) / wd)[..., None]
        sw = self.sw
        # two or three soft swells follow the ring out; their slopes take the low sun from the right
        dot = (self.X * .66 + (self.Z - 1) * .75) / np.maximum(self.rr, 1e-3)
        sl = np.zeros((H, W), f32); up = np.zeros((H, W), f32)
        for j, (c, wj, am) in enumerate(((.70, .060, 1.0), (.49, .058, .72), (.30, .052, .5))):
            rc = R * c * (1 + .035 * np.sin(3 * self.th + j * 2.1) + .02 * np.sin(7 * self.th - j)) + .006 * self.n
            q = (self.rr - rc) / (R * wj * (1 + .25 * np.sin(2 * self.th + j)))
            e = np.exp(-q * q / 2) * am * (.72 + .28 * np.sin(4 * self.th + j * 1.7 + 1)); sl += q * e * dot * 1.5; up += e
        sl = np.where(below, sl, 0)
        sw = sw + (rgb('#fff1d8') - sw) * (np.clip(sl, 0, 1) * .66 + np.clip(up, 0, 1) * .12)[..., None]
        sw = sw + (rgb('#b48fae') - sw) * (np.clip(-sl, 0, 1) * .58)[..., None]
        sw = sw * (1 - tm) + self.tro * tm
        img = sea * (1 - k) + sw * k
        # the foot of the wall: violet shade on the sea just outside it
        ft = np.exp(-np.maximum(s, 0) / .11) * smooth(-.01, .02, s) * below * (.55 + .45 * np.clip((self.y - m.TY) / 120, -1, 1))
        img = img * (1 - (ft * .30)[..., None] * np.array([.85, 1., .55], f32))
        # mist lying along the wall's near foot
        fm = np.exp(-((s - .075 + .02 * self.n2) / .075) ** 2) * smooth(-.05, .45, np.cos(self.th)) * (self.y > m.TY) * .82
        img = img + (mix(rgb('#dcbbb4'), rgb('#bd9db4'), .5) - img) * fm[..., None]
        L = np.array(self.cr[d], f32)
        far = (smooth(m.TY + 4, m.TY - 10, self.y))[..., None]                       # the far arc rolls away behind the tower
        img = comp.over(img, L * far)
        img = self.tower(img, d)
        L = L * (1 - far)
        # soft contact under the crest
        sh = cv2.GaussianBlur(L[..., 3], (0, 0), 14); sh = np.roll(sh, 10, 0)
        img = img * (1 - (sh * .16 * (1 - L[..., 3]))[..., None] * np.array([.8, 1., .7], f32))
        img = comp.over(img, L)
        img = comp.over(img, m.arcs_of_mist(self.ring, f, R))
        # low gold air
        g = np.exp(-(np.hypot((self.x - m.SUN[0]) / 1.7, self.y - m.SUN[1]) / 430) ** 1.5) * .5
        img = img + (rgb('#ffcf90') - img) * g[..., None]
        rr = np.hypot(self.x - m.SUN[0], self.y - m.SUN[1])
        a = np.clip(46 - rr + .5, 0, 1) * (self.y < m.HZ + 2)
        img = img + (np.array([1.2, 1.1, .88], f32) - img) * a[..., None]
        img = img + (rgb('#fff0c4') - img) * (np.exp(-(np.maximum(0, rr - 40) / 36) ** 1.2) * .8 * (1 - a))[..., None]
        return zoomto(img, m.TX + 260, 560, 1.045 - .03 * f / 26.)

    def tower(self, img, d):
        c, ox, oy = self.tw
        img = over_roi(img.copy(), c, ox, oy)
        # mist round its feet
        m = self.m
        n = np.exp(-(np.hypot((self.x - m.TX + 22) / 4.6, self.y - (m.TY + 34)) / 27) ** 2) * (1.0 + .22 * self.n)
        return img + (rgb('#f0d2ba') - img) * np.clip(n, 0, 1)[..., None]

class Shot:
    id = 's7'; n = 120
    finish = dict(hal_amt=.24, vig=.10)
    FOX = {'s7a': MA, 's7b': dict(x=930, y=1500, h=1150)}

    def setup(self):
        for n, fn in (('a', plate_a), ('b', plate_b), ('c', plate_c)): cached(n, fn, keys_only=True)
        self.cur = (None, None)

    def part(self, k):
        if self.cur[0] != k: self.cur = (k, {'a': A7, 'b': B7, 'c': C7}[k]().setup())     # one cut in memory at a time
        return self.cur[1]

    def frame(self, f):
        if f < 72: return self.part('a').frame(f)
        if f < 94: return self.part('b').frame(f - 72)
        return self.part('c').frame(f - 94)
