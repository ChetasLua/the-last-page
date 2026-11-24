"""S1  Before Dawn.  A sea of cloud in the last hour of night, the library tower on its cliff, one small light."""
import numpy as np, cv2, math, os
from lp import W, H, comp, clouds, plants
from lp.paint import f32, rgb, mix, ramp, smooth, fbm, grid, repaint, mottle, flat_dabs, Canvas, arcs, jit_color, lum
from lp.ink import spl, Cel
from lp.comp import ease, over

VER = 'b4'
HZ = 690.
MOON = (-.74, -.62, .20)

FAR = clouds.tones('#b9b4d8', '#a09bcb', '#8489bf', '#6c73b0')
MID = clouds.tones('#c0c7e6', '#979ccf', '#6a73b4', '#495394')
NEAR = clouds.tones('#97a1d2', '#6570b2', '#46518f', '#2c3570')
MIST_F, MIST_M, MIST_N = rgb('#8f90c6'), rgb('#6c74b3'), rgb('#424d8e')


# ================================================================= plates
def sky():
    x, y = grid()
    t = y / HZ
    base = ramp(np.clip(t, 0, 1), [(0, '#191f43'), (.16, '#1f2850'), (.40, '#2e3869'), (.66, '#485389'), (.84, '#6c70a6'), (.94, '#8c87b6'), (1, '#a596be')])
    # moon wash from the upper left, rose lying low on the left horizon
    g = comp.radial(-160, 130, 1500, 900, 1.7)[..., None]
    base = base + g * np.array([.030, .040, .062], f32)
    rose = np.exp(-((x - 560) / 620) ** 2) * np.exp(-np.maximum(0, (HZ - y)) / 62.)
    base = base + rose[..., None] * (rgb('#d9a8b6') - base) * .62
    # thin high veil behind the tower, so its dark side keeps its edge
    veil = np.exp(-((x - 1500) / 430) ** 2 - ((y - 300) / 250) ** 2) * (1 + .18 * fbm(H, W, (150, 50), (1, .5), 14, aspect=3.))
    base = base + veil[..., None] * np.array([.040, .046, .062], f32)
    # flat-brush sky, strokes nearly level
    ang = fbm(H, W, (260, 90), (1, .4), 3) * .07 - .03
    out = repaint(base, (58, 26), seed=4, angle=ang, thr=(0, .004), density=.9, aspect=6.5, jit=(.001, .004, .008), op=.55, dry=.18, curve=.12, bristle=.02)
    out = mottle(out, .007, 5, .002)
    return out


def stars(img):
    r = np.random.default_rng(12)
    pts = [(188, 128, 2.0), (402, 70, 1.3), (318, 236, 1.5), (610, 182, 1.8), (742, 88, 1.2), (96, 300, 1.2), (520, 330, 1.1), (868, 226, 1.4), (950, 70, 1.5), (250, 400, 1.0), (1560, 96, 1.3),
           (1718, 210, 1.6), (1880, 84, 1.2), (1640, 316, 1.0), (700, 300, .9), (1960, 250, 1.0), (1052, 168, .9), (430, 160, .9), (150, 210, .8), (830, 360, .8), (1450, 60, .9)]
    c = Cel(W, 480, 0, 0, ss=2)
    for (x, y, s) in pts:
        col = mix(rgb('#e3e8f8'), rgb('#f4e4da'), r.random() * .6)
        n = 7; a = np.linspace(0, 2 * np.pi, n, endpoint=False) + r.uniform(0, 1)
        q = np.c_[x + np.cos(a) * s * r.uniform(.7, 1.1, n), y + np.sin(a) * s * r.uniform(.7, 1.1, n)]
        c.fill(spl(q, closed=True, n=5), col, min(1., .48 + .26 * s))
    img = over(img, c.layer())
    for (x, y, s) in pts:
        if s > 1.35: img = img + comp.radial(x, y, 10 * s, power=2.6)[..., None] * np.array([.10, .115, .15], f32) * s * .5
    return img


def high_cloud(img):
    """long dry-brushed cirrus, barely there"""
    r = np.random.default_rng(8)
    cv = Canvas(img=img)
    for (x, y, L, w, a, c, op) in ((330, 446, 620, 7, -.035, '#6a70a8', .30), (620, 470, 420, 5, -.03, '#757aae', .28), (150, 498, 380, 6, -.02, '#7b7db0', .30), (1730, 420, 520, 6, -.03, '#5d659d', .26), (1900, 452, 360, 5, -.02, '#656ba2', .26),
                                  (820, 548, 460, 5, -.012, '#9089b6', .34), (420, 574, 560, 6, -.01, '#9c8fb8', .36), (1640, 560, 520, 5, -.012, '#8580b0', .32), (60, 604, 420, 5, -.005, '#b29cbe', .38), (760, 618, 620, 5, -.004, '#b79ebf', .40)):
        for k in range(9):
            K = 7; t = np.linspace(-.5, .5, K)
            l = L * r.uniform(.35, .9); dx = r.uniform(-.4, .4) * L * .5; dy = r.normal(0, w * .9)
            P = np.stack([x + dx + t * l, y + dy + a * t * l + np.cumsum(r.normal(0, .5, K))], -1)[None]
            cv.strokes(P, w * r.uniform(.25, .8), np.clip(rgb(c) * (1 + r.normal(0, .03)), 0, 1)[None], op=op * r.uniform(.4, 1), dry=.35, seed=k + int(x), shape='round', bristle=.04)
    return cv.img


def back_plate():
    img = high_cloud(sky())
    S = clouds.Sky(H, W, img, light=MOON, seed=1)
    S.deck(HZ, ('#b3a8cd', '#a29cca', '#9492c5'), ('#5d68ab', '#485396', '#363f7d'), seed=5, scale=1.7, amt=.56, cx=1180)
    S.strata(HZ - 34, HZ + 10, 34, ['#c6a5bf', '#b59fc4', '#a597c5', '#cfa9ba'], seed=6, w=(.05, .26), op=(.25, .65))
    # range on the horizon
    r = np.random.default_rng(4)
    def B(x, base, w, h, T, m, sd, **k): S.bank(x, base, w, h, T, m, seed=sd, **k)
    for (x, b, w, h) in ((60, 698, 170, 13), (250, 697, 120, 8), (470, 699, 210, 12), (690, 698, 160, 9), (905, 700, 230, 15), (1480, 699, 180, 11), (1660, 698, 150, 9), (1850, 700, 220, 16), (2010, 699, 120, 10)):
        B(x, b, w, h, FAR, '#a39dca', int(x), tiers=2, levels=1, n=5, pool=.3)
    # cloud towers that break the horizon
    B(300, 712, 460, 190, FAR, '#9a96c8', 24, profile=clouds.peak(.40, .17, .22, .70, .58, .14), n=7, levels=3, lit=.26, half=.30, contrast=1.15)
    B(620, 712, 230, 46, FAR, '#9a96c8', 31, n=5, levels=2)
    B(1812, 710, 340, 112, FAR * .985, '#9895c8', 27, profile=clouds.peak(.62, .2, .25, .2, .5, .13), n=6, levels=3, lit=.2)
    B(1010, 712, 250, 54, FAR, '#9995c8', 41, profile=clouds.peak(.35, .25, .3), n=5, levels=2)
    # rows stepping toward us
    T1 = FAR + (MID - FAR) * .35; m1 = MIST_F
    for (x, b, w, h, sd) in ((90, 727, 330, 34, 3), (520, 722, 260, 22, 6), (760, 730, 300, 30, 7), (1600, 728, 330, 34, 8), (1960, 726, 280, 30, 9), (1290, 724, 280, 24, 10)):
        B(x, b, w, h, T1, m1, sd + 60, n=6, levels=3, pool=.45)
    T2 = FAR + (MID - FAR) * .75; m2 = MIST_F + (MIST_M - MIST_F) * .5
    for (x, b, w, h, sd) in ((-40, 770, 460, 72, 3), (330, 752, 300, 34, 9), (1000, 762, 380, 52, 5), (1780, 772, 520, 78, 6)):
        B(x, b, w, h, T2, m2, sd + 80, n=6, levels=3, pool=.5)
    # the calm lake on the left stays empty; one small island in it
    S.mist(806, 560, 900, 26, '#8f92cb', .45, 3)
    B(742, 806, 210, 30, MID * .985, m2, 93, n=5, levels=3, pool=.55)
    T3 = MID; m3 = MIST_M
    B(1890, 872, 640, 120, T3, m3, 101, profile=clouds.peak(.35, .22, .3, .75, .6, .15), n=6, pool=.55)
    B(150, 876, 620, 128, T3, m3, 103, profile=clouds.peak(.30, .2, .28, .68, .62, .14), n=6, pool=.55)
    B(1000, 880, 420, 84, T3, m3, 108, profile=clouds.peak(.6, .24, .3), n=6, pool=.5)
    out = S.paint(5)
    # the far clouds near the rose take a little of it in their shadows
    x, y = grid()
    m = np.exp(-((x - 520) / 560) ** 2) * smooth(745, 690, y) * smooth(560, 640, y) * np.clip(S.cov, 0, 1)
    out = out + (rgb('#c9a0b8') - out) * (m * .30 * smooth(.80, .55, lum(out)))[..., None]
    return stars(out)


def front_plate(under):
    """clouds in front of the cliff foot, then the great soft masses under us: layered"""
    S = clouds.Sky(H, W, under, light=MOON, seed=2, layered=True)
    T3 = MID + (NEAR - MID) * .20; m3 = MIST_M + (MIST_N - MIST_M) * .25
    S.mist(952, 1370, 1000, 34, m3, .75, 2)
    S.bank(1395, 948, 430, 86, T3, m3, seed=203, profile=clouds.peak(.30, .18, .32, .72, .7, .16), n=6, pool=.5)
    S.bank(1120, 966, 520, 112, T3, m3, seed=201, profile=clouds.peak(.40, .2, .3, .80, .5, .12), n=6, pool=.6)
    S.bank(1640, 984, 620, 128, T3, m3, seed=204, profile=clouds.peak(.58, .22, .3, .2, .5, .14), n=6, pool=.6)
    S.mist(994, 1300, 1300, 36, m3, .6, 4)
    T4 = MID + (NEAR - MID) * .80; m4 = MIST_N + (MIST_M - MIST_N) * .30
    S.bank(640, 1020, 800, 138, T4, m4, seed=207, profile=clouds.peak(.28, .2, .25, .7, .6, .16), n=7, pool=.55, detail=.9)
    S.bank(1970, 1070, 720, 180, T4, m4, seed=210, profile=clouds.peak(.45, .25, .3), n=6, pool=.5, detail=.9, soft=.5)
    S.mist(1062, 900, 1800, 56, m4, .6, 6)
    # the great soft masses right under us: few lobes, low contrast, dark
    T5 = NEAR; m5 = MIST_N * .94
    S.bank(1150, 1196, 1500, 190, T5, m5, seed=245, profile=clouds.peak(.30, .17, .38, .70, .74, .18), n=9, levels=3, detail=.75, soft=1.3, ss=1, pool=.35, lit=.16, half=.30)
    S.bank(200, 1204, 1250, 256, T5 * .93, m5 * .95, seed=238, profile=clouds.peak(.36, .24, .34, .80, .62, .16), n=8, levels=3, detail=.7, soft=1.6, ss=1, pool=.3, lit=.15, half=.30)
    S.bank(1960, 1240, 980, 190, T5 * .88, m5 * .92, seed=239, n=6, levels=3, detail=.6, soft=1.8, ss=1, pool=.3, lit=.14, half=.28)
    S.mist(1160, 1024, 2300, 74, m5 * .86, .62, 8)
    out = S.paint(9, sizes=(15, 7, 3.6))
    # the bottom of the frame sinks into night
    y = np.arange(H, dtype=f32)[:, None]
    k = smooth(960, 1160, y + 24 * fbm(H, W, (220, 80), (1, .4), 3, aspect=2.5))[..., None]
    out = out * (1 - .24 * k) + rgb('#232a5c') * .10 * k
    return S.layer(out)


class Shot:
    id = 's1'; n = 168
    finish = dict(hal_amt=.22, vig=.12)
    CX, CY = 1200., 560.                                          # the push closes on the tower
    LAMP = [(84, 22), (99, 22), (114, 22), (129, 23)]             # (first frame, length) the light spends behind each window

    # ------------------------------------------------------------------ setup
    def setup(self):
        from shots import s1_tower as tw, s1_cliff as cl
        os.makedirs('cache', exist_ok=True)
        key = lambda k: f'cache/s1_{k}_{VER}.npy'
        def cached(k, fn):
            p = key(k)
            if os.path.exists(p): return np.load(p).astype(f32)
            a = fn(); tmp = p + f'.{os.getpid()}.npy'; np.save(tmp, a.astype(np.float16)); os.replace(tmp, p)
            return a.astype(f32)
        self.back = cached('back', back_plate)
        tower = tw.paint_tower()
        def mid():
            L = cl.cliff(); top = [tw.ivy_layer(tower), tw.balcony_ivy(), cl.dressing().layer(), cl.pines().layer()]
            L = over4(L, tw.gouache(tower)); occ = np.zeros((H, W), f32)
            for l in top: L = over4(L, l); occ = np.maximum(occ, l[..., 3])
            return np.dstack([L, occ])
        m = cached('mid', mid)
        self.mid, occ = m[..., :4].copy(), m[..., 4].copy()
        self.front = cached('front', lambda: front_plate(over(self.back, self.mid)))
        self.win = [self._window(w, occ) for w in sorted(tower.windows, key=lambda w: w['key'])]
        del tower
        self.grass = self._grass()
        self.delay = self._fade_map()
        self.wisps = [self._wisp(360, 34, 3, '#a3a3d4', .9), self._wisp(330, 40, 7, '#8f95ce', 1.0), self._wisp(260, 30, 11, '#aeaada', .9)]

    def _window(self, w, occ):
        """masks of one lit window in a small ROI of the tower plate"""
        o = w['outer']; pad = 46
        x0, y0 = np.floor(o.min(0)).astype(int) - pad; x1, y1 = np.ceil(o.max(0)).astype(int) + pad
        h, wd = y1 - y0, x1 - x0
        def mask(*polys, lines=()):
            c = Cel(wd, h, x0, y0, ss=2)
            for q in polys: c.fill(q, '#ffffff')
            for (P, lw) in lines: c.line(P, lw, '#ffffff', (0, 0), .2)
            return c.rgba()[..., 3]
        inner = w['inner']
        xi0, xi1 = inner[:, 0].min(), inner[:, 0].max(); yi0, yi1 = inner[:, 1].min(), inner[:, 1].max()
        r = np.random.default_rng(40 + w['key'])
        g = mask(inner)
        xm = (xi0 + xi1) / 2 + r.normal(0, .3)
        bars = [([[xm, yi0 - 3], [xm + .5, (yi0 + yi1) / 2], [xm, yi1 + 3]], 2.0)]
        for j in range(1, 5):
            yy = yi0 + (yi1 - yi0) * j / 5
            bars.append(([[xi0 - 2, yy], [xm, yy + r.normal(0, .6)], [xi1 + 2, yy]], 1.5))
        g = g * (1 - .92 * mask(lines=bars))
        # books left on the inner sill, dark against the light
        bk = []; x = xi0 + 2
        while x < xi1 - 4:
            bw = r.uniform(2.4, 4.6); bh = r.uniform(5, 11)
            if r.random() < .68: bk.append(np.array([[x, yi1 + 2], [x + r.normal(0, .6), yi1 - bh], [x + bw, yi1 - bh + r.normal(0, .5)], [x + bw, yi1 + 2]]))
            x += bw + .5
        if bk: g = g * (1 - .88 * mask(*bk))
        # old uneven panes
        yy, xx = np.mgrid[0:h, 0:wd].astype(f32); xx += x0; yy += y0
        pane = r.uniform(.74, 1.08, (5, 2))
        pj = np.clip(((yy - yi0) / (yi1 - yi0) * 5).astype(int), 0, 4); pi_ = (xx > xm).astype(int)
        g = g * cv2.GaussianBlur(pane[pj, pi_].astype(f32), (0, 0), .7)
        free = 1 - occ[y0:y1, x0:x1]
        om = mask(o); rev = np.clip(om - mask(inner), 0, 1)
        ring = np.clip(cv2.GaussianBlur(om, (0, 0), 7) * 1.6, 0, 1) * (1 - om)
        sill = mask(w['sill']) if w['sill'] is not None else om * 0
        return dict(box=(x0, y0, x1, y1), g=g * free, rev=rev * free, ring=ring, sill=sill * free, xx=xx, yy=yy, xi=(xi0, xi1), yi=(yi0, yi1), th=w['th'], leaf=occ[y0:y1, x0:x1] * np.clip(cv2.GaussianBlur(om, (0, 0), 4) * 1.3, 0, 1))

    def _grass(self):
        r = np.random.default_rng(17)
        def clump(n, xs, hs, lean, sz, y=(1178, 1225), hf=None):
            st = []
            for k in range(n):
                x = r.uniform(*xs); h = r.uniform(*hs) * (hf(x) if hf else 1)
                st.append((x, r.uniform(*y), h, r.uniform(*lean), r.uniform(*sz)))
            return sorted(st, key=lambda s: -s[2])
        left = plants.Pampas(clump(19, (-30, 372), (250, 560), (.03, .30), (.75, 1.1), hf=lambda x: 1.12 - .0016 * x), seed=4, leaves=5)
        right = plants.Pampas(clump(15, (1770, 2070), (190, 430), (.02, .30), (.7, 1.0), hf=lambda x: .62 + .0016 * (x - 1770)), seed=9, leaves=5)
        near = plants.Pampas([(-34, 1250, 830, .13, 1.9), (44, 1260, 640, .20, 1.75), (-8, 1260, 500, .30, 1.6), (116, 1270, 420, .34, 1.5)], seed=6, hairs=110, leaves=4,
                             pal=('#15132e', '#1d1a3c', '#2e2b58', '#50508a', '#8a90c2'))
        self.under = [self._undergrowth(0, 520, 21, lambda x: 150 * math.exp(-(max(x, 0) / 230.) ** 1.6) + 26), self._undergrowth(1640, W, 22, lambda x: 110 * math.exp(-(max(W - x, 0) / 190.) ** 1.6) + 18)]
        out = []
        for P, t_jit, stiff in ((left, 3., 1.), (right, 3., 1.1), (near, 2., .85)):
            x = P.st[:, 0]; ph = r.uniform(0, 6.28, P.n); tj = r.normal(0, t_jit, P.n)
            def force(t, x=x, ph=ph, tj=tj):
                q = t + tj
                base = .035 * np.sin(q * .085 + ph) + .02 * np.sin(q * .19 + ph * 2.3)
                g1 = .46 * np.exp(-((q - 58 - x / 27.) / 13.) ** 2)            # the gust that crosses the frame, left to right
                g2 = .20 * np.exp(-((q - 24 - x / 27.) / 9.) ** 2) + .24 * np.exp(-((q - 112 - x / 27.) / 16.) ** 2)
                return base + g1 + g2
            a = plants.springs(self.n + 4, force, k=.026 / stiff, c=.115)
            b = plants.springs(self.n + 4, force, k=.012, c=.10)               # plume hair trails
            out.append((P, a, b))
        return out

    def _undergrowth(self, x0, x1, seed, hf):
        """the dark bank of grass the stalks stand in: hundreds of blades, violet, few lit edges"""
        r = np.random.default_rng(seed); cv = Canvas(330, x1 - x0, layer=True)
        for lay in range(4):
            n = (260, 240, 170, 60)[lay]
            x = r.uniform(x0 - 20, x1 + 20, n); h = np.array([hf(v) for v in x]) * r.uniform(.45, 1.25, n) * (1.0, .95, .85, .9)[lay]
            keep = (h > 12) & (np.array([hf(v) for v in x]) > hf(x0 if x0 > 0 else x1) + 4); x, h = x[keep], h[keep]; n = len(x)
            A = -np.pi / 2 + .22 + r.normal(0, .34, n); ln = h * 1.15
            P = arcs(x - x0 + np.cos(A) * ln / 2, 330 + 6 + np.sin(A) * ln / 2, A, ln, r.normal(.7, .6, n), K=6)
            v = (.04, .12, .22, .50)[lay]
            col = jit_color(plants.pick(('#16142f', '#201d42', '#312f5c', '#54548e', '#8c92c2'), v + r.normal(0, .05, n)), r, .004, .01, .02)
            cv.strokes(P, r.uniform(1.4, 4.2, n) * (1., 1., .8, .45)[lay], col, op=r.uniform(.8, 1, n), seed=lay * 9 + seed, shape='taper', bristle=.05)
        return x0, H - 330, cv.layer()

    def _fade_map(self):
        """when each part of the painting wakes: horizon first, then crowns and sky, the near ground last"""
        x, y = grid()
        d = np.where(y > 655, (y - 655) / 500. * .95, (655 - y) / 655. * .62) + .10 * np.abs(x - 560) / 1100
        d = d + .03 * fbm(H, W, (260, 110), (1, .4), 2)
        return cv2.GaussianBlur(np.clip(d, 0, 1).astype(f32), (0, 0), 24)

    def _wisp(self, w, h, seed, col, op):
        """a torn shred of mist: long flat-brush lens strokes, crisp on top, dry at the ends"""
        r = np.random.default_rng(seed); hh = h * 4; cv = Canvas(hh, w + 160, layer=True, color=col)
        for k in range(34):
            K = 9; t = np.linspace(-.5, .5, K); q = r.normal(0, .20)
            L = w * r.uniform(.14, .50) * (1 - abs(q) * .9); cx = 80 + w * (.5 + q); cy = hh * .5 + h * (.22 * math.sin(q * 5 + seed) + r.normal(0, .16) * (1 - abs(q)))
            P = np.stack([cx + t * L, cy - h * .12 * np.cos(t * 3.1) * r.uniform(-.3, 1) + np.cumsum(r.normal(0, .35, K))], -1)[None]
            v = 1.07 if k % 3 == 2 else .96
            cv.strokes(P, h * r.uniform(.05, .15) * (.6 if v > 1 else 1), np.clip(rgb(col) * v * (1 + r.normal(0, .015)), 0, 1)[None], op=r.uniform(.16, .42) * op, dry=.3, seed=k, shape='round', bristle=.04)
        return cv2.GaussianBlur(cv.layer(), (0, 0), .9)

    # ------------------------------------------------------------------ per frame
    def lamp(self, f):
        """[(window, u across the window, intensity)], held on twos"""
        fd = (f // 2) * 2; out = []
        for k, (t0, n) in enumerate(self.LAMP):
            q = (fd - t0) / float(n)
            if q < -.001 or q > 1: continue
            u = -.55 + 2.1 * q
            I = math.exp(-((u - .5) / .62) ** 2) * float(np.clip(q / .09, .35, 1)) * float(np.clip((1 - q) / .12, 0, 1))
            I *= 1 + .05 * math.sin(fd * 1.7 + k)                              # flame breath
            out.append((k, u, I))
        return out

    def _light_windows(self, mid, f):
        for (k, u, I) in self.lamp(f):
            w = self.win[k]; x0, y0, x1, y1 = w['box']
            R = mid[y0:y1, x0:x1, :3]
            lx = w['xi'][0] + u * (w['xi'][1] - w['xi'][0]); wd = w['xi'][1] - w['xi'][0]; hh = w['yi'][1] - w['yi'][0]
            ly = w['yi'][0] + .64 * hh
            d2 = ((w['xx'] - lx) / (wd * .80)) ** 2 + ((w['yy'] - ly) / (hh * .50)) ** 2
            e = np.exp(-d2)
            L = I * (.40 + .72 * e)
            core = I * np.exp(-((w['xx'] - lx) / (wd * .17)) ** 2 - ((w['yy'] - ly) / (hh * .10)) ** 2) * (1. if -.05 < u < 1.05 else 0)
            col = ramp(np.clip(L, 0, 1.15), [(0, '#2a1f3e'), (.28, '#8a4d42'), (.55, '#dd8f4a'), (.85, '#ffc46b'), (1.1, '#ffe2a4')])
            a = (w['g'] * np.clip(L * 1.5, 0, 1))[..., None]
            R += (col - R) * a
            R += (rgb('#fff3d0') * 1.12 - R) * (w['g'] * np.clip(core * 1.2, 0, 1))[..., None]
            # reveal: the jamb opposite the flame takes the light
            side = np.clip(.45 + .75 * np.sign(.5 - u) * ((w['xx'] - (w['xi'][0] + w['xi'][1]) / 2) / wd) * min(1, abs(u - .5) * 1.6) * 1.6, .06, 1.2)
            top = np.clip(1.25 - (w['yy'] - w['yi'][0]) / hh * .55, .4, 1.3)
            rv = np.clip(w['rev'] * I * side * top, 0, 1)[..., None]
            R += (R * .6 + rgb('#f0a256') * .62 - R) * rv * .92
            R += R * (w['ring'] * I * .9)[..., None] * rgb('#ffb866') + (w['ring'] * I * .05)[..., None] * rgb('#ffa860')
            R += (rgb('#ffc98a') * .85 - R) * (w['sill'] * I * .55)[..., None]
            R += (R * rgb('#ffb866') * 1.5 + rgb('#d08a48') * .16) * (w['leaf'] * I * .75)[..., None]

    def frame(self, f):
        if f < 29: return np.zeros((H, W, 3), f32)
        t = f / (self.n - 1.)
        z = 1 - (1 - t) ** 1.25
        zb, zm, zf, zg = 1 + .011 * z, 1 + .030 * z, 1 + .033 * z, 1 + .058 * z
        mid = self.mid.copy()
        self._light_windows(mid, f)
        def cam(img, zz, it=cv2.INTER_LINEAR):
            M = np.float32([[1 / zz, 0, self.CX - self.CX / zz], [0, 1 / zz, self.CY - self.CY / zz]])
            return cv2.warpAffine(img, M, (W, H), flags=it | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101)
        img = cam(self.back, zb, cv2.INTER_CUBIC)
        img = self._put(img, self.wisps[0], 300 + .085 * f, 792, .8)
        img = over(img, cam(mid, zm, cv2.INTER_CUBIC))
        img = self._put(img, self.wisps[1], 1190 + .10 * f, 876, .6)
        img = over(img, cam(self.front, zf))
        img = self._put(img, self.wisps[2], 1500 + .12 * f, 846, .75)
        # glow of the small light in the night air
        for (k, u, I) in self.lamp(f):
            w = self.win[k]; gx = w['xi'][0] + np.clip(u, 0, 1) * (w['xi'][1] - w['xi'][0]); gy = w['yi'][0] + .6 * (w['yi'][1] - w['yi'][0])
            gx = self.CX + (gx - self.CX) * zm; gy = self.CY + (gy - self.CY) * zm
            X0, X1, Y0, Y1 = int(gx - 150), int(gx + 150), int(gy - 150), int(gy + 150)
            yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(f32)
            d = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)
            g = (.30 * np.clip(1 - d / 46, 0, 1) ** 2 + .13 * np.clip(1 - d / 145, 0, 1) ** 2) * I
            img[Y0:Y1, X0:X1] = comp.screen(img[Y0:Y1, X0:X1], g[..., None] * rgb('#ffb45e'))
        # pampas, on threes
        for (ux, uy, ul) in self.under:
            M = np.float32([[zg, 0, self.CX + (ux - self.CX) * zg - ux], [0, zg, self.CY + (uy - self.CY) * zg - uy]])
            uw = cv2.warpAffine(ul, M, (ul.shape[1], ul.shape[0]), flags=cv2.INTER_LINEAR)
            if ux == 0: pass
            img[uy:uy + ul.shape[0], ux:ux + ul.shape[1]] = over(img[uy:uy + ul.shape[0], ux:ux + ul.shape[1]], uw)
        d = (f // 3) * 3
        for gi, (P, a, b) in enumerate(self.grass):
            st = P.st; P.st = st.copy()
            P.st[:, 0] = self.CX + (st[:, 0] - self.CX) * zg; P.st[:, 1] = self.CY + (st[:, 1] - self.CY) * zg; P.st[:, 2] = st[:, 2] * zg; P.st[:, 4] = st[:, 4] * zg
            if gi == 1: x0, x1, y0 = 1660, W, 620
            elif gi == 0: x0, x1, y0 = 0, 620, 420
            else: x0, x1, y0 = 0, 620, 260
            cv = Canvas(H - y0, x1 - x0, layer=True)
            P.st[:, 0] -= x0; P.st[:, 1] -= y0
            P.draw(cv, np.stack([a[d], b[d] - .4 * a[d]], 1), seed=gi * 50)
            P.st = st
            l = cv.layer()
            if gi == 2: l = comp.bokeh(l, 3.4, 0)
            img[y0:, x0:x1] = over(img[y0:, x0:x1], l)
        return self._fade(img, f)

    def _put(self, img, spr, x, y, op):
        h, w = spr.shape[:2]; ix, iy = int(math.floor(x)), int(y - h / 2); fx = x - ix
        s = cv2.warpAffine(spr, np.float32([[1, 0, fx], [0, 1, 0]]), (w, h))
        X0, X1 = max(0, ix), min(W, ix + w)
        sl = s[:, X0 - ix:X1 - ix] * op
        img[iy:iy + h, X0:X1] = over(img[iy:iy + h, X0:X1], sl)
        return img

    def _fade(self, img, f):
        """the painting is lit from black through deep blue, the horizon first"""
        if f >= 84: return img
        q = np.clip((f - 30 - 20 * self.delay) / 32., 0, 1)
        q = q * q * (3 - 2 * q)
        L = lum(img)
        g = np.clip((L - .20 * (1 - q) ** 1.6) / (L + 1e-3), 0, 1) * q ** .9       # dark paint wakes last
        k = ((1 - smooth(.30, 1., q)) * .80)[..., None]
        blue = L[..., None] * np.array([.22, .40, 1.12], f32) + np.array([.0, .004, .022], f32) * (q * (1 - q) * 4)[..., None]
        return (img * (1 - k) + blue * k) * g[..., None]


def over4(a, b):
    """premult over premult"""
    return b + a * (1 - b[..., 3:4])
