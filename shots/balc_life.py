"""Balcony lane: wind, grass, ivy and paper, drawn new for every drawing (on twos)."""
import numpy as np, math, cv2
from lp.paint import f32, rgb, mix, Canvas, jit_color
from lp.ink import Cel, spl, resample, noise1


def hash01(*k):
    v = math.sin(sum((i + 1) * 12.9898 * (float(a) + 1.7) for i, a in enumerate(k))) * 43758.5453
    return v - math.floor(v)


def gust(x, f, span=1500., per=26., calm=.42):
    """wind strength 0..1.2 at screen x, gust fronts travelling to the right"""
    x = np.asarray(x, float)
    a = .5 + .5 * np.sin(2 * np.pi * (f / per - x / span))
    b = .5 + .5 * np.sin(2 * np.pi * (f / 9.4 - x / (span * .36)) + 1.3)
    return calm + (1 - calm) * a ** 1.6 + .20 * (b - .5)


# ------------------------------------------------------------------ grass
class Grass:
    """tufts = [(x, y, n, height, spread, depth)]  every blade is its own curve"""
    def __init__(self, tufts, seed=1, pal=None, wk=.055, wmin=.9, lean=.35):
        r = np.random.default_rng(seed); B = []
        for (x, y, n, h, sp, dep) in tufts:
            for j in range(n):
                u = r.normal(0, .5)
                L = h * r.uniform(.45, 1.0) * (1 - .3 * min(1, abs(u)))
                B.append((x + u * sp, y + r.normal(0, sp * .05) + abs(u) * sp * .04, L, -math.pi / 2 + u * .5 + r.normal(0, .12), r.uniform(.75, 1.25), r.uniform(0, 1), dep, r.uniform(0, 100), r.uniform(-.5, .5)))
        B = np.array(B, float).reshape(-1, 9); o = np.argsort(B[:, 6] + B[:, 5] * .4, kind='stable'); self.B = B[o]
        self.pal = pal or [('#5f6a4a', '#6f6f52', '#57635a'), ('#8a8750', '#94904f', '#7c7e52'), ('#b9ab5c', '#d2c067', '#c6b55f'), ('#ecd98a', '#f6e7a2', '#e4cf7c')]
        self.wk, self.wmin, self.lean = wk, wmin, lean
        tone = np.clip(self.B[:, 5], 0, .999)
        k = np.select([tone < .34, tone < .70, tone < .92], [0, 1, 2], 3)
        c = np.stack([rgb(self.pal[a][int(hash01(i, 3) * 3)]) for i, a in enumerate(k)]) if len(B) else np.zeros((0, 3), f32)
        self.col = jit_color(c, r, .006, .012, .02) if len(B) else c
        self.seed = seed

    def polys(self, f, wind=1.0, K=7, gfun=gust):
        B = self.B; d = f // 2; fd = d * 2
        s = np.linspace(0, 1, K)
        g = gfun(B[:, 0], fd + B[:, 7] * .03) * wind
        fl = np.sin(fd * .9 * B[:, 4] + B[:, 7]) * .16 * g + np.sin(fd * .47 + B[:, 7] * 2.) * .08
        # whole-blade lean plus a bend that grows to the tip; the tip lags (overlap)
        gl = gfun(B[:, 0], fd - 3 + B[:, 7] * .03) * wind
        bend0 = g * self.lean * B[:, 4]
        bend1 = gl * 1.15 * B[:, 4] + fl + B[:, 8] * .4
        th = B[:, 3][:, None] + bend0[:, None] * s[None] ** .6 + bend1[:, None] * s[None] ** 1.8
        step = (B[:, 2] / (K - 1))[:, None]
        px = np.cumsum(np.cos(th) * step, 1); py = np.cumsum(np.sin(th) * step, 1)
        P = np.stack([B[:, 0][:, None] + np.c_[np.zeros(len(B)), px[:, :-1]], B[:, 1][:, None] + np.c_[np.zeros(len(B)), py[:, :-1]]], -1)
        r = np.random.default_rng(self.seed * 1000 + d)
        P[:, 1:] += r.normal(0, .35, P[:, 1:].shape)
        return P

    def draw(self, cv_, f, wind=1.0, tint=None, op=.95, K=7, gfun=gust):
        if len(self.B) == 0: return cv_
        P = self.polys(f, wind, K, gfun)
        c = self.col if tint is None else np.clip(self.col * tint, 0, 1)
        cv_.strokes(P, np.maximum(self.wmin, self.B[:, 2] * self.wk * .5), c, op, 0., seed=self.seed, shape='taper', bristle=.05)
        # a few seed heads on the longest stalks
        sel = np.nonzero((self.B[:, 5] > .80) & (self.B[:, 2] > 34))[0][::2]
        if len(sel):
            tip = P[sel, -1]; e = P[sel, -1] - P[sel, -2]; e = e / (np.hypot(*e.T)[:, None] + 1e-6); L = self.B[sel, 2]
            for j in range(4):
                q = tip - e * (L * .055 * j)[:, None]; n = np.c_[-e[:, 1], e[:, 0]] * ((-1) ** j)
                A = q; B2 = q + (e * .7 + n * .55) * (L * .05)[:, None]
                cv_.strokes(np.stack([A + (B2 - A) * t for t in (0, .35, .7, 1)], 1), np.maximum(.7, L * .014), c[sel] * np.array([1.0, .93, .86], f32), .9, 0, seed=j, shape='leaf')
        return cv_


def cel_blade(c, P, w, fill, shade, light=None, ink='#5b4a34', lw=1.4, seed=0, alpha=1.0):
    """a big foreground blade: flat fill, one hard shadow half, a light edge, a broken contour"""
    P = resample(spl(P), 6.); n = len(P)
    T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
    t = np.linspace(0, 1, n)
    r = np.random.default_rng(seed)
    hw = w * .5 * (np.clip((1 - t) / .75, 0, 1) ** .85) * (.72 + .28 * np.clip(t / .25, 0, 1)) + .25
    hw = hw * (1 + .05 * noise1(n, 18, r))
    L, R = P + N * hw[:, None], P - N * hw[:, None]
    body = np.vstack([L, R[::-1]])
    c.fill(body, fill, alpha)
    mid = P + N * (hw * r.uniform(-.25, .15))[:, None]
    c.fill(np.vstack([mid, R[::-1]]), shade, alpha)
    if light is not None: c.line(P[int(n * .1):] + N[int(n * .1):] * (hw[int(n * .1):] * .78)[:, None], max(.8, w * .10), light, (.3, .5), seed=seed + 1, alpha=alpha)
    c.line(mid, max(.6, lw * .6), ink, (.2, .6), seed=seed + 2, alpha=.45 * alpha)
    if lw > 0:
        c.line(L, lw, ink, (.1, .3), seed=seed + 3, alpha=alpha); c.line(R, lw * 1.15, ink, (.1, .3), seed=seed + 4, alpha=alpha)
    return body


# -------------------------------------------------------------------- ivy
class Ivy:
    """stems = list of polylines (screen px, root first).  Leaves hang off them; free ends swing."""
    def __init__(self, stems, seed=1, leaf=11., gap=7., free=None, pal=None, spread=1.0, dens=1.0, taper=1.4):
        r = np.random.default_rng(seed); self.L = []; self.stems = []; self.seed = seed
        pal = pal or [('#4b5756', '#55604f', '#5a5a5e'), ('#6d774d', '#7c8052', '#63705a'), ('#a09c56', '#b3a75c', '#8f9152'), ('#dccb7a', '#ecdc92', '#cfbf6c')]
        rows = []
        for si, st in enumerate(stems):
            P = resample(spl(np.asarray(st, float)), 3.); n = len(P); s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
            fr = .25 if free is None else (free[si] if np.ndim(free) else free)
            self.stems.append((P, s / s[-1], fr, r.uniform(0, 99)))
            q = 0.
            while q < s[-1]:
                u = q / s[-1]
                cl = 1 + taper * max(0., 1 - 1.5 * u) + .5 * max(0., math.sin(u * 17 + si * 2.1)) * taper      # clumps at the root, thinning to the tip
                for k in range(int(1 + r.random() * .9 * dens + cl * 1.1)):
                    sz = leaf * r.uniform(.45, 1.2) * (1 - .3 * u)
                    sd = r.choice([-1, 1]); off = r.uniform(.0, 1.0) ** .8 * leaf * .9 * spread * cl
                    tone = r.random()
                    rows.append((si, u, sd * off, sz, r.uniform(-1.1, 1.1), tone, r.uniform(0, 99), r.uniform(.3, .5)))
                q += gap * r.uniform(.4, 1.6) / dens
        self.R = np.array(rows, float).reshape(-1, 8)
        # light sits on the outer, upper leaves; the dark ones are tucked in
        self.R = self.R[np.argsort(self.R[:, 5] + .25 * r.random(len(self.R)))]
        tone = self.R[:, 5]; k = np.select([tone < .36, tone < .72, tone < .92], [0, 1, 2], 3)
        self.col = jit_color(np.stack([rgb(pal[a][int(hash01(i, 7) * 3)]) for i, a in enumerate(k)]), r, .006, .012, .02) if len(self.R) else np.zeros((0, 3), f32)
        self.stemc = rgb('#6a5560')

    def draw(self, cv_, f, wind=1.0, tint=None, gfun=gust, op=.96):
        d = f // 2; fd = d * 2
        if len(self.R) == 0: return cv_
        base = np.zeros((len(self.R), 2)); tan = np.zeros((len(self.R), 2))
        for si, (P, u, fr, ph) in enumerate(self.stems):
            g = float(gfun(P[:, 0].mean(), fd)) * wind
            k = np.clip((u - (1 - fr)) / max(fr, 1e-3), 0, 1) ** 1.5 if fr > 0 else u * 0
            L = np.hypot(*np.diff(P, axis=0).T).sum()
            sw = (g * .55 + .13 * np.sin(fd * .6 + ph - u * 3.0) * (.4 + g)) * k * L * .42
            Q = P + np.c_[sw, -np.abs(sw) * .22 + 1.5 * k * np.sin(fd * .8 + ph + u * 5)]
            cv_.strokes(_chain(Q), 1.1, self.stemc if tint is None else np.clip(self.stemc * tint, 0, 1), .8, .1, seed=si, shape='round')
            m = self.R[:, 0] == si
            uu = self.R[m, 1]
            base[m] = np.c_[np.interp(uu, u, Q[:, 0]), np.interp(uu, u, Q[:, 1])]
            T = np.gradient(Q, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9
            tan[m] = np.c_[np.interp(uu, u, T[:, 0]), np.interp(uu, u, T[:, 1])]
        R = self.R
        nn = np.c_[-tan[:, 1], tan[:, 0]]
        c = base + nn * R[:, 2:3]
        g = gfun(c[:, 0], fd + R[:, 6] * .05) * wind
        a = np.arctan2(tan[:, 1], tan[:, 0]) + np.sign(R[:, 2]) * .5 + R[:, 4] * .6 + np.sin(fd * .85 + R[:, 6]) * .28 * g - .35 * g * np.sign(tan[:, 1] + 1e-3) * .6
        sz = R[:, 3] * (1 - .25 * np.abs(np.sin(fd * .55 + R[:, 6] * 3)) * np.clip(g, 0, 1))
        e = np.c_[np.cos(a), np.sin(a)]
        P = np.stack([c + e * (sz * t)[:, None] + np.c_[-e[:, 1], e[:, 0]] * (sz * .12 * math.sin(math.pi * (t + .5)) * np.sign(R[:, 4]))[:, None] for t in (-.5, -.15, .2, .55)], 1)
        col = self.col if tint is None else np.clip(self.col * tint, 0, 1)
        cv_.strokes(P, sz * R[:, 7] * .85, col, op, 0, seed=self.seed + 5, shape='leaf', bristle=.06)
        return cv_


def _chain(Q, K=5, st=4):
    out = []
    for i in range(0, len(Q) - 1, (K - 1) * st):
        idx = np.minimum(np.arange(K) * st + i, len(Q) - 1)
        out.append(Q[idx])
    return np.array(out, f32)


# ------------------------------------------------------------------ paper
def rot(yaw=0., pitch=0., roll=0.):
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.]]); Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]); Ry = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
    return Rz @ Rx @ Ry


class Sheet:
    """one paper design: outline in (u,v) metres on the sheet, faded writing / picture strokes on its face"""
    def __init__(self, w, h, seed, kind='text', torn=None):
        r = np.random.default_rng(seed); self.w, self.h, self.kind, self.seed = w, h, kind, seed
        hw, hh = w / 2, h / 2
        E = []
        def edge(a, b, amp, n, jag=False):
            a, b = np.array(a), np.array(b); e = b - a; nn = np.array([e[1], -e[0]]) / np.hypot(*e)
            nz = np.cumsum(r.normal(0, 1, n)); nz = (nz - np.linspace(nz[0], nz[-1], n)); nz /= np.abs(nz).max() + 1e-6
            j = r.uniform(-1, .3, n) if jag else r.normal(0, .25, n)
            return [a + e * t - nn * (abs(amp) * (z * .6 + q * .5) if jag else amp * (z + q * .5)) for t, z, q in zip(np.linspace(0, 1, n, endpoint=False), nz, j)]
        cs = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        for k in range(4):
            t = torn == k
            E += edge(cs[k], cs[(k + 1) % 4], (.035 * w if t else .004 * w), 16 if t else 9, t)
        self.E = np.array(E)
        if r.random() < .55 and torn is None:                      # a nibbled corner
            k = 9 * int(r.integers(4)); self.E[k] = self.E[k] * r.uniform(.86, .95)
        self.ink = []                                              # (polyline, width_m, tone)
        if kind in ('text', 'pict', 'letter'):
            top = -hh * .80; bot = hh * .82
            if kind == 'pict':
                self._pict(r, hw, hh); top = hh * .18
            if kind == 'letter': top = -hh * .45
            y = top
            lead = h * r.uniform(.055, .075)
            while y < bot:
                x = -hw * r.uniform(.74, .80) + (hw * .2 if r.random() < .12 else 0); xe = hw * (r.uniform(.45, .8) if r.random() < .8 else r.uniform(-.1, .4))
                while x < xe:                                       # words as broken hand strokes
                    wl = w * r.uniform(.035, .11); n = 7
                    t = np.linspace(0, 1, n)
                    self.ink.append((np.c_[x + wl * t, y + lead * .16 * np.sin(t * 22 + r.uniform(0, 6)) * r.uniform(.5, 1.1, n)], .0035 * w * r.uniform(.8, 1.4), 0))
                    x += wl + w * r.uniform(.015, .03)
                y += lead * r.uniform(.92, 1.1)
                if r.random() < .10: y += lead
        fine = []
        for pl, wd, tn in self.ink:
            pl = np.asarray(pl, float)
            if tn == 1 or len(pl) < 6:
                sg = [pl[0]]
                for a, b in zip(pl[:-1], pl[1:]):
                    k = max(1, int(np.hypot(*(b - a)) / (w * .02)))
                    sg += [a + (b - a) * (j + 1) / k + r.normal(0, w * .0012, 2) for j in range(k)]
                pl = np.array(sg)
            fine.append((pl, wd, tn))
        self.ink = fine
        # creases: straight folds across the sheet
        self.creases = []
        for k in range(int(r.integers(1, 3))):
            a = r.uniform(0, np.pi); p = r.uniform(-.25, .25, 2) * (w, h); d = np.array([math.cos(a), math.sin(a)])
            self.creases.append((p - d * w * r.uniform(.2, .5), p + d * w * r.uniform(.2, .5)))

    def _pict(self, r, hw, hh):
        k = int(r.integers(0, 5)); cx, cy, s = 0., -hh * .36, hw * .62
        ad = lambda pts, wd=.006, tone=1: self.ink.append((np.array(pts) * s + (cx, cy), wd * hw * 2, tone))
        fr = [(-1, -.72), (1, -.72), (1, .8), (-1, .8), (-1, -.72)]
        ad(fr, .004, 0)
        if k == 0:    # a tree on a hill
            ad([(-.9, .55), (-.3, .35), (.2, .42), (.9, .3)]); ad([(0, .4), (-.03, .1), (.04, -.2)], .012)
            for a in np.linspace(0, 6.28, 9): ad([(.02 + .38 * math.cos(a), -.25 + .3 * math.sin(a)), (.02 + .42 * math.cos(a + .4), -.25 + .34 * math.sin(a + .4)), (.02 + .36 * math.cos(a + .75), -.25 + .28 * math.sin(a + .75))])
        elif k == 1:  # a small sailing boat
            ad([(-.5, .25), (-.3, .45), (.3, .45), (.55, .25), (-.5, .25)], .008); ad([(0, .25), (.02, -.55)], .008); ad([(.02, -.55), (.48, .15), (.04, .15)]); ad([(.0, -.45), (-.35, .15), (-.02, .15)])
            for y in (.55, .65): ad([(-.85, y), (-.4, y - .04), (0, y), (.4, y - .04), (.85, y)], .004)
        elif k == 2:  # moon over mountains
            ad([(-.95, .6), (-.4, -.1), (-.1, .3), (.35, -.3), (.95, .6)], .008); a = np.linspace(0, 6.28, 14); ad(np.c_[-.45 + .16 * np.cos(a), -.42 + .16 * np.sin(a)])
            ad([(.1, .1), (.25, .25), (.35, .12), (.45, .28)], .005)
        elif k == 3:  # a long-tailed bird
            ad([(-.6, .1), (-.2, -.15), (.15, -.05), (.4, -.3), (.6, -.25), (.42, -.12), (.2, .2), (-.2, .25), (-.6, .1)], .008); ad([(-.6, .1), (-.9, .3)]); ad([(-.55, .15), (-.8, .5)]); ad([(-.1, .0), (-.3, -.45), (.1, -.1)])
        else:         # a flower with roots
            ad([(0, .7), (.05, .2), (-.02, -.2)], .008)
            for a in np.linspace(.3, 6.0, 7): ad([(0, -.3), (.22 * math.cos(a + .2), -.3 + .22 * math.sin(a + .2)), (.28 * math.cos(a), -.3 + .28 * math.sin(a)), (0, -.3)])
            ad([(.03, .3), (.35, .12), (.4, .25), (.04, .38)]); ad([(.0, .7), (-.2, .78)]); ad([(0, .7), (.18, .76)])


PAPER = dict(burn='#fff6da', lit='#faebca', trans='#e6bf91', half='#d0b9a6', sh='#bd9c9a', deep='#8b7b9c', ink='#6b4f4a', text='#a5866f', blank='#f2deb2')


def _clip_half(poly, n, c):
    """keep points with n.p >= c"""
    out = []
    m = len(poly)
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        da, db = n @ a - c, n @ b - c
        if da >= 0: out.append(a)
        if (da >= 0) != (db >= 0): out.append(a + (b - a) * (da / (da - db)))
    return np.array(out) if len(out) >= 3 else None


def paper(cel, sh, pos, R, proj, bend=0.6, beta=.4, curl=0., curl_at=.55, light=(.55, -.62, .56), eye=None, pal=PAPER, lw=1.7, seed=0, N=13, mode='back', boil=.0, detail=True, waves=0.):
    """draw a bent sheet.  pos metres (3,), R 3x3 sheet axes -> world, proj(P (n,3)) -> (xy (n,2), depth (n,)).
    bend: total radians of cylindrical bend along the direction beta on the sheet; curl: extra bend at the far end."""
    r = np.random.default_rng(seed)
    d = np.array([math.cos(beta), math.sin(beta)]); e = np.array([-d[1], d[0]])
    E = sh.E
    s = E @ d; s0, s1 = s.min(), s.max()
    # bend profile along s
    q = np.linspace(s0, s1, 80); u = (q - s0) / (s1 - s0)
    k = bend / (s1 - s0) * np.ones_like(q) + curl / (s1 - s0) / (1 - curl_at) * (u > curl_at) * np.clip((u - curl_at) / (1 - curl_at), 0, 1) * 2 + waves * np.sin(u * 7 + seed) * 6 / (s1 - s0)
    th = np.cumsum(k) * (q[1] - q[0]); th -= np.interp(0., q, th)
    X = np.cumsum(np.cos(th)) * (q[1] - q[0]); Z = np.cumsum(np.sin(th)) * (q[1] - q[0])
    X -= np.interp(0., q, X); Z -= np.interp(0., q, Z)
    def wrap(p):
        p = np.asarray(p, float); a = p @ d; b = p @ e
        xx = np.interp(a, q, X); zz = np.interp(a, q, Z)
        loc = np.c_[xx[:, None] * d[None] + b[:, None] * e[None], zz]
        return loc @ np.asarray(R).T + np.asarray(pos)
    L = np.asarray(light, float); L /= np.linalg.norm(L)
    cuts = np.linspace(s0, s1, N + 1)
    strips = []
    for i in range(N):
        a, b = cuts[i], cuts[i + 1]
        ov = (b - a) * .06
        pl = _clip_half(E, d, a - ov); pl = _clip_half(pl, -d, -(b + ov)) if pl is not None else None
        if pl is None: continue
        P3 = wrap(pl); xy, z = proj(P3)
        t = np.interp((a + b) / 2, q, th)
        nl = np.array([-math.sin(t) * d[0], -math.sin(t) * d[1], math.cos(t)]); nw = np.asarray(R) @ nl
        c3 = P3.mean(0); view = (np.asarray(eye, float) - c3) if eye is not None else np.array([0, -1., 0])
        view /= np.linalg.norm(view)
        front = nw @ view > 0
        nv = nw if front else -nw
        strips.append((z.mean(), i, xy, front, nv @ L, a, b, t, abs(nw @ view)))
    strips.sort(key=lambda v: -v[0])
    pal_c = {k_: (rgb(v) if isinstance(v, str) else v) for k_, v in pal.items()}
    def tone(dl, front):
        if mode == 'back':            # sun behind: paper glows through
            return 'burn' if dl > .42 else 'lit' if dl > .06 else 'sh' if dl > -.34 else 'trans'
        return 'burn' if dl > .82 else 'lit' if dl > .2 else 'sh' if dl > -.35 else 'deep'
    fr = {i: f for _, i, _, f, _, _, _, _, _ in strips}
    inkpts = [(np.asarray(pl), wd, tn) for pl, wd, tn in sh.ink] if detail else []
    ctx = cel.ctx
    nE = len(E); Es = s
    onefold = len(set(fr.values())) == 1
    seed_tone = sh.seed * 1.7
    for z, i, xy, front, dl, a, b, t, fac in strips:
        dl = dl + .20 * math.sin((i + .5) / N * 4.1 + seed_tone)                      # every sheet keeps one hard tone break
        col = pal_c[tone(dl, front)]
        if sh.kind == 'blank' and tone(dl, front) in ('lit', 'trans'): col = mix(col, pal_c['blank'], .5)
        cel.fill(xy, col)
        # writing on the face
        if detail and fac > .03 and (front or mode == 'back'):
            for pl, wd, tn in inkpts:
                sm = pl @ d; mid = (sm[:-1] + sm[1:]) / 2; ok = (mid >= a) & (mid < b)
                if not ok.any(): continue
                w3, _ = proj(wrap(pl))
                sc = np.hypot(*(w3[-1] - w3[0])) / (np.hypot(*(pl[-1] - pl[0])) + 1e-9)
                ctx.set_source_rgba(*[float(v) for v in mix(pal_c['text'], col, (.30 if tn == 0 else .12) if front else .55)], .9); ctx.set_line_width(max(.5, wd * sc)); ctx.set_line_cap(1)
                for j in np.nonzero(ok)[0]:
                    ctx.move_to(*w3[j]); ctx.line_to(*w3[j + 1])
                ctx.stroke()
        if front or True:
            for (ca, cb) in sh.creases:
                pl = np.linspace(ca, cb, 14); sm = pl @ d; mid = (sm[:-1] + sm[1:]) / 2; ok = (mid >= a) & (mid < b)
                if ok.any():
                    w3, _ = proj(wrap(pl)); ctx.set_source_rgba(*[float(v) for v in mix(col, pal_c['deep'], .35)], .8); ctx.set_line_width(max(.5, lw * .5))
                    for j in np.nonzero(ok)[0]: ctx.move_to(*w3[j]); ctx.line_to(*w3[j + 1])
                    ctx.stroke()
        # outline parts belonging to this strip, hand pressure
        if onefold: continue
        run = []
        for j in range(nE + 1):
            jj = j % nE
            s_ = Es[jj]
            inside = a - (b - a) * .6 <= s_ <= b + (b - a) * .6
            if inside and j < nE: run.append(E[jj])
            else:
                if len(run) >= 2: _edge(cel, proj(wrap(np.array(run)))[0], lw, pal_c['ink'], seed + i * 7 + j, boil)
                run = []
        if len(run) >= 2: _edge(cel, proj(wrap(np.array(run)))[0], lw, pal_c['ink'], seed + i, boil)
        for nb, sq in ((i - 1, a), (i + 1, b)):                     # contour where the sheet turns over
            if nb in fr and fr[nb] != front:
                sg = _clip_half(_clip_half(E, d, sq - 1e-3), -d, -(sq + 1e-3)) if True else None
                if sg is not None:
                    bb = sg @ e; pl = np.stack([d * sq + e * v for v in np.linspace(bb.min(), bb.max(), 5)])
                    _edge(cel, proj(wrap(pl))[0], lw, pal_c['ink'], seed + 91 + i, boil)
    if onefold:
        P = E[np.r_[np.arange(nE), 0]]
        cor = [0] + [j for j in range(1, nE) if np.dot(E[j] - E[j - 1], E[(j + 1) % nE] - E[j]) < .35 * np.hypot(*(E[j] - E[j - 1])) * np.hypot(*(E[(j + 1) % nE] - E[j]))] + [nE]
        for a, b in zip(cor[:-1], cor[1:]):
            if b - a >= 1: _edge(cel, proj(wrap(P[a:b + 1]))[0], lw, pal_c['ink'], seed + a, boil)
    return wrap


def _edge(cel, xy, lw, col, seed, boil):
    if len(xy) < 2: return
    if np.hypot(*np.ptp(xy, 0)) < 1.2: return
    if boil: xy = xy + np.random.default_rng(seed).normal(0, boil, xy.shape)
    if lw < .8 or len(xy) < 3:
        c = cel.ctx; c.set_source_rgba(float(col[0]), float(col[1]), float(col[2]), .9); c.set_line_width(max(.5, lw)); c.set_line_cap(1); c.move_to(*xy[0])
        for q in xy[1:]: c.line_to(*q)
        c.stroke(); return
    cel.line(xy, lw, col, (.03, .03), press=.3, seed=seed, tremor=.25, tmin=.5)
