"""The library tower of shot 1: laid out on a Camera3D rig, painted stone by stone in gouache."""
import numpy as np, cv2, math
from lp import W, H
from lp.paint import f32, rgb, mix, Canvas, arcs, jit_color, smooth, fbm, repaint
from lp.ink import Cel, spl, resample, noise1
from lp.cam import Camera3D
from lp import plants

HZ = 690.
CAM = Camera3D((0, 0, 0), (0, 1, 0), focal=2200., cy=HZ)
D = 128.
RS = .885                                        # slenderness
X0 = (1266 - 1024) / 2200. * D
PX = 2200. / D                                   # px per metre at the axis

STONE = ['#1f2147', '#292c57', '#373c6b', '#535a8a', '#787eac', '#a0a6ce', '#ced5ec']
COPPER = ['#1d2848', '#26395c', '#304d6d', '#466b80', '#7099a6', '#a2c0c9', '#d2e1e9']
WARM = rgb('#ffc46b')

zk = np.array([-2, 0, 4, 9.5, 15, 20, 23.5, 26, 29, 31.5, 34.5])
cxk = np.array([0, 0, .10, -.26, .20, -.30, -.12, .16, .36, .66, 1.25])
tik = np.radians([0, .3, -.8, 1.0, -1.3, 1.3, 2.0, 3.0, 3.6, 4.4, 6.0])


def axis(z):
    return np.interp(z, zk, cxk), np.interp(z, zk, tik)


def T(th, z, r, ax=None):
    """tower-local (theta rad 0 = facing us, z, r) -> screen px.  ax = (x, y) offset of a side axis"""
    th, z, r = np.broadcast_arrays(np.asarray(th, float), np.asarray(z, float), np.asarray(r, float))
    dx, ti = axis(z)
    r = r * RS
    x = r * np.sin(th); y = -r * np.cos(th)
    ox, oy = (0, 0) if ax is None else ax
    P = np.stack([X0 + dx + ox + x * np.cos(ti), D + oy + y, z + (x + ox) * np.sin(ti)], -1)
    return CAM.project(P)[0]


def light(th):
    """moon from the upper left and a little behind: 0..1 around the drum"""
    th = np.asarray(th, float)
    v = np.clip(-.95 * np.sin(th) - .16 * np.cos(th), 0, 1)
    refl = .16 * smooth(math.radians(52), math.radians(88), th)
    return .17 + .85 * v ** .70 + refl - .09 * np.clip(np.sin(th + .25), 0, 1)


def rough_line(P, r, amp=.7, n=4):
    P = np.asarray(P, float); out = []
    for a, b in zip(P[:-1], P[1:]):
        t = np.arange(n) / n; out.append(a + (b - a) * t[:, None] + r.normal(0, amp, (n, 2)) * (t[:, None] > 0))
    return np.vstack(out + [P[-1:]])


def limb_rim(p, R, z0, z1, th=-87, ax=None, w=2.0, a=.9, keep=.8):
    """crisp pale moon edge down the left silhouette, broken where the stones are chipped"""
    z = z0; r = p.r
    while z < z1 - .1:
        h = r.uniform(.35, 1.1); zz = np.linspace(z, min(z1, z + h), 4)
        if r.random() < keep:
            L = np.array([T(math.radians(th) + r.normal(0, .004), q, R(q), ax) for q in zz])
            p.c.line(L + [1.0, 0], w * r.uniform(.7, 1.25), p.col(STONE, .98, .01), (.2, .2), .4, seed=int(z * 31), alpha=a * r.uniform(.6, 1))
        z += h + r.uniform(.03, .16)


def poly(q, n=4, corners=True):
    return spl(q, closed=True, corners=range(len(q)) if corners else (), n=n)


class Painter:
    def __init__(self, seed=1):
        self.r = np.random.default_rng(seed)
        self.x0, self.y0, self.w, self.h = 940, 40, 720, 720
        self.c = Cel(self.w, self.h, self.x0, self.y0, ss=2)
        self.lab = np.zeros((self.h, self.w), np.int32); self.k = 0
        self.ang = np.zeros((self.h, self.w), f32)
        self.windows = []
        self.open = np.zeros((self.h, self.w), np.uint8)

    # --- primitives
    def fill(self, P, col, a=1.0, lab=True, ang=None):
        self.c.fill(P, col, a)
        if lab:
            self.k += 1
            q = np.round((np.asarray(P) - (self.x0, self.y0)) * 8).astype(np.int32)
            cv2.fillPoly(self.lab, [q], self.k, cv2.LINE_8, 3)
            if ang is not None: cv2.fillPoly(self.ang, [q], float(ang), cv2.LINE_8, 3)
        return self

    def col(self, pal, v, j=.04):
        v = v + self.r.normal(0, j)
        c = plants.pick(pal, np.array([np.clip(v, 0, 1)]))
        L = float(c.mean())
        return jit_color(c, self.r, .0015 + .008 * L * L, .010, .012)[0]

    # --- masonry
    def base_shell(self, R, z0, z1, th0=-92, th1=92, ax=None, pal=STONE, vf=None, nth=40, nz=10):
        """underpaint: the wall as flat vertical tone planks"""
        ths = np.radians(np.linspace(th0, th1, nth)); zs = np.linspace(z0, z1, nz)
        for a, b in zip(ths[:-1], ths[1:]):
            for u, v in zip(zs[:-1], zs[1:]):
                vv = float(light((a + b) / 2)) * .9 - .05
                if vf: vv += vf((a + b) / 2, (u + v) / 2)
                q = np.array([T(a - .003, u, R(u), ax), T(b + .006, u, R(u), ax), T(b + .006, v + .02, R(v), ax), T(a - .003, v + .02, R(v), ax)])
                self.c.fill(q, self.col(pal, vv, .012))

    def stones(self, R, z0, z1, th0=-92, th1=92, hrow=(.42, .68), wr=(1.2, 2.6), ax=None, pal=STONE, vf=None, gap=.035, skip=None, jit=.05, lo=.0, rough=.03):
        r = self.r
        z = z0
        while z < z1 - .08:
            h = min(r.uniform(*hrow), z1 - z)
            if z1 - (z + h) < hrow[0] * .5: h = z1 - z
            th = math.radians(th0) - r.uniform(0, .15)
            wav = r.uniform(0, 6.28)
            rowv = r.normal(0, .018)
            while th < math.radians(th1):
                rr = R(z + h / 2)
                dth = h * r.uniform(*wr) / rr
                a, b = th, th + dth
                th = b
                if a < math.radians(th0): a = math.radians(th0)
                b = min(b, math.radians(th1))
                if b - a < .02: continue
                if skip is not None and skip((a + b) / 2, z + h / 2): continue
                g = gap * r.uniform(.5, 1.4); ga = g / rr
                wv = lambda t, zz: zz + .035 * math.sin(t * 3.1 + wav) + r.normal(0, rough)
                m1, m2 = a + (b - a) * r.uniform(.3, .45), a + (b - a) * r.uniform(.55, .7)
                q = [(a + ga, wv(a, z) + g), (m1, wv(m1, z) + g * .6), (m2, wv(m2, z) + g * .6), (b - ga, wv(b, z) + g),
                     (b - ga + r.normal(0, .006), wv(b, z + h * .5)), (b - ga, wv(b, z + h) - g), (m2, wv(m2, z + h) - g * .5), (m1, wv(m1, z + h) - g * .5), (a + ga, wv(a, z + h) - g),
                     (a + ga + r.normal(0, .006), wv(a, z + h * .5))]
                P = np.array([T(t, zz, R(zz), ax) for t, zz in q])
                lv = float(light((a + b) / 2)); q_ = .38 + .62 * smooth(.2, .5, lv)
                v = lv + (rowv + r.normal(0, jit)) * q_
                if r.random() < .06: v += r.choice([-.10, .10]) * q_
                if vf: v += vf((a + b) / 2, z + h / 2)
                v = max(v, lo)
                c = self.col(pal, v, .01)
                # a few stones of another quarry
                u = r.random()
                if u < .10: c = mix(c, c * np.array([1.07, .985, .97], f32), q_)
                elif u < .20: c = mix(c, c * np.array([.94, 1.03, 1.015], f32), q_)
                d = P[3] - P[0]
                self.fill(poly(P, 3, False), c, ang=math.atan2(d[1], d[0]))
                # worn moonlit arris: upper-left edge of stones that face the moon
                if lv > .30 and r.random() < .75 * min(1, lv * 1.6):
                    e = np.array([T(a + ga * 1.5, z + h * r.uniform(.15, .4), R(z), ax), T(a + ga * 1.6, z + h - g * 1.6, R(z + h), ax), T(m1, z + h - g * 1.4, R(z + h), ax)])
                    self.c.line(spl(e, n=5), r.uniform(.7, 1.5), self.col(pal, min(1, v + .24), .03), (.3, .6), .3, seed=self.k, alpha=r.uniform(.5, .95))
                if r.random() < .5 * q_:
                    e = np.array([T(m1, z + g, R(z), ax), T(b - ga, z + g, R(z), ax), T(b - ga, z + h * .6, R(z), ax)])
                    self.c.line(spl(e, n=5), r.uniform(.6, 1.2), self.col(pal, max(0, v - .16), .03), (.4, .5), .3, seed=self.k + 7, alpha=r.uniform(.35, .75))
            z += h

    def band(self, R, z0, z1, out, th0=-92, th1=92, w=(.7, 1.3), ax=None, pal=STONE, dv=.06, vf=None, shadow=.55, gap=.03):
        """projecting string course with its cast shadow on the wall below"""
        r = self.r
        ths = np.radians(np.linspace(th0, th1, 40))
        if shadow > 0:
            lo = [T(t, z0 - shadow * (.75 + .25 * math.sin(t * 5 + 1) + .5 * max(0, -math.sin(t))), R(z0 - shadow), ax) for t in ths]
            up = [T(t, z0 + .02, R(z0), ax) for t in ths]
            P = np.vstack([up, lo[::-1]])
            self.c.fill(P, rgb('#20224a'), .42)
        th = math.radians(th0)
        while th < math.radians(th1):
            rr = R(z1) + out
            b = min(th + r.uniform(*w) / rr, math.radians(th1)); g = gap / rr * r.uniform(.6, 1.2)
            j = lambda: r.normal(0, .025)
            q = [(th + g, z0 + j()), (b - g, z0 + j()), (b - g, z1 + j()), (th + g, z1 + j())]
            P = np.array([T(t, zz, rr, ax) for t, zz in q])
            v = float(light((th + b) / 2)) + dv + r.normal(0, .04) + (vf((th + b) / 2, z0) if vf else 0)
            self.fill(P, self.col(pal, v, .01), ang=math.atan2(*(P[1] - P[0])[::-1]))
            lv = float(light((th + b) / 2)) + (vf((th + b) / 2, z0) if vf else 0)
            # lit top edge, dark lower edge
            self.c.line(np.array([P[3], (P[3] + P[2]) / 2, P[2]]), 1.5 if lv > .3 else .9, self.col(pal, min(1, v + (.30 if lv > .28 else .12)), .02), (.15, .15), .3, seed=self.k, alpha=.9 if lv > .25 else .5)
            self.c.line(np.array([P[0], (P[0] + P[1]) / 2, P[1]]), 1.1, self.col(pal, max(0, v - .22), .02), (.15, .15), .3, seed=self.k + 3, alpha=.7)
            th = b

    # --- openings
    def arch(self, th, z0, w, h, R, r_in=0., n=11, pointed=.12, ax=None, grow=0.0, zgrow=None):
        """arched outline on the wall at radius R - r_in (screen px, from lower left round to lower right)"""
        rr = R(z0 + h / 2)
        a = (w / 2 + grow); zg = grow if zgrow is None else zgrow
        hs = h - w / 2                                          # springing
        pts = [(-a, -zg), (-a, hs * .5), (-a, hs)]
        for t in np.linspace(math.pi, 0, n)[1:-1]:
            pts.append((a * math.cos(t), hs + (w / 2 * (1 + pointed * math.sin(t) ** 4) + grow) * math.sin(t)))
        pts += [(a, hs), (a, hs * .5), (a, -zg)]
        return np.array([T(th + u / rr, z0 + v, rr - r_in, ax) for u, v in pts])

    def window(self, th, z0, w, h, R, ax=None, depth=.5, trim=.34, sill=True, glass=True, mull=(1, 3), key=None, vdark=0.0, pal=STONE, stain=True, trimv=.07):
        r = self.r
        th = math.radians(th)
        rr = R(z0 + h / 2)
        lv = float(light(th))
        # soft contact shadow of the trim
        # voussoirs + jamb stones
        if trim > 0:
            hs = h - w / 2
            nj = max(3, int(hs / .5))
            for s in (-1, 1):
                for j in range(nj):
                    za, zb = hs * j / nj + .02, hs * (j + 1) / nj - .02
                    ln = trim * (1.25 if (j + (s > 0)) % 2 else .8) * r.uniform(.9, 1.15)
                    u0, u1 = s * (w / 2 - .02), s * (w / 2 + ln)
                    P = np.array([T(th + u / rr, z0 + z + r.normal(0, .02), rr + .03, ax) for u, z in ((u0, za), (u1, za), (u1, zb), (u0, zb))])
                    v = float(light(th + (u0 + u1) / 2 / rr)) + trimv + r.normal(0, .035) - vdark
                    self.fill(P, self.col(pal, v, .01), ang=0.)
                    if lv > .28: self.c.line(np.array([P[0] if s < 0 else P[3], P[3]]) if False else np.array([P[3], (P[2] + P[3]) / 2, P[2]]), .9, self.col(pal, min(1, v + .2)), (.2, .3), alpha=.6, seed=self.k)
            nv = 7 if w > .9 else 5
            ts = np.linspace(math.pi, 0, nv + 1)
            for j in range(nv):
                t0, t1 = ts[j] - .025, ts[j + 1] + .025
                key_ = (j == nv // 2)
                o = trim * (1.35 if key_ else 1.0) * r.uniform(.92, 1.1)
                q = []
                for t, e in ((t0, 0), ((t0 + t1) / 2, 0), (t1, 0), (t1, o), ((t0 + t1) / 2, o * 1.04), (t0, o)):
                    q.append(((w / 2 + e - .02) * math.cos(t), hs + (w / 2 * (1 + .12 * math.sin(t) ** 4) + e - .02) * math.sin(t)))
                P = np.array([T(th + u / rr, z0 + zz + r.normal(0, .015), rr + .03, ax) for u, zz in q])
                v = float(light(th)) + trimv + .10 * math.cos((t0 + t1) / 2 - 2.2) + r.normal(0, .035) - vdark
                self.fill(P, self.col(pal, v, .01), ang=0.)
                if math.cos((t0 + t1) / 2 - 2.2) > .3 and lv > .22:
                    self.c.line(P[[5, 4, 3]], 1.1, self.col(pal, min(1, v + .22)), (.2, .3), alpha=.8, seed=self.k)
        outer = self.arch(th, z0, w, h, R, 0, ax=ax)
        inner = self.arch(th, z0, w, h, R, depth, ax=ax, grow=-.03)
        cv2.fillPoly(self.open, [np.round((outer - (self.x0, self.y0)) * 8).astype(np.int32)], 255, cv2.LINE_8, 3)
        # reveal: side jamb + soffit in shadow, the far jamb picks a little sky
        self.fill(outer, self.col(STONE, .14 + .10 * lv, .01), ang=1.57)
        n = len(outer) // 2
        # limb-side jamb strip (faces us) is the darker one on the moon side
        jamb = np.vstack([outer[:4], inner[:4][::-1]]) if th < 0 else np.vstack([outer[-4:], inner[-4:][::-1]])
        self.fill(jamb, self.col(STONE, .20 + .05 * lv if th < 0 else .24, .01), ang=1.57)
        if glass:
            self.fill(inner, rgb('#1b1d42'), ang=1.57)
            mx0, mx1 = inner[0], inner[-1]
            # panes: a few catch the sky
            self.c.clip(inner)
            top = inner[:, 1].min(); bot = inner[:, 1].max(); x0 = inner[:, 0].min(); x1 = inner[:, 0].max()
            nx, ny = mull[0] + 1, mull[1] + 1
            for i in range(nx):
                for j in range(ny):
                    a0, a1 = x0 + (x1 - x0) * i / nx, x0 + (x1 - x0) * (i + 1) / nx
                    b0, b1 = top + (bot - top) * j / ny, top + (bot - top) * (j + 1) / ny
                    v = r.random()
                    cc = mix(rgb('#1b1d42'), rgb('#4a5590'), (.55 * r.random() if v < .55 else .12) * (.5 + .9 * lv) * (1.2 - .6 * j / ny))
                    self.c.fill([[a0, b0 + r.normal(0, .6)], [a1, b0 + r.normal(0, .6)], [a1, b1], [a0, b1]], cc, .95)
                    if lv > .4 and v < .18: self.c.line([[a0 + 1, b1 - 1], [a0 + (a1 - a0) * .5, b0 + 1]], 1.0, rgb('#8e9aca'), alpha=.6)
            for i in range(1, nx):
                xx = x0 + (x1 - x0) * i / nx + r.normal(0, .3)
                self.c.line([[xx, top - 2], [xx + r.normal(0, .5), (top + bot) / 2], [xx, bot + 2]], 1.3, rgb('#15163a'), (0, 0), .2, seed=self.k + i)
            for j in range(1, ny):
                yy = top + (bot - top) * j / ny
                self.c.line([[x0 - 1, yy], [(x0 + x1) / 2, yy + r.normal(0, .5)], [x1 + 1, yy]], 1.0, rgb('#15163a'), (0, 0), .2, seed=self.k + 20 + j)
            self.c.unclip()
            # shadow of the soffit on the glass
            self.c.clip(inner)
            self.c.fill(np.vstack([inner[:, :] + [0, -1], (inner + [0, (bot - top) * .16])[::-1]]), rgb('#12133a'), .35)
            self.c.unclip()
        else:
            self.fill(inner, rgb('#181a3c'), ang=1.57)
        # sill
        if sill:
            q = [(-w / 2 - .24, -.02), (w / 2 + .26, -.02), (w / 2 + .22, -.27), (-w / 2 - .2, -.27)]
            P = np.array([T(th + u / rr, z0 + z + r.normal(0, .015), rr + .16, ax) for u, z in q])
            v = lv + .10 - vdark
            sh = np.array([T(th + u / rr, z0 + z, rr, ax) for u, z in ((-w / 2 - .18, -.26), (w / 2 + .34, -.26), (w / 2 + .30, -.52), (-w / 2 - .05, -.46))])
            self.c.fill(sh, rgb('#1d1f46'), .5)
            self.fill(P, self.col(pal, v, .02), ang=0.)
            self.c.line(P[[0, 1]], 1.5, self.col(pal, min(1, v + .24)), (.1, .2), alpha=.95 if lv > .25 else .5, seed=self.k)
            if stain:
                for _ in range(r.integers(3, 7)):
                    u = r.uniform(-w / 2 - .2, w / 2 + .2); L = r.uniform(.5, 1.9)
                    q = np.array([T(th + (u + r.normal(0, .02) * k) / rr, z0 - .3 - L * k / 4, R(z0 - .3 - L * k / 4), ax) for k in range(5)])
                    self.c.line(q, r.uniform(1.2, 3.2), rgb('#23264e') if r.random() < .75 else rgb('#4a6476'), (.05, .7), .4, seed=self.k + _, alpha=r.uniform(.12, .34))
        if key is not None:
            self.windows.append(dict(key=key, outer=outer, inner=inner, jamb=jamb, th=th, c=inner.mean(0), sill=P if sill else None,
                                     ring=self.arch(th, z0, w, h, R, 0, ax=ax, grow=.5)))
        return outer, inner

    def stain_line(self, pts, w, col, a, seed=0):
        self.c.line(spl(np.array(pts), n=6), w, col, (.05, .7), .4, seed=seed, alpha=a)


def Rmain(z):
    return float(np.interp(z, [-1, 2, 4, 9.5, 15, 20], [7.2, 6.75, 6.55, 6.28, 6.08, 5.92]))


def Rgal(z): return 6.72 + .02 * (z - 20)


def Rdrum(z): return 5.28


def Rdome(z):
    return float(np.interp(z - .5, [25.9, 26.3, 26.9, 27.6, 28.4, 29.1, 29.8, 30.4, 30.9, 31.25, 31.4], [5.35, 5.6, 5.72, 5.66, 5.34, 4.82, 4.02, 3.05, 2.1, 1.35, 1.05]))


ZST = lambda th: np.interp(th, np.radians([-95, -72, -64, 36]), [7.7, 7.7, 7.3, .5])            # ramp height at the stair


def Rst(z): return 7.75 - .035 * max(0, z)


def paint_tower(seed=7):
    """returns premult RGBA full frame, window dicts"""
    p = Painter(seed); r = p.r; c = p.c
    # ---------------------------------------------------------------- dome (behind gallery cornice)
    dome(p)
    # ---------------------------------------------------------------- drum + shutters
    p.base_shell(Rdrum, 23.2, 26.5, vf=lambda t, z: -.06)
    p.stones(Rdrum, 23.2, 26.4, hrow=(.5, .8), vf=lambda t, z: -.03 - .10 * smooth(25.6, 26.4, z))
    for i, th in enumerate(np.arange(-80, 84, 18.0)):
        shutter(p, th + r.normal(0, 1), i)
    p.band(Rdrum, 26.2, 26.6, .36, shadow=.55, dv=.08)
    # ---------------------------------------------------------------- main shaft
    vf = lambda t, z: -.16 * smooth(18.6, 20.0, z) + .04 * smooth(3, 0, z)
    p.base_shell(Rmain, -1.2, 20.1, vf=vf)
    p.stones(Rmain, -1.2, 4.0, hrow=(.62, .95), wr=(1.1, 2.3), vf=vf)
    p.stones(Rmain, 4.0, 9.5, hrow=(.46, .72), vf=vf)
    p.stones(Rmain, 9.5, 20.0, hrow=(.40, .62), vf=vf)
    plaster(p, seed + 3)
    streaks(p)
    p.band(Rmain, 9.1, 9.55, .26)
    p.band(Rmain, 14.75, 15.2, .24)
    # ---------------------------------------------------------------- side turret
    turret(p)
    # windows, storey by storey
    for th, z0 in ((-30, 5.3), (27, 5.0)):
        p.window(th, z0, .95, 2.1, Rmain, mull=(1, 2))
    for k, (th, z0) in enumerate(((-49, 10.35), (-20, 10.5), (9, 10.72), (37, 10.95))):
        p.window(th, z0, 1.45, 3.15, Rmain, key=k, depth=.6, trim=.4, mull=(1, 4))
    for th, z0 in ((-22, 15.9), (4, 16.1), (29, 16.0)):
        p.window(th, z0, .8, 2.5, Rmain, mull=(0, 4), trim=.3)
    p.window(53, 16.0, .7, 1.9, Rmain, mull=(0, 2), trim=.26)
    # ---------------------------------------------------------------- gallery on corbels
    gallery(p)
    limb_rim(p, Rmain, 8.2, 14.6); limb_rim(p, Rmain, 15.2, 19.0); limb_rim(p, Rgal, 20.4, 22.9); limb_rim(p, Rdrum, 23.9, 26.1)
    # ---------------------------------------------------------------- balcony
    balcony(p)
    # ---------------------------------------------------------------- stair ramp round the base
    stair(p)
    limb_rim(p, Rst, -.6, 8.0, w=2.2)
    return p


def dome(p):
    r = p.r; c = p.c
    ths = np.radians(np.arange(-97, 97, 11.0)) + np.radians(2)
    zs = np.array([25.9, 26.6, 27.45, 28.25, 28.95, 29.6, 30.15, 30.6, 31.0, 31.4]) + .5
    miss = {(11, 2), (12, 3), (4, 5), (13, 4)}
    def dl(t, z):
        rr = Rdome(z); dr = (Rdome(z + .2) - Rdome(z - .2)) / .4
        nz = -dr / math.hypot(1, dr); nh = 1 / math.hypot(1, dr)
        return .10 + .84 * max(0., (-.93 * math.sin(t) - .20 * math.cos(t)) * nh + .28 * nz) ** 1.0 + .14 * float(smooth(1.0, 1.5, t))
    sil = np.vstack([[T(-math.pi / 2, z, Rdome(z)) for z in zs], [T(math.pi / 2, z, Rdome(z)) for z in zs[::-1]]])
    p.fill(sil, rgb('#161a38'))
    gore = r.normal(0, .035, len(ths))
    for i, (a, b) in enumerate(zip(ths[:-1], ths[1:])):
        for j, (u, v) in enumerate(zip(zs[:-1], zs[1:])):
            if b < -math.pi / 2 or a > math.pi / 2: continue
            aa, bb = max(a, -math.pi / 2), min(b, math.pi / 2)
            if bb - aa < .02: continue
            if (i, j) in miss:
                for q in (.2, .55, .9):                                     # bare battens in the hole
                    zz = u + (v - u) * q
                    c.line([T(aa, zz, Rdome(zz) - .22), T((aa + bb) / 2, zz + r.normal(0, .04), Rdome(zz) - .22), T(bb, zz, Rdome(zz) - .22)], 1.0, rgb('#2c2d5a'), (.05, .05), seed=i * 13 + j, alpha=.8)
                continue
            jz = lambda: r.normal(0, .03)
            ov = .06
            mth = (aa + bb) / 2
            q = [(aa + .003, u - ov + jz()), (mth, u - ov + jz()), (bb - .003, u - ov + jz()), (bb - .003, v + jz()), (mth, v + jz()), (aa + .003, v + jz())]
            P = np.array([T(t, z, Rdome(z) + .02) for t, z in q])
            vv = dl(mth, (u + v) / 2)
            sh = smooth(.2, .55, vv)
            vv = vv + gore[i] + r.normal(0, .012 + .022 * sh)
            cc = p.col(COPPER, vv, .008)
            k = r.random()
            if k < .10: cc = mix(cc, rgb('#43405f'), .30)
            elif k < .2: cc = mix(cc, rgb('#86aeb0'), .10)
            d = T(mth, v, Rdome(v)) - T(mth, u, Rdome(u))
            p.fill(P, cc, ang=math.atan2(d[1], d[0]))
            c.line(P[[0, 1, 2]], 1.1, p.col(COPPER, max(0, vv - .16)), (.2, .2), seed=p.k, alpha=.35 + .4 * float(sh))
            for _ in range(r.integers(0, 3)):
                t = r.uniform(aa, bb); L = r.uniform(.4, 1.6)
                z1 = v - r.uniform(0, .2); pts = [T(t + r.normal(0, .004), max(26.4, z1 - L * s_), Rdome(max(26.4, z1 - L * s_)) + .03) for s_ in (0, .35, .7, 1)]
                c.line(spl(np.array(pts), n=5), r.uniform(.8, 2.6), p.col(COPPER, np.clip(vv + r.choice([.13, -.10, .16]), 0, 1)), (.1, .7), .4, seed=p.k + _, alpha=r.uniform(.18, .45))
    for i, a in enumerate(ths):
        if abs(a) > math.pi / 2: continue
        zz = np.linspace(26.5, 31.8, 14)
        L = np.array([T(a + r.normal(0, .004), z, Rdome(z) + .07) for z in zz])
        v = dl(a, 28.6)
        c.line(spl(L, n=4) + [.9, .4], 1.7, p.col(COPPER, max(0, v - .18)), (.05, .1), .3, seed=i, alpha=.6)
        c.line(spl(L, n=4), 1.2, p.col(COPPER, min(1, v + .14)), (.05, .2), .5, seed=i + 50, alpha=.55 + .3 * float(v > .4))
    # moonlit contour, broken by the plates
    zz = np.linspace(26.6, 31.6, 24)
    L = np.array([T(-math.pi / 2 + .05, z, Rdome(z) + .03) for z in zz])
    for k0 in range(0, 22, 4):
        c.line(spl(L[k0:k0 + 4], n=4) + [1.0, .3], 2.2, p.col(COPPER, .98), (.1, .25), .4, seed=k0, alpha=.95)
    # THE WOUND: a torn hole near the crown, its ribs bare, cracks running down to the drum
    RD = lambda z: Rdome(z) + .05
    hole = [(-.46, 30.35), (-.30, 30.62), (-.10, 30.80), (.06, 30.35), (.20, 30.45), (.36, 30.30), (.30, 29.72), (.50, 29.35), (.40, 28.95), (.17, 28.82), (.06, 28.30), (-.10, 28.72), (-.22, 29.12), (-.44, 29.22), (-.36, 29.75)]
    Hp = np.array([T(t, z, RD(z)) for t, z in hole])
    p.fill(Hp, rgb('#0e112e'))
    c.clip(Hp)
    for a in (-.34, -.12, .10, .32):                                        # wooden ribs of the dome
        zz = np.linspace(28.0, 31.0, 9); L = np.array([T(a + r.normal(0, .004), z, Rdome(z) - .32) for z in zz])
        c.line(spl(L, n=4), 3.4, rgb('#2a2a55'), (0, 0), .2, seed=int(a * 90) + 200); c.line(spl(L, n=4) - [1.1, 0], 1.3, rgb('#6a71a0'), (0, 0), .4, seed=int(a * 90) + 201, alpha=.9)
    for z in (28.6, 29.15, 29.75, 30.3):
        L = np.array([T(t, z + .04 * math.sin(t * 9 + z), Rdome(z) - .26) for t in np.linspace(-.55, .55, 7)])
        c.line(spl(L, n=4), 2.0, rgb('#24244e'), (0, 0), .3, seed=int(z * 9)); c.line(spl(L, n=4) - [0, .9], .8, rgb('#565d90'), (.1, .1), .5, seed=int(z * 9) + 1, alpha=.8)
    c.unclip()
    lip = Hp[[3, 4, 5, 6, 7, 8, 9, 10, 11]]
    for k0 in range(0, 8, 2):
        c.line(rough_line(lip[k0:k0 + 3], r), 2.3, p.col(COPPER, .92), (.15, .3), .5, seed=k0 + 3, alpha=.95)
    c.line(rough_line(Hp[[14, 0, 1, 2]], r) + [.5, 1.2], 1.6, p.col(COPPER, .04), (.1, .2), .4, seed=4, alpha=.8)
    def crackline(pts, w0):
        n = len(pts); ww = w0 * np.linspace(1, .12, n) * r.uniform(.8, 1.25, n)
        A_ = np.array([T(t - w / 2 / Rdome(z), z + r.normal(0, .04), RD(z)) for (t, z), w in zip(pts, ww)]); B_ = np.array([T(t + w / 2 / Rdome(z), z + r.normal(0, .05), RD(z)) for (t, z), w in zip(pts, ww)])
        p.fill(np.vstack([A_, B_[::-1]]), rgb('#0f1230'))
        c.line(rough_line(B_, r) + [.5, .2], 1.5, p.col(COPPER, .80), (.05, .6), .5, seed=n + int(w0 * 50), alpha=.9)
    crackline([(.08, 28.6), (-.05, 28.05), (.16, 27.55), (.04, 27.1), (.18, 26.75), (.14, 26.4)], .52)
    crackline([(.44, 29.15), (.60, 28.75), (.55, 28.2), (.72, 27.75), (.68, 27.3)], .36)
    crackline([(-.12, 30.7), (-.05, 31.0), (-.18, 31.35)], .28)
    crackline([(-.42, 29.3), (-.58, 28.9), (-.54, 28.45), (-.66, 28.1)], .24)
    crackline([(-.02, 28.0), (-.18, 27.7), (-.20, 27.35)], .20)
    for (t, z, sz, a) in ((-.50, 29.55, .66, -.5), (.40, 28.75, .62, .5), (.50, 27.3, .46, .95), (.16, 30.35, .5, -.2)):
        o = T(t, z, Rdome(z) + .1); s_ = sz * PX
        q = np.array([[0, 0], [s_, -.12 * s_], [.95 * s_, .75 * s_], [.1 * s_, .8 * s_]]) @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) + o
        c.fill(q + [1.5, 2.5], rgb('#10132c'), .6)
        p.fill(q, p.col(COPPER, .55 if t < 0 else .34)); c.line(q[[3, 0, 1]], 1.3, p.col(COPPER, .92 if t < 0 else .70), (.1, .2), seed=9)
    # plates lifting off the silhouette
    for (t, z, dx, dy, v) in ((1.36, 29.2, 7, -3, .40), (1.45, 27.6, 5, -5, .42), (-1.38, 30.0, -5, -4, .9)):
        o = T(t, z, Rdome(z)); q = o + np.array([[-4, 3], [dx, dy], [dx + 2, dy + 5], [-2, 8]])
        p.fill(q, p.col(COPPER, v)); c.line(q[[0, 1]], 1.2, p.col(COPPER, min(1, v + .3)), (.1, .2), seed=int(z))
    # cupola
    o0, o1 = 31.85, 32.95
    for a in np.radians([-70, -25, 20, 64]):
        q = np.array([T(a - .16, o0, 1.1), T(a + .16, o0, 1.1), T(a + .13, o1, 1.05), T(a - .13, o1, 1.05)])
        p.fill(q, p.col(STONE, float(light(a)) + .05))
    bl = T(0, 32.75, 0)
    c.line([bl, bl + [0, 5]], 1.2, rgb('#2a3158')); c.fill(spl(bl + np.array([[-2.5, 4], [2.5, 4], [4, 9], [6.5, 13], [-6.5, 13], [-4, 9]]), closed=True, corners=(3, 4)), rgb('#28305a')); c.line(spl(bl + np.array([[-2.5, 4.5], [-4, 9], [-6, 12.5]])), 1.2, rgb('#8f9ac4'), (.2, .2))
    q = np.array([T(-1.5, 31.8, 1.4), T(-.6, 31.72, 1.45), T(.6, 31.72, 1.45), T(1.5, 31.8, 1.4), T(1.5, 32.1, 1.35), T(0, 32.0, 1.4), T(-1.5, 32.1, 1.35)])
    p.fill(q, p.col(COPPER, .42)); c.line(q[[0, 6, 5]], 1.2, p.col(COPPER, .85), (.1, .4))
    cap = np.array([T(-1.55, 32.9, 1.4), T(-.4, 32.8, 1.45), T(.6, 32.8, 1.45), T(1.55, 32.9, 1.4), T(1.4, 33.25, 1.05), T(1.2, 33.6, .5), T(0, 33.9, .16), T(-1.2, 33.6, .55), T(-1.4, 33.25, 1.1)])
    cp = spl(cap, closed=True, corners=(0, 3, 6), n=6)
    p.fill(cp, p.col(COPPER, .26)); c.shade(spl(np.vstack([cap[[0, 8, 7, 6]], T(-.3, 33.4, .4)[None], T(-.5, 32.85, .9)[None]]), closed=True, n=5), p.col(COPPER, .72), within=cp)
    f0 = T(0, 33.8, 0); f1 = T(0, 35.3, 0) + [3, 0]
    fin = spl(np.array([f0, (f0 + f1) / 2 + [-1.5, 0], f1]), n=8)
    c.line(fin, 2.6, rgb('#2a3350'), (0, .7), .2); c.line(fin - [.8, 0], .9, rgb('#a9b6d6'), (.1, .6), .2)
    b = f0 + [0, -7]
    c.fill(spl(b + np.array([[-3, 0], [0, -3], [3, 0], [0, 3]]), closed=True), rgb('#50708a')); c.fill(spl(b + np.array([[-3, 0], [-1, -2.6], [1, -2], [-1, 1]]), closed=True), rgb('#b7c8dc'))
    bb = f1 + [0, 4]
    bird = bb + np.array([[-11, -1], [-5, -3], [-2, -9], [3, -11], [1, -5], [8, -4], [13, -8], [10, -1], [3, 1.5], [-5, 1.5]]) * .85
    c.fill(spl(bird, closed=True, corners=(0, 3, 6)), rgb('#2b3352')); c.line(spl(bird[:4], n=5), 1.0, rgb('#b3bedc'), (.2, .2))


def shutter(p, thd, i):
    """closed louvred shutters between stone piers; one hangs, some slats are lost"""
    r = p.r; c = p.c
    th = math.radians(thd); R = 5.36; w = 1.16; z0, h = 23.75, 2.25
    lv = float(light(th))
    tilt = (0, 0, .03, 0, -.10, 0, 0, .05, 0, 0)[i % 10]
    def Q(u, z): return T(th + (u + tilt * (z - z0 - h)) / R, z + tilt * u * .8, R)
    hs = h - w * .42
    fr = [Q(-w / 2 - .12, z0 - .08), Q(w / 2 + .12, z0 - .08), Q(w / 2 + .12, z0 + hs)] + [Q((w / 2 + .12) * math.cos(a), z0 + hs + (w * .42 + .1) * math.sin(a)) for a in (.6, 1.2, 1.57, 1.95, 2.5)] + [Q(-w / 2 - .12, z0 + hs)]
    p.fill(np.array(fr), rgb('#171a3c'))
    wood = [['#232850', '#2d345f', '#415076', '#6f83a6', '#b0c0d8'], ['#272550', '#34325f', '#4a4c78', '#767ba6', '#b4b8d8']][i % 2]
    for s in (-1, 1):
        u0, u1 = (.025, w / 2 - .02) if s > 0 else (-w / 2 + .02, -.025)
        if i == 4 and s > 0: u0 += .14; u1 += .10
        vbase = lv * .92 + .03 + r.normal(0, .02)
        body = [Q(u0, z0), Q(u1, z0), Q(u1, z0 + hs)] + ([Q(w / 2 * math.cos(a), z0 + hs + w * .40 * math.sin(a)) for a in (.5, 1.0, 1.45)] if s > 0 else [Q(u1, z0 + h - .06)]) + ([Q(u0, z0 + h - .02)] if s > 0 else [Q(w / 2 * math.cos(a), z0 + hs + w * .40 * math.sin(a)) for a in (1.7, 2.15, 2.65)])
        if s < 0: body = [Q(u0, z0), Q(u1, z0), Q(u1, z0 + h - .02)] + [Q(w / 2 * math.cos(a), z0 + hs + w * .40 * math.sin(a)) for a in (1.7, 2.15, 2.65)]
        else: body = [Q(u0, z0), Q(u1, z0)] + [Q(w / 2 * math.cos(a), z0 + hs + w * .40 * math.sin(a)) for a in (.5, 1.0, 1.45)] + [Q(u0, z0 + h - .02)]
        body = np.array(body)
        p.fill(body, p.col(wood, vbase, .01), ang=0.)
        c.clip(body)
        n = 8
        for j in range(n):
            za = z0 + .12 + (h - .2) * j / n; zb = za + (h - .2) / n
            lost = (i, s, j) in ((3, 1, 3), (5, -1, 5), (7, 1, 2), (2, -1, 6), (3, 1, 4))
            if lost: c.fill([Q(u0 + .07, za), Q(u1 - .07, za), Q(u1 - .07, zb), Q(u0 + .07, zb)], rgb('#12143a')); continue
            c.line([Q(u0 + .06, za + r.normal(0, .02)), Q((u0 + u1) / 2, za), Q(u1 - .06, za + r.normal(0, .02))], 1.5, p.col(wood, max(0, vbase - .26), .02), (.05, .05), seed=p.k + j, alpha=.85)
            c.line([Q(u0 + .06, zb - .07), Q(u1 - .06, zb - .07)], 1.1, p.col(wood, min(1, vbase + .14), .03), (.1, .3), seed=p.k + j + 9, alpha=.75 if lv > .28 else .35)
        c.unclip()
        c.line(np.vstack([body, body[:1]]), 1.2, p.col(wood, max(0, vbase - .15)), (0, 0), .2, seed=p.k + 9, alpha=.8)
        if s < 0 and lv > .3: c.line(body[[0, -1]], 1.2, p.col(wood, min(1, vbase + .3)), (.1, .2), seed=p.k + 10, alpha=.9)
    # arch stones
    for a0 in np.linspace(.25, 2.9, 6)[:-1]:
        a1 = a0 + .46
        q = np.array([Q((w / 2 + .1) * math.cos(a0), z0 + hs + (w * .42 + .08) * math.sin(a0)), Q((w / 2 + .1) * math.cos(a1), z0 + hs + (w * .42 + .08) * math.sin(a1)), Q((w / 2 + .34) * math.cos(a1), z0 + hs + (w * .42 + .3) * math.sin(a1)), Q((w / 2 + .34) * math.cos(a0), z0 + hs + (w * .42 + .3) * math.sin(a0))])
        p.fill(q, p.col(STONE, lv + .05, .03))


def gallery(p):
    r = p.r; c = p.c
    # deep shadow under the overhang
    ths = np.radians(np.linspace(-92, 92, 40))
    P = np.vstack([[T(t, 20.15, 6.75) for t in ths], [T(t, 18.6 + .2 * math.sin(t * 9) - .5 * max(0, -math.sin(t)), Rmain(19)) for t in ths[::-1]]])
    c.fill(P, rgb('#1a1c42'), .62)
    # corbels
    for t in np.radians(np.arange(-86, 87, 8.6)):
        t += r.normal(0, .008)
        lv = float(light(t))
        w = .34
        f = lambda u, z, rr: T(t + u / rr, z, rr)
        face = np.array([f(-w / 2, 20.1, 6.74), f(w / 2, 20.1, 6.74), f(w / 2, 19.55, 6.6), f(w / 2, 19.05, 6.1), f(-w / 2, 19.05, 6.1), f(-w / 2, 19.55, 6.6)])
        p.fill(spl(face, closed=True, corners=(0, 1, 3, 4), n=5), p.col(STONE, lv * .8 + .02, .04))
        if lv > .25: c.line(face[[0, 5, 4]], 1.2, p.col(STONE, min(1, lv + .2)), (.1, .5), seed=p.k, alpha=.85)
        c.line(face[[1, 2, 3]], 1.3, rgb('#1b1d44'), (.1, .5), seed=p.k + 2, alpha=.6)
    p.base_shell(Rgal, 20.05, 23.2, vf=lambda t, z: -.02, nz=4)
    p.stones(Rgal, 20.45, 23.1, hrow=(.42, .6), vf=lambda t, z: .0)
    p.band(Rgal, 20.0, 20.45, .10, shadow=0, dv=.05)
    for t in np.arange(-81, 82, 13.5):
        p.window(t + r.normal(0, .7), 20.85, .78, 1.75, Rgal, mull=(0, 2), trim=.22, depth=.45, sill=False, stain=False, trimv=.05)
    p.band(Rgal, 23.0, 23.5, .30, shadow=.45, dv=.10)
    # broken parapet merlons on top (stones standing up, some gone)
    for t in np.radians(np.arange(-90, 91, 7.0)):
        if r.random() < .30: continue
        hh = r.uniform(.28, .62); w = r.uniform(.5, .78); rr = 6.95
        q = np.array([T(t - w / 2 / rr, 23.48, rr), T(t + w / 2 / rr, 23.48, rr), T(t + w / 2 / rr * r.uniform(.8, 1), 23.48 + hh, rr), T(t - w / 2 / rr * r.uniform(.8, 1), 23.48 + hh * r.uniform(.75, 1.05), rr)])
        v = float(light(t)) + .04 + r.normal(0, .05)
        p.fill(q, p.col(STONE, v), ang=0.)
        c.line(q[[0, 3, 2]], 1.2, p.col(STONE, min(1, v + .26)), (.1, .4), seed=p.k, alpha=.9 if t < -.1 else .45)


def balcony(p):
    r = p.r; c = p.c
    a0, a1 = -86, -33; z = 15.35; ro = 7.55
    ths = np.radians(np.linspace(a0, a1, 14))
    # door behind
    p.window(-58, 15.6, 1.15, 2.75, Rmain, mull=(1, 0), trim=.32, sill=False, depth=.5, stain=False)
    # cast shadow on the wall below
    P = np.vstack([[T(t, z - .1, Rmain(z)) for t in ths], [T(t + .10, z - 2.5 + .6 * abs(t + 1.0), Rmain(z - 2)) for t in ths[::-1]]])
    c.fill(P, rgb('#1c1e46'), .5)
    for t in np.radians([-79, -60, -41]):
        w = .42
        f = lambda u, zz, rr: T(t + u / rr, zz, rr)
        lv = float(light(t))
        side = np.array([f(w / 2, z, 7.2), f(w / 2, z - .5, 7.0), f(w / 2, z - 1.0, 6.5), f(w / 2, z - 1.75, 6.05), f(w / 2, z, 6.05)])
        p.fill(spl(side, closed=True, corners=(0, 3, 4), n=5), p.col(STONE, .12 + .12 * lv))
        face = np.array([f(-w / 2, z, 7.2), f(w / 2, z, 7.2), f(w / 2, z - .5, 7.0), f(w / 2, z - 1.0, 6.5), f(w / 2, z - 1.75, 6.05), f(-w / 2, z - 1.75, 6.05), f(-w / 2, z - 1.0, 6.5), f(-w / 2, z - .5, 7.0)])
        p.fill(spl(face, closed=True, corners=(0, 1, 4, 5), n=5), p.col(STONE, lv * .85))
        c.line(face[[0, 7, 6, 5]], 1.5, p.col(STONE, min(1, lv + .18)), (.1, .4), seed=p.k)
    under = np.vstack([[T(t, z, ro) for t in ths], [T(t, z, 6.05) for t in ths[::-1]]])
    p.fill(under, p.col(STONE, .10))
    # slab edge stones
    th = math.radians(a0)
    while th < math.radians(a1):
        b = min(th + r.uniform(.8, 1.3) / ro, math.radians(a1))
        q = np.array([T(th + .004, z, ro), T(b - .004, z, ro), T(b - .004, z + .38 + r.normal(0, .02), ro), T(th + .004, z + .38, ro)])
        v = float(light((th + b) / 2)) + .12 + r.normal(0, .03)
        p.fill(q, p.col(STONE, v), ang=0.); c.line(q[[3, 2]], 1.5, p.col(STONE, min(1, v + .2)), (.1, .1), seed=p.k)
        c.line(q[[0, 1]], 1.0, p.col(STONE, .12), (.1, .1), seed=p.k + 1, alpha=.8)
        th = b
    # balusters + rail (some lost)
    zr = z + .38
    for j, t in enumerate(np.radians(np.arange(a0 + 1, a1 + 1, 5.2))):
        if j in (4, 7): continue
        lv = float(light(t))
        tp = 1.02 if j % 5 else 1.18
        w = .13 if j % 5 else .26
        q = np.array([T(t - w / ro, zr, ro - .12), T(t + w / ro, zr, ro - .12), T(t + w * .55 / ro, zr + tp * .5, ro - .12), T(t + w * .9 / ro, zr + tp, ro - .12), T(t - w * .9 / ro, zr + tp, ro - .12), T(t - w * .55 / ro, zr + tp * .5, ro - .12)])
        p.fill(spl(q, closed=True, corners=(0, 1, 3, 4), n=4), p.col(STONE, lv - .06, .05), ang=1.57)
        c.line(q[[0, 5, 4]], 1.1, p.col(STONE, min(1, lv + .22)), (.1, .2), seed=p.k, alpha=.9)
    rl = np.array([T(t, zr + 1.02 + .05 * math.sin(t * 7), ro - .12) for t in np.radians(np.linspace(a0 + .5, a1, 16))])
    c.line(spl(rl, n=5), 4.6, p.col(STONE, .62, .0), (0.02, .02), .12, seed=5); c.line(spl(rl, n=5) + [0, 1.5], 1.6, p.col(STONE, .26, 0), (.05, .05), .2, seed=6, alpha=.9)
    c.line(spl(rl, n=5) + [0, -1.6], 1.5, p.col(STONE, .97, 0), (.1, .15), .4, seed=7)


def turret(p):
    r = p.r; c = p.c
    ta = math.radians(66); ax = (6.35 * math.sin(ta), -6.0 * math.cos(ta))
    R = lambda z: float(np.interp(z, [-1, 2, 15], [2.3, 2.1, 1.95]))
    vf = lambda t, z: -.48 * max(0., float(light(t)) - .16) * (1 - .35 * smooth(11, 14, z)) - .02
    # its shadow on the shaft
    q = np.vstack([[T(math.radians(36 + 2 * math.sin(z)), z, Rmain(z)) for z in np.linspace(-1, 14.5, 10)], [T(math.radians(58), z, Rmain(z)) for z in np.linspace(14.5, -1, 10)]])
    c.fill(spl(q, closed=True, corners=(0, 9, 10, 19), n=3), rgb('#1c1e46'), .45)
    p.base_shell(R, -1.2, 14.0, ax=ax, vf=vf, nth=16)
    p.stones(R, -1.2, 13.8, ax=ax, hrow=(.42, .66), wr=(1.2, 2.2), vf=vf)
    for z0, t in ((3.6, -34), (7.4, -12), (11.0, 6)):
        p.window(t, z0, .32, 1.15, R, ax=ax, mull=(0, 0), trim=.2, depth=.3, stain=True, vdark=.26)
    p.band(R, 5.9, 6.2, .12, ax=ax, shadow=.2, w=(.5, .9), vf=vf); p.band(R, 9.6, 9.9, .12, ax=ax, shadow=.2, w=(.5, .9), vf=vf)
    p.band(R, 13.6, 14.1, .28, ax=ax, shadow=.5, w=(.45, .8), vf=vf)
    # conical roof, bent like an old hat, tile rows
    rows = 14; z0, z1 = 14.0, 19.6
    rr = lambda q: 2.65 * (1 - q) ** 1.2 + .05
    bend = lambda q: .75 * q ** 2.2
    for j in range(rows):
        qa, qb = j / rows, (j + 1.25) / rows
        n = max(4, int(15 * (1 - qa)) + 3)
        tt = np.linspace(-math.pi / 2, math.pi / 2, n + 1) + (j % 2) * .5 * math.pi / n
        for a, b in zip(tt[:-1], tt[1:]):
            a = max(a, -math.pi / 2); b = min(b, math.pi / 2)
            if b - a < .05: continue
            f = lambda t, q: T(t, z0 + (z1 - z0) * min(q, 1), rr(min(q, .999)), (ax[0] + bend(min(q, 1)), ax[1]))
            P = np.array([f(a, qa) + [0, r.normal(0, .4)], f((a + b) / 2, qa) + [0, 1.1], f(b, qa), f(b, qb), f(a, qb)])
            v = float(light((a + b) / 2)) * (.45 + .3 * qa) - .0 + r.normal(0, .035)
            p.fill(P, p.col(['#21244a', '#2d3560', '#434f7c', '#6a7aa6', '#a3b4d4', '#c8d4e8'], v, .03))
            c.line(P[:3], 1.0, rgb('#191b3e'), (.1, .1), seed=p.k, alpha=.7)
            if v > .36: c.line(P[[0, 4]], 1.0, p.col(STONE, min(1, v + .3)), (.1, .3), seed=p.k + 1, alpha=.7)
    tip = T(0, z1, 0, (ax[0] + bend(1), ax[1]))
    c.line([tip + [0, 3], tip + [1, -6], tip + [2.5, -13]], 2.2, rgb('#27305a'), (0, .7)); c.line([tip + [-.8, 3], tip + [.3, -6], tip + [1.8, -12]], .8, rgb('#9fb0d4'), (.1, .6))


def stair(p):
    r = p.r; c = p.c
    a0, a1 = -91, 36
    th = np.radians(np.linspace(a0, a1, 30))
    # shadow of the ramp on the shaft above its parapet
    P = np.vstack([[T(t, ZST(t) + 2.1 + .12 * math.sin(t * 8), Rmain(ZST(t) + 1.5)) for t in th], [T(t, ZST(t) + .6, Rmain(ZST(t))) for t in th[::-1]]])
    c.fill(P, rgb('#1d1f47'), .42)
    vf = lambda t, z: .02
    # wall of the ramp, stones cut to the slope
    for a, b in zip(th[:-1], th[1:]):
        q = np.array([T(a - .002, -1.4, Rst(-1)), T(b + .004, -1.4, Rst(-1)), T(b + .004, ZST(b) + .95, Rst(ZST(b))), T(a - .002, ZST(a) + .95, Rst(ZST(a)))])
        c.fill(q, p.col(STONE, float(light((a + b) / 2)) * .9 - .05, .01))
    p.stones(Rst, -1.4, 10.2, a0, a1, hrow=(.5, .78), wr=(1.1, 2.2), skip=lambda t, z: z > ZST(t) + .62, vf=lambda t, z: .03 * smooth(3, 0, z))
    # arches beneath
    for t in (-67, -42, -20, 0):
        tt = math.radians(t); hh = max(1.6, ZST(tt) - 1.3)
        p.window(t, -.7 if hh < 4 else .1, 1.75, min(hh * .86, 5.0), Rst, glass=False, trim=.42, sill=False, depth=1.1, stain=False)
        # inner pier light
    # coping: lit, worn, broken blocks following the slope
    t = math.radians(a0)
    while t < math.radians(a1):
        b = min(t + r.uniform(.8, 1.35) / 7.7, math.radians(a1)); rr = 7.82
        q = np.array([T(t + .004, ZST(t) + .62, rr), T(b - .004, ZST(b) + .62, rr), T(b - .004, ZST(b) + 1.0 + r.normal(0, .05), rr), T(t + .004, ZST(t) + 1.0 + r.normal(0, .05), rr)])
        v = float(light((t + b) / 2)) + .10 + r.normal(0, .04)
        if r.random() > .06:
            p.fill(q, p.col(STONE, v), ang=math.atan2(*(q[1] - q[0])[::-1]))
            c.line(q[[0, 3, 2]], 1.6, p.col(STONE, min(1, v + .28)), (.05, .15), seed=p.k, alpha=.95 if t < 0 else .6)
            c.line(q[[0, 1]], 1.2, p.col(STONE, max(0, v - .3)), (.1, .1), seed=p.k + 1, alpha=.8)
        t = b


def plaster(p, seed):
    """old limewash left in quiet patches"""
    r = np.random.default_rng(seed); c = p.c
    for (th, z, w, h) in ((-.35, 17.4, 1.7, 1.6), (.42, 13.4, 1.2, 2.3), (-.70, 12.8, .7, 1.8), (.06, 7.6, 1.8, 1.0), (-.95, 17.8, .9, 1.4), (.20, 18.4, 1.2, .7)):
        m = 16; q = []
        for k in range(m):
            a = 2 * math.pi * k / m; f = r.uniform(.72, 1.15)
            q.append(T(th + w * math.cos(a) * f / Rmain(z), z + h * math.sin(a) * f, Rmain(z) + .02))
        v = float(light(th)) + .05
        c.fill(spl(np.array(q), closed=True, n=4), p.col(STONE, v, .02), .72)
        c.line(spl(np.array(q)[:7], n=4), 1.0, p.col(STONE, max(0, v - .2)), (.3, .3), seed=seed, alpha=.4)


def streaks(p):
    r = p.r
    for (z, n, L) in ((14.8, 34, 2.2), (9.2, 30, 2.4), (19.4, 22, 2.0)):
        for _ in range(n):
            t = r.uniform(-1.5, 1.5); ll = L * r.uniform(.25, 1)
            q = np.array([T(t + r.normal(0, .004), z - ll * s, Rmain(z - ll * s)) for s in np.linspace(0, 1, 5)])
            p.c.line(q, r.uniform(1.0, 4.0), rgb('#22254e') if r.random() < .7 else rgb('#56708a'), (.02, .75), .4, seed=_, alpha=r.uniform(.08, .28))


def gouache(p, seed=3):
    """brush pass over the tower cel: every stone keeps its edge, gets its own strokes"""
    f = p.c.rgba()
    a = f[..., 3]
    straight = f[..., :3] / np.maximum(a, 1e-3)[..., None]
    base = np.where(a[..., None] > .02, straight, rgb('#3b4272'))
    m = (a > .5).astype(f32)
    ang = p.ang + fbm(p.h, p.w, (30,), None, seed) * .12
    out = repaint(base, (4.2, 2.3), seed=seed, angle=ang, lab=p.lab, mask=m, thr=(.0, .03), density=1.15, aspect=3.2, jit=(.003, .008, .016), op=.6, dry=.35, curve=.25, bristle=.07)
    k = .8
    out = base * (1 - k) + out * k
    n = fbm(p.h, p.w, (26, 9, 3.2), (1, .6, .4), seed + 2)
    out = np.clip(out * (1 + .022 * n[..., None]), 0, 1)
    L = np.zeros((H, W, 4), f32)
    R = np.dstack([out * a[..., None], a]).astype(f32)
    gx, gy = np.meshgrid(np.arange(p.w), np.arange(p.h))                          # nothing on a hand-painted tower is ruled
    R = cv2.remap(R, (gx + fbm(p.h, p.w, (70, 24, 8), (1, .7, .45), seed + 31) * .62).astype(f32), (gy + fbm(p.h, p.w, (70, 24, 8), (1, .7, .4), seed + 32) * .5).astype(f32), cv2.INTER_CUBIC)
    L[p.y0:p.y0 + p.h, p.x0:p.x0 + p.w] = np.clip(R, 0, 1)
    return L


def ivy_points(p, seed=21):
    """leaf positions on the wall: stems climb the shadow-side front, strands hang from gallery and balcony"""
    r = np.random.default_rng(seed)
    X, Y, V = [], [], []
    stems = []
    op_ = cv2.dilate(p.open, np.ones((5, 5), np.uint8))
    def walk(th, z, dz, n, drift, spread, dens, taper=.6, up=1):
        path = []
        for k in range(n):
            q = k / n
            th += r.normal(drift, .020); z += up * dz * r.uniform(.6, 1.2)
            path.append((th, z, q))
            m = r.poisson(dens * (1 - taper * q))
            sp = spread * (1 - .55 * q)
            for _ in range(m):
                du = r.normal(0, sp); dv = r.normal(0, sp * .8)
                rr = Rmain(z)
                if z > 19.4: rr = 6.3
                t = th + du / rr; zz = z + dv
                if abs(t) > 1.52: continue
                pt = T(t, zz, (Rmain(zz) if zz < 19.6 else 6.75) + .12)
                iy, ix = int(pt[1] - p.y0), int(pt[0] - p.x0)
                if 0 <= iy < p.h and 0 <= ix < p.w and op_[iy, ix] and r.random() < .975: continue
                lv = float(light(t))
                v = .10 + .50 * max(0, lv - .10) + .26 * (-du * .6 + dv) / sp * .5 + r.normal(0, .05)
                X.append(pt[0] + r.normal(0, .8)); Y.append(pt[1] + r.normal(0, .8)); V.append(np.clip(v, 0, .95))
        stems.append(path)
    # climbing masses (above the ramp)
    for (t, z, n, dr, sp, de) in ((.30, 1.6, 44, .0015, .62, 26), (.46, 1.2, 46, .004, .70, 26), (.62, 1.0, 40, .006, .62, 22), (.10, 2.6, 26, -.004, .5, 18),
                                  (.36, 6.0, 40, .010, .42, 16), (.22, 8.0, 34, -.006, .30, 11), (.55, 9, 34, .006, .4, 14), (.72, 12, 28, -.004, .32, 12), (-.10, 4.0, 16, -.012, .34, 10)):
        walk(t, z, .29, n, dr, sp, de)
    # strands hanging from the gallery corbels
    for t in (.28, .40, .52, .66, .80, -.06, .95):
        walk(t, 20.2, .22, int(r.uniform(8, 22)), .0, .20, 10, .5, up=-1)
    return np.array(X), np.array(Y), np.array(V), stems


def ivy_layer(p, seed=21):
    cv = Canvas(H, W, layer=True)
    X, Y, V, stems = ivy_points(p, seed)
    # shadow the leaves throw on the wall
    plants.ivy(cv, X[::2] + 2.2, Y[::2] + 2.2, V[::2] * 0, seed=seed + 1, size=3.2, pal=('#161a3a', '#191d40'))
    plants.ivy(cv, X, Y, V, seed=seed, size=2.9)
    # strands and turret
    r = np.random.default_rng(seed + 4)
    L = cv.layer()
    return L


def balcony_ivy(seed=5):
    """a few strands trailing from the balcony and the landing on the moon side"""
    r = np.random.default_rng(seed); cv = Canvas(H, W, layer=True)
    X, Y, V = [], [], []
    for (t, z, rr, n) in ((-80, 15.5, 7.5, 13), (-72, 15.4, 7.5, 8), (-47, 15.4, 7.5, 16), (-39, 15.4, 7.5, 9), (-62, 15.4, 7.5, 5), (-84, 8.2, 7.8, 14), (-78, 8.2, 7.8, 22), (-70, 8.0, 7.8, 11), (-50, 6.0, 7.7, 10), (-8, 3.9, 7.8, 9), (12, 2.6, 7.8, 8)):
        th = math.radians(t); zz = z
        for k in range(n):
            zz -= .17; th += r.normal(0, .006)
            for _ in range(r.integers(3, 7)):
                pt = T(th + r.normal(0, .014), zz + r.normal(0, .08), rr)
                X.append(pt[0]); Y.append(pt[1]); V.append(np.clip(.12 + .50 * float(light(th)) + r.normal(0, .14), 0, 1))
    plants.ivy(cv, X, Y, V, seed=seed, size=2.5)
    return cv.layer()
