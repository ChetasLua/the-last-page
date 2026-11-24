"""Balcony lane: masonry painted block by block.  Work happens at ss=2 in a ROI, then is reduced.

Every block is a hand-cut polygon (jittered courses, chipped corners) with its own pigment.  Strokes run along the
block.  Light is painted as separate overlay marks so that the sun can arrive stone by stone."""
import numpy as np, math, cv2
from lp.paint import f32, rgb, mix, smooth, ramp, fbm, Canvas, arcs, jit_color, lum


def cut(u0, u1, v0, v1, r, gap=.012, chip=.06, pchip=.16, wob=.007, n=4):
    """hand-cut rectangle outline in metric (u,v): slightly proud corners, nicked edges, sometimes a chipped corner"""
    u0, u1, v0, v1 = u0 + gap, u1 - gap, v0 + gap, v1 - gap
    cs = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    out = []
    for k in range(4):
        a = np.array(cs[k]); b = np.array(cs[(k + 1) % 4]); p = np.array(cs[k - 1])
        e1 = (b - a); L1 = np.linalg.norm(e1) + 1e-6; e0 = (p - a); L0 = np.linalg.norm(e0) + 1e-6
        c = chip * r.uniform(.12, .3) if r.random() > pchip else chip * r.uniform(.8, 1.9)
        c0 = min(c * r.uniform(.6, 1.5), L0 * .4); c1 = min(c * r.uniform(.6, 1.5), L1 * .4)
        A = a + e0 / L0 * c0; B = a + e1 / L1 * c1
        out.append(A)
        if c > chip * .5: out.append((A + B) / 2 + (a - (A + B) / 2) * r.uniform(-.1, .55))
        else: out.append((A + B) / 2 + (a - (A + B) / 2) * .45)
        out.append(B)
        nn = np.array([-e1[1], e1[0]]) / L1
        for t in np.linspace(0, 1, n + 2)[1:-1]:
            q = a + e1 * t
            if np.linalg.norm(q - a) > c1 * 1.2 and np.linalg.norm(q - b) > c1 * 1.2:
                out.append(q + nn * abs(r.normal(0, wob)) + (nn * wob * 3.2 if r.random() < .06 else 0))
    return np.array(out)


def courses(u0, u1, v0, v1, r, ch=(.34, .5), bw=(.5, 1.15), skip=None):
    """running-bond blocks [(u0,u1,v0,v1,row)] over a rectangle"""
    out = []; v = v0; row = 0
    while v < v1 - .05:
        h = r.uniform(*ch); t = min(v1, v + h)
        if v1 - t < ch[0] * .45: t = v1
        u = u0 - r.uniform(0, bw[1]) * .8
        while u < u1:
            w = r.uniform(*bw) * (1 if r.random() > .12 else .6)
            a, b = max(u0, u), min(u1, u + w)
            if b - a > .05 and not (skip and skip(a, b, v, t)): out.append((a, b, v + r.normal(0, .004), t + r.normal(0, .004), row))
            u += w
        v = t; row += 1
    return out


class Stone:
    def __init__(self, x0, y0, w, h, ss=2, ground='#8d7690'):
        self.ox, self.oy, self.w, self.h, self.ss = x0, y0, w, h, ss
        self.cv = Canvas(h * ss, w * ss, layer=True)
        self.cv.img[:] = rgb(ground)                 # straight colour under partial strokes (a black ground would grey them)
        self.lab = np.zeros((h * ss, w * ss), np.int32)
        self.blocks = []

    def P(self, pts):
        return (np.asarray(pts, float) - (self.ox, self.oy)) * self.ss

    def poly(self, pts, lab):
        q = np.round(self.P(pts) * 16).astype(np.int32)
        cv2.fillPoly(self.lab, [q], int(lab), cv2.LINE_8, 4)

    def fill(self, pts, col, alpha=1.0):
        """flat antialiased polygon"""
        m = np.zeros((self.h * self.ss, self.w * self.ss), np.uint8)
        q = np.round(self.P(pts) * 16).astype(np.int32)
        x, y, bw, bh = cv2.boundingRect(np.round(self.P(pts)).astype(np.int32))
        X0, Y0 = max(0, x - 2), max(0, y - 2); X1, Y1 = min(m.shape[1], x + bw + 2), min(m.shape[0], y + bh + 2)
        if X1 <= X0 or Y1 <= Y0: return self
        cv2.fillPoly(m, [q], 255, cv2.LINE_AA, 4)
        a = (m[Y0:Y1, X0:X1].astype(f32) / 255. * alpha)
        c = rgb(col) if isinstance(col, str) else np.asarray(col, f32)
        im = self.cv.img[Y0:Y1, X0:X1]; im += (c - im) * a[..., None]
        A = self.cv.a[Y0:Y1, X0:X1]; A += (1 - A) * a
        return self

    def block(self, pts, col, r, ang=None, hw=None, jit=(.008, .02, .03), dens=1.0, dry=.2, grad=None, lab=None, streak=.0, edge=True, hi=None, lo=None, light=(.6, -.8), flat=None):
        """one block: pts screen polygon.  Strokes along `ang`.  grad: (dx,dy,amount) value drift across the block."""
        i = len(self.blocks) + 1 if lab is None else lab
        self.poly(pts, i)
        q = self.P(pts); mn = q.min(0); mx = q.max(0); sz = mx - mn
        col = rgb(col) if isinstance(col, str) else np.asarray(col, f32)
        if ang is None:
            d = q[np.argmax(np.hypot(*(q - q[0]).T))] - q[0]; ang = 0. if sz[0] > sz[1] else np.pi / 2
        self.fill(pts, col)
        th = min(sz[0], sz[1])
        hw = max(1.5, th * r.uniform(.07, .14)) if hw is None else hw * self.ss
        n = int(max(5, sz[0] * sz[1] / (hw * hw * 14) * dens))
        n = min(n, 260)
        x = r.uniform(mn[0], mx[0], n); y = r.uniform(mn[1], mx[1], n)
        c = np.repeat(col[None], n, 0)
        if grad is not None:
            t = ((x - (mn[0] + mx[0]) / 2) / (sz[0] + 1) * grad[0] + (y - (mn[1] + mx[1]) / 2) / (sz[1] + 1) * grad[1]) * 2
            c = c * (1 + (t * grad[2])[:, None].astype(f32))
        c = jit_color(c, r, *jit)
        L = np.maximum(hw * r.uniform(2.5, 7, n), 3)
        self.cv.strokes(arcs(x, y, ang + r.normal(0, .12, n), L, r.normal(0, .15, n)), hw * r.uniform(.5, 1.2, n), c, r.uniform(.35, .85, n),
                        r.uniform(0, 1, n) * dry * 2, seed=int(r.integers(1 << 20)), lab=self.lab, LAB=i, bristle=.05)
        if edge: self.edges(pts, col, r, i, hi, lo, light)
        self.blocks.append((np.asarray(pts, float), col, ang))
        return i

    def edges(self, pts, col, r, lab=None, hi=None, lo=None, light=(.6, -.8), wd=None):
        """broken light arris toward the light, darker arris away from it"""
        q = self.P(pts); n = len(q); ctr = q.mean(0)
        hi = np.clip(col * 1.12 + .035, 0, 1) if hi is None else (rgb(hi) if isinstance(hi, str) else hi)
        lo = col * .74 + np.array([-.01, -.015, .01]) if lo is None else (rgb(lo) if isinstance(lo, str) else lo)
        sz = np.ptp(q, 0).min(); wd = np.clip(sz * .035, .8, 3.2) * (1 if wd is None else wd)
        for k in range(n):
            a, b = q[k], q[(k + 1) % n]; e = b - a; L = np.hypot(*e)
            if L < 2: continue
            nn = np.array([e[1], -e[0]]) / L
            if np.dot(nn, (a + b) / 2 - ctr) < 0: nn = -nn
            s = nn[0] * light[0] + nn[1] * light[1]
            if abs(s) < .15 or r.random() < .16: continue
            t0, t1 = sorted(r.uniform(0, 1, 2)) if r.random() < .35 else (r.uniform(-.05, .08), r.uniform(.9, 1.05))
            A = a + e * t0 - nn * wd * .7; B = a + e * t1 - nn * wd * .7
            P = np.stack([A + (B - A) * t for t in (0, .33, .66, 1)])[None]
            self.cv.strokes(P, wd * r.uniform(.6, 1.1), np.clip(hi if s > 0 else lo, 0, 1), min(1, abs(s)) * r.uniform(.4, .8), r.uniform(0, .4), seed=int(r.integers(1 << 20)), shape='round',
                            lab=self.lab if lab is not None else None, LAB=lab, bristle=.04)
        return self

    def dabs(self, pts_xy, size, col, r, op=(.4, .9), val=.04, shape='round', lab=None, asp=1.4, ang=None, dry=.2):
        """little freehand dabs at screen points"""
        q = self.P(pts_xy); n = len(q)
        if n == 0: return self
        s = np.asarray(size) * self.ss
        c = np.broadcast_to(rgb(col) if isinstance(col, str) else np.asarray(col, f32), (n, 3))
        c = jit_color(c, r, .012, .03, val)
        an = r.uniform(0, np.pi, n) if ang is None else ang + r.normal(0, .3, n)
        self.cv.strokes(arcs(q[:, 0], q[:, 1], an, s * asp * r.uniform(.7, 1.3, n), r.normal(0, .8, n)), s * r.uniform(.5, 1, n), c, r.uniform(*op, n), r.uniform(0, 1, n) * dry,
                        seed=int(r.integers(1 << 20)), shape=shape, lab=self.lab if lab is not None else None, LAB=lab)
        return self

    def lichen(self, c, rad, r, cols=('#b9a765', '#a7a58a', '#c9b48a', '#8f9470'), n=26, lab=None):
        """a colony: ring of small blotches with a thin heart"""
        a = r.uniform(0, 2 * np.pi, n); q = np.sqrt(r.uniform(.12, 1, n))
        p = np.c_[c[0] + np.cos(a) * rad * q * r.uniform(.8, 1.2, n), c[1] + np.sin(a) * rad * .75 * q]
        col = np.stack([rgb(cols[int(k)]) for k in r.integers(0, len(cols), n)])
        self.dabs(p, np.maximum(.6, rad * r.uniform(.06, .22, n)), col, r, (.3, .8), .05, lab=lab)
        return self

    def crack(self, a, b, r, col='#57496a', w=1.1, lab=None, wig=.12):
        a = np.asarray(a, float); b = np.asarray(b, float); L = np.hypot(*(b - a)); n = max(4, int(L / 14))
        nn = np.array([-(b - a)[1], (b - a)[0]]) / (L + 1e-6)
        pts = np.array([a + (b - a) * t + nn * r.normal(0, wig * L / n * 1.8) * (0 < t < 1) for t in np.linspace(0, 1, n)])
        q = self.P(pts)
        K = 4
        for k in range(0, n - 1):
            s = np.stack([q[k] + (q[k + 1] - q[k]) * t for t in np.linspace(-.05, 1.05, K)])[None]
            self.cv.strokes(s, max(.5, w * self.ss * .5 * (1 - .75 * k / n) * r.uniform(.6, 1.2)), rgb(col), r.uniform(.5, .9), 0, seed=int(r.integers(1 << 20)), shape='round', lab=self.lab if lab is not None else None, LAB=lab)
        return self

    def layer(self):
        L = self.cv.layer()
        return cv2.resize(L, (self.w, self.h), interpolation=cv2.INTER_AREA).astype(f32)


def wash(layer, mask, col, op):
    """tint a premultiplied layer toward col by mask*op, alpha kept"""
    k = (mask * op)[..., None] if mask.ndim == 2 else mask * op
    layer[..., :3] += (np.asarray(col, f32) * layer[..., 3:4] - layer[..., :3]) * k
    return layer
