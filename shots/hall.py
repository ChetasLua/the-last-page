"""The reading hall, built once in metres and painted book by book for any camera.  Library lane helper (s2, s5).
World: X right, Y forward, Z up.  Back wall Y=17, left wall X=-7.6, right wall X=6.2.  One SUN_DIR for beam, patch and mullion shadows."""
import os, math, numpy as np, cv2
from numba import njit, prange
from lp import W, H, books as bk, comp
from lp.paint import rgb, mix, smooth, ramp, fbm, warp_by, repaint, Canvas, arcs, jit_color, mottle, flat_dabs, f32, grid, lum
from lp.ink import spl, noise1, resample
from lp.cam import Camera3D

XL, XR, YB = -7.6, 6.2, 17.0
HS = 2.5                                                  # storey height
SUN = np.array([5.8, -9.4, -6.9]); SUN = SUN / np.linalg.norm(SUN)
WIN = dict(x0=-6.3, x1=-3.5, z0=4.9, zs=7.1, z1=9.0)     # sill, spring line, crown of the pointed arch
MOON = rgb('#9fb4e0')
PC = np.array([1.1, 7.8, .4]); PW = np.array([-4.9, 16.6, 6.9])
WOOD = rgb('#6a4b3c')
UP = np.array([0, 0, 1.])
EY_ = np.array([0., 1., 0])


# ------------------------------------------------------------------ window opening (X, Z) mask
def window_mask(res=160):
    """moonlight transmitted through the broken window; (H, W) texture over the WIN rectangle, v from crown down"""
    w = WIN; tw, th = int((w['x1'] - w['x0']) * res), int((w['z1'] - w['z0']) * res)
    r = np.random.default_rng(41)
    m = np.zeros((th, tw), np.uint8)
    px = lambda X, Z: (int((X - w['x0']) * res), int((w['z1'] - Z) * res))
    out = window_outline()
    cv2.fillPoly(m, [np.array([px(*p) for p in out], np.int32)], 255, cv2.LINE_AA)
    m = m.astype(f32) / 255.
    # panes: leaded grid with different transmission, some broken out (1.0), some boarded or filthy
    nx, nz = 6, 10
    pane = np.zeros_like(m)
    for j in range(nz):
        for i in range(nx):
            x0, x1 = int(i * tw / nx), int((i + 1) * tw / nx); y0, y1 = int(j * th / nz), int((j + 1) * th / nz)
            broken = r.random() < .20 or (1 <= i <= 3 and 4 <= j <= 6)
            v = r.uniform(.70, .84)
            pane[y0:y1, x0:x1] = v
            if broken:
                c = np.array([(x0 + x1) / 2, (y0 + y1) / 2]); n = 11; a = np.sort(r.uniform(0, 2 * np.pi, n))
                rad = r.uniform(.75, 1.25, n)
                pts = np.stack([c[0] + np.cos(a) * rad * (x1 - x0) * .55, c[1] + np.sin(a) * rad * (y1 - y0) * .55], -1)
                pts[:, 0] = np.clip(pts[:, 0], x0, x1); pts[:, 1] = np.clip(pts[:, 1], y0, y1)
                cv2.fillPoly(pane, [pts.astype(np.int32)], 1.0, cv2.LINE_AA)
    g = np.ones_like(m)
    for i in range(1, nx):
        if i in (2, 4): continue
        x = int(i * tw / nx + r.normal(0, 2)); cv2.line(g, (x, 0), (x + int(r.normal(0, 4)), th), 0.2, 4, cv2.LINE_AA)
    for j in range(1, nz):
        if j == 5: continue
        y = int(j * th / nz + r.normal(0, 2)); cv2.line(g, (0, y), (tw, y + int(r.normal(0, 4))), 0.2, 4, cv2.LINE_AA)
    # broken lead where the hole is
    g[int(th * .30):int(th * .58), int(tw * .20):int(tw * .48)] = np.maximum(g[int(th * .30):int(th * .58), int(tw * .20):int(tw * .48)], .9)
    for x in (tw / 3., tw * 2 / 3.): cv2.line(g, (int(x), 0), (int(x), th), 0., int(.11 * res), cv2.LINE_AA)
    cv2.line(g, (0, int(th * .50)), (tw, int(th * .50)), 0., int(.10 * res), cv2.LINE_AA)
    for (cx, s) in ((tw / 6., 1), (tw / 2., 1), (tw * 5 / 6., 1)):
        a = np.linspace(0, np.pi, 18); rad = tw / 6.
        pts = np.stack([cx + np.cos(a) * rad, (w['z1'] - w['zs']) * res - np.sin(a) * rad * 1.15], -1)
        cv2.polylines(g, [pts.astype(np.int32)], False, 0., int(.07 * res), cv2.LINE_AA)
    m = m * pane * g
    # branches of the ivy that grows in at the lower right corner, and a bough outside
    for k in range(5):
        p = np.array([tw * r.uniform(.75, 1.05), th * r.uniform(.75, 1.0)]); a = r.uniform(-2.8, -1.6)
        pts = [p]
        for _ in range(9): a += r.normal(0, .35); p = p + 22 * np.array([math.cos(a), math.sin(a)]); pts.append(p)
        cv2.polylines(m, [np.array(pts, np.int32)], False, 0., 5, cv2.LINE_AA)
        for q in pts[2:]:
            for _ in range(3):
                c = q + r.normal(0, 13, 2); cv2.ellipse(m, (int(c[0]), int(c[1])), (int(r.uniform(6, 12)), int(r.uniform(4, 7))), r.uniform(0, 180), 0, 360, 0., -1, cv2.LINE_AA)
    return np.ascontiguousarray(m)


def spl3(P, n=10):
    P = np.asarray(P, float); m = len(P)
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]]); out = []
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    for i in range(m - 1):
        p0, p1, p2, p3 = Q[i:i + 4]
        out.append(.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out + [P[-1:]])


def window_outline():
    w = WIN; cx = (w['x0'] + w['x1']) / 2; a = (w['x1'] - w['x0']) / 2; h = w['z1'] - w['zs']
    c = (h * h - a * a) / (2 * a); R = a + c
    th = np.linspace(0, math.atan2(h, c), 10)
    L = [(cx + c - R * math.cos(t), w['zs'] + R * math.sin(t)) for t in th]
    return [(w['x0'], w['z0'])] + L + [(2 * cx - x, z) for (x, z) in L[::-1][1:]] + [(w['x1'], w['z0'])]


@njit(cache=True, fastmath=True, parallel=True)
def _boxvis(pos, dirx, diry, dirz, point, B, out):
    """shadow test of each world point against oriented boxes.  point: light position (3) or use direction when point[0]>1e8.
    B[n,15]: centre3, axis x3, y3, z3, half3"""
    h, w = pos.shape[0], pos.shape[1]
    for y in prange(h):
        for x in range(w):
            px = pos[y, x, 0]; py = pos[y, x, 1]; pz = pos[y, x, 2]
            if px > 1e8: continue
            if point[0] > 1e8:
                dx = dirx; dy = diry; dz = dirz; tm = 1e9
            else:
                dx = point[0] - px; dy = point[1] - py; dz = point[2] - pz; tm = 1.0
            ln = math.sqrt(dx * dx + dy * dy + dz * dz)
            v = 1.0
            for i in range(B.shape[0]):
                ox = px - B[i, 0]; oy = py - B[i, 1]; oz = pz - B[i, 2]
                t0 = 0.0; t1 = tm; ok = True
                for a in range(3):
                    ax = B[i, 3 + a * 3]; ay = B[i, 4 + a * 3]; az = B[i, 5 + a * 3]
                    e = B[i, 12 + a]
                    o = ox * ax + oy * ay + oz * az
                    d = dx * ax + dy * ay + dz * az
                    if abs(d) < 1e-9:
                        if abs(o) > e: ok = False; break
                    else:
                        q0 = (-e - o) / d; q1 = (e - o) / d
                        if q0 > q1:
                            q0, q1 = q1, q0
                        if q0 > t0: t0 = q0
                        if q1 < t1: t1 = q1
                        if t0 > t1: ok = False; break
                if ok and t1 > 1e-4 and t1 - t0 > .004 and t0 < tm and t0 * ln > .012:
                    v = 0.0; break
            out[y, x] = v


@njit(cache=True, fastmath=True, parallel=True)
def _beam(h, w, q, eye, R, f, cx, cy, sun, mask, win, yb, depth, n, out):
    """single scattering along each view ray through the prism of light behind the window mask (screen / q resolution)"""
    th, tw = mask.shape
    for j in prange(h):
        for i in range(w):
            sx = (i + .5) * q - .5; sy = (j + .5) * q - .5
            a = (sx - cx) / f; b = (sy - cy) / f
            dx = a * R[0, 0] + b * R[1, 0] + R[2, 0]; dy = a * R[0, 1] + b * R[1, 1] + R[2, 1]; dz = a * R[0, 2] + b * R[1, 2] + R[2, 2]
            ln = math.sqrt(dx * dx + dy * dy + dz * dz)
            tm = depth[j, i]
            acc = 0.0
            st = (tm - .5) / n
            jt = math.sin(i * 12.9898 + j * 78.233) * 43758.5453; jt = jt - math.floor(jt)
            for k in range(n):
                t = .5 + (k + jt) * st
                px = eye[0] + dx * t; py = eye[1] + dy * t; pz = eye[2] + dz * t
                s = (yb - py) / -sun[1]
                if s < 0: continue
                X = px - sun[0] * s; Z = pz - sun[2] * s
                u = (X - win[0]) / (win[1] - win[0]); v = (win[3] - Z) / (win[3] - win[2])
                if u < 0 or u >= 1 or v < 0 or v >= 1: continue
                m = mask[int(v * th), int(u * tw)]
                acc += m * math.exp(-s * .055) * st * ln
            out[j, i] = acc


# ------------------------------------------------------------------ builder
class Hall:
    def __init__(self, cam, h, w, ss=2, view='s2', layer=False):
        self.cam, self.h, self.w, self.view = cam, h, w, view
        self.P = bk.Painter(h, w, ss, base='#1c1a30', pid=True, layer=layer)
        self.planes = [(0., 0., 1., 0.)]; self.boxes = []
        self.r = np.random.default_rng(7)

    # planes + world light
    def use(self, p0, n, gain=1.0, hazeoff=0.):
        n = np.asarray(n, float); n = n / np.linalg.norm(n)
        self.planes.append((n[0], n[1], n[2], float(n @ np.asarray(p0, float))))
        self.P.cur_pid = len(self.planes) - 1
        return self.light(p0, n, gain, hazeoff)

    def light(self, p0, n, gain=1.0, hazeoff=0.):
        cam = self.cam; n = np.asarray(n, float); d = float(n @ np.asarray(p0, float))

        def pos(x, y):
            x = np.asarray(x, float); y = np.asarray(y, float)
            rd = np.stack([(x - cam.cx) / cam.f, (y - cam.cy) / cam.f, np.ones_like(x)], -1) @ cam.R
            den = rd @ n; den = np.where(np.abs(den) < 1e-4, 1e-4, den)
            t = (d - cam.eye @ n) / den
            t = np.clip(t, .3, 60)
            return cam.eye + rd * t[..., None]
        fn = lambda x, y: amb(pos(x, y), n) * gain
        air = (lambda x, y: haze(pos(x, y), cam.eye, hazeoff))
        side = -1 if (np.asarray([1., 0, 0]) @ n) <= .5 else 1
        return bk.Light(fn, '#0f0d1c', side, air)

    def pj(self, *pts):
        return self.cam.project(np.asarray(pts, float))[0]

    def vis(self, pts, m=60):
        p, z = self.cam.project(np.asarray(pts, float))
        if (z < .25).all(): return False
        p = p[z > .25]
        return p[:, 0].max() > -m and p[:, 0].min() < self.w + m and p[:, 1].max() > -m and p[:, 1].min() < self.h + m

    def quad(self, a, b, c, d, col, L, n=4, **kw):
        """strip between edges a->b and d->c (world points)"""
        K = 9; t = np.linspace(0, 1, K)[:, None]
        a, b, c, d = [np.asarray(q, float) for q in (a, b, c, d)]
        if not self.vis(np.stack([a, b, c, d])): return
        A = a + (b - a) * t; B = d + (c - d) * t
        pa, za = self.cam.project(A); pb, zb = self.cam.project(B)
        if min(za.min(), zb.min()) < .2:
            k = (za > .25) & (zb > .25)
            if k.sum() < 2: return
            pa, pb = pa[k], pb[k]
        self.P.strip(pa, pb, col, L, n=n, seed=int(self.r.integers(1 << 30)), **kw)

    # ---- walls of books
    def wall(self, p0, eu, n, bays, storeys, seed, gain=1.0, hole=None, rows=None, wood=WOOD * .8, hazeoff=0., back='#241d30'):
        """bays = list of u positions; hole(i, k) -> True skips a bay"""
        eu, n = np.asarray(eu, float), np.asarray(n, float)
        r = np.random.default_rng(seed)
        for k in range(storeys):
            for i in range(len(bays) - 1):
                if hole is not None and hole(i, k): continue
                u0, u1 = bays[i], bays[i + 1]
                z0 = k * HS
                q = [p0 + eu * u + UP * z for u in (u0, u1) for z in (z0, z0 + HS)]
                if not self.vis(q): continue
                L = self.use(p0, n, gain * r.uniform(.78, 1.12), hazeoff)
                T = bk.plane(self.cam, p0, eu, UP, n)
                rr = [.40, .37, .36, .34, .34, .33] if r.random() < .5 else [.46, .40, .38, .36, .34, .20] if r.random() < .5 else [.36, .36, .38, .34, .36, .34]
                rr = list(np.array(rr) * (2.16 / sum(rr)))
                if r.random() < .3: rr = [rr[0] + rr[1] * .5] + [rr[1] * .5 + rr[2]] + rr[3:]
                mossy = .7 if (p0[0] + eu[0] * u0 < -4 and 1 <= k <= 2 and r.random() < .5) else (.35 if r.random() < .14 else 0.)
                bk.paint_bay(self.P, T, u0, u1, z0 + .10, rr, L, seed * 131 + k * 17 + i, post=.10, board=.032, wood=wood, back=back, sag=.010,
                             book_kw=dict(moss=mossy, messy=r.uniform(.2, .8), fill=r.uniform(.82, .99)), empty=.05 if r.random() < .8 else .3)
                # plinth and cornice
                e = eu; w3 = lambda u, z, d=0.: p0 + e * u + UP * z + n * d
                self.quad(w3(u0, z0 + .10, .03), w3(u1, z0 + .10, .03), w3(u1, z0, .03), w3(u0, z0, .03), wood * .8, L, 2, hi=.25, lo=-.3)
                self.quad(w3(u0, z0 + HS, .06), w3(u1, z0 + HS, .06), w3(u1, z0 + 2.27, .05), w3(u0, z0 + 2.27, .05), wood * .92, L, 4, hi=.22, lo=-.45)
                self.quad(w3(u0, z0 + 2.31, .08), w3(u1, z0 + 2.31, .08), w3(u1, z0 + 2.26, .07), w3(u0, z0 + 2.26, .07), wood * 1.15, L, 1, hi=.25, lo=-.5)
        self.P.flush()

    def gallery(self, p0, eu, n, u0, u1, z, D=.9, posts=(), gain=1.0, rail=True, broken=(), hazeoff=0., under=True):
        eu, n = np.asarray(eu, float), np.asarray(n, float)
        w3 = lambda u, zz, d: p0 + eu * u + UP * zz + n * d
        r = np.random.default_rng(int(abs(z * 100 + u0 * 10)) + int(abs(n[0]) * 7))
        wood = WOOD * .85
        if under:
            L = self.use(w3(0, z - .14, 0), -UP, gain * .55, hazeoff)
            us = np.linspace(u0, u1, max(2, int((u1 - u0) / 1.2)) + 1)
            for a, b in zip(us[:-1], us[1:]): self.quad(w3(a, z - .14, 0), w3(b, z - .14, 0), w3(b, z - .14, D), w3(a, z - .14, D), wood * .5, L, 3, val=.08)
            # joists
            for u in np.arange(u0 + .3, u1, .42): self.quad(w3(u, z - .20, 0), w3(u, z - .20, D), w3(u + .09, z - .20, D), w3(u + .09, z - .20, 0), wood * .42, L, 1, lo=-.4)
        else:
            L = self.use(w3(0, z, 0), UP, gain, hazeoff)
            us = np.linspace(u0, u1, max(2, int((u1 - u0) / 1.2)) + 1)
            for a, b in zip(us[:-1], us[1:]): self.quad(w3(a, z, 0), w3(b, z, 0), w3(b, z, D), w3(a, z, D), wood * .9, L, 4, val=.08)
        L = self.use(w3(0, z, D), n, gain, hazeoff)
        us = np.linspace(u0, u1, max(2, int((u1 - u0) / 1.1)) + 1)
        for a, b in zip(us[:-1], us[1:]):
            self.quad(w3(a, z + .02, D), w3(b, z + .02, D), w3(b, z - .15, D), w3(a, z - .15, D), wood, L, 4, val=.09, hi=.45, lo=-.4)
            self.quad(w3(a, z + .035, D + .04), w3(b, z + .035, D + .04), w3(b, z - .02, D + .03), w3(a, z - .02, D + .03), wood * 1.25, L, 1, hi=.4, lo=-.4)
            self.quad(w3(a, z - .15, D - .02), w3(b, z - .15, D - .02), w3(b, z - .20, D - .04), w3(a, z - .20, D - .04), wood * .6, L, 1)
            for u in np.arange(a + .05, b, .14):              # dentils, some lost
                if r.random() < .1: continue
                self.quad(w3(u, z - .105, D + .02), w3(u + .06, z - .105, D + .02), w3(u + .06, z - .15, D + .02), w3(u, z - .15, D + .02), wood * r.uniform(.95, 1.35), L, 1)
        self.P.flush()
        for u in posts: self.column(w3(u, 0, D - .08), z - .20, n, eu, gain, hazeoff, z0=z - HS)
        if rail: self.rail(p0 + n * (D - .04), eu, n, u0, u1, z, posts, gain, broken, hazeoff)

    def rail(self, p0, eu, n, u0, u1, z, posts, gain, broken, hazeoff=0., hgt=.82):
        w3 = lambda u, zz, d=0.: p0 + eu * u + UP * zz + n * d
        L = self.use(w3(0, z), n, gain, hazeoff)
        r = np.random.default_rng(int(z * 100) + 5)
        wood = WOOD * .95
        prof = [(0, .55), (.06, .55), (.09, .3), (.17, .28), (.30, .62), (.42, .60), (.62, .28), (.80, .22), (.88, .42), (.94, .3), (1, .4)]
        for u in np.arange(u0 + .05, u1, .175):
            if any(a < u < b for a, b in broken) and r.random() < .72: continue
            if r.random() < .05: continue
            base = w3(u, z + .03)
            if not self.vis(base[None], 80): continue
            ht = hgt - .06; tilt = r.normal(0, .012)
            if r.random() < .03: ht *= r.uniform(.3, .7); tilt = r.normal(0, .12)
            hw = .040 * r.uniform(.85, 1.1)
            zf = np.array([p[0] for p in prof]); rf = np.array([p[1] for p in prof]) * 2 * hw * (1 + r.normal(0, .06, len(prof)))
            k = zf <= 1
            A = np.stack([w3(u + tilt * q * ht - rr, z + .03 + q * ht) for q, rr in zip(zf, rf)])
            B = np.stack([w3(u + tilt * q * ht + rr, z + .03 + q * ht) for q, rr in zip(zf, rf)])
            pa, pb = self.cam.project(A)[0], self.cam.project(B)[0]
            wpx = np.hypot(*(pa - pb).T).max()
            c = wood * r.uniform(.75, 1.2); mid = (pa[4] + pb[4]) / 2
            lab = self.P.poly(np.vstack([pa, pb[::-1]]), L(c * .9, *mid)[0])
            if wpx > 3:
                s = .25 if L.side < 0 else .75
                self.P.q((pa * (1 - s) + pb * s)[[0, 3, 5, 8, 10]], wpx * .13, L(c * 1.35, *mid)[0], .85, .25, lab)
                s = .85 if L.side < 0 else .15
                self.P.q((pa * (1 - s) + pb * s)[[0, 3, 5, 8, 10]], wpx * .15, L(c * .5, *mid)[0], .85, .1, lab)
        self.P.flush()
        us = np.linspace(u0, u1, max(2, int((u1 - u0) / 1.3)) + 1)
        for a, b in zip(us[:-1], us[1:]):
            if any(x0 < (a + b) / 2 < x1 for x0, x1 in broken): continue
            dz = r.normal(0, .006)
            self.quad(w3(a, z + hgt + .05 + dz, .03), w3(b, z + hgt + .05, .03), w3(b, z + hgt - .02, .03), w3(a, z + hgt - .02 + dz, .03), wood * 1.1, L, 2, hi=.5, lo=-.45)
            self.quad(w3(a, z + .10, .01), w3(b, z + .10, .01), w3(b, z + .05, .01), w3(a, z + .05, .01), wood, L, 1, hi=.3, lo=-.4)
        for u in posts:
            if any(a < u < b for a, b in broken): continue
            self.quad(w3(u - .06, z, .04), w3(u - .06, z + hgt + .14, .04), w3(u + .06, z + hgt + .14, .04), w3(u + .06, z, .04), wood, L, 3, hi=.35, lo=-.4)
            q, zz = self.cam.project(w3(u, z + hgt + .22, .04)); s = self.cam.f / max(.3, zz) * .075
            t = np.linspace(0, 2 * np.pi, 11, endpoint=False) + r.uniform(0, .5)
            lab = self.P.poly(np.stack([q[0] + np.cos(t) * s * r.uniform(.92, 1.06, 11), q[1] + np.sin(t) * s * 1.1 * r.uniform(.92, 1.06, 11)], -1), L(wood * .9, *q)[0])
            if s > 3: self.P.q(np.stack([q + [-s * .5, -s * .1], q + [-s * .3, -s * .5], q + [s * .1, -s * .65]]), s * .22, L(wood * 1.6, *q)[0], .8, .2, lab)
        self.P.flush()

    def column(self, base, ht, n, eu, gain=1.0, hazeoff=0., z0=0., wd=.22):
        """carved wooden post: plinth, fluted shaft, collar, leaf capital"""
        b = np.array([base[0], base[1], z0]); ht = ht - z0
        if not self.vis(np.stack([b, b + UP * ht])): return
        L = self.use(b, n, gain, hazeoff); wood = WOOD * .95
        w3 = lambda u, z, d=0.: b + eu * u + UP * z + n * d
        r = np.random.default_rng(int(abs(base[0] * 91 + base[1] * 37 + z0 * 11)))
        seg = [(0, .24, .75, .75, .9), (.24, .30, .62, .70, 1.2), (.30, ht - .42, .46, .42, 1.0), (ht - .42, ht - .37, .58, .58, 1.2), (ht - .37, ht - .10, .46, .86, .95), (ht - .10, ht, .95, .95, 1.15)]
        for (z0_, z1_, wa, wb, c) in seg:
            A = w3(-wd * wa, z0_), w3(-wd * wb, z1_); B = w3(wd * wa, z0_), w3(wd * wb, z1_)
            if L.side > 0: A, B = B, A
            self.quad(A[0], A[1], B[1], B[0], wood * c * r.uniform(.92, 1.08), L, 4, val=.08, hi=.4, lo=-.5)
        s = self.cam.scale(b + UP * ht * .5)
        if s * wd > 12:
            for q in (-.22, -.05, .14):                       # flutes
                self.quad(w3(wd * q - .012, .36), w3(wd * q - .010, ht - .48), w3(wd * q + .012, ht - .48), w3(wd * q + .012, .36), wood * .5, L, 1)
            for q in (-.6, -.2, .2, .6):                       # capital leaves
                p0 = self.pj(w3(wd * q * .5, ht - .36))[0]; p1 = self.pj(w3(wd * q * .9, ht - .22))[0]; p2 = self.pj(w3(wd * q * 1.25, ht - .13))[0]
                self.P.q(np.stack([p0, p1, p2]), s * .025, L(wood * 1.5, *p1)[0], .8, .2, -1, 'leaf')
            for q in (-.4, .0, .4):
                p0 = self.pj(w3(wd * q * .8, ht - .36))[0]; p1 = self.pj(w3(wd * q * 1.0, ht - .18))[0]
                self.P.q(np.stack([p0, (p0 + p1) / 2, p1]), s * .012, L(wood * .4, *p1)[0], .8, .2, -1, 'taper')
        self.P.flush()

    # ---- one book anywhere
    def book(self, c, L3, D3, th, yaw, item, pitch=0., roll=0., cast=True, gain=1.0):
        """box book: centre c, length L3 (spine direction), depth D3, thickness th.  yaw about Z; pitch/roll tip it."""
        cy, sy = math.cos(yaw), math.sin(yaw)
        ex = np.array([cy, sy, 0.]); ey = np.array([-sy, cy, 0.]); ez = UP.copy()
        if pitch:
            cp, sp = math.cos(pitch), math.sin(pitch); ex, ez = ex * cp + ez * sp, ez * cp - ex * sp
        if roll:
            cp, sp = math.cos(roll), math.sin(roll); ey, ez = ey * cp + ez * sp, ez * cp - ey * sp
        c = np.asarray(c, float); hx, hy, hz = L3 / 2, D3 / 2, th / 2
        if not self.vis(c[None], 200): return
        if cast: self.boxes.append(np.r_[c, ex, ey, ez, hx, hy, hz])
        r = np.random.default_rng(item.seed)
        P = self.P; col = item.col
        faces = [(ez, ex, ey, hz, hx, hy, 'cover'), (-ez, ex, -ey, hz, hx, hy, 'cover'), (-ey, ex, ez, hy, hx, hz, 'spine'), (ey, -ex, ez, hy, hx, hz, 'fore'),
                 (ex, ey, ez, hx, hy, hz, 'head'), (-ex, -ey, ez, hx, hy, hz, 'head')]
        for (nn, u, v, hn, hu, hv, kind) in faces:
            fc = c + nn * hn
            if (self.cam.eye - fc) @ nn <= 0: continue
            L = self.use(fc, nn, gain * (1.12 if nn[2] > .6 else .80 if abs(nn[0]) > .6 else .92))
            M = lambda s, t, fc=fc, u=u, v=v, hu=hu, hv=hv: self.cam.project(fc + np.asarray(s, float)[..., None] * u * hu + np.asarray(t, float)[..., None] * v * hv)[0]
            q = M(np.array([-1, -.3, .4, 1, 1, 1, .3, -.4, -1, -1]), np.array([-1, -1, -1, -1, 0, 1, 1, 1, 1, 0]) * 1.)
            q = q + r.normal(0, .25, q.shape)
            cen = q.mean(0)
            a = np.hypot(*(M(np.array(1.), np.array(0.)) - M(np.array(-1.), np.array(0.)))); b = np.hypot(*(M(np.array(0.), np.array(1.)) - M(np.array(0.), np.array(-1.))))
            if min(a, b) < .3: continue
            lc = lambda cc, p: L(np.clip(cc, 0, 1), *p)[0]
            S = lambda s0, t0, s1, t1, K=4: M(np.linspace(s0, s1, K), np.linspace(t0, t1, K))
            if kind in ('cover', 'spine'):
                lab = P.poly(q, lc(col, cen))
                n = 2 if min(a, b) < 8 else 4
                for _ in range(n):
                    t = r.uniform(-.8, .8)
                    p = S(r.uniform(-1, -.4), t, r.uniform(.4, 1), t + r.normal(0, .08))
                    P.q(p, max(.5, b * r.uniform(.10, .2)), lc(col * (1 + r.normal(0, .1)), p.mean(0)), .75, .25, lab)
                if kind == 'spine' and b > 2.2:
                    P.q(S(-1, .72, 1, .72), max(.4, b * .10), lc(col * 1.35 + .03, cen), .8, .25, lab)
                    P.q(S(-1, -.8, 1, -.8), max(.4, b * .14), lc(col * .55, cen), .85, .1, lab)
                    for (t, thk, sty) in item.bands:
                        x = t * 2 - 1
                        P.q(S(x, -1, x, 1, 3), max(.45, a * thk * .45), lc({0: col * 1.3 + .02, 1: mix(col, bk.GILT, .6), 2: col * .5, 3: mix(col, item.col2, .7)}[sty], M(np.array(x), np.array(0.))), .85, .15, lab)
                    if item.label is not None and b > 4:
                        t0, t1 = item.label[0] * 2 - 1, item.label[1] * 2 - 1
                        P.q(S(t0, 0, t1, 0, 3), b * .30, lc(mix(col, item.col2, .75), M(np.array((t0 + t1) / 2), np.array(0.))), .9, .0, lab)
                elif kind == 'cover' and min(a, b) > 5:
                    for (s0, t0, s1, t1, k) in ((-1, .9, 1, .9, 1.25), (-1, -.9, 1, -.9, .7), (-.93, -1, -.93, 1, .75), (.94, -1, .94, 1, 1.2)):
                        p = S(s0, t0, s1, t1); P.q(p, max(.4, min(a, b) * .035), lc(col * k + (.02 if k > 1 else 0), p.mean(0)), .75, .25, lab)
                    if r.random() < .6:
                        for (s0, t0, s1, t1) in ((-.7, -.6, .7, -.6), (-.7, .6, .7, .6), (-.7, -.6, -.7, .6), (.7, -.6, .7, .6)):
                            p = S(s0, t0, s1, t1); P.q(p, max(.35, min(a, b) * .018), lc(mix(col, bk.GILT, .4), p.mean(0)), .55, .5, lab)
            else:
                pg = rgb('#b9a98a') * r.uniform(.7, .95)
                lab = P.poly(q, lc(col * .8, cen))
                P.q(S(-.96, 0, .96, 0), max(.4, b * .37), lc(pg, cen), 1., 0., lab)
                for _ in range(int(min(8, b / 2.2))):
                    t = r.uniform(-.7, .7); p = S(-.95, t, .95, t + r.normal(0, .08)); P.q(p, max(.35, b * .035), lc(pg * r.uniform(.6, .95), p.mean(0)), .7, .3, lab)
                P.q(S(-1, .9, 1, .9), max(.4, b * .07), lc(col * 1.2, cen), .9, .1, lab)
                P.q(S(-1, -.92, 1, -.92), max(.4, b * .08), lc(col * .6, cen), .9, .1, lab)
        P.flush()

    def stack(self, x, y, n, seed, s=1.0, z=0., yaw=None, lean=0., gain=1.0):
        r = np.random.default_rng(seed)
        yaw0 = r.uniform(-.5, .5) if yaw is None else yaw
        for k in range(n):
            l = r.uniform(.24, .40) * s; d = l * r.uniform(.64, .8); th = r.uniform(.03, .085) * s
            it = bk.make_item(r, 'flat', l, th, moss=0)
            yy = yaw0 + r.normal(0, .18) + (r.uniform(-.8, .8) if r.random() < .15 else 0) + (np.pi if r.random() < .4 else 0)
            self.book((x + r.normal(0, .022) * s + lean * z, y + r.normal(0, .022) * s, z + th / 2), l, d, th, yy, it, roll=r.normal(0, .015), gain=gain)
            z += th * .99
        return z

    def heap(self, spots, seed):
        """tumbled books: list of (x, y, z)"""
        r = np.random.default_rng(seed)
        order = sorted(spots, key=lambda p: -np.linalg.norm(np.array(p) - self.cam.eye))
        for (x, y, z) in order:
            l = r.uniform(.22, .38); d = l * r.uniform(.64, .8); th = r.uniform(.03, .08)
            it = bk.make_item(r, 'flat', l, th)
            yw, pt, rl = r.uniform(0, 2 * np.pi), r.normal(0, .25), r.normal(0, .28)
            if len(spots) <= 2 and z < .01: pt = rl = 0.; z = -.1 * th          # loose ones lie flat on the boards
            self.book((x, y, z + th * .6), l, d, th, yw, it, pitch=pt, roll=rl)

    def mound(self, x, y, rx, ry, ht, n, seed):
        r = np.random.default_rng(seed); pts = []
        for _ in range(n):
            a = r.uniform(0, 2 * np.pi); q = math.sqrt(r.random())
            pts.append((x + math.cos(a) * q * rx, y + math.sin(a) * q * ry, max(0., ht * (1 - q) * r.uniform(.35, 1.0))))
        self.heap(pts, seed + 1)

    def tuft(self, x, y, seed, s=1.0, flowers=0):
        """ferns and grass that came up through the boards"""
        r = np.random.default_rng(seed); L = self.use((x, y, 0), -EY_, 1.2); sc = self.cam.scale(np.array([x, y, 0.]))
        b = self.pj((x, y, 0.))[0]
        for k in range(int(13 * s)):
            o = b + np.array([r.normal(0, .05), r.normal(0, .012)]) * sc; ln = r.uniform(.07, .24) * s * sc; a = -np.pi / 2 + r.normal(0, .55)
            c = rgb('#5f6d4c') * r.choice([.5, .75, 1., 1.3])
            tip = o + ln * np.array([math.cos(a), math.sin(a)]); md = (o + tip) / 2 + np.array([0, -ln * .12])
            self.P.q(np.stack([o, md, tip + [math.cos(a) * ln * .2, ln * .12]]), max(.5, sc * .012 * r.uniform(.5, 1.2)), L(c, *md)[0], .95, .1, -1, 'taper')
        for k in range(flowers):
            p = b + np.array([r.normal(0, .07), -r.uniform(.06, .16)]) * sc * s; w = max(.7, sc * .010)
            for j in range(3): self.P.q(np.stack([p + r.normal(0, w, 2), p, p + r.normal(0, w, 2)]), w * .7, L(rgb('#c9cbe0'), *p)[0] * 1.15, .95, 0., -1, 'round')
        self.P.flush()

    # ---- shared pieces
    def floor(self, x0=-9., x1=8.4, y0=1.0, y1=17.0, gap=.010, wob=0.):
        L = self.use((0, 0, 0), UP, 1.0)
        r = np.random.default_rng(77)
        x = x0
        while x < x1:
            wd = r.uniform(.20, .34)
            y = y0 - r.uniform(0, 2.)
            while y < y1:
                ln = r.uniform(1.6, 4.2); a, b = max(y, y0), min(y + ln, y1)
                if b > a:
                    c = np.array([.345, .305, .268]) * r.uniform(.74, 1.12) + r.normal(0, .006, 3) + r.normal(0, .010) * np.array([1, .3, -1])
                    zz = r.normal(0, .004)
                    # subdivide for light + perspective
                    cuts = np.geomspace(a + 4., b + 4., max(2, int((b - a) / .9) + 2)) - 4. if a < .5 else np.geomspace(a, b, max(2, int((b - a) / .9) + 2))
                    g = gap
                    for ya, yb in zip(cuts[:-1], cuts[1:]):
                        self.quad((x + g, ya, 0), (x + g, yb, 0), (x + wd - g, yb, 0), (x + wd - g, ya, 0), c, L, 4, val=.07, hi=.0, lo=-.12, seg=(.4, 1.), dry=.3, wob=wob)
                    sc = self.cam.scale(np.array([x, (a + b) / 2, 0.]))
                    for _ in range(int(min(16, sc * .05))):          # grain, scratches, worn pale streaks
                        u0 = r.uniform(.06, .94); ya_ = r.uniform(a, b); yb_ = min(b, ya_ + r.uniform(.15, .9))
                        pt = self.pj((x + wd * u0, ya_, 0), (x + wd * (u0 + r.normal(0, .03)), (ya_ + yb_) / 2, 0), (x + wd * (u0 + r.normal(0, .05)), yb_, 0))
                        kk = r.choice([.55, .72, 1.22, 1.4], p=[.35, .3, .2, .15])
                        self.P.q(pt, max(.4, self.cam.scale(np.array([x, ya_, 0.])) * r.uniform(.002, .007)), L(np.clip(c * kk, 0, 1), *pt.mean(0))[0], .6, .5, -1, 'taper' if r.random() < .6 else 'flat')
                    p = self.pj((x + g, b, 0), (x + wd - g, b, 0)); q = self.pj((x + wd * .5, b, 0))[0]
                    s = self.cam.scale(np.array([x, b, 0.]))
                    self.P.q(np.stack([p[0], (p[0] + p[1]) / 2, p[1]]), max(.4, s * .007), L(c * .35, *q)[0], .9, 0.)
                    if s > 60:
                        for e in (.2, .8):
                            for yy in (a + .06, b - .06):
                                n_ = self.pj((x + wd * e, yy, 0))[0]; self.P.q(np.stack([n_ - [1.2, 0], n_, n_ + [1.2, 0]]), max(.6, s * .006), L(c * .3, *n_)[0], .8, 0., -1, 'round')
                y += ln
            x += wd
        self.P.flush()

    def ladder(self, a, b, wd, eu, rungs, n, gain=1.0, missing=()):
        a, b, eu = np.asarray(a, float), np.asarray(b, float), np.asarray(eu, float)
        L = self.use(a, n, gain)
        c = WOOD * 1.1
        for k in range(rungs):
            if k in missing: continue
            t = (k + .7) / (rungs + .4); p = a + (b - a) * t + UP * self.r.normal(0, .012)
            h_ = .022
            self.quad(p - eu * wd / 2 + UP * h_, p + eu * wd / 2 + UP * (h_ + self.r.normal(0, .01)), p + eu * wd / 2 - UP * h_, p - eu * wd / 2 - UP * h_, c * self.r.uniform(.85, 1.15), L, 1, hi=.5, lo=-.4)
        for s in (-1, 1):
            o = eu * (wd / 2 * s)
            self.quad(a + o - eu * .03, b + o - eu * .03, b + o + eu * .03, a + o + eu * .03, c * .95, L, 2, hi=.5 if s < 0 else .2, lo=-.5)
        self.P.flush()

    def vine(self, pts, w0, w1, col, n, gain=1.0, leaves=0, seed=1, lcol='#5c6a4c', lsize=.06, droop=1.0, light=None):
        """a root or ivy stem along world points, with individually placed leaves"""
        p3 = spl3(pts)
        pr, z = self.cam.project(p3)
        if (z < .3).any() or not self.vis(p3): return
        r = np.random.default_rng(seed)
        L = light or self.use(p3[len(p3) // 2], n, gain)
        s = self.cam.f / z
        tw = np.linspace(w0, w1, len(p3)) * s / 2
        tg = np.gradient(pr, axis=0); tg /= np.hypot(*tg.T)[:, None] + 1e-6; nn = np.c_[-tg[:, 1], tg[:, 0]]
        wob = 1 + noise1(len(p3), 6, r) * .18
        A = pr + nn * (tw * wob)[:, None]; B = pr - nn * (tw * wob)[:, None]
        col = rgb(col) if isinstance(col, str) else col
        if tw.max() > .8: self.P.strip(A, B, col, L, n=3, val=.10, hi=.35, lo=-.45, seed=seed, seg=(.15, .5))
        else: self.P.q(pr[np.linspace(0, len(pr) - 1, 9).astype(int)], max(.4, tw.mean()), L(col, *pr.mean(0))[0], .9, .1)
        lc = rgb(lcol)
        for k in range(leaves):
            i = int(r.integers(0, len(p3))); q = p3[i] + r.normal(0, lsize * 1.1, 3); sp, zz = self.cam.project(q)
            sz = lsize * self.cam.f / max(.4, zz) * r.uniform(.6, 1.35)
            an = r.normal(np.pi / 2 * droop, .8)
            tone = r.choice([.55, .8, 1., 1.3], p=[.3, .3, .25, .15])
            P_ = arcs([sp[0]], [sp[1]], [an], [sz * 1.7], [r.normal(0, .5)], 4)
            self.P.q(P_[0], max(.6, sz * .5), L(np.clip(lc * tone + r.normal(0, .02, 3), 0, 1), *sp)[0], .95, .05, -1, 'leaf')
        self.P.flush()

    # ---- final data
    def finish(self):
        self.P.flush()
        img = self.P.image(); pid = self.P.pid1()
        return img, pid, np.array(self.planes, np.float64), np.array(self.boxes, np.float64).reshape(-1, 15)


def positions(cam, pid, planes):
    """per-pixel world position and normal from the plane ids"""
    rd = cam.rays(*pid.shape)
    n = planes[:, :3].astype(f32)[pid]; d = planes[:, 3].astype(f32)[pid]
    den = (rd * n).sum(-1); den = np.where(np.abs(den) < 1e-5, 1e-5, den)
    t = (d - (n @ cam.eye.astype(f32))) / den
    t = np.where((pid > 0) & (t > .05), t, 40.).astype(f32)
    t = np.minimum(t, 60.)
    return cam.eye.astype(f32) + rd * t[..., None], n, t


def amb(p, n):
    """painted ambient of the sleeping hall, rgb multiplier at world points (n: one normal or per point)"""
    p = np.asarray(p, float); n = np.broadcast_to(np.asarray(n, float), p.shape)
    z = p[..., 2:3]
    base = np.array([.235, .228, .365])
    top = 1 - .62 * smooth(3.0, 9.5, z)
    c = base * top
    v = PC - p; d = np.linalg.norm(v, axis=-1, keepdims=True); k = np.clip((v * n).sum(-1, keepdims=True) / (d + 1e-6) * .7 + .42, .10, 1)
    c = c + np.array([.44, .54, .78]) * 1.25 * k / (1 + (d / 2.6) ** 2)
    v = PW - p; d = np.linalg.norm(v, axis=-1, keepdims=True); k = np.clip((v * n).sum(-1, keepdims=True) / (d + 1e-6) * .6 + .5, .25, 1)
    c = c + np.array([.42, .52, .76]) * 1.3 * k / (1 + (d / 1.8) ** 2.2)
    up = (n[..., 2:3] > .6); lf = (n[..., :1] > .6)
    c = c * np.where(up, np.array([1.08, 1.12, 1.18]), 1.) * np.where(lf, .74, 1.)
    return c


def haze(p, eye, off=0.):
    d = np.linalg.norm(p - eye, axis=-1)
    k = (1 - np.exp(-np.maximum(0, d - 6.0) * .040)) + off
    z = np.clip((p[..., 2] - 1.0) / 8.5, 0, 1)
    c = ramp(z.astype(f32), [(0, '#3c4170'), (.4, '#2e3160'), (.8, '#1e1c38'), (1, '#1a182e')])
    g = np.exp(-(((p[..., 0] + 2.5) / 5.5) ** 2 + ((p[..., 2] - 3.6) / 4.2) ** 2))       # moon mist hangs where the beam crosses
    c = c + g[..., None] * np.array([.045, .060, .095])
    return c, np.clip(k, 0, .72)


def moonlight(cam, pid, planes, boxes, mask):
    """0..1 direct moonlight per pixel (window shape, mullions, every stack's own shadow)"""
    pos, n, t = positions(cam, pid, planes)
    s = (YB - pos[..., 1]) / -SUN[1]
    X = pos[..., 0] - SUN[0] * s; Z = pos[..., 2] - SUN[2] * s
    th, tw = mask.shape
    u = (X - WIN['x0']) / (WIN['x1'] - WIN['x0']) * tw; v = (WIN['z1'] - Z) / (WIN['z1'] - WIN['z0']) * th
    m = cv2.remap(cv2.GaussianBlur(mask, (0, 0), 2.2), u.astype(f32), v.astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    ndl = np.clip(-(n @ SUN.astype(f32)), 0, 1)
    m = m * (s > .05) * np.clip(ndl / .53, 0, 1.25) * (pid > 0)
    pp = (pos + n * .008).astype(f32); pp[m < .02] = 1e9
    vis = np.ones(m.shape, f32)
    if len(boxes): _boxvis(pp, *(-SUN), np.array([1e9, 0, 0]), boxes, vis)
    return (m * vis).astype(f32), pos, n, t


def beam(cam, h, w, t, mask, q=2, n=220):
    mask = np.ascontiguousarray(cv2.GaussianBlur(mask, (0, 0), 4))
    d = cv2.resize(np.minimum(t, 26).astype(f32), (w // q, h // q), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((h // q, w // q), f32)
    _beam(h // q, w // q, float(q), cam.eye, cam.R, cam.f, float(cam.cx), float(cam.cy), SUN, mask,
          np.array([WIN['x0'], WIN['x1'], WIN['z0'], WIN['z1']]), YB, d, n, out)
    return cv2.resize(cv2.GaussianBlur(out, (0, 0), 2.0), (w, h), interpolation=cv2.INTER_LINEAR)


# ================================================================== the room itself
BACK_BAYS = [0, 1.15, 4.25, 5.85, 7.35, 8.95, 10.5, 12.1, 13.8]        # u from XL; bay 1 is the window bay
EX, EY = np.array([1., 0, 0]), np.array([0, 1., 0])


def window(hh, gain=1.0):
    w = WIN; cam, P = hh.cam, hh.P
    p0 = np.array([0, YB + .02, 0]); L = hh.use(p0, -EY, gain)
    out = np.array(window_outline()); w3 = lambda x, z, d=0.: np.array([x, YB + d, z])
    r = np.random.default_rng(3)
    # night outside: deep reveal, then sky
    cx = (w['x0'] + w['x1']) / 2; cz = (w['z0'] + w['z1']) / 2
    fr = np.stack([cx + (out[:, 0] - cx) * 1.16, cz + (out[:, 1] - cz) * 1.09], -1)
    q = cam.project(np.stack([fr[:, 0], np.full(len(fr), YB - .05), fr[:, 1]], -1))[0]
    P.poly(q, rgb('#3a3d63'))
    # rough masonry of the window bay
    z = 4.9
    while z < 10.6:
        rh = r.uniform(.26, .44); x = XL + 1.0 + r.uniform(-.3, 0)
        while x < XL + 4.4:
            bw = r.uniform(.35, .85); g = .012
            hh.quad(w3(x + g, z + rh - g, .0), w3(x + bw - g, z + rh - g + r.normal(0, .01), .0), w3(x + bw - g, z + g, .0), w3(x + g, z + g + r.normal(0, .01), .0),
                    rgb('#8d93bd') * r.uniform(.30, .52), L, 3, val=.09, hi=.18, lo=-.22)
            x += bw
        z += rh
    P.flush()
    q = cam.project(np.stack([fr[:, 0], np.full(len(fr), YB - .05), fr[:, 1]], -1))[0]
    P.poly(q, rgb('#2c2f54'))
    # surround stones
    n = len(out); M = 26
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(np.vstack([out, out[:1]]), axis=0).T))]
    tt = np.linspace(0, s[-1], M + 1); ring = np.vstack([out, out[:1]])
    for a, b in zip(tt[:-1], tt[1:]):
        pa = np.array([np.interp(a + .02, s, ring[:, 0]), np.interp(a + .02, s, ring[:, 1])]); pb = np.array([np.interp(b - .02, s, ring[:, 0]), np.interp(b - .02, s, ring[:, 1])])
        k = r.uniform(1.12, 1.24); pc = np.array([cx, cz])
        oa = pc + (pa - pc) * np.array([k, (k - 1) * .6 + 1]); ob = pc + (pb - pc) * np.array([k, (k - 1) * .6 + 1])
        hh.quad(w3(*pa, -.06), w3(*pb, -.06), w3(*ob, -.06), w3(*oa, -.06), rgb('#8d93bd') * r.uniform(.42, .66), L, 3, val=.08, hi=.5, lo=-.35)
    P.flush()
    q = cam.project(np.stack([out[:, 0], np.full(n, YB), out[:, 1]], -1))[0]
    P.poly(q, rgb('#8fa0d4'))
    x0, y0 = q.min(0); x1, y1 = q.max(0)
    lab = P.n
    for _ in range(90):                                       # painted night sky: paler low, a torn cloud
        yy = r.uniform(y0, y1); xx = r.uniform(x0, x1); v = (yy - y0) / (y1 - y0)
        c = ramp(np.array([v]), [(0, '#8294cf'), (.45, '#a3b4e4'), (1, '#c6d0ee')])[0]
        P.q(arcs([xx], [yy], [r.normal(-.25, .2)], [r.uniform(30, 90)], [r.normal(0, .5)])[0], r.uniform(6, 15), np.clip(c * (1 + r.normal(0, .025)), 0, 1), .85, .2, lab)
    for _ in range(16):
        yy = y0 + (y1 - y0) * r.normal(.62, .10); xx = r.uniform(x0, x1)
        P.q(arcs([xx], [yy], [-.2], [r.uniform(40, 90)], [r.normal(0, .4)])[0], r.uniform(4, 9), rgb('#dfe6f8') * r.uniform(.95, 1.0), .7, .35, lab)
    P.flush()
    # tracery, lead and dirty panes: multiply the opening mask
    m = window_mask(160)
    tw, th = m.shape[1], m.shape[0]
    Hm = cam.H_plane((w['x0'], YB, w['z1']), EX, -UP, tw, th, w['x1'] - w['x0'], w['z1'] - w['z0'])
    S = np.diag([P.ss, P.ss, 1.]); wm = cv2.warpPerspective(m, S @ Hm, (P.w * P.ss, P.h * P.ss), flags=cv2.INTER_AREA)
    inw = np.zeros(wm.shape, np.uint8); cv2.fillPoly(inw, [P._fx(q)], 255, cv2.LINE_8, 4)
    k = (inw > 0)[..., None] * (1 - wm[..., None])
    dark = rgb('#262949')
    P.img[:] = P.img * (1 - k * .97) + dark * (k * .97)
    # sill with rubble and fallen glass
    hh.quad(w3(w['x0'] - .25, w['z0'] + .02, -.35), w3(w['x1'] + .25, w['z0'] + .02, -.35), w3(w['x1'] + .25, w['z0'] - .14, -.35), w3(w['x0'] - .25, w['z0'] - .14, -.35), rgb('#8d93bd') * .62, L, 4, hi=.7, lo=-.4)
    P.flush()


def columns_front(hh, p0, eu, n, posts, zs, D=.9, gain=1.0, hazeoff=0.):
    for z in zs:
        for u in posts: hh.column(np.asarray(p0) + eu * u + n * (D - .08), z - .20, n, eu, gain, hazeoff, z0=z - HS)


def assemble(hh, view='s2'):
    P = hh.P; low = view == 's2'
    r = np.random.default_rng(12)
    # room shell so no pixel is empty
    L = hh.use((0, YB + .3, 0), -EY); hh.quad((XL, YB + .3, 0), (XR, YB + .3, 0), (XR, YB + .3, 14), (XL, YB + .3, 14), rgb('#3a3550'), L, 12); P.flush()
    L = hh.use((0, 0, 0), UP); fq = hh.pj((-9, 1.2, 0), (-9, 17.3, 0), (8.4, 17.3, 0), (8.4, 1.2, 0)) if low else hh.pj((-9, 17.3, 0), (8.4, 17.3, 0), (8.4, 2.6, 0), (-9, 2.6, 0)); P.poly(fq, rgb('#16131f'))
    hh.floor(y0=1.0 if low else -2.0)
    # ---- back wall
    p0 = np.array([XL, YB, 0.])
    hh.wall(p0, EX, -EY, BACK_BAYS, 5, 1, hole=lambda i, k: i == 1 and k in (2, 3), hazeoff=.04)
    window(hh)
    posts = [4.25, 5.85, 7.35, 8.95, 10.5, 12.1]
    for z in (HS, 2 * HS, 3 * HS, 4 * HS):
        u0 = 4.25 if z == 3 * HS else 0.
        hh.gallery(p0, EX, -EY, u0, 13.8, z, D=.9, posts=posts, under=low or z > 8, hazeoff=.02,
                   broken=[(9.1, 10.4)] if z == 2 * HS else [(5.0, 5.8)] if z == HS else [])
    # ladder up the middle of the back wall
    hh.ladder((.75, YB - 1.25, 0), (.95, YB - .95, 2.55), .42, EX, 9, -EY, missing=(5,))
    hh.ladder((-1.9, YB - .85, 2.5), (-1.75, YB - .12, 4.75), .40, EX, 8, -EY)
    # ---- right wall
    pr = np.array([XR, 17., 0.]); er = -EY
    rb = [0, 1.7, 3.4, 5.1, 6.8, 8.5, 10.2, 12., 13.8]
    hh.wall(pr, er, -EX, rb, 5, 2)
    for z in (4 * HS, 3 * HS, 2 * HS, HS) if not low else (HS, 2 * HS, 3 * HS, 4 * HS):
        hh.gallery(pr, er, -EX, 0., 13.8 if low else 6.8, z, D=.9, posts=rb[1:-1] if low else rb[1:4], under=low or z > 8, broken=[(5.4, 6.9)] if z == HS and low else [])
    # ---- left wall
    pl = np.array([XL, 3.2, 0.]); lb = [0, 1.75, 3.5, 5.2, 6.9, 8.6, 10.3, 12., 13.8]
    if not low: pl = np.array([XL, -2.0, 0.]); lb = [0, 1.75, 3.5, 5.2, 6.95, 8.7, 10.4, 12.1, 13.8, 15.5, 17.2, 19.]
    hh.wall(pl, EY, EX, lb, 5, 3, gain=.92)
    for z in (HS, 2 * HS, 3 * HS, 4 * HS):
        hh.gallery(pl, EY, EX, 0., lb[-1], z, D=.9, posts=lb[1:-1], gain=.92, under=low or z > 8, broken=[(8.8, 10.0)] if z == 2 * HS else [])
    # ---- roots through the crack by the window, ivy over the left galleries
    rt = [
        ([(-7.3, 16.9, 5.2), (-7.2, 16.4, 4.4), (-7.0, 16.0, 3.3), (-6.7, 15.8, 2.5), (-6.5, 15.4, 1.4), (-6.2, 15.0, .6), (-5.6, 14.5, .1), (-4.6, 13.9, .05)], .30, .07, '#5a4a52', 26),
        ([(-7.2, 16.8, 5.0), (-6.4, 16.2, 4.7), (-5.5, 16.0, 4.95), (-4.9, 15.95, 4.5), (-4.5, 15.9, 3.3), (-4.4, 15.8, 2.6)], .18, .04, '#564652', 46),
        ([(-7.4, 16.6, 5.2), (-6.9, 15.2, 5.05), (-6.75, 13.9, 5.1), (-6.7, 12.6, 5.0), (-6.72, 11.8, 4.4), (-6.7, 11.5, 3.6)], .16, .03, '#504253', 60),
        ([(-6.0, 15.0, .4), (-5.4, 14.2, .12), (-5.2, 13.2, .05), (-4.2, 12.4, .04), (-3.6, 11.6, .02)], .13, .03, '#564652', 8),
    ]
    for k, (pts, a, b, c, nl) in enumerate(rt): hh.vine(pts, a, b, c, EX + -EY, 1.1, leaves=nl, seed=60 + k, lsize=.075)
    for k in range(13):                                        # hanging ivy strands from the sill and the second gallery
        x = r.uniform(-6.5, -3.2); z0 = 4.85 if k < 7 else 5.0; ln = r.uniform(.5, 1.9); y = 16.05 + r.normal(0, .08)
        pts = [(x, y, z0)]; 
        for q in np.linspace(.2, 1, 5): pts.append((x + r.normal(0, .06), y - .05 * q, z0 - ln * q))
        hh.vine(pts, .03, .012, '#3f4a42', -EY, 1.25, leaves=int(16 + ln * 14), seed=200 + k, lsize=.075, lcol='#66785a')
    for k in range(9):
        y = r.uniform(10.5, 15.5); ln = r.uniform(.4, 1.6); z0 = 5.0 if r.random() < .6 else 2.5
        pts = [(-6.68, y, z0)] + [(-6.66 + r.normal(0, .04), y + r.normal(0, .08), z0 - ln * q) for q in np.linspace(.25, 1, 4)]
        hh.vine(pts, .03, .012, '#3c4340', EX, 1.0, leaves=int(12 + ln * 12), seed=300 + k, lsize=.075, lcol='#566a52')
    # ---- chandelier, dark against the far wall
    chandelier(hh, np.array([2.75, 11.2, 5.0]))
    # ---- collapsed bookcase leaning into the right wall
    collapsed(hh)
    banner(hh, np.array([3.45, YB - .97, 0.]), EX, -EY, 1.05, 5.86, 2.95, '#6b3a44', 5, .95)
    # ---- stacks and heaps, far to near
    st = [(-5.2, 15.4, 7, .9), (-4.1, 14.6, 4, .9), (-2.6, 15.2, 9, 1.0), (-.9, 14.6, 5, .95), (1.9, 15.6, 10, .95), (3.1, 15.0, 6, 1.), (4.6, 15.4, 8, .9), (5.3, 14.2, 5, 1.),
          (-5.6, 12.2, 6, 1.0), (-3.6, 12.6, 4, 1.0), (.1, 12.9, 3, 1.0), (4.9, 12.2, 9, 1.0), (3.4, 11.9, 4, 1.),
          (-1.75, 10.5, 5, 1.0), (2.35, 9.6, 9, 1.0), (2.75, 9.3, 5, 1.0), (3.15, 8.1, 11, 1.05), (3.55, 7.75, 6, 1.0), (-4.3, 10.4, 8, 1.0), (-3.95, 10.1, 4, 1.0), (1.45, 8.9, 3, 1.0),
          (-1.0, 7.15, 7, 1.05), (-.50, 7.45, 3, 1.), (-.78, 6.85, 2, 1.), (4.1, 6.3, 12, 1.1), (3.7, 6.05, 5, 1.1), (-5.9, 9.9, 12, 1.0), (-5.6, 9.5, 7, 1.0)]
    table(hh, -.95, 1.15, 13.3, 14.2); chair(hh, 1.5, 12.9)
    sp = [(x, y, 'st', n, s, i) for i, (x, y, n, s) in enumerate(st)]
    hp = [[(4.0 + .2 * i, 9.2 - .35 * (i % 4), 0.) for i in range(7)], [(2.9, 10.6, 0.), (3.3, 10.2, 0.), (3.1, 9.9, .02), (3.6, 9.9, .12), (3.5, 10.4, .08)],
          [(-3.2, 9.3, 0.), (-2.9, 9.0, 0.), (-3.5, 8.9, 0.)], [(-1.95, 5.75, 0.), (-1.35, 6.2, 0.)], [(.9, 6.3, 0.), (1.36, 6.06, 0.)], [(-.2, 10.9, 0.), (.2, 11.2, 0.)], [(2.4, 6.5, 0.)]]
    for j, pts in enumerate(hp): sp.append((np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts]), 'hp', pts, 0, j))
    mounds = [(-2.3, 13.6, .9, .5, .3, 16), (-6.1, 12.0, 1.0, 1.4, .62, 46), (3.5, 9.15, 1.25, .75, .50, 46), (4.35, 8.95, .95, .5, .62, 46), (3.0, 9.9, .7, .5, .42, 22), (2.6, 15.6, 1.3, .7, .55, 40), (-4.0, 15.6, 1.3, .8, .4, 30), (5.2, 13.0, .8, 1.2, .6, 34), (-3.55, 8.0, .55, .5, .22, 16), (4.4, 7.6, .9, .8, .42, 34)]
    for j, m in enumerate(mounds): sp.append((m[0], m[1], 'md', m, 0, j))
    for s_ in sorted(sp, key=lambda q: -np.linalg.norm(np.array([q[0], q[1], 0]) - hh.cam.eye)):
        if s_[2] == 'md': hh.mound(*s_[3], 1300 + s_[5] * 9)
        elif s_[2] == 'st': hh.stack(s_[0], s_[1], s_[3], 500 + s_[5], s_[4], lean=r.normal(0, .06))
        else: hh.heap(s_[3], 700 + s_[5])
    litter(hh)
    tr = np.random.default_rng(66)
    for k in range(30):
        x, y = (tr.normal(.9, 1.3), tr.normal(8.6, 1.3)) if k < 20 else (tr.uniform(-5.5, 5), tr.uniform(11, 15.5))
        hh.tuft(x, y, 2000 + k, tr.uniform(.6, 1.25), int(tr.integers(0, 5)) if k < 20 else 0)
    left_case(hh)
    return hh


def chandelier(hh, c):
    L = hh.use(c, -EY, .62, -.06); P = hh.P; r = np.random.default_rng(9); col = rgb('#4a4048')
    top = c + UP * 7
    for q in np.linspace(0, 1, 40):
        p = hh.pj(c + (top - c) * q)[0]; P.q(np.stack([p + [-.6, -3], p + [.6, 0], p + [-.3, 3]]), 1.3, L(col * r.uniform(.6, 1.0), *p)[0], .9, .1, -1, 'round')
    for (rad, z) in ((1.0, 0.), (.55, .45)):
        t = np.linspace(0, 2 * np.pi, 41); ring = np.stack([c[0] + np.cos(t) * rad, c[1] + np.sin(t) * rad, np.full(41, c[2] + z)], -1)
        for half in (slice(0, 21), slice(20, 41)):
            a = hh.pj(ring[half] + UP * .05); b = hh.pj(ring[half] - UP * .05)
            P.strip(a[0], b[0], col * (.75 if half.start == 0 else 1.05), L, n=2, hi=.5, lo=-.4, seed=int(rad * 100) + half.start)
        for k in range(10 if rad > .8 else 6):
            a = k * 2 * np.pi / (10 if rad > .8 else 6) + .2
            p = c + np.array([math.cos(a) * rad, math.sin(a) * rad, z])
            hh.quad(p - EX * .03 + UP * .05, p - EX * .03 + UP * r.uniform(.1, .24), p + EX * .03 + UP * r.uniform(.1, .24), p + EX * .03 + UP * .05, rgb('#b9a98a') * .75, L, 1)
            hh.vine([p, p * .5 + c * .5 + UP * .55, c + UP * 1.15], .022, .022, col, -EY, .9, seed=k)
    P.flush()


def collapsed(hh):
    """a tall case that slid sideways into the right wall; boards snapped, what is left has slumped to the low side"""
    a = .40; ca, sa = math.cos(a), math.sin(a)
    eu = np.array([ca, -.10, -sa]); eu /= np.linalg.norm(eu); ev = np.array([sa, .02, ca]); ev -= eu * (ev @ eu); ev /= np.linalg.norm(ev)
    n = np.cross(eu, ev); n = n if n[1] < 0 else -n
    wd, ht = 2.1, 3.1
    foot = np.array([4.72, 9.6, 0.0]); p0 = foot - eu * wd
    G = .74; wood = WOOD * .78; P = hh.P
    hh.boxes.append(np.r_[p0 + eu * wd / 2 + ev * ht / 2 - n * .18, eu, ev, n, wd / 2, ht / 2, .18])
    r = np.random.default_rng(31)
    Le = hh.use(p0, -eu, 1.25)                                     # end panel turns to the moon pool
    hh.quad(p0, p0 + ev * ht, p0 + ev * ht - n * .36, p0 - n * .36, wood * 1.05, Le, 5, hi=.3, lo=-.3)
    L = hh.use(p0, n, G); T = bk.plane(hh.cam, p0, eu, ev, n)
    P.poly(T(np.array([0, wd, wd, 0]), np.array([0, 0, ht, ht]), 0.), L(rgb('#211a2c'), *T(wd / 2, ht / 2, 0.))[0]); bl = P.n
    for _ in range(60):                                            # bare back boards
        u = r.uniform(.05, wd - .05); v0 = r.uniform(0, ht * .85); p = T(np.array([u, u + r.normal(0, .01), u]), np.array([v0, v0 + .25, v0 + .55]), 0.)
        P.q(p, r.uniform(3, 9), L(rgb('#2b2233') * r.uniform(.55, 1.35), *p.mean(0))[0], .7, .35, bl)
    for u in (.36, .74, 1.1, 1.5, 1.82):
        p = T(np.array([u, u + .01, u]), np.array([0, ht / 2, ht]), 0.); P.q(p, 1.1, L(rgb('#1c1628'), *p.mean(0))[0], .8, .1, bl)
    rows = [.46, .42, .40, .40, .38, .36, .36]; v = .08; frac = [.97, .86, .62, .90, .50, .62, .34]
    for k, rh in enumerate(rows):
        broken = k in (2, 4); u0 = wd * (1 - frac[k]) - .06
        it = bk.make_row(900 + k, wd - u0 - .10, rh - .04, fill=.97, messy=.0)
        lean = 0.
        for i in it[::1]: i.lean = 0.
        for j, i in enumerate(it):                                 # everything leans downhill
            if i.kind == 'up':
                i.lean = float(np.clip(.60 - .50 * (j + 1) / len(it) + r.normal(0, .06), .02, .7)); i.u = i.u + i.w * .5
        bk.paint_row(P, T, it, L, v0=v + .03, sag=.02, width=wd - u0 - .10, seed=k, clear_h=rh - .04, u0=u0)
        if broken:
            m = r.uniform(.4, .6) * wd
            a0, b0 = T(np.array([.04, m]), np.array([v + .03, v - .22]), .02), T(np.array([.04, m]), np.array([v, v - .30]), .02)
            P.strip(a0, b0, wood * 1.2, L, n=2, hi=.35, lo=-.4, seed=k)
            a0, b0 = T(np.array([m + .07, wd - .04]), np.array([v - .07, v + .03]), .02), T(np.array([m + .05, wd - .04]), np.array([v - .14, v - .015]), .02)
            P.strip(a0, b0, wood * 1.05, L, n=2, hi=.35, lo=-.4, seed=k + 50)
            for q in range(4):                                     # splinters
                e = T(np.array([m + r.uniform(-.02, .06)] * 2) + np.array([0, r.uniform(.04, .1)]), np.array([v - .2, v - .2 + r.normal(0, .05)]), .03)
                P.q(np.stack([e[0], e.mean(0), e[1]]), 1.1, L(wood * 1.5, *e[0])[0], .9, .1, -1, 'taper')
        else:
            x = np.linspace(.04, wd - .04, 9); dv = -.02 * np.sin(np.pi * x / wd)
            P.strip(T(x, v + .035 + dv, .02), T(x, v - .015 + dv, .02), wood * r.uniform(1.0, 1.25), L, n=3, hi=.45, lo=-.4, seed=k)
        v += rh
    P.flush()
    for (u0, u1, v0, v1) in ((0, .09, 0, ht), (wd - .09, wd, 0, ht - .5)):
        hh.quad(p0 + eu * u0 + ev * v0 + n * .03, p0 + eu * u0 + ev * v1 + n * .03, p0 + eu * u1 + ev * v1 + n * .03, p0 + eu * u1 + ev * v0 + n * .03, wood, L, 5, hi=.55 if u0 == 0 else .1, lo=-.4, wob=.6)
    for (u0, u1, v0, v1, d) in ((0, 1.22, ht - .15, ht, 0.), (1.30, wd + .05, ht - .23, ht - .08, .0), (0, wd, 0, .09, 0.)):
        hh.quad(p0 + eu * u0 + ev * v1 + n * .05, p0 + eu * u1 + ev * (v1 - (.05 if u0 > 1 else 0)) + n * .05, p0 + eu * u1 + ev * (v0 - (.05 if u0 > 1 else 0)) + n * .05, p0 + eu * u0 + ev * v0 + n * .05, wood * 1.05, L, 4, hi=.5, lo=-.4, wob=.7)
        if v0 > 1: hh.quad(p0 + eu * (u0 - .03) + ev * (v1 + .05) + n * .08, p0 + eu * (u1 - .02) + ev * (v1 + .05) + n * .08, p0 + eu * (u1 - .02) + ev * (v1 - .01) + n * .08, p0 + eu * (u0 - .03) + ev * (v1 - .01) + n * .08, wood * 1.2, L, 1, hi=.5, lo=-.4, wob=.6)
    P.flush()
    for k in range(6):                                             # dead creeper over its shoulder
        u = r.uniform(0., .9); o = p0 + eu * u + ev * (ht + .03) + n * .08; ln = r.uniform(.4, 1.2)
        pts = [o] + [o + np.array([r.normal(0, .05), -.03, -ln * q]) for q in np.linspace(.2, 1, 5)]
        hh.vine(pts, .03, .012, '#3c3b40', n, .9, leaves=int(10 + ln * 10), seed=640 + k, lsize=.07, lcol='#55604c')


def banner(hh, p0, eu, n, w, z1, z0, col, seed, gain=1.0):
    """a tall faded cloth over a rail: big folds, torn hem, a worn border"""
    r = np.random.default_rng(seed); P = hh.P; L = hh.use(p0, n, gain); col = rgb(col)
    w3 = lambda u, z, d=0.: p0 + eu * u + UP * z + n * d
    us = np.linspace(0, w, 25)
    hem = z0 + r.normal(0, .05, 25).cumsum() * .35 + np.abs(r.normal(0, .06, 25)) + .25 * np.clip((us / w - .72) / .28, 0, 1)
    fold = np.sin(us / w * 2 * np.pi * 2.6 + r.uniform(0, 6)) * .035
    top = hh.pj(*[w3(u, z1 - .02 * math.sin(u * 9), f) for u, f in zip(us, fold)]); bot = hh.pj(*[w3(u + f * .6, z, f) for u, z, f in zip(us, hem, fold)])
    lab = P.poly(np.vstack([top, bot[::-1]]), L(col * .8, *top.mean(0))[0])
    for k in range(64):
        i = r.uniform(0, 24); i0 = int(i); fr = i - i0; i1 = min(24, i0 + 1)
        a = top[i0] * (1 - fr) + top[i1] * fr; b = bot[i0] * (1 - fr) + bot[i1] * fr
        q0, q1 = sorted(r.uniform(0, 1, 2)); q1 = min(1, q1 + .25)
        sl = math.cos(us[i0] / w * 2 * np.pi * 2.6)
        c = col * (1 + .30 * (fold[i0] / .035) + r.normal(0, .06)) * (1 - .25 * q0)
        P.q(np.stack([a + (b - a) * q for q in np.linspace(q0, q1, 4)]), max(1., np.hypot(*(top[1] - top[0])) * r.uniform(.4, 1.0)), L(np.clip(c, 0, 1), *(a + (b - a) * (q0 + q1) / 2))[0], .8, .2, lab)
    for q in (.07, .93):
        i0 = int(q * 24); P.q(np.stack([top[i0] + (bot[i0] - top[i0]) * t for t in np.linspace(.02, .95, 5)]), max(.8, np.hypot(*(top[1] - top[0])) * .35), L(mix(col, bk.GILT, .35), *top[i0])[0], .6, .4, lab)
    P.flush()


def table(hh, x0, x1, y0, y1, z=.80, seed=1):
    L = hh.use((x0, y0, 0), -EY, .8); wood = WOOD * .8
    for (x, y) in ((x0 + .12, y1 - .1), (x1 - .12, y1 - .1), (x0 + .12, y0 + .1), (x1 - .12, y0 + .1)):
        for (za, zb, wa, wb) in ((0, .12, .05, .06), (.12, .5, .04, .065), (.5, .62, .08, .06), (.62, z - .05, .06, .06)):
            hh.quad((x - wa, y, za), (x - wb, y, zb), (x + wb, y, zb), (x + wa, y, za), wood, L, 2, hi=.4, lo=-.45)
    hh.quad((x0 + .1, y0 + .1, .16), (x1 - .1, y0 + .1, .16), (x1 - .1, y0 + .1, .10), (x0 + .1, y0 + .1, .10), wood * .8, L, 1)
    Lt = hh.use((x0, y0, z), UP, 1.1); hh.quad((x0, y1, z), (x1, y1, z), (x1, y0, z), (x0, y0, z), wood * 1.15, Lt, 5)
    L = hh.use((x0, y0, 0), -EY, .8); hh.quad((x0, y0, z), (x1, y0, z), (x1, y0, z - .09), (x0, y0, z - .09), wood, L, 2, hi=.5, lo=-.4)
    hh.P.flush()
    hh.boxes.append(np.r_[(x0 + x1) / 2, (y0 + y1) / 2, z - .05, EX, EY, UP, (x1 - x0) / 2, (y1 - y0) / 2, .05])
    r = np.random.default_rng(seed)
    for k in range(4): hh.stack(r.uniform(x0 + .25, x1 - .25), r.uniform(y0 + .2, y1 - .15), int(r.integers(1, 7)), 70 + seed * 9 + k, .95, z=z)


def chair(hh, x, y, seed=3):
    """tipped over backwards"""
    L = hh.use((x, y, 0), -EY, .75); wood = WOOD * .78
    for (a, b) in (((0, .0, .05), (.85, .1, .10)), ((.05, .42, .05), (.9, .5, .10)), ((.42, .0, .10), (.46, .05, .55)), ((.45, .42, .10), (.5, .46, .52)), ((.05, .03, .08), (-.02, .05, .46)), ((.05, .45, .08), (-.02, .48, .44))):
        p, q = np.array([x, y, 0]) + a, np.array([x, y, 0]) + b; o = np.array([.022, 0, .022])
        hh.quad(p - o, q - o, q + o, p + o, wood, L, 1, hi=.4, lo=-.4)
    for t in (.55, .7, .85): hh.quad((x + t, y + .02, .09), (x + t + .03, y + .48, .10), (x + t + .07, y + .48, .10), (x + t + .04, y + .02, .09), wood * .9, L, 1)
    hh.P.flush()


def litter(hh):
    """loose sheets and dead leaves on the boards"""
    r = np.random.default_rng(88); L = hh.use((0, 0, 0), UP, 1.05); P = hh.P
    spots = [(r.normal(1.2, 2.2), r.normal(8.4, 1.7)) for _ in range(26)] + [(r.uniform(-5, 5.5), r.uniform(5, 15)) for _ in range(26)]
    for (x, y) in spots:
        a = r.uniform(0, np.pi); s = r.uniform(.07, .13)
        c = np.array([x, y, .004]); ex = np.array([math.cos(a), math.sin(a), 0]) * s; ey = np.array([-math.sin(a), math.cos(a), 0]) * s * .72
        q = np.stack([c - ex - ey, c + ex - ey * r.uniform(.8, 1.1), c + ex * r.uniform(.8, 1.1) + ey + UP * r.uniform(0, .05), c - ex + ey])
        if not hh.vis(q): continue
        p = hh.pj(*q); lab = P.poly(p, L(rgb('#b9a98a') * r.uniform(.6, .95), *p.mean(0))[0])
        P.q(np.stack([p[0] * .7 + p[1] * .3, p.mean(0), p[3] * .4 + p[2] * .6]), max(.5, np.hypot(*(p[0] - p[2])) * .1), L(rgb('#b9a98a') * .55, *p.mean(0))[0], .5, .4, lab)
    for _ in range(150):
        x, y = (r.normal(-4.6, 1.4), r.normal(14, 1.6)) if r.random() < .45 else (r.normal(.8, 2.6), r.normal(9.2, 2.4))
        if not (XL + 1 < x < XR - 1 and 4 < y < 16.5): continue
        p = hh.pj((x, y, 0.))[0]; s = hh.cam.scale(np.array([x, y, 0.])) * r.uniform(.025, .055)
        P.q(arcs([p[0]], [p[1]], [r.uniform(-.6, .6)], [s * 1.8], [r.normal(0, .8)], 4)[0], max(.5, s * .35), L(rgb('#6a5a48') * r.uniform(.5, 1.3), *p)[0], .9, .1, -1, 'leaf')
    P.flush()


def left_case(hh):
    """big freestanding case at left, the dark mass that frames the hall"""
    y = 6.85; x0, x1 = -7.9, -4.25; ht = 3.55; n = -EY
    p0 = np.array([x0, y, 0.]); P = hh.P; wood = WOOD * .78
    hh.boxes.append(np.r_[(x0 + x1) / 2, y + .28, ht / 2, EX, EY, UP, (x1 - x0) / 2, .28, ht / 2])
    Ls = hh.use((x1, y, 0), EX, .80)
    hh.quad((x1, y, 0), (x1, y, ht), (x1, y + .56, ht), (x1, y + .56, 0), wood * .9, Ls, 5, hi=.1, lo=-.2, val=.08)
    for (za, zb) in ((.25, 1.0), (1.12, 1.95), (2.07, 3.2)):      # sunk panels on the end
        hh.quad((x1 + .01, y + .10, za), (x1 + .01, y + .10, zb), (x1 + .01, y + .46, zb), (x1 + .01, y + .46, za), wood * .62, Ls, 3, hi=-.35, lo=.5)
    L = hh.use(p0, n, .56); T = bk.plane(hh.cam, p0, EX, UP, n)
    for i, (u0, u1) in enumerate(((0., 1.75), (1.75, 3.65))):
        bk.paint_bay(P, T, u0, u1, .14, [.50, .46, .42, .40, .42, .40, .40, .41], L, 4100 + i, post=.13, board=.045, wood=wood, sag=.016, book_kw=dict(messy=.6, moss=.2 if i == 0 else .0), empty=.0)
    w3 = lambda u, z, d=0.: p0 + EX * u + UP * z + n * d
    hh.quad(w3(0, .16, .05), w3(3.7, .16, .05), w3(3.7, 0, .05), w3(0, 0, .05), wood * .8, L, 4, hi=.25, lo=-.3)
    for (za, zb, d, c) in ((ht - .02, ht - .10, .12, 1.15), (ht - .10, ht - .22, .07, .85), (ht - .22, ht - .26, .05, 1.1)):
        hh.quad(w3(-.1, za, d), w3(3.76, za, d), w3(3.72, zb, d), w3(-.1, zb, d), wood * c, L, 3, hi=.35, lo=-.4)
    # carved end post
    s = 3.58
    for (za, zb, wa, wb, c) in ((0, .5, .13, .12, .9), (.5, .56, .15, .15, 1.2), (.56, 2.9, .10, .09, 1.0), (2.9, 2.98, .14, .14, 1.2), (2.98, 3.3, .09, .15, .95)):
        hh.quad(w3(s - wa, za, .08), w3(s - wb, zb, .08), w3(s + wb, zb, .08), w3(s + wa, za, .08), wood * c, L, 5, hi=.2, lo=-.2)
    for q in (-.04, .0, .04): hh.quad(w3(s + q - .009, .66, .085), w3(s + q - .009, 2.8, .085), w3(s + q + .009, 2.8, .085), w3(s + q + .009, .66, .085), wood * .5, L, 1)
    P.flush()
    Lt = hh.use((x0, y, ht), UP, .8); hh.quad((x0, y + .6, ht), (x1, y + .6, ht), (x1, y, ht), (x0, y, ht), wood * .9, Lt, 5); P.flush()
    # things piled on top
    r = np.random.default_rng(55)
    for k, (x, n_) in enumerate(((-7.1, 5), (-6.3, 3), (-5.2, 6), (-4.6, 2))): hh.stack(x, y + .22, n_, 800 + k, 1.0, z=ht, yaw=r.uniform(-.3, .3), gain=.5)


# ================================================================== painted light over the ambient plate
FLOORC = rgb('#0f0d1c')


def albedo_of(img, pos, n, eye):
    """undo haze and ambient: what the paint would be in white light (for moon patch and lantern)"""
    hc, hk = haze(pos, eye)
    a = np.clip(amb(pos, n), .05, 3).astype(f32)
    c = (img - hc.astype(f32) * hk[..., None].astype(f32)) / np.maximum(1 - hk[..., None], .3).astype(f32)
    return np.clip((c - FLOORC) / a, .02, .95).astype(f32)


def light_plate(img, pid, planes, boxes, cam, view='s2'):
    """returns plate (moonlit), albedo, pos, n"""
    h, w = img.shape[:2]
    mask = window_mask()
    m, pos, n, t = moonlight(cam, pid, planes, boxes, mask)
    m = warp_by(m, .55, 26, 5)
    m = cv2.GaussianBlur(m, (0, 0), .8)
    alb = albedo_of(img, pos, n, cam.eye)
    g = lum(alb)[..., None]
    alb2 = g * .84 + alb * .16 + .085
    brush = 1 + .07 * fbm(h, w, (60, 14, 5), (1, .8, .5), 91, aspect=3.5)[..., None]
    lit = alb2 * np.array([.56, .70, .95], f32) * 2.0 * brush
    out = img + lit * m[..., None]
    # bounce: the pool breathes into the air around it
    q = cv2.resize(lit * m[..., None], (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    bl = cv2.GaussianBlur(q, (0, 0), 5) * .30 + cv2.GaussianBlur(q, (0, 0), 16) * .55
    out = comp.screen(out, cv2.resize(bl, (w, h), interpolation=cv2.INTER_LINEAR))
    b = beam(cam, h, w, t, mask)
    nz = 1 + .16 * fbm(h, w, (220, 70), (1, .6), 17)
    b = np.clip(b * nz, 0, 3)
    b = b ** 1.2 * (1 - .45 * np.clip(1 - np.hypot(pos[..., 0] - PW[0], pos[..., 2] - PW[2]) / 2.2, 0, 1) * (pos[..., 1] > YB - 1.2))
    return out.astype(f32), alb, pos.astype(f32), n.astype(f32), b.astype(f32), m


# ================================================================== falling dust: soft hand-painted veils
def veil(img, p0, dirv, t, seed, speed=11., reach=520., w0=1.4, spread=.10, amt=.5, cols=('#ccd3ec', '#a6b0d4', '#7a84b0'), bright=None, grow=.25, fadeout=None):
    """a thin thread at the shelf edge that widens and fades as it falls.  Painted streaks in three values, patchy and
    drifting, redrawn on threes.  p0 px, dirv unit direction of fall, t frames since it started, bright(x, y) -> 0..1"""
    if t < 0: return img
    dr = int(t) // 3; tt = dr * 3 + 2.
    head = min(reach, speed * tt + .5 * grow * tt * tt)
    rs = np.random.default_rng(seed); r = np.random.default_rng(seed * 131 + dr * 7919)
    amt = amt * rs.uniform(.45, 1.0)                                             # no two veils as heavy
    if fadeout is not None: amt = amt * (1 - float(smooth(fadeout[0], fadeout[1], tt)))
    if head < 8 or amt <= .01: return img
    p0 = np.asarray(p0, float); dv = np.asarray(dirv, float); nv = np.array([-dv[1], dv[0]])
    bow = rs.normal(0, 9); ph = rs.uniform(0, 6.28); lean = rs.normal(0, .030); spread = spread * rs.uniform(.6, 1.35)
    base = lambda d: p0 + np.outer(d, dv) + np.outer(bow * (np.sin(d / reach * 3.0 + ph) - math.sin(ph)) + lean * d, nv)
    wid = lambda d: w0 + spread * d ** 1.04
    cl = rs.uniform(.15, 1.0, 14); gust = lambda d: np.interp(d - speed * tt * .85, np.linspace(-reach * 1.2, reach * .4, 14), cl)   # thick and thin, falling with it
    fade = lambda d: (.30 + .70 * smooth(0, 70, d)) * (1 - np.clip(d / (reach * 1.05), 0, 1)) ** 1.2 * (1 - smooth(head * .50, head * 1.0, d)) * (.30 + .70 * gust(d))
    e = base(np.linspace(0, head, 12)); pad = wid(head) * 1.6 + 26
    H_, W_ = img.shape[:2]
    x0, y0 = np.maximum(np.floor(e.min(0) - pad), 0).astype(int); x1, y1 = np.ceil(e.max(0) + pad).astype(int); x1, y1 = min(W_, x1), min(H_, y1)
    if x1 - x0 < 4 or y1 - y0 < 4: return img
    cv = Canvas(y1 - y0, x1 - x0, color=cols[1], layer=True)
    def pass_(n, hw, col, op, shape, dry, spr, ln0, ln1):
        aa = r.uniform(-.05, .92, n) * head; ln = r.uniform(ln0, ln1, n) * head + 12
        d = np.clip(aa[:, None] + ln[:, None] * np.linspace(0, 1, 6)[None], 0, head)
        u = r.normal(0, spr, n); wb = r.uniform(0, 6.28, n)
        P = np.stack([base(d[i]) + np.outer((u[i] + .16 * np.sin(d[i] * .021 + wb[i])) * wid(d[i]), nv) for i in range(n)]) - (x0, y0)
        dm = d.mean(1)
        cv.strokes(P, np.maximum(.45, hw * wid(dm) * r.uniform(.45, 1.5, n)), rgb(col), op=np.clip(op * fade(dm) * np.exp(-(u / .9) ** 2) * r.uniform(.4, 1.1, n), 0, 1), dry=dry, seed=seed + dr * 17 + n, shape=shape, bristle=.10)
    k = 1 + head / 220.
    pass_(int(4 * k) + 2, .50, cols[2], .24 * amt, 'round', .10, .30, .22, .42)
    pass_(int(8 * k) + 3, .17, cols[1], .40 * amt, 'round', .45, .55, .12, .30)
    pass_(int(7 * k) + 3, .055, cols[0], .62 * amt, 'taper', .50, .50, .08, .24)
    L = cv.layer()
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(f32)
    q = np.clip(((xx - p0[0]) * dv[0] + (yy - p0[1]) * dv[1]) / max(head, 1.), 0, 1)
    w = smooth(.05, .75, q)[..., None]
    L = cv2.GaussianBlur(L, (0, 0), .7) * (1 - w) + cv2.GaussianBlur(L, (0, 0), 1.3 + min(3.2, head * .006)) * w
    if bright is not None:
        db = np.linspace(0, head, 9); bb = np.array([bright(*p) for p in base(db)], f32)
        L = L * np.interp(q * head, db, bb).astype(f32)[..., None]
    s = img[y0:y1, x0:x1]; s *= 1 - L[..., 3:4]; s += L[..., :3]
    for _ in range(int(2 + head / 150.)):                                       # a few motes that strayed
        d = r.uniform(.25, .98) * head; p = base(np.array([d]))[0] + nv * r.normal(0, .9) * wid(d); rad = r.choice([.5, .8, 1.3], p=[.5, .35, .15])
        ix, iy = int(p[0]), int(p[1])
        if 6 < ix < W_ - 6 and 6 < iy < H_ - 6:
            gy, gx = np.mgrid[iy - 4:iy + 5, ix - 4:ix + 5].astype(f32); g = np.clip(rad + .7 - np.hypot(gx - p[0], gy - p[1]), 0, 1) * .38 * amt * (1 - .5 * d / reach)
            c = img[iy - 4:iy + 5, ix - 4:ix + 5]; c += (rgb(cols[0]) - c) * g[..., None]
    return img


def dust_puff(img, x, y, t, seed, s=1.0, amt=.5, cols=('#b9c1de', '#929cc4', '#6d7599'), life=15, flat=.36):
    """dust knocked out from under a book: low soft wisps that slide out sideways, curl and settle, redrawn on threes"""
    if t < 0 or t >= life: return img
    dr = int(t) // 3; k = (dr + .6) / (life / 3.)
    r = np.random.default_rng(seed * 97 + dr * 31)
    Rr = (16 + 50 * k ** .65) * s; pad = int(Rr * 2.4 + 20); H_, W_ = img.shape[:2]
    x0, y0, x1, y1 = max(0, int(x) - pad), max(0, int(y) - pad), min(W_, int(x) + pad), min(H_, int(y) + int(12 * s) + 10)
    if x1 - x0 < 8 or y1 - y0 < 8: return img
    cv = Canvas(y1 - y0, x1 - x0, color=cols[1], layer=True)
    for (n, hw, col, op, ln) in ((7, .17, cols[2], .30, .95), (10, .10, cols[1], .36, .75), (6, .035, cols[0], .46, .55)):
        sd = np.where(r.random(n) < .5, -1., 1.); dx = sd * Rr * r.uniform(.05, 1.25, n); up = r.random(n) ** 1.4 * (1 - np.abs(dx) / (1.6 * Rr))
        P = arcs(x + dx - x0, y - 1.5 * s - Rr * flat * up * 1.5 - y0, sd * -.14 * r.uniform(.2, 1.6, n), Rr * ln * r.uniform(.5, 1.2, n), -sd * r.uniform(.2, .9, n))
        cv.strokes(P, np.maximum(.5, hw * Rr * r.uniform(.5, 1.3, n)), rgb(col), op=op * amt * (1 - k) ** .8 * r.uniform(.4, 1.1, n), dry=.35, seed=seed + dr, shape='round', bristle=.08)
    L = cv2.GaussianBlur(cv.layer(), (0, 0), .9 + 1.8 * k * s)
    c = img[y0:y1, x0:x1]; c *= 1 - L[..., 3:4]; c += L[..., :3]
    return img


# ================================================================== the lectern and the giant open book (seen in the wide)
LEC = dict(x=-4.9, y=2.55, z0=.66, sl=math.radians(17))


def onb(X, t, h=0.):
    c, s_ = math.cos(LEC['sl']), math.sin(LEC['sl'])
    X, t, h = np.broadcast_arrays(*[np.asarray(v, float) for v in (X, t, h)])
    return np.stack([X, LEC['y'] - .22 + t * c - h * s_, LEC['z0'] + t * s_ + h * c], -1)


def pageh(d, side=-1):
    top = .118 if side < 0 else .132
    return .030 + (top - .030) * (1 - np.exp(-(d / .050) ** 1.35)) - .030 * smooth(.14, 1.14, d)


def lectern(hh):
    """stone step, carved arcade, sloping desk, oxblood binding, two pale pages"""
    P = hh.P; r = np.random.default_rng(77); Lx, Ly = LEC['x'], LEC['y']
    WD, WK, ST = rgb('#7c5a3c'), rgb('#5c4334'), rgb('#8f8d96')
    n_pg = np.array([0., -math.sin(LEC['sl']), math.cos(LEC['sl'])])
    hh.boxes.append(np.r_[Lx, Ly + .46, .06, EX, EY, UP, 1.75, .78, .06])
    hh.boxes.append(np.r_[Lx, Ly + .33, .38, EX, EY, UP, 1.0, .33, .26])
    hh.boxes.append(np.r_[onb(Lx, .46, -.03), EX, np.array([0, math.cos(LEC['sl']), math.sin(LEC['sl'])]), n_pg, 1.4, .56, .035])
    for (ins, z0, z1) in ((0., 0., .062), (.07, .062, .12)):
        x0, x1, y0, y1 = Lx - 1.75 + ins, Lx + 1.75 - ins, Ly - .34 + ins, Ly + 1.24 - ins
        L = hh.use((0, 0, z1), UP, 1.0); y = y0
        while y < y1 - .02:
            yb = min(y1, y + r.uniform(.34, .5)); x = x0 - r.uniform(0, .3)
            while x < x1 - .02:
                xb = min(x1, x + r.uniform(.45, .8)); a = max(x, x0) + .006
                hh.quad((a, y + .005, z1), (xb - .006, y + .005, z1), (xb - .006, yb - .005, z1), (a, yb - .005, z1), ST * r.uniform(.62, .82), L, 4, val=.07); x = xb
            y = yb
        L = hh.use((0, y0, 0), -EY, 1.0); x = x0
        while x < x1 - .02:
            xb = min(x1, x + r.uniform(.4, .72)); hh.quad((x + .005, y0, z1), (xb - .005, y0, z1), (xb - .005, y0, z0), (x + .005, y0, z0), ST * r.uniform(.75, .98), L, 2, hi=.5, lo=-.35); x = xb
        L = hh.use((x1, 0, 0), EX, 1.15); y = y0
        while y < y1 - .02:
            yb = min(y1, y + r.uniform(.4, .7)); hh.quad((x1, y + .005, z1), (x1, yb - .005, z1), (x1, yb - .005, z0), (x1, y + .005, z0), ST * r.uniform(.75, .98), L, 2, hi=.5, lo=-.35); y = yb
    P.flush()
    # ---- body
    bx0, bx1, by0, by1, zb, zt = Lx - 1., Lx + 1., Ly, Ly + .66, .12, .64
    Ls = hh.use((bx1, 0, 0), EX, 1.15)
    for ya, yb in zip(np.linspace(by0, by1, 5)[:-1], np.linspace(by0, by1, 5)[1:]): hh.quad((bx1, ya, zt), (bx1, yb, zt), (bx1, yb, zb), (bx1, ya, zb), WD * r.uniform(.75, .95), Ls, 4, val=.08, lo=-.3)
    Lf = hh.use((0, by0, 0), -EY, 1.55)
    hh.quad((bx0, by0, zt), (bx1, by0, zt), (bx1, by0, zb), (bx0, by0, zb), WD * .78, Lf, 6, val=.08)
    cols = np.linspace(bx0 + .05, bx1 - .05, 5)
    for i in range(4):
        a, b = cols[i] + .06, cols[i + 1] - .06; m = (a + b) / 2
        ar = [(a, zb + .10), (a, zb + .28)] + [(m + (a - m) * math.cos(q) ** .85, zb + .28 + .15 * math.sin(q)) for q in np.linspace(.1, 1.5, 6)] + [(m + (b - m) * math.cos(q) ** .85, zb + .28 + .15 * math.sin(q)) for q in np.linspace(1.5, .1, 6)] + [(b, zb + .28), (b, zb + .10)]
        pq = hh.pj(*[(x, by0 - .01, z) for x, z in ar]); lab = P.poly(pq, Lf(WK * .42, *pq.mean(0))[0])
        for x in np.linspace(a + .05, b - .05, 4):
            q = hh.pj((x, by0, zb + .12), (x, by0, zb + .25), (x, by0, zb + .36)); P.q(q, r.uniform(2, 4), Lf(WK * r.uniform(.45, .65), *q[1])[0], .8, .3, lab)
        sp = hh.pj((m, by0 - .012, zb + .13), (m + .01, by0 - .012, zb + .22), (m, by0 - .012, zb + .34)); P.q(sp, 1.0, Lf(WD * 1.0, *sp[1])[0], .9, .1, lab)
        for j in range(5):
            sd = 1 if j % 2 else -1; z = zb + .16 + .035 * j; q = hh.pj((m, by0 - .012, z), (m + sd * .04, by0 - .012, z + .03), (m + sd * .085, by0 - .012, z + .045)); P.q(q, 1.9, Lf(WD * 1.0, *q[1])[0], .9, .1, lab, 'leaf')
        ring = hh.pj(*[(x, by0 - .02, z) for x, z in ar[1:-1]])
        for j in range(0, len(ring) - 3, 2): P.q(ring[j:j + 4], 1.5, Lf(WD * 1.35, *ring[j])[0], .9, .1)
    for x in cols:
        for (za, zc, wa, wb, k) in ((zb, zb + .05, .065, .06, .95), (zb + .05, zb + .09, .045, .05, 1.2), (zb + .09, zb + .40, .036, .032, 1.0), (zb + .40, zb + .44, .052, .05, 1.25), (zb + .44, zt, .045, .062, 1.0)):
            hh.quad((x - wa, by0 - .04, za), (x - wb, by0 - .04, zc), (x + wb, by0 - .04, zc), (x + wa, by0 - .04, za), WD * k, Lf, 3, hi=-.35, lo=.5)
        for z in np.arange(zb + .12, zb + .38, .032): q = hh.pj((x - .034, by0 - .045, z), (x, by0 - .05, z + .012), (x + .034, by0 - .045, z + .03)); P.q(q, .8, Lf(WK * .55, *q[1])[0], .85, .1)
    for (z0, z1, d, k) in ((zb, zb + .06, .10, .95), (zb + .06, zb + .085, .085, 1.35), (zt - .05, zt + .02, .10, 1.0), (zt + .02, zt + .04, .12, 1.4)):
        hh.quad((bx0 - d, by0 - d, z1), (bx1 + d, by0 - d, z1), (bx1 + d, by0 - d, z0), (bx0 - d, by0 - d, z0), WD * k, Lf, 2, hi=.4, lo=-.4)
    P.flush()
    # ---- desk
    e0, e1 = Lx - 1.42, Lx + 1.42
    Lr = hh.use((e1, 0, 0), EX, 1.2); hh.quad(onb(e1, -.12, .04), onb(e1, 1.04, .04), onb(e1, 1.04, -.06), onb(e1, -.12, -.06), WD * 1.0, Lr, 3, hi=.4, lo=-.4)
    Lt = hh.use(onb(Lx, .4, 0.), n_pg, 1.05)
    for a, b in zip(np.linspace(e0, e1, 4)[:-1], np.linspace(e0, e1, 4)[1:]):
        for (t0, t1) in ((-.10, .30), (.30, .66), (.66, 1.04)): hh.quad(onb(a, t1, .04), onb(b, t1, .04), onb(b, t0, .04), onb(a, t0, .04), WD * r.uniform(.88, 1.05), Lt, 5, val=.07)
    Lf = hh.use(onb(Lx, -.12, 0.), -EY, 1.3)
    for (h0, h1, k) in ((-.06, -.018, .85), (-.018, .006, 1.35), (.006, .045, 1.0)): hh.quad(onb(e0, -.12, h1), onb(e1, -.12, h1), onb(e1, -.12, h0), onb(e0, -.12, h0), WD * k, Lf, 3, hi=.45, lo=-.4)
    for x in np.arange(e0 + .05, e1, .085):
        q = hh.pj(onb(x, -.125, -.036))[0]; P.q(np.stack([q + [-2, 0], q, q + [2, 0]]), 2.0, Lf(WD * r.uniform(1.0, 1.4), *q)[0], .9, 0., -1, 'round')
    P.flush()
    # ---- binding and page block
    ox = rgb('#6b2f38'); gx = Lx
    Lt = hh.use(onb(Lx, .45, .02), n_pg, 1.15); hh.quad(onb(gx - 1.26, .94, .05), onb(gx + 1.26, .94, .05), onb(gx + 1.26, .015, .05), onb(gx - 1.26, .015, .05), ox, Lt, 5)
    Lf = hh.use(onb(Lx, .015, 0.), -EY, 1.1); hh.quad(onb(gx - 1.26, .015, .055), onb(gx + 1.26, .015, .055), onb(gx + 1.26, .015, .038), onb(gx - 1.26, .015, .038), ox * 1.2, Lf, 1, hi=.4, lo=-.3)
    one = lambda c: bk.Light(np.asarray(c, f32), '#000000', -1)
    hh.page_ids = []
    for side in (-1, 1):
        d = np.r_[0., .012, .025, .045, .07, .11, .17, .27, .42, .6, .8, 1.0, 1.15]; h = pageh(d, side) + .045
        xs = gx + side * d
        hh.use(onb(gx, .05, .05), -EY); hh.page_ids.append(P.cur_pid)
        fr = [onb(x, .045, hh_) for x, hh_ in zip(xs, h)]; lo = [onb(x, .045, .055) for x in xs]
        ec = rgb('#8f8794') if side < 0 else rgb('#c79a62')
        A, B = hh.cam.project(np.array(fr))[0], hh.cam.project(np.array(lo))[0]
        lab = P.poly(np.vstack([A, B[::-1]]), ec)
        for k in range(9):
            q = r.uniform(.1, .95); pq = A * (1 - q) + B * q; i0 = int(r.integers(0, 5)); P.q(pq[i0:i0 + 5], r.uniform(.5, 1.0), ec * r.uniform(.62, 1.18), .8, .2, lab)
        hh.use(onb(gx, .45, .12), n_pg); hh.page_ids.append(P.cur_pid)
        for j in range(len(d) - 1):
            dm = (d[j] + d[j + 1]) / 2; sh = .50 + .50 * float(smooth(0., .13, dm)) - .05 * float(smooth(.5, 1.15, dm))
            c = (rgb('#a9abc2') * (1 - .22 * smooth(.35, 1.15, dm)) + rgb('#cfc7b6') * .06) * sh if side < 0 else (mix(rgb('#c9b897'), rgb('#eccb82'), smooth(.1, 1.0, dm)) * sh)
            t = np.linspace(.045, .905, 7)
            A = hh.cam.project(np.stack([onb(xs[j], tt, h[j]) for tt in t]))[0]; B = hh.cam.project(np.stack([onb(xs[j + 1], tt, h[j + 1]) for tt in t]))[0]
            lab = P.poly(np.vstack([A, B[::-1]]), c)
            for k in range(5):
                q = r.uniform(0, 1); pq = A * (1 - q) + B * q; i0 = int(r.integers(0, 3)); P.q(pq[i0:i0 + 5], max(.8, np.hypot(*(A[3] - B[3])) * r.uniform(.12, .3)), np.clip(c * (1 + r.normal(0, .06)) + r.normal(0, .012) * np.array([1, .2, -1]), 0, 1), .6, .25, lab)
        edge = hh.cam.project(np.stack([onb(x, .905, hh_) for x, hh_ in zip(xs, h)]))[0]; fe = hh.cam.project(np.stack([onb(x, .045, hh_) for x, hh_ in zip(xs, h)]))[0]
        for E_, k in ((edge, 1.0), (fe, 1.0)):
            for j in range(3, len(E_) - 4, 4): P.q(E_[j:j + 5], .9, (rgb('#d6d8e6') if side < 0 else rgb('#fff0c4')) * k, .8, .1)
        P.q(hh.cam.project(np.stack([onb(gx + side * 1.15, tt, pageh(1.15, side) + .045) for tt in np.linspace(.045, .905, 5)]))[0], .9, rgb('#cfd2e4') if side < 0 else rgb('#fff0c4'), .75, .1)
        for k in range(3):                                     # worn ripples in the paper
            t0 = r.uniform(.2, .8); pq = hh.cam.project(np.stack([onb(gx + side * q, t0 + .02 * math.sin(q * 7 + k), pageh(q, side) + .047) for q in np.linspace(.2, 1.05, 5)]))[0]; P.q(pq, .9, (rgb('#8f90a6') if side < 0 else rgb('#c9a868')), .45, .4)
    g = hh.cam.project(np.stack([onb(gx, tt, .06) for tt in np.linspace(.045, .905, 5)]))[0]; P.q(g, 1.5, rgb('#4a3a40'), .9, .1)
    rb = hh.cam.project(np.stack([onb(gx + .06, .05, .09), onb(gx + .07, -.10, .05), np.array([gx + .075, Ly - .36, .46]), np.array([gx + .085, Ly - .36, .34])]))[0]; P.q(spl(rb, n=3)[:9:2], 1.6, rgb('#8a2f3a'), .95, .05)
    P.flush()
    # candle stubs, sheets and a pile on the lantern side
    Lf = hh.use((0, Ly - .3, 0), -EY, 1.1)
    for (x, y, z, ht) in ((Lx - 1.5, Ly - .12, .12, .13), (Lx - 1.40, Ly - .20, .12, .08), (Lx - 1.31, Ly - .16, .12, .10), (Lx + 1.45, Ly - .16, .12, .14), (Lx + 1.36, Ly - .22, .12, .08)):
        hh.quad((x - .03, y, z), (x - .028, y, z + ht), (x + .028, y, z + ht), (x + .03, y, z), rgb('#c7b69a') * .8, Lf, 2, hi=.1, lo=.3); q = hh.pj((x, y, z + ht), (x, y, z + ht + .03)); P.q(np.stack([q[0], q.mean(0), q[1]]), .8, rgb('#2b2233'), .9)
    P.flush()
    hh.stack(Lx + 1.55, Ly + .78, 6, 8801, 1.0); hh.stack(Lx + 1.95, Ly + .95, 3, 8802, 1.0)
    Lp = hh.use((0, 0, 0), UP, 1.1)
    for (x, y, a) in ((Lx - .5, Ly - .75, .3), (Lx + .6, Ly - .62, -.4), (Lx + 2.4, Ly - .5, .8), (Lx + 1.6, Ly - .95, .2), (Lx - 1.5, Ly - .8, 1.0)):
        c = np.array([x, y, .004]); ex = np.array([math.cos(a), math.sin(a), 0]) * .13; ey = np.array([-math.sin(a), math.cos(a), 0]) * .095
        q = hh.pj(c - ex - ey, c + ex - ey * .9, c + ex + ey + UP * .02, c - ex + ey); P.poly(q, Lp(rgb('#b9a98a') * .8, *q.mean(0))[0])
    P.flush()
