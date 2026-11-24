"""s2 The Reading Hall (168 f).  Library lane.  Also paints cache/hall_bokeh.npy for the fox close-ups."""
import os, math, numpy as np, cv2
from lp import W, H, books as bk, comp, foxact
from lp.paint import rgb, mix, smooth, ramp, fbm, warp_by, repaint, Canvas, arcs, jit_color, mottle, flat_dabs, tooth, f32, grid, lum, scatter_mask
from lp.ink import Cel, spl, over_roi, noise1
from lp.cam import Camera3D
from . import hall

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CACHE = os.path.join(ROOT, 'cache')


def _save(name, a):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name); t = p + '.%d.tmp.npy' % os.getpid()
    np.save(t, a); os.replace(t, p)


# ============================================================ soft plate for the fox close-ups
def hall_bokeh(force=False):
    """shelves at the fox's eye level, thrown out of focus and gone over with a wet brush: big quiet shapes"""
    p = os.path.join(CACHE, 'hall_bokeh.npy')
    if os.path.exists(p) and not force: return np.load(p).astype(f32)
    cam = Camera3D((0, -2.45, .50), (.34, 0, .66), 1500, W, H, cx=W * .52, cy=H * .56)
    an = .30; eu = np.array([math.cos(an), math.sin(an), 0]); n = np.array([math.sin(an), -math.cos(an), 0.]); p0 = np.array([-2.35, -.72, 0.])
    P = bk.Painter(H, W, ss=1, base='#24223f')
    L = bk.Light(np.array([.46, .435, .575], f32), '#14122a', -1)
    T = bk.plane(cam, p0, eu, (0, 0, 1), n)
    us = [0., 1.45, 2.75, 4.2, 5.5, 7.0, 8.5]
    for i in range(len(us) - 1):
        bk.paint_bay(P, T, us[i], us[i + 1], .16, [.50, .45, .42, .43, .42, .42, .42], L, 1200 + i, post=.16, board=.055, wood='#5a4440', sag=.014, depth=.3, book_kw=dict(messy=.45, unit=1.35), empty=.0)
    q = lambda v0, v1, d: T(np.array([us[0], us[-1], us[-1], us[0]]), np.array([v1, v1, v0, v0]), d)
    P.strip(T(np.linspace(0, 8.5, 9), np.full(9, .16), .05), T(np.linspace(0, 8.5, 9), np.zeros(9), .05), rgb('#463239'), L, n=5, hi=.3, lo=-.3); P.flush()
    im = P.image()
    x, y = grid()
    base = T(np.linspace(0, 8.5, 60), np.zeros(60), .05); fy = np.interp(x[0], base[:, 0], base[:, 1])[None]
    fl = ramp(np.clip((y - fy) / 260., 0, 1), [(0, '#1f1c33'), (.3, '#2a2540'), (1, '#382c40')]) * _planks(x, y)
    m = smooth(-1, 1, y - fy)[..., None]; im = im * (1 - m) + fl * m
    # a few piles on the floor, left lost in shadow and right against the lantern
    P2 = bk.Painter(H, W, ss=1, base='#000000', layer=True)
    L2 = bk.Light(np.array([.42, .40, .53], f32), '#14122a', 1)
    r = np.random.default_rng(5)
    for (cx, by, s, k_) in ((300, 1165, .85, 5), (1870, 1110, 1.0, 6), (1600, 1080, .7, 3), (90, 1120, .7, 7)):
        v = 0.
        for k in range(k_):
            wd = r.uniform(250, 360) * s; th = r.uniform(34, 66) * s
            it = bk.make_item(r, 'flat', wd, th); dx = r.normal(0, 22 * s)
            T2 = lambda u, vv, d=0., cx=cx, by=by: np.stack(np.broadcast_arrays(cx + np.asarray(u, float), by - np.asarray(vv, float)), -1)
            it.u = -wd / 2 + dx; it.v = v; it.fore = r.random() < .4; v += th * .97
            bk.paint_row(P2, T2, [it], L2, tops=False)
    im = comp.over(im, P2.layer())
    # light: dark in the middle and right, a slab of moon from the upper left, the lantern low at the right
    F = np.ones((H, W, 3), f32) * .84
    F *= (1 - .40 * smooth(700, 1700, x) * smooth(900, 300, y))[..., None]
    mo = np.maximum(np.clip(1 - np.abs(x - 120 - y * .50) / (210 + y * .10), 0, 1), .8 * np.clip(1 - np.abs(x - 560 - y * .50) / (90 + y * .06), 0, 1)) ** .6 * np.clip(1.12 - y / 1200., 0, 1)
    F += mo[..., None] * np.array([.62, .70, .82], f32)
    wm = np.clip(1 - np.hypot((x - 1930) / 1080., (y - 1160) / 640.), 0, 1) ** 1.5
    F += wm[..., None] * np.array([2.5, 1.35, .42], f32)
    im = im * F; im = np.minimum(im, .34 + .34 * im + .5 * wm[..., None])
    im = comp.bokeh(np.dstack([im, np.ones((H, W, 1), f32)]), 10.5, 0)[..., :3]
    im = cv2.GaussianBlur(im, (0, 0), 2.4)
    an_ = np.where(y < fy, np.pi / 2, 0.08) + fbm(H, W, (200,), (1,), 8) * .08
    im = repaint(im, (17,), 3, angle=an_, thr=(0,), jit=(.003, .012, .012), op=.32, dry=.25, aspect=3.4, curve=.12, bristle=.05, density=.8)
    im = cv2.GaussianBlur(im, (0, 0), 1.2)
    im = comp.screen(im, (mo ** 1.5)[..., None] * rgb('#9fb4e0') * .16)
    im = comp.glow(im, 1990, 1150, 1080, '#ffb458', .32, 2.0); im = comp.glow(im, 2060, 1200, 520, '#ffd08a', .26, 1.8)
    im = 1 - (1 - np.clip(im, 0, 1)) * (1 - rgb('#1c1a30') * .18)
    im = mottle(np.clip(im, 0, 1), .010, 4, .004)
    _save('hall_bokeh.npy', im.astype(np.float16))
    return im


def _planks(x, y):
    """soft converging boards for the blurred floor"""
    u = (x - W * .60) / np.maximum(y - H * .56, 40.)
    return (1 - .16 * (np.abs(((u * 2.6) % 1.) - .5) > .44) + .06 * np.sin(np.floor(u * 2.6) * 12.9))[..., None]


# ============================================================ the shot
PW, PH = 2048, 1480
KEY = 'v8'


def camera(): return Camera3D((0, 0, 1.35), (0, 10, 1.35), 1400, PW, PH, cx=1188, cy=680)


def build():
    """paint the tall plate, the beam and the lantern-light buffers"""
    cam = camera()
    hh = hall.Hall(cam, PH, PW, 2); hall.assemble(hh, 's2'); img, pid, planes, boxes = hh.finish()
    out, alb, pos, n, b, m = hall.light_plate(img, pid, planes, boxes, cam)
    out = mottle(np.clip(out, 0, 1.2) / 1.2, .012, 3, .004) * 1.2
    _save(f's2_{KEY}_plate.npy', out.astype(np.float16)); _save(f's2_{KEY}_beam.npy', b.astype(np.float16))
    x0, x1, y0, y1 = ROI
    _save(f's2_{KEY}_alb.npy', alb[y0:y1, x0:x1].astype(np.float16))
    _save(f's2_{KEY}_pos.npy', pos[y0:y1, x0:x1].astype(np.float32)); _save(f's2_{KEY}_nrm.npy', n[y0:y1, x0:x1].astype(np.float16))
    bx = boxes[(np.hypot(boxes[:, 0] + 2.4, boxes[:, 1] - 5.9) < 5.5)]
    _save(f's2_{KEY}_boxes.npy', bx)
    # foreground stacks on their own plane, out of focus
    fh = hall.Hall(cam, PH, PW, 1, layer=True)
    for (x, y, k, sd, sc) in ((1.66, 2.75, 11, 1, 1.05), (1.22, 2.45, 4, 2, 1.0), (-1.72, 2.6, 5, 3, 1.0), (-2.24, 2.75, 9, 4, 1.05), (-2.55, 3.4, 6, 5, 1.0), (2.2, 3.5, 9, 6, 1.0)):
        fh.stack(x, y, k, 9000 + sd, sc)
    lay = fh.P.layer()
    lay[..., :3] = lay[..., :3] * np.array([.62, .62, .70], f32) + lay[..., 3:4] * rgb('#12101f') * .45
    lay[..., :3] = np.minimum(lay[..., :3], lay[..., 3:4] * rgb('#3a3850'))
    _save(f's2_{KEY}_fg.npy', comp.bokeh(lay, 4.2, .6).astype(np.float16))


ROI = (0, 1300, 330, 1240)                         # lantern light region of the plate (x0, x1, y0, y1)


class Shot:
    id = 's2'; n = 168
    finish = dict(hal_amt=.24, vig=.14, grain=.010)
    M = dict(x0=-90, x1=560, y=985, h=118)         # fox marks at the FINAL camera position; use Shot.marks(f) for any frame
    TOP0, TOP1 = 190., 14.

    @classmethod
    def top(cls, f):
        t = np.clip((f + 4.) / (cls.n + 6.), 0, 1); k = .30 * t + .70 * comp.ease(t)
        return cls.TOP0 + (cls.TOP1 - cls.TOP0) * float(k)

    @classmethod
    def marks(cls, f):
        m = dict(cls.M); m['y'] = cls.M['y'] + cls.TOP1 - cls.top(f); return m

    def setup(self):
        need = [f's2_{KEY}_{k}.npy' for k in ('plate', 'beam', 'alb', 'pos', 'nrm', 'boxes', 'fg')]
        if not all(os.path.exists(os.path.join(CACHE, q)) for q in need): build()
        hall_bokeh()
        L = lambda k: np.load(os.path.join(CACHE, f's2_{KEY}_{k}.npy'))
        self.plate = L('plate').astype(f32); self.b = L('beam').astype(f32); self.alb = L('alb').astype(f32)
        self.pos = L('pos'); self.nrm = L('nrm').astype(f32); self.boxes = L('boxes'); self.fg = L('fg').astype(f32)
        self.cam = camera()
        x0, x1, y0, y1 = ROI
        h, w = y1 - y0, x1 - x0
        self.tx = fbm(h, w, (68, 24, 9), (1, .8, .5), 71, aspect=3.2); self.tx2 = fbm(h, w, (120, 44), (1, .6), 72, aspect=2.2)
        r = np.random.default_rng(404)
        # dust: hand-scattered in the beam, each with its own size, depth and way of drifting
        bb = np.clip(self.b, 0, 1.6); bb[:40] = 0
        q = cv2.resize(bb, (PW // 4, PH // 4), interpolation=cv2.INTER_AREA) ** 1.1
        mx, my = scatter_mask(q, 74, r)
        self.motes = [(float(x * 4), float(y * 4), float(r.choice([.55, .8, 1.1, 1.5, 3.4], p=[.36, .32, .18, .09, .05])), r.normal(.10, .22), abs(r.normal(.16, .14)) + .02, r.uniform(0, 6.28), r.uniform(.35, 1.)) for x, y in zip(mx, my)]

    # ------------------------------------------------ light of the paper lantern
    def flame_world(self, x, y):
        c = self.cam; rd = np.array([(x - c.cx) / c.f, (y - c.cy) / c.f, 1.]) @ c.R
        fy = c.f * c.eye[2] / (self.M['y'] + self.TOP1 - c.cy)                 # depth of the fox's feet
        return c.eye + rd * (fy / rd[1]), fy

    def lantern(self, img, f, px, py, fx, fy_):
        """px, py: flame in plate px.  fx: fox body centre x, fy_: foot y (plate px)"""
        W3, fy = self.flame_world(px, py)
        W3 = W3 + np.array([0, -.04, .02])
        d = f // 3; r = np.random.default_rng(31 + d); fl = 1 + r.normal(0, .035)
        jit = r.normal(0, .006, 3)
        x0, x1, y0, y1 = ROI
        # only the neighbourhood of the flame is worked on
        cx_, cy_ = int(px) - x0, int(py) - y0; R = 640
        sx0, sx1, sy0, sy1 = max(0, cx_ - R), min(x1 - x0, cx_ + R), max(0, cy_ - 640), min(y1 - y0, cy_ + 300)
        if sx1 - sx0 < 8: return img
        S = (slice(sy0, sy1), slice(sx0, sx1)); pos, nrm = self.pos[S], self.nrm[S]
        v = W3 + jit + np.array([0, 0, .13], f32) * ((pos[..., 2:3] < .02) | (nrm[..., 2:3] > .6)) - pos
        dist = np.sqrt((v * v).sum(-1)) + 1e-6
        ndl = np.clip((v * nrm).sum(-1) / dist, 0, 1)
        E = (.22 + .78 * ndl ** .7) * (ndl > .0) / (1 + (dist / .84) ** 2.15)
        E = E * (1 - smooth(1.9, 2.75, dist)); near = E > .008
        pp = (pos + nrm * .014).astype(f32); pp[~near] = 1e9
        vis = np.ones(E.shape, f32)
        bx = self.boxes[np.hypot(self.boxes[:, 0] - W3[0], self.boxes[:, 1] - W3[1]) < 3.2]
        if len(bx): hall._boxvis(pp, 0., 0., 0., (W3 + jit).astype(np.float64), bx, vis)
        vis = cv2.GaussianBlur(cv2.medianBlur((vis * 255).astype(np.uint8), 5).astype(f32) / 255., (0, 0), 1.6)
        E = E * (.10 + .90 * vis)
        # painted steps in the falloff, broken by the brush
        e = np.clip(E * 1.9, 0, 1.25) + (self.tx[S] * .034 + self.tx2[S] * .026) * np.sqrt(np.clip(E, 0, 1))
        step = np.array([0, .05, .13, .27, .48, .78, 1.15]); lev = np.array([0, .035, .10, .23, .44, .76, 1.08])
        q = np.interp(e, step, lev); hard = np.zeros_like(e)
        for a, l0, l1 in zip(step[1:], lev[:-1], lev[1:]): hard += (l1 - l0) * smooth(a * .86, a * 1.02, e)
        q = .45 * q + .55 * hard
        # the fox's own shadow, long and soft to the left
        q = q * self.fox_shadow(E.shape, px, py, fx, fy_, x0 + sx0, y0 + sy0)
        q = cv2.GaussianBlur(q.astype(f32), (0, 0), 1.0)[..., None] * fl
        al = self.alb[S]
        al = al * .68 + lum(al)[..., None] * .20; al = al / (1 + np.maximum(al - .30, 0) * 1.5)
        warm = (al + .035) * rgb('#ffb458') * np.array([1.5, 1.28, 1.0], f32)
        t = img[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1]
        t += warm * q * 1.85 * (1 - .5 * np.clip(t, 0, 1))
        return img

    def fox_shadow(self, shape, px, py, fx, fy_, x0, y0):
        """light multiplier over a plate window at (x0, y0)"""
        h = self.M['h']
        m = np.zeros(shape, f32); soft = np.zeros(shape, f32)
        hl = .62 * h                                                          # virtual flame height for a storybook-long shadow
        def cast(X, Z):                                                       # silhouette point (px right of foot centre, px above the floor) -> floor px
            k = hl / max(hl - Z, .16 * hl)
            return np.array([px + (fx + X - px) * k, fy_ - (k - 1) * h * .035])
        P = lambda pts: (np.array([cast(*q) for q in pts]) - (x0, y0))
        body = P([(-.46 * h, .30 * h), (-.40 * h, .46 * h), (.0, .50 * h), (.38 * h, .46 * h), (.40 * h, .28 * h), (0., .26 * h)])
        body[:3, 1] -= 11; body[3:, 1] += 8
        tail = P([(-.42 * h, .34 * h), (-.8 * h, .44 * h), (-1.05 * h, .34 * h), (-.8 * h, .26 * h), (-.42 * h, .28 * h)])
        tail[:2, 1] -= 8
        for q in (body, tail): cv2.fillPoly(m, [np.round(np.clip(q, -4000, 4000) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, 4)
        for lx in (-.36, -.26, .22, .32):
            a = P([(lx * h - .035 * h, 0.), (lx * h - .03 * h, .30 * h), (lx * h + .03 * h, .30 * h), (lx * h + .035 * h, 0.)])
            a[1, 1] -= 5; a[2, 1] += 3
            cv2.fillPoly(m, [np.round(np.clip(a, -4000, 4000) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, 4)
        near = cv2.GaussianBlur(m, (0, 0), 2.6); far = cv2.GaussianBlur(m, (0, 0), 12)
        xx = np.arange(shape[1], dtype=f32)[None] + x0
        t = np.clip((px - xx) / (3.2 * h), 0, 1)
        c = near * (1 - t) + far * t
        # contact
        cv2.ellipse(soft, (int(fx - x0), int(fy_ - y0)), (int(.52 * h), 8), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        soft = cv2.GaussianBlur(soft, (0, 0), 5)
        return 1 - np.clip(.86 * c * (1 - .35 * t) + .55 * soft, 0, .95)

    # ------------------------------------------------ frame
    def frame(self, f):
        top = self.top(f); M = self.marks(f)
        lp_ = foxact.lantern_pos('s2', f, M)
        px, py = float(lp_[0]), float(lp_[1]) + top
        fx = px - .62 * M['h'] if f < 92 else px - (.62 + .10 * min(1, (f - 92) / 12.)) * M['h']
        foot = M['y'] + top
        img = self.plate.copy()
        img = self.lantern(img, f, px, py, fx, foot)
        # ambient contact under the fox
        h = M['h']; sh = np.zeros((60, 400), f32); cv2.ellipse(sh, (200, 30), (int(.5 * h), 9), 0, 0, 360, 1, -1, cv2.LINE_AA); sh = cv2.GaussianBlur(sh, (0, 0), 6)
        X0, Y0 = int(fx) - 200, int(foot) - 30; a0, a1 = max(0, X0), min(PW, X0 + 400)
        if a1 > a0: img[Y0:Y0 + 60, a0:a1] *= 1 - .42 * sh[:, a0 - X0:a1 - X0, None]
        img = comp.screen(np.clip(img, 0, 1), (self.b[..., None] * .40) * hall.MOON)
        self.dust(img, f)
        fr = cv2.warpAffine(img, np.float32([[1, 0, 0], [0, 1, -top]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fr = foxact.draw(fr, 's2', f, M)
        # glow of the paper lantern in the dusty air
        d = f // 3; r = np.random.default_rng(31 + d); fl = 1 + r.normal(0, .035)
        gx, gy = float(lp_[0]), float(lp_[1])
        fr = self.halo(fr, gx, gy, 300, '#ffab52', .26 * fl, 2.4); fr = self.halo(fr, gx, gy, 92, '#ffd48c', .42 * fl, 2.0)
        ft = self.TOP1 + (top - self.TOP1) * 1.38
        fg = cv2.warpAffine(self.fg, np.float32([[1, 0, 0], [0, 1, -ft]]), (W, H), flags=cv2.INTER_LINEAR)
        fr = fr * (1 - fg[..., 3:4]) + fg[..., :3]
        return fr.astype(f32)

    @staticmethod
    def halo(img, cx, cy, r, col, amt, power):
        x0, x1, y0, y1 = int(max(0, cx - r)), int(min(W, cx + r)), int(max(0, cy - r)), int(min(H, cy + r))
        if x1 <= x0 or y1 <= y0: return img
        y, x = np.mgrid[y0:y1, x0:x1].astype(f32)
        g = np.clip(1 - np.hypot(x - cx, y - cy) / r, 0, 1) ** power
        s = img[y0:y1, x0:x1]; img[y0:y1, x0:x1] = 1 - (1 - np.clip(s, 0, 1)) * (1 - g[..., None] * rgb(col) * amt)
        return img

    def dust(self, img, f):
        d = f // 3
        for k, (x, y, s, vx, vy, ph, a) in enumerate(self.motes):
            X = x + vx * 3 * d + math.sin(ph + d * .21) * 2.0 + 14 * vx * math.sin(ph * 2 + d * .07); Y = y + vy * 3 * d
            ix, iy = int(X), int(Y)
            if not (4 < ix < PW - 4 and 4 < iy < PH - 4): continue
            br = float(np.clip(self.b[iy, ix] * 1.1, 0, 1.2)) * a * (.72 + .28 * math.sin(ph * 3 + d * .9))
            if br < .05: continue
            R = int(s * 2 + 3); yy, xx = np.mgrid[iy - R:iy + R + 1, ix - R:ix + R + 1].astype(f32)
            e = 1 + .35 * math.sin(ph)                                        # motes are not round
            q = np.hypot((xx - X) * e, (yy - Y) / e)
            g = np.clip(s + .6 - q, 0, 1) if s < 3 else np.clip((s + 1.2 - q) / 2.2, 0, 1) * .30
            img[iy - R:iy + R + 1, ix - R:ix + R + 1] += g[..., None] * rgb('#dfe6fb') * min(.62, .50 * br)
        # one veil of dust, once, off the lip of the second gallery
        if 108 <= f < 162: hall.veil(img, (1236., 352.), (0., 1.), f - 108, 903, speed=8.5, reach=430., w0=1.7, spread=.095, amt=2.1, cols=('#dde3f6', '#b8c2e2', '#8f9ac8'), grow=.12, fadeout=(30, 56))
        return img


if __name__ == '__main__':
    import time
    import sys
    os.makedirs(os.path.join(ROOT, 'qa/s2'), exist_ok=True); t = time.time()
    if 'bokeh' in sys.argv: comp.save(os.path.join(ROOT, 'qa/s2/bokeh.jpg'), hall_bokeh(True)); print('bokeh', time.time() - t)
    else: build(); print('plate', time.time() - t)
