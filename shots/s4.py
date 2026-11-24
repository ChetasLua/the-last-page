"""s4  The Blank Book (168).  4a f0-56 the book on its lectern, 4b f56-112 the paw and the line, 4c f112-168 the bird leaves the page.
Owner: ink lane."""
import os, math, numpy as np, cv2
from lp import W, H, comp, foxact
from lp.paint import (rgb, mix, smooth, fbm, Canvas, arcs, repaint, jit_color, scatter_mask, mottle, tooth, f32, lum, ramp)
from lp.ink import Cel, spl, ribbon, xf, boil, over_roi, resample, noise1
from lp.cam import Camera3D
from lp.brush import Sheet, Stroke, blob, SparseInk, transmit, sheen, fibre_tile, tile_to
from lp import ink_letters as IL, sheet3d as S3

MOON, LAMP, AMB, DARK = rgb('#9fb4e0'), rgb('#ffb458'), rgb('#2c2a48'), rgb('#1c1a30')


def cached(name, ver, fn):
    os.makedirs('cache', exist_ok=True)
    p = f'cache/s4_{name}_{ver}.npz'
    if not os.path.exists(p):
        d = fn(); t = p[:-4] + '.tmp.npz'
        np.savez(t, **{k: (v.astype(np.float16) if v.dtype in (np.float32, np.float64) else v) for k, v in d.items()}); os.replace(t, p)
    z = np.load(p)
    return {k: (z[k].astype(f32) if z[k].dtype == np.float16 else z[k]) for k in z.files}


def line_layer(items, ss=1):
    """[(pts, w, col, alpha, taper)] hand lines as a layer"""
    c = Cel(ss=ss)
    for j, (pts, w, col, al, tp) in enumerate(items): c.line(pts, w, col, tp, seed=j * 13 + 5, alpha=al)
    return c.rgba()


# =====================================================================  4b  the page, close
class B:
    VER = 'c2'
    M = dict(px=760, py=700, scale=1.25, ang=-.62)
    TW, TH = 2500, 1700                                   # page texture
    XG = .210                                             # gutter line, metres right of frame centre
    P = {}

    @classmethod
    def cam(cls):
        pit = math.radians(52); fwd = np.array([.05, math.cos(pit), -math.sin(pit)]); fwd /= np.linalg.norm(fwd)
        D = .62
        return Camera3D(-fwd * D, (0, 0, 0), 4700 * D, cx=1010, cy=590, roll=math.radians(2.5))

    @staticmethod
    def gut(v): return .060 * (v - .5) + .020 * np.sin(np.pi * np.clip(v, -.2, 1.2) * 1.05) - .012

    @classmethod
    def surf(cls, side=-1, nu=181, nv=81):
        """left (side -1) or right page, metres.  u: 0 at the gutter outward"""
        u, v = np.meshgrid(np.linspace(0, 1, nu) ** 1.4, np.linspace(0, 1, nv))
        wid = .62; d = u * wid
        sk = cls.gut(v)                                     # the gutter runs aslant and bows
        x = cls.XG + sk + side * d
        y = .36 - v * .80
        z = .034 * (1 - np.exp(-(d / .018) ** 1.3)) - .012 * smooth(.02, .30, d) + (.0011 * np.sin(v * 9 + d * 14) + .0007 * np.sin(v * 17 - d * 30)) * smooth(.02, .2, d)
        if side > 0: z = z * 1.15 + .012 * smooth(.02, .14, d)
        z = z - .034
        return np.stack([x, y, z], -1), d, v

    @classmethod
    def pt(cls, d, v, side=-1):
        """point on the page surface (same formulas as surf)"""
        d, v = np.broadcast_arrays(np.asarray(d, float), np.asarray(v, float))
        x = cls.XG + cls.gut(v) + side * d; y = .36 - v * .80
        z = .034 * (1 - np.exp(-(d / .018) ** 1.3)) - .012 * smooth(.02, .30, d)
        if side > 0: z = z * 1.15 + .012 * smooth(.02, .14, d)
        return np.stack([x, y, z - .034], -1)

    @classmethod
    def build(cls):
        cam = cls.cam(); r = np.random.default_rng(41)
        # ---- geometry at 2x
        out, zb = None, None
        for sd in (-1, 1):
            S, d, v = cls.surf(sd); N = S3.normals(S)
            u_t = (-d if sd < 0 else d)
            att = np.dstack([u_t, v, N, S, np.full(d.shape, sd, float)])
            out, zb = S3.raster(cam, S, att, ss=1, out=out, zb=zb)
        d, v, N, X = out[..., 0], out[..., 1], out[..., 2:5], out[..., 5:8]
        right = out[..., 8] > 0
        # ---- light, as washes: moon-blue from the left, the lantern's amber from the right, violet between
        x, y = np.meshgrid(np.arange(W, dtype=f32), np.arange(H, dtype=f32))
        w1 = fbm(H, W, (380, 150, 60), (1, .5, .25), 7)
        ad = np.abs(d)
        lamp = np.array([.40, .03, .14]); L = lamp - X; dl = np.linalg.norm(L, axis=-1)
        nl = np.clip((N * L).sum(-1) / dl, 0, 1)
        am = 1 / (1 + (dl / .262) ** 2.6)                                        # 0 far .. 1 near the lamp
        s_ = np.clip(1.22 * am + .06 + .014 * w1, 0, 1)
        cols = [(0, '#938a90'), (.12, '#a59a92'), (.28, '#b9a992'), (.44, '#cdb88f'), (.62, '#dec28b'), (.80, '#ecc987'), (1, '#f8d796')]
        k = s_ * 5; kf = np.floor(k); s_p = (kf + smooth(.10, .90, k - kf)) / 5
        g = ramp(.30 * s_p + .70 * s_, cols)
        # moonlight: the left cools a little, the far corner sinks
        mo = smooth(1250, 0, x + .30 * y)
        g = g * (1 + mo[..., None] * np.array([-.050, -.018, .040], f32))
        g = g * (1 - .13 * (smooth(1, 0, x / 760) * smooth(420, 1152, y))[..., None] * np.array([1, 1, .8], f32))
        g = g * (1 - .06 * smooth(360, 0, y))[..., None]
        # curve into the gutter: a lit roll, then the turn away, then the crease
        tw_ = smooth(.085, .040, ad) * smooth(.016, .040, ad)
        gl = np.where(right, 0, 1)
        g = mix(g, np.minimum(g * np.array([1.10, 1.05, .96], f32) + .02, 1), (tw_ * gl * .8 * smooth(.2, .6, s_))[..., None])
        turn = smooth(.030, .004, ad) * gl
        g = mix(g, g * np.array([.56, .44, .46], f32), (turn * .92)[..., None])
        # facing page: its back is to the lamp, the moon finds it
        rp = ramp(np.clip(ad / .085 + .010 * w1, 0, 1), [(0, '#35211f'), (.05, '#5c4034'), (.13, '#946c49'), (.25, '#c89a62'), (.42, '#eac17e'), (.65, '#f6d390'), (1, '#fbe0a3')])
        rp = rp * (1 - .16 * smooth(500, 1152, y))[..., None]
        g = np.where(right[..., None], rp, g)
        g = mix(g, np.ones_like(g) * rgb('#2b1a1e'), (smooth(.0045, .0006, ad) * .9)[..., None])
        A = s_ * gl
        lab = np.where(right, 2, 1).astype(np.int32); lab[ad < .0012] = 3
        # broad calm washes along the page; nothing near the roll yet
        p0, _ = cam.project(np.array([[0, 0, 0], [.05, 0, 0]]))
        ax = math.atan2(*(p0[1] - p0[0])[::-1])
        guide = np.clip(g, 0, 1)
        g = repaint(guide, sizes=(44, 22), seed=5, angle=(ax + .10 * fbm(H, W, (320, 130), (1, .5), 15)).astype(f32), lab=lab, mask=((ad > .075) & ~right).astype(f32), thr=(0, .012), density=1.0, aspect=8.0,
                    jit=(.0015, .004, .0045), op=.62, dry=.14, bristle=.014, curve=.25)
        # the roll of the page and the facing page: long strokes that run along the fold, lit crest first, then the turn into the crease
        cv = Canvas(img=g)
        def along(side, d0, v0, v1, dj, n=26):
            v = np.linspace(v0, v1, n); dd = d0 + dj * np.linspace(-1, 1, n)
            return cam.project(cls.pt(np.maximum(dd, .0004), v, side))[0]
        for side, cnt, dm in ((-1, 620, .095), (1, 260, .075)):
            ds = np.sort(r.uniform(0, 1, cnt) ** 1.15 * dm)[::-1]
            for d0 in ds:
                v0 = r.uniform(-.05, .85); q = along(side, d0, v0, v0 + r.uniform(.22, .55), r.normal(0, .0012))
                L_ = 1 if side < 0 else 2
                for j in range(0, len(q) - 4, 4):
                    m = q[j + 2]; ix, iy = int(np.clip(m[0], 0, W - 1)), int(np.clip(m[1], 0, H - 1))
                    if not (0 <= m[0] < W and 0 <= m[1] < H) or lab[iy, ix] != L_: continue
                    cv.strokes(q[None, j:j + 5], r.uniform(3.5, 8.5) * (.55 + .45 * min(1, d0 / .03)), (guide[iy, ix] * (1 + r.normal(0, .024 - .014 * min(1, d0 / .07))))[None], op=float(.85 * smooth(dm * 1.0, dm * .45, d0) + .05), dry=.16, seed=int(r.integers(1 << 20)), lab=lab, LAB=np.array([L_]), bristle=.045)
        g = cv.img
        # the top leaf ends in a deckle edge just short of the fold; older leaves step down behind it into the crease
        lf = ~right
        wv = fbm(H, W, (150, 56, 18, 7), (1, .7, .45, .25), 91)
        prev = None
        for (dc, amp, col, rimc, ra) in ((.0150, .00135, '#b48a5e', '#ffe9b8', .85), (.0098, .0011, '#8a6250', '#dcae74', .5), (.0055, .0009, '#573a36', '#a5764f', .35)):
            de = ad - (dc + amp * wv); wv = np.roll(wv, 173, 0)
            px_ = cv2.GaussianBlur(np.abs(cv2.Sobel(ad, cv2.CV_32F, 1, 0, scale=1 / 8.)), (0, 0), 2) + 1e-6      # metres per px across the roll
            m_ = np.clip(.5 - de / px_, 0, 1) * lf
            tone = rgb(col) * (.80 + .32 * smooth(1152, 200, y) )[..., None]
            g = mix(g, tone * (1 + .05 * fbm(H, W, (60, 18), (1, .5), int(dc * 1e5)))[..., None], m_[..., None])
            k_ = de / px_
            brk = smooth(-1.1, .5, fbm(H, W, (90, 30), (1, .6), int(dc * 1e5) + 7))
            g = mix(g, np.ones_like(g) * rgb(rimc), (np.exp(-((k_ - 1.3) / 1.2) ** 2) * lf * ra * brk)[..., None])
            g = mix(g, g * np.array([.48, .38, .40], f32), (np.exp(-((k_ + 2.6) / 2.4) ** 2) * lf * .85)[..., None])
        g = S3.soften_edges(g, lab)
        # ---- real paper: fibres, laid texture, specks, all in page space
        tw, th = cls.TW, cls.TH
        U = np.clip((np.where(right, d, d) + .62) / 1.24, 0, 1) * (tw * 1.0 - 1); Vv = v * (th - 1)
        f1 = tile_to(fibre_tile(1024, 11, 5200, 14, 64), th, tw); f2 = tile_to(fibre_tile(1024, 23, 900, 18, 80), th, tw, 260, 80)
        f3 = fbm(th, tw, (3.4, 1.8), (1, .7), 31); f4 = fbm(th, tw, (13, 5), (1, .5), 33)
        uv = np.dstack([U, Vv])
        s1, s2, s3, s4 = (S3.tex(t.astype(f32), uv, cv2.INTER_LINEAR, cv2.BORDER_REFLECT) for t in (f1, f2, f3, f4))
        g = g * (1 - .030 * s1[..., None] * np.array([.75, .95, 1.2], f32)) + (s2 * (.004 + .016 * lum(g)))[..., None] * np.array([1, .95, .82], f32)
        g = g * (1 + .010 * s3 + .006 * s4)[..., None]
        # foxing: a few deliberate rusty blots, each with a darker heart, kept off the drawing
        fx = np.zeros((H, W), f32); fr_ = np.zeros((H, W), f32)
        for (cx, cy, rad) in ((318, 236, 9), (352, 262, 4), (300, 270, 3), (1588, 356, 8), (1556, 392, 3.5), (1180, 1004, 10), (1226, 1030, 4.5), (1150, 1038, 3), (214, 742, 5), (980, 120, 5.5), (1012, 138, 2.6), (1500, 880, 4), (640, 1010, 3.4), (1420, 170, 3)):
            th_ = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            q = np.c_[cx + np.cos(th_) * rad * 1.3 * (1 + r.normal(0, .16, 12)), cy + np.sin(th_) * rad * .9 * (1 + r.normal(0, .16, 12))]
            cv2.fillPoly(fx, [np.round(spl(q, closed=True, n=4) * 8).astype(np.int32)], float(r.uniform(.45, .8)), cv2.LINE_AA, 3)
            if rad > 4: cv2.circle(fr_, (int(cx + rad * .15), int(cy)), max(1, int(rad * .4)), .8, -1, cv2.LINE_AA)
        fx = .7 * cv2.GaussianBlur(fx, (0, 0), 1.3) + .45 * cv2.GaussianBlur(fx, (0, 0), 6); fr_ = cv2.GaussianBlur(fr_, (0, 0), 1.2)
        g = mix(g, g * np.array([.82, .68, .52], f32), np.clip(fx * .62, 0, .8)[..., None]); g = mix(g, g * np.array([.84, .70, .58], f32), np.clip(fr_ * .55, 0, .8)[..., None])
        # coarse dark flecks of bark in the pulp
        cv = Canvas(img=g); n = 26
        fxp, fyp = r.uniform(0, W, n), r.uniform(0, H, n)
        col = g[fyp.astype(int), fxp.astype(int)] * np.array([.62, .54, .52], f32)
        cv.strokes(arcs(fxp, fyp, r.uniform(0, 3.1, n), r.gamma(2, 4.5, n) + 3, r.normal(0, .7, n)), r.uniform(.5, 1.1, n), col, op=r.uniform(.25, .7, n), seed=3, shape='round')
        g = cv.img
        # the facing page: older leaves stepping up out of the valley, then its own lit face going out of focus toward the corner
        wv = fbm(H, W, (320, 120, 40), (1, .5, .22), 191)
        px_ = cv2.GaussianBlur(np.abs(cv2.Sobel(ad, cv2.CV_32F, 1, 0, scale=1 / 8.)), (0, 0), 2) + 1e-6
        for (dc, amp, ra) in ((.0050, .0004, .30), (.0090, .0005, .35), (.0145, .0006, .45)):
            k_ = (ad - (dc + amp * wv)) / px_; wv = np.roll(wv, 211, 0)
            g = mix(g, g * np.array([.62, .52, .50], f32), (np.exp(-((k_ + 2.2) / 2.2) ** 2) * right * .75)[..., None])
            g = mix(g, np.ones_like(g) * rgb('#f2d296'), (np.exp(-((k_ - 1.2) / 1.2) ** 2) * right * ra * smooth(-1., .6, fbm(H, W, (90, 30), (1, .6), int(dc * 1e5) + 9)))[..., None])
        far = (smooth(.004, .030, ad) * right)[..., None]
        soft_ = comp.bokeh(np.clip(g, 0, 1.2).astype(f32), 4.5, 0); soft2 = comp.bokeh(soft_, 7.5, 0)
        f2 = (smooth(.030, .075, ad) * right)[..., None]
        g = g * (1 - far) + (soft_ * (1 - f2) + soft2 * f2) * far
        # cockles: long soft ridges, one lit flank one shaded flank
        c = Cel(ss=1)
        for (pts, w_, al) in (([(90, 300), (420, 338), (760, 322), (1040, 356)], 6, .26), ([(300, 980), (640, 1010), (980, 986), (1330, 1030)], 6, .26), ([(980, 96), (1200, 140), (1400, 132), (1560, 168)], 5, .20)):
            q_ = spl(np.array(pts, float), n=20); q_ = q_ + np.c_[noise1(len(q_), 14, r), noise1(len(q_), 14, r)] * 3
            c.line(q_ + [0, 5], w_ * 1.3, '#6a5350', (.45, .45), seed=w_ + int(pts[0][0]), alpha=al, press=.5); c.line(q_ - [0, 3], w_ * 1.1, '#fff0c8', (.45, .45), seed=w_ + int(pts[0][1]), alpha=al * 1.3, press=.5)
        l = cv2.GaussianBlur(c.rgba(), (0, 0), 2.6); g = g * (1 - l[..., 3:]) + l[..., :3] * (.35 + .65 * lum(g)[..., None])
        # right page: sheaf of page edges just visible past the gutter
        return dict(plate=g.astype(f32), am=A.astype(f32), right=right.astype(np.uint8))

    @classmethod
    def ink(cls):
        """the line and the bird drawn on the page, warped to the screen"""
        cam = cls.cam(); S, d, v = cls.surf(-1)
        tw, th = cls.TW, cls.TH
        # sheet px: 1 px = .62/1900 m
        k = 1900 / .62
        att = np.dstack([(.62 - d) * k, v * .80 * k * 1.0])
        sw, sh_ = 1900, int(.80 * k)
        buf, zb = S3.raster(cam, S, att, ss=1)
        px, py = cls.M['px'], cls.M['py']
        f_ = lambda x, y: buf[int(y), int(x)]
        p0 = f_(px + 40, py - 14); cx = f_(1215, 520)
        run = []
        for (qx, qy, qr, *kd) in ((px + 6, py + 4, 5.5, 'w'), (px + 66, py + 4, 6.2, 'w'), (px + 126, py - 2, 4.2)):
            a = f_(qx, qy); run.append((a[0], a[1], qr) + tuple(kd))
        sc = 1.02
        # page foreshortening: draw the bird a little taller on the sheet so it reads true on screen
        st0 = [(x * sc + cx[0], y * sc * 1.16 + cx[1], rr * sc) + tuple(kk) for (x, y, rr, *kk) in IL.BIRD]
        st = Stroke(run + st0, step=.7, entry=-.2, load=(1.0, .42), depl=1.2, dwell=((.455, .10),))
        e = np.linspace(0, 1, 400); pos = e ** 1.55; pos = pos * .55 + .45 * (e * e * (3 - 2 * e))
        warp = lambda t: np.interp(t, pos, e)
        sh = Sheet(sw, sh_, K=5, margin=12, fib=fibre_tile())
        sh.add(st, 76, 105, warp)
        ex, ey = IL.BIRD_EYE
        sh.add(blob(ex * sc + cx[0], ey * sc * 1.16 + cx[1], 7.0 * sc, stretch=.34, ang=.9, seed=3), 105.6, 107)
        fr = 76 + 29 * warp(st.t)
        rr = np.random.default_rng(4)
        for (qx, qy) in ((1070, 712), (1300, 722)):
            a = f_(qx, qy); j = int(np.argmin(np.hypot(st.X[:, 0] - a[0], st.X[:, 1] - a[1])))
            sh.add(blob(a[0], a[1], rr.uniform(1.2, 2.6), stretch=.4, ang=rr.uniform(0, 3), seed=int(qx)), fr[j] + .5, fr[j] + 1.2)
        sh.bake()
        ox, oy, w, h = 700, 300, 900, 520
        b2, z2 = S3.raster(cam, S, att, w=w, h=h, ss=2, ox=ox, oy=oy)
        wm = [(S3.tex(D, b2) * (z2 > 0), S3.tex(np.minimum(F, 1e4), b2, cv2.INTER_NEAREST, val=1e4)) for D, F in sh.maps()]
        return SparseInk(wm, w, h, ox, oy, 2)

    @classmethod
    def setup(cls):
        if cls.P: return
        d = cached('b', cls.VER, cls.build)
        cls.P.update(d); cls.P['ink'] = cls.ink()

    @classmethod
    def paw_shadow(cls, img, f):
        """contact shadow: the lamp is at the right, so it lies to the left; it meets the paw as it lands"""
        M = cls.M; px, py, s = M['px'], M['py'], M['scale']
        fd = f - f % 2 - 56
        up = 1.0
        for a, v in ((2, 1.0), (4, .62), (6, .30), (8, .10), (10, 0.)):
            if fd >= a - 0: up = v
        if fd < 2: return img
        a = M['ang']; dx, dy = -math.cos(a) * 330 * s * up, -math.sin(a) * 330 * s * up
        ox, oy = -30 - 130 * up, 16 + 40 * up
        q = np.array([[96, -22], [88, -62], [44, -100], [-14, -92], [-80, -66], [-160, 10], [-380, 170], [-560, 300], [-520, 410], [-260, 280], [-100, 160], [0, 100], [74, 62]], float)
        q = spl(xf(q, s, tx=px + dx + ox, ty=py + dy + oy), closed=True)
        c = Cel(ss=1); c.fill(q, '#ffffff')
        m = cv2.GaussianBlur(c.rgba()[..., 3], (0, 0), 4 + 14 * up) * (.80 - .35 * up)
        return img * (1 - m[..., None] * (1 - np.array([.50, .46, .60], f32)))

    @classmethod
    def frame(cls, f):
        P = cls.P
        img = P['plate'].copy()
        d = f // 3
        fl = [1, .975, 1.0, .985, 1.01, .97, .99][d % 7]                         # lantern breathes on threes
        img *= (1 + (fl - 1) * np.clip(P['am'], 0, 1)[..., None] * np.array([1, .9, .6], f32))
        if f >= 76:
            ink = P['ink']; cov, wet = ink.at(f, dry_frames=20.)
            x0, y0, w, h = ink.ox, ink.oy, ink.w, ink.h
            sub = img[y0:y0 + h, x0:x0 + w]; A = P['am'][y0:y0 + h, x0:x0 + w]
            sub = sub * transmit(cov, wet) + (cov * .012)[..., None] * LAMP
            sh = sheen(cov, wet, light=(.75, -.62))
            sub = sub + (sh * (.25 + .9 * np.clip(A, 0, 1)))[..., None] * rgb('#ffdca0') + (sheen(cov, wet, light=(-.8, -.6)) * .22)[..., None] * MOON
            img[y0:y0 + h, x0:x0 + w] = sub
        img = cls.paw_shadow(img, f)
        img = foxact.draw(img, 's4b', f - 56, cls.M)
        # dust in the lantern light
        t = (f - 56) / 56.
        z = 1.0 + .055 * comp.ease(t)
        return comp.cam2d(img, W / 2 + 46 * comp.ease(t), H / 2 - 8 * comp.ease(t), zoom=z, interp=cv2.INTER_LANCZOS4)


# =====================================================================  4c  the bird leaves the page
from shots import s4_bird as birdact
BSC = birdact.BSC


class C:
    VER = 'c6'
    M = dict(x=620, y=980, h=520)
    HZ, F = .62, 2400.
    CX, CY = 1230, 333
    BP = np.array([-.005, 2.10])                           # where the bird lies on the page (metres), centre
    BW = .27                                               # bird length on the page, metres (500 units)
    P = {}

    @classmethod
    def cam(cls): return Camera3D((0, 0, cls.HZ), (0, 1, cls.HZ), cls.F, cx=cls.CX, cy=cls.CY)

    @classmethod
    def surf(cls, side=-1, nu=161, nv=61):
        u, v = np.meshgrid(np.linspace(0, 1, nu) ** 1.5, np.linspace(0, 1, nv))
        wid = 1.45 if side < 0 else 1.1
        d = u * wid
        Y = 1.30 + v * 1.48
        X = .34 + .03 * (Y - 2) + side * d
        z = .060 * (1 - np.exp(-(d / .030) ** 1.3)) - .012 * smooth(.04, .6, d)
        if side > 0: z = z * 1.25 + .02 * smooth(.02, .2, d)
        z = z - .050 + .003 * np.sin(X * 7 + Y * 3) * smooth(.1, .3, d) + (.022 * smooth(2.62, 2.78, Y) * np.sin(X * 3.3 + 1) ** 2)
        return np.stack([X, Y, z], -1), d, v

    @classmethod
    def bg(cls):
        from shots import s4_book
        return s4_book.hall_close()

    @classmethod
    def build(cls):
        cam = cls.cam(); r = np.random.default_rng(6)
        bg = cls.bg()
        out, zb = None, None
        for sd in (-1, 1):
            S, d, v = cls.surf(sd); N = S3.normals(S)
            out, zb = S3.raster(cam, S, np.dstack([np.full(d.shape, sd, float), d, N, S]), ss=1, out=out, zb=zb)
        msk = (zb > 0)
        right = out[..., 0] > 0; d = out[..., 1]; N = out[..., 2:5]; X = out[..., 5:8]
        x, y = np.meshgrid(np.arange(W, dtype=f32), np.arange(H, dtype=f32))
        w1 = fbm(H, W, (420, 150, 60), (1, .5, .25), 7, aspect=2.6)
        lamp = np.array([1.02, 2.02, .30]); L = lamp - X; dl = np.linalg.norm(L, axis=-1)
        am = 1 / (1 + (dl / .80) ** 2.5)
        nl = np.clip((N * L).sum(-1) / (dl + 1e-6), 0, 1)
        s_ = np.clip(1.12 * am * (.62 + .9 * nl) + .030 * w1, 0, 1)
        cols = [(0, '#5a5872'), (.10, '#767182'), (.22, '#99908f'), (.34, '#b9a68f'), (.48, '#d5b388'), (.68, '#ecc383'), (1, '#ffdc9a')]
        k = s_ * 7; kf = np.floor(k); s_p = (kf + smooth(.2, .8, k - kf)) / 7
        g = ramp(.55 * s_p + .45 * s_, cols)
        # moon pool near the fox, shaft landing on the page
        mp = np.exp(-(((x - 560) / 520) ** 2 + ((y - 940) / 90) ** 2))
        g = g * (1 + .30 * (mp * smooth(.5, .1, s_))[..., None] * np.array([.86, .96, 1.2], f32))
        turn = smooth(.055, .006, d) * (~right)
        g = mix(g, g * np.array([.60, .48, .52], f32), (turn * .8)[..., None])
        roll = smooth(.16, .09, d) * smooth(.035, .09, d) * (~right)
        g = mix(g, np.minimum(g * np.array([1.10, 1.06, .98], f32) + .015, 1), roll[..., None])
        rp = ramp(np.clip(d / .34 + .05 * w1, 0, 1), [(0, '#3a2838'), (.12, '#6f4a48'), (.30, '#b98355'), (.6, '#efc07a'), (1, '#ffe0a0')])
        g = np.where(right[..., None], rp, g)
        g = mix(g, np.ones_like(g) * rgb('#2a1a28'), (smooth(.010, .001, d) * msk)[..., None] * .9)
        lab = np.where(msk, np.where(right, 2, 1), 0).astype(np.int32)
        gd, _ = cam.project(np.array([[.34, 1.5, 0], [.36, 2.6, 0]])); ga = math.atan2(*(gd[1] - gd[0])[::-1]) + math.pi
        ang = .06 * fbm(H, W, (200, 80), (1, .5), 15) + (ga - .04) * smooth(.22, .08, d) * msk 
        g = repaint(np.clip(g, 0, 1), sizes=(26, 12, 6), seed=5, angle=ang, lab=lab, mask=msk.astype(f32), thr=(0, .010, .016), density=1.0, aspect=8.5, jit=(.003, .009, .010), op=.82, dry=.14, bristle=.026)
        g = S3.soften_edges(g, np.where(msk, lab, 1))
        # fibres in perspective
        tw = th = 2048
        Uv = np.dstack([(X[..., 0] + 1.3) / 2.8 * 4000 % (tw - 1), (X[..., 1] - 1.3) / 1.5 * 2140 % (th - 1)])
        f1 = tile_to(fibre_tile(1024, 11, 5200, 14, 64), th, tw); f3 = fbm(th, tw, (5, 2.4), (1, .7), 31)
        s1 = S3.tex(f1.astype(f32), Uv); s3 = S3.tex(f3, Uv)
        g = g * (1 - .055 * s1[..., None] * np.array([.75, .95, 1.2], f32)) * (1 + .013 * s3[..., None])
        # foxing + cockles in perspective
        fx = np.zeros((H, W), f32)
        for _ in range(120):
            Xp, Yp = r.uniform(-1.1, 1.2), r.uniform(1.45, 2.75)
            p, z = cam.project(np.array([Xp, Yp, 0.])); rad = r.gamma(1.3, .0022) * cls.F / Yp + .6
            cv2.ellipse(fx, (int(p[0]), int(p[1])), (max(1, int(rad * 1.3)), max(1, int(rad * .42))), 0, 0, 360, float(r.uniform(.3, .8)), -1, cv2.LINE_AA)
        fx = cv2.GaussianBlur(fx, (0, 0), 1.1); g = mix(g, g * np.array([.78, .64, .52], f32), (fx * .7)[..., None])
        c = Cel(ss=1)
        for (pts, w_, al) in (([(-1.0, 1.75), (-.5, 1.70), (-.05, 1.78), (.2, 1.74)], 4, .30), ([(-.7, 2.35), (-.3, 2.42), (.1, 2.38)], 3, .26), ([(-.2, 1.6), (.1, 1.64), (.28, 1.6)], 4, .24), ([(.5, 1.9), (.9, 2.0), (1.2, 1.96)], 3, .22)):
            q_, _ = cam.project(np.array([(a, b, .004) for a, b in pts])); q_ = spl(q_, n=16)
            c.line(q_ + [0, 3], w_, '#3c2b46', (.45, .45), seed=int(abs(pts[0][0]) * 90), alpha=al, press=.5); c.line(q_ - [0, 2], w_ * .8, '#ffe6b4', (.45, .45), seed=int(abs(pts[0][1]) * 90), alpha=al * .8, press=.5)
        l = cv2.GaussianBlur(c.rgba(), (0, 0), 1.6); g = g * (1 - l[..., 3:]) + l[..., :3] * (.35 + .65 * lum(g)[..., None])
        # far edge of the page: hand-cut, rim of moonlight; dark gap behind
        m = cv2.GaussianBlur(msk.astype(f32), (0, 0), .8)
        e = fbm(H, W, (30, 11, 4), (1, .6, .4), 61) * 1.5
        ys = np.where(msk.any(0), msk.argmax(0), H).astype(f32)
        edge = y - ys[None]
        a = np.clip(edge + e - 1.5 + .5, 0, 1) * msk
        rimc = np.exp(-((edge - 3) / 2.4) ** 2) * (.25 + .75 * smooth(1200, 500, x))
        g = g + (rimc * .20)[..., None] * MOON + (np.exp(-((edge - 3) / 2.2) ** 2) * smooth(1300, 1900, x) * .16)[..., None] * LAMP
        # the sheaf under the top leaf: a few older edges showing past it
        dist = cv2.distanceTransform((a < .5).astype(np.uint8), cv2.DIST_L2, 5)
        j1 = fbm(H, W, (110, 40, 12), (1, .6, .35), 71); j2 = fbm(H, W, (140, 46, 14), (1, .6, .35), 72); j3 = fbm(H, W, (160, 50, 14), (1, .6, .35), 73)
        lit_ = (.55 + .45 * smooth(1500, 300, x))[..., None]
        under = bg
        side_ = smooth(470, 240, x)
        for (w_, jj, col) in ((15, j3, '#37344f'), (10.5, j2, '#57546c'), (5.0, j1, '#7d7a90')):
            m_ = np.clip(w_ * side_ + 1.2 * jj * side_ - dist + .5, 0, 1) * (dist > 0) * (y < 1100)
            ln = np.exp(-((w_ * side_ + 1.2 * jj * side_ - dist) / 1.0) ** 2) * (dist > 0) * side_
            under = mix(under, rgb(col) * lit_, m_[..., None]); under = mix(under, np.ones_like(under) * rgb('#1f1c33'), (ln * .7)[..., None])
        img = under * (1 - a[..., None]) + g * a[..., None]
        # near lip of the page falls out of focus
        nb = comp.bokeh(img, 5.0, 0); t = smooth(1085, 1152, y)[..., None]
        img = img * (1 - t) + nb * t
        # what sits behind the fox is the plate; keep page alpha for shadows
        return dict(plate=img.astype(f32), page=(a).astype(f32), am=(s_ * a).astype(f32))

    # ---------------------------------------------------------------- the bird
    @classmethod
    def page_uv(cls, x0, y0, w, h, ss=2):
        """screen ROI -> bird-sprite px for a bird lying on the page"""
        cam = cls.cam()
        n = 41
        k = cls.BW / 500.                                  # metres per BIRD unit
        u, v = np.meshgrid(np.linspace(-262, 238, n), np.linspace(-160, 160, n))
        th = math.radians(8)
        X = cls.BP[0] + (u * math.cos(th) + v * math.sin(th)) * k; Y = cls.BP[1] - (-u * math.sin(th) + v * math.cos(th)) * k * 1.55
        S = np.stack([X, Y, np.zeros_like(X) + .012], -1)
        att = np.dstack([(u + 262) * BSC, (v + 160) * BSC])
        return S3.raster(cam, S, att, w=w, h=h, ss=ss, ox=x0, oy=y0)

    @classmethod
    def setup(cls):
        if cls.P: return
        cls.P.update(cached('c', cls.VER, cls.build))
        cls.P['sp'] = birdact.sprite(True); [birdact.sprite(b_) for b_ in birdact.BODY]
        cls.ROI = (880, 900, 640, 252)
        b, z = cls.page_uv(*cls.ROI)
        cls.P['uv'] = b; cls.P['uvm'] = (z > 0).astype(f32)

    @classmethod
    def ghost(cls, img, amt, shift=(0, 0), ink=1.0, boil_=None):
        """bird lying on the page: ink line (ink=1), and/or the pale place it left (amt)"""
        P = cls.P; sp = P['sp']; x0, y0, w, h = cls.ROI
        uv = P['uv']
        if boil_ is not None:
            uv = uv + boil_
        sub = img[y0:y0 + h, x0:x0 + w]
        if amt > 0:
            a = S3.down(S3.tex(sp['a'], uv) * P['uvm'], 2) * amt
            rim = S3.down(S3.tex(sp['rim'], uv) * P['uvm'], 2) * amt
            new = np.minimum(sub * np.array([1.13, 1.15, 1.20], f32) + .03, 1.0)
            sub = sub * (1 - a[..., None]) + new * a[..., None]
            sub = sub * (1 - .34 * rim[..., None] * np.array([.6, .8, .9], f32))
        if ink > 0:
            cov = S3.down(S3.tex(sp['cov'], uv) * P['uvm'], 2)
            sub = sub * (1 - ink * (1 - transmit(cov)))
        img[y0:y0 + h, x0:x0 + w] = sub
        return img

    @classmethod
    def frame(cls, f):
        P = cls.P; k = f - 112; d = k // 2
        img = P['plate'].copy()
        fl = [1, .975, 1.0, .985, 1.01, .97, .99][(f // 3) % 7]
        img *= (1 + (fl - 1) * np.clip(P['am'], 0, 1)[..., None] * np.array([1, .9, .6], f32))
        img = cls.fox_shadow(img)
        img = foxact.draw(img, 's4c', k, cls.M)
        img = birdact.draw(img, f, cls)
        return img

    @classmethod
    def fox_shadow(cls, img):
        M = cls.M; x, y, h = M['x'], M['y'], M['h']
        q = np.array([[.46, .0], [.36, -.030], [.10, -.034], [-.2, -.026], [-.42, -.012], [-.95, -.004], [-1.15, .012], [-.8, .030], [-.3, .040], [.1, .036], [.4, .024]], float)
        q = spl(q * h + (x, y), closed=True)
        c = Cel(ss=1); c.fill(q, '#ffffff'); m = cv2.GaussianBlur(c.rgba()[..., 3], (0, 0), 3.2) * .62 * cls.P['page']
        return img * (1 - m[..., None] * (1 - np.array([.52, .45, .60], f32)))


# =====================================================================  4a  the book on its lectern
from shots import s4_book as bookset


class A:
    VER = 'a5'
    M = dict(h=250, x_in=-160, floor_y=1040, x_crouch=470, x_land=935, x_paw=1080, page_y=(586, 578))
    P = {}

    @classmethod
    def setup(cls):
        if cls.P: return
        cls.P.update(cached('a', cls.VER, bookset.build))

    @classmethod
    def fox_pos(cls, f):
        """where the fox is (mirrors the lead's path through the marks): x, y, height above what it is over"""
        M = cls.M; fd = f - f % 2
        if fd < 14: t = fd / 14.; return M['x_in'] + (M['x_crouch'] - M['x_in']) * t ** .9, M['floor_y'], 0.
        if fd < 22: return M['x_crouch'], M['floor_y'], 0.
        if fd < 34:
            t = (fd - 22) / 12.; y0, y1 = M['floor_y'], M['page_y'][0]
            return M['x_crouch'] + (M['x_land'] - M['x_crouch']) * t, y0 + (y1 - y0) * t - 4 * t * (1 - t) * (M['h'] * .34), 1.
        t = float(comp.ease((fd - 41) / 9.)) if fd > 41 else 0.
        return M['x_land'] + (M['x_paw'] - M['x_land']) * t, M['page_y'][0] + (M['page_y'][1] - M['page_y'][0]) * t, 0.

    @classmethod
    def shadow(cls, img, f):
        M = cls.M; h = M['h']; x, y, air = cls.fox_pos(f); fd = f - f % 2
        P = cls.P
        def blot(cx, cy, w_, amt, soft, page, tilt=0.):
            q = np.array([[.50, .0], [.40, -.045], [.10, -.055], [-.2, -.05], [-.5, -.03], [-.95, -.012], [-1.05, .014], [-.7, .04], [-.3, .055], [.1, .052], [.42, .035]], float) * [w_, w_ * .9]
            q = boil(xf(q, 1, rot=tilt) + (cx, cy), .8, fd)
            c = Cel(int(w_ * 2.4), int(w_ * .8), int(cx - w_ * 1.5), int(cy - w_ * .4), ss=1); c.fill(spl(q, closed=True), '#ffffff')
            m = cv2.GaussianBlur(c.rgba()[..., 3], (0, 0), soft) * amt
            x0, y0 = max(0, c.ox), max(0, c.oy); x1, y1 = min(W, c.ox + c.w), min(H, c.oy + c.h)
            if x1 <= x0 or y1 <= y0: return
            m = m[y0 - c.oy:y1 - c.oy, x0 - c.ox:x1 - c.ox]
            pg = P['page'][y0:y1, x0:x1]; m = m * (pg if page else (1 - pg) * (1 - .0 * pg))
            img[y0:y1, x0:x1] *= (1 - m[..., None] * (1 - np.array([.44, .40, .52], f32)))
        if not air: blot(x, y + 1, h, .68, 2.6, y < 800, -.03)
        else:
            t = (fd - 22) / 12.
            blot(x - 40 * t, M['floor_y'] + 4, h * (1 - .25 * t), .55 * (1 - t) ** 1.4, 4 + 9 * t, False)
            blot(x, M['page_y'][0] + 2, h * (.7 + .3 * t), .62 * max(0, t * 1.6 - .6), 11 - 8 * t, True)
        return img

    @classmethod
    def motes(cls, img, f):
        """dust turning in the moon shaft: few, slow, on twos"""
        d = f // 2; r = np.random.default_rng(91)
        c = Cel(ss=1)
        for j in range(22):
            x0, y0 = r.uniform(230, 1010), r.uniform(60, 640); vx, vy = r.normal(.9, .4), r.normal(.75, .35); ph = r.uniform(0, 6)
            x = x0 + vx * d + 5 * math.sin(ph + d * .21); y = y0 + vy * d + 3 * math.sin(ph * 2 + d * .17)
            k = np.exp(-((x - 250 - y * .98) / 250.) ** 2) * (.45 + .4 * math.sin(ph + d * .33))
            if k > .05: c.fill(spl(np.array([[x - 1.7, y], [x, y - 1.2], [x + 1.6, y + .2], [x, y + 1.3]]) + r.normal(0, .2, (4, 2)), closed=True, n=4), '#c9d6f2', float(min(.75, k)))
        return comp.over(img, c.layer())

    @classmethod
    def frame(cls, f):
        P = cls.P; img = P['plate'].copy()
        fl = [1, .975, 1.0, .985, 1.012, .965, .99][(f // 3) % 7]
        img *= (1 + (fl - 1) * np.clip(P['am'], 0, 1)[..., None] * np.array([1, .9, .6], f32))
        img = cls.shadow(img, f)
        img = foxact.draw(img, 's4a', f, cls.M)
        img = cls.motes(img, f)
        # the near pile slides a little faster than the room as the camera eases right and in
        t = comp.ease(f / 56.)
        fg = cv2.warpAffine(P['fg'], np.float32([[1, 0, -34 * t], [0, 1, 5 * t]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        img = comp.over(img, fg)
        z = 1.0 + .030 * t
        return comp.cam2d(img, W / 2 + 22 * t, H / 2 - 8 * t, zoom=z, interp=cv2.INTER_LANCZOS4)


class Shot:
    id = 's4'; n = 168
    finish = dict(hal_amt=.22, vig=.12)
    M4a = A.M; M4b = B.M; M4c = C.M

    def setup(self):
        for k in (A, B, C): k.setup()

    def frame(self, f):
        return (A.frame(f) if f < 56 else B.frame(f) if f < 112 else C.frame(f)).astype(f32)
