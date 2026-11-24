"""S8b  Across the top of one wing: huge pages settle and overlap like feathers, sun raking from ahead."""
import numpy as np, cv2, math, os
from numba import njit, prange
from lp import W, H
from lp.paint import f32, rgb, mix, smooth, ramp, grid, fbm, Canvas, arcs, jit_color, _vn
from lp import comp, gb_sky as G, gb_art
from lp.cam import Camera3D
from lp.ink import spl, over_roi

VER = 'v7'
AW, AH = 720, 965
KINDS = [(3, 6), (0, 11), (3, 1), (2, 1), (0, 12), (3, 9), (0, 15), (3, 12), (6, 3), (3, 4), (0, 13), (9, 0), (3, 2), (0, 16), (3, 5), (8, 2), (3, 10), (0, 14), (9, 2), (2, 4), (0, 17), (3, 11), (2, 8), (5, 2), (3, 0), (3, 3)]


# ------------------------------------------------------------------ art: ink, wash, distance from the torn edge
def big_art():
    os.makedirs('cache', exist_ok=True)
    fn = f'cache/s8b_art_{AW}_{len(KINDS)}_{VER}.npy'
    if os.path.exists(fn): return np.load(fn)
    out = []
    for i, (k, s) in enumerate(KINDS):
        r = np.random.default_rng(500 + i)
        a = gb_art.page(k, s, AW, AH)
        # hand-cut / torn outline
        def edge(n, amp):
            t = np.linspace(0, 1, n); e = np.zeros(n)
            for fr, am in ((2.3, .5), (7, .28), (19, .16), (47, .10)): e += am * np.sin(t * fr * 6.283 + r.uniform(0, 6.3))
            return e * amp
        m = 10.
        P = np.vstack([np.c_[np.linspace(m, AW - m, 90), m + edge(90, 3.2)], np.c_[AW - m + edge(120, 3.4), np.linspace(m, AH - m, 120)],
                       np.c_[np.linspace(AW - m, m, 90), AH - m + edge(90, 3.6)], np.c_[m + edge(120, 3.4), np.linspace(AH - m, m, 120)]])
        M = np.zeros((AH, AW), np.uint8); cv2.fillPoly(M, [np.round(P * 8).astype(np.int32)], 255, cv2.LINE_AA, 3)
        # nibbled bites and a torn corner
        for _ in range(int(r.integers(1, 4))):
            side = int(r.integers(0, 4)); t = r.uniform(.15, .85)
            c = [(t * AW, m), (AW - m, t * AH), (t * AW, AH - m), (m, t * AH)][side]
            b = gb_art.blob(c[0], c[1], r.uniform(6, 16), r.uniform(6, 14), r, 9, .25)
            cv2.fillPoly(M, [np.round(b * 8).astype(np.int32)], 0, cv2.LINE_AA, 3)
        if r.random() < .5:
            q = r.uniform(26, 60); cx, cy = [(AW, AH), (0, AH), (AW, 0), (0, 0)][int(r.integers(0, 4))]; sx, sy = (-1 if cx else 1), (-1 if cy else 1)
            T = spl([(cx, cy), (cx + sx * q * 1.3, cy), (cx + sx * q * .75, cy + sy * q * .42), (cx + sx * q * .4, cy + sy * q * .8), (cx, cy + sy * q * 1.2)], closed=True, corners=(0, 1, 4))
            cv2.fillPoly(M, [np.round(T * 8).astype(np.int32)], 0, cv2.LINE_AA, 3)
        d = cv2.distanceTransform((M > 127).astype(np.uint8), cv2.DIST_L2, 5)
        d = np.clip(d * 5, 0, 255).astype(np.uint8)
        # creases: fold lines drawn, plus a tone step in the wash
        p = gb_art.Pen(AW, AH)
        for _ in range(int(r.integers(1, 4))):
            if r.random() < .5: y = r.uniform(20, 114); Q = spl([(0, y), (30, y + r.normal(0, 1.5)), (70, y + r.normal(0, 1.5)), (100, y + r.normal(0, 3))])
            else: x = r.uniform(10, 90); Q = spl([(x, 0), (x + r.normal(0, 2), 45), (x + r.normal(0, 2), 90), (x + r.normal(0, 4), 134)])
            p.line(Q, .34, .55, (.1, .1), .3, .6)
            p.line(Q + [0.5, 0.5], .8, .16, (.1, .1), .3, .6)
        cr = p.out()[..., 0]
        out.append(np.dstack([np.maximum(a[..., 0], (cr * .8).astype(np.uint8)), a[..., 1], d]))
    A = np.stack(out)
    t = fn + f'.{os.getpid()}.tmp.npy'; np.save(t, A); os.replace(t, fn)
    return A


_MIP = None


def mips():
    global _MIP
    if _MIP is None:
        A = big_art(); L = [A]
        for l in range(1, 4): L.append(np.stack([cv2.resize(a, (AW >> l, AH >> l), interpolation=cv2.INTER_AREA) for a in A]))
        off = np.cumsum([0] + [a.size for a in L[:-1]])
        _MIP = (np.concatenate([a.ravel() for a in L]), off.astype(np.int64), np.array([a.shape[2] for a in L], np.int64), np.array([a.shape[1] for a in L], np.int64), len(A))
    return _MIP


# ------------------------------------------------------------------ rasteriser for big bent pages (alpha-tested edges)
@njit(cache=True, fastmath=True, parallel=True)
def _rast(S, NL, ART, FLIP, AD, off0, aw, ah, Zi, ID, UU, VV, NB, nb):
    h, w = Zi.shape
    n, gr, gc = S.shape[0], S.shape[1], S.shape[2]
    bh = (h + nb - 1) // nb
    for b in prange(nb):
        y_lo = b * bh; y_hi = min(h, y_lo + bh) - 1
        for i in range(n):
            for a in range(gr - 1):
                for c in range(gc - 1):
                    for k in range(2):
                        if k == 0: a0, c0, a1, c1, a2, c2 = a, c, a + 1, c, a + 1, c + 1
                        else: a0, c0, a1, c1, a2, c2 = a, c, a + 1, c + 1, a, c + 1
                        x0 = S[i, a0, c0, 0]; y0 = S[i, a0, c0, 1]; z0 = S[i, a0, c0, 2]
                        x1 = S[i, a1, c1, 0]; y1 = S[i, a1, c1, 1]; z1 = S[i, a1, c1, 2]
                        x2 = S[i, a2, c2, 0]; y2 = S[i, a2, c2, 1]; z2 = S[i, a2, c2, 2]
                        if z0 <= 0 or z1 <= 0 or z2 <= 0: continue
                        iy0 = max(y_lo, int(math.floor(min(y0, min(y1, y2))))); iy1 = min(y_hi, int(math.ceil(max(y0, max(y1, y2)))))
                        if iy0 > iy1: continue
                        ix0 = max(0, int(math.floor(min(x0, min(x1, x2))))); ix1 = min(w - 1, int(math.ceil(max(x0, max(x1, x2)))))
                        if ix0 > ix1: continue
                        ar = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
                        if abs(ar) < 1e-6: continue
                        ia = 1.0 / ar
                        u0 = a0 / (gr - 1); u1 = a1 / (gr - 1); u2 = a2 / (gr - 1)
                        v0 = c0 / (gc - 1); v1 = c1 / (gc - 1); v2 = c2 / (gc - 1)
                        n0 = NL[i, a0, c0]; n1 = NL[i, a1, c1]; n2 = NL[i, a2, c2]
                        for y in range(iy0, iy1 + 1):
                            for x in range(ix0, ix1 + 1):
                                px = x + .5; py = y + .5
                                w1 = ((px - x0) * (y2 - y0) - (x2 - x0) * (py - y0)) * ia
                                w2 = ((x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)) * ia
                                w0 = 1 - w1 - w2
                                if w0 < -1e-4 or w1 < -1e-4 or w2 < -1e-4: continue
                                z = w0 * z0 + w1 * z1 + w2 * z2
                                if z <= Zi[y, x]: continue
                                # perspective-correct uv
                                u = (w0 * u0 * z0 + w1 * u1 * z1 + w2 * u2 * z2) / z
                                v = (w0 * v0 * z0 + w1 * v1 * z1 + w2 * v2 * z2) / z
                                uu = 1 - u if FLIP[i] & 1 else u
                                vv = 1 - v if FLIP[i] & 2 else v
                                tx = min(aw - 1, max(0, int(uu * aw))); ty = min(ah - 1, max(0, int(vv * ah)))
                                if AD[off0 + ((ART[i] * ah + ty) * aw + tx) * 3 + 2] < 2: continue
                                Zi[y, x] = z; ID[y, x] = i
                                UU[y, x] = int(min(1.0, max(0.0, u)) * 65535); VV[y, x] = int(min(1.0, max(0.0, v)) * 65535)
                                NB[y, x] = w0 * n0 + w1 * n1 + w2 * n2


@njit(cache=True, fastmath=True, parallel=True)
def _shadow(SM, Zi, ID, cam_f, cx, cy, lcx, lcy, lcz, TX, steps):
    """screen-space march toward the sun, at half the buffer resolution"""
    h, w = SM.shape
    H2, W2 = Zi.shape
    for y in prange(h):
        for x in range(w):
            yb = y * 2 + 1; xb = x * 2 + 1
            if ID[yb, xb] < 0: continue
            z = 1.0 / Zi[yb, xb]
            X = (xb - cx) / cam_f * z; Y = (yb - cy) / cam_f * z
            for k in range(steps.shape[0]):
                s = steps[k]
                zz = z + lcz * s
                if zz < .15: break
                xx = int((X + lcx * s) / zz * cam_f + cx); yy = int((Y + lcy * s) / zz * cam_f + cy)
                if xx < 0 or yy < 0 or xx >= W2 or yy >= H2: break
                if ID[yy, xx] < 0: continue
                df = zz - 1.0 / Zi[yy, xx]
                if df > .014 + .0018 * z + s * .03 and df < TX:
                    SM[y, x] = 255
                    break


@njit(cache=True, fastmath=True, parallel=True)
def _shade(OUT, Zi, ID, UU, VV, NB, C, ART, FLIP, AD, OFF, AWs, AHs, LV, WPX, LPX, HZ, hzc, cam_f, cx, cy, SM, seed, SIDE, NEAR, rx, ry):
    """C[n, 7, 3]: hot, lit, half, shade, cast, ink, trans"""
    h, w = Zi.shape
    for y in prange(h):
        for x in range(w):
            i = ID[y, x]
            if i < 0: continue
            z = 1.0 / Zi[y, x]
            u = UU[y, x] / 65535.0; v = VV[y, x] / 65535.0
            nl = NB[y, x] + (_vn(u * 3.1 + i * 7.3, v * 4.7 + i * 3.1, 5) - .5) * .10
            tone = 0 if nl > .95 else 1 if nl > .30 else 2 if nl > .08 else 3
            shadow = SM[y, x] > 127
            col = tone
            if shadow: col = 4
            elif NEAR[i] == 1 and tone < 3:
                ub = .60 + .36 * v * v + .030 * math.sin(v * 7.0 + i) + .05 * (_vn(v * 6, i * 1.7, 3) - .5)
                uq = u if SIDE[i] > 0 else 1 - u
                if uq > ub and v < .96: col = 4
            r = C[i, col, 0]; g = C[i, col, 1]; b = C[i, col, 2]
            if SIDE[i] < 0 and not shadow and tone >= 3:      # sun through the sheet
                r = C[i, 6, 0]; g = C[i, 6, 1]; b = C[i, 6, 2]
            # gouache: flat-brush drag along the page
            br = (_vn(u * 46 + i * 3.7, v * 5 + i, 11) * .6 + _vn(u * 13, v * 17 + i * 2.1, 13) * .4 - .5) * .035
            r *= 1 + br; g *= 1 + br; b *= 1 + br
            # art
            uu = u + (_vn(u * 7 + i, v * 10, seed + 5) - .5) * .014 + (_vn(u * 23 + i, v * 31, seed + 9) - .5) * .004
            vv = v + (_vn(u * 8, v * 9 + i, seed + 6) - .5) * .012 + (_vn(u * 27, v * 29 + i, seed + 11) - .5) * .004
            uu = 1 - uu if FLIP[i] & 1 else uu
            vv = 1 - vv if FLIP[i] & 2 else vv
            l = LV[i]; tw = AWs[l]; th = AHs[l]
            fx = uu * tw - .5; fy = vv * th - .5
            x0 = int(math.floor(fx)); y0 = int(math.floor(fy)); fx -= x0; fy -= y0
            a0 = 0.0; a1 = 0.0; a2 = 0.0
            for dy in range(2):
                for dx in range(2):
                    xi = min(tw - 1, max(0, x0 + dx)); yi = min(th - 1, max(0, y0 + dy))
                    q = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
                    o = OFF[l] + ((ART[i] * th + yi) * tw + xi) * 3
                    a0 += q * AD[o]; a1 += q * AD[o + 1]; a2 += q * AD[o + 2]
            a0 /= 255.0; a1 /= 255.0
            sc = LPX[i] / 965.0                        # px per atlas texel
            # keep the drawn line crisp when it is magnified
            if sc > .55:
                pr = _vn(u * 11 + i * 3.1, v * 15, 23)
                a0 = (a0 - .5 + (pr - .5) * .34) * max(1.15, min(2.2, sc)) + .5
                if a0 < 0: a0 = 0.0
                if a0 > 1: a0 = 1.0
            fade = 1 - HZ[i]
            ir = C[i, 5, 0]; ig = C[i, 5, 1]; ib = C[i, 5, 2]
            k = a1 * .34
            r += (ir - r) * k * .50; g += (ig - g) * k * .62; b += (ib - b) * k * .80
            k = a0 * .66 * (.35 + .65 * fade)
            r += (ir - r) * k; g += (ig - g) * k; b += (ib - b) * k
            # contour of the sheet
            dist = a2 / 5.0 * sc
            nz = _vn(u * 14 + i, v * 18, seed + 3)
            lw = (1.5 + 2.4 * nz) * (.5 + .5 * fade) * min(1.0, max(.35, LPX[i] / 420.0))
            if nz < .14: lw *= .3
            if NEAR[i] == 1: lw *= .30
            aa = lw - dist + .5
            if aa > 0:
                if aa > 1: aa = 1.0
                aa *= .9 * (.4 + .6 * fade)
                r += (ir * .8 - r) * aa; g += (ig * .72 - g) * aa; b += (ib * .8 - b) * aa
            # burning edge on the sun side
            rr = (4.5 + 7.0 * _vn(x * .01, y * .01, 31)) * min(1.0, LPX[i] / 500.0)
            xx = int(x + rx * rr); yy = int(y + ry * rr)
            if 0 <= xx < w and 0 <= yy < h and not shadow:
                j = ID[yy, xx]
                if j < 0 or (j != i and 1.0 / Zi[yy, xx] - z > .10 + .02 * z):
                    r = C[i, 0, 0]; g = C[i, 0, 1]; b = C[i, 0, 2]
            # air
            q = HZ[i]
            r += (hzc[0] - r) * q; g += (hzc[1] - g) * q; b += (hzc[2] - b) * q
            OUT[y, x, 0] = r; OUT[y, x, 1] = g; OUT[y, x, 2] = b; OUT[y, x, 3] = 1.0


# ------------------------------------------------------------------ the wing top
# world: x = chord (right = trailing edge), y = span (away), z = up
def surf(x, y):
    xx = x - cx_(y)
    return -.030 * np.maximum(xx + 1.5, 0) ** 2 - .012 * np.maximum(-xx - 3, 0) ** 2 + .0016 * y * y


def cx_(y):
    """the wing curves toward the sun as it goes out"""
    return .0105 * np.maximum(y - 3, 0) ** 2 * .9


ROWS = [  # chord of the base, length, width, sweep(deg), span step
    (1.7, 7.4, 2.5, 20, 1.55), (-.2, 5.2, 2.35, 17, 1.50), (-2.0, 4.1, 2.15, 14, 1.40), (-3.6, 3.4, 1.95, 12, 1.30), (-5.0, 2.8, 1.75, 10, 1.20), (-6.2, 2.4, 1.6, 8, 1.10), (-7.3, 2.1, 1.45, 7, 1.0), (-8.3, 1.8, 1.3, 5, .95)]


class WingTop:
    def __init__(self, seed=17):
        r = np.random.default_rng(seed)
        D = dict(b=[], L=[], W=[], th=[], row=[], art=[], flip=[], ta=[], rj=[], cu=[], j=[], tw=[])
        na = len(KINDS)
        for ri, (cb, L, Wd, sw, st) in enumerate(ROWS):
            y = -3.0 + r.uniform(0, 1) + .5 * st * (ri % 2); j = 0; last = -1
            while y < 46:
                g = 1 + .0 * y
                a = int(r.integers(0, na))
                while a == last: a = int(r.integers(0, na))
                last = a
                D['b'].append((cb + r.normal(0, .10), y)); D['L'].append(L * r.uniform(.92, 1.12)); D['W'].append(Wd * r.uniform(.92, 1.14))
                D['th'].append(math.radians(sw + np.interp(y, [0, 12, 40], [0, 8, 26])) + r.normal(0, .05)); D['row'].append(ri); D['art'].append(a); D['flip'].append(1 if r.random() < .6 else 2)
                D['rj'].append(r.normal(0, 1, 8)); D['cu'].append(r.uniform(0, 1)); D['j'].append(j); D['tw'].append(r.normal(0, .07))
                # arrival: a wave that comes in from the far trailing edge, row over row, and ends at our feet
                t = -30. if ri < 2 else 44 + 4.2 * ri + np.interp(y, [-1, 3, 6, 10, 15, 22, 30], [8.0, 6.0, 4.5, 2.0, -3.0, -12, -60]) + r.normal(0, .7)
                D['ta'].append(t)
                y += st * r.uniform(.88, 1.12) * (1 + .012 * max(y, 0)); j += 1
        # the wing tip far ahead: long fingers fanning out into the sky
        for k in range(13):
            u = k / 12.
            D['b'].append((np.interp(u, [0, 1], [1.2, -6.5]), np.interp(u, [0, 1], [41., 46.]))); D['L'].append(8.5 + 6.5 * math.sin(math.pi * min(1, u * .8 + .15))); D['W'].append(2.5)
            D['th'].append(math.radians(np.interp(u, [0, 1], [34, 112]))); D['row'].append(2); D['art'].append(int(r.integers(0, na))); D['flip'].append(1)
            D['rj'].append(r.normal(0, .4, 8)); D['cu'].append(.1); D['j'].append(k); D['tw'].append(.16); D['ta'].append(-60.)
        self.D = {k: np.array(v) for k, v in D.items()}
        self.n = len(D['L'])

    def mesh(self, d, gr=7, gc=9):
        """world mesh at drawing time d"""
        D = self.D; n = self.n
        a = np.linspace(-.5, .5, gr)[None, :, None]; l = np.linspace(0, 1, gc)[None, None, :]
        s = d - D['ta']                                   # time since touch-down
        fl = 9.0
        q = np.clip(1 + s / fl, 0, 1)                      # 0..1 in flight
        # hinge angle of the sheet and bend along it
        rest = .035 + .05 * D['cu'] ** 2
        e = q ** 2.2
        bounce = np.where(s > 0, np.exp(-np.maximum(s, 0) * .30) * np.sin(np.maximum(s, 0) * .85), 0)
        hinge = (1 - e) * 1.30 + rest + .17 * np.maximum(bounce, 0) - .5 * np.clip((-fl - s) / 12., 0, 1) * np.sin(s * .4 + D['rj'][:, 5] * 3)
        bend = (1 - e) * .95 * np.sin(q * 2.6 + .4) + .22 * bounce + .10 * (D['rj'][:, 0] * .5 + .3) + .035 * np.sin(d * .42 + D['rj'][:, 1] * 3) * (s > 6)
        offx = -(1 - q) ** 1.6 * 2.6; offz = (1 - q) ** 1.4 * (2.2 + .5 * D['rj'][:, 2]); offy = -(1 - q) ** 1.5 * 1.4
        roll = (1 - q) ** 1.5 * (.6 * D['rj'][:, 3])
        # before that: it comes in from behind our left shoulder, a tumbling sheet
        k = np.clip((-fl - s) / 12., 0, 1)
        offx = offx - 17 * k ** 1.15; offy = offy - 4.5 * k; offz = offz + (2.2 + 1.6 * D['rj'][:, 2]) * k ** 1.2
        roll = roll + 1.5 * k * np.sign(D['rj'][:, 3] + 1e-3) + .35 * np.sin(s * .55 + D['rj'][:, 1] * 4) * np.clip(-s / 9., 0, 1)
        hinge = hinge[:, None, None]; bend = bend[:, None, None]
        L = D['L'][:, None, None]; Wd = D['W'][:, None, None]
        # profile along the sheet: starts at -slope of surface, bends
        ang = hinge * (1 - .35 * l) - .06 + bend * (l ** 1.6)
        dl = 1.0 / (gc - 1)
        ca = np.cumsum(np.concatenate([np.zeros((n, 1, 1)), np.cos(ang[..., :-1])], -1), -1) * dl
        sa = np.cumsum(np.concatenate([np.zeros((n, 1, 1)), np.sin(ang[..., :-1])], -1), -1) * dl
        along = L * ca; up = L * sa
        # corner curl at the tip, and a soft ripple across
        cc = D['rj'][:, 4][:, None, None]
        up = up + L * .060 * np.clip(l - .55, 0, 1) ** 2 * 6 * (np.clip(a * 2 * np.sign(cc), 0, 1) ** 2) * np.abs(cc)
        up = up - Wd * .16 * np.abs(a * 2) ** 1.25 * (.5 + .5 * l) * (.75 + .3 * D['rj'][:, 7][:, None, None].clip(-1, 1)) + .03 * np.sin(a * 5 + D['rj'][:, 5][:, None, None] * 2) * l + roll[:, None, None] * a * Wd
        ac = a * Wd * (.86 + .14 * np.clip(l / .25, 0, 1)) * (1 - .34 * np.clip((l - .55) / .45, 0, 1) ** 2) + D['rj'][:, 6][:, None, None] * .05 * l
        along = along - L * .10 * (np.abs(a) * 2) ** 2 * np.clip((l - .6) / .4, 0, 1) ** 1.5
        th = D['th'][:, None, None] + D['tw'][:, None, None] * l
        bx = D['b'][:, 0] + cx_(D['b'][:, 1]); by = D['b'][:, 1]
        x = bx[:, None, None] + offx[:, None, None] + along * np.cos(th) + ac * np.sin(th)
        y = by[:, None, None] + offy[:, None, None] + along * np.sin(th) - ac * np.cos(th)
        ri = D['row'][:, None, None]
        base = surf(x, y) + .085 * ri + .0008 * np.maximum(y - 4, 0) ** 2 * math.sin((d - 50) * .085)
        ov = .016 * (D['j'][:, None, None] % 40) * 0 + a * -.055          # each sheet tips a little over the nearer one
        z = base + up + offz[:, None, None] + ov * 1.0 + (by[:, None, None] - y) * 0
        return np.stack([x, y, z], -1), q, s


CAMB = dict(eye=(-1.2, -1.2, 2.75), tgt=(2.6, 12., .1), f=1180.)
LW = np.array([-.60, .62, .36]); LW = LW / np.linalg.norm(LW)


def camera(f):
    t = (f - 48) / 28.
    e = np.array([-.9, -2.4, 4.1]) + np.array([.5, 2.2, -.35]) * t
    g = np.array([3.0, 11.5, -.1]) + np.array([.6, 2.2, -.05]) * t
    return Camera3D(e, g, 1180., cx=1024, cy=576, roll=-.05 + .03 * t)


PAL = ['#fff3d2', '#f7e3c0', '#dcb699', '#a08cae', '#a88aa8', '#7b4f3f', '#f3bf80']


def background():
    h, w = G.PH, G.PW; r = np.random.default_rng(61)
    hz = 330 + 99; sun = (470 + 176, 232 + 99)
    img = G.sky(h, w, [(0, '#8fbccb'), (.35, '#c6d3c6'), (.65, '#f4d9b0'), (1, '#fcca8c')], hz, sun, seed=14, warm='#ffecc0', warm_r=900, warm_a=.85, n=(700, 1200))
    img = G.paint_sun(img, sun, 40, r, rings=(('#fff3d4', 1.5, .62), ('#ffeac0', 2.3, .44), ('#ffe3b2', 3.6, .32), ('#fedcaa', 5.8, .22)))
    far = np.zeros((h, w, 4), f32); ft = G.T('#fff0d2', '#f9d9b2', '#e8bfa4', '#d3a9aa', '#c0a0b2')
    for pr, sd in (([(1300, 405, 120, 50), (1420, 395, 90, 60), (1180, 415, 90, 34)], 3), ([(700, 412, 130, 40), (560, 404, 90, 56), (820, 418, 90, 26)], 4), ([(2250, 400, 140, 70), (2120, 412, 110, 40), (1990, 420, 90, 30)], 5)):
        G.comp_over(far, G.cloud(h, w, pr, r, ft, (.7, -.4, 0), seed=sd, foot=(405, 445), thr=(.62, .44, .22, .0), mist='#f9d3a0', rimcol='#fff6da', rimw=3, kids=(9, 5), face=4, sizes=(8, 4, 3)))
    G.comp_over(img, far)
    def tf(d): return np.stack([mix(a, b, d ** .6) for a, b in zip(G.T('#fff0c8', '#fbd9aa', '#efc3a2', '#d9aeaa', '#cba6b2'), G.T('#f6dcc6', '#e0bcb4', '#bda3b8', '#9a8fb6', '#807fae'))])
    s = G.sea(h, w, hz, 1500, r, sun, tf, lambda d: mix(rgb('#fad0a0'), rgb('#8f88b6'), float(d) ** .5), rows=26, lobe=120, seed=23, lane=1.1, power=1.5)
    y = np.arange(h, dtype=f32)[:, None]
    s[..., :3] += (rgb('#fbd5a4') * s[..., 3:4] - s[..., :3]) * (np.exp(-np.maximum(0, y - hz) / 60.)[..., None] * .8)
    s[..., :3] += (rgb('#e9c7b4') * s[..., 3:4] - s[..., :3]) * .16
    G.comp_over(img, s)
    return img


class ShotB:
    def setup(self):
        fn = f'cache/s8b_bg_{VER}.npy'
        if not os.path.exists(fn):
            t = fn + f'.{os.getpid()}.tmp.npy'; np.save(t, background().astype(np.float16)); os.replace(t, fn)
        self.bg = np.load(fn).astype(f32); self.w = WingTop(); mips()

    def frame(self, f):
        d = f - (f % 2)
        cam = camera(f); t = (f - 48) / 28.
        img = comp.cam2d(self.bg, G.PW / 2 + 26 - 30 * t, G.PH / 2 + 4 + 16 * t, 1.02 + .025 * t, rot=.045 - .03 * t)
        D = self.w.D
        V, q, s = self.w.mesh(d)
        ok = (D['b'][:, 1] > cam.eye[1] - 4.5)
        # a page that has not yet taken off is not there
        ok &= s > -21.0
        XY, z = cam.project(V)
        ok &= (z.min((1, 2)) > .25) & (XY[..., 0].max((1, 2)) > -50) & (XY[..., 0].min((1, 2)) < W + 50) & (XY[..., 1].max((1, 2)) > -50) & (XY[..., 1].min((1, 2)) < H + 50)
        idx = np.nonzero(ok)[0]
        V, XY, z = V[idx], XY[idx], z[idx]; n = len(idx)
        r = np.random.default_rng(900 + d)
        XY = XY + r.normal(0, .35, XY.shape)                     # redrawn each time
        e1 = np.gradient(V, axis=1); e2 = np.gradient(V, axis=2)
        N = np.cross(e2, e1); N = N / (np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9)
        N = N * np.sign(N[:, 3:4, 4:5, 2:3] + 1e-9)
        vw = cam.eye - V[:, 3, 4]; side = np.sign((N[:, 3, 4] * vw).sum(-1))
        rr = np.random.default_rng(31)
        tilt = rr.normal(0, .075, self.w.n)[idx]
        NL = ((N @ LW) * side[:, None, None] + tilt[:, None, None] * np.minimum(1, q[idx])[:, None, None]).astype(f32)
        ss = 2
        S = np.empty(XY.shape[:3] + (3,), f32)
        S[..., :2] = XY * ss; S[..., 2] = np.where(z > .25, 1 / np.maximum(z, .25), -1)
        h_, w_ = H * ss, W * ss
        Zi = np.zeros((h_, w_), f32); ID = np.full((h_, w_), -1, np.int32); UU = np.zeros((h_, w_), np.uint16); VV = np.zeros((h_, w_), np.uint16); NB = np.zeros((h_, w_), f32)
        AD, OFF, AWs, AHs, NA = mips()
        art = D['art'][idx].astype(np.int64); flip = D['flip'][idx].astype(np.int64)
        _rast(S, NL, art, flip, AD, OFF[0], AW, AH, Zi, ID, UU, VV, NB, 16)
        wp = np.linalg.norm(XY[:, -1, 4] - XY[:, 0, 4], axis=-1) * ss; lp = np.linalg.norm(XY[:, 3, -1] - XY[:, 3, 0], axis=-1) * ss
        px = np.maximum(np.minimum(lp, wp * 1.34), 1).astype(f32)
        lv = np.clip(np.log2(965. / np.maximum(lp, wp * 1.34) * .9), 0, 3).astype(np.int64)
        zc = z[:, 3, 4]; hz = (np.clip((zc - 9) / 34., 0, 1) ** .9 * .72).astype(f32)
        C = np.tile(np.stack([rgb(c) for c in PAL])[None], (n, 1, 1)).astype(f32)
        jv = rr.normal(0, .022, (self.w.n, 1))[idx]; age = (rr.random((self.w.n, 1)) ** 1.8)[idx]
        C[:, 1] += (rgb('#ecc98e') - C[:, 1]) * age * .75; C[:, 2] += (rgb('#d3a680') - C[:, 2]) * age * .6
        C[:, :5] += (jv + np.array([[.35, -.05, -1.]]) * rr.normal(0, .012, (self.w.n, 1))[idx])[:, None]
        lc = cam.R @ LW
        # the fourteen nearest sheets get a hand-inked contour and their own shadow shape
        area = wp * lp; order = np.argsort(-np.where(z.min((1, 2)) > .4, area / np.maximum(z[:, 3, 4], .3), 0))
        near = np.zeros(n, np.int64); near[order[:14]] = 1
        OUT = np.zeros((h_, w_, 4), f32)
        SM = np.zeros((H, W), np.uint8)
        _shadow(SM, Zi, ID, cam.f * ss, cam.cx * ss, cam.cy * ss, float(lc[0]), float(lc[1]), float(lc[2]), 1.5, np.r_[np.arange(.02, 1.2, .02), np.arange(1.2, 2.6, .045)])
        SM = cv2.GaussianBlur(cv2.medianBlur(SM, 7), (0, 0), 2.6)
        SM = cv2.resize(SM, (w_, h_), interpolation=cv2.INTER_LINEAR)
        l2 = np.array([lc[0], lc[1]]); l2 = l2 / np.hypot(*l2)
        _shade(OUT, Zi, ID, UU, VV, NB, C, art, flip, AD, OFF, AWs, AHs, lv, (wp).astype(f32), px, hz, rgb('#f6d3a6').astype(f32), cam.f * ss, cam.cx * ss, cam.cy * ss,
               SM, d, side, near, float(l2[0]), float(l2[1]))
        lay = cv2.resize(OUT, (W, H), interpolation=cv2.INTER_AREA)
        img = comp.over(img, lay)
        return self.ink(img, ID, Zi, np.nonzero(near)[0], d)

    def ink(self, img, ID, Zi, who, d):
        """pressure contour: thick where the sheet turns under, thinning and lifting along the lit edge"""
        from lp.ink import Cel
        id1 = ID[::2, ::2]; z1 = Zi[::2, ::2]
        cel = Cel(W, H, ss=2); k = 0
        for i in who:
            m = (id1 == i).astype(np.uint8)
            if m.sum() < 500: continue
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cs:
                if len(c) < 70: continue
                P = c[:, 0].astype(float); N_ = len(P)
                g = np.exp(-np.linspace(-2, 2, 13) ** 2); g /= g.sum()
                Q = np.stack([np.convolve(np.pad(P[:, j], 6, 'wrap'), g, 'valid') for j in (0, 1)], 1)
                tg = np.roll(Q, -3, 0) - np.roll(Q, 3, 0); tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-6; nm = np.c_[tg[:, 1], -tg[:, 0]]
                # outward normal: leaves the mask
                a = np.round(Q + nm * 4).astype(int); inb = (a[:, 0] >= 0) & (a[:, 0] < W) & (a[:, 1] >= 0) & (a[:, 1] < H)
                ins = np.zeros(N_, bool); ins[inb] = m[a[inb, 1], a[inb, 0]] > 0
                if ins.mean() > .5: nm = -nm
                o = np.round(Q + nm * 4).astype(int); qi = np.round(Q - nm * 3).astype(int)
                inb = (o[:, 0] >= 0) & (o[:, 0] < W) & (o[:, 1] >= 0) & (o[:, 1] < H) & (qi[:, 0] >= 0) & (qi[:, 0] < W) & (qi[:, 1] >= 0) & (qi[:, 1] < H)
                own = np.zeros(N_, bool)
                own[inb] = z1[qi[inb, 1], qi[inb, 0]] > z1[o[inb, 1], o[inb, 0]] * 1.002          # we are in front here
                edge = (Q[:, 0] < 3) | (Q[:, 0] > W - 4) | (Q[:, 1] < 3) | (Q[:, 1] > H - 4); own &= ~edge
                # shade side (down, right) carries the weight; lit side is thin and broken
                wt = np.clip(.5 + .5 * (nm @ np.array([.62, .78])), 0, 1)
                idx = np.nonzero(own)[0]
                if len(idx) < 12: continue
                for run in np.split(idx, np.nonzero(np.diff(idx) > 4)[0] + 1):
                    if len(run) < 14: continue
                    q = Q[run]; w_ = wt[run]; mw = float(w_.mean()); t_ = np.linspace(0, 1, len(run))
                    wf = lambda t, w_=w_, t_=t_: .30 + 1.0 * np.interp(t, t_, w_)
                    cel.line(q, 4.6, '#6a4038', (.10, .22), .40, seed=int(i) * 7 + d + k, tremor=.9, alpha=.50 + .42 * mw, wfun=wf); k += 1
        l = cel.rgba()
        return img * (1 - l[..., 3:4]) + l[..., :3]
