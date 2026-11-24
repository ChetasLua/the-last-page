"""Book and shelf painter (library lane).  Every book is its own thing: width, height, lean, sag, swollen pages,
torn head, bands, worn unreadable title marks, moss.  Painted with real strokes (lp.paint), label-masked.

COORDINATES   shelf metres: u right, v up, d toward the viewer (0 = front of the row).  T(u, v, d) -> (..., 2) screen px.
    T = flat(x0, y0, ppm, vp=(vx, vy), k=.18)      shelf seen flat-on; (x0,y0) = px of u=0,v=0; ppm = px per metre;
                                                   vp = vanishing point px, k = how fast depth shrinks toward it
    T = plane(cam, p0, eu, ev, ed)                 any shelf in 3-D through lp.cam.Camera3D (world metres)
LIGHT         L = Light(fn, floor, side, air)      fn(x, y) -> (n,3) rgb multiplier at screen px (or a colour, constant);
                                                   side=-1 light from the left / +1 right; air(x,y) -> (colour, k) mist
PAINTING      P = Painter(h, w, base='#1c1a30')    supersampled surface (P.img is ss x bigger, P.image() -> 1x rgb,
                                                   P.layer() -> premultiplied RGBA when built with layer=True)
    items = make_row(seed, width, clear_h)         list[Item], metres; tune fill=, messy=, hue=, unit=, moss=
    paint_row(P, T, items, L, v0=0., sag=0.)       paints books (back to front, sides + tops where visible)
    paint_bay(P, T, u0, u1, v0, [row heights], L, seed, ...)   posts, boards, dark back, books, board shadows
    P.strip(A, B, col, L, n=5)                     a plank / beam / post between two screen polylines A and B
    P.glaze(poly, rgb_mul)                         a crisp multiply shadow
MOVING BOOKS  (inked cel, same item, same colours, so it can replace the painted one; leave it out with skip={index})
    paint_row(P, T, items, L, skip={3})            hole where book 3 was
    book_cel(cel, T, items[3], L, seed=drawing, du=, dv=, dd=, drot=, bend=)     pose offsets in metres / radians;
                                                   dd>0 pulls it out of the row; bend bows the spine (m at mid height)
    draw_order(T, items)                           indices back to front (use it when you draw a row of cels)
    book_free(cel, cx, cy, w, h, rot, item, L, open_=0., bend=0., seed=0)        a loose flying/falling closed book (px)
EXAMPLE       see qa/t_books.py
"""
import numpy as np, math, cv2
from dataclasses import dataclass, field
from .paint import f32, rgb, mix, Canvas, _strokes, _NOLAB, _NOA, arcs
from .ink import Cel, spl, resample, noise1

PAL = dict(oxblood='#6b3436', moss='#4c5540', indigo='#333a5e', ochre='#8a6a3a', plum='#54405c', grey='#5d6a66',
           bone='#b9a98a', brown='#5a4036', slate='#4a5670', rust='#8a5340', sage='#77805f', dusk='#6f5f7a')
_names = list(PAL)
_wts = np.array([.16, .10, .12, .09, .11, .10, .07, .10, .06, .05, .03, .03]); _wts = _wts / _wts.sum()
GILT = rgb('#c8a45a')


# ------------------------------------------------------------------ mapping
def flat(x0, y0, ppm, vp=None, k=.18):
    """flat-on shelf; d shrinks toward vp (px) by k per metre"""
    vp = (x0, y0) if vp is None else vp

    def T(u, v, d=0.):
        u, v, d = np.broadcast_arrays(*[np.asarray(q, float) for q in (u, v, d)])
        x = x0 + u * ppm; y = y0 - v * ppm
        s = 1. / np.maximum(.05, 1 - d * k)
        return np.stack([vp[0] + (x - vp[0]) * s, vp[1] + (y - vp[1]) * s], -1)
    T.vp = vp
    return T


def plane(cam, p0, eu, ev, ed):
    p0, eu, ev, ed = [np.asarray(q, float) for q in (p0, eu, ev, ed)]

    def T(u, v, d=0.):
        u, v, d = np.broadcast_arrays(*[np.asarray(q, float) for q in (u, v, d)])
        W3 = p0 + u[..., None] * eu + v[..., None] * ev + d[..., None] * ed
        return cam.project(W3)[0]
    T.cam, T.p0, T.eu, T.ev, T.ed = cam, p0, eu, ev, ed
    return T


class Light:
    def __init__(self, fn='#6a6a96', floor='#100e1e', side=-1, air=None):
        if not callable(fn):
            c = rgb(fn) if isinstance(fn, str) else np.asarray(fn, f32)
            fn = lambda x, y, c=c: np.broadcast_to(c, np.shape(x) + (3,))
        self.fn, self.side, self.air = fn, side, air
        self.floor = rgb(floor) if isinstance(floor, str) else np.asarray(floor, f32)

    def __call__(self, alb, x, y, k=1.0):
        """albedo (n,3) at screen px -> painted colour"""
        alb = np.atleast_2d(np.asarray(alb, f32))
        x = np.broadcast_to(np.asarray(x, f32), (len(alb),)); y = np.broadcast_to(np.asarray(y, f32), (len(alb),))
        k = np.broadcast_to(np.asarray(k, f32), (len(alb),))[:, None]
        c = self.floor + alb * k * np.asarray(self.fn(x, y), f32)
        if self.air is not None:
            ac, ak = self.air(x, y)
            ak = np.asarray(ak, f32).reshape(-1, 1)
            c = c * (1 - ak) + np.asarray(ac, f32) * ak
        return np.clip(c, 0, 1.4)


# ------------------------------------------------------------------ surface
class Painter:
    def __init__(self, h, w, ss=2, base='#1c1a30', layer=False, pid=False):
        self.h, self.w, self.ss = h, w, ss
        self.img = np.zeros((h * ss, w * ss, 3), f32)
        self.img[:] = rgb(base) if isinstance(base, str) else base
        self.a = np.zeros((h * ss, w * ss), f32) if layer else None
        self.lab = np.zeros((h * ss, w * ss), np.int32)
        self.pid = np.zeros((h * ss, w * ss), np.int16) if pid else None
        self.cur_pid = 0
        self.n = 1
        self._q = []                         # queued strokes
        self.seed = 1

    def _fx(self, pts):
        return np.round(np.clip(np.asarray(pts, float) * self.ss, -30000, 30000) * 16).astype(np.int32).reshape(-1, 1, 2)

    def poly(self, pts, col, own=True):
        """crisp fill; returns label id"""
        p = self._fx(pts)
        c = tuple(float(v) for v in (rgb(col) if isinstance(col, str) else col))
        cv2.fillPoly(self.img, [p], c, cv2.LINE_8, 4)
        if self.a is not None: cv2.fillPoly(self.a, [p], 1.0, cv2.LINE_8, 4)
        if self.pid is not None: cv2.fillPoly(self.pid, [p], self.cur_pid, cv2.LINE_8, 4)
        self.n += 1
        if own: cv2.fillPoly(self.lab, [p], int(self.n), cv2.LINE_8, 4)
        return self.n

    def q(self, P, hw, col, op=1., dry=0., lab=-1, shape='flat', bristle=.06):
        """queue one or many strokes (screen px)"""
        P = np.asarray(P, f32)
        if P.ndim == 2: P = P[None]
        self._q.append((P, np.broadcast_to(np.asarray(hw, f32), (len(P),)), np.broadcast_to(np.asarray(col, f32), (len(P), 3)),
                        np.broadcast_to(np.asarray(op, f32), (len(P),)), np.broadcast_to(np.asarray(dry, f32), (len(P),)),
                        np.broadcast_to(np.asarray(lab, np.int32), (len(P),)), shape, bristle))

    def flush(self):
        """rasterise queued strokes in order, grouped into runs of equal shape / point count"""
        i = 0; Q = self._q
        while i < len(Q):
            j = i
            while j < len(Q) and Q[j][6] == Q[i][6] and Q[j][0].shape[1] == Q[i][0].shape[1] and Q[j][7] == Q[i][7]: j += 1
            P = np.concatenate([q[0] for q in Q[i:j]]) * self.ss
            wd, col, op, dry, lab = [np.concatenate([q[k] for q in Q[i:j]]) for k in range(1, 6)]
            n = len(P)
            sd = (np.arange(n) * 7919 + self.seed * 104729).astype(np.int64) & 0x7ffffff
            self.seed += 1
            sh = {'flat': 0, 'round': 1, 'leaf': 2, 'taper': 3}[Q[i][6]]
            _strokes(self.img, self.a if self.a is not None else _NOA, P.astype(f32), (wd * self.ss).astype(f32).copy(), col.astype(f32).copy(),
                     op.astype(f32).copy(), dry.astype(f32).copy(), sd, self.lab, lab.astype(np.int32).copy(), self.a is not None, sh, P.shape[1], float(Q[i][7]))
            i = j
        self._q = []
        return self

    def glaze(self, pts, mul, op=1.0, soft=0.):
        """multiply a polygon by a colour (crisp, optional soft edge in px)"""
        p = np.asarray(pts, float) * self.ss
        m = int(soft * self.ss * 3) + 2
        x0, y0 = np.floor(p.min(0)).astype(int) - m; x1, y1 = np.ceil(p.max(0)).astype(int) + m
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(self.w * self.ss, x1), min(self.h * self.ss, y1)
        if x1 - x0 < 2 or y1 - y0 < 2: return
        k = np.zeros((y1 - y0, x1 - x0), f32)
        cv2.fillPoly(k, [np.round((p - (x0, y0)) * 16).astype(np.int32).reshape(-1, 1, 2)], 1.0, cv2.LINE_AA, 4)
        if soft > 0: k = cv2.GaussianBlur(k, (0, 0), soft * self.ss)
        mul = rgb(mul) if isinstance(mul, str) else np.asarray(mul, f32)
        s = self.img[y0:y1, x0:x1]
        s *= 1 - k[..., None] * op * (1 - mul)

    def strip(self, A, B, col, L=None, n=5, val=.05, hi=0., lo=0., dry=.2, seed=0, op=.8, cover=.9, seg=(.35, .95), fill=True, wob=0.):
        """plank-like surface between polylines A and B (screen px, same direction); strokes run along it.
        hi / lo: brightness multiplier added along edge A / B (painted arris)."""
        A, B = np.asarray(A, float), np.asarray(B, float)
        K = max(7, min(25, len(A)))
        rA = _rs(A, K); rB = _rs(B, K)
        r = np.random.default_rng(seed * 31 + 7)
        col = rgb(col) if isinstance(col, str) else np.asarray(col, f32)
        c0 = (rA.mean(0) + rB.mean(0)) / 2
        lit = (lambda a, p, k=1.: L(a, p[:, 0], p[:, 1], k)) if L is not None else (lambda a, p, k=1.: np.atleast_2d(a) * np.reshape(k, (-1, 1)))
        if wob > 0:
            A = A + r.normal(0, wob, A.shape); B = B + r.normal(0, wob, B.shape)
        ring = np.vstack([A, B[::-1]])
        lab = self.poly(ring, lit(col, c0[None])[0]) if fill else -1
        wpx = np.hypot(*(rA - rB).T)
        if fill and L is not None:                     # a first coat that follows the light along the length
            t = (np.arange(6) + .5) / 6
            for tt in t:
                i0 = np.clip(tt * (K - 1), 0, K - 1.001); k0 = int(i0)
                pa = rA[k0] + (rA[k0 + 1] - rA[k0]) * (i0 - k0); pb = rB[k0] + (rB[k0 + 1] - rB[k0]) * (i0 - k0)
                d = (rA[-1] - rA[0]) / 6 * .62
                m = (pa + pb) / 2
                Pp = np.stack([m - d, m, m + d])
                self.q(Pp, np.hypot(*(pa - pb)) * .75, lit(col, m[None])[0], 1., 0., lab)
        m = max(1, int(round(n * cover * 1.6)))
        q = np.clip((np.arange(m) + r.uniform(-.4, 1.4, m)) / m, .02, .98)
        for qq in r.permutation(q):
            ln = r.uniform(*seg); s0 = r.uniform(0, 1 - ln)
            tt = np.linspace(s0, s0 + ln, 5) * (K - 1)
            k0 = np.minimum(tt.astype(int), K - 2); fr = (tt - k0)[:, None]
            pa = rA[k0] * (1 - fr) + rA[k0 + 1] * fr; pb = rB[k0] * (1 - fr) + rB[k0 + 1] * fr
            P = pa * (1 - qq) + pb * qq
            a = np.clip(col * (1 + r.normal(0, val)) + r.normal(0, .008) * np.array([1, 0, -1]), 0, 1)
            sh = 1.0
            self.q(P, max(.35, wpx.mean() / n * r.uniform(.45, .9)), lit(a.astype(f32), P.mean(0)[None], sh)[0], op * r.uniform(.6, 1), dry * r.uniform(0, 1.6), lab)
        for e, amt, E in ((rA, hi, rB), (rB, lo, rA)):
            if amt == 0: continue
            w = max(.4, min(2.2, wpx.mean() * .07))
            P = e * .94 + E * .06
            for s0, s1 in ((0, .52), (.48, 1.)):
                tt = np.linspace(s0, s1, 5) * (K - 1); k0 = np.minimum(tt.astype(int), K - 2); fr = (tt - k0)[:, None]
                Pq = P[k0] * (1 - fr) + P[k0 + 1] * fr
                self.q(Pq, w, lit(np.clip(col * (1 + amt), 0, 1), Pq.mean(0)[None])[0], .85, .25, lab)
        return lab

    def image(self):
        self.flush()
        return cv2.resize(self.img, (self.w, self.h), interpolation=cv2.INTER_AREA) if self.ss > 1 else self.img.copy()

    def layer(self):
        self.flush()
        rgba = np.dstack([self.img * self.a[..., None], self.a])
        return cv2.resize(rgba, (self.w, self.h), interpolation=cv2.INTER_AREA) if self.ss > 1 else rgba

    def pid1(self):
        return self.pid[self.ss // 2::self.ss, self.ss // 2::self.ss].copy()


def _rs(A, K):
    if len(A) == K: return A
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(A, axis=0).T))]
    t = np.linspace(0, s[-1], K)
    return np.c_[np.interp(t, s, A[:, 0]), np.interp(t, s, A[:, 1])]


# ------------------------------------------------------------------ items
@dataclass
class Item:
    kind: str                # 'up' upright | 'flat' lying | 'face' cover facing out
    u: float; v: float       # pivot foot position
    w: float; h: float; d: float
    lean: float = 0.         # + = top to the right
    set: float = 0.          # setback from the row front, m
    col: np.ndarray = None
    col2: np.ndarray = None
    bands: tuple = ()
    label: tuple = None
    torn: float = 0.
    swell: float = 0.
    moss: float = 0.
    wear: float = .4
    fore: bool = False       # flat book shows its page edge
    seed: int = 0


def _colour(r, hue=None):
    nm = _names[r.choice(len(_names), p=_wts)] if hue is None or r.random() < .25 else hue[r.integers(len(hue))]
    c = rgb(PAL[nm]).copy()
    g = c.mean()
    c = g + (c - g) * r.uniform(.5, 1.02)
    c = c * r.uniform(.74, 1.22) + r.normal(0, .014, 3) + r.normal(0, .012) * np.array([1, .2, -1])
    return np.clip(c, .05, .82).astype(f32)


def make_item(r, kind='up', w=.045, h=.26, hue=None, moss=0.):
    col = _colour(r, hue)
    style = r.random()
    nb = 0
    if style < .62: nb = int(r.integers(2, 6))
    elif style < .8: nb = int(r.integers(1, 3))
    bands = []
    if nb:
        base = np.linspace(.1, .9, nb) if r.random() < .6 else np.sort(r.uniform(.07, .93, nb))
        sty = int(r.integers(0, 4))            # 0 raised, 1 gilt line, 2 dark line, 3 pale cloth
        for t in base:
            bands.append((float(np.clip(t + r.normal(0, .012), .04, .96)), float(r.uniform(.008, .024)), sty if r.random() < .85 else int(r.integers(0, 4))))
    label = None
    if r.random() < .38:
        t0 = r.uniform(.46, .74); label = (float(t0), float(t0 + r.uniform(.07, .2)), int(r.integers(0, 3)))
    c2 = _colour(r, ('ochre', 'bone', 'oxblood', 'brown', 'plum')) if r.random() < .8 else np.clip(col * 1.5, 0, .8)
    return Item(kind, 0., 0., float(w), float(h), float(r.uniform(.14, .23)), 0., float(abs(r.normal(0, .012))), col, c2.astype(f32), tuple(bands), label,
                torn=float(r.uniform(.2, 1) if r.random() < .22 else r.uniform(0, .12)), swell=float(r.normal(.02, .05)),
                moss=float(moss * r.uniform(0, 1.2)) if r.random() < .5 else 0., wear=float(r.uniform(.1, 1)), seed=int(r.integers(1 << 30)))


def make_row(seed, width, clear_h, fill=.94, messy=.4, hue=None, unit=1.0, moss=0., hmin=.52):
    """one shelf of individually different books.  unit scales thickness (folios 1.5, pocket books .7)"""
    r = np.random.default_rng(seed)
    out = []; u = r.uniform(0, .02) * unit
    while u < width - .012 * unit:
        m = r.random()
        room = width - u
        if m < .05 + .10 * (1 - fill) * 4:                      # gap
            u += min(room, r.uniform(.02, .16) * unit * (1 + 2 * (1 - fill))); continue
        if m < .05 + .16 * messy and room > .2 * unit:           # a flat pile
            L = r.uniform(.17, .28) * min(1, unit); L = min(L, room * .96)
            v = 0.; top = clear_h * r.uniform(.25, .85)
            lst = []
            while v < top:
                th = r.uniform(.02, .06) * unit
                if v + th > clear_h * .9: break
                it = make_item(r, 'flat', L * r.uniform(.72, 1.0), th, hue, moss)
                it.u = u + r.uniform(0, max(.001, L - it.w)); it.v = v; it.lean = r.normal(0, .012); it.fore = r.random() < .38
                it.set = float(r.uniform(-.015, .03))
                lst.append(it); v += th
            out += lst
            u += L + r.uniform(.002, .02); continue
        if m < .10 + .34 * messy and room > .18 * unit:          # leaning run
            n = int(r.integers(1, 6)); th = r.uniform(.07, .34) * (1 if r.random() < .5 else -1)
            hs = clear_h * r.uniform(hmin, .93, n) * .97; ws = r.gamma(3.2, .0135, n).clip(.014, .1) * unit
            if th > 0:
                uu = u
                for i in range(n):
                    it = make_item(r, 'up', ws[i], hs[i], hue, moss); it.lean = th + r.normal(0, .012)
                    uu += ws[i] / math.cos(it.lean); it.u = uu; out.append(it)
                u = uu + hs[-1] * math.sin(th)
            else:
                uu = u + hs[0] * math.sin(-th)
                for i in range(n):
                    it = make_item(r, 'up', ws[i], hs[i], hue, moss); it.lean = th + r.normal(0, .012)
                    it.u = uu; uu += ws[i] / math.cos(th); out.append(it)
                u = uu
            u += r.uniform(0, .01); continue
        if m < .955 or room < .22:                               # upright run, shelf-mates of related size and hue
            n = int(r.integers(2, 13)); hb = clear_h * r.uniform(hmin + .08, .94)
            fam = None if r.random() < .55 else [_names[r.choice(len(_names), p=_wts)] for _ in range(2)]
            for i in range(n):
                w = float(r.gamma(3.0, .0150)); w = min(max(w, .013), .115) * unit
                if u + w > width: w = width - u
                if w < .009 * unit: u = width; break
                h = min(clear_h * .965, max(clear_h * .4, hb * (1 + r.normal(0, .07))))
                it = make_item(r, 'up', w, h, fam or hue, moss); it.u = u + w; it.lean = r.normal(0, .010)
                if it.lean <= 0: it.u = u
                out.append(it); u += w + abs(r.normal(0, .0012))
            continue
        it = make_item(r, 'face', r.uniform(.13, .2) * min(unit, 1.1), clear_h * r.uniform(.6, .85), hue, moss)   # cover facing out
        it.u = u; it.lean = r.normal(0, .02); out.append(it); u += it.w + r.uniform(.005, .03)
    return out


# ------------------------------------------------------------------ geometry of one book
def frame(it, du=0., dv=0., drot=0., bend=0.):
    """F(s, t) -> (u, v): s across the spine 0..1, t up 0..1 (flat books: s along the length, t up the thickness)"""
    th = it.lean + drot
    w, h = it.w, it.h
    c, sn = math.cos(th), math.sin(th)
    if it.kind == 'up':
        rf = it.u if it.lean > 0 else it.u + w                     # right foot at rest (items that lean right store it)
        if th > 0: O = np.array([rf - w * c, it.v + w * sn])       # tips over its right foot
        else: O = np.array([rf - w, it.v])                         # or over its left
        ex = np.array([c, -sn]); ey = np.array([sn, c])
        sw = it.swell

        def F(s, t):
            s, t = np.broadcast_arrays(np.asarray(s, float), np.asarray(t, float))
            b = np.sin(np.pi * np.clip(t, 0, 1))
            ss = s + (s - .5) * sw * b
            uu = ss * w + bend * b * (1 if h > 0 else 1)
            return O[0] + du + uu * ex[0] + t * h * ey[0], O[1] + dv + uu * ex[1] + t * h * ey[1]
        return F
    O = np.array([it.u, it.v])

    def F(s, t):
        s, t = np.broadcast_arrays(np.asarray(s, float), np.asarray(t, float))
        x = s * w; y = t * h + bend * np.sin(np.pi * np.clip(s, 0, 1))
        return O[0] + du + x * c + y * sn * .3, O[1] + dv - x * sn * .6 + y * c
    return F


def outline_st(it, r):
    """closed outline in (s, t) with a worn, torn, hand-cut edge"""
    tj = .006 + .02 * it.wear
    if it.kind == 'up':
        n = 5
        top = []
        for s in np.linspace(0, 1, n):
            top.append((s, 1 + r.normal(0, tj * .5)))
        if it.torn > .15:
            k = int(r.integers(0, n - 1)); dd = it.torn * r.uniform(.02, .09)
            top[k] = (top[k][0], top[k][1] - dd); top[min(n - 1, k + 1)] = (top[min(n - 1, k + 1)][0], top[min(n - 1, k + 1)][1] - dd * r.uniform(.2, 1))
        j = lambda: r.normal(0, .022)
        pts = [(0 + j(), 0 + r.normal(0, .003)), (j(), .33), (j(), .7)] + top + [(1 + j(), .7), (1 + j(), .33), (1 + j(), 0 + r.normal(0, .003))]
        pts[3] = (pts[3][0] + abs(j()), pts[3][1]); pts[3 + n - 1] = (pts[3 + n - 1][0] - abs(j()), pts[3 + n - 1][1])
        return np.array(pts)
    j = lambda a=.03: r.normal(0, a)
    return np.array([(0 + j(.006), j()), (.35, j()), (.7, j()), (1 + j(.006), j()), (1 + j(.008), .5), (1 + j(.006), 1 + j()), (.66, 1 + j()), (.3, 1 + j()), (0 + j(.006), 1 + j()), (j(.008), .5)])


def draw_order(T, items, u0=0., v0=0., d0=0.):
    """indices back to front: a book's visible side is covered by its neighbour nearer the vanishing point"""
    n = len(items)
    if n == 0: return []
    u = np.array([i.u + u0 + (i.w / 2 if (i.kind != 'up' or i.lean <= 0) else -i.w / 2) for i in items]); v = np.full(n, v0 + .12)
    a = T(u, v, d0); dd = T(u, v, d0 - .2) - a; eu = T(u + .05, v, d0) - a; ev = T(u, v + .05, d0) - a
    det = eu[:, 0] * ev[:, 1] - eu[:, 1] * ev[:, 0]
    au = (dd[:, 0] * ev[:, 1] - dd[:, 1] * ev[:, 0]) / (det + 1e-9)
    for i in range(1, n):
        if items[i].kind == 'flat' and items[i].v > 0 and items[i - 1].kind == 'flat': au[i] = au[i - 1]
    return sorted(range(n), key=lambda i: (-abs(round(float(au[i]), 4)), items[i].v))


def paint_row(P, T, items, L, v0=0., sag=0., width=None, skip=(), depth=None, tops='auto', seed=0, lod=1.0, top_shadow=.22, clear_h=None, u0=0., d0=0.):
    """paint a row of items standing on v0 (metres).  sag: downward bow of the board at mid-span."""
    if not items: return
    width = max(i.u + i.w for i in items) if width is None else width

    def TT(u, v, d):
        q = np.clip(np.asarray(u, float) / max(width, 1e-3), 0, 1)
        return T(np.asarray(u) + u0, np.asarray(v) + v0 - sag * 4 * q * (1 - q), np.asarray(d) + d0)
    # which side of the books do we see?  compare u-shift of depth
    c0 = TT(width * .5, .1, 0.); c1 = TT(width * .5, .1, -.2)
    base_u = TT(width * .5 + .1, .1, 0.) - c0; base_v = TT(width * .5, .2, 0.) - c0
    dd = c1 - c0
    order = draw_order(T, items, u0, v0, d0)
    det = base_u[0] * base_v[1] - base_u[1] * base_v[0]
    au = (dd[0] * base_v[1] - dd[1] * base_v[0]) / (det + 1e-9)        # depth direction in (u, v) of the face
    av = (base_u[0] * dd[1] - base_u[1] * dd[0]) / (det + 1e-9)
    Q = []
    for oi, i in enumerate(order):
        if i in skip: continue
        it = items[i]
        r = np.random.default_rng(it.seed)
        F = frame(it)
        df = -it.set
        M = lambda st, d=df: TT(*F(st[..., 0], st[..., 1]), d)
        wpx = float(np.hypot(*(M(np.array([1., .5])) - M(np.array([0., .5])))))
        hpx = float(np.hypot(*(M(np.array([.5, 1.])) - M(np.array([.5, 0.])))))
        cen = M(np.array([.5, .5]))
        if not (-200 < cen[0] < P.w + 200 and -200 < cen[1] < P.h + 200): continue
        # depth faces
        dback = df - it.d * (1 if it.kind != 'face' else .2)
        if it.kind in ('up', 'flat'):
            e0, e1 = M(np.array([[0., .5], [1., .5]])), M(np.array([[0., .5], [1., .5]]), dback)
            ax = (e0[1] - e0[0]) / (np.hypot(*(e0[1] - e0[0])) + 1e-6)
            s = 1. if float((e1[1] - e0[1]) @ ax) > 0 else 0.
            e = np.array([[s, 0.], [s, .5], [s, 1.]])
            a_, b_ = M(e), M(e, dback)
            ln = np.hypot(*(a_ - b_).T).mean()
            if ln > .8:
                col = it.col * .62 if not (it.kind == 'flat') else (rgb('#b9a98a') * .6)
                cc = L(col, *b_.mean(0))[0]
                P.poly(np.vstack([a_, b_[::-1]]), cc)
        tf = M(np.array([.5, 1.])); tb = M(np.array([.5, 1.]), dback); bvn = base_v / (np.hypot(*base_v) + 1e-6)
        vis_top = float((tb - tf) @ bvn) > .6 and (tops is not False)
        if vis_top:
            e = np.array([[.03, 1.], [.5, 1.], [.97, 1.]])
            a_, b_ = M(e), M(e, dback)
            ln = np.hypot(*(a_ - b_).T).mean()
            if ln > .8:
                pg = rgb('#b9a98a') * (.62 + .25 * r.random()) if it.kind == 'up' else it.col * 1.15
                lb = P.poly(np.vstack([a_, b_[::-1]]), L(pg, *((a_ + b_) / 2).mean(0))[0])
                if wpx > 7:
                    for _ in range(3):
                        ss = r.uniform(.1, .9); pq = np.stack([M(np.array([ss, 1.])), M(np.array([ss, 1.]), dback)])
                        Q.append((np.stack([pq[0], pq.mean(0), pq[1]]), max(.4, wpx * .05), pg * r.uniform(.75, 1.1), .7, .2, lb, 1.))
        ring = outline_st(it, r)
        pts = M(ring)
        c = L(it.col, *cen)[0]
        lab = P.poly(pts, c)
        _book_marks(Q, it, M, wpx, hpx, lab, L, r, lod)
    _emit(P, Q, L)
    # shadow of the board above, across the heads (crisp, uneven)
    if top_shadow and clear_h:
        r = np.random.default_rng(seed + 5)
        n = 14; us = np.linspace(0, width, n)
        lo = clear_h * (1 - top_shadow * (1 + r.normal(0, .12, n)))
        pts = np.vstack([TT(us, np.full(n, clear_h * 1.02), 0.), TT(us, lo, 0.)[::-1]])
        P.glaze(pts, (.60, .58, .68), .85)


def _emit(P, Q, L):
    if not Q: return
    by = {}
    col = np.array([q[2] for q in Q], f32)
    cen = np.array([q[0].mean(0) for q in Q], f32)
    k = np.array([q[6] for q in Q], f32)
    cc = L(col, cen[:, 0], cen[:, 1], k)
    for q, c in zip(Q, cc):
        shape = q[7] if len(q) > 7 else 'flat'
        P.q(q[0], q[1], c, q[3], q[4], q[5], shape)
    P.flush()


def _ln(M, s0, t0, s1, t1, K=4, r=None, j=0.):
    s = np.linspace(s0, s1, K); t = np.linspace(t0, t1, K)
    if r is not None and j > 0: s = s + r.normal(0, j, K)
    return M(np.stack([s, t], -1))


def _book_marks(Q, it, M, wpx, hpx, lab, L, r, lod):
    """the brush marks of one spine, from broad to fine, scaled to its size on screen"""
    flat_ = it.kind == 'flat'
    side = L.side
    col = it.col
    W_ = hpx if flat_ else wpx              # thickness on screen
    ln = (lambda s0, t0, s1, t1, K=4, j=0.: _ln(M, t0, s0, t1, s1, K, r, j)) if flat_ else (lambda s0, t0, s1, t1, K=4, j=0.: _ln(M, s0, t0, s1, t1, K, r, j))
    if flat_ and it.fore:                   # page block seen edge-on
        pg = rgb('#b9a98a') * r.uniform(.72, 1.0)
        Q.append((ln(.5, .015, .5, .985), W_ * .36, pg, 1., 0., lab, .95))
        if W_ > 5:
            for _ in range(int(min(7, W_ / 3))):
                Q.append((ln(r.uniform(.2, .8), .02, r.uniform(.2, .8), .98, 5), max(.35, W_ * .04), pg * r.uniform(.62, .95), .7, .3, lab, 1.))
        Q.append((ln(.93, .0, .93, 1.), max(.5, W_ * .09), col * 1.25, .95, .05, lab, 1.))
        return
    sh_s, hi_s = (.86, .1) if side < 0 else (.14, .9)
    if flat_: sh_s, hi_s = .14, .9          # lying: light from above
    # broad coat with pigment variation
    n = 1 if W_ < 5 else 2 if W_ < 14 else 4
    for _ in range(n):
        s = r.uniform(.25, .75); t0 = r.uniform(0, .3); t1 = r.uniform(.65, 1)
        Q.append((ln(s, t0, s + r.normal(0, .05), t1), W_ * r.uniform(.16, .3), np.clip(col * (1 + r.normal(0, .09)) + r.normal(0, .01, 3), 0, 1), .75, .22, lab, 1.))
    # turning into shadow, and the lit arris
    Q.append((ln(sh_s, -.02, sh_s, 1.02), W_ * .17, col * np.array([.62, .60, .70]), .9, .05, lab, 1.))
    if W_ > 3.2:
        Q.append((ln(hi_s, .03, hi_s + r.normal(0, .02), .97), max(.4, W_ * .075), np.clip(col * 1.34 + .03, 0, 1), .8, .25 + .3 * it.wear, lab, 1.))
    if W_ > 9:
        Q.append((ln(sh_s + (.1 if sh_s > .5 else -.1), 0, sh_s + (.1 if sh_s > .5 else -.1), 1), max(.4, W_ * .05), col * np.array([.42, .40, .52]), .85, .1, lab, 1.))
        Q.append((ln(.5 * (hi_s + .5), .05, .5 * (hi_s + .5), .95), W_ * .1, np.clip(col * 1.14 + .012, 0, 1), .55, .4, lab, 1.))
    # foot and head dirt
    if W_ > 4:
        Q.append((ln(0, .03, 1, .03, 3), max(.5, hpx * .018 if not flat_ else wpx * .012), col * .62, .7, .2, lab, 1.))
    cross = hpx if not flat_ else wpx
    for (t, th, sty) in it.bands:
        hw = max(.42, cross * th * .5)
        if sty == 0:
            Q.append((ln(.02, t, .98, t, 3), hw, np.clip(col * 1.28 + .02, 0, 1), .85, .12, lab, 1.))
            if hw > .8: Q.append((ln(.02, t - th * .95, .98, t - th * .95, 3), hw * .7, col * .56, .85, .1, lab, 1.))
        elif sty == 1:
            Q.append((ln(.05, t, .95, t, 3), hw * .6, mix(col, GILT, .62), .8, .5, lab, 1.))
        elif sty == 2:
            Q.append((ln(.0, t, 1., t, 3), hw * .8, col * .5, .85, .15, lab, 1.))
        else:
            Q.append((ln(.0, t, 1., t, 3), hw * 1.5, mix(col, it.col2, .75), .8, .2, lab, 1.))
    if it.label is not None and W_ > 3:
        t0, t1, sty = it.label
        c2 = mix(col, it.col2 if sty < 2 else rgb('#b9a98a') * .8, .78)
        tm = (t0 + t1) / 2
        Q.append((ln(.5, t0, .5, t1, 3), W_ * .32, c2, .92, 0., lab, 1.))
        if W_ > 7:
            mk = mix(c2, GILT, .6) if c2.mean() < .42 else c2 * .45
            nrow = 2 if W_ < 18 else 3 if W_ < 60 else 4
            for k in range(nrow):
                tt = t0 + (t1 - t0) * (k + .8) / (nrow + .6)
                s0 = r.uniform(.24, .34); s1 = r.uniform(.6, .78)
                if W_ < 24: Q.append((ln(s0, tt, s1, tt, 3), max(.4, cross * (t1 - t0) * .045), mk, .75, .45, lab, 1.))
                else:
                    x = s0
                    while x < s1:                     # worn strokes that never become letters
                        dx = r.uniform(.03, .09)
                        Q.append((_ln(M, x, tt + r.normal(0, .003), x + dx * .7, tt + r.normal(0, .006), 3, r, .004), max(.5, cross * .004), mk * r.uniform(.8, 1.1), .6 * r.uniform(.4, 1), .6, lab, 1.))
                        x += dx
    # scuffs
    if W_ > 6:
        for _ in range(int(r.integers(0, 2 + 4 * it.wear))):
            t = r.choice([r.uniform(.0, .12), r.uniform(.88, 1.), r.uniform(0, 1)]); s = r.uniform(.05, .95)
            Q.append((ln(s, t, np.clip(s + r.normal(0, .14), 0, 1), np.clip(t + r.normal(0, .035), 0, 1), 3), max(.4, W_ * r.uniform(.02, .06)), np.clip(col * 1.3 + .05, 0, 1), .5, .6, lab, 1.))
    if it.moss > .12:
        g = rgb('#5f6b3e')
        asp = (wpx / max(hpx, 1.)) if not flat_ else (hpx / max(wpx, 1.))
        nc = 1 + int(it.moss * 2.2)
        for _ in range(nc):
            cs = r.uniform(-.1, 1.1); ct = abs(r.normal(0, .09)) if not flat_ else r.uniform(0, 1)
            if flat_: cs = 1.02
            rad = r.uniform(.25, .75) * (1 + it.moss)
            m = int(8 + min(40, W_ * 1.1))
            for k in range(m):
                a_ = r.uniform(0, 2 * np.pi); q_ = rad * math.sqrt(r.random())
                s_ = cs + math.cos(a_) * q_; t_ = ct + math.sin(a_) * q_ * asp * 1.5
                if flat_: s_ = cs - abs(math.sin(a_)) * q_ * .6; t_ = ct + math.cos(a_) * q_ * asp * 2
                tone = .62 + .55 * (math.sin(a_) * .5 + .5) + r.normal(0, .12)
                p = ln(s_ - .03, t_, s_ + .03, t_ + .004, 3)
                Q.append((p, max(.45, W_ * .045 * r.uniform(.6, 1.4)), g * tone, .9, .1, lab, 1., 'round'))
    if W_ > 26: _hero_marks(Q, it, M, wpx, hpx, lab, L, r)


def _hero_marks(Q, it, M, wpx, hpx, lab, L, r):
    """close-up leather: mottling, cracks along the hinge, rubbed bands, tooling ghosts, frayed head"""
    if it.kind == 'flat': return
    col = it.col
    ln = lambda s0, t0, s1, t1, K=4, j=0.: _ln(M, s0, t0, s1, t1, K, r, j)
    n = int(min(90, wpx * hpx / 900))
    for _ in range(n):                                    # dabs of worn pigment, along the form
        s, t = r.uniform(0, 1), r.uniform(0, 1)
        v = 1 + r.normal(0, .10)
        Q.append((ln(s, t, s + r.normal(0, .05), t + r.uniform(.03, .10), 4), wpx * r.uniform(.03, .09), np.clip(col * v + r.normal(0, .008, 3), 0, 1), .5, .45, lab, 1.))
    sh_s = .9 if L.side < 0 else .1
    for k in range(int(r.integers(5, 11))):               # hinge cracks
        s = (0.05 if r.random() < .5 else .95) + r.normal(0, .03); t = r.uniform(.03, .9)
        Q.append((ln(s, t, s + r.normal(0, .025), t + r.uniform(.02, .12), 5, .006), max(.6, wpx * .010), col * .45, .8, .25, lab, 1., 'taper'))
    for k in range(int(r.integers(3, 8))):
        s = r.uniform(.15, .85); t = r.uniform(.03, .92)
        Q.append((ln(s, t, s + r.normal(0, .12), t + r.normal(0, .03), 5, .012), max(.5, wpx * .007), col * .5, .6, .3, lab, 1., 'taper'))
    bs = sorted(it.bands)
    for (t, th, sty) in bs:
        if sty == 0:                                      # raised band: rubbed top, crisp under-shadow
            for _ in range(4):
                s = r.uniform(.1, .7)
                Q.append((ln(s, t + th * .25, s + r.uniform(.1, .3), t + th * .25, 3), max(.6, hpx * th * .16), np.clip(col * 1.5 + .05, 0, 1), .7, .5, lab, 1.))
            Q.append((ln(0, t - th * .9, 1, t - th * .9, 4), max(.7, hpx * th * .22), col * np.array([.40, .38, .5]), .85, .05, lab, 1.))
            Q.append((ln(0, t - th * 1.5, 1, t - th * 1.5, 4), max(.6, hpx * .0022), mix(col, GILT, .5), .6, .6, lab, 1.))
            Q.append((ln(0, t + th * 1.3, 1, t + th * 1.3, 4), max(.6, hpx * .0022), mix(col, GILT, .5), .6, .6, lab, 1.))
        elif sty == 1:
            Q.append((ln(.03, t + th * .8, .97, t + th * .8, 4), max(.5, hpx * .0025), mix(col, GILT, .6), .65, .6, lab, 1.))
    ed = [0.02] + [b[0] for b in bs] + [.98]
    for a, b in zip(ed[:-1], ed[1:]):                     # worn ornament in the panels
        if b - a < .1 or r.random() < .45: continue
        tm = (a + b) / 2
        if it.label is not None and it.label[0] - .04 < tm < it.label[1] + .04: continue
        g = mix(col, GILT, .5)
        aw = wpx / hpx; mo = int(r.integers(0, 3)); q = r.uniform(.14, .24)
        if mo == 0:                                       # rubbed lozenge
            c4 = np.array([[.5, tm + q * aw * 1.3], [.5 + q, tm], [.5, tm - q * aw * 1.3], [.5 - q, tm], [.5, tm + q * aw * 1.3]])
            for p0, p1 in zip(c4[:-1], c4[1:]):
                if r.random() < .2: continue
                Q.append((M(np.stack([p0, (p0 + p1) / 2, p1])), max(.6, wpx * .009), g, .6 * r.uniform(.4, 1), .6, lab, 1., 'round'))
        elif mo == 1:                                     # a worn flower of dots
            for j in range(6):
                an = j * np.pi / 3 + .3; P0 = np.array([.5 + math.cos(an) * q * .7, tm + math.sin(an) * q * .7 * aw])
                Q.append((M(np.stack([P0 - [.02, 0], P0, P0 + [.02, 0]])), max(.7, wpx * .018), g, .6 * r.uniform(.3, 1), .4, lab, 1., 'round'))
        else:                                             # short rules
            for j in (-1, 0, 1):
                Q.append((ln(.5 - q * (1 - .4 * abs(j)), tm + j * .012, .5 + q * (1 - .4 * abs(j)), tm + j * .012, 3), max(.6, hpx * .0016), g, .6 * r.uniform(.4, 1), .6, lab, 1.))
    for _ in range(12):                                   # frayed head cap and foot
        s = r.uniform(0, 1); t = 1 - abs(r.normal(0, .018)) if r.random() < .6 else abs(r.normal(0, .015))
        Q.append((ln(s, t, s + r.normal(0, .04), t + r.normal(0, .012), 3), max(.6, wpx * .018), np.clip(col * 1.45 + .08, 0, 1), .6, .5, lab, 1.))


def paint_bay(P, T, u0, u1, v0, rows, L, seed, post=.07, board=.028, back='#241d30', wood='#54403c', sag=.006, depth=.28, style=None,
              book_kw=None, boards=True, posts=True, empty=.05, lod=1.0, boost=1.0):
    """one bookcase bay: back, posts, boards with sag, individually different rows.  rows = list of row heights (m), bottom up."""
    r = np.random.default_rng(seed)
    wood = rgb(wood) if isinstance(wood, str) else np.asarray(wood, f32)
    back = rgb(back) if isinstance(back, str) else np.asarray(back, f32)
    top = v0 + sum(rows)
    cen = T((u0 + u1) / 2, (v0 + top) / 2, 0.)
    P.poly(T(np.array([u0, u1, u1, u0]), np.array([v0, v0, top, top]), 0.), L(back, *cen)[0])
    style = style or {}
    v = v0
    w = u1 - u0 - post
    for k, rh in enumerate(rows):
        ch = rh - board
        sg = sag * r.uniform(.2, 1.7) * (r.uniform(4, 6) if r.random() < .05 else 1)
        kw = dict(fill=r.uniform(.86, .99), messy=r.uniform(.15, .7), unit=float(np.clip(ch / .3, .7, 1.5)) * r.uniform(.8, 1.15)); kw.update(book_kw or {})
        if r.random() < .4: kw['hue'] = [_names[r.choice(len(_names), p=_wts)] for _ in range(2)]
        if r.random() >= empty:
            it = make_row(int(r.integers(1 << 30)), w, ch, **kw)
            paint_row(P, T, it, L, v0=v + board, sag=sg, width=w, seed=seed * 7 + k, lod=lod, clear_h=ch, u0=u0 + post / 2, top_shadow=r.uniform(.1, .3))
        if boards:
            us = np.linspace(u0, u1, 9); q = (us - u0) / (u1 - u0); dv = -sg * 4 * q * (1 - q)
            a = T(us, v + board + dv, .012); b = T(us, v + dv, .012)
            P.strip(a, b, wood * r.uniform(.85, 1.12) * boost, L, n=2, val=.07, hi=.28, lo=-.4, seed=seed * 13 + k)
        v += rh
    if posts:
        for uu in (u0, u1):
            a = T(np.array([uu - post / 2] * 2), np.array([v0, top]), .03); b = T(np.array([uu + post / 2] * 2), np.array([v0, top]), .03)
            if L.side > 0: a, b = b, a
            P.strip(a, b, wood * r.uniform(.8, 1.05) * boost, L, n=3, val=.07, hi=.22, lo=-.38, seed=seed * 17 + int(uu * 100) % 97)
    P.flush()


# ------------------------------------------------------------------ inked cels (moving books)
INK = '#241a26'


def _sh(c): return np.clip(np.asarray(c) * np.array([.60, .56, .70]), 0, 1)


def book_cel(cel, T, it, L, seed=0, du=0., dv=0., dd=0., drot=0., bend=0., v0=0., u0=0., lw=2.0, ink=INK, side=True, top=True, flatL=None, d0=0., smear=0., pitch=0., rigid=False):
    """the same item as an inked cel: flat colour, one hard shadow, line.  Coordinates through T like paint_row.
    pitch: rigid tip toward the viewer (radians) about the front foot edge.  rigid: dead straight spine edges."""
    r = np.random.default_rng(it.seed)
    rb = np.random.default_rng(it.seed % 9973 + seed * 101)
    F = frame(it, du, dv, drot, bend)
    df = -it.set + dd + d0
    cp, sp = math.cos(pitch), math.sin(pitch)

    def M(st, q=0.):                                  # q = metres behind the spine, square to it
        st = np.asarray(st, float); uu, vv = F(st[..., 0], st[..., 1] * cp)
        return T(uu + u0, vv + v0 + q * sp, df + st[..., 1] * it.h * sp - q * cp)
    cen = M(np.array([.5, .5]))
    lit = flatL if flatL is not None else (lambda c, k=1.: L(np.clip(c, 0, 1), cen[0], cen[1], k)[0])
    wpx = float(np.hypot(*(M(np.array([1., .5])) - M(np.array([0., .5])))))
    jit = lambda p: p + rb.normal(0, .35, p.shape)
    base = lit(it.col); sh = lit(_sh(it.col)); hi = lit(np.clip(it.col * 1.3 + .02, 0, 1))
    t = np.linspace(0, 1, 7)
    # depth faces
    for s in (0., 1.):
        e = np.stack([np.full(7, s), t], -1)
        a, b = M(e), M(e, it.d)
        ax = M(np.array([1., .5])) - M(np.array([0., .5])); ax = ax / (np.hypot(*ax) + 1e-6)
        vis = float((b - a).mean(0) @ ax) * (1 if s > .5 else -1)
        if side and vis > 1.0:
            q = jit(np.vstack([a, b[::-1]]))
            cel.fill(q, lit(it.col * .5)); cel.line(b, lw * .7, ink, (.1, .1), seed=seed + 3)
            if vis > 42:                              # a cover seen from the side: hard shadow and a worn rule
                cel.shade(np.vstack([a * .35 + b * .65 + rb.normal(0, .6, a.shape), b[::-1]]), lit(it.col * np.array([.36, .34, .42])), within=q)
                fr = np.stack([a[0] * .82 + b[0] * .18, a[-1] * .82 + b[-1] * .18, a[-1] * .16 + b[-1] * .84, a[0] * .16 + b[0] * .84])
                fr = fr * .86 + fr.mean(0) * .14; cel.line(np.vstack([fr, fr[:1]]), max(.8, lw * .5), lit(mix(it.col * .5, GILT * .7, .5)), (.1, .1), seed=seed + 9, alpha=.7)
            cel.line(np.vstack([a[-1:], b[-1:]]), lw * .7, ink, (.05, .1), seed=seed + 4); cel.line(np.vstack([a[:1], b[:1]]), lw * .7, ink, (.05, .1), seed=seed + 5)
    if top:
        e = np.stack([np.linspace(.03, .97, 5), np.ones(5)], -1)
        a, b = M(e), M(e, it.d)
        ay = M(np.array([.5, 1.])) - M(np.array([.5, .0])); ay = ay / (np.hypot(*ay) + 1e-6)
        if float((b - a).mean(0) @ ay) > 1.0:
            q = np.vstack([a, b[::-1]])
            cel.fill(q, lit(rgb('#b9a98a') * .85))
            cel.shade(np.vstack([a * .5 + b * .5, b[::-1]]), lit(rgb('#b9a98a') * .55), within=q)
            cel.line(np.vstack([a[:1], b, a[-1:]]), lw * .7, ink, (.05, .05), seed=seed + 8)
    st = outline_st(it, np.random.default_rng(it.seed))
    if it.kind == 'up':
        left = np.stack([np.interp(t, [0, .33, .7, 1], [st[0, 0], st[1, 0], st[2, 0], st[3, 0]]), t * st[3, 1]], -1)
        right = np.stack([np.interp(t, [0, .33, .7, 1], [st[-1, 0], st[-2, 0], st[-3, 0], st[7, 0]]), t * st[7, 1]], -1)
        topl = st[3:8]
        if rigid:
            left = np.stack([st[0, 0] + (st[3, 0] - st[0, 0]) * t, t * st[3, 1]], -1); right = np.stack([st[-1, 0] + (st[7, 0] - st[-1, 0]) * t, t * st[7, 1]], -1)
    else:
        left = np.array([[0, 0], [0, 1.]]); right = np.array([[1, 0], [1, 1.]]); topl = np.array([[0, 1], [.5, 1.], [1, 1.]])
    ring = M(np.vstack([left, topl[1:-1], right[::-1]]))
    ring = jit(ring)
    cel.fill(ring, base)
    ss = 1 if L.side < 0 else 0
    edge = np.stack([np.full(7, .70 if ss else .30) + rb.normal(0, .012, 7), np.linspace(-.02, 1.05, 7)], -1)
    far = np.stack([np.full(7, 1.2 if ss else -.2), np.linspace(-.02, 1.05, 7)], -1)
    cel.shade(np.vstack([M(edge), M(far)[::-1]]), sh, within=ring)
    hp = M(np.stack([np.full(6, .1 if ss else .9), np.linspace(.06, .93, 6)], -1))
    for (tb, th, sty) in it.bands:
        c = {0: hi, 1: lit(mix(it.col, GILT, .55)), 2: lit(it.col * .5), 3: lit(mix(it.col, it.col2, .75))}[sty]
        hh = th * (.9 if sty in (0, 3) else .5)
        q = M(np.array([[-.05, tb - hh], [.5, tb - hh * .8], [1.05, tb - hh], [1.05, tb + hh], [.5, tb + hh * 1.1], [-.05, tb + hh]]))
        cel.shade(q, c, within=ring)
        if sty == 0:
            q2 = M(np.array([[.70 if ss else -.05, tb - hh], [1.05 if ss else .3, tb - hh], [1.05 if ss else .3, tb + hh], [.70 if ss else -.05, tb + hh]]))
            cel.shade(q2, base, within=ring)
            cel.line(M(np.array([[0.0, tb - hh], [.5, tb - hh], [1., tb - hh]])), lw * .55, ink, (.2, .2), seed=seed + int(tb * 100), alpha=.75)
            if wpx > 40: cel.line(M(np.array([[0.05, tb + hh], [.5, tb + hh], [.9, tb + hh]])), lw * .35, ink, (.4, .4), seed=seed + int(tb * 90), alpha=.45)
    if it.label is not None:
        t0, t1, sty = it.label
        c2 = it.col2 if sty < 2 else rgb('#b9a98a') * .8
        q = M(np.array([[.14, t0], [.86, t0], [.86, t1], [.14, t1]])); cel.shade(q, lit(c2), within=ring)
        mk = lit(mix(c2, GILT, .6) if c2.mean() < .42 else c2 * .45)
        if wpx > 12:
            nrow = 2 if wpx < 30 else 3
            for k in range(nrow):
                tt = t0 + (t1 - t0) * (k + .8) / (nrow + .6); x = .24 + .04 * r.random()
                while x < .74:
                    dx = r.uniform(.05, .14)
                    cel.line(M(np.array([[x, tt], [x + dx * .35, tt + r.normal(0, .003)], [x + dx * .7, tt]])), max(.7, lw * .45), mk, (.3, .3), seed=seed + k, alpha=.8)
                    x += dx
        cel.line(np.vstack([q, q[:1]]), lw * .4, ink, (.05, .05), seed=seed + 11, alpha=.5)
    if wpx > 10: cel.line(hp, min(4., max(.6, wpx * .03)), hi, (.3, .3), seed=seed + 12, alpha=.8 if wpx < 60 else .55)
    if it.moss > .12:
        rr = np.random.default_rng(it.seed + 5)
        for _ in range(int(1 + it.moss * 4)):
            c_ = np.array([rr.uniform(0, 1), abs(rr.normal(0, .12))]); n = 9; an = np.linspace(0, 2 * np.pi, n, endpoint=False)
            rad = .16 * rr.uniform(.6, 1.4, n)
            q = M(np.stack([c_[0] + np.cos(an) * rad, c_[1] + np.sin(an) * rad * wpx / max(1., np.hypot(*(M(np.array([.5, 1.])) - M(np.array([.5, 0.])))))], -1))
            cel.shade(spl(q, closed=True, n=4), lit(rgb('#5f6b3e')), within=ring)
    nl = len(left)
    cel.line(ring[:nl], lw, ink, (.08, .12), seed=seed + 1)
    cel.line(ring[nl - 1:nl + len(topl) - 1 + 1], lw * .9, ink, (.1, .1), seed=seed + 2)
    cel.line(ring[nl + len(topl) - 2:], lw * 1.1, ink, (.1, .08), seed=seed + 6)
    cel.line(np.vstack([ring[-1:], ring[:1]]), lw * .8, ink, (.15, .15), seed=seed + 7)
    return ring


def book_free(cel, cx, cy, w, h, rot, it, L, open_=0., bend=0., seed=0, lw=2.0, thick=None, ink=INK, sx=1.0, sy=1.0, lit=None):
    """loose closed book (cover view, spine on the left edge) centred at px (cx, cy); open_ 0..1 splays the covers.
    sx/sy stretch along/across its own axes (smear drawings)."""
    rb = np.random.default_rng(it.seed % 997 + seed * 37)
    th = thick if thick is not None else w * .24
    lit = lit or (lambda c, k=1.: L(np.clip(c, 0, 1), cx, cy, k)[0])
    c, s = math.cos(rot), math.sin(rot)

    def M(P):
        P = np.asarray(P, float) + rb.normal(0, .004, np.shape(P))
        x = P[..., 0] * w * sx; y = P[..., 1] * h * sy
        x = x + bend * w * np.sin(np.pi * np.clip(P[..., 1] + .5, 0, 1))
        return np.stack([cx + x * c - y * s, cy + x * s + y * c], -1)
    k = th / w
    o = open_
    pages = M([[-.47, -.48], [.46 + .1 * o, -.48 - .12 * o], [.46 + .1 * o + k * .6, -.46 - k * .25 - .12 * o], [.47 + k * .6 + .08 * o, .44 - k * .2], [.46, .48], [-.47, .48]])
    back = M([[-.5, -.5], [.5 + k * .75 + .12 * o, -.5 - k * .4 - .2 * o], [.5 + k * .75 + .16 * o, .5 - k * .4], [.5, .5]])
    cel.fill(back, lit(_sh(it.col) * .9)); cel.line(np.vstack([back, back[:1]]), lw * .8, ink, (.05, .05), seed=seed)
    cel.fill(pages, lit(rgb('#b9a68e'))); cel.shade(pages[[2, 3, 4]] * .55 + pages[[1, 4, 4]] * .45, lit(rgb('#8f8290')), within=pages); cel.line(pages[1:5], lw * .6, ink, (.1, .1), seed=seed + 1)
    for q in np.linspace(.2, .85, 4 + int(3 * o)):
        a = pages[1] * (1 - q) + pages[2] * q; b = pages[4] * (1 - q) + pages[3] * q
        cel.line(np.stack([a, (a + b) / 2 + rb.normal(0, 1, 2), b]), lw * .35, '#7a6b68', (.2, .3), seed=seed + int(q * 30), alpha=.7)
    cov = M([[-.5, -.5], [-.2, -.505], [.2, -.5 - .05 * o], [.5, -.5 - .1 * o], [.505, 0.], [.5, .5 + .02 * o], [.1, .505], [-.5, .5]])
    cel.fill(cov, lit(it.col))
    cel.shade(M([[-.6, .18], [0, .24], [.6, .16], [.6, .7], [-.6, .7]]), lit(_sh(it.col)), within=cov)
    cel.shade(M([[-.5, -.6], [-.36, -.6], [-.37, .6], [-.5, .6]]), lit(_sh(it.col)), within=cov)
    for (tb, t_, sty) in it.bands[:4]:
        cel.shade(M([[-.52, -.5 + tb], [-.36, -.5 + tb], [-.36, -.5 + tb + .03], [-.52, -.5 + tb + .03]]), lit(np.clip(it.col * 1.3, 0, 1)), within=cov)
    fr = M([[-.26, -.36], [.36, -.37], [.37, .36], [-.25, .37]])
    cel.line(np.vstack([fr, fr[:1]]), lw * .6, lit(mix(it.col, GILT, .55)), (.1, .1), seed=seed + 9)
    cel.line(np.vstack([cov, cov[:1]]), lw, ink, (.05, .08), seed=seed + 2)
    cel.line(M([[-.36, -.49], [-.365, 0], [-.36, .49]]), lw * .6, ink, (.1, .2), seed=seed + 3, alpha=.8)
    return cov
