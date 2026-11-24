"""s6 sets (flock lane): the gouache quad painter, the circular stairwell (6b/6c) and the hall of towering shelves (6a)."""
import numpy as np, cv2, math
from numba import njit
from lp import W, H
from lp.paint import rgb, mix, f32, fbm, _strokes, _NOLAB, _NOA, smooth, jit_color
from lp.pages import fill_polys, hash01
from lp.cam import Camera3D

ALB = {  # book cloth and leather, faded
    'oxblood': '#7d4345', 'moss': '#5c6650', 'indigo': '#434c78', 'ochre': '#9b7a48', 'plum': '#65506e', 'greygreen': '#6f7c78',
    'bone': '#c6b79b', 'rust': '#8f5a42', 'slate': '#5a6484', 'umber': '#6b5245', 'rose': '#9a6f70', 'teal': '#4f6f74'}
ALB_P = np.array([.15, .12, .13, .10, .11, .10, .08, .06, .06, .05, .02, .02])
ALBA = np.stack([rgb(v) for v in ALB.values()])


# ------------------------------------------------------------------ label raster
@njit(cache=True, fastmath=True)
def _fill_lab(lab, V, N, ID):
    h, w = lab.shape
    xs = np.zeros(16, np.float32)
    for i in range(V.shape[0]):
        nv = N[i]; y0 = 1e9; y1 = -1e9; ok = True
        for k in range(nv):
            py = V[i, k, 1]
            if not (py > -1e6 and py < 1e6 and V[i, k, 0] > -1e6 and V[i, k, 0] < 1e6): ok = False
            if py < y0: y0 = py
            if py > y1: y1 = py
        if not ok: continue
        ya = max(0, int(math.ceil(y0 - .5))); yb = min(h - 1, int(math.floor(y1 - .5)))
        for y in range(ya, yb + 1):
            sy = y + .5; cnt = 0
            for k in range(nv):
                k2 = k + 1 if k + 1 < nv else 0
                ay = V[i, k, 1]; by = V[i, k2, 1]
                if (ay <= sy and by > sy) or (by <= sy and ay > sy):
                    if cnt < 16:
                        xs[cnt] = V[i, k, 0] + (sy - ay) / (by - ay) * (V[i, k2, 0] - V[i, k, 0]); cnt += 1
            for p in range(1, cnt):
                v = xs[p]; q = p - 1
                while q >= 0 and xs[q] > v:
                    xs[q + 1] = xs[q]; q -= 1
                xs[q + 1] = v
            for p in range(0, cnt - 1, 2):
                a = max(0, int(math.ceil(xs[p] - .5))); b = min(w - 1, int(math.floor(xs[p + 1] - .5)))
                for x in range(a, b + 1): lab[y, x] = ID[i]


def strokes_into(img, P, wd, col, op=1.0, dry=0.0, seed=0, shape=0, lab=None, LAB=None, bristle=.05):
    """lp.paint brush marks straight into a float32 frame (no copy)"""
    P = np.ascontiguousarray(P, f32); n, K, _ = P.shape
    if n == 0: return img
    g = lambda v, s=(n,): np.broadcast_to(np.asarray(v, f32), s).copy()
    sd = (np.arange(n) * 7919 + seed * 104729).astype(np.int64) & 0x7ffffff
    _strokes(img, _NOA, P, g(wd), g(col, (n, 3)), g(op), g(dry), sd, _NOLAB if lab is None else lab,
             np.full(n, -1, np.int32) if LAB is None else np.asarray(LAB, np.int32), False, shape, K, bristle)
    return img


class Quads:
    """painter-ordered quads: world corners (4,3), albedo, unit normal, kind, stroke count along the short side"""
    def __init__(s): s.P, s.A, s.N, s.K, s.S, s.M = [], [], [], [], [], []

    def add(s, P, alb, nrm, kind=0, strokes=2, mul=1.0):
        P = np.asarray(P, f32)
        if P.ndim == 2: P = P[None]
        n = len(P)
        s.P.append(P); s.A.append(np.broadcast_to(np.asarray(alb, f32), (n, 3))); s.N.append(np.broadcast_to(np.asarray(nrm, f32), (n, 3)))
        s.K.append(np.full(n, kind)); s.S.append(np.full(n, strokes)); s.M.append(np.broadcast_to(np.asarray(mul, f32), (n,)))

    def done(s):
        return [np.concatenate(v) for v in (s.P, s.A, s.N, s.K, s.S, s.M)]


def paint_quads(cam, w, h, ss, quads, shade, variants, bg, seed=1, jitter=.5, stroke_min=5.0, bristle=.06, mask_kinds=None, extra=None):
    """returns {variant: (h,w,3)} painted at ss and averaged down.  shade(variant, centre (n,3), albedo, normal, kind, mul) -> (n,3)"""
    P, A, N, K, S, M = quads
    cs = Camera3D(cam.eye, cam.eye + cam.R[2], cam.f * ss, int(w * ss), int(h * ss), cam.cx * ss, cam.cy * ss)
    cs.R = cam.R
    v, z = cs.project(P.astype(float))
    ok = (z > .25).all(1)
    x0, x1 = v[..., 0].min(1), v[..., 0].max(1); y0, y1 = v[..., 1].min(1), v[..., 1].max(1)
    ok &= (x1 > -20) & (x0 < cs.w + 20) & (y1 > -20) & (y0 < cs.h + 20)
    v = v[ok].astype(f32); P, A, N, K, S, M = P[ok], A[ok], N[ok], K[ok], S[ok], M[ok]
    r = np.random.default_rng(seed)
    n = len(v)
    v += r.normal(0, jitter * ss * .5, v.shape).astype(f32)
    ids = np.arange(1, n + 1, dtype=np.int32)
    lab = np.zeros((cs.h, cs.w), np.int32)
    _fill_lab(lab, v, np.full(n, 4, np.int32), ids)
    vis = np.bincount(lab.ravel(), minlength=n + 1)[1:] > 6
    C0 = P.mean(1)
    # strokes: along the long side
    e01 = np.linalg.norm(v[:, 1] - v[:, 0], axis=1) + np.linalg.norm(v[:, 2] - v[:, 3], axis=1)
    e12 = np.linalg.norm(v[:, 2] - v[:, 1], axis=1) + np.linalg.norm(v[:, 3] - v[:, 0], axis=1)
    e01 *= .5; e12 *= .5
    q = v.copy()
    swap = e01 > e12                       # make axis a (0->3 , 1->2) the long one
    q[swap] = np.roll(v[swap], -1, axis=1)
    long_, short = np.maximum(e01, e12), np.minimum(e01, e12)
    sp, sw, sl, sj = [], [], [], []
    for i in np.nonzero(vis & (short > stroke_min) & (S > 0))[0]:
        m = int(np.clip(round(S[i] * (1 + short[i] / 60.)), 1, 7)); k = int(np.clip(round(long_[i] / (short[i] * 4.5 + 40)), 1, 6))
        u = (np.arange(m) + .5) / m + r.normal(0, .16 / m, m)
        for uu in np.clip(u, .03, .97):
            for kk in range(k):
                t0 = max(-.03, (kk - .25) / k + r.normal(0, .05)); t1 = min(1.03, (kk + 1.25) / k + r.normal(0, .05))
                t = np.linspace(t0, t1, 4)
                du = r.normal(0, .07 / m)
                a = q[i, 0] + (q[i, 3] - q[i, 0]) * t[:, None]; b = q[i, 1] + (q[i, 2] - q[i, 1]) * t[:, None]
                pl = a + (b - a) * (uu + du * np.sin(t * 3 + kk))[:, None]
                if r.random() < .5: pl = pl[::-1]
                sp.append(pl); sw.append(short[i] / m * r.uniform(.45, .78)); sl.append(i); sj.append(uu)
    n0 = len(sp)
    # scumbles: drier, shorter, off-value marks dragged along the form of the larger pieces
    for i in np.nonzero(vis & (short > 22 * ss) & (S > 0))[0]:
        for kk in range(int(np.clip(long_[i] * short[i] / (5200 * ss * ss), 1, 9))):
            u0 = r.uniform(.05, .95); t0 = r.uniform(0, .8); t = np.linspace(t0, min(1.02, t0 + r.uniform(.15, .5)), 4)
            a = q[i, 0] + (q[i, 3] - q[i, 0]) * t[:, None]; b = q[i, 1] + (q[i, 2] - q[i, 1]) * t[:, None]
            sp.append(a + (b - a) * (u0 + r.normal(0, .02) * np.sin(t * 5))[:, None]); sw.append(min(short[i] * r.uniform(.06, .20), 16 * ss)); sl.append(i); sj.append(0)
    sp = np.array(sp, f32).reshape(-1, 4, 2); sw = np.array(sw, f32); sl = np.array(sl, int)
    ns = len(sp)
    isc = np.arange(ns) >= n0
    jv = np.where(isc, r.normal(0, .12, ns), r.normal(0, .055, ns))[:, None].astype(f32); jh = r.normal(0, .016, (ns, 1)).astype(f32) * np.array([[1.1, .1, -1.1]], f32)
    op = np.where(isc, r.uniform(.25, .6, ns), r.uniform(.45, .95, ns)); dry = np.where(isc, r.uniform(.25, .75, ns), np.clip(r.normal(.14, .22, ns), 0, .7))
    out = {}
    for nm in variants:
        col = np.clip(shade(nm, C0, A, N, K, M), 0, 1.4).astype(f32)
        img = np.empty((cs.h, cs.w, 3), f32); img[:] = bg[nm]
        fill_polys(img, v, 4, col)
        if ns:
            sc = np.clip(col[sl] * (1 + jv) + jh * (.3 + col[sl].mean(1, keepdims=True)), 0, 1.4)
            strokes_into(img, sp, sw, sc, op, dry, seed + 3, 0, lab, ids[sl], bristle)
        if extra is not None: extra(img, cs, nm, ss)
        out[nm] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        del img
    if mask_kinds is not None:
        kk = np.r_[False, np.isin(K, mask_kinds)]
        out['mask'] = cv2.resize(kk[lab].astype(f32), (w, h), interpolation=cv2.INTER_AREA)
    return out


# ================================================================== the stairwell
R0 = 5.2; RIN = 4.12; PITCH = 8.6; STEPS = 40; RISE = PITCH / STEPS; DTH = 2 * math.pi / STEPS; TH0 = 2.15
ZD = 22.0; DOME_H = 3.3; HAND = 1.0
SUN = np.array([.36, -.24, .90]); SUN = SUN / np.linalg.norm(SUN)
SUNB = np.array([.17, -.11, .95]); SUNB = SUNB / np.linalg.norm(SUNB)   # the light the flock catches: steeper
RO = 4.75                    # radius of the dome opening
NP_ = 12                     # dome panels
FLOORS = np.arange(0, 26, 4.4)
FOG_N, FOG_S, FOG_G = rgb('#5b5a8c'), rgb('#b69cb0'), rgb('#ffe1a0')


def zstair(th, n=0):
    """height of the stair (tread top) at angle th, turn n"""
    return (np.mod(HAND * (th - TH0), 2 * math.pi)) / (2 * math.pi) * PITCH + n * PITCH - 3.0


def cyl(th, z, w=0.0):
    th, z, w = np.broadcast_arrays(th, z, w)
    return np.stack([(R0 - w) * np.cos(th), (R0 - w) * np.sin(th), z], -1)


def cq(t0, t1, z0, z1, w0=0.0, w1=None):
    """quads on the cylinder: corners (t0,z0) (t1,z0) (t1,z1) (t0,z1)"""
    w1 = w0 if w1 is None else w1
    return np.stack([cyl(t0, z0, w0), cyl(t1, z0, w0), cyl(t1, z1, w1), cyl(t0, z1, w1)], -2)


def cn(th):
    th = np.asarray(th, float); return np.stack([-np.cos(th), -np.sin(th), np.zeros_like(th)], -1)


def sun_hit(P, sun=SUN):
    """where the ray to the sun meets the dome opening plane: radius, sector, distance to nearest rib, path length"""
    P = np.asarray(P, float)
    L = (ZD + 1.0 - P[..., 2]) / sun[2]
    X = P[..., 0] + sun[0] * L; Y = P[..., 1] + sun[1] * L
    rr = np.hypot(X, Y); a = np.mod(np.arctan2(Y, X), 2 * math.pi) / (2 * math.pi) * NP_
    k = np.floor(a).astype(int) % NP_
    rib = np.minimum(a - np.floor(a), np.ceil(a) - a) * 2 * math.pi / NP_ * rr
    return rr, k, rib, L


def stair_shadow(P, hmax=3.2, n=26):
    """1 where a stair turn blocks the sun above point P"""
    P = np.asarray(P, float); out = np.zeros(P.shape[:-1], bool); prev = None
    for h in np.linspace(.04, hmax, n):
        Q = P + SUN * (h / SUN[2])
        rr = np.hypot(Q[..., 0], Q[..., 1]); th = np.arctan2(Q[..., 1], Q[..., 0])
        d = np.mod(Q[..., 2] - zstair(th) + PITCH * .5, PITCH) - PITCH * .5
        if prev is not None: out |= (prev < 0) & (d >= 0) & (np.abs(d - prev) < 1) & (rr > RIN) & (rr < R0 + .1)
        prev = d
    return out


MOON = np.array([-.62, .30, .72]); MOON = MOON / np.linalg.norm(MOON)


def moon_band(th, z):
    """a slanting bar of moonlight that falls through a gap in the dome (hard painted edge)"""
    d = np.mod(th - (-.30 + .050 * (z - 12)) + math.pi, 2 * math.pi) - math.pi
    w = .15 + .012 * (z - 4)
    m = np.clip((w - np.abs(d)) / .03 + .5, 0, 1) * smooth(5.2, 6.4, z) * (1 - smooth(20.5, 22, z))
    d2 = d - w - .17
    m2 = np.clip((.07 - np.abs(d2)) / .03 + .5, 0, 1) * smooth(8, 9, z) * (1 - smooth(20.5, 22, z))
    return np.maximum(m, m2 * .75)


def shade_well(nm, C, A, N, K, M):
    z = C[:, 2]; th = np.arctan2(C[:, 1], C[:, 0])
    d = np.linalg.norm(C - np.array([.8, -.6, 0]), axis=1)
    ao = np.where(K == 3, .42, 1.0) * M                       # K: 0 book, 1 wood, 2 stone, 3 recess, 4 iron
    g = A.mean(1, keepdims=True)
    if nm == 'N':
        A = g + (A - g) * .80
        amb = mix(rgb('#36325e'), rgb('#7675b2'), smooth(3, 18, z)[:, None])
        side = .84 + .34 * np.clip(N @ np.array([-.75, .35, .2]), -1, 1)
        mb = moon_band(th, z) * np.where(K == 3, .35, 1)
        warm = np.exp(-np.maximum(z - .5, 0) / 3.6) * .62 * (.45 + .55 * np.clip(N @ np.array([.0, .3, -.95]), 0, 1))
        c = A * ao[:, None] * (amb * side[:, None] + warm[:, None] * rgb('#ffb458') * .9)
        c = c + A * ao[:, None] * mb[:, None] * rgb('#a3b6e2') * .70 + mb[:, None] * rgb('#6f7fc0') * .05
        c = c + rgb('#1c1a30') * .36
        # near the lens the shelves go soft and quiet: big dark masses, little contrast between spines
        e = ((1 - smooth(4.2, 9.2, z)) * np.where(K == 2, .30, .74))[:, None]
        c = c + (rgb('#2b2846') * (.80 + .20 * ao[:, None]) + c.mean(1, keepdims=True) * .10 - c) * e
        k = (1 - np.exp(-np.maximum(0, d - 6) * .060))[:, None]
        return c * (1 - k) + (FOG_N * .96) * k
    if nm == 'S':
        A = g + (A - g) * .72
        amb = mix(rgb('#6f68ac'), rgb('#c9a9bb'), smooth(4, 22, z)[:, None])
        bounce = np.clip(-(N @ SUN) * .45 + .18, 0, 1)[:, None] * smooth(0, 18, z)[:, None]
        c = A * ao[:, None] * (amb * .98 + bounce * rgb('#ffcf93') * .55) + rgb('#2e2a58') * .22
        e = ((1 - smooth(4.2, 9.2, z)) * np.where(K == 2, .24, .62))[:, None]
        c = c + (rgb('#4b4478') * (.78 + .22 * ao[:, None]) + c.mean(1, keepdims=True) * .12 - c) * e
        k = (1 - np.exp(-np.maximum(0, d - 4.5) * .060))[:, None]
        return c * (1 - k) + mix(rgb('#9788b4'), rgb('#e0c0b4'), smooth(8, 21, z)[:, None]) * k
    A = g + (A - g) * .62
    A = A * np.array([1.04, .97, .90])
    sunl = np.clip(N @ SUN, .0, 1) * .85 + .62
    c = A * (.40 + .60 * ao[:, None]) * (sunl[:, None] * rgb('#ffcb80') * 1.60) + rgb('#e08058') * .13
    c = np.where((K == 3)[:, None], c * np.array([1.10, .86, .92]) + rgb('#a05548') * .10, c)
    k = (1 - np.exp(-np.maximum(0, d - 4.5) * .058))[:, None]
    c = c * (1 - k) + FOG_G * 1.02 * k
    e = ((1 - smooth(4.2, 8.6, z)) * np.where(K == 2, .12, .34))[:, None]
    c = c + (rgb('#d9a468') * (.72 + .28 * ao[:, None]) - c) * e
    return np.minimum(c, 1.0 + (c - 1.0) * .35)


def books_row(r, u0, u1, zlo, clear, wmin=.028, wmax=.072, gap=.05, mood=None):
    """generate one shelf of books: arrays u0,u1,height,tilt,albedo,depth"""
    u = u0 + r.uniform(0, .04); B = []
    ci = int(r.choice(len(ALBA), p=ALB_P / ALB_P.sum())); run = int(r.integers(2, 8)); hb = r.uniform(.72, .95); wb = r.uniform(wmin, wmax)
    tilt = 0.0
    if mood == 'empty': return np.zeros((0, 6))
    stop = u0 + (u1 - u0) * (r.uniform(.25, .7) if mood == 'half' else 1.0)
    if mood == 'half' and r.random() < .5: u += (u1 - u0) * r.uniform(.2, .5); stop = u1
    if mood == 'lean': gap = .22
    while u < min(u1, stop) - wmin:
        if run <= 0:
            ci = int(r.choice(len(ALBA), p=ALB_P / ALB_P.sum())); run = int(r.integers(1, 8)); hb = r.uniform(.60, .96); wb = r.uniform(wmin, wmax)
            tilt = r.normal(0, .16) if r.random() < .14 else 0.0
            if mood == 'lean': tilt = r.choice([-1, 1]) * r.uniform(.14, .42) if r.random() < .75 else 0.0
            if r.random() < gap: u += r.uniform(.04, .16)
        w = wb * r.uniform(.85, 1.15); h = clear * hb * r.uniform(.96, 1.02)
        if u + w > u1: break
        B.append((u, u + w * .95, h, tilt + r.normal(0, .012), ci, r.uniform(.0, .05))); u += w; run -= 1
    return np.array(B) if B else np.zeros((0, 6))


def build_well(seed=11, zlo=1.2, detail=True):
    """all quads of the shelf wall (stairs, dome and rails are drawn live)"""
    r = np.random.default_rng(seed); Q = Quads()
    wood = rgb('#7d5a48'); wood_d = rgb('#54403f'); stone = rgb('#9a93a4')
    NB = 8; pil = .42                          # pilasters
    bay = 2 * math.pi / NB
    rows = np.arange(zlo, ZD, .40)[::-1]
    # back wall
    for j, z in enumerate(rows):
        fl = np.floor((z + .2) / 4.4); band = (z - fl * 4.4) < .55 or (fl + 1) * 4.4 - z < .42
        for b in range(NB):
            t0, t1 = b * bay + pil / 2 / R0, (b + 1) * bay - pil / 2 / R0
            seg = np.linspace(t0, t1, 12)
            if band: continue
            Q.add(cq(seg[:-1], seg[1:], z, z + .40, -.22), wood_d, cn((seg[:-1] + seg[1:]) / 2), 3, 1, np.repeat(r.uniform(.62, .86, 4), 3)[:11])
            # underside of the board above
            Q.add(np.stack([cyl(seg[:-1], z + .365, .03), cyl(seg[1:], z + .365, .03), cyl(seg[1:], z + .365, -.22), cyl(seg[:-1], z + .365, -.22)], -2),
                  wood_d, np.array([0, 0, -1.]), 3, 1, np.repeat(r.uniform(.55, .75, 4), 3)[:11])
            for c in range(2):
                a0, a1 = t0 + (t1 - t0) * c / 2 + .03 / R0, t0 + (t1 - t0) * (c + 1) / 2 - .03 / R0
                far = z > 13
                rm = np.random.default_rng(int(b * 7919 + c * 331 + j * 17 + seed)); u_ = rm.random()
                # some half bays hold tall folios over two shelves, some are cupboards, some have been emptied
                if u_ < .07 and not far:
                    for q_ in range(3):
                        w0, w1 = a0 + (a1 - a0) * q_ / 3 + .01 / R0, a0 + (a1 - a0) * (q_ + 1) / 3 - .01 / R0
                        Q.add(cq(np.array([w0]), np.array([w1]), z + .02, z + .34, .015), wood * rm.uniform(.85, 1.0), cn((w0 + w1) / 2), 1, 2, rm.uniform(.62, .80))
                        Q.add(cq(np.array([w0 + .05 / R0]), np.array([w1 - .05 / R0]), z + .07, z + .29, .03), wood * 1.08, cn((w0 + w1) / 2), 1, 1, rm.uniform(.74, .92))
                        Q.add(cq(np.array([(w0 + w1) / 2 - .012 / R0]), np.array([(w0 + w1) / 2 + .012 / R0]), z + .17, z + .20, .05), rgb('#c9a35a'), cn((w0 + w1) / 2), 4, 0, .8)
                    continue
                mood = 'empty' if u_ < .13 else 'half' if u_ < .30 else 'lean' if u_ < .50 else None
                if u_ > .86 and not far:      # a pile lying flat
                    zc = z; w_ = (a1 - a0) * rm.uniform(.22, .34); x_ = a0 + (a1 - a0 - w_) * rm.random()
                    for q_ in range(int(rm.integers(2, 5))):
                        t_ = rm.uniform(.045, .075); ww = w_ * rm.uniform(.8, 1.0); xo = x_ + (w_ - ww) * rm.random()
                        alb_ = np.clip(ALBA[int(rm.integers(0, 12))] * rm.uniform(.85, 1.1), 0, 1)
                        Q.add(cq(np.array([xo]), np.array([xo + ww]), zc, zc + t_, -.01), alb_, cn(xo), 0, 1, rm.uniform(.85, 1.05)); Q.add(cq(np.array([xo + ww * .86]), np.array([xo + ww]), zc + .005, zc + t_ - .008, -.005), rgb('#c9b894'), cn(xo), 0, 0, .8); zc += t_
                    if x_ - a0 > (a1 - a0) * .3: a1 = x_ - .02 / R0
                    else: a0 = x_ + w_ + .02 / R0
                    if a1 - a0 < .10: continue
                B = books_row(r, a0 * R0, a1 * R0, z, .335, *((.045, .10) if far else (.030, .076)), mood=mood)
                if not len(B): continue
                th0, th1, hh, tl, ci, dp = B.T; th0 = th0 / R0; th1 = th1 / R0
                P = np.stack([cyl(th0, z, -dp), cyl(th1, z, -dp), cyl(th1 + tl * hh / R0, z + hh, -dp), cyl(th0 + tl * hh / R0, z + hh, -dp)], -2)
                alb = jit_color(ALBA[ci.astype(int)], r, .014, .03, .10)
                tm = (th0 + th1) / 2
                Q.add(P, alb, cn(tm), 0, 1, r.uniform(.86, 1.12, len(P)))
                if z < 15:
                    def sub(x0, x1, y0, y1, a, m, st=1, sel=slice(None)):
                        p = P[sel]; l0, l1 = p[:, 0] + (p[:, 1] - p[:, 0]) * x0, p[:, 0] + (p[:, 1] - p[:, 0]) * x1
                        h0, h1 = p[:, 3] + (p[:, 2] - p[:, 3]) * x0, p[:, 3] + (p[:, 2] - p[:, 3]) * x1
                        q = np.stack([l0 + (h0 - l0) * y0, l1 + (h1 - l1) * y0, l1 + (h1 - l1) * y1, l0 + (h0 - l0) * y1], 1)
                        Q.add(q, a[sel] if np.ndim(a) == 2 else a, cn(tm[sel]), 0, st, m)
                    sub(.72, 1.0, 0, 1, alb, .66, 1)                           # hard turned-away side
                    sub(.0, .16, .02, .98, np.clip(alb * 1.18 + .03, 0, 1), 1.0, 0)
                    if 6.0 < z < 9.5:
                        s = np.nonzero(r.random(len(P)) < .42)[0]
                        if len(s):
                            y = r.uniform(.60, .80, 1)[0]
                            sub(.04, .96, y, y + .05, np.clip(alb * .5 + rgb('#c9a35a') * .55, 0, 1), .95, 0, s); sub(.04, .96, y - .12, y - .085, np.clip(alb * .5 + rgb('#c9a35a') * .55, 0, 1), .9, 0, s)
                        s = np.nonzero(r.random(len(P)) < .24)[0]
                        if len(s): sub(.10, .86, .34, .52, rgb('#c9b894'), .9, 0, s)
                    sub(-.02, 1.02, .80, 1.02, alb, .50, 1, np.nonzero(r.random(len(P)) < .9)[0])  # shadow of the shelf above
            # board front
            Q.add(cq(seg[:-1], seg[1:], z - .035, z + .0, .03), wood, cn((seg[:-1] + seg[1:]) / 2), 1, 1, np.repeat(r.uniform(.85, 1.05, 4), 3)[:11])
        if not band:
            for b in range(NB):
                tm = (b + .5) * bay
                Q.add(cq(np.array([tm - .035 / R0]), np.array([tm + .035 / R0]), z - .03, z + .40, .035), wood, cn(tm), 1, 1, r.uniform(.8, 1))
        # pilasters, drawn as stone courses
        for b in range(NB):
            t = b * bay; m = r.uniform(.82, 1.08); dz = r.uniform(-.02, .02)
            if not band or True:
                Q.add(cq(np.array([t - pil / 2 / R0]), np.array([t - pil * .18 / R0]), z - .02, z + .42, -.02, -.02), stone, cn(t - .05), 2, 1, m * .74)
                Q.add(cq(np.array([t - pil * .22 / R0]), np.array([t + pil * .22 / R0]), z + dz, z + .41 + dz, .10), stone * r.uniform(.94, 1.05), cn(t), 2, 2, m)
                Q.add(cq(np.array([t + pil * .18 / R0]), np.array([t + pil / 2 / R0]), z - .02, z + .42, -.02, -.02), stone, cn(t + .05), 2, 1, m * .60)
        # storey bands: cornice, frieze, shadow
        for fz in FLOORS:
            if abs(fz - z) < .2001 and z > 2:
                def ring(n0, z0, z1, w0, w1, lo, hi, st=2, tint=1.0, nrm=None):
                    e = np.cumsum(np.clip(r.normal(1, .28, n0), .45, 1.8)); e = np.r_[0, e] / e[-1] * 2 * math.pi + r.uniform(0, 6)
                    sub_ = 3; a_ = np.concatenate([np.linspace(e[i], e[i + 1], sub_ + 1)[:-1] for i in range(n0)]); b__ = np.r_[a_[1:], a_[0] + 2 * math.pi]
                    m_ = np.repeat(r.uniform(lo, hi, n0), sub_); tt_ = np.repeat(1 + r.normal(0, .02, (n0, 1)) * np.array([[1, 0, -1]]), sub_, 0)
                    if nrm is None: P_ = cq(a_, b__, z0, z1, w0, w1); n_ = cn((a_ + b__) / 2)
                    else: P_ = np.stack([cyl(a_, z0, w0), cyl(b__, z0, w0), cyl(b__, z1, w1), cyl(a_, z1, w1)], -2); n_ = nrm
                    Q.add(P_, np.clip(stone * tint * tt_, 0, 1), n_, 2, st, m_)
                ring(64, fz + .30, fz + .55, .04, .04, .70, .84, 1)
                ring(52, fz - .02, fz - .02, .36, .0, .42, .52, 2, 1.0, np.array([0, 0, -1.]))
                ring(46, fz - .02, fz + .20, .36, .36, .90, 1.06, 2)
                ring(58, fz + .18, fz + .30, .40, .40, 1.00, 1.12, 1, 1.06)
                ring(70, fz - .30, fz - .02, .10, .12, .58, .70, 1)
                ring(60, fz - .42, fz - .30, .03, .03, .46, .56, 1)
                nd = 150; cb = np.linspace(0, 2 * math.pi, nd + 1)[:-1] + r.normal(0, .004, nd); hw = .055 / R0
                Q.add(np.stack([cyl(cb - hw, fz - .15, .12), cyl(cb + hw, fz - .15, .12), cyl(cb + hw, fz - .02, .27), cyl(cb - hw, fz - .02, .27)], -2), stone, cn(cb), 2, 0, r.uniform(.64, .80, nd))
    for (t_, z0, z1, lean) in ((-.62, 4.3, 8.5, .05), (.95, 8.9, 13.0, -.05), (2.75, 4.6, 8.6, .04)):        # ladders left against the shelves
        n_ = 10; zz = np.linspace(z0, z1, n_ + 1); tt = t_ + lean * (zz - z0) / (z1 - z0); ww = .20 / R0; wo = .46 - .28 * (zz - z0) / (z1 - z0)
        for sd_ in (-1, 1):
            Q.add(np.stack([cyl(tt[:-1] + sd_ * ww - .022 / R0, zz[:-1], wo[:-1]), cyl(tt[:-1] + sd_ * ww + .022 / R0, zz[:-1], wo[:-1]), cyl(tt[1:] + sd_ * ww + .022 / R0, zz[1:], wo[1:]), cyl(tt[1:] + sd_ * ww - .022 / R0, zz[1:], wo[1:])], -2), rgb('#a87c58'), cn(t_), 1, 1, r.uniform(.86, 1.05, n_))
        rg = np.arange(z0 + .2, z1 - .1, .30); tr = np.interp(rg, zz, tt); wr = np.interp(rg, zz, wo)
        Q.add(np.stack([cyl(tr - ww, rg, wr), cyl(tr + ww, rg + r.normal(0, .008, len(rg)), wr), cyl(tr + ww, rg + .045, wr), cyl(tr - ww, rg + .045, wr)], -2), rgb('#9a7050'), cn(t_), 1, 1, r.uniform(.80, 1.0, len(rg)))
    return Q.done()


def ivy_well(seed=41):
    """hanging ivy: world polylines of stems and leaf positions"""
    r = np.random.default_rng(seed); stems, leaves = [], []
    for (t0, spread, z0, n) in ((-.95, .20, 8.78, 9), (.42, .16, 13.1, 7), (1.75, .22, 8.75, 8), (2.45, .16, 13.1, 7), (3.75, .24, 8.8, 11), (4.75, .18, 13.15, 8), (-1.75, .18, 13.1, 7), (5.6, .14, 17.5, 6), (2.0, .16, 17.5, 6)):
        for k in range(n):
            th = t0 + r.normal(0, spread); L = r.uniform(.5, 3.0) * (1.15 if abs(th - t0) < spread * .6 else .6)
            m = 12; zz = z0 - np.linspace(0, L, m) ; sw = np.cumsum(r.normal(0, .018, m)) / R0 * 3; wo = .40 - .10 * np.linspace(0, 1, m) + r.normal(0, .02)
            P = cyl(th + sw, zz, wo); stems.append(P)
            for q in range(int(L * 15) + 5):
                u = r.random() ** .85 * (m - 1); i = int(u); p = P[i] + (P[min(i + 1, m - 1)] - P[i]) * (u - i)
                leaves.append(np.r_[p + r.normal(0, .075, 3), r.uniform(.055, .125) * (1 - .45 * u / m), r.uniform(0, 6.28), r.random()])
        for q in range(46):                       # the clump it hangs from
            leaves.append(np.r_[cyl(t0 + r.normal(0, spread * 1.15), z0 + abs(r.normal(0, .13)), .42 + r.normal(0, .05)), r.uniform(.07, .14), r.uniform(0, 6.28), r.random()])
    return stems, np.array(leaves)


def paint_ivy(img, cs, nm, ss):
    stems, LV = ivy_well()
    pal = {'N': ('#2c3550', '#3f5468', '#6f8ca6'), 'S': ('#565a82', '#74809a', '#a9a4b4'), 'G': ('#8a7638', '#c4a24e', '#f3dc8e')}[nm]
    pal = [rgb(q) for q in pal]
    fog = {'N': FOG_N * .96, 'S': rgb('#c4a6b4'), 'G': FOG_G}[nm]
    for P in stems:
        v, z = cs.project(P); d = np.linalg.norm(P.mean(0) - cs.eye); kf = 1 - math.exp(-max(0, d - 6) * .058)
        q = np.stack([v[:-1:1][:11], v[1:][:11]], 1); t = np.linspace(0, 1, 4)
        strokes_into(img, (q[:, :1] + (q[:, 1:] - q[:, :1]) * t[None, :, None]).astype(f32), np.clip(.011 * cs.f / z[:11], .5, 4), pal[0] * .8 * (1 - kf) + fog * kf, .9, .0, 3, 1)
    v, z = cs.project(LV[:, :3]); d = np.linalg.norm(LV[:, :3] - cs.eye, axis=1); o = np.argsort(-z)
    th = np.arctan2(LV[:, 1], LV[:, 0]); sunny = np.clip(-(np.cos(th) * SUN[0] + np.sin(th) * SUN[1]) * 2, 0, 1) if nm == 'G' else np.clip(moon_band(th, LV[:, 2]) + .12, 0, 1) if nm == 'N' else np.full(len(th), .4)
    r = np.random.default_rng(5)
    for lay in range(2):
        px = LV[:, 3] * cs.f / z * (1.0 if lay == 0 else .62); a = LV[:, 4] + lay * .5
        kf = (1 - np.exp(-np.maximum(0, d - 6) * .058))[:, None]
        tone = np.where((LV[:, 5] + lay * .25)[:, None] < .55, pal[0], pal[1]); tone = np.where((LV[:, 5][:, None] * .5 + sunny[:, None] * .9 + lay * .25) > 1.0, pal[2], tone)
        c = (tone * (1 + r.normal(0, .06, (len(tone), 1)))) * (1 - kf) + fog * kf
        t = np.linspace(-.5, .5, 4)
        P = np.stack([np.stack([v[:, 0] + np.cos(a + .5 * q) * px * q * 2, v[:, 1] + np.sin(a + .5 * q) * px * q * 2], -1) for q in t], 1)
        sel = o if lay == 0 else o[r.random(len(o)) < .55]
        strokes_into(img, P[sel].astype(f32), np.clip(px * .42, .7, 40)[sel], c[sel], .95, 0.0, 9 + lay, 2)


# -------------------------------------------------------------- live stairs
class Stairs:
    """painted stone steps and iron rail, redrawn in true perspective every frame"""
    def __init__(s, seed=5):
        r = np.random.default_rng(seed)
        s.n = n = int((ZD + 3.0) / RISE)
        i = np.arange(n)
        s.th = TH0 + HAND * i * DTH; s.z = -3.0 + i * RISE
        s.jr = r.normal(0, .022, n); s.jz = r.normal(0, .012, n); s.worn = r.uniform(.02, .07, n)
        s.val = 1 + r.normal(0, .035, n) + .05 * np.sin(i * .37); s.hue = r.normal(0, .012, n)
        g = np.convolve(r.uniform(.16, .50, 3 * n + 20), np.ones(3) / 3, 'same'); s.cut = np.stack([g[3 * np.arange(n) + q] for q in range(4)], -1); s.chip = r.random((n, 4)); s.seed = r.integers(0, 1 << 30, n)
        s.crack = [None if r.random() > .30 else (r.uniform(.1, .9), np.cumsum(r.normal(0, .07, 6)), r.uniform(.25, .9)) for _ in range(n)]
        s.st = []
        for k in range(n):
            m = 13
            s.st.append(np.c_[r.uniform(.06, .94, m), r.uniform(.06, .94, m), r.uniform(.18, .55, m), r.uniform(.035, .10, m), r.normal(0, .06, m), r.uniform(.18, .55, m), r.normal(0, .05, m), r.uniform(.0, .6, m)])

    def draw(s, img, cam, k_day=0.0, lit_fn=None, zcull=0.5, strokes=True, fog_n=.056, warm=1.0, rail=True, flat=None):
        """k_day 0 night .. 1 day ambience.  lit_fn(P (n,3)) -> bool mask of sunlit points"""
        e = cam.eye
        order = np.argsort(-s.z)
        stone = rgb('#a79fb0'); iron = rgb('#262339')
        th1 = s.th + HAND * DTH
        ri = RIN + s.jr; zt = s.z + s.jz; zb = zt - .20
        pt = lambda rad, th, z: np.stack([rad * np.cos(th), rad * np.sin(th), z], -1)
        mid = (s.th + th1) / 2
        inner_c = pt(ri, mid, zt - .1)
        nin = np.stack([-np.cos(mid), -np.sin(mid), np.zeros(s.n)], -1)
        facing = nin @ SUN
        lit = (lit_fn(inner_c + nin * .05) & (facing > .05)) if lit_fn is not None else np.zeros(s.n, bool)
        lit_r = lit_fn(inner_c + np.array([0, 0, .9])) if lit_fn is not None else np.zeros(s.n, bool)
        moonf = np.clip(nin @ MOON, 0, 1) * moon_band(mid, zt)

        def sh(i, alb, ao, nz, z, P):
            d = np.linalg.norm(P - e)
            wm = math.exp(-max(z - .5, 0) / 4.0) * .85 * (max(0., -nz) * .8 + .2) * warm
            t = float(smooth(3, 19, z))
            cN = alb * ao * (mix(rgb('#4a4678'), rgb('#6f6ea8'), t) * .98 + rgb('#ffa850') * wm) + rgb('#1c1a30') * .35
            if nz == 0: cN = cN + alb * moonf[i] * rgb('#a8bde8') * .9
            kf = 1 - math.exp(-max(0, d - 6) * fog_n)
            cN = cN * (1 - kf) + FOG_N * .96 * kf
            if k_day <= 0: return cN
            t = float(np.clip((z - 4) / 18, 0, 1))
            cS = alb * ao * (mix(rgb('#6660a8'), rgb('#b598b8'), t) + rgb('#ffc98a') * .15 * max(0., -nz + .5)) + rgb('#2e2a58') * .22
            kf = 1 - math.exp(-max(0, d - 4.5) * .055)
            cS = cS * (1 - kf) + mix(rgb('#8a7cb0'), rgb('#d8b8b4'), t) * kf
            return cN * (1 - k_day) + cS * k_day
        from lp.pages import edge_quads
        for i in order:
            if zt[i] - e[2] < zcull or zb[i] > ZD - .1: continue
            a, b = s.th[i], th1[i]; A = (a + b) / 2
            L = lambda q: a + (b - a) * q
            c = s.cut[i]
            X = np.array([pt(ri[i], a, zb[i]), pt(R0, a, zb[i]), pt(R0, b, zb[i]), pt(ri[i], b, zb[i]),            # 0-3 soffit
                          pt(ri[i], a, zt[i]), pt(ri[i], b, zt[i]), pt(ri[i], a, zb[i] - RISE), pt(R0, a, zb[i] - RISE),   # 4-7
                          pt(ri[i] + c[0], a, zb[i]), pt(ri[i] + c[1], L(.36), zb[i]), pt(ri[i] + c[2], L(.70), zb[i]), pt(ri[i] + c[3], b, zb[i]),  # 8-11 light wedge
                          pt(ri[i], A, zt[i] - s.worn[i]), pt(ri[i], L(.22), zt[i] - s.worn[i] * .6), pt(ri[i], L(.80), zt[i] - s.worn[i] * .5),   # 12-14
                          pt(R0, L(.0), zb[i]), pt(ri[i], L(.12), zb[i]), pt(R0 - .30, L(.0), zb[i]), pt(R0 - .22, L(.5), zb[i]), pt(R0 - .26, L(1.), zb[i])])   # 15-19
            v, z = cam.project(X)
            if z.min() < .3: continue
            if v[:, 0].max() < -60 or v[:, 0].min() > cam.w + 60 or v[:, 1].max() < -60 or v[:, 1].min() > cam.h + 60: continue
            sc = cam.f / z[0]
            alb = np.clip(stone * s.val[i] + s.hue[i] * np.array([1, .1, -1]), 0, 1)
            Pm = X[0]
            cs_ = sh(i, alb, .60, -1, zb[i], Pm); cs2 = sh(i, alb, .80, -1, zb[i], Pm); cs0 = sh(i, alb, .44, -1, zb[i], Pm)
            ce = sh(i, alb, 1.06, 0, zb[i], Pm); cr = sh(i, alb, .40, 0, zb[i], Pm)
            if lit[i]:
                kf = 1 - math.exp(-max(0, np.linalg.norm(Pm - e) - 4.5) * .05)
                ce = (alb * rgb('#ffd98a') * 1.68) * (1 - kf) + FOG_G * kf
                cs2 = cs2 * .84 + rgb('#ff9f62') * .15; cs_ = cs_ * .93 + rgb('#ff9f62') * .05
            if flat is not None: cs_, cs2, cs0, ce, cr = flat[0], flat[0], flat[0], flat[1], flat[0]
            V = np.zeros((8, 6, 2), f32); N = np.zeros(8, np.int32); C = np.zeros((8, 3), f32)
            def setp(k, pts, cc): V[k, :len(pts)] = pts; N[k] = len(pts); C[k] = cc
            setp(0, v[[0, 1, 7, 6]], cr)                                    # riser below the leading edge
            setp(1, v[[0, 1, 2, 3]], cs_)                                   # soffit
            setp(2, v[[17, 18, 19, 2, 1]], cs0)                             # dark against the wall
            setp(3, v[[0, 8, 9, 10, 11, 3]], cs2)                           # brighter inner part
            V[4, :6] = v[[0, 3, 5, 14, 12, 13]]; N[4] = 6; C[4] = ce         # worn end face
            V[5, :3] = v[[0, 13, 4]]; N[5] = 3; C[5] = ce
            fill_polys(img, V[:6], N[:6], C[:6])
            if strokes and sc > 26:
                pa, pb, pc, pd = v[0], v[1], v[2], v[3]
                S = s.st[i]
                t = np.clip(S[:, 1:2] + S[:, 2:3] * np.linspace(-.5, .5, 4)[None], .02, .98)
                ta = np.clip(S[:, 0:1] + S[:, 6:7] * np.sin(np.linspace(0, 3, 4))[None], .03, .97)
                aa = pa + (pb - pa) * t[..., None]; bb = pd + (pc - pd) * t[..., None]
                SP = aa + (bb - aa) * ta[..., None]
                tn = (S[:, 1:2] + S[:, 6:7]).clip(0, 1)
                cbase = np.where(tn < .3, cs2, np.where(tn < .78, cs_, cs0))
                strokes_into(img, SP, np.maximum(.8, S[:, 3] * np.hypot(*(pd - pa)) * .8), np.clip(cbase * (1 + S[:, 4:5]), 0, 1.4), S[:, 5], S[:, 7], int(s.seed[i] & 0xfff), 0)
                q = np.array([.04, .30, .55])[:, None] + np.linspace(0, .36, 4)[None]
                lo = v[0] + (v[3] - v[0]) * q[..., None]; hi = v[4] + (v[5] - v[4]) * q[..., None]
                SP = lo + (hi - lo) * (.28 + .45 * S[:3, 0])[:, None, None]
                strokes_into(img, SP, max(.7, np.hypot(*(v[4] - v[0])) * .16), np.clip(ce * (1 + S[:3, 4:5] * .9), 0, 1.4), .55, .3, int(s.seed[i] & 0xfff) + 1, 0)
                if s.crack[i] is not None and sc > 55:
                    q0, w_, ln = s.crack[i]
                    t = np.linspace(.0, ln, 6); ta = np.clip(q0 + w_, .04, .96)
                    cp = (pa + (pb - pa) * t[:, None]) * (1 - ta[:, None]) + (pd + (pc - pd) * t[:, None]) * ta[:, None]
                    fill_polys(img, edge_quads(cp[:-1], cp[1:], np.linspace(.014, .005, 5).astype(f32) * sc, .8), 6, (cs0 * .72)[None] * np.ones((5, 1)), .85)
            if flat is not None:
                if rail: s._rail(img, cam, i, a, b, A, zt, 0.0, False, flat[0])
                continue
            j = np.array([[v[0], v[1]], [v[0], v[3]], [v[4], v[13]], [v[13], v[12]], [v[12], v[14]], [v[14], v[5]]])
            wj = np.array([.020, .010, .011, .014, .012, .009]) * sc
            dk = cs0 * .62; hi = np.minimum(ce * 1.16, 1.1)
            fill_polys(img, edge_quads(j[:, 0], j[:, 1], np.clip(wj, .5, 8).astype(f32), .4), 6, np.array([dk, dk, hi, hi, hi, hi]), np.array([.9, .6, .7, .8, .75, .6]))
            if flat is not None:
                continue
            if rail: s._rail(img, cam, i, a, b, A, zt, k_day, bool(lit_r[i]), iron)
        return img

    def _rail(s, img, cam, i, a, b, A, zt, k_day, litr, iron):
        from lp.pages import edge_quads
        pt = lambda rad, th, z: np.stack([rad * np.cos(th), rad * np.sin(th), z], -1)
        ra = RIN + .10; e = cam.eye
        za = s.z[i] - RISE * .5; zb_ = s.z[i] + RISE * .5
        n = 3
        tt = np.linspace(-.02, 1.02, n + 1)
        H_ = [pt(ra, a + (b - a) * t, za + (zb_ - za) * t + .95) for t in tt]
        Lw = [pt(ra, a + (b - a) * t, za + (zb_ - za) * t + .16) for t in tt]
        bal = []
        for q in (.25, .75):
            th = a + (b - a) * q; z0 = zt[i] if q < .9 else zt[i]
            zh = za + (zb_ - za) * q
            bal += [pt(ra, th, z0), pt(ra, th, zh + .5), pt(ra, th, zh + .94)]
        vv, zz = cam.project(np.array(H_ + Lw + bal))
        if zz.min() < .25: return
        kf = 1 - math.exp(-max(0, np.linalg.norm(H_[0] - e) - 5) * .040)
        tz = float(np.clip((zt[i] - 4) / 18, 0, 1))
        ci = iron * (1 - kf) + mix(FOG_N * .60, mix(rgb('#6a6098'), rgb('#a58aa8'), tz), k_day) * kf
        s2 = cam.f / zz[0]
        m = n + 1
        hr, lr, bl = vv[:m], vv[m:2 * m], vv[2 * m:]
        P0, P1, WW = [], [], []
        for q in range(2):
            x = bl[q * 3:q * 3 + 3]; big = (i % 5 == 0 and q == 0)
            P0 += [x[0], x[1]]; P1 += [x[1], x[2]]; WW += [.034 if not big else .075, .028 if not big else .065]
            kn = (x[2] - x[0]) * .035
            if s2 > 34: P0.append(x[1] - kn); P1.append(x[1] + kn); WW.append(.075 if not big else .12)
        if s2 > 30:         # a drawn lattice wave between the uprights
            P0 += [bl[0], bl[1], bl[3], bl[4]]; P1 += [bl[4], bl[3], bl[1], bl[5]]; WW += [.0, .015, .015, .0]
        P0 += list(lr[:-1]); P1 += list(lr[1:]); WW += [.030] * n
        P0 += list(hr[:-1]); P1 += list(hr[1:]); WW += [.088] * n
        WW = np.array(WW, f32); keep = WW > 0
        P0, P1, WW = np.array(P0)[keep], np.array(P1)[keep], WW[keep]
        tap = .92
        fill_polys(img, edge_quads(P0, P1, np.clip(WW * s2, .5, 70), tap), 6, ci[None] * np.ones((len(P0), 1)))
        # one broken light along the handrail
        if litr and k_day > 0: hc = rgb('#ffe3a0') * 1.04
        else: hc = mix(rgb('#8890c6') * (.55 + .45 * float(moon_band(A, zt[i]))), FOG_N, kf) * (1 - .25 * k_day)
        if s2 > 12 and (hash01(i, 3) < .86):
            d = hr[-1] - hr[0]; nr = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-6)
            if nr @ np.array([.5, -.86]) < 0: nr = -nr
            off = nr * .026 * s2
            fill_polys(img, edge_quads(hr[:-1] + off, hr[1:] + off, np.clip(np.full(n, .020 * s2, f32), .4, 7), .85), 6, hc[None] * np.ones((n, 1)), .85)
            fill_polys(img, edge_quads(bl[[0, 3]] + off * .3, bl[[2, 5]] + off * .3, np.clip(np.full(2, .008 * s2, f32), .3, 3), .3), 6, hc[None] * np.ones((2, 1)), .5)


# ================================================================== the copper dome
class Dome:
    """underside of the old copper dome: ribs, riveted shutter plates that strain, bend and are flung out"""
    TIERS = (.02, .46, .86, 1.24)

    def __init__(s, seed=21):
        r = np.random.default_rng(seed); s.r = r
        s.ph = [r.uniform(0, 1, 40) for _ in range(NP_ * 3)]
        s.val = r.uniform(.82, 1.15, (NP_, 3)); s.green = r.uniform(.15, .9, (NP_, 3))
        # break order: frame each plate goes
        base = np.array([116, 117, 119, 121, 118, 116, 120, 126, 131, 124, 117, 128])
        s.tb = np.stack([base + 2, base, base + np.array([1, 0, 1, 0, 1, 0, 2, 1, 2, 0, 1, 1])], 1) + r.integers(0, 2, (NP_, 3))
        s.ax = r.normal(0, 1, (NP_, 3, 3)); s.ax /= np.linalg.norm(s.ax, axis=-1, keepdims=True)
        s.sp = r.uniform(.8, 1.35, (NP_, 3)); s.om = r.uniform(.10, .30, (NP_, 3)) * r.choice([-1, 1], (NP_, 3))

    @staticmethod
    def surf(a, ph, off=0.0):
        rr = (R0 - .02 + off) * np.cos(ph); z = ZD + (DOME_H + off) * np.sin(ph)
        return np.stack([rr * np.cos(a), rr * np.sin(a), z + np.zeros_like(rr)], -1)

    def open_frac(s, f):
        return (f >= s.tb[:, 1]).astype(float)

    def plate(s, k, t, f):
        """4x? world corner grid (5 points around) of plate k tier t at frame f, plus state (0 shut, 1 bent, 2 flying)"""
        da = 2 * math.pi / NP_; m = .022
        a0, a1 = k * da + m, (k + 1) * da - m; p0, p1 = s.TIERS[t] + .012, s.TIERS[t + 1] - .012
        am = (a0 + a1) / 2; pm = (p0 + p1) / 2
        A = np.array([a0, am, a1, a1, a1, am, a0, a0]); Pp = np.array([p0, p0, p0, pm, p1, p1, p1, pm])
        P = s.surf(A, Pp); C = s.surf(np.array([am]), np.array([pm]))[0]
        nrm = C - np.array([0, 0, ZD - 1.5]); nrm /= np.linalg.norm(nrm)
        tb = s.tb[k, t]; st = 0
        if f < tb:
            if f >= 106:
                b = np.clip((f - 106) / 10., 0, 1) ** 1.5 * .20 * (1 + .5 * ((f // 2 + k + t) % 2))
                w = np.array([0, 1, 0, .6, 0, 1, 0, .6]); P = P + nrm * (b * w)[:, None] * (1 if (k + t) % 3 else .4)
            return P, C, nrm, 0
        q = f - tb
        if q < 2:       # bends on its outer hinge
            hinge = (P[0] + P[2]) / 2; ax = P[2] - P[0]; ax /= np.linalg.norm(ax); ang = -(.35 + .5 * q)
            v = P - hinge; cc, ss = math.cos(ang), math.sin(ang)
            P = hinge + v * cc + np.cross(ax, v) * ss + ax * (v @ ax)[:, None] * (1 - cc)
            P = P + nrm * (np.array([0, .3, 0, .1, .25, .55, -.05, .1]) * .5)[:, None]
            return P, C, nrm, 1
        d = (1 - math.exp(-q * .085)) * 11.5 * s.sp[k, t]
        out = np.array([math.cos(am), math.sin(am), 0]) * .55 + np.array([0, 0, 1.0])
        ang = s.om[k, t] * q + .8; ax = s.ax[k, t]
        v = (P - C) * 1.0
        v = v + np.cross(nrm, v) * .0 + nrm * (np.array([0, .35, 0, .15, .3, .7, -.1, .2]) * .6)[:, None]
        cc, ss = math.cos(ang), math.sin(ang)
        v = v * cc + np.cross(ax, v) * ss + ax * (v @ ax)[:, None] * (1 - cc)
        return C + v + out * d, C, nrm, 2

    def mask(s, cam, w, h):
        a = np.linspace(0, 2 * math.pi, 73)[:-1]
        v, z = cam.project(s.surf(a, np.zeros_like(a)))
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(v * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
        return m.astype(f32) / 255., v

    def draw(s, img, cam, f, k_day, flick=0.0):
        """paints over the disc of the opening"""
        from lp.pages import edge_quads
        h, w = img.shape[:2]
        M, ring = s.mask(cam, w, h)
        x0, x1 = int(max(0, ring[:, 0].min() - 4)), int(min(w, ring[:, 0].max() + 4)); y0, y1 = int(max(0, ring[:, 1].min() - 4)), int(min(h, ring[:, 1].max() + 4))
        if x1 <= x0 or y1 <= y0: return img
        sub = np.ascontiguousarray(img[y0:y1, x0:x1].copy())
        off = np.array([x0, y0], f32)
        pj = lambda P: cam.project(P)[0].astype(f32) - off
        # sky
        gold = f >= 116
        cen = pj(np.array([[0, 0, ZD + 6.]]))[0]
        yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(f32)
        rad = np.hypot(xx - cen[0], yy - cen[1]) / (ring[:, 0].max() - ring[:, 0].min()) * 2
        if gold:
            q = np.clip(rad, 0, 1.2)[..., None]
            hot = float(np.clip((f - 122) / 34., 0, 1))
            c0, c1, c2 = rgb('#fffbe6') * (1.06 + .10 * hot), rgb('#ffedb8') * (1.0 + .12 * hot), rgb('#ffd488') * (1 + .16 * hot)
            sky = np.where(q < .5, c0 + (c1 - c0) * (q / .5), c1 + (c2 - c1) * np.clip((q - .5) / .6, 0, 1))
            n = fbm(y1 - y0, x1 - x0, (60, 22), (1, .5), 4)[..., None]
            sky = sky + (np.clip((n - .55) * 7, 0, 1)) * (rgb('#fffaf0') * 1.03 - sky) * .55 * (1 - hot * .7)
            sub[:] = sky
        else:
            sub[:] = rgb('#39406f')
        cam_fwd = cam.R[2]
        polys, cols = [], []
        stroke_jobs = []
        cop, ver, dk = rgb('#6d4f4c'), rgb('#54877f'), rgb('#232848')
        fly, shut = [], []
        for k in range(NP_):
            for t in range(3):
                P, C, nrm, st = s.plate(k, t, f)
                (fly if st else shut).append((np.linalg.norm(P.mean(0) - cam.eye), k, t, P, st))
        for _, k, t, P, st in sorted(fly, key=lambda q: -q[0]):
            v = pj(P); dist = np.linalg.norm(P.mean(0) - cam.eye)
            ctr = v.mean(0)
            far = float(np.clip((dist - 23) / 16, 0, 1))
            base = mix(mix(rgb('#8a5c58'), rgb('#6f6a86'), s.green[k, t]), rgb('#f3c987'), far * .8)
            tri = np.array([np.r_[v[[0, 1, 2, 3]], v[[7]]][:6], ], f32)
            fill_polys(sub, np.array([v[:8]]), 8, base[None])
            fill_polys(sub, np.array([[v[1], v[2], v[3], v[4], v[5]]]), 5, (base * .78)[None])
            hi = rgb('#fff3c4') * 1.08
            e = [(0, 1), (1, 2), (4, 5), (5, 6)]
            fill_polys(sub, edge_quads(v[[a for a, b in e]], v[[b for a, b in e]], np.full(4, 2.6, f32), .5), 6, hi[None] * np.ones((4, 1)))
            fill_polys(sub, edge_quads(v[[0, 2, 3, 6]], v[[7, 3, 4, 7]], np.full(4, 1.6, f32), .5), 6, mix(rgb('#6a3f3c'), base, far)[None] * np.ones((4, 1)))
        leak = float(np.clip((f - 104) / 10., 0, 1)) if f < 124 else 0
        for _, k, t, P, st in sorted(shut, key=lambda q: -q[0]):
            v = pj(P)
            tl = (k + .5) / NP_ * 2 * math.pi
            sidev = .5 + .5 * math.cos(tl - 2.6)
            base = mix(cop, ver, s.green[k, t]) * (.52 + .34 * sidev) * s.val[k, t] * np.array([.80, .80, 1.10])
            base = mix(base, FOG_N * .8, .38)
            if k_day > 0:
                b2 = mix(cop, ver, s.green[k, t]) * np.array([1.05, .78, .80]) * .92 * s.val[k, t]; b2 = mix(b2, rgb('#b58a92'), .35)
                base = mix(base, b2, k_day)
            fill_polys(sub, np.array([v]), 8, base[None])
            fill_polys(sub, np.array([[v[1], v[2], v[3], v[4], v[5]]]), 5, (base * .88)[None])
            # verdigris drips and cross bars
            ph = s.ph[k * 3 + t]
            SP, SW, SC = [], [], []
            for j in range(7):
                u, l0, l1 = ph[j], ph[j + 8] * .6, .4 + ph[j + 16] * .6
                a, b = v[0] + (v[2] - v[0]) * u, v[6] + (v[4] - v[6]) * u
                SP.append(np.array([b + (a - b) * (l0 + (l1 - l0) * q) for q in np.linspace(0, 1, 4)])); SW.append(1.2 + 3.2 * ph[j + 24]); SC.append(mix(base, ver * (.62 + .5 * k_day), .45) * (.9 + .35 * ph[j + 30]))
            strokes_into(sub, np.array(SP, f32), np.array(SW), np.clip(np.array(SC), 0, 1.2), .55, .25, 31 + k * 3 + t, 0)
            for u in (.33, .68):
                a, b = v[0] + (v[6] - v[0]) * u, v[2] + (v[4] - v[2]) * u
                fill_polys(sub, edge_quads(a[None], b[None], np.array([3.0], f32), .9), 6, (base * .66)[None])
                fill_polys(sub, edge_quads(a[None] + [0, -1.6], b[None] + [0, -1.6], np.array([1.0], f32), .9), 6, (base * 1.28)[None])
            # rivets
            pts = np.concatenate([v[0] + (v[6] - v[0]) * np.linspace(.08, .92, 5)[:, None], v[2] + (v[4] - v[2]) * np.linspace(.08, .92, 5)[:, None]])
            fill_polys(sub, pts[:, None] + np.array([[-1.2, 0], [0, -1.2], [1.2, 0], [0, 1.2]], f32)[None], 4, (base * 1.25)[None] * np.ones((10, 1)))
            # seam light: moon before, gold when the shutters strain
            if not gold or f < tb_min(s, k, t):
                g = mix(rgb('#9fb4e0') * .8, rgb('#ffe9a8') * 1.1, min(1, leak * 1.6)) if leak > 0 else rgb('#93a6d8') * .72
                on = (hash01(k * 5 + t, 3) < .30 + leak * .8)
                if on:
                    wd = .9 + leak * (2.2 + 3.6 * hash01(k * 7 + t, f // 2))
                    fill_polys(sub, edge_quads(v[[0]], v[[6]], np.array([wd], f32), .35), 6, g[None])
                    if leak > .3: fill_polys(sub, edge_quads(v[[6]], v[[4]], np.array([wd * .8], f32), .35), 6, g[None])
        # ribs and boss
        for k in range(NP_):
            a = k * 2 * math.pi / NP_; hw = .105
            brk = f >= s.tb[k, 2] + 8 and k in (1, 4, 8, 10)
            ph_ = np.linspace(.0, 1.25 if not brk else .62, 9)
            dw = hw / np.maximum(.5, R0 * np.cos(ph_))
            o = -.20
            Pl, Pr = pj(s.surf(a - dw, ph_, o)), pj(s.surf(a + dw, ph_, o)); Pb = pj(s.surf(a + dw * 1.2, ph_, 0)); Pb2 = pj(s.surf(a - dw * 1.2, ph_, 0))
            sidev = .5 + .5 * math.cos(a - 2.6)
            c = mix(rgb('#4b3f50'), rgb('#4d6672'), hash01(k, 9)) * (.55 + .5 * sidev); c = mix(c, FOG_N * .75, .35)
            if k_day > 0: c = mix(c, mix(rgb('#8d5f58'), rgb('#e8b47c'), float(np.clip((f - 124) / 40., 0, .75))), k_day)
            for (A_, B_, m_) in ((Pb2, Pl, .72), (Pr, Pb, .58), (Pl, Pr, 1.0)):
                q = np.stack([A_[:-1], B_[:-1], B_[1:], A_[1:]], 1)
                fill_polys(sub, q, 4, (c * m_)[None] * np.ones((len(q), 1)))
            hc = mix(rgb('#8d9ad0') * .75, rgb('#fff2c0') * 1.1, k_day)
            e = Pl if sidev > .5 else Pr
            fill_polys(sub, edge_quads(e[:-1], e[1:], np.full(8, 1.5 + k_day, f32), .9), 6, hc[None] * np.ones((8, 1)), .7)
        bo = pj(s.surf(np.linspace(0, 2 * math.pi, 25)[:-1] + .2, np.full(24, 1.22) + .012 * np.sin(np.arange(24) * 2.1), -.22))
        if f < 121:
            c = mix(rgb('#3a3f62'), rgb('#7d5a56'), k_day)
            fill_polys(sub, bo[None], 24, c[None])
            fill_polys(sub, (bo.mean(0) + (bo - bo.mean(0)) * .80)[None] + [1, 1.5], 24, (c * .84)[None])
            fill_polys(sub, (bo.mean(0) + (bo - bo.mean(0)) * .50)[None] + [1.5, 2], 24, (c * .66)[None])
            fill_polys(sub, edge_quads(bo, np.roll(bo, -1, 0), np.full(24, 1.8, f32), .9), 6, (c * 1.45)[None] * np.ones((24, 1)))
            for q in range(8): fill_polys(sub, edge_quads(bo[[q * 3]], (bo.mean(0) + (bo[[q * 3]] - bo.mean(0)) * .52), np.array([1.6], f32), .5), 6, (c * .70)[None])
        m = M[y0:y1, x0:x1, None]
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - m) + sub * m
        return img


def tb_min(s, k, t):
    return s.tb[k, t]


# ================================================================== the hall of towering shelves (6a)
class Wall:
    def __init__(s, o, e, n, length, name):
        s.o, s.e, s.n, s.len, s.name = np.asarray(o, float), np.asarray(e, float) / np.linalg.norm(e), np.asarray(n, float) / np.linalg.norm(n), length, name

    def W(s, u, v, w=0.0):
        u, v, w = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(w, float))
        return s.o + u[..., None] * s.e + v[..., None] * np.array([0, 0, 1.]) + w[..., None] * s.n

    def q(s, u0, u1, v0, v1, w0=0.0, w1=None):
        w1 = w0 if w1 is None else w1
        return np.stack([s.W(u0, v0, w0), s.W(u1, v0, w0), s.W(u1, v1, w1), s.W(u0, v1, w1)], -2)


HALL_EYE = np.array([0, 0, 1.05])
WA = Wall((-4.6, 3.6, 0), (.80, .60, 0), (.60, -.80, 0), 11.6, 'A')
WB = Wall((3.02, 3.4, 0), (.36, .933, 0), (-.933, .36, 0), 3.8, 'B')
WC = Wall((2.0, 16.5, 0), (1, .0, 0), (0, -1, 0), 14.0, 'C')
TIERS = np.array([3.62, 7.35, 11.1])
WIN = (6.4, 8.55, 3.2, 8.3)                # window on wall C: u0 u1 sill spring
HMOON = np.array([.42, .55, .72]); HMOON = HMOON / np.linalg.norm(HMOON)


def hall_cam(dx=0., dy=0., push=0.0):
    e = HALL_EYE + np.array([.05, .9, .03]) * push
    c = Camera3D(e, e + np.array([.0, 1, .295]), 1250., W, H, 1024 + dx, 590 + dy)
    return c


def shaft_w(P):
    """painted bar of moonlight crossing the near wall (world x-z-y plane of the beam)"""
    k = P[..., 0] * .74 - P[..., 2] * .66 - (P[..., 1] - 7.) * .10
    m = np.clip((.80 - np.abs(k + 1.75)) / .12 + .5, 0, 1)
    m2 = np.clip((.20 - np.abs(k + .55)) / .10 + .5, 0, 1) * .8
    return np.maximum(m, m2)


def shade_hall(nm, C, A, N, K, M):
    d = np.linalg.norm(C - HALL_EYE, axis=1)
    ao = np.where(K == 3, .40, 1.0) * M
    g = A.mean(1, keepdims=True); A = g + (A - g) * .70
    up = np.clip(C[:, 2] / 9., 0, 1)
    amb = mix(rgb('#4a487c'), rgb('#36345f'), up[:, None])
    side = .80 + .34 * np.clip(N @ np.array([.75, -.35, .55]), -1, 1)
    mb = shaft_w(C) * np.where(K == 3, .30, 1) * (K != 5) * np.clip(.25 + N @ np.array([.3, -.75, .6]), .2, 1.1)
    dl = np.linalg.norm((C - np.array([-1.9, 4.3, .2])) * np.array([1, 1, 1.15]), axis=1)
    warm = np.exp(-dl / 1.25) * 1.0
    c = A * ao[:, None] * (amb * side[:, None] + warm[:, None] * rgb('#ffb458') * 1.1)
    c = c + A * ao[:, None] * mb[:, None] * rgb('#93a8dc') * .62 + mb[:, None] * rgb('#5f6cb0') * .05
    c = c + rgb('#1c1a30') * .34
    k = (1 - np.exp(-np.maximum(0, d - 5.0) * .062))[:, None]
    f = mix(rgb('#33345f'), rgb('#55609a'), np.clip(smooth(5, 17, d) , 0, 1)[:, None])
    return c * (1 - k) + f * k


class Hall:
    def __init__(s, seed=17):
        s.r = np.random.default_rng(seed)
        s.B = dict(w=[], u0=[], u1=[], v=[], h=[], tl=[], col=[], dp=[])
        s.Q = Quads()
        s._wall(WC, 0, rows=(.0, 12.5), detail=0)
        s._wall(WA, 1, rows=(.0, 11.5), detail=2)
        s._wall(WB, 2, rows=(.0, 8.5), detail=2)
        s._floor()
        s.quads = s.Q.done()
        s.B = {k: np.concatenate(v) for k, v in s.B.items()}

    def _floor(s):
        r, Q = s.r, s.Q
        wood = rgb('#6a4b3c')
        P, M = [], []
        d = np.array([.10, .995, 0]); d /= np.linalg.norm(d); e = np.array([d[1], -d[0], 0])
        for x in np.arange(-9, 12, .23):
            y = 3.2 + r.uniform(0, 1.5)
            while y < 17:
                ln = r.uniform(1.2, 2.6); o = e * x + d * y
                P.append([o, o + e * .222, o + e * .222 + d * ln, o + d * ln]); M.append(r.uniform(.50, .86)); y += ln + .01
        P = np.array(P); o = np.argsort(-P[:, :, 1].mean(1))
        Q.add(P[o], wood, np.array([0, 0, 1.]), 1, 1, np.array(M)[o])

    def _wall(s, wl, wid, rows, detail):
        r, Q = s.r, s.Q
        wood, wood_d, wood_l = rgb('#7a5646'), rgb('#4a3838'), rgb('#8d6650')
        bay, post = 1.72, .17
        nb = int(wl.len / bay)
        pitch = .415
        zs = []
        base = .22
        tier0 = 0.0
        for ti, tz in enumerate(np.r_[TIERS, TIERS[-1] + 3.7]):
            z0 = base + (0 if ti == 0 else TIERS[ti - 1] + .06)
            n = int((tz - .28 - z0) / pitch)
            zs.append([z0 + (tz - .28 - z0) / n * k for k in range(n)] + [tz - .28])
        for ti in range(len(zs) - 1, -1, -1):
            zz = zs[ti]
            if zz[0] > rows[1]: continue
            tz = (np.r_[TIERS, TIERS[-1] + 3.7])[ti]
            p_ = zz[1] - zz[0]
            for b in range(nb - 1, -1, -1):
                u0, u1 = b * bay + post / 2, (b + 1) * bay - post / 2
                if wid == 0 and not (u1 < WIN[0] - .25 or u0 > WIN[1] + .25) and zz[0] < WIN[3] + 1.2 and zz[-1] > WIN[2]:
                    for zq in zz[:-1]: Q.add(wl.q(u0, u1, zq, zq + p_, .0), rgb('#6b6680'), wl.n, 2, 2, r.uniform(.6, .78))
                    if zz[0] > WIN[2] - 1: continue
                for j in range(len(zz) - 2, -1, -1):
                    z = zz[j]
                    Q.add(wl.q(u0, u1, z, z + p_, -.26), wood_d, wl.n, 3, 1, r.uniform(.55, .85))
                    Q.add(np.stack([wl.W(u0, z + p_ - .03, .02), wl.W(u1, z + p_ - .03, .02), wl.W(u1, z + p_ - .03, -.26), wl.W(u0, z + p_ - .03, -.26)], -2), wood_d, np.array([0, 0, -1.]), 3, 1, r.uniform(.5, .7))
                    Q.add(np.stack([wl.W(u0, z, .0), wl.W(u0, z + p_, .0), wl.W(u0, z + p_, -.26), wl.W(u0, z, -.26)], -2), wood_d, wl.e, 3, 1, r.uniform(.7, .9))
                    Q.add(np.stack([wl.W(u1, z, .0), wl.W(u1, z + p_, .0), wl.W(u1, z + p_, -.26), wl.W(u1, z, -.26)], -2), wood_d, -wl.e, 3, 1, r.uniform(.8, 1.1))
                    B = books_row(r, u0 + .01, u1 - .02, z, p_ - .075, *((.030, .078) if detail else (.05, .12)))
                    if not len(B): continue
                    a0, a1, hh, tl, ci, dp = B.T
                    sag = -.014 * np.sin(np.pi * (a0 - u0) / (u1 - u0)); v = z + sag
                    alb = jit_color(ALBA[ci.astype(int)], r, .014, .03, .10)
                    P = np.stack([wl.W(a0, v, -dp), wl.W(a1, v, -dp), wl.W(a1 + tl * hh, v + hh, -dp), wl.W(a0 + tl * hh, v + hh, -dp)], -2)
                    Q.add(P, alb, wl.n, 0, 1, r.uniform(.86, 1.12, len(P)))
                    for k_, v_ in (('w', np.full(len(P), wid)), ('u0', a0), ('u1', a1), ('v', v), ('h', hh), ('tl', tl), ('col', alb), ('dp', dp)): s.B[k_].append(v_)
                    if detail:
                        def sub(x0, x1, y0, y1, a, m, st=1, sel=slice(None)):
                            p = P[sel]; l0, l1 = p[:, 0] + (p[:, 1] - p[:, 0]) * x0, p[:, 0] + (p[:, 1] - p[:, 0]) * x1
                            h0, h1 = p[:, 3] + (p[:, 2] - p[:, 3]) * x0, p[:, 3] + (p[:, 2] - p[:, 3]) * x1
                            q = np.stack([l0 + (h0 - l0) * y0, l1 + (h1 - l1) * y0, l1 + (h1 - l1) * y1, l0 + (h0 - l0) * y1], 1)
                            Q.add(q, a[sel] if np.ndim(a) == 2 else a, wl.n, 0, st, m)
                        sub(.0, .24, 0, 1, alb, .62, 1)
                        sub(.80, .96, .02, .98, np.clip(alb * 1.20 + .03, 0, 1), 1.0, 0)
                        gl = np.clip(alb * .5 + rgb('#c9a35a') * .55, 0, 1)
                        for y in (.13, .70, .80):
                            sl = np.nonzero(r.random(len(P)) < .40)[0]
                            if len(sl): yy = y + r.normal(0, .03); sub(.03, .97, yy, yy + .030, gl, .95, 0, sl)
                        sl = np.nonzero(r.random(len(P)) < .22)[0]
                        if len(sl): sub(.12, .88, .40, .56, rgb('#c9b894'), .88, 0, sl)
                        sub(-.02, 1.02, .84, 1.03, alb, .50, 1)
                    seg = np.linspace(u0 - .02, u1 + .02, 7); sg = -.014 * np.sin(np.pi * np.linspace(0, 1, 7))
                    for a_, b_, s0, s1 in zip(seg[:-1], seg[1:], sg[:-1], sg[1:]):
                        Q.add(np.stack([wl.W(a_, z + s0 - .032, .025), wl.W(b_, z + s1 - .032, .025), wl.W(b_, z + s1, .025), wl.W(a_, z + s0, .025)])[None], wood, wl.n, 1, 1, r.uniform(.86, 1.04))
            # posts with a carved capital
            for b in range(nb, -1, -1):
                u = b * bay
                if wid == 0 and WIN[0] - .3 < u < WIN[1] + .3 and zz[0] < WIN[3] + 1.2 and zz[-1] > WIN[2]: continue
                pz = np.linspace(zz[0] - .25, tz - .1, 7)
                for a_, b_ in zip(pz[:-1], pz[1:]):
                    Q.add(wl.q(u - post / 2, u - post * .22, a_, b_, .02, .02)[None], wood, wl.n - wl.e * .5, 1, 1, r.uniform(.55, .66))
                    Q.add(wl.q(u - post * .26, u + post * .26, a_, b_, .075)[None], wood_l, wl.n, 1, 2, r.uniform(.88, 1.06))
                    Q.add(wl.q(u + post * .22, u + post / 2, a_, b_, .02, .02)[None], wood, wl.n + wl.e * .5, 1, 1, r.uniform(.72, .84))
                for k_, (vv, ww, oo) in enumerate(((.62, .11, .10), (.48, .15, .14), (.34, .20, .19)) if wid != 2 else ()):
                    Q.add(wl.q(u - ww * .9, u + ww * .9, tz - vv, tz - vv + .13, .05 + oo * .4, .09 + oo * .6)[None], wood_l, wl.n, 1, 1, r.uniform(.78, 1.0))
            # frieze, gallery slab and balustrade
            U = np.arange(0, nb * bay + .01, bay / 2)
            for a_, b_ in zip(U[:-1], U[1:]):
                if wid == 0 and WIN[0] - .4 < (a_ + b_) / 2 < WIN[1] + .4 and WIN[2] < tz < WIN[3] + 1.2: continue
                Q.add(wl.q(a_, b_, tz - .28, tz - .02, .05)[None], wood, wl.n, 7, 2, r.uniform(.70, .84))
                pr = .78 if detail else .5
                for x0, x1 in ((.05, .30), (.30, .55), (.55, pr)):
                    Q.add(np.stack([wl.W(a_, tz - .04, x1), wl.W(b_, tz - .04, x1), wl.W(b_, tz - .04, x0), wl.W(a_, tz - .04, x0)])[None], wood_d, np.array([0, 0, -1.]), 7, 2, r.uniform(.62, .80))
                Q.add(np.stack([wl.W(a_ - .045, tz - .20, .05), wl.W(a_ + .045, tz - .20, .05), wl.W(a_ + .045, tz - .10, pr - .06), wl.W(a_ - .045, tz - .10, pr - .06)])[None], wood_d, np.array([0, 0, -1.]), 7, 1, .62)
                Q.add(wl.q(a_, b_, tz - .06, tz + .10, pr)[None], wood_l, wl.n, 7, 2, r.uniform(.84, 1.02))
                Q.add(wl.q(a_, b_, tz + .07, tz + .10, pr + .02)[None], wood_l * 1.1, wl.n, 7, 1, r.uniform(.95, 1.1))
                nbz = 6 if detail else 3
                for k_ in range(nbz):
                    uu = a_ + (b_ - a_) * (k_ + .5) / nbz
                    for (z0, z1, hw0) in ((.10, .24, .018), (.24, .46, .033), (.46, .74, .020), (.74, .86, .030)):
                        Q.add(wl.q(uu - hw0, uu + hw0, tz + z0, tz + z1, pr - .05)[None], wood, wl.n, 7, 1, r.uniform(.70, .92))
                Q.add(wl.q(a_, b_, tz + .86, tz + .95, pr - .02)[None], wood_l, wl.n, 7, 1, r.uniform(.92, 1.08))
                Q.add(wl.q(a_ - .05, a_ + .05, tz + .02, tz + 1.0, pr - .03)[None], wood_l, wl.n, 7, 1, r.uniform(.8, .95))
        if wid == 0:
            u0, u1, v0, v1 = WIN; um = (u0 + u1) / 2; rad = (u1 - u0) / 2
            a = np.linspace(0, math.pi, 13)
            pu, pv = um + rad * np.cos(a), v1 + rad * np.sin(a)
            glass = rgb('#b9c8ec')
            rowsz = np.linspace(v0, v1, 7)
            for a_, b_ in zip(rowsz[:-1], rowsz[1:]):
                for k_ in range(4):
                    x0, x1 = u0 + (u1 - u0) * k_ / 4, u0 + (u1 - u0) * (k_ + 1) / 4
                    Q.add(wl.q(x0 + .03, x1 - .03, a_ + .03, b_ - .03, .02)[None], glass * r.uniform(.92, 1.05), wl.n, 6, 2, 1.0)
            for k_ in range(12):
                Q.add(np.stack([wl.W(um, v1, .02), wl.W(pu[k_], pv[k_], .02), wl.W((pu[k_] + pu[k_ + 1]) / 2, (pv[k_] + pv[k_ + 1]) / 2, .02), wl.W(pu[k_ + 1], pv[k_ + 1], .02)])[None], glass * r.uniform(.92, 1.04), wl.n, 6, 1, 1.0)
            for k_ in range(1, 4):
                x = u0 + (u1 - u0) * k_ / 4; Q.add(wl.q(x - .035, x + .035, v0, v1 + rad * math.sqrt(max(0, 1 - ((x - um) / rad) ** 2)), .05)[None], wood_d, wl.n, 1, 0, .6)


def shade_hall2(nm, C, A, N, K, M):
    c = shade_hall(nm, C, A, N, K, M)
    g = K == 6
    if g.any(): c[g] = A[g] * .80
    return c


def open_books(hall, state, drawing, cam, which=(1, 2), only=None, off=None, grow=1.0):
    """polygon batch for every book at a hard drawn state: 1 covers bulge, 2 covers flung wide, 3 pages ripping free, 4 emptied and fluttering.
    returns V (n,M,2), N, kind (0 cover 1 cover shade 2 page lit 3 page half 4 page shade 5 ink), depth, book index"""
    B = hall.B; walls = {0: WC, 1: WA, 2: WB}
    Vs, Ks, Ds, Is = [], [], [], []
    for wid in which:
        ix = np.nonzero(B['w'] == wid)[0]
        if only is not None: ix = ix[only[ix]]
        wl = walls[wid]
        u0, u1, v, h, tl, dp = (B[k][ix] for k in ('u0', 'u1', 'v', 'h', 'tl', 'dp'))
        uc = (u0 + u1) / 2; wd = u1 - u0
        h1, h2, h3, h4 = hash01(ix, 1), hash01(ix, 2), hash01(ix, 3), hash01(ix, 4)
        jd = hash01(ix, 100 + drawing) - .5
        # off(ix) -> du, dv, dw, roll: books that have left the shelf
        du, dv, dw, dr_ = off(ix) if off is not None else (np.zeros(len(ix)),) * 4
        mid = wl.W(uc + du, v + h * .5 + dv, .1 + dw)
        sp, z = cam.project(mid)
        ok = (z > .6) & (sp[:, 0] > -160) & (sp[:, 0] < W + 160) & (sp[:, 1] > -160) & (sp[:, 1] < H + 160)
        sel = np.nonzero(ok)[0]
        ix, u0, u1, v, h, tl, dp, uc, wd, h1, h2, h3, h4, jd, z, du, dv, dw, dr_ = (a[sel] for a in (ix, u0, u1, v, h, tl, dp, uc, wd, h1, h2, h3, h4, jd, z, du, dv, dw, dr_))
        # no two books burst alike: a few hardly move, some twist right over, sizes scatter
        h5, h6, h7 = hash01(ix, 6), hash01(ix, 7), hash01(ix, 8)
        h = h * (1 if state == 1 else np.where(h6 < .14, 1.22, np.where(h6 > .84, .74, .88 + .24 * h5)))
        roll = (h5 - .5) * (.14 if state == 1 else np.where(h7 < .22, 1.25, .46)) + dr_
        vc = v + h * .5; cr_, sr_ = np.cos(roll), np.sin(roll)

        def quad(a0, w0, b0, w1, vlo0, vhi0, vlo1, vhi1, kind, order):
            def T_(a_, v_, w_): a_, v_ = a_ - uc, v_ - vc; return wl.W(uc + du + a_ * cr_ - v_ * sr_, vc + dv + a_ * sr_ + v_ * cr_, w_ + dw)
            P = np.stack([T_(a0, vlo0, w0), T_(b0, vlo1, w1), T_(b0, vhi1, w1), T_(a0, vhi0, w0)], 1)
            p, zz = cam.project(P)
            Vs.append(p); Ks.append(np.broadcast_to(kind, (len(p),)).astype(int)); Ds.append(z * 100 - order); Is.append(ix)
        if state == 1:
            bl = 1.35 + .9 * h1; o = .08 + .09 * h2; lift = .016 * h3
            sk = (h4 - .5) * .10 + tl * h
            lo, hi = v + lift, v + lift + h * (1 + .03 * jd)
            quad(uc - wd * .5 * bl - .034, .01, uc - wd * .36, o, lo, hi - .012, lo, hi, 2, 1)
            quad(uc + wd * .36, o, uc + wd * .5 * bl + .034, .01, lo, hi, lo, hi - .012, 3, 2)
            quad(uc - wd * .5 * bl, o * .55, uc - wd * .1, o + .01, lo - .006, hi + .004, lo - .010, hi + .010, 1, 3)
            quad(uc - wd * .16, o + .01, uc + wd * .5 * bl, o * .6, lo - .010, hi + .010, lo - .006, hi + .004, 0, 4)
            continue
        big = h1 < .42
        huge = hash01(ix, 5) < .10
        L = np.where(huge, .33, np.where(big, .21, .105)) * (.62 + .85 * h2 ** 1.4) * grow * (1.05 if state == 3 else 1.0)
        ph = np.radians(np.where(big, 50 + 56 * h3, 18 + 56 * h3)) * (1.10 if state >= 3 else 1.0)
        pop = np.where(big, .10, .05) * (1 + .4 * jd) + np.where(hash01(ix, 9) < .20, .05 + .30 * hash01(ix, 10), 0) * (state >= 2); lift = (h4 - .3) * .08 + jd * .02
        lo, hi = v + lift * .5, v + h + lift * .5
        tilt = (h2 - .5) * np.where(big, .22, .08)
        for sg, kd in ((-1, 1), (1, 0)):
            a1 = uc + sg * np.sin(ph + (h3 - .5) * .2 * sg) * L; w1 = pop + np.cos(ph) * L
            quad(uc + sg * wd * .18, np.full_like(uc, .03), a1, w1, lo, hi, lo + tilt * sg * L - .012, hi + tilt * sg * L + .02, kd, 0)
        if state in (2, 3):     # dark throat of the open book, so the fan has gaps
            quad(uc - np.sin(ph * .7) * L * .62, pop * .55 + np.cos(ph * .7) * L * .55, uc + np.sin(ph * .7) * L * .62, pop * .55 + np.cos(ph * .7) * L * .55, lo + .012, hi - .016, lo + .012, hi - .016, 5, .5)
        nblade = 5 if state in (2, 3) else 3
        for q in range(nblade):
            f = (q + .5) / nblade * 2 - 1
            hh = hash01(ix, 20 + q); flut = (hash01(ix, 300 + q * 7 + drawing) - .5)
            an = f * ph * .82 + flut * (.22 if state < 4 else .55)
            ln = L * (.80 + .25 * hh) * (1.0 if state == 2 else 1.28 if state == 3 else .9)
            a1 = uc + np.sin(an) * ln; w1 = pop * .8 + np.cos(an) * ln
            am = uc + np.sin(an * .6) * ln * .5; wm = pop * .6 + np.cos(an * .6) * ln * .5
            rise = (hh - .30) * .07 + np.abs(f) * .05 * (.4 + hh) + (.06 * hh if state == 3 else 0)
            kind = 2 if q % 3 == 1 else 3 if q % 3 == 0 else 4
            kind = np.where(hash01(ix, 40 + q) < .5, kind, 2 + (q % 3))
            gap = 0.0
            if state == 3: gap = np.where(hh < .5, .10 + .25 * hash01(ix, 60 + q), 0.)
            show = np.ones(len(ix), bool) if state < 4 else hash01(ix, 80 + q) < .6
            c0 = uc + np.sin(an) * gap; g0 = .03 + np.cos(an) * gap
            quad(np.where(show, c0, uc), np.where(show, g0, 0), np.where(show, am + np.sin(an) * gap, uc), np.where(show, wm + np.cos(an) * gap, 0), lo + .01, hi - .012, lo + .012 + rise * .4, hi - .008 + rise * .5, kind, 1 + (nblade - abs(q - nblade / 2)) * .1)
            quad(np.where(show, am + np.sin(an) * gap, uc), np.where(show, wm + np.cos(an) * gap, 0), np.where(show, a1 + np.sin(an) * gap, uc), np.where(show, w1 + np.cos(an) * gap, 0), lo + .012 + rise * .4, hi - .008 + rise * .5, lo + .02 + rise - tilt * .1, hi - .012 + rise * 1.4 + tilt * .1, np.where(hh < .4, np.minimum(kind + 1, 4), kind), 1.05 + (nblade - abs(q - nblade / 2)) * .1)
    V = np.concatenate(Vs); K = np.concatenate(Ks); D = np.concatenate(Ds); I = np.concatenate(Is)
    o = np.argsort(-D, kind='stable')
    return V[o].astype(f32), K[o], D[o] / 100., I[o]
