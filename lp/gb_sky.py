"""Sky plates for s8: gouache sky, painted sun, cirrus, designed cumulus and cloud sea.
A cloud is a relief of hand-placed lobes lit as a few big masses first and cauliflower crowns second; tones are flat
shapes with hard painted edges, then gone over with bristled strokes that follow the lobes and never cross a silhouette."""
import numpy as np, math, cv2
from numba import njit, prange
from .paint import f32, rgb, mix, smooth, ramp, grid, fbm, Canvas, arcs, jit_color, _vn, mottle

PW, PH = 2400, 1350


def T(*h): return np.stack([rgb(x) for x in h])


# ------------------------------------------------------------------ relief
@njit(cache=True, fastmath=True)
def _relief(Z, ID, A, NX, NY, NZ, X, Y, RX, RY, ZO, S, wob, flat):
    h, w = Z.shape
    for i in range(X.shape[0]):
        cx = X[i]; cy = Y[i]; rx = RX[i]; ry = RY[i]; s = S[i]
        x0 = max(0, int(cx - rx * 1.2)); x1 = min(w - 1, int(cx + rx * 1.2) + 1)
        y0 = max(0, int(cy - ry * 1.2)); y1 = min(h - 1, int(cy + ry * 1.2) + 1)
        rm = .5 * (rx + ry)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                dx = (x - cx) / rx; dy = (y - cy) / ry
                d = math.sqrt(dx * dx + dy * dy)
                if d > 1.2: continue
                ux = dx / (d + 1e-6); uy = dy / (d + 1e-6)
                n = _vn(ux * 2.1 + 10, uy * 2.1 + 10, s) * .55 + _vn(ux * 5.3 + 30, uy * 5.3 + 30, s + 1) * .45
                R = 1 + wob * (n - .5) * 2
                q = d / R
                if q >= 1.0 + 1.0 / rm: continue
                a = (1 - q) * rm * R / 1.3 + .5
                if a > 1: a = 1.0
                if a <= 0: continue
                if a > A[y, x]: A[y, x] = a
                qq = min(q, .999)
                nz = math.sqrt(1 - qq * qq)
                z = ZO[i] + nz * rm * flat
                if z > Z[y, x]:
                    Z[y, x] = z; ID[y, x] = i
                    NX[y, x] = ux * qq; NY[y, x] = uy * qq; NZ[y, x] = nz


def rim_lobes(prim, r, kids=(9, 6), face=5, size=(.42, .40), ctr=None, spread=(-2.9, -.25), minr=5., down=.0):
    """prim: list of (x, y, rx, ry).  Children ride the rim that faces out of the mass; a few bulge from the face."""
    P = np.asarray(prim, float)
    if ctr is None: ctr = np.array([P[:, 0].mean(), P[:, 1].max()])
    out = []
    for k, (x, y, rx, ry) in enumerate(P): out.append((x, y, rx, ry, .0 + .15 * ry, k))
    for k, (x, y, rx, ry) in enumerate(P):
        o = math.atan2(y - ctr[1], x - ctr[0]) if abs(x - ctr[0]) + abs(y - ctr[1]) > 1 else -math.pi / 2
        o = .55 * o + .45 * (-math.pi / 2)
        for j in range(kids[0]):
            a = o + (j / max(1, kids[0] - 1) - .5) * 3.0 + r.normal(0, .13)
            q = r.uniform(.70, .96); s = size[0] * r.uniform(.5, 1.25) * (1.0 if r.random() < .75 else .55)
            cx, cy = x + math.cos(a) * rx * q, y + math.sin(a) * ry * q
            crx, cry = rx * s, ry * s * r.uniform(.85, 1.1)
            out.append((cx, cy, crx, cry, ry * .30 * r.uniform(.4, 1.1), k))
            for i in range(kids[1]):
                b = a + r.normal(0, .95); q2 = r.uniform(.72, 1.0); s2 = size[1] * r.uniform(.45, 1.2)
                if min(crx, cry) * s2 < minr: continue
                out.append((cx + math.cos(b) * crx * q2, cy + math.sin(b) * cry * q2, crx * s2, cry * s2 * r.uniform(.85, 1.1), ry * .30 + cry * .3 * r.uniform(.2, 1.), k))
        for j in range(face):
            a = r.uniform(0, 2 * math.pi); q = math.sqrt(r.uniform(.0, .5))
            s = size[0] * r.uniform(.5, 1.15)
            cx, cy = x + math.cos(a) * rx * q, y + math.sin(a) * ry * q - ry * .1
            out.append((cx, cy, rx * s, ry * s * .9, ry * r.uniform(.72, 1.0), k))
            for i in range(kids[1] // 2):
                b = -math.pi / 2 + r.normal(0, 1.0); s2 = size[1] * r.uniform(.5, 1.1)
                if min(rx, ry) * s * s2 < minr: continue
                out.append((cx + math.cos(b) * rx * s * .85, cy + math.sin(b) * ry * s * .8, rx * s * s2, ry * s * s2, ry * r.uniform(.95, 1.2), k))
    return np.array(out)


def relief(h, w, L, seed=0, wob=.07, flat=1.0):
    Z = np.full((h, w), -1e6, f32); ID = np.full((h, w), -1, np.int32); A = np.zeros((h, w), f32)
    N = [np.zeros((h, w), f32) for _ in range(3)]
    o = np.argsort(L[:, 4]); L = L[o]
    _relief(Z, ID, A, *N, L[:, 0], L[:, 1], L[:, 2], L[:, 3], L[:, 4], ((np.arange(len(L)) * 613 + seed * 977) & 0xffffff).astype(np.int64), wob, flat)
    return Z, ID, A, N, L


def light_mass(L, ID, A, N, prim, ldir, seed=0, big=.62, own=.46, ao=None, rough=.10, amb=.18, cast=.0, Z=None):
    """-> s in about -1..1: mass light + lobe light + brush-wobbled noise"""
    h, w = A.shape
    ldir = np.asarray(ldir, float); ldir = ldir / np.linalg.norm(ldir)
    P = np.asarray(prim, float)
    k = np.where(ID >= 0, L[np.maximum(ID, 0), 5].astype(int), 0)
    x, y = grid(h, w)
    bx = np.clip((x - P[k, 0].astype(f32)) / P[k, 2].astype(f32), -1.2, 1.2) * .85; by = np.clip((y - P[k, 1].astype(f32)) / P[k, 3].astype(f32), -1.2, 1.2) * .85
    bz = np.sqrt(np.clip(1 - bx * bx - by * by, .02, 1))
    sb = bx * ldir[0] + by * ldir[1] + bz * ldir[2]
    so = N[0] * ldir[0] + N[1] * ldir[1] + N[2] * ldir[2]
    lit = np.clip(sb * .9 + .45, 0, 1)
    s = big * sb + own * so * (.35 + .65 * lit)
    if cast > 0 and Z is not None:
        # hard cast shadow: march toward the light over the relief
        sh = np.zeros_like(A)
        for d in (6, 14, 26, 44, 70):
            M = np.float32([[1, 0, -ldir[0] * d], [0, 1, -ldir[1] * d]])
            Zs = cv2.warpAffine(np.maximum(Z, -500), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            sh = np.maximum(sh, ((Zs - Z) > d * (.35 + abs(ldir[2]))).astype(f32))
        s = s - cast * sh
    n = fbm(h, w, (46, 17, 7), (1, .6, .4), seed + 3)
    s = s + rough * n + amb
    if ao is not None: s = s - ao
    return s.astype(f32)


def flat_tones(s, tones, thr=(.62, .34, .06), edge=.05, seed=0):
    """posterise s into flat painted tones: tones[0] lightest.  edge width varies: crisp in places, lost in others"""
    h, w = s.shape
    v = np.clip(.55 + .5 * fbm(h, w, (120, 50), (1, .5), seed + 5), .18, 1.8) * edge
    out = np.empty((h, w, 3), f32); out[:] = tones[-1]
    for i in range(len(tones) - 2, -1, -1):
        k = np.clip((s - thr[i]) / v + .5, 0, 1)[..., None]
        out += (tones[i] - out) * k
    return out


def over_strokes(img, A, ang, r, sizes=(15, 8, 4.5), dens=.9, jit=(.006, .014, .014), op=.8, dry=.12, aspect=3.4, thr=.012, seed=0, bristle=.04, wander=.25):
    """bristled strokes on a mass, colours from its own flat tones, clipped to the silhouette (A)"""
    h, w = A.shape
    lab = (A > .5).astype(np.int32)
    g = img.copy()
    cv = Canvas(h, w, img=img)
    for li, hw in enumerate(sizes):
        step = hw * 1.5 / math.sqrt(dens)
        yy, xx = np.mgrid[step * .5:h:step, step * .5:w:step].astype(f32)
        x = (xx + r.uniform(-.6, .6, xx.shape) * step).ravel().clip(0, w - 1); y = (yy + r.uniform(-.6, .6, xx.shape) * step).ravel().clip(0, h - 1)
        ix, iy = x.astype(int), y.astype(int)
        keep = lab[iy, ix] > 0
        ref = g if hw < 6 else cv2.GaussianBlur(g, (0, 0), hw * .22)
        if li > 0:
            e = np.abs(cv2.GaussianBlur(cv.img, (0, 0), max(.8, hw * .4)) - ref).mean(-1)
            keep &= e[iy, ix] > thr * r.uniform(.3, 1.7, x.size)
        x, y, ix, iy = x[keep], y[keep], ix[keep], iy[keep]
        if not len(x): continue
        o = r.permutation(len(x)); x, y, ix, iy = x[o], y[o], ix[o], iy[o]
        col = jit_color(ref[iy, ix], r, *jit)
        P = arcs(x, y, ang[iy, ix] + r.normal(0, wander, len(x)), hw * aspect * r.uniform(.6, 1.4, len(x)), r.normal(0, .55, len(x)))
        cv.strokes(P, hw * r.uniform(.6, 1.2, len(x)), col, op=op * r.uniform(.7, 1, len(x)), dry=dry * r.uniform(0, 2, len(x)), seed=seed + li * 31, lab=lab, LAB=np.ones(len(x), np.int32), bristle=bristle)
    return cv.img


def cloud(h, w, prim, r, tones, ldir, seed=0, kids=(9, 6), face=5, foot=None, thr=(.62, .34, .06), sizes=(13, 7, 4), ctr=None, size=(.42, .40), wob=.07, big=.62, own=.46,
          rough=.09, cast=.22, ao_h=None, edge=.05, mist=None, amb=.18, rimcol=None, rimw=0, minr=5., op=.8, flat=1.0):
    """one cumulus mass -> premult RGBA.  foot=(y0, y1): lost lower edge"""
    L = rim_lobes(prim, r, kids, face, size, ctr, minr=minr)
    Z, ID, A, N, L = relief(h, w, L, seed, wob, flat)
    x, y = grid(h, w)
    P = np.asarray(prim, float)
    top = (P[:, 1] - P[:, 3]).min(); bot = (P[:, 1] + P[:, 3] * .3).max()
    ao = None
    if ao_h is not None: ao = np.clip((y - top) / (bot - top), 0, 1) ** 1.4 * ao_h
    s = light_mass(L, ID, A, N, prim, ldir, seed, big, own, ao, rough, amb, cast, Z)
    img = flat_tones(s, tones, thr, edge, seed)
    if rimcol is not None and rimw > 0:
        l2 = np.asarray(ldir[:2], float); l2 = l2 / (np.hypot(*l2) + 1e-6)
        M = np.float32([[1, 0, -l2[0] * rimw], [0, 1, -l2[1] * rimw]])
        As = cv2.warpAffine(A, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        rk = np.clip((A - As) * 1.4, 0, 1) * np.clip(.6 + .5 * fbm(h, w, (40, 14), (1, .6), seed + 8), 0, 1)
        img += (rgb(rimcol) - img) * rk[..., None]
    ang = np.arctan2(N[1], N[0]) + np.pi / 2
    m = A > .02
    ys, xs = np.nonzero(m)
    if len(ys):
        ya, yb, xa, xb = max(0, ys.min() - 4), min(h, ys.max() + 5), max(0, xs.min() - 4), min(w, xs.max() + 5)
        sub = over_strokes(img[ya:yb, xa:xb].copy(order='C'), A[ya:yb, xa:xb], ang[ya:yb, xa:xb], r, sizes, seed=seed, op=op)
        img[ya:yb, xa:xb] = sub
    if mist is not None and foot is not None:
        k = smooth(foot[0] - (foot[1] - foot[0]) * .6, foot[1], y + 22 * fbm(h, w, (150, 60, 24), (1, .6, .35), seed + 12)) * .85
        img += (rgb(mist) - img) * k[..., None]
    if foot is not None:
        n = fbm(h, w, (190, 70, 26, 10), (1, .7, .45, .25), seed + 11, aspect=2.2)
        A = A * (1 - smooth(foot[0], foot[1], y + n * (foot[1] - foot[0]) * .32))
    return np.dstack([img * A[..., None], A]).astype(f32)


def comp_over(base, l):
    base *= 1 - l[..., 3:4]; base += l[..., :base.shape[2]]
    return base


# ------------------------------------------------------------------ sky
def sky(h, w, stops, hz, sun, seed=1, warm='#fff0c8', warm_r=900, warm_a=.75, n=(1500, 2600)):
    r = np.random.default_rng(seed)
    x, y = grid(h, w)
    t = y / hz
    g = ramp(np.clip(t, 0, 1), stops)
    d = np.sqrt(((x - sun[0]) / 1.35) ** 2 + ((y - sun[1]) * 1.0) ** 2)
    k = np.clip(1 - d / warm_r, 0, 1) ** 1.7 * warm_a
    g += (rgb(warm) - g) * k[..., None]
    # break the gradient into painted bands
    cv = Canvas(h, w, img=g)
    for pas, (cnt, hw, ln, op, dry) in enumerate(((n[0], (26, 52), (260, 700), .55, .10), (n[1], (10, 24), (120, 420), .5, .25), (900, (3, 9), (60, 240), .35, .5))):
        px = r.uniform(-60, w + 60, cnt).astype(f32); py = (r.uniform(0, 1, cnt) ** .9 * (hz + 60)).astype(f32)
        ix = px.astype(int).clip(0, w - 1); iy = py.astype(int).clip(0, h - 1)
        # strokes bend round the sun
        a = np.arctan2(py - sun[1], px - sun[0]) + np.pi / 2
        dd = np.hypot(px - sun[0], py - sun[1])
        flat = np.clip(1 - dd / 520, 0, 1) ** 1.2
        a = np.where(np.cos(a) < 0, a + np.pi, a); a = (a + np.pi / 2) % np.pi - np.pi / 2
        an = a * (.10 + .9 * flat) + r.normal(0, .05, cnt)
        col = jit_color(g[np.clip(iy + r.normal(0, 14, cnt).astype(int), 0, h - 1), ix], r, .004, .008, .010)
        P = arcs(px, py, an, r.uniform(*ln, cnt) * (1 - .6 * flat), r.normal(0, .12, cnt) + .8 * flat * np.sign(sun[1] - py))
        cv.strokes(P, r.uniform(*hw, cnt) * (1 - .4 * flat), col, op=op * r.uniform(.5, 1, cnt), dry=dry * r.uniform(0, 1.6, cnt), seed=seed + pas * 13, bristle=.022)
    return cv.img


def paint_sun(img, sun, rad, r, core='#fffaf0', rings=(('#fff3cf', 1.55, .62), ('#ffe9b8', 2.4, .42), ('#ffe2a8', 3.7, .30), ('#fddca2', 5.6, .20)), seed=3):
    h, w = img.shape[:2]
    cv = Canvas(h, w, img=img)
    for j, (c, k, op) in reversed(list(enumerate(rings))):
        R0, R1 = rad * (rings[j - 1][1] if j else 1.0), rad * k
        cnt = int(26 + 14 * k)
        th = r.uniform(0, 2 * np.pi, cnt); q = r.uniform(R0 * .98, R1, cnt)
        hw = (R1 - R0) * r.uniform(.22, .5, cnt)
        P = arcs(sun[0] + np.cos(th) * q, sun[1] + np.sin(th) * q, th + np.pi / 2, q * r.uniform(.5, 1.1, cnt), r.uniform(.5, 1.1, cnt))
        col = jit_color(np.tile(rgb(c), (cnt, 1)), r, .004, .006, .006)
        cv.strokes(P, hw, col, op=op * r.uniform(.4, .9, cnt), dry=r.uniform(0, .35, cnt), seed=seed + j, bristle=.03)
    img = cv.img
    x, y = grid(h, w)
    th = np.arctan2(y - sun[1], x - sun[0])
    R = rad * (1 + .012 * np.sin(th * 3 + 1) + .008 * np.sin(th * 7 + 2))
    k = np.clip((R - np.hypot(x - sun[0], y - sun[1])) / 1.6 + .5, 0, 1)
    img += (rgb(core) * 1.06 - img) * k[..., None]
    return img


def cirrus(img, groups, r, seed=5):
    """groups: dict(x, y, ang, n, len, spread, col, op, curve)"""
    h, w = img.shape[:2]
    cv = Canvas(h, w, img=img)
    for gi, g in enumerate(groups):
        n = g['n']
        u = r.normal(0, 1, n) * g['spread'][0]; v = r.normal(0, 1, n) * g['spread'][1]
        c, s = math.cos(g['ang']), math.sin(g['ang'])
        px = g['x'] + u * c - v * s; py = g['y'] + u * s + v * c
        ln = g['len'] * r.uniform(.35, 1.2, n)
        an = g['ang'] + r.normal(0, .06, n) + v / (g['spread'][1] + 1) * g.get('fan', .05)
        P = arcs(px, py, an, ln, g['curve'] * r.uniform(.6, 1.3, n), K=7)
        hw = g.get('hw', 9) * r.uniform(.25, 1.2, n) ** 1.3
        col = jit_color(np.tile(rgb(g['col']), (n, 1)), r, .006, .01, .012)
        o = r.permutation(n)
        cv.strokes(P[o], hw[o], col[o], op=g['op'] * r.uniform(.25, 1, n), dry=r.uniform(.2, .7, n), seed=seed + gi * 7, shape='leaf', bristle=.03)
    return cv.img


# ------------------------------------------------------------------ cloud sea
def ridge_field(seed, cells=((3.1, 1.0), (7.3, .5), (15., .25))):
    """large-scale mound height over (x, depth): elongated ridges and open valleys"""
    r = np.random.default_rng(seed)
    ph = r.uniform(0, 100, (len(cells), 4))

    def f(x, dep, w):
        u = x / w; v = 0.
        for (c, a), p in zip(cells, ph):
            v += a * math.sin(u * c + p[0] + 2.2 * math.sin(dep * 5.1 + p[1])) * math.cos(dep * c * 2.3 + p[2] + 1.3 * math.sin(u * c * .8 + p[3]))
        return .5 + .55 * v
    return f


def sea(h, w, hz, bottom, r, sun, tone_fn, mist_fn, rows=14, lobe=240, power=1.75, squash=.50, seed=0, height=None, thr=(.66, .48, .26, .04), back=-.22, lane=1.0, gap=.30):
    """sea of cloud, painted bank by bank from the horizon, back-lit.  premult RGBA, lost at the horizon"""
    y_ = np.arange(h, dtype=f32)
    base = np.stack([mist_fn(float(np.clip((yy - hz) / (bottom - hz), 0, 1))) for yy in y_]).astype(f32)
    img = np.empty((h, w, 3), f32); img[:] = base[:, None]
    # quiet painted mist between the banks
    cv = Canvas(h, w, img=img); n = 1300
    py = hz + (bottom - hz) * r.uniform(0, 1, n) ** 1.5; px = r.uniform(0, w, n); k = np.clip((py - hz) / (bottom - hz), 0, 1)
    col = jit_color(img[np.clip(py + r.normal(0, 22, n) * k, 0, h - 1).astype(int), px.astype(int)], r, .008, .012, .022)
    cv.strokes(arcs(px, py, r.normal(0, .05, n), 60 + 420 * k * r.uniform(.5, 1.2, n), r.normal(0, .14, n)), 1.5 + 22 * k * r.uniform(.4, 1.2, n), col, op=.6, dry=.3 * r.random(n), seed=seed, bristle=.03)
    img = cv.img
    AA = np.zeros((h, w), f32); AA[:] = np.clip((y_ - hz + 3) / 7., 0, 1)[:, None]
    rf = height if height is not None else ridge_field(seed + 4)
    for k in range(rows):
        d = (k + .5) / rows
        dep = d ** power
        yb = hz + (bottom - hz) * dep
        R = lobe * (.030 + .97 * dep ** .95)
        prim = []
        x = -R * 2
        while x < w + R * 2:
            ht = rf(x, dep, w)
            if ht > gap:
                g = min(1.35, (ht - gap) / (.95 - gap) + .18)
                rr = R * r.uniform(.6, 1.15) * (.50 + .62 * g)
                prim.append((x + r.normal(0, R * .10), yb + r.normal(0, R * .06) - R * squash * .30 * g, rr, rr * squash * r.uniform(.85, 1.2) * (.62 + .5 * g)))
            x += R * r.uniform(.6, 1.2)
        if not prim: continue
        ya = int(max(0, min(p[1] - p[3] * 2 for p in prim))); yc = int(min(h, max(p[1] + p[3] * 1.5 for p in prim) + 2))
        if yc - ya < 4: continue
        pr = [(p[0], p[1] - ya, p[2], p[3]) for p in prim]
        L = rim_lobes(pr, r, (8, 5) if R > 34 else (5, 2) if R > 12 else (3, 0), 3 if R > 50 else 1, (.42, .42), minr=3.0)
        Z, ID, A, N, L = relief(yc - ya, w, L, seed + k * 37, .08)
        xg = (sun[0] - np.arange(w)) / w
        x_, yy_ = grid(yc - ya, w)
        s1 = light_mass(L, ID, A, N, pr, (.60, -.74, back), seed + k, .50, .56, None, .085, .20, .0)
        s2 = light_mass(L, ID, A, N, pr, (-.60, -.74, back), seed + k, .50, .56, None, .085, .20, .0)
        t = np.clip(.5 + xg * 1.8, 0, 1).astype(f32)[None, :]
        s = s1 * t + s2 * (1 - t)
        tones = tone_fn(d)
        col = flat_tones(s, tones, thr, .07, seed + k)
        ln = np.exp(-((np.arange(w) - sun[0]) / (70 + 430 * dep)) ** 2).astype(f32)[None, :, None] * (.75 * (1 - dep) + .25) * lane
        col += (rgb('#fff4d2') - col) * np.clip((s - .30) / .30, 0, 1)[..., None] * ln
        ang = np.arctan2(N[1], N[0]) + np.pi / 2
        if R > 10:
            q = np.clip(R / 120, .28, 1.1)
            col = over_strokes(col.copy(order='C'), A, ang, r, (14 * q, 7 * q, 3.8 * q) if R > 44 else (6 * q + 1, 3 * q + .8), seed=seed + k * 5, op=.72, dens=.85)
        foot = np.array([p[1] for p in pr]).mean(); fr = np.array([p[3] for p in pr]).mean()
        n = fbm(yc - ya, w, (R * 1.6 + 6, R * .6 + 4, R * .25 + 3), (1, .6, .35), seed + k * 3, aspect=2.)
        A = A * (1 - smooth(foot - fr * .05, foot + fr * .95, yy_ + n * fr * .35))
        img[ya:yc] += (col - img[ya:yc]) * A[..., None]
        AA[ya:yc] = np.maximum(AA[ya:yc], A)
    return np.dstack([img * AA[..., None], AA])


# ------------------------------------------------------------------ effects clouds (drawn per frame)
@njit(cache=True, fastmath=True)
def _puffs(img, X, Y, RX, RY, TONE, S, A0, lx, ly, wob):
    """opaque-topped lobes with a lost lower edge, four flat tones; drawn in order"""
    h, w = img.shape[0], img.shape[1]
    for i in range(X.shape[0]):
        cx = X[i]; cy = Y[i]; rx = RX[i]; ry = RY[i]; s = S[i]; op = A0[i]
        if op <= .01 or rx < .8: continue
        x0 = max(0, int(cx - rx * 1.2)); x1 = min(w - 1, int(cx + rx * 1.2) + 1)
        y0 = max(0, int(cy - ry * 1.2)); y1 = min(h - 1, int(cy + ry * 1.2) + 1)
        rm = .5 * (rx + ry)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                dx = (x - cx) / rx; dy = (y - cy) / ry
                d = math.sqrt(dx * dx + dy * dy)
                if d > 1.2: continue
                ux = dx / (d + 1e-6); uy = dy / (d + 1e-6)
                n = _vn(ux * 2.1 + 10, uy * 2.1 + 10, s) * .55 + _vn(ux * 5.3 + 30, uy * 5.3 + 30, s + 1) * .45
                R = 1 + wob * (n - .5) * 2
                q = d / R
                if q >= 1.0: continue
                a = (1 - q) * rm * R / 1.4
                if a > 1: a = 1.0
                f = (dy + .12) / .72
                if f > 0:
                    if f > 1: f = 1.0
                    a *= 1 - f * f * (3 - 2 * f)
                nz = math.sqrt(max(0., 1 - q * q))
                v = (ux * q * lx + uy * q * ly) + nz * .10 + (_vn(x * .05, y * .06, s + 9) - .5) * .30
                k = 0 if v > .52 else 1 if v > .24 else 2 if v > -.10 else 3
                a *= op
                for c in range(3): img[y, x, c] += (TONE[i, k, c] - img[y, x, c]) * a


def puffs(img, L, tones, light=(.5, -.86), seed=0, op=1.0, wob=.09):
    L = np.asarray(L, f32); n = len(L)
    T_ = np.broadcast_to(np.asarray(tones, f32), (n, 4, 3)).copy()
    lx, ly = np.asarray(light, float) / np.hypot(*light)
    _puffs(img, L[:, 0], L[:, 1], L[:, 2], L[:, 3], T_, ((np.arange(n) * 613 + seed * 977) & 0xffffff).astype(np.int64), np.broadcast_to(np.asarray(op, f32), (n,)).copy(), float(lx), float(ly), float(wob))
    return img


def mass_layer(h, w, prims, tones_by, ldir, r, seed=0, kids=(7, 4), face=2, thr=(.64, .44, .20, .0), sizes=(9, 5), fade=(.10, .95), size=(.44, .42), op=.75, rough=.07, minr=3.):
    """a moving bank of cumulus (drawn per drawing), each mass with its own lost foot.  prims rows: x, y, rx, ry, alpha.
    tones_by: (5, 3) tones.  -> premult RGBA"""
    Pm = np.asarray(prims, float)
    L = rim_lobes([tuple(p[:4]) for p in Pm], r, kids, face, size, minr=minr)
    Z, ID, A, N, L = relief(h, w, L, seed, .08)
    s = light_mass(L, ID, A, N, Pm[:, :4], ldir, seed, .52, .54, None, rough, .18, .0)
    col = flat_tones(s, tones_by, thr, .07, seed)
    ang = np.arctan2(N[1], N[0]) + np.pi / 2
    if sizes: col = over_strokes(col.copy(order='C'), A, ang, r, sizes, seed=seed, op=op, dens=.8)
    k = np.where(ID >= 0, L[np.maximum(ID, 0), 5].astype(int), 0)
    x, y = grid(h, w)
    q = (y - Pm[k, 1].astype(f32)) / Pm[k, 3].astype(f32)
    n = fbm(h, w, (46, 18, 7), (1, .6, .4), seed + 3)
    A = A * (1 - smooth(fade[0], fade[1], q + .30 * n)) * Pm[k, 4].astype(f32)
    return np.dstack([col * A[..., None], A]).astype(f32)
