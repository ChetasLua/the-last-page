"""S8a  The flock streams in from every side and draws the Great Bird in currents of paper."""
import numpy as np, cv2, math, os
from lp import W, H
from lp.paint import f32, rgb, mix, smooth, ramp, grid, fbm, Canvas, arcs, jit_color
from lp import comp, gb_sky as G, greatbird as gb
from lp.cam import Camera3D
from lp.ink import spl
try:
    from lp import pages as PG
except Exception:
    PG = None

VER = 'v4'
PALB = dict(lit='#f6e3c4', half='#d8b3a4', shade='#9d8bb0', deep='#766a9c', ink='#5a4262', text='#8a7090', rim='#fff6e0')
PALA = dict(lit='#fff0cf', half='#f8cd92', shade='#c9a3a4', deep='#9483ab', ink='#6b4552', text='#9a7a80', rim='#fffaf0')
HZ = 1010
CYC = np.array(getattr(PG, 'CYCLE', (0, 1, 2, 3, 4, 5, 6)))
C0 = np.array([.1, 31., 9.1])


def camera(f):
    t = f / 48.
    e = comp.ease(t)
    return Camera3D((-1.3 + 1.9 * e, -1.6 + 2.4 * e, -.5 + .7 * e), (-1.3 + 1.9 * e - .02 * 100 * (1 - e) * .5, 100, -.5 + .7 * e), 1400, cx=1024, cy=HZ)


def background():
    h, w = G.PH, G.PW; r = np.random.default_rng(71)
    hz = HZ + 99; sun = (1500 + 176, 884 + 99)
    img = G.sky(h, w, [(0, '#4f8bbd'), (.22, '#6fa6c6'), (.42, '#a3c6cb'), (.58, '#d6d6c0'), (.72, '#f4d8ae'), (.87, '#fbcf96'), (1, '#fcc98a')], hz, sun, seed=24, warm='#ffe9c0', warm_r=860, warm_a=.9)
    img = G.paint_sun(img, sun, 44, r, rings=(('#fff3d4', 1.5, .62), ('#ffeac0', 2.3, .44), ('#ffe3b2', 3.6, .32), ('#fedcaa', 5.8, .22)))
    img = G.cirrus(img, [dict(x=700, y=300, ang=-.34, n=190, len=620, spread=(480, 60), col='#eee8d8', op=.13, curve=.30, hw=5),
                         dict(x=1500, y=240, ang=-.28, n=150, len=560, spread=(400, 44), col='#eae7dc', op=.11, curve=.26, hw=5),
                         dict(x=1100, y=560, ang=-.16, n=120, len=640, spread=(540, 30), col='#f9e8cc', op=.18, curve=.14, hw=5),
                         dict(x=1300, y=800, ang=-.06, n=70, len=460, spread=(600, 20), col='#fde4ba', op=.30, curve=.05, hw=5)], r)
    far = np.zeros((h, w, 4), f32); ft = G.T('#fff0d2', '#f9d9b2', '#e8bfa4', '#d3a9aa', '#c0a0b2')
    S = lambda pr, sd, ld, foot, tn, **k: G.comp_over(far, G.cloud(h, w, pr, r, tn, ld, seed=sd, foot=foot, **k))
    S([(1180, 1085, 150, 60), (1330, 1075, 120, 80), (1020, 1095, 150, 46), (1500, 1095, 150, 44)], 5, (.75, -.35, 0), (1090, 1150), ft, thr=(.62, .44, .22, .0), mist='#f9d3a0', ao_h=.3, rimcol='#fff6da', rimw=4, kids=(10, 5), face=4, sizes=(10, 5, 3))
    S([(2180, 1010, 190, 150), (2040, 1060, 160, 100), (2360, 960, 180, 200), (2250, 860, 110, 104), (2380, 800, 120, 120), (1920, 1090, 130, 64)], 6, (-.85, -.28, .0), (1060, 1170), ft, thr=(.63, .44, .20, -.04), mist='#f6cfa2', ao_h=.36, rimcol='#fffbe6', rimw=5, kids=(10, 6), face=5, sizes=(13, 7, 4))
    tn = G.T('#fff2d6', '#fbdab4', '#e3b6a3', '#b893a9', '#8f80ab')
    S([(260, 1080, 300, 170), (60, 980, 240, 230), (220, 820, 190, 180), (440, 960, 170, 150), (70, 720, 180, 180), (250, 660, 120, 120), (140, 590, 104, 104), (600, 1060, 150, 110), (520, 880, 90, 84), (760, 1110, 110, 60), (-20, 860, 160, 220)], 33, (.82, -.30, .02), (1060, 1200), tn,
      thr=(.66, .46, .20, -.10), mist='#f4cda4', ao_h=.42, rimcol='#fffbe6', rimw=5, kids=(11, 6), face=7, amb=.2, sizes=(15, 8, 4.5))
    G.comp_over(img, far)
    def tf(d): return np.stack([mix(a, b, d ** .7) for a, b in zip(G.T('#fff0c8', '#fbd9aa', '#efc3a2', '#d9aeaa', '#cba6b2'), G.T('#fdeaca', '#f0c8aa', '#c79fab', '#9a86af', '#776fa1'))])
    s = G.sea(h, w, hz + 26, 1700, r, sun, tf, lambda d: mix(rgb('#fad3a0'), rgb('#8d81b1'), float(d) ** .62), rows=15, lobe=300, seed=43)
    G.comp_over(img, s)
    return img


# ------------------------------------------------------------------ currents
def bez(P, n=60):
    return spl(np.asarray(P, float), n=n) if len(P) > 2 else np.asarray(P, float)


def spl3(P, n=24):
    """Catmull-Rom in 3D"""
    P = np.asarray(P, float); out = []
    for i in range(len(P) - 1):
        p0 = P[max(i - 1, 0)]; p1 = P[i]; p2 = P[i + 1]; p3 = P[min(i + 2, len(P) - 1)]
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out + [P[-1:]])


def arc_resample(P, step):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    n = max(4, int(d[-1] / step) + 1)
    x = np.linspace(0, d[-1], n)
    return np.stack([np.interp(x, d, P[:, j]) for j in range(3)], -1), d[-1]


def orbit(a0, turns, r0, r1, tilt, n=40, c=C0):
    a = a0 + np.linspace(0, turns * 2 * np.pi, n); R = np.linspace(r0, r1, n)
    P = np.stack([np.cos(a) * R, np.sin(a) * R * .75, np.sin(a + tilt) * R * .42], -1)
    return P + c


RIVERS = {   # name: control points before the orbit, orbit args
    'low': ([(-20, 13, -13), (-19, 18, -6), (-13, 23, 1.5), (-5, 25, 5)], (-1.6, .95, 6.0, 1.6, .3)),
    'left': ([(-62, 56, -2), (-46, 50, 5), (-32, 46, 3), (-20, 41, 4), (-10, 34, 8)], (3.0, .80, 7.5, 1.8, 1.2)),
    'right': ([(58, 48, 4), (42, 44, -4), (29, 38, -3), (18, 30, 3), (9, 25, 8)], (-.9, .85, 7.0, 1.8, 2.0)),
    'top': ([(48, 86, 52), (36, 70, 45), (26, 56, 31), (17, 46, 24), (8, 42, 18)], (1.0, .75, 7.5, 1.7, .8)),
    'under': ([(16, 74, -18), (13, 60, -8), (14, 50, 0), (9, 42, 5), (3, 39, 6)], (1.5, .7, 6, 1.5, 2.6)),
    'nw': ([(-60, 96, 50), (-44, 80, 46), (-30, 62, 33), (-20, 48, 22), (-9, 40, 16)], (2.4, .75, 7.5, 1.7, .2)),
}


class Currents:
    def __init__(self, B):
        self.B = B
        r = np.random.default_rng(909)
        from shots.s8 import K, plume_curves, pose_c
        p = gb.pose(elev=(.42, .32, .12), elevL=(.42, .32, .12), twist=(.10, .02), yaw=math.radians(150), pos=tuple(C0), pitch=.62, adv=.40, bend=.10, fold=.04, fan=1.0, nod=-.13, turn=-.1, tailfan=.9, tailpitch=-.1)
        self.p = p
        D = B.design_lines(p)
        cv = plume_curves(B, 84)
        sh = np.asarray(p['pos']) - np.asarray(pose_c(84)['pos'])
        jobs = []                                    # (river, curve, normal, t_start, speed, spacing, size, kind)
        for k, (P, n) in enumerate(D['spine']): jobs.append((('low', 'under')[k % 2], P, n, 9 + .5 * k, 1.25, .26, .27, 'spine'))
        nb = len(D['bone']) // 2
        for k, (P, n) in enumerate(D['bone']): jobs.append((('left' if k >= nb else 'right'), P, n, 18 + 1.0 * (k % nb), 1.15, .26, .28, 'bone'))
        na = len(D['arc']) // 2
        for k, (P, n) in enumerate(D['arc']): jobs.append((('nw' if k >= na else 'top'), P, n, 22 + 1.6 * (k % na), 1.0, .30, .25, 'arc'))
        nr = len(D['rib']) // 2
        bone = [D['bone'][1][0], D['bone'][nb + 1][0]]
        for k, (P, n) in enumerate(D['rib']):
            j = k % nr; wi = k // nr
            # the rib is fed along the bone: follow the bone to the root of the feather first
            Bn = bone[wi]; i0 = int(np.argmin(np.linalg.norm(Bn - P[0], axis=1)))
            path = np.vstack([Bn[:i0 + 1], P[1:]]) if i0 > 1 else P
            riv = ('right', 'top', 'low') if wi == 0 else ('left', 'nw', 'low')
            jobs.append((riv[j % 3], path, n, 19.5 + .22 * j + r.uniform(0, 1.0), 1.28, .34, .30 if j > nr * .55 else .25, 'rib'))
        for k, (P, ref) in enumerate(cv): jobs.append(('under', P + sh, np.array([0, -1, .3]), 20 + 1.0 * k, 1.25, .40, .32, 'plume'))
        hp = lambda q: B.head_pt(p, np.array(q, float)) @ gb.body_R(p).T + np.asarray(p['pos'])
        for k, q in enumerate(([(0, -.6, .2), (0, -.1, .42), (0, .4, .42), (0, .9, .18), (0, 1.8, -.12), (0, 3.2, -.40)], [(.2, -.6, 0), (.3, 0, .05), (.28, .5, .0), (.12, 1.1, -.08), (.05, 2.0, -.22), (0, 3.2, -.40)],
                               [(-.2, -.6, 0), (-.3, 0, .05), (-.28, .5, .0), (-.12, 1.1, -.08), (-.05, 2.0, -.22), (0, 3.2, -.40)], [(0, -.6, -.25), (0, .1, -.3), (0, .7, -.18), (0, 1.6, -.2), (0, 3.2, -.40)],
                               [(0, .1, .45), (0, -.5, 1.0), (0, -1.4, 1.5), (0, -2.4, 1.4), (0, -3.2, .6)], [(.1, .1, .45), (.2, -.5, .85), (.25, -1.2, 1.2), (.3, -2.0, 1.0), (.3, -2.7, .3)])):
            jobs.append(('low', np.vstack([D['spine'][k % 7][0][60:150:12], spl3(hp(q), 8)]), np.array([0, -1., .4]), 13 + .6 * k, 1.5, .22, .20, 'head'))
        self.jobs = jobs
        self.rivers = {}
        for k, (pts, ob) in RIVERS.items():
            o = orbit(*ob); self.rivers[k] = spl3(np.vstack([pts, o[::6], o[-1:]]), 20)
        # one page table
        T = dict(path=[], slot=[], t0=[], v=[], size=[], lat=[], job=[], sd=[], bird=[], LA=[])
        self.paths = []
        for j, (rv, P, n, t0, v, sp, sz, kind) in enumerate(jobs):
            R = self.rivers[rv]
            # join river to the curve with a short spline
            J = spl3(np.vstack([R[-24:-4:6], P[:1], P[1:4:2]]), 8)
            A = np.vstack([R[:-24], J])
            A, LA = arc_resample(A, .35)
            Q, LQ = arc_resample(P, .2)
            path = np.vstack([A, Q[1:]]); d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            self.paths.append((path, d, LA, LA + LQ, np.asarray(n, float)))
            m = int(LQ / sp)
            for i in range(m):
                slot = LA + LQ - i * LQ / m
                T['path'].append(j); T['slot'].append(slot); T['job'].append(kind); T['size'].append(sz * r.uniform(.75, 1.3))
                # head of stream reaches the curve start at t0; the rest string out behind
                T['t0'].append(t0 + i * sp * .50 / v + r.normal(0, .35)); T['v'].append(v); T['lat'].append(r.normal(0, 1, 3)); T['sd'].append(int(r.integers(0, 1 << 20))); T['bird'].append(r.random() < .72); T['LA'].append(LA)
        # free flyers: keep the rivers full
        names = list(self.rivers)
        for k, nm in enumerate(names):
            A, LA = arc_resample(self.rivers[nm], .35); d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(A, axis=0), axis=1))]
            self.paths.append((A, d, LA, LA, np.array([0, 0, 1.])))
            for i in range(300):
                T['path'].append(len(self.paths) - 1); T['slot'].append(1e9); T['job'].append('free'); T['size'].append(r.uniform(.18, .40) * (1.5 if r.random() < .08 else 1))
                T['t0'].append(-34 + 68 * r.random() ** .8); T['v'].append(r.uniform(1.9, 2.5)); T['lat'].append(r.normal(0, 1, 3)); T['sd'].append(int(r.integers(0, 1 << 20))); T['bird'].append(r.random() < .85); T['LA'].append(LA)
        self.T = {k: np.array(v) for k, v in T.items()}

    def state(self, f):
        T = self.T; n = len(T['slot'])
        free = T['job'] == 'free'
        dt = np.maximum(T['t0'] - f, 0)
        s = np.where(free, T['v'] * (f - T['t0']), T['LA'] + np.where(f < T['t0'], -T['v'] * dt * (1 + 1.5 * dt / (dt + 9.)), T['v'] * (f - T['t0'])))
        arr = np.maximum(0, s - T['slot'])
        ok = (s > 0) & (s < T['LA'] + np.where(free, 0, .8))
        s = np.minimum(s, T['slot'])
        pos = np.zeros((n, 3)); tg = np.zeros((n, 3)); nm = np.zeros((n, 3))
        for j, (path, d, LA, LT, nrm) in enumerate(self.paths):
            m = (T['path'] == j) & ok
            if not m.any(): continue
            x = s[m]
            for c in range(3):
                pos[m, c] = np.interp(x, d, path[:, c]); tg[m, c] = np.interp(x + .4, d, path[:, c]) - np.interp(x - .4, d, path[:, c])
            nm[m] = nrm
        tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-6
        # river breadth: wide far out, a thread at the knot, on the line once it is drawing
        rem = np.clip((T['LA'] - s) / 34., 0, 1)
        br = .06 + 1.15 * rem ** 1.1 + np.where(free, .16, 0)
        # a river is a twisted rope: birds wind round its core
        u1 = np.cross(tg, np.array([0, 0, 1.])); u1 /= np.linalg.norm(u1, axis=1, keepdims=True) + 1e-6; u2 = np.cross(u1, tg)
        a = (T['sd'] % 628) / 100. + s * .16
        rad = br * (.35 + .75 * abs(T['lat'][:, 0]) ** .8)
        pos = pos + u1 * (np.cos(a) * rad)[:, None] + u2 * (np.sin(a) * rad * .8)[:, None] + T['lat'] * (.05 + .22 * rem)[:, None]
        return pos, tg, nm, ok, s, arr


def flock(img, cam, pos, fwd, up, span, design, phase, seed, d, L, **k):
    """the flock lane's drawn birds; plain folded chevrons if that module is missing"""
    try:
        return PG.flock(img, cam, pos, fwd, up, span, design, phase, seed, d, L, **k)
    except Exception:
        return chevrons(img, cam, pos, fwd, span, phase, seed)


def sheets(img, cam, pos, nrm, axis, size, bend, seed, d, L, **k):
    try:
        return PG.sheets(img, cam, pos, nrm, axis, size, bend, seed, d, L, **k)
    except Exception:
        return chevrons(img, cam, pos, axis, np.asarray(size) * .5, np.asarray(seed) % 5, seed)


def chevrons(img, cam, pos, fwd, span, phase, seed):
    from lp.ink import Cel
    xy, z = cam.project(pos); xf, _ = cam.project(np.asarray(pos) + np.asarray(fwd) * .5)
    t = xf - xy; t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-6); px = np.asarray(span) * cam.f / np.maximum(z, .5)
    c = Cel(W, H, ss=1)
    for i in np.argsort(-z):
        if z[i] < 1 or not (-60 < xy[i, 0] < W + 60 and -60 < xy[i, 1] < H + 60): continue
        a = t[i] * px[i] * .55; b = np.array([-t[i, 1], t[i, 0]]) * px[i]; up = np.array([0, -1.]) * px[i] * (.55 - .22 * (int(phase[i]) % 5))
        for s_, col in ((1, '#fdeac6'), (-1, '#b79bb0' if int(seed[i]) % 3 else '#f3c88e')):
            c.fill(np.array([xy[i] + a, xy[i] + b * s_ + up, xy[i] - a]), col)
    l = c.rgba()
    return img * (1 - l[..., 3:4]) + l[..., :3]


def rodrigues(ax, an):
    ax = gb._nrm(ax); c, s = np.cos(an), np.sin(an); x, y, z = ax[:, 0], ax[:, 1], ax[:, 2]; C = 1 - c
    return np.stack([np.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], -1), np.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], -1), np.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1)], 1)


class Assembly:
    """the same bird as 8c, page for page: small and dense at the root first, then every feather lengthens from its quill to its tip"""

    def __init__(self, B, p):
        from shots.s8 import plume_curves, pose_c
        cv = plume_curves(B, 84); sh = np.asarray(p['pos']) - np.asarray(pose_c(84)['pos'])
        cv = [(P + sh, ref) for P, ref in cv]
        # plumes must start exactly at this pose's tail root
        roots, R = B.rib_roots(p)
        cv = [(P - P[0] + roots[k], R @ np.array([1., 0, 0])) for k, (P, ref) in enumerate(cv)]
        self.sc = sc = B.build(p, cv); self.p = p
        n = len(sc['V']); r = np.random.default_rng(31)
        c = sc['V'][:, 1, 1]
        ta = np.full(n, 20.); mid = c.copy(); o = 0
        wmid = lambda P, q: .35 * C0 + .65 * P
        for wi in (0, 1):
            w = B.wings[wi]; n1 = len(w['FI']); nc = len(w['C']['q']); qf = w['q'][w['FI']]; sg = np.clip(w['SG'].mean((1, 2)), 0, 1); tg = np.abs(w['TG'].mean((1, 2))); pr = w['prim'][w['FI']]
            base = np.zeros((len(w['q']), 3))
            for k in range(len(w['q'])):
                m = np.nonzero(w['FI'] == k)[0]; base[k] = c[o + m[np.argmin(sg[m])]]
            ta[o:o + n1] = 17.0 + 9.5 * qf + (9.0 + 4.5 * pr) * sg ** 1.05 + 3.0 * tg * np.clip(sg * 1.6, 0, 1)
            plate = np.arange(n1) >= w['nplate']; ta[o:o + n1][plate & (np.arange(n1) < w['nrach'])] += 4.0
            mid[o:o + n1] = base[w['FI']] * .75 + c[o:o + n1] * .25
            C = w['C']; ta[o + n1:o + n1 + nc] = 14.0 + 12.5 * np.clip(C['q'], 0, 1.05) + 2.2 * (1 - C['row'] / 9.) * .6 + 3.0 * C['big']
            cm = c[o + n1:o + n1 + nc]; mid[o + n1:o + n1 + nc] = C0 * .45 + cm * .55 + (base.mean(0) - C0) * .05
            o += n1 + nc
        nb = len(B.body['s']); s_ = B.body['s']; ta[o:o + nb] = 9.5 + np.where(s_ > .4, 19 * (s_ - .4), 14 * (.4 - s_)); mid[o:o + nb] = c[o:o + nb] * .7 + C0 * .3 + np.array([0, 0, -.6]); o += nb
        nt = len(B.tail['FI']); sg = np.clip(B.tail['SG'].mean((1, 2)), 0, 1); ta[o:o + nt] = 16 + 11 * sg; mid[o:o + nt] = roots.mean(0) * .6 + c[o:o + nt] * .4; o += nt
        nh = n - o
        hd = B.head_pages(p); nh = len(hd['V']); ta[o:o + nh] = 21.5; mid[o:o + nh] = c[o:o + nh] * .9 + C0 * .1; o += nh
        for k, rb in enumerate(B.rib):
            m = len(rb['SG']); sg = np.clip(rb['SG'].mean((1, 2)), 0, 1); tg = np.abs(rb['TG'].mean((1, 2)))
            ta[o:o + m] = 15.5 + .35 * k + 21 * sg ** .9 + 1.6 * tg; P = cv[k][0]; d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            x = np.maximum(sg * d[-1] - 4.5, 0); mid[o:o + m] = np.stack([np.interp(x, d, P[:, j]) for j in range(3)], -1); o += m
        ta[o:] = 25 + r.uniform(0, 3, n - o); mid[o:] = c[o:] + np.array([0, 0, .3])
        # rough hand of chance: a few stragglers keep every edge soft
        rag = r.normal(0, .9, n) + 4.0 * (r.random(n) < .06) * r.random(n)
        solid = np.isin(sc['part'], (7, 9)); rag[solid] *= .25
        self.ta = np.minimum(ta + rag, 45.5); self.T = r.uniform(6.5, 10.5, n); self.mid = mid; self.c = c
        a = r.uniform(0, 6.283, n); self.src = C0 + np.stack([np.cos(a) * 2.6, np.sin(a) * 2.0, np.sin(a + .7) * 1.4], -1) * r.uniform(.4, 1.1, n)[:, None]
        far = sc['part'] != 4
        self.src = np.where(far[:, None], self.src * .55 + self.mid * .45, self.src)
        self.ax = r.normal(0, 1, (n, 3)); self.an = r.normal(0, 1.5, n); self.ph = r.uniform(0, 6.283, n)

    def scene(self, d):
        sc = self.sc; t = np.clip((d - (self.ta - self.T)) / self.T, 0, 1)
        on = d > self.ta - self.T
        e = 1 - (1 - t) ** 1.7
        pos = ((1 - e) ** 2)[:, None] * self.src + (2 * e * (1 - e))[:, None] * self.mid + (e ** 2)[:, None] * self.c
        since = np.maximum(d - self.ta, 0)
        an = self.an * (1 - e) ** .8 + np.where(t >= 1, .22 * np.exp(-since / 3.2) * np.sin(since * 1.1 + self.ph), 0) * (sc['part'] != 9)
        R = rodrigues(self.ax, an); sz = (.42 + .58 * e)[:, None, None, None]
        V = np.einsum('nij,nabj->nabi', R, (sc['V'] - self.c[:, None, None]) * sz) + pos[:, None, None]
        V = np.where(on[:, None, None, None], V, np.array([0, -50., 0]))
        out = dict(sc); out['V'] = V; out['N'] = np.einsum('nij,nj->ni', R, sc['N']); out['NV'] = np.einsum('nij,nabj->nabi', R, sc['NV'])
        return out

    def palette(self, f):
        """sunlit and pale while the pages are still loose, then the shadow side settles into violet and the first primaries light up"""
        k = float(comp.ease((f - 31) / 15.)); P_ = gb.PAL
        pale = dict(t1='#fbe3bb', t2='#eccab0', shade='#cfb4b6', deep='#bb9fac', bounce='#d8bfba', sh_step='#b99aa4', dp_step='#a4889f', lip='#eadad0', ink_sh='#946f76', fline_sh='#8a646c', half='#e6c8a8', bold_sh='#96788a')
        return {n: mix(rgb(v), rgb(P_[n]), k) for n, v in pale.items()}


class ShotA:
    def setup(self, B):
        fn = f'cache/s8a_bg_{VER}.npy'
        if not os.path.exists(fn):
            t = fn + f'.{os.getpid()}.tmp.npy'; np.save(t, background().astype(np.float16)); os.replace(t, fn)
        self.bg = np.load(fn).astype(f32); self.B = B; self.C = Currents(B); self.A = Assembly(B, self.C.p)

    def threads(self, img, cam, f):
        """the drawing itself: light drawn along every current as its head passes"""
        P, wd, col, op, dry = [], [], [], [], []
        r = np.random.default_rng(5)
        for j, (rv, Pn, n, t0, v, sp, sz, kind) in enumerate(self.C.jobs):
            path, d, LA, LT, _ = self.C.paths[j]
            head = LA + v * (f - t0)
            if head < LA - 14: continue
            hd = min(head, LT)
            reps = {'spine': 1, 'bone': 2, 'rib': 2, 'arc': 1, 'plume': 2, 'head': 1}[kind]
            if f > t0 + (LT - LA) / v + 5: continue
            for k in range(reps):
                a = max(LA - 10 * (f < t0 + 8) * r.random(), hd - r.uniform(5, 18)); b = hd - r.uniform(0, .8)
                if head > LA: a = max(LA - 1.0, hd - (LT - LA) * r.uniform(.25, .6))
                if b - a < .6: continue
                x = np.linspace(a, b, 8)
                X = np.stack([np.interp(x, d, path[:, c]) for c in range(3)], -1) + r.normal(0, .09, 3)
                xy, z = cam.project(X)
                if z.min() < 1: continue
                w = {'spine': 3.0, 'bone': 5.0, 'rib': 2.8, 'arc': 2.8, 'plume': 3.6, 'head': 1.6}[kind]
                P.append(xy); wd.append(w * r.uniform(.4, 1.2) * 31. / z.mean()); col.append(rgb(['#fff3d2', '#ffe0a4', '#f6c98e', '#fffaf0'][int(r.integers(0, 4))])); op.append(r.uniform(.35, .8)); dry.append(r.uniform(0, .45))
        if not P: return img
        cv = Canvas(H, W, img=img)
        cv.strokes(np.array(P), np.array(wd), np.array(col), op=np.array(op), dry=np.array(dry), seed=3, shape='leaf', bristle=.04)
        return cv.img

    def rivers_mist(self, img, cam, f):
        """broad dry strokes that give each river a body"""
        P, wd, col, op, dry = [], [], [], [], []
        r = np.random.default_rng(15)
        for nm, R in self.C.rivers.items():
            d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(R, axis=0), axis=1))]
            for k in range(46):
                c0 = r.uniform(.1, 1.02) * d[-1]; L = r.uniform(4, 10); flow = (f * .9 + r.uniform(0, 60)) % 60. / 60.
                x = np.linspace(c0 - L / 2, c0 + L / 2, 8) + (flow - .5) * 10
                rem = np.clip((d[-1] - c0) / 30., 0, 1)
                off = r.normal(0, 1, 3) * (.15 + .75 * rem)
                X = np.stack([np.interp(np.clip(x, 0, d[-1]), d, R[:, c]) for c in range(3)], -1) + off
                xy, z = cam.project(X)
                if z.min() < 4: continue
                P.append(xy); wd.append(r.uniform(.12, .55) * 1400 / z.mean() * .60); col.append(rgb(['#fbe6c2', '#f7d7ae', '#d9bfc4', '#fff0d0'][k % 4])); op.append(r.uniform(.10, .30) * math.sin(math.pi * flow) * (.45 if nm in ('right', 'low') else 1)); dry.append(r.uniform(.4, .8))
        cv = Canvas(H, W, img=img)
        cv.strokes(np.array(P), np.array(wd), np.array(col), op=np.array(op), dry=np.array(dry), seed=13, bristle=.03)
        return cv.img

    def bird(self, img, cam, f):
        d2 = f - f % 2
        sc = self.A.scene(d2)
        lay, ox, oy = gb.draw(sc, cam, haze=(32.5, 46, .58), L=(.60, .42, .68), sun2d=(.72, -.60), rim=4.0, seed=d2 // 2, boil=.4, pal=self.A.palette(f))
        if f > 22 and lay.shape[0] > 4: self.B.draw_head_marks(lay, ox, oy, self.A.p, cam, None, seed=d2)
        from lp.ink import over_roi
        return over_roi(img, lay, ox, oy)

    def swirl(self, img, cam, f, front=False):
        """the knot turning: curved dry-brush air, drawn on twos"""
        d = f - f % 2
        k = float(np.interp(f, [2, 10, 20, 30], [0, 1, .9, 0]))
        if k <= 0: return img
        r = np.random.default_rng(78 if front else 77); P, wd, col, op, dry = [], [], [], [], []
        for i in range(34 if front else 64):
            R = r.uniform(4.2, 10.5) if front else r.uniform(1.6, 8.2); a0 = r.uniform(0, 6.283) + d * (.30 * 4.0 / (R + 1.5)); arc = r.uniform(.5, 1.5); tilt = r.normal(.6, .35)
            a = a0 + np.linspace(0, arc, 9)
            rr = R * (1 - .10 * np.linspace(0, 1, 9))
            X = np.stack([np.cos(a) * rr, np.sin(a) * rr * .8, np.sin(a + tilt) * rr * .42], -1) + C0
            xy, z = cam.project(X)
            P.append(xy); wd.append(r.uniform(.03, .13 if front else .22) * 1400 / z.mean()); col.append(rgb(['#fff0d0', '#fbd9a6', '#cdb2c4', '#f3c795'][i % 4])); op.append(k * r.uniform(.20, .55)); dry.append(r.uniform(.2, .7))
        cv = Canvas(H, W, img=img)
        cv.strokes(np.array(P), np.array(wd), np.array(col), op=np.array(op), dry=np.array(dry), seed=int(d), shape='leaf', bristle=.04)
        return cv.img

    def frame(self, f):
        cam = camera(f); t = comp.ease(f / 48.)
        img = comp.cam2d(self.bg, G.PW / 2 - 24 + 40 * t, G.PH / 2 + 26 - 34 * t, 1.015 + .025 * t)
        img = self.rivers_mist(img, cam, f)
        img = self.swirl(img, cam, f)
        d = f // 2
        pos, tg, nm, ok, s, arr = self.C.state(f)
        T = self.C.T
        idx = np.nonzero(ok)[0]
        cz = cam.cam(pos[idx])[:, 2]
        idx = idx[(cz > 5.0)]
        img = self.threads(img, cam, f)
        pos, tg, nm, s = pos[idx], tg[idx], nm[idx], s[idx]
        sd = T['sd'][idx]; bird = T['bird'][idx]; size = T['size'][idx]; settled = np.clip(arr[idx] / 3., 0, 1)
        drawing_on = (s > T['LA'][idx])
        bird = bird & ~drawing_on
        L = np.array([.58, -.30, .62])
        up = np.array([0, 0, 1.]) + .55 * np.stack([np.sin(sd * .7 + d * .6), np.cos(sd * 1.3 + d * .5), 0 * sd], -1)
        # inside the knot everything leans with the turn, so light falls on it as on one form
        rad = pos - C0; rl = np.linalg.norm(rad, axis=1, keepdims=True) + 1e-6
        kn = (np.clip((10.5 - rl[:, 0]) / 4., 0, 1) * (~drawing_on))[:, None]
        rad = rad / rl
        up = up * (1 - .85 * kn) + (rad * .9 + np.array([0, -.25, .45])) * (.85 * kn)
        # draw back to front in three depth slabs so pages and birds interleave
        z = cam.cam(pos)[:, 2]
        order = np.argsort(-z); slabs = np.array_split(order, 6)
        done = False
        for sl in slabs:
            if len(sl) == 0: continue
            if not done and z[sl].mean() < 31.5: img = self.bird(img, cam, f); done = True
            b = sl[bird[sl]]; p = sl[~bird[sl]]
            fog = (rgb('#f6dcb6'), 36., .012)
            if len(p):
                nmk = nm[p] * (1 - kn[p]) + (rad[p] * .9 + np.array([0, -.25, .45])) * kn[p]
                u = nmk + .45 * (1 - .6 * kn[p]) * np.stack([np.sin(sd[p] * .9 + d * .7), np.cos(sd[p] * 1.7 + d * .55), np.sin(sd[p] * .31 + d * .8)], -1) * (1 - .55 * settled[p])[:, None]
                img = sheets(img, cam, pos[p], gb._nrm(u), tg[p], size[p] * 1.5, .5 + .3 * np.sin(sd[p]), sd[p], d, L, pal=((sd[p] % 10 < 3) & (settled[p] < .5)).astype(int), pals=[PALA, PALB], fog=fog, trans=True)
            if len(b):
                ph = CYC[(d + sd[b]) % 7]
                img = flock(img, cam, pos[b], tg[b], gb._nrm(up[b]), size[b] * 1.15, sd[b] % 13, ph, sd[b], d, L, pal=(sd[b] % 10 < 5).astype(int), pals=[PALA, PALB], fog=fog, trans=False)
        if not done: img = self.bird(img, cam, f)
        img = self.swirl(img, cam, f, True)
        img = self.near(img, cam, f, d, L)
        return img

    def near(self, img, cam, f, d, L):
        """a few big drawn birds sweep past the lens and join"""
        r = np.random.default_rng(64)
        pos, tg, sz, sd, ph = [], [], [], [], []
        ways = [(-9, 8.5, -6.4, 2.4), (8.5, 9.5, -4.6, 1.8), (-8, 11, 4.6, 2.2), (9, 10, 6.0, 2.6), (-3, 8, -5.6, 1.6), (5, 13, -7.5, 2.2), (-12, 14, 1.0, 2.2), (2.5, 9, -6.0, 2.0), (11, 14, 2.5, 2.4), (-6, 9, 6.5, 2.2), (6, 8.5, -5.8, 1.6)]
        for k, (x, y, z_, wob) in enumerate(ways):
            t0 = -6 + 4.6 * k + r.uniform(-1, 1); dur = 22.
            u = (f - t0) / dur
            if u < 0 or u > 1: continue
            A = np.array([x, y, z_]); M_ = np.array([x * .55 + wob, y + 7, z_ * .45 + 3.0]); E = C0 + np.array([x * .25, -2, -2.])
            q = u ** 1.25
            P = lambda q: (1 - q) ** 2 * A + 2 * q * (1 - q) * M_ + q * q * E
            pos.append(P(q)); v = P(min(1, q + .02)) - P(max(0, q - .02)); tg.append(v / np.linalg.norm(v)); sz.append(1.05); sd.append(300 + k * 17); ph.append(CYC[(d + k * 2) % 7])
        if not pos: return img
        pos = np.array(pos); tg = np.array(tg)
        up = gb._nrm(np.array([0, 0, 1.]) + .5 * np.stack([np.sin(np.arange(len(pos)) * 2.1 + d * .12), np.zeros(len(pos)), np.zeros(len(pos))], -1))
        return flock(img, cam, pos, tg, up, np.array(sz), np.array([0, 1, 6, 3, 8, 2, 4, 9, 12, 5, 7])[np.array([(s_ - 300) // 17 for s_ in sd])], np.array(ph), np.array(sd), d, L, pal='dawn')
