"""The Great Bird: one colossal bird whose every feather is made of book pages.

BirdModel.build(pose, ribbons) -> world-space pages (n, 3, 3, 3) laid in overlapping rows on a posed skeleton.
draw(...) -> premultiplied RGBA cel, flat tones + hard cast shadows + warm ink, deferred at 2x.
Bird space: +x right, +y beak, +z up.  Half span about 12 units."""
import numpy as np, math, cv2
from numba import njit, prange
from . import W, H
from .paint import f32, rgb, mix, _vn
from .paint import rgb as _rgb
from . import gb_art
from .cam import rot_x, rot_y, rot_z

ARM = 8.8
NJ = 10


def _nrm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


# ------------------------------------------------------------------ pose
DEF = dict(elev=(.55, .55, .55), sweep=(.46, .16, -.14), twist=(.10, .0), fan=1.0, bend=.25, bend2=.06, fold=0., adv=0.,
           ruffle=0.0, neck=0.0, turn=0.0, nod=0.0, tailfan=1.0, tailpitch=0.0,
           pos=(0, 0, 0), yaw=0., pitch=0., roll=0., elevL=None, bendL=None)


def pose(**k):
    p = dict(DEF); p.update(k); return p


def body_R(p):
    """bird -> world"""
    return rot_z(p['yaw']) @ rot_x(p['pitch']) @ rot_y(p['roll'])


_TQ = np.linspace(-.12, 1.16, 100)


def wing_table(p, side):
    el = p['elev'] if side > 0 or p['elevL'] is None else p['elevL']
    q = _TQ
    e = np.interp(q, [-.12, .0, .14, .36, .56, .72, .88, 1.06], [el[0] * .75, el[0] * .9, el[0], el[0] * .55 + el[1] * .45, el[1], el[1] * .5 + el[2] * .5, el[2], el[2] * 1.06])
    f = p['fold']
    sw = p['sweep']
    b = np.interp(q, [-.12, .1, .34, .5, .70, .80, 1.06], [sw[0] - .5 * f, sw[0] - .5 * f, sw[0] * .75 + sw[1] * .25 + .1 * f, sw[1] + .35 * f, sw[1] * .6 + sw[2] * .4 - .3 * f, sw[2] - .75 * f, sw[2] * 1.15 - .9 * f])
    t = np.interp(q, [-.1, .6, 1.06], [p['twist'][0], (p['twist'][0] + p['twist'][1]) * .5, p['twist'][1]])
    k = np.array([1, 4, 7, 8, 7, 4, 1.]); k /= k.sum()
    sm = lambda a: np.convolve(np.pad(a, 3, 'edge'), k, 'valid')
    for _ in range(2): e, b, t = sm(e), sm(b), sm(t)
    o = np.zeros_like(q)
    out = np.stack([np.cos(e), o, np.sin(e)], -1)
    nr0 = np.stack([-np.sin(e), o, np.cos(e)], -1)
    bk0 = np.stack([o, o - 1, o], -1)
    ct, st = np.cos(t)[:, None], np.sin(t)[:, None]
    back = bk0 * ct - nr0 * st
    nrm = nr0 * ct + bk0 * st
    d = out * np.cos(b)[:, None] - back * np.sin(b)[:, None]
    dq = q[1] - q[0]
    B = np.cumsum(d, 0) * dq * ARM
    i0 = int(np.argmin(abs(q)))
    B = B - B[i0]
    if p['adv']:
        c, s = math.cos(p['adv']), math.sin(p['adv'])
        Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.]])        # row vectors: x -> x c + y s
        B, out, back, nrm = B @ Rz, out @ Rz, back @ Rz, nrm @ Rz
    return np.concatenate([B + np.array([.62, 1.15, .34]), out, back, nrm], 1)


def wt(tab, q):
    """interpolate the wing table at q (any shape) -> B, out, back, nrm"""
    q = np.clip(q, _TQ[0], _TQ[-1])
    x = (q - _TQ[0]) / (_TQ[1] - _TQ[0])
    i = np.minimum(x.astype(int), len(_TQ) - 2); f = (x - i)[..., None]
    r = tab[i] * (1 - f) + tab[i + 1] * f
    return r[..., :3], r[..., 3:6], r[..., 6:9], r[..., 9:]


def theta(q):
    """feather direction, radians from chord-back toward span-out"""
    return np.interp(q, [-.1, 0, .15, .35, .55, .70, .78, .86, .93, 1.0, 1.06], np.radians([-26, -20, -11, -4, 4, 14, 25, 41, 58, 76, 86]))


def h_up(q, c):
    Hq = np.interp(q, [-.1, 0, .3, .7, 1], [.62, .55, .40, .20, .05])
    pr = np.interp(c, [-.46, -.43, -.36, -.22, .05, .5, 1.2, 2.2, 3.5], [.0, .25, .52, .80, 1., .80, .42, .16, .04])
    return Hq * pr


def h_dn(q, c):
    return .55 * h_up(q, c)


def join(parts):
    """concatenate page sets; flat pages get one normal per page"""
    for q in parts:
        if 'NV' not in q: q['NV'] = np.broadcast_to(q['N'][:, None, None], (len(q['V']), 3, 3, 3))
    return {k: np.concatenate([q[k] for q in parts]) for k in ('V', 'N', 'NV', 'fid', 'thin', 'part', 'art', 'two')}


# ------------------------------------------------------------------ static layout
class BirdModel:
    def __init__(self, seed=11, dens=1.0):
        self.dens = dens
        r = np.random.default_rng(seed)
        self.wings = [self._wing(np.random.default_rng(seed + 1)), self._wing(np.random.default_rng(seed + 2))]
        self.body = self._body(r)
        self.tail = self._tail(r)
        self.rib = self._ribbons(r)
        self.headf = self._head()

    # ---- wings
    def _wing(self, r):
        w = {}
        # flight feathers
        nS, nP = 15, 11
        qs = np.linspace(.035, .70, nS) + r.normal(0, .004, nS)
        Ls = np.interp(qs, [0, .15, .4, .7], [3.3, 4.3, 4.75, 5.05]) * (1 + r.normal(0, .02, nS))
        qp = np.linspace(.735, .965, nP) + r.normal(0, .002, nP)
        kp = np.linspace(0, 1, nP)
        Lp = np.interp(kp, [0, .2, .45, .7, .85, 1], [5.35, 6.1, 7.0, 7.4, 7.1, 6.0]) * (1 + r.normal(0, .015, nP))
        q = np.r_[qs, qp]
        w['q'] = q; w['L'] = np.r_[Ls, Lp]
        w['th'] = np.r_[theta(qs), np.interp(kp, [0, .2, .4, .6, .8, 1], np.radians([18, 31, 45, 59, 72, 84]))] + r.normal(0, .012, len(q))
        w['prim'] = np.r_[np.zeros(nS), .25 + .75 * kp]
        w['curve'] = np.r_[np.full(nS, .10), np.interp(kp, [0, 1], [.16, .36])]
        w['hw'] = np.r_[np.full(nS, .46), np.interp(kp, [0, .5, 1], [.48, .46, .38])] * (1 + r.normal(0, .03, nS + nP))
        w['emarg'] = np.r_[np.zeros(nS), np.interp(kp, [0, .25, 1], [.05, .4, .5])]
        w['tilt'] = np.r_[np.full(nS, .05), np.full(nP, .10)] + r.normal(0, .03, nS + nP)
        w['bj'] = r.normal(0, .05, nS + nP)
        nf = nS + nP
        # pages on feathers
        FI, SG, TG, LG, ART = [], [], [], [], []
        g3 = np.array([0, .5, 1.])
        for k in range(nf):
            M = int(round(w['L'][k] / .38 * self.dens))
            for side in (1, -1):
                for m in range(M):
                    s0 = .10 + .86 * (m + (0.0 if side > 0 else .45)) / M + r.normal(0, .004)
                    ds = (1.75 * .86 / M) * r.uniform(.85, 1.2)
                    sh = .22 / w['L'][k] * r.uniform(.6, 1.6) + .02
                    e = r.uniform(.88, 1.06)
                    # page grid (along a: toward the vane edge; b: along the shaft)
                    a, b = np.meshgrid(g3, g3, indexing='ij')
                    ss = s0 + b * ds * (1 - .1 * a) + sh * a + r.normal(0, .0025, (3, 3)) * a
                    tt = side * (a * e - .07 * (1 - a))
                    ss = np.minimum(ss, 1.0 + .004 * a)
                    lift = .035 * b + .02 * a * r.uniform(-.2, 1.2) * b + r.uniform(-.004, .006)
                    FI.append(k); SG.append(ss); TG.append(tt); LG.append(lift); ART.append(r.integers(0, 1 << 20))
            # tip page
            a, b = np.meshgrid(np.array([-1., 0, 1]), g3, indexing='ij')
            FI.append(k); SG.append(.86 + .152 * b - .012 * abs(a) * b); TG.append(a * .98); LG.append(np.full((3, 3), -.012)); ART.append(r.integers(0, 1 << 20))
        w['nplate'] = len(FI)
        for j, k in enumerate((1, 5, 9, 13)):
            ds = 1.36 / w['L'][k]; s0 = r.uniform(.52, .56); art = gb_art.BOLD0 + (j * 5 + int(r.integers(0, 8)) * 3) % 24
            for sgn in (1, -1):
                a, b = np.meshgrid(np.array([-1., 0, 1]), g3, indexing='ij')
                FI.append(k); SG.append(s0 + b * ds + a * r.normal(0, .006)); TG.append(a * 2.15 + r.normal(0, .06)); LG.append(np.full((3, 3), sgn * .14) + sgn * .03 * b); ART.append(art)
        w['nrach'] = len(FI)
        for k in range(nf):
            for sgn in (1, -1):
                for m in range(4):
                    a, b = np.meshgrid(np.array([-1., 0, 1]), g3, indexing='ij')
                    s0_ = (.60 if sgn > 0 else .52) if k < nS else .30
                    s = s0_ + (.985 - s0_) / 4 * (m + b)
                    FI.append(k); SG.append(s); TG.append(a * .055 * (1.12 - s)); LG.append(np.full((3, 3), sgn * .085)); ART.append(-1)
        w['FI'] = np.array(FI); w['SG'] = np.array(SG); w['TG'] = np.array(TG); w['LG'] = np.array(LG); w['Fart'] = np.array(ART); w['rf'] = r.random(len(FI)) ** 2 * 2
        # coverts: (tip chord, length, width, theta gain)
        up = [(2.95, 2.1, .70, .94), (2.30, 1.65, .62, .92), (1.78, 1.35, .56, .88), (1.36, 1.1, .50, .82), (1.00, .92, .45, .75), (.70, .76, .40, .65), (.42, .62, .36, .52), (.17, .5, .32, .4), (-.07, .4, .29, .25), (-.26, .32, .26, .1)]
        dn = [(2.55, 1.9, .66, .92), (1.90, 1.5, .58, .88), (1.36, 1.2, .52, .82), (.90, .95, .46, .7), (.48, .74, .40, .5), (.12, .56, .34, .3), (-.18, .42, .29, .15)]
        C = {k: [] for k in ('q', 'c0', 'L', 'W', 'kth', 'row', 'sd', 'art', 'lf', 'jt', 'skew', 'rot')}
        for sd, rows in ((1, up), (-1, dn)):
            for ri, (ct, L, Wd, kt) in enumerate(rows):
                span = np.interp(ri, [0, 3, 9], [1.04, 1.03, 1.0])
                n = int(ARM * (span + .08) / (Wd * .70) * self.dens)
                qq = np.linspace(-.075, span, n) + r.normal(0, .0055, n) + (.5 * (ri % 2)) * (span / n)
                taper = np.interp(qq, [-.08, .0, .55, .70, .80, .90, 1.04], [.92, 1.0, 1.0, .88, .62, .40, .16])
                for i in range(n):
                    C['q'].append(qq[i]); C['L'].append(L * taper[i] * r.uniform(.86, 1.2)); C['W'].append(Wd * r.uniform(.9, 1.28) * (.8 + .2 * taper[i]))
                    ctip = -.40 + (ct + .40) * taper[i] + r.normal(0, .035)
                    C['c0'].append(ctip); C['kth'].append(kt); C['row'].append(ri); C['sd'].append(sd); C['art'].append(r.integers(0, 1 << 20))
                    C['lf'].append(.05 + .05 * r.random() ** 3 + (.10 if r.random() < .05 else 0)); C['jt'].append(r.normal(0, .03, (3, 3)))
                    C['skew'].append(r.normal(0, .10)); C['rot'].append(r.normal(0, .07))
        w['C'] = {k: np.array(v) for k, v in C.items()}
        C = w['C']
        for sd, plan in ((1, ((0, 3), (3, 3))), (-1, ((0, 3), (3, 3)))):
            for ri, cnt in plan:
                idx = np.nonzero((C['sd'] == sd) & (C['row'] == ri) & (C['q'] > .05) & (C['q'] < .70))[0]
                off = {0: .05, 3: .42}[ri]
                pick = idx[np.clip(((np.arange(cnt) + off + .25) / (cnt + .25) * len(idx)).astype(int), 0, len(idx) - 1)]
                C['W'][pick] = 1.86; C['L'][pick] = 1.42; C['lf'][pick] = .05; C['skew'][pick] = 0; C['rot'][pick] *= .5; C['jt'][pick] *= .25
                C['art'][pick] = gb_art.BOLD0 + (np.arange(cnt) * 7 + ri * 3 + (11 if sd > 0 else 0) + int(r.integers(0, 24))) % 24
        C['big'] = (C['art'] >= gb_art.BOLD0).astype(float)
        return w

    def wing_pages(self, p, side, wi):
        w = self.wings[wi]
        tab = wing_table(p, side)
        # flight feathers
        bend = p['bend'] if side > 0 or p['bendL'] is None else p['bendL']
        B, out, back, nr = wt(tab, w['q'])
        nf = len(w['q'])
        sj = np.linspace(0, 1, NJ + 1)
        th = w['th'] * (1 + (p['fan'] - 1) * w['prim']) - p['fold'] * .25 * w['prim']
        thj = th[:, None] - w['curve'][:, None] * sj[None] ** 1.2
        bt = (p['bend2'] * (1 - w['prim']) + bend * w['prim'])[:, None] * (sj[None] ** 1.35) + (w['bj'][:, None] * sj[None]) - .03
        dirv = back[:, None] * np.cos(thj)[..., None] + out[:, None] * np.sin(thj)[..., None]
        T = dirv * np.cos(bt)[..., None] + nr[:, None] * np.sin(bt)[..., None]
        N = nr[:, None] * np.cos(bt)[..., None] - dirv * np.sin(bt)[..., None]
        Lat = out[:, None] * np.cos(thj)[..., None] - back[:, None] * np.sin(thj)[..., None]
        c, s = np.cos(w['tilt'])[:, None, None], np.sin(w['tilt'])[:, None, None]
        Lat, N = Lat * c + N * s, N * c - Lat * s
        step = (w['L'] / NJ)[:, None, None]
        P = np.cumsum(np.concatenate([np.zeros((nf, 1, 3)), (T[:, :-1] + T[:, 1:]) * .5 * step], 1), 1)
        base = B + back * (.15 - .1 * w['prim'])[:, None] - nr * (np.arange(nf) * .010 + .02)[:, None]
        P = P + base[:, None]
        fi = w['FI'][:, None, None] * np.ones((1, 3, 3), int)
        x = np.clip(w['SG'], 0, 1.02) * NJ
        j = np.clip(x.astype(int), 0, NJ - 1); fr = (x - j)[..., None]
        lerp = lambda A: A[fi, j] * (1 - fr) + A[fi, j + 1] * fr
        sv = np.clip(w['SG'], 0, 1)
        em = w['emarg'][fi]; pr_ = w['prim'][fi]
        wid = w['hw'][fi] * (.55 + .45 * np.clip(sv / .22, 0, 1)) * (1 - em * np.clip((sv - .42) / .28, 0, 1)) * np.clip((1.015 - sv) / (.16 + .16 * pr_), 0, 1) ** (.55 + .15 * pr_)
        t = w['TG']; wid = wid * np.where(t > 0, .74, 1.16)
        LG = w['LG'] * (1 + p['ruffle'] * 1.5 * w['rf'][:, None, None] * np.clip(abs(t), .1, 1))
        V = lerp(P) + (t * wid)[..., None] * lerp(Lat) + LG[..., None] * lerp(N)
        n1 = V.shape[0]
        fid1 = 100 + 1000 * wi + w['FI']
        NV1 = _nrm(lerp(N)); NF = NV1[:, 1, 1]
        thin1 = np.ones(n1, np.int8)
        part1 = np.full(n1, 1, np.int8); part1[w['nrach']:] = 9
        thin1[w['nplate']:w['nrach']] = 3
        thin1[w['nrach']:] = 2
        # coverts
        C = w['C']
        a = np.array([-.5, 0, .5])[None, :, None]; l = np.array([0, .55, 1.])[None, None, :]
        q = C['q'][:, None, None]; L = C['L'][:, None, None]; Wd = C['W'][:, None, None]
        kk = C['kth'] + (1 - C['kth']) * np.clip((C['q'] - .66) / .26, 0, 1)
        tc = theta(C['q'])[:, None, None] * kk[:, None, None] + C['rot'][:, None, None]
        la = l - 1.0                      # measured back from the tip
        al = la * L * (1 + C['skew'][:, None, None] * a) + C['jt'] * L * .5 * (l > .9)
        wdl = Wd * (1 - .06 * l + .0) * a * 1.0 + C['jt'] * Wd * .4
        cc = C['c0'][:, None, None] + al * np.cos(tc) - wdl * np.sin(tc)
        qq = q + (al * np.sin(tc) + L * np.sin(tc) * .62 + wdl * np.cos(tc)) / ARM
        cc = np.maximum(cc, -.46 * np.interp(qq, [.55, .72, .85, 1.04], [1., .72, .40, .05]))
        B2, o2, b2, n2 = wt(tab, qq)
        sd = C['sd'][:, None, None]
        hgt = np.where(sd > 0, h_up(qq, cc), h_dn(qq, cc))
        lift = C['lf'][:, None, None] * l ** 1.5 * (.8 + .5 * a * C['skew'][:, None, None] * 4) * (1 + p['ruffle'] * 2.2 * (C['art'][:, None, None] % 97) / 97.) + np.where(sd > 0, .065, .095) * np.clip((cc + .40) / .55, 0, 1) + .012 + .10 * C['big'][:, None, None]
        V2 = B2 + b2 * cc[..., None] + n2 * (sd * (hgt + lift))[..., None]
        NV2 = _nrm(n2 * sd[..., None]); N2 = NV2[:, 1, 1]
        # slope of the covert surface: leading edge rounds over
        e1 = V2[:, 2, 1] - V2[:, 0, 1]; e2 = V2[:, 1, 2] - V2[:, 1, 0]
        ng = np.cross(e1, e2); ng = _nrm(ng); ng *= np.sign((ng * N2).sum(-1, keepdims=True) + 1e-9)
        NV2 = _nrm(NV2 * .80 + ng[:, None, None] * .20); N2 = NV2[:, 1, 1]
        nc = V2.shape[0]
        # inner coverts hug the body: fine
        VV = np.concatenate([V, V2]); NN = np.concatenate([NF, N2]); NV = np.concatenate([NV1, NV2])
        fid = np.r_[fid1, 10000 + 4000 * wi + np.arange(nc)]
        thin = np.r_[thin1, np.zeros(nc, np.int8)]
        part = np.r_[part1, np.where(C['sd'] > 0, 2, 3)].astype(np.int8)
        art = np.r_[w['Fart'], C['art']]
        loc1 = .04 + .42 * np.clip((w['SG'].mean((1, 2)) - .62) / .3, 0, 1) * np.clip(w['prim'][w['FI']] * 3, .35, 1)
        loc2 = np.where(C['sd'] > 0, np.interp(C['row'], [0, 1, 2, 4, 9], [.0, .22, .36, .62, .95]), .1)
        loc2 = np.where(C['big'] > 0, np.maximum(loc2, .34), loc2); loc1[w['nplate']:w['nrach']] = .30
        two = np.r_[loc1, loc2].astype(f32)
        if side < 0:
            m_ = np.array([-1., 1, 1]); VV = VV * m_; NN = NN * m_; NV = NV * m_
        return dict(V=VV, N=NN, NV=NV, fid=fid, thin=thin, part=part, art=art, two=two)

    def tips(self, p):
        """world tip positions of the flight feathers: [2][nf, 3]"""
        out = []
        R = body_R(p); X = np.asarray(p['pos'], float)
        for wi, side in ((0, 1), (1, -1)):
            w = self.wings[wi]; tab = wing_table(p, side)
            bend = p['bend'] if side > 0 or p['bendL'] is None else p['bendL']
            B, o, bk, nr = wt(tab, w['q'])
            th = w['th'] * (1 + (p['fan'] - 1) * w['prim']) - p['fold'] * .25 * w['prim'] - w['curve'] * .5
            be = ((p['bend2'] * (1 - w['prim']) + bend * w['prim']) * .55)[:, None]
            d = bk * np.cos(th)[:, None] + o * np.sin(th)[:, None]
            P = B + (d * np.cos(be) + nr * np.sin(be)) * w['L'][:, None]
            if side < 0: P = P * np.array([-1., 1, 1])
            out.append(P @ R.T + X)
        return out

    def design_lines(self, p):
        """the drawing of the bird: world polylines {name: [(P[m,3], normal[3])...]} used by the currents that sketch it"""
        R = body_R(p); X = np.asarray(p['pos'], float)
        out = dict(spine=[], bone=[], rib=[], arc=[])
        w3 = lambda P: P @ R.T + X
        G, T, Xs, U, rx, rz = self.spine(p)
        for k in range(7):
            a = k / 7. * 2 * np.pi
            out['spine'].append((w3(G + Xs * (np.cos(a) * rx)[:, None] + U * (np.sin(a) * rz)[:, None]), R @ (Xs[80] * math.cos(a) + U[80] * math.sin(a))))
        o, x, t, u = self.head_frame(p)
        for wi, side in ((0, 1), (1, -1)):
            m_ = np.array([1., 1, 1]) if side > 0 else np.array([-1., 1, 1])
            w = self.wings[wi]; tab = wing_table(p, side)
            bend = p['bend'] if side > 0 or p['bendL'] is None else p['bendL']
            q = np.linspace(-.02, 1.0, 40); B, o_, bk, nr = wt(tab, q)
            nn = R @ (nr[20] * m_)
            for c in (-.32, .0): out['bone'].append((w3((B + bk * c) * m_), nn))
            for c in (.9, 1.8, 2.8): out['arc'].append((w3((B + bk * (c * np.interp(q, [0, .7, 1], [1, .9, .3]))[:, None]) * m_), nn))
            B, o_, bk, nr = wt(tab, w['q'])
            sj = np.linspace(0, 1, 14)
            th = w['th'] * (1 + (p['fan'] - 1) * w['prim']) - p['fold'] * .25 * w['prim']
            for k in range(len(w['q'])):
                tj = th[k] - w['curve'][k] * sj ** 1.2
                be = (p['bend2'] * (1 - w['prim'][k]) + bend * w['prim'][k]) * sj ** 1.35
                d = bk[k] * np.cos(tj)[:, None] + o_[k] * np.sin(tj)[:, None]
                tn = d * np.cos(be)[:, None] + nr[k] * np.sin(be)[:, None]
                P = B[k] + np.cumsum(np.vstack([np.zeros(3), tn[:-1]]), 0) * w['L'][k] / 13.
                out['rib'].append((w3(P * m_), R @ (nr[k] * m_)))
        return out

    # ---- body and neck: one long tube of small pages
    def _body(self, r):
        s = []; D = {k: [] for k in ('s', 'ph', 'L', 'W', 'art', 'lf', 'jt')}
        pos = .02
        while pos < .985:
            rad = np.interp(pos, [0, .12, .3, .45, .58, .7, .85, 1], [.4, .75, 1.1, 1.15, .75, .48, .36, .30])
            L = np.interp(pos, [0, .3, .5, .62, .8, 1], [1.15, 1.05, .9, .62, .46, .34]); Wd = L * .52
            n = max(6, int(2 * np.pi * rad / (Wd * .78)))
            off = r.uniform(0, 1)
            for k in range(n):
                D['s'].append(pos + r.normal(0, .004)); D['ph'].append((k + off) / n * 2 * np.pi + r.normal(0, .05)); D['L'].append(L * r.uniform(.85, 1.25)); D['W'].append(Wd * r.uniform(.8, 1.45))
                D['art'].append(r.integers(0, 1 << 20)); D['lf'].append(.03 + .06 * r.random() ** 2); D['jt'].append(r.normal(0, .05, (3, 3)))
            pos += L * .36 / 13.2
        return {k: np.array(v) for k, v in D.items()}

    NK0 = np.array([[-3.6, -.10], [-2.6, -.02], [-1.3, .0], [.0, -.04], [1.15, -.06], [2.15, .22], [2.85, .95], [3.0, 1.85], [2.88, 2.80], [2.98, 3.72], [3.45, 4.42], [4.12, 4.62], [4.72, 4.30]])
    NK1 = np.array([[-3.6, -.05], [-2.6, .0], [-1.3, .0], [.0, -.02], [1.15, .02], [2.3, .16], [3.25, .45], [4.1, .80], [4.95, 1.16], [5.8, 1.50], [6.6, 1.72], [7.2, 1.78], [7.7, 1.78]])

    def spine(self, p):
        from .ink import spl
        C = self.NK0 * (1 - p['neck']) + self.NK1 * p['neck']
        S = spl(C, n=14)
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(S, axis=0).T))]
        t = np.linspace(0, 1, 161) * d[-1]
        y = np.interp(t, d, S[:, 0]); z = np.interp(t, d, S[:, 1])
        s = np.linspace(0, 1, 161)
        turn = p['turn'] * np.clip((s - .5) / .5, 0, 1) ** 1.5
        # neck turn: shear sideways progressively
        dy = np.gradient(y); dz = np.gradient(z)
        x = np.cumsum(np.hypot(dy, dz) * np.sin(turn))
        y = y[0] + np.cumsum(dy * np.cos(turn)) - dy[0]
        G = np.stack([x, y, z], -1)
        T = _nrm(np.gradient(G, axis=0))
        X = _nrm(np.stack([np.cos(turn), -np.sin(turn), 0 * s], -1)); X = _nrm(X - T * (X * T).sum(-1, keepdims=True))
        U = np.cross(X, T)
        rx = np.interp(s, [0, .06, .16, .30, .40, .48, .56, .64, .74, .88, 1], [.22, .50, .82, 1.02, 1.06, .92, .70, .50, .37, .28, .225])
        rz = np.interp(s, [0, .06, .16, .30, .40, .48, .56, .64, .74, .88, 1], [.16, .40, .78, 1.08, 1.22, 1.08, .80, .54, .39, .30, .24])
        return G, T, X, U, rx, rz

    def body_pages(self, p):
        D = self.body
        G, T, X, U, rx, rz = self.spine(p)
        a = np.array([-.5, 0, .5])[None, :, None]; l = np.array([0, .55, 1.])[None, None, :]
        SL = 13.2
        s = D['s'][:, None, None] - l * D['L'][:, None, None] / SL + D['jt'] * .02
        rr = np.interp(D['s'], np.linspace(0, 1, 161), (rx + rz) / 2)[:, None, None]
        ph = D['ph'][:, None, None] + a * D['W'][:, None, None] / np.maximum(rr, .2) * (1 - .1 * l) + D['jt'] * .12
        s = np.clip(s, 0, 1)
        i = np.minimum((s * 160).astype(int), 159); f = (s * 160 - i)
        g = lambda A: A[i] * (1 - f)[..., None] + A[i + 1] * f[..., None] if A.ndim == 2 else A[i] * (1 - f) + A[i + 1] * f
        RX, RZ = g(rx), g(rz)
        lift = D['lf'][:, None, None] * l ** 1.4 + .01
        cp, sp = np.cos(ph), np.sin(ph)
        V = g(G) + g(X) * (cp * (RX + lift))[..., None] + g(U) * (sp * (RZ + lift))[..., None]
        NV = _nrm(g(X) * (cp / RX)[..., None] + g(U) * (sp / RZ)[..., None]); N = NV[:, 1, 1]
        n = len(V)
        return dict(V=V, N=N, NV=NV, fid=30000 + np.arange(n), thin=np.zeros(n, np.int8), part=np.full(n, 4, np.int8), art=D['art'], two=(.12 + .5 * np.clip(np.sin(D['ph']), 0, 1) * np.interp(D['s'], [0, .4, .6, 1], [1, 1, .4, .1])).astype(f32))

    # ---- tail fan
    def _tail(self, r):
        n = 12
        D = dict(ang=np.linspace(-.72, .72, n) + r.normal(0, .015, n), L=5.6 - 1.7 * abs(np.linspace(-1, 1, n)) + r.normal(0, .1, n), hw=np.full(n, .46))
        FI, SG, TG, LG, ART = [], [], [], [], []
        g3 = np.array([0, .5, 1.])
        for k in range(n):
            M = int(D['L'][k] / .40 * self.dens)
            for side in (1, -1):
                for m in range(M):
                    a, b = np.meshgrid(g3, g3, indexing='ij')
                    s0 = .04 + .92 * (m + (0 if side > 0 else .45)) / M; ds = 1.8 * .92 / M * r.uniform(.85, 1.2)
                    FI.append(k); SG.append(np.minimum(s0 + b * ds + .07 * a * r.uniform(.6, 1.4), 1.0 + .003 * a)); TG.append(side * (a * r.uniform(.88, 1.08) - .07 * (1 - a))); LG.append(.035 * b); ART.append(r.integers(0, 1 << 20))
        D.update(FI=np.array(FI), SG=np.array(SG), TG=np.array(TG), LG=np.array(LG), art=np.array(ART))
        return D

    def tail_pages(self, p):
        D = self.tail
        n = len(D['ang'])
        sj = np.linspace(0, 1, NJ + 1)
        an = D['ang'] * p['tailfan']
        pit = p['tailpitch']
        pj = pit * (.3 + .7 * sj)[None] + .10 * abs(D['ang'])[:, None]
        thj = an[:, None] * (1 + .12 * sj[None])
        T = np.stack([np.sin(thj) * np.cos(pj), -np.cos(thj) * np.cos(pj), np.sin(pj)], -1)
        step = (D['L'] / NJ)[:, None, None]
        P = np.cumsum(np.concatenate([np.zeros((n, 1, 3)), (T[:, :-1] + T[:, 1:]) * .5 * step], 1), 1)
        P = P + np.stack([np.sin(an) * .25, np.full(n, -3.05), .02 - .02 * abs(np.arange(n) - (n - 1) / 2)], -1)[:, None]
        Lat = np.stack([np.cos(thj), np.sin(thj), 0 * thj], -1)
        N = np.cross(Lat, T); N *= np.sign(N[..., 2:3] + 1e-9)
        fi = D['FI'][:, None, None] * np.ones((1, 3, 3), int)
        x = np.clip(D['SG'], 0, 1.02) * NJ
        j = np.clip(x.astype(int), 0, NJ - 1); fr = (x - j)[..., None]
        lerp = lambda A: A[fi, j] * (1 - fr) + A[fi, j + 1] * fr
        sv = np.clip(D['SG'], 0, 1)
        wid = D['hw'][fi] * (.4 + .6 * np.clip(sv / .4, 0, 1)) * np.sqrt(np.clip((1.03 - sv) / .18, 0, 1))
        V = lerp(P) + (D['TG'] * wid)[..., None] * lerp(Lat) + D['LG'][..., None] * lerp(N)
        m = len(V)
        return dict(V=V, N=_nrm(lerp(N)[:, 1, 1]), NV=_nrm(lerp(N)), fid=40000 + D['FI'], thin=np.ones(m, np.int8), part=np.full(m, 5, np.int8), art=D['art'], two=(.35 * np.clip((D['SG'].mean((1, 2)) - .5) / .5, 0, 1)).astype(f32))

    # ---- streaming plumes: immensely long feathers of pages (world-space centrelines supplied by the act)
    def _ribbons(self, r):
        out = []
        g3 = np.array([0, .5, 1.])
        for k in range(9):
            L = [13, 18, 23, 15, 27, 16, 25, 19, 14][k]
            w0 = [.40, .46, .50, .42, .56, .44, .52, .46, .40][k]
            SG, TG, LG, A = [], [], [], []
            M = int(L / .58 * self.dens)
            for side in (1, -1):
                for m in range(M):
                    a, b = np.meshgrid(g3, g3, indexing='ij')
                    s0 = .99 * (m + (0 if side > 0 else .5)) / M
                    ds = (1.75 / M) * r.uniform(.85, 1.2); sh = .55 / L * r.uniform(.8, 1.3)
                    loose = max(0., s0 - .88) / .12
                    SG.append(s0 + b * ds * (1 - .12 * a) + sh * a + r.normal(0, .06 / L, (3, 3)) * a)
                    TG.append(side * (a * r.uniform(.90, 1.08) - .08 * (1 - a)) + loose * r.normal(0, .18)); LG.append(.05 * b + .06 * a * b * r.uniform(-.4, 1.2) + loose * r.normal(0, .12)); A.append(r.integers(0, 1 << 20))
            # fan of pages at the end
            SG = [np.clip(q, 0, 1) for q in SG]
            nr_ = len(SG)
            for sgn in (1, -1):
                for m in range(16):
                    a, b = np.meshgrid(np.array([-1., 0, 1]), g3, indexing='ij')
                    s = (m + b) / 16.6
                    SG.append(s); TG.append(a * .065 * (1.2 - s)); LG.append(np.full((3, 3), sgn * .16)); A.append(-1)
            out.append(dict(nr=nr_, L=L, SG=np.array(SG), TG=np.array(TG), LG=np.array(LG), art=np.array(A), w0=w0, tw=r.uniform(-1, 1), root=np.array([(k - 4) * .13, -3.2, .06 * math.cos(k * 2)])))
        return out

    def ribbon_pages(self, curves, flutter=0.0):
        """curves: list of (P[m,3] dense world polyline from the root, lateral reference) per plume"""
        VV, NN, FF, AA, LO, PT = [], [], [], [], [], []
        for k, (R, cu) in enumerate(zip(self.rib, curves)):
            P, ref = cu
            d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            # dense frames
            m = 72; x = np.linspace(0, d[-1], m)
            C = np.stack([np.interp(x, d, P[:, c]) for c in range(3)], -1)
            T = _nrm(np.gradient(C, axis=0))
            X = _nrm(ref - T * (T @ ref)[:, None]); N = np.cross(X, T)
            q = x / d[-1]
            ang = R['tw'] * .25 + (.12 + .5 * q) * np.sin(x * .30 + k * 1.7 + flutter)
            X, N = X * np.cos(ang)[:, None] + N * np.sin(ang)[:, None], N * np.cos(ang)[:, None] - X * np.sin(ang)[:, None]
            s = np.clip(R['SG'], 0, .9999) * (m - 1)
            i = np.minimum(s.astype(int), m - 2); f = (s - i)[..., None]
            lerp = lambda A: A[i] * (1 - f) + A[i + 1] * f
            sv = np.clip(R['SG'], 0, 1)
            wd = R['w0'] * np.interp(sv, [0, .05, .2, .45, .7, .85, .95, 1.0], [.30, .55, .72, .95, 1.1, .95, .5, .08])
            V = lerp(C) + lerp(X) * (R['TG'] * wd)[..., None] + lerp(N) * (R['LG'] - .012 * k)[..., None]
            VV.append(V); NN.append(_nrm(lerp(N))); FF.append(np.full(len(V), 50000 + k)); AA.append(R['art']); LO.append(.45 * sv[:, 1, 1] ** 1.5); PT.append(np.where(np.arange(len(V)) >= R['nr'], 9, 6))
        V = np.concatenate(VV); n = len(V)
        pt = np.concatenate(PT).astype(np.int8)
        NV = np.concatenate(NN)
        return dict(V=V, N=NV[:, 1, 1], NV=NV, fid=np.concatenate(FF), thin=np.where(pt == 9, 2, 1).astype(np.int8), part=pt, art=np.concatenate(AA), two=np.concatenate(LO).astype(f32))

    # ---- head: folded paper, a faceted loft
    ST = np.array([[-.62, -.05, .205, .225], [-.30, .03, .265, .285], [.06, .07, .30, .325], [.40, .06, .285, .30], [.74, .0, .215, .225], [1.02, -.06, .140, .145],
                   [1.45, -.11, .090, .090], [2.15, -.20, .052, .052], [3.25, -.40, .007, .007]])

    def _head(self):
        return None

    def head_frame(self, p):
        G, T, X, U, rx, rz = self.spine(p)
        o = G[-1]; t = T[-1]; x = X[-1]; u = U[-1]
        n = p['nod']
        t, u = t * math.cos(n) + u * math.sin(n), u * math.cos(n) - t * math.sin(n)
        return o, x, t, u

    def head_pt(self, p, v):
        """head-local (side, forward, up) -> bird space"""
        o, x, t, u = self.head_frame(p)
        v = np.asarray(v, float)
        return o + v[..., 0:1] * x + v[..., 1:2] * t + v[..., 2:3] * u

    def head_pages(self, p):
        S = self.ST; K = 8
        an = (np.arange(K + 1) + .5) / K * 2 * np.pi
        ring = np.stack([np.stack([np.cos(an) * s[2], np.full(K + 1, s[0]), s[1] + np.sin(an) * s[3]], -1) for s in S])
        g = np.array([0, .5, 1.])
        VV, NN, LO, FI = [], [], [], []
        for i in range(len(S) - 1):
            for k in range(K):
                A, B, C, D = ring[i, k], ring[i + 1, k], ring[i + 1, k + 1], ring[i, k + 1]
                a, b = np.meshgrid(g, g, indexing='ij'); a = a[..., None]; b = b[..., None]
                VV.append((A * (1 - b) + B * b) * (1 - a) + (D * (1 - b) + C * b) * a)
                m = (A + B + C + D) / 4; n = np.array([m[0], (S[i, 2] - S[i + 1, 2]) * .9, m[2] - S[i, 1]]); NN.append(n / np.linalg.norm(n))
                LO.append(1.25 if i >= 5 else .0); FI.append(60000 + (i * 10 + k if i < 5 else 80 + (k in (0, 1, 2, 3))))
        V = np.array(VV); n = len(V)
        o, x, t, u = self.head_frame(p)
        R = np.stack([x, t, u])
        return dict(V=V @ R + o, N=np.array(NN) @ R, fid=np.array(FI), thin=np.zeros(n, np.int8), part=np.full(n, 7, np.int8), art=np.full(n, -1), two=np.array(LO, f32))

    def draw_head_marks(self, layer, ox, oy, p, cam, M=None, lw=1.0, seed=0, lit=False):
        """calm ink eye, gape line, nostril: projected drawing on top of the cel"""
        from .ink import Cel, spl
        R = body_R(p); X = np.asarray(p['pos'], float)
        o, x, t, u = self.head_frame(p)
        R_ = R
        pj = lambda P: (lambda q: q if M is None else q @ np.asarray(M)[:, :2].T + np.asarray(M)[:, 2])(cam.project(np.asarray(P) @ R_.T + X)[0])
        cl = [pj(C) for C, _, _ in self.crest_lines(p)]
        sc0 = float(np.hypot(*(cl[0][3] - cl[0][0]))) / .9
        if sc0 > 5 and cam.project((self.head_pt(p, np.array([0, 0, 0.]))[None] @ R.T + X))[1][0] > .5:
            al = np.vstack(cl); x0, y0 = np.floor(al.min(0) - 10).astype(int); x1, y1 = np.ceil(al.max(0) + 10).astype(int); x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
            if x1 - x0 > 6 and y1 - y0 > 6:
                cel = Cel(x1 - x0, y1 - y0, x0, y0, ss=3); kk = float(np.clip(sc0 / 40., .5, 1.5)) * lw
                for i, q in enumerate(cl):
                    cel.line(spl(q[:9]), 1.5 * kk, '#8a5a48' if not lit else '#94644c', (.05, .75), seed=seed + 40 + i, alpha=.8)
                f = cel.rgba(); lh, lw_ = layer.shape[:2]
                ax0, ay0 = max(x0, ox), max(y0, oy); ax1, ay1 = min(x1, ox + lw_), min(y1, oy + lh)
                if ax1 > ax0 and ay1 > ay0:
                    s_ = f[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]; b_ = layer[ay0 - oy:ay1 - oy, ax0 - ox:ax1 - ox]
                    b_[:] = b_ * (1 - s_[..., 3:4]) + s_
        for sd in (1, -1):
            c = self.head_pt(p, np.array([.32 * sd, .36, .12])) @ R.T + X
            nrm = (R @ (x * sd))
            v = cam.eye - c; v = v / np.linalg.norm(v)
            if nrm @ v < .10: continue
            def pr(P):
                q, z = cam.project(self.head_pt(p, np.array([[a * sd, b, d] for a, b, d in P])) @ R.T + X)
                return q if M is None else q @ np.asarray(M)[:, :2].T + np.asarray(M)[:, 2]
            e = pr([(.335, .16, .10), (.335, .60, .05)]); sc = float(np.hypot(*(e[1] - e[0]))) / .44      # px per unit
            if sc < 9: continue
            x0, y0 = np.floor(e.min(0) - sc * 3.2 - 8).astype(int); x1, y1 = np.ceil(e.max(0) + sc * 3.2 + 8).astype(int)
            x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
            if x1 - x0 < 6 or y1 - y0 < 6: continue
            cel = Cel(x1 - x0, y1 - y0, x0, y0, ss=3)
            ink = '#34213f'
            k = float(np.clip(sc / 45., .35, 1.6)) * lw
            # gape, nostril
            cel.line(spl(pr([(.17, .72, -.10), (.12, 1.15, -.12), (.07, 2.0, -.215), (.010, 3.1, -.385)])), 1.4 * k, '#54304a', (.05, .5), seed=seed + 1)
            cel.line(spl(pr([(.125, 1.16, -.03), (.11, 1.30, -.05), (.095, 1.42, -.08)])), 1.2 * k, ink, (.2, .4), seed=seed + 2)
            # eye: a long almond, half lidded, looking ahead and a little down
            white = spl(pr([(.300, .16, .085), (.302, .28, .145), (.296, .42, .135), (.272, .60, .045), (.292, .44, .03), (.302, .28, .045)]), closed=True, corners=(0, 3))
            cel.fill(white, '#f3dcc0')
            iris = spl(pr([(.304, .30, .14), (.303, .40, .138), (.296, .455, .085), (.297, .40, .036), (.304, .31, .045), (.306, .275, .09)]), closed=True)
            cel.shade(iris, ink, within=white)
            cel.fill(spl(pr([(.306, .375, .118), (.305, .40, .122), (.304, .405, .10), (.305, .38, .098)]), closed=True), '#fff0d4')
            cel.line(spl(pr([(.28, -.26, .13), (.298, .02, .10), (.306, .18, .105), (.306, .29, .155), (.298, .43, .143), (.270, .62, .035)])), 2.3 * k, ink, (.55, .25), seed=seed + 3)
            cel.line(spl(pr([(.300, .20, .065), (.302, .36, .02), (.286, .52, .03)])), 1.1 * k, '#6b4a66', (.4, .5), seed=seed + 4)
            cel.line(spl(pr([(.29, .10, .21), (.29, .30, .245), (.27, .52, .20)])), 1.0 * k, '#6b4a66', (.4, .5), seed=seed + 5)
            f = cel.rgba()
            # clip to the bird's own alpha
            lh, lw_ = layer.shape[:2]
            ax0, ay0 = max(x0, ox), max(y0, oy); ax1, ay1 = min(x1, ox + lw_), min(y1, oy + lh)
            if ax1 <= ax0 or ay1 <= ay0: continue
            s = f[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]; b = layer[ay0 - oy:ay1 - oy, ax0 - ox:ax1 - ox]
            s = s * b[..., 3:4]
            b[..., :3] = b[..., :3] * (1 - s[..., 3:4]) + s[..., :3]
        return layer

    # ---- everything in world space
    def build(self, p, curves=None, flutter=0.0):
        parts = [self.wing_pages(p, 1, 0), self.wing_pages(p, -1, 1), self.body_pages(p), self.tail_pages(p), self.head_pages(p)]
        R = body_R(p); X = np.asarray(p['pos'], float)
        for q in parts:
            q['V'] = q['V'] @ R.T + X; q['N'] = q['N'] @ R.T
            if 'NV' in q: q['NV'] = q['NV'] @ R.T
        if curves is None: curves = self.default_curves(p)
        parts.append(self.ribbon_pages(curves, flutter))
        cr = self.crest(p, R, X)
        parts.append(cr)
        return join(parts)

    def crest_lines(self, p):
        """centrelines of the crest plumes in bird space: rise out of the crown, sweep back"""
        o, x, t, u = self.head_frame(p); sw = p['ruffle'] * .5
        out = []
        for k, (dx, L, up, dr, y0) in enumerate(((0, 3.4, .86, 1.05, .30), (.10, 2.8, .62, 1.1, .20), (-.10, 3.0, .72, 1.15, .22), (.18, 2.1, .40, .95, .08), (-.18, 2.3, .46, 1.0, .10))):
            q = np.linspace(0, 1, 13)
            z = .345 + up * (np.sin(np.minimum(q * 1.9, 3.1)) - .10 * np.exp(-q * 14)) * np.clip(q * 5, 0, 1) ** .7 - dr * q ** 2.2 - sw * q ** 2 + .04 * np.sin(q * 7 + k * 2) * q
            C = o + x * (dx * (.75 + 2.1 * q))[:, None] + t * (y0 - q * L * .95 + .05 * np.sin(q * 5 + k) * q)[:, None] + u * z[:, None]
            out.append((C, x, u))
        return out

    def crest(self, p, R, X):
        """long plumes swept back from the crown, each with a solid quill and a small folded leaf"""
        VV, NN, FF, PT = [], [], [], []
        for k, (C, x, u) in enumerate(self.crest_lines(p)):
            q = np.linspace(0, 1, 13)
            wd = np.interp(q, [0, .10, .35, .62, .76, .88, 1], [.12, .115, .085, .06, .075, .17, .01])
            for j in range(6):
                i = j * 2
                V = np.array([[C[i + b] + x * (a - 1) * wd[i + b] + u * (.45 * abs(a - 1) * wd[i + b]) for b in range(3)] for a in range(3)])
                VV.append(V); n = np.cross(x, C[i + 2] - C[i]); NN.append(n / np.linalg.norm(n)); FF.append(61000 + k); PT.append(8)
        # a small collar of folded leaves holds them to the skull
        o, x, t, u = self.head_frame(p)
        for k, (dx, y, L_, w_, z0) in enumerate(((0, .46, .62, .13, .30), (.13, .36, .56, .12, .27), (-.13, .36, .56, .12, .27), (.0, .26, .70, .14, .33), (.22, .20, .50, .11, .20), (-.22, .20, .50, .11, .20))):
            b = np.linspace(0, 1, 3)
            V = np.array([[o + x * (dx * (1 + .3 * bb) + (a - 1) * w_ * (1 - .55 * bb ** 2)) + t * (y - L_ * bb) + u * (z0 + .16 * bb - .02 * abs(a - 1)) for bb in b] for a in range(3)])
            VV.append(V); NN.append(u * .9 - t * .2); FF.append(61020 + k); PT.append(7)
        V = np.array(VV) @ R.T + X; n = len(V); pt = np.array(PT, np.int8)
        return dict(V=V, N=_nrm(np.array(NN)) @ R.T, fid=np.array(FF), thin=np.where(pt == 8, 1, 0).astype(np.int8), part=pt, art=np.full(n, -1), two=np.full(n, .35, f32))

    def rib_roots(self, p):
        R = body_R(p); X = np.asarray(p['pos'], float)
        return np.array([r['root'] for r in self.rib]) @ R.T + X, R

    def default_curves(self, p, droop=.10, f=0.0):
        roots, R = self.rib_roots(p)
        out = []
        for k, (o, rb) in enumerate(zip(roots, self.rib)):
            s = np.linspace(0, 1, 40)
            d = R @ np.array([(k - 4) * .12, -1., .03 * (k % 3 - 1)])
            side = R @ np.array([1., 0, 0])
            P = o + d * (s * rb['L'])[:, None]
            P = P + np.array([0, 0, -1.]) * (droop * rb['L'] * s ** 2)[:, None]
            P = P + side * (.085 * rb['L'] * s ** 1.2 * np.sin(s * 4.2 + k * .9 - f))[:, None] + np.array([0, 0, 1.]) * (.05 * rb['L'] * s * np.sin(s * 3.6 + k * 1.1 - f * 1.1))[:, None]
            out.append((P, side))
        return out


# ------------------------------------------------------------------ rasteriser
@njit(cache=True, fastmath=True, parallel=True)
def _raster(S, PID, THIN, FIDP, Zi, ID, UU, VV, CNT, LASTF, nb):
    """S[n, r, c, 3] = x, y, 1/z in buffer px.  grid u along axis 1, v along axis 2"""
    h, w = Zi.shape
    n, gr, gc = S.shape[0], S.shape[1], S.shape[2]
    bh = (h + nb - 1) // nb
    for b in prange(nb):
        y_lo = b * bh; y_hi = min(h, y_lo + bh) - 1
        for i in range(n):
            mn = 1e9; mx = -1e9; ok = True
            for a in range(gr):
                for c in range(gc):
                    if S[i, a, c, 2] <= 0: ok = False
                    if S[i, a, c, 1] < mn: mn = S[i, a, c, 1]
                    if S[i, a, c, 1] > mx: mx = S[i, a, c, 1]
            if not ok or mx < y_lo or mn > y_hi: continue
            for a in range(gr - 1):
                for c in range(gc - 1):
                    for k in range(2):
                        if k == 0:
                            a0, c0, a1, c1, a2, c2 = a, c, a + 1, c, a + 1, c + 1
                        else:
                            a0, c0, a1, c1, a2, c2 = a, c, a + 1, c + 1, a, c + 1
                        x0 = S[i, a0, c0, 0]; y0 = S[i, a0, c0, 1]; z0 = S[i, a0, c0, 2]
                        x1 = S[i, a1, c1, 0]; y1 = S[i, a1, c1, 1]; z1 = S[i, a1, c1, 2]
                        x2 = S[i, a2, c2, 0]; y2 = S[i, a2, c2, 1]; z2 = S[i, a2, c2, 2]
                        ar = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
                        if abs(ar) < 1e-6: continue
                        ix0 = max(0, int(math.floor(min(x0, min(x1, x2))))); ix1 = min(w - 1, int(math.ceil(max(x0, max(x1, x2)))))
                        iy0 = max(y_lo, int(math.floor(min(y0, min(y1, y2))))); iy1 = min(y_hi, int(math.ceil(max(y0, max(y1, y2)))))
                        if ix0 > ix1 or iy0 > iy1: continue
                        ia = 1.0 / ar
                        u0 = a0 / (gr - 1); u1 = a1 / (gr - 1); u2 = a2 / (gr - 1)
                        v0 = c0 / (gc - 1); v1 = c1 / (gc - 1); v2 = c2 / (gc - 1)
                        for y in range(iy0, iy1 + 1):
                            for x in range(ix0, ix1 + 1):
                                px = x + .5; py = y + .5
                                w1 = ((px - x0) * (y2 - y0) - (x2 - x0) * (py - y0)) * ia
                                w2 = ((x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)) * ia
                                w0 = 1 - w1 - w2
                                if w0 < -1e-4 or w1 < -1e-4 or w2 < -1e-4: continue
                                z = w0 * z0 + w1 * z1 + w2 * z2
                                if THIN[i] == 1:
                                    if LASTF[y, x] != FIDP[i] and CNT[y, x] < 200:
                                        CNT[y, x] += 1; LASTF[y, x] = FIDP[i]
                                elif THIN[i] == 0 and CNT[y, x] < 200: CNT[y, x] += 3
                                if z > Zi[y, x]:
                                    Zi[y, x] = z; ID[y, x] = PID[i]
                                    UU[y, x] = int(min(1.0, max(0.0, w0 * u0 + w1 * u1 + w2 * u2)) * 65535)
                                    VV[y, x] = int(min(1.0, max(0.0, w0 * v0 + w1 * v1 + w2 * v2)) * 65535)


@njit(cache=True, fastmath=True, parallel=True)
def _shade(OUT, Zi, ID, UU, VV, CNT, C0, CC, CT1, CT2, CINK, CFL, ARTA, LINA, LW, WPX, LPX, FID, ART, FLIP, THIN, NLG, THR, BDX, BDY, ROFF, RH, RW, ADATA, LV, AW, AH, SEED,
           lx, ly, dsh, ssh, rx_, ry_, rimw, ocol, rimcol, rimon, ow, sw, dz, CURL, LIP):
    h, w = Zi.shape
    for y in prange(h):
        for x in range(w):
            i = ID[y, x]
            if i < 0: continue
            zi = Zi[y, x]
            z = 1.0 / zi
            u = UU[y, x] / 65535.0; v = VV[y, x] / 65535.0
            thin = THIN[i] == 1 or THIN[i] == 3
            # mass light at this pixel (bilinear over the page's own 3x3 normals), then flat tones
            gu = min(1.9999, u * 2); gv = min(1.9999, v * 2)
            ia_ = int(gu); ib_ = int(gv); fa = gu - ia_; fb = gv - ib_
            nl = (NLG[i, ia_, ib_] * (1 - fa) + NLG[i, ia_ + 1, ib_] * fa) * (1 - fb) + (NLG[i, ia_, ib_ + 1] * (1 - fa) + NLG[i, ia_ + 1, ib_ + 1] * fa) * fb
            nl += (_vn(x * .012, y * .012, 41) - .5) * .10
            if nl > THR[i, 0]: tone = 0
            elif nl > THR[i, 1]: tone = 1
            elif nl > THR[i, 2]: tone = 2
            elif nl > THR[i, 3]: tone = 3
            else: tone = 4
            r = C0[i, tone, 0]; g = C0[i, tone, 1]; b = C0[i, tone, 2]
            tl = 0 if tone < 3 else 1
            # transmitted light through single sheets
            if thin and tone >= 3:
                c = CNT[y, x]
                if c <= 1: r = CT1[i, 0]; g = CT1[i, 1]; b = CT1[i, 2]; tl = 0
                elif c == 2: r = CT2[i, 0]; g = CT2[i, 1]; b = CT2[i, 2]; tl = 0
            # dog-eared corner
            cu = CURL[i]
            fold = 0.0
            if cu > 0.0:
                q = (u - (1 - cu)) / cu + (v - (1 - cu)) / cu
                if q > 1.0:
                    r = CC[i, tone, 0]; g = CC[i, tone, 1]; b = CC[i, tone, 2]
                fold = abs(q - 1.0) * cu * min(WPX[i], LPX[i]) * .7
            # cast shadow from a page nearer the sun (hard, designed)
            cast = False
            if tone <= 2:
                for k in range(2):
                    dd = dsh[k] * (ssh[k] if ssh[k] > 0 else 1.0)
                    if k == 0: dd = min(dsh[0], 0.26 * min(WPX[i], LPX[i]) + 1.5)
                    xx = int(x + lx * dd); yy = int(y + ly * dd)
                    if xx < 0 or yy < 0 or xx >= w or yy >= h: continue
                    j = ID[yy, xx]
                    if j < 0 or j == i: continue
                    zz = 1.0 / Zi[yy, xx]
                    df = z - zz
                    if df > dz[k, 0] and df < dz[k, 1]:
                        cast = True
                if cast:
                    r = CC[i, tone, 0]; g = CC[i, tone, 1]; b = CC[i, tone, 2]
            elif not thin or CNT[y, x] > 2:
                # in shade: the overlap still reads as a darker step and a pale lip of paper
                dd = min(dsh[0] * .8, 0.16 * LPX[i] + 1.0)
                xx = int(x + BDX[i] * dd); yy = int(y + BDY[i] * dd)
                if 0 <= xx < w and 0 <= yy < h:
                    j = ID[yy, xx]
                    if j >= 0 and j != i and FID[j] != FID[i]:
                        df = z - 1.0 / Zi[yy, xx]
                        if df > dz[0, 0] * .5 and df < dz[0, 1]:
                            r = CC[i, tone, 0]; g = CC[i, tone, 1]; b = CC[i, tone, 2]
                if v > 1 - 2.2 / max(LPX[i], 8.0) * LIP and v < 1 - .9 / max(LPX[i], 8.0) * LIP and LPX[i] > 14:
                    r += (CC[i, 5, 0] - r) * .55; g += (CC[i, 5, 1] - g) * .55; b += (CC[i, 5, 2] - b) * .55
            # page art
            ai = ART[i]
            if ai >= 0 and ARTA[i] > 0.01:
                uu = u; vv = v
                fl = FLIP[i]
                if fl & 1: uu = 1 - uu
                if fl & 2: vv = 1 - vv
                if fl & 4:          # engravings stand upright when the wing is raised: their top is toward the wing tip
                    tt_ = uu; uu = vv; vv = 1 - tt_
                l = LV[i]
                tw = AW[l]; th = AH[l]
                fx = uu * tw - .5; fy = vv * th - .5
                x0 = int(math.floor(fx)); y0 = int(math.floor(fy))
                fx -= x0; fy -= y0
                a0 = 0.0; a1 = 0.0
                for dy in range(2):
                    for dx in range(2):
                        xi = min(tw - 1, max(0, x0 + dx)); yi = min(th - 1, max(0, y0 + dy))
                        wt_ = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
                        o = ROFF[l] + ((ai * th + yi) * tw + xi) * 2
                        a0 += wt_ * ADATA[o]; a1 += wt_ * ADATA[o + 1]
                a0 = min(1.0, a0 / 255.0 * 1.55) * ARTA[i]; a1 = a1 / 255.0 * .42
                ir = CINK[i, tl, 0]; ig = CINK[i, tl, 1]; ib = CINK[i, tl, 2]
                k = a1 * .5
                r += (ir - r) * k * .55; g += (ig - g) * k * .62; b += (ib - b) * k * .75
                r += (ir - r) * a0; g += (ig - g) * a0; b += (ib - b) * a0
            # page edge line (drawn inside the page that owns it)
            la = LINA[i]
            if la > 0.02:
                e1 = u * WPX[i]; e2 = (1 - u) * WPX[i]; e3 = (1 - v) * LPX[i]
                e = e1 if THIN[i] != 1 else 99.0; sdx = v * LPX[i]
                if e2 < e: e = e2; sdx = v * LPX[i] + 300
                if e3 < e: e = e3; sdx = u * WPX[i] + 600
                if cu > 0 and q > 1.0: e = 99.0
                if cu > 0 and fold < e: e = fold
                nz = _vn(sdx * .035, i * 1.7, SEED)
                wl = LW[i] * (.45 + 1.0 * nz)
                if nz < .16: wl *= .25
                aa = (wl - e) + .5
                if aa > 0:
                    if aa > 1: aa = 1.0
                    aa *= la * (.62 if tl == 1 else 1.0)
                    r += (CINK[i, tl, 0] - r) * aa; g += (CINK[i, tl, 1] - g) * aa; b += (CINK[i, tl, 2] - b) * aa
            # silhouettes and feather edges
            nz = _vn(x * .021, y * .021, SEED + 7)
            R1 = sw * (.45 + .8 * nz); R2 = ow[min(5, int(LINA[i] * 5.99))] * (.45 + .8 * nz)
            sil = False; fe = False
            fi = FID[i]
            for k in range(8):
                dxk = (1., -1., 0., 0., .7, -.7, .7, -.7)[k]; dyk = (0., 0., 1., -1., .7, .7, -.7, -.7)[k]
                xx = int(x + .5 + dxk * R1); yy = int(y + .5 + dyk * R1)
                if 0 <= xx < w and 0 <= yy < h:
                    j = ID[yy, xx]
                    if j < 0 or (1.0 / Zi[yy, xx] - z) > dz[2, 0]:
                        sil = True
                xx = int(x + .5 + dxk * R2); yy = int(y + .5 + dyk * R2)
                if 0 <= xx < w and 0 <= yy < h:
                    j = ID[yy, xx]
                    if j >= 0 and FID[j] != fi and Zi[yy, xx] < zi: fe = True
            if fe and la > .05:
                k = .75 * min(1.0, max(0.0, la * 1.9 - .12))
                r += (CFL[i, tl, 0] - r) * k; g += (CFL[i, tl, 1] - g) * k; b += (CFL[i, tl, 2] - b) * k
            # hot rim on the sun side
            rim = False
            if rimon:
                rr = rimw * (.35 + 1.1 * _vn(x * .016 + 9, y * .016, SEED + 3))
                xx = int(x + rx_ * rr); yy = int(y + ry_ * rr)
                if 0 <= xx < w and 0 <= yy < h:
                    j = ID[yy, xx]
                    if j < 0 or (1.0 / Zi[yy, xx] - z) > dz[2, 0]: rim = True
                    elif thin and FID[j] != fi and Zi[yy, xx] < zi * .9995 and tone >= 3 and rr > 2.0:
                        # feather edges catch light too, thinner
                        xx = int(x + rx_ * rr * .45); yy = int(y + ry_ * rr * .45)
                        j = ID[yy, xx]
                        if j < 0 or (FID[j] != fi and Zi[yy, xx] < zi * .9995): rim = True
            if sil:
                k = .95
                oc = ocol[1] if rim else (ocol[2] if (thin and tone >= 3) else ocol[0])
                r += (oc[0] - r) * k; g += (oc[1] - g) * k; b += (oc[2] - b) * k
                if rim: r = rimcol[0] * .55 + r * .45; g = rimcol[1] * .55 + g * .45; b = rimcol[2] * .55 + b * .45
            elif rim:
                r = rimcol[0]; g = rimcol[1]; b = rimcol[2]
            OUT[y, x, 0] = r; OUT[y, x, 1] = g; OUT[y, x, 2] = b; OUT[y, x, 3] = 1.0


_AT = None


def _atlas():
    global _AT
    if _AT is None:
        m = gb_art.atlas(256, 343, levels=4) + gb_art.atlas(128, 172, gb_art.SMALL, 's', levels=4) + gb_art.atlas(256, 343, gb_art.BOLD, 'b', levels=4)
        off = []; o = 0
        for a in m: off.append(o); o += a.size
        _AT = (np.concatenate([a.ravel() for a in m]), np.array(off, np.int64), np.array([a.shape[2] for a in m], np.int64), np.array([a.shape[1] for a in m], np.int64), m[0].shape[0])
    return _AT


PAL = dict(
    hot='#fff3cf', lit='#fae7c0', half='#e0bd9c', shade='#9b8aaa', deep='#766a98',
    t1='#f9c684', t2='#dc9a80', bounce='#ad94ac',
    old='#e6bb7c', old2='#cd9b78', cast_lit='#b592a4', cast_half='#9d86a6', sh_step='#8676a3', dp_step='#665b8f', lip='#c3b2c4', ink_lit='#94644c', ink_sh='#51407a', fline='#74463e', fline_sh='#4c3a6c', haze='#f2dcb4', bold_sh='#3a2b5e')


def draw(scene, cam, M=None, L=(.55, -.35, .76), sun2d=(.8, -.5), pal=None, haze=(26, 60, .0), ss=2, seed=0, lw=1.0, boil=.0, art=1.0, rim=3.6, glow=1.0,
         roi_pad=8, eye_fn=None, cast=(3.4, 9.0), dzs=None, size=(W, H), outline=1.0, curl=.16, thr_lit=.22, jitn=.03, thr_hot=.95):
    """scene: dict from BirdModel.build.  M: 2x3 affine applied to projected px (2D stand camera).  -> premult RGBA f32 (H,W,4) layer roi, ox, oy"""
    P = dict(PAL); P.update(pal or {})
    rgb = lambda c: _rgb(c) if isinstance(c, str) else np.asarray(c, f32)
    V = scene['V']; n = len(V)
    XY, z = cam.project(V)
    if M is not None:
        XY = XY @ np.asarray(M)[:, :2].T + np.asarray(M)[:, 2]
    if boil > 0:
        r = np.random.default_rng(4000 + seed)
        wp = np.linalg.norm(XY[:, 2, 1] - XY[:, 0, 1], axis=-1)
        XY = XY + r.normal(0, 1, XY.shape) * np.minimum(boil, wp * .03)[:, None, None, None]
    WW, HH = size
    good = (z.min((1, 2)) > .2)
    good &= ~((scene['part'] == 9) & (cam.f / np.maximum(z[:, 1, 1], .2) * (np.hypot(M[0][0], M[1][0]) if M is not None else 1.0) < 41))      # too far off for shafts
    mn = XY.min((1, 2)); mx = XY.max((1, 2))
    good &= (mx[:, 0] > 0) & (mx[:, 1] > 0) & (mn[:, 0] < WW) & (mn[:, 1] < HH)
    if not good.any(): return np.zeros((2, 2, 4), f32), 0, 0
    x0, y0 = np.floor(np.maximum(mn[good].min(0) - roi_pad, 0)).astype(int); x1, y1 = np.ceil(np.minimum(mx[good].max(0) + roi_pad, [WW, HH])).astype(int)
    w, h = (x1 - x0) * ss, (y1 - y0) * ss
    if w < 4 or h < 4: return np.zeros((2, 2, 4), f32), 0, 0
    # per-page shading -------------------------------------------------
    ctr = V[:, 1, 1]
    N = scene['N'].copy()
    tov = cam.eye - ctr; tov /= np.linalg.norm(tov, axis=1, keepdims=True)
    facing = (N * tov).sum(1)
    back = facing < 0
    Ls = np.asarray(L, float); Ls = Ls / np.linalg.norm(Ls)
    rk = lambda k: np.random.default_rng(7700 + k)          # one stream per attribute: a page keeps its looks when others come and go
    part = scene['part']
    sg = np.where(back, -1., 1.)
    NLG = (scene['NV'] @ Ls) * sg[:, None, None] + (rk(0).normal(0, 1, n) * np.where(part == 4, .035, np.where((part == 2) | (part == 1), jitn, .03)))[:, None, None]
    up = N[:, 2] * sg
    thr = np.tile(np.array([[thr_hot, thr_lit, min(-.02, thr_lit - .24) if thr_lit < .3 else thr_lit - .26, -2.]]), (n, 1))
    thr[rk(1).random(n) < .18, 0] = thr_hot - .15
    thr[(part == 3), 3] = -.40
    thr[(part == 4), 3] = -.66
    thr[(part == 7), 3] = -.60
    base = np.stack([rgb(P[k]) for k in ('hot', 'lit', 'half', 'shade', 'deep')])
    C0 = np.tile(base[None], (n, 1, 1)).astype(float)
    lo = scene['two'][:, None].astype(float)
    for k in (0, 1): C0[:, k] = C0[:, k] * (1 - lo) + rgb(P['old']) * lo
    C0[:, 2] = C0[:, 2] * (1 - lo * .5) + rgb(P['old2']) * lo * .5
    for k in (3, 4): C0[:, k] *= (1 - .10 * np.minimum(lo, 1))
    bn = (up < -.62) & (part != 1)
    C0[bn, 3] = rgb(P['bounce'])
    CC = np.tile(np.stack([rgb(P[k]) for k in ('cast_lit', 'cast_lit', 'cast_half', 'sh_step', 'dp_step', 'lip')])[None], (n, 1, 1)).astype(float)
    CK = np.tile(np.stack([rgb(P['ink_lit']), rgb(P['ink_sh'])])[None], (n, 1, 1)).astype(float)
    CF = np.tile(np.stack([rgb(P['fline']), rgb(P['fline_sh'])])[None], (n, 1, 1)).astype(float)
    ra = part == 9
    for k in range(5): C0[ra, k] = C0[ra, k] * .35 + CF[ra, 0 if k < 3 else 1] * .65
    T1 = np.tile(rgb(P['t1']), (n, 1)).astype(float); T2 = np.tile(rgb(P['t2']), (n, 1)).astype(float)
    lz = part == 10
    if pal and lz.any():          # loose sheets keep their own paper colours
        C0[lz] = np.stack([_rgb(PAL[k]) for k in ('hot', 'lit', 'half', 'shade', 'deep')])[None]
        CC[lz] = np.stack([_rgb(PAL[k]) for k in ('cast_lit', 'cast_lit', 'cast_half', 'sh_step', 'dp_step', 'lip')])[None]
    T1[lz] = _rgb(PAL['t1']) * .45 + _rgb(PAL['lit']) * .55; T2[lz] = T1[lz] * .6 + _rgb(PAL['shade']) * .4
    # individual pages: old paper varies
    jv = rk(2).normal(0, .024, n)[:, None]; jw = rk(3).normal(0, .014, n)[:, None] + .05 * (rk(4).random(n) < .10)[:, None]
    tint = np.array([[.4, -.1, -1.1]])
    for A in (T1, T2): A += jv + jw * tint
    for A in (C0, CC): A += (jv + jw * tint)[:, None]
    # distance: air eats contrast
    zc = z[:, 1, 1]
    hz = np.clip((zc - haze[0]) / max(haze[1] - haze[0], 1e-3), 0, 1)[:, None] * haze[2]
    if 'hx' in scene: hz = np.clip(hz + scene['hx'][:, None], 0, .85)
    hc = rgb(P['haze'])[None]
    for A in (T1, T2): A += (hc - A) * hz
    for A in (C0, CC, CK, CF): A += (hc[:, None] - A) * hz[:, None]
    wp = (np.linalg.norm(XY[:, 2, 1] - XY[:, 0, 1], axis=-1) * ss).astype(f32); lp = (np.linalg.norm(XY[:, 1, 2] - XY[:, 1, 0], axis=-1) * ss).astype(f32)
    sz = np.minimum(wp, lp)
    lina = (part != 9) * np.clip((sz - 5) / 34., 0, 1) ** .8 * (1 - .75 * hz[:, 0]) * .72 * lw
    arta = np.clip((sz - 8) / 46., 0, 1) ** .8 * .80 * art * (1 - .7 * hz[:, 0])
    LWa = np.clip(sz * .022, .7, 2.2) * ss * .5 * lw
    AD, OFF, AW, AH, NA = _atlas()
    lb = np.clip(np.log2(np.maximum(256. / np.maximum(wp, 1), 343. / np.maximum(lp, 1)) * .75), 0, 3).astype(np.int64)
    ls = 4 + np.clip(np.log2(np.maximum(128. / np.maximum(wp, 1), 172. / np.maximum(lp, 1)) * .8), 0, 3).astype(np.int64)
    lv = np.where(np.minimum(wp, lp * .75) < 130, ls, lb)
    a = scene['art']; isb = a >= gb_art.BOLD0
    ai = np.where(a < 0, -1, np.where(isb, a - gb_art.BOLD0, a % NA)).astype(np.int64); fl = np.where(isb, 4, (np.abs(a) // 64) % 4).astype(np.int64)
    ai = np.where(((np.abs(a) // 512) % 9 == 0) & ~isb, -1, ai)
    lv = np.where(isb, 8 + np.clip(np.log2(np.maximum(256. / np.maximum(lp, 1), 343. / np.maximum(wp, 1)) * .72), 0, 3).astype(np.int64), lv)
    arta = np.where(isb, np.clip((sz - 8) / 40., 0, 1) * .84 * (1 - .6 * hz[:, 0]), arta)
    CK[isb, 0] = rgb('#86573f'); CK[isb, 1] = rgb(P['bold_sh']); lina = np.where(isb, np.maximum(lina, .55), lina)
    cu = np.where((np.abs(a) // 4096) % 9 == 0, curl * (.6 + ((np.abs(a) // 99) % 10) / 10.), 0.) * (sz > 16) * (a >= 0) * (~isb)
    bd = XY[:, 1, 0] - XY[:, 1, 2]; bd = bd / (np.linalg.norm(bd, axis=1, keepdims=True) + 1e-6)
    S = np.empty(XY.shape[:3] + (3,), f32)
    S[..., 0] = (XY[..., 0] - x0) * ss; S[..., 1] = (XY[..., 1] - y0) * ss; S[..., 2] = np.where(good[:, None, None], 1. / np.maximum(z, .1), -1)
    Zi = np.zeros((h, w), f32); ID = np.full((h, w), -1, np.int32); UU = np.zeros((h, w), np.uint16); VV = np.zeros((h, w), np.uint16); CNT = np.zeros((h, w), np.uint8)
    idx = np.nonzero(good)[0]
    LF = np.full((h, w), -1, np.int32)
    _raster(S[idx], idx.astype(np.int32), scene['thin'][idx], scene['fid'][idx].astype(np.int32), Zi, ID, UU, VV, CNT, LF, 12)
    OUT = np.zeros((h, w, 4), f32)
    l2 = np.array([Ls @ cam.R[0], Ls @ cam.R[1]]); l2 = l2 / (np.hypot(*l2) + 1e-6)
    s2 = np.asarray(sun2d, float); s2 = s2 / np.hypot(*s2)
    D = float(np.median(zc[good]))
    dzz = np.array([[.004, .9], [.35, 6.], [.55, 0]]) if dzs is None else np.asarray(dzs, float)
    pxu = cam.f / D * (np.hypot(M[0][0], M[1][0]) if M is not None else 1.0)     # px per unit
    ksz = np.clip(pxu / 62., .12, 2.2)
    _shade(OUT, Zi, ID, UU, VV, CNT, C0.astype(f32), CC.astype(f32), T1.astype(f32), T2.astype(f32), CK.astype(f32), CF.astype(f32), arta.astype(f32), lina.astype(f32), LWa.astype(f32),
           wp, lp, scene['fid'].astype(np.int64), ai, fl, scene['thin'], NLG.astype(f32), thr.astype(f32), bd[:, 0].astype(f32), bd[:, 1].astype(f32), OFF, 0, 0, AD, lv, AW, AH, seed,
           float(l2[0]), float(l2[1]), np.array([cast[0] * ss * ksz * 2.2, cast[1] * ss * ksz]), np.array([1., 1.]), float(s2[0]), float(s2[1]), float(rim * ss * ksz ** .6),
           np.stack([rgb('#6b4250'), rgb('#d08a52'), rgb('#94563f')]).astype(np.float64), rgb(P['hot']).astype(np.float64), rim > 0,
           np.array([.5, .7, .9, 1.0, 1.1, 1.2]) * ss * .9 * ksz ** .4 * outline, float(1.35 * ss * ksz ** .5 * outline), dzz, cu.astype(f32), float(ss))
    out = cv2.resize(OUT, (w // ss, h // ss), interpolation=cv2.INTER_AREA) if ss > 1 else OUT
    return out, x0, y0
