"""Side view of the fox (faces right in model space).  Units: standing height to ear tip = 100, ground y = 0, up is negative.
Every pose is a dict; outlines are rebuilt from the pose as continuous contours (no rotated parts)."""
import numpy as np, math
from .ink import Cel, spl, resample, noise1
from .paint import rgb, mix
from .fox import PAL, fur

HS = 1.14
STAND = dict(hip=(-19, -38), sh=(16, -38.5), head=(30.5, -59), hang=.04, belly=0., arch=0., fn=(16.5, 0), ff=(21, 0), hn=(-22, 0), hf=(-14.5, 0),
             fnp=0., ffp=0., hnp=0., hfp=0., tail=((-.15, 13), (.25, 14), (.1, 15), (-.35, 14), (-.45, 10)), tb=3.35, earb=0., ears=0., blink=0., mouth=0., look=(0., 0.), wind=0., puff=0.)


def pose(base=STAND, **kw):
    p = dict(base); p.update(kw); return p


def lerp(a, b, t):
    out = {}
    for k in a:
        va, vb = a[k], b.get(k, a[k])
        if isinstance(va, (int, float)): out[k] = va + (vb - va) * t
        else: out[k] = tuple(map(tuple, (np.array(va, float) * (1 - t) + np.array(vb, float) * t))) if np.ndim(va) == 2 else tuple(np.array(va, float) * (1 - t) + np.array(vb, float) * t)
    return out


def ik2(a, b, l1, l2, bend):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a; L = np.hypot(*d); L2 = min(L, (l1 + l2) * .999)
    if L < 1e-6: L = 1e-6
    u = d / L
    x = (l1 * l1 - l2 * l2 + L2 * L2) / (2 * L2); y = math.sqrt(max(0, l1 * l1 - x * x))
    n = np.array([-u[1], u[0]]) * bend
    e = a + u * x + n * y
    if L > L2: b = a + u * L2
    return e, b


def edges(J, W, n=10, full=False):
    """variable width limb through joints J with half widths W -> (left edge, right edge)"""
    J = np.asarray(J, float)
    c = spl(J, n=n)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]; d /= d[-1]
    dj = np.r_[0, np.cumsum(np.hypot(*np.diff(J, axis=0).T))]; dj /= dj[-1]
    w = np.interp(d, dj, W)
    k = np.array([1, 2, 3, 2, 1]) / 9.
    w = np.convolve(np.pad(w, 2, 'edge'), k, 'valid')
    T = np.gradient(c, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9
    N = np.c_[T[:, 1], -T[:, 0]]
    if full: return c + N * w[:, None], c - N * w[:, None], c, w
    return c + N * w[:, None], c - N * w[:, None]


def runs(E, c, w, eps=1.25):
    """parts of an offset edge that are really on the outside of the limb"""
    d = np.sqrt(((E[:, None] - c[None]) ** 2).sum(-1)) - w[None]
    ok = d.min(1) > -eps
    out, cur = [], []
    for i, k in enumerate(ok):
        if k: cur.append(i)
        elif cur: out.append(cur); cur = []
    if cur: out.append(cur)
    return [r for r in out if len(r) >= 4]


def side(cel, x, y, h, P, pal='lantern', lw=2.0, seed=0, flip=False, key=1, rim=True, detail=1, carry=None, under=False):
    """draw the fox.  (x, y) ground origin px, h standing height px.  carry: callback(cel, mouth_xy_px, scale) drawn behind the near jaw.
    returns dict of useful screen points (mouth, eye, nose, paw_n)."""
    C = PAL[pal] if isinstance(pal, str) else pal
    r = np.random.default_rng(5000 + abs(seed))
    k = h / 100.
    sx = -k if flip else k
    wob = .16 if h < 300 else .10
    S = lambda q: np.asarray(q, float).reshape(-1, 2) * [sx, k] + [x, y]
    J = lambda q: np.asarray(q, float) + r.normal(0, wob, np.shape(q))
    L = C['line']
    ks = key * (-1 if flip else 1)          # light side in model space (+1 = from the right/front)
    hip, sh, hd = np.array(P['hip'], float), np.array(P['sh'], float), np.array(P['head'], float)
    wv = P['wind']
    tip_l = np.array([0., 0.])

    def line(q, w=1.0, taper=(.2, .2), al=1.0, col=None, sd=0, press=.28):
        cel.line(S(q), lw * w, col or L, taper, press, seed=seed * 13 + sd, alpha=al)

    # ------------------------------------------------ limbs
    def paw(foot, ang, sc=1.0):
        ca, sa = math.cos(ang), math.sin(ang)
        q = np.array([[-3.2, -3.6], [1.5, -4.6], [5.6, -3.4], [7.2, -1.4], [6.4, .2], [2, .4], [-2.6, .3], [-3.6, -1.4]], float) * sc
        q = q + r.normal(0, wob * .6, q.shape)
        return np.c_[q[:, 0] * ca - q[:, 1] * sa, q[:, 0] * sa + q[:, 1] * ca] + foot

    def front(foot, pa, near):
        j = sh + [0.5, 6] + ([0, 0] if near else [1.5, 1])
        j = j + [(foot[0] - j[0]) * .38, 0]
        wrist = np.array(foot, float) + [-.6 * math.cos(pa), -4.2]
        if pa: wrist = np.array(foot, float) + np.array([-2.2 * math.sin(abs(pa)) * 1.4, -4.2 * math.cos(pa)])
        e, wr = ik2(j, wrist, 13.2, 18.2, 1)
        ft = wr + (np.array(foot, float) - wrist)
        lo = wr + (e - wr) * .12
        A, B, cc, ww = edges(J([j + [0, -6], j, e, (e + wr) / 2, wr, ft + [0, -1.5]]), [8.5, 7.6, 4.9, 3.9, 3.1, 3.2], full=True)
        return A, B, paw(ft, pa), e, wr, ft, cc, ww

    def hind(foot, pa, near):
        j = hip + [0, 4] + ([0, 0] if near else [3, .5])
        j = j + [(foot[0] + 3 - j[0]) * .2, 0]
        ft = np.array(foot, float)
        hock = ft + np.array([-5.0 - 3.5 * math.sin(pa), -13.2 * math.cos(pa * .55)])
        kn, hk = ik2(j, hock, 16.5, 15.0, -1)
        ft = ft + (hk - hock)
        A, B, cc, ww = edges(J([j + [-1, -7], j + [0, -1], kn * .55 + j * .45, kn * .9 + hk * .1, hk, (hk + ft) / 2, ft + [0, -1.6]]), [10.5, 11.8, 9.6, 5.8, 3.3, 2.9, 3.0], full=True)
        return A, B, paw(ft, pa * .6), kn, hk, ft, cc, ww

    def sock(A, B, pw, frac, near):
        n = len(A); i = int(n * frac)
        dk, dks = (C['drk'], C['drk_sh'])
        top = np.vstack([A[i:i + 1], (A[i:i + 1] + B[i:i + 1]) / 2 + [0, 1.5], B[i:i + 1]])
        ft = fur(S(spl(top)), 3.2 * k, 2.6 * k, .2, seed + i, side=1 if not flip else -1, taper=False) if h > 160 else S(top)
        poly = np.vstack([ft, S(B[i:]), S(A[i:])[::-1]])
        cel.fill(poly, dk if near else dks)
        cel.fill(S(spl(pw, closed=True)), dk if near else dks)
        return i

    def leg(L_, near, frac, openfrac, oA=None, oB=None):
        A, B, pw, *_ = L_
        body = np.vstack([A, B[::-1]])
        cel.fill(S(body), C['fur'] if near else C['fur_sh'])
        if near:
            # shadow along the trailing edge
            m = (A + B) / 2
            E, Q = (A, B) if ks > 0 else (B, A)
            tt_ = np.linspace(0, 1, len(A))[:, None]
            wq = np.clip((tt_ - .30) / .30, 0, 1) ** .8 * .62
            i0 = int(len(A) * .30)
            q = np.vstack([E[i0:], (E * (1 - wq) + m * wq)[i0:][::-1]])
            cel.shade(S(q), C['fur_sh'], within=S(body))
        i = sock(A, B, pw, frac, near)
        nA = int(len(A) * (openfrac if oA is None else oA)); nB = int(len(B) * (openfrac if oB is None else oB))
        for E_, n_, sd_, w_ in ((A, nA, 3, .95), (B, nB, 4, 1.0)):
            for rr in runs(E_, L_[6], L_[7]):
                q_ = [i_ for i_ in rr if i_ >= n_]
                if len(q_) < 4: continue
                line(E_[q_], w_, (.35 if q_[0] <= n_ + 1 else .12, .05 if q_[-1] >= len(E_) - 2 else .12), sd=sd_)
        p = spl(pw, closed=True)
        line(np.vstack([p[int(len(p) * .05):], ]), .95, (.1, .25), sd=5)
        if h > 180:
            for t in (3.6, 5.2):
                q = pw[6] * 0 + L_[5] + np.array([[t, -2.6], [t + .6, -.6]])
                line(q, .6, (.3, .3), .7, sd=6)

    Fn, Ff = front(P['fn'], P['fnp'], True), front(P['ff'], P['ffp'], False)
    Hn, Hf = hind(P['hn'], P['hnp'], True), hind(P['hf'], P['hfp'], False)

    # ------------------------------------------------ tail
    def tail():
        a = P['tb']; p = hip + [-10.0, -4.0]; pts = [hip + [1.5, -.5], hip + [-4.5, -2.2], p]
        for (da, ln) in P['tail']:
            a += da; p = p + np.array([math.cos(a), math.sin(a)]) * ln; pts.append(p)
        pf = 1 + P['puff']
        W = np.interp(np.linspace(0, 1, len(pts)), [0, .25, .50, .72, .88, 1], np.array([3.6, 5.0, 9.2, 10.2, 6.4, .4]) * np.r_[1, 1, pf, pf, pf, pf])
        A, B = edges(J(pts), W, 10)
        tipv = pts[-1] + (pts[-1] - pts[-2]) / (np.hypot(*(pts[-1] - pts[-2])) + 1e-6) * 4
        out = np.vstack([A, tipv[None], B[::-1]])
        o = S(out)
        n = len(A)
        # furry silhouettes: cut locks into the outer halves of both edges
        sd1 = 1 if not flip else -1
        c0, c1 = int(n * .35), int(n * .90)
        Af = fur(S(A[c0:c1]), 1.9 * k, 6.4 * k, 1.2, 41, side=sd1)
        Bf = fur(S(B[c0:c1]), 1.9 * k, 6.4 * k, 1.2, 43, side=-sd1)
        Af = np.vstack([S(A[:c0]), Af, S(A[c1:])]); Bf = np.vstack([S(B[:c0]), Bf, S(B[c1:])])
        shape = np.vstack([Af, S(tipv), Bf[::-1]])
        cel.fill(shape, C['fur'])
        cel.clip(shape)
        # shadow: lower side
        low = (B if B[n // 2, 1] > A[n // 2, 1] else A)
        mid = (A + B) / 2
        q = np.vstack([mid * .75 + low * .25, (low * 2.2 - mid * 1.2)[::-1]])
        cel.fill(fur(S(q), 2.2 * k, 9 * k, .8, 7 + 73), C['fur_sh'])
        # white tip
        i = int(n * .70)
        e = np.vstack([A[i - 2:i - 1] * 1.6 - mid[i:i + 1] * .6, A[i:i + 1], mid[i + 4 if i + 4 < n else n - 1][None] + 0, B[i:i + 1], B[i - 2:i - 1] * 1.6 - mid[i:i + 1] * .6])
        ee = fur(S(spl(e)), 6.5 * k, 6.5 * k, -.2, 7 + 74, side=-sd1, taper=False)
        end = S(np.vstack([B[i:] * 1.8 - mid[i:] * .8, tipv[None] * 1.0 + (tipv - mid[i]) * .6, (A[i:] * 1.8 - mid[i:] * .8)[::-1]]))
        cel.fill(np.vstack([ee, end]), C['wht'])
        q2 = np.vstack([mid[i - 2:] * .8 + low[i - 2:] * .2, (low[i - 2:] * 2.2 - mid[i - 2:] * 1.2)[::-1]])
        cel.clip(np.vstack([ee, end])); cel.fill(S(q2), C['wht_sh']); cel.unclip()
        cel.unclip()
        m1 = len(Af); cel.line(Af[:], lw * 1.0, L, (.3, .06), .3, seed=seed + 75)
        cel.line(Bf[:], lw * 1.0, L, (.3, .06), .3, seed=seed + 76)
        cel.line(np.vstack([Af[-1:], S(tipv)]), lw * .9, L, (.1, .3), seed=seed + 77); cel.line(np.vstack([Bf[-1:], S(tipv)]), lw * .9, L, (.1, .3), seed=seed + 78)
        if detail:
            for t0 in (.45, .6):
                i0 = int(n * t0)
                q = np.array([mid[i0] * .8 + A[i0] * .2, mid[i0 + 3] * .9 + A[i0 + 3] * .1 + [0, -.5], mid[min(n - 1, i0 + 6)]])
                line(spl(q), .55, (.5, .5), .5, sd=79 + i0)
        return shape

    # ------------------------------------------------ torso + neck contour
    ha = P['hang']; ch, shh = math.cos(ha), math.sin(ha)
    HX = lambda q: (np.asarray(q, float).reshape(-1, 2) * HS) @ np.array([[ch, shh], [-shh, ch]]) + hd
    mid = (hip + sh) / 2
    ar, be = P['arch'], P['belly']
    nape = HX([[-11.5, 2]])[0]; throat = HX([[2, 13]])[0]
    bd = sh - hip; bl = np.hypot(*bd); u = bd / bl; nrm = np.array([u[1], -u[0]])     # nrm points up
    if nrm[1] > 0: nrm = -nrm
    B_ = lambda a, b: hip + u * a * bl + nrm * b
    neck_mid = nape * .52 + B_(.80, 11.5) * .48 + np.array([-1.2, -.6])
    top = [nape, neck_mid, B_(.80, 10.6), B_(.50, 9.2 + ar), B_(.22, 9.8 + ar), B_(-.02, 9.6), B_(-.22, 6.0), B_(-.31, -.5)]
    bot = [B_(-.27, -7), B_(-.05, -10.5), B_(.22, -8.2 - be), B_(.52, -11.0 - be), B_(.82, -15.5 - be * .5), B_(1.10, -14.5), B_(1.32, -5.5), (throat * .45 + B_(1.36, 3) * .55), throat]
    tt = spl(J(top), n=8); bb = spl(J(bot), n=8)
    torso = np.vstack([tt, bb])

    tail_first = True
    tail()
    leg(Hf, False, .60, .40); leg(Ff, False, .56, .45)

    Ts = S(torso)
    cel.fill(Ts, C['fur'])
    # belly / chest white bib with fur cut
    cel.clip(Ts)
    bib = np.array([HX([[-7, 14]])[0], (HX([[-5, 15]])[0] + B_(1.04, 3)) / 2, B_(1.05, 0), B_(1.0, -6), B_(.92, -11), B_(.78, -12.6), B_(.52, -9.4 - be), B_(.22, -7.0 - be), B_(.02, -9)])
    be_ = fur(S(spl(bib)), 3.4 * k, 7.5 * k, .7, 7 + 81, side=(1 if not flip else -1))
    bibp = np.vstack([be_, S(np.array([B_(0, -40), B_(1.2, -40), B_(1.8, -10), throat + [14, 0]]))])
    cel.fill(bibp, C['wht'])
    # cel shadow: underside of the body, cut as fur
    if under:
        se = [B_(-.45, 3), B_(-.1, 2.5), B_(.3, 3.5), B_(.7, 3), B_(1.0, 2), B_(1.15, 6), neck_mid + nrm * -7]
        cap = [neck_mid + [0, -20], B_(.5, 30), B_(-.6, 25)]
    else:
        se = [B_(-.45, -1.5), B_(-.1, -2.6), B_(.25, -2.6), B_(.6, -4.2), B_(.86, -7.2), B_(1.04, -10), B_(1.16, -18)]
        cap = [B_(1.1, -60), B_(-.6, -60), B_(-.6, -3)]
    sfe = fur(S(spl(np.array(se))), 2.4 * k, 8 * k, .8, 7 + 83, side=(-1 if not flip else 1) if not under else (1 if not flip else -1))
    shp = np.vstack([sfe, S(np.array(cap))])
    cel.clip(shp); cel.fill(Ts, C['fur_sh']); cel.fill(bibp, C['wht_sh']); cel.unclip()
    # neck shadow under the head
    q = HX([[-12, 6], [-4, 16], [6, 15], [10, 22], [-4, 25], [-14, 16]])
    cel.fill(S(spl(q, closed=True)), C['fur_sh'], .55); cel.clip(bibp); cel.fill(S(spl(q, closed=True)), C['wht_sh'], .9); cel.unclip()
    cel.unclip()
    nt = len(tt)
    line(tt[8:], 1.0, (.4, .05), sd=21)
    line(np.vstack([tt[-2:], bb[:int(len(bb) * .05)]]), 1.0, (.05, .5), sd=22)
    line(bb[int(len(bb) * .24):int(len(bb) * .56)], .95, (.3, .4), sd=23)
    if detail:
        for (a, b) in ((1.12, 1), (1.18, -4), (1.06, -8)):
            q = np.array([B_(a, b), B_(a, b) + [.4, 2.2], B_(a, b) + [-.6, 4.4]]); line(spl(q), .55, (.5, .3), .55, sd=25 + int(b))
    # near limbs
    leg(Hn, True, .60, .42, .22, .12)
    line(np.vstack([tt[int(nt * .86):], bb[:2]]), 1.0, (.2, .15), sd=33)
    q = np.array([B_(.02, 4), B_(.13, -2), B_(.12, -8)]); line(spl(q), .6, (.6, .5), .6, sd=31)
    leg(Fn, True, .55, .40, .40, .30)
    cf = bb[int(len(bb) * .60):][::-1]
    cff = fur(S(cf[int(len(cf) * .10):int(len(cf) * .92)]), 3.0 * k, 9.5 * k, .9, 31, side=(-1 if not flip else 1))
    cel.fill(np.vstack([cff, S(cf[int(len(cf) * .10):int(len(cf) * .92)])[::-1]]), C['wht'] if ks > 0 else C['wht_sh'])
    cel.line(cff, lw, L, (.3, .3), .28, seed=seed + 24)

    # ------------------------------------------------ head
    bk = P['earb']                                      # 0 up .. 1 flat back

    def ear(near):
        ox, oy = (-5.5, -8.5) if near else (1.8, -9.5)
        a = -1.05 * bk + (-.10 if near else .06) + P['ears'] * .25 + wv * -.10 - ha * .35
        shp_ = np.array([[-7.4, 2], [-9.0, -7], [-7.6, -15.5], [-3.4, -22.6], [-1.4, -24.0], [.6, -22.4], [5.6, -13.5], [7.6, -5], [7.2, 2.5]], float) * [1 if near else .92, 1 - .12 * bk]
        ca, sa = math.cos(a), math.sin(a)
        R = lambda q: HX(np.c_[np.asarray(q, float)[:, 0] * ca - np.asarray(q, float)[:, 1] * sa + ox, np.asarray(q, float)[:, 0] * sa + np.asarray(q, float)[:, 1] * ca + oy])
        e = spl(J(R(shp_)), n=8); es = S(e)
        cel.fill(es, C['fur'] if near else C['fur_sh'])
        cel.clip(es)
        if near and ks > 0 and not under:
            cel.fill(S(spl(HX(np.array([[-30, -60], [-9.5, -60], [-8.2, -30], [-6.8, -17], [-6.0, -8], [-6.4, 10], [-30, 10]], float)), closed=True, corners=(0, 1, 5, 6))), C['fur_sh'])
        t = np.array([[-14, -12.5], [-6, -13], [0, -15.5], [5, -12], [14, -10], [14, -40], [-14, -40]], float)
        tp = S(R(t)); te = fur(S(spl(R(t[:5]))), 2.6 * k, 4.5 * k, .0, 7 + 91, side=(-1 if not flip else 1), taper=False)
        cel.fill(np.vstack([te, S(R(t[5:]))]), C['drk'] if near else C['drk_sh'])
        if near:
            inn = np.array([[-2.2, 3], [-2.6, -6], [-1.0, -14.5], [-.2, -19.6], [3.6, -12.5], [5.6, -5], [5.8, 3]], float)
            ii = S(spl(R(inn), n=8)); cel.fill(ii, C['ear'])
            cel.shade(S(spl(R(np.array([[-3, 4], [-3.4, -6], [-.6, -19], [.6, -12], [.6, -4], [.4, 4]], float)), closed=True, corners=(0, 2, 5))), C['ear_sh'], within=ii)
            f1 = np.array([[-.8, 4], [1.0, -3], [2.6, -9.5], [3.8, -3], [5.4, 4]], float)
            cel.fill(S(spl(R(f1), corners=(2,))), C['wht'])
            if not under:
                pass
        if near:
            fl_ = fur(S(R(spl(np.array([[10, 1.5], [5.5, -2.0], [0, -2.6], [-5, -2.0], [-11, 1.5]], float)))), 3.0 * k, 3.6 * k, .1, 19, side=(1 if not flip else -1), taper=False)
            base_ = np.vstack([fl_, S(R([[-11, 12], [10, 12]]))])
            cel.fill(base_, C['fur_sh'] if under else C['fur'])
            if ks > 0 and not under: cel.clip(base_); cel.fill(S(HX([[-30, -40], [-6.2, -40], [-6.2, 12], [-30, 12]])), C['fur_sh']); cel.unclip()
        cel.unclip()
        ne_ = len(e)
        line(e[int(ne_ * (.05 if near else 0)):int(ne_ * (.93 if near else 1))], 1.0, (.18, .25), sd=41 + near)
        if rim and near:
            rp = np.array([[-8.4, -2], [-9.3, -7], [-7.9, -15.8], [-3.6, -22.9], [-1.4, -24.3], [-3.4, -20], [-6.4, -14], [-7.8, -6]], float) if ks > 0 else \
                np.array([[7.5, 1], [7.9, -5], [5.9, -13.7], [.8, -22.7], [-1.2, -24.2], [1.2, -18.6], [4.6, -11.8], [6.4, -4]], float)
            cel.shade(S(spl(R(rp), closed=True, corners=(0, 4))), C['rim'], within=es, alpha=.9)

    ear(False)
    mo = P['mouth']
    # head silhouette (profile)
    crown = [[-13.5, 6], [-14.2, -2], [-11, -10], [-4, -13.4], [3, -12.6], [8.2, -10.0], [12, -6.8], [16.2, -4.4], [20.2, -3.0], [22.8, -2.3], [24.2, -.8], [24.0, 1.2], [22.6, 2.8]]
    jaw_up = [[21.4, 4.8], [17.6, 6.8], [13, 7.8], [9.2, 8.6]]
    jaw_lo = [[9.0, 8.4], [13, 7.9 + 2.2 * mo], [17, 7.1 + 4.4 * mo], [19.0, 6.2 + 5.2 * mo], [18.6, 8.6 + 5.4 * mo], [14.4, 11.0 + 4.6 * mo], [8.5, 12.8 + 2 * mo]]
    ruff = [[5, 15.2], [1.5, 18.2 + wv], [-.2, 15.6], [-4, 19.6 + wv], [-5.4, 16], [-10.5, 18.6], [-10.2, 13], [-15.5, 13.4 + wv * .6], [-13.2, 8.5]]
    full = np.array(crown + jaw_up + [[10, 12.5]] + ruff, float)
    # lower jaw behind
    jl = S(spl(HX(J(jaw_lo)), closed=True, corners=(0, 3)))
    if mo > .04:
        mi = np.array([[9.4, 8.0], [14, 7.4], [19.2, 5.8], [18.2, 7.4 + 4.9 * mo], [14, 9.4 + 3.2 * mo]], float)
        cel.fill(S(spl(HX(mi), closed=True, corners=(0, 2, 3))), C['mouth'])
        tg = np.array([[11, 10.0], [15, 8.8 + 3.0 * mo], [18.6, 8.6 + 4.4 * mo], [15, 10.6 + 4.2 * mo]], float)
        cel.fill(S(spl(HX(tg), closed=True)), C['tongue'])
    cel.fill(jl, C['wht_sh'] if not under else C['wht'])
    line(spl(HX(J(jaw_lo[2:]))), .9, (.2, .4), sd=51)
    if carry is not None:
        m = S(HX([[18.2, 7.4]]))[0]; carry(cel, m, k)
    nc = len(crown) + len(jaw_up) + 1
    ci = (12, 13, nc + 1, nc + 3, nc + 5, nc + 7)
    fp = HX(J(full)); sil = spl(fp, closed=True, corners=ci, n=8); ss = S(sil)
    cel.fill(ss, C['fur'])
    cel.clip(ss)
    # white: muzzle side, cheek, ruff
    we = np.array([[25, 2.8], [20, .4], [15.5, -.4], [12.4, .8], [10.2, 3.8], [6, 6.4], [1, 5.4], [-4, 2.6], [-8.5, 1.6], [-12.5, 5], [-17, 9]], float)
    wf = S(spl(HX(we)))
    c0 = int(len(wf) * .34)
    wf = np.vstack([wf[:c0], fur(wf[c0:], 2.0 * k, 5.5 * k, .6, 7 + 95, side=(-1 if not flip else 1))])
    wm = np.vstack([wf, S(HX([[-24, 28], [30, 28]]))])
    cel.fill(wm, C['wht'])
    # brow patch
    cel.fill(S(spl(HX([[3.8, -9.0], [6.4, -9.8], [8.6, -8.2], [7.4, -7.2], [4.8, -7.6]]), closed=True)), mix(rgb(C['wht']), rgb(C['fur_hi']), .7))
    # cel shadow
    if under:
        se = np.array([[26, -3.5], [17, -4.6], [10.5, -7.0], [5, -9.6], [-2, -8.6], [-7, -4.5], [-9, 1], [-9, 8], [-12, 22]], float)
        cap = [[-30, 22], [-30, -30], [26, -30]]
        sd_ = (1 if not flip else -1)
    elif ks > 0:
        se = np.array([[-3, -30], [-5.5, -10], [-6.5, -3], [-4.5, 4], [-1.5, 9], [4, 12.2], [12, 12.6], [20, 8.8], [30, 9]], float)
        cap = [[30, 30], [-30, 30], [-30, -30]]
        sd_ = (1 if not flip else -1)
    else:
        se = np.array([[30, 4.0], [22, 4.2], [14, 5.6], [8, 9.0], [2, 11.0], [-6, 11.4], [-20, 12]], float)
        cap = [[-20, 30], [30, 30]]
        sd_ = (-1 if not flip else 1)
    sf = fur(S(spl(HX(se))), 2.2 * k, 6.5 * k, .6, 7 + 97, side=sd_)
    sp_ = np.vstack([sf, S(HX(cap))])
    cel.clip(sp_); cel.fill(ss, C['fur_sh']); cel.fill(wm, C['wht_sh']); cel.unclip()
    cel.unclip()
    # ink: crown, muzzle, jaw, ruff (nib lifts)
    nS = len(full)
    def seg(a, b, w=1.0, tp=(.2, .2)):
        idx = list(range(a, b + 1)); cs = [q for q, i in enumerate(idx) if i in ci]
        line(spl(fp[idx], corners=cs, n=8), w, tp, sd=60 + a)
    seg(2, 10, 1.05, (.3, .05)); seg(12, nc - 3, .95, (.05, .3))
    seg(nc - 1, nc + 1, 1.0, (.4, .04)); seg(nc + 1, nc + 3, 1.0, (.04, .04)); seg(nc + 3, nc + 5, 1.0, (.04, .04)); seg(nc + 5, nc + 7, 1.0, (.04, .04)); seg(nc + 7, nS - 1, .9, (.04, .5))
    # nose
    nz = np.array([[20.6, -2.8], [22.9, -3.0], [25.0, -1.1], [24.6, 1.2], [22.6, 2.7], [21.4, 1.2], [21.3, -.9]], float)
    cel.fill(S(spl(HX(J(nz)), closed=True)), C['nose'])
    cel.fill(S(spl(HX([[21.6, -2.0], [23.0, -2.2], [23.8, -1.3], [22.6, -1.1]]), closed=True)), mix(rgb(C['glint']), rgb(C['nose']), .45))
    line(spl(HX([[22.4, 2.9], [21.2, 5.0], [17.4, 6.9], [13, 7.8], [9.4, 8.2], [8.0, 7.0]])), .85, (.1, .35), sd=70)
    # eye
    bl = P['blink']; lx, ly = P['look']
    top_ = np.array([[12.2, -.6], [11.4, -3.6], [9.2, -5.9], [6.4, -6.1], [4.0, -4.4], [2.7, -1.8]], float)
    bot_ = np.array([[2.7, -1.8], [3.8, .4], [6.2, 1.6], [9.0, 1.5], [11.2, .8], [12.2, -.6]], float)
    clo_ = np.array([[12.2, -.4], [10.8, .4], [8.8, 1.1], [6.4, 1.1], [4.2, .3], [2.4, -1.4]], float)
    for q_ in (top_, bot_, clo_): q_ -= [7.4, -2.2]; q_ *= .88; q_ += [6.6, -2.7]
    tl = top_ * (1 - bl) + clo_ * bl
    op_ = np.vstack([spl(HX(tl)), spl(HX(bot_))]); ops = S(op_)
    if bl < .9:
        cel.clip(ops); cel.fill(ops, C['scl'])
        th = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        ic = np.array([7.0 + lx * 1.1, -2.6 + ly * 1.1])
        E = lambda ax, ay, dx=0, dy=0: S(HX(np.c_[ic[0] + dx + ax * np.cos(th), ic[1] + dy + ay * np.sin(th)]))
        ir = E(3.3, 3.8); cel.fill(ir, C['iris'])
        cel.shade(E(2.7, 1.9, .2, 2.0), C['iris_lo'], within=ir); cel.shade(E(4.4, 2.3, 0, -2.9), C['iris_hi'], within=ir)
        cel.fill(E(1.8, 2.6, .3, -.1), C['pupil'])
        ls = np.vstack([spl(HX(tl + [0, -4])), spl(HX(tl + [0, .6 if h < 170 else 1.4]))[::-1]]); cel.fill(S(ls), mix(rgb(C['iris_hi']), rgb(C['pupil']), .6), .7)
        cel.fill(E(.95, 1.15, -.9, -1.4), C['glint'])
        cel.fill(E(.6, .45, 1.2, 1.5), '#ffd98a' if pal == 'lantern' else mix(rgb(C['iris_lo']), rgb(C['glint']), .6))
        cel.unclip()
    if bl > .05:
        cel.fill(S(np.vstack([spl(HX(top_ + [0, -.3])), spl(HX(tl))[::-1]])), C['fur'])
    wfn = lambda q: .45 + .95 * np.sin(np.pi * np.clip(q * .9 + .05, 0, 1)) ** .7
    small = h < 170
    cel.line(S(spl(HX(tl))), lw * ((1.0 if small else 1.9) if bl < .9 else 1.3), C['lash'], (.12, .18), seed=seed + 85, wfun=wfn, tremor=.15)
    line(spl(HX([[10.8, -1.3], [12.2, -.1], [13.2, 1.3]])), .8 if small else 1.25, (.05, .7), col=C['lash'], sd=86)
    if bl < .9: line(spl(HX(bot_)), .45 if small else .8, (.25, .25), .9, col=C['lash'], sd=87)
    if detail and h > 220:
        for q in ([[16.4, 3.2], [17.0, 3.8]], [[18.4, 2.8], [19.0, 3.4]], [[17.4, 4.6], [17.9, 5.1]]): line(HX(q), .8, (.2, .2), .6, sd=88)
        for (x1, y1, x2, y2) in ((17.2, 4.2, 8, 11.0), (16.8, 4.8, 6, 8.2), (17.6, 3.6, 8, 4.4)):
            cel.line(S(spl(HX([[x1, y1], [(x1 + x2) / 2 - 1, (y1 + y2) / 2 + 1.6], [x2 - 4 - wv * 4, y2 + 1.4 + wv * .5]]))), lw * .45, mix(rgb(C['wht']), rgb(L), .45), (.02, .7), seed=seed + 89, alpha=.75, press=.1)
        for (a, b) in ((-3, 9.5), (-7.5, 10.5), (2, 12), (-1, 13.8)):
            line(spl(HX([[a, b], [a - 1.4, b + 1.0], [a - 2.2, b + 2.6]])), .55, (.5, .25), .55, sd=90 + int(a))
        for (a, b) in ((-2, -10), (1.5, -9.6), (-5, -8.6)):
            line(spl(HX([[a, b], [a - 1.2, b + 1.2]])), .45, (.5, .5), .4, sd=95 + int(a))
    if rim:
        rp = [[-13.8, 4], [-14.4, -2], [-11.2, -10.2], [-4, -13.9], [-4.4, -12.6], [-10, -9.0], [-12.9, -2]] if ks > 0 else [[4, -12.5], [8.6, -10.0], [12.2, -7.0], [20.6, -3.2], [23, -2.4], [20, -2.0], [11.6, -5.6], [7.8, -8.8]]
        cel.shade(S(spl(HX(rp), closed=True, corners=(0, 3) if ks > 0 else (0, 4))), C['rim'], within=ss, alpha=.85)
        q = np.vstack([tt[10:], (tt[10:] - nrm * 1.5)[::-1]])
        cel.shade(S(q), C['rim'], within=Ts, alpha=.75)
    ear(True)
    return dict(mouth=S(HX([[19.5, 8.6]]))[0], eye=S(HX([[8, -2]]))[0], nose=S(HX([[25.6, 2]]))[0], paw=S(Fn[5])[0], head=S(hd)[0])


def trot(ph, stride=88., lift=9., **kw):
    """trot drawing at phase ph (0..1); body travels `stride` units per cycle"""
    half = stride * .5 * .5

    def foot(p, cx):
        p = p % 1.
        if p < .5: t = p / .5; return np.array([cx + half * (1 - 2 * t), 0.]), 0.
        t = (p - .5) / .5; e = t * t * (3 - 2 * t)
        return np.array([cx + half * (-1 + 2 * e), -lift * math.sin(math.pi * t) ** .8]), (.75 * math.sin(math.pi * min(1, t * 1.25)) if t < .8 else -.25 * math.sin(math.pi * (t - .8) / .2))
    fn, a1 = foot(ph, 21); hf, a2 = foot(ph + .02, -19)
    ff, a3 = foot(ph + .5, 25); hn, a4 = foot(ph + .52, -23)
    bob = 1.7 * math.cos(4 * math.pi * (ph - .12))
    p = pose(hip=(-21, -40 + bob), sh=(20, -42 + bob * .8), head=(38, -61 + bob * .5 - .6 * math.sin(4 * math.pi * ph)), hang=.10 + .03 * math.sin(4 * math.pi * ph + .6),
             fn=tuple(fn), ff=tuple(ff), hn=tuple(hn), hf=tuple(hf), fnp=a1, ffp=a3, hnp=a4, hfp=a2,
             tail=((-.10 + .07 * math.sin(4 * math.pi * ph), 13), (.16 + .08 * math.sin(4 * math.pi * ph - .8), 14), (.10 + .10 * math.sin(4 * math.pi * ph - 1.5), 15), (-.18 + .12 * math.sin(4 * math.pi * ph - 2.3), 14), (-.28 + .14 * math.sin(4 * math.pi * ph - 3), 10)),
             tb=3.30, earb=.10)
    p.update(kw); return p
