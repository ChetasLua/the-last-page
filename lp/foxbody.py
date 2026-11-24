"""Fox: bust for close-ups, sitting three-quarter front body, paw from above, tail tip.  All on-model with lp/fox.head."""
import numpy as np, math
from .ink import Cel, spl, resample
from .paint import rgb, mix
from .fox import PAL, fur, head, _proj


def bust(cel, cx, cy, s=2.6, pal='lantern', seed=0, lw=3.2, key=1, wind=0.0, breath=0.0, lean=0.0, under=True, rim=True, turn=1.0, rim_k=None):
    """chest, neck and shoulders below a head() at the same (cx, cy, s).  turn: -1..1 three-quarter twist (+ faces right)"""
    C = PAL[pal] if isinstance(pal, str) else pal
    r = np.random.default_rng(6000 + abs(seed))
    L = C['line']; kx = -key; wv = wind; rk = -key if rim_k is None else rim_k
    S = lambda P: np.asarray(P, float) * s + [cx, cy]
    J = lambda P: np.asarray(P, float) + r.normal(0, .45, np.shape(P))
    b = breath; q_ = abs(turn)
    # viewer-left is the near shoulder and back, viewer-right the chest (mirrored when turn < 0)
    back = np.array([[-64, 62], [-84, 100], [-104, 136 - b], [-140, 186 - b], [-176, 252], [-198, 330], [-206, 420]], float)
    chest = np.array([[66, 62], [84, 92], [104, 134 - b], [114, 182 - b], [112, 232], [100, 290], [92, 350], [94, 420]], float)
    sym_b = np.array([[-64, 62], [-88, 104], [-108, 150 - b], [-132, 206 - b], [-150, 270], [-160, 340], [-164, 420]], float)
    sym_c = np.array([[66, 62], [88, 98], [104, 140 - b], [120, 186 - b], [136, 236], [148, 290], [156, 350], [160, 420]], float)
    back = back * q_ + sym_b * (1 - q_); chest = chest * q_ + sym_c * (1 - q_)
    out = np.vstack([back, chest[::-1]])
    if turn < 0: out = (out * [-1, 1])[::-1]
    out[:, 0] += lean
    o = spl(J(out), n=10, corners=(6, 7)); os_ = S(o)
    cel.fill(os_, C['fur'])
    cel.clip(os_)
    sg = 1 if turn >= 0 else -1
    cxo = 24 * q_ * sg
    # shoulder / back shadow
    w0 = 1.0 - .06 * q_ * (1 if kx * sg < 0 else 0)
    q = np.array([[kx * 74 * w0, 70], [kx * 84 * w0, 130], [kx * 100 * w0, 190], [kx * 112 * w0, 262], [kx * 104 * w0, 330], [kx * 112 * w0, 430]], float)
    e = fur(S(spl(q)), 9 * s, 26 * s, .8, 7 + 3, side=-kx)
    cel.fill(np.vstack([e, S([[kx * 320, 440], [kx * 320, 40]])]), C['fur_sh'])
    # near foreleg contour
    leg = np.array([[-96 * sg, 246], [-74 * sg, 286], [-66 * sg, 340], [-70 * sg, 430]], float) if q_ > .4 else None
    # white ruff in two overlapping tiers of locks
    for tier, (y0, y1, w0, w1) in enumerate(((150, 316, 86, 84), (86, 194, 92, 104))):
        off = lambda y: cxo * np.clip((y - 70) / 200., 0, 1.3)
        left = np.array([[-w0, 66], [-w1, y0 * .5 + 60], [-w1 * .88, (y0 + y1) / 2 + b], [-w1 * .5, y1 - 26 + b], [0, y1 + b]], float)
        right = left[::-1] * [-1, 1]
        left[:, 0] += off(left[:, 1]); right[:, 0] += off(right[:, 1])
        pl = fur(S(spl(J(left))), 12 * s, 27 * s, .9, 7 + 11 + tier, side=-1)
        pr = fur(S(spl(J(right)))[::-1], 12 * s, 27 * s, .9, 7 + 17 + tier, side=1)[::-1]
        if wv:
            for P in (pl, pr): P[:, 0] += wv * s * 13 * np.clip((P[:, 1] - cy) / (s * 300), 0, 1) ** 1.2
        shape = np.vstack([pl, pr, S([[90, 40], [-90, 40]])])
        cel.fill(shape, C['wht'] if tier else mix(rgb(C['wht']), rgb(C['wht_sh']), .24))
        cel.clip(shape)
        q = np.array([[kx * 12, 60], [kx * 44, 120], [kx * 36, 170], [kx * 46, 230], [kx * 32, 340]], float); q[:, 0] += off(q[:, 1])
        e = fur(S(spl(q)), 10 * s, 24 * s, .9, 7 + 23 + tier, side=-kx)
        cel.fill(np.vstack([e, S([[kx * 300, 360], [kx * 300, 40]])]), C['wht_sh'])
        if tier:
            q = np.array([[-110, 82], [-60, 108], [-24, 124], [14, 126], [54, 112], [110, 84]], float)
            e = fur(S(spl(q)), 9 * s, 20 * s, .4, 7 + 29, side=-1)
            cel.fill(np.vstack([e, S([[130, 30], [-130, 30]])]), C['wht_sh'])
        cel.unclip()
        cel.line(pl[int(len(pl) * .18):int(len(pl) * .975)], lw * .85, L, (.3, .12), .3, seed=seed + 31 + tier)
        cel.line(pr[int(len(pr) * .025):int(len(pr) * .82)], lw * .85, L, (.12, .3), .3, seed=seed + 37 + tier)
        for (x, y, dx, dy) in ((-34, 150, -6, 26), (18, 140, 2, 30), (52, 160, 6, 22), (-8, 176, 1, 22)):
            yy = y + (90 if tier == 0 else 0)
            t = np.array([[x, yy], [x + dx * .4 + 2, yy + dy * .5], [x + dx, yy + dy]], float); t[:, 0] += off(t[:, 1])
            cel.line(S(spl(J(t))), lw * .5, L, (.5, .25), seed=seed + 41 + x, alpha=.5)
    if leg is not None:
        cel.line(S(spl(J(leg))), lw * .9, L, (.6, .05), .3, seed=seed + 47)
        for (x, y) in ((-128, 214), (-146, 262), (-120, 300)):
            t = np.array([[x * sg, y], [(x - 5) * sg, y + 12], [(x - 4) * sg, y + 26]], float)
            cel.line(S(spl(J(t))), lw * .55, L, (.5, .3), seed=seed + 48 + x, alpha=.55)
    cel.unclip()
    n = len(o); nb = int(n * 6.0 / (len(out) - 1))
    cel.line(os_[int(nb * .16):nb], lw, L, (.3, .05), .3, seed=seed + 51)
    cel.line(os_[nb + int((n - nb) * .13):int(n * .97)], lw, L, (.05, .3), .3, seed=seed + 52)
    if rim:
        E = out[:7] if (rk < 0) else out[7:][::-1]
        E = spl(E[1:], n=8)
        T = np.gradient(E, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9
        N = np.c_[T[:, 1], -T[:, 0]]
        if (N[:, 0] * rk).mean() > 0: N = -N
        w = 5.5 * np.sin(np.pi * np.linspace(.04, .96, len(E)))[:, None] ** .5
        cel.shade(S(np.vstack([E, (E + N * w)[::-1]])), C['rim'], within=os_, alpha=.85)


def sit(cel, x, y, h, pal='gold', lw=2.4, seed=0, key=1, hd=None, tail=0.0, reach=None, breath=0.0, rim=True, under=False, wind=0.0, back=1.0, rim_k=None, tside=1, lean=(0., 0.)):
    """sitting three-quarter front.  (x, y) ground between the front paws, h = px to ear tips.
    hd: dict passed to head().  reach: None or (px, py, lift) screen target for the viewer's-right front paw.  tail: tip swing -1..1"""
    C = PAL[pal] if isinstance(pal, str) else pal
    r = np.random.default_rng(7000 + abs(seed))
    k = h / 470.
    L = C['line']; kx = -key; rk = -key if rim_k is None else rim_k
    S = lambda P: np.asarray(P, float).reshape(-1, 2) * k + [x, y]
    J = lambda P: np.asarray(P, float) + r.normal(0, .45, np.shape(P))
    b = breath * 2.2
    hd = dict(hd or {})
    hs = hd.pop('s', .735) * k
    hx, hy = hd.pop('pos', (4, -340))
    hx, hy = (hx + lean[0]) * k + x, (hy - b + lean[1]) * k + y

    def ln(P, w=1., tp=(.25, .25), al=1., sd=0): cel.line(P, lw * w, L, tp, .28, seed=seed * 7 + sd, alpha=al)

    # ---- tail, wrapped from behind on the viewer's right, tip forward
    tw = tail
    Tp = np.array([[60, -60], [112, -46], [160, -34], [196 + 6 * tw, -44 - 14 * tw], [226 + 8 * tw, -78 - 44 * tw], [238 + 2 * tw, -118 - 62 * tw], [226 - 22 * tw, -158 - 66 * tw]], float) * [tside, 1]
    from .foxside import edges
    A, B = edges(J(Tp), [16, 24, 33, 38, 36, 24, 2], 10)
    n = len(A)
    c0, c1 = int(n * .25), int(n * .92)
    Af = np.vstack([S(A[:c0]), fur(S(A[c0:c1]), 9 * k, 30 * k, 1.1, 7 + 5, side=1), S(A[c1:])])
    Bf = np.vstack([S(B[:c0]), fur(S(B[c0:c1]), 9 * k, 30 * k, 1.1, 7 + 6, side=-1), S(B[c1:])])
    shape = np.vstack([Af, Bf[::-1]]); mid = (A + B) / 2
    cel.fill(shape, C['fur']); cel.clip(shape)
    low = B if B[n // 2, 1] > A[n // 2, 1] else A
    q = np.vstack([mid * .7 + low * .3, (low * 2.4 - mid * 1.4)[::-1]])
    cel.fill(fur(S(q), 8 * k, 34 * k, .8, 7 + 7), C['fur_sh'])
    i = int(n * .66)
    e = np.vstack([A[i - 3:i - 2] * 1.8 - mid[i:i + 1] * .8, A[i:i + 1], mid[i + 5:i + 6], B[i:i + 1], B[i - 3:i - 2] * 1.8 - mid[i:i + 1] * .8])
    ee = fur(S(spl(e)), 20 * k, 20 * k, -.2, 7 + 8, side=1, taper=False)
    wt = np.vstack([ee, S(np.vstack([B[i:] * 1.9 - mid[i:] * .9, mid[-1:] + (mid[-1:] - mid[i]) * .5, (A[i:] * 1.9 - mid[i:] * .9)[::-1]]))])
    cel.fill(wt, C['wht']); cel.clip(wt); cel.fill(S(np.vstack([mid[i - 2:] * .7 + low[i - 2:] * .3, (low[i - 2:] * 2.4 - mid[i - 2:] * 1.4)[::-1]])), C['wht_sh']); cel.unclip()
    cel.unclip()
    ln(Af, 1., (.2, .05), sd=1); ln(Bf, 1., (.2, .05), sd=2)
    if rim:
        up_ = A if low is B else B
        rq = np.vstack([up_[c0:], (up_ * .86 + mid * .14)[c0:][::-1]])
        cel.shade(S(rq), C['rim'], within=shape, alpha=.8)

    # ---- far haunch + back
    body = np.array([[-40, -284 - b], [-58, -236 - b], [-67, -192], [-70, -152], [-82, -122], [-108, -96], [-121, -60], [-118, -24], [-100, -4], [-60, 1], [60, 1], [98, -4], [115, -26], [117, -62], [104, -98], [80, -124], [72, -152], [69, -192], [61, -236 - b], [44, -284 - b]], float)
    lk = lambda Q: np.asarray(Q, float) + np.array(lean)[None] * np.clip(-np.asarray(Q, float)[:, 1:2] / 284., 0, 1) ** 1.2
    body = lk(body)
    bo = spl(J(body), n=9); bs = S(bo)
    cel.fill(bs, C['fur'])
    cel.clip(bs)
    q = np.array([[kx * 60, -300], [kx * 46, -230], [kx * 52, -180], [kx * 66, -130], [kx * 62, -80], [kx * 74, -30], [kx * 70, 20]], float)
    e = fur(S(spl(q)), 9 * k, 30 * k, .8, 7 + 11, side=-kx)
    cel.fill(np.vstack([e, S([[kx * 200, 20], [kx * 200, -300]])]), C['fur_sh'])
    cel.unclip()
    nB = len(bo)
    ln(bs[int(nB * .03):int(nB * .46)], 1., (.25, .2), sd=3); ln(bs[int(nB * .56):int(nB * .97)], 1., (.2, .25), sd=4)
    # haunch lines and hind paws
    for sd in (-1, 1):
        hl = np.array([[sd * 50, -112], [sd * 66, -104], [sd * 75, -80], [sd * 72, -46], [sd * 54, -18]], float)
        ln(S(spl(J(hl))), .8, (.5, .3), .85, sd=6 + sd)
        hp = np.array([[sd * 40, 1], [sd * 44, -13], [sd * 60, -20], [sd * 84, -17], [sd * 97, -13], [sd * 106, -6], [sd * 105, 2], [sd * 96, 5], [sd * 88, 3], [sd * 78, 6], [sd * 66, 4], [sd * 54, 5]], float)
        hs_ = S(spl(J(hp), closed=True))
        cel.fill(hs_, C['drk'] if sd * key > 0 else C['drk_sh']); ln(np.vstack([hs_, hs_[:1]]), .95, (0, 0), sd=9 + sd)
        for t in (72, 84): ln(S(spl([[sd * t, -6], [sd * (t + 3), 1]])), .6, (.3, .3), .7, sd=11 + t)

    # ---- front legs (standing one, then the one that can reach)
    def fleg(x0, x1, top, ft, near, sd, wid=1.0):
        A, Bq = edges(J(np.vstack([lk([[x0, top], [x0 * .9 + x1 * .1, top + 50]]) * [.94, 1], [[x1 * .7 + x0 * .3, (top + ft) / 2 + 22], [x1, ft - 18], [x1, ft - 6]]])), np.array([26, 24, 18, 15, 15.5]) * wid)
        leg = np.vstack([A, Bq[::-1]]); ls = S(leg)
        cel.fill(ls, C['fur'] if near else mix(rgb(C['fur']), rgb(C['fur_sh']), .5))
        cel.clip(ls)
        E = A if key < 0 else Bq
        m = (A + Bq) / 2
        q = np.vstack([E * 1.5 - m * .5, (m * .5 + E * .5)[::-1]]) if False else np.vstack([(Bq if kx < 0 else A) * 1.6 - m * .6, (m * .45 + (Bq if kx < 0 else A) * .55)[::-1]])
        cel.fill(S(q), C['fur_sh'])
        i = int(len(A) * .52)
        e = np.vstack([A[i:i + 1] * 1.6 - m[i:i + 1] * .6, m[i:i + 1] + [0, 4], Bq[i:i + 1] * 1.6 - m[i:i + 1] * .6])
        ee = fur(S(spl(e)), 13 * k, 10 * k, .1, 7 + 13 + sd, side=1, taper=False)
        cel.fill(np.vstack([ee, S(np.vstack([Bq[i:] * 2 - m[i:], (A[i:] * 2 - m[i:])[::-1]]))]), C['drk'])
        cel.clip(np.vstack([ee, S(np.vstack([Bq[i:] * 2 - m[i:], (A[i:] * 2 - m[i:])[::-1]]))])); cel.fill(S(q), C['drk_sh']); cel.unclip()
        cel.unclip()
        pw = np.array([[x1 - 18, ft - 9], [x1 - 15, ft - 20], [x1, ft - 25], [x1 + 15, ft - 20], [x1 + 19, ft - 8], [x1 + 14, ft + 3], [x1, ft + 5], [x1 - 14, ft + 3]], float)
        ps = S(spl(J(pw), closed=True)); cel.fill(ps, C['drk'])
        cel.shade(S([[x1 - 24 if kx < 0 else x1 + 4, ft - 30], [x1 - 5 if kx < 0 else x1 + 30, ft - 30], [x1 - 7 if kx < 0 else x1 + 30, ft + 8], [x1 - 24 if kx < 0 else x1 + 6, ft + 8]]), C['drk_sh'], within=ps)
        o0 = int(len(A) * .30)
        ln(S(A[o0:]), .95, (.4, .05), sd=15 + sd); ln(S(Bq[o0:]), .95, (.4, .05), sd=17 + sd)
        ln(np.vstack([ps, ps[:1]])[4:], .95, (.1, .1), sd=19 + sd)
        for t in (-5, 6): ln(S(spl([[x1 + t, ft - 10], [x1 + t * 1.1, ft - 2], [x1 + t * 1.05, ft + 4]])), .6, (.3, .2), .75, sd=21 + t)

    fleg(-28, -27, -175, 0, True, 1)
    if reach is None:
        fleg(28, 27, -175, 0, True, 2)
    if rim:
        q = np.array([[rk * 52, -240], [rk * 62, -222], [rk * 69, -186], [rk * 73, -150], [rk * 106, -98], [rk * 118, -60], [rk * 111, -62], [rk * 100, -96], [rk * 66, -150], [rk * 62, -186]], float)
        cel.shade(S(spl(lk(q), closed=True, corners=(0, 5))), C['rim'], within=bs, alpha=.9)
    if reach is not None:
        px, py, lift = reach
        tx, ty = (px - x) / k, (py - y) / k
        s0 = lk(np.array([[26., -182.]]))[0]
        d = np.array([tx, ty - 8 - lift]) - s0; Ld = np.hypot(*d)
        from .foxside import ik2
        e_, w_ = ik2(s0, s0 + d, 94, 106, 1)
        A, Bq = edges(J([s0 + [-6, -12], s0 + [2, 14], e_, (e_ + w_) / 2, w_]), [32, 28, 19, 15.5, 15.5])
        leg = np.vstack([A, Bq[::-1]]); ls = S(leg); m = (A + Bq) / 2
        cel.fill(ls, C['fur']); cel.clip(ls)
        lowE = A if A[len(A) // 2, 1] > Bq[len(A) // 2, 1] else Bq
        cel.fill(S(np.vstack([lowE * 1.6 - m * .6, (m * .45 + lowE * .55)[::-1]])), C['fur_sh'])
        i = int(len(A) * .56)
        e = np.vstack([A[i:i + 1] * 1.6 - m[i:i + 1] * .6, m[i + 2:i + 3], Bq[i:i + 1] * 1.6 - m[i:i + 1] * .6])
        ee = fur(S(spl(e)), 13 * k, 10 * k, .1, 7 + 53, side=1, taper=False)
        dk = np.vstack([ee, S(np.vstack([Bq[i:] * 2 - m[i:], (A[i:] * 2 - m[i:])[::-1]]))])
        cel.fill(dk, C['drk']); cel.clip(dk); cel.fill(S(np.vstack([lowE * 1.6 - m * .6, (m * .45 + lowE * .55)[::-1]])), C['drk_sh']); cel.unclip(); cel.unclip()
        pw = np.array([[-15, -10], [-8, -18], [8, -17], [22, -11], [27, -2], [22, 7], [4, 9], [-13, 6]], float) + w_ + [3, 2]
        ps = S(spl(J(pw), closed=True)); cel.fill(ps, C['drk'])
        ln(S(A[int(len(A) * .2):]), .95, (.4, .05), sd=61); ln(S(Bq[int(len(Bq) * .2):]), .95, (.4, .05), sd=62); ln(np.vstack([ps, ps[:1]])[3:], .95, (.1, .1), sd=63)
        for t in (10, 18): ln(S(spl([[t, -8] + w_, [t + 4, 0] + w_, [t + 2, 6] + w_])), .6, (.3, .2), .75, sd=64 + t)
    # ---- chest ruff (white), spikes over the legs
    for tier, (y1, w1) in enumerate(((-92, 50), (-150, 64))):
        left = np.array([[-34, -292 - b], [-w1 * .92, -250 - b], [-w1 - 4, -200], [-w1 * .62, y1 - 22], [0, y1]], float)
        right = lk(left[::-1] * [-1, 1]); left = lk(left)
        pl = fur(S(spl(left)), 13 * k, 32 * k, .9, 7 + 31 + tier, side=-1); pr = fur(S(spl(right))[::-1], 13 * k, 32 * k, .9, 7 + 33 + tier, side=1)[::-1]
        if wind:
            for P in (pl, pr): P[:, 0] += wind * k * 9 * np.clip((P[:, 1] - (y - 262 * k)) / (170 * k), 0, 1)
        sh_ = np.vstack([pl, pr])
        cel.fill(sh_, C['wht'] if tier else mix(rgb(C['wht']), rgb(C['wht_sh']), .4)); cel.clip(sh_)
        q = np.array([[kx * 4, -290], [kx * 24, -230], [kx * 18, -190], [kx * 28, -150], [kx * 12, -80]], float)
        e = fur(S(spl(q)), 8 * k, 26 * k, .9, 7 + 35 + tier, side=-kx)
        cel.fill(np.vstack([e, S([[kx * 160, -60], [kx * 160, -300]])]), C['wht_sh'])
        if tier: cel.fill(np.vstack([fur(S(spl([[-70, -262], [-30, -236], [6, -230], [40, -238], [74, -262]])), 8 * k, 22 * k, .4, 7 + 37, side=-1), S([[80, -300], [-80, -300]])]), C['wht_sh'])
        cel.unclip()
        ln(pl[int(len(pl) * .2):int(len(pl) * .97)], .85, (.35, .12), sd=41 + tier); ln(pr[int(len(pr) * .03):int(len(pr) * .8)], .85, (.12, .35), sd=43 + tier)
    head(cel, hx, hy, s=hs, pal=pal, lw=lw * .95, seed=seed, key=key, rim=rim, under=under, wind=wind, rim_k=rim_k, **hd)


def paw_top(cel, px, py, s=1.0, ang=-.72, press=1.0, pal='lantern', seed=0, lw=3.0, key=1):
    """fore-paw seen from above and behind, reaching away along ang (radians, 0 = right).  (px, py) = centre of the pad."""
    C = PAL[pal] if isinstance(pal, str) else pal
    r = np.random.default_rng(8000 + abs(seed)); L = C['line']
    ca, sa = math.cos(ang + math.pi / 2), math.sin(ang + math.pi / 2)
    def S(P):
        P = np.asarray(P, float).reshape(-1, 2) * s
        return np.c_[P[:, 0] * ca - P[:, 1] * sa + px, P[:, 0] * sa + P[:, 1] * ca + py]
    J = lambda P: np.asarray(P, float) + r.normal(0, .8, np.shape(P))
    sp = press
    legL = np.array([[-62 - 6 * sp, 26], [-57, 70], [-58, 130], [-72, 230], [-104, 390], [-150, 640], [-196, 980], [-230, 1500]], float)
    legR = np.array([[62 + 6 * sp, 26], [58, 70], [60, 130], [74, 230], [92, 390], [112, 640], [140, 980], [170, 1500]], float)
    legL[3:, 0] += 18; legR[3:, 0] += 22
    EL, ER = spl(J(legL), n=12), spl(J(legR), n=12)
    cL, cR = int(len(EL) * .40), int(len(ER) * .46)
    fL = np.vstack([S(EL[:cL]), fur(S(EL[cL:]), 9 * s, 74 * s, 1.0, 7 + 9, side=-1)])
    fR = np.vstack([S(ER[:cR]), fur(S(ER[cR:]), 9 * s, 74 * s, 1.0, 7 + 10, side=1)])
    ls = np.vstack([fL, fR[::-1]])
    cel.fill(ls, C['fur']); cel.clip(ls)
    q = np.array([[-40 * key, 20], [-18 * key, 120], [-26 * key, 230], [-24 * key, 380], [-30 * key, 640], [-30 * key, 1500]], float)
    e = fur(S(spl(q)), 15 * s, 46 * s, .6, 7 + 3, side=key)
    cel.fill(np.vstack([e, S([[-360 * key, 1500], [-360 * key, 20]])]), C['fur_sh'])
    brk = fur(S(spl([[-130, 236], [-60, 262], [0, 250], [60, 270], [150, 240]])), 34 * s, 34 * s, .0, 7 + 5, side=-1, taper=False)
    dk = np.vstack([brk, S([[200, -60], [-200, -60]])])
    cel.fill(dk, C['drk']); cel.clip(dk); cel.fill(np.vstack([e, S([[-360 * key, 1500], [-360 * key, 20]])]), C['drk_sh']); cel.unclip()
    cel.unclip()
    cel.line(fL, lw, L, (.06, .02), .3, seed=seed + 11); cel.line(fR, lw, L, (.06, .02), .3, seed=seed + 12)
    # pad and four toes, spread by pressure
    toes = [(-50 - 6 * sp, -18, 24, 34, -.30), (-20 - 2 * sp, -44 - 2 * sp, 25, 38, -.10), (18 + 2 * sp, -46 - 2 * sp, 25, 38, .10), (50 + 6 * sp, -20, 24, 34, .30)]
    back = np.array([[-66 - 6 * sp, 10], [-64, -14], [-40, -34], [0, -42], [40, -34], [64, -14], [66 + 6 * sp, 10], [64, 44], [0, 62], [-64, 44]], float)
    bs = S(spl(J(back), closed=True)); cel.fill(bs, C['drk'])
    for (tx, ty, rx, ry, a) in (toes[0], toes[3], toes[1], toes[2]):
        th = np.linspace(0, 2 * np.pi, 14, endpoint=False)
        t = np.c_[rx * np.cos(th) * (1 + .08 * np.sin(th)), ry * np.sin(th)]
        c_, s_ = math.cos(a), math.sin(a)
        t = np.c_[t[:, 0] * c_ - t[:, 1] * s_ + tx, t[:, 0] * s_ + t[:, 1] * c_ + ty]
        ts = spl(S(J(t)), closed=True)
        cel.fill(ts, C['drk'])
        cel.shade(S(np.array([[tx - 40 * key, ty - 60], [tx - 8 * key, ty - 60], [tx - 4 * key, ty], [tx - 10 * key, ty + 50], [tx - 40 * key, ty + 50]])), C['drk_sh'], within=ts)
        nT = len(ts); ro = np.roll(ts, -int(nT * .06), 0)
        cel.line(ro[int(nT * .42):], lw * .95, L, (.15, .15), .3, seed=seed + int(tx) + 100)
        # claw
        cl = np.array([[tx - 5 + a * 30, ty - ry + 6], [tx + a * 40, ty - ry - 13], [tx + 5 + a * 30, ty - ry + 5]], float)
        cel.fill(S(spl(cl, corners=(1,))), mix(rgb(C['drk_sh']), rgb(C['wht']), .25)); cel.line(S(spl(cl, corners=(1,))), lw * .55, L, (.2, .2), seed=seed + 120)
    hi = np.array([[-30 * key + 40 * key, -30], [24 * key + 20 * key, -8], [30 * key + 18 * key, 30], [40 * key, 30], [40 * key, 0]], float)
    for (x0, y0) in ((-22, 4), (4, -2), (28, 8), (-6, 26), (18, 34)):
        cel.line(S(spl(J([[x0, y0], [x0 + 2, y0 + 10], [x0 - 1, y0 + 22]]))), lw * .55, mix(rgb(C['drk']), rgb(C['fur']), .35), (.5, .4), seed=seed + 130 + x0, alpha=.8)
    cel.fill(S(spl([[44 * key, 20], [58 * key, 34], [58 * key, 70], [60 * key, 130], [52 * key, 130], [50 * key, 70]], closed=True)), mix(rgb(C['drk']), rgb(C['fur_hi']), .25), .8)


def tail_tip(cel, P, w=70., pal='gold', seed=0, lw=3.0, white=.55):
    """tail lying/curling into frame along spine P (screen points, root first)"""
    from .foxside import edges
    C = PAL[pal]; r = np.random.default_rng(8500 + abs(seed)); L = C['line']
    P = np.asarray(P, float) + r.normal(0, .8, np.shape(P))
    m_ = len(P)
    A, B = edges(P, np.interp(np.linspace(0, 1, m_), [0, .35, .7, .88, 1], np.array([.85, 1.0, .95, .6, .02]) * w), 12)
    n = len(A); c1 = int(n * .93); mid = (A + B) / 2
    Af = np.vstack([fur(A[:c1], w * .24, w * .7, 1.1, 7 + 1, side=1), A[c1:]]); Bf = np.vstack([fur(B[:c1], w * .24, w * .7, 1.1, 7 + 2, side=-1), B[c1:]])
    sh = np.vstack([Af, Bf[::-1]])
    cel.fill(sh, C['fur']); cel.clip(sh)
    low = B if B[n // 2, 1] > A[n // 2, 1] else A
    sq = fur(np.vstack([mid * .72 + low * .28, (low * 2.4 - mid * 1.4)[::-1]]), w * .2, w * .8, .8, 7 + 3)
    cel.fill(sq, C['fur_sh'])
    i = int(n * white)
    e = np.vstack([A[i - 3:i - 2] * 1.8 - mid[i:i + 1] * .8, A[i:i + 1], mid[i + 5:i + 6], B[i:i + 1], B[i - 3:i - 2] * 1.8 - mid[i:i + 1] * .8])
    ee = fur(spl(e), w * .6, w * .5, -.2, 7 + 4, side=1, taper=False)
    wt = np.vstack([ee, B[i:] * 1.9 - mid[i:] * .9, mid[-1:] * 2 - mid[-6:-5], (A[i:] * 1.9 - mid[i:] * .9)[::-1]])
    cel.fill(wt, C['wht']); cel.clip(wt); cel.fill(sq, C['wht_sh']); cel.unclip(); cel.unclip()
    cel.line(Af, lw, L, (.02, .05), .3, seed=seed + 5); cel.line(Bf, lw, L, (.02, .05), .3, seed=seed + 6)
    for t0 in (.25, .4, .72, .82):
        i0 = int(n * t0); q = np.array([mid[i0] * .7 + A[i0] * .3, mid[min(n - 1, i0 + 5)] * .85 + A[min(n - 1, i0 + 5)] * .15, mid[min(n - 1, i0 + 10)]])
        cel.line(spl(q), lw * .55, L, (.5, .4), seed=seed + 7 + i0, alpha=.5)
