"""The fox.  One character, one model sheet, bespoke drawings per view.

head(): 2.5-D front / three-quarter head (control points carry depth, so a turn is a redraw).
Flat cel colour + one hard shadow + optional rim, fur-cut colour breaks, inked with pressure lines."""
import numpy as np, math
from .ink import Cel, spl, xf, noise1, resample
from .paint import rgb, mix

PAL = {
    'day': dict(fur='#d9793f', fur_sh='#b25c42', fur_hi='#eb9a5a', wht='#f8ecd8', wht_sh='#cfb7b4', drk='#4d3633', drk_sh='#38282c',
                ear='#eccbb4', ear_sh='#be918c', nose='#362a2c', line='#4b2c25', iris='#a8651f', iris_lo='#e6ab47', iris_hi='#6c3a18',
                pupil='#22150f', scl='#f1e2c6', rim='#fff0cc', glint='#fffdf2', lash='#301b18', mouth='#7a3a35', tongue='#d98578'),
    'lantern': dict(fur='#cf7436', fur_sh='#8a4c4c', fur_hi='#eda25c', wht='#f6dcb0', wht_sh='#ab8f98', drk='#4a3234', drk_sh='#31253a',
                    ear='#dba884', ear_sh='#8c5c63', nose='#2e2129', line='#3c2223', iris='#a5611c', iris_lo='#efb44c', iris_hi='#5c3017',
                    pupil='#1b1012', scl='#e9cfa4', rim='#9db4dc', glint='#fff8e6', lash='#25151a', mouth='#6a2f2e', tongue='#c97660'),
    'moon': dict(fur='#8d5e62', fur_sh='#5f4767', fur_hi='#ad7c76', wht='#bdb7c8', wht_sh='#817c9c', drk='#3a2e4a', drk_sh='#2a2340',
                 ear='#9c8190', ear_sh='#6a587c', nose='#251d30', line='#2a1f36', iris='#a5713c', iris_lo='#d9ad66', iris_hi='#5a3a34',
                 pupil='#17101e', scl='#b5aec2', rim='#b6caee', glint='#f2f6ff', lash='#1e152a', mouth='#4f3040', tongue='#936270'),
    'gold': dict(fur='#e68c48', fur_sh='#b8685a', fur_hi='#ffbd70', wht='#fff0d0', wht_sh='#d9b5aa', drk='#57393a', drk_sh='#432d3a',
                 ear='#f3caa4', ear_sh='#c89288', nose='#3c282b', line='#5a3229', iris='#b46c22', iris_lo='#f6c25c', iris_hi='#74421e',
                 pupil='#271612', scl='#fbeac6', rim='#fff2c0', glint='#fffef6', lash='#381d1c', mouth='#83453e', tongue='#e0927a'),
}


def fur(P, depth=6., step=14., lean=.6, seed=0, side=1, jit=.5, taper=True):
    """cut a smooth polyline into hand-drawn fur locks: irregular, directional, concave-sided spikes."""
    P = resample(P, 1.5)
    r = np.random.default_rng(seed)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    Ltot = s[-1]
    if Ltot < step * 1.2: return P
    T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9
    k = max(3, int(step / 3)); ker = np.ones(k) / k
    if len(T) > 2 * k:
        T = np.c_[np.convolve(np.pad(T[:, 0], k, 'edge'), ker, 'same')[k:-k], np.convolve(np.pad(T[:, 1], k, 'edge'), ker, 'same')[k:-k]]
        T /= np.hypot(*T.T)[:, None] + 1e-9
    N = np.c_[T[:, 1], -T[:, 0]] * side
    f = lambda arr, q: np.array([np.interp(q, s, arr[:, 0]), np.interp(q, s, arr[:, 1])])
    out = [P[0]]
    q = 0.; i = 0
    while q < Ltot - step * .35:
        big = r.random() < .42
        st = step * (1.25 if big else .62) * (1 + r.uniform(-jit, jit) * .6)
        q2 = min(Ltot, q + st)
        e = 1.0 if not taper else min(1., min(q + st * .5, Ltot - q - st * .5) / (Ltot * .12 + 1e-6) + .3)
        d = depth * (1.2 if big else .6) * r.uniform(.75, 1.15) * max(e, .2)
        a, b = f(P, q), f(P, q2)
        tm, nm = f(T, (q + q2) / 2), f(N, (q + q2) / 2)
        tip = f(P, q + st * .62) + nm * d + tm * d * lean
        c1 = f(P, q + st * .34) + nm * d * .18
        c2 = f(P, q + st * .80) + nm * d * .10 + tm * d * lean * .25
        for u in (.3, .6, .85):
            out.append(a * (1 - u) ** 2 + 2 * c1 * u * (1 - u) + tip * u * u)
        out.append(tip)
        for u in (.35, .7):
            out.append(tip * (1 - u) ** 2 + 2 * c2 * u * (1 - u) + b * u * u)
        out.append(b)
        q = q2
    out.append(P[-1])
    return np.array(out)


def _proj(P, yaw, pitch, roll, s, cx, cy):
    P = np.asarray(P, float)
    if P.shape[-1] == 2: P = np.concatenate([P, np.zeros(P.shape[:-1] + (1,))], -1)
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    cyw, syw = math.cos(yaw), math.sin(yaw)
    x1 = x * cyw + z * syw; z1 = z * cyw - x * syw
    cp, sp = math.cos(pitch), math.sin(pitch)
    y1 = y * cp - z1 * sp; z2 = z1 * cp + y * sp
    k = 1 + z2 * .00045
    x1, y1 = x1 * k, y1 * k
    c, si = math.cos(roll), math.sin(roll)
    return np.stack([(x1 * c - y1 * si) * s + cx, (x1 * si + y1 * c) * s + cy], -1)


def head(cel, cx, cy, s=3.0, yaw=0.35, pitch=0.0, roll=0.0, pal='lantern', ear_l=(0, 0), ear_r=(0, 0), blink=0.0,
         look=(0, 0), sniff=0.0, brow=0.0, mouth=0.0, pupil=1.0, key=1, rim=True, seed=0, lw=3.2, wide=0.0, lantern=True, wind=0.0, under=True, rim_k=None, fseed=7):
    """ear_x = (swivel -1..1, flatten 0..1); key=+1 light from right, -1 from left; brow: + worried.
    under=True: lantern-from-below shadow design (top of skull in shade); False: side light."""
    C = PAL[pal] if isinstance(pal, str) else pal
    r = np.random.default_rng(4000 + seed)

    def B(P):
        P = np.asarray(P, float).copy()
        P[:, :2] += r.normal(0, .32, (len(P), 2)); return P
    pr = lambda P: _proj(P, yaw, pitch, roll, s, cx, cy)
    L = C['line']
    sn = sniff
    wv = wind
    rk = -key if rim_k is None else rim_k

    def ear(side):
        sw, fl = (ear_l if side < 0 else ear_r)
        k = 1 - .26 * abs(sw)
        e = lambda y: np.clip(-y / 128., 0, 2) ** 1.3
        sh = sw * 22

        def place(q, z, depth=.10):
            q = np.asarray(q, float).copy()
            q[:, 0] = q[:, 0] * k + sh * e(q[:, 1])
            q = q * [1 - .10 * fl, 1 - .16 * fl]
            a = .34 + fl * 1.0 + .08 * sw
            X = q[:, 0] * math.cos(a) - q[:, 1] * math.sin(a); Y = q[:, 0] * math.sin(a) + q[:, 1] * math.cos(a)
            return np.c_[side * (X + 60 + 8 * fl), Y - 52 + 12 * fl, z - np.clip(-q[:, 1], 0, 200) * (depth + .55 * fl)]
        out = [[40, 16], [47, -18], [46, -56], [35, -94], [17, -124], [8, -130], [0, -126], [-16, -100], [-29, -66], [-37, -30], [-40, 6]]
        cp = pr(B(place(out, 0)))
        shape = spl(cp, n=9)
        cel.fill(shape, C['fur'])
        cel.clip(shape)
        # dark tip, fur-cut break
        brk = spl(pr(place([[62, -74], [44, -74], [22, -88], [2, -92], [-18, -84], [-50, -76]], 4)))
        brk = fur(brk, 9 * s, 13 * s, .1, fseed + 5, side=-side)
        tip = np.vstack([brk, pr(place([[-60, -180], [60, -180]], 4))])
        cel.fill(tip, C['drk'])
        # inner opening
        kk = 1 - .62 * abs(sw)
        inn = np.array([[27, 12], [33, -20], [32, -54], [23, -86], [10, -108], [4, -112], [-6, -92], [-15, -60], [-20, -26], [-20, 10]], float)
        inn[:, 0] = inn[:, 0] * kk + sw * -6
        inn[:, 0] *= 1 - .3 * fl
        inner = spl(pr(place(inn, 6)), n=9)
        cel.fill(inner, C['ear'])
        cel.clip(inner)
        shx = np.array([[-26, 14], [-12, -20], [-4, -58], [6, -96], [8, -120], [-30, -110]], float) if side * key > 0 else \
            np.array([[34, 14], [14, -18], [16, -56], [12, -92], [6, -120], [44, -110]], float)
        shx[:, 0] = shx[:, 0] * kk + sw * -6
        cel.fill(spl(pr(place(shx, 7)), closed=True, corners=(0, 4, 5)), C['ear_sh'])
        cel.unclip()
        # fluff from the inner base
        for j, (bx, tx, ty, w_) in enumerate([(-18, -10, -38, 9), (16, 22, -44, 9), (-9, -3, -60, 9), (9, 13, -66, 9), (0, 6, -78, 8)]):
            t = np.array([[bx - w_, 16], [bx - w_ * .5, ty * .45], [tx, ty], [bx + w_ * .7, ty * .4], [bx + w_, 16]], float)
            t[:, 0] = t[:, 0] * kk + sw * -6
            tp = spl(pr(B(place(t, 9))), corners=(2,))
            cel.fill(tp, mix(rgb(C['wht']), rgb(C['ear']), .25) if j > 1 else mix(rgb(C['wht_sh']), rgb(C['ear']), .35))
            cel.line(spl(pr(B(place(t[1:4], 9))), corners=(1,)), lw * .4, L, (.5, .5), seed=seed + j + 31, alpha=.42)
        # whole-ear shadow on the side away from the key
        if side * key < 0 or under:
            es = np.array([[70, 24], [30, 14], [38, -20], [38, -58], [28, -94], [8, -134], [70, -150]], float)
            cel.fill(fur(spl(pr(place(es, 1)), closed=True, corners=(0, 5, 6)), 2 * s, 14 * s, .3, fseed + 3), C['drk_sh'] if False else C['fur_sh'], .8)
            cel.fill(spl(pr(place([[60, -78], [40, -78], [32, -96], [10, -134], [60, -150]], 2)), closed=True, corners=(0, 1, 3, 4)), C['drk_sh'], .9)
        cel.unclip()
        cel.line(inner, lw * .5, L, (.4, .4), seed=seed + 7 + side, alpha=.6)
        cel.line(shape, lw * 1.2, L, (.12, .1), press=.35, seed=seed + 11 + side)
        if rim and side * rk > 0:
            rp = place([[41, 2], [50, -20], [49, -57], [38, -95], [20, -125], [8, -131], [16, -116], [32, -90], [43, -55], [43, -18]], 0)
            cel.shade(spl(pr(rp), closed=True, corners=(0, 5)), C['rim'], within=shape, alpha=.9)

    far = 1 if yaw > 0 else -1
    ear(far); ear(-far)

    # -------------------------------------------------- silhouette
    ws = np.clip(wv, -1.6, 1.6)
    R = [[0, -86, 50], [28, -84, 40], [58, -72, 22], [86, -48, 2], [102, -18, -10], [109, 8, -16],
         [116 + ws * 1.5, 28, -18], [130 + ws * 4, 52 + abs(ws), -22], [114, 46, -14], [118 + ws * 4, 69, -18], [105, 62, -8], [104 + ws * 5, 93, -8],
         [89, 79, 2], [82 + ws * 3, 99, 4], [66, 91, 18], [44, 102, 44], [22, 112, 66], [0, 116, 76]]
    n2 = len(R)
    Lf = [[-x, y, z] for x, y, z in R[::-1]][1:-1]
    full = np.array(R + Lf, float)
    n = len(full)
    # the two cheeks are drawn, not mirrored
    for i, (dx, dy) in {7: (5, -6), 9: (-4, 4), 11: (3, 5), 13: (-4, -5)}.items(): full[n - i, :2] += (dx, dy)
    if ws: full[n2:, 0] += ws * 5
    cidx = [7, 9, 11, 13, n - 7, n - 9, n - 11, n - 13]
    fb = B(full)
    sil = spl(pr(fb), closed=True, corners=cidx, n=8)
    cel.fill(sil, C['fur'])
    cel.clip(sil)

    # -------------------------------------------------- white mask with fur-cut edge
    masks = []
    for sd in (-1, 1):
        edge = np.array([[0, 46, 98], [11, 30, 90], [17, 20, 82], [27, 28, 78], [38, 38, 70], [54, 43, 60], [72, 38, 46], [86, 26, 28], [100, 10, 4], [118, 14, -14]], float) * [sd, 1, 1]
        e = spl(pr(edge), n=8)
        e1 = fur(e[int(len(e) * .28):], 6.2 * s, 13 * s, .9, fseed + 41 + sd, side=1 if sd > 0 else -1)
        e = np.vstack([e[:int(len(e) * .28)], e1])
        m = np.vstack([e, pr(np.array([[170 * sd, 40, -20], [150 * sd, 160, 0], [0, 160, 50], [0, 100, 90]], float))])
        masks.append(m)
        cel.fill(m, C['wht'])
        bsp = np.array([[25, -32 - brow * 3, 68], [33, -42 - brow * 4, 62], [43, -41 - brow * 2, 58], [47, -33, 58], [37, -28 - brow, 64]], float) * [sd, 1, 1]
        cel.fill(fur(spl(pr(bsp), closed=True), 2.2 * s, 9 * s, .3, fseed + 44), mix(rgb(C['wht']), rgb(C['fur_hi']), .75))

    # -------------------------------------------------- cel shadow
    kx = -key
    if under:
        edge = np.array([[kx * 150, 92, 0], [kx * 104, 80, 0], [kx * 92, 62, 8], [kx * 94, 44, 10], [kx * 86, 22, 24], [kx * 83, 0, 36], [kx * 82, -20, 40],
                         [kx * 70, -38, 46], [kx * 50, -50, 54], [kx * 26, -58, 60], [0, -62, 64], [-kx * 30, -60, 58], [-kx * 56, -54, 44], [-kx * 80, -44, 24], [-kx * 110, -30, 0]], float)
    else:
        edge = np.array([[kx * 70, 150, 0], [kx * 78, 92, 8], [kx * 84, 74, 10], [kx * 86, 52, 14], [kx * 78, 30, 32], [kx * 76, 4, 42], [kx * 70, -18, 48],
                         [kx * 58, -40, 52], [kx * 40, -60, 52], [kx * 22, -78, 52], [kx * 8, -100, 48]], float)
    e = fur(spl(pr(edge), n=8), 8.0 * s, 19 * s, .7, fseed + 51, side=1)
    cap = pr(np.array([[-kx * 160, -160, 0], [kx * 190, -160, 0], [kx * 190, 160, 0]], float)) if under else pr(np.array([[kx * 190, -160, 0], [kx * 190, 160, 0]], float))
    shp = np.vstack([e, cap])
    cel.clip(shp)
    cel.fill(sil, C['fur_sh'])
    for m in masks: cel.fill(m, C['wht_sh'])
    cel.unclip()
    # chin shadow / under-cheek
    ch = np.array([[-96, 92, 0], [-70, 84, 16], [-44, 92, 44], [-24, 99, 62], [0, 101, 70], [24, 99, 62], [44, 92, 44], [70, 84, 16], [96, 92, 0], [60, 150, 10], [-60, 150, 10]], float)
    cel.fill(fur(spl(pr(ch), closed=True, corners=(0, 8, 9, 10)), 4 * s, 14 * s, .4, fseed + 53), C['wht_sh'], .9)

    # -------------------------------------------------- muzzle
    my = -sn * 5
    fr_ = float(np.clip(abs(yaw) / .22, 0, 1))                  # 0 frontal .. 1 clearly turned
    nside = -far                                                # the side of the muzzle turned toward us
    # form shadow where the muzzle leaves the near cheek (or the side away from the light when frontal)
    ss_ = kx
    q = np.array([[ss_ * 20, 44, 94], [ss_ * 26, 54 + my, 100], [ss_ * 33, 65 + my, 98], [ss_ * 36, 76, 92], [ss_ * 31, 88, 90], [ss_ * 40, 83, 76], [ss_ * 45, 70, 72], [ss_ * 40, 57, 76], [ss_ * 29, 46, 86]], float)
    e = spl(pr(q[5:]), n=8)
    e = fur(e, 2.6 * s, 9 * s, .4, fseed + 57, side=ss_)
    cel.fill(np.vstack([spl(pr(q[:6]), n=8), e]), mix(rgb(C['wht']), rgb(C['wht_sh']), .62))
    for sd in (-1, 1):
        tear = np.array([[20, 17, 76], [16, 28, 86], [12, 40 + my * .5, 98], [9, 48 + my, 108], [14, 41 + my * .5, 96], [21, 29, 84], [25, 20, 74]], float) * [sd, 1, 1]
        cel.fill(spl(pr(tear), closed=True, corners=(3,)), mix(rgb(C['fur_sh']), rgb(C['drk']), .25), .75)
    cel.unclip()
    # far contour of the muzzle, over the far cheek
    if fr_ > .25:
        fm = np.array([[19, 24, 80], [25, 38, 92], [33, 52 + my * .6, 102], [38, 65 + my * .6, 104], [37, 78, 100], [29, 90, 94], [14, 98, 98]], float) * [far, 1, 1]
        fp_ = spl(pr(B(fm)), n=8)
        cel.line(fp_[int(len(fp_) * .16):], lw * .95, L, (.45, .5), seed=seed + 61, alpha=.5 + .5 * fr_)
        nm = np.array([[30, 84, 88], [22, 93, 92], [12, 98, 96]], float) * [nside, 1, 1]
        cel.line(spl(pr(B(nm))), lw * .6, L, (.5, .6), seed=seed + 62, alpha=.55)
    else:
        for sd in (-1, 1):
            mz = np.array([[34, 64 + my, 84], [37, 76, 82], [30, 89, 86], [17, 97, 92]], float) * [sd, 1, 1]
            cel.line(spl(pr(B(mz))), lw * .6, L, (.6, .6), seed=seed + 63 + sd, alpha=.6)
    if sn > .15:
        for sd in (-1, 1):
            for j in range(2):
                wr = np.array([[12 + j * 4, 38 + j * 6 + my, 98], [18 + j * 3, 35 + j * 6 + my, 92], [23 + j * 2, 38 + j * 6 + my, 86]], float) * [sd, 1, 1]
                cel.line(spl(pr(wr)), lw * .45, L, (.5, .5), seed=seed + 66 + j, alpha=.7 * sn)
    mo = mouth
    mth = np.array([[-33, 76, 82], [-27, 83, 90], [-12, 87, 106], [0, 80, 116], [12, 87, 106], [27, 83, 90], [33, 76, 82]], float)
    if mo > .05:
        low = np.array([[24, 84 + 5 * mo, 92], [14, 92 + 13 * mo, 98], [0, 94 + 16 * mo, 100], [-14, 92 + 13 * mo, 98], [-24, 84 + 5 * mo, 92]], float)
        msh = spl(pr(np.vstack([mth[1:-1], low])), closed=True, corners=(0, 2, 4), n=8)
        cel.fill(msh, C['mouth'])
        tg = np.array([[-13, 99 + 6 * mo, 98], [0, 91 + 8 * mo, 100], [13, 99 + 6 * mo, 98], [0, 100 + 19 * mo, 98]], float)
        cel.shade(spl(pr(tg), closed=True), C['tongue'], within=msh)
        cel.line(spl(pr(B(low))), lw * .75, L, (.2, .2), seed=seed + 71)
        ch_ = np.array([[16, 99 + 13 * mo, 92], [0, 103 + 17 * mo, 94], [-16, 99 + 13 * mo, 92]], float)
        cel.line(spl(pr(B(ch_))), lw * .55, L, (.5, .5), seed=seed + 74, alpha=.6)
    # mouth: the near half is long, the far half tucks round the muzzle
    for sd in (-1, 1):
        half = mth[3:] if sd > 0 else mth[:4][::-1]
        k_ = 4 if (sd == nside or fr_ < .3) else 3
        cel.line(spl(pr(B(half[:k_]))), lw * .85, L, (.08, .55), seed=seed + 72 + sd)
    cel.line(pr(np.array([[0, 72 + my, 120], [0, 76, 119], [0, 80, 116]], float)), lw * .75, L, (.1, .3), seed=seed + 75)
    ny = 60 + my * 1.3; nw = 1.15 + .10 * sn
    nose = np.array([[0, ny - 10, 120], [11 * nw, ny - 9, 117], [18 * nw, ny - 3, 112], [16 * nw, ny + 5, 114], [7, ny + 11, 120], [0, ny + 13, 121],
                     [-7, ny + 11, 120], [-16 * nw, ny + 5, 114], [-18 * nw, ny - 3, 112], [-11 * nw, ny - 9, 117]], float)
    ns = spl(pr(B(nose)), closed=True)
    cel.fill(ns, C['nose'])
    hl = np.array([[-8, ny - 6, 120], [-1, ny - 7, 121], [5, ny - 5.5, 121], [0, ny - 4, 121], [-6, ny - 4, 121]], float) + [4 * key, 0, 0]
    cel.fill(spl(pr(hl), closed=True), mix(rgb(C['glint']), rgb(C['nose']), .5), .9)
    for sd in (-1, 1):
        nos = np.array([[6, ny + 2, 121], [11 * nw, ny + 1, 118], [12 * nw, ny + 4, 117], [7, ny + 5, 120]], float) * [sd, 1, 1]
        cel.fill(spl(pr(nos), closed=True), mix(rgb(C['nose']), np.zeros(3), .6))
        for (x, y) in ((22, 70), (27, 75)):
            p = pr(np.array([[x * sd, y + my, 98]], float))[0]
            d = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]]) * lw * .28 + p
            cel.fill(spl(d, closed=True), L, .45)

    # -------------------------------------------------- eyes
    def eye(sd):
        ex, ey, ez = 50 * sd, 2, 64
        op = 1 + .09 * wide
        top = np.array([[-24, 17], [-22, 2], [-13, -14], [0, -20.5], [13, -18], [23, -10], [29, 0]], float)
        bot = np.array([[29, 0], [27, 11], [17, 21], [3, 25], [-11, 24], [-20, 21], [-24, 17]], float)
        top[:, 1] *= op; bot[2:5, 1] += 1.2 * wide
        ke = .90; top *= ke; bot *= ke
        top[1:4, 1] -= brow * np.array([1.5, 2.2, 1.2]); top[4:, 1] += brow * np.array([1, 2.2, 2.0])
        clo = np.array([[-23, 16], [-16, 18.5], [-8, 20.5], [2, 21], [12, 19], [20, 15], [27, 8]], float)
        t = top * (1 - blink) + clo * blink

        def pl(q):
            q = np.asarray(q, float)
            return pr(np.c_[ex + sd * q[:, 0], ey + q[:, 1], ez - (q[:, 0] + 25) * .24 - ((q[:, 0] + 25) / 30) ** 2 * 2.2])
        opening = np.vstack([spl(pl(t)), spl(pl(bot))])
        lit = sd * key > 0
        if blink < .93:
            cel.clip(opening)
            cel.fill(opening, C['scl'])
            icx, icy = 2.0 + look[0] * 5 * sd, 1.5 + look[1] * 5
            th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            E = lambda ax, ay, dx=0, dy=0: pl(np.c_[icx + dx + ax * np.cos(th), icy + dy + ay * np.sin(th)])
            iris = E(20.2, 21.5)
            cel.fill(iris, C['iris'])
            cel.shade(E(16.5, 10.5, 0, 11.5), C['iris_lo'], within=iris)
            cel.shade(E(26, 13, 0, -15.5), C['iris_hi'], within=iris)
            cel.line(np.vstack([iris, iris[:1]]), lw * .8, C['iris_hi'], (0, 0), seed=seed + 81, tremor=.15)
            cel.fill(E(10.8 * pupil, 14.0 * pupil, 0, -.5), C['pupil'])
            lidsh = np.vstack([spl(pl(t + [0, -20])), spl(pl(t + [0, 6.5]))[::-1]])
            cel.fill(lidsh, mix(rgb(C['iris_hi']), rgb(C['pupil']), .6), .72)
            gx = -1 if key > 0 else 1                     # sky / moon light side in screen space
            cel.fill(E(4.9, 5.8, gx * sd * 6.6, -7.6), C['glint'])
            if lantern:
                cel.fill(E(3.6, 2.7, key * sd * 7.2, 9.2), '#ffd489')
            else:
                cel.fill(E(2.1, 1.7, key * sd * 7.2, 9.0), mix(rgb(C['iris_lo']), rgb(C['glint']), .55))
            cel.unclip()
        else:
            cel.fill(np.vstack([spl(pl(top)), spl(pl(bot))]), C['fur'])
        if blink > .05:
            lidp = np.vstack([spl(pl(top + [0, -1.5])), spl(pl(t))[::-1]])
            cel.fill(lidp, C['fur'])
            cel.line(spl(pl(top[1:6])), lw * .45, L, (.5, .5), seed=seed + 84, alpha=.4 * min(1, blink * 1.5))
        # dark rim: heavy above, tapering to both corners; short tear stroke at the inner corner
        lash = np.vstack([t, t[-1:] + [[3.5, -1 + 5 * blink]]])
        wf = lambda q: .35 + 1.0 * np.sin(np.pi * np.clip(q * .9 + .05, 0, 1)) ** .7
        cel.line(spl(pl(lash)), lw * (2.1 if blink < .9 else 1.6), C['lash'], (.12, .2), seed=seed + 85 + sd, wfun=wf, tremor=.2)
        cel.line(spl(pl(np.array([[-22, 12], [-26, 18], [-29, 25], [-31, 31]], float))), lw * 1.5, C['lash'], (.05, .75), seed=seed + 88)
        if blink < .9:
            cel.line(spl(pl(bot)), lw * .85, C['lash'], (.3, .3), seed=seed + 91, alpha=.9)
        # upper-lid light fur patch edge (short broken strokes, not a human crease)
        for (a, b) in (((-14, -21), (-4, -26)), ((4, -27), (14, -24))):
            q = np.array([a, ((a[0] + b[0]) / 2, min(a[1], b[1]) - 1.2), b], float) + [0, -1 - brow * 2]
            cel.line(spl(pl(q)), lw * .45, L, (.5, .5), seed=seed + 93, alpha=.45)
    eye(far); eye(-far)

    # -------------------------------------------------- ink
    def contour(a, b, w=1.0, tap=(.12, .12)):
        idx = [k % n for k in range(a, b + 1)]
        cs = [k for k, q in enumerate(idx) if q in cidx]
        cel.line(spl(pr(fb[idx]), corners=cs, n=8), lw * w * 1.2, L, tap, press=.35, seed=seed + 100 + a)
    contour(4, 7, 1.05, (.4, .05)); contour(7, 9, .95, (.05, .06)); contour(9, 11, .95, (.05, .06)); contour(11, 13, .9, (.05, .1)); contour(13, 14, .7, (.05, .8))
    contour(n - 7, n - 4, 1.05, (.05, .4)); contour(n - 9, n - 7, .95, (.06, .05)); contour(n - 11, n - 9, .95, (.06, .05)); contour(n - 13, n - 11, .9, (.1, .05)); contour(n - 14, n - 13, .7, (.8, .05))
    contour(16, 18, .8, (.55, .55))
    for (x, y, dx, dy) in ((58, 93, 6, 9), (-57, 94, -5, 10), (40, 101, 3, 8), (-41, 102, -3, 9)):      # short locks instead of a jaw line
        t = np.array([[x, y, 24], [x + dx * .5, y + dy * .55, 22], [x + dx, y + dy, 20]], float)
        cel.line(spl(pr(B(t))), lw * .65, L, (.4, .35), seed=seed + 121 + x, alpha=.7)
    contour(1, 2, .7, (.5, .5)); contour(n - 2, n - 1, .7, (.5, .5))
    # crown: three soft fur points breaking the skull line
    for j, (a, tx, ty, w_) in enumerate(((-9, -12, -97, 7), (1, 3, -103, 8), (10, 18, -97, 7))):
        tp = np.array([[a - w_, -82, 50], [a - w_ * .4, -90, 50], [tx + wv * 12, ty, 48], [a + w_ * .5, -89, 50], [a + w_, -81, 50]], float)
        tp = spl(pr(B(tp)), corners=(2,))
        cel.fill(tp, C['fur_sh'] if under else C['fur'])
        cel.line(tp[3:-3], lw * .75, L, (.4, .4), seed=seed + 130 + j)
    for sd in (-1, 1):
        for (x, y, dx, dy, z) in ((96, 48, 14, 4, 12), (88, 66, 15, 8, 12), (74, 80, 12, 7, 20), (104, 28, 13, 2, 4)):
            t = np.array([[x * sd, y, z], [(x + dx * .55) * sd, y + dy * .15, z - 4], [(x + dx) * sd, y + dy, z - 9]], float)
            cel.line(spl(pr(B(t))), lw * .55, L, (.5, .25), seed=seed + 141 + x, alpha=.55)
    for (x, y) in ((-9, -62), (3, -56), (12, -63), (-2, -46)):
        t = np.array([[x, y, 58], [x + 1, y + 5, 60], [x, y + 10, 62]], float)
        cel.line(spl(pr(t)), lw * .45, L, (.5, .5), seed=seed + 150 + x, alpha=.4)

    if rim:
        rp = np.array([[rk * 84, -50, 2], [rk * 102, -18, -10], [rk * 109, 8, -16], [rk * 116, 28, -18], [rk * 128, 50, -22],
                       [rk * 114, 33, -16], [rk * 106, 12, -13], [rk * 100, -17, -6], [rk * 83, -46, 3]], float)
        cel.shade(spl(pr(rp), closed=True, corners=(0, 4)), C['rim'], within=sil, alpha=.85)
        rp2 = np.array([[rk * 2, -87, 50], [rk * 28, -85, 40], [rk * 56, -74, 22], [rk * 30, -79, 42], [rk * 12, -83, 48]], float)
        cel.shade(spl(pr(rp2), closed=True, corners=(0, 2)), C['rim'], within=sil, alpha=.8)

    for sd in (-1, 1):
        for j, (y0, dx, dy) in enumerate(((71, 92, -14), (76, 100, 6), (80, 84, 24))):
            q = np.array([[24 * sd, y0 + my, 88], [(24 + dx * .5) * sd, y0 + dy * .25 + my - 5 + r.normal(0, 1), 60], [(24 + dx) * sd, y0 + dy + my * 2 + wv * 10, 20]], float)
            cel.line(spl(pr(q)), lw * .42, mix(rgb(C['wht']), rgb(C['line']), .4), (.02, .7), seed=seed + 160 + j, alpha=.7, press=.1)
    return sil
