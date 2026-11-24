"""s6 The Pages Burst (168 f).  6a the shelves explode, 6b one page folds itself into a bird, 6c the flock breaks the dome.
Owner: flock lane.  Sets live in shots/s6_set.py, paper in lp/pages.py."""
import os, math
import numpy as np, cv2
from lp import W, H, comp
from lp import pages as PG
from lp.paint import rgb, mix, f32, fbm, smooth, lum
from lp.ink import Cel, spl, over_roi
from lp.cam import Camera3D
from lp.pages import fill_polys, edge_quads, hash01
from shots import s6_set as ST

VER = 'v8'
CACHE = os.path.join(os.path.dirname(__file__), '..', 'cache')


def cached(name, fn):
    p = os.path.join(CACHE, f's6_{name}_{VER}.npy')
    if os.path.exists(p):
        try: return np.load(p)
        except Exception: pass
    a = fn(); t = p + '.%d.npy' % os.getpid(); np.save(t, a); os.replace(t, p)
    return a


def u16(a): return (np.clip(a, 0, 1.6) * 40000).astype(np.uint16)


def shake(f, t0, amp, dur, seed=0, step=2):
    """drawn jolts held on twos, decaying"""
    if f < t0 or f >= t0 + dur: return np.zeros(2)
    k = (f - t0) // step
    r = np.random.default_rng(seed * 97 + k); a = r.uniform(0, 6.28)
    return np.array([math.cos(a), math.sin(a) * .8]) * amp * (1 - (f - t0) / dur) ** 1.3


# ================================================================================================ 6c the vortex
class Vortex:
    """thousands of birds and loose pages on rising helical arms, in family groups that peel off the shelves"""
    F0, F1 = 78, 168

    def __init__(s, seed=3):
        r = np.random.default_rng(seed)
        f = np.arange(0, 240)
        v = 5.0 + 9.5 * np.clip((f - s.F0) / 90., 0, 1.2) ** 1.5 + 3.0 * np.clip((f - 120) / 20., 0, 1)
        s.Z = np.cumsum(v / 24.); s.Z -= s.Z[s.F0]
        om = .75 + 1.1 * np.clip((f - s.F0) / 90., 0, 1.2)
        s.Om = np.cumsum(om / 24.); s.Om -= s.Om[s.F0]
        span = s.Z[s.F1] + 30
        P = dict(z0=[], arm=[], dth=[], rho=[], join=[], size=[], seed=[], grp=[], kind=[], ph=[], lag=[])
        g = 0
        n_tot = 0
        while n_tot < 11500:
            m = int(np.clip(r.lognormal(2.5, .8), 3, 70)); kind = 1 if r.random() < .07 else 0       # group of loose pages or of birds
            z0 = r.uniform(-s.Z[s.F1] - 3, 26); arm = int(r.integers(0, 3)); a0 = r.normal(0, .125)
            if r.random() < .035: a0 = r.uniform(-1.0, 1.0)
            rho0 = np.clip(r.normal(2.35, .42), .9, 3.5)
            des = int(r.integers(0, len(PG.NAMES))); dm = r.random() < .55
            ln, wd = .17 * m ** .75, .040 * m ** .35
            jz = np.clip(r.normal(6.6, 3.6), .3, 21)
            for q in range(m):
                P['z0'].append(z0 + r.normal(0, 1) * ln * .55); P['arm'].append(arm); P['dth'].append(a0 + r.normal(0, wd)); P['rho'].append(np.clip(rho0 + r.normal(0, .24), .45, 3.75))
                P['join'].append(jz + r.normal(0, .8)); P['size'].append((r.uniform(.13, .235) if r.random() < .94 else r.uniform(.26, .36)))
                P['seed'].append(int(r.integers(1, 1 << 28))); P['grp'].append(g); P['kind'].append(kind if r.random() < .93 else 1 - kind)
                P['ph'].append(int(r.integers(0, 3)) + g); P['lag'].append(des if dm and r.random() < .7 else int(r.integers(0, len(PG.NAMES))))
            g += 1; n_tot += m
        for k, v in P.items(): setattr(s, k, np.array(v))
        s.n = len(s.z0)

    def lim(s, f):
        """the shut dome: the current piles up beneath it, then is let go"""
        return 18.2 - 1.8 * float(np.clip((f - 98) / 10., 0, 1)) if f < 121 else 16.4 + (f - 120.) ** 2.0

    def path(s, f, idx=slice(None)):
        Zf = np.interp(f, np.arange(len(s.Z)), s.Z); Of = np.interp(f, np.arange(len(s.Om)), s.Om)
        z = s.z0[idx] + Zf
        l = s.lim(f); k0 = l - 4.5
        zz = np.where(z > k0, k0 + 4.5 * (1 - np.exp(-(z - k0) / 4.5 * .55)) if f < 121 else k0 + (z - k0) / (1 + np.maximum(0, (z - k0)) / 30.), z)
        th = s.arm[idx] * 2 * math.pi / 3 + 3.5 * np.log(np.maximum(zz, 0) + 2.3) + Of + s.dth[idx] * (1 + .9 * smooth(13, 19, zz))
        # slow breathing of every group, so arms are never clean tubes
        rho = s.rho[idx] * (1 + .10 * np.sin(zz * .8 + s.grp[idx] * 1.7)) * (1 - .24 * smooth(10, 22, zz))
        # the crowd recoils into a crown while the shutters strain
        rc = float(np.clip((f - 100) / 10., 0, 1) * np.clip((127 - f) / 6., 0, 1)); rho = rho + (np.maximum(rho, 2.9 + .45 * np.sin(s.grp[idx] * 2.1)) - rho) * rc * smooth(8, 13.5, zz)
        jn = np.clip((s.join[idx] - zz) / 1.6, 0, 1); rho = rho + (4.7 - rho) * jn ** 1.4; th = th - jn * .22
        z2 = zz + .22 * np.sin(zz * 1.9 + s.grp[idx] * 2.3) - jn * .3
        zz = np.where((f < 121) & (z - k0 > 5.5), -99, zz)
        return np.stack([rho * np.cos(th), rho * np.sin(th), z2], -1), zz


class Six:
    EYE = np.array([.9, -.6, 0.])

    def ez(s, f):
        t = np.clip((f - 78) / 72., 0, 1); e = 1.5 * t ** 1.6
        if f > 146: e += 6.2 * ((f - 146) / 22.) ** 2.4
        return e

    def cam(s, f):
        e = s.EYE + [0, 0, s.ez(f)]
        roll = math.radians(-8 + 15 * np.clip((f - 78) / 90., 0, 1) ** 1.25)
        foc = float(comp.lerp_keys(f, [(111, 1000), (117, 1290), (121, 1340), (124, 975), (129, 1000)])) + (330 * ((f - 148) / 20.) ** 2 if f > 148 else 0)
        sh = shake(f, 108, 1.6, 10, 5) + shake(f, 116, 6.0, 16, 8) + shake(f, 126, 2.5, 8, 9)
        return Camera3D(e, e + np.array([.03, .12, 1.0]), foc, W, H, 1180 + sh[0] - 60 * np.clip((f - 140) / 28., 0, 1) ** 2, 606 + sh[1] + 26 * np.clip((f - 140) / 28., 0, 1) ** 2, roll)


PLATE = (2800, 1900, 1250., 1585., 965.)


def plate_cam():
    return Camera3D(Six.EYE, Six.EYE + np.array([.03, .12, 1.0]), PLATE[2], PLATE[0], PLATE[1], PLATE[3], PLATE[4])


def cyl_hit(cam, x, y):
    """world point where pixel rays meet the shelf wall"""
    d = np.stack([(x - cam.cx) / cam.f, (y - cam.cy) / cam.f, np.ones_like(x)], -1) @ cam.R
    e = cam.eye
    a = d[..., 0] ** 2 + d[..., 1] ** 2 + 1e-9; b = d[..., 0] * e[0] + d[..., 1] * e[1]; c = e[0] ** 2 + e[1] ** 2 - ST.R0 ** 2
    t = (-b + np.sqrt(np.maximum(b * b - a * c, 0))) / a
    return e + d * t[..., None]


def opened(dome, f):
    """how far sunlight has come down each dome sector (metres of path), in held steps"""
    steps = [(-99, 0.), (0, 5.), (2, 12.), (4, 18.5), (6, 25.), (8, 40.)]
    out = np.zeros(ST.NP_)
    for k in range(ST.NP_):
        t0 = max(dome.tb[k, 1], 119) + (k * 7 % 4) - 1
        t0 = max(t0, 120)
        out[k] = comp.keyed(f - t0, steps)
    return out


class Shot:
    id = 's6'; n = 168
    finish = dict(hal_amt=.24, vig=.12)

    def setup(s):
        os.makedirs(CACHE, exist_ok=True)
        s.st, s.dome, s.six = ST.Stairs(), ST.Dome(), Six()
        s.vx = Vortex()
        names = ['N', 'S', 'G']
        if not all(os.path.exists(os.path.join(CACHE, f's6_w{n}_{VER}.npy')) for n in names):
            Q = ST.build_well(); cam = plate_cam()
            o = ST.paint_quads(cam, PLATE[0], PLATE[1], 1.5, Q, ST.shade_well, names, {'N': rgb('#24223e'), 'S': rgb('#5a5490'), 'G': rgb('#a0744c')}, extra=ST.paint_ivy)
            for n in names: cached('w' + n, lambda: u16(o[n]))
        s.well = {n: np.load(os.path.join(CACHE, f's6_w{n}_{VER}.npy')) for n in names}
        lm = cached('lit', s._litmaps)
        s.M0, s.Lm, s.Km = lm[0].astype(f32), lm[1].astype(f32), lm[2].astype(f32)
        s.setup_a() if hasattr(s, 'setup_a') else None
        s.setup_b() if hasattr(s, 'setup_b') else None

    def _litmaps(s):
        cam = plate_cam()
        y, x = np.mgrid[0:PLATE[1], 0:PLATE[0]].astype(np.float64)
        P = cyl_hit(cam, x, y)
        rr, k, rib, L = ST.sun_hit(P)
        m = ((rr < ST.RO) & (rr > .5) & (rib > .13)).astype(f32)
        nrm = -(np.cos(np.arctan2(P[..., 1], P[..., 0])) * ST.SUN[0] + np.sin(np.arctan2(P[..., 1], P[..., 0])) * ST.SUN[1])
        m *= (nrm > .02)
        idx = np.nonzero(m)
        sh = np.zeros(len(idx[0]), bool)
        PP = P[idx]
        for a in range(0, len(PP), 400000): sh[a:a + 400000] = ST.stair_shadow(PP[a:a + 400000])
        m[idx] *= ~sh
        # brushy, hand cut edge
        m = cv2.GaussianBlur(m, (0, 0), 2.0); m = np.clip((m + fbm(PLATE[1], PLATE[0], (26, 9), (1, .6), 7) * .07 - .5) * 2.6 + .5, 0, 1)
        n = fbm(PLATE[1], PLATE[0], (90, 30), (1, .5), 9)
        return np.stack([m, np.clip(L + n * .55, -5, 200), k]).astype(np.float16)

    # -------------------------------------------------------------------------------------------- 6c
    def wall(s, cam, f, kday):
        g = 8
        x = (np.arange(W // g) + .5) * g - .5; y = (np.arange(H // g) + .5) * g - .5
        X, Y = np.meshgrid(x, y)
        P = cyl_hit(cam, X, Y)
        pc = plate_cam()
        uv, z = pc.project(P)
        mx = cv2.resize(uv[..., 0].astype(f32), (W, H), interpolation=cv2.INTER_CUBIC); my = cv2.resize(uv[..., 1].astype(f32), (W, H), interpolation=cv2.INTER_CUBIC)
        rm = lambda a, i=cv2.INTER_LINEAR: cv2.remap(a, mx, my, i, borderMode=cv2.BORDER_REPLICATE)
        img = rm(s.well['N']).astype(f32) / 40000.
        if kday > 0:
            S = rm(s.well['S']).astype(f32) / 40000.
            img = img * (1 - kday) + S * kday
            Lf = opened(s.dome, f)
            if Lf.max() > 0:
                M = rm(s.M0); L = rm(s.Lm); K = rm(s.Km, cv2.INTER_NEAREST).astype(np.int32).clip(0, ST.NP_ - 1)
                fr = np.clip((Lf[K].astype(f32) - L) * 6 + .5, 0, 1)
                M = M * fr
                G = rm(s.well['G']).astype(f32) / 40000.
                img = img + (G - img) * M[..., None]
        return img

    def lit(s, P, Lf, sun=ST.SUN):
        rr, k, rib, L = ST.sun_hit(P, sun)
        return (rr < ST.RO) & (rr > .2) & (rib > .08) & (L < Lf[k])

    def beams(s, img, cam, f, Lf, amt=1.0):
        g = 2
        m = np.zeros((H // g, W // g), f32)
        r = np.random.default_rng(44)
        for k in range(ST.NP_):
            parts = [(.10, .90, .9, 4.6, .13)] + [tuple(sorted(r.uniform(.12, .88, 2))) + (r.uniform(.8, 2.6), r.uniform(3.0, 4.6), r.uniform(.06, .15)) for _ in range(3)]
            if Lf[k] <= 0: continue
            for (a0, a1, r0, r1, al) in parts:
                if a1 - a0 < .05: a1 = a0 + .05
                a = (k + np.array([a0, (a0 + a1) / 2, a1, a1, (a0 + a1) / 2, a0])) * 2 * math.pi / ST.NP_
                rad = np.array([r0, r0, r0, r1, r1, r1])
                Q = np.stack([rad * np.cos(a), rad * np.sin(a), np.full(6, ZD1)], -1)
                # each ray stops at the wall, at the light front or just above the lens
                d = -ST.SUN
                aa = d[0] ** 2 + d[1] ** 2; bb = Q[:, 0] * d[0] + Q[:, 1] * d[1]; cc = Q[:, 0] ** 2 + Q[:, 1] ** 2 - 5.0 ** 2
                tw = (-bb + np.sqrt(np.maximum(bb * bb - aa * cc, 0))) / aa
                tl = (ZD1 - cam.eye[2] - .45) / ST.SUN[2]
                t = np.minimum(np.minimum(tw, Lf[k] * (1 + .04 * math.sin(k * 3.1))), tl)
                E = Q + d * t[:, None]
                v, _ = cam.project(np.concatenate([Q, E]))
                hull = cv2.convexHull((v / g).astype(np.float32))[:, 0]
                lay = np.zeros_like(m); cv2.fillPoly(lay, [np.round(hull * 16).astype(np.int32)], al, cv2.LINE_AA, 4)
                m += lay
        m = cv2.GaussianBlur(m, (0, 0), .8)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        return comp.screen(img, (np.clip(m, 0, .6) * amt)[..., None] * rgb('#ffd98a') * 1.02)

    def vortex_at(s, f, cam, lo, hi, light_day, Lf, img, kday, post=False, skein=None):
        """draw the part of the flock between camera distances lo..hi (far to near)"""
        vx = s.vx
        d = f // 2 * 2
        P, zz = vx.path(d)
        Pn, _ = vx.path(d + 1.0)
        dist = np.linalg.norm(P - cam.eye, axis=1); dz = P[:, 2] - cam.eye[2]
        inside = P[:, 2] < 60
        vis = (dz > .35) & (dist >= lo) & (dist < hi) & inside & (zz > vx.join - 1.6)
        idx = np.nonzero(vis)[0]
        if len(idx) == 0: return img
        # near birds are drawn on ones
        if hi < 5:
            P[idx], _ = vx.path(f, idx); Pn[idx], _ = vx.path(f + 1.0, idx)
        s2, z = cam.project(P[idx]); keep = z > .3
        idx, s2, z = idx[keep], s2[keep], z[keep]
        on = (s2[:, 0] > -400) & (s2[:, 0] < W + 400) & (s2[:, 1] > -400) & (s2[:, 1] < H + 400)
        if post:
            c = cam.project(np.array([0, 0, ST.ZD + 1.]))[0]; on &= (np.hypot(s2[:, 0] - c[0], s2[:, 1] - c[1]) < cam.f * ST.R0 / (ST.ZD - cam.eye[2]) * 1.06) & (P[idx, 2] > 12) & (hash01(vx.seed[idx], 40) < .80 - .55 * float(np.clip((f - 140) / 24., 0, 1)))
        idx, s2, z = idx[on], s2[on], z[on]
        if len(idx) == 0: return img
        if skein is not None and not post and len(skein[0]):        # the followers part round the leading flock so the eye can hold it
            sk2, skr = skein; clear = np.ones(len(idx), bool)
            for q2, r2 in zip(sk2, skr): clear &= ((s2 - q2) ** 2).sum(1) > r2 * r2
            idx, s2, z = idx[clear], s2[clear], z[clear]
            if len(idx) == 0: return img
        vel = Pn[idx] - P[idx]
        v2 = cam.project(Pn[idx])[0] - s2
        px = vx.size[idx] * cam.f / z
        sp = np.hypot(v2[:, 0], v2[:, 1])
        sm = v2 * np.clip((sp / np.maximum(px, 1) - .30) * 1.0, 0, .55)[:, None]
        sm = np.where((px > 34)[:, None], sm, 0)
        rad = P[idx].copy(); rad[:, 2] = 0; rad /= np.linalg.norm(rad, axis=1, keepdims=True) + 1e-9
        tang = np.stack([-rad[:, 1], rad[:, 0], np.zeros(len(rad))], -1)
        h1, h2, h3 = hash01(vx.grp[idx], 71)[:, None], hash01(vx.seed[idx], 72)[:, None], hash01(vx.seed[idx], 73)[:, None]
        head = tang + np.array([0, 0, 1.]) * (.25 + .55 * h1) - rad * (.10 + .30 * (h2 - .5)) + (h3 - .5) * .35 * np.array([0, 0, 1.])
        upw = np.array([0, 0, 1.]) * (.9 + .4 * h2) - rad * (.35 + .7 * h3) - tang * .3
        tilt = (hash01(vx.seed[idx], 91)[:, None] - .5) * vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-9)
        ddraw = f // 2
        fast = px < 60
        ph = np.where(fast, (ddraw + vx.ph[idx] + hash01(vx.seed[idx], 2) * 2).astype(int), (f // 2 + vx.ph[idx])) % 7
        ph = np.array(PG.CYCLE)[ph.astype(int)]
        glide = hash01(vx.grp[idx], (f // 18)) < .12; ph = np.where(glide, 7, ph)
        ll = s.lit(P[idx], Lf, ST.SUNB) if kday > 0 else np.zeros(len(idx), bool)
        # palette: 0 deep violet 1 violet 2 cream (lantern-warm, low) 3 ember 4 gold
        pals = ['deep', 'violet', 'warm', 'ember2', 'gold'] if not post else ['burn'] * 5
        h = hash01(vx.seed[idx], 66)
        zz_ = P[idx, 2]
        night = np.where(zz_ < 4.2 + h * 1.5, 2, np.where(h < .10, 0, 1))
        wm = np.clip((f - 136) / 18., 0, 1) * .85
        day = np.where(ll, 4, np.where((zz_ > 15 - 6 * h) | (hash01(vx.seed[idx], 68) < wm), 3, 1))
        pid = np.where(hash01(vx.seed[idx], 67) < kday * 1.15, day, night) if kday > 0 else night
        pid = np.where(ll, 4, pid)
        L = ST.SUN if kday > .25 else np.array([-.45, .25, .86])
        fogc = mix(ST.FOG_N * 1.12, rgb('#f2cf9e'), kday)
        k = vx.kind[idx] == 0
        fog = (mix(ST.FOG_N * 1.02, rgb('#f2cf9e'), kday), 7.0, mix(.050, .022, kday)) if not post else (rgb('#fff3cf'), 10., .030 + .05 * float(np.clip((f - 138) / 26., 0, 1)))
        pg = ~k
        if pg.any():
            i2 = idx[pg]
            sd = vx.seed[i2]
            a = hash01(sd, 5) * 6.28 + ddraw * (hash01(sd, 6) - .5) * 1.5; b = hash01(sd, 7) * 6.28 + ddraw * (hash01(sd, 8) - .5) * 1.3
            nr = np.stack([np.cos(a) * np.sin(b), np.sin(a) * np.sin(b), np.cos(b)], -1)
            ax = np.stack([-np.sin(a), np.cos(a), np.zeros_like(a)], -1) + .3 * np.stack([np.cos(b * 2), np.sin(b), np.cos(a)], -1)
            PG.sheets(img, cam, P[i2], nr, ax, vx.size[i2] * 1.25, .8 + 1.2 * hash01(sd, 9), sd, ddraw, L, pid[pg], pals=s.pals(pals), smear=sm[pg], trans=True, fog=fog, lod=(64, 125 if f < 146 else 170))
        if k.any():
            i1 = idx[k]
            PG.flock(img, cam, P[i1], head[k], upw[k] + .4 * tilt[k], vx.size[i1], vx.lag[i1], ph[k], vx.seed[i1], ddraw, L, pid[k], pals=s.pals(pals), smear=sm[k], trans=True,
                     lod=(19, 118 if f < 146 else 165), fog=fog, rim=1.0 if kday > .25 else .7, tier=True)
        return img

    _pl = None

    def pals(s, names):
        if s._pl is None:
            w = dict(PG.PAL['cream']); w.update(lit='#ecdcc4', half='#c2aba8', shade='#8077a0', deep='#524a7c', ink='#1f1a36', rim='#f2dcc0', text='#5a4862')
            e = dict(PG.PAL['ember'], lit='#ffe8b8', half='#f3c893', shade='#cfa39c', deep='#9a84a2', ink='#5e3c48')
            b = dict(PG.PAL['gold'], lit='#fff6dc', half='#ffe9ba', shade='#f4cf98', deep='#e3b386', ink='#cf9763', rim='#fffbe9')
            l1 = dict(PG.PAL['cream'], lit='#fbf4e6', half='#e6dbdf', shade='#b7abcb', deep='#857ba8', ink='#2a2348', rim='#ffffff', text='#6a5c78')
            l2 = dict(PG.PAL['gold'], lit='#fffaf0', half='#fff0cc', shade='#ffdfa4', deep='#e6b88a', ink='#a0643f', rim='#fffdf4')
            s._pl = {'warm': w, 'ember2': e, 'burn': b, 'lead': l1, 'leadg': l2}
        return [s._pl.get(n, n) for n in names]

    LEAD = [  # birds that cross close to the lens: f0, f1, three screen points, span0, span1, design, seed, phase offset, light state key
        (84, 104, (-340, 260), (420, 60), (1010, 250), 470, 60, 'gull', 23, 4), (99, 116, (2420, 1080), (1800, 520), (1380, 330), 520, 80, 'heron', 31, 0),
        (103, 118, (1250, 1600), (820, 900), (640, 620), 430, 90, 'swallow', 37, 3), (123, 140, (2350, -260), (1500, 120), (1120, 250), 520, 80, 'pheasant', 41, 2), (128, 144, (-300, 1300), (330, 800), (560, 420), 560, 90, 'engraved', 45, 5),
        (133, 150, (1750, 1560), (1500, 900), (1350, 620), 480, 90, 'crane', 51, 0), (141, 156, (-260, 120), (520, 60), (960, 260), 520, 100, 'owl', 57, 3), (146, 160, (900, 1600), (560, 980), (640, 560), 600, 120, 'gull', 63, 1), (150, 165, (2380, 700), (1700, 820), (1330, 600), 520, 120, 'swallow', 67, 4)]

    def lead(s, img, f, kday):
        for k, (f0, f1, a, b, c, s0, s1, des, sd, po) in enumerate(s.LEAD):
            if not f0 <= f < f1: continue
            u = (f - f0 + .3) / (f1 - f0); t = 1 - (1 - u) ** 1.9
            a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
            B = lambda t: (1 - t) ** 2 * a + 2 * t * (1 - t) * b + t * t * c
            P, D = B(t), 2 * (1 - t) * (b - a) + 2 * t * (c - b)
            sp = s0 * (s1 / s0) ** (t ** .8)
            d = f // 2; ph = PG.CYCLE[(d + po) % 7]
            dn = D / np.hypot(*D)
            v = B(min(1, 1 - (1 - min(1, u + 1. / (f1 - f0))) ** 1.9)) - P
            sm = v * .40 if np.hypot(*v) > sp * .20 else None
            fw = np.array([dn[0], dn[1], .45]); up = np.array([-.25 * dn[1], -.25, .85])
            gold = kday > .8 and (k % 3 != 0)
            pal = 'gold' if gold else s._pl['warm'] if kday < .3 and P[1] > 500 else s.BPAL if kday < .3 else 'violet'
            img = PG.put_bird(img, des, ph, PG.basis(fw, up), P, sp, pal=pal, seed=sd, drawing=d, light=(.45, -.30, .60) if kday > .3 else (.1, -.3, .8) if P[1] < 500 else (.2, .6, -.4), trans=True, rim=1.0, smear=sm, persp=.22)
            img = np.ascontiguousarray(img, f32)
        return img

    # the leading flock: a skein of 44 pale birds, the hero crane at its point, climbing the gap between two arms
    def skein_pos(s, f):
        n = 44
        zL = comp.lerp_keys(f, [(74, 1.5), (78, 3.4), (90, 6.6), (102, 10.2), (112, 13.2), (119, 14.8), (122, 16.4), (128, 21.5), (136, 31), (146, 46), (160, 70)], e=lambda t: t)
        r = np.random.default_rng(404)
        i = np.arange(n)
        # a crowded head, a long uneven left file, a short right one, a ragged tail
        file = np.where(i == 0, 0, np.where((i % 5 == 2) | ((i % 2 == 0) & (i < 14)), 1, -1))
        rank = np.zeros(n)
        for sgn, st in ((-1, .55), (1, .80)):
            m = np.nonzero(file == sgn)[0]; gap = .62 + .05 * np.arange(len(m)) ** .8 + r.normal(0, .22, len(m)) + (r.random(len(m)) < .18) * .8
            rank[m] = st + np.cumsum(np.maximum(gap, .28)) - gap[0]
        z = zL - rank * .42 + r.normal(0, .06, n)
        wid = np.minimum(.14 + .052 * rank, .66) * (1 - .5 * smooth(12, 20, rank))
        rho = 2.35 - .5 * smooth(9, 18, z) + file * wid + r.normal(0, .085, n)
        zz = np.maximum(z, -1.8)
        th = math.pi / 3 * 3.0 + s.SK0 + 3.5 * np.log(np.maximum(zz, 0) + 2.3) + (np.minimum(zz, 0)) * 1.2 + s.vx.Om[int(f)] + r.normal(0, .030, n)
        return np.stack([rho * np.cos(th), rho * np.sin(th), z], -1), z, rank

    SK0 = .0

    def skein(s, img, cam, f, kday, Lf, draw=True):
        d = f // 2 * 2
        P, z, rank = s.skein_pos(d); Pn, _, _ = s.skein_pos(d + 1)
        ok = (P[:, 2] - cam.eye[2] > .7) & (P[:, 2] < 60)
        s2, zc = cam.project(P)
        ok &= zc > .4
        on = ok & (s2[:, 0] > -200) & (s2[:, 0] < W + 200) & (s2[:, 1] > -200) & (s2[:, 1] < H + 200)
        size = np.where(rank < .2, .35, .262 - .0030 * rank) * (1 + .34 * (hash01(np.arange(44), 5) - .5))
        px = size * cam.f / np.maximum(zc, .3)
        if not draw: return s2[on], np.clip(px[on] * 1.15, 12, 120)
        idx = np.nonzero(on)[0]
        if len(idx) == 0: return img
        vel = Pn[idx] - P[idx]; rad = P[idx].copy(); rad[:, 2] = 0; rad /= np.linalg.norm(rad, axis=1, keepdims=True) + 1e-9
        tang = np.stack([-rad[:, 1], rad[:, 0], np.zeros(len(rad))], -1)
        head = tang + np.array([0, 0, .38]) - rad * .12; upw = np.array([0, 0, 1.]) * 1.0 - rad * .45 - tang * .25
        des = np.where(hash01(idx, 9) < .72, PG.IDX['crane'], np.where(hash01(idx, 10) < .5, PG.IDX['heron'], PG.IDX['gull']))
        des[idx == 0] = PG.IDX['crane']
        ph = np.array(PG.CYCLE)[((f // 2) + (rank[idx] * .6).astype(int) + (hash01(idx, 21) * 2.2).astype(int)) % 7]        # the beat runs down the line
        sd = np.where(idx == 0, 4, idx * 977 + 31)
        ll = s.lit(P[idx], Lf, ST.SUNB) if kday > 0 else np.zeros(len(idx), bool)
        pid = np.where(ll | (kday > .75), 1, 0)
        L = ST.SUN if kday > .25 else np.array([-.35, .2, -.62])
        return PG.flock(img, cam, P[idx], head, upw, size[idx], des, ph, sd, f // 2, L, pid, pals=s.pals(['lead', 'leadg']), trans=True, lod=(9, 74), fog=(mix(rgb('#8f8cc0'), rgb('#ffe9b8'), kday), 9., .020), rim=1.0, tier=True)

    def splinters(s, img, cam, f, kday):
        """broken copper and lath thrown down the shaft"""
        r = np.random.default_rng(91); n = 84
        t0 = 117 + r.choice([0, 0, 1, 2, 3, 5, 9, 12], n); a = r.uniform(0, 6.283, n); rad = 4.4 * np.sqrt(r.random(n))
        p0 = np.stack([rad * np.cos(a), rad * np.sin(a), np.full(n, ST.ZD + .6)], -1)
        v = np.stack([r.normal(0, 1.5, n), r.normal(0, 1.5, n), -r.uniform(9, 24, n)], -1); sz = r.uniform(.10, .46, n); sd = r.integers(0, 1 << 30, n)
        d = f // 2 * 2 if f > 126 else f
        V, N, C, Z = [], [], [], []
        for i in range(n):
            t = (d - t0[i]) / 24.
            if t <= 0 or t > 1.8: continue
            P = p0[i] + v[i] * t - np.array([0, 0, 4.]) * t * t
            if P[2] - cam.eye[2] < .5 or np.hypot(P[0], P[1]) > 4.6: continue
            s2, z = cam.project(P); s3, _ = cam.project(P + v[i] / 24.)
            px = sz[i] * cam.f / z
            if px < 1.5: continue
            rr = np.random.default_rng(int(sd[i])); m = int(rr.integers(4, 6)); th = np.sort(rr.uniform(0, 6.283, m)) + t * rr.uniform(-9, 9)
            sq = .35 + .65 * abs(math.sin(t * rr.uniform(6, 14)))
            el = rr.uniform(1.2, 3.2)
            pts = np.stack([np.cos(th) * px * .5 * el * rr.uniform(.6, 1, m), np.sin(th) * px * .28 * sq * rr.uniform(.6, 1.1, m)], -1)
            dr = s3 - s2; u = dr / (np.hypot(*dr) + 1e-6); an = math.atan2(u[1], u[0]) * .5 + rr.uniform(0, 3); ca, sa = math.cos(an), math.sin(an)
            pts = pts @ np.array([[ca, sa], [-sa, ca]]); al = pts @ u; pts = pts + u * (np.clip(-al, 0, None) * min(1.2, np.hypot(*dr) / (px + 1)))[:, None] + s2
            base = mix(rgb('#4a3347'), rgb('#5f7f80'), rr.random() * .5); lit = rr.random() < .6
            V.append(np.vstack([pts, np.repeat(pts[-1:], 6 - m, 0)])); N.append(m); C.append(mix(base, rgb('#f2c078'), .25 * kday)); Z.append(z)
            e = edge_quads(pts[:2], pts[1:3], np.full(2, np.clip(px * .05, .7, 4), f32), .5)
            for q in e: V.append(q); N.append(6); C.append(rgb('#fff0bd') if lit else rgb('#d79a66')); Z.append(z - .001)
        if not V: return img
        o = np.argsort(-np.array(Z), kind='stable')
        return fill_polys(img, np.array(V)[o], np.array(N)[o], np.array(C)[o])

    def moonbeam(s, img, cam):
        g = 2; m = np.zeros((H // g, W // g), f32)
        for e0, e1, al in ((-1., 1., .085), (-.55, .1, .05), (1.5, 2.1, .06)):
            z = np.array([6.6, 10., 14., 19.]); w = .15 + .012 * (z - 4)
            P = np.concatenate([ST.cyl(-.30 + .050 * (z - 12) + w * e, z, .05) for e in (e0, e1)])
            S = P + ST.MOON * ((ST.ZD + .4 - P[:, 2:3]) / ST.MOON[2])
            rr = np.hypot(S[:, 0], S[:, 1]); S[:, :2] *= np.minimum(1, 4.9 / rr)[:, None]
            v, _ = cam.project(np.concatenate([P, S]))
            lay = np.zeros_like(m); cv2.fillPoly(lay, [np.round(cv2.convexHull((v / g).astype(np.float32))[:, 0] * 16).astype(np.int32)], al, cv2.LINE_AA, 4); m += lay
        return np.ascontiguousarray(comp.screen(img, cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * rgb('#9fb4e0')), f32)

    def scraps(s, img, cam, f, kday, Lf):
        """torn corners and dust that hang in the shaft and catch the light"""
        r = np.random.default_rng(19); n = 300
        a = r.uniform(0, 6.283, n); rad = 3.7 * np.sqrt(r.random(n)); z = r.uniform(.8, 21, n); d = f // 2 * 2
        z = cam.eye[2] + .7 + np.mod(z + (d - 78) / 24. * r.uniform(.6, 2.6, n), 20.5 - cam.eye[2]); a = a + (d - 78) / 24. * r.uniform(.15, .5, n)
        P = np.stack([rad * np.cos(a), rad * np.sin(a), z], -1); sd = r.integers(1, 1 << 28, n)
        ll = s.lit(P, Lf, ST.SUNB) if kday > 0 else np.zeros(n, bool)
        sz = r.uniform(.018, .075, n)
        e = r.uniform(0, 6.28, n) + d * r.normal(0, .12, n)
        nr = np.stack([np.cos(e), np.sin(e * 1.3), .5 * np.sin(e)], -1); ax = np.stack([-np.sin(e), np.cos(e), np.cos(e * .7)], -1)
        keep = ll | (r.random(n) < .35)
        return PG.sheets(img, cam, P[keep], nr[keep], ax[keep], sz[keep], .8, sd[keep], f // 2, ST.SUN, np.where(ll, 1, 0)[keep], pals=['violet', 'gold'], lod=(30, 300), trans=True, near=False)

    def impact_c(s, f):
        """the shutters give way: two drawn frames"""
        inv = f == 119
        cam = s.six.cam(f)
        dark, mid, lite = rgb('#1a1338'), rgb('#42357a'), rgb('#ffeec4')
        bg = dark if not inv else lite
        img = np.empty((H, W, 3), f32); img[:] = bg
        c = cam.project(np.array([0, 0, ST.ZD + 1.]))[0]
        r = np.random.default_rng(70 + f)
        rr = cam.f * ST.R0 / (ST.ZD - cam.eye[2])
        V, C = [], []; q = 0.
        ray = lambda q, rad: c + np.array([math.cos(q), math.sin(q)]) * rad
        while q < 6.283 + .2:
            w = r.choice([.03, .07, .15, .26], p=[.3, .3, .25, .15]) * r.uniform(.7, 1.3); r0 = rr * r.uniform(.75, 1.5)
            V.append([ray(q, r0), ray(q - w * .5, (r0 + 700) * .6), ray(q - w, 1900), ray(q + w, 1900), ray(q + w * .35, (r0 + 900) * .6)]); C.append(mid if not inv else (rgb('#ffd88a') if r.random() < .6 else rgb('#d9bde0'))); q += w + r.uniform(.05, .30)
        fill_polys(img, np.array(V), 5, np.array(C))
        # shelves survive only as broken rings of marks
        yy, xx = np.mgrid[0:H, 0:W].astype(f32)
        wall = s.wall(cam, f, 0.0); Lm = lum(wall); dv = Lm - cv2.GaussianBlur(Lm, (0, 0), 14)
        t = np.clip(dv * 34 + .28, 0, 1)[..., None] * (.75 if not inv else .85)
        img = img + ((mid * 1.35 if not inv else rgb('#8d78bd')) - img) * t
        img = np.ascontiguousarray(img, f32)
        s.st.draw(img, cam, 0.0, None, strokes=False, flat=((rgb('#2a2056'), lite) if not inv else (rgb('#4a3b86'), rgb('#fffaf0'))))
        M, ring = s.dome.mask(cam, W, H)
        hot = rgb('#fffbe8') if not inv else rgb('#fffdf4')
        img = np.ascontiguousarray(img * (1 - M[..., None]) + hot * M[..., None], f32)
        # shutters as hard black petals
        for k in range(ST.NP_):
            for tt in range(3):
                P, _, _, st = s.dome.plate(k, tt, f + (1 if inv else 0))
                v, _ = cam.project(P); v = c + (v - c) * (1.0 if st == 0 else 1.10 + .10 * hash01(k, tt))
                fill_polys(img, v[None], 8, (dark if not inv else rgb('#33276a'))[None])
        # the flock as flat paper cut-outs
        vx = s.vx; P, zz = vx.path(f); Pn, _ = vx.path(f + 1.0)
        dz = P[:, 2] - cam.eye[2]; idx = np.nonzero((dz > .5) & (zz > vx.join) & (dz < 19))[0]
        s2, z = cam.project(P[idx]); sn, _ = cam.project(Pn[idx]); px = vx.size[idx] * cam.f / z
        u = s2 - c; rd = np.hypot(u[:, 0], u[:, 1]); keep = (px > 9) & (rd > rr * .8); idx, s2, sn, px, u, rd = idx[keep], s2[keep], sn[keep], px[keep], u[keep], rd[keep]
        pal = dict(lit=lite if not inv else rgb('#2a2056'), half=lite if not inv else rgb('#2a2056'), shade=rgb('#9a86cc') if not inv else rgb('#6a58a6'), deep=mid if not inv else rgb('#6a58a6'), ink=bg, rim=lite, text=bg)
        h = vx.seed[idx]; rad = P[idx].copy(); rad[:, 2] = 0; rad /= np.linalg.norm(rad, axis=1, keepdims=True) + 1e-9
        tang = np.stack([-rad[:, 1], rad[:, 0], np.zeros(len(rad))], -1)
        PG.flock(img, cam, P[idx], tang + np.array([0, 0, .3]), np.array([0, 0, 1.]) - rad * .5, vx.size[idx], vx.lag[idx], np.array(PG.CYCLE)[(vx.ph[idx] + f // 2) % 7], h, f, np.array([.2, -.2, -.95]), pal, smear=u / rd[:, None] * (px * .8)[:, None], lod=(6, 400))
        k = 90; th = r.uniform(0, 6.283, k); l = r.uniform(60, 330, k); g = rr * r.uniform(.2, 1.0, k)
        Pq = np.stack([np.stack([c[0] + np.cos(th) * (g + l * q), c[1] + np.sin(th) * (g + l * q)], -1) for q in np.linspace(0, 1, 4)], 1)
        ST.strokes_into(img, Pq.astype(f32), r.uniform(2, 11, k), hot, 1.0, r.uniform(.3, .9, k), 5, 3)
        return img

    def fc(s, f):
        if f in (118, 119): return s.impact_c(f)
        cam = s.six.cam(f)
        kday = float(comp.keyed(f, [(0, 0), (116, .12), (118, .22), (120, .42), (122, .56), (124, .68), (126, .80), (129, .90), (132, 1.0)]))
        Lf = opened(s.dome, f)
        s.pals(['warm'])
        img = s.wall(cam, f, kday)
        img = np.ascontiguousarray(img, f32)
        s.dome.draw(img, cam, f, kday)
        lf = (lambda P: s.lit(P, Lf)) if Lf.max() > 0 else None
        s.st.draw(img, cam, kday, lf, warm=1 - .75 * kday)
        sk = s.skein(img, cam, f, kday, Lf, draw=False)
        for lo, hi in ((15, 70), (9, 15)):
            s.vortex_at(f, cam, lo, hi, None, Lf, img, kday, skein=sk)
        if Lf.max() > 0: img = np.ascontiguousarray(s.beams(img, cam, f, Lf), f32)
        if f < 118: img = s.moonbeam(img, cam)
        if f >= 117: s.splinters(img, cam, f, kday)
        s.scraps(img, cam, f, kday, Lf)
        for lo, hi in ((5, 9), (2.6, 5.)):
            s.vortex_at(f, cam, lo, hi, None, Lf, img, kday, skein=sk)
        img = s.skein(img, cam, f, kday, Lf)
        s.vortex_at(f, cam, .5, 2.6, None, Lf, img, kday, skein=sk)
        img = s.lead(img, f, kday)
        img = s.glow_c(img, cam, f)
        if f >= 126:      # the flock pouring out through the light itself
            img = np.ascontiguousarray(img, f32)
            s.vortex_at(f, cam, 11.5, 70, None, Lf, img, kday, post=True)
        return img

    def glow_c(s, img, cam, f):
        c = cam.project(np.array([0, 0, ST.ZD + 2.]))[0]
        if f >= 112:
            a = comp.lerp_keys(f, [(112, .0), (116, .10), (119, .20), (120, .70), (122, .60), (126, .50), (136, .56), (150, .74), (160, 1.05), (168, 1.7)])
            rr = comp.lerp_keys(f, [(112, 260), (119, 420), (120, 820), (126, 780), (134, 850), (150, 1080), (160, 1500), (168, 2300)])
            q = 4
            y, x = np.mgrid[0:H // q, 0:W // q].astype(f32)
            d = np.hypot(x - c[0] / q, y - c[1] / q) * q
            g = np.clip(1 - d / float(rr), 0, 1) ** 2.4 * float(a) + np.clip(1 - d / (float(rr) * .42), 0, 1) ** 2 * float(a) * .8
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
            if 118 < f < 140:      # keep the bent plates readable against the sky
                M, _ = s.dome.mask(cam, W, H); g = g * (1 - M[..., None] * .80 * float(np.clip((131 - f) / 8., 0, 1)))
            hot = rgb('#fff3cf') * 1.035
            img = comp.screen(np.clip(img, 0, 1.2), np.minimum(g, .80) * rgb('#ffe2a4'))
            k = 1 - np.exp(-np.maximum(g - .42, 0) * 1.5); img = img * (1 - k) + hot * k
        return img

    def frame(s, f):
        if f < 42: return s.fa(f)
        if f < 78: return s.fb(f)
        return s.fc(f)

    # -------------------------------------------------------------------------------------------- 6a
    FOCUS = np.array([1020., 600.])

    def setup_a(s):
        s.hall = ST.Hall()

        def paint():
            o = ST.paint_quads(ST.hall_cam(), W, H, 2.0, s.hall.quads, ST.shade_hall2, ['N'], {'N': rgb('#1e1c36')}, mask_kinds=(7,))
            return np.dstack([s._hall_glaze(o['N']), o['mask']]).astype(np.float16)
        hp = cached('hall', paint); s.hallp, s.hallm = hp[..., :3], hp[..., 3:]
        B = s.hall.B
        walls = {0: ST.WC, 1: ST.WA, 2: ST.WB}
        C = np.zeros((len(B['w']), 3)); Nn = np.zeros_like(C)
        for k, wl in walls.items():
            m = B['w'] == k; C[m] = wl.W((B['u0'] + B['u1'])[m] / 2, (B['v'] + B['h'] * .5)[m], .05); Nn[m] = wl.n
        n = len(C)
        s.bl = ST.shade_hall('N', C, np.ones((n, 3)), Nn, np.zeros(n), np.ones(n)).astype(f32)     # light on white paper at each book
        s.bc = ST.shade_hall('N', C, B['col'], Nn, np.zeros(n), np.ones(n)).astype(f32)
        s.bC = C; s.bN = Nn
        # the paper storm
        r = np.random.default_rng(12)
        sp, z = ST.hall_cam().project(C)
        vis = (z > .5) & (sp[:, 0] > -80) & (sp[:, 0] < W + 80) & (sp[:, 1] > -80) & (sp[:, 1] < H + 40) & (B['w'] > 0)
        ids = np.nonzero(vis)[0]
        wt = 1.0 / np.clip(z[ids], 3, 14)
        # 1 the blast: everything at once, fast and straight
        m1 = 1000
        b1 = r.choice(ids, m1, p=wt / wt.sum())
        # 2 rivers: nine gusts, each a string of pages pulled off one stretch of shelf along one curved path
        G = 10
        gb, gt, gs, gk = [], [], [], []
        heads = [(260, 500), (560, 880), (760, 620), (1080, 820), (420, 180), (1250, 600), (900, 300), (180, 900), (1330, 900), (1880, 420)]
        for g in range(G):
            d2 = ((sp[ids] - np.array(heads[g])) ** 2).sum(1); near = ids[np.argsort(d2)[:110]]
            k = int([130, 170, 130, 140, 100, 100, 130, 120, 90, 80][g]); t0 = [7, 8, 9, 13, 7, 16, 10, 18, 20, 9][g]; dur = [24, 24, 20, 22, 20, 18, 22, 18, 18, 20][g]
            q = np.sort(r.random(k))
            gb.append(r.choice(near, k)); gt.append(t0 + q * dur + r.normal(0, .35, k)); gs.append(np.full(k, g)); gk.append(q)
        s.pb = np.concatenate([b1] + gb); m2 = len(s.pb) - m1; m = m1 + m2
        s.pg = np.concatenate([np.full(m1, -1)] + gs).astype(int)
        s.pt0 = np.concatenate([5.2 + np.abs(r.normal(0, 1.1, m1)) + (r.random(m1) < .18) * r.uniform(0, 7, m1)] + gt)
        s.pseed = r.integers(1, 1 << 28, m)
        s.psz = np.where(s.pg < 0, r.uniform(.17, .30, m), r.uniform(.21, .34, m))
        s.psw = r.normal(0, 1, (m, 3))
        s.pn = r.uniform(.6, 1.7, m); s.pv = r.uniform(.8, 1.35, m)
        # blast targets fan out all round the lens
        a = r.uniform(0, 6.283, m); q = np.sqrt(r.uniform(.35, 2.2, m))
        s.ptgt = np.stack([.3 + np.cos(a) * 2.7 * q, np.full(m, -.9), 1.5 + np.sin(a) * 1.9 * q], -1)
        ex = np.array([[-2.2, -1.0, 2.8], [2.6, -1.0, 3.3], [.9, -1.0, 3.4], [-1.2, -1.0, 2.4], [2.2, -.8, 3.6], [-2.6, -1.0, 1.9], [1.5, -1., 2.6], [2.9, -.9, 2.0], [-.4, -1.0, 3.0], [-2.0, -.8, 3.2]])
        c2 = np.array([[-.6, 2.6, 1.6], [.4, 2.4, .8], [-1.4, 3.0, 2.9], [1.5, 2.6, 1.5], [-.2, 2.8, 3.4], [1.6, 3.4, 2.6], [-1.6, 2.6, 1.3], [.4, 2.6, 1.9], [1.8, 2.8, .9], [.2, 2.4, 2.8]])
        g = np.maximum(s.pg, 0)
        s.pc2 = c2[g] + s.psw * .15; s.pex = ex[g] + s.psw[:, ::-1] * .24
        s.pm0 = np.array([s.bC[s.pb[s.pg == q]].mean(0) for q in range(G)])          # each gust gathers to a throat in front of its shelf
        s.pm1 = s.pm0 + np.array([s.bN[s.pb[s.pg == q]].mean(0) for q in range(G)]) * 1.35 + np.array([0, 0, .25])
        s.pT = np.array([.95, 1.0, .9, .95, .85, .9, .9, .85, .9, .9])[g] * 24

    def _hall_glaze(s, img):
        """moon beam in the air, mist at the far end, loose gouache glazes, fallen books in the foreground"""
        img = np.ascontiguousarray(img, f32)
        r = np.random.default_rng(8)
        # far mist
        y, x = np.mgrid[0:H, 0:W].astype(f32)
        m = np.exp(-((x - 1640) / 330) ** 2) * np.exp(-((y - 820) / 420) ** 2)
        img = img + (rgb('#5f6fae') - img) * (m * .34)[..., None]
        # beam
        for q, al in (([[1640, 500], [1800, 560], [1180, 1180], [760, 1180]], .13), ([[1630, 560], [1700, 600], [980, 1180], [820, 1180]], .09), ([[1700, 640], [1820, 690], [1440, 1180], [1290, 1180]], .10)):
            sh = comp.shaft(H, W, q, 5, 3, .25); img = comp.screen(img, (sh * al)[..., None] * rgb('#a9bdf0'))
        # loose glazes along the perspective of the near wall
        n = 70
        px = r.uniform(0, 1500, n); py = r.uniform(0, 1050, n); ang = np.arctan2(1000 - py, 2690 - px) * .9
        P = np.stack([np.stack([px + np.cos(ang) * d, py + np.sin(ang) * d], -1) for d in (-260, -90, 90, 260)], 1)
        ST.strokes_into(img, P.astype(f32), r.uniform(18, 60, n), np.where(r.random((n, 1)) < .5, rgb('#2b2950'), rgb('#4a4f88')), r.uniform(.04, .12, n), r.uniform(.2, .6, n), 5, 0)
        # foreground: fallen volumes, a dark mass at the lower left and right
        c = Cel(W, H)
        def vol(x, y, w, h, rot, col, pg):
            co, si = math.cos(rot), math.sin(rot); T = lambda a, b: (x + a * co - b * si, y + a * si + b * co)
            d = .34 * h
            c.fill([T(0, 0), T(w, 0), T(w + d, -d * .8), T(d, -d * .8)], mix(rgb(col), rgb('#5a5f9a'), .22))
            c.fill([T(w, 0), T(w + d, -d * .8), T(w + d, h - d * .8), T(w, h)], rgb(pg) * .62)
            c.fill([T(0, 0), T(w, 0), T(w, h), T(0, h)], rgb(col))
            c.fill([T(0, h * .12), T(w, h * .12), T(w, h * .24), T(0, h * .24)], mix(rgb(col), rgb(pg), .25))
            c.line([T(0, 0), T(w * .5, -1), T(w, 0)], 2.2, mix(rgb(col), rgb('#8f97cc'), .45), seed=int(x))
        for (x0, y0, w, h, rot, col) in ((-40, 1078, 330, 62, .02, '#2a2442'), (-20, 1022, 280, 58, -.03, '#33263c'), (10, 972, 230, 48, .05, '#22253f'), (250, 1106, 260, 50, -.06, '#2b2440'),
                                         (1760, 1090, 330, 70, .03, '#241f38'), (1800, 1030, 270, 60, -.04, '#2c2540'), (1620, 1112, 190, 46, .10, '#26233c'), (1850, 984, 220, 46, .06, '#201e36')):
            vol(x0, y0, w, h, rot, col, '#8f86a6')
        return over_roi(img, c.rgba(), 0, 0)

    def page_at(s, f, idx=slice(None)):
        t = np.maximum(0, f - s.pt0[idx]) / 24.
        p0 = s.bC[s.pb[idx]]; n = s.bN[s.pb[idx]]
        # blast
        out = n * (1.5 * s.pn[idx] * (1 - np.exp(-11 * t)))[:, None]
        go = 11.5 * s.pv[idx] * np.maximum(t, 0) ** 1.28
        dr = s.ptgt[idx] - p0; L = np.linalg.norm(dr, axis=1, keepdims=True)
        A = p0 + out + dr / L * go[:, None] + np.array([0, 0, 1.]) * (.35 * s.psw[idx, 1] * np.minimum(t * 2, 1))[:, None]
        # rivers: cubic paths
        u = np.clip((t * 24) / s.pT[idx], 0, 1.3)[:, None] ** 1.25
        p1 = s.pm1[np.maximum(s.pg[idx], 0)] + (p0 - s.pm0[np.maximum(s.pg[idx], 0)]) * .35 + s.psw[idx] * .06; p2, p3 = s.pc2[idx], s.pex[idx]
        B = (1 - u) ** 3 * p0 + 3 * u * (1 - u) ** 2 * p1 + 3 * u * u * (1 - u) * p2 + u ** 3 * p3
        return np.where((s.pg[idx] < 0)[:, None], A, B), t

    _fly = None

    def fly_ids(s):
        if s._fly is None:
            r = np.random.default_rng(88); cam = ST.hall_cam(); sp, z = cam.project(s.bC)
            ok = np.nonzero((s.hall.B['w'] == 1) & (z > 3) & (z < 10.5) & (sp[:, 0] > 120) & (sp[:, 0] < 1500) & (sp[:, 1] > 60) & (sp[:, 1] < 1080))[0]
            pick = []
            for i in r.permutation(ok):
                if len(pick) >= 34: break
                if all(np.hypot(*(sp[i] - sp[j])) > 120 for j in pick): pick.append(i)
            s._fly = np.array(pick)
        return s._fly

    def fly_off(s, ix, f):
        """books flung clean off the shelf, the first of them before anything else has opened"""
        h = lambda q: hash01(ix, 200 + q)
        t0 = np.where(h(1) < .35, 2.0, np.where(h(1) < .7, 3.0, 4.6)); t = np.maximum(0, f + .6 - t0) / 24.
        dw = (1.0 + 1.9 * h(2)) * (1 - np.exp(-7.5 * t)) + 4.2 * t * t
        dv = (.6 + 1.6 * h(3)) * t * 2.2 - 5.0 * t * t * h(6); du = (h(4) - .42) * 4.2 * t
        return du, dv, dw, (h(5) - .5) * 11.0 * t

    def books_layer(s, img, state, d, tonefn=None, f_=None):
        cam = ST.hall_cam()
        nB = len(s.hall.B['w']); hb = hash01(np.arange(nB), 77); fly = s.fly_ids()
        # ragged timing: early birds, late ones, and a handful that leave the shelf altogether
        mix_ = {1: ((1, .10, .86), (2, .86, 1.0)), 2: ((1, .0, .13), (2, .13, .88), (3, .88, 1.0)), 3: ((2, .0, .20), (3, .20, 1.0))}.get(state, ((state, 0., 1.01),))
        parts = []
        for st_, lo_, hi_ in mix_:
            m_ = (hb >= lo_) & (hb < hi_ + 1e-9); m_[fly] = False
            if m_.any(): parts.append(ST.open_books(s.hall, st_, d, cam, only=m_))
        if f_ is not None and f_ < 16:
            m_ = np.zeros(nB, bool); m_[fly] = True
            parts.append(ST.open_books(s.hall, 3 if f_ >= 6 else 2, d + 7, cam, only=m_, off=lambda ix: s.fly_off(ix, f_), grow=1.30))
        V, K, D, I = (np.concatenate([p_[q] for p_ in parts]) for q in range(4))
        o_ = np.argsort(-D, kind='stable'); V, K, D, I = V[o_], K[o_], D[o_], I[o_]
        if tonefn is not None: C = tonefn(K, I)
        else:
            pl = {2: rgb('#f1e4c6'), 3: rgb('#cbb9a8'), 4: rgb('#8f86a6')}
            lt = np.clip(s.bl[I] * 1.25 + .20, 0, 1.1)
            C = np.where((K == 0)[:, None], np.clip(s.bc[I] * 1.25 + .02, 0, 1), s.bc[I] * .62); C[K == 5] = rgb('#2c2746') * (.7 + .5 * np.clip(s.bl[I[K == 5]], 0, 1))
            for k, c in pl.items():
                m = K == k; C[m] = c * lt[m] * PG.tints(I[m] * 17 + 3)
        # ink on the bigger, nearer volumes, interleaved so overlaps stay right
        sz = np.abs(V[:, 2] - V[:, 0]).max(1)
        big = (sz > 26) & (D < 9)
        E = edge_quads(V.reshape(-1, 2), np.roll(V, -1, 1).reshape(-1, 2), np.repeat(np.clip(sz * .014, .5, 1.5), 4).astype(f32), .4).reshape(-1, 4, 6, 2)
        V6 = np.concatenate([V, V[:, -1:], V[:, -1:]], 1)[:, None]
        allV = np.concatenate([V6, E], 1).reshape(-1, 6, 2)
        ink = rgb('#211b33')
        allC = np.concatenate([C[:, None], np.broadcast_to(C[:, None] * .40 + ink * .6, (len(C), 4, 3))], 1).reshape(-1, 3)
        Nn = np.concatenate([np.full((len(V), 1), 4), np.where(big[:, None], 6, 0) * np.ones((1, 4), int)], 1).reshape(-1)
        return fill_polys(img, allV, Nn, allC)

    def speed(s, img, f, n, cols, op, r0=(260, 900), ln=(150, 520), wd=(1.5, 7), seed=0, focus=None, dry=.45):
        r = np.random.default_rng(900 + seed * 31 + f // 2)
        c = s.FOCUS if focus is None else focus
        a = r.uniform(0, 6.283, n); a = a + np.sin(a * 3) * .2
        q0 = r.uniform(*r0, n) * (1 + .25 * np.abs(np.cos(a))); L = r.uniform(*ln, n)
        t = np.linspace(0, 1, 4)
        P = np.stack([np.stack([c[0] + np.cos(a + k * .012 * np.sin(a * 5)) * (q0 + L * k) * 1.15, c[1] + np.sin(a) * (q0 + L * k) * .92], -1) for k in t], 1)
        col = np.stack([rgb(q) for q in cols])[r.integers(0, len(cols), n)]
        return ST.strokes_into(img, P.astype(f32), r.uniform(*wd, n), col, op * r.uniform(.4, 1, n), dry * r.uniform(.5, 1.5, n), f // 2, 1)

    def dust(s, img, f):
        """flat drawn puffs knocked off the shelves"""
        r = np.random.default_rng(51)
        cam = ST.hall_cam()
        ids = s.pb[:64]
        c = Cel(W, H, ss=1)
        for k, i in enumerate(ids):
            t0 = 4 + (k % 9) * .9; a = (f // 2 * 2 - t0)
            rs = r.uniform(.12, .34); sx = r.normal(0, 1); pr = r.uniform(.8, 1.3)
            if a < 0 or a > 24: continue
            p, z = cam.project(s.bC[i] + s.bN[i] * (.15 + .5 * (1 - math.exp(-a * .25))) + np.array([0, 0, .012 * a]))
            R = rs * cam.f / z * (1 - math.exp(-a * .30)) * pr
            if R < 4 or R > 190: continue
            fade = max(0., 1 - a / 25.)
            rr = np.random.default_rng(k * 71 + 5)
            nl = 7; th = np.linspace(0, 6.283, nl, endpoint=False) + rr.uniform(0, 1)
            lob = R * (.72 + .30 * rr.random(nl)) * (1 + .06 * np.random.default_rng(k * 7 + f // 2).normal(0, 1, nl))
            pts = []
            for q in range(nl):
                pts.append(p + np.array([math.cos(th[q]), math.sin(th[q]) * .8]) * lob[q]); tm = th[q] + math.pi / nl
                pts.append(p + np.array([math.cos(tm), math.sin(tm) * .8]) * lob[q] * (.80 - .25 * (1 - fade)))
            lt = float(s.bl[i].mean())
            body = spl(pts, closed=True, n=5)
            base = mix(rgb('#5b5788'), rgb('#a6a3c8'), min(1, lt * 1.2))
            c.fill(body, base, .50 * fade ** .6)
            c.shade(spl(np.array(pts) * .72 + p * .28 + np.array([-R * .16, -R * .2]), closed=True, n=4), mix(base, rgb('#d8d2e2'), .45), within=body, alpha=.5 * fade ** .6)
        return over_roi(img, c.rgba(), 0, 0)

    FG = [  # foreground sheets whipping past the lens: f0, frames, start, end, size0, size1, spin, seed, bend
        (10, 4, (760, 300), (-420, -520), 620, 1250, .9, 5, 1.5), (13, 4, (1560, 820), (2500, 1500), 640, 1300, -1.1, 9, 1.4), (17, 5, (520, 860), (-520, 1500), 600, 1250, 1.2, 14, 1.6),
        (22, 5, (1600, 360), (2600, -420), 620, 1300, 1.0, 27, 1.5), (28, 5, (420, 420), (-560, -480), 640, 1350, -1.3, 33, 1.4),
        (33, 5, (1620, 900), (2450, 1450), 640, 1350, .9, 38, 1.5), (37, 5, (700, 980), (80, -560), 900, 1600, -1.0, 44, 1.4)]

    def foreground(s, img, f):
        for k, (f0, dur, a, b, s0, s1, spin, sd, bend) in enumerate(s.FG):
            if not (f0 <= f < f0 + dur): continue
            t = (f - f0 + .35) / dur; e = t ** 1.5
            at = np.array(a) * (1 - e) + np.array(b) * e; sz = s0 * (s1 / s0) ** e
            vel = (np.array(b) - np.array(a)) * 1.5 * t ** .5 / dur
            an = spin * t * 1.2 + sd
            R = PG.basis((math.cos(an), math.sin(an), .55 * math.sin(an * 1.7)), (.45 * math.cos(sd), .5 * math.sin(an), -.85))
            g = 2
            c = Cel(W // g + 160, H // g + 160, -80, -80, ss=1)
            PG.page(c, R, at / g, sz / g, pal='cream', seed=sd, drawing=f, bend=bend, scurve=.7 * (-1) ** k, light=(.55, -.72, -.25), text=True, smear=vel / g * .6, lw=2.2)
            L = c.rgba()
            L[..., :3] *= np.array([.97, .95, 1.0], f32)
            ln = int(np.clip(np.hypot(*vel) / g * .16, 3, 34)) | 1
            kr = np.zeros((ln, ln), f32); u = vel / (np.hypot(*vel) + 1e-6) * ln / 2
            cv2.line(kr, (int(round(ln / 2 - u[0])), int(round(ln / 2 - u[1]))), (int(round(ln / 2 + u[0])), int(round(ln / 2 + u[1]))), 1.0, 1, cv2.LINE_AA)
            kr /= kr.sum() + 1e-6
            L = cv2.filter2D(L, -1, kr, borderType=cv2.BORDER_CONSTANT)
            L = cv2.filter2D(L, -1, comp.disc_kernel(2.6), borderType=cv2.BORDER_CONSTANT)
            L = cv2.resize(L[80:-80, 80:-80], (W, H), interpolation=cv2.INTER_LINEAR)
            img = comp.over(img, L)
        return img

    def impact_a(s, f):
        """two drawn impact frames: violet with cream paper, then its inverse"""
        inv = f == 9
        dark, mid, lite = rgb('#1b1436'), rgb('#43377c'), rgb('#f4e8cc')
        bg, fg, half = (dark, lite, rgb('#8d7cc0')) if not inv else (lite, rgb('#2a2050'), rgb('#9c86c4'))
        img = np.empty((H, W, 3), f32); img[:] = bg
        r = np.random.default_rng(60 + f)
        F = s.FOCUS + [40, -60]
        ray = lambda q, rad: F + np.array([math.cos(q) * 1.12, math.sin(q) * .94]) * rad
        # broad uneven wedges
        V, C = [], []
        q = r.uniform(0, 1)
        while q < 6.283 + .2:
            w = r.choice([.02, .05, .11, .20], p=[.35, .3, .22, .13]) * r.uniform(.7, 1.3); r0 = r.uniform(160, 520)
            V.append([ray(q, r0), ray(q - w * .5, (r0 + 700) * .6), ray(q - w, 1700), ray(q + w, 1700), ray(q + w * .35, (r0 + 900) * .6)]); C.append(mid if not inv else (rgb('#cbb9d6') if r.random() < .6 else half)); q += w + r.uniform(.05, .34)
        fill_polys(img, np.array(V), 5, np.array(C))
        # the architecture survives as hard silhouettes
        m = np.array(s.hallm, f32); m = np.clip((m - .4) * 4, 0, 1)
        img = img * (1 - m) + (rgb('#2d2460') if not inv else rgb('#6a58a6')) * m
        # every page a sliver thrown out from the focus: small at the heart, huge at the rim
        n = 430; a = r.uniform(0, 6.283, n); a = a + .16 * np.sin(a * 5 + 1); rad = 120 + 1150 * r.random(n) ** .8
        V, C = [], []
        o = np.argsort(rad)
        for i in o:
            sz = (6 + rad[i] * .055) * r.lognormal(0, .5); ln = sz * r.uniform(1.6, 4.5)
            p = ray(a[i], rad[i]); u = ray(a[i], rad[i] + 10) - p; u /= np.hypot(*u); t = r.normal(0, .22); u = np.array([u[0] * math.cos(t) - u[1] * math.sin(t), u[0] * math.sin(t) + u[1] * math.cos(t)]); v = np.array([-u[1], u[0]])
            k = r.uniform(.25, .9)
            V.append([p - u * ln * .6, p + v * sz * .4 * k - u * ln * .05, p + u * ln * .5 + v * sz * .55 * k, p + u * ln * .62, p - v * sz * .45 + u * ln * .2]); C.append(fg)
            V.append([p - u * ln * .6, p + v * sz * .0 + u * ln * .05, p + u * ln * .62, p - v * sz * .45 + u * ln * .2, p - v * sz * .45 + u * ln * .2]); C.append(half if r.random() < .75 else fg)
        fill_polys(img, np.array(V), 5, np.array(C))
        img = s.speed(img, f * 2, 130, ['#f4e8cc', '#8d7cc0'] if not inv else ['#2a2050', '#6a58a6'], 1.0, (300, 1000), (240, 760), (1.5, 9), 3, focus=F, dry=.75)
        # heart of the blast: loose loaded-brush marks, no outline
        k = 64; th = r.uniform(0, 6.283, k); l = r.uniform(60, 250, k)
        P = np.stack([np.stack([F[0] + np.cos(th) * (8 + l * t) * 1.25, F[1] + np.sin(th) * (8 + l * t)], -1) for t in np.linspace(0, 1, 4)], 1)
        return ST.strokes_into(img, P.astype(f32), r.uniform(6, 26, k), rgb('#fffaf0') if not inv else rgb('#2a2050'), 1.0, r.uniform(.1, .6, k), 5, 3)

    def fa(s, f):
        if f in (8, 9): return s.impact_a(f)
        img = np.array(s.hallp, f32)
        d = f // 2
        cam = ST.hall_cam()
        if f >= 2:
            st = 1 if f < 4 else 2 if f < 6 else 3 if f < 8 else 3 if f < 12 else 4
            base = img
            img = s.books_layer(np.array(img, f32, order='C'), st, d if st < 4 else 40 + d % 3, f_=d * 2 if f > 7 else f)
            img = img * (1 - s.hallm) + base * s.hallm
            if f >= 10:     # the air is full of dust: shelves sink back
                k = .50 * min(1, (f - 9) / 6) + .26 * float(np.clip((f - 17) / 9., 0, 1))
                yy, xx = np.mgrid[0:H // 8, 0:W // 8].astype(f32) * 8
                km = cv2.resize(.62 + .5 * np.exp(-((xx - 1150) / 700) ** 2 - ((yy - 560) / 460) ** 2), (W, H))[..., None] * k
                img = np.ascontiguousarray(img + (rgb('#504c86') - img) * np.clip(km, 0, .86), f32)
        if 4 <= f < 26: img = s.dust(img, f)
        if f >= 4:
            am = comp.lerp_keys(f, [(4, 14), (6, 46), (10, 60), (16, 26), (22, 8), (26, 0)])
            if am > 1: img = s.speed(np.ascontiguousarray(img), f, int(am), ['#e9dcc4', '#a79fc6', '#cfc3d6'], .55)
        if f >= 6:
            q = f if f < 14 else d * 2
            P, t = s.page_at(q); Pn, _ = s.page_at(q + 1.0)
            on = np.nonzero((t > 0) & (P[:, 1] > .9))[0]
            P, Pn = P[on], Pn[on]
            p, z = cam.project(P); p2, _ = cam.project(Pn); v = p2 - p; px = s.psz[on] * cam.f / np.maximum(z, .3)
            sp = np.hypot(v[:, 0], v[:, 1])
            sm = v * np.clip((sp / np.maximum(px, 1) - .25) * 1.2, 0, 1.0 if f < 16 else .6)[:, None]
            h = s.pseed[on]
            a = hash01(h, 1) * 6.28 + d * (hash01(h, 2) - .5) * 1.7; b = hash01(h, 3) * 6.28 + d * (hash01(h, 4) - .5) * 1.5
            nr = np.stack([np.cos(a) * np.sin(b), np.cos(b), np.sin(a) * np.sin(b)], -1); ax = np.stack([-np.sin(a), .4 * np.sin(b), np.cos(a)], -1)
            # river pages ride their current: long edge down the flow, rolling slowly from one page to the next
            riv = s.pg[on] >= 0
            vd = Pn - P; vd /= np.linalg.norm(vd, axis=1, keepdims=True) + 1e-9
            sd_ = np.cross(vd, [0, 0, 1.]); sd_ /= np.linalg.norm(sd_, axis=1, keepdims=True) + 1e-9; up_ = np.cross(sd_, vd)
            rl = (s.pt0[on] * .35 + s.pg[on] + d * .28 + .9 * hash01(h, 12))[:, None]
            nr = np.where(riv[:, None], up_ * np.cos(rl) + sd_ * np.sin(rl) + vd * ((hash01(h, 13) - .5) * .7)[:, None], nr); ax = np.where(riv[:, None], vd + sd_ * ((hash01(h, 14) - .5) * .8)[:, None], ax)
            lt = s.bl[s.pb[on]].mean(1)
            core = np.exp(-((p[:, 0] - 1120) / 620) ** 2 - ((p[:, 1] - 560) / 430) ** 2)
            pid = np.where(hash01(h, 8) < .30 + .75 * core + .25 * lt + .45 * (s.pg[on] >= 0), 1, 0); pid = np.where((z < 1.9) & (hash01(h, 10) < .85 - 1.1 * core), 2, pid)
            img = np.ascontiguousarray(img, f32)
            bd = .8 + 1.3 * hash01(h, 9)
            for lo, hi in ((5.5, 40), (2.4, 5.5), (1.0, 2.4)):
                m = (z >= lo) & (z < hi)
                if m.any():
                    PG.sheets(img, cam, P[m], nr[m], ax[m], s.psz[on][m], bd[m], h[m], d, np.array([.42, -.60, .68]), pid[m], pals=['violet', 'cream', 'deep'], smear=sm[m], lod=(9, 175), fog=(rgb('#4a4f88'), 6., .05))
        # stand camera: drawn jolts and a slow push
        sh = shake(f, 2, 2.0, 6, 3) + shake(f, 5, 4.2, 22, 4)
        zm = 1.004 + .030 * comp.ease_in(f / 42., 1.4)
        img = comp.cam2d(img, 1024 + sh[0], 576 + sh[1], zm)
        return s.foreground(img, f)
    # -------------------------------------------------------------------------------------------- 6b
    def setup_b(s):
        def paint():
            h2 = 1560
            cam = Camera3D([1.5, -1.7, 1.0], [-2.1, 3.0, 7.2], 1180., W, h2, 1080, 760)
            Q = ST.build_well(seed=29, zlo=.6)
            o = ST.paint_quads(cam, W, h2, 1.4, Q, ST.shade_well, ['N'], {'N': rgb('#24223e')})['N']
            o = np.ascontiguousarray(o, f32)
            s.st.draw(o, cam, 0.0, None, warm=1.0)
            # a painted, out of focus background: large loose marks first, then the lens
            r = np.random.default_rng(3); n = 420
            x, y = r.uniform(0, W, n), r.uniform(0, h2, n)
            c = o[y.astype(int), x.astype(int)]
            from lp.paint import arcs
            o = ST.strokes_into(o, arcs(x, y, r.uniform(0, 3.1, n), r.uniform(40, 130, n), r.normal(0, .5, n)), r.uniform(5, 15, n), c * r.uniform(.94, 1.07, (n, 1)), r.uniform(.15, .5, n), .3, 5, 0)
            o = comp.bokeh(np.dstack([o, np.ones((h2, W), f32)]), 5.6, .8)[..., :3]
            yy, xx = np.mgrid[0:h2, 0:W].astype(f32)
            # value plan: dark sides, a paler violet well of air left of centre, cold moon above, lantern warmth below
            air = np.exp(-((xx - 830) / 620) ** 2 - ((yy - 620) / 620) ** 2)
            o = o * (.82 + .24 * air[..., None]) + rgb('#5a5896') * (air * .22)[..., None] + rgb('#1c1a30') * .10
            o = comp.screen(o, (np.clip(1 - np.hypot(xx - 560, yy + 120) / 1150, 0, 1) ** 2 * .46)[..., None] * rgb('#8fa4dc'))
            for q, al in (([[180, -60], [430, -60], [1250, 1620], [760, 1620]], .115), ([[520, -60], [590, -60], [1480, 1620], [1330, 1620]], .085), ([[-160, 100], [-60, 60], [520, 1620], [300, 1620]], .07)):
                o = comp.screen(o, (comp.shaft(h2, W, q, 9, 5, .22) * al)[..., None] * rgb('#a3b4ea'))
            o = comp.screen(o, (np.clip(1 - np.hypot(xx - 1060, (yy - 1640) * 1.15) / 900, 0, 1) ** 1.8 * .55)[..., None] * rgb('#d88a4c'))
            return o.astype(np.float16)
        s.bgb = cached('bgb', paint)
        r = np.random.default_rng(77)
        n, ng = 1500, 70
        s.qx = dict(x=r.normal(0, 1, n), y=r.uniform(3.4, 15, n), z0=r.uniform(-16, 9, n), v=r.uniform(4.2, 7.5, n), seed=r.integers(1, 1 << 28, n), kind=r.random(n), size=r.uniform(.16, .27, n), g=r.integers(0, ng, n), des=r.integers(0, len(PG.NAMES), n))
        g = s.qx['g']; gx = np.where(r.random(ng) < .5, -1, 1) * r.uniform(.25, 2.0, ng); gz = -r.uniform(0, 1, ng) ** .75 * 24 + 7; gy = r.uniform(3.6, 14, ng); gv = r.uniform(5.0, 8.0, ng)
        s.qx['y'] = np.clip(gy[g] + r.normal(0, .5, n), 3.3, 16); s.qx['z0'] = gz[g] + r.normal(0, 1.1, n); s.qx['v'] = gv[g] + r.normal(0, .2, n); s.qx['x'] = gx[g] * .36 * s.qx['y'] + r.normal(0, .42, n)
        s.qx['des'] = np.where(r.random(n) < .65, (g * 5) % len(PG.NAMES), s.qx['des'])

    BPAL = dict(PG.PAL['cream'], lit='#f3ead9', half='#c9bcc0', shade='#857cab', deep='#5a5389', ink='#241d35', rim='#eef2ff', text='#5c4a66')

    def stream(s, img, f, lo, hi):
        q = s.qx; d = f // 2
        cam = Camera3D([0, 0, 0], [0, 1, .12], 1250., W, H, 1000, 600)
        t = (d * 2 - 42) / 24.
        z = q['z0'] + q['v'] * (t + .18 * t * t)
        sw = .5 * np.sin(z * .55 + q['g'] * 1.9)
        P = np.stack([q['x'] + sw + .10 * z * np.sign(q['x']) * .25, q['y'], z], -1)
        sel = np.nonzero((q['y'] >= lo) & (q['y'] < hi) & (np.abs(z) < q['y'] * .75 + 1))[0]
        if len(sel) == 0: return img
        if hi < 6:      # nothing small crosses in front of the hero page
            sc, _z = cam.project(P[sel]); sel = sel[np.hypot((sc[:, 0] - 1030) / 1.1, sc[:, 1] - (570 if f < 66 else 330)) > 290]
            if len(sel) == 0: return img
        # sheets fold as they rise: birdness grows with height and time
        bird = q['kind'][sel] < np.clip(.18 + .10 * z[sel] + .35 * t, .08, .93)
        h = q['seed'][sel]
        vel = np.stack([.5 * .55 * np.cos(z[sel] * .55 + q['g'][sel] * 1.9) * q['v'][sel], (hash01(h, 3) - .5) * 1.5, q['v'][sel]], -1)
        fog = (rgb('#4c4a82'), 4.5, .075)
        L = np.array([-.55, -.25, .8])
        pid = np.where(P[sel, 2] < -.14 * q['y'][sel] + .3 * (hash01(h, 5) - .5) * q['y'][sel] * .3, 2, np.where(hash01(h, 6) < .5, 1, 0))
        pals = s.pals(['violet', 'cream', 'warm'])
        b = sel[bird]
        hb = h[bird]
        if (~bird).any():
            pg = sel[~bird]; hp = h[~bird]
            a = hash01(hp, 1) * 6.28 + d * (hash01(hp, 2) - .5) * 1.1; e = hash01(hp, 4) * 6.28 + d * .18
            nr = np.stack([np.cos(a) * np.sin(e), -np.abs(np.cos(e)) - .2, np.sin(a) * np.sin(e)], -1); ax = np.stack([-np.sin(a), .2 * np.sin(e), np.cos(a)], -1)
            PG.sheets(img, cam, P[pg], nr, ax, q['size'][pg] * 1.35, .8 + 1.2 * hash01(hp, 9), hp, d, L, pid[~bird], pals=pals, fog=fog, lod=(48, 120))
        if len(b):
            hd = vel[bird] * np.array([1.0, 1.0, .55]); hd[:, 0] += (hash01(q['g'][b], 8) - .5) * 5.5
            up = np.stack([(hash01(hb, 10) - .5) * .8, -.75 + hash01(hb, 11) * .5, np.full(len(b), .75)], -1)
            ph = np.array(PG.CYCLE)[(d + q['g'][b] + (hash01(hb, 12) * 3).astype(int)) % 7]
            PG.flock(img, cam, P[b], hd, up, q['size'][b], q['des'][b], ph, hb, d, L, pid[bird], pals=pals, fog=fog, lod=(17, 92), rim=.8, tier=True)
        return img

    MID = [  # other sheets caught at a fold stage: x, y at f42, rise px/frame, size, rot, first stage, seed, flip, frames per stage
        (330, 830, 12, 215, .5, 0, 3, False, 4), (1790, 1010, 14, 250, -.6, 0, 7, True, 5), (1830, 560, 10, 175, .9, 1, 11, False, 6), (540, 1260, 19, 270, -.3, 0, 15, True, 6), (170, 420, 9, 140, 1.4, 2, 19, False, 5),
        (1340, 1380, 22, 230, .4, 0, 23, False, 7), (1720, 1440, 24, 285, -.9, 1, 29, True, 5), (760, 230, 8, 112, .7, 3, 31, False, 4), (1480, 290, 8, 128, -1.2, 2, 37, True, 7), (80, 1180, 16, 190, 1.1, 1, 41, False, 5),
        (930, 1640, 25, 250, -1.5, 0, 43, True, 8), (1960, 1150, 17, 165, .3, 2, 47, False, 6), (400, 1640, 27, 210, 2.2, 0, 53, False, 5), (1180, 160, 7, 100, .3, 3, 59, True, 5), (640, 1900, 33, 255, .9, 1, 61, False, 6),
        (1500, 1960, 34, 275, -.4, 0, 67, True, 7), (1890, 1760, 30, 205, 1.8, 2, 71, False, 5), (210, 2000, 35, 240, -.8, 1, 73, True, 6), (1120, 2150, 37, 215, .2, 2, 79, False, 5), (760, 2300, 40, 280, 1.3, 1, 83, True, 6)]

    _mi = None

    def _mink(s, sd):
        if s._mi is None: s._mi = {}
        if sd not in s._mi: s._mi[sd] = PG.page_ink(sd, sd % 5)
        return s._mi[sd]

    def mids(s, img, f, big):
        d = f // 2
        order = ('taut', 'crease', 'half', 'kite', 'petal')
        for k, (x, y, v, sz, rot, st, sd, fl, fps) in enumerate(s.MID):
            if (sz >= 200) != big: continue
            yy = y - v * (d * 2 - 42) - .10 * (d * 2 - 42) ** 2; xx = x + 22 * math.sin(d * .35 + k)
            if yy < -300 or yy > H + 300: continue
            stg = st + (f - 42 + (k * 3) % 5) // fps
            rr = rot + .10 * math.sin(d * .6 + k * 2) + (hash01(k * 5 + stg, 3) - .5) * 1.9
            if stg < 5 and sz >= 130:
                S = PG.Sheet(sd) if stg == 0 else PG.origami_stage(order[stg], sd, .75 if (f // 2) % 2 and stg in (1, 4) else 1.0)
                h_ = lambda q: hash01(sd * 3 + stg, q) - .5
                R = PG.rot3(-14 + 46 * h_(1), 44 * h_(2) * 1.2, math.degrees(rr) - (0, 0, 35, 45, 45)[stg]) @ np.diag([-1. if fl else 1., 1, 1])
                img = PG.put_origami(img, S, (xx, yy), sz, R, pal=s.BPAL if yy < 800 else s._pl['warm'], seed=sd, drawing=d, light=(-.5, -.72, -.45) if yy < 800 else (-.25, .30, -.86), curl=(.2 * h_(3) + .12, .25 * h_(4), .3 * h_(5)) if stg == 0 else (.08, -.05, .05), ss=1, ink_rows=s._mink(sd) if sz > 150 else None)
            elif stg < 5: img = PG.put_fold(img, order[stg], (xx, yy), sz, rot=rr, pal=s.BPAL if yy < 800 else s._pl['warm'], seed=sd, drawing=d, flip=fl, shear=.12 * math.sin(k * 1.7))
            else:
                ph = 8 if stg == 5 else PG.CYCLE[(d + k) % 7]
                fw = (.75 * (-1 if fl else 1), -.55, .30); up = (.15, -.75, -.60)
                img = PG.put_bird(img, (k * 5) % 13, ph, PG.basis(fw, up), (xx, yy), sz * .52, pal=s.BPAL if yy < 800 else s._pl['warm'], seed=sd, drawing=d, light=(-.55, -.75, -.35), rim=.9)
        return img

    def ticks(s, img, pts, seed, ln=60, n=3, col='#e9e6f6'):
        """small drawn snap marks"""
        r = np.random.default_rng(seed); P = []
        for (x, y, a) in pts:
            for k in range(n):
                aa = a + (k - (n - 1) / 2) * .34 + r.normal(0, .05); l = ln * r.uniform(.6, 1.2); g = r.uniform(16, 30)
                t = np.linspace(0, 1, 4)
                P.append(np.stack([x + np.cos(aa + .05 * t) * (g + l * t), y + np.sin(aa + .05 * t) * (g + l * t)], -1))
        return ST.strokes_into(np.ascontiguousarray(img, f32), np.array(P, f32), r.uniform(1.4, 3.4, len(P)), rgb(col), .80, .0, seed, 1)

    _hink = None

    def hero(s, img, f):
        B = s.BPAL; L = (-.55, -.72, -.42)
        keys = [(42, 'fly'), (44, 'taut'), (47, 'crease'), (49, 'half'), (52, 'kite'), (54, 'petal'), (56, 'blade'), (59, 'spread'), (62, 'top'), (64, 'down'), (65, 'bot'), (67, 'fold'), (69, 'wrist'), (70, 'top'), (72, 'down'), (73, 'bot'), (75, 'wrist'), (76, 'top')]
        key = comp.keyed(f, keys); kf = [k for k, _ in keys if k <= f][-1]; age = f - kf
        # a tiny settle inside each hold: drawings trade places on twos, never slide
        cx, cy = 1010., 590.
        warm = ('#e59a62', .20)
        if key == 'fly':
            at = (1250, 900) if f == 42 else (1090, 700)
            R = PG.basis((.35, -.9, .3), (-.35, .25, -.9)) if f == 42 else PG.basis((.2, -.95, .15), (-.2, .3, -.93))
            img = s.speed(np.ascontiguousarray(img, f32), f, 12, ['#cfc6e2', '#e9dcc4'], .6, (80, 220), (90, 230), (1.2, 4.5), 8, focus=np.array(at) + [40, 110])
            return PG.put_page(img, R, at, 392, pal=B, seed=4, drawing=f, bend=1.5 if f == 42 else 1.15, scurve=-1.1, beta=1.2, light=L, smear=(-46, -96) if f == 42 else (-30, -58), torn=True, art=2, rim=.8)
        # rotation (nod, turn, spin), size, offset, the page's own curl, how far the last motion has gone
        tab = dict(taut=((-20, 16, -9), 396, (0, 0), (.24, -.13, .17)), crease=((-12, 25, -16), 394, (2, -4), (.12, -.08, .07)), half=((-27, -12, -33), 392, (-4, -10), (.07, -.05, .05)),
                   kite=((-28, -12, -43), 392, (2, -18), (.05, -.04, .04)), petal=((-24, -8, -42), 392, (6, -24), (.05, -.03, .05)))
        if key in tab:
            (ax, ay, az), sz, (dx, dy), curl = tab[key]
            if age == 0: az += 4; sz *= 1.03
            t = {0: .72, 1: .72}.get(age, 1.0) if key in ('crease', 'half', 'petal') else {0: .90}.get(age, 1.0)
            at = (cx + dx, cy + dy)
            if s._hink is None: s._hink = PG.page_ink(4, 2)
            S = PG.Sheet(4) if key == 'taut' else PG.origami_stage(key, 4, t)
            if key == 'taut' and age == 0: curl = (-.16, .10, -.22)          # one drawing bowed the other way, then it snaps
            img = PG.put_origami(img, S, at, sz, PG.rot3(ax, ay, az), pal=B, seed=4, drawing=f // 2, light=L, ink_rows=s._hink, curl=curl, bounce='#c79a86', lw=2.5, rim=.9)
            if age == 0:
                if key == 'taut': pts = [(at[0] - 190, at[1] - 190, 3.9), (at[0] + 150, at[1] - 230, 5.4), (at[0] + 190, at[1] + 190, .8), (at[0] - 150, at[1] + 230, 2.3)]
                elif key == 'crease': pts = [(at[0] + 90, at[1] - 140, 5.2), (at[0] + 160, at[1] + 40, .1)]
                elif key == 'half': pts = [(at[0] - 40, at[1] + 130, 1.9), (at[0] + 180, at[1] - 40, -.2)]
                elif key == 'kite': pts = [(at[0] - 110, at[1] - 20, 3.2), (at[0] + 110, at[1] - 20, .0)]
                else: pts = [(at[0] - 190, at[1] - 70, 3.4)]
                img = s.ticks(img, pts, f * 7, 46 if key == 'taut' else 36)
            return img
        # it is a crane now
        ph = dict(blade=8, spread=7, top=0, down=2, bot=3, fold=4, wrist=5)[key]
        pos = comp.keyed(f, [(56, (1015, 600)), (59, (1010, 575)), (62, (1004, 602)), (64, (1000, 520)), (65, (1006, 455)), (67, (1022, 400)), (69, (1036, 372)), (70, (1048, 380)), (72, (1070, 262)), (73, (1098, 150)), (75, (1130, 40)), (76, (1160, -60)), (77, (1190, -170))])
        sp = comp.keyed(f, [(56, 216), (59, 228), (62, 225), (64, 230), (67, 236), (70, 246), (72, 262), (75, 290), (77, 320)])
        if key == 'spread' and age == 0: ph = 9
        fw = comp.keyed(f, [(56, (.72, -.28, -.18)), (59, (.70, -.34, -.22)), (64, (.60, -.55, -.16)), (70, (.52, -.70, -.05)), (75, (.42, -.82, .10))]); up = comp.keyed(f, [(56, (.10, -.82, -.55)), (64, (.22, -.62, -.72)), (72, (.30, -.45, -.82))])
        sm = {'down': (-16, 98)}.get(key)
        if key == 'down': img = s.speed(np.ascontiguousarray(img, f32), f, 16, ['#e9dcc4', '#b9b2d6'], .7, (120, 230), (70, 170), (1.2, 4.6), 12, focus=np.array(pos) + [0, -20])
        img = PG.put_bird(img, 'crane', ph, PG.basis(fw, up), pos, sp, pal=B, seed=4, drawing=f if key == 'down' else f // 2, light=L, rim=1.0, smear=sm, persp=.20, lw=2.4, bounce='#c79a86')
        if key == 'spread' and age == 0: img = s.ticks(img, [(pos[0] - 215, pos[1] - 80, 3.5), (pos[0] + 215, pos[1] - 100, 5.6), (pos[0] - 40, pos[1] + 120, 1.8)], 99, 50, 3)
        return img

    BFG = [(43, 8, (1650, 1600), (1990, -500), 640, 3, .9), (50, 8, (120, 1700), (-60, -520), 760, 8, -.7), (57, 7, (1900, 1650), (1940, -560), 620, 13, .8), (63, 8, (300, 1700), (200, -560), 780, 17, -.9), (70, 8, (1640, 1700), (2060, -420), 800, 22, .7)]

    def fb(s, f):
        # background pans down as we are carried up with the paper
        oy = int(round((1560 - H) - 18 - 330 * ((f - 42) / 36.) ** 1.15))
        img = np.array(s.bgb[max(0, oy):max(0, oy) + H], f32, order='C')
        s.pals(['warm'])
        img = s.stream(img, f, 8.5, 99)
        img = comp.screen(img, np.full((1, 1, 1), .045, f32) * rgb('#8f9ad8'))
        img = np.ascontiguousarray(img, f32)
        img = s.stream(img, f, 5.2, 8.5)
        img = s.mids(img, f, False)
        # a soft dark halo of air keeps the hero page clear of the crowd
        yy, xx = np.mgrid[0:H // 8, 0:W // 8].astype(f32) * 8
        m = np.exp(-((xx - 1010) / 360) ** 2 - ((yy - 570) / 320) ** 2)
        m = cv2.resize(m, (W, H))[..., None] * (.50 if f < 66 else .50 * max(0, 1 - (f - 66) / 8))
        img = np.ascontiguousarray(img * (1 - m) + rgb('#39366a') * m, f32)
        img = s.hero(img, f)
        img = s.mids(np.ascontiguousarray(img, f32), f, True)
        img = s.stream(np.ascontiguousarray(img, f32), f, 3.0, 5.2)
        # near, out of focus
        for k, (f0, dur, a, b, sz, sd, spin) in enumerate(s.BFG):
            if not (f0 <= f < f0 + dur): continue
            t = (f - f0 + .3) / dur
            at = np.array(a) * (1 - t) + np.array(b) * t; vel = (np.array(b) - np.array(a)) / dur
            an = sd + spin * t
            R = PG.basis((math.cos(an), math.sin(an), .5 * math.sin(an * 1.3)), (.4 * math.cos(sd), .5 * math.sin(an), -.8))
            g = 2; c = Cel(W // g + 160, H // g + 160, -80, -80, ss=1)
            pal = dict(PG.PAL['deep'], lit='#a79fc4', half='#77709e', shade='#4b4677', deep='#36325c')
            if k in (1, 3, 4): PG.bird(c, (3, 0, 1, 6, 8)[k], PG.CYCLE[(f // 2) % 7], R, at / g, sz * .42 / g, pal=pal, seed=sd, drawing=f, light=(-.5, -.8, -.3), smear=vel / g * .5, rim=.8)
            else: PG.page(c, R, at / g, sz / g, pal=pal, seed=sd, drawing=f, bend=1.3, light=(-.5, -.8, -.3), smear=vel / g * .4, rim=.8)
            Ly = c.rgba()
            Ly = cv2.GaussianBlur(Ly, (0, 0), sigmaX=1.2, sigmaY=7)
            Ly = cv2.filter2D(Ly, -1, comp.disc_kernel(3.6), borderType=cv2.BORDER_CONSTANT)
            img = comp.over(img, cv2.resize(Ly[80:-80, 80:-80], (W, H), interpolation=cv2.INTER_LINEAR))
        # lantern warmth from far below, cold edge from above
        yy, xx = np.mgrid[0:H // 8, 0:W // 8].astype(f32) * 8
        g1 = cv2.resize(np.clip(1 - np.hypot(xx - 1040, (yy - 1420) * 1.1) / 820, 0, 1) ** 1.8, (W, H))[..., None]
        return comp.screen(img, g1 * rgb('#e08a46') * .34)


ZD1 = ST.ZD + 1.0
