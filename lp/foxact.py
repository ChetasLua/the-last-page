"""Fox acting per shot.  Owner: lead.  Shots call foxact.draw(img, key, f, M) at the right depth and
foxact.lantern_pos(key, f, M) for the light."""
import numpy as np, math
from .ink import Cel, spl, over_roi
from .paint import rgb, f32, mix
from . import W, H, foxside as fs, foxbody as fb, fox, props
from .comp import ease

ACT = {}


def pseed(*v):
    """a held pose keeps its drawing; a new pose is a new drawing"""
    import zlib
    q = repr([[round(float(a), 2) for a in np.ravel(x)] for x in v])
    return zlib.crc32(q.encode()) % 9973


def posekey(P):
    return [np.ravel(P[k]) for k in sorted(P)]


def act(k):
    def d(fn): ACT[k] = fn; return fn
    return d


def _y(M, f):
    y = M['y']
    return float(y(f)) if callable(y) else float(y)


def _x(M, f, k='x'):
    v = M[k]
    return float(v(f)) if callable(v) else float(v)


def put(img, cel, tint=None, glow=None):
    """cel over picture, with the compositor's para: a gentle light gradient so cel and background sit together.
    tint = (rgb multiply at p0, rgb multiply at p1, p0, p1) in screen px"""
    s = cel.rgba()
    if tint is not None:
        c0, c1, p0, p1 = tint
        yy, xx = np.mgrid[0:s.shape[0], 0:s.shape[1]].astype(f32)
        d = np.array(p1, float) - np.array(p0, float)
        t = np.clip(((xx + cel.ox - p0[0]) * d[0] + (yy + cel.oy - p0[1]) * d[1]) / (d @ d), 0, 1)[..., None]
        s = s.copy(); s[..., :3] *= (np.asarray(c0, f32) * (1 - t) + np.asarray(c1, f32) * t)
    return over_roi(img, s, cel.ox, cel.oy)


def roi(x, y, h, a=1.35, b=1.35, up=1.3, dn=.35):
    return Cel(int(h * (a + b)) + 16, int(h * (up + dn)) + 16, int(x - h * a), int(y - h * up))


def keypose(f, keys, e=ease):
    """[(frame, pose)] -> pose, eased between keys; holds when two keys carry the same pose"""
    if f <= keys[0][0]: return keys[0][1]
    for (a, pa), (b, pb) in zip(keys[:-1], keys[1:]):
        if f < b: return fs.lerp(pa, pb, float(e((f - a) / float(b - a))))
    return keys[-1][1]


# ===================================================== s2 reading hall
LS = .0032
S2_STOP, S2_LOW, S2_SET, S2_UP, S2_LOOK = 88, 96, 104, 112, 126


def _s2_x(f, M):
    x0, x1 = M['x0'], M['x1']
    t = np.clip(f / float(S2_STOP), 0, 1)
    t = t if t < .80 else .80 + .2 * (1 - (1 - (t - .80) / .2) ** 1.35)
    return x0 + (x1 - x0) * t


def lantern_pos(key, f, M):
    """screen px of the lantern flame centre (or None)"""
    if key == 's2':
        h = M['h']; y = _y(M, f); fd = f - f % 2
        x = _s2_x(fd, M)
        mx, my = x + .53 * h, y - .30 * h
        if fd < S2_STOP: my += .012 * h * math.sin(fd * .9)
        if fd >= S2_LOW:
            k = float(ease((fd - S2_LOW) / float(S2_SET - S2_LOW)))
            mx += .02 * h * k; my += .162 * h * k
        return np.array([mx, my])
    return M.get('lantern')


def _hang(h, rot=0.):
    def c(cel, m, k): props.lantern(cel, m[0] - 104 * h * LS * math.sin(rot), m[1] + 104 * h * LS * math.cos(rot), h * LS, rot=rot, lw=1.1)
    return c


@act('s2')
def s2(img, f, M):
    h = M['h']; fd = f - f % 2; y = _y(M, f); d = f // 2
    x = _s2_x(fd, M)
    P0 = fs.pose(earb=.08)
    LOW = fs.pose(sh=(16.5, -34.5), head=(36, -53), hang=.5, earb=.12, tail=((-.30, 13), (.2, 14), (.15, 15), (-.1, 14), (-.2, 10)))
    UP0 = fs.pose(head=(30.5, -60), hang=.02)
    UP1 = fs.pose(head=(28.5, -61.5), hang=-.50, earb=.10, tail=((.15, 13), (.28, 14), (.1, 15), (-.30, 14), (-.40, 10)))
    UP2 = fs.pose(head=(28, -61.8), hang=-.60, earb=.0, ears=.35, tail=((.28, 13), (.30, 14), (.05, 15), (-.35, 14), (-.50, 10)))
    if fd < S2_STOP:
        ph = (x - M['x0']) / (88. * h / 100)
        P = fs.trot(ph, carry_dummy=0) if False else fs.trot(ph)
        if fd > S2_STOP - 10: P = fs.lerp(P, P0, (fd - (S2_STOP - 10)) / 10.)
    else:
        P = keypose(fd, [(S2_STOP, P0), (S2_LOW - 2, P0), (S2_SET, LOW), (S2_SET + 4, LOW), (S2_UP, UP0), (S2_LOOK - 6, UP0), (S2_LOOK + 8, UP1), (150, UP1), (153, UP2), (168, UP2)])
        if 116 <= fd < 119: P = fs.pose(P, blink=1.)
    c = roi(x, y, h, 1.5, 1.5, 1.4, .3)
    gait = 2 * math.pi * (x - M['x0']) / (88. * h / 100)
    sw = .20 * math.sin(gait * 2 - 1.2) if fd < S2_STOP - 6 else .20 * math.exp(-(fd - S2_STOP + 6) / 5.) * math.cos((fd - S2_STOP) * .62)
    if fd >= S2_LOW: sw = 0.
    carry = _hang(h, sw) if fd < S2_SET else None
    if fd >= S2_SET: props.lantern(c, x + .55 * h, y, h * LS, lw=1.1, seed=d % 4)
    fs.side(c, x, y, h, P, 'lantern', max(1.5, h * .0135), pseed(*posekey(P)), key=1, carry=carry, rim=True)
    return put(img, c, ((.62, .60, .78), (1.04, 1.0, .95), (x - h, y - h), (x + .6 * h, y)))


# ===================================================== s4a hop on to the book
@act('s4a')
def s4a(img, f, M):
    h = M['h']; fd = f - f % 2; d = f // 2
    fy = M['floor_y']; py0, py1 = M['page_y']
    xi, xc, xl, xp = M['x_in'], M['x_crouch'], M['x_land'], M['x_paw']
    CR = fs.pose(hip=(-17, -27), sh=(16, -23), head=(31, -40), hang=-.35, fn=(22, 0), ff=(27, 0), hn=(-10, 0), hf=(-4, 0), earb=.55, tail=((-.45, 13), (.25, 14), (.2, 15), (.1, 14), (-.1, 10)), belly=-1.)
    TK = fs.pose(hip=(-20, -34), sh=(17, -46), head=(31, -68), hang=-.45, fn=(34, -34), ff=(38, -30), hn=(-34, -2), hf=(-30, -1), fnp=.5, ffp=.5, hnp=.5, hfp=.5, earb=.45, tail=((.2, 13), (.15, 14), (.1, 15), (.05, 14), (0, 10)))
    AIR = fs.pose(hip=(-24, -40), sh=(18, -46), head=(35, -62), hang=.02, fn=(46, -34), ff=(50, -30), hn=(-50, -30), hf=(-46, -26), fnp=.3, ffp=.3, hnp=.9, hfp=.9, earb=.7, tail=((.35, 13), (.05, 14), (-.1, 15), (-.15, 14), (-.1, 10)))
    LD = fs.pose(hip=(-20, -37), sh=(17, -27), head=(32, -44), hang=.25, fn=(24, 0), ff=(29, 0), hn=(-14, -4), hf=(-9, -2), earb=.35, tail=((.7, 13), (.35, 14), (.05, 15), (-.2, 14), (-.3, 10)))
    ST = fs.pose(head=(31, -56), hang=.28, earb=.12)
    W1 = fs.pose(head=(32, -54), hang=.36, fn=(27, -5), fnp=.5, hf=(-8, -3), earb=.1)
    W2 = fs.pose(head=(32, -54), hang=.38, ff=(30, -5), ffp=.5, hn=(-14, -3), earb=.1)
    PW = fs.pose(head=(31, -55), hang=.52, fn=(33, -13), fnp=-.15, sh=(15.5, -38), hip=(-19, -37), earb=.05, ears=.3, tail=((.1, 13), (.28, 14), (.1, 15), (-.32, 14), (-.42, 10)))
    rot = 0.
    if fd < 14:
        t = fd / 14.; x = xi + (xc - xi) * (t ** .9); y = fy
        ph = fd / 9.
        P = fs.trot(ph, 100., 11.); P = fs.pose(P, head=(33, -58), hang=-.02, earb=.35)
    elif fd < 22:
        P = keypose(fd, [(14, CR), (22, CR)]); x, y = xc + (2 if fd < 18 else 0), fy
        if fd >= 18: P = fs.pose(CR, hip=(-18, -25.5), sh=(15, -21.5), head=(30, -38.5))
    elif fd < 34:
        t = (fd - 22) / 12.
        x = xc + (xl - xc) * t
        y0, y1 = fy, py0
        y = y0 + (y1 - y0) * t - 4 * t * (1 - t) * (h * .34)
        P = TK if fd < 26 else AIR if fd < 30 else fs.lerp(AIR, LD, .5)
    else:
        if fd < 38: P = LD
        else: P = keypose(fd, [(38, LD), (41, ST), (44, W1), (46, ST), (48, W2), (50, ST), (54, PW), (56, PW)])
        t = float(ease((fd - 41) / 9.)) if fd > 41 else 0.
        x = xl + (xp - xl) * t; y = py0 + (py1 - py0) * t
    c = roi(x, y, h, 1.6, 1.6, 1.4, .4)
    if 22 <= fd < 30:      # dry-brush smear behind the leap
        r = np.random.default_rng(70 + d); C = fox.PAL['lantern']
        vx, vy = (xl - xc) / 12. * 1.6, -(h * .34) * 4 * (1 - 2 * (fd - 22) / 12.) / 12. * 1.6 + (py0 - fy) / 12 * 1.6
        for j in range(6):
            ox, oy = r.uniform(-.42, .2) * h, r.uniform(-.62, -.12) * h
            c.line(np.array([[x + ox - vx * r.uniform(1, 2.2), y + oy - vy * r.uniform(1, 2.2)], [x + ox, y + oy]]), r.uniform(3, 8) * h / 250, C['fur'] if j % 3 else C['fur_sh'], (.8, .1), seed=j, alpha=.5)
    fs.side(c, x, y, h, P, 'lantern', 2.1, pseed(*posekey(P)), key=1, rim=True)
    return put(img, c, ((.70, .68, .86), (1.04, 1.0, .95), (x - h, y - h), (x + .6 * h, y)))


# ===================================================== s4b paw on the page
@act('s4b')
def s4b(img, f, M):
    fd = f - f % 2; px, py, s = M['px'], M['py'], M.get('scale', 1.0)
    ang = M.get('ang', -.62)
    if fd < 2: return img
    k = [(2, 1.0), (4, .62), (6, .30), (8, .10), (10, .0)]
    off = 1.0
    for a, v in k:
        if fd >= a: off = v
    press = 1.0 if fd >= 8 else .2
    if fd in (10, 11): press = 1.5
    dx, dy = -math.cos(ang) * 330 * s * off, -math.sin(ang) * 330 * s * off - 60 * s * math.sin(math.pi * min(1, off))
    c = Cel(W, H)
    seed = (f // 4) if fd > 14 else f // 2
    fb.paw_top(c, px + dx, py + dy, s, ang, press, 'lantern', pseed(off, press), 3.0, key=1)
    return put(img, c, ((.80, .78, .92), (1.05, 1.0, .94), (px - 500, py + 200), (px + 250, py - 100)))


# ===================================================== s4c watching the bird
@act('s4c')
def s4c(img, f, M):
    h, x, y = M['h'], M['x'], M['y']; fd = f - f % 2; d = f // 2
    A = fs.pose(head=(30, -56), hang=.40, earb=.05, ears=.3)
    B = fs.pose(head=(29.5, -58), hang=.22, earb=.0, ears=.4, sh=(15.5, -39))
    C_ = fs.pose(head=(28, -60.5), hang=-.18, earb=.05, ears=.2, sh=(15, -39.5), mouth=.15)
    D = fs.pose(head=(27, -62.5), hang=-.52, earb=.12, sh=(14.5, -40), hip=(-19, -37.5), mouth=.35, tail=((.05, 13), (.28, 14), (.12, 15), (-.32, 14), (-.45, 10)))
    E_ = fs.pose(D, tail=((.22, 13), (.30, 14), (.05, 15), (-.40, 14), (-.55, 10)), head=(26.6, -63), hang=-.58)
    P = keypose(fd, [(0, A), (10, A), (13, B), (22, B), (30, C_), (36, D), (46, D), (52, E_), (56, E_)])
    if 16 <= fd < 20: P = fs.pose(P, blink=1. if fd < 19 else .5)
    c = roi(x, y, h, 1.5, 1.3, 1.35, .3)
    fs.side(c, x, y, h, P, 'lantern', 2.7, pseed(*posekey(P)), key=1, rim=True)
    return put(img, c, ((.72, .70, .88), (1.05, 1.0, .95), (x - h * .8, y - h), (x + .6 * h, y)))


# ===================================================== s5 inserts
def closeup(img, x, y, s, f, hd, pal='lantern', key=1, breath=0., lw=3.3, wind=0., tint=None, seed=0, under=True, rim_k=None, turn=1.0, body_seed=None):
    c = Cel(W, H)
    fb.bust(c, x, y, s, pal, seed if body_seed is None else body_seed, lw, key, wind, breath, under=under, rim_k=rim_k, turn=turn)
    fox.head(c, x, y, s=s, pal=pal, seed=seed, lw=lw, key=key, wind=wind, under=under, rim_k=rim_k, **hd)
    return put(img, c, tint)


@act('s5e')
def s5e(img, f, M):
    fd = f - f % 2; d = f // 2
    k = 0. if fd < 6 else .35 if fd < 8 else .75 if fd < 10 else 1.0
    kk = k if fd < 12 else 1.0 + (.06 if fd in (12, 13) else 0)
    hd = dict(yaw=.30, pitch=.10 - .16 * k, roll=-.03 * k, ear_l=(.1, .92 * kk), ear_r=(-.1, .92 * kk), brow=k, wide=.6 + .6 * k, pupil=1.0 - .18 * k, look=(.5 - .3 * k, -.75), mouth=.0)
    s = 2.35
    tint = ((.78, .74, .92), (1.05, .99, .92), (300, 100), (1500, 1100))
    return closeup(img, 1010, 560 + 22 * k, s, f, hd, tint=tint, seed=pseed(k, kk), body_seed=3)


@act('s5i')
def s5i(img, f, M):
    x, y, h = _x(M, f), _y(M, f), M['h']
    P = fs.pose(hip=(-18, -32), sh=(16, -30), head=(30, -48), hang=-.28, earb=.85, fn=(21, 0), ff=(26, 0), hn=(-14, 0), hf=(-8, 0), tail=((-.45, 13), (.2, 14), (.2, 15), (.15, 14), (0, 10)))
    c = roi(x, y, h, 1.5, 1.5, 1.4, .3)
    if M.get('draw_lantern', True): props.lantern(c, x + .75 * h, y, h * LS, lw=1.1)
    fs.side(c, x, y, h, P, 'lantern', max(1.4, h * .0135), 3, key=1, rim=True)
    return put(img, c, ((.62, .60, .78), (1.04, 1.0, .95), (x - h, y - h), (x + .6 * h, y)))


# ===================================================== s7 sunrise
@act('s7a')
def s7a(img, f, M):
    # wind comes from behind (left to right): ears pushed forward, tail lifted and thrown over the back
    x, y, h = _x(M, f), _y(M, f), M['h']; d = f // 2; fd = f - f % 2
    w = math.sin(d * .9) * .5 + math.sin(d * .37 + 1) * .5
    A = fs.pose(head=(30, -58), hang=.05, earb=.0, ears=.5, wind=-1.)
    B = fs.pose(head=(29, -61), hang=-.28, earb=.0, ears=.55, wind=-1.)
    C_ = fs.pose(head=(28, -62), hang=-.46, earb=.0, ears=.65, wind=-1., mouth=.35)
    P = keypose(fd, [(0, A), (14, A), (22, B), (44, B), (52, C_), (72, C_)])
    P['tail'] = tuple((a_ + .13 * math.sin(d * .95 - .7 * i) + .10 * w, l) for i, (a_, l) in enumerate(((.18, 13), (.36, 14), (.42, 15), (.40, 14), (.30, 10))))
    P['ears'] = P['ears'] + .25 * w
    c = roi(x, y, h, 1.6, 1.4, 1.4, .3)
    fs.side(c, x, y, h, P, 'gold', max(1.4, h * .014), d, key=1, rim=True)
    return put(img, c, ((.86, .78, .86), (1.08, 1.03, .95), (x - .8 * h, y), (x + .6 * h, y - h)))


@act('s7b')
def s7b(img, f, M):
    fd = f - f % 2; d = f // 2
    w = .8 + .5 * math.sin(d * 1.1) + .25 * math.sin(d * 2.3 + 1)
    mo = 0. if fd < 8 else .18 if fd < 10 else .40
    hd = dict(yaw=.42, pitch=.30 + .05 * ease(fd / 10.), roll=-.05, ear_l=(-.30 + .16 * math.sin(d * 1.3), .10 + .06 * math.sin(d * 1.1 + .5)), ear_r=(.34 + .16 * math.sin(d * 1.2), .22 + .10 * math.sin(d * 1.1)), wide=.9,
              look=(.8, -.55), mouth=mo, pupil=.92, lantern=False, brow=.15)
    if fd == 4: hd['blink'] = .55
    tint = ((.90, .82, .90), (1.07, 1.02, .94), (300, 1000), (1500, 200))
    return closeup(img, M.get('hx', 980), M.get('hy', 560), 2.25, f, hd, 'gold', key=1, lw=3.2, wind=1.5 * w, tint=tint, seed=d, under=False, rim_k=1)


# ===================================================== s9 last page
@act('s9a')
def s9a(img, f, M):
    x, y, h = _x(M, f), _y(M, f), M['h']; d = f // 2; fd = f - f % 2
    px, py = M['page']
    side = 1 if px > x else -1
    UPL = dict(yaw=.34 * side, pitch=.30, look=(.6 * side, -.7), ear_l=(0, 0), ear_r=(0, 0))
    FOL = dict(yaw=.44 * side, pitch=.02, look=(.8 * side, .1))
    DWN = dict(yaw=.36 * side, pitch=-.42, look=(.5 * side, .9), ear_l=(.15, 0), ear_r=(-.15, 0))
    US0 = dict(yaw=.06 * side, pitch=-.06, look=(0, .0), ear_l=(-.05, 0), ear_r=(.05, 0))
    US1 = dict(yaw=.0, pitch=.0, look=(0, 0))
    def L(a, b, t):
        out = {}
        for k in set(a) | set(b):
            va, vb = np.array(a.get(k, (0, 0) if k in ('look', 'ear_l', 'ear_r') else 0.), float), np.array(b.get(k, (0, 0) if k in ('look', 'ear_l', 'ear_r') else 0.), float)
            v = va * (1 - t) + vb * t; out[k] = tuple(v) if v.ndim else float(v)
        return out
    def K(keys):
        if fd <= keys[0][0]: return dict(keys[0][1])
        for (a, pa), (b, pb) in zip(keys[:-1], keys[1:]):
            if fd < b: return L(pa, pb, float(ease((fd - a) / float(b - a))))
        return dict(keys[-1][1])
    hd = K([(0, UPL), (24, UPL), (34, FOL), (46, DWN), (74, DWN), (84, US0), (88, US1), (112, US1), (122, DWN), (150, DWN)])
    for a in (40, 79, 102):
        if a <= fd < a + 2: hd['blink'] = .5
        elif a + 2 <= fd < a + 6: hd['blink'] = 1.
        elif a + 6 <= fd < a + 8: hd['blink'] = .5
    if 60 <= fd < 66: hd['sniff'] = (1, .2, 1)[(fd - 60) // 2]
    if 92 <= fd < 112: hd['roll'] = -.05 * float(ease((fd - 92) / 8.))
    reach = None; lean = (0., 0.)
    tx, ty = px - side * M.get('reach_dx', 78), py + 2
    if fd >= 122:
        t = float(ease((fd - 122) / 18.))
        sx, sy = x + 27 * h / 470, y - 4
        lift = 46 * math.sin(math.pi * min(1, t ** .8))
        reach = (sx + (tx - sx) * t, sy + (ty - sy) * t, lift)
        lean = (44. * t * side, 34. * t)
    br = .5 + .5 * math.sin(fd * .13)
    tl = .0 if fd < 70 else .25 * math.sin((fd - 70) * .06)
    c = Cel(int(h * 1.9), int(h * 1.5), int(x - h * .95), int(y - h * 1.25))
    hd['lantern'] = False
    fb.sit(c, x, y, h, 'gold', max(2.2, h * .0052), pseed(*[np.ravel(hd[k]) for k in sorted(hd) if k not in ('blink',)], reach if reach is not None else 0.), key=M.get('key', 1), hd=hd, reach=reach, tail=tl, breath=br, rim_k=1, tside=-side, lean=lean)
    return put(img, c, ((.86, .77, .84), (1.08, 1.03, .95), (x - .5 * h, y), (x + .5 * h, y - h)))


@act('s9b')
def s9b(img, f, M):
    fd = f - f % 2; px, py, s = M['px'], M['py'], M.get('scale', 1.0); ang = M.get('ang', -.55)
    off = 0.
    for a, v in ((0, .55), (2, .28), (4, .10), (6, .0)):
        if fd >= a: off = v
    press = .4 if fd < 6 else 1.4 if fd < 9 else 1.
    c = Cel(W, H)
    dx, dy = -math.cos(ang) * 300 * s * off, -math.sin(ang) * 300 * s * off - 46 * s * math.sin(math.pi * min(1, off * 1.2))
    fb.paw_top(c, px + dx, py + dy, s, ang, press, 'gold', pseed(off, press), 2.8, key=1)
    tx, ty = M.get('tail', (1830, 1010))
    # tail: rest, one flick (anticipation, snap, overshoot, settle)
    fl = 0.
    for a, v in ((116, -.18), (119, .55), (121, 1.15), (124, .8), (127, .25), (130, -.10), (133, 0.)):
        if fd >= a: fl = v
    base = np.array([[W + 60, ty + 110], [tx + 150, ty + 104], [tx + 60, ty + 86], [tx - 18, ty + 34], [tx - 34, ty - 44], [tx + 6, ty - 116]], float)
    sw = np.array([[0, 0], [0, -4], [0, -14], [8, -34], [34, -50], [78, -46]], float)
    P = base + sw * fl * 1.3
    fb.tail_tip(c, P, M.get('tail_w', 58), 'gold', pseed(fl), 2.8)
    if 119 <= fd < 123:
        r = np.random.default_rng(fd)
        for j in range(5):
            a = P[4] + r.normal(0, 14, 2); c.line(np.array([a, a + [-30 - 20 * r.random(), 26 + 10 * r.random()]]), 3, fox.PAL['gold']['wht_sh'], (.6, .2), seed=j, alpha=.6)
    return put(img, c, ((.94, .88, .92), (1.05, 1.02, .97), (0, H), (W, 0)))


def standin(img, x, y, h):
    c = roi(x, y, h); fs.side(c, x, y, h, fs.STAND, 'lantern', 2, 0); return put(img, c)


def draw(img, key, f, M):
    if key in ACT: return ACT[key](img, f, M)
    if 'x' in M: return standin(img, _x(M, f), _y(M, f), M['h'])
    return img
