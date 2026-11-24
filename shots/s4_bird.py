"""s4c bird acting: the ink line shivers, lifts a wing, peels off the page and flies, still paper and ink.  Owner: ink lane."""
import math, numpy as np, cv2
from lp import W, H, comp, ink_letters as IL, sheet3d as S3
from lp.paint import rgb, mix, smooth, f32, fbm, lum
from lp.ink import Cel, spl, ribbon, xf, boil, over_roi, resample, noise1
from lp.brush import Sheet, Stroke, blob, ink_at, transmit

BSC = 2.0
PAP, PAP_S, PAP_H, INK = '#f1e2bf', '#b39fae', '#fff2d2', '#1b171b'
NWING = 18                                        # BIRD control points before the wing loop starts
_s = {}


def outline(n=10):
    """silhouette of the one-line bird (BIRD units), closed"""
    q = np.array([(-236, 124), (-200, 94), (-128, 48), (-66, 14), (-8, -26), (44, -70), (90, -111), (136, -119), (170, -100), (206, -72), (178, -52), (176, -20),
                  (190, 14), (172, 76), (98, 122), (18, 114), (-46, 98), (-124, 102), (-212, 134)], float)
    return spl(q, closed=True, corners=(9, 18, 0), n=n)


# four drawn bodies.  ink = one brush line; out = paper silhouette; sh = hard shadow; eye; shoulder(s); wrot = how far the wings turn with it
BODY = {
 'level': dict(ink=IL.BIRD[:NWING], out=None, eye=IL.BIRD_EYE + (7.4,), shd=[(-260, 160), (-250, 60), (-130, 66), (-30, 60), (40, 76), (120, 78), (176, 50), (200, 90), (110, 150)], shc=(0, 1, 7, 8),
               sho=(30, -6), wrot=0.),
 # neck out, belly drawn in, tail in line: the pull of the down-stroke
 'stretch': dict(ink=[(-262, 36, 2.2), (-156, 12, 4.2), (-62, -22, 5.8), (30, -56, 5.2), (98, -86, 6.2), (146, -92, 5.6), (182, -80, 3.6), (230, -70, 2, 'c'), (188, -57, 2.4), (166, -40, 4.0), (152, -6, 8.0), (112, 40, 12.0), (44, 64, 12.5), (-34, 62, 8.5), (-118, 56, 5.5), (-196, 64, 4.2), (-286, 84, 2.2, 'c'), (-206, 58, 3.6), (-124, 40, 2.4)],
               out=[(-304, 92), (-272, 24), (-154, 0), (-62, -34), (30, -68), (96, -100), (148, -106), (190, -92), (238, -73), (198, -48), (178, -28), (164, 2), (122, 52), (48, 78), (-34, 76), (-118, 70), (-196, 80)], cor=(8, 0, 1),
               eye=(132, -70, 6.8), shd=[(-310, 110), (-270, 46), (-120, 44), (0, 40), (90, 34), (150, 8), (186, 30), (90, 110)], shc=(0, 1, 6, 7), sho=(26, -20), wrot=.06),
 # thrown back on its tail, breast out, beak up: the launch
 'back': dict(ink=[(-150, 180, 2.2), (-128, 116, 4.4), (-98, 42, 5.8), (-74, -24, 6.4), (-42, -92, 5.8), (-2, -150, 6.6), (38, -178, 6.0), (74, -180, 4.0), (124, -204, 2, 'c'), (94, -162, 2.4), (90, -134, 3.8), (104, -84, 7.5), (106, -12, 12.5), (72, 60, 15), (4, 106, 11.5), (-48, 116, 6.5), (-90, 148, 4.4), (-128, 200, 2.2, 'c'), (-112, 152, 3.6), (-92, 110, 2.4)],
               out=[(-164, 190), (-142, 112), (-112, 36), (-86, -30), (-52, -100), (-10, -162), (36, -192), (78, -194), (134, -214), (108, -162), (104, -136), (118, -84), (120, -8), (84, 70), (8, 122), (-40, 130), (-80, 162), (-120, 216)], cor=(8, 0, 17),
               eye=(58, -150, 7.2), shd=[(-170, 220), (-130, 110), (-92, 60), (-60, 40), (-10, 70), (50, 46), (96, -14), (140, 0), (90, 110), (10, 150), (-60, 170)], shc=(0, 1, 6, 7), sho=(0, -40), wrot=-.52),
 # three-quarter back view, banking off into the hall
 'bank': dict(ink=[(-150, 58, 2.2), (-98, 34, 4.4), (-48, -12, 5.8), (2, -56, 5.8), (44, -98, 6.2), (80, -110, 5.6), (108, -96, 3.8), (138, -84, 2, 'c'), (110, -74, 2.4), (106, -54, 3.8), (98, -8, 7.5), (66, 46, 10.5), (12, 76, 9.5), (-38, 86, 6), (-94, 116, 4.2), (-130, 142, 2.2, 'c'), (-112, 112, 3.2), (-76, 78, 2.6), (-48, 56, 1.6)],
               out=[(-168, 56), (-104, 20), (-56, -24), (-4, -70), (40, -112), (82, -124), (118, -106), (148, -84), (118, -64), (118, -46), (112, -2), (78, 58), (16, 90), (-34, 100), (-90, 130), (-134, 160), (-156, 108)], cor=(7, 0, 15, 16),
               eye=(96, -88, 5.4), shd=[(-180, 60), (-120, 70), (-40, 44), (30, 30), (70, -14), (100, -40), (130, 0), (70, 110), (-60, 150), (-150, 170)], shc=(0, 5, 6, 9), sho=(6, -34), wrot=0.),
}


def sprite(wing=True):
    """paper skin with the brush line on it, flat: alpha, ink coverage, torn-edge rim, hard shadow shape.
    True = the bird as drawn on the page (wing inked); or a BODY name"""
    if wing in _s: return _s[wing]
    B = BODY['level'] if wing is True else BODY[wing]
    w, h = int(620 * BSC), int(500 * BSC); ox, oy = 340 * BSC, 250 * BSC
    if wing is True: w, h, ox, oy = int(500 * BSC), int(320 * BSC), 262 * BSC, 160 * BSC
    r = np.random.default_rng(8)
    q0 = outline() if B['out'] is None else spl(np.array(B['out'], float), closed=True, corners=B['cor'], n=10)
    o = resample(q0 * BSC + (ox, oy), 5.0)
    o = o + np.c_[noise1(len(o), 5.5, r), noise1(len(o), 5.5, r)] * 2.2 + np.c_[noise1(len(o), 1.6, r), noise1(len(o), 1.6, r)] * .7
    m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [np.round(o * 8).astype(np.int32)], 255, cv2.LINE_AA, 3)
    a = m.astype(f32) / 255.
    I = IL.BIRD if wing is True else B['ink']
    b = [(x * BSC + ox, y * BSC + oy, rr * BSC) + tuple(k) for (x, y, rr, *k) in I]
    sh = Sheet(w, h, K=4, margin=10); sh.add(Stroke(b, step=.7, load=(1, .45), entry=-.2, depl=1.2), 0, 1)
    ex, ey, er = B['eye']
    sh.add(blob(ex * BSC + ox, ey * BSC + oy, er * BSC, stretch=.34, ang=.9, seed=3), 0, 1); sh.bake()
    cov, _ = ink_at(sh.maps(), 999)
    e = cv2.GaussianBlur(a, (0, 0), 2.4); rim = np.clip((1 - e) * 2.0, 0, 1) * a
    # one hard cel shadow: under the belly and toward the tail (lamp is at the right)
    s = np.zeros((h, w), np.uint8); cv2.fillPoly(s, [np.round((spl(np.array(B['shd'], float), closed=True, corners=B['shc']) * BSC + (ox, oy)) * 8).astype(np.int32)], 255, cv2.LINE_AA, 3)
    _s[wing] = dict(a=a, cov=cov, rim=rim, sh=s.astype(f32) / 255., w=w, h=h, ox=ox, oy=oy)
    return _s[wing]


def cel_tex(sp):
    """premult RGBA texture of the paper bird (cel colours)"""
    k = 'tex%d' % id(sp)
    if k not in _s:
        c = np.ones(sp['a'].shape + (3,), f32) * rgb(PAP)
        c = mix(c, np.ones_like(c) * rgb(PAP_S), (sp['sh'] * .92)[..., None])
        c = mix(c, np.ones_like(c) * rgb(PAP_H), (sp['rim'] * .75 * (1 - sp['sh']))[..., None])
        fib = fbm(c.shape[0], c.shape[1], (4, 2), (1, .6), 3)
        c = c * (1 + .012 * fib[..., None])
        c = c * (1 - (1 - transmit(sp['cov'])))
        _s[k] = np.dstack([c * sp['a'][..., None], sp['a']]).astype(f32)
    return _s[k]


# ------------------------------------------------------------ wings (BIRD units, shoulder at 0,0; facing right)
# each: outline (root front .. tip .. feather ends .. root back), corner indices, quill lines
WINGS = [
    # 0 up: thrown high, a little back
    dict(p=[(34, 20), (54, -40), (58, -120), (30, -206), (-4, -252), (-14, -206), (-40, -218), (-44, -166), (-72, -168), (-66, -118), (-92, -108), (-74, -60), (-84, -38), (-52, 14)], c=(4, 5, 6, 7, 8, 9, 10, 11, 12)),
    # 1 down, through the middle: broad, fingers spread (the smear drawing)
    dict(p=[(36, 2), (92, -6), (172, -8), (250, 14), (292, 52), (244, 44), (258, 80), (204, 66), (206, 102), (150, 82), (140, 118), (94, 92), (66, 112), (0, 58)], c=(4, 5, 6, 7, 8, 9, 10, 11, 12)),
    # 2 down: full reach forward and under
    dict(p=[(40, 0), (98, 10), (164, 66), (214, 144), (222, 204), (182, 168), (174, 210), (138, 164), (120, 200), (92, 148), (64, 168), (50, 116), (14, 118), (-22, 40)], c=(4, 5, 6, 7, 8, 9, 10, 11, 12)),
    # 3 down and back, pulling through
    dict(p=[(36, 10), (66, 52), (74, 112), (46, 186), (4, 226), (-2, 182), (-34, 206), (-38, 158), (-66, 172), (-62, 124), (-86, 122), (-70, 84), (-82, 64), (-40, 24)], c=(4, 5, 6, 7, 8, 9, 10, 11, 12)),
    # 4 recovery: wrist up, hand folded back
    dict(p=[(36, 12), (64, -30), (58, -84), (14, -92), (-60, -60), (-152, 14), (-112, 18), (-104, 48), (-70, 42), (-54, 66), (-18, 42)], c=(5, 6, 7, 8, 9, 10)),
    # 5 recovery: wrist high, hand trailing, about to open
    dict(p=[(36, 16), (60, -44), (64, -124), (22, -164), (-50, -150), (-150, -102), (-110, -86), (-114, -50), (-74, -42), (-66, -6), (-42, 18)], c=(5, 6, 7, 8, 9, 10)),
]


def wing(cel, P, k, T, far=False, seed=0, stretch=1.0):
    """draw wing pose k; T maps BIRD-unit points (shoulder origin) to screen"""
    w = WINGS[k] if not isinstance(k, dict) else k
    q = np.array(w['p'], float)
    q = q * (1.16 if not isinstance(k, dict) else 1.0)
    if far and isinstance(k, dict): q = q * .97
    elif far:
        a_ = .30 if k in (0, 5, 4) else -.16 if k == 1 else -.26
        ca, sa = math.cos(a_), math.sin(a_)
        q = q @ np.array([[ca, sa], [-sa, ca]]) * [.96, .92] + [30, -14]
    q = boil(q, 1.3, seed)
    o = T(spl(q, closed=False, corners=w['c'], n=7))
    col, shd = (PAP, PAP_S) if not far else ('#cbb7ae', '#9a879d')
    cel.fill(o, col)
    # hard shadow along the trailing half
    n = len(q); root = (q[0] + q[-1]) / 2
    lead = q[1:max(w['c'][0], 4) + 1]
    tip = q[w['c'][0]]
    mid = np.array([root * (1 - t) + tip * t for t in (0., .35, .7, 1.0)])
    off = (q[-3] - root) * .10
    sh = np.vstack([mid + off, q[w['c'][0] + 1:][::1]])
    cel.shade(T(spl(sh, closed=True, corners=tuple(range(4, len(sh))), n=6)), shd, within=o, alpha=.95)
    # quills: single brush lines from the wrist into each finger
    fing = [j for j in w['c'] if (j - w['c'][0]) % 2 == 0]
    cel.clip(o)
    for i, j in enumerate(fing):
        a = root + (q[j] - root) * .30 + (tip - root) * .10; b = root + (q[j] - root) * .90
        m = (a + b) / 2 + np.array([-(b - a)[1], (b - a)[0]]) * .06
        cel.line(T(spl(np.array([a, m, b]), n=8)), 2.6 * P['lw'], INK, (.15, .7), seed=seed + i, press=.4, alpha=.95 if not far else .7)
    cel.unclip()
    # the brush contour: heavy on the leading edge, broken on the feathers
    L = T(spl(q[:w['c'][0] + 1], n=8))
    cel.line(L, 8.0 * P['lw'], INK, (.30, .55), seed=seed + 31, press=.35, wfun=lambda t: .55 + .75 * np.sin(np.pi * np.clip(t * 1.05, 0, 1)) ** .8, tmin=.15, alpha=1 if not far else .85)
    for i, (a, b) in enumerate(zip(w['c'][:-1], w['c'][1:])):
        cel.line(T(spl(q[a:b + 1], n=6)), 2.8 * P['lw'], INK, (.25, .5), seed=seed + 50 + i, press=.4, alpha=.92 if not far else .7)
    cel.line(T(spl(q[w['c'][-1]:], n=6)), 3.0 * P['lw'], INK, (.3, .6), seed=seed + 80, press=.4, alpha=.9 if not far else .6)


def body_layer(x, y, s, rot, sx=1.0, sy=1.0, body='level'):
    """drawn body as an RGBA ROI: (rgba, ox, oy)"""
    sp = sprite(body); tex = cel_tex(sp)
    R = int(330 * s * max(sx, sy)) + 6
    ss = 2
    c_, s_ = math.cos(rot), math.sin(rot)
    A = np.array([[c_ * sx, -s_ * sy], [s_ * sx, c_ * sy]]) * s
    k = 1. / BSC
    M = np.zeros((2, 3)); M[:, :2] = A * k * ss
    M[:, 2] = np.array([R, R]) * ss - M[:, :2] @ np.array([sp['ox'], sp['oy']])
    f_ = k * s * ss
    if f_ < .75:                                      # pre-shrink so thin ink lines survive
        tex = cv2.GaussianBlur(tex, (0, 0), .45 / f_)
    big = cv2.warpAffine(tex, M.astype(f32), (2 * R * ss, 2 * R * ss), flags=cv2.INTER_LINEAR)
    return cv2.resize(big, (2 * R, 2 * R), interpolation=cv2.INTER_AREA), int(round(x - R)), int(round(y - R))


# banking wings, seen from behind: (left wing, right wing) about their own shoulders
BANK = [
    (dict(p=[(34, 14), (8, -62), (-44, -142), (-112, -200), (-158, -214), (-142, -168), (-172, -156), (-140, -120), (-158, -94), (-114, -72), (-122, -40), (-72, -24), (-48, 24)], c=(4, 5, 6, 7, 8, 9, 10, 11)),
     dict(p=[(-22, 12), (8, -60), (56, -132), (112, -176), (150, -186), (130, -146), (152, -128), (118, -104), (128, -78), (90, -62), (90, -34), (52, -14), (30, 18)], c=(4, 5, 6, 7, 8, 9, 10, 11))),
    (dict(p=[(26, -18), (-40, -62), (-132, -72), (-222, -40), (-262, -8), (-212, -2), (-228, 30), (-172, 24), (-180, 54), (-124, 40), (-122, 66), (-68, 46), (-30, 36)], c=(4, 5, 6, 7, 8, 9, 10, 11)),
     dict(p=[(-16, -12), (48, -62), (118, -80), (186, -66), (218, -46), (178, -34), (190, -10), (142, -10), (144, 12), (98, 8), (92, 28), (46, 18), (16, 26)], c=(4, 5, 6, 7, 8, 9, 10, 11))),
    (dict(p=[(22, -12), (-40, -22), (-112, 18), (-172, 88), (-192, 140), (-154, 112), (-154, 148), (-114, 114), (-108, 142), (-72, 106), (-56, 128), (-28, 82), (-14, 38)], c=(4, 5, 6, 7, 8, 9, 10, 11)),
     dict(p=[(-14, -8), (44, -26), (108, -2), (156, 54), (172, 98), (140, 78), (140, 108), (106, 82), (100, 106), (68, 78), (56, 94), (30, 60), (16, 30)], c=(4, 5, 6, 7, 8, 9, 10, 11))),
]


SHOULDER = np.array([6., -34.])                       # centre of the smear arc


def flying(img, x, y, s, rot, k, seed=0, sx=1.0, sy=1.0, far=True, lw=1.0, body='level', wscale=1.0, wr=0.):
    B = BODY[body]
    c_, s_ = math.cos(rot), math.sin(rot)
    wa = B['wrot'] + wr; cw, sw = math.cos(wa) * wscale, math.sin(wa) * wscale

    def mkT(sho):
        sho = np.asarray(sho, float)
        def T(q):
            q = np.asarray(q, float)
            q = np.stack([q[..., 0] * cw - q[..., 1] * sw, q[..., 0] * sw + q[..., 1] * cw], -1) + sho
            q = q * [sx, sy] * s
            return np.stack([q[..., 0] * c_ - q[..., 1] * s_ + x, q[..., 0] * s_ + q[..., 1] * c_ + y], -1)
        return T
    P = dict(lw=lw * max(s, .5))
    R = int(470 * s) + 12
    def mk(fn):
        c = Cel(2 * R, 2 * R, int(x - R), int(y - R)); fn(c); over_roi(img, c.rgba(), c.ox, c.oy)
    if body == 'bank':
        Lw, Rw = BANK[k]
        mk(lambda c: wing(c, P, Rw, mkT((62, -48)), True, seed + 5))
        b, ox, oy = body_layer(x, y, s, rot, sx, sy, body); over_roi(img, b, ox, oy)
        mk(lambda c: wing(c, P, Lw, mkT((-8, -30)), False, seed))
        return img
    T = mkT(B['sho'])
    if far: mk(lambda c: wing(c, P, k, T, True, seed + 5))
    b, ox, oy = body_layer(x, y, s, rot, sx, sy, body); over_roi(img, b, ox, oy)
    mk(lambda c: wing(c, P, k, T, False, seed))
    return img


# ----------------------------------------------------------------- acting
# wing lift while the bird still lies on the page: angle of the paper wing
LIFT = [(118, 20), (120, 52), (122, 38)]
# peel: frame -> (phi head, phi tail, bend line head, bend line tail)   degrees / units from the belly edge
PEEL = [(124, (62, 6, 168, 240)), (126, (92, 40, 96, 200)), (128, (108, 78, 24, 120)), (130, (96, 100, -70, 40))]
#  f: (x, y, rot, scale, wing pose, sx, sy, body)
FLY = [(132, (0, 0, -.02, .617, 4, 1.0, 1.0, 'level')),            # in-between: on its feet, the wing comes away from the side
       (134, (1216, 986, .04, .62, 5, 1.08, .88, 'level')),         # crouch
       (136, (1224, 918, .10, .63, 0, .95, 1.06, 'back')),          # thrown up on its tail, wings high
       (138, (1250, 816, .06, .645, 1, .96, 1.07, 'back')),         # first beat, smear
       (140, (1284, 742, -.50, .65, 2, 1, 1, 'stretch')), (142, (1318, 706, -.42, .65, 3, 1, 1, 'stretch')),
       (144, (1346, 696, -.22, .65, 4, 1.01, .99, 'level')), (146, (1370, 704, -.12, .65, 5, 1.02, .97, 'level')),
       (148, (1394, 682, .04, .65, 0, .97, 1.04, 'back')), (150, (1426, 596, -.02, .65, 1, .97, 1.05, 'back')),
       (152, (1466, 522, -.56, .645, 2, 1, 1, 'stretch')), (154, (1508, 478, -.46, .635, 3, 1, 1, 'stretch')),
       (156, (1548, 456, -.26, .61, 5, 1, 1, 'level')),
       (158, (1592, 424, -.16, .58, 0, 1, 1, 'bank')), (160, (1650, 352, -.26, .53, 1, 1, 1, 'bank')), (162, (1716, 276, -.34, .47, 2, 1, 1, 'bank')),
       (164, (1784, 212, -.30, .41, 0, 1, 1, 'bank')), (166, (1858, 138, -.40, .35, 1, 1, 1, 'bank'))]


def key(f, K):
    v = None
    for a, b in K:
        if f >= a: v = b
    return v


def page_pt(C, u, v, z=0.):
    """BIRD units on the page (z metres up) -> screen"""
    k = C.BW / 500.; th = math.radians(8)
    X = C.BP[0] + (u * math.cos(th) + v * math.sin(th)) * k; Y = C.BP[1] - (-u * math.sin(th) + v * math.cos(th)) * k * 1.55
    p, _ = C.cam().project(np.stack([X, Y, np.zeros_like(X) + .012 + z], -1)); return p


def peel_mesh(C, st):
    ph, pt, bh, bt = st
    n, m = 61, 61
    k = C.BW / 500.; th = math.radians(8)
    u, v = np.meshgrid(np.linspace(-262, 238, n), np.linspace(118, -158, m))
    g = smooth(-190, 120, u)
    phi = np.radians(pt + (ph - pt) * g); bl = bt + (bh - bt) * g
    s = 118 - v
    ang = phi * smooth(bl - 50, bl + 70, s)
    ds = (118 + 158) / (m - 1.)
    lie = 1.55 + (1.02 - 1.55) * np.clip(ang / math.radians(80), 0, 1)
    yy = np.cumsum(np.cos(ang) * ds * lie, 0); zz = np.cumsum(np.sin(ang) * ds, 0)
    v0 = 118.
    X = C.BP[0] + (u * math.cos(th) + v0 * math.sin(th)) * k; Y0 = C.BP[1] - (-u * math.sin(th) + v0 * math.cos(th)) * k * 1.55
    P = np.stack([X + math.sin(th) * -yy * k * .3, Y0 + yy * k, zz * k + .012], -1)
    tx = np.dstack([(u + 262) * BSC, (v + 160) * BSC, ang])
    return P, tx, u, v


def cast(P, lamp, zmax=.25):
    z = np.minimum(P[..., 2:3], zmax)
    q = lamp + (P - lamp) * (lamp[2] / np.maximum(lamp[2] - z, .03))
    q[..., 2] = .006
    return q


def peel(img, C, st, f):
    cam = C.cam(); P, tx, u, v = peel_mesh(C, st)
    sp = sprite(True); tex = cel_tex(sp)
    x0, y0, w, h = 900, 740, 560, 400
    # shadow on the page first (lamp at the right)
    lamp = np.array([1.02, 2.02, .42])
    b, z = S3.raster(cam, cast(P, lamp), tx, w=700, h=h, ss=1, ox=x0 - 140, oy=y0)
    a = S3.tex(sp['a'], b[..., :2]) * (z > 0) * smooth(.12, .5, b[..., 2])
    a = cv2.GaussianBlur(a, (0, 0), 3.4) * .58
    sub = img[y0:y0 + h, x0 - 140:x0 + 560]
    sub *= (1 - a[..., None] * (1 - np.array([.52, .44, .58], f32)))
    b, z = S3.raster(cam, P, tx, w=w, h=h, ss=2, ox=x0, oy=y0)
    t = S3.tex(tex, b[..., :2]) * (z > 0)[..., None]
    cov = S3.tex(sp['cov'], b[..., :2]) * (z > 0)
    up = smooth(.22, .30, b[..., 2])[..., None]                       # hard turn from page to paper-in-light
    lay = t * up
    lie = S3.down(cov * (1 - up[..., 0]), 2)
    back = smooth(2.0, 2.06, b[..., 2])[..., None]
    lay[..., :3] = lay[..., :3] * (1 - back * (1 - np.array([.80, .74, .84], f32)))
    sub = img[y0:y0 + h, x0:x0 + w]
    sub = sub * (1 - (1 - transmit(lie)))
    img[y0:y0 + h, x0:x0 + w] = sub
    lay[..., :3] *= np.array([1.02, .985, .925], f32)
    over_roi(img, S3.down(lay, 2), x0, y0)
    # where the shoulder is now, and the local frame for the wing
    i, j = int(np.argmin(np.abs(v[:, 0] - 4))), int(np.argmin(np.abs(u[0] - 40)))
    p, _ = cam.project(P[i, j]); pu, _ = cam.project(P[i, j + 2])
    return p


def wing_flap(img, C, deg, f, shadow=True):
    """the ink wing as a flap of paper lifting off the page"""
    q = np.array([(96, 8), (70, -20), (18, -16), (-40, 18), (-112, 50), (-40, 68), (24, 68), (78, 46)], float)
    q = boil(q, 1.0, f // 2)
    a = math.radians(deg); S = np.array([92., 10.])
    r = S[0] - q[:, 0]; bend = r / 204.
    t = float(smooth(4, 46, deg))
    P0 = page_pt(C, q[:, 0], q[:, 1])
    S0 = page_pt(C, S[:1], S[1:])[0]
    px = C.F / C.BP[1] * C.BW / 500.
    d = q - S
    an = math.radians(deg) * (.55 + .45 * np.clip(-d[:, 0] / 204., 0, 1))       # the tip leads, the root follows: a bend, not a hinge
    P1 = S0 + np.c_[d[:, 0] * np.cos(an) - d[:, 1] * np.sin(an), d[:, 0] * np.sin(an) * 1.0 + d[:, 1] * np.cos(an)] * px
    P = P0 * (1 - t) + P1 * t
    o = spl(P, closed=True, corners=(4,), n=9)
    c = Cel(520, 420, 960, 760)
    if shadow:
        sh = P0 + [-30 * math.sin(a), 4]
        cs = Cel(520, 420, 960, 760, ss=1); cs.fill(spl(sh, closed=True, corners=(4,), n=9), '#ffffff')
        m = cv2.GaussianBlur(cs.rgba()[..., 3], (0, 0), 2.6) * .5
        sub = img[760:1180, 960:1480]; m = m[:sub.shape[0]]
        sub *= (1 - m[..., None] * (1 - np.array([.52, .44, .58], f32)))
    c.fill(o, PAP)
    c.shade(spl(np.vstack([P[3], P[4], P[5], P[6], (P[5] + P[2]) / 2]), closed=True, corners=(1,), n=8), PAP_S, within=o)
    lw = .62
    c.line(spl(P[[0, 1, 2, 3, 4]], n=10), 7.5 * lw, INK, (.2, .6), seed=f // 2, press=.4)
    c.line(spl(P[[4, 5, 6, 7, 0]], n=10), 9.0 * lw, INK, (.35, .25), seed=f // 2 + 3, press=.4)
    m_ = (P[4] + P[5]) / 2
    c.line(spl(np.array([P[4], (P[4] + P[6]) / 2 + [0, -6], (P[6] + P[1]) / 2]), n=9), 3.2 * lw, INK, (.1, .7), seed=f // 2 + 6)
    over_roi(img, c.rgba(), c.ox, c.oy)
    return img


def drops(img, f, C):
    """ink shaken off by the beats; falls, lands, stays"""
    fd = f - f % 2
    r = np.random.default_rng(12)
    c = Cel(); any_ = False
    D = [(138, 1330, 800, 5.0, -3, 4.2), (138, 1200, 770, -4.0, -5, 3.0), (138, 1382, 770, 8.5, -1, 2.4), (140, 1290, 880, 1.0, 2, 3.4), (150, 1500, 600, 6, 1, 3.4), (150, 1420, 690, -2, 4, 2.6), (160, 1660, 400, 2, 2, 2.2)]
    for (t0, x, y, vx, vy, rad) in D:
        if fd < t0: continue
        t = fd - t0
        X, Y = x + vx * t, y + vy * t + 1.15 * t * t
        ground = 985 + (x - 1200) * .10 + rad * 9
        if Y > ground and 400 < X < 1560:
            q = np.c_[np.cos(np.linspace(0, 2 * np.pi, 9)[:-1]) * rad * 1.5, np.sin(np.linspace(0, 2 * np.pi, 9)[:-1]) * rad * .46]
            c.fill(spl(q + np.random.default_rng(int(x)).normal(0, .3, q.shape) + (X, ground), closed=True), INK, .92); any_ = True
        elif Y < H + 20:
            VY = vy + 2.3 * t; sp = math.hypot(vx, VY); an = math.atan2(VY, vx); L = rad * (1 + min(2.2, sp / 14.))
            q = np.array([[-L * 1.5, 0], [-L * .3, rad * .62], [L * .6, rad * .75], [L, 0], [L * .6, -rad * .75], [-L * .3, -rad * .62]])
            c.fill(spl(xf(q, 1, rot=an, tx=X, ty=Y), closed=True, corners=(0,)), INK); any_ = True
    return comp.over(img, c.layer()) if any_ else img


def smear(img, x, y, s, rot, seed, warm=1.0):
    """dry-brush speed lines swept by the wing on the fastest drawing"""
    r = np.random.default_rng(seed)
    c_, s_ = math.cos(rot), math.sin(rot)
    T = lambda q: np.stack([(q[..., 0] * c_ - q[..., 1] * s_) * s + x, (q[..., 0] * s_ + q[..., 1] * c_) * s + y], -1)
    c = Cel(int(900 * s) + 20, int(900 * s) + 20, int(x - 450 * s), int(y - 450 * s))
    a0, a1 = math.radians(-98), math.radians(-14)
    for j in range(15):
        R = r.uniform(120, 330); b0 = a0 + (a1 - a0) * r.uniform(0, .45); b1 = b0 + (a1 - a0) * r.uniform(.25, .55)
        t = np.linspace(b0, min(b1, a1), 12)
        q = np.c_[np.cos(t) * R, np.sin(t) * R] + SHOULDER
        ink = j % 4 == 0
        c.line(T(q), (r.uniform(3, 12) if not ink else r.uniform(2, 5)) * s, INK if ink else (PAP if j % 3 else PAP_S), (.75, .2), seed=seed + j, press=.6, alpha=.75 if not ink else .85, tmin=.05)
    # stretched wing ghost: the paper dragged along the arc
    t = np.linspace(a0 + .25, a1 - .02, 20)
    out = np.c_[np.cos(t) * (270 + 40 * np.sin(t * 2)), np.sin(t) * (270 + 30 * np.sin(t * 2))] + SHOULDER
    inn = np.c_[np.cos(t) * (90 + 110 * np.linspace(1, 0, 20)), np.sin(t) * (90 + 110 * np.linspace(1, 0, 20))] + SHOULDER
    c.fill(T(np.vstack([out, inn[::-1]])), PAP_S, .55)
    over_roi(img, c.rgba(), c.ox, c.oy)
    return img


def grade(img, before, x, y):
    """one light on the bird: lantern-warm near the page, moon-cool as it climbs"""
    t = float(np.clip((y - 250) / 700., 0, 1))
    d = img - before
    m = (np.abs(d).sum(-1, keepdims=True) > .004)
    tint = np.array([.84, .86, .99], f32) * (1 - t) + np.array([1.02, .985, .925], f32) * t
    return np.where(m, img * tint, img)


def draw(img, f, C):
    fd = f - f % 2
    if f < 118:
        # shiver: two nervous redraws, a hold, two more
        j = {112: 0, 114: 3.4, 116: -4.6}.get(fd, 0)
        sh = None
        if j:
            b = C.P['uv']
            n = np.random.default_rng(fd)
            sh = np.dstack([j * 2.2 * np.sin(b[..., 1] / 70. + n.uniform(0, 6)), j * 2.6 * np.sin(b[..., 0] / 90. + n.uniform(0, 6))]).astype(f32)
        return C.ghost(img, 0., ink=1.0, boil_=sh)
    if f < 124:
        b = C.P['uv']; n = np.random.default_rng(fd)
        sh = np.dstack([3.2 * np.sin(b[..., 1] / 60. + n.uniform(0, 6)), 4.0 * np.sin(b[..., 0] / 90. + n.uniform(0, 6))]).astype(f32)
        img = C.ghost(img, 0., ink=1.0, boil_=sh if fd != 122 else None)
        return wing_flap(img, C, key(f, LIFT), f)
    img = C.ghost(img, 1., ink=0.10)
    if f < 132:
        p = peel(img, C, key(f, PEEL), f)
        return img
    x, y, rot, s, pose, sx, sy, body = key(f, FLY)
    n = np.random.default_rng(fd)
    if fd == 132:                                                   # exactly where the peeled paper stood
        P_, tx_, u_, v_ = peel_mesh(C, PEEL[-1][1])
        i, j = int(np.argmin(np.abs(v_[:, 0]))), int(np.argmin(np.abs(u_[0])))
        p, _ = C.cam().project(P_[i, j]); x, y = p
    rot += n.normal(0, .008); y += n.normal(0, 1.0) if fd > 132 else 0
    # its shadow slides away to the left over the page and lets go
    k = max(0, (fd - 134) / 2)
    if k < 6:
        sxp = 1190 - 46 * k; syp = 1052 - 3 * k
        e = np.c_[np.cos(np.linspace(0, 2 * np.pi, 13)[:-1]) * (120 - 6 * k), np.sin(np.linspace(0, 2 * np.pi, 13)[:-1]) * (22 - k)]
        c = Cel(520, 220, int(sxp - 260), int(syp - 110), ss=1); c.fill(spl(e * (1 + n.normal(0, .06, e.shape)) + (sxp, syp), closed=True), '#ffffff')
        m = cv2.GaussianBlur(c.rgba()[..., 3], (0, 0), 4 + 2.4 * k) * max(0, .60 - .10 * k)
        y0, y1 = c.oy, min(H, c.oy + 220)
        sub = img[y0:y1, c.ox:c.ox + 520]; pg = C.P['page'][y0:y1, c.ox:c.ox + 520]
        sub *= (1 - (m[:y1 - y0] * pg)[..., None] * (1 - np.array([.52, .44, .58], f32)))
    img = drops(img, f, C)
    before = img.copy()
    if pose == 1 and body == 'back': img = smear(img, x, y, s, rot - .30, fd)
    img = flying(img, x, y, s, rot, pose, seed=fd, sx=sx, sy=sy, body=body, wscale=.86 if fd == 132 else 1.0, wr=.10 if fd == 132 else 0., far=fd != 132)
    return grade(img, before, x, y)


if __name__ == '__main__':
    g = np.ones((1080, 1800, 3), f32) * rgb('#2b2945')
    L = [(132, 'level', 4), (134, 'level', 5), (136, 'back', 0), (138, 'back', 1), (140, 'stretch', 2), (142, 'stretch', 3), (144, 'level', 4), (156, 'level', 5), (158, 'bank', 0), (160, 'bank', 1), (162, 'bank', 2)]
    for i, (f, bd, k) in enumerate(L):
        r_ = [v for a_, v in FLY if a_ == f][0][2]
        flying(g, 220 + (i % 4) * 440, 200 + (i // 4) * 340, .58, r_, k, seed=i, body=bd)
    comp.save('qa/s4/bird_poses.jpg', np.clip(g, 0, 1))
