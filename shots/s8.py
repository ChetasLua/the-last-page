"""S8  The Great Bird.  8a the flock draws it, 8b across one wing, 8c the widest image: one wingbeat, and away."""
import numpy as np, cv2, math, os
from scipy.interpolate import PchipInterpolator, CubicSpline
from lp import W, H
from lp.paint import f32, rgb, mix, smooth, ramp, grid, fbm, Canvas, arcs, jit_color, mottle
from lp import comp, gb_sky as G, greatbird as gb
from lp.cam import Camera3D, rot_z, rot_x, rot_y
from lp.ink import Cel, spl, over_roi, noise1

D2R = math.radians
PW, PH = G.PW, G.PH
OX, OY = (PW - W) // 2, (PH - H) // 2            # frame origin inside a plate
VER = 'v7'


def cached(name, fn):
    os.makedirs('cache', exist_ok=True)
    p = f'cache/s8_{name}_{VER}.npy'
    if not os.path.exists(p):
        a = fn(); t = p + f'.{os.getpid()}.tmp.npy'
        np.save(t, a.astype(np.float16)); os.replace(t, p)
    return np.load(p)


def keyset(keys):
    """dict param -> PCHIP over (frame, value) keys"""
    out = {}
    for k, v in keys.items():
        t = np.array([a for a, _ in v], float); y = np.array([np.atleast_1d(b) for _, b in v], float)
        out[k] = (PchipInterpolator(t, y, axis=0), t[0], t[-1])
    return out


def kv(ks, k, f):
    p, a, b = ks[k]
    r = p(np.clip(f, a, b))
    return r if len(r) > 1 else float(r[0])


# ====================================================================== 8c
HZ = 771
SUN = (1560, 640)
CAMC = Camera3D((0, 0, 0), (0, 100, 0), 1400, cx=1024, cy=HZ)


def tower_cel():
    """the library far below: 1 unit = 1 px, foot at (0, 0)"""
    c = Cel(150, 150, -75, -128, ss=5)
    st, sm, sd = '#f0d3a6', '#c9a596', '#8f779b'
    def pole(P, col): c.fill(np.array(P, float), col)
    rock = spl([[-34, 18], [-30, 2], [-24, -4], [-12, -7], [10, -6], [22, -3], [33, 5], [40, 20]], corners=(0, 7))
    c.fill(np.vstack([rock, [[40, 30], [-34, 30]]]), '#8b7596')
    c.fill(spl([[8, -6], [22, -3], [33, 5], [40, 20], [40, 30], [20, 30], [16, 10], [12, 2]], closed=True, corners=(0, 3, 4, 5)), '#b99a94')
    for x in (-26, -18, -9, -1, 6): c.line([[x, 0 + abs(x) * .1], [x + 1, 12], [x - 1, 26]], .9, '#6f5f8a', (.1, .4))
    c.fill(spl([[-30, 2], [-24, -5], [-12, -8], [10, -7], [24, -3], [30, 2], [20, 0], [0, -3], [-18, -1]], closed=True), '#8f9452')
    c.fill(spl([[6, -8], [24, -4], [30, 2], [20, 0], [8, -4]], closed=True), '#cdbb68')
    for (x, y, s) in ((-29, -3, 1.), (27, -4, .9), (33, 4, .7), (-36, 8, .7)):
        c.line([[x, y + 5 * s], [x + 2 * s, y - 2 * s]], 1.2 * s, '#4f4460', (0, .2))
        c.fill(spl([[x - 9 * s, y - 2 * s], [x - 3 * s, y - 7 * s], [x + 5 * s, y - 8 * s], [x + 10 * s, y - 3 * s], [x, y - 1 * s]], closed=True), '#5d6b54')
        c.fill(spl([[x, y - 7 * s], [x + 5 * s, y - 8 * s], [x + 10 * s, y - 4 * s], [x + 4 * s, y - 4 * s]], closed=True), '#a9a35c')
    # body
    body = spl([[-14, 0], [-13, -22], [-12, -44], [-12.5, -60], [12, -60], [12, -44], [13.5, -22], [15, 0]], closed=True, corners=(0, 3, 4, 7))
    c.fill(body, sm)
    c.shade(np.array([[-16, 4], [-16, -62], [-4, -62], [-3, -30], [-2, 4]]), sd, within=body)
    c.shade(np.array([[7, 4], [6.5, -30], [6, -62], [18, -62], [18, 4]]), st, within=body)
    # turret
    t = np.array([[13, 0], [13, -36], [20, -36], [21, 0]]); c.fill(t, sm); c.fill(np.array([[18, 0], [17.5, -36], [20, -36], [21, 0]]), st)
    c.fill(np.array([[11, -36], [16.5, -54], [22, -36]]), '#7d6f93'); c.fill(np.array([[17, -36], [16.5, -54], [22, -36]]), '#c99f8a')
    for y in (-16, -28, -40): 
        for x in (-8, 0, 7): c.fill(spl([[x - 1.2, y], [x - 1.2, y - 5], [x, y - 6.6], [x + 1.2, y - 5], [x + 1.2, y]], closed=True, corners=(0, 4)), '#5c4a78')
    c.fill(np.array([[-15, -46], [-15, -49], [15, -49], [15, -46]]), '#d8b799')
    for x in np.arange(-11, 12, 3.6): c.fill(np.array([[x, -50.5], [x, -57], [x + 1.6, -57], [x + 1.6, -50.5]]), '#65527f')
    c.fill(np.array([[-15.5, -58], [-15.5, -62], [15.5, -62], [15.5, -58]]), '#e4c6a0'); c.fill(np.array([[-15.5, -58], [-15.5, -62], [-3, -62], [-3, -58]]), '#a38aa0')
    dome = spl([[-13, -62], [-15.5, -69], [-14, -77], [-8, -86], [0, -91], [8, -86], [14, -77], [15.5, -69], [13, -62]], closed=True, corners=(0, 8))
    c.fill(dome, '#77a08a')
    c.shade(spl([[-18, -60], [-18, -90], [-3, -93], [-6, -80], [-5, -60]], closed=True, corners=(0, 1, 2, 4)), '#4f7478', within=dome)
    c.shade(spl([[6, -60], [8, -74], [4, -90], [18, -92], [20, -60]], closed=True, corners=(0, 2, 3, 4)), '#c3cf9c', within=dome)
    for x in (-9, -3, 3, 9): c.line(spl([[x, -62], [x * 1.12, -72], [x * .6, -84], [0, -91]]), .6, '#47696e', (.1, .3), alpha=.7)
    c.fill(np.array([[-3, -90], [-3, -95], [3, -95], [3, -90]]), '#d8b799'); c.fill(np.array([[-4.5, -95], [0, -101], [4.5, -95]]), '#6c9684'); c.fill(np.array([[0, -95], [0, -101], [4.5, -95]]), '#c3cf9c')
    c.line([[0, -100], [0.4, -109]], 1.0, '#f3d9a4', (0, .4))
    ink = '#5b4163'
    for P in (body, dome): c.line(np.vstack([P, P[:1]]), .55, ink, (0, 0), alpha=.75)
    return c.rgba(), c.ox, c.oy


def plates_c():
    r = np.random.default_rng(21)
    h, w = PH, PW; hz = HZ + OY; sun = (SUN[0] + OX, SUN[1] + OY)
    img = G.sky(h, w, [(0, '#548fbf'), (.20, '#7aaec8'), (.40, '#aecbcc'), (.56, '#dfd8bf'), (.70, '#f6d6ab'), (.86, '#fbcf95'), (1, '#fcc98a')], hz, sun, seed=4, warm='#ffe9c0', warm_r=700, warm_a=.85)
    img = G.paint_sun(img, sun, 44, r, rings=(('#fff3d4', 1.5, .62), ('#ffeac0', 2.3, .44), ('#ffe3b2', 3.6, .32), ('#fedcaa', 5.8, .22)))
    img = G.cirrus(img, [dict(x=1500, y=240, ang=.26, n=120, len=520, spread=(430, 44), col='#f3ead6', op=.20, curve=-.26, hw=7),
                         dict(x=880, y=180, ang=.20, n=90, len=460, spread=(380, 34), col='#eeeadd', op=.17, curve=-.22, hw=6),
                         dict(x=1180, y=430, ang=.12, n=90, len=560, spread=(480, 24), col='#fbe9ca', op=.26, curve=-.12, hw=6),
                         dict(x=2000, y=470, ang=.16, n=60, len=420, spread=(260, 24), col='#f9e3c4', op=.24, curve=-.14, hw=6),
                         dict(x=700, y=610, ang=.04, n=60, len=420, spread=(520, 18), col='#fde4ba', op=.30, curve=-.05, hw=5)], r)
    sky = img
    far = np.zeros((h, w, 4), f32)
    S = lambda pr, sd, ld, ft, tn, **k: G.comp_over(far, G.cloud(h, w, pr, r, tn, ld, seed=sd, foot=ft, **k))
    ft = G.T('#fff0d2', '#f9d9b2', '#e8bfa4', '#d3a9aa', '#c0a0b2')
    S([(1120, 852, 110, 34), (1230, 842, 86, 40), (1010, 862, 100, 24), (1390, 862, 130, 22)], 52, (.75, -.35, .0), (845, 885), ft, thr=(.62, .44, .22, .0), mist='#f9d3a0', ao_h=.3, rimcol='#fff6da', rimw=3, kids=(9, 5), face=4, sizes=(9, 5, 3))
    S([(1990, 800, 140, 84), (2110, 770, 110, 100), (1880, 838, 120, 50), (2240, 800, 150, 96), (2090, 720, 66, 56), (2180, 735, 70, 56), (2350, 840, 120, 60)], 51, (-.80, -.32, .0), (800, 885), ft, thr=(.62, .44, .22, .0), mist='#f8cf9c', ao_h=.34, rimcol='#fff6da', rimw=4, kids=(10, 5), face=5, sizes=(10, 5, 3))
    S([(640, 862, 160, 30), (800, 858, 110, 24), (520, 850, 90, 40)], 57, (.75, -.35, .0), (850, 888), ft, thr=(.62, .44, .22, .0), mist='#f9d3a0', rimcol='#fff6da', rimw=3, kids=(8, 4), face=3, sizes=(8, 4, 3))
    tn = G.T('#fff2d6', '#fbdab4', '#e3b6a3', '#b893a9', '#8f80ab')
    S([(250, 850, 230, 150), (80, 780, 200, 200), (170, 640, 160, 160), (330, 720, 140, 130), (60, 560, 150, 160), (190, 500, 112, 112), (95, 430, 92, 96), (480, 815, 110, 90), (20, 660, 150, 200), (395, 640, 74, 70), (560, 850, 80, 50)], 31, (.80, -.30, .02), (800, 930), tn,
      thr=(.66, .46, .20, -.10), mist='#f4cda4', ao_h=.4, rimcol='#fffbe6', rimw=5, kids=(10, 6), face=6, amb=.2)
    def tf(d): return np.stack([mix(a, b, d ** .7) for a, b in zip(G.T('#fff0c8', '#fbd9aa', '#efc3a2', '#d9aeaa', '#cba6b2'), G.T('#fdeaca', '#f0c8aa', '#c79fab', '#9a86af', '#776fa1'))])
    def mf(d): return mix(rgb('#fad3a0'), rgb('#8d81b1'), float(d) ** .62) + .05 * math.sin(math.pi * min(1., float(d) * 1.6)) * np.array([1., .1, .3], f32)
    rf0 = G.ridge_field(7)
    def rf(x, dep, w_):
        v = rf0(x, dep, w_)
        # a river of shadow winding from the tower toward the sun; tall banks at lower left
        cx = .10 + .62 * (1 - dep) ** .8 + .07 * math.sin(dep * 7)
        v -= .55 * math.exp(-((x / w_ - cx) / (.05 + .07 * dep)) ** 2) * (dep > .03)
        return v + .10
    sea = G.sea(h, w, hz, 1430, r, sun, tf, mf, rows=17, seed=3, height=rf)
    # haze where the sea meets the sky
    y = np.arange(h, dtype=f32)[:, None]
    k = np.exp(-np.maximum(0, y - hz) / 26.)[..., None] * .75
    sea[..., :3] += (rgb('#fbd5a2') * sea[..., 3:4] - sea[..., :3]) * k
    # the tower, with a bank drawn up round its foot
    t, ox, oy = tower_cel()
    ts = .62; t = cv2.resize(t, None, fx=ts, fy=ts, interpolation=cv2.INTER_AREA); ox, oy = int(ox * ts), int(oy * ts)
    t[..., :3] = t[..., :3] * .86 + rgb('#e9c3a6') * t[..., 3:4] * .14
    tx, ty = 318 + OX, 900 + OY
    over4(sea, t, tx + ox, ty + oy)
    bank = G.cloud(h, w, [(tx - 50, ty + 26, 56, 20), (tx + 40, ty + 28, 50, 18), (tx - 4, ty + 34, 60, 17), (tx + 92, ty + 38, 50, 16), (tx - 112, ty + 38, 56, 19)], r, tf(.62), (.6, -.74, -.2), seed=77, foot=(ty + 30, ty + 60), thr=(.66, .48, .26, .04), kids=(8, 5), face=2, sizes=(7, 4, 2.6), mist='#b99db3')
    G.comp_over(sea, bank)
    # near bank, its own parallax
    near = np.zeros((h, w, 4), f32)
    nt = G.T('#fbe3c4', '#e3b7a8', '#b08fb0', '#877aad', '#6a659b')
    G.comp_over(near, G.cloud(h, w, [(1900, 1330, 260, 110), (2180, 1300, 260, 150), (1640, 1370, 240, 80), (2400, 1330, 200, 150), (1400, 1390, 200, 60)], r, nt, (-.35, -.82, -.28), seed=91, foot=(1320, 1420), thr=(.66, .50, .30, .06), kids=(10, 6), face=5, sizes=(17, 9, 4.5), rimcol='#fff2cf', rimw=4))
    G.comp_over(near, G.cloud(h, w, [(60, 1330, 220, 90), (270, 1370, 200, 70), (-40, 1280, 160, 90)], r, nt, (.45, -.80, -.25), seed=92, foot=(1330, 1420), thr=(.66, .50, .30, .06), kids=(9, 6), face=4, sizes=(15, 8, 4.5), rimcol='#fff2cf', rimw=4))
    def tl(d): return np.stack([mix(a, b, d ** .7) for a, b in zip(G.T('#d9bcb8', '#c4a7b4', '#a992b2', '#9184b0', '#837cad'), G.T('#b9a3be', '#9c8fb8', '#817cae', '#6a69a0', '#5d5f99'))])
    low = G.sea(h, w, hz, 1500, np.random.default_rng(8), sun, tl, lambda d: mix(rgb('#c2a6b4'), rgb('#6d6ca4'), float(d) ** .5), rows=20, lobe=130, seed=13, lane=.25)
    return sky, far, sea, near, low[..., :3]


def over4(dst, src, ox, oy):
    """premult over premult in place"""
    h, w = dst.shape[:2]; sh, sw = src.shape[:2]
    x0, y0 = max(0, ox), max(0, oy); x1, y1 = min(w, ox + sw), min(h, oy + sh)
    if x1 <= x0 or y1 <= y0: return
    s = src[y0 - oy:y1 - oy, x0 - ox:x1 - ox]; d = dst[y0:y1, x0:x1]
    dst[y0:y1, x0:x1] = d * (1 - s[..., 3:4]) + s


# ---------------------------------------------------------------- act 8c
K = keyset(dict(
    elev=[(76, (.04, -.08, -.22)), (84, (.34, .22, .00)), (90, (.52, .47, .28)), (93, (.56, .50, .46)), (96, (.55, .48, .40)), (100, (.57, .50, .43)), (102, (.61, .54, .50)),
          (104, (.46, .52, .56)), (106, (.08, .26, .44)), (108, (-.30, -.10, .14)), (110, (-.52, -.44, -.22)), (112, (-.58, -.60, -.50)), (115, (-.55, -.62, -.66)), (119, (-.51, -.56, -.58)), (123, (-.44, -.50, -.52)),
          (129, (.05, -.10, -.34)), (133, (.30, .20, -.02)), (137, (.40, .34, .30)), (140, (.22, .28, .34)), (144, (-.16, -.08, .10)), (147, (-.24, -.24, -.20)), (152, (.10, .04, -.06)), (158, (.24, .18, .10)), (166, (.20, .16, .10)), (174, (.30, .22, .08)), (179, (.10, .10, .12)), (184, (.16, .10, .02)), (188, (.40, .34, .20)), (192, (.52, .46, .34))],
    adv=[(76, .06), (84, .34), (90, .47), (94, .55), (100, .57), (102, .63), (104, .50), (106, .32), (108, .06), (111, -.12), (116, -.18), (124, -.12), (131, .05), (137, .16), (143, -.06), (150, .02), (192, -.05)],
    bend=[(76, -.28), (84, -.42), (90, -.10), (93, .44), (96, .30), (100, .26), (102, .16), (104, .70), (106, 1.10), (108, 1.00), (110, .62), (112, .14), (115, -.20), (119, -.10), (124, -.06), (129, -.40), (134, -.10), (137, .24), (140, .55), (143, .60), (147, .16), (152, .22), (160, .34), (192, .26)],
    fold=[(76, .22), (84, .32), (90, .06), (93, 0), (124, 0), (129, .38), (134, .12), (138, 0), (192, .06)],
    fan=[(76, .92), (84, .88), (92, 1.0), (94, 1.10), (98, 1.04), (102, 1.06), (106, 1.0), (112, 1.06), (124, 1.0), (130, .86), (138, 1.04), (150, 1.0), (192, .98)],
    ruffle=[(76, 0), (100, 0), (103, .2), (106, 1.0), (110, .9), (116, .35), (124, .1), (138, .1), (141, .5), (146, .1), (192, 0)],
    pitch=[(76, .60), (92, .68), (100, .70), (104, .64), (110, .44), (116, .36), (122, .22), (128, -.12), (132, -.14), (136, .02), (140, .36), (144, .74), (149, .76), (154, .46), (159, .18), (164, .08), (169, .20), (174, .34), (180, .46), (192, .42)],
    yaw=[(76, 152), (100, 150), (116, 150), (122, 142), (128, 108), (132, 84), (136, 62), (140, 36), (144, 2), (149, -44), (154, -70), (159, -80), (164, -77), (169, -66), (175, -40), (180, -27), (185, -20), (192, -19)],
    roll=[(76, 0), (119, 0), (126, .40), (131, .66), (136, .74), (140, .70), (144, .62), (149, .74), (154, .88), (159, .90), (164, .90), (169, .76), (174, .36), (179, -.06), (185, -.20), (192, -.10)],
    neck=[(76, .0), (100, .0), (106, .12), (114, .06), (120, .10), (130, .62), (138, .9), (192, .95)],
    nod=[(76, -.20), (92, -.30), (102, -.36), (108, -.22), (116, -.20), (126, -.02), (192, .05)],
    turn=[(76, -.10), (100, -.14), (118, -.10), (130, 0), (192, 0)],
    tailfan=[(76, .85), (100, .95), (106, 1.3), (116, .9), (127, .24), (192, .24)],
    tailpitch=[(76, -.1), (100, -.2), (106, .35), (116, .1), (130, 0), (192, 0)],
    L=[(76, (.60, .42, .68)), (130, (.60, .42, .68)), (140, (.80, .10, .50)), (147, (.94, -.16, .34)), (166, (.94, -.14, .34)), (176, (.62, .34, .55)), (192, (.5, .6, .5))],
))
PATH_T = [76, 84, 92, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 149, 154, 159, 164, 169, 175, 180, 185, 192]
PATH_P = np.array([(.0, 27.6, 2.75), (.08, 27.6, 2.55), (.12, 27.6, 2.2), (.12, 27.6, 2.0), (.12, 27.6, 2.1), (.10, 27.4, 3.1), (.06, 27.2, 4.4), (.02, 27.2, 5.2), (-.1, 27.5, 5.5), (-.7, 28.4, 5.35),
                   (-3.0, 30.6, 4.3), (-7.4, 34.2, 2.7), (-12.6, 38.6, 1.7), (-16.6, 42.8, 2.4), (-17.2, 46.0, 5.6), (-14.2, 48.0, 10.4), (-8.4, 49.2, 14.5), (-.4, 50.2, 16.9), (8.6, 52, 17.0), (17.6, 57, 15.7), (30, 80, 15.3), (45, 118, 16.6), (64, 168, 18.3), (97, 232, 22.0)])
_PX = PchipInterpolator(PATH_T, PATH_P, axis=0)
_PXs = CubicSpline(PATH_T[9:], PATH_P[9:], axis=0)


def path(f):
    f = float(np.clip(f, 76, 192))
    a = _PX(f)
    if f <= 124: return a
    k = min(1., (f - 124) / 6.)
    return a * (1 - k) + _PXs(f) * k


def pose_c(f):
    p = gb.pose(elev=tuple(kv(K, 'elev', f)), twist=(.10, .02), yaw=D2R(kv(K, 'yaw', f)), pos=tuple(path(f)),
                **{k: kv(K, k, f) for k in ('adv', 'bend', 'fold', 'fan', 'pitch', 'roll', 'neck', 'nod', 'turn', 'tailfan', 'tailpitch', 'ruffle')})
    # the far wing lags a hair: two wings never match exactly
    e = kv(K, 'elev', f - .7); p['elevL'] = tuple(e); p['bendL'] = kv(K, 'bend', f - .7)
    return p


HANG = np.array([(0, 0, 0), (.5, .3, -2.1), (.1, .8, -4.2), (-1.8, 1.4, -6.0), (-5.0, 2.2, -6.8), (-8.8, 3.0, -6.3), (-12.4, 3.8, -4.6), (-15., 4.4, -2.4), (-17.5, 5, -.4)])
_SEQ = None


def plume_curves(B, f):
    """plumes are ropes: they follow exactly where the tail root has been, then the slow S they hung in"""
    p = pose_c(f); roots, R = B.rib_roots(p)
    side = R @ np.array([1., 0, 0])
    hist = np.arange(f - .5, 103.9, -.5) if f > 104.5 else np.zeros(0)
    pr = []
    for t in hist:
        rr, _ = B.rib_roots(pose_c(t)); pr.append(rr)
    pr = np.array(pr) if len(pr) else np.zeros((0, len(roots), 3))
    f0 = min(f, 104.)
    r0, R0 = B.rib_roots(pose_c(f0))
    out = []
    from lp.ink import spl as _spl
    for k, rb in enumerate(B.rib):
        u = (k - 4) / 4.
        s = HANG * np.array([-1., 1, 1])           # the S drifts to the right, toward the light
        q = np.linspace(0, 1, len(s))
        sc_ = [.80, .92, 1.0, .86, 1.06, .90, 1.04, .94, .82][k]
        ph = u * .62 * q ** .8 - .10 + .05 * math.sin(k * 2.3)
        x_, z_ = s[:, 0] * sc_, s[:, 2] * sc_
        s = np.c_[x_ * np.cos(ph) - z_ * np.sin(ph), s[:, 1] + u * 1.6 * q, x_ * np.sin(ph) + z_ * np.cos(ph)]
        s = s + np.c_[.55 * np.sin(q * 4.4 + k * 1.1 - f0 * .045) * q, 0 * q, .60 * np.sin(q * 3.8 + k * 1.9 - f0 * .05) * q]
        c = np.stack([np.interp(np.linspace(0, 1, 70), q, s[:, j]) for j in range(3)], -1)
        for j in range(3): c[:, j] = np.convolve(np.pad(c[:, j], 6, 'edge'), np.ones(13) / 13, 'valid')
        c = c - c[0] + r0[k]
        P = np.vstack([roots[k][None], pr[:, k], c[1:]]) if len(pr) else np.vstack([roots[k][None], c[1:]])
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        x = np.linspace(0, min(rb['L'], d[-1]), 60)
        P = np.stack([np.interp(x, d, P[:, j]) for j in range(3)], -1)
        # slack: smooth, spread with distance, ripple running backward
        g_ = np.exp(-np.linspace(-2, 2, 19) ** 2); g_ /= g_.sum()
        for _ in range(1 if f < 106 else 3 + int(np.clip((f - 123) * 1.1, 0, 12))):
            S_ = P.copy()
            for j in range(3): S_[:, j] = np.convolve(np.pad(P[:, j], 9, 'reflect', reflect_type='odd'), g_, 'valid')
            kk = np.clip(np.arange(len(P)) / 8., 0, 1)[:, None]
            P = P * (1 - kk) + S_ * kk
        if f > 124:
            # under way the air takes over: they stream straight out behind, each with its own lag, in long S curves
            lag = 5.0 + 1.1 * ((k * 5) % 7) / 6. * 4
            ks = np.linspace(0, 1, len(P))
            vel = np.array([path(f - lag * q) - path(f - lag * q - 2.5) for q in ks]); vel = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-6)
            st = np.vstack([np.zeros(3), np.cumsum(-vel[:-1] * (x[-1] / (len(P) - 1)), 0)]) + roots[k]
            wk = float(np.interp(f, [124, 134, 141, 166, 176], [0, .45, .92, .92, .6]))
            P = P * (1 - wk) + st * wk
        a = (x / rb['L'])
        if f > 104:
            g = min(1., (f - 104) / 22.)
            g2 = float(np.interp(f, [126, 140], [0, 1]))
            P = P + g * (np.array([0, 0, 1.]) * (np.sin(a * 5.2 - f * .32 + k * 1.3) * (.55 + .55 * g2) * a)[:, None] + side * ((u * (1.5 + 2.2 * g2) * a ** .9 + np.sin(a * 4.5 - f * .27 + k) * (.4 + .5 * g2) * a))[:, None])
        out.append((P, side))
    return out


def loose_pages(f, n=360, seed=5):
    """pages torn off by the stroke, and the ones always drifting round the bird"""
    r = np.random.default_rng(seed)
    VV, NN, FF, AA = [], [], [], []
    g = np.array([-.5, 0, .5])
    p6 = np.asarray(path(106.))
    for k in range(n):
        kind = 0 if k < 62 else 1 if k < 76 else 2            # torn off / drifting / left along the arc
        tb = r.uniform(103.5, 113) if kind == 0 else r.uniform(20, 76) if kind == 1 else ARC0 + 2 + 44 * r.random() ** 1.7
        ph = r.uniform(0, 6.28, 4); sz = .20 + .55 * r.random() ** 2.0; fr = r.uniform(.5, 1.1)
        u1, u2, u3 = r.random(3); n3 = r.normal(0, 1, 3); n4 = r.normal(0, 1, 3)
        t = f - tb
        if kind == 2: sz = .26 + .62 * r.random() ** 2.2
        if t < 0: continue
        fall = 0.
        if kind == 0:
            sd = -1 if u1 < .5 else 1
            q = .25 + 1.1 * u2; ang = D2R(150)
            lat = np.array([math.cos(ang), math.sin(ang), 0]) * sd
            p0 = p6 + lat * (2 + 12.5 * q) + np.array([0, 0, -1.5 + 5.5 * u3 - 1.2 * (tb - 106)]) + n3 * .8
            v0 = np.array([0, 0, -.5 - 1.1 * u1]) + lat * (.25 * u3) + n4 * .14
            dr = .10
        elif kind == 1:
            a = -.7 + 2.0 * u1 + (3.14 if u3 < .4 else 0); R0 = 9 + 9 * u2
            p0 = p6 + np.array([math.cos(a) * R0, math.sin(a) * R0 * .9, -5 + 9 * u3 * u2]); v0 = np.array([-math.sin(a), math.cos(a) * .6, .12]) * .10
            dr = .004
        else:
            q = path(tb); v = np.asarray(path(tb + .5)) - np.asarray(path(tb - .5))
            old = float(np.clip((160 - tb) / 22., .25, 1))
            p0 = q + n3 * np.array([2.0, 1.5, 1.8]) * (.45 + 1.1 * u1 * old); v0 = v * .30 * u2 + n4 * .07
            dr = .09
            tf = max(0., f - (172 + 8 * u3 - 6 * old)); fall = .5 * .030 * tf ** 1.75
        if t < 0: continue
        pos = p0 + v0 * (1 - math.exp(-dr * t)) / dr + np.array([.012 * t + .6 * math.sin(t * .09 + ph[0]), .4 * math.sin(t * .07 + ph[1]), (.5 * math.sin(t * .11 + ph[2])) - (.010 * t if kind == 0 else .004 * t if kind == 2 else 0) - fall])
        an = np.array([ph[0] + t * fr * .22, ph[1] + t * fr * .15, ph[2] + t * fr * .11])
        Rm = rot_z(an[0]) @ rot_x(an[1]) @ rot_y(an[2])
        a, b = np.meshgrid(g, g, indexing='ij')
        c = .34 * math.sin(t * .30 + ph[3])
        V = np.stack([a * sz * .74, b * sz, c * sz * (4 * b * b - .5) + .30 * sz * a * b * math.sin(ph[0])], -1) @ Rm.T + pos
        VV.append(V); NN.append(Rm[:, 2]); FF.append(70000 + k); AA.append(int(r.integers(0, 1 << 20)))
    if not VV: return None
    m = len(VV)
    return dict(V=np.array(VV), N=np.array(NN), fid=np.array(FF), thin=np.ones(m, np.int8), part=np.full(m, 10, np.int8), art=np.array(AA), two=np.full(m, .18, f32))


# -------------------------------------------------------------- trail of light, mist and paper
ARC0 = 141.5            # where the arc of light begins: low on the left


def trail_strokes():
    """a dozen torn streamers, each with its own wander, length and gaps"""
    r = np.random.default_rng(88)
    ts = np.arange(130, 203.01, .5)
    C = np.array([path(min(x, 192.)) + max(0, x - 192.) * (path(192.) - path(191.)) for x in ts])
    xy, z = CAMC.project(C); ppu = 1400. / np.maximum(C[:, 1], 1)
    tg = np.gradient(xy, axis=0); tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-6; nm = np.c_[-tg[:, 1], tg[:, 0]]      # +normal = inside of the arc (lower right)
    S = dict(P=[], w=[], c=[], op=[], dry=[], t=[], sh=[], n=[], s=[], lay=[], brk=[])
    def add(t0, dt, offn, w, col, op, dry, sh, lay):
        q = np.linspace(t0 - dt / 2, t0 + dt / 2, 9); q = np.clip(q, ARC0 - .5, 202)
        f_ = lambda A: np.interp(q, ts, A)
        pp = f_(ppu); o = offn(q) * pp
        P = np.c_[f_(xy[:, 0]) + f_(nm[:, 0]) * o, f_(xy[:, 1]) + f_(nm[:, 1]) * o]
        S['P'].append(P); S['w'].append(w * pp[4]); S['c'].append(col); S['op'].append(op); S['dry'].append(dry); S['t'].append(min(t0 + dt * .30, 191)); S['sh'].append(sh)
        S['n'].append((f_(nm[:, 0])[4], f_(nm[:, 1])[4])); S['s'].append(float(offn(np.array([t0]))[0])); S['lay'].append(lay); S['brk'].append(r.random())
    J = lambda h, v=.012: np.clip(rgb(h) + r.normal(0, v, 3), 0, 1)
    for j in range(13):
        a, b, c, d = r.normal(0, .85), r.uniform(.10, .42), r.uniform(.14, .30), r.uniform(0, 6.3)
        e, g, h_ = r.uniform(.2, .55), r.uniform(0, 6.3), r.uniform(.35, .75)           # gap pattern
        t_in, t_out = ARC0 + r.uniform(0, 12) * (j % 3 > 0), 192 - r.uniform(0, 14) * (j % 4 > 1)
        wt_ = r.uniform(.5, 1.3) * (1.25 if abs(a) < .5 else .75)
        off = lambda q, a=a, b=b, c=c, d=d: a * (1 - .25 * np.clip((q - 170) / 22, 0, 1)) + b * np.sin(q * c + d) + .10 * np.sin(q * 1.1 + d * 2)
        on = lambda t, e=e, g=g, h_=h_: math.sin(t * e + g) + .6 * math.sin(t * e * 2.3 + g * 1.7) > -h_
        n = int(16 * wt_) + 7
        for k in range(n):
            t = t_in + (t_out - t_in) * r.random()
            if not on(t): continue
            dt = r.uniform(1.8, 5.5); w = r.uniform(.06, .19) * wt_; m = r.normal(0, .05)
            o2 = lambda q, off=off, m=m, w=w: off(q) + w * 1.0 + m
            add(t, dt * r.uniform(.6, 1.), o2, w * r.uniform(.7, 1.2), J(['#cda4b4', '#bd9cba', '#dcb4b4'][k % 3]), r.uniform(.30, .56), r.uniform(.1, .5), 2, 0)
            add(t + r.normal(0, .8), dt, lambda q, off=off, m=m: off(q) + m, w * r.uniform(.7, 1.1), J(['#fbdca2', '#f9d08e', '#fde6ba', '#f6d0a6'][k % 4]), r.uniform(.36, .62), r.uniform(.1, .5), 2, 1)
            if r.random() < .75: add(t + r.normal(0, .8), dt * r.uniform(.7, 1.3), lambda q, off=off, m=m, w=w: off(q) - w * .7 + m, w * r.uniform(.25, .6), J(['#fff4d8', '#fff9e8', '#ffefca'][k % 3]), r.uniform(.40, .75), r.uniform(.0, .4), 2, 2)
            if r.random() < .16: add(t, dt * 1.8, lambda q, off=off, m=m: off(q) - .05 + m, r.uniform(.015, .035), J('#fffdf2'), r.uniform(.5, .85), 0, 2, 3)
    for k in range(110):      # mist the old end dissolves into
        t0 = ARC0 + 34 * r.random() ** 1.2; a = r.normal(0, 1.5)
        add(t0, r.uniform(2.5, 6), lambda q, a=a: a + 0 * q, r.uniform(.14, .50), J(['#f7e3c9', '#efd3c0', '#dcc3cb'][k % 3]), r.uniform(.10, .26), r.uniform(.5, .85), 0, 4)
    return {k: np.array(v) for k, v in S.items()}


_TR = None


def draw_trail(img, f, M):
    global _TR
    if f < ARC0: return img
    if _TR is None: _TR = trail_strokes()
    S = _TR
    d3 = f - f % 3
    # its own light, transmitted: a soft bloom under the paint
    cps = [t for t in np.arange(ARC0, min(f, 192.) + .01, 1.)]
    if len(cps) > 2:
        q4 = np.zeros((H // 4, W // 4), f32)
        C = np.array([path(t) for t in cps]); xy, _ = CAMC.project(C); xy = xy @ M[:, :2].T + M[:, 2]; pp = 1400. / C[:, 1] * M[0, 0]
        for i in range(len(cps) - 1):
            fresh = .45 + .55 * math.exp(-(f - cps[i]) / 26.)
            cv2.line(q4, tuple(np.int32(xy[i] / 4)), tuple(np.int32(xy[i + 1] / 4)), float(fresh), int(max(1, pp[i] * 2.0 / 4)), cv2.LINE_AA)
        q4 = cv2.GaussianBlur(q4, (0, 0), 13) * .8 + cv2.GaussianBlur(q4, (0, 0), 5) * .4
        g = cv2.resize(q4, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
        img = 1 - (1 - img) * (1 - np.clip(g * rgb('#ffd49a') * .17, 0, .8))
        img = img.astype(f32)
    for lay in (4, 0, 1, 2, 3):
        for sh in (0, 2):
            m = (S['lay'] == lay) & (S['sh'] == sh) & (S['t'] < f)
            if not m.any(): continue
            age = f - S['t'][m]; ageb = np.maximum(d3 - S['t'][m], 0)
            P = S['P'][m].copy()
            # the band opens and drifts as it ages; each stroke slides on its own
            P = P + (S['n'][m] * (S['s'][m] * age * .20)[:, None])[:, None] + np.stack([age * .12, age * -.06], -1)[:, None]
            wob = np.sin(np.arange(9)[None] * .7 + S['t'][m][:, None] * 3. + ageb[:, None] * .05)
            P = P + S['n'][m][:, None] * (wob * np.minimum(ageb * .16, 9.)[:, None])[..., None]
            # fresh strokes are still being laid; old ones break up and shorten from both ends
            g = np.clip(age / 4., .25, 1)
            brk = np.clip((age - 9 - 20 * S['brk'][m]) / 18., 0, 1) * (1. if lay != 4 else 0.)
            a0 = .5 * brk * .75; a1 = g * (1 - .5 * brk * .75)
            Q = np.empty_like(P)
            for i in range(len(P)):
                x = np.linspace(a0[i], max(a1[i], a0[i] + .05), 9) * 8; j = np.minimum(x.astype(int), 7); fr = (x - j)[:, None]
                Q[i] = P[i][j] * (1 - fr) + P[i][j + 1] * fr
            Q = Q @ M[:, :2].T + M[:, 2]
            op = S['op'][m] * np.clip(age / 2., 0, 1) * (1 - .66 * brk)
            if lay == 4: op = S['op'][m] * np.clip((age - 6) / 16., 0, 1) * (1 - .25 * np.clip((age - 30) / 24., 0, 1))
            dry = np.clip(S['dry'][m] + .45 * brk, 0, .9)
            wd = np.maximum(S['w'][m], .7) * M[0, 0] * (1 + .006 * age) * (1 - .30 * brk)
            o = np.argsort(S['t'][m])
            cv = Canvas(H, W, img=img)
            cv.strokes(Q[o], wd[o], S['c'][m][o], op=op[o], dry=dry[o], seed=9 + lay * 5 + sh, shape='flat' if sh == 0 else 'leaf', bristle=.03)
            img = cv.img
    return img


# -------------------------------------------------------------- size: the bird's shadow on the sea, air in front of its tail
def sea_shadow(img, f, sc, M, seed):
    """the bird's huge soft violet shadow, hand-edged, sliding over the cloud tops"""
    k = float(np.interp(f, [76, 80, 140, 156], [.0 if f < 76 else 1, 1, 1, 0]))
    if k <= 0: return img
    V = sc['V'][(sc['part'] != 10)][:, ::2, ::2].reshape(-1, 4, 3)
    h = V[..., 2] - SEA_Z
    G_ = V + h[..., None] * np.array([-.26, 1.75, -1.0])
    ok = (G_[..., 1].min(1) > 9)
    G_ = G_[ok][:, [0, 1, 3, 2]]
    pp = 1400. / G_[..., 1]
    xy = np.stack([1024 + G_[..., 0] * pp, HZ - SEA_Z * pp], -1) @ M[:, :2].T + M[:, 2]
    m = np.zeros((H // 4, W // 4), np.uint8)
    q = np.round(xy / 4 * 8).astype(np.int32)
    for p_ in q: cv2.fillConvexPoly(m, p_, 255, cv2.LINE_AA, 3)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 7)))
    m = cv2.GaussianBlur(m.astype(f32) / 255., (0, 0), 6.0, sigmaY=2.8)
    r_ = fbm(H // 4, W // 4, (36, 13, 5), (1, .6, .4), 41)
    m = np.clip((m + .075 * r_ * (m > .03) - .10) / .26, 0, 1)
    m = cv2.resize(cv2.GaussianBlur(m, (0, 0), 1.4), (W, H), interpolation=cv2.INTER_LINEAR)
    y = np.arange(H, dtype=f32)[:, None]
    m = m * np.clip((y - (HZ * M[1, 1] + M[1, 2] + 14)) / 40., 0, 1) * k * .86
    sh = img * np.array([.60, .56, .80], f32) + rgb('#8b7bb6') * .09
    return (img + (sh - img) * m[..., None]).astype(f32)


def strand_layer():
    """three thin torn wisps, nearer than the bird: soft-edged, broken, painted thinly"""
    h, w = 190, 1500
    out = np.zeros((h, w, 4), f32)
    for j, (x0, ln, y0, th_) in enumerate(((40, 560, 92, 1.0), (640, 330, 112, .62), (1060, 400, 74, .8))):
        r = np.random.default_rng(17 + j)
        c = Canvas(h, w, color='#f6dcc2', layer=True)
        ax = lambda u: y0 + 9 * np.sin(u * 4.2 + .6 + j) - 14 * (u - .5)
        th = lambda u: np.sin(np.pi * np.clip(u, 0, 1)) ** 1.1 * th_
        for lay, (n, cols, dy, sp, hw, op) in enumerate(((30, ('#b79ab8', '#c7a9ba', '#a990b8'), 8, 5, (1.5, 5.5), (.22, .5)), (36, ('#f3cfb4', '#f7dcc0', '#e6bfb8'), 1, 5, (1.5, 5), (.30, .65)), (30, ('#fff2d8', '#ffe9c4', '#fff8e8'), -5, 3.5, (1, 3.6), (.45, .85)))):
            u = r.beta(1.4, 1.4, n)
            u = u[np.sin(u * 23 + j * 2) > -.55]; n = len(u)                 # gaps
            y = ax(u) + (dy + r.normal(0, 1, n) * sp) * th(u); x = x0 + u * ln
            P = arcs(x, y, r.normal(-.03, .06, n), r.uniform(24, 110, n) * (.35 + .65 * th(u)), r.normal(0, .22, n), K=6)
            c.strokes(P, r.uniform(*hw, n) * (.35 + .65 * th(u)), jit_color(np.stack([rgb(cols[i % 3]) for i in range(n)]), r, .005, .01, .012), op=r.uniform(*op, n), dry=r.uniform(.2, .7, n), seed=3 + lay + 10 * j, shape='leaf', bristle=.05)
        l = np.dstack([c.img * c.a[..., None], c.a])
        out = l + out * (1 - l[..., 3:4])
    out = out.astype(f32)
    return cv2.GaussianBlur(out, (0, 0), 1.5)                                # nearer than the plane of focus


def front_strand(img, f, M, lay):
    """drifts slowly; the stroke's air tears it open and it is gone"""
    k = float(np.interp(f, [76, 108, 115], [.92, .92, 0]))
    if k <= 0: return img
    t = max(0., f - 108.)
    z = M[0, 0] * 1.16
    sx = z * (1 + .030 * t ** 1.3); sy = z * (1 + .02 * t)
    cx = 1190 + (f - 76) * 1.5; cy = 1000 + t * 3.0
    X = np.array([cx, cy]) @ M[:, :2].T + M[:, 2]
    A = np.float32([[sx, 0, X[0] - 750 * sx], [0, sy, X[1] - 95 * sy]])
    l = cv2.warpAffine(lay, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return comp.over(img, l, k)


# -------------------------------------------------------------- the shock in the cloud sea
SEA_Z, SEA_C = -6.2, np.array([.2, 35.5])


def shock_r(f):
    t = max(0., f - 108.5)
    a = 9.0 * (1 - math.exp(-t / 3.2)) * (1 - .40 * float(comp.ease((f - 124) / 30.)))
    ring = 1.6 + a * .9 + 20.0 * (1 - math.exp(-t / 14.))
    return t, a, ring


def shock_streaks(img, f, M):
    """air tearing outward over the cloud tops: dry-brush strokes, drawn on twos"""
    d = f - (f % 2); t, a, ring = shock_r(d)
    if t <= 0 or t > 26: return img
    r = np.random.default_rng(int(d) * 3 + 1)
    k0 = math.sin(math.pi * min(1., (t + 2) / 28.)) ** .7
    P, wd, col, op = [], [], [], []
    for k in range(190):
        th = r.uniform(0, 2 * np.pi); rr = a * .9 + (ring * 1.12 - a * .9) * r.uniform(.0, 1.) ** .8; L = r.uniform(.8, 4.2) * (.5 + .5 * min(1, t / 6.))
        x = np.linspace(rr - L * .5, rr + L * .5, 6)
        bow = .18 * np.sin(np.linspace(0, 3.1, 6)) * L * .35
        wx = SEA_C[0] + np.cos(th) * x; wy = SEA_C[1] + np.sin(th) * x; wz = SEA_Z + .25 + bow
        if wy.min() < 11: continue
        pp = 1400. / wy
        q = np.c_[1024 + wx * pp, HZ - wz * pp] @ M[:, :2].T + M[:, 2]
        P.append(q); wd.append(r.uniform(.03, .12) * pp.mean()); col.append(rgb(['#fff4dc', '#fde2bd', '#e9c9c0', '#c9b0c4'][int(r.integers(0, 4))])); op.append(k0 * r.uniform(.25, .70))
    if not P: return img
    cv = Canvas(H, W, img=img)
    cv.strokes(np.array(P), np.array(wd), np.array(col), op=np.array(op), dry=r.uniform(.25, .7, len(P)), seed=int(d), shape='leaf', bristle=.05)
    return cv.img


def shock_warp(sea, f, M, low=None):
    """the banks are pushed outward round the hole"""
    t, a, ring = shock_r(f)
    if t <= 0: return sea
    iz = 1. / M[0, 0]
    y0 = int(max(790, HZ * M[1, 1] + M[1, 2] + 6))
    yy, xx = np.mgrid[y0:H, 0:W].astype(f32)
    X = (xx - M[0, 2]) * iz; Y = (yy - M[1, 2]) * iz
    dep = -SEA_Z * 1400. / np.maximum(Y - HZ, 4.)
    wx = (X - 1024) * dep / 1400.
    dx = wx - SEA_C[0]; dy = dep - SEA_C[1]
    rho = np.sqrt(dx * dx + dy * dy) + 1e-4
    th = np.arctan2(dy, dx)
    aa = a * (1 + .05 * np.sin(th * 5 + 1) + .035 * np.sin(th * 11 + 2.2))
    heal = 1 - float(comp.ease((f - 128) / 60.))
    rs = rho * (1 - .40 * heal * np.exp(-(rho / np.maximum(aa * 1.45, .05)) ** 2))
    # a travelling swell outside
    rs = rs - .95 * np.exp(-((rho - ring) / 2.4) ** 2) * math.exp(-t / 24.)
    rs = np.maximum(rs, .02)
    k = rs / rho
    sx = SEA_C[0] + dx * k; sy = np.maximum(SEA_C[1] + dy * k, 3.)
    u = (1024 + sx / sy * 1400.) * M[0, 0] + M[0, 2]; v = (HZ - SEA_Z * 1400. / sy) * M[1, 1] + M[1, 2]
    far = np.clip((dep - 60) / 30., 0, 1)
    u = u * (1 - far) + xx * far; v = v * (1 - far) + yy * far
    out = sea.copy()
    roi = cv2.remap(sea, u.astype(f32), v.astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out[y0:] = roi
    kz = math.exp(-t / 22.) * min(1., t / 2.)
    nz = 1 + .035 * np.sin(th * 6 + 1.1) + .022 * np.sin(th * 13 + .4) + .012 * np.sin(th * 29)
    rq = rho / (max(ring, .1) * nz)
    sk = .5 + .5 * np.sin(th * 41 + np.sin(rho * .9) * 2 + 2 * np.sin(th * 5))
    trough = np.clip((rq - .60 - .07 * sk) / .05, 0, 1) * np.clip((.90 - rq) / .03, 0, 1) * kz
    out[y0:, :, :3] += (out[y0:, :, :3] * np.array([.74, .68, .86], f32) + rgb('#8d7bb0') * .10 - out[y0:, :, :3]) * (trough * .85)[..., None]
    crest = np.clip((rq - .905) / .02, 0, 1) * np.clip((1.02 + .05 * sk - rq) / .03, 0, 1) * kz
    out[y0:, :, :3] += (rgb('#fff1d2') - out[y0:, :, :3]) * (crest * .62)[..., None]
    # the hole: a torn cumulus wall, and the next deck of cloud far down inside it
    farw = np.clip(dy / np.maximum(rho, .01), -1, 1)                          # +1 far wall (faces us)
    fs = np.clip(abs(farw) * 1.5 - .30, 0, 1)
    n = .10 * np.sin(th * 4 + 2) + .07 * np.sin(th * 9 + .6) + fs * (.05 * np.sin(th * 17 + rho * 1.1) + .035 * np.sin(th * 31 + 1.7) + .03 * np.sin(th * 53 + rho * 2.))
    q = rho / max(a, .05) + n * .42
    op = float(np.clip(t / 2.5, 0, 1)) * (1 - float(comp.ease((f - 128) / 22.)))
    st = np.sin(th * 38 + np.sin(rho * 1.7) * 1.2 + 2.5 * np.sin(th * 7)) * .5 + .5
    st2 = np.sin(th * 19 + 1.3 + np.sin(rho * 1.1)) * .5 + .5
    lw = np.roll(low[y0:], -14, 0) * np.array([.90, .87, .97], f32) if low is not None else np.full(out[y0:, :, :3].shape, .5, f32)
    inn = np.clip((.80 - q - .05 * st * fs) / .02 + .5, 0, 1) * op
    wall = np.clip((1.03 - q) / .02 + .5, 0, 1) * op
    fw = np.clip((farw + .55 + .2 * (st2 - .5)) / .5, 0, 1)
    wc = rgb('#8f82b1') * (1 - fw[..., None]) + rgb('#b59ab2') * fw[..., None]
    out[y0:, :, :3] += (wc - out[y0:, :, :3]) * wall[..., None]
    k1 = wall * np.clip((.955 - q - .05 * st * fs) / .015 + .5, 0, 1); out[y0:, :, :3] += (mix(rgb('#8378ac'), rgb('#9a88b2'), .5) - out[y0:, :, :3]) * (k1 * .85)[..., None]
    k2 = wall * np.clip((q - .84 - .10 * st2 * fs) / .015 + .5, 0, 1) * np.clip((st - .55) / .2, 0, 1) * fw; out[y0:, :, :3] += (rgb('#cfaeb6') - out[y0:, :, :3]) * (k2 * .8)[..., None]
    out[y0:, :, :3] += (lw - out[y0:, :, :3]) * inn[..., None]
    w2 = np.clip((1.05 - q) / .015 + .5, 0, 1) * np.clip((q - .985 + .035 * st * fs) / .02 + .5, 0, 1) * np.clip((farw + .5) / .3, 0, 1) * op
    out[y0:, :, :3] += (rgb('#fde6c4') - out[y0:, :, :3]) * (w2 * .95)[..., None]
    out[y0:, :, :3] *= np.where(roi[..., 3:4] < .999, roi[..., 3:4], 1)      # nothing is painted above the horizon
    return out


def shock_ring(img, f, M):
    """the ring of air made visible: a wall of torn cumulus rolling out, painted on twos"""
    d = f - (f % 2)
    t, a, ring = shock_r(d)
    if t <= 0 or t > 46: return img
    y0, hh = 800, H - 800
    fade = float(np.clip((46 - t) / 22., 0, 1))
    for rg, (rad, n, hgt, sd) in enumerate(((ring, 58, 1.0, 1), (a * 1.06, 46, .34, 2))):
        rr = np.random.default_rng(404 + sd)
        th = np.linspace(0, 2 * np.pi, n, endpoint=False) + rr.normal(0, .05, n)
        R = rad * (1 + rr.normal(0, .03, n) + .035 * np.sin(th * 5 + sd) + .02 * np.sin(th * 9 + 2))
        sz = (.75 + 1.0 * rr.random(n) ** 1.3) * (.50 + .80 * (1 - math.exp(-t / 3.0))) * (1 - .55 * float(np.clip((t - 9) / 34., 0, 1))) * hgt
        keep = rr.random(n) > .05 + .006 * t
        wx = SEA_C[0] + np.cos(th) * R; wy = SEA_C[1] + np.sin(th) * R
        half = [(wy >= SEA_C[1]), (wy < SEA_C[1])]
        for hi, hm in enumerate(half):
            prims = []
            for i in np.argsort(-wy):
                if not hm[i] or not keep[i] or wy[i] < 12.5: continue
                pp = 1400. / wy[i]
                sx, sy = (1024 + wx[i] * pp) * M[0, 0] + M[0, 2], (HZ - SEA_Z * pp) * M[1, 1] + M[1, 2]
                s0 = sz[i] * pp * M[0, 0]
                if sx < -s0 * 2 or sx > W + s0 * 2 or sy - s0 > H: continue
                lean = math.cos(th[i]) * s0 * .25
                prims.append((sx + lean, sy - y0 - s0 * .20, s0 * 1.25, s0 * .74, fade))
                if rr.random() < .6: prims.append((sx + lean * 1.8 + rr.normal(0, s0 * .4), sy - y0 - s0 * .52, s0 * .62, s0 * .48, fade))
            if not prims: continue
            dp = .12 if hi else .55
            tn = np.stack([mix(p, q, dp) for p, q in zip(G.T('#fff6e0', '#f9d8b6', '#cfa4b0', '#9f84b0', '#7f74ac'), G.T('#fff6da', '#fde0b4', '#e6b9aa', '#bf9db2', '#a68fb4'))])
            q = 1.0 if hi else .7
            l = G.mass_layer(hh, W, prims, tn, (.52, -.78, -.12), np.random.default_rng(50 + sd * 7 + hi), seed=60 + sd * 3 + hi, sizes=(11 * q, 6 * q, 3.4 * q), fade=(.34, .84))
            img[y0:] = comp.over(img[y0:], l)
    return img


def smears(B, img, f, M):
    """dry-brush air behind the fastest drawings of the stroke"""
    if not (103 <= f <= 112 or 138 <= f <= 143): return img
    k0 = float(np.interp(f, [103, 105, 108, 111, 113, 137, 139, 141, 144], [.2, .8, 1, .45, 0, 0, .7, .6, 0]))
    ts = np.linspace(f, f - 2.6, 7)
    T_ = [B.tips(pose_c(t)) for t in ts]
    r = np.random.default_rng(700 + f)
    P, wd, col, op, dry = [], [], [], [], []
    for wi in (0, 1):
        A = np.array([t[wi] for t in T_])                    # [7, nf, 3]
        for k in range(A.shape[1]):
            if k < 15 and k % 2: continue
            xy, z = CAMC.project(A[:, k])
            xy = xy @ M[:, :2].T + M[:, 2]
            ln = np.linalg.norm(xy[0] - xy[-1])
            if ln < 70: continue
            for j in range(2 if k < 15 else 4):
                o = r.normal(0, 15, 2)
                P.append(xy + o); wd.append(min(ln * .075, r.uniform(3, 11) if j else r.uniform(8, 20))); col.append(rgb(['#f9cf92', '#fff1d2', '#e2a583', '#fbe3b4'][int(r.integers(0, 4))])); op.append(k0 * r.uniform(.30, .70)); dry.append(r.uniform(.30, .70))
    if not P: return img
    cv = Canvas(H, W, img=img)
    cv.strokes(np.array(P), np.array(wd), np.array(col), op=np.array(op), dry=np.array(dry), seed=f, shape='leaf', bristle=.05)
    return cv.img


def cam_c(f):
    z = float(comp.lerp_keys(f, [(76, 1.06), (118, 1.0), (136, .985), (192, .955)]))
    cx = float(comp.lerp_keys(f, [(76, PW / 2 + 34), (118, PW / 2), (150, PW / 2 - 12), (192, PW / 2 + 34)]))
    cy = float(comp.lerp_keys(f, [(76, PH / 2 - 50), (118, PH / 2 - 36), (150, PH / 2 - 20), (192, PH / 2 - 14)]))
    return cx, cy, z


def M_c(f):
    cx, cy, z = cam_c(f)
    return np.array([[z, 0, (OX - cx) * z + W / 2], [0, z, (OY - cy) * z + H / 2]])


def exit_pal(d):
    """against the sun the sheets stop glowing and go to a violet silhouette"""
    k = float(comp.ease((d - 172) / 12.))
    if k <= 0: return None
    P_ = gb.PAL; tgt = dict(t1='#8a6f94', t2='#75648f', shade='#67598f', deep='#544a82', bounce='#75658f', half='#8d7397', lit='#e9c9a4', old='#d9ac84', old2='#8d6f8c', sh_step='#594f86', dp_step='#4a4279', lip='#8f7ba2')
    return {n: mix(rgb(P_[n]), rgb(v), k * .92) for n, v in tgt.items()}


class Shot:
    id = 's8'; n = 192
    finish = dict(hal_amt=.22, vig=.10)

    def setup(self):
        self.B = gb.BirdModel()
        names = ('c_sky', 'c_far', 'c_sea', 'c_near', 'c_low')
        if not all(os.path.exists(f'cache/s8_{n}_{VER}.npy') for n in names):
            for n, a in zip(names, plates_c()):
                t = f'cache/s8_{n}_{VER}.{os.getpid()}.tmp.npy'; np.save(t, a.astype(np.float16)); os.replace(t, f'cache/s8_{n}_{VER}.npy')
        self.P = {n: np.load(f'cache/s8_{n}_{VER}.npy', mmap_mode='r') for n in names}
        self._f = {}; self.strand = strand_layer()
        from shots.s8a import ShotA
        from shots.s8b import ShotB
        self.A_, self.B_ = ShotA(), ShotB()
        self.A_.setup(self.B); self.B_.setup()

    def plate(self, n):
        if n not in self._f: self._f[n] = np.asarray(self.P[n]).astype(f32)
        return self._f[n]

    def stand(self, name, f, par):
        cx, cy, z = cam_c(f)
        zz = 1 + (z - 1) * par; px = PW / 2 + (cx - PW / 2) * par; py = PH / 2 + (cy - PH / 2) * par
        return comp.cam2d(self.plate(name), px, py, zz, border=cv2.BORDER_REPLICATE if name == 'c_sky' else cv2.BORDER_CONSTANT)

    def frame(self, f):
        if f < 48: return self.A_.frame(f)
        if f < 76: return self.B_.frame(f)
        return self.frame_c(f)

    # ------------------------------------------------------------ 8c
    def frame_c(self, f):
        M = M_c(f)
        img = self.stand('c_sky', f, .45)
        img = comp.over(img, self.stand('c_far', f, .6))
        sea = shock_warp(self.stand('c_sea', f, 1.0), f, M, self.stand('c_low', f, 1.0)[..., :3] if f > 109 else None)
        img = comp.over(img, sea)
        img = shock_ring(img, f, M)
        img = shock_streaks(img, f, M)
        # drawings: twos while it hangs, ones through the stroke and the flight
        fast = 101 <= f <= 116 or 122 <= f <= 178
        d = f if fast else f - (f % 2)
        p = pose_c(d)
        cu = plume_curves(self.B, d)
        sc = self.B.build(p, cu, flutter=d * .06)
        img = sea_shadow(img, f, sc, M, d)
        img = draw_trail(img, f, M)
        # air: the lowest plumes and the far wing tip go a step paler
        rz = p['pos'][2] - 3.0
        hx = np.where(sc['part'] == 6, .46 * np.clip((rz - sc['V'][:, 1, 1, 2] - 1.5) / 6.5, 0, 1), 0.) * float(np.interp(d, [118, 132], [1, 0]))
        hx = hx + np.where(sc['part'] != 6, .42 * np.clip((sc['V'][:, 1, 1, 1] - p['pos'][1] - 1.6) / 6.0, 0, 1) ** 1.2, 0.) * float(np.interp(d, [120, 134], [1, 0]))
        sc['hx'] = hx
        lo = loose_pages(d)
        if lo is not None:
            hx = np.r_[sc['hx'], np.zeros(len(lo['V']))]; sc = gb.join([sc, lo]); sc['hx'] = hx
        dist = p['pos'][1]
        sun2 = np.array([SUN[0], SUN[1]]) - np.array([CAMC.project(np.array(p['pos']))[0][0], CAMC.project(np.array(p['pos']))[0][1]])
        air = float(np.clip((dist - 60) / 300., 0, 1)) * .22
        lay, ox, oy = gb.draw(sc, CAMC, M=M, haze=(dist + 2, dist + 30, .5), L=tuple(kv(K, 'L', d)), sun2d=(.72, -.60), rim=4.2, seed=d // 2, boil=.4 if not fast else .25, ss=2, thr_lit=float(np.interp(d, [132, 146, 168, 178], [.22, .37, .37, .22])), pal=exit_pal(d), jitn=float(np.interp(d, [136, 146, 170, 178], [.03, .07, .07, .03])), thr_hot=float(np.interp(d, [136, 146, 170, 178], [.95, .62, .62, .95])))
        self.B.draw_head_marks(lay, ox, oy, p, CAMC, M, seed=d)
        if d > 174:
            # into the light: the sun swells a little and the bird is the darkest, clearest thing on it
            k = float(comp.ease((d - 174) / 14.))
            sx, sy = np.array([1535 + 0., 0]) * 0, 0
            cx_, cy_, zz_ = cam_c(f); z_ = 1 + (zz_ - 1) * .45
            sx = (SUN[0] + OX - PW / 2 - (cx_ - PW / 2) * .45) * z_ + W / 2; sy = (SUN[1] + OY - PH / 2 - (cy_ - PH / 2) * .45) * z_ + H / 2
            img = comp.glow(img, sx, sy, 330, '#fff0c8', .42 * k, 2.2).astype(f32)
        if air > 0: lay[..., :3] += (rgb('#f7d3a6') * lay[..., 3:4] - lay[..., :3]) * air
        img = smears(self.B, img, f, M)
        over_roi(img, lay, ox, oy)
        img = front_strand(img, f, M, self.strand)
        img = comp.over(img, self.stand('c_near', f, 1.5))
        return img
