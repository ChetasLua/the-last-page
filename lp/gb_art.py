"""Page art for the Great Bird: faded handwriting, columns, old engravings.  Everything is drawn (pressure ribbons),
nothing spells a word.  page(kind, seed, w, h) -> uint8 (h, w, 2): ink line coverage, pale wash coverage."""
import numpy as np, cairo, math, os, cv2
from .ink import spl, ribbon

# ------------------------------------------------------------ vector helpers (unit page 100 x 134)


class Pen:
    def __init__(self, w, h, UW=100., UH=134.):
        self.w, self.h = w, h
        self.s = cairo.ImageSurface(cairo.FORMAT_A8, w, h)
        self.g = cairo.ImageSurface(cairo.FORMAT_A8, w, h)
        self.c = cairo.Context(self.s); self.k = cairo.Context(self.g)
        for c in (self.c, self.k):
            c.scale(w / UW, h / UH); c.set_antialias(cairo.ANTIALIAS_BEST)
        self.q = 0
        self.wm = 1.0; self.hm = 1.0

    def poly(self, P, a=1.0, wash=False):
        c = self.k if wash else self.c
        P = np.asarray(P, float)
        c.new_path(); c.move_to(*P[0])
        for p in P[1:]: c.line_to(p[0], p[1])
        c.close_path(); c.set_source_rgba(0, 0, 0, a); c.fill()

    def erase(self, P):
        for c in (self.c, self.k):
            c.save(); c.set_operator(cairo.OPERATOR_CLEAR)
            c.new_path(); c.move_to(*P[0])
            for p in P[1:]: c.line_to(p[0], p[1])
            c.close_path(); c.fill(); c.restore()

    def line(self, P, w=.7, a=1.0, taper=(.2, .25), tremor=.16, press=.3):
        self.q += 1
        w = w * self.wm
        P = np.asarray(P, float)
        # ribbon samples at 1.6 units; feed it finer geometry by scaling
        R = ribbon(P * 6, w * 6, taper, press, self.q * 17 + 5, tremor * 4)
        if R is None: return
        self.poly(R / 6, a)

    def out(self):
        o = np.zeros((self.h, self.w, 2), np.uint8)
        for i, s in enumerate((self.s, self.g)):
            s.flush()
            o[..., i] = np.ndarray((self.h, s.get_stride()), np.uint8, s.get_data())[:, :self.w]
        return o


# ------------------------------------------------------------ handwriting
_L = [  # letter skeletons, x 0..1, y 0 base .. 1 x-height (up), >1 ascender, <0 descender
    [(0, 0), (.25, .9), (.45, .95), (.5, .1), (.75, .1), (1, .5)],
    [(0, .1), (.3, 1), (.38, .2), (.62, 1.), (.7, .1), (1, .3)],
    [(0, .4), (.5, 1), (.2, .9), (.15, .2), (.5, 0), (1, .4)],
    [(0, .2), (.45, 2.2), (.3, 2.1), (.35, .2), (.6, 0), (1, .4)],
    [(0, .3), (.4, 1), (.1, .5), (.4, 0), (.6, .9), (.5, -1.1), (.2, -1), (.9, .4)],
    [(0, .5), (.3, .9), (.6, .5), (.3, .4), (.2, .1), (.6, 0), (1, .4)],
    [(0, .2), (.3, .9), (.35, 0), (.7, .1), (1, .6)],
    [(0, .3), (.35, 1.9), (.35, .1), (.6, 0), (1, .4)],
    [(0, .6), (.35, 1), (.6, .5), (.3, 0), (.1, .5), (.4, 1), (1, .8)],
    [(0, .1), (.2, .9), (.28, .1), (.45, .9), (.55, .1), (.72, .9), (.8, .05), (1, .3)],
]


def writing(p, x0, x1, y0, y1, r, lead=5.2, xh=1.55, w=.42, a=.9, ragged=.12, slant=.28):
    y = y0; ink = 1.; nl = max(1, int((y1 - y0) / lead)); li = 0
    fx, fy = r.uniform(x0, x1), r.uniform(y0, y1)                 # a patch where age has rubbed the page pale
    while y < y1:
        u = li / nl
        xe = x1 - (x1 - x0) * (r.uniform(0, ragged) if r.random() < .78 else r.uniform(.3, .75))
        if li > nl - 3: xe = x0 + (xe - x0) * r.uniform(.35, .8)
        x = x0 + (r.uniform(2, 6) if r.random() < .14 else r.uniform(-.5, 1.0))
        base = y + r.normal(0, .22); tilt = r.normal(-.004, .010); bow = r.normal(0, .5); k0 = r.uniform(.88, 1.12)
        x00 = x
        while x < xe - 3:
            n = int(r.integers(2, 9)); cw = xh * k0 * r.uniform(.8, 1.15)
            v_ = (x - x00) / max(x1 - x0, 1)
            pts = []
            for k in range(n):
                g = _L[int(r.integers(0, len(_L)))]
                sc = r.uniform(.8, 1.2) * (1 - .25 * k / n) * k0
                for (gu, gv) in g:
                    pts.append((x + (k + gu) * cw + gv * xh * slant * sc + r.normal(0, .06), base + tilt * (x - x0) + bow * math.sin(v_ * 3.1) - gv * xh * sc * .82 + r.normal(0, .06)))
            if x + n * cw > xe + 2: break
            rub = math.exp(-(((x - fx) / 24.) ** 2 + ((y - fy) / 18.) ** 2))
            al = a * ink * (1 - .28 * v_) * (1 - .80 * rub) * r.uniform(.85, 1.05)
            if al > .08: p.line(spl(pts, n=4), w * (.62 + .5 * ink) * r.uniform(.8, 1.15), al, (.06, .45), .05, .8)
            if r.random() < .05: p.line([(x + cw * .2, base + .9), (x + n * cw * .9, base + 1.3)], w * .7, al * .7, (.05, .6))
            x += n * cw + xh * r.uniform(.9, 1.7)
            ink -= r.uniform(.04, .12)
            if ink < .34: ink = r.uniform(.92, 1.05)
        y += lead * r.uniform(.92, 1.08); li += 1


def printed(p, x0, x1, y0, y1, r, lead=3.2, w=.95, a=.55):
    """printed text as short broken dashes"""
    y = y0
    while y < y1:
        xe = x1 if r.random() < .84 else x0 + (x1 - x0) * r.uniform(.3, .8)
        x = x0 + (3 if r.random() < .14 else 0)
        while x < xe - 1:
            n = r.uniform(1.6, 7.)
            e = min(x + n, xe)
            p.line([(x, y + r.normal(0, .06)), ((x + e) / 2, y + r.normal(0, .12)), (e, y + r.normal(0, .06))], w * r.uniform(.8, 1.1), a * r.uniform(.7, 1.1), (.03, .03), .12, .35)
            x = e + r.uniform(.9, 1.6)
        y += lead


def hatch(p, poly, ang, gap, r, w=.35, a=.8, cov=1.0, curve=0.):
    """hatch lines clipped to a polygon"""
    P = np.asarray(poly, float); c = P.mean(0)
    R = np.hypot(*(P - c).T).max() * 1.1
    gap = gap * p.hm; w = w / max(p.wm, 1) * (1 + .6 * (p.wm - 1)) if p.wm > 1 else w
    d = np.array([math.cos(ang), math.sin(ang)]); n = np.array([-d[1], d[0]])
    k = p.c; k.save(); k.new_path(); k.move_to(*P[0])
    for q in P[1:]: k.line_to(q[0], q[1])
    k.close_path(); k.clip()
    t = -R
    while t < R:
        if r.random() < cov:
            a0, a1 = -R * r.uniform(.7, 1.), R * r.uniform(.7, 1.)
            m = c + n * t
            p.line(spl([m + d * a0, m + n * curve * R * .2, m + d * a1]), w * r.uniform(.7, 1.2), a, (.1, .3))
        t += gap * r.uniform(.8, 1.2)
    k.restore()


def blob(cx, cy, rx, ry, r, n=9, j=.07, rot=0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False) + rot
    q = 1 + r.normal(0, j, n)
    return spl(np.c_[cx + np.cos(t) * rx * q, cy + np.sin(t) * ry * q], closed=True)


# ------------------------------------------------------------ engravings, drawn in box (x, y, s) of size s
def E_moon(p, x, y, s, r):
    f = lambda u, v: (x + u * s, y + v * s)
    t = np.linspace(.7, 2 * np.pi - .7, 20)
    outer = np.c_[np.cos(t) * .34, np.sin(t) * .34]
    t2 = np.linspace(2 * np.pi - .95, .95, 16)
    inner = np.c_[np.cos(t2) * .27 + .13, np.sin(t2) * .27]
    C = np.array([f(.5 + a, .42 + b) for a, b in np.vstack([outer, inner])])
    p.erase(C); p.poly(C, .12, wash=True)
    p.line(np.vstack([C, C[:1]]), .6)
    hatch(p, C, .5, .03 * s, r, .3, .7, .5)
    for k in range(9):
        u, v = r.uniform(.05, .95), r.uniform(.05, .72)
        if (u - .5) ** 2 + (v - .42) ** 2 < .15: continue
        q = .022 + r.uniform(0, .02)
        for a in (0, 1.57, .78, 2.36): p.line([f(u - math.cos(a) * q, v - math.sin(a) * q), f(u + math.cos(a) * q, v + math.sin(a) * q)], .3)
    for k in range(5):
        v = .78 + .045 * k; a, b = r.uniform(.0, .2), r.uniform(.75, 1.)
        p.line(spl([f(a + (b - a) * t, v + .02 * math.sin(t * 14 + k)) for t in np.linspace(0, 1, 9)]), .5)


def E_ship(p, x, y, s, r):
    f = lambda u, v: (x + u * s, y + v * s)
    hull = spl([f(.16, .6), f(.3, .63), (f(.6, .63)), f(.86, .55), f(.8, .7), f(.7, .77), f(.35, .78), f(.22, .71)], closed=True, corners=(0, 3))
    p.erase(hull); p.poly(hull, .5, True); p.line(np.vstack([hull, hull[:1]]), .6); hatch(p, hull, .05, .025 * s, r, .32, .8)
    for (u, h) in ((.36, .1), (.55, .02), (.72, .16)):
        p.line([f(u, .62), f(u + .01, h)], .55, 1, (0, .2))
        for (a, b, w) in ((h + .03, h + .2, .10), (h + .22, .56, .13)):
            sail = spl([f(u - w, a), f(u, a + .02), f(u + w, a), f(u + w * 1.25, (a + b) / 2), f(u + w * 1.05, b), f(u, b - .03), f(u - w * .95, b), f(u - w * .75, (a + b) / 2)], closed=True, corners=(0, 2, 4, 6))
            p.erase(sail); p.poly(sail, .18, True); p.line(np.vstack([sail, sail[:1]]), .5)
            hatch(p, np.array([f(u + w * .4, a), f(u + w * 1.2, a), f(u + w * 1.3, b), f(u + w * .5, b)]), 1.5, .022 * s, r, .25, .7)
        p.line(spl([f(u + .01, h), f(u + .06, h - .015), f(u + .1, h + .015)]), .4)
    for k in range(7):
        v = .77 + .034 * k; a, b = r.uniform(-.02, .2), r.uniform(.8, 1.02)
        p.line(spl([f(a + (b - a) * t, v + .014 * math.sin(t * 20 + k * 2)) for t in np.linspace(0, 1, 15)]), .45, .9)


def E_tree(p, x, y, s, r):
    f = lambda u, v: (x + u * s, y + v * s)
    tr = spl([f(.4, .95), f(.46, .8), f(.46, .6), f(.4, .45), f(.5, .5), f(.58, .42), f(.55, .6), f(.56, .8), f(.64, .95)], closed=True, corners=(0, 3, 5, 8))
    p.erase(tr); p.poly(tr, .5, True); p.line(np.vstack([tr, tr[:1]]), .55); hatch(p, tr, 1.5, .02 * s, r, .3, .8)
    for (u, v, q) in ((.28, .42, .16), (.72, .4, .17), (.38, .24, .18), (.62, .22, .18), (.5, .36, .17), (.18, .56, .1), (.83, .55, .1), (.5, .12, .12)):
        t = np.linspace(0, 2 * np.pi, 40)
        sc = 1 + .12 * np.abs(np.sin(t * 5.5 + r.uniform(0, 6)))
        C = np.array([f(u + math.cos(a) * q * b, v + math.sin(a) * q * .8 * b) for a, b in zip(t, sc)])
        p.erase(C); p.poly(C, .28, True); p.line(C, .5)
        hatch(p, np.array([f(u - q, v + .02), f(u + q, v - .02), f(u + q * .8, v + q * .8), f(u - q * .8, v + q * .8)]), .25, .022 * s, r, .28, .75, .8, .4)
    p.line(spl([f(.02, .95), f(.3, .93), f(.6, .96), f(.98, .93)]), .55)
    for k in range(14):
        u = r.uniform(.05, .95); p.line([f(u, .955), f(u + r.normal(0, .012), .925 - r.uniform(0, .02))], .3)


def E_mount(p, x, y, s, r):
    f = lambda u, v: (x + u * s, y + v * s)
    S = blob(*f(.74, .2), .1 * s, .1 * s, r, 12, .02); p.erase(S); p.line(np.vstack([S, S[:1]]), .45)
    for a in np.linspace(0, 2 * np.pi, 15)[:-1]: p.line([f(.74 + math.cos(a) * .13, .2 + math.sin(a) * .13), f(.74 + math.cos(a) * .19, .2 + math.sin(a) * .19)], .35)
    for (P, d) in (([(0, .7), (.18, .42), (.28, .5), (.45, .2), (.56, .36), (.62, .33), (.85, .62), (1, .7)], 0), ([(0, .86), (.2, .64), (.36, .72), (.62, .5), (.8, .7), (1, .6), (1, .86)], 1)):
        q = np.array([f(*k) for k in P])
        c = np.vstack([q, [f(1, .92), f(0, .92)]]); p.erase(c); p.line(q, .6, 1, (.02, .02))
        for i in range(1, len(P) - 1):
            if P[i][1] < P[i - 1][1] and P[i][1] < P[i + 1][1]:
                u, v = P[i]; e = P[i + 1]
                sh = np.array([f(u, v), f(*e), f((u + e[0]) / 2 - .02, e[1] + .12), f(u + .01, v + .2)])
                p.poly(sh, .4, True); hatch(p, sh, 1.1, .02 * s, r, .33, .85)
    for k in range(4):
        v = .9 + .03 * k; p.line(spl([f(r.uniform(0, .2), v), f(.5, v + .01), f(r.uniform(.7, 1), v)]), .4)


def E_fox(p, x, y, s, r):
    f = lambda u, v: (x + u * s, y + v * s)
    tail = spl([f(.55, .86), f(.74, .88), f(.92, .74), (f(.94, .52)), f(.85, .36), f(.82, .56), f(.72, .68), f(.58, .72)], closed=True, corners=(4,))
    body = spl([f(.3, .92), f(.3, .7), f(.32, .5), f(.4, .42), f(.5, .46), f(.6, .62), f(.63, .8), f(.6, .92)], closed=True, corners=(0, 7))
    head = spl([f(.1, .42), f(.2, .36), f(.24, .28), f(.25, .12), f(.34, .22), f(.42, .24), f(.52, .12), f(.52, .32), f(.46, .44), f(.34, .5), f(.22, .47)], closed=True, corners=(0, 3, 6))
    for P in (tail, body, head):
        p.erase(P); p.poly(P, .3, True); p.line(np.vstack([P, P[:1]]), .6)
    hatch(p, tail, 1.9, .026 * s, r, .3, .75, .8, .5); hatch(p, np.array([f(.45, .5), f(.64, .6), f(.64, .92), f(.5, .92)]), 1.5, .024 * s, r, .3, .75)
    p.line(spl([f(.28, .33), f(.31, .345), f(.335, .33)]), .7); p.line([f(.1, .42), f(.14, .425)], 1.1)
    p.line([f(.36, .92), f(.37, .7)], .4); p.line([f(.45, .92), f(.455, .74)], .4)
    for k in range(16):
        u = r.uniform(.05, .95); p.line([f(u, .95), f(u + r.normal(0, .02), .9 - r.uniform(0, .03))], .3)
    p.line(spl([f(0, .95), f(.5, .94), f(1, .955)]), .5)


def E_stars(p, x, y, s, r):
    f = lambda u, v: (x + u * s, y + v * s)
    p.erase(blob(*f(.5, .5), .48 * s, .48 * s, r, 16, .006))
    for q in (.48, .44, .3, .16):
        C = blob(*f(.5, .5), q * s, q * s, r, 16, .006); p.line(np.vstack([C, C[:1]]), .4 if q != .48 else .6)
    for a in np.linspace(0, np.pi, 7)[:-1]:
        d = np.array([math.cos(a), math.sin(a)])
        p.line([f(.5 + d[0] * .44, .5 + d[1] * .44), f(.5 - d[0] * .44, .5 - d[1] * .44)], .25, .7)
    pts = []
    for k in range(11):
        a, q = r.uniform(0, 6.28), .4 * math.sqrt(r.uniform(.02, 1)); pts.append((.5 + math.cos(a) * q, .5 + math.sin(a) * q))
    pts.sort()
    p.line([f(*q) for q in pts[:6]], .3, .9, (.02, .02)); p.line([f(*q) for q in pts[6:]], .3, .9, (.02, .02))
    for (u, v) in pts:
        q = r.uniform(.016, .034)
        S = [f(u + math.cos(a) * (q if i % 2 == 0 else q * .4), v + math.sin(a) * (q if i % 2 == 0 else q * .4)) for i, a in enumerate(np.linspace(0, 2 * np.pi, 11)[:-1])]
        p.poly(S, .95)


def E_castle(p, x, y, s, r):
    f = lambda u, v: (x + u * s, y + v * s)
    hill = spl([f(0, .95), f(.15, .8), f(.35, .7), f(.6, .68), f(.85, .8), f(1, .95)]); p.line(hill, .6)
    for (u, w, t, roof) in ((.24, .07, .4, 1), (.4, .1, .3, 0), (.56, .075, .12, 1), (.7, .09, .36, 0), (.82, .05, .48, 1)):
        B = np.array([f(u - w, .76), f(u - w, t), f(u + w, t), f(u + w, .76)]); p.erase(B); p.poly(B, .22, True); p.line(B, .55, 1, (.02, .02))
        hatch(p, np.array([f(u + w * .2, t), f(u + w, t), f(u + w, .76), f(u + w * .3, .76)]), 1.5, .02 * s, r, .28, .8)
        if roof:
            R = np.array([f(u - w * 1.3, t), f(u, t - .2), f(u + w * 1.3, t)]); p.erase(R); p.poly(R, .5, True); p.line(np.vstack([R, R[:1]]), .5); hatch(p, R, 1.2, .02 * s, r, .3, .9)
            p.line(spl([f(u, t - .2), f(u + .05, t - .23), f(u + .09, t - .2)]), .4)
        else:
            c = []
            for k in range(6):
                a = u - w * 1.1 + w * 2.2 * k / 6; c += [f(a, t), f(a, t - .035), f(a + w * .2, t - .035), f(a + w * .2, t)]
            p.line(c, .5, 1, (0, 0))
        for k in range(2):
            v = t + .1 + .13 * k
            if v < .7: p.line([f(u - w * .2, v), f(u - w * .2, v + .05)], .8)
    hatch(p, np.array([f(.1, .86), f(.4, .74), f(.9, .84), f(.95, .95), f(.1, .95)]), .4, .03 * s, r, .3, .6, .7)


def E_whale(p, x, y, s, r):
    f = lambda u, v: (x + u * s, y + v * s)
    W = spl([f(.04, .55), f(.1, .4), f(.3, .32), f(.55, .36), f(.75, .5), f(.86, .52), f(.97, .38), f(.96, .56), f(.9, .64), f(.97, .76), f(.84, .68), f(.6, .68), f(.3, .72), f(.1, .68)], closed=True, corners=(6, 9))
    p.erase(W); p.poly(W, .3, True); p.line(np.vstack([W, W[:1]]), .65)
    for k in range(9):
        u = .1 + .045 * k; p.line(spl([f(u - .02, .56 + .004 * k), f(u + .12, .63), f(u + .3, .67)]), .3, .8)
    hatch(p, np.array([f(.2, .36), f(.55, .36), f(.8, .5), f(.5, .44)]), .3, .022 * s, r, .3, .8)
    p.line(spl([f(.04, .56), f(.14, .575), f(.24, .555)]), .5); p.poly(blob(*f(.21, .5), .012 * s, .012 * s, r, 7), 1)
    p.line(spl([f(.4, .66), f(.5, .8), f(.6, .82), f(.54, .7)]), .5)
    for q in (-1, 0, 1): p.line(spl([f(.2, .33), f(.2 + .03 * q, .2), f(.2 + .12 * q, .12), f(.2 + .16 * q, .18)]), .45)
    for k in range(6):
        v = .82 + .03 * k; p.line(spl([f(t, v + .014 * math.sin(t * 22 + k * 2.1)) for t in np.linspace(r.uniform(0, .1), r.uniform(.85, 1), 15)]), .42, .9)


def E_sun(p, x, y, s, r):
    f = lambda u, v: (x + u * s, y + v * s)
    C = blob(*f(.5, .5), .2 * s, .2 * s, r, 14, .01); p.line(np.vstack([C, C[:1]]), .6)
    C2 = blob(*f(.5, .5), .165 * s, .165 * s, r, 14, .01); p.line(np.vstack([C2, C2[:1]]), .3)
    for k in range(16):
        a = k * 2 * np.pi / 16; b = .06
        if k % 2 == 0:
            R = spl([f(.5 + math.cos(a - b * 2) * .22, .5 + math.sin(a - b * 2) * .22), f(.5 + math.cos(a - .08) * .34, .5 + math.sin(a - .08) * .34), f(.5 + math.cos(a + .05) * .46, .5 + math.sin(a + .05) * .46), f(.5 + math.cos(a + .06) * .33, .5 + math.sin(a + .06) * .33), f(.5 + math.cos(a + b * 2) * .22, .5 + math.sin(a + b * 2) * .22)])
            p.poly(R, .3, True); p.line(R, .5)
        else:
            for d in (-.05, 0, .05): p.line([f(.5 + math.cos(a + d) * .23, .5 + math.sin(a + d) * .23), f(.5 + math.cos(a + d) * .36, .5 + math.sin(a + d) * .36)], .35)
    hatch(p, np.array([f(.5, .36), f(.66, .4), f(.68, .6), f(.52, .66)]), .9, .02 * s, r, .28, .7)


def E_compass(p, x, y, s, r):
    f = lambda u, v: (x + u * s, y + v * s)
    for q in (.46, .42, .2): C = blob(*f(.5, .5), q * s, q * s, r, 18, .004); p.line(np.vstack([C, C[:1]]), .45)
    for k in range(8):
        a = k * np.pi / 4 - np.pi / 2; L = .4 if k % 2 == 0 else .26; w = .07
        T = f(.5 + math.cos(a) * L, .5 + math.sin(a) * L); c = f(.5, .5)
        A = f(.5 + math.cos(a + .6) * w, .5 + math.sin(a + .6) * w); B = f(.5 + math.cos(a - .6) * w, .5 + math.sin(a - .6) * w)
        p.poly([T, A, c], .85); p.line([T, B, c, T], .4, 1, (0, 0))
    for k in range(48):
        a = k * 2 * np.pi / 48; p.line([f(.5 + math.cos(a) * .42, .5 + math.sin(a) * .42), f(.5 + math.cos(a) * .455, .5 + math.sin(a) * .455)], .25)


def E_bird(p, x, y, s, r):
    """three swallows and a feather"""
    f = lambda u, v: (x + u * s, y + v * s)
    for (cx, cy, q, fl) in ((.46, .46, 1., 1), (.16, .20, .42, 1), (.84, .24, .36, -1), (.80, .80, .40, 1)):
        g = lambda u, v: f(cx + (u - .5) * q * .92 * fl, cy + (v - .5) * q * .92)
        body = spl([g(.02, .40), g(.12, .34), g(.24, .36), g(.40, .48), g(.62, .60), g(.80, .74), g(.98, .96), g(.82, .84), g(.72, .80), g(.88, 1.0), g(.66, .80), g(.42, .68), g(.22, .54), g(.10, .46)], closed=True, corners=(0, 6, 8, 9))
        w1 = spl([g(.30, .42), g(.34, .20), g(.52, .04), g(.84, -.06), g(.70, .10), g(.64, .16), g(.70, .24), g(.60, .30), g(.62, .40), g(.50, .52)], closed=True, corners=(0, 3, 5, 7, 9))
        w2 = spl([g(.26, .52), g(.16, .66), g(.12, .86), g(.22, 1.08), g(.30, .92), g(.36, .86), g(.40, .72), g(.40, .62)], closed=True, corners=(0, 3, 5, 7))
        for P in (w2, body, w1):
            p.erase(P); p.poly(P, .28, True); p.line(np.vstack([P, P[:1]]), .5 * q + .14)
        for k in range(5): p.line(spl([g(.40 + .03 * k, .40 + .02 * k), g(.46 + .05 * k, .26 + .03 * k), g(.56 + .05 * k, .10 + .05 * k)]), .28 * q + .1, .8)
        p.line([g(.12, .39), g(.14, .395)], .8 * q + .2)


def E_hills(p, x, y, s, r):
    f = lambda u, v: (x + u * s, y + v * s)
    for k, v in enumerate((.35, .5, .66, .82)):
        ph = r.uniform(0, 6); L = np.array([f(t, v + .05 * math.sin(t * 5 + ph) + .02 * math.sin(t * 13 + ph * 2)) for t in np.linspace(0, 1, 18)])
        c = np.vstack([L, [f(1, 1), f(0, 1)]]); p.erase(c); p.line(L, .55, 1, (.03, .03))
        for j in range(8):
            u = r.uniform(.02, .98); i = int(u * 17); X, Y = L[i]
            if r.random() < .5:
                p.line([(X, Y), (X + .01 * s, Y - .07 * s)], .5); C = blob(X + .01 * s, Y - .09 * s, .025 * s, .04 * s, r, 7, .1); p.poly(C, .6, True); p.line(np.vstack([C, C[:1]]), .35)
            else:
                p.line(spl([(X, Y + .03 * s), (X + .05 * s, Y + .05 * s), (X + .1 * s, Y + .11 * s)]), .3, .8)
    for k in range(3): p.line(spl([f(.1 + .25 * k, .14 + .03 * k), f(.16 + .25 * k, .1 + .03 * k), f(.22 + .25 * k, .14 + .03 * k)]), .4)


def E_flower(p, x, y, s, r):
    f = lambda u, v: (x + u * s, y + v * s)
    p.line(spl([f(.5, .98), f(.47, .75), f(.5, .5), f(.49, .35)]), .7, 1, (0, .1))
    for k in range(7):
        a = -np.pi / 2 + (k - 3) * .62; c = (.5, .3)
        P = spl([f(*c), f(c[0] + math.cos(a - .3) * .15, c[1] + math.sin(a - .3) * .15), f(c[0] + math.cos(a) * .27, c[1] + math.sin(a) * .27), f(c[0] + math.cos(a + .3) * .15, c[1] + math.sin(a + .3) * .15)], closed=True, corners=(0, 2))
        p.erase(P); p.poly(P, .25, True); p.line(np.vstack([P, P[:1]]), .5)
        for d in (-.12, 0, .12): p.line([f(c[0] + math.cos(a + d) * .08, c[1] + math.sin(a + d) * .08), f(c[0] + math.cos(a + d) * .2, c[1] + math.sin(a + d) * .2)], .28)
    for (sx, v) in ((1, .8), (-1, .68), (1, .58)):
        P = spl([f(.48, v), f(.48 + sx * .16, v - .12), f(.48 + sx * .34, v - .1), f(.48 + sx * .2, v + .0)], closed=True, corners=(0, 2))
        p.erase(P); p.poly(P, .4, True); p.line(np.vstack([P, P[:1]]), .5); p.line([f(.48, v), f(.48 + sx * .3, v - .09)], .3)
        hatch(p, P, .9 * sx, .022 * s, r, .25, .7)


ENG = [E_moon, E_ship, E_tree, E_mount, E_fox, E_stars, E_castle, E_whale, E_sun, E_compass, E_bird, E_hills, E_flower]


def stains(p, r):
    for k in range(int(r.integers(1, 4))):
        C = blob(r.uniform(0, 100), r.uniform(0, 134), r.uniform(6, 26), r.uniform(6, 22), r, 11, .14)
        p.poly(C, r.uniform(.10, .22), True)
    # foxed edge
    for (a, b) in (((0, 0), (100, 0)), ((100, 0), (100, 134)), ((100, 134), (0, 134)), ((0, 134), (0, 0))):
        if r.random() < .7:
            t = np.linspace(0, 1, 14)[:, None]; A = np.array(a) * (1 - t) + np.array(b) * t
            n = np.array([50, 67]) - A.mean(0); n = n / np.hypot(*n)
            B = A + n * (r.uniform(1.5, 6, 14)[:, None])
            p.poly(np.vstack([A, spl(B)[::-1]]), r.uniform(.12, .3), True)


def small(kind, seed, w=256, h=343):
    """pages that are only 15-60 px tall on screen: rows of faint script, never little pictures"""
    r = np.random.default_rng(9100 + seed * 77 + kind)
    p = Pen(w, h); p.wm = 2.3

    def scrib(x0, x1, y0, n, lead, a=.85, end=1.0):
        for j in range(n):
            y = y0 + j * lead + r.normal(0, .6); xe = x0 + (x1 - x0) * (r.uniform(.86, 1.) if r.random() < .8 else r.uniform(.35, .75)) * (1 if j < n - 1 else end)
            x = x0 + (4 if r.random() < .15 else 0); ink = 1.
            while x < xe - 5:
                L = min(r.uniform(9, 30), xe - x); t = np.linspace(0, 1, int(L / 2.2) + 4)
                p.line(np.c_[x + t * L, y + 1.25 * np.sin(t * L * 1.25 + r.uniform(0, 6)) * (1 - .5 * t) + .4 * t], .60 * r.uniform(.75, 1.15), a * ink, (.1, .35), .12, .45)
                x += L + r.uniform(3, 6); ink = max(.35, ink - .12)
    if kind == 0: scrib(12, 88, 16, 12, 9.3, .9, .6)
    elif kind == 1: scrib(12, 46, 16, 12, 9.2, .8); scrib(55, 88, 16, 12, 9.2, .8)
    elif kind == 2: scrib(18, 82, 30, 7, 11., .85, .4)
    elif kind == 3: scrib(12, 88, 60, 7, 9.6, .85, .5)
    stains(p, r)
    return p.out()


def bold(kind, seed, w=256, h=343):
    """one old engraving: a drawn picture in a ruled frame, light subject on a hatched ground"""
    r = np.random.default_rng(5200 + seed * 31 + kind)
    p = Pen(w, h); p.wm = 2.7; p.hm = 2.0
    E = BOLDSET[seed % len(BOLDSET)]
    if kind == 0:     # tall frame
        F = spl([(8, 8), (50, 7.2), (92, 8), (93, 67), (92, 126), (50, 127), (8, 126), (7.2, 67)], closed=True, corners=(0, 2, 4, 6)); bx = (12, 28, 76)
    elif kind == 1:   # arched
        F = spl([(8, 126), (7.5, 80), (8, 42), (20, 18), (50, 7), (80, 18), (92, 42), (92.5, 80), (92, 126), (50, 127)], closed=True, corners=(0, 8)); bx = (12, 34, 76)
    else:             # oval
        F = blob(50, 67, 44, 60, r, 18, .004); bx = (16, 32, 68)
    # ruled sky
    k = p.c; k.save(); k.new_path(); k.move_to(*F[0])
    for q in F[1:]: k.line_to(q[0], q[1])
    k.close_path(); k.clip()
    y = 8.
    while y < 127:
        x = 6.
        while x < 94:
            L = r.uniform(18, 60); p.line(spl([(x, y + r.normal(0, .3)), (x + L * .5, y + r.normal(0, .4)), (x + L, y + r.normal(0, .3))]), .26, .60, (.1, .3)); x += L + r.uniform(0, 5)
        y += 3.1
    k.restore()
    E(p, *bx, r)
    p.line(np.vstack([F, F[:1]]), .55, 1., (0, 0)); G = (F - [50, 67]) * [1.055, 1.035] + [50, 67]; p.line(np.vstack([G, G[:1]]), .22, .9, (0, 0))
    return p.out()


SMALL = [(0, 1), (0, 2), (1, 1), (2, 1), (0, 3), (3, 1), (0, 4), (1, 2), (0, 5), (2, 2), (0, 6), (1, 3), (0, 7), (3, 2), (0, 8), (2, 3), (1, 4), (0, 9), (0, 10), (3, 3), (1, 5), (0, 11), (2, 4), (0, 12), (0, 13), (1, 6), (3, 4), (0, 14), (2, 5), (0, 15), (1, 7), (0, 16)]


BOLDSET = [E_moon, E_ship, E_tree, E_mount, E_castle, E_whale, E_fox, E_stars]
BOLD = [(0, 1), (1, 0), (0, 2), (0, 5), (1, 4), (2, 6), (0, 3), (1, 1), (2, 0), (0, 4), (1, 2), (0, 7), (2, 5), (1, 3), (0, 6), (2, 1), (0, 0), (1, 5), (2, 4), (1, 6), (2, 2), (2, 3), (1, 7), (2, 7)]
BOLD0 = 1 << 24          # art codes from here on index BOLD


def page(kind, seed, w=256, h=343):
    r = np.random.default_rng(3100 + seed * 77 + kind)
    p = Pen(w, h)
    k = kind
    eng = lambda: ENG[int(r.integers(0, len(ENG)))] if seed < 0 else ENG[(seed * 5 + kind * 3) % len(ENG)]
    if k == 0:      # letter
        writing(p, 12, 88, 18, 122, r, 6.4, 1.9, .6)
    elif k == 1:    # two printed columns
        printed(p, 11, 47, 16, 122, r); printed(p, 54, 90, 16, 122, r)
        p.line([(50.5, 14), (50.4, 122)], .25, .5)
    elif k == 2:    # engraving over writing
        eng()(p, 20, 12, 60, r); writing(p, 12, 88, 82, 124, r, 6.2, 1.9, .6)
    elif k == 3:    # full plate
        q = spl([(9, 9), (50, 8.4), (91, 9), (91.4, 67), (91, 125), (50, 125.6), (9, 125), (8.6, 67)], closed=True, corners=(0, 2, 4, 6))
        p.line(np.vstack([q, q[:1]]), .7); eng()(p, 12, 22, 76, r)
        writing(p, 26, 74, 112, 120, r, 6, 1.7, .5)
    elif k == 4:    # big drop letter + print
        d = blob(22, 27, 9, 10, r, 8, .1); p.line(np.vstack([d, d[:1]]), .8); hatch(p, d, .8, 1.6, r, .4, .8)
        printed(p, 35, 89, 18, 37, r, 3.6); printed(p, 11, 89, 40.5, 122, r, 3.6)
    elif k == 5:    # nearly blank, a few lines
        writing(p, 24, 78, 40, 70, r, 7.4, 2.2, .65); p.line(spl([(38, 88), (50, 86), (63, 89)]), .5)
    elif k == 6:    # star chart / map plate, margin notes
        E_stars(p, 14, 10, 72, r); writing(p, 14, 88, 92, 124, r, 5.6, 1.6, .5)
    elif k == 7:    # printed + small engraving at foot
        printed(p, 12, 88, 14, 70, r, 3.4); eng()(p, 26, 76, 48, r)
    elif k == 8:    # music-like staves (drawn lines and dots)
        for j in range(6):
            y0 = 16 + j * 18.5
            for q in range(5): p.line(spl([(10, y0 + q * 1.9), (50, y0 + q * 1.9 + r.normal(0, .2)), (90, y0 + q * 1.9)]), .25, .8, (.02, .02))
            x = 16
            while x < 88:
                yy = y0 + r.uniform(0, 8); p.poly(blob(x, yy, 1.1, .8, r, 7, .05), .95); p.line([(x + 1, yy), (x + 1, yy - 5.5 * (1 if r.random() < .6 else -1))], .3); x += r.uniform(4, 9)
    elif k == 9:    # plate with two engravings side by side (bestiary)
        eng()(p, 10, 12, 44, r); ENG[(seed * 7 + 4) % len(ENG)](p, 50, 14, 42, r)
        writing(p, 12, 88, 66, 84, r, 6.2, 1.8, .55); ENG[(seed * 3 + 7) % len(ENG)](p, 26, 80, 46, r)
    stains(p, r)
    return p.out()


# art sets: (kind, seed)
WIDE = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 1), (4, 2), (5, 1), (6, 1), (7, 0), (7, 1), (8, 1), (9, 0), (9, 1), (9, 2), (0, 4), (4, 3), (3, 5), (3, 6), (2, 5), (7, 4), (1, 4), (3, 7), (2, 9)]


def atlas(w, h, lst=WIDE, key='w', levels=1):
    """uint8 [N, h, w, 2], cached.  levels>1 returns a list of mips (area filtered)"""
    os.makedirs('cache', exist_ok=True)
    fn = f'cache/s8_art_{key}_{w}x{h}_{len(lst)}_v7.npy'
    if os.path.exists(fn): A = np.load(fn)
    else:
        A = np.stack([(small if key[0] == 's' else bold if key[0] == 'b' else page)(k, s, w, h) for k, s in lst])
        tmp = fn + '.%d.tmp.npy' % os.getpid(); np.save(tmp, A); os.replace(tmp, fn)
    if levels == 1: return A
    out = [A]
    for l in range(1, levels):
        out.append(np.stack([cv2.resize(a, (max(2, w >> l), max(2, h >> l)), interpolation=cv2.INTER_AREA) for a in A]))
    return out
