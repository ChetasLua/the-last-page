"""The sound of every shot, placed in its room. Times are film seconds, taken from the shots' own keys."""
import json
import os
import numpy as np
from core import *
from inst import modes
from sfx import *
from rooms import Out


STROKES = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strokes.json')))      # tip speed and pressure of the drawn lines
OFF2 = 17.5        # scene level that reads as dBFS in the last epoch's mix


def env_gain(x, t0, pts, kind='db'):
    return x * curve(pts, x.shape[-1], t0, SR, kind)


def press(x, ceil=4.4, drive=1.0):
    """Tape-like saturation for the two great impacts: denser, not taller."""
    return ceil * np.tanh(x * drive / ceil)


def swirl(rate0, rate1, t0, t1, amp=.85):
    """pan centre orbiting faster and faster"""
    def f(t):
        u = np.clip((t - t0) / (t1 - t0), 0, 1)
        ph = 2 * np.pi * (rate0 * (t - t0) + .5 * (rate1 - rate0) * u * (t - t0))
        return amp * np.sin(ph) * np.clip(u * 3, .25, 1)
    return f


# ================================================================ s1 cloud sea at night
def s1():
    o = Out()
    g = [(0, 0), (.12, .24), (1.0, .34), (2.2, .72), (3.2, .34), (4.5, .98), (5.7, .42), (6.6, .74), (7.2, .6)]
    w = wind(0, 7.14, g, 11, whistle=.28, wf=(430, 700))
    w = env_gain(w, 0, [(0, -60), (.05, -18), (.35, -2), (1.4, 0), (6.95, 0), (7.05, -26), (7.14, -70)])
    o.put('sfx', 0, w, -1.5, room='sky', wet=-10)
    gr = grass(.4, 7.1, [(.4, 0), (1.6, .12), (2.6, .55), (3.5, .35), (4.6, .85), (5.8, .45), (7.1, .6)], 12, dens=2600)
    gr = env_gain(gr, .4, [(.4, -40), (2.0, -4), (6.95, 0), (7.05, -30), (7.1, -70)])
    o.put('sfx', .4, gr, 6.5, room='sky', wet=-16)
    # a loose stem tapping stone, once or twice
    for t, p in ((2.9, .5), (3.08, .52), (4.75, -.6), (4.86, -.58), (5.02, -.6)):
        x = modes(ns(.06), [1900, 2900, 4300, 6100], [.6, 1, .5, .3], [.012, .01, .007, .005], int(t * 100), att=.0002)
        o.put('sfx', t, x * .03, 0, p=p, room='sky', wet=-8)
    return o.pack()


# ================================================================ interior bed, s2-s4
def bed():
    o = Out()
    # big dusty room: low air, far wind on the roof, specks of wood noise
    t0, t1 = 6.94, 20.66
    n = ns(t1 - t0)
    z = np.stack([shaped(n, 21 + k, [20, 35, 60, 110, 220, 500, 1200, 4000, 9000, 20000], [-30, -8, 0, -2, -9, -18, -27, -36, -42, -60]) for k in (0, 1)])
    c = shaped(n, 24, [20, 35, 60, 110, 220, 500, 20000], [-30, -6, 0, -3, -14, -40, -80])
    z = (z + .8 * c) * (1 + .25 * np.stack([slow(n, .7, 25), slow(n, .7, 26)]))
    z = env_gain(z, t0, [(6.94, -50), (7.06, 0), (20.3, 0), (20.6, -20)])
    o.put('sfx', t0, z * .034, 0)
    w = wind(6.95, 20.6, [(6.95, .4), (9, .25), (11, .5), (13, .2), (15, .45), (17.5, .2), (20.6, .3)], 27)
    w = lp(w, 340, 2)
    o.put('sfx', 6.95, env_gain(w, 6.95, [(6.95, -40), (7.1, 0), (20.6, 0)]), -17, room='hall', wet=-6)
    r = rng(28)
    for t in (8.2, 9.63, 12.9, 14.35, 15.05, 17.1, 17.8):
        m = modes(ns(.1), wood_modes(int(t * 10), f0=220, K=10, T=.05)[0], wood_modes(int(t * 10), f0=220, K=10, T=.05)[1], wood_modes(int(t * 10), f0=220, K=10, T=.05)[2], 29, att=.0003)
        o.put('sfx', t, m / np.abs(m).max() * .010 * r.uniform(.5, 1.2), 0, p=r.uniform(-.9, .9), room='hall', wet=-2)
    # lantern flame: far, then close in s3
    fl = flame(13.5, 31)
    fl = env_gain(fl, 7.05, [(7.05, -22), (11, -14), (13.98, -13), (14.02, -2), (17.98, -2), (18.02, -12), (20.5, -14)])
    o.put('sfx', 7.05, fl, 2, room='hall', wet=-16)
    return o.pack()


# ================================================================ s2 the fox pads in
def s2():
    o = Out()
    # the drawn gait is a trot: diagonal pairs, 103.8 px a cycle over 650 px in 88 frames, easing over the last fifth
    r = rng(41)
    fr = np.linspace(0, 88, 8801)
    u = fr / 88.0
    x = 650 * np.where(u < .8, u, .8 + .2 * (1 - (1 - (u - .8) / .2) ** 1.35))
    ph = x / (88.0 * 1.18)
    k = 0
    for beat in np.arange(.5, ph[-1], .5):
        t = 7.0 + np.interp(beat, ph, fr) / 24.0
        w = (t - 7.0) / 3.67
        p = -.92 + .80 * w
        v = .40 + .20 * w + r.normal(0, .03)
        slow_ = .8 if t > 10.3 else 1.0
        # fore paw, and the diagonal hind a hair before it
        o.put('sfx', t, paw(50 + k * 2, v * slow_, 'wood'), 3.5, p=p + (.03 if k % 2 else -.01), room='hall', wet=-5.5 - 2.5 * w)
        o.put('sfx', t - r.uniform(.010, .026), paw(51 + k * 2, v * .72 * slow_, 'wood', claws=bool(k % 2)), 2.5, p=p - .05, room='hall', wet=-5.5 - 2.5 * w)
        k += 1
    o.put('sfx', 10.667, paw(90, .40, 'wood'), -2.0, p=-.12, room='hall', wet=-8)
    for t, f, d, g in ((7.36, 1850, .13, .6), (7.84, 1990, .11, .45), (8.55, 1820, .14, .75), (9.26, 1960, .11, .55), (9.74, 1830, .15, .8), (10.45, 1900, .12, .7), (11.02, 1740, .24, .8)):
        o.put('sfx', t, squeak(int(t * 100), f, d), -9 + 20 * np.log10(g), p=-.9 + .8 * np.clip((t - 7) / 3.7, 0, 1), room='hall', wet=-4)
    # fur and breath around the stop
    o.put('sfx', 10.86, fur(.30, 95, 1800), -8, p=-.1, room='hall', wet=-10)
    o.put('sfx', 11.315, lantern_down(96), -3.5, p=-.06, room='hall', wet=-5)
    o.put('sfx', 11.66, fur(.22, 98, 2100), -9, p=-.1, room='hall', wet=-10)
    o.put('sfx', 12.06, fur(.42, 97, 2300), -8, p=-.1, room='hall', wet=-10)
    return o.pack()


# ================================================================ s3 close-up
def s3():
    o = Out()
    o.put('sfx', 14.60, fur(.17, 101, 3300), 6.0, p=-.45, room='room', wet=-12)
    o.put('sfx', 15.43, fur(.15, 102, 3600), 5.5, p=.45, room='room', wet=-12)
    for q, t in enumerate((16.0, 16.165, 16.335)):
        o.put('sfx', t, sniff(103 + q, (.8, 1.0, .85)[q]), 4.0, p=.03, room='room', wet=-13)
    # breath out, small
    n = ns(.5)
    tt = np.arange(n) / SR
    br = svf(white(n, 107), 1100 - 500 * tt / .5, .8) * smoothstep(tt / .08) * np.exp(-np.maximum(tt - .1, 0) / .13) * .014
    o.put('sfx', 17.46, br, 0, p=.02, room='room', wet=-14)
    return o.pack()


# ================================================================ s4 leap, ink, flight
def s4a():
    o = Out()
    r = rng(111)
    for k, (t, v) in enumerate(((18.02, .52), (18.10, .48), (18.21, .56), (18.29, .50), (18.40, .62), (18.47, .52), (18.57, .50), (18.60, .42))):
        o.put('sfx', t, paw(112 + k, v, 'wood'), 1.0, p=-.5 + .4 * (t - 18) / .6, room='hall', wet=-7)
    # crouch, spring, air
    o.put('sfx', 18.62, fur(.24, 119, 1700), -7, p=-.08, room='hall', wet=-12)
    for k, (t, v) in enumerate(((18.905, .9), (18.925, .85))):
        o.put('sfx', t, paw(124 + k, v, 'wood'), 1.5, p=-.05, room='hall', wet=-7)
    o.put('sfx', 18.89, whoosh(.62, [(0, 420), (.18, 1400), (.62, 480)], .9, [(0, 0), (.12, 1), (.35, .55), (.62, 0)], [(0, -.05), (.62, .18)], 121) * .055, 1, room='hall', wet=-12)
    for k, (t, v) in enumerate(((19.415, 1.0), (19.435, .8), (19.475, .7), (19.50, .6))):
        o.put('sfx', t, paw(130 + k, v, 'paper'), 4.5, p=.08, room='hall', wet=-9)
    o.put('sfx', 19.42, rustle(.5, 136, .9, 'thick', [(0, 0), (.02, 1), (.15, .35), (.5, 0)]), -2, p=0, room='hall', wet=-10)
    for k, (t, v) in enumerate(((19.90, .55), (20.07, .50))):
        o.put('sfx', t, paw(140 + k, v, 'paper'), 3.5, p=.12 + .04 * k, room='hall', wet=-9)
        o.put('sfx', t + .01, rustle(.22, 143 + k, .35, 'thick', [(0, 0), (.02, 1), (.22, 0)]), -3, room='hall', wet=-12)
    o.put('sfx', 20.18, fur(.22, 146, 2000), -8, p=.1, room='hall', wet=-12)
    return o.pack()


def s4b():
    o = Out()
    # the touch that stops the world: almost nothing
    n = ns(.12)
    t = np.arange(n) / SR
    x = bp(white(n, 151), 250, 1800, 1) * np.exp(-t / .010) * (1 - np.exp(-t / .003)) * .006
    o.put('sfx', 20.667, x, 0, p=.1, ep=3)
    n3 = ns(1.0)
    o.put('sfx', 20.55, np.stack([lp(white(n3, 153 + k), 900, 1) for k in (0, 1)]) * .0016 * fade(n3, .1, .35), 0, ep=3)
    bd_ = STROKES['bird']
    tb, yb = brush_path(bd_['t'], bd_['v'], bd_['r'], bd_['x'], 152, -.25, .35)
    o.put('sfx', tb, yb, 8.5, room='hall', wet=-15, ep=1)
    o.put('sfx', 22.40, brush([(0, .06, .6, .22, .23)], 153, size=.6), 3, room='hall', wet=-14, ep=1)      # the eye
    # shiver: two nervous redraws, a wing lifts
    for k, (t, v) in enumerate(((22.75, .35), (22.79, .25), (22.835, .42), (22.88, .28), (22.92, .55), (23.0, .65), (23.085, .40))):
        o.put('sfx', t, flap(160 + k, .03, v), -3, p=.16 + .03 * (-1) ** k, room='hall', wet=-13, ep=1)
    o.put('sfx', 22.74, rustle(.44, 170, .55, 'thin', [(0, 0), (.03, .6), (.12, .5), (.2, .9), (.36, .8), (.44, 0)]), -4, p=.1, room='hall', wet=-13, ep=1)
    o.put('sfx', 23.15, peel(.36, 171), 1.5, p=.1, room='hall', wet=-11, ep=1)
    # flight, away up and to the right: three strokes, a glide between
    for k, t in enumerate((23.54, 24.05, 24.555)):
        u = k / 2.0
        p = .14 + .62 * u
        f = lp(flap(180 + k, .42, 1.25 - .55 * u), 14000 * (1 - .6 * u), 1)
        o.put('sfx', t, f, 7, p=p, room='hall', wet=-10 + 6 * u, ep=1)
        o.put('sfx', t + .075, lp(flap(184 + k, .20, .8 - .35 * u), 12000 * (1 - .6 * u), 1), 5, p=p + .04, room='hall', wet=-9 + 6 * u, ep=1)
        o.put('sfx', t + .26, lp(flap(188 + k, .25, .7 - .3 * u, up=True), 9000 * (1 - .6 * u), 1), 5, p=p + .1, room='hall', wet=-8 + 6 * u, ep=1)
        o.put('sfx', t - .02, rustle(.42, 192 + k, .5 - .2 * u, 'thin', [(0, 0), (.06, 1), (.2, .4), (.42, 0)]), -5, p=p, room='hall', wet=-9 + 6 * u, ep=1)
    # the room comes back after the silence
    n = ns(4.7)
    z = np.stack([shaped(n, 191 + q, [20, 35, 60, 110, 220, 500, 1200, 4000, 20000], [-30, -8, 0, -2, -9, -18, -27, -38, -60]) for q in (0, 1)])
    o.put('sfx', 21.0, env_gain(z, 21.0, [(21.0, -60), (21.3, -34), (22.6, -12), (23.6, -2), (25.7, 0)]) * .030, 0, ep=1)
    return o.pack()


# ================================================================ s5 tremor
def s5():
    o = Out()
    e = 1
    # 5a one book, on its drawn jolts
    for k, (f_, v) in enumerate(((10, .6), (12, .4), (14, .25), (19, .7), (21, .55), (23, .85), (25, .5), (27, .3), (29, .9), (30, .7), (31, 1.0), (32, .8), (33, 1.0))):
        r = rng(200 + k)
        n = ns(.16)
        h = modes(n, np.array([128, 236, 372, 610, 955, 1430]) * r.uniform(.96, 1.04), np.array([.8, 1, .8, .5, .35, .2]) * db(r.normal(0, 2, 6)), [.055, .05, .04, .026, .018, .012], 200 + k, att=.0006, phase=0)
        h += bp(white(n, 220 + k), 700, 4800, 1) * np.exp(-np.arange(n) / SR / .0032) * .3
        o.put('sfx', 25.0 + f_ / 24.0, h * fade(n, .0002, .03) * v * .12, 10, p=.05, room='hall', wet=-8, ep=e)
    o.put('sfx', 25.36, rustle(1.05, 202, .30, 'thick', [(0, 0), (.1, .6), (.2, .1), (.42, .7), (.65, .2), (.85, .9), (1.05, 0)]), -3, room='hall', wet=-10, ep=e)
    # 5b the tremble runs down the row toward us, then stays in the walls
    for k in range(14):
        t_ = 26.42 + k * .078
        o.put('sfx', t_, rattle(t_, 27.78, [(t_, 24), (27.78, 18)], [(t_, .9), (t_ + .25, .45), (27.7, .5), (27.78, .1)], 260 + k, (.62 - .098 * k, .62 - .098 * k), 1, f=(300 - 4 * k, 1900)), 3.5 + .30 * k, room='hall', wet=-9, ep=e)
    o.put('sfx', 27.75, rattle(27.75, 31.34, [(27.75, 9), (28.8, 8), (29.8, 12), (30.7, 20)], [(27.75, .25), (29.0, .25), (30.0, .6), (30.6, 1.1), (30.72, 1.0), (30.76, .25), (31.2, .35), (31.34, .1)], 203, (-.85, .85), 9), 2.0, room='hall', wet=-9, ep=e)
    # farther shelves join
    o.put('sfx', 27.9, lp(rattle(27.9, 31.34, [(27.9, 4), (30.7, 15)], [(27.9, .05), (29.5, .35), (30.7, 1.0), (30.75, .2), (31.2, .3)], 204, (-1, 1), 12, f=(180, 900)), 2400, 1), -1, room='tower', wet=-5, ep=e)
    # dust
    du = crackle(4.4, [(0, 20), (.5, 900), (1.5, 500), (3.0, 900), (4.4, 1400)], 205, 'dust', tail=2.2, width=1.6)
    n = du.shape[1]
    hs = np.stack([hp(white(n, 206 + k), 4500, 2) * np.clip(.5 + .6 * slow(n, 3, 208 + k), 0, 2) for k in (0, 1)]) * .010
    o.put('sfx', 27.0, (du * .5 + hs) * curve([(27, 0), (27.4, 1), (28.4, .55), (30, .8), (31.3, 1)], n, 27.0), 3, room='hall', wet=-12, ep=e)
    # the creak: stick-slip climbing the tower, storey after storey (every third frame)
    f, a, tt = wood_modes(211, f0=52, K=40, hi=2600, T=.30)
    c1 = stickslip(1.5, [(0, 13), (.3, 22), (.75, 46), (1.2, 88), (1.5, 52)], f, a, tt, 212, jitter=.09, gain=[(0, .2), (.25, 1), (1.1, 1), (1.5, .15)])
    n = len(c1)
    c1 *= np.clip(.62 + .75 * slow(n, 8, 2121), .08, 1.4)
    f, a, tt = wood_modes(213, f0=150, K=36, hi=4200, T=.11)
    c2 = stickslip(1.15, [(0, 48), (.35, 84), (.8, 176), (1.1, 250), (1.15, 220)], f, a, tt, 214, jitter=.05, gain=[(0, 0), (.25, .6), (.95, 1), (1.15, 0)])
    n = len(c2)
    c2 *= np.clip(.5 + .9 * slow(n, 10, 2141), .0, 1.4)
    o.put('sfx', 27.72, c1, -8.5, p=-.1, room='tower', wet=-6, ep=e)
    o.put('sfx', 27.95, c2, -15, p=.2, room='tower', wet=-4, ep=e)
    for k in range(6):
        f, a, tt = wood_modes(270 + k, f0=64 + 13 * k, K=26, hi=2200 + 400 * k, T=.22)
        kn = stickslip(.22, [(0, 70 + 26 * k), (.22, 30 + 14 * k)], f, a, tt, 280 + k, jitter=.15, gain=[(0, 1), (.08, .6), (.22, 0)])
        o.put('sfx', 27.79 + .125 * k, kn, -11.5 - .5 * k, p=(-.5, .45, -.3, .55, -.15, .3)[k], room='tower', wet=-7 + .9 * k, ep=e)
    n = ns(1.9)
    t = np.arange(n) / SR
    gr = np.zeros(n)
    f0 = 33 + 24 * smoothstep(t / 1.4) + 1.5 * slow(n, 6, 215)
    ph = np.cumsum(f0) / SR
    for k in range(1, 16):
        gr += np.sin(2 * np.pi * k * ph + k) / k ** .9 * (1 + .5 * slow(n, 7, 216 + k))
    gr = lp(gr, 420, 1) * np.sin(np.pi * np.clip(t / 1.9, 0, 1)) ** 1.2 * (1 + .4 * slow(n, 16, 240))
    o.put('sfx', 27.62, gr * .16, -8, room='tower', wet=-5, ep=e)
    f, a, tt = wood_modes(217, f0=110, K=30, hi=3600, T=.2)
    for q, (t_, d, g, p) in enumerate(((29.05, .5, -21, -.6), (29.7, .6, -19, .5), (30.2, .5, -16, -.2))):
        c = stickslip(d, [(0, 14 + 4 * q), (d, 46 + 12 * q)], f, a, tt, 218 + q, gain=[(0, 0), (d * .4, 1), (d, .2)])
        o.put('sfx', t_, c, g, p=p, room='tower', wet=-3, ep=e)
    # low tower rumble growing
    n = ns(4.0)
    z = np.stack([shaped(n, 221 + k, [16, 26, 40, 70, 120, 260, 600, 20000], [-30, -5, 0, -3, -12, -26, -44, -90]) for k in (0, 1)])
    o.put('sfx', 27.4, env_gain(z, 27.4, [(27.4, -40), (28.6, -14), (30, -6), (30.70, 0), (30.76, -10), (31.3, -6), (31.4, -10)]) * .16, 0, ep=e)
    # flame leans
    fl = flame(1.9, 231, [(0, 0), (.45, .1), (.7, .35), (.95, .7), (1.25, 1), (1.5, .4), (1.9, 0)])
    o.put('sfx', 28.5, fl, 13, p=.05, room='hall', wet=-14, ep=e)
    o.put('sfx', 28.85, whoosh(1.1, [(0, 160), (.45, 420), (1.1, 200)], .8, [(0, 0), (.4, 1), (1.1, 0)], [(0, -.3), (1.1, .3)], 232) * .07, 0, room='hall', wet=-10, ep=e)
    o.put('sfx', 30.04, fur(.20, 233, 3000), 3, p=0, room='room', wet=-12, ep=e)
    # three hard cuts
    for q, t in enumerate((30.75, 30.96, 31.17)):
        o.put('sfx', t, jolt(240 + q, (.75, .85, 1.0)[q]), 3.0, p=(-.35, .35, 0)[q], room='room', wet=-14, ep=e)
        o.put('sfx', t, slap(250 + q, .8), -1, p=(-.4, .4, .05)[q], ep=e)
        o.put('sfx', t, subhit(.32, 92, 46, .03, .10, 1), -10, ep=e)
    # one speck of dust lands in the silence
    x = modes(ns(.03), [5200, 7400, 9800], [1, .6, .4], [.005, .004, .003], 260, att=.0001) * .0024
    o.put('sfx', 31.66, x * 3.0, 0, p=.25, room='hall', wet=-6, ep=3)
    return o.pack()


# ================================================================ s6 the burst
def s6():
    o = Out()
    r = rng(301)
    t0 = 32.08
    # covers: one hard front, then a spray
    core = Track(32.0, 34.2)
    core.add(t0, jolt(302, 1.0), float(db(10)))
    core.add(t0, subhit(1.3, 105, 37, .085, .36, 1, 1.8), float(db(1.0)))
    for k in range(9):
        core.add(t0 + k * .0013, slap(310 + k, r.uniform(.3, 1.0)), float(db(6)), r.uniform(-.8, .8))
    for k in range(170):
        d = r.exponential(.17)
        if d > .9:
            continue
        o.put('sfx', t0 + .012 + d, lp(slap(330 + k, r.uniform(.1, 1.0)), r.uniform(2200, 13000), 1), -6.5 - 10 * d - r.exponential(4.0), p=np.clip(r.normal(0, .6), -1, 1), room='tower', wet=-8)
    core.add(32.208, jolt(303, 1.0), float(db(8)), -.2)
    core.add(32.208, subhit(.9, 84, 35, .07, .28, 1, 1.6), float(db(-1)))
    n = ns(1.2)
    t = np.arange(n) / SR
    blast = np.stack([shaped(n, 480 + k, [30, 80, 200, 600, 1500, 3500, 8000, 16000], [-8, -1, 0, -2, -5, -9, -18, -30]) for k in (0, 1)])
    blast *= (1 - np.exp(-t / .0012)) * (1.0 * np.exp(-t / .055) + .22 * np.exp(-t / .30))
    core.add(t0, blast, float(db(9.0)))
    core.add(32.208, blast[:, :ns(.5)] * fade(ns(.5), .001, .3), float(db(5.0)))
    o.put('sfx', 32.0, press(core.x, 4.0, 1.2), 0, room='tower', wet=-13, ep=2)

    def door(x, tx):
        k = ns(t0 - tx)
        x = x.copy()
        x[:, :k] = 0
        x[:, k:k + 96] *= np.linspace(0, 1, 96)
        return x
    # the roar: a crowd of fluttering sheets over moving air
    sw = swirl(.55, 1.5, 35.25, 36.95, .98)

    def wd(t):
        return np.where((t > 35.2) & (t < 36.9), .30, .9)

    def pc(t):
        return np.where(t < 35.25, 0, np.where(t < 37.0, sw(t), 0))
    tx = 31.5
    rate = [(tx, 420), (32.3, 300), (33.0, 150), (34.6, 90), (35.25, 120), (36.9, 330), (37.3, 260), (39.0, 160), (39.4, 100)]
    one = [(tx, 1), (40, 1)]
    far = flock(tx, 39.2, rate, one, 500, pan=pc, width=wd, sg=1.25)
    big = flock(tx, 39.2, [(a, b * .10) for a, b in rate], one, 520, pan=pc, width=wd, speed=(.42, .75), big=1.0, sg=1.0)
    lev = [(tx, -24), (32.1, -22), (32.3, -19), (32.5, -20), (33.1, -26), (33.6, -33), (33.9, -36.5), (35.1, -36.5), (35.35, -32), (36.0, -27), (36.8, -22.5), (37.15, -24), (37.8, -30), (38.6, -34), (39.05, -36), (39.2, -42), (41, -70)]
    far = eq(eq(far, 6500, -5, .7, 'hs'), 650, 3, .6)
    # sheets rise on a spiral: pitch of the mass climbs with it
    roar = mass(far / far.std() + .62 * big / big.std(), 531, .55)
    roar = eq(hp(roar, 240, 1), 12000, -5, .7, 'hs')
    o.put('sfx', tx, door(ride(roar, tx, lev), tx), OFF2, room='tower', wet=-14, ep=2)
    air = whoosh(7.7, [(0, 520), (.6, 480), (2.1, 330), (3.7, 380), (5.3, 1200), (5.7, 900), (7.7, 600)], .5, [(0, 1), (7.4, 1), (7.7, .4)], 0.0, 541, corr=.15, air=.12)
    air = mass(air, 545, .22, (3, 8))
    o.put('sfx', tx, door(ride(air, tx, [(tx, -30), (32.1, -26), (32.25, -22), (32.7, -28), (33.6, -40), (35.1, -42), (36.0, -32), (36.85, -25), (37.3, -26), (38.5, -34), (39.2, -50)]), tx), OFF2, room='tower', wet=-14, ep=2)
    # single sheets close to us
    nf = near(32.1, 39.0, [(32.1, 24), (32.7, 15), (33.4, 8), (33.8, 3), (35.0, 3), (35.4, 8), (36.6, 17), (37.4, 11), (39, 5)], 547, gain=[(32.1, 1.6), (33, 1.2), (35, .8), (36.7, 1.5), (39, .8)])
    o.put('sfx', 32.1, nf, 7.5, room='tower', wet=-11)
    # two spiral arms circling opposite ways
    for q in range(2):
        d = 1.85
        sw2 = [(x, (1 if q else -1) * .95 * np.sin(2 * np.pi * (.6 * x + .42 * x * x) + q * 2.1)) for x in np.linspace(0, d, 60)]
        a = whoosh(d, [(0, 420), (1.0, 1000), (1.6, 2300), (d, 2700)], 1.8, [(0, 0), (.6, .5), (1.6, 1.0), (1.7, .6), (d, 0)], sw2, 550 + q, corr=.6, air=.2)
        o.put('sfx', 35.2, a * .38, 7.5, room='tower', wet=-13)
    # crackling fringe
    ck = crackle(7.0, [(0, 4200), (.3, 2400), (1.6, 900), (3.1, 800), (4.8, 2600), (5.4, 1700), (7.0, 500)], 560, 'thin', tail=1.35, width=1.7, gcap=16)
    ck = lp(ck, 9500, 1)
    o.put('sfx', t0, ride(ck, t0, [(32.08, -24), (32.5, -28), (33.6, -38), (35.2, -38), (36.8, -28), (37.4, -31), (39, -42)], .6), OFF2, room='tower', wet=-13)
    # folds: every three frames, plus smaller neighbours
    for k in range(13):
        t = 33.75 + k * .125
        y, pre = snap(570 + k, .35 + .5 * (k % 4 == 0))
        y = press(y * 1.15, 1.15)
        o.put('sfx', t - pre, y, 4.5 + (2 if k % 4 == 0 else 0) + (2.0 if k == 12 else 0), p=(-.55, .45, -.2, .65)[k % 4], room='tower', wet=-9)
        for j in range(3):
            y, pre = snap(590 + k * 5 + j, .1)
            o.put('sfx', t - pre + r.uniform(.012, .1), y, -9 - r.uniform(0, 6), p=r.uniform(-1, 1), room='tower', wet=-7)
    # copper shutters
    f, a, tt = metal_modes(601, f0=74, K=48, hi=3200, T=.9)
    g1 = stickslip(.62, [(0, 16), (.3, 34), (.55, 70), (.62, 90)], f, a, tt, 602, gain=[(0, .1), (.3, .8), (.6, 1)])
    o.put('sfx', 36.34, g1, -2.5, p=.35, room='tower', wet=-6)
    f, a, tt = metal_modes(603, f0=260, K=40, hi=6000, T=.5)
    g2 = stickslip(.5, [(0, 40), (.5, 150)], f, a, tt, 604, gain=[(0, .1), (.4, 1), (.5, 1)])
    o.put('sfx', 36.42, g2, -8, p=.4, room='tower', wet=-6)
    o.put('sfx', 36.50, lp(jolt(604, 1.0), 900, 1), -2, p=0, room='tower', wet=-5)
    o.put('sfx', 36.835, lp(jolt(606, 1.0), 2400, 1), 1, p=0, room='tower', wet=-7)
    o.put('sfx', 37.25, lp(jolt(607, 1.0), 2000, 1), -3, p=.3, room='tower', wet=-6)
    o.put('sfx', 36.90, jolt(605, 1.0), 2.5, p=.2, room='tower', wet=-8)
    n = ns(.25)
    t = np.arange(n) / SR
    ckk = hp(white(n, 606), 700, 1) * (np.exp(-t / .004) + .25 * np.exp(-t / .03)) * .6
    o.put('sfx', 36.90, ckk, -1, p=.25, room='tower', wet=-8)
    for q, (t, f0, g, p, T) in enumerate(((36.903, 96, -7, .3, 1.5), (36.94, 148, -9, -.2, 1.2), (37.03, 210, -11, .7, 1.0), (37.12, 120, -10, -.6, 1.3), (37.31, 300, -16, .1, .8), (37.46, 260, -18, -.75, .8), (37.7, 340, -22, .5, .6))):
        o.put('sfx', t, clang(610 + q, f0, T), g + 4.5, p=p, room='tower', wet=-7)
    o.put('sfx', 36.90, subhit(1.0, 84, 40, .07, .28, 1, 1.5), -6)
    for q in range(26):
        t = 37.0 + r.exponential(.35)
        if t < 38.6:
            x = modes(ns(.2), np.exp(r.uniform(np.log(700), np.log(6500), 6)), r.normal(0, 1, 6), r.uniform(.02, .12, 6), 630 + q, att=.0002)
            o.put('sfx', t, x * .05, 0 - 6 * (t - 37), p=r.uniform(-.9, .9), room='tower', wet=-5)
    return o.pack()


# ================================================================ s7 sunrise
def s7():
    o = Out()
    w = wind(36.95, 48.3, [(36.95, .5), (37.6, .75), (39.0, 1.0), (40.2, .75), (41.4, 1.05), (42.6, .7), (43.3, 1.1), (44.2, .6), (46.5, .75), (47.2, .5), (48.0, .22), (48.3, .1)], 701, whistle=.10, wf=(380, 620))
    w = env_gain(w, 36.95, [(36.95, -30), (37.3, -9), (38.95, -6), (39.0, 0), (41.6, 0), (42.6, -4), (42.9, -8), (43.15, 0), (43.7, -3), (44.2, -9), (46, -9), (47.2, -10), (48.15, -24), (48.2, -28)])
    o.put('sfx', 36.95, hp(w, 60, 1), .5, room='sky', wet=-10)
    g = grass(38.95, 47.3, [(38.95, .7), (39.3, 1.0), (40.2, .75), (41.4, 1.1), (42.6, .5), (43.0, .4), (43.5, 1.3), (44.2, .45), (47.3, .25)], 711, dens=4200)
    o.put('sfx', 38.95, env_gain(g, 38.95, [(38.95, -12), (39.05, 0), (43.95, 0), (44.1, -8), (47.3, -20)]), 11, room='sky', wet=-14)
    # grass whips
    r = rng(712)
    for k in range(16):
        t = r.uniform(39.2, 43.8)
        x = whoosh(.22, [(0, 1600), (.1, 4200), (.22, 2200)], 1.2, [(0, 0), (.08, 1), (.22, 0)], float(r.uniform(-.9, .9)), 720 + k)
        o.put('sfx', t, x * .06, -2 - r.uniform(0, 6), room='sky', wet=-12)
    # flock pouring out past the camera at the cut
    fl = flock(38.98, 43.2, [(38.98, 420), (39.5, 320), (41, 130), (43.2, 40)], [(38, 1), (44, 1)], 740, pan=[(38.9, -.2), (40.5, .1), (43, .4)], width=1.0, sg=1.1)
    fl = mass(eq(hp(fl, 260, 1), 8000, -5, .7, 'hs'), 742, .4)
    o.put('sfx', 38.98, ride(fl, 38.98, [(38.98, -24), (39.5, -27), (40.5, -33), (42, -40), (43.2, -50), (45, -80)]), OFF2, room='sky', wet=-9)
    o.put('sfx', 39.0, near(39.0, 42.0, [(39, 16), (40, 8), (42, 2)], 744, gain=[(39, 1.2), (42, .5)]), 2, room='sky', wet=-9)
    o.put('sfx', 38.85, whoosh(2.2, [(0, 300), (.2, 1300), (2.2, 320)], .55, [(0, 0), (.16, 1), (.8, .4), (2.2, 0)], 0.0, 745, corr=.1) * .34, -5, room='sky', wet=-12)
    # the ring of wind: low to high, centre to edges
    for q, s in enumerate((-1, 1)):
        a = whoosh(2.4, [(0, 70), (.35, 190), (.9, 900), (1.6, 2600), (2.4, 4200)], .6, [(0, 0), (.16, .85), (.45, 1.0), (.95, .55), (1.5, .16), (2.4, 0)],
                   [(0, 0), (.3, s * .12), (1.3, s * .92), (2.4, s)], 750 + q, corr=.75, air=.2)
        o.put('sfx', 42.92, a * .42, 6.5, room='sky', wet=-10)
    n = ns(1.5)
    t = np.arange(n) / SR
    lo = np.stack([bp(white(n, 760 + k), 34, 120, 2) for k in (0, 1)]) * (1 - np.exp(-t / .06)) * np.exp(-t / .42)
    o.put('sfx', 42.94, lo * .9, 5)
    o.put('sfx', 43.1, grass(43.1, 44.6, [(43.1, .2), (43.35, 1.4), (44.0, .9), (44.6, .2)], 765, dens=6000), 9, room='sky', wet=-14)
    return o.pack()


# ================================================================ s8 the great bird
def s8():
    o = Out()
    r = rng(801)
    sw = swirl(.3, .9, 44.0, 46.4, .95)
    fl = flock(43.6, 47.4, [(43.6, 60), (44.6, 160), (45.8, 420), (46.6, 620), (47.1, 380), (47.4, 120)], [(43, 1), (48, 1)], 810,
               pan=lambda t: np.where(t < 46.0, sw(t), np.interp(t, [46.0, 46.3, 46.75, 47.3], [-.5, -.8, .7, .95])), width=.45, speed=(1.0, 1.9), sg=1.1)
    fl = mass(eq(hp(fl, 350, 1), 9000, -3, .7, 'hs'), 812, .42, (7, 15))
    o.put('sfx', 43.6, ride(fl, 43.6, [(43.6, -60), (44.0, -42), (45.0, -35), (45.9, -28), (46.5, -24.5), (46.9, -25.5), (47.2, -33), (47.6, -44), (48.1, -80)]), OFF2, room='sky', wet=-9)
    # glitter: edges catching light
    ck = crackle(3.3, [(0, 60), (1.0, 350), (2.2, 1300), (2.9, 900), (3.3, 100)], 820, 'thin', tail=1.4, width=1.6, rr=(1.0, 1.7), gcap=14)
    o.put('sfx', 44.0, ride(ck, 44.0, [(44, -48), (45, -40), (46.4, -30), (47, -34), (47.3, -60)], .6), OFF2, room='sky', wet=-9)
    o.put('sfx', 44.0, near(44.2, 47.1, [(44.2, 3), (45.5, 8), (46.4, 18), (47.1, 6)], 822, size=(.05, .3), gain=[(44, .35), (46, .9), (46.6, 1.4), (47.1, .6)]), -1, room='sky', wet=-9)
    o.put('sfx', 44.0, whoosh(2.4, [(0, 260), (2.0, 1200), (2.4, 1500)], .6, [(0, 0), (1.2, .4), (2.1, 1), (2.4, 0)], [(0, -.3), (1.2, .3), (2.4, -.4)], 830) * .16, 0, room='sky', wet=-10)
    # close pass
    cp = whoosh(1.7, [(0, 200), (.45, 520), (.68, 1300), (.9, 640), (1.7, 220)], .55, [(0, 0), (.4, .35), (.66, 1.0), (.95, .5), (1.7, 0)], [(0, -.9), (.5, -.6), (.68, .1), (.9, .7), (1.7, .95)], 840, corr=.5)
    o.put('sfx', 45.9, cp * .55, -2.5, room='sky', wet=-13)
    for q, t in enumerate((46.18, 46.42, 46.62, 46.86)):
        o.put('sfx', t, flap(850 + q, .9, 1.2), 8, p=(-.6, -.25, .35, .75)[q], room='sky', wet=-13)
    cl = flock(45.95, 47.1, [(45.95, 20), (46.5, 140), (46.8, 120), (47.1, 20)], [(45, 1), (48, 1)], 860, pan=[(46, -.8), (46.45, -.3), (46.65, .3), (47.0, .9)], width=.5, speed=(.5, .9), big=1.0)
    o.put('sfx', 45.95, ride(hp(cl, 110, 1), 45.95, [(45.95, -40), (46.3, -28), (46.6, -22), (46.85, -26), (47.2, -40), (49, -90)], .3), OFF2, room='sky', wet=-13)
    # wings rise: paper under tension, air drawn in
    f, a, tt = wood_modes(870, f0=420, K=30, hi=6500, T=.03)
    te = stickslip(.92, [(0, 40), (.6, 90), (.92, 150)], f, a, tt, 871, jitter=.5, gain=[(0, .05), (.5, .6), (.85, 1), (.92, .4)])
    o.put('sfx', 47.2, te, -17, room='sky', wet=-9)
    ins = whoosh(1.0, [(0, 130), (.6, 420), (.92, 1300), (1.0, 1500)], .6, [(0, 0), (.4, .18), (.8, .62), (.91, 1.0), (.95, .0), (1.0, 0)], 0.0, 875, corr=.4, air=.15)
    o.put('sfx', 47.2, ins * .17, 1)
    rv = rustle(.92, 876, .5, 'thick', [(0, 0), (.6, .5), (.88, 1), (.92, 0)])
    o.put('sfx', 47.2, rv, 0, room='sky', wet=-12)
    # ------- THE WINGBEAT 48.2
    T = 48.2
    core = Track(T - .1, T + 3.6)
    core.add(T, subhit(1.7, 96, 35, .13, .46, 1, 1.35), float(db(3.0)))
    n = ns(3.4)
    t = np.arange(n) / SR
    # the sail itself: a huge slack membrane filled in one stroke
    for ch, dt_ in ((0, 0), (1, 3)):
        mem = modes(ns(1.4), np.array([52, 71, 97, 128, 166, 214, 276, 352, 450]) * (1 + .004 * dt_), np.array([.8, 1, 1, .9, .75, .6, .45, .3, .2]), np.array([.62, .58, .5, .44, .38, .32, .26, .2, .16]), 878 + dt_, att=.006)
        st = np.zeros((2, len(mem)))
        st[ch] = mem
        core.add(T, st, float(db(7.5)))
    z = noise2(n, 881, .45)
    fc = 150 + 1500 * np.exp(-t / .10) + 420 * np.exp(-t / .7)
    body = np.stack([svf(z[c], fc, .75, 'lp') for c in (0, 1)])
    e = (1 - np.exp(-t / .009)) * (1.0 * np.exp(-t / .17) + .20 * np.exp(-t / .9)) + .42 * np.exp(-((t - .17) / .085) ** 2)
    body = hp(body, 48, 1) * e * (1 + .18 * np.stack([slow(n, 7, 885), slow(n, 7, 886)]))
    core.add(T, body, float(db(19.5)))
    # the sail cracks taut
    q = ns(.4)
    tq = np.arange(q) / SR
    sn = noise2(q, 890, .3)
    sn = np.stack([lp(hp(sn[c], 500, 1), 9000, 1) for c in (0, 1)]) * (np.exp(-tq / .007) * 1.0 + np.exp(-tq / .05) * .25)
    core.add(T, sn, float(db(5.0)))
    for k in range(9):
        y, pre = snap(895 + k, .95)
        core.add(T - pre + abs(r.normal(0, .009)), lp(y, 7500, 1), float(db(1 - .6 * k)), r.uniform(-.8, .8))
    o.put('sfx', T - .1, press(core.x, 4.6, 1.25), 0, room='sky', wet=-12)
    tx = 47.7
    fl = flock(tx, T + 3.9, [(tx, 620), (T + .25, 460), (T + 1.0, 260), (T + 3.9, 120)], [(47, 1), (53, 1)], 910, pan=[(T, 0), (T + .7, -.6), (T + 1.3, -.7), (T + 2.4, .3), (T + 3.5, .95)], width=lambda t: np.where(t < T + .5, .95, .4), sg=1.1)
    fl = mass(eq(hp(fl, 180, 1), 8000, -4, .7, 'hs'), 912, .42)
    fl = ride(fl, tx, [(tx, -18), (48.25, -18), (48.4, -18.5), (48.6, -20), (49.0, -24), (49.8, -27), (51, -30), (52, -36), (53.5, -70)])
    k = ns(T - tx)
    fl[:, :k] = 0
    fl[:, k:k + 200] *= np.linspace(0, 1, 200)
    o.put('sfx', tx, fl, OFF2, room='sky', wet=-10)
    o.put('sfx', T, near(T + .02, 51.5, [(T, 26), (T + .6, 14), (51.5, 4)], 915, size=(.2, .7), gain=[(T, 1.7), (T + .7, 1.0), (51.5, .5)]), 4, room='sky', wet=-9)
    # shock ring spreading left and right
    for q, s in enumerate((-1, 1)):
        a = whoosh(2.2, [(0, 1100), (.2, 560), (1.0, 380), (2.2, 240)], .55, [(0, 0), (.03, 1), (.35, .55), (1.0, .25), (2.2, 0)], [(0, 0), (.15, s * .4), (.6, s * .95), (2.2, s)], 920 + q, corr=.6)
        o.put('sfx', T, a * .5, 1.5, room='sky', wet=-10)
    for k, (t_, g_, p_) in enumerate(((49.70, 1.0, -.35), (50.62, .45, .35))):
        o.put('sfx', t_, whoomph(960 + k, g_), 6.5, p=0, room='sky', wet=-10)
        o.put('sfx', t_ + .02, flap(965 + k, 1.0, 1.0 * g_), 11, p=p_, room='sky', wet=-9)
    # tearing across the sky, left to right
    tw = whoosh(3.6, [(0, 420), (.8, 800), (1.5, 1700), (2.0, 1500), (2.9, 800), (3.6, 340)], .5, [(0, 0), (.5, .35), (1.4, 1.0), (2.2, .8), (3.0, .35), (3.6, 0)],
                [(0, -.9), (.8, -.75), (1.6, 0), (2.5, .7), (3.6, .95)], 930, corr=.35)
    o.put('sfx', 48.75, tw * .42, 3.5, room='sky', wet=-11)
    sc = whoosh(3.2, [(0, 1500), (1.2, 4200), (2.0, 3600), (3.2, 1200)], 1.1, [(0, 0), (1.1, 1.0), (1.9, .6), (3.2, 0)], [(0, -.9), (1.3, 0), (2.4, .85), (3.2, .95)], 935, corr=.5)
    o.put('sfx', 48.95, sc * .13, 2, room='sky', wet=-9)
    for k, t in enumerate((50.1, 51.2)):
        o.put('sfx', t, flap(940 + k, 1.0, .8 - .3 * k), 5 - 1.5 * k, p=(0, .85)[k], room='sky', wet=-9)
    return o.pack()


# ================================================================ s9 the last page
def s9():
    o = Out()
    w = wind(48.2, 65.0, [(48.2, .5), (49.2, .9), (50.8, .85), (52.0, .55), (53.5, .34), (55.2, .26), (57, .36), (58.2, .28), (60.5, .22), (62.6, .30), (63.6, .42), (64.4, .25), (65, 0)], 1001)
    w = env_gain(w, 48.2, [(48.2, -8), (48.8, -3), (51.2, -2), (52.2, -5), (54, -7), (58.2, -7.5), (58.32, -13), (61, -14), (63.2, -10), (63.7, -8), (64.4, -15), (64.9, -38), (65.0, -70)])
    o.put('sfx', 48.2, hp(w, 85, 1), 2.0, room='sky', wet=-9)
    g = grass(51.8, 64.95, [(51.8, .5), (53, .3), (55.2, .22), (57, .42), (58.2, .3), (60.5, .16), (62.4, .3), (63.6, .5), (64.9, 0)], 1011, dens=1900)
    o.put('sfx', 51.8, env_gain(g, 51.8, [(51.8, -20), (52.6, 0), (58.2, 0), (58.3, -8), (63, -6), (64, -5), (64.95, -50)]), 11, room='sky', wet=-14)
    # last pages
    for q, (t, d, a, p) in enumerate(((52.05, .5, .7, -.5), (52.42, .4, .5, .55), (52.85, .45, .6, -.15), (53.22, .35, .45, .35), (53.55, .3, .4, -.05))):
        o.put('sfx', t, rustle(d, 1020 + q, a, 'thin'), 6, p=p, room='sky', wet=-10)
        for k in range(3):
            o.put('sfx', t + .05 + k * .085, flap(1030 + q * 4 + k, .08, .5 * a), 9, p=p, room='sky', wet=-9)
    # last one: slides the air, lands
    o.put('sfx', 53.62, whoosh(.34, [(0, 900), (.25, 1900), (.34, 800)], 1.0, [(0, 0), (.2, 1), (.34, 0)], [(0, -.2), (.34, .05)], 1050) * .05, 2)
    o.put('sfx', 53.90, paw(1051, .55, 'paper', claws=False), 7.0, p=.04, room='room', wet=-13)
    o.put('sfx', 53.905, rustle(.30, 1052, .95, 'thick', [(0, 0), (.015, 1), (.10, .35), (.30, 0)]), 0, p=.04, room='room', wet=-12)
    for q, t in enumerate((54.5, 54.67)):
        o.put('sfx', t, sniff(1056 + q, .7), -6, p=.02, room='room', wet=-14)
    o.put('sfx', 55.46, fur(.26, 1054, 2500), -4, p=-.02, room='room', wet=-14)
    o.put('sfx', 57.02, fur(.36, 1055, 1900), -5, p=.04, room='room', wet=-14)
    # ------- the page
    o.put('sfx', 58.38, fur(.14, 1060, 1800), -5, p=.05, room='room', wet=-14)
    o.put('sfx', 58.50, paw(1061, .62, 'paper', claws=False), 6.0, p=.05, room='room', wet=-12)
    o.put('sfx', 58.505, rustle(.18, 1062, .35, 'thick', [(0, 0), (.012, 1), (.18, 0)]), -2, p=.05, room='room', wet=-13)
    # ink blooms: a drop taken by fibre
    n = ns(.45)
    t = np.arange(n) / SR
    bl = modes(n, [310, 462, 705], [1, .6, .3], [.06, .05, .03], 1063, att=.006, phase=0) * .05
    bl += svf(white(n, 1064), 900 + 2400 * t / .45, 1.0) * smoothstep(t / .05) * np.exp(-t / .13) * .016
    o.put('sfx', 58.66, bl, 7, p=.02, room='room', wet=-13)
    ti = STROKES['title']
    tb, yb = brush_path(ti['t'], ti['v'], ti['r'], ti['x'], 1065, -.5, .5)
    o.put('sfx', tb, yb, 10.5, room='room', wet=-13)
    # hanko
    n = ns(.35)
    t = np.arange(n) / SR
    hk = modes(n, np.array([88, 137, 212, 331, 520, 845, 1380]), [.8, 1, .8, .5, .32, .18, .08], [.07, .075, .06, .05, .035, .025, .015], 1070, att=.0022, phase=0) * .34
    hk += bp(white(n, 1071), 300, 2400, 1) * np.exp(-t / .009) * .10
    o.put('sfx', 61.50, hk, 1.5, p=.22, room='room', wet=-12)
    o.put('sfx', 61.56, rustle(.12, 1072, .45, 'thick', [(0, 0), (.02, 1), (.12, 0)]), -5, p=.22, room='room', wet=-14)
    # lifted: sticky paste lets go
    n = ns(.08)
    t = np.arange(n) / SR
    o.put('sfx', 61.74, svf(white(n, 1073), 1200 + 2600 * t / .08, 1.6) * np.sin(np.pi * t / .08) ** 2 * .022, 5, p=.22, room='room', wet=-14)
    ln = [(t_ - 61.9, max(u_ - t_, .030), .55, (x_ - .5) * 1.2, (x_ - .5) * 1.2 + .02) for t_, u_, x_ in STROKES['line']]
    o.put('sfx', 61.9, brush(ln, 1075, size=.45), 2.5, room='room', wet=-14)
    o.put('sfx', 63.08, fur(.16, 1079, 1400), -3, p=-.2, room='room', wet=-13)
    o.put('sfx', 63.19, fur(.34, 1080, 1700), 5.0, p=-.2, room='room', wet=-13)
    o.put('sfx', 63.23, rustle(.22, 1081, .10, 'thick', [(0, 0), (.1, 1), (.22, 0)]), -4, p=-.2, room='room', wet=-14)
    return o.pack()


PARTS = [s1, bed, s2, s3, s4a, s4b, s5, s6, s7, s8, s9]
