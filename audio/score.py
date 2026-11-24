"""THE LAST PAGE, music.  Motif (degrees): 1 5 4 2 | 3 7 6 5 - a question of eight notes whose last note only becomes the tonic
when the ink touches the page.  D minor (solo felt piano) -> D major sunrise -> F major flight -> one F."""
import numpy as np
from core import *
from inst import *
from rooms import *

K = nt


def line(*a):
    """('D4', t, dur, dyn[, opts]) -> (t, dur, midi, dyn, opts)"""
    return [(x[1], x[2], K(x[0]), x[3], x[4] if len(x) > 4 else {}) for x in a]


PL = dict(vn1=16, vn2=14, vla=12, vc=10, cb=7)
KD = dict(vn1='vn', vn2='vn', vla='va', vc='vc', cb='cb')


WETX = [0.0]
BAL = dict(timp=9.0, bd=8.0, glock=8.0, harp=-2.5, choir=2.5)      # section faders


def seatput(out, name, t0, g, gdb, ep=None, wet=None):
    s = seat(name)
    gdb = gdb + BAL.get(name, 0.0)
    for i in range(len(g)):
        h = s[i % len(s)]
        st = np.stack([conv(g[i], h[0]), conv(g[i], h[1])])
        out.put('music', t0, st, gdb, room='stage', wet=(WET[name] if wet is None else wet) + WETX[0], ep=ep)


def S(out, name, notes, gdb=0.0, seed=1, n=None, ep=None, wet=None, **kw):
    t0, g = strings(notes, KD[name], n or PL[name], 4, seed=seed, **kw)
    gdb = gdb + (9.0 if kw.get('art') == 'spic' else 0.0)
    seatput(out, name, t0, g, gdb, ep, wet)


def BRS(out, name, notes, gdb=0.0, seed=1, n=None, ep=None, **kw):
    n = n or len(SEATS[name])
    t0, g = brass(notes, name, n, seed, **kw)
    seatput(out, name, t0, g, gdb, ep)


def CH(out, part, notes, gdb=0.0, seed=1, n=9, ep=None, **kw):
    t0, g = choir(notes, part, n, 4, seed=seed, **kw)
    if part in 'TB':
        g = g[::-1]
    seatput(out, 'choir', t0, g, gdb, ep)


def hit(out, name, t, x, gdb=0.0, ep=None):
    seatput(out, name, t, x[None] if x.ndim == 1 else x, gdb, ep)


def harp(out, notes, gdb=0.0, seed=500, ep=None):
    """[(t, midi, vel)] into one buffer"""
    t0 = min(n[0] for n in notes) - .01
    t1 = max(n[0] for n in notes) + 4.5
    b = np.zeros(ns(t1 - t0))
    for q, (t, m, v) in enumerate(notes):
        x = harp_note(m, v, seed + q)
        i = ns(t - t0)
        e = min(len(b), i + len(x))
        b[i:e] += x[:e - i]
    seatput(out, 'harp', t0, b[None], gdb, ep)


def gliss(t0, t1, lo, hi, scale, v0, v1, curve_=1.15):
    ms = [m for m in range(K(lo), K(hi) + 1) if m % 12 in [x % 12 for x in scale]]
    u = np.linspace(0, 1, len(ms)) ** (1 / curve_)
    r = rng(int(t0 * 100))
    return [(t0 + (t1 - t0) * a + r.normal(0, .004), m, v0 + (v1 - v0) * a) for a, m in zip(u, ms)]


def roll(out, name, t0, t1, fn, v0, v1, gdb=0.0, rate=14.0, seed=700, ep=None, shape=1.6):
    b = np.zeros(ns(t1 - t0 + 4.0))
    r = rng(seed)
    t = 0.0
    q = 0
    while t < t1 - t0:
        u = t / (t1 - t0)
        v = v0 + (v1 - v0) * u ** shape
        x = fn(v * r.uniform(.82, 1.1), seed + q)
        i = ns(t)
        b[i:i + len(x)] += x[:len(b) - i]
        t += 1 / rate * r.uniform(.88, 1.12)
        q += 1
    seatput(out, name, t0, b[None], gdb, ep)


# ------------------------------------------------------------------ piano bus
_SB = {}


def piano(out, notes, gdb=0.0, seed=40, felt=1.0, ep=None, wet=-13.0, halo=-21.0, peds=()):
    """[(t, name, vel, hold)]  close felt piano: strings + soundboard + open-string halo, in a small room and the stage."""
    if not _SB:
        _SB['b'], _SB['s'] = soundboard_ir(), sympathetic_ir()
    t0 = min([n[0] for n in notes] + [p[0] for p in peds]) - .05
    t1 = max(n[0] + min(n[3] + 1.5, 17) for n in notes)
    tr = Track(t0, t1 + 1.0)
    for q, (t, nm, v, h) in enumerate(notes):
        tr.add(t, piano_note(K(nm), v, h, seed + q * 5, felt))
    n = tr.x.shape[1]
    # the pedal: a soft wooden thump, and the strings breathing as the dampers leave them
    br = np.zeros((2, n))
    for q, (t, v) in enumerate(peds):
        k = ns(.4)
        th = modes(k, np.array([56, 81, 119, 168, 260]), [1, .9, .6, .35, .15], [.12, .11, .09, .07, .05], seed + 400 + q, att=.004, phase=0) * .010 * v
        th += bp(white(k, seed + 410 + q), 150, 1800, 1) * np.exp(-np.arange(k) / SR / .030) * (1 - np.exp(-np.arange(k) / SR / .012)) * .0035 * v
        tr.add(t, th, 1.0, -.1)
        i = ns(t - t0)
        e = ns(.06)
        br[:, i:i + e] += bp(white(e, seed + 420 + q), 120, 1600, 1) * np.hanning(e) * .020 * v
    y = tr.x
    b = np.stack([conv(y[c], _SB['b'][c])[:n] for c in (0, 1)])
    w = np.stack([conv(y.mean(0) + br[c], _SB['s'][c])[:n] for c in (0, 1)])
    # halo follows the pedal: it dies with the last note held
    z = .80 * y + .34 * b + float(db(halo)) * w
    z = hp(z, 48, 2)
    z = eq(z, 240, 1.5, .8)
    if felt > .5:
        z = eq(z, 3200, -1.5, .7)
        z = eq(z, 7500, 4.0, .7, 'hs')       # air of a close microphone
    mid, side = (z[0] + z[1]) / 2, (z[0] - z[1]) / 2
    z = np.stack([mid + .62 * side, mid - .62 * side])
    out.put('music', t0, z, gdb, room='stage', wet=wet, ep=ep)
    out.put('music', t0, z, gdb, room='room', wet=-13.0, ep=ep, dry=None)


# ================================================================== A  solo felt piano, D minor (7.6 - 20.6)
def part_a():
    out = Out()
    N_ = [(7.62, 'D4', .42, 4.4), (8.74, 'A4', .47, 3.3), (10.44, 'G4', .33, 1.55), (11.03, 'E4', .36, 1.25),
          (12.37, 'F4', .40, 3.3), (12.41, 'D3', .17, 3.2),
          (13.52, 'C5', .48, 2.2), (15.16, 'Bb4', .34, .60), (15.80, 'A4', .40, 4.75), (15.85, 'G2', .18, 4.7), (15.92, 'Bb3', .18, 4.6),
          (17.46, 'E5', .27, 3.1),
          # the leap
          (18.60, 'E4', .22, .40), (18.92, 'A4', .30, 1.2), (19.01, 'D5', .34, 1.2), (19.11, 'F5', .37, 1.2), (19.23, 'A5', .40, 1.3), (19.43, 'D3', .22, 1.1), (19.46, 'A3', .21, 1.1)]
    piano(out, N_, 1.0, 40, ep=0, peds=((7.40, 1.0), (12.16, .8), (15.62, .8), (18.52, .6)))
    harp(out, [(18.93, K('D4'), .22), (19.02, K('F4'), .25), (19.12, K('A4'), .28), (19.24, K('E5'), .32)], -5, 520, ep=0)
    # a held breath of muted violins under the raised paw
    S(out, 'vn1', [(19.55, 1.3, K('A5'), (.10, .34), dict(art='trem')), ], -9, 3, n=8, ep=0, art='trem', sord=True)
    S(out, 'vn2', [(19.75, 1.1, K('E5'), (.10, .32), dict(art='trem')), ], -10, 4, n=8, ep=0, art='trem', sord=True)
    # the light behind the windows, far away
    # one note of the motif for each window it passes
    for t, m, d, v in ((3.46, 'D5', .62, .20), (4.08, 'A5', .62, .16), (4.71, 'G5', .62, .17), (5.33, 'E5', .80, .15)):
        out.put('music', t, bowed_bar(K(m), d, v, int(t * 10)), -15, room='stage', wet=-3)
    return out.pack()


# ================================================================== B  the bird leaves the page (21.2 - 25)
def part_b():
    out = Out()
    for t, m, d, v in ((21.5, 'A4', 1.5, .22), (22.1, 'E5', 1.4, .20), (22.85, 'D6', .8, .14)):
        out.put('music', t, bowed_bar(K(m), d, v, int(t * 10)), -1.5, room='stage', wet=-5, ep=1)
    arp = ['Bb2', 'F3', 'Bb3', 'D4', 'F4', 'A4', 'C5', 'D5', 'F5', 'A5', 'C6']
    ts = 23.46 + .62 * np.linspace(0, 1, len(arp)) ** .85
    harp(out, [(t, K(m), .34 + .012 * i) for i, (t, m) in enumerate(zip(ts, arp))], -2, 540, ep=1)
    cel = np.zeros(ns(4.5))
    for q, (t, m, v) in enumerate(((23.80, 'F5', .42), (24.08, 'C6', .5), (24.46, 'Bb5', .38), (24.60, 'G5', .40), (24.90, 'A5', .22))):
        for oc, g in ((0, 1.0), (12, .35)):
            x = glock_note(K(m) + oc, v, 560 + q * 3 + oc, soft=True)
            i = ns(t - 23.7)
            cel[i:i + len(x)] += g * x[:len(cel) - i]
    hit(out, 'glock', 23.7, cel, -1.0, ep=1)
    e = [(23.3, -40), (23.55, -5), (24.6, 0), (25.1, -9), (25.6, -60)]
    for nm, m, sd in (('vn1', 'C5', 11), ('vn2', 'A4', 12), ('vla', 'F4', 13), ('vc', 'D3', 14)):
        S(out, nm, [(23.50, 1.8, K(m), (.42, .46))], -5, sd, n=8, ep=1, sord=True, expr=e)
    S(out, 'cb', [(23.50, 1.7, K('Bb1'), (.36, .3))], -8, 15, n=4, ep=1, expr=e)
    return out.pack()


# ================================================================== C  tremor: the motif in the cellars (25 - 31.4)
def part_c():
    out = Out()
    low = line(('D2', 25.25, 1.22, (.26, .40)), ('A2', 26.47, 1.40, (.42, .50)), ('G2', 27.87, .46, .46), ('Eb2', 28.33, 1.02, (.48, .56)),
               ('F2', 29.35, .72, (.54, .60)), ('C3', 30.07, .68, (.62, .82)))
    S(out, 'vc', low, 3.5, 21, ep=1, art='leg', vib=.7)
    S(out, 'cb', [(t, d, m - 12 if m > 38 else m - 12, v, o) for t, d, m, v, o in low], 3.0, 22, ep=1, art='leg', vib=.4)
    S(out, 'cb', [(25.0, 5.7, K('D2'), (.12, .48))], -5, 23, n=5, ep=1, art='trem')
    # violas / seconds: a tremolo that climbs by semitones, sul ponticello
    up = ['A3', 'Bb3', 'B3', 'C4', 'C#4', 'D4', 'Eb4', 'E4', 'F4', 'F#4', 'G4', 'Ab4']
    t, dt = 27.4, .40
    n1, n2 = [], []
    for i, m in enumerate(up):
        d = dt * (1.0 - .042 * i)
        v = .18 + .055 * i
        n1.append((t, d, K(m), v, dict(art='trem')))
        n2.append((t + .06, d, K(m) + 1 + 5, v, dict(art='trem')))
        t += d
        if t > 30.62:
            break
    bd = [(24, 0), (30.70, 0), (30.75, -9), (31.3, -5)]
    S(out, 'vla', n1, -2, 24, ep=1, art='trem', bright=1.45, expr=bd)
    S(out, 'vn2', n2, -4, 25, ep=1, art='trem', bright=1.5, expr=bd)
    S(out, 'vn1', [(28.6, 2.1, K('Bb5'), (.08, .55)), ], -8, 26, n=7, ep=1, art='trem', bright=1.3)
    S(out, 'vn1', [(28.9, 1.8, K('A5'), (.08, .55)), ], -8, 27, n=7, ep=1, art='trem', bright=1.3)
    # three cuts
    for q, t in enumerate((30.75, 30.96, 31.17)):
        v = (.70, .80, .95)[q]
        for nm, ms in (('cb', ['D1']), ('vc', ['D2', 'Ab2']), ('vla', ['Eb3', 'B3']), ('vn2', ['F4', 'A4']), ('vn1', ['Eb5', 'E5'])):
            for j, m in enumerate(ms):
                S(out, nm, [(t, .10, K(m) if K(m) > 27 else K(m), v, dict(art='spic'))], -7 if nm in ('cb', 'vc') else -9, 30 + q * 9 + j * 3 + len(nm), n=5, ep=1,
                  art='spic', tail=.4, bright=1.3)
        BRS(out, 'tbn', [(t, .11, K('D2'), v, dict(art='marc')), ], -12, 41 + q, ep=1, n=3)
        hit(out, 'timp', t, timp(K('D2'), v, 44 + q, tail=1.2), -6, ep=1)
    roll(out, 'timp', 29.9, 30.72, lambda v, s: timp(K('D2'), v, s, 1.4), .10, .42, -5, seed=720, ep=1)
    return out.pack()


# ================================================================== D  the burst and the vortex (32.08 - 39)
def cells(t, dur, notes, v0, v1):
    k = len(notes)
    return [(t + dur * i / k, dur / k * .9, K(m), v0 + (v1 - v0) * i / k, dict(art='spic')) for i, m in enumerate(notes)]


def part_d1():
    out = Out()
    # impact
    for nm, ms, g in (('cb', ['D1'], 6), ('vc', ['D2', 'A2'], 5), ('vla', ['Eb4', 'A3'], 2), ('vn2', ['A4', 'Eb5'], 1), ('vn1', ['Bb5', 'Eb6'], 1)):
        for j, m in enumerate(ms):
            S(out, nm, [(32.08, 1.45, K(m), (1.0, .34), dict(art='marc' if nm in ('cb', 'vc') else 'trem'))], g, 60 + len(nm) * 3 + j, n=PL[nm] // len(ms) + 1,
              ep=2, art='trem', bright=1.2, expr=[(32.0, 0), (32.3, 0), (33.6, -11), (34, -20)])
    BRS(out, 'tbn', [(32.08, 1.3, K('D2'), (.98, .30))], 0, 66, ep=2, art='marc', expr=[(32.0, 0), (32.35, 0), (33.5, -12)])
    BRS(out, 'tbn', [(32.08, 1.3, K('Ab2'), (.95, .30))], -3, 67, ep=2, art='marc', expr=[(32.0, 0), (32.35, 0), (33.5, -12)])
    BRS(out, 'tuba', [(32.08, 1.5, K('D1'), (.95, .4))], -1, 68, ep=2, art='marc', expr=[(32.0, 0), (32.4, 0), (33.6, -10)])
    BRS(out, 'horn', [(32.08, 1.1, K('Eb3'), (.95, .3))], -2, 69, ep=2, art='marc', expr=[(32.0, 0), (32.3, 0), (33.3, -14)])
    hit(out, 'timp', 32.08, timp(K('D2'), 1.0, 70, 3.2), -1, ep=2)
    hit(out, 'bd', 32.08, bassdrum(1.0, 71), 0, ep=2)
    # ostinato = the motif in sixteenths, one note per fold (3 frames)
    A, B = ['D', 'A', 'G', 'E'], ['F', 'C', 'Bb', 'A']

    def cell(c, base):
        o, out_, last = base, [], None
        for q, x in enumerate(c):
            up = {'D': 0, 'A': 0, 'G': 0, 'E': 0, 'F': 0, 'C': 1, 'Bb': 0}.get(x, 0)
            out_.append(f'{x}{base + up}')
        return out_
    o1 = cells(33.75, .5, cell(A, 4), .50, .56) + cells(34.25, .5, cell(B, 4), .56, .62) + cells(34.75, .5, cell(A, 4), .62, .68)
    b0 = 35.25
    bt = [.46, .43, .40, .36]
    seq = [['F4', 'C5', 'Bb4', 'G4'], ['Ab4', 'Eb5', 'Db5', 'Bb4'], ['B4', 'F#5', 'E5', 'C#5'], ['A4', 'E5', 'G5', 'Bb5']]
    t = b0
    for q in range(4):
        o1 += cells(t, bt[q], seq[q], .70 + .06 * q, .76 + .06 * q)
        t += bt[q]
    S(out, 'vla', o1, 1, 80, ep=2, art='spic', bright=1.25)
    S(out, 'vn2', [(a, b, m + (12 if a > 34.7 else 0), v, o) for a, b, m, v, o in o1 if a > 34.2], 0, 81, ep=2, art='spic', bright=1.25)
    S(out, 'vn1', [(a, b, m + 12, v, o) for a, b, m, v, o in o1 if a > 35.2], 0, 82, ep=2, art='spic', bright=1.2)
    S(out, 'vc', [(a, b, m - 12, v, o) for a, b, m, v, o in o1], -1, 83, ep=2, art='spic', bright=1.2)
    # bass pedal then the minor-third staircase
    bass = [(33.75, 1.48, K('D2'), (.5, .62))]
    t = b0
    for q, m in enumerate(('F2', 'Ab2', 'B2', 'A2')):
        bass.append((t, bt[q] * .96, K(m), (.72 + .06 * q, .80 + .06 * q)))
        t += bt[q]
    S(out, 'cb', [(a, b, m - 12, v) for a, b, m, v in bass], 0, 84, ep=2, art='det')
    BRS(out, 'tuba', bass, -8, 85, ep=2, art='ten')
    chords = [['F3', 'A3', 'D4'], ['Ab3', 'C4', 'Eb4'], ['B3', 'D4', 'F#4'], ['A3', 'C#4', 'G4']]
    t = b0
    tb = [[], [], []]
    hn = []
    for q in range(4):
        v = .62 + .09 * q
        for j in range(3):
            # dotted push: long-short on every beat
            tb[j] += [(t, bt[q] * .56, K(chords[q][j]), v, dict(art='marc')), (t + bt[q] * .75, bt[q] * .20, K(chords[q][j]), v * .9, dict(art='marc'))]
        lo, hi = [('F4', 'C5'), ('Ab4', 'Eb5'), ('B4', 'F#5'), ('A4', 'E5')][q]
        hn += [(t, bt[q] * .46, K(lo) - 12 + (0 if q else 0), v, dict(art='marc')), (t + bt[q] * .5, bt[q] * .46, K(hi) - 12, v + .06, dict(art='marc'))]
        t += bt[q]
    for j in range(3):
        BRS(out, 'tbn', tb[j], -9, 86 + j, ep=2, n=2)
    BRS(out, 'horn', hn, -3, 89, ep=2)
    BRS(out, 'tpt', [(a, b, m + 12, v * .92, o) for a, b, m, v, o in hn[4:]], -9, 90, ep=2)
    # high violins wind upward
    S(out, 'vn1', [(35.25, .46, K('D6'), .45), (35.71, .43, K('Eb6'), .52), (36.14, .40, K('F#6'), .60), (36.54, .34, K('G6'), .70)], -7, 91, n=8, ep=2, art='trem')
    t = 33.75
    r = rng(92)
    while t < 36.88:
        g = np.interp(t, [33.75, 35.25, 36.9], [.30, .5, .85])
        acc = 1.0 if abs((t - 33.75) / .5 - round((t - 33.75) / .5)) < .05 else .6
        hit(out, 'timp', t, timp(K('D2') if t < 35.2 else K('F2') if t < 35.7 else K('Ab2') if t < 36.1 else K('B2') if t < 36.5 else K('A2'), g * acc, 93 + int(t * 10), 1.2), -4, ep=2)
        t += .25 if t < 35.2 else .2
    hit(out, 'bd', 35.25, bassdrum(.7, 94), -3, ep=2)
    hit(out, 'bd', 36.90, bassdrum(1.0, 95), -1, ep=2)
    return out.pack()


def part_d2():
    """gold light: Bb maj9 -> C(sus-3) -> D, 37.1 / 38.05 / 39.0"""
    out = Out()
    c = [(36.85, -18), (37.2, -11), (38.05, -6), (38.8, 0), (39.0, 0)]
    vn1 = [(36.92, 1.13, K('F5'), (.55, .66)), (38.05, .95, K('G5'), (.68, .92))]
    vn2 = [(36.92, 1.13, K('A4'), (.55, .66)), (38.05, .95, K('G4'), (.68, .9))]
    vn2b = [(36.92, 1.13, K('C5'), (.55, .66)), (38.05, .95, K('D5'), (.68, .9))]
    vla = [(36.92, 1.13, K('D4'), (.55, .66)), (38.05, .40, K('F4'), .72), (38.45, .55, K('E4'), (.72, .92))]
    vc = [(36.92, 1.13, K('F3'), (.6, .68)), (38.05, .95, K('G3'), (.7, .95))]
    cb = [(36.92, 1.13, K('Bb1'), (.6, .68)), (38.05, .95, K('C2'), (.7, .95))]
    S(out, 'vn1', vn1, 3, 110, ep=2, expr=c, tail=.5)
    S(out, 'vn1', [(a, b, m + 12, (v[0] * .8, v[1] * .85) if isinstance(v, tuple) else v, dict(art='trem')) for a, b, m, v in vn1], -6, 111, n=8, ep=2, expr=c, art='trem', tail=.5)
    S(out, 'vn2', vn2, -2, 112, n=8, ep=2, expr=c, tail=.5)
    S(out, 'vn2', vn2b, -2, 113, n=8, ep=2, expr=c, tail=.5)
    S(out, 'vla', vla, 0, 114, ep=2, expr=c, tail=.5)
    S(out, 'vc', vc, 0, 115, ep=2, expr=c, tail=.5)
    S(out, 'cb', cb, 1, 116, ep=2, expr=c, tail=.5)
    cc = [(36.9, -20), (37.4, -9), (38.1, -4), (39.0, 0)]
    CH(out, 'S', [(37.1, 1.35, K('D5'), (.4, .72), dict(vowel='u>o')), (38.45, .55, K('E5'), (.74, .9), dict(vowel='o>a'))], 1, 120, ep=2, expr=cc, tail=.4)
    CH(out, 'A', [(37.1, .95, K('A4'), (.4, .6), dict(vowel='u>o')), (38.05, .95, K('G4'), (.62, .9), dict(vowel='o>a'))], 1, 121, ep=2, expr=cc, tail=.4)
    CH(out, 'T', [(37.1, .95, K('F4'), (.4, .6), dict(vowel='u>o')), (38.05, .95, K('D4'), (.62, .9), dict(vowel='o>a'))], 1, 122, ep=2, expr=cc, tail=.4)
    CH(out, 'B', [(37.1, .95, K('Bb2'), (.4, .6), dict(vowel='u>o')), (38.05, .95, K('C3'), (.62, .9), dict(vowel='o>a'))], 2, 123, ep=2, expr=cc, tail=.4)
    hc = [(36.9, -14), (37.3, -8), (38.2, -3), (39.0, 0)]
    BRS(out, 'horn', [(37.1, .95, K('D4'), (.45, .62), dict(art='leg')), (38.05, .40, K('F4'), .70, dict(art='leg')), (38.45, .55, K('E4'), (.72, .95), dict(art='leg'))], -5, 133, n=2, ep=2, expr=hc)
    for j, ms in enumerate((('Bb3', 'C4'), ('F4', 'G4'))):
        BRS(out, 'horn', [(37.1, .95, K(ms[0]), (.45, .62), dict(art='leg')), (38.05, .95, K(ms[1]), (.64, .95), dict(art='leg'))], -5, 130 + j, n=2, ep=2, expr=hc)
    for j, ms in enumerate((('Bb2', 'C3'), ('F3', 'G3'))):
        BRS(out, 'tbn', [(37.1, .95, K(ms[0]), (.4, .6), dict(art='leg')), (38.05, .95, K(ms[1]), (.62, .95), dict(art='leg'))], -8, 135 + j, n=2, ep=2, expr=hc)
    BRS(out, 'tuba', [(37.1, .95, K('Bb1'), (.4, .6)), (38.05, .95, K('C2'), (.62, .9))], -8, 138, ep=2, expr=hc)
    BRS(out, 'tpt', [(38.05, .95, K('G4'), (.30, .85), dict(art='swell'))], -8, 139, ep=2, n=2, expr=hc)
    roll(out, 'timp', 37.9, 38.98, lambda v, s: timp(K('A2'), v, s, 1.6), .08, .72, -3, seed=740, ep=2)
    harp(out, gliss(38.30, 38.99, 'C3', 'A6', [0, 2, 4, 7, 9], .3, .62), 0, 580, ep=2)
    sw = cymbal_roll(1.25, 901)
    out.put('music', 37.78, sw, -13, room='stage', wet=-10, ep=2)
    # pick-up arpeggio under the light
    harp(out, [(37.12 + i * .075, K(m), .42) for i, m in enumerate(['Bb2', 'F3', 'Bb3', 'D4', 'F4', 'A4', 'C5', 'F5', 'A5'])], -1, 590, ep=2)
    return out.pack()


_CY = {}


def cym(kind=0):
    if kind not in _CY:
        _CY[kind] = cymbal_ir(31) if kind == 0 else cymbal_ir(33, lo=90, hi=9000, K=800, t_lo=8, t_hi=1.2, L=7.0, bloom=.10)
    return _CY[kind]


def cymbal_roll(dur, seed, dark=0):
    """Soft mallets on a suspended cymbal: many small strikes, brightness blooming with level; choked at the end by the next hit's mask."""
    r = rng(seed)
    n = ns(dur + 3.0)
    ex = np.zeros(n)
    t = 0.0
    while t < dur:
        u = t / dur
        i = ns(t)
        m = ns(.006)
        ex[i:i + m] += r.normal(0, 1, m) * np.hanning(m) * (.04 + u ** 2.2) * r.uniform(.6, 1.1)
        t += r.uniform(.028, .05)
    c = cym(dark)
    y = np.stack([conv(ex, c[k])[:n] for k in (0, 1)])
    br = curve([(0, 1500), (dur * .6, 4500), (dur, 16000), (dur + .2, 16000), (dur + 3, 6000)], n)
    y = np.stack([svf(y[k], br, .6, 'lp') for k in (0, 1)])
    return y * 11.0 * fade(n, .01, 1.0)


def crash(out, t, gdb, seed, dark=0, ep=2, bright=15000):
    m = ns(.004)
    ex = rng(seed).normal(0, 1, m) * np.hanning(m)
    c = cym(dark)
    y = np.stack([conv(ex, c[k]) for k in (0, 1)])
    y = lp(y, bright, 1)
    out.put('music', t, y * 10.0, gdb, room='stage', wet=-11, ep=ep)


# ================================================================== E  sunrise, D major over a D pedal (39 - 44)
def part_e1():
    out = Out()
    x = [(38.9, 0), (40.2, 0), (41.6, -2), (43.0, -4), (43.6, -8), (44.2, -13)]
    q = [(39.0, 1.8), (40.8, 1.2), (42.0, 1.0), (43.0, 1.05)]

    def P(ns_, v=((.95, .82), (.80, .84), (.78, .62), (.60, .35)), **o):
        return [(q[i][0], q[i][1], K(ns_[i]), v[i], o) for i in range(4)]
    S(out, 'vn1', P(['A5', 'B5', 'Bb5', 'A5']), 4, 150, ep=2, expr=x)
    S(out, 'vn1', P(['A6', 'B6', 'Bb6', 'A6'], ((.7, .55), (.55, .5), (.45, .35), (.3, .2)), art='trem'), -9, 151, n=8, ep=2, expr=x, art='trem')
    S(out, 'vn2', P(['F#5', 'G#5', 'G5', 'F#5']), 2.5, 152, ep=2, expr=x)
    S(out, 'vn2', P(['D5', 'E5', 'E5', 'D5'], art='trem'), -5, 153, n=8, ep=2, expr=x, art='trem')
    S(out, 'vla', P(['F#4', 'G#4', 'G4', 'F#4'], art='trem'), -3, 154, n=6, ep=2, expr=x, art='trem')
    S(out, 'vla', P(['E4', 'E4', 'E4', 'E4']), -3, 155, n=7, ep=2, expr=x)
    cel = line(('D3', 39.0, .55, .9), ('A3', 39.55, .45, .88), ('F#4', 40.0, .8, (.9, .85)), ('E4', 40.8, .55, .85), ('B3', 41.35, .35, .78), ('G#3', 41.7, .3, .76),
               ('Bb3', 42.0, .5, .76), ('D4', 42.5, .5, .74), ('C#4', 43.0, .45, .62), ('A3', 43.45, .7, (.55, .35)))
    S(out, 'vc', cel, 2, 156, ep=2, expr=x, vib=1.2)
    S(out, 'cb', [(39.0, 2.6, K('D1') + 12, (.95, .78)), (41.6, 2.4, K('D2'), (.74, .4))], -2, 157, ep=2, expr=x)
    S(out, 'cb', [(39.0, 2.6, K('D1'), (.95, .7)), (41.6, 2.3, K('D1'), (.7, .3))], 0, 158, n=4, ep=2, expr=x)
    return out.pack()


def part_e2():
    out = Out()
    x = [(38.9, 0), (40.2, 0), (41.6, -1.5), (43.0, -4), (43.6, -9), (44.2, -14)]
    q = [(39.0, 1.8), (40.8, 1.2), (42.0, 1.0), (43.0, .95)]

    def P(ns_, v=((.95, .82), (.80, .82), (.76, .62), (.55, .3)), **o):
        return [(q[i][0], q[i][1], K(ns_[i]), v[i], o) for i in range(4)]
    vow = ['a', 'a', 'a>o', 'o>u']
    for part, ms, sd in (('S', ['F#5', 'G#5', 'G5', 'F#5'], 160), ('A', ['D5', 'E5', 'E5', 'D5'], 161), ('T', ['A3', 'B3', 'Bb3', 'A3'], 162), ('B', ['D3', 'D3', 'D3', 'D3'], 163)):
        CH(out, part, [(q[i][0], q[i][1], K(ms[i]), ((.9, .75), (.75, .75), (.7, .55), (.5, .3))[i], dict(vowel=vow[i])) for i in range(4)], 1.5, sd, n=10, ep=2, expr=x)
    # horns: the motif's leap alone, a call
    hn = line(('D4', 39.42, .46, .78, dict(art='ten')), ('A4', 39.88, .92, (.85, .95), dict(art='leg')), ('G#4', 40.8, .55, .85, dict(art='leg')),
              ('B4', 41.35, .65, (.82, .78), dict(art='leg')), ('Bb4', 42.0, .5, .74, dict(art='leg')), ('G4', 42.5, .5, .66, dict(art='leg')), ('F#4', 43.0, .9, (.55, .25), dict(art='leg')))
    BRS(out, 'horn', hn, 0, 165, ep=2, expr=x)
    for j, ms in enumerate((['A3', 'B3', 'Bb3', 'A3'], ['F#3', 'G#3', 'G3', 'F#3'], ['D3', 'E3', 'E3', 'D3'])):
        BRS(out, 'tbn', P(ms, ((.88, .62), (.6, .6), (.55, .4), (.35, .2)), art='leg'), -8, 166 + j, n=2, ep=2, expr=x)
    BRS(out, 'tuba', [(39.0, 2.6, K('D2'), (.88, .6)), (41.7, 1.6, K('D2'), (.55, .25))], -7, 170, ep=2, expr=x)
    for j, m in enumerate(('A4', 'D5', 'F#5')):
        BRS(out, 'tpt', [(39.0, 1.75, K(m), (.86, .4), dict(art='ten'))], -10, 171 + j, n=1, ep=2, expr=[(39, 0), (39.5, -1), (40.7, -12)])
    hit(out, 'timp', 39.0, timp(K('D2'), .95, 175, 3.5), 0, ep=2)
    roll(out, 'timp', 39.12, 40.6, lambda v, s: timp(K('D2'), v, s, 1.6), .42, .10, -5, seed=760, ep=2, shape=.7)
    hit(out, 'bd', 39.0, bassdrum(.9, 176), -1, ep=2)
    crash(out, 39.0, -11, 177)
    ar = []
    for t0, ch in ((39.0, ['D2', 'A2', 'D3', 'F#3', 'A3', 'D4', 'E4', 'F#4', 'A4', 'D5', 'F#5', 'A5', 'D6']),
                   (39.9, ['A5', 'F#5', 'E5', 'D5', 'A4', 'F#4', 'E4', 'D4']), (40.8, ['E3', 'B3', 'E4', 'G#4', 'B4', 'E5', 'G#5', 'B5']),
                   (42.0, ['G3', 'D4', 'G4', 'Bb4', 'E5', 'G5', 'Bb5', 'E6']), (43.0, ['D3', 'A3', 'D4', 'F#4', 'A4', 'E5', 'A5', 'D6'])):
        ar += [(t0 + i * (.062 if t0 == 39.0 else .105), K(m), (.62 if t0 == 39.0 else .46) - .01 * i - (.12 if t0 > 41.9 else 0)) for i, m in enumerate(ch)]
    harp(out, ar, 0, 600, ep=2)
    gl = np.zeros(ns(9))
    for q_, (t, m, v) in enumerate(((39.0, 'F#6', .5), (39.0, 'D6', .4), (39.45, 'A6', .3), (40.8, 'G#6', .34), (41.1, 'B6', .26), (42.0, 'Bb6', .26), (42.5, 'G6', .2), (43.0, 'A6', .18))):
        y = glock_note(K(m), v, 610 + q_)
        i = ns(t - 39)
        gl[i:i + len(y)] += y[:len(gl) - i]
    hit(out, 'glock', 39.0, gl, -3, ep=2)
    return out.pack()


# ================================================================== F  the flock gathers, the held breath (44 - 48.2)
def part_f():
    out = Out()
    x = [(43.9, -6), (45.0, -4), (46.0, -1.5), (46.6, 0), (47.2, 0), (47.34, -3), (47.8, -13), (48.06, -18), (48.14, -60)]
    T = [(44.0, 1.0), (45.0, 1.0), (46.0, 1.2), (47.2, .85)]

    def P(ms, v=((.42, .52), (.54, .64), (.68, .86), (.88, .5)), **o):
        return [(T[i][0], T[i][1], K(ms[i]), v[i], o) for i in range(4)]
    S(out, 'vn1', line(('D5', 44.0, 1.0, (.42, .52)), ('F5', 45.0, 1.0, (.54, .64)), ('G5', 46.0, .6, .70), ('Bb5', 46.6, .6, (.74, .86)), ('C6', 47.2, .85, (.9, .5))), 1, 200, ep=2, expr=x, tail=.35)
    S(out, 'vn2', P(['Bb4', 'D5', 'Eb5', 'G5'], art='trem'), -2, 201, ep=2, expr=x, art='trem', tail=.35)
    S(out, 'vn2', P(['F4', 'A4', 'C5', 'D5'], art='trem'), -4, 202, n=8, ep=2, expr=x, art='trem', tail=.35)
    S(out, 'vla', P(['D4', 'F4', 'G4', 'Bb4']), -1, 203, ep=2, expr=x, tail=.35)
    S(out, 'vc', line(('F3', 44.0, .5, .5), ('Bb3', 44.5, .5, .54), ('A3', 45.0, .5, .58), ('D4', 45.5, .5, .64), ('C4', 46.0, .6, .70), ('Eb4', 46.6, .6, (.74, .84)), ('F4', 47.2, .85, (.86, .45))), 0, 204, ep=2, expr=x, tail=.35)
    S(out, 'cb', P(['D2', 'G1', 'Ab1', 'C2']), 1, 205, ep=2, expr=x, tail=.35)
    # horns climb by the motif's own fifth
    hn = line(('Bb3', 44.5, .24, .50, dict(art='ten')), ('F4', 44.74, .62, (.56, .5), dict(art='leg')), ('G3', 45.5, .24, .58, dict(art='ten')), ('D4', 45.74, .5, (.64, .6), dict(art='leg')),
              ('Ab3', 46.3, .28, .70, dict(art='ten')), ('Eb4', 46.58, .60, (.78, .86), dict(art='leg')), ('C4', 47.2, .30, .84, dict(art='ten')), ('G4', 47.5, .5, (.6, .35), dict(art='leg')))
    BRS(out, 'horn', hn, -1, 210, ep=2, expr=[(44, 0), (47.2, 0), (47.5, -6), (48.05, -22), (48.12, -60)], tail=.2)
    for j, ms in enumerate((['Bb2', 'D3', 'C3', 'Bb2'], ['F3', 'F3', 'Eb3', 'F3'])):
        BRS(out, 'tbn', P(ms, ((.3, .4), (.42, .5), (.55, .78), (.75, .3)), art='leg'), -8, 211 + j, n=2, ep=2, expr=x, tail=.2)
    CH(out, 'S', [(45.0, 1.0, K('A4'), (.25, .45), dict(vowel='u')), (46.0, 1.2, K('C5'), (.5, .8), dict(vowel='u>a')), (47.2, .8, K('D5'), (.85, .4), dict(vowel='a>u'))], 0, 215, ep=2, expr=x, tail=.25)
    CH(out, 'A', [(45.0, 1.0, K('F4'), (.25, .45), dict(vowel='u')), (46.0, 1.2, K('G4'), (.5, .8), dict(vowel='u>a')), (47.2, .8, K('G4'), (.85, .4), dict(vowel='a>u'))], 0, 216, ep=2, expr=x, tail=.25)
    CH(out, 'T', [(45.0, 1.0, K('D4'), (.25, .45), dict(vowel='u')), (46.0, 1.2, K('Eb4'), (.5, .8), dict(vowel='u>a')), (47.2, .8, K('F4'), (.85, .4), dict(vowel='a>u'))], 0, 217, ep=2, expr=x, tail=.25)
    # harp sixteenths: paper glitter
    ar = []
    pat = [['Bb3', 'D4', 'F4', 'Bb4', 'D5', 'F5', 'D5', 'Bb4'], ['G3', 'Bb3', 'D4', 'F4', 'A4', 'D5', 'F5', 'A5'], ['Ab3', 'C4', 'Eb4', 'G4', 'C5', 'Eb5', 'G5', 'C6', 'Eb6', 'G6']]
    for i in range(3):
        k = len(pat[i])
        ar += [(T[i][0] + j * T[i][1] / k, K(m), .34 + .08 * i + .02 * (j == 0)) for j, m in enumerate(pat[i])]
    ar += gliss(47.55, 48.17, 'C4', 'C7', [0, 2, 5, 7, 10], .18, .60, .8)
    harp(out, ar, 0, 640, ep=2)
    gl = np.zeros(ns(6))
    for q_, (t, m, v) in enumerate(((44.5, 'F6', .2), (45.0, 'A6', .22), (45.5, 'D6', .2), (46.0, 'G6', .28), (46.3, 'C6', .2), (46.6, 'Bb6', .3), (46.9, 'Eb6', .22), (47.2, 'C7', .34))):
        y = glock_note(K(m), v, 660 + q_)
        i = ns(t - 44)
        gl[i:i + len(y)] += y[:len(gl) - i]
    hit(out, 'glock', 44.0, gl, -3, ep=2)
    roll(out, 'timp', 46.0, 47.2, lambda v, s: timp(K('C3'), v, s, 1.2), .10, .55, -4, seed=780, ep=2)
    roll(out, 'timp', 47.75, 48.12, lambda v, s: timp(K('C3'), v, s, .5), .08, .30, -8, seed=790, ep=2)
    y = cymbal_roll(.62, 903)
    m = ns(.66)
    y = y[:, :m + ns(.04)] * fade(m + ns(.04), .01, .05)
    out.put('music', 47.5, y, -17, room='stage', wet=-16, ep=2)
    return out.pack()


# ================================================================== G  the motif at full scale, F major (48.2 - 52)
BT = 3.6 / 7
TH = [('F', 0, 1), ('C', 1, 1.5), ('Bb', 2.5, .5), ('G', 3, 1), ('A', 4, 1), ('E', 5, 1.5), ('D', 6.5, .5), ('C', 7, 2.4)]
OCT = [0, 1, 0, 0, 0, 1, 1, 1]
HX = [(47.9, 0), (51.2, 0), (51.9, -1.0), (52.4, -4), (53.1, -10), (53.9, -18), (54.6, -30)]


def th(base, v, art='det', k=8):
    return [(48.4 + b * BT, d * BT, K(f'{n}{base + OCT[i]}'), v[i], dict(art=art, swell=.14 if d > 1 else .06)) for i, (n, b, d) in enumerate(TH[:k])]


VV = [.95, (.95, 1.04), .92, (.92, .95), .97, (1.0, 1.1), .98, (.86, .40)]


def part_g1():
    out = Out()
    WETX[0] = -3.0
    S(out, 'vn1', th(5, VV), 5.0, 230, ep=2, expr=HX, vib=1.25, loose=.8)
    S(out, 'vn2', th(5, VV), 3.0, 231, n=8, ep=2, expr=HX, vib=1.2, loose=.8)
    S(out, 'vn2', th(4, VV), 2.5, 232, n=10, ep=2, expr=HX, vib=1.2, loose=.8)
    S(out, 'vla', th(4, VV), 1.0, 233, n=7, ep=2, expr=HX, vib=1.2, loose=.8)
    # inner sixteenths: the air
    pat = [(0, ['F4', 'A4', 'C5', 'A4']), (1, ['D4', 'F4', 'A4', 'C5']), (2, ['A4', 'F4', 'D4', 'F4']), (3, ['G4', 'Bb4', 'D5', 'Bb4']), (4, ['F4', 'Bb4', 'D5', 'F5']),
           (5, ['E4', 'G4', 'C5', 'E5']), (6, ['G4', 'C5', 'D4', 'Bb4'])]
    six = []
    for b, ms in pat:
        six += [(48.4 + (b + j / 4) * BT, BT / 4 * .92, K(m), .78 + .06 * (j == 0), dict(art='spic')) for j, m in enumerate(ms)]
    S(out, 'vla', six, -1, 234, n=7, ep=2, art='spic', expr=HX, bright=1.2)
    S(out, 'vn2', [(a, b, m + 12, v, o) for a, b, m, v, o in six], -4, 235, n=6, ep=2, art='spic', expr=HX, bright=1.1)
    # cellos: counter-melody, then the suspension
    ct = line(('C4', 48.4, BT, .9), ('D4', 48.4 + BT, BT / 2, .9), ('E4', 48.4 + 1.5 * BT, BT / 2, .92), ('F4', 48.4 + 2 * BT, BT, .94), ('Bb3', 48.4 + 3 * BT, BT, .94),
              ('D4', 48.4 + 4 * BT, BT, .96), ('G4', 48.4 + 5 * BT, BT * .5, .98), ('F4', 48.4 + 5.5 * BT, BT * .5, .96), ('E4', 48.4 + 6 * BT, BT * .5, .92), ('F4', 48.4 + 6.5 * BT, BT * .5, .88), ('A3', 52.0, 1.2, (.75, .30)))
    S(out, 'vc', ct, -1, 236, n=8, ep=2, expr=HX, vib=1.25)
    bs = [('F2', -.39, 1.36), ('D2', 1, 2), ('G1', 3, 1), ('Bb1', 4, 1), ('C2', 5, 2), ('F1', 7, 2.9)]
    bl = [(48.4 + b * BT - (.03 if b == 7 else 0), d * BT * .97, K(m), .96 if b < 7 else (.8, .4)) for m, b, d in bs]
    S(out, 'cb', bl, 2, 237, ep=2, art='det', expr=HX)
    S(out, 'vc', [(a, b, m + 12 if m < K('C2') else m, v) for a, b, m, v in bl], 0, 238, n=6, ep=2, art='det', expr=HX)
    return out.pack()


def part_g2():
    out = Out()
    WETX[0] = -3.0
    hn = line(('C4', 48.4, BT, .9, dict(art='ten')), ('D4', 48.4 + BT, BT / 2, .9, dict(art='leg')), ('E4', 48.4 + 1.5 * BT, BT / 2, .92, dict(art='leg')), ('F4', 48.4 + 2 * BT, BT, (.94, .98), dict(art='leg')),
              ('Bb4', 48.4 + 3 * BT, BT, .98, dict(art='ten')), ('D5', 48.4 + 4 * BT, BT, 1.0, dict(art='leg')), ('C5', 48.4 + 5 * BT, BT * .5, .98, dict(art='leg')), ('Bb4', 48.4 + 5.5 * BT, BT * .5, .95, dict(art='leg')),
              ('G4', 48.4 + 6 * BT, BT * .5, .92, dict(art='leg')), ('F4', 48.4 + 6.5 * BT, BT * .5, .88, dict(art='leg')), ('A4', 52.0, 1.15, (.70, .22), dict(art='leg')))
    BRS(out, 'horn', hn, -1.5, 240, ep=2, expr=HX)
    BRS(out, 'tpt', [(a, b * .97, m, v, dict(art='ten')) for a, b, m, v, o in th(4, [.86, (.86, .96), .82, .84, .88, (.92, 1.0), .85, (.5, .2)])], -1.0, 241, ep=2, expr=HX, n=3)
    ch = [(-.39, 1.36, ['F2', 'C3', 'A3']), (1, 1.5, ['D3', 'A3', 'F3']), (2.5, .5, ['D3', 'Bb3', 'F3']), (3, 1, ['G2', 'Bb3', 'F3']), (4, 1, ['Bb2', 'D3', 'A3']), (5, 1.5, ['C3', 'G3', 'E3']),
          (6.5, .5, ['C3', 'Bb3', 'F3']), (7, 2.2, ['F2', 'C3', 'A3'])]
    for j in range(3):
        BRS(out, 'tbn', [(48.4 + b * BT, d * BT * .95, K(c[j]), .82 if b < 7 else (.55, .25), dict(art='marc' if b < 0 else 'ten')) for b, d, c in ch], -7, 242 + j, n=2, ep=2, expr=HX)
    BRS(out, 'tuba', [(48.4 + b * BT, d * BT * .95, K(m), .9 if b < 7 else (.5, .3)) for m, b, d in (('F1', -.39, 1.36), ('D2', 1, 2), ('G1', 3, 1), ('Bb1', 4, 1), ('C2', 5, 2), ('F1', 7, 1.8))], -5, 246, ep=2, expr=HX)
    sat = dict(S=['C5', 'C5', 'D5', 'D5', 'D5', 'E5', 'D5', 'C5'], A=['A4', 'A4', 'Bb4', 'Bb4', 'A4', 'C5', 'Bb4', 'A4'], T=['F4', 'F4', 'F4', 'F4', 'F4', 'G4', 'F4', 'F4'], B=['F3', 'D3', 'D3', 'G3', 'Bb2', 'C3', 'C3', 'F3'])
    for q, (p, ms) in enumerate(sat.items()):
        CH(out, p, [(48.4 + ch[i][0] * BT if i else 48.4, (ch[i][1] if i else 1) * BT, K(m), .92 if i < 7 else (.6, .25), dict(vowel='a' if i < 7 else 'a>u')) for i, m in enumerate(ms)], -1.0, 250 + q, n=10, ep=2, expr=HX)
    # the instrument that first spoke, now a concert grand in octaves under the tutti
    pn = []
    for i, (n_, b, d) in enumerate(TH):
        for o in (0, 12):
            m = K(f'{n_}{4 + OCT[i]}') + o
            pn.append((48.4 + b * BT, m, .82 if i < 7 else .5, d * BT + .1))
    tr = Track(48.3, 58.0)
    for q, (t, m, v, h) in enumerate(pn):
        tr.add(t, piano_note(m, v, h if q < 14 else 2.0, 900 + q * 5, felt=0.0))
    for t, m, v in ((48.4, 'F2', .8), (48.4, 'C3', .7), (48.4 + 3 * BT, 'G2', .75), (48.4 + 5 * BT, 'C2', .8)):
        tr.add(t, piano_note(K(m), v, 1.0, 960 + K(m), felt=0.0))
    seatput(out, 'piano', 48.3, tr.x, -1.5, 2)
    gl = np.zeros(ns(8))
    for i, (n_, b, d) in enumerate(TH):
        y = glock_note(K(f'{n_}{6 + OCT[i]}'), .62 if i < 7 else .3, 970 + i)
        k = ns(b * BT)
        gl[k:k + len(y)] += y[:len(gl) - k]
    hit(out, 'glock', 48.4, gl, -2.5, ep=2)
    for t, m, v in ((48.4, 'F2', .9), (48.4 + BT, 'D2', .7), (48.4 + 3 * BT, 'G2', .78), (48.4 + 4 * BT, 'Bb2', .8), (52.0, 'F2', .38)):
        hit(out, 'timp', t, timp(K(m), v, 990 + int(t * 10), 2.4), -1, ep=2)
    roll(out, 'timp', 48.4 + 5 * BT, 51.95, lambda v, s: timp(K('C3'), v, s, 1.2), .62, .30, -5, seed=800, ep=2, shape=.6)
    hit(out, 'timp', 48.2, timp(K('F2'), 1.0, 995, 3.0), 0, ep=2)
    hit(out, 'bd', 48.2, bassdrum(1.0, 996), 2, ep=2)
    hit(out, 'bd', 48.4 + 5 * BT, bassdrum(.6, 997), -4, ep=2)
    crash(out, 48.4, -8, 998)
    crash(out, 48.4 + 5 * BT, -13, 999)
    harp(out, gliss(48.42, 48.4 + BT, 'F3', 'F6', [5, 7, 9, 10, 0, 2, 4], .5, .7) + gliss(48.4 + 4.2 * BT, 48.4 + 5 * BT, 'C4', 'E7', [5, 7, 9, 10, 0, 2, 4], .45, .7)
         + [(52.0 + i * .13, K(m), .34 - .015 * i) for i, m in enumerate(['F2', 'C3', 'F3', 'A3', 'C4', 'F4', 'A4', 'C5', 'F5', 'A5'])], 1, 680, ep=2)
    return out.pack()


# ================================================================== H  quiet, and the last note (52 - 65)
def part_h():
    out = Out()
    x = [(51.8, -30), (52.1, -4), (53.2, -2), (54.0, -5), (54.9, -13), (56.0, -22), (57.4, -33), (58.2, -60)]
    q = [(52.0, 1.30), (53.30, .85), (54.15, 2.4)]

    def P(ms, v=((.52, .5), (.5, .46), (.4, .2))):
        return [(q[i][0], q[i][1], K(ms[i]), v[i]) for i in range(len(ms))]
    S(out, 'vn1', P(['C6', 'Db6', 'C6']), 7, 270, n=10, ep=2, sord=True, expr=x)
    S(out, 'vn2', P(['A5', 'Bb5', 'A5']), 5, 271, n=10, ep=2, sord=True, expr=x)
    S(out, 'vla', P(['F4', 'F4', 'F4']), 5, 272, n=8, ep=2, sord=True, expr=x)
    S(out, 'vc', P(['C4', 'Db4', 'C4']), 6, 273, n=6, ep=2, sord=True, expr=x)
    S(out, 'cb', [(52.0, 2.6, K('F2'), (.46, .3))], 4, 274, n=4, ep=2, expr=x)
    piano(out, [(53.60, 'F4', .34, 4.6), (54.56, 'C5', .38, 3.7), (55.62, 'Bb4', .28, 2.7), (55.98, 'G4', .30, 2.4),
                (56.74, 'A4', .33, 1.75), (56.80, 'D3', .18, 1.7), (57.46, 'E5', .35, 1.05), (58.02, 'D5', .24, .46)], 14.0, 300, ep=2, wet=-14, peds=((53.42, 1.0), (56.58, .7)))
    piano(out, [(58.70, 'F4', .46, 15.0)], 15.0, 330, ep=2, wet=-12, halo=-19, peds=((58.48, .7),))
    return out.pack()


PARTS = [part_a, part_b, part_c, part_d1, part_d2, part_e1, part_e2, part_f, part_g1, part_g2, part_h]
