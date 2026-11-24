"""Sync table, click hunt, silence and stereo checks on the rendered files."""
import sys
import numpy as np
from report import *

EV = [  # (name, time, stem, band lo, band hi)
    ('lantern set', 11.315, 's', 200, 3000), ('piano 1', 7.62, 'm', 200, 3000), ('piano looks up', 12.37, 'm', 250, 3000),
    ('sniff 1', 16.0, 's', 1500, 9000), ('spring', 18.905, 's', 60, 1500), ('landing', 19.415, 's', 60, 2500), ('ink starts', 21.167, 's', 1500, 9000),
    ('peel', 23.15, 's', 1500, 9000), ('wing 1', 23.54, 's', 300, 6000), ('book jolt', 25.417, 's', 200, 3000), ('cut 1', 30.75, 's', 40, 3000),
    ('cut 2', 30.96, 's', 40, 3000), ('cut 3', 31.17, 's', 40, 3000), ('burst', 32.08, 's', 40, 8000), ('burst (music)', 32.08, 'm', 400, 6000),
    ('fold 1', 33.75, 's', 3000, 16000), ('fold 7', 34.50, 's', 3000, 16000), ('fold 13', 35.25, 's', 3000, 16000), ('vortex beat', 35.25, 'm', 40, 2000),
    ('shutters', 36.90, 's', 40, 9000), ('gold light (harp)', 37.12, 'm', 900, 5000), ('sunrise', 39.0, 'm', 60, 12000), ('wind ring', 42.94, 's', 60, 400),
    ('wingbeat', 48.2, 's', 40, 6000), ('wingbeat (drums)', 48.2, 'm', 30, 400), ('theme', 48.4, 'm', 900, 8000), ('resolution', 52.0, 'm', 40, 130),
    ('page lands', 53.9, 's', 300, 8000), ('paw', 58.5, 's', 60, 1200), ('last note', 58.70, 'm', 250, 3000), ('brush', 58.833, 's', 1000, 9000),
    ('hanko', 61.5, 's', 60, 1200), ('tail flick', 63.19, 's', 1000, 8000)]


def onset(x, t, lo, hi, win=.085):
    a = max(t - .5, 0)
    y = bp(x[:, ns(a):ns(t + .5)].mean(0), lo, hi, 2)
    h = 48
    e = np.sqrt(np.convolve(y ** 2, np.ones(240) / 240, 'full')[:len(y)][::h] + 1e-14)
    L = 20 * np.log10(e + 1e-7)
    k = 8
    d = np.zeros(len(L))
    d[k:] = L[k:] - L[:-k]
    # energy-weighted rise so silence-to-nothing does not win
    w = np.maximum(d, 0) * (e / e.max()) ** .5
    i0, i1 = int((t - a - win) * 1000), int((t - a + win) * 1000)
    i0 = max(i0, 0)
    j = i0 + np.argmax(w[i0:i1])
    # walk back to where the rise begins
    q = j
    while q > 1 and L[q - 1] < L[q] and L[j] - L[q] < 9:
        q -= 1
    return a + q / 1000.0


if __name__ == '__main__':
    x, m, s = load('soundtrack'), load('stem_music'), load('stem_sfx')
    assert x.shape == (2, 3120000), x.shape
    print('stems sum to mix, max error', np.abs(m + s - x).max())
    bad = 0
    for nm, t, st, lo, hi in EV:
        o = onset(s if st == 's' else m, t, lo, hi)
        f = (o - t) * 24
        flag = '' if abs(f) <= 1.0 else '  <<<'
        bad += bool(flag)
        print(f'{nm:18s} map {t:7.3f}  heard {o:7.3f}  {f:+5.2f} frames{flag}')
    print('outside one frame:', bad)
    # clicks: a sample whose 2nd difference is far above its neighbourhood, with no broadband burst around it
    for nm, y in (('music', m), ('sfx', s)):
        for ch in range(2):
            d = np.abs(np.diff(y[ch], 2))
            k = 480
            ref = np.sqrt(np.convolve(d ** 2, np.ones(k) / k, 'same'))
            r = d / (ref + 2e-6)
            i = np.flatnonzero((r > 17) & (d > 4e-4))
            if len(i):
                g = [i[0]]
                for v in i[1:]:
                    if v - g[-1] > 2400:
                        g.append(v)
                print(nm, 'LR'[ch], 'suspects', [(round(v / SR, 3), round(float(r[v]), 1), round(float(d[v]), 4)) for v in g[:14]], len(g))
    z = lambda a, b: 20 * np.log10(np.sqrt((x[:, ns(a):ns(b)] ** 2).mean()) + 1e-9)
    print('silence 20.70-21.15 %.1f dBFS, 31.42-31.98 %.1f dBFS, first 10 ms %.1f, last 20 ms %.1f' % (z(20.70, 21.15), z(31.42, 31.98), z(0, .01), z(64.98, 65)))
    for a, b in ((0.5, 7), (7.6, 18), (25, 31), (32.1, 33.5), (35.2, 37), (39, 42), (44, 47), (48.2, 49), (49, 52), (54, 58), (59, 63)):
        y = x[:, ns(a):ns(b)]
        mid, side = y.mean(0), (y[0] - y[1]) / 2
        print(f'{a:5.1f}-{b:5.1f} corr {np.corrcoef(y)[0,1]:+.2f}  side/mid {10*np.log10((side**2).mean()/(mid**2).mean()):6.1f} dB   L-R balance {10*np.log10((y[0]**2).mean()/(y[1]**2).mean()):+.1f} dB')
