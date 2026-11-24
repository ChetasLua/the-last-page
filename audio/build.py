"""Rebuild the soundtrack: ./.mm/env/bin/python audio/build.py   (writes out/soundtrack.wav, out/stem_music.wav, out/stem_sfx.wav)"""
import os
import sys
import time
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(k, '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from core import *
import rooms

ROOT = os.path.dirname(HERE)

# ---- mix ---------------------------------------------------------------- stem / epoch / room
MUSIC_EP = {0: -13.8, 1: -25.0, 2: -26.5, 3: -14.0}      # solo piano world | tremor | tutti | silences
SFX_EP = {0: -13.8, 1: -15.0, 2: -17.5, 3: -14.0}
RET = dict(stage=0.0, hall=0.0, tower=0.0, sky=0.0, room=0.0)
# fader rides (absolute seconds, dB)
RIDE_M = [(0, 0), (31.9, 0), (32.0, 1.2), (33.6, 1), (35.2, 2), (36.9, 2.5), (38.0, 1), (39.0, .5), (40.8, 1), (43.0, 0), (46, 0), (48.0, 0), (48.4, .5), (51.5, .5), (52.6, 0), (65, 0)]
RIDE_S = [(0, 0), (31.9, 0), (32.0, -1.0), (33.0, -.5), (34, 0), (48.1, 0), (48.2, .6), (48.7, 0), (65, 0)]
# silences: epoch windows (fade in start, in end, fade out start, out end)
WIN = [(-1, 0, 20.50, 20.62), (20.58, 20.60, 31.30, 31.395), (32.062, 32.079, 64.90, 65.0), (20.5, 20.55, 32.0, 32.08)]


def job(name):
    mod, fn = name.split('.')
    m = __import__(mod)
    t = time.time()
    if mod == 'score':
        m.WETX[0] = 0.0
    d = getattr(m, fn)()
    return name, d, time.time() - t


def main(only=None):
    T0 = time.time()
    import score
    import scenes
    jobs = ['score.' + f.__name__ for f in score.PARTS] + ['scenes.' + f.__name__ for f in scenes.PARTS]
    order = ['scenes.s6', 'score.part_g2', 'score.part_a', 'scenes.s8', 'score.part_d2', 'scenes.s9']
    jobs = order + [j for j in jobs if j not in order]
    acc, tim = {}, {}
    with ProcessPoolExecutor(4) as ex:
        for name, d, t in ex.map(job, jobs):
            tim[name] = t
            for k, v in d.items():
                if k in acc:
                    acc[k] += v
                else:
                    acc[k] = v.astype(np.float64)
    print('parts %.1fs' % (time.time() - T0), {k: round(v, 1) for k, v in tim.items() if v > 2})
    irs = rooms.all_rooms()
    bus = {'music': np.zeros((2, N)), 'sfx': np.zeros((2, N))}

    def wet(item):
        (stem, e, room), x = item
        if room == 'dry':
            return stem, e, x
        h = irs[room]
        y = np.stack([conv(x[c], h[c])[:x.shape[1]] for c in (0, 1)])
        return stem, e, y * float(db(RET[room]))
    with ThreadPoolExecutor(3) as ex:
        for stem, e, y in ex.map(wet, list(acc.items())):
            a, b = rooms.EP[e]
            g = float(db((MUSIC_EP if stem == 'music' else SFX_EP)[e]))
            t = np.arange(y.shape[1]) / SR + a
            w = WIN[e]
            win = smoothstep((t - w[0]) / (w[1] - w[0])) * (1 - smoothstep((t - w[2]) / (w[3] - w[2])))
            i = ns(a)
            bus[stem][:, i:i + y.shape[1]] += y * win * g
    bus['music'] *= curve(RIDE_M, N, kind='db')
    bus['sfx'] *= curve(RIDE_S, N, kind='db')
    # console: the same linear shaping on both stems
    for k in bus:
        x = signal.sosfilt(signal.butter(2, 22, 'highpass', fs=SR, output='sos'), bus[k], axis=1)
        bus[k] = x
    bus['music'] = eq(bus['music'], 9000, 1.5, .7, 'hs')
    # one centre for the low end, a little less width than the raw rooms give
    for k, (f, w) in dict(music=(130, .80), sfx=(95, .92)).items():
        mid, side = (bus[k][0] + bus[k][1]) / 2, (bus[k][0] - bus[k][1]) / 2
        side = hp(side, f, 2) * w
        bus[k] = np.stack([mid + side, mid - side])
    mix = bus['music'] + bus['sfx']
    I0, _, _ = loudness(mix)
    g = float(db(np.clip(TARGET_PRE - I0, -TRIM, TRIM)))
    mix = mix * g
    lg = limiter_gain(mix, -1.4)
    gr = -20 * np.log10(lg.min())
    out = {k: bus[k] * g * lg for k in bus}
    hit = np.flatnonzero(lg < .84)
    sec = sorted(set((hit // (SR // 10)).tolist()))
    print('limiter activity (>1.5 dB) at', [round(q / 10, 1) for q in sec])
    edge = np.ones(N)
    edge[:96] = np.linspace(0, 1, 96)
    k = ns(.1)
    edge[-k:] *= np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    out = {k: (v * edge).astype(np.float32) for k, v in out.items()}
    full = (out['music'].astype(np.float64) + out['sfx']).astype(np.float32)
    os.makedirs(os.path.join(ROOT, 'out'), exist_ok=True)
    for nm, x in (('soundtrack', full), ('stem_music', out['music']), ('stem_sfx', out['sfx'])):
        tmp = os.path.join(ROOT, 'out', nm + '.tmp.wav')
        write_wav(tmp, x)
        os.replace(tmp, os.path.join(ROOT, 'out', nm + '.wav'))
    print('pre %.1f LUFS, gain %.1f dB, limiter max reduction %.1f dB, reduced >1 dB for %.2f s' % (I0, 20 * np.log10(g), gr, (lg < .89).sum() / SR))
    import check
    check.levels(full.astype(np.float64), 'soundtrack')
    print('total %.1fs' % (time.time() - T0))


TARGET_PRE, TRIM = -15.5, 1.5

if __name__ == '__main__':
    main()
