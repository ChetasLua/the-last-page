"""Per-second table and whole-film picture."""
import sys
import numpy as np
import soundfile as sf
from core import *
from check import *
ROOT = __file__.rsplit('/audio/', 1)[0]


def load(n):
    x, _ = sf.read(f'{ROOT}/out/{n}.wav', always_2d=True, dtype='float64')
    return x.T


if __name__ == '__main__':
    x, m, s = load('soundtrack'), load('stem_music'), load('stem_sfx')
    levels(x, 'mix')
    r = lambda y, a, b: 20 * np.log10(np.sqrt((y[:, ns(a):ns(b)] ** 2).mean()) + 1e-9)
    print(' t    mix   mus   sfx    peak')
    for i in range(65):
        print(f'{i:2d} {r(x,i,i+1):6.1f}{r(m,i,i+1):6.1f}{r(s,i,i+1):6.1f} {20*np.log10(np.abs(x[:,ns(i):ns(i+1)]).max()+1e-9):6.1f}', end='  |' if i % 2 == 0 else '\n')
    print()
    for a, b in ((20.66, 21.18), (31.42, 31.98), (58.3, 58.68), (64.8, 65.0)):
        print(f'quiet {a}-{b}: mix {r(x,a,b):.1f} dBFS  mus {r(m,a,b):.1f} sfx {r(s,a,b):.1f}')
    if len(sys.argv) > 1:
        a, b = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (0, 65)
        src = dict(mix=x, mus=m, sfx=s)[sys.argv[4] if len(sys.argv) > 4 else 'mix']
        spectro(src, sys.argv[1], a, b, H=370, dbr=96 if b - a > 30 else 84)
