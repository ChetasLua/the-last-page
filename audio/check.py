"""Pictures and numbers: spectrogram PNG, per-second levels, loudness, onset table."""
import sys
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw
from scipy import signal
from core import *


def cmap(v):
    """v 0..1 -> rgb (dark blue, violet, orange, cream)."""
    xp = [0, .2, .4, .6, .8, 1]
    c = np.array([[4, 4, 18], [32, 20, 80], [120, 30, 110], [215, 80, 60], [250, 180, 70], [255, 252, 225]], float)
    return np.stack([np.interp(v, xp, c[:, k]) for k in range(3)], -1).astype(np.uint8)


def spectro(x, path, t0=0.0, t1=None, W=1500, H=420, fmin=30, fmax=20000, dbr=84, top=None, marks=(), nfft=4096, ref=None, lin=False):
    """Two stacked log-f spectrograms (L above R if stereo) plus an RMS strip."""
    if x.ndim == 1:
        x = x[None]
    t1 = t1 or x.shape[1] / SR
    x = x[:, ns(t0):ns(t1)]
    n = x.shape[1]
    hop = max(1, n // W)
    win = np.hanning(nfft)
    ims = []
    S = []
    for ch in x:
        p = np.pad(ch, (nfft // 2, nfft))
        cols = []
        for i in range(W):
            seg = p[i * hop:i * hop + nfft]
            cols.append(np.abs(np.fft.rfft(seg * win)) / (nfft / 4))
        S.append(np.array(cols).T)
    f = np.fft.rfftfreq(nfft, 1 / SR)
    fy = np.linspace(fmin, fmax, H) if lin else np.exp(np.linspace(np.log(fmin), np.log(fmax), H))
    mx = top if top is not None else 20 * np.log10(max(s.max() for s in S) + 1e-9)
    for s in S:
        e = np.stack([np.interp(fy, f, s[:, i]) for i in range(W)], 1)
        v = np.clip((20 * np.log10(e + 1e-9) - (mx - dbr)) / dbr, 0, 1)
        ims.append(cmap(v[::-1]))
    strip = np.full((110, W, 3), 12, np.uint8)
    img = Image.fromarray(np.concatenate(ims + [strip], 0))
    d = ImageDraw.Draw(img)
    for c in range(len(S)):
        for fr in (50, 100, 200, 500, 1000, 2000, 4000, 8000, 16000):
            if fmin < fr < fmax:
                y = (fr - fmin) / (fmax - fmin) if lin else np.log(fr / fmin) / np.log(fmax / fmin)
                yy = c * H + H - 1 - int(y * H)
                d.line([(0, yy), (6, yy)], fill=(255, 255, 255))
                d.text((8, yy - 6), str(fr), fill=(170, 220, 255))
    base = H * len(S)
    w = max(64, hop)
    pk = np.array([np.abs(x[:, i * hop:i * hop + w]).max() for i in range(W)])
    rm = np.array([np.sqrt((x[:, i * hop:i * hop + w] ** 2).mean()) for i in range(W)])
    for arr_, col in ((pk, (90, 90, 140)), (rm, (255, 200, 90))):
        yy = base + 108 - np.clip((20 * np.log10(arr_ + 1e-6) + 80) / 80, 0, 1) * 106
        d.line([(i, float(yy[i])) for i in range(W)], fill=col)
    st = t1 - t0
    step = next(s for s in (.01, .02, .05, .1, .2, .5, 1, 2, 5) if st / s < 26)
    for k in range(int(np.ceil(t0 / step)), int(t1 / step) + 1):
        X = (k * step - t0) / st * W
        d.line([(X, base), (X, base + 8)], fill=(255, 255, 255))
        d.text((X + 2, base + 1), f'{k*step:g}', fill=(200, 200, 200))
    for m in marks:
        X = (m - t0) / st * W
        d.line([(X, 0), (X, base)], fill=(0, 255, 180))
    img.save(path)


def levels(x, name=''):
    I, ST, M = loudness(x)
    print(f'{name} integrated {I:.2f} LUFS  max short-term {ST:.2f}  max momentary {M.max():.2f}  LRA {lra(x):.1f}  true peak {true_peak(x):.2f} dBTP  '
          f'sample peak {20*np.log10(np.abs(x).max()+1e-9):.2f}  DC {x.mean(1)}')


def bands(x, t0, t1):
    s = x[:, ns(t0):ns(t1)].mean(0)
    f, p = signal.welch(s, SR, nperseg=8192)
    out = []
    for a, b in ((20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)):
        out.append(10 * np.log10(p[(f >= a) & (f < b)].sum() * (f[1] - f[0]) + 1e-14))
    return np.array(out)


if __name__ == '__main__':
    path = sys.argv[1]
    x, _ = sf.read(path, always_2d=True)
    x = x.T
    levels(x, path)
