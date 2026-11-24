"""python tools/still.py s1 40 [80 ...] [--crop x,y[,w,h]] [--sheet] [--raw] [--scale .5]
Renders finished stills of a shot to qa/<shot>/.  Crops are 1:1 600x450."""
import sys, os, importlib, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, cv2
cv2.setNumThreads(4)
from lp import comp
ap = argparse.ArgumentParser(); ap.add_argument('shot'); ap.add_argument('frames', nargs='+', type=int)
ap.add_argument('--crop', action='append', default=[]); ap.add_argument('--sheet', action='store_true'); ap.add_argument('--raw', action='store_true')
ap.add_argument('--scale', type=float, default=1.0); ap.add_argument('--out', default=None)
a = ap.parse_args()
m = importlib.import_module('shots.' + a.shot); S = m.Shot(); t = time.time(); S.setup(); print('setup %.1fs' % (time.time() - t))
d = a.out or f'qa/{a.shot}'; os.makedirs(d, exist_ok=True); ims = []
for f in a.frames:
    t = time.time(); im = S.frame(f); t1 = time.time() - t
    if not a.raw: im = comp.finish(im, f, **getattr(S, 'finish', {}))
    im8 = comp.to8(np.clip(im, 0, 1), f); print(f, '%.2fs' % t1, im8.shape)
    if a.sheet: ims.append(cv2.resize(im8, (640, 360), interpolation=cv2.INTER_AREA))
    else:
        o = im8 if a.scale == 1 else cv2.resize(im8, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        comp.save(f'{d}/f{f:04d}.jpg', o)
    for k, c in enumerate(a.crop):
        v = [int(q) for q in c.split(',')]; w, h = (v[2], v[3]) if len(v) > 2 else (600, 450)
        comp.save(f'{d}/f{f:04d}_c{k}.jpg', im8[v[1]:v[1] + h, v[0]:v[0] + w])
if a.sheet:
    n = 3; rows = [np.hstack((ims[i:i + n] + [np.zeros_like(ims[0])] * n)[:n]) for i in range(0, len(ims), n)]
    comp.save(f'{d}/sheet.jpg', np.vstack(rows)); print(f'{d}/sheet.jpg')
