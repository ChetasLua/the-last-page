"""python tools/look.py out.jpg s1:40 s2:80 ...  contact sheet of finished frames (2 per row, 960 wide)"""
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, cv2
cv2.setNumThreads(4)
from lp import comp
out = sys.argv[1]; cache = {}; ims = []
W2 = 960
for t in sys.argv[2:]:
    k, f = t.split(':'); f = int(f)
    if k not in cache:
        S = importlib.import_module('shots.' + k).Shot(); S.setup(); cache[k] = S
    S = cache[k]; im = comp.to8(comp.finish(S.frame(f), f, **getattr(S, 'finish', {})), f)
    ims.append(cv2.resize(im, (W2, W2 * 9 // 16), interpolation=cv2.INTER_AREA))
n = 2
while len(ims) % n: ims.append(np.zeros_like(ims[0]))
comp.save(out, np.vstack([np.hstack(ims[i:i + n]) for i in range(0, len(ims), n)]))
