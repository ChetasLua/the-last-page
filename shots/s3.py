"""s3 The Fox: close-up, 96 frames."""
import numpy as np, cv2, math, os
from lp import W, H, foxact, comp, props
from lp.paint import rgb, f32, fbm, grid, smooth, mottle, repaint
from lp.ink import Cel, spl, over_roi
from shots import s2


class Shot:
    id = 's3'; n = 96
    finish = dict(hal_amt=.24, vig=.14)
    def setup(self):
        bg = s2.hall_bokeh().astype(f32)
        self.bg = bg
        x, y = grid()
        g = np.clip(1 - np.hypot((x - 1750) / 1150, (y - 1180) / 820), 0, 1) ** 1.7
        self.warm = g[..., None] * rgb('#ffb458')
        m = np.clip(1 - np.hypot((x - 120) / 1100, (y - 40) / 900), 0, 1) ** 1.8
        self.cool = m[..., None] * rgb('#8fa8dc')
        r = np.random.default_rng(31)
        self.motes = np.c_[r.uniform(0, W, 46), r.uniform(0, H, 46), r.uniform(1.2, 5.5, 46), r.uniform(0, 6.28, 46), r.uniform(.2, 1, 46)]

    def frame(self, f):
        fd = f - f % 2; d = f // 2
        fl = [.0, .03, -.02, .015, -.035, .02, .0, -.015][(f // 3) % 8]
        # slow truck-in on ones over the held painting
        z = 1.0 + .022 * f / 96.
        img = comp.cam2d(self.bg, W / 2 + 8, H / 2 + 4, z * 1.02)
        img = comp.screen(img, self.warm * (.40 + fl)); img = comp.screen(img, self.cool * .16)
        # ---- acting
        L = lambda a, b, v: v if a <= fd < b else 0.
        earL = 0.; earR = 0.
        for a, v in ((14, .35), (17, .9), (20, .78), (32, .4), (35, .0)):
            if fd >= a: earL = v
        for a, v in ((34, -.4), (37, -.95), (40, -.82), (56, -.35), (60, 0.)):
            if fd >= a: earR = v
        sn = 0.
        if 46 <= fd < 62: sn = [.3, 1, .25, 1, .2, .9, .3, .1][(fd - 46) // 2]
        bl = 0.
        for a, v in ((66, .5), (68, 1.), (75, .55), (77, 0.)):
            if fd >= a: bl = v
        afraid = float(comp.ease((fd - 80) / 8.)) if fd > 80 else 0.
        lookx, looky = .45, -.62
        if fd >= 24: lookx, looky = .25, -.70
        if fd >= 42: lookx, looky = .55, -.35
        if fd >= 77: lookx, looky = .40, -.95
        br = math.sin(fd * .16) * .5 + .5
        hd = dict(yaw=.36 + .02 * (fd >= 42), pitch=.10 + .05 * sn + .07 * afraid, roll=-.035 * (14 <= fd < 34) + .05 * (34 <= fd < 60) - .04 * afraid, ear_l=(earL, .30 * afraid), ear_r=(earR, .30 * afraid), blink=bl, look=(lookx, looky),
                  sniff=sn, brow=.25 + .75 * afraid, wide=.2 + .8 * afraid, pupil=1. - .06 * afraid)
        # shared drawing between pose changes: boil on fours while holding
        seed = foxact.pseed(hd['roll'], hd['pitch'], hd['yaw'], sn)
        zc = z * 1.0
        cx, cy, s = 930 - 10 * afraid, 556 + 10 * afraid, 2.42
        cx = W / 2 + (cx - W / 2) * zc; cy = H / 2 + (cy - H / 2) * zc
        tint = ((.76, .74, .95), (1.06 + fl * .5, 1.0, .92), (350, 60), (1450, 1120))
        img = foxact.closeup(img, cx, cy - 2 * br, s * zc, f, hd, tint=tint, seed=seed, breath=2.2 * br, body_seed=11)
        # the lantern itself, soft in the near foreground
        c = Cel(700, 700, 1500, 640)
        props.lantern(c, 1835 + (zc - 1) * 800, 1270 + (zc - 1) * 600, 5.4, lean=.15 * math.sin(f // 3 * 1.3), seed=f // 3 % 5, lw=4)
        lay = comp.bokeh(comp.blur(c.rgba(), 2.0), 15.5, 1.2)
        img = over_roi(img, lay, 1500, 640)
        g = comp.radial(1835, 1080, 520, power=2.2)[..., None] * rgb('#ffb04e') * (.62 + fl * 2)
        img = comp.screen(img, g)
        # lantern air glow over cel, sparse dust on threes
        img = comp.screen(img, self.warm * (.10 + fl * .5))
        t3 = f // 3
        c = Cel(W, H)
        for (x, y, r_, ph, al) in self.motes:
            xx = (x + 14 * math.sin(ph + t3 * .10) + t3 * .7) % W; yy = (y - t3 * 1.1 + 9 * math.sin(ph * 2 + t3 * .13)) % H
            warm = np.clip((xx / W) * .6 + (yy / H) * .6, 0, 1)
            col = rgb('#a9b8e0') * (1 - warm) + rgb('#ffd79a') * warm
            c.fill(spl([[xx - r_, yy], [xx, yy - r_ * .8], [xx + r_, yy + r_ * .1], [xx, yy + r_]], closed=True), col, .10 + .30 * al * (.5 + .5 * math.sin(ph + t3 * .4)))
        lay = c.rgba()
        img = comp.screen(img, lay[..., :3])
        return img
