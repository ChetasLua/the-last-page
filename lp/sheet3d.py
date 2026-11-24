"""Curved sheets of paper in a Camera3D: mesh -> per-pixel (u, v, depth, light).  Owner: ink lane."""
import numpy as np, cv2, math
from numba import njit
from .paint import f32


@njit(cache=True, fastmath=True)
def _tris(V, A, F, out, zb, w, h):
    """V[n,3] = screen x, y, 1/z;  A[n,c] attributes;  F[m,3] indices"""
    C = A.shape[1]
    for t in range(F.shape[0]):
        a, b, c = F[t, 0], F[t, 1], F[t, 2]
        x0, y0, z0 = V[a, 0], V[a, 1], V[a, 2]
        x1, y1, z1 = V[b, 0], V[b, 1], V[b, 2]
        x2, y2, z2 = V[c, 0], V[c, 1], V[c, 2]
        if z0 <= 0 or z1 <= 0 or z2 <= 0: continue
        ar = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(ar) < 1e-9: continue
        ix0 = max(0, int(math.floor(min(x0, x1, x2)))); ix1 = min(w - 1, int(math.ceil(max(x0, x1, x2))))
        iy0 = max(0, int(math.floor(min(y0, y1, y2)))); iy1 = min(h - 1, int(math.ceil(max(y0, y1, y2))))
        for y in range(iy0, iy1 + 1):
            for x in range(ix0, ix1 + 1):
                w0 = ((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)) / ar
                w1 = ((x2 - x) * (y0 - y) - (x0 - x) * (y2 - y)) / ar
                w2 = 1 - w0 - w1
                if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6: continue
                iz = w0 * z0 + w1 * z1 + w2 * z2
                if iz <= zb[y, x]: continue
                zb[y, x] = iz
                for k in range(C):
                    out[y, x, k] = (w0 * z0 * A[a, k] + w1 * z1 * A[b, k] + w2 * z2 * A[c, k]) / iz


def grid_faces(nu, nv):
    i = np.arange(nu * nv).reshape(nv, nu)
    a, b, c, d = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, 1:].ravel(), i[1:, :-1].ravel()
    return np.vstack([np.c_[a, b, c], np.c_[a, c, d]]).astype(np.int64)


def raster(cam, P, attrs, w=None, h=None, ss=1, ox=0, oy=0, out=None, zb=None):
    """P: (nv, nu, 3) world grid, attrs: (nv, nu, c).  ROI w x h at (ox, oy) screen px, ss x supersampled.
    returns attr image (h*ss, w*ss, c) and 1/z buffer (0 = empty)"""
    w = cam.w if w is None else w; h = cam.h if h is None else h
    nv, nu = P.shape[:2]
    s, z = cam.project(P.reshape(-1, 3))
    V = np.c_[(s[:, 0] - ox + .5) * ss - .5, (s[:, 1] - oy + .5) * ss - .5, np.where(z > .05, 1 / np.maximum(z, .05), -1)]
    A = np.asarray(attrs, float).reshape(nv * nu, -1)
    if out is None:
        out = np.zeros((h * ss, w * ss, A.shape[1]), f32); zb = np.zeros((h * ss, w * ss), f32)
    _tris(V.astype(np.float64), A.astype(np.float64), grid_faces(nu, nv), out, zb, w * ss, h * ss)
    return out, zb


def normals(P):
    du = np.gradient(P, axis=1); dv = np.gradient(P, axis=0)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    return np.where(n[..., 2:3] < 0, -n, n)


def tex(img, uv, interp=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, val=0):
    """sample a texture at per-pixel texture px uv"""
    return cv2.remap(img, uv[..., 0].astype(f32), uv[..., 1].astype(f32), interp, borderMode=border, borderValue=val)


def down(img, ss):
    if ss == 1: return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w // ss, h // ss), interpolation=cv2.INTER_AREA)


def soften_edges(img, lab, s=.85):
    """anti-alias the label-masked paint edges only; the strokes themselves stay crisp"""
    e = np.zeros(lab.shape, np.uint8)
    d = lab[:, 1:] != lab[:, :-1]; e[:, 1:] |= d; e[:, :-1] |= d
    d = lab[1:] != lab[:-1]; e[1:] |= d; e[:-1] |= d
    m = np.clip(cv2.GaussianBlur(cv2.dilate(e, np.ones((3, 3), np.uint8)).astype(f32), (0, 0), .8), 0, 1)[..., None]
    return img * (1 - m) + cv2.GaussianBlur(img, (0, 0), s) * m
