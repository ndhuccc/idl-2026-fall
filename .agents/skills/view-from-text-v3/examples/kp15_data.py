"""Numbers and curves on the KP15 pages (recomputed)."""
import numpy as np
from math import comb
from kp15_nums import *
def sweep():
    Xtr, ytr = moons(30, 0.2, 0); Xte, yte = moons(500, 0.2, 1); out = {}
    for r in range(1, 11):
        tr_, te_, k = fit_acc(r, Xtr, ytr, Xte, yte); out[r] = (k, float(tr_), float(te_), float(np.linalg.cond(poly_features(Xtr, r))))
    return Xtr, ytr, out
def boundary(r, lo=(-1.6, -1.1), hi=(2.6, 1.6), n=300):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Xtr, ytr = moons(30, 0.2, 0); w = np.linalg.lstsq(poly_features(Xtr, r), ytr, rcond=None)[0]
    xs = np.linspace(lo[0], hi[0], n); ys = np.linspace(lo[1], hi[1], n); XX, YY = np.meshgrid(xs, ys)
    G = (poly_features(np.c_[XX.ravel(), YY.ravel()], r) @ w).reshape(XX.shape); cs = plt.contour(XX, YY, G, [0]); segs = [s for s in cs.allsegs[0] if len(s) > 5]; plt.close("all"); return segs
def hyper_regions(w=0.9):
    """B (g<0) corner regions of g = 1/4 - 2uv (u = x1-.5, v = x2-.5) inside |u|,|v| <= w, as point lists (x1, x2, smooth)."""
    us = np.linspace(1 / (8 * w), w, 14); up = [(0.5 + u, 0.5 + 1 / (8 * u), True) for u in us]
    ur = up + [(0.5 + w, 0.5 + w, False)]
    ll = [(1 - x, 1 - y, s) for x, y, s in ur]
    return ur, ll
def memory(): k = k_terms(10, 10); return k, k * 1000 * 8 / 1e9, k * k * 8 / 1e9
def kernel_check():
    x = np.array([1., 2.]); z = np.array([3., -1.]); phi = lambda v: np.array([1, np.sqrt(2) * v[0], np.sqrt(2) * v[1], v[0] ** 2, v[1] ** 2, np.sqrt(2) * v[0] * v[1]])
    return (1 + x @ z) ** 2, float(phi(x) @ phi(z)), phi(x), phi(z)
if __name__ == "__main__":
    Xtr, ytr, o = sweep(); print({r: (k, round(a, 3), round(b, 3), "%.3g" % c) for r, (k, a, b, c) in o.items()}); print(memory(), kernel_check()[:2], [len(s) for s in boundary(3)], [len(s) for s in boundary(1)])
    print([k_terms(l, r) for l in (2, 10, 100) for r in (2, 3, 10)]); print(k_terms(4, 3))
