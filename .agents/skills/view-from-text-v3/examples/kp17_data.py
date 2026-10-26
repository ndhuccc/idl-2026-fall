"""All numbers on the KP17 pages, recomputed (functions of kp17_nums.py)."""
import numpy as np
from kp17_nums import *
from itertools import combinations_with_replacement as cwr
_cache = {}
def poly_errs(): return {r: sup_err(poly_ls(r)) for r in range(1, 12)}
def runge_curves(r=12, n=241):
    g = lambda x: 1 / (1 + 25 * x ** 2); xs = np.linspace(-1, 1, n); nodes = np.linspace(-1, 1, r + 1)
    ci = np.polynomial.chebyshev.chebfit(nodes, g(nodes), r); xd = np.linspace(-1, 1, 400); cl = np.polynomial.chebyshev.chebfit(xd, g(xd), r)
    return xs, g(xs), np.polynomial.chebyshev.chebval(xs, ci), np.polynomial.chebyshev.chebval(xs, cl)
def runge_table():
    g = lambda x: 1 / (1 + 25 * x ** 2); xs = np.linspace(-1, 1, 2001); out = []
    for r in (4, 8, 12, 16, 20):
        nodes = np.linspace(-1, 1, r + 1); ci = np.polynomial.chebyshev.chebfit(nodes, g(nodes), r); xd = np.linspace(-1, 1, 400); cl = np.polynomial.chebyshev.chebfit(xd, g(xd), r)
        out.append((r, float(np.max(np.abs(np.polynomial.chebyshev.chebval(xs, ci) - g(xs)))), float(np.max(np.abs(np.polynomial.chebyshev.chebval(xs, cl) - g(xs))))))
    return out
def mlp_curves():
    xs = np.linspace(0, 1, 201); res = {}
    for k in (1, 2, 4):
        (e, p), _ = best_mlp(k); res[k] = (e, mlp_fn(p, k, xs))
    return xs, target(xs), res
def mlp_table(): return {k: best_mlp(k)[0][0] for k in (1, 2, 3, 4, 6, 8)}
def restarts(k): return best_mlp(k)[1]
def equal_params(): return [(7, sup_err(poly_ls(6)), best_mlp(2)[0][0]), (13, sup_err(poly_ls(12)), best_mlp(4)[0][0]), (19, sup_err(poly_ls(18)), best_mlp(6)[0][0])]
def ridge():
    rng = np.random.default_rng(0); l = 5; w = rng.normal(size=l); w /= np.linalg.norm(w); Xtr = rng.normal(size=(3000, l)); Xte = rng.normal(size=(2000, l)); f = lambda X: np.tanh(2 * X @ w)
    def pf(X, r):
        cols = [np.ones(len(X))]
        for d in range(1, r + 1):
            for c in cwr(range(X.shape[1]), d): cols.append(np.prod(X[:, list(c)], axis=1))
        return np.c_[tuple(cols)]
    poly = []
    for r in (1, 2, 3, 5):
        P = pf(Xtr, r); cf = np.linalg.lstsq(P, f(Xtr), rcond=None)[0]; poly.append((r, P.shape[1], float(np.sqrt(np.mean((pf(Xte, r) @ cf - f(Xte)) ** 2)))))
    return poly
def halving(): return {l: 2 ** (l / 2) for l in (2, 5, 10, 20)}
def hat(n, m=2001):
    h = lambda x: 2 * np.minimum(x, 1 - x); x = np.linspace(0, 1, m); y = x.copy()
    for _ in range(n): y = h(y)
    return x, y
def hat_nodes(n): return 2 * n, 2 ** n - 1
if __name__ == "__main__":
    print({r: float(f"{v:.3g}") for r, v in poly_errs().items()}); print(runge_table()); print({k: round(v, 4) for k, v in mlp_table().items()})
    print(equal_params()); print(ridge()); print(halving()); print([hat_nodes(n) for n in (3, 5, 6, 10)]); print(round(restarts(2)[0], 3), [round(e, 3) for e in restarts(2)], [round(e, 3) for e in restarts(3)])
    print([ (n, 2*n, 2**n-1) for n in (3,6,10)])
