import numpy as np
from scipy.optimize import minimize
sig = lambda v: 1 / (1 + np.exp(-v))
def target(x): return np.sin(2 * np.pi * x) + 0.3 * x ** 2
XG = np.linspace(0, 1, 401)
def poly_ls(r, x=None, y=None):
    x = XG if x is None else x; y = target(x) if y is None else y
    c = np.polynomial.chebyshev.chebfit(2 * x - 1, y, r); return lambda t: np.polynomial.chebyshev.chebval(2 * t - 1, c)
def sup_err(f, g=target, x=XG): return float(np.max(np.abs(f(x) - g(x))))
def mlp_unpack(p, k): return p[:k], p[k:2 * k], p[2 * k:3 * k], p[3 * k]   # a, b, w_out, c0
def mlp_fn(p, k, x): a, b, w, c = mlp_unpack(p, k); return c + (w[None, :] * sig(x[:, None] * a[None, :] + b[None, :])).sum(1)
def fit_mlp(k, x, y, seed, maxiter=3000):
    rng = np.random.default_rng(seed); p0 = np.r_[rng.normal(0, 8, k), rng.normal(0, 4, k), rng.normal(0, 1, k), 0.0]
    def loss(p):
        a, b, w, c = mlp_unpack(p, k); z = x[:, None] * a[None, :] + b[None, :]; h = sig(z); pred = c + h @ w; e = pred - y
        L = 0.5 * np.mean(e ** 2); dh = (e[:, None] * w[None, :]) * h * (1 - h)
        return L, np.r_[(dh * x[:, None]).mean(0), dh.mean(0), (e[:, None] * h).mean(0), e.mean()]
    r = minimize(loss, p0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter)); return r.x, r.fun
def best_mlp(k, restarts=8, x=None, y=None):
    x = XG if x is None else x; y = target(x) if y is None else y; best = None; res = []
    for s in range(restarts):
        p, L = fit_mlp(k, x, y, s); e = float(np.max(np.abs(mlp_fn(p, k, x) - y))); res.append(e)
        if best is None or e < best[0]: best = (e, p)
    return best, res
if __name__ == "__main__":
    print("poly LS sup err", {r: round(sup_err(poly_ls(r)), 4) for r in (1, 2, 3, 4, 5, 6, 8, 10, 12)})
    runge = lambda x: 1 / (1 + 25 * x ** 2); xs = np.linspace(-1, 1, 2001)
    for r in (4, 8, 12, 16, 20):
        nodes = np.linspace(-1, 1, r + 1); c = np.polynomial.chebyshev.chebfit(nodes, runge(nodes), r); fi = lambda t: np.polynomial.chebyshev.chebval(t, c)
        xd = np.linspace(-1, 1, 400); cl = np.polynomial.chebyshev.chebfit(xd, runge(xd), r); fl = lambda t: np.polynomial.chebyshev.chebval(t, cl)
        print("runge r", r, "equispaced interp sup", round(float(np.max(np.abs(fi(xs) - runge(xs)))), 3), "LS dense sup", round(float(np.max(np.abs(fl(xs) - runge(xs)))), 3))
    for k in (1, 2, 3, 4, 6, 8):
        (e, p), res = best_mlp(k); print("mlp k", k, "params", 3 * k + 1, "best sup err", round(e, 4), "all", np.round(res, 3).tolist())
    # equal-params: poly r params r+1
    for k in (2, 4, 6): print("params", 3 * k + 1, "poly degree", 3 * k, "sup", round(sup_err(poly_ls(3 * k)), 6))
    # ridge in l dims
    rng = np.random.default_rng(0)
    for l in (5,):
        w = rng.normal(size=l); w /= np.linalg.norm(w); Xtr = rng.normal(size=(3000, l)); Xte = rng.normal(size=(2000, l)); f = lambda X: np.tanh(2 * X @ w)
        from itertools import combinations_with_replacement as cwr
        def pf(X, r):
            cols = [np.ones(len(X))]
            for d in range(1, r + 1):
                for c in cwr(range(X.shape[1]), d): cols.append(np.prod(X[:, list(c)], axis=1))
            return np.c_[tuple(cols)]
        for r in (1, 2, 3, 5):
            P = pf(Xtr, r); cf = np.linalg.lstsq(P, f(Xtr), rcond=None)[0]; print("ridge l", l, "poly r", r, "terms", P.shape[1], "test rmse", round(float(np.sqrt(np.mean((pf(Xte, r) @ cf - f(Xte)) ** 2))), 4))
        for k in (1, 2, 4):
            best = None
            for s in range(5):
                r2 = np.random.default_rng(s); p0 = r2.normal(0, 0.5, k * l + k + k + 1)
                def loss(p):
                    W = p[:k * l].reshape(k, l); b = p[k * l:k * l + k]; v = p[k * l + k:k * l + 2 * k]; c = p[-1]; H = np.tanh(Xtr @ W.T + b); e = H @ v + c - f(Xtr); L = 0.5 * np.mean(e ** 2)
                    dH = (e[:, None] * v[None, :]) * (1 - H ** 2); gW = (dH.T @ Xtr) / len(Xtr); return L, np.r_[gW.ravel(), dH.mean(0), (e[:, None] * H).mean(0), e.mean()]
                rr = minimize(loss, p0, jac=True, method="L-BFGS-B", options=dict(maxiter=500)); W = rr.x[:k * l].reshape(k, l); b = rr.x[k * l:k * l + k]; v = rr.x[k * l + k:k * l + 2 * k]; c = rr.x[-1]
                te = float(np.sqrt(np.mean((np.tanh(Xte @ W.T + b) @ v + c - f(Xte)) ** 2)))
                best = te if best is None else min(best, te)
            print("ridge l", l, "mlp k", k, "params", k * l + 2 * k + 1, "test rmse", round(best, 4))
    print("halving factor 2^(l/2):", {l: 2 ** (l / 2) for l in (2, 5, 10, 20)})
    h = lambda x: 2 * np.minimum(x, 1 - x); x = np.linspace(0, 1, 100001)
    for n in (1, 2, 3, 4):
        y = x.copy()
        for _ in range(n): y = h(y)
        d = np.diff(y) / np.diff(x); br = int((np.abs(np.diff(np.sign(np.round(d)))) > 0).sum()); print("hat^n n", n, "pieces", br + 1)
