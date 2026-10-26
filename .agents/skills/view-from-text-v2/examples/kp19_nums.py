import numpy as np
from math import comb
from sklearn.svm import SVC
from kp21_nums import rings, gk
from kp20_nums import lin, quad
def svc_fit(X, y, Xs, ys, K, C):
    m = SVC(C=C, kernel="precomputed").fit(K(X, X), y)
    return (int((m.predict(K(X, X)) == y).sum()), [int((m.predict(K(A, X)) == b).sum()) for A, b in zip(Xs, ys)], len(m.support_), m)
if __name__ == "__main__":
    phi = lambda v: np.array([v[0] ** 2, np.sqrt(2) * v[0] * v[1], v[1] ** 2]); x, z = np.array([1, 2.]), np.array([2, 1.]); print("identity", phi(x) @ phi(z), (x @ z) ** 2, phi(x), phi(z))
    Xt, yt = rings(500, 2); idx = np.r_[0:100, 500:600]; print("lift 195/200:", int(((Xt[idx, 0] ** 2 + Xt[idx, 1] ** 2 > 2.5) == (yt[idx] > 0)).sum()))
    print("G2 eig", np.linalg.eigvalsh(np.array([[4., 1], [1, 4]])))
    for seed_try in (0,):
        rng = np.random.default_rng(seed_try); P = rng.standard_normal((30, 2)); print("30 pts eig min gauss(.5)", np.linalg.eigvalsh(gk(P, P, 0.5)).min(), "quad", np.linalg.eigvalsh(quad(P, P)).min())
    X, y = rings(60, 1); Xt, yt = rings(500, 2); g = lambda A, B: gk(A, B, 0.5)
    for nm, K in (("lin", lin), ("quad", quad), ("gauss", g)):
        tr, te, ns, m = svc_fit(X, y, [Xt], [yt], K, 10); print(nm, tr, te, ns)
    tr, te, ns, m = svc_fit(X, y, [Xt], [yt], g, 10); a = m.dual_coef_.ravel(); sv = X[m.support_]; dec = (a @ g(sv, Xt)) + m.intercept_[0]; print("dec diff", np.abs(dec - m.decision_function(g(Xt, X))).max())
    X, y = rings(60, 3, 0.45); Xv, yv = rings(500, 5, 0.45); Xt, yt = rings(500, 4, 0.45)
    for gm in (0.01, 0.1, 0.5, 2, 10, 50):
        tr, te, ns, m = svc_fit(X, y, [Xv, Xt], [yv, yt], lambda A, B: gk(A, B, gm), 10); print("gamma", gm, tr, te, ns)
    for C in (0.01, 0.1, 1, 10, 1000):
        tr, te, ns, m = svc_fit(X, y, [Xv, Xt], [yv, yt], lambda A, B: gk(A, B, 0.5), C); print("C", C, tr, te, ns)
    for l, q in ((2, 2), (10, 3), (100, 5), (784, 5)): print(l, q, comb(l + q, q))
    P3 = np.array([[-1, 0], [1.6, 0.9], [0.1, -0.8]]); print("tanh eig", np.linalg.eigvalsh(np.tanh(2 * P3 @ P3.T + 1)))
    rng = np.random.default_rng(5); nt = ng = 0
    for _ in range(200):
        P = rng.standard_normal((10, 2)); nt += np.linalg.eigvalsh(np.tanh(2 * P @ P.T + 1)).min() < -1e-12; ng += np.linalg.eigvalsh(gk(P, P, 0.5)).min() < -1e-12
    print("tanh neg", nt, "gauss neg", ng)
    x, z = np.array([1, 2.]), np.array([3, 1.]); ph = lambda v: np.array([1, np.sqrt(2) * v[0], np.sqrt(2) * v[1], v[0] ** 2, np.sqrt(2) * v[0] * v[1], v[1] ** 2]); print((x @ z + 1) ** 2, ph(x).round(3), ph(z).round(3), ph(x) @ ph(z))
