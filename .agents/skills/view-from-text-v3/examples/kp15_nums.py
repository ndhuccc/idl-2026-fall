import numpy as np, itertools
from math import comb
def k_terms(l, r): return comb(l + r, r)
def poly_features(X, r):
    n, l = X.shape; cols = []
    for deg in range(r + 1):
        for c in itertools.combinations_with_replacement(range(l), deg):
            col = np.ones(n)
            for i in c: col = col * X[:, i]
            cols.append(col)
    return np.c_[tuple(cols)]
def moons(n, noise, seed):
    rng = np.random.default_rng(seed); t = rng.uniform(0, np.pi, n)
    a = np.c_[np.cos(t), np.sin(t)]; b = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    X = np.r_[a, b] + rng.normal(0, noise, (2 * n, 2)); y = np.r_[np.ones(n), -np.ones(n)]; return X, y
def fit_acc(r, Xtr, ytr, Xte, yte):
    P = poly_features(Xtr, r); w = np.linalg.lstsq(P, ytr, rcond=None)[0]
    tr = (np.sign(P @ w) == ytr).mean(); te = (np.sign(poly_features(Xte, r) @ w) == yte).mean(); return tr, te, P.shape[1]
if __name__ == "__main__":
    print("k table", {(l, r): k_terms(l, r) for l in (2, 3, 10, 100) for r in (1, 2, 3, 5, 10) if (l, r) in {(2,1),(2,2),(2,3),(2,5),(2,10),(3,2),(10,2),(10,3),(10,10),(100,2),(100,3)}})
    print("k l=2 r=1..10", [k_terms(2, r) for r in range(1, 11)])
    k = k_terms(10, 10); print("mem Phi (1000 samples) GB", k * 1000 * 8 / 1e9, "Phi^T Phi GB", k * k * 8 / 1e9, "k", k)
    Xtr, ytr = moons(30, 0.2, 0); Xte, yte = moons(500, 0.2, 1)
    for r in range(1, 11): print(r, fit_acc(r, Xtr, ytr, Xte, yte))
    for r in range(1, 9):
        P = poly_features(Xtr, r); print("cond Phi", r, P.shape, "%.3g" % np.linalg.cond(P))
    x = np.array([1., 2.]); z = np.array([3., -1.]); phi = lambda v: np.array([1, np.sqrt(2) * v[0], np.sqrt(2) * v[1], v[0] ** 2, v[1] ** 2, np.sqrt(2) * v[0] * v[1]])
    print("kernel", (1 + x @ z) ** 2, phi(x) @ phi(z), phi(x))
    g = lambda x1, x2: -0.25 + x1 + x2 - 2 * x1 * x2
    print([g(*p) for p in ((0,0),(0,1),(1,0),(1,1),(0.5,0.5))])
