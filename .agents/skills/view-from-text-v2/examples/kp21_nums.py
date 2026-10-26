import numpy as np
from sklearn.svm import SVC
def rings(n, seed, sd=0.25):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, 2 * n); r = np.r_[rng.normal(1, sd, n), rng.normal(2, sd, n)]
    X = np.c_[r * np.cos(th), r * np.sin(th)]; y = np.r_[-np.ones(n), np.ones(n)]; return X, y
def gk(A, B, g): d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1); return np.exp(-g * d)
def svc(X, y, Xt, yt, C, g):
    m = SVC(C=C, kernel="precomputed").fit(gk(X, X, g), y); return len(m.support_), float(np.mean(m.predict(gk(Xt, X, g)) == yt))
if __name__ == "__main__":
    X, y = rings(60, 1); Xt, yt = rings(500, 2)
    print("C=10 g=0.5", svc(X, y, Xt, yt, 10, 0.5), "(19, 0.968)")
    for g in (0.01, 0.5, 50):
        print(g, [svc(X, y, Xt, yt, C, g) for C in (0.01, 1, 10, 1000)])

def part_b():
    rng = np.random.default_rng(3); X = rng.standard_normal((30, 2)); y = rng.standard_normal(30); C = 0.7
    ph = lambda X: np.c_[np.ones(len(X)), np.sqrt(2) * X[:, 0], np.sqrt(2) * X[:, 1], X[:, 0] ** 2, np.sqrt(2) * X[:, 0] * X[:, 1], X[:, 1] ** 2]
    P = ph(X); w = np.linalg.solve(P.T @ P + C * np.eye(6), P.T @ y); K = P @ P.T; a = np.linalg.solve(K + C * np.eye(30), y)
    Xn = rng.standard_normal((5, 2)); return np.abs(w - P.T @ a).max(), np.abs(ph(Xn) @ w - ph(Xn) @ P.T @ a).max()
def part_hand(C):
    K = np.array([[1., 1], [1, 2]]); y = np.array([1., -1]); a = np.linalg.solve(K + C * np.eye(2), y); return a, a[0] + a[1] * np.array([1, 2])
def sin9(seed=0):
    rng = np.random.default_rng(seed); x = np.linspace(0, 2 * np.pi, 9) + rng.normal(0, 0.05, 9); y = np.sin(x) + rng.normal(0, 0.2, 9); return x, y
def part_sin(C, s=0.8, order=0):
    x, y = sin9(); k = lambda A, B: np.exp(-(A[:, None] - B[None, :]) ** 2 / (2 * s * s)); K = k(x, x); a = np.linalg.solve(K + C * np.eye(9), y)
    xg = np.linspace(0, 2 * np.pi, 200); g = k(xg, x) @ a
    return np.mean((K @ a - y) ** 2), np.mean((g - np.sin(xg)) ** 2), int(np.sum(np.abs(a) > 1e-9)), np.abs(a).max(), np.linalg.cond(K)
def part_dict(th):
    X, y = rings(60, 1); order = np.random.default_rng(0).permutation(120); D = []
    for i in order:
        if not D or gk(X[i:i + 1], X[D], 0.5).max() <= th: D.append(i)
    return len(D)
if __name__ == "__main__":
    print("6d", part_b(), "(1.9e-14, 2.2e-14)")
    for C in (0, 1): print("hand C", C, part_hand(C))
    for C in (0, 0.01, 0.1, 1, 10, 100): print("sin C", C, [round(float(v), 4) for v in part_sin(C)])
    print("dict", [part_dict(t) for t in (0.99, 0.9, 0.7, 0.5)], "(90,38,17,9)")
