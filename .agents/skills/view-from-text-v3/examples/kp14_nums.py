"""KP14 numbers (recomputed): Cover counting, probabilities, Monte-Carlo separability by LP, lift, generalisation."""
import numpy as np
from math import comb
from scipy.optimize import linprog
def O(N, l): return 2 * sum(comb(N - 1, i) for i in range(l + 1))
def P(N, l): return O(N, l) / 2 ** N
def separable(X, y):
    n, d = X.shape; A = -(y[:, None] * np.c_[X, np.ones(n)])
    r = linprog(np.zeros(d + 1), A_ub=A, b_ub=-np.ones(n), bounds=[(None, None)] * (d + 1), method="highs"); return r.status == 0
def quad(X): return np.c_[X, X ** 2, X[:, 0] * X[:, 1]]
def experiments():
    rng = np.random.default_rng(1); out = {}
    for N in (4, 6):
        c = 0
        for _ in range(1500):
            X = rng.standard_normal((N, 2)); y = rng.choice([-1., 1.], N); c += separable(X, y)
        out[N] = c / 1500
    c0 = c1 = 0
    for _ in range(1500):
        X = rng.standard_normal((6, 2)); y = rng.choice([-1., 1.], 6); c0 += separable(X, y); c1 += separable(quad(X), y)
    out["lift"] = (c0 / 1500, c1 / 1500)
    ok = 0; acc = []; tr = []
    for _ in range(300):
        X = rng.standard_normal((5, 2)); y = rng.choice([-1., 1.], 5); F = quad(X)
        A = -(y[:, None] * np.c_[F, np.ones(5)]); r = linprog(np.zeros(6), A_ub=A, b_ub=-np.ones(5), bounds=[(None, None)] * 6, method="highs")
        if r.status != 0: continue
        ok += 1; w = r.x; tr.append(float(np.mean(np.sign(np.c_[F, np.ones(5)] @ w) == y)))
        Xn = rng.standard_normal((200, 2)); yn = rng.choice([-1., 1.], 200); acc.append(float(np.mean(np.sign(np.c_[quad(Xn), np.ones(200)] @ w) == yn)))
    out["gen"] = (ok, float(np.mean(tr)), float(np.mean(acc)))
    return out
if __name__ == "__main__":
    print([O(N, 2) for N in range(1, 9)], [round(P(N, 2), 4) for N in range(1, 11)])
    print(experiments())
    print([(l, round(P(round(1.5 * (l + 1)), l), 2) if (1.5 * (l + 1)) % 1 == 0 else None) for l in (1, 19, 99)])
    print([(l, round(P(100, l), 4)) for l in (40, 48, 49, 56, 60, 68)], [O(2 * l + 2, l) for l in range(1, 6)], O(5, 2), P(5, 2), O(5, 3), P(5, 3))
