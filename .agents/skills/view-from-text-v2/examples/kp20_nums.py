import numpy as np
from kp21_nums import rings, gk
def lin(A, B): return A @ B.T
def quad(A, B): return (A @ B.T + 1) ** 2
def nearest_mean(X, y, Xt, K, theta_on=True):
    """class 1 = +1 (outer), class 2 = -1 (inner). Returns predicted labels and (mean K inside class1, class2)."""
    c1, c2 = X[y > 0], X[y < 0]; s2 = K(Xt, c2).mean(1); s1 = K(Xt, c1).mean(1)
    m1, m2 = K(c1, c1).mean(), K(c2, c2).mean(); theta = (m2 - m1) / 2
    lhs = s2 - s1; pred = np.where(lhs > (theta if theta_on else 0), -1, 1); return pred, m1, m2
def kperc(X, y, K, max_epochs=500):
    G = K(X, X); a = np.zeros(len(X)); errs = []
    for ep in range(max_epochs):
        e = 0
        for i in range(len(X)):
            g = (a * y) @ G[:, i] + (a * y).sum()
            if y[i] * g <= 0: a[i] += 1; e += 1
        errs.append(e)
        if e == 0: break
    return a, errs
def kpred(X, y, a, Xt, K): return np.sign((a * y) @ K(X, Xt) + (a * y).sum())
if __name__ == "__main__":
    X, y = rings(60, 1); Xt, yt = rings(500, 2)
    g = lambda A, B: gk(A, B, 0.5)
    for nm, K in (("linear", lin), ("quad", quad), ("gauss", g)):
        p, m1, m2 = nearest_mean(X, y, Xt, K); print(nm, int((p == yt).sum()), round(m1, 3), round(m2, 3))
    p, m1, m2 = nearest_mean(X, y, Xt, g, False); print("gauss no theta", int((p == yt).sum()), "2theta", round(m2 - m1, 3))
    # hand trace
    P = np.array([[1, 0], [-1, 0], [0, 1], [0, -1.]]); yy = np.array([1, 1, -1, -1.]); a, errs = kperc(P, yy, quad); print("hand", a, errs)
    for sd in (0.15, 0.25):
        X, y = rings(60, 1, sd); Xt, yt = rings(500, 2, sd)
        a, errs = kperc(X, y, g); p = kpred(X, y, a, Xt, g); print("kperc sd", sd, "epochs", len(errs), "alpha>0", int((a > 0).sum()), "correct", int((p == yt).sum()), "first errs", errs[:6])
        a, errs = kperc(X, y, lin, 500); p = kpred(X, y, a, Xt, lin); print(" linear perc epochs", len(errs), "last err", errs[-1], "correct", int((p == yt).sum()))
    X, y = rings(60, 1, 0.25)
    for gm in (0.1, 2, 8):
        a, errs = kperc(X, y, lambda A, B: gk(A, B, gm)); print("gamma", gm, "epochs", len(errs))
    z = np.array([-2., 0, 2]); print("hinge", np.maximum(0, 1 - z), "exp", np.exp(-z).round(3), "logit", np.log1p(np.exp(-z)).round(3), np.log1p(np.exp(2)) / np.log(2), np.log1p(np.exp(1)) / np.log(2), np.log1p(np.exp(1)), np.exp(2) / 3)
