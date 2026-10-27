"""KP13 numbers (recomputed)."""
import numpy as np
XA = np.array([-3., -2., 2., 3.]); XB = np.array([-1., 0., 1.])
X1 = np.r_[XA, XB]; T1 = np.r_[np.ones(4), -np.ones(3)]
def best_threshold():
    best = (0, None)
    cand = np.r_[X1 - 0.5, X1 + 0.5]
    for th in cand:
        for sgn in (1, -1):
            pred = np.where(sgn * (X1 - th) > 0, 1, -1); acc = int((pred == T1).sum())
            if acc > best[0]: best = (acc, (float(th), sgn))
    return best
def lstsq_1d():
    Phi = np.c_[np.ones(7), X1, X1 ** 2]; w = np.linalg.lstsq(Phi, T1, rcond=None)[0]; return w, Phi @ w
def rings(seed=0):
    rng = np.random.default_rng(seed); n = 100
    tA = rng.uniform(0, 2 * np.pi, n); tB = rng.uniform(0, 2 * np.pi, n); rA = rng.normal(1.0, 0.15, n); rB = rng.normal(2.2, 0.2, n)
    X = np.r_[np.c_[rA * np.cos(tA), rA * np.sin(tA)], np.c_[rB * np.cos(tB), rB * np.sin(tB)]]; t = np.r_[np.ones(n), -np.ones(n)]
    return X, t, rA, rB
def ring_fit():
    X, t, rA, rB = rings()
    out = {}
    for name, F in (("lin", np.c_[np.ones(200), X]), ("quad", np.c_[np.ones(200), X, X ** 2, X[:, 0] * X[:, 1]])):
        w = np.linalg.lstsq(F, t, rcond=None)[0]; out[name] = (int((np.sign(F @ w) == t).sum()), w)
    w = out["quad"][1]; r = float(np.sqrt(-w[0] / ((w[3] + w[4]) / 2)))
    return out, r, float(rA.mean()), float(rB.mean())
def multi():
    x = np.array([-3, -2.5, -2, -0.5, 0, 0.5, 2, 2.5, 3.]); cls = np.repeat([0, 1, 2], 3); T = np.eye(3)[cls]
    res = {}
    for name, F in (("quad", np.c_[np.ones(9), x, x ** 2]), ("lin", np.c_[np.ones(9), x])):
        W = np.linalg.lstsq(F, T, rcond=None)[0]; G = F @ W; res[name] = (int((G.argmax(1) == cls).sum()), W, G)
    return x, cls, res
def sig(v): return 1 / (1 + np.exp(-v))
if __name__ == "__main__":
    print(best_threshold(), lstsq_1d())
    print(ring_fit()); print(multi()[2]["quad"][0], multi()[2]["lin"][0]); print(multi()[2]["lin"][2][4])
    print(sig(2), 2 * sig(1), [l * (l + 3) // 2 for l in (2, 10, 100)])
