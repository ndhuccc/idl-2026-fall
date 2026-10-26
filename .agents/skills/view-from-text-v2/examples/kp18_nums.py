import numpy as np
def unit(X): return X / np.linalg.norm(X, axis=1, keepdims=True)
def pnn(Xtr, ytr, Xte, h, prior=None, norm=True):
    if norm: Xtr, Xte = unit(Xtr), unit(Xte)
    cl = np.unique(ytr); sc = []
    for c in cl:
        P = Xtr[ytr == c]; Y = np.exp((Xte @ P.T - 1) / h ** 2); sc.append((np.mean(ytr == c) if prior is None else prior) * Y.mean(1))
    return cl[np.argmax(np.array(sc), 0)]
def dirdata(n, seed, variant=0):
    rng = np.random.default_rng(seed)
    if variant == 0:
        a = np.r_[rng.normal(40, 25, n), rng.normal(110, 25, n)]; r = rng.uniform(0.5, 3.0, 2 * n)
    elif variant == 1:
        a = np.r_[rng.normal(40, 25, n), rng.normal(110, 25, n)]; r = np.r_[rng.uniform(0.5, 3.0, n), rng.uniform(0.5, 3.0, n)]
    else:
        a1 = rng.normal(40, 25, n); r1 = rng.uniform(0.5, 3.0, n); a2 = rng.normal(110, 25, n); r2 = rng.uniform(0.5, 3.0, n); a = np.r_[a1, a2]; r = np.r_[r1, r2]
    t = np.deg2rad(a); return np.c_[r * np.cos(t), r * np.sin(t)], np.r_[np.zeros(n), np.ones(n)], a
if __name__ == "__main__":
    x = np.array([0.6, 0.8]); xi = np.array([1., 0]); print("dist2", ((x - xi) ** 2).sum(), 2 - 2 * xi @ x)
    h = 0.8
    for x in (np.array([0.6, 0.8]), np.array([1.2, 1.6])):
        d = ((x - xi) ** 2).sum(); print(d, round(np.exp(-d / (2 * h * h)), 4), round(np.exp((xi @ x - 1) / h ** 2), 4))
    th = np.deg2rad([20, 40, 60, 100, 120]); X = np.c_[np.cos(th), np.sin(th)]; xt = np.array([np.cos(np.deg2rad(70)), np.sin(np.deg2rad(70))]); y = np.exp((X @ xt - 1) / 0.25)
    print("toy", (X @ xt).round(4), y.round(4), 0.5 / 3 * y[:3].sum(), 0.5 / 2 * y[3:].sum())
    for var in (0, 1, 2):
        Xtr, ytr, a = dirdata(50, 0, var); Xte, yte, ae = dirdata(500, 1, var)
        base = np.mean((ae > 75) == (yte == 1))
        r = [(h_, round(np.mean(pnn(Xtr, ytr, Xtr, h_) == ytr) * 100), round(np.mean(pnn(Xtr, ytr, Xte, h_) == yte) * 100, 1)) for h_ in (0.02, 0.05, 0.1, 0.3, 1, 5)]
        print("variant", var, "baseline", round(base * 100, 1), r)
    # check variant 0 first
    Xte, yte, ae = dirdata(500, 1, 0); ang = np.degrees(np.arctan2(Xte[:, 1], Xte[:, 0])); print("cut at 75 (arctan2):", np.mean((ang > 75) == (yte == 1)))
    for n in (2, 5, 10, 25, 50, 200):
        acc = [np.mean(pnn(*dirdata(n, s, 0)[:2], Xte, 0.3) == yte) for s in range(100, 120)]; print("n", n, round(np.mean(acc) * 100, 1), round(np.std(acc), 3))
    rng = np.random.default_rng(0); th = rng.uniform(0, 2 * np.pi, 400); r = np.r_[rng.normal(1.0, 0.15, 200), rng.normal(2.2, 0.2, 200)]; X = np.c_[r * np.cos(th), r * np.sin(th)]; y = np.r_[np.zeros(200), np.ones(200)]; idx = rng.permutation(400)
    print("rings", np.mean(pnn(X[idx[:200]], y[idx[:200]], X[idx[200:]], 0.3) == y[idx[200:]]))
    for h in (1, 0.2):
        xA = np.array([1, 0.]); xB = np.array([0, 1.]); x = np.array([0.8, 0.6]); yA = np.exp((xA @ x - 1) / h ** 2); yB = np.exp((xB @ x - 1) / h ** 2); print("exercise h", h, yA, yB, 0.5 * yA, 0.5 * yB, yA / yB)
