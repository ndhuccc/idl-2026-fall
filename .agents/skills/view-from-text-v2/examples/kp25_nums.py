import numpy as np
def alpha(P): return 0.5 * np.log((1 - P) / P)
def gauss20(n, seed, a=None, flip=0.0):
    a = 2 / np.sqrt(20) if a is None else a; r = np.random.default_rng(seed); y = np.r_[-np.ones(n // 2), np.ones(n - n // 2)]; X = r.normal(0, 1, (n, 20)) + y[:, None] * a
    yl = y.copy()
    if flip > 0: m = r.random(n) < flip; yl[m] = -yl[m]
    return X, yl, y
def best_stump(X, y, w):
    N, d = X.shape; best = (1.0, 0, 0.0, 1)
    for j in range(d):
        o = np.argsort(X[:, j]); xs = X[o, j]; ys = y[o]; ws = w[o]
        # predict +1 for x > thr: error = weight of (+1 labels at or below thr) + weight of (-1 labels above thr)
        cp = np.cumsum(ws * (ys == 1)); cn = np.cumsum(ws * (ys == -1)); tot_n = cn[-1]
        err = cp + (tot_n - cn)                      # threshold after position i
        i = int(np.argmin(np.minimum(err, 1 - err)))
        e = err[i]; sgn = 1 if e <= 1 - e else -1; e = min(e, 1 - e)
        if e < best[0]: thr = (xs[i] + xs[min(i + 1, N - 1)]) / 2; best = (e, j, thr, sgn)
    return best
def adaboost(X, y, K, seed=0, track=None, Xt=None, yt=None, stumps="random"):
    N, d = X.shape; w = np.ones(N) / N; F = np.zeros(N); Ft = np.zeros(len(Xt)) if Xt is not None else None; rng = np.random.default_rng(seed); out = []; als = []; Ps = []
    for m in range(1, K + 1):
        if stumps == "best": P, j, thr, s = best_stump(X, y, w); h = np.where(X[:, j] > thr, 1, -1); hm = s * h
        else:
            j = rng.integers(0, d); h = np.sign(X[:, j]); h[h == 0] = 1; P = w[h != y].sum(); s = 1
            if P > 0.5: s = -1; P = 1 - P
            hm = s * h
        P = min(max(P, 1e-12), 1 - 1e-12); a = alpha(P)
        w = w * np.exp(-y * a * hm); w /= w.sum(); F += a * hm; als.append(a); Ps.append(P)
        if Xt is not None:
            ht = np.where(Xt[:, j] > thr, 1, -1) if stumps == 'best' else np.where(Xt[:, j] > 0, 1, -1); Ft += a * s * ht
        if track is not None and m in track: out.append((m, float((np.sign(F) != y).mean()), float((np.sign(Ft) != yt).mean()) if Xt is not None else None, F.copy(), w.copy()))
    return out, np.array(als), np.array(Ps)
if __name__ == "__main__":
    print([round(alpha(p), 4) for p in (0.1, 0.3, 0.5, 0.7)])
    a = alpha(0.3); wm = 0.1 * np.exp(a); wc = 0.1 * np.exp(-a); Z = 3 * wm + 7 * wc; print("hand a", a, wm, wc, Z, wm / Z, wc / Z, 3 * wm / Z)
    print("practice P=.2", alpha(0.2), 0.2 * 2, 0.8 * 0.5 / 4 * 4 / 4)
    X, y, _ = gauss20(200, 1); Xt, yt, _ = gauss20(400, 2)
    tr = [1, 10, 50, 100, 200, 500, 1000, 2000]
    out, als, Ps = adaboost(X, y, 2000, 0, tr, Xt, yt, stumps="best")
    for m, e1, e2, F, w in out: 
        mg = y * F / als[:m].sum(); print("round", m, "train err", e1, "test err", e2, "frac margin<0.1", (mg < 0.1).mean(), "min margin", mg.min())
    print("alpha first", als[:5], "P first", Ps[:5], "mean P", Ps.mean(), "single stump test err", np.mean([ (np.sign(Xt[:, j]) != yt).mean() for j in range(20)]))
    # first round train-zero
    out, als, Ps = adaboost(X, y, 2000, 0, list(range(1, 2001)), Xt, yt, stumps="best"); z = [m for m, e1, *_ in out if e1 == 0]; print("train zero first at", z[0] if z else None, "test at then", [e2 for m, e1, e2, *_ in out if m == (z[0] if z else 0)], "test min", min(e2 for _, _, e2, *_ in out), "at", min(out, key=lambda t: t[2])[0])
    # noisy labels
    for flip in (0.0, 0.15):
        res = []
        for s in range(5):
            X, yl, y0 = gauss20(200, 10 + s, flip=flip); Xt, ytl, yt0 = gauss20(400, 50 + s)
            o, al, Pp = adaboost(X, yl, 2000, s, [10, 100, 500, 2000], Xt, yt0, stumps="best")
            res.append([(e1, e2) for _, e1, e2, *_ in o])
            if s == 0 and flip > 0:
                m_ = X.shape[0]; noisy = yl != y0; wfin = o[-1][4]; print("weight on noisy labels after 2000:", wfin[noisy].sum(), "share of samples", noisy.mean(), "max weight", wfin.max(), "mean", 1 / m_)
        res = np.array(res); print("flip", flip, "mean (train,test) at 10,100,500,2000:", res.mean(0).round(3).tolist())
