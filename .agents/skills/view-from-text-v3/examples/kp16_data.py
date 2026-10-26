"""Numbers on the KP16 pages (recomputed)."""
import numpy as np
from kp16_nums import *
def xor_table():
    C = np.array([[1., 1], [0, 0]]); X = np.array([[0, 0.], [1, 1], [1, 0], [0, 1]]); Y = rbf_feats(X, C, 0.5); return X, Y, Y.sum(1) - 1
def xor_curve():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xs = np.linspace(-1.2, 2.2, 341); XX, YY = np.meshgrid(xs, xs); g = np.exp(-((XX - 1) ** 2 + (YY - 1) ** 2)) + np.exp(-(XX ** 2 + YY ** 2)) - 1
    cs = plt.contour(XX, YY, g, [0]); segs = cs.allsegs[0]; plt.close("all"); return max(segs, key=len)
def rings_k():
    Xtr, ytr = rings(100, 0); Xte, yte = rings(500, 1); out = {}
    for k in (1, 2, 4, 8, 16, 32):
        res = []
        for seed in range(20):
            rng = np.random.default_rng(seed); C = Xtr[rng.choice(len(Xtr), k, replace=False)]; res.append(fit_acc(C, 0.5, Xtr, ytr, Xte, yte)[1])
        out[k] = float(np.mean(res))
    rng = np.random.default_rng(0); C8 = Xtr[rng.choice(len(Xtr), 8, replace=False)]
    return Xtr, ytr, C8, out
def moons_cmp():
    Xtr, ytr = moons(60, 0.2, 0); Xte, yte = moons(500, 0.2, 1); out = {}
    for k in (2, 4, 6, 8, 12):
        rr, kk = [], []
        for seed in range(20):
            rng = np.random.default_rng(seed); C = Xtr[rng.choice(len(Xtr), k, replace=False)]; rr.append(fit_acc(C, 0.3, Xtr, ytr, Xte, yte)[1]); kk.append(fit_acc(kmeans(Xtr, k, seed), 0.3, Xtr, ytr, Xte, yte)[1])
        out[k] = (float(np.mean(rr)), float(np.mean(kk)), float(np.std(rr)), float(np.std(kk)))
    return Xtr, ytr, kmeans(Xtr, 8, 0), out
def coverage():
    rng = np.random.default_rng(0); out = {}
    for l in (1, 2, 5, 10, 20):
        Cc = rng.uniform(0, 1, (100, l)); T = rng.uniform(0, 1, (2000, l)); d = np.sqrt(((T[:, None] - Cc[None]) ** 2).sum(-1)).min(1)
        out[l] = (float(np.median(d)), float((np.exp(-d ** 2 / (2 * 0.09)) > 0.1).mean()))
    return out
def practice():
    C = np.array([[1., 1], [0, 0]]); Y = rbf_feats(np.array([[0.5, 0.5], [3, 3]]), C, 0.5); return Y, Y.sum(1) - 1
if __name__ == "__main__":
    print(xor_table()); print(len(xor_curve())); print({k: round(v, 3) for k, v in rings_k()[3].items()}); print({k: tuple(round(x, 3) for x in v) for k, v in moons_cmp()[3].items()}); print({l: (round(a, 3), b) for l, (a, b) in coverage().items()}); print(practice())
