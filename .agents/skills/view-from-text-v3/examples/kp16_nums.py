import numpy as np
def rbf_feats(X, C, s2): d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1); return np.exp(-d2 / (2 * s2))
def design(X, C, s2): return np.c_[np.ones(len(X)), rbf_feats(X, C, s2)]
def moons(n, noise, seed):
    rng = np.random.default_rng(seed); t = rng.uniform(0, np.pi, n)
    a = np.c_[np.cos(t), np.sin(t)]; b = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    X = np.r_[a, b] + rng.normal(0, noise, (2 * n, 2)); y = np.r_[np.ones(n), -np.ones(n)]; return X, y
def rings(n, seed):
    rng = np.random.default_rng(seed); th = rng.uniform(0, 2 * np.pi, 2 * n); r = np.r_[rng.normal(1.0, 0.15, n), rng.normal(2.2, 0.2, n)]
    return np.c_[r * np.cos(th), r * np.sin(th)], np.r_[np.ones(n), -np.ones(n)]
def kmeans(X, k, seed, it=50):
    rng = np.random.default_rng(seed); C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(it):
        a = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (a == j).any(): C[j] = X[a == j].mean(0)
    return C
def fit_acc(C, s2, Xtr, ytr, Xte, yte):
    P = design(Xtr, C, s2); w = np.linalg.lstsq(P, ytr, rcond=None)[0]
    return (np.sign(P @ w) == ytr).mean(), (np.sign(design(Xte, C, s2) @ w) == yte).mean()
if __name__ == "__main__":
    C = np.array([[1., 1], [0, 0]]); X = np.array([[0, 0.], [1, 1], [1, 0], [0, 1]]); Y = rbf_feats(X, C, 0.5); print("xor map", Y.round(3), "g", (Y.sum(1) - 1).round(3))
    print("response sigma2=0.5 at d=0,1,2,3:", [round(float(np.exp(-d * d / 1.0)), 6) for d in (0, 1, 2, 3)])
    print("practice", rbf_feats(np.array([[0.5, 0.5], [3, 3]]), C, 0.5), (rbf_feats(np.array([[0.5, 0.5], [3, 3]]), C, 0.5).sum(1) - 1))
    # rings: random centers from training
    Xtr, ytr = rings(100, 0); Xte, yte = rings(500, 1)
    for k in (1, 2, 4, 8, 16, 32):
        res = []
        for seed in range(20):
            rng = np.random.default_rng(seed); C = Xtr[rng.choice(len(Xtr), k, replace=False)]; res.append(fit_acc(C, 0.5, Xtr, ytr, Xte, yte))
        print("rings random k", k, np.mean(res, 0).round(3))
    # moons: random vs kmeans
    Xtr, ytr = moons(60, 0.2, 0); Xte, yte = moons(500, 0.2, 1)
    for k in (2, 4, 6, 8, 12):
        rr, kk = [], []
        for seed in range(20):
            rng = np.random.default_rng(seed); C = Xtr[rng.choice(len(Xtr), k, replace=False)]; rr.append(fit_acc(C, 0.3, Xtr, ytr, Xte, yte)[1]); kk.append(fit_acc(kmeans(Xtr, k, seed), 0.3, Xtr, ytr, Xte, yte)[1])
        print("moons k", k, "random test", round(float(np.mean(rr)), 3), "kmeans test", round(float(np.mean(kk)), 3), "sd", round(float(np.std(rr)), 3), round(float(np.std(kk)), 3))
    # coverage
    rng = np.random.default_rng(0)
    for l in (1, 2, 5, 10, 20):
        Cc = rng.uniform(0, 1, (100, l)); T = rng.uniform(0, 1, (2000, l)); d = np.sqrt(((T[:, None] - Cc[None]) ** 2).sum(-1)).min(1)
        print("l", l, "median nearest dist", round(float(np.median(d)), 3), "response s=0.3", float(np.exp(-np.median(d) ** 2 / (2 * 0.09))), "frac resp>0.1", float((np.exp(-d ** 2 / (2 * 0.09)) > 0.1).mean()))
    print("grid centers m=5:", {l: 5 ** l for l in (2, 5, 10)})
