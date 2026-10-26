import numpy as np
def trend(n=20, seed=0, noise=0.3, out=True):
    rng = np.random.default_rng(seed); x = np.sort(rng.uniform(0, 4, n)); y = 2 * x + 1 + rng.normal(0, noise, n)
    if out: x = np.r_[x, 5.0]; y = np.r_[y, 2 * 5 + 1 + 10]
    return x, y
def ols(x, y): A = np.c_[x, np.ones(len(x))]; return np.linalg.lstsq(A, y, rcond=None)[0]
def huber_irls(x, y, c=1.0, it=100):
    A = np.c_[x, np.ones(len(x))]; w = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(it):
        e = y - A @ w; wt = np.where(np.abs(e) <= c, 1.0, c / np.maximum(np.abs(e), 1e-12)); W = np.diag(wt); w = np.linalg.solve(A.T @ W @ A, A.T @ W @ y)
    return w
def huber(e, c=1.0): a = np.abs(e); return np.where(a <= c, 0.5 * e ** 2, c * a - 0.5 * c ** 2)
def eins(e, eps=0.5): return np.maximum(0, np.abs(e) - eps)
def svr_lin(x, y, eps=0.5, C=10.0):
    from sklearn.svm import SVR
    m = SVR(kernel="linear", C=C, epsilon=eps).fit(x[:, None], y); return np.r_[m.coef_.ravel(), m.intercept_], m
def ridge_lin(x, y, C):
    A = np.c_[x, np.ones(len(x))]; return np.linalg.solve(A.T @ A + C * np.eye(2), A.T @ y)
if __name__ == "__main__":
    xc, yc = trend(out=False); xo, yo = trend()
    print("clean ols", ols(xc, yc), "outlier ols", ols(xo, yo))
    for c in (0.2, 0.5, 1, 2, 5, 20): print("huber c", c, huber_irls(xo, yo, c))
    for eps in (0.1, 0.5, 1.0): w, m = svr_lin(xo, yo, eps); print("svr eps", eps, w, "nSV", len(m.support_))
    for C in (0.1, 1, 10): print("ridge C", C, ridge_lin(xo, yo, C))
    print("huber e=0.5,3,10:", huber(np.array([0.5, 3, 10.])), "sq", 0.5 * np.array([0.5, 3, 10.]) ** 2, "eins", eins(np.array([0.3, 2.0])))
    # SVR sparsity on sin
    from sklearn.svm import SVR
    rng = np.random.default_rng(1); xs = np.sort(rng.uniform(0, 2 * np.pi, 60)); ys = np.sin(xs) + rng.normal(0, 0.15, 60); xg = np.linspace(0, 2 * np.pi, 300)
    for eps in (0.01, 0.1, 0.2, 0.3, 0.5, 1.0):
        m = SVR(kernel="rbf", gamma=0.5, C=10, epsilon=eps).fit(xs[:, None], ys); print("eps", eps, "nSV", len(m.support_), "mse vs sin", round(np.mean((m.predict(xg[:, None]) - np.sin(xg)) ** 2), 4), "train frac inside tube", np.mean(np.abs(m.predict(xs[:, None]) - ys) <= eps + 1e-9))
    # ridge primal/dual
    rng = np.random.default_rng(2); X = rng.normal(size=(15, 4)); y = rng.normal(size=15); C = 0.8
    wp = np.linalg.solve(X.T @ X + C * np.eye(4), X.T @ y); wd = X.T @ np.linalg.solve(X @ X.T + C * np.eye(15), y); print("primal-dual maxdiff", np.abs(wp - wd).max())
    # near-collinear
    rng = np.random.default_rng(3); t = rng.normal(size=30); X2 = np.c_[t, t + 1e-4 * rng.normal(size=30)]; y2 = t + 0.1 * rng.normal(size=30)
    print("cond XtX", np.linalg.cond(X2.T @ X2), "w ols", np.linalg.lstsq(X2, y2, rcond=None)[0])
    for C in (1e-3, 0.1, 1): print("ridge", C, np.linalg.solve(X2.T @ X2 + C * np.eye(2), X2.T @ y2), "cond", np.linalg.cond(X2.T @ X2 + C * np.eye(2)))
    # 2nd data draw to check variability of ols outlier effect
    for sd in range(3):
        a, b = trend(seed=sd, out=False), trend(seed=sd); print(sd, "slope clean", ols(*a)[0], "outlier", ols(*b)[0], "huber", huber_irls(*b)[0])
