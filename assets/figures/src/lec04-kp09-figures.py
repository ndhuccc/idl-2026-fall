"""KP09 figures (cross-validation) in two skins from the same data.
    assets/figures/<id>.png (light, lecture)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp09-folds (5-fold rotation and the LOOCV strip), lec04-kp09-cv-curve (5-fold CV error against polynomial degree).
Toy data: f(x) = 1 + 2x - 1.5x^2, sigma = 0.3 (sigma^2 = 0.09), n = 100 points, x ~ Uniform(-1, 1), noise from a fixed seed.
Folds: one random permutation (seed 5) cut into 5 blocks of 20. LOOCV is computed twice (100 brute-force refits and the hat-matrix shortcut
e_i / (1 - h_ii) of KP07) as a cross-check. The experiments on the luck of a single split and on the choice of K are repeated over many seeds.
Run:  python3 assets/figures/src/lec04-kp09-figures.py     (prints every number the lecture and the deck quote)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

SIGMA = 0.3
N = 100
DEGS = list(range(1, 9))
XG = np.linspace(-1, 1, 2001)


def f(x):
    return 1.0 + 2.0 * np.asarray(x) - 1.5 * np.asarray(x) ** 2


def make(seed, n=N):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    return x, f(x) + rng.normal(0, SIGMA, n)


X, Y = make(9)
PERM = np.random.default_rng(5).permutation(N)
FOLDS = PERM.reshape(5, 20)


def fit_mse(xtr, ytr, xte, yte, d):
    return float(np.mean((np.polyval(np.polyfit(xtr, ytr, d), xte) - yte) ** 2))


def kfold_errors(x, y, d, perm, K):
    """MSE on each held-out fold when the model of degree d is trained on the other K-1 folds."""
    parts = np.array_split(perm, K)
    out = []
    for k in range(K):
        te = parts[k]
        tr = np.concatenate([parts[j] for j in range(K) if j != k])
        out.append(fit_mse(x[tr], y[tr], x[te], y[te], d))
    return np.array(out)


def loocv_brute(x, y, d):
    n = len(x)
    e = [fit_mse(np.delete(x, i), np.delete(y, i), x[i:i + 1], y[i:i + 1], d) for i in range(n)]
    return float(np.mean(e))


def loocv_shortcut(x, y, d):
    """For least squares the leave-one-out residual is e_i / (1 - h_ii), with h_ii the hat-matrix diagonal (KP07)."""
    V = np.vander(x, d + 1)
    H = V @ np.linalg.pinv(V)
    e = y - H @ y
    return float(np.mean((e / (1 - np.diag(H))) ** 2))


def true_mse(x, y, d):
    """Expected squared error of the degree-d fit on fresh noisy data, x ~ Uniform(-1, 1): mean (fhat - f)^2 + sigma^2."""
    return float(np.mean((np.polyval(np.polyfit(x, y, d), XG) - f(XG)) ** 2) + SIGMA ** 2)


def folds(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, 16)
    W, H = 1060, 340
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    x0, bw, bh, gap = 150, 120, 36, 6
    ax.text(x0 - 10, 18, "5-fold (K=5, 20 points per fold)", ha="left", va="center", color=c["text"], fontsize=14, fontweight="bold")
    for r in range(5):
        y = 38 + r * (bh + 8)
        ax.text(x0 - 14, y + bh / 2, f"round {r + 1}", ha="right", va="center", color=c["muted"], fontsize=13)
        for k in range(5):
            test = k == r
            ax.add_patch(Rectangle((x0 + k * (bw + gap), y), bw, bh, facecolor=c["warn"] if test else c["ref"], alpha=0.95 if test else 0.55,
                                   edgecolor="none"))
            ax.text(x0 + k * (bw + gap) + bw / 2, y + bh / 2, ("test" if test else "train") + f" (fold {k + 1})", ha="center", va="center",
                    color="#ffffff" if test else c["text"], fontsize=12)
    ax.text(x0 + 5 * (bw + gap) + 4, 38 + 2 * (bh + 8) + bh / 2, r"$\to E_1,\dots,E_5$", ha="left", va="center", color=c["text"], fontsize=14)
    ax.text(x0 - 10, 18, "", color=c["text"])
    y2 = 38 + 5 * (bh + 8) + 26
    ax.text(x0 - 10, y2 - 8, "LOOCV (K=n=100, shown with 20 blocks)", ha="left", va="center", color=c["text"], fontsize=14, fontweight="bold")
    nb = 20
    wb = (5 * bw + 4 * gap) / nb
    for k in range(nb):
        test = k == 7
        ax.add_patch(Rectangle((x0 + k * wb, y2 + 6), wb - 2, bh, facecolor=c["warn"] if test else c["ref"], alpha=0.95 if test else 0.55, edgecolor="none"))
    ax.text(x0 - 14, y2 + 6 + bh / 2, "round i", ha="right", va="center", color=c["muted"], fontsize=13)
    ax.text(x0 + 5 * (bw + gap) + 4, y2 + 6 + bh / 2, "n fits, 99 points each", ha="left", va="center", color=c["text"], fontsize=14)
    finish(fig, "lec04-kp09-folds", dark, c)
    return W, H


def cv_curve(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H, left=104)
    errs = {d: kfold_errors(X, Y, d, PERM, 5) for d in DEGS}
    cv = np.array([errs[d].mean() for d in DEGS])
    best = DEGS[int(np.argmin(cv))]
    for d in DEGS:
        ax.scatter([d] * 5, errs[d], s=26, color=c["muted"], alpha=0.8, zorder=3, label="fold error $E_k$" if d == 1 else None)
    ax.plot(DEGS, cv, "-o", color=c["ref"], lw=2.4, ms=6, zorder=4, label="CV error (mean of 5)")
    ax.axhline(SIGMA ** 2, color=c["ok"], ls="--", lw=1.4)
    ax.annotate(r"noise floor $\sigma^2=0.09$", (4.2, SIGMA ** 2), xytext=(0, -20), textcoords="offset points", color=c["ok"], fontsize=13)
    ax.scatter([best], [cv[best - 1]], s=190, facecolors="none", edgecolors=c["main"], linewidths=2.4, zorder=5)
    ax.annotate(f"min at degree {best}", (best, cv[best - 1]), xytext=(30, 52), textcoords="offset points", color=c["text"], fontsize=13,
                arrowprops=dict(arrowstyle="->", color=c["muted"]))
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks([0.06, 0.1, 0.2, 0.4])
    ax.set_yticklabels(["0.06", "0.1", "0.2", "0.4"])
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("MSE (log scale)")
    ax.set_xticks(DEGS)
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    clean(ax)
    finish(fig, "lec04-kp09-cv-curve", dark, c)
    return W, H


def report():
    print(f"data: n={N}, x~U(-1,1), noise seed 9, sigma={SIGMA}; fold permutation seed 5 -> 5 folds of 20; first fold indices {FOLDS[0][:6].tolist()}...")
    print("degree | five fold errors E_1..E_5 | CV mean | LOOCV brute | LOOCV shortcut | true MSE of full-data fit | train MSE")
    for d in DEGS:
        e = kfold_errors(X, Y, d, PERM, 5)
        tr = float(np.mean((np.polyval(np.polyfit(X, Y, d), X) - Y) ** 2))
        print(f"{d:5d}  {np.round(e, 4).tolist()}  {e.mean():.4f}  {loocv_brute(X, Y, d):.4f}  {loocv_shortcut(X, Y, d):.4f}  {true_mse(X, Y, d):.4f}  {tr:.4f}")
    cv = [kfold_errors(X, Y, d, PERM, 5).mean() for d in DEGS]
    lo = [loocv_shortcut(X, Y, d) for d in DEGS]
    print("5-fold picks degree", DEGS[int(np.argmin(cv))], "| LOOCV picks degree", DEGS[int(np.argmin(lo))])
    d2 = kfold_errors(X, Y, 2, PERM, 5)
    print("degree 2 fold errors:", np.round(d2, 4).tolist(), "sum", round(float(d2.sum()), 4), "mean", round(float(d2.mean()), 4),
          "min", round(float(d2.min()), 4), "max", round(float(d2.max()), 4))

    # luck of a single 80/20 split: 300 random splits of the same data
    rng = np.random.default_rng(100)
    val2, picks = [], []
    for _ in range(300):
        p = rng.permutation(N)
        tr, te = p[:80], p[80:]
        v = [fit_mse(X[tr], Y[tr], X[te], Y[te], d) for d in DEGS]
        val2.append(v[1])
        picks.append(DEGS[int(np.argmin(v))])
    val2 = np.array(val2)
    cnt = {d: picks.count(d) for d in DEGS}
    print("single 80/20 split, degree 2 validation MSE over 300 splits: mean %.4f std %.4f min %.4f max %.4f" % (val2.mean(), val2.std(), val2.min(), val2.max()))
    print("degree chosen by one 80/20 split (counts of 300):", cnt)
    picks5 = []
    for _ in range(300):
        p = rng.permutation(N)
        picks5.append(DEGS[int(np.argmin([kfold_errors(X, Y, d, p, 5).mean() for d in DEGS]))])
    print("degree chosen by 5-fold CV over 300 reshuffles (counts of 300):", {d: picks5.count(d) for d in DEGS})
    cv2 = np.array([kfold_errors(X, Y, 2, rng.permutation(N), 5).mean() for _ in range(300)])
    print("degree 2: 5-fold CV mean over 300 reshuffles of the same data: mean %.4f std %.4f min %.4f max %.4f" % (cv2.mean(), cv2.std(), cv2.min(), cv2.max()))

    # choice of K: 300 fresh datasets, degree 2, estimate of the error of the full-data fit
    Ks = [2, 5, 10, 20, 100]
    est = {K: [] for K in Ks}
    truth = []
    for s in range(300):
        x, y = make(1000 + s)
        truth.append(true_mse(x, y, 2))
        pr = np.random.default_rng(2000 + s).permutation(N)
        for K in Ks:
            est[K].append(loocv_shortcut(x, y, 2) if K == N else kfold_errors(x, y, 2, pr, K).mean())
    truth = np.array(truth)
    print("choice of K, degree 2, 300 datasets of n=100; target = exact test MSE of the full-data fit (mean %.4f)" % truth.mean())
    print("K   fits   mean estimate   mean(estimate-target)   std(estimate)   std(estimate-target)")
    for K in Ks:
        e = np.array(est[K])
        print(f"{K:3d}  {K:4d}   {e.mean():.4f}          {np.mean(e - truth):+.4f}                {e.std():.4f}         {np.std(e - truth):.4f}")


if __name__ == "__main__":
    for dk in (False, True):
        folds(dk)
        cv_curve(dk)
    report()
