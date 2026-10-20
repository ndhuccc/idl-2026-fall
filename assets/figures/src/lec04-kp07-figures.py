"""KP07 figures (training-point sensitivity and model complexity, Case II) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp07-shift (two panels: degree 1 and degree 4, one label moved), lec04-kp07-hat (diagonal of the hat matrix).
Prints every number the lecture and the deck quote.
Setup: five training points x = (-2,-1,0,1,2); the label of the middle point is moved up.
Run:  py assets/figures/src/lec04-kp07-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

X = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
Y = np.array([0.3, 1.1, 1.6, 1.0, 0.6])        # fixed labels (any values work: the derivatives do not depend on y)
MID = 2                                         # index of the point that is moved
SHIFT_FIG = 0.5                                 # shift drawn in the figure (the lecture quotes 0.1; the responses scale linearly)
SHIFT_TXT = 0.1
GRID = np.linspace(-2.4, 2.4, 300)


def design(x, deg):
    return np.vander(x, deg + 1, increasing=True)


def hat(deg):
    D = design(X, deg)
    return D @ np.linalg.solve(D.T @ D, D.T)


def fit_pred(deg, y, xs):
    D = design(X, deg)
    beta = np.linalg.solve(D.T @ D, D.T @ y)
    return design(xs, deg) @ beta


def numbers():
    out = {}
    for deg in (1, 4):
        H = hat(deg)
        out[deg] = dict(diag=np.diag(H), trace=np.trace(H))
        y2 = Y.copy()
        y2[MID] += SHIFT_TXT
        out[deg]["move"] = fit_pred(deg, y2, X)[MID] - fit_pred(deg, Y, X)[MID]
        out[deg]["rss"] = float(np.sum((Y - fit_pred(deg, Y, X)) ** 2))
    return out


def shift(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 900, 340
    fig = plt.figure(figsize=(W / 100, H / 100))
    L, B, T_, PW, GAP = 74, 62, 34, 376, 36
    y2 = Y.copy()
    y2[MID] += SHIFT_FIG
    axs = []
    for k, deg in enumerate((1, 4)):
        ax = fig.add_axes([(L + k * (PW + GAP)) / W, B / H, PW / W, (H - B - T_) / H])
        base, moved = fit_pred(deg, Y, GRID), fit_pred(deg, y2, GRID)
        ax.plot(GRID, base, color=c["ref"], lw=2.2, label="original")
        ax.plot(GRID, moved, color=c["main"], lw=2.2, ls="--", label="label moved")
        ax.scatter(np.delete(X, MID), np.delete(Y, MID), s=46, color=c["text"], zorder=5)
        ax.scatter([X[MID]], [Y[MID]], s=60, color=c["ref"], zorder=6)
        ax.scatter([X[MID]], [y2[MID]], s=60, color=c["main"], zorder=6)
        ax.annotate("", xy=(X[MID], y2[MID] - 0.02), xytext=(X[MID], Y[MID] + 0.02), arrowprops=dict(arrowstyle="->", color=c["warn"], lw=1.6))
        d = fit_pred(deg, y2, X)[MID] - fit_pred(deg, Y, X)[MID]
        ax.set_title("degree %d:  $\\Delta\\hat f=%.2f$ for a move of %.1f" % (deg, d, SHIFT_FIG), fontsize=15, color=c["text"])
        ax.set_xlabel("$x$")
        ax.set_ylim(-0.2, 2.5)
        ax.set_xlim(-2.4, 2.4)
        if k == 0:
            ax.set_ylabel("$y$")
            ax.legend(loc="lower center", frameon=False, fontsize=12)
        clean(ax)
        axs.append(ax)
    finish(fig, "lec04-kp07-shift", dark, c)
    return W, H


def hat_bars(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 330
    fig, ax = axes(W, H, left=92, bottom=64, top=26)
    n = numbers()
    x = np.arange(5)
    w = 0.36
    ax.bar(x - w / 2, n[1]["diag"], w, color=c["ref"], label=r"degree 1, $\sum=2$")
    ax.bar(x + w / 2, n[4]["diag"], w, color=c["main"], label=r"degree 4, $\sum=5$")
    ax.set_xticks(x)
    ax.set_xticklabels(["$x=%d$" % v for v in X])
    ax.set_ylabel(r"$\partial\hat f_i/\partial y_i=h_{ii}$")
    ax.set_ylim(0, 1.3)
    ax.legend(loc="upper left", frameon=False, fontsize=13, ncol=2)
    clean(ax)
    finish(fig, "lec04-kp07-hat", dark, c)
    return W, H


def sigma_estimators(seed=21, reps=200000, sigma=0.3):
    """Linear truth y = 1 + 0.5 x + noise on the five points: compare RSS/(n-1), RSS/(n-2) and the interpolating fit."""
    rng = np.random.default_rng(seed)
    D = design(X, 1)
    P = D @ np.linalg.solve(D.T @ D, D.T)
    e = rng.normal(0.0, sigma, (reps, 5))
    res = e - e @ P                                   # residuals of the linear fit (P is symmetric, truth is in the column space)
    rss = np.sum(res ** 2, axis=1)
    return rss.mean() / 4, rss.mean() / 3, sigma ** 2


if __name__ == "__main__":
    n = numbers()
    print("x =", X.tolist(), " y =", Y.tolist(), " moved point x = %g" % X[MID])
    for deg in (1, 4):
        print("degree %d: h_ii = %s  trace = %.4f  prediction change for a label move of %.1f = %.4f  RSS = %.4f"
              % (deg, np.round(n[deg]["diag"], 4).tolist(), n[deg]["trace"], SHIFT_TXT, n[deg]["move"], n[deg]["rss"]))
    print("closed form degree 1: 1/5 + x^2/10 =", np.round(0.2 + X ** 2 / 10, 4).tolist())
    a, b, s2 = sigma_estimators()
    print("sigma^2 = %.4f: mean of RSS/(n-1) = %.4f, mean of RSS/(n-p) with p = 2 -> %.4f  (200000 repeats, linear truth)" % (s2, a, b))
    print("penalty 2 sigma^2 p, sigma^2 = 0.09: p = 2 -> %.3f, p = 5 -> %.3f" % (2 * 0.09 * 2, 2 * 0.09 * 5))
    for fn in (shift, hat_bars):
        for dark in (False, True):
            fn(dark)
    print("figures written")
