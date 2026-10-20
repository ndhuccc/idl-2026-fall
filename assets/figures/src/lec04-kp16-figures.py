"""KP16 figures (Bagging and variance reduction) in two skins from the same data.
    assets/figures/<id>.png (light, lecture)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp16-var-vs-k, lec04-kp16-bagging-demo.   Prints every number the lecture and the deck quote.
Formula: Var(mean of k errors) = s/k + (k-1)c/k  (equal variance s, equal pairwise covariance c).
Demo: true f(x) = 1 + 2x - 1.5x^2 on [-1, 1], noise sigma = 0.3, n = 20 training points, degree-9 polynomial, B = 30 bootstrap fits.
Run:  python3 assets/figures/src/lec04-kp16-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SIGMA = 0.3
N_TRAIN, B = 20, 30


def f(x):
    return 1.0 + 2.0 * np.asarray(x) - 1.5 * np.asarray(x) ** 2


def bag_var(s, c, k):
    """Variance of the average of k errors with variance s and pairwise covariance c."""
    return s / k + (k - 1) / k * c


def nn_predict(x, y, xg):
    """1-nearest-neighbour regression: copy the label of the closest training point (a very flexible, very unstable learner)."""
    return y[np.abs(np.asarray(xg)[:, None] - x[None, :]).argmin(1)]


def one_training_set(seed, n=N_TRAIN):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1, 1, n))
    return x, f(x) + rng.normal(0, SIGMA, n)


def bagged(x, y, xg, rng, nb=B, learner=nn_predict):
    preds = []
    for _ in range(nb):
        idx = rng.integers(0, len(x), len(x))
        preds.append(learner(x[idx], y[idx], xg))
    return np.array(preds)


# ------------------------------------------------------------------ figure 1: variance against k
def var_vs_k(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    k = np.arange(1, 51)
    for ratio, col in ((0.0, c["vec"]), (0.3, c["ref"]), (0.8, c["warn"])):
        ax.plot(k, bag_var(1.0, ratio, k), color=col, lw=2.6, label=rf"$c/s={ratio:g}$")
    ax.scatter([10, 10], [bag_var(1, 0, 10), bag_var(1, 0.8, 10)], color=c["main"], zorder=5, s=46)
    ax.annotate(r"$0.1$", (10, 0.1), xytext=(10, 8), textcoords="offset points", color=c["main"], fontsize=14)
    ax.annotate(r"$0.82$", (10, 0.82), xytext=(8, 8), textcoords="offset points", color=c["main"], fontsize=14)
    ax.set_xlabel(r"number of models $k$")
    ax.set_ylabel(r"$\mathrm{Var}$ of the average ($s=1$)")
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", frameon=False, fontsize=13)
    clean(ax)
    finish(fig, "lec04-kp16-var-vs-k", dark, c)
    return W, H


# ------------------------------------------------------------------ figure 2: bagging 1-nearest-neighbour regression
def bagging_demo(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    xg = np.linspace(-1, 1, 400)
    x, y = one_training_set(0)
    rng = np.random.default_rng(100)
    preds = bagged(x, y, xg, rng)
    for p in preds[:10]:
        ax.step(xg, p, color=c["ref"], lw=0.8, alpha=0.40, where="mid")
    ax.step(xg, nn_predict(x, y, xg), color=c["warn"], lw=1.8, ls="--", where="mid", label="single 1-NN fit")
    ax.plot(xg, preds.mean(axis=0), color=c["vec"], lw=2.8, label=rf"bagged average ($B={B}$)")
    ax.plot(xg, f(xg), color=c["text"], lw=1.6, label=r"true $f$")
    ax.scatter(x, y, color=c["edge"], s=26, zorder=5, label=r"training points ($n=20$)")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-3.1, 3.3)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.legend(loc="upper left", frameon=False, fontsize=11, ncol=1)
    clean(ax)
    finish(fig, "lec04-kp16-bagging-demo", dark, c)
    return W, H


def report():
    print("== formula Var = s/k + (k-1)c/k  (s = 1)")
    for k, cc in ((10, 0.0), (10, 0.8), (10, 0.3), (1, 0.8), (50, 0.8), (50, 0.0)):
        print(f"  k={k:2d} c={cc}: {bag_var(1.0, cc, k):.4f}")
    print("  limit k->inf with c=0.8:", 0.8, "; c=0.3:", 0.3)
    print("  check against derivation: k*(1/k)^2*s + k(k-1)*(1/k)^2*c =", 10 * 0.01 * 1 + 90 * 0.01 * 0.8)
    n = 1000
    print("== bootstrap: P(point included) = 1-(1-1/n)^n =", round(1 - (1 - 1 / n) ** n, 4), " 1-1/e =", round(1 - np.exp(-1), 4))
    for n in (5, 10, 20, 100):
        print(f"  n={n}: {1 - (1 - 1 / n) ** n:.4f}")
    rng = np.random.default_rng(0)
    frac = [len(np.unique(rng.integers(0, 20, 20))) / 20 for _ in range(4000)]
    print("  simulated mean fraction of distinct points, n=20:", round(float(np.mean(frac)), 4), " theory", round(1 - (1 - 1 / 20) ** 20, 4))
    print("== demo: 1-NN regression, variance over 300 training sets (n=20, sigma 0.3, B=30 bootstrap models)")
    xs = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
    singles, bags = [], []
    rng = np.random.default_rng(7)
    for t in range(300):
        x, y = one_training_set(1000 + t)
        singles.append(nn_predict(x, y, xs))
        bags.append(bagged(x, y, xs, rng).mean(axis=0))
    singles, bags = np.array(singles), np.array(bags)
    fx = f(xs)
    for j, xv in enumerate(xs):
        print(f"  x={xv:+.1f}: var single {singles[:, j].var():.4f}  var bagged {bags[:, j].var():.4f}  ratio {bags[:, j].var() / singles[:, j].var():.3f}"
              f"   bias^2 single {(singles[:, j].mean() - fx[j]) ** 2:.5f} bagged {(bags[:, j].mean() - fx[j]) ** 2:.5f}")
    print("  mean over the 5 x: var single", round(float(singles.var(axis=0).mean()), 4), " var bagged", round(float(bags.var(axis=0).mean()), 4),
          " ratio", round(float(bags.var(axis=0).mean() / singles.var(axis=0).mean()), 3))
    print("  mean squared error vs f:", round(float(((singles - fx) ** 2).mean()), 4), "single;", round(float(((bags - fx) ** 2).mean()), 4), "bagged;",
          " noise floor sigma^2 =", SIGMA ** 2, " (MSE vs f, not vs y)")
    print("  squared error vs a fresh y at those x: single", round(float(((singles - fx) ** 2).mean() + SIGMA ** 2), 4), " bagged", round(float(((bags - fx) ** 2).mean() + SIGMA ** 2), 4))
    # --- caution: bagging a high-degree polynomial does NOT help
    print("== caution: bagging a degree-5 polynomial (n=20, 300 training sets, B=30)")
    S, G = [], []
    rng = np.random.default_rng(7)
    for t in range(300):
        x, y = one_training_set(1000 + t)
        S.append(np.polyval(np.polyfit(x, y, 5), xs))
        P = []
        for _ in range(B):
            i = rng.integers(0, len(x), len(x))
            if len(np.unique(x[i])) <= 7:
                continue
            P.append(np.polyval(np.polyfit(x[i], y[i], 5), xs))
        G.append(np.mean(P, axis=0))
    S, G = np.array(S), np.array(G)
    print("  mean squared error vs f over the 5 x: single", round(float(((S - fx) ** 2).mean()), 4), " bagged", round(float(((G - fx) ** 2).mean()), 4),
          " (bootstrap repeats make some polynomial fits wild at the edges)")


if __name__ == "__main__":
    sizes = {"lec04-kp16-var-vs-k": var_vs_k(False), "lec04-kp16-bagging-demo": bagging_demo(False)}
    var_vs_k(True)
    bagging_demo(True)
    print("sizes", sizes)
    report()
