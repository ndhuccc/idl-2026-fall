"""KP05 figures (test-error decomposition, Case I) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp05-bar, lec04-kp05-mc.   Prints every number the lecture and the deck quote.
Toy data: f(x) = 1 + 2x - 1.5x^2 on [-1, 1], noise sigma = 0.3 (sigma^2 = 0.09); the model is a straight line fitted on n = 15 points.
Run:  py assets/figures/src/lec04-kp05-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

SIGMA = 0.3
SIG2 = SIGMA ** 2
NTR, NTE_BIG, REPS = 15, 20000, 300


def f(x):
    return 1.0 + 2.0 * np.asarray(x) - 1.5 * np.asarray(x) ** 2


# ------------------------------------------------------------------ the worked example of the source (GUIDE 5)
N_TE, S2, ERR_EMP = 50, 4.0, 300.0
ERR_TRUE = ERR_EMP - N_TE * S2          # 100


def bar(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 300
    fig, ax = axes(W, H, left=30, bottom=60, top=20)
    ax.barh([0], [ERR_TRUE], color=c["ref"], height=0.5, label=r"$\mathbf{Err}=100$ (true error)")
    ax.barh([0], [N_TE * S2], left=[ERR_TRUE], color=c["warn"], height=0.5, label=r"$n_{te}\sigma^2=50\cdot4=200$ (noise floor)")
    ax.text(ERR_TRUE / 2, 0, "100", ha="center", va="center", color="#ffffff", fontsize=16, fontweight="bold")
    ax.text(ERR_TRUE + N_TE * S2 / 2, 0, "200", ha="center", va="center", color="#ffffff", fontsize=16, fontweight="bold")
    ax.annotate("", xy=(ERR_EMP, 0.42), xytext=(0, 0.42), arrowprops=dict(arrowstyle="<->", color=c["text"], lw=1.4))
    ax.text(ERR_EMP / 2, 0.5, r"$\mathbf{err}=300$ (what we can compute)", ha="center", va="bottom", color=c["text"], fontsize=14)
    ax.set_xlim(0, 320)
    ax.set_ylim(-1.15, 0.95)
    ax.set_yticks([])
    ax.set_xlabel("sum of squared errors over 50 test points")
    ax.legend(loc="lower center", frameon=False, fontsize=12, ncol=1, bbox_to_anchor=(0.5, 0.0))
    clean(ax)
    ax.grid(False)
    finish(fig, "lec04-kp05-bar", dark, c)
    return W, H


# ------------------------------------------------------------------ Monte Carlo: a straight line fitted on 15 points, scored on fresh points
def monte_carlo(seed=5):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(REPS):
        x = rng.uniform(-1, 1, NTR)
        y = f(x) + rng.normal(0, SIGMA, NTR)
        b1, b0 = np.polyfit(x, y, 1)
        xt = rng.uniform(-1, 1, NTE_BIG)
        yt = f(xt) + rng.normal(0, SIGMA, NTE_BIG)
        fh = b0 + b1 * xt
        out.append((np.mean((fh - yt) ** 2), np.mean((fh - f(xt)) ** 2)))
    return np.array(out)


def mc(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    r = monte_carlo()
    lhs, rhs = r[:, 0], r[:, 1] + SIG2
    ax.scatter(rhs, lhs, s=14, color=c["main"], alpha=0.75, label="300 fitted lines")
    lo, hi = min(rhs.min(), lhs.min()) - 0.01, max(rhs.max(), lhs.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], color=c["ok"], lw=1.8, ls="--", label=r"$y=x$")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"mean$(\hat f-f)^2+\sigma^2$")
    ax.set_ylabel(r"mean$(\hat f-y)^2$ on fresh points")
    ax.legend(loc="upper left", frameon=False, fontsize=13)
    clean(ax)
    finish(fig, "lec04-kp05-mc", dark, c)
    return W, H


def report():
    print("source example: n_te =", N_TE, " sigma^2 =", S2, " err =", ERR_EMP, " -> Err =", ERR_TRUE, " per point:", ERR_EMP / N_TE, ERR_TRUE / N_TE, " floor per point", S2)
    print("stack: ", ERR_TRUE, "+", N_TE * S2, "=", ERR_TRUE + N_TE * S2)
    r = monte_carlo()
    lhs, rhs = r[:, 0], r[:, 1] + SIG2
    print(f"toy: sigma={SIGMA} sigma^2={SIG2}; line on n={NTR} points; {REPS} repeats, {NTE_BIG} fresh test points each")
    print(f"  mean over repeats of mean(f^-y)^2        = {lhs.mean():.4f}")
    print(f"  mean over repeats of mean(f^-f)^2        = {r[:, 1].mean():.4f}")
    print(f"  mean over repeats of mean(f^-f)^2+sigma^2 = {rhs.mean():.4f}")
    print(f"  max |lhs - rhs| over repeats = {np.max(np.abs(lhs - rhs)):.4f}; correlation {np.corrcoef(lhs, rhs)[0, 1]:.4f}")
    # cross term explicitly
    rng = np.random.default_rng(7)
    x = rng.uniform(-1, 1, NTR); y = f(x) + rng.normal(0, SIGMA, NTR)
    b1, b0 = np.polyfit(x, y, 1)
    xt = rng.uniform(-1, 1, 400000); eps = rng.normal(0, SIGMA, 400000)
    fh = b0 + b1 * xt
    print(f"  one fitted line: mean(eps*(f^-f)) on 400000 fresh points = {np.mean(eps * (fh - f(xt))):.5f}  (cross term, should be ~0)")
    print(f"  mean(eps^2) = {np.mean(eps ** 2):.4f}")


if __name__ == "__main__":
    for dk in (False, True):
        sizes = [bar(dk), mc(dk)]
    report()
