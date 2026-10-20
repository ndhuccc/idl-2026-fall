"""KP03 figures (covariance and linear combinations) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp03-scatter, lec04-kp03-bars, lec04-kp03-parabola.   Prints every number the lecture and the deck quote.
Run:  python3 assets/figures/src/lec04-kp03-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

VX, VY = 4.0, 9.0                         # Var X, Var Y of the source example
COVS = (3.0, 0.0, -3.0)                   # Cov > 0, = 0, < 0
N = 20000                                 # sample size per panel (statistics); only the first NPLOT points are drawn
NPLOT = 300
SEEDS = (11, 12, 13)


def panels():
    """Sample N points of (X, Y) for each covariance; return rows (cov, samples, sample stats)."""
    out = []
    for cov, seed in zip(COVS, SEEDS):
        rng = np.random.default_rng(seed)
        xy = rng.multivariate_normal([0, 0], [[VX, cov], [cov, VY]], size=N)
        x, y = xy[:, 0], xy[:, 1]
        s = x + y
        out.append(dict(cov=cov, x=x, y=y, var_sum=float(s.var()), cov_hat=float(np.mean((x - x.mean()) * (y - y.mean()))),
                        theory=VX + VY + 2 * cov, var_x=float(x.var()), var_y=float(y.var())))
    return out


def scatter(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, 16)
    W, H = 1100, 330
    fig = plt.figure(figsize=(W / 100, H / 100))
    titles = {3.0: r"Cov$(X,Y)=+3$", 0.0: r"Cov$(X,Y)=0$", -3.0: r"Cov$(X,Y)=-3$"}
    for i, d in enumerate(panels()):
        ax = fig.add_axes([(62 + i * 356) / W, 60 / H, 290 / W, 214 / H])
        ax.plot(d["x"][:NPLOT], d["y"][:NPLOT], "o", color=c["main"], ms=3.5, alpha=0.55)
        ax.axhline(0, color=c["muted"], lw=0.8, alpha=0.6)
        ax.axvline(0, color=c["muted"], lw=0.8, alpha=0.6)
        ax.set_xlim(-8, 8)
        ax.set_ylim(-11, 11)
        ax.set_xlabel("$X$", labelpad=1)
        if i == 0:
            ax.set_ylabel("$Y$", labelpad=1)
        ax.set_title(titles[d["cov"]], fontsize=14, color=c["text"], pad=22)
        ax.text(0.5, 1.03, rf"Var$(X+Y)$: theory {d['theory']:.0f}, sample {d['var_sum']:.1f}", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=11.5, color=c["muted"])
        clean(ax)
    finish(fig, "lec04-kp03-scatter", dark, c)
    return W, H


def bars(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 600, 380
    fig, ax = axes(W, H, left=70, bottom=62, right=20, top=26)
    ds = panels()
    order = [ds[2], ds[1], ds[0]]                     # Cov -3, 0, +3
    xs = np.arange(3)
    ax.bar(xs - 0.2, [d["theory"] for d in order], 0.38, color=c["ref"], alpha=0.85, label="theory")
    ax.bar(xs + 0.2, [d["var_sum"] for d in order], 0.38, color=c["main"], alpha=0.85, label="sample")
    ax.axhline(VX + VY, color=c["ok"], lw=2, ls="--")
    ax.text(-0.5, VX + VY + 0.5, r"Var$X$+Var$Y=13$", color=c["ok"], fontsize=12, ha="left", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([r"Cov$=-3$", r"Cov$=0$", r"Cov$=+3$"])
    ax.set_ylabel(r"Var$(X+Y)$")
    ax.set_ylim(0, 24)
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    clean(ax)
    finish(fig, "lec04-kp03-bars", dark, c)
    return W, H


def parabola(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 520, 380
    fig, ax = axes(W, H, left=70, bottom=62, right=24, top=26)
    xs = np.arange(-2, 3)
    ys = xs ** 2
    xx = np.linspace(-2.4, 2.4, 100)
    ax.plot(xx, xx ** 2, color=c["muted"], lw=1.2, ls=":")
    ax.plot(xs, ys, "o", color=c["main"], ms=10)
    ax.axvline(0, color=c["muted"], lw=0.8, alpha=0.6)
    ax.set_xlabel("$X$")
    ax.set_ylabel(r"$Y=X^2$")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-0.6, 6.4)
    ax.text(0, 5.7, r"Cov$(X,Y)=0$ but $Y$ is a function of $X$", ha="center", color=c["text"], fontsize=13)
    clean(ax)
    finish(fig, "lec04-kp03-parabola", dark, c)
    return W, H


def report():
    print(f"source example: Var X={VX}, Var Y={VY}; a=b=1")
    for d in panels():
        print(f"  Cov {d['cov']:+.0f}: theory Var(X+Y) = {VX} + {VY} + 2*({d['cov']:+.0f}) = {d['theory']:.0f}; sample (n={N}) Var X {d['var_x']:.2f}, "
              f"Var Y {d['var_y']:.2f}, Cov {d['cov_hat']:+.2f}, Var(X+Y) {d['var_sum']:.2f}; check Var X+Var Y+2Cov = {d['var_x'] + d['var_y'] + 2 * d['cov_hat']:.2f}")
    print(f"  correlation for Cov +3: {3 / np.sqrt(VX * VY):.2f}; Cov -3: {-3 / np.sqrt(VX * VY):.2f}")
    # general weights, e.g. a=2, b=-1 with Cov=+3 / -3
    for cov in (3, -3):
        a, b = 2, -1
        print(f"  a={a}, b={b}, Cov {cov:+d}: a^2*4 + b^2*9 + 2ab*Cov = {a * a * VX + b * b * VY + 2 * a * b * cov:.0f}")
    # zero covariance but dependent: X uniform on {-2..2}, Y = X^2
    x = np.arange(-2, 3)
    y = x ** 2
    print(f"zero-cov example: E X = {x.mean():.1f}, E Y = {y.mean():.1f}, E XY = {(x * y).mean():.1f}, Cov = {(x * y).mean() - x.mean() * y.mean():.1f}; "
          f"P(Y=4) = {np.mean(y == 4):.1f}, P(Y=4 | X=2) = 1; E XY = E X^3")
    # equal-weights special case: k variables, variance s, pairwise covariance c, weights 1/k
    for k, s, c_ in ((4, 1.0, 0.5), (10, 1.0, 0.0), (10, 1.0, 0.8)):
        theory = s / k + (k - 1) * c_ / k
        cov = np.full((k, k), c_) + np.eye(k) * (s - c_)
        w = np.full(k, 1 / k)
        direct = float(w @ cov @ w)
        rng = np.random.default_rng(5)
        sim = float(rng.multivariate_normal(np.zeros(k), cov, size=200000).mean(axis=1).var())
        print(f"  equal weights k={k}, s={s}, c={c_}: s/k + (k-1)c/k = {theory:.4f}; w'Cw = {direct:.4f}; simulated {sim:.4f}")
    # ordered pairs: k=3, a_i = 1
    print("ordered pairs for k=3: i != j gives", 3 * 2, "terms = 2 * C(3,2) =", 2 * 3)


if __name__ == "__main__":
    sizes = {}
    for fn in (scatter, bars, parabola):
        sizes[fn.__name__] = [fn(True), fn(False)][0]
    print("sizes", sizes)
    report()
