"""KP06 figures (Stein's lemma and SURE) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp06-stein, lec04-kp06-sure.   Prints every number the lecture and the deck quote.
Run:  py assets/figures/src/lec04-kp06-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

SIGMAS = [0.5, 1.0, 1.5, 2.0]
M = 4_000_000                         # Monte-Carlo sample size per sigma
# SURE example of the source (GUIDE 6)
N_TR, MSE_TR, S2 = 30, 3.0, 1.0
ERR_EMP = N_TR * MSE_TR               # 90
SENS = [0.5, 0.8]                     # mean sensitivity d f_i / d y_i


def stein_numbers(seed=11):
    """E((z-mu) g(z)) versus sigma^2 E(g'(z)) for g(z) = tanh(z), mu = 0; and the linear case g = c z."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in SIGMAS:
        z = rng.normal(0.0, s, M)
        lhs = np.mean(z * np.tanh(z))
        rhs = s ** 2 * np.mean(1.0 - np.tanh(z) ** 2)
        rows.append((s, lhs, rhs))
    return rows


def sure_numbers():
    out = []
    for a in SENS:
        pen = 2 * S2 * N_TR * a
        out.append((a, pen, ERR_EMP - N_TR * S2 + pen))
    return out


def stein(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 340
    fig, ax = axes(W, H, left=92, bottom=64, top=22)
    rows = stein_numbers()
    x = np.arange(len(rows))
    w = 0.36
    ax.bar(x - w / 2, [r[1] for r in rows], w, color=c["ref"], label=r"$\mathbb{E}\,[z\,g(z)]$")
    ax.bar(x + w / 2, [r[2] for r in rows], w, color=c["main"], label=r"$\sigma^2\,\mathbb{E}\,[g'(z)]$")
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\sigma=%g$" % r[0] for r in rows])
    ax.set_ylabel(r"value, $g(z)=\tanh z$")
    ax.legend(loc="upper left", frameon=False, fontsize=13)
    clean(ax)
    finish(fig, "lec04-kp06-stein", dark, c)
    return W, H


def sure(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 340
    fig, ax = axes(W, H, left=92, bottom=64, top=30)
    rs = sure_numbers()
    labels = [r"$\mathbf{err}$", r"$\mathbf{Err}$, $s=0.5$", r"$\mathbf{Err}$, $s=0.8$"]
    vals = [ERR_EMP] + [r[2] for r in rs]
    cols = [c["ref"], c["ok"], c["warn"]]
    ax.bar(range(3), vals, color=cols, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 2.5, "%g" % v, ha="center", va="bottom", color=c["text"], fontsize=15, fontweight="bold")
    ax.axhline(ERR_EMP, color=c["muted"], lw=1.2, ls="--")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 128)
    ax.set_ylabel("sum of squared errors, $n_{tr}=30$")
    clean(ax)
    finish(fig, "lec04-kp06-sure", dark, c)
    return W, H


if __name__ == "__main__":
    print("Stein check, g = tanh, mu = 0, M = %d samples per sigma" % M)
    for s, l, r in stein_numbers():
        print("  sigma %.1f: E(z g) = %.4f   sigma^2 E(g') = %.4f   gap %.4f" % (s, l, r, abs(l - r)))
    rng = np.random.default_rng(3)
    c_, s_ = 2.0, 1.5
    z = rng.normal(0.0, s_, M)
    print("linear g = c z, c = 2, sigma = 1.5: E(z c z) = %.4f, sigma^2 c = %.4f" % (np.mean(z * c_ * z), s_ ** 2 * c_))
    print("SURE example: n_tr = %d, mean MSE = %g (err = %g), sigma^2 = %g" % (N_TR, MSE_TR, ERR_EMP, S2))
    for a, pen, e in sure_numbers():
        print("  mean sensitivity %.1f: penalty 2 s^2 sum = %g, Err = %g - %g + %g = %g (per point %.2f)" % (a, pen, ERR_EMP, N_TR * S2, pen, e, e / N_TR))
    for fn in (stein, sure):
        for dark in (False, True):
            fn(dark)
    print("figures written")
