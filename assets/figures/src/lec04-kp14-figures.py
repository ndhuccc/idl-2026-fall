"""KP14 figures (weight decay and input noise) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp14-penalty, lec04-kp14-saturation.   Prints every number the lecture and the deck quote.
Toy problem: one training pair (x, y) = (1.5, 2), model y_hat = w x, squared loss J(w) = (1.5 w - 2)^2, unregularized w* = 4/3.
Weight decay: J + (alpha/2) w^2 with alpha = 0.5.   Input noise eps ~ N(0, sigma^2), sigma^2 = 0.25: E[(w (x+eps) - y)^2] = J(w) + sigma^2 w^2.
Run:  python3 assets/figures/src/lec04-kp14-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

X, Y = 1.5, 2.0
ALPHA, SIG2 = 0.5, 0.25


def J(w):
    return (X * w - Y) ** 2


def wd(w, alpha=ALPHA):
    return 0.5 * alpha * w ** 2


def noise(w, s2=SIG2):
    return s2 * w ** 2


def argmin(f):
    ws = np.linspace(0, 3, 300001)
    return float(ws[np.argmin(f(ws))])


# ------------------------------------------------------------------ figure 1: both penalties give the same objective
def penalty(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    w = np.linspace(0, 3, 400)
    ax.plot(w, J(w), color=c["muted"], lw=2.2, label=r"$J(w)$ only, min at $1.333$")
    ax.plot(w, J(w) + wd(w), color=c["main"], lw=3.0, label=r"$J+\frac{\alpha}{2}w^2$ ($\alpha=0.5$), min at $1.2$")
    ax.plot(w, J(w) + noise(w), color=c["vec"], lw=2.2, ls=(0, (1, 3)), dash_capstyle="round",
            label=r"$J+\sigma^2 w^2$ ($\sigma^2=0.25$), min at $1.2$")
    ax.plot(w, J(w) + noise(w, 0.5), color=c["ref"], lw=2.0, ls="--", label=r"$J+\sigma^2 w^2$ ($\sigma^2=0.5$), min at $1.091$")
    pts = [(argmin(J), c["muted"]), (argmin(lambda v: J(v) + wd(v)), c["main"]), (argmin(lambda v: J(v) + noise(v, 0.5)), c["ref"])]
    for wm, col in pts:
        ax.plot([wm], [0], "o", color=col, ms=9, clip_on=False, zorder=5)
    ax.set_xlabel(r"weight $w$")
    ax.set_ylabel("objective")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 9)
    ax.legend(loc="upper left", bbox_to_anchor=(0.3, 1.0), frameon=False, fontsize=12)
    clean(ax)
    finish(fig, "lec04-kp14-penalty", dark, c)
    return W, H


# ------------------------------------------------------------------ figure 2: small weights keep tanh near its linear part
def saturation(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    x = np.linspace(-3, 3, 600)
    spec = ((0.5, c["vec"]), (1.0, c["ref"]), (4.0, c["main"]))
    ax.axhspan(0.95, 1.25, color=c["warn"], alpha=0.15, lw=0)
    ax.axhspan(-1.25, -0.95, color=c["warn"], alpha=0.15, lw=0)
    for wv, col in spec:
        ax.plot(x, np.tanh(wv * x), color=col, lw=2.6, label=rf"$w={wv:g}$")
    ax.axhline(0.95, color=c["warn"], lw=1.0, ls="--")
    ax.axhline(-0.95, color=c["warn"], lw=1.0, ls="--")
    ax.text(-2.95, 1.13, r"saturated: $|\tanh|>0.95$", color=c["warn"], fontsize=12, va="center")
    ax.set_xlabel(r"input $x$")
    ax.set_ylabel(r"$\tanh(wx)$")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.25, 1.25)
    ax.legend(loc="lower right", frameon=False, fontsize=13)
    clean(ax)
    finish(fig, "lec04-kp14-saturation", dark, c)
    return W, H


def report():
    print("J(w) = (1.5 w - 2)^2, w* = 4/3 =", round(4 / 3, 4))
    print("weight decay penalty at w=2, alpha=0.5:", wd(2.0), " noise penalty at w=2, sigma^2=0.25:", noise(2.0))
    print("minimiser of J + (alpha/2) w^2:", round(argmin(lambda v: J(v) + wd(v)), 4), " closed form 6/5 =", 6 / 5)
    print("minimiser of J + sigma^2 w^2 (0.25):", round(argmin(lambda v: J(v) + noise(v)), 4), " (0.5):", round(argmin(lambda v: J(v) + noise(v, 0.5)), 4), " closed form 3/(2.25+0.5) =", round(3 / 2.75, 4))
    rng = np.random.default_rng(0)
    e = rng.normal(0, np.sqrt(SIG2), 2_000_000)
    print("Monte Carlo E[(2(x+eps)-y)^2] =", round(float(np.mean((2.0 * (X + e) - Y) ** 2)), 3), " theory J(2)+sigma^2*4 =", J(2.0) + 4 * SIG2)
    eta, w, g = 0.1, 2.0, 1.0
    print("one step w <- (1-eta*alpha) w - eta*g:", (1 - eta * ALPHA) * w - eta * g, " check w - eta*(g + alpha w):", w - eta * (g + ALPHA * w))
    thr = np.arctanh(0.95)
    for wv in (0.5, 1.0, 4.0):
        print(f"w={wv}: tanh(wx) exceeds 0.95 for |x|>{thr / wv:.3f}; share of x in [-3,3]: {max(0.0, (3 - thr / wv) / 3):.3f}")


if __name__ == "__main__":
    for dark in (False, True):
        penalty(dark)
        saturation(dark)
    report()
