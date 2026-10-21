"""Weight-decay figure for KP05 (L2 regularization): small weights keep tanh near its linear part, in two skins.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Id: lec04-kp05-saturation.   Prints the numbers the lecture and the deck quote.
Run:  python3 assets/figures/src/lec04-kp05-saturation-figures.py
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


# ------------------------------------------------------------------ figure: small weights keep tanh near its linear part
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
    finish(fig, "lec04-kp05-saturation", dark, c)
    return W, H


def report():
    print("J(w) = (1.5 w - 2)^2, w* = 4/3 =", round(4 / 3, 4))
    print("minimiser of J + (alpha/2) w^2:", round(argmin(lambda v: J(v) + wd(v)), 4), " closed form 6/5 =", 6 / 5)
    eta, w, g = 0.1, 2.0, 1.0
    print("one step w <- (1-eta*alpha) w - eta*g:", (1 - eta * ALPHA) * w - eta * g, " check w - eta*(g + alpha w):", w - eta * (g + ALPHA * w))
    thr = np.arctanh(0.95)
    for wv in (0.5, 1.0, 4.0):
        print(f"w={wv}: tanh(wx) exceeds 0.95 for |x|>{thr / wv:.3f}; share of x in [-3,3]: {max(0.0, (3 - thr / wv) / 3):.3f}")


if __name__ == "__main__":
    for dark in (False, True):
        saturation(dark)
    report()
