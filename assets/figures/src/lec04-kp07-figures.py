"""KP07 figures (regularization J + alpha*Omega and sparsity) in two skins from the same data.
    assets/figures/<id>.png (light, lecture)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp07-tradeoff, lec04-kp07-shrink.   Prints every number the lecture and the deck quote.
One-weight demo: J(w) = (w - w*)^2 with w* = 2, Omega(w) = w^2  =>  w_dagger = w*/(1+alpha).
Run:  python3 assets/figures/src/lec04-kp07-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

WSTAR = 2.0


def wdag(alpha, wstar=WSTAR):
    return wstar / (1.0 + alpha)


def tradeoff(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    w = np.linspace(-0.3, 3.0, 500)
    a = 1.0
    J = (w - WSTAR) ** 2
    P = a * w ** 2
    T = J + P
    ax.plot(w, J, color=c["ref"], lw=2.4, label=r"$J(w)=(w-w^{\ast})^2$")
    ax.plot(w, P, color=c["warn"], lw=2.4, label=r"$\alpha\Omega(w)=\alpha w^2$, $\alpha=1$")
    ax.plot(w, T, color=c["main"], lw=2.8, label=r"$\tilde J=J+\alpha\Omega$")
    wd = wdag(a)
    ax.plot([WSTAR], [0], "o", color=c["ref"], ms=9)
    ax.plot([wd], [(wd - WSTAR) ** 2 + a * wd ** 2], "o", color=c["main"], ms=9)
    ax.annotate(r"$w^{\ast}=2$", (WSTAR, 0), xytext=(6, 14), textcoords="offset points", color=c["ref"], fontsize=14)
    ax.annotate(r"$w^{\dagger}=1$", (wd, (wd - WSTAR) ** 2 + a * wd ** 2), xytext=(0, 16), textcoords="offset points", color=c["main"], fontsize=14, ha="center")
    ax.set_xlabel(r"weight $w$")
    ax.set_ylabel("objective")
    ax.set_xlim(-0.3, 3.0)
    ax.set_ylim(0, 8)
    ax.legend(loc="upper center", frameon=False, fontsize=12)
    clean(ax)
    finish(fig, "lec04-kp07-tradeoff", dark, c)
    return W, H


def shrink(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    al = np.linspace(0, 10, 400)
    ax.plot(al, wdag(al), color=c["main"], lw=2.8, label=r"$w^{\dagger}=w^{\ast}/(1+\alpha)$")
    ax.axhline(0, color=c["muted"], lw=1.0, ls="--")
    for a, off in ((0.5, (8, 8)), (1.0, (8, 8)), (3.0, (8, 10))):
        ax.plot([a], [wdag(a)], "o", color=c["ref"], ms=8)
        ax.annotate(rf"$\alpha={a:g}$: {wdag(a):.3f}", (a, wdag(a)), xytext=off, textcoords="offset points", color=c["ref"], fontsize=13)
    ax.set_xlabel(r"strength $\alpha$")
    ax.set_ylabel(r"regularized solution $w^{\dagger}$")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.1, 2.2)
    ax.legend(loc="upper right", frameon=False, fontsize=13)
    clean(ax)
    finish(fig, "lec04-kp07-shrink", dark, c)
    return W, H


def report():
    print("balance example: complex model J=10, Omega=4; simple model J=11, Omega=1")
    for a in (0.5, 2):
        print(f"  alpha={a}: complex {10 + a * 4}, simple {11 + a * 1}, gap {10 + a * 4 - (11 + a)}")
    print("break-even alpha: 10+4a = 11+a -> a =", 1 / 3)
    print("one weight: J=(w-2)^2, Omega=w^2")
    for a in (0, 0.5, 1, 3, 9):
        w = wdag(a)
        print(f"  alpha={a}: w_dag={w:.4f}, J={(w - WSTAR) ** 2:.4f}, alpha*Omega={a * w * w:.4f}, total={(w - WSTAR) ** 2 + a * w * w:.4f}")
    # numeric check of the closed form
    ws = np.linspace(-1, 4, 500001)
    for a in (0.5, 1, 3):
        print(f"  grid argmin alpha={a}: {ws[np.argmin((ws - 2) ** 2 + a * ws ** 2)]:.4f} (formula {wdag(a):.4f})")
    # l0 / l1 / l2 of an example vector
    x = np.array([3.0, 0.0, -4.0, 0.0])
    print("x=(3,0,-4,0): l0=", int(np.count_nonzero(x)), " l1=", float(np.abs(x).sum()), " l2=", float(np.sqrt((x ** 2).sum())))
    y = np.array([1.0, 1.0, 1.0, 1.0])
    print("y=(1,1,1,1): l0=", int(np.count_nonzero(y)), " l1=", float(np.abs(y).sum()), " l2=", float(np.sqrt((y ** 2).sum())))


if __name__ == "__main__":
    for dk in (False, True):
        tradeoff(dk)
        shrink(dk)
    report()
