"""KP02 (Gradient descent and learning rate) figures on f(w) = 5 w^2 - 4 w + 1 (minimum w* = 0.4, f* = 0.2).

Two figures in two skins from the SAME data:
    assets/figures/<id>.png   light skin, used by lectures/lec03-kp02-gradient-descent-and-learning-rate.md
    assets/figures/<id>.svg   dark skin (deck canvas), embedded in slides/lec03-kp02-gradient-descent-and-learning-rate.jbd.json
Figure ids: lec03-kp02-one-step, lec03-kp02-step-regimes.  All numbers are computed here with exact fractions.
Run:  py assets/figures/src/lec03-kp02-lr-figures.py
"""
import base64
import os
import sys
from fractions import Fraction as Fr

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, ".agents", "skills", "view-from-text"))
import matplotlib.pyplot as plt  # noqa: E402
from jbd_figure import svg_data_uri, use_jbd_style  # noqa: E402

OUT = os.path.join(ROOT, "assets", "figures")
DARK = dict(text="#f2ede4", muted="#b7b0a5", main="#d97757", ref="#6a9fd6", vec="#5fb8a8", ok="#7dbb8c", warn="#e07a70", edge="#e0995f", grid="#b7b0a5")
LIGHT = dict(text="#1c1712", muted="#5a544b", main="#c4572f", ref="#2b6cb0", vec="#1f8a79", ok="#2f855a", warn="#c0392b", edge="#b7791f", grid="#8a8378", bg="#ffffff")


def f(w):
    return 5 * w * w - 4 * w + 1


def fp(w):
    return 10 * w - 4


def skin(c, dark):
    use_jbd_style(18)
    plt.rcParams.update({"text.color": c["text"], "axes.labelcolor": c["text"], "axes.edgecolor": c["muted"], "xtick.color": c["muted"],
                        "ytick.color": c["muted"], "grid.color": c["grid"], "svg.fonttype": "path",
                        "figure.facecolor": "none" if dark else c["bg"], "axes.facecolor": "none" if dark else c["bg"], "savefig.transparent": dark})


def finish(fig, key, dark, c):
    if dark:
        uri = svg_data_uri(fig)
        with open(os.path.join(OUT, key + ".svg"), "wb") as fh:
            fh.write(base64.b64decode(uri.split(",", 1)[1]))
    else:
        fig.savefig(os.path.join(OUT, key + ".png"), dpi=160, facecolor=c["bg"])
    plt.close(fig)


def one_step(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 560, 440
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([86 / W, 70 / H, (W - 86 - 26) / W, (H - 70 - 28) / H])
    w = np.linspace(-0.4, 2.3, 400)
    ax.plot(w, f(w), color=c["ref"], lw=2.6)
    w0, w1, ws = 2.0, 2.0 - 0.05 * fp(2.0), 0.4
    ax.plot([w0], [f(w0)], "o", color=c["main"], ms=9)
    ax.plot([w1], [f(w1)], "o", color=c["vec"], ms=9)
    ax.plot([ws], [f(ws)], "*", color=c["ok"], ms=15)
    ax.annotate("", xy=(w1, f(w0) + 1.6), xytext=(w0, f(w0) + 1.6), arrowprops=dict(arrowstyle="-|>", color=c["vec"], lw=2.4, mutation_scale=18))
    ax.annotate(r"$w_0=2$, $f=13$", (w0, f(w0)), xytext=(-12, -6), textcoords="offset points", ha="right", color=c["main"], fontsize=15)
    ax.annotate(r"$w_1=1.2$, $f=3.4$", (w1, f(w1)), xytext=(10, -22), textcoords="offset points", color=c["vec"], fontsize=15)
    ax.annotate(r"$w^*=0.4$, $f^*=0.2$", (ws, f(ws)), xytext=(6, -30), textcoords="offset points", color=c["ok"], fontsize=15)
    ax.set_xlim(-0.4, 2.3)
    ax.set_ylim(-4, 24)
    ax.set_xlabel(r"$w$")
    ax.set_ylabel(r"$f(w)$")
    ax.grid(True, alpha=0.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    finish(fig, "lec03-kp02-one-step", dark, c)
    return W, H


def regimes(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 560, 400
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([92 / W, 74 / H, (W - 92 - 170) / W, (H - 74 - 26) / H])
    t = np.arange(7)
    for eta, col in ((Fr(1, 50), c["ref"]), (Fr(1, 10), c["ok"]), (Fr(1, 4), c["warn"])):
        e = [float(Fr(8, 5) * (1 - 10 * eta) ** k) for k in t]
        ax.plot(t, e, "o-", color=col, lw=2.3, ms=5.5, label=rf"$\eta={float(eta):.2f}$")
    ax.axhline(0, color=c["muted"], lw=1, ls="--")
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel(r"error $w_t-w^*$ (symlog)")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(-30, 30)
    ax.set_yticks([-10, -1, 0, 1, 10])
    ax.set_yticklabels(["-10", "-1", "0", "1", "10"])
    ax.set_xlim(-0.2, 6.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=14, ncol=1, handlelength=1.3, handletextpad=0.5)
    ax.grid(True, alpha=0.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    finish(fig, "lec03-kp02-step-regimes", dark, c)
    return W, H


if __name__ == "__main__":
    for dark in (False, True):
        s1 = one_step(dark)
        s2 = regimes(dark)
    print("sizes", s1, s2)
    w = Fr(2)
    print("f'(2)", fp(w), "f(2)", f(w), "w1", w - Fr(1, 20) * fp(w), "f(w1)", f(w - Fr(1, 20) * fp(w)), "f(0.4)", f(Fr(2, 5)))
    for eta in (Fr(1, 50), Fr(1, 10), Fr(1, 4)):
        fac = 1 - 10 * eta
        print("eta", float(eta), "factor", float(fac), "e_t", [float(Fr(8, 5) * fac ** k) for k in range(6)], "gap_t", [float(5 * (Fr(8, 5) * fac ** k) ** 2) for k in range(3)])
    w = Fr(2)
    row = []
    for k in range(6):
        row.append((k, float(w), float(f(w))))
        w = w - Fr(1, 20) * fp(w)
    print("eta=0.05 iterates", row)
    for eta, label in ((Fr(1, 10), "0.10"), (Fr(1), "1"), (Fr(11, 10), "1.10")):
        w = Fr(4)
        seq = []
        for k in range(4):
            seq.append((float(w), float(w * w)))
            w = w - eta * 2 * w
        print("w^2 eta", label, seq)
    print("L=10: 1/L", 1 / 10, "2/L", 2 / 10)
