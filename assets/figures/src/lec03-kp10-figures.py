"""KP10 figures (AdaGrad) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec03-kp10-path, lec03-kp10-eff-step, lec03-kp10-shrink.   Prints every number the lecture and the deck quote.
Run:  py assets/figures/src/lec03-kp10-figures.py
"""
import base64
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, ".agents", "skills", "view-from-text"))
import matplotlib.pyplot as plt  # noqa: E402
from jbd_figure import svg_data_uri, use_jbd_style  # noqa: E402

OUT = os.path.join(ROOT, "assets", "figures")
DARK = dict(text="#f2ede4", muted="#b7b0a5", main="#d97757", ref="#6a9fd6", vec="#5fb8a8", ok="#7dbb8c", warn="#e07a70", edge="#e0995f", grid="#b7b0a5")
LIGHT = dict(text="#1c1712", muted="#5a544b", main="#c4572f", ref="#2b6cb0", vec="#1f8a79", ok="#2f855a", warn="#c0392b", edge="#b7791f", grid="#8a8378", bg="#ffffff")

W0, ETA, ETA_GD, K = np.array([10.0, 2.0]), 1.0, 0.07, 30


def grad(w):
    return np.array([w[0], 25.0 * w[1]])


def adagrad(eta=ETA, k=K, eps=1e-8):
    w, G = W0.copy(), np.zeros(2)
    P, E, GG, Gr = [w.copy()], [], [], []
    for _ in range(k):
        g = grad(w)
        G = G + g * g
        e = eta / np.sqrt(eps + G)
        w = w - e * g
        P.append(w.copy())
        E.append(e)
        GG.append(G.copy())
        Gr.append(g)
    return np.array(P), np.array(E), np.array(GG), np.array(Gr)


def gd(eta=ETA_GD, k=K):
    w, P = W0.copy(), [W0.copy()]
    for _ in range(k):
        w = w - eta * grad(w)
        P.append(w.copy())
    return np.array(P)


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


def clean(ax):
    ax.grid(True, alpha=0.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def axes(W, H):
    fig = plt.figure(figsize=(W / 100, H / 100))
    return fig, fig.add_axes([78 / W, 66 / H, (W - 78 - 24) / W, (H - 66 - 28) / H])


def path_fig(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    P, _, _, _ = adagrad()
    Q = gd()
    W, H = 640, 380
    fig, ax = axes(W, H)
    xs, ys = np.linspace(-0.3, 10.6, 300), np.linspace(-2.3, 2.7, 300)
    X, Y = np.meshgrid(xs, ys)
    ax.contour(X, Y, 0.5 * (X ** 2 + 25 * Y ** 2), levels=[2, 8, 20, 50, 100], colors=c["muted"], linewidths=0.9, alpha=0.6)
    ax.plot(Q[:, 0], Q[:, 1], "-o", color=c["vec"], lw=1.4, ms=3.5, label=rf"GD, $\eta={ETA_GD}$")
    ax.plot(P[:, 0], P[:, 1], "-o", color=c["main"], lw=2.2, ms=4.5, label=rf"AdaGrad, $\eta={ETA:g}$")
    ax.plot([10], [2], "s", color=c["text"], ms=7)
    ax.annotate("start (10, 2)", (10, 2), xytext=(-8, 10), textcoords="offset points", ha="right", color=c["text"], fontsize=13)
    ax.set_xlabel(r"$x$ (flat direction)")
    ax.set_ylabel(r"$y$ (steep direction)")
    ax.set_xlim(-0.3, 10.6)
    ax.set_ylim(-2.3, 2.7)
    ax.legend(loc="lower left", frameon=False, fontsize=13)
    clean(ax)
    finish(fig, "lec03-kp10-path", dark, c)


def eff_fig(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    _, E, _, _ = adagrad()
    W, H = 640, 380
    fig, ax = axes(W, H)
    k = np.arange(1, K + 1)
    ax.plot(k, E[:, 0], "-o", color=c["ref"], lw=2.2, ms=4, label=r"AdaGrad, $x$: $\eta/\sqrt{G_x}$")
    ax.plot(k, E[:, 1], "-o", color=c["main"], lw=2.2, ms=4, label=r"AdaGrad, $y$: $\eta/\sqrt{G_y}$")
    ax.axhline(ETA_GD, color=c["warn"], ls="--", lw=1.4)
    ax.annotate(rf"GD: constant {ETA_GD}", (6, ETA_GD), xytext=(0, 7), textcoords="offset points", ha="left", color=c["warn"], fontsize=13)
    ax.set_yscale("log")
    ax.set_xlabel(r"step $k$")
    ax.set_ylabel("effective step")
    ax.set_xlim(0, K + 1)
    ax.set_ylim(0.012, 0.16)
    ax.set_yticks([0.02, 0.05, 0.1])
    ax.set_yticklabels(["0.02", "0.05", "0.1"])
    ax.minorticks_off()
    ax.legend(loc="upper right", frameon=False, fontsize=13)
    clean(ax)
    finish(fig, "lec03-kp10-eff-step", dark, c)


def shrink_fig(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    k = np.logspace(0, 4, 200)
    ax.plot(k, 1 / np.sqrt(k), color=c["ref"], lw=2.4, label=r"$g=1$ every step: $1/\sqrt{k}$")
    ax.plot(k, 1 / np.sqrt(2500 + k - 1), color=c["main"], lw=2.4, label=r"one early $g=50$, then $g=1$")
    for kk in (100, 10000):
        ax.plot([kk], [1 / math.sqrt(kk)], "o", color=c["ref"], ms=7)
        ax.plot([kk], [1 / math.sqrt(2500 + kk - 1)], "o", color=c["main"], ms=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"step $k$")
    ax.set_ylabel(r"effective step ($\eta=1$)")
    ax.set_xlim(0.8, 14000)
    ax.set_ylim(0.005, 1.4)
    ax.legend(loc="upper right", frameon=False, fontsize=13)
    clean(ax)
    finish(fig, "lec03-kp10-shrink", dark, c)


def numbers():
    P, E, GG, Gr = adagrad()
    np.set_printoptions(precision=6, suppress=True)
    print("== bowl f=0.5(x^2+25y^2), start (10,2), AdaGrad eta=1, eps=1e-8")
    for i in range(3):
        print(f"step {i + 1}: g={Gr[i]}  G={GG[i]}  eff={E[i]}  move={E[i] * Gr[i]}  w_next={P[i + 1]}")
    print("eff at k=10:", E[9], " k=30:", E[29], " ratio y/x at k=1,30:", E[0, 1] / E[0, 0], E[29, 1] / E[29, 0])
    print("w after 5/10/20/30:", P[5], P[10], P[20], P[30])
    print("G_y step1 share after 6 steps:", 2500 / GG[5, 1], " after 30:", 2500 / GG[29, 1], " G_x after 30:", GG[29, 0], " G_y after 30:", GG[29, 1])
    Q = gd()
    print("GD eta=0.07 after 30:", Q[30], " first iterates:", Q[:4].tolist())
    print("== GD stability: limit 2/25 =", 2 / 25)
    for eta in (0.01, 0.07, 0.1):
        fx, fy = 1 - eta, 1 - 25 * eta
        n10 = math.log(0.1) / math.log(abs(fx))
        print(f"eta={eta}: factor x={fx:.3f} y={fy:.3f}  steps for x to fall to 1/10: {math.ceil(n10)}  y after 30: {2 * fy ** 30:.4g}")
    print("== epsilon, first appearance of a tiny gradient g=0.001, eta=1")
    for eps in (1e-8, 1e-6, 1e-4):
        g = 0.001
        print(f"eps={eps:g}: step = {g / math.sqrt(eps + g * g):.4f}")
    print("g=0 first step: with eps=0 -> 0/0 (nan); with eps=1e-8 ->", 0.0 / math.sqrt(1e-8))
    print("== worked example (lecture): g1=(10,50), g2=(9,25)")
    g1, g2 = np.array([10.0, 50.0]), np.array([9.0, 25.0])
    G1, G2 = g1 ** 2, g1 ** 2 + g2 ** 2
    print("G after 1:", G1, " after 2:", G2, " sqrt:", np.sqrt(G2), " eff2:", 1 / np.sqrt(G2), " move2:", g2 / np.sqrt(G2))
    print("== sparse vs frequent, g=1 when present, eta=1")
    cnt = 0
    for t in range(1, 101):
        if t in (1, 50, 100):
            cnt += 1
            print(f"rare coordinate at t={t}: appearance #{cnt}, eff step = {1 / math.sqrt(cnt):.4f};  frequent coordinate eff at t={t}: {1 / math.sqrt(t):.4f}")
    print("== shrink: g=1 const: eff k=100,10000:", 1 / 10, 1 / 100, " with early spike 50:", 1 / math.sqrt(2500 + 99), 1 / math.sqrt(2500 + 9999))
    print("   ratio at k=100:", (1 / math.sqrt(2599)) / 0.1, " k=10000:", (1 / math.sqrt(12499)) / 0.01)


if __name__ == "__main__":
    for dark in (False, True):
        path_fig(dark)
        eff_fig(dark)
        shrink_fig(dark)
    numbers()
