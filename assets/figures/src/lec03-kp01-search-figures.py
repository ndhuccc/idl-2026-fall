"""KP01 (Optimization as parameter search) figures: one toy bowl f(x, y) = (x^2 + 25 y^2) / 2, theta_0 = (10, 2).

Two figures, each in two skins from the SAME data:
    assets/figures/<id>.png   light skin (white page), used by lectures/lec03-kp01-optimization-as-search.md
    assets/figures/<id>.svg   dark skin (deck canvas #151513), embedded in slides/lec03-kp01-optimization-as-search.jbd.json
Figure ids: lec03-kp01-contours, lec03-kp01-newton-gd.
Every number printed at the end is computed here; the lecture and the deck quote those numbers.
Run:  py assets/figures/src/lec03-kp01-search-figures.py
"""
import base64
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, ".agents", "skills", "view-from-text"))
import matplotlib.pyplot as plt  # noqa: E402
from jbd_figure import svg_data_uri, use_jbd_style  # noqa: E402

OUT = os.path.join(ROOT, "assets", "figures")
LAM = np.array([1.0, 25.0])
TH0 = np.array([10.0, 2.0])
ETA = 0.06

DARK = dict(text="#f2ede4", muted="#b7b0a5", main="#d97757", ref="#6a9fd6", vec="#5fb8a8", ok="#7dbb8c", warn="#e07a70", grid="#b7b0a5", bg="#151513")
LIGHT = dict(text="#1c1712", muted="#5a544b", main="#c4572f", ref="#2b6cb0", vec="#1f8a79", ok="#2f855a", warn="#c0392b", grid="#8a8378", bg="#ffffff")


def f(p):
    return 0.5 * float(np.sum(LAM * p * p))


def grad(p):
    return LAM * p


def gd_path(k):
    p = [TH0.copy()]
    for _ in range(k):
        p.append(p[-1] - ETA * grad(p[-1]))
    return np.array(p)


def skin(c, dark):
    use_jbd_style(18)
    plt.rcParams.update({"text.color": c["text"], "axes.labelcolor": c["text"], "axes.edgecolor": c["muted"], "xtick.color": c["muted"],
                        "ytick.color": c["muted"], "grid.color": c["grid"], "svg.fonttype": "path",
                        "figure.facecolor": "none" if dark else c["bg"], "axes.facecolor": "none" if dark else c["bg"],
                        "savefig.transparent": dark})


def finish(fig, key, dark, c):
    if dark:
        uri = svg_data_uri(fig)
        with open(os.path.join(OUT, key + ".svg"), "wb") as fh:
            fh.write(base64.b64decode(uri.split(",", 1)[1]))
    else:
        fig.savefig(os.path.join(OUT, key + ".png"), dpi=160, facecolor=c["bg"])
    plt.close(fig)


def contours(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 470
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([78 / W, 66 / H, (W - 78 - 24) / W, (H - 66 - 26) / H])
    xr, yr = (-1.0, 11.0), (-4.2, 4.2)
    g1, g2 = np.meshgrid(np.linspace(*xr, 500), np.linspace(*yr, 300))
    F = 0.5 * (LAM[0] * g1 ** 2 + LAM[1] * g2 ** 2)
    ax.contour(g1, g2, F, levels=[4, 12, 30, 60, 100, 160], colors=c["muted"], linewidths=1.3, alpha=0.65)
    g = grad(TH0)
    n = g / np.linalg.norm(g)
    th1 = TH0 - ETA * g
    ax.plot(0, 0, marker="*", color=c["ok"], ms=14, ls="none")
    ax.annotate("", xy=TH0 + 2.0 * n, xytext=TH0, arrowprops=dict(arrowstyle="-|>", color=c["warn"], lw=2.6, mutation_scale=20))
    ax.annotate("", xy=TH0 - 2.0 * n, xytext=TH0, arrowprops=dict(arrowstyle="-|>", color=c["vec"], lw=2.6, mutation_scale=20))
    ax.plot(*TH0, marker="o", color=c["text"], ms=8, ls="none", zorder=5)
    ax.plot(*th1, marker="o", mfc="none", mec=c["main"], mew=2.2, ms=9, ls="none", zorder=5)
    ax.annotate(r"$\theta_t=\theta_0$", TH0, xytext=(-12, 2), textcoords="offset points", ha="right", color=c["text"], fontsize=15)
    ax.annotate(r"$\nabla f$ (uphill)", TH0 + 2.0 * n, xytext=(-8, 0), textcoords="offset points", ha="right", color=c["warn"], fontsize=14)
    ax.annotate(r"$-\nabla f$ (downhill)", TH0 - 2.0 * n, xytext=(-8, -16), textcoords="offset points", ha="right", color=c["vec"], fontsize=14)
    ax.annotate(r"$\theta_{t+1}$", th1, xytext=(-8, -20), textcoords="offset points", ha="right", color=c["main"], fontsize=15)
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([-2, 0, 2])
    ax.grid(True, alpha=0.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    finish(fig, "lec03-kp01-contours", dark, c)
    return W, H, dict(grad=g.tolist(), th1=th1.tolist(), f0=f(TH0), f1=f(th1))


def newton_gd(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    path = gd_path(30)
    W, H = 1160, 340
    ml, mb, mr, mt, gap = 70, 74, 20, 36, 80
    pw = (W - ml - mr - gap) / 2
    xr, yr = (-1.0, 11.0), (-2.7, 2.7)
    ph = pw * (yr[1] - yr[0]) / (xr[1] - xr[0])
    fig = plt.figure(figsize=(W / 100, H / 100))
    axes = [fig.add_axes([(ml + i * (pw + gap)) / W, mb / H, pw / W, ph / H]) for i in range(2)]
    g1, g2 = np.meshgrid(np.linspace(*xr, 500), np.linspace(*yr, 300))
    F = 0.5 * (LAM[0] * g1 ** 2 + LAM[1] * g2 ** 2)
    for ax in axes:
        ax.contour(g1, g2, F, levels=[4, 12, 30, 60, 100], colors=c["muted"], linewidths=1.3, alpha=0.6)
        ax.plot(0, 0, marker="*", color=c["ok"], ms=14, ls="none")
        ax.plot(*TH0, marker="o", color=c["text"], ms=8, ls="none", zorder=5)
        ax.annotate(r"$\theta_0=(10,\,2)$", TH0, xytext=(-6, -26), textcoords="offset points", ha="right", color=c["text"], fontsize=15)
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x$")
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([-2, 0, 2])
        ax.grid(True, alpha=0.18)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    a, b = axes
    a.set_ylabel(r"$y$")
    a.annotate("", xy=(0.25, 0.04), xytext=TH0, arrowprops=dict(arrowstyle="-|>", color=c["ref"], lw=2.8, mutation_scale=20))
    a.set_title(r"curvature reshapes the step: $-H^{-1}\nabla f$", color=c["text"], fontsize=15)
    n = grad(TH0) / np.linalg.norm(grad(TH0))
    b.annotate("", xy=TH0 - 2.6 * n, xytext=TH0, arrowprops=dict(arrowstyle="-|>", color=c["vec"], lw=2.8, mutation_scale=20))
    b.plot(path[:13, 0], path[:13, 1], "-o", color=c["main"], lw=1.7, ms=4.2)
    b.set_title(r"direction of $-\nabla f$, then 12 steps with $\eta=0.06$", color=c["text"], fontsize=15)
    finish(fig, "lec03-kp01-newton-gd", dark, c)
    return W, H, dict(f1=f(path[1]), f10=f(path[10]), f30=f(path[30]), th2=path[2].tolist())


def cost_numbers():
    rows = []
    for d in (10 ** 2, 10 ** 4, 10 ** 6):
        rows.append(dict(d=d, g_bytes=4 * d, h_bytes=4 * d * d, ops=d ** 3, secs=d ** 3 / 1e13))
    return rows


if __name__ == "__main__":
    for dark in (False, True):
        s1 = contours(dark)
        s2 = newton_gd(dark)
    print("contours", s1)
    print("newton_gd", s2)
    for r in cost_numbers():
        print(r)
    h = np.diag(LAM)
    newton = TH0 - np.linalg.solve(h, grad(TH0))
    print("grad0", grad(TH0), "newton step ->", newton, "f(newton)", f(newton))
    print("stability limit 2/lambda_max =", 2 / LAM.max())
