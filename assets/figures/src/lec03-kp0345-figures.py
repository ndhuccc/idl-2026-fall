"""KP03-KP05 figures (stopping criteria, line search, momentum) on the toy bowl f(x, y) = (x^2 + 25 y^2) / 2 and on f(w) = 5 w^2 - 4 w + 1.

Each figure is drawn in two skins from the SAME data:
    assets/figures/<id>.png   light skin (lectures)      assets/figures/<id>.svg   dark skin (deck canvas)
Ids: lec03-kp03-meters, lec03-kp03-saddle, lec03-kp04-backtracking, lec03-kp04-zigzag, lec03-kp05-paths.
Run:  py assets/figures/src/lec03-kp0345-figures.py   (prints every number the lectures and decks quote)
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
LAM = np.array([1.0, 25.0])
TH0 = np.array([10.0, 2.0])
ETA = 0.06
DARK = dict(text="#f2ede4", muted="#b7b0a5", main="#d97757", ref="#6a9fd6", vec="#5fb8a8", ok="#7dbb8c", warn="#e07a70", edge="#e0995f", grid="#b7b0a5")
LIGHT = dict(text="#1c1712", muted="#5a544b", main="#c4572f", ref="#2b6cb0", vec="#1f8a79", ok="#2f855a", warn="#c0392b", edge="#b7791f", grid="#8a8378", bg="#ffffff")
STATS = {}


def f2(p):
    return 0.5 * float(np.sum(LAM * p * p))


def g2(p):
    return LAM * p


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


def gd_path(k, eta=ETA):
    p = [TH0.copy()]
    for _ in range(k):
        p.append(p[-1] - eta * g2(p[-1]))
    return np.array(p)


def mom_path(k, eta=ETA, alpha=0.8):
    p, v = [TH0.copy()], np.zeros(2)
    for _ in range(k):
        v = alpha * v - eta * g2(p[-1])
        p.append(p[-1] + v)
    return np.array(p)


# ------------------------------------------------------------------ KP03
def meters(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    K, eps = 80, 0.2
    path = gd_path(K)
    gn = np.array([np.linalg.norm(g2(p)) for p in path])
    df = np.array([abs(f2(path[k + 1]) - f2(path[k])) for k in range(K)])
    k_df = int(np.argmax(df <= eps))
    k_gn = int(np.argmax(gn <= eps))
    STATS["meters"] = dict(eps=eps, k_df=k_df, k_gn=k_gn, g0=gn[0], g30=gn[30], df0=df[0], df10=df[10], df30=df[30], gn10=gn[10], gn_at_kdf=gn[k_df],
                           df_at_kgn=df[k_gn - 1], f_at_kdf=f2(path[k_df]))
    W, H = 640, 400
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([86 / W, 70 / H, (W - 86 - 24) / W, (H - 70 - 28) / H])
    ax.semilogy(range(K + 1), gn, color=c["main"], lw=2.4, label=r"$\|\nabla f\|$")
    ax.semilogy(range(1, K + 1), df, color=c["ref"], lw=2.4, label=r"$|f_{k+1}-f_k|$")
    ax.axhline(eps, color=c["muted"], ls="--", lw=1.3)
    ax.annotate(r"$\varepsilon=0.2$", (K * 0.62, eps), xytext=(0, 6), textcoords="offset points", color=c["muted"], fontsize=14)
    ax.plot([k_df + 1], [df[k_df]], "o", color=c["ref"], ms=8)
    ax.plot([k_gn], [gn[k_gn]], "o", color=c["main"], ms=8)
    ax.set_xlabel(r"step $k$")
    ax.set_xlim(0, K)
    ax.set_ylim(1e-3, 80)
    ax.legend(loc="upper right", frameon=False, fontsize=14)
    clean(ax)
    finish(fig, "lec03-kp03-meters", dark, c)


def saddle(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 360
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([86 / W, 70 / H, (W - 86 - 24) / W, (H - 70 - 28) / H])
    t = np.linspace(-1.5, 1.5, 300)
    ax.plot(t, t ** 2, color=c["ref"], lw=2.6, label=r"$f(x,0)=x^2$")
    ax.plot(t, -t ** 2, color=c["warn"], lw=2.6, label=r"$f(0,y)=-y^2$")
    ax.plot([0], [0], "o", color=c["text"], ms=9)
    ax.annotate(r"$\nabla f(0,0)=(0,0)$", (0, 0), xytext=(14, 8), textcoords="offset points", color=c["text"], fontsize=14)
    ax.set_xlabel(r"$x$ or $y$")
    ax.set_ylabel(r"$f$")
    ax.set_ylim(-2.6, 2.6)
    ax.legend(loc="upper center", frameon=False, fontsize=14, ncol=2)
    clean(ax)
    finish(fig, "lec03-kp03-saddle", dark, c)


# ------------------------------------------------------------------ KP04
def fw(w):
    return 5 * w * w - 4 * w + 1


def backtracking(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    w0, g = Fr(2), Fr(16)
    rows = []
    for eta in (Fr(1), Fr(1, 2), Fr(1, 4), Fr(1, 8)):
        w = w0 - eta * g
        rows.append(dict(eta=str(eta), w=float(w), f=float(fw(w)), armijo=float(fw(w0) - Fr(1, 10) * eta * g * g), ok=bool(fw(w) <= fw(w0) - Fr(1, 10) * eta * g * g)))
    STATS["backtracking"] = rows
    W, H = 640, 400
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([86 / W, 70 / H, (W - 86 - 24) / W, (H - 70 - 28) / H])
    e = np.linspace(0, 0.62, 300)
    ax.plot(e, [float(fw(w0 - Fr(float(x)) * g)) for x in e], color=c["ref"], lw=2.6, label=r"$f(w-\eta g)$")
    ax.plot(e, 13 - 0.1 * e * 256, color=c["muted"], ls="--", lw=1.6, label=r"Armijo line, $c=0.1$")
    for r in rows[1:]:
        x = float(Fr(r["eta"]))
        ax.plot([x], [r["f"]], "o", color=c["ok"] if r["ok"] else c["warn"], ms=10)
        ax.annotate(rf"$\eta={r['eta']}$", (x, r["f"]), xytext=(-62 if r["eta"] == "1/2" else 8, 8), textcoords="offset points", color=c["ok"] if r["ok"] else c["warn"], fontsize=13)
    ax.text(0.02, 150, r"$\eta=1$: $f=1037$ (off chart)", color=c["warn"], fontsize=13, ha="left")
    ax.set_xlabel(r"trial step $\eta$")
    ax.set_ylabel(r"$f(w_0-\eta g)$")
    ax.set_xlim(0, 0.62)
    ax.set_ylim(-10, 230)
    ax.legend(loc="upper left", frameon=False, fontsize=13)
    clean(ax)
    finish(fig, "lec03-kp04-backtracking", dark, c)


def exact_ls_path(K):
    p = [TH0.copy()]
    gs = []
    for _ in range(K):
        g = g2(p[-1])
        gs.append(g)
        eta = float(g @ g) / float(g @ (LAM * g))
        p.append(p[-1] - eta * g)
    return np.array(p), gs


def zigzag(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    K = 8
    path, gs = exact_ls_path(K)
    g0, g1 = gs[0], gs[1]
    STATS["zigzag"] = dict(eta0=float(g0 @ g0) / float(g0 @ (LAM * g0)), th1=path[1].tolist(), th2=path[2].tolist(), g0=g0.tolist(), g1=g1.tolist(), dot01=float(g0 @ g1),
                           f=[f2(q) for q in path[:K + 1]])
    W, H = 640, 330
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([78 / W, 66 / H, (W - 78 - 24) / W, (H - 66 - 26) / H])
    xr, yr = (-1.0, 11.0), (-2.7, 2.7)
    g1m, g2m = np.meshgrid(np.linspace(*xr, 500), np.linspace(*yr, 300))
    ax.contour(g1m, g2m, 0.5 * (LAM[0] * g1m ** 2 + LAM[1] * g2m ** 2), levels=[1, 4, 12, 30, 60, 100], colors=c["muted"], linewidths=1.2, alpha=0.6)
    ax.plot(path[:, 0], path[:, 1], "-o", color=c["main"], lw=1.8, ms=5)
    ax.plot(0, 0, marker="*", color=c["ok"], ms=14, ls="none")
    ax.plot(*TH0, marker="o", color=c["text"], ms=8, ls="none", zorder=5)
    ax.annotate(r"$\theta_0$", TH0, xytext=(-14, 6), textcoords="offset points", color=c["text"], fontsize=15)
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([-2, 0, 2])
    clean(ax)
    finish(fig, "lec03-kp04-zigzag", dark, c)


# ------------------------------------------------------------------ KP05
def paths(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    K = 40
    gd, mo = gd_path(K), mom_path(K)
    STATS["paths"] = dict(gd=[f2(gd[k]) for k in (1, 2, 10, 20, 40)], mom=[f2(mo[k]) for k in (1, 2, 10, 20, 40)], mo1=mo[1].tolist(), mo2=mo[2].tolist(),
                          gd2=gd[2].tolist(), steady=ETA / (1 - 0.8))
    W, H = 640, 330
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([78 / W, 66 / H, (W - 78 - 24) / W, (H - 66 - 26) / H])
    xr, yr = (-1.0, 11.0), (-2.7, 2.7)
    g1m, g2m = np.meshgrid(np.linspace(*xr, 500), np.linspace(*yr, 300))
    ax.contour(g1m, g2m, 0.5 * (LAM[0] * g1m ** 2 + LAM[1] * g2m ** 2), levels=[1, 4, 12, 30, 60, 100], colors=c["muted"], linewidths=1.2, alpha=0.6)
    ax.plot(gd[:, 0], gd[:, 1], "-o", color=c["warn"], lw=1.6, ms=3.5, label="gradient descent")
    ax.plot(mo[:, 0], mo[:, 1], "-o", color=c["vec"], lw=1.8, ms=3.5, label=r"momentum, $\alpha=0.8$")
    ax.plot(0, 0, marker="*", color=c["ok"], ms=14, ls="none")
    ax.plot(*TH0, marker="o", color=c["text"], ms=8, ls="none", zorder=5)
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([-2, 0, 2])
    ax.legend(loc="lower left", frameon=False, fontsize=13)
    clean(ax)
    finish(fig, "lec03-kp05-paths", dark, c)


def stability_cases():
    rows = []
    for eta, alpha in ((0.06, 0.0), (0.06, 0.8), (0.09, 0.0), (0.09, 0.8), (0.2, 0.8), (0.2, 0.95)):
        p = mom_path(40, eta, alpha)
        rows.append((eta, alpha, float(np.linalg.norm(p[-1])), f2(p[-1]) if np.isfinite(f2(p[-1])) else float("inf")))
    return rows


if __name__ == "__main__":
    for dark in (False, True):
        meters(dark)
        saddle(dark)
        backtracking(dark)
        zigzag(dark)
        paths(dark)
    for k, v in STATS.items():
        print(k, v)
    print("stability (eta, alpha, |theta_40|, f_40):")
    for r in stability_cases():
        print("  ", r)
    print("limits: GD 2/lmax =", 2 / 25, " momentum alpha=0.8 2(1+a)/lmax =", 2 * 1.8 / 25, " alpha=0.95:", 2 * 1.95 / 25)
    print("rate (kappa-1)/(kappa+1) for kappa=25:", 24 / 26, " 0.923^100 =", (24 / 26) ** 100, " optimal eta 2/(1+25)=", 2 / 26)
    print("y factor at eta=0.09:", 1 - 0.09 * 25, "  x factor", 1 - 0.09)
