"""KP11 figures (RMSProp and Adam) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec03-kp11-effective-step, lec03-kp11-bias-step.   Prints every number the lecture and deck quote.
Toy bowl f(x,y)=0.5(x^2+25y^2) from (10,2).  AdaGrad: G_j=sum g_j^2, w_j <- w_j - eta g_j/sqrt(eps+G_j).
RMSProp: v_j <- gamma v_j + (1-gamma) g_j^2, same update with v in place of G.  eta=0.5, gamma=0.9, eps=1e-8.
Run:  py assets/figures/src/lec03-kp11-figures.py
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

ETA, GAMMA, EPS, T = 0.5, 0.9, 1e-8, 40


def grad(w):
    return np.array([w[0], 25 * w[1]])


def f(w):
    return 0.5 * (w[0] ** 2 + 25 * w[1] ** 2)


def run(kind, steps=T):
    w = np.array([10.0, 2.0])
    acc = np.zeros(2)
    ws, effs, losses = [w.copy()], [], [f(w)]
    for _ in range(steps):
        g = grad(w)
        acc = acc + g * g if kind == "ada" else GAMMA * acc + (1 - GAMMA) * g * g
        eff = ETA / np.sqrt(EPS + acc)
        w = w - eff * g
        ws.append(w.copy())
        effs.append(eff.copy())
        losses.append(f(w))
    return np.array(ws), np.array(effs), np.array(losses)


def adam_1d(g_fn, w0, steps, eta, b1=0.9, b2=0.999, eps=1e-8, correct=True):
    w, m, v = w0, 0.0, 0.0
    rows = []
    for t in range(1, steps + 1):
        g = g_fn(w)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t) if correct else m
        vh = v / (1 - b2 ** t) if correct else v
        step = eta * mh / (math.sqrt(vh) + eps)
        w = w - step
        rows.append(dict(t=t, g=g, m=m, v=v, mh=mh, vh=vh, step=step, w=w))
    return rows


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


def effective_step(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    _, ea, _ = run("ada")
    _, er, _ = run("rms")
    W, H = 640, 380
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([104 / W, 66 / H, (W - 104 - 24) / W, (H - 66 - 28) / H])
    k = np.arange(1, T + 1)
    ax.semilogy(k, ea[:, 0], color=c["ref"], lw=2.6, label="AdaGrad")
    ax.semilogy(k, er[:, 0], color=c["main"], lw=2.6, label="RMSProp")
    for arr, col, dy in ((ea, c["ref"], 14), (er, c["main"], -26)):
        ax.plot([T], [arr[-1, 0]], "o", color=col, ms=8)
        ax.annotate(f"{arr[-1, 0]:.4f}", (T, arr[-1, 0]), xytext=(-8, dy), textcoords="offset points", ha="right", color=col, fontsize=14)
    kmin = int(np.argmin(er[:, 0])) + 1
    ax.plot([kmin], [er[kmin - 1, 0]], "o", color=c["main"], ms=8)
    ax.annotate(f"min {er[kmin - 1, 0]:.4f}", (kmin, er[kmin - 1, 0]), xytext=(6, -24), textcoords="offset points", color=c["main"], fontsize=14)
    ax.set_xlabel(r"step $k$")
    ax.set_ylabel("effective step size (x)")
    ax.set_xlim(0, T + 1)
    ax.set_ylim(0.006, 1.0)
    ax.legend(loc="upper left", frameon=False, fontsize=14)
    clean(ax)
    finish(fig, "lec03-kp11-effective-step", dark, c)
    return W, H


def bias_ratio(t, b1=0.9, b2=0.999):
    return (1 - b1 ** t) / math.sqrt(1 - b2 ** t)


def bias_step(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([78 / W, 66 / H, (W - 78 - 24) / W, (H - 66 - 28) / H])
    t = np.unique(np.round(np.logspace(0, math.log10(3000), 400)).astype(int))
    r = np.array([bias_ratio(int(i)) for i in t])
    ax.semilogx(t, r, color=c["warn"], lw=2.6, label="without correction")
    ax.axhline(1, color=c["ok"], ls="--", lw=2.0, label="with correction")
    ip = int(np.argmax(r))
    ax.plot([1], [r[0]], "o", color=c["warn"], ms=8)
    ax.annotate(f"t=1: {r[0]:.2f}", (1, r[0]), xytext=(8, -24), textcoords="offset points", color=c["warn"], fontsize=14)
    ax.plot([t[ip]], [r[ip]], "o", color=c["warn"], ms=8)
    ax.annotate(f"peak {r[ip]:.2f} at t={t[ip]}", (t[ip], r[ip]), xytext=(12, 8), textcoords="offset points", color=c["warn"], fontsize=14)
    ax.set_xlabel(r"step $t$ (log scale)")
    ax.set_ylabel(r"step size / $\eta$")
    ax.set_xlim(0.9, 3500)
    ax.set_ylim(0, 8.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.76), frameon=False, fontsize=14)
    clean(ax)
    finish(fig, "lec03-kp11-bias-step", dark, c)
    return W, H


if __name__ == "__main__":
    for dark in (False, True):
        effective_step(dark)
        bias_step(dark)
    print(f"bowl: eta={ETA} gamma={GAMMA} eps={EPS} start=(10,2) steps={T}")
    for kind in ("ada", "rms"):
        ws, ef, ls = run(kind)
        print(kind, "w_k:", {k: np.round(ws[k], 3).tolist() for k in (1, 2, 3, 5, 10, 20, 40)})
        print(kind, "eff_x:", {k: round(float(ef[k - 1, 0]), 4) for k in (1, 2, 3, 5, 10, 20, 40)}, "argmin k", int(np.argmin(ef[:, 0])) + 1, "min", round(float(ef[:, 0].min()), 4))
        print(kind, "eff_y:", {k: round(float(ef[k - 1, 1]), 4) for k in (1, 2, 3, 5, 10, 20, 40)})
        print(kind, "loss:", {k: round(float(ls[k]), 4) for k in (0, 1, 5, 10, 20, 40)})
    print("RMSProp loss at 100,150,200 more steps:", [round(float(x), 5) for x in run("rms", 200)[2][[100, 150, 200]]])
    print("AdaGrad loss at 100,150,200:", [round(float(x), 5) for x in run("ada", 200)[2][[100, 150, 200]]])
    ETA = 1.0  # KP10 used eta=1: RMSProp on the same bowl bounces much more
    print("eta=1 (as in KP10): AdaGrad step 1:", np.round(run("ada", 1)[0][1], 3).tolist(), " RMSProp loss at 1,5,20:", [round(float(x), 3) for x in run("rms", 20)[2][[1, 5, 20]]], " RMSProp first x move:", round(float(10 - run("rms", 1)[0][1][0]), 3))
    ETA = 0.5
    print("eff x ratio rms/ada at k=40:", round(float(run("rms")[1][-1, 0] / run("ada")[1][-1, 0]), 1))
    print("weights gamma^k, gamma=0.9:", {k: round(0.9 ** k, 4) for k in (1, 5, 10, 22)})
    for g in (0.9, 0.99, 0.999):
        print(f"gamma={g}: window 1/(1-g)={1 / (1 - g):.0f}, half-life ln0.5/ln g={math.log(0.5) / math.log(g):.2f}")
    # EMA of squared gradient when g drops from 10 to 1 after 20 steps
    v, vs = 0.0, {}
    for t in range(1, 41):
        v = 0.9 * v + 0.1 * (100 if t <= 20 else 1)
        vs[t] = v
    print("v (g=10 for 20 steps, then g=1):", {t: round(vs[t], 3) for t in (20, 21, 25, 30, 40)}, " AdaGrad G would be", 20 * 100 + 1 * 20)
    print("Adam 1-D (g = x, x0 = 10, eta = 0.5, b1=.9, b2=.999):")
    for corr in (True, False):
        for r_ in adam_1d(lambda x: x, 10.0, 3, 0.5, correct=corr):
            print("  corrected  " if corr else "  uncorrected", {k: round(float(v_), 5) for k, v_ in r_.items()})
    print("first-step uncorrected ratio b1=.9,b2=.999:", round(bias_ratio(1), 4), "= 0.1/sqrt(0.001) =", round(0.1 / math.sqrt(0.001), 4))
    print("first-step uncorrected ratio b1=.9,b2=.9:", round(bias_ratio(1, 0.9, 0.9), 4), "= 0.1/sqrt(0.1)")
    ts = list(range(1, 3001))
    rr = [bias_ratio(i) for i in ts]
    ip = int(np.argmax(rr))
    print("peak ratio", round(rr[ip], 3), "at t", ts[ip], " ratio at t=10,50,100,1000,3000:", [round(bias_ratio(i), 3) for i in (10, 50, 100, 1000, 3000)])
    print("m factors 1/(1-0.9^t) t=1,2,10,50:", [round(1 / (1 - 0.9 ** i), 4) for i in (1, 2, 10, 50)], " v factors 1/(1-0.999^t) t=1,2,3000:", [round(1 / (1 - 0.999 ** i), 2) for i in (1, 2, 3000)])
    print("m example g=4, t=1: m=", 0.1 * 4, "mhat=", 0.1 * 4 / (1 - 0.9))
    print("Adam on bowl (both coords), eta=0.5, first 3 steps:")
    w = np.array([10.0, 2.0])
    m = np.zeros(2)
    v = np.zeros(2)
    for t in range(1, 4):
        g = grad(w)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w = w - 0.5 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        print("  t", t, "w", np.round(w, 4).tolist(), "loss", round(float(f(w)), 4))
