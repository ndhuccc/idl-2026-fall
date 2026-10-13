"""KP06-KP08 figures (backpropagation, SGD, mini-batch) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec03-kp06-sigmoid, lec03-kp07-sgd-noise, lec03-kp08-batch-std.   Prints every number the lectures and decks quote.
Run:  py assets/figures/src/lec03-kp0608-figures.py
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
STATS = {}


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


def sig(a):
    return 1 / (1 + np.exp(-a))


# ------------------------------------------------------------------ KP06
def sigmoid(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 600, 360
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([78 / W, 66 / H, (W - 78 - 24) / W, (H - 66 - 28) / H])
    a = np.linspace(-8, 8, 400)
    ax.plot(a, sig(a), color=c["ref"], lw=2.6, label=r"$\sigma(a)$")
    ax.plot(a, sig(a) * (1 - sig(a)), color=c["main"], lw=2.6, label=r"$\sigma'(a)$")
    for x, col in ((0.5, c["vec"]), (5, c["warn"])):
        y = float(sig(x) * (1 - sig(x)))
        ax.plot([x], [y], "o", color=col, ms=9)
        ax.annotate(rf"$\sigma'({x:g})={y:.4f}$", (x, y), xytext=(-10 if x == 5 else 10, 14), ha="right" if x == 5 else "left", textcoords="offset points", color=col, fontsize=13)
    ax.set_xlabel(r"pre-activation $a$")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="center left", frameon=False, fontsize=14)
    clean(ax)
    finish(fig, "lec03-kp06-sigmoid", dark, c)
    return W, H


def kp06_numbers():
    # linear example from the source
    x, w1, w2, y = 1.0, 2.0, 3.0, 0.0
    a = w1 * x
    yh = w2 * a
    L = 0.5 * (yh - y) ** 2
    d_out = yh - y
    g2, g1 = d_out * a, d_out * w2 * x
    eps = 1e-3
    Lf = lambda w: 0.5 * (w2 * w * x - y) ** 2
    cd1 = (Lf(w1 + eps) - Lf(w1 - eps)) / (2 * eps)
    STATS["linear"] = dict(a=a, yh=yh, L=L, d_out=d_out, g2=g2, g1=g1, cd_w1=cd1)
    # sigmoid hidden unit
    x, w1, w2, y = 1.0, 0.5, 2.0, 1.0
    a = w1 * x
    z = float(sig(a))
    yh = w2 * z
    L = 0.5 * (yh - y) ** 2
    d_out = yh - y
    gw2 = d_out * z
    sp = z * (1 - z)
    d_h = sp * w2 * d_out
    gw1 = d_h * x
    Lf = lambda w: 0.5 * (w2 * float(sig(w * x)) - y) ** 2
    e = 1e-4
    cd = (Lf(w1 + e) - Lf(w1 - e)) / (2 * e)
    STATS["sigmoid"] = dict(a=a, z=z, yh=yh, L=L, d_out=d_out, gw2=gw2, sp=sp, d_h=d_h, gw1=gw1, cd_w1=cd, err=abs(cd - gw1),
                            sp5=float(sig(5) * (1 - sig(5))), sp0=0.25)


# ------------------------------------------------------------------ KP07
A = np.array([2.0, 4.0, 6.0, 8.0])


def sgd_runs(K=80, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=K)
    w_gd, w_c, w_d = [0.0], [0.0], [0.0]
    for k in range(K):
        w_gd.append(w_gd[-1] - 0.2 * (w_gd[-1] - A.mean()))
        w_c.append(w_c[-1] - 0.2 * (w_c[-1] - A[idx[k]]))
        w_d.append(w_d[-1] - 1.0 / (k + 1) * (w_d[-1] - A[idx[k]]))
    return np.array(w_gd), np.array(w_c), np.array(w_d), idx


def sgd_noise(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    gd, cst, dec, idx = sgd_runs()
    STATS["sgd"] = dict(first_idx=idx[:6].tolist(), const_last40_std=float(np.std(cst[40:])), const_last40_mean=float(np.mean(cst[40:])), dec_last=float(dec[-1]),
                        gd_last=float(gd[-1]), theory_std=0.2 * math.sqrt(5) / math.sqrt(0.2 * 1.8), w1_const=cst[1], w1_dec=dec[1])
    W, H = 640, 380
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([78 / W, 66 / H, (W - 78 - 24) / W, (H - 66 - 28) / H])
    k = np.arange(len(gd))
    ax.plot(k, gd, color=c["muted"], lw=2.2, label=r"full gradient, $\eta=0.2$")
    ax.plot(k, cst, color=c["warn"], lw=1.6, label=r"SGD, $\eta=0.2$")
    ax.plot(k, dec, color=c["vec"], lw=2.2, label=r"SGD, $\eta_k=1/(k+1)$")
    ax.axhline(5, color=c["ok"], ls="--", lw=1.3)
    ax.annotate(r"$w^*=5$", (60, 5), xytext=(0, 8), textcoords="offset points", color=c["ok"], fontsize=14)
    ax.set_xlabel(r"step $k$")
    ax.set_ylabel(r"$w_k$")
    ax.set_xlim(0, len(gd) - 1)
    ax.set_ylim(-0.3, 8.6)
    ax.legend(loc="lower right", frameon=False, fontsize=13)
    clean(ax)
    finish(fig, "lec03-kp07-sgd-noise", dark, c)
    return W, H


# ------------------------------------------------------------------ KP08
def batch_std(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    n = 64
    g = np.arange(1, n + 1) - 32.5
    sigma = float(np.sqrt(np.mean(g ** 2)))
    bs = np.arange(1, n + 1)
    wor = sigma * np.sqrt((n - bs) / (bs * (n - 1)))
    wr = sigma / np.sqrt(bs)
    STATS["batch"] = dict(sigma=sigma, wor={int(b): float(wor[b - 1]) for b in (1, 8, 32, 64)}, wr={int(b): float(wr[b - 1]) for b in (1, 8, 32, 64)})
    W, H = 640, 380
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([78 / W, 66 / H, (W - 78 - 24) / W, (H - 66 - 28) / H])
    ax.plot(bs, wr, color=c["ref"], lw=2.4, label=r"with replacement: $\sigma/\sqrt{b}$")
    ax.plot(bs, wor, color=c["main"], lw=2.4, label="without replacement")
    for b in (1, 8, 32, 64):
        ax.plot([b], [wor[b - 1]], "o", color=c["main"], ms=7)
    ax.set_xlabel(r"batch size $b$ ($n=64$)")
    ax.set_ylabel("std of the batch gradient")
    ax.set_xlim(0, 66)
    ax.set_ylim(-0.5, 20)
    ax.legend(loc="upper right", frameon=False, fontsize=13)
    clean(ax)
    finish(fig, "lec03-kp08-batch-std", dark, c)
    return W, H


if __name__ == "__main__":
    kp06_numbers()
    for dark in (False, True):
        sigmoid(dark)
        sgd_noise(dark)
        batch_std(dark)
    for k, v in STATS.items():
        print(k, {a: (round(b, 6) if isinstance(b, float) else b) for a, b in v.items()})
    g4 = np.array([3.0, 1.0, -1.0, -3.0])
    import itertools
    pairs = [g4[list(p)].mean() for p in itertools.combinations(range(4), 2)]
    print("pair means", pairs, "std", float(np.std(pairs)), "formula", math.sqrt(5 / 2 * (4 - 2) / 3), "std b=1", float(np.std(g4)))
    print("epoch n=1000 b=32:", 1000 // 32, "full batches, remainder", 1000 - 32 * (1000 // 32))
    print("cost: finite diff W=1e6 ->", 1e6 * 1e6, "ops; backprop ~ few*1e6")
