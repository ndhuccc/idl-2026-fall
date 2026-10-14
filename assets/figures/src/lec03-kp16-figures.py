"""KP16 figure (training, validation and test errors answer different questions) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Id: lec03-kp16-three-sets.   Prints every number the lecture and the deck quote.
Run:  py assets/figures/src/lec03-kp16-figures.py
"""
import base64
import os
import sys

import numpy as np
from numpy.polynomial import legendre as Lg

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, ".agents", "skills", "view-from-text"))
import matplotlib.pyplot as plt  # noqa: E402
from jbd_figure import svg_data_uri, use_jbd_style  # noqa: E402

OUT = os.path.join(ROOT, "assets", "figures")
DARK = dict(text="#f2ede4", muted="#b7b0a5", main="#d97757", ref="#6a9fd6", vec="#5fb8a8", ok="#7dbb8c", warn="#e07a70", edge="#e0995f", grid="#b7b0a5")
LIGHT = dict(text="#1c1712", muted="#5a544b", main="#c4572f", ref="#2b6cb0", vec="#1f8a79", ok="#2f855a", warn="#c0392b", edge="#b7791f", grid="#8a8378", bg="#ffffff")
STATS = {}
NOISE = 0.2          # standard deviation of the label noise, so the best possible test MSE is NOISE**2 = 0.04


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


def experiment():
    rng = np.random.default_rng(5)

    def data(n):
        x = rng.uniform(-1, 1, n)
        return x, np.sin(3 * x) + NOISE * rng.standard_normal(n)

    xt, yt = data(15)
    xv, yv = data(15)
    xs, ys = data(2000)
    rows = []
    for d in range(0, 25):
        w = np.linalg.lstsq(Lg.legvander(xt, d), yt, rcond=None)[0]       # minimum-norm solution once d+1 > 15
        mse = lambda x, y: float(np.mean((Lg.legvander(x, d) @ w - y) ** 2))
        rows.append((d, mse(xt, yt), mse(xv, yv), mse(xs, ys)))
    return rows


def three_sets(dark, rows):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 760, 340
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([92 / W, 66 / H, (W - 92 - 24) / W, (H - 66 - 28) / H])
    d = [r[0] for r in rows]
    tr = np.maximum([r[1] for r in rows], 1e-5)
    va = [r[2] for r in rows]
    te = [r[3] for r in rows]
    ax.semilogy(d, tr, "o-", color=c["ref"], lw=2.4, ms=5, label="training")
    ax.semilogy(d, va, "s-", color=c["vec"], lw=2.4, ms=5, label="validation")
    ax.semilogy(d, te, "^-", color=c["main"], lw=2.4, ms=5, label="test")
    bv = int(np.argmin(va))
    bt = int(np.argmin(tr[:15] if False else np.array([r[1] for r in rows])))
    ax.axvline(bv, color=c["ok"], lw=1.6, ls="--")
    ax.text(bv + 0.35, 3e2, f"validation picks degree {bv}", color=c["ok"], fontsize=12)
    ax.axvline(14.5, color=c["muted"], lw=1.2, ls=":")
    ax.text(14.9, 3e-5 * 3, "15 parameters = 15 samples\n(training error 0 drawn at 1e-5)", color=c["muted"], fontsize=11)
    ax.set_xlabel("polynomial degree (number of parameters minus 1)", fontsize=13)
    ax.set_ylabel("mean squared error", fontsize=13)
    ax.set_ylim(1e-5, 1e4)
    ax.set_xlim(-0.5, 24.5)
    ax.legend(loc="lower left", frameon=False, fontsize=12)
    clean(ax)
    finish(fig, "lec03-kp16-three-sets", dark, c)
    return W, H


if __name__ == "__main__":
    rows = experiment()
    for dark in (False, True):
        three_sets(dark, rows)
    va = [r[2] for r in rows]
    bv = int(np.argmin(va))
    tr = [r[1] for r in rows]
    first_zero = next(r[0] for r in rows if r[1] < 1e-9)
    print("rows (degree, train, val, test):")
    for r in rows:
        print("  %2d  train %.6f  val %.4f  test %.4f" % r)
    print("noise floor (test MSE of the true curve):", NOISE ** 2)
    print("validation picks degree", bv, "val %.4f test %.4f" % (rows[bv][2], rows[bv][3]))
    print("first degree with training MSE below 1e-9:", first_zero, "test %.4f" % rows[first_zero][3], "val %.4f" % rows[first_zero][2])
    print("training MSE always decreases with degree:", all(tr[i + 1] <= tr[i] + 1e-12 for i in range(len(tr) - 1)))
    print("worst test MSE: degree", int(np.argmax([r[3] for r in rows])), "%.2f" % max(r[3] for r in rows))
