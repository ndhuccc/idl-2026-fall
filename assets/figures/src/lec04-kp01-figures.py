"""KP01 figures (data splits and the two kinds of cheating) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp01-split-roles, lec04-kp01-cheating-scores.   Prints every number the lecture and the deck quote.

Numbers
  * split of 100 examples: 60 / 20 / 20.
  * cheating #1 demo : 60 points in the unit square with random +-1 labels, a 1-nearest-neighbour classifier
                       (it memorises); accuracy on its own training points vs on 1000 fresh random points.
  * cheating #2 demo : 200 random linear classifiers, 20 random-label test points; keep the best on those 20, then
                       evaluate it on 1000 fresh random points.
  * overlap demo     : toy regression f(x)=1+2x-1.5x^2, sigma=0.3 (noise floor 0.09), 20 training points, polynomial
                       degrees 1,2,3,5,9; validation = the training points themselves (T cap V != empty) vs 20 clean points.
Run:  py assets/figures/src/lec04-kp01-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

SIGMA = 0.3


def f_true(x):
    return 1.0 + 2.0 * np.asarray(x) - 1.5 * np.asarray(x) ** 2


def make(n, seed):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1, 1, n))
    return x, f_true(x) + rng.normal(0, SIGMA, n)


# ------------------------------------------------------------------ numbers
def one_nn(Xtr, ytr, Xq):
    d = ((Xq[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    return ytr[d.argmin(1)]


def cheating_numbers():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (60, 2))
    y = rng.choice([-1, 1], 60)
    acc_overlap = float((one_nn(X, y, X) == y).mean())
    Xf = rng.uniform(0, 1, (1000, 2))
    yf = rng.choice([-1, 1], 1000)
    acc_fresh = float((one_nn(X, y, Xf) == yf).mean())
    # cheating 2: 200 random linear classifiers selected on a 20-point test set
    rng = np.random.default_rng(1)
    Xt = rng.uniform(0, 1, (20, 2))
    yt = rng.choice([-1, 1], 20)
    Xr = rng.uniform(0, 1, (1000, 2))
    yr = rng.choice([-1, 1], 1000)
    W = rng.normal(size=(200, 2))
    b = rng.normal(size=200) * 0.5
    pred_t = np.sign(Xt @ W.T + b)
    pred_t[pred_t == 0] = 1
    acc_t = (pred_t == yt[:, None]).mean(0)
    best = int(acc_t.argmax())
    pred_r = np.sign(Xr @ W[best] + b[best])
    acc_r = float((pred_r == yr).mean())
    return dict(acc_overlap=acc_overlap, acc_fresh=acc_fresh, best_test=float(acc_t.max()), best_fresh=acc_r,
                mean_all=float(acc_t.mean()), best_idx=best)


def poly_numbers(degrees=(1, 2, 3, 5, 9)):
    xt, yt = make(20, 1)
    xv, yv = make(20, 2)
    xs, ys = make(2000, 3)
    rows = []
    for d in degrees:
        c = np.polyfit(xt, yt, d)
        mse = lambda x, y: float(np.mean((np.polyval(c, x) - y) ** 2))  # noqa: E731
        rows.append((d, mse(xt, yt), mse(xv, yv), mse(xs, ys)))
    return rows


# ------------------------------------------------------------------ figure 1: roles of the three piles
def split_roles(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 720, 300
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    x0, x1, y0, h = 40, 680, 150, 56
    scale = (x1 - x0) / 100.0
    parts = [("T", 60, c["ref"], "fit the weights"), ("V", 20, c["vec"], "tune and\nstop early"),
             ("R", 20, c["ok"], "report\nonce")]
    x = x0
    for name, n, col, role in parts:
        w = n * scale
        ax.add_patch(FancyBboxPatch((x + 2, y0), w - 4, h, boxstyle="round,pad=0,rounding_size=8", fc=col, ec="none", alpha=0.28))
        ax.add_patch(FancyBboxPatch((x + 2, y0), w - 4, h, boxstyle="round,pad=0,rounding_size=8", fc="none", ec=col, lw=2.2))
        ax.text(x + w / 2, y0 + h / 2, rf"$\mathcal{{{name}}}$: {n}", ha="center", va="center", fontsize=17, color=c["text"])
        ax.annotate("", xy=(x + w / 2, y0 - 8), xytext=(x + w / 2, y0 - 38), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
        ax.text(x + w / 2, y0 - 66, role, ha="center", va="center", fontsize=13.5, color=c["text"], linespacing=1.15)
        x += w
    ax.text((x0 + x1) / 2, y0 + h + 36, r"$\mathcal{D}=\mathcal{T}\cup\mathcal{V}\cup\mathcal{R}$,   pairwise disjoint, 100 examples",
            ha="center", va="center", fontsize=16, color=c["text"])
    finish(fig, "lec04-kp01-split-roles", dark, c)
    return W, H


# ------------------------------------------------------------------ figure 2: scores that cheating inflates
def cheating_scores(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 700, 400
    fig, ax = axes(W, H, left=86, bottom=104, right=24, top=28)
    n = cheating_numbers()
    labels = ["1-NN on its\nown training\npoints", "1-NN on\n1000 fresh\npoints",
              "best of 200\non 20 test\npoints", "same model\non 1000 fresh\npoints"]
    vals = [n["acc_overlap"], n["acc_fresh"], n["best_test"], n["best_fresh"]]
    cols = [c["warn"], c["ok"], c["warn"], c["ok"]]
    xs = [0, 1, 2.6, 3.6]
    ax.bar(xs, [v * 100 for v in vals], color=cols, width=0.72, alpha=0.9)
    for xx, v in zip(xs, vals):
        ax.text(xx, v * 100 + 2, f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=14, color=c["text"])
    ax.axhline(50, color=c["muted"], ls="--", lw=1.3)
    ax.text(1.8, 51.5, "chance", ha="center", va="bottom", fontsize=12.5, color=c["muted"])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 118)
    ax.set_xlim(-0.6, 4.2)
    clean(ax)
    ax.grid(False)
    finish(fig, "lec04-kp01-cheating-scores", dark, c)
    return W, H


def report():
    n = cheating_numbers()
    print("cheating #1: 1-NN on random labels, 60 train points")
    print(f"  accuracy on its own training points = {n['acc_overlap']:.3f}; on 1000 fresh points = {n['acc_fresh']:.3f}")
    print("cheating #2: 200 random linear classifiers, 20 test points with random labels")
    print(f"  mean accuracy over the 200 = {n['mean_all']:.3f}; best on the test set = {n['best_test']:.3f} (model #{n['best_idx']}); same model on 1000 fresh points = {n['best_fresh']:.3f}")
    print("overlap demo: polynomial degree, train MSE (= 'validation' if T cap V != empty), clean validation MSE (n=20), test MSE (n=2000)")
    rows = poly_numbers()
    for d, tr, va, te in rows:
        print(f"  degree {d}: train {tr:.4f}  clean validation {va:.4f}  test {te:.4f}")
    print("  chosen with overlapping validation: degree", min(rows, key=lambda r: r[1])[0], "; with clean validation: degree", min(rows, key=lambda r: r[2])[0],
          "; best on the 2000-point test set: degree", min(rows, key=lambda r: r[3])[0])
    # split arithmetic
    print("split 100 = 60 + 20 + 20 =", 60 + 20 + 20, "; ratios", 60 / 100, 20 / 100, 20 / 100)


if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), ".."), exist_ok=True)
    sizes = {}
    for fn in (split_roles, cheating_scores):
        for dark in (False, True):
            sizes[fn.__name__] = fn(dark)
    print("figure sizes (px):", sizes)
    report()
