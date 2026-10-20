"""KP17 figures (Dropout) in two skins from the same data.
    assets/figures/<id>.png (light, lecture)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp17-network (full network and one random mask), lec04-kp17-mask-hist (a linear unit under 20000 random masks).
Prints every number the lecture and the deck quote.   p = KEEP probability (nn.Dropout in PyTorch takes q = 1 - p, the drop probability).
Run:  python3 assets/figures/src/lec04-kp17-figures.py
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

P = 0.5
H_EX = np.array([2.0, -1.0, 3.0, 1.0])      # four hidden activations of the hand example
W_EX = np.array([1.0, 2.0, 1.0, -1.0])      # weights of one linear unit that reads them


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# ------------------------------------------------------------------ figure 1: network with a mask
def network(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H, left=10, bottom=10, right=10, top=34)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    mask = np.array([1, 0, 1, 1, 0])
    ys_in, ys_h, ys_o = np.linspace(1.3, 4.9, 4), np.linspace(0.9, 5.3, 5), [3.1]
    for off, title, use_mask in ((0.2, "full network", False), (5.4, r"one mask $m$ (keep prob. $p$)", True)):
        xi, xh, xo = off, off + 2.0, off + 4.0
        ax.text(off + 2.0, 6.05, title, ha="center", va="center", color=c["text"], fontsize=13)
        for j, yh in enumerate(ys_h):
            alive = (not use_mask) or mask[j] == 1
            for yi in ys_in:
                if alive:
                    ax.plot([xi, xh], [yi, yh], color=c["muted"], lw=0.8, alpha=0.55, zorder=1)
            if alive:
                ax.plot([xh, xo], [yh, ys_o[0]], color=c["muted"], lw=0.8, alpha=0.55, zorder=1)
        ax.scatter([xi] * 4, ys_in, s=170, color=c["ref"], zorder=3)
        ax.scatter([xo], ys_o, s=170, color=c["vec"], zorder=3)
        for j, yh in enumerate(ys_h):
            alive = (not use_mask) or mask[j] == 1
            if alive:
                ax.scatter([xh], [yh], s=170, color=c["main"], zorder=3)
            else:
                ax.scatter([xh], [yh], s=170, facecolor="none", edgecolor=c["muted"], lw=1.4, zorder=3)
                ax.plot([xh - 0.16, xh + 0.16], [yh - 0.16, yh + 0.16], color=c["warn"], lw=2.4, zorder=4)
                ax.plot([xh - 0.16, xh + 0.16], [yh + 0.16, yh - 0.16], color=c["warn"], lw=2.4, zorder=4)
    finish(fig, "lec04-kp17-network", dark, c)
    return W, H


# ------------------------------------------------------------------ figure 2: a linear unit under many masks
def mask_hist(dark, n_units=20, n_masks=20000, seed=23):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    rng = np.random.default_rng(seed)
    h = rng.normal(0, 1, n_units)
    w = rng.normal(0, 1, n_units)
    z_full = float(w @ h)
    masks = (rng.random((n_masks, n_units)) < P).astype(float)
    z = masks @ (w * h)
    ax.hist(z, bins=60, color=c["ref"], alpha=0.75, density=True)
    ax.axvline(z_full, color=c["warn"], lw=2.2, ls="--", label=r"full network $z$")
    ax.axvline(P * z_full, color=c["vec"], lw=2.6, label=r"$p\,z$ (= mean over masks)")
    ax.set_xlabel(r"pre-activation $z$ under a random mask")
    ax.set_ylabel("density")
    ax.set_ylim(0, 0.30)
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    clean(ax)
    finish(fig, "lec04-kp17-mask-hist", dark, c)
    return W, H, z_full, float(z.mean()), float(z.std())


def report():
    print("== hand example: h =", H_EX, " w =", W_EX, " p =", P)
    zf = float(W_EX @ H_EX)
    print("  full z = w.h =", zf, " terms w*h =", W_EX * H_EX)
    m = np.array([1, 0, 1, 1])
    print("  one mask (1,0,1,1): z =", float(W_EX @ (m * H_EX)))
    allz = []
    for bits in itertools.product([0, 1], repeat=4):
        allz.append(float(W_EX @ (np.array(bits) * H_EX)))
    allz = np.array(allz)
    print("  all 16 masks: mean =", allz.mean(), " p*z =", P * zf, " std =", round(float(allz.std()), 4),
          " theory std sqrt(p(1-p)*sum (w h)^2) =", round(float(np.sqrt(P * (1 - P) * np.sum((W_EX * H_EX) ** 2))), 4), " sum (wh)^2 =", float(np.sum((W_EX * H_EX) ** 2)))
    print("  sorted masked z values:", sorted(allz.tolist()))
    print("  inverted dropout: mean of (m*h/p) =", np.mean([np.array(b) * H_EX / P for b in itertools.product([0, 1], repeat=4)], axis=0), "(equals h)")
    print("  number of sub-networks for n hidden units: 2^n:", {n: 2 ** n for n in (4, 100)})
    print("  per-layer example 1000 hidden units: 2^1000 ~ 10^%.1f" % (1000 * np.log10(2)))
    print("== simulation: 20 units, 20000 masks")
    W_, H_, zf2, mean2, std2 = mask_hist(False)
    print(f"  full z = {zf2:.4f}  p*z = {P * zf2:.4f}  mean over masks = {mean2:.4f}  std = {std2:.4f}")
    # ---------------- geometric mean claim
    rng = np.random.default_rng(3)
    print("== weight scaling vs geometric mean of sub-network outputs (p = 0.5)")
    W1 = rng.normal(0, 1, (3, 4))
    x = np.array([1.0, -0.5, 2.0, 0.5])
    outs = np.array([softmax(W1 @ (np.array(b) * x)) for b in itertools.product([0, 1], repeat=4)])
    gm = np.exp(np.log(outs).mean(axis=0))
    gm /= gm.sum()
    ws = softmax(P * (W1 @ x))
    print("  softmax regression, dropout on the 4 inputs: weight scaling", np.round(ws, 6), " normalised geometric mean", np.round(gm, 6),
          " max|diff| =", float(np.abs(ws - gm).max()), " arithmetic mean", np.round(outs.mean(axis=0), 4))
    # one hidden layer of 6 relu units, dropout on hidden units only: exact too (output layer is softmax regression on the hidden vector)
    rng = np.random.default_rng(5)
    A1, A2 = rng.normal(0, 1, (6, 4)), rng.normal(0, 1, (3, 6))
    a = np.maximum(A1 @ x, 0)
    outs = np.array([softmax(A2 @ (np.array(b) * a)) for b in itertools.product([0, 1], repeat=6)])
    gm = np.exp(np.log(outs).mean(axis=0))
    gm /= gm.sum()
    ws = softmax(P * (A2 @ a))
    print("  one ReLU hidden layer, dropout on it: max|weight scaling - geometric mean| =", float(np.abs(ws - gm).max()))
    # two hidden layers: dropout on both -> only approximate
    rng = np.random.default_rng(11)
    B1, B2, B3 = rng.normal(0, 1, (5, 4)), rng.normal(0, 1, (5, 5)), rng.normal(0, 1, (3, 5))
    a1 = np.maximum(B1 @ x, 0)
    ws_out = softmax(P * (B3 @ np.maximum(P * (B2 @ (P * a1)), 0)))
    logs, ar = [], []
    for b1 in itertools.product([0, 1], repeat=5):
        for b2 in itertools.product([0, 1], repeat=5):
            a2 = np.maximum(B2 @ (np.array(b1) * a1), 0)
            o = softmax(B3 @ (np.array(b2) * a2))
            logs.append(np.log(o))
            ar.append(o)
    gm = np.exp(np.mean(logs, axis=0))
    gm /= gm.sum()
    print("  two hidden layers, dropout on both (1024 masks): weight scaling", np.round(ws_out, 5), " geometric mean", np.round(gm, 5),
          " max|diff| =", round(float(np.abs(ws_out - gm).max()), 5), " arithmetic mean", np.round(np.mean(ar, axis=0), 5))


if __name__ == "__main__":
    sizes = {"lec04-kp17-network": network(False), "lec04-kp17-mask-hist": mask_hist(False)[:2]}
    network(True)
    mask_hist(True)
    print("sizes", sizes)
    report()
