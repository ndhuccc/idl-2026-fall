"""KP01 figures (flattening versus local scanning) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec06-kp01-flatten-neighbors, lec06-kp01-sliding-kernel, lec06-kp01-weight-counts, lec06-kp01-shift-equivariance.
Prints every number the lecture quotes (all computed here, nothing typed by hand) and asserts them.

Numbers
  * flattening a 4x4 grid row by row: index of (i, j) is 4i + j; neighbours of (1, 1); the pair (0, 3), (1, 0).
  * 3x3 kernel with ones on the corners and the centre slid over a 5x5 checkerboard: 3x3 feature map.
  * same-size 4x4 (and 8x8) layer: weights of a fully connected layer, connections of a 3x3 local layer that touch
    real pixels (no padded zeros), and weights after sharing (bias excluded).
  * larger images: 28x28 to 512 units, 224x224x3 to 512 units, 64 kernels of 3x3x3.
  * 1-D shift: x = [2, 1, 3] and w = [4, 5] (same numbers as KP04) on a row of six cells, shifted right by one.
  * practice: 8x8 counts, 5x5 flatten indices, a 4x4 image with a 2x2 kernel, the edge kernel [1, -1].
Run:  py assets/figures/src/lec06-kp01-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from lec04_common import DARK, LIGHT, finish, skin  # noqa: E402


# ------------------------------------------------------------------ numbers
def corr2d(I, K):
    """valid sliding multiply-and-add: s[i,j] = sum_m sum_n I[i+m, j+n] K[m,n]"""
    k = K.shape[0]
    H, W = I.shape
    return np.array([[(I[i:i + k, j:j + k] * K).sum() for j in range(W - k + 1)] for i in range(H - k + 1)])


def corr1d(x, w):
    k = len(w)
    return np.array([sum(x[t + m] * w[m] for m in range(k)) for t in range(len(x) - k + 1)])


def local_links(n, k):
    """same-size layer on an n x n image, k x k window centred on each output: links that touch a real pixel"""
    p = k // 2
    return sum(1 for i in range(n) for j in range(n) for a in range(-p, p + 1) for b in range(-p, p + 1)
               if 0 <= i + a < n and 0 <= j + b < n)


def links_per_output(n, k):
    p = k // 2
    return np.array([[sum(1 for a in range(-p, p + 1) for b in range(-p, p + 1) if 0 <= i + a < n and 0 <= j + b < n)
                      for j in range(n)] for i in range(n)])


CHECK = (np.indices((5, 5)).sum(0) % 2 == 0).astype(int)        # 1 where i + j is even
KX = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])                  # ones on the corners and the centre
X1 = np.array([2, 1, 3]); W1 = np.array([4, 5])
ROW = np.array([0, 2, 1, 3, 0, 0]); ROW_SHIFT = np.roll(ROW, 1)   # x placed on six cells, then moved right by one
P3_I = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]]); P3_K = np.array([[1, 0], [0, 1]])
P4_X = np.array([0, 0, 5, 5, 0, 0]); P4_W = np.array([1, -1])


def numbers():
    idx = lambda i, j, n=4: n * i + j
    nb = {"up": idx(0, 1), "down": idx(2, 1), "left": idx(1, 0), "right": idx(1, 2)}
    print("flatten 4x4: (1,1) ->", idx(1, 1), "neighbours", nb, " (0,3) ->", idx(0, 3), " (1,0) ->", idx(1, 0))
    assert idx(1, 1) == 5 and nb == {"up": 1, "down": 9, "left": 4, "right": 6} and idx(1, 0) - idx(0, 3) == 1
    S = corr2d(CHECK, KX)
    print("checkerboard row 0:", CHECK[0].tolist(), " feature map:", S.tolist(), " s[0,0] terms:", (CHECK[0:3, 0:3] * KX).tolist())
    assert S.tolist() == [[5, 0, 5], [0, 5, 0], [5, 0, 5]]
    for n in (4, 8):
        fc, loc, sh = (n * n) ** 2, local_links(n, 3), 9
        print(f"same-size {n}x{n}: fully connected {fc}, local 3x3 {loc} (source method {n*n*9}), shared {sh}")
    assert (local_links(4, 3), local_links(8, 3), 16 * 16, 64 * 64) == (100, 484, 256, 4096)
    L = links_per_output(4, 3)
    print("links per output 4x4:", L.tolist(), " corners/edges/inside:", 4 * 4, 8 * 6, 4 * 9)
    assert L.sum() == 100 and 4 * 4 + 8 * 6 + 4 * 9 == 100
    print("28x28 -> 512:", 784 * 512, "  224x224x3 =", 224 * 224 * 3, " -> 512:", 224 * 224 * 3 * 512, "  64 kernels 3x3x3:", 64 * 27)
    assert (784 * 512, 224 * 224 * 3, 224 * 224 * 3 * 512, 64 * 27) == (401408, 150528, 77070336, 1728)
    s, s2 = corr1d(ROW, W1), corr1d(ROW_SHIFT, W1)
    print("1-D: x", ROW.tolist(), "->", s.tolist(), "   shifted", ROW_SHIFT.tolist(), "->", s2.tolist())
    assert s.tolist() == [10, 13, 19, 12, 0] and s2.tolist() == [0, 10, 13, 19, 12] and np.array_equal(s2[1:], s[:-1])
    assert corr1d(X1, W1).tolist() == [13, 19]
    # practice
    print("P1 8x8: fully connected", 64 * 64, "local", local_links(8, 3), "shared", 9)
    pairs = [(idx(i, 4, 5), idx(i + 1, 0, 5)) for i in range(4)]
    print("P2 pairs adjacent in the strip but at opposite ends of two rows:", pairs)
    assert pairs == [(4, 5), (9, 10), (14, 15), (19, 20)]
    print("P2 5x5: (2,3) ->", idx(2, 3, 5), " up", idx(1, 3, 5), " down", idx(3, 3, 5), " (0,4),(1,0) ->", idx(0, 4, 5), idx(1, 0, 5))
    assert (idx(2, 3, 5), idx(1, 3, 5), idx(3, 3, 5), idx(0, 4, 5), idx(1, 0, 5)) == (13, 8, 18, 4, 5)
    P3 = corr2d(P3_I, P3_K)
    print("P3 4x4 with 2x2 diagonal kernel:", P3.tolist())
    assert P3.tolist() == [[2, 1, 0], [1, 2, 1], [0, 1, 2]]
    a, b = corr1d(P4_X, P4_W), corr1d(np.roll(P4_X, 1), P4_W)
    print("P4 edge kernel [1,-1]:", P4_X.tolist(), "->", a.tolist(), "  shifted", np.roll(P4_X, 1).tolist(), "->", b.tolist())
    assert a.tolist() == [0, -5, 0, 5, 0] and b.tolist() == [0, 0, -5, 0, 5]


# ------------------------------------------------------------------ drawing helpers
def cell(ax, x, y, v, face, c, size=1.0, lw=1.2, fs=None, bold=False, tc=None):
    ax.add_patch(Rectangle((x, y), size, size, facecolor=face, edgecolor=c["muted"], lw=lw))
    if v is not None:
        ax.text(x + size / 2, y + size / 2, f"{v}", ha="center", va="center", color=tc or c["text"],
                fontsize=fs, fontweight="bold" if bold else "normal")


def tint(hexcol, a, dark):
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    base = (0.12, 0.11, 0.10) if dark else (1, 1, 1)
    return tuple(base[i] * (1 - a) + (r, g, b)[i] * a for i in range(3))


# ------------------------------------------------------------------ figures
def fig_flatten(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    W, H = 1000, 440
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.9])
    ax.set_xlim(-0.6, 17.6); ax.set_ylim(-3.9, 4.6); ax.set_aspect("equal"); ax.axis("off")
    base = tint(c["muted"], 0.12, dark); me = tint(c["main"], 0.6, dark)
    nbr = tint(c["ref"], 0.45, dark); far = tint(c["warn"], 0.4, dark)
    role = {5: me, 1: nbr, 9: nbr, 4: nbr, 6: nbr, 3: far}
    for i in range(4):
        for j in range(4):
            k = 4 * i + j
            cell(ax, j, 3 - i, k, role.get(k, base), c, fs=14, bold=k == 5)
    ax.text(2.0, 4.3, "$4\\times4$ image, cell number $=4p+q$", ha="center", fontsize=14)
    ax.add_patch(FancyArrowPatch((4.3, 1.3), (5.0, 0.2), arrowstyle="-|>", mutation_scale=22, lw=2, color=c["muted"]))
    ax.text(4.6, 1.5, "flatten row by row", ha="left", fontsize=13, color=c["muted"])
    ax.text(6.0, 3.5, "orange: cell 5", ha="left", va="center", fontsize=13, color=c["main"])
    ax.text(6.0, 2.8, "blue: its 4 neighbours in the image", ha="left", va="center", fontsize=13, color=c["ref"])
    ax.text(6.0, 2.1, "red: cell 3, at the far end of row 0 from cell 4", ha="left", va="center", fontsize=13, color=c["warn"])
    y0 = -1.0
    for k in range(16):
        cell(ax, 1.3 + k, y0, k, role.get(k, base), c, fs=13, bold=k == 5)
    def brace(a, b, y, txt, col, up=False):
        xa, xb = 1.3 + a + 0.5, 1.3 + b + 0.5
        d = 0.35 if up else -0.35
        ax.plot([xa, xa, xb, xb], [y, y + d, y + d, y], color=col, lw=1.8)
        ax.text((xa + xb) / 2, y + 2.4 * d, txt, ha="center", va="center", fontsize=13, color=col)
    brace(1, 5, y0 - 0.1, "up: 4 apart", c["ref"])
    brace(5, 9, y0 - 0.1, "down: 4 apart", c["ref"])
    brace(4, 6, y0 + 1.1, "", c["ref"], up=True)
    ax.text(1.3 + 5.5 + 1.6, y0 + 1.75, "left, right: 1 apart", ha="left", va="center", fontsize=13, color=c["ref"])
    for ya, yb in ((y0 - 0.05, y0 - 0.4), (y0 - 1.3, y0 - 2.3)):
        ax.plot([1.3 + 4.0, 1.3 + 4.0], [ya, yb], color=c["warn"], lw=1.4, ls=":")
    ax.text(1.3 + 4.0, y0 - 2.6, "cells 3 and 4: side by side in the strip, far apart in the image", ha="left", va="center", fontsize=13, color=c["warn"])
    finish(fig, "lec06-kp01-flatten-neighbors", dark, c)


def fig_sliding(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    W, H = 960, 400
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.88])
    ax.set_xlim(-0.4, 14.6); ax.set_ylim(-1.6, 5.6); ax.set_aspect("equal"); ax.axis("off")
    S = corr2d(CHECK, KX)
    win = tint(c["edge"], 0.45, dark); base = tint(c["muted"], 0.12, dark)
    for i in range(5):
        for j in range(5):
            cell(ax, j, 4 - i, int(CHECK[i, j]), win if i < 3 and j < 3 else base, c, fs=15)
    ax.add_patch(Rectangle((0, 2), 3, 3, fill=False, edgecolor=c["main"], lw=3))
    ax.text(2.5, 5.25, "image $I$ ($5\\times5$)", ha="center", fontsize=14)
    ax.text(5.6, 2.5, "$\\star$", ha="center", va="center", fontsize=24)
    for m in range(3):
        for n in range(3):
            cell(ax, 6.4 + n, 3.5 - m, int(KX[m, n]), tint(c["ref"], 0.45 if KX[m, n] else 0.1, dark), c, fs=15)
    ax.text(7.9, 5.25, "kernel $K$ ($3\\times3$)", ha="center", fontsize=14)
    ax.text(10.2, 2.5, "$=$", ha="center", va="center", fontsize=22)
    for i in range(3):
        for j in range(3):
            on = (i, j) == (0, 0)
            cell(ax, 11 + j, 3.5 - i, int(S[i, j]), tint(c["ok"], 0.55 if on else 0.2, dark), c, fs=15, bold=on)
    ax.add_patch(Rectangle((11, 3.5), 1, 1, fill=False, edgecolor=c["main"], lw=3))
    ax.text(12.5, 5.25, "feature map $s$ ($3\\times3$)", ha="center", fontsize=14)
    t = (CHECK[0:3, 0:3] * KX)
    ones = [int(CHECK[m, n]) for m in range(3) for n in range(3) if KX[m, n]]
    ax.text(7.0, -0.9, "$s[0,0]=" + "+".join(str(v) for v in ones) + f"={S[0, 0]}$   (only the five 1s of $K$ count)",
            ha="center", va="center", fontsize=17, color=c["main"])
    assert t.sum() == S[0, 0]
    finish(fig, "lec06-kp01-sliding-kernel", dark, c)


def fig_counts(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    W, H = 960, 380
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([0.30, 0.16, 0.66, 0.74])
    labels = ["fully connected\n(every output to every pixel)", "sparse: $3\\times3$ neighbours only\n(own weights per output)", "sparse + shared\n(one $3\\times3$ kernel everywhere)"]
    vals = [16 * 16, local_links(4, 3), 9]
    cols = [c["warn"], c["edge"], c["ok"]]
    y = np.arange(3)[::-1]
    ax.barh(y, vals, color=[tint(cc, 0.7, dark) for cc in cols], edgecolor=c["muted"], height=0.6)
    for yy, v in zip(y, vals):
        ax.text(v + 4, yy, f"{v}", va="center", ha="left", fontsize=16, fontweight="bold", color=c["text"])
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlim(0, 300); ax.set_xlabel("number of different weights ($4\\times4$ image, $4\\times4$ output, no bias)", fontsize=13)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.annotate("", xy=(100, 1.35), xytext=(256, 1.35), arrowprops=dict(arrowstyle="-|>", color=c["main"], lw=2))
    ax.text(178, 1.55, "sparse", ha="center", fontsize=13, color=c["main"])
    ax.annotate("", xy=(9, 0.35), xytext=(100, 0.35), arrowprops=dict(arrowstyle="-|>", color=c["main"], lw=2))
    ax.text(55, 0.55, "shared", ha="center", fontsize=13, color=c["main"])
    finish(fig, "lec06-kp01-weight-counts", dark, c)


def fig_shift(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    W, H = 960, 420
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.9])
    ax.set_xlim(-2.4, 13.4); ax.set_ylim(-0.4, 6.4); ax.set_aspect("equal"); ax.axis("off")
    s, s2 = corr1d(ROW, W1), corr1d(ROW_SHIFT, W1)
    xin = tint(c["ref"], 0.4, dark); zero = tint(c["muted"], 0.1, dark); out = tint(c["ok"], 0.35, dark); new = tint(c["warn"], 0.3, dark)
    rows = [(5.0, ROW, "input $x$", False), (3.6, ROW_SHIFT, "shifted input", False), (2.0, s, "output $s$", True), (0.4, s2, "new output", True)]
    for y, vals, lab, isout in rows:
        ax.text(-0.3, y + 0.5, lab, ha="right", va="center", fontsize=14)
        for t, v in enumerate(vals):
            face = (new if (vals is s2 and t == 0) else out) if isout else (xin if v else zero)
            cell(ax, t + (0.5 if isout else 0), y, int(v), face, c, fs=15)
    for t in range(1, 4):
        ax.add_patch(FancyArrowPatch((t + 0.5, 4.95), (t + 1.5, 4.65), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=c["ref"]))
    for t in range(4):
        ax.add_patch(FancyArrowPatch((t + 1.0, 1.95), (t + 2.0, 1.45), arrowstyle="-|>", mutation_scale=14, lw=1.4, color=c["main"]))
    ax.text(6.6, 4.4, "input moves right by 1", ha="left", va="center", fontsize=14, color=c["ref"])
    ax.text(6.6, 1.7, "output moves right by 1", ha="left", va="center", fontsize=14, color=c["main"])
    ax.text(6.6, 0.9, "red: new value at the border", ha="left", va="center", fontsize=13, color=c["warn"])
    ax.text(6.6, 6.0, "kernel $w=[4,5]$:  $s[t]=x[t]\\,w[0]+x[t+1]\\,w[1]$", ha="left", va="center", fontsize=13, color=c["text"])
    finish(fig, "lec06-kp01-shift-equivariance", dark, c)


if __name__ == "__main__":
    numbers()
    for dark in (False, True):
        fig_flatten(dark)
        fig_sliding(dark)
        fig_counts(dark)
        fig_shift(dark)
    print("wrote lec06-kp01-flatten-neighbors, lec06-kp01-sliding-kernel, lec06-kp01-weight-counts, lec06-kp01-shift-equivariance (.png light, .svg dark)")
