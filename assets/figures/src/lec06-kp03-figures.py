"""KP03 figures (the three stages of a convolutional layer) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec06-kp03-prewitt-responses, lec06-kp03-three-stages, lec06-kp03-learned-kernel.
Prints every number the lecture quotes (all computed here, nothing typed by hand) and asserts them.

Numbers
  * the two Prewitt kernels on a 3x3 patch that is dark on top and bright below, and on its transpose.
  * a 6x6 image (a bright block of 2s in rows 1-3, columns 0-3) through the three stages:
    horizontal Prewitt (valid sliding multiply-and-add, kernel not flipped, as in KP01) -> ReLU -> 2x2 max pooling.
  * the gradient of s[i, j] with respect to K[m, n] is the pixel I[i+m, j+n] (checked by finite differences).
  * practice: the patch [[2,2,2],[2,2,2],[0,0,0]]; ReLU + pooling of a given 4x4 map; two 1-D kernels in a row
    with and without ReLU in between; the vertical Prewitt kernel on the main image.
Run:  py assets/figures/src/lec06-kp03-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

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


def relu(a):
    return np.maximum(0, a)


def maxpool2(r):
    """non-overlapping 2x2 blocks, keep the largest number of each block"""
    h, w = r.shape
    return np.array([[r[2 * u:2 * u + 2, 2 * v:2 * v + 2].max() for v in range(w // 2)] for u in range(h // 2)])


KH = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])      # horizontal Prewitt: top row minus bottom row
KV = KH.T.copy()                                          # vertical Prewitt: left column minus right column
XP = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])          # dark on top, bright below
IMG = np.zeros((6, 6), dtype=int); IMG[1:4, 0:4] = 2      # bright block: rows 1-3, columns 0-3
P1 = np.array([[2, 2, 2], [2, 2, 2], [0, 0, 0]])
P2 = np.array([[-3, 1, 0, 2], [4, -1, -2, 5], [0, 0, -6, 1], [2, -4, 3, 3]])
P3_X = np.array([1, -3, 2, 5, -4]); P3_A = np.array([1, 1]); P3_B = np.array([1, -1])


def numbers():
    assert KV.tolist() == [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
    rh, rv = int((XP * KH).sum()), int((XP * KV).sum())
    th, tv = int((XP.T * KH).sum()), int((XP.T * KV).sum())
    print("patch", XP.tolist(), "-> Kh", rh, " Kv", rv, "   transposed", XP.T.tolist(), "-> Kh", th, " Kv", tv)
    assert (rh, rv, th, tv) == (-6, 0, 0, -6)
    print("Kh row by row on the patch:", [int((XP[m] * KH[m]).sum()) for m in range(3)])
    assert [int((XP[m] * KH[m]).sum()) for m in range(3)] == [0, 0, -6]
    print("image rows:", IMG.tolist())
    S = corr2d(IMG, KH); R = relu(S); Y = maxpool2(R)
    print("conv with Kh:", S.tolist()); print("ReLU:", R.tolist()); print("2x2 max pool:", Y.tolist())
    assert S.tolist() == [[-6, -6, -4, -2], [0, 0, 0, 0], [6, 6, 4, 2], [6, 6, 4, 2]]
    assert R.tolist() == [[0, 0, 0, 0], [0, 0, 0, 0], [6, 6, 4, 2], [6, 6, 4, 2]]
    assert Y.tolist() == [[0, 0], [6, 4]]
    print("s[2,0] window:", IMG[2:5, 0:3].tolist(), " s[0,0] window:", IMG[0:3, 0:3].tolist(), " s[0,3] window:", IMG[0:3, 3:6].tolist())
    assert (S[2, 0], S[0, 0], S[0, 3]) == (6, -6, -2)
    print("sizes: image", IMG.shape, "conv", S.shape, "pool", Y.shape, " 6-3+1 =", 6 - 3 + 1)
    assert S.shape == (4, 4) and Y.shape == (2, 2)
    # gradient of s[i,j] w.r.t. K[m,n] = I[i+m, j+n]  (finite differences, s is linear in K)
    G = np.zeros((3, 3))
    for m in range(3):
        for n in range(3):
            E = np.zeros((3, 3)); E[m, n] = 1
            G[m, n] = corr2d(IMG, KH + E)[2, 0] - S[2, 0]
    print("d s[2,0] / d K =", G.astype(int).tolist(), " (window", IMG[2:5, 0:3].tolist(), ")")
    assert np.array_equal(G, IMG[2:5, 0:3]) and G[0, 0] == 2
    # practice
    print("P1", P1.tolist(), "-> Kh", int((P1 * KH).sum()), " Kv", int((P1 * KV).sum()))
    assert (int((P1 * KH).sum()), int((P1 * KV).sum())) == (6, 0)
    print("P2 ReLU", relu(P2).tolist(), " pool", maxpool2(relu(P2)).tolist())
    assert relu(P2).tolist() == [[0, 1, 0, 2], [4, 0, 0, 5], [0, 0, 0, 1], [2, 0, 3, 3]]
    assert maxpool2(relu(P2)).tolist() == [[4, 5], [2, 3]]
    a = corr1d(P3_X, P3_A); b = corr1d(a, P3_B); one = corr1d(P3_X, np.array([1, 0, -1])); br = corr1d(relu(a), P3_B)
    print("P3 x", P3_X.tolist(), " [1,1] ->", a.tolist(), " then [1,-1] ->", b.tolist(), " one kernel [1,0,-1] ->", one.tolist(),
          "  with ReLU in between:", relu(a).tolist(), "->", br.tolist())
    assert a.tolist() == [-2, -1, 7, 1] and b.tolist() == [-1, -8, 6] and one.tolist() == [-1, -8, 6]
    assert relu(a).tolist() == [0, 0, 7, 1] and br.tolist() == [0, -7, 6]
    S4 = corr2d(IMG, KV)
    print("P4 Kv conv", S4.tolist(), " ReLU", relu(S4).tolist(), " pool", maxpool2(relu(S4)).tolist())
    assert S4.tolist() == [[0, 0, 4, 4], [0, 0, 6, 6], [0, 0, 4, 4], [0, 0, 2, 2]]
    assert np.array_equal(relu(S4), S4) and maxpool2(relu(S4)).tolist() == [[0, 6], [0, 4]]
    print("P4 s[1,2] window:", IMG[1:4, 2:5].tolist())
    assert (IMG[1:4, 2:5] * KV).sum() == 6


# ------------------------------------------------------------------ drawing helpers
def tint(hexcol, a, dark):
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    base = (0.12, 0.11, 0.10) if dark else (1, 1, 1)
    return tuple(base[i] * (1 - a) + (r, g, b)[i] * a for i in range(3))


def cell(ax, x, y, v, face, c, size=1.0, lw=1.2, fs=None, bold=False, tc=None):
    ax.add_patch(Rectangle((x, y), size, size, facecolor=face, edgecolor=c["muted"], lw=lw))
    if v is not None:
        ax.text(x + size / 2, y + size / 2, f"{v}", ha="center", va="center", color=tc or c["text"],
                fontsize=fs, fontweight="bold" if bold else "normal")


def grid(ax, M, x0, y0, c, dark, size=1.0, fs=14, colour=None):
    """draw matrix M with its row 0 on top; colour(v) -> face colour"""
    h, w = M.shape
    for i in range(h):
        for j in range(w):
            cell(ax, x0 + j * size, y0 + (h - 1 - i) * size, int(M[i, j]), colour(M[i, j]), c, size=size, fs=fs)


def signed(c, dark):
    pos, neg, zero = tint(c["ok"], 0.5, dark), tint(c["warn"], 0.45, dark), tint(c["muted"], 0.1, dark)
    return lambda v: pos if v > 0 else (neg if v < 0 else zero)


# ------------------------------------------------------------------ figures
def fig_prewitt(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    W, H = 960, 440
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.94])
    ax.set_xlim(-0.5, 16.5); ax.set_ylim(-0.9, 7.2); ax.set_aspect("equal"); ax.axis("off")
    kc = lambda v: tint(c["ref"], 0.45, dark) if v > 0 else (tint(c["warn"], 0.4, dark) if v < 0 else tint(c["muted"], 0.1, dark))
    px = lambda v: tint(c["edge"], 0.15 + 0.3 * v, dark)
    # kernels on the left
    grid(ax, KH, 0.0, 3.8, c, dark, size=0.8, fs=13, colour=kc); ax.text(1.2, 6.5, "$K_{\\mathrm{h}}$ (horizontal)", ha="center", fontsize=13)
    grid(ax, KV, 0.0, 0.0, c, dark, size=0.8, fs=13, colour=kc); ax.text(1.2, 2.7, "$K_{\\mathrm{v}}$ (vertical)", ha="center", fontsize=13)
    # patches on top
    grid(ax, XP, 5.2, 4.3, c, dark, size=0.7, fs=12, colour=px); ax.text(6.25, 6.75, "patch $X$: dark top, bright bottom", ha="center", fontsize=12)
    grid(ax, XP.T, 11.2, 4.3, c, dark, size=0.7, fs=12, colour=px); ax.text(12.25, 6.75, "transposed: bright right", ha="center", fontsize=12)
    # response table
    vals = [[int((XP * KH).sum()), int((XP.T * KH).sum())], [int((XP * KV).sum()), int((XP.T * KV).sum())]]
    for r, yy in enumerate((2.6, 0.2)):
        for col, xx in enumerate((4.7, 10.7)):
            v = vals[r][col]
            ax.add_patch(FancyBboxPatch((xx, yy), 3.1, 1.4, boxstyle="round,pad=0.05", facecolor=signed(c, dark)(v) if v else tint(c["muted"], 0.08, dark),
                                        edgecolor=c["muted"], lw=1.2))
            ax.text(xx + 1.55, yy + 0.7, f"{v}", ha="center", va="center", fontsize=20, fontweight="bold")
    ax.text(3.0, 3.3, "$\\to$", ha="center", va="center", fontsize=20); ax.text(3.0, 0.9, "$\\to$", ha="center", va="center", fontsize=20)
    ax.text(10.0, -0.35, "each box: multiply the kernel and the patch cell by cell, add all 9", ha="center", va="center", fontsize=12, color=c["muted"])
    ax.text(15.0, 3.3, "big |value|:\nedge found", ha="left", va="center", fontsize=11, color=c["muted"])
    ax.text(15.0, 0.9, "0: no edge of\nthis direction", ha="left", va="center", fontsize=11, color=c["muted"])
    finish(fig, "lec06-kp03-prewitt-responses", dark, c)


def fig_stages(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    W, H = 1000, 420
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([0.005, 0.02, 0.99, 0.94])
    ax.set_xlim(-0.3, 21.3); ax.set_ylim(-1.7, 7.4); ax.set_aspect("equal"); ax.axis("off")
    S = corr2d(IMG, KH); R = relu(S); Y = maxpool2(R)
    sz = 0.85
    grid(ax, IMG, 0.0, 0.4, c, dark, size=sz, fs=12, colour=lambda v: tint(c["edge"], 0.55 if v else 0.08, dark))
    ax.text(2.55, 6.0, "image $I$ ($6\\times6$)", ha="center", fontsize=13)
    x1 = 6.6
    grid(ax, S, x1, 1.25, c, dark, size=sz, fs=12, colour=signed(c, dark))
    ax.text(x1 + 1.7, 6.0, "1. convolution", ha="center", fontsize=13, fontweight="bold")
    ax.text(x1 + 1.7, 5.35, "$K_{\\mathrm{h}}$ slides: $s$ ($4\\times4$)", ha="center", fontsize=12)
    x2 = 12.0
    grid(ax, R, x2, 1.25, c, dark, size=sz, fs=12, colour=signed(c, dark))
    ax.text(x2 + 1.7, 6.0, "2. detector", ha="center", fontsize=13, fontweight="bold")
    ax.text(x2 + 1.7, 5.35, "ReLU: $\\max(0,a)$", ha="center", fontsize=12)
    for k in (2,):
        ax.plot([x2 + k * sz] * 2, [1.25, 1.25 + 4 * sz], color=c["main"], lw=2.2)
    ax.plot([x2, x2 + 4 * sz], [1.25 + 2 * sz] * 2, color=c["main"], lw=2.2)
    x3 = 18.2
    grid(ax, Y, x3, 2.1, c, dark, size=sz, fs=13, colour=signed(c, dark))
    ax.text(x3 + 0.85, 6.0, "3. pooling", ha="center", fontsize=13, fontweight="bold")
    ax.text(x3 + 0.85, 5.35, "max of each $2\\times2$", ha="center", fontsize=12)
    for xa, xb in ((5.4, x1 - 0.2), (x1 + 4 * sz + 0.2, x2 - 0.2), (x2 + 4 * sz + 0.2, x3 - 0.2)):
        ax.add_patch(FancyArrowPatch((xa, 2.95), (xb, 2.95), arrowstyle="-|>", mutation_scale=20, lw=2, color=c["muted"]))
    ax.text(x1 + 1.7, 0.55, "green $+$: bright above dark\nred $-$: dark above bright", ha="center", va="top", fontsize=11, color=c["muted"])
    ax.text(x2 + 1.7, 0.55, "negatives become 0", ha="center", va="top", fontsize=11, color=c["muted"])
    ax.text(x3 + 0.85, 1.6, "$2\\times2$", ha="center", va="top", fontsize=11, color=c["muted"])
    finish(fig, "lec06-kp03-three-stages", dark, c)


def fig_learned(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    W, H = 960, 380
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.94])
    ax.set_xlim(-1.4, 19.4); ax.set_ylim(-1.0, 6.6); ax.set_aspect("equal"); ax.axis("off")
    kc = lambda v: tint(c["ref"], 0.45, dark) if v > 0 else (tint(c["warn"], 0.4, dark) if v < 0 else tint(c["muted"], 0.1, dark))
    grid(ax, KH, 0.3, 1.2, c, dark, size=1.0, fs=15, colour=kc)
    ax.text(1.8, 5.6, "hand-set: $K_{\\mathrm{h}}$", ha="center", fontsize=14, fontweight="bold", color=c["warn"])
    ax.text(1.8, 4.75, "fixed by hand", ha="center", fontsize=13, color=c["muted"])
    ax.add_patch(FancyArrowPatch((4.0, 2.7), (5.6, 2.7), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=c["main"]))
    ax.text(4.7, 3.2, "make\nthem\nweights", ha="center", va="bottom", fontsize=12, color=c["main"])
    for m in range(3):
        for n in range(3):
            cell(ax, 5.8 + 1.2 * n, 3.2 - 1.2 * m + 0.2, None, tint(c["ok"], 0.3, dark), c, size=1.2)
            ax.text(6.4 + 1.2 * n, 3.8 - 1.2 * m + 0.2, f"$K[{m},{n}]$", ha="center", va="center", fontsize=12)
    ax.text(7.6, 5.6, "learned: $K[m,n]$", ha="center", fontsize=14, fontweight="bold", color=c["ok"])
    ax.text(7.6, 4.75, "9 weights learned from data", ha="center", fontsize=13, color=c["muted"])
    for x0, txt in ((11.0, "conv $\\to$ ReLU\n$\\to$ pooling $\\to\\cdots$"), (15.6, "loss $e$")):
        ax.add_patch(FancyBboxPatch((x0, 1.8), 3.2, 1.8, boxstyle="round,pad=0.08", facecolor=tint(c["muted"], 0.1, dark), edgecolor=c["muted"], lw=1.2))
        ax.text(x0 + 1.6, 2.7, txt, ha="center", va="center", fontsize=13)
    ax.add_patch(FancyArrowPatch((9.5, 3.25), (10.9, 3.25), arrowstyle="-|>", mutation_scale=18, lw=2, color=c["ok"]))
    ax.add_patch(FancyArrowPatch((14.3, 3.25), (15.5, 3.25), arrowstyle="-|>", mutation_scale=18, lw=2, color=c["ok"]))
    ax.text(12.6, 4.2, "forward", ha="center", fontsize=12, color=c["ok"])
    ax.add_patch(FancyArrowPatch((17.2, 1.7), (7.6, 0.05), connectionstyle="arc3,rad=-0.18", arrowstyle="-|>", mutation_scale=18, lw=2, color=c["warn"]))
    ax.text(13.0, -0.55, "backward: $\\partial e/\\partial K[m,n]$, then $K[m,n]\\leftarrow K[m,n]-\\eta\\,\\partial e/\\partial K[m,n]$",
            ha="center", va="center", fontsize=14, color=c["warn"])
    finish(fig, "lec06-kp03-learned-kernel", dark, c)


if __name__ == "__main__":
    numbers()
    for dark in (False, True):
        fig_prewitt(dark)
        fig_stages(dark)
        fig_learned(dark)
    print("wrote lec06-kp03-prewitt-responses, lec06-kp03-three-stages, lec06-kp03-learned-kernel (.png light, .svg dark)")
