"""KP04 figures (discrete convolution and cross-correlation) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec06-kp04-flip-1d, lec06-kp04-rotate-180, lec06-kp04-2d-correlation.
Prints every number the lecture quotes (all computed here, nothing typed by hand).

Numbers
  * 1-D main example: x = [2, 1, 3], w = [4, 5]; valid convolution (flip first), valid cross-correlation (no flip),
    and cross-correlation with the flipped kernel [5, 4].
  * 180-degree rotation of the 3x3 kernel with entries 1..9 (source KP04 figure).
  * 2-D main example: I = [[1,2,0],[1,1,2],[0,1,1]], K = 2x2 identity (source KP04 Fig. 1), valid cross-correlation.
  * practice: x = [1,0,2], w = [3,4]; Prewitt horizontal kernel on the KP03 3x3 patch; asymmetric K = [[1,2],[0,0]]
    on I; kernels [1,2,1] and [1,0,-1] on [3,1,4,1,5]; valid output lengths.
Run:  py assets/figures/src/lec06-kp04-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from lec04_common import DARK, LIGHT, finish, skin  # noqa: E402


# ------------------------------------------------------------------ numbers
def corr1d(x, w):
    """valid cross-correlation: s[t] = sum_m x[t+m] w[m]"""
    k = len(w)
    return np.array([sum(x[t + m] * w[m] for m in range(k)) for t in range(len(x) - k + 1)])


def conv1d_full(x, w):
    """full convolution: (x*w)[t] = sum_tau x[tau] w[t-tau]"""
    L, k = len(x), len(w)
    return np.array([sum(x[tau] * w[t - tau] for tau in range(L) if 0 <= t - tau < k) for t in range(L + k - 1)])


def conv1d_valid(x, w):
    k = len(w)
    return conv1d_full(x, w)[k - 1:len(x)]


def corr2d(I, K):
    """valid cross-correlation: s[i,j] = sum_m sum_n I[i+m, j+n] K[m,n]"""
    k = K.shape[0]
    H, W = I.shape
    return np.array([[(I[i:i + k, j:j + k] * K).sum() for j in range(W - k + 1)] for i in range(H - k + 1)])


def conv2d_valid(I, K):
    """valid convolution = cross-correlation with the kernel rotated by 180 degrees"""
    return corr2d(I, np.rot90(K, 2))


def conv2d_full(I, K):
    """full 2-D convolution by its definition: (I*K)[i,j] = sum_p sum_q I[p,q] K[i-p, j-q]"""
    H, W = I.shape
    k = K.shape[0]
    out = np.zeros((H + k - 1, W + k - 1), dtype=I.dtype)
    for i in range(H + k - 1):
        for j in range(W + k - 1):
            out[i, j] = sum(I[p, q] * K[i - p, j - q] for p in range(H) for q in range(W)
                            if 0 <= i - p < k and 0 <= j - q < k)
    return out


def corr1d_full(x, w):
    """full cross-correlation: every shift t = -(k-1) .. n-1 with any overlap, zeros outside the signal"""
    k = len(w)
    return np.array([sum(x[t + m] * w[m] for m in range(k) if 0 <= t + m < len(x)) for t in range(-(k - 1), len(x))])


X1 = np.array([2, 1, 3])
W1 = np.array([4, 5])
I2 = np.array([[1, 2, 0], [1, 1, 2], [0, 1, 1]])
K2 = np.eye(2, dtype=int)
K9 = np.arange(1, 10).reshape(3, 3)


def numbers():
    print("1-D full convolution x*w      :", conv1d_full(X1, W1).tolist())
    print("1-D valid convolution x*w     :", conv1d_valid(X1, W1).tolist())
    print("1-D valid cross-correlation   :", corr1d(X1, W1).tolist())
    print("1-D corr with flipped [5,4]   :", corr1d(X1, W1[::-1]).tolist())
    assert np.array_equal(conv1d_valid(X1, W1), np.convolve(X1, W1, "valid"))
    assert np.array_equal(corr1d(X1, W1), np.correlate(X1, W1, "valid"))
    print("rot180 of 1..9                :", np.rot90(K9, 2).tolist())
    print("2-D corr I, identity          :", corr2d(I2, K2).tolist())
    print("2-D conv I, identity          :", conv2d_valid(I2, K2).tolist())
    print("2-D window (0,1) terms        :", (I2[0:2, 1:3] * K2).tolist())
    Ka = np.array([[1, 2], [0, 0]])
    for name, KK in (("identity", K2), ("[[1,2],[0,0]]", Ka)):
        F = conv2d_full(I2, KK)
        k = KK.shape[0]
        print(f"2-D full conv I*{name}: {F.tolist()}  entries [k-1:H, k-1:W] = {F[k-1:3, k-1:3].tolist()}"
              f"  corr with rot180 = {corr2d(I2, np.rot90(KK, 2)).tolist()}")
        assert np.array_equal(F[k - 1:3, k - 1:3], corr2d(I2, np.rot90(KK, 2)))
    print("(I*K)[0,0] by definition (Ka) :", conv2d_full(I2, Ka)[0, 0])
    print("full corr x*w, w*x            :", corr1d_full(X1, W1).tolist(), corr1d_full(W1, X1).tolist())
    assert np.array_equal(corr1d_full(X1, W1), np.correlate(X1, W1, "full"))
    print("conv commutative              :", conv1d_full(X1, W1).tolist(), conv1d_full(W1, X1).tolist())
    # practice
    print("P1 x=[1,0,2], w=[3,4] conv    :", conv1d_valid(np.array([1, 0, 2]), np.array([3, 4])).tolist(),
          "corr:", corr1d(np.array([1, 0, 2]), np.array([3, 4])).tolist())
    Hp = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    Xp = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    print("P2 Prewitt H on KP03 patch    : corr", corr2d(Xp, Hp).tolist(), "conv", conv2d_valid(Xp, Hp).tolist(),
          "rot180(H)", np.rot90(Hp, 2).tolist())
    Ka = np.array([[1, 2], [0, 0]])
    print("P3 K=[[1,2],[0,0]] on I       : corr", corr2d(I2, Ka).tolist(), "conv", conv2d_valid(I2, Ka).tolist(),
          "rot180(K)", np.rot90(Ka, 2).tolist())
    xs = np.array([3, 1, 4, 1, 5])
    for k in ([1, 2, 1], [1, 0, -1]):
        k = np.array(k)
        print("P4 kernel", k.tolist(), "corr", corr1d(xs, k).tolist(), "conv", conv1d_valid(xs, k).tolist())
    Fp = conv2d_full(Xp, Hp)
    print("P2 full conv (X*H)[2,2]       :", Fp[2, 2])
    assert Fp[2, 2] == 6 and corr2d(Xp, Hp)[0, 0] == -6
    print("valid lengths L=7,k=3:", 7 - 3 + 1, " 5x5 with 3x3:", 5 - 3 + 1)


# ------------------------------------------------------------------ drawing helpers
def cell(ax, x, y, v, face, c, size=1.0, lw=1.2, fs=None, bold=False):
    ax.add_patch(Rectangle((x, y), size, size, facecolor=face, edgecolor=c["muted"], lw=lw))
    ax.text(x + size / 2, y + size / 2, f"{v}", ha="center", va="center", color=c["text"],
            fontsize=fs, fontweight="bold" if bold else "normal")


def tint(hexcol, a, dark):
    """blend a colour toward the background (light) or keep it translucent-looking (dark)."""
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    base = (0.12, 0.11, 0.10) if dark else (1, 1, 1)
    return tuple(base[i] * (1 - a) + (r, g, b)[i] * a for i in range(3))


# ------------------------------------------------------------------ figures
def fig_flip_1d(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=18)
    W, H = 1000, 400
    fig = plt.figure(figsize=(W / 100, H / 100))
    col = {4: tint(c["edge"], 0.55, dark), 5: tint(c["ref"], 0.45, dark)}
    xface = tint(c["muted"], 0.18, dark)
    # (title, kernel as laid down, output, first output index t, kernel label)
    # convolution keeps the index of its definition: the valid part starts at t = k-1 = 1
    panels = [("convolution $x*w$ (flip $w$ to $[5,4]$)", W1[::-1], conv1d_valid(X1, W1), len(W1) - 1, "$w$ flipped"),
              ("cross-correlation $x\\star w$, $w=[4,5]$", W1, corr1d(X1, W1), 0, "$w$"),
              ("cross-correlation $x\\star\\tilde w$, $\\tilde w=[5,4]$", W1[::-1], corr1d(X1, W1[::-1]), 0, "$\\tilde w$")]
    for p, (title, kern, out, t0, _klab) in enumerate(panels):
        ax = fig.add_axes([0.005 + p * 0.333, 0.02, 0.325, 0.86])
        ax.set_xlim(-1.0, 5.4); ax.set_ylim(-5.3, 1.7); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=14, color=c["text"])
        ax.text(-0.5, 0.5, "$x$", ha="center", va="center", fontsize=17)
        for i, v in enumerate(X1):
            cell(ax, i + 0.2, 0, v, xface, c, fs=17)
        for r in range(2):
            y = -1.4 - 1.3 * r
            for m, v in enumerate(kern):
                cell(ax, r + m + 0.2, y, v, col[int(v)], c, fs=17)
            terms = " + ".join(f"{X1[r + m]}\\cdot{kern[m]}" for m in range(2))
            ax.text(-0.5, y + 0.5, f"$t={t0 + r}$", ha="center", va="center", fontsize=15, color=c["text"])
            ax.text(r + 2.3, y + 0.5, f"${terms}$", ha="left", va="center", fontsize=16, color=c["text"])
        ax.text(-0.5, -4.2 + 0.5, "out", ha="center", va="center", fontsize=15)
        for t, v in enumerate(out):
            cell(ax, t + 0.2, -4.2, v, tint(c["ok"] if p != 1 else c["warn"], 0.35, dark), c, bold=True, fs=17)
        if p == 2:
            ax.text(2.4, -4.2 + 0.5, "= convolution", ha="left", va="center", fontsize=15, color=c["ok"])
    finish(fig, "lec06-kp04-flip-1d", dark, c)
    return W, H


def fig_rotate_180(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    W, H = 760, 380
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.86])
    ax.set_xlim(-0.5, 9.5); ax.set_ylim(-1.1, 3.6); ax.set_aspect("equal"); ax.axis("off")
    cmap = plt.get_cmap("Blues" if not dark else "Oranges")
    R = np.rot90(K9, 2)
    for ox, M, lab in ((0.3, K9, "$K$"), (6.2, R, "$\\tilde K$ (rotated)")):
        for r in range(3):
            for q in range(3):
                v = int(M[r, q])
                face = cmap(0.12 + 0.07 * v)
                cell(ax, ox + q, 2 - r, v, face, c, bold=v in (1, 9))
        ax.text(ox + 1.5, 3.3, lab, ha="center", va="center", fontsize=15)
    ax.add_patch(FancyArrowPatch((3.6, 1.5), (5.9, 1.5), arrowstyle="-|>", mutation_scale=24, lw=2.2, color=c["main"]))
    ax.text(4.75, 2.0, "rotate $180^\\circ$", ha="center", va="center", fontsize=14, color=c["main"])
    ax.text(4.75, -0.6, "$\\tilde K[m,n]=K[k-1-m,\\,k-1-n]$", ha="center", va="center", fontsize=12, color=c["muted"])
    finish(fig, "lec06-kp04-rotate-180", dark, c)
    return W, H


def fig_2d_correlation(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    W, H = 900, 380
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.86])
    ax.set_xlim(-0.6, 12.6); ax.set_ylim(-1.4, 3.7); ax.set_aspect("equal"); ax.axis("off")
    S = corr2d(I2, K2)
    hi = tint(c["edge"], 0.5, dark); base = tint(c["muted"], 0.15, dark)
    for r in range(3):
        for q in range(3):
            inwin = r in (0, 1) and q in (1, 2)
            cell(ax, q, 2 - r, int(I2[r, q]), hi if inwin else base, c, bold=inwin)
    ax.add_patch(Rectangle((1, 1), 2, 2, fill=False, edgecolor=c["main"], lw=3))
    ax.text(1.5, 3.35, "$I$", ha="center", fontsize=15)
    ax.text(3.75, 1.5, "$\\star$", ha="center", va="center", fontsize=26)
    for r in range(2):
        for q in range(2):
            cell(ax, 4.5 + q, 1.5 - r, int(K2[r, q]), tint(c["ref"], 0.4 if K2[r, q] else 0.12, dark), c)
    ax.text(5.5, 3.35, "$K$", ha="center", fontsize=15)
    ax.text(7.25, 1.5, "$=$", ha="center", va="center", fontsize=24)
    for r in range(2):
        for q in range(2):
            on = (r, q) == (0, 1)
            cell(ax, 8 + q, 1.5 - r, int(S[r, q]), tint(c["ok"], 0.55 if on else 0.2, dark), c, bold=on)
    ax.add_patch(Rectangle((9, 1.5), 1, 1, fill=False, edgecolor=c["main"], lw=3))
    ax.text(9.0, 3.35, "$s=I\\star K$", ha="center", fontsize=15)
    t = I2[0:2, 1:3]
    ax.text(5.6, -0.9, f"$s[0,1]={t[0,0]}\\cdot1+{t[0,1]}\\cdot0+{t[1,0]}\\cdot0+{t[1,1]}\\cdot1={S[0,1]}$",
            ha="center", va="center", fontsize=14, color=c["main"])
    finish(fig, "lec06-kp04-2d-correlation", dark, c)
    return W, H


if __name__ == "__main__":
    numbers()
    for dark in (False, True):
        fig_flip_1d(dark)
        fig_rotate_180(dark)
        fig_2d_correlation(dark)
    print("wrote lec06-kp04-flip-1d, lec06-kp04-rotate-180, lec06-kp04-2d-correlation (.png light, .svg dark)")
