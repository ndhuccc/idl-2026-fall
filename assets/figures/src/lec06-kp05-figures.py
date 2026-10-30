"""KP05 figures (sparse connectivity and parameter sharing) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec06-kp05-three-layers, lec06-kp05-weight-matrix, lec06-kp05-counts, lec06-kp05-k-to-m.
Prints every number the lecture quotes (all computed here, nothing typed by hand) and asserts them.

Numbers
  * 1-D layer with m = 6 inputs and a window of k = 3: the valid output has n = m - k + 1 = 4 units.
    fully connected: m*n weights; locally connected (sparse, not shared): k*n weights; convolutional (sparse and shared): k weights.
    connections (paths from an input into an output) and multiply-adds are m*n, k*n and k*n.
  * 28 x 28 image, 5 x 5 window, valid output 24 x 24 (one kernel, bias excluded).
  * m = 6 with k = 1..6: n = 7 - k, and at k = m every count equals m.
  * the 224 x 224 numbers of the source (50,000 and 5,555) are rechecked.
  * practice: m = 8, k = 3; the three counts for a 10-input row with k = 4.
Run:  py assets/figures/src/lec06-kp05-figures.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402


# ------------------------------------------------------------------ numbers
def counts_1d(m, k):
    n = m - k + 1
    return dict(m=m, k=k, n=n, fc=m * n, local=k * n, conv=k, conn_fc=m * n, conn_sparse=k * n, madds_fc=m * n, madds_sparse=k * n)


def numbers():
    N = {}
    N["row"] = counts_1d(6, 3)
    assert N["row"] == dict(m=6, k=3, n=4, fc=24, local=12, conv=3, conn_fc=24, conn_sparse=12, madds_fc=24, madds_sparse=12)
    # 28 x 28 with 5 x 5, valid output 24 x 24
    H = 28; k = 5; ho = H - k + 1; n = ho * ho; m = H * H
    N["img"] = dict(m=m, n=n, k=k * k, ho=ho, fc=m * n, local=k * k * n, conv=k * k, ratio_fc_conv=m * n // (k * k), ratio_fc_local=m / (k * k))
    assert (m, n, ho) == (784, 576, 24) and N["img"]["fc"] == 451584 and N["img"]["local"] == 14400 and N["img"]["conv"] == 25
    assert N["img"]["ratio_fc_conv"] == 18063 and abs(m * n / 25 - 18063.36) < 1e-9 and m / 25 == 31.36
    # the source's 224 x 224 numbers
    N["src"] = dict(pixels=224 * 224, rgb=224 * 224 * 3, ratio_exact=224 * 224 / 9, ratio_50000=50000 / 9)
    assert N["src"]["pixels"] == 50176 and N["src"]["rgb"] == 150528 and abs(N["src"]["ratio_exact"] - 5575.11) < 0.01 and abs(N["src"]["ratio_50000"] - 5555.56) < 0.01
    # m = 6, k = 1..6
    N["sweep"] = [counts_1d(6, kk) for kk in range(1, 7)]
    assert [r["n"] for r in N["sweep"]] == [6, 5, 4, 3, 2, 1]
    assert [r["fc"] for r in N["sweep"]] == [36, 30, 24, 18, 12, 6] and [r["local"] for r in N["sweep"]] == [6, 10, 12, 12, 10, 6] and [r["conv"] for r in N["sweep"]] == [1, 2, 3, 4, 5, 6]
    assert N["sweep"][-1]["fc"] == N["sweep"][-1]["local"] == N["sweep"][-1]["conv"] == 6
    # convolution written as a matrix (cross-correlation as in KP04): M = 6, k = 3, w = [1, 2, -1]
    import numpy as np
    x = np.array([2, 1, 3, 0, 1, 4]); w = np.array([1, 2, -1]); Wm = np.zeros((4, 6), dtype=int)
    for t in range(4): Wm[t, t:t + 3] = w
    N["ex"] = dict(s=(Wm @ x).tolist(), W=Wm.tolist())
    assert N["ex"]["s"] == np.correlate(x, w, "valid").tolist() == [1, 7, 2, -2]
    assert int((Wm != 0).sum()) == 12 and len(set(Wm[Wm != 0].tolist())) == 3
    # the two savings: sparsity M/k, sharing P, product M*P/k
    r = N["row"]; assert r["fc"] / r["local"] == r["m"] / r["k"] == 2 and r["local"] / r["conv"] == r["n"] == 4 and r["fc"] / r["conv"] == r["m"] * r["n"] / r["k"] == 8
    # practice
    x4 = np.array([0, 5, 5, 0]); W4 = np.array([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]]); N["p4"] = (W4 @ x4).tolist(); assert N["p4"] == [-5, 0, 5] == np.correlate(x4, [1, -1], "valid").tolist()
    N["p3"] = counts_1d(100, 5); assert (N["p3"]["n"], N["p3"]["fc"], N["p3"]["local"], N["p3"]["conv"]) == (96, 9600, 480, 5)
    N["p1"] = counts_1d(8, 3); assert (N["p1"]["n"], N["p1"]["fc"], N["p1"]["local"], N["p1"]["conv"]) == (6, 48, 18, 3)
    N["p2"] = counts_1d(10, 4); assert (N["p2"]["n"], N["p2"]["fc"], N["p2"]["local"], N["p2"]["conv"]) == (7, 70, 28, 4)
    return N


def show(N):
    print("row m=6,k=3:", N["row"])
    print("image 28x28, 5x5:", N["img"])
    print("source 224x224:", N["src"])
    for r in N["sweep"]: print("sweep", r)
    print("p1", N["p1"]); print("p2", N["p2"]); print("p3", N["p3"]); print("p4", N["p4"]); print("ex", N["ex"])


# ------------------------------------------------------------------ figures
def tint_pal(c):
    return [c["main"], c["ref"], c["vec"], c["edge"], c["ok"], c["warn"]]


def fig_three(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    W, Hh = 760, 330
    fig, ax = axes(W, Hh, left=8, bottom=8, right=8, top=8)
    ax.set_xlim(0, W); ax.set_ylim(0, Hh); ax.axis("off")
    m, k = 6, 3; n = m - k + 1
    pal = tint_pal(c)
    titles = [("fully connected", "24 weights"), ("sparse, not shared", "12 weights"), ("sparse and shared", "3 weights")]
    for p in range(3):
        x0 = 10 + p * 250
        xin = [x0 + 20 + i * 36 for i in range(m)]
        xout = [x0 + 20 + 36 + j * 36 for j in range(n)]
        yin, yout = 70, 220
        for j in range(n):
            wins = range(m) if p == 0 else range(j, j + k)
            for i in wins:
                if p == 0: col = c["muted"]; a = 0.45
                elif p == 1: col = pal[(j * k + (i - j)) % 6] if False else pal[(j * 3 + (i - j)) % 6]; a = 0.95
                else: col = pal[i - j]; a = 0.95
                ax.plot([xin[i], xout[j]], [yin + 8, yout - 8], color=col, lw=1.8 if p else 1.0, alpha=a, zorder=2)
        for x in xin: ax.add_patch(Circle((x, yin), 8, fc=c["bg"] if not dark else "none", ec=c["text"], lw=1.6, zorder=3))
        for x in xout: ax.add_patch(Circle((x, yout), 8, fc=c["bg"] if not dark else "none", ec=c["text"], lw=1.6, zorder=3))
        ax.text(x0 + 130, 285, titles[p][0], ha="center", va="center", fontsize=15, color=c["text"], fontweight="bold")
        ax.text(x0 + 130, 30, titles[p][1], ha="center", va="center", fontsize=15, color=c["main"], fontweight="bold")
    finish(fig, "lec06-kp05-three-layers", dark, c)
    return W, Hh


def fig_matrix(dark):
    """The same three layers as P x M weight matrices: zeros = no connection (sparse), equal colours = equal weights (shared)."""
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    W, Hh = 760, 250
    fig, ax = axes(W, Hh, left=8, bottom=8, right=8, top=8)
    ax.set_xlim(0, W); ax.set_ylim(0, Hh); ax.axis("off")
    m, k = 6, 3; n = m - k + 1
    pal = tint_pal(c)
    cell = 30
    titles = ["fully connected", "sparse, not shared", "sparse and shared"]
    for p in range(3):
        x0 = 30 + p * 250; y0 = 70
        idx = 0
        for j in range(n):
            for i in range(m):
                inside = i >= j and i < j + k
                x = x0 + i * cell; y = y0 + (n - 1 - j) * cell
                if p == 0: col = plt.cm.hsv(idx / 24 * 0.85); idx += 1; on = True
                elif p == 1:
                    on = inside
                    if on: col = plt.cm.hsv(idx / 12 * 0.85); idx += 1
                else:
                    on = inside
                    if on: col = pal[i - j]
                if on:
                    ax.add_patch(plt.Rectangle((x, y), cell - 3, cell - 3, fc=col, ec="none", alpha=0.9))
                else:
                    ax.add_patch(plt.Rectangle((x, y), cell - 3, cell - 3, fc="none", ec=c["muted"], lw=0.8, alpha=0.5))
                    ax.text(x + (cell - 3) / 2, y + (cell - 3) / 2, "0", ha="center", va="center", fontsize=11, color=c["muted"])
        ax.text(x0 + m * cell / 2, 215, titles[p], ha="center", va="center", fontsize=15, color=c["text"], fontweight="bold")
        ax.text(x0 + m * cell / 2, 30, ["24 different numbers", "12 different numbers", "3 different numbers"][p], ha="center", va="center", fontsize=14, color=c["main"], fontweight="bold")
    finish(fig, "lec06-kp05-weight-matrix", dark, c)
    return W, Hh


def fig_counts(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    N = numbers()["img"]
    W, Hh = 740, 300
    fig, ax = axes(W, Hh, left=190, bottom=64, right=40, top=20)
    labs = ["fully connected", "sparse, not shared", "sparse and shared"]
    vals = [N["fc"], N["local"], N["conv"]]
    cols = [c["warn"], c["edge"], c["ok"]]
    ax.barh(range(3), vals, color=cols, height=0.55)
    ax.set_xscale("log"); ax.set_xlim(1, 4e6)
    ax.set_yticks(range(3)); ax.set_yticklabels(labs); ax.invert_yaxis()
    for i, v in enumerate(vals): ax.text(v * 1.25, i, f"{v:,}", va="center", color=c["text"], fontsize=15, fontweight="bold")
    ax.set_xlabel("number of weights (log scale)")
    clean(ax)
    finish(fig, "lec06-kp05-counts", dark, c)
    return W, Hh


def fig_sweep(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    S = numbers()["sweep"]
    W, Hh = 640, 340
    fig, ax = axes(W, Hh, left=70, bottom=60, right=24, top=22)
    ks = [r["k"] for r in S]
    for key, col, lab, mk in (("fc", c["warn"], "fully connected", "o"), ("local", c["edge"], "sparse, not shared", "s"), ("conv", c["ok"], "sparse and shared", "^")):
        ax.plot(ks, [r[key] for r in S], color=col, marker=mk, lw=2.4, ms=8, label=lab)
    ax.annotate("k = M: all three equal 6", xy=(6, 6), xytext=(3.9, 23), color=c["text"], fontsize=14, fontweight="bold", arrowprops=dict(arrowstyle="->", color=c["text"], lw=1.6))
    ax.set_xlabel("window size k  (M = 6 inputs, P = 7 - k outputs)"); ax.set_ylabel("number of weights")
    ax.set_xticks(ks); ax.set_ylim(0, 44); ax.legend(frameon=False, loc="upper right", fontsize=13)
    clean(ax)
    finish(fig, "lec06-kp05-k-to-m", dark, c)
    return W, Hh


if __name__ == "__main__":
    N = numbers(); show(N)
    for f in (fig_three, fig_matrix, fig_counts, fig_sweep):
        for dark in (False, True): f(dark)
    print("figures written")
