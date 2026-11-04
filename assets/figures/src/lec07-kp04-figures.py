"""Week 7 KP04 figures (Inception and GoogLeNet) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec07-kp04-modules (early module and final module with 1x1 reductions), lec07-kp04-cost (multiply-adds and output channels).
Source (CCC-LNS IDL_115 Ch.4 KP14): early module = 1x1, 3x3, 5x5 convolutions and a 3x3 max pooling side by side, concatenated; the final module adds a 1x1
  reduction before the 3x3 and 5x5 convolutions and after the pooling; GoogLeNet = stacked modules, ReLU everywhere, input 224x224x3 minus the RGB mean.
  The source gives NO numbers of channels, layers or parameters.
[Added] the example of the module "3a" (input 28x28x192; 1x1: 64, 3x3 reduce: 96, 3x3: 128, 5x5 reduce: 16, 5x5: 32, pool projection: 32) and all counts below.
Run:  py assets/figures/src/lec07-kp04-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

H = W_ = 28
CIN = 192
C1, R3, C3, R5, C5, PP = 64, 96, 128, 16, 32, 32


def macs(h, w, k, cin, cout):
    """multiply-adds of a k x k convolution with 'same' output size: h * w * cout * k * k * cin."""
    return h * w * cout * k * k * cin


def weights(k, cin, cout):
    return k * k * cin * cout


def numbers():
    N = {}
    # naive module: 1x1 (64), 3x3 (128), 5x5 (32) directly on 192 channels, plus the 3x3 max pooling which keeps all 192 channels
    naive = dict(c1=macs(H, W_, 1, CIN, C1), c3=macs(H, W_, 3, CIN, C3), c5=macs(H, W_, 5, CIN, C5))
    assert naive == dict(c1=9633792, c3=173408256, c5=120422400)
    naive_total = sum(naive.values()); assert naive_total == 303464448
    out_naive = C1 + C3 + C5 + CIN; assert out_naive == 416
    # final module: 1x1 reductions in front of 3x3 and 5x5, 1x1 projection after the pooling
    red = dict(c1=macs(H, W_, 1, CIN, C1), r3=macs(H, W_, 1, CIN, R3), c3=macs(H, W_, 3, R3, C3), r5=macs(H, W_, 1, CIN, R5), c5=macs(H, W_, 5, R5, C5), pp=macs(H, W_, 1, CIN, PP))
    assert red == dict(c1=9633792, r3=14450688, c3=86704128, r5=2408448, c5=10035200, pp=4816896)
    red_total = sum(red.values()); assert red_total == 128049152 and round(naive_total / red_total, 2) == 2.37
    out_red = C1 + C3 + C5 + PP; assert out_red == 256
    N["naive"] = naive; N["naive_total"] = naive_total; N["red"] = red; N["red_total"] = red_total; N["out_naive"] = out_naive; N["out_red"] = out_red
    # 5x5 branch alone
    five_direct = macs(H, W_, 5, CIN, C5); five_red = macs(H, W_, 1, CIN, R5) + macs(H, W_, 5, R5, C5)
    assert five_direct == 120422400 and five_red == 12443648 and round(five_direct / five_red, 2) == 9.68
    N["five"] = (five_direct, five_red)
    # weights of the whole module
    wn = weights(1, CIN, C1) + weights(3, CIN, C3) + weights(5, CIN, C5)
    wr = weights(1, CIN, C1) + weights(1, CIN, R3) + weights(3, R3, C3) + weights(1, CIN, R5) + weights(5, R5, C5) + weights(1, CIN, PP)
    assert (wn, wr) == (387072, 163328)
    N["w"] = (wn, wr)
    # the cost of a 1x1 reduction in weights: the 5x5 branch
    assert weights(5, CIN, C5) == 153600 and weights(1, CIN, R5) + weights(5, R5, C5) == 15872
    # a 1x1 convolution is a per-position linear map across channels: tiny example 2 positions x 3 channels -> 2 channels
    X = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 1.0]]); Wm = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]); Y = X @ Wm
    assert Y.tolist() == [[4.0, -1.0], [1.0, 0.0]]
    N["one"] = (X, Wm, Y)
    # depth growth over modules is NOT given by the source; nothing else to assert
    return N


def show(N):
    print("naive", N["naive"], N["naive_total"], "channels", N["out_naive"]); print("reduced", N["red"], N["red_total"], "channels", N["out_red"])
    print("5x5 branch", N["five"], "weights", N["w"])


def box(ax, x, y, w, h, text, col, fs=10, fc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc or "none", ec=col, lw=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=col)


def fig_modules(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); W, Hh = 940, 420
    fig = plt.figure(figsize=(W / 100, Hh / 100)); ax = fig.add_axes([0.01, 0.01, 0.98, 0.98]); ax.set_xlim(0, 20); ax.set_ylim(0, 9); ax.axis("off")
    for k, (ttl, red) in enumerate((("early module", False), ("final module", True))):
        x0 = 0.3 + k * 10.0
        ax.text(x0 + 4.6, 8.55, ttl, ha="center", fontsize=13, color=c["text"], fontweight="bold")
        box(ax, x0 + 1.5, 0.2, 6.2, 0.8, "previous layer", c["muted"], 11)
        cols = [c["ref"], c["main"], c["vec"], c["edge"]]
        xs = [x0 + 0.1, x0 + 2.45, x0 + 4.8, x0 + 7.15]
        if not red:
            labs = ["$1\\times1$", "$3\\times3$", "$5\\times5$", "max pool"]
            for i in range(4):
                box(ax, xs[i], 3.2, 2.0, 1.0, labs[i], cols[i], 10); ax.annotate("", (xs[i] + 1.0, 3.2), (x0 + 4.6, 1.0), arrowprops=dict(arrowstyle="->", color=c["muted"], lw=1.2))
                ax.annotate("", (x0 + 4.6, 6.1), (xs[i] + 1.0, 4.2), arrowprops=dict(arrowstyle="->", color=c["muted"], lw=1.2))
        else:
            labs = ["$1\\times1$", "$3\\times3$", "$5\\times5$", "$3\\times3$ max pool"]
            for i in range(4):
                if i in (1, 2):
                    box(ax, xs[i], 2.0, 2.0, 0.85, "$1\\times1$ reduce", c["ok"], 9); box(ax, xs[i], 3.6, 2.0, 0.85, labs[i], cols[i], 10)
                    ax.annotate("", (xs[i] + 1.0, 2.0), (x0 + 4.6, 1.0), arrowprops=dict(arrowstyle="->", color=c["muted"], lw=1.2)); ax.annotate("", (xs[i] + 1.0, 3.6), (xs[i] + 1.0, 2.85), arrowprops=dict(arrowstyle="->", color=c["muted"], lw=1.2))
                    ax.annotate("", (x0 + 4.6, 6.1), (xs[i] + 1.0, 4.45), arrowprops=dict(arrowstyle="->", color=c["muted"], lw=1.2))
                elif i == 0:
                    box(ax, xs[i], 2.0, 2.0, 0.85, labs[i], cols[i], 10); ax.annotate("", (xs[i] + 1.0, 2.0), (x0 + 4.6, 1.0), arrowprops=dict(arrowstyle="->", color=c["muted"], lw=1.2)); ax.annotate("", (x0 + 4.6, 6.1), (xs[i] + 1.0, 2.85), arrowprops=dict(arrowstyle="->", color=c["muted"], lw=1.2))
                else:
                    box(ax, xs[i], 2.0, 2.0, 0.85, "max pool", cols[i], 10); box(ax, xs[i], 3.6, 2.0, 0.85, "$1\\times1$ reduce", c["ok"], 9)
                    ax.annotate("", (xs[i] + 1.0, 2.0), (x0 + 4.6, 1.0), arrowprops=dict(arrowstyle="->", color=c["muted"], lw=1.2)); ax.annotate("", (xs[i] + 1.0, 3.6), (xs[i] + 1.0, 2.85), arrowprops=dict(arrowstyle="->", color=c["muted"], lw=1.2))
                    ax.annotate("", (x0 + 4.6, 6.1), (xs[i] + 1.0, 4.45), arrowprops=dict(arrowstyle="->", color=c["muted"], lw=1.2))
        box(ax, x0 + 1.5, 6.1, 6.2, 0.9, "concatenate along channels", c["text"], 11)
    finish(fig, "lec07-kp04-modules", dark, c)
    return W, Hh


def fig_cost(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); N = numbers(); W, Hh = 900, 400
    fig = plt.figure(figsize=(W / 100, Hh / 100))
    ax = fig.add_axes([0.09, 0.15, 0.40, 0.70]); clean(ax)
    ax.bar([0, 1], [N["naive_total"] / 1e6, N["red_total"] / 1e6], color=[c["warn"], c["ok"]], width=0.55)
    for i, v in enumerate((N["naive_total"], N["red_total"])): ax.text(i, v / 1e6 + 6, f"{v / 1e6:.1f}M", ha="center", fontsize=12, color=c["text"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["early", "with $1\\times1$\nreduction"]); ax.set_ylabel("multiply-adds (millions)"); ax.set_ylim(0, 350); ax.set_title("cost of one module", fontsize=12, color=c["text"])
    ax2 = fig.add_axes([0.60, 0.15, 0.36, 0.70]); clean(ax2)
    ax2.bar([0, 1], [N["out_naive"], N["out_red"]], color=[c["warn"], c["ok"]], width=0.55)
    for i, v in enumerate((N["out_naive"], N["out_red"])): ax2.text(i, v + 8, str(v), ha="center", fontsize=12, color=c["text"])
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["early", "with $1\\times1$\nreduction"]); ax2.set_ylabel("output channels"); ax2.set_ylim(0, 480); ax2.set_title("channels after concatenation", fontsize=12, color=c["text"])
    finish(fig, "lec07-kp04-cost", dark, c)
    return W, Hh


if __name__ == "__main__":
    N = numbers(); show(N)
    for f in (fig_modules, fig_cost):
        for dark in (False, True): f(dark)
    print("figures written")
