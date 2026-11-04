"""Week 7 KP06 figures (ResNet: degradation and residual learning) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec07-kp06-block (a residual block with x = 1.0, F = 0.5), lec07-kp06-depth (product of the per-layer derivatives against depth).
Source (CCC-LNS IDL_115 Ch.4 KP16, first half): beyond about 20 layers the training accuracy saturates and then gets worse (degradation);
  H(x) = F(x) + x (formulas 9, 10), identity shortcut, residual block (Fig. 18); easier gradient flow to shallow layers; ResNet-18/34/50/101;
  interactive demo x = 1.0, F = 0.5 -> H = 1.5, and F = 0 -> H = x.
[Added]: the derivative argument with assumed per-layer derivatives, the weight count of a block with C = 64, and all other numbers below.
Run:  py assets/figures/src/lec07-kp06-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402


def numbers():
    N = {}
    x, F = 1.0, 0.5; assert x + F == 1.5 and 1.0 + 0.0 == 1.0
    L = np.arange(0, 21)
    plain = {a: float(a) ** L for a in (0.8, 0.5)}; res = {f: (1 + f) ** L for f in (0.05, -0.05)}
    assert round(plain[0.8][20], 4) == 0.0115 and f"{plain[0.5][20]:.2e}" == "9.54e-07"
    assert round(res[0.05][20], 3) == 2.653 and round(res[-0.05][20], 3) == 0.358
    N["depth"] = (L, plain, res)
    C = 64; w2 = 2 * 9 * C * C; assert w2 == 73728
    N["block_w"] = w2
    # one block: F(x) = relu(conv(relu(conv(x)))) with all weights 0 -> F = 0 -> H = x (identity); a plain stack with all weights 0 gives 0
    assert 0.0 + 1.0 == 1.0
    # forward values of the toy chain: x0 = 1, L blocks each adding F = 0.05 * x_l  -> x_L = 1.05 ** L
    assert round(1.05 ** 10, 3) == 1.629
    return N


def show(N):
    L, plain, res = N["depth"]
    print({a: float(f"{v[20]:.4g}") for a, v in plain.items()}, {f: float(f"{v[20]:.4g}") for f, v in res.items()}, "block weights", N["block_w"])


def box(ax, x, y, w, h, text, col, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="none", ec=col, lw=1.8)); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=col)


def fig_block(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); W, H = 900, 420
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.01, 0.01, 0.98, 0.98]); ax.set_xlim(0, 18); ax.set_ylim(0, 9); ax.axis("off")
    arr = dict(arrowstyle="->", color=c["muted"], lw=1.6)
    ax.text(1.2, 7.6, "$x$", ha="center", fontsize=15, color=c["text"]); ax.annotate("", (3.6, 7.6), (1.7, 7.6), arrowprops=arr)
    box(ax, 3.6, 7.0, 3.0, 1.2, "$3\\times3$ conv, ReLU", c["ref"], 11); ax.annotate("", (7.4, 7.6), (6.6, 7.6), arrowprops=arr)
    box(ax, 7.4, 7.0, 3.0, 1.2, "$3\\times3$ conv", c["ref"], 11); ax.annotate("", (12.9, 6.1), (10.4, 7.2), arrowprops=arr)
    ax.text(11.3, 8.3, "$\\mathcal{F}(x)$", ha="center", fontsize=14, color=c["ref"])
    ax.plot([1.7, 1.7], [7.6, 3.8], color=c["main"], lw=2.2); ax.plot([1.7, 12.9], [3.8, 3.8], color=c["main"], lw=2.2); ax.annotate("", (12.9, 4.7), (12.9, 3.8), arrowprops=dict(arrowstyle="->", color=c["main"], lw=2.2))
    ax.text(6.5, 3.15, "identity shortcut: carries $x$", ha="center", fontsize=12, color=c["main"])
    ax.add_patch(plt.Circle((12.9, 5.4), 0.7, fc="none", ec=c["text"], lw=2)); ax.text(12.9, 5.4, "$+$", ha="center", va="center", fontsize=18, color=c["text"])
    ax.annotate("", (15.2, 5.4), (13.6, 5.4), arrowprops=arr); ax.text(16.2, 5.4, "$\\mathcal{H}(x)$", ha="center", va="center", fontsize=15, color=c["text"])
    ax.text(9.0, 1.9, "$x=1.0,\\ \\mathcal{F}=0.5\\ \\Rightarrow\\ \\mathcal{H}=1.5$        $\\mathcal{F}=0\\ \\Rightarrow\\ \\mathcal{H}=x$", ha="center", fontsize=13, color=c["ok"])
    ax.text(9.0, 0.7, "$\\mathcal{H}(x)=\\mathcal{F}(x)+x$", ha="center", fontsize=14, color=c["text"])
    finish(fig, "lec07-kp06-block", dark, c)
    return W, H


def fig_depth(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); N = numbers(); L, plain, res = N["depth"]; W, H = 860, 400
    fig, ax = axes(W, H, left=96, bottom=64, right=24, top=30); clean(ax)
    ax.semilogy(L, plain[0.5], color=c["warn"], lw=2.5, label="plain layers, derivative $0.5$ each")
    ax.semilogy(L, plain[0.8], color=c["edge"], lw=2.5, label="plain layers, derivative $0.8$ each")
    ax.semilogy(L, res[-0.05], color=c["ref"], lw=2.5, label="residual blocks, $1-0.05$ each")
    ax.semilogy(L, res[0.05], color=c["ok"], lw=2.5, label="residual blocks, $1+0.05$ each")
    ax.set_xlabel("depth (number of layers or blocks)"); ax.set_ylabel("derivative from output to input (log scale)"); ax.set_ylim(1e-7, 20); ax.set_xticks(range(0, 21, 4))
    ax.annotate("$9.5\\times10^{-7}$", (20, 9.54e-7), (14.5, 2e-6), color=c["warn"], fontsize=11, arrowprops=dict(arrowstyle="->", color=c["warn"]))
    ax.annotate("$0.358$", (20, 0.358), (16.2, 0.04), color=c["ref"], fontsize=11, arrowprops=dict(arrowstyle="->", color=c["ref"]))
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    finish(fig, "lec07-kp06-depth", dark, c)
    return W, H


if __name__ == "__main__":
    N = numbers(); show(N)
    for f in (fig_block, fig_depth):
        for dark in (False, True): f(dark)
    print("figures written")
