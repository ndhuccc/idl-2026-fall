"""Week 7 KP03 figures (VGG: stacks of 3x3 kernels) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec07-kp03-rf (receptive field of stacked 3x3 layers), lec07-kp03-params (stack of 3x3 against one big kernel), lec07-kp03-vgg16 (weights of VGG-16).
Source (CCC-LNS IDL_115 Ch.4 KP13): almost all kernels 3x3 (some 1x1), stride 1, padding 1, max pooling 2x2 stride 2, FC 4096-4096-1000,
  no LRN, SGD with momentum 0.9, batch 256, L2 weight decay 5e-4, learning rate 1e-2 divided by 10 on a plateau. The source gives NO layer counts.
[Added] everything below about receptive fields, parameter counts and the VGG-16 configuration (13 conv layers) is computed here.
Run:  py assets/figures/src/lec07-kp03-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402

from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

VGG16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


def rf(L, k=3):
    """receptive field (side) of L stacked k x k stride-1 layers."""
    r = 1
    for _ in range(L): r += k - 1
    return r


def vgg16():
    """per-layer weights (no bias), output side, with size 224 -> 112 -> 56 -> 28 -> 14 -> 7; also the total with biases."""
    n, cin, rows, nb = 224, 3, [], 0
    i = 0
    for v in VGG16:
        if v == "M": n //= 2; continue
        i += 1; rows.append((f"conv{i}", 3 * 3 * cin * v, n, v)); nb += v; cin = v
    assert n == 7 and cin == 512
    fc = [("fc1", 7 * 7 * 512 * 4096), ("fc2", 4096 * 4096), ("fc3", 4096 * 1000)]
    return rows, fc, nb + 4096 + 4096 + 1000


def numbers():
    N = {}
    assert [rf(L) for L in (1, 2, 3, 4)] == [3, 5, 7, 9] and all(rf(L) == 2 * L + 1 for L in range(1, 10))
    C = 64
    stack = {L: 9 * L * C * C for L in (1, 2, 3)}; single = {L: (2 * L + 1) ** 2 * C * C for L in (1, 2, 3)}
    assert stack == {1: 36864, 2: 73728, 3: 110592} and single == {1: 36864, 2: 102400, 3: 200704}
    assert round(18 / 25, 2) == 0.72 and round(27 / 49, 3) == 0.551 and single[3] - stack[3] == 90112
    N["stack"] = stack; N["single"] = single
    rows, fc, nbias = vgg16()
    convw = sum(r[1] for r in rows); fcw = sum(w for _, w in fc)
    assert len(rows) == 13 and [r[1] for r in rows][:3] == [1728, 36864, 73728]
    assert convw == 14710464 and fcw == 123633664 and convw + fcw == 138344128
    assert convw + fcw + nbias == 138357544                       # the well-known total of VGG-16 with biases
    assert round(100 * fcw / (convw + fcw), 1) == 89.4 and fc[0][1] == 102760448 and round(100 * fc[0][1] / (convw + fcw), 1) == 74.3
    N["rows"] = rows; N["fc"] = fc; N["convw"] = convw; N["fcw"] = fcw; N["nbias"] = nbias
    # pooling halves the side: 224 -> 7 after five poolings
    assert [224 // 2 ** i for i in range(6)] == [224, 112, 56, 28, 14, 7]
    # same-size convolution: (n + 2*1 - 3) / 1 + 1 = n ; a 3x3 conv with stride 1 and padding 0 would shrink by 2
    assert all((n + 2 * 1 - 3) // 1 + 1 == n for n in (224, 112, 56, 28, 14, 7)) and (224 - 3) // 1 + 1 == 222
    # training bookkeeping (source: batch 256; ILSVRC about 1.2 million images)
    assert -(-1_200_000 // 256) == 4688 and round(1_200_000 / 256, 1) == 4687.5
    lr = [1e-2 / 10 ** i for i in range(3)]; assert [float(f"{x:.0e}") for x in lr] == [1e-2, 1e-3, 1e-4]
    return N


def show(N):
    print("conv", [(r[0], r[1], r[2]) for r in N["rows"]]); print("fc", N["fc"], "convw", N["convw"], "fcw", N["fcw"], "total", N["convw"] + N["fcw"], "with bias", N["convw"] + N["fcw"] + N["nbias"])
    print("stack", N["stack"], "single", N["single"])


def fig_rf(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); W, H = 860, 400
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.02, 0.02, 0.96, 0.96]); ax.set_xlim(-1.2, 8.2); ax.set_ylim(-0.6, 5.2); ax.axis("off")
    for L in range(4):
        y = 0.2 + 1.35 * L; n = 7 - 2 * L
        for j in range(n):
            x = 0.5 + L + j
            ax.plot(x, y, "o", ms=11, color=c["main"] if (L == 3) else c["ref"], mfc="none" if L else c["ref"], mew=2)
        ax.text(-0.9, y, ["input", "layer 1", "layer 2", "layer 3"][L], ha="right" if False else "left", va="center", fontsize=11, color=c["muted"])
    # draw only the cone of the top cell (layer 3, x = 3.5 + 0)
    cone = {3: [3.5]}
    for L in (3, 2, 1):
        nxt = []
        for x in cone[L]:
            for d in (-1, 0, 1):
                ax.plot([x, x + d], [0.2 + 1.35 * L, 0.2 + 1.35 * (L - 1)], color=c["main"], lw=1.6, alpha=0.8); nxt.append(x + d)
        cone[L - 1] = sorted(set(nxt))
    for x in cone[0]: ax.plot(x, 0.2, "o", ms=11, color=c["main"])
    ax.text(3.5, -0.45, "one cell of layer $3$ sees $7$ input cells: $2L+1$ with $L=3$", ha="center", fontsize=12, color=c["text"])
    ax.text(7.8, 1.55, "a layer-$1$ cell sees $3$", ha="right", fontsize=11, color=c["muted"]); ax.text(7.8, 2.9, "a layer-$2$ cell sees $5$", ha="right", fontsize=11, color=c["muted"])
    finish(fig, "lec07-kp03-rf", dark, c)
    return W, H


def fig_params(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); N = numbers(); W, H = 860, 400
    fig, ax = axes(W, H, left=96, bottom=64, right=24, top=30); clean(ax)
    xs = np.arange(3); w = 0.36
    ax.bar(xs - w / 2, [N["stack"][L] for L in (1, 2, 3)], w, color=c["ok"], label="stack of $3\\times3$ layers")
    ax.bar(xs + w / 2, [N["single"][L] for L in (1, 2, 3)], w, color=c["warn"], label="one big kernel, same receptive field")
    for i, L in enumerate((1, 2, 3)):
        ax.text(i - w / 2, N["stack"][L] + 3000, f"{N['stack'][L]:,}", ha="center", fontsize=10, color=c["text"]); ax.text(i + w / 2, N["single"][L] + 3000, f"{N['single'][L]:,}", ha="center", fontsize=10, color=c["text"])
    ax.set_xticks(xs); ax.set_xticklabels(["$3\\times3$ vs $3\\times3$\n($L=1$)", "$2\\times(3\\times3)$ vs $5\\times5$", "$3\\times(3\\times3)$ vs $7\\times7$"]); ax.set_ylabel("weights ($C=64$ in and out)"); ax.set_ylim(0, 240000)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    finish(fig, "lec07-kp03-params", dark, c)
    return W, H


def fig_vgg16(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); N = numbers(); W, H = 940, 400
    fig, ax = axes(W, H, left=96, bottom=64, right=20, top=36); clean(ax)
    names = [r[0].replace("conv", "c") for r in N["rows"]] + ["fc1", "fc2", "fc3"]
    vals = [r[1] for r in N["rows"]] + [w for _, w in N["fc"]]
    ax.bar(range(16), vals, color=[c["ref"]] * 13 + [c["warn"]] * 3, width=0.65)
    ax.set_yscale("log"); ax.set_ylim(1e3, 6e8); ax.set_xticks(range(16)); ax.set_xticklabels(names, fontsize=10); ax.set_ylabel("weights (log scale)")
    ax.text(6, 1.8e8, "13 conv layers: $14{,}710{,}464$ ($10.6\\%$)", ha="center", fontsize=12, color=c["ref"]); ax.text(13.6, 3.3e8, "3 FC: $123{,}633{,}664$ ($89.4\\%$)", ha="center", fontsize=12, color=c["warn"])
    ax.text(13, 1.45e8, "fc1 alone: $102{,}760{,}448$", ha="center", fontsize=10, color=c["text"])
    finish(fig, "lec07-kp03-vgg16", dark, c)
    return W, H


if __name__ == "__main__":
    N = numbers(); show(N)
    for f in (fig_rf, fig_params, fig_vgg16):
        for dark in (False, True): f(dark)
    print("figures written")
