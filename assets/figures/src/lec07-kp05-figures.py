"""Week 7 KP05 figures (U-Net) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec07-kp05-unet (the U with sizes and channels), lec07-kp05-transpose (a 2x2 transposed convolution with stride 2).
Source (CCC-LNS IDL_115 Ch.4 KP15): contracting path = two 3x3 conv + ReLU, 2x2 max pooling stride 2, channels doubled at each downsampling;
  expansive path = two 3x3 conv + ReLU, 2x2 transposed convolution (no pooling) that halves the channels, concatenation with the cropped feature map of the same level;
  unpadded convolutions (hence the crop); output with c channels (one per class), softmax, pixel-wise cross-entropy; batch size 1; momentum 0.9.
[Added] (not in the source): the size chain 572 -> 388, the channel numbers 64..1024, the parameter count and all examples below.
Run:  py assets/figures/src/lec07-kp05-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from lec04_common import DARK, LIGHT, finish, skin  # noqa: E402

CH = [64, 128, 256, 512, 1024]


def valid3(n, times=2):
    """side after `times` unpadded 3x3 convolutions."""
    return n - 2 * times


def chain(n0=572, padded=False):
    """returns encoder sizes (after the two convs of each level), the bottom, decoder sizes (after up + two convs) and the crops."""
    enc, n = [], n0
    for lvl in range(4):
        n = n if padded else valid3(n); enc.append(n); n //= 2
    bottom = n if padded else valid3(n)
    dec, crops, n = [], [], bottom
    for lvl in range(3, -1, -1):
        n *= 2; crops.append(enc[lvl] - n if not padded else 0)      # total pixels cut from the encoder map of this level
        n = n if padded else valid3(n); dec.append(n)
    return enc, bottom, dec, crops


def transpose2(x, k):
    """2x2 transposed convolution with stride 2: each input value times the kernel, written in its own non-overlapping 2x2 block."""
    h, w = x.shape; out = np.zeros((2 * h, 2 * w))
    for i in range(h):
        for j in range(w): out[2 * i:2 * i + 2, 2 * j:2 * j + 2] += x[i, j] * k
    return out


def unet_weights():
    """weights (no bias) of the original layout: 1 input channel, 2 output classes."""
    c, t = 1, 0
    for ch in CH:
        t += 9 * c * ch + 9 * ch * ch; c = ch
    for ch in CH[-2::-1]:
        t += 2 * 2 * c * ch                 # 2x2 up-convolution c -> ch
        t += 9 * (2 * ch) * ch + 9 * ch * ch  # two 3x3 convs after concatenation (2*ch in)
        c = ch
    t += 1 * 1 * 64 * 2
    return t


def numbers():
    N = {}
    enc, bottom, dec, crops = chain()
    assert enc == [568, 280, 136, 64] and bottom == 28 and dec == [52, 100, 196, 388] and crops == [8, 32, 80, 176]
    assert [(e - 2 * 0) for e in enc][3] - 2 * 4 == 56            # crop 4 on each side: 64 -> 56
    assert [(enc[3] - 56) // 2, (enc[2] - 104) // 2, (enc[1] - 200) // 2, (enc[0] - 392) // 2] == [4, 16, 40, 88]
    N["valid"] = (enc, bottom, dec)
    encp, bp, decp, _ = chain(256, padded=True)
    assert encp == [256, 128, 64, 32] and bp == 16 and decp == [32, 64, 128, 256]
    N["padded"] = (encp, bp, decp)
    x = np.array([[1.0, 2.0], [3.0, 4.0]]); K = np.array([[1.0, 0.0], [0.0, 1.0]]); y = transpose2(x, K)
    assert y.tolist() == [[1, 0, 2, 0], [0, 1, 0, 2], [3, 0, 4, 0], [0, 3, 0, 4]]
    y2 = transpose2(x, np.ones((2, 2))); assert y2.tolist() == [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
    N["tc"] = (x, K, y, y2)
    p = np.array([0.7, 0.5, 0.9, 0.2]); loss = float(np.mean(-np.log(p))); assert abs(loss - 0.6911) < 1e-4
    N["loss"] = (p, loss); assert [round(float(-np.log(v)), 4) for v in p] == [0.3567, 0.6931, 0.1054, 1.6094]
    p2 = np.array([0.5, 0.5, 0.25]); assert abs(float(np.mean(-np.log(p2))) - 0.9242) < 1e-4
    W = unet_weights(); assert W == 31023808
    N["weights"] = W
    assert 3.0e7 < W < 3.2e7
    # channels after each step of the decoder: up-conv halves, concatenation doubles back, two convs bring it down again
    assert [2 * c // 2 for c in CH[-2::-1]] == [512, 256, 128, 64]
    return N


def show(N):
    print("valid", N["valid"]); print("padded", N["padded"]); print("transposed", N["tc"][2].tolist(), N["tc"][3].tolist()); print("loss", N["loss"]); print("weights", N["weights"])


def fig_unet(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); N = numbers(); W, H = 980, 470
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.01, 0.01, 0.98, 0.98]); ax.set_xlim(0, 20); ax.set_ylim(0, 10.3); ax.axis("off")
    enc, bottom, dec = N["valid"]
    xs_l = [0.6, 2.4, 4.2, 6.0]; xs_r = [14.4, 12.6, 10.8, 9.0]; ys = [7.9, 6.0, 4.1, 2.2]
    wd = 1.25
    sizes_l = [f"{enc[i]}" for i in range(4)]
    for i in range(4):
        ax.add_patch(Rectangle((xs_l[i], ys[i]), wd, 0.95, fc="none", ec=c["ref"], lw=2)); ax.text(xs_l[i] + wd / 2, ys[i] + 0.47, f"{CH[i]}", ha="center", va="center", fontsize=11, color=c["ref"])
        ax.text(xs_l[i] + wd / 2, ys[i] + 1.15, f"{enc[i]}²", ha="center", fontsize=10, color=c["muted"])
        ax.add_patch(Rectangle((xs_r[i], ys[i]), wd, 0.95, fc="none", ec=c["ok"], lw=2)); ax.text(xs_r[i] + wd / 2, ys[i] + 0.47, f"{CH[i]}", ha="center", va="center", fontsize=11, color=c["ok"])
        ax.text(xs_r[i] + wd / 2, ys[i] + 1.15, f"{dec[3 - i]}²", ha="center", fontsize=10, color=c["muted"])
        ax.annotate("", (xs_r[i], ys[i] + 0.47), (xs_l[i] + wd, ys[i] + 0.47), arrowprops=dict(arrowstyle="->", color=c["main"], lw=1.4, ls="--"))
    ax.text(10.0, 8.75, "copy and crop (skip connection)", ha="center", fontsize=11, color=c["main"])
    for i in range(3):
        ax.annotate("", (xs_l[i + 1] + 0.2, ys[i + 1] + 0.95), (xs_l[i] + 1.0, ys[i]), arrowprops=dict(arrowstyle="->", color=c["muted"], lw=1.4))
        ax.annotate("", (xs_r[i] + 0.2, ys[i] + 0.95 + 0.0), (xs_r[i + 1] + 1.0, ys[i + 1] + 0.95), arrowprops=dict(arrowstyle="->", color=c["muted"], lw=1.4))
    ax.add_patch(Rectangle((7.6, 0.45), wd, 0.95, fc="none", ec=c["edge"], lw=2)); ax.text(7.6 + wd / 2, 0.92, f"{CH[4]}", ha="center", va="center", fontsize=11, color=c["edge"]); ax.text(7.6 + wd / 2, 1.6, f"{bottom}²", ha="center", fontsize=10, color=c["muted"])
    ax.annotate("", (7.6, 0.92), (6.0 + 1.0, ys[3]), arrowprops=dict(arrowstyle="->", color=c["muted"], lw=1.4)); ax.annotate("", (9.0 + 0.2, ys[3] + 0.0), (7.6 + 1.1, 1.4), arrowprops=dict(arrowstyle="->", color=c["muted"], lw=1.4))
    ax.text(0.6, 10.0, "contracting path: $2\\times(3\\times3)$ conv, then $2\\times2$ pooling", fontsize=11, color=c["ref"]); ax.text(11.2, 10.0, "expansive path: $2\\times2$ up-conv, concat, $2\\times(3\\times3)$ conv", fontsize=11, color=c["ok"])
    ax.text(16.9, 7.0, "output\n$388\\times388$", ha="center", fontsize=11, color=c["text"]); ax.text(1.2, 9.0 if False else 0.4, "input $572\\times572$", fontsize=11, color=c["text"])
    finish(fig, "lec07-kp05-unet", dark, c)
    return W, H


def fig_transpose(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); N = numbers(); x, K, y, y2 = N["tc"]; W, H = 900, 340
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.01, 0.01, 0.98, 0.98]); ax.set_xlim(0, 18); ax.set_ylim(0, 7); ax.axis("off")
    def grid(M, x0, y0, s, col, fs=13):
        h, w = M.shape
        for i in range(h):
            for j in range(w):
                ax.add_patch(Rectangle((x0 + j * s, y0 + (h - 1 - i) * s), s, s, fc="none", ec=col, lw=1.6)); ax.text(x0 + j * s + s / 2, y0 + (h - 1 - i) * s + s / 2, f"{M[i, j]:g}", ha="center", va="center", fontsize=fs, color=c["text"])
    grid(x, 0.6, 2.3, 1.5, c["ref"]); ax.text(2.1, 6.0, "input $2\\times2$", ha="center", fontsize=12, color=c["ref"])
    ax.text(4.2, 3.8, "$\\otimes$", ha="center", fontsize=18, color=c["text"]); grid(K, 4.9, 2.9, 1.0, c["main"]); ax.text(5.9, 6.0, "kernel $2\\times2$", ha="center", fontsize=12, color=c["main"])
    ax.text(7.5, 3.8, "$\\to$", ha="center", fontsize=20, color=c["text"]); grid(y, 8.4, 1.4, 1.2, c["ok"]); ax.text(10.8, 6.6, "output $4\\times4$", ha="center", fontsize=12, color=c["ok"])
    ax.text(14.9, 3.2, "each input value\ntimes the kernel,\nwritten in its own\n$2\\times2$ block", ha="center", fontsize=11, color=c["muted"])
    finish(fig, "lec07-kp05-transpose", dark, c)
    return W, H


if __name__ == "__main__":
    N = numbers(); show(N)
    for f in (fig_unet, fig_transpose):
        for dark in (False, True): f(dark)
    print("figures written")
