"""Week 7 KP02 figures (ImageNet, ILSVRC and AlexNet) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec07-kp02-params (weights per layer of AlexNet), lec07-kp02-loss (gradient of -P against -log P).
Every number the lecture quotes is computed here and asserted; nothing is typed by hand.

Source (CCC-LNS IDL_115 Ch.4 KP12) gives the kernels:  conv1 96 x (11x11x3) stride 4;  conv2 256 x (5x5x48);  conv3 384 x (3x3x256);
  conv4 384 x (3x3x192);  conv5 256 x (3x3x192);  FC 4096, 4096, 1000;  about 60 million parameters, 650 thousand neurons;
  LRN k=2, n=5, alpha=1e-4, beta=0.75;  pooling width 3 stride 2 after layers 1, 2, 5.
[Added] (not in the source): the sizes of the feature maps (input 227, paddings 0, 2, 1, 1, 1), hence the 6 x 6 x 256 = 9216 inputs of FC6.
Run:  py assets/figures/src/lec07-kp02-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402

from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402


def out_size(n, k, s, p=0):
    """side of the output of a convolution or pooling: floor((n + 2p - k) / s) + 1; also returns whether it is an integer division."""
    return (n + 2 * p - k) // s + 1, (n + 2 * p - k) % s == 0


def lrn(a, i, k=2.0, n=5, alpha=1e-4, beta=0.75):
    """local response normalisation of the activation of kernel i; a = activations of all kernels at one position; n/2 taken as 2."""
    h = n // 2
    s = sum(a[j] ** 2 for j in range(max(0, i - h), min(len(a) - 1, i + h) + 1))
    return a[i] / (k + alpha * s) ** beta


def numbers():
    N = {}
    # weights per layer from the kernel sizes of the source (no bias)
    conv = [("conv1", 96 * 11 * 11 * 3), ("conv2", 256 * 5 * 5 * 48), ("conv3", 384 * 3 * 3 * 256), ("conv4", 384 * 3 * 3 * 192), ("conv5", 256 * 3 * 3 * 192)]
    assert [w for _, w in conv] == [34848, 307200, 884736, 663552, 442368]
    # [Added] feature-map sizes: 227 -> conv1 55 -> pool 27 -> conv2 27 -> pool 13 -> conv3,4,5 13 -> pool 6
    chain = [("input", 227)]
    n, ok = out_size(227, 11, 4); assert (n, ok) == (55, True); chain.append(("conv1", n))
    n, _ = out_size(n, 3, 2); assert n == 27; chain.append(("pool1", n))
    n, _ = out_size(n, 5, 1, 2); assert n == 27; chain.append(("conv2", n))
    n, _ = out_size(n, 3, 2); assert n == 13; chain.append(("pool2", n))
    for nm in ("conv3", "conv4", "conv5"):
        n, _ = out_size(n, 3, 1, 1); assert n == 13; chain.append((nm, n))
    n, _ = out_size(n, 3, 2); assert n == 6; chain.append(("pool5", n))
    assert out_size(224, 11, 4) == (54, False)          # the source says 224: (224 - 11) / 4 is not an integer
    N["chain"] = chain
    fc_in = 6 * 6 * 256; assert fc_in == 9216
    fc = [("fc6", fc_in * 4096), ("fc7", 4096 * 4096), ("fc8", 4096 * 1000)]
    assert [w for _, w in fc] == [37748736, 16777216, 4096000]
    layers = conv + fc
    tot = sum(w for _, w in layers); convtot = sum(w for _, w in conv); fctot = sum(w for _, w in fc)
    assert (convtot, fctot, tot) == (2332704, 58621952, 60954656)
    assert round(100 * fctot / tot, 1) == 96.2 and round(100 * convtot / tot, 1) == 3.8
    N["layers"] = layers; N["tot"] = tot; N["convtot"] = convtot; N["fctot"] = fctot
    # neurons (feature-map cells) with the sizes above
    cells = [55 * 55 * 96, 27 * 27 * 256, 13 * 13 * 384, 13 * 13 * 384, 13 * 13 * 256, 4096, 4096, 1000]
    assert cells == [290400, 186624, 64896, 64896, 43264, 4096, 4096, 1000] and sum(cells) == 659272
    N["cells"] = sum(cells)
    # dataset sizes and top-k
    assert 1000 * 1000 == 10 ** 6 and abs(1.2e6 / 1000 - 1200) < 1e-9
    # LRN, k=2, n=5, alpha=1e-4, beta=0.75
    alone = [0, 0, 100, 0, 0]; crowd = [100] * 5; ex = [0, 3, 4, 0, 0]
    N["lrn"] = dict(alone=lrn(alone, 2), crowd=lrn(crowd, 2), ex=lrn(ex, 2), const=2 ** -0.75)
    assert round(N["lrn"]["alone"], 1) == 43.9 and round(N["lrn"]["crowd"], 1) == 23.2 and round(N["lrn"]["ex"], 3) == 2.376 and round(2 ** -0.75, 4) == 0.5946
    # loss: -P against -log P for the true class; P = softmax output
    P = np.array([0.01, 0.1, 0.5, 0.9]); g_P = P * (1 - P); g_log = 1 - P            # |d loss / d o_l|
    assert np.allclose(g_P, [0.0099, 0.09, 0.25, 0.09]) and np.allclose(g_log, [0.99, 0.9, 0.5, 0.1])
    N["loss"] = dict(P=P, g_P=g_P, g_log=g_log)
    o = np.array([2.0, 1.0, 0.1]); p = np.exp(o) / np.exp(o).sum(); assert np.allclose(p, [0.6590, 0.2424, 0.0986], atol=1e-4)
    N["softmax"] = p
    return N


def show(N):
    print("chain", N["chain"]); print("layers", N["layers"], "conv", N["convtot"], "fc", N["fctot"], "total", N["tot"], "cells", N["cells"])
    print("lrn", {k: round(v, 4) for k, v in N["lrn"].items()}); print("softmax", np.round(N["softmax"], 4).tolist())


def fig_params(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); N = numbers(); W, H = 900, 400
    fig, ax = axes(W, H, left=96, bottom=60, right=30, top=30); clean(ax)
    names = [n for n, _ in N["layers"]]; vals = [w for _, w in N["layers"]]
    cols = [c["ref"]] * 5 + [c["warn"]] * 3
    ax.bar(range(8), vals, color=cols, width=0.6)
    ax.set_yscale("log"); ax.set_ylim(1e4, 3e8)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.25, f"{v:,}", ha="center", fontsize=9, color=c["text"], rotation=0)
    ax.set_xticks(range(8)); ax.set_xticklabels(names); ax.set_ylabel("weights (log scale)")
    ax.text(2, 1.1e8, "5 conv layers: $2{,}332{,}704$ ($3.8\\%$)", ha="center", fontsize=12, color=c["ref"])
    ax.text(6, 1.1e8, "3 FC layers: $58{,}621{,}952$ ($96.2\\%$)", ha="center", fontsize=12, color=c["warn"])
    finish(fig, "lec07-kp02-params", dark, c)
    return W, H


def fig_loss(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); W, H = 760, 380
    fig, ax = axes(W, H); clean(ax)
    P = np.linspace(0.001, 0.999, 400)
    ax.plot(P, P * (1 - P), lw=2.5, color=c["warn"], label="$-\\mathbb{P}$ : $\\mathbb{P}(1-\\mathbb{P})$")
    ax.plot(P, 1 - P, lw=2.5, color=c["ok"], label="$-\\log\\mathbb{P}$ : $1-\\mathbb{P}$")
    ax.plot([0.01], [0.0099], "o", color=c["warn"]); ax.plot([0.01], [0.99], "o", color=c["ok"])
    ax.annotate("$\\mathbb{P}=0.01$: $0.0099$", (0.01, 0.0099), (0.30, 0.07), color=c["warn"], fontsize=11, arrowprops=dict(arrowstyle="->", color=c["warn"]))
    ax.annotate("$\\mathbb{P}=0.01$: $0.99$", (0.01, 0.99), (0.14, 0.93), color=c["ok"], fontsize=11, arrowprops=dict(arrowstyle="->", color=c["ok"]))
    ax.set_xlabel("probability of the true class $\\mathbb{P}$"); ax.set_ylabel("size of the gradient on $o_\\ell$"); ax.set_ylim(0, 1.1)
    ax.legend(loc="center right", frameon=False, fontsize=11)
    finish(fig, "lec07-kp02-loss", dark, c)
    return W, H


if __name__ == "__main__":
    N = numbers(); show(N)
    for f in (fig_params, fig_loss):
        for dark in (False, True): f(dark)
    print("figures written")
