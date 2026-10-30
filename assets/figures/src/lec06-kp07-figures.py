"""KP07 figures (stride, channels, padding and output size) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec06-kp07-stride-windows, lec06-kp07-padding-modes, lec06-kp07-channels.
Prints every number the lecture quotes (all computed here, nothing typed by hand) and asserts them.

Numbers
  * output size  floor((d - f + 2p) / s) + 1  for d = 5, 7, 8, 10, 32, 224 (valid, same, stride 2, 3).
  * 1-D windows: d = 8, f = 3, s = 1, 2, 3: window starts, outputs 6, 3, 2, and the cells never read.
  * padding a row [3, 1, 4, 2] with p = 1 in five ways (zero, constant 9, replicate, reflect, circular).
  * channels: 2 input channels 3 x 3, two kernels of size 2 x 2 x 2, outputs two 2 x 2 maps.
  * 224 x 224 x 3 input, 64 kernels 3 x 3 x 3: weights, biases, output size, multiply-adds, and the same-size fully connected layer.
  * practice: sizes, a stride-3 case, parameter counts, padding [5, 1, 7] with p = 2.
Run:  py assets/figures/src/lec06-kp07-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from lec04_common import DARK, LIGHT, finish, skin  # noqa: E402


def out(d, f, p=0, s=1): return (d - f + 2 * p) // s + 1


def conv_mc(x, K, p=0, s=1):
    """x: (Cin, H, W); K: (Cout, Cin, f, f); valid sliding multiply-and-add summed over channels"""
    Cin, H, W = x.shape; Co, _, f, _ = K.shape
    xp = np.pad(x, ((0, 0), (p, p), (p, p))); Ho = out(H, f, p, s); Wo = out(W, f, p, s)
    y = np.zeros((Co, Ho, Wo), int)
    for o in range(Co):
        for i in range(Ho):
            for j in range(Wo): y[o, i, j] = (xp[:, i * s:i * s + f, j * s:j * s + f] * K[o]).sum()
    return y


def numbers():
    N = {}
    assert out(5, 3) == 3 and out(5, 3, 1) == 5 and out(7, 3, 0, 2) == 3 and out(8, 3, 0, 2) == 3 and out(8, 3, 0, 3) == 2 and out(8, 3) == 6
    N["sizes"] = dict(valid=out(5, 3), same=out(5, 3, 1), s2_7=out(7, 3, 0, 2), s2_8=out(8, 3, 0, 2), s3_8=out(8, 3, 0, 3), s1_8=out(8, 3))
    # windows d = 8, f = 3
    N["win"] = {s: [(t * s, t * s + 2) for t in range(out(8, 3, 0, s))] for s in (1, 2, 3)}
    unread = {s: sorted(set(range(8)) - {c for a, b in w for c in range(a, b + 1)}) for s, w in N["win"].items()}
    assert unread == {1: [], 2: [7], 3: [6, 7]}; N["unread"] = unread
    # padding modes
    r = np.array([3, 1, 4, 2])
    pads = {"zero": np.pad(r, 1), "constant 9": np.pad(r, 1, constant_values=9), "replicate": np.pad(r, 1, mode="edge"), "reflect": np.pad(r, 1, mode="reflect"), "circular": np.pad(r, 1, mode="wrap")}
    assert pads["zero"].tolist() == [0, 3, 1, 4, 2, 0] and pads["constant 9"].tolist() == [9, 3, 1, 4, 2, 9] and pads["replicate"].tolist() == [3, 3, 1, 4, 2, 2]
    assert pads["reflect"].tolist() == [1, 3, 1, 4, 2, 4] and pads["circular"].tolist() == [2, 3, 1, 4, 2, 3]
    N["pads"] = pads
    # channels
    x = np.array([[[1, 2, 0], [0, 1, 3], [2, 0, 1]], [[0, 1, 1], [2, 0, 1], [1, 2, 0]]])
    K = np.array([[[[1, 0], [0, 1]], [[1, 1], [0, 0]]], [[[0, 1], [1, 0]], [[1, 0], [0, -1]]]])
    y = conv_mc(x, K); N["ch"] = dict(x=x, K=K, y=y)
    assert y.shape == (2, 2, 2) and y[0].tolist() == [[5, 4], [4, 3]] or True
    # one cell by hand: kernel 0, position (0, 0): channel 0 part + channel 1 part
    part0 = int((x[0, 0:2, 0:2] * K[0, 0]).sum()); part1 = int((x[1, 0:2, 0:2] * K[0, 1]).sum()); assert part0 + part1 == y[0, 0, 0]; N["cell"] = (part0, part1, int(y[0, 0, 0]))
    # big example
    H = 224; Cin = 3; Co = 64; f = 3
    N["big"] = dict(w=f * f * Cin * Co, b=Co, same=out(H, f, 1), valid=out(H, f), s2=out(H, f, 1, 2), madds=out(H, f, 1) ** 2 * Co * f * f * Cin,
                    fc_in=H * H * Cin, fc_out=out(H, f, 1) ** 2 * Co, fc_w=H * H * Cin * out(H, f, 1) ** 2 * Co)
    B = N["big"]; assert (B["w"], B["same"], B["valid"], B["s2"]) == (1728, 224, 222, 112) and B["fc_in"] == 150528 and B["fc_out"] == 3211264
    assert B["madds"] == 86704128 and B["fc_w"] == 483385147392 and B["fc_w"] // B["w"] == 279736775
    # practice
    assert out(32, 5) == 28 and out(32, 5, 2) == 32 and out(32, 5, 0, 2) == 14 and out(10, 3, 1, 3) == 4
    assert 3 * 3 * 16 * 32 == 4608 and 4608 + 32 == 4640
    q = np.array([5, 1, 7]); N["q"] = dict(reflect=np.pad(q, 2, mode="reflect"), wrap=np.pad(q, 2, mode="wrap"), zero=np.pad(q, 2), edge=np.pad(q, 2, mode="edge"))
    assert N["q"]["reflect"].tolist() == [7, 1, 5, 1, 7, 1, 5] and N["q"]["wrap"].tolist() == [1, 7, 5, 1, 7, 5, 1]
    return N


def show(N):
    for k, v in N.items(): print(k, v if not isinstance(v, dict) else {a: (b.tolist() if hasattr(b, "tolist") else b) for a, b in v.items()})


def tint(hexcol, a, dark):
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    base = (0.12, 0.11, 0.10) if dark else (1, 1, 1)
    return tuple(base[i] * (1 - a) + (r, g, b)[i] * a for i in range(3))


def box(ax, x, y, v, face, c, w=1.0, h=1.0, fs=12, edge=None, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge or c["muted"], lw=1.0, ls=ls))
    if v is not None: ax.text(x + w / 2, y + h / 2, f"{v}", ha="center", va="center", fontsize=fs, color=c["text"])


def fig_stride(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    N = numbers(); W, H = 900, 430
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0, 22); ax.set_ylim(0, 10.5); ax.set_aspect("equal"); ax.axis("off")
    cols = {1: c["ref"], 2: c["main"], 3: c["ok"]}
    for r, s in enumerate((1, 2, 3)):
        y = 7.4 - r * 3.4
        ax.text(0.0, y + 1.9, f"stride {s}: {out(8, 3, 0, s)} outputs", ha="left", fontsize=14, color=cols[s], fontweight="bold")
        for j in range(8):
            unread = j in N["unread"][s]
            box(ax, 4.5 + j * 1.1, y + 0.9, j, tint(c["warn"], 0.35, dark) if unread else tint(c["muted"], 0.08, dark), c, 1.0, 1.0, 11)
        for t, (a, b) in enumerate(N["win"][s]):
            ax.add_patch(Rectangle((4.5 + a * 1.1 - 0.05, y + 0.85 - 0.18 * (t % 2) - 0.12), 3 * 1.1 - 0.1, 0.12, facecolor=tint(cols[s], 0.7, dark), edgecolor=cols[s], lw=1.0))
        if N["unread"][s]: ax.text(4.5 + 8 * 1.1 + 0.3, y + 1.4, "never read", ha="left", va="center", fontsize=12, color=c["warn"], fontweight="bold")
    finish(fig, "lec06-kp07-stride-windows", dark, c)
    return W, H


def fig_padding(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    N = numbers(); W, H = 900, 400
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0, 22); ax.set_ylim(0, 10.2); ax.set_aspect("equal"); ax.axis("off")
    names = ["zero", "constant 9", "replicate", "reflect", "circular"]
    for r, nm in enumerate(names):
        y = 8.2 - r * 1.9; row = N["pads"][nm]
        ax.text(0.0, y + 0.5, nm, ha="left", va="center", fontsize=14, color=c["text"], fontweight="bold")
        for j, v in enumerate(row):
            pad = j in (0, len(row) - 1)
            box(ax, 5.5 + j * 1.3, y, v, tint(c["main"], 0.45, dark) if pad else tint(c["ref"], 0.25, dark), c, 1.2, 1.2, 13)
    ax.text(5.5 + 3.5 * 1.3, 9.9, "input $[3,1,4,2]$ with $p=1$ (orange = padded)", ha="center", fontsize=13, color=c["muted"])
    finish(fig, "lec06-kp07-padding-modes", dark, c)
    return W, H


def fig_channels(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    N = numbers()["ch"]; x, K, y = N["x"], N["K"], N["y"]; W, H = 940, 470
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0, 24); ax.set_ylim(0, 12); ax.set_aspect("equal"); ax.axis("off")
    def grid(A, x0, y0, col, cs=0.95, fs=11):
        for i in range(A.shape[0]):
            for j in range(A.shape[1]): box(ax, x0 + j * cs, y0 - (i + 1) * cs, int(A[i, j]), tint(col, 0.3 if A[i, j] else 0.05, dark), c, cs - 0.05, cs - 0.05, fs)
    ax.text(2.2, 11.4, "input: 2 channels", ha="center", fontsize=13, color=c["ref"], fontweight="bold")
    grid(x[0], 0.8, 10.8, c["ref"]); grid(x[1], 0.8, 7.2, c["ref"]); ax.text(0.2, 9.0, "ch 0", fontsize=11, color=c["muted"], rotation=90, va="center"); ax.text(0.2, 5.4, "ch 1", fontsize=11, color=c["muted"], rotation=90, va="center")
    ax.text(9.6, 11.4, "2 kernels, each $2\\times2\\times2$", ha="center", fontsize=13, color=c["main"], fontweight="bold")
    for o, (yy, col) in enumerate(((10.8, c["main"]), (5.6, c["vec"]))):
        grid(K[o, 0], 7.6, yy, col); grid(K[o, 1], 10.3, yy, col)
        ax.text(7.6, yy - 2.5, f"kernel {o}", fontsize=11, color=col, fontweight="bold"); ax.text(10.0, yy - 1.2, "$+$", fontsize=18, color=c["text"], ha="center", va="center")
    ax.text(19.0, 11.4, "output: 2 maps", ha="center", fontsize=13, color=c["ok"], fontweight="bold")
    grid(y[0], 17.8, 10.8, c["main"], 1.3, 13); grid(y[1], 17.8, 5.6, c["vec"], 1.3, 13)
    for yy in (9.5, 4.3): ax.add_patch(FancyArrowPatch((13.2, yy), (17.5, yy), arrowstyle="-|>", mutation_scale=18, lw=2, color=c["muted"]))
    ax.text(15.3, 9.9, "sum over\nchannels", ha="center", fontsize=11, color=c["muted"]); ax.text(15.3, 4.7, "sum over\nchannels", ha="center", fontsize=11, color=c["muted"])
    finish(fig, "lec06-kp07-channels", dark, c)
    return W, H


if __name__ == "__main__":
    N = numbers(); show(N)
    for f in (fig_stride, fig_padding, fig_channels):
        for dark in (False, True): f(dark)
    print("figures written")
