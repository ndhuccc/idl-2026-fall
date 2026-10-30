"""KP02 figures (LeNet and the classic CNN skeleton) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec06-kp02-pipeline, lec06-kp02-params.
Prints every number the lecture quotes (all computed here, nothing typed by hand) and asserts them.

Numbers
  * sizes: 28x28x1 -> conv 5x5 pad 2 x6 -> 28x28x6 -> pool -> 14x14x6 -> conv 5x5 x16 -> 10x10x16 -> pool -> 5x5x16 -> 400 -> 120 -> 84 -> 10
  * parameters per layer (conv: k*k*c_in*c_out + c_out; fc: n_in*n_out + n_out; LeNet's pooling kept 2 per map)
    total 61,750; conv+pool 2,616; fully connected 59,134 (95.8%); with parameter-free pooling total 61,706
  * flatten-first MLP 784-120-84-10: 105,214; flatten right after pool 1: 14*14*6*120+120 = 141,240
  * original C3 connection table: 1,516 -> total 60,850 (the figure the original paper quotes)
  * practice: P1 sizes for a 32x32 input, P2 conv 2 as a conv vs as an fc layer, P3 MLP vs LeNet, P4 early flatten.
Run:  py assets/figures/src/lec06-kp02-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from lec04_common import DARK, LIGHT, finish, skin  # noqa: E402


def out_side(n, k, p, s): return (n + 2 * p - k) // s + 1


def conv_p(k, ci, co): return k * k * ci * co + co


def fc_p(i, o): return i * o + o


def numbers():
    N = {}
    s1 = out_side(28, 5, 2, 1); s2 = s1 // 2; s3 = out_side(s2, 5, 0, 1); s4 = s3 // 2
    assert (s1, s2, s3, s4) == (28, 14, 10, 5) and s4 * s4 * 16 == 400
    N["sizes"] = [("Input", 28, 1), ("Conv1", 28, 6), ("Pool1", 14, 6), ("Conv2", 10, 16), ("Pool2", 5, 16)]
    N["acts"] = [28 * 28, 28 * 28 * 6, 14 * 14 * 6, 10 * 10 * 16, 5 * 5 * 16]; assert N["acts"] == [784, 4704, 1176, 1600, 400]
    P = [("Conv1", conv_p(5, 1, 6)), ("Pool1", 6 * 2), ("Conv2", conv_p(5, 6, 16)), ("Pool2", 16 * 2), ("FC1", fc_p(400, 120)), ("FC2", fc_p(120, 84)), ("Out", fc_p(84, 10))]
    assert [v for _, v in P] == [156, 12, 2416, 32, 48120, 10164, 850]
    tot = sum(v for _, v in P); cp = sum(v for n, v in P[:4]); fc = sum(v for n, v in P[4:])
    assert (tot, cp, fc) == (61750, 2616, 59134) and round(100 * fc / tot, 1) == 95.8
    assert tot - 12 - 32 == 61706 and round(100 * fc / 61706, 1) == 95.8
    N["params"] = P; N["tot"], N["cp"], N["fc"] = tot, cp, fc
    mlp = fc_p(784, 120) + fc_p(120, 84) + fc_p(84, 10); assert (fc_p(784, 120), mlp) == (94200, 105214)
    assert mlp - tot == 43464 and round(100 * (mlp - tot) / mlp, 1) == 41.3
    N["mlp"] = mlp
    early = fc_p(14 * 14 * 6, 120); assert early == 141240 and round(early / 48120, 1) == 2.9
    N["early"] = early
    c3 = 6 * (3 * 25 + 1) + 6 * (4 * 25 + 1) + 3 * (4 * 25 + 1) + 1 * (6 * 25 + 1); assert c3 == 1516
    assert 156 + 12 + c3 + 32 + 48120 + 10164 + 850 == 60850; N["c3"] = c3
    # practice
    assert out_side(32, 5, 0, 1) == 28 and 5 * 5 * 6 * 16 == 2400 and 14 * 14 * 6 * (10 * 10 * 16) == 1881600 and 1881600 // 2400 == 784
    return N


def show(N):
    for k, v in N.items(): print(k, v)


def tint(hexcol, a, dark):
    h = hexcol.lstrip("#"); r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    base = (0.12, 0.11, 0.10) if dark else (1, 1, 1)
    return tuple(base[i] * (1 - a) + (r, g, b)[i] * a for i in range(3))


def fig_pipeline(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); N = numbers(); W, H = 980, 430
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0, 49); ax.set_ylim(0, 21.5); ax.axis("off")
    cols = [c["ref"], c["main"], c["vec"], c["main"], c["vec"]]
    xs = [0.8, 8.0, 16.0, 21.4, 26.4]
    names = ["input", "conv 1", "pool 1", "conv 2", "pool 2"]
    for k, ((nm, s, ch), x) in enumerate(zip(N["sizes"], xs)):
        h = 0.6 + s * 0.26; w = h * 0.7
        y0 = 12.0 - h / 2
        for d in range(min(ch, 4)):
            ax.add_patch(Rectangle((x + d * 0.45, y0 + d * 0.45), w, h, facecolor=tint(cols[k], 0.3 + 0.1 * d, dark), edgecolor=cols[k], lw=1.4))
        xc = x + w / 2 + 0.7
        ax.text(xc, 6.4, names[k], ha="center", fontsize=12, color=c["muted"])
        ax.text(xc, 4.6 if k != 3 else 3.1, f"${s}\\times{s}\\times{ch}$", ha="center", fontsize=13, color=c["text"], fontweight="bold")
    ax.text(32.6, 6.4, "flatten", ha="center", fontsize=12, color=c["muted"]); ax.text(32.6, 4.6, "$400$", ha="center", fontsize=13, color=c["text"], fontweight="bold")
    ax.add_patch(Rectangle((31.9, 9.0), 1.4, 9.0, facecolor=tint(c["warn"], 0.3, dark), edgecolor=c["warn"], lw=1.4))
    for k, (n, hh, x) in enumerate((("$120$", 7.5, 36.0), ("$84$", 6.0, 40.4), ("$10$", 3.0, 44.6))):
        ax.add_patch(Rectangle((x, 12.5 - hh / 2 + 2), 1.4, hh, facecolor=tint(c["warn"], 0.3, dark), edgecolor=c["warn"], lw=1.4))
        ax.text(x + 0.7, 4.6, n, ha="center", fontsize=13, color=c["text"], fontweight="bold"); ax.text(x + 0.7, 6.4, ["FC 1", "FC 2", "output"][k], ha="center", fontsize=12, color=c["muted"])
    ax.text(15.0, 20.4, "convolution and pooling: space is kept", ha="center", fontsize=13, color=c["main"], fontweight="bold")
    ax.text(40.2, 20.4, "fully connected: decide", ha="center", fontsize=13, color=c["warn"], fontweight="bold")
    ax.text(24.5, 1.4, "$28\\times28\\times1\\to28\\times28\\times6\\to14\\times14\\times6\\to10\\times10\\times16\\to5\\times5\\times16\\to400\\to120\\to84\\to10$", ha="center", fontsize=12, color=c["muted"])
    finish(fig, "lec06-kp02-pipeline", dark, c)
    return W, H


def fig_params(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); N = numbers(); W, H = 940, 430
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 48); ax.set_ylim(0, 11.5); ax.axis("off")
    mx = max(v for _, v in N["params"])
    for r, (nm, v) in enumerate(N["params"]):
        y = 10.0 - r * 1.45; col = c["ref"] if r < 4 else c["warn"]
        ax.text(0.0, y + 0.45, nm, ha="left", va="center", fontsize=13, color=c["text"], fontweight="bold")
        L = max(0.25, 32.0 * v / mx)
        ax.add_patch(Rectangle((6.0, y), L, 0.9, facecolor=tint(col, 0.5, dark), edgecolor=col, lw=1.4))
        ax.text(6.0 + L + 0.5, y + 0.45, f"{v:,}", ha="left", va="center", fontsize=12, color=c["text"])
    ax.text(0.0, -0.1, f"total {N['tot']:,}: conv + pool {N['cp']:,}, fully connected {N['fc']:,} (95.8%)", ha="left", fontsize=13, color=c["muted"])
    finish(fig, "lec06-kp02-params", dark, c)
    return W, H


if __name__ == "__main__":
    N = numbers(); show(N)
    for f in (fig_pipeline, fig_params):
        for dark in (False, True): f(dark)
    print("figures written")
