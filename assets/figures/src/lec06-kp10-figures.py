"""KP10 figures (infinitely strong prior, dynamic pooling, SPP, global average pooling) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec06-kp10-prior-matrices, lec06-kp10-gap.
Prints every number the lecture quotes (all computed here, nothing typed by hand) and asserts them.

Numbers
  * an 8 x 8 weight matrix (8 outputs, 8 inputs): fully connected 64 free weights; a width-3 band has 22 cells that may be non-zero
    and 42 cells fixed to 0 (an infinitely strong prior); sharing the band leaves 3 free weights.
  * global average pooling: a 7 x 7 x 512 map -> 512 numbers; the weights of a 1000-class layer: 25,088,000 -> 512,000 (49 x fewer, 98 % less).
  * spatial pyramid pooling with 4 x 4, 2 x 2, 1 x 1 bins: 21 bins per channel whatever the size of the map (13 x 13 or 10 x 10).
  * dynamic pooling: k-means (k = 2) on 8 feature positions groups them by content, a fixed grid of width 4 cuts the first group in two (maxima 8, 6, 9 against 8, 9).
  * practice: P1 band counts, P2 GAP weights, P3 GAP of three 2 x 2 maps, P4 SPP bin sizes.
Run:  py assets/figures/src/lec06-kp10-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from lec04_common import DARK, LIGHT, finish, skin  # noqa: E402


def band(n, k):
    M = np.zeros((n, n), int)
    for i in range(n):
        for j in range(n):
            if abs(i - j) <= k // 2: M[i, j] = 1
    return M


def bins(n, b):
    """window and stride of one SPP level with b bins on a map of side n."""
    win = int(np.ceil(n / b)); st = n // b
    return win, st


def kmeans1d(x, k=2, it=20):
    x = np.asarray(x, float); c = np.array([x.min(), x.max()], float)
    for _ in range(it):
        lab = np.abs(x[:, None] - c[None]).argmin(1); c = np.array([x[lab == j].mean() for j in range(k)])
    return lab


def numbers():
    N = {}
    B8 = band(8, 3); assert int(B8.sum()) == 22 and int((B8 == 0).sum()) == 42 and 8 * 8 == 64
    assert int(band(10, 3).sum()) == 28 and int((band(10, 3) == 0).sum()) == 72
    N["band"] = B8
    flat, gap = 7 * 7 * 512, 512; assert (flat, gap) == (25088, 512) and (flat * 1000, gap * 1000) == (25088000, 512000) and flat // gap == 49
    assert round(100 * (1 - gap / flat), 1) == 98.0
    N["gap"] = dict(flat=flat, gap=gap, w_flat=flat * 1000, w_gap=gap * 1000)
    # SPP
    spp = {}
    for n in (13, 10):
        lv = {}
        for b in (4, 2, 1): w, s = bins(n, b); lv[b] = (w, s, [(i * s, i * s + w - 1) for i in range(b)])
        spp[n] = lv
    assert spp[13][4][:2] == (4, 3) and spp[13][2][:2] == (7, 6) and spp[13][1][:2] == (13, 13) and spp[10][4][:2] == (3, 2) and spp[10][2][:2] == (5, 5) and spp[10][1][:2] == (10, 10)
    assert 16 + 4 + 1 == 21 and 21 * 256 == 5376
    N["spp"] = spp
    # dynamic pooling
    pos = np.array([2, 3, 4, 5, 6, 7, 20, 21]); val = np.array([3, 8, 1, 6, 2, 5, 4, 9]); lab = kmeans1d(pos)
    grid = {}
    for p, v in zip(pos, val): grid.setdefault(int(p) // 4, []).append(int(v))
    km = {int(j): [int(v) for v, l in zip(val, lab) if l == j] for j in (0, 1)}
    assert sorted(len(v) for v in km.values()) == [2, 6] and len(grid) == 3 and [max(v) for _, v in sorted(grid.items())] == [8, 6, 9] and sorted(max(v) for v in km.values()) == [8, 9]
    N["dyn"] = dict(pos=pos, val=val, lab=lab, grid=grid, km=km)
    # practice
    assert int(band(8, 3).sum()) == 22 and int(band(10, 3).sum()) == 28
    maps = [np.array([[1, 3], [5, 7]]), np.array([[0, 2], [2, 0]]), np.array([[4, 4], [4, 4]])]; g = [float(m.mean()) for m in maps]; assert g == [4.0, 1.0, 4.0]; N["p3"] = g
    return N


def show(N):
    for k, v in N.items(): print(k, v if not isinstance(v, dict) else {a: (b.tolist() if hasattr(b, "tolist") else b) for a, b in v.items()})


def tint(hexcol, a, dark):
    h = hexcol.lstrip("#"); r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    base = (0.12, 0.11, 0.10) if dark else (1, 1, 1)
    return tuple(base[i] * (1 - a) + (r, g, b)[i] * a for i in range(3))


def fig_prior(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); N = numbers(); W, H = 940, 400; B = N["band"]
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0, 24); ax.set_ylim(0, 10.2); ax.set_aspect("equal"); ax.axis("off")
    rng = np.random.default_rng(3)
    for k, (ttl, col) in enumerate((("weak prior\n(fully connected)", c["ref"]), ("strong prior\n(weights pushed to 0)", c["main"]), ("infinitely strong prior\n(fixed to 0)", c["ok"]))):
        x0 = 0.4 + k * 8.0; ax.text(x0 + 3.2, 9.5, ttl, ha="center", va="center", fontsize=12, color=col, fontweight="bold")
        for i in range(8):
            for j in range(8):
                if k == 0: a = 0.45
                elif k == 1: a = 0.12 + 0.1 * rng.random()
                else: a = 0.6 if B[i, j] else 0.0
                ax.add_patch(Rectangle((x0 + j * 0.8, 7.9 - (i + 1) * 0.8 + 0.4), 0.78, 0.78, facecolor=tint(col, a, dark) if a else tint(c["muted"], 0.05, dark), edgecolor=c["muted"], lw=0.5))
    ax.text(3.6, 0.5, "64 free weights", ha="center", fontsize=12, color=c["ref"]); ax.text(11.6, 0.5, "64 weights, small but free", ha="center", fontsize=12, color=c["main"]); ax.text(19.6, 0.5, "22 free, 42 fixed to $0$", ha="center", fontsize=12, color=c["ok"])
    finish(fig, "lec06-kp10-prior-matrices", dark, c)
    return W, H


def fig_gap(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); N = numbers()["gap"]; W, H = 900, 380
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0, 22); ax.set_ylim(0, 9.5); ax.set_aspect("equal"); ax.axis("off")
    for r, (nm, wts, col, ln) in enumerate((("flatten $7\\times7\\times512=25{,}088$ numbers", N["w_flat"], c["warn"], 17.0), ("global average pooling: $512$ numbers", N["w_gap"], c["ok"], 0.35))):
        y = 6.2 - r * 3.6
        ax.text(0.0, y + 1.6, nm, ha="left", fontsize=13, color=col, fontweight="bold")
        ax.add_patch(Rectangle((0.0, y), max(ln, 0.12), 1.2, facecolor=tint(col, 0.5, dark), edgecolor=col, lw=1.5))
        ax.text(0.0, y - 0.7, f"1000-class layer: {wts:,} weights", ha="left", va="center", fontsize=12, color=c["text"])
    ax.text(0.0, 0.4, "$49\\times$ fewer weights, $98\\%$ less", ha="left", fontsize=12, color=c["muted"])
    finish(fig, "lec06-kp10-gap", dark, c)
    return W, H


if __name__ == "__main__":
    N = numbers(); show(N)
    for f in (fig_prior, fig_gap):
        for dark in (False, True): f(dark)
    print("figures written")
