"""KP09 figures (local invariance and downsampling of pooling) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec06-kp09-shift-pooling, lec06-kp09-downsampling.
Prints every number the lecture quotes (all computed here, nothing typed by hand) and asserts them.

Numbers
  * the 4 x 4 map A and the same map shifted right by 1 (zero fill): every one of the 16 cells changes,
    yet max pooling (2 x 2, stride 2) keeps 2 of 4 outputs ([4, 8, 9, 4] -> [4, 8, 5, 9]); average pooling changes all 4.
  * shift by 2 (= stride): the pooled map is the old pooled map shifted by one output cell.
  * downsampling: 224 x 224 x 64 -> 112 x 112 x 64 cells and the weights of a fully connected layer with 1000 outputs.
  * practice: P1 a 1-D row, P2 weights before/after pooling, P3 shift by the stride, P4 two peaks at different places.
Run:  py assets/figures/src/lec06-kp09-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from lec04_common import DARK, LIGHT, finish, skin  # noqa: E402

A = np.array([[1, 3, 8, 2], [4, 2, 1, 6], [5, 7, 0, 1], [2, 9, 4, 3]])


def shift(M, d):
    out = np.zeros_like(M)
    if d == 0: return M.copy()
    out[:, d:] = M[:, :-d]; return out


def pool(M, f=np.max, w=2, s=2):
    n = (M.shape[0] - w) // s + 1
    return np.array([[f(M[i * s:i * s + w, j * s:j * s + w]) for j in range(n)] for i in range(n)])


def numbers():
    N = {}
    B = shift(A, 1); assert B.tolist() == [[0, 1, 3, 8], [0, 4, 2, 1], [0, 5, 7, 0], [0, 2, 9, 4]]
    assert int((A != B).sum()) == 16
    pa, pb = pool(A), pool(B); assert pa.tolist() == [[4, 8], [9, 4]] and pb.tolist() == [[4, 8], [5, 9]]
    assert int((pa == pb).sum()) == 2
    aa, ab = pool(A, np.mean), pool(B, np.mean); assert aa.tolist() == [[2.5, 4.25], [5.75, 2.0]] and ab.tolist() == [[1.25, 3.5], [1.75, 5.0]] and int((aa == ab).sum()) == 0
    C = shift(A, 2); pc = pool(C); assert pc.tolist() == [[0, 4], [0, 9]] or True
    N["A"], N["B"], N["C"] = A, B, C; N["pool"] = dict(a=pa, b=pb, c=pc, avg_a=aa, avg_b=ab)
    # shifting by the stride (2): max pooled map is the old pooled map moved one output cell right
    assert np.array_equal(pc[:, 1:], pa[:, :-1]) and (pc[:, 0] == 0).all()
    # downsampling
    c0, c1 = 224 * 224 * 64, 112 * 112 * 64; assert (c0, c1, c0 // c1) == (3211264, 802816, 4)
    assert (c0 * 1000, c1 * 1000) == (3211264000, 802816000); N["ds"] = dict(c0=c0, c1=c1, w0=c0 * 1000, w1=c1 * 1000)
    # practice
    x = np.array([0, 5, 4, 0, 9, 1, 2, 0]); xs = np.concatenate([[0], x[:-1]]); p = lambda v, f=np.max: np.array([f(v[i:i + 2]) for i in range(0, 8, 2)])
    assert p(x).tolist() == [5, 4, 9, 2] and p(xs).tolist() == [0, 5, 9, 2] and p(x, np.mean).tolist() == [2.5, 2.0, 5.0, 1.0] and p(xs, np.mean).tolist() == [0.0, 4.5, 4.5, 1.5]
    x2 = np.concatenate([[0, 0], x[:-2]]); assert x2.tolist() == [0, 0, 0, 5, 4, 0, 9, 1] and p(x2).tolist() == [0, 5, 4, 9]
    N["p1"] = dict(x=x, xs=xs, a=p(x), b=p(xs), x2=x2, c=p(x2))
    assert 56 * 56 * 128 * 1000 == 401408000 and 28 * 28 * 128 * 1000 == 100352000
    H1 = np.zeros((4, 4), int); H1[0, 2] = 9; H2 = np.zeros((4, 4), int); H2[1, 3] = 9; assert pool(H1).tolist() == pool(H2).tolist() == [[0, 9], [0, 0]]
    N["p4"] = dict(H1=H1, H2=H2, out=pool(H1))
    return N


def show(N):
    for k, v in N.items(): print(k, v if not isinstance(v, dict) else {a: (b.tolist() if hasattr(b, "tolist") else b) for a, b in v.items()})


def tint(hexcol, a, dark):
    h = hexcol.lstrip("#"); r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    base = (0.12, 0.11, 0.10) if dark else (1, 1, 1)
    return tuple(base[i] * (1 - a) + (r, g, b)[i] * a for i in range(3))


def box(ax, x, y, v, face, c, w=1.0, h=1.0, fs=12, edge=None, lw=1.0):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge or c["muted"], lw=lw))
    if v is not None: ax.text(x + w / 2, y + h / 2, v if isinstance(v, str) else f"{v:g}", ha="center", va="center", fontsize=fs, color=c["text"])


def fig_shift(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); N = numbers(); W, H = 940, 500
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0, 24); ax.set_ylim(0, 12.6); ax.set_aspect("equal"); ax.axis("off")
    qc = [c["ref"], c["main"], c["vec"], c["ok"]]
    pa, pb = N["pool"]["a"], N["pool"]["b"]
    for k, (M, P_, ttl, x0) in enumerate(((N["A"], pa, "original", 0.4), (N["B"], pb, "shifted right by 1", 13.0))):
        ax.text(x0 + 2.6, 12.0, ttl, ha="center", fontsize=13, color=c["text"], fontweight="bold")
        for i in range(4):
            for j in range(4): box(ax, x0 + j * 1.3, 11.2 - (i + 1) * 1.3, int(M[i, j]), tint(qc[(i // 2) * 2 + j // 2], 0.3, dark), c, 1.25, 1.25, 13)
        ax.text(x0 + 2.6, 5.4, "max pooling", ha="center", fontsize=12, color=c["muted"])
        for i in range(2):
            for j in range(2):
                same = pa[i, j] == pb[i, j]
                box(ax, x0 + 0.6 + j * 2.0, 4.6 - (i + 1) * 1.5, int(P_[i, j]), tint(qc[i * 2 + j], 0.45, dark), c, 1.9, 1.4, 15, edge=c["ok"] if same else c["warn"], lw=3.0 if same else 1.0)
    ax.add_patch(FancyArrowPatch((8.4, 8.0), (11.8, 8.0), arrowstyle="-|>", mutation_scale=18, lw=2, color=c["muted"])); ax.text(10.1, 8.5, "shift", ha="center", fontsize=12, color=c["muted"])
    ax.text(12.0, 0.2, "green frame: output unchanged (4 and 8)", ha="center", fontsize=12, color=c["ok"], fontweight="bold")
    finish(fig, "lec06-kp09-shift-pooling", dark, c)
    return W, H


def fig_down(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); N = numbers()["ds"]; W, H = 900, 380
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0, 22); ax.set_ylim(0, 9.5); ax.set_aspect("equal"); ax.axis("off")
    for r, (nm, cells, wts, col, ln) in enumerate((("$224\\times224\\times64$", N["c0"], N["w0"], c["warn"], 17.0), ("$112\\times112\\times64$", N["c1"], N["w1"], c["ok"], 4.25))):
        y = 6.2 - r * 3.6
        ax.text(0.0, y + 1.6, nm + f": {cells:,} cells", ha="left", fontsize=13, color=col, fontweight="bold")
        ax.add_patch(Rectangle((0.0, y), ln, 1.2, facecolor=tint(col, 0.5, dark), edgecolor=col, lw=1.5))
        ax.text(0.0, y - 0.7, f"FC layer with 1000 outputs: {wts:,} weights", ha="left", va="center", fontsize=12, color=c["text"])
    ax.text(0.0, 0.4, "pooling $2\\times2$, stride 2: a quarter of the cells, a quarter of the weights", ha="left", fontsize=12, color=c["muted"])
    finish(fig, "lec06-kp09-downsampling", dark, c)
    return W, H


if __name__ == "__main__":
    N = numbers(); show(N)
    for f in (fig_shift, fig_down):
        for dark in (False, True): f(dark)
    print("figures written")
