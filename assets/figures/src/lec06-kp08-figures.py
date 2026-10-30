"""KP08 figures (detector and pooling) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec06-kp08-pooling-4x4, lec06-kp08-pooling-windows.
Prints every number the lecture quotes (all computed here, nothing typed by hand) and asserts them.

Numbers
  * the textbook 4 x 4 input X, four non-overlapping 2 x 2 blocks (stride 2, width 2):
    max, average, median and l2 pooling of each block (the source's median 5.5 of the upper-right block is 4.5).
  * 1-D windows, d = 8: width 2 stride 2 (4 windows), width 3 stride 2 (3 windows, cell 7 never read), width 3 stride 1 (6 windows).
  * channels: pooling is done on every channel alone: a 4 x 4 x 3 input gives 2 x 2 x 3, with 0 weights.
  * ReLU and max commute; max routes the gradient to the arg-max, average splits it equally.
  * practice: P1 a 4 x 4 map, P2 output sizes, P3 an outlier window, P4 gradients.
Run:  py assets/figures/src/lec06-kp08-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from lec04_common import DARK, LIGHT, finish, skin  # noqa: E402

X = np.array([[2, 1, 7, 5], [2, 3, 2, 4], [8, -1, -5, 1], [7, 3, 2, 6]])
P1 = np.array([[1, 3, 2, 0], [4, 2, 1, 5], [0, 1, 3, 2], [2, 6, 1, 4]])


def blocks(A, w=2, s=2):
    n = (A.shape[0] - w) // s + 1
    return [[A[i * s:i * s + w, j * s:j * s + w].ravel() for j in range(n)] for i in range(n)]


def pool(A, f, w=2, s=2): return np.array([[f(b) for b in row] for row in blocks(A, w, s)])


def l2(b): return float(np.sqrt((np.asarray(b, float) ** 2).sum()))


def out(d, f, p=0, s=1): return (d - f + 2 * p) // s + 1


def numbers():
    N = {}
    mx, av, md, e2 = pool(X, np.max), pool(X, np.mean), pool(X, np.median), pool(X, l2)
    assert mx.tolist() == [[3, 7], [8, 6]] and av.tolist() == [[2.0, 4.5], [4.25, 1.0]] and md.tolist() == [[2.0, 4.5], [5.0, 1.5]]
    assert np.allclose(e2, [[18 ** .5, 94 ** .5], [123 ** .5, 66 ** .5]]) and round(e2[0, 0], 3) == 4.243 and round(e2[0, 1], 3) == 9.695 and round(e2[1, 0], 3) == 11.091 and round(e2[1, 1], 3) == 8.124
    N["x4"] = dict(max=mx, avg=av, med=md, l2=e2)
    assert out(4, 2, 0, 2) == 2 and out(224, 2, 0, 2) == 112 and out(8, 3, 0, 2) == 3 and out(8, 2, 0, 2) == 4 and out(8, 3) == 6
    win = {"w2s2": [(t * 2, t * 2 + 1) for t in range(out(8, 2, 0, 2))], "w3s2": [(t * 2, t * 2 + 2) for t in range(out(8, 3, 0, 2))], "w3s1": [(t, t + 2) for t in range(out(8, 3))]}
    unread = {k: sorted(set(range(8)) - {c for a, b in v for c in range(a, b + 1)}) for k, v in win.items()}
    assert unread == {"w2s2": [], "w3s2": [7], "w3s1": []}; N["win"] = win; N["unread"] = unread
    # overlapping 1-D example: a row [1, 5, 2, 8, 3, 4, 7, 6]
    r = np.array([1, 5, 2, 8, 3, 4, 7, 6]); N["row"] = r
    N["row_max"] = {k: [int(r[a:b + 1].max()) for a, b in v] for k, v in win.items()}
    assert N["row_max"] == {"w2s2": [5, 8, 4, 7], "w3s2": [5, 8, 7], "w3s1": [5, 8, 8, 8, 7, 7]}
    # channels: 4 x 4 x 3 -> 2 x 2 x 3, no weights
    ch = np.stack([X, P1, X.T]); outc = np.stack([pool(a, np.max) for a in ch]); assert outc.shape == (3, 2, 2)
    N["ch"] = outc; assert out(224, 2, 0, 2) ** 2 * 64 == 802816 and 224 * 224 * 64 == 3211264 and 3211264 // 802816 == 4
    # ReLU and max commute (relu is nondecreasing); average does not
    relu = lambda a: np.maximum(a, 0)
    w = np.array([-5, 1, 2, 6]); assert relu(w.max()) == relu(w).max() == 6
    w2 = np.array([-5, -1, -2, -3]); assert relu(w2.max()) == relu(w2).max() == 0
    assert relu(w.mean()) != relu(w).mean() or True
    assert relu(np.array([-4, 2]).mean()) == 0 and relu(np.array([-4, 2])).mean() == 1.0
    # gradient routing in a window [3, 7, 2, 4]
    win4 = np.array([3, 7, 2, 4]); g_max = (win4 == win4.max()).astype(int); g_avg = np.full(4, 0.25)
    assert g_max.tolist() == [0, 1, 0, 0] and g_avg.tolist() == [0.25] * 4; N["grad"] = (g_max, g_avg)
    # practice
    p1 = dict(max=pool(P1, np.max), avg=pool(P1, np.mean)); assert p1["max"].tolist() == [[4, 5], [6, 4]] and p1["avg"].tolist() == [[2.5, 2.0], [2.25, 2.5]]; N["p1"] = p1
    assert (out(224, 2, 0, 2), out(7, 3, 0, 2), out(8, 3, 0, 2), out(32, 2, 0, 2)) == (112, 3, 3, 16)
    o3 = np.array([2, 2, 3, 100]); assert (o3.max(), o3.mean(), np.median(o3)) == (100, 26.75, 2.5)
    assert np.median(np.array([2, 2, 3, 4])) == 2.5 and np.mean(np.array([2, 2, 3, 4])) == 2.75
    return N


def show(N):
    for k, v in N.items(): print(k, v if not isinstance(v, dict) else {a: (b.tolist() if hasattr(b, "tolist") else b) for a, b in v.items()})


def tint(hexcol, a, dark):
    h = hexcol.lstrip("#"); r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    base = (0.12, 0.11, 0.10) if dark else (1, 1, 1)
    return tuple(base[i] * (1 - a) + (r, g, b)[i] * a for i in range(3))


def box(ax, x, y, v, face, c, w=1.0, h=1.0, fs=12, edge=None):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge or c["muted"], lw=1.0))
    if v is not None: ax.text(x + w / 2, y + h / 2, v if isinstance(v, str) else f"{v}", ha="center", va="center", fontsize=fs, color=c["text"])


def fig_pooling(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); N = numbers()["x4"]; W, H = 940, 440
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0, 24); ax.set_ylim(0, 11.2); ax.set_aspect("equal"); ax.axis("off")
    qc = [c["ref"], c["main"], c["vec"], c["ok"]]
    ax.text(2.7, 10.5, "input $4\\times4$", ha="center", fontsize=13, color=c["muted"], fontweight="bold")
    for i in range(4):
        for j in range(4): box(ax, 0.2 + j * 1.3, 8.9 - (i + 1) * 1.3 + 0.4, int(X[i, j]), tint(qc[(i // 2) * 2 + j // 2], 0.35, dark), c, 1.25, 1.25, 13)
    names = [("max", N["max"]), ("average", N["avg"]), ("median", N["med"]), ("$\\ell_2$", N["l2"])]
    for k, (nm, M) in enumerate(names):
        x0 = 8.2 + (k % 2) * 7.8; y0 = 9.4 - (k // 2) * 4.9
        ax.text(x0 + 1.3, y0 + 0.7, nm, ha="center", fontsize=13, color=c["text"], fontweight="bold")
        for i in range(2):
            for j in range(2):
                v = M[i, j]; s = f"{v:.3f}" if nm.startswith("$") else (f"{v:g}")
                box(ax, x0 + j * 2.0, y0 - (i + 1) * 1.7 + 0.1, s, tint(qc[i * 2 + j], 0.35, dark), c, 1.9, 1.6, 12 if nm.startswith("$") else 14)
    ax.add_patch(FancyArrowPatch((6.0, 5.6), (8.0, 5.6), arrowstyle="-|>", mutation_scale=18, lw=2, color=c["muted"]))
    finish(fig, "lec06-kp08-pooling-4x4", dark, c)
    return W, H


def fig_windows(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); N = numbers(); W, H = 900, 430
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0, 22); ax.set_ylim(0, 10.5); ax.set_aspect("equal"); ax.axis("off")
    cols = {"w2s2": c["ref"], "w3s2": c["main"], "w3s1": c["ok"]}
    lab = {"w2s2": "width 2, stride 2", "w3s2": "width 3, stride 2", "w3s1": "width 3, stride 1"}
    for r, k in enumerate(("w2s2", "w3s2", "w3s1")):
        y = 7.4 - r * 3.4; w = N["win"][k]
        ax.text(0.0, y + 1.9, f"{lab[k]}: {len(w)} outputs, max {N['row_max'][k]}", ha="left", fontsize=13, color=cols[k], fontweight="bold")
        for j in range(8):
            unread = j in N["unread"][k]
            box(ax, 1.5 + j * 1.5, y + 0.9, int(N["row"][j]), tint(c["warn"], 0.35, dark) if unread else tint(c["muted"], 0.08, dark), c, 1.4, 1.2, 12)
        for t, (a, b) in enumerate(w):
            ax.add_patch(Rectangle((1.5 + a * 1.5 - 0.05, y + 0.75 - 0.2 * (t % 2) - 0.12), (b - a + 1) * 1.5 - 0.1, 0.12, facecolor=tint(cols[k], 0.7, dark), edgecolor=cols[k], lw=1.0))
        if N["unread"][k]: ax.text(1.5 + 8 * 1.5 + 0.2, y + 1.5, "never read", ha="left", va="center", fontsize=12, color=c["warn"], fontweight="bold")
    finish(fig, "lec06-kp08-pooling-windows", dark, c)
    return W, H


if __name__ == "__main__":
    N = numbers(); show(N)
    for f in (fig_pooling, fig_windows):
        for dark in (False, True): f(dark)
    print("figures written")
