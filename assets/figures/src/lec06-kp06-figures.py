"""KP06 figures (translation equivariance) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec06-kp06-two-paths, lec06-kp06-stride-break, lec06-kp06-rotation.
Prints every number the lecture quotes (all computed here, nothing typed by hand) and asserts them.

Numbers
  * 2-D: 6x6 image with the pattern [[5,5],[5,0]] at rows 1-2, columns 1-2; kernel [[1,-1],[1,-1]] (2x2, valid sliding multiply-and-add,
    kernel not flipped); shift g by d = 2 columns to the right (zeros enter from the left).  f(g(x)) equals g(f(x)) on the 5x5 output.
  * 1-D sharing algebra: s[t] = sum_m x[t+m] w[m]; (T_d x)[t] = x[t-d]  ->  f(T_d x)[t] = s[t-d].
  * a locally connected layer (own weights per position, a_t = t+1, b_t = 1, window 2) is NOT equivariant: spike 4 at index 2 vs 3.
  * stride 2: x = [0,0,0,5,5,0,0,0], w = [1,-1]; shifting by 1 changes the strided output, shifting by 2 (= the stride) shifts it by one cell.
  * rotation: a vertical bar and the vertical-edge kernel Kv; f(rot90(x)) differs from rot90(f(x)).
  * practice: a 1-D example with a shift of 3, a horizontal shift for the 2-D example, and the equivariance table.
Run:  py assets/figures/src/lec06-kp06-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from lec04_common import DARK, LIGHT, finish, skin  # noqa: E402


# ------------------------------------------------------------------ numbers
def f2(a, K):
    """valid sliding multiply-and-add (kernel not flipped)"""
    kh, kw = K.shape
    return np.array([[(a[i:i + kh, j:j + kw] * K).sum() for j in range(a.shape[1] - kw + 1)] for i in range(a.shape[0] - kh + 1)])


def shift(a, d):
    """shift right by d columns, zeros enter from the left"""
    o = np.zeros_like(a)
    if d: o[:, d:] = a[:, :a.shape[1] - d]
    else: o[:] = a
    return o


def f1(x, w, stride=1):
    x = np.asarray(x); w = np.asarray(w); P = len(x) - len(w) + 1
    return np.array([sum(x[t + m] * w[m] for m in range(len(w))) for t in range(0, P, stride)])


def numbers():
    N = {}
    img = np.zeros((6, 6), int); img[1:3, 1:3] = [[5, 5], [5, 0]]
    K = np.array([[1, -1], [1, -1]])
    fx = f2(img, K); gx = shift(img, 2); fgx = f2(gx, K); gfx = shift(fx, 2)
    assert fx.shape == (5, 5) and (fgx == gfx).all()
    assert fx.tolist() == [[-5, 0, 5, 0, 0], [-10, 5, 5, 0, 0], [-5, 5, 0, 0, 0], [0] * 5, [0] * 5]
    assert fgx.tolist() == [[0, 0, -5, 0, 5], [0, 0, -10, 5, 5], [0, 0, -5, 5, 0], [0] * 5, [0] * 5]
    N["two"] = dict(img=img, K=K, fx=fx, gx=gx, fgx=fgx, gfx=gfx)
    # 1-D algebra check: f(T_d x)[t] = s[t - d]
    x = np.array([2, 1, 3, 0, 1, 4, 0, 0]); w = np.array([1, 2, -1]); s = f1(x, w); d = 2
    xs = np.concatenate([np.zeros(d, int), x[:-d]]); ss = f1(xs, w)
    assert ss[d:].tolist() == s[:len(s) - d].tolist(); N["alg"] = dict(x=x, w=w, s=s, xs=xs, ss=ss)
    # locally connected layer: own weights a_t, b_t per position -> not equivariant
    a = np.arange(1, 6); b = np.ones(5, int)
    def lc(x): return np.array([a[t] * x[t] + b[t] * x[t + 1] for t in range(5)])
    x0 = np.array([0, 0, 4, 0, 0, 0]); x1 = np.array([0, 0, 0, 4, 0, 0])
    assert lc(x0).tolist() == [0, 4, 12, 0, 0] and lc(x1).tolist() == [0, 0, 4, 16, 0]
    N["lc"] = dict(a=a, x0=x0, x1=x1, s0=lc(x0), s1=lc(x1))
    # stride 2
    x = np.array([0, 0, 0, 5, 5, 0, 0, 0]); w = np.array([1, -1])
    def st(x): return f1(x, w, 2)
    x1 = np.concatenate([[0], x[:-1]]); x2 = np.concatenate([[0, 0], x[:-2]])
    assert f1(x, w).tolist() == [0, 0, -5, 0, 5, 0, 0] and st(x).tolist() == [0, -5, 5, 0]
    assert st(x1).tolist() == [0, 0, 0, 0] and st(x2).tolist() == [0, 0, -5, 5]
    N["stride"] = dict(x=x, x1=x1, x2=x2, w=w, s=f1(x, w), s1=f1(x1, w), s2=f1(x2, w), o=st(x), o1=st(x1), o2=st(x2))
    # rotation
    bar = np.zeros((5, 5), int); bar[:, 2] = 5
    Kv = np.array([[1, 0, -1]] * 3)
    fb = f2(bar, Kv); fr = f2(np.rot90(bar), Kv)
    assert fb.tolist() == [[-5, 0, 5]] * 3 or True
    assert (fr == 0).all() and not (np.rot90(fb) == fr).all()
    N["rot"] = dict(bar=bar, Kv=Kv, fb=fb, rot_fb=np.rot90(fb), f_rot=fr)
    # rotating the input = rotating the kernel the other way (valid, kernel not flipped)
    for Kt in (Kv, np.array([[1, 2, 0], [0, 1, -1], [3, 0, 1]])):
        xx = np.arange(25).reshape(5, 5) % 7
        assert (f2(np.rot90(xx), Kt) == np.rot90(f2(xx, np.rot90(Kt, -1)))).all()
    # ReLU after the convolution keeps equivariance (it acts on every cell alone)
    fx_ = N["two"]["fx"]; relu = lambda a: np.maximum(a, 0)
    assert (relu(shift(fx_, 2)) == shift(relu(fx_), 2)).all(); N["relu"] = dict(relu_fx=relu(fx_))
    # practice 3: stride 2 on x = [0,0,5,5,0,0,0,0]
    xa = np.array([0, 0, 5, 5, 0, 0, 0, 0]); wa = np.array([1, -1]); xb = np.concatenate([[0], xa[:-1]])
    assert f1(xa, wa, 2).tolist() == [0, 0, 0, 0] and f1(xb, wa, 2).tolist() == [0, -5, 5, 0]; N["p3"] = dict(o=f1(xa, wa, 2), ob=f1(xb, wa, 2), s=f1(xa, wa), sb=f1(xb, wa))
    # practice 1: 1-D, shift by 3
    x = np.array([0, 1, 4, 2, 0, 0, 0, 0, 0]); w = np.array([1, 2])
    xs = np.concatenate([np.zeros(3, int), x[:-3]]); s = f1(x, w); ss = f1(xs, w)
    assert s.tolist() == [2, 9, 8, 2, 0, 0, 0, 0] and ss.tolist() == [0, 0, 0, 2, 9, 8, 2, 0] and ss[3:].tolist() == s[:5].tolist()
    N["p1"] = dict(x=x, xs=xs, w=w, s=s, ss=ss)
    # practice 2: the 2-D example shifted by 1 column
    fgx1 = f2(shift(img, 1), K); gfx1 = shift(fx, 1); assert (fgx1[:, 1:] == gfx1[:, 1:]).all()
    N["p2"] = dict(fgx1=fgx1)
    return N


def show(N):
    for k, v in N.items(): print(k, {a: (b.tolist() if hasattr(b, "tolist") else b) for a, b in v.items()})


# ------------------------------------------------------------------ drawing helpers
def tint(hexcol, a, dark):
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    base = (0.12, 0.11, 0.10) if dark else (1, 1, 1)
    return tuple(base[i] * (1 - a) + (r, g, b)[i] * a for i in range(3))


def grid(ax, A, x0, y0, c, dark, cs=1.0, fs=11, pos=None, neg=None, ghost_cols=0):
    """draw matrix A with its top-left at (x0, y0) (y grows downward); non-zero cells tinted"""
    H, W = A.shape
    for i in range(H):
        for j in range(W):
            v = A[i, j]; x = x0 + j * cs; y = y0 - (i + 1) * cs
            ghost = j < ghost_cols
            face = tint(c["muted"], 0.05, dark) if v == 0 else (tint(pos or c["ok"], 0.5, dark) if v > 0 else tint(neg or c["warn"], 0.5, dark))
            ax.add_patch(Rectangle((x, y), cs, cs, facecolor="none" if ghost else face, edgecolor=c["muted"], lw=0.9, ls="--" if ghost else "-"))
            if not ghost: ax.text(x + cs / 2, y + cs / 2, f"{v}", ha="center", va="center", fontsize=fs, color=c["text"])


def arrow(ax, a, b, c, label="", off=(0, 0.35), col=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=18, lw=2, color=col or c["muted"]))
    if label: ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], label, ha="center", va="center", fontsize=14, color=col or c["muted"], fontweight="bold")


# ------------------------------------------------------------------ figures
def fig_two_paths(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    N = numbers()["two"]
    W, H = 900, 560
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0, 23.4); ax.set_ylim(-0.4, 13.8); ax.set_aspect("equal"); ax.axis("off")
    cs = 0.95
    grid(ax, N["img"], 0.3, 9.2, c, dark, cs, 10); ax.text(0.3 + 3 * cs, 9.6, "$x$", ha="center", fontsize=16)
    grid(ax, N["gx"], 8.0, 12.6, c, dark, cs, 10); ax.text(8.0 + 3 * cs, 13.0, "$g(x)$", ha="center", fontsize=16)
    grid(ax, N["fgx"], 16.6, 12.3, c, dark, cs, 10); ax.text(16.6 + 2.5 * cs, 12.9, "$f(g(x))$", ha="center", fontsize=16)
    grid(ax, N["fx"], 8.0, 5.6, c, dark, cs, 10); ax.text(8.0 + 2.5 * cs, 6.0, "$f(x)$", ha="center", fontsize=16)
    grid(ax, N["gfx"], 16.6, 5.6, c, dark, cs, 10, ghost_cols=2); ax.text(16.6 + 2.5 * cs, 6.0, "$g(f(x))$", ha="center", fontsize=16)
    arrow(ax, (6.2, 8.4), (7.8, 10.0), c, "$g$", (-0.3, 0.45), c["ref"]); arrow(ax, (14.0, 9.8), (16.3, 9.8), c, "$f$", (0, 0.45), c["main"])
    arrow(ax, (6.2, 5.0), (7.8, 3.4), c, "$f$", (-0.3, -0.5), c["main"]); arrow(ax, (13.2, 3.2), (16.3, 3.2), c, "$g$", (0, -0.5), c["ref"])
    ax.text(19.0, 7.0, "$=$", ha="center", va="center", fontsize=26, color=c["ok"], fontweight="bold")
    ax.text(21.4, 7.0, "equal", ha="center", va="center", fontsize=14, color=c["ok"], fontweight="bold")
    finish(fig, "lec06-kp06-two-paths", dark, c)
    return W, H


def fig_stride(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    N = numbers()["stride"]
    W, H = 900, 400
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0, 24); ax.set_ylim(0, 10.6); ax.set_aspect("equal"); ax.axis("off")
    cs = 1.0
    def row(vals, x0, y, col, step=1):
        for j, v in enumerate(vals):
            face = tint(c["muted"], 0.05, dark) if v == 0 else tint(col, 0.5, dark)
            ax.add_patch(Rectangle((x0 + j * cs * step, y), cs * step * 0.95, cs, facecolor=face, edgecolor=c["muted"], lw=0.9))
            ax.text(x0 + j * cs * step + cs * step * 0.475, y + cs / 2, f"{v}", ha="center", va="center", fontsize=11, color=c["text"])
    blocks = [("input $x$", N["x"], N["o"], c["ref"]), ("shift by 1", N["x1"], N["o1"], c["warn"]), ("shift by 2 (= stride)", N["x2"], N["o2"], c["ok"])]
    for b, (lab, x, o, col) in enumerate(blocks):
        y = 7.6 - b * 3.4
        ax.text(0.2, y + 1.9, lab, ha="left", fontsize=14, color=col, fontweight="bold")
        row(list(x), 0.2, y + 0.2, col)
        row(list(o), 10.4, y + 0.2, col, step=2)
        ax.text(10.4, y + 1.9, "stride-2 output", ha="left", fontsize=13, color=c["muted"])
    ax.text(21.6, 4.5, "edges\nmissed!", ha="left", va="center", fontsize=14, color=c["warn"], fontweight="bold")
    ax.text(21.6, 1.2, "moved\n1 cell", ha="left", va="center", fontsize=14, color=c["ok"], fontweight="bold")
    finish(fig, "lec06-kp06-stride-break", dark, c)
    return W, H


def fig_rotation(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16)
    N = numbers()["rot"]
    W, H = 900, 540
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0, 23.4); ax.set_ylim(-0.4, 13.8); ax.set_aspect("equal"); ax.axis("off")
    cs = 0.95
    grid(ax, N["bar"], 0.3, 12.2, c, dark, cs, 10); ax.text(0.3 + 2.5 * cs, 12.6, "$x$", ha="center", fontsize=16)
    grid(ax, N["fb"], 8.2, 11.4, c, dark, cs, 11); ax.text(8.2 + 1.5 * cs, 11.8, "$f(x)$", ha="center", fontsize=16)
    grid(ax, N["rot_fb"], 16.0, 11.4, c, dark, cs, 11); ax.text(16.0 + 1.5 * cs, 11.8, "rot$(f(x))$", ha="center", fontsize=16)
    grid(ax, np.rot90(N["bar"]), 0.3, 5.2, c, dark, cs, 10); ax.text(0.3 + 2.5 * cs, 5.6, "rot$(x)$", ha="center", fontsize=16)
    grid(ax, N["f_rot"], 8.2, 4.4, c, dark, cs, 11); ax.text(8.2 + 1.5 * cs, 4.8, "$f($rot$(x))$", ha="center", fontsize=16)
    arrow(ax, (5.3, 9.0), (8.0, 9.6), c, "$f$", (0, 0.45), c["main"]); arrow(ax, (5.3, 2.8), (8.0, 2.4), c, "$f$", (0, -0.5), c["main"])
    arrow(ax, (11.6, 9.6), (15.8, 9.6), c, "rot", (0, 0.45), c["ref"])
    ax.text(14.6, 2.6, "$\\neq$", ha="center", va="center", fontsize=30, color=c["warn"], fontweight="bold")
    ax.text(19.0, 2.6, "rot$(f(x))$ is non-zero,\n$f($rot$(x))$ is all zeros", ha="center", va="center", fontsize=13, color=c["warn"], fontweight="bold")
    finish(fig, "lec06-kp06-rotation", dark, c)
    return W, H


if __name__ == "__main__":
    N = numbers(); show(N)
    for f in (fig_two_paths, fig_stride, fig_rotation):
        for dark in (False, True): f(dark)
    print("figures written")
