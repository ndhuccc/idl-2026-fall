"""Figures for Chapter 2 KP02-KP04 (Perceptron learning rule, signum derivations, XOR and generalization limits). Labels are English.

  lec02-kp02-or-training    : OR learned from zero weights; decision line after pass 1, 2, 3
  lec02-kp03-signum-geometry: normal vector and signed distance d = y(w.x+w0)/|w|
  lec02-kp04-xor-limit      : XOR has no separating line; updates per pass never reach 0
  lec02-kp04-two-boundaries : two zero-error boundaries disagree on a new point
Reuses the helpers (themes, save) of lec02-kp01-figures.py; the dark SVG for slides goes to $DECK_WORK/assets/figures/.
"""
import importlib.util
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("figs_common", os.path.join(HERE, "lec02-kp01-figures.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
px, setup, save, arrow = C.px, C.setup, C.save, C.arrow


def step(s):
    return 1 if s >= 0 else 0


def dots(ax, t, pts, bad=(), size=300):
    for (x, y, o) in pts:
        ax.scatter([x], [y], s=size, color=t["main"] if o else "none", edgecolor=t["main"] if o else t["ref"], lw=2.8, zorder=3)
    for (x, y) in bad:
        ax.scatter([x], [y], s=size * 2.6, facecolor="none", edgecolor=t["warn"], lw=2.6, zorder=2)


def or_training(theme):
    C.FM = 1.4 if theme == "dark" else 1.0
    W, H = 1160, 400
    fig, t = setup(theme, W, H)
    fs = px(19)
    OR = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    states = [(1, (0, 1), 0), (2, (1, 1), 0), (3, (1, 1), -1)]
    for k, (n, w, w0) in enumerate(states):
        ax = fig.add_axes([0.05 + k * 0.325, 0.25, 0.27, 0.5])
        ax.set_facecolor("none")
        g = np.linspace(-0.5, 1.5, 200)
        X, Y = np.meshgrid(g, g)
        S = w[0] * X + w[1] * Y + w0
        ax.contourf(X, Y, S, levels=[0, 1e9], colors=[t["main"]], alpha=0.14)
        ax.contour(X, Y, S, levels=[0], colors=[t["edge"]], linewidths=3)
        bad = [(x, y) for (x, y, o) in OR if step(w[0] * x + w[1] * y + w0) != o]
        dots(ax, t, OR, bad)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect("equal")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.tick_params(labelsize=fs)
        ax.set_xlabel(r"$x_1$", fontsize=fs)
        if k == 0:
            ax.set_ylabel(r"$x_2$", fontsize=fs)
        ax.set_title(f"after pass {n}\n" + rf"$\mathbf{{w}}=({w[0]},{w[1]}),\ w_0={w0}$", fontsize=fs, pad=8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.text(0.5, 0.03, "filled = label 1, open = label 0, red ring = wrong", ha="center",
             va="center", fontsize=px(19), color=t["muted"])
    save(fig, "lec02-kp02-or-training", theme)



def signum_geometry(theme):
    C.FM = 1.3 if theme == "dark" else 1.0
    W, H = 760, 560
    fig, t = setup(theme, W, H)
    fs = px(19)
    ax = fig.add_axes([0.09, 0.2, 0.88, 0.77])
    ax.set_facecolor("none")
    w = np.array([1.0, 1.0])
    w0 = -1.5
    nrm = np.linalg.norm(w)
    xs = np.linspace(-0.6, 2.2, 50)
    ax.fill_between(xs, 1.5 - xs, 2.4, color=t["main"], alpha=0.14)
    ax.plot(xs, 1.5 - xs, color=t["edge"], lw=3)
    pts = [(0, 0, -1), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    for (x, y, lab) in pts:
        d = lab * (w[0] * x + w[1] * y + w0) / nrm
        ok = d >= 0
        ax.scatter([x], [y], s=300, color=t["main"] if lab > 0 else "none", edgecolor=t["main"] if lab > 0 else t["ref"], lw=2.8, zorder=3)
        if not ok:
            ax.scatter([x], [y], s=780, facecolor="none", edgecolor=t["warn"], lw=2.6, zorder=2)
        off = {(0, 0): (-0.12, -0.22, "right"), (0, 1): (-0.16, 0.12, "right"), (1, 0): (-0.16, -0.27, "right"), (1, 1): (0.0, 0.17, "center")}[(x, y)]
        ax.text(x + off[0], y + off[1], rf"$d={d:.2f}$", fontsize=fs, color=t["ok"] if ok else t["warn"], ha=off[2])
    foot = np.array([1.0, 0.5])
    u = w / nrm
    arrow(ax, tuple(foot), tuple(foot + 0.8 * u), t["aux"], lw=3, ms=22)
    ax.text(foot[0] + 0.8 * u[0] + 0.06, foot[1] + 0.8 * u[1] + 0.02, r"$\mathbf{w}$", fontsize=px(24), color=t["aux"])
    tan = np.array([1, -1]) / np.sqrt(2)
    a, b = 0.16 * u, 0.16 * tan
    ax.plot([foot[0] + a[0], foot[0] + a[0] + b[0], foot[0] + b[0]], [foot[1] + a[1], foot[1] + a[1] + b[1], foot[1] + b[1]], color=t["text"], lw=1.6)
    ax.set_xlim(-0.6, 2.2)
    ax.set_ylim(-0.6, 2.0)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.tick_params(labelsize=fs)
    ax.set_xlabel(r"$x_1$", fontsize=fs)
    ax.set_ylabel(r"$x_2$", fontsize=fs)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.text(0.55, 0.02, r"$d=y\,(\mathbf{w}^\top\mathbf{x}+w_0)/\|\mathbf{w}\|$ ;  $d<0$ means wrong side (red ring)", ha="center", va="center",
             fontsize=px(19), color=t["muted"])
    save(fig, "lec02-kp03-signum-geometry", theme)



def xor_limit(theme):
    C.FM = 1.5 if theme == "dark" else 1.0
    W, H = 1100, 420
    fig, t = setup(theme, W, H)
    fs = px(19)
    ax = fig.add_axes([0.07, 0.17, 0.36, 0.64])
    ax.set_facecolor("none")
    xs = np.linspace(-0.5, 1.5, 50)
    ax.plot(xs, 0.5 - xs, color=t["edge"], lw=3, ls="--")
    ax.plot(xs, 1.5 - xs, color=t["warn"], lw=3, ls=":")
    dots(ax, t, [(0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1)])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=fs)
    ax.set_xlabel(r"$x_1$", fontsize=fs)
    ax.set_ylabel(r"$x_2$", fontsize=fs)
    ax.set_title("XOR: no line fits", fontsize=fs, pad=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    bx = fig.add_axes([0.58, 0.17, 0.39, 0.64])
    bx.set_facecolor("none")
    mistakes = [3, 3, 4, 4, 4, 4, 4, 4, 4, 4]
    bx.bar(range(1, 11), mistakes, color=t["warn"], width=0.7)
    bx.set_ylim(0, 5.2)
    bx.set_xticks([1, 5, 10])
    bx.set_yticks([0, 2, 4])
    bx.tick_params(labelsize=fs)
    bx.set_xlabel("pass over the 4 samples", fontsize=fs)
    bx.set_ylabel("updates in the pass", fontsize=fs)
    bx.set_title("Perceptron on XOR never stops", fontsize=fs, pad=8)
    for s_ in ("top", "right"):
        bx.spines[s_].set_visible(False)
    save(fig, "lec02-kp04-xor-limit", theme)


def train_boundaries():
    rng = np.random.default_rng(3)
    A = rng.normal([1.5, 1.5], 0.45, (12, 2))
    B = rng.normal([3.5, 3.2], 0.45, (12, 2))
    data = [(p, 0) for p in A] + [(p, 1) for p in B]
    n = len(data)
    out = {}
    for s in (0, 2):
        r = np.random.default_rng(100 + s)
        init = r.normal(0, 1, 3)
        order = list(r.permutation(n))
        w, w0 = init[:2].copy(), float(init[2])
        for _ in range(500):
            ch = 0
            for i in order:
                x, y = data[i]
                o = step(w @ x + w0)
                if o != y:
                    w = w + (y - o) * x
                    w0 += y - o
                    ch += 1
            if ch == 0:
                break
        out[s] = (w, w0)
    return data, out


def two_boundaries(theme):
    C.FM = 1.25 if theme == "dark" else 1.0
    W, H = 1160, 420
    fig, t = setup(theme, W, H)
    fs = px(19)
    data, bs = train_boundaries()
    test = np.array([2.5, 2.4])
    for k, s in enumerate((0, 2)):
        w, w0 = bs[s]
        ax = fig.add_axes([0.06 + k * 0.47, 0.23, 0.42, 0.6])
        ax.set_facecolor("none")
        assert all(step(w @ x + w0) == y for x, y in data)
        for x, y in data:
            ax.scatter([x[0]], [x[1]], s=70, color=t["main"] if y else "none", edgecolor=t["main"] if y else t["ref"], lw=2)
        xs = np.linspace(0, 5, 50)
        ax.plot(xs, -(w[0] * xs + w0) / w[1], color=t["edge"], lw=3)
        p = step(w @ test + w0)
        ax.scatter([test[0]], [test[1]], marker="*", s=420, color=t["ok"], edgecolor=t["text"], lw=1.2, zorder=4)
        ax.text(test[0] + 0.15, test[1] - 0.42, f"new point: class {p}", fontsize=fs, color=t["ok"])
        ax.set_xlim(0.2, 4.8)
        ax.set_ylim(0.4, 4.6)
        ax.tick_params(labelsize=fs)
        ax.set_xlabel(r"$x_1$", fontsize=fs)
        if k == 0:
            ax.set_ylabel(r"$x_2$", fontsize=fs)
        ax.set_title(f"boundary {'AB'[k]}: zero training error", fontsize=fs, pad=8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.text(0.5, 0.025, "same data, same rule; only the starting weights and sample order differ", ha="center", va="center", fontsize=px(19),
             color=t["muted"])
    save(fig, "lec02-kp04-two-boundaries", theme)
    return bs, test


if __name__ == "__main__":
    for th in ("light", "dark"):
        or_training(th)
        signum_geometry(th)
        xor_limit(th)
        bs, test = two_boundaries(th)
    for s, (w, w0) in bs.items():
        print("boundary", "AB"[(0, 2).index(s)], "w=", np.round(w, 3), "w0=", round(w0, 3), "pred", step(w @ test + w0))
    print("ok")

