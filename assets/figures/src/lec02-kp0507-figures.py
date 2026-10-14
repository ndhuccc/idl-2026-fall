"""Figures for Chapter 2 KP05-KP07 (multilayer network and nonlinearity, activation functions, loss functions). Labels are English.

  lec02-kp05-xor-network      : a 2-2-1 network that computes XOR (lecture only; the slides draw it with native nodes)
  lec02-kp05-collapse-vs-relu : same random weights, identity activation vs ReLU
  lec02-kp06-act-relu-family  : ReLU, Leaky ReLU, ELU, softplus
  lec02-kp07-loss-regression  : MSE, MAE, Huber against the residual
  lec02-kp07-loss-hinge       : hinge loss against y*f with the 0-1 loss
Reuses the helpers of lec02-kp01-figures.py; the dark SVG for slides goes to $DECK_WORK/assets/figures/.
"""
import importlib.util
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("figs_common", os.path.join(HERE, "lec02-kp01-figures.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
px, setup, save, arrow = C.px, C.setup, C.save, C.arrow


def clean(ax, fs):
    ax.set_facecolor("none")
    ax.tick_params(labelsize=fs)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def xor_network(theme):
    if theme == "dark":
        return
    C.FM = 1.0
    W, H = 1100, 420
    fig, t = setup(theme, W, H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    pos = {"x1": (110, 290), "x2": (110, 130), "h1": (500, 300), "h2": (500, 120), "o": (900, 210)}
    names = {"x1": r"$x_1$", "x2": r"$x_2$", "h1": "OR", "h2": "NAND", "o": "AND"}
    cols = {"x1": t["ref"], "x2": t["ref"], "h1": t["main"], "h2": t["main"], "o": t["aux"]}
    edges = [("x1", "h1", "1"), ("x2", "h1", "1"), ("x1", "h2", "-1"), ("x2", "h2", "-1"), ("h1", "o", "1"), ("h2", "o", "1")]
    for a, b, lab in edges:
        (x0, y0), (x1_, y1_) = pos[a], pos[b]
        ax.plot([x0 + 48, x1_ - 52], [y0, y1_], color=t["muted"], lw=2.4, zorder=1)
        k = 0.30 if (a, b) in (("x1", "h1"), ("x2", "h2"), ("h1", "o")) else 0.72
        ax.text(x0 + 48 + k * (x1_ - 52 - x0 - 48), y0 + k * (y1_ - y0) + 14, lab, fontsize=px(20), color=t["text"], ha="center",
                bbox=dict(fc="white", ec="none", pad=1.5))
    for k, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 50, fc=t["soft"], ec=cols[k], lw=3.2, zorder=2))
        ax.text(x, y, names[k], ha="center", va="center", fontsize=px(22), color=t["text"], zorder=3)
    ax.text(500, 388, "hidden layer", ha="center", fontsize=px(20), color=t["muted"])
    ax.text(110, 388, "input", ha="center", fontsize=px(20), color=t["muted"])
    ax.text(900, 388, "output", ha="center", fontsize=px(20), color=t["muted"])
    ax.text(500, 40, "bias: OR $-1$,  NAND $+1.5$,  AND $-2$;  each unit fires when its sum is $\\geq 0$", ha="center", fontsize=px(19),
            color=t["muted"])
    save(fig, "lec02-kp05-xor-network", theme)


def net_out(X, W1, W2, W3, relu):
    h1 = X @ W1
    h2 = (np.maximum(0, h1) if relu else h1) @ W2
    return (np.maximum(0, h2) if relu else h2) @ W3   # no bias, ReLU on both hidden layers, linear output


def collapse_vs_relu(theme):
    C.FM = 1.25 if theme == "dark" else 1.0
    W, H = 1160, 440
    fig, t = setup(theme, W, H)
    fs = px(19)
    rng = np.random.default_rng(2)
    W1 = rng.normal(size=(2, 4))
    W2 = rng.normal(size=(4, 3))
    W3 = rng.normal(size=(3, 1))
    g = np.linspace(-3, 3, 300)
    gx, gy = np.meshgrid(g, g)
    X = np.stack([gx.ravel(), gy.ravel()], axis=1)
    V = W1 @ W2 @ W3
    assert np.abs(net_out(X, W1, W2, W3, False) - X @ V).max() < 1e-9
    for k, (relu, title) in enumerate([(False, "identity activation: collapses to one line"), (True, "ReLU activation: the boundary bends")]):
        ax = fig.add_axes([0.07 + k * 0.48, 0.2, 0.4, 0.62])
        z = net_out(X, W1, W2, W3, relu).reshape(gx.shape)
        ax.contourf(gx, gy, z, levels=[-1e9, 1e-9, 1e9], colors=[t["ref"], t["main"]], alpha=0.22)
        cs = ax.contour(gx, gy, z, levels=[1e-9], colors=[t["edge"]], linewidths=0)
        for sg in cs.allsegs[0]:
            if len(sg) > 40:   # drop tiny contour fragments around flat (dead-ReLU) patches
                ax.plot(sg[:, 0], sg[:, 1], color=t["edge"], lw=3.2, solid_capstyle="round")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$", fontsize=fs)
        if k == 0:
            ax.set_ylabel(r"$x_2$", fontsize=fs)
        ax.set_title(title, fontsize=fs, pad=8)
        clean(ax, fs)
    fig.text(0.5, 0.04, "same random weights $W_1,W_2,W_3$ (3 layers); boundary is where the output equals 0", ha="center", va="center",
             fontsize=px(19), color=t["muted"])
    save(fig, "lec02-kp05-collapse-vs-relu", theme)



def act_relu_family(theme):
    C.FM = 1.4 if theme == "dark" else 1.0
    W, H = 760, 520
    fig, t = setup(theme, W, H)
    fs = px(19)
    ax = fig.add_axes([0.1, 0.16, 0.86, 0.74])
    a = np.linspace(-3, 3, 600)
    ax.plot(a, np.maximum(0, a), color=t["main"], lw=3.2, label="ReLU")
    ax.plot(a, np.where(a < 0, 0.01 * a, a), color=t["ref"], lw=2.4, ls="--", label="Leaky ReLU (0.01)")
    ax.plot(a, np.where(a < 0, np.exp(a) - 1, a), color=t["aux"], lw=2.8, label="ELU ($\\alpha=1$)")
    ax.plot(a, np.log1p(np.exp(a)), color=t["edge"], lw=2.8, ls=":", label="softplus")
    ax.axhline(0, color=t["muted"], lw=1)
    ax.axvline(0, color=t["muted"], lw=1)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.3, 3.2)
    ax.set_xlabel(r"$a$", fontsize=fs)
    ax.set_ylabel(r"$\sigma(a)$", fontsize=fs)
    ax.legend(fontsize=fs * 0.9, loc="upper left", frameon=False)
    clean(ax, fs)
    save(fig, "lec02-kp06-act-relu-family", theme)


def loss_regression(theme):
    C.FM = 1.35 if theme == "dark" else 1.0
    W, H = 760, 520
    fig, t = setup(theme, W, H)
    fs = px(19)
    ax = fig.add_axes([0.1, 0.16, 0.86, 0.74])
    e = np.linspace(-3, 3, 600)
    delta = 1.0
    ax.plot(e, e ** 2, color=t["main"], lw=3.2, label="MSE  $e^2$")
    ax.plot(e, np.abs(e), color=t["ref"], lw=3, label="MAE  $|e|$")
    ax.plot(e, np.where(np.abs(e) <= delta, 0.5 * e ** 2, delta * (np.abs(e) - 0.5 * delta)), color=t["aux"], lw=3, ls="--",
            label=r"Huber  ($\delta=1$)")
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 6)
    ax.set_xlabel(r"residual $e=f_o-y$", fontsize=fs)
    ax.set_ylabel("loss", fontsize=fs)
    ax.legend(fontsize=fs * 0.9, loc="upper center", frameon=False)
    clean(ax, fs)
    save(fig, "lec02-kp07-loss-regression", theme)


def loss_hinge(theme):
    C.FM = 1.35 if theme == "dark" else 1.0
    W, H = 760, 520
    fig, t = setup(theme, W, H)
    fs = px(19)
    ax = fig.add_axes([0.1, 0.16, 0.86, 0.74])
    m = np.linspace(-2, 3, 600)
    ax.plot(m, np.maximum(0, 1 - m), color=t["main"], lw=3.2, label=r"hinge  $[1-yf]_+$")
    ax.plot(m, (m < 0).astype(float), color=t["muted"], lw=2.2, ls=":", label="0-1 loss (mistake count)")
    for v, name in [(1.5, "1.5"), (0.3, "0.3"), (-0.5, "-0.5")]:
        ax.scatter([v], [max(0, 1 - v)], s=110, color=t["edge"], zorder=4)
    ax.text(0.33, 0.78, "0.7", fontsize=fs, color=t["edge"])
    ax.text(-0.45, 1.6, "1.5", fontsize=fs, color=t["edge"])
    ax.text(1.55, 0.1, "0", fontsize=fs, color=t["edge"])
    ax.axvline(1, color=t["ok"], lw=1.6, ls="--")
    ax.text(1.05, 2.0, "margin $m=1$", fontsize=fs, color=t["ok"])
    ax.set_xlim(-2, 3)
    ax.set_ylim(-0.1, 3.1)
    ax.set_xlabel(r"$y\,f_o(\mathbf{x})$", fontsize=fs)
    ax.set_ylabel("loss", fontsize=fs)
    ax.legend(fontsize=fs * 0.9, loc="upper right", frameon=False)
    clean(ax, fs)
    save(fig, "lec02-kp07-loss-hinge", theme)


if __name__ == "__main__":
    for th in ("light", "dark"):
        xor_network(th)
        collapse_vs_relu(th)
        act_relu_family(th)
        loss_regression(th)
        loss_hinge(th)
    print("ok")
