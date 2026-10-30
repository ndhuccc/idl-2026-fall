"""Figures for Chapter 2 KP08-KP10 (ADALINE and logistic regression, RBF and SOM, universal approximation). Labels are English.

  lec02-kp08-adaline-or       : OR data, Perceptron boundary vs ADALINE (LMS) boundary, and the LMS loss per iteration
  lec02-kp08-logistic-newton  : 1-D data, logistic fit after 0, 1, 2 Newton steps
  lec02-kp09-rbf-xor          : XOR in the input space and in the space of two Gaussian basis functions
  lec02-kp09-som-grid         : an 8x8 SOM grid before and after training on a unit square
  lec02-kp10-uat-staircase    : sum of steep sigmoids approximating sin(2 pi x), error vs number of units
Reuses the helpers of lec02-kp01-figures.py; the dark SVG for slides goes to $DECK_WORK/assets/figures/.
"""
import importlib.util
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("figs_common", os.path.join(HERE, "lec02-kp01-figures.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
px, setup, save = C.px, C.setup, C.save


def clean(ax, fs):
    ax.set_facecolor("none")
    ax.tick_params(labelsize=fs)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))


def adaline_or(theme):
    C.FM = 1.25 if theme == "dark" else 1.0
    W, H = 1160, 440
    fig, t = setup(theme, W, H)
    fs = px(19)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 1, 1, 1], float)
    ax = fig.add_axes([0.07, 0.2, 0.36, 0.66])
    xs = np.linspace(-0.4, 1.4, 50)
    ax.plot(xs, 1 - xs, color=t["ref"], lw=3, ls="--", label="Perceptron (KP02)  $x_1+x_2=1$")
    ax.plot(xs, 0.5 - xs, color=t["main"], lw=3.2, label="ADALINE (LMS)  $x_1+x_2=0.5$")
    for (a, b), c in zip(X, y):
        ax.scatter([a], [b], s=190, color=t["main"] if c else t["ref"], edgecolor=t["text"], zorder=3, marker="o" if c else "s")
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-0.4, 2.0)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$", fontsize=fs)
    ax.set_ylabel(r"$x_2$", fontsize=fs)
    ax.legend(fontsize=fs * 0.82, loc="upper right", frameon=False)
    ax.set_title("OR: the LMS line sits between the classes", fontsize=fs, pad=8)
    clean(ax, fs)
    # loss per iteration
    A = np.c_[X, np.ones(4)]
    w = np.zeros(3)
    loss = [0.5 * ((y - A @ w) ** 2).sum()]
    for _ in range(12):
        w = w + 0.1 * A.T @ (y - A @ w)
        loss.append(0.5 * ((y - A @ w) ** 2).sum())
    ax2 = fig.add_axes([0.56, 0.2, 0.4, 0.66])
    ax2.plot(range(len(loss)), loss, color=t["main"], lw=3, marker="o", ms=7)
    ax2.axhline(0.125, color=t["ok"], lw=1.8, ls="--")
    ax2.text(12, 0.2, "least-squares minimum $0.125$", fontsize=fs * 0.85, color=t["ok"], ha="right")
    ax2.text(0.4, loss[0] - 0.12, "$1.5$", fontsize=fs * 0.9, color=t["text"])
    ax2.text(1.35, loss[1] + 0.05, "$0.34$", fontsize=fs * 0.9, color=t["text"])
    ax2.set_xlabel("iteration ($\\eta=0.1$)", fontsize=fs)
    ax2.set_ylabel("LMS loss $e$", fontsize=fs)
    ax2.set_title("the loss falls smoothly", fontsize=fs, pad=8)
    clean(ax2, fs)
    save(fig, "lec02-kp08-adaline-or", theme)


def logistic_newton(theme):
    C.FM = 1.35 if theme == "dark" else 1.0
    W, H = 760, 520
    fig, t = setup(theme, W, H)
    fs = px(19)
    ax = fig.add_axes([0.1, 0.16, 0.86, 0.74])
    x = np.array([0, 1, 2, 3.0])
    y = np.array([0, 1, 0, 1.0])
    g = np.linspace(-0.5, 3.5, 300)
    B = np.c_[np.ones(4), x]
    b = np.zeros(2)
    curves = []
    for k in range(3):
        curves.append((k, b.copy()))
        p = sigmoid(B @ b)
        H_ = B.T @ np.diag(p * (1 - p)) @ B
        b = b + np.linalg.solve(H_, B.T @ (y - p))
    cols = [t["muted"], t["ref"], t["main"]]
    labs = ["start $\\beta=(0,0)$", "1 step $(-1.200,\\,0.800)$", "2 steps $(-1.356,\\,0.904)$"]
    for (k, bb), c, lab in zip(curves, cols, labs):
        ax.plot(g, sigmoid(bb[0] + bb[1] * g), color=c, lw=3 if k else 2.2, ls="--" if k == 0 else "-", label=lab)
    ax.scatter(x, y, s=170, color=t["aux"], edgecolor=t["text"], zorder=4)
    ax.axhline(0.5, color=t["edge"], lw=1.4, ls=":")
    ax.text(3.45, 0.52, "threshold 0.5", fontsize=fs * 0.85, color=t["edge"], ha="right")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel(r"$x$", fontsize=fs)
    ax.set_ylabel(r"$P(y=1\mid x)$", fontsize=fs)
    ax.legend(fontsize=fs * 0.8, loc="upper left", frameon=False)
    clean(ax, fs)
    save(fig, "lec02-kp08-logistic-newton", theme)


def rbf_xor(theme):
    C.FM = 1.25 if theme == "dark" else 1.0
    W, H = 1160, 440
    fig, t = setup(theme, W, H)
    fs = px(19)
    P = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    cls = np.array([0, 1, 1, 0])
    mu = np.array([[0, 0], [1, 1]], float)
    ax = fig.add_axes([0.07, 0.2, 0.36, 0.66])
    th = np.linspace(0, 2 * np.pi, 100)
    for m in mu:
        ax.plot(m[0] + 0.7 * np.cos(th), m[1] + 0.7 * np.sin(th), color=t["edge"], lw=2, ls="--")
        ax.scatter([m[0]], [m[1]], s=90, color=t["edge"], marker="x", zorder=5, linewidths=3)
    for p_, c in zip(P, cls):
        ax.scatter([p_[0]], [p_[1]], s=190, color=t["main"] if c else t["ref"], edgecolor=t["text"], zorder=3, marker="o" if c else "s")
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$", fontsize=fs)
    ax.set_ylabel(r"$x_2$", fontsize=fs)
    ax.set_title("XOR with two Gaussian centres (×)", fontsize=fs, pad=8)
    clean(ax, fs)
    ax2 = fig.add_axes([0.56, 0.2, 0.4, 0.66])
    Phi = np.exp(-((P[:, None, :] - mu[None]) ** 2).sum(-1))
    s = np.linspace(0, 1.1, 10)
    ax2.plot(s, 0.9355 - s, color=t["ok"], lw=3, label=r"$\phi_1+\phi_2=0.936$")
    for ph, c in zip(Phi, cls):
        ax2.scatter([ph[0]], [ph[1]], s=190, color=t["main"] if c else t["ref"], edgecolor=t["text"], zorder=3, marker="o" if c else "s")
    ax2.annotate("(0,1) and (1,0)\nland on the same point", (0.368, 0.368), (0.5, 0.62), fontsize=fs * 0.82, color=t["text"],
                 arrowprops=dict(arrowstyle="->", color=t["muted"]))
    ax2.set_xlim(0, 1.15)
    ax2.set_ylim(0, 1.15)
    ax2.set_aspect("equal")
    ax2.set_xlabel(r"$\phi_1$", fontsize=fs)
    ax2.set_ylabel(r"$\phi_2$", fontsize=fs)
    ax2.legend(fontsize=fs * 0.82, loc="upper right", frameon=False)
    ax2.set_title("in basis-function space a line works", fontsize=fs, pad=8)
    clean(ax2, fs)
    save(fig, "lec02-kp09-rbf-xor", theme)


def som_grid(theme):
    C.FM = 1.25 if theme == "dark" else 1.0
    W, H = 1160, 440
    fig, t = setup(theme, W, H)
    fs = px(19)
    rng = np.random.default_rng(0)
    n = 8
    gi, gj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    w0 = 0.5 + 0.06 * rng.normal(size=(n, n, 2))
    w = w0.copy()
    T = 4000
    for tau in range(T):
        x = rng.random(2)
        d = ((w - x) ** 2).sum(-1)
        zi, zj = np.unravel_index(d.argmin(), d.shape)
        eta = 0.5 * (1 - tau / T) + 0.01
        sig = 3.0 * (1 - tau / T) + 0.4          # neighbourhood shrinks gradually
        h = np.exp(-((gi - zi) ** 2 + (gj - zj) ** 2) / (2 * sig ** 2))
        w += eta * h[..., None] * (x - w)
    for k, (ww, title) in enumerate([(w0, "start: small random weights"), (w, "after 4000 inputs")]):
        ax = fig.add_axes([0.07 + k * 0.48, 0.14, 0.4, 0.72])
        for i in range(n):
            ax.plot(ww[i, :, 0], ww[i, :, 1], color=t["ref"], lw=1.8)
            ax.plot(ww[:, i, 0], ww[:, i, 1], color=t["ref"], lw=1.8)
        ax.scatter(ww[..., 0].ravel(), ww[..., 1].ravel(), s=34, color=t["main"], zorder=3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=fs, pad=8)
        clean(ax, fs)
    save(fig, "lec02-kp09-som-grid", theme)


def uat_staircase(theme):
    C.FM = 1.25 if theme == "dark" else 1.0
    W, H = 1160, 440
    fig, t = setup(theme, W, H)
    fs = px(19)
    f = lambda x: np.sin(2 * np.pi * x)  # noqa: E731
    xs = np.linspace(0, 1, 4001)

    def approx(N):
        tk = np.linspace(0, 1, N + 1)
        vals = f((tk[:-1] + tk[1:]) / 2)
        F = vals[0] + sum((vals[i] - vals[i - 1]) * sigmoid(2000 * (xs - tk[i])) for i in range(1, N))
        return F

    ax = fig.add_axes([0.07, 0.2, 0.4, 0.66])
    ax.plot(xs, f(xs), color=t["text"], lw=2.4, label=r"target $f(x)=\sin 2\pi x$")
    ax.plot(xs, approx(5), color=t["ref"], lw=2.2, label="$N=5$ units")
    ax.plot(xs, approx(20), color=t["main"], lw=2.2, label="$N=20$ units")
    ax.set_xlabel(r"$x$", fontsize=fs)
    ax.legend(fontsize=fs * 0.8, loc="upper right", frameon=False)
    ax.set_title("sums of steep sigmoids", fontsize=fs, pad=8)
    clean(ax, fs)
    Ns = np.array([3, 5, 10, 20, 40, 80])
    errs = np.array([np.abs(f(xs) - approx(N)).max() for N in Ns])
    ax2 = fig.add_axes([0.58, 0.2, 0.38, 0.66])
    ax2.loglog(Ns, errs, color=t["main"], lw=3, marker="o", ms=8)
    for N, e in zip(Ns, errs):
        if N in (10, 20, 40):
            ax2.text(N, e * 1.18, f"{e:.3f}", fontsize=fs * 0.85, color=t["text"], ha="center")
    ax2.set_xlabel("number of units $N$", fontsize=fs)
    ax2.set_ylabel(r"$\sup|f-F|$", fontsize=fs)
    ax2.set_title("the error halves when $N$ doubles", fontsize=fs, pad=8)
    ax2.minorticks_off()
    ax2.set_xticks([3, 10, 40, 80])
    ax2.set_xticklabels(["3", "10", "40", "80"])
    clean(ax2, fs)
    save(fig, "lec02-kp10-uat-staircase", theme)


if __name__ == "__main__":
    for th in ("light", "dark"):
        adaline_or(th)
        logistic_newton(th)
        rbf_xor(th)
        som_grid(th)
        uat_staircase(th)
    print("ok")
