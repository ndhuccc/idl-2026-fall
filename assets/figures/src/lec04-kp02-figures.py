"""KP02 figures (bias, variance and MSE) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp02-darts, lec04-kp02-triangle, lec04-kp02-shrink.   Prints every number the lecture and the deck quote.
Run:  python3 assets/figures/src/lec04-kp02-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

# ---- four-dart example (source: true value 10, estimates 7, 9, 8, 10)
X_TRUE = 10.0
EST = np.array([7.0, 9.0, 8.0, 10.0])
# ---- shrinkage example: estimate mu = 3 from n = 4 draws with sigma = 4; B = 0.8 * sample mean
MU, SIG, N, SHR = 3.0, 4.0, 4, 0.8


def four_darts():
    """Population-style numbers of the four-dart example (1/n variance, as in the source)."""
    m = EST.mean()
    bias = m - X_TRUE
    var = float(np.mean((EST - m) ** 2))
    mse = float(np.mean((EST - X_TRUE) ** 2))
    return m, bias, var, mse


# ------------------------------------------------------------------ figure 1: four dartboards
BOARDS = (("low bias, low variance", (0.1, 0.0), 0.30),
          ("low bias, high variance", (0.0, 0.0), 1.00),
          ("high bias, low variance", (1.5, 0.8), 0.30),
          ("high bias, high variance", (1.5, 0.8), 1.00))


def board_stats(seed=7, n=40):
    rng = np.random.default_rng(seed)
    out = []
    for name, b, sd in BOARDS:
        p = np.array(b) + rng.normal(0, sd, (n, 2))
        m = p.mean(0)
        var = float(np.mean(np.sum((p - m) ** 2, axis=1)))
        bias2 = float(np.sum(m ** 2))
        mse = float(np.mean(np.sum(p ** 2, axis=1)))
        out.append((name, p, var, bias2, mse))
    return out


def darts(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, 16)
    W, H = 1100, 330
    fig = plt.figure(figsize=(W / 100, H / 100))
    th = np.linspace(0, 2 * np.pi, 200)
    for i, (name, p, var, bias2, mse) in enumerate(board_stats()):
        ax = fig.add_axes([(14 + i * 272) / W, 14 / H, 258 / W, 262 / H])
        for r in (1, 2, 3):
            ax.plot(r * np.cos(th), r * np.sin(th), color=c["muted"], lw=1.0, alpha=0.7)
        ax.plot(0, 0, "+", color=c["ok"], ms=12, mew=2.2)
        ax.plot(p[:, 0], p[:, 1], "o", color=c["main"], ms=4.5, alpha=0.9)
        m = p.mean(0)
        ax.plot(m[0], m[1], "x", color=c["ref"], ms=11, mew=2.6)
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(name, fontsize=13, color=c["text"], pad=3)
        ax.text(0, -3.4, rf"Var={var:.2f}  Bias$^2$={bias2:.2f}  MSE={mse:.2f}", ha="center", va="top", fontsize=11.5, color=c["muted"])
    finish(fig, "lec04-kp02-darts", dark, c)
    return W, H


# ------------------------------------------------------------------ figure 2: Pythagoras
def triangle(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 560, 400
    fig, ax = axes(W, H, left=30, bottom=30, right=20, top=20)
    _, bias, var, mse = four_darts()
    a, b, h = abs(bias), np.sqrt(var), np.sqrt(mse)       # legs |Bias|, sqrt(Var); hypotenuse sqrt(MSE)
    ax.fill([0, a, a], [0, 0, b], color=c["ref"], alpha=0.14)
    ax.plot([0, a], [0, 0], color=c["ref"], lw=3)
    ax.plot([a, a], [0, b], color=c["vec"], lw=3)
    ax.plot([0, a], [0, b], color=c["warn"], lw=3, ls="--")
    s = 0.12
    ax.plot([a - s, a - s, a], [0, s, s], color=c["muted"], lw=1.6)
    ax.text(a / 2, -0.12, rf"$|\mathrm{{Bias}}|={a:.2f}$", ha="center", va="top", color=c["ref"], fontsize=15)
    ax.text(a + 0.06, b / 2, rf"$\sqrt{{\mathrm{{Var}}}}={b:.3f}$", ha="left", va="center", color=c["vec"], fontsize=15)
    ax.text(a / 2 - 0.12, b / 2 + 0.2, rf"$\sqrt{{\mathrm{{MSE}}}}={h:.3f}$", ha="right", va="bottom", color=c["warn"], fontsize=15, rotation=0)
    ax.text(a / 2, b + 0.5, r"$(\sqrt{\mathrm{MSE}})^2=(\sqrt{\mathrm{Var}})^2+|\mathrm{Bias}|^2$", ha="center", color=c["text"], fontsize=15)
    ax.set_xlim(-0.65, 2.25)
    ax.set_ylim(-0.4, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")
    finish(fig, "lec04-kp02-triangle", dark, c)
    return W, H


# ------------------------------------------------------------------ figure 3: unbiased vs shrunk estimator
def shrink_stats(seed=3, runs=200000):
    rng = np.random.default_rng(seed)
    m = rng.normal(MU, SIG, (runs, N)).mean(axis=1)
    rows = {}
    for key, k in (("A", 1.0), ("B", SHR)):
        e = k * m
        rows[key] = (e, float(e.mean() - MU), float(e.var()), float(np.mean((e - MU) ** 2)))
    return rows


def shrink(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    rows = shrink_stats()
    bins = np.linspace(-6, 12, 80)
    for key, col, lab in (("A", c["ref"], "A"), ("B", c["main"], "B")):
        e, bias, var, mse = rows[key]
        ax.hist(e, bins=bins, density=True, color=col, alpha=0.35, label=lab)
        txt = (r"A: sample mean" "\n" rf"unbiased, MSE={mse:.2f}") if key == "A" else (r"B: $0.8\times$ sample mean" "\n" rf"biased, MSE={mse:.2f}")
        ax.text(*((5.6, 0.17) if key == "A" else (-5.7, 0.27)), txt, color=col, fontsize=14, va="center")
    ax.axvline(MU, color=c["ok"], lw=2, ls="--")
    ax.annotate(r"true $\mu=3$", (MU, 0.345), xytext=(5, 0), textcoords="offset points", color=c["ok"], fontsize=14)
    ax.set_xlabel(r"estimate $\widehat{\mu}$")
    ax.set_ylabel("density")
    ax.set_xlim(-6, 12)
    ax.set_ylim(0, 0.38)
    clean(ax)
    finish(fig, "lec04-kp02-shrink", dark, c)
    return W, H


def report():
    m, bias, var, mse = four_darts()
    print(f"four darts: X={X_TRUE}, estimates {EST.tolist()}, mean {m}, bias {bias}, bias^2 {bias ** 2}, var {var}, mse {mse}, var+bias^2 {var + bias ** 2}")
    print(f"triangle legs: |bias| {abs(bias):.3f}, sqrt(var) {np.sqrt(var):.3f}, hypotenuse sqrt(mse) {np.sqrt(mse):.3f}; check {np.sqrt(var) ** 2 + bias ** 2:.3f}")
    for name, p, v, b2, ms in board_stats():
        print(f"board {name}: var {v:.3f}, bias^2 {b2:.3f}, mse {ms:.3f}, var+bias^2 {v + b2:.3f}")
    print(f"shrink example: mu={MU}, sigma={SIG}, n={N}, shrink {SHR}")
    print(f"  A theory: bias 0, var sigma^2/n = {SIG ** 2 / N}, mse {SIG ** 2 / N}")
    print(f"  B theory: bias {SHR * MU - MU:.2f}, bias^2 {(SHR * MU - MU) ** 2:.2f}, var {SHR ** 2 * SIG ** 2 / N:.2f}, mse {SHR ** 2 * SIG ** 2 / N + (SHR * MU - MU) ** 2:.2f}")
    for key, (e, bias_, var_, mse_) in shrink_stats().items():
        print(f"  {key} simulated (200000 runs): bias {bias_:.3f}, var {var_:.3f}, mse {mse_:.3f}")
    # break-even: shrink factor k minimising MSE = k^2 s2/n + (k-1)^2 mu^2
    s2n = SIG ** 2 / N
    kopt = MU ** 2 / (MU ** 2 + s2n)
    print(f"  best shrink factor k* = mu^2/(mu^2+sigma^2/n) = {kopt:.4f}, mse at k* = {kopt ** 2 * s2n + (kopt - 1) ** 2 * MU ** 2:.3f}")


if __name__ == "__main__":
    sizes = {}
    for fn in (darts, triangle, shrink):
        finish_sizes = [fn(True), fn(False)]
        sizes[fn.__name__] = finish_sizes[0]
    print("sizes", sizes)
    report()
