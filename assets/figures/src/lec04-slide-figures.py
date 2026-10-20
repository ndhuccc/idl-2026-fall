"""Slide-size redraws of the chapter-4 figures (dark skin only), drawn 1:1 with the image node.

Why: CLARITY.md C5 asks for figures drawn at display size, with every label >= 16 px on the slide
(the lecture figures are drawn larger and shrunk on the page, so their labels end up around 10 px).
The data and seeds are taken from the lecture scripts (assets/figures/src/lec04-kpNN-figures.py), so the numbers are identical.

    assets/figures/<name>-slide.svg   dark skin, used only by slides/lec04-kpNN-*.jbd.json
Run:  python3 assets/figures/src/lec04-slide-figures.py [kp01 kp02 ...]   (no argument: all)
Prints the display size (W x H px) of every figure: use it as the image node size.
"""
import base64
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from lec04_common import DARK, OUT, axes, clean, skin  # noqa: E402
from jbd_figure import svg_data_uri  # noqa: E402

c = DARK
MIN_PX = 16


def load(kp):
    """Import the lecture script of one KP (file names contain a dash) to reuse its data functions."""
    path = [f for f in os.listdir(HERE) if f.startswith(f"lec04-{kp}-") and f.endswith("-figures.py")][0]
    spec = importlib.util.spec_from_file_location(f"lec04_{kp}", os.path.join(HERE, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save(fig, name):
    with open(os.path.join(OUT, name + "-slide.svg"), "wb") as fh:
        fh.write(base64.b64decode(svg_data_uri(fig).split(",", 1)[1]))
    plt.close(fig)


def start(W, H, base=16):
    skin(c, True, base)
    return plt.figure(figsize=(W / 100, H / 100))


# ======================================================================= KP01
def kp01_split_roles():
    W, H = 640, 250
    fig = start(W, H)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    x0, x1, y0, h = 20, 620, 110, 54
    scale = (x1 - x0) / 100.0
    x = x0
    for name, n, col, role in (("T", 60, c["ref"], "fit the weights"), ("V", 20, c["vec"], "tune and\nstop early"), ("R", 20, c["ok"], "report\nonce")):
        w = n * scale
        ax.add_patch(FancyBboxPatch((x + 2, y0), w - 4, h, boxstyle="round,pad=0,rounding_size=8", fc=col, ec="none", alpha=0.28))
        ax.add_patch(FancyBboxPatch((x + 2, y0), w - 4, h, boxstyle="round,pad=0,rounding_size=8", fc="none", ec=col, lw=2.2))
        ax.text(x + w / 2, y0 + h / 2, rf"$\mathcal{{{name}}}$: {n}", ha="center", va="center", fontsize=18, color=c["text"])
        ax.annotate("", xy=(x + w / 2, y0 - 6), xytext=(x + w / 2, y0 - 30), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
        ax.text(x + w / 2, y0 - 62, role, ha="center", va="center", fontsize=17, color=c["text"], linespacing=1.15)
        x += w
    ax.text((x0 + x1) / 2, y0 + h + 40, r"$\mathcal{D}=\mathcal{T}\cup\mathcal{V}\cup\mathcal{R}$:  100 examples, no overlap", ha="center", va="center", fontsize=18, color=c["text"])
    save(fig, "lec04-kp01-split-roles")
    return W, H


def kp01_cheating_scores():
    m = load("kp01")
    n = m.cheating_numbers()
    W, H = 600, 400
    skin(c, True, 16)
    fig, ax = axes(W, H, left=70, bottom=140, right=16, top=20)
    labels = ["1-NN\ntrain pts", "1-NN\nfresh pts", "best of\n200, test", "same one\nfresh pts"]
    vals = [n["acc_overlap"], n["acc_fresh"], n["best_test"], n["best_fresh"]]
    cols = [c["warn"], c["ok"], c["warn"], c["ok"]]
    xs = [0, 1, 2.45, 3.45]
    ax.bar(xs, [v * 100 for v in vals], color=cols, width=0.7, alpha=0.9)
    for xx, v in zip(xs, vals):
        ax.text(xx, v * 100 + 2, f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=17, color=c["text"])
    ax.axhline(50, color=c["muted"], ls="--", lw=1.3)
    ax.text(4.0, 114, "-- chance 50%", ha="right", va="center", fontsize=16, color=c["muted"])
    ax.text(0.5, -49, "cheat #1: train = test", ha="center", va="top", fontsize=16, color=c["warn"], clip_on=False)
    ax.text(2.95, -49, "cheat #2: pick by test", ha="center", va="top", fontsize=16, color=c["warn"], clip_on=False)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=16)
    ax.tick_params(axis="x", pad=4)
    ax.set_ylabel("accuracy (%)", fontsize=17)
    ax.set_ylim(0, 122); ax.set_xlim(-0.6, 4.05)
    clean(ax); ax.grid(False)
    save(fig, "lec04-kp01-cheating-scores")
    return W, H


# ======================================================================= KP02
def kp02_boards():
    k = load("kp02")
    W, H = 600, 440
    fig = start(W, H, 18)
    lg = fig.add_axes([0, (H - 34) / H, 1, 34 / H]); lg.axis("off"); lg.set_xlim(0, W); lg.set_ylim(0, 1)
    for x, mk, col, ms, lab in ((24, "+", c["ok"], 13, "target X"), (170, "x", c["ref"], 11, "mean of estimates"), (420, "o", c["main"], 7, "one estimate")):
        lg.plot(x + 8, .5, mk, color=col, ms=ms, mew=2.4)
        lg.text(x + 24, .5, lab, va="center", fontsize=16, color=c["text"])
    st = {n: (p, v, b, m) for n, p, v, b, m in k.board_stats()}
    cells = {(0, 0): "low bias, low variance", (0, 1): "high bias, low variance", (1, 0): "low bias, high variance", (1, 1): "high bias, high variance"}
    x0, cw, chh, bs = 34, 283, 186, 150
    th = np.linspace(0, 2 * np.pi, 200)
    for (r, kk), name in cells.items():
        p, v, b2, m = st[name]
        left = x0 + kk * cw; top = H - 68 - r * chh
        ax = fig.add_axes([left / W, (top - bs - 8) / H, bs / W, bs / H])
        for rr in (1, 2, 3):
            ax.plot(rr * np.cos(th), rr * np.sin(th), color=c["muted"], lw=1.0, alpha=.7)
        ax.plot(0, 0, "+", color=c["ok"], ms=12, mew=2.2)
        ax.plot(p[:, 0], p[:, 1], "o", color=c["main"], ms=4, alpha=.9)
        mm = p.mean(0); ax.plot(mm[0], mm[1], "x", color=c["ref"], ms=11, mew=2.6)
        ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6); ax.set_aspect("equal"); ax.axis("off")
        tx = left + bs + 8
        for j, (lab, val) in enumerate((("Var", v), ("Bias$^2$", b2), ("MSE", m))):
            fig.text(tx / W, (top - 38 - j * 30) / H, f"{lab} {val:.2f}", fontsize=17, color=c["text"] if lab == "MSE" else c["muted"], va="center",
                     fontweight="bold" if lab == "MSE" else "normal")
    for kk, lab in enumerate(("low bias", "high bias")):
        fig.text((x0 + kk * cw + 150) / W, (H - 51) / H, lab, fontsize=18, color=c["text"], ha="center", va="center", fontweight="bold")
    for r, lab in enumerate(("low var.", "high var.")):
        fig.text(22 / W, (H - 68 - r * chh - 80) / H, lab, fontsize=18, color=c["text"], ha="center", va="center", rotation=90, fontweight="bold")
    save(fig, "lec04-kp02-boards")
    return W, H


def kp02_numline():
    k = load("kp02")
    W, H = 1160, 160
    fig = start(W, H, 18)
    ax = fig.add_axes([40 / W, 28 / H, (W - 80) / W, 120 / H])
    est = k.EST; X = k.X_TRUE; m, bias, var, mse = k.four_darts()
    ax.set_xlim(6.4, 10.6); ax.set_ylim(-1, 3.2)
    ax.axhline(0, color=c["muted"], lw=1.2)
    for t in range(7, 11):
        ax.text(t, -0.35, str(t), ha="center", va="top", fontsize=18, color=c["muted"]); ax.plot([t, t], [-0.1, 0.1], color=c["muted"], lw=1.2)
    ax.plot(est, np.zeros(4), "o", color=c["main"], ms=11)
    ax.plot([X], [0], "+", color=c["ok"], ms=20, mew=3)
    ax.plot([m], [0], "x", color=c["ref"], ms=16, mew=3.2)
    ax.annotate("", xy=(m, 1.5), xytext=(X, 1.5), arrowprops=dict(arrowstyle="<|-|>", color=c["warn"], lw=2.4, mutation_scale=18))
    ax.text((m + X) / 2, 1.65, f"Bias = {bias:.1f}", ha="center", va="bottom", fontsize=18, color=c["warn"])
    ax.text(X, 0.28, "X = 10", ha="center", va="bottom", fontsize=18, color=c["ok"])
    ax.text(m, 0.28, f"mean {m:.1f}", ha="center", va="bottom", fontsize=18, color=c["ref"])
    ax.text(6.45, 1.65, "o = estimates 7, 9, 8, 10", ha="left", va="bottom", fontsize=18, color=c["main"])
    ax.axis("off")
    save(fig, "lec04-kp02-numline")
    return W, H


def kp02_triangle():
    k = load("kp02")
    W, H = 560, 330
    fig = start(W, H, 18)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    _, bias, var, mse = k.four_darts()
    a, b, h = abs(bias), np.sqrt(var), np.sqrt(mse)
    sc = 150  # px per unit
    ox, oy = 125, 80
    P = lambda x, y: (ox + x * sc, oy + y * sc)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.fill(*zip(P(0, 0), P(a, 0), P(a, b)), color=c["ref"], alpha=0.14)
    ax.plot(*zip(P(0, 0), P(a, 0)), color=c["ref"], lw=3)
    ax.plot(*zip(P(a, 0), P(a, b)), color=c["vec"], lw=3)
    ax.plot(*zip(P(0, 0), P(a, b)), color=c["warn"], lw=3, ls="--")
    s = 0.1
    ax.plot(*zip(P(a - s, 0), P(a - s, s), P(a, s)), color=c["muted"], lw=1.6)
    ax.text(*P(a / 2, -0.12), r"$|\mathrm{Bias}|=1.5$", ha="center", va="top", color=c["ref"], fontsize=18)
    ax.text(*P(a + 0.06, b / 2), r"$\sqrt{\mathrm{Var}}=1.118$", ha="left", va="center", color=c["vec"], fontsize=18)
    ax.text(*P(a / 2 - 0.08, b / 2 + 0.12), r"$\sqrt{\mathrm{MSE}}=1.871$", ha="right", va="bottom", color=c["warn"], fontsize=18)
    save(fig, "lec04-kp02-triangle")
    return W, H


def kp02_shrink():
    k = load("kp02")
    W, H = 640, 380
    fig = start(W, H, 18)
    ax = fig.add_axes([88 / W, 72 / H, (W - 108) / W, (H - 100) / H])
    rows = k.shrink_stats()
    bins = np.linspace(-6, 12, 80)
    for key, col in (("A", c["ref"]), ("B", c["main"])):
        e, bias, var, mse = rows[key]
        ax.hist(e, bins=bins, density=True, color=col, alpha=0.35)
    ax.text(12, 0.215, "A: sample mean\nunbiased, MSE=4", color=c["ref"], fontsize=17, va="center", ha="right")
    ax.text(-5.8, 0.30, "B: 0.8 x sample mean\nbiased, MSE=2.92", color=c["main"], fontsize=17, va="center", ha="left")
    ax.axvline(k.MU, color=c["ok"], lw=2, ls="--")
    ax.text(k.MU + 0.2, 0.355, r"true $\mu=3$", color=c["ok"], fontsize=17, va="center")
    ax.set_xlabel(r"estimate $\widehat{\mu}$", fontsize=18)
    ax.set_ylabel("density", fontsize=18)
    ax.set_xlim(-6, 12); ax.set_ylim(0, 0.38)
    clean(ax)
    save(fig, "lec04-kp02-shrink")
    return W, H


# ======================================================================= KP03
def kp03_scatter():
    k = load("kp03")
    W, H = 1160, 330
    fig = start(W, H, 18)
    titles = {3.0: r"Cov$(X,Y)=+3$", 0.0: r"Cov$(X,Y)=0$", -3.0: r"Cov$(X,Y)=-3$"}
    for i, d in enumerate(k.panels()):
        ax = fig.add_axes([(76 + i * 388) / W, 68 / H, 300 / W, 176 / H])
        ax.plot(d["x"][:k.NPLOT], d["y"][:k.NPLOT], "o", color=c["main"], ms=3.5, alpha=0.55)
        ax.axhline(0, color=c["muted"], lw=0.8, alpha=0.6)
        ax.axvline(0, color=c["muted"], lw=0.8, alpha=0.6)
        ax.set_xlim(-8, 8); ax.set_ylim(-11, 11)
        ax.set_xlabel("$X$", labelpad=1, fontsize=18)
        if i == 0:
            ax.set_ylabel("$Y$", labelpad=1, fontsize=18)
        ax.set_title(titles[d["cov"]], fontsize=19, color=c["text"], pad=36)
        ax.text(0.5, 1.04, rf"Var$(X+Y)={d['theory']:.0f}$", transform=ax.transAxes, ha="center", va="bottom", fontsize=18, color=c["ok"])
        clean(ax)
    save(fig, "lec04-kp03-scatter")
    return W, H


def kp03_parabola():
    W, H = 520, 380
    fig = start(W, H, 18)
    ax = fig.add_axes([84 / W, 70 / H, (W - 108) / W, (H - 100) / H])
    xs = np.arange(-2, 3); ys = xs ** 2
    xx = np.linspace(-2.4, 2.4, 100)
    ax.plot(xx, xx ** 2, color=c["muted"], lw=1.2, ls=":")
    ax.plot(xs, ys, "o", color=c["main"], ms=11)
    ax.axvline(0, color=c["muted"], lw=0.8, alpha=0.6)
    ax.set_xlabel("$X$", fontsize=18); ax.set_ylabel(r"$Y=X^2$", fontsize=18)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-0.6, 7.2)
    ax.text(0, 6.4, r"5 points, Cov$(X,Y)=0$", ha="center", color=c["text"], fontsize=17)
    clean(ax)
    save(fig, "lec04-kp03-parabola")
    return W, H


def kp03_bars():
    k = load("kp03")
    W, H = 560, 300
    fig = start(W, H, 18)
    ax = fig.add_axes([76 / W, 60 / H, (W - 90) / W, (H - 84) / H])
    ds = k.panels()
    order = [ds[2], ds[1], ds[0]]
    xs = np.arange(3)
    ax.bar(xs - 0.2, [d["theory"] for d in order], 0.38, color=c["ref"], alpha=0.85, label="theory")
    ax.bar(xs + 0.2, [d["var_sum"] for d in order], 0.38, color=c["main"], alpha=0.85, label="sample")
    for i, d in enumerate(order):
        ax.text(i - 0.2, d["theory"] + 0.4, f"{d['theory']:.0f}", ha="center", fontsize=17, color=c["text"])
        ax.text(i + 0.2, d["var_sum"] + 0.4, f"{d['var_sum']:.1f}", ha="center", fontsize=17, color=c["text"])
    ax.axhline(13, color=c["ok"], lw=2, ls="--")
    ax.text(-0.55, 14.0, "4+9=13", color=c["ok"], fontsize=17, ha="left")
    ax.set_xticks(xs); ax.set_xticklabels([r"Cov$=-3$", r"Cov$=0$", r"Cov$=+3$"], fontsize=17)
    ax.set_ylabel(r"Var$(X+Y)$", fontsize=18)
    ax.set_ylim(0, 27); ax.set_xlim(-0.6, 2.6)
    ax.legend(loc="upper left", frameon=False, fontsize=17)
    clean(ax)
    save(fig, "lec04-kp03-bars")
    return W, H


# ======================================================================= KP04
def kp04_model():
    k = load("kp04")
    W, H = 600, 400
    fig = start(W, H, 18)
    ax = fig.add_axes([74 / W, 80 / H, (W - 92) / W, (H - 100) / H])
    x, y = k.make(k.N_TRAIN, k.SEED)
    xx = np.linspace(-1, 1, 200)
    coef = np.polyfit(x, y, 2)
    ax.fill_between(xx, k.f(xx) - k.SIGMA, k.f(xx) + k.SIGMA, color=c["ref"], alpha=0.16, lw=0)
    ax.plot(xx, k.f(xx), color=c["ref"], lw=2.6, label=r"true $f$, band $\pm\sigma$")
    ax.plot(x, y, "o", color=c["main"], ms=6, alpha=0.85, label=r"observed $y=f+\varepsilon$")
    ax.plot(xx, np.polyval(coef, xx), color=c["ok"], lw=2.2, ls="--", label=r"fitted $\hat f$")
    ax.set_xlabel("$x$", fontsize=18); ax.set_ylabel("$y$", fontsize=18)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-4.2, 2.4)
    ax.legend(loc="lower center", frameon=False, fontsize=17)
    clean(ax)
    save(fig, "lec04-kp04-model")
    return W, H


def kp04_hist():
    k = load("kp04")
    W, H = 600, 380
    fig = start(W, H, 18)
    ax = fig.add_axes([84 / W, 74 / H, (W - 100) / W, (H - 94) / H])
    rng = np.random.default_rng(8)
    e = rng.normal(0, k.SIGMA, 2000)
    ax.hist(e, bins=36, density=True, color=c["main"], alpha=0.5)
    t = np.linspace(-1.1, 1.1, 300)
    ax.plot(t, np.exp(-t ** 2 / (2 * k.SIGMA ** 2)) / (k.SIGMA * np.sqrt(2 * np.pi)), color=c["ref"], lw=2.4)
    ax.axvline(0, color=c["muted"], lw=0.8, alpha=0.7)
    ax.set_xlabel(r"noise $\varepsilon$", fontsize=18); ax.set_ylabel("density", fontsize=18)
    ax.set_xlim(-1.1, 1.1)
    ax.text(0.04, 0.95, rf"mean $={e.mean():+.3f}$", transform=ax.transAxes, va="top", fontsize=17, color=c["text"])
    ax.text(0.04, 0.84, rf"$\mathbb{{E}}(\varepsilon^2)\approx{np.mean(e ** 2):.3f}$", transform=ax.transAxes, va="top", fontsize=17, color=c["text"])
    ax.text(0.04, 0.73, rf"$\sigma^2={k.SIGMA ** 2:.2f}$", transform=ax.transAxes, va="top", fontsize=17, color=c["ref"])
    clean(ax)
    save(fig, "lec04-kp04-hist")
    return W, H


def kp04_thermo():
    k = load("kp04")
    W, H = 600, 380
    fig = start(W, H, 18)
    ax = fig.add_axes([92 / W, 74 / H, (W - 108) / W, (H - 94) / H])
    r = k.readings()
    sq = (r - 37.0) ** 2
    n = np.arange(1, len(sq) + 1)
    run = np.cumsum(sq) / n
    ax.semilogx(n, run, color=c["main"], lw=2.2)
    ax.axhline(0.04, color=c["ok"], lw=2, ls="--")
    ax.plot([3], [run[2]], "o", color=c["warn"], ms=9)
    ax.text(3.6, 0.11, "3 readings: 0.0167", color=c["warn"], fontsize=17, va="center")
    ax.text(9000, 0.047, r"$\sigma^2=0.04$", color=c["ok"], fontsize=17, va="bottom", ha="right")
    ax.set_xlabel("number of readings", fontsize=18); ax.set_ylabel(r"mean of $(y-37.0)^2$", fontsize=18)
    ax.set_xlim(1, 10000); ax.set_ylim(0, 0.2)
    clean(ax)
    save(fig, "lec04-kp04-thermo")
    return W, H


# ======================================================================= KP05
def kp05_bar():
    W, H = 600, 260
    fig = start(W, H, 18)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    x0, sc, y0, h = 20, 560 / 300.0, 70, 70
    wb, wr = 100 * sc, 200 * sc
    ax.add_patch(FancyBboxPatch((x0, y0), wb - 3, h, boxstyle="round,pad=0,rounding_size=8", fc=c["ref"], ec=c["ref"], alpha=0.35, lw=2))
    ax.add_patch(FancyBboxPatch((x0 + wb + 3, y0), wr - 3, h, boxstyle="round,pad=0,rounding_size=8", fc=c["warn"], ec=c["warn"], alpha=0.35, lw=2))
    ax.text(x0 + wb / 2, y0 + h / 2, "Err\n100", ha="center", va="center", fontsize=18, color=c["text"], linespacing=1.1)
    ax.text(x0 + wb + wr / 2, y0 + h / 2, "noise floor  200", ha="center", va="center", fontsize=18, color=c["text"])
    ax.annotate("", xy=(x0 + wb + wr, y0 + h + 28), xytext=(x0, y0 + h + 28), arrowprops=dict(arrowstyle="<|-|>", color=c["text"], lw=1.8))
    ax.text(x0 + (wb + wr) / 2, y0 + h + 40, "err = 300  (what we can compute)", ha="center", va="bottom", fontsize=18, color=c["text"])
    ax.text(x0, y0 - 14, "true error", ha="left", va="top", fontsize=17, color=c["ref"])
    ax.text(x0 + wb + wr, y0 - 14, r"$n_{te}\sigma^2=50\times4$", ha="right", va="top", fontsize=17, color=c["warn"])
    save(fig, "lec04-kp05-bar")
    return W, H


def kp05_mc():
    k = load("kp05")
    W, H = 600, 340
    fig = start(W, H, 18)
    ax = fig.add_axes([84 / W, 72 / H, (W - 102) / W, (H - 92) / H])
    r = k.monte_carlo()
    lhs, rhs = r[:, 0], r[:, 1] + k.SIG2
    ax.scatter(rhs, lhs, s=16, color=c["main"], alpha=0.75, label="300 fitted lines")
    lo, hi = min(rhs.min(), lhs.min()) - 0.01, max(rhs.max(), lhs.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], color=c["ok"], lw=1.8, ls="--", label=r"$y=x$")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"mean$(\hat f-f)^2+\sigma^2$", fontsize=18)
    ax.set_ylabel(r"mean$(\hat f-y)^2$", fontsize=18)
    ax.legend(loc="upper left", frameon=False, fontsize=17)
    clean(ax)
    save(fig, "lec04-kp05-mc")
    return W, H


FIGS = {"kp01": (kp01_split_roles, kp01_cheating_scores),
        "kp02": (kp02_boards, kp02_numline, kp02_triangle, kp02_shrink),
        "kp03": (kp03_scatter, kp03_parabola, kp03_bars),
        "kp04": (kp04_model, kp04_hist, kp04_thermo),
        "kp05": (kp05_bar, kp05_mc)}

if __name__ == "__main__":
    want = sys.argv[1:] or list(FIGS)
    for kp in want:
        for fn in FIGS[kp]:
            print(fn.__name__, fn())
