"""KP05 figures (overfitting, underfitting, generalization) in two skins from the same data.
    assets/figures/<id>.png (light, lecture)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp05-fits (three fits on the same 12 training points), lec04-kp05-train-test (train vs test MSE against degree).
Toy data: f(x) = 1 + 2x - 1.5x^2 (true model is quadratic), sigma = 0.3 (noise floor sigma^2 = 0.09), 12 fixed-design points x = linspace(-1, 1, 12),
noise drawn with seed 11. "Test MSE" = E[(fhat - y)^2] on fresh noisy data = mean (fhat - f)^2 + sigma^2 on a dense grid (KP04 identity), cross-checked
against a fresh noisy test set. Bias^2 and variance come from refitting on 2000 fresh noise draws with the design fixed.
Run:  python3 assets/figures/src/lec04-kp05-figures.py     (prints every number the lecture and the deck quote)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

SIGMA = 0.3
N = 12
X = np.linspace(-1, 1, N)
XG = np.linspace(-1, 1, 2001)


def f(x):
    return 1.0 + 2.0 * np.asarray(x) - 1.5 * np.asarray(x) ** 2


Y = f(X) + np.random.default_rng(11).normal(0, SIGMA, N)
rt = np.random.default_rng(77)
XT = np.sort(rt.uniform(-1, 1, 8))              # eight fresh test points shown as hollow circles
YT = f(XT) + rt.normal(0, SIGMA, 8)


def fit(d, y=Y):
    return np.polyfit(X, y, d)


def train_mse(d):
    return float(np.mean((np.polyval(fit(d), X) - Y) ** 2))


def test_mse(d):
    """Expected squared error on fresh noisy data: mean (fhat - f)^2 + sigma^2."""
    return float(np.mean((np.polyval(fit(d), XG) - f(XG)) ** 2) + SIGMA ** 2)


def fresh_test_mse(d, m=200000, seed=5):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-1, 1, m)
    ys = f(xs) + rng.normal(0, SIGMA, m)
    return float(np.mean((np.polyval(fit(d), xs) - ys) ** 2))


def bias_var(d, reps=2000):
    P = np.array([np.polyval(np.polyfit(X, f(X) + np.random.default_rng(10 + s).normal(0, SIGMA, N), d), XG) for s in range(reps)])
    return float(np.mean((P.mean(0) - f(XG)) ** 2)), float(np.mean(P.var(0)))


def fits(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, 16)
    W, H = 1100, 340
    fig = plt.figure(figsize=(W / 100, H / 100))
    spec = ((1, "underfit", c["warn"]), (2, "good fit", c["ok"]), (9, "overfit", c["ref"]))
    for k, (d, name, col) in enumerate(spec):
        ax = fig.add_axes([(60 + k * 355) / W, 84 / H, 300 / W, 200 / H])
        ax.plot(XG, f(XG), color=c["muted"], lw=1.4, ls="--", label="true $f$")
        ax.plot(XG, np.polyval(fit(d), XG), color=col, lw=2.6, label=f"degree {d}")
        ax.scatter(X, Y, s=34, color=c["main"], zorder=5, label="train")
        ax.scatter(XT, YT, s=44, facecolors="none", edgecolors=c["text"], linewidths=1.6, zorder=5, label="test")
        ax.set_ylim(-1.6, 2.2)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel("$x$")
        if k == 0:
            ax.set_ylabel("$y$")
            h, l = ax.get_legend_handles_labels()
            hh = [h[0], h[2], h[3]]
            ll = ["true $f$", "train", "test"]
        ax.set_title(f"{name}: degree {d}\ntrain {train_mse(d):.3f}   test {test_mse(d):.3f}", fontsize=12, color=c["text"])
        clean(ax)
    fig.legend(hh, ll, loc="lower center", frameon=False, fontsize=11, ncol=3, bbox_to_anchor=(0.5, 0.0))
    finish(fig, "lec04-kp05-fits", dark, c)
    return W, H


def train_test(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    ds = np.arange(1, 11)
    tr = [train_mse(d) for d in ds]
    te = [test_mse(d) for d in ds]
    ax.semilogy(ds, tr, "-o", color=c["main"], lw=2.4, ms=6, label="train MSE")
    ax.semilogy(ds, te, "-o", color=c["ref"], lw=2.4, ms=6, label="test MSE")
    ax.axhline(SIGMA ** 2, color=c["ok"], ls="--", lw=1.4)
    ax.annotate(r"noise floor $\sigma^2=0.09$", (6.4, SIGMA ** 2), xytext=(0, -18), textcoords="offset points", color=c["ok"], fontsize=13)
    best = int(ds[int(np.argmin(te))])
    ax.annotate(f"best test: degree {best}", (best, min(te)), xytext=(18, 38), textcoords="offset points", color=c["text"], fontsize=13,
                arrowprops=dict(arrowstyle="->", color=c["muted"]))
    ax.set_xlabel("polynomial degree (complexity)")
    ax.set_ylabel("MSE (log scale)")
    ax.set_xticks(ds)
    ax.legend(loc="lower left", frameon=False, fontsize=13)
    clean(ax)
    finish(fig, "lec04-kp05-train-test", dark, c)
    return W, H


def report():
    print("true f(x)=1+2x-1.5x^2, sigma=0.3 (sigma^2=0.09), n=12 fixed design x=linspace(-1,1,12), noise seed 11")
    print("noise realisation: mean eps^2 on the 12 points =", round(float(np.mean((Y - f(X)) ** 2)), 4))
    print("degree  train   test(exact)  test(fresh set)   bias^2   var    bias^2+var+sigma^2")
    for d in list(range(1, 11)):
        b2, v = bias_var(d) if d in (1, 2, 3, 9) else (float("nan"), float("nan"))
        print(f"{d:5d}  {train_mse(d):.4f}  {test_mse(d):.4f}     {fresh_test_mse(d):.4f}         {b2:.4f}  {v:.4f}  {b2 + v + SIGMA ** 2:.4f}")
    print("expected train MSE of a p-parameter fit = sigma^2 (n-p)/n :", {d: round(SIGMA ** 2 * (N - d - 1) / N, 4) for d in (1, 2, 9)})
    print("test points (hollow):", np.round(XT, 3), np.round(YT, 3))


if __name__ == "__main__":
    for dk in (False, True):
        fits(dk)
        train_test(dk)
    report()
