"""KP04 figures (true model and noise floor) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp04-model, lec04-kp04-hist, lec04-kp04-thermo.   Prints every number the lecture and the deck quote.
Run:  python3 assets/figures/src/lec04-kp04-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

SIGMA = 0.3                                           # noise standard deviation of the toy regression data
N_TRAIN, SEED = 30, 15                                # training points of the model figure


def f(x):
    """True model of the chapter-4 toy data: f(x) = 1 + 2x - 1.5 x^2."""
    return 1.0 + 2.0 * np.asarray(x) - 1.5 * np.asarray(x) ** 2


def make(n, seed, sigma=SIGMA):
    """n points, x ~ U(-1, 1) sorted, y = f(x) + N(0, sigma^2)."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1, 1, n))
    return x, f(x) + rng.normal(0, sigma, n)


def readings(n_total=10000, seed=21):
    """Thermometer: f = 37.0, sigma = 0.2. The first three readings are those of the worked example, the rest are simulated."""
    rng = np.random.default_rng(seed)
    rest = 37.0 + rng.normal(0, 0.2, n_total - 3)
    return np.concatenate([[36.9, 37.2, 37.0], rest])


def model(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    x, y = make(N_TRAIN, SEED)
    xx = np.linspace(-1, 1, 200)
    coef = np.polyfit(x, y, 2)
    ax.fill_between(xx, f(xx) - SIGMA, f(xx) + SIGMA, color=c["ref"], alpha=0.16, lw=0)
    ax.plot(xx, f(xx), color=c["ref"], lw=2.6, label=r"true $f$ with band $\pm\sigma$")
    ax.plot(x, y, "o", color=c["main"], ms=5.5, alpha=0.85, label=r"observed $y=f+\varepsilon$")
    ax.plot(xx, np.polyval(coef, xx), color=c["ok"], lw=2.2, ls="--", label=r"fitted $\hat f$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-3.2, 2.4)
    ax.legend(loc="lower center", frameon=False, fontsize=12, ncol=1)
    clean(ax)
    finish(fig, "lec04-kp04-model", dark, c)
    return W, H


def hist(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 600, 380
    fig, ax = axes(W, H)
    rng = np.random.default_rng(8)
    e = rng.normal(0, SIGMA, 2000)
    ax.hist(e, bins=36, density=True, color=c["main"], alpha=0.5)
    t = np.linspace(-1.1, 1.1, 300)
    ax.plot(t, np.exp(-t ** 2 / (2 * SIGMA ** 2)) / (SIGMA * np.sqrt(2 * np.pi)), color=c["ref"], lw=2.4)
    ax.axvline(0, color=c["muted"], lw=0.8, alpha=0.7)
    ax.set_xlabel(r"noise $\varepsilon$")
    ax.set_ylabel("density")
    ax.set_xlim(-1.1, 1.1)
    ax.text(0.04, 0.95, rf"mean $=${e.mean():+.3f}", transform=ax.transAxes, va="top", fontsize=13, color=c["text"])
    ax.text(0.04, 0.86, rf"$\mathbb{{E}}(\varepsilon^2)\approx${np.mean(e ** 2):.3f}", transform=ax.transAxes, va="top", fontsize=13, color=c["text"])
    ax.text(0.04, 0.77, rf"$\sigma^2=${SIGMA ** 2:.2f}", transform=ax.transAxes, va="top", fontsize=13, color=c["ref"])
    clean(ax)
    finish(fig, "lec04-kp04-hist", dark, c)
    return W, H


def thermo(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 600, 380
    fig, ax = axes(W, H)
    r = readings()
    sq = (r - 37.0) ** 2
    run = np.cumsum(sq) / np.arange(1, len(sq) + 1)
    n = np.arange(1, len(sq) + 1)
    ax.semilogx(n, run, color=c["main"], lw=2.2)
    ax.axhline(0.04, color=c["ok"], lw=2, ls="--")
    ax.plot([3], [run[2]], "o", color=c["warn"], ms=9)
    ax.annotate(r"3 readings: 0.0167", (3, run[2]), xytext=(1.3, 0.11), textcoords="data", color=c["warn"], fontsize=13,
                arrowprops=dict(arrowstyle="->", color=c["warn"], lw=1.6))
    ax.text(9000, 0.0455, r"$\sigma^2=0.04$", color=c["ok"], fontsize=13, va="bottom", ha="right")
    ax.set_xlabel("number of readings")
    ax.set_ylabel(r"mean of $(y-37.0)^2$")
    ax.set_xlim(1, 10000)
    ax.set_ylim(0, 0.2)
    clean(ax)
    finish(fig, "lec04-kp04-thermo", dark, c)
    return W, H


def report():
    x, y = make(N_TRAIN, SEED)
    coef = np.polyfit(x, y, 2)
    res = y - f(x)
    print(f"model figure: n={N_TRAIN}, seed={SEED}, f(x)=1+2x-1.5x^2, sigma={SIGMA}, sigma^2={SIGMA ** 2:.2f}")
    print(f"  fitted quadratic coefficients (x^2, x, 1): {coef[0]:.3f}, {coef[1]:.3f}, {coef[2]:.3f} (true -1.5, 2, 1)")
    print(f"  noise in this sample: mean {res.mean():+.3f}, mean of squares {np.mean(res ** 2):.4f}; train MSE of the fit {np.mean((y - np.polyval(coef, x)) ** 2):.4f}")
    xt = np.linspace(-1, 1, 20001)
    rng = np.random.default_rng(99)
    xs = rng.uniform(-1, 1, 200000)
    ys = f(xs) + rng.normal(0, SIGMA, xs.size)
    print(f"  fresh test MSE of the fitted curve (200000 points) {np.mean((ys - np.polyval(coef, xs)) ** 2):.4f}; of the true f {np.mean((ys - f(xs)) ** 2):.4f} (floor {SIGMA ** 2:.2f})")
    rng = np.random.default_rng(8)
    e = rng.normal(0, SIGMA, 2000)
    print(f"hist figure: 2000 draws, mean {e.mean():+.4f}, mean of squares {np.mean(e ** 2):.4f}")
    # thermometer
    r3 = np.array([36.9, 37.2, 37.0])
    print(f"thermometer: f=37.0, sigma=0.2, sigma^2={0.2 ** 2:.2f}; three readings squared deviations {np.round((r3 - 37.0) ** 2, 4).tolist()}, mean {np.mean((r3 - 37.0) ** 2):.4f}")
    rr = readings()
    run = np.cumsum((rr - 37.0) ** 2) / np.arange(1, len(rr) + 1)
    for k in (3, 10, 30, 100, 1000, 10000):
        print(f"  mean squared deviation after {k:5d} readings: {run[k - 1]:.4f}")
    # reversal: a model that memorises the noise (1-nearest-neighbour: every training point is reproduced exactly)
    xtr, ytr = make(N_TRAIN, SEED)
    rng = np.random.default_rng(99)                  # the same fresh test set as above
    xs = rng.uniform(-1, 1, 200000)
    ys = f(xs) + rng.normal(0, SIGMA, xs.size)
    nn = np.array([ytr[np.abs(xtr - v).argmin()] for v in xs[:20000]])
    print(f"memorising model (1-nearest-neighbour, n={N_TRAIN}): train MSE 0.0000 (floor {SIGMA ** 2:.2f}); fresh test MSE {np.mean((ys[:20000] - nn) ** 2):.4f} (20000 points, about 2 sigma^2 = {2 * SIGMA ** 2:.2f})")
    coef2 = np.polyfit(xtr, ytr, 2)
    print(f"  quadratic fit on the same data: train MSE {np.mean((ytr - np.polyval(coef2, xtr)) ** 2):.4f}, fresh test MSE {np.mean((ys - np.polyval(coef2, xs)) ** 2):.4f}")
    # floor for ANY predictor independent of the new noise: E(y-g)^2 = E(f-g)^2 + sigma^2
    g = 0.9 + 2.1 * xs - 1.2 * xs ** 2
    print(f"floor check with another predictor g: E(f-g)^2 = {np.mean((f(xs) - g) ** 2):.4f}; E(y-g)^2 = {np.mean((ys - g) ** 2):.4f}; sum E(f-g)^2 + sigma^2 = {np.mean((f(xs) - g) ** 2) + SIGMA ** 2:.4f}")


if __name__ == "__main__":
    sizes = {}
    for fn in (model, hist, thermo):
        sizes[fn.__name__] = [fn(True), fn(False)][0]
    print("sizes", sizes)
    report()
