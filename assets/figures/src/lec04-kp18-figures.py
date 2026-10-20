"""KP18 figures (Batch normalization) in two skins from the same data.
    assets/figures/<id>.png (light, lecture)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp18-hist-bn (a skewed feature before / after batch normalization), lec04-kp18-batch-noise (how noisy the batch statistics are against batch size).
Prints every number the lecture and the deck quote.
Run:  python3 assets/figures/src/lec04-kp18-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

EPS = 1e-5


def bn(x, eps=EPS):
    """Batch normalization of one feature over a mini-batch (1/b variance, as in the source)."""
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    return (x - mu) / np.sqrt(var + eps), mu, var


def skew(x):
    z = (x - x.mean()) / x.std()
    return float((z ** 3).mean())


def skewed_feature(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=0.5, sigma=0.8, size=n)


# ------------------------------------------------------------------ figure 1: histogram before / after
def hist_bn(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig = plt.figure(figsize=(W / 100, H / 100))
    x = skewed_feature()
    xh, mu, var = bn(x)
    for k, (data, title, col, xl) in enumerate(((x, rf"before: mean ${mu:.2f}$, std ${np.sqrt(var):.2f}$", c["ref"], (0, 14)),
                                                (xh, r"after: mean $0$, std $1$", c["main"], (-2, 7)))):
        ax = fig.add_axes([(80 + k * 280) / W, 66 / H, 245 / W, 245 / H])
        ax.hist(data, bins=40, color=col, alpha=0.85)
        ax.set_title(title, fontsize=13, color=c["text"])
        ax.set_xlim(*xl)
        ax.set_xlabel(r"feature value $x$" if k == 0 else r"normalized value $\hat x$")
        if k == 0:
            ax.set_ylabel("count")
        else:
            ax.set_yticklabels([])
        clean(ax)
    finish(fig, "lec04-kp18-hist-bn", dark, c)
    return W, H


# ------------------------------------------------------------------ figure 2: noise of the batch statistics
def batch_noise_data(bs=(2, 4, 8, 16, 32, 64, 128, 256), nb=5000, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for b in bs:
        X = rng.normal(0, 1, (nb, b))
        sd = X.std(axis=1)          # 1/b form
        mu = X.mean(axis=1)
        out.append((b, float(sd.mean()), float(sd.std()), float(mu.std())))
    return out


def batch_noise(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    d = batch_noise_data()
    b = np.array([r[0] for r in d])
    m = np.array([r[1] for r in d])
    s = np.array([r[2] for r in d])
    mu_sd = np.array([r[3] for r in d])
    ax.axhline(1.0, color=c["muted"], lw=1.2, ls=":")
    ax.fill_between(b, m - s, m + s, color=c["ref"], alpha=0.25)
    ax.plot(b, m, color=c["ref"], lw=2.4, marker="o", label=r"batch std $\hat\sigma$ (mean $\pm$ 1 sd over batches)")
    ax.plot(b, mu_sd, color=c["main"], lw=2.4, marker="s", label=r"spread of the batch mean $\hat\mu$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(b)
    ax.set_xticklabels([str(v) for v in b])
    ax.set_xlabel(r"batch size $b$")
    ax.set_ylabel("estimate of a $N(0,1)$ feature")
    ax.set_ylim(0, 1.6)
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    clean(ax)
    finish(fig, "lec04-kp18-batch-noise", dark, c)
    return W, H


def report():
    print("== hand example b=4: x = 2,4,6,8")
    x = np.array([2.0, 4.0, 6.0, 8.0])
    xh0, mu, var = bn(x, eps=0.0)
    xh, _, _ = bn(x)
    print("  mu =", mu, " sigma^2 (1/b) =", var, " deviations", x - mu, " squares", (x - mu) ** 2)
    print("  x_hat (eps=0)    =", np.round(xh0, 6))
    print("  x_hat (eps=1e-5) =", np.round(xh, 6), " max diff", float(np.abs(xh - xh0).max()))
    print("  mean of x_hat", float(xh.mean()), " var of x_hat (1/b)", float((xh ** 2).mean()), " theory var/(var+eps) =", var / (var + EPS))
    g, bt = 2.0, 1.0
    print("  gamma=2, beta=1: y =", np.round(g * xh + bt, 4), " mean", float((g * xh + bt).mean()), " var", float(((g * xh + bt) - (g * xh + bt).mean()).var()))
    x2 = 100 * x
    xh2, mu2, var2 = bn(x2)
    print("  second feature x2 = 100*x: mu", mu2, " sigma^2", var2, " x_hat2 =", np.round(xh2, 6), " max|x_hat2 - x_hat| =", float(np.abs(xh2 - xh).max()))
    print("== skewed small batch: 1,1,2,2,3,10")
    s6 = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 10.0])
    s6h, m6, v6 = bn(s6)
    print("  mu", round(float(m6), 4), " sigma^2", round(float(v6), 4), " x_hat", np.round(s6h, 4), " skewness before", round(skew(s6), 4), " after", round(skew(s6h), 4))
    print("== lognormal feature, b = 1000")
    x = skewed_feature()
    xh, mu, var = bn(x)
    print(f"  before: mean {mu:.4f} std {np.sqrt(var):.4f} skewness {skew(x):.4f}   after: mean {xh.mean():.2e} std {xh.std():.4f} skewness {skew(xh):.4f}")
    print("== batch of size 1 :", bn(np.array([3.0]))[0], "(x_hat = 0 for any single value)")
    print("== batch statistics noise (N(0,1) feature, 5000 batches per size)")
    for b, m, s, ms in batch_noise_data():
        print(f"  b={b:3d}: mean batch std {m:.4f} (sd {s:.4f})   sd of batch mean {ms:.4f}  (1/sqrt(b) = {1 / np.sqrt(b):.4f})")


if __name__ == "__main__":
    sizes = {"lec04-kp18-hist-bn": hist_bn(False), "lec04-kp18-batch-noise": batch_noise(False)}
    hist_bn(True)
    batch_noise(True)
    print("sizes", sizes)
    report()
