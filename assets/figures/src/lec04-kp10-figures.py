"""KP10 figures (why the true error rises) in two skins from the same data.
    assets/figures/<id>.png (light, lecture)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp10-three-curves (err, penalty and err+penalty: the five-stage numbers of the source on the left, a computed polynomial experiment on the right),
     lec04-kp10-overlap (training error, a validation set that overlaps the training set, and a clean validation set).
Computed experiment: f(x) = 1 + 2x - 1.5x^2, sigma = 0.3 (sigma^2 = 0.09), fixed design x = linspace(-1, 1, 12), polynomial degree d = 1..10 (p = d + 1 parameters).
For a least-squares fit the sensitivity sum is sum_i d fhat_i / d y_i = tr(H) = p (KP07), so the penalty per point is 2 sigma^2 p / n. The script checks by
simulation (4000 fresh noise draws, same x) that  E[test error at the same x with fresh y] = E[err] + 2 sigma^2 p / n  for every degree.
Overlap demo: n = 15 training points x = linspace(-1, 1, 15), noise seed 21; "clean" validation = 2000 fresh points, "half-overlap" validation = 10 of the
training points + 10 fresh points, "full overlap" validation = the training set itself.
Run:  python3 assets/figures/src/lec04-kp10-figures.py     (prints every number the lecture and the deck quote)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

SIGMA = 0.3
S2 = SIGMA ** 2
N = 12
X = np.linspace(-1, 1, N)
DEGS = list(range(1, 11))
STAGE_ERR = np.array([10, 7, 5, 4, 3])
STAGE_PEN = np.array([0, 1, 3, 6, 10])


def f(x):
    return 1.0 + 2.0 * np.asarray(x) - 1.5 * np.asarray(x) ** 2


def hat(x, d):
    V = np.vander(x, d + 1)
    return V @ np.linalg.pinv(V)


def experiment(reps=4000):
    """Per degree: mean err/n, penalty 2 s2 p/n, simulated in-sample test error (fresh y at the same x), simulated mean (fhat - f)^2 at the design."""
    rng = np.random.default_rng(31)
    out = {}
    E = rng.normal(0, SIGMA, (reps, N))
    E2 = rng.normal(0, SIGMA, (reps, N))
    for d in DEGS:
        H = hat(X, d)
        Y = f(X)[None, :] + E
        Yfit = Y @ H.T
        err = np.mean((Yfit - Y) ** 2)
        test_in = np.mean((Yfit - (f(X)[None, :] + E2)) ** 2)
        true_err = np.mean((Yfit - f(X)[None, :]) ** 2)
        out[d] = dict(err=err, pen=2 * S2 * np.trace(H) / N, trace=np.trace(H), test_in=test_in, true_err=true_err)
    return out


EXP = experiment()


def three_curves(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, 16)
    W, H = 1100, 380
    fig = plt.figure(figsize=(W / 100, H / 100))
    # left: the five stages of the source (sigma^2 = 1)
    ax = fig.add_axes([70 / W, 70 / H, 440 / W, 250 / H])
    st = np.arange(1, 6)
    tot = STAGE_ERR + STAGE_PEN
    ax.plot(st, STAGE_ERR, "-o", color=c["main"], lw=2.4, ms=6, label="err")
    ax.plot(st, STAGE_PEN, marker="o", color=c["warn"], lw=2.4, ms=6, ls="--", label="penalty")
    ax.plot(st, tot, "-o", color=c["ref"], lw=2.8, ms=7, label="err + penalty")
    ax.axvspan(1.75, 3.25, color=c["ok"], alpha=0.16, lw=0)
    ax.text(2.5, 20.5, "lowest: 8", ha="center", color=c["ok"], fontsize=13)
    ax.set_ylim(-0.5, 25)
    ax.set_xticks(st)
    ax.set_xlabel("training stage")
    ax.set_ylabel(r"error ($\sigma^2=1$)")
    ax.set_title("five stages of the source", fontsize=13, color=c["text"])
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    clean(ax)
    # right: computed polynomial experiment (per point)
    ax = fig.add_axes([620 / W, 70 / H, 440 / W, 250 / H])
    ds = np.array(DEGS)
    err = np.array([EXP[d]["err"] for d in DEGS])
    pen = np.array([EXP[d]["pen"] for d in DEGS])
    ti = np.array([EXP[d]["test_in"] for d in DEGS])
    ax.plot(ds, err, "-o", color=c["main"], lw=2.4, ms=5, label="err")
    ax.plot(ds, pen, marker="o", color=c["warn"], lw=2.4, ms=5, ls="--", label="penalty")
    ax.plot(ds, err + pen, "-o", color=c["ref"], lw=2.8, ms=6, label="err + penalty")
    ax.scatter(ds, ti, s=70, facecolors="none", edgecolors=c["text"], linewidths=1.5, zorder=6, label="simulated test")
    ax.set_xticks(ds)
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("error per point")
    ax.set_title("computed: 12 points, $\\sigma^2=0.09$", fontsize=13, color=c["text"])
    ax.set_ylim(-0.01, 0.56)
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    clean(ax)
    finish(fig, "lec04-kp10-three-curves", dark, c)
    return W, H


def overlap_curves():
    """Training error, half-overlap validation, clean validation, all against the degree (n=15)."""
    n = 15
    x = np.linspace(-1, 1, n)
    y = f(x) + np.random.default_rng(21).normal(0, SIGMA, n)
    rng = np.random.default_rng(22)
    xf = rng.uniform(-1, 1, 10)
    yf = f(xf) + rng.normal(0, SIGMA, 10)
    xc = np.random.default_rng(23).uniform(-1, 1, 2000)
    yc = f(xc) + np.random.default_rng(24).normal(0, SIGMA, 2000)
    sel = np.random.default_rng(25).choice(n, 10, replace=False)
    xh, yh = np.concatenate([x[sel], xf]), np.concatenate([y[sel], yf])
    res = {"train": [], "half": [], "clean": []}
    for d in DEGS:
        p = np.polyfit(x, y, d)
        res["train"].append(float(np.mean((np.polyval(p, x) - y) ** 2)))
        res["half"].append(float(np.mean((np.polyval(p, xh) - yh) ** 2)))
        res["clean"].append(float(np.mean((np.polyval(p, xc) - yc) ** 2)))
    return res


OV = overlap_curves()


def overlap(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H, left=96)
    ds = np.array(DEGS)
    ax.semilogy(ds, OV["train"], "-o", color=c["main"], lw=2.4, ms=6, label="training set (fully overlapping)")
    ax.semilogy(ds, OV["half"], "-o", color=c["edge"], lw=2.4, ms=6, label="validation, half overlapping")
    ax.semilogy(ds, OV["clean"], "-o", color=c["ref"], lw=2.4, ms=6, label="clean validation")
    ax.axhline(S2, color=c["ok"], ls="--", lw=1.3)
    ax.annotate("clean: lowest at degrees 2 to 4", (3, OV["clean"][2]), xytext=(10, 62), textcoords="offset points", color=c["text"], fontsize=12,
                arrowprops=dict(arrowstyle="->", color=c["muted"]))
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("MSE (log scale)")
    ax.set_xticks(ds)
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.legend(loc="lower left", frameon=False, fontsize=11)
    clean(ax)
    finish(fig, "lec04-kp10-overlap", dark, c)
    return W, H


def report():
    print("source five-stage table (sigma^2 = 1): err", STAGE_ERR.tolist(), "penalty", STAGE_PEN.tolist(), "err+penalty", (STAGE_ERR + STAGE_PEN).tolist(),
          "-> minimum", int((STAGE_ERR + STAGE_PEN).min()), "at stages", (np.where(STAGE_ERR + STAGE_PEN == (STAGE_ERR + STAGE_PEN).min())[0] + 1).tolist())
    print("computed: n=12 fixed design, sigma^2=0.09, per point; err = mean train MSE over 4000 noise draws, pen = 2 s2 tr(H)/n,")
    print("          test_in = error at the same x with fresh y, true = mean (fhat - f)^2 at the design")
    print("degree p  tr(H)   err     pen     err+pen   test_in   test_in-(err+pen)   true_err   err+pen-s2")
    for d in DEGS:
        e = EXP[d]
        print(f"{d:5d} {d + 1:2d}  {e['trace']:.3f}  {e['err']:.4f}  {e['pen']:.4f}  {e['err'] + e['pen']:.4f}    {e['test_in']:.4f}    {e['test_in'] - e['err'] - e['pen']:+.4f}"
              f"            {e['true_err']:.4f}     {e['err'] + e['pen'] - S2:.4f}")
    tot = [EXP[d]["err"] + EXP[d]["pen"] for d in DEGS]
    print("minimum of err+pen at degree", DEGS[int(np.argmin(tot))])
    print("overlap demo (n=15 training points, seed 21): degree | train | half-overlap val | clean val (2000 fresh)")
    for i, d in enumerate(DEGS):
        print(f"{d:5d}  {OV['train'][i]:.4f}  {OV['half'][i]:.4f}  {OV['clean'][i]:.4f}")
    print("argmin: train", DEGS[int(np.argmin(OV['train']))], "half-overlap", DEGS[int(np.argmin(OV['half']))], "clean", DEGS[int(np.argmin(OV['clean']))])


if __name__ == "__main__":
    for dk in (False, True):
        three_curves(dk)
        overlap(dk)
    report()
