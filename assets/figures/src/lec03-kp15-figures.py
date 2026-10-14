"""KP15 figures (convergence of over-parameterized networks) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec03-kp15-interpolants, lec03-kp15-width, lec03-kp15-separation.   Prints every number the lecture and the deck quote.
Run:  py assets/figures/src/lec03-kp15-figures.py
"""
import base64
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, ".agents", "skills", "view-from-text"))
import matplotlib.pyplot as plt  # noqa: E402
from jbd_figure import svg_data_uri, use_jbd_style  # noqa: E402

OUT = os.path.join(ROOT, "assets", "figures")
DARK = dict(text="#f2ede4", muted="#b7b0a5", main="#d97757", ref="#6a9fd6", vec="#5fb8a8", ok="#7dbb8c", warn="#e07a70", edge="#e0995f", grid="#b7b0a5")
LIGHT = dict(text="#1c1712", muted="#5a544b", main="#c4572f", ref="#2b6cb0", vec="#1f8a79", ok="#2f855a", warn="#c0392b", edge="#b7791f", grid="#8a8378", bg="#ffffff")
STATS = {}


def skin(c, dark):
    use_jbd_style(18)
    plt.rcParams.update({"text.color": c["text"], "axes.labelcolor": c["text"], "axes.edgecolor": c["muted"], "xtick.color": c["muted"],
                        "ytick.color": c["muted"], "grid.color": c["grid"], "svg.fonttype": "path",
                        "figure.facecolor": "none" if dark else c["bg"], "axes.facecolor": "none" if dark else c["bg"], "savefig.transparent": dark})


def finish(fig, key, dark, c):
    if dark:
        uri = svg_data_uri(fig)
        with open(os.path.join(OUT, key + ".svg"), "wb") as fh:
            fh.write(base64.b64decode(uri.split(",", 1)[1]))
    else:
        fig.savefig(os.path.join(OUT, key + ".png"), dpi=160, facecolor=c["bg"])
    plt.close(fig)


def clean(ax):
    ax.grid(True, alpha=0.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------------ experiment A: over-parameterized least squares (n=3, p=10)
def exp_a():
    rng = np.random.default_rng(3)
    n, p = 3, 10
    X = rng.standard_normal((n, p))
    w_true = np.zeros(p)
    w_true[0] = 2.0
    y = X @ w_true
    lam = np.linalg.eigvalsh(X @ X.T)
    eta = 1.0 / lam.max()
    Xp = np.linalg.pinv(X)
    w_mn = Xp @ y

    def gd(w0, steps=4000):
        w, losses = w0.copy(), []
        for _ in range(steps):
            r = X @ w - y
            losses.append(0.5 * float(r @ r))
            w = w - eta * (X.T @ r)
        losses.append(0.5 * float(np.sum((X @ w - y) ** 2)))
        return w, np.array(losses)

    w0a = np.zeros(p)
    w0b = rng.standard_normal(p)
    w0c = rng.standard_normal(p)
    wa, la = gd(w0a)
    wb, lb = gd(w0b)
    wc, lc = gd(w0c)
    test = lambda w: float(np.sum((w - w_true) ** 2))        # exact test MSE for x ~ N(0, I)
    sols = {"GD from 0": wa, "GD from random 1": wb, "GD from random 2": wc}
    STATS["A"] = dict(n=n, p=p, null_dim=p - n, eta=eta, lam_max=float(lam.max()), lam_min=float(lam.min()),
                      loss0=la[0], loss_100=la[100], loss_end_a=la[-1], loss_end_b=lb[-1], loss_end_c=lc[-1],
                      steps_to_1e6_a=int(np.argmax(la < 1e-6)), steps_to_1e6_b=int(np.argmax(lb < 1e-6)),
                      gap_a_vs_pinv=float(np.linalg.norm(wa - w_mn)), norm_mn=float(np.linalg.norm(w_mn)),
                      norm_b=float(np.linalg.norm(wb)), norm_c=float(np.linalg.norm(wc)),
                      test_mn=test(w_mn), test_a=test(wa), test_b=test(wb), test_c=test(wc), test_true=0.0,
                      rank=int(np.linalg.matrix_rank(X)))
    # null-space check for random-init GD: w_end = w0 + X^+ (y - X w0)
    pred_b = w0b + Xp @ (y - X @ w0b)
    STATS["A"]["formula_gap_b"] = float(np.linalg.norm(wb - pred_b))
    return dict(la=la, lb=lb, lc=lc, tests=[test(wa), test(wb), test(wc)], labels=["GD from 0\n(min-norm)", "GD from\nrandom 1", "GD from\nrandom 2"])


def interpolants(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    r = exp_a()
    W, H = 760, 330
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax1 = fig.add_axes([92 / W, 70 / H, 290 / W, (H - 70 - 30) / H])
    k = np.arange(len(r["la"]))
    ax1.semilogy(k, np.maximum(r["la"], 1e-14), color=c["ref"], lw=2.6, label="from 0")
    ax1.semilogy(k, np.maximum(r["lb"], 1e-14), color=c["main"], lw=2.6, label="random init 1")
    ax1.semilogy(k, np.maximum(r["lc"], 1e-14), color=c["vec"], lw=2.6, ls="--", label="random init 2")
    ax1.set_xlim(0, 160)
    ax1.set_ylim(1e-14, 30)
    ax1.set_xlabel("GD step", fontsize=14)
    ax1.set_ylabel("training loss", fontsize=14)
    ax1.legend(loc="upper right", frameon=False, fontsize=12)
    clean(ax1)
    ax2 = fig.add_axes([470 / W, 70 / H, 270 / W, (H - 70 - 30) / H])
    cols = [c["ref"], c["main"], c["vec"]]
    ax2.bar(range(3), r["tests"], color=cols, width=0.62)
    for i, v in enumerate(r["tests"]):
        ax2.text(i, v + 0.05, f"{v:.2f}", ha="center", color=c["text"], fontsize=13)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["from 0", "random 1", "random 2"], fontsize=12)
    ax2.set_ylabel("test loss", fontsize=14)
    ax2.set_ylim(0, max(r["tests"]) * 1.35)
    clean(ax2)
    finish(fig, "lec03-kp15-interpolants", dark, c)
    return W, H


# ------------------------------------------------------------------ experiment B: shallow quadratic network, width k
def run_width(k, data_seed, init_seed, steps=6000, eta=0.05):
    d, n = 2, 3
    rd = np.random.default_rng(1000 + data_seed)
    X = rd.standard_normal((n, d))
    y = rd.standard_normal(n)
    v = np.concatenate([np.ones((k + 1) // 2), -np.ones(k // 2)])
    ri = np.random.default_rng(init_seed)
    Wm = 0.5 * ri.standard_normal((k, d))
    loss = np.inf
    for _ in range(steps):
        z = X @ Wm.T                    # n x k
        pred = (z ** 2) @ v
        res = pred - y
        loss = 0.5 * float(res @ res) / n
        G = (2 * (res[:, None] * z * v[None, :])).T @ X / n      # k x d
        Wm = Wm - eta * G
        if not np.isfinite(loss) or loss > 1e6:
            break
    z = X @ Wm.T
    res = (z ** 2) @ v - y
    return 0.5 * float(res @ res) / n


_CACHE = {}


def good_data(count=10):
    """first `count` data seeds whose three samples give a well-conditioned system for M (so 12,000 steps are enough to see the end)."""
    seeds = []
    for ds in range(200):
        X = np.random.default_rng(1000 + ds).standard_normal((3, 2))
        A = np.stack([X[:, 0] ** 2, 2 * X[:, 0] * X[:, 1], X[:, 1] ** 2], 1)
        if np.linalg.cond(A) < 12:
            seeds.append(ds)
        if len(seeds) == count:
            break
    return seeds


def target_matrix(ds):
    """the 2x2 symmetric matrix M* with x_i^T M* x_i = y_i for the three samples of data seed ds (a quadratic network computes x^T M x)."""
    rd = np.random.default_rng(1000 + ds)
    X = rd.standard_normal((3, 2))
    y = rd.standard_normal(3)
    A = np.stack([X[:, 0] ** 2, 2 * X[:, 0] * X[:, 1], X[:, 1] ** 2], 1)
    m = np.linalg.solve(A, y)
    return np.array([[m[0], m[1]], [m[1], m[2]]])


def reachable(k, ds):
    """can a width-k network with ceil(k/2) positive and floor(k/2) negative output weights represent M* exactly?"""
    M = target_matrix(ds)
    ev = np.linalg.eigvalsh(M)
    npos, nneg = (k + 1) // 2, k // 2
    need_pos, need_neg = int((ev > 1e-12).sum()), int((ev < -1e-12).sum())
    return need_pos <= npos and need_neg <= nneg


def exp_b():
    if "b" in _CACHE:
        return _CACHE["b"]
    cache_file = os.environ.get("KP15_B_CACHE")           # optional: skip the 4-minute run when only the plot changes
    if cache_file and os.path.exists(cache_file):
        import json
        d = json.load(open(cache_file, encoding="utf-8"))
        STATS["B"] = d["stats"]
        _CACHE["b"] = {int(k): v for k, v in d["frac"].items()}
        return _CACHE["b"]
    widths = [1, 2, 4, 8]
    seeds = good_data()
    frac, failed, feas = {}, {}, {}
    for k in widths:
        ok = tot = 0
        failed[k] = []
        for ds in seeds:
            for isd in range(10):
                tot += 1
                good = run_width(k, ds, 7 + isd, steps=12000, eta=0.03) < 1e-6
                ok += good
                if not good:
                    failed[k].append((ds, 7 + isd))
        frac[k] = ok / tot
        feas[k] = sum(reachable(k, ds) for ds in seeds) / len(seeds)
    retry = {}
    for k in (4, 8):                      # failures where a zero-loss point exists: rerun with a smaller step
        retry[k] = (len(failed[k]), sum(run_width(k, ds, i, steps=60000, eta=0.01) < 1e-6 for ds, i in failed[k]))
    STATS["B"] = dict(widths=widths, data_seeds=seeds, runs_per_width=tot, frac=frac, data_with_exact_fit=feas, retry_small_step=retry)
    _CACHE["b"] = frac
    if cache_file:
        import json
        json.dump({"stats": STATS["B"], "frac": frac}, open(cache_file, "w", encoding="utf-8"))
    return frac


def width(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    frac = exp_b()
    W, H = 640, 330
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([86 / W, 70 / H, (W - 86 - 24) / W, (H - 70 - 30) / H])
    ks = list(frac)
    vals = [100 * frac[k] for k in ks]
    cols = [c["warn"] if k < 4 else c["ok"] for k in ks]
    ax.bar(range(len(ks)), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center", color=c["text"], fontsize=14)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"k={k}" for k in ks], fontsize=14)
    ax.axvline(1.5, color=c["muted"], lw=1.2, ls=":")
    ax.text(1.44, 104, r"$k\geq 2d=4$", color=c["muted"], fontsize=13, ha="right")
    ax.set_ylim(0, 112)
    ax.set_ylabel("runs reaching loss < 1e-6 (%)", fontsize=13)
    clean(ax)
    finish(fig, "lec03-kp15-width", dark, c)
    return W, H


# ------------------------------------------------------------------ experiment C: steps grow like 1/delta^2 as two samples approach
def steps_for(delta, p=50, seed=11, eps=1e-6):
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(p)
    x1 /= np.linalg.norm(x1)
    z = rng.standard_normal(p)
    z -= (z @ x1) * x1
    z /= np.linalg.norm(z)
    x2 = x1 + delta * z                    # ||x1 - x2|| = delta exactly
    X = np.stack([x1, x2])
    y = np.array([1.0, -1.0])
    G = X @ X.T
    lam = np.linalg.eigvalsh(G)
    eta = 1.0 / lam.max()
    w = np.zeros(p)
    loss0 = 0.5 * float(y @ y)
    k = 0
    while True:
        r = X @ w - y
        loss = 0.5 * float(r @ r)
        if loss < eps * loss0 or k > 5_000_000:
            return k, float(lam.min()), float(lam.max())
        w = w - eta * (X.T @ r)
        k += 1


def separation(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    ds = [1.0, 0.5, 0.25, 0.125, 0.0625]
    rows = [steps_for(d) for d in ds]
    ks = np.array([r[0] for r in rows], float)
    slope = float(np.polyfit(np.log(ds), np.log(ks), 1)[0])
    STATS["C"] = dict(deltas=ds, steps=[int(k) for k in ks], lam_min=[r[1] for r in rows], lam_max=[r[2] for r in rows], slope=slope,
                      ratios=[float(ks[i + 1] / ks[i]) for i in range(len(ks) - 1)])
    W, H = 600, 330
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([92 / W, 70 / H, (W - 92 - 24) / W, (H - 70 - 30) / H])
    ax.loglog(ds, ks, "o-", color=c["main"], lw=2.6, ms=9, label="steps to 1e-6 of the initial loss")
    ref = ks[0] * (np.array(ds) / ds[0]) ** -2
    ax.loglog(ds, ref, "--", color=c["ref"], lw=2.2, label=r"slope $-2$")
    for d, k in zip(ds, ks):
        ax.annotate(f"{int(k):,}", (d, k), xytext=(8, -16), textcoords="offset points", color=c["text"], fontsize=12)
    ax.set_xlim(0.05, 1.7)
    ax.set_xlabel(r"separation $\delta=\|x_1-x_2\|$", fontsize=14)
    ax.set_ylabel("GD steps", fontsize=14)
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    clean(ax)
    finish(fig, "lec03-kp15-separation", dark, c)
    return W, H


if __name__ == "__main__":
    for dark in (False, True):
        interpolants(dark)
        width(dark)
        separation(dark)
    for k, v in STATS.items():
        print(k, {a: (round(b, 6) if isinstance(b, float) else b) for a, b in v.items()})
