"""KP09 figures (SGD convergence analysis) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec03-kp09-rates, lec03-kp09-band, lec03-kp09-decay.   Prints every number the lecture and the deck quote.
Toy problem (same as KP07/KP08): f_i(w) = (w - a_i)^2 / 2, a = (2,4,6,8), w* = 5, F = mean f_i, F* = 2.5, sigma^2 = 5, L = mu = 1, w0 = 0.
Run:  py assets/figures/src/lec03-kp09-figures.py
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

A = np.array([2.0, 4.0, 6.0, 8.0])
WSTAR, W0 = 5.0, 0.0
SIG2 = float(np.mean((A - WSTAR) ** 2))          # variance of the single-example gradient at w*: 5
E0 = W0 - WSTAR                                   # initial error -5


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


def axes(W, H):
    fig = plt.figure(figsize=(W / 100, H / 100))
    return fig, fig.add_axes([86 / W, 66 / H, (W - 86 - 24) / W, (H - 66 - 28) / H])


# ------------------------------------------------------------------ exact recursions
def exact_residual(eta, K):
    """E[F(w_k) - F*] for k = 0..K (eta: callable k -> step). e_{k+1} = (1-eta)e_k + eta*xi, xi has mean 0 and variance SIG2."""
    m, v = E0, 0.0
    res = [0.5 * (m * m + v)]
    for k in range(K):
        s = eta(k)
        m = (1 - s) * m
        v = (1 - s) ** 2 * v + s * s * SIG2
        res.append(0.5 * (m * m + v))
    return np.array(res), m, v


def gd_residual(eta, K):
    return np.array([0.5 * (E0 * (1 - eta) ** t) ** 2 for t in range(K + 1)])


def mc_residual(eta, K, runs=40000, seed=1):
    rng = np.random.default_rng(seed)
    w = np.full(runs, W0)
    out = [float(np.mean(0.5 * (w - WSTAR) ** 2))]
    for k in range(K):
        i = rng.integers(0, 4, size=runs)
        s = eta(k)
        w = w - s * (w - A[i])
        out.append(float(np.mean(0.5 * (w - WSTAR) ** 2)))
    return np.array(out)


def plateau(eta):
    var = eta * SIG2 / (2 - eta)
    return dict(std=math.sqrt(var), res=0.5 * var)


# ------------------------------------------------------------------ figure 1: rates of full gradient descent
def rates(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    t = np.arange(1, 41)
    gd = gd_residual(0.2, 40)[1:]
    bound = E0 ** 2 / (2 * 0.2 * t)
    ax.semilogy(t, bound, color=c["ref"], lw=2.4, label=r"upper bound $\|w_0-w^{\ast}\|^2/(2\eta t)$")
    ax.semilogy(t, gd, color=c["main"], lw=2.6, label=r"full gradient, $\eta=0.2$: $12.5\cdot0.64^t$")
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel(r"$F(w_t)-F^{\ast}$")
    ax.set_xlim(1, 40)
    ax.set_ylim(1e-6, 1e2)
    ax.legend(loc="lower left", frameon=False, fontsize=13)
    clean(ax)
    finish(fig, "lec03-kp09-rates", dark, c)
    return W, H


# ------------------------------------------------------------------ figure 2: the stagnation band
def trajectories(K=160, seed=3):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=K)
    out = {}
    for eta in (0.2, 0.05):
        w = [W0]
        for k in range(K):
            w.append(w[-1] - eta * (w[-1] - A[idx[k]]))
        out[eta] = np.array(w)
    return out, idx


def band(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    tr, _ = trajectories()
    k = np.arange(len(tr[0.2]))
    for eta, col in ((0.2, c["warn"]), (0.05, c["vec"])):
        sd = plateau(eta)["std"]
        ax.fill_between(k, WSTAR - sd, WSTAR + sd, color=col, alpha=0.16, lw=0)
        ax.plot(k, tr[eta], color=col, lw=1.7, label=rf"$\eta={eta:g}$, band $\pm${sd:.3f}")
    ax.axhline(WSTAR, color=c["ok"], ls="--", lw=1.3)
    ax.annotate(r"$w^{\ast}=5$", (3, WSTAR), xytext=(0, 8), textcoords="offset points", color=c["ok"], fontsize=14)
    ax.set_xlabel(r"step $k$")
    ax.set_ylabel(r"$w_k$")
    ax.set_xlim(0, len(k) - 1)
    ax.set_ylim(-0.3, 9.2)
    ax.legend(loc="lower right", frameon=False, fontsize=13)
    clean(ax)
    finish(fig, "lec03-kp09-band", dark, c)
    return W, H


# ------------------------------------------------------------------ figure 3: schedules
SCHED = {
    "fixed": lambda k: 0.2,
    "inv": lambda k: 1.0 / (k + 1),
    "sqrt": lambda k: 1.0 / math.sqrt(k + 1),
    "sq": lambda k: 0.5 / (k + 1) ** 2,
}


def decay(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    K = 1000
    ks = np.arange(1, K + 1)
    spec = (("sq", c["warn"], r"$\eta_k=0.5/(k+1)^2$: $\sum\eta<\infty$"),
            ("fixed", c["edge"], r"fixed $\eta=0.2$"),
            ("sqrt", c["ref"], r"$\eta_k=1/\sqrt{k+1}$"),
            ("inv", c["vec"], r"$\eta_k=1/(k+1)$"))
    for key, col, lab in spec:
        r, _, _ = exact_residual(SCHED[key], K)
        ax.loglog(ks, r[1:], color=col, lw=2.4, label=lab)
    ax.set_xlabel(r"step $k$")
    ax.set_ylabel(r"$E[F(w_k)]-F^{\ast}$")
    ax.set_xlim(1, K)
    ax.set_ylim(1e-3, 1e1)
    ax.legend(loc="lower left", frameon=False, fontsize=12)
    clean(ax)
    finish(fig, "lec03-kp09-decay", dark, c)
    return W, H


# ------------------------------------------------------------------ numbers
def report():
    print("sigma^2 =", SIG2, " sigma =", round(math.sqrt(SIG2), 3), " F* =", float(np.mean(0.5 * (A - WSTAR) ** 2)), " L = mu = 1")
    # --- full gradient descent
    gd = gd_residual(0.2, 60)
    print("GD eta=0.2: error factor", 1 - 0.2, " residual factor", (1 - 0.2) ** 2)
    for t in (0, 1, 5, 10, 20):
        tight = 12.5 / t if t else float("nan")
        print(f"  t={t:2d} error {abs(E0) * 0.8 ** t:.4f} residual {gd[t]:.6f} bound |w0-w*|^2/(2 eta t) = {62.5 / t if t else float('nan'):.3f}; source form 2L|w0-w*|^2/(t+1) = {50 / (t + 1):.3f}; eta=1/L form L|w0-w*|^2/(2t) = {tight:.3f}")
    print("  bound 62.5/t holds t>=1:", bool(np.all(gd[1:] <= 62.5 / np.arange(1, 61))), "; source-form 50/(t+1) holds in this example too:", bool(np.all(gd <= 50 / (np.arange(61) + 1))))
    for eps in (1e-3, 1e-2):
        t_gd = next(t for t in range(1000) if 12.5 * 0.64 ** t <= eps)
        t_b = math.ceil(62.5 / eps)
        print(f"  residual <= {eps}: GD needs t={t_gd}; the guarantee 62.5/t needs t>={t_b}")
    # --- fixed step SGD
    print("fixed-step SGD stationary values (std = sigma*sqrt(eta/(2-eta)), residual = eta*sigma^2/(2(2-eta))):")
    for eta in (0.4, 0.2, 0.1, 0.05):
        p = plateau(eta)
        n10 = math.ceil(math.log(10) / -math.log(1 - eta))
        print(f"  eta={eta}: std {p['std']:.4f} residual {p['res']:.4f} res/eta {p['res'] / eta:.4f} steps for error x0.1: {n10} half-life {math.log(2) / -math.log(1 - eta):.2f}")
    ex, m, v = exact_residual(SCHED["fixed"], 400)
    mc = mc_residual(SCHED["fixed"], 100)
    print(f"  exact E residual k=100: {ex[100]:.5f}, Monte Carlo (40000 runs): {mc[100]:.5f}; k=400 exact {ex[400]:.5f} (plateau {plateau(0.2)['res']:.5f})")
    tr, idx = trajectories()
    for eta in (0.2, 0.05):
        rng = np.random.default_rng(11)
        w, xs = 0.0, []
        ii = rng.integers(0, 4, size=300000)
        for k in range(300000):
            w = w - eta * (w - A[ii[k]])
            xs.append(w)
        print(f"  eta={eta}: empirical std of w_k (k>1000) {np.std(xs[1000:]):.4f}; plotted-run std over k>=60 {np.std(tr[eta][60:]):.3f}")
    # --- decaying schedules
    inv, _, v_inv = exact_residual(SCHED["inv"], 1000)
    mc_inv = mc_residual(SCHED["inv"], 100)
    print("eta_k=1/(k+1): exact residual k=1,10,100,1000:", [round(float(inv[k]), 5) for k in (1, 10, 100, 1000)], " 2.5/k:", [round(2.5 / k, 5) for k in (1, 10, 100, 1000)])
    print(f"  Monte Carlo k=100: {mc_inv[100]:.5f}; error std at k=100 = sqrt(5/100) = {math.sqrt(5 / 100):.4f}")
    sq, _, _ = exact_residual(SCHED["sqrt"], 1000)
    print("eta_k=1/sqrt(k+1): residual k=10,100,1000:", [round(float(sq[k]), 5) for k in (10, 100, 1000)], " vs 1.25/sqrt(k):", [round(1.25 / math.sqrt(k), 5) for k in (10, 100, 1000)])
    s2, m2, v2 = exact_residual(SCHED["sq"], 100000)
    prod = math.sin(math.pi / math.sqrt(2)) / (math.pi / math.sqrt(2))
    print(f"eta_k=0.5/(k+1)^2: sum eta = {0.5 * math.pi ** 2 / 6:.4f}; prod(1-eta_k) = {prod:.4f}; mean error -> {E0 * prod:.4f}; residual k=10,100,1000,1e5:",
          [round(float(s2[k]), 4) for k in (10, 100, 1000, 100000)], " mean part 0.5*m^2 =", round(0.5 * m2 * m2, 4), " var", round(v2, 4))
    # --- Robbins-Monro sums
    K = 10000
    k = np.arange(K)
    for name, f in (("fixed 0.2", lambda x: 0.2 + 0 * x), ("1/(k+1)", lambda x: 1 / (x + 1)), ("1/sqrt(k+1)", lambda x: 1 / np.sqrt(x + 1)), ("0.5/(k+1)^2", lambda x: 0.5 / (x + 1) ** 2)):
        e = f(k)
        print(f"  sums to K=10^4 {name:12s}: sum eta {e.sum():10.3f}   sum eta^2 {np.sum(e ** 2):10.4f}")
    print("  pi^2/6 =", round(math.pi ** 2 / 6, 4))
    # --- worked example eta_k = 2/(k+2)
    f2 = lambda kk: 2.0 / (kk + 2)
    e = f2(np.arange(1000000))
    print("worked example eta_k=2/(k+2): eta_0 =", f2(0), " sum eta^2 exact = 4*(pi^2/6-1) =", round(4 * (math.pi ** 2 / 6 - 1), 4), " numeric 1e6 terms", round(float(np.sum(e ** 2)), 4),
          " sum eta (1e6 terms) =", round(float(e.sum()), 2))
    r2, m3, v3 = exact_residual(f2, 1000)
    print("  exact residual k=10,100,1000:", [round(float(r2[k]), 5) for k in (10, 100, 1000)], " c^2 sigma^2/((2c-1)k)/2 with c=2:", [round(4 * 5 / 3 / k / 2, 5) for k in (10, 100, 1000)])
    mc2 = mc_residual(f2, 100)
    print(f"  Monte Carlo k=100: {mc2[100]:.5f}")
    print("  eta_k=2/(k+2) first steps:", [round(f2(kk), 4) for kk in range(5)], " variance factor (k/(k+2))^2 at k=3:", round((3 / 5) ** 2, 4))
    for cc in (0.4, 1.0, 2.0):
        rc, _, _ = exact_residual(lambda kk, cc=cc: min(1.0, cc / (kk + 1)), 100000)
        slope = math.log(rc[100000] / rc[10000]) / math.log(10)
        print(f"  eta_k={cc}/(k+1): residual k=1e3 {rc[1000]:.5f}, 1e4 {rc[10000]:.5f}, 1e5 {rc[100000]:.5f}; log-log slope between 1e4 and 1e5 {slope:.3f}")
    for pw in (0.4, 0.7, 1.0):
        e = 1 / (np.arange(2000000) + 1.0) ** pw
        print(f"  eta_k=1/(k+1)^{pw}: sum eta (2e6 terms) {e.sum():.1f}, sum eta^2 {np.sum(e ** 2):.3f}")
    eta_max = 0.2 / 5.1
    print(f"practice: plateau residual <= 0.05 needs eta <= {eta_max:.4f}; check plateau({eta_max:.4f}) = {plateau(eta_max)['res']:.4f}; eta=0.04 -> {plateau(0.04)['res']:.4f}, eta=0.05 -> {plateau(0.05)['res']:.4f}")
    # --- cost comparison (toy)
    k_sgd = math.ceil(2.5 / 0.01)
    t_gd = next(t for t in range(1000) if 12.5 * 0.64 ** t <= 0.01)
    print(f"cost to reach residual 0.01: GD {t_gd} steps x n sample-gradients; SGD eta=1/(k+1): k={k_sgd} samples; crossover n = {k_sgd / t_gd:.1f}")
    print("  n=1e7 B=128 : n/B =", 1e7 / 128, " GD 16 steps x 1e7 =", t_gd * 10 ** 7)
    print("  plateau ratio eta 0.1 / 0.05:", round(plateau(0.1)["res"] / plateau(0.05)["res"], 3),
          " steps ratio:", round(math.ceil(math.log(10) / -math.log(0.9)) / math.ceil(math.log(10) / -math.log(0.95)), 3))


if __name__ == "__main__":
    for dark in (False, True):
        rates(dark)
        band(dark)
        decay(dark)
    report()
