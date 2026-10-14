"""KP14 figures (Efficient SAM: SWP, SDS; Adaptive SAM: ASAM) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec03-kp14-swp-unbiased, lec03-kp14-sds-select, lec03-kp14-asam-scale.   Prints every number the lecture and deck quote.

Settings (shared with KP13): toy bowl f(x,y)=0.5(x^2+25y^2), start (10,2), rho=0.5 (first-order SAM perturbation eps*=rho g/||g||).
SWP: beta=0.5.  SDS: four-sample example f_i(w)=0.5(w-a_i)^2, a=(2,4,6,8), w=7, alpha=1.
ASAM: two-layer linear net L(w1,w2)=0.5(w1 w2 - 1)^2 at the minimizer (a, 1/a); T_w=diag(|w|+gamma), gamma=0.01, rho_ASAM=0.1.
eta (step size) for the bowl run quoted in the lecture: eta=0.05.
Run:  py assets/figures/src/lec03-kp14-figures.py
"""
import base64
import itertools
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

RHO, BETA, ETA = 0.5, 0.5, 0.05
GAMMA_T, RHO_A = 0.01, 0.1
A_SAMP = np.array([2.0, 4.0, 6.0, 8.0])
W_SDS, ALPHA = 7.0, 1.0


# ------------------------------------------------------------------ SWP
def bowl_grad(w):
    return np.array([w[0], 25 * w[1]])


def bowl_f(w):
    return 0.5 * (w[0] ** 2 + 25 * w[1] ** 2)


def eps_star(g, rho=RHO):
    return rho * g / np.linalg.norm(g)


def quartic_grad(w):
    return w ** 3          # f_q = 1/4 (x^4 + y^4)


def masks():
    return [np.array(m, float) for m in itertools.product((1, 0), repeat=2)]


def mask_prob(m, beta=BETA):
    return float(np.prod(np.where(m == 1, beta, 1 - beta)))


def exact_expect(fn, w, es, beta=BETA):
    """Exact E over the 4 masks of fn(w + m*es/beta)."""
    return sum(mask_prob(m, beta) * fn(w + m * es / beta) for m in masks())


def swp_curves(rng, grid, R=300):
    """Relative error of the N-sample mean, averaged (RMS) over R repetitions, for three quantities."""
    w_b = np.array([10.0, 2.0])
    es_b = eps_star(bowl_grad(w_b))
    w_q = np.array([1.0, 1.0])
    es_q = eps_star(quartic_grad(w_q) * 0 + np.array([1.0, 1.0]))   # g = grad f_q(1,1) = (1,1)
    ms = masks()
    pm = np.array([mask_prob(m) for m in ms])
    tgt_eps = es_b
    tgt_gb = bowl_grad(w_b + es_b)
    tgt_gq = quartic_grad(w_q + es_q)
    out = {"eps": [], "gb": [], "gq": []}
    for N in grid:
        cnt = rng.multinomial(N, pm, size=R) / N                      # R x 4 frequencies of the four masks
        e_b = np.array([[sum(c[i] * ms[i][j] * es_b[j] / BETA for i in range(4)) for j in range(2)] for c in cnt])
        gb = np.array([sum(c[i] * bowl_grad(w_b + ms[i] * es_b / BETA) for i in range(4)) for c in cnt])
        gq = np.array([sum(c[i] * quartic_grad(w_q + ms[i] * es_q / BETA) for i in range(4)) for c in cnt])
        for key, arr, tg in (("eps", e_b, tgt_eps), ("gb", gb, tgt_gb), ("gq", gq, tgt_gq)):
            err = np.linalg.norm(arr - tg, axis=1) / np.linalg.norm(tg)
            out[key].append(float(np.sqrt(np.mean(err ** 2))))
    return out


# ------------------------------------------------------------------ SDS
def sds_numbers():
    g_i = W_SDS - A_SAMP
    g = g_i.mean()
    eps = RHO * np.sign(g)
    delta = 0.5 * (W_SDS + eps - A_SAMP) ** 2 - 0.5 * (W_SDS - A_SAMP) ** 2
    return g_i, g, eps, delta


def cost_ratio(r, fwd=1.0, bwd=2.0):
    """Cost of one SDS step relative to SGD: pass 1 (fwd+bwd), perturbed forward on all, backward on a fraction r."""
    return (fwd + bwd + fwd + r * bwd) / (fwd + bwd)


# ------------------------------------------------------------------ ASAM
def L2(w):
    return 0.5 * (w[..., 0] * w[..., 1] - 1.0) ** 2


def sharp_sam(w, rho=RHO, nth=3600, nr=60):
    """max over the disc ||e||<=rho of L(w+e)-L(w), brute force on a polar grid (boundary + interior)."""
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    rr = np.linspace(rho / nr, rho, nr)
    R, TH = np.meshgrid(rr, th)
    pts = np.stack([w[0] + R * np.cos(TH), w[1] + R * np.sin(TH)], axis=-1)
    return float(L2(pts).max() - L2(w))


def sharp_asam(w, rho=RHO_A, gamma=GAMMA_T, nth=3600, nr=60):
    """max over {||T^-1 e||<=rho} of L(w+e)-L(w), T=diag(|w|+gamma): e = T u with ||u||<=rho."""
    t = np.abs(w) + gamma
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    rr = np.linspace(rho / nr, rho, nr)
    R, TH = np.meshgrid(rr, th)
    pts = np.stack([w[0] + t[0] * R * np.cos(TH), w[1] + t[1] * R * np.sin(TH)], axis=-1)
    return float(L2(pts).max() - L2(w))


# ------------------------------------------------------------------ drawing
def skin(c, dark):
    use_jbd_style(16)
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


def fig_swp(dark, curves, grid):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 750, 330
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax1 = fig.add_axes([74 / W, 58 / H, 270 / W, (H - 58 - 22) / H])
    ax2 = fig.add_axes([(74 + 270 + 96) / W, 58 / H, 290 / W, (H - 58 - 22) / H])
    w_b = np.array([10.0, 2.0])
    es = eps_star(bowl_grad(w_b))
    for m in masks():
        e = m * es / BETA
        ax1.plot([e[0]], [e[1]], "o", color=c["ref"], ms=13, alpha=0.9)
        left = e[0] == 0
        ax1.annotate(f"m=({int(m[0])},{int(m[1])})", e, xytext=(-12 if left else 12, 14), textcoords="offset points",
                     ha="right" if left else "left", fontsize=12, color=c["ref"])
    ax1.plot([es[0]], [es[1]], "*", color=c["main"], ms=24, zorder=5)
    ax1.annotate(r"mean = $\epsilon^{\ast}$", es, xytext=(22, 26), textcoords="offset points", fontsize=13, color=c["main"],
                 arrowprops=dict(arrowstyle="-", color=c["main"], lw=1.2))
    th = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(RHO * np.cos(th), RHO * np.sin(th), color=c["muted"], lw=1.6, ls="--")
    ax1.text(-0.5, 0.58, r"$\|\epsilon\|=\rho$", color=c["muted"], fontsize=13)
    ax1.set_xlabel(r"$\hat\epsilon_x$")
    ax1.set_ylabel(r"$\hat\epsilon_y$")
    ax1.set_xlim(-0.55, 0.95)
    ax1.set_ylim(-0.2, 1.3)
    ax1.set_aspect("equal")
    clean(ax1)
    ax2.loglog(grid, curves["eps"], color=c["ref"], lw=2.6, label=r"mean of $\hat\epsilon$ (bowl)")
    ax2.loglog(grid, curves["gb"], color=c["ok"], lw=2.6, ls="--", label="mean gradient, bowl")
    ax2.loglog(grid, curves["gq"], color=c["warn"], lw=2.6, label="mean gradient, quartic")
    ax2.set_xlabel("number of masks $N$")
    ax2.set_ylabel("relative error")
    ax2.legend(loc="lower left", frameon=False, fontsize=12)
    ax2.set_ylim(1e-4, 1.0)
    clean(ax2)
    finish(fig, "lec03-kp14-swp-unbiased", dark, c)
    return W, H


def fig_sds(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 750, 330
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax1 = fig.add_axes([74 / W, 58 / H, 270 / W, (H - 58 - 22) / H])
    ax2 = fig.add_axes([(74 + 270 + 96) / W, 58 / H, 290 / W, (H - 58 - 22) / H])
    _, _, _, delta = sds_numbers()
    xs = np.arange(1, 5)
    cols = [c["main"] if d > ALPHA else c["ref"] for d in delta]
    ax1.bar(xs, delta, color=cols, width=0.62)
    ax1.axhline(ALPHA, color=c["edge"], ls="--", lw=2.2)
    ax1.text(4.45, ALPHA + 0.1, r"$\alpha=1$", color=c["edge"], fontsize=14, ha="right")
    ax1.axhline(0, color=c["muted"], lw=1.2)
    for x, d in zip(xs, delta):
        ax1.text(x, d + (0.12 if d >= 0 else -0.33), f"{d:.3f}", ha="center", color=c["text"], fontsize=13)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"$a={int(a)}$" for a in A_SAMP])
    ax1.set_ylabel(r"$\ell_i(w+\epsilon^{\ast})-\ell_i(w)$")
    ax1.set_ylim(-1.0, 3.3)
    clean(ax1)
    r = np.linspace(0, 1, 101)
    ax2.plot(r, [cost_ratio(x) for x in r], color=c["main"], lw=2.6)
    ax2.axhline(2.0, color=c["muted"], ls="--", lw=1.8)
    ax2.text(0.02, 2.04, "vanilla SAM = 2", color=c["muted"], fontsize=12)
    ax2.axhline(1.0, color=c["muted"], ls=":", lw=1.8)
    ax2.text(0.02, 1.04, "SGD = 1", color=c["muted"], fontsize=12)
    ax2.plot([0.5], [cost_ratio(0.5)], "o", color=c["main"], ms=10)
    ax2.annotate(f"r=0.5: {cost_ratio(0.5):.2f}", (0.5, cost_ratio(0.5)), xytext=(8, -28), textcoords="offset points", fontsize=13, color=c["main"])
    ax2.set_xlabel("kept fraction $r=|B^+|/|B|$")
    ax2.set_ylabel("step cost / SGD step")
    ax2.set_ylim(0.9, 2.15)
    clean(ax2)
    finish(fig, "lec03-kp14-sds-select", dark, c)
    return W, H


def fig_asam(dark, scales, ss, sa):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 750, 330
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax1 = fig.add_axes([74 / W, 58 / H, 270 / W, (H - 58 - 22) / H])
    ax2 = fig.add_axes([(74 + 270 + 96) / W, 58 / H, 290 / W, (H - 58 - 22) / H])
    # left: neighbourhoods at (4, 1/4) over contours of L
    xs = np.linspace(2.4, 5.6, 400)
    ys = np.linspace(-0.7, 1.7, 300)
    X, Y = np.meshgrid(xs, ys)
    ax1.contour(X, Y, 0.5 * (X * Y - 1) ** 2, levels=[0.02, 0.1, 0.4, 1.0, 2.5], colors=c["muted"], linewidths=1.0, alpha=0.7)
    xv = np.linspace(2.4, 5.6, 200)
    ax1.plot(xv, 1 / xv, color=c["ok"], lw=2.4)
    ax1.text(2.5, 0.5, r"$w_1w_2=1$", color=c["ok"], fontsize=13)
    a = 4.0
    w = np.array([a, 1 / a])
    th = np.linspace(0, 2 * np.pi, 300)
    ax1.plot(w[0] + RHO * np.cos(th), w[1] + RHO * np.sin(th), color=c["ref"], lw=2.6)
    t = np.abs(w) + GAMMA_T
    ax1.plot(w[0] + RHO_A * t[0] * np.cos(th), w[1] + RHO_A * t[1] * np.sin(th), color=c["main"], lw=2.6)
    ax1.plot([w[0]], [w[1]], "o", color=c["text"], ms=8)
    ax1.annotate("SAM disc", (w[0] + 0.36, w[1] + 0.36), xytext=(14, 26), textcoords="offset points", fontsize=13, color=c["ref"],
                 arrowprops=dict(arrowstyle="-", color=c["ref"], lw=1.2))
    ax1.annotate("ASAM ellipse", (w[0] + 0.3, w[1] - 0.02), xytext=(-24, -38), ha="right", textcoords="offset points", fontsize=13, color=c["main"],
                 arrowprops=dict(arrowstyle="-", color=c["main"], lw=1.2))
    ax1.set_xlabel(r"$w_1$")
    ax1.set_ylabel(r"$w_2$")
    ax1.set_xlim(2.4, 5.6)
    ax1.set_ylim(-0.7, 1.7)
    ax1.set_aspect("equal")
    clean(ax1)
    ax2.semilogy(scales, ss, "o-", color=c["ref"], lw=2.6, ms=7, label="SAM sharpness")
    ax2.semilogy(scales, sa, "o-", color=c["main"], lw=2.6, ms=7, label="ASAM sharpness")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(scales)
    ax2.set_xticklabels([f"{int(s)}" if s >= 1 else f"1/{int(round(1 / s))}" for s in scales], fontsize=12)
    ax2.set_xlabel(r"scale $a$ of the point $(a,\,1/a)$")
    ax2.set_ylabel("sharpness")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 0.93), frameon=False, fontsize=12)
    clean(ax2)
    finish(fig, "lec03-kp14-asam-scale", dark, c)
    return W, H


if __name__ == "__main__":
    rng = np.random.default_rng(14)
    grid = np.unique(np.round(np.logspace(1, 5, 25)).astype(int))
    curves = swp_curves(rng, grid)
    scales = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    ss = [sharp_sam(np.array([s, 1 / s])) for s in scales]
    sa = [sharp_asam(np.array([s, 1 / s])) for s in scales]
    for dark in (False, True):
        fig_swp(dark, curves, grid)
        fig_sds(dark)
        fig_asam(dark, scales, ss, sa)

    print(f"settings: rho={RHO} beta={BETA} eta={ETA} gamma_T={GAMMA_T} rho_ASAM={RHO_A} alpha={ALPHA}")
    # ---- SWP on the bowl
    w = np.array([10.0, 2.0])
    g = bowl_grad(w)
    es = eps_star(g)
    print("bowl: start", w.tolist(), "f", bowl_f(w), "g", g.tolist(), "||g||", round(float(np.linalg.norm(g)), 4), "eps*", np.round(es, 5).tolist(), "||eps*||", RHO)
    print("SAM point w+eps*", np.round(w + es, 4).tolist(), "g(w+eps*)", np.round(bowl_grad(w + es), 4).tolist())
    for m in masks():
        e = m * es / BETA
        print("  mask", m.astype(int).tolist(), "prob", mask_prob(m), "eps_hat", np.round(e, 5).tolist(), "||eps_hat||", round(float(np.linalg.norm(e)), 4))
    print("exact E[eps_hat]", np.round(exact_expect(lambda x: x - w, w, es), 6).tolist(), "= eps*", np.round(es, 6).tolist())
    ex = exact_expect(bowl_grad, w, es)
    print("bowl: exact E[grad(w+eps_hat)]", np.round(ex, 5).tolist(), " grad(w+eps*)", np.round(bowl_grad(w + es), 5).tolist(), "(gradient is linear -> equal)")
    print("bowl: E[f(w+eps_hat)]", round(exact_expect(bowl_f, w, es), 5), " f(w+eps*)", round(float(bowl_f(w + es)), 5), " f(w)", bowl_f(w))
    print("E||eps_hat||^2 =", round(exact_expect(lambda x: float(np.sum((x - w) ** 2)), w, es), 5), " rho^2/beta =", RHO ** 2 / BETA, " Var per coord:", np.round(es ** 2 * (1 / BETA - 1), 6).tolist())
    r2 = np.random.default_rng(7)
    mm = (r2.random((10000, 2)) < BETA).astype(float)
    print("Monte Carlo N=10000 mean eps_hat", np.round((mm * es / BETA).mean(axis=0), 4).tolist(), "target", np.round(es, 4).tolist())
    wq = np.array([1.0, 1.0])
    esq = eps_star(quartic_grad(wq))
    exq = exact_expect(quartic_grad, wq, esq)
    print("quartic f=1/4(x^4+y^4) at (1,1): g", quartic_grad(wq).tolist(), "eps*", np.round(esq, 4).tolist(), " E[grad(w+eps_hat)]", np.round(exq, 4).tolist(),
          " grad(w+eps*)", np.round(quartic_grad(wq + esq), 4).tolist(), " rel gap", round(float(np.linalg.norm(exq - quartic_grad(wq + esq)) / np.linalg.norm(quartic_grad(wq + esq))), 4))
    for key in ("eps", "gb", "gq"):
        print("rel RMS error", key, {int(n): round(v, 5) for n, v in zip(grid, curves[key]) if int(n) in (10, 100, 1000, 10000, 100000)})
    # SWP trajectory comparison on the bowl: SAM (full eps*) vs SWP-SAM (random mask), eta=0.05, 60 steps
    def run(kind, steps=60, seed=0):
        rg = np.random.default_rng(seed)
        x = np.array([10.0, 2.0])
        for _ in range(steps):
            gg = bowl_grad(x)
            e = eps_star(gg)
            if kind == "swp":
                e = (rg.random(2) < BETA) * e / BETA
            x = x - ETA * bowl_grad(x + e)
        return x
    print(f"bowl run eta={ETA} 60 steps: SAM", np.round(run('sam'), 4).tolist(), "f", round(float(bowl_f(run('sam'))), 5),
          "| SWP (5 seeds) f:", [round(float(bowl_f(run('swp', seed=s))), 5) for s in range(5)])
    # ---- SDS
    g_i, gb, e1, delta = sds_numbers()
    print("SDS: w", W_SDS, "a", A_SAMP.tolist(), "g_i", g_i.tolist(), "batch g", gb, "eps*", e1, "delta_i", np.round(delta, 4).tolist(), "alpha", ALPHA,
          "B+", [int(a) for a, d in zip(A_SAMP, delta) if d > ALPHA], "B-", [int(a) for a, d in zip(A_SAMP, delta) if d < ALPHA])
    print("  delta_i = g_i*eps + eps^2/2 :", np.round(g_i * e1 + 0.5 * e1 ** 2, 4).tolist(), " sum/4 =", round(float(delta.mean()), 4), "(= batch loss increase)")
    print("  cost model (fwd=1, bwd=2): SGD", cost_ratio(0) * 3 - 1, "->", 3, "; SAM 6 (ratio 2.0); SDS ratio r=1,0.75,0.5,0.25,0:", [round(cost_ratio(x), 4) for x in (1, 0.75, 0.5, 0.25, 0)],
          " lower bound 4/3 =", round(4 / 3, 4), " with bwd=3fwd r=0.5:", round(cost_ratio(0.5, 1.0, 3.0), 4))
    # ---- ASAM
    print("ASAM: L=0.5(w1w2-1)^2, minimizers (a,1/a); SAM rho", RHO, " ASAM rho", RHO_A, " gamma", GAMMA_T)
    print("  scales", scales.tolist())
    print("  SAM sharpness  ", [round(x, 5) for x in ss])
    print("  ASAM sharpness ", [round(x, 5) for x in sa])
    print("  ASAM max/min over scales", round(max(sa) / min(sa), 4), " SAM max/min", round(max(ss) / min(ss), 2), " SAM a=1 vs a=4", round(ss[2], 5), round(ss[4], 5), "ratio", round(ss[4] / ss[2], 2))
    sa0 = [sharp_asam(np.array([s, 1 / s]), gamma=0.0) for s in scales]
    print("  ASAM gamma=0:  ", [round(x, 5) for x in sa0], " max/min", round(max(sa0) / min(sa0), 6))
    sa1 = [sharp_asam(np.array([s, 1 / s]), gamma=1.0) for s in scales]
    print("  ASAM gamma=1:  ", [round(x, 5) for x in sa1], " max/min", round(max(sa1) / min(sa1), 3))
    # condition (56) check
    wv = np.array([0.5, 2.0])
    Av = np.diag([4.0, 0.25])
    for gam in (0.0, GAMMA_T):
        T = lambda x: np.diag(np.abs(x) + gam)
        lhs = np.linalg.inv(T(Av @ wv)) @ Av
        rhs = np.linalg.inv(T(wv))
        print(f"  (56) gamma={gam}: w={wv.tolist()}, A=diag(4,0.25): T_Aw^-1 A =", np.round(np.diag(lhs), 5).tolist(), " T_w^-1 =", np.round(np.diag(rhs), 5).tolist(), " max diff", round(float(np.abs(lhs - rhs).max()), 6))
    # eps* formula for p=2 : check against brute force
    wv = np.array([1.0, -2.0, 0.5])
    gv = np.array([0.3, 0.1, -0.8])
    Tv = np.abs(wv) + GAMMA_T
    e_formula = RHO_A * Tv ** 2 * gv / np.linalg.norm(Tv * gv)
    rg = np.random.default_rng(3)
    U = rg.normal(size=(2_000_000, 3))
    U = RHO_A * U / np.linalg.norm(U, axis=1, keepdims=True)
    best = (U * Tv) @ gv
    print("  eps* formula rho T^2 g/||Tg||:", np.round(e_formula, 5).tolist(), " g.eps formula", round(float(gv @ e_formula), 6), " best of 2e6 random boundary points", round(float(best.max()), 6),
          " rho*||Tg|| =", round(float(RHO_A * np.linalg.norm(Tv * gv)), 6), " ||T^-1 eps*|| =", round(float(np.linalg.norm(e_formula / Tv)), 6))
    # ASAM first-order perturbation on the bowl, for the worked example
    wb = np.array([10.0, 2.0])
    gbv = bowl_grad(wb)
    Tb = np.abs(wb) + GAMMA_T
    ea = RHO_A * Tb ** 2 * gbv / np.linalg.norm(Tb * gbv)
    print("  bowl ASAM eps* (rho_ASAM=0.1, gamma=0.01):", np.round(ea, 5).tolist(), " relative to w:", np.round(ea / wb, 5).tolist(), " SAM eps*:", np.round(es, 5).tolist(), " relative:", np.round(es / wb, 5).tolist())
    # ---- extra numbers quoted in the lecture (worked solution, cost model variants, first-order estimates, practice answers)
    print("worked solution: bowl w=(10,2), g=(10,50), ||g||=sqrt(2600)=", round(math.sqrt(2600), 4), " eps*=0.5*g/||g||=", np.round(es, 5).tolist())
    m10 = np.array([1.0, 0.0])
    e10 = m10 * es / BETA
    print("  mask (1,0): eps_hat", np.round(e10, 5).tolist(), " w+eps_hat", np.round(w + e10, 4).tolist(), " grad", np.round(bowl_grad(w + e10), 4).tolist(),
          " vs grad(w+eps*)", np.round(bowl_grad(w + es), 4).tolist(), " vs grad(w)", g.tolist())
    for m in masks():
        print("  mask", m.astype(int).tolist(), " grad(w+eps_hat)", np.round(bowl_grad(w + m * es / BETA), 4).tolist())
    print("  quartic: eps_hat for the 4 masks gives grad x-comp (1+e)^3:", [round(float((1 + m[0] * esq[0] / BETA) ** 3), 4) for m in masks()], " mean", round(float(exq[0]), 4), " at w+eps*:", round(float((1 + esq[0]) ** 3), 4))
    print("cost model variants: bwd=3fwd: SGD", 4, " SAM 8, SDS r=0.5:", round(cost_ratio(0.5, 1.0, 3.0), 4), " r=0:", round(cost_ratio(0, 1.0, 3.0), 4), " r=1:", round(cost_ratio(1, 1.0, 3.0), 4),
          "| bwd=2fwd r=0.25:", round(cost_ratio(0.25), 4))
    for s in (1.0, 4.0, 8.0):
        print(f"  first-order estimate a={s}: SAM 0.5*rho^2*(a^2+a^-2) = {0.5 * RHO ** 2 * (s ** 2 + s ** -2):.4f}   ASAM rho_A^2 = {RHO_A ** 2:.4f}")
    # practice answers
    es_p = np.array([0.2, -0.4, 0.1])
    print("practice 1: beta=0.25 eps*", es_p.tolist(), " mask (1,0,1) eps_hat", (np.array([1, 0, 1]) * es_p / 0.25).tolist(), " E||eps_hat||^2 =", round(float(np.sum(es_p ** 2) / 0.25), 4), " ||eps*||^2", round(float(np.sum(es_p ** 2)), 4))
    d3 = np.array([1.4, 0.2, 0.9, -0.1, 0.6])
    print("practice 2: deltas", d3.tolist(), " alpha=0.5 -> B+ size", int((d3 > 0.5).sum()), " r=", (d3 > 0.5).mean(), " cost ratio", round(cost_ratio((d3 > 0.5).mean()), 4))
    wp, gp = np.array([2.0, 0.5]), np.array([0.4, 1.2])
    Tp = np.abs(wp)
    epp = 0.1 * Tp ** 2 * gp / np.linalg.norm(Tp * gp)
    print("practice 3: w", wp.tolist(), "g", gp.tolist(), "T=diag", Tp.tolist(), " Tg", (Tp * gp).tolist(), " ||Tg||", round(float(np.linalg.norm(Tp * gp)), 4), " eps*", np.round(epp, 5).tolist(), " ||T^-1 eps*||", round(float(np.linalg.norm(epp / Tp)), 5),
          " SAM eps* (rho=0.1)", np.round(0.1 * gp / np.linalg.norm(gp), 5).tolist())
