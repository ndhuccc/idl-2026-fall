"""KP15 figures (early stopping) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp15-paths (weight-space paths on elliptical contours), lec04-kp15-valcurve (validation-error U-curve).   Prints every number the lecture and the deck quote.
Part A: quadratic J(w) = 1/2 (w-w*)^T H (w-w*), H = diag(1, 0.25), w* = (1.8, 2.6), gradient descent from w = 0 with eta = 0.1;
        ridge path w_dagger(alpha) = (lambda_j/(lambda_j+alpha)) w*_j  for alpha from large to small.
Part B: computed demo (fixed seed): 13 Legendre features, 20 noisy training points of sin(1.5 pi x), noise std 0.3, gradient descent from w = 0.
Run:  python3 assets/figures/src/lec04-kp15-figures.py
"""
import os
import sys

import numpy as np
from numpy.polynomial import legendre as L

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

# ------------------------------------------------------------------ part A: two-direction quadratic
LAM = np.array([1.0, 0.25])
WSTAR = np.array([1.8, 2.6])
ETA, T_STOP, ALPHA = 0.1, 10, 1.0


def gd_path(t_max):
    """Iterates w^(0..t_max) of gradient descent from the origin: w <- w - eta * H (w - w*)."""
    w = np.zeros(2)
    out = [w.copy()]
    for _ in range(t_max):
        w = w - ETA * LAM * (w - WSTAR)
        out.append(w.copy())
    return np.array(out)


def ridge(alpha):
    """Weight-decay (ridge) minimiser of J + alpha/2 |w|^2: each eigen-direction is multiplied by lambda/(lambda+alpha)."""
    return LAM / (LAM + alpha) * WSTAR


def J(w):
    d = np.asarray(w) - WSTAR
    return 0.5 * np.sum(LAM * d * d, axis=-1)


# ------------------------------------------------------------------ part B: validation curve demo
def demo(t_max=20000):
    rng = np.random.default_rng(3)
    n, m, deg = 20, 400, 12
    f = lambda x: np.sin(1.5 * np.pi * x)  # noqa: E731
    xt = np.sort(rng.uniform(-1, 1, n))
    yt = f(xt) + rng.normal(0, 0.3, n)
    xv = np.linspace(-1, 1, m)
    yv = f(xv) + rng.normal(0, 0.3, m)
    feat = lambda x: np.stack([L.legval(x, [0] * k + [1]) for k in range(deg + 1)], 1)  # noqa: E731
    Xt, Xv = feat(xt), feat(xv)
    H = Xt.T @ Xt / n
    eta = 0.9 / np.linalg.eigvalsh(H).max()
    w = np.zeros(deg + 1)
    tr, va, ws = [], [], []
    for _ in range(t_max + 1):
        tr.append(np.mean((Xt @ w - yt) ** 2))
        va.append(np.mean((Xv @ w - yv) ** 2))
        ws.append(w.copy())
        w = w - eta * Xt.T @ (Xt @ w - yt) / n
    return dict(tr=np.array(tr), va=np.array(va), ws=np.array(ws), eta=eta, H=H, Xt=Xt, yt=yt, Xv=Xv, yv=yv)


# ------------------------------------------------------------------ figure 1: paths in weight space
def paths(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 600, 440
    fig, ax = axes(W, H)
    g1, g2 = np.meshgrid(np.linspace(-0.6, 3.8, 400), np.linspace(-0.6, 4.6, 400))
    z = 0.5 * (LAM[0] * (g1 - WSTAR[0]) ** 2 + LAM[1] * (g2 - WSTAR[1]) ** 2)
    ax.contour(g1, g2, z, levels=sorted([0.3, 0.9, 1.6, 3.6, float(J(np.zeros(2)))]), colors=[c["muted"]], linewidths=1.0, alpha=0.7)
    p = gd_path(60)
    ax.plot(p[:, 0], p[:, 1], color=c["main"], lw=1.4, alpha=0.5)
    ax.plot(p[:T_STOP + 1, 0], p[:T_STOP + 1, 1], "o-", color=c["main"], lw=2.4, ms=4.5, label=r"gradient descent, $t=0..10$")
    ax.plot(*p[T_STOP], "o", color=c["main"], ms=11, mfc="none", mew=2.2)
    al = np.logspace(2, -2, 200)
    r = np.array([ridge(a) for a in al])
    ax.plot(r[:, 0], r[:, 1], color=c["ref"], lw=2.4, ls="--", label=r"weight decay, $\alpha:\infty\to0$")
    ax.plot(*ridge(ALPHA), "s", color=c["ref"], ms=10)
    ax.plot(*WSTAR, "*", color=c["ok"], ms=15)
    ax.plot(0, 0, "o", color=c["text"], ms=7)
    ax.annotate(r"$w^{*}$", WSTAR, xytext=(8, 6), textcoords="offset points", color=c["ok"], fontsize=15)
    ax.annotate(r"$w^{(10)}$", p[T_STOP], xytext=(12, -16), textcoords="offset points", color=c["main"], fontsize=14)
    ax.annotate(r"$w^{\dagger}(1)$", ridge(ALPHA), xytext=(-52, 6), textcoords="offset points", color=c["ref"], fontsize=14)
    ax.annotate(r"start $0$", (0, 0), xytext=(8, -16), textcoords="offset points", color=c["text"], fontsize=13)
    ax.set_xlabel(r"$w_1$ (curvature $\lambda_1=1$)")
    ax.set_ylabel(r"$w_2$ (curvature $\lambda_2=0.25$)")
    ax.set_xlim(-0.6, 3.8)
    ax.set_ylim(-0.6, 4.6)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    clean(ax)
    finish(fig, "lec04-kp15-paths", dark, c)
    return W, H


# ------------------------------------------------------------------ figure 2: validation error U curve
def valcurve(dark, d):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    t = np.arange(len(d["va"]))
    ts = np.maximum(t, 1)
    ax.semilogx(ts[1:], d["tr"][1:], color=c["ref"], lw=2.4, label="training error")
    ax.semilogx(ts[1:], d["va"][1:], color=c["main"], lw=2.4, label="validation error")
    ax.axhline(0.09, color=c["muted"], lw=1.2, ls=":")
    ax.text(45, 0.103, r"noise floor $\sigma^2=0.09$", color=c["muted"], fontsize=12)
    tb = int(np.argmin(d["va"]))
    ax.axvline(tb, color=c["ok"], lw=1.6, ls="--")
    ax.plot(tb, d["va"][tb], "o", color=c["ok"], ms=10)
    ax.annotate(rf"stop at $t={tb}$", (tb, d["va"][tb]), xytext=(14, 54), textcoords="offset points", color=c["ok"], fontsize=14,
                arrowprops=dict(arrowstyle="->", color=c["ok"], lw=1.4))
    ax.set_xlabel(r"gradient-descent step $t$ (log scale)")
    ax.set_ylabel("mean squared error")
    ax.set_xlim(1, 20000)
    ax.set_ylim(0, 0.7)
    ax.legend(loc="upper left", bbox_to_anchor=(0.46, 1.0), frameon=False, fontsize=13)
    clean(ax)
    finish(fig, "lec04-kp15-valcurve", dark, c)
    return W, H


def report(d):
    print("lambda =", LAM, " w* =", WSTAR, " eta =", ETA, " t =", T_STOP, " alpha = 1/(t eta) =", 1 / (T_STOP * ETA))
    print("GD closed form check, direction 1:", round(float(gd_path(T_STOP)[T_STOP, 0]), 4), "=", round(float((1 - (1 - ETA * LAM[0]) ** T_STOP) * WSTAR[0]), 4))
    print("table  lambda | eta*lambda | (1-eta*lambda)^10 | alpha/(lambda+alpha) | GD factor 1-(1-eta*lambda)^10 | WD factor lambda/(lambda+alpha)")
    for lam in (0.1, 0.25, 0.5, 1.0, 4.0):
        a, b = (1 - ETA * lam) ** T_STOP, ALPHA / (lam + ALPHA)
        print(f"   {lam:5.2f} | {ETA * lam:5.3f} | {a:6.4f} | {b:6.4f} | {1 - a:6.4f} | {1 - b:6.4f}")
    w10, wr = gd_path(T_STOP)[T_STOP], ridge(ALPHA)
    print("w(10) =", np.round(w10, 4), " w_dagger(1) =", np.round(wr, 4), " w* =", WSTAR, " J(0) =", round(float(J(np.zeros(2))), 4),
          " J(w10) =", round(float(J(w10)), 4), " J(wdagger) =", round(float(J(wr)), 4))
    print("distance |w(10)-w_dagger(1)| =", round(float(np.linalg.norm(w10 - wr)), 4), " |w*| =", round(float(np.linalg.norm(WSTAR)), 4))
    print("stability limit 2/lambda_max =", 2 / LAM.max(), " (eta =", ETA, ")")
    va, tr = d["va"], d["tr"]
    tb = int(np.argmin(va))
    print(f"demo: eta={d['eta']:.4f}  best validation step t={tb}: val={va[tb]:.4f} train={tr[tb]:.4f}; t=20000: val={va[-1]:.4f} train={tr[-1]:.4f}; t=0 val={va[0]:.4f}")
    # patience rule: evaluate every 10 steps, stop after 20 evaluations (200 steps) without a new best, return the best weights
    best, best_t, bad = np.inf, 0, 0
    for t in range(0, len(va), 10):
        if va[t] < best - 1e-12:
            best, best_t, bad = va[t], t, 0
        else:
            bad += 1
            if bad >= 20:
                break
    print(f"patience rule (check every 10 steps, patience 20 checks): stops at t={t}, returns weights of t={best_t} with val={best:.4f}")
    # weight decay with alpha = 1/(t eta) on the same demo
    H, Xt, yt, Xv, yv = d["H"], d["Xt"], d["yt"], d["Xv"], d["yv"]
    n = len(yt)
    for tt in (tb, 100, 1000):
        a = 1 / (tt * d["eta"])
        wr = np.linalg.solve(H + a * np.eye(len(H)), Xt.T @ yt / n)
        wg = d["ws"][tt]
        print(f"  t={tt}: alpha=1/(t eta)={a:.4f}  GD val={va[tt]:.4f}  ridge val={np.mean((Xv @ wr - yv) ** 2):.4f}  rel. weight distance={np.linalg.norm(wg - wr) / np.linalg.norm(wr):.3f}")


if __name__ == "__main__":
    d = demo()
    for dark in (False, True):
        paths(dark)
        valcurve(dark, d)
    report(d)
