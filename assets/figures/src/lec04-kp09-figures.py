"""KP09 figures (L1 regularization, soft threshold and the diamond) in two skins from the same data.
    assets/figures/<id>.png (light, lecture)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp09-softthreshold, lec04-kp09-diamond.   Prints every number the lecture and the deck quote.
One coordinate: J = 1/2 h (x - x*)^2, penalty alpha |x|  =>  x_dag = sign(x*) max(|x*| - alpha/h, 0)   (h = 2, alpha = 1).
Two coordinates: J = 1/2 (x - x*)^T H (x - x*), x* = (1.6, 0.9), H = [[2, 1.2], [1.2, 1.5]]; constraints |x|_1 <= 1 and |x|_2 <= 1.
Run:  python3 assets/figures/src/lec04-kp09-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

H1, ALPHA = 2.0, 1.0                                 # one-coordinate example
XSTAR = np.array([1.6, 0.9])
HM = np.array([[2.0, 1.2], [1.2, 1.5]])              # a non-diagonal Hessian
HM_B = np.array([[1.5, -0.9], [-0.9, 1.0]])          # same x*, different orientation (L1 point is not on the axis)
R = 1.0                                              # radius of both constraint sets


def soft(x, thr):
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)


def ridge1(x, h=H1, a=ALPHA):
    return h / (h + a) * x


def jval(x, xs=XSTAR, h=HM):
    d = np.asarray(x) - xs
    return 0.5 * np.einsum("...i,ij,...j->...", d, h, d)


def l1_constrained(xs=XSTAR, h=HM, r=R, n=200001):
    """Minimizer of J over |x|_1 <= r (x* outside): scan the four edges of the diamond."""
    best = (np.inf, None)
    t = np.linspace(0, 1, n)
    for s1 in (1, -1):
        for s2 in (1, -1):
            x = np.stack([s1 * r * t, s2 * r * (1 - t)], axis=1)
            v = jval(x, xs, h)
            i = int(v.argmin())
            if v[i] < best[0]:
                best = (float(v[i]), x[i])
    return best


def l2_constrained(xs=XSTAR, h=HM, r=R, n=400001):
    th = np.linspace(0, 2 * np.pi, n)
    x = r * np.stack([np.cos(th), np.sin(th)], axis=1)
    v = jval(x, xs, h)
    i = int(v.argmin())
    return float(v[i]), x[i]


def softthreshold(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig, ax = axes(W, H)
    thr = ALPHA / H1
    xs = np.linspace(-2, 2, 801)
    ax.plot(xs, xs, color=c["muted"], lw=1.4, ls="--", label=r"$x^{\dagger}=x^{\ast}$ (no penalty)")
    ax.plot(xs, ridge1(xs), color=c["ref"], lw=2.2, label=r"$\ell_2$: $\dfrac{h}{h+\alpha}x^{\ast}$")
    ax.plot(xs, soft(xs, thr), color=c["main"], lw=3.0, label=r"$\ell_1$: soft threshold")
    for v in (-thr, thr):
        ax.axvline(v, color=c["edge"], lw=1.2, ls=":")
    for xv, dx, dy in ((1.2, 8, -16), (0.3, 10, -18), (-0.8, -86, 6)):
        ax.plot([xv], [soft(xv, thr)], "o", color=c["ok"], ms=9, zorder=5)
        ax.annotate(f"{xv:g} -> {soft(xv, thr):g}", (xv, soft(xv, thr)), xytext=(dx, dy), textcoords="offset points", color=c["ok"], fontsize=13)
    ax.annotate(r"$\pm\alpha/h=\pm0.5$", (thr, -1.78), xytext=(6, 0), textcoords="offset points", color=c["edge"], fontsize=13)
    ax.set_xlabel(r"unregularized $x^{\ast}$")
    ax.set_ylabel(r"regularized $x^{\dagger}$")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    clean(ax)
    finish(fig, "lec04-kp09-softthreshold", dark, c)
    return W, H


def diamond(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig = plt.figure(figsize=(W / 100, H / 100))
    pw, ph = 296, 296
    boxes = [(24, 52), (W - 24 - pw, 52)]
    g1, g2 = np.meshgrid(np.linspace(-1.6, 2.4, 500), np.linspace(-1.6, 2.4, 500))
    Jg = jval(np.stack([g1, g2], axis=-1))
    th = np.linspace(0, 2 * np.pi, 400)
    solutions = []
    for k, (x0, y0) in enumerate(boxes):
        ax = fig.add_axes([x0 / W, y0 / H, pw / W, ph / H])
        if k == 0:
            val, xm = l2_constrained()
            ax.fill(R * np.cos(th), R * np.sin(th), color=c["vec"], alpha=0.16)
            ax.plot(R * np.cos(th), R * np.sin(th), color=c["vec"], lw=2.4)
            ttl = r"$\ell_2$ ball: circle"
        else:
            val, xm = l1_constrained()
            dm = np.array([[R, 0], [0, R], [-R, 0], [0, -R], [R, 0]])
            ax.fill(dm[:, 0], dm[:, 1], color=c["vec"], alpha=0.16)
            ax.plot(dm[:, 0], dm[:, 1], color=c["vec"], lw=2.4)
            ttl = r"$\ell_1$ ball: diamond"
        solutions.append(xm)
        ax.contour(g1, g2, Jg, levels=[0.25, 0.6, 1.1], colors=c["ref"], linewidths=1.1, alpha=0.7)
        ax.contour(g1, g2, Jg, levels=[val], colors=c["main"], linewidths=2.0)
        ax.axhline(0, color=c["muted"], lw=0.9)
        ax.axvline(0, color=c["muted"], lw=0.9)
        ax.plot([XSTAR[0]], [XSTAR[1]], "o", color=c["ref"], ms=8)
        ax.plot([xm[0]], [xm[1]], "o", color=c["main"], ms=10, zorder=6)
        ax.annotate(r"$\mathbf{x}^{\ast}$", XSTAR, xytext=(6, 6), textcoords="offset points", color=c["ref"], fontsize=15)
        fig.text((x0 + pw / 2) / W, 20 / H, rf"minimizer $({xm[0]:.2f},\,{xm[1]:.2f})$", color=c["main"], fontsize=15, ha="center")
        ax.set_title(ttl, color=c["text"], fontsize=15, pad=6)
        ax.set_xlim(-1.6, 2.4)
        ax.set_ylim(-1.6, 2.4)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    finish(fig, "lec04-kp09-diamond", dark, c)
    return W, H, solutions


def report():
    np.set_printoptions(precision=4, suppress=True)
    thr = ALPHA / H1
    print(f"one coordinate: h={H1}, alpha={ALPHA}, threshold alpha/h={thr}")
    for xv in (1.2, 0.3, -0.8):
        print(f"  x*={xv}: soft threshold -> {soft(xv, thr):.4f}   ridge h/(h+alpha) x* -> {ridge1(xv):.4f}")
    # brute-force check of the closed form: minimize 1/2 h (x-x*)^2 + alpha |x| on a fine grid
    grid = np.linspace(-3, 3, 600001)
    for xv in (1.2, 0.3, -0.8):
        obj = 0.5 * H1 * (grid - xv) ** 2 + ALPHA * np.abs(grid)
        print(f"  grid argmin for x*={xv}: {grid[obj.argmin()]:.4f}")
    # 2-D constrained picture
    print("2-D: x* =", XSTAR, " H =", HM.tolist(), " r =", R, " |x*|_1 =", float(np.abs(XSTAR).sum()), " |x*|_2 =", round(float(np.linalg.norm(XSTAR)), 4))
    v1, x1 = l1_constrained()
    v2, x2 = l2_constrained()
    print(f"  l1 ball: minimizer {x1} (J={v1:.4f})   l2 ball: minimizer {x2} (J={v2:.4f})   J(0)={float(jval(np.zeros(2))):.4f}")
    v1b, x1b = l1_constrained(h=HM_B)
    v2b, x2b = l2_constrained(h=HM_B)
    print(f"  same x*, H={HM_B.tolist()}: l1 minimizer {x1b} (not on an axis), l2 minimizer {x2b}")
    # how often does the l1 minimizer sit on an axis?
    rng = np.random.default_rng(13)
    trials, hit1, hit2, used = 3000, 0, 0, 0
    t = np.linspace(0, 1, 4001)
    th = np.linspace(0, 2 * np.pi, 20001)
    circle = np.stack([np.cos(th), np.sin(th)], axis=1)
    for _ in range(trials):
        xs = rng.uniform(-3, 3, 2)
        if np.abs(xs).sum() <= R:
            continue
        a = rng.normal(size=(2, 2))
        h = a @ a.T + 0.1 * np.eye(2)
        used += 1
        best = (np.inf, None)
        for s1 in (1, -1):
            for s2 in (1, -1):
                x = np.stack([s1 * R * t, s2 * R * (1 - t)], axis=1)
                v = jval(x, xs, h)
                i = int(v.argmin())
                if v[i] < best[0]:
                    best = (v[i], x[i])
        hit1 += int(min(abs(best[1][0]), abs(best[1][1])) < 1e-9)
        v = jval(circle, xs, h)
        xm = circle[int(v.argmin())]
        hit2 += int(min(abs(xm[0]), abs(xm[1])) < 1e-3)
    print(f"  random ellipses (seed 13, {used} used): l1 minimizer exactly on an axis in {hit1} ({100 * hit1 / used:.1f}%), l2 minimizer within 1e-3 of an axis in {hit2} ({100 * hit2 / used:.1f}%)")
    # a diagonal-Hessian assumption check: lasso with a non-diagonal H is not coordinate-wise soft thresholding
    alpha = 1.0
    xs = XSTAR.copy()
    naive = soft(xs, alpha / np.diag(HM))
    x = xs.copy()
    for _ in range(2000):                               # coordinate descent on 1/2 (x-x*)^T H (x-x*) + alpha |x|_1
        for j in range(2):
            rj = HM[j] @ (x - xs) - HM[j, j] * (x[j] - xs[j])
            x[j] = soft(xs[j] - rj / HM[j, j], alpha / HM[j, j])
    print(f"  non-diagonal H, alpha=1: coordinate-wise soft threshold {naive}  true lasso solution {x}  (objective {float(jval(naive) + alpha * np.abs(naive).sum()):.4f} vs {float(jval(x) + alpha * np.abs(x).sum()):.4f})")
    # l1 is not rotation invariant
    rot = np.deg2rad(30)
    Rm = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    v = np.array([3.0, 0.0])
    print("  |v|_1 for v=(3,0):", float(np.abs(v).sum()), " after rotating by 30 deg:", round(float(np.abs(Rm @ v).sum()), 4), " (|v|_2 stays", round(float(np.linalg.norm(Rm @ v)), 4), ")")


if __name__ == "__main__":
    for dk in (False, True):
        softthreshold(dk)
        diamond(dk)
    report()
