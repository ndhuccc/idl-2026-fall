"""KP08 figures (L2 regularization shrinks along the Hessian eigen-directions) in two skins from the same data.
    assets/figures/<id>.png (light, lecture)    assets/figures/<id>.svg (dark, deck)
Ids: lec04-kp08-ellipse, lec04-kp08-factor, lec04-kp08-dof.   Prints every number the lecture and the deck quote.
Setup: J(x) = 1/2 (x-x*)^T H (x-x*), penalty (alpha/2)||x||^2  =>  x_dag = (H + alpha I)^-1 H x*.
H = U diag(4, 0.5) U^T with U a rotation by -25 degrees; x* = U (10, 10)^T; alpha = 1.
Run:  python3 assets/figures/src/lec04-kp08-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

LAM = np.array([4.0, 0.5])
ALPHA = 1.0
THETA = np.deg2rad(-25.0)
U = np.array([[np.cos(THETA), -np.sin(THETA)], [np.sin(THETA), np.cos(THETA)]])   # columns = eigenvectors u1, u2
H = U @ np.diag(LAM) @ U.T
XSTAR_EIG = np.array([10.0, 10.0])
XSTAR = U @ XSTAR_EIG


def factor(lam, alpha):
    return lam / (lam + alpha)


def xdag(alpha, x=XSTAR, h=H):
    return np.linalg.solve(h + alpha * np.eye(len(x)), h @ x)


def dof(alpha, lam=LAM):
    return float(np.sum(factor(lam, alpha)))


def ellipse(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H_PX = 640, 380
    fig, ax = axes(W, H_PX, left=40, bottom=36, right=16, top=14)
    xd = xdag(ALPHA)
    gx, gy = np.meshgrid(np.linspace(-6, 20, 600), np.linspace(-12, 16, 600))
    d = np.stack([gx - XSTAR[0], gy - XSTAR[1]], axis=-1)
    Jv = 0.5 * np.einsum("...i,ij,...j->...", d, H, d)
    Jd = 0.5 * (xd - XSTAR) @ H @ (xd - XSTAR)
    J0 = 0.5 * XSTAR @ H @ XSTAR
    ax.contour(gx, gy, Jv, levels=[8, 40, 90, 150], colors=c["ref"], linewidths=1.2, alpha=0.55)
    ax.contour(gx, gy, Jv, levels=[Jd], colors=c["main"], linewidths=1.8)
    ax.contour(gx, gy, Jv, levels=[J0], colors=c["muted"], linewidths=1.2, linestyles="--")
    t = np.array([-14.0, 14.0])
    for k, col in ((0, c["ok"]), (1, c["vec"])):
        ax.plot(t * U[0, k], t * U[1, k], color=col, lw=1.6, ls=(0, (6, 4)))
    # components along the eigen-directions: origin -> along u1 -> x* (or x_dag)
    for comp, col, ls in ((XSTAR_EIG, c["ref"], "-"), (U.T @ xd, c["main"], "-")):
        p1 = U[:, 0] * comp[0]
        ax.plot([0, p1[0]], [0, p1[1]], color=col, lw=2.4, ls=ls)
        tip = U @ comp
        ax.plot([p1[0], tip[0]], [p1[1], tip[1]], color=col, lw=2.4, ls=ls)
    ax.plot([0], [0], "o", color=c["text"], ms=7)
    ax.plot([XSTAR[0]], [XSTAR[1]], "o", color=c["ref"], ms=10)
    ax.plot([xd[0]], [xd[1]], "o", color=c["main"], ms=10)
    ax.annotate(r"$\mathbf{x}^{\ast}$", XSTAR, xytext=(-4, 12), textcoords="offset points", color=c["ref"], fontsize=17, ha="center")
    ax.annotate(r"$\mathbf{x}^{\dagger}$", xd, xytext=(-44, -2), textcoords="offset points", color=c["main"], fontsize=17)
    ax.annotate(r"$\mathbf{0}$", (0, 0), xytext=(-16, -18), textcoords="offset points", color=c["text"], fontsize=15)
    ax.annotate(r"$\mathbf{u}_1$ ($\lambda=4$)", (11.2, -6.8), color=c["ok"], fontsize=13)
    ax.annotate(r"$\mathbf{u}_2$ ($\lambda=0.5$)", (3.6, 9.6), xytext=(8, -2), textcoords="offset points", color=c["vec"], fontsize=13)
    ax.set_xlim(-5, 18)
    ax.set_ylim(-8, 12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    finish(fig, "lec04-kp08-ellipse", dark, c)
    return W, H_PX


def factor_fig(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H_PX = 640, 380
    fig, ax = axes(W, H_PX)
    al = np.linspace(0, 10, 500)
    for lam, col, tag in ((4.0, c["ok"], r"$\lambda=4$"), (0.5, c["vec"], r"$\lambda=0.5$")):
        ax.plot(al, factor(lam, al), color=col, lw=2.8, label=tag)
        ax.plot([ALPHA], [factor(lam, ALPHA)], "o", color=col, ms=9)
        ax.annotate(f"{factor(lam, ALPHA):.3f}", (ALPHA, factor(lam, ALPHA)), xytext=(14, 8), textcoords="offset points", color=col, fontsize=14)
    ax.axvline(ALPHA, color=c["muted"], lw=1.0, ls="--")
    ax.set_xlabel(r"strength $\alpha$")
    ax.set_ylabel(r"shrink factor $\lambda/(\lambda+\alpha)$")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", frameon=False, fontsize=14)
    clean(ax)
    finish(fig, "lec04-kp08-factor", dark, c)
    return W, H_PX


def dof_fig(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H_PX = 640, 380
    fig, ax = axes(W, H_PX)
    al = np.linspace(0, 10, 500)
    ax.plot(al, [dof(a) for a in al], color=c["main"], lw=2.8, label=r"$\sum_j \lambda_j/(\lambda_j+\alpha)$")
    for a, off in ((0, (10, -2)), (1, (10, 6)), (3, (10, 6)), (10, (-46, 12))):
        ax.plot([a], [dof(a)], "o", color=c["ref"], ms=8)
        ax.annotate(f"{dof(a):.3f}", (a, dof(a)), xytext=off, textcoords="offset points", color=c["ref"], fontsize=14)
    ax.set_xlabel(r"strength $\alpha$")
    ax.set_ylabel("effective parameters")
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(0, 2.3)
    ax.legend(loc="upper right", frameon=False, fontsize=14)
    clean(ax)
    finish(fig, "lec04-kp08-dof", dark, c)
    return W, H_PX


def report():
    np.set_printoptions(precision=4, suppress=True)
    print("lambda =", LAM, " alpha =", ALPHA, " rotation -25 deg")
    print("H =\n", H)
    print("eigenvalues of H:", np.linalg.eigvalsh(H)[::-1])
    print("x* (original coords) =", XSTAR, "  in eigenbasis =", U.T @ XSTAR)
    print("shrink factors at alpha=1:", factor(LAM, ALPHA), " effective parameters:", round(dof(ALPHA), 4))
    xd = xdag(ALPHA)
    xe = np.diag(factor(LAM, ALPHA)) @ XSTAR_EIG
    print("x_dag direct (H+aI)^-1 H x* =", xd, " in eigenbasis =", U.T @ xd, " eigen formula =", xe, " max diff:", float(np.abs(U.T @ xd - xe).max()))
    grad = H @ (xd - XSTAR) + ALPHA * xd
    print("gradient of J+(alpha/2)|x|^2 at x_dag:", grad)
    print("|x*| =", float(np.linalg.norm(XSTAR)), " |x_dag| =", float(np.linalg.norm(xd)))
    print("J(x*) = 0, J(x_dag) =", float(0.5 * (xd - XSTAR) @ H @ (xd - XSTAR)), " J(0) =", float(0.5 * XSTAR @ H @ XSTAR))
    print("table alpha: factor(4), factor(0.5), effective parameters")
    for a in (0, 0.5, 1, 3, 10):
        print(f"  alpha={a}: {factor(4.0, a):.4f}  {factor(0.5, a):.4f}  dof={dof(a):.4f}")
    print("alpha for which the lambda=0.5 factor is 0.5:", 0.5, " (factor = lambda/(lambda+alpha) = 1/2 iff alpha = lambda)")
    # diagonal-H check from the lecture's 1-D link with KP07: J=(w-w*)^2 => H=2, penalty alpha_K w^2 = (alpha/2) w^2 with alpha=2 alpha_K
    for ak in (0.5, 1, 3):
        print(f"  KP07 link: alpha_K={ak}: 2/(2+2*alpha_K)={2 / (2 + 2 * ak):.4f}  =  1/(1+alpha_K)={1 / (1 + ak):.4f}")
    # ridge regression is exactly the quadratic case
    rng = np.random.default_rng(12)
    n, p = 30, 3
    X = rng.normal(size=(n, p)) * np.array([3.0, 1.0, 0.3])
    y = X @ np.array([1.0, -2.0, 4.0]) + 0.5 * rng.normal(size=n)
    Hx = X.T @ X
    w_ols = np.linalg.solve(Hx, X.T @ y)
    lam, _ = np.linalg.eigh(Hx)
    s = np.linalg.svd(X, compute_uv=False)
    print("ridge regression check: n=30, p=3, X columns scaled 3, 1, 0.3")
    print("  eigenvalues of X^T X:", lam[::-1], " squared singular values:", s ** 2)
    for a in (1.0, 10.0):
        w_r = np.linalg.solve(Hx + a * np.eye(p), X.T @ y)
        w_f = np.linalg.solve(Hx + a * np.eye(p), Hx @ w_ols)
        hat = X @ np.linalg.solve(Hx + a * np.eye(p), X.T)
        print(f"  alpha={a}: ridge w = {w_r}  formula (H+aI)^-1 H w_ols = {w_f}  max diff {float(np.abs(w_r - w_f).max()):.2e}")
        print(f"    trace of hat matrix = {np.trace(hat):.4f}   sum lambda/(lambda+alpha) = {float(np.sum(lam / (lam + a))):.4f}")
    print("  alpha=0: trace of hat matrix =", round(float(np.trace(X @ np.linalg.solve(Hx, X.T))), 4), "(= p)")


if __name__ == "__main__":
    for dk in (False, True):
        ellipse(dk)
        factor_fig(dk)
        dof_fig(dk)
    report()
