"""KP12 figures (flat minima and the motivation for SAM) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec03-kp12-two-valleys, lec03-kp12-rescale, lec03-kp12-sam-objective.   Prints every number the lecture and deck quote.
1-D landscape: L(w) = min( 0.5*20*(w+1)^2 , 0.5*0.6*(w-1.5)^2 )  (sharp valley k=20 at w=-1, flat valley k=0.6 at w=1.5), rho=0.3.
2-layer linear net y = w2*w1*x, one sample (x,y)=(1,1):  L(w1,w2) = 0.5*(w1*w2-1)^2  (Hessian at a minimum = [[w2^2,1],[1,w1^2]]).
SAM objective (motivation only):  L_SAM(w) = max_{|e|<=rho} L(w+e).
Run:  py assets/figures/src/lec03-kp12-figures.py
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

K_SHARP, K_FLAT, W_SHARP, W_FLAT, RHO = 20.0, 0.6, -1.0, 1.5, 0.3


def land(w):
    w = np.asarray(w, dtype=float)
    return np.minimum(0.5 * K_SHARP * (w - W_SHARP) ** 2, 0.5 * K_FLAT * (w - W_FLAT) ** 2)


def land_sam(w, rho=RHO, n=1201):
    e = np.linspace(-rho, rho, n)
    w = np.atleast_1d(np.asarray(w, dtype=float))
    return np.array([land(wi + e).max() for wi in w])


def two_layer(w1, w2):
    return 0.5 * (w1 * w2 - 1.0) ** 2


def hess(w1, w2):
    return np.array([[w2 ** 2, 2 * w1 * w2 - 1.0], [2 * w1 * w2 - 1.0, w1 ** 2]])


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


def two_valleys(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([78 / W, 66 / H, (W - 78 - 24) / W, (H - 66 - 28) / H])
    w = np.linspace(-1.9, 3.6, 1500)
    ax.plot(w, land(w), color=c["text"], lw=2.4)
    for w0, col, lab in ((W_SHARP, c["ref"], "sharp  k=20"), (W_FLAT, c["main"], "flat  k=0.6")):
        ax.axvspan(w0 - RHO, w0 + RHO, color=col, alpha=0.16, lw=0)
        ax.plot([w0], [0], "o", color=col, ms=9)
        ax.plot([w0 + RHO], [float(land(w0 + RHO))], "s", color=col, ms=9)
        rise = float(land(w0 + RHO))
        ax.annotate(f"+{rise:g}", (w0 + RHO, rise), xytext=(12, 16) if w0 > 0 else (12, 4), textcoords="offset points", color=col, fontsize=15)
        ax.text(w0, 2.28, lab, color=col, ha="center", fontsize=15)
    ax.set_xlabel(r"weight $w$")
    ax.set_ylabel(r"loss $L(w)$")
    ax.set_xlim(-1.9, 3.6)
    ax.set_ylim(-0.1, 2.6)
    clean(ax)
    finish(fig, "lec03-kp12-two-valleys", dark, c)


def rescale(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([66 / W, 62 / H, (W - 66 - 14) / W, (H - 62 - 14) / H])
    g1 = np.linspace(0.02, 4.9, 500)
    g2 = np.linspace(-0.35, 2.35, 400)
    A, B = np.meshgrid(g1, g2)
    ax.contour(A, B, two_layer(A, B), levels=[0.02, 0.1, 0.4, 1.0, 2.5], colors=[c["muted"]], linewidths=1.1, alpha=0.8)
    xs = np.linspace(0.43, 4.9, 300)
    ax.plot(xs, 1 / xs, color=c["ok"], lw=2.6)
    th = np.linspace(0, 2 * np.pi, 200)
    for (a, b), col, off, ha in (((1.0, 1.0), c["ref"], (26, 34), "left"), ((4.0, 0.25), c["main"], (-6, 30), "right")):
        ax.plot(a + RHO * np.cos(th), b + RHO * np.sin(th), color=col, lw=2.2)
        ax.plot([a], [b], "o", color=col, ms=9)
        lam = float(np.linalg.eigvalsh(hess(a, b)).max())
        ax.annotate(rf"$\lambda_{{\max}}={lam:.2f}$", (a, b), xytext=off, textcoords="offset points", ha=ha, color=col, fontsize=16)
    ax.set_aspect("equal")
    ax.set_xlim(0, 4.9)
    ax.set_ylim(-0.35, 2.35)
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    clean(ax)
    finish(fig, "lec03-kp12-rescale", dark, c)


def sam_objective(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, H = 640, 380
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([78 / W, 66 / H, (W - 78 - 24) / W, (H - 66 - 28) / H])
    w = np.linspace(-1.9, 3.6, 900)
    ax.plot(w, land(w), color=c["muted"], lw=2.2, label=r"$L(w)$")
    ax.plot(w, land_sam(w), color=c["main"], lw=2.6, ls="--", label=r"$\max_{|\epsilon|\leq\rho}L(w+\epsilon)$")
    for w0, col, txt in ((W_SHARP, c["ref"], (-1.68, 0.78)), (W_FLAT, c["vec"], (1.5, -0.27))):
        v = float(land_sam([w0])[0])
        ax.plot([w0], [0], "o", color=col, ms=9)
        ax.plot([w0], [v], "D", color=col, ms=9)
        ax.annotate(f"{v:g}", (w0, v), xytext=txt, textcoords="data", ha="center", va="center", color=col, fontsize=15,
                    arrowprops=dict(arrowstyle="-", color=col, lw=1.2, shrinkA=6, shrinkB=6))
    ax.set_xlabel(r"weight $w$")
    ax.set_ylabel("loss")
    ax.set_xlim(-1.9, 3.6)
    ax.set_ylim(-0.4, 2.6)
    ax.legend(loc="upper right", frameon=False, fontsize=14)
    clean(ax)
    finish(fig, "lec03-kp12-sam-objective", dark, c)


if __name__ == "__main__":
    for dark in (False, True):
        two_valleys(dark)
        rescale(dark)
        sam_objective(dark)
    print(f"1-D: sharp k={K_SHARP} at w={W_SHARP}, flat k={K_FLAT} at w={W_FLAT}, rho={RHO}")
    print("L at both minima:", float(land(W_SHARP)), float(land(W_FLAT)))
    for name, k in (("sharp", K_SHARP), ("flat", K_FLAT)):
        print(f"{name}: worst-case rise in ball 0.5*k*rho^2 =", round(0.5 * k * RHO ** 2, 4), " ratio sharp/flat =", round(K_SHARP / K_FLAT, 2),
              " radius allowed for rise 0.027 =", round(math.sqrt(2 * 0.027 / k), 4))
    print("curve check L(w*+rho):", round(float(land(W_SHARP + RHO)), 4), round(float(land(W_FLAT + RHO)), 4), " both sides sharp:", round(float(land(W_SHARP - RHO)), 4), " flat:", round(float(land(W_FLAT - RHO)), 4))
    xc = (K_SHARP * W_SHARP * 0 + 0)  # crossing point of the two parabolas between the valleys
    ws = np.linspace(W_SHARP, W_FLAT, 200001)
    d = 0.5 * K_SHARP * (ws - W_SHARP) ** 2 - 0.5 * K_FLAT * (ws - W_FLAT) ** 2
    ic = int(np.argmin(np.abs(d)))
    print("parabolas cross at w =", round(float(ws[ic]), 4), " loss there =", round(float(0.5 * K_SHARP * (ws[ic] - W_SHARP) ** 2), 4))
    print("L_SAM at sharp, flat minimum:", [round(float(x), 4) for x in land_sam([W_SHARP, W_FLAT])])
    wg = np.linspace(-1.9, 3.6, 5501)
    ls = land_sam(wg)
    print("argmin of L_SAM on grid:", round(float(wg[int(np.argmin(ls))]), 3), "value", round(float(ls.min()), 4),
          " ; argmin of L on grid (ties at 0):", round(float(wg[int(np.argmin(land(wg)))]), 3))
    print("local minimum of L_SAM in sharp half (w<-0.3):", round(float(wg[wg < -0.3][int(np.argmin(ls[wg < -0.3]))]), 3), round(float(ls[wg < -0.3].min()), 4))
    # --- two-layer linear net -------------------------------------------------
    print("two-layer net y = w2*w1*x, one sample (1,1), L = 0.5*(w1*w2-1)^2")
    for a in (1.0, 4.0, 0.25):
        w1, w2 = a, 1.0 / a
        H = hess(w1, w2)
        ev, evec = np.linalg.eigh(H)
        print(f"  (w1,w2)=({w1:g},{w2:g}) w1*w2={w1 * w2:g} loss={two_layer(w1, w2):.4g}  H=", np.round(H, 4).tolist(), " eig=", np.round(ev, 4).tolist(),
              " trace=", round(float(np.trace(H)), 4), " top eigvec=", np.round(evec[:, 1], 4).tolist())
    print("  lambda_max(alpha)=alpha^2+alpha^-2, alpha=1,2,4,10:", [round(a * a + a ** -2, 4) for a in (1, 2, 4, 10)], " ratio alpha=4 vs 1:", round((16 + 1 / 16) / 2, 3))
    rng = np.random.default_rng(0)
    for a in (1.0, 4.0):
        w1, w2 = a, 1.0 / a
        H = hess(w1, w2)
        lam = float(np.linalg.eigvalsh(H).max())
        # true max of L over the disk of radius rho (polar grid), compared with the quadratic bound 0.5*lam*rho^2
        r = np.linspace(0, RHO, 301)[:, None]
        t = np.linspace(0, 2 * np.pi, 3601)[None, :]
        vals = two_layer(w1 + r * np.cos(t), w2 + r * np.sin(t))
        print(f"  alpha={a:g}: quadratic worst case 0.5*lam*rho^2 = {0.5 * lam * RHO ** 2:.4f}   true max of L over disk rho={RHO}: {vals.max():.5f}")
        sig = 0.1
        d = rng.normal(0, sig, size=(400000, 2))
        mc = float(two_layer(w1 + d[:, 0], w2 + d[:, 1]).mean())
        print(f"           random perturbation sigma={sig}: 0.5*sigma^2*Tr(H) = {0.5 * sig ** 2 * np.trace(H):.5f}   Monte Carlo mean of true L = {mc:.5f}")
    # prediction invariance: y = w2*w1*x for several x
    xs_ = np.array([-2.0, 0.5, 3.0])
    print("  predictions w2*w1*x at x=-2,0.5,3 for alpha=1 and 4:", (1.0 * 1.0 * xs_).tolist(), (0.25 * 4.0 * xs_).tolist())
    # worked example: bound on the quadratic form
    HA, HB = hess(1.0, 1.0), hess(4.0, 0.25)
    for nm, Hm in (("A", HA), ("B", HB)):
        ev, evec = np.linalg.eigh(Hm)
        v = evec[:, 1]
        d = RHO * v
        print(f"  H_{nm}: dL along top eigvec with |dw|={RHO}: {0.5 * d @ Hm @ d:.4f}; along the other eigvec: {0.5 * (RHO * evec[:, 0]) @ Hm @ (RHO * evec[:, 0]):.4f}")
    # direction (1,0)*rho for H_B: shows the bound is not tight in other directions
    for nm, Hm in (("A", HA), ("B", HB)):
        e1 = np.array([RHO, 0.0])
        e2 = np.array([0.0, RHO])
        print(f"  H_{nm}: dw=(0.3,0): {0.5 * e1 @ Hm @ e1:.4f}   dw=(0,0.3): {0.5 * e2 @ Hm @ e2:.4f}")
    # a 2-D check of the bound 0.5*dw^T H dw <= 0.5*lam_max*|dw|^2 on random vectors
    mx = 0.0
    for Hm in (HA, HB):
        lam = float(np.linalg.eigvalsh(Hm).max())
        v = rng.normal(size=(100000, 2))
        v *= RHO / np.linalg.norm(v, axis=1, keepdims=True)
        q = 0.5 * np.einsum("ni,ij,nj->n", v, Hm, v)
        mx = max(mx, float((q / (0.5 * lam * RHO ** 2)).max()))
    print("  max over random dw of quadratic form / bound (should be <= 1):", round(mx, 6))
    # flat-vs-sharp ranking by sigma: allowed radius for a fixed rise
    print("  radius for rise 0.05 in the 1-D valleys:", [round(math.sqrt(2 * 0.05 / k), 4) for k in (K_SHARP, K_FLAT)])
