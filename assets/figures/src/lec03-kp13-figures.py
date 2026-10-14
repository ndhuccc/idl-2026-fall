"""KP13 figures (vanilla SAM) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec03-kp13-worst-perturbation, lec03-kp13-worst-loss.   Prints every number the lecture and deck quote.
Toy bowl f(w)=0.5 w^T H w, H=diag(1,25), start (10,2).  SAM radius rho=0.5, learning rate eta=0.02 (KP13 and KP14 share them).
Plain GD:  w <- w - eta * grad f(w).
Vanilla SAM (p=2):  eps = rho g/|g|,  w <- w - eta * grad f(w + eps)   (the second-order term of d eps/dw is dropped).
The exact inner maximum of the quadratic over the ball is solved by bisection on the Lagrange multiplier (trust-region form).
Run:  py assets/figures/src/lec03-kp13-figures.py
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

D = np.array([1.0, 25.0])
H = np.diag(D)
RHO, ETA, T = 0.5, 0.02, 100
W0 = np.array([10.0, 2.0])


def f(w):
    return 0.5 * float(w @ H @ w)


def grad(w):
    return H @ w


def eps_star(w, rho=RHO):
    g = grad(w)
    return rho * g / np.linalg.norm(g)


def exact_max(w, rho=RHO):
    """max_{|e|<=rho} f(w+e) for the quadratic f: boundary point e=(lam I-H)^-1 H w with |e|=rho, lam>lambda_max."""
    hw = D * w
    top = int(np.argmax(D))
    dm = D[top]
    if abs(hw[top]) < 1e-12:                       # hard case: no component along the top eigenvector
        e = np.zeros(2)
        for i in range(2):
            if i != top:
                e[i] = hw[i] / (dm - D[i])
        r2 = rho ** 2 - float(e @ e)
        if r2 >= 0:
            e[top] = math.sqrt(r2)
            return e, f(w + e)
    lo, hi = dm + 1e-12, 1e7
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.sqrt(np.sum((hw / (mid - D)) ** 2)) > rho:
            lo = mid
        else:
            hi = mid
    e = hw / (hi - D)
    return e, f(w + e)


def worst(w, rho=RHO):
    return exact_max(w, rho)[1]


def run(kind, steps=T):
    w = W0.copy()
    ws = [w.copy()]
    for _ in range(steps):
        if kind == "gd":
            w = w - ETA * grad(w)
        else:
            w = w - ETA * grad(w + eps_star(w))
        ws.append(w.copy())
    return np.array(ws)


def true_sam_grad(w, rho=RHO):
    """exact gradient of Phi(w)=f(w+eps*(w)): (I + d eps*/dw) grad f(w+eps*), d eps*/dw = rho/|g| (I - gh gh^T) H."""
    g = grad(w)
    n = np.linalg.norm(g)
    gh = g / n
    J = rho / n * (np.eye(2) - np.outer(gh, gh)) @ H
    return (np.eye(2) + J.T) @ grad(w + eps_star(w, rho))


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


def worst_perturbation(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    W, Hh = 560, 480
    fig = plt.figure(figsize=(W / 100, Hh / 100))
    ax = fig.add_axes([70 / W, 62 / Hh, (W - 70 - 20) / W, (Hh - 62 - 24) / Hh])
    xs = np.linspace(9.2, 10.9, 200)
    ys = np.linspace(1.3, 2.75, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.5 * (X ** 2 + 25 * Y ** 2)
    cs = ax.contour(X, Y, Z, levels=[90, 100, 110, 120, 130, 140], colors=c["muted"], linewidths=1.2, alpha=0.7)
    ax.clabel(cs, fmt="%d", fontsize=12, colors=c["muted"])
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(W0[0] + RHO * np.cos(th), W0[1] + RHO * np.sin(th), color=c["ref"], lw=2.4)
    es = eps_star(W0)
    ex, _ = exact_max(W0)
    ax.annotate("", xy=(W0[0] + es[0], W0[1] + es[1]), xytext=(W0[0], W0[1]), arrowprops=dict(arrowstyle="-|>", color=c["main"], lw=2.8, shrinkA=0, shrinkB=0))
    ax.plot([W0[0]], [W0[1]], "o", color=c["text"], ms=8)
    ax.annotate(r"$w$  (L=100)", (W0[0], W0[1]), xytext=(14, -26), textcoords="offset points", ha="left", color=c["text"], fontsize=15)
    ax.plot([W0[0] + es[0]], [W0[1] + es[1]], "o", color=c["main"], ms=9)
    ax.annotate(r"$w+\epsilon^{\ast}$  (L=%.2f)" % f(W0 + es), (W0[0] + es[0], W0[1] + es[1]), xytext=(14, 4), textcoords="offset points", color=c["main"], fontsize=15)
    ax.plot([W0[0] + ex[0]], [W0[1] + ex[1]], "x", color=c["ok"], ms=11, mew=2.6)
    ax.annotate("exact max %.2f" % f(W0 + ex), (W0[0] + ex[0], W0[1] + ex[1]), xytext=(14, -22), textcoords="offset points", color=c["ok"], fontsize=15)
    ax.set_aspect("equal")
    ax.set_xlim(9.2, 10.9)
    ax.set_ylim(1.3, 2.75)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    clean(ax)
    finish(fig, "lec03-kp13-worst-perturbation", dark, c)
    return W, Hh


def worst_loss(dark):
    c = DARK if dark else LIGHT
    skin(c, dark)
    wg, wsam = run("gd"), run("sam")
    lg = np.array([worst(w) for w in wg])
    ls = np.array([worst(w) for w in wsam])
    W, Hh = 640, 380
    fig = plt.figure(figsize=(W / 100, Hh / 100))
    ax = fig.add_axes([90 / W, 66 / Hh, (W - 90 - 24) / W, (Hh - 66 - 28) / Hh])
    k = np.arange(T + 1)
    ax.semilogy(k, lg, color=c["ref"], lw=2.6, label="plain GD")
    ax.semilogy(k, ls, color=c["main"], lw=2.6, label="vanilla SAM")
    ax.axhline(0.5 * D[1] * RHO ** 2, color=c["muted"], ls="--", lw=1.8)
    ax.annotate("floor %.3f" % (0.5 * D[1] * RHO ** 2), (2, 0.5 * D[1] * RHO ** 2), xytext=(0, 8), textcoords="offset points", ha="left", color=c["muted"], fontsize=14)
    ax.annotate("%.2f" % ls[-1], (T, ls[-1]), xytext=(-6, 10), textcoords="offset points", ha="right", color=c["main"], fontsize=14)
    ax.annotate("%.2f" % lg[-1], (T, lg[-1]), xytext=(-6, 8), textcoords="offset points", ha="right", color=c["ref"], fontsize=14)
    ax.set_xlabel(r"step $k$")
    ax.set_ylabel(r"worst loss in the $\rho$-ball")
    ax.set_xlim(0, T)
    ax.set_ylim(2.5, 160)
    ax.legend(loc="upper right", frameon=False, fontsize=14)
    clean(ax)
    finish(fig, "lec03-kp13-worst-loss", dark, c)
    return W, Hh


def pnorm_eps(g, rho, p):
    q = p / (p - 1)
    return rho * np.sign(g) * np.abs(g) ** (q - 1) / (np.sum(np.abs(g) ** q) ** (1 / p))


if __name__ == "__main__":
    for dark in (False, True):
        worst_perturbation(dark)
        worst_loss(dark)
    np.set_printoptions(suppress=True)
    r = lambda a, n=4: np.round(a, n).tolist()
    print(f"bowl f=0.5(x^2+25y^2), start {W0.tolist()}, rho={RHO}, eta={ETA}, steps={T}")

    # ---- page 1/2: start point
    g = grad(W0)
    ng = np.linalg.norm(g)
    es = eps_star(W0)
    ex, fmax = exact_max(W0)
    print("g =", g.tolist(), " |g| =", round(ng, 4), " g/|g| =", r(g / ng), " eps* =", r(es), " |eps*| =", round(float(np.linalg.norm(es)), 6))
    print("L(w) =", f(W0), " first-order estimate L+rho|g| =", round(f(W0) + RHO * ng, 4), " sharpness estimate rho|g| =", round(RHO * ng, 4))
    print("L(w+eps*) =", round(f(W0 + es), 4), " exact max =", round(fmax, 4), " exact sharpness =", round(fmax - f(W0), 4),
          " gap exact - L(w+eps*) =", round(fmax - f(W0 + es), 4), " first-order underestimate =", round(fmax - f(W0) - RHO * ng, 4))
    print("exact maximizer eps =", r(ex), " |eps| =", round(float(np.linalg.norm(ex)), 6),
          " angle between eps* and exact (deg) =", round(math.degrees(math.acos(float(es @ ex) / (np.linalg.norm(es) * np.linalg.norm(ex)))), 3))
    print("second-order term 0.5 eps*^T H eps* =", round(0.5 * float(es @ H @ es), 4), " (= L(w+eps*) - L(w) - eps*^T g =", round(f(W0 + es) - f(W0) - float(es @ g), 4), ")")
    # Cauchy-Schwarz check by random directions on the circle
    rng = np.random.default_rng(0)
    ang = rng.uniform(0, 2 * np.pi, 200000)
    dirs = RHO * np.stack([np.cos(ang), np.sin(ang)], 1)
    print("Cauchy-Schwarz: max over 200000 random eps with |eps|=rho of eps.g =", round(float((dirs @ g).max()), 4), " <= rho|g| =", round(RHO * ng, 4),
          " | actual max of L over 20000 of the same samples =", round(max(f(W0 + e) for e in dirs[:20000]), 4))
    print("p-norm check (g=(10,50)): p=2", r(pnorm_eps(g, RHO, 2)), "|eps|_2 =", round(float(np.linalg.norm(pnorm_eps(g, RHO, 2))), 4),
          "| p=3", r(pnorm_eps(g, RHO, 3)), " |eps|_3 =", round(float(np.sum(np.abs(pnorm_eps(g, RHO, 3)) ** 3) ** (1 / 3)), 4),
          " eps.g =", round(float(pnorm_eps(g, RHO, 3) @ g), 4), " rho|g|_q =", round(RHO * float(np.sum(np.abs(g) ** 1.5) ** (1 / 1.5)), 4),
          "| p=inf eps = rho sign(g) =", (RHO * np.sign(g)).tolist(), " eps.g =", float(RHO * np.sign(g) @ g), " rho|g|_1 =", RHO * float(np.abs(g).sum()))

    # ---- rho sweep: how good is the first-order picture
    print("rho sweep at the start (L=100): rho | first-order L+rho|g| | L(w+eps*) | exact max | exact - L(w+eps*)")
    for rho in (0.1, 0.5, 1.0, 2.0, 5.0):
        e = eps_star(W0, rho)
        fm = worst(W0, rho)
        print("  ", rho, round(f(W0) + rho * ng, 3), round(f(W0 + e), 3), round(fm, 3), round(fm - f(W0 + e), 4))

    # ---- page 3: one step of GD and SAM
    w_adv = W0 + es
    g_adv = grad(w_adv)
    w_sam = W0 - ETA * g_adv
    w_gd = W0 - ETA * g
    print("w+eps* =", r(w_adv), " grad at w+eps* =", r(g_adv), " |.| =", round(float(np.linalg.norm(g_adv)), 4))
    print("GD step:  w1 =", r(w_gd), " L(w1) =", round(f(w_gd), 4), " worst(w1) =", round(worst(w_gd), 4))
    print("SAM step: w1 =", r(w_sam), " L(w1) =", round(f(w_sam), 4), " worst(w1) =", round(worst(w_sam), 4))
    print("SAM minus GD gradient =", r(g_adv - g), " ratio of |g_adv|/|g| =", round(float(np.linalg.norm(g_adv) / ng), 4))
    # the dropped term
    tg = true_sam_grad(W0)
    h = 1e-5
    fd = np.array([(f(W0 + h * np.eye(2)[i] + eps_star(W0 + h * np.eye(2)[i])) - f(W0 - h * np.eye(2)[i] + eps_star(W0 - h * np.eye(2)[i]))) / (2 * h) for i in range(2)])
    print("true gradient of Phi(w)=L(w+eps*(w)) at start:", r(tg), " finite difference:", r(fd), " SAM approx:", r(g_adv),
          " rel. diff |true-approx|/|true| =", round(float(np.linalg.norm(tg - g_adv) / np.linalg.norm(tg)), 4),
          " cosine =", round(float(tg @ g_adv / (np.linalg.norm(tg) * np.linalg.norm(g_adv))), 5))
    wg, wsam = run("gd"), run("sam")
    print("path (k: GD w, L, worst | SAM w, L, worst):")
    for k in (0, 1, 2, 3, 5, 10, 20, 30, 50, 60, 100):
        print("  ", k, r(wg[k], 3), round(f(wg[k]), 3), round(worst(wg[k]), 3), "|", r(wsam[k], 3), round(f(wsam[k]), 3), round(worst(wsam[k]), 3))
    wl_g = np.array([worst(w) for w in wg])
    wl_s = np.array([worst(w) for w in wsam])
    cross = [k for k in range(2, T + 1) if wl_s[k] > wl_g[k]]
    print("SAM worst < GD worst for k in 2..%d except from first crossover k =" % T, cross[0] if cross else None, "; min SAM worst =", round(float(wl_s.min()), 4), "at k =", int(wl_s.argmin()),
          "; ratio worst GD/SAM at k=30,60:", round(float(wl_g[30] / wl_s[30]), 3), round(float(wl_g[60] / wl_s[60]), 3))
    xs_ = wsam[:, 0]
    k_unst = int(np.argmax(xs_ < 6.25 / 1.5))
    print("y-mode factor of SAM near y=0: y' = y (1 - 25 eta (1 + rho*25/(25 x)... )) = y (0.5 - 6.25/x); unstable when x < 6.25/1.5 =", round(6.25 / 1.5, 3),
          "first k with x below that:", k_unst, " x there:", round(float(xs_[k_unst]), 3),
          " |y| at k=10,20,40,60,80:", [float(f"{abs(wsam[k][1]):.3g}") for k in (10, 20, 40, 60, 80)],
          " factor at x(k=60), x(k=80):", round(0.5 - 6.25 / float(xs_[60]), 3), round(0.5 - 6.25 / float(xs_[80]), 3))
    print("|y| of SAM path k=80..90:", [round(abs(float(wsam[k][1])), 6) for k in range(80, 91)])
    for k in (30, 100):
        tgk = true_sam_grad(wsam[k])
        ak = grad(wsam[k] + eps_star(wsam[k]))
        print(f"  dropped-term check on SAM path k={k}: true", r(tgk), "approx", r(ak), "rel diff", round(float(np.linalg.norm(tgk - ak) / np.linalg.norm(tgk)), 4))
    wlong = run("sam", 400)
    wglong = run("gd", 400)
    print("k=400: SAM w =", r(wlong[400], 4), " L =", round(f(wlong[400]), 4), " worst =", round(worst(wlong[400]), 4), "| GD L =", f"{f(wglong[400]):.2e}", " worst =", round(worst(wglong[400]), 4))
    print("SAM last 4 y values:", [round(float(wlong[k][1]), 4) for k in (397, 398, 399, 400)],
          " predicted 2-cycle |y| = 25*eta*rho/(2-25*eta) =", round(25 * ETA * RHO / (2 - 25 * ETA), 4))
    print("worst-loss floor at minimum 0.5*25*rho^2 =", 0.5 * 25 * RHO ** 2, "; first-order sharpness estimate at w=0 is rho|g| = 0 while exact sharpness =", worst(np.zeros(2)))
    # ---- worked example (lecture): w=(4,1)
    w = np.array([4.0, 1.0])
    g = grad(w)
    ng = np.linalg.norm(g)
    e = eps_star(w)
    wa = w + e
    ga = grad(wa)
    print("WORKED w=(4,1): g =", g.tolist(), " |g| =", round(ng, 4), " eps* =", r(e), " w+eps* =", r(wa), " grad there =", r(ga),
          " SAM w1 =", r(w - ETA * ga), " GD w1 =", r(w - ETA * g), " L(w) =", f(w), " L(w+eps*) =", round(f(wa), 4),
          " first-order L+rho|g| =", round(f(w) + RHO * ng, 4), " exact max =", round(worst(w), 4),
          " L(SAM w1) =", round(f(w - ETA * ga), 4), " L(GD w1) =", round(f(w - ETA * g), 4),
          " worst(SAM w1) =", round(worst(w - ETA * ga), 4), " worst(GD w1) =", round(worst(w - ETA * g), 4))
    # ---- practice answers
    g3 = np.array([3.0, 4.0])
    print("PRACTICE 1: g=(3,4), rho=0.5: eps* =", r(RHO * g3 / np.linalg.norm(g3)), " rho|g| =", RHO * np.linalg.norm(g3))
    k_, w_, rho_ = 20.0, 1.0, 0.3
    print("PRACTICE 2: 1-D k=20, w=1, rho=0.3: eps* =", rho_, " SAM grad = k(w+eps*) =", k_ * (w_ + rho_), " plain grad =", k_ * w_,
          " | k=0.6: SAM grad =", 0.6 * (w_ + rho_), " plain =", 0.6 * w_, " | exact max of f(w+e)-f(w) = k(w*rho+rho^2/2) =", k_ * (w_ * rho_ + rho_ ** 2 / 2), " first-order k w rho =", k_ * w_ * rho_)
    gi = np.array([3.0, -4.0])
    ei = RHO * np.sign(gi)
    print("PRACTICE 3: g=(3,-4), rho=0.5, p=inf: eps* =", ei.tolist(), " eps.g =", float(ei @ gi), " p=2: eps.g =", RHO * np.linalg.norm(gi), " |eps*|_2 for p=inf =", round(float(np.linalg.norm(ei)), 4))
