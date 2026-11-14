"""lec13-kp04-figures.py — 第13週 KP04 講義的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec13-kp04-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
來源的玩具賽局 min_x max_y xy；[補充] 同時更新與交替更新的比較、迭代平均，以及 KP01 一維小 GAN 的寬度擺盪。"""
import json, sys, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"
args = sys.argv[1:]
REPORT = None
if "--report" in args:
    i = args.index("--report"); REPORT = Path(args[i + 1]); del args[i:i + 2]
OUT = Path(args[0]) if args else Path(__file__).resolve().parent.parent
OUT.mkdir(parents=True, exist_ok=True)
num = {}
sig = lambda t: 1.0 / (1.0 + np.exp(-t))
ETA, T = 0.15, 100


def simultaneous(eta=ETA, steps=T, p=(1.0, 0.0)):
    x, y = p; out = [(x, y)]
    for _ in range(steps):
        x, y = x - eta * y, y + eta * x          # x descends on xy (grad y), y ascends (grad x), both from the old point
        out.append((x, y))
    return np.array(out)


def alternating(eta=ETA, steps=T, p=(1.0, 0.0)):
    x, y = p; out = [(x, y)]
    for _ in range(steps):
        x = x - eta * y                           # x moves first
        y = y + eta * x                           # y responds to the new x
        out.append((x, y))
    return np.array(out)


S, A = simultaneous(), alternating()
r = lambda P: np.hypot(P[:, 0], P[:, 1])
M = np.array([[1, -ETA], [ETA, 1 - ETA ** 2]]); ev = np.linalg.eigvals(M)
num["toy_game"] = dict(game="min_x max_y V=xy", start=[1.0, 0.0], eta=ETA, steps=T,
                       simultaneous_radius={str(t): round(float(r(S)[t]), 3) for t in (0, 10, 50, 100)},
                       simultaneous_factor_per_step=round(math.sqrt(1 + ETA ** 2), 4),
                       alternating_radius_range=[round(float(r(A).min()), 3), round(float(r(A).max()), 3)],
                       alternating_radius_at={str(t): round(float(r(A)[t]), 3) for t in (0, 10, 50, 100)},
                       alternating_eigen_modulus=[round(float(abs(e)), 6) for e in ev],
                       alternating_steps_per_turn=round(float(2 * math.pi / abs(np.angle(ev[0]))), 1),
                       alternating_mean_of_iterates=[round(float(v), 3) for v in A.mean(0)],
                       simultaneous_after_100_other_eta={str(e): round((1 + e * e) ** 50, 3) for e in (0.05, 0.3)})
assert abs(r(S)[100] - (1 + ETA ** 2) ** 50) < 1e-9 and np.allclose(abs(ev), 1)
A_long = alternating(steps=4000)
num["toy_game"]["alternating_mean_of_4000_iterates"] = [round(float(v), 4) for v in A_long.mean(0)]

# ---------- KP01 toy GAN: width oscillation (same settings as KP01) ----------
def train(steps=3000, lrD=0.1, lrG=0.005, n=256, k=10, seed=0):
    rng = np.random.default_rng(seed); a, b, w = 0.0, 1.0, np.zeros(3)
    F = lambda x: np.stack([np.ones_like(x), x, x * x], 1); out = []
    for t in range(steps + 1):
        out.append((t, a, abs(b)))
        if t == steps:
            break
        for _ in range(k):
            xr = rng.normal(2, 0.5, n); xf = a + b * rng.normal(size=n)
            w = w + lrD * (F(xr).T @ (1 - sig(F(xr) @ w)) / n - F(xf).T @ sig(F(xf) @ w) / n)
        z = rng.normal(size=n); xf = a + b * z; d = sig(F(xf) @ w); gx = (1 - d) * (w[1] + 2 * w[2] * xf)
        a += lrG * np.mean(gx); b += lrG * np.mean(gx * z)
    return np.array(out)


G = train()
bb = G[:, 2]
i1 = int(np.argmin(bb[:800])); i2 = int(np.argmax(bb[i1:2000]) + i1); i3 = int(np.argmin(bb[i2:2500]) + i2)
num["kp01_gan_width"] = dict(settings="as KP01: k=10, lr 0.1/0.005, batch 256, seed 0",
                             first_min=dict(step=i1, sd=round(float(bb[i1]), 3)), next_max=dict(step=i2, sd=round(float(bb[i2]), 3)),
                             next_min=dict(step=i3, sd=round(float(bb[i3]), 3)), sd_at_3000=round(float(bb[-1]), 3))
assert i1 == 523
G1 = train(k=1)
num["kp01_gan_width_k1"] = dict(settings="same but k=1 discriminator update per step", sd_at_3000=round(float(G1[-1, 2]), 3))
num["kp01_gan_width_last500"] = dict(min=round(float(bb[-500:].min()), 3), max=round(float(bb[-500:].max()), 3))

# ---------- figure 1: lec13-kp04-vector-field ----------
g = np.linspace(-1.6, 1.6, 15); X, Y = np.meshgrid(g, g)
fig, ax = plt.subplots(figsize=(5.4, 5.0))
cs = ax.contourf(X, Y, X * Y, levels=np.linspace(-2.6, 2.6, 27), cmap="RdBu_r", alpha=0.55)
ax.quiver(X, Y, -Y, X, color="#334155", width=0.004, scale=28)
ax.plot([0], [0], marker="*", ms=16, color="#B45309"); ax.text(0.08, -0.22, "Nash\nequilibrium", color="#B45309", fontsize=9.5)
fig.colorbar(cs, ax=ax, label=r"$V(x,y)=xy$")
ax.set_xlabel(r"$x$ (minimiser, like $G$)"); ax.set_ylabel(r"$y$ (maximiser, like $D$)")
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")
ax.set_title(r"update direction $(-\partial V/\partial x,\ +\partial V/\partial y)=(-y,\ x)$", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "lec13-kp04-vector-field.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec13-kp04-trajectories ----------
th = np.linspace(0, 2 * np.pi, 400)
fig, axs = plt.subplots(1, 3, figsize=(12.5, 5.0), sharex=True, sharey=True)
for ax, P, ttl, col in zip(axs, (None, S, A), ("continuous time", rf"simultaneous, $\eta={ETA}$", rf"alternating, $\eta={ETA}$"),
                          ("#0F766E", "#BE123C", "#1D4ED8")):
    ax.plot(np.cos(th), np.sin(th), color="#CBD5E1", lw=1, ls=":")
    if P is None:
        ax.plot(np.cos(th), np.sin(th), color=col, lw=2.2)
        ax.text(0, -2.9, r"radius stays $1$", ha="center", fontsize=10, color=col)
    else:
        ax.plot(P[:, 0], P[:, 1], "-", color=col, lw=1.2); ax.plot(P[:, 0], P[:, 1], ".", color=col, ms=3)
        ax.plot(P[-1, 0], P[-1, 1], "o", color=col, ms=6)
        rr = float(np.hypot(*P[-1]))
        ax.text(0, -2.9, rf"after {T} steps: radius ${rr:.2f}$", ha="center", fontsize=10, color=col)
    ax.plot([0], [0], marker="*", ms=13, color="#B45309"); ax.plot([1], [0], "s", color="black", ms=5)
    ax.set_title(ttl, fontsize=11); ax.set_aspect("equal"); ax.set_xlim(-3.3, 3.3); ax.set_ylim(-3.3, 3.3); ax.grid(alpha=0.25)
    ax.set_xlabel(r"$x$")
axs[0].set_ylabel(r"$y$")
fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.12, wspace=0.08); fig.savefig(OUT / "lec13-kp04-trajectories.png", dpi=200); plt.close(fig)

# ---------- figure 3: lec13-kp04-gan-oscillation ----------
fig, ax = plt.subplots(figsize=(7.6, 3.6))
ax.plot(G[:, 0], G[:, 2], color="#B45309", lw=2.2, label=r"generator width $|b|$")
ax.axhline(0.5, color="#1D4ED8", ls="--", lw=1.4); ax.text(2990, 0.53, "real width = 0.5", color="#1D4ED8", ha="right", fontsize=9.5)
for st in (i1, i2, i3):
    ax.plot([st], [bb[st]], "o", color="#BE123C"); ax.text(st + 30, bb[st] + (0.04 if st != i1 else -0.08), rf"step {st}: ${bb[st]:.3f}$", fontsize=9)
ax.set_xlim(0, 3000); ax.set_ylim(0, 1.1); ax.set_xlabel("training step"); ax.set_ylabel(r"$|b|$"); ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "lec13-kp04-gan-oscillation.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
