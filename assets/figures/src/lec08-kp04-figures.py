"""lec08-kp04-figures.py — 第8週 KP04 講義的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec08-kp04-figures.py [輸出資料夾]（預設：與講義同層的 ../assets/figures/）"""
import json, math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["mathtext.fontset"] = "cm"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent  # 課程 repo：assets/figures/src/ 的上一層 assets/figures/
OUT.mkdir(parents=True, exist_ok=True)

# ---------- numbers ----------
# (1) gamma^k: the same factor multiplied k times
gammas = [0.8, 0.85, 1.0, 1.15, 1.2]
ks = [1, 5, 10, 20, 50]
power = {str(g): {str(k): g ** k for k in ks} for g in gammas}

# (2) tanh derivative 1 - tanh(i)^2
dtanh = {str(i): 1 - math.tanh(i) ** 2 for i in (0, 0.5, 1, 2, 3)}

# (3) scalar RNN with zero inputs after h_0: h_k = tanh(W h_{k-1}),
#     dh_K/dh_0 = prod_{k=1..K} W (1 - h_k^2); compare with W^K
K = 20
def chain(W, h0):
    h, prod, prods, hs, fac = h0, 1.0, [1.0], [h0], []
    for _ in range(K):
        h = math.tanh(W * h)
        f = W * (1 - h * h)
        prod *= f
        hs.append(h); fac.append(f); prods.append(prod)
    return dict(W=W, h0=h0, h=hs, factor=fac, prod=prods, W_pow_K=W ** K)
cases = [chain(0.9, 0.5), chain(1.5, 0.5), chain(1.5, 0.01)]

# (4) gradient clipping by norm: if ||g|| > c, g <- c g / ||g||
g = np.array([6.0, 8.0]); c = 5.0
gn = float(np.linalg.norm(g))
g_clip = g * (c / gn) if gn > c else g
clip = dict(g=g.tolist(), norm=gn, c=c, scale=c / gn, g_clip=g_clip.tolist(),
            norm_clip=float(np.linalg.norm(g_clip)), small=[0.3, 0.4], small_norm=0.5)

# (5) practice: gamma = W * tanh' = 0.5 * 0.8, three steps; 0.95^k thresholds
prac = dict(gamma=0.5 * 0.8, gamma3=(0.5 * 0.8) ** 3, p095_20=0.95 ** 20, p095_100=0.95 ** 100,
            p105_20=1.05 ** 20, k_half_085=math.log(0.5) / math.log(0.85))

num = dict(power=power, dtanh=dtanh, K=K,
           cases=[dict(W=cs["W"], h0=cs["h0"], h_last=cs["h"][-1], factor_first=cs["factor"][0],
                       factor_last=cs["factor"][-1], prod_last=cs["prod"][-1], prod_max=max(cs["prod"]),
                       k_at_max=int(np.argmax(cs["prod"])), W_pow_K=cs["W_pow_K"]) for cs in cases],
           clip=clip, practice=prac)
print(json.dumps(num, indent=1))

# ---------- figure 1: gamma^k on a log scale ----------
fig, ax = plt.subplots(figsize=(9, 5))
kk = np.arange(0, 51)
cols = {0.8: "#1D4ED8", 0.85: "#60A5FA", 1.0: "#334155", 1.15: "#F97316", 1.2: "#B91C1C"}
for gm in gammas:
    ax.semilogy(kk, gm ** kk, color=cols[gm], lw=2.2, ls="--" if gm == 1.0 else "-",
                label=rf"$\gamma={gm:g}$")
for gm in (0.8, 1.2):
    v = gm ** 20
    ax.plot([20], [v], "o", color=cols[gm], ms=7)
    ax.annotate(rf"$\gamma^{{20}}={v:.4f}$" if v < 1 else rf"$\gamma^{{20}}={v:.2f}$", (20, v),
                xytext=(-128, 10 if v > 1 else -26), textcoords="offset points", fontsize=12, color=cols[gm])
ax.axvline(20, color="#94A3B8", lw=1, ls=":")
ax.set_xlabel(r"number of steps $k=T-t$", fontsize=12)
ax.set_ylabel(r"$\gamma^k$ (log scale)", fontsize=12)
ax.set_ylim(1e-5, 2e4); ax.set_xlim(0, 50)
ax.text(28, 2e3, "exploding", ha="center", fontsize=13, color="#B91C1C")
ax.text(28, 3e-5, "vanishing", ha="center", fontsize=13, color="#1D4ED8")
ax.legend(loc="upper left", fontsize=11, frameon=False)
ax.grid(alpha=0.3, which="major")
fig.tight_layout(); fig.savefig(OUT / "lec08-kp04-power-decay.png", dpi=200); plt.close(fig)

# ---------- figure 2: tanh' <= 1 and real chains of a scalar RNN ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
ii = np.linspace(-4, 4, 400)
a1.plot(ii, 1 - np.tanh(ii) ** 2, color="#16A34A", lw=2.4)
a1.axhline(1, color="#334155", lw=1, ls="--")
for i0 in (0, 1, 2):
    v = 1 - math.tanh(i0) ** 2
    a1.plot([i0], [v], "o", color="#16A34A", ms=7)
    a1.annotate(rf"$i={i0}$: ${v:.4f}$", (i0, v), xytext=(12, 4), textcoords="offset points", fontsize=12)
a1.set_xlabel(r"net input $i$", fontsize=12); a1.set_ylabel(r"$\tanh'(i)=1-\tanh^2(i)$", fontsize=12)
a1.set_ylim(0, 1.15); a1.set_title(r"(a) $\tanh'$ is never above $1$", fontsize=13)
a1.grid(alpha=0.3)
kk = np.arange(0, K + 1)
ccol = ["#1D4ED8", "#B91C1C", "#F97316"]
labs = [r"$W=0.9,\ h_0=0.5$", r"$W=1.5,\ h_0=0.5$", r"$W=1.5,\ h_0=0.01$"]
for cs, cl, lb in zip(cases, ccol, labs):
    a2.semilogy(kk, cs["prod"], color=cl, lw=2.2, marker="o", ms=3.5, label=lb + r"  (actual)")
a2.semilogy(kk, 0.9 ** kk, color="#1D4ED8", lw=1.2, ls=":", label=r"$0.9^k$")
a2.semilogy(kk, 1.5 ** kk, color="#B91C1C", lw=1.2, ls=":", label=r"$1.5^k$")
a2.set_xlabel(r"step $k$", fontsize=12); a2.set_ylabel(r"$|\partial h_k/\partial h_0|$ (log scale)", fontsize=12)
a2.set_title(r"(b) scalar RNN, zero inputs: $\tanh'$ pulls the chain down", fontsize=13)
a2.set_ylim(1e-9, 1e4); a2.set_xlim(0, K)
a2.legend(loc="lower left", fontsize=10, frameon=False)
a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "lec08-kp04-tanh-factor.png", dpi=200); plt.close(fig)

# ---------- figure 3: clipping by norm ----------
fig, ax = plt.subplots(figsize=(6.4, 6))
th = np.linspace(0, 2 * np.pi, 300)
ax.plot(c * np.cos(th), c * np.sin(th), color="#64748B", lw=1.4, ls="--")
ax.text(-5.7, 5.4, rf"dashed circle: $\|\mathbf{{g}}\|=c={c:g}$", fontsize=12, color="#64748B")
ax.annotate("", xy=tuple(g), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#B91C1C", mutation_scale=18))
ax.annotate("", xy=tuple(g_clip), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=3.2, color="#16A34A", mutation_scale=18))
ax.text(g[0] + 0.2, g[1] - 0.1, rf"$\mathbf{{g}}=({g[0]:g},{g[1]:g})$" + "\n" + rf"$\|\mathbf{{g}}\|={gn:g}$", fontsize=12, color="#B91C1C")
ax.text(g_clip[0] + 0.9, g_clip[1] - 0.9, rf"clipped $({g_clip[0]:g},{g_clip[1]:g})$" + "\n" + rf"length ${c:g}$",
        fontsize=12, color="#16A34A")
ax.annotate("", xy=(0.3, 0.4), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=2, color="#1D4ED8", mutation_scale=14))
ax.text(0.4, -0.9, r"$(0.3,0.4)$: kept", fontsize=11, color="#1D4ED8")
ax.set_xlim(-6, 10); ax.set_ylim(-6, 10); ax.set_aspect("equal")
ax.axhline(0, color="#CBD5E1", lw=1); ax.axvline(0, color="#CBD5E1", lw=1)
ax.set_title("same direction, length cut to the threshold", fontsize=13)
ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "lec08-kp04-clipping.png", dpi=200); plt.close(fig)
print("figures ->", OUT)
