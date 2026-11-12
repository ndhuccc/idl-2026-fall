"""lec08-kp05-figures.py — 第8週 KP05 講義的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec08-kp05-figures.py [輸出資料夾]（預設：與講義同層的 ../assets/figures/）"""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams["mathtext.fontset"] = "cm"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent  # 課程 repo：assets/figures/src/ 的上一層 assets/figures/
OUT.mkdir(parents=True, exist_ok=True)
r4 = lambda v: round(float(v), 4)

# ---------- 1. eigenvalue view: lambda^15 ----------
K = 15
powers = {str(l): r4(l ** K) for l in (0.75, 0.98, 1.0, 1.25)}

# ---------- 2. W + I : a 2x2 example ----------
W2 = np.array([[-0.10, 0.08], [0.08, -0.10]])
ev_W = np.sort(np.linalg.eigvalsh(W2))[::-1]
ev_WI = np.sort(np.linalg.eigvalsh(W2 + np.eye(2)))[::-1]
WI15 = np.linalg.matrix_power(W2 + np.eye(2), K)
W15 = np.linalg.matrix_power(W2, K)
bad = dict(lam_W=0.5, lam_WI=1.5, pow10=r4(1.5 ** 10))          # practice 3
identity = dict(W=W2.tolist(), eig_W=[r4(v) for v in ev_W], eig_WI=[r4(v) for v in ev_WI],
                eig_WI_pow15=[r4(v ** K) for v in ev_WI],
                normWI15=r4(np.linalg.norm(WI15, 2)), normW15=float(f"{np.linalg.norm(W15, 2):.3e}"),
                params_W_p4=16, extra_params_I=0, practice3=bad)

# ---------- 3. long delays ----------
lam = 0.75
delay = dict(lam=lam, T=K, hops_standard=K, hops_delay3=K // 3,
             standard=r4(lam ** K), delay3=r4(lam ** (K // 3)),
             ratio=round(lam ** (K // 3) / lam ** K, 2), identity098=r4(0.98 ** K),
             extra_params_W3_p4=16,
             practice1=dict(lam=0.8, T=12, standard=r4(0.8 ** 12), delay3=r4(0.8 ** 4)))

# ---------- 4. leaky units ----------
taus = [1, 2, 5, 10]
def decay(tau, n):   # zero candidate, h0 = 1
    return [(1 - 1 / tau) ** t for t in range(n + 1)]
def step(tau, n):    # candidate = 1 from t = 1, h0 = 0
    return [1 - (1 - 1 / tau) ** t for t in range(n + 1)]
leaky = dict(
    keep={str(t): r4(1 - 1 / t) for t in taus},
    decay_tau2=[r4(v) for v in decay(2, 2)],
    decay_tau10=[r4(v) for v in decay(10, 2)], decay_tau10_h10=r4(decay(10, 10)[10]),
    halflife_exact={str(t): r4(math.log(0.5) / math.log(1 - 1 / t)) for t in (2, 5, 10)},
    halflife_approx_tau_ln2={str(t): r4(t * math.log(2)) for t in (2, 5, 10)},
    step_tau10=dict(h1=r4(step(10, 1)[1]), h2=r4(step(10, 2)[2]), h10=r4(step(10, 10)[10])),
    step_tau1_h1=r4(step(1, 1)[1]),
    grad_path_tau10_15=r4(0.9 ** K),
    practice2=dict(tau=5, h1=r4(0.8), h2=r4(0.8 ** 2), h3=r4(0.8 ** 3), h4=r4(0.8 ** 4),
                   halflife=r4(math.log(0.5) / math.log(0.8))))

# ---------- 5. echo state network ----------
rng = np.random.default_rng(0)
N, density, rho_target = 100, 0.1, 0.9
Wres = rng.uniform(-1, 1, (N, N)) * (rng.random((N, N)) < density)
eig_raw = np.linalg.eigvals(Wres)
Wres *= rho_target / np.max(np.abs(eig_raw))
eig = np.linalg.eigvals(Wres)
Uin = rng.uniform(-0.5, 0.5, N)
Tlen, washout = 2200, 200
x = rng.uniform(-1, 1, Tlen)
H = np.zeros((Tlen, N)); h = np.zeros(N)
for t in range(Tlen):
    h = np.tanh(Wres @ h + Uin * x[t]); H[t] = h
# echo state property: two different initial states, same input
ha, hb = rng.uniform(-1, 1, N), rng.uniform(-1, 1, N)
gap = []
for t in range(60):
    ha = np.tanh(Wres @ ha + Uin * x[t]); hb = np.tanh(Wres @ hb + Uin * x[t])
    if t + 1 in (1, 10, 30, 60):
        gap.append((t + 1, float(f"{np.linalg.norm(ha - hb):.3e}")))
delays = [1, 2, 5, 10, 15, 20, 30]
Ntr = 1500
r2 = {}
for k in delays:
    idx = np.arange(washout, Tlen)
    Xf = np.hstack([H[idx], np.ones((len(idx), 1))]); yf = x[idx - k]
    Xtr, ytr, Xte, yte = Xf[:Ntr], yf[:Ntr], Xf[Ntr:], yf[Ntr:]
    lam_r = 1e-6
    Vb = np.linalg.solve(Xtr.T @ Xtr + lam_r * np.eye(N + 1), Xtr.T @ ytr)
    pred = Xte @ Vb
    r2[k] = 1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2)
nnz = int(np.count_nonzero(Wres))
esn = dict(N=N, density=density, nonzero_Wres=nnz, rho=r4(np.max(np.abs(eig))),
           trained_params=N + 1, fixed_params_Wres=nnz, fixed_params_U=N,
           train_steps=Ntr, test_steps=Tlen - washout - Ntr, washout=washout,
           init_gap=gap, recall_R2={str(k): r4(v) for k, v in r2.items()})

source_check = dict(p08_20=r4(0.8 ** 20), p09_20=r4(0.9 ** 20))
num = dict(source_check=source_check, eigen_powers_k15=powers, identity=identity, delay=delay, leaky=leaky, esn=esn)
print(json.dumps(num, indent=1, ensure_ascii=False))

# ---------- figure 1: retention over k steps ----------
ks = np.arange(0, 16)
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.semilogy(ks, 0.75 ** ks, "o-", color="#DC2626", lw=2, ms=4, label=r"standard RNN: $0.75^{k}$")
kd = np.arange(0, 16, 3)
ax.semilogy(kd, 0.75 ** (kd // 3), "s-", color="#2563EB", lw=2, ms=6, label=r"3-step delay $\mathbf{W}_3$: $0.75^{k/3}$")
ax.semilogy(ks, 0.98 ** ks, "^-", color="#16A34A", lw=2, ms=5, label=r"close to identity: $0.98^{k}$")
for yv, c, dy in ((0.75 ** 15, "#DC2626", 1.0), (0.75 ** 5, "#2563EB", 1.0), (0.98 ** 15, "#16A34A", 1.0)):
    ax.annotate(f"{yv:.4f}", (15, yv), xytext=(15.25, yv * dy), va="center", fontsize=12, color=c)
ax.set_xlim(0, 17.2); ax.set_ylim(0.008, 1.5)
ax.set_xlabel(r"steps back in time $k$", fontsize=12)
ax.set_ylabel("gradient factor left (log scale)", fontsize=12)
ax.set_xticks(range(0, 16, 3))
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=11, loc="lower left", frameon=False)
fig.tight_layout(); fig.savefig(OUT / "lec08-kp05-retention.png", dpi=200); plt.close(fig)

# ---------- figure 2: leaky units ----------
cols = {1: "#DC2626", 2: "#EA580C", 5: "#2563EB", 10: "#16A34A"}
n = 20; tt = np.arange(n + 1)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for tau in taus:
    axs[0].plot(tt, decay(tau, n), "o-", ms=3, lw=2, color=cols[tau], label=rf"$\tau={tau}$")
    axs[1].plot(tt, step(tau, n), "o-", ms=3, lw=2, color=cols[tau], label=rf"$\tau={tau}$")
axs[0].axhline(0.5, color="#64748B", ls="--", lw=1); axs[0].text(19.8, 0.52, "half", ha="right", color="#64748B", fontsize=11)
axs[0].annotate(r"$0.9^{10}=%.4f$" % (0.9 ** 10), (10, 0.9 ** 10), xytext=(11.5, 0.6), fontsize=12,
                arrowprops=dict(arrowstyle="->", color="#16A34A"), color="#16A34A")
axs[0].set_title(r"(a) remembering: candidate $=0$, $h_0=1$", fontsize=13)
axs[1].annotate(r"$1-0.9^{10}=%.4f$" % (1 - 0.9 ** 10), (10, 1 - 0.9 ** 10), xytext=(4.5, 0.15), fontsize=12,
                arrowprops=dict(arrowstyle="->", color="#16A34A"), color="#16A34A")
axs[0].legend(fontsize=11, frameon=False, loc="upper right"); axs[1].legend(fontsize=11, frameon=False, loc="lower right")
axs[1].set_title(r"(b) reacting: candidate $=1$, $h_0=0$", fontsize=13)
for a in axs:
    a.set_xlabel(r"time step $t$", fontsize=12); a.set_ylabel(r"state $h_t$", fontsize=12)
    a.set_ylim(-0.05, 1.08); a.set_xlim(0, n); a.grid(alpha=0.3); pass
fig.tight_layout(); fig.savefig(OUT / "lec08-kp05-leaky.png", dpi=200); plt.close(fig)

# ---------- figure 3: ESN ----------
fig = plt.figure(figsize=(15, 4.8))
ax = fig.add_axes([0.01, 0.06, 0.36, 0.88]); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.add_patch(Circle((5, 3.1), 2.2, fc="#F1F5F9", ec="#64748B", lw=1.5, ls="--"))
pts = np.random.default_rng(3).uniform(-1, 1, (40, 2)); pts = pts[np.hypot(*pts.T) < 0.85][:14] * 2.2 + [5, 3.1]
for i in range(len(pts)):
    for j in np.random.default_rng(i).choice(len(pts), 2, replace=False):
        if j != i:
            ax.annotate("", xy=pts[j], xytext=pts[i], arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=0.8))
ax.scatter(*pts.T, s=60, color="#64748B", zorder=3)
ax.text(5, 0.45, r"reservoir $\mathbf{W}_{\mathrm{res}}$: random, sparse, FIXED", ha="center", fontsize=12, color="#334155")
ax.add_patch(FancyBboxPatch((0.2, 2.75), 1.2, 0.7, boxstyle="round,pad=0.05", fc="#DCFCE7", ec="#334155"))
ax.text(0.8, 3.1, r"$\mathbf{x}_t$", ha="center", va="center", fontsize=15)
ax.add_patch(FancyArrowPatch((1.45, 3.1), (2.85, 3.1), arrowstyle="-|>", mutation_scale=14, lw=1.6, color="#64748B"))
ax.text(2.1, 3.4, r"$\mathbf{U}$ fixed", ha="center", fontsize=11, color="#64748B")
ax.add_patch(FancyBboxPatch((8.6, 2.75), 1.2, 0.7, boxstyle="round,pad=0.05", fc="#FEF3C7", ec="#334155"))
ax.text(9.2, 3.1, r"$\mathbf{y}_t$", ha="center", va="center", fontsize=15)
ax.add_patch(FancyArrowPatch((7.15, 3.1), (8.55, 3.1), arrowstyle="-|>", mutation_scale=16, lw=3, color="#EA580C"))
ax.text(7.85, 3.55, r"$\mathbf{V}$ trained", ha="center", fontsize=12, color="#EA580C")
ax.text(5, 5.65, "(a) only the readout is trained", ha="center", fontsize=13)
ax2 = fig.add_axes([0.42, 0.22, 0.24, 0.66])
th = np.linspace(0, 2 * np.pi, 300)
ax2.plot(np.cos(th), np.sin(th), color="#334155", lw=1.2, label="unit circle")
ax2.plot(rho_target * np.cos(th), rho_target * np.sin(th), "--", color="#EA580C", lw=1.2, label=rf"$\rho={rho_target}$")
ax2.scatter(eig.real, eig.imag, s=12, color="#2563EB", label=r"eigenvalues of $\mathbf{W}_{\mathrm{res}}$")
ax2.set_aspect("equal"); ax2.set_xlim(-1.15, 1.15); ax2.set_ylim(-1.15, 1.15)
ax2.set_xlabel("real part", fontsize=11); ax2.set_ylabel("imaginary part", fontsize=11)
ax2.set_title(r"(b) spectral radius $\rho<1$", fontsize=13); ax2.grid(alpha=0.3)
ax2.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
ax3 = fig.add_axes([0.74, 0.14, 0.245, 0.74])
ax3.plot(delays, [r2[k] for k in delays], "o-", color="#2563EB", lw=2)
for k in (1, 10, 20):
    ax3.annotate(f"{r2[k]:.4f}", (k, r2[k]), xytext=(k + 1.2, r2[k] + 0.06), fontsize=11)
ax3.set_xlabel(r"delay $k$ (target $y_t=x_{t-k}$)", fontsize=11); ax3.set_ylabel(r"test $R^2$", fontsize=11)
ax3.set_ylim(-0.15, 1.12); ax3.set_xlim(0, 32); ax3.grid(alpha=0.3)
ax3.set_title("(c) the echo fades with delay", fontsize=13)
fig.savefig(OUT / "lec08-kp05-esn.png", dpi=200); plt.close(fig)
print("figures ->", OUT)
