"""lec13-kp07-figures.py — 第13週 KP07 講義的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec13-kp07-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
需要 numpy、matplotlib 與 PyTorch（CPU 版即可；約一分鐘）。
[補充] 沿用 KP03 的一維例子：真資料 N(2,0.5^2)，生成資料 N(a,1)。
(1) JS 散度與 Wasserstein 距離隨 a 變化，以及它們對 a 的斜率；(2) a=-3 時的最優判別器 D* 與最優評論家 f*；
(3) 用樣本訓練一個加梯度懲罰的評論家，估計 W 並和精確值比較；(4) 三種生成器損失對分數的梯度大小；(5) 推土小例子。"""
import json, sys, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn

plt.rcParams["mathtext.fontset"] = "cm"
args = sys.argv[1:]
REPORT = None
if "--report" in args:
    i = args.index("--report"); REPORT = Path(args[i + 1]); del args[i:i + 2]
OUT = Path(args[0]) if args else Path(__file__).resolve().parent.parent
OUT.mkdir(parents=True, exist_ok=True)
num = {}

x = np.linspace(-12, 14, 52001); dx = x[1] - x[0]
pdf = lambda m, s: np.exp(-(x - m) ** 2 / (2 * s * s)) / (s * math.sqrt(2 * math.pi))
cdf = lambda m, s: 0.5 * (1 + np.vectorize(math.erf)((x - m) / (s * math.sqrt(2))))
P = pdf(2, 0.5); FP = cdf(2, 0.5)
def jsd(a):
    Q = pdf(a, 1.0); M = 0.5 * (P + Q)
    t = lambda A: np.sum(np.where(A > 1e-300, A * np.log(np.maximum(A, 1e-300) / np.maximum(M, 1e-300)), 0)) * dx
    return float(0.5 * t(P) + 0.5 * t(Q))
def w1(a): return float(np.sum(np.abs(FP - cdf(a, 1.0))) * dx)

# ---------- (1) JSD and W as the generator centre a moves ----------
A = np.round(np.arange(-4, 6.001, 0.05), 2)
J = np.array([jsd(a) for a in A]); W = np.array([w1(a) for a in A])
dJ = np.gradient(J, A); dW = np.gradient(W, A)
pick = lambda arr, a: round(float(arr[np.argmin(np.abs(A - a))]), 4)
num["jsd_w"] = dict(setting="real N(2,0.5^2), generated N(a,1)",
                    table={str(a): dict(JSD=pick(J, a), W=pick(W, a), dJSD_da=pick(dJ, a), dW_da=pick(dW, a)) for a in (-3, -1, 0, 1, 1.5, 2, 3)},
                    W_min=dict(a=float(A[np.argmin(W)]), W=round(float(W.min()), 4)))
assert abs(pick(W, -3) - 5.0) < 0.01

# ---------- (2) optimal D* and optimal critic f* at a=-3 ----------
a0 = -3.0; Q0 = pdf(a0, 1.0); FQ0 = cdf(a0, 1.0)
Dstar = P / np.maximum(P + Q0, 1e-300)
slope = np.sign(FQ0 - FP)                       # f*' = sign(F_g - F_data) maximises E_data f - E_g f
fstar = np.cumsum(slope) * dx; fstar -= np.interp(0.0, x, fstar)
num["far_case"] = dict(a=a0, W_exact=round(w1(a0), 4), share_of_x_where_slope_is_plus1=round(float((slope[(x > -8) & (x < 6)] > 0).mean()), 4),
                       Dstar_at={str(v): float(f"{np.interp(v, x, Dstar):.3g}") for v in (-4, -3, -2, 0, 1, 2)},
                       dual_check=round(float(np.sum(fstar * P) * dx - np.sum(fstar * Q0) * dx), 4))

# ---------- (3) a learned critic with gradient penalty (two penalty weights) ----------
torch.set_num_threads(1)
def critic(lam, steps=3000, n=512):
    torch.manual_seed(0); g = torch.Generator().manual_seed(0)
    C = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(C.parameters(), 1e-3, betas=(0.5, 0.9))
    for t in range(steps):
        xr = 2 + 0.5 * torch.randn(n, 1, generator=g); xf = a0 + torch.randn(n, 1, generator=g)
        e = torch.rand(n, 1, generator=g); xh = (e * xr + (1 - e) * xf).requires_grad_(True)
        gr = torch.autograd.grad(C(xh).sum(), xh, create_graph=True)[0]
        loss = -(C(xr).mean() - C(xf).mean()) + lam * ((gr.abs() - 1) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        xr = 2 + 0.5 * torch.randn(20000, 1, generator=g); xf = a0 + torch.randn(20000, 1, generator=g)
        est = float(C(xr).mean() - C(xf).mean())
        xs = torch.linspace(-6, 5, 1101).view(-1, 1); fc = C(xs).squeeze(1).numpy()
    sl = np.diff(fc) / np.diff(xs.squeeze(1).numpy())
    return est, xs, fc, sl
crit = {lam: critic(lam) for lam in (10, 100)}
num["learned_critic"] = dict(net="MLP 1-64-64-1 ReLU", penalty="lam*(|f'(x_hat)|-1)^2 on random interpolates", steps=3000, batch=512, exact=round(w1(a0), 3),
                             **{f"lam{lam}": dict(estimate=round(v[0], 3), slope_range_on_minus6_to_5=[round(float(v[3].min()), 2), round(float(v[3].max()), 2)])
                                for lam, v in crit.items()})
num["learned_critic"]["uniform_slope_theory"] = {f"lam{lam}": dict(k=round(1 + 5 / (2 * lam), 4), estimate=round(5 * (1 + 5 / (2 * lam)), 3)) for lam in (10, 100)}
est, xs, fc, slopes = crit[100]

# ---------- (4) generator-gradient size versus the discriminator's raw score ----------
s = np.linspace(-6, 3, 901); sig = 1 / (1 + np.exp(-s))
num["score_grad"] = dict(note="magnitude of d(generator loss)/d(score) for one fake sample",
                         saturating_at={str(v): round(float(1 / (1 + math.exp(-v))), 4) for v in (-6, -3, 0)},
                         nonsaturating_at={str(v): round(float(1 - 1 / (1 + math.exp(-v))), 4) for v in (-6, -3, 0)},
                         lsgan_c1_at={str(v): round(abs(v - 1), 4) for v in (-6, -3, 0)})

# ---------- (5) moving earth: a tiny worked example ----------
num["earth_example"] = dict(piles_p={"0": 0.5, "1": 0.5}, piles_q={"3": 0.5, "4": 0.5}, W=0.5 * 3 + 0.5 * 3,
                            note="each half-unit of earth moves 3 to the right")

pts = np.linspace(-1, 6, 70001); Fp = 0.5 * (pts >= 0) + 0.5 * (pts >= 2); Fq = 0.5 * (pts >= 1) + 0.5 * (pts >= 5)
num["practice"] = dict(plan_a=0.5 * 1 + 0.5 * 3, plan_b=0.5 * 5 + 0.5 * 1, W_by_cdf=round(float(np.sum(np.abs(Fp - Fq)) * (pts[1] - pts[0])), 3),
                       critic_3x=3 * 2 - 3 * (-3), s_minus6=dict(saturating=round(1 / (1 + math.exp(6)), 4), nonsaturating=round(1 - 1 / (1 + math.exp(6)), 4), lsgan=7))

# ---------- figure 1: lec13-kp07-jsd-vs-w ----------
fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.2))
ax = axs[0]
ax.plot(A, J, color="#7C3AED", lw=2.4, label=r"$\mathrm{JSD}$")
ax.plot(A, W, color="#0F766E", lw=2.4, label=r"$W$ (earth mover)")
ax.axhline(math.log(2), color="#94A3B8", ls=":", lw=1.2); ax.text(4.2, math.log(2) + 0.12, r"$\log 2$", color="#64748B", fontsize=10)
ax.axvline(2, color="#2563EB", ls="--", lw=1.0); ax.text(2.1, 2.9, "real centre", color="#2563EB", fontsize=9)
ax.set_xlabel(r"generator centre $a$"); ax.set_ylabel("distance"); ax.set_ylim(0, 6.2); ax.set_xlim(-4, 6)
ax.set_title("value", fontsize=10); ax.legend(loc="upper center", fontsize=9); ax.grid(alpha=0.3)
ax = axs[1]
ax.plot(A, dJ, color="#7C3AED", lw=2.4, label=r"$d\,\mathrm{JSD}/da$")
ax.plot(A, dW, color="#0F766E", lw=2.4, label=r"$dW/da$")
ax.axhline(0, color="#334155", lw=0.8); ax.axvline(2, color="#2563EB", ls="--", lw=1.0)
ax.axvspan(-4, -0.5, color="#FDE68A", alpha=0.35); ax.text(-3.9, 0.75, "far left:\nJSD slope near 0,\nW slope still -1", fontsize=8.6, color="#92400E")
ax.set_xlabel(r"generator centre $a$"); ax.set_ylabel("slope (gradient)"); ax.set_ylim(-1.3, 1.3); ax.set_xlim(-4, 6)
ax.set_title("slope: what the generator feels", fontsize=10); ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
fig.suptitle(r"real $\mathcal{N}(2,0.5^2)$ vs generated $\mathcal{N}(a,1)$", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "lec13-kp07-jsd-vs-w.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- figure 2: lec13-kp07-critic ----------
fig, ax = plt.subplots(figsize=(8.8, 4.3))
m = (x > -6.5) & (x < 5.5)
ax.fill_between(x[m], P[m] / P.max() * 0.9, color="#93C5FD", alpha=0.5, label=r"real $\mathcal{N}(2,0.5^2)$ (scaled)")
ax.fill_between(x[m], Q0[m] / Q0.max() * 0.9, color="#FDBA74", alpha=0.5, label=r"generated $\mathcal{N}(-3,1)$ (scaled)")
ax.plot(x[m], Dstar[m], color="#7C3AED", lw=2.4, label=r"optimal discriminator $D^*$ (0 to 1)")
ax2 = ax.twinx()
ax2.plot(x[m], fstar[m], color="#0F766E", lw=2.4, ls="--", label=r"optimal critic $f^*$ (slope 1)")
ax2.plot(xs.squeeze(1).numpy(), fc - np.interp(0.0, xs.squeeze(1).numpy(), fc), color="#BE123C", lw=1.6, ls=":", label=r"learned critic, $\lambda=100$ (shifted)")
ax.set_ylim(0, 1.05); ax2.set_ylim(-7.5, 6.5); ax.set_xlim(-6.5, 5.5)
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$D^*(x)$ and densities"); ax2.set_ylabel(r"critic value $f(x)$")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.2, ncol=2); ax.grid(alpha=0.3)
ax.set_title(r"far apart ($a=-3$): $D^*$ is flat at 0 over the fakes; the critic keeps a slope", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "lec13-kp07-critic.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- figure 3: lec13-kp07-score-gradient ----------
fig, ax = plt.subplots(figsize=(8.4, 4.0))
ax.plot(s, sig, color="#BE123C", lw=2.4, label=r"saturating $\log(1-\sigma(s))$: size $\sigma(s)$")
ax.plot(s, 1 - sig, color="#B45309", lw=2.4, ls="--", label=r"non-saturating $-\log\sigma(s)$: size $1-\sigma(s)$")
ax.plot(s, np.abs(s - 1), color="#0F766E", lw=2.4, label=r"LSGAN $\frac{1}{2}(s-1)^2$: size $|s-1|$")
ax.set_xlim(-6, 3); ax.set_ylim(0, 7.4); ax.set_xlabel(r"discriminator's raw score $s$ for a fake sample (low = clearly fake)")
ax.set_ylabel("size of generator's gradient")
ax.legend(loc="upper right", fontsize=8.8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "lec13-kp07-score-gradient.png", dpi=200, bbox_inches="tight"); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
