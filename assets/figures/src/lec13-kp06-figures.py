"""lec13-kp06-figures.py — 第13週 KP06 講義的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec13-kp06-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
需要 numpy、matplotlib 與 PyTorch（CPU 版即可；單執行緒約需數分鐘）。
[補充] (1) 雙峰分布用一個常態分布去配：正向 KL 與反向 KL 的最佳解；
(2) 二維 8 個高斯環的小 GAN（生成器與判別器都是兩層 128 單元的 MLP），標準 GAN 與特徵匹配各跑 5 個種子；
(3) 一個批次裡的統計量：涵蓋幾個模式、樣本彼此平均距離。"""
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

# ---------- (1) forward vs reverse KL, fitting one Gaussian to two bumps ----------
xg = np.linspace(-10, 10, 20001); dx = xg[1] - xg[0]
Nd = lambda m, s: np.exp(-(xg - m) ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi))
pd = 0.5 * Nd(-2, 0.5) + 0.5 * Nd(2, 0.5)
def kl(a, b):
    m = a > 1e-300
    return float(np.sum(a[m] * np.log(a[m] / np.maximum(b[m], 1e-300))) * dx)
best = (1e9, 0, 0)
for m in np.arange(0, 3.001, 0.01):              # by symmetry the left mode gives the same value
    for s in np.arange(0.2, 3.001, 0.01):
        r = kl(Nd(m, s), pd)
        if r < best[0]:
            best = (r, m, s)
fwd_s = math.sqrt(0.25 + 4.0)                   # forward KL over Gaussians = moment matching: mean 0, var 0.25+4
num["kl_fit"] = dict(target="0.5 N(-2,0.5^2) + 0.5 N(2,0.5^2)",
                     forward=dict(mean=0.0, sd=round(fwd_s, 3), forward_kl=round(kl(pd, Nd(0, fwd_s)), 3), reverse_kl=round(kl(Nd(0, fwd_s), pd), 3)),
                     reverse=dict(mean=round(best[1], 2), sd=round(best[2], 2), reverse_kl=round(best[0], 3), forward_kl=round(kl(pd, Nd(best[1], best[2])), 2)),
                     log2=round(math.log(2), 3))
assert abs(best[1] - 2) < 0.02 and abs(best[2] - 0.5) < 0.02

# ---------- (2) ring of 8 Gaussians ----------
torch.set_num_threads(1)
ANG = torch.arange(8) * 2 * math.pi / 8
CEN = torch.stack([2 * torch.cos(ANG), 2 * torch.sin(ANG)], 1)
def real(n, g): return CEN[torch.randint(0, 8, (n,), generator=g)] + 0.05 * torch.randn(n, 2, generator=g)
def coverage(x):
    d = torch.cdist(x, CEN); m = d.min(1); ok = m.values < 0.15
    counts = torch.bincount(m.indices[ok], minlength=8)
    return int((counts >= 0.02 * len(x)).sum()), float(ok.float().mean())
class Disc(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()); s.o = nn.Linear(128, 1)
    def forward(s, x): return s.o(s.f(x)).squeeze(1)
def train(seed, mode, steps=5000, lrG=1e-4, lrD=1e-3, zdim=2, n=256):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    G = nn.Sequential(nn.Linear(zdim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 2)); D = Disc()
    oG = torch.optim.Adam(G.parameters(), lrG, betas=(0.5, 0.999)); oD = torch.optim.Adam(D.parameters(), lrD, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss(); zf = torch.randn(2000, zdim, generator=g); log = []
    for t in range(steps + 1):
        if t % 250 == 0:
            with torch.no_grad(): log.append((t,) + coverage(G(zf)))
        if t == steps: break
        xr = real(n, g); xf = G(torch.randn(n, zdim, generator=g))
        lD = bce(D(xr), torch.ones(n)) + bce(D(xf.detach()), torch.zeros(n)); oD.zero_grad(); lD.backward(); oD.step()
        xf = G(torch.randn(n, zdim, generator=g))
        if mode == "fm":   # feature matching: match the mean of D's last hidden layer
            lG = ((D.f(xr).mean(0) - D.f(xf).mean(0)) ** 2).sum()
        else:              # non-saturating loss (KP02)
            lG = bce(D(xf), torch.ones(n))
        oG.zero_grad(); lG.backward(); oG.step()
    with torch.no_grad(): xs = G(zf).numpy()
    return log, xs
SEEDS = range(5); runs = {}
for mode in ("std", "fm"):
    for s in SEEDS:
        runs[(mode, s)] = train(s, mode)
num["ring"] = dict(settings="8 Gaussians on a circle of radius 2 (sd 0.05); G,D: MLP 2x128 (G tanh, D ReLU); z dim 2; Adam(0.5,0.999) lrG 1e-4 lrD 1e-3; "
                            "batch 256; 1 D step per G step; 5000 steps; seeds 0-4; a mode counts as covered if >=2% of 2000 samples lie within 0.15 of its centre",
                   final_modes={m: [runs[(m, s)][0][-1][1] for s in SEEDS] for m in ("std", "fm")},
                   final_on_mode_share={m: [round(runs[(m, s)][0][-1][2], 3) for s in SEEDS] for m in ("std", "fm")})
num["ring"]["trajectories_every_250_steps"] = {m: {str(sd): [r[1] for r in runs[(m, sd)][0]] for sd in SEEDS} for m in ("std", "fm")}
SHOW = 1
assert num["ring"]["final_modes"]["std"][SHOW] < 8 and num["ring"]["final_modes"]["fm"][SHOW] == 8

# ---------- (3) batch statistics ----------
gb = torch.Generator().manual_seed(123)
xr64 = real(64, gb).numpy(); xc64 = runs[("std", SHOW)][1][:64]
def close(a, r=0.3):   # how many OTHER samples in the batch lie within r: the idea behind minibatch discrimination
    d = np.linalg.norm(a[:, None] - a[None], axis=2); return round(float(((d < r).sum(1) - 1).mean()), 2)
num["batch"] = dict(batch=64, radius=0.3, prob_real_batch_misses_a_mode_upper=float(f"{8 * (7 / 8) ** 64:.2e}"),
                    close_neighbours_real=close(xr64), close_neighbours_std_seed1=close(xc64),
                    close_neighbours_theory_8_modes=round(63 / 8, 2), close_neighbours_theory_4_modes=round(63 / 4, 2))

r2 = float((CEN ** 2).sum(1).mean()) + 2 * 0.05 ** 2   # E||x||^2 of the ring data
num["practice"] = dict(neighbours_8=round(63 / 8, 2), neighbours_2=round(63 / 2, 2), fm_mean_only=0.0,
                       ring_E_norm2=round(r2, 3), fm_with_norm2=round(r2 ** 2, 2))

# ---------- figure 1: lec13-kp06-kl-direction ----------
fig, ax = plt.subplots(figsize=(8.6, 4.0))
ax.fill_between(xg, pd, color="#93C5FD", alpha=0.55, label=r"target $p$: two bumps at $\pm2$")
ax.plot(xg, Nd(0, fwd_s), color="#B45309", lw=2.4, label=r"min forward KL $D_{\mathrm{KL}}(p\,\Vert\,q)$: $\mathcal{N}(0,\ 2.06^2)$")
ax.plot(xg, Nd(best[1], best[2]), color="#BE123C", lw=2.4, ls="--", label=r"min reverse KL $D_{\mathrm{KL}}(q\,\Vert\,p)$: $\mathcal{N}(2,\ 0.5^2)$")
ax.set_xlim(-5.5, 5.5); ax.set_ylim(0, 0.9); ax.set_xlabel(r"$x$"); ax.set_ylabel("density")
ax.text(-2, 0.47, "covers both,\nbut puts mass\nin the gap", ha="center", fontsize=9, color="#B45309")
ax.text(3.5, 0.62, "sits on one bump,\nignores the other", ha="center", fontsize=9, color="#BE123C")
ax.legend(loc="upper left", fontsize=8.6, frameon=True); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "lec13-kp06-kl-direction.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec13-kp06-ring-samples ----------
fig, axs = plt.subplots(1, 3, figsize=(12.0, 4.3), sharex=True, sharey=True)
xr_show = real(2000, torch.Generator().manual_seed(7)).numpy()
panels = [(xr_show, "real data: 8 modes", "#2563EB"),
          (runs[("std", SHOW)][1], f"standard GAN (seed {SHOW}): {num['ring']['final_modes']['std'][SHOW]} of 8 modes", "#BE123C"),
          (runs[("fm", SHOW)][1], f"feature matching (seed {SHOW}): {num['ring']['final_modes']['fm'][SHOW]} of 8 modes", "#15803D")]
for ax, (pts, title, col) in zip(axs, panels):
    ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.35, color=col)
    ax.scatter(CEN[:, 0], CEN[:, 1], s=60, facecolors="none", edgecolors="#334155", lw=1.2)
    ax.set_title(title, fontsize=10); ax.set_aspect("equal"); ax.grid(alpha=0.3)
axs[0].set_xlim(-3, 3); axs[0].set_ylim(-3, 3)
for ax in axs: ax.set_xlabel(r"$x_1$")
axs[0].set_ylabel(r"$x_2$")
fig.suptitle("2000 samples after 5000 steps; open circles mark the 8 real mode centres", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "lec13-kp06-ring-samples.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- figure 3: lec13-kp06-coverage ----------
fig, ax = plt.subplots(figsize=(8.6, 4.0))
for mode, col, lab in (("std", "#BE123C", "standard GAN"), ("fm", "#15803D", "feature matching")):
    for s in SEEDS:
        L = np.array(runs[(mode, s)][0])
        ax.plot(L[:, 0], L[:, 1] + (0.06 if mode == "fm" else -0.06), color=col, lw=1.6, alpha=0.8, label=lab if s == 0 else None)
ax.set_xlim(0, 5000); ax.set_ylim(-0.5, 8.6); ax.set_yticks(range(0, 9))
ax.set_xlabel("training step"); ax.set_ylabel("modes covered (of 8)")
ax.set_title("5 seeds each (lines nudged apart vertically so they do not hide each other)", fontsize=9.5)
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "lec13-kp06-coverage.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
