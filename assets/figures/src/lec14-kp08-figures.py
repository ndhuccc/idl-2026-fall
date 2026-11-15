"""lec14-kp08-figures.py — 第14週 KP08 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec14-kp08-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
內容：
1. 一步的近似：sqrt(1 - beta) 與 1 - beta / 2。
2. [補充] DDPM 的連續版（VP-SDE）：beta(t) = 0.1 + 19.9 t，t 在 [0, 1]；abar(t) = exp(-(0.1 t + 9.95 t^2))，和離散 DDPM 比較。
3. [補充] 一維兩堆資料 0.5 N(-2, 0.3^2) + 0.5 N(2, 0.3^2)：
   前向 SDE（Euler–Maruyama）、逆向 SDE 與機率流 ODE（用精確的分數），比較生成結果與軌跡。
4. 練習題的數字。"""
import json, sys
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
r4 = lambda v: round(float(v), 4)

# ---------- 1. sqrt(1 - beta) ≈ 1 - beta/2 ----------
num["one_step_approx"] = {str(b): dict(exact=round(float(np.sqrt(1 - b)), 6), approx=round(1 - b / 2, 6), error=float(f"{(1 - b / 2) - np.sqrt(1 - b):.2g}"))
                          for b in (0.0001, 0.02, 0.36)}

# ---------- 2. VP-SDE vs discrete DDPM ----------
BMIN, BMAX = 0.1, 20.0
beta_c = lambda t: BMIN + (BMAX - BMIN) * t
abar_c = lambda t: np.exp(-(BMIN * t + 0.5 * (BMAX - BMIN) * t * t))
T = 1000
beta_d = np.linspace(1e-4, 0.02, T); abar_d = np.cumprod(1 - beta_d)
num["vp_vs_ddpm"] = {str(t): dict(continuous=float(f"{abar_c(t / T):.4g}"), discrete=float(f"{abar_d[t - 1]:.4g}")) for t in (100, 300, 600, 1000)}
assert abs(abar_c(0.3) - abar_d[299]) < 0.005

# ---------- 3. SDE / ODE on two piles ----------
MU, S = 2.0, 0.3


def score(x, t):
    ab = abar_c(t)
    v = ab * S * S + (1 - ab)
    comps = [np.exp(-(x - s_ * np.sqrt(ab) * MU) ** 2 / (2 * v)) for s_ in (-1, 1)]
    return sum(-(x - s_ * np.sqrt(ab) * MU) / v * c for s_, c in zip((-1, 1), comps)) / (comps[0] + comps[1])


N = 20000; K = 1000; dt = 1.0 / K
rng = np.random.default_rng(0)
x0 = rng.choice([-MU, MU], N) + S * rng.normal(size=N)
x = x0.copy(); fwd = {}
for k in range(K):                                          # forward SDE: dx = -1/2 beta x dt + sqrt(beta) dw
    t = k * dt
    x = x - 0.5 * beta_c(t) * x * dt + np.sqrt(beta_c(t) * dt) * rng.normal(size=N)
    if k + 1 in (300, 600, 1000):
        fwd[str(k + 1)] = dict(std=r4(x.std()), theory=r4(np.sqrt(abar_c((k + 1) / K) * (MU ** 2 + S ** 2) + 1 - abar_c((k + 1) / K))))
num["forward_sde"] = dict(steps=K, samples=N, seed=0, std_at_step=fwd)
for v in fwd.values():
    assert abs(v["std"] - v["theory"]) < 0.03


def reverse(xT, steps, mode, seed, keep_paths=0):
    g = np.random.default_rng(seed); x = xT.copy(); h = 1.0 / steps
    paths = [x[:keep_paths].copy()] if keep_paths else None
    for k in range(steps, 0, -1):
        t = k * h; b = beta_c(t)
        if mode == "sde":                                   # dx = [f - g^2 score] dt + g dw_bar, run with dt = -h
            x = x + (0.5 * b * x + b * score(x, t)) * h + (np.sqrt(b * h) * g.normal(size=x.size) if k > 1 else 0.0)
        else:                                               # probability-flow ODE: dx = [f - 1/2 g^2 score] dt
            x = x + (0.5 * b * x + 0.5 * b * score(x, t)) * h
        if keep_paths:
            paths.append(x[:keep_paths].copy())
    return x, (np.array(paths) if keep_paths else None)


xT = np.random.default_rng(1).normal(size=N)
gen = {}
for mode, steps in (("sde", 1000), ("ode", 1000), ("ode", 100), ("sde", 100)):
    xg, _ = reverse(xT, steps, mode, seed=2)
    r = xg[xg > 0]
    gen[f"{mode}_{steps}"] = dict(frac_right=r4((xg > 0).mean()), right_mean=r4(r.mean()), right_std=r4(r.std()), between=r4((np.abs(xg) < 1).mean()))
xg2, _ = reverse(xT, 1000, "ode", seed=99)
x_ode, _ = reverse(xT, 1000, "ode", seed=2)
num["reverse"] = dict(start="x_T ~ N(0,1), seed 1", samples=N, results=gen, ode_same_start_same_result=bool(np.allclose(xg2, x_ode)))
for k in ("sde_1000", "ode_1000"):
    assert abs(gen[k]["right_std"] - 0.3) < 0.03 and 0.48 < gen[k]["frac_right"] < 0.52
assert num["reverse"]["ode_same_start_same_result"]

# paths for the figure: 6 starts
starts = np.array([-1.2, -0.4, 0.05, 0.3, 0.9, 1.5])
_, p_sde = reverse(starts, 1000, "sde", seed=5, keep_paths=6)
_, p_sde2 = reverse(starts, 1000, "sde", seed=6, keep_paths=6)
_, p_ode = reverse(starts, 1000, "ode", seed=5, keep_paths=6)
num["paths"] = dict(starts=starts.tolist(), sde_ends=[round(float(v), 2) for v in p_sde[-1]], sde_ends_other_seed=[round(float(v), 2) for v in p_sde2[-1]],
                    ode_ends=[round(float(v), 2) for v in p_ode[-1]])

# ---------- 4. 練習題 ----------
ab = 0.5
num["practice"] = dict(entropy_1d=r4(0.5 * np.log(2 * np.pi * np.e * (1 - ab))), posterior_mean_gaussian_data=r4(np.sqrt(ab) * 1.0),
                       posterior_var_gaussian_data=r4(1 - ab),
                       em_step=r4(1.0 + (-0.5 * 10 * 1.0) * 0.01 + np.sqrt(10) * np.sqrt(0.01) * 0.5),
                       dsm_target_example=r4(-(1.0 - np.sqrt(0.5) * 2.0) / (1 - 0.5)))

# ---------- figure 1: lec14-kp08-paths ----------
fig, axs = plt.subplots(1, 2, figsize=(11.6, 3.9), sharey=True)
tt = np.linspace(1, 0, 1001)
for p, c in ((p_sde, "#7C3AED"), (p_sde2, "#C4B5FD")):
    for j in range(6):
        axs[0].plot(tt, p[:, j], color=c, lw=0.8)
for j in range(6):
    axs[1].plot(tt, p_ode[:, j], color="#059669", lw=1.4)
axs[0].set_title("reverse SDE: same 6 starts, two runs (dark / light)", fontsize=10.5)
axs[1].set_title("probability-flow ODE: one smooth path per start", fontsize=10.5)
for ax in axs:
    ax.set_xlim(1, 0); ax.set_xlabel("time $t$ (runs from 1 down to 0)")
    ax.axhline(2, color="#9CA3AF", lw=0.6, ls="--"); ax.axhline(-2, color="#9CA3AF", lw=0.6, ls="--"); ax.set_ylim(-3.6, 3.6)
axs[0].set_ylabel("$x$")
fig.tight_layout(); fig.savefig(OUT / "lec14-kp08-paths.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec14-kp08-samples ----------
fig, ax = plt.subplots(figsize=(9.0, 3.6))
g = np.linspace(-3.5, 3.5, 600)
dens = 0.5 * (np.exp(-(g + MU) ** 2 / (2 * S * S)) + np.exp(-(g - MU) ** 2 / (2 * S * S))) / (S * np.sqrt(2 * np.pi))
xs_sde, _ = reverse(xT, 1000, "sde", seed=2)
bins = np.linspace(-3.5, 3.5, 141)
ax.hist(xs_sde, bins=bins, density=True, color="#C4B5FD", alpha=0.85, label="reverse SDE (1000 steps)")
ax.hist(x_ode, bins=bins, density=True, histtype="step", color="#059669", lw=1.5, label="probability-flow ODE (1000 steps)")
ax.plot(g, dens, color="#111827", lw=1.3, ls="--", label="true data density")
ax.set_xlim(-3.5, 3.5); ax.set_ylim(0, 1.0); ax.set_xlabel("$x_0$"); ax.set_ylabel("density")
ax.legend(loc="upper center", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "lec14-kp08-samples.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
