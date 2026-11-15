"""lec14-kp02-figures.py — 第14週 KP02 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec14-kp02-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
內容：DDPM 線性排程（beta 從 1e-4 到 0.02、T=1000）的 alpha、alpha-bar；兩步合併的手算；
[補充] 一維兩堆資料 0.5 N(-2, 0.3^2) + 0.5 N(2, 0.3^2) 逐步加雜訊與一步跳到第 t 步的比較。"""
import json, sys, time
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

# ---- 手算 1：KP01 的 beta = 0.36 三步，改用閉式一步算 ----
a = 0.64
abar3 = a ** 3
eps = [0.5, -1.0, 0.0]
noise = 0.8 ** 2 * 0.6 * eps[0] + 0.8 * 0.6 * eps[1] + 0.6 * eps[2]   # 三次雜訊合起來
x3 = np.sqrt(abar3) * 2.0 + noise
num["hand_kp01"] = dict(alpha=a, abar_3=r4(abar3), sqrt_abar_3=r4(np.sqrt(abar3)), sqrt_1m_abar_3=r4(np.sqrt(1 - abar3)),
                        combined_noise=r4(noise), eps_bar=r4(noise / np.sqrt(1 - abar3)), x3=r4(x3),
                        noise_var_check=r4(0.8 ** 4 * 0.36 + 0.8 ** 2 * 0.36 + 0.36))
assert num["hand_kp01"]["x3"] == 0.736 and num["hand_kp01"]["noise_var_check"] == r4(1 - abar3)

# ---- 手算 2：兩步合併的一般式，代入 beta1=0.1, beta2=0.2 ----
b1, b2 = 0.1, 0.2
a1, a2 = 1 - b1, 1 - b2
num["two_step"] = dict(beta1=b1, beta2=b2, alpha1=a1, alpha2=a2, abar2=r4(a1 * a2), mean_coef=r4(np.sqrt(a1 * a2)),
                       var_parts=[r4(a2 * b1), r4(b2)], var_total=r4(a2 * b1 + b2), one_minus_abar2=r4(1 - a1 * a2))
assert num["two_step"]["var_total"] == num["two_step"]["one_minus_abar2"] == 0.28

# ---- 練習題：來源習題 alpha_t = 0.9^t, x0 = 1 ----
al = [0.9 ** t for t in (1, 2)]
ab = np.cumprod(al)
num["exercise"] = dict(alpha=[r4(v) for v in al], abar=[r4(v) for v in ab],
                       mean=[r4(np.sqrt(v)) for v in ab], var=[r4(1 - v) for v in ab], std=[r4(np.sqrt(1 - v)) for v in ab])
assert num["exercise"]["abar"] == [0.9, 0.729]

# ---- DDPM 線性排程 ----
T = 1000
beta = np.linspace(1e-4, 0.02, T); alpha = 1 - beta; abar = np.cumprod(alpha)
tt = np.arange(1, T + 1)
cross = int(tt[np.argmin(np.abs(abar - 0.5))])
sched = {str(t): dict(beta=r4(beta[t - 1]), alpha=r4(alpha[t - 1]), abar=float(f"{abar[t - 1]:.4g}"),
                      sqrt_abar=r4(np.sqrt(abar[t - 1])), sqrt_1m_abar=r4(np.sqrt(1 - abar[t - 1])))
         for t in (1, 10, 100, 250, 500, 750, 1000)}
num["linear_schedule"] = dict(T=T, beta_first=1e-4, beta_last=0.02, table=sched, t_half=cross,
                              abar_at_half=r4(abar[cross - 1]), sum_beta=r4(beta.sum()),
                              exp_minus_sum_beta=float(f"{np.exp(-beta.sum()):.4g}"))
assert 4.0e-5 < abar[-1] < 4.1e-5 and 250 <= cross <= 260
# 來源互動展示的設定：beta 從 0.01 到 0.2、T = 100
b_demo = np.linspace(0.01, 0.2, 100)
num["source_demo"] = dict(beta_first=0.01, beta_last=0.2, T=100, abar_T=float(f"{np.prod(1 - b_demo):.3g}"))
assert np.prod(1 - b_demo) < 0.05

# ---- 練習題：beta 固定 0.36 時，幾步後 sqrt(abar) < 0.01；DDPM 排程第 500 步的一步跳 ----
k = 1
while 0.8 ** k >= 0.01:
    k += 1
x500 = np.sqrt(abar[499]) * 2.0 + np.sqrt(1 - abar[499]) * 0.5
num["practice"] = dict(steps_until_below_001=k, value_at_k=r4(0.8 ** k), value_at_k_minus_1=r4(0.8 ** (k - 1)),
                       x500_from_x0_2_eps_05=r4(x500))
assert k == 21 and num["practice"]["x500_from_x0_2_eps_05"] == 1.0406

# ---- [補充] 逐步 vs 一步跳：兩堆資料 ----
MU, S, N = 2.0, 0.3, 50000
CHK = (100, 300, 600)
rng = np.random.default_rng(0)
x0 = rng.choice([-MU, MU], N) + S * rng.normal(size=N)
t0 = time.perf_counter()
xs = x0.copy(); step = {}
for t in range(1, max(CHK) + 1):
    xs = np.sqrt(alpha[t - 1]) * xs + np.sqrt(beta[t - 1]) * rng.normal(size=N)
    if t in CHK:
        step[t] = xs.copy()
rng2 = np.random.default_rng(1)
jump = {t: np.sqrt(abar[t - 1]) * x0 + np.sqrt(1 - abar[t - 1]) * rng2.normal(size=N) for t in CHK}
st = lambda v: dict(mean=round(float(v.mean()), 3), std=round(float(v.std()), 3))
theory_std = {t: float(np.sqrt(abar[t - 1] * (MU ** 2 + S ** 2) + 1 - abar[t - 1])) for t in CHK}
num["step_vs_jump"] = dict(data="0.5 N(-2, 0.3^2) + 0.5 N(2, 0.3^2)", samples=N, seeds=[0, 1],
                           step={str(t): st(step[t]) for t in CHK}, jump={str(t): st(jump[t]) for t in CHK},
                           theory_std={str(t): round(theory_std[t], 3) for t in CHK},
                           random_draws_per_sample={"step": {str(t): t for t in CHK}, "jump": 1})
for t in CHK:
    assert abs(step[t].std() - theory_std[t]) < 0.02 and abs(jump[t].std() - theory_std[t]) < 0.02
# 用同一組 x0，分別「一步跳」到第 300 步和第 310 步：兩者並不在同一條路徑上
TA, TB = 300, 310
rng3 = np.random.default_rng(2)
ja = np.sqrt(abar[TA - 1]) * x0 + np.sqrt(1 - abar[TA - 1]) * rng3.normal(size=N)
jb = np.sqrt(abar[TB - 1]) * x0 + np.sqrt(1 - abar[TB - 1]) * rng3.normal(size=N)
pa = ja.copy(); pb = pa.copy()                      # 從第 300 步沿著前向過程再走 10 步
for t in range(TA + 1, TB + 1):
    pb = np.sqrt(alpha[t - 1]) * pb + np.sqrt(beta[t - 1]) * rng3.normal(size=N)
num["path_vs_marginal"] = dict(t_a=TA, t_b=TB, rms_diff_two_jumps=round(float(np.sqrt(((jb - ja) ** 2).mean())), 3),
                               rms_diff_along_path=round(float(np.sqrt(((pb - pa) ** 2).mean())), 3),
                               std_b_two_jumps=round(float(jb.std()), 3), std_b_along_path=round(float(pb.std()), 3))
P = num["path_vs_marginal"]
assert P["rms_diff_two_jumps"] > 3 * P["rms_diff_along_path"] and abs(P["std_b_two_jumps"] - P["std_b_along_path"]) < 0.02

# ---------- figure 1: lec14-kp02-schedule ----------
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0))
ax = axs[0]
ax.plot(tt, beta, color="#DC2626", lw=2, label=r"$\beta_t$ (left axis)")
ax.set_xlabel("step $t$"); ax.set_ylabel(r"$\beta_t$", color="#DC2626"); ax.set_ylim(0, 0.021)
ax2 = ax.twinx(); ax2.plot(tt, alpha, color="#2563EB", lw=2, ls="--", label=r"$\alpha_t=1-\beta_t$ (right axis)")
ax2.set_ylabel(r"$\alpha_t$", color="#2563EB"); ax2.set_ylim(0.978, 1.0005)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=9)
ax.set_title(r"one step: $\beta_t$ grows from $0.0001$ to $0.02$", fontsize=10.5)
ax = axs[1]
ax.plot(tt, np.sqrt(abar), color="#2563EB", lw=2, label=r"$\sqrt{\bar\alpha_t}$: weight of $x_0$")
ax.plot(tt, np.sqrt(1 - abar), color="#7C3AED", lw=2, ls="--", label=r"$\sqrt{1-\bar\alpha_t}$: weight of noise")
ax.axvline(cross, color="#6B7280", lw=0.8)
ax.plot([cross], [np.sqrt(0.5)], "o", color="#111827", ms=4)
ax.text(cross + 12, 0.03, f"$t={cross}$: equal weights $\\sqrt{{0.5}}$", fontsize=9, color="#374151")
ax.set_xlabel("step $t$"); ax.set_ylim(0, 1.03); ax.legend(loc="center right", fontsize=9)
ax.set_title(r"after $t$ steps: $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "lec14-kp02-schedule.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec14-kp02-step-vs-jump ----------
fig, axs = plt.subplots(1, 3, figsize=(12.0, 3.7), sharey=True)
bins = np.linspace(-4.5, 4.5, 91)
for ax, t in zip(axs, CHK):
    ax.hist(step[t], bins=bins, density=True, color="#2563EB", alpha=0.45, label=f"step by step ($t$ draws)")
    ax.hist(jump[t], bins=bins, density=True, histtype="step", color="#7C3AED", lw=1.6, label="one jump (1 draw)")
    ax.set_title(f"$t={t}$:  std {step[t].std():.2f} vs {jump[t].std():.2f}", fontsize=10.5)
    ax.set_xlabel("$x_t$"); ax.set_xlim(-4.5, 4.5)
axs[0].set_ylabel("density"); axs[0].legend(loc="upper center", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "lec14-kp02-step-vs-jump.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
