"""lec14-kp04-figures.py — 第14週 KP04 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec14-kp04-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
內容：
1. 來源互動展示的數字：x0 = 1、x_t = 0.5、beta_t = 0.2、abar_{t-1} = 0.6，算真正的去噪分布 q(x_{t-1} | x_t, x0)，
   並用抽樣（只留下 x_t 落在 0.5 附近的樣本）核對；同時算出來源展示公式（少了根號係數）的結果。
2. 把 x0 換成雜訊後的均值公式（式 37）用同一組數字核對。
3. [補充] DDPM 排程、兩堆資料：三種預測目標（雜訊、x0、去噪均值的修正量）在各步的大小；beta-tilde 與 beta 的比較。"""
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

# ---------- 1. 來源展示的數字 ----------
x0, xt, b, abp = 1.0, 0.5, 0.2, 0.6
a = 1 - b; abt = abp * a
# 兩個「關於 x_{t-1} 的常態」
m1, v1 = xt / np.sqrt(a), b / a                  # 來自 q(x_t | x_{t-1})：把 x_t 往回放大
m2, v2 = np.sqrt(abp) * x0, 1 - abp              # 來自 q(x_{t-1} | x0)
prec = 1 / v1 + 1 / v2
v_post = 1 / prec; m_post = v_post * (m1 / v1 + m2 / v2)
c_xt = np.sqrt(a) * (1 - abp) / (1 - abt); c_x0 = np.sqrt(abp) * b / (1 - abt)
beta_tilde = (1 - abp) * b / (1 - abt)
mu35 = c_xt * xt + c_x0 * x0
eps = (xt - np.sqrt(abt) * x0) / np.sqrt(1 - abt)
mu37 = (xt - b / np.sqrt(1 - abt) * eps) / np.sqrt(a)
mu_source_demo = (b * x0 + a * (1 - abp) * xt) / (1 - abt)
# 抽樣核對
rng = np.random.default_rng(0); n = 4_000_000
xp = np.sqrt(abp) * x0 + np.sqrt(1 - abp) * rng.normal(size=n)
xn = np.sqrt(a) * xp + np.sqrt(b) * rng.normal(size=n)
keep = np.abs(xn - xt) < 0.01
mc = xp[keep]
num["demo"] = dict(x0=x0, x_t=xt, beta_t=b, alpha_t=r4(a), abar_prev=abp, abar_t=r4(abt),
                   from_xt=dict(mean=r4(m1), var=r4(v1)), from_x0=dict(mean=r4(m2), var=r4(v2)),
                   precision_sum=r4(prec), post_mean=r4(m_post), post_var=r4(v_post), post_std=r4(np.sqrt(v_post)),
                   coef_xt=r4(c_xt), coef_x0=r4(c_x0), beta_tilde=r4(beta_tilde), mu_eq35=r4(mu35),
                   eps=r4(eps), noise_coef=r4(b / np.sqrt(1 - abt)), mu_eq37=r4(mu37),
                   source_demo_formula_mean=r4(mu_source_demo),
                   mc=dict(samples=n, window=0.01, kept=int(keep.sum()), mean=r4(mc.mean()), var=r4(mc.var())))
assert abs(m_post - mu35) < 1e-12 and abs(mu35 - mu37) < 1e-12 and abs(v_post - beta_tilde) < 1e-12
assert abs(mc.mean() - m_post) < 0.01 and abs(mc.var() - v_post) < 0.01
assert abs(mu_source_demo - mu35) > 0.03

# ---------- 練習題 1、2 ----------
b_, abp_ = 0.1, 0.5; a_ = 1 - b_; abt_ = abp_ * a_
cxt = np.sqrt(a_) * (1 - abp_) / (1 - abt_); cx0 = np.sqrt(abp_) * b_ / (1 - abt_)
mu_p = cxt * 0.4 + cx0 * 1.0
eps_p = (0.4 - np.sqrt(abt_)) / np.sqrt(1 - abt_)
mu_p37 = (0.4 - b_ / np.sqrt(1 - abt_) * eps_p) / np.sqrt(a_)
num["practice"] = dict(alpha=r4(a_), abar=r4(abt_), beta_tilde=r4((1 - abp_) * b_ / (1 - abt_)), coef_xt=r4(cxt), coef_x0=r4(cx0),
                       mu=r4(mu_p), eps=r4(eps_p), noise_term=r4(b_ / np.sqrt(1 - abt_) * eps_p), mu_eq37=r4(mu_p37))
assert abs(mu_p - mu_p37) < 1e-12

# ---------- 3. [補充] 三種預測目標的大小（DDPM 排程，兩堆資料） ----------
T = 1000
beta = np.linspace(1e-4, 0.02, T); alpha = 1 - beta; abar = np.cumprod(alpha)
abar_prev = np.concatenate([[1.0], abar[:-1]])
MU, S, N = 2.0, 0.3, 20000
rng = np.random.default_rng(1)
X0 = rng.choice([-MU, MU], N) + S * rng.normal(size=N)
E = rng.normal(size=N)
TS = np.unique(np.round(np.geomspace(1, T, 40)).astype(int))
sd_eps, sd_x0, sd_corr = [], [], []
for t in TS:
    ab, abp_, bt, at = abar[t - 1], abar_prev[t - 1], beta[t - 1], alpha[t - 1]
    Xt = np.sqrt(ab) * X0 + np.sqrt(1 - ab) * E
    mu = (np.sqrt(at) * (1 - abp_) * Xt + np.sqrt(abp_) * bt * X0) / (1 - ab)
    sd_eps.append(E.std()); sd_x0.append(X0.std()); sd_corr.append((mu - Xt).std())
sd_corr = np.array(sd_corr)
pick = {str(t): r4(sd_corr[list(TS).index(t)]) for t in (1, 10, 100, 1000) if t in TS}
num["targets"] = dict(data="0.5 N(-2, 0.3^2) + 0.5 N(2, 0.3^2)", samples=N, seed=1, steps=TS.tolist(),
                      std_eps=r4(np.mean(sd_eps)), std_x0=r4(np.mean(sd_x0)), std_mean_correction=pick,
                      correction_min=r4(sd_corr.min()), correction_max=r4(sd_corr.max()))
assert sd_corr.max() < 0.05 and abs(np.mean(sd_eps) - 1) < 0.02
# 網路誤差換算成平均誤差的倍數：猜 eps 時 c_eps，猜 x0 時 c_x0
c_eps = beta / (np.sqrt(alpha) * np.sqrt(1 - abar))
c_x0 = np.sqrt(abar_prev) * beta / (1 - abar)
num["error_amplification"] = dict(eps_min=r4(c_eps.min()), eps_max=r4(c_eps.max()),
                                  x0_at_1=r4(c_x0[0]), x0_at_1000=float(f"{c_x0[-1]:.2g}"),
                                  ratio_eps_to_correction=[round(1 / sd_corr.max()), round(1 / sd_corr.min())],
                                  ratio_x0_to_correction=[round(2.02 / sd_corr.max()), round(2.02 / sd_corr.min())])
assert 0.005 < c_eps.min() and c_eps.max() < 0.021 and abs(c_x0[0] - 1) < 1e-9 and c_x0[-1] < 2e-4

# beta-tilde vs beta
bt_tilde = (1 - abar_prev) * beta / (1 - abar)
num["beta_tilde"] = {str(t): dict(beta=r4(beta[t - 1]), beta_tilde=float(f"{bt_tilde[t - 1]:.4g}"),
                                  ratio=r4(bt_tilde[t - 1] / beta[t - 1])) for t in (1, 2, 10, 100, 1000)}
assert bt_tilde[0] == 0.0 and bt_tilde[-1] / beta[-1] > 0.999

# ---------- figure 1: lec14-kp04-posterior ----------
fig, ax = plt.subplots(figsize=(9.6, 4.2))
g = np.linspace(-1.2, 2.6, 800)
pdf = lambda m, v: np.exp(-(g - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)
ax.hist(mc, bins=60, density=True, color="#C4B5FD", alpha=0.7, label=r"samples with $x_t\approx0.5$")
ax.plot(g, pdf(m1, v1), color="#2563EB", lw=1.8, ls="--", label=r"from $x_t$: mean $x_t/\sqrt{\alpha_t}$")
ax.plot(g, pdf(m2, v2), color="#059669", lw=1.8, ls="-.", label=r"from $x_0$: mean $\sqrt{\bar\alpha_{t-1}}\,x_0$")
ax.plot(g, pdf(m_post, v_post), color="#7C3AED", lw=2.4, label=r"$q(x_{t-1}\mid x_t,x_0)$")
for m, c in ((m1, "#2563EB"), (m2, "#059669"), (m_post, "#7C3AED")):
    ax.axvline(m, color=c, lw=0.8, ls=":")
ax.text(m1 - 0.03, 0.1, f"{m1:.3f}", color="#2563EB", ha="right", fontsize=9)
ax.text(m2 + 0.03, 0.1, f"{m2:.3f}", color="#059669", ha="left", fontsize=9)
ax.text(m_post, 1.08, f"{m_post:.3f}", color="#6D28D9", ha="center", fontsize=9)
ax.set_xlabel("$x_{t-1}$"); ax.set_ylabel("density"); ax.set_ylim(0, 1.18); ax.set_xlim(-1.2, 2.6)
ax.legend(loc="upper right", fontsize=8.8)
ax.set_title(r"$x_0=1,\ x_t=0.5,\ \beta_t=0.2,\ \bar\alpha_{t-1}=0.6$", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "lec14-kp04-posterior.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec14-kp04-target-scale ----------
fig, ax = plt.subplots(figsize=(9.0, 3.9))
ax.plot(TS, sd_eps, color="#7C3AED", lw=2.2, label=r"target $\epsilon$")
ax.plot(TS, sd_x0, color="#059669", lw=2.0, ls="-.", label=r"target $x_0$")
ax.plot(TS, sd_corr, color="#F59E0B", lw=2.0, ls="--", label=r"mean correction $\tilde\mu-x_t$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 5)
ax.set_xlabel("step $t$ (log scale)"); ax.set_ylabel("std over samples (log scale)")
ax.legend(loc="center right", fontsize=9)
ax.set_title("how large is the thing the network must output?", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "lec14-kp04-target-scale.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
