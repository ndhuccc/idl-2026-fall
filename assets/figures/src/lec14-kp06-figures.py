"""lec14-kp06-figures.py — 第14週 KP06 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec14-kp06-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
內容：
1. 學變異數：beta_t = 0.2、abar_{t-1} = 0.6 時 beta-tilde 與對數插值（來源互動展示的數字）；DDPM 排程下 beta-tilde / beta。
2. cosine 排程（s = 0.008）與線性排程的比較。
3. PriorGrad 的小例子：資料 N(0.5, 0.8^2)（來源互動展示的數字）。
4. [補充] 少步數生成：資料 0.5 N(-2, 0.3^2) + 0.5 N(2, 0.3^2)，用「完美的雜訊猜測」E[eps | x_t]（代替訓練好的網路，
   只看取樣方法本身的誤差），比較 DDPM（跳步，變異數用 beta-tilde）與 DDIM（eta = 0，確定性）在 K 步時的樣本品質。"""
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

# ---------- 1. 學變異數 ----------
b, abp = 0.2, 0.6
abt = abp * (1 - b)
bt = (1 - abp) / (1 - abt) * b
interp = {str(v): r4(np.exp(v * np.log(b) + (1 - v) * np.log(bt))) for v in (0.0, 0.5, 1.0, 1.2)}
num["learned_variance_demo"] = dict(beta=b, abar_prev=abp, abar_t=r4(abt), beta_tilde=r4(bt), interp=interp,
                                    geometric_mean=r4(np.sqrt(b * bt)), arithmetic_mean=r4((b + bt) / 2))
assert abs(interp["0.5"] - np.sqrt(b * bt)) < 1e-4 and interp["1.2"] > b

T = 1000
beta_lin = np.linspace(1e-4, 0.02, T); abar_lin = np.cumprod(1 - beta_lin)
abar_prev_lin = np.concatenate([[1.0], abar_lin[:-1]])
bt_lin = (1 - abar_prev_lin) / (1 - abar_lin) * beta_lin
num["beta_tilde_ratio_linear"] = {str(t): r4(bt_lin[t - 1] / beta_lin[t - 1]) for t in (2, 10, 50, 100, 1000)}

# ---------- 2. cosine 排程 ----------
s = 0.008
f = lambda t: np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
tt = np.arange(0, T + 1)
abar_cos_full = f(tt) / f(0)
beta_cos = np.clip(1 - abar_cos_full[1:] / abar_cos_full[:-1], 0, 0.999)
abar_cos = np.cumprod(1 - beta_cos)
def t_where(ab, level):
    return int(np.argmax(ab < level) + 1)
num["schedules"] = dict(
    linear=dict(t_half=t_where(abar_lin, 0.5), t_below_001=t_where(abar_lin, 0.01), frac_steps_below_001=r4((abar_lin < 0.01).mean()),
                abar_T=float(f"{abar_lin[-1]:.3g}"), beta_first=1e-4, beta_last=0.02),
    cosine=dict(s=s, t_half=t_where(abar_cos, 0.5), t_below_001=t_where(abar_cos, 0.01), frac_steps_below_001=r4((abar_cos < 0.01).mean()),
                abar_T=float(f"{abar_cos[-1]:.3g}"), beta_first=float(f"{beta_cos[0]:.3g}"), beta_last=r4(beta_cos[-1]),
                n_clipped=int((beta_cos >= 0.999).sum())),
    sqrt_abar_at={str(t): dict(linear=r4(np.sqrt(abar_lin[t - 1])), cosine=r4(np.sqrt(abar_cos[t - 1]))) for t in (100, 250, 500, 750, 900)})
SC = num["schedules"]
assert SC["linear"]["t_half"] < SC["cosine"]["t_half"] and SC["linear"]["frac_steps_below_001"] > 2 * SC["cosine"]["frac_steps_below_001"]

# ---------- 3. PriorGrad ----------
mu, sd = 0.5, 0.8
kl_std = 0.5 * (sd ** 2 + mu ** 2 - 1 - np.log(sd ** 2))
num["priorgrad_demo"] = dict(data_mean=mu, data_std=sd, kl_data_to_standard_normal=r4(kl_std), kl_centered_to_data_prior=0.0)

# ---------- 重要性抽樣的玩具例子：總損失 = L1 + L2 ----------
def is_demo(L1, L2):
    tot = L1 + L2
    uni = [2 * L1, 2 * L2]                          # 均勻抽 t，乘 2 才是總和的不偏估計
    p1 = L1 / tot; imp = [L1 / p1, L2 / (1 - p1)]
    var = lambda vals, ps: sum(pp * (v - tot) ** 2 for v, pp in zip(vals, ps))
    return dict(L1=L1, L2=L2, total=tot, uniform_values=uni, uniform_var=r4(var(uni, [0.5, 0.5])),
                p=[r4(p1), r4(1 - p1)], importance_values=[r4(v) for v in imp], importance_var=r4(var(imp, [p1, 1 - p1])))
num["importance_demo"] = is_demo(10.0, 0.1)
num["practice3"] = is_demo(4.0, 1.0)
assert num["importance_demo"]["importance_var"] == 0 and num["practice3"]["uniform_var"] == 9.0

# ---------- 練習題 1、2、4 ----------
b1, abp1 = 0.1, 0.5
bt1 = (1 - abp1) / (1 - abp1 * (1 - b1)) * b1
num["practice1"] = dict(beta_tilde=r4(bt1), v=0.25, sigma2=r4(np.exp(0.25 * np.log(b1) + 0.75 * np.log(bt1))))
num["practice2"] = dict(abar_half_T=r4(f(T / 2) / f(0)), sqrt=r4(np.sqrt(f(T / 2) / f(0))))
xt4, abt4, abs4, e4 = 0.5, 0.5, 0.8, 0.3
x0h4 = (xt4 - np.sqrt(1 - abt4) * e4) / np.sqrt(abt4)
num["practice4"] = dict(x0_hat=r4(x0h4), x_s=r4(np.sqrt(abs4) * x0h4 + np.sqrt(1 - abs4) * e4))

# ---------- 4. [補充] 少步數生成：DDPM vs DDIM ----------
MU, S = 2.0, 0.3


def best_eps(xt, ab):
    m = np.sqrt(ab) * np.array([-MU, MU])[:, None]
    v = ab * S * S + (1 - ab)
    lw = -(xt[None, :] - m) ** 2 / (2 * v)
    r = np.exp(lw - lw.max(0)); r /= r.sum(0)
    x0c = np.array([-MU, MU])[:, None] + (np.sqrt(ab) * S * S / v) * (xt[None, :] - m)
    return (xt - np.sqrt(ab) * (r * x0c).sum(0)) / np.sqrt(1 - ab)


def sample(K, method, n=20000, seed=3):
    rng = np.random.default_rng(seed)
    ts = np.unique(np.round(np.linspace(T, 1, K)).astype(int))[::-1]     # K steps from T down to 1
    x = rng.normal(size=n)
    for j, t in enumerate(ts):
        ab_t = abar_lin[t - 1]
        ab_s = abar_lin[ts[j + 1] - 1] if j + 1 < len(ts) else 1.0
        e = best_eps(x, ab_t)
        x0h = (x - np.sqrt(1 - ab_t) * e) / np.sqrt(ab_t)
        if method == "ddim":
            x = np.sqrt(ab_s) * x0h + np.sqrt(1 - ab_s) * e
        else:                                                             # DDPM ancestral step from t to s
            a_ts = ab_t / ab_s; b_ts = 1 - a_ts
            mean = (np.sqrt(ab_s) * b_ts / (1 - ab_t)) * x0h + (np.sqrt(a_ts) * (1 - ab_s) / (1 - ab_t)) * x
            var = (1 - ab_s) / (1 - ab_t) * b_ts
            x = mean + (np.sqrt(var) * rng.normal(size=n) if ab_s < 1.0 else 0.0)
    return x


from math import erf
NS = 200000
grid_q = np.linspace(-4.5, 4.5, 90001)
cdf = 0.5 * (0.5 * (1 + np.vectorize(erf)((grid_q + MU) / (S * np.sqrt(2)))) + 0.5 * (1 + np.vectorize(erf)((grid_q - MU) / (S * np.sqrt(2)))))
u = (np.arange(NS) + 0.5) / NS
TRUE_Q = np.interp(u, cdf, grid_q)                                     # quantiles of the true data distribution
w1 = lambda x: float(np.abs(np.sort(x) - TRUE_Q).mean())               # 1-D Wasserstein-1 distance to the true distribution
KS = [5, 10, 20, 50, 100, 1000]
res = {}
for mth in ("ddpm", "ddim"):
    res[mth] = {}
    for K in KS:
        x = sample(K, mth, n=NS)
        r = x[x > 0]
        res[mth][str(K)] = dict(w1=float(f"{w1(x):.2g}"), right_std=round(float(r.std()), 2), between=r4((np.abs(x) < 1).mean()))
gd = np.random.default_rng(100)
data_draw = gd.choice([-MU, MU], NS) + S * gd.normal(size=NS)
num["few_steps"] = dict(data="0.5 N(-2, 0.3^2) + 0.5 N(2, 0.3^2)", eps="perfect guess E[eps|x_t]", schedule="linear 1e-4..0.02, T=1000",
                        steps=KS, samples=NS, seed=3, ddpm_variance="beta-tilde between chosen steps (= DDIM eta=1)", ddim_eta=0,
                        distance="1-D Wasserstein-1 to the true distribution (exact quantiles)",
                        reference_w1_real_data_sample=float(f"{w1(data_draw):.2g}"), results=res)
R = res
assert R["ddim"]["50"]["w1"] < R["ddpm"]["50"]["w1"] and R["ddim"]["100"]["w1"] < R["ddpm"]["100"]["w1"] and R["ddpm"]["1000"]["w1"] < 0.02

# ---------- figure 1: lec14-kp06-schedules ----------
fig, axs = plt.subplots(1, 2, figsize=(11.4, 3.9))
x = np.arange(1, T + 1) / T
ax = axs[0]
ax.plot(x, np.sqrt(abar_lin), color="#2563EB", lw=2, label="linear")
ax.plot(x, np.sqrt(abar_cos), color="#DC2626", lw=2, ls="--", label=f"cosine ($s={s}$)")
ax.set_xlabel("$t/T$"); ax.set_ylabel(r"$\sqrt{\bar\alpha_t}$ (weight of $x_0$)"); ax.set_ylim(0, 1.03)
ax.legend(loc="upper right", fontsize=9.5); ax.set_title("how fast the data are erased", fontsize=10.5)
ax = axs[1]
ax.plot(x, beta_lin, color="#2563EB", lw=2, label="linear")
ax.plot(x, beta_cos, color="#DC2626", lw=2, ls="--", label="cosine (clipped at 0.999)")
ax.set_yscale("log"); ax.set_ylim(1e-5, 1.5)
ax.set_xlabel("$t/T$"); ax.set_ylabel(r"$\beta_t$ (log scale)")
ax.legend(loc="lower right", fontsize=9.5); ax.set_title("noise added per step", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "lec14-kp06-schedules.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec14-kp06-few-steps ----------
fig, ax = plt.subplots(figsize=(8.8, 3.9))
ax.plot(KS, [R["ddpm"][str(k)]["w1"] for k in KS], "o-", color="#7C3AED", lw=2, label=r"DDPM ($\eta=1$, stochastic)")
ax.plot(KS, [R["ddim"][str(k)]["w1"] for k in KS], "s--", color="#059669", lw=2, label=r"DDIM ($\eta=0$, deterministic)")
ax.axhline(num["few_steps"]["reference_w1_real_data_sample"], color="#9CA3AF", lw=1, ls=":", label=f"a real data sample of the same size")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("number of generation steps $K$ (log scale)"); ax.set_ylabel("distance to true distribution (log)")
ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
ax.legend(loc="upper right", fontsize=9.5)
ax.set_title(r"same perfect $\epsilon$-guess, fewer steps", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "lec14-kp06-few-steps.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
