"""lec14-kp03-figures.py — 第14週 KP03 講義的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec14-kp03-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
[補充] 1. 為什麼逆向一步可以假設是常態：兩堆資料 0.5 N(-2,0.3^2)+0.5 N(2,0.3^2) 走一步，
       小步（beta=0.01）與大步（beta=0.5）時真正的逆向分布 q(x0 | x1)。
[補充] 2. 三步的小擴散模型（beta = 0.2, 0.4, 0.6），資料 N(0,1)、x0 = 1：
       模型 A 的逆向一步 = 真正的逆向 N(sqrt(alpha_t) x_t, beta_t)；模型 B 把平均改成 0.8 倍。
       用常態分布的公式算出 log p(x0) 與 ELBO 的每一項，並用蒙地卡羅抽樣核對。"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["mathtext.fontset"] = "cm"
args = sys.argv[1:]
REPORT = None
if "--report" in args:
    i = args.index("--report"); REPORT = Path(args[i + 1]); del args[i:i + 2]
OUT = Path(args[0]) if args else Path(__file__).resolve().parent.parent
OUT.mkdir(parents=True, exist_ok=True)
num = {}
r4 = lambda v: round(float(v), 4)
LOG2PI = np.log(2 * np.pi)

# ---------- [補充] 1：小步與大步的真正逆向分布 ----------
MU, S = 2.0, 0.3
grid = np.linspace(-4, 4, 8001); dx = grid[1] - grid[0]
prior = 0.5 * np.exp(-(grid + MU) ** 2 / (2 * S * S)) + 0.5 * np.exp(-(grid - MU) ** 2 / (2 * S * S))
rev = {}
for name, beta, x1 in (("small", 0.01, 1.5), ("large", 0.5, 0.0)):
    lik = np.exp(-(x1 - np.sqrt(1 - beta) * grid) ** 2 / (2 * beta))
    post = prior * lik; post /= post.sum() * dx
    m = float((grid * post).sum() * dx); sd = float(np.sqrt(((grid - m) ** 2 * post).sum() * dx))
    gauss = np.exp(-(grid - m) ** 2 / (2 * sd * sd)) / np.sqrt(2 * np.pi) / sd
    tv = 0.5 * float(np.abs(post - gauss).sum() * dx)            # total variation distance to the moment-matched normal
    peaks = int(((post[1:-1] > post[:-2]) & (post[1:-1] > post[2:]) & (post[1:-1] > 0.05 * post.max())).sum())
    rev[name] = dict(beta=beta, x1=x1, mean=round(m, 3), std=round(sd, 3), peaks=peaks, tv_to_normal=round(tv, 3), post=post, gauss=gauss)
num["reverse_shape"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("post", "gauss")} for k, v in rev.items()}
assert rev["small"]["peaks"] == 1 and rev["large"]["peaks"] == 2
assert rev["small"]["tv_to_normal"] < 0.02 and rev["large"]["tv_to_normal"] > 0.3

# ---------- [補充] 2：三步小模型的 ELBO ----------
betas = np.array([0.2, 0.4, 0.6]); alphas = 1 - betas; abar = np.cumprod(alphas); T = 3
X0 = 1.0


def kl_gauss(m1, v1, m2, v2):                       # KL(N(m1,v1) || N(m2,v2)) for scalars
    return 0.5 * (np.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1)


def post_coef(t):                                    # q(x_{t-1} | x_t, x0) = N(A x0 + B x_t, V), t >= 2（公式見 KP04）
    ab_t, ab_p = abar[t - 1], abar[t - 2]
    A = np.sqrt(ab_p) * betas[t - 1] / (1 - ab_t)
    B = np.sqrt(alphas[t - 1]) * (1 - ab_p) / (1 - ab_t)
    V = (1 - ab_p) * betas[t - 1] / (1 - ab_t)
    return A, B, V


def model(scale):                                    # p(x_{t-1} | x_t) = N(c_t x_t, s2_t)
    c = scale * np.sqrt(alphas); s2 = betas.copy()
    return c, s2


def elbo_terms(scale, x0=X0):
    c, s2 = model(scale)
    LT = kl_gauss(np.sqrt(abar[-1]) * x0, 1 - abar[-1], 0.0, 1.0)
    mids = {}
    for t in range(2, T + 1):
        A, B, V = post_coef(t)
        mt, vt = np.sqrt(abar[t - 1]) * x0, 1 - abar[t - 1]          # x_t ~ q(x_t | x0)
        k = B - c[t - 1]
        e_sq = (A * x0 + k * mt) ** 2 + k * k * vt                 # E[(mu_tilde - c x_t)^2]
        mids[t] = 0.5 * (np.log(s2[t - 1] / V) + (V + e_sq) / s2[t - 1] - 1)
    m1, v1 = np.sqrt(abar[0]) * x0, 1 - abar[0]
    L0 = -0.5 * (LOG2PI + np.log(s2[0])) - ((x0 - c[0] * m1) ** 2 + c[0] ** 2 * v1) / (2 * s2[0])
    elbo = -LT - sum(mids.values()) + L0
    var = 1.0                                                      # marginal of x0 under the model: start var(x_T)=1
    for t in range(T, 0, -1):
        var = c[t - 1] ** 2 * var + s2[t - 1]
    logp = -0.5 * (LOG2PI + np.log(var)) - x0 * x0 / (2 * var)
    return dict(KL_T=LT, KL_mid=mids, recon=L0, elbo=elbo, logp=logp, model_var_x0=var)


def elbo_mc(scale, x0=X0, n=400000, seed=0):        # E_q[log p(x_{0:T}) - log q(x_{1:T}|x0)] by sampling
    rng = np.random.default_rng(seed); c, s2 = model(scale)
    xs = [np.full(n, x0)]
    lq = np.zeros(n)
    for t in range(1, T + 1):
        nxt = np.sqrt(alphas[t - 1]) * xs[-1] + np.sqrt(betas[t - 1]) * rng.normal(size=n)
        lq += -0.5 * (LOG2PI + np.log(betas[t - 1])) - (nxt - np.sqrt(alphas[t - 1]) * xs[-1]) ** 2 / (2 * betas[t - 1])
        xs.append(nxt)
    lp = -0.5 * (LOG2PI) - xs[T] ** 2 / 2
    for t in range(T, 0, -1):
        lp += -0.5 * (LOG2PI + np.log(s2[t - 1])) - (xs[t - 1] - c[t - 1] * xs[t]) ** 2 / (2 * s2[t - 1])
    v = lp - lq
    return float(v.mean()), float(v.std() / np.sqrt(n))


res = {}
for name, scale in (("A_exact", 1.0), ("B_mean_x0.8", 0.8)):
    e = elbo_terms(scale); mc, se = elbo_mc(scale)
    res[name] = dict(mean_scale=scale, KL_T=r4(e["KL_T"]), KL_2=r4(e["KL_mid"][2]), KL_3=r4(e["KL_mid"][3]),
                     recon=r4(e["recon"]), elbo=r4(e["elbo"]), log_p=r4(e["logp"]), gap=r4(e["logp"] - e["elbo"]),
                     model_var_x0=r4(e["model_var_x0"]), elbo_monte_carlo=r4(mc), mc_std_error=r4(se))
    assert abs(mc - e["elbo"]) < 4 * se + 1e-3
    assert e["elbo"] <= e["logp"] + 1e-9
num["tiny_model"] = dict(betas=betas.tolist(), alphas=[r4(a) for a in alphas], abar=[r4(a) for a in abar],
                         data="N(0, 1)", x0=X0, models=res, mc_samples=400000, mc_seed=0)
A_, B_ = res["A_exact"], res["B_mean_x0.8"]
assert abs(A_["gap"]) < 1e-4 and abs(A_["log_p"] - (-0.5 * LOG2PI - 0.5)) < 1e-4
assert A_["KL_2"] > 0.01 and A_["KL_3"] > 0.01 and B_["gap"] > 0.01   # 完美模型的每步 KL 仍大於 0：p 看不到 x0
assert B_["elbo"] < A_["elbo"]

# DDPM 排程（T=1000）下第一項 KL(q(x_T | x0) || N(0,1))，x0 = 1 的一維情形
ab_T = float(np.prod(1 - np.linspace(1e-4, 0.02, 1000)))
num["ddpm_KL_T_x0_1"] = float(f"{kl_gauss(np.sqrt(ab_T), 1 - ab_T, 0.0, 1.0):.3g}")
assert num["ddpm_KL_T_x0_1"] < 1e-4

num["practice3_abar5_beta05"] = r4(0.5 ** 5)
assert num["practice3_abar5_beta05"] == 0.0312

# ---------- figure 1: lec14-kp03-two-chains ----------
fig, ax = plt.subplots(figsize=(10.5, 3.2)); ax.set_xlim(0, 10.5); ax.set_ylim(0, 3.2); ax.axis("off")
xs_ = [0.4, 2.6, 4.8, 7.0, 9.2]; labels = [r"$x_0$", r"$x_1$", r"$\cdots$", r"$x_{T-1}$", r"$x_T$"]
fills = ["#DBEAFE", "#E0E7FF", "#FFFFFF", "#F3E8FF", "#FEF3C7"]
for x, lab, fc in zip(xs_, labels, fills):
    if lab == r"$\cdots$":
        ax.text(x + 0.45, 1.6, lab, ha="center", va="center", fontsize=16); continue
    ax.add_patch(FancyBboxPatch((x, 1.2), 0.9, 0.8, boxstyle="round,pad=0.04", fc=fc, ec="#374151", lw=1.3))
    ax.text(x + 0.45, 1.6, lab, ha="center", va="center", fontsize=14)
for i in range(4):
    a, b = xs_[i] + 0.95, xs_[i + 1] - 0.05
    ax.annotate("", (b, 2.25), (a, 2.25), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2563EB", connectionstyle="arc3,rad=-0.25"))
    ax.annotate("", (a, 0.95), (b, 0.95), arrowprops=dict(arrowstyle="->", lw=1.5, color="#7C3AED", connectionstyle="arc3,rad=-0.25"))
ax.text(5.25, 2.95, r"forward (fixed): $q(x_t\mid x_{t-1})=\mathcal{N}(\sqrt{\alpha_t}\,x_{t-1},\ \beta_t I)$", ha="center", fontsize=11, color="#1D4ED8")
ax.text(5.25, 0.15, r"backward (learned): $p_\theta(x_{t-1}\mid x_t)=\mathcal{N}(\mu_\theta(x_t,t),\ \sigma_t^2 I)$, start $p(x_T)=\mathcal{N}(0,I)$", ha="center", fontsize=11, color="#6D28D9")
fig.tight_layout(); fig.savefig(OUT / "lec14-kp03-two-chains.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec14-kp03-small-vs-large-step ----------
fig, axs = plt.subplots(1, 2, figsize=(11.0, 3.8))
for ax, key, title in ((axs[0], "small", r"small step: $\beta=0.01$, observed $x_1=1.5$"),
                       (axs[1], "large", r"large step: $\beta=0.5$, observed $x_1=0$")):
    r = rev[key]
    ax.plot(grid, r["post"], color="#7C3AED", lw=2, label=r"true $q(x_0\mid x_1)$")
    ax.plot(grid, r["gauss"], color="#F59E0B", lw=1.6, ls="--", label="best single normal")
    ax.set_title(title, fontsize=10.5); ax.set_xlabel("$x_0$"); ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(0, max(r["post"].max(), r["gauss"].max()) * 1.15)
axs[0].set_ylabel("density")
axs[0].legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "lec14-kp03-small-vs-large-step.png", dpi=200); plt.close(fig)

# ---------- figure 3: lec14-kp03-elbo-terms ----------
fig, ax = plt.subplots(figsize=(9.6, 4.0))
cats = [r"$-\mathrm{KL}_T$", r"$-\mathrm{KL}_{t=2}$", r"$-\mathrm{KL}_{t=3}$", "recon.\n" + r"$\log p_\theta(x_0\mid x_1)$", "ELBO\n(sum)", r"$\log p_\theta(x_0)$"]
xpos = np.arange(len(cats)); w = 0.36
for k, (name, col) in enumerate((("A_exact", "#2563EB"), ("B_mean_x0.8", "#F59E0B"))):
    r = res[name]
    vals = [-r["KL_T"], -r["KL_2"], -r["KL_3"], r["recon"], r["elbo"], r["log_p"]]
    lab = "model A: exact reverse" if k == 0 else r"model B: means $\times0.8$"
    bars = ax.bar(xpos + (k - 0.5) * w, vals, w, color=col, alpha=0.85, label=lab)
    for xb, v in zip(xpos + (k - 0.5) * w, vals):
        ax.text(xb, v - 0.06 if v < 0 else v + 0.02, f"{v:.3f}", ha="center", va="top" if v < 0 else "bottom", fontsize=8)
ax.axhline(0, color="#374151", lw=0.8)
ax.set_xticks(xpos); ax.set_xticklabels(cats, fontsize=9.5)
ax.set_ylim(-1.95, 0.25); ax.set_ylabel("value (nats)")
ax.legend(loc="lower left", fontsize=9)
ax.set_title(r"3-step model, data $\mathcal{N}(0,1)$, $x_0=1$: ELBO $\leq \log p_\theta(x_0)$", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "lec14-kp03-elbo-terms.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
