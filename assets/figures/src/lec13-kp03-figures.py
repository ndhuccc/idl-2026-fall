"""lec13-kp03-figures.py — 第13週 KP03 講義的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec13-kp03-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
[補充] 例子沿用 KP01：真資料 N(2, 0.5^2)；生成器的分布用常態 N(a, s^2)。積分用格點（-12 到 16，200001 點）。"""
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
N = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))
X = np.linspace(-12, 16, 200001); DX = X[1] - X[0]


def kl(p, q):
    return float(np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300) / np.maximum(q, 1e-300)), 0.0)) * DX)


def jsd(p, q):
    m = (p + q) / 2
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ---------- pointwise maximisation f(y) = a log y + b log(1-y) ----------
f = lambda y, a, b: a * math.log(y) + b * math.log(1 - y)
pw = []
for a, b in ((0.6, 0.2), (0.3, 0.3), (0.0001, 0.5)):
    ys = a / (a + b); pw.append(dict(a=a, b=b, y_star=round(ys, 4), f_at_y_star=round(f(ys, a, b), 4)))
num["pointwise"] = pw
num["pointwise_check"] = dict(f_06_02_at_05=round(f(0.5, 0.6, 0.2), 4), f_06_02_at_09=round(f(0.9, 0.6, 0.2), 4))

# ---------- optimal discriminator for p_data = N(2, 0.5^2), p_g = N(0, 1) ----------
w_star = [-2 * 4 + math.log(2), 8.0, -1.5]          # logit of D* = log p_data - log p_g = -1.5 x^2 + 8 x - 8 + log 2
pd, pg = N(X, 2, 0.5), N(X, 0, 1)
Dstar = pd / (pd + pg)
assert np.allclose(Dstar[np.abs(X) < 6], sig(w_star[0] + w_star[1] * X + w_star[2] * X * X)[np.abs(X) < 6], atol=1e-9)
# fit a logistic regression with features (1, x, x^2) on many samples by Newton's method
rng = np.random.default_rng(3); n = 200000
xs = np.r_[rng.normal(2, 0.5, n), rng.normal(0, 1, n)]; ys = np.r_[np.ones(n), np.zeros(n)]
F = np.stack([np.ones_like(xs), xs, xs * xs], 1); w = np.zeros(3)
for _ in range(30):
    p = sig(F @ w); g = F.T @ (ys - p); H = (F * (p * (1 - p))[:, None]).T @ F; w = w + np.linalg.solve(H, g)
V_direct = float(np.sum(pd * np.log(Dstar + 1e-300)) * DX + np.sum(pg * np.log(1 - Dstar + 1e-300)) * DX)
J = jsd(pd, pg)
assert abs(V_direct - (-math.log(4) + 2 * J)) < 1e-6
num["optimal_D_example"] = dict(p_data="N(2, 0.5^2)", p_g="N(0, 1)", w_star=[round(v, 4) for v in w_star],
                                w_fitted_logistic=[round(float(v), 3) for v in w], fit_samples_each=n, fit_seed=3,
                                D_star_at={str(x0): round(float(sig(w_star[0] + w_star[1] * x0 + w_star[2] * x0 * x0)), 4) for x0 in (0.0, 1.0, 2.0, 3.0)},
                                V_D_star=round(V_direct, 4), JSD=round(J, 4))

# ---------- JSD between N(2, 0.5^2) and N(a, 0.5^2) as the generator moves ----------
shifts = {}
for a in (2, 1.5, 1, 0, -1, -3):
    j = jsd(pd, N(X, a, 0.5)); shifts[str(a)] = dict(JSD=round(j, 4), V=round(-math.log(4) + 2 * j, 4))
num["jsd_shift"] = dict(p_data="N(2, 0.5^2)", p_g="N(a, 0.5^2)", by_a=shifts, log2=round(math.log(2), 4), log4=round(math.log(4), 4))
num["kl_check_a1"] = dict(KL_pd_m=round(kl(pd, (pd + N(X, 1, 0.5)) / 2), 4), KL_pg_m=round(kl(N(X, 1, 0.5), (pd + N(X, 1, 0.5)) / 2), 4))

# ---------- figure 1: lec13-kp03-pointwise ----------
y = np.linspace(0.005, 0.995, 600)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for (a, b), col in (((0.6, 0.2), "#1D4ED8"), ((0.3, 0.3), "#B45309")):
    ax.plot(y, a * np.log(y) + b * np.log(1 - y), color=col, lw=2.4, label=rf"$a={a},\ b={b}$")
    ys_ = a / (a + b); ax.plot([ys_], [f(ys_, a, b)], "o", color=col, ms=7)
    ax.annotate(rf"$y^*={ys_:.2f}$", xy=(ys_, f(ys_, a, b)), xytext=(ys_ - 0.12, f(ys_, a, b) + 0.35), color=col, fontsize=10.5,
                arrowprops=dict(arrowstyle="-", color=col, lw=1))
ax.set_xlim(0, 1); ax.set_ylim(-2.5, 0.2); ax.grid(alpha=0.3)
ax.set_xlabel(r"$y=D(\mathbf{x})$"); ax.set_ylabel(r"$a\log y+b\log(1-y)$"); ax.legend(loc="lower center", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "lec13-kp03-pointwise.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec13-kp03-optimal-d ----------
xx = np.linspace(-3, 4.5, 800)
fig, ax = plt.subplots(figsize=(7.4, 3.9))
ax.fill_between(xx, N(xx, 2, 0.5), color="#93C5FD", alpha=0.55, label=r"$p_{\mathrm{data}}=\mathcal{N}(2,0.5^2)$")
ax.plot(xx, N(xx, 0, 1), color="#B45309", lw=2.2, label=r"$p_g=\mathcal{N}(0,1)$")
ax.set_ylim(0, 1.0); ax.set_xlim(-3, 4.5); ax.set_xlabel(r"$x$"); ax.set_ylabel("density"); ax.grid(alpha=0.25)
ax2 = ax.twinx()
ax2.plot(xx, sig(w_star[0] + w_star[1] * xx + w_star[2] * xx * xx), color="#7C3AED", lw=2.6, label=r"$D^*=p_{\mathrm{data}}/(p_{\mathrm{data}}+p_g)$")
ax2.plot(xx, sig(w[0] + w[1] * xx + w[2] * xx * xx), color="#F472B6", lw=1.8, ls="--", label="logistic fit on samples")
ax2.axhline(0.5, color="#94A3B8", lw=1, ls=":"); ax2.text(-2.9, 0.52, r"$D=0.5$", color="#64748B", fontsize=9); ax2.set_ylim(0, 1.05); ax2.set_ylabel(r"$D(x)$", color="#7C3AED")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.6)
fig.tight_layout(); fig.savefig(OUT / "lec13-kp03-optimal-d.png", dpi=200); plt.close(fig)

# ---------- figure 3: lec13-kp03-jsd-shift ----------
aa = np.linspace(-4, 8, 241)
js = np.array([jsd(pd, N(X, a, 0.5)) for a in aa])
fig, ax = plt.subplots(figsize=(7.4, 3.8))
ax.plot(aa, js, color="#0F766E", lw=2.6, label=r"$\mathrm{JSD}(p_{\mathrm{data}}\,\Vert\,p_g)$")
ax.axhline(math.log(2), color="#BE123C", ls="--", lw=1.4); ax.text(7.9, math.log(2) + 0.02, r"$\log 2=0.693$", color="#BE123C", ha="right")
for a in (2, 1, 0, -1):
    j = shifts[str(a)]["JSD"]; ax.plot([a], [j], "o", color="#0F766E"); ax.text(a + 0.18, {2: 0.03, 1: 0.30, 0: 0.56, -1: 0.62}[a], rf"${j:.3f}$", fontsize=9.5)
ax.axvspan(-4, -0.5, color="#FDE68A", alpha=0.35); ax.axvspan(4.5, 8, color="#FDE68A", alpha=0.35)
ax.text(-3.9, 0.35, "almost no overlap:\nnearly flat,\nnear-zero gradient", fontsize=9); ax.text(4.6, 0.35, "almost no overlap:\nnearly flat,\nnear-zero gradient", fontsize=9)
ax.set_xlim(-4, 8); ax.set_ylim(-0.03, 0.78); ax.grid(alpha=0.3)
ax.set_xlabel(r"generator mean $a$  ($p_g=\mathcal{N}(a,0.5^2)$, real mean $2$)"); ax.set_ylabel("Jensen-Shannon divergence")
ax.legend(loc="lower left", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "lec13-kp03-jsd-shift.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
