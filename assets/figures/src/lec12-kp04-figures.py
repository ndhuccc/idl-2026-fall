"""lec12-kp04-figures.py — 第12週 KP04 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec12-kp04-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
例子 [補充]：8 個數，兩群常態混合（標準差固定 1、兩群各占一半），未知的是兩群的中心；潛在變數是「每個點屬於哪一群」。"""
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

X = np.array([-2.3, -1.7, -1.2, -0.4, 0.6, 1.3, 1.9, 2.6])
N = lambda x, m: np.exp(-0.5 * (x - m) ** 2) / math.sqrt(2 * math.pi)


def loglik(mu):
    return float(np.sum(np.log(0.5 * N(X, mu[0]) + 0.5 * N(X, mu[1]))))


def resp(mu):
    a, b = 0.5 * N(X, mu[0]), 0.5 * N(X, mu[1])
    return a / (a + b)      # probability each point belongs to group 1


def elbo(r, mu):            # L(q, theta) with q = responsibilities r (group 1) and 1-r (group 2)
    t1 = r * (np.log(0.5 * N(X, mu[0])) - np.log(np.clip(r, 1e-300, 1)))
    t2 = (1 - r) * (np.log(0.5 * N(X, mu[1])) - np.log(np.clip(1 - r, 1e-300, 1)))
    return float(np.sum(t1 + t2))


mu = np.array([-0.2, 0.3])
hist = [dict(t=0, mu=mu.round(4).tolist(), loglik=round(loglik(mu), 4))]
stairs = []                 # (label, ELBO, loglik at current theta)
r = resp(mu)
first_E = dict(x0=float(X[3]), r_group1=round(float(r[3]), 4),
               num=round(float(0.5 * N(X[3], mu[0])), 4), den=round(float(0.5 * N(X[3], mu[0]) + 0.5 * N(X[3], mu[1])), 4),
               all_r=r.round(4).tolist())
for t in range(1, 21):
    r = resp(mu)                                    # E-step
    e_after_E = elbo(r, mu); ll_old = loglik(mu)
    assert abs(e_after_E - ll_old) < 1e-9           # E-step closes the gap
    new = np.array([np.sum(r * X) / np.sum(r), np.sum((1 - r) * X) / np.sum(1 - r)])   # M-step
    e_after_M = elbo(r, new); ll_new = loglik(new)
    assert e_after_M >= e_after_E - 1e-12 and ll_new >= ll_old - 1e-12 and ll_new >= e_after_M - 1e-12
    stairs.append(dict(t=t, elbo_after_E=round(e_after_E, 4), elbo_after_M=round(e_after_M, 4), loglik_after_M=round(ll_new, 4),
                       gap_after_M=round(ll_new - e_after_M, 4)))
    mu = new
    hist.append(dict(t=t, mu=mu.round(4).tolist(), loglik=round(ll_new, 4)))
num["data"] = X.tolist()
num["init_mu"] = [-0.2, 0.3]
num["first_E_step_point_x4"] = first_E
num["first_M_step_mu"] = hist[1]["mu"]
num["history"] = hist[:6] + hist[-1:]
num["stairs_first3"] = stairs[:3]
num["final"] = hist[-1]
num["monotone"] = all(hist[k + 1]["loglik"] >= hist[k]["loglik"] for k in range(len(hist) - 1))
# hard-label comparison: if the labels were known (left 4 / right 4), the MLE is just two group means
num["known_labels_means"] = [round(float(X[:4].mean()), 4), round(float(X[4:].mean()), 4)]

# ---------- practice answers ----------
d = (2.6 + 0.2) ** 2 / 2 - (2.6 - 0.3) ** 2 / 2
num["practice"] = dict(q1_exponent=round(d, 4), q1_r=round(1 / (1 + math.exp(d)), 3),
                       q2=[round((0 * 1 + 2 * 0.5 + 4 * 0) / 1.5, 3), round((0 * 0 + 2 * 0.5 + 4 * 1) / 1.5, 3)])
assert abs(num["practice"]["q1_r"] - first_E["all_r"][7]) < 1e-3

# ---------- figure 1: lec12-kp04-em-fit ----------
xs = np.linspace(-4.5, 4.5, 500)
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, k in zip(axs, (0, 1, 20)):
    m = hist[k]["mu"]; rr = resp(np.array(m))
    ax.plot(xs, 0.5 * N(xs, m[0]), color="#1D4ED8", lw=2, label="group 1")
    ax.plot(xs, 0.5 * N(xs, m[1]), color="#D97706", lw=2, label="group 2")
    ax.scatter(X, np.zeros_like(X) - 0.02, c=rr, cmap="coolwarm_r", vmin=0, vmax=1, s=55, edgecolor="k", zorder=5)
    ax.set_title(rf"round $t={k}$: $\mu_1={m[0]:.2f}$, $\mu_2={m[1]:.2f}$" + "\n" + rf"$\log p(X)={hist[k]['loglik']:.4f}$", fontsize=10)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-0.06, 0.25); ax.set_xlabel(r"$x$"); ax.grid(alpha=0.3)
axs[0].set_ylabel("density (each group weighted 0.5)"); axs[0].legend(loc="upper left", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "lec12-kp04-em-fit.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec12-kp04-em-stairs ----------
fig, ax = plt.subplots(figsize=(7.4, 4.2))
T = 6
ll = [hist[k]["loglik"] for k in range(T + 1)]
ax.plot(range(T + 1), ll, color="#15803D", lw=2.2, marker="o", label=r"$\log p(X\mid\theta^{(t)})$")
px, py = [], []
for s in stairs[:T]:
    t = s["t"]
    px += [t - 1, t]; py += [s["elbo_after_E"], s["elbo_after_M"]]
ax.plot(px, py, color="#0F766E", lw=1.8, ls="--", marker="s", ms=4, label=r"ELBO after E-step (left) and after M-step (right)")
for s in stairs[:T]:
    t = s["t"]
    if s["gap_after_M"] > 0.01:
        ax.annotate("", xy=(t, s["loglik_after_M"]), xytext=(t, s["elbo_after_M"]), arrowprops=dict(arrowstyle="<->", color="#B91C1C", lw=1.2))
ax.text(1.06, (stairs[0]["loglik_after_M"] + stairs[0]["elbo_after_M"]) / 2, f"gap {stairs[0]['gap_after_M']:.4f}", color="#B91C1C", fontsize=9, va="center")
ax.set_xlabel(r"round $t$"); ax.set_ylabel("value"); ax.set_xticks(range(T + 1)); ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8.8)
fig.tight_layout(); fig.savefig(OUT / "lec12-kp04-em-stairs.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
