"""lec13-kp08-figures.py — 第13週 KP08（選讀）講義的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec13-kp08-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
[補充] (1) 兩點分布 p=(0.5,0.5)、q=(0.8,0.2) 的幾種 f-散度，用 f 公式與直接公式各算一次；
(2) Pearson chi^2 的 Fenchel 共軛與下界：任何 T 都是下界，T=f'(p/q) 時相等；
(3) 從 1 維雜訊映到 2 維的生成器：樣本全落在一條曲線上；用 MLE 時真資料點的概似是 0。"""
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

# ---------- (1) f-divergences on a two-point example ----------
p = np.array([0.5, 0.5]); q = np.array([0.8, 0.2]); u = p / q
F = {
    "KL(p||q)":        lambda u: u * np.log(u),
    "reverse KL(q||p)": lambda u: -np.log(u),
    "Pearson chi2":    lambda u: (u - 1) ** 2,
    "total variation": lambda u: 0.5 * np.abs(u - 1),
    "2 JSD":           lambda u: u * np.log(u) - (u + 1) * np.log((u + 1) / 2),
}
via_f = {k: float(np.sum(q * f(u))) for k, f in F.items()}
m = 0.5 * (p + q)
direct = {"KL(p||q)": float(np.sum(p * np.log(p / q))), "reverse KL(q||p)": float(np.sum(q * np.log(q / p))),
          "Pearson chi2": float(np.sum((p - q) ** 2 / q)), "total variation": float(0.5 * np.sum(np.abs(p - q))),
          "2 JSD": float(np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))}
for k in F: assert abs(via_f[k] - direct[k]) < 1e-12
num["f_divergences"] = dict(p=p.tolist(), q=q.tolist(), ratio_u=u.tolist(), values={k: round(v, 4) for k, v in via_f.items()},
                           JSD=round(direct["2 JSD"] / 2, 4))

# ---------- (2) Fenchel conjugate of f(u)=(u-1)^2 and the variational lower bound ----------
fstar = lambda t: t + t * t / 4          # sup_u { u t - (u-1)^2 }, attained at u = 1 + t/2
tt = np.linspace(-3, 4, 7001)
uu = np.linspace(-20, 20, 400001)
for t in (-2.0, 0.0, 1.5, 3.0):          # brute-force check of the conjugate
    assert abs(np.max(uu * t - (uu - 1) ** 2) - fstar(t)) < 1e-6
Tstar = 2 * (u - 1)                      # f'(p/q)
bound = lambda T: float(np.sum(p * T) - np.sum(q * fstar(T)))
cs = np.linspace(0, 2, 201); bvals = np.array([bound(c * Tstar) for c in cs])
num["fenchel_chi2"] = dict(fstar="t + t^2/4", Tstar=Tstar.round(4).tolist(), bound_at_Tstar=round(bound(Tstar), 4),
                           chi2=round(via_f["Pearson chi2"], 4), bound_at_T_zero=round(bound(0 * Tstar), 4),
                           bound_at_half=round(bound(0.5 * Tstar), 4), bound_at_1_5=round(bound(1.5 * Tstar), 4),
                           best_scale=round(float(cs[np.argmax(bvals)]), 2))
assert abs(bound(Tstar) - via_f["Pearson chi2"]) < 1e-12

# ---------- (3) a generator from 1-D noise into 2-D ----------
rng = np.random.default_rng(0)
xr = rng.normal(size=(600, 2)) * [1.0, 0.6] @ np.array([[0.8, 0.6], [-0.6, 0.8]])
z = rng.uniform(-1, 1, 600)
G = lambda z: np.stack([2.2 * z, 0.9 * np.sin(2.4 * z)], 1)
xg = G(z)
zz = np.linspace(-1, 1, 20001); curve = G(zz)
dmin = np.array([np.min(np.linalg.norm(curve - x, axis=1)) for x in xr])
num["manifold"] = dict(real="600 points, 2-D Gaussian", generator="z ~ U(-1,1) (1-D), G(z)=(2.2 z, 0.9 sin 2.4 z)",
                       share_real_within_0p01_of_curve=round(float((dmin < 0.01).mean()), 4),
                       median_distance_real_to_curve=round(float(np.median(dmin)), 3))

# ---------- figure 1: lec13-kp08-f-functions ----------
fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.2))
ax = axs[0]; ug = np.linspace(0.05, 4, 800)
cols = {"KL(p||q)": "#2563EB", "reverse KL(q||p)": "#BE123C", "Pearson chi2": "#15803D", "total variation": "#B45309", "2 JSD": "#7C3AED"}
labs = {"KL(p||q)": r"KL: $u\log u$", "reverse KL(q||p)": r"reverse KL: $-\log u$", "Pearson chi2": r"Pearson $\chi^2$: $(u-1)^2$",
        "total variation": r"total variation: $\frac{1}{2}|u-1|$", "2 JSD": r"$2\,$JSD: $u\log u-(u+1)\log\frac{u+1}{2}$"}
for k in F: ax.plot(ug, F[k](ug), color=cols[k], lw=2.2, label=labs[k])
ax.axvline(1, color="#94A3B8", ls=":", lw=1.2); ax.text(1.06, 2.75, r"$u=1$: all are $0$", fontsize=9, color="#475569")
ax.set_xlim(0, 4); ax.set_ylim(-0.6, 3); ax.set_xlabel(r"density ratio $u=p(x)/q(x)$"); ax.set_ylabel(r"$f(u)$")
ax.set_title("five convex functions with $f(1)=0$", fontsize=10); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=8.2, ncol=2); ax.grid(alpha=0.3)
ax = axs[1]; keys = list(F); vals = [via_f[k] for k in keys]
bars = ax.bar(range(len(keys)), vals, color=[cols[k] for k in keys], alpha=0.85)
for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
ax.set_xticks(range(len(keys))); ax.set_xticklabels(["KL", "reverse\nKL", r"$\chi^2$", "TV", "2 JSD"], fontsize=9)
ax.set_ylim(0, 0.68); ax.set_ylabel(r"$D_f(p\,\Vert\,q)=\sum_x q(x)\,f(p(x)/q(x))$")
ax.set_title(r"same pair $p=(0.5,0.5)$, $q=(0.8,0.2)$: five different numbers", fontsize=10); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(OUT / "lec13-kp08-f-functions.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- figure 2: lec13-kp08-fenchel-bound ----------
fig, ax = plt.subplots(figsize=(8.4, 4.0))
ax.plot(cs, bvals, color="#15803D", lw=2.4, label=r"lower bound $\mathbb{E}_p[T]-\mathbb{E}_q[f^*(T)]$ with $T=c\cdot T^*$")
ax.axhline(via_f["Pearson chi2"], color="#334155", ls="--", lw=1.4, label=r"true $\chi^2(p\,\Vert\,q)=0.5625$")
ax.plot([1], [bound(Tstar)], "o", color="#BE123C", ms=8); ax.annotate(r"$c=1$: $T=f'(p/q)$, bound $=$ divergence", xy=(1, bound(Tstar)), xytext=(1.12, 0.2),
                                                                    fontsize=9, color="#BE123C", arrowprops=dict(arrowstyle="->", color="#BE123C"))
ax.set_xlim(0, 2); ax.set_ylim(-0.05, 0.7); ax.set_xlabel(r"scale $c$ of the test function $T$"); ax.set_ylabel("value")
ax.legend(loc="lower center", fontsize=8.8); ax.grid(alpha=0.3)
ax.set_title(r"every $T$ gives a lower bound; the best $T$ reaches the divergence", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "lec13-kp08-fenchel-bound.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- figure 3: lec13-kp08-implicit-manifold ----------
fig, ax = plt.subplots(figsize=(6.8, 5.0))
ax.scatter(xr[:, 0], xr[:, 1], s=8, alpha=0.4, color="#2563EB", label="real data (2-D cloud)")
ax.scatter(xg[:, 0], xg[:, 1], s=8, alpha=0.6, color="#BE123C", label=r"generator samples $G(z)$, $z$ 1-D")
ax.plot(curve[:, 0], curve[:, 1], color="#BE123C", lw=0.8, alpha=0.6)
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); ax.legend(loc="upper left", fontsize=8.8)
ax.set_title("all fakes lie on one curve: in 2-D they have no density", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "lec13-kp08-implicit-manifold.png", dpi=200, bbox_inches="tight"); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
