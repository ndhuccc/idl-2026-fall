"""lec12-kp02-figures.py — 第12週 KP02 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec12-kp02-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。"""
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


def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


z = np.linspace(-12, 12, 24001); dz = z[1] - z[0]   # wide grid: the KL of a wide q needs the tails

# ---------- [Added] linear Gaussian model: z ~ N(0,1), x|z ~ N(z, 0.5^2), observe x = 1.2 ----------
s, x0 = 0.5, 1.2
prior = npdf(z, 0, 1); lik = npdf(x0, z, s); joint = prior * lik
px_num = joint.sum() * dz
px_exact = float(npdf(x0, 0, math.sqrt(1 + s ** 2)))
post = joint / px_num
mu_post, var_post = x0 / (1 + s ** 2), s ** 2 / (1 + s ** 2)
assert abs(px_num - px_exact) < 1e-6
assert abs((z * post).sum() * dz - mu_post) < 1e-6 and abs(((z - mu_post) ** 2 * post).sum() * dz - var_post) < 1e-6
num["linear"] = dict(sigma=s, x=x0, p_x=round(px_exact, 4), post_mean=round(mu_post, 4), post_var=round(var_post, 4),
                     post_sd=round(math.sqrt(var_post), 4), prior_at_zero=round(float(npdf(0, 0, 1)), 4))
# a check at one point z=1: prior, likelihood, joint, posterior density
z1 = 1.0
pr1, li1 = float(npdf(z1, 0, 1)), float(npdf(x0, z1, s))
num["linear_point_z1"] = dict(prior=round(pr1, 4), likelihood=round(li1, 4), joint=round(pr1 * li1, 4),
                             posterior=round(pr1 * li1 / px_exact, 4))

# ---------- grid cost in d dimensions ----------
pts, rate = 100, 1e9
num["grid_cost"] = {str(d): pts ** d for d in (1, 2, 10)}
num["grid_years_d10"] = round(pts ** 10 / rate / (365.25 * 24 * 3600))

# ---------- [Added] nonlinear model: x|z ~ N(z^2, 0.3^2), observe x = 1  -> posterior has two peaks ----------
s2, x1 = 0.3, 1.0
joint2 = npdf(z, 0, 1) * npdf(x1, z ** 2, s2)
px2 = joint2.sum() * dz; post2 = joint2 / px2
peak = float(z[np.argmax(post2)])
mass_pos = float(post2[z > 0].sum() * dz)


logpost2 = (-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi)) + (-0.5 * ((x1 - z ** 2) / s2) ** 2 - math.log(s2 * math.sqrt(2 * math.pi))) - math.log(px2)


def kl_q_p(m, sd):      # log posterior in closed form (no underflow in the tails)
    q = npdf(z, m, sd); mask = q > 1e-300
    return float((q[mask] * (np.log(q[mask]) - logpost2[mask])).sum() * dz)


best = None              # the posterior is symmetric, so search around the right-hand peak
for m in np.linspace(0.5, 1.3, 161):
    for sd in np.linspace(0.05, 0.6, 111):
        k = kl_q_p(m, sd)
        if best is None or k < best[0]:
            best = (k, m, sd)
kl_best, m_best, sd_best = best
m_best = abs(m_best)  # the posterior is symmetric; show the right-hand peak
# the Gaussian with the posterior's own mean and variance (covers both peaks)
mean2 = float((z * post2).sum() * dz); sd2 = float(np.sqrt(((z - mean2) ** 2 * post2).sum() * dz))
kl_wide = kl_q_p(mean2, sd2)
qw = npdf(z, mean2, sd2); integ = np.where(qw > 1e-300, qw * (np.log(np.maximum(qw, 1e-300)) - logpost2), 0.0)
parts = {"between |z|<0.5": (np.abs(z) < 0.5), "peaks 0.5<=|z|<=1.5": (np.abs(z) >= 0.5) & (np.abs(z) <= 1.5), "tails |z|>1.5": (np.abs(z) > 1.5)}
num["wide_q_kl_parts"] = {k: dict(kl=round(float(integ[m].sum() * dz), 3), q_mass=round(float(qw[m].sum() * dz), 3)) for k, m in parts.items()}
qb = npdf(z, m_best, sd_best)
num["best_q_mass_z_positive"] = round(float(qb[z > 0].sum() * dz), 4)
num["nonlinear"] = dict(sigma=s2, x=x1, p_x=round(float(px2), 4), peaks=[round(-abs(peak), 3), round(abs(peak), 3)], mass_z_positive=round(mass_pos, 4),
                        best_q=dict(mean=round(abs(float(m_best)), 3), sd=round(float(sd_best), 3), kl=round(kl_best, 4)),
                        wide_q=dict(mean=round(mean2, 3), sd=round(sd2, 3), kl=round(kl_wide, 4)))
assert kl_best < kl_wide and abs(mass_pos - 0.5) < 1e-3

# ---------- [Added] KL on two outcomes: q=[0.5,0.5], p=[0.9,0.1]; not symmetric ----------
qd, pd = np.array([0.5, 0.5]), np.array([0.9, 0.1])
kl_qp = float(np.sum(qd * np.log(qd / pd))); kl_pq = float(np.sum(pd * np.log(pd / qd)))
num["kl_discrete"] = dict(q=qd.tolist(), p=pd.tolist(), terms_qp=[round(float(v), 4) for v in qd * np.log(qd / pd)],
                          kl_qp=round(kl_qp, 4), kl_pq=round(kl_pq, 4), ln2=round(math.log(2), 4))

# ---------- practice answers ----------
num["practice"] = dict(q1_post=[2.5 / (1 + s ** 2), s ** 2 / (1 + s ** 2)],
                       q2_kl=round(0.8 * math.log(0.8 / 0.5) + 0.2 * math.log(0.2 / 0.5), 4),
                       q3=dict(diag=8, full=4 + 4 * 5 // 2))

# ---------- parameter count: diagonal vs full covariance ----------
num["param_count"] = {str(d): dict(diag=2 * d, full=d + d * (d + 1) // 2) for d in (2, 10)}

# ---------- figure 1: lec12-kp02-bayes-posterior ----------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(z, prior, color="#64748B", lw=2, ls="--", label=r"prior $p(z)=\mathcal{N}(0,1)$")
ax.plot(z, lik, color="#D97706", lw=2, label=r"likelihood $p(x{=}1.2\mid z)$ as a function of $z$")
ax.plot(z, post, color="#1D4ED8", lw=2.6, label=rf"posterior $p(z\mid x{{=}}1.2)=\mathcal{{N}}({mu_post:.2f},\,{var_post:.2f})$")
ax.axvline(mu_post, color="#1D4ED8", lw=1, ls=":")
ax.scatter([z1, z1, z1], [pr1, li1, pr1 * li1 / px_exact], color=["#64748B", "#D97706", "#1D4ED8"], zorder=5, s=30)
ax.set_xlim(-3, 3.5); ax.set_ylim(0, 1.05); ax.set_xlabel(r"$z$"); ax.set_ylabel("density")
ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8.8)
fig.tight_layout(); fig.savefig(OUT / "lec12-kp02-bayes-posterior.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec12-kp02-best-gaussian ----------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.fill_between(z, post2, color="#BFDBFE", label=r"true posterior $p(z\mid x{=}1)$, $x\mid z\sim\mathcal{N}(z^2,0.3^2)$")
ax.plot(z, post2, color="#1D4ED8", lw=1.5)
ax.plot(z, npdf(z, m_best, sd_best), color="#15803D", lw=2.6,
        label=rf"best Gaussian $q$: $\mathcal{{N}}({abs(m_best):.2f},\,{sd_best:.2f}^2)$, KL$={kl_best:.2f}$")
ax.plot(z, npdf(z, mean2, sd2), color="#B91C1C", lw=2, ls="--",
        label=rf"wide Gaussian: $\mathcal{{N}}({mean2:.2f},\,{sd2:.2f}^2)$, KL$={kl_wide:.2f}$")
ax.set_xlim(-2.5, 2.5); ax.set_ylim(0, 2.8); ax.set_xlabel(r"$z$"); ax.set_ylabel("density")
ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8.8)
fig.tight_layout(); fig.savefig(OUT / "lec12-kp02-best-gaussian.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
