"""lec12-kp03-figures.py — 第12週 KP03 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec12-kp03-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
模型沿用 KP02：z ~ N(0,1)，x|z ~ N(z, 0.5^2)，觀察 x = 1.2；近似後驗 q = N(m, s^2)。所有量都有公式解。"""
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

sig2, x = 0.25, 1.2
mu_p, v_p = x / (1 + sig2), sig2 / (1 + sig2)          # true posterior N(0.96, 0.2)
log_px = -0.5 * math.log(2 * math.pi * (1 + sig2)) - x ** 2 / (2 * (1 + sig2))


def recon(m, s2):      # E_q[log p(x|z)]
    return -0.5 * math.log(2 * math.pi * sig2) - ((x - m) ** 2 + s2) / (2 * sig2)


def kl_prior(m, s2):   # KL(q || N(0,1))
    return 0.5 * (s2 + m ** 2 - 1 - math.log(s2))


def kl_post(m, s2):    # KL(q || N(mu_p, v_p))
    return 0.5 * (s2 / v_p + (m - mu_p) ** 2 / v_p - 1 + math.log(v_p / s2))


def elbo(m, s2):
    return recon(m, s2) - kl_prior(m, s2)


# Monte Carlo check of the closed forms with samples from q
rng = np.random.default_rng(0)
rows = {}
for name, (m, s2) in {"posterior": (mu_p, v_p), "prior": (0.0, 1.0), "shifted": (0.5, v_p)}.items():
    e, k = elbo(m, s2), kl_post(m, s2)
    assert abs(e + k - log_px) < 1e-12
    zs = rng.normal(m, math.sqrt(s2), 200000)
    logq = -0.5 * np.log(2 * np.pi * s2) - (zs - m) ** 2 / (2 * s2)
    logjoint = -0.5 * np.log(2 * np.pi) - zs ** 2 / 2 - 0.5 * np.log(2 * np.pi * sig2) - (x - zs) ** 2 / (2 * sig2)
    mc = float(np.mean(logjoint - logq))
    assert abs(mc - e) < 0.01
    rows[name] = dict(m=round(m, 4), s2=round(s2, 4), recon=round(recon(m, s2), 4), kl_prior=round(kl_prior(m, s2), 4),
                      elbo=round(e, 4), kl_post=round(k, 4), elbo_plus_kl=round(e + k, 4), monte_carlo_elbo=round(mc, 4))
num["log_px"] = round(log_px, 4)
num["p_x"] = round(math.exp(log_px), 4)
num["q_examples"] = rows
num["source_notes_example"] = dict(log_px=0.0, kl=0.7, elbo=-0.7)

# figure data: q = N(m, v_p) for m in a range
ms = np.linspace(-1.0, 2.5, 400)
E = np.array([elbo(m, v_p) for m in ms]); R = np.array([recon(m, v_p) for m in ms]); KP = np.array([kl_prior(m, v_p) for m in ms])
assert abs(ms[np.argmax(E)] - mu_p) < 0.01
num["argmax_m"] = round(float(ms[np.argmax(E)]), 2)

# ---------- figure 1: lec12-kp03-elbo-gap ----------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.axhline(log_px, color="#15803D", lw=2.4, label=rf"$\log p(x)={log_px:.4f}$ (does not depend on $q$)")
ax.plot(ms, E, color="#0F766E", lw=2.4, label=r"ELBO$(q)$, $q=\mathcal{N}(m,\,0.2)$")
ax.fill_between(ms, E, log_px, color="#99F6E4", alpha=0.6, label=r"gap $=D_{\mathrm{KL}}(q\,\Vert\,p(z\mid x))$")
for name, col in (("posterior", "#DC2626"), ("shifted", "#7C3AED")):
    r = rows[name]
    ax.scatter([r["m"]], [r["elbo"]], color=col, zorder=5, s=40)
ax.annotate("q = posterior: gap 0", (mu_p, log_px), xytext=(1.25, -1.35), fontsize=9, color="#DC2626",
            arrowprops=dict(arrowstyle="->", color="#DC2626"))
ax.annotate(f"m = 0.5: gap {rows['shifted']['kl_post']:.4f}", (0.5, rows["shifted"]["elbo"]), xytext=(-0.95, -2.35), fontsize=9,
            color="#7C3AED", arrowprops=dict(arrowstyle="->", color="#7C3AED"))
ax.set_xlim(-1, 2.5); ax.set_ylim(-5.5, -1.0); ax.set_xlabel(r"mean $m$ of $q$"); ax.set_ylabel("value")
ax.grid(alpha=0.3); ax.legend(loc="lower center", fontsize=8.8)
fig.tight_layout(); fig.savefig(OUT / "lec12-kp03-elbo-gap.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec12-kp03-elbo-terms ----------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(ms, R, color="#1D4ED8", lw=2.2, label=r"reconstruction $\mathbb{E}_q[\log p(x\mid z)]$")
ax.plot(ms, -KP, color="#D97706", lw=2.2, label=r"$-D_{\mathrm{KL}}(q\,\Vert\,p(z))$, stay near the prior")
ax.plot(ms, E, color="#0F766E", lw=2.8, label="ELBO = sum of the two")
ax.axvline(0, color="#D97706", lw=1, ls=":"); ax.axvline(x, color="#1D4ED8", lw=1, ls=":"); ax.axvline(mu_p, color="#0F766E", lw=1, ls=":")
ax.text(-0.35, 0.05, r"$m=0$", color="#D97706", fontsize=9); ax.text(x + 0.04, 0.05, r"$m=x=1.2$", color="#1D4ED8", fontsize=9)
ax.text(mu_p - 0.48, 0.05, r"best $m=0.96$", color="#0F766E", fontsize=9)
ax.set_xlim(-1, 2.5); ax.set_ylim(-5.6, 0.45); ax.set_xlabel(r"mean $m$ of $q$"); ax.set_ylabel("value")
ax.grid(alpha=0.3); ax.legend(loc="lower center", fontsize=8.8)
fig.tight_layout(); fig.savefig(OUT / "lec12-kp03-elbo-terms.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
