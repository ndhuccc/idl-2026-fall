"""lec12-kp05-figures.py — 第12週 KP05 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec12-kp05-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
需要同資料夾的 lec12_vae_toy.py（小 VAE，numpy 手寫反向傳播）。"""
import json, sys, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lec12_vae_toy as T  # noqa: E402

plt.rcParams["mathtext.fontset"] = "cm"
args = sys.argv[1:]
REPORT = None
if "--report" in args:
    i = args.index("--report"); REPORT = Path(args[i + 1]); del args[i:i + 2]
OUT = Path(args[0]) if args else Path(__file__).resolve().parent.parent
OUT.mkdir(parents=True, exist_ok=True)
num = {}

# ---------- [Added] one forward pass by hand: d = 2, p = 1 ----------
x = np.array([1.0, 2.0])
mu = 0.5 * x[0] + 0.25 * x[1]                 # encoder mean head
logvar = math.log(0.25)                        # encoder log-variance head (here a constant)
sd = math.exp(0.5 * logvar); eps = 0.6
z = mu + sd * eps
xh = np.array([0.8, 1.6]) * z                  # decoder
sq = float(np.sum((x - xh) ** 2))
recon_nll = 0.5 * sq + math.log(2 * math.pi)   # -log N(x; x_hat, I) in 2 dims
kl = 0.5 * (sd ** 2 + mu ** 2 - 1 - logvar)
num["forward_pass"] = dict(x=x.tolist(), mu=mu, logvar=round(logvar, 4), sigma=sd, eps=eps, z=round(z, 4), x_hat=xh.round(4).tolist(),
                           squared_error=round(sq, 4), recon_nll=round(recon_nll, 4), kl=round(kl, 4), neg_elbo_one_sample=round(recon_nll + kl, 4))
# generation: z from the prior
num["generate_example"] = dict(z=-0.5, x_hat=(np.array([0.8, 1.6]) * -0.5).tolist())

# ---------- parameter counts of the encoder's covariance head ----------
num["cov_params"] = {str(p): dict(diag=p, full=p * (p + 1) // 2) for p in (4, 32, 100)}

# ---------- [Added] train the toy VAE ----------
num["grad_check_max_rel_err"] = float(f"{T.grad_check():.2e}")
X, P, hist = T.train()
_, MU, LV = T.encode(X, P); SD = np.exp(0.5 * LV)
mu1, mu2 = MU[:150, 0], MU[150:, 0]
rng = np.random.default_rng(7); zs = rng.normal(size=(2000, 1)); _, xs = T.decode(zs, P)
r = np.linalg.norm(xs, axis=1); ang = np.degrees(np.mod(np.arctan2(xs[:, 1], xs[:, 0]), 2 * np.pi))
on = np.abs(r - 1) < 0.1; inarc = ((ang >= 18) & (ang <= 108)) | ((ang >= 180) & (ang <= 270))
zgrid = np.linspace(-4, 4, 801)
agg = np.mean(np.exp(-0.5 * ((zgrid[:, None] - MU[:, 0]) / SD[:, 0]) ** 2) / (SD[:, 0] * np.sqrt(2 * np.pi)), axis=1)
num["toy_vae"] = dict(n=300, latent=1, hidden=T.H, sigma_x2=T.SIGMA_X2, steps=20000, final_loss=round(hist[-1][1], 3),
                      final_recon=round(hist[-1][2], 3), final_kl=round(hist[-1][3], 3),
                      mean_range_arc1=[round(float(mu1.min()), 2), round(float(mu1.max()), 2)],
                      mean_range_arc2=[round(float(mu2.min()), 2), round(float(mu2.max()), 2)],
                      sd_range=[round(float(SD.min()), 3), round(float(SD.max()), 3)],
                      prior_samples=2000, share_on_circle=round(float(on.mean()), 3), share_on_data_arcs=round(float((on & inarc).mean()), 3))

# ---------- figure 1: lec12-kp05-architecture ----------
fig, ax = plt.subplots(figsize=(10.5, 3.6)); ax.set_xlim(0, 10.5); ax.set_ylim(0, 3.6); ax.axis("off")


def box(x0, y0, w, h, text, fc, ec, fs=11):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.05,rounding_size=0.12", fc=fc, ec=ec, lw=1.8))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(x0, y0, x1, y1, col="#334155", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=col, lw=1.8, ls=ls))


box(0.1, 1.55, 0.8, 0.7, r"$\mathbf{x}$", "#F1F5F9", "#475569", 14)
box(1.3, 1.25, 1.7, 1.3, "encoder\n" + r"$\theta_e$", "#EDE9FE", "#7C3AED")
box(3.4, 2.15, 1.3, 0.6, r"$\boldsymbol{\mu}$", "#EDE9FE", "#7C3AED", 13)
box(3.4, 1.05, 1.3, 0.6, r"$\log\boldsymbol{\sigma}^2$", "#EDE9FE", "#7C3AED", 12)
box(5.2, 1.4, 1.5, 1.0, r"sample $\mathbf{z}$" + "\n" + r"$\sim\mathcal{N}(\boldsymbol{\mu},\mathrm{diag}\,\boldsymbol{\sigma}^2)$", "#CCFBF1", "#0F766E", 10)
box(7.1, 1.25, 1.7, 1.3, "decoder\n" + r"$\theta_d$", "#DBEAFE", "#1D4ED8")
box(9.2, 1.55, 1.1, 0.7, r"$\hat{\mathbf{x}}$", "#F1F5F9", "#475569", 14)
arrow(0.9, 1.9, 1.3, 1.9); arrow(3.0, 2.2, 3.4, 2.45); arrow(3.0, 1.6, 3.4, 1.35)
arrow(4.7, 2.45, 5.2, 2.1); arrow(4.7, 1.35, 5.2, 1.7); arrow(6.7, 1.9, 7.1, 1.9); arrow(8.8, 1.9, 9.2, 1.9)
box(5.25, 0.1, 1.4, 0.6, r"prior $\mathcal{N}(\mathbf{0},I)$", "#FEF3C7", "#B45309", 10)
ax.annotate("", xy=(7.1, 1.45), xytext=(6.65, 0.4), arrowprops=dict(arrowstyle="->", color="#B45309", lw=1.8, ls="--"))
ax.text(7.25, 0.35, "generation: skip the encoder", color="#B45309", fontsize=10)
ax.text(5.95, 3.05, r"training: maximise ELBO $=\mathbb{E}_q[\log p(\mathbf{x}\mid\mathbf{z})]-D_{\mathrm{KL}}(q\,\Vert\,p(\mathbf{z}))$",
        ha="center", fontsize=11, color="#0F172A")
fig.tight_layout(); fig.savefig(OUT / "lec12-kp05-architecture.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec12-kp05-latent-coverage ----------
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw=dict(width_ratios=[1.2, 1]))
ax = axs[0]
order = np.argsort(MU[:, 0])
for k, j in enumerate(order[::6]):
    col = "#1D4ED8" if j < 150 else "#7C3AED"
    ax.plot([MU[j, 0] - 2 * SD[j, 0], MU[j, 0] + 2 * SD[j, 0]], [k * 0.02 + 0.55] * 2, color=col, lw=1.2, alpha=0.8)
    ax.scatter([MU[j, 0]], [k * 0.02 + 0.55], color=col, s=8)
ax.plot(zgrid, agg, color="#0F766E", lw=2.4, label=r"average of all $q(z\mid\mathbf{x})$")
ax.plot(zgrid, np.exp(-0.5 * zgrid ** 2) / np.sqrt(2 * np.pi), color="#B45309", lw=2, ls="--", label=r"prior $\mathcal{N}(0,1)$")
ax.plot([], [], color="#1D4ED8", lw=1.2, label=r"arc 1: mean $\pm2$ sd")
ax.plot([], [], color="#7C3AED", lw=1.2, label=r"arc 2: mean $\pm2$ sd")
ax.set_xlim(-4, 4); ax.set_ylim(0, 1.65); ax.set_yticks([0, 0.2, 0.4]); ax.set_xlabel(r"code $z$"); ax.set_title("code space (every 6th training point)", fontsize=11)
ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8.5)
ax = axs[1]
th = np.linspace(0, 2 * np.pi, 400); ax.plot(np.cos(th), np.sin(th), color="#CBD5E1", lw=1, ls=":")
ax.scatter(X[:, 0], X[:, 1], s=8, color="#93C5FD", label="training data")
ax.scatter(xs[:, 0], xs[:, 1], s=4, color="#B45309", alpha=0.35, label=r"decoded $z\sim\mathcal{N}(0,1)$, 2000 draws")
ax.set_aspect("equal"); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.grid(alpha=0.3)
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); ax.set_title(f"generation: {100 * on.mean():.1f}% land on the circle", fontsize=11)
ax.legend(loc="upper left", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "lec12-kp05-latent-coverage.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
