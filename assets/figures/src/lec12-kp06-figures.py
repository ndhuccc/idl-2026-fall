"""lec12-kp06-figures.py — 第12週 KP06 講義的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec12-kp06-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
需要 lec12_vae_toy.py（KP05 的小 VAE）；在課程 repo 裡它和本檔放在同一個資料夾，在任務資料夾裡它在 ../lec12-kp05-vae-architecture/。
抽樣估計沿用 KP02、KP03 的一維模型：z ~ N(0,1)，x|z ~ N(z, 0.5^2)，觀察 x = 1.2；近似後驗 q = N(0.5, 0.2)。"""
import json, sys, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
for d in (HERE, HERE.parent / "lec12-kp05-vae-architecture"):
    sys.path.insert(0, str(d))
import lec12_vae_toy as T  # noqa: E402

plt.rcParams["mathtext.fontset"] = "cm"
args = sys.argv[1:]
REPORT = None
if "--report" in args:
    i = args.index("--report"); REPORT = Path(args[i + 1]); del args[i:i + 2]
OUT = Path(args[0]) if args else HERE.parent
OUT.mkdir(parents=True, exist_ok=True)
num = {}

# ---------- [Added] Type 1 vs Type 2 Monte Carlo estimates of the ELBO ----------
sig2, x = 0.25, 1.2
m, s2 = 0.5, 0.2
recon_exact = -0.5 * math.log(2 * math.pi * sig2) - ((x - m) ** 2 + s2) / (2 * sig2)
kl_exact = 0.5 * (s2 + m ** 2 - 1 - math.log(s2))
elbo = recon_exact - kl_exact
log_px = -0.5 * math.log(2 * math.pi * 1.25) - x ** 2 / 2.5


def logN(v, mu, var):
    return -0.5 * np.log(2 * np.pi * var) - (v - mu) ** 2 / (2 * var)


def type1(z):   # log p(x, z) - log q(z)
    return logN(z, 0, 1) + logN(x, z, sig2) - logN(z, m, s2)


def type2(z):   # log p(x | z) - analytic KL
    return logN(x, z, sig2) - kl_exact


rng = np.random.default_rng(0)
ells = [1, 10, 100, 1000]
REP = 20000


def spread(mq, vq):
    kq = 0.5 * (vq + mq ** 2 - 1 - math.log(vq))
    t1 = lambda z: logN(z, 0, 1) + logN(x, z, sig2) - logN(z, mq, vq)
    t2 = lambda z: logN(x, z, sig2) - kq
    zb = rng.normal(mq, math.sqrt(vq), 1_000_000)
    e = -0.5 * math.log(2 * math.pi * sig2) - ((x - mq) ** 2 + vq) / (2 * sig2) - kq
    assert abs(np.mean(t1(zb)) - e) < 0.005 and abs(np.mean(t2(zb)) - e) < 0.005
    rows = []
    for l in ells:
        z = rng.normal(mq, math.sqrt(vq), (REP, l))
        rows.append(dict(ell=l, type1_sd=round(float(t1(z).mean(1).std()), 4), type2_sd=round(float(t2(z).mean(1).std()), 4)))
    return dict(q=[mq, vq], elbo=round(e, 4), kl=round(kq, 4), per_sample_sd_type1=round(float(np.std(t1(zb))), 4),
                per_sample_sd_type2=round(float(np.std(t2(zb))), 4), by_ell=rows)


CASES = {"wide": (m, s2), "narrow": (0.96, 0.05)}
sp = {k: spread(*v) for k, v in CASES.items()}
z5 = np.array([0.9, 0.2, 0.6, 0.1, 0.7])
num["mc"] = dict(model="z~N(0,1), x|z~N(z,0.25), x=1.2", log_px=round(log_px, 4), q_wide=sp["wide"], q_narrow=sp["narrow"],
                 q_posterior_type1_sd=0.0,
                 five_draws_q_wide=z5.tolist(), five_type1=type1(z5).round(4).tolist(), five_type1_mean=round(float(type1(z5).mean()), 4),
                 five_type2=type2(z5).round(4).tolist(), five_type2_mean=round(float(type2(z5).mean()), 4))
assert abs(sp["wide"]["elbo"] - elbo) < 1e-4

# ---------- Lemma 1 / Lemma 2 checks ----------
def kl1(m1, s1, m2, s2_):
    return math.log(s2_ / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2_ ** 2) - 0.5
mu_v, var_v = np.array([0.5, -0.3]), np.array([0.1, 0.2])
kl_notes = 0.5 * (-np.log(var_v).sum() - 2 + var_v.sum() + (mu_v ** 2).sum())
kl_sum = sum(kl1(mu_v[k], math.sqrt(var_v[k]), 0, 1) for k in range(2))
assert abs(kl_notes - kl_sum) < 1e-12
zz = np.random.default_rng(1).normal(2, 0.5, 2_000_000)
kl_mc = float(np.mean(logN(zz, 2, 0.25) - logN(zz, 0, 1)))
num["lemma"] = dict(demo_mu1_2_sigma1_05=round(kl1(2, 0.5, 0, 1), 4), demo_source_claims=2.443, demo_mc_check=round(kl_mc, 3),
                    notes_example=round(float(kl_notes), 4), notes_example_sum_of_1d=round(float(kl_sum), 4),
                    kp05_example=round(kl1(1, 0.5, 0, 1), 4))

# ---------- [Added] reparameterization: samples and gradients of E[z^2] ----------
mu_r, sd_r = 0.5, 0.8
eps = np.random.default_rng(2).normal(size=10000)
zr = mu_r + sd_r * eps
g_mu = float(np.mean(2 * zr * 1.0)); g_sd = float(np.mean(2 * zr * eps))
num["reparam"] = dict(mu=mu_r, sigma=sd_r, n=10000, sample_mean=round(float(zr.mean()), 3), sample_sd=round(float(zr.std()), 3),
                      notes_eps=-0.3, notes_z=round(mu_r + sd_r * -0.3, 4),
                      f="E[z^2] = mu^2 + sigma^2", grad_mu_exact=2 * mu_r, grad_sigma_exact=2 * sd_r,
                      grad_mu_reparam=round(g_mu, 3), grad_sigma_reparam=round(g_sd, 3))

# ---------- [Added] train the toy VAE (KP05) with beta = 1 and beta = 0 ----------
num["grad_check_max_rel_err"] = float(f"{T.grad_check():.2e}")
X, P1, h1 = T.train(beta=1.0)
X, P0, h0 = T.train(beta=0.0)
_, MU0, LV0 = T.encode(X, P0); SD0 = np.exp(0.5 * LV0)
_, MU1, LV1 = T.encode(X, P1); SD1 = np.exp(0.5 * LV1)
zs = np.random.default_rng(7).normal(size=(2000, 1))
def share(P):
    _, xs = T.decode(zs, P); r = np.linalg.norm(xs, axis=1)
    ang = np.degrees(np.mod(np.arctan2(xs[:, 1], xs[:, 0]), 2 * np.pi))
    on = np.abs(r - 1) < 0.1; inarc = ((ang >= 18) & (ang <= 108)) | ((ang >= 180) & (ang <= 270))
    return float(on.mean()), float((on & inarc).mean())
s1, s0 = share(P1), share(P0)
pick = lambda h, it: next(r for r in h if r[0] == it)
num["toy_training"] = dict(
    beta1=dict(it1=[round(v, 3) for v in pick(h1, 1)[1:]], it2000=[round(v, 3) for v in pick(h1, 2000)[1:]], it20000=[round(v, 3) for v in pick(h1, 20000)[1:]],
               mean_range=[round(float(MU1.min()), 2), round(float(MU1.max()), 2)], sd_median=round(float(np.median(SD1)), 3),
               share_on_circle=round(s1[0], 3), share_on_arcs=round(s1[1], 3)),
    beta0=dict(it20000=[round(v, 3) for v in pick(h0, 20000)[1:]], mean_range=[round(float(MU0.min()), 1), round(float(MU0.max()), 1)],
               sd_median=round(float(np.median(SD0)), 4), share_on_circle=round(s0[0], 3), share_on_arcs=round(s0[1], 3)),
    note="values are [loss, recon, KL] per point; recon = ||x-x_hat||^2/(2*0.01)")

# ---------- figure 1: lec12-kp06-mc-estimates ----------
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
L = np.array(ells)
for ax, key, ttl in zip(axs, ("wide", "narrow"), (r"$q=\mathcal{N}(0.5,\ 0.2)$", r"narrow $q=\mathcal{N}(0.96,\ 0.05)$")):
    d = sp[key]
    ax.loglog(L, [r["type1_sd"] for r in d["by_ell"]], "o-", color="#B45309", lw=2, label=r"Type 1: $\log p(x,z)-\log q(z)$")
    ax.loglog(L, [r["type2_sd"] for r in d["by_ell"]], "s-", color="#0F766E", lw=2, label=r"Type 2: $\log p(x\mid z)-$ exact KL")
    ax.set_xlabel(r"samples per estimate $\ell$"); ax.grid(alpha=0.3, which="both")
    ax.set_title(ttl + f", ELBO = {d['elbo']:.4f}", fontsize=10.5)
axs[0].set_ylabel("spread (sd) of the ELBO estimate"); axs[0].legend(fontsize=9, loc="lower left")
axs[0].text(10, 0.75, r"both fall as $1/\sqrt{\ell}$", color="#334155", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "lec12-kp06-mc-estimates.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec12-kp06-reparameterization ----------
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axs[0]; ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
def node(x0, y0, t, fc, ec):
    ax.text(x0, y0, t, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.45", fc=fc, ec=ec, lw=1.6))
def arr(a, b, col="#334155", ls="-"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=col, lw=1.6, ls=ls))
ax.text(5, 5.6, r"before: $z\sim\mathcal{N}(\mu,\sigma^2)$  (random node, no path back)", ha="center", fontsize=10, color="#B91C1C")
node(1.3, 4.5, r"$\mu,\sigma$", "#EDE9FE", "#7C3AED"); node(5, 4.5, r"sample $z$", "#FEE2E2", "#B91C1C"); node(8.7, 4.5, r"loss", "#F1F5F9", "#475569")
arr((2.1, 4.5), (4.0, 4.5)); arr((6.0, 4.5), (8.0, 4.5))
ax.text(3.05, 3.9, r"$\partial z/\partial\mu$ = ?", ha="center", fontsize=10, color="#B91C1C")
ax.text(5, 2.75, r"after: $z=\mu+\sigma\,\epsilon,\ \epsilon\sim\mathcal{N}(0,1)$", ha="center", fontsize=10, color="#0F766E")
node(1.3, 1.5, r"$\mu,\sigma$", "#EDE9FE", "#7C3AED"); node(5, 1.5, r"$z=\mu+\sigma\epsilon$", "#CCFBF1", "#0F766E"); node(8.7, 1.5, r"loss", "#F1F5F9", "#475569")
node(5, 0.25, r"$\epsilon$ (outside input)", "#FEF3C7", "#B45309")
arr((2.1, 1.5), (3.75, 1.5)); arr((6.25, 1.5), (8.0, 1.5)); arr((5, 0.6), (5, 1.15), "#B45309")
ax.annotate("", xy=(2.1, 1.05), xytext=(3.75, 1.05), arrowprops=dict(arrowstyle="->", color="#1D4ED8", lw=1.4, ls="--"))
ax.text(2.9, 0.45, r"$\partial z/\partial\mu=1$" + "\n" + r"$\partial z/\partial\sigma=\epsilon$", ha="center", fontsize=9.5, color="#1D4ED8")
ax = axs[1]
ax.hist(zr, bins=60, density=True, color="#14B8A6", alpha=0.45, label=r"$\mu+\sigma\epsilon$, 10000 draws")
g = np.linspace(-3, 3.5, 400)
ax.plot(g, np.exp(-(g - mu_r) ** 2 / (2 * sd_r ** 2)) / (sd_r * np.sqrt(2 * np.pi)), color="#1D4ED8", lw=2, label=r"$\mathcal{N}(0.5,\ 0.8^2)$")
for v in (mu_r - sd_r, mu_r, mu_r + sd_r):
    ax.axvline(v, color="#B91C1C", ls="--", lw=1)
ax.set_xlabel(r"$z$"); ax.set_ylabel("density"); ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc="upper left")
ax.set_title(f"sample mean {zr.mean():.3f}, sd {zr.std():.3f}", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "lec12-kp06-reparameterization.png", dpi=200); plt.close(fig)

# ---------- figure 3: lec12-kp06-training-curves ----------
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axs[0]
it = np.array([r[0] for r in h1])
ax.semilogy(it, [r[2] for r in h1], color="#1D4ED8", lw=2, label="reconstruction term")
ax.semilogy(it, [r[3] for r in h1], color="#B91C1C", lw=2, label="KL term")
ax.semilogy(it, [r[1] for r in h1], color="#0F766E", lw=2, ls="--", label=r"loss $=-$ELBO estimate")
ax.set_xlabel("training step"); ax.set_ylabel("per point (log scale)"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)
ax.set_title(r"toy VAE, $\beta=1$", fontsize=11)
ax = axs[1]
_, xs0 = T.decode(zs, P0)
th = np.linspace(0, 2 * np.pi, 400); ax.plot(np.cos(th), np.sin(th), color="#CBD5E1", lw=1, ls=":")
ax.scatter(X[:, 0], X[:, 1], s=8, color="#93C5FD", label="training data")
ax.scatter(xs0[:, 0], xs0[:, 1], s=4, color="#B91C1C", alpha=0.35, label=r"decoded $z\sim\mathcal{N}(0,1)$")
ax.set_aspect("equal"); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc="upper left")
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
ax.set_title(rf"$\beta=0$ (no KL): {100 * s0[0]:.0f}% on the circle, {100 * s0[1]:.0f}% on the data arcs", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "lec12-kp06-training-curves.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
