"""lec12-kp07-figures.py — 第12週 KP07 講義的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec12-kp07-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
需要 lec12_vae_toy.py（KP05 的小 VAE）；在課程 repo 裡它和本檔放在同一個資料夾，在任務資料夾裡它在 ../lec12-kp05-vae-architecture/。"""
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

# ---------- [Added] test phase with the beta = 1 toy VAE: reconstruction errors ----------
X, P, _ = T.train(beta=1.0)


def recon(pts):          # reconstruct with z = mu (the encoder's mean)
    _, mu, _ = T.encode(pts, P); _, xh = T.decode(mu, P)
    return mu[:, 0], xh, np.sum((pts - xh) ** 2, 1)


_, _, e_train = recon(X)
deg = lambda a: [math.cos(math.radians(a)), math.sin(math.radians(a))]
tests = {"new point on arc 1 (60 deg)": deg(60), "gap point (315 deg)": deg(315), "centre (0, 0)": [0.0, 0.0]}
TP = np.array(list(tests.values()))
zt, xt, et = recon(TP)
num["test_phase"] = dict(model="toy VAE of KP05, beta = 1, reconstruct with z = mu",
                         train_sq_err_median=round(float(np.median(e_train)), 4), train_sq_err_max=round(float(e_train.max()), 4),
                         tests={k: dict(x=[round(v, 3) for v in TP[i]], z=round(float(zt[i]), 3), x_hat=xt[i].round(3).tolist(),
                                        sq_err=round(float(et[i]), 4)) for i, k in enumerate(tests)})

# ---------- [Added] why squared error blurs: two sharp "images" ----------
A = (np.arange(20) >= 8).astype(float); B = (np.arange(20) >= 12).astype(float); M = (A + B) / 2
num["blur"] = dict(pixels=20, edge_A=8, edge_B=12, diff_pixels=int(np.sum(A != B)),
                   expected_err_guess_A=float(0.5 * 0 + 0.5 * np.sum((A - B) ** 2)),
                   expected_err_average=float(0.5 * np.sum((A - M) ** 2) + 0.5 * np.sum((B - M) ** 2)),
                   two_pixel=dict(images=[[1, 0], [0, 1]], guess_err=1.0, average=[0.5, 0.5], average_err=0.5))

# ---------- [Added] beta sweep ----------
BETAS = [0, 0.3, 1, 3, 10, 30, 100]
zs = np.random.default_rng(7).normal(size=(2000, 1))
sweep, samples = [], {}
for b in BETAS:
    Xb, Pb, hb = T.train(beta=float(b))
    _, MU, LV = T.encode(Xb, Pb)
    _, xs = T.decode(zs, Pb); r = np.linalg.norm(xs, axis=1)
    ang = np.degrees(np.mod(np.arctan2(xs[:, 1], xs[:, 0]), 2 * np.pi))
    on = np.abs(r - 1) < 0.1; inarc = ((ang >= 18) & (ang <= 108)) | ((ang >= 180) & (ang <= 270))
    _, recon_last, kl_last = hb[-1][1:]
    sweep.append(dict(beta=b, recon=round(recon_last, 3), kl=round(kl_last, 3), sq_err=round(recon_last * 2 * T.SIGMA_X2, 4),
                      sd_median=round(float(np.median(np.exp(0.5 * LV))), 3),
                      mean_range=[round(float(MU.min()), 2), round(float(MU.max()), 2)],
                      share_on_arcs=round(float((on & inarc).mean()), 3), decoded_spread=round(float(xs.std(0).mean()), 3)))
    samples[b] = xs
num["beta_sweep"] = sweep
num["data_mean"] = X.mean(0).round(3).tolist()
num["data_total_variance"] = round(float(np.sum(X.var(0))), 4)

# ---------- figure 1: lec12-kp07-test-phase ----------
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw=dict(width_ratios=[1, 1.15]))
ax = axs[0]
th = np.linspace(0, 2 * np.pi, 400); ax.plot(np.cos(th), np.sin(th), color="#CBD5E1", lw=1, ls=":")
ax.scatter(X[:, 0], X[:, 1], s=7, color="#93C5FD", label="training data")
cols = ["#15803D", "#B45309", "#B91C1C"]
for i, k in enumerate(tests):
    ax.scatter(*TP[i], s=70, marker="o", facecolor="white", edgecolor=cols[i], lw=2, zorder=5)
    ax.scatter(*xt[i], s=70, marker="X", color=cols[i], zorder=6)
    ax.annotate("", xy=xt[i], xytext=TP[i], arrowprops=dict(arrowstyle="->", color=cols[i], lw=1.6))
ax.scatter([], [], s=50, facecolor="white", edgecolor="#334155", label=r"test input $\mathbf{x}$")
ax.scatter([], [], s=50, marker="X", color="#334155", label=r"reconstruction $\hat{\mathbf{x}}$")
ax.set_aspect("equal"); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc="upper left")
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); ax.set_title(r"reconstruction: $\mathbf{x}\to z=\mu\to\hat{\mathbf{x}}$", fontsize=11)
ax = axs[1]
bins = np.logspace(-5, 1, 40)
ax.hist(e_train, bins=bins, color="#93C5FD", edgecolor="#1D4ED8", label="300 training points")
for i, k in enumerate(tests):
    ax.axvline(et[i], color=cols[i], lw=2, label=f"{k}: {et[i]:.3f}")
ax.set_xscale("log"); ax.set_xlabel(r"squared reconstruction error $\Vert\mathbf{x}-\hat{\mathbf{x}}\Vert^2$"); ax.set_ylabel("count")
ax.grid(alpha=0.3); ax.legend(fontsize=8.2, loc="upper left"); ax.set_ylim(0, 95)
ax.set_title("anomaly detection: large error = unusual input", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "lec12-kp07-test-phase.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec12-kp07-blur ----------
fig, axs = plt.subplots(3, 1, figsize=(8.6, 3.9), sharex=True)
for ax, img, ttl in zip(axs, (A, B, M), (r"sharp image A (edge at pixel 8)", r"sharp image B (edge at pixel 12)",
                                          r"average of A and B: what squared error prefers")):
    ax.imshow(img[None, :], cmap="gray_r", vmin=0, vmax=1, aspect="auto")
    for j, v in enumerate(img):
        ax.text(j, 0, f"{v:g}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else "#0F172A")
    ax.set_yticks([]); ax.set_title(ttl, fontsize=10, loc="left")
axs[-1].set_xticks(range(20)); axs[-1].set_xlabel("pixel")
fig.tight_layout(); fig.savefig(OUT / "lec12-kp07-blur.png", dpi=200); plt.close(fig)

# ---------- figure 3: lec12-kp07-beta-sweep ----------
fig = plt.figure(figsize=(11, 4.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1])
ax = fig.add_subplot(gs[0])
xpos = np.arange(len(BETAS))
ax.plot(xpos, [s["sq_err"] for s in sweep], "o-", color="#1D4ED8", lw=2, label=r"squared error $\Vert\mathbf{x}-\hat{\mathbf{x}}\Vert^2$")
ax.plot(xpos, [s["kl"] for s in sweep], "s-", color="#B91C1C", lw=2, label="KL term")
ax.set_yscale("log"); ax.set_xticks(xpos); ax.set_xticklabels([f"{b:g}" for b in BETAS])
ax.set_xlabel(r"KL weight $\beta$"); ax.set_ylabel("per point (log scale)"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8.8, loc="center left")
ax.set_title(r"larger $\beta$: smaller KL, worse reconstruction", fontsize=10.5)
for k, b in enumerate((10, 100)):
    ax2 = fig.add_subplot(gs[k + 1])
    ax2.plot(np.cos(th), np.sin(th), color="#CBD5E1", lw=1, ls=":")
    ax2.scatter(X[:, 0], X[:, 1], s=6, color="#93C5FD")
    ax2.scatter(samples[b][:, 0], samples[b][:, 1], s=4, color="#B45309", alpha=0.35)
    ax2.set_aspect("equal"); ax2.set_xlim(-1.6, 1.6); ax2.set_ylim(-1.6, 1.6); ax2.grid(alpha=0.3)
    s = next(s for s in sweep if s["beta"] == b)
    ax2.set_title(rf"$\beta={b}$: {100 * s['share_on_arcs']:.0f}% on the data arcs", fontsize=10.5)
    ax2.set_xlabel(r"$x_1$")
fig.tight_layout(); fig.savefig(OUT / "lec12-kp07-beta-sweep.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
