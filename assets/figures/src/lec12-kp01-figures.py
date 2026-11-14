"""lec12-kp01-figures.py — 第12週 KP01 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec12-kp01-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。"""
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

# ---------- source example (CCC-LNS Ch09 professor notes): x=[1,2,1], k=1, linear ----------
x = np.array([1.0, 2.0, 1.0]); We = np.array([0.5, 0.5, 0.5]); Wd = np.array([0.5, 1.0, 0.5])
z = We @ x; xh = Wd * z
assert z == 2.0 and np.allclose(xh, x)
x2 = np.array([2.0, 1.0, 1.0]); z2 = We @ x2; xh2 = Wd * z2; err2 = float(np.sum((x2 - xh2) ** 2))
num["source_example"] = dict(z=z, x_hat=xh.tolist(), loss=0.0)
num["added_second_input"] = dict(x=x2.tolist(), z=z2, x_hat=xh2.tolist(), loss=round(err2, 4))

# ---------- [Added] linear bottleneck on 2-D points: k=1 keeps the best line (PCA) ----------
rng = np.random.default_rng(0)
t = rng.uniform(-2.5, 2.5, 10)
X = np.c_[t, 0.5 * t] + rng.normal(0, 0.25, (10, 2))
mu = X.mean(0); C = (X - mu).T @ (X - mu) / len(X)
evals, evecs = np.linalg.eigh(C); u = evecs[:, -1] * np.sign(evecs[0, -1])
Z1 = (X - mu) @ u; R1 = mu + np.outer(Z1, u)
mse_pca = float(np.mean(np.sum((X - R1) ** 2, 1)))
ux = np.array([1.0, 0.0]); Rx = mu + np.outer((X - mu) @ ux, ux)
mse_x = float(np.mean(np.sum((X - Rx) ** 2, 1)))
total = float(np.mean(np.sum((X - mu) ** 2, 1)))
assert abs(mse_pca - evals[0]) < 1e-9 and abs(total - evals.sum()) < 1e-9
num["linear_k1"] = dict(n=10, seed=0, direction=u.round(4).tolist(), mse_best_line=round(mse_pca, 4), mse_x_axis=round(mse_x, 4),
                        mse_k2=0.0, total_variance=round(total, 4), kept_share=round(1 - mse_pca / total, 4))

# ---------- [Added] nonlinear AE on two arcs: codes leave a gap the decoder never learned ----------
rng = np.random.default_rng(1)
a1 = rng.uniform(0.10 * np.pi, 0.60 * np.pi, 150); a2 = rng.uniform(1.00 * np.pi, 1.50 * np.pi, 150)
ang = np.r_[a1, a2]; D = np.c_[np.cos(ang), np.sin(ang)] + rng.normal(0, 0.02, (300, 2))
H = 32
P = {"W1": rng.normal(0, 0.5, (2, H)), "b1": np.zeros(H), "W2": rng.normal(0, 0.5, (H, 1)), "b2": np.zeros(1),
     "W3": rng.normal(0, 0.5, (1, H)), "b3": np.zeros(H), "W4": rng.normal(0, 0.5, (H, 2)), "b4": np.zeros(2)}


def enc(X, P):
    h = np.tanh(X @ P["W1"] + P["b1"]); return h, h @ P["W2"] + P["b2"]


def dec(Zc, P):
    g = np.tanh(Zc @ P["W3"] + P["b3"]); return g, g @ P["W4"] + P["b4"]


m = {k: np.zeros_like(v) for k, v in P.items()}; v2 = {k: np.zeros_like(v) for k, v in P.items()}
lr, b1_, b2_ = 3e-3, 0.9, 0.999
for it in range(1, 12001):
    h, Zc = enc(D, P); g, Y = dec(Zc, P)
    dY = 2 * (Y - D) / len(D)
    G = {"W4": g.T @ dY, "b4": dY.sum(0)}
    dg = dY @ P["W4"].T * (1 - g ** 2)
    G["W3"] = Zc.T @ dg; G["b3"] = dg.sum(0)
    dZ = dg @ P["W3"].T
    G["W2"] = h.T @ dZ; G["b2"] = dZ.sum(0)
    dh = dZ @ P["W2"].T * (1 - h ** 2)
    G["W1"] = D.T @ dh; G["b1"] = dh.sum(0)
    for k in P:
        m[k] = b1_ * m[k] + (1 - b1_) * G[k]; v2[k] = b2_ * v2[k] + (1 - b2_) * G[k] ** 2
        P[k] -= lr * (m[k] / (1 - b1_ ** it)) / (np.sqrt(v2[k] / (1 - b2_ ** it)) + 1e-8)
_, Zc = enc(D, P); _, Y = dec(Zc, P)
mse_train = float(np.mean(np.sum((Y - D) ** 2, 1)))
z1, z2c = Zc[:150, 0], Zc[150:, 0]
lo1, hi1, lo2, hi2 = z1.min(), z1.max(), z2c.min(), z2c.max()
if lo2 > hi1:
    g_lo, g_hi = hi1, lo2
else:
    g_lo, g_hi = hi2, lo1
assert g_hi > g_lo, "codes of the two arcs overlap"
zg = np.linspace(g_lo, g_hi, 41)[1:-1, None]
_, Yg = dec(zg, P)
off = np.abs(np.linalg.norm(Yg, axis=1) - 1)
zmid = np.array([[0.5 * (g_lo + g_hi)]]); _, ymid = dec(zmid, P)
ang_gap = np.degrees(np.mod(np.arctan2(Yg[:, 1], Yg[:, 0]), 2 * np.pi))
zlo_all, zhi_all = min(lo1, lo2), max(hi1, hi2)
ze_hi = np.linspace(zhi_all, zhi_all + 1.6, 60)[:, None]; _, Ye_hi = dec(ze_hi, P)
ze_lo = np.linspace(zlo_all - 1.6, zlo_all, 60)[:, None]; _, Ye_lo = dec(ze_lo, P)
r_ext = dict(hi=round(float(np.linalg.norm(Ye_hi[-1])), 3), lo=round(float(np.linalg.norm(Ye_lo[0])), 3))
num["two_arcs"] = dict(n=300, seed=1, hidden=H, steps=12000, mse_train=round(mse_train, 4),
                       code_range_arc1=[round(float(lo1), 3), round(float(hi1), 3)], code_range_arc2=[round(float(lo2), 3), round(float(hi2), 3)],
                       gap=[round(float(g_lo), 3), round(float(g_hi), 3)], gap_mid_code=round(float(zmid[0, 0]), 3),
                       gap_mid_decoded=ymid[0].round(3).tolist(), gap_mid_radius=round(float(np.linalg.norm(ymid[0])), 3),
                       gap_max_off_circle=round(float(off.max()), 3), gap_angle_deg=[round(float(ang_gap.min()), 1), round(float(ang_gap.max()), 1)],
                       data_angle_deg=[[18, 108], [180, 270]], extrap_codes=[round(float(zlo_all - 1.6), 3), round(float(zhi_all + 1.6), 3)],
                       extrap_end_radius=r_ext, train_max_off_circle=round(float(np.abs(np.linalg.norm(Y, axis=1) - 1).max()), 3))

# ---------- figure 1: lec12-kp01-linear-bottleneck ----------
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
for ax, R, col, ttl, mse in ((axs[0], R1, "#15803D", "best line ($k=1$, same as PCA)", mse_pca),
                             (axs[1], Rx, "#B91C1C", "a worse line: horizontal through the mean", mse_x)):
    d = u if R is R1 else ux
    s = np.linspace(-4, 4, 2)
    ax.plot(mu[0] + s * d[0], mu[1] + s * d[1], color=col, lw=2)
    for p, r in zip(X, R):
        ax.plot([p[0], r[0]], [p[1], r[1]], color="#64748B", lw=1, ls="--")
    ax.scatter(X[:, 0], X[:, 1], s=36, color="#1D4ED8", zorder=3, label=r"input $\mathbf{x}$")
    ax.scatter(R[:, 0], R[:, 1], s=36, facecolor="white", edgecolor=col, lw=1.8, zorder=4, label=r"reconstruction $\hat{\mathbf{x}}$")
    ax.set_title(ttl + f"\nmean $\\|\\mathbf{{x}}-\\hat{{\\mathbf{{x}}}}\\|^2={mse:.4f}$", fontsize=11)
    ax.set_aspect("equal"); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.2, 2.2); ax.grid(alpha=0.3)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
axs[0].legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "lec12-kp01-linear-bottleneck.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec12-kp01-latent-holes ----------
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axs[0]
th = np.linspace(0, 2 * np.pi, 400); ax.plot(np.cos(th), np.sin(th), color="#CBD5E1", lw=1, ls=":")
ax.scatter(D[:, 0], D[:, 1], s=8, color="#93C5FD", label="training data")
for lo, hi in ((lo1, hi1), (lo2, hi2)):
    zz = np.linspace(lo, hi, 200)[:, None]; _, yy = dec(zz, P)
    ax.plot(yy[:, 0], yy[:, 1], color="#1D4ED8", lw=2.5)
ax.plot([], [], color="#1D4ED8", lw=2.5, label=r"$g(z)$, $z$ used in training")
ax.plot(Yg[:, 0], Yg[:, 1], color="#DC2626", lw=2.5, label=r"$g(z)$, $z$ in the gap")
ax.scatter([ymid[0, 0]], [ymid[0, 1]], s=60, color="#DC2626", zorder=5)
ax.plot(Ye_hi[:, 0], Ye_hi[:, 1], color="#D97706", lw=2.5, ls="--", label=r"$g(z)$, $z$ beyond the used range")
ax.plot(Ye_lo[:, 0], Ye_lo[:, 1], color="#D97706", lw=2.5, ls="--")
ax.set_aspect("equal"); ax.set_xlim(-2.0, 2.0); ax.set_ylim(-2.0, 2.0); ax.grid(alpha=0.3)
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); ax.set_title("input space", fontsize=11)
ax.legend(loc="upper left", fontsize=8.5)
ax = axs[1]
ax.scatter(z1, np.zeros_like(z1) + 0.15, s=10, color="#1D4ED8", alpha=0.6, label="codes of arc 1")
ax.scatter(z2c, np.zeros_like(z2c) - 0.15, s=10, color="#7C3AED", alpha=0.6, label="codes of arc 2")
ax.axvspan(g_lo, g_hi, color="#FECACA", alpha=0.6, label="gap: no training code")
ax.axvspan(zhi_all, zhi_all + 1.6, color="#FDE68A", alpha=0.6, label="beyond the used range")
ax.axvspan(zlo_all - 1.6, zlo_all, color="#FDE68A", alpha=0.6)
ax.scatter([zmid[0, 0]], [0], s=60, color="#DC2626", zorder=5)
ax.set_ylim(-1, 1); ax.set_yticks([]); ax.set_xlabel(r"code $z$ ($k=1$)"); ax.set_title("code space", fontsize=11)
ax.legend(loc="upper center", fontsize=8.5, ncol=1); ax.grid(alpha=0.3, axis="x")
fig.tight_layout(); fig.savefig(OUT / "lec12-kp01-latent-holes.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
