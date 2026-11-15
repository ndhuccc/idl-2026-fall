"""lec14-kp01-figures.py — 第14週 KP01 講義的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec14-kp01-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
[補充] 一維玩具資料：x0 ~ 0.5 N(-2, 0.3^2) + 0.5 N(2, 0.3^2)。
前向：x_t = sqrt(1 - beta_t) x_{t-1} + sqrt(beta_t) eps，beta_t 從 1e-4 線性增加到 0.02，T = 1000（DDPM 論文的設定）。
逆向：因為資料是兩個常態的混合，「給定 x_t 時 x0 的平均」算得出來，所以可以用一個「完美去噪器」跑逆向，
代替還沒學會的神經網路（真正的模型要靠訓練去逼近它，見 KP04–KP05）。"""
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

# ---- 手算例子：beta = 0.36 的三步加噪（sqrt(1-beta)=0.8, sqrt(beta)=0.6）----
eps_hand = [0.5, -1.0, 0.0]
x = 2.0; hand = []
for k, e in enumerate(eps_hand, 1):
    x = 0.8 * x + 0.6 * e
    hand.append(dict(t=k, eps=e, x=round(x, 4), signal=round(2.0 * 0.8 ** k, 4)))
num["hand_example"] = dict(beta=0.36, scale=0.8, noise_scale=0.6, x0=2.0, steps=hand,
                           signal_after_20=round(2.0 * 0.8 ** 20, 4), var_check=round(0.8 ** 2 + 0.6 ** 2, 4))
assert [h["x"] for h in hand] == [1.9, 0.92, 0.736]
assert num["hand_example"]["signal_after_20"] == 0.0231 and num["hand_example"]["var_check"] == 1.0

# ---- 一維玩具資料的前向與逆向 ----
T = 1000
beta = np.linspace(1e-4, 0.02, T)          # beta[t-1] 是第 t 步
alpha = 1 - beta
abar = np.cumprod(alpha)
MU, S, N = 2.0, 0.3, 20000
SNAP = (0, 100, 300, 600, 1000)


def sample_x0(rng, n):
    return rng.choice([-MU, MU], n) + S * rng.normal(size=n)


def x0_hat(xt, ab):                         # E[x0 | x_t]，資料為兩個常態的混合時有閉式
    m = np.sqrt(ab) * np.array([-MU, MU])[:, None]
    v = ab * S * S + (1 - ab)
    logw = -(xt[None, :] - m) ** 2 / (2 * v)
    r = np.exp(logw - logw.max(0)); r /= r.sum(0)
    comp = np.array([-MU, MU])[:, None] + (np.sqrt(ab) * S * S / v) * (xt[None, :] - m)
    return (r * comp).sum(0)


rng = np.random.default_rng(0)
x0 = sample_x0(rng, N)
fwd = {0: x0.copy()}
xs = x0.copy()
for t in range(1, T + 1):
    xs = np.sqrt(alpha[t - 1]) * xs + np.sqrt(beta[t - 1]) * rng.normal(size=N)
    if t in SNAP:
        fwd[t] = xs.copy()
corr_T = float(np.corrcoef(x0, fwd[T])[0, 1])

rng = np.random.default_rng(1)
xs = rng.normal(size=N)
bwd = {T: xs.copy()}
for t in range(T, 0, -1):
    ab = abar[t - 1]
    eps_hat = (xs - np.sqrt(ab) * x0_hat(xs, ab)) / np.sqrt(1 - ab)
    mean = (xs - beta[t - 1] / np.sqrt(1 - ab) * eps_hat) / np.sqrt(alpha[t - 1])
    xs = mean + (np.sqrt(beta[t - 1]) * rng.normal(size=N) if t > 1 else 0.0)
    if t - 1 in SNAP:
        bwd[t - 1] = xs.copy()

stat = lambda a: dict(mean=round(float(a.mean()), 3), std=round(float(a.std()), 3), frac_pos=round(float((a > 0).mean()), 3))
right = bwd[0][bwd[0] > 0]
num["toy"] = dict(data="0.5 N(-2, 0.3^2) + 0.5 N(2, 0.3^2)", T=T, beta_first=1e-4, beta_last=0.02, samples=N,
                  forward_seed=0, backward_seed=1,
                  abar={str(t): float(f"{abar[t - 1]:.4g}") for t in SNAP if t > 0},
                  sqrt_abar={str(t): round(float(np.sqrt(abar[t - 1])), 4) for t in SNAP if t > 0},
                  forward={str(t): stat(fwd[t]) for t in SNAP},
                  backward={str(t): stat(bwd[t]) for t in SNAP},
                  corr_x0_xT=round(corr_T, 3),
                  corr_x0_xT_theory=round(float(np.sqrt(abar[-1]) * np.sqrt(MU**2 + S**2) / np.sqrt(abar[-1] * (MU**2 + S**2) + 1 - abar[-1])), 3),
                  backward_right_mode=dict(mean=round(float(right.mean()), 3), std=round(float(right.std()), 3)))
F, B = num["toy"]["forward"], num["toy"]["backward"]
assert abs(F["1000"]["mean"]) < 0.03 and abs(F["1000"]["std"] - 1) < 0.02 and abs(corr_T) < 0.03
assert 0.47 < B["0"]["frac_pos"] < 0.53 and abs(num["toy"]["backward_right_mode"]["mean"] - 2) < 0.03
assert abs(num["toy"]["backward_right_mode"]["std"] - 0.3) < 0.03
assert num["toy"]["corr_x0_xT_theory"] == 0.013 and num["toy"]["corr_x0_xT"] == 0.022

# 幾條個別樣本的軌跡（前向從 +2 出發；逆向從同一個 x_T 出發、跑兩次不同的隨機雜訊）
rng = np.random.default_rng(7)
paths_f = np.empty((4, T + 1)); paths_f[:, 0] = [2.0, 2.0, -2.0, -2.0]
for t in range(1, T + 1):
    paths_f[:, t] = np.sqrt(alpha[t - 1]) * paths_f[:, t - 1] + np.sqrt(beta[t - 1]) * rng.normal(size=4)
start = np.full(6, 0.3)
paths_b = np.empty((6, T + 1)); paths_b[:, T] = start
for t in range(T, 0, -1):
    ab = abar[t - 1]; xt = paths_b[:, t]
    eps_hat = (xt - np.sqrt(ab) * x0_hat(xt, ab)) / np.sqrt(1 - ab)
    mean = (xt - beta[t - 1] / np.sqrt(1 - ab) * eps_hat) / np.sqrt(alpha[t - 1])
    paths_b[:, t - 1] = mean + (np.sqrt(beta[t - 1]) * rng.normal(size=6) if t > 1 else 0.0)
num["paths"] = dict(seed=7, backward_start=0.3, backward_ends=[round(float(v), 2) for v in paths_b[:, 0]],
                    n_end_positive=int((paths_b[:, 0] > 0).sum()))
assert 1 <= num["paths"]["n_end_positive"] <= 5

# ---------- figure 1: lec14-kp01-pipelines ----------
fig, ax = plt.subplots(figsize=(9.6, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")
C = dict(data="#DBEAFE", fixed="#E5E7EB", noise="#FEF3C7", learned="#EDE9FE", vae="#BFDBFE")


def box(x, y, w, h, txt, fc, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=fc, ec="#374151", lw=1.3, ls=ls))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=10.5)


def arrow(x1, x2, y):
    ax.annotate("", (x2, y), (x1, y), arrowprops=dict(arrowstyle="->", lw=1.4, color="#374151"))


ax.text(0.05, 4.15, "Diffusion model", fontsize=12, weight="bold")
y = 2.75
box(0.1, y, 1.4, 0.9, "data\n$x_0$", C["data"])
box(2.1, y, 1.9, 0.9, "fixed encoder\nadd noise, $T$ steps", C["fixed"], ls="--")
box(4.6, y, 1.4, 0.9, "noise\n$x_T$", C["noise"])
box(6.6, y, 1.9, 0.9, "learned decoder\n$p_\\theta(x_{t-1}\\mid x_t)$", C["learned"])
box(9.0, y, 0.9, 0.9, "new\n$x_0$", C["data"])
for a, b in [(1.5, 2.1), (4.0, 4.6), (6.0, 6.6), (8.5, 9.0)]:
    arrow(a, b, y + 0.45)
ax.text(5.3, y - 0.3, "latents $x_1,\\ldots,x_T$ have the same size as $x_0$", ha="center", fontsize=9.5, color="#374151")
ax.text(0.05, 1.75, "VAE", fontsize=12, weight="bold")
y = 0.45
box(0.1, y, 1.4, 0.9, "data\n$x$", C["data"])
box(2.1, y, 1.9, 0.9, "learned encoder\n$q_\\phi(z\\mid x)$", C["vae"])
box(4.6, y, 1.4, 0.9, "latent\n$z$ (small)", C["fixed"])
box(6.6, y, 1.9, 0.9, "learned decoder\n$p_\\theta(x\\mid z)$", C["vae"])
box(9.0, y, 0.9, 0.9, "$\\hat{x}$", C["data"])
for a, b in [(1.5, 2.1), (4.0, 4.6), (6.0, 6.6), (8.5, 9.0)]:
    arrow(a, b, y + 0.45)
ax.text(5.3, y - 0.3, "both networks are trained together", ha="center", fontsize=9.5, color="#374151")
fig.tight_layout(); fig.savefig(OUT / "lec14-kp01-pipelines.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec14-kp01-forward-backward ----------
fig, axs = plt.subplots(2, 5, figsize=(12.5, 5.0), sharex=True, sharey=True)
bins = np.linspace(-4, 4, 81); grid = np.linspace(-4, 4, 400)
std_normal = np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi)
for j, t in enumerate(SNAP):
    for row, (src, col) in enumerate([(fwd, "#2563EB"), (bwd, "#7C3AED")]):
        ax = axs[row, j]
        ax.hist(src[t], bins=bins, density=True, color=col, alpha=0.75)
        ax.plot(grid, std_normal, color="#6B7280", ls=":", lw=1.2)
        s = num["toy"]["forward" if row == 0 else "backward"][str(t)]
        ax.set_title(f"$t={t}$" + (r",  $\sqrt{\bar\alpha_t}=$" + f"{np.sqrt(abar[t - 1]):.3f}" if t else ""), fontsize=10)
        ax.text(-3.8, 1.12, f"std {s['std']:.2f}", fontsize=8.5)
        ax.set_ylim(0, 1.3); ax.set_xlim(-4, 4)
    axs[1, j].set_xlabel("$x$")
axs[0, 0].set_ylabel("forward (fixed)\ndensity"); axs[1, 0].set_ylabel("backward (perfect denoiser)\ndensity")
axs[0, 4].plot([], [], color="#6B7280", ls=":", label=r"$\mathcal{N}(0,1)$"); axs[0, 4].legend(loc="upper right", fontsize=8.5)
fig.text(0.5, 0.965, "forward: read left to right            backward: read right to left (starts at $t=1000$)", ha="center", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "lec14-kp01-forward-backward.png", dpi=200); plt.close(fig)

# ---------- figure 3: lec14-kp01-trajectories ----------
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
tt = np.arange(T + 1)
for p in paths_f:
    axs[0].plot(tt, p, lw=0.8, color="#2563EB", alpha=0.8)
axs[0].set_title("forward: 4 samples start at $\\pm 2$", fontsize=10.5); axs[0].set_xlabel("step $t$"); axs[0].set_ylabel("$x_t$")
for p in paths_b:
    axs[1].plot(tt, p, lw=0.8, color="#7C3AED", alpha=0.8)
axs[1].set_xlim(T, 0)
axs[1].set_title("backward: 6 runs from the same $x_T=0.3$", fontsize=10.5); axs[1].set_xlabel("step $t$ (runs from 1000 down to 0)")
for ax in axs:
    ax.set_ylim(-4.2, 4.2); ax.axhline(2, color="#9CA3AF", lw=0.6, ls="--"); ax.axhline(-2, color="#9CA3AF", lw=0.6, ls="--")
fig.tight_layout(); fig.savefig(OUT / "lec14-kp01-trajectories.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
