"""lec13-kp01-figures.py — 第13週 KP01 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec13-kp01-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
[補充] 一維小 GAN：真資料 x ~ N(2, 0.5^2)；生成器 G(z) = a + b z，z ~ N(0, 1)；
判別器 D(x) = sigmoid(w0 + w1 x + w2 x^2)。交替做梯度上升：每步先更新 D 十次，再更新 G 一次。"""
import json, sys, math
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
sig = lambda t: 1.0 / (1.0 + np.exp(-t))
MU, SD = 2.0, 0.5
SNAPS = (0, 50, 150, 400, 1000, 6000)


def train(steps=6000, lrD=0.1, lrG=0.005, n=256, k=10, seed=0):
    rng = np.random.default_rng(seed)
    a, b, w = 0.0, 1.0, np.zeros(3)
    F = lambda x: np.stack([np.ones_like(x), x, x * x], 1)
    out = {}
    traj = []
    for t in range(steps + 1):
        traj.append(abs(b))
        if t in SNAPS:
            out[t] = (a, b, w.copy())
        if t == steps:
            break
        for _ in range(k):                      # discriminator: ascend log D(real) + log(1 - D(fake))
            xr = rng.normal(MU, SD, n); xf = a + b * rng.normal(size=n)
            dr, df = sig(F(xr) @ w), sig(F(xf) @ w)
            w = w + lrD * (F(xr).T @ (1 - dr) / n - F(xf).T @ df / n)
        z = rng.normal(size=n); xf = a + b * z   # generator: ascend log D(G(z))
        df = sig(F(xf) @ w); gx = (1 - df) * (w[1] + 2 * w[2] * xf)
        a += lrG * np.mean(gx); b += lrG * np.mean(gx * z)
    out['traj'] = traj
    return out


snaps = train()
traj = snaps.pop('traj')
ev = np.random.default_rng(99); xr_ev = ev.normal(MU, SD, 100000); z_ev = ev.normal(size=100000)
rows = []
for t, (a, b, w) in snaps.items():
    D = lambda x: sig(w[0] + w[1] * x + w[2] * x * x)
    rows.append(dict(step=t, a=round(a, 3), b=round(abs(b), 3), D_real_mean=round(float(D(xr_ev).mean()), 3),
                     D_fake_mean=round(float(D(a + b * z_ev).mean()), 3), D_at_2=round(float(D(2.0)), 3), D_at_0=round(float(D(0.0)), 3)))
num["toy_gan"] = dict(real="N(2, 0.5^2)", generator="G(z)=a+bz, z~N(0,1), start a=0, b=1",
                      discriminator="D(x)=sigmoid(w0+w1 x+w2 x^2), start w=0", k_D_steps_per_G_step=10, lr_D=0.1, lr_G=0.005,
                      batch=256, steps=6000, seed=0, eval_samples=100000, eval_seed=99, snapshots=rows,
                      sd_min=dict(step=int(np.argmin(traj[:2000])), sd=round(float(min(traj[:2000])), 3)),
                      sd_max_after_min=dict(step=int(np.argmax(traj[600:2000]) + 600), sd=round(float(max(traj[600:2000])), 3)),
                      w_at_400=[round(float(v), 3) for v in snaps[400][2]], w_at_1000=[round(float(v), 3) for v in snaps[1000][2]])
r = {s["step"]: s for s in rows}
assert r[0]["D_real_mean"] == 0.5 and abs(r[6000]["a"] - 2) < 0.02 and abs(r[6000]["b"] - 0.5) < 0.02
assert abs(r[6000]["D_real_mean"] - 0.5) < 0.02 and abs(r[6000]["D_fake_mean"] - 0.5) < 0.02
# one discriminator reading by hand at step 50
a50, b50, w50 = snaps[50]
num["step50_by_hand"] = dict(w=[round(float(v), 3) for v in w50],
                             logit_at_2=round(float(w50[0] + 2 * w50[1] + 4 * w50[2]), 3), D_at_2=round(float(sig(w50[0] + 2 * w50[1] + 4 * w50[2])), 3),
                             logit_at_0=round(float(w50[0]), 3), D_at_0=round(float(sig(w50[0])), 3))

# ---------- figure 1: lec13-kp01-architecture ----------
fig, ax = plt.subplots(figsize=(10.5, 4.0)); ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.0); ax.axis("off")


def box(x0, y0, w, h, text, fc, ec, fs=11):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.05,rounding_size=0.12", fc=fc, ec=ec, lw=1.8))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(p, q, col="#334155", ls="-", lw=1.8):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", color=col, lw=lw, ls=ls))


box(0.1, 0.55, 1.6, 0.8, r"noise $\mathbf{z}\sim p_z$", "#FEF3C7", "#B45309", 11)
box(2.2, 0.45, 1.7, 1.0, "generator\n" + r"$G$", "#DBEAFE", "#1D4ED8")
box(4.4, 0.55, 1.6, 0.8, r"fake $G(\mathbf{z})$", "#F1F5F9", "#475569", 11)
box(0.1, 2.55, 1.6, 0.8, "real data", "#DCFCE7", "#15803D", 11)
box(2.3, 2.55, 1.5, 0.8, r"real $\mathbf{x}$", "#F1F5F9", "#475569", 11)
box(6.6, 1.4, 1.7, 1.2, "discriminator\n" + r"$D$", "#EDE9FE", "#7C3AED")
box(8.8, 1.55, 1.6, 0.9, r"$D(\cdot)\in[0,1]$" + "\n" + "P(real)", "#F1F5F9", "#475569", 10)
arrow((1.7, 0.95), (2.2, 0.95)); arrow((3.9, 0.95), (4.4, 0.95)); arrow((1.7, 2.95), (2.3, 2.95))
arrow((6.0, 0.95), (6.6, 1.7)); arrow((3.8, 2.95), (6.6, 2.3)); arrow((8.3, 2.0), (8.8, 2.0))
ax.annotate("", xy=(3.05, 0.42), xytext=(7.45, 1.38), arrowprops=dict(arrowstyle="->", color="#BE123C", lw=1.6, ls="--",
            connectionstyle="arc3,rad=-0.35"))
ax.text(8.0, 0.55, "G learns only\nthrough D's feedback", color="#BE123C", fontsize=9.5, ha="left")
ax.text(7.45, 3.15, "D: tell real from fake", color="#7C3AED", fontsize=10, ha="center")
ax.text(3.05, 1.62, "G: fool D", color="#1D4ED8", fontsize=10, ha="center")
fig.tight_layout(); fig.savefig(OUT / "lec13-kp01-architecture.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec13-kp01-toy-training ----------
xs = np.linspace(-3, 5, 801)
npdf = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))
fig, axs = plt.subplots(1, 5, figsize=(16, 3.7), sharey=True)
for ax, t in zip(axs, (50, 150, 400, 1000, 6000)):
    a, b, w = snaps[t]; b = abs(b)
    ax.fill_between(xs, npdf(xs, MU, SD), color="#93C5FD", alpha=0.5, label=r"real $p_{\mathrm{data}}$")
    ax.plot(xs, npdf(xs, a, b), color="#B45309", lw=2.2, label=r"generator $p_g$")
    ax2 = ax.twinx(); ax2.plot(xs, sig(w[0] + w[1] * xs + w[2] * xs * xs), color="#7C3AED", lw=2, ls="--", label=r"$D(x)$")
    ax2.axhline(0.5, color="#94A3B8", lw=1, ls=":"); ax2.set_ylim(0, 1.02); ax2.set_yticks([0, 0.5, 1])
    ax2.text(4.9, 0.52, r"$D=0.5$", color="#64748B", fontsize=8, ha="right")
    if t == 6000:
        ax2.set_ylabel(r"$D(x)$", color="#7C3AED")
    s = r[t]
    ax.set_title(f"step {t}: " + rf"$G(z)={s['a']:.2f}+{s['b']:.2f}z$", fontsize=10.5)
    ax.text(-2.8, 1.45, rf"avg $D$: real {s['D_real_mean']:.2f}, fake {s['D_fake_mean']:.2f}", fontsize=9)
    ax.set_xlim(-3, 5); ax.set_ylim(0, 1.6); ax.set_xlabel(r"$x$"); ax.grid(alpha=0.25)
axs[0].set_ylabel("density")
h1, l1 = axs[0].get_legend_handles_labels(); axs[0].legend(h1 + [plt.Line2D([], [], color="#7C3AED", ls="--")], l1 + [r"$D(x)$"], loc="upper left", fontsize=8.5, bbox_to_anchor=(0, 0.88))
fig.tight_layout(); fig.savefig(OUT / "lec13-kp01-toy-training.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
