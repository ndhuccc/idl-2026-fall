"""lec13-kp02-figures.py — 第13週 KP02 講義的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec13-kp02-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
[補充] 沿用 KP01 的一維小 GAN（真資料 N(2, 0.5^2)、G(z)=a+bz、D(x)=sigmoid(w0+w1 x+w2 x^2)），
但生成器從 a=-3 出發，比較兩種生成器損失：飽和的 log(1-D(G(z)))（要變小）與非飽和的 -log D(G(z))（要變小）。"""
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
sig = lambda t: 1.0 / (1.0 + np.exp(-t))
V = lambda dr, df: math.log(dr) + math.log(1 - df)

# ---------- value of the game for a few (D(x), D(G(z))) pairs ----------
pairs = [(0.9, 0.1), (0.5, 0.5), (0.9, 0.9), (0.99, 0.01)]
num["value_examples"] = [dict(D_real=a, D_fake=b, V=round(V(a, b), 4)) for a, b in pairs]
# a batch example: two real, two fake scores
dr, df = [0.8, 0.6], [0.3, 0.1]
Vb = np.mean(np.log(dr)) + np.mean(np.log(1 - np.array(df)))
bce = -(np.mean(np.log(dr)) + np.mean(np.log(1 - np.array(df))))
num["batch_example"] = dict(D_real=dr, D_fake=df, V=round(float(Vb), 4), D_loss_bce_sum_of_two_means=round(float(bce), 4),
                            term_real=round(float(np.mean(np.log(dr))), 4), term_fake=round(float(np.mean(np.log(1 - np.array(df)))), 4))

# ---------- gradient w.r.t. the discriminator logit h for the two generator losses ----------
gr = {}
for d in (0.01, 0.1, 0.5, 0.9):
    gr[str(d)] = dict(saturating=round(d, 4), non_saturating=round(1 - d, 4))   # |d/dh log(1-sig(h))| = D ; |d/dh -log sig(h)| = 1-D
num["grad_wrt_logit"] = gr
num["grad_wrt_D_at_0.01"] = dict(saturating=round(1 / (1 - 0.01), 4), non_saturating=round(1 / 0.01, 1))


# ---------- toy GAN: the two generator losses from a far start ----------
def train(loss, steps=2000, lrD=0.1, lrG=0.005, n=256, k=10, seed=0, a0=-3.0):
    rng = np.random.default_rng(seed); a, b, w = a0, 1.0, np.zeros(3)
    F = lambda x: np.stack([np.ones_like(x), x, x * x], 1); traj = []; dm = 0.5
    for t in range(steps + 1):
        traj.append((t, a, abs(b), dm))
        if t == steps:
            break
        for _ in range(k):
            xr = rng.normal(2, 0.5, n); xf = a + b * rng.normal(size=n)
            w = w + lrD * (F(xr).T @ (1 - sig(F(xr) @ w)) / n - F(xf).T @ sig(F(xf) @ w) / n)
        z = rng.normal(size=n); xf = a + b * z; d = sig(F(xf) @ w); dm = float(d.mean()); dl = w[1] + 2 * w[2] * xf
        gx = (1 - d) * dl if loss == "non_saturating" else d * dl     # ascent direction on x for each loss
        a += lrG * np.mean(gx); b += lrG * np.mean(gx * z)
    return traj


tr = {L: train(L) for L in ("non_saturating", "saturating")}
pick = lambda L, t: tr[L][t]
num["toy_far_start"] = dict(real="N(2, 0.5^2)", start="a=-3, b=1", k=10, lr_D=0.1, lr_G=0.005, batch=256, seed=0, steps=2000,
                           at={L: {str(t): [round(pick(L, t)[1], 3), round(pick(L, t)[2], 3)] for t in (0, 250, 500, 1000, 2000)} for L in tr},
                           mean_D_on_fakes_saturating={str(t): round(pick('saturating', t)[3], 4) for t in (100, 500, 1000, 2000)},
                           note='a, |b| at the given steps; mean_D is the batch mean of D(G(z)) used by the previous generator update')
sd = [r_[3] for r_ in tr['saturating'][1:]]
num['toy_far_start']['saturating_mean_D_stats'] = dict(max_steps_1_to_1000=round(max(sd[:1000]), 4), max_all=round(max(sd), 4), min_all=round(min(sd), 4))
assert pick("non_saturating", 500)[1] > 1.9 and pick("saturating", 2000)[1] < -2
# ---------- figure 1: lec13-kp02-value-surface ----------
g = np.linspace(0.01, 0.99, 400); DF, DR = np.meshgrid(g, g)
Z = np.log(DR) + np.log(1 - DF)
fig, ax = plt.subplots(figsize=(5.6, 5.0))
cs = ax.contourf(DF, DR, Z, levels=np.linspace(-9.2, 0, 24), cmap="viridis")
ax.contour(DF, DR, Z, levels=[-4, -2, -1.386, -0.5], colors="white", linewidths=0.8)
fig.colorbar(cs, ax=ax, label=r"$V=\log D(\mathbf{x})+\log(1-D(G(\mathbf{z})))$")
for (a, b), lab in zip(((0.1, 0.9), (0.5, 0.5)), (r"$V=-0.21$", r"$V=-1.39$")):
    ax.plot(a, b, "o", color="white", mec="black", ms=7); ax.text(a + 0.03, b - 0.07, lab, color="black", fontsize=10, bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
ax.annotate("", xy=(0.20, 0.96), xytext=(0.42, 0.70), arrowprops=dict(arrowstyle="->", color="#F472B6", lw=2.2))
ax.text(0.38, 0.80, "D: max V", color="#F472B6", fontsize=10.5)
ax.annotate("", xy=(0.80, 0.30), xytext=(0.55, 0.30), arrowprops=dict(arrowstyle="->", color="#FDE68A", lw=2.2))
ax.text(0.56, 0.22, "G: min V", color="#FDE68A", fontsize=10.5)
ax.set_xlabel(r"score on fakes $D(G(\mathbf{z}))$"); ax.set_ylabel(r"score on real $D(\mathbf{x})$")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(OUT / "lec13-kp02-value-surface.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec13-kp02-saturation ----------
d = np.linspace(0.001, 0.999, 800)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
ax = axs[0]
ax.plot(d, np.log(1 - d), color="#BE123C", lw=2.4, label=r"saturating: $\log(1-D)$ (G minimises)")
ax.plot(d, -np.log(d), color="#15803D", lw=2.4, label=r"non-saturating: $-\log D$ (G minimises)")
ax.axvspan(0, 0.1, color="#FDE68A", alpha=0.4); ax.text(0.12, 4.2, "early training:\nD(G(z)) near 0", fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(-5, 5); ax.set_xlabel(r"$D(G(\mathbf{z}))$"); ax.set_ylabel("generator loss"); ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8.8)
ax = axs[1]
ax.plot(d, d, color="#BE123C", lw=2.4, label=r"saturating: $D$")
ax.plot(d, 1 - d, color="#15803D", lw=2.4, label=r"non-saturating: $1-D$")
for x0 in (0.01,):
    ax.plot([x0], [x0], "o", color="#BE123C"); ax.plot([x0], [1 - x0], "o", color="#15803D")
    ax.text(x0 + 0.04, 0.15, r"$0.01$", color="#BE123C"); ax.text(x0 + 0.04, 0.84, r"$0.99$", color="#15803D")
ax.axvspan(0, 0.1, color="#FDE68A", alpha=0.4)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.set_xlabel(r"$D(G(\mathbf{z}))$"); ax.set_ylabel(r"size of gradient w.r.t. logit $h$")
ax.legend(loc="center right", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "lec13-kp02-saturation.png", dpi=200); plt.close(fig)

# ---------- figure 3: lec13-kp02-far-start ----------
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for L, col, lab in (("non_saturating", "#15803D", r"non-saturating $-\log D(G(z))$"), ("saturating", "#BE123C", r"saturating $\log(1-D(G(z)))$")):
    t = [r[0] for r in tr[L]]; a = [r[1] for r in tr[L]]
    ax.plot(t, a, color=col, lw=2.2, label=lab)
ax.axhline(2, color="#1D4ED8", ls="--", lw=1.4); ax.text(1990, 2.12, "real mean = 2", color="#1D4ED8", ha="right", fontsize=9.5)
ax.set_xlim(0, 2000); ax.set_ylim(-3.5, 3); ax.set_xlabel("training step"); ax.set_ylabel(r"generator mean $a$")
ax.legend(loc="center right", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "lec13-kp02-far-start.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
