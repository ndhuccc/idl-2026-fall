"""lec13-kp05-figures.py — 第13週 KP05 講義的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec13-kp05-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
[補充] 轉置卷積的輸出大小、DCGAN 生成器各層的形狀、100 維標準常態向量的長度與兩種插值的長度變化（種子 0）。"""
import json, sys, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams["mathtext.fontset"] = "cm"
args = sys.argv[1:]
REPORT = None
if "--report" in args:
    i = args.index("--report"); REPORT = Path(args[i + 1]); del args[i:i + 2]
OUT = Path(args[0]) if args else Path(__file__).resolve().parent.parent
OUT.mkdir(parents=True, exist_ok=True)
num = {}

# ---------- transposed convolution output size and a DCGAN-style generator ----------
tsize = lambda i, s, p, k: (i - 1) * s - 2 * p + k
shapes = [(4, 1024)]
for ch in (512, 256, 128, 3):
    shapes.append((tsize(shapes[-1][0], 2, 1, 4), ch))
num["dcgan_generator"] = dict(z_dim=100, first="project z to 4x4x1024", kernel=4, stride=2, padding=1,
                              shapes=[f"{s}x{s}x{c}" for s, c in shapes], formula="out=(in-1)*stride-2*padding+kernel")
assert [s for s, _ in shapes] == [4, 8, 16, 32, 64]

# ---------- latent vectors: norms and interpolation ----------
p = 100
rng = np.random.default_rng(0)
Z = rng.normal(size=(100000, p)); nz = np.linalg.norm(Z, axis=1)
z1, z2 = rng.normal(size=p), rng.normal(size=p)
c = float(z1 @ z2 / (np.linalg.norm(z1) * np.linalg.norm(z2))); th = math.acos(c)
lerp = lambda a: (1 - a) * z1 + a * z2
slerp = lambda a: (math.sin((1 - a) * th) * z1 + math.sin(a * th) * z2) / math.sin(th)
al = np.linspace(0, 1, 101)
nl = np.array([np.linalg.norm(lerp(a)) for a in al]); ns = np.array([np.linalg.norm(slerp(a)) for a in al])
num["latent_norms"] = dict(p=p, samples=100000, seed=0, mean=round(float(nz.mean()), 3), sd=round(float(nz.std()), 3),
                           pct1=round(float(np.percentile(nz, 1)), 2), pct99=round(float(np.percentile(nz, 99)), 2), sqrt_p=10.0,
                           share_below_8=round(float((nz < 8).mean()), 4))
num["interpolation"] = dict(norm_z1=round(float(np.linalg.norm(z1)), 3), norm_z2=round(float(np.linalg.norm(z2)), 3), cosine=round(c, 4),
                            theta_deg=round(math.degrees(th), 2),
                            lerp={str(a): round(float(np.linalg.norm(lerp(a))), 3) for a in (0, 0.25, 0.5, 0.75, 1)},
                            slerp={str(a): round(float(np.linalg.norm(slerp(a))), 3) for a in (0, 0.25, 0.5, 0.75, 1)},
                            share_of_prior_below_lerp_mid=round(float((nz < np.linalg.norm(lerp(0.5))).mean()), 5))
num["lerp_min"] = dict(alpha=round(float(al[nl.argmin()]), 2), norm=round(float(nl.min()), 3))
num["latent_norms"]["min_sample"] = round(float(nz.min()), 3)
num["sqrt_p_over_2"] = round(math.sqrt(p / 2), 3)
# unit orthogonal case from the source demo
u1, u2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
num["orthogonal_unit_mid"] = round(float(np.linalg.norm(0.5 * u1 + 0.5 * u2)), 4)

# ---------- [Added] cycle-consistency example ----------
x = np.array([0.2, 0.6, 0.9]); back = np.array([0.25, 0.5, 0.9])
num["cycle_l1_example"] = dict(x=x.tolist(), F_G_x=back.tolist(), L1=round(float(np.abs(back - x).sum()), 4))
# practice answers
tconv = lambda n, k=4, s=2, q=1: (n - 1) * s - 2 * q + k
conv = lambda n, k=4, s=2, q=1: (n + 2 * q - k) // s + 1
n, layers = 4, 0
while n < 128:
    n = tconv(n); layers += 1
a60 = np.array([1.0, 0.0]); b60 = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)])
num["practice"] = dict(tconv_16=tconv(16), disc_conv_64=conv(64), layers_4_to_128=layers,
                       lerp_mid_60deg=round(float(np.linalg.norm(0.5 * a60 + 0.5 * b60)), 4),
                       cycle_q3=round(float(np.abs(np.array([0.3, 0.4]) - np.array([0.1, 0.4])).sum()), 4))

# ---------- figure 1: lec13-kp05-cgan ----------
fig, ax = plt.subplots(figsize=(10.5, 4.0)); ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.0); ax.axis("off")


def box(x0, y0, w, h, text, fc, ec, fs=11):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.05,rounding_size=0.12", fc=fc, ec=ec, lw=1.8))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(p0, p1, col="#334155"):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="->", color=col, lw=1.8))


box(0.1, 0.4, 1.3, 0.7, r"noise $\mathbf{z}$", "#FEF3C7", "#B45309")
box(0.1, 1.45, 1.3, 0.7, r"label $\mathbf{y}$", "#FCE7F3", "#BE185D")
box(1.9, 0.75, 1.6, 1.0, "generator\n" + r"$G(\mathbf{z}\mid\mathbf{y})$", "#DBEAFE", "#1D4ED8")
box(4.0, 0.9, 1.4, 0.7, r"fake $\hat{\mathbf{x}}$", "#F1F5F9", "#475569")
box(4.0, 2.55, 1.4, 0.7, r"real $\mathbf{x}$", "#F1F5F9", "#475569")
box(6.0, 1.45, 1.9, 1.1, "discriminator\n" + r"$D(\mathbf{x}\mid\mathbf{y})$", "#EDE9FE", "#7C3AED")
box(8.4, 1.6, 1.9, 0.8, "P(real and\nmatches y)", "#F1F5F9", "#475569", 10)
arrow((1.4, 0.75), (1.9, 1.05)); arrow((1.4, 1.8), (1.9, 1.45)); arrow((3.5, 1.25), (4.0, 1.25))
arrow((5.4, 1.25), (6.0, 1.75)); arrow((5.4, 2.9), (6.0, 2.3)); arrow((7.9, 2.0), (8.4, 2.0))
ax.plot([0.75, 0.75, 6.95], [2.15, 3.55, 3.55], color="#BE185D", lw=1.6, ls="--")
ax.annotate("", xy=(6.95, 2.57), xytext=(6.95, 3.55), arrowprops=dict(arrowstyle="->", color="#BE185D", lw=1.6, ls="--"))
ax.text(3.0, 3.65, r"the same label $\mathbf{y}$ also goes to $D$", color="#BE185D", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "lec13-kp05-cgan.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec13-kp05-dcgan-shapes ----------
fig, ax = plt.subplots(figsize=(10.5, 3.6)); ax.set_xlim(0, 10.5); ax.set_ylim(0, 3.6); ax.axis("off")
ax.add_patch(Rectangle((0.2, 1.2), 0.25, 1.2, fc="#FEF3C7", ec="#B45309")); ax.text(0.33, 0.9, r"$\mathbf{z}$: 100", ha="center", fontsize=10)
xs = [1.0, 3.0, 4.9, 6.7, 8.5]
for (s, ch), x0 in zip(shapes, xs):
    h = 0.35 + 2.5 * (s / 64); w = 0.25 + 1.0 * (ch / 1024)
    col = "#1D4ED8" if ch != 3 else "#15803D"
    ax.add_patch(Rectangle((x0, 1.8 - h / 2), w, h, fc=col, alpha=0.18, ec=col, lw=1.6))
    ax.text(x0 + w / 2, 1.8 - h / 2 - 0.3, rf"${s}\times{s}\times{ch}$", ha="center", fontsize=10)
for x0, x1 in zip([0.45] + [xs[k] + 0.25 + 1.0 * (shapes[k][1] / 1024) for k in range(4)], xs):
    ax.annotate("", xy=(x1 - 0.05, 1.8), xytext=(x0 + 0.05, 1.8), arrowprops=dict(arrowstyle="->", color="#334155", lw=1.5))
ax.text(5.2, 3.35, r"after projecting $\mathbf{z}$: transposed conv, kernel $4$, stride $2$, padding $1$; size doubles each layer", ha="center", fontsize=10)
ax.text(9.15, 1.8, "image\n(tanh)", ha="left", va="center", fontsize=9.5, color="#15803D")
fig.tight_layout(); fig.savefig(OUT / "lec13-kp05-dcgan-shapes.png", dpi=200); plt.close(fig)

# ---------- figure 3: lec13-kp05-lerp-slerp ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
ax = axs[0]
ax.hist(nz, bins=80, color="#93C5FD", edgecolor="white", density=True)
ax.axvline(float(np.linalg.norm(lerp(0.5))), color="#BE123C", lw=2, ls="--")
ax.text(float(np.linalg.norm(lerp(0.5))) + 0.1, 0.45, "Lerp\nmidpoint", color="#BE123C", fontsize=9.5)
ax.set_xlabel(r"length $\Vert\mathbf{z}\Vert$ of $\mathbf{z}\sim\mathcal{N}(\mathbf{0},I)$, $p=100$"); ax.set_ylabel("density")
ax.set_xlim(6, 13); ax.grid(alpha=0.3); ax.set_title(r"100000 samples: lengths crowd near $\sqrt{100}=10$", fontsize=10.5)
ax = axs[1]
ax.axhspan(float(np.percentile(nz, 1)), float(np.percentile(nz, 99)), color="#93C5FD", alpha=0.35, label="prior: middle 98%")
ax.plot(al, nl, color="#BE123C", lw=2.4, label="Lerp (straight line)")
ax.plot(al, ns, color="#15803D", lw=2.4, label="Slerp (arc)")
ax.set_xlim(0, 1); ax.set_ylim(4.6, 12.5); ax.set_xlabel(r"interpolation weight $\alpha$"); ax.set_ylabel(r"$\Vert\mathbf{z}(\alpha)\Vert$")
ax.legend(loc="lower left", fontsize=8.4, ncol=3, columnspacing=0.8, handlelength=1.4); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "lec13-kp05-lerp-slerp.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
