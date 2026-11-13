"""lec10-kp02-figures.py — 第10週 KP02 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec10-kp02-figures.py [輸出資料夾]（預設：上一層的 assets/figures/）"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["mathtext.fontset"] = "cm"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT.mkdir(parents=True, exist_ok=True)

# ---------- numbers: linear RNN h_t = 0.5 h_{t-1} + x_t, h_0 = 0 ----------
r = 0.5


def context_weights(T):
    """weight of x_t inside c = h_T, for t = 1..T"""
    return [r ** (T - t) for t in range(1, T + 1)]


num = {"r": r}
num["T2"] = context_weights(2)
num["first_word_weight"] = {str(T): context_weights(T)[0] for T in (2, 5, 10, 20)}
xs = [1.0, 1.0, 1.0]
h = 0.0
hs = []
for x in xs:
    h = r * h + x
    hs.append(h)
num["practice1"] = dict(x=xs, h=hs, c=hs[-1], weights=context_weights(3), share_x1=round(context_weights(3)[0] / hs[-1], 4))
num["practice2_T"] = next(T for T in range(1, 50) if r ** (T - 1) < 0.01)
num["practice2_weight"] = r ** (num["practice2_T"] - 1)
assert num["first_word_weight"]["10"] == 0.5 ** 9 and num["practice2_T"] == 8
print(json.dumps(num, indent=1))


def box(ax, x, y, txt, fc, w=0.9, h=0.55, fs=14, ec="#334155"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03,rounding_size=0.1",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b, color="#334155", lw=1.6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=lw, color=color))


# ---------- figure 1: Seq2Seq with one context vector ----------
fig, ax = plt.subplots(figsize=(11, 4.0))
ax.set_xlim(-0.4, 11.4); ax.set_ylim(-0.6, 3.5); ax.axis("off")
src = ["Je", "suis", "étudiant"]
for k, w in enumerate(src):
    X = 0.7 + 1.6 * k
    box(ax, X, 0.2, w, "#DCFCE7", w=1.2)
    box(ax, X, 1.5, rf"$\mathbf{{h}}_{k+1}$", "#DBEAFE")
    arrow(ax, (X, 0.48), (X, 1.22))
    if k:
        arrow(ax, (X - 1.6 + 0.45, 1.5), (X - 0.45, 1.5), color="#1D4ED8")
box(ax, 5.55, 1.5, r"$\mathbf{c}=\mathbf{h}_3$", "#FDE68A", w=1.25, h=0.75, ec="#B45309", fs=15)
arrow(ax, (3.9 + 0.45, 1.5), (5.55 - 0.63, 1.5), color="#B45309", lw=2.2)
ax.text(5.55, 2.25, "fixed size", ha="center", fontsize=11, color="#B45309")
tgt = ["I", "am", "a", "student"]
for k, w in enumerate(tgt):
    X = 6.9 + 1.25 * k
    box(ax, X, 1.5, rf"$\mathbf{{l}}_{k+1}$", "#EDE9FE", w=0.75)
    box(ax, X, 2.85, w, "#FEF3C7", w=1.05)
    arrow(ax, (X, 1.78), (X, 2.57))
    if k:
        arrow(ax, (X - 1.25 + 0.38, 1.5), (X - 0.38, 1.5), color="#6D28D9")
        ax.add_patch(FancyArrowPatch((X - 1.25 + 0.3, 2.57), (X - 0.3, 1.78), arrowstyle="-|>", mutation_scale=10,
                                     lw=1.2, color="#047857"))
    ax.add_patch(FancyArrowPatch((5.55, 1.12), (X, 1.22), arrowstyle="-|>", mutation_scale=10, lw=1.0,
                                 color="#B45309", ls="--", connectionstyle="arc3,rad=0.25"))
ax.text(2.3, -0.45, "encoder: reads the source, step by step", ha="center", fontsize=11)
ax.text(8.8, -0.45, "decoder: writes the target; green = previous word fed back", ha="center", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "lec10-kp02-seq2seq-bottleneck.png", dpi=200); plt.close(fig)

# ---------- figure 2: how much of each word survives in c ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, T in zip(axes, (5, 20)):
    w = context_weights(T)
    ax.bar(range(1, T + 1), w, color=["#DC2626"] + ["#1D4ED8"] * (T - 1))
    ax.set_yscale("log"); ax.set_ylim(1e-6, 2); ax.set_xticks([1] + list(range(5, T + 1, 5)) if T > 5 else list(range(1, T + 1)))
    ax.set_xlabel("position $t$ of the source word"); ax.set_title(rf"$T={T}$: weight of $x_1$ is " + (rf"${w[0]:g}$" if w[0] > 1e-3 else r"$1.9\times10^{-6}$"), fontsize=12)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel(r"weight of $x_t$ inside $\mathbf{c}=h_T$ (log scale)")
fig.suptitle(r"Linear encoder $h_t=0.5\,h_{t-1}+x_t$: early words fade as $0.5^{T-t}$", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "lec10-kp02-early-words-fade.png", dpi=200); plt.close(fig)
