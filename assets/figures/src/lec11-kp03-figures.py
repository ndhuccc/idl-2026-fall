"""lec11-kp03-figures.py — 第11週 KP03 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec11-kp03-figures.py [輸出資料夾]（預設：上一層的 assets/figures/）"""
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


def softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


num = {}
# ---------- illustrative example [Added]: the query "it" over (animal, it, tired) ----------
# head 1 (who does "it" refer to?) scores 3, 0, 0; head 2 (what is said about "it"?) scores 0, 0, 3; d_k = 2
a1 = softmax(np.array([3.0, 0, 0]) / np.sqrt(2))
a2 = softmax(np.array([0, 0, 3.0]) / np.sqrt(2))
# one head that must serve both relations: the two query directions added, d_k = 4
a_single = softmax(np.array([3.0, 0, 3.0]) / np.sqrt(4))
V = np.eye(3)                      # value of each word = its own label (animal, it, tired)
num["head1"] = a1.round(4).tolist(); num["head2"] = a2.round(4).tolist(); num["single"] = a_single.round(4).tolist()
num["head1_out"] = (a1 @ V).round(4).tolist(); num["head2_out"] = (a2 @ V).round(4).tolist()
num["concat"] = np.concatenate([a1 @ V, a2 @ V]).round(4).tolist()
num["single_out"] = (a_single @ V).round(4).tolist()
assert abs(a1[0] - 0.8066) < 1e-4 and abs(a_single[0] - 0.4498) < 1e-4
# ---------- source concat example ----------
H1 = np.array([[1.0, 0.0], [0.5, 0.5]]); H2 = np.array([[0.0, 2.0], [1.0, 1.0]])
num["source_concat"] = np.concatenate([H1, H2], axis=1).tolist()
# ---------- Transformer-Base: cost of 8 heads vs 1 full head ----------
d, h, n = 512, 8, 10
dk = d // h
params_multi = 3 * h * d * dk + d * d          # W_Q,i W_K,i W_V,i for each head + W_O (no biases)
params_single = 3 * d * d + d * d
mults_multi = h * n * n * dk                   # QK^T for every head
mults_single = n * n * d
num["base"] = dict(d_model=d, heads=h, d_k=dk, n=n, params_multi=params_multi, params_single=params_single,
                   score_mults_multi=mults_multi, score_mults_single=mults_single, attention_maps_multi=h, entries_multi=h * n * n,
                   entries_single=n * n)
assert params_multi == params_single == 1048576 and mults_multi == mults_single == 51200
# too many heads
num["heads_512"] = dict(h=512, d_k=d // 512)
print(json.dumps(num, indent=1))

# ---------- figure 1: two heads vs one head for the query "it" ----------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
words = ["animal", "it", "tired"]
for ax, w, title, col in ((axes[0], a1, "head 1: who is \"it\"?", "#1D4ED8"), (axes[1], a2, "head 2: what about \"it\"?", "#15803D"),
                          (axes[2], a_single, "one head doing both", "#BE123C")):
    ax.bar(range(3), w, color=col, width=0.6)
    for j, v in enumerate(w):
        ax.text(j, v + 0.02, f"{v:.4f}", ha="center", fontsize=11)
    ax.set_xticks(range(3), words, fontsize=11); ax.set_ylim(0, 1.0); ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("attention weight of the query \"it\"")
fig.tight_layout(); fig.savefig(OUT / "lec11-kp03-heads-vs-single.png", dpi=200); plt.close(fig)

# ---------- figure 2: shape flow of multi-head attention ----------
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 11); ax.set_ylim(0, 4.8); ax.axis("off")


def box(x, y, w, hh, txt, fc, ec, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + hh / 2, txt, ha="center", va="center", fontsize=fs)


def arr(a, b, color="#334155"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.3, color=color))


box(0.2, 1.9, 1.5, 1.0, "$X$\n$n\\times 512$", "#F1F5F9", "#64748B", 12)
ys = [3.9, 3.25, 2.6, 1.55, 0.9]
labels = ["head 1", "head 2", "head 3", "...", "head 8"]
for y, lab in zip(ys, labels):
    if lab == "...":
        ax.text(4.0, y + 0.25, r"$\vdots$", ha="center", fontsize=16)
        continue
    box(2.6, y, 2.8, 0.5, f"{lab}: attention on $n\\times 64$", "#DBEAFE", "#1D4ED8", 10)
    arr((1.72, 2.4), (2.58, y + 0.25))
    arr((5.42, y + 0.25), (6.18, 2.4))
ax.text(2.1, 4.55, r"$W_{Q,i},\ W_{K,i},\ W_{V,i}$: $512\times 64$ each", fontsize=10, color="#1D4ED8")
box(6.2, 1.9, 1.6, 1.0, "concat\n$n\\times 512$", "#FEF3C7", "#B45309", 11)
arr((7.82, 2.4), (8.38, 2.4))
box(8.4, 1.9, 1.0, 1.0, "$W_O$\n$512\\times512$", "#EDE9FE", "#7C3AED", 10)
arr((9.42, 2.4), (9.88, 2.4))
box(9.9, 1.9, 1.0, 1.0, "output\n$n\\times 512$", "#DCFCE7", "#15803D", 10)
ax.text(5.5, 0.25, r"$8\times 64 = 512$: same total width as one full head", ha="center", fontsize=11, color="#334155")
fig.tight_layout(); fig.savefig(OUT / "lec11-kp03-mha-shapes.png", dpi=200); plt.close(fig)
