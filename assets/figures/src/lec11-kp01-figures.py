"""lec11-kp01-figures.py — 第11週 KP01 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec11-kp01-figures.py [輸出資料夾]（預設：上一層的 assets/figures/）"""
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


def softmax_rows(S):
    E = np.exp(S - S.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


num = {}
# ---------- Transformer-Base settings (source) and a per-layer parameter count [Added] ----------
d, h, dff, N, V = 512, 8, 2048, 6, 37000
dk = d // h
attn = 4 * (d * d + d)                 # W_Q, W_K, W_V, W_O with biases
ffn = (d * dff + dff) + (dff * d + d)  # two linear layers with biases
ln = 2 * d                             # gamma and beta
enc_layer = attn + ffn + 2 * ln
dec_layer = 2 * attn + ffn + 3 * ln
emb = V * d
num["base"] = dict(d_model=d, heads=h, d_k=dk, d_ff=dff, N=N, vocab=V)
num["params"] = dict(attention=attn, ffn=ffn, layernorm=ln, encoder_layer=enc_layer, decoder_layer=dec_layer,
                     encoder_stack=N * enc_layer, decoder_stack=N * dec_layer, embedding=emb,
                     ffn_share_of_encoder_layer=round(ffn / enc_layer, 4))
assert dk == 64 and dff == 4 * d and enc_layer == 3152384 and dec_layer == 4204032
# shapes for a 4-word sentence
num["shapes"] = dict(n=4, input=[4, d], after_attention=[4, d], after_ffn=[4, d], ffn_hidden=[4, dff])

# ---------- self-attention ignores word order [Added]: W_Q = W_K = W_V = I ----------
words = ["wo", "da", "ni"]          # 我、打、你
E = {"wo": np.array([1.0, 0.0]), "da": np.array([0.0, 1.0]), "ni": np.array([1.0, 1.0])}


def self_attn(order):
    X = np.stack([E[w] for w in order])
    A = softmax_rows(X @ X.T / np.sqrt(2))
    return {w: z for w, z in zip(order, A @ X)}


Z1 = self_attn(["wo", "da", "ni"])   # 我打你
Z2 = self_attn(["ni", "da", "wo"])   # 你打我
for w in words:
    assert np.allclose(Z1[w], Z2[w])
num["permutation"] = {w: Z1[w].round(4).tolist() for w in words}
print(json.dumps(num, indent=1))

# ---------- figure 1: the architecture ----------
fig, ax = plt.subplots(figsize=(10.5, 7.2))
ax.set_xlim(0, 10.5); ax.set_ylim(0, 7.4); ax.axis("off")


def box(x, y, w, hgt, txt, fc, ec, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, hgt, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + hgt / 2, txt, ha="center", va="center", fontsize=fs)


def arr(a, b, color="#334155", lw=1.5, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


# encoder tower
ax.add_patch(FancyBboxPatch((0.9, 1.75), 3.4, 3.05, boxstyle="round,pad=0.05", fc="#EFF6FF", ec="#1D4ED8", lw=1.6, ls="--"))
ax.text(0.65, 3.3, r"$\times N$", fontsize=14, color="#1D4ED8", ha="center")
box(1.1, 0.5, 3.0, 0.55, "input embedding  +  positional encoding", "#F1F5F9", "#64748B", 10)
box(1.1, 1.95, 3.0, 0.6, "multi-head self-attention", "#DBEAFE", "#1D4ED8")
box(1.1, 2.6, 3.0, 0.38, "add & norm", "#FEF3C7", "#B45309", 10)
box(1.1, 3.35, 3.0, 0.6, "feed-forward", "#DCFCE7", "#15803D")
box(1.1, 4.0, 3.0, 0.38, "add & norm", "#FEF3C7", "#B45309", 10)
arr((2.6, 1.06), (2.6, 1.93))
arr((2.6, 2.99), (2.6, 3.33))
arr((2.6, 4.39), (2.6, 5.3))
ax.text(2.6, 0.15, "source: Je suis etudiant", ha="center", fontsize=10, color="#334155")
ax.text(2.6, 5.45, r"encoder output  $n\times d_{\mathrm{model}}$", ha="center", fontsize=11, color="#1D4ED8")
ax.text(2.6, 6.7, "ENCODER", ha="center", fontsize=14, color="#1D4ED8", weight="bold")
# decoder tower
ax.add_patch(FancyBboxPatch((6.1, 1.75), 3.4, 4.25, boxstyle="round,pad=0.05", fc="#F5F3FF", ec="#7C3AED", lw=1.6, ls="--"))
ax.text(9.8, 3.9, r"$\times N$", fontsize=14, color="#7C3AED", ha="center")
box(6.3, 0.5, 3.0, 0.55, "output embedding  +  positional encoding", "#F1F5F9", "#64748B", 10)
box(6.3, 1.95, 3.0, 0.6, "masked multi-head self-attention", "#EDE9FE", "#7C3AED", 10)
box(6.3, 2.6, 3.0, 0.38, "add & norm", "#FEF3C7", "#B45309", 10)
box(6.3, 3.35, 3.0, 0.6, "cross-attention", "#EDE9FE", "#7C3AED")
box(6.3, 4.0, 3.0, 0.38, "add & norm", "#FEF3C7", "#B45309", 10)
box(6.3, 4.55, 3.0, 0.6, "feed-forward", "#DCFCE7", "#15803D")
box(6.3, 5.2, 3.0, 0.38, "add & norm", "#FEF3C7", "#B45309", 10)
box(6.3, 6.35, 3.0, 0.5, "linear  +  softmax", "#FCE7F3", "#BE185D")
arr((7.8, 1.06), (7.8, 1.93))
arr((7.8, 2.99), (7.8, 3.33))
arr((7.8, 4.39), (7.8, 4.53))
arr((7.8, 5.59), (7.8, 6.33))
ax.text(7.8, 0.15, "previous outputs, shifted right", ha="center", fontsize=10, color="#334155")
ax.text(7.8, 7.05, "next-word probabilities", ha="center", fontsize=11, color="#BE185D")
# encoder -> cross-attention (K, V)
arr((4.15, 5.3), (6.28, 3.65), color="#1D4ED8", lw=2.0, rad=-0.15)
ax.text(5.15, 4.75, r"$K, V$", fontsize=13, color="#1D4ED8", ha="center")
ax.text(7.95, 3.16, r"$Q$ (from decoder)", fontsize=10, color="#7C3AED", ha="left", va="center")
ax.text(7.8, 7.35, "DECODER", ha="center", fontsize=14, color="#7C3AED", weight="bold")
fig.tight_layout(); fig.savefig(OUT / "lec11-kp01-architecture.png", dpi=200); plt.close(fig)

# ---------- figure 2: permutation ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
cols = {"wo": "#1D4ED8", "da": "#15803D", "ni": "#BE185D"}
names = {"wo": "wo (I)", "da": "da (hit)", "ni": "ni (you)"}
for ax, order, title in ((axes[0], ["wo", "da", "ni"], 'order 1: "wo da ni"'), (axes[1], ["ni", "da", "wo"], 'order 2: "ni da wo"')):
    Z = self_attn(order)
    for pos, w in enumerate(order):
        for k in range(2):
            x = pos * 3 + k
            ax.bar(x, Z[w][k], width=0.8, color=cols[w], alpha=1.0 if k == 0 else 0.55)
            ax.text(x, Z[w][k] + 0.02, f"{Z[w][k]:.4f}", ha="center", fontsize=9)
        ax.text(pos * 3 + 0.5, -0.12, f"position {pos + 1}\n{names[w]}", ha="center", va="top", fontsize=10, color=cols[w])
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_ylim(0, 1.0); ax.set_xlim(-0.8, 8.8)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel(r"new representation $\mathbf{z}$ (two components)")
fig.tight_layout(); fig.subplots_adjust(bottom=0.24)
fig.savefig(OUT / "lec11-kp01-permutation.png", dpi=200); plt.close(fig)
