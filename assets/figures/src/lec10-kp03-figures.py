"""lec10-kp03-figures.py — 第10週 KP03 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec10-kp03-figures.py [輸出資料夾]（預設：上一層的 assets/figures/）"""
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
# source's Bahdanau example: two encoder states, scores 1.0 and 2.0
H2 = np.array([[1.0, 0.0], [0.0, 1.0]])
a = softmax(np.array([1.0, 2.0]))
num["bahdanau"] = dict(e=[1.0, 2.0], exp=[round(float(np.exp(1)), 3), round(float(np.exp(2)), 3)],
                       denom=round(float(np.exp(1) + np.exp(2)), 3), a=a.round(4).tolist(), c=(a @ H2).round(4).tolist())
# translation example [Added]: encoder states one-hot, decoder queries
src = ["Je", "suis", "étudiant"]
tgt = ["I", "am", "a", "student"]
H = np.eye(3)
Lq = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.8, 1.2], [0.0, 0.0, 2.0]])
S = Lq @ H.T
A = np.vstack([softmax(r) for r in S])
C = A @ H
num["translation"] = dict(src=src, tgt=tgt, queries=Lq.tolist(), scores=S.tolist(), A=A.round(4).tolist(),
                          row_sums=A.sum(1).round(6).tolist(), contexts=C.round(4).tolist(), n_scores=int(S.size))
# score functions (source numbers) + additive [Added]
q, k = np.array([1.0, 2.0]), np.array([2.0, 1.0])
Wa = np.diag([1.0, 2.0])
num["scores"] = dict(dot=float(q @ k), general=float(q @ Wa @ k), scaled=round(float(q @ k / np.sqrt(2)), 4),
                     additive=round(float(np.ones(2) @ np.tanh(q + k)), 4), tanh_3=round(float(np.tanh(3)), 4))
# practice [Added]
num["practice1"] = dict(e=[0.0, 1.0, 3.0], a=softmax(np.array([0.0, 1.0, 3.0])).round(4).tolist())
p1 = softmax(np.array([0.0, 1.0, 3.0]))
num["practice1"]["c"] = (p1 @ np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])).round(4).tolist()
num["practice2"] = dict(Tx=20, Ty=25, n=20 * 25)
assert np.allclose(num["bahdanau"]["a"], [0.2689, 0.7311])
print(json.dumps(num, indent=1, ensure_ascii=False))

# ---------- figure 1: alignment matrix ----------
fig, ax = plt.subplots(figsize=(5.8, 5.2))
im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ax.text(j, i, f"{A[i, j]:.4f}", ha="center", va="center", fontsize=12, color="white" if A[i, j] > 0.55 else "#0F172A")
ax.set_xticks(range(3), src, fontsize=12); ax.set_yticks(range(4), tgt, fontsize=12)
ax.set_xlabel(r"source word $j$ (encoder state $\mathbf{h}_j$)"); ax.set_ylabel(r"target word $i$ being written")
ax.set_title(r"Attention weights $a_{ij}$ (each row sums to 1)", fontsize=12)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(OUT / "lec10-kp03-alignment-matrix.png", dpi=200); plt.close(fig)

# ---------- figure 2: one decoding step ----------
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.set_xlim(-0.3, 10.3); ax.set_ylim(-0.3, 4.4); ax.axis("off")


def box(x, y, t, fc, w=1.5, h=0.62, fs=14, ec="#334155"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03,rounding_size=0.1", fc=fc, ec=ec, lw=1.4))
    ax.text(x, y, t, ha="center", va="center", fontsize=fs)


row = A[3]
xs = [1.2, 3.4, 5.6]
for x, w, wgt in zip(xs, src, row):
    box(x, 0.4, rf"$\mathbf{{h}}$: {w}", "#DBEAFE", w=1.9)
    ax.add_patch(FancyArrowPatch((8.2, 3.45), (x, 0.75), arrowstyle="-", lw=0.6 + 9 * wgt, color="#1D4ED8", alpha=0.75,
                                 connectionstyle="arc3,rad=0.12"))
    ax.text(x + 0.05, 1.25, f"$a={wgt:.4f}$", ha="center", fontsize=12, color="#1D4ED8",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
box(8.2, 3.75, r"$\mathbf{l}_{3}$ (query)", "#EDE9FE", w=2.0)
box(8.2, 1.6, r"$\mathbf{c}_4=\sum_j a_{4j}\mathbf{h}_j$", "#FDE68A", w=2.6, ec="#B45309")
ax.text(8.2, 2.25, "weighted sum of the three h", ha="center", fontsize=10, color="#B45309")
box(8.2, 0.4, 'write "student"', "#FEF3C7", w=2.2)
ax.add_patch(FancyArrowPatch((8.2, 1.27), (8.2, 0.72), arrowstyle="-|>", mutation_scale=12, color="#334155"))
ax.text(0.0, 3.95, "compare the query with every h; line width = weight", fontsize=11, color="#475569")
fig.tight_layout(); fig.savefig(OUT / "lec10-kp03-one-step.png", dpi=200); plt.close(fig)
