"""lec11-kp05-figures.py — 第11週 KP05 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec11-kp05-figures.py [輸出資料夾]（預設：上一層的 assets/figures/）"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT.mkdir(parents=True, exist_ok=True)
num = {}


def softmax_rows(S):
    S = np.where(np.isneginf(S), -np.inf, S)
    E = np.exp(S - S.max(axis=-1, keepdims=True))
    return E / E.sum(axis=-1, keepdims=True)


# ---------- causal mask: source example (scores already scaled) ----------
S = np.array([[2.0, 3, 1], [1, 4, 5], [2, 1, 3]])
n = 3
M = np.where(np.arange(n)[None, :] > np.arange(n)[:, None], -np.inf, 0.0)
A = softmax_rows(S + M)
A_full = softmax_rows(S)
num["mask"] = dict(S=S.tolist(), weights=A.round(4).tolist(), weights_without_mask=A_full.round(4).tolist())
assert abs(A[1, 0] - 0.0474) < 1e-4 and abs(A[2, 2] - 0.6652) < 1e-4
# [Added] changing the future word does not change earlier outputs
rng = np.random.default_rng(1)
X = rng.normal(0, 1, (3, 2)); X2 = X.copy(); X2[2] = [5.0, -5.0]   # replace word 3


def causal_self_attn(X):
    Sx = X @ X.T / np.sqrt(2)
    return softmax_rows(Sx + M) @ X


Z1, Z2 = causal_self_attn(X), causal_self_attn(X2)
num["future_change"] = dict(rows_1_2_unchanged=bool(np.array_equal(Z1[:2], Z2[:2])), row_3_changed=bool(not np.allclose(Z1[2], Z2[2])))
assert num["future_change"]["rows_1_2_unchanged"] and num["future_change"]["row_3_changed"]

# ---------- cross-attention: source example ----------
q = np.array([1.0, 1.0]); K = np.array([[2.0, 0], [0, 2], [2, 2]])
sc = K @ q / np.sqrt(2)
w = softmax_rows(sc[None])[0]
num["cross"] = dict(scores=(K @ q).tolist(), scaled=sc.round(4).tolist(), weights=w.round(4).tolist())
# shapes: Base, T_x = 5 source words, T_y = 4 target words
num["cross_shapes"] = dict(T_x=5, T_y=4, Q=[4, 64], K=[5, 64], weights=[4, 5], out=[4, 64], mask_self=[4, 4])

# ---------- output layer ----------
d, V = 512, 37000
num["output"] = dict(d_model=d, vocab=V, linear_params_no_bias=d * V)
# temperature: source two-logit example
z2 = np.array([4.0, 2.0])
num["temp_source"] = {str(T): softmax_rows((z2 / T)[None])[0].round(4).tolist() for T in (0.5, 1.0, 2.0)}
# [Added] five candidate words after "the cat sat on the"
words = ["mat", "floor", "sofa", "roof", "car"]
z = np.array([3.0, 2.0, 1.5, 0.5, -1.0])
dist = {}
for T in (0.5, 1.0, 2.0):
    p = softmax_rows((z / T)[None])[0]
    dist[str(T)] = p.round(4).tolist()
num["words"] = words; num["logits"] = z.tolist(); num["temp_words"] = dist
p1 = softmax_rows(z[None])[0]
order = np.argsort(-p1); cum = np.cumsum(p1[order])
k_p = int(np.searchsorted(cum, 0.9) + 1)
nucleus = [words[i] for i in order[:k_p]]
renorm = (p1[order[:k_p]] / p1[order[:k_p]].sum())
num["top_p_0.9"] = dict(cumulative=cum.round(4).tolist(), kept=nucleus, renormalized=renorm.round(4).tolist())
top2 = p1[order[:2]] / p1[order[:2]].sum()
num["top_k_2"] = dict(kept=[words[i] for i in order[:2]], renormalized=top2.round(4).tolist())
print(json.dumps(num, indent=1))

# ---------- figure 1: causal mask ----------
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
lab = ["pos 1", "pos 2", "pos 3"]
panels = [(S, "scores  $QK^\\top/\\sqrt{d_k}$", "s"), (S + M, "scores + mask $M$", "m"), (A, "weights after softmax", "w")]
for ax, (Mtx, title, kind) in zip(axes, panels):
    show = np.where(np.isneginf(Mtx), np.nan, Mtx)
    cmap = plt.get_cmap("Blues" if kind == "w" else "Oranges").copy(); cmap.set_bad("#334155")
    ax.imshow(show, cmap=cmap, vmin=0, vmax=1 if kind == "w" else 6)
    for i in range(3):
        for j in range(3):
            v = Mtx[i, j]
            if np.isneginf(v):
                ax.text(j, i, r"$-\infty$", ha="center", va="center", color="white", fontsize=15)
            else:
                txt = f"{v:.4f}" if kind == "w" else f"{v:g}"
                dark = (kind == "w" and v > 0.55) or (kind != "w" and v > 3.8)
                ax.text(j, i, txt, ha="center", va="center", fontsize=13, color="white" if dark else "#0F172A")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"key {l}" for l in lab]); ax.set_yticks(range(3)); ax.set_yticklabels([f"query {l}" for l in lab])
    ax.set_title(title, fontsize=12)
    ax.tick_params(length=0)
for ax in axes[1:]:
    ax.plot([0.5, 2.5, 2.5, 1.5, 1.5, 0.5, 0.5], [-0.5, -0.5, 1.5, 1.5, 0.5, 0.5, -0.5], color="#B91C1C", lw=2.5)
axes[2].text(2.65, -0.15, "future:\nweight 0", color="#B91C1C", fontsize=10, va="top")
fig.tight_layout(); fig.savefig(OUT / "lec11-kp05-causal-mask.png", dpi=200); plt.close(fig)

# ---------- figure 2: temperature and top-p ----------
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), sharey=True)
cols = {"0.5": "#7C3AED", "1.0": "#1D4ED8", "2.0": "#B45309"}
for ax, T in zip(axes, ("0.5", "1.0", "2.0")):
    p = np.array(dist[T])
    bars = ax.bar(range(5), p, color=cols[T], alpha=0.9)
    for i, v in enumerate(p):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_xticks(range(5)); ax.set_xticklabels(words)
    ax.set_title(f"temperature T = {T}", fontsize=12); ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.3)
    if T == "1.0":
        for i in range(5):
            if words[i] not in nucleus:
                bars[i].set_hatch("//"); bars[i].set_alpha(0.25)
        ax.text(2.0, 0.84, "top-p = 0.9 keeps mat, floor, sofa\n(hatched bars are dropped)", ha="center", fontsize=9.5, color="#1E3A8A")
axes[0].set_ylabel(r"probability of the next word")
fig.suptitle('next word after "the cat sat on the ..."', fontsize=12, y=0.99)
fig.tight_layout(); fig.savefig(OUT / "lec11-kp05-temperature.png", dpi=200); plt.close(fig)
