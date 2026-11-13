"""lec10-kp04-figures.py — 第10週 KP04 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec10-kp04-figures.py [輸出資料夾]（預設：上一層的 assets/figures/）"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT.mkdir(parents=True, exist_ok=True)


def softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
V = np.array([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])
q = np.array([2.0, 1.0])
num = {"K": K.tolist(), "V": V.tolist(), "q": q.tolist()}
dot = K @ q
cos = dot / (np.linalg.norm(K, axis=1) * np.linalg.norm(q))
scaled = dot / np.sqrt(2)
for name, s in (("dot", dot), ("scaled", scaled), ("cosine", cos)):
    w = softmax(s)
    num[name] = dict(scores=s.round(4).tolist(), weights=w.round(4).tolist(), output=(w @ V).round(4).tolist())
num["hard"] = dict(index=int(np.argmax(dot)), output=V[np.argmax(dot)].tolist())
num["exp"] = [round(float(np.exp(x)), 3) for x in (2, 1, 3)]
num["denom"] = round(float(np.exp(2) + np.exp(1) + np.exp(3)), 3)
# additive similarity of the source: w_q^T q + w_k^T k_i  (q-term cancels in softmax)
wk = np.array([1.0, 1.0])
add_scores = {}
for qq, wq in (([2.0, 1.0], [1.0, 0.0]), ([0.0, 3.0], [1.0, 0.0])):
    s = np.array(wq) @ np.array(qq) + K @ wk
    add_scores[str(qq)] = dict(scores=s.round(4).tolist(), weights=softmax(s).round(4).tolist())
num["additive_source_form"] = add_scores
# practice
num["practice1"] = (np.array([0.7, 0.2, 0.1]) @ np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])).round(4).tolist()
q3 = np.array([0.0, 2.0])
w3 = softmax(K @ q3)
num["practice3"] = dict(scores=(K @ q3).tolist(), weights=w3.round(4).tolist(), output=(w3 @ V).round(4).tolist())
assert np.allclose(num["dot"]["weights"], [0.2447, 0.09, 0.6652], atol=1e-4)
print(json.dumps(num, indent=1))

# ---------- figure 1: hard lookup vs soft lookup ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
labels = [r"$\mathbf{k}_1$", r"$\mathbf{k}_2$", r"$\mathbf{k}_3$"]
hard = np.zeros(3); hard[num["hard"]["index"]] = 1
for ax, w, title, col, out in ((axes[0], hard, "hard lookup: take only the best match", "#64748B", V[num["hard"]["index"]]),
                               (axes[1], softmax(dot), "soft lookup (attention): blend all values", "#2563EB", softmax(dot) @ V)):
    bars = ax.bar(range(3), w, color=[col] * 3, width=0.55)
    for b, v in zip(bars, w):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.4f}", ha="center", fontsize=11)
    ax.set_xticks(range(3), labels, fontsize=13); ax.set_ylim(0, 1.18); ax.grid(axis="y", alpha=0.3)
    ax.set_title(title, fontsize=12)
    ax.text(1, 1.06, rf"output $=[{out[0]:.4f},\ {out[1]:.4f}]^\top$", ha="center", fontsize=11, color=col)
axes[0].set_ylabel("weight on each value")
fig.suptitle(r"Query $\mathbf{q}=[2,1]^\top$; scores $\mathbf{q}^\top\mathbf{k}_i = 2,\ 1,\ 3$", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "lec10-kp04-hard-vs-soft.png", dpi=200); plt.close(fig)

# ---------- figure 2: three similarity measures ----------
fig, ax = plt.subplots(figsize=(8.5, 3.9))
x = np.arange(3); wbar = 0.26
for k, (name, col) in enumerate((("dot", "#DC2626"), ("scaled", "#2563EB"), ("cosine", "#059669"))):
    w = num[name]["weights"]
    bars = ax.bar(x + (k - 1) * wbar, w, wbar, color=col, label={"dot": r"dot $\mathbf{q}^\top\mathbf{k}$",
                  "scaled": r"scaled $\mathbf{q}^\top\mathbf{k}/\sqrt{2}$", "cosine": "cosine"}[name])
    for b, v in zip(bars, w):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
ax.set_xticks(x, labels, fontsize=13); ax.set_ylim(0, 0.8); ax.set_ylabel("softmax weight"); ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper left", fontsize=10)
ax.set_title(r"Same query and keys, three similarity measures", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "lec10-kp04-similarity-measures.png", dpi=200); plt.close(fig)
