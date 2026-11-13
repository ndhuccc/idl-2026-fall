"""lec10-kp05-figures.py — 第10週 KP05 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec10-kp05-figures.py [輸出資料夾]（預設：上一層的 assets/figures/）"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT.mkdir(parents=True, exist_ok=True)


def softmax_rows(S):
    E = np.exp(S - S.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


num = {}
# --- three-word example [Added]: X rows = KP01 vectors, W_Q = W_K = W_V = I
X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
Q = K = V = X
S = Q @ K.T / np.sqrt(2)
A = softmax_rows(S)
Z = A @ V
num["three_words"] = dict(X=X.tolist(), QKt=(Q @ K.T).tolist(), S=S.round(4).tolist(), A=A.round(4).tolist(),
                          row_sums=A.sum(1).round(6).tolist(), Z=Z.round(4).tolist())
# --- source example (professor's notes): Q = I, K = [[1,1],[0,2]], V = diag(10,20)
Q2 = np.eye(2); K2 = np.array([[1.0, 1.0], [0.0, 2.0]]); V2 = np.diag([10.0, 20.0])
S2 = Q2 @ K2.T / np.sqrt(2)
A2 = softmax_rows(S2)
num["source_example"] = dict(QKt=(Q2 @ K2.T).tolist(), S=S2.round(4).tolist(), A=A2.round(4).tolist(), Z=(A2 @ V2).round(4).tolist())
# --- why sqrt(d_k): source numbers d_k = 64, raw scores [16, 0]
raw = np.array([[16.0, 0.0]]); a_raw = softmax_rows(raw)[0]; a_sc = softmax_rows(raw / 8)[0]
num["scaling_example"] = dict(raw=a_raw.tolist(), raw_grad=float(a_raw[0] * (1 - a_raw[0])),
                              scaled=a_sc.round(4).tolist(), scaled_grad=round(float(a_sc[0] * (1 - a_sc[0])), 4))
# --- simulation [Added]: random q, k ~ N(0, I), 8 keys, 2000 trials per d_k
rng = np.random.default_rng(0)
dks = [4, 16, 64, 256, 512]
sim = {}
for d in dks:
    q = rng.standard_normal((2000, 1, d)); k = rng.standard_normal((2000, 8, d))
    s = (q * k).sum(-1)
    sim[d] = dict(std_dot=round(float(s.std()), 3), sqrt_d=round(d ** 0.5, 3),
                  max_w_raw=round(float(softmax_rows(s).max(1).mean()), 4),
                  max_w_scaled=round(float(softmax_rows(s / np.sqrt(d)).max(1).mean()), 4),
                  max_w_div_d=round(float(softmax_rows(s / d).max(1).mean()), 4))
num["simulation"] = {str(d): v for d, v in sim.items()}
num["simulation_setup"] = "seed 0; 2000 trials; one query and 8 keys per trial; entries i.i.d. N(0,1); mean of the largest softmax weight"
# practice [Added]
P = np.array([[2.0, 0.0], [0.0, 2.0]])
num["practice2"] = dict(QKt=(P @ P.T).tolist(), A=softmax_rows(P @ P.T / np.sqrt(2)).round(4).tolist())
assert np.allclose(num["source_example"]["A"], [[0.6698, 0.3302], [0.3302, 0.6698]], atol=1e-4)
print(json.dumps(num, indent=1))

# ---------- figure 1: the three-word attention matrix ----------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axes[0]
im = ax.imshow(A, cmap="Purples", vmin=0, vmax=1)
names = [r"word 1: $[1,0]$", r"word 2: $[0,1]$", r"word 3: $[1,1]$"]
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{A[i, j]:.4f}", ha="center", va="center", fontsize=12, color="white" if A[i, j] > 0.45 else "#1E1B4B")
ax.set_xticks(range(3), ["key 1", "key 2", "key 3"]); ax.set_yticks(range(3), names)
ax.set_title(r"$\mathbf{A}=\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{2})$, rows sum to 1", fontsize=12)
ax = axes[1]
ax.plot(*X.T, "o", color="#1E293B", ms=8)
for (x, y), (zx, zy), lab in zip(X, Z, ("1", "2", "3")):
    ax.annotate("", xy=(zx, zy), xytext=(x, y), arrowprops=dict(arrowstyle="->", color="#7C3AED", lw=1.6))
    ax.text(x + 0.04, y - 0.1, rf"$\mathbf{{x}}_{lab}$", fontsize=13)
    ax.plot(zx, zy, "*", color="#7C3AED", ms=14)
    ax.text(zx + 0.04, zy + 0.04, rf"$\mathbf{{z}}_{lab}$", fontsize=13, color="#7C3AED")
ax.set_xlim(-0.25, 1.35); ax.set_ylim(-0.25, 1.35); ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_title(r"each word moves toward the words it attends to", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "lec10-kp05-three-word-attention.png", dpi=200); plt.close(fig)

# ---------- figure 2: why divide by sqrt(d_k) ----------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
ax = axes[0]
ax.plot(dks, [sim[d]["std_dot"] for d in dks], "o-", color="#DC2626", label=r"std of $\mathbf{q}^\top\mathbf{k}$ (simulated)")
ax.plot(dks, [sim[d]["sqrt_d"] for d in dks], "--", color="#64748B", label=r"$\sqrt{d_k}$")
ax.set_xscale("log", base=2); ax.set_xticks(dks, [str(d) for d in dks]); ax.set_xlabel(r"dimension $d_k$")
ax.set_title(r"inner products grow like $\sqrt{d_k}$", fontsize=12); ax.grid(alpha=0.3); ax.legend(fontsize=10)
ax = axes[1]
ax.plot(dks, [sim[d]["max_w_raw"] for d in dks], "o-", color="#DC2626", label="without scaling")
ax.plot(dks, [sim[d]["max_w_scaled"] for d in dks], "s-", color="#2563EB", label=r"divided by $\sqrt{d_k}$")
ax.axhline(1 / 8, color="#94A3B8", ls=":", label="uniform (1/8)")
ax.set_xscale("log", base=2); ax.set_xticks(dks, [str(d) for d in dks]); ax.set_xlabel(r"dimension $d_k$")
ax.set_ylim(0, 1.05); ax.set_ylabel("mean largest weight (8 keys)")
ax.set_title("without scaling, softmax becomes almost one-hot", fontsize=12); ax.grid(alpha=0.3); ax.legend(fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "lec10-kp05-scaling.png", dpi=200); plt.close(fig)
