"""lec11-kp07-figures.py — 第11週 KP07 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec11-kp07-figures.py [輸出資料夾]（預設：上一層的 assets/figures/）"""
import json, sys, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams["mathtext.fontset"] = "cm"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT.mkdir(parents=True, exist_ok=True)
num = {}

# ---------- sparse patterns on n = 16 ----------
n, w = 16, 2                       # window: |i - j| <= w
I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
full = np.ones((n, n), int)
window = (np.abs(I - J) <= w).astype(int)
glob = window.copy(); glob[0, :] = 1; glob[:, 0] = 1     # token 0 is a global token
num["sparse_n16"] = dict(n=n, w=w, full=int(full.sum()), window=int(window.sum()), window_global=int(glob.sum()),
                         window_share=round(window.sum() / full.sum(), 4), window_global_share=round(glob.sum() / full.sum(), 4))


def hops(A):
    """longest shortest path (in layers) between any two tokens when each layer follows edges of A"""
    m = A.shape[0]; D = np.full((m, m), np.inf); np.fill_diagonal(D, 0)
    D[A > 0] = np.minimum(D[A > 0], 1)
    for k in range(m):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return int(D.max())


num["sparse_n16"].update(layers_full=hops(full), layers_window=hops(window), layers_window_global=hops(glob))
# source notes BigBird example, n=1000
num["bigbird_source"] = dict(per_token=2 * 16 + 2 * 4 + 4, total=1000 * 44, sparsity=round(1 - 44000 / 1e6, 4))

# ---------- associativity ----------
def cost(n, d): return dict(standard=4 * n * n * d, reordered=4 * n * d * d, ratio=n / d)
num["assoc_4096_256"] = cost(4096, 256)
num["assoc_10000_64"] = cost(10000, 64)
rng = np.random.default_rng(0)
Q = rng.normal(0, 1, (6, 3)); K = rng.normal(0, 1, (6, 3)); V = rng.normal(0, 1, (6, 2))
num["assoc_equal_without_softmax"] = bool(np.allclose((Q @ K.T) @ V, Q @ (K.T @ V)))
num["intermediate_shapes"] = dict(QKt=[6, 6], KtV=[3, 2])

# ---------- positive random features (FAVOR+ style, i.i.d. Gaussian w, antithetic pairs) ----------
def phi_pos(X, W):
    m = W.shape[0]
    pre = np.exp(-np.sum(X ** 2, axis=1, keepdims=True) / 2) / np.sqrt(2 * m)
    P = X @ W.T
    return pre * np.concatenate([np.exp(P), np.exp(-P)], axis=1)


def phi_trig(X, W):
    m = W.shape[0]
    pre = np.exp(np.sum(X ** 2, axis=1, keepdims=True) / 2) / np.sqrt(m)
    P = X @ W.T
    return pre * np.concatenate([np.sin(P), np.cos(P)], axis=1)


d = 4
q = rng.normal(0, 1, d) * 0.6
Kx = rng.normal(0, 1, (8, d)) * 0.6
exact_scores = np.exp(Kx @ q)
exact = exact_scores / exact_scores.sum()
num["query_keys_scale"] = 0.6
num["exact_weights"] = exact.round(4).tolist()


def approx_weights(m, kind, seed):
    W = np.random.default_rng(seed).normal(0, 1, (m, d))
    f = phi_pos if kind == "pos" else phi_trig
    s = f(Kx, W) @ f(q[None], W)[0]
    return s / s.sum(), s


approx, spread = {}, {}
for m in (16, 256, 4096):
    runs = np.array([approx_weights(m, "pos", 500 + sd)[0] for sd in range(50)])
    approx[str(m)] = runs.mean(0).round(4).tolist(); spread[str(m)] = runs.std(0).round(4).tolist()
num["favor_weights_mean_of_50"] = approx; num["favor_weights_std_of_50"] = spread
# error vs m: mean over 50 seeds of L1 distance between approximate and exact weights; also how often trig gives a negative score
ms = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
err_pos, err_trig, neg_trig, neg_pos = [], [], [], []
for m in ms:
    ep, et, nt, npos = [], [], 0, 0
    for seed in range(50):
        a, s = approx_weights(m, "pos", 1000 + seed); ep.append(np.abs(a - exact).sum()); npos += int((s < 0).any())
        b, s2 = approx_weights(m, "trig", 1000 + seed); et.append(np.abs(b - exact).sum()); nt += int((s2 < 0).any())
    err_pos.append(float(np.mean(ep))); err_trig.append(float(np.mean(et))); neg_trig.append(nt / 50); neg_pos.append(npos / 50)
num["error_vs_m"] = dict(m=ms, positive=[round(e, 4) for e in err_pos], trig=[round(e, 4) for e in err_trig],
                         trig_runs_with_negative_score=neg_trig, positive_runs_with_negative_score=neg_pos)
# source notes FAVOR+ scalar example with and without the 1/sqrt(2m) factor
x = 1.0; wv = 0.8
raw = (math.exp(-0.5) * math.exp(wv)) ** 2 + (math.exp(-0.5) * math.exp(-wv)) ** 2
num["favor_source_scalar"] = dict(target=round(math.e, 4), without_norm=round(raw, 4), with_norm=round(raw / 2, 4))
print(json.dumps(num, indent=1))

# ---------- figure 1: sparse patterns ----------
fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.5))
cm = ListedColormap(["#F1F5F9", "#1D4ED8"])
titles = [f"full: {full.sum()} pairs, 1 layer",
          f"sliding window $w={w}$: {window.sum()} pairs, {num['sparse_n16']['layers_window']} layers",
          f"window + 1 global token: {glob.sum()} pairs, {num['sparse_n16']['layers_window_global']} layers"]
for ax, M, t in zip(axes, (full, window, glob), titles):
    ax.imshow(M, cmap=cm, vmin=0, vmax=1)
    ax.set_xticks([0, 5, 10, 15]); ax.set_yticks([0, 5, 10, 15])
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True); ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6); ax.tick_params(which="minor", length=0)
    ax.set_xlabel("key position"); ax.set_title(t, fontsize=10.5)
axes[0].set_ylabel("query position")
axes[2].add_patch(plt.Rectangle((-0.5, -0.5), 1, n, fill=False, ec="#B45309", lw=2.2))
axes[2].add_patch(plt.Rectangle((-0.5, -0.5), n, 1, fill=False, ec="#B45309", lw=2.2))
axes[2].text(8, 3.2, "token 0 sees all\nand is seen by all", color="#B45309", fontsize=9.5, ha="left")
fig.tight_layout(); fig.savefig(OUT / "lec11-kp07-sparse-patterns.png", dpi=200); plt.close(fig)

# ---------- figure 2: random features ----------
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
ax = axes[0]
xk = np.arange(8); wdt = 0.2
ax.bar(xk - 1.5 * wdt, exact, wdt, color="#0F172A", label="exact softmax")
for i, (m, col) in enumerate(((16, "#FCA5A5"), (256, "#F87171"), (4096, "#B91C1C"))):
    ax.bar(xk + (i - 0.5) * wdt, approx[str(m)], wdt, color=col, yerr=spread[str(m)], capsize=2,
           error_kw=dict(lw=0.8, ecolor="#334155"), label=f"positive features, $m={m}$")
ax.set_xticks(xk); ax.set_xticklabels([f"$k_{i+1}$" for i in range(8)])
ax.set_ylabel("attention weight of one query"); ax.set_title("mean of 50 draws; bars show the spread", fontsize=12)
ax.set_ylim(0, 0.5); ax.legend(fontsize=9.5, ncol=2, loc="upper left"); ax.grid(axis="y", alpha=0.3)
ax = axes[1]
ax.loglog(ms, err_trig, "-s", color="#7C3AED", lw=2, ms=5, label="trig features (can be negative)")
ax.loglog(ms, err_pos, "-o", color="#B91C1C", lw=2, ms=5, label="positive features (no orthogonalization)")
ref = err_pos[-1] * np.sqrt(ms[-1] / np.array(ms))
ax.loglog(ms, ref, ":", color="#475569", lw=1.4, label=r"reference slope $1/\sqrt{m}$")
ax.set_xlabel("number of random features $m$"); ax.set_ylabel("L1 error, summed over 8 weights\n(mean of 50 draws)")
ax.set_title("both errors shrink as $m$ grows", fontsize=12); ax.grid(alpha=0.3, which="major"); ax.legend(fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "lec11-kp07-random-features.png", dpi=200); plt.close(fig)
