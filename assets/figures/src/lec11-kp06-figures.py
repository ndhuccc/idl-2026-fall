"""lec11-kp06-figures.py — 第11週 KP06 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec11-kp06-figures.py [輸出資料夾]（預設：上一層的 assets/figures/）"""
import json, sys, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams["mathtext.fontset"] = "cm"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT.mkdir(parents=True, exist_ok=True)
num = {}

# ---------- path length and sequential steps ----------
def rnn_path(n): return n - 1
def cnn_path(n, k=3): return math.ceil((n - 1) / (k - 1))          # contiguous kernels, one layer widens reach by k-1
num["path_n8"] = dict(rnn=rnn_path(8), cnn_k3=cnn_path(8), self_attention=1)
num["path_n100"] = dict(rnn=rnn_path(100), cnn_k3=cnn_path(100), self_attention=1)
num["sequential_n100"] = dict(rnn=100, self_attention=1)

# ---------- FLOPs and memory (source notes formula: 4 n^2 d + 6 n d^2; Q,K,V projections only) ----------
def flops(n, d): return 4 * n * n * d, 6 * n * d * d
d = 512
rows = {}
for n in (128, 768, 1024, 4096, 16384, 32768):
    a, b = flops(n, d)
    rows[n] = dict(attn_term=a, proj_term=b, total=a + b, attn_share=round(a / (a + b), 4),
                   matrix_entries=n * n, matrix_bytes_fp32=4 * n * n)
num["flops_d512"] = {str(k): v for k, v in rows.items()}
num["crossover_n"] = 1.5 * d
assert abs(rows[1024]["total"] / 1e9 - 3.758) < 1e-3
num["ratio_1024_to_32768_attn_term"] = rows[32768]["attn_term"] / rows[1024]["attn_term"]
# source demo: 2n^2d + 4nd^2 versus 4nd^2 at n=16384, d=512
n = 16384
num["source_demo_ratio"] = round((2 * n * n * d + 4 * n * d * d) / (4 * n * d * d), 4)
# ---------- BERT vs GPT supervision on a 4-word sentence and on 512 tokens ----------
num["supervision"] = dict(bert_masked_fraction=0.15, bert_targets_512=round(0.15 * 512, 1), gpt_targets_512=512)
print(json.dumps(num, indent=1))

# ---------- figure 1: path length ----------
fig = plt.figure(figsize=(12.4, 4.4))
ax = fig.add_axes([0.02, 0.06, 0.52, 0.86]); ax.axis("off"); ax.set_xlim(-0.5, 6.0); ax.set_ylim(-0.8, 4.0)
xs = np.arange(6)
for y, title, col in ((3.0, "RNN: 5 steps from word 1 to word 6", "#B91C1C"), (0.6, "self-attention: 1 step between any two words", "#1D4ED8")):
    ax.text(2.75, y + 0.75 if y > 2 else y - 0.6, title, ha="center", fontsize=12, color=col)
    for x in xs:
        ax.add_patch(plt.Circle((x, y), 0.22, fc="white", ec=col, lw=2))
        ax.text(x, y, f"{x+1}", ha="center", va="center", fontsize=11)
for x in xs[:-1]:
    ax.add_patch(FancyArrowPatch((x + 0.24, 3.0), (x + 0.76, 3.0), arrowstyle="-|>", mutation_scale=12, color="#B91C1C", lw=1.8))
for i in xs:
    for j in xs:
        if j > i:
            hi = 0.25 + 0.17 * (j - i)
            lw, colr, al = (2.6, "#1D4ED8", 1.0) if (i, j) == (0, 5) else (1.0, "#93C5FD", 0.9)
            ax.add_patch(FancyArrowPatch((i, 0.82), (j, 0.82), connectionstyle=f"arc3,rad=-{0.12 + 0.02*(j-i)}", arrowstyle="-", color=colr, lw=lw, alpha=al))
ax2 = fig.add_axes([0.62, 0.15, 0.36, 0.72])
ns = np.arange(2, 101)
ax2.plot(ns, [rnn_path(n) for n in ns], color="#B91C1C", lw=2.2, label="RNN:  $n-1$")
ax2.plot(ns, [cnn_path(n) for n in ns], color="#15803D", lw=2.2, label="CNN, kernel 3:  $\\lceil (n-1)/2 \\rceil$")
ax2.plot(ns, np.ones_like(ns), color="#1D4ED8", lw=2.6, label="self-attention:  $1$")
ax2.set_xlabel("sentence length $n$"); ax2.set_ylabel("longest path (steps)")
ax2.set_title("longest path between two words", fontsize=12)
ax2.grid(alpha=0.3); ax2.legend(fontsize=10, loc="upper left")
fig.savefig(OUT / "lec11-kp06-path-length.png", dpi=200); plt.close(fig)

# ---------- figure 2: cost vs n ----------
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3))
ns = np.logspace(np.log10(64), np.log10(65536), 200)
a = 4 * ns ** 2 * d; b = 6 * ns * d * d
ax = axes[0]
ax.loglog(ns, a, color="#B91C1C", lw=2.2, label=r"scores and weighting: $4n^2d$")
ax.loglog(ns, b, color="#15803D", lw=2.2, label=r"projections: $6nd^2$")
ax.loglog(ns, a + b, color="#0F172A", lw=1.4, ls="--", label="total")
ax.axvline(768, color="#475569", ls=":", lw=1.2)
ax.text(800, 2e8, r"$n=1.5d=768$", fontsize=10, color="#334155")
ax.set_xlabel("sentence length $n$  ($d=512$)"); ax.set_ylabel("FLOPs per self-attention layer")
ax.set_title("compute: the $n^2$ term takes over", fontsize=12); ax.grid(alpha=0.3, which="major"); ax.legend(fontsize=9.5, loc="upper left")
ax = axes[1]
mem = 4 * ns ** 2
ax.loglog(ns, mem, color="#7C3AED", lw=2.4)
for n_, lab in ((1024, "4.19 MB"), (32768, "4.29 GB")):
    ax.plot(n_, 4 * n_ ** 2, "o", color="#7C3AED")
    ax.annotate(f"n = {n_}: {lab}", (n_, 4 * n_ ** 2), xytext=(n_ * 0.06, 4 * n_ ** 2 * 3), fontsize=10,
                arrowprops=dict(arrowstyle="-", color="#7C3AED"))
ax.set_xlabel("sentence length $n$"); ax.set_ylabel("bytes (float32)")
ax.set_title(r"memory: one $n\times n$ weight table", fontsize=12); ax.grid(alpha=0.3, which="major")
fig.tight_layout(); fig.savefig(OUT / "lec11-kp06-cost-vs-n.png", dpi=200); plt.close(fig)
