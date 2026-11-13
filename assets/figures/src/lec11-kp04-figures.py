"""lec11-kp04-figures.py — 第11週 KP04 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec11-kp04-figures.py [輸出資料夾]（預設：上一層的 assets/figures/）"""
import json, sys, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT.mkdir(parents=True, exist_ok=True)
num = {}

# ---------- residual: source scalar example ----------
x = 2.0; F = 0.1 * x ** 2; dF = 0.2 * x
num["residual_scalar"] = dict(x=x, F=F, y=x + F, dF=dF, grad_with_residual=1 + dF, grad_without=dF)
assert abs(1 + dF - 1.4) < 1e-12

# ---------- residual: gradient through a deep stack [Added] ----------
# each sublayer has a random Jacobian J_l (64x64) with spectral size about 0.3;
# plain stack: gradient = J_L ... J_1 g ; residual stack: (I + J_L) ... (I + J_1) g
rng = np.random.default_rng(0)
dim, L = 64, 24
Js = [rng.normal(0, 0.3 / np.sqrt(dim), (dim, dim)) for _ in range(L)]
g = rng.normal(0, 1, dim); g /= np.linalg.norm(g)
plain, resid = [1.0], [1.0]
gp, gr = g.copy(), g.copy()
for J in Js:
    gp = J.T @ gp; gr = gr + J.T @ gr
    plain.append(float(np.linalg.norm(gp))); resid.append(float(np.linalg.norm(gr)))
num["deep_stack"] = dict(dim=dim, jacobian_scale=0.3, depth=list(range(L + 1)),
                         plain_at_12=round(plain[12], 10), residual_at_12=round(resid[12], 4),
                         plain_at_24=float(f"{plain[24]:.3e}"), residual_at_24=round(resid[24], 4))

# ---------- LayerNorm: source example ----------
def ln(v, eps=0.0):
    mu = v.mean(); var = ((v - mu) ** 2).mean()
    return (v - mu) / np.sqrt(var + eps), mu, var
xh, mu, var = ln(np.array([1.0, 3.0, 5.0]))
num["ln_source"] = dict(x=[1, 3, 5], mu=mu, var=round(var, 4), sigma=round(math.sqrt(var), 4), xhat=xh.round(4).tolist())
assert abs(xh[2] - 1.2247) < 1e-4
# source's sigma without 1/H
num["ln_source_sigma_without_1_over_H"] = round(math.sqrt(8), 4)

# ---------- LN vs BN on a small batch [Added] ----------
X = np.array([[2.0, 4.0, 6.0, 8.0],
              [1.0, 1.0, 3.0, 3.0],
              [10.0, 0.0, 10.0, 0.0]])        # 3 tokens (rows) x 4 features (columns)
LN = np.stack([ln(r)[0] for r in X])
BN = (X - X.mean(0)) / X.std(0)
num["lnbn"] = dict(X=X.tolist(), LN=LN.round(4).tolist(), BN=BN.round(4).tolist(),
                   LN_row_means=LN.mean(1).round(4).tolist(), LN_row_std=LN.std(1).round(4).tolist(),
                   BN_col_means=BN.mean(0).round(4).tolist())
# BN with batch of one token: (x - x)/sqrt(0 + eps) = 0
one = X[:1]
num["bn_batch_of_one"] = ((one - one.mean(0)) / np.sqrt(one.var(0) + 1e-5)).round(4).tolist()
num["ln_batch_of_one"] = ln(X[0])[0].round(4).tolist()

# ---------- FFN ----------
d, dff = 512, 2048
attn = 4 * (d * d + d); ffn = (d * dff + dff) + (dff * d + d); lnp = 2 * d
layer = attn + ffn + 2 * lnp
num["ffn"] = dict(d_model=d, d_ff=dff, attention=attn, ffn=ffn, layernorms=2 * lnp, encoder_layer=layer,
                  ffn_share=round(ffn / layer, 4), attention_share=round(attn / layer, 4))
assert layer == 3152384
# FFN on 6 layers + embeddings 37000*512 (shared)
emb = 37000 * 512
num["ffn"]["embedding"] = emb
dec_layer = 2 * attn + ffn + 3 * lnp
total = 6 * layer + 6 * dec_layer + emb          # same count as KP01
num["ffn"].update(decoder_layer=dec_layer, ffn_share_of_decoder_layer=round(ffn / dec_layer, 4), model_total=total, ffn_all_layers=12 * ffn, ffn_share_of_model=round(12 * ffn / total, 4))
assert total == 63082496
# GELU source example
Phi = 0.5 * (1 + math.erf(2.0 / math.sqrt(2)))
num["gelu"] = dict(Phi2=round(Phi, 5), gelu2=round(2 * Phi, 4), y=round(2 * Phi * 3, 4),
                   relu2_y=6.0, gelu_min=None)
xs = np.linspace(-3, 3, 60001)
gel = xs * 0.5 * (1 + np.vectorize(math.erf)(xs / math.sqrt(2)))
num["gelu"]["gelu_min"] = [round(float(xs[gel.argmin()]), 3), round(float(gel.min()), 4)]
# position-wise: applying FFN row by row equals applying it to the matrix
W1 = rng.normal(0, 0.5, (4, 8)); W2 = rng.normal(0, 0.5, (8, 4))
f = lambda A: np.maximum(0, A @ W1) @ W2
num["positionwise_same"] = bool(np.allclose(f(X), np.stack([f(r[None])[0] for r in X])))
# stacking shapes
num["stack"] = dict(N=6, sublayers=12, add_norm_per_encoder_layer=2)
print(json.dumps(num, indent=1))

# ---------- figure 1: gradient norm through a deep stack ----------
fig, ax = plt.subplots(figsize=(8.6, 4.6))
dep = np.arange(L + 1)
ax.semilogy(dep, resid, "-o", color="#1D4ED8", ms=4, lw=2, label=r"with residual: $\prod (I + J_l)$")
ax.semilogy(dep, plain, "-s", color="#B91C1C", ms=4, lw=2, label=r"without residual: $\prod J_l$")
ax.axvline(12, color="#475569", ls="--", lw=1)
ax.text(11.7, 1e-10, "12 sublayers\n(6 encoder layers)", fontsize=10, color="#334155", ha="right", va="center")
ax.annotate(f"{resid[12]:.2f}", (12, resid[12]), xytext=(13.5, resid[12] * 6), fontsize=10, color="#1D4ED8",
            arrowprops=dict(arrowstyle="-", color="#1D4ED8"))
ax.annotate(f"{plain[12]:.1e}", (12, plain[12]), xytext=(14, plain[12] * 30), fontsize=10, color="#B91C1C",
            arrowprops=dict(arrowstyle="-", color="#B91C1C"))
ax.set_xlabel("number of sublayers the gradient passes back through")
ax.set_ylabel(r"gradient size $\|\partial \mathcal{L}/\partial x\|$ (start = 1)")
ax.set_ylim(1e-15, 1e2); ax.grid(alpha=0.3, which="major"); ax.legend(loc="lower left", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "lec11-kp04-residual-gradient.png", dpi=200); plt.close(fig)

# ---------- figure 2: LN vs BN ----------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
titles = ["input X  (3 tokens x 4 features)", "LayerNorm: each ROW -> mean 0, std 1", "BatchNorm: each COLUMN -> mean 0, std 1"]
mats = [X, LN, BN]
for ax, M, t, k in zip(axes, mats, titles, range(3)):
    vmax = 10 if k == 0 else 1.8
    ax.imshow(M, cmap="RdBu_r" if k else "Greys", vmin=-vmax if k else 0, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            dark = (k == 0 and v > 6) or (k > 0 and abs(v) > 1.1)
            ax.text(j, i, f"{v:g}" if k == 0 else f"{v:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if dark else "#0F172A")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"f{j+1}" for j in range(4)])
    ax.set_yticks(range(3)); ax.set_yticklabels([f"token {i+1}" for i in range(3)])
    ax.set_title(t, fontsize=11)
for i in range(3):
    axes[1].add_patch(plt.Rectangle((-0.48, i - 0.46), 3.96, 0.92, fill=False, ec="#7C3AED", lw=2))
for j in range(4):
    axes[2].add_patch(plt.Rectangle((j - 0.46, -0.48), 0.92, 2.96, fill=False, ec="#1D4ED8", lw=2))
fig.tight_layout(); fig.savefig(OUT / "lec11-kp04-ln-vs-bn.png", dpi=200); plt.close(fig)
