"""lec08-kp02-figures.py — 第8週 KP02 講義的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec08-kp02-figures.py [輸出資料夾]（預設：與講義同層的 ../assets/figures/）"""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["mathtext.fontset"] = "cm"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent  # 課程 repo：assets/figures/src/ 的上一層 assets/figures/
OUT.mkdir(parents=True, exist_ok=True)
r4 = lambda v: round(float(v), 4)

# ---------- numbers ----------
# (1) total loss and shared-weight gradient (source professor's notes, 3 steps)
Ls = [0.50, 0.30, 0.10]
gW = [0.05, 0.15, 0.20]
loss = dict(L=Ls, cum=[r4(sum(Ls[:k + 1])) for k in range(3)], total=r4(sum(Ls)),
            gW=gW, gW_total=r4(sum(gW)), terminal_only_total=Ls[2])

# (2) running example for delta_t (graduate KP02 / source notes): V=1.0, W=0.5, tanh' = 0.8, dL_t/dy_t = 1.0
V, W, dtanh, T = 1.0, 0.5, 0.8, 3
def deltas(g, V, W, dtanh):
    """g[t] = dL_t/dy_t for t = 1..T (list index 0..T-1); returns delta_1..delta_T and the two parts."""
    d = [0.0] * len(g); out_part = [0.0] * len(g); fut_part = [0.0] * len(g)
    for k in reversed(range(len(g))):
        out_part[k] = g[k] * V
        fut_part[k] = d[k + 1] * dtanh * W if k + 1 < len(g) else 0.0
        d[k] = out_part[k] + fut_part[k]
    return [r4(v) for v in d], [r4(v) for v in out_part], [r4(v) for v in fut_part]
allout = deltas([1.0, 1.0, 1.0], V, W, dtanh)
lastonly = deltas([0.0, 0.0, 1.0], V, W, dtanh)
ex = dict(V=V, W=W, dtanh=dtanh, factor=r4(W * dtanh), h_for_dtanh=r4(math.sqrt(1 - dtanh)),
          all_steps=dict(delta=allout[0], output_part=allout[1], future_part=allout[2]),
          last_only=dict(delta=lastonly[0], output_part=lastonly[1], future_part=lastonly[2]))

# (3) MSE output gradient: L_t = 0.5 (y_t - r_t)^2  ->  dL_t/dy_t = y_t - r_t
mse = dict(y=1.5, r=0.5, L=r4(0.5 * (1.5 - 0.5) ** 2), dLdy=r4(1.5 - 0.5))

# (4) practice numbers
p1 = dict(L=[0.40, 0.25, 0.15], total=r4(0.40 + 0.25 + 0.15), g=[0.10, 0.20, 0.30], g_total=r4(0.10 + 0.20 + 0.30))
d2 = deltas([1.0, 1.0, 1.0], 2.0, 0.5, 0.6)
p2 = dict(V=2.0, W=0.5, dtanh=0.6, factor=r4(0.5 * 0.6), delta=d2[0], output_part=d2[1], future_part=d2[2])
d3 = deltas([0.0, 0.0, 1.0], 1.0, 0.5, 1 - 0.0 ** 2)
p3 = dict(V=1.0, W=0.5, h=0.0, dtanh=1.0, delta=d3[0])
tanh_d = dict(h=0.6, dtanh=r4(1 - 0.6 ** 2))

# (5) vector check by finite differences (p=2, q=1, T=3): column-vector recursion vs. the source's ordering
rng = np.random.default_rng(seed=0)
p, d, q = 2, 1, 1
Wm, Um, Vm = rng.normal(0, 0.8, (p, p)), rng.normal(0, 1, (p, d)), rng.normal(0, 1, (q, p))
xs, rs = rng.normal(0, 1, (T, d)), rng.normal(0, 1, (T, q))
def forward(h0):
    hs, L, h = [], 0.0, h0
    for t in range(T):
        h = np.tanh(Wm @ h + Um @ xs[t]); y = Vm @ h
        L += 0.5 * float(np.sum((y - rs[t]) ** 2)); hs.append(h)
    return hs, L
hs, _ = forward(np.zeros(p))
def loss_from_h1(h1):  # total loss as a function of h_1 (step 1 output + all later steps)
    L, h = 0.5 * float(np.sum((Vm @ h1 - rs[0]) ** 2)), h1
    for t in range(1, T):
        h = np.tanh(Wm @ h + Um @ xs[t]); L += 0.5 * float(np.sum((Vm @ h - rs[t]) ** 2))
    return L
eps, num = 1e-6, np.zeros(p)
for j in range(p):
    e = np.zeros(p); e[j] = eps
    num[j] = (loss_from_h1(hs[0] + e) - loss_from_h1(hs[0] - e)) / (2 * eps)
dcol = [None] * T; drow_src = [None] * T
for t in reversed(range(T)):
    g = Vm @ hs[t] - rs[t]
    dcol[t] = Vm.T @ g + (Wm.T @ ((1 - hs[t + 1] ** 2) * dcol[t + 1]) if t + 1 < T else 0)
    drow_src[t] = g @ Vm + (drow_src[t + 1] @ Wm @ np.diag(1 - hs[t + 1] ** 2) if t + 1 < T else 0)
check = dict(seed=0, numeric_delta1=[r4(v) for v in num], column_formula=[r4(v) for v in dcol[0]],
             source_ordering=[r4(v) for v in drow_src[0]],
             err_column=float(np.max(np.abs(num - dcol[0]))), err_source=r4(np.max(np.abs(num - drow_src[0]))))

nums = dict(loss=loss, example=ex, mse=mse, practice1=p1, practice2=p2, practice3=p3, tanh_deriv=tanh_d, vector_check=check)
print(json.dumps(nums, indent=1))

# ---------- figure 1: total loss and shared-weight gradient ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axs[0]; ts = [1, 2, 3]
ax.bar(ts, Ls, width=0.55, color="#3B82F6", label=r"step loss $\mathcal{L}_t$")
ax.plot(ts, loss["cum"], "o-", color="#DC2626", lw=2, label=r"running sum $\mathcal{L}_1+\dots+\mathcal{L}_t$")
for t, v in zip(ts, Ls): ax.text(t, v / 2, f"{v:.2f}", ha="center", va="center", color="white", fontsize=12)
for t, v in zip(ts, loss["cum"]): ax.text(t - 0.08, v + 0.05, f"{v:.2f}", ha="right", fontsize=12, color="#DC2626")
ax.set_xticks(ts); ax.set_xticklabels([r"$t=1$", r"$t=2$", r"$t=3$"], fontsize=12)
ax.set_ylim(0, 1.2); ax.set_xlim(0.4, 3.8); ax.grid(axis="y", alpha=0.3)
ax.set_title(r"(a) total loss $\mathcal{L}=\sum_t \mathcal{L}_t = 0.90$", fontsize=13)
ax.legend(loc="upper left", fontsize=11, frameon=False)
ax = axs[1]
bottom = 0.0
cols = ["#93C5FD", "#60A5FA", "#2563EB"]
for t, v, c in zip(ts, gW, cols):
    ax.bar(1, v, bottom=bottom, width=0.8, color=c, edgecolor="white")
    ax.text(1, bottom + v / 2, rf"step {t}: ${v:.2f}$", ha="center", va="center", fontsize=11,
            color="white" if t == 3 else "#0F172A")
    bottom += v
for t, v, c in zip(ts, gW, cols):
    ax.bar(1.9 + 0.55 * t, v, width=0.45, color=c)
    ax.text(1.9 + 0.55 * t, v + 0.01, f"{v:.2f}", ha="center", fontsize=11)
ax.text(1, bottom + 0.015, rf"sum $={loss['gW_total']:.2f}$", ha="center", fontsize=13, color="#16A34A")
ax.set_xticks([1, 2.45, 3.0, 3.55])
ax.set_xticklabels([r"$\partial\mathcal{L}/\partial W$", r"$\partial\mathcal{L}_1/\partial W$",
                    r"$\partial\mathcal{L}_2/\partial W$", r"$\partial\mathcal{L}_3/\partial W$"], fontsize=11)
ax.set_ylim(0, 0.5); ax.set_xlim(0.5, 4.0); ax.grid(axis="y", alpha=0.3)
ax.set_title(r"(b) shared $W$: the step gradients add up", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "lec08-kp02-loss-sum.png", dpi=200); plt.close(fig)

# ---------- figure 2: the two backward paths into h_t ----------
def box(ax, x, y, txt, fc, w=1.0, h=0.6, fs=15):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fc, ec="#334155", lw=1.4))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs)
def arrow(ax, a, b, color, lw=2.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=lw, color=color, linestyle=ls))
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.set_xlim(0.2, 11.2); ax.set_ylim(-0.9, 4.2); ax.axis("off")
X = [2.0, 5.6, 9.2]
d_all = ex["all_steps"]
for k, x in enumerate(X, start=1):
    box(ax, x, 0.8, rf"$h_{k}$", "#DBEAFE")
    box(ax, x, 2.9, rf"$y_{k}$", "#FEF3C7")
    ax.text(x, 3.55, rf"$\partial\mathcal{{L}}_{k}/\partial y_{k}=1.0$", ha="center", fontsize=12, color="#B45309")
    # backward from y_t to h_t (orange)
    arrow(ax, (x + 0.15, 2.58), (x + 0.15, 1.12), "#EA580C")
    ax.text(x + 0.28, 1.85, rf"$\times V$: $+{d_all['output_part'][k-1]:.2f}$", ha="left", va="center", fontsize=12, color="#EA580C")
    ax.text(x, 0.12, rf"$\delta_{k}={d_all['delta'][k-1]:.2f}$", ha="center", va="top", fontsize=14, color="#0F172A")
# backward from h_{t+1} to h_t (green)
for k in (1, 2):
    a, b = X[k] - 0.52, X[k - 1] + 0.52
    arrow(ax, (a, 0.8), (b, 0.8), "#16A34A")
    ax.text((a + b) / 2, 1.05, rf"$\times W(1-h_{k+1}^2)=\times{ex['factor']:.1f}$", ha="center", fontsize=12, color="#16A34A")
    ax.text((a + b) / 2, 0.5, rf"$+{d_all['future_part'][k-1]:.2f}$", ha="center", fontsize=12, color="#16A34A")
ax.text(10.0, 0.8, "no step 4:\nonly one path", ha="left", va="center", fontsize=11, color="#64748B")
ax.annotate("", xy=(1.0, -0.65), xytext=(10.2, -0.65), arrowprops=dict(arrowstyle="->", lw=1.6, color="#64748B"))
ax.text(5.6, -0.55, r"backward order: $t=3\rightarrow2\rightarrow1$", ha="center", va="bottom", fontsize=12, color="#64748B")
fig.tight_layout(); fig.savefig(OUT / "lec08-kp02-two-paths.png", dpi=200); plt.close(fig)

# ---------- figure 3: delta_t split into its two parts, all-step vs last-step-only ----------
fig, ax = plt.subplots(figsize=(10, 4.4))
labels = [r"$\delta_3$", r"$\delta_2$", r"$\delta_1$"]
order = [2, 1, 0]
xa = np.arange(3) * 1.0
for j, k in enumerate(order):
    for off, case, edge in ((-0.2, ex["all_steps"], "#334155"), (0.2, ex["last_only"], "#334155")):
        o, f_ = case["output_part"][k], case["future_part"][k]
        ax.bar(xa[j] + off, o, width=0.36, color="#FB923C", edgecolor=edge)
        ax.bar(xa[j] + off, f_, bottom=o, width=0.36, color="#4ADE80", edgecolor=edge)
        ax.text(xa[j] + off, o + f_ + 0.03, f"{case['delta'][k]:.2f}", ha="center", fontsize=12)
ax.set_xticks(xa); ax.set_xticklabels(labels, fontsize=14)
for j in range(3):
    ax.text(xa[j] - 0.2, -0.13, "every step", ha="center", fontsize=10, color="#334155")
    ax.text(xa[j] + 0.2, -0.13, "last step only", ha="center", fontsize=10, color="#334155")
ax.set_ylim(-0.2, 1.95); ax.set_xlim(-0.6, 2.6); ax.axhline(0, color="#334155", lw=1)
ax.tick_params(axis="x", pad=14)
ax.set_ylabel(r"state gradient $\delta_t$", fontsize=12); ax.grid(axis="y", alpha=0.3)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#FB923C", "#4ADE80")],
          labels=[r"path 1: own output $V\,\partial\mathcal{L}_t/\partial y_t$", r"path 2: from the future $W(1-h_{t+1}^2)\,\delta_{t+1}$"],
          loc="upper left", fontsize=11, frameon=False)
ax.set_title(r"computed backward from $t=3$ to $t=1$ ($V=1.0,\ W=0.5,\ 1-h^2=0.8$)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "lec08-kp02-delta-parts.png", dpi=200); plt.close(fig)
print("figures ->", OUT)
