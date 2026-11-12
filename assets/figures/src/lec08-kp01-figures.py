"""lec08-kp01-figures.py — 第8週 KP01 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec08-kp01-figures.py [輸出資料夾]（預設：與講義同層的 ../assets/figures/）"""
import json, math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["mathtext.fontset"] = "cm"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent  # 課程 repo：assets/figures/src/ 的上一層 assets/figures/
OUT.mkdir(parents=True, exist_ok=True)

# ---------- numbers ----------
W, U, V, bi, by, h0 = 0.8, 1.0, 1.2, 0.0, 0.0, 0.0
xs = [1.0, -0.5]
def run(W):
    h, rows = h0, []
    for x in xs:
        wh, ux = W * h, U * x
        i = wh + ux + bi
        hn = math.tanh(i)
        rows.append(dict(Wh=wh, Ux=ux, i=i, h=hn, y=V * hn + by))
        h = hn
    return rows
mem, nomem = run(W), run(0.0)
d, p, q = 3, 4, 2
shared = p * d + p * p + q * p + p + q
num = dict(mem=mem, nomem=nomem, dims=dict(d=d, p=p, q=q, U=p * d, W=p * p, V=q * p, bi=p, by=q, shared=shared),
           per_step={T: shared * T for T in (10, 100)})
print(json.dumps(num, indent=1))
pass  # 數字印在標準輸出；任務另存 reports/<unit>/數值驗算.json

def box(ax, x, y, txt, fc, w=0.9, h=0.55, fs=15):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fc, ec="#334155", lw=1.4))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs)
def arrow(ax, a, b, lab=None, off=(0, 0), color="#334155", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.6, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if lab:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], lab, ha="center", va="center", fontsize=14, color=color)

# ---------- figure 1: folded vs unrolled ----------
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.set_xlim(-0.6, 11.2); ax.set_ylim(-0.5, 3.6); ax.axis("off")
# folded
box(ax, 0.6, 0.4, r"$\mathbf{x}_t$", "#DCFCE7"); box(ax, 0.6, 1.7, r"$\mathbf{h}_t$", "#DBEAFE"); box(ax, 0.6, 3.0, r"$\mathbf{y}_t$", "#FEF3C7")
arrow(ax, (0.6, 0.68), (0.6, 1.42), r"$\mathbf{U}$", (-0.32, 0)); arrow(ax, (0.6, 1.98), (0.6, 2.72), r"$\mathbf{V}$", (-0.32, 0))
ax.add_patch(FancyArrowPatch((1.05, 1.85), (1.05, 1.55), arrowstyle="-|>", mutation_scale=14, lw=1.6, color="#1D4ED8",
                             connectionstyle="arc3,rad=-2.2"))
ax.text(1.5, 1.7, r"$\mathbf{W}$", fontsize=14, color="#1D4ED8", va="center")
ax.text(0.9, -0.35, "(a) folded: one loop", ha="center", fontsize=13)
ax.annotate("", xy=(3.0, 1.7), xytext=(2.1, 1.7), arrowprops=dict(arrowstyle="->", lw=2.2, color="#64748B"))
ax.text(2.55, 1.95, "unroll", ha="center", fontsize=12, color="#64748B")
# unrolled
xs_ = [5.2, 7.4, 9.6]
box(ax, 3.6, 1.7, r"$\mathbf{h}_0$", "#F1F5F9", w=0.7)
prev = (3.95, 1.7)
for k, X in enumerate(xs_, start=1):
    box(ax, X, 0.4, rf"$\mathbf{{x}}_{k}$", "#DCFCE7"); box(ax, X, 1.7, rf"$\mathbf{{h}}_{k}$", "#DBEAFE"); box(ax, X, 3.0, rf"$\mathbf{{y}}_{k}$", "#FEF3C7")
    arrow(ax, (X, 0.68), (X, 1.42), r"$\mathbf{U}$", (-0.3, 0)); arrow(ax, (X, 1.98), (X, 2.72), r"$\mathbf{V}$", (-0.3, 0))
    arrow(ax, prev, (X - 0.45, 1.7), r"$\mathbf{W}$", (0, 0.22), color="#1D4ED8")
    prev = (X + 0.45, 1.7)
ax.text(7.2, -0.35, "(b) unrolled over t = 1, 2, 3: the same U, W, V at every step", ha="center", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "lec08-kp01-unrolled.png", dpi=200); plt.close(fig)

# ---------- figure 2: what builds i_2, with and without memory ----------
fig, ax = plt.subplots(figsize=(9, 4.4))
labels = [r"$W=0.8$" + "\n(with memory)", r"$W=0$" + "\n(no memory)"]
for row, (lab, r) in enumerate(zip(labels, [mem[1], nomem[1]])):
    y = 1 - row
    ax.barh(y + 0.22, r["Wh"], height=0.2, color="#2563EB")
    ax.barh(y, r["Ux"], height=0.2, color="#EA580C")
    ax.barh(y - 0.22, r["i"], height=0.2, color="#16A34A")
    ax.text(r["Wh"] + (0.02 if r["Wh"] >= 0 else -0.02), y + 0.22, f"$Wh_1={r['Wh']:+.4f}$", va="center",
            ha="left" if r["Wh"] >= 0 else "right", fontsize=12)
    ax.text(r["Ux"] - 0.02, y, f"$Ux_2={r['Ux']:+.4f}$", va="center", ha="right", fontsize=12)
    ax.text(r["i"] + (0.02 if r["i"] >= 0 else -0.02), y - 0.22, rf"$i_2={r['i']:+.4f}\;\Rightarrow\;h_2={r['h']:+.4f}$",
            va="center", ha="left" if r["i"] >= 0 else "right", fontsize=12)
ax.axvline(0, color="#334155", lw=1)
ax.set_yticks([1, 0]); ax.set_yticklabels(labels, fontsize=12)
ax.set_xlim(-1.35, 1.35); ax.set_ylim(-0.5, 1.5)
ax.set_xlabel("contribution to the net input at step 2", fontsize=12)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#2563EB", "#EA580C", "#16A34A")],
          labels=[r"old memory $Wh_1$", r"new input $Ux_2$", r"sum $i_2$"], loc="upper left", fontsize=11, frameon=False)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "lec08-kp01-memory.png", dpi=200); plt.close(fig)
print("figures ->", OUT)
