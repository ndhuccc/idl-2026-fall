"""lec10-kp01-figures.py — 第10週 KP01 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec10-kp01-figures.py [輸出資料夾]（預設：上一層的 assets/figures/）"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams["mathtext.fontset"] = "cm"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT.mkdir(parents=True, exist_ok=True)

# ---------- numbers ----------
H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


def attend(q):
    s = H @ np.asarray(q, float)
    w = softmax(s)
    return s, w, w @ H


num = {}
for name, q in (("q1", [2, 0]), ("q2", [0, 2]), ("practice_q", [1, 1]), ("zero_q", [0, 0])):
    s, w, c = attend(q)
    num[name] = dict(q=q, scores=s.round(4).tolist(), weights=w.round(4).tolist(), output=c.round(4).tolist(),
                     weight_sum=round(float(w.sum()), 6))
num["average"] = H.mean(0).round(4).tolist()
num["exp2"] = round(float(np.exp(2)), 3)
num["denominator_q1"] = round(float(2 * np.exp(2) + 1), 3)
base = np.array([2.0, 1.0, 0.0])
num["sharpness"] = {str(k): softmax(k * base).round(6).tolist() for k in (1, 3, 10)}
num["shift"] = softmax(base + 5).round(4).tolist()
assert np.allclose(num["q1"]["weights"], [0.4683, 0.0634, 0.4683])
assert np.allclose(num["q1"]["output"], [0.9366, 0.5317])
print(json.dumps(num, indent=1))

# ---------- figure 1: weighted average in the plane ----------
fig, ax = plt.subplots(figsize=(6.4, 6.0))
tri = Polygon(H, closed=True, fc="#E0E7FF", ec="#6366F1", lw=1.2, alpha=0.6)
ax.add_patch(tri)
for (x, y), lab, off in zip(H, (r"$\mathbf{h}_1$", r"$\mathbf{h}_2$", r"$\mathbf{h}_3$"), ((0.06, -0.09), (-0.16, 0.04), (0.05, 0.05))):
    ax.plot(x, y, "o", color="#1E293B", ms=8, zorder=5)
    ax.text(x + off[0], y + off[1], lab, fontsize=16)
avg = H.mean(0)
ax.plot(*avg, "s", color="#64748B", ms=10, zorder=6)
ax.text(avg[0] - 0.05, avg[1], "average (1/3 each)", fontsize=11, color="#475569", va="center", ha="right")
for key, col, lab, toff in (("q1", "#DC2626", r"$\mathbf{q}=[2,0]^\top$", (0.04, -0.13)), ("q2", "#2563EB", r"$\mathbf{q}=[0,2]^\top$", (-0.45, 0.13))):
    c = np.array(num[key]["output"])
    ax.plot(*c, "*", color=col, ms=18, zorder=7)
    ax.text(c[0] + toff[0], c[1] + toff[1], lab + "\n" + rf"$\mathbf{{c}}=[{c[0]:.4f},\ {c[1]:.4f}]^\top$", fontsize=11, color=col)
    ax.annotate("", xy=c, xytext=avg, arrowprops=dict(arrowstyle="->", color=col, lw=1.6, ls="--"))
ax.set_xlim(-0.25, 1.35); ax.set_ylim(-0.25, 1.35); ax.set_aspect("equal")
ax.set_xlabel("first component"); ax.set_ylabel("second component"); ax.grid(alpha=0.3)
ax.set_title("Outputs stay inside the triangle; the query moves them", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "lec10-kp01-weighted-average.png", dpi=200); plt.close(fig)

# ---------- figure 2: scaling the scores sharpens softmax ----------
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
for ax, k in zip(axes, (1, 3, 10)):
    w = softmax(k * base)
    bars = ax.bar([0, 1, 2], w, color=["#DC2626", "#F59E0B", "#94A3B8"], width=0.6)
    for b, v in zip(bars, w):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, (f"{v:.4f}" if 1e-3 <= v <= 0.999 else (f"{v:.5f}" if v > 0.999 else f"{v:.1e}")), ha="center", fontsize=11)
    ax.set_xticks([0, 1, 2], [rf"$s_1={2*k:g}$", rf"$s_2={1*k:g}$", rf"$s_3={0*k:g}$"])
    ax.set_title(rf"scores $\times{k}$", fontsize=13); ax.set_ylim(0, 1.12); ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("softmax weight")
fig.suptitle(r"Same ranking $[2,1,0]$, larger scale $\Rightarrow$ sharper weights", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "lec10-kp01-softmax-sharpness.png", dpi=200); plt.close(fig)
