"""lec10-kp06-figures.py — 第10週 KP06 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec10-kp06-figures.py [輸出資料夾]（預設：上一層的 assets/figures/）"""
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


sig = lambda x: 1 / (1 + np.exp(-x))
num = {}
# --- spatial attention over a 7x7 grid [Added]: scores = 4 * exp(-dist^2 / 2) around a centre
G = 7
yy, xx = np.mgrid[0:G, 0:G]


def grid_weights(cy, cx):
    s = 4.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 2.0)
    return softmax(s.ravel()).reshape(G, G)


Wel = grid_weights(2, 3)   # "elephant": upper middle
Wwa = grid_weights(5.5, 3)  # "water": bottom
num["grid"] = dict(size=G, positions=G * G, elephant_max=round(float(Wel.max()), 4), elephant_sum=round(float(Wel.sum()), 6),
                   elephant_top9=round(float(np.sort(Wel.ravel())[::-1][:9].sum()), 4), uniform=round(1 / (G * G), 4),
                   water_max=round(float(Wwa.max()), 4))
num["show_attend_tell"] = dict(grid=14, positions=14 * 14, words=10, scores=14 * 14 * 10)
num["speech"] = dict(seconds=3, frame_ms=10, frames=300, chars=20, scores=300 * 20)
# --- SENet: source example (C = 2) and a 4-channel example [Added]
num["se_source"] = dict(z=[2.0, 0.0], scores=[2.0, -2.0], s=[round(float(sig(2.0)), 4), round(float(sig(-2.0)), 4)],
                        X1_scaled=round(float(2 * sig(2.0)), 4))
z4 = np.array([2.0, 0.5, 1.0, 0.0]); e4 = np.array([2.0, -1.0, 0.5, -2.0])
s4 = sig(e4)
num["se_four"] = dict(z=z4.tolist(), scores=e4.tolist(), s=s4.round(4).tolist(), s_sum=round(float(s4.sum()), 4),
                      scaled_means=(z4 * s4).round(4).tolist())
# practice
num["practice2"] = dict(scores=[0.0, 3.0, -1.0], s=[round(float(sig(x)), 4) for x in (0.0, 3.0, -1.0)])
assert abs(num["se_source"]["s"][0] - 0.8808) < 1e-4
print(json.dumps(num, indent=1))

# ---------- figure 1: spatial attention maps for two words ----------
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
for ax, Wm, word in zip(axes, (Wel, Wwa), ("elephant", "water")):
    im = ax.imshow(Wm, cmap="magma", vmin=0, vmax=Wel.max())
    ax.set_xticks(range(G)); ax.set_yticks(range(G))
    ax.set_title(f'writing "{word}": max weight {Wm.max():.4f}', fontsize=12)
    for i in range(G + 1):
        ax.axhline(i - 0.5, color="white", lw=0.3); ax.axvline(i - 0.5, color="white", lw=0.3)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="attention weight")
fig.suptitle(r"Spatial attention over a $7\times7$ grid of CNN features (49 weights, sum = 1)", fontsize=12)
fig.savefig(OUT / "lec10-kp06-spatial-attention.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- figure 2: channel attention (squeeze, excitation, scale) ----------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True)
ch = [f"ch {i + 1}" for i in range(4)]
for ax, vals, title, col, fmt in ((axes[0], z4, r"squeeze: channel means $z_c$", "#64748B", "{:.2f}"),
                                  (axes[1], s4, r"excitation: gates $s_c=\sigma(\cdot)$", "#D97706", "{:.4f}"),
                                  (axes[2], z4 * s4, r"scale: $s_c z_c$ (mean after gating)", "#2563EB", "{:.4f}")):
    bars = ax.bar(ch, vals, color=col, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, fmt.format(v), ha="center", fontsize=10)
    ax.set_title(title, fontsize=11); ax.set_ylim(0, 2.3); ax.grid(axis="y", alpha=0.3)
axes[1].axhline(1, color="#94A3B8", ls=":", lw=1)
fig.suptitle("Channel attention (SENet): each channel gets its own gate in (0, 1); gates need not sum to 1", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "lec10-kp06-channel-attention.png", dpi=200); plt.close(fig)
