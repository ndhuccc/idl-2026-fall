"""lec11-kp02-figures.py — 第11週 KP02 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec11-kp02-figures.py [輸出資料夾]（預設：上一層的 assets/figures/）"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT.mkdir(parents=True, exist_ok=True)


def pe(pos, d):
    v = np.zeros(d)
    for i in range(d // 2):
        w = 1.0 / 10000 ** (2 * i / d)
        v[2 * i] = np.sin(pos * w)
        v[2 * i + 1] = np.cos(pos * w)
    return v


num = {}
# source worked example: d = 4, pos = 1
p1 = pe(1, 4)
num["d4_pos1"] = p1.round(5).tolist()
num["d4_pos0"] = pe(0, 4).tolist()
num["d4_pos2"] = pe(2, 4).round(4).tolist()
num["omega_d4"] = [1.0, 0.01]
assert np.allclose(p1, [0.8415, 0.5403, 0.0100, 0.99995], atol=5e-5)
# wavelengths for d = 512
d = 512
num["wavelength_d512"] = dict(i0=round(2 * np.pi, 4), i255=round(2 * np.pi * 10000 ** (510 / 512), 1), limit=round(2 * np.pi * 10000, 1))
# relative position: dot product depends only on k
for d_ in (4, 64):
    dots = {f"pos{p}_k4": round(float(pe(p, d_) @ pe(p + 4, d_)), 4) for p in (0, 10, 37)}
    num[f"dot_d{d_}"] = dots
    assert len(set(dots.values())) == 1
num["norm_sq_d64"] = round(float(pe(5, 64) @ pe(5, 64)), 6)          # always d/2
num["cos_sim_d64_pos10_14"] = round(float(pe(10, 64) @ pe(14, 64) / 32), 4)
ks = np.arange(0, 61)
sim64 = np.array([pe(0, 64) @ pe(k, 64) / 32 for k in ks])
num["cos_sim_d64"] = {str(k): round(float(sim64[k]), 4) for k in (0, 1, 2, 4, 10, 20, 40)}
# rotation matrix for one pair: R_k [sin, cos] -> shifted
w, k, pos = 1.0, 4, 10
R = np.array([[np.cos(w * k), np.sin(w * k)], [-np.sin(w * k), np.cos(w * k)]])
assert np.allclose(R @ np.array([np.sin(w * pos), np.cos(w * pos)]), [np.sin(w * (pos + k)), np.cos(w * (pos + k))])
num["rotation_check"] = "R_k [sin(w pos), cos(w pos)] = [sin(w(pos+k)), cos(w(pos+k))] verified for w=1, k=4, pos=10"
print(json.dumps(num, indent=1))

# ---------- figure 1: the PE matrix and three dimensions ----------
P = np.stack([pe(p, 64) for p in range(50)])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axes[0]
im = ax.imshow(P, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xlabel(r"dimension of $\mathbf{p}_{pos}$  ($d=64$)"); ax.set_ylabel(r"position $pos$")
ax.set_title("each row is one position's code", fontsize=12)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
ax = axes[1]
xs = np.linspace(0, 49, 600)
for dim, col in ((0, "#BE123C"), (10, "#1D4ED8"), (40, "#15803D")):
    i = dim // 2
    ax.plot(xs, np.sin(xs / 10000 ** (2 * i / 64)), color=col, lw=2, label=f"dim {dim} (pair i={i}): wavelength {2*np.pi*10000**(2*i/64):.1f}")
ax.set_xlabel(r"position $pos$"); ax.set_ylabel("value"); ax.set_ylim(-1.15, 1.6); ax.set_xlim(0, 49)
ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3)
ax.set_title("low dims oscillate fast, high dims slowly", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "lec11-kp02-pe-waves.png", dpi=200); plt.close(fig)

# ---------- figure 2: similarity depends only on the offset k ----------
fig, ax = plt.subplots(figsize=(8.5, 4))
ax.plot(ks, sim64, color="#7C3AED", lw=2.2, label=r"$\mathbf{p}_{0}^\top\mathbf{p}_{k}\,/\,(d/2)$")
for p0, mk, col, ms in ((10, "o", "#BE123C", 11), (37, "s", "#15803D", 6)):
    sims = [pe(p0, 64) @ pe(p0 + k, 64) / 32 for k in ks[::5]]
    ax.plot(ks[::5], sims, mk, color=col, ms=ms, mfc="none", mew=1.6, label=rf"start at $pos={p0}$")
ax.axhline(0, color="#94A3B8", lw=1)
ax.set_xlabel(r"offset $k$ between two positions"); ax.set_ylabel("cosine similarity")
ax.set_xlim(0, 60); ax.set_ylim(-0.1, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=10)
ax.set_title(r"$d=64$: similarity depends only on $k$, not on where you start", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "lec11-kp02-relative-offset.png", dpi=200); plt.close(fig)
