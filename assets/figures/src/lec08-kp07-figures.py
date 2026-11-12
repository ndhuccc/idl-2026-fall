"""lec08-kp07-figures.py — 第8週 KP07 講義（LSTM 變體、GRU、MGU）的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec08-kp07-figures.py [輸出資料夾]（預設：與講義同層的 ../assets/figures/）"""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent  # 課程 repo：assets/figures/src/ 的上一層 assets/figures/
OUT.mkdir(parents=True, exist_ok=True)
sech2 = lambda c: 1.0 / math.cosh(c) ** 2   # = 1 - tanh^2(c), without round-off to 0
sig = lambda z: 1.0 / (1.0 + math.exp(-z))
r4 = lambda v: round(v, 4)

# ---------- 1. cell state with and without a forget gate (source: 50 steps, i*c~ = 0.5, f = 0.7) ----------
N, add, f = 50, 0.5, 0.7
c_orig, c_forget = [0.0], [0.0]
for t in range(N):
    c_orig.append(c_orig[-1] + add)          # 1997: c_t = c_{t-1} + i_t c~_t
    c_forget.append(f * c_forget[-1] + add)  # with forget gate: c_t = f c_{t-1} + i_t c~_t
c_orig, c_forget = np.array(c_orig), np.array(c_forget)
limit = add / (1 - f)
cell = dict(N=N, add=add, f=f, c50_orig=r4(c_orig[-1]), c50_forget=r4(c_forget[-1]), limit=r4(limit),
            c10_orig=r4(c_orig[10]), c10_forget=r4(c_forget[10]),
            tanh_c50_orig=r4(math.tanh(c_orig[-1])), tanh_c50_forget=r4(math.tanh(c_forget[-1])),
            slope_orig=float(f"{sech2(c_orig[-1]):.3e}"), slope_forget=r4(sech2(c_forget[-1])),
            slope_c10_orig=float(f"{sech2(c_orig[10]):.3e}"))

# ---------- 2. scalar GRU, one step ----------
h_prev, x = 0.8, 1.0
Wr, Ur, br = 1.0, -1.0, 0.0
Wz, Uz, bz = -1.0, 1.0, 0.0
Wc, Uc, bc = 1.0, -1.5, 0.0
a_r = Wr * h_prev + Ur * x + br; r = sig(a_r)
a_z = Wz * h_prev + Uz * x + bz; z = sig(a_z)
a_c = Wc * (r * h_prev) + Uc * x + bc; hc = math.tanh(a_c)
h = (1 - z) * h_prev + z * hc
# no reset (r = 1) for comparison
a_c1 = Wc * h_prev + Uc * x + bc; hc1 = math.tanh(a_c1); h1 = (1 - z) * h_prev + z * hc1
gru = dict(h_prev=h_prev, x=x, a_r=r4(a_r), r=r4(r), a_z=r4(a_z), z=r4(z), rh=r4(r * h_prev), a_c=r4(a_c), hc=r4(hc),
           keep=r4((1 - z) * h_prev), write=r4(z * hc), h=r4(h),
           no_reset=dict(a_c=r4(a_c1), hc=r4(hc1), write=r4(z * hc1), h=r4(h1)),
           z0_h=h_prev, z1_h=r4(hc), one_minus_z=r4(1 - z),
           rounding_gap=r4(abs((r4((1 - z) * h_prev) + r4(z * hc)) - r4(h))))
# source worked numbers: h_{t-1}=0.8, r=1, h~=-0.4, z=0.7
gru["source_example_h"] = r4((1 - 0.7) * 0.8 + 0.7 * (-0.4))

# ---------- 3. scalar MGU, same step: one gate f plays both roles (W_f, U_f = W_z, U_z) ----------
Wf, Uf, bf = Wz, Uz, bz
a_f = Wf * h_prev + Uf * x + bf; fg = sig(a_f)
a_cm = Wc * (fg * h_prev) + Uc * x + bc; hcm = math.tanh(a_cm)
hm = (1 - fg) * h_prev + fg * hcm
mgu = dict(a_f=r4(a_f), f=r4(fg), fh=r4(fg * h_prev), a_c=r4(a_cm), hc=r4(hcm), keep=r4((1 - fg) * h_prev),
           write=r4(fg * hcm), h=r4(hm))

# ---------- 4. parameter counts (recurrent cell only: one block = W (p x p) + U (p x d) + b (p)) ----------
def counts(d, p, q=None, r_proj=None):
    blk = p * p + p * d + p
    c = dict(d=d, p=p, pp=p * p, pd=p * d, block=blk, RNN=blk, MGU=2 * blk, GRU=3 * blk, LSTM=4 * blk,
             LSTM_peephole=4 * blk + 3 * p, LSTM_full_gate_recurrence_extra=9 * p * p,
             GRU_vs_LSTM_saving=r4(1 - 3 / 4), MGU_vs_LSTM_saving=r4(1 - 2 / 4), MGU_vs_GRU_saving=r4(1 - 2 / 3),
             GRU_vs_peephole_LSTM_saving=r4(1 - 3 * blk / (4 * blk + 3 * p)),
             two_bias=dict(LSTM=4 * (blk + p), GRU=3 * (blk + p), MGU=2 * (blk + p),
                           GRU_vs_LSTM_saving=r4(1 - 3 * (blk + p) / (4 * (blk + p)))))
    c["MGU_saved_vs_LSTM"] = c["LSTM"] - c["MGU"]
    if q is not None:
        out = q * p + q
        c["with_output"] = dict(q=q, out=out, LSTM=4 * blk + out, GRU=3 * blk + out, MGU=2 * blk + out,
                                GRU_vs_LSTM_saving=r4(1 - (3 * blk + out) / (4 * blk + out)))
    if r_proj is not None:
        rr = r_proj
        lstmp = 4 * (p * rr + p * d + p) + rr * p
        c["LSTMP"] = dict(r=rr, gates=4 * (p * rr + p * d + p), proj=rr * p, total=lstmp,
                          saving_vs_LSTM=r4(1 - lstmp / (4 * blk)))
    return c
small = counts(3, 4, q=2)
big = counts(256, 256, r_proj=128)
# ---------- practice answers ----------
ex1 = counts(5, 8)
x0 = 0.0
ar0 = Wr * h_prev + Ur * x0 + br; r0 = sig(ar0)
az0 = Wz * h_prev + Uz * x0 + bz; z0 = sig(az0)
ac0 = Wc * (r0 * h_prev) + Uc * x0 + bc; hc0 = math.tanh(ac0)
ex2 = dict(x=x0, a_r=r4(ar0), r=r4(r0), a_z=r4(az0), z=r4(z0), a_c=r4(ac0), hc=r4(hc0),
           keep=r4((1 - z0) * h_prev), write=r4(z0 * hc0), h=r4((1 - z0) * h_prev + z0 * hc0))
lim9 = add / (1 - 0.9)
ex3 = dict(f=0.9, limit=r4(lim9), tanh_limit=r4(math.tanh(lim9)), slope=float(f"{sech2(lim9):.3e}"))
practice = dict(q1=dict(block=ex1["block"], LSTM=ex1["LSTM"], GRU=ex1["GRU"], MGU=ex1["MGU"]), q2=ex2, q3=ex3)
num = dict(cell=cell, gru=gru, mgu=mgu, params_small=small, params_big=big, practice=practice)
print(json.dumps(num, indent=1, ensure_ascii=False))

# ---------- figure 1: cell state growth ----------
t = np.arange(N + 1)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axs[0]
ax.plot(t, c_orig, color="#DC2626", lw=2.2, label=r"no forget gate ($f_t=1$, 1997)")
ax.plot(t, c_forget, color="#2563EB", lw=2.2, label=r"forget gate $f_t=0.7$ (2000)")
ax.axhline(limit, color="#2563EB", ls="--", lw=1)
ax.text(30, limit + 1.0, rf"limit $0.5/(1-0.7)={limit:.4f}$", color="#2563EB", fontsize=11)
ax.annotate(rf"$c_{{50}}={c_orig[-1]:.1f}$", xy=(N, c_orig[-1]), xytext=(38, 24.5), color="#DC2626", fontsize=11, ha="right", va="center",
            arrowprops=dict(arrowstyle="->", color="#DC2626"))
ax.set_xlim(0, N); ax.set_ylim(0, 27)
ax.set_xlabel(r"time step $t$", fontsize=12); ax.set_ylabel(r"cell state $c_t$", fontsize=12)
ax.set_title(r"(a) $c_t=f_t\,c_{t-1}+0.5$", fontsize=12)
ax.legend(loc="upper left", fontsize=10, frameon=False); ax.grid(alpha=0.3)
ax = axs[1]
ax.semilogy(t[1:], 1 / np.cosh(c_orig[1:]) ** 2, color="#DC2626", lw=2.2, label=r"$f_t=1$")
ax.semilogy(t[1:], 1 / np.cosh(c_forget[1:]) ** 2, color="#2563EB", lw=2.2, label=r"$f_t=0.7$")
ax.annotate(r"$%.2f\times10^{%d}$" % (sech2(c_orig[-1]) / 10 ** math.floor(math.log10(sech2(c_orig[-1]))), math.floor(math.log10(sech2(c_orig[-1])))), xy=(N, sech2(c_orig[-1])), xytext=(33, 1e-21), color="#DC2626", fontsize=11, ha="center",
            arrowprops=dict(arrowstyle="->", color="#DC2626"))
ax.text(N - 0.5, 0.25, rf"${sech2(c_forget[-1]):.4f}$", color="#2563EB", fontsize=11, ha="right")
ax.set_xlim(0, N); ax.set_ylim(1e-23, 3)
ax.set_xlabel(r"time step $t$", fontsize=12); ax.set_ylabel(r"slope of $\tanh$ at $c_t$:  $1-\tanh^2(c_t)$", fontsize=12)
ax.set_title(r"(b) how much $\tanh(c_t)$ can still change", fontsize=12)
ax.legend(loc="lower left", fontsize=10, frameon=False); ax.grid(alpha=0.3, which="major")
fig.tight_layout(); fig.savefig(OUT / "lec08-kp07-cell-growth.png", dpi=200); plt.close(fig)

# ---------- figure 2: GRU convex combination ----------
zz = np.linspace(0, 1, 401)
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(zz, (1 - zz) * h_prev + zz * hc, color="#16A34A", lw=2.4, label=rf"with reset $r_t={r:.4f}$:  $\tilde{{h}}_t={hc:.4f}$")
ax.plot(zz, (1 - zz) * h_prev + zz * hc1, color="#9333EA", lw=1.8, ls="--", label=rf"no reset $r_t=1$:  $\tilde{{h}}_t={hc1:.4f}$")
ax.axhline(0, color="#334155", lw=0.8)
ax.plot([0], [h_prev], "o", color="#334155", ms=8); ax.text(0.02, h_prev + 0.06, rf"$z_t=0$: keep $h_{{t-1}}={h_prev}$", fontsize=11)
ax.plot([1], [hc], "o", color="#16A34A", ms=8); ax.text(0.98, hc - 0.13, rf"$z_t=1$: take $\tilde{{h}}_t$", fontsize=11, ha="right", color="#16A34A")
ax.axvline(z, color="#EA580C", lw=1, ls=":")
ax.plot([z], [h], "s", color="#EA580C", ms=9)
ax.annotate(rf"$z_t={z:.4f}$:  $h_t={h:.4f}$", xy=(z, h), xytext=(z + 0.06, h + 0.35), fontsize=12, color="#EA580C",
            arrowprops=dict(arrowstyle="->", color="#EA580C"))
ax.plot([z], [h1], "s", color="#9333EA", ms=7)
ax.annotate(rf"$h_t={h1:.4f}$", xy=(z, h1), xytext=(z - 0.36, h1 - 0.32), fontsize=11, color="#9333EA",
            arrowprops=dict(arrowstyle="->", color="#9333EA"))
ax.set_xlim(-0.03, 1.03); ax.set_ylim(-1.0, 1.05)
ax.set_xlabel(r"update gate $z_t$", fontsize=12); ax.set_ylabel(r"new state $h_t=(1-z_t)\,h_{t-1}+z_t\,\tilde{h}_t$", fontsize=12)
ax.legend(loc="lower left", fontsize=10.5, frameon=False); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "lec08-kp07-gru-mix.png", dpi=200); plt.close(fig)

# ---------- figure 3: parameter counts, p = d = 256 ----------
names = ["RNN", "MGU", "GRU", "LSTM"]
vals = [big[n] for n in names]
blocks = [1, 2, 3, 4]
gates = ["0 gates", "1 gate", "2 gates", "3 gates"]
cols = ["#94A3B8", "#16A34A", "#2563EB", "#DC2626"]
fig, ax = plt.subplots(figsize=(9.5, 4.2))
ys = np.arange(len(names))[::-1]
ax.barh(ys, vals, color=cols, height=0.6)
for y, v, b, g in zip(ys, vals, blocks, gates):
    ax.text(v + 8000, y, f"{v:,}   ({b} block{'s' if b > 1 else ''}, {g})", va="center", fontsize=11)
ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=12)
ax.set_xlim(0, 820000)
ax.set_xlabel(r"parameters of the recurrent cell, $p=d=256$;  one block $=p^2+pd+p=131{,}328$", fontsize=11)
ax.ticklabel_format(axis="x", style="plain")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.grid(axis="x", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "lec08-kp07-params.png", dpi=200); plt.close(fig)
print("figures ->", OUT)
