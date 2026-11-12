"""lec08-kp06-figures.py — 第8週 KP06 講義（LSTM 三個門與細胞狀態傳送帶）的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec08-kp06-figures.py [輸出資料夾]（預設：與講義同層的 ../assets/figures/）"""
import json, math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams["mathtext.fontset"] = "cm"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent  # 課程 repo：assets/figures/src/ 的上一層 assets/figures/
OUT.mkdir(parents=True, exist_ok=True)

def sig(z): return 1.0 / (1.0 + math.exp(-z))
R = lambda v, n=4: round(v, n)

# ---------- source examples (教授祕笈) ----------
src = dict(sig_p2197=R(sig(2.197)), sig_m2197=R(sig(-2.197)), exp_m2197=R(math.exp(-2.197)), exp_p2197=R(math.exp(2.197)))
s_c = 0.8 * 2.0 + 0.5 * 0.6
src.update(keep=R(0.8 * 2.0), write=R(0.5 * 0.6), c=R(s_c), tanh_c=R(math.tanh(s_c)), h=R(0.9 * math.tanh(s_c)),
           h_3dp=round(0.9 * math.tanh(s_c), 3))

# ---------- own scalar running example (one full LSTM step, d = p = q = 1) ----------
P = dict(Wf=0.5, Uf=1.0, pf=0.2, bf=0.5,
         Wi=0.5, Ui=0.5, pi=0.1, bi=-0.5,
         Wc=0.4, Uc=0.6, bc=0.0,
         Wo=0.5, Uo=0.5, po=0.1, bo=0.0,
         V=1.2, by=0.0)
h0, c0, x1 = 0.5, 2.0, 1.0
def step(h, c, x, P, peep=True):
    k = 1.0 if peep else 0.0
    parts_f = [P["Wf"] * h, P["Uf"] * x, k * P["pf"] * c, P["bf"]]
    zf = sum(parts_f); f = sig(zf)
    parts_i = [P["Wi"] * h, P["Ui"] * x, k * P["pi"] * c, P["bi"]]
    zi = sum(parts_i); i = sig(zi)
    zc = P["Wc"] * h + P["Uc"] * x + P["bc"]; ct = math.tanh(zc)
    keep, write = f * c, i * ct
    cn = keep + write
    parts_o = [P["Wo"] * h, P["Uo"] * x, k * P["po"] * cn, P["bo"]]
    zo = sum(parts_o); o = sig(zo)
    tc = math.tanh(cn); hn = o * tc; y = P["V"] * hn + P["by"]
    return dict(parts_f=[R(v) for v in parts_f], zf=R(zf), f=R(f), parts_i=[R(v) for v in parts_i], zi=R(zi), i=R(i),
                zc=R(zc), ctilde=R(ct), keep=R(keep), write=R(write), c=R(cn), parts_o=[R(v) for v in parts_o], zo=R(zo),
                o=R(o), tanh_c=R(tc), h=R(hn), y=R(y), raw=dict(f=f, i=i, o=o, c=cn, h=hn))
ex = step(h0, c0, x1, P)
ex_nopeep = step(h0, c0, x1, P, peep=False)
# plain RNN of KP01 on the same input, for comparison: h = tanh(0.8*h0 + 1.0*x1)
rnn_h = math.tanh(0.8 * h0 + 1.0 * x1)

# ---------- gradient along the belt vs. plain RNN ----------
ks = list(range(0, 51))
belt = {f: [f ** k for k in ks] for f in (1.0, 0.99, 0.9)}
rnn_bound = [0.8 ** k for k in ks]
grad = dict(f099_50=R(0.99 ** 50), f09_50_4dp=R(0.9 ** 50), f09_50=f"{0.9 ** 50:.4g}", rnn08_50=f"{0.8 ** 50:.4g}",
            f09_10=R(0.9 ** 10), rnn08_10=R(0.8 ** 10),
            sig_max_slope=0.25, tanh_max_slope=1.0)

# ---------- practice ----------
pr1 = dict(c=R(0.05 * 2.0 + 0.95 * (-0.6)), keep=R(0.05 * 2.0), write=R(0.95 * (-0.6)))
k_half = next(k for k in range(1, 100) if 0.9 ** k < 0.5)
pr2 = dict(k_half=k_half, f09_k=R(0.9 ** k_half), f09_km1=R(0.9 ** (k_half - 1)))
d, p, q = 3, 4, 2
lstm_blocks = 4 * (p * p + p * d + p)
pr3 = dict(per_block=p * p + p * d + p, four_blocks=lstm_blocks, peep=3 * p, out=q * p + q,
           total=lstm_blocks + 3 * p + q * p + q, rnn_total=p * d + p * p + q * p + p + q)
pr3["ratio"] = round(pr3["total"] / pr3["rnn_total"], 1)
hadamard = [a * b for a, b in zip((0.5, 1.0), (4.0, 6.0))]

num = dict(source=src, params=P, start=dict(h0=h0, c0=c0, x1=x1),
           step={k: v for k, v in ex.items() if k != "raw"},
           no_peephole={k: v for k, v in ex_nopeep.items() if k in ("zf", "f", "zi", "i", "c", "zo", "o", "h")},
           rnn_kp01_h=R(rnn_h), hadamard_example=hadamard, gradient=grad, practice=dict(q1=pr1, q2=pr2, q3=pr3))
print(json.dumps(num, indent=1, ensure_ascii=False))

# ---------- figure 1: sigmoid as a soft switch ----------
fig, ax = plt.subplots(figsize=(9, 4.6))
zz = [-6 + 12 * j / 400 for j in range(401)]
ax.axhspan(0, 0.1, color="#FEE2E2", alpha=0.8); ax.axhspan(0.9, 1.0, color="#DCFCE7", alpha=0.8)
ax.plot(zz, [sig(z) for z in zz], color="#1E3A8A", lw=2.4, label=r"$\mathrm{sig}(z)=1/(1+e^{-z})$")
ax.text(0.3, 0.03, "nearly closed (pass < 10%)", fontsize=11, color="#B91C1C", va="bottom")
ax.text(-5.8, 0.93, "nearly open (pass > 90%)", fontsize=11, color="#15803D", va="bottom")
for z, lab in ((2.197, "0.90"), (-2.197, "0.10")):
    ax.plot([z], [sig(z)], "o", color="#64748B", ms=7)
    ax.annotate(rf"$z={z:+.3f}\Rightarrow {lab}$", (z, sig(z)), xytext=((2.6, 0.78) if z > 0 else (-5.6, 0.22)),
                fontsize=11, color="#475569", arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=0.8))
cols = dict(f="#DC2626", i="#2563EB", o="#16A34A")
for g, dy in (("f", -0.02), ("i", 0.13), ("o", -0.16)):
    z, v = ex["z" + g], ex[g]
    ax.plot([z], [v], "s", color=cols[g], ms=9, zorder=5)
    ax.annotate(rf"${g}_t={v:.4f}$", (z, v), xytext=({"f": z - 2.6, "i": z - 1.7, "o": z + 0.25}[g], v + dy), fontsize=12, color=cols[g],
                arrowprops=dict(arrowstyle="-", color=cols[g], lw=0.8))
ax.set_xlim(-6, 6); ax.set_ylim(-0.03, 1.03)
ax.set_xlabel(r"gate net input $z$", fontsize=12); ax.set_ylabel(r"gate value (fraction let through)", fontsize=12)
ax.grid(alpha=0.3); ax.legend(loc="center right", fontsize=11, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "lec08-kp06-sigmoid-gates.png", dpi=200); plt.close(fig)

# ---------- figure 2: the LSTM cell with the running example's numbers ----------
def box(ax, x, y, txt, fc, w=1.0, h=0.62, fs=14):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03,rounding_size=0.1", fc=fc, ec="#334155", lw=1.3))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs)
def node(ax, x, y, sym):
    ax.add_patch(Circle((x, y), 0.26, fc="white", ec="#334155", lw=1.5, zorder=4))
    ax.text(x, y, sym, ha="center", va="center", fontsize=15, zorder=5)
def arr(ax, a, b, color="#334155", lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=lw, color=color, linestyle=ls, zorder=3))
fig, ax = plt.subplots(figsize=(12, 5.6))
ax.set_xlim(-0.6, 12.6); ax.set_ylim(-0.9, 4.4); ax.axis("off")
BY = 3.4
ax.add_patch(FancyBboxPatch((0.1, BY - 0.12), 11.0, 0.24, boxstyle="round,pad=0,rounding_size=0.1", fc="#FEF3C7", ec="none", zorder=1))
ax.text(-0.15, BY, r"$c_{t-1}$", ha="right", va="center", fontsize=15)
ax.text(-0.15, BY - 0.42, f"{c0:.4f}", ha="right", va="center", fontsize=11, color="#92400E")
node(ax, 2.6, BY, r"$\times$"); node(ax, 5.2, BY, r"$+$")
arr(ax, (0.1, BY), (2.34, BY), color="#B45309", lw=3); arr(ax, (2.86, BY), (4.94, BY), color="#B45309", lw=3)
arr(ax, (5.46, BY), (11.6, BY), color="#B45309", lw=3)
ax.text(11.7, BY, r"$c_t$", ha="left", va="center", fontsize=15)
ax.text(11.7, BY - 0.42, f"{ex['c']:.4f}", ha="left", va="center", fontsize=11, color="#92400E")
ax.text(3.9, BY + 0.38, rf"keep $f_tc_{{t-1}}={ex['keep']:.4f}$", ha="center", fontsize=11, color="#92400E")
ax.text(7.0, BY + 0.38, "cell state: the conveyor belt (only $\\times$ and $+$)", ha="center", fontsize=11, color="#92400E")
# input bus
GY, IY = 1.2, -0.25
ax.plot([0.6, 9.4], [IY, IY], color="#64748B", lw=2)
ax.text(0.5, IY, r"$[h_{t-1},\,x_t]$", ha="right", va="center", fontsize=13)
ax.text(0.5, IY - 0.42, f"[{h0}, {x1}]", ha="right", va="center", fontsize=11, color="#475569")
gates = [(2.6, r"$f_t$" + "\nsig", "#FEE2E2", ex["f"]), (4.4, r"$i_t$" + "\nsig", "#DBEAFE", ex["i"]),
         (6.0, r"$\tilde c_t$" + "\ntanh", "#EDE9FE", ex["ctilde"]), (8.6, r"$o_t$" + "\nsig", "#DCFCE7", ex["o"])]
for x, t, fc, v in gates:
    arr(ax, (x, IY), (x, GY - 0.36), color="#64748B")
    box(ax, x, GY, t, fc, w=0.95, h=0.72, fs=12)
    ax.text(x + 0.55, GY - 0.48, f"{v:.4f}", ha="left", va="center", fontsize=11, color="#334155")
arr(ax, (2.6, GY + 0.36), (2.6, BY - 0.26))
node(ax, 5.2, 2.35, r"$\times$")
arr(ax, (4.4, GY + 0.36), (5.0, 2.17)); arr(ax, (6.0, GY + 0.36), (5.4, 2.17))
arr(ax, (5.2, 2.61), (5.2, BY - 0.26))
ax.text(5.55, 2.72, rf"write $i_t\tilde c_t={ex['write']:.4f}$", ha="left", fontsize=11, color="#1E3A8A")
# output branch
box(ax, 10.2, 2.35, "tanh", "#F1F5F9", w=0.85, h=0.5, fs=12)
arr(ax, (10.2, BY - 0.12), (10.2, 2.6))
node(ax, 10.2, GY, r"$\times$")
arr(ax, (10.2, 2.1), (10.2, GY + 0.26)); arr(ax, (9.08, GY), (9.94, GY))
ax.text(10.75, 2.35, f"{ex['tanh_c']:.4f}", ha="left", va="center", fontsize=11, color="#334155")
arr(ax, (10.46, GY), (11.6, GY), color="#1D4ED8", lw=2)
ax.text(11.7, GY, r"$h_t$", ha="left", va="center", fontsize=15, color="#1D4ED8")
ax.text(11.7, GY - 0.42, f"{ex['h']:.4f}", ha="left", va="center", fontsize=11, color="#1D4ED8")
# peepholes (dashed)
ax.plot([1.4, 1.4], [BY - 0.12, GY + 0.1], color="#9CA3AF", lw=1.2, ls="--")
arr(ax, (1.4, GY + 0.1), (2.12, GY + 0.1), color="#9CA3AF", lw=1.2, ls="--")
ax.plot([1.4, 3.7], [GY + 0.62, GY + 0.62], color="#9CA3AF", lw=1.2, ls="--")
arr(ax, (3.7, GY + 0.62), (3.93, GY + 0.3), color="#9CA3AF", lw=1.2, ls="--")
ax.plot([9.6, 9.6], [BY - 0.12, GY + 0.62], color="#9CA3AF", lw=1.2, ls="--")
arr(ax, (9.6, GY + 0.62), (9.07, GY + 0.36), color="#9CA3AF", lw=1.2, ls="--")
ax.text(1.32, 2.3, "peephole", rotation=90, ha="right", va="center", fontsize=10, color="#6B7280")
ax.text(6.0, -0.75, r"one step with $h_{t-1}=0.5,\ c_{t-1}=2.0,\ x_t=1.0$; dashed = peephole (sees $c_{t-1}$ for $f,i$; $c_t$ for $o$)",
        ha="center", fontsize=11, color="#475569")
fig.tight_layout(); fig.savefig(OUT / "lec08-kp06-cell.png", dpi=200); plt.close(fig)

# ---------- figure 3: gradient factor along the belt vs. plain RNN ----------
fig, ax = plt.subplots(figsize=(9, 4.6))
for f, c, lab in ((1.0, "#B45309", r"belt, $f_t=1$: $1^k$"), (0.99, "#D97706", r"belt, $f_t=0.99$: $0.99^k$"),
                  (0.9, "#F59E0B", r"belt, $f_t=0.9$: $0.9^k$")):
    ax.semilogy(ks, belt[f], color=c, lw=2.4, label=lab)
ax.semilogy(ks, rnn_bound, color="#2563EB", lw=2.4, ls="--", label=r"plain RNN bound, $W=0.8$: $0.8^k$")
for y, txt, col, fy in ((1.0, "1.0000", "#B45309", 1.35), (0.99 ** 50, f"{0.99 ** 50:.4f}", "#D97706", 0.45),
                        (0.9 ** 50, f"{0.9 ** 50:.4f}", "#F59E0B", 1.8), (0.8 ** 50, r"$1.43\times10^{-5}$", "#2563EB", 0.3)):
    ax.plot([50], [y], "o", color=col); ax.text(49.2, y * fy, txt, ha="right", fontsize=11, color=col)
ax.set_xlabel(r"number of steps travelled back, $k$", fontsize=12)
ax.set_ylabel("gradient factor (log scale)", fontsize=12)
ax.set_xlim(0, 52); ax.set_ylim(1e-6, 3); ax.grid(alpha=0.3, which="both")
ax.legend(loc="lower left", fontsize=11, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "lec08-kp06-cec.png", dpi=200); plt.close(fig)
print("figures ->", OUT)
