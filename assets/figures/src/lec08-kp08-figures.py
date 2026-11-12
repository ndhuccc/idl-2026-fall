"""lec08-kp08-figures.py — 第8週 KP08 講義的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec08-kp08-figures.py [輸出資料夾]（預設：與講義同層的 ../assets/figures/）"""
import json, math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["mathtext.fontset"] = "cm"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent  # 課程 repo：assets/figures/src/ 的上一層 assets/figures/
OUT.mkdir(parents=True, exist_ok=True)

# ---------- numbers: scalar BiRNN ----------
Wf, Uf = 0.5, 1.0          # forward chain
Wb, Ub = 0.6, 0.8          # backward chain (its own, different weights)
Vf, Vb, by = 1.0, 1.0, 0.0
seqA = [1.0, 0.5, 1.0]
seqB = [1.0, 0.5, -1.0]

def birnn(xs):
    T = len(xs)
    hf, h = [], 0.0
    fwd_i = []
    for x in xs:                       # t = 1 .. T, starts from h_0 = 0
        i = Wf * h + Uf * x; h = math.tanh(i); fwd_i.append(i); hf.append(h)
    hb, h = [0.0] * T, 0.0
    bwd_i = [0.0] * T
    for t in reversed(range(T)):       # t = T .. 1, starts from h_{T+1} = 0
        i = Wb * h + Ub * xs[t]; h = math.tanh(i); bwd_i[t] = i; hb[t] = h
    y = [Vf * a + Vb * b + by for a, b in zip(hf, hb)]
    return dict(x=xs, fwd_i=fwd_i, hf=hf, bwd_i=bwd_i, hb=hb, y=y)

A, B = birnn(seqA), birnn(seqB)

# source worked example (t = 2): h_f = 0.60, h_b = -0.40, V = 1, b_y = 0.1
src_y2 = 1.0 * 0.60 + 1.0 * (-0.40) + 0.1

# parameter counts
def counts(d, p, q):
    one_dir = p * d + p * p + p                  # U, W, b_i of one chain
    uni = one_dir + q * p + q
    bi = 2 * one_dir + 2 * q * p + q
    return dict(d=d, p=p, q=q, one_dir=one_dir, uni=uni, bi=bi)
cnt_main, cnt_prac = counts(3, 4, 2), counts(5, 8, 3)

# ---------- numbers: ELMo weighted sum ----------
h1, h2 = [1.0, 2.0], [3.0, 0.0]
def elmo(s, gamma):
    return [gamma * (s[0] * a + s[1] * b) for a, b in zip(h1, h2)]
qa = dict(s=[0.3, 0.7], gamma=1.0, y=elmo([0.3, 0.7], 1.0))
pos = dict(s=[0.6, 0.4], gamma=1.0, y=elmo([0.6, 0.4], 1.0))
prac = dict(s=[0.5, 0.5], gamma=2.0, y=elmo([0.5, 0.5], 2.0))
w_raw = [0.0, math.log(0.7 / 0.3)]
z = sum(math.exp(w) for w in w_raw)
softmax = [math.exp(w) / z for w in w_raw]

products = dict(fwd_Wh2=Wf * A["hf"][0], bwd_Wh2_A=Wb * A["hb"][2], bwd_Wh2_B=Wb * B["hb"][2],
                Ux2_fwd=Uf * seqA[1], Ux2_bwd=Ub * seqA[1], bwd_Wh1_A=Wb * A["hb"][1], fwd_Wh3_A=Wf * A["hf"][1])
num = dict(products=products, params=dict(Wf=Wf, Uf=Uf, Wb=Wb, Ub=Ub, Vf=Vf, Vb=Vb, by=by), A=A, B=B,
           diff_t2_y=A["y"][1] - B["y"][1], src_y2=src_y2, counts_main=cnt_main, counts_practice=cnt_prac,
           elmo=dict(h1=h1, h2=h2, qa=qa, pos=pos, practice=prac, w_raw=w_raw, softmax=softmax))
print(json.dumps(num, indent=1))

def box(ax, x, y, txt, fc, w=1.15, h=0.6, fs=13, ec="#334155"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs)
def arrow(ax, a, b, color="#334155", lw=1.6, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))

BLUE, ORANGE = "#1D4ED8", "#C2410C"
# ---------- figure 1: BiRNN, two chains, sequence A ----------
fig, ax = plt.subplots(figsize=(11, 5.8))
ax.set_xlim(-1.2, 11.6); ax.set_ylim(-0.75, 5.0); ax.axis("off")
X = [2.4, 5.4, 8.4]
yx, yb, yf, yy = 0.2, 1.6, 2.85, 4.25
dx, bw, bh = 0.72, 1.3, 0.78
for k, xc in enumerate(X):
    box(ax, xc, yx, rf"$x_{k+1}={A['x'][k]:.1f}$", "#DCFCE7")
    box(ax, xc + dx, yf, rf"$\overrightarrow{{h}}_{k+1}$" + "\n" + rf"${A['hf'][k]:.4f}$", "#DBEAFE", w=bw, h=bh, ec=BLUE)
    box(ax, xc - dx, yb, rf"$\overleftarrow{{h}}_{k+1}$" + "\n" + rf"${A['hb'][k]:.4f}$", "#FFEDD5", w=bw, h=bh, ec=ORANGE)
    box(ax, xc, yy, rf"$y_{k+1}={A['y'][k]:.4f}$", "#FEF3C7", w=1.9)
    arrow(ax, (xc - 0.3, yx + 0.32), (xc - dx + 0.1, yb - bh / 2 - 0.02), color="#15803D")
    arrow(ax, (xc + 0.3, yx + 0.32), (xc + dx - 0.1, yf - bh / 2 - 0.02), color="#15803D")
    arrow(ax, (xc - dx, yb + bh / 2 + 0.02), (xc - 0.35, yy - 0.32), color=ORANGE)
    arrow(ax, (xc + dx, yf + bh / 2 + 0.02), (xc + 0.35, yy - 0.32), color=BLUE)
box(ax, 0.2, yf, r"$\overrightarrow{h}_0=0$", "#F1F5F9", w=1.3)
prev = 0.85
for xc in X:
    arrow(ax, (prev, yf), (xc + dx - bw / 2 - 0.02, yf), color=BLUE, lw=2.0); prev = xc + dx + bw / 2 + 0.02
box(ax, 10.6, yb, r"$\overleftarrow{h}_4=0$", "#F1F5F9", w=1.3)
prev = 9.95
for xc in reversed(X):
    arrow(ax, (prev, yb), (xc - dx + bw / 2 + 0.02, yb), color=ORANGE, lw=2.0); prev = xc - dx - bw / 2 - 0.02
ax.text(10.4, yf + 0.65, r"forward: $t=1\to3$", color=BLUE, fontsize=12, ha="center", va="center")
ax.text(0.2, yb - 0.65, r"backward: $t=3\to1$", color=ORANGE, fontsize=12, ha="center", va="center")
ax.text(5.4, -0.62, r"forward $\overrightarrow{W}=0.5,\ \overrightarrow{U}=1.0$;   backward $\overleftarrow{W}=0.6,\ \overleftarrow{U}=0.8$;   "
        r"$y_t=1.0\cdot\overrightarrow{h}_t+1.0\cdot\overleftarrow{h}_t+0$", ha="center", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "lec08-kp08-birnn.png", dpi=200); plt.close(fig)

# ---------- figure 2: same beginning, different ending ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
ts = [1, 2, 3]; wbar = 0.36
for ax, key, title, col in [(axs[0], "hf", r"forward state $\overrightarrow{h}_t$ (sees $x_1..x_t$)", BLUE),
                            (axs[1], "hb", r"backward state $\overleftarrow{h}_t$ (sees $x_t..x_3$)", ORANGE)]:
    for j, (S, lab, alpha, hatch) in enumerate([(A, r"A: $x_3=+1.0$", 1.0, None), (B, r"B: $x_3=-1.0$", 0.45, "//")]):
        xs_ = [t + (j - 0.5) * wbar for t in ts]
        ax.bar(xs_, S[key], width=wbar, color=col, alpha=alpha, hatch=hatch, edgecolor="#334155", label=lab)
        for xv, v in zip(xs_, S[key]):
            ax.text(xv, v + (0.04 if v >= 0 else -0.04), f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=9.5)
    ax.axhline(0, color="#334155", lw=1)
    ax.set_xticks(ts); ax.set_xticklabels([r"$t=1$", r"$t=2$", r"$t=3$"], fontsize=12)
    ax.set_title(title, fontsize=12.5); ax.set_ylim(-1.0, 1.15); ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=10.5, frameon=False)
axs[0].set_ylabel("state value", fontsize=12)
fig.suptitle(r"A $=(1.0,\,0.5,\,1.0)$ and B $=(1.0,\,0.5,\,-1.0)$ differ only in the last input", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "lec08-kp08-later-context.png", dpi=200); plt.close(fig)

# ---------- figure 3: ELMo, stacked layers and weighted sum ----------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.0), gridspec_kw=dict(width_ratios=[1.05, 1]))
axL.set_xlim(0, 6); axL.set_ylim(-0.3, 6.0); axL.axis("off")
cx = 3.5
box(axL, cx, 0.3, r"token at position $t$", "#DCFCE7", w=2.8)
box(axL, cx, 1.8, r"layer 1 BiLSTM:  $\mathbf{h}_t^{(1)}$ = [forward ; backward]", "#DBEAFE", w=4.4, fs=12)
box(axL, cx, 3.3, r"layer 2 BiLSTM:  $\mathbf{h}_t^{(2)}$ = [forward ; backward]", "#E0E7FF", w=4.4, fs=12)
box(axL, cx, 5.1, r"$\mathbf{y}_t^{\mathrm{ELMo}}=\gamma\,(s_1\mathbf{h}_t^{(1)}+s_2\mathbf{h}_t^{(2)})$", "#FEF3C7", w=4.4, fs=12.5)
arrow(axL, (cx, 0.62), (cx, 1.48)); arrow(axL, (cx, 2.12), (cx, 2.98))
axL.add_patch(FancyArrowPatch((1.25, 1.8), (1.25, 5.1), arrowstyle="-|>", mutation_scale=14, lw=1.6, color="#64748B",
                              connectionstyle="arc3,rad=-0.45"))
axL.text(0.72, 3.45, r"$s_1$", fontsize=14, color="#64748B")
arrow(axL, (cx, 3.62), (cx, 4.78), color="#64748B"); axL.text(cx + 0.15, 4.15, r"$s_2$", fontsize=14, color="#64748B")
axL.text(cx, -0.3, "(a) every layer is kept, then mixed", ha="center", fontsize=12)
# right: geometry
P1, P2 = h1, h2
axR.plot([P1[0], P2[0]], [P1[1], P2[1]], color="#94A3B8", lw=1.5, ls="--", zorder=1)
axR.scatter(*P1, s=90, color="#2563EB", zorder=3); axR.scatter(*P2, s=90, color="#7C3AED", zorder=3)
axR.annotate(r"$\mathbf{h}_t^{(1)}=(1.0,\,2.0)$", P1, xytext=(8, 8), textcoords="offset points", fontsize=12)
axR.annotate(r"$\mathbf{h}_t^{(2)}=(3.0,\,0.0)$", P2, xytext=(10, 8), textcoords="offset points", fontsize=12)
for case, col, lab, off in [(pos, "#16A34A", "POS tagging", (10, 6)), (qa, "#EA580C", "question answering", (10, 6))]:
    axR.scatter(*case["y"], s=110, marker="D", color=col, zorder=4)
    axR.annotate(rf"$s=({case['s'][0]:.1f},{case['s'][1]:.1f})$: $({case['y'][0]:.1f},\,{case['y'][1]:.1f})$" + "\n" + lab,
                 case["y"], xytext=off, textcoords="offset points", fontsize=11, color=col)
axR.set_xlim(0, 4.6); axR.set_ylim(-0.4, 2.8); axR.set_aspect("equal")
axR.set_xlabel("component 1", fontsize=12); axR.set_ylabel("component 2", fontsize=12); axR.grid(alpha=0.3)
axR.set_title(r"(b) $\gamma=1$, $s_1+s_2=1$: the result lies on the segment", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "lec08-kp08-elmo-mix.png", dpi=200); plt.close(fig)
print("figures ->", OUT)
