"""lec08-kp03-figures.py — 第8週 KP03 講義的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec08-kp03-figures.py [輸出資料夾]（預設：與講義同層的 ../assets/figures/）
主例沿用 KP02 的 3 步純量例子：V=1.0、W=0.5、每步 1-h_t^2=0.8、每步 dL_t/dy_t=1.0。"""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["mathtext.fontset"] = "cm"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent  # 課程 repo：assets/figures/src/ 的上一層 assets/figures/
OUT.mkdir(parents=True, exist_ok=True)
r4 = lambda v: round(float(v), 4)

# ---------- running example (scalar, T = 3) ----------
# KP02 fixes V=1.0, W=0.5, (1-h_t^2)=0.8 and dL_t/dy_t=1.0 at every step.
# Completion [Added]: U=1.0, b_i=b_y=0, h_0=0 and inputs chosen so that h_t = sqrt(0.2) at every step;
# targets y*_t = y_t - 1 with L_t = 0.5 (y_t - y*_t)^2, so that dL_t/dy_t = 1.0.
s = math.sqrt(0.2)
P0 = dict(W=0.5, U=1.0, V=1.0, bi=0.0, by=0.0)
xs = [math.atanh(s) / P0["U"], (math.atanh(s) - P0["W"] * s) / P0["U"], (math.atanh(s) - P0["W"] * s) / P0["U"]]
targets = [P0["V"] * s - 1.0] * 3
Tn = 3

def forward(P):
    h, y = [0.0], []
    for x in xs:
        h.append(math.tanh(P["W"] * h[-1] + P["U"] * x + P["bi"]))
        y.append(P["V"] * h[-1] + P["by"])
    L = sum(0.5 * (yy - tt) ** 2 for yy, tt in zip(y, targets))
    return h, y, L

def bptt(P, e_override=None):
    h, y, L = forward(P)
    e = [y[t] - targets[t] for t in range(Tn)] if e_override is None else list(e_override)
    delta, g = [0.0] * Tn, [0.0] * Tn
    for t in reversed(range(Tn)):
        future = P["W"] * g[t + 1] if t < Tn - 1 else 0.0   # delta_t = V e_t + W g_{t+1}
        delta[t] = P["V"] * e[t] + future
        g[t] = (1 - h[t + 1] ** 2) * delta[t]               # g_t = (1-h_t^2) delta_t
    contrib = dict(V=[e[t] * h[t + 1] for t in range(Tn)], W=[g[t] * h[t] for t in range(Tn)],
                   U=[g[t] * xs[t] for t in range(Tn)], bi=list(g), by=list(e))
    G = {k: sum(v) for k, v in contrib.items()}
    return dict(h=h, y=y, L=L, e=e, delta=delta, g=g, contrib=contrib, G=G)

R = bptt(P0)
# finite-difference check of every gradient
fd = {}
for k in P0:
    eps = 1e-6
    Pp, Pm = dict(P0), dict(P0); Pp[k] += eps; Pm[k] -= eps
    fd[k] = (forward(Pp)[2] - forward(Pm)[2]) / (2 * eps)

eta = 0.1
P1 = {k: P0[k] - eta * R["G"][k] for k in P0}
L1 = forward(P1)[2]

def trajectory(eta, n=12):
    P, out = dict(P0), []
    for _ in range(n):
        r = bptt(P); out.append(r["L"])
        P = {k: P[k] - eta * r["G"][k] for k in P}
    return out
etas = [0.1, 0.3, 0.5]
traj = {e_: trajectory(e_) for e_ in etas}

# terminal-only variant (practice 3): only step 3 has an output error
RT = bptt(P0, e_override=[0.0, 0.0, 1.0])

# ---------- p = 2 Kronecker / vec example (source numbers) ----------
gk = np.array([0.5, 0.2]); hk = np.array([1.0, 0.0])
Wm = np.array([[0.3, -0.1], [0.4, 0.2]])         # any W, only for checking (h^T kron I) vec(W) = W h  [Added]
vec = lambda M: M.reshape(-1, order="F")          # stack columns
K = np.kron(hk.reshape(1, -1), np.eye(2))         # h^T kron I, size 2 x 4
check_forward = dict(K=K.tolist(), K_vecW=(K @ vec(Wm)).tolist(), W_h=(Wm @ hk).tolist())
grad_vec = K.T @ gk
grad_mat = grad_vec.reshape(2, 2, order="F")
outer = np.outer(gk, hk)

# practice numbers
pr1 = dict(W_new=0.80 - 0.1 * 0.50, W_new_eta10=0.80 - 10.0 * 0.50)
pr2 = dict(contrib=(0.5 * np.array([0.6, -0.2])).tolist())

num = dict(
    example=dict(params=P0, s=s, x=[r4(v) for v in xs], targets=[r4(v) for v in targets], h=[r4(v) for v in R["h"]],
                 y=[r4(v) for v in R["y"]], one_minus_h2=[r4(1 - v ** 2) for v in R["h"][1:]], e=R["e"], L=r4(R["L"]),
                 delta=[r4(v) for v in R["delta"]], g=[r4(v) for v in R["g"]],
                 contrib={k: [r4(v) for v in vv] for k, vv in R["contrib"].items()},
                 grad={k: r4(v) for k, v in R["G"].items()}, finite_diff={k: r4(v) for k, v in fd.items()}),
    update=dict(eta=eta, new={k: r4(v) for k, v in P1.items()}, step={k: r4(-eta * R["G"][k]) for k in P0}, L_new=r4(L1)),
    trajectories={str(k): [r4(v) for v in vv] for k, vv in traj.items()},
    terminal_only=dict(delta=[r4(v) for v in RT["delta"]], g=[r4(v) for v in RT["g"]], grad={k: r4(v) for k, v in RT["G"].items()}),
    kron=dict(g=gk.tolist(), h=hk.tolist(), W_check=Wm.tolist(), check_forward=check_forward, grad_vec=grad_vec.tolist(),
              grad_mat=grad_mat.tolist(), outer=outer.tolist(), equal=bool(np.allclose(grad_mat, outer))),
    practice=dict(q1=pr1, q2=pr2),
)
print(json.dumps(num, indent=1, ensure_ascii=False))

# ---------- figure 1: per-step contributions summed into each gradient ----------
cols = {0: "#93C5FD", 1: "#2563EB", 2: "#1E3A8A"}
names = ["V", "W", "U", "bi", "by"]
labels = [r"$\partial\mathcal{L}/\partial V$", r"$\partial\mathcal{L}/\partial W$", r"$\partial\mathcal{L}/\partial U$",
          r"$\partial\mathcal{L}/\partial b_i$", r"$\partial\mathcal{L}/\partial b_y$"]
fig, ax = plt.subplots(figsize=(10, 4.8))
for j, k in enumerate(names):
    bottom = 0.0
    for t in range(Tn):
        v = R["contrib"][k][t]
        ax.bar(j, v, bottom=bottom, width=0.55, color=cols[t], edgecolor="white", lw=1.2)
        if v > 0.12:
            ax.text(j, bottom + v / 2, f"{v:.4f}", ha="center", va="center", fontsize=10,
                    color="white" if t > 0 else "#0F172A")
        elif v == 0:
            ax.text(j + 0.31, bottom + 0.05, "t=1: 0", ha="left", va="bottom", fontsize=10, color="#64748B")
        bottom += v
    ax.text(j, bottom + 0.06, f"sum = {bottom:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=13)
ax.set_ylim(0, 3.7); ax.set_xlim(-0.6, 4.75)
ax.set_ylabel("gradient (sum of per-step terms)", fontsize=12)
ax.legend(handles=[Rectangle((0, 0), 1, 1, color=cols[t]) for t in range(Tn)], labels=["step t=1", "step t=2", "step t=3"],
          loc="upper left", fontsize=11, frameon=False)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "lec08-kp03-step-sums.png", dpi=200); plt.close(fig)

# ---------- figure 2: vec, Kronecker and outer product for p = 2 ----------
cell_cols = ["#FDE68A", "#FCA5A5", "#A7F3D0", "#C4B5FD"]   # colour of entries (1,1),(2,1),(1,2),(2,2)
def grid(ax, x0, y0, M, colors, fs=13, w=0.9):
    M = np.atleast_2d(M)
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            ax.add_patch(Rectangle((x0 + c * w, y0 - (r + 1) * w), w, w, fc=colors[r][c], ec="#334155", lw=1.2))
            ax.text(x0 + (c + 0.5) * w, y0 - (r + 0.5) * w, M[r][c], ha="center", va="center", fontsize=fs)
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.set_xlim(0, 13.2); ax.set_ylim(-0.9, 4.3); ax.axis("off")
# (a) W -> vec(W)
cW = [[cell_cols[0], cell_cols[2]], [cell_cols[1], cell_cols[3]]]
grid(ax, 0.3, 3.0, [[r"$w_{11}$", r"$w_{12}$"], [r"$w_{21}$", r"$w_{22}$"]], cW)
ax.annotate("", xy=(3.1, 2.1), xytext=(2.2, 2.1), arrowprops=dict(arrowstyle="->", lw=2, color="#64748B"))
ax.text(2.65, 2.35, "vec", ha="center", fontsize=12, color="#64748B")
grid(ax, 3.3, 3.85, [[r"$w_{11}$"], [r"$w_{21}$"], [r"$w_{12}$"], [r"$w_{22}$"]],
     [[cell_cols[0]], [cell_cols[1]], [cell_cols[2]], [cell_cols[3]]], w=0.8)
ax.text(1.9, -0.55, "(a) vec stacks the columns of $\\mathbf{W}$", ha="center", fontsize=12)
# (b) h^T kron I (2 x 4)
hv = [f"{v:.1f}" for v in hk]
Ksym = [[r"$h_1$", "0", r"$h_2$", "0"], ["0", r"$h_1$", "0", r"$h_2$"]]
Kc = [["#DBEAFE" if Ksym[r][c] != "0" else "#F8FAFC" for c in range(4)] for r in range(2)]
grid(ax, 5.2, 3.0, Ksym, Kc, w=0.8)
ax.text(6.8, 3.25, r"$\mathbf{h}_{t-1}^\top\otimes\mathbf{I}_2$  ($2\times4$)", ha="center", fontsize=12)
ax.text(6.8, 1.0, r"$[\,h_1\mathbf{I}_2\ |\ h_2\mathbf{I}_2\,]$", ha="center", fontsize=13)
ax.text(6.8, 0.35, r"example: $\mathbf{h}_{t-1}=[1.0,\ 0.0]^\top$", ha="center", fontsize=12)
ax.text(6.8, -0.55, "(b) Kronecker form: $(2\\times4)$ times vec$(\\mathbf{W})$ gives $\\mathbf{W}\\mathbf{h}_{t-1}$", ha="center", fontsize=12)
# (c) gradient folded back = outer product
ax.annotate("", xy=(10.15, 2.1), xytext=(8.75, 2.1), arrowprops=dict(arrowstyle="->", lw=2, color="#64748B"))
ax.text(9.45, 2.35, r"vec$^{-1}$", ha="center", fontsize=12, color="#64748B")
ov = [[f"{outer[r, c]:.1f}" for c in range(2)] for r in range(2)]
grid(ax, 10.4, 3.0, ov, cW)
ax.text(11.3, 3.25, r"$\mathbf{g}_t\mathbf{h}_{t-1}^\top$  ($2\times2$)", ha="center", fontsize=12)
ax.text(11.3, 0.75, r"$\mathbf{g}_t=[0.5,\ 0.2]^\top$", ha="center", fontsize=12)
ax.text(11.3, -0.55, "(c) folded back: same size as $\\mathbf{W}$", ha="center", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "lec08-kp03-vec-kron.png", dpi=200); plt.close(fig)

# ---------- figure 3: learning rate decides whether the loss goes down ----------
fig, ax = plt.subplots(figsize=(8.5, 4.6))
mk = {0.1: ("#16A34A", "o"), 0.3: ("#2563EB", "s"), 0.5: ("#DC2626", "^")}
for e_ in etas:
    c, m = mk[e_]
    ax.plot(range(len(traj[e_])), traj[e_], color=c, marker=m, lw=1.8, ms=6, label=rf"$\eta={e_}$")
ax.set_yscale("log"); ax.set_ylim(1e-4, 1e4)
ax.set_xlabel("iteration (one full forward pass + BPTT + update)", fontsize=12)
ax.set_ylabel(r"total loss $\mathcal{L}$ (log scale)", fontsize=12)
ax.annotate(rf"start $\mathcal{{L}}={traj[0.1][0]:.1f}$", xy=(0, traj[0.1][0]), xytext=(0.6, 30), fontsize=11,
            arrowprops=dict(arrowstyle="->", color="#334155"))
ax.annotate(rf"after 1 step: {traj[0.1][1]:.4f}", xy=(1, traj[0.1][1]), xytext=(2.0, 0.6), fontsize=11, color="#16A34A",
            arrowprops=dict(arrowstyle="->", color="#16A34A"))
ax.legend(loc="upper left", fontsize=11, frameon=False)
ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(OUT / "lec08-kp03-learning-rate.png", dpi=200); plt.close(fig)
print("figures ->", OUT)
