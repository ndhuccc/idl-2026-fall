"""lec14-kp07-figures.py — 第14週 KP07 講義的所有數字與兩張圖（圖內只用英文與數學符號）。
用法：python3 lec14-kp07-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
內容：
1. [補充] 分數與雜訊的關係：兩堆資料的 x_t 分布 q_t 的分數 d/dx log q_t(x)，等於 -E[eps | x_t] / sqrt(1 - abar_t)。
2. [補充] 一維條件生成：類別 c = right 是 N(2, 0.3^2)、c = left 是 N(-2, 0.3^2)，無條件是兩者各半。
   用完美的雜訊猜測（有條件 eps_c、無條件 eps_u 都算得出來），以引導強度 gamma 組合
   eps_tilde = eps_u + gamma (eps_c - eps_u)，DDPM 排程（T=1000、sigma^2 = beta）跑演算法 2。
3. [補充] ILVR 的低頻替換：長度 4 的手算例子（下採樣 2 倍）與一條長度 64 的訊號示意圖。
4. [補充] cross-attention 的小例子：一個影像位置對三個文字詞元。"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"
args = sys.argv[1:]
REPORT = None
if "--report" in args:
    i = args.index("--report"); REPORT = Path(args[i + 1]); del args[i:i + 2]
OUT = Path(args[0]) if args else Path(__file__).resolve().parent.parent
OUT.mkdir(parents=True, exist_ok=True)
num = {}
r4 = lambda v: round(float(v), 4)

T = 1000
beta = np.linspace(1e-4, 0.02, T); alpha = 1 - beta; abar = np.cumprod(alpha)
MU, S = 2.0, 0.3


def eps_mix(x, ab, w_right=0.5, MU=MU, S=S):
    """E[eps | x_t] for data w_left N(-MU,S^2) + w_right N(MU,S^2)；w_right = 1 或 0 就是單一類別。"""
    ws = np.array([1 - w_right, w_right])
    m = np.sqrt(ab) * np.array([-MU, MU])[:, None]
    v = ab * S * S + (1 - ab)
    with np.errstate(divide="ignore"):
        lw = np.log(ws)[:, None] - (x[None, :] - m) ** 2 / (2 * v)
    r = np.exp(lw - lw.max(0)); r /= r.sum(0)
    x0c = np.array([-MU, MU])[:, None] + (np.sqrt(ab) * S * S / v) * (x[None, :] - m)
    return (x - np.sqrt(ab) * (r * x0c).sum(0)) / np.sqrt(1 - ab)


def score_mix(x, ab):                                   # d/dx log q_t(x) 直接微分混合常態的密度
    v = ab * S * S + (1 - ab)
    comps = [np.exp(-(x - s_ * np.sqrt(ab) * MU) ** 2 / (2 * v)) for s_ in (-1, 1)]
    dens = 0.5 * (comps[0] + comps[1])
    ddens = 0.5 * sum(-(x - s_ * np.sqrt(ab) * MU) / v * c for s_, c in zip((-1, 1), comps))
    return ddens / dens


# ---------- 1. 分數與雜訊 ----------
xs = np.array([-1.0, 0.0, 0.5, 1.5])
chk = {}
for t in (100, 300):
    ab = abar[t - 1]
    sc = score_mix(xs, ab); ep = -eps_mix(xs, ab) / np.sqrt(1 - ab)
    chk[str(t)] = dict(x=xs.tolist(), score=[r4(v) for v in sc], minus_eps_over_std=[r4(v) for v in ep])
    assert np.allclose(sc, ep, atol=1e-8)
num["score_eps_check"] = chk

# ---------- 2. 條件生成與引導（兩類互相重疊：N(-1, 0.8^2) 與 N(1, 0.8^2)） ----------
MC, SC = 1.0, 0.8
def generate(gamma, n=20000, seed=4, cond="right"):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    for t in range(T, 0, -1):
        ab = abar[t - 1]
        eu = eps_mix(x, ab, 0.5, MC, SC)
        if cond is None:
            e = eu
        else:
            ec = eps_mix(x, ab, 1.0 if cond == "right" else 0.0, MC, SC)
            e = eu + gamma * (ec - eu)
        mean = (x - beta[t - 1] / np.sqrt(1 - ab) * e) / np.sqrt(alpha[t - 1])
        x = mean + (np.sqrt(beta[t - 1]) * rng.normal(size=n) if t > 1 else 0.0)
    return x


GAMMAS = (0.0, 1.0, 3.0, 7.0)
gen = {"uncond": generate(0.0, cond=None)}
for g in GAMMAS:
    gen[str(g)] = generate(g)
st = lambda x: dict(frac_right=r4((x > 0).mean()), mean=r4(x.mean()), std=r4(x.std()))
num["guidance"] = dict(classes="right: N(1, 0.8^2), left: N(-1, 0.8^2); unconditional = half each", sampler="DDPM, T=1000, sigma^2=beta",
                       eps="perfect guesses", formula="eps_u + gamma (eps_c - eps_u)", samples=20000, seed=4,
                       results={k: st(v) for k, v in gen.items()})
G = num["guidance"]["results"]
from math import erf
num["guidance"]["class_frac_positive_true"] = r4(0.5 * (1 + erf(MC / (SC * np.sqrt(2)))))
assert 0.47 < G["uncond"]["frac_right"] < 0.53 and abs(G["1.0"]["frac_right"] - num["guidance"]["class_frac_positive_true"]) < 0.01
assert abs(G["1.0"]["mean"] - MC) < 0.03 and abs(G["1.0"]["std"] - SC) < 0.03
assert G["7.0"]["std"] < G["3.0"]["std"] < G["1.0"]["std"] and G["7.0"]["frac_right"] > G["3.0"]["frac_right"] > G["1.0"]["frac_right"]
assert 0.45 < G["0.0"]["frac_right"] < 0.55

# ---------- 3. ILVR ----------
def lowpass(v, n=2):
    return np.repeat(v.reshape(-1, n).mean(1), n)


xh = np.array([3.0, 1.0, 0.0, 2.0]); yh = np.array([0.0, 2.0, 4.0, 4.0])
res_h = lowpass(yh) + xh - lowpass(xh)
num["ilvr_hand"] = dict(x=xh.tolist(), y=yh.tolist(), lowpass_x=lowpass(xh).tolist(), lowpass_y=lowpass(yh).tolist(),
                        result=res_h.tolist(), lowpass_result=lowpass(res_h).tolist())
assert np.allclose(lowpass(res_h), lowpass(yh))
rng = np.random.default_rng(12)
L = 64; idx = np.arange(L)
x_sig = np.cumsum(rng.normal(size=L)) * 0.35; x_sig -= x_sig.mean()
x_sig += 0.5 * rng.normal(size=L)
y_sig = 1.5 * np.sin(2 * np.pi * idx / L) + 0.4 * np.sin(2 * np.pi * 9 * idx / L)
N_DS = 8
out_sig = lowpass(y_sig, N_DS) + x_sig - lowpass(x_sig, N_DS)
num["ilvr_signal"] = dict(length=L, downsample=N_DS, seed=12,
                          max_abs_lowpass_diff_to_y=float(f"{np.abs(lowpass(out_sig, N_DS) - lowpass(y_sig, N_DS)).max():.1e}"),
                          max_abs_highpass_diff_to_x=float(f"{np.abs((out_sig - lowpass(out_sig, N_DS)) - (x_sig - lowpass(x_sig, N_DS))).max():.1e}"))

xp = np.array([1.0, 3.0, 2.0, 2.0]); yp = np.array([4.0, 0.0, 1.0, 1.0])
rp = lowpass(yp) + xp - lowpass(xp)
num["practice1"] = dict(lowpass_x=lowpass(xp).tolist(), lowpass_y=lowpass(yp).tolist(), result=rp.tolist())
assert rp.tolist() == [1.0, 3.0, 1.0, 1.0]
num["practice2"] = {str(gm): r4(0.4 + gm * (0.1 - 0.4)) for gm in (0, 1, 3)}

# ---------- 4. cross-attention ----------
q = np.array([1.0, 0.0])
K = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
V = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
sc_ = K @ q / np.sqrt(2)
w_ = np.exp(sc_) / np.exp(sc_).sum()
num["cross_attention"] = dict(q=q.tolist(), K=K.tolist(), V=V.tolist(), scores=[r4(v) for v in sc_], weights=[r4(v) for v in w_], output=[r4(v) for v in w_ @ V])

# ---------- figure 1: lec14-kp07-guidance ----------
fig, axs = plt.subplots(1, 4, figsize=(13.0, 3.5), sharey=True)
g = np.linspace(-3.5, 3.5, 500)
dens_r = np.exp(-(g - MC) ** 2 / (2 * SC * SC)) / (SC * np.sqrt(2 * np.pi))
panels = [("uncond", "unconditional"), ("1.0", r"conditional, $\gamma=1$"), ("3.0", r"guided, $\gamma=3$"), ("7.0", r"guided, $\gamma=7$")]
bins = np.linspace(-3.5, 3.5, 141)
for ax, (k, title) in zip(axs, panels):
    ax.hist(gen[k], bins=bins, density=True, color="#C4B5FD", alpha=0.9)
    ax.plot(g, dens_r, color="#111827", lw=1.2, ls="--", label=r"class $c$: $\mathcal{N}(1,0.8^2)$")
    r = G[k]
    ax.set_title(title, fontsize=10.5)
    ax.text(-3.3, 1.02, f"$x_0>0$: {r['frac_right'] * 100:.1f}%\nmean {r['mean']:.2f}, std {r['std']:.2f}", fontsize=9, va="top")
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(0, 1.1); ax.set_xlabel("$x_0$")
axs[0].set_ylabel("density"); axs[1].legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0.0, 0.8))
fig.tight_layout(); fig.savefig(OUT / "lec14-kp07-guidance.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec14-kp07-ilvr ----------
fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.4), sharey=True)
for ax, (sig, col, title) in zip(axs, [(x_sig, "#7C3AED", "generated $x$"), (y_sig, "#059669", "reference $y$"),
                                      (out_sig, "#DC2626", r"$\phi_N(y)+x-\phi_N(x)$")]):
    ax.plot(idx, sig, color=col, lw=1.4, label="signal")
    ax.step(idx, lowpass(sig, N_DS), where="mid", color="#111827", lw=1.6, ls="--", label=r"low-pass $\phi_N$")
    ax.set_title(title, fontsize=10.5); ax.set_xlabel("position")
axs[0].legend(loc="lower left", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "lec14-kp07-ilvr.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
