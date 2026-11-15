"""lec14-kp05-figures.py — 第14週 KP05 講義的所有數字與三張圖（圖內只用英文與數學符號）。
用法：python3 lec14-kp05-figures.py [輸出資料夾] [--report 數值檔.json]
預設輸出到上一層（課程 repo 放在 assets/figures/src/ 時就是 assets/figures/）；只有給 --report 才另存數值 JSON。
只需要 numpy 與 matplotlib（網路與 Adam 都用 numpy 手寫）。在一般筆電上約需兩分鐘。
內容：
1. 來源互動展示的數字：epsilon = 0.5、epsilon_theta = 0.2、beta_t = 0.2、abar_t = 0.5，算 KL（變異數取 beta_t）與簡化損失。
2. DDPM 排程下，完整 ELBO 每一步的權重 w_t = beta_t / (2 alpha_t (1 - abar_t))（sigma_t^2 = beta_t）。
3. [補充] 一維小 DDPM：資料 0.5 N(-2, 0.3^2) + 0.5 N(2, 0.3^2)，T = 1000，
   epsilon_theta 是兩層、每層 64 個神經元的多層感知器，用演算法 1（簡化損失）訓練，再用演算法 2 生成。
   另外用「完美的猜測」E[epsilon | x_t]（資料簡單所以算得出來）算出損失的下限。"""
import json, sys, time
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

# ---------- 1. 來源展示的數字 ----------
e, eh, b, ab = 0.5, 0.2, 0.2, 0.5
a = 1 - b
coef = b / (np.sqrt(a) * np.sqrt(1 - ab))                 # 平均差 = coef * (eps - eps_theta)
dmu = coef * (e - eh)
kl = dmu ** 2 / (2 * b)                                   # sigma^2 = beta, 共變異數相同時的 KL
w = b ** 2 / (2 * b * a * (1 - ab))
num["demo"] = dict(eps=e, eps_theta=eh, beta_t=b, alpha_t=r4(a), abar_t=ab, mean_coef=r4(coef), mean_diff=r4(dmu),
                   kl=r4(kl), weight=r4(w), simple_loss=r4((e - eh) ** 2), weighted_loss=r4(w * (e - eh) ** 2))
assert abs(kl - w * (e - eh) ** 2) < 1e-12

# 練習題 1：eps = -1、eps_theta = -0.4、beta = 0.1、abar_t = 0.6，sigma^2 = beta
e1, h1, b1, ab1 = -1.0, -0.4, 0.1, 0.6; a1 = 1 - b1
c1 = b1 / (np.sqrt(a1) * np.sqrt(1 - ab1)); d1 = c1 * (e1 - h1)
w1 = b1 / (2 * a1 * (1 - ab1))
num["practice1"] = dict(mean_coef=r4(c1), mean_diff=r4(d1), kl=r4(d1 ** 2 / (2 * b1)), weight=r4(w1), simple=r4((e1 - h1) ** 2), weighted=r4(w1 * (e1 - h1) ** 2))
assert abs(d1 ** 2 / (2 * b1) - w1 * (e1 - h1) ** 2) < 1e-12
# 練習題 2：演算法 2 的一步，x_t = 0.5、beta = 0.2、abar_t = 0.5、eps_theta = 0.3、z = -1、sigma^2 = beta
x2, b2, ab2, h2, z2 = 0.5, 0.2, 0.5, 0.3, -1.0; a2 = 1 - b2
mean2 = (x2 - b2 / np.sqrt(1 - ab2) * h2) / np.sqrt(a2)
num["practice2"] = dict(noise_coef=r4(b2 / np.sqrt(1 - ab2)), mean=r4(mean2), sigma=r4(np.sqrt(b2)), x_prev=r4(mean2 + np.sqrt(b2) * z2))

# ---------- 2. DDPM 排程與權重 ----------
T = 1000
beta = np.linspace(1e-4, 0.02, T); alpha = 1 - beta; abar = np.cumprod(alpha)
wt = beta / (2 * alpha * (1 - abar))
num["weights"] = {str(t): r4(wt[t - 1]) for t in (1, 2, 10, 100, 500, 1000)}
num["weights_ratio_first_to_last"] = round(float(wt[0] / wt[-1]), 1)
assert wt[0] > 0.49 and wt[-1] < 0.011

# ---------- 3. 一維小 DDPM ----------
MU, S = 2.0, 0.3


def sample_data(rng, n):
    return rng.choice([-MU, MU], n) + S * rng.normal(size=n)


def best_eps(xt, t):                                      # E[eps | x_t]，資料為兩個常態的混合時有閉式
    ab_ = abar[t - 1]
    m = np.sqrt(ab_)[None, :] * np.array([-MU, MU])[:, None]
    v = ab_ * S * S + (1 - ab_)
    lw = -(xt[None, :] - m) ** 2 / (2 * v[None, :])
    r = np.exp(lw - lw.max(0)); r /= r.sum(0)
    x0c = np.array([-MU, MU])[:, None] + (np.sqrt(ab_) * S * S / v)[None, :] * (xt[None, :] - m)
    x0h = (r * x0c).sum(0)
    return (xt - np.sqrt(ab_) * x0h) / np.sqrt(1 - ab_)


def floor_exact(t):                                     # E||eps - E[eps|x_t]||^2 = 1 - E[(E[eps|x_t])^2]，對 x_t 做數值積分
    ab_ = abar[t - 1]
    v = ab_ * S * S + (1 - ab_)
    g = np.linspace(-6, 6, 6001); dx = g[1] - g[0]
    dens = 0.5 * (np.exp(-(g + np.sqrt(ab_) * MU) ** 2 / (2 * v)) + np.exp(-(g - np.sqrt(ab_) * MU) ** 2 / (2 * v))) / np.sqrt(2 * np.pi * v)
    be = best_eps(g, np.full(g.size, t))
    return float(1 - (be ** 2 * dens).sum() * dx)


FLOOR = np.array([floor_exact(t) for t in range(1, T + 1)])

FREQ = np.array([0.5, 1, 2, 4, 8])


def feats(xt, t):
    s = t / T
    return np.concatenate([xt[:, None], s[:, None], np.sin(np.pi * FREQ[None] * s[:, None]), np.cos(np.pi * FREQ[None] * s[:, None])], 1)


H = 64
rng = np.random.default_rng(0)
D_IN = 2 + 2 * len(FREQ)
P = {"W1": rng.normal(size=(D_IN, H)) * np.sqrt(1 / D_IN), "b1": np.zeros(H),
     "W2": rng.normal(size=(H, H)) * np.sqrt(1 / H), "b2": np.zeros(H),
     "W3": rng.normal(size=(H, 1)) * np.sqrt(1 / H) * 0.1, "b3": np.zeros(1)}
silu = lambda z: z / (1 + np.exp(-z))
dsilu = lambda z: (1 / (1 + np.exp(-z))) * (1 + z * (1 - 1 / (1 + np.exp(-z))))


def forward(X):
    z1 = X @ P["W1"] + P["b1"]; h1 = silu(z1)
    z2 = h1 @ P["W2"] + P["b2"]; h2 = silu(z2)
    return (h2 @ P["W3"] + P["b3"])[:, 0], (X, z1, h1, z2, h2)


def backward(cache, g):                                   # g = dLoss/dout, shape (n,)
    X, z1, h1, z2, h2 = cache
    g = g[:, None]
    G = {"W3": h2.T @ g, "b3": g.sum(0)}
    d2 = (g @ P["W3"].T) * dsilu(z2)
    G["W2"] = h1.T @ d2; G["b2"] = d2.sum(0)
    d1 = (d2 @ P["W2"].T) * dsilu(z1)
    G["W1"] = X.T @ d1; G["b1"] = d1.sum(0)
    return G


STEPS, BATCH, LR = 12000, 1024, 3e-3
m_ = {k: np.zeros_like(v) for k, v in P.items()}; v_ = {k: np.zeros_like(v) for k, v in P.items()}
losses = []
t0 = time.perf_counter()
for it in range(1, STEPS + 1):                           # 演算法 1
    x0 = sample_data(rng, BATCH)
    t = rng.integers(1, T + 1, BATCH)
    eps = rng.normal(size=BATCH)
    xt = np.sqrt(abar[t - 1]) * x0 + np.sqrt(1 - abar[t - 1]) * eps
    out, cache = forward(feats(xt, t))
    diff = out - eps
    losses.append(float((diff ** 2).mean()))
    G = backward(cache, 2 * diff / BATCH)
    lr = LR * (0.1 ** (it / STEPS))                       # 學習率從 3e-3 指數遞減到 3e-4
    for k in P:
        m_[k] = 0.9 * m_[k] + 0.1 * G[k]; v_[k] = 0.999 * v_[k] + 0.001 * G[k] ** 2
        P[k] -= lr * (m_[k] / (1 - 0.9 ** it)) / (np.sqrt(v_[k] / (1 - 0.999 ** it)) + 1e-8)
train_sec = time.perf_counter() - t0

# 評估：簡化損失、損失下限、各步的損失
ev = np.random.default_rng(5); NE = 200000
x0 = sample_data(ev, NE); t = ev.integers(1, T + 1, NE); eps = ev.normal(size=NE)
xt = np.sqrt(abar[t - 1]) * x0 + np.sqrt(1 - abar[t - 1]) * eps
L_net = float(((forward(feats(xt, t))[0] - eps) ** 2).mean())
L_floor = float(FLOOR.mean())                           # 精確值：t 從 1 到 T 平均
L_floor_mc = float(((best_eps(xt, t) - eps) ** 2).mean())
TS = np.unique(np.concatenate([np.round(np.geomspace(1, T, 30)).astype(int), [10, 100]]))
per_net, per_floor = [], []
for tt in TS:
    x0 = sample_data(ev, 20000); e_ = ev.normal(size=20000); ts = np.full(20000, tt)
    xt_ = np.sqrt(abar[tt - 1]) * x0 + np.sqrt(1 - abar[tt - 1]) * e_
    per_net.append(float(((forward(feats(xt_, ts))[0] - e_) ** 2).mean()))
    per_floor.append(FLOOR[tt - 1])
per_net, per_floor = np.array(per_net), np.array(per_floor)
idx = {k: list(TS).index(k) for k in (1, 10, 100, 1000) if k in TS}
num["tiny_ddpm"] = dict(data="0.5 N(-2, 0.3^2) + 0.5 N(2, 0.3^2)", T=T, net="MLP 12-64-64-1, SiLU", inputs="x_t, t/T, sin/cos(pi f t/T) for f in 0.5,1,2,4,8",
                        steps=STEPS, batch=BATCH, lr="3e-3 decaying to 3e-4 (Adam)", seed=0, train_seconds=round(train_sec, 1),
                        loss_first_100=r4(np.mean(losses[:100])), loss_last_500=r4(np.mean(losses[-500:])),
                        eval_samples=NE, eval_seed=5, eval_loss_net=r4(L_net), eval_loss_floor=r4(L_floor),
                        per_t={str(k): dict(net=r4(per_net[i]), floor=r4(per_floor[i])) for k, i in idx.items()})
full_t = np.arange(1, T + 1)
m1 = (full_t >= 100) & (full_t <= 300); m2 = (full_t >= 250) & (full_t <= 500)
i_dip = int(full_t[m1][np.argmin(FLOOR[m1])]); i_peak = int(full_t[m2][np.argmax(FLOOR[m2])])
num["tiny_ddpm"]["floor_dip"] = dict(t=i_dip, floor=r4(FLOOR[i_dip - 1]))
num["tiny_ddpm"]["floor_bump"] = dict(t=i_peak, floor=r4(FLOOR[i_peak - 1]))
num["tiny_ddpm"]["eval_loss_floor_mc"] = r4(L_floor_mc)
num["tiny_ddpm"]["floor_exact_at"] = {str(k): r4(FLOOR[k - 1]) for k in (1, 10, 100, 189, 1000)}
assert L_floor - 0.005 < L_net < L_floor + 0.03 and per_floor[idx[1]] > 0.9 and per_floor[idx[1000]] < 0.01

# 演算法 2：生成
def generate(eps_fn, n, seed, var="beta"):
    g = np.random.default_rng(seed)
    x = g.normal(size=n)
    for tt in range(T, 0, -1):
        ts = np.full(n, tt)
        mean = (x - beta[tt - 1] / np.sqrt(1 - abar[tt - 1]) * eps_fn(x, ts)) / np.sqrt(alpha[tt - 1])
        if tt > 1:
            sig2 = beta[tt - 1] if var == "beta" else (1 - abar[tt - 2]) / (1 - abar[tt - 1]) * beta[tt - 1]
            x = mean + np.sqrt(sig2) * g.normal(size=n)
        else:
            x = mean
    return x


NG = 20000
gen_net = generate(lambda x, ts: forward(feats(x, ts))[0], NG, 11)
gen_net_tilde = generate(lambda x, ts: forward(feats(x, ts))[0], NG, 11, var="tilde")
gen_best = generate(best_eps, NG, 11)
def gstat(x):
    r = x[x > 0]; l = x[x < 0]
    return dict(frac_right=r4((x > 0).mean()), right_mean=r4(r.mean()), right_std=r4(r.std()), left_mean=r4(l.mean()), left_std=r4(l.std()),
                between_frac=r4((np.abs(x) < 1).mean()))
num["generation"] = dict(samples=NG, seed=11, sigma2="beta_t (also beta_tilde)", net=gstat(gen_net), net_beta_tilde=gstat(gen_net_tilde),
                         perfect=gstat(gen_best), data_between_frac=float(f"{0.5 * (1 + __import__('math').erf((-1 / S) / np.sqrt(2))):.2g}"))
G_ = num["generation"]["net"]
assert 0.45 < G_["frac_right"] < 0.55 and abs(G_["right_mean"] - 2) < 0.1 and abs(G_["right_std"] - 0.3) < 0.08

# ---------- figure 1: lec14-kp05-training ----------
fig, axs = plt.subplots(1, 2, figsize=(11.8, 3.9))
ax = axs[0]
k = 100
sm = np.convolve(losses, np.ones(k) / k, mode="valid")
ax.plot(np.arange(k, STEPS + 1), sm, color="#7C3AED", lw=1.6, label="training loss (moving avg. of 100)")
ax.axhline(L_floor, color="#059669", ls="--", lw=1.4, label=f"floor with perfect guess: {L_floor:.3f}")
ax.set_xlabel("training iteration"); ax.set_ylabel(r"$\|\epsilon-\epsilon_\theta\|^2$ (avg.)"); ax.set_ylim(0, 1.05)
ax.legend(loc="upper right", fontsize=9); ax.set_title("Algorithm 1 on 1-D data", fontsize=10.5)
ax = axs[1]
ax.plot(TS, per_net, color="#7C3AED", lw=2, label=r"trained $\epsilon_\theta$")
ax.plot(full_t, FLOOR, color="#059669", lw=1.6, ls="--", label=r"perfect guess $E[\epsilon\mid x_t]$")
ax.set_xscale("log"); ax.set_ylim(0, 1.08); ax.set_xlabel("step $t$ (log scale)"); ax.set_ylabel("loss at step $t$")
ax.legend(loc="lower left", fontsize=9); ax.set_title("which steps are hard?", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "lec14-kp05-training.png", dpi=200); plt.close(fig)

# ---------- figure 2: lec14-kp05-samples ----------
fig, ax = plt.subplots(figsize=(9.0, 3.8))
g = np.linspace(-3.5, 3.5, 600)
dens = 0.5 * (np.exp(-(g + MU) ** 2 / (2 * S * S)) + np.exp(-(g - MU) ** 2 / (2 * S * S))) / (S * np.sqrt(2 * np.pi))
ax.hist(gen_net, bins=np.linspace(-3.5, 3.5, 141), density=True, color="#C4B5FD", alpha=0.85, label=f"Algorithm 2 with trained net ({NG} samples)")
ax.plot(g, dens, color="#111827", lw=1.6, label="true data density")
ax.set_xlabel("$x_0$"); ax.set_ylabel("density"); ax.set_xlim(-3.5, 3.5); ax.set_ylim(0, 1.0)
ax.legend(loc="upper center", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "lec14-kp05-samples.png", dpi=200); plt.close(fig)

# ---------- figure 3: lec14-kp05-weights ----------
fig, ax = plt.subplots(figsize=(8.6, 3.6))
tt = np.arange(1, T + 1)
ax.plot(tt, wt, color="#F59E0B", lw=2, label=r"full ELBO weight $w_t=\frac{\beta_t}{2\alpha_t(1-\bar\alpha_t)}$")
ax.plot(tt, np.ones(T), color="#7C3AED", lw=2, ls="--", label="simplified loss: weight 1")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(3e-3, 3)
ax.set_xlabel("step $t$ (log scale)"); ax.set_ylabel("weight (log scale)")
ax.legend(loc="upper right", bbox_to_anchor=(1, 0.88), fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "lec14-kp05-weights.png", dpi=200); plt.close(fig)

if REPORT:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(num, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(num, indent=1))
