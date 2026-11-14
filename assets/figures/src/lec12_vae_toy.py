"""lec12_vae_toy.py — 第12週 KP05、KP06 共用的小 VAE（只用 numpy，手寫反向傳播）。
資料沿用 KP01：300 個點落在單位圓的兩段圓弧上（18°–108° 與 180°–270°），亂數種子 1。
編碼器：2 → 32（tanh）→ (mu, logvar)，潛在維度 1；解碼器：1 → 32（tanh）→ 2。
損失（每個點的負 ELBO，只差一個常數）：||x - x_hat||^2 / (2 sigma_x^2) + KL(N(mu, sd^2) || N(0, 1))，sigma_x = 0.1。"""
import numpy as np

SIGMA_X2 = 0.01
H = 32


def two_arcs(seed=1):
    rng = np.random.default_rng(seed)
    a1 = rng.uniform(0.10 * np.pi, 0.60 * np.pi, 150); a2 = rng.uniform(1.00 * np.pi, 1.50 * np.pi, 150)
    ang = np.r_[a1, a2]
    return np.c_[np.cos(ang), np.sin(ang)] + rng.normal(0, 0.02, (300, 2))


def init(seed=2):
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(0, 0.5, (2, H)), "b1": np.zeros(H), "Wm": rng.normal(0, 0.5, (H, 1)), "bm": np.zeros(1),
            "Wv": rng.normal(0, 0.1, (H, 1)), "bv": np.zeros(1),
            "W3": rng.normal(0, 0.5, (1, H)), "b3": np.zeros(H), "W4": rng.normal(0, 0.5, (H, 2)), "b4": np.zeros(2)}


def encode(X, P):
    h = np.tanh(X @ P["W1"] + P["b1"])
    return h, h @ P["Wm"] + P["bm"], h @ P["Wv"] + P["bv"]


def decode(Z, P):
    g = np.tanh(Z @ P["W3"] + P["b3"])
    return g, g @ P["W4"] + P["b4"]


def loss_and_grads(X, P, eps, beta=1.0):
    """Mean over the batch of recon + beta*KL, with z = mu + sd*eps (reparameterization)."""
    n = len(X)
    h, mu, lv = encode(X, P); sd = np.exp(0.5 * lv)
    z = mu + sd * eps
    g, xh = decode(z, P)
    recon = np.sum((X - xh) ** 2, 1, keepdims=True) / (2 * SIGMA_X2)
    kl = 0.5 * (sd ** 2 + mu ** 2 - 1 - lv)
    L = float(np.mean(recon + beta * kl))
    dxh = (xh - X) / SIGMA_X2 / n
    G = {"W4": g.T @ dxh, "b4": dxh.sum(0)}
    dg = dxh @ P["W4"].T * (1 - g ** 2)
    G["W3"] = z.T @ dg; G["b3"] = dg.sum(0)
    dz = dg @ P["W3"].T
    dmu = dz + beta * mu / n
    dlv = dz * eps * 0.5 * sd + beta * 0.5 * (sd ** 2 - 1) / n
    G["Wm"] = h.T @ dmu; G["bm"] = dmu.sum(0); G["Wv"] = h.T @ dlv; G["bv"] = dlv.sum(0)
    dh = (dmu @ P["Wm"].T + dlv @ P["Wv"].T) * (1 - h ** 2)
    G["W1"] = X.T @ dh; G["b1"] = dh.sum(0)
    return L, float(np.mean(recon)), float(np.mean(kl)), G


def grad_check(seed=3):
    X = two_arcs()[:20]; P = init(); eps = np.random.default_rng(seed).normal(size=(20, 1))
    _, _, _, G = loss_and_grads(X, P, eps)
    worst = 0.0
    for k in ("W1", "Wv", "W3", "bm"):
        idx = tuple(0 for _ in P[k].shape)
        old = P[k][idx]; P[k][idx] = old + 1e-6; lp = loss_and_grads(X, P, eps)[0]
        P[k][idx] = old - 1e-6; lm = loss_and_grads(X, P, eps)[0]; P[k][idx] = old
        num = (lp - lm) / 2e-6
        worst = max(worst, abs(num - G[k][idx]) / max(1e-8, abs(num) + abs(G[k][idx])))
    return worst


def train(steps=20000, beta=1.0, seed=2, lr=3e-3, log_every=500):
    X = two_arcs(); P = init(seed); rng = np.random.default_rng(seed + 100)
    m = {k: np.zeros_like(v) for k, v in P.items()}; v2 = {k: np.zeros_like(v) for k, v in P.items()}
    hist = []
    for it in range(1, steps + 1):
        eps = rng.normal(size=(len(X), 1))
        L, R, K, G = loss_and_grads(X, P, eps, beta)
        for k in P:
            m[k] = 0.9 * m[k] + 0.1 * G[k]; v2[k] = 0.999 * v2[k] + 0.001 * G[k] ** 2
            P[k] -= lr * (m[k] / (1 - 0.9 ** it)) / (np.sqrt(v2[k] / (1 - 0.999 ** it)) + 1e-8)
        if it == 1 or it % log_every == 0:
            hist.append((it, L, R, K))
    return X, P, hist
