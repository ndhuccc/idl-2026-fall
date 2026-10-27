"""KP06 numbers (recomputed): 2-2-1 network, x = (1, 0), t = 1, a = 1, eta = 0.3."""
import math
f = lambda v, a=1.0: 1 / (1 + math.exp(-a * v))
fp = lambda v, a=1.0: a * f(v, a) * (1 - f(v, a))
W = {"h1": [-0.10, 0.20, 0.40], "h2": [0.10, -0.30, 0.20], "out": [-0.20, 0.50, -0.40]}
X = [1, 0]; T = 1; ETA = 0.3
def fwd(W_):
    v1 = W_["h1"][0] + W_["h1"][1] * X[0] + W_["h1"][2] * X[1]; v2 = W_["h2"][0] + W_["h2"][1] * X[0] + W_["h2"][2] * X[1]
    y1, y2 = f(v1), f(v2); vo = W_["out"][0] + W_["out"][1] * y1 + W_["out"][2] * y2; o = f(vo)
    return dict(v1=v1, v2=v2, y1=y1, y2=y2, vo=vo, o=o)
def E(W_, t=T): return 0.5 * (t - fwd(W_)["o"]) ** 2
def grads(W_, t=T):
    g = fwd(W_); e = t - g["o"]; do = e * g["o"] * (1 - g["o"])
    d1 = g["y1"] * (1 - g["y1"]) * do * W_["out"][1]; d2 = g["y2"] * (1 - g["y2"]) * do * W_["out"][2]
    g.update(e=e, do=do, d1=d1, d2=d2, fp1=g["y1"] * (1 - g["y1"]), fp2=g["y2"] * (1 - g["y2"]), fpo=g["o"] * (1 - g["o"])); return g
def inputs(g): return {"out": [1, g["y1"], g["y2"]], "h1": [1, X[0], X[1]], "h2": [1, X[0], X[1]]}
def deltas(g): return {"out": g["do"], "h1": g["d1"], "h2": g["d2"]}
def update(W_, t=T):
    g = grads(W_, t); I = inputs(g); D = deltas(g); return {k: [w + ETA * D[k] * i for w, i in zip(W_[k], I[k])] for k in W_}, g
def numeric(W_, k, i, h=1e-6):
    Wp = {a: b[:] for a, b in W_.items()}; Wm = {a: b[:] for a, b in W_.items()}; Wp[k][i] += h; Wm[k][i] -= h; return (E(Wp) - E(Wm)) / (2 * h)
if __name__ == "__main__":
    g = grads(W); print({k: round(v, 5) for k, v in g.items()}, "E", E(W))
    W2, _ = update(W); print(W2, "E after", E(W2), "o after", fwd(W2)["o"])
    I, D = inputs(g), deltas(g); mx = 0
    for k in W:
        for i in range(3):
            an = -D[k] * I[k][i]; nu = numeric(W, k, i); mx = max(mx, abs(an - nu)); print(k, i, "grad", round(an, 4), "dw", round(ETA * D[k] * I[k][i], 4), "num", round(nu, 6))
    print("max diff", mx)
    print("chain", -g["e"] * g["fpo"] * g["y1"], g["e"], g["fpo"], g["y1"])
    print("fp(3)", [fp(3, a) for a in (0.7, 2, 8)], "peaks", [a / 4 for a in (0.7, 2, 8)])
    g0 = grads(W, 0); print("t=0 e", g0["e"], "do", g0["do"], [ETA * g0["do"] * i for i in inputs(g0)["out"]])
