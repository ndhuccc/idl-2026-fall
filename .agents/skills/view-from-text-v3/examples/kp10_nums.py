"""Numbers on the KP10 pages (all recomputed)."""
import numpy as np
from math import comb
def M(K, d): return sum(comb(K, m) for m in range(d + 1))
def params(d, k, o=1): return (d + 1) * k + (k + 1) * o
TRAIN = [0.50, 0.30, 0.20, 0.15, 0.10]; VAL = [0.55, 0.35, 0.25, 0.28, 0.40]
def best(val): return int(np.argmin(val)) + 1
def sine_demo(seed=7, n=10, deg=9):
    rng = np.random.default_rng(seed); x = np.linspace(0.04, 0.96, n); y = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, n)
    c = np.polyfit(x, y, deg); return x, y, c
def R(K, d):
    if K == 0 or d == 0: return 1
    return R(K - 1, d) + R(K - 1, d - 1)
if __name__ == "__main__":
    print("table", [[M(K, d) for K in (1, 2, 3, 4)] for d in (1, 2, 3)])
    print("R == M", all(R(K, d) == M(K, d) for K in range(8) for d in range(6)))
    print("params 2-3-1", params(2, 3), "hidden", (2 + 1) * 3, "out", (3 + 1) * 1)
    print("best", best(VAL), "rise", round(VAL[4] - VAL[2], 2), "min val", min(VAL), "train min epoch", best(TRAIN))
    print("ex1", best([.45, .25, .30]), round(.30 - .25, 2), "ex2", M(3, 2), "mse", (1 ** 2 + (-1) ** 2) / 2)
    print("3k+1", [(k, 3 * k + 1) for k in (3, 5, 15)])
    print("err", 1 / 10, 2 / 10)
    x, y, c = sine_demo(); print(np.abs(np.polyval(c, np.linspace(0, 1, 200))).max())
