"""Numbers on the KP11 pages (all recomputed)."""
import numpy as np
from math import sqrt
def sal(H, w): return 0.5 * H * w * w
def rho(w, tau=1.0): return w * w / (tau * tau + w * w)
def drho(w, tau=1.0): return 2 * tau * tau * w / (tau * tau + w * w) ** 2
def total(data, P, a): return data + a * P
CENTERS_A = [(0.4, 0.9), (2.0, 1.8), (2.3, 2.3), (2.6, 1.8)]
CENTERS_B = [(1.5, 1.0), (1.9, 1.0), (1.5, 3.0), (3.3, 2.6)]
def data(seed=20261025):
    rng = np.random.default_rng(seed); out = []
    for lab, cs in ((0, CENTERS_A), (1, CENTERS_B)):
        for c in cs: out.append((lab, rng.normal(c, sqrt(0.08), (50, 2))))
    return out
def cascade_S(v, e):
    v = np.array(v, float); e = np.array(e, float); return abs(((v - v.mean()) * (e - e.mean())).sum())
def comp(d2):  # H=[[4,1],[1,2]], delta w1=-0.5
    return 0.5 * 4 * 0.25 + 1 * (-0.5) * d2 + 0.5 * 2 * d2 * d2
def logistic(r): return 1 / (1 + np.exp(-r))
if __name__ == "__main__":
    print("sA", sal(100, .2), "sB", sal(.02, 1), "ratio", sal(100, .2) / sal(.02, 1))
    print("J A", total(.10, 4, .1), "J B", total(.20, 1, .1), "alpha0 pick", .10 < .20)
    print("rho", [round(rho(w), 4) for w in (.2, 1, 3)], "drho(0)", drho(0), "drho peak at tau/sqrt3", 1 / sqrt(3), drho(1 / sqrt(3)))
    print("A cubed", 2 ** 3)
    print("S", cascade_S([-1, 1], [-1, 1]), cascade_S([1, 1], [-1, 1]))
    print("counts", 25 / 480, 480 - 25, (480 - 25) / 480, 8 * 50, sqrt(.08), "params", 9 + 8 + 3)
    print("comp", comp(0), comp(.25), "min check", min((comp(d), d) for d in np.linspace(-1, 1, 2001)))
    print("C", sal(8, .5), "D", sal(.1, 2), "rho(2),tau=2", rho(2, 2))
    print("fd", (0.01 - 0 + 0.01) / 0.01)
    for lab, X in data(): print(lab, X.mean(0).round(2), X.std(0).round(3))
