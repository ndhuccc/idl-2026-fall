"""Builds kp25_cache.json: error curves per round for the figures (uses kp25_nums.py; same settings as the lecture)."""
import json, numpy as np
from kp25_nums import gauss20, adaboost, alpha
X, y, _ = gauss20(200, 1); Xt, yt, _ = gauss20(400, 2)
stump = [float((np.sign(Xt[:, j]) != yt).mean()) for j in range(20)]
rounds = list(range(1, 2001))
out, als, Ps = adaboost(X, y, 2000, 0, rounds, Xt, yt, stumps="best")
tr = [o[1] for o in out]; te = [o[2] for o in out]
cs = np.cumsum(als); minm = []; frac = []
for m, e1, e2, F, w in out:
    mg = y * F / cs[m - 1]; minm.append(float(mg.min())); frac.append(float((mg < 0.1).mean()))
noise = {}
for flip in (0.0, 0.15):
    acc = np.zeros(2000)
    for s in range(5):
        Xn, yl, y0 = gauss20(200, 10 + s, flip=flip); Xtn, ytl, yt0 = gauss20(400, 50 + s)
        o, al, Pp = adaboost(Xn, yl, 2000, s, rounds, Xtn, yt0, stumps="best")
        acc += np.array([e[2] for e in o])
    noise[str(flip)] = (acc / 5).tolist()
json.dump({"stump": stump, "train": tr, "test": te, "minm": minm, "frac": frac, "noise": noise, "alpha": [float(a) for a in als[:5]]}, open("kp25_cache.json", "w"))
print("done")
