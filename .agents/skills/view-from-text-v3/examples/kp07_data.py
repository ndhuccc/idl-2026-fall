"""Curves and tables on the KP07 pages (recomputed from kp07_nums)."""
import os, pickle
import numpy as np
import kp07_nums as N
_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kp07_cache.pkl")
def data():
    if os.path.exists(_C): return pickle.load(open(_C, "rb"))
    d = {}
    W, h, e = N.train_batch(N.W0, 1.5); d["main"] = (h, e, W)
    d["zero"] = N.train_batch(np.zeros((3, 3)), 1.5, maxE=3707)[1]
    A = np.random.RandomState(0).uniform(-3, 3, (3, 3)); B = np.random.RandomState(4).uniform(-3, 3, (3, 3))
    d["A"] = N.train_batch(A, 3.0, maxE=20000)[1]; d["B"] = N.train_batch(B, 3.0, maxE=20000)[1:3]
    Wa = N.train_batch(A, 3.0, maxE=20000)[0]; d["A_out"] = [float(o) for o in N.evalJ(Wa)[2]]
    G = sum(N.grads(Wa, x, y)[0] for x, y in zip(N.X, N.Y)) / 4; d["A_grad"] = float(np.linalg.norm(G))
    W2, h2, e2 = N.train_batch(Wa, 3.0, maxE=60000); d["A_more"] = h2[-1]
    d["eta"] = {eta: N.vtrain(N.W0[None], eta, 80000)[1][0] for eta in (0.1, 0.5, 1.5, 3, 6, 30, 100, 200)}
    Wr = np.stack([np.random.RandomState(s).uniform(-3, 3, (3, 3)) for s in range(300)]); _, dd, _ = N.vtrain(Wr, 3, 20000); d["seeds300"] = int((dd > 0).sum())
    Ws = np.stack([np.random.RandomState(s).uniform(-0.5, 0.5, (3, 3)) for s in range(20)]); _, d20, _ = N.vtrain(Ws, 1.5, 40000); d["small20"] = (int(d20.min()), int(d20.max()), int((d20 > 0).sum()))
    d["pat"] = N.train_pattern(N.W0, 1.5)[2]; d["batch6"] = N.train_batch(N.W0, 6.0)[2]
    d["patA"] = N.train_pattern(A, 1.5, maxE=20000)[1][-1]
    pickle.dump(d, open(_C, "wb")); return d
def epoch1():
    rows = []; G = np.zeros((3, 3))
    for x, y in zip(N.X, N.Y):
        g, do, dh, h, o = N.grads(N.W0, x, y); G += g; rows.append((x, y, h, o, do, dh))
    dW = 1.5 * G / 4; J0 = N.evalJ(N.W0)[0]; W1, _ = N.batch_epoch(N.W0, 1.5); return rows, dW, J0, N.evalJ(W1)[0]
if __name__ == "__main__":
    d = data(); print({k: (v if not hasattr(v, "__len__") or len(v) < 10 else len(v)) for k, v in d.items()})
