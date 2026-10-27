"""KP09 numbers (recomputed)."""
import numpy as np
def sig(v): return 1 / (1 + np.exp(-v))
X4 = np.array([-2., -1., 1., 2.]); T4 = np.array([0., 0., 1., 1.])
def train(loss, w0=-3.0, b0=0.0, eta=0.5, max_ep=200000, track=None):
    w, b = w0, b0; hist = []; epochs = None
    for ep in range(1, max_ep + 1):
        y = sig(b + w * X4); d = (T4 - y) * (y * (1 - y) if loss == "mse" else 1.0)
        w += eta * np.mean(d * X4); b += eta * np.mean(d)
        yy = sig(b + w * X4); ok = int(((yy > .5) == (T4 > .5)).sum())
        if track is not None and ep <= track: hist.append(float(sig(b + w * 2.0)))
        if ok == 4 and epochs is None:
            epochs = ep
            if track is None: break
    return epochs, hist
def lsfit():
    x = np.array([-3., -2., -1., 1., 2., 3., 40.]); t = np.array([0, 0, 0, 1, 1, 1, 1.]); A = np.c_[x, np.ones(7)]
    a, b = np.linalg.lstsq(A, t, rcond=None)[0]; return x, t, float(a), float(b), (0.5 - b) / a
def logit_fit(iters=20000, eta=0.5):
    x = np.array([-3., -2., -1., 1., 2., 3., 40.]); t = np.array([0, 0, 0, 1, 1, 1, 1.]); w = b = 0.0
    for _ in range(iters):
        y = sig(b + w * x); d = t - y; w += eta * np.mean(d * x); b += eta * np.mean(d)
    return w, b, -b / w
def posterior(n): prec = 0.25 + n; return float(np.sqrt(1 / prec)), 1.5 * n / prec
if __name__ == "__main__":
    print(train("ce")[0], train("mse")[0])
    print({w: (train("mse", w)[0], train("ce", w)[0]) for w in (-1, -2, -3, -4, -6, -8, -10)})
    print(lsfit()[2:], logit_fit()); print([posterior(n) for n in (0, 1, 5, 20)])
    for net in (-10, -5, -2, 0, 2, 5): y = sig(net); print(net, round(float(y), 6), round(float((1 - y) * y * (1 - y)), 5), round(float(1 - y), 4))
    y = np.array([.9, .8, .3, .1]); t = np.array([1, 1, 0, 0]); p = np.where(t == 1, y, 1 - y); print(p, p.prod(), -np.log(p.prod()), -np.log(p), -np.log(p).sum())
    v = np.array([2, 1, .1]); e = np.exp(v); s = e / e.sum(); print(e, e.sum(), s, -np.log(s[0]), np.array([1, 0, 0]) - s)
    y4 = sig(-4); print(y4, (1 - y4) * y4 * (1 - y4), 1 - y4, (1 - y4) / ((1 - y4) * y4 * (1 - y4)), 0.5 * (1 - .001) ** 2, -np.log(.001), (1 - sig(-10)) / ((1 - sig(-10)) * sig(-10) * (1 - sig(-10))))
