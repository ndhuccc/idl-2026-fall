"""KP08 numbers (recomputed): J = 1/2 (w1^2 + 20 w2^2), start (-4, 1), mu = 0.08."""
import numpy as np
W0 = np.array([-4.0, 1.0]); A = np.array([1.0, 20.0]); MU = 0.08; TOL = 1e-3
def J(w): return 0.5 * float(np.sum(A * w * w))
def grad(w): return A * w
def plain(mu=MU, tol=TOL, maxn=3000):
    w = W0.copy(); path = [w.copy()]
    for t in range(maxn):
        if np.linalg.norm(w) < tol: return t, np.array(path)
        w = w - mu * grad(w); path.append(w.copy())
        if np.linalg.norm(w) > 1e12: break
    return (len(path) - 1 if np.linalg.norm(w) < tol else None), np.array(path)
def momentum(alpha, mu=MU, tol=TOL, maxn=3000, full=False):
    w = W0.copy(); dw = np.zeros(2); path = [w.copy()]; ds = []
    for t in range(maxn):
        if np.linalg.norm(w) < tol: return t, np.array(path), ds
        g = grad(w); dw = alpha * dw - mu * g; w = w + dw; path.append(w.copy()); ds.append((g, dw.copy()))
    return None, np.array(path), ds
def adaptive(mu0, ri=1.05, rd=0.7, c=1.04, tol=TOL, maxn=3000):
    w = W0.copy(); mu = mu0; Jp = J(w); path = [w.copy()]; mus = [mu]; rej = 0; n = 0
    while np.linalg.norm(w) >= tol and n < maxn:
        wn = w - mu * grad(w); Jn = J(wn); n += 1
        if Jn / Jp > c: mu *= rd; rej += 1; path.append(w.copy()); mus.append(mu); continue   # rejected step: keep w
        mu = mu * ri if Jn / Jp < 1 else mu; w = wn; Jp = Jn; path.append(w.copy()); mus.append(mu)
    return n, rej, np.array(path), mus
def quickprop(mu=MU, amax=1.75, tol=TOL, maxn=200):
    w = W0.copy(); g0 = grad(w); dw = -mu * g0; w = w + dw; path = [w.copy()]; gp = g0
    for t in range(maxn):
        if np.linalg.norm(w) < tol: return len(path), np.array(path)
        g = grad(w); den = gp - g
        a = np.where(np.abs(den) > 1e-12, g / np.where(np.abs(den) > 1e-12, den, 1), 0.0); a = np.minimum(a, amax)
        dw = a * dw; gp = g; w = w + dw; path.append(w.copy())
    return None, np.array(path)
def blowup(mu=0.12, n=60):
    w = W0.copy()
    for _ in range(n): w = w - mu * grad(w)
    return float(np.linalg.norm(w))
if __name__ == "__main__":
    print("plain", plain()[0], "plain .02", plain(0.02)[0], "plain .12", plain(0.12)[0], "blow60", "%.3g" % blowup())
    for a in (0, .3, .5, .6, .7, .8, .9, .95): print("alpha", a, momentum(a)[0])
    n, p, ds = momentum(0.6)
    for t in range(3): print(t + 1, ds[t][0], ds[t][1], p[t + 1])
    for m in (0.02, 0.12, 0.08):
        n, rej, p, mus = adaptive(m); print("adaptive", m, n, rej, len(p) - 1)
    print("quick", quickprop()[0], quickprop()[1][:4])
    # fixed gradient weights
    for a in (.6, .8):
        d = 0; 
        for _ in range(80): d = a * d - 1
        print("fixed", a, -d, 1 / (1 - a))
    # ex: practice
    g = np.array([-3.1936, -12.0]); d3 = 0.6 * np.array([0.4864, 0]) - 0.08 * g; print(d3, np.array([-3.1936, -0.6]) + d3)
    print([0.6 ** t for t in range(6)], [0.8 ** t for t in range(6)])
