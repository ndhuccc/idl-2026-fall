"""Week 7 KP01 figures (weight initialization) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec07-kp01-signal (signal size layer by layer), lec07-kp01-jl (random projection keeps distances), lec07-kp01-pca (PCA kernels).
Every number the lecture quotes is computed here and asserted; nothing is typed by hand. Everything is synthetic and seeded.

Setup used by the lecture ([Added]: the source gives no numbers)
  * signal: a stack of 10 fully connected layers, width n = 100, 1000 random inputs ~ N(0,1), no bias, weights ~ N(0, std^2).
    We print the rms (root of the mean of squares) of the activations after layers 0, 1, 2, 5, 10 (0 = the input). The tanh stacks use inputs ~ N(0, 0.1^2) so that tanh starts in its linear range.
  * zero-init symmetry: a net with 2 inputs, 2 hidden ReLU units, 1 output, every weight 0.5; the two hidden units get identical gradients.
  * Johnson-Lindenstrauss: 50 points in d = 1000 dimensions, Gaussian random projection to k dimensions, relative error of the 1225 pairwise distances.
  * PCA initialisation: a 64x64 synthetic image, 3x3 patches (s = 3), original patches plus patches shifted by 1 pixel, PCA -> 9 eigenvectors of size 3x3.
Run:  py assets/figures/src/lec07-kp01-figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402

from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402

LAYERS, WIDTH, BATCH = 10, 100, 1000


def relu(z):
    return np.maximum(z, 0.0)


def signal(act, std, x0_std=1.0, seed=0):
    """root-mean-square (rms) of the activations after each layer (index 0 = input)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((BATCH, WIDTH)) * x0_std
    out = [float(np.sqrt((x ** 2).mean()))]
    for _ in range(LAYERS):
        x = act(x @ (rng.standard_normal((WIDTH, WIDTH)) * std))
        out.append(float(np.sqrt((x ** 2).mean())))
    return np.array(out)


def pdist(X):
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    return D[np.triu_indices(len(X), 1)]


def pca_kernels():
    rng = np.random.default_rng(7)
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.sin(xx / 9.0) + np.cos(yy / 11.0) + (xx > 30) * 1.2 + ((yy > 20) & (yy < 24)) * 1.5 + 0.15 * rng.standard_normal((n, n))
    s = 3
    P = []
    for dy, dx in ((0, 0), (1, 0), (0, 1), (1, 1)):          # original patches and patches shifted by 1 pixel
        for i in range(0, n - s - 1, s):
            for j in range(0, n - s - 1, s):
                P.append(img[i + dy:i + dy + s, j + dx:j + dx + s].ravel())
    P = np.array(P)
    P = P - P.mean(0)
    w, V = np.linalg.eigh(P.T @ P / len(P))
    idx = np.argsort(w)[::-1]
    return P.shape[0], w[idx], V[:, idx]


def numbers():
    N = {}
    sig = {}
    sig["relu_0.01"] = signal(relu, 0.01)
    sig["relu_0.1"] = signal(relu, 0.1)            # = Xavier for n_in = n_out = 100: sqrt(2/(100+100)) = 0.1
    sig["relu_he"] = signal(relu, np.sqrt(2 / WIDTH))
    sig["relu_1"] = signal(relu, 1.0)
    sig["tanh_0.01"] = signal(np.tanh, 0.01, x0_std=0.1)
    sig["tanh_xavier"] = signal(np.tanh, np.sqrt(2 / (2 * WIDTH)), x0_std=0.1)
    sig["tanh_1"] = signal(np.tanh, 1.0, x0_std=0.1)
    assert abs(np.sqrt(2 / (WIDTH + WIDTH)) - 0.1) < 1e-12 and abs(np.sqrt(2 / WIDTH) - 0.1414) < 1e-4
    r = sig
    f = lambda s_, a=WIDTH: float(np.sqrt(a / 2) * s_)                                 # theory: rms factor per ReLU layer = sqrt(n/2) * std
    for key, s_ in (("relu_0.01", 0.01), ("relu_0.1", 0.1), ("relu_he", np.sqrt(2 / WIDTH)), ("relu_1", 1.0)):
        assert abs(np.log10(r[key][10] / r[key][0]) - 10 * np.log10(f(s_))) < 0.5, key     # measured within a factor ~3 of theory after 10 layers
    assert abs(f(0.01) - 0.0707) < 1e-4 and abs(f(0.1) - 0.7071) < 1e-4 and abs(f(np.sqrt(2 / WIDTH)) - 1.0) < 1e-12 and abs(f(1.0) - 7.0711) < 1e-4
    assert r["relu_0.01"][10] < 1e-10                                                  # vanishes
    assert 0.02 < r["relu_0.1"][10] < 0.05                                             # Xavier on ReLU: shrinks about 0.71x per layer
    assert all(0.9 < v < 1.7 for v in r["relu_he"])                # He keeps the size
    assert r["relu_1"][10] > 1e7                                                       # explodes
    assert r["tanh_0.01"][10] < 1e-9 and r["tanh_1"][10] > 0.9 and 0.08 < r["tanh_xavier"][10] < 0.12   # tanh: vanishes / saturates near +-1 / Xavier keeps the size
    N["signal"] = sig
    # zero (equal) init symmetry: x = (1, 2), all weights 0.5, hidden h_j = relu(0.5*1 + 0.5*2) = 1.5, output y = 0.5*h1 + 0.5*h2 = 1.5, target t = 3
    x = np.array([1.0, 2.0]); W1 = np.full((2, 2), 0.5); w2 = np.full(2, 0.5); t = 3.0
    z = W1 @ x; h = relu(z); y = w2 @ h; dy = y - t                               # loss = (y-t)^2 / 2
    g2 = dy * h; gh = dy * w2; g1 = np.outer(gh * (z > 0), x)
    assert list(z) == [1.5, 1.5] and y == 1.5 and dy == -1.5
    assert list(g2) == [-2.25, -2.25] and np.allclose(g1[0], g1[1]) and np.allclose(g1[0], [-0.75, -1.5])
    N["sym"] = dict(z=z, y=float(y), dy=float(dy), g2=g2, g1=g1)
    # Johnson-Lindenstrauss
    rng = np.random.default_rng(1); d, npts = 1000, 50
    X = rng.standard_normal((npts, d)); D0 = pdist(X); assert len(D0) == 1225
    jl = {}
    for k in (10, 50, 100, 400):
        P = rng.standard_normal((d, k)) / np.sqrt(k); ratio = pdist(X @ P) / D0
        jl[k] = (float(ratio.min()), float(ratio.max()), float(abs(ratio - 1).max()))
    assert jl[10][2] > jl[50][2] > jl[100][2] > jl[400][2] and jl[400][2] < 0.15 and jl[10][2] > 0.5
    N["jl"] = jl
    # PCA initialisation
    npatch, w, V = pca_kernels(); assert V.shape == (9, 9) and npatch == 4 * 20 * 20
    assert np.allclose(np.linalg.norm(V, axis=0), 1) and np.allclose(V.T @ V, np.eye(9), atol=1e-9)
    ratio = w / w.sum(); assert ratio[0] > 0.5 and abs(ratio.sum() - 1) < 1e-9
    N["pca"] = dict(npatch=npatch, ratio=ratio, V=V)
    # practice answers
    xav = lambda a, b: float(np.sqrt(2 / (a + b))); he = lambda a: float(np.sqrt(2 / a))
    assert round(xav(256, 256), 4) == 0.0625 and round(he(256), 4) == 0.0884 and round(xav(784, 100), 4) == 0.0476 and round(he(784), 4) == 0.0505
    assert abs(np.sqrt(100 / 2) * 0.05 - 2 ** -1.5) < 1e-12 and abs((np.sqrt(100 / 2) * 0.05) ** 10 - 2 ** -15) < 1e-15 and round(2 ** -15, 7) == 3.05e-05 and round((1 / np.sqrt(2)) ** 10, 5) == 0.03125
    return N


def show(N):
    s = N["signal"]
    for k, v in s.items():
        print("signal", k, [float(f"{v[i]:.4g}") for i in (0, 1, 2, 5, 10)], "ratio l=10:", float(f"{v[10] / v[0]:.4g}"))
    print("sym", {a: (b.tolist() if hasattr(b, "tolist") else b) for a, b in N["sym"].items()})
    print("jl", {k: tuple(round(x, 3) for x in v) for k, v in N["jl"].items()})
    print("pca patches", N["pca"]["npatch"], "ratio", np.round(N["pca"]["ratio"], 3).tolist())


def fig_signal(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); S = numbers()["signal"]; W, H = 940, 400
    fig = plt.figure(figsize=(W / 100, H / 100)); xs = np.arange(LAYERS + 1)
    spec = (("ReLU layers: rms of the activations", (("relu_0.01", "std $0.01$ (too small)", c["warn"]), ("relu_0.1", "std $0.1$ (Xavier)", c["ref"]), ("relu_he", "std $\\sqrt{2/n}\\approx0.141$ (He)", c["ok"]), ("relu_1", "std $1$ (too large)", c["edge"]))),
            ("tanh layers: rms of the activations", (("tanh_0.01", "std $0.01$ (too small)", c["warn"]), ("tanh_xavier", "std $0.1$ (Xavier)", c["ref"]), ("tanh_1", "std $1$ (too large)", c["edge"]))))
    for p, (ttl, lines) in enumerate(spec):
        ax = fig.add_axes([(0.075 + p * 0.5), 0.14, 0.40, 0.70]); clean(ax)
        for key, lab, col in lines:
            ax.semilogy(xs, np.maximum(S[key], 1e-16), marker="o", ms=4, lw=2, color=col, label=lab)
        ax.set_ylim(1e-14, 1e17) if p == 0 else ax.set_ylim(1e-14, 10)
        ax.set_title(ttl, fontsize=12, color=c["text"]); ax.set_xlabel("layer $l$"); ax.set_xticks(range(0, 11, 2))
        ax.legend(fontsize=9, loc="upper left" if p == 0 else "lower left", frameon=False)
    finish(fig, "lec07-kp01-signal", dark, c)
    return W, H


def fig_jl(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); J = numbers()["jl"]; W, H = 760, 380
    fig, ax = axes(W, H); clean(ax)
    ks = list(J)
    ax.bar(range(len(ks)), [J[k][2] for k in ks], color=[c["warn"], c["edge"], c["ref"], c["ok"]], width=0.55)
    for i, k in enumerate(ks):
        ax.text(i, J[k][2] + 0.015, f"{J[k][2]:.2f}", ha="center", fontsize=12, color=c["text"])
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([f"$k={k}$" for k in ks]); ax.set_ylabel("largest relative error"); ax.set_ylim(0, 0.85)
    ax.set_xlabel("target dimension $k$ ($d=1000$, $50$ points)")
    finish(fig, "lec07-kp01-jl", dark, c)
    return W, H


def fig_pca(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); P = numbers()["pca"]; W, H = 900, 340
    fig = plt.figure(figsize=(W / 100, H / 100))
    for i in range(9):
        ax = fig.add_axes([0.02 + i * 0.108, 0.30, 0.095, 0.55])
        v = P["V"][:, i].reshape(3, 3); m = np.abs(v).max()
        ax.imshow(v, cmap="RdBu_r" if not dark else "coolwarm", vmin=-m, vmax=m); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(c["muted"])
        fig.text(0.02 + i * 0.108 + 0.0475, 0.14, f"#{i + 1}\n{100 * P['ratio'][i]:.1f}%", ha="center", va="center", fontsize=11, color=c["text"])
    fig.text(0.5, 0.93, "the nine $3\\times3$ eigenvectors of the patches, ordered by the share of variance", ha="center", fontsize=12, color=c["text"])
    finish(fig, "lec07-kp01-pca", dark, c)
    return W, H


if __name__ == "__main__":
    N = numbers(); show(N)
    for f in (fig_signal, fig_jl, fig_pca):
        for dark in (False, True): f(dark)
    print("figures written")
