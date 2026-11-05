"""Week 7 KP07 figures (DenseNet, EfficientNet, ConvNeXt and the chapter summary) in two skins from the same data.
    assets/figures/<id>.png (light, lectures)    assets/figures/<id>.svg (dark, deck)
Ids: lec07-kp07-dense (connections of a dense block, l = 4), lec07-kp07-growth (input channels of each layer for k0 = 16, k = 12).
Source (CCC-LNS IDL_115 Ch.4 KP16, second half): in a dense block every layer feeds all later layers; a standard l-layer network has about l weight matrices and DenseNet has l(l+1)/2;
  ResNet adds (x_l = F(x_{l-1}) + x_{l-1}, formula 11), DenseNet concatenates (x_l = F_l([x_0, ..., x_{l-1}]), formula 12); F_l = batch norm + 3x3 conv + ReLU;
  growth rate k: layer l sees k0 + k(l-1) input maps, k = 12 is usually enough; transition layer = batch norm, 1x1 conv, 2x2 average pooling;
  EfficientNet (2019): compound scaling of depth, width, resolution; ConvNeXt (2022): modernised CNN against Transformers. The source gives NO numbers for these.
[Added]: the example k0 = 16, k = 12 with 6 layers, the transition 88 -> 44, and the constants of compound scaling (alpha 1.2, beta 1.1, gamma 1.15, from the EfficientNet paper).
Run:  py assets/figures/src/lec07-kp07-figures.py
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc  # noqa: E402

from lec04_common import DARK, LIGHT, axes, clean, finish, skin  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), os.path.join(HERE, name)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def numbers():
    N = {}
    dense = lambda l: l * (l + 1) // 2
    assert [(l, l, dense(l)) for l in (4, 5, 6, 10)] == [(4, 4, 10), (5, 5, 15), (6, 6, 21), (10, 10, 55)]
    assert sum(range(1, 5)) == 10                                     # layer i receives i inputs (x0 .. x_{i-1})
    N["links"] = {l: dense(l) for l in (4, 5, 6, 10, 12)}
    k0, k, L = 16, 12, 6
    cin = [k0 + k * (i - 1) for i in range(1, L + 1)]; assert cin == [16, 28, 40, 52, 64, 76]
    out = k0 + k * L; assert out == 88
    w = [9 * c * k for c in cin]; assert w == [1728, 3024, 4320, 5616, 6912, 8208] and sum(w) == 29808 == 108 * sum(cin)
    N["growth"] = (k0, k, L, cin, out, w)
    assert 88 * 44 == 3872 and 56 // 2 == 28
    a, b, g = 1.2, 1.1, 1.15
    assert round(a * b ** 2 * g ** 2, 4) == 1.9203
    phi = {p: (a ** p, b ** p, g ** p, (a * b ** 2 * g ** 2) ** p) for p in (1, 2)}
    assert round(phi[2][0], 3) == 1.44 and round(phi[2][1], 3) == 1.21 and round(phi[2][2], 4) == 1.3225 and round(phi[2][3], 3) == 3.687
    N["scaling"] = phi
    # the week's architectures: weights computed in the scripts of KP02, KP03 and KP05
    k2 = load("lec07-kp02-figures.py").numbers(); k3 = load("lec07-kp03-figures.py").numbers(); k5 = load("lec07-kp05-figures.py").numbers()
    N["alex"] = k2["tot"]; N["vgg"] = k3["convw"] + k3["fcw"]; N["unet"] = k5["weights"]
    assert (N["alex"], N["vgg"], N["unet"]) == (60954656, 138344128, 31023808)
    return N


def show(N):
    print("links", N["links"]); print("growth", N["growth"]); print("scaling", {p: tuple(round(x, 4) for x in v) for p, v in N["scaling"].items()}); print("week", N["alex"], N["vgg"], N["unet"])


def fig_dense(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); W, H = 900, 400
    fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0.01, 0.01, 0.98, 0.98]); ax.set_xlim(-0.6, 9.6); ax.set_ylim(-2.4, 4.8); ax.axis("off")
    xs = np.arange(5) * 2.0 + 0.6
    names = ["$x_0$", "$x_1$", "$x_2$", "$x_3$", "$x_4$"]
    for i in range(5):
        ax.plot(xs[i], 0, "o", ms=20, mfc="none", mec=c["ref"] if i else c["text"], mew=2.4); ax.text(xs[i], 0, names[i], ha="center", va="center", fontsize=12, color=c["text"])
    for i in range(5):
        for j in range(i + 1, 5):
            r = (xs[j] - xs[i]) / 2; ctr = (xs[i] + xs[j]) / 2
            adj = j - i == 1
            ax.add_patch(Arc((ctr, 0.0), 2 * r, 2 * r * 0.9, theta1=0, theta2=180, color=c["main"] if not adj else c["ok"], lw=2.2 if adj else 1.4, alpha=1 if adj else 0.75))
    ax.text(4.6, 4.45, "dense block, $l=4$: $4+3+2+1=10$ links", ha="center", fontsize=13, color=c["main"])
    ax.text(4.6, -1.4, "a standard network has only the $4$ green links between neighbours", ha="center", fontsize=12, color=c["ok"])
    finish(fig, "lec07-kp07-dense", dark, c)
    return W, H


def fig_growth(dark):
    c = DARK if dark else LIGHT
    skin(c, dark, base_px=16); N = numbers(); k0, k, L, cin, out, w = N["growth"]; W, H = 860, 380
    fig, ax = axes(W, H, left=86, bottom=64, right=24, top=34); clean(ax)
    ax.bar(range(1, L + 1), cin, color=c["ref"], width=0.55)
    for i, v in enumerate(cin, 1): ax.text(i, v + 2, str(v), ha="center", fontsize=12, color=c["text"])
    ax.axhline(out, color=c["main"], lw=2, ls="--"); ax.text(L + 0.45, out + 2, f"$k_0+kL={out}$ at the end of the block", ha="right", fontsize=12, color=c["main"])
    ax.set_xticks(range(1, L + 1)); ax.set_xlabel("layer $\\ell$ in the dense block ($k_0=16$, $k=12$)"); ax.set_ylabel("input feature maps $k_0+k(\\ell-1)$"); ax.set_ylim(0, 100)
    finish(fig, "lec07-kp07-growth", dark, c)
    return W, H


if __name__ == "__main__":
    N = numbers(); show(N)
    for f in (fig_dense, fig_growth):
        for dark in (False, True): f(dark)
    print("figures written")
