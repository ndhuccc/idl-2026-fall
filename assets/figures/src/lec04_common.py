"""Shared helpers for the chapter-4 (regularization and overfitting) figure scripts.

Every figure script draws each figure in two skins from the same data:
    assets/figures/<id>.png   light skin, used by the lecture notes
    assets/figures/<id>.svg   dark skin, redrawn for the JianBauDau slides

Typical use inside assets/figures/src/lec04-kpNN-figures.py:

    from lec04_common import DARK, LIGHT, axes, clean, finish, skin
    def my_figure(dark):
        c = DARK if dark else LIGHT
        skin(c, dark)
        W, H = 640, 380                      # node size in JianBauDau px (figsize = W/100 x H/100)
        fig, ax = axes(W, H)
        ...                                  # draw with analytic coordinates
        clean(ax)
        finish(fig, "lec04-kpNN-<code>", dark, c)
        return W, H

Text inside figures: Latin letters, digits and math only (no Chinese).
"""
import base64
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, ".agents", "skills", "view-from-text"))
import matplotlib.pyplot as plt  # noqa: E402
from jbd_figure import svg_data_uri, use_jbd_style  # noqa: E402

OUT = os.path.join(ROOT, "assets", "figures")
DARK = dict(text="#f2ede4", muted="#b7b0a5", main="#d97757", ref="#6a9fd6", vec="#5fb8a8", ok="#7dbb8c", warn="#e07a70", edge="#e0995f", grid="#b7b0a5")
LIGHT = dict(text="#1c1712", muted="#5a544b", main="#c4572f", ref="#2b6cb0", vec="#1f8a79", ok="#2f855a", warn="#c0392b", edge="#b7791f", grid="#8a8378", bg="#ffffff")


def skin(c, dark, base_px=18):
    """Apply the dark (slides) or light (lecture) matplotlib skin."""
    use_jbd_style(base_px)
    plt.rcParams.update({"text.color": c["text"], "axes.labelcolor": c["text"], "axes.edgecolor": c["muted"], "xtick.color": c["muted"],
                         "ytick.color": c["muted"], "grid.color": c["grid"], "svg.fonttype": "path",
                         "figure.facecolor": "none" if dark else c["bg"], "axes.facecolor": "none" if dark else c["bg"],
                         "savefig.transparent": dark})


def finish(fig, key, dark, c):
    """Write assets/figures/<key>.svg (dark) or <key>.png (light) and close the figure."""
    if dark:
        uri = svg_data_uri(fig)
        with open(os.path.join(OUT, key + ".svg"), "wb") as fh:
            fh.write(base64.b64decode(uri.split(",", 1)[1]))
    else:
        fig.savefig(os.path.join(OUT, key + ".png"), dpi=160, facecolor=c["bg"])
    plt.close(fig)


def clean(ax):
    """Light grid, no top/right spines."""
    ax.grid(True, alpha=0.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def axes(W, H, left=86, bottom=66, right=24, top=28):
    """One axes filling a W x H px canvas with fixed pixel margins."""
    fig = plt.figure(figsize=(W / 100, H / 100))
    return fig, fig.add_axes([left / W, bottom / H, (W - left - right) / W, (H - bottom - top) / H])
