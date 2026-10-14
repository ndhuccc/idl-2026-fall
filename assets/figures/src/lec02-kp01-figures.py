"""Figures for Chapter 2 KP01 (biological neuron and the McCulloch-Pitts unit). Fixed teaching values only; labels are English.

  lec02-kp01-neuron-anatomy : dendrites, soma, axon, synapse and the add-then-threshold mechanism
  lec02-kp01-mp-threshold   : worked example x=(1,0,1), z=2, thresholds 1, 2, 3
  lec02-kp01-mp-gates       : AND (theta=2), OR (theta=1) and NOT (w=-1, w0=0.5) as one neuron

Lecture version: cream background, PNG (dpi 175) + SVG next to this folder, like the Chapter 1 figures.
If the environment variable DECK_WORK is set, a dark transparent SVG for the slides is also written to $DECK_WORK/assets/figures/.
"""
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
THEMES = {
    "light": dict(bg="#f6f4ee", text="#263849", muted="#677782", main="#d16b50", ref="#315f85", aux="#158b84", warn="#b3382c",
                  ok="#2f855a", edge="#b7791f", soft="#ffffff", onfill="#ffffff"),
    "dark": dict(bg="none", text="#f2ede4", muted="#b7b0a5", main="#d97757", ref="#6a9fd6", aux="#5fb8a8", warn="#e07a70",
                 ok="#7dbb8c", edge="#e0995f", soft="#2a241d", onfill="#1c1712"),
}
FM = 1.0


def px(size_px):
    return size_px * 0.72 * FM


def setup(theme, w, h):
    t = THEMES[theme]
    plt.rcParams.update({"mathtext.fontset": "cm", "svg.fonttype": "none" if theme == "light" else "path", "font.family": "DejaVu Sans",
                         "text.color": t["text"], "axes.labelcolor": t["text"], "axes.edgecolor": t["muted"],
                         "xtick.color": t["muted"], "ytick.color": t["muted"]})
    fig = plt.figure(figsize=(w / 100, h / 100))
    fig.patch.set_facecolor(t["bg"])
    return fig, t


def save(fig, name, theme):
    if theme == "light":
        fig.savefig(ROOT / f"{name}.png", dpi=175, facecolor=fig.get_facecolor())
        svg = ROOT / f"{name}.svg"
        fig.savefig(svg, facecolor=fig.get_facecolor())
        svg.write_text(re.sub(r"(?m)[ \t]+$", "", svg.read_text(encoding="utf-8")), encoding="utf-8")
    else:
        d = os.environ.get("DECK_WORK")
        if d:
            out = Path(d) / "assets" / "figures"
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"{name}.svg", transparent=True)
    plt.close(fig)


def arrow(ax, p, q, color, lw=2.4, ms=18):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw))


def anatomy(theme):
    global FM
    FM = 1.75 if theme == "dark" else 1.0
    W, H = 1100, 440
    fig, t = setup(theme, W, H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    dark = theme == "dark"
    fs = px(20)
    sx, sy = 430, 170
    for y0, yb in [(300, 215), (170, 170), (50, 125)]:
        ax.plot([90, 200, 320], [y0, y0 + (yb - y0) * 0.4, yb], color=t["ref"], lw=3.4, solid_capstyle="round")
        ax.plot([90, 55], [y0, y0 + 22], color=t["ref"], lw=2.4)
        ax.plot([90, 55], [y0, y0 - 22], color=t["ref"], lw=2.4)
        arrow(ax, (14, y0), (50, y0), t["muted"], lw=1.8, ms=14)
    ax.add_patch(Ellipse((sx, sy), 220, 150, fc=t["soft"], ec=t["main"], lw=3.4, zorder=2))
    ax.add_patch(Circle((sx, sy), 34, fc="none", ec=t["edge"], lw=2.6, zorder=3))
    ax.text(430, 226, "Nucleus", color=t["edge"], fontsize=px(15 if dark else 16), ha="center", va="center", zorder=4)
    ax.plot([540, 760], [170, 170], color=t["aux"], lw=7, solid_capstyle="round")
    arrow(ax, (620, 200), (720, 200), t["aux"], lw=3, ms=22)
    for dy in (-34, 0, 34):
        ax.plot([760, 800], [170, 170 + dy], color=t["aux"], lw=3.4)
        ax.add_patch(Circle((806, 170 + dy), 9, fc=t["aux"], ec=t["aux"]))
    ax.plot([826, 880], [170, 170], color=t["edge"], lw=3, ls=(0, (3, 3)))
    ax.text(853, 200, r"$\rightleftarrows$", ha="center", va="center", fontsize=fs, color=t["edge"])
    ax.add_patch(Ellipse((985, 170), 150, 110, fc=t["soft"], ec=t["muted"], lw=2.4))
    ax.text(985, 170, "next\nneuron", ha="center", va="center", fontsize=px(18), color=t["muted"])
    kw = dict(fontsize=fs, ha="center", va="center")
    ax.text(150, 385, "Dendrites" if dark else "Dendrites\n(receive signals)", color=t["ref"], **kw)
    ax.text(430, 385, "Soma" if dark else "Soma\n(adds signals up)", color=t["main"], **kw)
    ax.text(650, 112, "Axon" if dark else "Axon\n(carries the pulse)", color=t["aux"], **kw)
    ax.text(880, 262, "Synapse" if dark else "Synapse\n(chemical gap)", color=t["edge"], **kw)
    msg = "add up  →  ≥ threshold ?  →  fire / silent" if dark else "many inputs add up  →  total ≥ threshold ?  →  fire / stay silent"
    ax.text(550, 32, msg, ha="center", va="center", fontsize=px(22), color=t["text"])
    save(fig, "lec02-kp01-neuron-anatomy", theme)


def threshold(theme):
    global FM
    FM = 1.65 if theme == "dark" else 1.0
    W, H = 1100, 440
    fig, t = setup(theme, W, H)
    ax = fig.add_axes([0.04, 0.2, 0.92, 0.7])
    x = np.array([1, 0, 1])
    left = 0
    cols = [t["ref"], t["muted"], t["aux"]]
    for i, v in enumerate(x):
        if v:
            ax.barh(0, v, left=left, height=0.5, color=cols[i], ec=t["text"], lw=1.5)
            ax.text(left + v / 2, 0, rf"$x_{i + 1}=1$", ha="center", va="center", color=t["onfill"], fontsize=px(22), fontweight="bold")
            left += v
    ax.text(0.5, -0.43, r"$x_2=0$" if theme == "dark" else r"$x_2=0$ adds nothing", ha="center", va="center", color=t["muted"], fontsize=px(18))
    for th, ok in [(1, True), (2, True), (3, False)]:
        c = t["ok"] if ok else t["warn"]
        ax.plot([th, th], [-0.3, 0.45], color=c, lw=3, ls="--")
        ax.text(th, 0.5, rf"$\theta={th}$" + "\n" + (r"$o=1$ fires" if ok else r"$o=0$ silent"), ha="center", va="bottom", color=c,
                fontsize=px(20), fontweight="bold")
    ax.set_xlim(-0.05, 3.5)
    ax.set_ylim(-0.55, 1.0)
    ax.set_yticks([])
    ax.set_xticks([0, 1, 2, 3])
    ax.tick_params(labelsize=px(20))
    ax.set_xlabel(r"total  $z=x_1+x_2+x_3=2$" if theme == "dark" else r"accumulated total  $z=x_1+x_2+x_3$  (here $z=2$)", fontsize=px(20))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_facecolor("none")
    save(fig, "lec02-kp01-mp-threshold", theme)


def gates(theme):
    global FM
    FM = 1.5 if theme == "dark" else 1.0
    W, H = 1160, 400
    fig, t = setup(theme, W, H)
    fs = px(19)

    def plane(rect, title, pts, theta):
        ax = fig.add_axes(rect)
        ax.set_facecolor("none")
        xs = np.linspace(-0.4, 1.4, 50)
        ax.fill_between(xs, theta - xs, 1.5, color=t["main"], alpha=0.14)
        ax.plot(xs, theta - xs, color=t["edge"], lw=3)
        for (x1, x2, o) in pts:
            ax.scatter([x1], [x2], s=300, color=t["main"] if o else "none", edgecolor=t["main"] if o else t["ref"], lw=2.8, zorder=3)
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(-0.4, 1.4)
        ax.set_aspect("equal")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel(r"$x_1$", fontsize=fs)
        ax.set_ylabel(r"$x_2$", fontsize=fs)
        ax.set_title(title, fontsize=fs, pad=8)
        ax.tick_params(labelsize=fs)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    plane([0.07, 0.29, 0.24, 0.51], r"AND:  $\theta=2$", [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)], 2)
    plane([0.40, 0.29, 0.24, 0.51], r"OR:  $\theta=1$", [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)], 1)
    n = fig.add_axes([0.73, 0.29, 0.24, 0.51])
    n.set_facecolor("none")
    n.axvspan(-0.4, 0.5, color=t["main"], alpha=0.14)
    n.axvline(0.5, color=t["edge"], lw=3)
    for xv, ov in [(0, 1), (1, 0)]:
        n.scatter([xv], [0], s=300, color=t["main"] if ov else "none", edgecolor=t["main"] if ov else t["ref"], lw=2.8, zorder=3)
    n.set_xlim(-0.4, 1.4)
    n.set_ylim(-0.6, 0.6)
    n.set_yticks([])
    n.set_xticks([0, 1])
    n.set_xlabel(r"$x$", fontsize=fs)
    n.tick_params(labelsize=fs)
    n.set_title(r"NOT:  $w=-1,\ w_0=0.5$", fontsize=fs, pad=8)
    for s in ("top", "right", "left"):
        n.spines[s].set_visible(False)
    fig.text(0.5, 0.025, "filled = output 1,  open = output 0;  shaded side = where the neuron fires", ha="center", va="center",
             fontsize=px(19), color=t["muted"])
    save(fig, "lec02-kp01-mp-gates", theme)


if __name__ == "__main__":
    for th in ("light", "dark"):
        anatomy(th)
        threshold(th)
        gates(th)
    print("ok")
