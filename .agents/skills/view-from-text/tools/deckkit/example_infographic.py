"""example_infographic.py -- the reference "one page, one figure" layout (CLARITY.md C23): an information graphic made only of
native JianBauDau nodes (nativefig.NFig). Copy this file, change the data and the strings, keep the structure:

  legend (top right)  ->  N coloured columns: header chip, tinted panel with the example drawn with icons, the substitution as KaTeX,
  one big result chip, one plain-words comment  ->  a formula bar (formula + what each symbol means)  ->  one-line takeaway.

Rules it follows: every label >= 16 px; dark text on bright fills, light text on tinted panels; one colour per column role;
no drop shadows or gradients (the player has only flat fills); numbers computed in code, never typed; Chinese and English strings
come from one dictionary (KP_LANG=zh|en); the figure's alt text lists every number and the conclusion.

Run:  python3 example_infographic.py out.png      (light PNG for the lecture; in a deck: impfull().to_nodes(page, 60, y, "dark"))
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nativefig import NFig

A_C, B_C = "#6a9fd6", "#d97757"; RED = "#e07a70"; GRN = "#7dbb8c"; VIO2 = "#a98bd0"


def H(c):
    c = np.asarray(c, float); p = c[c > 0] / c.sum(); return float(-(p * np.log2(p)).sum())


def impfull():
    zh = os.environ.get("KP_LANG", "zh") != "en"
    L = {
     "legend": ("第一類", "第二類", "第三類") if zh else ("class 1", "class 2", "class 3"),
     "hd": ("三類混合：(4,4,2)", "三類一樣多：(1,1,1)", "全是同一類：(5,0,0)") if zh else ("Mixed: (4,4,2)", "Equal: (1,1,1)", "One class: (5,0,0)"),
     "tot": ("共 10 點", "共 3 點", "共 5 點") if zh else ("10 points", "3 points", "5 points"),
     "prop": ("比例 0.4、0.4、0.2", "比例 1/3、1/3、1/3", "比例 1、0、0") if zh else ("p = 0.4, 0.4, 0.2", "p = 1/3, 1/3, 1/3", "p = 1, 0, 0"),
     "cm": ("有混合，但三類比例不完全相同", "每類機會相同，最難猜下一點", "每一點都屬同一類，沒有混雜") if zh else ("mixed, but the shares differ", "equal shares: hardest to guess", "every point is one class: no mixing"),
     "note": "約定：$0\\log_2 0=0$" if zh else "convention: $0\\log_2 0=0$",
     "def": "$p_i$＝該類點數 ÷ 節點總點數；$t$ 是節點" if zh else "$p_i$ = points of class $i$ ÷ points in node $t$",
     "tk": "看的是各類的比例，不是總點數：三類越平均，不純度越高。" if zh else "It is the class shares, not the total: the more even the classes, the larger the impurity.",
    }
    alt = ("三個節點的不純度：(4,4,2) 為 0.529+0.529+0.464=1.522；(1,1,1) 為 log₂3=1.585（三類時的最大值）；(5,0,0) 為 0（完全純）；公式 I(t)=−Σ pᵢ log₂ pᵢ，pᵢ＝該類點數÷節點總點數；越平均不純度越高。" if zh else
           "Impurity of three nodes: (4,4,2) is 0.529+0.529+0.464=1.522; (1,1,1) is log₂3=1.585 (the maximum for three classes); (5,0,0) is 0 (pure); formula I(t)=−Σ pᵢ log₂ pᵢ with pᵢ = class points ÷ node points; the more even, the larger.")
    f = NFig("lec05-kp23-impfull", 1160, 452, alt)
    cols = ((RED, (224, 122, 112)), (A_C, (106, 159, 214)), (GRN, (125, 187, 140)))
    cw, gap = 372, 22; ink = "#1c1712"
    # legend
    lx = 740
    for i, nm in enumerate(L["legend"]):
        x = lx + i * 140
        (f.ellipse(x, 14, 17, A_C) if i == 0 else f.square(x, 14, 15, B_C) if i == 1 else f.tri(x, 14, 17, VIO2)); f.text(x + 14, 14, nm, "TXT", 16, False, "left")
    cnts = ((4, 4, 2), (1, 1, 1), (5, 0, 0)); Is = [abs(H(list(c))) for c in cnts]
    eq1 = ("$I=-(2\\times0.4\\log_2 0.4+0.2\\log_2 0.2)$", "$I=-3\\times\\frac{1}{3}\\log_2\\frac{1}{3}=\\log_2 3$", "$I=-1\\times\\log_2 1=0$")
    eq2 = ("$\\approx 0.529+0.529+0.464$", "", "")
    res = ("$I\\approx 1.522$", "$I\\approx 1.585$", "$I=0$")
    for k, ((c, rgb), cnt) in enumerate(zip(cols, cnts)):
        x = k * (cw + gap)
        f.rect(x, 34, cw, 38, fill=c, radius=12, text=L["hd"][k], tcolor=ink, size=20, bold=True)
        f.rect(x, 78, cw, 276, fill=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.12)", stroke=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.65)", sw=2, radius=14)
        f.rect(x + 12, 88, cw - 24, 96, stroke="MUT", sw=1.5, radius=10)
        # icons: rows of classes
        iy = 88 + 48 - (len([v for v in cnt if v]) - 1) * 13
        for ci, n in enumerate(cnt):
            if n == 0: continue
            for j in range(n):
                cx = x + 30 + j * 28; cy = iy + ci * 26
                if k == 1: cx = x + 34 + ci * 40; cy = 136
                if k == 2: cy = 136
                (f.ellipse(cx, cy, 17, A_C) if ci == 0 else f.square(cx, cy, 15, B_C) if ci == 1 else f.tri(cx, cy, 17, VIO2))
        f.text(x + 168, 118, L["tot"][k], "TXT", 18, True, "left"); f.text(x + 168, 150, L["prop"][k], "TXT", 16, False, "left")
        f.text(x + cw / 2, 204, eq1[k], "TXT", 18, w=cw - 20, math=True)
        if eq2[k]: f.text(x + cw / 2, 232, eq2[k], "TXT", 18, w=cw - 20, math=True)
        f.rect(x + 76, 248, cw - 152, 58, fill=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.30)", radius=12, text=res[k], tcolor="TXT", size=26, bold=True, math=True)
        f.text(x + cw / 2, 320, L["cm"][k], "TXT", 16, w=cw - 16)
        if k == 2: f.text(x + cw / 2, 342, L["note"], "MUT", 16, w=cw - 16, math=True)
    f.rect(0, 362, 1160, 44, fill="rgba(106,159,214,0.14)", radius=12)
    f.text(250, 382, "$I(t)=-\\sum_i p_i\\log_2 p_i$", "TXT", 22, w=360, math=True)
    f.text(600, 384, L["def"], "TXT", 18, w=540, ha="left", math=True)
    f.rect(0, 412, 1160, 40, fill="rgba(224,200,95,0.16)", stroke="rgba(224,200,95,0.55)", sw=1.5, radius=12, text=L["tk"], tcolor="TXT", size=20, bold=True)
    return f


if __name__ == "__main__":
    impfull().to_png(sys.argv[1] if len(sys.argv) > 1 else "example_infographic.png")
