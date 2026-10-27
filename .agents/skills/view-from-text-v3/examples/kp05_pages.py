"""KP05 pages designed in HTML (view-from-text-v3, route A). pageN(lang) -> body HTML of a 1160x480 figure."""
import os, sys, math
import numpy as np
SK = "/home/user/pr-2026-fall/.agents/skills"
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, SK + "/view-from-text-v3/examples"); sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, HERE)
from kp17_pages import Axes, tr, take, legend, fmt, sci, txt, named, card, head, chip, formula, svg, BL, RD, VI, GR, YE, OR, TE, INK, TXT, MUT, hexc, build, native, _sup, rgba, rgb
from kp15_pages import box, cardH
import kp05_nums as K
W, H = 1160, 480
CA, CB = "#4f86bf", "#c4572e"
CAt, CBt = "#7fb0e6", "#e8896a"
LINE, ANC = "#e0c85f", "#b79be0"
GT, GG = "#5fb8a8", "#b7b0a5"

def plane(x, y, w, h, lo=-0.5, hi=1.6):
    return Axes(x, y, w, h, (lo, hi), (lo, hi))
def pt(ax, p, dim=False, lab=True, dy=34, r=16):
    ca = K.TRUTH[p] == "A"; col = CA if ca else CB
    if dim: ax.raw(f'<circle cx="{ax.X(p[0]):.1f}" cy="{ax.Y(p[1]):.1f}" r="{r}" fill="#1c1712" stroke="{GG}" stroke-width="3" stroke-dasharray="5 4"/>')
    else: ax.dot(p[0], p[1], col, r)
    ax.text(p[0], p[1], K.TRUTH[p], "#f2ede4" if not dim else GG, "middle", 16, True, 0, 6)
    if lab: ax.text(p[0], p[1], f"({p[0]},{p[1]})", "#f2ede4", "middle", 16, False, 0, dy)
def poly(ax, pts, col, op, stroke=None, dash=None):
    s = " ".join(f"{ax.X(a):.1f},{ax.Y(b):.1f}" for a, b in pts)
    st = f' stroke="{stroke}" stroke-width="3"' + (f' stroke-dasharray="{dash}"' if dash else "") if stroke else ""
    ax.raw(f'<polygon points="{s}" fill="{col}" opacity="{op}"{st}/>')
def axlab(ax, a="x₁", b="x₂"):
    ax.text(ax.xr[1] - 0.02, ax.yr[0] + 0.05, a, GG, "end", 16, False, 0, -10); ax.text(ax.xr[0] + 0.02, ax.yr[1] - 0.02, b, GG, "start", 16, False, 8, 10)
def arrow(x1, x2, y): return f'<line x1="{x1}" y1="{y}" x2="{x2-8}" y2="{y}" stroke="#b7b0a5" stroke-width="3"/><polygon points="{x2},{y} {x2-12},{y-7} {x2-12},{y+7}" fill="#b7b0a5"/>'
def blk(x, y, w, h, c1, c2, label, lab2=None):
    o = box(x, y, w, h, c1, c2, 135, shadow=False, r=12) + txt(x, y + (12 if lab2 else h // 2 - 14), w, label, 17, TXT, True, "center", 28)
    if lab2: o += txt(x, y + 42, w, lab2, 16, TXT, False, "center", 26)
    return o
def row(x, y, w, h, role, a=.16): return card(x, y, w, h, role, a)

# ------------------------------------------------ 1: master unit splits the points into two groups
def page1(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 640, 480, "blue", t("四個點與主要節點的線 x₁ + x₂ = 0.5", "The four points and the master unit's line x₁ + x₂ = 0.5"))
    ax = plane(70, 56, 500, 296)
    poly(ax, [(-0.5, 1.0), (-0.5, 1.6), (1.6, 1.6), (1.6, -0.5), (1.0, -0.5)], GT, .13); poly(ax, [(-0.5, 1.0), (-0.5, -0.5), (1.0, -0.5)], GG, .10)
    ax.frame([0, 1], [0, 1]); ax.line([(-0.5, 1.0), (1.0, -0.5)], LINE, 4)
    ax.text(1.55, 1.5, t("z₁ = 1 群", "z₁ = 1 group"), GT, "end", 18, True, 0, 6); ax.text(-0.45, -0.38, t("z₁ = 0 群", "z₁ = 0 group"), GG, "start", 18, True, 0, 6)
    for p in K.PTS: pt(ax, p, dy=-26 if p[1] == 1 else 38)
    axlab(ax); o += ax.html()
    o += txt(16, 394, 608, t("訓練集：N = 4 個帶標籤的點，A 是 (0,1)、(1,0)，B 是 (0,0)、(1,1)（XOR）。", "Training set: N = 4 labelled points, A at (0,1), (1,0) and B at (0,0), (1,1) (XOR)."), 17, TXT, False, "center", 52)
    o += txt(16, 440, 608, t("黃線：x₁ + x₂ = 0.5", "Yellow line: x₁ + x₂ = 0.5"), 16, LINE, True, "center", 24)
    o += cardH(656, 0, 504, 480, "teal", t("主要節點怎麼算", "How the master unit computes"))
    o += named(672, 46, 472, t("主要節點：每一層第一個負責分類的單元，把所有點分成輸出 1 與輸出 0 兩群。", "Master unit: the first classifying unit of each layer; it splits all points into an output-1 group and an output-0 group."), "term:" + t("主要節點", "Master unit"), 16, TXT, 72)
    o += named(672, 124, 472, t("階梯函數 f(t)：t > 0 得 1，否則得 0。", "Step function f(t): 1 if t > 0, otherwise 0."), "term:" + t("階梯函數", "Step function"), 16, TXT, 30)
    o += card(672, 160, 472, 52, "violet", .22) + formula(672, 164, 472, r"z_1=f(x_1+x_2-0.5)", 24, 44)
    for xx, lab, w_ in ((676, t("輸入", "input"), 100), (784, t("分數", "score"), 200), (992, "z₁", 140)): o += txt(xx, 220, w_, lab, 16, MUT, True, "center", 24)
    for i, p in enumerate(K.PTS):
        y0 = 248 + i * 34; o += row(672, y0, 472, 30, "teal" if K.z1(p) else "grey", .2 if K.z1(p) else .1)
        o += txt(676, y0 + 2, 100, f"({p[0]},{p[1]})", 17, TXT, False, "center", 26) + txt(784, y0 + 2, 200, f"{p[0]}+{p[1]}−0.5 = {K.s1(p):+.1f}".replace("+-", "−"), 17, TXT, False, "center", 26) + txt(992, y0 + 2, 140, str(K.z1(p)), 18, TXT, True, "center", 26)
    o += card(672, 392, 472, 66, "yellow", .14) + txt(680, 400, 456, t("0 群 1 點，1 群 3 點：先分成兩群。〔程式計算〕", "0 group: 1 point; 1 group: 3 points. Two groups first. (computed)"), 18, TXT, True, "center", 50)
    return o

# ------------------------------------------------ 2: mixed group
def page2(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 600, 480, "red", t("主要節點的 1 群裡，兩類都有", "In the master unit's 1 group both classes appear"))
    ax = plane(60, 56, 480, 296)
    ax.frame([0, 1], [0, 1]); ax.line([(-0.5, 1.0), (1.0, -0.5)], LINE, 3)
    poly(ax, [(-0.5, 1.3), (1.3, 1.3), (1.3, -0.5)], RD, .10, RD, "9 7")
    for p in K.PTS: pt(ax, p, dy=-26 if p[1] == 1 else 38)
    ax.text(1.55, 1.5, t("混雜群", "mixed group"), "#f08a80", "end", 18, True, 0, 4); axlab(ax); o += ax.html()
    o += txt(16, 394, 568, t("紅色虛線框：1 群的三個點，同時有 A 和 B。", "Red dashed frame: the three points of the 1 group, with both A and B."), 17, TXT, True, "center", 52)
    o += cardH(616, 0, 544, 480, "orange", t("逐群檢查標籤", "Check the labels group by group"))
    o += row(632, 48, 512, 66, "green", .18) + txt(640, 54, 496, t("0 群：(0,0)，真值 B，判 B，對。", "0 group: (0,0), truth B, called B: right."), 17, TXT, True, "left", 28) + txt(640, 84, 496, t("只有一個點，而且是 B：這一群是純的。", "One point, and it is B: this group is pure."), 16, MUT, False, "left", 26)
    o += row(632, 126, 512, 136, "red", .18) + txt(640, 132, 496, t("1 群：三個點一律判 A", "1 group: all three are called A"), 17, TXT, True, "left", 28)
    for i, (p, ok) in enumerate((((0, 1), 1), ((1, 0), 1), ((1, 1), 0))):
        o += txt(648, 164 + i * 30, 480, f"{p}  " + t("真值", "truth") + f" {K.TRUTH[p]}，" + t("判 A", "called A") + "　" + ("✓ " + t("對", "right") if ok else "✗ " + t("錯", "wrong")), 17, TXT if ok else "#f08a80", not ok, "left", 26)
    o += named(632, 274, 512, t("混雜群：同一個輸出群裡，兩個類別的點都有。", "Mixed group: an output group that contains points of both classes."), "term:" + t("混雜群", "Mixed group"), 16, TXT, 30)
    o += card(632, 328, 512, 130, "yellow", .14) + txt(640, 332, 496, t("答對 3 點、答錯 1 點", "3 points right, 1 point wrong"), 19, TXT, True, "center", 30) + txt(640, 364, 496, t("混雜群有 3 點：(0,1)、(1,0)、(1,1)。〔程式計算〕", "The mixed group has 3 points: (0,1), (1,0), (1,1). (computed)"), 17, TXT, False, "center", 30) + txt(640, 418, 496, t("接下來要把這 3 點裡的 B 分出來。", "Next we split the B out of these 3 points."), 17, MUT, False, "center", 52)
    return o

# ------------------------------------------------ 3: ancillary unit
def page3(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 600, 480, "violet", t("再加一條輔助線 x₁ + x₂ = 1.5", "Add an ancillary line x₁ + x₂ = 1.5"))
    ax = plane(60, 56, 480, 296)
    ax.frame([0, 1], [0, 1]); ax.line([(-0.5, 1.0), (1.0, -0.5)], LINE, 4); ax.line([(-0.1, 1.6), (1.6, -0.1)], ANC, 4, "10 7")
    pt(ax, (0, 0), dim=True, dy=38)
    for p in ((0, 1), (1, 0), (1, 1)): pt(ax, p, dy=-26 if p[1] == 1 else 38)
    ax.text(1.55, 1.52, t("紫虛線：輔助節點", "purple dashed: ancillary unit"), ANC, "end", 17, True, 0, 0); ax.text(-0.45, -0.38, t("黃線：主要節點", "yellow: master unit"), LINE, "start", 17, True, 0, 6); axlab(ax); o += ax.html()
    o += txt(40, 394, 520, t("虛線圈的 (0,0) 不在混雜群，不參與訓練；這條線是手動選定的機制示範，不是跑過某個訓練演算法。", "The dashed (0,0) is outside the mixed group and takes no part in training; the line is picked by hand to show the mechanism, not the result of a training algorithm."), 16, TXT, False, "center", 72)
    o += cardH(616, 0, 544, 480, "teal", t("輔助節點 z₂ 的算法", "How the ancillary unit z₂ computes"))
    o += named(632, 46, 512, t("輔助節點：只針對混雜群的點設計的新增單元，多給每個點一個位元。", "Ancillary unit: an added unit designed only for the points of the mixed group; it gives each point one more bit."), "term:" + t("輔助節點", "Ancillary unit"), 16, TXT, 72)
    o += card(632, 124, 512, 52, "violet", .22) + formula(632, 128, 512, r"z_2=f(1.5-x_1-x_2)", 24, 44)
    for xx, lab, w_ in ((636, t("輸入", "input"), 90), (734, t("分數", "score"), 200), (950, "z₂", 70), (1060, "", 70)): o += txt(xx, 184, w_, lab, 16, MUT, True, "center", 24)
    for i, p in enumerate(((0, 1), (1, 0), (1, 1), (0, 0))):
        y0 = 212 + i * 36; used = i < 3; o += row(632, y0, 512, 32, "teal" if K.z2(p) else "orange", .2 if used else .06)
        sc = f"1.5−{p[0]}−{p[1]} = {K.s2(p):+.1f}".replace("+-", "−")
        o += txt(636, y0 + 3, 90, f"({p[0]},{p[1]})", 17, TXT if used else MUT, False, "center", 26) + txt(734, y0 + 3, 200, sc, 17, TXT if used else MUT, False, "center", 26) + txt(950, y0 + 3, 70, str(K.z2(p)), 18, TXT if used else MUT, True, "center", 26)
        if not used: o += txt(1036, y0 + 3, 100, t("不訓練", "unused"), 16, MUT, False, "center", 26)
    o += card(632, 366, 512, 92, "yellow", .14) + txt(640, 372, 496, t("線夾在 (1,1) 與 (0,1)、(1,0) 之間：(0,1)、(1,0) 得 +0.5、位元 1；(1,1) 得 −0.5、位元 0。〔程式計算〕", "The line lies between (1,1) and (0,1), (1,0): (0,1), (1,0) score +0.5 and bit 1; (1,1) scores −0.5 and bit 0. (computed)"), 17, TXT, True, "center", 76)
    return o

# ------------------------------------------------ 4: codes and purification
def page4(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 560, 480, "teal", t("每個點的編碼 z₁z₂", "The code z₁z₂ of each point"))
    for xx, lab, w_ in ((14, t("輸入", "input"), 100), (118, "z₁", 70), (190, "z₂", 70), (264, t("編碼", "code"), 110), (380, t("真值", "truth"), 100)): o += txt(xx, 48, w_, lab, 16, MUT, True, "center", 24)
    for i, p in enumerate(K.PTS):
        y0 = 78 + i * 46; ca = K.TRUTH[p] == "A"; o += row(14, y0, 532, 40, "blue" if ca else "orange", .2)
        o += txt(14, y0 + 7, 100, f"({p[0]},{p[1]})", 17, TXT, False, "center", 26) + txt(118, y0 + 7, 70, str(K.z1(p)), 17, TXT, False, "center", 26) + txt(190, y0 + 7, 70, str(K.z2(p)), 17, TXT, False, "center", 26)
        o += txt(264, y0 + 7, 110, K.code(p), 20, TXT, True, "center", 26) + txt(380, y0 + 7, 100, K.TRUTH[p], 18, TXT, True, "center", 26)
    o += named(16, 276, 528, t("純化：新增位元，讓不同類的點不再共用完全相同的編碼。", "Purification: add bits so that points of different classes no longer share exactly the same code."), "term:" + t("純化", "Purification"), 16, TXT, 72)
    o += card(16, 366, 528, 92, "yellow", .14) + txt(24, 372, 512, t("兩個 A 共用 11 沒關係，因為同類；B 佔 01 與 10 兩個碼也沒關係：純化不要求一類只有一個碼。", "The two A sharing 11 is fine, since they are the same class; B using the two codes 01 and 10 is fine too: purification does not require one code per class."), 16, TXT, False, "center", 76)
    o += cardH(576, 0, 584, 480, "blue", t("按 z₁ 分群與按完整編碼分群", "Groups by z₁ alone and by the full code"))
    o += row(592, 48, 552, 190, "red", .12) + txt(600, 54, 536, t("只看主要節點的位元 z₁", "Only the master unit's bit z₁"), 17, TXT, True, "left", 28)
    o += row(608, 90, 240, 66, "green", .2) + txt(612, 94, 232, "z₁ = 0", 16, MUT, True, "center", 24) + txt(612, 120, 232, t("(0,0)：B　純", "(0,0): B  pure"), 17, TXT, False, "center", 26)
    o += row(864, 90, 264, 114, "red", .24) + txt(868, 94, 256, "z₁ = 1", 16, MUT, True, "center", 24) + txt(868, 120, 256, t("(0,1) A　(1,0) A<br>(1,1) B　混雜", "(0,1) A  (1,0) A<br>(1,1) B  mixed"), 17, TXT, False, "center", 52)
    o += txt(600, 206, 536, t("混雜 3 點", "3 mixed points"), 19, "#f08a80", True, "center", 28)
    o += row(592, 250, 552, 208, "green", .12) + txt(600, 256, 536, t("看完整編碼 z₁z₂", "The full code z₁z₂"), 17, TXT, True, "left", 28)
    for i, (c, s_) in enumerate((("01", t("(0,0)：B", "(0,0): B")), ("11", t("(0,1)、(1,0)：A、A", "(0,1), (1,0): A, A")), ("10", t("(1,1)：B", "(1,1): B")))):
        xx = 608 + i * 180; o += row(xx, 296, 168, 98, "green", .24) + txt(xx, 300, 168, t("編碼 ", "code ") + c, 16, MUT, True, "center", 24) + txt(xx, 326, 168, s_, 16, TXT, False, "center", 48)
    o += txt(600, 408, 536, t("三組全純：混雜 0 點。〔程式計算〕", "All three groups pure: 0 mixed points. (computed)"), 19, "#7dbb8c", True, "center", 30)
    return o

# ------------------------------------------------ 5: new code plane
def page5(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 640, 480, "blue", t("新的 (z₁, z₂) 平面：一條線 z₁ + z₂ = 1.5", "The new (z₁, z₂) plane: one line z₁ + z₂ = 1.5"))
    ax = Axes(80, 56, 500, 296, (-0.5, 1.6), (-0.5, 1.6)); ax.frame([0, 1], [0, 1])
    poly(ax, [(0.0, 1.5), (-0.5, 1.5), (-0.5, 1.6), (1.6, 1.6), (1.6, -0.5), (1.5, 0.0)][0:0] or [(-0.1, 1.6), (1.6, 1.6), (1.6, -0.1)], CA, .14)
    poly(ax, [(-0.5, -0.5), (-0.5, 1.6), (-0.1, 1.6), (1.6, -0.1), (1.6, -0.5)], CB, .10)
    ax.line([(-0.1, 1.6), (1.6, -0.1)], LINE, 4)
    ax.raw(f'<circle cx="{ax.X(0):.1f}" cy="{ax.Y(0):.1f}" r="16" fill="none" stroke="{GG}" stroke-width="3" stroke-dasharray="5 4"/>'); ax.text(0, 0, t("沒有點", "no point"), GG, "middle", 16, False, 0, 40)
    ax.dot(0, 1, CB, 16); ax.text(0, 1, "B", "#f2ede4", "middle", 16, True, 0, 6); ax.text(0, 1, "01", "#f2ede4", "middle", 16, False, 0, -26)
    ax.dot(1, 0, CB, 16); ax.text(1, 0, "B", "#f2ede4", "middle", 16, True, 0, 6); ax.text(1, 0, "10", "#f2ede4", "middle", 16, False, 0, 36)
    ax.dot(1, 1, CA, 16); ax.text(1, 1, "A", "#f2ede4", "middle", 16, True, 0, 6); ax.text(1, 1, t("11（兩個 A）", "11 (two A)"), "#f2ede4", "middle", 16, False, 0, -26)
    ax.text(1.55, 1.5, t("判 A", "call A"), CAt, "end", 18, True, 0, 6); ax.text(-0.45, -0.38, t("判 B", "call B"), CBt, "start", 18, True, 0, 6)
    ax.text(1.55, -0.38, "z₁", GG, "end", 16, False, 0, 6); ax.text(-0.45, 1.5, "z₂", GG, "start", 16, False, 0, 6); o += ax.html()
    o += txt(16, 394, 608, t("橫軸 z₁，縱軸 z₂：兩個 A 合併在 11，00 沒有點。", "Horizontal z₁, vertical z₂: the two A merge at 11, and 00 has no point."), 17, TXT, False, "center", 52)
    o += cardH(656, 0, 504, 480, "violet", t("新的主要節點：z₁ + z₂ − 1.5", "The new master unit: z₁ + z₂ − 1.5"))
    o += card(672, 52, 472, 52, "violet", .22) + formula(672, 56, 472, r"s=z_1+z_2-1.5\quad(s>0\Rightarrow \mathrm{A})", 22, 44)
    for xx, lab, w_ in ((676, t("編碼", "code"), 90), (774, t("算式", "working"), 220), (1002, "s", 70), (1076, t("判", "call"), 60)): o += txt(xx, 114, w_, lab, 16, MUT, True, "center", 24)
    for i, c in enumerate(("11", "01", "10")):
        y0 = 142 + i * 44; ok = K.pred3(c) == ("A" if c == "11" else "B"); o += row(672, y0, 472, 38, "blue" if K.pred3(c) == "A" else "orange", .2)
        o += txt(676, y0 + 6, 90, c, 18, TXT, True, "center", 26) + txt(774, y0 + 6, 220, f"{c[0]}+{c[1]}−1.5", 17, TXT, False, "center", 26) + txt(1002, y0 + 6, 70, f"{K.s3(c):+.1f}".replace("+-", "−"), 18, TXT, True, "center", 26) + txt(1076, y0 + 6, 60, K.pred3(c), 18, TXT, True, "center", 26)
    o += card(672, 286, 472, 172, "green", .16) + txt(680, 294, 456, t(f"四個點全對：{K.final_correct()} / 4", f"All four right: {K.final_correct()} / 4"), 22, TXT, True, "center", 34) + txt(680, 336, 456, t("兩個 A 合併在 11 沒有關係，因為同類；(0,0) 的編碼是 01，判 B，也對。〔程式計算〕", "Merging the two A at 11 is fine, since they are the same class; (0,0) has code 01, is called B, and is right. (computed)"), 17, TXT, False, "center", 100)
    return o

# ------------------------------------------------ 6: why it stops
def bars(ax, vals, cols, names):
    for i, (v, c) in enumerate(zip(vals, cols)):
        a = 0.2 + i; ax.rect(a, a + 0.6, 0, max(v, 0.02), c, .9); ax.text(a + 0.3, v, str(v), "#f2ede4", "middle", 20, True, 0, -10)
def page6(lang):
    t = tr(lang); o = ""
    m0, _ = K.mixed_count(K.z1); m1, _ = K.mixed_count(K.code); c0, c1 = K.master_correct(), K.final_correct(); N = len(K.PTS)
    o += cardH(0, 0, 380, 480, "red", t("進展一：混雜群的點數", "Progress 1: points in the mixed group"))
    ax = Axes(60, 64, 290, 290, (0, 2), (0, 4.6)); ax.frame([], [0, 1, 2, 3, 4]); bars(ax, [m0, m1], [RD, GR], None); o += ax.html()
    o += txt(60, 366, 140, t("只有主要節點", "master unit only"), 16, MUT, False, "center", 48) + txt(210, 366, 140, t("加上輔助節點", "plus ancillary unit"), 16, MUT, False, "center", 48)
    o += txt(16, 424, 348, f"{m0} → {m1}：" + t("嚴格變小", "strictly smaller"), 19, TXT, True, "center", 28)
    o += cardH(396, 0, 380, 480, "blue", t("進展二：答對的點數", "Progress 2: points answered right"))
    ax = Axes(456, 64, 290, 290, (0, 2), (0, 4.6)); ax.frame([], [0, 1, 2, 3, 4]); bars(ax, [c0, c1], [BL, GR], None)
    ax.line([(0, N), (2, N)], YE, 3, "9 6"); ax.text(0.02, N, f"N = {N}", YE, "start", 17, True, 0, -10); o += ax.html()
    o += txt(456, 366, 140, t("只有主要節點", "master unit only"), 16, MUT, False, "center", 48) + txt(606, 366, 140, t("加上輔助節點", "plus ancillary unit"), 16, MUT, False, "center", 48)
    o += txt(412, 424, 348, f"{c0} → {c1}：" + t("至少多對一點", "at least one more right"), 19, TXT, True, "center", 28)
    o += cardH(792, 0, 368, 480, "violet", t("為什麼會停", "Why it stops"))
    o += card(808, 50, 336, 84, "violet", .22) + formula(808, 56, 336, r"C_{k+1}\geq C_k+1,\quad C_k\leq N", 24, 72)
    o += txt(808, 142, 336, t("Cₖ：第 k 層主要節點答對的點數；N：訓練點總數（本例 4）。", "Cₖ: points the layer-k master units get right; N: number of training points (here 4)."), 16, TXT, False, "left", 76)
    o += txt(808, 212, 336, t("整數不可能一直增加，所以會停。", "An integer cannot keep increasing, so it stops."), 18, TXT, True, "center", 30)
    o += card(808, 282, 336, 176, "yellow", .14) + txt(816, 286, 320, t("條件：這是所引構造的要求，不是本例驗出的一般結論，本頁不證明。數的是混雜群的點數，不是錯誤率；也不是說任何不可分的子群都能被完美分開。〔程式計算〕", "Condition: required by the cited construction, not verified here as a general result and not proved on this page. It counts mixed-group points, not an error rate, and does not say every inseparable subgroup can be split perfectly. (computed)"), 16, TXT, False, "left", 170)
    return o

# ------------------------------------------------ 7: stopping conditions
def page7(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 520, 480, "red", t("同一個輸入，兩個標籤", "One input, two labels"))
    o += f'<div style="position:absolute;left:92px;top:104px;width:84px;height:84px;background:{CB};border-radius:10px"></div><div style="position:absolute;left:104px;top:116px;width:60px;height:60px;background:{CA};border-radius:30px"></div>'
    o += txt(30, 204, 220, t("圓點 A 與方塊 B<br>疊在同一個位置", "circle A and square B<br>at the same place"), 17, TXT, False, "center", 54)
    o += svg(0, 0, W, H, arrow(262, 322, 146))
    o += box(330, 96, 170, 100, "#5a544b", "#3a352d", 135, shadow=False, r=12) + txt(330, 110, 170, t("確定性網路", "Deterministic<br>network"), 17, TXT, True, "center", 52 if lang == "en" else 28) + txt(330, 150 if lang == "en" else 146, 170, t("同一輸入<br>只有一個輸出", "one input,<br>one output") if lang == "zh" else "", 16, TXT, False, "center", 48 if lang == "zh" else 24)
    o += row(16, 260, 488, 90, "red", .22) + txt(24, 266, 472, t("A 與 B 不可能兩筆都對", "A and B cannot both be right"), 20, TXT, True, "center", 32) + txt(24, 302, 472, t("再增加多少節點也沒有用。", "No number of added units helps."), 17, MUT, False, "center", 30)
    o += txt(16, 372, 488, t("條件 1：標籤不能互相矛盾。", "Condition 1: labels must not contradict."), 19, TXT, True, "center", 30)
    o += cardH(536, 0, 624, 480, "orange", t("停止是有條件的", "Stopping has conditions"))
    o += row(552, 52, 592, 72, "grey", .12) + txt(560, 58, 576, t("1. 標籤不矛盾：相同的輸入不能被標成不同的類。", "1. Consistent labels: the same input cannot carry two different classes."), 17, TXT, False, "left", 60)
    o += row(552, 134, 592, 72, "grey", .12) + txt(560, 140, 576, t("2. 節點依規則增加，才有嚴格進展；隨便增加神經元不保證會停。", "2. Units are added by rule for strict progress; adding neurons at random does not guarantee a stop."), 17, TXT, False, "left", 60)
    o += named(552, 218, 592, t("泛化：對沒看過的資料也判對的能力。", "Generalization: the ability to classify data not seen in training."), "term:" + t("泛化", "Generalization"), 16, TXT, 30)
    o += card(552, 262, 592, 196, "red", .18) + txt(560, 268, 576, t("常見迷思：既然最後能把訓練點全分對，新資料也一定對。", "Myth: if the training points end up all right, new data must be right too."), 18, TXT, True, "center", 56) + txt(560, 330, 576, t("全對只涉及給定的這批點：沒看過的位置不保證也對，也不保證網路最小。", "All right concerns only the given points: unseen places are not guaranteed, and the network is not guaranteed to be small."), 17, TXT, False, "center", 100)
    return o

# ------------------------------------------------ 8: nearest neighbour bisector
def page8(lang):
    t = tr(lang); o = ""; a, b, x0 = K.nearest_x1(); d = math.sqrt(0.5 ** 2 + 0.7 ** 2)
    o += cardH(0, 0, 560, 480, "blue", t("兩個訓練點 u、v 與它們的垂直平分線", "Two training points u, v and their perpendicular bisector"))
    ax = Axes(70, 56, 440, 270, (-0.6, 1.6), (-0.4, 1.0)); ax.frame([0, 1], [0, 0.5, 1])
    ax.line([(0.5, -0.4), (0.5, 1.0)], LINE, 4)
    ax.line([(0.5, 0.7), (0, 0)], "#f2ede4", 2, "6 5"); ax.line([(0.5, 0.7), (1, 0)], "#f2ede4", 2, "6 5")
    ax.dot(0, 0, "#2f7f73", 14); ax.text(0, 0, "u", "#f2ede4", "middle", 17, True, 0, 6); ax.dot(1, 0, "#2f7f73", 14); ax.text(1, 0, "v", "#f2ede4", "middle", 17, True, 0, 6)
    ax.dot(0.5, 0.7, "#f2ede4", 8); ax.text(0.5, 0.7, "x", "#f2ede4", "start", 16, True, 12, -4)
    ax.text(0.25, 0.37, f"d = {d:.3f}", "#f2ede4", "end", 16, False, -4, 0); ax.text(0.75, 0.37, f"d = {d:.3f}", "#f2ede4", "start", 16, False, 4, 0)
    ax.text(0.5, 0.95, "x₁ = 0.5", LINE, "start", 17, True, 8, 8); ax.text(0, 0, "(0,0)", "#f2ede4", "middle", 16, False, 0, 36); ax.text(1, 0, "(1,0)", "#f2ede4", "middle", 16, False, 0, 36)
    o += ax.html()
    o += txt(16, 366, 528, t("線上任一點 x（例如 (0.5, 0.7)）到 u 和 v 的距離相同。〔程式計算〕", "Any point x on the line (for example (0.5, 0.7)) is equally far from u and v. (computed)"), 17, TXT, False, "center", 52)
    o += named(16, 424, 528, t("垂直平分線：到兩點等距的點的集合（高維時是超平面）。", "Perpendicular bisector: all points equally far<br>from two points (a hyperplane in higher dimensions)."), "term:" + t("垂直平分線", "Perpendicular bisector"), 16, TXT, 36)
    o += cardH(576, 0, 584, 480, "violet", t("等距條件展開，二次項消掉", "Expand the equal-distance condition: the quadratic terms cancel"))
    o += card(592, 50, 552, 56, "violet", .22) + formula(592, 54, 552, r"\|\mathbf{x}-\mathbf{u}\|^2=\|\mathbf{x}-\mathbf{v}\|^2", 24, 48)
    o += card(592, 114, 552, 56, "violet", .22) + formula(592, 118, 552, r"\Leftrightarrow\ 2(\mathbf{v}-\mathbf{u})^T\mathbf{x}=\|\mathbf{v}\|^2-\|\mathbf{u}\|^2", 22, 48)
    o += card(592, 180, 552, 84, "teal", .16) + txt(600, 184, 536, t("取 u = (0,0)、v = (1,0)：", "Take u = (0,0), v = (1,0):"), 17, TXT, True, "left", 28) + formula(600, 212, 536, r"2x_1=1\ \Rightarrow\ x_1=0.5", 24, 46)
    o += txt(592, 274, 552, t("N 個點的點對數：N(N−1)/2", "Number of pairs among N points: N(N−1)/2"), 17, TXT, True, "center", 28)
    for i, n in enumerate((4, 10, 100)):
        xx = 600 + i * 184; o += row(xx, 308, 172, 64, "orange", .2) + txt(xx, 312, 172, f"N = {n}", 16, MUT, True, "center", 24) + txt(xx, 338, 172, f"{n}×{n-1}÷2 = {K.pairs(n):,}", 16, TXT, True, "center", 26)
    o += card(592, 384, 552, 82, "yellow", .14) + txt(600, 390, 536, t("這只是兩點的距離比較，不是完整的最近鄰網路實驗；點對多不表示都要保留，也不是最小網路大小。", "This is only a distance comparison of two points, not a full nearest-neighbour network experiment; many pairs do not mean all are kept, nor is it a minimal network size."), 16, TXT, False, "center", 72)
    return o

# ------------------------------------------------ 9: three layers suffice != at most three layers
def page9(lang):
    t = tr(lang); o = ""; a = ""
    o += cardH(0, 0, 1160, 220, "blue", t("兩段：先長出二元編碼，再用偵測器合併", "Two stages: grow binary codes, then merge with detectors"))
    o += blk(20, 64, 140, 90, "#5a544b", "#3a352d", t("輸入 x", "input x")) + blk(210, 64, 270, 90, "#4f86bf", "#2f4f7a", t("第 1 層", "Layer 1"), t("主要節點 + 輔助節點", "master + ancillary units")) + blk(530, 64, 170, 90, "#4f9a68", "#2c6140", t("二元編碼", "binary codes"), "z₁z₂…") + blk(750, 64, 230, 90, "#d9822b", "#a8452b", t("偵測器加或合併", "detectors, OR-merge"), t("（KP04）", "(KP04)")) + blk(1030, 64, 110, 90, "#5a544b", "#3a352d", t("輸出", "output"))
    a += arrow(160, 210, 109) + arrow(480, 530, 109) + arrow(700, 750, 109) + arrow(980, 1030, 109); o += svg(0, 0, W, H, a)
    o += txt(210, 168, 270, t("逐層增長，直到純化", "grows layer by layer until pure"), 16, MUT, False, "center", 26) + txt(750, 168, 230, t("後面各層的輸入是二元編碼", "later layers take binary codes"), 16, MUT, False, "center", 26)
    o += card(0, 236, 570, 150, "green", .16) + txt(8, 242, 554, t("存在性：適當的三層表示足夠", "Existence: three suitable layers suffice"), 19, TXT, True, "center", 30) + txt(8, 282, 554, t("用 KP04 的偵測器加或合併，就能分類純化後的編碼。", "KP04's detectors with an OR-merge classify the purified codes."), 19, TXT, False, "center", 76)
    o += card(590, 236, 570, 150, "red", .16) + txt(598, 242, 554, t("程序：增長沒有層數上限", "Procedure: growth has no depth bound"), 19, TXT, True, "center", 30) + txt(598, 282, 554, t("一層一層增長的程序，沒有保證在三層內停止。", "The layer-by-layer growing procedure is not guaranteed to stop within three layers."), 19, TXT, False, "center", 76)
    o += card(0, 400, 1160, 70, "yellow", .14) + txt(16, 412, 1128, t("三層足夠不等於最多三層：存在性與程序的深度保證不是同一個定理。", "Three layers suffice is not at most three layers: existence and a depth guarantee for the procedure are not the same theorem."), 19, TXT, True, "center", 46)
    return o

# ------------------------------------------------ 10: exercises
def page10(lang):
    t = tr(lang); o = ""
    q1 = [t("01：(0,0)　真值 B　純", "01: (0,0)  truth B  pure"), t("11：(0,1)、(1,0)　A、A　純", "11: (0,1), (1,0)  A, A  pure"), t("10：(1,1)　真值 B　純", "10: (1,1)  truth B  pure")]
    q2 = [t("相同的輸入 → 相同的輸出", "same input → same output"), t("確定性網路只給一個答案", "a deterministic network gives one answer")]
    qs = [("blue", t("題 1", "Q1"), t("依完整編碼把四個點分組，列出每組的真值，並算混雜群的點數。", "Group the four points by the full code, list each group's truth and count the points in the mixed group."), q1, t("混雜 0 點（只看 z₁ 是 3 點）〔程式計算〕", "0 mixed points (3 with z₁ alone) (computed)"), t("核對：3→0 數的是還混雜的點；錯誤率數答錯的點，主要節點單獨答錯 1 點。", "Check: 3→0 counts the still-mixed points; the error rate counts wrong points, and the master unit alone gets 1 wrong.")),
          ("red", t("題 2", "Q2"), t("若同一個輸入出現矛盾標籤，新增再多節點，兩筆都能分對嗎？", "If one input carries contradictory labels, can adding any number of units get both right?"), q2, t("不能：不可能同時對 A 與 B。", "No: it cannot be right for both A and B."), t("核對：和停止的條件 1 一致：標籤不能互相矛盾。", "Check: consistent with condition 1: labels must not contradict."))]
    for i, (role, name, q, rows_, ans, chk) in enumerate(qs):
        x = i * 590; o += card(x, 0, 570, 470, role) + head(x, 0, 570, role, name, 36, 18)
        o += txt(x + 16, 50, 538, q, 19, TXT, True, "left", 84)
        o += chip(x + 16, 146, 130, "grey", t("解題示範", "Solution"), False, 30, 16)
        o += card(x + 16, 190, 538, 150, "grey", .1)
        for j, s_ in enumerate(rows_): o += txt(x + 28, 202 + j * 42 + (21 if len(rows_) == 2 else 0), 514, s_, 18, TXT, False, "left", 30)
        o += chip(x + 16, 356, 538, role, ans, True, 42, 19)
        o += txt(x + 16, 410, 538, chk, 16, MUT, False, "left", 48)
    return o

PAGES_DESIGN = [(f"p{i}", globals()[f"page{i}"]) for i in range(1, 11)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:]
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp05_" + fid, fn, W, H, out)
