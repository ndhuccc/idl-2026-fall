import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *
from lec01_kp03_nums import *

KP = "KP03"; N = 9
RIGHT = "#0ea5e9"

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    def apple(cx, cy, r, z=3):
        g.ellipse(cx, cy, r, r * 0.93, fill="#dc2626", stroke="#991b1b", sw=3, z=z, gradientTo="#f87171", gradientType="radial")
        g.line([(cx, cy - r * 0.9), (cx + 6, cy - r * 1.25)], "#78350f", sw=6, z=z + 1); g.poly([(cx + 6, cy - r * 1.15), (cx + r * 0.6, cy - r * 1.4), (cx + r * 0.75, cy - r * 1.05), (cx + 12, cy - r * 0.95)], fill="#22c55e", stroke="#15803d", sw=2, z=z + 1)
    def cucumber(cx, cy, w, h, z=3):
        g.rect(cx - w / 2, cy - h / 2, w, h, fill="#16a34a", stroke="#14532d", sw=3, radius=h / 2, z=z, gradientTo="#4ade80", gradientAngle=0)
        for k in range(5): g.ellipse(cx - w / 2 + (k + 1) * w / 6, cy, 3, fill="#bbf7d0", z=z + 1)

    # ============ 1:同一顆水果,兩種問法 ============
    g = new(1, T("KP03 標籤決定分類或回歸", "KP03 The label decides: classification or regression"))
    g.beat = 1
    card(g, 50, 120, 570, 500, BLUE); head(g, 70, 126, 530, T("問法 1：它是哪一類？", "Question 1: which kind is it?"), BLUE)
    pill(g, 460, 132, 140, T("比喻 · 示意", "Analogy"), GRAY, size=16)
    apple(335, 340, 90)
    for i, (nm, on) in enumerate((("apple", 1), ("cucumber", 0), ("pear", 0))):
        x = 100 + i * 170; g.rect(x, 500, 150, 70, fill=BLUEL if on else "#f1f5f9", stroke=BLUE if on else GRAY, sw=4 if on else 2, radius=12, z=3)
        g.text(x, 500, 150, 70, T(["蘋果", "黃瓜", "梨子"][i], nm), size=24, color=INK if on else MUTED, bold=bool(on), align="center", vcenter=True, z=8)
    g.beat = 2
    card(g, 660, 120, 570, 500, ORG); head(g, 680, 126, 530, T("問法 2：直徑是多少？", "Question 2: how wide is it?"), ORG)
    apple(945, 340, 90)
    g.line([(855, 470), (1035, 470)], ORG, sw=5, z=4); g.line([(855, 458), (855, 482)], ORG, sw=5, z=4); g.line([(1035, 458), (1035, 482)], ORG, sw=5, z=4)
    g.rect(880, 500, 130, 70, fill=ORGL, stroke=ORG, sw=4, radius=12, z=3); g.text(880, 500, 130, 70, r"$7.2$ cm", size=28, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.beat = 3
    banner(g, T("同樣的輸入，要的答案型態不同", "Similar inputs, different kinds of answer"))
    pages.append((g, T("標籤決定任務", "The label decides the task"), {
        0: T("問水果「是哪一類」與「直徑是多少」，可以看相似的輸入，卻需要不同的答案。（水果圖是比喻。）", "Asking which fruit it is and asking its diameter can use similar inputs but need different answers. (The fruit is an analogy.)"),
        1: T("第一種問法：從有限的類別裡選一個。", "The first question: choose one from a finite set of categories."),
        2: T("第二種問法：回答一個量測出來的數值，例如 7.2 公分。", "The second: answer a measured value, such as 7.2 cm."),
        3: T("類名有多少種，不等於直徑能取多少個值。判斷任務要看標籤的意義。", "The number of category names does not count the possible diameters. The label's meaning identifies the task.")}))

    # ============ 2:監督式 = 輸入配標籤 ============
    g = new(2, T("監督式學習：每筆輸入都配一個標籤", "Supervised learning: every input comes with a label"))
    g.beat = 1
    card(g, 90, 110, 1100, 120, GRAY)
    g.text(110, 114, 1060, 112, r"$\mathcal D=\{(\mathbf x_i,y_i)\}_{i=1}^{n}$", size=46, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    for i in range(3):
        x = 130 + i * 360
        g.rect(x, 260, 150, 150, fill=BLUEL, stroke=BLUE, sw=3, radius=14, z=3); g.text(x, 260, 150, 80, rf"$\mathbf x_{i + 1}$", size=34, color=INK, align="center", vcenter=True, z=8)
        g.text(x, 330, 150, 60, T("輸入", "input"), size=20, color=BLUE, bold=True, align="center", z=8)
        arrow(g, (x + 156, 335), (x + 206, 335), GRAY, sw=4, head=14)
        g.rect(x + 210, 260, 130, 150, fill=ORGL, stroke=ORG, sw=3, radius=14, z=3); g.text(x + 210, 260, 130, 80, rf"$y_{i + 1}$", size=34, color=INK, align="center", vcenter=True, z=8)
        g.text(x + 210, 330, 130, 60, T("標籤", "label"), size=20, color=ORG, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(130, 450, 1020, 150, fill=CARD, stroke=GRAY, sw=2, radius=14, z=2)
    g.text(150, 480, 980, 100, T("標籤不必是文字：\n可以是類別名稱，也可以是量測到的數值", "A label need not be text:\nit can be a category name or a measured value"), size=28, color=INK, bold=True, align="center", z=8)
    banner(g, T("有標籤，才有「答案」可以拿來訓練", "With labels there is an “answer” to train on"))
    pages.append((g, T("輸入與標籤成對", "Inputs paired with labels"), {
        0: T("監督式學習使用輸入與目標成對的資料。目標又叫標籤。", "Supervised learning uses paired inputs and targets. A target is also called a label."),
        1: T("每一筆輸入 x，都配一個標籤 y。", "Each input x comes with a label y."),
        2: T("標籤不必是文字：可以是類別名稱，也可以是量測到的數值。", "A label need not be text: it can be a category name or a measured value."),
        3: T("有標籤，才有答案可以拿來訓練。", "Labels give us answers to train on.")}))

    # ============ 3:分類 vs 回歸 輸出型態 ============
    g = new(3, T("分類選一個類別，回歸給一個實數", "Classification picks a class; regression, a number"))
    g.beat = 1
    card(g, 50, 130, 570, 470, BLUE); head(g, 70, 136, 530, T("分類：有限類別之一", "Classification: one of finitely many classes"), BLUE, 22)
    g.text(70, 194, 530, 72, r"$f_\theta^{\mathrm{cls}}:\mathbb R^d\to\{\ell_1,\ldots,\ell_m\}$", size=32, color=INK, align="center", vcenter=True, z=8)
    for i in range(4):
        x = 100 + i * 125; hi = i == 1
        g.ellipse(x + 50, 380, 48, fill=BLUEL if not hi else BLUE, stroke=BLUE, sw=4, z=3); g.text(x, 340, 100, 80, rf"$\ell_{i + 1}$" if i < 3 else r"$\ell_m$", size=30, color=INK if not hi else "#ffffff", bold=True, align="center", vcenter=True, z=8)
    star(g, 275, 305, 16, GOLD, z=6, stroke="#b45309", sw=2); g.text(100, 480, 470, 90, T("只能選其中 1 個\n（這裡示意選到 $\\ell_2$）", "choose exactly 1\n(here $\\ell_2$ is picked)"), size=22, color=INK, bold=True, align="center", z=8)
    g.beat = 2
    card(g, 660, 130, 570, 470, ORG); head(g, 680, 136, 530, T("回歸：一個實數", "Regression: one real number"), ORG, 22)
    g.text(680, 194, 530, 72, r"$f_\theta^{\mathrm{reg}}:\mathbb R^d\to\mathbb R$", size=32, color=INK, align="center", vcenter=True, z=8)
    g.line([(700, 400), (1190, 400)], ORG, sw=5, z=4, arrow="end", head=20)
    for t in range(0, 9):
        g.line([(720 + t * 55, 392), (720 + t * 55, 408)], ORG, sw=3, z=4)
    g.ellipse(720 + 3.6 * 55, 400, 14, fill=ORG, stroke="#ffffff", sw=3, z=6); g.text(745, 330, 260, 40, r"$7.2$ cm", size=26, color=ORG, bold=True, align="center", vcenter=True, z=8)
    g.text(690, 480, 510, 90, T("連續的數線上，\n任一位置都可能", "on a continuous line,\nany position is possible"), size=22, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    banner(g, T("輸出型態：有限類別，還是實數？", "Output type: a finite set of classes, or a real number?"))
    pages.append((g, T("兩種輸出型態", "Two output types"), {
        0: T("正式地說：分類把輸入映射到 m 個類別之一；這裡的回歸映射到一個實數。", "Formally: classification maps an input to one of m classes; the regression here maps to one real number."),
        1: T("分類：從有限個類別裡選一個。這裡示意選到第二類。", "Classification: pick one of finitely many classes; here the second is picked as a sketch."),
        2: T("回歸：輸出是數線上的一個實數，例如 7.2 公分。", "Regression: the output is a real number on a line, such as 7.2 cm."),
        3: T("兩者可以看同樣的輸入特徵，仍然不是同一件事。", "They can use the same input features and still be different tasks.")}))

    # ============ 4:3 筆例子 ============
    g = new(4, T("類別與直徑，不是同一把尺", "Categories and diameters are not one scale"))
    rows = [("apple", T("蘋果", "apple"), T("紅、大", "red, large"), 7.2, BLUE), ("cucumber", T("黃瓜", "cucumber"), T("綠、長", "green, elongated"), 6.8, GRN), ("apple", T("蘋果", "apple"), T("紅、中", "red, medium"), 6.5, BLUE)]
    g.beat = 1
    pill(g, 1030, 96, 200, T("假想教學樣本", "hypothetical samples"), GRAY, size=16)
    for i, (kind, nm, desc, dia, col) in enumerate(rows):
        y = 140 + i * 150
        g.rect(50, y, 1180, 130, fill=CARD, stroke=GRAY, sw=2, radius=14, z=1)
        g.text(70, y, 60, 130, str(i + 1), size=36, color=MUTED, bold=True, align="center", vcenter=True, z=8)
        if kind == "apple":
            g.ellipse(190, y + 66, 48, 45, fill="#dc2626", stroke="#991b1b", sw=3, z=3, gradientTo="#f87171", gradientType="radial"); g.poly([(194, y + 24), (222, y + 12), (230, y + 30), (200, y + 34)], fill="#22c55e", stroke="#15803d", sw=2, z=4)
        else:
            g.rect(130, y + 36, 120, 56, fill="#16a34a", stroke="#14532d", sw=3, radius=28, z=3, gradientTo="#4ade80", gradientAngle=0)
        g.text(280, y + 10, 220, 50, desc, size=24, color=INK, bold=True, vcenter=True, z=8)
        g.rect(280, y + 70, 160, 44, fill=BLUEL if kind == "apple" else GRNL, stroke=col, sw=3, radius=22, z=3); g.text(280, y + 70, 160, 44, nm, size=22, color=col, bold=True, align="center", vcenter=True, z=8)
    g.beat = 2
    for i, (_, _, _, dia, col) in enumerate(rows):
        y = 140 + i * 150
        g.rect(560, y + 40, dia * 62, 50, fill=ORGL, stroke=ORG, sw=3, radius=6, z=3)
        g.text(560 + dia * 62 + 12, y + 40, 150, 50, rf"${dia:.1f}$ cm", size=26, color=ORG, bold=True, vcenter=True, z=8)
    g.text(560, 590 - 8, 540, 30, T("直徑（柱長依比例）", "diameter (bars to scale)"), size=18, color=MUTED, z=8)
    g.beat = 3
    g.rect(990, 296, 220, 40, fill=REDL, stroke=RED, sw=3, radius=10, z=5); g.text(990, 296, 220, 40, T("第 2 筆其實是黃瓜", "sample 2 is a cucumber"), size=18, color=RED, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("類別是名字；直徑的差值有公分的意義", "A class is a name; a diameter difference is meaningful in cm"))
    pages.append((g, T("類別與直徑", "Class vs diameter"), {
        0: T("看講義的例子：3 筆假想教學樣本，保留原數字。描述詞還需要編碼，不能直接當數值向量。注意：來源標題說是 3 顆蘋果，但第 2 筆其實標成黃瓜。", "The note's example: 3 hypothetical teaching samples with the original numbers. Descriptions need encoding before they become vectors. Note: the source calls them 3 apples, but the second is labeled a cucumber."),
        1: T("右邊是它們的直徑：7.2、6.8、6.5 公分，柱長依比例。", "On the right are the diameters: 7.2, 6.8 and 6.5 cm, bars to scale."),
        2: T("類別 1、類別 2 只是名字，不能解讀成第二類是第一類的兩倍。直徑的差值，例如 7.2 減 6.5 等於 0.7，才有公分的意義。", "Class 1 and class 2 are names, not a claim that one is twice the other. A diameter difference such as 7.2 minus 6.5 equals 0.7 cm is meaningful."),
        3: T("標籤即使寫成數字，也要先問它是編號還是量測值。", "Even when a label is written as a number, ask whether it is an identifier or a measurement.")}))

    # ============ 5:編號 vs 量測 ============
    g = new(5, T("標籤寫成數字，也要先問：是編號還是量測？", "A numeric label: identifier or measurement?"))
    g.beat = 1
    card(g, 50, 130, 570, 470, RED); head(g, 70, 136, 530, T("編號（只是名字）", "Identifier (just a name)"), RED, 24)
    for i, (nm, cl) in enumerate(((r"$\ell_1$", BLUEL), (r"$\ell_2$", ORGL))):
        g.ellipse(210 + i * 200, 300, 60, fill=cl, stroke=INK, sw=3, z=3); g.text(150 + i * 200, 260, 120, 80, nm, size=36, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.text(100, 400, 470, 50, rf"$\ell_2=2\times\ell_1$ ？", size=32, color=INK, align="center", vcenter=True, z=8); cross(g, 290, 470, 50, RED, z=8)
    g.text(100, 540, 470, 40, T("第二類不是第一類的 2 倍", "class 2 is not twice class 1"), size=20, color=RED, bold=True, align="center", z=8)
    g.beat = 2
    card(g, 660, 130, 570, 470, GRN); head(g, 680, 136, 530, T("量測（差值有單位意義）", "Measurement (differences carry units)"), GRN, 24)
    for i, (v, nm) in enumerate(((7.2, "A"), (6.5, "B"))):
        g.rect(720, 220 + i * 90, v * 56, 60, fill=GRNL, stroke=GRN, sw=3, radius=6, z=3); g.text(720 + v * 56 + 10, 220 + i * 90, 120, 60, rf"${v:.1f}$ cm", size=24, color=INK, bold=True, vcenter=True, z=8)
    g.text(700, 410, 500, 50, r"$7.2-6.5=0.7$ cm", size=32, color=INK, align="center", vcenter=True, z=8); tick(g, 930, 480, 46, GRN, z=8)
    g.beat = 3
    banner(g, T("同樣是數字：編號不能做算術，量測可以", "Both are numbers: identifiers don't support arithmetic, measurements do"))
    pages.append((g, T("編號與量測", "Identifier vs measurement"), {
        0: T("類別編號只是名字：第二類不是第一類的兩倍，編號不能做算術。", "A class identifier is just a name: class 2 is not twice class 1, so identifiers don't support arithmetic."),
        1: T("量測值不同：直徑 7.2 減 6.5 等於 0.7 公分，差值有單位意義。", "A measurement differs: 7.2 minus 6.5 equals 0.7 cm, and the difference carries units."),
        2: T("所以標籤寫成數字時，要先問它是編號還是量測值。", "So when a label is a number, first ask whether it is an identifier or a measurement."),
        3: T("（直徑都是講義的假想教學樣本，不是真實量測。）", "(The diameters are the note's hypothetical teaching samples, not real measurements.)")}))

    # ============ 6:同一分數,12 點 ============
    g = new(6, T("同一個分數：看正負，或保留大小", "One score: take its sign, or keep its size"))
    g.beat = 1
    card(g, 50, 120, 560, 500, BLUE); head(g, 70, 126, 520, T(r"分數 $s=x_1+x_2-5$", r"Score $s=x_1+x_2-5$"), BLUE)
    px0, py0, S = 110, 560, 38
    g.line([(px0, py0), (px0 + 10 * S + 10, py0)], MUTED, sw=3, z=3); g.line([(px0, py0), (px0, py0 - 10 * S - 10)], MUTED, sw=3, z=3)
    for t in (0, 5, 10): g.text(px0 + t * S - 20, py0 + 4, 40, 26, str(t), size=16, color=MUTED, align="center", z=8); g.text(px0 - 44, py0 - t * S - 14, 36, 26, str(t), size=16, color=MUTED, align="right", z=8)
    g.text(px0 + 10 * S + 14, py0 - 38, 60, 30, r"$x_1$", size=22, color=MUTED, z=8); g.text(px0 + 10, py0 - 10 * S - 12, 60, 30, r"$x_2$", size=22, color=MUTED, z=8)
    g.beat = 2
    g.poly([(px0, py0 - 5 * S), (px0 + 5 * S, py0), (px0, py0)], fill="rgba(29,78,216,0.10)", z=2); g.line([(px0, py0 - 5 * S), (px0 + 5 * S, py0)], RED, sw=5, z=4)
    g.text(px0 + 4.2 * S, py0 - 3.4 * S, 260, 54, T(r"邊界 $x_1+x_2=5$", r"boundary $x_1+x_2=5$"), size=20, color=RED, bold=True, z=8)
    for (x, y), c in zip(pts, cls): g.ellipse(px0 + x * S, py0 - y * S, 10, fill=BLUE if c == "l1" else ORG, stroke="#ffffff", sw=2, z=5)
    g.beat = 3
    card(g, 650, 120, 580, 500, GRN); head(g, 670, 126, 540, T("保留大小：12 個分數", "Keep the size: 12 scores"), GRN)
    bx0, zero, bu = 700, 400, 17
    g.line([(bx0 - 10, zero), (bx0 + 12 * 41, zero)], MUTED, sw=3, z=3)
    for i, v in enumerate(s):
        h = abs(v) * bu; col = BLUE if v < 0 else ORG
        g.rect(bx0 + i * 41, zero - (h if v >= 0 else 0), 32, max(h, 3), fill=col, stroke="#ffffff", sw=1, radius=3, z=4)
        g.text(bx0 + i * 41 - 4, zero + 150 if False else 560, 40, 24, str(i + 1), size=16, color=MUTED, align="center", z=8)
    g.text(bx0 + 0 * 41 - 20, zero + 1.5 * bu + 6, 80, 26, f(s[0]), size=16, color=BLUE, bold=True, align="center", z=8)
    g.text(bx0 + 6 * 41 - 16, zero - s[6] * bu - 28, 60, 26, f(s[6]), size=16, color=ORG, bold=True, align="center", z=8)
    g.text(670, 585, 540, 26, T("樣本 1 到 12 的分數；橫線是 0", "scores of samples 1 to 12; the line is 0"), size=16, color=MUTED, align="center", z=8)
    banner(g, T("同一組 12 點、同一組參數：只有輸出規則不同", "Same 12 points, same parameters: only the output rule differs"))
    pages.append((g, T("同一個分數", "One score"), {
        0: T("換成講義的 12 個無單位教學點，驗算運算。沿用 KP02 的加權和，權重都是 1，偏移 b 設成負 5。偏移從預設 0 改成負 5，只是為了在同一片點裡看到正負分數，不是訓練結果。", "Now the note's 12 unitless points. Reuse KP02's weighted sum with both weights 1 and offset b set to minus 5. Changing b from 0 to minus 5 only exposes both signs; it is not a training result."),
        1: T("分類看分數的正負：邊界是 x1 加 x2 等於 5。分數不小於 0 的點歸第二類，小於 0 的歸第一類。", "Classification looks at the sign of the score: the boundary is x1 plus x2 equals 5. Non-negative scores go to class 2, negative to class 1."),
        2: T("12 個點裡，只有第 1 筆的分數是負 1.5，歸第一類；其他 11 筆都不小於 0。落在邊界上的兩點，分數是 0，歸第二類。", "Of the 12 points only the first has a negative score, minus 1.5, so it is class 1; the other 11 are non-negative. The two points on the boundary have score 0 and go to class 2."),
        3: T("回歸則保留分數本身：右邊是 12 個分數的大小。這些不是真實水果的標籤。", "Regression keeps the score itself: on the right are the 12 scores. These are not real fruit labels.")}))

    # ============ 7:兩個代入 ============
    g = new(7, T("同一個分數，兩條輸出規則", "One score, two output rules"))
    g.beat = 1
    for i, ((x, y), idx, col, light) in enumerate(((pts[0], 0, BLUE, BLUEL), (pts[6], 6, ORG, ORGL))):
        X0 = 50 + i * 600
        card(g, X0, 104, 580, 232, col); head(g, X0 + 20, 108, 540, T(rf"第 {idx + 1} 筆：$({x},{y})$", rf"Sample {idx + 1}: $({x},{y})$"), col, 22)
        g.text(X0 + 20, 152, 540, 60, rf"$s({x},{y})={x}+{y}-5={x + y}-5={f(s[idx])}$", size=22, color=INK, align="center", vcenter=True, z=8)
        g.rect(X0 + 40, 232, 240, 90, fill=light, stroke=col, sw=3, radius=12, z=3); g.text(X0 + 40, 234, 240, 30, T("分類輸出", "classification"), size=18, color=col, bold=True, align="center", vcenter=True, z=8)
        g.text(X0 + 40, 262, 240, 56, r"$\ell_1$" if i == 0 else r"$\ell_2$", size=34, color=INK, bold=True, align="center", vcenter=True, z=8)
        g.rect(X0 + 300, 232, 240, 90, fill=GRNL, stroke=GRN, sw=3, radius=12, z=3); g.text(X0 + 300, 234, 240, 30, T("回歸輸出", "regression"), size=18, color=GRN, bold=True, align="center", vcenter=True, z=8)
        g.text(X0 + 300, 262, 240, 56, rf"${f(s[idx])}$", size=34, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.beat = 2
    ax0, ax1, ay = 140, 1140, 438; sc = (ax1 - ax0) / 14.0
    X = lambda v: ax0 + (v + 2) * sc
    g.line([(ax0, ay), (ax1, ay)], MUTED, sw=4, z=3, arrow="end", head=18)
    for t in range(-2, 13, 2): g.line([(X(t), ay - 7), (X(t), ay + 7)], MUTED, sw=3, z=3); g.text(X(t) - 24, ay + 12, 48, 26, str(t), size=16, color=MUTED, align="center", z=8)
    g.poly([(ax0, ay - 40), (X(0), ay - 40), (X(0), ay), (ax0, ay)], fill="rgba(29,78,216,0.12)", z=2); g.poly([(X(0), ay - 40), (ax1, ay - 40), (ax1, ay), (X(0), ay)], fill="rgba(194,65,12,0.12)", z=2)
    g.line([(X(0), ay - 60), (X(0), ay + 10)], RED, sw=5, z=4)
    g.text(X(0) - 120, ay - 98, 240, 32, T("分類邊界：$s=0$", "boundary: $s=0$"), size=20, color=RED, bold=True, align="center", z=8)
    g.text(ax0 + 20, ay - 34, 250, 30, T(r"$s<0$ → $\ell_1$", r"$s<0$ → $\ell_1$"), size=18, color=BLUE, bold=True, z=8); g.text(X(0) + 20, ay - 34, 300, 30, T(r"$s\ge0$ → $\ell_2$", r"$s\ge0$ → $\ell_2$"), size=18, color=ORG, bold=True, z=8)
    g.ellipse(X(s[0]), ay, 13, fill=BLUE, stroke="#ffffff", sw=3, z=6); g.text(X(s[0]) - 90, ay + 40, 180, 30, T("第 1 筆 −1.5", "sample 1: −1.5"), size=18, color=BLUE, bold=True, align="center", z=8)
    g.ellipse(X(s[6]), ay, 13, fill=ORG, stroke="#ffffff", sw=3, z=6); g.text(X(s[6]) - 90, ay + 40, 180, 30, T("第 7 筆 10.5", "sample 7: 10.5"), size=18, color=ORG, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(110, 528, 1060, 100, fill=CARD, stroke=GRAY, sw=2, radius=14, z=2)
    g.text(130, 530, 1020, 96, r"$f_\theta^{\mathrm{cls}}(\mathbf x)=\begin{cases}\ell_2,&s(\mathbf x)\ge0\\\ell_1,&s(\mathbf x)<0\end{cases}\qquad f_\theta^{\mathrm{reg}}(\mathbf x)=s(\mathbf x)$", size=26, color=INK, align="center", vcenter=True, z=8)
    pill(g, 1020, 346, 210, T("不是真實水果標籤", "not real fruit labels"), GRAY, size=16)
    banner(g, T("這些數字不能拿來算辨識正確率或直徑誤差", "These values cannot give accuracy or diameter error"))
    pages.append((g, T("兩條輸出規則", "Two output rules"), {
        0: T("第 1 筆 (1.5, 2.0)：分數是 1.5 加 2.0 減 5，等於負 1.5。分類輸出第一類，回歸輸出負 1.5。第 7 筆 (8.0, 7.5)：分數是 10.5，分類輸出第二類，回歸輸出 10.5。", "Sample 1 (1.5, 2.0): the score is 1.5 plus 2.0 minus 5, equal to minus 1.5. Classification outputs class 1, regression outputs minus 1.5. Sample 7 (8.0, 7.5): score 10.5, class 2, regression 10.5."),
        1: T("把兩個分數放在數線上：0 是分類邊界，左邊歸第一類，右邊歸第二類。", "Put the two scores on a line: 0 is the classification boundary, class 1 on the left and class 2 on the right."),
        2: T("兩條規則：分類是分數不小於 0 選第二類、小於 0 選第一類；回歸直接輸出分數。這些不是實際水果的標籤，也不是公分的預測，不能拿來算辨識正確率或直徑誤差。", "Two rules: classification picks class 2 for a non-negative score and class 1 for a negative one; regression outputs the score. These are not observed fruit labels or cm predictions, so they cannot establish accuracy or diameter error."),
        3: T("所有數值由程式重算。", "All values were computed by program.")}))

    # ============ 8:機率不等於回歸 ============
    g = new(8, T("看見機率，不等於任務就是回歸", "Seeing a probability does not make it regression"))
    pill(g, 1090, 100, 140, T("講義補充", "Added note"), GRAY, size=16)
    g.beat = 1
    boxes = [(80, T("輸入", "input"), BLUE, BLUEL, r"$\mathbf x$"), (390, T("內部分數／機率", "score / probability"), ORG, ORGL, r"$0.8$"), (700, T("選出類別", "chosen class"), GRN, GRNL, r"$\ell_2$")]
    for i, (x, nm, col, light, sym) in enumerate(boxes):
        if i == 0: g.beat = 1
        if i == 1: g.beat = 2
        if i == 2: g.beat = 3
        g.rect(x, 170, 240, 220, fill=light, stroke=col, sw=3, radius=16, z=3); g.text(x, 180, 240, 50, nm, size=20, color=col, bold=True, align="center", vcenter=True, z=8); g.text(x, 240, 240, 130, sym, size=56, color=INK, bold=True, align="center", vcenter=True, z=8)
        if i == 1:
            g.rect(x + 30, 355, 180, 14, fill="#e2e8f0", radius=7, z=4); g.rect(x + 30, 355, 180 * 0.8, 14, fill=ORG, radius=7, z=5); g.ellipse(x + 30 + 180 * 0.8, 362, 11, fill=CARD, stroke=ORG, sw=4, z=6)
        if i == 0:
            for k in range(3): g.ellipse(x + 60 + k * 60, 372, 9, fill=BLUE, z=5)
        if i == 2:
            star(g, x + 120, 372, 13, GOLD, z=6, stroke="#b45309", sw=2)
        if i < 2: arrow(g, (x + 250, 280), (x + 300, 280), GRAY)
    g.beat = 3
    g.rect(980, 170, 250, 220, fill=CARD, stroke=GRAY, sw=2, radius=16, z=2, dash=True); g.text(990, 215, 230, 140, T("任務看的是：\n最後要預測什麼", "The task depends on\nwhat is finally predicted"), size=20, color=INK, bold=True, align="center", z=8)
    g.text(80, 440, 1120, 80, T("二元事件的機率雖然是連續數值，也可以從「有沒有發生」的標籤估計出來", "A probability of a binary event is a continuous value, yet it can be estimated from “did it happen” labels"), size=26, color=INK, bold=True, align="center", z=8)
    g.text(80, 540, 1120, 60, T("先寫出要預測的目標，再說輸出怎麼用", "State the target first, then how the output is used"), size=24, color=MUTED, align="center", z=8)
    banner(g, T("內部算出數字，不等於最後的輸出規則", "Numbers inside do not fix the final output rule"))
    pages.append((g, T("機率與任務", "Probability and task"), {
        0: T("常見迷思：看到內部的數值，就以為任務一定是回歸。這一頁是講義的補充。分類器可以先算類別分數或機率，再選出類別。（0.8 是示意，不是講義的數字。）", "Common misconception: seeing an internal value and concluding the task is regression. This page is a supplement. A classifier can compute class scores or probabilities, then choose a class. (0.8 is a sketch, not the note's number.)"),
        1: T("輸入先算出一個內部分數或機率。", "The input first yields an internal score or probability."),
        2: T("再依規則選出類別。二元事件的機率雖然是連續數值，也可以從有事件標籤的分類資料估計，所以來源說降雨機率必是回歸，說得太武斷。", "Then a rule chooses the class. A binary-event probability is continuous yet can be estimated from categorical event labels, so the claim that rain probability must be regression is too strong."),
        3: T("判斷時，先寫出要預測的目標，再說輸出如何使用。把直徑分箱會變成另一個分類任務，評分規則與資訊粒度都改變了。分類與回歸是常見的監督任務，不是全部。", "State the target, then how the output is used. Binning diameters makes a different classification task with a different scoring rule and resolution. Classification and regression are common supervised tasks, not all of them.")}))

    # ============ 9:拿掉標籤 ============
    g = new(9, T("如果拿掉標籤，只剩輸入呢？", "What if we remove the labels and keep only inputs?"))
    g.beat = 1
    card(g, 50, 130, 520, 470, BLUE); head(g, 70, 136, 480, T("有標籤：輸入 + 答案", "Labeled: input + answer"), BLUE, 24)
    for i in range(4):
        y = 210 + i * 90
        g.rect(90, y, 160, 70, fill=BLUEL, stroke=BLUE, sw=3, radius=10, z=3); g.text(90, y, 160, 70, rf"$\mathbf x_{i + 1}$", size=28, color=INK, align="center", vcenter=True, z=8)
        arrow(g, (254, y + 35), (296, y + 35), GRAY, sw=4, head=12); g.rect(300, y, 220, 70, fill=ORGL, stroke=ORG, sw=3, radius=10, z=3); g.text(300, y, 220, 70, rf"$y_{i + 1}$", size=28, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    arrow(g, (580, 365), (700, 365), ORG, sw=6, head=22); g.text(570, 300, 140, 50, T("拿掉標籤", "drop labels"), size=22, color=ORG, bold=True, align="center", z=8)
    card(g, 710, 130, 520, 470, GRAY); head(g, 730, 136, 480, T("沒有標籤：只剩輸入", "Unlabeled: inputs only"), MUTED, 24)
    for i in range(4):
        y = 210 + i * 90
        g.rect(780, y, 160, 70, fill=BLUEL, stroke=BLUE, sw=3, radius=10, z=3); g.text(780, y, 160, 70, rf"$\mathbf x_{i + 1}$", size=28, color=INK, align="center", vcenter=True, z=8)
        arrow(g, (944, y + 35), (986, y + 35), GRAY, sw=4, head=12, dash=True); g.rect(990, y, 180, 70, fill="#f1f5f9", stroke=GRAY, sw=3, radius=10, dash=True, z=3); g.text(990, y, 180, 70, "?", size=34, color=RED, bold=True, align="center", vcenter=True, z=8)
    mascot(g, 1130, 430, 0.0 + 0.0 + 0.0 + 0.4, "confused") if False else None
    g.beat = 3
    banner(g, T(r"下一站 KP04：沒有答案時，資料本身有什麼結構？$\to$", r"Next, KP04: with no answers, what structure is in the data? $\to$"))
    pages.append((g, T("拿掉標籤", "Remove the labels"), {
        0: T("兩種任務的共同點，是有目標可以用來訓練。", "What the two tasks share is a target to train on."),
        1: T("如果拿掉標籤，只保留輸入，會發生什麼？", "What if we remove the labels and keep only the inputs?"),
        2: T("沒有答案可以對照了。下一個單元就要改問：資料本身有什麼結構？", "There is no answer to compare against. The next unit asks what structure the data itself has."),
        3: T("講義的 4 題小練習留在講義裡。", "The note's practice questions stay in the note.")}))
    return pages
