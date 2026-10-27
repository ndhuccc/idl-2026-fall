"""KP01 pages designed in HTML (view-from-text-v3, route A). pageN(lang) -> body HTML of a 1160x480 figure."""
import os, sys
SK = os.environ.get("COURSE_SKILLS") or (os.path.join(os.path.abspath(os.environ["COURSE_REPO"]), ".agents", "skills") if os.environ.get("COURSE_REPO") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, SK + "/view-from-text-v3/examples"); sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, HERE)
from kp02_pages import *          # helpers: Axes, region, poly_svg, line_svg, arrow, axes_svg, mark, ring, txt, card, ...
import kp01_nums as N
W, H = 1160, 480
def plane(ax, truth, line=None, shade=None, wrong=(), r=13, labels=False):
    """Axes + (optional) shaded positive side + line + the four points coloured by `truth` (1 = A)."""
    s = ""
    if shade: s += poly_svg(ax, region(ax, [shade]), BL, .26)
    s += axes_svg(ax, [0, 1], [0, 1], "x₁", "x₂")
    if line: s += line_svg(ax, *line)
    for p, v in zip(N.PTS, truth): s += mark(ax, p[0], p[1], "A" if v else "B", r)
    for p in wrong: s += ring(ax, p[0], p[1], r + 10)
    return s
PRED = lambda w, strict=True: N.preds(w, strict)
def ab(v): return "A" if v else "B"
# ------------------------------------------------ 1: three labelings
def page1(lang):
    t = tr(lang); o = ""
    items = [("OR", t("只要有一個輸入為 1 就輸出 1", "outputs 1 if any input is 1"), N.OR, "blue"), ("AND", t("兩個輸入都為 1 才輸出 1", "outputs 1 only if both are 1"), N.AND, "teal"), ("XOR", t("兩個輸入不同才輸出 1", "outputs 1 only if inputs differ"), N.XOR, "orange")]
    for i, (name, desc, tv, role) in enumerate(items):
        x0 = i * 392; o += card(x0, 0, 376, 336, role) + head(x0, 0, 376, role, name, 36, 18)
        o += txt(x0 + 8, 42, 360, desc, 16, TXT, False, "center", 24)
        ax = Axes(x0 + 94, 74, 188, 188, XR, XR); o += svg(0, 0, W, H, plane(ax, tv, r=12))
        o += chip(x0 + 88, 300, 200, role, t("輸出 ", "outputs ") + " ".join(str(v) for v in tv), True, 30, 18)
    o += card(0, 348, 570, 124, "orange", .14) + named(10, 354, 540, t("標籤：每個點要分到的類別：<br>A（圓點，輸出 1）或 B（方點，輸出 0）。", "Label: the class a point must go to:<br>A (circle, output 1) or B (square, output 0)."), "term:" + t("標籤", "Label"), 16, TXT, 48)
    o += named(10, 414, 540, t("互斥或 XOR：兩個輸入不同才輸出 1，相同輸出 0。", "Exclusive OR (XOR): outputs 1 only if the two<br>inputs differ, otherwise 0."), "term:" + t("互斥或", "Exclusive OR"), 16, TXT, 48)
    o += card(590, 348, 570, 124, "yellow", .14) + txt(600, 356, 550, t("三格的四個點座標完全相同，只有標籤不同。<br>輸出依 (0,0)、(0,1)、(1,0)、(1,1) 的順序：<br>OR 為 0111、AND 為 0001、XOR 為 0110。〔程式計算〕", "The four points are identical; only the labels differ.<br>Outputs in the order (0,0), (0,1), (1,0), (1,1):<br>OR is 0111, AND is 0001, XOR is 0110.<br>(computed)"), 16, TXT, True, "center", 96)
    return o
# ------------------------------------------------ 2: score, rule, OR line
def page2(lang):
    t = tr(lang); o = ""; w = N.LINES["or"]; sc = [N.score(*w, p) for p in N.PTS]
    o += cardH(0, 0, 620, 480, "blue", t("每個點打一個分數，分數為正就判 A", "Give every point a score: positive means class A"))
    o += card(14, 50, 592, 70, "blue", .16) + formula(14, 58, 592, r"g(\mathbf{x})=w_1x_1+w_2x_2+w_0", 26, 54)
    o += card(14, 130, 592, 110, "blue", .16) + formula(14, 136, 592, r"\widehat y=1\ (A),\quad g(\mathbf{x})>0", 22, 44) + formula(14, 186, 592, r"\widehat y=0\ (B),\quad g(\mathbf{x})\leq 0", 22, 44)
    o += named(14, 252, 580, t("分數：把點代入 g 得到的數；正負決定點在線的哪一側。<br>剛好等於 0 也判 B，不能看完真實標籤再改規則。", "Score: the number you get by putting a point into g;<br>its sign says which side of the line it is on. Exactly 0<br>is also B, and the rule may not change after seeing<br>the labels."), "term:" + t("分數", "Score"), 16, TXT, 96)
    o += named(14, 352, 580, t("線性可分：存在一組係數 w₁、w₂、w₀，能把所有點依這條規則分對。<br>重點是「存在」，不只是目前找到的那一組。", "Linearly separable: there exists a set of coefficients<br>w₁, w₂, w₀ that classifies every point correctly by this<br>rule. The point is existence, not just the set found."), "term:" + t("線性可分", "Linearly separable"), 16, TXT, 96)
    o += cardH(636, 0, 524, 480, "teal", t("例：OR 取 g = x₁ + x₂ − 0.5", "Example: OR with g = x₁ + x₂ − 0.5"))
    ax = Axes(700, 82, 300, 300, XR, XR)
    s = plane(ax, N.OR, (1, 1, 0.5, "#e0c85f", 4), (1, 1, 0.5, True), r=14)
    o += svg(0, 0, W, H, s)
    for p, v in zip(N.PTS, sc): o += txt(ax.X(p[0]) + 18, ax.Y(p[1]) - 34 if p[1] == 1 else ax.Y(p[1]) + 8, 80, sg(v), 17, "#e0c85f", True, "left", 26)
    o += txt(1052, 120, 100, t("線<br>x₁ + x₂ = 0.5", "line<br>x₁ + x₂ = 0.5"), 16, "#e0c85f", True, "left", 52) if False else ""
    o += txt(646, 44, 504, t("黃線：x₁ + x₂ = 0.5，藍色是正側", "yellow line: x₁ + x₂ = 0.5; blue is the positive side"), 16, "#e0c85f", True, "center", 26)
    o += txt(646, 422, 504, t(f"四個點的分數依序是 {', '.join(sg(v) for v in sc)}，後三個為正，判 A，剛好是 OR。〔程式計算〕", f"The four scores are {', '.join(sg(v) for v in sc)}; the last three are positive and called A: exactly OR. (computed)"), 16, TXT, False, "center", 48)
    return o
# ------------------------------------------------ 3: same line on XOR
def page3(lang):
    t = tr(lang); o = ""; w = N.LINES["or"]; pr = PRED(w); wrong = [p for p, a, b in zip(N.PTS, pr, N.XOR) if a != b]
    o += cardH(0, 0, 650, 480, "orange", t("同一條線 g = x₁ + x₂ − 0.5 拿去分 XOR", "The same line g = x₁ + x₂ − 0.5 applied to XOR"))
    cols = [(14, 80, t("輸入", "input")), (98, 140, t("代入", "substitute")), (240, 80, t("分數", "score")), (330, 110, t("預測", "pred.")), (450, 100, t("XOR 真值", "XOR class"))]
    for x0, w_, lab in cols: o += txt(x0, 48, w_, lab, 16, MUT, True, "center", 24)
    for i, p in enumerate(N.PTS):
        y0 = 78 + i * 58; v = N.score(*w, p); ok = pr[i] == N.XOR[i]; o += card(10, y0, 630, 50, "green" if ok else "red", .2)
        o += txt(14, y0 + 11, 80, f"({p[0]}, {p[1]})", 17, TXT, True, "center", 28) + txt(98, y0 + 11, 150, f"{p[0]} + {p[1]} {M} 0.5", 17, TXT, False, "center", 28) + txt(252, y0 + 11, 80, sg(v), 17, TXT, True, "center", 28)
        o += chip(346, y0 + 9, 70, "blue" if pr[i] else "orange", ab(pr[i]), True, 32, 18) + txt(430, y0 + 11, 130, ab(N.XOR[i]) + ("" if ok else " ✗"), 17, TXT, True, "center", 28)
    o += txt(14, 318, 622, t("預測恰好等於 OR；對 XOR 錯 1 點。", "The predictions equal OR; on XOR one point is wrong."), 18, TXT, True, "center", 28)
    o += card(60, 360, 530, 90, "yellow", .16) + formula(60, 370, 530, r"\frac{3}{4}=0.75=75\%,\quad 75\%+25\%=100\%", 24, 72)
    o += cardH(666, 0, 494, 480, "blue", t("四個點與那條線", "The four points and the line"))
    ax = Axes(716, 62, 310, 310, XR, XR)
    o += svg(0, 0, W, H, plane(ax, N.XOR, (1, 1, 0.5, "#b7b0a5", 4), (1, 1, 0.5, True), wrong, r=14))
    o += txt(1036, 100, 118, t("(1, 1) 是 B，卻和 A 在同一側", "(1, 1) is B yet on the A side"), 16, "#f0a98b", True, "left", 120)
    o += txt(676, 408, 474, t("這只是這一條線的結果，還不能說所有的線都不行。", "That is the result for this one line only; it does not yet show that every line fails."), 16, TXT, False, "center", 48)
    return o
# ------------------------------------------------ 4: three more lines
def page4(lang):
    t = tr(lang); o = ""
    specs = [("attempt1", r"x_1-0.5", (1, 0, 0.5), (1, 0, 0.5, True), "blue"), ("attempt2", r"x_1+x_2-1.5", (1, 1, 1.5), (1, 1, 1.5, True), "teal"), ("attempt3", r"x_1-x_2", (1, -1, 0), (1, -1, 0, True), "violet")]
    for i, (key, tex, ln, sh, role) in enumerate(specs):
        w = N.LINES[key]; pr = PRED(w); wrong = [p for p, a, b in zip(N.PTS, pr, N.XOR) if a != b]; x0 = i * 392
        o += card(x0, 0, 376, 400, role) + head(x0, 0, 376, role, t(f"第 {i+1} 條線", f"Line {i+1}"), 36, 18)
        o += formula(x0, 40, 376, r"g=" + tex, 22, 36)
        ax = Axes(x0 + 78, 90, 200, 200, XR, XR); o += svg(0, 0, W, H, plane(ax, N.XOR, (*ln, "#e0c85f", 4), sh, wrong, r=12))
        o += txt(x0 + 10, 318, 356, t("預測 ", "predicted ") + "(" + ",".join(str(v) for v in pr) + ")", 17, TXT, False, "center", 28)
        o += chip(x0 + 98, 352, 180, "red", t(f"錯 {len(wrong)} 點", f"{len(wrong)} wrong"), True, 36, 20)
    o += card(0, 410, 1160, 66, "yellow", .14) + txt(10, 416, 1140, t("三條都有錯，這不是證明：沒試到的線太多。<br>預測依 (0,0)、(0,1)、(1,0)、(1,1) 的順序；第三條線上兩個點分數為 0，依規則判 B。〔程式計算〕", "All three have errors, but this is no proof: too many lines are untried.<br>Order (0,0), (0,1), (1,0), (1,1); two points on the third line score 0 and are called B. (computed)"), 16, TXT, False, "center", 54)
    return o
# ------------------------------------------------ 5: A diagonal
def page5(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 560, 480, "blue", t("兩個 A 點連成對角線，中點是 M", "The two A points form a diagonal with midpoint M"))
    ax = Axes(90, 60, 380, 380, XR, XR)
    s = axes_svg(ax, [0, 1], [0, 1], "x₁", "x₂") + f'<line x1="{ax.X(0)}" y1="{ax.Y(1)}" x2="{ax.X(1)}" y2="{ax.Y(0)}" stroke="#9cc3ea" stroke-width="4" stroke-dasharray="10 7"/>'
    for p, v in zip(N.PTS, N.XOR): s += mark(ax, p[0], p[1], ab(v), 14)
    s += f'<ellipse cx="{ax.X(.5)}" cy="{ax.Y(.5)}" rx="9" ry="9" fill="#e0c85f"/>'
    o += svg(0, 0, W, H, s)
    o += txt(ax.X(0.5) + 20, ax.Y(0.5) - 14, 150, "M = (½, ½)", 18, "#e0c85f", True, "left", 28)
    o += cardH(576, 0, 584, 480, "violet", t("假設有一組係數把 XOR 全分對", "Suppose some coefficients classify XOR perfectly"))
    o += card(590, 50, 556, 62, "blue", .16) + formula(590, 56, 556, r"g(0,1)=w_2+w_0>0,\quad g(1,0)=w_1+w_0>0", 20, 48)
    o += txt(590, 120, 556, t("正數加正數仍為正，兩式相加：", "A positive plus a positive stays positive; add the two:"), 17, TXT, False, "center", 28)
    o += card(590, 154, 556, 62, "yellow", .18) + formula(590, 160, 556, r"(w_2+w_0)+(w_1+w_0)=w_1+w_2+2w_0>0", 21, 48)
    o += named(590, 232, 556, t("中點 M：兩個 A 點連線的正中央，M = (½, ½)。", "Midpoint M: the exact middle of the segment joining the two A points, M = (½, ½)."), "term:" + t("中點", "Midpoint"), 16, TXT, 48)
    o += card(590, 292, 556, 80, "violet", .16) + formula(590, 300, 556, r"g(M)=\frac{w_1+w_2+2w_0}{2}>0", 26, 64)
    o += txt(590, 384, 556, t("g 是線性的，所以 g(M) 剛好是兩端分數的平均，也是正的。", "g is linear, so g(M) is exactly the average of the two end scores, and it is positive too."), 17, TXT, True, "center", 52)
    o += txt(590, 438, 556, t("下一頁：B 的兩個點會得到什麼條件？", "Next: what condition do the two B points give?"), 16, MUT, False, "center", 26)
    return o
# ------------------------------------------------ 6: contradiction
def page6(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 560, 480, "orange", t("兩條對角線共用中點 M", "Both diagonals share the midpoint M"))
    ax = Axes(90, 60, 380, 380, XR, XR)
    s = axes_svg(ax, [0, 1], [0, 1], "x₁", "x₂") + f'<line x1="{ax.X(0)}" y1="{ax.Y(1)}" x2="{ax.X(1)}" y2="{ax.Y(0)}" stroke="#9cc3ea" stroke-width="4" stroke-dasharray="10 7"/><line x1="{ax.X(0)}" y1="{ax.Y(0)}" x2="{ax.X(1)}" y2="{ax.Y(1)}" stroke="#f0a98b" stroke-width="4" stroke-dasharray="10 7"/>'
    for p, v in zip(N.PTS, N.XOR): s += mark(ax, p[0], p[1], ab(v), 14)
    s += f'<ellipse cx="{ax.X(.5)}" cy="{ax.Y(.5)}" rx="9" ry="9" fill="#e0c85f"/>'
    o += svg(0, 0, W, H, s)
    o += txt(ax.X(0.5) + 20, ax.Y(0.5) - 14, 60, "M", 20, "#e0c85f", True, "left", 28)
    o += cardH(576, 0, 584, 480, "red", t("B 的兩個點相加卻不大於 0", "The B points add up to at most 0"))
    o += card(590, 50, 556, 62, "orange", .16) + formula(590, 56, 556, r"g(0,0)=w_0\leq 0,\quad g(1,1)=w_1+w_2+w_0\leq 0", 20, 48)
    o += card(590, 124, 556, 62, "orange", .16) + formula(590, 130, 556, r"w_0+(w_1+w_2+w_0)=w_1+w_2+2w_0\leq 0", 21, 48)
    o += card(590, 204, 556, 60, "blue", .16) + txt(598, 208, 540, t("A 要求：w₁ + w₂ + 2w₀ > 0", "A requires: w₁ + w₂ + 2w₀ > 0"), 20, TXT, True, "center", 52)
    o += card(590, 274, 556, 60, "orange", .16) + txt(598, 278, 540, t("B 要求：w₁ + w₂ + 2w₀ ≤ 0", "B requires: w₁ + w₂ + 2w₀ ≤ 0"), 20, TXT, True, "center", 52)
    o += chip(590, 346, 556, "red", t("同一個式子：矛盾，不存在能全對的係數", "Same expression: a contradiction"), True, 50, 21)
    o += txt(590, 408, 556, t("這是對所有線的證明，不只是排除圖上幾條候選線。幾何上，g(M) 既是 A 兩端分數的平均，也是 B 兩端分數的平均，不可能一個為正、一個不大於 0。", "This is a proof for every line, not just for a few candidates. Geometrically g(M) is both the average of the A end scores and of the B end scores, so it cannot be positive and at most 0 at once."), 16, TXT, False, "center", 72)
    return o
# ------------------------------------------------ 7: training cannot fix it
def page7(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 640, 480, "red", t("錯誤數隨訓練下降，但永遠到不了 0（示意）", "Errors fall with training but never reach 0 (schematic)"))
    ax = Axes(86, 70, 520, 300, (0, 9), (-0.4, 3.6)); s = axes_svg(ax, [], [0, 1, 2, 3], t("訓練回合", "training rounds"), t("錯誤數", "errors"))
    steps = [3, 3, 2, 2, 1, 1, 1, 1, 1]; pts = [(i + .5, v) for i, v in enumerate(steps)]
    for (a, b), (c, d) in zip(pts, pts[1:]): s += f'<line x1="{ax.X(a):.1f}" y1="{ax.Y(b):.1f}" x2="{ax.X(c):.1f}" y2="{ax.Y(d):.1f}" stroke="{RD}" stroke-width="4"/>'
    for a, b in pts: s += f'<circle cx="{ax.X(a):.1f}" cy="{ax.Y(b):.1f}" r="7" fill="{RD}"/>'
    s += f'<line x1="{ax.x}" y1="{ax.Y(0):.1f}" x2="{ax.x+ax.w}" y2="{ax.Y(0):.1f}" stroke="#7bbb8c" stroke-width="4" stroke-dasharray="12 8"/>'
    o += svg(0, 0, W, H, s)
    o += txt(ax.X(3.2), ax.Y(0) - 34, 280, t("0（永遠達不到）", "0 (never reached)"), 18, "#9be0ad", True, "left", 28)
    o += txt(14, 396, 612, t("示意圖，不是真實實驗。", "A schematic, not a real experiment."), 16, MUT, False, "center", 26)
    o += txt(14, 424, 612, t(f"對係數網格窮舉（−4 到 4，間隔 0.5），XOR 最少仍錯 {N.best_over_grid()} 點。〔程式計算〕", f"Searching a grid of coefficients (−4 to 4, step 0.5),<br>XOR still has at least {N.best_over_grid()} error. (computed)"), 16, TXT, True, "center", 48)
    o += cardH(656, 0, 504, 480, "orange", t("表達能力與訓練是兩個問題", "Capacity and training are two different questions"))
    o += named(670, 50, 476, t("表達能力：模型的邊界形狀本身能不能表示出答案。", "Representational capacity: whether the shape of the model's boundary can express the answer at all."), "term:" + t("表達能力", "Representational capacity"), 16, TXT, 72)
    o += card(670, 136, 476, 150, "red", .16) + txt(678, 144, 460, t("常見迷思：多跑幾輪、調小學習率或重新初始化，應該就能全對。", "Common myth: more rounds, a smaller learning rate or a fresh start should get all points right."), 17, TXT, True, "center", 52) + txt(678, 210, 460, t("這些只是在同一類直線裡找錯誤少的參數；這一類直線裡沒有全對的答案。", "These only look for fewer errors among the same family of lines, and that family has no perfect answer."), 16, TXT, False, "center", 72)
    o += card(670, 300, 476, 160, "yellow", .16) + txt(678, 308, 460, t("訓練有沒有找到最好的線，和最好的線夠不夠用，是兩個不同的問題。最低錯誤是 1 點，不是 0。", "Whether training finds the best line and whether the best line is good enough are different questions. The lowest error is 1 point, not 0."), 18, TXT, True, "center", 140)
    return o
# ------------------------------------------------ 8: strip
def page8(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 600, 480, "teal", t("兩條平行線夾出一條帶", "Two parallel lines make a strip"))
    ax = Axes(80, 56, 380, 380, XR, XR)
    s = poly_svg(ax, region(ax, [(1, 1, 0.5, True), (1, 1, 1.5, False)]), BL, .36) + axes_svg(ax, [0, 1], [0, 1], "x₁", "x₂") + line_svg(ax, 1, 1, 0.5, TE, 4) + line_svg(ax, 1, 1, 1.5, VI, 4)
    for p, v in zip(N.PTS, N.XOR): s += mark(ax, p[0], p[1], ab(v), 14)
    o += svg(0, 0, W, H, s)
    o += txt(ax.X(0.9), ax.Y(0.4), 80, "", 16) if False else ""
    o += txt(476, 140, 118, "s = 0.5", 16, "#8fe0d0", True, "left", 24) + txt(476, 180, 118, "s = 1.5", 16, "#c4b0e6", True, "left", 24) + txt(476, 240, 118, t("兩個 A 在帶內，兩個 B 在帶外", "A inside, B outside"), 16, TXT, False, "left", 72)
    o += cardH(616, 0, 544, 480, "violet", t("令 s = x₁ + x₂", "Let s = x₁ + x₂"))
    sv = [p[0] + p[1] for p in N.PTS]
    for i, (p, v) in enumerate(zip(N.PTS, sv)):
        x0 = 630 + i * 129; role = "blue" if N.XOR[i] else "orange"
        o += card(x0, 52, 121, 100, role, .2) + txt(x0, 58, 121, f"({p[0]}, {p[1]})", 16, MUT, True, "center", 24) + txt(x0, 86, 121, f"s = {v}", 20, TXT, True, "center", 30) + txt(x0, 120, 121, ab(N.XOR[i]), 18, TXT, True, "center", 28)
    o += txt(630, 166, 516, t("四個點的 s 只會是 0、1、2，A 恰好是 s = 1。", "The s values are only 0, 1, 2, and A is exactly s = 1."), 17, TXT, False, "center", 28)
    o += card(630, 208, 516, 80, "yellow", .18) + formula(630, 216, 516, r"\widehat y=1\Leftrightarrow  0.5<s\leq 1.5", 30, 64)
    o += txt(630, 300, 516, t("用兩個判斷的交集選出（補充）。", "Selected by the intersection of two judgments (supplementary)."), 16, MUT, False, "center", 26)
    o += card(630, 338, 516, 124, "violet", .16) + txt(638, 346, 500, t("改變的是模型允許的邊界組合，不是換一組舊係數。這只驗證四個布林點，沒有規定連續平面上 XOR 的唯一延伸。下一個知識點問：兩個判斷要怎麼用神經元做出來？", "What changes is the allowed combination of boundaries, not the old coefficients. Only the four Boolean points are checked. Next KP: how do neurons build the two judgments?"), 16, TXT, False, "center", 108)
    return o
# ------------------------------------------------ 9: ties
def page9(lang):
    t = tr(lang); o = ""; w = N.LINES["attempt3"]; ps = PRED(w); pg = PRED(w, False)
    o += cardH(0, 0, 660, 480, "yellow", t("實作：邊界上的點，> 與 ≥ 預測不同", "Implementation: points on the boundary, > and ≥ differ"))
    o += formula(14, 46, 632, r"g=x_1-x_2", 24, 40)
    for x0, w_, lab in ((14, 90, t("輸入", "input")), (108, 90, t("分數", "score")), (206, 140, t("用 >（本講義）", "with > (this KP)")), (350, 140, t("用 ≥", "with ≥")), (494, 150, t("XOR 真值", "XOR class"))): o += txt(x0, 94, w_, lab, 16, MUT, True, "center", 24)
    for i, p in enumerate(N.PTS):
        y0 = 124 + i * 56; v = N.score(*w, p); tie = v == 0; o += card(10, y0, 640, 48, "yellow" if tie else "grey", .22 if tie else .1)
        o += txt(14, y0 + 9, 90, f"({p[0]}, {p[1]})", 17, TXT, True, "center", 28) + txt(108, y0 + 9, 90, sg(v), 17, TXT, True, "center", 28)
        o += chip(236, y0 + 8, 80, "blue" if ps[i] else "orange", ab(ps[i]), True, 32, 18) + chip(380, y0 + 8, 80, "blue" if pg[i] else "orange", ab(pg[i]), True, 32, 18) + txt(494, y0 + 9, 150, ab(N.XOR[i]), 17, TXT, True, "center", 28)
    e1, e2 = N.errors(ps, N.XOR), N.errors(pg, N.XOR)
    o += txt(14, 360, 632, t(f"錯誤數：用 > 是 {e1} 點，用 ≥ 是 {e2} 點。〔程式計算〕", f"Errors: {e1} with >, {e2} with ≥. (computed)"), 18, TXT, True, "center", 28)
    o += txt(14, 400, 632, t("測試必須包含邊界上的點，免得程式用 >= 而文字用 >。這裡的半整數可精確表示；一般實數資料仍要注意浮點誤差。", "Tests must include points on the boundary, so that code using >= does not disagree with text using >. Half-integers are exact here; with general real data watch for floating-point error."), 16, MUT, False, "center", 72)
    o += cardH(676, 0, 484, 480, "blue", t("線 x₁ = x₂ 上的兩個點", "Two points on the line x₁ = x₂"))
    ax = Axes(750, 56, 320, 320, XR, XR)
    o += svg(0, 0, W, H, plane(ax, N.XOR, (1, -1, 0, "#e0c85f", 4), (1, -1, 0, True), [(0, 0), (1, 1)], r=14))
    o += txt(686, 414, 464, t("黃圈：分數剛好為 0 的 (0, 0) 與 (1, 1)", "Rings: (0, 0) and (1, 1), whose score is exactly 0"), 16, TXT, False, "center", 48)
    return o
# ------------------------------------------------ 10: exercise 1
def page10(lang):
    t = tr(lang); o = ""; w = N.LINES["and"]; sc = [N.score(*w, p) for p in N.PTS]; pr = PRED(w)
    o += card(0, 0, 680, 470, "blue") + head(0, 0, 680, "blue", t("小練習 1：用 g = x₁ + x₂ − 1.5 核對 AND 與 XOR", "Exercise 1: check AND and XOR with g = x₁ + x₂ − 1.5"), 36, 18)
    o += txt(14, 46, 652, t("題目：算四個輸入的分數與預測，分別核對 AND 與 XOR。", "Question: compute the scores and predictions, then check AND and XOR."), 17, TXT, True, "left", 28)
    o += chip(14, 92, 130, "grey", t("解題示範", "Solution"), False, 30, 16)
    o += txt(154, 92, 520, t("① 符號式　② 代入四點　③ 預測　④ 核對", "① formula　② substitute　③ predict　④ check"), 16, MUT, False, "left", 30)
    o += formula(14, 126, 652, r"g=x_1+x_2-1.5,\quad g>0\Rightarrow A", 22, 40)
    for i, p in enumerate(N.PTS):
        x0 = 14 + i * 166; role = "blue" if pr[i] else "orange"
        o += card(x0, 170, 158, 112, role, .18) + txt(x0, 176, 158, f"({p[0]}, {p[1]})", 16, MUT, True, "center", 24) + txt(x0, 204, 158, f"{p[0]}+{p[1]}{M}1.5 = {sg(sc[i])}", 16, TXT, False, "center", 26) + chip(x0 + 34, 240, 90, role, ab(pr[i]), True, 32, 18)
    ea, ex = N.errors(pr, N.AND), N.errors(pr, N.XOR)
    o += card(14, 296, 320, 150, "green", .2) + txt(22, 304, 304, t(f"AND 真值 B、B、B、A<br>預測 B、B、B、A<br>全對（錯 {ea} 點）", f"AND truth B, B, B, A<br>predicted B, B, B, A<br>all right ({ea} wrong)"), 18, TXT, True, "center", 120)
    o += card(346, 296, 320, 150, "red", .22) + txt(354, 304, 304, t(f"XOR 真值 B、A、A、B<br>預測 B、B、B、A<br>錯 {ex} 點", f"XOR truth B, A, A, B<br>predicted B, B, B, A<br>{ex} wrong"), 18, TXT, True, "center", 120)
    o += cardH(696, 0, 464, 470, "teal", t("同一條線：AND 全對，XOR 錯 3 點", "One line: AND all right, XOR 3 wrong"))
    a1 = Axes(738, 100, 180, 180, XR, XR); a2 = Axes(960, 100, 180, 180, XR, XR)
    o += svg(0, 0, W, H, plane(a1, N.AND, (1, 1, 1.5, "#e0c85f", 4), (1, 1, 1.5, True), [], r=11) + plane(a2, N.XOR, (1, 1, 1.5, "#e0c85f", 4), (1, 1, 1.5, True), [(0, 1), (1, 0), (1, 1)], r=11))
    o += txt(738, 50, 180, "AND", 20, TXT, True, "center", 28) + txt(960, 50, 180, "XOR", 20, TXT, True, "center", 28)
    o += txt(706, 336, 444, t("左：AND 被同一條線全分對。右：XOR 的 (0,1)、(1,0) 被判 B，(1,1) 被判 A，紅圈是錯的點。〔程式計算〕", "Left: the same line classifies AND perfectly. Right: XOR's (0,1), (1,0) are called B and (1,1) is called A; red rings mark the errors. (computed)"), 16, TXT, False, "center", 96)
    return o
# ------------------------------------------------ 11: exercises 2 and 3
def page11(lang):
    t = tr(lang); o = ""
    o += card(0, 0, 580, 470, "red") + head(0, 0, 580, "red", t("小練習 2：XOR 失敗的原因是什麼？", "Exercise 2: why does XOR fail?"), 36, 18)
    o += txt(14, 46, 552, t("題目：是樣本少、雜訊、學習率，還是邊界形狀受限？", "Question: too few samples, noise, the learning rate, or a limited boundary shape?"), 17, TXT, True, "left", 52)
    rows = [(t("樣本少？", "Too few samples?"), t("四個輸入就是全部資料，沒有缺漏", "The four inputs are all the data; nothing is missing"), False), (t("雜訊？", "Noise?"), t("標籤直接取自 XOR 真值，沒有錯的標籤", "The labels come straight from XOR; none is wrong"), False), (t("學習率？", "Learning rate?"), t("第 6 頁的矛盾對任何係數都成立，與怎麼調整無關", "The page 6 contradiction holds for any coefficients"), False), (t("邊界形狀受限？", "Limited boundary shape?"), t("排除前三項後，只剩下「一條線」這個形狀", "With the first three excluded, only the single-line shape remains"), True)]
    for i, (a, b, ans) in enumerate(rows):
        y0 = 108 + i * 84; role = "green" if ans else "grey"; o += card(14, y0, 552, 76, role, .2 if ans else .1)
        o += txt(24, y0 + 6, 400, a, 17, TXT, True, "left", 28) + chip(470, y0 + 8, 86, "green" if ans else "red", t("答案", "answer") if ans else t("排除", "excluded"), True, 30, 16) + txt(24, y0 + 38, 532, b, 16, TXT, False, "left", 34)
    o += txt(14, 446, 552, "", 16, TXT, False, "left", 24)
    o += card(596, 0, 564, 470, "orange") + head(596, 0, 564, "orange", t("小練習 3：OR、XOR、XNOR 哪個能用一條線分開？", "Exercise 3: which of OR, XOR, XNOR is separable by one line?"), 36, 18)
    o += txt(610, 46, 536, t("XNOR：輸入相同輸出 1。提示：先畫標籤。", "XNOR: outputs 1 when the inputs are equal. Hint: draw the labels first."), 17, TXT, True, "left", 52)
    items = [("OR", N.OR, "blue", t("可以：g = x₁ + x₂ − 0.5，分數 −0.5、0.5、0.5、1.5，預測 B、A、A、A，與 OR 相同", "Yes: g = x₁ + x₂ − 0.5 gives B, A, A, A, the same as OR"), True), ("XOR", N.XOR, "orange", t("不行：第 5、6 頁的矛盾", "No: the contradiction of pages 5 and 6"), False), ("XNOR", N.XNOR, "violet", t("不行：(0,0)、(1,1) 為 1，(0,1)、(1,0) 為 0，同類仍在對角，只是標籤互換，同樣的矛盾還在", "No: same-class points still sit on the diagonals, only the labels swap, so the same contradiction remains"), False)]
    for i, (name, tv, role, txt_, ok) in enumerate(items):
        y0 = 108 + i * 102; o += card(610, y0, 536, 94, role, .16)
        ax = Axes(622, y0 + 10, 74, 74, XR, XR); o += svg(0, 0, W, H, plane(ax, tv, None, None, [], r=7) if False else "".join(mark(ax, p[0], p[1], ab(v), 7) for p, v in zip(N.PTS, tv)) + f'<rect x="{ax.x}" y="{ax.y}" width="{ax.w}" height="{ax.h}" fill="none" stroke="#6b645a" stroke-width="2"/>')
        o += chip(706, y0 + 8, 90, role, name, True, 28, 17) + chip(1036, y0 + 8, 100, "green" if ok else "red", t("可分", "yes") if ok else t("不可分", "no"), True, 28, 16) + txt(706, y0 + 40, 432, txt_, 16, TXT, False, "left", 52)
    o += card(610, 420, 536, 40, "yellow", .16) + txt(618, 424, 520, t("結論：三者中只有 OR 能用一條線分開。〔程式窮舉驗證〕", "Conclusion: only OR is separable. (checked by search)"), 16, TXT, True, "center", 32)
    return o
PAGES_DESIGN = [(f"p{i}", globals()[f"page{i}"]) for i in range(1, 12)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:]
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp01_" + fid, fn, W, H, out)
