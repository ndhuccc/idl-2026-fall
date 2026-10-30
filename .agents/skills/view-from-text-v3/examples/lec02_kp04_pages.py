import math, sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *

KP = "KP04"; N = 8
XOR = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
def run_xor(rounds=10):
    w = [0, 0, 0]; cnt = []; ends = []
    for _ in range(rounds):
        c = 0
        for (x1, x2), y in XOR:
            o = 1 if w[0] * x1 + w[1] * x2 + w[2] >= 0 else 0
            if o != y: w = [w[0] + (y - o) * x1, w[1] + (y - o) * x2, w[2] + (y - o)]; c += 1
        cnt.append(c); ends.append(tuple(w))
    return cnt, ends
CNT, ENDS = run_xor()
assert CNT == [3, 3, 4, 4, 4, 4, 4, 4, 4, 4] and all(e == (-1, 0, 0) for e in ENDS[3:])
GRID = list(itertools.product(range(-3, 4), repeat=3))
BEST = max(sum((1 if a[0] * x1 + a[1] * x2 + a[2] >= 0 else 0) == y for (x1, x2), y in XOR) for a in GRID)
assert len(GRID) == 343 and BEST == 3
WA, WA0, WB, WB0, NEWP = (4.03, 3.17), -15.22, (0.94, 0.95), -5.04, (2.5, 2.4)
SA = WA[0] * NEWP[0] + WA[1] * NEWP[1] + WA0; SB = WB[0] * NEWP[0] + WB[1] * NEWP[1] + WB0
assert round(SA, 2) == 2.46 and round(SB, 2) == -0.41

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    def sq(g, x0, y0, side, wline=None, bad=None):
        sc = side / 1.9; u = lambda v: x0 + sc * 0.35 + v * sc; vv = lambda v: y0 + side - sc * 0.35 - v * sc
        g.rect(x0, y0, side, side, fill=CARD, stroke=GRAY, sw=3, radius=14, z=1)
        if wline:
            a, b_, c = wline; pts = []
            for xx in (-0.35, 1.55):
                if abs(b_) > 1e-9:
                    yy = -(a * xx + c) / b_
                    if -0.35 <= yy <= 1.55: pts.append((xx, yy))
            for yy in (-0.35, 1.55):
                if abs(a) > 1e-9:
                    xx = -(b_ * yy + c) / a
                    if -0.35 <= xx <= 1.55 and all(abs(xx - p[0]) + abs(yy - p[1]) > 1e-6 for p in pts): pts.append((xx, yy))
            if len(pts) >= 2: g.line([(u(pts[0][0]), vv(pts[0][1])), (u(pts[1][0]), vv(pts[1][1]))], RED, sw=5, dash=True, z=5)
        for p, y in XOR:
            g.ellipse(u(p[0]), vv(p[1]), 22, fill=(GRN if y else CARD), stroke=(GRN if y else BLUE), sw=4, z=6)
            if bad and p in bad: g.ellipse(u(p[0]), vv(p[1]), 34, stroke=RED, sw=4, z=7)
        return u, vv

    # ============ 1: XOR on the diagonal ============
    g = new(1, T("KP04 單層 Perceptron 解不了 XOR", "KP04 One Perceptron cannot solve XOR"))
    g.beat = 1
    u, vv = sq(g, 60, 120, 480)
    g.line([(u(0), vv(1)), (u(1), vv(0))], GREEN if False else GRN, sw=4, dash=True, z=3); g.line([(u(0), vv(0)), (u(1), vv(1))], BLUE, sw=4, dash=True, z=3)
    g.text(60, 604, 480, 30, T("實心:類 1　空心:類 0", "solid: class 1 · open: class 0"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.beat = 2
    g.rect(600, 140, 630, 150, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3)
    g.text(610, 148, 610, 134, T("同一類的兩個點\n在正方形的對角線上", "the two points of a class\nsit on a diagonal of the square"), size=28, color=BLUE, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(600, 320, 630, 260, fill=REDL, stroke=RED, sw=3, radius=16, z=3)
    g.text(610, 330, 610, 60, T("KP02 的結論反過來", "KP02's conclusion, reversed"), size=26, color=RED, bold=True, align="center", z=8)
    g.text(610, 400, 610, 170, T("可分 → 保證停\n不可分 → 一定停不下來", "separable → guaranteed to stop\nnot separable → never stops"), size=30, color=INK, bold=True, align="center", z=8)
    banner(g, T("第 1 章已證明:沒有任何 $w_1,w_2,w_0$ 做得出 XOR", "Chapter 1 proved it: no $w_1,w_2,w_0$ makes XOR"))
    pages.append((g, T("XOR 的四個點", "The four XOR points"), {
        0: T("XOR 的四個點：0 0 與 1 1 是類 0，0 1 與 1 0 是類 1。", "The four XOR points: 0 0 and 1 1 are class 0; 0 1 and 1 0 are class 1."),
        1: T("同一類的兩個點，剛好在正方形的對角線上。", "The two points of a class sit exactly on a diagonal of the square."),
        2: T("第一章已經證明沒有任何權重做得出 XOR。對學習規則來說，KP02 的結論反過來：不可分，一定停不下來。", "Chapter 1 already proved no weights make XOR. For the learning rule, KP02's conclusion reverses: not separable means it never stops."),
        3: T("", "")}))

    # ============ 2: it never stops ============
    g = new(2, T("實跑:每一輪都有更新", "Run it: every pass makes updates"))
    g.beat = 1
    base = 520; bw = 90; sc_ = 70
    g.line([(80, base), (1180, base)], INK, sw=3, z=3)
    for i, c in enumerate(CNT):
        x = 100 + i * 108
        g.beat = 1 if i < 3 else 2
        g.rect(x, base - c * sc_, bw, c * sc_, fill=(ORGL), stroke=ORG, sw=3, radius=8, z=3)
        g.text(x, base - c * sc_ - 40, bw, 36, str(c), size=26, color=ORG, bold=True, align="center", z=8)
        g.text(x, base + 6, bw, 34, T(f"第 {i + 1} 輪", f"pass {i + 1}"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.text(80, 112, 1100, 40, T("每一輪的更新次數", "number of updates in each pass"), size=22, color=MUTED, bold=True, z=8)
    g.beat = 3
    g.rect(190, 560, 900, 60, fill="#fef3c7", stroke=GOLD, sw=3, radius=14, z=3)
    g.text(190, 560, 900, 60, T("第 4 輪起，每輪結束權重都回到同一組 $(w_1,w_2,w_0)=(-1,0,0)$", "from pass 4 on, every pass ends at the same $(w_1,w_2,w_0)=(-1,0,0)$"), size=22, color=INK, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("永遠不會出現「一整輪沒有更新」", "a pass with no update never happens"))
    pages.append((g, T("實跑 XOR", "Running XOR"), {
        0: T("從全 0 開始，依序學 XOR，學習率 1。前兩輪各更新 3 次。", "Start from zeros and learn XOR in order with learning rate 1. The first two passes make 3 updates each."),
        1: T("從第三輪起，每輪都更新 4 次，一直持續到第十輪。", "From the third pass every pass makes 4 updates, through pass ten."),
        2: T("從第四輪起，每輪結束時權重都回到同一組 負 1、0、0：輪內更新四次又繞回原處，永遠沒有一整輪沒更新。", "From pass four on every pass ends at the same weights minus 1, 0, 0: four updates inside a pass loop back to the start, so there is never a pass without an update."),
        3: T("", "")}))

    # ============ 3: grid search ============
    g = new(3, T("搜遍所有權重，最多對 3 個", "Searching all weights: at most 3 of 4 right"))
    g.beat = 1
    sq(g, 60, 120, 440, (1, 1, -0.5), bad=[(1, 1)])
    g.text(60, 570, 440, 36, T("一個例子:$(1,1,-0.5)$ 錯 1 個", "one example: $(1,1,-0.5)$ gets 1 wrong"), size=20, color=INK, bold=True, align="center", z=8)
    g.beat = 2
    g.rect(560, 130, 670, 190, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3)
    g.text(570, 138, 650, 50, T("格點 $[-3,3]^3$", "grid $[-3,3]^3$"), size=26, color=BLUE, bold=True, align="center", z=8)
    g.text(570, 196, 650, 110, T("$7^3=343$ 組權重\n全部算一遍", "$7^3=343$ weight triples\nall evaluated"), size=30, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(560, 350, 670, 220, fill=REDL, stroke=RED, sw=3, radius=16, z=3)
    g.text(570, 360, 650, 60, T("最多只對", "at most only"), size=26, color=RED, bold=True, align="center", z=8)
    g.text(570, 424, 650, 130, T("3 / 4", "3 / 4"), size=64, color=RED, bold=True, align="center", z=8)
    banner(g, T("不是資料太少或學習率不對，是表示力的限制", "not too little data or a bad learning rate: a limit of expressiveness"))
    pages.append((g, T("格點搜尋", "Grid search"), {
        0: T("單一神經元的邊界是一條直線，不論怎麼畫，總有一個點被分錯。", "One neuron's boundary is a straight line; however you draw it, some point is wrong."),
        1: T("用程式在負 3 到正 3 的整數格點上，把 343 組權重全部算一遍。", "A program evaluates all 343 weight triples on the integer grid from minus 3 to plus 3."),
        2: T("最多只能讓四筆中的三筆正確。這不是資料太少或學習率不對，是表示力的限制。", "At most 3 of 4 are right. This is not too little data or a bad learning rate; it is a limit of expressiveness."),
        3: T("", "")}))

    # ============ 4: separable but different boundaries ============
    g = new(4, T("資料可分，邊界也可能不同", "Separable data, yet different boundaries"))
    X0, Y0, SW, SH = 60, 110, 720, 500
    ux = lambda v: X0 + 40 + v * 100; uy = lambda v: Y0 + SH - 40 - v * 80
    g.beat = 1
    g.rect(X0, Y0, SW, SH, fill=CARD, stroke=GRAY, sw=3, radius=14, z=1)
    g.ellipse(ux(0.9), uy(0.9), 75, 60, fill="#dbeafe", stroke=BLUE, sw=3, z=2, fillOpacity=0.7); g.text(ux(0.9) - 100, uy(0.9) - 24, 200, 56, T("類 0(示意)", "class 0 (schematic)"), size=20, color=BLUE, bold=True, align="center", z=8)
    g.ellipse(ux(5.4), uy(5.2), 75, 60, fill="#dcfce7", stroke=GRN, sw=3, z=2, fillOpacity=0.7); g.text(ux(5.4) - 100, uy(5.2) - 24, 200, 56, T("類 1(示意)", "class 1 (schematic)"), size=20, color=GRN, bold=True, align="center", z=8)
    g.beat = 2
    la = lambda x: (-WA0 - WA[0] * x) / WA[1]; lb = lambda x: (-WB0 - WB[0] * x) / WB[1]
    g.line([(ux(0.2), uy(la(0.2))), (ux(3.7), uy(la(3.7)))], RED, sw=5, z=5); g.text(ux(3.7) + 6, uy(la(3.7)) - 14, 60, 36, "A", size=30, color=RED, bold=True, z=8)
    g.line([(ux(0.5), uy(lb(0.5))), (ux(5.3), uy(lb(5.3)))], ORG, sw=5, z=5); g.text(ux(5.3) - 2, uy(lb(5.3)) - 4, 60, 36, "B", size=30, color=ORG, bold=True, z=8)
    g.beat = 3
    g.ellipse(ux(NEWP[0]), uy(NEWP[1]), 14, fill=GOLD, stroke=INK, sw=3, z=7); g.text(ux(NEWP[0]) + 22, uy(NEWP[1]) - 12, 220, 56, T("新點 $(2.5,2.4)$", "new point $(2.5,2.4)$"), size=20, color=INK, bold=True, z=8)
    g.rect(830, 140, 400, 440, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3)
    g.text(840, 150, 380, 120, T("兩條線的訓練誤差都是 0", "both lines have training error 0"), size=26, color=BLUE, bold=True, align="center", z=8)
    g.text(840, 300, 380, 260, T("只是起點與資料順序不同:Perceptron 一得到全對就停，不挑最好的一條", "they differ only in start and data order: the Perceptron stops at all-right and does not pick the best line"), size=22, color=INK, bold=True, align="center", z=8)
    banner(g, T("全對只代表停下來，不代表停在好的位置", "all-right only means it stopped, not that it stopped well"))
    pages.append((g, T("邊界取決於起點", "The start decides"), {
        0: T("兩類資料線性可分，同一批資料、同一條規則。", "Two classes, linearly separable; same data, same rule."),
        1: T("只換起點與資料順序，得到邊界 A 與邊界 B。兩條線都讓訓練誤差為 0。示意圖的點群只是示意。", "Changing only the start and the data order gives boundary A and boundary B. Both have training error 0. The blobs here are schematic."),
        2: T("但對新點 2.5、2.4，兩條線給出不同的答案。Perceptron 一得到全對就停了，不會再挑最好的一條。", "But for the new point 2.5, 2.4 the two lines disagree. The Perceptron stops at all-right and never picks the best one."),
        3: T("", "")}))

    # ============ 5: different answers ============
    g = new(5, T("同一個新點，兩個相反的答案", "One new point, two opposite answers"))
    g.beat = 1
    g.rect(50, 130, 540, 330, fill=REDL, stroke=RED, sw=3, radius=16, z=3)
    g.text(50, 140, 540, 50, T("邊界 A", "boundary A"), size=28, color=RED, bold=True, align="center", z=8)
    g.text(60, 200, 520, 100, r"$\mathbf{w}=(4.03,3.17),\ w_0=-15.22$", size=26, color=INK, align="center", vcenter=True, z=8)
    g.text(60, 310, 520, 70, rf"$s=4.03\cdot2.5+3.17\cdot2.4-15.22={SA:.2f}$", size=24, color=INK, align="center", vcenter=True, z=8)
    g.text(60, 390, 520, 60, T("判為類 1", "class 1"), size=32, color=GRN, bold=True, align="center", z=8)
    g.beat = 2
    g.rect(690, 130, 540, 330, fill=ORGL, stroke=ORG, sw=3, radius=16, z=3)
    g.text(690, 140, 540, 50, T("邊界 B", "boundary B"), size=28, color=ORG, bold=True, align="center", z=8)
    g.text(700, 200, 520, 100, r"$\mathbf{w}=(0.94,0.95),\ w_0=-5.04$", size=26, color=INK, align="center", vcenter=True, z=8)
    g.text(700, 310, 520, 70, rf"$s=0.94\cdot2.5+0.95\cdot2.4-5.04={SB:.2f}$".replace("=-", "=-"), size=24, color=INK, align="center", vcenter=True, z=8)
    g.text(700, 390, 520, 60, T("判為類 0", "class 0"), size=32, color=BLUE, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(190, 490, 900, 110, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(210, 500, 860, 90, T("兩個都對訓練資料 100% 正確，你要相信哪一個？\n要靠沒看過的資料來決定(之後的驗證)", "both are 100% right on training data: which do you trust?\nunseen data must decide (validation, later)"), size=24, color=INK, bold=True, align="center", z=8)
    banner(g, T("零訓練誤差不蘊含泛化", "zero training error does not imply generalization"))
    pages.append((g, T("相反的答案", "Opposite answers"), {
        0: T("邊界 A 的權重 4.03、3.17，偏置負 15.22。新點的 s 等於 2.46，判為類 1。", "Boundary A has weights 4.03, 3.17 and bias minus 15.22. At the new point s is 2.46, so class 1."),
        1: T("邊界 B 的權重 0.94、0.95，偏置負 5.04。新點的 s 等於負 0.41，判為類 0。", "Boundary B has weights 0.94, 0.95 and bias minus 5.04. At the new point s is minus 0.41, so class 0."),
        2: T("兩個都對訓練資料完全正確，卻對新點給出相反的答案。零訓練誤差不蘊含泛化。", "Both are perfect on the training data yet give opposite answers on the new point. Zero training error does not imply generalization."),
        3: T("", "")}))

    # ============ 6: history timeline ============
    g = new(6, T("兩個限制，一段歷史", "Two limits, one piece of history"))
    ev = [("1958", T("Rosenblatt 提出 Perceptron", "Rosenblatt proposes the Perceptron"), T("第一個會學習的神經元", "the first neuron that learns"), BLUE, BLUEL),
          ("1960", T("Widrow 與 Hoff 提出 ADALINE", "Widrow and Hoff propose ADALINE"), T("改用連續的 LMS 損失", "uses a continuous LMS loss"), GRN, GRNL),
          ("1969", T("Minsky 與 Papert 分析單層限制", "Minsky and Papert analyze the single-layer limit"), T("XOR 成為最有名的反例", "XOR becomes the best-known counterexample"), RED, REDL),
          (T("之後", "later"), T("多層加非線性活化", "multiple layers with nonlinear activation"), T("繞過直線限制", "gets around the straight-line limit"), PUR, PURL)]
    ids = []
    for i, (yr, a, b_, col, light) in enumerate(ev):
        g.beat = i + 1 if i < 3 else 3
        x = 50 + i * 300
        ids.append(g.mark()); g.rect(x, 170, 270, 330, fill=light, stroke=col, sw=3, radius=16, z=3)
        g.text(x, 180, 270, 60, yr, size=34, color=col, bold=True, align="center", z=8); g.text(x + 10, 250, 250, 120, a, size=22, color=INK, bold=True, align="center", z=8); g.text(x + 10, 390, 250, 100, b_, size=20, color=MUTED, bold=True, align="center", z=8)
        if i: g.link(ids[i - 1], ids[i], 4, 12, GRAY, 5)
    g.beat = 4
    banner(g, T("「一本書造成整個研究寒冬」是過度簡化;XOR 限制本身是可驗算的事實", "“one book caused a whole winter” oversimplifies; the XOR limit itself is a checkable fact"))
    pages.append((g, T("歷史", "History"), {
        0: T("1958 年 Rosenblatt 提出 Perceptron，第一個會學習的神經元。", "In 1958 Rosenblatt proposed the Perceptron, the first neuron that learns."),
        1: T("1960 年 Widrow 與 Hoff 提出 ADALINE，改用連續的 LMS 損失。", "In 1960 Widrow and Hoff proposed ADALINE, using a continuous LMS loss."),
        2: T("1969 年 Minsky 與 Papert 分析單層的限制，XOR 成為最有名的反例。之後的出路是多層加非線性活化。", "In 1969 Minsky and Papert analyzed the single-layer limit, and XOR became the best-known counterexample. The way out afterwards was multiple layers with nonlinear activation."),
        3: T("第一章提醒過：XOR 限制是可驗算的數學事實，但一本書單獨造成整個研究寒冬是過度簡化。", "Chapter 1 reminded us: the XOR limit is a checkable mathematical fact, but a single book causing a whole research winter is an oversimplification.")}))

    # ============ 7: two limits, two exits ============
    g = new(7, T("兩個限制，兩條出路", "Two limits, two ways out"))
    g.beat = 1
    l1 = g.mark(); g.rect(50, 140, 540, 150, fill=REDL, stroke=RED, sw=3, radius=16, z=3)
    g.text(50, 150, 540, 130, T("限制 1\n邊界只是一條直線", "limit 1\nthe boundary is only a straight line"), size=28, color=RED, bold=True, align="center", z=8)
    g.beat = 2
    r1 = g.mark(); g.rect(690, 140, 540, 150, fill=PURL, stroke=PUR, sw=3, radius=16, z=3)
    g.text(690, 150, 540, 130, T("出路\n多層加非線性(KP05)", "way out\nlayers with nonlinearity (KP05)"), size=28, color=PUR, bold=True, align="center", z=8)
    g.link(l1, r1, 4, 12, PUR, 5)
    g.beat = 3
    l2 = g.mark(); g.rect(50, 360, 540, 150, fill=REDL, stroke=RED, sw=3, radius=16, z=3)
    g.text(50, 370, 540, 130, T("限制 2\n只求全對，不挑好的一條", "limit 2\nit wants all-right, not the best line"), size=28, color=RED, bold=True, align="center", z=8)
    g.beat = 4
    r2 = g.mark(); g.rect(690, 360, 540, 150, fill=GRNL, stroke=GRN, sw=3, radius=16, z=3)
    g.text(690, 370, 540, 130, T("出路\n連續損失的 ADALINE(KP08)", "way out\nADALINE with a continuous loss (KP08)"), size=28, color=GRN, bold=True, align="center", z=8)
    g.link(l2, r2, 4, 12, GRN, 5)
    banner(g, T("直線限制靠結構解決，泛化限制靠換一種損失解決", "structure fixes the line limit; a different loss fixes the generalization limit"))
    pages.append((g, T("兩條出路", "Two ways out"), {
        0: T("第一個限制：邊界只是一條直線。", "The first limit: the boundary is only a straight line."),
        1: T("出路是多層加非線性活化，這是下一個單元 KP05。", "The way out is layers with nonlinear activation, the next unit KP05."),
        2: T("第二個限制：Perceptron 只求全對，不挑好的一條。", "The second limit: the Perceptron wants all-right and does not pick the best line."),
        3: T("出路是換一種會挑好直線的方法，例如連續損失的 ADALINE，在 KP08。", "The way out is a method that picks a good line, such as ADALINE with a continuous loss, in KP08.")}))

    # ============ 8: summary ============
    g = new(8, T("可分只保證會停，不保證停得好", "Separable only guarantees it stops, not that it stops well"))
    g.beat = 1
    g.rect(50, 130, 570, 230, fill=REDL, stroke=RED, sw=3, radius=16, z=3)
    g.text(60, 140, 550, 60, T("迷思", "myth"), size=24, color=RED, bold=True, align="center", z=8)
    g.text(60, 196, 550, 160, T("「資料可分，Perceptron 就一定學得好」\n可分只保證會停，不保證停在哪裡", "“separable data means the Perceptron learns well”\nseparable only guarantees it stops, not where"), size=24, color=INK, bold=True, align="center", z=8)
    g.beat = 2
    g.rect(660, 130, 570, 230, fill=ORGL, stroke=ORG, sw=3, radius=16, z=3)
    g.text(670, 140, 550, 60, T("迷思", "myth"), size=24, color=ORG, bold=True, align="center", z=8)
    g.text(670, 196, 550, 160, T("「學不會 XOR 是資料太少或學習率不對」\n不是，是表示力的限制", "“XOR fails from too little data or a bad rate”\nno: it is a limit of expressiveness"), size=24, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(50, 400, 1000, 200, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(70, 410, 960, 180, T("XOR 可以寫成「至少一個為 1，而且不是兩個都為 1」。能不能把幾個神經元接起來，讓前一個的輸出當下一個的輸入？", "XOR can be written “at least one is 1, and not both are 1”. Can we connect several neurons so that one's output is the next one's input?"), size=24, color=INK, bold=True, align="center", z=8)
    mascot(g, 1100, 410, 0.55, "confused")
    banner(g, T("下一個單元:把神經元接成層", "next unit: stack neurons into layers"))
    pages.append((g, T("重點與問題", "Summary and question"), {
        0: T("第一個迷思：資料可分，Perceptron 就一定學得好。可分只保證會停，不保證停在哪裡。", "First myth: separable data means the Perceptron learns well. Separable only guarantees it stops, not where."),
        1: T("第二個迷思：學不會 XOR 是因為資料太少或學習率不對。不是，是表示力的限制。", "Second myth: XOR fails because of too little data or a bad rate. No, it is a limit of expressiveness."),
        2: T("留給下一個單元：XOR 可以寫成至少一個為 1，而且不是兩個都為 1。能不能把幾個神經元接起來？", "For the next unit: XOR is at least one is 1 and not both are 1. Can we connect several neurons?"),
        3: T("", "")}))
    return pages
