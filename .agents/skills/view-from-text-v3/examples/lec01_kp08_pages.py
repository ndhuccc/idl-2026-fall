import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *
from lec01_kp08_nums import *

KP = "KP08"; N = 9

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    # ============ 1:時間軸 ============
    g = new(1, T("KP08 從邏輯單元走向可學權重", "KP08 From logic units to learnable weights"))
    ev = [(1943, BLUE, T("邏輯形式", "logic form"), T("全或無單元\n組成邏輯網路", "all-or-none units\nin logic networks")), (1949, ORG, T("連結可塑性", "plastic links"), T("Hebb：活動關聯\n如何改變連結", "Hebb: how co-activity\nchanges links")), (1958, GRN, T("感知機", "Perceptron"), T("Rosenblatt：可調連結\n從例子辨識", "Rosenblatt: adjustable\nlinks learn from examples")),
          (1960, PUR, "ADALINE", T("Widrow–Hoff：\nLMS 更新", "Widrow–Hoff:\nLMS update")), (1969, RED, T("單層限制", "single-layer limit"), T("Minsky–Papert：\n特定架構的能力限制", "Minsky–Papert:\nlimits of specific architectures")), (GOLD and 1986, GOLD, T("多層誤差傳遞", "multilayer error flow"), T("Rumelhart 等：\n多層誤差反傳", "Rumelhart et al.:\nmultilayer backprop"))]
    g.beat = 1
    g.line([(80, 300), (1200, 300)], GRAY, sw=7, z=2, arrow="end", head=22)
    for i, (yr, col, nm, sub) in enumerate(ev):
        g.beat = 1 if i < 2 else (2 if i < 4 else 3)
        x = 130 + i * 205
        g.ellipse(x, 300, 20, fill=col, stroke="#ffffff", sw=4, z=5)
        g.text(x - 70, 215, 140, 40, str(yr), size=28, color=col if col != GOLD else "#92400e", bold=True, align="center", vcenter=True, z=8)
        g.rect(x - 95, 345, 190, 200, fill=CARD, stroke=col, sw=3, radius=14, z=3)
        g.text(x - 90, 355, 180, 60, nm, size=21 if zh else 18, color=col if col != GOLD else "#92400e", bold=True, align="center", z=8); g.text(x - 90, 425, 180, 110, sub, size=17, color=INK, align="center", z=8)
    g.beat = 3
    banner(g, T("問題從「能算什麼」，走向「怎麼調權重」", "The question moved from “what can it compute” to “how are weights adjusted”"))
    pages.append((g, T("早期時間軸", "The early timeline"), {
        0: T("先看時間軸。箭頭表示課堂問題逐步改變，不是每篇文獻的直接承繼關係。1943 年 McCulloch 與 Pitts 研究由全或無單元組成的邏輯網路；這是計算模型，還沒有從樣本調參數的訓練法。", "Start with the timeline. The arrow shows the classroom question changing, not direct lineage between papers. In 1943 McCulloch and Pitts studied logic networks of all-or-none units; a model of computation with no learning from samples yet."),
        1: T("1949 年 Hebb 討論活動關聯如何改變連結，是學習規則的生物學靈感；「一起活躍，連結變強」是後來的概括，不是完整的最佳化公式。1958 年 Rosenblatt 的感知機，讓可調連結接受例子。", "In 1949 Hebb discussed how co-activity changes connections, a biological inspiration for learning rules; “fire together, wire together” is a later summary, not an optimization formula. In 1958 Rosenblatt's perceptron let adjustable links learn from examples."),
        2: T("1960 年 Widrow 與 Hoff 的 ADALINE；1969 年 Minsky 與 Papert 分析特定感知機架構的限制；1986 年多層誤差反傳的文獻是重要里程碑。", "In 1960 Widrow and Hoff's ADALINE; in 1969 Minsky and Papert analyzed limits of specific perceptron architectures; the 1986 multilayer error back-propagation paper is a milestone."),
        3: T("從「能算什麼」走向「如何調權重」，不表示每項工作都由前一項直接推導。", "Moving from what can be computed to how weights are adjusted does not mean each work follows directly from the previous one.")}))

    # ============ 2:門檻單元 ============
    g = new(2, T("門檻單元：加權和過了門檻就輸出 1", "Threshold unit: output 1 past the threshold"))
    g.beat = 1
    for i, nm in enumerate((r"$x_1$", r"$x_2$")):
        g.ellipse(130, 250 + i * 200, 46, fill=BLUEL, stroke=BLUE, sw=4, z=3); g.text(84, 210 + i * 200, 92, 80, nm, size=34, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.beat = 2
    for i, nm in enumerate((r"$w_1$", r"$w_2$")):
        g.line([(180, 250 + i * 200), (440, 350)], ORG, sw=6, arrow="end", head=18, z=4); g.text(250, 245 + i * 150 + (0 if i == 0 else 20), 90, 40, nm, size=26, color=ORG, bold=True, align="center", vcenter=True, z=8)
    g.ellipse(520, 350, 90, fill=ORGL, stroke=ORG, sw=5, z=3); g.text(440, 300, 160, 100, r"$\sum+b$", size=34, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.beat = 3
    arrow(g, (620, 350), (730, 350), GRAY, sw=6, head=20)
    g.poly([(740, 270), (740, 430), (860, 350)], fill=GRNL, stroke=GRN, sw=5, z=3); g.text(735, 320, 90, 60, r"$>0?$", size=28, color=INK, bold=True, align="center", vcenter=True, z=8)
    arrow(g, (868, 350), (960, 350), GRN, sw=6, head=20); g.rect(970, 300, 180, 100, fill=GRNL, stroke=GRN, sw=4, radius=14, z=3); g.text(970, 300, 180, 100, r"$\hat y\in\{0,1\}$", size=26, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.rect(130, 520, 1020, 100, fill=CARD, stroke=GRAY, sw=2, radius=14, z=2); g.text(150, 524, 980, 92, r"$\hat y=\mathbf 1[\mathbf w^{\top}\mathbf x+b>0]$", size=40, color=INK, align="center", vcenter=True, z=8)
    banner(g, T("AND、OR 要選不同的權重與門檻；這是現代教學的等價寫法", "AND and OR need different weights and thresholds; a modern teaching form"))
    pages.append((g, T("門檻單元", "The threshold unit"), {
        0: T("對二元輸入 x，門檻單元把輸入先乘上權重，再加偏置。", "For a binary input x, a threshold unit multiplies inputs by weights and adds a bias."),
        1: T("權重 w 決定每個輸入的貢獻，偏置 b 平移門檻。", "The weights w set each input's contribution; the bias b shifts the threshold."),
        2: T("總和大於 0 就輸出 1，否則輸出 0。指示函數的寫法是現代教學的等價寫法，不是聲稱 McCulloch 與 Pitts 的原文把所有權重固定相等，或已給出感知機更新。用它表示 AND、OR 要選不同的權重與門檻；NOT 可用抑制性連結或有符號權重表示。", "If the sum is greater than 0 the output is 1, otherwise 0. The indicator notation is a modern teaching form, not a claim that the 1943 paper fixed all weights equal or gave the Perceptron update. AND and OR need different weights and thresholds; NOT can use inhibitory links or signed weights."),
        3: T("這個單元算得了邏輯，卻還沒有「怎麼學」。", "This unit can compute logic but has no way of learning yet.")}))

    # ============ 3:XOR 四點 ============
    g = new(3, T("XOR：對角線的兩個 1，被兩個 0 夾住", "XOR: two 1s on the diagonal, separated by two 0s"))
    pill(g, 890, 62, 140, T("講義補充", "Added note"), GRAY, size=16)
    g.beat = 1
    card(g, 50, 120, 640, 480, BLUE); head(g, 70, 126, 600, T("四個輸入與標籤", "Four inputs and labels"), BLUE, 22)
    ox, oy, S = 160, 540, 280
    g.line([(ox - 40, oy), (ox + S + 100, oy)], MUTED, sw=3, z=2, arrow="end", head=14); g.line([(ox, oy + 40), (ox, oy - S - 70)], MUTED, sw=3, z=2, arrow="end", head=14)
    g.text(ox + S + 90, oy + 6, 60, 30, r"$x_1$", size=22, color=MUTED, z=8); g.text(ox + 10, oy - S - 90, 60, 30, r"$x_2$", size=22, color=MUTED, z=8)
    for (a_, b_), yv in XOR.items():
        cx_, cy_ = ox + a_ * S, oy - b_ * S
        g.ellipse(cx_, cy_, 34, fill=ORGL if yv else BLUEL, stroke=ORG if yv else BLUE, sw=4, z=5); g.text(cx_ - 30, cy_ - 24, 60, 48, str(yv), size=30, color=INK, bold=True, align="center", vcenter=True, z=8)
        g.text(cx_ + 42, cy_ - 14, 100, 30, rf"$({a_},{b_})$", size=16, color=MUTED, z=8)
    g.beat = 2
    g.line([(ox + 40, oy - S * 0.62), (ox + S - 40, oy - S * 0.62 + 0)], RED, sw=5, dash=True, z=6)
    g.line([(ox + 30, oy - 20), (ox + S + 30, oy - S + 30)], RED, sw=5, dash=True, z=6)
    g.beat = 3
    card(g, 730, 120, 500, 480, RED); head(g, 750, 126, 460, T("任何一條直線", "Any straight line"), RED, 22)
    cross(g, 920, 220, 70, RED, z=7)
    g.text(750, 330, 460, 240, T("都至少留下一個錯點\n\n正式的論證，\n要看四條不等式", "leaves at least one point wrong\n\nthe formal argument\nneeds four inequalities"), size=24, color=INK, bold=True, align="center", z=8)
    banner(g, T("單一線性邊界，分不開 XOR 的四個點", "One linear boundary cannot separate XOR's four points"))
    pages.append((g, T("XOR 四點", "XOR's four points"), {
        0: T("這是講義的補充：XOR 的輸入與標籤是 (0,0) 對 0、(1,0) 對 1、(0,1) 對 1、(1,1) 對 0。橘色是 1，藍色是 0。", "A supplement: XOR's inputs and labels are (0,0) to 0, (1,0) to 1, (0,1) to 1, (1,1) to 0. Orange is 1, blue is 0."),
        1: T("對角的兩個 1，被另外兩個 0 夾住。試任何一條直線，例如這兩條虛線，都至少留下一個錯點。", "The two 1s on a diagonal are flanked by the two 0s. Try any straight line, such as these two dashed ones; at least one point is always wrong."),
        2: T("格點搜尋也找不到分得開的直線；但正式論證靠下一頁的四條不等式。", "A grid search also finds no separating line, but the formal argument rests on the four inequalities on the next page."),
        3: T("圖上的虛線只是示意。", "The dashed lines are only a sketch.")}))

    # ============ 4:四條不等式 ============
    g = new(4, T("四條不等式互相矛盾", "Four inequalities that contradict each other"))
    g.beat = 1
    ineq = [(r"$b\le0$", T("$(0,0)\\mapsto0$", "$(0,0)\\mapsto0$"), BLUE), (r"$w_1+b>0$", T("$(1,0)\\mapsto1$", "$(1,0)\\mapsto1$"), ORG), (r"$w_2+b>0$", T("$(0,1)\\mapsto1$", "$(0,1)\\mapsto1$"), ORG), (r"$w_1+w_2+b\le0$", T("$(1,1)\\mapsto0$", "$(1,1)\\mapsto0$"), BLUE)]
    for i, (f_, lab, col) in enumerate(ineq):
        g.beat = 1
        x = 50 + i * 300
        g.rect(x, 130, 280, 150, fill=BLUEL if col == BLUE else ORGL, stroke=col, sw=3, radius=14, z=3); g.text(x, 138, 280, 60, f_, size=30, color=INK, bold=True, align="center", vcenter=True, z=8); g.text(x, 214, 280, 50, lab, size=20, color=col, bold=True, align="center", vcenter=True, z=8)
    g.beat = 2
    g.line([(340, 280), (340, 340)], ORG, sw=4, z=4); g.line([(640, 280), (640, 340)], ORG, sw=4, z=4); g.line([(340, 340), (640, 340)], ORG, sw=4, z=4); arrow(g, (490, 340), (490, 380), ORG, sw=4, head=14)
    g.rect(200, 385, 580, 80, fill=ORGL, stroke=ORG, sw=3, radius=14, z=3); g.text(200, 387, 580, 76, r"$w_1+w_2+2b>0$", size=34, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.beat = 3
    g.rect(130, 490, 1020, 120, fill=REDL, stroke=RED, sw=4, radius=14, z=3); g.text(150, 494, 980, 112, r"$w_1+w_2+b=(w_1+w_2+2b)-b>0$" + "\n" + T(r"與最後一式 $\le0$ 矛盾", r"contradicts the last inequality $\le0$"), size=26, color=INK, bold=True, align="center", z=8)
    star(g, 1090, 470, 26, GOLD, z=6, stroke="#b45309", sw=2) if False else None
    cross(g, 1100, 440, 44, RED, z=8)
    banner(g, T("所以單一線性邊界，無法給這四點正確標籤", "So no single linear boundary labels these four points correctly"))
    pages.append((g, T("四條不等式", "Four inequalities"), {
        0: T("假設單一門檻在 w1 x1 加 w2 x2 加 b 大於 0 時輸出 1。四個點要求：b 不大於 0；w1 加 b 大於 0；w2 加 b 大於 0；w1 加 w2 加 b 不大於 0。", "Suppose one threshold outputs 1 when w1 x1 plus w2 x2 plus b is greater than 0. The four points require: b at most 0; w1 plus b greater than 0; w2 plus b greater than 0; w1 plus w2 plus b at most 0."),
        1: T("把中間兩式相加：w1 加 w2 加 2b 大於 0。", "Adding the middle two: w1 plus w2 plus 2b is greater than 0."),
        2: T("而 w1 加 w2 加 b 等於 w1 加 w2 加 2b 減 b，因為 b 不大於 0，所以它大於 0，與最後一式矛盾。", "And w1 plus w2 plus b equals w1 plus w2 plus 2b minus b; since b is at most 0, it is greater than 0, contradicting the last inequality."),
        3: T("多層且有非線性時，表達能力不同，但仍須解決訓練。這個推論由程式與代數雙重檢查。", "With multiple layers and nonlinearity the expressiveness differs, yet training must still be solved. The deduction was checked by both program and algebra.")}))

    # ============ 5:表示 vs 學習 ============
    g = new(5, T("「能表示」與「能學到」是兩個問題", "“Can represent” and “can learn” are two questions"))
    g.beat = 1
    card(g, 50, 130, 570, 420, BLUE); head(g, 70, 136, 530, T("能表示？", "Can it represent?"), BLUE, 28)
    for i in range(3): g.ellipse(160, 260 + i * 80, 30, fill=BLUEL, stroke=BLUE, sw=3, z=3); g.ellipse(310, 300 + (i % 2) * 90, 30, fill=ORGL, stroke=ORG, sw=3, z=3); g.line([(190, 260 + i * 80), (280, 300 + (i % 2) * 90)], GRAY, sw=2, z=2)
    g.ellipse(470, 345, 30, fill=GRNL, stroke=GRN, sw=3, z=3)
    for j in range(2): g.line([(340, 300 + j * 90), (440, 345)], GRAY, sw=2, z=2)
    g.text(70, 440, 530, 90, T("多層、有非線性：\n表達能力不同，XOR 可以表示", "multiple layers + nonlinearity:\nXOR becomes representable"), size=22, color=INK, bold=True, align="center", z=8)
    g.beat = 2
    card(g, 660, 130, 570, 420, ORG); head(g, 680, 136, 530, T("能學到？", "Can it learn?"), ORG, 28)
    g.rect(740, 230, 410, 100, fill=ORGL, stroke=ORG, sw=3, radius=14, z=3); g.text(740, 230, 410, 100, T("要有可用的\n學習規則", "a workable\nlearning rule"), size=26, color=INK, bold=True, align="center", z=8)
    g.line([(740, 400), (1150, 400)], ORG, sw=6, z=4, arrow="end", head=20); g.ellipse(840, 400, 14, fill=ORG, z=5)
    g.text(680, 440, 530, 90, T("1986 年的多層誤差反傳\n是重要里程碑", "the 1986 multilayer back-propagation\nis a landmark"), size=22, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    g.text(50, 570, 1180, 60, T("1969 年的 XOR 限制是可驗算的數學事實；「一本書單獨造成 AI 冬天」是過度簡化", "The 1969 XOR limitation is a checkable fact; “one book caused the whole AI winter” oversimplifies"), size=20, color=MUTED, bold=True, align="center", z=8)
    banner(g, T("表達能力不等於有訓練演算法", "Expressiveness is not a training algorithm"))
    pages.append((g, T("表示與學習", "Representing vs learning"), {
        0: T("多層、有非線性時，網路的表達能力不同，XOR 可以表示。", "With several layers and nonlinearity the expressive power differs and XOR can be represented."),
        1: T("但能表示，不等於能學到：還需要可用的學習規則。1986 年 Rumelhart、Hinton 與 Williams 的多層誤差反傳文獻是重要里程碑，不是說反向模式的所有構想直到 1986 年才出現。", "But representable does not mean learnable: a workable learning rule is also needed. The 1986 Rumelhart, Hinton and Williams paper is a landmark, not a claim that every idea behind reverse mode appeared only then."),
        2: T("1969 年 Minsky 與 Papert 的《Perceptrons》分析特定感知機架構的能力限制。XOR 限制是可驗算的數學事實；說這本書單獨造成整個 AI 冬天則是過度簡化，多層方法、資料、算力、研究資源等因素各自影響後續發展。", "Minsky and Papert's 1969 Perceptrons analyzed the limits of specific perceptron architectures. The XOR limitation is a checkable mathematical fact; saying the book alone caused the AI winter oversimplifies, since multilayer methods, data, compute and research funding each mattered."),
        3: T("講義的口訣：形式決定能表示什麼，損失與更新決定怎樣學。", "The note's rule of thumb: architecture sets capacity; objective and update decide how it learns.")}))

    # ============ 6:兩種更新規則 ============
    g = new(6, T("感知機看分類對錯，ADALINE 看連續誤差", "Perceptron: right or wrong; ADALINE: error"))
    g.beat = 1
    card(g, 50, 130, 570, 470, BLUE); head(g, 70, 136, 530, T("感知機", "Perceptron"), BLUE, 28)
    g.rect(80, 205, 510, 70, fill=BLUEL, stroke=BLUE, sw=3, radius=12, z=3); g.text(90, 205, 490, 70, r"$\hat y=\operatorname{sign}(a),\quad a=\mathbf w^{\top}\mathbf x+b$", size=26, color=INK, align="center", vcenter=True, z=8)
    arrow(g, (335, 278), (335, 295), BLUE, sw=4, head=12)
    g.rect(80, 300, 510, 130, fill=CARD, stroke=BLUE, sw=2, radius=12, z=3); g.text(90, 304, 490, 58, r"$\mathbf w\leftarrow\mathbf w+\eta\,y\,\mathbf x$", size=26, color=INK, align="center", vcenter=True, z=8); g.text(90, 364, 490, 58, r"$b\leftarrow b+\eta\,y$", size=26, color=INK, align="center", vcenter=True, z=8)
    g.text(80, 450, 510, 100, T(r"只在分錯時更新：$y\,a\le0$", r"update only on a mistake: $y\,a\le0$"), size=24, color=BLUE, bold=True, align="center", z=8)
    g.beat = 2
    card(g, 660, 130, 570, 470, ORG); head(g, 680, 136, 530, "ADALINE", ORG, 28)
    g.rect(690, 205, 510, 70, fill=ORGL, stroke=ORG, sw=3, radius=12, z=3); g.text(700, 205, 490, 70, r"$L=\frac12(y-a)^2$", size=28, color=INK, align="center", vcenter=True, z=8)
    arrow(g, (945, 278), (945, 295), ORG, sw=4, head=12)
    g.rect(690, 300, 510, 130, fill=CARD, stroke=ORG, sw=2, radius=12, z=3); g.text(700, 304, 490, 58, r"$\mathbf w\leftarrow\mathbf w+\eta\,(y-a)\,\mathbf x$", size=26, color=INK, align="center", vcenter=True, z=8); g.text(700, 364, 490, 58, r"$b\leftarrow b+\eta\,(y-a)$", size=26, color=INK, align="center", vcenter=True, z=8)
    g.text(690, 450, 510, 100, T("在閾值之前的線性輸出 $a$ 上\n最小化平方誤差（LMS / Widrow–Hoff）", "minimizes squared error on the linear\noutput $a$ before the threshold (LMS / Widrow–Hoff)"), size=21, color=ORG, bold=True, align="center", z=8)
    g.beat = 3
    banner(g, T("兩者的決策邊界在這裡都仍是線性的", "Both decision boundaries are still linear here"))
    pages.append((g, T("兩種更新", "Two updates"), {
        0: T("設 y 是正負 1 的標籤，a 是權重加權和加偏置。感知機先用 a 的正負號分類；典型的錯分更新是 w 加 eta 乘 y 乘 x，只在 y 乘 a 不大於 0 時更新。", "Let y be a plus-or-minus-1 label and a the weighted sum plus bias. The Perceptron classifies by the sign of a; its typical mistake update is w plus eta times y times x, applied only when y times a is at most 0."),
        1: T("ADALINE 在閾值之前的線性輸出 a 上最小化平方誤差，L 等於 1/2 乘 y 減 a 的平方；LMS 或 Widrow–Hoff 更新是 w 加 eta 乘 (y 減 a) 乘 x。", "ADALINE minimizes squared error on the linear output a before the threshold, L equal to one half of y minus a squared; the LMS or Widrow–Hoff update is w plus eta times (y minus a) times x."),
        2: T("兩者在這裡的決策邊界都仍是線性的。差別在更新的目標。", "Both have linear decision boundaries here; the difference is the update objective."),
        3: T("MADALINE 使用多個 ADALINE；把單元堆起來，和能用誤差有效訓練所有層，不是同一件事。", "MADALINE uses several ADALINEs; stacking units is not the same as training all layers effectively with error.")}))

    # ============ 7:受控樣本 ============
    g = new(7, T("同一個已分對的樣本：感知機不動，ADALINE 還會修正", "A correct sample: Perceptron idle, ADALINE acts"))
    pill(g, 1090, 84, 140, T("講義補充", "Added note"), GRAY, size=16)
    g.beat = 1
    card(g, 50, 124, 1180, 116, GRAY); g.text(70, 124, 1140, 112, T(r"$\mathbf x=(1,0),\ y=+1,\ \mathbf w=(0.2,0),\ b=0,\ \eta=0.1\ \Rightarrow\ a=0.2>0$", r"$\mathbf x=(1,0),\ y=+1,\ \mathbf w=(0.2,0),\ b=0,\ \eta=0.1\ \Rightarrow\ a=0.2>0$"), size=30, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    arrow(g, (335, 242), (335, 266), BLUE, sw=4, head=12)
    card(g, 50, 270, 570, 330, BLUE); head(g, 70, 276, 530, T("感知機：已分對，不更新", "Perceptron: already correct, no update"), BLUE, 22)
    g.rect(120, 350, 190, 70, fill=BLUEL, stroke=BLUE, sw=3, radius=12, z=3); g.text(120, 350, 190, 70, r"$\Delta\mathbf w=(0,0)$", size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.rect(340, 350, 230, 70, fill=BLUEL, stroke=BLUE, sw=3, radius=12, z=3); g.text(340, 350, 230, 70, r"$\Delta b=0$", size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
    tick(g, 320, 470, 56, GRN, z=8)
    g.beat = 3
    arrow(g, (945, 242), (945, 266), ORG, sw=4, head=12)
    card(g, 660, 270, 570, 330, ORG); head(g, 680, 276, 530, T("ADALINE：連續誤差 $y-a=0.8$", "ADALINE: continuous error $y-a=0.8$"), ORG, 22)
    g.rect(690, 350, 250, 70, fill=ORGL, stroke=ORG, sw=3, radius=12, z=3); g.text(690, 350, 250, 70, r"$\Delta\mathbf w=(0.08,\,0)$", size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.rect(960, 350, 240, 70, fill=ORGL, stroke=ORG, sw=3, radius=12, z=3); g.text(960, 350, 240, 70, r"$\Delta b=0.08$", size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.text(690, 450, 510, 60, r"$\mathbf w=(0.28,\,0),\ b=0.08$", size=26, color=ORG, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("這只展示目標不同，不能推出 ADALINE 一定泛化較好", "This only shows the objectives differ; it does not show ADALINE generalizes better"))
    pages.append((g, T("受控樣本", "A controlled sample"), {
        0: T("這是講義的補充：取 x 等於 (1, 0)，y 等於正 1，w 等於 (0.2, 0)，b 等於 0，學習率 0.1。a 等於 0.2，大於 0，這一筆已經分對。", "A supplement: take x equal to (1, 0), y equal to plus 1, w equal to (0.2, 0), b equal to 0 and a learning rate of 0.1. Then a is 0.2, greater than 0, so this sample is already correct."),
        1: T("感知機已經分對，所以不更新：權重和偏置的變化都是 0。", "The Perceptron is already correct, so it does not update: the weight and bias changes are both 0."),
        2: T("ADALINE 還有誤差 y 減 a 等於 1 減 0.2 等於 0.8。權重變化是 0.1 乘 0.8 乘 (1, 0)，等於 (0.08, 0)；偏置加 0.08；新的 w 是 (0.28, 0)，b 是 0.08。", "ADALINE still has error y minus a equal to 1 minus 0.2, which is 0.8. The weight change is 0.1 times 0.8 times (1, 0), equal to (0.08, 0); the bias gains 0.08; the new w is (0.28, 0) and b is 0.08."),
        3: T("此例只展示目標不同，不能推出 ADALINE 一定泛化較好。數值由程式重算。", "This example only shows different objectives; it does not show ADALINE generalizes better. Values were recomputed by program.")}))

    # ============ 8:堆起來不等於會訓練 ============
    g = new(8, T("把單元堆起來，不等於每一層都能被訓練", "Stacking units is not the same as training every layer"))
    pill(g, 960, 96, 270, T("示意：不是講義的資料", "sketch: not the note's data"), GRAY, size=16)
    g.beat = 1
    card(g, 50, 140, 570, 460, PUR); head(g, 70, 146, 530, T("MADALINE：多個 ADALINE", "MADALINE: several ADALINEs"), PUR, 22)
    for c in range(3):
        for r in range(2): g.ellipse(170 + c * 140, 280 + r * 110, 34, fill=PURL, stroke=PUR, sw=3, z=3)
    g.ellipse(330, 480, 34, fill=GRNL, stroke=GRN, sw=3, z=3)
    for c in range(3):
        for r in range(2): g.line([(170 + c * 140, 314 + r * 110 - 34 + 34), (330, 450)], "#c4b5fd", sw=2, z=2)
    g.beat = 2
    card(g, 660, 140, 570, 460, ORG); head(g, 680, 146, 530, T("要用誤差訓練各層", "Training the layers with error"), ORG, 22)
    for i in range(3): g.rect(720 + i * 160, 290, 120, 100, fill=ORGL, stroke=ORG, sw=3, radius=12, z=3); g.text(720 + i * 160, 290, 120, 100, T(f"第 {i + 1} 層", f"layer {i + 1}"), size=22, color=INK, bold=True, align="center", vcenter=True, z=8)
    for i in range(2): arrow(g, (845 + i * 160, 440), (765 + i * 160 if False else 725 + i * 160, 440), RED, sw=4, head=14)
    g.text(680, 470, 530, 100, T("誤差要能一路傳回各層", "the error must flow back to every layer"), size=22, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    banner(g, T("1986 年是重要里程碑，不是「此前沒有任何相關想法」", "1986 is a landmark, not “no related idea existed before”"))
    pages.append((g, T("堆疊與訓練", "Stacking vs training"), {
        0: T("Widrow 與 Hoff 在 1960 年提出 ADALINE 與 LMS 的相關工作；MADALINE 使用多個 ADALINE。圖是示意。", "Widrow and Hoff proposed ADALINE and the LMS work in 1960; MADALINE uses several ADALINEs. The figure is a sketch."),
        1: T("把單元堆起來，與能用誤差有效訓練所有層，不是同一件事。", "Stacking units is not the same as training all layers effectively with error."),
        2: T("1986 年 Rumelhart、Hinton、Williams 的多層誤差反傳文獻是重要里程碑，不是說反向模式的所有構想直到 1986 年才出現。", "The 1986 Rumelhart, Hinton and Williams paper is a landmark, not a claim that every reverse-mode idea first appeared in 1986."),
        3: T("多單元結構與有效訓練各層，是兩個不同的問題。", "Multiunit structure and effective training are two different problems.")}))

    # ============ 9 ============
    g = new(9, T("形式決定能表示什麼，損失與更新決定怎樣學", "Form sets capacity; loss and update set learning"))
    g.beat = 1
    for i, (col, light, ttl, sub) in enumerate(((BLUE, BLUEL, T("形式", "Form"), T("門檻單元、層數、非線性\n→ 能表示什麼", "threshold units, depth, nonlinearity\n→ what can be represented")), (ORG, ORGL, T("損失與更新", "Loss and update"), T("分類對錯、平方誤差\n→ 怎樣學", "right or wrong, squared error\n→ how it learns")))):
        g.beat = i + 1
        x = 60 + i * 600
        g.rect(x, 150, 560, 280, fill=light, stroke=col, sw=3, radius=18, z=3); g.text(x + 20, 170, 520, 70, ttl, size=34, color=col, bold=True, align="center", z=8); g.text(x + 30, 270, 500, 140, sub, size=24, color=INK, bold=True, align="center", z=8)
        star(g, x + 280, 140, 16, GOLD, z=6)
    g.beat = 3
    g.text(60, 470, 1160, 110, T("講義的 4 題自我檢核留在講義。", "The note's 4 self-check questions stay in the note."), size=22, color=MUTED, bold=True, align="center", z=8)
    mascot(g, 560, 450, 0.5, "proud")
    banner(g, T(r"下一站 KP09：影像、序列與生成任務，怎樣改變架構與訓練？$\to$", r"Next, KP09: how images, sequences and generation reshape architecture and training $\to$"))
    pages.append((g, T("口訣與銜接", "Rule of thumb"), {
        0: T("口訣：形式決定能表示什麼；損失與更新決定怎樣學。", "Rule of thumb: the form decides what can be represented; the loss and update decide how it learns."),
        1: T("形式：門檻單元、層數與非線性。", "Form: threshold units, depth and nonlinearity."),
        2: T("損失與更新：感知機看分類對錯，ADALINE 看平方誤差。", "Loss and update: the Perceptron looks at right or wrong, ADALINE at squared error."),
        3: T("下一個單元，比較影像、序列與生成任務怎樣共同改變架構與訓練。", "The next unit follows how images, sequences and generation reshape architecture and training together.")}))
    return pages
