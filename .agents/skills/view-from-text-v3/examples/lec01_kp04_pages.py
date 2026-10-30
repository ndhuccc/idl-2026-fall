import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *
from lec01_kp04_nums import *

KP = "KP04"; N = 9
GC = ["#1d4ed8", "#c2410c", "#15803d", "#6d28d9", "#b91c1c"]

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    def cross_mark(g, x, y, r, color, z=7):
        g.line([(x - r, y - r), (x + r, y + r)], color, sw=4, z=z); g.line([(x - r, y + r), (x + r, y - r)], color, sw=4, z=z)
    def axes(g, ox, oy, w, h, xmax, ymax, xt=(0, 5, 10, 15), yt=(0, 2, 4)):
        g.line([(ox, oy), (ox + w, oy)], MUTED, sw=3, z=3); g.line([(ox, oy), (ox, oy - h)], MUTED, sw=3, z=3)
        for t in xt: g.text(ox + t / xmax * w - 18, oy + 4, 36, 24, str(t), size=16, color=MUTED, align="center", z=8)
        for t in yt: g.text(ox - 34, oy - t / ymax * h - 12, 28, 24, str(t), size=16, color=MUTED, align="right", z=8)
        return (lambda x: ox + x / xmax * w), (lambda y: oy - y / ymax * h)

    # ============ 1 ============
    g = new(1, T("KP04 無標籤資料也能找結構", "KP04 Unlabeled data can still reveal structure"))
    g.beat = 1
    card(g, 50, 120, 570, 500, GRAY); head(g, 70, 126, 480, T("沒有名字的一堆水果", "A pile of unnamed fruit"), MUTED, 22); pill(g, 480, 132, 120, T("比喻 · 示意", "Analogy"), GRAY, size=16)
    pos = [(150, 250), (250, 220), (200, 330), (330, 300), (430, 240), (500, 330), (170, 440), (280, 430), (390, 450), (480, 460), (350, 360), (120, 360)]
    for i, (x, y) in enumerate(pos): g.ellipse(x, y, 26, fill="#e2e8f0", stroke=GRAY, sw=3, z=3)
    g.beat = 2
    arrow(g, (630, 370), (700, 370), ORG, sw=6, head=22); g.text(620, 300, 100, 60, T("依外觀\n分堆", "group by\nlook"), size=20, color=ORG, bold=True, align="center", z=8)
    card(g, 710, 120, 520, 500, ORG); head(g, 730, 126, 480, T("分成幾堆", "Sorted into piles"), ORG, 22)
    groups = [((820, 280), 90, BLUE, 5), ((1100, 270), 80, ORG, 4), ((960, 470), 100, GRN, 3)]
    for (cx, cy), r, col, k in groups:
        g.ellipse(cx, cy, r + 30, stroke=col, sw=5, dash=True, z=3)
        for j in range(k): a = j * 2 * math.pi / k; g.ellipse(cx + r * 0.55 * math.cos(a), cy + r * 0.55 * math.sin(a), 22, fill=col, stroke="#ffffff", sw=2, z=5)
    g.beat = 3
    banner(g, T("沒有答案可對照，也仍要說清楚怎麼比「相似」", "No answers to check against, but we still must define “similar”"))
    pages.append((g, T("無標籤與分群", "Unlabeled data and clustering"), {
        0: T("KP03 最後把標籤拿掉。這個單元問：只剩輸入，還能做什麼？先看分群。（水果是比喻，不是講義的資料。）", "KP03 ended by removing the labels. This unit asks what we can do with inputs only. First, clustering. (The fruit is an analogy, not the note's data.)"),
        1: T("一堆沒有名字的水果。", "A pile of unnamed fruit."),
        2: T("依外觀把它們分成幾堆，這就是分群。", "Grouping them by appearance is clustering."),
        3: T("來源用各 5 個蘋果和黃瓜的比喻，是不先告訴類名，不是說沒有任何先備假設。仍要決定怎麼量相似度、分幾群。", "The source's 5 apples and 5 cucumbers illustrate withholding category names, not removing all assumptions. Similarity and the number of groups must still be specified.")}))

    # ============ 2 ============
    g = new(2, T("分群給的是群號，不是類名", "Clustering gives group numbers, not class names"))
    g.beat = 1
    card(g, 90, 110, 1100, 100, GRAY); g.text(110, 112, 1060, 96, r"$f_\theta:\mathbb R^d\to\{\ell_1,\ldots,\ell_m\}$", size=44, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    card(g, 50, 250, 570, 350, BLUE); head(g, 70, 256, 530, T("分類：類名是事先給的", "Classification: names are given"), BLUE, 22)
    g.ellipse(95, 385, 16, fill="#64748b", z=5); arrow(g, (118, 385), (172, 385), GRAY, sw=4, head=14)
    for i, nm in enumerate((T("蘋果", "apple"), T("黃瓜", "cucumber"))): g.ellipse(250 + i * 190, 385, 56, fill=BLUEL, stroke=BLUE, sw=3, z=3); g.text(194 + i * 190, 345, 112, 80, nm, size=24 if zh else 20, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.text(70, 470, 530, 90, T("訓練時有「正確類名」可對照", "training has “correct names” to compare"), size=22, color=INK, bold=True, align="center", z=8)
    card(g, 660, 250, 570, 350, ORG); head(g, 680, 256, 530, T("分群：只有群號", "Clustering: only group numbers"), ORG, 22)
    g.ellipse(705, 385, 16, fill="#64748b", z=5); arrow(g, (728, 385), (782, 385), GRAY, sw=4, head=14)
    for i in range(2): g.ellipse(840 + i * 190, 385, 56, fill=ORGL, stroke=ORG, sw=3, z=3); g.text(784 + i * 190, 345, 112, 80, rf"$\ell_{i + 1}$", size=36, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.text(680, 470, 530, 90, T("群號不是事先提供的真實類名", "group numbers are not ground-truth names"), size=22, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    banner(g, T("要先決定：怎麼量相似度、分幾群、怎麼評分", "Still to decide: the similarity, the cluster count, the scoring"))
    pages.append((g, T("群號不是類名", "Group numbers are not names"), {
        0: T("非監督式學習不使用每筆的目標類別來訓練。分群的輸出形式像分類：d 維輸入，對應到 m 個群號之一。", "Unsupervised learning does not use target classes for training. Clustering resembles classification in form: a d-dimensional input maps to one of m group identifiers."),
        1: T("分類的類名是事先提供的，訓練時有正確的名字可以對照。", "In classification the names are supplied; training has correct names to compare."),
        2: T("分群只有群號，群號不是預先提供的真實類名。", "Clustering has only identifiers, not supplied ground-truth names."),
        3: T("所以仍須先決定怎麼量相似度、要分幾群、如何接受一個分群結果。", "So we must still decide how to measure similarity, how many groups, and how to accept a result.")}))

    # ============ 3 ============
    g = new(3, T("同樣 15 個點，可以看成不同粒度的堆", "The same 15 points can form piles at different granularity"))
    g.beat = 1
    card(g, 50, 120, 760, 500, BLUE); head(g, 70, 126, 700, T("講義的 15 個教學點（無單位）", "The note's 15 teaching points (unitless)"), BLUE, 22)
    X, Y = axes(g, 110, 520, 660, 330, 15, 4, xt=(0, 5, 10, 15), yt=(0, 2, 4))
    for x, y in P15: g.ellipse(X(x), Y(y), 11, fill="#64748b", stroke="#ffffff", sw=2, z=5)
    g.text(110 + 665, 524, 50, 26, r"$x_1$", size=20, color=MUTED, z=8); g.text(120, 175, 50, 26, r"$x_2$", size=20, color=MUTED, z=8)
    g.beat = 2
    for (a, b_), col in (((0.6, 5.6), BLUE), ((9.4, 14.6), GRN)): g.rect(X(a) - 8, Y(3.6), X(b_) - X(a) + 16, Y(1.4) - Y(3.6), stroke=col, sw=4, dash=True, radius=20, z=3, fill="transparent")
    g.beat = 3
    card(g, 840, 120, 390, 500, ORG); head(g, 860, 126, 350, T("目測", "By eye"), ORG, 22)
    g.text(860, 190, 350, 340, T("左邊 9 點靠得近，\n右邊 6 點靠得近。\n\n但「幾堆」\n要不要再細分？\n\n要由準則決定。", "9 points close on the left,\n6 close on the right.\n\nBut how many piles,\nand how finely to split?\n\nA criterion decides."), size=22, color=INK, bold=True, z=8)
    banner(g, T("座標不變，只改分的粒度", "Same coordinates, only the granularity changes"))
    pages.append((g, T("15 個教學點", "The 15 teaching points"), {
        0: T("看講義來源互動的 15 個無單位教學點。它們不是有真實類別的測量。", "Look at the source's 15 unitless teaching points. They are not measurements with known true classes."),
        1: T("目測可以先看出左邊一團、右邊一團。", "By eye, one cluster on the left and one on the right."),
        2: T("但要不要再細分，沒有標準答案，要由準則決定。", "Whether to split further has no single answer; a criterion decides."),
        3: T("座標不變，只改要分幾群。", "The coordinates stay fixed; only the cluster count changes.")}))

    # ============ 4:k-means 兩步 ============
    g = new(4, T("k-means 輪流做兩件事：指派、更新", "k-means alternates two steps: assign, update"))
    asg0 = [min(range(2), key=lambda j: d(p, INIT[2][j])) for p in P15]
    cen1 = [tuple(sum(p[k] for p, a in zip(P15, asg0) if a == j) / asg0.count(j) for k in (0, 1)) for j in range(2)]
    assert [round(v, 3) for c in cen1 for v in c] == [3.833, 2.333, 11.833, 2.333]
    def panel(x0, ttl, col, centers, drawmove, beat):
        g.beat = beat
        card(g, x0, 120, 570, 440, col); head(g, x0 + 20, 126, 530, ttl, col, 22)
        X, Y = axes(g, x0 + 50, 470, 480, 240, 15, 4, xt=(0, 5, 10, 15), yt=())
        for (x, y), a in zip(P15, asg0): g.ellipse(X(x), Y(y), 9, fill=GC[a], stroke="#ffffff", sw=2, z=5)
        for j, c in enumerate(centers): cross_mark(g, X(c[0]), Y(c[1]), 11, GC[j], 7)
        if drawmove:
            for j in range(2): g.line([(X(INIT[2][j][0]), Y(INIT[2][j][1])), (X(cen1[j][0]), Y(cen1[j][1]))], GC[j], sw=4, arrow="end", head=14, dash=True, z=6)
    panel(50, T("① 指派：每點交給最近的中心", "1. Assign: each point to its nearest center"), BLUE, INIT[2], False, 1)
    panel(660, T("② 更新：中心移到所負責點的平均", "2. Update: move each center to its points' mean"), ORG, cen1, True, 2)
    g.beat = 1; g.text(50, 575, 570, 50, T("× 是初始中心（來源固定）", "× marks the fixed initial centers"), size=20, color=MUTED, align="center", z=8)
    g.beat = 2; g.text(660, 575, 570, 50, T(r"新中心約 $(3.83,\,2.33)$ 與 $(11.83,\,2.33)$", r"new centers about $(3.83,\,2.33)$ and $(11.83,\,2.33)$"), size=20, color=MUTED, align="center", z=8)
    g.beat = 3
    banner(g, T("重複到指派不再改變：這是演算法，不是查類名表", "Repeat until assignments stop changing: an algorithm, not a lookup"))
    pages.append((g, T("k-means 兩步", "Two steps of k-means"), {
        0: T("來源用 k-means。指派：把每個點交給距離最近的中心。更新：把每個中心移到它負責的點的平均。這一頁用 m 等於 2、來源固定的初始中心。", "The source uses k-means. Assignment gives each point to its nearest center; update moves each center to the mean of its points. This page uses m equal to 2 and the source's fixed initial centers."),
        1: T("指派：藍色點離左邊的初始中心近，橘色點離右邊的近。", "Assignment: blue points are closer to the left initial center, orange to the right."),
        2: T("更新：兩個中心移到各自點的平均，約 3.83 與 11.83。", "Update: each center moves to the mean of its points, about 3.83 and 11.83."),
        3: T("重複這兩步，直到指派不再改變。圖中中心以叉號標示，不把中心稱為某類水果的真值。", "Repeat until assignments stop changing. Centers are marked with crosses, not interpreted as true fruit categories.")}))

    # ============ 5:m=2,3,5 ============
    g = new(5, T("群數不同，同一批點分成不同的堆", "Different cluster counts split the same points differently"))
    for mi, m in enumerate((2, 3, 5)):
        g.beat = mi + 1
        x0 = 50 + mi * 400
        card(g, x0, 120, 380, 440, GC[mi])
        head(g, x0 + 15, 126, 350, T(f"$m={m}$ 群", f"$m={m}$ clusters"), GC[mi], 24)
        asg, cen = RES[m]
        X, Y = axes(g, x0 + 35, 400, 320, 128, 15, 4, xt=(0, 5, 10, 15), yt=())
        for (x, y), a in zip(P15, asg): g.ellipse(X(x), Y(y), 8, fill=GC[a], stroke="#ffffff", sw=2, z=5)
        for j, c in enumerate(cen): cross_mark(g, X(c[0]), Y(c[1]), 9, GC[j], 7)
        g.text(x0 + 15, 440, 350, 40, T("各群人數：", "sizes:") + " " + " / ".join(str(v) for v in sizes[m]), size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
        g.text(x0 + 15, 490, 350, 56, T("每圖仍是 15 個點", "each panel still has 15 points"), size=18, color=MUTED, align="center", z=8)
    g.beat = 3
    banner(g, T("m 增加，也不一定形成巢狀的群：初始化與局部解會影響", "More clusters need not be nested: initialization and local solutions matter"))
    pages.append((g, T("不同群數", "Different cluster counts"), {
        0: T("m 等於 2：初始中心固定，最終分成 9 點和 6 點。", "m equal to 2: with the fixed initial centers, the groups end with 9 and 6 points."),
        1: T("m 等於 3：分成 3、6、6 點。講義的敘述寫「左側 1 點與 6 點」，和來源的 15 點程式不一致；按原座標與原種子重算是 3、6、6，這是講義更正的地方。", "m equal to 3: the groups have 3, 6 and 6 points. The source narrative mentions 1 and 6 left-side points, inconsistent with its 15-point code; recomputing from the original coordinates and seeds gives 3, 6, 6."),
        2: T("m 等於 5：每群 3 點。座標與範圍相同；群號跨圖不代表同一類。", "m equal to 5: 3 points per group. Axes are identical; group identifiers do not mean the same category across panels."),
        3: T("這個例子可以細分，不代表任何 k-means 在 m 增加時都會形成巢狀群；初始化與局部解也可能改變結果。", "This example admits finer groups, but arbitrary k-means solutions need not be nested as m grows; initialization and local solutions matter.")}))

    # ============ 6:距離 ============
    g = new(6, T("距離要按指定的中心算，不照錯誤的表", "Distances come from the specified centers"))
    g.beat = 1
    card(g, 50, 120, 560, 500, BLUE); head(g, 70, 126, 520, T("2 個指定中心、5 個點", "2 specified centers, 5 points"), BLUE, 22)
    X, Y = axes(g, 110, 570, 440, 360, 8, 8, xt=(0, 2, 4, 6, 8), yt=(0, 2, 4, 6, 8))
    cross_mark(g, X(A[0]), Y(A[1]), 13, BLUE, 7); cross_mark(g, X(B[0]), Y(B[1]), 13, ORG, 7)
    g.text(X(A[0]) + 14, Y(A[1]) + 6, 80, 28, "A", size=22, color=BLUE, bold=True, z=8); g.text(X(B[0]) + 14, Y(B[1]) - 34, 80, 28, "B", size=22, color=ORG, bold=True, z=8)
    g.beat = 2
    for q, nr in zip(Q5, near):
        c = A if nr == "A" else B
        if q != c: g.line([(X(q[0]), Y(q[1])), (X(c[0]), Y(c[1]))], BLUE if nr == "A" else ORG, sw=3, dash=True, z=4)
        g.ellipse(X(q[0]), Y(q[1]), 9, fill=BLUE if nr == "A" else ORG, stroke="#ffffff", sw=2, z=5)
    g.beat = 3
    card(g, 650, 120, 580, 500, ORG); head(g, 670, 126, 540, T("先算距離，再比較", "Compute first, then compare"), ORG, 22)
    g.text(670, 180, 540, 70, r"$\operatorname{dist}((1,1),(6,6))=\sqrt{25+25}=\sqrt{50}\approx7.0711$", size=22, color=INK, align="center", vcenter=True, z=8)
    hdr = (T("輸入", "input"), T("到 A", "to A"), T("到 B", "to B"), T("最近", "nearest"))
    cx = [680, 800, 930, 1070]
    for k, h in enumerate(hdr): g.text(cx[k], 268, 130 if k else 110, 30, h, size=18, color=MUTED, bold=True, z=8)
    for i, (q, (da, db), nr) in enumerate(zip(Q5, dist, near)):
        y = 304 + i * 46
        g.rect(670, y - 4, 540, 40, fill="#fff7ed" if i == 3 else "#f8fafc", stroke=RED if i == 3 else "transparent", sw=2 if i == 3 else 0, radius=8, z=2)
        g.text(cx[0], y, 110, 32, rf"$({q[0]},{q[1]})$", size=20, color=INK, vcenter=True, z=8); g.text(cx[1], y, 120, 32, rf"${da:.4f}$", size=20, color=INK, vcenter=True, z=8)
        g.text(cx[2], y, 120, 32, rf"${db:.4f}$", size=20, color=RED if i == 3 else INK, bold=(i == 3), vcenter=True, z=8); g.text(cx[3], y, 80, 32, nr, size=22, color=BLUE if nr == "A" else ORG, bold=True, vcenter=True, z=8)
    g.text(670, 548, 540, 60, T("(6, 6) 就是中心 B：距離必為 0，來源表的 1.4 不成立", "(6, 6) is center B: its distance is exactly 0, so the source's 1.4 cannot hold"), size=20, color=RED, bold=True, z=8)
    banner(g, T("固定中心與點，重算距離；不換中心來遷就錯表", "Keep centers and points, recompute; never change centers to fit a wrong table"))
    pages.append((g, T("按中心重算距離", "Recompute distances"), {
        0: T("來源教授資料的 5 點例子：指定中心 A 是 (1,1)、B 是 (6,6)，但來源表列出的距離不一致。這一頁固定原中心與原點，重算歐氏距離。", "The guide's 5-point example: the specified centers are A at (1,1) and B at (6,6), but the source table lists inconsistent distances. We keep centers and points and recompute Euclidean distances."),
        1: T("每個點連到最近的中心：藍色虛線去 A，橘色虛線去 B。", "Each point connects to its nearest center: blue dashed lines to A, orange to B."),
        2: T("逐步算：第一點到 B 是 根號 25 加 25，等於根號 50，約 7.0711。特別檢查 (6,6)：它就是 B 中心，到 B 的距離必為 0，來源表的 1.4 不成立。這 5 點只做一次最近中心指派，和前面 15 點跑到固定指派的互動不同。", "Step by step: the first point to B is the square root of 25 plus 25, which is the square root of 50, about 7.0711. Check (6,6): it is center B, so its distance to B must be 0, and the source's 1.4 cannot hold. These 5 points get one assignment, unlike the converged 15-point example."),
        3: T("所有距離由程式重算。", "All distances were recomputed by program.")}))

    # ============ 7:降維 ============
    g = new(7, T("降維：把 $d$ 個座標縮成 $p$ 個", "Dimensionality reduction: from $d$ coordinates to $p$"))
    pill(g, 890, 62, 140, T("講義補充", "Added note"), GRAY, size=16)
    g.beat = 1
    card(g, 90, 110, 1100, 90, GRAY); g.text(110, 112, 1060, 86, r"$f_\theta:\mathbb R^d\to\mathbb R^p,\qquad p<d$", size=40, color=INK, align="center", vcenter=True, z=8)
    card(g, 50, 230, 760, 370, BLUE); head(g, 70, 236, 720, T(r"2 維：$(x_1,x_2)$", r"2-D: $(x_1,x_2)$"), BLUE, 22)
    X, Y = axes(g, 110, 520, 660, 220, 15, 4, xt=(0, 5, 10, 15), yt=())
    for x, y in P15: g.ellipse(X(x), Y(y), 10, fill="#64748b", stroke="#ffffff", sw=2, z=5)
    g.beat = 2
    for x, y in P15: g.line([(X(x), Y(y) + 12), (X(x), 520 - 1)], "#93c5fd", sw=2, dash=True, z=3)
    g.beat = 3
    card(g, 840, 230, 390, 370, ORG); head(g, 860, 236, 350, T("只留第一座標", "Keep only the first"), ORG, 22)
    g.text(860, 290, 350, 90, r"$z=f(\mathbf x)=x_1$", size=36, color=INK, align="center", vcenter=True, z=8)
    g.text(860, 390, 350, 120, T("這是人為指定的\n特徵選擇，\n不是 PCA，也不是學來的", "a prescribed\nfeature selection,\nnot PCA, not learned"), size=20, color=INK, bold=True, align="center", z=8)
    banner(g, T("低維表示叫嵌入；保留什麼，由準則決定", "The low-dimensional representation is an embedding; a criterion decides what stays"))
    pages.append((g, T("降維與嵌入", "Reduction and embedding"), {
        0: T("分群保留原座標、增加群號；降維是把 d 維資料變成 p 維的表示，p 小於 d。這個低維表示叫嵌入。", "Clustering keeps the coordinates and adds group identifiers; dimensionality reduction maps d-dimensional data to a p-dimensional embedding with p less than d."),
        1: T("這一頁是講義的補充：用同樣 15 點，做最簡單的指定投影。", "This page is a supplement: the simplest prescribed projection on the same 15 points."),
        2: T("每個點垂直往下投到橫軸，只保留第一座標。", "Each point drops straight down onto the horizontal axis, keeping only the first coordinate."),
        3: T("這是特徵選擇的一種降維，不是 PCA，也不是宣稱投影由資料學得。要保留哪些變化，取決於準則。", "This is dimensionality reduction by feature selection, not PCA and not a learned projection. The criterion decides which changes are kept.")}))

    # ============ 8:碰撞 ============
    g = new(8, T("降維會合併差異：15 筆擠進 10 個位置", "Reduction merges differences: 15 samples, 10 positions"))
    g.beat = 1
    card(g, 50, 120, 1180, 330, BLUE); head(g, 70, 126, 700, T("只保留 $x_1$ 之後的一維位置", "The 1-D positions after keeping only $x_1$"), BLUE, 22)
    X, Y = axes(g, 110, 350, 1060, 0, 15, 1, xt=(0, 5, 10, 15), yt=())
    g.line([(110, 350), (1170, 350)], MUTED, sw=4, z=3)
    for x in sorted(set(xs)):
        k = mult[x]
        for j in range(k): g.ellipse(X(x), 330 - j * 30, 12, fill=ORG if k == 2 else "#64748b", stroke="#ffffff", sw=2, z=5)
    g.beat = 2
    for x in sorted(set(xs)):
        if mult[x] == 2:
            g.rect(X(x) - 24, 252, 48, 30, fill=REDL, stroke=RED, sw=2, radius=8, z=6); g.text(X(x) - 24, 252, 48, 30, "×2", size=18, color=RED, bold=True, align="center", vcenter=True, z=8)
    g.beat = 3
    card(g, 50, 470, 1180, 140, ORG)
    g.text(70, 478, 1140, 60, r"$(1,2)\mapsto1,\qquad(1,3)\mapsto1$", size=34, color=INK, align="center", vcenter=True, z=8)
    g.text(70, 538, 1140, 60, T("不同的輸入得到相同的輸出：第二座標的差異不再可辨；15 筆樣本都還在，只是落在 10 個位置", "Different inputs, same output: the second coordinate's difference is lost; all 15 samples remain, in 10 positions"), size=20, color=INK, bold=True, align="center", z=8)
    banner(g, T("碰撞不是刪掉 5 筆樣本，而是它們不能再被分辨", "Collisions do not delete 5 samples; they make them indistinguishable"))
    pages.append((g, T("資訊損失", "Information loss"), {
        0: T("保留第一座標以後，15 個點落在一條線上。", "After keeping the first coordinate, the 15 points sit on one line."),
        1: T("橘色的 5 個位置，各有兩筆樣本重疊，標上乘 2。其他 5 個位置各只有一筆。垂直方向不編碼新特徵，只是為了看見重疊。", "The 5 orange positions each hold two samples, marked times 2; the other 5 hold one each. The vertical stacking encodes no feature; it only shows the overlap."),
        2: T("例如 (1,2) 和 (1,3) 都變成 1：第二座標的差異不再可辨。15 筆樣本仍然存在，落在 10 個位置，碰撞不是刪掉 5 筆樣本。", "For example (1,2) and (1,3) both become 1: the second-coordinate difference is lost. All 15 samples remain, in 10 positions; collisions do not delete 5 samples."),
        3: T("所有位置與次數由程式重算。", "All positions and counts were computed by program.")}))

    # ============ 9 ============
    g = new(9, T("無標籤，也要說明保留什麼", "Even without labels, say what to preserve"))
    g.beat = 1
    for i, (ttl, sub, col, light) in enumerate(((T("相似度", "similarity"), T("距離怎麼量？", "how is distance measured?"), BLUE, BLUEL), (T("群數 $m$", "cluster count $m$"), T("要分幾堆？", "how many piles?"), ORG, ORGL), (T("保留準則", "preservation criterion"), T("降維要留住什麼？", "what must reduction keep?"), GRN, GRNL))):
        g.beat = i + 1
        x = 60 + i * 400
        g.rect(x, 150, 360, 250, fill=light, stroke=col, sw=3, radius=18, z=3); g.text(x, 170, 360, 60, ttl, size=30, color=col, bold=True, align="center", vcenter=True, z=8); g.text(x + 20, 290, 320, 90, sub, size=24, color=INK, bold=True, align="center", z=8)
        if i == 0: g.ellipse(x + 100, 255, 14, fill=BLUE, z=5); g.ellipse(x + 260, 255, 14, fill=BLUE, z=5); g.line([(x + 118, 255), (x + 242, 255)], BLUE, sw=4, arrow="both", head=14, z=4)
        if i == 1:
            for k in range(3): g.ellipse(x + 110 + k * 70, 255, 18, fill=[BLUE, ORG, GRN][k], z=5)
        if i == 2: star(g, x + 180, 255, 22, GOLD, z=6, stroke="#b45309", sw=2)
    g.beat = 3
    g.rect(60, 430, 1160, 150, fill=CARD, stroke=GRAY, sw=2, radius=14, z=2)
    g.text(80, 440, 1120, 130, T("無監督降維只看輸入；監督降維可以加入類別或其他側資訊。\n費雪判別分析用到標籤，和「沒有標籤」的設定不同。", "Unsupervised reduction uses inputs only; supervised reduction can use class labels or other side information.\nFisher discriminant analysis uses labels, unlike the unlabeled setting."), size=22, color=INK, bold=True, align="center", z=8)
    banner(g, T(r"下一站 KP05：額外的回饋，能幫我們保留什麼結構？$\to$", r"Next, KP05: what structure can extra feedback help preserve? $\to$"))
    pages.append((g, T("無標籤也要有準則", "Criteria without labels"), {
        0: T("沒有訓練標籤，仍然要指定假設與評分準則。", "Without target labels, assumptions and scoring criteria are still necessary."),
        1: T("第一，相似度：距離怎麼量？", "First, similarity: how is distance measured?"),
        2: T("第二，群數：要分幾堆？降維時還要說明要保留什麼。", "Second, the cluster count; for reduction, say what to preserve."),
        3: T("無監督降維只看輸入；監督降維可以加入類別或其他側資訊。下一個單元問：額外的回饋能幫我們保留什麼？講義的 4 題小練習留在講義。", "Unsupervised reduction uses inputs; supervised reduction can use labels or side information. The next unit asks what structure extra feedback helps preserve. The note's practice questions stay in the note.")}))
    return pages
