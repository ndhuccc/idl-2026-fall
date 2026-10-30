import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *
from lec01_kp05_nums import *

KP = "KP05"; N = 9

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    def scatter8(g, ox, oy, w, h, color_cls=True):
        S = 40
        cx, cy = ox + w / 2, oy + h / 2
        g.line([(ox, cy), (ox + w, cy)], MUTED, sw=3, z=3, arrow="end", head=14); g.line([(cx, oy + h), (cx, oy)], MUTED, sw=3, z=3, arrow="end", head=14)
        g.text(ox + w - 40, cy + 6, 40, 26, r"$x_1$", size=20, color=MUTED, z=8); g.text(cx + 8, oy - 4, 40, 26, r"$x_2$", size=20, color=MUTED, z=8)
        for (x, y) in C1: g.ellipse(cx + x * 40, cy - y * 70, 11, fill=BLUE, stroke="#ffffff", sw=2, z=5)
        for (x, y) in C2: g.ellipse(cx + x * 40, cy - y * 70, 11, fill=ORG, stroke="#ffffff", sw=2, z=5)
        return cx, cy

    # ============ 1 ============
    g = new(1, T("KP05 回饋型態改變學習目標", "KP05 The kind of feedback changes the goal"))
    cards = [(BLUE, BLUEL, T("資料自己的形狀", "the data's own shape"), T("只有輸入點", "inputs only")), (ORG, ORGL, T("標籤（側資訊）", "labels (side information)"), T("告訴你哪些區別要留", "say which distinctions to keep")), (GRN, GRNL, T("獎勵", "rewards"), T("評估行動的後果", "judge the consequences of actions"))]
    for i, (col, light, ttl, sub) in enumerate(cards):
        g.beat = i + 1
        x = 50 + i * 400
        card(g, x, 130, 380, 470, col); g.rect(x, 130, 380, 70, fill=light, radius=16, z=2); g.rect(x, 170, 380, 30, fill=light, z=2); g.text(x + 10, 130, 360, 70, ttl, size=24, color=col, bold=True, align="center", vcenter=True, z=8)
        if i == 0:
            for (px, py) in [(100, 300), (150, 340), (130, 400), (250, 310), (290, 360), (300, 430), (200, 450), (110, 470)]: g.ellipse(x + px, py, 14, fill="#64748b", stroke="#ffffff", sw=2, z=5)
        if i == 1:
            for k, (px, py, c) in enumerate([(100, 300, BLUE), (150, 340, BLUE), (130, 400, BLUE), (250, 310, ORG), (290, 360, ORG), (300, 430, ORG), (200, 450, BLUE), (110, 470, ORG)]): g.ellipse(x + px, py, 14, fill=c, stroke="#ffffff", sw=2, z=5)
            g.poly([(x + 175, 250), (x + 215, 250), (x + 235, 270), (x + 215, 290), (x + 175, 290)], fill=GOLD, stroke="#b45309", sw=3, z=6)
        if i == 2:
            g.ellipse(x + 190, 340, 60, fill=GOLD, stroke="#b45309", sw=5, z=5); g.text(x + 130, 300, 120, 80, "+1", size=40, color=INK, bold=True, align="center", vcenter=True, z=8)
            g.line([(x + 110, 450), (x + 190, 420), (x + 270, 450)], GRN, sw=6, arrow="end", head=18, z=6)
        g.text(x + 20, 520, 340, 60, sub, size=22, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    banner(g, T("先問：回饋的訊號從哪裡來？", "First ask: where does the feedback signal come from?"))
    pages.append((g, T("三種回饋", "Three kinds of feedback"), {
        0: T("KP04 說明把座標變少會遺失某些差異。這個單元問：保留什麼？答案取決於回饋型態。（圖示是示意。）", "KP04 showed that dropping coordinates erases some distinctions. This unit asks what to preserve; the answer depends on the kind of feedback. (Icons are a sketch.)"),
        1: T("只有輸入點：只能看資料自己的形狀。", "Inputs only: we can only look at the data's own shape."),
        2: T("有標籤：標籤告訴表示模型，哪些區別值得保留。", "With labels: labels tell the representation which distinctions are worth preserving."),
        3: T("獎勵：評估行動的後果，常須把數步合起來看。", "Rewards: they judge the consequences of actions, often over several steps.")}))

    # ============ 2 ============
    g = new(2, T("同一組 8 個點，兩個類別", "The same 8 points, two classes"))
    pill(g, 890, 62, 140, T("講義補充", "Added note"), GRAY, size=16)
    g.beat = 1
    card(g, 50, 120, 640, 480, BLUE); head(g, 70, 126, 600, T("指定的教學點（不是真實量測）", "Prescribed teaching points (not measurements)"), BLUE, 22)
    cx, cy = scatter8(g, 110, 190, 520, 380)
    g.beat = 2
    g.text(cx - 220, cy - 215, 440, 30, T("類別 2：$x_2=0.8,\\ 1.2$", "class 2: $x_2=0.8,\\ 1.2$"), size=18, color=ORG, bold=True, align="center", z=8)
    g.text(cx - 220, cy + 190, 440, 30, T("類別 1：$x_2=-1.2,\\ -0.8$", "class 1: $x_2=-1.2,\\ -0.8$"), size=18, color=BLUE, bold=True, align="center", z=8)
    g.text(cx - 280, cy - 20, 90, 28, "$-3$", size=18, color=MUTED, align="center", z=8); g.text(cx + 190, cy - 20, 90, 28, "$3$", size=18, color=MUTED, align="center", z=8)
    g.beat = 3
    card(g, 730, 120, 500, 480, GRAY); head(g, 750, 126, 460, T("程式算出的量（$1/8$ 平均）", "Computed by program (averaged by $1/8$)"), MUTED, 22)
    for i, (nm, v) in enumerate(((r"$C$", r"$\operatorname{diag}(9,\,1.04)$"), (r"$S_W$", r"$\operatorname{diag}(9,\,0.04)$"), (r"$S_B$", r"$\operatorname{diag}(0,\,1)$"))):
        y = 200 + i * 110
        g.rect(760, y, 440, 90, fill=[BLUEL, ORGL, GRNL][i], stroke=[BLUE, ORG, GRN][i], sw=3, radius=12, z=3); g.text(770, y, 110, 90, nm, size=32, color=INK, bold=True, align="center", vcenter=True, z=8); g.text(880, y, 310, 90, v, size=28, color=INK, align="center", vcenter=True, z=8)
    g.text(750, 540, 460, 50, T("總平均 $(0,0)$；類中心 $(0,-1)$、$(0,1)$", "overall mean $(0,0)$; class centers $(0,-1)$, $(0,1)$"), size=18, color=MUTED, align="center", z=8)
    banner(g, T("變異最大的方向，不一定是分得開類別的方向", "The direction of greatest variance need not separate the classes"))
    pages.append((g, T("8 點控制例", "The 8-point control case"), {
        0: T("這是講義的補充：同一組 8 個指定教學點，用來看變異和分離為什麼可能指向不同方向。它們不是訓練結果，也不是真實量測。", "A supplement: the same 8 prescribed points show why variance and separation can point in different directions. They are neither a training result nor real measurements."),
        1: T("類別 1 的點在 x2 等於負 1.2 與負 0.8；類別 2 在 0.8 與 1.2；x1 都是負 3 或 3。", "Class 1 has x2 equal to minus 1.2 and minus 0.8; class 2 has 0.8 and 1.2; x1 is minus 3 or 3 throughout."),
        2: T("以 1/8 平均：總共變異數 C 是對角矩陣 9 與 1.04；類內散布 S_W 是 9 與 0.04；類間散布 S_B 是 0 與 1。這些由程式重算。", "Averaging by 1/8: total covariance C is diagonal 9 and 1.04; within-class scatter S_W is 9 and 0.04; between-class scatter S_B is 0 and 1. All computed by program."),
        3: T("看 x1 方向變異最大，看 x2 方向才分得開兩類。", "Variance is largest along x1, but only x2 separates the classes.")}))

    # ============ 3 & 4 ============
    def proj_page(n, title_zh, title_en, col, light, axis, notes, formula, which):
        g = new(n, T(title_zh, title_en))
        g.beat = 1
        card(g, 50, 120, 560, 480, col); head(g, 70, 126, 520, T("原來的 8 點", "The original 8 points"), col, 22)
        cx, cy = scatter8(g, 100, 190, 470, 380)
        if which == "pca": g.line([(cx - 230, cy), (cx + 230, cy)], RED, sw=6, z=6, arrow="end", head=22)
        else: g.line([(cx, cy + 190), (cx, cy - 190)], RED, sw=6, z=6, arrow="end", head=22)
        g.text(100, 580 - 6, 470, 30, T(r"$\mathbf w=(1,0)$" if which == "pca" else r"$\mathbf w=(0,1)$", r"$\mathbf w=(1,0)$" if which == "pca" else r"$\mathbf w=(0,1)$"), size=22, color=RED, bold=True, align="center", z=8)
        g.beat = 2
        arrow(g, (620, 360), (690, 360), ORG, sw=6, head=20)
        card(g, 700, 120, 530, 480, col); head(g, 720, 126, 490, T("投影到這條軸之後", "After projecting onto this axis"), col, 22)
        if which == "pca":
            g.line([(740, 420), (1190, 420)], MUTED, sw=4, z=3)
            for x in (-3, 3):
                px = 965 + x * 70
                for j in range(2): g.ellipse(px - 11, 400 - j * 30, 10, fill=BLUE, stroke="#ffffff", sw=2, z=5); g.ellipse(px + 11, 400 - j * 30, 10, fill=ORG, stroke="#ffffff", sw=2, z=5)
                g.rect(px - 26, 305, 52, 30, fill=REDL, stroke=RED, sw=2, radius=8, z=6); g.text(px - 26, 305, 52, 30, "×2", size=18, color=RED, bold=True, align="center", vcenter=True, z=8)
                g.text(px - 30, 428, 60, 28, rf"${x}$", size=18, color=MUTED, align="center", z=8)
            g.text(740, 480, 450, 90, T("兩類都落在 $-3$ 與 $3$：\n各位置各有 2 筆，類別混在一起", "Both classes land on $-3$ and $3$:\n2 samples at each, classes mixed"), size=22, color=INK, bold=True, align="center", z=8)
        else:
            g.line([(740, 420), (1190, 420)], MUTED, sw=4, z=3)
            sx = lambda v: 965 + v * 150
            for v in (-1.2, -0.8): g.ellipse(sx(v), 400, 12, fill=BLUE, stroke="#ffffff", sw=2, z=5); g.text(sx(v) - 30, 428, 60, 28, rf"${v}$", size=16, color=MUTED, align="center", z=8)
            for v in (0.8, 1.2): g.ellipse(sx(v), 400, 12, fill=ORG, stroke="#ffffff", sw=2, z=5); g.text(sx(v) - 30, 428, 60, 28, rf"${v}$", size=16, color=MUTED, align="center", z=8)
            g.line([(965, 330), (965, 430)], GRAY, sw=3, dash=True, z=4)
            g.text(740, 480, 450, 90, T("類別 1 在左、類別 2 在右：\n兩類分得開", "Class 1 on the left, class 2 on the right:\nthe classes separate"), size=22, color=INK, bold=True, align="center", z=8)
        g.beat = 3
        f1, f2 = formula
        g.rect(740, 170, 450, 130, fill=light, stroke=col, sw=3, radius=12, z=3); g.text(750, 176, 430, 66, f1, size=26, color=INK, align="center", vcenter=True, z=8); g.text(750, 244, 430, 50, f2, size=26, color=INK, align="center", vcenter=True, z=8)
        banner(g, T(axis[0], axis[1]))
        pages.append((g, T(title_zh.split("：")[0], title_en.split(":")[0]), notes))
        return g
    proj_page(3, "只有輸入點：PCA 選變異最大的方向", "Inputs only: PCA picks the direction of greatest variance", BLUE, BLUEL,
              (r"選 $x_1$：變異 9 比 1.04 大，但兩類混在一起", r"It keeps $x_1$: variance 9 beats 1.04, yet the classes mix"), {
        0: T("沒有標籤時，主成分分析 PCA 保留最大變異的方向。", "Without labels, PCA keeps the direction of greatest variance."),
        1: T("投影到 x1 軸：兩類的點都落在負 3 和 3，每個位置各有兩筆。這裡的上下偏移只是為了看見兩類，不是第二個維度。", "Projecting onto x1: both classes land on minus 3 and 3, two samples at each position. The vertical offset only shows both classes; it is not a second dimension."),
        2: T("PCA 選 w 等於 (1,0)，因為這個方向的變異是 9，比 x2 方向的 1.04 大。", "PCA picks w equal to (1,0) because its variance is 9, larger than 1.04 along x2."),
        3: T("結果：變異最大，卻沒有把兩類分開。", "The result: greatest variance, yet the classes are not separated.")}, (r"$\max_{\|\mathbf w\|=1}\ \mathbf w^{\top}C\,\mathbf w$", r"$9>1.04$"), "pca")
    proj_page(4, "有標籤：FDA 選分得開類別的方向", "With labels: FDA separates the classes", ORG, ORGL,
              (r"選 $x_2$：$J=25$ 比 $0$ 大，兩類分開", r"It keeps $x_2$: $J=25$ beats $0$, and the classes separate"), {
        0: T("有了類別標籤，Fisher 判別分析 FDA 比較類間與類內散布，優先找能分開類別的方向。", "With class labels, Fisher discriminant analysis compares between-class and within-class scatter and favors directions that separate classes."),
        1: T("投影到 x2 軸：類別 1 落在負 1.2 與負 0.8，類別 2 落在 0.8 與 1.2，兩類分開。", "Projecting onto x2: class 1 lands on minus 1.2 and minus 0.8, class 2 on 0.8 and 1.2; the classes separate."),
        2: T("FDA 選 w 等於 (0,1)：J 是 1 除以 0.04 等於 25；(1,0) 的 J 是 0 除以 9 等於 0。分母必須大於零。", "FDA picks w equal to (0,1): J is 1 over 0.04, which is 25; for (1,0), J is 0 over 9, which is 0. The denominator must be positive."),
        3: T("標籤告訴表示：這個區別值得保留。", "The labels tell the representation that this distinction is worth keeping.")}, (r"$J(\mathbf w)=\frac{\mathbf w^{\top}S_B\mathbf w}{\mathbf w^{\top}S_W\mathbf w}$", r"$J(0,1)=25,\ J(1,0)=0$"), "fda")

    # ============ 5:兩個準則 ============
    g = new(5, T("兩個目標回答不同的問題", "The two objectives answer different questions"))
    g.beat = 1
    for i, (col, light, ttl, f_, sub) in enumerate(((BLUE, BLUEL, "PCA", r"$\max\ \mathbf w^{\top}C\,\mathbf w$", T("看變異：資料往哪個方向散得最開？", "variance: where is the data most spread?")), (ORG, ORGL, "FDA", r"$\max\ \dfrac{\mathbf w^{\top}S_B\mathbf w}{\mathbf w^{\top}S_W\mathbf w}".replace("\\dfrac", "\\frac") + "$", T("看類別：類間分得開、類內擠得近？", "classes: far apart between, tight within?")))):
        g.beat = i + 1
        x = 50 + i * 600
        card(g, x, 130, 580, 380, col); head(g, x + 20, 136, 540, ttl, col, 30)
        g.rect(x + 40, 210, 500, 110, fill=light, stroke=col, sw=3, radius=14, z=3); g.text(x + 50, 210, 480, 110, f_ if i == 0 else r"$\max\ \frac{\mathbf w^{\top}S_B\mathbf w}{\mathbf w^{\top}S_W\mathbf w}$", size=34, color=INK, align="center", vcenter=True, z=8)
        g.text(x + 40, 345, 500, 90, sub, size=24, color=INK, bold=True, align="center", z=8)
        g.text(x + 40, 440, 500, 50, T("在 8 點例中：選 $(1,0)$", "In the 8-point case: picks $(1,0)$") if i == 0 else T("在 8 點例中：選 $(0,1)$", "In the 8-point case: picks $(0,1)$"), size=22, color=col, bold=True, align="center", z=8)
    g.beat = 3
    g.text(50, 535, 1180, 90, T("兩式都不是所有降維方法共用的目標：LLE 看局部重建；\n標籤標錯時，投影準則也可能朝錯方向調整", "Neither is a universal objective: LLE preserves local reconstruction;\nwrong labels can steer the criterion the wrong way"), size=22, color=MUTED, bold=True, align="center", z=8)
    banner(g, T("標籤告訴表示：哪些區別值得保留", "Labels tell the representation which distinctions to keep"))
    pages.append((g, T("兩個準則", "Two criteria"), {
        0: T("PCA 最大化投影後的變異；FDA 最大化類間散布與類內散布的比值。兩者回答不同的問題。", "PCA maximizes projected variance; FDA maximizes the ratio of between-class to within-class scatter. They answer different questions."),
        1: T("PCA 看變異，FDA 看類別。", "PCA looks at variance, FDA at classes."),
        2: T("常見迷思：無監督降維並不一律是推近相似、拉遠相異，PCA 看變異，LLE 看局部重建。標籤也不保證更好的表示，因為標錯時，投影準則可能朝錯方向調整。", "Common misconception: unsupervised reduction does not always pull similar points together and push others apart; PCA looks at variance, LLE at local reconstruction. Labels do not guarantee a better representation: wrong labels can steer the criterion the wrong way."),
        3: T("兩式都不是所有降維方法共用的目標。", "Neither formula is a universal objective for all reduction methods.")}))

    # ============ 6:代理與環境 ============
    g = new(6, T("強化學習：行動會改變接下來看到的東西", "Reinforcement learning: actions change what comes next"))
    g.beat = 1
    card(g, 60, 220, 360, 360, BLUE); head(g, 80, 226, 320, T("代理", "Agent"), BLUE, 26)
    mascot(g, 180, 300, 0.7, "proud")
    g.beat = 2
    card(g, 860, 220, 360, 360, GRN); head(g, 880, 226, 320, T("環境", "Environment"), GRN, 26)
    g.rect(930, 400, 220, 120, fill="#fef3c7", stroke="#b45309", sw=4, radius=6, z=3); g.poly([(920, 400), (1040, 320), (1160, 400)], fill=RED, stroke="#7f1d1d", sw=4, z=4); g.rect(1020, 450, 40, 70, fill="#7c2d12", z=4)
    g.line([(430, 330), (850, 330)], BLUE, sw=7, arrow="end", head=24, z=6)
    g.text(450, 270, 380, 50, T(r"行動 $a_t\in\mathcal A$", r"action $a_t\in\mathcal A$"), size=26, color=BLUE, bold=True, align="center", vcenter=True, z=8)
    g.beat = 3
    g.line([(850, 470), (430, 470)], GRN, sw=7, arrow="end", head=24, z=6)
    g.text(450, 490, 380, 50, T(r"新狀態 $s_{t+1}$ 與獎勵 $r_{t+1}$", r"new state $s_{t+1}$ and reward $r_{t+1}$"), size=24, color=GRN, bold=True, align="center", vcenter=True, z=8)
    g.rect(540, 400, 60, 50, fill=GRNL, stroke=GRN, sw=3, radius=10, z=5); g.text(540, 400, 60, 50, "+", size=36, color=GRN, bold=True, align="center", vcenter=True, z=8)
    g.rect(620, 400, 60, 50, fill=REDL, stroke=RED, sw=3, radius=10, z=5); g.text(620, 400, 60, 50, "−", size=36, color=RED, bold=True, align="center", vcenter=True, z=8)
    g.text(150, 110, 980, 90, T("代理觀察狀態 $s_t$，依策略選行動；環境回傳新狀態與獎勵", "The agent observes $s_t$ and picks an action by its policy; the environment returns a new state and a reward"), size=24, color=INK, bold=True, align="center", z=8)
    banner(g, T("獎勵可正可負：評估行動的後果，不是對照標準答案", "Rewards can be positive or negative: they judge consequences, not match a key"))
    pages.append((g, T("代理與環境", "Agent and environment"), {
        0: T("表示學習觀察既有資料；強化學習的代理會採取行動。前者不是後者必需的前置步驟。", "Representation learning analyzes existing data; in reinforcement learning the agent takes actions. The former is not a required step for the latter."),
        1: T("環境：行動會改變它的狀態。", "The environment: actions change its state."),
        2: T("代理依策略選一個行動，送到環境。", "The agent picks an action by its policy and sends it to the environment."),
        3: T("環境回傳新狀態與獎勵。綠色加號與紅色減號表示獎勵可以有不同符號，不表示每一步都有兩項。", "The environment returns a new state and a reward. The green plus and red minus show rewards may have either sign, not that every step has two terms.")}))

    # ============ 7:兩條路徑 ============
    g = new(7, T("玩具收拾：兩條路徑，每一步都有獎勵", "Tidying toys: two paths, a reward at every step"))
    pill(g, 1030, 96, 200, T("設計者給定的數字", "set by the designer"), GRAY, size=16)
    paths = [("A", T("先收玩具", "tidy the toy"), T("再坐下", "then sit"), 1, 3, BLUE, BLUEL), ("B", T("先咬玩具", "bite the toy"), T("再亂跑", "then run around"), 2, -5, ORG, ORGL)]
    for i, (nm, s1, s2, a, b_, col, light) in enumerate(paths):
        g.beat = i + 1
        y = 140 + i * 230
        card(g, 50, y, 1180, 200, col)
        g.ellipse(120, y + 100, 40, fill=col, z=3); g.text(80, y + 60, 80, 80, nm, size=36, color="#ffffff", bold=True, align="center", vcenter=True, z=8)
        g.rect(200, y + 50, 280, 100, fill=light, stroke=col, sw=3, radius=14, z=3); g.text(200, y + 50, 280, 100, s1, size=26, color=INK, bold=True, align="center", vcenter=True, z=8)
        g.line([(490, y + 100), (610, y + 100)], GRAY, sw=6, arrow="end", head=20, z=5)
        g.rect(620, y + 50, 280, 100, fill=light, stroke=col, sw=3, radius=14, z=3); g.text(620, y + 50, 280, 100, s2, size=26, color=INK, bold=True, align="center", vcenter=True, z=8)
        for (bx, v) in ((430, a), (850, b_)):
            g.ellipse(bx + 40, y + 50, 40, fill=GRNL if v > 0 else REDL, stroke=GRN if v > 0 else RED, sw=4, z=6); g.text(bx, y + 20, 80, 60, (f"+{v}" if v > 0 else f"−{abs(v)}"), size=30, color=GRN if v > 0 else RED, bold=True, align="center", vcenter=True, z=8)
        g.text(955, y + 56, 250, 90, T("圓圈：每一步的獎勵", "circles: the reward at each step"), size=18, color=MUTED, z=8)
    g.beat = 3
    banner(g, T("哪一條比較好？先別只看第一步", "Which is better? Don't look at the first step alone"))
    pages.append((g, T("兩條路徑", "Two paths"), {
        0: T("玩具收拾例：稱讚收好可設正獎勵，咬壞玩具可設負獎勵。數字是設計者給定的，不是孩子或機器人的實測。", "The toy-tidying example: praise for tidying can be a positive reward, biting a toy a negative one. The numbers are set by the designer, not measured from a child or a robot."),
        1: T("路徑 A：先收玩具，獎勵正 1；再坐下，獎勵正 3。", "Path A: tidy the toy for plus 1, then sit for plus 3."),
        2: T("路徑 B：先咬玩具，獎勵正 2；再亂跑，獎勵負 5。", "Path B: bite the toy for plus 2, then run around for minus 5."),
        3: T("哪一條比較好？先別只看第一步。", "Which is better? Do not judge by the first step alone.")}))

    # ============ 8:回報 ============
    g = new(8, T("要比較整段回報：即時排名會翻轉", "Compare the whole return: the immediate ranking flips"))
    g.beat = 1
    card(g, 50, 120, 560, 420, GRAY); head(g, 70, 126, 520, T("只看第一步 $r_1$", "First step only, $r_1$"), MUTED, 22)
    U = 50; base = 440
    g.line([(100, base), (560, base)], MUTED, sw=3, z=3)
    for nm, v, x, col in (("A", 1, 160, BLUE), ("B", 2, 380, ORG)):
        g.rect(x, base - v * U, 110, v * U, fill=col, stroke="#ffffff", sw=2, radius=4, z=4); g.text(x, base - v * U - 40, 110, 36, rf"$+{v}$", size=26, color=col, bold=True, align="center", vcenter=True, z=8); g.text(x, base + 6, 110, 34, rf"${nm}$", size=26, color=INK, bold=True, align="center", vcenter=True, z=8)
    star(g, 435, base - 2 * U - 76, 20, GOLD, z=6, stroke="#b45309", sw=2)
    g.text(70, 480, 520, 50, T(r"$2>1$：B 的即時獎勵較高", r"$2>1$: B pays more right now"), size=22, color=INK, bold=True, align="center", z=8)
    g.beat = 2
    card(g, 650, 120, 580, 420, BLUE); head(g, 670, 126, 540, T(r"整段回報 $G=r_1+r_2$", r"Whole return $G=r_1+r_2$"), BLUE, 22)
    zero = 340; U2 = 28
    g.line([(700, zero), (1180, zero)], MUTED, sw=3, z=3)
    for nm, (a, b_), x in (("A", (1, 3), 760), ("B", (2, -5), 1000)):
        g.rect(x, zero - a * U2, 110, a * U2, fill=BLUEL, stroke=BLUE, sw=2, z=4)
        if b_ > 0: g.rect(x, zero - (a + b_) * U2, 110, b_ * U2, fill=ORGL, stroke=ORG, sw=2, z=4)
        else: g.rect(x, zero, 110, abs(b_) * U2, fill=REDL, stroke=RED, sw=2, z=4)
        tot = a + b_
        g.text((x + 118) if nm == "A" else (x - 128), (zero - (a + b_) * U2 + 10) if tot > 0 else zero + 40, 120, 40, rf"$G_{nm}={tot}$", size=24, color=INK, bold=True, vcenter=True, z=8)
    g.text(700, 490, 500, 40, r"$G_A=1+3=4,\quad G_B=2+(-5)=-3$", size=22, color=INK, align="center", vcenter=True, z=8)
    star(g, 815, zero - 4 * U2 - 30, 18, GOLD, z=6, stroke="#b45309", sw=2)
    g.beat = 3
    g.text(50, 560, 1180, 60, T("在這兩條固定路徑中，A 的回報較高；這不證明代理已學會 A，也沒有給路徑的機率", "In these two fixed paths A has the higher return; this does not show the agent has learned A, nor give path probabilities"), size=20, color=MUTED, bold=True, align="center", z=8)
    banner(g, T("第一步 $2>1$，但兩步後 $4>-3$", "First step $2>1$, yet after two steps $4>-3$"))
    pages.append((g, T("整段回報", "The whole return"), {
        0: T("只看第一步：B 的即時獎勵 2 比 A 的 1 高。", "Looking at the first step only: B's immediate reward 2 beats A's 1."),
        1: T("改看兩步的總和，G 等於 r1 加 r2。A 是 1 加 3 等於 4；B 是 2 加負 5 等於負 3。這裡不打折扣，相當於折扣因子等於 1；多步時一般用折扣因子 gamma。強化學習關心的是策略下的預期累積回報。", "Now the two-step sum, G equals r1 plus r2. A is 1 plus 3, which is 4; B is 2 plus minus 5, which is minus 3. There is no discounting here, equivalent to a discount factor of 1; with more steps a discount gamma is used. RL cares about the expected return under a policy."),
        2: T("所以即時排名是 B 較高，整段回報卻是 A 較高。圖程式直接用 1、3、2、負 5 計算，保留負號。", "So the immediate ranking favors B, while the whole return favors A. The figure is computed directly from 1, 3, 2 and minus 5, keeping the sign."),
        3: T("這個算式沒有證明代理已學會 A，也沒有給每條路徑的機率。", "This calculation does not show the agent has learned A, and gives no path probabilities.")}))

    # ============ 9 ============
    g = new(9, T("回饋設計錯了，會把代理帶到錯的地方", "A badly designed feedback leads the agent astray"))
    g.beat = 1
    card(g, 50, 130, 570, 400, ORG); head(g, 70, 136, 530, T("錯誤的標籤", "Wrong labels"), ORG, 24)
    for k, (px, py, c) in enumerate([(130, 260, BLUE), (200, 320, BLUE), (150, 390, ORG), (330, 270, ORG), (420, 340, ORG), (360, 430, BLUE)]): g.ellipse(px + 60, py, 22, fill=c, stroke="#ffffff", sw=3, z=5)
    cross_ = lambda x, y, r: (g.line([(x - r, y - r), (x + r, y + r)], RED, sw=5, z=7), g.line([(x - r, y + r), (x + r, y - r)], RED, sw=5, z=7))
    cross_(420 + 60 - 60 + 0, 430, 10); g.text(70, 460, 530, 60, T("標錯時，投影準則可能朝錯方向調整", "wrong labels can steer the projection the wrong way"), size=20, color=INK, bold=True, align="center", z=8)
    g.beat = 2
    card(g, 660, 130, 570, 400, GRN); head(g, 680, 136, 530, T("不當的獎勵", "A poor reward"), GRN, 24)
    for k in range(3): g.ellipse(790 + k * 90, 460 - 40 + 0, 0.1, fill=GRN, z=1) if False else None
    g.poly([(1170, 250), (1200, 300), (1140, 300)], fill=RED, z=5); g.text(1140, 262, 60, 40, "!", size=24, color="#ffffff", bold=True, align="center", vcenter=True, z=8)
    g.rect(740, 340, 420, 80, fill="#fef3c7", stroke="#b45309", sw=4, radius=6, z=3); g.text(740, 340, 420, 80, T("床底", "under the bed"), size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.line([(740, 300), (800, 380)], GRN, sw=6, z=5, arrow="end", head=18); g.ellipse(790, 280, 26, fill=GOLD, stroke="#b45309", sw=4, z=6); g.text(764, 262, 52, 36, "+1", size=20, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.text(680, 440, 530, 60, T("只獎勵「快收拾」，代理可能把玩具掃到床底", "rewarding only “quick tidying” may make it sweep toys under the bed"), size=20, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    g.text(50, 550, 1180, 70, T("口訣：先問訊號從哪來——資料幾何、側資訊，還是行動後的獎勵？", "Rule of thumb: ask where the feedback comes from: data geometry, side information, or rewards after actions"), size=22, color=INK, bold=True, align="center", z=8)
    banner(g, T(r"下一站 KP06：監督式分類工具各有什麼假設？$\to$", r"Next, KP06: what do the supervised classifiers assume? $\to$"))
    pages.append((g, T("回饋也會出錯", "Feedback can go wrong"), {
        0: T("錯誤的類別標籤和錯誤的獎勵設計，都提供回饋，但作用在不同的問題上。", "Wrong class labels and a wrongly designed reward both provide feedback, but they act on different problems."),
        1: T("標籤標錯時，投影準則可能朝錯方向調整。", "With wrong labels, the projection criterion can be steered the wrong way."),
        2: T("獎勵設計不當時，例如只獎勵快收拾，代理可能把玩具掃到床底。獎勵函數應該反映真正的目標。", "With a poorly designed reward, such as only rewarding quick tidying, the agent may sweep toys under the bed. A reward should reflect the real goal."),
        3: T("口訣：先問訊號從哪來？資料幾何、側資訊，還是行動後的獎勵。講義的 4 題自我檢核留在講義。", "Rule of thumb: ask where the signal comes from: data geometry, side information, or rewards after actions. The note's 4 self-check questions stay in the note.")}))
    return pages
