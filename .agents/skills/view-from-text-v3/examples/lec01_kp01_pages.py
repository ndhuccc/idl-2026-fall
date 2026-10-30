import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *
from lec01_kp01_nums import *

def make(lang):
    zh = lang == "zh"
    T = lambda a, b: a if zh else b
    pages = []

    def new(n, title):
        g = PG(); P = g
        P.rect(0, 0, 1280, 720, fill=BG, z=0)
        P.text(50, 22, 1010, 72, title, size=36, color=INK, bold=True, vcenter=True, z=8)
        P.text(1060, 36, 170, 30, f"KP01 · {n}/9", size=18, color=MUTED, align="right", z=8)
        return g

    def banner(g, s):
        g.rect(50, 640, 1180, 60, fill=BAN, radius=14, z=3, shadowColor="rgba(30,58,138,0.35)", shadowBlur=12, shadowOffsetY=4)
        g.text(70, 640, 1140, 60, s, size=24, color="#ffffff", bold=True, align="center", vcenter=True, z=8)

    def pill(g, x, y, w, s, fill, color="#ffffff", size=18):
        g.rect(x, y, w, 30, fill=fill, radius=15, z=6); g.text(x, y, w, 30, s, size=size, color=color, bold=True, align="center", vcenter=True, z=8)

    # ============ 頁 1:地圖 ============
    g = new(1, T("KP01 選模型要有評分標準", "KP01 Model selection needs a scoring rule"))
    roads = [[(0, 120), (150, 110), (260, 170), (380, 160), (500, 200)], [(120, 0), (140, 120), (200, 260), (190, 415)],
             [(330, 0), (340, 160), (420, 300), (500, 330)], [(0, 300), (130, 290), (260, 330), (360, 415)]]
    samp = [(a + (c - a) * t / 20, b + (d - b) * t / 20) for r in roads for (a, b), (c, d) in zip(r, r[1:]) for t in range(21)]
    near = lambda x, y, m: any(math.hypot(x - sx, y - sy) < m for sx, sy in samp)
    def terrain(ox, oy, detail):
        g.rect(ox, oy, 500, 415, fill="#ecfccb", stroke="#d9f99d", sw=2, radius=12, z=2)
        if detail: g.poly([(ox, oy + 330), (ox + 80, oy + 350, True), (ox + 200, oy + 385, True), (ox + 300, oy + 405, True), (ox + 300, oy + 415), (ox, oy + 415)], fill="#bae6fd", z=2)
        for r in roads: g.line([(ox + x, oy + y) for x, y in r], "#cbd5e1" if detail else "#64748b", sw=11 if not detail else 9, smooth=True, z=3)
        if detail:
            n = 0
            for i in range(9):
                for j in range(8):
                    x = 18 + i * 52 + (j % 3) * 7; y = 14 + j * 48 + (i % 2) * 9
                    if x < 475 and y < 395 and not near(x + 12, y + 10, 30) and not (y > 300 and x < 300):
                        g.rect(ox + x, oy + y, 24 + (i * j) % 3 * 5, 20 + (i + j) % 3 * 4, fill=["#fecaca", "#fde68a", "#e2e8f0"][(i + j) % 3], stroke="#94a3b8", sw=1, z=4); n += 1
            for i in range(22):
                x = 14 + (i * 83) % 470; y = 10 + (i * 57) % 380
                if not near(x, y, 22) and not (y > 320 and x < 310): g.ellipse(ox + x, oy + y, 9, fill="#4ade80", stroke="#15803d", sw=1.5, z=4)
    g.beat = 1
    g.rect(50, 120, 540, 500, fill=CARD, stroke=GRAY, sw=2, radius=16, z=1, shadowColor="rgba(15,23,42,0.12)", shadowBlur=14, shadowOffsetY=4)
    g.text(70, 128, 380, 34, T("真實世界：細節很多", "The real world: full of detail"), size=20, color=MUTED, bold=True, vcenter=True, z=8)
    pill(g, 460, 130, 110, T("比喻 · 示意", "Analogy"), GRAY, size=16)
    terrain(70, 190, True)
    g.beat = 2
    g.line([(600, 400), (680, 400)], ORG, sw=5, arrow="end", head=20, z=6)
    g.text(596, 340, 90, 56, T("省略\n細節", "Drop\ndetail"), size=18, color=ORG, bold=True, align="center", z=8)
    g.rect(690, 120, 540, 500, fill=CARD, stroke=BLUE, sw=3, radius=16, z=1, shadowColor="rgba(29,78,216,0.18)", shadowBlur=14, shadowOffsetY=4)
    g.text(710, 128, 500, 34, T("地圖（模型）：只留要用的", "The map (model): keep only what is used"), size=20, color=BLUE, bold=True, vcenter=True, z=8)
    terrain(710, 190, False)
    route = [(750, 307), (860, 300), (970, 360), (1090, 350), (1165, 380)]
    g.line(route, GRN, sw=7, dash=True, smooth=True, z=5, arrow="end", head=22)
    g.ellipse(750, 307, 17, fill=RED, stroke="#ffffff", sw=3, z=6); g.text(733, 291, 34, 32, "A", size=18, color="#ffffff", bold=True, align="center", vcenter=True, z=8)
    star(g, 1165, 380, 24, GOLD, z=6, stroke="#b45309", sw=2); g.text(1135, 412, 60, 28, "B", size=18, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    banner(g, T("省略不是錯：關鍵是留下任務需要的規律", "Leaving things out is fine: keep the patterns the task needs"))
    pages.append((g, T("選模型要有評分標準", "Model selection needs a scoring rule"), {
        0: T("先看一張地圖。地圖把道路以外的細節省掉，反而好用。（比喻，不是講義的實驗。）", "Start with a map. It drops detail beyond the roads, which makes it easier to use. (An analogy, not an experiment from the note.)"),
        1: T("真實世界什麼都有：建築、樹、河。", "The real world has everything: buildings, trees, a river."),
        2: T("地圖只留道路和地標，足夠從 A 走到 B。模型也一樣，有取捨。", "The map keeps roads and landmarks, enough to get from A to B. A model also makes trade-offs."),
        3: T("省略本身不是錯，關鍵是留下任務需要的規律。地圖的比喻只說明表示有取捨，不代表模型一定掌握真實世界的因果。", "Omission is not a flaw; keep the patterns the task needs. The analogy only shows that representation involves trade-offs, not that a model knows real-world causes.")}))

    # ============ 頁 2:擬合與泛化 ============
    g = new(2, T("好模型要過兩關：擬合與泛化", "A good model passes two tests: fitting and generalizing"))
    def sheet(x, color, light, title, tag, sub, symbol):
        g.rect(x, 120, 570, 500, fill=CARD, stroke=color, sw=3, radius=16, z=1, shadowColor="rgba(15,23,42,0.12)", shadowBlur=14, shadowOffsetY=4)
        g.rect(x, 120, 570, 56, fill=light, stroke=color, sw=0, radius=16, z=2)
        g.text(x + 20, 120, 390, 56, title, size=24 if zh else 20, color=color, bold=True, vcenter=True, z=8)
        pill(g, x + 420, 133, 130, T("比喻 · 示意", "Analogy"), GRAY, size=16)
        for r in range(4):
            y = 200 + r * 56
            g.rect(x + 30, y + 12, 250 - r * 25, 14, fill="#e2e8f0", radius=7, z=3)
            symbol(x + 380, y)
        pill(g, x + 30, 436, 160, tag, color, size=20)
        g.text(x + 30, 480, 380, 80, sub, size=22, color=INK, bold=True, z=8)
    g.beat = 1
    sheet(50, BLUE, BLUEL, T("練習題（已經看過）", "Practice problems (already seen)"), T("擬合", "Fitting"), T("調到貼合已看過的資料", "match data already seen"), lambda x, y: tick(g, x, y + 4, 34))
    mascot(g, 440, 450, 0.5, "proud")
    g.beat = 2
    sheet(660, ORG, ORGL, T("新題目（沒看過）", "New problems (never seen)"), T("泛化", "Generalizing"), T("沒看過的資料上仍有效", "still works on unseen data"), lambda x, y: g.text(x + 4, y - 2, 50, 50, "?", size=40, color=RED, bold=True, align="center", vcenter=True, z=8))
    mascot(g, 1050, 450, 0.5, "confused")
    g.beat = 3
    banner(g, T("練習答得準，不保證新題也答得準：兩種不同的證據", "Aced practice does not guarantee new problems: two different kinds of evidence"))
    pages.append((g, T("擬合與泛化", "Fitting and generalizing"), {
        0: T("模型要過兩關。", "A model has to pass two tests."),
        1: T("第一關是擬合：調整模型，讓它貼合已經看過的訓練資料。就像把練習題背熟。", "Test one is fitting: adjust the model to match the training data it has seen, like memorizing practice problems."),
        2: T("第二關是泛化：碰到沒看過的新資料，仍然有效。背熟練習題的人，不一定會解新題。", "Test two is generalizing: stay effective on data it has not seen. Memorizing practice problems does not guarantee solving new ones."),
        3: T("擬合與泛化是兩種不同的證據，不能互相取代。（比喻，不是講義的實驗。）", "Fitting and generalizing are two different kinds of evidence. (An analogy, not an experiment from the note.)")}))

    # ============ 頁 3:三個角色 ============
    g = new(3, T("比較模型前，先定三件事", "Before comparing models, fix three things"))
    xs3 = [50, 470, 890]
    def box(i, color, light, num, name, math_, sentence, icon):
        x = xs3[i]
        g.rect(x, 130, 340, 470, fill=CARD, stroke=color, sw=3, radius=18, z=1, shadowColor="rgba(15,23,42,0.12)", shadowBlur=14, shadowOffsetY=4)
        g.rect(x, 130, 340, 70, fill=light, radius=18, z=2); g.rect(x, 170, 340, 30, fill=light, z=2)
        g.ellipse(x + 40, 165, 22, fill=color, z=3); g.text(x + 18, 148, 44, 34, num, size=22, color="#ffffff", bold=True, align="center", vcenter=True, z=8)
        g.text(x + 76, 135, 250, 60, name, size=26, color=color, bold=True, vcenter=True, z=8)
        g.text(x + 20, 210, 300, 56, math_, size=30, color=INK, align="center", vcenter=True, z=8)
        icon(x, 300)
        g.text(x + 20, 490, 300, 90, sentence, size=22, color=INK, bold=True, align="center", z=8)
    def icon_models(x, y):
        g.rect(x + 20, y + 10, 90, 90, fill="#f1f5f9", stroke=BLUE, sw=2, radius=10, z=3); g.line([(x + 30, y + 90), (x + 100, y + 25)], BLUE, sw=4, z=4)
        g.rect(x + 125, y + 10, 90, 90, fill="#f1f5f9", stroke=BLUE, sw=2, radius=10, z=3)
        g.line([(x + 170, y + 25), (x + 148, y + 60), (x + 192, y + 60)], BLUE, sw=3, z=4); g.line([(x + 148, y + 60), (x + 138, y + 88)], BLUE, sw=3, z=4); g.line([(x + 148, y + 60), (x + 158, y + 88)], BLUE, sw=3, z=4)
        g.line([(x + 192, y + 60), (x + 182, y + 88)], BLUE, sw=3, z=4); g.line([(x + 192, y + 60), (x + 202, y + 88)], BLUE, sw=3, z=4)
        for cx, cy in [(170, 25), (148, 60), (192, 60)]: g.ellipse(x + cx, y + cy, 7, fill=BLUE, z=5)
        g.rect(x + 230, y + 10, 90, 90, fill="#f1f5f9", stroke=BLUE, sw=2, radius=10, z=3)
        pts = [(244, 30), (244, 55), (244, 80), (276, 40), (276, 70), (308, 55)]
        for a, b in [(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5)]: g.line([(x + pts[a][0], y + pts[a][1]), (x + pts[b][0], y + pts[b][1])], "#93c5fd", sw=2, z=4)
        for cx, cy in pts: g.ellipse(x + cx, y + cy, 6, fill=BLUE, z=5)
        g.text(x + 20, y + 110, 300, 26, T("直線、決策樹、神經網路", "line, tree, neural network"), size=16, color=MUTED, align="center", z=8)
    def icon_meter(x, y):
        g.rect(x + 40, y + 50, 260, 34, fill=GRN, gradientTo=RED, gradientAngle=90, radius=17, z=3)
        g.poly([(x + 70, y + 34), (x + 96, y + 34), (x + 83, y + 56)], fill=INK, z=5)
        g.text(x + 40, y + 94, 120, 28, T("低 = 好", "low = good"), size=16, color=GRN, bold=True, z=8); g.text(x + 180, y + 94, 120, 28, T("高 = 差", "high = bad"), size=16, color=RED, bold=True, align="right", z=8)
    def icon_search(x, y):
        for k, (cx, cy) in enumerate([(70, 40), (130, 90), (200, 50), (250, 100), (110, 30)]): g.ellipse(x + cx, y + cy, 9, fill=BLUEL, stroke=BLUE, sw=2, z=3)
        g.ellipse(x + 200, y + 50, 38, stroke=GRN, sw=6, z=4, fill="rgba(21,128,61,0.10)"); g.line([(x + 228, y + 78), (x + 270, y + 124)], GRN, sw=9, z=4)
        star(g, x + 200, y + 50, 12, GOLD, z=5)
    g.beat = 1; box(0, BLUE, BLUEL, "1", T("候選模型", "Candidates"), r"$\mathcal F$", T("誰可以被選？\n（假設集合）", "Who may be chosen?\n(hypothesis class)"), icon_models)
    g.beat = 2; g.line([(395, 365), (465, 365)], GRAY, sw=5, arrow="end", head=18, z=6); box(1, ORG, ORGL, "2", T("評分", "Scoring"), r"$S(f)$", T("怎麼比高低？\n（越低越好）", "How do we compare?\n(lower is better)"), icon_meter)
    g.beat = 3; g.line([(815, 365), (885, 365)], GRAY, sw=5, arrow="end", head=18, z=6); box(2, GRN, GRNL, "3", T("搜尋", "Search"), T("找低分者", "find low scores"), T("怎麼找到低分者？\n（不決定重視什麼）", "How do we find them?\n(does not decide what matters)"), icon_search)
    banner(g, T("評分規則先講清楚，才談誰比較好", "State the scoring rule first, then ask who is better"))
    pages.append((g, T("三個角色", "Three roles"), {
        0: T("比較模型之前，先講清楚三件事。", "Before comparing models, fix three things."),
        1: T("第一，候選：假設集合 F，是允許被選的模型。線性模型、決策樹、神經網路是不同的集合。選一個集合，不代表資料真的屬於它。", "First, the candidates: the hypothesis class F, the models you allow. Linear models, trees and networks are different classes. Choosing a class does not mean the data really belongs to it."),
        2: T("第二，評分：給每個候選一個可比較的數字，本單元統一用越低越好。", "Second, scoring: give each candidate a comparable number; this unit uses lower is better."),
        3: T("第三，搜尋：實際去找低分的候選。搜尋不會替你決定重視哪些錯誤。評分規則要先講清楚。（比喻圖示，不是講義的實驗。）", "Third, search: actually look for low-scoring candidates. Search does not decide which errors matter, so state the scoring rule first. (Illustrative icons, not an experiment from the note.)")}))

    # ============ 頁 4:一行公式 ============
    g = new(4, T("把三件事寫成一行", "Write the three things as one line"))
    g.beat = 1
    g.rect(90, 130, 1100, 150, fill=CARD, stroke=GRAY, sw=2, radius=18, z=1, shadowColor="rgba(15,23,42,0.12)", shadowBlur=14, shadowOffsetY=4)
    g.text(110, 140, 1060, 130, r"$\textcolor{#6d28d9}{\widehat f}\in\textcolor{#15803d}{\arg\min}_{\,f\in\textcolor{#1d4ed8}{\mathcal F}}\ \textcolor{#c2410c}{S(f)}$", size=60, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    cards = [(PUR, PURL, r"$\widehat f$", T("選出的模型", "the selected model"), T("最後挑中的那一個", "the one we end up with")),
             (GRN, GRNL, r"$\arg\min_{f\in\mathcal F}$", T("在候選裡找最小", "find the minimum among candidates"), T("先限定範圍，再找最低", "restrict first, then find the lowest")),
             (ORG, ORGL, r"$S(f)$", T("分數，越低越好", "the score, lower is better"), T("每個候選一個數字", "one number per candidate"))]
    for i, (c, l, m, a, b) in enumerate(cards):
        x = 60 + i * 400
        g.rect(x, 330, 360, 270, fill=CARD, stroke=c, sw=3, radius=16, z=1, shadowColor="rgba(15,23,42,0.12)", shadowBlur=12, shadowOffsetY=4)
        g.rect(x, 330, 360, 90, fill=l, radius=16, z=2); g.rect(x, 390, 360, 30, fill=l, z=2)
        g.text(x + 10, 332, 340, 86, m, size=36, color=c, align="center", vcenter=True, z=8)
        g.text(x + 20, 440, 320, 62, a, size=24, color=INK, bold=True, align="center", z=8)
        g.text(x + 20, 510, 320, 70, b, size=20, color=MUTED, align="center", z=8)
    g.beat = 3
    banner(g, T("先限定比較範圍，再從中選出分數最低的模型", "Restrict the candidates, then pick a lowest-scoring model"))
    pages.append((g, T("一行公式", "One line"), {
        0: T("三件事可以寫成一行。", "The three things fit on one line."),
        1: T("f 帽，屬於，在 F 裡讓分數 S 最小的候選。", "f-hat belongs to the set of candidates in F that attain the smallest score S."),
        2: T("三個部分：f 帽是選出的模型；argmin 是在候選裡找最小；S 是分數，越低越好。用屬於符號，是因為可能有並列最低的情況。", "Three parts: f-hat is the selected model; argmin finds the minimum among candidates; S is the score, lower is better. Membership allows ties."),
        3: T("白話讀法：先限定範圍，再從中選出分數最低的模型。", "Read it as: restrict the candidates, then choose a lowest-scoring model.")}))

    # ============ 頁 5:分數 = 誤差 + 懲罰 ============
    g = new(5, T("分數 = 訓練誤差 + 複雜度的代價", "Score = training error + the cost of complexity"))
    g.beat = 1
    g.rect(130, 120, 1020, 110, fill=CARD, stroke=GRAY, sw=2, radius=18, z=1, shadowColor="rgba(15,23,42,0.12)", shadowBlur=14, shadowOffsetY=4)
    g.text(150, 126, 980, 98, r"$S(f)=\textcolor{#1d4ed8}{E_{\mathrm{train}}(f)}+\textcolor{#c2410c}{\lambda\,k(f)}$", size=50, color=INK, align="center", vcenter=True, z=8)
    g.text(150, 238, 980, 36, T(r"$\lambda=0.1$：每多 1 個參數，分數加 $0.1$", r"$\lambda=0.1$: each extra parameter adds $0.1$ to the score"), size=24, color=ORG, bold=True, align="center", vcenter=True, z=8)
    g.beat = 2
    g.rect(180, 330, 560, 64, fill=BLUEL, stroke=BLUE, sw=3, radius=8, z=2); g.rect(740, 330, 360, 64, fill=ORGL, stroke=ORG, sw=3, radius=8, z=2)
    g.text(180, 330, 560, 64, T(r"訓練誤差 $E_{\mathrm{train}}$：錯得多不多", r"Training error $E_{\mathrm{train}}$: how wrong"), size=22, color=BLUE, bold=True, align="center", vcenter=True, z=8)
    g.text(740, 330, 360, 64, T(r"複雜度懲罰 $\lambda k$", r"Complexity penalty $\lambda k$"), size=22, color=ORG, bold=True, align="center", vcenter=True, z=8)
    g.line([(180, 312), (1100, 312)], MUTED, sw=3, z=3); g.line([(180, 312), (180, 322)], MUTED, sw=3, z=3); g.line([(1100, 312), (1100, 322)], MUTED, sw=3, z=3)
    g.text(540, 280, 200, 30, T(r"分數 $S(f)$", r"Score $S(f)$"), size=22, color=INK, bold=True, align="center", z=8)
    pill(g, 1000, 262, 100, T("示意", "Sketch"), GRAY, size=16)
    g.beat = 3
    for i in range(6):
        x = 190 + i * 150
        g.ellipse(x + 40, 490, 34, fill=CARD, stroke=ORG, sw=4, z=3); g.line([(x + 40, 490), (x + 58, 470)], ORG, sw=5, z=4)
        g.ellipse(x + 40, 564, 18, fill=GOLD, stroke="#b45309", sw=2, z=4); g.text(x + 20, 548, 40, 32, "0.1", size=16, color=INK, bold=True, align="center", vcenter=True, z=8)
        g.line([(x + 40, 526), (x + 40, 544)], GRAY, sw=2, dash=True, z=3)
    g.text(1090, 470, 150, 100, T("每個旋鈕（參數）收一枚硬幣", "each knob (parameter) costs one coin"), size=18, color=MUTED, z=8)
    banner(g, T("錯誤有代價，變複雜也有代價：兩者一起算", "Errors cost, and so does complexity: count both"))
    pages.append((g, T("分數怎麼算", "How the score is computed"), {
        0: T("評分函數可以只量錯誤，也可以加上複雜度的懲罰。", "A scoring function can measure error alone or add a complexity penalty."),
        1: T("分數等於訓練誤差加上 lambda 乘以參數個數 k。本例 lambda 是 0.1，也就是每多一個參數，分數加 0.1。參數個數只是這個例子用的複雜度代理，不是通用定義。", "The score is the training error plus lambda times the parameter count k. Here lambda is 0.1: each extra parameter adds 0.1. Parameter count is only this example's proxy for complexity."),
        2: T("把它畫成一條長條：藍色是訓練誤差，橘色是懲罰。（長度是示意，不依比例。）", "Draw it as a bar: blue is training error, orange is the penalty. (Lengths are a sketch, not to scale.)"),
        3: T("可以想成每個旋鈕收一枚硬幣。錯誤有代價，變複雜也有代價，兩者一起算。", "Think of each knob costing a coin. Errors cost, complexity costs; count both.")}))

    # ============ 頁 6:排名翻轉 ============
    g = new(6, T("加上懲罰，排名就翻過來", "Add the penalty and the ranking flips"))
    U = 150; base = 500
    def panel(x, ttl, color):
        g.rect(x, 120, 570, 500, fill=CARD, stroke=color, sw=3, radius=16, z=1, shadowColor="rgba(15,23,42,0.12)", shadowBlur=14, shadowOffsetY=4)
        g.text(x + 20, 126, 530, 44, ttl, size=24, color=color, bold=True, vcenter=True, z=8); g.line([(x + 40, base), (x + 530, base)], MUTED, sw=3, z=4)
    g.beat = 1
    panel(50, T("① 只看訓練誤差", "1. Training error only"), BLUE)
    for name, v, x in (("A", EA, 150), ("B", EB, 380)):
        h = float(v) * U; g.rect(x, base - h, 120, h, fill=BLUEL if name == "B" else BLUE, stroke=BLUE, sw=2, radius=4, z=3)
        g.text(x, base - h - 38, 120, 34, rf"${f(v)}$", size=26, color=BLUE, bold=True, align="center", vcenter=True, z=8); g.text(x, base + 6, 120, 34, rf"${name}$", size=26, color=INK, bold=True, align="center", vcenter=True, z=8)
    star(g, 210, base - float(EA) * U - 74, 26, GOLD, z=6, stroke="#b45309", sw=2)
    g.text(80, 550, 510, 50, T(rf"$0.5<1.0$：A 最低", rf"$0.5<1.0$: A is lowest"), size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.beat = 2
    panel(660, T("② 加上參數懲罰", "2. Add the parameter penalty"), ORG)
    for name, e_, p_, s_, x, k in (("A", EA, penA, SA, 760, kA), ("B", EB, penB, SB, 990, kB)):
        he, hp = float(e_) * U, float(p_) * U
        g.rect(x, base - he, 120, he, fill=BLUEL, stroke=BLUE, sw=2, z=3); g.rect(x, base - he - hp, 120, hp, fill=ORGL, stroke=ORG, sw=2, z=3)
        g.text(x, base - he - hp - 38, 120, 34, rf"${f(s_)}$", size=26, color=INK, bold=True, align="center", vcenter=True, z=8); g.text(x, base + 6, 120, 34, rf"${name}$", size=26, color=INK, bold=True, align="center", vcenter=True, z=8)
        g.text(x + 124, base - he - hp / 2 - 14, 80, 28, rf"${f(p_)}$", size=18, color=ORG, bold=True, z=8) if name == "A" else g.text(x + 124, base - he - hp / 2 - 14, 80, 28, rf"${f(p_)}$", size=18, color=ORG, bold=True, z=8)
    star(g, 1050, base - float(SB) * U - 74, 26, GOLD, z=6, stroke="#b45309", sw=2)
    g.text(676, 548, 540, 32, rf"$S(A)=0.5+0.1\times10=1.5$", size=22, color=INK, align="center", vcenter=True, z=8); g.text(676, 580, 540, 32, rf"$S(B)=1.0+0.1\times2=1.2$", size=22, color=INK, align="center", vcenter=True, z=8)
    g.text(680, 174, 530, 28, T("藍：訓練誤差　橘：參數懲罰", "blue: training error   orange: parameter penalty"), size=16, color=MUTED, z=8)
    g.line([(625, 330), (655, 330)], RED, sw=6, arrow="end", head=18, z=6)
    g.beat = 3
    banner(g, T("同一組候選，評分規則一改，贏家就變了", "Same candidates, new scoring rule, new winner"))
    pages.append((g, T("排名翻轉", "The ranking flips"), {
        0: T("同樣兩個候選 A 和 B，來自講義的例子。A 的訓練誤差 0.5、10 個參數；B 的訓練誤差 1.0、2 個參數。", "The same two candidates A and B from the note: A has training error 0.5 and 10 parameters; B has 1.0 and 2."),
        1: T("只看訓練誤差：0.5 小於 1.0，A 最低。", "Training error alone: 0.5 is less than 1.0, so A is lowest."),
        2: T("加上懲罰：A 是 0.5 加 0.1 乘 10，等於 1.5；B 是 1.0 加 0.1 乘 2，等於 1.2。1.2 小於 1.5，這個評分選 B。柱高依數值比例。", "With the penalty: A is 0.5 plus 0.1 times 10, which is 1.5; B is 1.0 plus 0.1 times 2, which is 1.2. Since 1.2 is less than 1.5, this score picks B. Heights are proportional."),
        3: T("這不是把 A 的誤差算錯，而是評分規則改了。所有數值已用程式重算。", "This is not an arithmetic error in A; the scoring rule changed. All values were recomputed by program.")}))

    # ============ 頁 7:不保證泛化 ============
    g = new(7, T("分數最低，還不等於會泛化", "The lowest score does not mean it generalizes"))
    g.beat = 1
    g.rect(50, 120, 520, 500, fill=CARD, stroke=BLUE, sw=3, radius=16, z=1, shadowColor="rgba(15,23,42,0.12)", shadowBlur=14, shadowOffsetY=4)
    g.text(70, 126, 480, 44, T("已知：訓練資料", "Known: the training data"), size=24, color=BLUE, bold=True, vcenter=True, z=8)
    for name, e_, p_, x in (("A", EA, penA, 130), ("B", EB, penB, 330)):
        he, hp = float(e_) * 120, float(p_) * 120; b0 = 470
        g.rect(x, b0 - he, 90, he, fill=BLUEL, stroke=BLUE, sw=2, z=3); g.rect(x, b0 - he - hp, 90, hp, fill=ORGL, stroke=ORG, sw=2, z=3)
        g.text(x, b0 - he - hp - 36, 90, 32, rf"$S={f(e_ + p_)}$", size=22, color=INK, bold=True, align="center", vcenter=True, z=8); g.text(x, b0 + 6, 90, 32, rf"${name}$", size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.line([(100, 470), (540, 470)], MUTED, sw=3, z=4); star(g, 375, 250, 20, GOLD, z=6, stroke="#b45309", sw=2)
    g.text(80, 540, 460, 62, T("B 的訓練分數較低", "B has the lower training score"), size=22, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.beat = 2
    g.rect(710, 120, 520, 500, fill="#e2e8f0", fillOpacity=0.7, stroke=RED, sw=3, dash=True, radius=16, z=1)
    g.text(730, 126, 480, 44, T("未見資料：新資料", "Unseen: new data"), size=24, color=RED, bold=True, vcenter=True, z=8)
    for name, x in (("A", 790), ("B", 990)):
        g.rect(x, 300, 90, 170, fill="#f1f5f9", stroke=GRAY, sw=3, dash=True, radius=4, z=3); g.text(x, 340, 90, 80, "?", size=64, color=RED, bold=True, align="center", vcenter=True, z=8)
        g.text(x, 476, 90, 32, rf"${name}$", size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.rect(840, 200, 200, 52, fill=GOLD, stroke="#b45309", sw=3, radius=10, z=5); g.text(840, 200, 200, 52, T("尚未量測", "not measured"), size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.text(740, 540, 460, 62, T("講義沒有 A、B 的測試結果", "The note gives no test results for A or B"), size=22, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.beat = 3
    g.line([(585, 340), (700, 340)], GRAY, sw=7, dash=True, arrow="end", head=22, z=6)
    cross(g, 614, 300, 46, RED, z=7); g.text(560, 250, 170, 30, T("不能直接推", "no direct step"), size=18, color=RED, bold=True, align="center", z=8)
    mascot(g, 596, 400, 0.45, "confused")
    banner(g, T(r"B 的分數更低 $\neq$ B 在新資料上更好", r"B scoring lower $\neq$ B being better on new data"))
    pages.append((g, T("不保證泛化", "No guarantee of generalization"), {
        0: T("選出 B 之後，下一個問題是：它能不能用在新資料。", "After choosing B, the next question is whether it works on new data."),
        1: T("左邊是已算出的訓練分數：A 是 1.5，B 是 1.2。", "On the left are the computed training scores: A is 1.5, B is 1.2."),
        2: T("右邊是沒看過的新資料：講義沒有給 A、B 的測試結果，所以標成尚未量測。", "On the right is unseen data: the note gives no test results for A or B, so they are marked as not measured."),
        3: T("因此只能說 B 的指定分數更低，不能說 B 已證明泛化更好。懲罰改變搜尋的偏好，不會自動產生測試證據；也不能由此推出參數越少一定越好。", "So we can say B has the lower specified score, not that B has been shown to generalize better. A penalty changes the search preference but creates no test evidence; fewer parameters are not automatically better.")}))

    # ============ 頁 8:三份資料 ============
    g = new(8, T("訓練、驗證、測試各管一件事", "Training, validation and test each do one job"))
    def stack(i, color, light, name, job, lock=False):
        x = 60 + i * 400
        for k in range(3):
            g.rect(x + 24 - k * 10, 190 + k * 16, 290, 270, fill=CARD if k == 2 else light, stroke=color, sw=3, radius=12, z=2 + k)
        g.text(x + 24, 214, 290, 40, name, size=28, color=color, bold=True, align="center", vcenter=True, z=8)
        for r in range(5): g.rect(x + 54, 280 + r * 30, 230 - (r % 2) * 50, 14, fill=light, radius=7, z=6)
        g.text(x - 6, 500, 350, 100, job, size=22, color=INK, bold=True, align="center", z=8)
        if lock:
            g.rect(x + 120, 350, 110, 88, fill=GOLD, stroke="#b45309", sw=3, radius=12, z=9); g.line([(x + 142, 350), (x + 142, 310), (x + 208, 310), (x + 208, 350)], "#b45309", sw=9, z=8)
            g.ellipse(x + 175, 394, 11, fill=INK, z=10)
    g.beat = 1; stack(0, BLUE, BLUEL, T("訓練集", "Training set"), T("拿來擬合模型\n（調參數）", "fit the model\n(adjust parameters)"))
    g.beat = 2; g.line([(385, 330), (445, 330)], GRAY, sw=5, arrow="end", head=18, z=6); stack(1, ORG, ORGL, T("驗證集", "Validation set"), T("調設定、選模型\n（可反覆使用）", "tune settings, pick a model\n(may be used repeatedly)"))
    g.beat = 3; g.line([(785, 330), (845, 330)], GRAY, sw=5, arrow="end", head=18, z=6); stack(2, RED, REDL, T("測試集", "Test set"), T("固定模型後，最後檢查\n（不拿來挑設定）", "final check of the fixed model\n(not for choosing settings)"), True)
    pill(g, 1100, 130, 130, T("講義補充", "Added note"), GRAY, size=16)
    banner(g, T("反覆拿測試集挑設定，它就不再獨立了", "Choose settings on the test set again and again and it stops being independent"))
    pages.append((g, T("三份資料的分工", "Three datasets, three jobs"), {
        0: T("實作時，資料有三種用途。這一頁是講義的補充，講義沒有指定切分比例。", "In practice, data serves three purposes. This page is a supplement; the note prescribes no split ratio."),
        1: T("訓練集用來擬合模型。", "The training set is used to fit the model."),
        2: T("驗證集協助調整設定與選模型，可以反覆使用。", "The validation set helps tune settings and select models, and can be reused."),
        3: T("測試集留給固定好的模型做最後評估，所以帶一把鎖。反覆用測試集挑設定，它就失去獨立檢驗的用途。", "The test set evaluates the fixed model at the end, hence the lock. Selecting settings on it repeatedly destroys its independent role.")}))

    # ============ 頁 9:argmin vs min ============
    g = new(9, T("argmin 給模型，min 給分數", "argmin returns a model, min returns a value"))
    base9 = 480; U9 = 140
    g.beat = 1
    g.rect(50, 120, 460, 480, fill=CARD, stroke=GRAY, sw=2, radius=16, z=1, shadowColor="rgba(15,23,42,0.12)", shadowBlur=14, shadowOffsetY=4)
    g.text(70, 126, 420, 40, T(r"兩個候選的分數 $S$", r"Scores $S$ of the two candidates"), size=22, color=MUTED, bold=True, vcenter=True, z=8)
    for name, s_, x in (("A", SA, 110), ("B", SB, 320)):
        h = float(s_) * U9; win = name == "B"
        g.rect(x, base9 - h, 100, h, fill=PURL if win else "#e2e8f0", stroke=PUR if win else GRAY, sw=3, radius=4, z=3)
        g.text(x, base9 - h - 38, 100, 34, rf"${f(s_)}$", size=26, color=PUR if win else INK, bold=True, align="center", vcenter=True, z=8); g.text(x, base9 + 6, 100, 34, rf"${name}$", size=26, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.line([(80, base9), (480, base9)], MUTED, sw=3, z=4); star(g, 370, base9 - float(SB) * U9 - 70, 22, GOLD, z=6, stroke="#b45309", sw=2)
    g.beat = 2
    g.line([(515, 300), (590, 230)], GRN, sw=5, arrow="end", head=18, z=6); g.line([(515, 330), (590, 430)], PUR, sw=5, arrow="end", head=18, z=6)
    g.rect(600, 140, 630, 190, fill=GRNL, stroke=GRN, sw=3, radius=16, z=1); g.text(620, 148, 590, 42, T("數值（最小分數）", "a value (the minimum score)"), size=24, color=GRN, bold=True, vcenter=True, z=8)
    g.text(620, 196, 590, 110, r"$\min_{f\in\mathcal F}S(f)=1.2$", size=48, color=INK, align="center", vcenter=True, z=8)
    g.rect(600, 360, 630, 190, fill=PURL, stroke=PUR, sw=3, radius=16, z=1); g.text(620, 368, 590, 42, T("模型（達到最小的候選）", "a model (a candidate attaining it)"), size=24, color=PUR, bold=True, vcenter=True, z=8)
    g.text(620, 416, 590, 110, r"$\arg\min_{f\in\mathcal F}S(f)=\{B\}$", size=48, color=INK, align="center", vcenter=True, z=8)
    g.text(600, 560, 630, 70, T("本例只有 2 個候選，各算一次就比完；寫出 argmin 不等於已經找到。", "With only 2 candidates, scoring each once settles it; writing argmin does not mean it has been found."), size=20, color=MUTED, z=8)
    g.beat = 3
    banner(g, T(r"下一站 KP02：資料怎麼寫成向量與矩陣 $\to$", r"Next, KP02: how data is written as vectors and matrices $\to$"))
    pages.append((g, T("argmin 與 min", "argmin and min"), {
        0: T("最後把算出來的分數放在一起：A 是 1.5，B 是 1.2。", "Finally, put the computed scores together: A is 1.5, B is 1.2."),
        1: T("最小分數的柱子是 B。", "The lowest bar is B."),
        2: T("min 給你最小分數 1.2，是一個數值；argmin 給你達到它的候選，是模型 B。型態不同。若有並列，要說明怎麼選；若最低分沒有候選達到，argmin 可能是空集合。寫出 argmin 只定義目標，不代表演算法已找到；這裡只有兩個候選，各算一次就比完。", "min gives the minimum score 1.2, a value; argmin gives the candidate attaining it, the model B. Their types differ. Ties need a selection rule, and if no candidate attains the infimum, argmin may be empty. Writing argmin only defines the target; with two candidates, scoring each once settles it."),
        3: T("理解了選模型的框架，下一個問題是：模型到底接收什麼？KP02 把一個樣本寫成向量，把一批樣本排成矩陣。", "With the selection framework in place, ask what the model receives. KP02 writes one sample as a vector and several samples as a matrix.")}))
    return pages
