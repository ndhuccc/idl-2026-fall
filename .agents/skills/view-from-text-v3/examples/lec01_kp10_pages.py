import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *
from lec01_kp10_nums import *

KP = "KP10"; N = 8

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    # ============ 1:兩條線 ============
    g = new(1, T("KP10 用共同框架選下一步", "KP10 One shared map for choosing the next step"))
    g.beat = 1
    for i, (col, light, nm) in enumerate(((BLUE, BLUEL, T("輸入", "input")), (ORG, ORGL, T("可用的回饋", "available feedback")), (GRN, GRNL, T("輸出", "output")))):
        x = 60 + i * 400
        g.rect(x, 140, 340, 130, fill=light, stroke=col, sw=3, radius=16, z=3); g.text(x, 140, 340, 130, nm, size=30, color=col, bold=True, align="center", vcenter=True, z=8)
        if i < 2: arrow(g, (x + 348, 205), (x + 392, 205), GRAY, sw=5, head=16)
    g.text(60, 280, 1160, 40, T("先寫清楚這三件事，任務才算定義好", "write these three down first to define the task"), size=22, color=MUTED, bold=True, align="center", vcenter=True, z=8)
    g.beat = 2
    for i, (col, light, nm, sym) in enumerate(((BLUE, BLUEL, T("假設", "hypotheses"), r"$\mathcal H$"), (ORG, ORGL, T("評分", "score"), r"$S(h;D)$"), (GRN, GRNL, T("搜尋", "search"), T("程序", "procedure")))):
        x = 60 + i * 400
        g.rect(x, 350, 340, 160, fill=light, stroke=col, sw=3, radius=16, z=3); g.text(x, 360, 340, 70, nm, size=28, color=col, bold=True, align="center", vcenter=True, z=8); g.text(x, 435, 340, 60, sym, size=30, color=INK, align="center", vcenter=True, z=8)
        if i < 2: arrow(g, (x + 348, 430), (x + 392, 430), GRAY, sw=5, head=16)
    g.beat = 3
    g.rect(60, 540, 1160, 70, fill=REDL, stroke=RED, sw=3, radius=14, z=3); g.text(80, 540, 1120, 70, T("最後：用沒用來擬合的資料，或新的互動，檢查結果", "Finally: check on data not used for fitting, or on new interaction"), size=24, color=RED, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("輸入、回饋、輸出 → 假設、評分、搜尋 → 新情境檢查", "input, feedback, output → hypotheses, score, search → new-condition check"))
    pages.append((g, T("兩段框架", "A two-part frame"), {
        0: T("這是第一章的最後一個單元。先寫清楚輸入、回饋與輸出，任務才算定義好。", "This is the chapter's last unit. First write down the input, feedback and output; only then is the task defined."),
        1: T("再沿用 KP01 的三步：指定候選的假設集合，用資料定義評分，最後用實際的搜尋程序去找低分的候選。", "Then reuse KP01's three parts: specify the hypothesis class, define a score from data, and use an actual search procedure to find low-scoring candidates."),
        2: T("最後用沒用來擬合的資料，或新的互動，檢查結果。", "Finally check on data not used for fitting, or on new interaction."),
        3: T("先定義任務，再選模型，再用擬合以外的資料驗證。", "Define the task, then choose a model, then validate beyond the training examples.")}))

    # ============ 2:任務地圖 ============
    g = new(2, T("資料與回饋，決定這是什麼任務", "Data and feedback decide what the task is"))
    g.beat = 1
    g.poly([(640, 120), (780, 170), (640, 220), (500, 170)], fill=PURL, stroke=PUR, sw=3, z=3); g.text(520, 142, 240, 56, T("有什麼回饋？", "what feedback?"), size=22, color=PUR, bold=True, align="center", vcenter=True, z=8)
    branches = [(BLUE, BLUEL, T("有類別標籤", "class labels"), T("分類", "classification"), 120), (ORG, ORGL, T("有連續數值", "numeric targets"), T("回歸", "regression"), 420), (GRN, GRNL, T("沒有標籤", "no labels"), T("分群 / 降維", "clustering / reduction"), 720), (RED, REDL, T("行動改變狀態、有獎勵", "actions change state, with rewards"), T("互動決策（強化學習）", "interactive decisions (RL)"), 1010)]
    for i, (col, light, cond, res, x) in enumerate(branches):
        g.beat = 1 if i < 2 else 2
        g.line([(640 if i in (1, 2) else 640, 220), (x + 100, 300)], col, sw=4, arrow="end", head=14, z=4)
        g.rect(x - 40, 300, 280, 100, fill=CARD, stroke=col, sw=3, radius=14, z=3); g.text(x - 40, 300, 280, 100, cond, size=21, color=col, bold=True, align="center", vcenter=True, z=8) if False else g.text(x - 36, 306, 272, 90, cond, size=20, color=col, bold=True, align="center", z=8)
        g.rect(x - 40, 440, 280, 90, fill=light, stroke=col, sw=3, radius=14, z=3); g.text(x - 36, 446, 272, 80, res, size=22, color=INK, bold=True, align="center", z=8)
        g.line([(x + 100, 404), (x + 100, 436)], col, sw=4, arrow="end", head=12, z=4)
    g.beat = 3
    g.text(50, 546, 1180, 80, T("分群、一般的無標籤降維常屬非監督；帶標籤的降維（KP05 的 FDA）提醒我們：\n方法名稱不能取代資料與回饋條件", "clustering and plain unlabeled reduction are usually unsupervised; supervised reduction (KP05's FDA) reminds us:\na method's name cannot replace the data and feedback conditions"), size=19, color=MUTED, bold=True, align="center", z=8)
    banner(g, T("每一條分支，都要回到同一個問題：要評估什麼？", "Every branch returns to one question: what will be evaluated?"))
    pages.append((g, T("任務地圖", "The task map"), {
        0: T("同一份影像或向量，可以服務不同的問題。要先問：有什麼回饋？", "The same images or vectors can serve different problems. First ask: what feedback is there?"),
        1: T("有類別標籤，可能做分類；有連續的年齡之類的數值，可能做回歸。", "With class labels you might classify; with continuous values such as age you might regress."),
        2: T("暫時沒有標籤、只想整理結構，可以考慮分群或降維；若代理的行動會改變下一個狀態並收到獎勵，才進入互動決策。", "With no labels, to organize structure, consider clustering or reduction; only when an agent's actions change the next state and bring rewards do we enter interactive decision-making."),
        3: T("方法名稱不能取代資料與回饋條件；每一條分支都要回到要評估什麼。", "A method's name cannot replace data and feedback conditions; every branch must return to what is evaluated.")}))

    # ============ 3:三步與 argmin ============
    g = new(3, T("三步流程：先定目標，再找人去達成", "Three parts: fix the target, then try to reach it"))
    g.beat = 1
    card(g, 90, 112, 1100, 150, GRAY); g.text(110, 116, 1060, 142, r"$h^{*}\in\arg\min_{h\in\mathcal H}\ S(h;D)$", size=42, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    for i, (col, light, sym, nm, sub) in enumerate(((BLUE, BLUEL, r"$\mathcal H$", T("假設集合", "hypothesis class"), T("誰可以被選", "who may be chosen")), (ORG, ORGL, r"$S(h;D)$", T("損失／評分", "loss / score"), T("用資料 $D$ 定義", "defined from the data $D$")), (GRN, GRNL, T("搜尋", "search"), T("實際的程序", "an actual procedure"), T("嘗試找到低分者", "tries to find low scores")))):
        x = 60 + i * 400
        g.rect(x, 290, 360, 240, fill=light, stroke=col, sw=3, radius=16, z=3); star(g, x + 40, 322, 14, col, z=6) if False else g.ellipse(x + 40, 322, 14, fill=col, z=6)
        if i < 2: arrow(g, (x + 366, 410), (x + 394, 410), GRAY, sw=4, head=12)
        g.text(x, 300, 360, 70, sym, size=34, color=col, bold=True, align="center", vcenter=True, z=8) if i < 2 else g.text(x, 300, 360, 70, nm, size=30, color=col, bold=True, align="center", vcenter=True, z=8)
        g.text(x, 380, 360, 50, nm if i < 2 else "", size=24, color=INK, bold=True, align="center", vcenter=True, z=8); g.text(x + 20, 440, 320, 70, sub, size=20, color=INK, align="center", z=8)
    g.beat = 3
    g.rect(90, 555, 1100, 65, fill=REDL, stroke=RED, sw=3, radius=14, z=3); g.text(110, 555, 1060, 65, T("這個記號寫的是「目標」：不是梯度下降或樹搜尋，也不保證找到全域解", "the notation states a target, not an algorithm or a guarantee"), size=22, color=RED, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("輸入與輸出的型態、限制與資料切分，會改變可選的 $\\mathcal H$ 與 $S$", "Input and output types, constraints and data splits change the feasible $\\mathcal H$ and $S$"))
    pages.append((g, T("三步與目標", "The three parts and the target"), {
        0: T("沿用 KP01：先定假設集合 H，再用資料 D 定義損失或評分 S，最後用實際的搜尋程序尋找低分的候選。若最小解存在，可用 h 星屬於 argmin 表示目標。", "Reusing KP01: first the hypothesis class H, then a loss or score S defined from data D, finally an actual search procedure for low-scoring candidates. If a minimizer exists, h-star in the argmin states the target."),
        1: T("三個角色各做一件事。", "Each of the three roles does one job."),
        2: T("這個記號本身不是梯度下降、樹搜尋或任何具體演算法，也不保證一定找到全域解。輸入表示 x 與輸出 y 的型態、限制與資料切分，會改變可選的 H 與 S。", "The notation itself is not gradient descent, tree search or any concrete algorithm, and it does not guarantee a global solution. The type of input x and output y, constraints and data splits change the feasible H and S."),
        3: T("這個框架可以套在任何任務上。", "The frame applies to any task.")}))

    # ============ 4-6:三個情境 ============
    def scenario(n, title_zh, title_en, col, light, rows, notes, icon, label_zh, label_en):
        g = new(n, T(title_zh, title_en))
        pill(g, 960, 96, 270, T("講義補充：候選，不是標準答案", "added: candidates, not answers"), GRAY, size=16)
        icon(g)
        labels = (T("輸入與回饋", "input and feedback"), T("輸出與候選", "output and candidates"), T("評分與檢查", "score and check"))
        for i, (nm, sub) in enumerate(zip(labels, rows)):
            g.beat = i + 1
            x = 50 + i * 400
            g.rect(x, 330, 380, 270, fill=CARD, stroke=col, sw=3, radius=16, z=3); g.rect(x, 330, 380, 60, fill=light, radius=16, z=4); g.rect(x, 370, 380, 20, fill=light, z=4)
            g.text(x + 10, 330, 360, 60, nm, size=22, color=col, bold=True, align="center", vcenter=True, z=8); g.text(x + 20, 420, 340, 170, sub, size=24 if zh else 20, color=INK, bold=True, align="center", z=8)
            if i < 2: arrow(g, (x + 386, 465), (x + 414, 465), GRAY, sw=4, head=12)
        g.beat = 3
        banner(g, T("實際選擇，要看資料量、成本、錯誤類型與未見資料的表現", "The real choice depends on data size, cost, error types and unseen-data performance"))
        pages.append((g, T(label_zh, label_en), notes))
    def icon_spam(g):
        g.beat = 0
        g.rect(430, 130, 420, 180, fill=BLUEL, stroke=BLUE, sw=4, radius=14, z=3); g.poly([(430, 130), (640, 250), (850, 130)], fill="#bfdbfe", stroke=BLUE, sw=4, z=4)
        g.rect(560, 215, 160, 60, fill=REDL, stroke=RED, sw=3, radius=10, z=5); g.text(560, 215, 160, 60, "SPAM", size=22, color=RED, bold=True, align="center", vcenter=True, z=8)
        g.ellipse(470, 290, 14, fill=BLUE, z=6); g.ellipse(810, 290, 14, fill=RED, z=6); star(g, 640, 160, 14, GOLD, z=6)
    def icon_sensor(g):
        g.beat = 0
        for k, (x, y, c) in enumerate([(480, 200, BLUE), (520, 240, BLUE), (560, 190, BLUE), (700, 260, ORG), (740, 210, ORG), (770, 270, ORG), (640, 150, GRN), (620, 290, "#64748b")]): g.ellipse(x, y, 14, fill=c, stroke="#ffffff", sw=2, z=5)
        g.ellipse(520, 215, 70, stroke=BLUE, sw=4, z=3, fill="transparent"); g.ellipse(735, 240, 70, stroke=ORG, sw=4, z=3, fill="transparent")
    def icon_robot(g):
        g.beat = 0
        mascot(g, 590, 120, 0.5, "proud")
    scenario(4, "垃圾郵件：有標籤，是分類", "Spam: labeled data, a classification", BLUE, BLUEL,
             (T("郵件特徵\n+ 人工的類別標籤", "mail features\n+ human class labels"), T("類別\n邏輯迴歸、樹或神經網路", "a class\nlogistic model, tree or network"), T("按錯誤代價選分類指標\n留出新郵件驗證", "pick a class metric by error cost\ntest on new mail")),
             {0: T("第一個情境：垃圾郵件。這三個情境是講義的補充，每一列只給候選，不是標準答案。", "First scenario: spam. The three scenarios are supplements; each row gives only candidates, not answers."), 1: T("輸入是郵件特徵與人工的類別標籤。", "The input is mail features with human class labels."), 2: T("輸出是類別，候選有邏輯迴歸、樹或神經網路。", "The output is a class, with candidates such as logistic models, trees or networks."), 3: T("評分按錯誤代價選分類指標，並留出新郵件驗證。實際選擇要看資料量、成本、錯誤類型及未見資料表現。", "Score by a class metric chosen from error costs and test on new mail. The real choice depends on data size, cost, error types and unseen-data performance.")}, icon_spam, "垃圾郵件", "Spam")
    scenario(5, "感測器整理：沒有標籤，是分群或降維", "Sensor structure: no labels, clustering or reduction", ORG, ORGL,
             (T("時序特徵\n沒有人工標籤", "time-series features\nno human labels"), T("群或低維表示\nK-means、PCA 等候選", "groups or a representation\ncandidates: K-means, PCA, …"), T("查穩定性、重建或下游用途\n沒有單一「正確標籤」準確率", "check stability, reconstruction or downstream use\nthere is no single “correct-label” accuracy")),
             {0: T("第二個情境：感測器整理。", "Second scenario: organizing sensor data."), 1: T("輸入是時序特徵，沒有人工標籤。", "The input is time-series features without human labels."), 2: T("輸出是群或低維表示，候選有 K-means、PCA 等。", "The output is groups or a low-dimensional representation, with candidates such as K-means and PCA."), 3: T("分群不等於沒有評分：要查穩定性、重建或下游用途；因為沒有人工標籤，沒有單一正確標籤準確率。", "Clustering is not scoring-free: check stability, reconstruction or downstream use; with no human labels there is no single correct-label accuracy.")}, icon_sensor, "感測器整理", "Sensor data")
    scenario(6, "掃地機器人：行動有後果，是互動決策", "Cleaning robot: interactive decisions", GRN, GRNL,
             (T("狀態、行動\n與後續獎勵", "states, actions\nand later rewards"), T("策略\n值函數或策略模型候選", "a policy\nvalue-function or policy-model candidates"), T("比較累積回報與安全條件\n觀察新環境", "compare return and safety\nobserve a new environment")),
             {0: T("第三個情境：掃地機器人。掃到灰塵和碰牆，可以回不同的訊號；數字是設計者給定的，不是實測。", "Third scenario: a cleaning robot. Sweeping dust and hitting a wall can return different signals; numbers are designer-set, not measured."), 1: T("輸入是狀態、行動與後續的獎勵。", "The input is states, actions and the rewards that follow."), 2: T("輸出是策略，候選有值函數或策略模型。", "The output is a policy, with value-function or policy-model candidates."), 3: T("評分比較累積回報與安全條件，並觀察新環境。強化學習不等於所有負值回饋。", "Score by comparing return and safety conditions and observing a new environment. Reinforcement learning is not all negative feedback.")}, icon_robot, "掃地機器人", "Cleaning robot")

    # ============ 7:歷史是工具箱 ============
    g = new(7, T("歷史是工具箱，不是誰淘汰誰的階梯", "History is a toolbox, not a ladder of replacement"))
    g.beat = 1
    card(g, 50, 130, 570, 470, RED); head(g, 70, 136, 530, T("不要這樣排", "Not like this"), RED, 24)
    for i, nm in enumerate(("Bayes", "SVM", "Ridge", "PCA", "CNN", "Transformer")):
        y = 540 - i * 62; g.rect(110 + i * 55, y, 150, 50, fill=REDL, stroke=RED, sw=2, radius=8, z=3); g.text(110 + i * 55, y, 150, 50, nm, size=18, color=INK, bold=True, align="center", vcenter=True, z=8)
    cross(g, 480, 200, 60, RED, z=8)
    g.beat = 2
    card(g, 660, 130, 570, 470, GRN); head(g, 680, 136, 530, T("要分開問四個維度", "Ask four dimensions separately"), GRN, 24)
    for i, (nm, col, light) in enumerate(((T("任務", "task"), BLUE, BLUEL), (T("模型", "model"), ORG, ORGL), (T("損失", "loss"), PUR, PURL), (T("搜尋", "search"), GRN, GRNL))):
        x = 700 + (i % 2) * 260; y = 210 + (i // 2) * 170
        g.rect(x, y, 240, 140, fill=light, stroke=col, sw=3, radius=14, z=3); g.text(x, y, 240, 140, nm, size=30, color=col, bold=True, align="center", vcenter=True, z=8)
    for i, col in enumerate((BLUE, ORG, PUR, GRN)): g.ellipse(700 + (i % 2) * 260 + 30, 210 + (i // 2) * 170 + 30, 11, fill=col, z=6)
    g.beat = 3
    banner(g, T("KP08 分開問「表達」與「訓練」；KP09 分開問「資料結構」與「學習目標」", "KP08 separates representation from training; KP09 separates data structure from objective"))
    pages.append((g, T("工具箱", "A toolbox"), {
        0: T("把 Bayes、SVM、Ridge、PCA、CNN、Transformer 直接排成誰淘汰誰的世代序列，會混淆任務、模型、損失及搜尋四個維度。", "Lining up Bayes, SVM, Ridge, PCA, CNN and Transformer as a succession of replacements mixes up four dimensions: task, model, loss and search."),
        1: T("正確的問法，是分開問任務、模型、損失與搜尋。", "The right way is to ask task, model, loss and search separately."),
        2: T("KP08 的感知機與 ADALINE，提醒我們分開問表達與訓練；KP09 的 CNN、Transformer、生成模型與偏好方法，提醒我們分開問資料結構與學習目標。常見迷思：分群不等於沒有評分；強化學習不等於所有負值回饋；神經網路也不自動免除資料表示、損失設計與泛化檢查。", "KP08's Perceptron and ADALINE ask us to separate representation from training; KP09's CNN, Transformer, generative and preference methods ask us to separate data structure from learning objective. Misconceptions: clustering is not scoring-free; RL is not all negative feedback; neural networks do not waive data representation, loss design or generalization checks."),
        3: T("歷史標示工具的位置，但不構成必須依序取代的鏈。", "History places tools but does not form a mandatory replacement chain.")}))

    # ============ 8:回看十個 KP ============
    g = new(8, T("回看十個 KP：這一章收成一張決策圖", "Looking back at ten KPs: the chapter as one decision map"))
    pill(g, 1090, 84, 140, T("講義補充", "Added note"), GRAY, size=16)
    groups = [(BLUE, BLUEL, "KP01", T("選擇框架", "the frame"), 0, 1), (ORG, ORGL, "KP02", T("資料映射", "data mappings"), 1, 1), (GRN, GRNL, "KP03–05", T("標籤、無標籤、環境回饋", "labels, no labels, environment"), 2, 3), (PUR, PURL, "KP06–07", T("分類、回歸、表示工具", "classification, regression, representation"), 5, 2), (RED, REDL, "KP08–09", T("網路的表達、訓練、架構歷史", "network capacity, training, architecture history"), 7, 2), (GOLD, "#fef3c7", "KP10", T("決策圖", "the decision map"), 9, 1)]
    for i, (col, light, kp, nm, start, n) in enumerate(groups):
        g.beat = 1 if i < 3 else (2 if i < 5 else 3)
        x = 40 + start * 118; w = n * 118 - 10
        g.rect(x, 200 + (i % 2) * 130, w, 110, fill=light, stroke=col, sw=3, radius=14, z=3); g.text(x + 4, 204 + (i % 2) * 130, w - 8, 40, kp, size=22, color=col if col != GOLD else "#92400e", bold=True, align="center", vcenter=True, z=8); g.text(x + 6, 246 + (i % 2) * 130, w - 12, 60, nm, size=16, color=INK, bold=True, align="center", z=8)
    g.line([(60, 450), (1200, 450)], GRAY, sw=5, z=3, arrow="end", head=20)
    for k in range(10): g.ellipse(100 + k * 118, 450, 9, fill=GRAY, z=5)
    g.beat = 3
    g.text(50, 480, 1180, 100, T("第 2 章：先把一個可學習的神經元寫清楚\n第 3 章：多層網路的梯度與反向傳播", "Chapter 2: write one trainable neuron clearly\nChapter 3: gradients and backpropagation for multilayer networks"), size=26, color=INK, bold=True, align="center", z=8)
    mascot(g, 560, 540, 0.0 + 0.0 + 0.4, "proud") if False else None
    banner(g, T(r"輸入、回饋、輸出 $\to$ 假設、評分、搜尋 $\to$ 新情境檢查", r"input, feedback, output $\to$ hypotheses, score, search $\to$ new-condition check"))
    pages.append((g, T("回看與下一章", "Review and next chapter"), {
        0: T("這個單元把前面十個知識點收成一張決策圖。KP01 是選擇框架，KP02 是資料映射，KP03 到 05 是標籤、無標籤與環境回饋。這一頁是講義的補充。", "This unit gathers the ten knowledge points into one decision map. KP01 is the selection frame, KP02 data mappings, KP03 to 05 labels, no labels and environment feedback. This page is a supplement."),
        1: T("KP06 與 07 是分類、回歸與表示工具；KP08 與 09 是神經網路的表達、訓練與架構歷史。", "KP06 and 07 are classification, regression and representation tools; KP08 and 09 are neural-network capacity, training and architecture history."),
        2: T("KP10 把它們收成一張決策圖。第 2 章先把一個可學習的神經元寫清楚，接著第 3 章處理多層網路的梯度與反向傳播。口訣：輸入、回饋、輸出，假設、評分、搜尋，新情境檢查。", "KP10 gathers them into one decision map. Chapter 2 first writes one trainable neuron clearly, then Chapter 3 handles gradients and backpropagation for multilayer networks. Rule of thumb: input, feedback, output; hypotheses, score, search; new-condition check."),
        3: T("講義的 4 題自我檢核留在講義。", "The note's 4 self-check questions stay in the note.")}))
    return pages
