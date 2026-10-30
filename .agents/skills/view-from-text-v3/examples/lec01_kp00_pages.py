import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *

KP = "KP00"; N = 6
# (tag, group colour, light colour, zh title, en title)
KPS = [("KP01", BLUE, BLUEL, "選模型要有評分標準", "Model selection needs a scoring rule"),
       ("KP02", ORG, ORGL, "資料是向量，模型是映射", "Data are vectors, models are mappings"),
       ("KP03", ORG, ORGL, "標籤決定分類或回歸", "Labels split classification and regression"),
       ("KP04", ORG, ORGL, "無標籤資料也能找結構", "Unlabeled data can reveal structure"),
       ("KP05", ORG, ORGL, "回饋型態改變學習目標", "Feedback changes the objective"),
       ("KP06", GRN, GRNL, "分類工具各有不同假設", "Classifiers make different assumptions"),
       ("KP07", GRN, GRNL, "回歸與降維保留不同資訊", "Regression and reduction keep different things"),
       ("KP08", PUR, PURL, "從邏輯單元走向可學權重", "From logic units to learnable weights"),
       ("KP09", PUR, PURL, "架構與訓練共同推動復興", "Architecture and training drive revival"),
       ("KP10", RED, REDL, "用共同框架選下一步", "One shared map for the next step")]

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    # ============ 1: the one question ============
    g = new(1, T("KP00 第一章：從任務到可學網路", "KP00 Chapter 1: from a task to a trainable network"))
    g.beat = 1
    card(g, 50, 120, 1180, 120, GOLD)
    g.text(70, 150, 1140, 80, T("怎樣替一件新任務，選出在沒看過的資料上也可靠的模型？", "How do we pick, for a new task, a model that stays reliable on data it has not seen?"), size=30, color=INK, bold=True, align="center", z=8)
    g.beat = 2
    stages = [(BLUE, BLUEL, T("認清任務", "Recognize the task"), T("輸入 · 回饋 · 輸出", "input · feedback · output"), "KP02–KP05"),
              (GRN, GRNL, T("選出模型", "Pick a model"), T("假設 · 評分 · 搜尋", "hypotheses · score · search"), "KP01、KP06–KP09" if zh else "KP01, KP06–KP09"),
              (RED, REDL, T("檢查結果", "Check the result"), T("未見資料或新互動", "unseen data or new interaction"), "KP01、KP10" if zh else "KP01, KP10")]
    ids1 = []
    for i, (col, light, nm, sub, kps) in enumerate(stages):
        x = 50 + i * 410
        ids1.append(g.mark()); g.rect(x, 290, 360, 210, fill=light, stroke=col, sw=3, radius=16, z=3)
        g.text(x, 300, 360, 56, nm, size=30, color=col, bold=True, align="center", vcenter=True, z=8)
        g.text(x + 10, 362, 340, 60, sub, size=22, color=INK, bold=True, align="center", z=8)
        pill(g, x + 80, 450, 200, kps, col, size=18)
        if i: g.link(ids1[i - 1], ids1[i], 4, 12, GRAY, 5)
    g.beat = 3
    g.text(50, 530, 960, 90, T("第一章的十個單元，都在回答這個問題", "The ten units of this chapter all answer this one question"), size=26, color=MUTED, bold=True, align="left", vcenter=True, z=8)
    mascot(g, 1090, 520, 0.4, "proud")
    banner(g, T("先認清任務，再選模型，最後用沒用來擬合的資料檢查", "recognize the task, pick a model, check on data not used for fitting"))
    pages.append((g, T("本章的一個問題", "The chapter's one question"), {
        0: T("教材程度是大學生，全章三小時。先看整章只問一個問題。", "Level: undergraduate; the chapter takes three hours. First, the one question the whole chapter asks."),
        1: T("問題是：碰到一件新任務，怎樣選出在沒看過的資料上也可靠的模型。", "The question: facing a new task, how do we pick a model that is reliable on data it has not seen?"),
        2: T("回答分三段：先認清任務，也就是輸入、回饋與輸出；再用假設、評分、搜尋選出模型；最後用未見資料或新的互動檢查。", "The answer has three parts: recognize the task (input, feedback, output); pick a model with hypotheses, a score and a search; finally check on unseen data or new interaction."),
        3: T("十個單元就是照這三段排的。", "The ten units follow these three parts.")}))

    # ============ 2: roadmap (snake) ============
    g = new(2, T("十個單元，由左到右、再繞回來", "Ten units: left to right, then back"), )
    card_ix = {}
    def kcard(i, x, y):
        tag, col, light, tz, te = KPS[i]
        card_ix[i] = g.mark(); g.rect(x, y, 216, 140, fill=light, stroke=col, sw=3, radius=16, z=3)
        g.text(x + 12, y + 8, 192, 34, tag, size=22, color=col, bold=True, vcenter=True, z=8)
        g.text(x + 12, y + 44, 192, 90, tz if zh else te, size=20, color=INK, bold=True, z=8)
    g.beat = 1
    for i in range(5):
        kcard(i, 50 + i * 241, 130)
        if i: g.link(card_ix[i - 1], card_ix[i], 4, 12)
    g.beat = 2
    for j, i in enumerate(range(5, 10)):
        x = 50 + (4 - j) * 241
        kcard(i, x, 340)
        g.link(card_ix[i - 1], card_ix[i], 8 if j == 0 else 12, 0 if j == 0 else 4)
    g.beat = 3
    leg = [(BLUE, BLUEL, T("框架", "frame")), (ORG, ORGL, T("資料與任務", "data and tasks")), (GRN, GRNL, T("方法", "methods")), (PUR, PURL, T("歷史", "history")), (RED, REDL, T("總地圖", "full map"))]
    for k, (col, light, nm) in enumerate(leg):
        x = 50 + k * 241
        g.rect(x, 520, 216, 50, fill=light, stroke=col, sw=3, radius=25, z=3); g.text(x, 520, 216, 50, nm, size=22, color=col, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("框架 → 資料與任務 → 方法 → 歷史 → 總地圖", "frame → data and tasks → methods → history → full map"))
    pages.append((g, T("十個單元的路線", "The ten-unit route"), {
        0: T("這是全章的路線。先看第一排。", "This is the route of the whole chapter. Start with the first row."),
        1: T("第一排：KP01 先講選模型要有評分標準；KP02 到 KP05 看資料與任務：向量、標籤、無標籤、回饋型態。", "Row one: KP01 says model selection needs a scoring rule; KP02 to KP05 cover data and tasks: vectors, labels, no labels, kinds of feedback."),
        2: T("第二排從右邊接回來：KP06、KP07 是分類、回歸與降維的方法；KP08、KP09 是神經網路的歷史；KP10 把全部收進一張地圖。", "Row two comes back from the right: KP06 and KP07 are methods for classification, regression and reduction; KP08 and KP09 are neural-network history; KP10 gathers everything into one map."),
        3: T("顏色代表五段分組，後面的單元都用同樣的顏色。", "Colors mark the five groups and stay the same in later units.")}))

    # ============ 3: time bar ============
    g = new(3, T("3 小時裡，每個單元約 15 分鐘", "In 3 hours, each unit takes about 15 minutes"))
    g.beat = 1
    for i in range(10):
        tag, col, light, tz, te = KPS[i]
        x = 53 + i * 118
        g.rect(x, 150, 112, 120, fill=light, stroke=col, sw=3, radius=14, z=3)
        g.text(x, 156, 112, 40, tag, size=22, color=col, bold=True, align="center", vcenter=True, z=8)
        g.text(x, 206, 112, 50, T("15 分", "15 min"), size=22, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.beat = 2
    groups = [(0, 1, BLUE, T("框架", "frame")), (1, 5, ORG, T("資料與任務", "data and tasks")), (5, 7, GRN, T("方法", "methods")), (7, 9, PUR, T("歷史", "history")), (9, 10, RED, T("總地圖", "full map"))]
    for a, b_, col, nm in groups:
        x0 = 53 + a * 118; x1 = 53 + b_ * 118 - 6
        g.line([(x0, 290), (x0, 306), (x1, 306), (x1, 290)], col, sw=4, z=5)
        g.text(x0, 316, x1 - x0, 40, nm, size=22, color=col, bold=True, align="center", vcenter=True, z=8)
        g.text(x0, 358, x1 - x0, 36, f"{(b_ - a) * 15} " + T("分鐘", "min"), size=22, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.beat = 3
    g.rect(150, 450, 980, 150, fill=CARD, stroke=GOLD, sw=3, radius=16, z=3, shadowColor="rgba(15,23,42,0.12)", shadowBlur=14, shadowOffsetY=4)
    g.text(170, 460, 940, 60, r"$10\times 15 = 150$ " + T("分鐘", "minutes"), size=36, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.text(170, 528, 940, 60, T("全章 3 小時，其中淨講授 150 分鐘", "3 hours for the chapter, 150 minutes of pure teaching"), size=24, color=MUTED, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("資料與任務佔最多：4 個單元、60 分鐘", "data and tasks take the most: 4 units, 60 minutes"))
    pages.append((g, T("時間分配", "Time plan"), {
        0: T("全章三小時，每個單元配置約十五分鐘。", "The chapter has three hours, with about fifteen minutes per unit."),
        1: T("十個單元一樣長，每個十五分鐘。", "All ten units get the same fifteen minutes."),
        2: T("分組後：框架十五分鐘、資料與任務六十分鐘、方法三十分鐘、歷史三十分鐘、總地圖十五分鐘。", "By group: frame 15 minutes, data and tasks 60, methods 30, history 30, full map 15."),
        3: T("合計淨講授一百五十分鐘。資料與任務佔最多，因為後面全部都靠它。", "Pure teaching totals 150 minutes. Data and tasks take the most because everything later depends on them.")}))

    # ============ 4: task fork ============
    g = new(4, T("KP02–KP05 先認清任務", "KP02–KP05 recognize the task first"))
    g.beat = 1
    dia = g.mark(); g.poly([(640, 118), (790, 168), (640, 218), (490, 168)], fill=PURL, stroke=PUR, sw=3, z=3)
    g.text(520, 140, 240, 56, T("有什麼回饋？", "what feedback?"), size=24, color=PUR, bold=True, align="center", vcenter=True, z=8)
    br = [(BLUE, BLUEL, T("類別標籤", "class labels"), T("分類", "classification"), "KP03"),
          (ORG, ORGL, T("連續數值", "numeric targets"), T("回歸", "regression"), "KP03"),
          (GRN, GRNL, T("沒有標籤", "no labels"), T("分群 / 降維", "clustering / reduction"), "KP04"),
          (RED, REDL, T("行動改變狀態、有獎勵", "actions change state, with rewards"), T("互動決策", "interactive decisions"), "KP05")]
    for i, (col, light, cond, res, kp) in enumerate(br):
        g.beat = 2 if i < 2 else 3
        x = 50 + i * 300
        cb = g.mark(); g.rect(x, 290, 270, 100, fill=CARD, stroke=col, sw=3, radius=14, z=3); g.text(x + 6, 296, 258, 88, cond, size=22, color=col, bold=True, align="center", z=8)
        g.link(dia, cb, 8, 0, col)
        rb = g.mark(); g.rect(x, 436, 270, 90, fill=light, stroke=col, sw=3, radius=14, z=3); g.text(x + 6, 442, 258, 78, res, size=24, color=INK, bold=True, align="center", z=8)
        g.link(cb, rb, 8, 0, col)
        pill(g, x + 85, 540, 100, kp, col, size=18)
    banner(g, T("看清楚資料與回饋，才知道在解哪一種問題", "see the data and the feedback first, then name the problem"))
    pages.append((g, T("任務怎麼分", "How tasks split"), {
        0: T("KP02 到 KP05 要練的是先認清任務：問的第一句話是，有什麼回饋？", "KP02 to KP05 practice recognizing the task. The first question is: what feedback is there?"),
        1: T("有類別標籤是分類，有連續數值是回歸，這是 KP03。", "Class labels mean classification, numeric targets mean regression. That is KP03."),
        2: T("沒有標籤可以分群或降維，這是 KP04；行動會改變狀態而且有獎勵，就是互動決策，這是 KP05。", "No labels can mean clustering or reduction, which is KP04. Actions that change the state and bring rewards mean interactive decisions, which is KP05."),
        3: T("KP02 先把資料寫成向量、把模型寫成映射，這四種任務才有共同的語言。", "KP02 first writes data as vectors and models as mappings, so the four tasks share one language.")}))

    # ============ 5: methods and history ============
    g = new(5, T("KP06–KP09 方法與歷史", "KP06–KP09 methods and history"))
    cards5 = [(5, T("分類沒有單一途徑，看假設與未見資料的表現", "No single route to classification; judge by assumptions and unseen data")),
              (6, T("正則化管係數，降維改變資料的表示", "Regularization controls coefficients; reduction changes the representation")),
              (7, T("「能表示」和「能學到」是兩個問題", "“Can represent” and “can learn” are two questions")),
              (8, T("沒有單一演算法喚醒深度學習", "No single algorithm woke up deep learning"))]
    g.beat = 1
    for a, b_, col, nm in ((0, 2, GRN, T("方法", "methods")), (2, 4, PUR, T("歷史", "history"))):
        x0 = 50 + a * 301; x1 = 50 + b_ * 301 - 26
        g.line([(x0, 150), (x0, 136), (x1, 136), (x1, 150)], col, sw=4, z=5)
        g.text(x0, 100, x1 - x0, 32, nm, size=22, color=col, bold=True, align="center", vcenter=True, z=8)
    c5 = []
    for j, (i, s) in enumerate(cards5):
        g.beat = 1 if j < 2 else 2
        tag, col, light, tz, te = KPS[i]; x = 50 + j * 301
        c5.append(g.mark()); g.rect(x, 160, 275, 260, fill=light, stroke=col, sw=3, radius=16, z=3)
        g.text(x + 14, 168, 247, 34, tag, size=22, color=col, bold=True, vcenter=True, z=8)
        g.text(x + 14, 206, 247, 86, tz if zh else te, size=21, color=INK, bold=True, z=8)
        g.text(x + 14, 300, 247, 112, s, size=19, color=MUTED, z=8)
        if j: g.link(c5[j - 1], c5[j], 4, 12)
    g.beat = 3
    tag, col, light, tz, te = KPS[9]
    k10 = g.mark(); g.rect(240, 484, 800, 110, fill=light, stroke=col, sw=3, radius=16, z=3)
    g.text(260, 490, 760, 40, T("KP10 把兩條線索收進同一張地圖", "KP10 gathers both threads into one map"), size=24, color=col, bold=True, align="center", vcenter=True, z=8)
    g.text(260, 534, 760, 50, T("統計方法與神經網路，放進同一組候選來比較", "statistical methods and neural networks, compared as candidates of one set"), size=21, color=INK, align="center", vcenter=True, z=8)
    for c in c5: g.link(c, k10, 8, 0, RED)
    banner(g, T("方法教你怎麼選，歷史教你為什麼這樣選", "methods show how to choose; history shows why"))
    pages.append((g, T("方法與歷史", "Methods and history"), {
        0: T("KP06 到 KP09 分成方法和歷史兩組。", "KP06 to KP09 split into two groups: methods and history."),
        1: T("方法：KP06 比較分類工具的不同假設；KP07 看正則化與降維各自保留什麼。", "Methods: KP06 compares the assumptions of classifiers; KP07 shows what regularization and reduction each keep."),
        2: T("歷史：KP08 說明網路能表示某個函數，和能從資料學到它是兩件事；KP09 說明復興是架構、訓練、算力一起推動的。", "History: KP08 shows that a network representing a function and learning it from data are two things; KP09 shows the revival came from architecture, training and compute together."),
        3: T("KP10 把這兩條線索，加上前面的任務，收進同一張地圖。", "KP10 gathers these two threads, plus the tasks before them, into one map.")}))

    # ============ 6: can-do ============
    g = new(6, T("學完第一章，你能做到", "After chapter 1 you can"))
    items = [(BLUE, BLUEL, T("把一個新情境寫成輸入、回饋與輸出", "Write a new situation as input, feedback and output")),
             (GRN, GRNL, T("用假設、評分、搜尋比較統計方法與神經網路", "Compare statistical methods and neural networks by hypotheses, score and search")),
             (RED, REDL, T("說出要用哪種沒看過的資料，檢查結果", "Say which unseen data will check the result"))]
    for i, (col, light, s) in enumerate(items):
        g.beat = i + 1
        y = 120 + i * 150
        g.rect(50, y, 930, 120, fill=light, stroke=col, sw=3, radius=16, z=3)
        tick(g, 80, y + 38, 44, col, z=6)
        g.text(150, y + 20, 810, 84, s, size=26, color=INK, bold=True, z=8)
    g.beat = 3
    mascot(g, 1050, 330, 0.62, "proud")
    banner(g, T("練習：拿一個新情境，把三件事各寫一行", "practice: take a new situation and write one line for each"))
    pages.append((g, T("學完能做到", "What you can do"), {
        0: T("最後看這章結束後，你應該能做到的三件事。", "Finally, the three things you should be able to do after this chapter."),
        1: T("第一，把一個新情境寫成輸入、回饋與輸出。", "First, write a new situation as input, feedback and output."),
        2: T("第二，用假設、評分、搜尋，把統計方法和神經網路放在同一組候選來比較。", "Second, compare statistical methods and neural networks as one set of candidates, by hypotheses, score and search."),
        3: T("第三，說出要用哪種沒看過的資料來檢查結果。這就是 KP10 的練習。", "Third, say which unseen data will check the result. That is KP10's exercise.")}))
    return pages
