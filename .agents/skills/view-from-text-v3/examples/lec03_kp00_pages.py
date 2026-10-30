import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec03_kit import *

KP = "KP00"; N = 6
KPS = [("KP01", BLUE, BLUEL, "訓練是參數搜尋", "Training as search"), ("KP02", BLUE, BLUEL, "梯度下降與學習率", "GD and learning rate"), ("KP03", BLUE, BLUEL, "停止準則", "Stopping criteria"), ("KP04", BLUE, BLUEL, "線搜索", "Line search"), ("KP05", BLUE, BLUEL, "動量", "Momentum"), ("KP06", BLUE, BLUEL, "反向傳播", "Backpropagation"), ("KP07", ORG, ORGL, "隨機梯度下降", "SGD"), ("KP08", ORG, ORGL, "小批次與 epoch", "Minibatch and epoch"),
       ("KP09", ORG, ORGL, "SGD 收斂分析", "SGD convergence"), ("KP10", GRN, GRNL, "AdaGrad", "AdaGrad"), ("KP11", GRN, GRNL, "RMSProp 與 Adam", "RMSProp and Adam"), ("KP12", PUR, PURL, "平坦極小值", "Flat minima"), ("KP13", PUR, PURL, "Vanilla SAM", "Vanilla SAM"), ("KP14", PUR, PURL, "高效與自適應 SAM", "Efficient, adaptive SAM"), ("KP15", RED, REDL, "過參數化的收斂", "Over-parameterized convergence"), ("KP16", RED, REDL, "全章總結", "Chapter synthesis")]
GROUPS = [(0, 6, BLUE, BLUEL, "一階最佳化基礎", "first-order basics"), (6, 9, ORG, ORGL, "隨機與小批次", "stochastic, minibatch"), (9, 11, GRN, GRNL, "自適應步長", "adaptive steps"), (11, 14, PUR, PURL, "平坦與 SAM", "flatness and SAM"), (14, 16, RED, REDL, "理論與總結", "theory and synthesis")]
SELF = set(range(11, 15))   # KP12-KP15 are self-study (0-based 11..14)
INCLASS = [i for i in range(16) if i not in SELF]; assert len(INCLASS) == 12 and len(INCLASS) * 15 == 180 and sum(b - a for a, b, *_ in GROUPS) == 16

def make(lang):
    d = D(lang, KP, N); T = d.T
    g = d.page("KP00 第3章:怎樣把訓練走到低處", "KP00 Chapter 3: how training walks downhill")
    g.beat = 1
    card(g, 50, 120, 1180, 120, GOLD)
    g.text(70, 150, 1140, 80, T("一個很大的損失，怎樣一步步走到低處，而且負擔得起、走得穩、走得聰明？", "How does a huge loss get walked down step by step: affordably, steadily and cleverly?"), size=30, color=INK, bold=True, align="center", z=8)
    g.beat = 2
    stages = [(BLUE, BLUEL, T("走一步", "one step"), T("方向 · 步長 · 梯度", "direction · step · gradient"), "KP01–KP06"), (ORG, ORGL, T("算得起", "affordable"), T("SGD · 小批次 · 收斂", "SGD · minibatch · convergence"), "KP07–KP09"), (GRN, GRNL, T("走得聰明", "clever"), T("自適應 · SAM", "adaptive · SAM"), "KP10–KP14"), (RED, REDL, T("為什麼有效", "why it works"), T("條件式保證 · 驗證", "conditional guarantees · validation"), "KP15–KP16")]
    ids = []
    for i, (col, light, nm, sub, kps) in enumerate(stages):
        x = 50 + i * 305
        ids.append(g.mark()); g.rect(x, 290, 270, 210, fill=light, stroke=col, sw=3, radius=16, z=3)
        g.text(x + 6, 300, 258, 56, nm, size=26, color=col, bold=True, align="center", z=8); g.text(x + 6, 364, 258, 76, sub, size=20, color=INK, bold=True, align="center", z=8)
        pill(g, x + 55, 452, 160, kps, col, size=17)
        if i: g.link(ids[i - 1], ids[i], 4, 12, GRAY, 5)
    g.beat = 3
    g.text(50, 530, 960, 90, T("第三章的十六個單元,都在回答這個問題", "the sixteen units of chapter 3 all answer this question"), size=26, color=MUTED, bold=True, vcenter=True, z=8)
    mascot(g, 1090, 520, 0.4, "proud")
    banner(g, T("先走一步,再算得起,再走得聰明,最後問「為什麼有效」", "one step, then affordable, then clever, finally “why does it work”"))
    d.done(g, "本章的一個問題", "The one question", {0: ("教材程度是大學生。先看整章只問一個問題。", "Level: undergraduate. First, the one question the chapter asks."), 1: ("一個很大的損失，怎樣一步步走到低處，而且負擔得起、走得穩、走得聰明？", "How does a huge loss get walked down step by step, affordably, steadily and cleverly?"), 2: ("分四段：先走一步，方向、步長與梯度；再算得起，隨機梯度、小批次與收斂；再走得聰明，自適應步長與 SAM；最後為什麼有效，條件式的保證與驗證。", "Four parts: first one step, with direction, step and gradient; then affordability, with SGD, minibatches and convergence; then cleverness, with adaptive steps and SAM; finally why it works, with conditional guarantees and validation."), 3: ("十六個單元就是照這四段排的。", "The sixteen units follow these four parts.")})

    g = d.page("十六個單元,兩排走完", "Sixteen units, two rows")
    cid = {}
    for i in range(16):
        row, col_ = divmod(i, 8); xx = 50 + (col_ if row == 0 else 7 - col_) * 148; y = 130 + row * 240
        tag, col, light, tz, te = KPS[i]
        g.beat = 1 if row == 0 else 2
        cid[i] = g.mark(); g.rect(xx, y, 136, 170, fill=light, stroke=col, sw=3, radius=14, z=3, **({"dash": True} if i in SELF else {}))
        g.text(xx + 8, y + 6, 120, 34, tag, size=20, color=col, bold=True, vcenter=True, z=8); g.text(xx + 8, y + 46, 120, 118, tz if d.zh else te, size=17, color=INK, bold=True, z=8)
        if i and i != 8: g.link(cid[i - 1], cid[i], 4 if row == 0 else 12, 12 if row == 0 else 4, GRAY, 3)
        if i == 8: g.link(cid[7], cid[8], 8, 0, GRAY, 3)
    g.beat = 3
    for k, (a, b_, col, light, nz, ne) in enumerate(GROUPS):
        x = 50 + k * 238
        g.rect(x, 560, 222, 56, fill=light, stroke=col, sw=3, radius=28, z=3); g.text(x + 4, 560, 214, 56, nz if d.zh else ne, size=18, color=col, bold=True, align="center", z=8)
    banner(g, T("虛線框:KP12–KP15 是自主學習", "dashed frames: KP12–KP15 are self-study"))
    d.done(g, "路線", "The route", {0: ("第一排：KP01 訓練是參數搜尋，KP02 梯度下降與學習率，KP03 停止準則，KP04 線搜索，KP05 動量，KP06 反向傳播，KP07 隨機梯度下降，KP08 小批次與 epoch。", "Row one: KP01 training as search, KP02 gradient descent and learning rate, KP03 stopping criteria, KP04 line search, KP05 momentum, KP06 backpropagation, KP07 SGD, KP08 minibatches and epochs."), 1: ("第二排從右邊接回來：KP09 SGD 收斂分析，KP10 AdaGrad，KP11 RMSProp 與 Adam，KP12 到 KP14 平坦極小值與 SAM，KP15 過參數化的收斂，KP16 全章總結。", "Row two returns from the right: KP09 SGD convergence, KP10 AdaGrad, KP11 RMSProp and Adam, KP12 to KP14 flat minima and SAM, KP15 over-parameterized convergence, KP16 the synthesis."), 2: ("顏色代表五段分組，虛線框的 KP12 到 KP15 是自主學習。", "Colors mark five groups; the dashed KP12 to KP15 are self-study.")})

    g = d.page("3 小時裡,課堂講授 12 個單元", "In 3 hours, 12 units are taught in class")
    g.beat = 1
    for i in range(16):
        tag, col, light, tz, te = KPS[i]
        x = 50 + i * 74
        selfstudy = i in SELF
        g.rect(x, 150, 68, 120, fill=(CARD if selfstudy else light), stroke=(GRAY if selfstudy else col), sw=3, radius=10, z=3, **({"dash": True} if selfstudy else {}))
        g.text(x, 156, 68, 36, tag.replace("KP", ""), size=22, color=(GRAY if selfstudy else col), bold=True, align="center", vcenter=True, z=8)
        g.text(x, 202, 68, 50, "—" if selfstudy else "15", size=22, color=(GRAY if selfstudy else INK), bold=True, align="center", vcenter=True, z=8)
    g.beat = 2
    g.line([(50, 296), (50, 310), (50 + 11 * 74 + 68, 310), (50 + 11 * 74 + 68, 296)], BLUE, sw=4, z=5) if False else None
    g.line([(50, 290), (50, 306), (50 + 10 * 74 + 68, 306), (50 + 10 * 74 + 68, 290)], BLUE, sw=4, z=5)
    g.text(50, 314, 10 * 74 + 68, 40, T("KP01–KP11:11 個單元", "KP01–KP11: 11 units"), size=22, color=BLUE, bold=True, align="center", z=8)
    g.line([(50 + 15 * 74, 290), (50 + 15 * 74, 306), (50 + 15 * 74 + 68, 306), (50 + 15 * 74 + 68, 290)], RED, sw=4, z=5)
    g.text(50 + 15 * 74 - 20, 314, 110, 40, "KP16", size=22, color=RED, bold=True, align="center", z=8)
    g.line([(50 + 11 * 74, 290), (50 + 11 * 74, 306), (50 + 14 * 74 + 68, 306), (50 + 14 * 74 + 68, 290)], GRAY, sw=4, dash=True, z=5)
    g.text(50 + 11 * 74 - 30, 314, 4 * 74 - 10, 40, T("自主學習 KP12–15", "self-study KP12–15"), size=18, color=MUTED, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(150, 430, 980, 150, fill=CARD, stroke=GOLD, sw=3, radius=16, z=3, shadowColor="rgba(15,23,42,0.12)", shadowBlur=14, shadowOffsetY=4)
    g.text(170, 440, 940, 60, r"$12\times 15 = 180$ " + T("分鐘", "minutes"), size=36, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.text(170, 508, 940, 60, T("課堂 3 小時:KP01–KP11 與 KP16;KP12–KP15 自主學習", "3 hours in class: KP01–KP11 and KP16; KP12–KP15 are self-study"), size=24, color=MUTED, bold=True, align="center", z=8)
    banner(g, T("每個單元約 15 分鐘", "about 15 minutes per unit"))
    d.done(g, "時間分配", "Time plan", {0: ("第三章有十六個單元，每個約十五分鐘。KP12 到 KP15 是自主學習，虛線。", "Chapter three has sixteen units of about fifteen minutes each. KP12 to KP15 are self-study, drawn dashed."), 1: ("課堂講授範圍是 KP01 到 KP11，再加 KP16。", "The in-class range is KP01 to KP11 plus KP16."), 2: ("共十二個單元，一百八十分鐘，也就是三小時。", "Twelve units, 180 minutes, three hours."), 3: ("", "")})

    g = d.page("一路上的幾個年份", "Some dates along the way")
    ev = [("1951", "Robbins " + T("與", "and ") + "Monro", T("隨機逼近:步長 $\\sum\\eta=\\infty$、$\\sum\\eta^2<\\infty$", "stochastic approximation: step conditions"), "KP09", ORG, ORGL), ("1986", "Rumelhart, Hinton, Williams", T("反向傳播系統化並普及", "backpropagation systematized and popularized"), "KP06", BLUE, BLUEL), ("2011", "Duchi " + T("等人", "et al."), T("AdaGrad:逐座標的學習率", "AdaGrad: per-coordinate rates"), "KP10", GRN, GRNL), ("2012", "Hinton", T("RMSProp:出自網路課程講義", "RMSProp: from online course notes"), "KP11", GRN, GRNL), ("2017", "Keskar " + T("等人", "et al."), T("大批量與尖銳解的關聯(有條件)", "large batches and sharp minima (conditional)"), "KP12", PUR, PURL), ("2021", "Foret, Kwon " + T("等人", "et al."), T("SAM 與自適應的 ASAM", "SAM and the adaptive ASAM"), "KP12–14", PUR, PURL)]
    ids = []
    for i, (yr, who, what, kp, col, light) in enumerate(ev):
        g.beat = 1 if i < 3 else 2
        x = 50 + i * 200
        ids.append(g.mark()); g.rect(x, 150, 180, 360, fill=light, stroke=col, sw=3, radius=16, z=3)
        g.text(x, 160, 180, 56, yr, size=32, color=col, bold=True, align="center", z=8); g.text(x + 6, 226, 168, 90, who, size=18, color=INK, bold=True, align="center", z=8); g.text(x + 6, 320, 168, 120, what, size=17, color=INK, align="center", z=8)
        pill(g, x + 30, 456, 120, kp, col, size=17)
        if i: g.link(ids[i - 1], ids[i], 4, 12, GRAY, 4)
    banner(g, T("年份依時間排;藥丸是它在本章出現的單元", "ordered by date; the pill shows the unit where it appears"))
    d.done(g, "年份", "Dates", {0: ("1951 年 Robbins 與 Monro 的隨機逼近，給出步長條件；1986 年 Rumelhart、Hinton 與 Williams 系統化並普及反向傳播；2011 年 Duchi 等人的 AdaGrad。", "1951 Robbins and Monro's stochastic approximation gives the step conditions; 1986 Rumelhart, Hinton and Williams systematized and popularized backpropagation; 2011 Duchi and colleagues' AdaGrad."), 1: ("2012 年 Hinton 的 RMSProp，出自網路課程講義；2017 年 Keskar 等人觀察大批量與較尖銳的解有關，是有條件的；2021 年 Foret 等人的 SAM 與 Kwon 等人的 ASAM。", "2012 Hinton's RMSProp, from online course notes; 2017 Keskar and colleagues linked large batches with sharper solutions, conditionally; 2021 Foret and colleagues' SAM and Kwon and colleagues' ASAM.")})

    g = d.page("幾個轉折,一個接一個", "Turning points, one after another")
    tp = [(BLUE, BLUEL, T("只知方向,步多長", "only a direction: how far?"), T("學習率與 $L$-smooth、線搜索", "learning rate, $L$-smoothness, line search"), "KP01 → KP02"), (ORG, ORGL, T("全資料太貴", "all the data is too costly"), T("每步抽一筆或一小批", "draw one sample or a small batch"), "KP06 → KP07"), (GRN, GRNL, T("一個學習率不夠", "one learning rate is not enough"), T("逐座標自適應:AdaGrad → Adam", "per-coordinate: AdaGrad → Adam"), "KP09 → KP10"), (PUR, PURL, T("低損失還不夠", "a low loss is not enough"), T("看鄰域最壞損失:SAM", "the worst neighborhood loss: SAM"), "KP11 → KP12")]
    for i, (col, light, a, b_, kps) in enumerate(tp):
        g.beat = i + 1
        y = 120 + i * 125
        l = g.mark(); g.rect(60, y, 480, 100, fill=CARD, stroke=col, sw=3, radius=16, z=3); g.text(60, y + 24, 480, 60, a, size=24 if d.zh else 22, color=col, bold=True, align="center", z=8)
        r_ = g.mark(); g.rect(740, y, 480, 100, fill=light, stroke=col, sw=3, radius=16, z=3); g.text(740, y + 14, 480, 80, b_, size=22, color=INK, bold=True, align="center", z=8)
        g.link(l, r_, 4, 12, col, 5)
        pill(g, 560, y + 8, 160, kps, col, size=16)
    banner(g, T("每個轉折,都是上一個單元留下的問題", "each turn answers the question the previous unit left over"))
    d.done(g, "轉折", "Turning points", {0: ("第一個轉折：KP01 只知道方向，步長多長，KP02 的學習率與 L-smooth 回答。", "First turn: KP01 knows only a direction, so how far; KP02's learning rate and L-smoothness answer."), 1: ("第二個轉折：KP06 算得出梯度，但全資料太貴，KP07 每步抽一筆或一小批。", "Second turn: KP06 computes the gradient but all the data is too costly; KP07 draws one sample or a small batch."), 2: ("第三個轉折：KP09 之後一個學習率不夠，KP10 逐座標自適應。", "Third turn: after KP09 one learning rate is not enough; KP10 adapts per coordinate."), 3: ("第四個轉折：KP11 之後低損失還不夠，KP12 看鄰域最壞損失，也就是 SAM。", "Fourth turn: after KP11 a low loss is not enough; KP12 looks at the worst neighborhood loss, which is SAM.")})

    g = d.page("學完第三章,你能做到", "After chapter 3 you can")
    items = [(BLUE, BLUEL, T("手算梯度下降、動量與反向傳播,並選學習率", "compute gradient descent, momentum and backpropagation by hand, and choose a learning rate")), (GRN, GRNL, T("說明 SGD、小批次、AdaGrad、RMSProp、Adam 各解決什麼問題", "explain what SGD, minibatches, AdaGrad, RMSProp and Adam each solve")), (RED, REDL, T("說出收斂定理保證什麼、不保證什麼,並用驗證與測試證據評估模型", "say what convergence theorems guarantee and do not, and judge models with validation and test evidence"))]
    for i, (col, light, s_) in enumerate(items):
        g.beat = i + 1
        y = 120 + i * 150
        g.rect(50, y, 930, 120, fill=light, stroke=col, sw=3, radius=16, z=3)
        tick(g, 80, y + 38, 44, col, z=6)
        g.text(150, y + 14, 810, 96, s_, size=24, color=INK, bold=True, z=8)
    g.beat = 3
    mascot(g, 1050, 330, 0.62, "proud")
    banner(g, T("下一章:正則化與過擬合", "next chapter: regularization and overfitting"))
    d.done(g, "學完能做到", "What you can do", {0: ("最後看這章結束後你能做到的三件事。第一，手算梯度下降、動量與反向傳播，並選學習率。", "Finally three things you can do. First, compute gradient descent, momentum and backpropagation by hand and choose a learning rate."), 1: ("第二，說明隨機梯度、小批次、AdaGrad、RMSProp 與 Adam 各解決什麼問題。", "Second, explain what SGD, minibatches, AdaGrad, RMSProp and Adam each solve."), 2: ("第三，說出收斂定理保證什麼、不保證什麼，並用驗證與測試證據評估模型。下一章是正則化與過擬合。", "Third, say what convergence theorems guarantee and do not, and judge models with validation and test evidence. The next chapter is regularization and overfitting."), 3: ("", "")})
    return d.pages
