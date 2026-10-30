import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *

KP = "KP00"; N = 6
# (tag, colour, light, zh title, en title)
KPS = [("KP01", BLUE, BLUEL, "神經元加總訊號", "A neuron adds signals"),
       ("KP02", BLUE, BLUEL, "猜錯才改權重", "Fix weights when wrong"),
       ("KP03", BLUE, BLUEL, "標籤 ±1 的更新", "Updates with ±1 labels"),
       ("KP04", BLUE, BLUEL, "單層解不了 XOR", "One layer cannot do XOR"),
       ("KP05", ORG, ORGL, "多層與非線性", "Layers and nonlinearity"),
       ("KP06", ORG, ORGL, "活化函數", "Activation functions"),
       ("KP07", ORG, ORGL, "損失函數", "Loss functions"),
       ("KP08", GRN, GRNL, "ADALINE 與邏輯回歸", "ADALINE and logistic regression"),
       ("KP09", GRN, GRNL, "RBF 與 SOM", "RBF and SOM"),
       ("KP10", RED, REDL, "萬有逼近定理", "Universal approximation")]
GROUPS = [(0, 4, BLUE, BLUEL, "單一神經元", "one neuron"), (4, 7, ORG, ORGL, "多層與組件", "layers and parts"), (7, 9, GRN, GRNL, "連續學習與其他模型", "continuous learning, other models"), (9, 10, RED, REDL, "定理", "theorem")]
assert sum(b - a for a, b, *_ in GROUPS) == 10

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    # ============ 1: chapter question ============
    g = new(1, T("KP00 第2章:從一顆神經元到萬能逼近", "KP00 Chapter 2: from one neuron to any function"))
    g.beat = 1
    card(g, 50, 120, 1180, 120, GOLD)
    g.text(70, 150, 1140, 80, T("一顆會判斷的神經元，怎樣一步步變成能表示任何函數的網路？", "How does one neuron that makes a decision become a network that can represent any function?"), size=30, color=INK, bold=True, align="center", z=8)
    g.beat = 2
    stages = [(BLUE, BLUEL, T("單一神經元", "one neuron"), T("加總 · 過門檻 · 自己學", "sum · threshold · learn itself"), "KP01–KP04"),
              (ORG, ORGL, T("疊成網路", "stack into a network"), T("多層 · 活化函數 · 損失函數", "layers · activation · loss"), "KP05–KP07"),
              (GRN, GRNL, T("連續學習與其他模型", "continuous learning, other models"), "ADALINE · RBF · SOM", "KP08–KP10")]
    ids1 = []
    for i, (col, light, nm, sub, kps) in enumerate(stages):
        x = 50 + i * 410
        ids1.append(g.mark()); g.rect(x, 290, 360, 210, fill=light, stroke=col, sw=3, radius=16, z=3)
        g.text(x + 10, 300, 340, 70, nm, size=28, color=col, bold=True, align="center", z=8)
        g.text(x + 10, 376, 340, 70, sub, size=22, color=INK, bold=True, align="center", z=8)
        pill(g, x + 80, 452, 200, kps, col, size=18)
        if i: g.link(ids1[i - 1], ids1[i], 4, 12, GRAY, 5)
    g.beat = 3
    g.text(50, 530, 960, 90, T("第二章的十個單元，都在回答這個問題", "The ten units of this chapter all answer this question"), size=26, color=MUTED, bold=True, vcenter=True, z=8)
    mascot(g, 1090, 520, 0.4, "proud")
    banner(g, T("先看一顆神經元，再疊成網路，最後問「能表示什麼」", "one neuron first, then a network, finally “what can it represent?”"))
    pages.append((g, T("本章的一個問題", "The chapter's one question"), {
        0: T("教材程度是大學生，全章三小時。先看整章只問一個問題。", "Level: undergraduate; the chapter takes three hours. First, the one question the whole chapter asks."),
        1: T("一顆會判斷的神經元，怎樣一步步變成能表示任何函數的網路？", "How does one neuron that makes a decision become a network that can represent any function?"),
        2: T("回答分三段：先看一顆神經元，從加總、過門檻到自己學；再疊成網路，講多層、活化函數與損失函數；最後是連續學習的模型與萬有逼近定理。", "Three parts: one neuron first, from summing and thresholding to learning itself; then stacking into a network with layers, activations and losses; finally continuous-learning models and the universal approximation theorem."),
        3: T("十個單元就是照這三段排的。", "The ten units follow these parts.")}))

    # ============ 2: roadmap ============
    g = new(2, T("十個單元，由左到右、再繞回來", "Ten units: left to right, then back"))
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
    for k, (a, b_, col, light, nz, ne) in enumerate(GROUPS):
        x = 50 + k * 295
        g.rect(x, 520, 270, 56, fill=light, stroke=col, sw=3, radius=28, z=3); g.text(x + 6, 520, 258, 56, nz if zh else ne, size=20, color=col, bold=True, align="center", vcenter=True, z=8) if zh else g.text(x + 6, 524, 258, 52, ne, size=18, color=col, bold=True, align="center", z=8)
    banner(g, T("單一神經元 → 多層與組件 → 連續學習 → 定理", "one neuron → layers and parts → continuous learning → theorem"))
    pages.append((g, T("十個單元的路線", "The ten-unit route"), {
        0: T("這是全章的路線。第一排：KP01 神經元加總訊號，KP02、KP03 感知器的學習規則，KP04 單層解不了 XOR，KP05 多層與非線性。", "This is the route of the chapter. Row one: KP01 a neuron adds signals, KP02 and KP03 the Perceptron's learning rule, KP04 one layer cannot do XOR, KP05 layers and nonlinearity."),
        1: T("第二排從右邊接回來：KP06 活化函數，KP07 損失函數，KP08 ADALINE 與邏輯回歸，KP09 RBF 與 SOM，KP10 萬有逼近定理。", "Row two comes back from the right: KP06 activation functions, KP07 loss functions, KP08 ADALINE and logistic regression, KP09 RBF and SOM, KP10 universal approximation."),
        2: T("顏色代表四段分組：單一神經元、多層與組件、連續學習與其他模型、定理。", "Colors mark four groups: one neuron, layers and parts, continuous learning and other models, the theorem."),
        3: T("", "")}))

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
    for a, b_, col, light, nz, ne in GROUPS:
        x0 = 53 + a * 118; x1 = 53 + b_ * 118 - 6
        g.line([(x0, 290), (x0, 306), (x1, 306), (x1, 290)], col, sw=4, z=5)
        g.text(x0 - 10, 312, x1 - x0 + 20, 56, nz if zh else ne, size=20 if zh else 18, color=col, bold=True, align="center", z=8)
        g.text(x0, 370, x1 - x0, 36, f"{(b_ - a) * 15} " + T("分鐘", "min"), size=22, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.beat = 3
    g.rect(150, 450, 980, 150, fill=CARD, stroke=GOLD, sw=3, radius=16, z=3, shadowColor="rgba(15,23,42,0.12)", shadowBlur=14, shadowOffsetY=4)
    g.text(170, 460, 940, 60, r"$10\times 15 = 150$ " + T("分鐘", "minutes"), size=36, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.text(170, 528, 940, 60, T("全章 3 小時，其中淨講授 150 分鐘", "3 hours for the chapter, 150 minutes of pure teaching"), size=24, color=MUTED, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("單一神經元佔最多:4 個單元、60 分鐘", "one neuron takes the most: 4 units, 60 minutes"))
    pages.append((g, T("時間分配", "Time plan"), {
        0: T("全章三小時，每個單元配置約十五分鐘。", "The chapter has three hours, with about fifteen minutes per unit."),
        1: T("分組後：單一神經元六十分鐘、多層與組件四十五分鐘、連續學習與其他模型三十分鐘、定理十五分鐘。", "By group: one neuron 60 minutes, layers and parts 45, continuous learning and other models 30, the theorem 15."),
        2: T("合計淨講授一百五十分鐘。", "Pure teaching totals 150 minutes."),
        3: T("", "")}))

    # ============ 4: timeline ============
    g = new(4, T("一路上的幾個年份", "Some dates along the way"))
    ev = [("1943", T("McCulloch 與 Pitts", "McCulloch and Pitts"), T("加總過門檻的神經元", "the sum-and-threshold neuron"), "KP01", BLUE, BLUEL),
          ("1958", T("Rosenblatt", "Rosenblatt"), T("Perceptron 能自己學", "the Perceptron learns"), "KP02", BLUE, BLUEL),
          ("1960", T("Widrow 與 Hoff", "Widrow and Hoff"), T("ADALINE:連續的 LMS 損失", "ADALINE: the continuous LMS loss"), "KP08", GRN, GRNL),
          ("1969", T("Minsky 與 Papert", "Minsky and Papert"), T("分析單層限制，XOR 成反例", "analyze the single-layer limit; XOR the counterexample"), "KP04", BLUE, BLUEL),
          ("1982", "Kohonen", T("SOM:不用標籤", "SOM: no labels"), "KP09", GRN, GRNL),
          ("1988", "Broomhead " + T("與", "and ") + "Lowe", T("RBF 網路", "the RBF network"), "KP09", GRN, GRNL)]
    ids = []
    for i, (yr, who, what, kp, col, light) in enumerate(ev):
        g.beat = 1 if i < 3 else 2
        x = 50 + i * 200
        ids.append(g.mark()); g.rect(x, 150, 180, 360, fill=light, stroke=col, sw=3, radius=16, z=3)
        g.text(x, 160, 180, 56, yr, size=32, color=col, bold=True, align="center", z=8); g.text(x + 6, 226, 168, 70, who, size=20, color=INK, bold=True, align="center", z=8); g.text(x + 6, 306, 168, 130, what, size=19, color=INK, align="center", z=8)
        pill(g, x + 40, 456, 100, kp, col, size=18)
        if i: g.link(ids[i - 1], ids[i], 4, 12, GRAY, 4)
    g.beat = 3
    banner(g, T("年份依時間排;「對應單元」指它在本章出現的地方", "ordered by date; the pill shows the unit where it appears in this chapter"))
    pages.append((g, T("年份", "Dates"), {
        0: T("1943 年 McCulloch 與 Pitts 的神經元，1958 年 Rosenblatt 的 Perceptron 能自己學，1960 年 Widrow 與 Hoff 的 ADALINE 用連續的 LMS 損失。", "1943 the McCulloch–Pitts neuron; 1958 Rosenblatt's Perceptron learns; 1960 Widrow and Hoff's ADALINE with the continuous LMS loss."),
        1: T("1969 年 Minsky 與 Papert 分析單層限制，XOR 成為反例；1982 年 Kohonen 的 SOM 不用標籤；1988 年 Broomhead 與 Lowe 的 RBF 網路。", "1969 Minsky and Papert analyze the single-layer limit with XOR as the counterexample; 1982 Kohonen's SOM needs no labels; 1988 Broomhead and Lowe's RBF network."),
        2: T("", ""), 3: T("", "")}))

    # ============ 5: three turning points ============
    g = new(5, T("三個轉折，一個接一個", "Three turning points, one after another"))
    tp = [(BLUE, BLUEL, T("權重要人算", "weights are set by hand"), T("讓它自己學", "let it learn itself"), "KP01 → KP02"),
          (ORG, ORGL, T("單層做不到 XOR", "one layer cannot do XOR"), T("多層加非線性", "layers with nonlinearity"), "KP04 → KP05"),
          (GRN, GRNL, T("只問對錯，不能一步步走", "only right or wrong: no steady steps"), T("改用連續損失", "use a continuous loss"), "KP07 → KP08")]
    for i, (col, light, a, b_, kps) in enumerate(tp):
        g.beat = i + 1
        y = 130 + i * 150
        l = g.mark(); g.rect(60, y, 480, 120, fill=CARD, stroke=col, sw=3, radius=16, z=3); g.text(60, y + 32, 480, 70, a, size=24 if zh else 22, color=col, bold=True, align="center", z=8)
        r_ = g.mark(); g.rect(740, y, 480, 120, fill=light, stroke=col, sw=3, radius=16, z=3); g.text(740, y + 32, 480, 70, b_, size=24, color=INK, bold=True, align="center", z=8)
        g.link(l, r_, 4, 12, col, 5)
        pill(g, 560, y + 8, 160, kps, col, size=17)
    banner(g, T("每個轉折，都是上一個單元留下的問題", "each turn answers the question the previous unit left over"))
    pages.append((g, T("三個轉折", "Three turns"), {
        0: T("第一個轉折：KP01 的神經元權重要人事先算好，KP02 讓它自己學。", "First turn: KP01's neuron weights must be worked out by a person; KP02 lets it learn them."),
        1: T("第二個轉折：KP04 單層做不到 XOR，KP05 用多層加非線性解決。", "Second turn: KP04 shows one layer cannot do XOR; KP05 solves it with layers and nonlinearity."),
        2: T("第三個轉折：KP07 的損失與只問對錯的做法不同；KP08 的 ADALINE 與邏輯回歸改用連續損失，才能一步步往下走。", "Third turn: KP07's losses differ from the right-or-wrong approach; KP08's ADALINE and logistic regression use continuous losses, allowing steady downhill steps."),
        3: T("", "")}))

    # ============ 6: can do ============
    g = new(6, T("學完第二章，你能做到", "After chapter 2 you can"))
    items = [(BLUE, BLUEL, T("手算一顆神經元，並用 Perceptron 規則更新權重", "compute a neuron by hand and update weights with the Perceptron rule")),
             (ORG, ORGL, T("說明為什麼需要非線性，並為任務選活化函數與損失函數", "explain why nonlinearity is needed and choose an activation and a loss for a task")),
             (RED, REDL, T("說出萬有逼近定理保證什麼、不保證什麼", "say what the universal approximation theorem guarantees and what it does not"))]
    for i, (col, light, s_) in enumerate(items):
        g.beat = i + 1
        y = 120 + i * 150
        g.rect(50, y, 930, 120, fill=light, stroke=col, sw=3, radius=16, z=3)
        tick(g, 80, y + 38, 44, col, z=6)
        g.text(150, y + 20, 810, 84, s_, size=26, color=INK, bold=True, z=8)
    g.beat = 3
    mascot(g, 1050, 330, 0.62, "proud")
    banner(g, T("下一章:這些權重怎麼自己學?反向傳播與梯度下降", "next chapter: how do these weights learn themselves? backpropagation and gradient descent"))
    pages.append((g, T("學完能做到", "What you can do"), {
        0: T("最後看這章結束後，你應該能做到三件事。第一，手算一顆神經元，並用 Perceptron 規則更新權重。", "Finally, three things you should be able to do. First, compute a neuron by hand and update weights with the Perceptron rule."),
        1: T("第二，說明為什麼需要非線性，並為任務選活化函數與損失函數。", "Second, explain why nonlinearity is needed and choose an activation and a loss for a task."),
        2: T("第三，說出萬有逼近定理保證什麼、不保證什麼。下一章是反向傳播與梯度下降。", "Third, say what the universal approximation theorem guarantees and what it does not. The next chapter is backpropagation and gradient descent."),
        3: T("", "")}))
    return pages
