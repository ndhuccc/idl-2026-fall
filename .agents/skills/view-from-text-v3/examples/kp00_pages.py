"""KP00 pages (chapter roadmap) designed in HTML (view-from-text-v3, route A). pageN(lang) -> body HTML of a 1160x480 figure."""
import os, sys
SK = "/home/user/pr-2026-fall/.agents/skills"
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, SK + "/view-from-text-v3/examples"); sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, HERE)
sys.path.insert(0, "/tmp/claude-0/-home-user/45feda6c-d91f-5d4d-a7e1-176e77f608e6/scratchpad/agentC/kp02")
from kp02_pages import *
import kp00_nums as N
W, H = 1160, 480
KPS = [("KP01", "blue"), ("KP02", "teal"), ("KP03", "violet"), ("KP04", "orange"), ("KP05", "green")]
def kname(lang, i):
    zh = ["一條線不夠", "先編碼再分類", "編碼與極限", "逐碼偵測", "從點長網路"]; en = ["One line is not enough", "Encode first, then classify", "Codes and limits", "Detect code by code", "Grow a network from points"]
    return (zh if lang == "zh" else en)[i]
QA = [  # question, answer, left over  (zh, en)
 (("XOR 有任何一條線分得開嗎？", "Does any single line separate XOR?"), ("兩個 A 的分數和必為正，兩個 B 的分數和必不大於 0，卻是同一個式子，矛盾。", "The two A scores must add up to a positive number and the two B scores to at most 0, yet it is the same expression: a contradiction."), ("要改的是邊界的組合，例如兩條線夾出一條帶。", "What must change is the combination of boundaries, for example two lines that make a strip.")),
 (("兩個判斷怎麼用網路做出來？", "How do we build the two judgments as a network?"), ("兩條線各給一個位元，四個點只得三種編碼，輸出層一條線分開 A 與 B。", "Two lines give one bit each, the four points get only three codes, and one output line separates A from B."), ("線更多時，輸出層還夠用嗎？", "With more lines, is the output layer still enough?")),
 (("更多線之後，一個切面分得開所有標籤嗎？", "With more lines, can one cutting plane separate every labeling?"), ("有些分得開，有些不行（底面 A、B 沿對角交錯）。", "Some labelings yes, some no (A and B alternate along the diagonal of the base)."), ("切不開的標籤要再加一層。", "A labeling that cannot be cut needs another layer.")),
 (("怎樣在編碼之上再加一層？", "How do we add another layer on top of the codes?"), ("偵測器分數 = 0.5 − 不吻合的位元數，輸出層用「或」合併。", "Detector score = 0.5 − the number of mismatched bits, and the output layer merges with OR."), ("只證明夠用，網路要怎麼從點長出來？", "It only proves sufficiency; how does a network grow from the points?")),
 (("只有帶標籤的點，網路怎麼長出來？", "With only labeled points, how does a network grow?"), ("主要節點分一次，輔助節點處理混雜群，增加位元純化。", "A primary node splits once, auxiliary nodes handle mixed groups, and added bits purify them."), ("全對只是這批點；先固定架構再學權重。", "All right only on this batch of points; fix the architecture first, then learn the weights.")),
]
def qa_row(x, y, w, h, i, lang, size=16):
    t = tr(lang); role = KPS[i][1]; q, a, l = QA[i]; o = card(x, y, w, h, role, .14) + head(x, y, 150, role, f"{KPS[i][0]} {kname(lang, i)}" if False else KPS[i][0], 30, 17)
    c = (w - 160) // 3
    for j, (lab, txt_, rr) in enumerate(((t("問", "Question"), q, "grey"), (t("答", "Answer"), a, "green"), (t("留下", "Left over"), l, "yellow"))):
        xx = x + 160 + j * c; o += chip(xx, y + 6, 110, rr, lab, False, 26, 16) + txt(xx, y + 42, c - 40, txt_[0 if lang == "zh" else 1], size, TXT, False, "left", h - 50)
    o += txt(x + 6, y + 40, 146, kname(lang, i), 18, TXT, True, "left", 80)
    return o
# ------------------------------------------------ 1: the one question
def page1(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 560, 480, "blue", t("起點是 XOR：兩個輸入不同時是 A", "The starting point is XOR: A when the two inputs differ"))
    o += chip(14, 50, 532, "yellow", t("一條線分不開的標籤，網路怎樣分開？", "One line cannot separate them: how can a network?"), True, 48, 20)
    o += txt(14, 114, 100, "x₁", 17, MUT, True, "center", 26) + txt(120, 114, 100, "x₂", 17, MUT, True, "center", 26) + txt(250, 114, 200, t("類別", "class"), 17, MUT, True, "center", 26)
    for i, p in enumerate(N.PTS):
        c = N.xor_class(*p); y0 = 146 + i * 54; role = "blue" if c == "A" else "orange"; o += card(14, y0, 532, 46, role, .18)
        o += txt(14, y0 + 9, 100, str(p[0]), 18, TXT, True, "center", 28) + txt(120, y0 + 9, 100, str(p[1]), 18, TXT, True, "center", 28) + chip(300, y0 + 7, 100, role, c, True, 32, 18)
    o += named(14, 372, 532, t("互斥或 XOR：兩個輸入不同才是 A，相同是 B。", "Exclusive OR (XOR): A only if the two inputs differ, B if they are equal."), "term:" + t("互斥或", "Exclusive OR"), 16, TXT, 48)
    o += txt(14, 422, 532, t("線性分類器（第三章）用一條線把兩類分開；分數為正判 A，否則判 B。", "A linear classifier (chapter 3) separates two classes with one line; a positive score means A, otherwise B."), 16, MUT, False, "center", 48)
    o += cardH(576, 0, 584, 480, "orange", t("兩個 A 在一條對角線，兩個 B 在另一條", "The two A points on one diagonal, the two B points on the other"))
    ax = Axes(640, 56, 300, 300, XR, XR)
    s = axes_svg(ax, [0, 1], [0, 1], "x₁", "x₂") + f'<line x1="{ax.X(0)}" y1="{ax.Y(1)}" x2="{ax.X(1)}" y2="{ax.Y(0)}" stroke="#9cc3ea" stroke-width="4" stroke-dasharray="10 7"/><line x1="{ax.X(0)}" y1="{ax.Y(0)}" x2="{ax.X(1)}" y2="{ax.Y(1)}" stroke="#f0a98b" stroke-width="4" stroke-dasharray="10 7"/>'
    for p in N.PTS: s += mark(ax, p[0], p[1], N.xor_class(*p), 14)
    o += svg(0, 0, W, H, s)
    o += txt(960, 100, 190, t("任何一條直線，都沒辦法把兩個 A 放一側、兩個 B 放另一側", "No straight line can put the two A points on one side and the two B points on the other"), 18, TXT, True, "left", 220)
    o += txt(586, 410, 564, t("KP01 用分數相加證明這件事；之後四個知識點一步步補上網路。", "KP01 proves this by adding scores; the next four KPs build the network step by step."), 16, MUT, False, "center", 48)
    return o
# ------------------------------------------------ 2: the roadmap (also the lecture figure)
def page2(lang):
    t = tr(lang); o = ""; s = ""
    leave = [QA[i][2] for i in range(5)]
    for i, (k, role) in enumerate(KPS):
        x0 = i * 232; o += card(x0, 0, 214, 180, role, .16) + head(x0, 0, 214, role, k, 34, 18) + txt(x0 + 6, 44, 202, kname(lang, i), 19, TXT, True, "center", 56)
        o += txt(x0 + 6, 100, 202, QA[i][0][0 if lang == "zh" else 1], 16, MUT, False, "center", 72)
        o += card(x0, 214, 214, 196, "yellow", .14) + txt(x0 + 6, 222, 202, t("留下的問題", "Question left over"), 17, MUT, True, "center", 26) + txt(x0 + 6, 256, 202, leave[i][0 if lang == "zh" else 1], 18, TXT, True, "center", 140)
        s += arrow(x0 + 107, 182, x0 + 107, 212, "#b7b0a5", 3, 11)
        if i < 4: s += arrow(x0 + 216, 214 + 98, x0 + 232, 214 + 98, "#e0c85f", 3, 10)
    o += svg(0, 0, W, H, s)
    o += card(0, 426, 1160, 46, "yellow", .14) + txt(10, 434, 1140, t("每一個知識點都從上一個留下的問題出發；由左到右讀。", "Every KP starts from the question the previous one left over; read from left to right."), 18, TXT, True, "center", 30)
    return o
# ------------------------------------------------ 3: five words
def page3(lang):
    t = tr(lang); o = ""
    words = [(("位元", "Bit"), ("只有 0 或 1 的數；一條線判斷「點在正側嗎」，答案記成一個位元。", "A number that is only 0 or 1; a line asks “is the point on the positive side?” and the answer is recorded as one bit."), "blue"),
             (("隱藏層", "Hidden layer"), ("夾在輸入與最後輸出之間的一排計算單元；每個單元問一個小問題。", "The row of computing units between the inputs and the final output; each unit asks one small question."), "teal"),
             (("編碼", "Code"), ("把一個點的所有位元連起來，例如 (1, 0)，像替點貼上「位置標籤」。", "All bits of a point written together, for example (1, 0), like a position label attached to the point."), "violet"),
             (("虛擬碼", "Unrealized code"), ("寫得出來、卻沒有任何輸入點得到的編碼。", "A code that can be written down but that no input point gets."), "orange"),
             (("偵測器", "Detector"), ("一個單元，只在輸入的編碼剛好等於某個目標碼時才回答「是」。", "A unit that answers “yes” only when the input code equals one particular target code."), "green")]
    pos = [(0, 0), (390, 0), (780, 0), (0, 240), (390, 240)]
    for (nm, df, role), (x0, y0) in zip(words, pos):
        o += card(x0, y0, 380, 226, role, .14) + head(x0, y0, 380, role, t(*nm), 36, 20) + named(x0 + 12, y0 + 50, 320, t(*df), "term:" + t(*nm), 19, TXT, 160)
    o += card(780, 240, 380, 226, "yellow", .14) + txt(790, 252, 360, t("用白話先說五個詞；<br>後面的 KP 會逐一展開。<br>位元與隱藏層從 KP02 起，<br>編碼與虛擬碼在 KP03，<br>偵測器在 KP04。", "Five words in plain language first;<br>later KPs unfold each one.<br>Bit and hidden layer: KP02;<br>code and unrealized code: KP03;<br>detector: KP04."), 18, TXT, True, "center", 190)
    return o
# ------------------------------------------------ 4,5,6: Q / A / left over
def rows_page(idx, extra=None):
    def pg(lang):
        t = tr(lang); o = ""; n = len(idx); hh = 214 if n == 2 else 150
        for r, i in enumerate(idx): o += qa_row(0, r * (hh + 14), 1160, hh, i, lang, 20 if n == 2 else 17)
        if extra: o += extra(lang)
        return o
    return pg
def core_figs(lang):
    t = tr(lang); o = card(0, 164, 1160, 306, "grey", .08) + txt(10, 170, 1140, t("五個核心圖：各對應一個知識點", "Five core figures, one per KP"), 17, MUT, True, "center", 26)
    items = [(t("兩條平行線夾出一條帶，A 在帶內", "Two parallel lines make a strip; A is inside"), 0), (t("把它做成兩層網路，選出同一條帶", "Built as a two-layer network, it selects the same strip"), 1), (t("立方體與一個切面的極限", "The cube and the limit of one cutting plane"), 2), (t("在編碼上再加一層，圈出 A 區", "One more layer on the codes outlines the A region"), 3), (t("沒有現成網路時，新編碼平面長出來", "With no network given, the new code plane grows"), 4)]
    for (txt_, i) in items:
        x0 = i * 232; role = KPS[i][1]; o += card(x0 + 4, 206, 222, 252, role, .16) + head(x0 + 4, 206, 222, role, KPS[i][0], 32, 18) + txt(x0 + 10, 252, 210, kname(lang, i), 19, TXT, True, "center", 60) + txt(x0 + 10, 320, 210, txt_, 17, MUT, False, "center", 120)
    return o
page4 = rows_page([0, 1]); page5 = rows_page([2, 3])
def page6(lang):
    t = tr(lang); o = qa_row(0, 0, 1160, 150, 4, lang, 17) + core_figs(lang)
    return o
# ------------------------------------------------ 7: four stages
def page7(lang):
    t = tr(lang); o = ""; s = ""
    st = [(t("輸入", "Input"), t("點的座標 (x₁, x₂)", "the coordinates (x₁, x₂)"), "grey"), (t("隱藏編碼", "Hidden code"), t("每條線一個位元，連成編碼", "one bit per line, joined into a code"), "teal"), (t("偵測器", "Detectors"), t("只在編碼等於目標碼時回答「是」", "answer “yes” only for the target code"), "orange"), (t("輸出", "Output"), t("用「或」合併偵測器，判 A 或 B", "merge the detectors with OR: A or B"), "blue")]
    for i, (a, b, role) in enumerate(st):
        x0 = i * 290; o += card(x0, 0, 262, 250, role, .16) + head(x0, 0, 262, role, a, 38, 20) + txt(x0 + 8, 60, 246, b, 20, TXT, False, "center", 110) + chip(x0 + 71, 190, 120, role, t(f"階段 {i+1}", f"stage {i+1}"), True, 32, 16)
        if i < 3: s += arrow(x0 + 264, 125, x0 + 288, 125, "#b7b0a5", 3, 10)
    o += svg(0, 0, W, H, s)
    o += txt(10, 262, 1140, t("每一階段只接上一階段的結果。", "Each stage takes only the result of the previous stage."), 18, TXT, True, "center", 30)
    o += card(0, 308, 1160, 160, "red", .16) + txt(10, 316, 1140, t("常見迷思：構造出夠用的網路，就表示權重已學到、或這是最小的網路。", "Common myth: having built a network that suffices means the weights are learned, or that it is the smallest network."), 19, TXT, True, "center", 36) + txt(10, 372, 1140, t("構造只證明夠用。", "A construction only proves sufficiency."), 22, "#f0a98b", True, "center", 40)
    return o
# ------------------------------------------------ 8: four reminders + next chapter
def page8(lang):
    t = tr(lang); o = ""
    rem = [(t("固定編碼上，一個切面有極限。", "On fixed codes, one cutting plane has a limit."), "blue"), (t("矛盾標籤不可能全對。", "Contradictory labels cannot all be right."), "teal"), (t("全對只是這批點全對。", "All right means all right on this batch of points."), "violet"), (t("構造證明夠用，不代表權重已學到或網路最小。", "A construction proves sufficiency, not that the weights are learned or the network is smallest."), "orange")]
    for i, (a, role) in enumerate(rem):
        x0 = (i % 2) * 580; y0 = (i // 2) * 176; o += card(x0, y0, 570, 164, role, .16) + chip(x0 + 12, y0 + 12, 60, role, str(i + 1), True, 40, 22) + txt(x0 + 84, y0 + 14, 474, a, 24, TXT, True, "left", 140)
    o += card(0, 364, 1160, 100, "yellow", .16) + txt(10, 372, 1140, t("下一章：先固定網路架構，再用代價函數學權重（反向傳播，KP06 起）。", "Next chapter: fix the network architecture first, then learn the weights with a cost function (backpropagation, from KP06)."), 20, TXT, True, "center", 84)
    return o
PAGES_DESIGN = [(f"p{i}", globals()[f"page{i}"]) for i in range(1, 9)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:]
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp00_" + fid, fn, W, H, out)
