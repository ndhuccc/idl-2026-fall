import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE; SK = "/home/user/pr-2026-fall/.agents/skills"
for p in ("view-from-text-v3/tools", "view-from-text-v2/tools", "view-from-text-v2/examples"): sys.path.insert(0, os.path.join(SK, p))
sys.path.insert(0, HERE)
import kp00_pages as KP
from htmlkit import native
JD = os.path.join(HERE, "out")
UNIT_ZH, UNIT_EN = "第四章路線圖", "Ch. 4 roadmap"
TITLE_ZH = "圖型識別 第四章 KP00：第四章路線圖（大三）"; TITLE_EN = "Pattern Recognition, Chapter 4, KP00: the chapter 4 roadmap (undergraduate)"
NOTE_ZH = "教材程度：大學三年級（undergrad）。依講義 lec05-kp00 以 view-from-text-v3 製作：先規劃頁面，每頁以一張原生資訊圖（HTML 設計後轉成原生節點）為主，名詞在同一頁圖內定義。共 8 頁；本單元是導覽，只做回顧與白話定義，不新增內容。"
NOTE_EN = "Level: undergraduate (year 3). Made from the lecture lec05-kp00 with view-from-text-v3: pages were planned first and each page is mainly one native infographic (designed in HTML and converted to native nodes), with the terms defined on the same page. 8 pages; this unit is an overview that only reviews and defines terms in plain language, adding no new content."
def fig(n, alt_zh, alt_en):
    return lambda: native(f"kp00_p{n}", f"lec05-kp00-p{n}", KP.W, KP.H, JD, alt_en if os.environ.get("KP_LANG") == "en" else alt_zh)
P = []
NXT = {1: ("路線圖", "the roadmap"), 2: ("五個詞", "the five words"), 3: ("KP01 與 KP02", "KP01 and KP02"), 4: ("KP03 與 KP04", "KP03 and KP04"), 5: ("KP05 與核心圖", "KP05 and the core figures"), 6: ("整個網路", "the whole network"), 7: ("四個提醒", "the four reminders")}
PREV = {2: ("本章的問題", "the chapter's question"), 3: ("路線圖", "the roadmap"), 4: ("五個詞", "the five words"), 5: ("KP01 與 KP02", "KP01 and KP02"), 6: ("KP03 與 KP04", "KP03 and KP04"), 7: ("KP05 與核心圖", "KP05 and the core figures"), 8: ("整個網路", "the whole network")}
def add(n, lab, title, bridge, notes, alt):
    zh, en = notes
    if n in PREV: zh = f"承接上一頁的「{PREV[n][0]}」。" + zh; en = f"Following the previous page on {PREV[n][1]}. " + en
    if n == 8: zh += "到這裡這個單元就結束了；之後請接著讀下一單元。"; en += " That ends this unit; next, move on to the next unit."
    if n in NXT: zh += f"接下來的下一頁是「{NXT[n][0]}」。"; en += f" Next, the next page covers {NXT[n][1]}."
    P.append(dict(lab=lab, title=title, bridge=bridge, notes=(zh, en), layout="freeform", spec=fig(n, *alt)))
add(1, ("本章的問題", "The question"),
    ("第四章只問一件事：XOR 這種一條線分不開的標籤，網路怎樣分開？", "Chapter 4 asks one question: how can a network separate labels, like XOR, that one line cannot?"),
    ("第三章的線性分類器用一條線決定類別。標籤交錯時，這條線還夠用嗎？", "The linear classifier of chapter 3 decides the class with one line. Is one line still enough when the labels are interleaved?"),
    ("這是第四章路線圖的第一頁。先備三件事：線性分類器、分數、座標與表格。線性分類器用一條線把兩類分開，分數為正判 A、否則判 B。本章的起點是互斥或 XOR：兩個輸入相同時是 B、不同時是 A，(0,0) 與 (1,1) 是 B，(0,1) 與 (1,0) 是 A。兩個 A 在一條對角線上，兩個 B 在另一條對角線上，任何一條直線都沒辦法把兩個 A 放一側、兩個 B 放另一側。本章只問一件事：一條線分不開的標籤，網路怎樣分開？",
     "This is the first page of the chapter 4 roadmap. Three prerequisites: linear classifiers, scores, and coordinates with small tables. A linear classifier separates two classes with one line, calling A when the score is positive and B otherwise. The starting point of the chapter is exclusive OR, XOR: B when the two inputs are equal and A when they differ, so (0,0) and (1,1) are B and (0,1) and (1,0) are A. The two A points lie on one diagonal and the two B points on the other, and no straight line can put the two A points on one side and the two B points on the other. The chapter asks one question: how does a network separate labels that one line cannot?"),
    ("XOR table (0,0) B, (0,1) A, (1,0) A, (1,1) B and the plane with the A points on one diagonal and the B points on the other; no line separates them.",) * 2)
add(2, ("路線圖", "Roadmap"),
    ("路線圖：五個知識點一個接一個，每個都從上一個留下的問題出發", "Roadmap: five KPs one after another, each starting from the question the previous one left over"),
    ("網路怎樣分開一條線分不開的標籤？先看五個知識點怎麼串成一條路。", "How can a network separate labels that one line cannot? How do five KPs chain into one route?"),
    ("這一頁是全章的路線圖。本章有五個知識點，一個接著一個，每一個都從上一個留下的問題出發。KP01 一條線不夠，問 XOR 有任何一條線分得開嗎，留下要改邊界的組合。KP02 先編碼再分類，留下線更多時輸出層還夠用嗎。KP03 編碼與極限，留下切不開的標籤要再加一層。KP04 逐碼偵測，留下只證明夠用、網路要怎麼從點長出來。KP05 從點長網路，留下全對只是這批點，先固定架構再學權重。",
     "This page is the roadmap of the whole chapter. The chapter has five KPs, one after another, and each starts from the question the previous one left over. KP01, one line is not enough, asks whether any line separates XOR and leaves the need to change the combination of boundaries. KP02, encode first and then classify, leaves the question whether the output layer is still enough with more lines. KP03, codes and limits, leaves the point that a labeling that cannot be cut needs another layer. KP04, detecting code by code, leaves the fact that it only proves sufficiency and the question of how a network grows from points. KP05, growing a network from points, leaves the point that all right holds only for this batch, so fix the architecture first and then learn the weights."),
    ("Roadmap of KP01 to KP05 in a row; under each KP the question it leaves over, which the next KP takes up.",) * 2)
add(3, ("五個詞", "Five words"),
    ("先用白話說五個詞：位元、隱藏層、編碼、虛擬碼、偵測器", "Five words first, in plain language: bit, hidden layer, code, unrealized code, detector"),
    ("五個知識點一個接一個，後面的知識點會用到哪些詞？", "Five KPs chain one after another. Which words do the later KPs rely on?"),
    ("這一頁用白話先說五個詞，後面的知識點會逐一展開。位元：只有 0 或 1 的數；一條線判斷點在正側嗎，答案記成一個位元。隱藏層：夾在輸入與最後輸出之間的一排計算單元，每個單元問一個小問題。編碼：把一個點的所有位元連起來，例如 (1, 0)，像替點貼上位置標籤。虛擬碼：寫得出來、卻沒有任何輸入點得到的編碼。偵測器：一個單元，只在輸入的編碼剛好等於某個目標碼時才回答是。",
     "This page gives five words in plain language first; later KPs unfold each one. Bit: a number that is only 0 or 1; a line asks whether the point is on the positive side and the answer is recorded as one bit. Hidden layer: the row of computing units between the inputs and the final output, each asking one small question. Code: all bits of a point written together, for example (1, 0), like a position label attached to the point. Unrealized code: a code that can be written down but that no input point gets. Detector: a unit that answers yes only when the input code equals one particular target code."),
    ("Five definition cards: bit, hidden layer, code, unrealized code, detector.",) * 2)
add(4, ("KP01 與 KP02", "KP01 and KP02"),
    ("KP01 證明一條線不夠，KP02 用兩條線的位元編碼把 XOR 分開", "KP01 shows one line is not enough; KP02 separates XOR with the bit codes of two lines"),
    ("位元、隱藏層與編碼是後面的基本詞。KP01 與 KP02 各自問什麼、答什麼、留下什麼？", "Bit, hidden layer and code are the basic words from here on. What do KP01 and KP02 each ask, answer and leave over?"),
    ("這一頁是 KP01 與 KP02 的問、答、留下。KP01 問 XOR 有任何一條線分得開嗎；答：兩個 A 的分數和必為正，兩個 B 的分數和必不大於 0，卻是同一個式子，矛盾；留下：要改的是邊界的組合，例如兩條線夾出一條帶。KP02 問兩個判斷怎麼用網路做出來；答：兩條線各給一個位元，四個點只得三種編碼，輸出層一條線分開 A 與 B；留下：線更多時，輸出層還夠用嗎？",
     "This page gives the question, answer and what is left over for KP01 and KP02. KP01 asks whether any line separates XOR; the answer is that the two A scores must add up to a positive number and the two B scores to at most 0, yet it is the same expression, a contradiction; left over: what must change is the combination of boundaries, for example two lines that make a strip. KP02 asks how to build the two judgments as a network; the answer is that two lines give one bit each, the four points get only three codes, and one output line separates A from B; left over: with more lines, is the output layer still enough?"),
    ("Question, answer and left-over rows for KP01 and KP02.",) * 2)
add(5, ("KP03 與 KP04", "KP03 and KP04"),
    ("KP03 說明一個切面有極限，KP04 用偵測器在編碼上再加一層", "KP03 shows one cutting plane has a limit; KP04 adds a layer of detectors on the codes"),
    ("KP01 證明一條線不夠，KP02 用兩條線的位元編碼把 XOR 分開，卻留下線更多時輸出層還夠用嗎。KP03 與 KP04 怎麼接下去？", "KP01 shows one line is not enough and KP02 separates XOR with the bit codes of two lines, but leaves open whether the output layer still suffices with more lines. How do KP03 and KP04 continue?"),
    ("這一頁是 KP03 與 KP04 的問、答、留下。KP03 問更多線之後，一個切面分得開所有標籤嗎；答：有些分得開，有些不行，例如底面 A、B 沿對角交錯；留下：切不開的標籤要再加一層。KP04 問怎樣在編碼之上再加一層；答：偵測器分數等於 0.5 減不吻合的位元數，輸出層用或合併；留下：只證明夠用，網路要怎麼從點長出來？",
     "This page gives the question, answer and what is left over for KP03 and KP04. KP03 asks whether one cutting plane can separate every labeling once there are more lines; the answer is that some can and some cannot, for example when A and B alternate along the diagonal of the base; left over: a labeling that cannot be cut needs another layer. KP04 asks how to add another layer on top of the codes; the answer is that the detector score equals 0.5 minus the number of mismatched bits and the output layer merges with OR; left over: it only proves sufficiency, so how does a network grow from points?"),
    ("Question, answer and left-over rows for KP03 and KP04.",) * 2)
add(6, ("KP05 與核心圖", "KP05, figures"),
    ("KP05 說明只有點時網路怎麼長出來；五個核心圖各對應一個知識點", "KP05 grows a network from labeled points; five core figures, one per KP"),
    ("KP03 說明一個切面有極限，KP04 用偵測器在編碼上再加一層，卻只證明夠用。KP05 怎麼讓網路從點長出來？", "KP03 shows one cutting plane has a limit and KP04 adds a layer of detectors on the codes, but only proves sufficiency. How does KP05 grow the network from points?"),
    ("這一頁先看 KP05。問：只有帶標籤的點，網路怎麼長出來；答：主要節點分一次，輔助節點處理混雜群，增加位元純化；留下：全對只是這批點，先固定架構再學權重。下面五張核心圖各對應一個知識點：KP01 兩條平行線夾出一條帶，A 在帶內；KP02 把它做成兩層網路；KP03 說明一個切面的極限；KP04 在編碼上再加一層；KP05 說明沒有現成網路時怎麼長。",
     "This page first covers KP05. Question: with only labeled points, how does a network grow; answer: a primary node splits once, auxiliary nodes handle the mixed groups and added bits purify them; left over: all right holds only for this batch of points, so fix the architecture first and then learn the weights. The five core figures below each match one KP: KP01 two parallel lines make a strip with A inside; KP02 builds it as a two-layer network; KP03 shows the limit of one cutting plane; KP04 adds another layer on the codes; KP05 shows how a network grows when none is given."),
    ("KP05 question, answer and left-over row, and five core-figure cards for KP01 to KP05.",) * 2)
add(7, ("整個網路", "Whole network"),
    ("回顧：整個網路分四個階段，每一階段只接上一階段的結果", "Review: the whole network has four stages, each taking only the result of the previous stage"),
    ("五個核心圖各對應一個知識點。把它們合起來，整個網路長什麼樣子？", "The five core figures each match one KP. Put together, what does the whole network look like?"),
    ("這一頁回顧整個網路的四個階段：輸入是點的座標；隱藏編碼是每條線一個位元連成編碼；偵測器只在編碼等於目標碼時回答是；輸出用或合併偵測器，判 A 或 B。每一階段只接上一階段的結果。常見迷思：構造出夠用的網路，就表示權重已學到、或這是最小的網路。其實構造只證明夠用。",
     "This page reviews the four stages of the whole network: the input is the coordinates of the point; the hidden code is one bit per line joined into a code; the detectors answer yes only when the code equals the target code; the output merges the detectors with OR and calls A or B. Each stage takes only the result of the previous stage. Common myth: having built a network that suffices means the weights are learned or that the network is smallest. In fact a construction only proves sufficiency."),
    ("Four stages: input, hidden code, detectors, output; myth: a construction only proves sufficiency.",) * 2)
add(8, ("四個提醒", "Reminders"),
    ("四個提醒：切面有極限、矛盾不可能全對、全對只是這批點、構造只證明夠用", "Four reminders: limited planes, contradictions, per-batch success, sufficiency is not learning"),
    ("整個網路分四個階段，每一階段只接上一階段的結果，但構造出夠用的網路不代表權重已學到。本章留下什麼、接下來學什麼？", "The whole network has four stages, each taking only the result of the previous one, but a construction that suffices does not mean the weights are learned. What does the chapter leave over, and what comes next?"),
    ("這一頁是四個提醒。第一，固定編碼上，一個切面有極限。第二，矛盾標籤不可能全對。第三，全對只是這批點全對。第四，構造證明夠用，不代表權重已學到或網路最小。下一章：先固定網路架構，再用代價函數學權重，也就是反向傳播，從 KP06 開始。",
     "This page gives the four reminders. First, on fixed codes one cutting plane has a limit. Second, contradictory labels cannot all be right. Third, all right means all right on this batch of points. Fourth, a construction proves sufficiency, not that the weights are learned or the network is smallest. Next chapter: fix the network architecture first and then learn the weights with a cost function, that is, backpropagation, starting at KP06."),
    ("Four numbered reminders and the next chapter: fix the architecture, then learn weights by backpropagation from KP06.",) * 2)
PAGES = P
