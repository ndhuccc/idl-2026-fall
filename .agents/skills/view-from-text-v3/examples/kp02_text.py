import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE; SK = os.environ.get("COURSE_SKILLS") or (os.path.join(os.path.abspath(os.environ["COURSE_REPO"]), ".agents", "skills") if os.environ.get("COURSE_REPO") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
for p in ("view-from-text-v3/tools", "view-from-text-v2/tools", "view-from-text-v2/examples"): sys.path.insert(0, os.path.join(SK, p))
sys.path.insert(0, HERE)
import kp02_pages as KP
from htmlkit import native
JD = os.path.join(HERE, "out")
UNIT_ZH, UNIT_EN = "兩層感知器", "Two-layer net"
TITLE_ZH = "圖型識別 第四章 KP02：兩層感知器（大三）"; TITLE_EN = "Pattern Recognition, Chapter 4, KP02: the two-layer perceptron (undergraduate)"
NOTE_ZH = "教材程度：大學三年級（undergrad）。依講義 lec05-kp02 以 view-from-text-v3 製作：先規劃頁面，每頁以一張原生資訊圖（HTML 設計後轉成原生節點）為主，名詞在同一頁圖內定義。共 11 頁，所有數字皆由程式計算。"
NOTE_EN = "Level: undergraduate (year 3). Made from the lecture lec05-kp02 with view-from-text-v3: pages were planned first and each page is mainly one native infographic (designed in HTML and converted to native nodes), with the terms defined on the same page. 11 pages; every number is computed by program."
def fig(n, alt_zh, alt_en):
    return lambda: native(f"kp02_p{n}", f"lec05-kp02-p{n}", KP.W, KP.H, JD, alt_en if os.environ.get("KP_LANG") == "en" else alt_zh)
P = []
NXT = {1: ("階梯規則", "the step rule"), 2: ("以 (0, 1) 走一遍", "walking (0, 1) through"), 3: ("四個點的編碼", "the codes of the four points"), 4: ("隱藏空間", "the hidden space"), 5: ("輸出分數的驗算", "checking the output scores"), 6: ("回到原平面", "back in the original plane"), 7: ("合併的後果", "the consequence of merging"), 8: ("小練習 1", "exercise 1"), 9: ("小練習 2", "exercise 2"), 10: ("去掉階梯", "removing the step")}
PREV = {2: ("兩層網路的接線", "the wiring of the two-layer network"), 3: ("階梯規則", "the step rule"), 4: ("以 (0, 1) 走一遍", "walking (0, 1) through"), 5: ("四個點的編碼", "the codes of the four points"), 6: ("隱藏空間", "the hidden space"), 7: ("輸出分數的驗算", "checking the output scores"), 8: ("回到原平面", "back in the original plane"), 9: ("合併的後果", "the consequence of merging"), 10: ("小練習 1", "exercise 1"), 11: ("小練習 2", "exercise 2")}
def add(n, lab, title, bridge, notes, alt):
    zh, en = notes
    if n in PREV: zh = f"承接上一頁的「{PREV[n][0]}」。" + zh; en = f"Following the previous page on {PREV[n][1]}. " + en
    if n == 11: zh += "到這裡這個單元就結束了；之後請接著讀下一單元。"; en += " That ends this unit; next, move on to the next unit."
    if n in NXT: zh += f"接下來的下一頁是「{NXT[n][0]}」。"; en += f" Next, the next page covers {NXT[n][1]}."
    P.append(dict(lab=lab, title=title, bridge=bridge, notes=(zh, en), layout="freeform", spec=fig(n, *alt)))
add(1, ("網路接線", "Wiring"),
    ("兩層網路：隱藏層的兩個單元各問一個是非題，輸出層把兩個答案合成判決", "Two-layer network: two hidden units each ask a yes-or-no question; the output unit combines them"),
    ("KP01 的結論是兩條平行線夾出一條帶，帶內剛好是 XOR 的 A。怎麼把這件事做成網路？", "KP01 ended with two parallel lines enclosing a strip that holds exactly the A points of XOR. How do we turn that into a network?"),
    ("這是 KP02 的第一頁。先備三件事：分數與正側、KP01 的 XOR 與帶、階梯。一條線答不了 A 還是 B，就先用兩條線各問一個是非題：隱藏單元 1 的分數是 g1 等於 x1 加 x2 減 0.5，隱藏單元 2 的分數是 g2 等於 x1 加 x2 減 1.5。每個單元各輸出一個位元 y1、y2，再接到一個輸出單元，它的分數是 y1 減 y2 減 0.5。隱藏層是先問小問題的一排單元；輸出層是最後在編碼上做分類的單元；隱藏層加輸出層就是兩層網路，輸入只提供座標，不算一層。",
     "This is the first page of KP02. Three prerequisites: score and positive side, the XOR and strip of KP01, and the step. One line cannot answer A or B, so first ask two yes-or-no questions with two lines: hidden unit 1 has the score g1 equal to x1 plus x2 minus 0.5 and hidden unit 2 has g2 equal to x1 plus x2 minus 1.5. Each unit outputs one bit, y1 and y2, which feed one output unit whose score is y1 minus y2 minus 0.5. The hidden layer is the row of units that asks the small questions; the output layer is the unit that classifies the codes; hidden plus output is a two-layer network, and the inputs only supply coordinates and are not a layer."),
    ("Wiring: inputs x1, x2 feed hidden unit 1 (g1 = x1 + x2 - 0.5, bit y1) and hidden unit 2 (g2 = x1 + x2 - 1.5, bit y2), which feed the output unit g_out = y1 - y2 - 0.5.",) * 2)
add(2, ("階梯規則", "Step rule"),
    ("每條線問「點在正側嗎」，用階梯規則把分數記成一個位元：正記 1、否則記 0", "Each line asks whether a point is on its positive side; the step rule records 1 if so, otherwise 0"),
    ("隱藏層的兩個單元各問一個是非題。每個單元怎麼把一個分數變成一個位元？", "The two hidden units each ask a yes-or-no question. How does each unit turn a score into a bit?"),
    ("這一頁定義階梯規則。對每條線 gj，點在正側也就是分數大於 0 時記 1，否則記 0，剛好等於 0 也記 0。得到的 0 或 1 叫位元。左圖是同一個平面，兩條線 x1 加 x2 等於 0.5 與 1.5，各自的正側分別著色並以箭頭標出，兩色重疊處是兩條線都在正側。每個點會得到兩個位元 y1 與 y2。",
     "This page defines the step rule. For each line gj, a point on the positive side, that is with score above 0, gets 1 and otherwise 0, and exactly 0 also gives 0. The resulting 0 or 1 is a bit. The left plot is one plane with the lines x1 plus x2 equal to 0.5 and 1.5; the positive side of each is shaded and marked by an arrow, and where the colours overlap both lines are positive. Every point gets two bits, y1 and y2."),
    ("Plane with the lines x1+x2 = 0.5 and 1.5, positive sides shaded with arrows; step rule y_j = 1 if g_j > 0 else 0.",) * 2)
add(3, ("走一個點", "One point"),
    ("以 (0, 1) 為例：g₁ = 0.5 > 0 得 y₁ = 1，g₂ = −0.5 ≤ 0 得 y₂ = 0，隱藏編碼是 (1, 0)", "Take (0, 1): g₁ = 0.5 > 0 gives y₁ = 1, g₂ = −0.5 ≤ 0 gives y₂ = 0, so the hidden code is (1, 0)"),
    ("階梯規則把分數記成一個位元：正記 1、否則記 0。拿一個具體的點 (0, 1) 走一遍，會得到什麼編碼？", "The step rule records a score as one bit: 1 if positive, otherwise 0. What code does the concrete point (0, 1) get?"),
    ("這一頁拿點 (0, 1) 走一遍。代入第一條線：g1 等於 0 加 1 減 0.5 等於 0.5，大於 0，所以 y1 等於 1。代入第二條線：g2 等於 0 加 1 減 1.5 等於負 0.5，小於等於 0，所以 y2 等於 0。兩個位元連起來 y 等於 (1, 0) 叫隱藏編碼，像替點貼上兩張位置標籤。流程由上而下：輸入、兩個分數、階梯規則、隱藏編碼（程式計算）。",
     "This page walks the point (0, 1) through. Into the first line: g1 equals 0 plus 1 minus 0.5 equal to 0.5, above 0, so y1 equals 1. Into the second line: g2 equals 0 plus 1 minus 1.5 equal to minus 0.5, at most 0, so y2 equals 0. The two bits written together, y equal to (1, 0), form the hidden code, like two position labels attached to the point. The flow reads top to bottom: input, two scores, step rule, hidden code (computed by program)."),
    ("Flow for (0,1): g1 = 0.5 > 0 gives y1 = 1; g2 = -0.5 <= 0 gives y2 = 0; hidden code (1, 0); the point lies between the two lines.",) * 2)
add(4, ("四點編碼", "Four codes"),
    ("四個點只有三種編碼：兩個 A 都得到 (1, 0)，落在兩條線之間的帶", "Four points get only three codes: both A points get (1, 0) and lie in the band between the two lines"),
    ("點 (0, 1) 的隱藏編碼是 (1, 0)。其他三個點各得到什麼編碼？", "The hidden code of the point (0, 1) is (1, 0). What code does each of the other three points get?"),
    ("這一頁算出四個點的分數與編碼。(0, 0)：g1 為負 0.5、g2 為負 1.5，編碼 (0, 0)，類別 B。(0, 1) 與 (1, 0)：g1 為 0.5、g2 為負 0.5，編碼都是 (1, 0)，類別 A。(1, 1)：g1 為 1.5、g2 為 0.5，編碼 (1, 1)，類別 B。兩條線把平面切成三條帶，編碼依序是 (0, 0)、(1, 0)、(1, 1)，兩個 A 都在中間那條（程式計算）。",
     "This page computes the scores and codes of the four points. (0, 0): g1 is minus 0.5 and g2 is minus 1.5, code (0, 0), class B. (0, 1) and (1, 0): g1 is 0.5 and g2 is minus 0.5, code (1, 0) for both, class A. (1, 1): g1 is 1.5 and g2 is 0.5, code (1, 1), class B. The two lines cut the plane into three bands with codes (0, 0), (1, 0), (1, 1), and both A points are in the middle band (computed by program)."),
    ("Table of g1, g2 and codes: (0,0) -> (0,0) B; (0,1) and (1,0) -> (1,0) A; (1,1) -> (1,1) B; plane cut into three bands.",) * 2)
add(5, ("隱藏空間", "Hidden space"),
    ("在編碼的平面上，一條輸出線 y₂ = y₁ − 0.5 就分開 A 和 B", "In the plane of codes, one output line y₂ = y₁ − 0.5 separates A from B"),
    ("四個點只有三種編碼，兩個 A 都得到 (1, 0)。把編碼畫成新平面，輸出層能用一條線分類嗎？", "Four points get only three codes, and both A points get (1, 0). Drawn as a new plane, can one line classify the codes?"),
    ("這一頁把每個點換成編碼 (y1, y2)，畫在新的平面上，叫隱藏空間。(0, 0) 是 B，(1, 0) 是 A，它是兩個輸入合併的位置，(1, 1) 是 B；(0, 1) 沒有輸入落在那裡，所以畫成空心。輸出層的分數是 g_out 等於 y1 減 y2 減 0.5。令它等於 0，移項得 y2 等於 y1 減 0.5，是隱藏空間裡的一條線。g_out 大於 0 等價於 y2 小於 y1 減 0.5，也就是在線的下方，判 A。兩個 A 合併不要緊，因為它們同類。",
     "This page replaces every point by its code (y1, y2) and draws it in a new plane, the hidden space. (0, 0) is B, (1, 0) is A and is where two inputs merged, (1, 1) is B; nothing lands on (0, 1), so it is drawn hollow. The output score is g_out equal to y1 minus y2 minus 0.5. Setting it to 0 and rearranging gives y2 equal to y1 minus 0.5, a line in the hidden space. g_out above 0 is equivalent to y2 below y1 minus 0.5, that is, below the line, called A. Merging the two A points is harmless because they are the same class."),
    ("Hidden plane (y1, y2): B at (0,0) and (1,1), A at (1,0), hollow (0,1); output line y2 = y1 - 0.5 with A below it.",) * 2)
add(6, ("逐點驗算", "Check scores"),
    ("輸出分數只有 (1, 0) 為正，四個輸入的預測 (0, 1, 1, 0) 剛好是 XOR", "Only the code (1, 0) has a positive output score, so the predictions (0, 1, 1, 0) are exactly XOR"),
    ("輸出線 y₂ = y₁ − 0.5 分開 A 和 B。把三個編碼逐一代入，輸出分數與預測是什麼？", "In the plane of codes (the hidden space), the output line y₂ = y₁ − 0.5 separates A from B. What are the output scores and predictions for each code?"),
    ("這一頁逐點驗算輸出層。(0, 0)：0 減 0 減 0.5 等於負 0.5，判 B。(1, 0)：1 減 0 減 0.5 等於 0.5，判 A，這是兩個 A 輸入的位置。(1, 1)：1 減 1 減 0.5 等於負 0.5，判 B。四個輸入的預測依序是 0、1、1、0（A 記 1、B 記 0），與 XOR 完全一致，四點全對。這只是這四點的驗算，不是新資料上的準確率（程式計算）。",
     "This page checks the output layer point by point. (0, 0): 0 minus 0 minus 0.5 equals minus 0.5, called B. (1, 0): 1 minus 0 minus 0.5 equals 0.5, called A, the position of the two A inputs. (1, 1): 1 minus 1 minus 0.5 equals minus 0.5, called B. The predictions of the four inputs are 0, 1, 1, 0 (A as 1, B as 0), exactly XOR, all four right. This only checks these four points and is not accuracy on new data (computed by program)."),
    ("Output scores: (0,0) -0.5 B; (1,0) 0.5 A; (1,1) -0.5 B; four-input predictions (0, 1, 1, 0) equal XOR, all four correct.",) * 2)
add(7, ("回到原平面", "Original plane"),
    ("回到原平面：整個網路選出 0.5 < x₁ + x₂ ≤ 1.5 的帶，帶內判 A", "Back in the original plane the network selects the strip 0.5 < x₁ + x₂ ≤ 1.5 and calls it A"),
    ("預測 (0, 1, 1, 0) 剛好是 XOR。整個網路翻回原來的平面，選出的是什麼區域？", "The predictions (0, 1, 1, 0) are exactly XOR. Translated back to the original plane, what region does the whole network select?"),
    ("這一頁把網路翻回原來的平面。y1 等於 1 就是 x1 加 x2 大於 0.5；y2 等於 0 就是 x1 加 x2 小於等於 1.5。輸出分數大於 0 等價於編碼剛好是 (1, 0)。所以整個網路選出 0.5 小於 x1 加 x2 小於等於 1.5 的帶，這就是 KP01 的帶：隱藏層做兩個判斷，輸出層做兩個都成立。代入核對：兩個 A 的 x1 加 x2 等於 1，在帶內；兩個 B 的是 0 與 2，在帶外（程式計算）。",
     "This page translates the network back to the original plane. y1 equal to 1 means x1 plus x2 above 0.5; y2 equal to 0 means x1 plus x2 at most 1.5. An output score above 0 is equivalent to the code being exactly (1, 0). So the whole network selects the strip 0.5 below x1 plus x2 at most 1.5, which is the strip of KP01: the hidden layer makes two judgments and the output layer says both hold. Check: the two A points have x1 plus x2 equal to 1, inside the strip, and the two B points have 0 and 2, outside (computed by program)."),
    ("Original plane with the strip 0.5 < x1 + x2 <= 1.5 shaded; the two A points inside, the two B points outside; logic y1 = 1 and y2 = 0.",) * 2)
add(8, ("合併的後果", "Merging"),
    ("同類合併沒關係；不同類的點得到同一個編碼，輸出層就必錯一個", "Merging same-class points is harmless; if two classes share one code, one must be wrong"),
    ("網路選出的帶裡，兩個 A 合併成同一個編碼。所有的合併都沒關係嗎？", "In the strip the network selects, the two A points merge into one code. Is every kind of merge harmless?"),
    ("這一頁看合併的後果。本例兩個 A 合併成編碼 (1, 0)，輸出層把它判成 A，兩點都對。常見迷思是以為任何合併都無妨。若不同類的點得到同一個編碼，叫編碼衝突，輸出層只看編碼，同一個編碼只能得到一個判決，A 與 B 至少錯一個，後面怎麼畫線都救不回來。右邊是示意圖，不是本例。另外要注意 y1、y2 只是在正側嗎的位元，不是類別的機率；這個構造只證明 XOR 表示得出來，別的任務要先檢查隱藏編碼能否被輸出層分開。",
     "This page looks at the consequence of merging. In this example the two A points merge into code (1, 0), the output layer calls it A, and both are right. A common myth is that any merge is harmless. If points of different classes get the same code, called a code conflict, the output layer sees only the code and one code gets one verdict, so A or B is wrong and no later line can repair it. The right panel is a schematic, not this example. Note also that y1 and y2 are only on-the-positive-side bits, not class probabilities; this construction only shows that XOR can be represented, and for other tasks one must first check that the output layer can separate the hidden codes."),
    ("Same-class merge: (0,1) and (1,0), both A, share code (1,0), called A, both right. Different-class merge (schematic): A and B share code (1,1), one must be wrong.",) * 2)
add(9, ("延伸：小練習一", "Extension: Task 1"),
    ("練習 1：常數改成 +0.5，三個分數都為正，四個輸入全判 A，錯 2 點", "Exercise 1: with constant +0.5 all scores are positive, all four inputs are called A, 2 wrong"),
    ("異類合併輸出層必錯一個。若保留隱藏編碼、把輸出分數的常數改成 +0.5，預測會變成什麼？", "Merging two classes into one code makes the output layer wrong on one. If we keep the codes and change the output constant to +0.5, what are the predictions?"),
    ("這是第一題小練習。保留隱藏編碼，把輸出分數改成 g_out 等於 y1 減 y2 加 0.5，四個輸入的預測是什麼？解題示範：符號式如上；代入三個位置，(0, 0) 得 0.5，(1, 0) 得 1.5，(1, 1) 得 0.5；三個分數都為正，四個輸入全判 A。核對：XOR 真值是 B、A、A、B，兩個 B 被判成 A，錯 2 點。偏置把線平移成 y2 等於 y1 加 0.5，線不再分開 B 與 A（程式計算）。",
     "This is the first exercise. Keep the hidden codes and change the output score to g_out equal to y1 minus y2 plus 0.5; what are the four predictions? Solution: the formula is as above; substituting the three positions gives 0.5 at (0, 0), 1.5 at (1, 0) and 0.5 at (1, 1); all three scores are positive, so all four inputs are called A. Check: the XOR truth is B, A, A, B, so the two B points are called A, 2 wrong. The bias shifts the line to y2 equal to y1 plus 0.5, which no longer separates B from A (computed by program)."),
    ("Exercise 1: g_out = y1 - y2 + 0.5 gives 0.5, 1.5, 0.5 at (0,0), (1,0), (1,1); all four inputs predicted A, the two B points wrong (2 errors); line y2 = y1 + 0.5.",) * 2)
add(10, ("延伸：小練習二", "Extension: Task 2"),
    ("練習 2：被合併的 (0, 1) 與 (1, 0) 同為 A，錯的原因是偏置平移，不是合併", "Exercise 2: the merged (0, 1) and (1, 0) are both A; the errors come from the shifted line"),
    ("常數 +0.5 讓四個輸入全判 A，錯 2 點。被合併的輸入是誰，這次合併為什麼不造成衝突？", "The constant +0.5 calls all four inputs A with 2 wrong. Which inputs are merged, and why does this merge cause no conflict?"),
    ("這是第二題小練習：哪些不同的輸入被合併？為什麼這次合併不造成衝突？解題示範：被合併的輸入是 (0, 1) 與 (1, 0)，它們的隱藏編碼都是 (1, 0)；兩個點的 XOR 真值都是 A，同類的點合併，輸出層只要把這個編碼判成 A 就對了，不造成衝突。核對：這次輸出線讓 (0, 0) 與 (1, 1) 兩個 B 也判成 A，錯的原因是偏置把線平移，不是合併（程式計算）。",
     "This is the second exercise: which different inputs are merged, and why does this merge cause no conflict? Solution: the merged inputs are (0, 1) and (1, 0), whose hidden codes are both (1, 0); both points are class A in XOR, merging same-class points is harmless, and the output layer just calls this code A. Check: with this output line the two B points (0, 0) and (1, 1) are also called A; the cause is that the bias shifts the line, not the merge (computed by program)."),
    ("Exercise 2: (0,1) and (1,0) both map to code (1,0) and are both A, so no conflict; with constant +0.5 all three positions score 0.5, 1.5, 0.5 and are called A; the two B points are wrong.",) * 2)
add(11, ("延伸：階梯不能省", "Extension: Step"),
    ("去掉階梯，g_out 恆為 0.5，每個點都判 A：線性式的組合還是線性式", "Without the step g_out is always 0.5 and every point is called A: linear forms stay linear"),
    ("偏置平移是錯 2 點的原因，不是合併。若連階梯都去掉，網路會變成什麼？", "The shifted line, not the merge, caused the 2 errors. If we remove the step as well, what does the network become?"),
    ("這一頁說明階梯不能省。去掉階梯，y1 等於 x1 加 x2 減 0.5，y2 等於 x1 加 x2 減 1.5，代入 g_out 等於 y1 減 y2 減 0.5，得 (x1 加 x2 減 0.5) 減 (x1 加 x2 減 1.5) 減 0.5 等於 0.5，和 x 無關，每個點都判 A，XOR 錯 2 點。線性式的組合還是線性式，非線性要靠階梯（程式計算）。這是補充的推導。",
     "This page shows that the step cannot be dropped. Without the step, y1 equals x1 plus x2 minus 0.5 and y2 equals x1 plus x2 minus 1.5; substituting into g_out equal to y1 minus y2 minus 0.5 gives (x1 plus x2 minus 0.5) minus (x1 plus x2 minus 1.5) minus 0.5, which equals 0.5 and does not depend on x, so every point is called A and XOR gets 2 points wrong. A combination of linear forms is still linear, and the nonlinearity comes from the step (computed by program). This derivation is supplementary."),
    ("Without the step g_out = (x1+x2-0.5) - (x1+x2-1.5) - 0.5 = 0.5 at all four points; the whole plane predicts A and the two B points are wrong (2 errors).",) * 2)
PAGES = P
