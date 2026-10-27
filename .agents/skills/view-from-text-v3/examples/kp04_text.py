import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE; SK = os.environ.get("COURSE_SKILLS") or (os.path.join(os.path.abspath(os.environ["COURSE_REPO"]), ".agents", "skills") if os.environ.get("COURSE_REPO") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
for p in ("view-from-text-v3/tools", "view-from-text-v2/tools", "view-from-text-v2/examples"): sys.path.insert(0, os.path.join(SK, p))
sys.path.insert(0, HERE)
import kp04_pages as KP
from htmlkit import native
JD = os.path.join(HERE, "out")
UNIT_ZH, UNIT_EN = "三層感知器", "Three-layer perceptron"
TITLE_ZH = "圖型識別 第四章 KP04：三層感知器（大三）"; TITLE_EN = "Pattern Recognition, Chapter 4, KP04: three-layer perceptron (undergraduate)"
NOTE_ZH = "教材程度：大學三年級（undergrad）。依講義 lec05-kp04 以 view-from-text-v3 製作：先規劃頁面，每頁以一張原生資訊圖（HTML 設計後轉成原生節點）為主，名詞在同一頁圖內定義。共 10 頁，所有數字皆由程式計算。"
NOTE_EN = "Level: undergraduate (year 3). Made from the lecture lec05-kp04 with view-from-text-v3: pages were planned first and each page is mainly one native infographic (designed in HTML and converted to native nodes), with the terms defined on the same page. 10 pages; every number is computed by program."
def fig(n, alt_zh, alt_en):
    return lambda: native(f"kp04_p{n}", f"lec05-kp04-p{n}", KP.W, KP.H, JD, alt_en if os.environ.get("KP_LANG") == "en" else alt_zh)
P = []
LAB = {1: ("目標碼", "the target codes"), 2: ("三步走", "the three steps"), 3: ("偵測 110", "detecting 110"), 4: ("三個偵測器", "the three detectors"), 5: ("走過網路", "the path through the network"), 6: ("或合併規則", "the OR merge rule"), 7: ("八個角檢查", "the check of eight corners"), 8: ("輸入平面", "the input plane"), 9: ("重疊長方形", "the overlapping rectangles"), 10: ("小練習", "the exercises")}
def add(n, lab, title, bridge, notes, alt):
    zh, en = notes
    if n > 1: zh = f"承接上一頁的「{LAB[n-1][0]}」。" + zh; en = f"Following the previous page on {LAB[n-1][1]}. " + en
    if n == 10: zh += "到這裡這個單元就結束了；之後請接著讀下一單元。"; en += " That ends this unit; next, move on to the next unit."
    else: zh += f"接下來的下一頁是「{LAB[n+1][0]}」。"; en += f" Next, the next page covers {LAB[n+1][1]}."
    P.append(dict(lab=lab, title=title, bridge=bridge, notes=(zh, en), layout="freeform", spec=fig(n, *alt)))
add(1, ("目標碼", "Target codes"),
    ("A 角是 000、110、111，分散在立方體各處：底面的 A 與 B 沿對角交錯，一個切面切不開", "The A corners 000, 110, 111 are scattered, and A and B alternate on the bottom face: one cut fails"),
    ("上一個知識點最後，A 是 000、110、111 時，立方體底面的 A 與 B 交錯，和 XOR 一樣。最後一層能一刀切開嗎？", "At the end of the previous unit, with A at 000, 110, 111, A and B alternate on the cube's bottom face like XOR. Can the last layer separate them with one cut?"),
    ("這是 KP04 的第一頁。先備三件事：編碼與立方體的角（三條線給三個位元，連起來對應 8 個角之一）、輸出層是一刀（只能用一個平面把角分成兩邊）、階梯規則。上一個知識點最後，A 是 000、110、111：底面的 000 與 110 是 A，100 與 010 是 B，沿對角交錯，和 XOR 一樣，最後一層切不開；101 沒有任何輸入會到達。目標碼指標為 A 的角的編碼。解法的想法是不要一刀分完（程式計算）。",
     "This is the first page of KP04. Prerequisites: codes and cube corners (three lines give three bits, which together name one of the 8 corners), the output layer is one cut (one plane splitting the corners into two sides), and the step rule. At the end of the previous unit A is 000, 110, 111: on the bottom face 000 and 110 are A while 100 and 010 are B, alternating along the diagonal like XOR, so the last layer cannot cut them; 101 is reached by no input. A target code is the code of a corner labelled A. The idea of the fix is not to finish in one cut (computed by program)."),
    ("Cube with A at 000, 110, 111 and B elsewhere, 101 hollow, bottom face framed in red where A and B alternate.",) * 2)
add(2, ("三步走", "Three steps"),
    ("三步走：先產生編碼，每個目標碼配一個偵測器，任一個啟動就判 A", "Three steps: make the code, give each target code a detector, and call A if any one fires"),
    ("A 角是 000、110、111，底面的 A 與 B 沿對角交錯，一個切面切不開。不要一刀分完，改成幾步？", "The A corners 000, 110, 111 alternate with B on the bottom face, so one cut fails. If not one cut, how many steps?"),
    ("這一頁把網路分成三步：第一隱藏層用三條線產生編碼，第二隱藏層為每個目標碼放一個偵測器，輸出層做或合併。三層網路指兩個隱藏層加一個輸出層；偵測器是一個階梯單元，輸入碼剛好等於某個目標碼時輸出 1；或合併是任何一個偵測器輸出 1 就判 A，全部是 0 才判 B。上一個知識點的兩層網路中間少了偵測器這一層（程式計算）。",
     "This page splits the network into three steps: the first hidden layer makes the code with three lines, the second hidden layer holds one detector per target code, and the output layer does an OR merge. A three-layer network means two hidden layers plus an output layer; a detector is a step unit that outputs 1 exactly when the input code equals one target code; an OR merge calls A when any detector outputs 1 and B only when all are 0. The previous unit's two-layer network lacked the detector layer in the middle (computed by program)."),
    ("Block diagram: input x, first hidden layer making the code, second hidden layer of detectors, output layer with OR merge, then A or B.",) * 2)
add(3, ("偵測 110", "Detecting 110"),
    ("偵測 110：每錯一個位元，分數就少 1，只有 110 本身得 +0.5", "Detecting 110: every wrong bit costs 1 point, and only 110 itself scores +0.5"),
    ("三步走：先產生編碼，每個目標碼配一個偵測器，任一個啟動就判 A。偵測器的分數怎麼算？", "Three steps: make the code, give each target code a detector, and call A if any one fires. How is a detector's score computed?"),
    ("這一頁算偵測器的分數。對目標碼 t，h 等於各位元的 s 乘 y 的總和，減去 n1 減 0.5；t 的位元是 1 時 s 取正 1，是 0 時取負 1，n1 是 t 中 1 的個數。因為 s 乘 y 的總和等於 n1 減 m，m 是不吻合的位元數，所以 h 等於 0.5 減 m：全吻合得 0.5，差 1 位得負 0.5，差 2 位得負 1.5，差 3 位得負 2.5。例如 110：y 等於 110 得 0.5，010 得負 0.5（程式計算）。",
     "This page computes a detector's score. For target code t, h equals the sum of s times y over the bits minus (n1 minus 0.5); s is plus 1 where the bit of t is 1 and minus 1 where it is 0, and n1 is the number of 1s in t. Since the sum of s times y equals n1 minus m, with m the number of mismatched bits, h equals 0.5 minus m: a full match scores 0.5, one bit off scores minus 0.5, two off minus 1.5 and three off minus 2.5. For example for 110: y equal to 110 scores 0.5 and 010 scores minus 0.5 (computed by program)."),
    ("Cube with the score of the 110 detector at each corner, only 110 positive; formula h = 0.5 - m with table of m = 0..3 giving +0.5, -0.5, -1.5, -2.5.",) * 2)
add(4, ("三個偵測器", "Three detectors"),
    ("同一個做法套用到 000、110、111：每個偵測器只在自己的角得 +0.5", "The same recipe for 000, 110 and 111: each detector scores +0.5 only at its own corner"),
    ("偵測 110：每錯一個位元，分數就少 1，只有 110 本身得 +0.5。其他目標碼怎麼做？", "Detecting 110: every wrong bit costs 1 point and only 110 itself scores +0.5. How do the other target codes work?"),
    ("這一頁把同一個做法套到三個目標碼：h000 等於負 y1 負 y2 負 y3 加 0.5；h110 等於 y1 加 y2 減 y3 減 1.5；h111 等於 y1 加 y2 加 y3 減 2.5。000 沒有 1，偏置是正 0.5；111 有三個 1，偏置是負 2.5。右邊的表列出八個編碼對三個偵測器的分數，正分數只出現在自己的角（程式計算）。",
     "This page applies the same recipe to the three target codes: h000 is minus y1 minus y2 minus y3 plus 0.5; h110 is y1 plus y2 minus y3 minus 1.5; h111 is y1 plus y2 plus y3 minus 2.5. 000 has no 1, so its bias is plus 0.5; 111 has three 1s, so its bias is minus 2.5. The table lists the scores of the three detectors on the eight codes; a positive score appears only at its own corner (computed by program)."),
    ("Three detector formulas and a table of eight codes by three detectors; positive scores only at 000 for h000, 110 for h110, 111 for h111.",) * 2)
add(5, ("走過網路", "Through the net"),
    ("收到編碼 110：只有 110 的偵測器啟動，輸出分數 q = 0.5，判 A", "Code 110 arrives: only the 110 detector fires, and the output score q = 0.5 gives A"),
    ("同一個做法套用到 000、110、111，每個偵測器只在自己的角得 +0.5。收到編碼 110，三個偵測器各給什麼？", "The same recipe for 000, 110 and 111 gives each detector +0.5 only at its own corner. When code 110 arrives, what does each detector give?"),
    ("這一頁代入 y 等於 (1,1,0)：三個分數是 h000 等於負 1.5、h110 等於 0.5、h111 等於負 0.5；轉成位元 b 等於 f(h)，得 (0,1,0)。輸出層把位元相加再減 0.5：0 加 1 加 0 減 0.5 等於 0.5，大於 0，判 A。橘色粗線是啟動的路徑（程式計算）。",
     "This page substitutes y equal to (1,1,0): the three scores are h000 equal to minus 1.5, h110 equal to 0.5 and h111 equal to minus 0.5; turned into bits b equal to f(h) they give (0,1,0). The output layer adds the bits and subtracts 0.5: 0 plus 1 plus 0 minus 0.5 equals 0.5, above 0, so A. The thick orange lines are the active path (computed by program)."),
    ("Network diagram for y=(1,1,0): detector scores -1.5, 0.5, -0.5, bits 0,1,0, output q = 0.5 > 0 gives A.",) * 2)
add(6, ("或合併規則", "OR merge rule"),
    ("合併只要一條規則：q = k − 0.5，k = 0 判 B、k ≥ 1 判 A", "One rule merges them: q = k − 0.5, with k = 0 called B and k ≥ 1 called A"),
    ("收到編碼 110：只有 110 的偵測器啟動，q = 0.5 判 A。不同的啟動情形，輸出層怎麼統一處理？", "Code 110 arrives and only the 110 detector fires, with q = 0.5 called A. For other firing cases, how does the output layer treat them all alike?"),
    ("這一頁統一合併規則：輸出分數 q 等於三個偵測器的位元相加再減 0.5，也就是 k 減 0.5，k 是輸出 1 的偵測器個數。k 等於 0 得負 0.5 判 B；k 至少 1 得 0.5、1.5、2.5 判 A。這就是或合併。零點在 k 等於 0 與 k 等於 1 之間。每個編碼最多讓一個偵測器啟動，所以實際只出現 k 等於 0 或 1（程式計算）。",
     "This page unifies the merge rule: the output score q is the sum of the three detector bits minus 0.5, that is k minus 0.5, where k is the number of detectors outputting 1. k equal to 0 gives minus 0.5 and B; k at least 1 gives 0.5, 1.5, 2.5 and A. This is the OR merge. The zero lies between k equal to 0 and k equal to 1. Each code fires at most one detector, so only k equal to 0 or 1 occur (computed by program)."),
    ("Number line of q = k - 0.5 with zero between k=0 and k=1, B on the left and A on the right; table k 0..3 giving -0.5, 0.5, 1.5, 2.5.",) * 2)
add(7, ("八個角檢查", "Eight corners"),
    ("八個角逐一檢查：只有 000、110、111 判 A，其餘 5 個編碼判 B", "All eight corners checked: only 000, 110, 111 are called A, the other five are B"),
    ("合併只要一條規則：q = k − 0.5，k = 0 判 B、k ≥ 1 判 A。把八個編碼逐一檢查，結果對嗎？", "One rule merges them: q = k − 0.5, with k = 0 called B and k ≥ 1 called A. Checking all eight codes one by one, is the result right?"),
    ("這一頁不只驗算成功的輸入，把八個編碼全查一遍。010：三個分數是負 0.5、負 0.5、負 1.5，位元全為 0，q 等於負 0.5，判 B。虛擬碼 101 也全為 0，判 B；沒有輸入會得到它，不影響分類。只有 000、110、111 各讓自己的偵測器啟動並判 A（程式計算）。",
     "This page does not only check the successful inputs but all eight codes. For 010 the three scores are minus 0.5, minus 0.5, minus 1.5, all bits are 0, q is minus 0.5 and the call is B. The unused code 101 also gives all 0 and B; no input reaches it, so classification is unaffected. Only 000, 110, 111 each fire their own detector and are called A (computed by program)."),
    ("Table of the eight codes with detector scores, bits, k, q and call; only 000, 110, 111 are A, 101 is unused.",) * 2)
add(8, ("輸入平面", "Input plane"),
    ("翻回輸入平面：三層網路把 000、110、111 三個編碼區都分對", "Back in the input plane: the three-layer network gets the code regions 000, 110, 111 right"),
    ("八個角逐一檢查，只有 000、110、111 判 A。翻回輸入平面，這個結果是什麼樣子？", "Checking the eight corners one by one, only 000, 110, 111 are called A. Back in the input plane, what does this look like?"),
    ("這一頁翻回輸入平面：三條線 g1、g2、g3 切出 8 個編碼區，A 區是 000、110、111 三個編碼區的聯集（110 與 111 相鄰），B 區的 010 夾在其中；單一切面做不到，三層網路做到了。圖上的三個 A 點與一個 B 點，經程式驗算全被分對。常見迷思是以為存在三層構造，網路就已學到權重、或這是最小的網路；構造只證明夠用，權重怎麼從帶標籤的點得到，是下一個知識點（程式計算）。",
     "This page goes back to the input plane: the three lines g1, g2, g3 cut 8 code regions, the A region is the union of the code regions 000, 110, 111 (110 and 111 are adjacent), and B's 010 is squeezed in between; a single cut cannot do this but the three-layer network does. The three A points and one B point in the figure are all classified right by the program. A common myth is that once a three-layer construction exists the weights have been learned or the network is minimal; the construction only shows it suffices, and how weights come from labelled points is the next unit (computed by program)."),
    ("Input plane with three orange lines cutting eight code regions, A region blue and B region orange, three A points and one B point all right.",) * 2)
add(9, ("重疊長方形", "Overlapping rects"),
    ("重疊的長方形要 7 個偵測器，不是 2 個：偵測器的個數看細分後的格子", "Overlapping rectangles need 7 detectors, not 2: detectors count the refined cells"),
    ("三層網路把 000、110、111 三個編碼區都分對。如果 A 區是兩個重疊的長方形，要幾個偵測器？", "The three-layer network gets the code regions 000, 110, 111 right. If the A region is two overlapping rectangles, how many detectors are needed?"),
    ("這一頁是延伸：兩個重疊的長方形各用 4 條線，共 8 條線把平面切成格子，A 佔其中 7 格，所以要 7 個偵測器，不是 2 個。原長方形的個數不等於細分後的編碼數，做法不變：每個編碼區配一個偵測器，再或合併（程式計算）。",
     "This page is an extension: two overlapping rectangles use 4 lines each, 8 lines in all, to cut the plane into cells; A occupies 7 of them, so 7 detectors are needed, not 2. The number of original rectangles is not the number of codes after refining, and the recipe is unchanged: one detector per code region, then an OR merge (computed by program)."),
    ("Two overlapping rectangles in orange and purple refined into seven numbered cells, so seven detectors.",) * 2)
add(10, ("小練習", "Exercises"),
    ("兩題小練習：h₀₁₁ = −y₁ + y₂ + y₃ − 1.5，010 的分數是 −0.5", "Two exercises: h₀₁₁ = −y₁ + y₂ + y₃ − 1.5, and the score at 010 is −0.5"),
    ("重疊的長方形要 7 個偵測器，不是 2 個，偵測器的個數看細分後的格子。用 011 的偵測器自己算一次，結果是什麼？", "Overlapping rectangles need 7 detectors, not 2, since detectors count the refined cells. Work out the 011 detector yourself: what is the result?"),
    ("兩題小練習。題 1：011 有兩個 1，權重 (負 1, 正 1, 正 1)，偏置是負 (2 減 0.5) 等於負 1.5，所以 h011 等於負 y1 加 y2 加 y3 減 1.5；y 等於 011 得 0 加 1 加 1 減 1.5 等於 0.5，y 等於 111 得負 1 加 1 加 1 減 1.5 等於負 0.5，所以 011 輸出 1，111 輸出 0。題 2：010 與 011 差 1 位，0.5 減 1 等於負 0.5；代入 0 加 1 加 0 減 1.5 也是負 0.5，兩種算法一致（程式計算）。",
     "Two exercises. Q1: 011 has two 1s, weights (minus 1, plus 1, plus 1) and bias minus (2 minus 0.5) equal to minus 1.5, so h011 is minus y1 plus y2 plus y3 minus 1.5; y equal to 011 gives 0 plus 1 plus 1 minus 1.5 equal to 0.5 and y equal to 111 gives minus 1 plus 1 plus 1 minus 1.5 equal to minus 0.5, so 011 outputs 1 and 111 outputs 0. Q2: 010 differs from 011 in one bit, 0.5 minus 1 equals minus 0.5; substituting 0 plus 1 plus 0 minus 1.5 is also minus 0.5, so the two ways agree (computed by program)."),
    ("Exercise 1: h011 = -y1+y2+y3-1.5 scoring 0.5 at 011 and -0.5 at 111; exercise 2: the score at 010 is -0.5 both ways.",) * 2)
PAGES = P
