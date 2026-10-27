import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE; SK = os.environ.get("COURSE_SKILLS") or (os.path.join(os.path.abspath(os.environ["COURSE_REPO"]), ".agents", "skills") if os.environ.get("COURSE_REPO") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
for p in ("view-from-text-v3/tools", "view-from-text-v2/tools", "view-from-text-v2/examples"): sys.path.insert(0, os.path.join(SK, p))
sys.path.insert(0, HERE)
import kp01_pages as KP
from htmlkit import native
JD = os.path.join(HERE, "out")
UNIT_ZH, UNIT_EN = "XOR 與一條線的極限", "XOR and one line"
TITLE_ZH = "圖型識別 第四章 KP01：XOR 與一條線的極限（大三）"; TITLE_EN = "Pattern Recognition, Chapter 4, KP01: XOR and the limit of one line (undergraduate)"
NOTE_ZH = "教材程度：大學三年級（undergrad）。依講義 lec05-kp01 以 view-from-text-v3 製作：先規劃頁面，每頁以一張原生資訊圖（HTML 設計後轉成原生節點）為主，名詞在同一頁圖內定義。共 11 頁，所有數字皆由程式計算。"
NOTE_EN = "Level: undergraduate (year 3). Made from the lecture lec05-kp01 with view-from-text-v3: pages were planned first and each page is mainly one native infographic (designed in HTML and converted to native nodes), with the terms defined on the same page. 11 pages; every number is computed by program."
def fig(n, alt_zh, alt_en):
    return lambda: native(f"kp01_p{n}", f"lec05-kp01-p{n}", KP.W, KP.H, JD, alt_en if os.environ.get("KP_LANG") == "en" else alt_zh)
P = []
NXT = {1: ("分數與規則", "scores and the rule"), 2: ("同一條線分 XOR", "the same line on XOR"), 3: ("換三條線", "three other lines"), 4: ("A 的對角線", "the A diagonal"), 5: ("B 的矛盾", "the contradiction from B"), 6: ("訓練補不了", "why training cannot help"), 7: ("兩條線夾出的帶", "the strip between two lines"), 8: ("邊界上的點", "points on the boundary"), 9: ("小練習 1", "exercise 1"), 10: ("小練習 2 與 3", "exercises 2 and 3")}
PREV = {2: ("三種標籤", "the three labelings"), 3: ("分數與規則", "scores and the rule"), 4: ("同一條線分 XOR", "the same line on XOR"), 5: ("換三條線", "three other lines"), 6: ("A 的對角線", "the A diagonal"), 7: ("B 的矛盾", "the contradiction from B"), 8: ("訓練補不了", "why training cannot help"), 9: ("兩條線夾出的帶", "the strip between two lines"), 10: ("邊界上的點", "points on the boundary"), 11: ("小練習 1", "exercise 1")}
def add(n, lab, title, bridge, notes, alt):
    zh, en = notes
    if n in PREV: zh = f"承接上一頁的「{PREV[n][0]}」。" + zh; en = f"Following the previous page on {PREV[n][1]}. " + en
    if n == 11: zh += "到這裡這個單元就結束了；之後請接著讀下一單元。"; en += " That ends this unit; next, move on to the next unit."
    if n in NXT: zh += f"接下來的下一頁是「{NXT[n][0]}」。"; en += f" Next, the next page covers {NXT[n][1]}."
    P.append(dict(lab=lab, title=title, bridge=bridge, notes=(zh, en), layout="freeform", spec=fig(n, *alt)))
add(1, ("三種標籤", "Three labelings"),
    ("同樣四個點只換標籤：OR、AND 看似一條線能分，XOR 把同類放在對角", "Same four points, new labels: one line seems enough for OR and AND, but XOR puts classes diagonally"),
    ("第三章讓模型用一條線決定類別。若只換四個點的標籤，同一個模型還答得了嗎？", "Chapter 3 let a model decide the class with one line. If only the labels of the same four points change, can that model still answer?"),
    ("這是 KP01 的第一頁。先備三件事：線性分類器、分數、真值表。四個輸入是 (0,0)、(0,1)、(1,0)、(1,1)，兩個輸入各是 0 或 1。標籤是每個點要分到的類別：A 是圓點、輸出 1，B 是方點、輸出 0。OR 是只要有一個輸入為 1 就輸出 1，輸出依序為 0111；AND 是兩個輸入都為 1 才輸出 1，輸出 0001；互斥或 XOR 是兩個輸入不同才輸出 1，輸出 0110。三格座標完全相同，只有標籤不同。OR 的 A 占三個角，AND 的 A 只占右上角，看起來都能用一條線隔開；XOR 把同類放在對角，不確定（程式計算）。",
     "This is the first page of KP01. Three prerequisites: linear classifiers, scores and truth tables. The four inputs are (0,0), (0,1), (1,0), (1,1), each coordinate 0 or 1. A label is the class a point must go to: A is a circle with output 1 and B is a square with output 0. OR outputs 1 if any input is 1, giving 0111 in order; AND outputs 1 only if both inputs are 1, giving 0001; exclusive OR, XOR, outputs 1 only if the inputs differ, giving 0110. The three panels have identical coordinates and differ only in labels. The A points of OR take three corners and those of AND only the top right, so one line looks enough for both; XOR puts same-class points on the diagonals, which is unclear (computed by program)."),
    ("Three panels of the same four points labelled OR (0111), AND (0001) and XOR (0110); A circles, B squares.",) * 2)
add(2, ("分數與規則", "Score and rule"),
    ("分數為正就判 A：OR 取 g = x₁ + x₂ − 0.5，三個 A 都在正側", "Positive score means A: OR with g = x₁ + x₂ − 0.5 puts all three A points on the positive side"),
    ("OR 的 A 占三個角，AND 的 A 只占右上角，XOR 把同類放在對角。一條線怎麼打分數、怎麼決定類別？", "The A points of OR take three corners, AND only one, and XOR puts same-class points on the diagonals. How does one line give scores and decide the class?"),
    ("這一頁把規則說清楚。分數是 g 等於 w1 x1 加 w2 x2 加 w0，預測是分數大於 0 判 A、否則判 B，剛好等於 0 也判 B，不能看完真實標籤再改規則；g 等於 0 就是圖上的線。線性可分是指存在一組係數能把所有點依這條規則分對，重點是存在，不只是目前找到的那一組。例子：OR 取 w1、w2 都是 1、w0 是負 0.5，四個點的分數是負 0.5、0.5、0.5、1.5，後三個為正，判 A，剛好是 OR（程式計算）。",
     "This page states the rule. The score is g equal to w1 x1 plus w2 x2 plus w0 and the prediction is A if the score is above 0 and B otherwise, exactly 0 also B, and the rule may not change after seeing the labels; g equal to 0 is the line in the plot. Linearly separable means that some set of coefficients classifies every point correctly by this rule; the point is existence, not just the set found so far. Example: OR with w1 and w2 equal to 1 and w0 equal to minus 0.5 gives scores minus 0.5, 0.5, 0.5, 1.5; the last three are positive and called A, exactly OR (computed by program)."),
    ("OR line x1 + x2 = 0.5 with the positive side shaded; scores -0.5, 0.5, 0.5, 1.5; the three A points are on the positive side.",) * 2)
add(3, ("同線分 XOR", "Same line, XOR"),
    ("同一條線，OR 全對，XOR 錯 1 點：正確率 3/4 = 75%", "The same line gets OR right but XOR wrong at 1 point: accuracy 3/4 = 75%"),
    ("OR 的分數為 −0.5、0.5、0.5、1.5，三個 A 都在正側。同一條線拿去分 XOR，會怎樣？", "The OR scores are −0.5, 0.5, 0.5, 1.5 and all three A points are on the positive side. What happens when the same line is applied to XOR?"),
    ("這一頁把同一條線 g 等於 x1 加 x2 減 0.5 拿去分 XOR，逐點代入：(0,0) 分數負 0.5，預測 B，真值 B；(0,1) 分數 0.5，預測 A，真值 A；(1,0) 分數 0.5，預測 A，真值 A；(1,1) 分數 1.5，預測 A，真值 B，錯。預測恰好等於 OR，對 XOR 錯 1 點，正確率是 3 除以 4 等於 0.75，也就是 75%，加上錯誤的 25% 剛好 100%（程式計算）。這只是這一條線的結果，還不能說所有線都不行。",
     "This page applies the same line g equal to x1 plus x2 minus 0.5 to XOR and substitutes point by point: (0,0) scores minus 0.5, predicted B, truth B; (0,1) scores 0.5, predicted A, truth A; (1,0) scores 0.5, predicted A, truth A; (1,1) scores 1.5, predicted A, truth B, wrong. The predictions equal OR and XOR has 1 error, accuracy 3 over 4 equal to 0.75, that is 75%, and with the 25% errors it adds up to 100% (computed by program). This is only the result for this one line and does not yet show that every line fails."),
    ("Table for g = x1 + x2 - 0.5 on XOR: scores -0.5, 0.5, 0.5, 1.5; predictions B, A, A, A; (1,1) is wrong; accuracy 3/4 = 75%.",) * 2)
add(4, ("換三條線", "Three more lines"),
    ("換三條線試試：分別錯 2、3、1 點，但試了三條不是證明", "Three other lines make 2, 3 and 1 errors, but trying three lines is not a proof"),
    ("同一條線對 XOR 錯 1 點，正確率 75%。換一條線會不會好？", "The same line gets XOR wrong at 1 point, 75% accuracy. Would another line do better?"),
    ("這一頁換三條線。第一條 g 等於 x1 減 0.5，預測依序為 0、0、1、1，錯 2 點。第二條 g 等於 x1 加 x2 減 1.5，預測 0、0、0、1，只切下右上角，錯的是兩個 A 加一個 B，共 3 點。第三條 g 等於 x1 減 x2，預測 0、0、1、0，線上兩個點分數為 0，依規則判 B，錯 1 點。三條都有錯，但這不是證明：沒試到的線太多（程式計算）。",
     "This page tries three other lines. The first, g equal to x1 minus 0.5, predicts 0, 0, 1, 1 in order and has 2 errors. The second, g equal to x1 plus x2 minus 1.5, predicts 0, 0, 0, 1, cutting off only the top right corner; the errors are the two A points and one B point, 3 in all. The third, g equal to x1 minus x2, predicts 0, 0, 1, 0; the two points on the line score 0 and are called B by the rule, giving 1 error. All three have errors, but this is not a proof: too many lines are left untried (computed by program)."),
    ("Three lines on XOR: x1 - 0.5 predicts (0,0,1,1) with 2 errors; x1 + x2 - 1.5 predicts (0,0,0,1) with 3 errors; x1 - x2 predicts (0,0,1,0) with 1 error.",) * 2)
add(5, ("A 的對角線", "A diagonal"),
    ("兩個 A 點的分數相加必為正，連線的中點 M 的分數也為正", "The two A scores add up to a positive number, and so does the score at the midpoint M"),
    ("三條線試了都有錯，但沒試到的線太多。怎樣一次排除所有的線？", "All three lines tried have errors, but too many lines are untried. How can we rule out every line at once?"),
    ("這一頁假設有任意一組 w1、w2、w0 能把 XOR 全分對，列出它必須滿足的條件。A 的兩個點：g(0,1) 等於 w2 加 w0 大於 0，g(1,0) 等於 w1 加 w0 大於 0。正數加正數仍為正，兩式相加得 w1 加 w2 加 2 w0 大於 0。幾何意思：兩個 A 點連成對角線，中點 M 是 (1/2, 1/2)；g 是線性的，所以 g(M) 等於 w1 加 w2 加 2 w0 再除以 2，剛好是兩個分數的平均，也是正的（程式計算）。",
     "This page supposes that some w1, w2, w0 classify XOR perfectly and lists the conditions they must meet. For the two A points: g(0,1) equals w2 plus w0 and must be above 0, and g(1,0) equals w1 plus w0 and must be above 0. A positive plus a positive stays positive, so adding gives w1 plus w2 plus 2 w0 above 0. Geometrically the two A points form a diagonal whose midpoint M is (1/2, 1/2); g is linear, so g(M) equals w1 plus w2 plus 2 w0 over 2, exactly the average of the two scores, and it is positive too (computed by program)."),
    ("Diagonal between the two A points (0,1) and (1,0) with midpoint M = (1/2, 1/2); g(0,1) > 0, g(1,0) > 0 so w1 + w2 + 2w0 > 0 and g(M) > 0.",) * 2)
add(6, ("B 的矛盾", "B contradiction"),
    ("兩個 B 點的分數相加不大於 0，與 A 的同一個式子矛盾：沒有任何一條線能全對", "The B scores add up to at most 0, contradicting A on the same expression: no line works"),
    ("兩個 A 點的分數相加必為正，中點 M 的分數也為正。B 的兩個點會得到什麼條件？", "The two A scores add up to a positive number, and so does the score at the midpoint M. What condition do the two B points give?"),
    ("這一頁看 B 的兩個點，分數依固定規則必須不大於 0：g(0,0) 等於 w0 不大於 0，g(1,1) 等於 w1 加 w2 加 w0 不大於 0。不大於 0 的數相加不可能為正：w0 加上 w1 加 w2 加 w0 等於 w1 加 w2 加 2 w0，不大於 0。這和 A 的結果是完全相同的式子：A 要求它大於 0，B 要求它不大於 0，矛盾。所以不存在能全對的係數，這是對所有線的證明。幾何上兩條對角線共用中點 M，g(M) 不可能一個為正、一個不大於 0。",
     "This page looks at the two B points, whose scores must be at most 0 by the fixed rule: g(0,0) equals w0 at most 0 and g(1,1) equals w1 plus w2 plus w0 at most 0. Numbers at most 0 cannot add up to a positive: w0 plus w1 plus w2 plus w0 equals w1 plus w2 plus 2 w0, at most 0. This is exactly the same expression as for A: A requires it above 0 and B requires it at most 0, a contradiction. So no coefficients classify everything correctly, a proof for every line. Geometrically both diagonals share the midpoint M, and g(M) cannot be positive for one and at most 0 for the other."),
    ("Both diagonals share the midpoint M; A needs w1 + w2 + 2w0 > 0 and B needs w1 + w2 + 2w0 <= 0, a contradiction for every line.",) * 2)
add(7, ("訓練補不了", "Training cannot help"),
    ("不管怎麼訓練，一條線都無法把 XOR 全分對：最低錯誤是 1 點，不是 0", "However it is trained, one line cannot classify XOR perfectly: the lowest error is 1, not 0"),
    ("同一個式子，A 要求大於 0，B 要求不大於 0，不存在能全對的係數。每條線都不行，繼續訓練有用嗎？", "The same expression must be above 0 for A and at most 0 for B, so no coefficients work. If every line fails, does more training help?"),
    ("這一頁處理常見迷思：多跑幾輪、調小學習率或重新初始化，應該就能全對。這些操作只是在同一類直線裡找錯誤少的參數，可能把錯 3 點降成錯 1 點，但這一類直線裡沒有全對的答案，最低錯誤是 1 點，不是 0。左圖是示意，不是真實實驗；我另外對係數網格（負 4 到 4，間隔 0.5）窮舉，XOR 最少仍錯 1 點（程式計算）。表達能力是模型的邊界形狀本身能不能表示出答案；訓練有沒有找到最好的線，和最好的線夠不夠用，是兩個不同的問題。",
     "This page addresses a common myth: more rounds, a smaller learning rate or a fresh initialization should get everything right. These only search for fewer errors among the same family of lines, perhaps reducing 3 errors to 1, but that family has no perfect answer and the lowest error is 1, not 0. The left chart is a schematic, not a real experiment; separately I searched a grid of coefficients (minus 4 to 4, step 0.5) and XOR still has at least 1 error (computed by program). Representational capacity is whether the shape of the model's boundary can express the answer at all; whether training finds the best line and whether the best line is good enough are two different questions."),
    ("Schematic error curve falling from 3 to 1 but never reaching the dashed 0 line; grid search over coefficients still leaves at least 1 XOR error.",) * 2)
add(8, ("兩線夾出帶", "Strip"),
    ("要改的是邊界的組合：兩條平行線夾出 0.5 < s ≤ 1.5 的帶，剛好選出兩個 A", "Combine boundaries: two parallel lines make the strip 0.5 < s ≤ 1.5 holding exactly the A points"),
    ("多訓練也補不了，最低錯誤是 1 點，不是 0。能改什麼才能分開 XOR？", "More training cannot help: the lowest error is 1, not 0. What must change for XOR to be separated?"),
    ("這一頁說明要改的是邊界的組合。前面排除的是一個線性分數加固定門檻，沒有排除多條邊界的組合。令 s 等於 x1 加 x2，XOR 四個點的 s 依序是 0、1、1、2，A 恰好是 s 等於 1，所以用兩個判斷的交集選出：預測為 1 若且唯若 0.5 小於 s 且 s 小於等於 1.5（補充，程式計算）。這只驗證四個布林點，沒有規定連續平面上 XOR 的唯一延伸。改變的是模型允許的邊界組合，不是換一組舊係數。下一個知識點問：兩個判斷要怎麼用神經元做出來？",
     "This page shows that what must change is the combination of boundaries. What was ruled out is one linear score with a fixed threshold, not combinations of several boundaries. Let s equal x1 plus x2; the s values of the four XOR points are 0, 1, 1, 2 and A is exactly s equal to 1, so select it with the intersection of two judgments: the prediction is 1 if and only if 0.5 is below s and s is at most 1.5 (supplementary, computed by program). This verifies only the four Boolean points and does not fix a unique extension of XOR to the continuous plane. What changes is the set of boundary combinations the model allows, not a new set of old coefficients. The next KP asks how neurons build the two judgments."),
    ("Strip 0.5 < x1 + x2 <= 1.5 between two parallel lines; s values 0, 1, 1, 2; the two A points inside, the two B points outside.",) * 2)
add(9, ("延伸：實作", "Extension: Code"),
    ("實作要測邊界上的點：g = x₁ − x₂ 用 > 錯 1 點，用 ≥ 錯 3 點", "Test points on the boundary: with g = x₁ − x₂, > gives 1 error but ≥ gives 3"),
    ("兩條平行線夾出 0.5 < s ≤ 1.5 的帶。邊界上的點用 > 或 ≥ 會得到相同的預測嗎？", "Two parallel lines make the strip 0.5 < s ≤ 1.5. Do > and ≥ give the same predictions for points exactly on a boundary?"),
    ("這是實作的提醒。逐點算出分數後，用 score 大於 0 產生預測，再與真值比較。測試必須包含邊界上的點，免得程式用大於等於而文字用大於。以 g 等於 x1 減 x2 為例，(0,0) 與 (1,1) 的分數剛好是 0：用大於，預測 0、0、1、0，錯 1 點；用大於等於，預測 1、0、1、1，錯 3 點（程式計算）。這裡的半整數可精確表示；一般實數資料仍要注意浮點誤差。",
     "This is an implementation reminder. After computing the scores point by point, produce predictions with score above 0 and compare them with the truth. Tests must include points on the boundary so that code using greater-or-equal does not disagree with text using greater-than. Take g equal to x1 minus x2: the scores at (0,0) and (1,1) are exactly 0; with greater-than the predictions are 0, 0, 1, 0 with 1 error; with greater-or-equal they are 1, 0, 1, 1 with 3 errors (computed by program). Half-integers are exact here; with general real data watch for floating-point error."),
    ("g = x1 - x2: the points (0,0) and (1,1) lie on the boundary with score 0; using > gives 1 error and using >= gives 3 errors.",) * 2)
add(10, ("延伸：小練習一", "Extension: Task 1"),
    ("練習 1：g = x₁ + x₂ − 1.5 對 AND 全對，對 XOR 錯 3 點", "Exercise 1: g = x₁ + x₂ − 1.5 classifies AND perfectly but gets 3 XOR points wrong"),
    ("測邊界上的點很重要：用 > 錯 1 點、用 ≥ 錯 3 點。同一條線 g = x₁ + x₂ − 1.5 拿去核對 AND 與 XOR，會是什麼？", "Testing points on the boundary matters: > gives 1 error but ≥ gives 3. What does the line g = x₁ + x₂ − 1.5 give against AND and XOR?"),
    ("這是第一題小練習：用 g 等於 x1 加 x2 減 1.5 算四個輸入，分別核對 AND 與 XOR。解題示範：符號式如上，g 大於 0 判 A；代入四點，分數依序是負 1.5、負 0.5、負 0.5、0.5；只有 (1,1) 為正，預測 B、B、B、A。核對：AND 的真值是 B、B、B、A，全對；XOR 的 (0,1)、(1,0) 被判 B、(1,1) 被判 A，錯 3 點（程式計算）。",
     "This is the first exercise: compute the four inputs with g equal to x1 plus x2 minus 1.5 and check against AND and XOR. Solution: the formula is as above and g above 0 means A; substituting the four points gives scores minus 1.5, minus 0.5, minus 0.5, 0.5; only (1,1) is positive, so the predictions are B, B, B, A. Check: the AND truth is B, B, B, A, all right; for XOR, (0,1) and (1,0) are called B and (1,1) is called A, 3 wrong (computed by program)."),
    ("Exercise 1: g = x1 + x2 - 1.5 gives scores -1.5, -0.5, -0.5, 0.5 and predictions B, B, B, A; AND is all right; XOR has 3 errors.",) * 2)
add(11, ("延伸：小練習二三", "Extension: Task 2-3"),
    ("練習 2、3：XOR 失敗是因為邊界形狀受限；三者中只有 OR 能用一條線分開", "Exercises 2 and 3: XOR fails because the boundary shape is limited; only OR is separable by one line"),
    ("g = x₁ + x₂ − 1.5 對 AND 全對、對 XOR 錯 3 點。XOR 失敗的原因是什麼，OR、XOR、XNOR 又哪個能用一條線分開？", "The line g = x₁ + x₂ − 1.5 gets AND right and XOR 3 wrong. Why does XOR fail, and which of OR, XOR, XNOR can one line separate?"),
    ("這是第二、三題小練習。題 2：XOR 失敗是樣本少、雜訊、學習率，還是邊界形狀受限？逐一排除：四個輸入就是全部資料；標籤直接取自 XOR 真值沒有錯；矛盾對任何係數都成立，與學習率無關；剩下的只有一條線這個邊界形狀受限。題 3：OR、XOR、XNOR 哪個能用一條線分開？OR 可以，取 g 等於 x1 加 x2 減 0.5，預測 B、A、A、A，與 OR 相同；XNOR 不行，同類仍在對角，只是標籤互換，同樣的矛盾還在；結論是三者中只有 OR 能用一條線分開（程式窮舉驗證）。",
     "These are exercises 2 and 3. Q2: does XOR fail because of few samples, noise, the learning rate or a limited boundary shape? Excluding one by one: the four inputs are all the data; the labels come straight from XOR and none is wrong; the contradiction holds for any coefficients, so the learning rate is irrelevant; only the limited boundary shape of a single line remains. Q3: which of OR, XOR, XNOR can one line separate? OR can, with g equal to x1 plus x2 minus 0.5 predicting B, A, A, A like OR; XNOR cannot, since same-class points still sit on the diagonals with only the labels swapped, so the same contradiction remains; the conclusion is that only OR is separable by one line (verified by exhaustive search)."),
    ("Exercise 2: excluding few samples, noise and learning rate leaves a limited boundary shape; exercise 3: OR is separable, XOR and XNOR are not.",) * 2)
PAGES = P
