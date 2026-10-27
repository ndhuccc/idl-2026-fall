import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE; SK = os.environ.get("COURSE_SKILLS") or (os.path.join(os.path.abspath(os.environ["COURSE_REPO"]), ".agents", "skills") if os.environ.get("COURSE_REPO") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
for p in ("view-from-text-v3/tools", "view-from-text-v2/tools", "view-from-text-v2/examples"): sys.path.insert(0, os.path.join(SK, p))
sys.path.insert(0, HERE)
import kp03_pages as KP
from htmlkit import native
JD = os.path.join(HERE, "out")
UNIT_ZH, UNIT_EN = "多條線的編碼與兩層網路的極限", "Codes of several lines and the limit of two-layer networks"
TITLE_ZH = "圖型識別 第四章 KP03：多條線的編碼與兩層網路的極限（大三）"; TITLE_EN = "Pattern Recognition, Chapter 4, KP03: codes of several lines and the limit of two-layer networks (undergraduate)"
NOTE_ZH = "教材程度：大學三年級（undergrad）。依講義 lec05-kp03 以 view-from-text-v3 製作：先規劃頁面，每頁以一張原生資訊圖（HTML 設計後轉成原生節點）為主，名詞在同一頁圖內定義。共 11 頁，所有數字皆由程式計算。"
NOTE_EN = "Level: undergraduate (year 3). Made from the lecture lec05-kp03 with view-from-text-v3: pages were planned first and each page is mainly one native infographic (designed in HTML and converted to native nodes), with the terms defined on the same page. 11 pages; every number is computed by program."
def fig(n, alt_zh, alt_en):
    return lambda: native(f"kp03_p{n}", f"lec05-kp03-p{n}", KP.W, KP.H, JD, alt_en if os.environ.get("KP_LANG") == "en" else alt_zh)
P = []
LAB = {1: ("兩個空間", "the two spaces"), 2: ("七塊地", "the seven plots"), 3: ("虛擬碼 101", "the code 101"), 4: ("編碼立方體", "the code cube"), 5: ("逐角計分", "the corner scores"), 6: ("A 區的缺口", "the notch of region A"), 7: ("第二組標籤", "the second labelling"), 8: ("共用中點", "the shared midpoint"), 9: ("再加一層", "the extra layer"), 10: ("超立方體", "the hypercube"), 11: ("小練習", "the exercises")}
def add(n, lab, title, bridge, notes, alt):
    zh, en = notes
    if n > 1: zh = f"承接上一頁的「{LAB[n-1][0]}」。" + zh; en = f"Following the previous page on {LAB[n-1][1]}. " + en
    if n == 11: zh += "到這裡這個單元就結束了；之後請接著讀下一單元。"; en += " That ends this unit; next, move on to the next unit."
    else: zh += f"接下來的下一頁是「{LAB[n+1][0]}」。"; en += f" Next, the next page covers {LAB[n+1][1]}."
    P.append(dict(lab=lab, title=title, bridge=bridge, notes=(zh, en), layout="freeform", spec=fig(n, *alt)))
add(1, ("兩個空間", "Two spaces"),
    ("每個點在三條線上得到三個位元，連起來是編碼，住進立方體的一個角", "Each point gets three bits from three lines; the bits form its code, a corner of the cube"),
    ("KP02 用兩條線把 XOR 變成三個位置，再用最後一層分開。線再多，最後一層還夠用嗎？", "KP02 turned XOR into three positions with two lines and separated them with the last layer. With more lines, is the last layer still enough?"),
    ("這是 KP03 的第一頁。先備三件事：KP02 的做法（每條隱藏線給一個位元，位元連起來是編碼，輸出層在編碼上再切一刀）、立方體的角（三個位元共 2 的 3 次方等於 8 種組合，正好是立方體 8 個角的座標）、平面把空間分成兩邊。這一頁先擺工具：分數是把點代入線的式子得到的數；階梯規則是分數大於 0 記 1，其餘記 0；位元是點在線的正側記 1；一個點的三個位元連起來是編碼。圖上三個點分別落在立方體的角 110、000、011（程式計算）。",
     "This is the first page of KP03. Prerequisites: the method of KP02 (each hidden line gives one bit, the bits form a code, and the output layer cuts once more on the code), the corners of a cube (three bits give 2 to the power 3, that is 8 combinations, exactly the coordinates of the 8 corners), and the fact that a plane splits space into two sides. This page lays out the tools: a score is the number obtained by plugging a point into a line; the step rule gives 1 for a score above 0 and 0 otherwise; a bit is 1 on the positive side of a line; and the three bits of a point in a row form its code. The three points in the figure land at the cube corners 110, 000 and 011 (computed by program)."),
    ("Left: input plane with three lines and three marked points; middle: the code cube with the same points at corners 110, 000, 011; right: definitions of score, step rule, bit and code.",) * 2)
add(2, ("七塊地", "Seven plots"),
    ("三條線切出七塊地，每塊有自己的編碼", "Three lines cut seven plots, each with its own code"),
    ("每個點在三條線上得到三個位元，連起來是編碼，住進立方體的一個角。三條具體的線，會把平面切成幾塊地？", "Each point gets three bits from three lines; the bits form its code, a corner of the cube. How many plots do three concrete lines cut the plane into?"),
    ("這一頁用三條具體的線：g1 等於 x1，g2 等於 x2，g3 等於負 x1 加 x2 減 1；係數是教學選擇，不是原圖的方程式。區域是拿到同一個編碼的所有點合起來的一塊地。例子：點 (1,1) 的分數是 1、1、負 1，位元是 1、1、0（負 1 小於等於 0 記 0），編碼是 110。七種編碼出現，101 不出現（程式計算）。",
     "This page uses three concrete lines: g1 equals x1, g2 equals x2 and g3 equals minus x1 plus x2 minus 1; the coefficients are a teaching choice, not the equations of the original figure. A cell is all the points that get the same code, together forming one plot. For example the point (1,1) scores 1, 1, minus 1, so its bits are 1, 1, 0 (minus 1 is at most 0 and gives 0) and its code is 110. Seven codes occur and 101 does not (computed by program)."),
    ("Input plane cut by three orange lines into seven colored plots labelled by code; g1=x1, g2=x2, g3=-x1+x2-1; example (1,1) has code 110.",) * 2)
add(3, ("虛擬碼 101", "Code 101"),
    ("101 沒人拿得到：三個條件互相矛盾，圖上只有 7 塊地", "No point receives 101: its three conditions contradict, so the plane has only 7 plots"),
    ("三條線切出七塊地，每塊有自己的編碼；三位元能寫 8 種編碼，圖上只有 7 塊。101 為什麼沒人拿得到？", "Three lines cut seven plots, each with its own code; three bits can write 8 codes but the plane has only 7 plots. Why can no point receive 101?"),
    ("這一頁說明 101 為什麼沒人拿得到。編碼 101 要求同時滿足三件事：第 1 位是 1，即 x1 大於 0；第 2 位是 0，即 x2 小於等於 0；第 3 位是 1，即 x2 大於 x1 加 1。前兩個讓 x2 小於等於 0 且小於 x1 加 1，第三個不可能成立。實際碼是至少有一個輸入點會得到的編碼，共 7 個；虛擬碼是寫得出來卻沒有任何輸入點得到的編碼，這裡是 101。這不是資料還沒蒐集到：不論蒐集多少點，101 都不會出現（程式計算）。",
     "This page explains why no point receives 101. The code 101 needs three things at once: bit 1 is 1, that is x1 above 0; bit 2 is 0, that is x2 at most 0; bit 3 is 1, that is x2 above x1 plus 1. The first two make x2 at most 0 and below x1 plus 1, so the third cannot hold. A realized code is one that at least one input point receives, 7 in all; an unrealized code can be written but no input point receives it, here 101. This is not missing data: however many points are collected, 101 never appears (computed by program)."),
    ("Four panels: x1>0 right half, x2<=0 lower half, above the line g3, and all three stacked with empty intersection; definitions of realized and unrealized codes.",) * 2)
add(4, ("編碼立方體", "Code cube"),
    ("把編碼看成立方體的角：最後一層是在角上切一刀，q = 0 是一個平面", "Codes as cube corners: the last layer is one cut, and q = 0 is a plane"),
    ("101 沒人拿得到：三個條件互相矛盾，圖上只有 7 塊地。這些編碼放進立方體，輸出層的一刀要切在哪裡？", "No point receives 101: its three conditions contradict, so the plane has only 7 plots. Put these codes in the cube: where does the output layer's cut go?"),
    ("這一頁把編碼看成立方體的角：三位元的 8 種編碼正好是 8 個角的座標，角的顏色與上一頁平面上各塊地相同，101 是空心圓。輸出分數 q 等於 a1 y1 加 a2 y2 加 a3 y3 加 a0，q 大於 0 判 A，否則判 B；a_i 是每一位的票數，a0 是起始分。q 等於 0 是立方體裡的一個平面，把角分成兩邊，這就是切一刀。條件只需對實際碼成立（程式計算）。",
     "This page treats codes as cube corners: the 8 three-bit codes are exactly the coordinates of the 8 corners, the corner colors match the plots of the previous page, and 101 is a hollow circle. The output score q equals a1 y1 plus a2 y2 plus a3 y3 plus a0, with q above 0 called A and otherwise B; a_i is the vote of each bit and a0 the starting score. q equal to 0 is a plane in the cube that splits the corners into two sides, which is one cut. The condition only has to hold for realized codes (computed by program)."),
    ("Cube with eight corners colored like the plots, 101 hollow; definition of the output score q = a1y1 + a2y2 + a3y3 + a0 and the cut q = 0.",) * 2)
add(5, ("逐角計分", "Corner scores"),
    ("逐角計分：q = −y₁ − y₂ + y₃ + 0.5，A 全在一側、B 全在另一側，這一刀切得開", "Score every corner: q = −y₁ − y₂ + y₃ + 0.5 puts all A on one side and all B on the other"),
    ("把編碼看成立方體的角：最後一層是在角上切一刀，q = 0 是一個平面。一個角一個角代入，這一刀切得開嗎？", "Codes are cube corners: the last layer is one cut, and q = 0 is a plane. Plugging in corner by corner, does this cut separate them?"),
    ("這一頁沿用講義的輸出式 q 等於負 y1 負 y2 加 y3 加 0.5，一個角一個角代入：000 得 0.5、001 得 1.5、011 得 0.5，都判 A；010、100 得負 0.5，110 得負 1.5，111 得負 0.5，都判 B。101 的分數是 0.5，但沒有輸入會得到它，不影響分類。A 全在平面一側、B 全在另一側，叫線性可分（程式計算）。",
     "This page keeps the lecture's output formula q equal to minus y1 minus y2 plus y3 plus 0.5 and substitutes corner by corner: 000 gives 0.5, 001 gives 1.5 and 011 gives 0.5, all A; 010 and 100 give minus 0.5, 110 gives minus 1.5 and 111 gives minus 0.5, all B. The score of 101 is 0.5, but no input gets it, so classification is unaffected. A all on one side of a plane and B all on the other is called linearly separable (computed by program)."),
    ("Cube with the cut plane q=0 and corner scores, A circles on one side and B squares on the other; table of seven corners with scores 0.5, 1.5, -0.5, 0.5, -0.5, -1.5, -0.5.",) * 2)
add(6, ("A 區的缺口", "Notch of region A"),
    ("翻回平面：A 區由三塊地併成，中間有缺口，不是凸的", "Back in the plane: region A is three plots joined, with a notch, so it is not convex"),
    ("逐角計分：A 全在一側、B 全在另一側，這一刀切得開。把這一刀切回輸入平面，A 的區域長什麼樣？", "Score every corner: all A on one side and all B on the other, so the cut separates them. Cut back in the input plane, what does the A region look like?"),
    ("這一頁翻回平面。A 類是 000、001、011 三塊地的聯集，也就是把幾塊區域併成一塊。凸的意思是區域內任取兩點，連線都不跑出區域。取 P(負 0.15, 負 1) 與 Q(負 0.15, 2)：兩點都在 A 區，連線 PQ 卻經過 B 塊的 R(負 0.15, 0.62)，所以 A 區不是凸的。常見迷思是兩層網路只會分類凸區域：單一編碼的區域是凸的，但最後一層可以接受多個編碼，其聯集可以有缺口（程式計算）。",
     "This page goes back to the plane. Class A is the union of the plots 000, 001 and 011, that is, several regions joined into one. Convex means that for any two points of the region the segment between them stays inside. Take P(minus 0.15, minus 1) and Q(minus 0.15, 2): both are in A, yet the segment PQ passes through R(minus 0.15, 0.62) of the B plot, so region A is not convex. A common myth says two-layer networks only classify convex regions: one code's region is convex but the last layer can accept several codes, whose union may have notches (computed by program)."),
    ("Input plane with A region in blue and B region in orange, points P and Q in A and R in B on the dashed segment PQ; definitions of union and convex.",) * 2)
add(7, ("第二組標籤", "Second labelling"),
    ("換一組標籤：底面四個角 A、B 交錯，XOR 又出現了", "A second labelling: the bottom face alternates A and B, and XOR returns"),
    ("A 區由三塊地併成，中間有缺口，不是凸的。換一組標籤，這一刀還切得開嗎？", "Region A is three plots joined with a notch, so it is not convex. With a second labelling, can this cut still separate them?"),
    ("這一頁改成 A 等於 {000, 110, 111}，其餘為 B。立方體底面 y3 等於 0 的四個角：A 在 000 與 110，B 在 010 與 100，沿對角交錯。在輸入平面上，這四個角就是兩條座標軸切出的四個象限，和 KP01 的 XOR 一樣：不可分，也是一個反例（程式計算）。",
     "This page sets A to {000, 110, 111} and the rest to B. On the cube's bottom face y3 equal to 0 the four corners are: A at 000 and 110, B at 010 and 100, alternating along the diagonal. In the input plane these four corners are the four quadrants cut by the two coordinate axes, just like XOR in KP01: nonseparable, and a counterexample (computed by program)."),
    ("Input plane colored by the second labelling, cube with the bottom face framed in red, and the flattened bottom face with A at 000, 110 and B at 010, 100.",) * 2)
add(8, ("共用中點", "Shared midpoint"),
    ("兩條對角線共用中點：兩個 A 的分數和大於 0，兩個 B 的分數和不大於 0，矛盾", "Diagonals share a midpoint: two A scores sum above 0, two B scores at most 0, a contradiction"),
    ("換一組標籤：底面四個角 A、B 交錯，XOR 又出現了。怎麼證明任何一刀都切不開？", "A second labelling: the bottom face alternates A and B, and XOR returns. How do we prove that no cut can separate them?"),
    ("這一頁證明。兩個 A 角相加：q(000) 加 q(110) 等於 a0 加 (a1 加 a2 加 a0)，即 a1 加 a2 加 2 a0，必須大於 0。兩個 B 角相加：q(010) 加 q(100) 等於 (a2 加 a0) 加 (a1 加 a0)，同樣是 a1 加 a2 加 2 a0，必須小於等於 0。同一個式子既要大於 0 又要小於等於 0，矛盾，所以任何 a0、a1、a2 都切不開。幾何意思：兩條對角線交於中點 M，q 是線性的，q(M) 既是兩個 A 分數的平均也是兩個 B 分數的平均（程式計算）。",
     "This page gives the proof. Adding the two A corners: q(000) plus q(110) equals a0 plus (a1 plus a2 plus a0), that is a1 plus a2 plus 2 a0, which must be above 0. Adding the two B corners: q(010) plus q(100) equals (a2 plus a0) plus (a1 plus a0), the same a1 plus a2 plus 2 a0, which must be at most 0. One expression cannot be both above 0 and at most 0, a contradiction, so no a0, a1, a2 separates them. Geometrically the two diagonals meet at the midpoint M and, q being linear, q(M) is both the average of the two A scores and of the two B scores (computed by program)."),
    ("Bottom face with the two diagonals 000-110 in blue and 010-100 in orange crossing at M; proof that a1+a2+2a0 must be both above 0 and at most 0.",) * 2)
add(9, ("再加一層", "Extra layer"),
    ("不是網路不行，是這一刀不夠：每個目標角用一個平面切下來，再加一層就行", "Not hopeless: one cut is not enough, but cutting off each target corner and adding a layer works"),
    ("兩條對角線共用中點，兩個 A 的分數和大於 0，兩個 B 的分數和不大於 0，矛盾。既然這一刀不夠，怎麼辦？", "Two diagonals share a midpoint, the two A scores sum above 0 and the two B scores at most 0, a contradiction. If this cut is not enough, what now?"),
    ("這一頁說明不是網路不行，是這一刀不夠。被擋住的是這組固定的線加上這一刀，不是所有兩層網路：換一組線，編碼不同，反例也會不同。任何一個角都能用一個平面單獨切下來：平面 y1 加 y2 減 y3 等於 1.5 只切下 110，y1 加 y2 加 y3 等於 2.5 只切下 111，負 y1 負 y2 負 y3 等於負 0.5 只切下 000。對每個目標角各切一刀、再把結果併起來，就能認出整個 A 集合，這就是再加一層（程式計算）。",
     "This page explains that the network is not hopeless; this one cut is not enough. What is blocked is this fixed set of lines plus one cut, not every two-layer network: another set of lines gives other codes and other counterexamples. Any corner can be cut off alone by one plane: the plane y1 plus y2 minus y3 equal to 1.5 cuts off only 110, y1 plus y2 plus y3 equal to 2.5 only 111, and minus y1 minus y2 minus y3 equal to minus 0.5 only 000. One cut per target corner, with the results merged, recognises the whole A set, which is one more layer (computed by program)."),
    ("Three cubes, each with one target corner starred (110, 111, 000) and the plane that cuts it off: y1+y2-y3=1.5, y1+y2+y3=2.5, -y1-y2-y3=-0.5.",) * 2)
add(10, ("延伸：超立方體", "Extension: hypercube"),
    ("每多一個位元，角數加倍：p 條線給 2ᵖ 個角", "Each extra bit doubles the corners: p lines give 2ᵖ corners"),
    ("不是網路不行，是這一刀不夠：每個目標角用一個平面切下來，再加一層就行。線有 p 條時，編碼住在哪裡？", "Not hopeless: one cut is not enough, but cutting off each target corner and adding a layer works. With p lines, where do the codes live?"),
    ("這一頁是延伸：一般寫法。輸入有 d 個座標、有 p 條線，編碼 y 等於 f(g1(x)) 到 f(gp(x)) 組成的向量，屬於 {0,1} 的 p 次方，有 2 的 p 次方個角，是 p 維超立方體的角。一條線段有 2 個角，正方形有 4 個，立方體有 8 個，每多一位元角數加倍。本講義的例子 d 等於 2、p 等於 3（程式計算）。",
     "This page is the extension: the general form. With d input coordinates and p lines, the code y is the vector of f(g1(x)) to f(gp(x)), lying in {0,1} to the power p, which has 2 to the power p corners, the corners of a p-dimensional hypercube. A segment has 2 corners, a square 4 and a cube 8, doubling with every extra bit. This unit's example has d equal to 2 and p equal to 3 (computed by program)."),
    ("A segment with 2 corners, a square with 4 and a cube with 8; the general code y in {0,1}^p and the step function f.",) * 2)
add(11, ("延伸：小練習", "Extension: exercises"),
    ("兩題小練習：101 不影響預測；只改 a₃ 消除不了底面的矛盾", "Two exercises: 101 changes no prediction; changing only a₃ cannot remove the contradiction"),
    ("每多一個位元，角數加倍，p 條線給 2ᵖ 個角。用虛擬碼和底面四個角，自己算一次，結果是什麼？", "Each extra bit doubles the corners: p lines give 2ᵖ corners. Work it out yourself with the unrealized code and the four bottom corners: what is the result?"),
    ("兩題小練習。題 1：101 要求 g1 等於 x1 大於 0、g2 等於 x2 小於等於 0、g3 大於 0；由第三式 x2 大於 x1 加 1 大於 1，與 x2 小於等於 0 矛盾，沒有任何輸入得到 101，所以把 101 指定為 A 或 B 都不影響預測。題 2：兩個和都是 a1 加 a2 加 2 a0；y3 等於 0 時 a3 乘 0 等於 0，a3 消失，矛盾依舊；取 a1 等於 a2 等於 0、a0 等於 1，四個角的分數都是 1，B 被判成 A，與 a3 無關（程式計算）。",
     "Two exercises. Q1: 101 needs g1 equal to x1 above 0, g2 equal to x2 at most 0 and g3 above 0; the third gives x2 above x1 plus 1, above 1, which contradicts x2 at most 0, so no input gets 101 and assigning 101 to A or B changes no prediction. Q2: both sums are a1 plus a2 plus 2 a0; with y3 equal to 0, a3 times 0 is 0, so a3 drops out and the contradiction stays; with a1 equal to a2 equal to 0 and a0 equal to 1, all four corners score 1 and B is called A whatever a3 is (computed by program)."),
    ("Exercise 1: no input gets 101 so its label changes no prediction; exercise 2: both sums equal a1+a2+2a0 independent of a3, so the contradiction stays.",) * 2)
PAGES = P
