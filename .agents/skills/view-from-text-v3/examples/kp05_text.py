import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE; SK = os.environ.get("COURSE_SKILLS") or (os.path.join(os.path.abspath(os.environ["COURSE_REPO"]), ".agents", "skills") if os.environ.get("COURSE_REPO") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
for p in ("view-from-text-v3/tools", "view-from-text-v2/tools", "view-from-text-v2/examples"): sys.path.insert(0, os.path.join(SK, p))
sys.path.insert(0, HERE)
import kp05_pages as KP
from htmlkit import native
JD = os.path.join(HERE, "out")
UNIT_ZH, UNIT_EN = "構造式方法", "Constructive methods"
TITLE_ZH = "圖型識別 第四章 KP05：構造式方法（大三）"; TITLE_EN = "Pattern Recognition, Chapter 4, KP05: constructive methods (undergraduate)"
NOTE_ZH = "教材程度：大學三年級（undergrad）。依講義 lec05-kp05 以 view-from-text-v3 製作：先規劃頁面，每頁以一張原生資訊圖（HTML 設計後轉成原生節點）為主，名詞在同一頁圖內定義。共 10 頁，所有數字皆由程式計算。"
NOTE_EN = "Level: undergraduate (year 3). Made from the lecture lec05-kp05 with view-from-text-v3: pages were planned first and each page is mainly one native infographic (designed in HTML and converted to native nodes), with the terms defined on the same page. 10 pages; every number is computed by program."
def fig(n, alt_zh, alt_en):
    return lambda: native(f"kp05_p{n}", f"lec05-kp05-p{n}", KP.W, KP.H, JD, alt_en if os.environ.get("KP_LANG") == "en" else alt_zh)
P = []
LAB = {1: ("主要節點分兩群", "the master unit's two groups"), 2: ("混雜群", "the mixed group"), 3: ("輔助節點", "the ancillary unit"), 4: ("純化後的編碼", "the purified codes"), 5: ("新編碼平面", "the new code plane"), 6: ("停止的理由", "why it stops"), 7: ("停止的條件", "the stopping conditions"), 8: ("垂直平分線", "the perpendicular bisector"), 9: ("三層足夠", "three layers suffice"), 10: ("小練習", "the exercises")}
def add(n, lab, title, bridge, notes, alt):
    zh, en = notes
    if n > 1: zh = f"承接上一頁的「{LAB[n-1][0]}」。" + zh; en = f"Following the previous page on {LAB[n-1][1]}. " + en
    if n == 10: zh += "到這裡這個單元就結束了；之後請接著讀下一單元。"; en += " That ends this unit; next, move on to the next unit."
    else: zh += f"接下來的下一頁是「{LAB[n+1][0]}」。"; en += f" Next, the next page covers {LAB[n+1][1]}."
    P.append(dict(lab=lab, title=title, bridge=bridge, notes=(zh, en), layout="freeform", spec=fig(n, *alt)))
add(1, ("主要節點分兩群", "Master unit"),
    ("主要節點 z₁ = f(x₁ + x₂ − 0.5) 把四點分成兩群：(0,0) 在 0 群，其餘三點在 1 群", "One master unit splits four points: (0,0) falls in the 0 group, the other three in the 1 group"),
    ("實務只有帶標籤的點，沒有目標區域的方程式。要從點開始建網路，第一步做什麼？", "In practice we only have labelled points, not the equation of the target region. To build a network from them, what is the first step?"),
    ("這是 KP05 的第一頁。先備三件事：階梯單元把分數變成 0 或 1 的位元、編碼與純度（兩個不同類的點若編碼完全相同，輸出層就分不開）、XOR 四點（0,0 與 1,1 是 B，0,1 與 1,0 是 A）。KP04 的構造知道目標區域的方程式，實務上卻只有點與標籤。這一頁取 z1 等於 f(x1 加 x2 減 0.5)：(0,0) 的分數是負 0.5，位元 0；(0,1) 與 (1,0) 是 0.5，(1,1) 是 1.5，位元都是 1。所以 0 群有 1 個點，1 群有 3 個點（程式計算）。",
     "This is the first page of KP05. Prerequisites: a step unit turns a score into a bit 0 or 1; codes and purity (if two points of different classes have exactly the same code, the output layer cannot separate them); and the four XOR points (0,0 and 1,1 are B, 0,1 and 1,0 are A). The construction of KP04 knew the equation of the target region, but in practice we only have points and labels. This page takes z1 equal to f of x1 plus x2 minus 0.5: (0,0) scores minus 0.5, bit 0; (0,1) and (1,0) score 0.5 and (1,1) scores 1.5, all bit 1. So the 0 group has 1 point and the 1 group has 3 points (computed by program)."),
    ("The four XOR points with the master unit's line x1+x2=0.5; table of scores -0.5, 0.5, 0.5, 1.5 and bits 0,1,1,1.",
     "The four XOR points with the master unit's line x1+x2=0.5; table of scores -0.5, 0.5, 0.5, 1.5 and bits 0,1,1,1."))
add(2, ("混雜群", "Mixed group"),
    ("主要節點只分對 4 點中的 3 點：1 群裡有兩個 A 和一個 B，混雜 3 點", "The master unit gets 3 of 4 right: its 1 group holds two A and one B, 3 mixed points"),
    ("主要節點把四點分成 0 群與 1 群，(0,0) 在 0 群，其餘三點在 1 群。這樣分得完全對嗎？", "The master unit split four points into a 0 group and a 1 group, with (0,0) in the 0 group and the other three in the 1 group. Is that split fully right?"),
    ("這一頁逐群檢查標籤。0 群只有 (0,0)，真值是 B，判 B 對。1 群的三個點一律判 A：(0,1) 與 (1,0) 是 A，對；(1,1) 是 B，錯。所以答對 3 點、答錯 1 點。混雜群是同一個輸出群裡兩個類別都有；這裡的混雜群有 3 個點（程式計算）。",
     "This page checks the labels group by group. The 0 group holds only (0,0), truth B, called B, right. The three points of the 1 group are all called A: (0,1) and (1,0) are A and right, (1,1) is B and wrong. So 3 points are right and 1 is wrong. A mixed group is an output group containing both classes; here it has 3 points (computed by program)."),
    ("The 1 group's three points ringed in red; 0 group right; (0,1),(1,0) right and (1,1) wrong; 3 of 4 right, mixed group of 3 points.",) * 2)
add(3, ("輔助節點", "Ancillary unit"),
    ("輔助線 x₁ + x₂ = 1.5 把混雜群裡右上角的 (1,1) 分出來：z₂ 是 1、1、0", "The ancillary line x₁ + x₂ = 1.5 splits off (1,1), the upper-right B: z₂ is 1, 1, 0"),
    ("主要節點只分對 3 點，1 群裡有兩個 A 和一個 B。怎樣把右上角的 B 分出來？", "The master unit got only 3 points right, and its 1 group holds two A and one B. How do we split off the upper-right B?"),
    ("這一頁為了示範，手動選定一條輔助線：z2 等於 f(1.5 減 x1 減 x2)，線 x1 加 x2 等於 1.5 剛好夾在 (1,1) 與 (0,1)、(1,0) 之間。(0,1) 與 (1,0) 得 0.5、位元 1；(1,1) 得負 0.5、位元 0。(0,0) 不在混雜群、不參與訓練，它的 z2 是 1 只是順便算出的值。這是機制示範，不是實際跑過某個訓練演算法（程式計算）。",
     "For the demonstration this page picks one ancillary line by hand: z2 equal to f of 1.5 minus x1 minus x2, and the line x1 plus x2 equal to 1.5 lies between (1,1) and (0,1), (1,0). (0,1) and (1,0) score 0.5 and bit 1; (1,1) scores minus 0.5 and bit 0. (0,0) is not in the mixed group and takes no part in training; its z2 equal to 1 is merely computed on the side. This shows the mechanism and is not the result of running a training algorithm (computed by program)."),
    ("Master line and ancillary dashed line x1+x2=1.5; z2 table: (0,1) 0.5 bit 1, (1,0) 0.5 bit 1, (1,1) -0.5 bit 0, (0,0) 1.5 bit 1 unused.",) * 2)
add(4, ("純化後的編碼", "Purified codes"),
    ("加上輔助位元後四點的編碼是 01、11、11、10，不同類不再共用同一個碼", "With the extra bit the codes are 01, 11, 11, 10, and no two classes share a code"),
    ("輔助線 x₁ + x₂ = 1.5 把 (1,1) 分出來，z₂ 是 1、1、0。兩個位元合起來，每個點的編碼變成什麼？", "The ancillary line x₁ + x₂ = 1.5 splits off (1,1), with z₂ equal to 1, 1, 0. Put the two bits together: what is the code of each point?"),
    ("這一頁把兩個位元接起來：(0,0) 是 01，(0,1) 與 (1,0) 是 11，(1,1) 是 10。純化是新增位元，讓不同類的點不再共用完全相同的編碼。只看 z1，混雜群有 3 點；看完整編碼，三組 01、11、10 全純，混雜 0 點。純化不要求一類只有一個碼：B 佔 01 與 10 兩個碼沒有關係（程式計算）。",
     "This page joins the two bits: (0,0) is 01, (0,1) and (1,0) are 11, and (1,1) is 10. Purification adds bits so points of different classes no longer share exactly the same code. With z1 alone the mixed group has 3 points; with the full code the three groups 01, 11, 10 are all pure, so 0 points are mixed. Purification does not require one code per class: B using both 01 and 10 is fine (computed by program)."),
    ("Table of codes 01, 11, 11, 10 with truths B, A, A, B; groups by z1 alone has 3 mixed points, groups by full code has 0.",) * 2)
add(5, ("新編碼平面", "New code plane"),
    ("在新編碼 (z₁, z₂) 上，一條線 z₁ + z₂ = 1.5 就分對四個點", "On the new codes (z₁, z₂), one line z₁ + z₂ = 1.5 classifies all four points"),
    ("四點的編碼是 01、11、11、10，不同類不再共用同一個碼。有了新編碼，還要再加節點嗎？", "The codes of the four points are 01, 11, 11, 10, and no two classes share a code. With the new codes, do we still need more units?"),
    ("這一頁在新的 (z1, z2) 平面上看：兩個 A 合併在 11，B 在 01 與 10，00 沒有點。新的主要節點取 z1 加 z2 減 1.5，大於 0 判 A：11 得 0.5 判 A，01 與 10 得負 0.5 判 B。四個點全對；兩個 A 合併在 11 沒有關係，因為同類（程式計算）。",
     "This page looks at the new (z1, z2) plane: the two A merge at 11, B sits at 01 and 10, and 00 has no point. The new master unit takes z1 plus z2 minus 1.5 and calls A above 0: 11 scores 0.5 and is A, 01 and 10 score minus 0.5 and are B. All four points are right; merging the two A at 11 is fine because they are the same class (computed by program)."),
    ("The (z1,z2) plane with B at 01 and 10, two A merged at 11, no point at 00, and the line z1+z2=1.5; scores 0.5, -0.5, -0.5; 4 of 4 right.",) * 2)
add(6, ("停止的理由", "Why it stops"),
    ("會停：混雜群從 3 點降到 0 點，答對的點從 3 增加到 4，而點數有上限 4", "It stops: the mixed group shrinks from 3 to 0, correct points rise from 3 to 4, capped at N = 4"),
    ("新編碼 (z₁, z₂) 上，一條線 z₁ + z₂ = 1.5 就分對四個點。這個過程一定會停嗎？", "On the new codes (z₁, z₂), one line z₁ + z₂ = 1.5 classifies all four points. Will this process always stop?"),
    ("這一頁說明停止的理由。進展一：同一層裡還需要分辨的混雜群嚴格變小，本例從 3 點降到 0 點。進展二：在適當的層間構造下，新主要節點保留原來答對的點並至少多對一點；令 Ck 為第 k 層主要節點答對的點數，則 C(k+1) 不小於 Ck 加 1，且 Ck 不超過 N；本例從 3 增加到 4。這個條件是所引構造要求的，不是本例驗出的一般結論，本頁不證明。整數不可能一直增加，所以會停。注意 3 到 0 數的是混雜群的點數，不是錯誤率（程式計算）。",
     "This page gives the reason for stopping. Progress one: within a layer the mixed group still to be resolved shrinks strictly, here from 3 points to 0. Progress two: under a suitable construction between layers, the new master unit keeps the points already right and gets at least one more; let Ck be the number of points the layer-k master units get right, then C(k+1) is at least Ck plus 1 and Ck is at most N; here it rises from 3 to 4. This condition is required by the cited construction and is not a general conclusion verified by this example, and this page does not prove it. An integer cannot keep increasing, so it stops. Note that 3 to 0 counts mixed-group points, not an error rate (computed by program)."),
    ("Two bar charts: mixed-group points 3 to 0, and correct points 3 to 4 against the dashed line N=4; condition C(k+1) >= C(k)+1 and C(k) <= N.",) * 2)
add(7, ("停止的條件", "Stopping conditions"),
    ("停止有條件：標籤不能矛盾，而且訓練全對不保證新資料也對", "Stopping has conditions: consistent labels, and all-correct training does not guarantee new data"),
    ("混雜群從 3 點降到 0 點，答對的點從 3 增加到 4，點數上限是 4，所以會停。這個停止的理由什麼時候不成立？", "The mixed group shrank from 3 to 0 and correct points rose from 3 to 4, capped at N = 4, so it stops. When does this reason for stopping fail?"),
    ("這一頁說停止的條件。第一，標籤不能互相矛盾：兩筆完全相同的輸入被標成不同類時，確定性的網路對同一個輸入只有一個輸出，不可能兩筆都對。第二，節點要依規則增加才有嚴格進展；隨便增加神經元不保證會停。常見迷思是既然最後能把訓練點全分對，新資料也一定對；全對只涉及給定的這批點，不保證沒看過的位置也對，也不保證網路最小。泛化指的就是對沒看過的資料也判對的能力。",
     "This page gives the conditions for stopping. First, labels must not contradict: when two identical inputs carry different classes, a deterministic network has only one output for that input, so it cannot be right on both. Second, units must be added by rule for strict progress; adding neurons at random does not guarantee a stop. A common myth says that if the training points end up all right, new data must be right too; all right concerns only the given points and does not guarantee unseen places, nor a minimal network. Generalization is the ability to classify data not seen in training."),
    ("A circle A and a square B at the same position into a deterministic network with one output; conditions: consistent labels, rule-based growth; myth that all-correct training means correct on new data.",) * 2)
add(8, ("垂直平分線", "Bisector line"),
    ("另一條路：兩個訓練點 u、v 的垂直平分線 x₁ = 0.5，線上每點到兩者等距", "Another route: the bisector x₁ = 0.5 of points u and v is equidistant from both"),
    ("標籤不能矛盾，而且訓練全對不保證新資料也對。除了一層層長網路，還有沒有另一條路，用訓練點的距離直接分？", "Stopping has conditions: consistent labels, and all-correct training does not guarantee new data. Is there another route that splits by distance between training points?"),
    ("這一頁是延伸：最近鄰的另一條路，用兩個訓練點的垂直平分線比距離。令 u、v 為兩個訓練點，等距條件展開後共同的二次項消掉，得 2 乘 (v 減 u) 轉置 x 等於 v 的長度平方減 u 的長度平方。取 u 為 (0,0)、v 為 (1,0)，得 2 x1 等於 1，即 x1 等於 0.5；線上的點 (0.5, 0.7) 到兩者的距離都是 0.860。N 個點最多有 N(N 減 1)/2 個點對：4、10、100 個點是 6、45、4950；這不表示都要保留，也不是最小網路大小。這只是兩點的距離比較，不是完整的最近鄰網路實驗（程式計算）。",
     "This page is an extension: another route, nearest neighbour, comparing distances through the perpendicular bisector of two training points. Let u and v be two training points; expanding the equal-distance condition cancels the common quadratic term and gives 2 times (v minus u) transpose x equal to the squared length of v minus the squared length of u. Take u as (0,0) and v as (1,0): 2 x1 equal to 1, that is, x1 equal to 0.5; the point (0.5, 0.7) on the line is 0.860 from both. N points have at most N(N minus 1)/2 pairs: 6, 45 and 4950 for 4, 10 and 100 points; this does not mean all are kept, nor is it a minimal network size. This is only a distance comparison of two points, not a full nearest-neighbour network experiment (computed by program)."),
    ("Training points u=(0,0), v=(1,0) with bisector x1=0.5 and a point (0.5,0.7) at distance 0.860 from both; derivation 2x1=1; pair counts 6, 45, 4950.",) * 2)
add(9, ("三層足夠", "Three layers suffice"),
    ("三層足夠不等於最多三層：存在性成立，但增長的程序沒有層數上限", "Three layers suffice, not at most three: the growing procedure has no depth bound"),
    ("兩個訓練點 u、v 的垂直平分線 x₁ = 0.5，線上每點到兩者等距。這樣的網路最多需要幾層？", "The bisector x₁ = 0.5 of points u and v is equidistant from both. How many layers does such a network need at most?"),
    ("這一頁區分存在性與程序。第一層之後的輸入是二元編碼，用 KP04 的偵測器加或合併就能分類純化後的編碼，說明適當的三層表示足夠；但一層一層增長的程序並不保證在三層內停止。存在性與程序的深度保證不是同一個定理。",
     "This page separates existence from procedure. After the first layer the inputs are binary codes, and KP04's detectors with an OR-merge classify the purified codes, which shows that a suitable three-layer representation suffices; but the layer-by-layer growing procedure is not guaranteed to stop within three layers. Existence and a depth guarantee for the procedure are not the same theorem."),
    ("Block diagram input, layer 1 master and ancillary units, binary codes, detectors with OR-merge, output; existence of three layers versus no depth bound for the procedure.",) * 2)
add(10, ("小練習", "Exercises"),
    ("兩題小練習：完整編碼分組後混雜 0 點；矛盾標籤不可能兩筆都對", "Two exercises: the full code leaves 0 mixed points; contradictory labels cannot both be right"),
    ("三層足夠不等於最多三層：存在性成立，但增長的程序沒有層數上限。用編碼分組和矛盾標籤，自己算一次，結果是什麼？", "Three layers suffice, not at most three: the growing procedure has no depth bound. Work it out yourself with code groups and contradictory labels: what is the result?"),
    ("兩題小練習。題 1：依完整編碼分組，01 是 (0,0)，B；11 是 (0,1) 與 (1,0)，A、A；10 是 (1,1)，B；全純，混雜 0 點。只看 z1 的混雜是 3 點；3 到 0 數的是還混雜的點，錯誤率數答錯的點，主要節點單獨答錯 1 點。題 2：相同輸入得到相同輸出，確定性網路只能給一個答案，不可能同時對 A 與 B（程式計算）。",
     "Two exercises. Q1: grouped by the full code, 01 is (0,0), B; 11 is (0,1) and (1,0), A, A; 10 is (1,1), B; all pure, 0 mixed points. With z1 alone 3 points are mixed; 3 to 0 counts the still-mixed points, while the error rate counts wrong points, and the master unit alone gets 1 wrong. Q2: the same input gives the same output, so a deterministic network gives one answer and cannot be right on both A and B (computed by program)."),
    ("Exercise 1: groups 01 B, 11 A A, 10 B, all pure, 0 mixed points; Exercise 2: contradictory labels cannot both be right.",) * 2)
PAGES = P
