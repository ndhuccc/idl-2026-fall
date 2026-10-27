import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE; SK = "/home/user/pr-2026-fall/.agents/skills"
for p in ("view-from-text-v3/tools", "view-from-text-v2/tools", "view-from-text-v2/examples"): sys.path.insert(0, os.path.join(SK, p))
sys.path.insert(0, HERE)
import kp15_pages as KP
from htmlkit import native
JD = os.path.join(HERE, "out")
UNIT_ZH, UNIT_EN = "多項式分類器", "The polynomial classifier"
TITLE_ZH = "圖型識別 第四章 KP15：多項式分類器（大三）"; TITLE_EN = "Pattern Recognition, Chapter 4, KP15: the polynomial classifier (undergraduate)"
NOTE_ZH = "教材程度：大學三年級（undergrad）。依講義 lec05-kp15 以 view-from-text-v3 製作：先規劃頁面，每頁以一張原生資訊圖（HTML 設計後轉成原生節點）為主，名詞在同一頁圖內定義。共 8 頁，所有數字皆由程式計算。"
NOTE_EN = "Level: undergraduate (year 3). Made from the lecture lec05-kp15 with view-from-text-v3: pages were planned first and each page is mainly one native infographic (designed in HTML and converted to native nodes), with the terms defined on the same page. 8 pages; every number is computed by program."
def fig(n, alt_zh, alt_en):
    return lambda: native(f"kp15_p{n}", f"lec05-kp15-p{n}", KP.W, KP.H, JD, alt_en if os.environ.get("KP_LANG") == "en" else alt_zh)
P = []
NXT = {1: ("用交互項解 XOR", "solving XOR with the interaction term"), 2: ("項數公式", "the term-count formula"), 3: ("項數爆炸與記憶體", "the term explosion and memory"), 4: ("雙月資料實驗", "the two-moons experiment"), 5: ("階數與過度配適", "degree and overfitting"), 6: ("核方法預告", "the kernel preview"), 7: ("小練習", "the exercises")}
PREV = {2: ("兩個輸入的二階展開", "the second-order expansion of two inputs"), 3: ("用交互項解 XOR", "solving XOR with the interaction term"), 4: ("項數公式", "the term-count formula"), 5: ("項數爆炸與記憶體", "the term explosion and memory"), 6: ("雙月資料實驗", "the two-moons experiment"), 7: ("階數與過度配適", "degree and overfitting"), 8: ("核方法預告", "the kernel preview")}
def add(n, lab, title, bridge, notes, alt):
    zh, en = notes
    if n in PREV: zh = f"承接上一頁的「{PREV[n][0]}」。" + zh; en = f"Following the previous page on {PREV[n][1]}. " + en
    if n == 8: zh += "到這裡這個單元就結束了；之後請接著讀下一單元。"; en += " That ends this unit; next, move on to the next unit."
    if n in NXT: zh += f"接下來的下一頁是「{NXT[n][0]}」。"; en += f" Next, the next page covers {NXT[n][1]}."
    P.append(dict(lab=lab, title=title, bridge=bridge, notes=(zh, en), layout="freeform", spec=fig(n, *alt)))
add(1, ("二階展開", "Second-order expansion"),
    ("兩個輸入的二階多項式有 6 項：偏置、兩個一次項、一個交互項、兩個平方項", "A second-order polynomial of two inputs has 6 terms: bias, two linear, one interaction, two square terms"),
    ("線性分類器只看一次項。若把乘積也當成新座標，判別函數能多出什麼？", "A linear classifier sees only first-order terms. What do we gain if products also count as new coordinates?"),
    ("這是 KP15 的第一頁。先備是線性判別函數與最小二乘（KP13）。多項式分類器：判別函數用最高 r 階的多項式，也就是把次數總和不超過 r 的乘積項當作新座標，再學一個線性分類器。兩個輸入、二階時，g(x) 等於 w0 加 w1 x1 加 w2 x2 加 w12 x1 x2 加 w11 x1 平方加 w22 x2 平方，共 6 項：偏置 1 項、一次項 2 項、交互項 1 項、平方項 2 項。交互項是兩個以上不同輸入相乘的項。把新座標 y 記成 [x1, x2, x1x2, x1 平方, x2 平方]，g 對 6 個權重仍然是線性，所以前面的線性方法都能直接套用。",
     "This is the first page of KP15. The prerequisite is the linear discriminant function and least squares (KP13). A polynomial classifier uses a polynomial of degree at most r as the discriminant, that is, it takes the product terms whose degrees sum to at most r as new coordinates and then learns a linear classifier. With two inputs and second order, g(x) equals w0 plus w1 x1 plus w2 x2 plus w12 x1 x2 plus w11 x1 squared plus w22 x2 squared, 6 terms: 1 bias, 2 linear, 1 interaction and 2 square terms. An interaction term multiplies two or more different inputs. Writing the new coordinates y as [x1, x2, x1x2, x1 squared, x2 squared], g is still linear in the 6 weights, so every linear method from before applies directly."),
    ("Six blocks: bias 1, linear x1 and x2, interaction x1x2 (highlighted), squares x1^2 and x2^2; g = w0 + w1x1 + w2x2 + w12x1x2 + w11x1^2 + w22x2^2, 6 terms.",) * 2)
add(2, ("XOR 與交互項", "XOR and interaction"),
    ("加上交互項 x₁x₂，XOR 就能一條線分開：g = ¼ − 2uv，雙曲線分出兩區", "With the interaction term x1x2, XOR splits with one line: g = ¼ − 2uv, a hyperbola separates two regions"),
    ("兩個輸入的二階展開多了交互項 x₁x₂。XOR 這種直線分不開的資料，它能分開嗎？", "The second-order expansion of two inputs gained the interaction term x1x2. Can it separate XOR data that no straight line separates?"),
    ("這一頁用 XOR 看交互項的威力。四個點 (0,0) 與 (1,1) 屬 B 類，(0,1) 與 (1,0) 屬 A 類，沒有任何一條直線能分開。取 u 等於 x1 減 ½、v 等於 x2 減 ½，判別函數 g 等於 ¼ 減 2uv；回到 x，g 等於負 ¼ 加 x1 加 x2 減 2 x1 x2。程式算出 g(0,0)、g(0,1)、g(1,0)、g(1,1) 依序是 負 0.25、0.75、0.75、負 0.25，所以 g 大於 0 為 A、小於 0 為 B，四點全對。左邊是舉升後的座標表，右邊是雙曲線切出的兩個區域，曲線的邊界在原本的座標裡是彎的，但在 6 維新座標裡是一個平面。",
     "This page uses XOR to see the power of the interaction term. Points (0,0) and (1,1) are class B, (0,1) and (1,0) are class A, and no straight line separates them. Take u equal to x1 minus a half and v equal to x2 minus a half; the discriminant is g equal to a quarter minus 2uv, which in x is minus a quarter plus x1 plus x2 minus 2 x1 x2. The program gives g(0,0), g(0,1), g(1,0), g(1,1) as minus 0.25, 0.75, 0.75, minus 0.25, so g above 0 is A and below 0 is B, all four correct. The left shows the lifted coordinate table and the right the two regions cut by the hyperbola; the boundary is curved in the original coordinates but is a flat plane in the new coordinates."),
    ("XOR points lifted with x1x2; g = 1/4 - 2uv gives -0.25, 0.75, 0.75, -0.25 at (0,0),(0,1),(1,0),(1,1) and the hyperbola regions A/B.",) * 2)
add(3, ("項數公式", "Term-count formula"),
    ("l 個輸入、最高 r 階共有 k = C(l + r, r) 項；l = 2 時 r = 1 到 5 是 3、6、10、15、21", "With l inputs and degree up to r there are k = C(l + r, r) terms; for l = 2 and r = 1 to 5: 3, 6, 10, 15, 21"),
    ("XOR 靠交互項分開，兩個輸入、二階有 6 項。輸入更多、階數更高時，項數有幾項？", "The interaction term separates XOR, and two inputs at second order have 6 terms. How many terms are there with more inputs and a higher degree?"),
    ("這一頁給項數公式。l 個輸入、最高 r 階的多項式，共有 k 等於 C(l 加 r, r) 項，也就是 (l 加 r) 階乘除以 r 階乘再除以 l 階乘。l 等於 2 時，r 從 1 到 5 的項數是 3、6、10、15、21（程式計算）。例如 r 等於 2 就是前一頁的 6 項。來源是把「次數總和不超過 r」的乘積項數出來，等同於把 r 個球分到 l 加 1 個格子的組合數。",
     "This page gives the term-count formula. A polynomial of l inputs and degree at most r has k equal to C(l plus r, r) terms, that is, (l plus r) factorial over r factorial times l factorial. For l equal to 2 and r from 1 to 5 the counts are 3, 6, 10, 15, 21 (computed by program). For example r equal to 2 is the 6 terms of the previous page. The count comes from enumerating the product terms whose degrees sum to at most r, which equals the number of ways to put r balls into l plus 1 boxes."),
    ("Bars of k = C(l+r, r) for l=2, r=1..5: 3, 6, 10, 15, 21.",) * 2)
add(4, ("項數爆炸", "Term explosion"),
    ("l = 10、r = 10 時有 184,756 項；1000 筆資料的設計矩陣約 1.48 GB，ΦᵀΦ 約 273 GB", "At l = 10 and r = 10 there are 184,756 terms: Φ takes about 1.48 GB and ΦᵀΦ about 273 GB"),
    ("項數公式 k = C(l + r, r)：l = 2 時 r = 1 到 5 只有 3 到 21 項。輸入多、階數高時會怎樣？", "With k = C(l + r, r) terms, l = 2 and degree up to r = 5 gives just 21 terms. What happens with many inputs and a high degree?"),
    ("這一頁看項數爆炸。左圖是對數座標，l 等於 2、10、100 三條線，r 從 1 到 10；l 等於 10、r 等於 10 時有 184,756 項。右邊算記憶體，每個數用 8 位元組：1000 筆資料、設計矩陣 Φ 是 1000 乘 184,756，約 1.48 GB；KP13 的正規方程要 ΦᵀΦ，是 184,756 乘 184,756，約 273 GB，差約 185 倍，一般電腦裝不下。l 等於 2、r 等於 10 時也只有 66 項，所以是輸入變多才爆炸（程式計算）。",
     "This page looks at the term explosion. The left chart is on a log scale with three lines for l equal to 2, 10 and 100 and r from 1 to 10; at l equal to 10 and r equal to 10 there are 184,756 terms. On the right, memory at 8 bytes per number: with 1000 samples the design matrix Φ is 1000 by 184,756, about 1.48 GB; the normal equation of KP13 needs ΦᵀΦ, which is 184,756 by 184,756, about 273 GB, roughly 185 times larger and beyond an ordinary computer. At l equal to 2 and r equal to 10 there are only 66 terms, so the explosion comes from more inputs (computed by program)."),
    ("Log plot of k vs r for l=2,10,100; l=10, r=10 gives 184,756 terms; memory Phi 1.48 GB vs PhiT Phi 273 GB.",) * 2)
add(5, ("雙月資料", "Two moons"),
    ("雙月資料：r = 3（10 項）訓練正確率 90.0%、新點 93.2%；r = 1 的直線只有 78.3% 與 84.9%", "Two moons: r = 3 (10 terms) gets 93.2% on new points; the line at r = 1 only 84.9%"),
    ("項數隨輸入與階數爆炸。若輸入只有兩個，階數該取多高才夠？", "At l = 10 and r = 10 there are 184,756 terms and the design matrix needs 1.48 GB. With only two inputs, how high a degree is enough?"),
    ("這一頁用雙月資料實驗。訓練集每類 30 點、共 60 點：A 類取上半圓，B 類取平移的下半圓，各加 0.2 的雜訊；測試集每類 500 點，另用一個種子產生。r 從 1 到 5 的項數、訓練正確率與新點正確率（程式計算）：r 等於 1 是 3 項、78.3%、84.9%；r 等於 3 是 10 項、90.0%、93.2%。實線是 r 等於 3 的邊界，虛線是 r 等於 1 的直線；彎的邊界跟著兩個月牙走，直線切不出來。",
     "This page uses the two-moons experiment. The training set has 30 points per class, 60 in all: class A on the upper half circle, class B on the shifted lower half circle, each with noise 0.2; the test set has 500 points per class from another seed. For r from 1 to 5, the terms, training accuracy and new-point accuracy (computed by program): r equal to 1 is 3 terms, 78.3% and 84.9%; r equal to 3 is 10 terms, 90.0% and 93.2%. The solid curve is the boundary at r equal to 3 and the dashed line is r equal to 1; the curved boundary follows the two crescents while the line cannot."),
    ("Two moons with r=3 boundary (solid) and r=1 line (dashed); table r=1..5 with terms, training and test accuracy.",) * 2)
add(6, ("過度配適", "Overfitting"),
    ("階數再升，訓練正確率到 100%，新點卻從 93.2% 掉到 55.5%；r = 10 的條件數達 8.72×10⁸", "Higher degrees reach 100% training but only 55.5% on new points; the condition number hits 8.72×10⁸"),
    ("雙月資料 r = 3 時新點正確率 93.2%。階數繼續升高，是不是越高越好？", "On two moons, r = 3 with 10 terms gets 93.2% on new points. Does a higher degree keep helping?"),
    ("這一頁把階數升到 10。同一份雙月資料，r 從 1 到 10 的項數、訓練與新點正確率（程式計算）：r 等於 5 是 21 項、96.7%、91.2%；r 等於 8 是 36 項、98.3%、92.0%；r 等於 9 是 55 項、100%、63.7%；r 等於 10 是 66 項、100%、55.5%。過度配適：模型有餘裕把每個訓練點連雜訊都背下來，訓練到 100%，新點卻接近亂猜；r 等於 9 的 55 項和 r 等於 10 的 66 項已接近或超過 60 個訓練點。條件數是矩陣對微小誤差的放大倍率：r 等於 1 是 3.25，r 等於 6 是 3.44×10⁴，r 等於 8 是 4.42×10⁶，r 等於 10 是 8.72×10⁸，解越來越不穩。",
     "This page raises the degree to 10. On the same two-moons data, the terms, training and new-point accuracy for r from 1 to 10 (computed by program): r equal to 5 is 21 terms, 96.7% and 91.2%; r equal to 8 is 36 terms, 98.3% and 92.0%; r equal to 9 is 55 terms, 100% and 63.7%; r equal to 10 is 66 terms, 100% and 55.5%. Overfitting: the model has room to memorize every training point, noise included, so training reaches 100% while new points are near random guessing; 55 terms at r equal to 9 and 66 at r equal to 10 already approach or exceed the 60 training points. The condition number is the factor by which a matrix magnifies small errors: 3.25 at r equal to 1, 3.44×10⁴ at r equal to 6, 4.42×10⁶ at r equal to 8 and 8.72×10⁸ at r equal to 10, so the solution becomes less and less stable."),
    ("Accuracy curves for r=1..10 (training rises to 100%, test peaks 93.2% at r=3 and falls to 55.5%) and condition numbers 3.25, 3.44e4, 4.42e6, 8.72e8.",) * 2)
add(7, ("核方法預告", "Kernel preview"),
    ("先算內積再平方：(1 + x·z)² = 4，與逐項相乘的 6 項和一樣，卻只做一次內積", "Square after the inner product: (1 + x·z)² = 4, the same as the 6-term sum but with one inner product"),
    ("階數升高會過度配適，項數又爆炸到 184,756 項。能不能不寫出新座標，也得到同樣的內積？", "A higher degree reaches 100% training but only 55.5% on new points, and the terms explode. Can we get the same inner product without writing out the terms?"),
    ("這一頁預告核方法，細節在 KP19。取 l 等於 2、二階、x 等於 (1, 2)、z 等於 (3, 負 1)。直接算：(1 加 x·z) 的平方，x·z 等於 3 減 2 等於 1，所以是 (1 加 1) 的平方等於 4。造新座標再逐項相乘：φ(x) 等於 (1, √2 x1, √2 x2, x1 平方, x2 平方, √2 x1x2)，φ(x) 與 φ(z) 逐項相乘為 1、6、負 4、9、4、負 12，加起來也是 4（程式計算）。結果一樣，但左邊只做一次內積、一個次方；l 等於 10、r 等於 10 時照樣只算這麼多，不必展開 184,756 項。",
     "This page previews the kernel method; the details are in KP19. Take l equal to 2, second order, x equal to (1, 2) and z equal to (3, minus 1). Directly: the square of (1 plus x·z), where x·z is 3 minus 2 equal to 1, so (1 plus 1) squared is 4. With the new coordinates and term-by-term products: φ(x) equals (1, √2 x1, √2 x2, x1 squared, x2 squared, √2 x1x2), and the products of φ(x) and φ(z) term by term are 1, 6, minus 4, 9, 4, minus 12, which also sum to 4 (computed by program). The results agree, but the left side needs only one inner product and one square; at l equal to 10 and r equal to 10 it costs the same and never expands 184,756 terms."),
    ("Direct (1+x.z)^2 = 2^2 = 4 versus the 6-term inner product 1+6-4+9+4-12 = 4 with x=(1,2), z=(3,-1).",) * 2)
add(8, ("小練習", "Exercises"),
    ("兩題小練習：l = 4、r = 3 共 k = 35 項；g(1.5, 1.5) = −1.75 < 0，屬 B 類", "Two exercises: l = 4 and r = 3 give k = 35 terms; g(1.5, 1.5) = −1.75 < 0, class B"),
    ("核方法預告：先算內積再平方，與 6 項逐項相乘得到同樣的 4。用項數公式與判別函數算一次，答案是多少？", "The kernel preview: squaring after the inner product matches the 6-term sum, both 4. What do the term-count formula and the discriminant give?"),
    ("兩題小練習。題 1：4 個輸入、最高 3 階共有幾項？k 等於 C(4 加 3, 3) 等於 C(7, 3) 等於 7 階乘除以 3 階乘 4 階乘，也就是 5040 除以 6 乘 24，等於 35 項；核對：35 就是 C(7, 3)。題 2：g 等於 負 ¼ 加 x1 加 x2 減 2 x1 x2，點 (1.5, 1.5) 屬哪一類（g 大於 0 為 A）？g(1.5, 1.5) 等於 負 0.25 加 3 減 4.5 等於 負 1.75，小於 0，屬 B 類；核對：點在右上角的 B 區，和 (1,1) 同一區（程式計算）。",
     "Two exercises. Q1: how many terms with 4 inputs and degree at most 3? k equals C(4 plus 3, 3) equals C(7, 3), which is 7 factorial over 3 factorial times 4 factorial, that is 5040 over 6 times 24, equal to 35 terms; check: 35 is C(7, 3). Q2: with g equal to minus a quarter plus x1 plus x2 minus 2 x1 x2, which class is the point (1.5, 1.5) (g above 0 is A)? g(1.5, 1.5) equals minus 0.25 plus 3 minus 4.5, equal to minus 1.75, which is below 0, so class B; check: the point lies in the upper-right B region, the same region as (1,1) (computed by program)."),
    ("Exercise 1: l=4, r=3: k=C(7,3)=35. Exercise 2: g(1.5,1.5) = -0.25+3-4.5 = -1.75 < 0, class B.",) * 2)
PAGES = P
