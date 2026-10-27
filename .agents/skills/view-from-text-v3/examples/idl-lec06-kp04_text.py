import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
SK = "/workspace/idl-2026-fall/.agents/skills"
for p in ("view-from-text-v3/tools", "view-from-text-v2/tools", "view-from-text-v2/examples"): sys.path.insert(0, os.path.join(SK, p))
sys.path.insert(0, HERE)
from kp04_kit import native
JD = os.path.join(HERE, "out")
UNIT_ZH, UNIT_EN = "離散卷積與互相關", "Conv vs. corr"
TITLE_ZH = "深度學習導論 第6週 KP04：離散卷積與互相關（大學部）"; TITLE_EN = "Introduction to Deep Learning, Week 6, KP04: discrete convolution and cross-correlation (undergraduate)"
NOTE_ZH = "教材程度：大學部。依已確認的講義 第六週 KP04 製作：先規劃頁面，每頁以一張原生資訊圖為主，名詞在同一頁定義。主線 12 頁，課堂 25 分鐘；另有 2 頁延伸（交換律、練習），不計時。標「補充」的內容是課程來源沒有、講義補上的。練習頁只給提示，不給答案。"
NOTE_EN = "Level: undergraduate. Made from the confirmed lecture lec06-kp04: the pages were planned first, each page is mainly one native infographic, and terms are defined on the page where they are used. 12 main pages for a 25-minute class, plus 2 extension pages (commutativity, exercises) that are not timed. Content marked Added is not in the course source and was added by the lecture. The exercise page gives hints only, no answers."
def fig(n, alt_zh, alt_en): return lambda: native(f"kp04_p{n}", f"lec06-kp04-p{n}", JD, alt_en if os.environ.get("KP_LANG") == "en" else alt_zh)
NXT = {1: ("卷積的下標", "the subscript of convolution"), 2: ("完整卷積算例", "the full convolution worked out"), 3: ("互相關", "cross-correlation"), 4: ("翻核即相等", "flipping makes them equal"),
       5: ("二維翻兩次", "flipping twice in 2-D"), 6: ("二維互相關算例", "a 2-D cross-correlation worked out"), 7: ("對稱核與編號", "the symmetric kernel and the numbering"), 8: ("同一群函數", "the same set of functions"),
       9: ("翻轉何時有差", "when the flip matters"), 10: ("程式裡的名字", "the names in code"), 11: ("重點與迷思", "the summary and misconceptions"), 12: ("延伸：交換律", "the extension on commutativity"), 13: ("延伸：練習", "the extension exercises")}
PRV = {n + 1: v for n, v in {1: ("加權和", "the weighted sum"), 2: ("卷積的下標", "the subscript of convolution"), 3: ("完整卷積算例", "the full convolution"), 4: ("互相關", "cross-correlation"), 5: ("翻核即相等", "flipping makes them equal"),
       6: ("二維翻兩次", "flipping twice in 2-D"), 7: ("二維互相關算例", "the 2-D worked example"), 8: ("對稱核與編號", "the symmetric kernel"), 9: ("同一群函數", "the same set of functions"), 10: ("翻轉何時有差", "when the flip matters"),
       11: ("程式裡的名字", "the names in code"), 12: ("重點與迷思", "the summary"), 13: ("延伸：交換律", "commutativity")}.items()}
P = []
def add(n, lab, title, bridge, notes, alt):
    zh, en = notes
    if n in PRV: zh = f"承接上一頁的「{PRV[n][0]}」。" + zh; en = f"Following the previous page on {PRV[n][1]}. " + en
    if n in NXT: zh += f"接下來的下一頁是「{NXT[n][0]}」。"; en += f" Next, the next page covers {NXT[n][1]}."
    if n == 14: zh += "到這裡 KP04 就結束了；下一個單元 KP05 談稀疏連接與權重共享。"; en += " That ends KP04; the next unit, KP05, covers sparse connectivity and parameter sharing."
    P.append(dict(lab=lab, title=title, bridge=bridge, notes=(zh, en), layout="freeform", spec=fig(n, *alt)))

add(1, ("加權和", "Weighted sum"),
    ("每個輸出是附近幾格的加權和：$x=[2,1,3]$ 配 $w=[4,5]$ 只有 $2$ 個有效位置", "Each output is a weighted sum of nearby cells: $x=[2,1,3]$ with $w=[4,5]$ has $2$ valid positions"),
    ("KP03 用 Prewitt 核在 $3\\times3$ 的小窗裡逐格乘加。把這個動作寫成算式時，核要照原樣蓋，還是先翻面？", "KP03 multiplied and added cell by cell in a $3\\times3$ window with the Prewitt kernel. Written as a formula, does the kernel go down as it is, or flipped?"),
    ("這是 KP04 的第一頁。先備三件事：KP03 的小窗滑動，也就是拿 Prewitt 核在影像上逐格相乘再相加；加權和，也就是每個數乘上一個權重再全部加起來；第3章的反向傳播，它讓核的數字可以從資料學出來。"
     "這一課只用最小的例子：一維訊號 x 等於 2、1、3，共 3 格；核 w 等於 4、5，共 2 個權重。核也叫濾波器，就是一組固定的權重。把核蓋在訊號上，對齊的格子相乘再相加，就得到一個輸出；核往右挪一格，再算一次。核必須完整落在訊號內，所以只有 3 減 2 加 1 等於 2 個位置，這種輸出叫有效輸出；一般地，長 n 的訊號配長 k 的核，有 n 減 k 加 1 個。"
     "右邊是問題所在：同一個位置，照原樣蓋是 2 乘 4 加 1 乘 5 等於 13；先左右翻面再蓋是 2 乘 5 加 1 乘 4 等於 14。兩個數字不一樣。可以用一個比喻記：理論上的做法像烙餅，先翻面再壓下去；程式的做法像蓋橡皮章，拿起來直接蓋。這個比喻只對應核的方向。",
     "This is the first page of KP04. Prerequisites: the sliding window of KP03, which multiplies the Prewitt kernel with the image cell by cell and adds up; a weighted sum, where each number is multiplied by a weight and everything is added; and backpropagation from Chapter 3, which lets the numbers of a kernel be learned from data. "
     "This lesson uses the smallest example: the 1-D signal x is 2, 1, 3, three cells, and the kernel w is 4, 5, two weights. A kernel, also called a filter, is a fixed set of weights. Lay the kernel on the signal, multiply the aligned cells and add: that is one output; move the kernel one cell to the right and compute again. The kernel must lie fully inside the signal, so there are only 3 minus 2 plus 1, that is 2 positions; this is the valid output, and in general a signal of length n with a kernel of length k has n minus k plus 1 of them. "
     "The right side shows the question: at the same position, laying the kernel down as it is gives 2 times 4 plus 1 times 5, which is 13; flipping it first gives 2 times 5 plus 1 times 4, which is 14. The numbers differ. An analogy: the theory is like a pancake, flipped before it is pressed; the code is like a rubber stamp, stamped directly. The analogy covers only the direction of the kernel."),
    ("一維訊號 x=[2,1,3]、核 w=[4,5]；核有兩個擺放位置，3-2+1=2 個有效位置；第一個位置照原樣蓋得 2·4+1·5=13，先翻面得 2·5+1·4=14，兩者不同。",
     "1-D signal x=[2,1,3], kernel w=[4,5]; two placements, 3-2+1=2 valid positions; at the first position the kernel as it is gives 2·4+1·5=13, flipped gives 2·5+1·4=14; they differ."))

add(2, ("卷積的下標", "Subscript"),
    ("卷積的下標 $t-\\τ$：先把核翻面，再滑到位置 $t$", "The subscript $t-\\tau$ of convolution: flip the kernel, then slide it to $t$"),
    ("同一個位置，照原樣蓋得 $13$、先翻面得 $14$。課本的卷積選哪一種？式子 $w[t-\\τ]$ 裡哪一步是翻面？", "At the same position, the kernel as it is gives $13$ and flipped gives $14$. Which one does the textbook pick, and where in $w[t-\\tau]$ is the flip?"),
    ("訊號處理課本把一維離散卷積定義成：x 卷積 w 在位置 t 的值，等於對 τ 從負無窮加到正無窮，x 的 τ 乘 w 的 t 減 τ。τ 是求和指標，掃過輸入的每一格，和輸出位置 t 無關；訊號與核以外的格子都當成 0。"
     "關鍵在 w 的下標 t 減 τ。看下面三條軸：第一條是原本的核，4 在 0、5 在 1；第二條是 w 的負 τ，負號把核沿軸鏡射，5 跑到負 1、4 留在 0；第三條是 w 的 1 減 τ，加上 t 等於 1，把鏡射後的核挪到位置 1，於是 5 在 0、4 在 1，正好是翻過來的 5、4。所以卷積的指標是相減的，核要先翻轉，再滑動。翻轉核的意思，就是把核的順序倒過來，4、5 變成 5、4。",
     "Signal-processing textbooks define 1-D discrete convolution: x convolved with w at position t equals the sum over tau from minus infinity to infinity of x of tau times w of t minus tau. Tau is the summation index: it sweeps every input cell and is not the output position t; cells outside the signal and the kernel count as 0. "
     "The key is the subscript t minus tau. Look at the three axes: the first is the kernel as given, 4 at 0 and 5 at 1; the second is w of minus tau, where the minus sign mirrors the kernel, so 5 moves to minus 1 and 4 stays at 0; the third is w of 1 minus tau, where adding t equal to 1 moves the mirrored kernel to position 1, so 5 is at 0 and 4 at 1, exactly the flipped 5, 4. So the indices of convolution subtract, and the kernel is flipped before it slides. Flipping the kernel means reversing its order: 4, 5 becomes 5, 4."),
    ("卷積定義 (x*w)[t]=Σ x[τ] w[t−τ]；−τ 鏡射、+t 平移；三條 τ 軸：w[τ] 為 4 在 0、5 在 1；w[−τ] 為 5 在 −1、4 在 0；w[1−τ] 為 5 在 0、4 在 1；軸外當成 0。",
     "Convolution (x*w)[t]=Σ x[τ] w[t−τ]; −τ mirrors, +t shifts; axes: w[τ] has 4 at 0 and 5 at 1; w[−τ] has 5 at −1 and 4 at 0; w[1−τ] has 5 at 0 and 4 at 1; cells outside count as 0."))

add(3, ("完整卷積算例", "Full convolution"),
    ("逐項代入：完整卷積是 $[8,14,17,15]$，核整個落在訊號內的只有 $[14,17]$", "Term by term: the full convolution is $[8,14,17,15]$; only $[14,17]$ has the kernel fully inside"),
    ("卷積的 $w[t-\\τ]$ 就是把 $[4,5]$ 翻成 $[5,4]$ 再挪到 $t$。主例子代進去，每個 $t$ 算出多少？哪些才是有效輸出？", "Convolution says: flip the kernel, then slide it to $t$. With the main example, what does each $t$ give, and which are valid outputs?"),
    ("把主例子 x 等於 2、1、3，w 等於 4、5 逐項代進定義，t 從 0 到 3。這是講義補上的計算，所以右上角標了補充。每一列都是翻過來的核 5、4 挪到位置 t。t 等於 0：只有 x 的 0 乘 w 的 0，2 乘 4 等於 8。t 等於 1：2 乘 5 加 1 乘 4 等於 14。t 等於 2：1 乘 5 加 3 乘 4 等於 17。t 等於 3：3 乘 5 等於 15。"
     "核和訊號至少重疊一格的位置全部列出來，叫完整卷積，共有 n 加 k 減 1 等於 4 個數，就是 8、14、17、15。t 等於 0 和 3 時，核有一格懸在訊號外面，畫成淡色；只有 t 等於 1、2 時，核完整落在訊號內，所以有效輸出是 14、17，共 n 減 k 加 1 等於 2 個。再看 t 等於 1 那一列：x 的 0 乘的是 w 的 1 等於 5，x 的 1 乘的是 w 的 0 等於 4，等於把核翻成 5、4 之後才對齊。",
     "Substitute the main example, x equal to 2, 1, 3 and w equal to 4, 5, into the definition term by term, for t from 0 to 3. This computation was added by the lecture, hence the Added chip. Every row is the flipped kernel 5, 4 moved to position t. At t equal to 0 only x of 0 times w of 0 remains: 2 times 4 is 8. At t equal to 1: 2 times 5 plus 1 times 4 is 14. At t equal to 2: 1 times 5 plus 3 times 4 is 17. At t equal to 3: 3 times 5 is 15. "
     "Listing every position where kernel and signal overlap in at least one cell gives the full convolution, n plus k minus 1, that is 4 numbers: 8, 14, 17, 15. At t equal to 0 and 3 one cell of the kernel hangs outside the signal, drawn faint; only at t equal to 1 and 2 does the kernel lie fully inside, so the valid output is 14, 17, n minus k plus 1, that is 2 numbers. In the row t equal to 1, x of 0 is multiplied by w of 1, which is 5, and x of 1 by w of 0, which is 4: the kernel was flipped to 5, 4 before it was aligned."),
    ("完整卷積逐項：t=0 為 2·4=8；t=1 為 2·5+1·4=14；t=2 為 1·5+3·4=17；t=3 為 3·5=15；完整卷積 [8,14,17,15]，n+k−1=4 個；t=1,2 是有效輸出 [14,17]，n−k+1=2 個。",
     "Full convolution term by term: t=0 gives 2·4=8; t=1 gives 2·5+1·4=14; t=2 gives 1·5+3·4=17; t=3 gives 3·5=15; full convolution [8,14,17,15], n+k−1=4 numbers; t=1,2 are the valid output [14,17], n−k+1=2 numbers."))

add(4, ("互相關", "Cross-correlation"),
    ("互相關不翻核、指標相加：同一組 $x$、$w$ 得到 $[13,19]$", "Cross-correlation adds the indices and does not flip: the same $x$, $w$ give $[13,19]$"),
    ("卷積先翻核，主例子的有效輸出是 $[14,17]$。如果不翻核、直接把核蓋上去，數字會變成多少？", "Convolution flips the kernel first, and the valid output of the main example is $[14,17]$. If the kernel goes down without a flip, what are the numbers?"),
    ("程式裡多半把翻轉省略。不翻核、直接把核蓋上去的運算叫互相關，寫成星號 ⋆：x 互相關 w 在 t 的值，等於 m 從 0 加到 k 減 1，x 的 t 加 m 乘 w 的 m。這裡的指標是相加的：w 的 0 永遠對齊 x 的 t，w 的 1 對齊 x 的 t 加 1，核的順序不變，圖上橘色的 4 永遠在前。t 從 0 數到 n 減 k，正好是有效輸出。"
     "代入主例子：t 等於 0，2 乘 4 加 1 乘 5 等於 13；t 等於 1，1 乘 4 加 3 乘 5 等於 19。互相關得到 13、19，卷積得到 14、17，同一組 x 和 w，數字確實不同。常見錯誤是把式子裡的 w 的 t 減 τ 看成 w 的 τ 減 t，以為核照原樣滑過去；那樣算出來的其實就是互相關。",
     "Code usually skips the flip. The operation that lays the kernel down without flipping is cross-correlation, written with a star: x cross-correlated with w at t equals the sum over m from 0 to k minus 1 of x of t plus m times w of m. Here the indices add: w of 0 always lines up with x of t and w of 1 with x of t plus 1, so the order of the kernel is kept, and the orange 4 always comes first. t runs from 0 to n minus k, which is exactly the valid output. "
     "With the main example: at t equal to 0, 2 times 4 plus 1 times 5 is 13; at t equal to 1, 1 times 4 plus 3 times 5 is 19. Cross-correlation gives 13, 19 and convolution gives 14, 17: the same x and w, really different numbers. A common error is to read w of t minus tau as w of tau minus t and slide the kernel as it is; that computes cross-correlation."),
    ("互相關 (x⋆w)[t]=Σ_{m=0}^{k−1} x[t+m] w[m]；不翻核；t=0 為 2·4+1·5=13，t=1 為 1·4+3·5=19；互相關 [13,19] 與卷積 [14,17] 不同；常見錯誤：把 w[t−τ] 讀成 w[τ−t]。",
     "Cross-correlation (x⋆w)[t]=Σ_{m=0}^{k−1} x[t+m] w[m]; no flip; t=0 gives 2·4+1·5=13, t=1 gives 1·4+3·5=19; correlation [13,19] differs from convolution [14,17]; common error: reading w[t−τ] as w[τ−t]."))

add(5, ("翻核即相等", "Flip and they match"),
    ("把核翻成 $\\tilde w=[5,4]$ 再做互相關，正好得到卷積的 $[14,17]$", "Flip the kernel to $\\tilde w=[5,4]$ and cross-correlate: exactly the convolution's $[14,17]$"),
    ("互相關得 $[13,19]$，卷積得 $[14,17]$，差別只在核有沒有翻。把核先翻成 $[5,4]$ 再做互相關，會得到什麼？", "Cross-correlation gives $[13,19]$ and convolution $[14,17]$; the only difference is the flip. What if we flip the kernel to $[5,4]$ first and then cross-correlate?"),
    ("這是本單元最重要的一頁。三欄用同一條訊號 2、1、3，橘色是 4、藍色是 5。左欄是卷積，核已經翻成 5、4，依卷積的定義標成 t 等於 1、2，得到 14、17。中欄是互相關，用原本的 4、5，得到 13、19。右欄還是互相關，但把核換成翻轉後的 w̃ 等於 5、4，得到 2 乘 5 加 1 乘 4 等於 14，1 乘 5 加 3 乘 4 等於 17，和左欄一模一樣。"
     "一般地，令 w̃ 的 m 等於 w 的 k 減 1 減 m，就是把核左右翻；那麼 x 互相關 w̃ 在 t，等於 x 卷積 w 在 t 加 k 減 1。右邊的 k 減 1 只是編號的平移：卷積的有效輸出從 k 減 1 開始編號，互相關從 0 開始。這裡 k 等於 2，所以卷積的 t 等於 1、2 對應互相關的 t 等於 0、1。兩種運算的乘加骨架相同，只差核的方向。",
     "This is the key page of the unit. All three columns use the same signal 2, 1, 3; orange is 4 and blue is 5. The left column is convolution: the kernel is already flipped to 5, 4 and, following the definition, labelled t equal to 1 and 2; it gives 14, 17. The middle column is cross-correlation with the original 4, 5 and gives 13, 19. The right column is cross-correlation again, but with the flipped kernel w tilde equal to 5, 4: 2 times 5 plus 1 times 4 is 14 and 1 times 5 plus 3 times 4 is 17, exactly the left column. "
     "In general, let w tilde of m be w of k minus 1 minus m, which flips the kernel; then x cross-correlated with w tilde at t equals x convolved with w at t plus k minus 1. The k minus 1 on the right is only a shift of numbering: the valid output of convolution is numbered from k minus 1 and that of cross-correlation from 0. Here k is 2, so t equal to 1, 2 of convolution correspond to t equal to 0, 1 of cross-correlation. The multiply-and-add skeleton is the same; only the direction of the kernel differs."),
    ("三欄對照：卷積 x*w（核翻成 [5,4]，t=1,2）得 [14,17]；互相關 x⋆w（w=[4,5]，t=0,1）得 [13,19]；互相關 x⋆w̃（w̃=[5,4]）得 2·5+1·4=14、1·5+3·4=17，即 [14,17]；(x⋆w̃)[t]=(x*w)[t+k−1]，k=2。",
     "Three columns: convolution x*w (kernel flipped to [5,4], t=1,2) gives [14,17]; correlation x⋆w (w=[4,5], t=0,1) gives [13,19]; correlation x⋆w̃ (w̃=[5,4]) gives 2·5+1·4=14 and 1·5+3·4=17, i.e. [14,17]; (x⋆w̃)[t]=(x*w)[t+k−1] with k=2."))

add(6, ("二維翻兩次", "Flip twice in 2-D"),
    ("二維卷積兩個下標都相減：左右翻再上下翻，等於把核轉 $180^\\circ$", "Both 2-D subscripts subtract: flip left-right, then up-down, which rotates the kernel by $180^\\circ$"),
    ("一維只要把 $[4,5]$ 翻成 $[5,4]$，互相關就等於卷積。影像有高和寬兩個方向，核要翻幾次？", "In 1-D, flipping $[4,5]$ to $[5,4]$ makes cross-correlation equal convolution. An image has two directions, height and width: how many flips does the kernel need?"),
    ("影像有高和寬兩個方向，所以核也是二維的 K。二維卷積的定義：I 卷積 K 在 i、j 的值，等於對 p、q 求和，I 的 p、q 乘 K 的 i 減 p、j 減 q。p、q 是掃過影像各格的求和指標，i、j 是輸出的位置。兩個下標都是相減，所以核在列方向翻一次，在行方向也翻一次。"
     "看這個 3 乘 3 的核，數字 1 到 9 依序排列，顏色由淺到深。先左右翻，第一列變成 3、2、1；再上下翻，第一列變成 9、8、7，最後一列是 3、2、1。合起來就是轉 180 度：原本左上的 1 跑到右下，右下的 9 跑到左上。寫成式子，K̃ 的 m、n 等於 K 的 k 減 1 減 m、k 減 1 減 n，m、n 掃過核的各格，k 是核的邊長，這裡是 3。",
     "An image has two directions, height and width, so the kernel K is 2-D too. 2-D convolution: I convolved with K at i, j equals the sum over p and q of I of p, q times K of i minus p, j minus q. p and q are summation indices that sweep the image, and i, j is the output position. Both subscripts subtract, so the kernel is flipped once along the rows and once along the columns. "
     "Look at this 3 by 3 kernel with the numbers 1 to 9 in order, shaded from light to dark. Flip left-right: the first row becomes 3, 2, 1. Then flip up-down: the first row becomes 9, 8, 7 and the last row 3, 2, 1. Together this is a 180-degree rotation: the 1 at the top left moves to the bottom right and the 9 at the bottom right to the top left. As a formula, K tilde of m, n equals K of k minus 1 minus m, k minus 1 minus n, where m, n sweep the kernel cells and k is the side of the kernel, here 3."),
    ("二維卷積 (I*K)[i,j]=Σ_p Σ_q I[p,q] K[i−p, j−q]；3×3 核三列 [1,2,3]、[4,5,6]、[7,8,9]；左右翻後 [3,2,1]、[6,5,4]、[9,8,7]；再上下翻得 K̃ 三列 [9,8,7]、[6,5,4]、[3,2,1]，即轉 180°；K̃[m,n]=K[k−1−m, k−1−n]，k=3。",
     "2-D convolution (I*K)[i,j]=Σ_p Σ_q I[p,q] K[i−p, j−q]; 3×3 kernel rows [1,2,3], [4,5,6], [7,8,9]; after left-right [3,2,1], [6,5,4], [9,8,7]; after up-down K̃ rows [9,8,7], [6,5,4], [3,2,1], a 180° rotation; K̃[m,n]=K[k−1−m, k−1−n], k=3."))

add(7, ("二維互相關算例", "2-D correlation"),
    ("二維互相關：核框住四格乘加，$s[0,1]=2\\cdot1+0\\cdot0+1\\cdot0+2\\cdot1=4$", "2-D cross-correlation: the kernel frames four cells, $s[0,1]=2\\cdot1+0\\cdot0+1\\cdot0+2\\cdot1=4$"),
    ("二維卷積把核轉 $180^\\circ$；互相關則不轉，直接讓核的左上角對齊 $(i,j)$。實際怎麼算一格？", "2-D convolution rotates the kernel by $180^\\circ$; cross-correlation does not and puts the kernel's top-left corner at $(i,j)$. How is one cell computed?"),
    ("二維互相關不翻核，直接把核的左上角對齊影像的 i、j：s 的 i、j 等於 I 互相關 K 在 i、j，等於 m、n 各從 0 加到 k 減 1，I 的 i 加 m、j 加 n 乘 K 的 m、n。這是來源式子的改正版，來源寫錯的地方講義已經更正。輸出 s 叫特徵圖，s 的 i、j 是它在位置 i、j 的值。"
     "例子：3 乘 3 影像 I 的三列是 1、2、0，1、1、2，0、1、1；核 K 是 2 乘 2 單位矩陣，對角線是 1，其餘是 0。輸出大小是 3 減 2 加 1 乘 3 減 2 加 1，等於 2 乘 2。以 s 的 0、1 為例，黃框蓋住第 0、1 列與第 1、2 行，框裡是 2、0、1、2。逐格相乘再相加：2 乘 1 加 0 乘 0 加 1 乘 0 加 2 乘 1 等於 4。其他三格算法相同，輸出兩列是 2、4 與 2、2。",
     "2-D cross-correlation does not flip the kernel; it puts the kernel's top-left corner at i, j of the image: s of i, j equals I cross-correlated with K at i, j, the sum over m and n from 0 to k minus 1 of I of i plus m, j plus n times K of m, n. This is the corrected form of the source formula. The output s is called a feature map, and s of i, j is its value at position i, j. "
     "Example: the 3 by 3 image I has rows 1, 2, 0; 1, 1, 2; 0, 1, 1, and the kernel K is the 2 by 2 identity, 1 on the diagonal and 0 elsewhere. The output size is 3 minus 2 plus 1 times 3 minus 2 plus 1, that is 2 by 2. Take s of 0, 1: the yellow frame covers rows 0 and 1 and columns 1 and 2, holding 2, 0, 1, 2. Multiply cell by cell and add: 2 times 1 plus 0 times 0 plus 1 times 0 plus 2 times 1 is 4. The other three cells work the same way, and the output rows are 2, 4 and 2, 2."),
    ("二維互相關 s[i,j]=Σ_m Σ_n I[i+m, j+n] K[m,n]；影像 I 三列 [1,2,0]、[1,1,2]、[0,1,1]；K 為 2×2 單位矩陣；黃框第 0、1 列與第 1、2 行；s[0,1]=I[0,1]K[0,0]+I[0,2]K[0,1]+I[1,1]K[1,0]+I[1,2]K[1,1]=2·1+0·0+1·0+2·1=4；輸出大小 (3−2+1)×(3−2+1)=2×2；輸出兩列 [2,4]、[2,2]。",
     "2-D cross-correlation s[i,j]=Σ_m Σ_n I[i+m, j+n] K[m,n]; image I rows [1,2,0], [1,1,2], [0,1,1]; K the 2×2 identity; yellow frame rows 0,1 and columns 1,2; s[0,1]=I[0,1]K[0,0]+I[0,2]K[0,1]+I[1,1]K[1,0]+I[1,2]K[1,1]=2·1+0·0+1·0+2·1=4; size (3−2+1)×(3−2+1)=2×2; output rows [2,4] and [2,2]."))

add(8, ("對稱核與編號", "Symmetric kernel"),
    ("單位矩陣轉 $180^\\circ$ 不變，所以這個例子的卷積也是 $[2,4]$、$[2,2]$", "The identity is unchanged by a $180^\\circ$ rotation, so convolution here also gives $[2,4]$, $[2,2]$"),
    ("二維互相關算出輸出兩列 $[2,4]$、$[2,2]$。改用卷積、核先轉 $180^\\circ$，結果會不同嗎？二維也有編號平移嗎？", "2-D cross-correlation gave the output rows $[2,4]$ and $[2,2]$. With convolution the kernel is rotated first: does the result change, and does 2-D also shift the numbering?"),
    ("這個例子有一個陷阱。單位矩陣轉 180 度之後還是自己，K̃ 等於 K，所以用卷積算也得到 2、4 與 2、2。核轉 180 度後不變時，兩種運算一樣；要看出差別，必須換一個轉了會變的核，這留給練習第 3 題，答案請自己算。"
     "右邊是講義補上的關係，所以標了補充。和一維一樣，用翻轉的核做互相關，就等於做卷積，只差編號平移 k 減 1：I 互相關 K̃ 在 i、j，等於 I 卷積 K 在 i 加 k 減 1、j 加 k 減 1。依卷積的定義，完整卷積的位置是 i、j 從 0 到 3；有效輸出是中間這塊，從 i、j 等於 k 減 1 等於 1 開始。把這塊重新從 0 編號，就是互相關的 s。講義寫卷積的有效輸出時，都是這種重新編號後的版本。",
     "This example hides a trap. The identity rotated by 180 degrees is itself, K tilde equals K, so convolution also gives 2, 4 and 2, 2. When the kernel does not change under rotation, the two operations agree; to see a difference you need a kernel that changes, which is left for Exercise 3. "
     "The right side is a relation added by the lecture, hence the Added chip. As in 1-D, cross-correlation with the flipped kernel is convolution up to a shift of numbering by k minus 1: I cross-correlated with K tilde at i, j equals I convolved with K at i plus k minus 1, j plus k minus 1. By the definition of convolution, the full convolution sits at i, j from 0 to 3, and the valid output is the middle block, starting at i, j equal to k minus 1, which is 1. Renumbering that block from 0 gives the cross-correlation s. Whenever the lecture writes the valid output of convolution, it means this renumbered version."),
    ("左：2×2 單位矩陣 K 轉 180° 不變，K̃=K，所以卷積也得 [2,4]、[2,2]；練習第 3 題換成會變的核。右（補充）：(I⋆K̃)[i,j]=(I*K)[i+k−1, j+k−1]；完整卷積位置 i,j=0..3，有效區塊 i,j=1,2 的值 2,4,2,2，重新從 0 編號得 s。",
     "Left: the 2×2 identity K is unchanged by a 180° rotation, K̃=K, so convolution also gives [2,4], [2,2]; Exercise 3 uses a kernel that changes. Right (Added): (I⋆K̃)[i,j]=(I*K)[i+k−1, j+k−1]; full convolution positions i,j=0..3; valid block i,j=1,2 holds 2,4,2,2 and is renumbered from 0 as s."))

add(9, ("同一群函數", "Same functions"),
    ("用 $K$ 做卷積等於用 $\\tilde K$ 做互相關：兩群網路是同一群函數", "Convolution with $K$ equals cross-correlation with $\\tilde K$: the two families of networks are one set of functions"),
    ("二維也一樣：用 $\\tilde K$ 做互相關就是用 $K$ 做卷積，只差編號。既然數字不同，框架省略翻轉不會出錯嗎？", "The identity survives rotation, and in 2-D only the numbering shifts. Since the numbers usually differ, doesn't skipping the flip cause errors?"),
    ("關鍵在於卷積神經網路的核不是人手寫的，而是用反向傳播學出來的。前面已經看到：用 K 做卷積，和用 K̃ 做互相關，輸出完全相同。左邊每一個核，在右邊都有一個翻轉後的核和它配對，而且輸出一樣：4、5 的卷積和 5、4 的互相關都是 14、17；5、4 的卷積和 4、5 的互相關都是 13、19。"
     "所以所有卷積網路和所有互相關網路，是同一群函數，只是同一組數字換了排列。兩群函數完全一樣，所以表達能力相同；表達能力指的是一種網路能表示的所有函數。梯度下降只負責在這群函數裡，沿著損失下降的方向調整核的數字，找一個損失小的；它不在乎慣例叫什麼名字，所以不翻也找得到。可以想成慣用左手或右手寫字，練習夠了都寫得好；這個比喻只說結果一樣好，兩種慣例學到的核互為翻轉。",
     "The key is that the kernel of a convolutional network is not written by hand but learned by backpropagation. We already saw that convolution with K and cross-correlation with K tilde give exactly the same output. Every kernel on the left pairs with its flip on the right, with the same output: convolution with 4, 5 and cross-correlation with 5, 4 both give 14, 17; convolution with 5, 4 and cross-correlation with 4, 5 both give 13, 19. "
     "So all convolution networks and all cross-correlation networks are the same set of functions; only the arrangement of the same numbers differs. Because the two sets are identical, the expressive power is the same; expressive power means all the functions a kind of network can represent. Gradient descent only adjusts the kernel numbers downhill on the loss and searches this set for one with a small loss; it does not care what the convention is called, so it finds one without the flip too. Think of writing with the left or the right hand: with enough practice both write well. The analogy only says the results are equally good; the kernels the two conventions learn are flips of each other."),
    ("左：卷積網路的核 [4,5]、[5,4]、K；右：互相關網路的核 [5,4]、[4,5]、K̃；配對的輸出分別是 [14,17]、[13,19]、相同；推論：兩群是同一群函數，所以表達能力相同；梯度下降只在這群函數裡找損失小的。",
     "Left: convolution kernels [4,5], [5,4], K; right: correlation kernels [5,4], [4,5], K̃; paired outputs [14,17], [13,19], the same; so the two sets are the same set of functions and the expressive power is equal; gradient descent only searches this set for a small loss."))

add(10, ("翻轉何時有差", "When flip matters"),
    ("核是手設計的、或要跨框架搬權重時，翻不翻就有差別", "With a hand-designed kernel, or when weights move between frameworks, the flip does matter"),
    ("核是學出來的，兩群函數一樣，所以不翻也能訓練。如果核不是學出來的呢？什麼情況下翻不翻有差別？", "The kernel is learned and the two sets of functions are the same, so training works without the flip. What if the kernel is not learned? When does the flip matter?"),
    ("前一頁的推論有一個假設：核的數字必須是學出來的。有兩種情況，翻不翻就有差別。第一種，核是人手設計的固定濾波器，例如 KP03 的水平 Prewitt 核，三列是 1、1、1，0、0、0，負 1、負 1、負 1。把它轉 180 度，正負號全反，等於負 H；藍色的正數和紅色的負數整個交換，偵測到的邊緣也跟著變號。練習第 2 題讓你用一個小塊自己驗證，答案這裡不說。"
     "第二種是講義補上的情況：把一個框架訓練好的權重，搬到使用另一種慣例的程式裡。這時核必須先轉 180 度，否則輸出會錯。",
     "The argument of the previous page has one assumption: the numbers of the kernel are learned. In two cases the flip matters. First, the kernel is a fixed, hand-designed filter, such as the horizontal Prewitt kernel of KP03 with rows 1, 1, 1; 0, 0, 0; minus 1, minus 1, minus 1. Rotate it by 180 degrees and every sign reverses: it becomes minus H, the blue positives and red negatives swap, and the detected edges change sign too. Exercise 2 lets you check this on a small patch; the answer is not given here. "
     "Second, a case added by the lecture: weights trained in one framework are moved into code that uses the other convention. Then the kernel must be rotated by 180 degrees first, otherwise the output is wrong."),
    ("左：水平 Prewitt 核 H 三列 [1,1,1]、[0,0,0]、[−1,−1,−1]，轉 180° 後三列 [−1,−1,−1]、[0,0,0]、[1,1,1]，rot180(H)=−H，偵測到的邊緣變號；右（補充）：框架 A 的核 K 要先轉 180° 再放進另一種慣例的框架 B；前提：核的數字必須是學出來的。",
     "Left: horizontal Prewitt H rows [1,1,1], [0,0,0], [−1,−1,−1]; rotated by 180° the rows are [−1,−1,−1], [0,0,0], [1,1,1], rot180(H)=−H, edges change sign; right (Added): kernel K from framework A must be rotated by 180° before framework B with the other convention; assumption: the kernel is learned."))

add(11, ("程式裡的名字", "Names in code"),
    ("程式裡的 conv2d 名叫卷積，實際做的是互相關", "conv2d in code is called convolution but computes cross-correlation"),
    ("固定的核翻轉後意思會變，搬權重也要先轉 $180^\\circ$。所以得先知道：常用的函式各做哪一種運算？", "A fixed kernel changes meaning when flipped, and moved weights need a $180^\\circ$ rotation. So which operation does each common function compute?"),
    ("與實作的連結。PyTorch 的 torch.nn.functional.conv2d 和 TensorFlow 的 tf.nn.conv2d，名字叫卷積，實際做的是互相關，不翻核。NumPy 的情形是講義補上的，所以標了補充：np.convolve 會翻核，是真正的卷積；np.correlate 不翻，是互相關。講義的程式就是用這兩個函式核對前面的數字。論文的公式常寫星號 ∗ 的卷積，程式裡多半算的是互相關。"
     "到這裡，滑著乘加已經有了精確的算式。可是同一個乘加，也可以寫成一個巨大的全連接層。和全連接層比，卷積到底省下了什麼？這是 KP05 的主題。",
     "Connection to code. PyTorch's torch.nn.functional.conv2d and TensorFlow's tf.nn.conv2d are called convolution but compute cross-correlation, without the flip. The NumPy case was added by the lecture, hence the Added chips: np.convolve flips the kernel and is true convolution; np.correlate does not flip and is cross-correlation. The lecture's program uses these two functions to check the earlier numbers. Papers often write the asterisk of convolution, while the code mostly computes cross-correlation. "
     "At this point the sliding multiply-and-add has an exact formula. But the same multiply-and-add can also be written as one huge fully connected layer. Compared with a fully connected layer, what does convolution save? That is the topic of KP05."),
    ("函式對照表：torch.nn.functional.conv2d、tf.nn.conv2d 名叫卷積，實際做互相關（不翻核）；np.convolve（補充）做卷積（翻核）；np.correlate（補充）做互相關；論文常寫 *，程式多半是 ⋆；留給 KP05：卷積比全連接層省下什麼。",
     "Function table: torch.nn.functional.conv2d and tf.nn.conv2d are called convolution but compute cross-correlation (no flip); np.convolve (Added) computes convolution (flips); np.correlate (Added) computes cross-correlation; papers write *, code mostly ⋆; left for KP05: what convolution saves over a fully connected layer."))

add(12, ("重點與迷思", "Summary"),
    ("帶走一句話：翻不翻只差核的方向，核是學出來的，所以不翻也能訓練", "One sentence to keep: the flip only changes the kernel's direction, and a learned kernel needs no flip"),
    ("框架的 conv2d 多半做互相關，名字卻叫卷積。把整個單元收成一句話：翻不翻，到底差在哪？", "conv2d is named convolution but correlates. In one sentence for the whole unit: what does the flip change?"),
    ("一句話重點：理論上的卷積要先把核翻轉再滑動；深度學習程式通常不翻轉，實際算的是互相關。翻轉只是核的數字換個排列，兩種寫法的表達能力相同，而核是學出來的，所以不翻也能訓練。上排三條式子是一維卷積、一維互相關和翻核的關係，主例子分別是 14、17，13、19，14、17。"
     "下排是四個常見迷思。第一，核一定要先轉 180 度，否則網路無法訓練：錯，不翻也能訓練，網路會學到翻轉後的核。第二，式子右邊的求和指標就是輸出：錯，求和指標只負責掃過各格。第三，函式叫 conv2d，所以是真正的卷積：錯，它多半做互相關。第四，試一個例子結果一樣，兩者就相同：錯，核轉 180 度不變時才一樣，例如單位矩陣。",
     "One-sentence summary: convolution in theory flips the kernel before sliding; deep-learning code usually does not flip and actually computes cross-correlation. Flipping only rearranges the kernel's numbers, so the two forms have the same expressive power, and since the kernel is learned, training works without the flip. The top row shows 1-D convolution, 1-D cross-correlation and the flip relation, with main-example results 14, 17; 13, 19; and 14, 17. "
     "The bottom rows are four common misconceptions. One: the kernel must be rotated by 180 degrees or the network cannot train; wrong, it trains without the flip and learns the flipped kernel. Two: the summation index on the right is the output; wrong, it only sweeps the cells. Three: a function named conv2d computes true convolution; wrong, it mostly computes cross-correlation. Four: one example agreed, so the two are the same; wrong, they agree only when the kernel is unchanged by rotation, like the identity."),
    ("上：一維卷積 (x*w)[t]=Σ x[τ]w[t−τ] 主例子 [14,17]；一維互相關 (x⋆w)[t]=Σ x[t+m]w[m] 主例子 [13,19]；(x⋆w̃)[t]=(x*w)[t+k−1] 主例子 [14,17]。下：四個迷思與更正：不翻也能訓練；求和指標只掃過各格；conv2d 多半做互相關；核轉 180° 不變時兩者才一樣。",
     "Top: 1-D convolution (x*w)[t]=Σ x[τ]w[t−τ], example [14,17]; 1-D cross-correlation (x⋆w)[t]=Σ x[t+m]w[m], example [13,19]; (x⋆w̃)[t]=(x*w)[t+k−1], example [14,17]. Bottom: four misconceptions corrected: training works without the flip; the summation index only sweeps cells; conv2d mostly computes cross-correlation; the two agree only for a kernel unchanged by rotation."))

add(13, ("延伸：交換律", "Extension: commute"),
    ("延伸：卷積可交換，$x*w=w*x$；互相關交換後順序整個倒過來", "Extension: convolution commutes, $x*w=w*x$; swapping cross-correlation reverses the order"),
    ("卷積先翻核、互相關不翻，主例子是 $[14,17]$ 與 $[13,19]$。除了翻核，兩者的數學性質還有什麼不同？", "A learned kernel needs no flip, so the flip is settled. With the main example $[14,17]$ and $[13,19]$, how else do convolution and cross-correlation differ?"),
    ("這是延伸頁，不計入課堂時間，需要一點訊號與系統的背景。卷積滿足交換律，兩個運算元對調結果不變：完整卷積 x 卷積 w 和 w 卷積 x 都是 8、14、17、15。互相關一般不滿足。下面的完整互相關是講義補上的，所以標了補充：它和完整卷積一樣，列出核和訊號有重疊的所有位置，訊號外補 0，t 從負的 k 減 1 到 n 減 1。"
     "主例子中，x 互相關 w 是 10、13、19、12；把較長的 x 當成核，w 互相關 x 卻是 12、19、13、10，順序整個倒過來。在傅立葉域，卷積對應逐點相乘，互相關則多一步複共軛。這些性質在訊號處理很重要，但對學一個核的卷積神經網路沒有影響。",
     "This is an extension page, not timed, and it needs some background in signals and systems. Convolution is commutative, swapping the operands keeps the result: the full convolutions x with w and w with x are both 8, 14, 17, 15. Cross-correlation in general is not. The full cross-correlation below was added by the lecture, hence the Added chip: like the full convolution, it lists every position where kernel and signal overlap, with zeros outside the signal, t from minus the quantity k minus 1 up to n minus 1. "
     "In the main example, x cross-correlated with w is 10, 13, 19, 12; treating the longer x as the kernel, w cross-correlated with x is 12, 19, 13, 10, the order reversed. In the Fourier domain convolution is pointwise multiplication, while cross-correlation needs an extra complex conjugate. These properties matter in signal processing but do not affect a convolutional network that learns its kernel."),
    ("延伸：完整卷積 x*w=[8,14,17,15]，w*x=[8,14,17,15]，相同；完整互相關（補充）x⋆w=[10,13,19,12]，w⋆x=[12,19,13,10]，順序倒過來；交換律、完整互相關定義（t 從 −(k−1) 到 n−1）、傅立葉域說明。",
     "Extension: full convolution x*w=[8,14,17,15] and w*x=[8,14,17,15], equal; full cross-correlation (Added) x⋆w=[10,13,19,12] and w⋆x=[12,19,13,10], reversed; definitions of commutativity and full cross-correlation (t from −(k−1) to n−1); Fourier-domain note."))

add(14, ("延伸：練習", "Extension: exercises"),
    ("延伸：四題練習，先判斷要不要翻核，再自己算", "Extension: four exercises; first decide whether to flip the kernel, then compute"),
    ("卷積可交換，互相關交換後順序倒過來。換你動手：給一組新的數字，要不要翻核？", "Convolution commutes, and swapped cross-correlation reverses the order. Your turn: with new numbers, is the kernel flipped or not?"),
    ("這是延伸頁，不計入課堂時間。四題都只給提示，請先自己算，再用 np.convolve 和 np.correlate 核對。第 1 題：x 等於 1、0、2，w 等於 3、4，算卷積與互相關的有效輸出；提示是卷積先把核翻成 4、3。第 2 題是講義補上的題目：一個 3 乘 3 小塊和水平 Prewitt 核，比較兩種運算的正負號；提示是先把核轉 180 度。"
     "第 3 題：用前面的影像 I，換成核 K 兩列 1、2 與 0、0，這個核轉了會變，所以兩種運算會不同；提示是轉 180 度後的核兩列是 0、0 與 2、1。第 4 題也是講義補上的：訊號 3、1、4、1、5 配兩個長度 3 的核，想想哪一個核翻轉前後相同。",
     "This is an extension page, not timed. All four exercises give hints only; compute first, then check with np.convolve and np.correlate. Exercise 1: x is 1, 0, 2 and w is 3, 4; compute the valid outputs of convolution and of cross-correlation; the hint is that convolution first flips the kernel to 4, 3. Exercise 2 was added by the lecture: a 3 by 3 patch with the horizontal Prewitt kernel; compare the signs of the two operations; the hint is to rotate the kernel by 180 degrees first. "
     "Exercise 3: use the earlier image I with the kernel K of rows 1, 2 and 0, 0; this kernel changes under rotation, so the two operations differ; the hint is that the rotated kernel has rows 0, 0 and 2, 1. Exercise 4 was also added by the lecture: the signal 3, 1, 4, 1, 5 with two kernels of length 3; think about which kernel is the same before and after flipping."),
    ("四題練習（只給提示）：1. x=[1,0,2]、w=[3,4]，提示卷積先把核翻成 [4,3]；2.（補充）小塊 X 三列 [0,0,0]、[1,1,1]、[2,2,2] 與 Prewitt H，提示先轉 180°；3. 影像 I 配 K 兩列 [1,2]、[0,0]，提示轉後兩列 [0,0]、[2,1]；4.（補充）訊號 [3,1,4,1,5] 配核 [1,2,1]、[1,0,−1]，提示比較翻轉前後。",
     "Four exercises (hints only): 1. x=[1,0,2], w=[3,4], hint: convolution flips the kernel to [4,3]; 2. (Added) patch X rows [0,0,0], [1,1,1], [2,2,2] with Prewitt H, hint: rotate by 180° first; 3. image I with K rows [1,2], [0,0], hint: rotated rows [0,0], [2,1]; 4. (Added) signal [3,1,4,1,5] with kernels [1,2,1], [1,0,−1], hint: compare before and after flipping."))
TITLES = {1: ("每個輸出是附近幾格的加權和，核的方向會改變數字", "Each output is a weighted sum of nearby cells"),
          2: ("卷積的下標相減：先把核翻面，再滑到輸出位置", "Convolution subtracts indices: flip, then slide"),
          3: ("逐項代入：完整卷積有四個數，有效輸出只有中間兩個", "Full convolution has four numbers, two valid"),
          4: ("互相關不翻核、指標相加，主例子得到不同的數字", "Cross-correlation adds indices and skips the flip"),
          5: ("把核翻面再做互相關，正好得到卷積的有效輸出", "Correlate with the flipped kernel: convolution"),
          6: ("二維兩個下標都相減：左右翻再上下翻，就是轉 180 度", "In 2-D, flip both ways: a 180-degree rotation"),
          7: ("二維互相關：核框住四格逐格乘加，右上格是 4", "2-D cross-correlation: four cells give 4"),
          8: ("單位矩陣轉了不變，所以這個例子兩種運算一樣", "The identity survives rotation, so both agree"),
          9: ("用核做卷積等於用翻面的核做互相關：同一群函數", "Conv with K equals correlation with its flip"),
          10: ("核是手設計的或要搬權重時，翻不翻就有差別", "Hand-made kernels and moved weights need the flip"),
          11: ("程式裡的 conv2d 名叫卷積，實際做的是互相關", "conv2d is named convolution but correlates"),
          12: ("一句話：翻不翻只差核的方向，核學得到所以不必翻", "One sentence: a learned kernel needs no flip"),
          13: ("延伸：卷積可交換，互相關交換後順序倒過來", "Extension: convolution commutes; correlation does not"),
          14: ("延伸：四題練習，先判斷要不要翻核再自己算", "Extension: four exercises, hints only")}
for i, pg in enumerate(P, 1): pg["title"] = TITLES[i]     # plain-text titles (audit D2: no math in a page title; 6-26 characters)
PAGES = P
