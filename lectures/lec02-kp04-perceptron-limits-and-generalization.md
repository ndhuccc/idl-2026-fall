# 第2週 KP04：單層 Perceptron 解不了 XOR，可分也不保證泛化（One Perceptron cannot solve XOR, and separable data do not guarantee generalization）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)
>
> 週次：第2週（2026-09-17）｜主題：神經網路基礎：感知器與多層感知器
>
> Week 2 (2026-09-17) | Topic: Neural network fundamentals: perceptron and multilayer perceptron

本講義是第2章的第四個知識點（KP04），承接 KP03 留下的問題：Perceptron 這麼會學，是不是什麼分類問題都學得會？第1章 KP08 已經用四條不等式**證明**單一線性門檻分不開 XOR；這一節不再重證，而是看這個限制在**學習過程**中長什麼樣（停不下來），並看第二個、比較隱微的限制：資料可分，學到的邊界也可能不好。先備知識是 KP02 與第1章 KP08。內容依第2章主文重新講解，來源讀取日期為 2026-10-14；新增的釐清標為 `[補充]`。

This is the fourth knowledge point (KP04) of Chapter 2. It picks up the question KP03 left open: the Perceptron learns well, but can it learn any classification problem? Chapter 1 KP08 already **proved** with four inequalities that a single linear threshold cannot split XOR; this note does not prove it again but shows what the limit looks like in the **learning process** (it never stops), and then a second, subtler limit: even when the data are separable, the learned boundary may be poor. It assumes KP02 and Chapter 1 KP08. It re-explains the main text of Chapter 2, accessed on 2026-10-14; clarifications that are new are marked `[Added]`.

[下載本單元投影片](../slides/lec02-kp04-perceptron-limits-and-generalization.jbd.json)

## KP04 單層 Perceptron 解不了 XOR，可分也不保證泛化（One Perceptron cannot solve XOR, and separable data do not guarantee generalization）

**建議時間 / Time**：15 分鐘　**對應實作 / Related practice**：讓 Perceptron 學 XOR，數每輪的更新次數；比較同一批資料上兩條零訓練誤差的邊界。

### 學習目標（Learning objectives）

讀完之後，你能說出 XOR 為什麼讓 Perceptron 學不停（引用第1章 KP08 的證明）；能解讀「每輪更新次數永遠不歸零」；能說明資料線性可分時 Perceptron 為何仍可能泛化不佳；能說出兩個限制各自需要什麼解法。

After reading, you can say why XOR keeps a Perceptron from ever stopping (citing the proof of Chapter 1 KP08); read "the number of updates per pass never reaches zero"; explain why a Perceptron may still generalize poorly on linearly separable data; and say what each of the two limits needs as a remedy.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $(x_1,x_2)\to y$ | XOR：兩輸入不同才輸出 1 | XOR: outputs 1 only when the two inputs differ |
| 每輪更新次數 | 一輪（四筆）中猜錯而更新的筆數 | number of samples in one pass (four) that were wrong and updated |
| 零訓練誤差 | 訓練資料全部分對 | every training sample is classified right |
| 泛化 | 對沒看過的資料也分得對 | being right on data not seen in training |

### 1. XOR：證明在第1章，這裡看它「學不停」（XOR: proved in Chapter 1; here we watch it never stop）

XOR 的四個點：$(0,0),(1,1)\to0$、$(0,1),(1,0)\to1$，同一類的兩個點在正方形的**對角線**上。第1章 KP08 已證明沒有 $w_1,w_2,b$ 同時滿足四個條件（把中間兩式相加與其餘兩式矛盾），所以單一線性門檻做不出 XOR。對學習規則來說，這表示 KP02 的結論「可分才保證停」反過來：不可分，一定停不下來。

The four XOR points are $(0,0),(1,1)\to0$ and $(0,1),(1,0)\to1$, and the two points of one class lie on the **diagonals** of the square. Chapter 1 KP08 proved that no $w_1,w_2,b$ satisfy all four conditions (adding the middle two conditions contradicts the other two), so a single linear threshold cannot make XOR. For the learning rule this reverses the conclusion of KP02, "stops if separable": if not separable, it can never stop.

實跑：從全 0 開始，依 $(0,0),(0,1),(1,0),(1,1)$ 的順序反覆學 XOR（$\eta=1$）。每一輪的更新次數是 $3,3,4,4,4,4,4,4,4,4$〔程式計算〕。從第 4 輪起，每輪結束時權重都回到同一組 $(w_1,w_2)=(-1,0)$、$w_0=0$：輪內更新 4 次又繞回原處，永遠不會出現「一整輪沒有更新」。我們另用程式在 $[-3,3]$ 的格點上搜尋所有權重，最多只能讓 4 筆中的 3 筆正確〔程式計算〕。

A real run: from all zeros, repeat the order $(0,0),(0,1),(1,0),(1,1)$ to learn XOR ($\eta=1$). The number of updates per pass is $3,3,4,4,4,4,4,4,4,4$ (computed by program). From pass 4 on, the weights at the end of every pass are the same, $(w_1,w_2)=(-1,0)$ with $w_0=0$: the 4 updates inside a pass bring them back where they started, and a pass with no update never appears. We also searched all weights on a grid over $[-3,3]$ by program, and at most 3 of the 4 points can be right (computed by program).

![左：XOR 的四個點與兩條嘗試的直線；右：Perceptron 在 XOR 上每一輪的更新次數](../assets/figures/lec02-kp04-xor-limit.png)

**圖說 `lec02-kp04-xor-limit`**：先看左圖，虛線讓 1 個點錯，點線讓 2 個點錯，沒有一條線全對；再看右圖，每一輪都至少更新 3 次，10 輪後仍沒有停下來。 / **Caption `lec02-kp04-xor-limit`:** Look first at the left panel: the dashed line gets 1 point wrong and the dotted line 2, and no line gets all four right; then at the right panel: every pass has at least 3 updates and after 10 passes it still has not stopped.

### 2. 可分也不夠：邊界取決於起點（Separable is not enough: the boundary depends on the start）

第二個限制比較隱微。就算資料線性可分，Perceptron 一得到「全部分對」就停了，不會再去挑最好的一條線，所以最後的邊界取決於初始權重與資料順序。下圖用同一批資料（兩類各 12 點，固定亂數種子）與同一條規則，只換起點與順序，得到邊界 A 與 B；兩條都讓訓練誤差為 0。但對新的點 $(2.5,2.4)$，A 判為類 1，B 判為類 0〔程式計算：A 的 $\mathbf{w}=(4.03,3.17)$、$w_0=-15.22$；B 的 $\mathbf{w}=(0.94,0.95)$、$w_0=-5.04$〕。

The second limit is subtler. Even when the data are linearly separable, the Perceptron stops as soon as everything is right and does not look for the best line, so the final boundary depends on the initial weights and the data order. The figure below uses the same data (12 points per class, fixed random seed) and the same rule, changing only the start and the order, and gets boundaries A and B; both have zero training error. But for the new point $(2.5,2.4)$, A says class 1 and B says class 0 (computed by program: A has $\mathbf{w}=(4.03,3.17)$, $w_0=-15.22$; B has $\mathbf{w}=(0.94,0.95)$, $w_0=-5.04$).

![同一批資料上兩條零訓練誤差的邊界，對新點的判斷不同](../assets/figures/lec02-kp04-two-boundaries.png)

**圖說 `lec02-kp04-two-boundaries`**：先看兩張圖的點是否完全相同（是）；再看兩條線，A 靠近左下的藍點，B 緊貼右上的橘點；最後看綠色星號，它被 A 判為類 1，被 B 判為類 0。 / **Caption `lec02-kp04-two-boundaries`:** Look first at whether the points in the two panels are identical (they are); then at the two lines, A close to the blue points at lower left and B hugging the orange points at upper right; finally at the green star, which A calls class 1 and B calls class 0.

### 3. 兩個限制，兩條出路（Two limits, two ways out）

| 年份 | 事件 | 意義 |
|---|---|---|
| 1958 | Rosenblatt 提出 Perceptron | 第一個會學習的神經元 |
| 1960 | Widrow 與 Hoff 提出 ADALINE | 改用連續的 LMS 損失 |
| 1969 | Minsky 與 Papert 分析單層限制 | XOR 成為最有名的反例 |
| 之後 | 多層加非線性活化 | 繞過直線限制 |

直線限制要靠多層與非線性（KP05）；泛化的限制要靠換一種會挑「好」直線的方法，例如用連續損失的 ADALINE（KP08）。第1章 KP08 提醒過：XOR 限制是可驗算的數學事實，但「一本書單獨造成整個研究寒冬」是過度簡化。

The straight-line limit needs more layers and nonlinearity (KP05); the generalization limit needs a method that picks a "good" line, such as ADALINE with a continuous loss (KP08). Chapter 1 KP08 reminded us that the XOR limitation is a verifiable mathematical fact, while "one book alone caused a whole research winter" is an oversimplification.

### 本單元重點整理（KP04 summary）

**一句話重點**：單一個 Perceptron 的邊界是直線，所以解不了 XOR（學不停）；就算資料可分，學到的直線也可能因起點不同而泛化不佳。

**One-sentence summary**: The boundary of one Perceptron is a straight line, so it cannot solve XOR (it never stops); and even on separable data the learned line may generalize poorly depending on where it started.

**白話說明**：XOR 的同類點在對角線上，拿尺在紙上怎麼畫都有一個點被分錯。另外，Perceptron 只求「全對」，不求「最好」，所以兩次訓練可以停在不同的線上，對新點的答案相反。

**Plain explanation**: The points of one XOR class sit on the diagonals, so however you draw a line with a ruler, one point is wrong. Also, the Perceptron only asks for "all right", not "best", so two trainings can stop at different lines that give opposite answers on a new point.

**正式定義**：XOR 不可由 $f(w_1x_1+w_2x_2+w_0)$ 算出（第1章 KP08 的證明）；Perceptron 在不可分資料上每輪都有更新；零訓練誤差不蘊含泛化。

**Formal definition**: XOR cannot be computed as $f(w_1x_1+w_2x_2+w_0)$ (the proof of Chapter 1 KP08); a Perceptron on non-separable data has updates in every pass; zero training error does not imply generalization.

**例子**：XOR 每輪更新 $3,3,4,4,\dots$；格點搜尋最多對 3/4；兩條零誤差邊界對 $(2.5,2.4)$ 給出類 1 與類 0。

**Example**: XOR has $3,3,4,4,\dots$ updates per pass; a grid search gets at most 3/4 right; two zero-error boundaries give class 1 and class 0 for $(2.5,2.4)$.

**常見迷思**：

- 「資料線性可分，Perceptron 就一定學得好」——不對。可分只保證它會**停**，不保證停在哪裡。
- 「Perceptron 學不會 XOR，是因為資料太少或學習率不對」——不是，是表示力的限制：沒有任何權重能做出 XOR。

**Common misconceptions**:

- "If the data are linearly separable, the Perceptron learns them well." No. Separability only guarantees that it **stops**, not where.
- "The Perceptron fails on XOR because of too little data or a wrong learning rate." No, it is a limit of expressive power: no weights at all can make XOR.

**圖**：本單元兩張圖，`lec02-kp04-xor-limit`（XOR 四點與每輪更新次數，功能 1、2）、`lec02-kp04-two-boundaries`（兩條邊界與新點，功能 1、3），以 `assets/figures/src/lec02-kp0204-figures.py` 繪製。歷史與出路在簡報用原生節點畫成時間軸。

**Figures**: Two figures in this unit, `lec02-kp04-xor-limit` (four XOR points and updates per pass; functions 1, 2) and `lec02-kp04-two-boundaries` (two boundaries and a new point; functions 1, 3), drawn by `assets/figures/src/lec02-kp0204-figures.py`. The history and the ways out are drawn as a timeline with native nodes in the slides.

**想深入**（選讀）：一條直線把平面分成兩個**凸**的半平面，而 XOR 兩類各自的凸包（兩點連線）在正方形中央相交，所以不可能被一條直線分開。這是第1章不等式證明的幾何版。

**Going deeper** (optional): A line splits the plane into two **convex** half-planes, and the convex hulls of the two XOR classes (the segments joining their points) cross at the centre of the square, so no line can separate them. This is the geometric version of the inequality proof of Chapter 1.

**小練習**：

1. 用第1章 KP08 的不等式法，證明單一神經元也做不出 XNOR（$(0,0),(1,1)\to1$、$(0,1),(1,0)\to0$）。（提示：把兩個輸出為 1 的點的條件相加，再與兩個輸出為 0 的條件比較。）
2. 為什麼 AND 做得出來而 XOR 做不出來？（提示：畫出輸出為 1 的點，看能不能用一條直線把它們與其餘的點隔開。）
3. 兩個 Perceptron 都對訓練資料 100% 正確，你要如何決定相信哪一個？（提示：想想「沒看過的資料」可以怎麼取得與使用。）

**Practice**:

1. Use the inequality method of Chapter 1 KP08 to show that one neuron cannot make XNOR either ($(0,0),(1,1)\to1$ and $(0,1),(1,0)\to0$). (Hint: add the conditions of the two points with output 1, then compare with the conditions of the two points with output 0.)
2. Why can AND be made but XOR cannot? (Hint: draw the points with output 1 and check whether one line can separate them from the rest.)
3. Two Perceptrons are both 100% right on the training data; how would you decide which to trust? (Hint: think about how unseen data could be obtained and used.)

**解題示範**（第 1 題）：

1. 條件：$(0,0)\to1$：$w_0\ge0$；$(1,1)\to1$：$w_1+w_2+w_0\ge0$；$(0,1)\to0$：$w_2+w_0<0$；$(1,0)\to0$：$w_1+w_0<0$。
2. 把後兩式相加：$w_1+w_2+2w_0<0$，即 $w_1+w_2+w_0<-w_0$。
3. 由第一式 $w_0\ge0$ 得 $-w_0\le0$，所以 $w_1+w_2+w_0<0$。
4. 這與第二式 $w_1+w_2+w_0\ge0$ 矛盾，所以這樣的權重不存在，XNOR 也做不出來。

**Worked solution** (exercise 1):

1. Conditions: $(0,0)\to1$: $w_0\ge0$; $(1,1)\to1$: $w_1+w_2+w_0\ge0$; $(0,1)\to0$: $w_2+w_0<0$; $(1,0)\to0$: $w_1+w_0<0$.
2. Add the last two: $w_1+w_2+2w_0<0$, that is $w_1+w_2+w_0<-w_0$.
3. The first condition $w_0\ge0$ gives $-w_0\le0$, so $w_1+w_2+w_0<0$.
4. This contradicts the second condition $w_1+w_2+w_0\ge0$, so such weights do not exist and XNOR cannot be made either.

**自我檢核**（來源四題，不附答案）：

| 題 / Question | A | B | C | D | 提示 / Hint |
|---|---|---|---|---|---|
| 1. Perceptron 演算法本質上只適用於？ / What is the algorithm essentially limited to? | 多分類任意資料 | 二元且線性可分的資料 | 任何迴歸問題 | 需要非線性邊界的資料 | 看第 1 節。 / See section 1. |
| 2. Minsky 與 Papert（1969）用哪個反例？ / Which counterexample did they use? | AND | OR | XOR | NOT | 同類點在對角線。 / The classes sit on the diagonals. |
| 3. 即使資料線性可分，為何仍可能泛化差？ / Why can it generalize poorly on separable data? | 收斂太慢 / Slow convergence | 邊界取決於初始權重，可能貼近某一類 / The boundary depends on the start and may hug one class | 損失函數連續 / The loss is continuous | 它需要非線性活化 / It needs a nonlinear activation | 看第 2 節的兩條邊界。 / See the two boundaries. |
| 4. ADALINE 改善泛化靠的是？ / What lets ADALINE improve generalization? | 改用非線性活化 | 用 LMS 最小化連續損失 | 堆疊更多層 | 把邊界改成曲線 | KP08 的主題。 / The topic of KP08. |

**留給下一個知識點的問題**：KP01 已經知道 AND、OR、NOT 都能由一個神經元做出來，而 XOR 可以寫成「至少一個為 1，而且不是兩個都為 1」。能不能把幾個神經元接起來，讓前一個的輸出當下一個的輸入？

**Question for the next knowledge point**: KP01 showed that AND, OR and NOT can each be made by one neuron, and XOR can be written as "at least one is 1, and not both are 1". Could we connect several neurons so that the output of one is the input of the next?

### 來源更正與取捨（Corrections to the source）

- 來源說 Minsky 與 Papert「嚴格證明」並使神經網路研究進入「第一次寒冬」。本講義保留 XOR 的數學結論，並依第1章 KP08 的處理，把「單獨造成寒冬」說成過度簡化。
- 來源的 XOR 論證只說「任一直線都分不開」。本講義引用第1章 KP08 的不等式證明，並補上實跑的更新次數與格點搜尋（`[補充]`，數字由程式驗算）。
- 來源說 ADALINE「泛化更好」；本講義改為有條件的說法：連續損失讓邊界不再貼著資料，通常較好，但不是必然（KP08 會用數字說明）。

- The source says Minsky and Papert "rigorously proved" it and that neural network research then entered "the first winter". This note keeps the mathematical conclusion about XOR and, following Chapter 1 KP08, calls "one book alone caused the winter" an oversimplification.
- The source's XOR argument only says "no line can separate it". This note cites the inequality proof of Chapter 1 KP08 and adds the updates per pass of a real run and a grid search (`[Added]`, numbers verified by program).
- The source says ADALINE "generalizes better"; this note makes it conditional: a continuous loss keeps the boundary from hugging the data and is usually better, but not necessarily (KP08 will show it with numbers).

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 互斥或 | XOR (exclusive or) |
| 線性可分 | linearly separable |
| 泛化 | generalization |
| 零訓練誤差 | zero training error |
| 凸包 | convex hull |
| 半平面 | half-plane |
