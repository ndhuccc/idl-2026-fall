# 第2週 KP02：猜錯才改權重，Perceptron 自己學出權重（A Perceptron changes its weights only when it is wrong）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)
>
> 週次：第2週（2026-09-17）｜主題：神經網路基礎：感知器與多層感知器
>
> Week 2 (2026-09-17) | Topic: Neural network fundamentals: perceptron and multilayer perceptron

本講義是第2章「感知器與多層感知器」的第二個知識點（KP02），承接 KP01 留下的問題：神經元的權重要人事先算好，能不能只給資料與正確答案，讓它自己把權重改對？第1章 KP08 已寫出感知機「只在錯分時更新」的 $\pm1$ 形式；這一節用 0/1 標籤把規則完整走一遍，從全 0 學出 OR 閘。先備知識是 KP01 與向量加法。內容依第2章主文重新講解，來源讀取日期為 2026-10-14；新增的釐清標為 `[補充]`。

This is the second knowledge point (KP02) of Chapter 2. It picks up the question KP01 left open: the weights of the neuron must be worked out by a person, so can we give it only data and the right answers and let it correct its own weights? Chapter 1 KP08 already wrote the "update only when wrong" rule of the perceptron in $\pm1$ form; this note walks through the rule fully with 0/1 labels and learns an OR gate from all zeros. It assumes KP01 and vector addition. It re-explains the main text of Chapter 2, accessed on 2026-10-14; clarifications that are new are marked `[Added]`.

[下載本單元投影片](../slides/lec02-kp02-perceptron-learning-rule.jbd.json)

## KP02 猜錯才改權重（A Perceptron changes its weights only when it is wrong）

**建議時間 / Time**：15 分鐘　**對應實作 / Related practice**：手算 Perceptron 學 OR 的每一次更新。

### 學習目標（Learning objectives）

讀完之後，你能寫出 $\mathbf{w}\leftarrow\mathbf{w}+\eta(y_i-o_i)\mathbf{x}_i$ 並說明三種誤差各自的動作；能說明為什麼一次更新一定讓這一筆的加權和往對的方向移動；能手算 Perceptron 從全 0 學 OR 的 5 次更新；能說出這條規則何時保證停下。

After reading, you can write $\mathbf{w}\leftarrow\mathbf{w}+\eta(y_i-o_i)\mathbf{x}_i$ and say what each of the three error cases does; explain why one update always moves the weighted sum of that sample toward the right side; compute by hand the 5 updates with which a Perceptron learns OR from all zeros; and say when the rule is guaranteed to stop.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{x}_i$ | 第 $i$ 筆輸入，已補上 $x_{i0}=1$（偏置併入 $\mathbf{w}$） | the $i$-th input, with $x_{i0}=1$ added (bias folded into $\mathbf{w}$) |
| $y_i\in\{0,1\}$ | 第 $i$ 筆的正確標籤 | correct label of the $i$-th sample |
| $o_i=f(\mathbf{w}^\top\mathbf{x}_i)$ | 目前權重的輸出，$f$ 是 KP01 的階梯 | output under the current weights, with $f$ the step of KP01 |
| $\eta>0$ | 學習率：每次修正的步長 | learning rate: the size of one correction |
| $y_i-o_i$ | 誤差，只可能是 $0,+1,-1$ | error, which can only be $0$, $+1$ or $-1$ |

### 1. 回饋迴圈：算、比、猜錯才改（A feedback loop: compute, compare, fix only when wrong）

KP01 的神經元每個權重都要人手算。Rosenblatt 在 1958 年的想法是加一個回饋：每看一筆資料，先算和 $s$、過階梯得到 $o_i$，再和正確標籤 $y_i$ 比；猜對就什麼都不做，猜錯才把權重往正確的方向推一步。這是**監督式**規則，由標籤與錯誤驅動，和古典 Hebbian 學習（一起活躍、連結變強）本質不同。

Every weight of the KP01 neuron had to be worked out by hand. Rosenblatt's idea of 1958 adds feedback: for each sample, compute the sum $s$, pass it through the step to get $o_i$, and compare with the correct label $y_i$; if right, do nothing; if wrong, push the weights one step in the right direction. This is a **supervised** rule, driven by labels and mistakes, essentially different from classical Hebbian learning (neurons that fire together strengthen their link).

### 2. 更新規則：三種誤差，三種動作（The update rule: three errors, three actions）

$$
\mathbf{w}\leftarrow\mathbf{w}+\eta\,(y_i-o_i)\,\mathbf{x}_i
$$

| 誤差 $y_i-o_i$ | 情形 | 對 $\mathbf{w}$ 的動作 | 加權和 $s$ |
|---|---|---|---|
| $0$ | 猜對 | 不動 | 不變 |
| $+1$ | 該放電卻沒放 | $\mathbf{w}+\eta\mathbf{x}_i$ | 上升 |
| $-1$ | 不該放電卻放了 | $\mathbf{w}-\eta\mathbf{x}_i$ | 下降 |

方向一定對：更新後 $s'=\mathbf{w}'^\top\mathbf{x}_i=s+\eta(y_i-o_i)\|\mathbf{x}_i\|^2$，而 $x_{i0}=1$ 使 $\|\mathbf{x}_i\|^2\ge1$。所以誤差為 $+1$ 時 $s$ 一定上升，為 $-1$ 時一定下降。也可以一次看一批 $b$ 筆再更新：$\mathbf{w}\leftarrow\mathbf{w}+\eta\sum_{i=1}^{b}(y_i-o_i)\mathbf{x}_i$；$b=1$ 就是逐筆版。

The direction is always right: after the update $s'=\mathbf{w}'^\top\mathbf{x}_i=s+\eta(y_i-o_i)\|\mathbf{x}_i\|^2$, and $x_{i0}=1$ makes $\|\mathbf{x}_i\|^2\ge1$. So $s$ rises when the error is $+1$ and falls when it is $-1$. You can also look at a batch of $b$ samples before updating: $\mathbf{w}\leftarrow\mathbf{w}+\eta\sum_{i=1}^{b}(y_i-o_i)\mathbf{x}_i$; $b=1$ is the one-at-a-time version.

### 3. 手算：從全 0 學 OR（By hand: learning OR from all zeros）

資料依序是 $(0,0)\to0$、$(0,1)\to1$、$(1,0)\to1$、$(1,1)\to1$。從 $(w_1,w_2,w_0)=(0,0,0)$ 開始，$\eta=1$，$f(s)=1$ 當 $s\ge0$。第一筆：$s=0$，$o=f(0)=1$，但 $y=0$，誤差 $-1$；$w_1,w_2$ 加上 $-1\cdot0=0$ 不變，$w_0$ 加上 $-1\cdot1=-1$，變成 $-1$。逐筆算下去，全部 5 次更新〔程式計算，每一步可手算驗證〕：

The data come in the order $(0,0)\to0$, $(0,1)\to1$, $(1,0)\to1$, $(1,1)\to1$. Start from $(w_1,w_2,w_0)=(0,0,0)$ with $\eta=1$ and $f(s)=1$ when $s\ge0$. First sample: $s=0$ and $o=f(0)=1$, but $y=0$, so the error is $-1$; $w_1,w_2$ get $-1\cdot0=0$ added and stay, and $w_0$ gets $-1\cdot1=-1$ added and becomes $-1$. Continuing, all 5 updates (computed by program; every step can be verified by hand):

| 輪 | 猜錯的點 | $y$ | 更新前 $(w_1,w_2,w_0)$ | $s$ | $o$ | 更新後 |
|---|---|---|---|---|---|---|
| 1 | (0,0) | 0 | (0,0,0) | 0 | 1 | (0,0,−1) |
| 1 | (0,1) | 1 | (0,0,−1) | −1 | 0 | (0,1,0) |
| 2 | (0,0) | 0 | (0,1,0) | 0 | 1 | (0,1,−1) |
| 2 | (1,0) | 1 | (0,1,−1) | −1 | 0 | (1,1,0) |
| 3 | (0,0) | 0 | (1,1,0) | 0 | 1 | (1,1,−1) |

第 1 輪的 $(1,0)$、$(1,1)$ 與第 2、3 輪的其餘點都已猜對，不更新；第 4 輪四筆全對，停止。最後的權重 $(1,1,-1)$ 正是 KP01 手算的 OR 閘（$\theta=1$）。差別是：這次是機器自己找到的。

In pass 1 the points $(1,0)$ and $(1,1)$, and the remaining points of passes 2 and 3, were already right and did not update; pass 4 is all right and learning stops. The final weights $(1,1,-1)$ are exactly the OR gate we worked out by hand in KP01 ($\theta=1$). The difference: this time the machine found them itself.

![從全 0 學 OR：第 1、2、3 輪結束後的決策線](../assets/figures/lec02-kp02-or-training.png)

**圖說 `lec02-kp02-or-training`**：先看左圖，第 1 輪結束線是水平的，$(0,0)$ 還被劃進放電區（紅圈）；再看中圖，線轉成斜線但仍穿過 $(0,0)$；最後看右圖，線移到 $x_1+x_2=1$，四個點全對。每次更新都把線推向猜錯的點。 / **Caption `lec02-kp02-or-training`:** Look first at the left panel: after pass 1 the line is horizontal and $(0,0)$ is still in the firing zone (red ring); the middle panel shows the line diagonal but still through $(0,0)$; the right panel shows the line at $x_1+x_2=1$ with all four points right. Each update pushes the line toward a point it got wrong.

### 4. 什麼時候停、什麼時候不停（When it stops and when it does not）

這條規則假設標籤只有兩類、輸入是數值向量，而且資料**線性可分**。[補充] 線性可分時，Perceptron 一定在有限次更新後停下（感知器收斂定理，本講義不證明）；不可分時永遠停不下來，KP04 的 XOR 就是這種情形。實作是雙層迴圈：外層是輪，內層逐筆，整輪沒有更新就停止。[補充] 若權重從 0 開始，$\eta$ 取任何正數只是把 $\mathbf{w}$ 整個放大或縮小，每個 $s$ 的正負不變，預測完全相同。

The rule assumes two classes, numeric input vectors and data that are **linearly separable**. [Added] On separable data the Perceptron always stops after finitely many updates (the perceptron convergence theorem, not proved in this note); on non-separable data it never stops, and the XOR of KP04 is such a case. In code it is a double loop, passes outside and samples inside, stopping after a pass with no update. [Added] If the weights start at 0, any positive $\eta$ only scales $\mathbf{w}$ up or down, the sign of every $s$ is unchanged, and the predictions are exactly the same.

### 本單元重點整理（KP02 summary）

**一句話重點**：Perceptron 每看一筆資料，猜對就不動，猜錯就把權重往正確方向推一步：$\mathbf{w}\leftarrow\mathbf{w}+\eta(y_i-o_i)\mathbf{x}_i$；線性可分時有限步後停下。

**One-sentence summary**: For each sample the Perceptron leaves its weights alone when right and pushes them one step in the right direction when wrong: $\mathbf{w}\leftarrow\mathbf{w}+\eta(y_i-o_i)\mathbf{x}_i$; on linearly separable data it stops after finitely many steps.

**白話說明**：像調一組旋鈕：試聽對了不碰，太小聲就把相關的旋鈕轉大，太大聲就轉小。輸入為 0 的那一路，權重完全不動。

**Plain explanation**: Like adjusting a bank of knobs: leave them alone if it sounds right, turn the relevant ones up if too quiet and down if too loud. A path whose input is 0 gets no change at all.

**正式定義**：$o_i=f(\mathbf{w}^\top\mathbf{x}_i)$，$\mathbf{w}\leftarrow\mathbf{w}+\eta(y_i-o_i)\mathbf{x}_i$；批次版 $\mathbf{w}\leftarrow\mathbf{w}+\eta\sum_{i=1}^{b}(y_i-o_i)\mathbf{x}_i$。

**Formal definition**: $o_i=f(\mathbf{w}^\top\mathbf{x}_i)$ and $\mathbf{w}\leftarrow\mathbf{w}+\eta(y_i-o_i)\mathbf{x}_i$; the batch version is $\mathbf{w}\leftarrow\mathbf{w}+\eta\sum_{i=1}^{b}(y_i-o_i)\mathbf{x}_i$.

**例子**：OR 資料、從 $(0,0,0)$、$\eta=1$：5 次更新後得 $(1,1,-1)$，第 4 輪全對。

**Example**: OR data from $(0,0,0)$ with $\eta=1$: after 5 updates the weights are $(1,1,-1)$ and pass 4 is all right.

**常見迷思**：

- 「Perceptron 每看一筆就把那一筆改對，而且不會再錯」——不一定。上表第 1 輪 $(0,0)$ 被修好，卻被 $(0,1)$ 的更新打壞，第 2 輪再錯。一次更新只保證**這一筆**的 $s$ 往對的方向走。
- 「學習率越大學得越快」——從 0 開始時 $\eta$ 不影響預測；一般情形步長過大會來回跳。

**Common misconceptions**:

- "A Perceptron fixes each sample as soon as it sees it and never gets it wrong again." Not necessarily. In the table, $(0,0)$ is fixed in pass 1 but spoiled by the update for $(0,1)$ and wrong again in pass 2. One update only guarantees that **this** sample's $s$ moves the right way.
- "A larger learning rate learns faster." From a zero start $\eta$ does not change the predictions; in general a step that is too large makes the weights jump back and forth.

**圖**：本單元一張圖，`lec02-kp02-or-training`（決策線隨輪次移動，功能 1、3），以 `assets/figures/src/lec02-kp0204-figures.py` 繪製。回饋迴圈的流程圖在簡報中用原生節點與連線畫出。

**Figure**: One figure in this unit, `lec02-kp02-or-training` (the decision line moving across passes; functions 1, 3), drawn by `assets/figures/src/lec02-kp0204-figures.py`. The feedback-loop diagram is drawn with native nodes and connectors in the slides.

**想深入**（選讀）：來源指出 Rosenblatt 最初靈感來自 Hebbian 學習，並強調輸入與輸出神經元之間的隨機連接；最後採用的規則是由標籤與錯誤驅動的監督式規則。

**Going deeper** (optional): The source notes that Rosenblatt's first inspiration was Hebbian learning, with random connections between input and output neurons; the rule finally adopted is a supervised rule driven by labels and mistakes.

**小練習**：

1. 權重 $(w_1,w_2,w_0)=(1,1,-1)$，輸入 $(1,0)$，標籤 $y=0$，$\eta=1$。輸出是多少？要更新嗎？更新後的權重是多少？（提示：先算 $s$，再看 $y-o$。）
2. OR 資料的順序改成 $(1,1)$、$(1,0)$、$(0,1)$、$(0,0)$，從全 0 開始，第一次更新發生在哪一筆？（提示：全 0 時每筆的 $s$ 都是 0，$o$ 都是 1。）
3. 為什麼權重從 0 開始時，$\eta=2$ 與 $\eta=1$ 的預測完全相同？（提示：算算兩組權重是不是剛好差 2 倍。）

**Practice**:

1. With weights $(w_1,w_2,w_0)=(1,1,-1)$, input $(1,0)$, label $y=0$ and $\eta=1$: what is the output, is an update needed, and what are the weights afterwards? (Hint: compute $s$, then look at $y-o$.)
2. Change the OR data order to $(1,1)$, $(1,0)$, $(0,1)$, $(0,0)$ and start from all zeros. At which sample does the first update happen? (Hint: with all-zero weights every $s$ is 0 and every $o$ is 1.)
3. Why do $\eta=2$ and $\eta=1$ give exactly the same predictions when the weights start at 0? (Hint: check whether the two sets of weights differ by exactly a factor of 2.)

**解題示範**（第 1 題）：

1. 加權和：$s=1\cdot1+1\cdot0-1=0$。
2. 輸出：$o=f(0)=1$，但 $y=0$，所以猜錯，誤差 $y-o=-1$，要更新。
3. 更新：$\mathbf{x}=(1,0)$，補上 $x_0=1$ 得 $(x_1,x_2,x_0)=(1,0,1)$；$\mathbf{w}\leftarrow(1,1,-1)+1\cdot(-1)\cdot(1,0,1)=(0,1,-2)$。
4. 檢查：新的 $s=0\cdot1+1\cdot0-2=-2<0$，輸出 0，與標籤一致；$s$ 下降了 $2=\eta\|\mathbf{x}\|^2$。

**Worked solution** (exercise 1):

1. Sum: $s=1\cdot1+1\cdot0-1=0$.
2. Output: $o=f(0)=1$ but $y=0$, so it is wrong with error $y-o=-1$ and must be updated.
3. Update: $\mathbf{x}=(1,0)$, and adding $x_0=1$ gives $(x_1,x_2,x_0)=(1,0,1)$; $\mathbf{w}\leftarrow(1,1,-1)+1\cdot(-1)\cdot(1,0,1)=(0,1,-2)$.
4. Check: the new $s=0\cdot1+1\cdot0-2=-2<0$, so the output is 0, matching the label; $s$ fell by $2=\eta\|\mathbf{x}\|^2$.

**自我檢核**（來源四題，不附答案）：

| 題 / Question | A | B | C | D | 提示 / Hint |
|---|---|---|---|---|---|
| 1. Perceptron 於 1958 年由誰提出？ / Who proposed the Perceptron in 1958? | Widrow 與 Hoff | Rosenblatt（Cornell Aeronautical Laboratory） | McCulloch 與 Pitts | Minsky 與 Papert | 本講義第 1 節。 / Section 1. |
| 2. Perceptron 的學習規則與 Hebbian learning 的關係？ / How does its rule relate to Hebbian learning? | 完全相同 / Identical | 監督式，本質與古典 Hebbian 不同 / Supervised, essentially different | 完全不學習 / No learning | Hebbian 是後繼 / Hebbian came later | 看有沒有標籤。 / Are labels used? |
| 3. 當 $o_i=y_i$ 時，式 $\mathbf{w}\leftarrow\mathbf{w}+\eta(y_i-o_i)\mathbf{x}_i$ 會發生什麼？ / What happens when $o_i=y_i$? | 權重放大 $\eta$ 倍 / Scaled by $\eta$ | 更新量為 $\mathbf{0}$，不更新 / Zero update | 權重反轉 / Flipped | 學習率歸零 / $\eta$ becomes 0 | 先算 $y_i-o_i$。 / Compute $y_i-o_i$. |
| 4. 批次版與單樣本版的關係？ / How do the batch and one-sample versions relate? | 互相矛盾 / Contradictory | 批次版對各樣本誤差項求和，單樣本版去掉求和 / Batch sums the error terms; one-sample drops the sum | 批次版不收斂 / Batch never converges | 只在分類時等價 / Equal only for classification | 看 $b=1$。 / Look at $b=1$. |

**留給下一個知識點的問題**：標籤用 0 和 1 時，誤差有三種情形。如果改用 $+1$ 和 $-1$ 當標籤，更新規則能不能寫得更短，而且看得出幾何意義？

**Question for the next knowledge point**: With labels 0 and 1 the error has three cases. If we use $+1$ and $-1$ as labels, can the update rule be written more briefly and reveal its geometric meaning?

### 來源更正與取捨（Corrections to the source）

- 來源把 Perceptron 的「錯了就改」描述成「本質上和 Hebbian 不同」，本講義保留這個結論並加上差別：Perceptron 由標籤與錯誤驅動。
- 來源提到 1958 年的新聞報導與「能走路、說話」的誇大說法，本講義略去，因為與學習規則無關，也不是可驗算的事實。
- 來源沒有給 Perceptron 的逐步數值例，上面的 OR 例子是為了驗算規則而補充（`[補充]`），數字由程式驗證。

- The source describes the Perceptron's "fix it when wrong" as "essentially different from Hebbian"; this note keeps that conclusion and adds the difference: the Perceptron is driven by labels and mistakes.
- The source mentions the news reports of 1958 and their exaggerated "walk and talk" claims; this note omits them because they have nothing to do with the learning rule and are not verifiable facts.
- The source gives no step-by-step numeric example for the Perceptron; the OR example above is added to check the rule (`[Added]`), with numbers verified by program.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 學習率 | learning rate |
| 學習規則 | learning rule |
| 誤差 | error |
| 輪 | pass (epoch) |
| 線性可分 | linearly separable |
| 監督式學習 | supervised learning |
| 感知器收斂定理 | perceptron convergence theorem |
| 批次 | batch |
