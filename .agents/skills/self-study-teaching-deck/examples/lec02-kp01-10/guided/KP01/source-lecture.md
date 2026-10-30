# 第2週 KP01：神經元把訊號加總，過門檻才放電（A neuron adds its signals and fires only past a threshold）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)
>
> 週次：第2週（2026-09-17）｜主題：神經網路基礎：感知器與多層感知器
>
> Week 2 (2026-09-17) | Topic: Neural network fundamentals: perceptron and multilayer perceptron

本講義是第2章「感知器與多層感知器」的第一個知識點（KP01）。第1章 KP08 已經從歷史角度介紹過 McCulloch–Pitts 單元是邏輯閘；這一節只做三件事：看懂人工神經元模仿了生物神經元的什麼、把它寫成一條式子、並用同一顆神經元手算出邏輯閘。先備知識是向量內積與不等式。內容依第2章主文與教授備課資料重新講解，來源讀取日期為 2026-10-14；新增的釐清標為 `[補充]`。

This is the first knowledge point (KP01) of Chapter 2, "Perceptron and multilayer perceptron". Chapter 1 KP08 already introduced the McCulloch–Pitts unit as a logic gate from a historical angle; this note does three things only: see what an artificial neuron copies from a biological one, write it as one formula, and compute logic gates by hand with the same neuron. It assumes vector inner products and inequalities. It re-explains the main text and teaching guide of Chapter 2, accessed on 2026-10-14; clarifications that are new are marked `[Added]`.

[下載本單元投影片](../slides/lec02-kp01-biological-neuron-and-mp-unit.jbd.json)

## KP01 神經元把訊號加總，過門檻才放電（A neuron adds its signals and fires only past a threshold）

**建議時間 / Time**：15 分鐘　**對應實作 / Related practice**：手算一顆神經元的輸出，並調閾值做出 AND 與 OR。

### 學習目標（Learning objectives）

讀完之後，你能說出生物神經元的四個部件與「累積、過門檻、才放電」的機制；能把它對應到加權和、偏置與活化函數三個元件；能手算 $o=f(\sum_jw_jx_j+w_0)$；能只改閾值或權重，讓同一顆神經元變成 AND、OR、NOT。

After reading, you can name the four parts of a biological neuron and the mechanism "accumulate, pass a threshold, then fire"; map it onto the three parts weighted sum, bias and activation function; compute $o=f(\sum_jw_jx_j+w_0)$ by hand; and turn the same neuron into AND, OR and NOT by changing only the threshold or the weights.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $x_j$ | 第 $j$ 路輸入，本節只取 0 或 1 | the $j$-th input, only 0 or 1 in this note |
| $w_j$ | 第 $j$ 路的權重，代表這一路多重要 | weight of the $j$-th input, its importance |
| $w_0$ | 偏置，等於負的閾值 $-\theta$ | bias, equal to the negative threshold $-\theta$ |
| $\theta$ | 閾值：加權和要達到的門檻 | threshold the weighted sum must reach |
| $s=\sum_jw_jx_j+w_0$ | 活化前的加權和 | pre-activation weighted sum |
| $f(s)$ | 二元階梯函數：$s\ge0$ 輸出 1，否則 0 | binary step: 1 when $s\ge0$, otherwise 0 |
| $o$ | 輸出，1 代表放電、0 代表沉默 | output; 1 means fire, 0 means silent |

### 1. 生物神經元：累加之後才決定放不放電（A biological neuron sums first, then decides）

一顆神經元有四個要認識的部件。**樹突**（dendrites）像天線，接收其他神經元的訊號；**細胞體**（soma）把收到的訊號加在一起，細胞核在它裡面；**軸突**（axon）把放電的脈衝送出去；**突觸**（synapse）是軸突末端與下一顆神經元之間的微小縫隙，訊號在這裡以化學方式接力。

A neuron has four parts to know. **Dendrites** act like antennas and receive signals from other neurons; the **soma** adds the received signals together, with the nucleus inside it; the **axon** carries the firing pulse outward; the **synapse** is the tiny gap between the end of an axon and the next neuron, where the signal is relayed chemically.

關鍵在機制：細胞體把各路訊號累積起來，只有**總量**跨過門檻，神經元才放電；沒跨過就完全沉默。所以它是開關，不是音量鈕：任何單一訊號再大，總量不夠就沒有反應。人工神經元模仿的正是這個機制，而不是外形。

The key is the mechanism: the soma accumulates the signals, and the neuron fires only when the **total** crosses a threshold; otherwise it stays completely silent. It is a switch, not a volume knob: however loud one signal is, there is no reaction if the total is too small. An artificial neuron copies this mechanism, not the shape.

![生物神經元的樹突、細胞體、軸突與突觸，以及「累積、過門檻、放電」的流程](../assets/figures/lec02-kp01-neuron-anatomy.png)

**圖說 `lec02-kp01-neuron-anatomy`**：先看左邊三條樹突，各帶一路訊號；再看中間細胞體，訊號在這裡相加；最後看右邊，放電後脈衝沿軸突經突觸傳給下一顆神經元。底部一行是整個機制的縮寫。 / **Caption `lec02-kp01-neuron-anatomy`:** Look first at the three dendrites, each carrying one signal; then at the soma where the signals add; finally at the axon and synapse that pass the pulse on. The bottom line is the whole mechanism.

### 2. 寫成式子：加權和、偏置、活化函數（Writing it as a formula）

把「多路訊號」寫成輸入 $x_1,\dots,x_d$，「每一路的重要性」寫成權重 $w_1,\dots,w_d$，「門檻」搬進偏置 $w_0=-\theta$，「放電或沉默」寫成階梯函數 $f$。McCulloch 與 Pitts 在 1943 年提出的神經元就是：

Write the "many signals" as inputs $x_1,\dots,x_d$, the "importance of each signal" as weights $w_1,\dots,w_d$, move the "threshold" into the bias $w_0=-\theta$, and write "fire or stay silent" as the step function $f$. The neuron proposed by McCulloch and Pitts in 1943 is:

$$
o=f\!\left(\sum_{j=1}^{d}w_jx_j+w_0\right),\qquad f(s)=\begin{cases}1&s\ge0\\0&s<0\end{cases}
$$

第1章 KP08 提過原始的 McCulloch–Pitts 單元輸入是 0 或 1、權重固定、不學習；式子本身允許任意實數權重，這一點留給 KP02 使用。$w_0=-\theta$ 這個換法很重要：「總和 $\sum x_j\ge\theta$」與「$\sum x_j+w_0\ge0$」是同一個條件。

Chapter 1 KP08 noted that the original McCulloch–Pitts unit has 0/1 inputs, fixed weights and no learning; the formula itself allows any real weights, which KP02 will use. The switch $w_0=-\theta$ matters: "the sum $\sum x_j\ge\theta$" and "$\sum x_j+w_0\ge0$" are the same condition.

| 生物機制 | 數學元件 | 符號 |
|---|---|---|
| 多路訊號 / signals | 輸入 / input | $x_1,\dots,x_d$ |
| 每路的重要性 / importance | 權重 / weight | $w_1,\dots,w_d$ |
| 門檻 / threshold | 偏置 / bias | $w_0=-\theta$ |
| 放電或沉默 / fire or not | 活化函數 / activation | $f(s)$ |

### 3. 手算一遍：同一組輸入，只換門檻（A hand computation: same inputs, different thresholds）

主例子：三路輸入 $\mathbf{x}=(1,0,1)$，權重全為 1。總和 $z=1+0+1=2$，$x_2=0$ 不貢獻。對三個門檻各算一次，$s=z-\theta$：

Main example: three inputs $\mathbf{x}=(1,0,1)$ with all weights 1. The total is $z=1+0+1=2$, and $x_2=0$ contributes nothing. Evaluate three thresholds with $s=z-\theta$:

| $\theta$ | $w_0=-\theta$ | $s=z+w_0$ | $o=f(s)$ |
|---|---|---|---|
| 1 | $-1$ | $2-1=1$ | 1 |
| 2 | $-2$ | $2-2=0$ | 1 |
| 3 | $-3$ | $2-3=-1$ | 0 |

總和不變，只有門檻在變。$\theta=2$ 時 $s=0$ 剛好達標，所以仍然放電，這就是階梯函數用 $s\ge0$ 的原因。

The total never changes; only the threshold does. At $\theta=2$ we get $s=0$, which just reaches the bar, so the neuron still fires; this is why the step uses $s\ge0$.

![三路輸入的總和 z=2，與 θ=1、2、3 三條門檻線](../assets/figures/lec02-kp01-mp-threshold.png)

**圖說 `lec02-kp01-mp-threshold`**：先看橫條的長度，它就是總和 $z=2$（$x_1$ 與 $x_3$ 各貢獻 1）；再看三條虛線，落在橫條右端或內側的門檻，神經元就放電。 / **Caption `lec02-kp01-mp-threshold`:** Look first at the length of the bar, the total $z=2$; then at the three dashed lines: a threshold at or inside the right end of the bar means the neuron fires.

### 4. 只改閾值或權重，就是邏輯閘（Changing only the threshold or a weight gives logic gates）

兩個輸入、權重全為 1：$\theta=2$ 要兩路都為 1 才放電，這是 **AND**（$s=x_1+x_2-2$）；$\theta=1$ 只要有一路為 1 就放電，這是 **OR**（$s=x_1+x_2-1$）。

With two inputs and all weights 1, $\theta=2$ needs both inputs to be 1, which is **AND** ($s=x_1+x_2-2$); $\theta=1$ needs at least one input to be 1, which is **OR** ($s=x_1+x_2-1$).

| $(x_1,x_2)$ | AND：$s$，$o$ | OR：$s$，$o$ |
|---|---|---|
| (0,0) | $-2$，0 | $-1$，0 |
| (0,1) | $-1$，0 | $0$，1 |
| (1,0) | $-1$，0 | $0$，1 |
| (1,1) | $0$，1 | $1$，1 |

**[補充] NOT 閘。** 取 $w=-1$、$w_0=0.5$：$x=0$ 時 $s=0.5$，$o=1$；$x=1$ 時 $s=-0.5$，$o=0$。輸入變大要讓 $s$ 變小，所以權重必須是負的；全部權重為正的神經元做不出 NOT。

**[Added] The NOT gate.** Take $w=-1$ and $w_0=0.5$: at $x=0$ we get $s=0.5$ and $o=1$; at $x=1$ we get $s=-0.5$ and $o=0$. A larger input must lower $s$, so the weight must be negative; a neuron with all-positive weights cannot make NOT.

![AND、OR、NOT 三個邏輯閘的輸入點、決策線與放電區域](../assets/figures/lec02-kp01-mp-gates.png)

**圖說 `lec02-kp01-mp-gates`**：先看左邊兩張圖的四個角，實心點輸出 1、空心點輸出 0；再看斜線 $x_1+x_2=\theta$，線上與右上方是放電區。AND 與 OR 只差線的位置；最右邊 NOT 只有一個輸入，所以邊界是一條垂直線。 / **Caption `lec02-kp01-mp-gates`:** Look first at the four corners of the two left panels, filled for output 1 and open for output 0; then at the line $x_1+x_2=\theta$, with the firing zone on and above it. AND and OR differ only in where the line sits; NOT, on the right, has one input, so its boundary is a vertical line.

### 本單元重點整理（KP01 summary）

**一句話重點**：人工神經元模仿的是「把訊號加起來、過門檻才放電」的機制，寫成 $o=f(\sum_jw_jx_j+w_0)$；只改閾值或權重，同一顆神經元就能當 AND、OR、NOT，但權重要由人事先算好。

**One-sentence summary**: An artificial neuron copies the mechanism "add up the signals and fire only past a threshold", written $o=f(\sum_jw_jx_j+w_0)$; changing only the threshold or the weights turns the same neuron into AND, OR and NOT, but a person must work out the weights beforehand.

**白話說明**：每顆神經元先問「證據加起來夠不夠」，夠就回答 1，不夠就回答 0。證據的重要性由權重決定，門檻由偏置決定。

**Plain explanation**: Each neuron first asks "do the pieces of evidence add up to enough?" and answers 1 if so, 0 otherwise. The importance of each piece of evidence is set by the weights, and the bar by the bias.

**正式定義**：$o=f(\sum_{j=1}^{d}w_jx_j+w_0)$，$f(s)=1$ 當 $s\ge0$、否則 0，$w_0=-\theta$。

**Formal definition**: $o=f(\sum_{j=1}^{d}w_jx_j+w_0)$, with $f(s)=1$ when $s\ge0$ and 0 otherwise, and $w_0=-\theta$.

**例子**：$\mathbf{x}=(1,0,1)$、權重全為 1：$z=2$；$\theta=1,2,3$ 的 $s$ 是 $1,0,-1$，輸出 $1,1,0$。兩輸入：AND 用 $w_0=-2$，OR 用 $w_0=-1$，NOT 用 $w=-1$、$w_0=0.5$。

**Example**: $\mathbf{x}=(1,0,1)$ with all weights 1: $z=2$; for $\theta=1,2,3$ the sums $s$ are $1,0,-1$ and the outputs $1,1,0$. With two inputs, AND uses $w_0=-2$, OR uses $w_0=-1$, and NOT uses $w=-1$, $w_0=0.5$.

**常見迷思**：

- 「輸入越強，神經元輸出越大」——不是，輸出只回答「有沒有跨過門檻」。以 $\theta=2$ 為例，$z=2$ 與 $z=3$ 都輸出 1，$z=1$ 輸出 0。
- 「門檻要用 $>$，剛好等於不算」——本講義用 $s\ge0$，$z=\theta$ 時仍放電；換成 $>$ 只是約定不同，但同一份計算要一致。

**Common misconceptions**:

- "A stronger input gives a larger output." No: the output only says whether the threshold was crossed. At $\theta=2$, $z=2$ and $z=3$ both give 1 and $z=1$ gives 0.
- "The threshold test must be $>$, and equality does not count." This note uses $s\ge0$, so the neuron still fires at $z=\theta$; using $>$ is only another convention, but one computation must use one convention throughout.

**圖**：本單元三張圖：`lec02-kp01-neuron-anatomy`（機制，功能 1）、`lec02-kp01-mp-threshold`（總和與門檻，功能 1、2）、`lec02-kp01-mp-gates`（決策線，功能 1、3），以 `assets/figures/src/lec02-kp01-figures.py` 繪製。

**Figures**: Three figures: `lec02-kp01-neuron-anatomy` (the mechanism; function 1), `lec02-kp01-mp-threshold` (total versus threshold; functions 1, 2) and `lec02-kp01-mp-gates` (decision lines; functions 1, 3), drawn by `assets/figures/src/lec02-kp01-figures.py`.

**想深入**（選讀）：McCulloch 與 Pitts 證明，把這種神經元連成網路，可以算出任何邏輯函數。所以這顆神經元已經有計算力；缺的是**學習**，也就是權重不用人算。下一個知識點的 Perceptron 補上這一環。

**Going deeper** (optional): McCulloch and Pitts showed that networks of such neurons can compute any logic function. So the neuron already has computing power; what it lacks is **learning**, meaning weights that need not be worked out by a person. The Perceptron of the next knowledge point fills this gap.

**小練習**：

1. 三路輸入 $\mathbf{x}=(1,1,0)$，權重全為 1，$\theta=2$。輸出是多少？（提示：先算 $z$，再算 $s=z-\theta$。）
2. 三路輸入、權重全為 1，要「至少兩路為 1 才放電」，$\theta$ 該取多少？（提示：試 $(1,1,0)$ 與 $(1,0,0)$，找能分開它們的 $\theta$。）
3. 為什麼全部權重為正的神經元做不出 NOT？（提示：輸入由 0 變 1 時，$s$ 變大還是變小？）

**Practice**:

1. Three inputs $\mathbf{x}=(1,1,0)$, all weights 1 and $\theta=2$. What is the output? (Hint: compute $z$ first, then $s=z-\theta$.)
2. With three inputs and all weights 1, which $\theta$ makes the neuron fire only when at least two inputs are 1? (Hint: try $(1,1,0)$ and $(1,0,0)$ and find the $\theta$ that separates them.)
3. Why can a neuron with all-positive weights not make NOT? (Hint: when an input goes from 0 to 1, does $s$ grow or shrink?)

**解題示範**（第 1 題）：

1. 總和：$z=1+1+0=2$。
2. 偏置 $w_0=-\theta=-2$，所以 $s=z+w_0=2-2=0$。
3. 階梯函數在 $s\ge0$ 輸出 1，所以 $o=f(0)=1$。
4. 檢查：$z=2$ 剛好等於 $\theta$，與第 3 節 $\theta=2$ 的列一致。

**Worked solution** (exercise 1):

1. Total: $z=1+1+0=2$.
2. The bias is $w_0=-\theta=-2$, so $s=z+w_0=2-2=0$.
3. The step outputs 1 when $s\ge0$, so $o=f(0)=1$.
4. Check: $z=2$ equals $\theta$ exactly, consistent with the $\theta=2$ row of section 3.

**自我檢核**（來源四題，不附答案）：

| 題 / Question | A | B | C | D | 提示 / Hint |
|---|---|---|---|---|---|
| 1. 生物神經元中，負責把收到的訊號整合起來的是？ / Which part integrates the received signals? | 樹突 / Dendrites | 突觸 / Synapse | 細胞體 / Soma | 軸突 / Axon | 想「累積」發生在哪裡。 / Where does the accumulation happen? |
| 2. 人工神經元主要模仿生物神經元的什麼？ / What does the artificial neuron mainly copy? | 完整的細胞代謝 / Cell metabolism | 累積訊號、過閾值、開關式發火 / Sum, threshold, switch-like firing | 神經元的形狀 / The shape | 遺傳物質複製 / DNA copying | 看圖底部那一行。 / Read the bottom line of the figure. |
| 3. McCulloch–Pitts 模型由誰於何年提出？ / Who proposed the model, and when? | Rosenblatt, 1958 | McCulloch 與 Pitts, 1943 | Widrow 與 Hoff, 1960 | Minsky 與 Papert, 1969 | 本講義第 2 節。 / Section 2 of this note. |
| 4. 式中 $f$ 在這個模型裡是？ / What is $f$ in this model? | Sigmoid | ReLU | 二元階梯函數 / Binary step | tanh | 看符號表。 / See the symbol table. |

**留給下一個知識點的問題**：這顆神經元的權重與閾值都要人事先算好。能不能只給它資料與正確答案，讓它自己把權重改對？

**Question for the next knowledge point**: The weights and threshold of this neuron must be worked out by a person beforehand. Can we give it only data and the right answers, and let it correct its own weights?

### 來源更正與取捨（Corrections to the source）

- 來源把 NOT 閘寫成「等權、$\theta=0.5$」。權重全為正時，輸入變大不可能讓輸出由 1 變 0，所以本講義改用負權重（`[補充]`）。
- 來源把 McCulloch–Pitts 寫成「所有權重相等且固定」，但式 (1) 同時寫成可學習的權重。本講義說明：原始單元的權重固定，式子本身則允許任意權重，後者留給 KP02。
- 來源的互動示範用 $z>\theta$ 放電，圖與式 (1) 用「達到」門檻。本講義統一用 $s\ge0$（$z\ge\theta$ 放電）。

- The source writes NOT as an "equal-weight, $\theta=0.5$" gate. With all weights positive, a larger input can never turn the output from 1 to 0, so this note uses a negative weight (`[Added]`).
- The source describes McCulloch–Pitts as "all weights equal and fixed", yet writes formula (1) with learnable weights. This note says: the original unit has fixed weights, while the formula itself allows any weights, which KP02 uses.
- The source's interactive demo fires on $z>\theta$, while its figure and formula (1) use "reaches the threshold". This note uses $s\ge0$ throughout (fire when $z\ge\theta$).

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 樹突 | dendrites |
| 細胞體 | soma |
| 軸突 | axon |
| 突觸 | synapse |
| 閾值 | threshold |
| 加權和 | weighted sum |
| 偏置 | bias |
| 權重 | weight |
| 活化函數 | activation function |
| 二元階梯函數 | binary step function |
| 邏輯閘 | logic gate |
