# 第8週 KP01：序列與循環網路，用同一組權重記住過去（Sequences and recurrent networks: remembering the past with one set of weights）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第8週「循環神經網路：RNN、LSTM 與 GRU」的第一個知識點（KP01）。前幾週的網路一次看一張圖，看完就忘。這一節問：如果資料是一句話、一段聲音，前後順序很重要，網路要怎麼「記得」前面看過什麼？

This is the first knowledge point (KP01) of Week 8, "Recurrent neural networks: RNN, LSTM and GRU". The networks of the previous weeks look at one image at a time and forget it afterwards. This note asks: if the data is a sentence or a piece of sound, where order matters, how can a network "remember" what it has seen before?

先備知識是第2週的多層感知器（MLP）與 $\tanh$ 激活函數，以及矩陣乘法的維度規則。第一次出現的詞與符號都會定義。所有數字都由 `assets/figures/src/lec08-kp01-figures.py` 算出，可以自己重算。

The prerequisites are the multilayer perceptron (MLP) and the $\tanh$ activation of Week 2, and the dimension rule of matrix multiplication. Every term and symbol is defined when it first appears. Every number is computed by `assets/figures/src/lec08-kp01-figures.py`, so you can recompute it yourself.

## KP01 序列與循環網路，用同一組權重記住過去（Sequences and recurrent networks: remembering the past with one set of weights）

### 學習目標（Learning objectives）

讀完之後，你能說出前饋網路處理序列時的兩個困難：長度不固定、沒有記憶。你能寫出 RNN 的三條前向公式，並說出 $\mathbf{U}$、$\mathbf{W}$、$\mathbf{V}$ 各自的工作與維度。你能解釋「參數共享」為什麼讓參數個數不隨序列長度增加。你也能用純量例子手算兩個時間步，並說明 $W=0$ 時網路為什麼會失去記憶。

After reading, you can name the two difficulties a feedforward network has with sequences: lengths that vary and no memory. You can write the three forward equations of an RNN and state the job and the dimensions of $\mathbf{U}$, $\mathbf{W}$ and $\mathbf{V}$. You can explain why "parameter sharing" keeps the number of parameters from growing with the sequence length. You can also compute two time steps of a scalar example by hand and explain why the network loses its memory when $W=0$.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $t$ | 時間步，第幾個元素（$t=1,2,\dots,T$） | time step, which element ($t=1,2,\dots,T$) |
| $T$ | 序列長度 | sequence length |
| $\mathbf{x}_t\in\mathbb{R}^{d}$ | 第 $t$ 步的輸入，$d$ 維 | input at step $t$, of dimension $d$ |
| $\mathbf{h}_t\in\mathbb{R}^{p}$ | 第 $t$ 步的隱藏狀態，$p$ 維 | hidden state at step $t$, of dimension $p$ |
| $\mathbf{y}_t\in\mathbb{R}^{q}$ | 第 $t$ 步的輸出，$q$ 維 | output at step $t$, of dimension $q$ |
| $\mathbf{i}_t\in\mathbb{R}^{p}$ | 第 $t$ 步進入 $\tanh$ 之前的淨輸入 | net input at step $t$, before $\tanh$ |
| $\mathbf{U}\in\mathbb{R}^{p\times d}$ | 輸入到隱藏狀態的權重 | input-to-hidden weights |
| $\mathbf{W}\in\mathbb{R}^{p\times p}$ | 上一步狀態到這一步狀態的權重 | hidden-to-hidden (state transition) weights |
| $\mathbf{V}\in\mathbb{R}^{q\times p}$ | 隱藏狀態到輸出的權重 | hidden-to-output weights |
| $\mathbf{b}_i,\ \mathbf{b}_y$ | 隱藏層與輸出層的偏置 | biases of the hidden and output layers |

### 1. 前饋網路讀序列，卡在哪裡（Where a feedforward network gets stuck on sequences）

**序列資料**（sequence data）是有先後順序、而且長度不固定的資料。一句話是一串詞，一段語音是一串聲音片段，一檔股票是一串每天的價格。順序一變，意思就變：「警察追小偷」和「小偷追警察」用的字一模一樣，意思卻相反。

**Sequence data** is data with an order and a length that is not fixed. A sentence is a string of words, speech is a string of sound segments, a stock is a string of daily prices. Change the order and the meaning changes: "the police chase the thief" and "the thief chases the police" use exactly the same words but mean opposite things.

第2週的前饋網路（feedforward network，又稱 MLP）有兩個困難。第一，它的輸入層有固定的寬度；句子有時 $5$ 個詞，有時 $50$ 個詞，塞不進同一個輸入層。第二，它一次只看一筆輸入，看完就丟；讀到第三個詞時，它已經不記得前兩個詞是什麼。

The feedforward network of Week 2 (also called an MLP) has two difficulties. First, its input layer has a fixed width; a sentence may have $5$ words or $50$, which cannot fit the same input layer. Second, it looks at one input at a time and then drops it; when it reaches the third word it no longer remembers the first two.

有時要記的東西隔得很遠。「我在法國出生……（中間很多句）……所以我會說法語。」要猜出最後的「法語」，網路得記住很前面的「法國」。這叫**長距離依賴**（long-range dependency）。本週後面的 KP04 會看到，這件事對 RNN 也很難。

Sometimes what must be remembered lies far back. "I was born in France ... (many sentences) ... so I speak French." To guess the final "French", the network must remember "France" from long before. This is called a **long-range dependency**. KP04 later this week shows that this is hard even for an RNN.

常見錯誤：以為把整句話攤平成一個長向量，丟進 MLP 就好。這樣句子長度一變，輸入層就得重新設計；而且同一個詞出現在不同位置，會由不同的權重處理，學到的東西無法在位置之間共用。

Common error: thinking that you can flatten the whole sentence into one long vector and feed it to an MLP. Then every change in sentence length forces a new input layer; and the same word at different positions is handled by different weights, so what is learned cannot be shared across positions.

### 2. 把過去壓成一個狀態（Squeezing the past into one state）

要解決這兩個困難，借用**動態系統**（dynamical system）的想法：系統在每一刻有一個**狀態**，下一刻的狀態由「上一刻的狀態」和「這一刻的輸入」決定。寫成式子：

To solve both difficulties, borrow the idea of a **dynamical system**: the system has a **state** at every moment, and the next state is decided by "the previous state" and "the current input". As an equation:

$$
\mathbf{h}_t=f_\theta(\mathbf{h}_{t-1},\mathbf{x}_t).
$$

這裡的 $\mathbf{h}_t$ 叫**隱藏狀態**（hidden state）：它是一個向量，把第 $1$ 步到第 $t$ 步看過的東西濃縮成一份摘要。$f_\theta$ 是一個帶參數 $\theta$ 的函數。每走一步，網路就拿舊摘要和新輸入，算出新摘要。序列多長都沒關係，一步一步做下去就好。

Here $\mathbf{h}_t$ is called the **hidden state**: a vector that condenses what has been seen from step $1$ to step $t$ into one summary. $f_\theta$ is a function with parameters $\theta$. At every step the network takes the old summary and the new input and computes a new summary. The length of the sequence does not matter; it just goes on step by step.

類比 `[補充]`：隱藏狀態像一鍋一直在熬的高湯。昨天的湯底是 $\mathbf{h}_{t-1}$，今天加進去的新食材是 $\mathbf{x}_t$，熬完就是今天的湯 $\mathbf{h}_t$。這個類比的侷限是：真的高湯只會越加越多，而 $\mathbf{h}_t$ 的大小固定，加新東西時舊東西會被擠掉一部分。

Analogy `[Added]`: the hidden state is like a stock that is always simmering. Yesterday's stock is $\mathbf{h}_{t-1}$, today's new ingredient is $\mathbf{x}_t$, and after simmering you get today's stock $\mathbf{h}_t$. The limit of the analogy: a real stock only grows, while $\mathbf{h}_t$ has a fixed size, so adding something new pushes part of the old out.

![左邊是摺疊的 RNN：x_t 經 U 進入 h_t，h_t 經 V 得到 y_t，h_t 有一個標 W 的自迴圈。中間一個箭頭標 unroll。右邊是展開三步：h_0 經 W 到 h_1、再經 W 到 h_2、再經 W 到 h_3；每個 h_t 下方有 x_t 經 U 進來，上方經 V 得到 y_t。每條同類箭頭都標同一個字母（Folded RNN with a W self-loop on the left; on the right the same RNN unrolled over three steps, every arrow of the same kind carrying the same letter U, W or V）](../assets/figures/lec08-kp01-unrolled.png)

**圖說 `lec08-kp01-unrolled`**〔功能：一目了然〕：先看左邊的摺疊圖：只有一個隱藏狀態方塊，藍色自迴圈 $\mathbf{W}$ 表示「這一步的狀態會送回來，給下一步用」。再看右邊的展開圖：把迴圈沿時間攤開，$\mathbf{h}_0\to\mathbf{h}_1\to\mathbf{h}_2\to\mathbf{h}_3$ 一路傳下去。最後數一數字母：三條藍色水平箭頭都是 $\mathbf{W}$，三條往上都是 $\mathbf{V}$，三條往下進來都是 $\mathbf{U}$，沒有 $\mathbf{W}_1,\mathbf{W}_2$ 這種分別。

**Caption `lec08-kp01-unrolled`** [Function: seeing it at a glance]: First look at the folded graph on the left: there is only one hidden-state box, and the blue self-loop $\mathbf{W}$ means "the state of this step is sent back for the next step". Then look at the unrolled graph on the right: the loop is laid out along time, and $\mathbf{h}_0\to\mathbf{h}_1\to\mathbf{h}_2\to\mathbf{h}_3$ passes along. Finally count the letters: the three blue horizontal arrows are all $\mathbf{W}$, the three upward ones all $\mathbf{V}$ and the three incoming ones all $\mathbf{U}$; there is no $\mathbf{W}_1,\mathbf{W}_2$ distinction.

把迴圈沿時間攤開成一排的做法叫**時間展開**（unrolling in time）。展開之後，RNN 看起來就像一個很深的前饋網路，每一「層」是一個時間步。KP02 訓練 RNN 時，會用到這個展開圖。

Laying the loop out in a row along time is called **unrolling in time**. After unrolling, the RNN looks like a very deep feedforward network in which every "layer" is one time step. KP02 uses this unrolled graph to train the RNN.

### 3. 參數共享：每一步用同一組權重（Parameter sharing: the same weights at every step）

剛才的 $f_\theta$ 沒有下標 $t$，這是刻意的。如果每一步各用一套參數 $\theta_t$，會有兩個問題：參數個數跟著序列長度變多；而且遇到比訓練時更長的序列，後面那幾步根本沒有參數可用。

The $f_\theta$ above has no subscript $t$, and that is deliberate. If every step used its own parameters $\theta_t$, there would be two problems: the number of parameters would grow with the sequence length; and for a sequence longer than any seen in training, the last steps would have no parameters at all.

RNN 的做法是**參數共享**（parameter sharing）：所有時間步都用同一組 $\mathbf{U}$、$\mathbf{W}$、$\mathbf{V}$ 與偏置。這和第6週卷積「同一個核在影像各處滑動」是同一個想法：卷積在空間上共用權重，RNN 在時間上共用權重 `[補充]`。

The RNN's answer is **parameter sharing**: all time steps use the same $\mathbf{U}$, $\mathbf{W}$, $\mathbf{V}$ and biases. This is the same idea as the Week 6 convolution, where "one kernel slides over the whole image": convolution shares weights across space, the RNN shares weights across time `[Added]`.

用數字看差別 `[補充]`：設 $d=3$、$p=4$、$q=2$。$\mathbf{U}$ 有 $4\times3=12$ 個數，$\mathbf{W}$ 有 $4\times4=16$ 個，$\mathbf{V}$ 有 $2\times4=8$ 個，偏置 $\mathbf{b}_i$ 有 $4$ 個、$\mathbf{b}_y$ 有 $2$ 個，共 $42$ 個參數。

Look at the difference in numbers `[Added]`: let $d=3$, $p=4$, $q=2$. $\mathbf{U}$ has $4\times3=12$ numbers, $\mathbf{W}$ has $4\times4=16$, $\mathbf{V}$ has $2\times4=8$, the bias $\mathbf{b}_i$ has $4$ and $\mathbf{b}_y$ has $2$, which is $42$ parameters in all.

| 序列長度 $T$ | 共享權重的參數個數 | 每步各一套的參數個數 |
|---|---|---|
| $10$ | $42$ | $42\times10=420$ |
| $100$ | $42$ | $42\times100=4200$ |

| Sequence length $T$ | Parameters with shared weights | Parameters with one set per step |
|---|---|---|
| $10$ | $42$ | $42\times10=420$ |
| $100$ | $42$ | $42\times100=4200$ |

共享的版本不管 $T$ 是多少，都是 $42$ 個。

The shared version has $42$ no matter what $T$ is.

常見錯誤：看到展開圖有三個 $\mathbf{h}$ 方塊，就以為那是三個不同的網路。其實全程只有一個網路，同一組權重在時間軸上重複使用了三次。

Common error: seeing three $\mathbf{h}$ boxes in the unrolled graph and thinking they are three different networks. There is only one network throughout, and the same weights are reused three times along the time axis.

### 4. RNN 的三條前向公式（The three forward equations of an RNN）

把 $f_\theta$ 寫清楚，標準 RNN 每一步做三件事：

Written out, a standard RNN does three things at every step:

$$
\begin{aligned}
\mathbf{i}_t&=\mathbf{W}\mathbf{h}_{t-1}+\mathbf{U}\mathbf{x}_t+\mathbf{b}_i,\\
\mathbf{h}_t&=\tanh(\mathbf{i}_t),\\
\mathbf{y}_t&=\mathbf{V}\mathbf{h}_t+\mathbf{b}_y.
\end{aligned}
$$

第一條把「舊記憶」$\mathbf{W}\mathbf{h}_{t-1}$ 和「新輸入」$\mathbf{U}\mathbf{x}_t$ 加起來，得到淨輸入 $\mathbf{i}_t$。第二條用 $\tanh$ 把每個分量壓到 $(-1,1)$ 之間，得到新狀態；這樣狀態不會一步一步越滾越大。第三條把狀態轉成這一步的輸出。

The first equation adds the "old memory" $\mathbf{W}\mathbf{h}_{t-1}$ and the "new input" $\mathbf{U}\mathbf{x}_t$ to get the net input $\mathbf{i}_t$. The second squeezes every component into $(-1,1)$ with $\tanh$ to get the new state; this keeps the state from growing step after step. The third turns the state into the output of this step.

三個矩陣各管一條路：$\mathbf{U}$ 管「輸入進到狀態」，$\mathbf{W}$ 管「狀態傳到下一步」，$\mathbf{V}$ 管「狀態變成輸出」。它們的大小由維度決定，不必相同：$\mathbf{U}$ 是 $p\times d$，$\mathbf{W}$ 是 $p\times p$，$\mathbf{V}$ 是 $q\times p$。用矩陣乘法的規則檢查：$\mathbf{U}\mathbf{x}_t$ 是 $(p\times d)(d\times1)=p\times1$，和 $\mathbf{W}\mathbf{h}_{t-1}$ 的 $(p\times p)(p\times1)=p\times1$ 一樣大，才能相加。

Each of the three matrices owns one path: $\mathbf{U}$ owns "input into the state", $\mathbf{W}$ owns "state to the next step", $\mathbf{V}$ owns "state into the output". Their sizes are set by the dimensions and need not be equal: $\mathbf{U}$ is $p\times d$, $\mathbf{W}$ is $p\times p$, $\mathbf{V}$ is $q\times p$. Check with the rule of matrix multiplication: $\mathbf{U}\mathbf{x}_t$ is $(p\times d)(d\times1)=p\times1$, the same size as $\mathbf{W}\mathbf{h}_{t-1}$, which is $(p\times p)(p\times1)=p\times1$, so they can be added.

如果要做分類，輸出 $\mathbf{y}_t$ 再經過第2週的 softmax，變成機率：第 $k$ 類的機率是

For classification the output $\mathbf{y}_t$ passes through the softmax of Week 2 and becomes probabilities: the probability of class $k$ is

$$
\hat{y}_{t,k}=\frac{\exp(y_{t,k})}{\sum_{j=1}^{q}\exp(y_{t,j})}.
$$

常見錯誤：以為 $\mathbf{U}$、$\mathbf{W}$、$\mathbf{V}$ 一定是同樣大小的方陣。只有 $\mathbf{W}$ 一定是方陣，因為它把 $p$ 維的狀態變回 $p$ 維的狀態。

Common error: thinking that $\mathbf{U}$, $\mathbf{W}$ and $\mathbf{V}$ must be square matrices of the same size. Only $\mathbf{W}$ must be square, because it maps a $p$-dimensional state back to a $p$-dimensional state.

### 5. 手算兩步：記憶真的傳下去了（Two steps by hand: the memory really passes on）

用最小的例子把公式走一遍。取純量 $d=p=q=1$，參數 $W=0.8$、$U=1.0$、$V=1.2$，偏置 $b_i=b_y=0$，初始狀態 $h_0=0$。輸入序列是 $x_1=1.0$、$x_2=-0.5$。

Run the equations on the smallest example. Take scalars $d=p=q=1$, parameters $W=0.8$, $U=1.0$, $V=1.2$, biases $b_i=b_y=0$ and initial state $h_0=0$. The input sequence is $x_1=1.0$, $x_2=-0.5$.

第 $1$ 步：

Step $1$:

$$
\begin{aligned}
i_1&=0.8\times0+1.0\times1.0=1.0,\\
h_1&=\tanh(1.0)=0.7616,\\
y_1&=1.2\times0.7616=0.9139.
\end{aligned}
$$

第 $2$ 步：上一步的 $h_1=0.7616$ 被 $W$ 帶進來。

Step $2$: the previous $h_1=0.7616$ is carried in by $W$.

$$
\begin{aligned}
i_2&=0.8\times0.7616+1.0\times(-0.5)=0.6093-0.5=0.1093,\\
h_2&=\tanh(0.1093)=0.1088,\\
y_2&=1.2\times0.1088=0.1306.
\end{aligned}
$$

注意第 $2$ 步的輸入是負的，$x_2=-0.5$，但狀態 $h_2$ 仍是正的。原因是舊記憶 $Wh_1=+0.6093$ 比新輸入 $Ux_2=-0.5$ 大，抵銷之後還剩 $+0.1093$。

Notice that the input of step $2$ is negative, $x_2=-0.5$, yet the state $h_2$ is still positive. The reason is that the old memory $Wh_1=+0.6093$ is larger than the new input $Ux_2=-0.5$, and $+0.1093$ is left after they cancel.

對照 `[補充]`：把 $W$ 改成 $0$，切斷「狀態傳到下一步」這條路。第 $1$ 步不變，但第 $2$ 步變成 $i_2=0+1.0\times(-0.5)=-0.5$，$h_2=\tanh(-0.5)=-0.4621$，連正負號都翻了。$W=0$ 時，$h_t=\tanh(Ux_t)$ 只看當下的輸入，RNN 退化成沒有記憶的前饋網路。

Compare `[Added]`: change $W$ to $0$, cutting the path "state to the next step". Step $1$ does not change, but step $2$ becomes $i_2=0+1.0\times(-0.5)=-0.5$ and $h_2=\tanh(-0.5)=-0.4621$; even the sign flips. With $W=0$, $h_t=\tanh(Ux_t)$ looks only at the current input, and the RNN degenerates into a feedforward network without memory.

![兩組水平長條圖。上面一組是 W 等於 0.8：藍色舊記憶 W h_1 等於 +0.6093，橘色新輸入 U x_2 等於 -0.5000，綠色加總 i_2 等於 +0.1093，得到 h_2 等於 +0.1088。下面一組是 W 等於 0：藍色是 0，橘色 -0.5000，綠色加總 -0.5000，得到 h_2 等於 -0.4621（Two groups of horizontal bars: with W = 0.8 the old memory +0.6093 outweighs the new input −0.5 and the sum is +0.1093; with W = 0 the sum is −0.5 and h_2 = −0.4621）](../assets/figures/lec08-kp01-memory.png)

**圖說 `lec08-kp01-memory`**〔功能：精準度〕：先看上面一組（$W=0.8$）：藍色的舊記憶比橘色的新輸入長，所以綠色的加總落在 $0$ 的右邊。再看下面一組（$W=0$）：藍色長條消失了，綠色只剩橘色那一段，落到 $0$ 的左邊。兩組的橘色一樣長，差別完全來自藍色，也就是 $W$ 帶進來的記憶。

**Caption `lec08-kp01-memory`** [Function: precision]: First look at the upper group ($W=0.8$): the blue old memory is longer than the orange new input, so the green sum lies to the right of $0$. Then look at the lower group ($W=0$): the blue bar has vanished, the green is just the orange part and lies to the left of $0$. The orange bars of the two groups are equally long; the whole difference comes from the blue, that is, from the memory carried in by $W$.

### 本單元重點整理（KP01 summary）

**一句話重點**：RNN 用隱藏狀態 $\mathbf{h}_t$ 把過去濃縮成一份摘要，每一步以 $\mathbf{h}_t=\tanh(\mathbf{W}\mathbf{h}_{t-1}+\mathbf{U}\mathbf{x}_t+\mathbf{b}_i)$ 更新，並在所有時間步共用同一組 $\mathbf{U}$、$\mathbf{W}$、$\mathbf{V}$，所以能處理任意長度的序列，參數個數也不變。

**One-sentence summary**: an RNN condenses the past into one summary, the hidden state $\mathbf{h}_t$, updates it at every step by $\mathbf{h}_t=\tanh(\mathbf{W}\mathbf{h}_{t-1}+\mathbf{U}\mathbf{x}_t+\mathbf{b}_i)$, and shares the same $\mathbf{U}$, $\mathbf{W}$, $\mathbf{V}$ across all time steps, so it handles sequences of any length with a fixed number of parameters.

**白話說明**：前饋網路像一次只看一張照片，看完就忘；RNN 像邊看邊做筆記，每看一個新東西，就把它和舊筆記合在一起，改寫成新筆記。筆記本只有一本，寫筆記的規則也只有一套。

**Plain explanation**: a feedforward network is like looking at one photo at a time and forgetting it; an RNN is like taking notes while reading, merging each new item with the old notes into new notes. There is only one notebook and only one set of note-taking rules.

**正式定義**：$\mathbf{i}_t=\mathbf{W}\mathbf{h}_{t-1}+\mathbf{U}\mathbf{x}_t+\mathbf{b}_i$，$\mathbf{h}_t=\tanh(\mathbf{i}_t)$，$\mathbf{y}_t=\mathbf{V}\mathbf{h}_t+\mathbf{b}_y$；$\mathbf{U}\in\mathbb{R}^{p\times d}$，$\mathbf{W}\in\mathbb{R}^{p\times p}$，$\mathbf{V}\in\mathbb{R}^{q\times p}$，所有 $t$ 共用。

**Formal definition**: $\mathbf{i}_t=\mathbf{W}\mathbf{h}_{t-1}+\mathbf{U}\mathbf{x}_t+\mathbf{b}_i$, $\mathbf{h}_t=\tanh(\mathbf{i}_t)$, $\mathbf{y}_t=\mathbf{V}\mathbf{h}_t+\mathbf{b}_y$; $\mathbf{U}\in\mathbb{R}^{p\times d}$, $\mathbf{W}\in\mathbb{R}^{p\times p}$, $\mathbf{V}\in\mathbb{R}^{q\times p}$, shared by all $t$.

**例子**：$W=0.8$、$U=1.0$、$V=1.2$、$h_0=0$，輸入 $1.0$、$-0.5$：$h_1=0.7616$、$y_1=0.9139$，$i_2=0.1093$、$h_2=0.1088$、$y_2=0.1306$。$W=0$ 時 $h_2=-0.4621$。$d=3,p=4,q=2$ 時參數 $42$ 個，與 $T$ 無關。

**Example**: $W=0.8$, $U=1.0$, $V=1.2$, $h_0=0$ and inputs $1.0$, $-0.5$: $h_1=0.7616$, $y_1=0.9139$, $i_2=0.1093$, $h_2=0.1088$, $y_2=0.1306$. With $W=0$, $h_2=-0.4621$. With $d=3,p=4,q=2$ there are $42$ parameters, independent of $T$.

**常見迷思**：

- 「展開圖的每一步是不同的網路。」錯，全程只有一組權重，重複使用。
- 「$\mathbf{U}$、$\mathbf{W}$、$\mathbf{V}$ 一定一樣大。」錯，只有 $\mathbf{W}$ 一定是 $p\times p$ 的方陣。
- 「序列越長，RNN 的參數越多。」錯，參數個數只看 $d$、$p$、$q$。

**Common misconceptions**:

- "Every step of the unrolled graph is a different network." Wrong: there is only one set of weights, reused.
- "$\mathbf{U}$, $\mathbf{W}$ and $\mathbf{V}$ must have the same size." Wrong: only $\mathbf{W}$ must be a $p\times p$ square matrix.
- "The longer the sequence, the more parameters an RNN has." Wrong: the number of parameters depends only on $d$, $p$ and $q$.

**圖**：本單元有兩張圖。

- `lec08-kp01-unrolled`：摺疊與展開的 RNN，每一步共用 $\mathbf{U}$、$\mathbf{W}$、$\mathbf{V}$。
- `lec08-kp01-memory`：第 $2$ 步的淨輸入由舊記憶與新輸入組成，$W=0$ 時記憶消失。

**Figures** — this unit has two figures.

- `lec08-kp01-unrolled`: the folded and the unrolled RNN, sharing $\mathbf{U}$, $\mathbf{W}$, $\mathbf{V}$ at every step.
- `lec08-kp01-memory`: the net input of step $2$ is made of old memory and new input, and the memory vanishes when $W=0$.

### 小練習（Practice）

1. 設 $d=5$、$p=8$、$q=3$。寫出 $\mathbf{U}$、$\mathbf{W}$、$\mathbf{V}$ 的大小，並算出含偏置的參數總數。提示：偏置 $\mathbf{b}_i$ 有 $p$ 個、$\mathbf{b}_y$ 有 $q$ 個。
2. 沿用第 5 節的參數（$W=0.8$、$U=1.0$、$V=1.2$、$h_0=0$），輸入改成 $x_1=0.5$、$x_2=0.5$。算出 $h_1$、$h_2$。提示：先算 $i_1$，再用 $h_1$ 算 $i_2$。
3. 自動駕駛的連續影像畫面算是序列資料嗎？如果要用 RNN 處理，每一步的輸入 $\mathbf{x}_t$ 可以是什麼？提示：想想第6、7週的 CNN 能把一張圖變成什麼。

1. Let $d=5$, $p=8$, $q=3$. Write the sizes of $\mathbf{U}$, $\mathbf{W}$, $\mathbf{V}$ and compute the total number of parameters including biases. Hint: $\mathbf{b}_i$ has $p$ entries and $\mathbf{b}_y$ has $q$.
2. Keep the parameters of Section 5 ($W=0.8$, $U=1.0$, $V=1.2$, $h_0=0$) and change the inputs to $x_1=0.5$, $x_2=0.5$. Compute $h_1$ and $h_2$. Hint: compute $i_1$ first, then use $h_1$ to compute $i_2$.
3. Are the consecutive frames of a self-driving car's camera sequence data? If an RNN processes them, what could the input $\mathbf{x}_t$ of each step be? Hint: think about what the CNN of Weeks 6 and 7 can turn an image into.

**第 1 題解答** `[補充]`：$\mathbf{U}$ 是 $8\times5$（$40$ 個），$\mathbf{W}$ 是 $8\times8$（$64$ 個），$\mathbf{V}$ 是 $3\times8$（$24$ 個），偏置 $8+3=11$ 個，合計 $139$ 個。

**Solution to question 1** `[Added]`: $\mathbf{U}$ is $8\times5$ ($40$), $\mathbf{W}$ is $8\times8$ ($64$), $\mathbf{V}$ is $3\times8$ ($24$), and the biases are $8+3=11$, $139$ in all.

**第 2 題解答** `[補充]`：$i_1=0.5$，$h_1=\tanh(0.5)=0.4621$；$i_2=0.8\times0.4621+0.5=0.8697$，$h_2=\tanh(0.8697)=0.7012$。兩次輸入一樣，狀態卻不一樣，因為第二次多了舊記憶。

**Solution to question 2** `[Added]`: $i_1=0.5$, $h_1=\tanh(0.5)=0.4621$; $i_2=0.8\times0.4621+0.5=0.8697$, $h_2=\tanh(0.8697)=0.7012$. The two inputs are equal but the states differ, because the second time the old memory is added.

**第 3 題解答** `[補充]`：是，前後畫面有時間上的連續性。常見的做法是先用 CNN 把每一張畫面變成一個特徵向量，當作 $\mathbf{x}_t$，再由 RNN 把畫面之間的變化串起來。

**Solution to question 3** `[Added]`: yes, consecutive frames are continuous in time. A common design first turns every frame into a feature vector with a CNN, uses it as $\mathbf{x}_t$, and lets the RNN connect the changes between frames.

### 小測驗（Quiz，來源原題）

來源的兩題自我檢測保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The two self-check questions of the source are kept with all options; the source page does not mark the answers directly, and the explanations below follow the source text.

1. 為什麼傳統前饋神經網路（FNN）不適合直接處理變長文字序列？
   - (A) FNN 的權重無法更新
   - (B) FNN 假設輸入獨立同分布，缺乏儲存歷史上下文的記憶機制與變長處理能力
   - (C) FNN 無法使用反向傳播演算法
   - (D) FNN 只支援複數運算
2. 在標準 RNN 前向傳播公式中，權重矩陣 $\mathbf{W}$ 的主要作用是什麼？
   - (A) 將輸入映射到輸出空間
   - (B) 在時間步之間傳遞並轉移隱藏狀態 $\mathbf{h}_{t-1}$ 到 $\mathbf{h}_t$
   - (C) 對輸出向量進行 softmax 歸一化
   - (D) 計算即時損失

1. Why is a traditional feedforward network (FNN) unsuitable for processing variable-length text sequences directly?
   - (A) The weights of an FNN cannot be updated
   - (B) An FNN assumes independent, identically distributed inputs and lacks a memory for past context and the ability to handle variable lengths
   - (C) An FNN cannot use backpropagation
   - (D) An FNN only supports complex arithmetic
2. In the forward equations of a standard RNN, what is the main role of the weight matrix $\mathbf{W}$?
   - (A) Mapping the input to the output space
   - (B) Passing the hidden state between time steps, from $\mathbf{h}_{t-1}$ to $\mathbf{h}_t$
   - (C) Normalizing the output vector with softmax
   - (D) Computing the instantaneous loss

**解析** `[補充]`：第 1 題選 (B)：第 1 節的兩個困難正是「長度不固定」與「沒有記憶」；(A)(C) 錯，因為 FNN 的權重用反向傳播更新得很好；(D) 與題目無關。第 2 題選 (B)：第 4 節說 $\mathbf{W}$ 管「狀態傳到下一步」；把輸入送進狀態的是 $\mathbf{U}$，送出輸出的是 $\mathbf{V}$。

**Explanations** `[Added]`: question 1 is B: the two difficulties of Section 1 are exactly "lengths that vary" and "no memory"; A and C are wrong, since an FNN's weights are updated perfectly well by backpropagation; D is irrelevant. Question 2 is B: Section 4 says that $\mathbf{W}$ owns "state to the next step"; $\mathbf{U}$ brings the input into the state and $\mathbf{V}$ produces the output.

### 留給下一單元的問題（The question for the next unit）

同一個 $\mathbf{W}$ 在每一步都用了一次。如果網路在第 $3$ 步預測錯了，誤差要怎麼一路倒回去，告訴這個共用的 $\mathbf{W}$ 該怎麼改？下一單元（KP02）的隨時間反向傳播（BPTT）回答這個問題。

The same $\mathbf{W}$ is used once at every step. If the network makes a wrong prediction at step $3$, how does the error travel back and tell this shared $\mathbf{W}$ how to change? Backpropagation through time (BPTT) in the next unit (KP02) answers this question.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第8章「循環神經網絡與長短期記憶網絡」§1、§2.1–2.2 與其教授祕笈（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch08-rnn-lstm/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §1 and §2.1–2.2 of Chapter 8, "Recurrent neural networks and long short-term memory networks", of CCC-LNS IDL_115 and its professor's notes (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch08-rnn-lstm/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源寫 $h_2=\tanh(0.1093)\approx0.1089$。程式算得 $\tanh(0.10928)=0.10884$，四捨五入到小數第四位是 $0.1088$，本講義用 $0.1088$；$y_2$ 也隨之是 $0.1306$。
- 來源用粗斜體 `\boldsymbol` 表示向量；本課程網站不支援，改用 `\mathbf`，意義相同。
- 來源的 softmax 式子分母寫成 $\sum_j\exp(y_{t,j})$、分子只寫第 $1$ 類；本講義寫成任一類 $k$ 的一般形式。
- 本講義補上：參數個數的數字例（$d=3,p=4,q=2$ 共 $42$ 個）、$W=0$ 的數值對照（來源只說「失去記憶」）、與第6週卷積的比較、三題小練習與解答。
- 取捨：來源的兩題自我檢測保留原題；來源的互動滑桿展示無法放進講義，改用 `lec08-kp01-memory` 圖呈現同一個算例。

- The source writes $h_2=\tanh(0.1093)\approx0.1089$. The program gives $\tanh(0.10928)=0.10884$, which rounds to $0.1088$ at four decimals, so this note uses $0.1088$; $y_2$ becomes $0.1306$ accordingly.
- The source writes vectors with bold italic `\boldsymbol`; the course website does not support it, so this note uses `\mathbf` with the same meaning.
- The source's softmax has the denominator $\sum_j\exp(y_{t,j})$ but writes only class $1$ in the numerator; this note writes the general form for any class $k$.
- This note adds: the parameter-count example ($42$ for $d=3,p=4,q=2$), the numerical comparison for $W=0$ (the source only says "loses memory"), the comparison with the Week 6 convolution, and three practice questions with solutions.
- Choices: the two self-check questions of the source are kept as asked; the source's interactive sliders cannot go into a note, so the `lec08-kp01-memory` figure shows the same example.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 序列資料 | sequence data |
| 前饋網路 | feedforward network |
| 長距離依賴 | long-range dependency |
| 動態系統 | dynamical system |
| 隱藏狀態 | hidden state |
| 時間展開 | unrolling in time |
| 參數共享 | parameter sharing |
| 循環神經網路 | recurrent neural network (RNN) |
| 淨輸入 | net input |
| 隨時間反向傳播 | backpropagation through time (BPTT) |
| 卷積神經網路 | convolutional neural network (CNN) |
| 長短期記憶網路（KP06） | long short-term memory (LSTM) |
| 門控循環單元（KP07） | gated recurrent unit (GRU) |
| 前饋神經網路（來源的縮寫） | feedforward neural network (FNN) |
| 本課程的網頁教材 | CCC-LNS |
