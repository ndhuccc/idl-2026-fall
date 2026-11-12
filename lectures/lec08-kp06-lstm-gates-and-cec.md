# 第8週 KP06：LSTM 用三個門與加法傳送帶，讓記憶與梯度走得更遠（LSTM: three gates and an additive conveyor belt carry memory and gradients further）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第8週的第六個知識點（KP06），接在 KP05 之後。KP05 的滲漏單元，保留比例是固定的，沒辦法看情況決定「該記住還是該忘掉」。網路能不能自己學會，每一步動態地調整這個比例？

This is the sixth knowledge point (KP06) of Week 8, following KP05. The keep ratio of the KP05 leaky unit is fixed, so it cannot decide case by case whether "now is the time to remember or to forget". Can the network learn to adjust this ratio by itself, step by step?

先備知識有四項：KP01 的 RNN 前向公式、KP04 的梯度消失、KP05 的滲漏單元，以及第2週的 sigmoid 與 $\tanh$。第一次出現的詞與符號都會定義。所有數字都由 `assets/figures/src/lec08-kp06-figures.py` 算出。你可以自己重算。

The prerequisites are the RNN forward equations of KP01, the vanishing gradient of KP04, the leaky unit of KP05, and the sigmoid and $\tanh$ functions of Week 2. Every term and symbol is defined when it first appears. Every number is computed by `assets/figures/src/lec08-kp06-figures.py`, so you can recompute it yourself.

## KP06 LSTM 用三個門與加法傳送帶，讓記憶與梯度走得更遠（LSTM: three gates and an additive conveyor belt carry memory and gradients further）

### 學習目標（Learning objectives）

讀完之後，你能寫出遺忘門、輸入門、輸出門的算式，說出它們為何都用 sigmoid，並說明窺視孔連線讓門多看到什麼。你能寫出細胞狀態的加法更新式 $\mathbf{c}_t=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t$，以及隱藏狀態 $\mathbf{h}_t=\mathbf{o}_t\odot\tanh(\mathbf{c}_t)$。你能用純量例子手算一整步 LSTM。你也能說明加法傳送帶為何讓梯度不易消失，以及何時仍會衰減。

After reading, you can write the equations of the forget gate, the input gate and the output gate, say why all of them use the sigmoid, and explain what a peephole connection lets a gate see. You can write the additive cell-state update $\mathbf{c}_t=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t$ and the hidden state $\mathbf{h}_t=\mathbf{o}_t\odot\tanh(\mathbf{c}_t)$. You can compute one full LSTM step of a scalar example by hand. You can also explain why the additive conveyor belt keeps the gradient from vanishing easily, and when it still decays.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{x}_t\in\mathbb{R}^{d}$ | 第 $t$ 步的輸入 | input at step $t$ |
| $\mathbf{h}_t\in\mathbb{R}^{p}$ | 隱藏狀態，對外輸出的短期狀態 | hidden state, the short-term state shown outside |
| $\mathbf{c}_t\in\mathbb{R}^{p}$ | 細胞狀態，內部的長期記憶（傳送帶） | cell state, the internal long-term memory (conveyor belt) |
| $\tilde{\mathbf{c}}_t\in(-1,1)^{p}$ | 候選記憶，這一步想寫入的新內容 | candidate memory, the new content proposed at this step |
| $\mathbf{f}_t\in(0,1)^{p}$ | 遺忘門：舊記憶保留多少 | forget gate: how much old memory is kept |
| $\mathbf{i}_t\in(0,1)^{p}$ | 輸入門：新內容寫入多少（**不是** KP01 的淨輸入） | input gate: how much new content is written (**not** the net input of KP01) |
| $\mathbf{o}_t\in(0,1)^{p}$ | 輸出門：記憶對外公開多少 | output gate: how much memory is shown outside |
| $\mathbf{W}_f,\mathbf{W}_i,\mathbf{W}_o,\mathbf{W}_c\in\mathbb{R}^{p\times p}$ | 各門與候選記憶看 $\mathbf{h}_{t-1}$ 的權重 | weights from $\mathbf{h}_{t-1}$ to each gate and to the candidate |
| $\mathbf{U}_f,\mathbf{U}_i,\mathbf{U}_o,\mathbf{U}_c\in\mathbb{R}^{p\times d}$ | 各門與候選記憶看 $\mathbf{x}_t$ 的權重 | weights from $\mathbf{x}_t$ to each gate and to the candidate |
| $\mathbf{p}_f,\mathbf{p}_i,\mathbf{p}_o\in\mathbb{R}^{p}$ | 窺視孔權重，讓門看細胞狀態 | peephole weights, letting a gate look at the cell state |
| $\mathbf{b}_f,\mathbf{b}_i,\mathbf{b}_o,\mathbf{b}_c\in\mathbb{R}^{p}$ | 各門與候選記憶的偏置（$\mathbf{b}_i$ 是輸入門的偏置） | biases of the gates and the candidate ($\mathbf{b}_i$ is the input gate's bias) |
| $\mathbf{V}\in\mathbb{R}^{q\times p}$，$\mathbf{b}_y$ | 隱藏狀態到輸出的權重與偏置，同 KP01 | hidden-to-output weights and bias, as in KP01 |
| $\sigma(z)$ | sigmoid 函數 $1/(1+e^{-z})$（來源寫成 $\operatorname{sig}$） | sigmoid function $1/(1+e^{-z})$ (the source writes $\operatorname{sig}$) |
| $z$ | 門的淨輸入，進入 sigmoid 之前的加總 | net input of a gate, the sum before the sigmoid |
| $\odot$ | 逐元素乘積，同位置的分量相乘 | element-wise product, multiplying entries at the same position |

### 1. 從固定的滲漏到會學的開關（From a fixed leak to learnable switches）

先回答 KP05 的問題。滲漏單元每一步做 $h_t=(1-\frac{1}{\tau})h_{t-1}+\frac{1}{\tau}\tanh(\cdots)$。舊狀態保留 $1-\frac{1}{\tau}$，新內容寫入 $\frac{1}{\tau}$。這兩個比例由 $\tau$ 決定，訓練好之後就固定不變。

First, answer the question of KP05. A leaky unit computes $h_t=(1-\frac{1}{\tau})h_{t-1}+\frac{1}{\tau}\tanh(\cdots)$ at every step. It keeps $1-\frac{1}{\tau}$ of the old state and writes $\frac{1}{\tau}$ of the new content. Both ratios are set by $\tau$ and stay fixed once training is done.

**長短期記憶網路**（long short-term memory，LSTM）把這兩個固定比例，換成「每一步重新計算的開關」。這種開關叫做**門**（gate）：它是一個每個分量都在 $0$ 到 $1$ 之間的向量。門乘在一條資訊流上，$0$ 表示全擋，$1$ 表示全放，$0.7$ 表示放行七成。

**Long short-term memory** (LSTM) replaces the two fixed ratios with "switches recomputed at every step". Such a switch is called a **gate**: a vector whose every component lies between $0$ and $1$. A gate multiplies a flow of information; $0$ blocks it completely, $1$ lets it all through, and $0.7$ lets seventy percent through.

門要能落在 $0$ 到 $1$ 之間，所以用 **sigmoid 函數** $\sigma(z)=\frac{1}{1+e^{-z}}$，它的輸出永遠在 $(0,1)$ 之內。$\tanh$ 會出現負值，ReLU 沒有上限，都不適合當「放行比例」。門的值是連續的，例如 $0.85$，所以叫**軟開關**（soft switch）；連續才能對它微分，才能用反向傳播訓練。

A gate must lie between $0$ and $1$, so it uses the **sigmoid function** $\sigma(z)=\frac{1}{1+e^{-z}}$. Its output is always inside $(0,1)$. $\tanh$ produces negative values and ReLU has no upper limit, so neither suits a "pass ratio". A gate's value is continuous, for example $0.85$, so it is called a **soft switch**; being continuous, it can be differentiated and trained by backpropagation.

門乘到向量上，用的是**逐元素乘積**（element-wise product），記作 $\odot$。做法是：兩個同樣長的向量，同一位置的分量相乘。例如 $(0.5,1)\odot(4,6)=(2,6)$ `[補充]`。所以門的每個分量，各自控制記憶的一個分量。

A gate is applied to a vector by the **element-wise product**, written $\odot$: for two vectors of the same length, multiply the entries at the same position. For example $(0.5,1)\odot(4,6)=(2,6)$ `[Added]`. So each component of a gate controls one component of the memory on its own.

常見錯誤：以為門的輸出只會是 $0$ 或 $1$，像電燈開關一樣。其實門是連續的比例。如果真的只輸出 $0$ 或 $1$，它的導數處處是 $0$，梯度就傳不回去，門也學不會。

Common error: thinking that a gate outputs only $0$ or $1$, like a light switch. In fact a gate is a continuous ratio. If it really output only $0$ or $1$, its derivative would be $0$ everywhere, no gradient could pass back, and the gate could not learn.

### 2. 三個門的算式與窺視孔（The equations of the three gates and the peepholes）

有了「門」這個工具，LSTM 用三個門管三件事。**遺忘門**（forget gate）$\mathbf{f}_t$ 決定舊記憶 $\mathbf{c}_{t-1}$ 保留多少。**輸入門**（input gate）$\mathbf{i}_t$ 決定新內容寫入多少。**輸出門**（output gate）$\mathbf{o}_t$ 決定記憶對外公開多少：

With the gate as a tool, an LSTM uses three gates for three jobs. The **forget gate** $\mathbf{f}_t$ decides how much of the old memory $\mathbf{c}_{t-1}$ is kept. The **input gate** $\mathbf{i}_t$ decides how much new content is written. The **output gate** $\mathbf{o}_t$ decides how much of the memory is shown outside:

$$
\begin{aligned}
\mathbf{f}_t&=\sigma\left(\mathbf{W}_f\mathbf{h}_{t-1}+\mathbf{U}_f\mathbf{x}_t+\mathbf{p}_f\odot\mathbf{c}_{t-1}+\mathbf{b}_f\right),\\
\mathbf{i}_t&=\sigma\left(\mathbf{W}_i\mathbf{h}_{t-1}+\mathbf{U}_i\mathbf{x}_t+\mathbf{p}_i\odot\mathbf{c}_{t-1}+\mathbf{b}_i\right),\\
\mathbf{o}_t&=\sigma\left(\mathbf{W}_o\mathbf{h}_{t-1}+\mathbf{U}_o\mathbf{x}_t+\mathbf{p}_o\odot\mathbf{c}_{t}+\mathbf{b}_o\right).
\end{aligned}
$$

三條式子長得一樣，只是各用各的權重。每個門都看三樣東西：上一步的隱藏狀態 $\mathbf{h}_{t-1}$、這一步的輸入 $\mathbf{x}_t$、細胞狀態。括號裡的加總就是門的淨輸入 $z$，再經 sigmoid 變成 $(0,1)$ 的比例。

The three equations look the same; each just has its own weights. Every gate looks at three things: the previous hidden state $\mathbf{h}_{t-1}$, the current input $\mathbf{x}_t$ and the cell state. The sum inside the brackets is the gate's net input $z$, which the sigmoid turns into a ratio in $(0,1)$.

符號提醒 `[補充]`：這裡的 $\mathbf{i}_t$ 是輸入門，$\mathbf{b}_i$ 是輸入門的偏置。它們和 KP01 的淨輸入 $\mathbf{i}_t$、隱藏層偏置 $\mathbf{b}_i$ 同名，但意思不同。本單元沿用來源與文獻的寫法，在 LSTM 裡 $\mathbf{i}_t$ 一律指輸入門。

Notation warning `[Added]`: here $\mathbf{i}_t$ is the input gate and $\mathbf{b}_i$ is its bias. They share names with the net input $\mathbf{i}_t$ and the hidden bias $\mathbf{b}_i$ of KP01, but mean different things. This unit follows the source and the literature: inside an LSTM, $\mathbf{i}_t$ always means the input gate.

用維度檢查一次：$\mathbf{W}_f\mathbf{h}_{t-1}$ 是 $(p\times p)(p\times1)=p\times1$，$\mathbf{U}_f\mathbf{x}_t$ 是 $(p\times d)(d\times1)=p\times1$。$\mathbf{p}_f\odot\mathbf{c}_{t-1}$ 也是 $p$ 維。所以三個門都是 $p$ 維，和記憶一樣長，才能逐元素相乘。

Check the dimensions once: $\mathbf{W}_f\mathbf{h}_{t-1}$ is $(p\times p)(p\times1)=p\times1$ and $\mathbf{U}_f\mathbf{x}_t$ is $(p\times d)(d\times1)=p\times1$. $\mathbf{p}_f\odot\mathbf{c}_{t-1}$ is also $p$-dimensional. So all three gates have $p$ components, as long as the memory, which is what element-wise multiplication needs.

**窺視孔連線**（peephole connection）是 $\mathbf{p}\odot\mathbf{c}$ 這一項。少了它，門只看得到 $\mathbf{h}_{t-1}$，而 $\mathbf{h}_{t-1}$ 是被輸出門打過折的記憶。窺視孔讓門直接「偷看」細胞記憶本身。權重 $\mathbf{p}$ 是向量，不是矩陣：第 $j$ 個門分量只看第 $j$ 個記憶分量 `[補充]`。

The **peephole connection** is the term $\mathbf{p}\odot\mathbf{c}$. Without it a gate sees only $\mathbf{h}_{t-1}$, which is memory already discounted by the output gate. A peephole lets the gate "peek" at the cell memory itself. The weight $\mathbf{p}$ is a vector, not a matrix: gate component $j$ only looks at memory component $j$ `[Added]`.

注意輸出門的窺視孔看的是 $\mathbf{c}_t$，不是 $\mathbf{c}_{t-1}$。遺忘門和輸入門要決定「怎麼改記憶」，只能看改之前的 $\mathbf{c}_{t-1}$。輸出門要決定「公開多少」，在記憶改好之後才算，所以看得到新的 $\mathbf{c}_t$。

Note that the output gate's peephole looks at $\mathbf{c}_t$, not $\mathbf{c}_{t-1}$. The forget and input gates decide "how to change the memory", so they can only see $\mathbf{c}_{t-1}$ before the change. The output gate decides "how much to show", is computed after the memory is updated, and so can see the new $\mathbf{c}_t$.

來源的數字例：若某個門的淨輸入 $z=2.197$，則 $e^{-2.197}=0.1111$，門值是 $\frac{1}{1+0.1111}=0.9000$，放行九成。若 $z=-2.197$，則 $e^{2.197}=8.998$，門值是 $\frac{1}{1+8.998}=0.1000$，只放行一成。

The source's numerical example: if a gate has net input $z=2.197$, then $e^{-2.197}=0.1111$ and the gate value is $\frac{1}{1+0.1111}=0.9000$, letting ninety percent through. If $z=-2.197$, then $e^{2.197}=8.998$ and the gate value is $\frac{1}{1+8.998}=0.1000$, letting only ten percent through.

![一條 S 形的 sigmoid 曲線，橫軸是門的淨輸入 z，縱軸是放行比例。上方綠色帶是放行超過 90% 的幾乎全開區，下方紅色帶是放行不到 10% 的幾乎全關區。灰點標出 z 等於 +2.197 得 0.90、z 等於 -2.197 得 0.10。三個方塊標出第 4 節例子的門：紅色 f_t 等於 0.8957、藍色 i_t 等於 0.6106、綠色 o_t 等於 0.7251（Sigmoid curve with a green nearly-open band above 0.9 and a red nearly-closed band below 0.1; grey dots at z = +2.197 and −2.197 give 0.90 and 0.10; squares mark the example gates f = 0.8957, i = 0.6106, o = 0.7251）](../assets/figures/lec08-kp06-sigmoid-gates.png)

**圖說 `lec08-kp06-sigmoid-gates`**〔功能：精準度〕：先看曲線的形狀：淨輸入 $z$ 越大，門越接近 $1$；越小，越接近 $0$；中間平滑過渡，沒有跳躍。再看兩條色帶：$z$ 超過 $2.197$ 就進入綠色的幾乎全開區，低於 $-2.197$ 就進入紅色的幾乎全關區。最後看三個方塊：第 4 節例子的遺忘門 $0.8957$ 接近全開，輸入門 $0.6106$ 和輸出門 $0.7251$ 只開一部分。

**Caption `lec08-kp06-sigmoid-gates`** [Function: precision]: First look at the shape of the curve: the larger the net input $z$, the closer the gate is to $1$; the smaller, the closer to $0$; in between it changes smoothly, with no jump. Then look at the two bands: above $z=2.197$ the gate is in the green nearly-open zone, below $z=-2.197$ in the red nearly-closed zone. Finally look at the three squares: in the example of Section 4 the forget gate $0.8957$ is nearly open, while the input gate $0.6106$ and the output gate $0.7251$ are only partly open.

類比（來源）：把 LSTM 細胞想成存知識的城堡，有三位門衛。遺忘門是後門的清潔隊，決定丟掉多少舊記憶。輸入門是前門的檢驗員，決定讓多少新資訊進來。輸出門是頂樓的發報員，決定對外公開多少。

Analogy (source): think of an LSTM cell as a castle that stores knowledge, with three guards. The forget gate is the cleaning crew at the back door, deciding how much outdated memory to throw away. The input gate is the inspector at the front door, deciding how much new information may enter. The output gate is the messenger on the top floor, deciding how much to announce.

這個類比的侷限 `[補充]`：真的門衛只會「放」或「擋」，LSTM 的門卻常常只放一部分。而且每個記憶分量，各有一位門衛。

The limit of the analogy `[Added]`: a real guard only "lets in" or "blocks", while an LSTM gate often lets only a part through, and every memory component has its own guard.

### 3. 候選記憶、細胞狀態傳送帶與輸出（Candidate memory, the cell-state conveyor belt and the output）

三個門只是比例。它們要乘在什麼東西上？答案是兩條資訊流：新內容與舊記憶。新內容叫**候選記憶**（candidate memory，來源也稱 block input），記作 $\tilde{\mathbf{c}}_t$。它的算法和 KP01 的 RNN 一樣：

The three gates are only ratios. What do they multiply? Two flows of information: new content and old memory. The new content is called the **candidate memory** (the source also calls it the block input) $\tilde{\mathbf{c}}_t$, computed just like the RNN of KP01:

$$
\tilde{\mathbf{c}}_t=\tanh\left(\mathbf{W}_c\mathbf{h}_{t-1}+\mathbf{U}_c\mathbf{x}_t+\mathbf{b}_c\right).
$$

這裡用 $\tanh$，不用 sigmoid。候選記憶是「內容」，不是「比例」：它要能是負的，才能把記憶往下調。

It uses $\tanh$, not the sigmoid. The candidate is "content", not a "ratio": it must be able to be negative, so that it can push the memory down.

接著是 LSTM 的核心式子。**細胞狀態**（cell state）$\mathbf{c}_t$ 是 LSTM 內部的長期記憶。它每一步這樣更新：

Next comes the core equation of the LSTM. The **cell state** $\mathbf{c}_t$ is the LSTM's internal long-term memory, updated at every step by

$$
\mathbf{c}_t=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t.
$$

第一項是「保留下來的舊記憶」，由遺忘門打折。第二項是「寫進來的新內容」，由輸入門打折。兩項**相加**，就是新記憶。和 KP05 比較 `[補充]`：遺忘門取代了固定的 $1-\frac{1}{\tau}$，輸入門取代了固定的 $\frac{1}{\tau}$。而且兩者每一步都依輸入重新決定。

The first term is "the old memory that is kept", discounted by the forget gate. The second is "the new content written in", discounted by the input gate. The two are **added** to give the new memory. Compare with KP05: the forget gate replaces the fixed $1-\frac{1}{\tau}$, the input gate replaces the fixed $\frac{1}{\tau}$, and both are decided anew from the input at every step `[Added]`.

這條式子常被比喻成**傳送帶**（conveyor belt）。舊記憶 $\mathbf{c}_{t-1}$ 放在帶子上往前走。經過遺忘門像經過一把刮刀，刮掉過時的部分。經過輸入門像經過一隻手臂，把新零件 $\tilde{\mathbf{c}}_t$ 裝上去。帶子上只有「乘一個比例」和「加一項」，沒有矩陣乘法，也沒有 $\tanh$ 擠壓。

This equation is often pictured as a **conveyor belt**: the old memory $\mathbf{c}_{t-1}$ sits on the belt and moves forward. Passing the forget gate is like passing a scraper that removes the outdated part. Passing the input gate is like passing an arm that mounts the new part $\tilde{\mathbf{c}}_t$. The belt has only "multiply by a ratio" and "add a term"; there is no matrix multiplication and no $\tanh$ squashing.

記憶改好之後，才決定對外公開什麼。對外的**隱藏狀態**（hidden state）與輸出是：

Only after the memory is updated does the cell decide what to show. The visible **hidden state** and the output are

$$
\begin{aligned}
\mathbf{h}_t&=\mathbf{o}_t\odot\tanh(\mathbf{c}_t),\\
\mathbf{y}_t&=\mathbf{V}\mathbf{h}_t+\mathbf{b}_y.
\end{aligned}
$$

細胞狀態一路用加法累積，數值可能變大，例如累積到 $15$。先用 $\tanh$ 壓回 $(-1,1)$，再乘輸出門，$\mathbf{h}_t$ 就不會太大。$\mathbf{y}_t$ 的算法和 KP01 完全一樣。

The cell state keeps adding up and may grow large, for example to $15$. Squashing it back into $(-1,1)$ with $\tanh$ and then multiplying by the output gate keeps $\mathbf{h}_t$ from getting too large. $\mathbf{y}_t$ is computed exactly as in KP01.

來源把 $\mathbf{h}_t$ 稱為 block output。一步 LSTM 的計算順序是：先算 $\mathbf{f}_t$、$\mathbf{i}_t$、$\tilde{\mathbf{c}}_t$，再算 $\mathbf{c}_t$。接著算要看 $\mathbf{c}_t$ 的 $\mathbf{o}_t$，最後算 $\mathbf{h}_t$ 與 $\mathbf{y}_t$。

The source calls $\mathbf{h}_t$ the block output. The order of one LSTM step is: first $\mathbf{f}_t$, $\mathbf{i}_t$ and $\tilde{\mathbf{c}}_t$, then $\mathbf{c}_t$. Next comes $\mathbf{o}_t$, which needs $\mathbf{c}_t$, and finally $\mathbf{h}_t$ and $\mathbf{y}_t$.

常見錯誤：以為 $\mathbf{c}_t$ 和 $\mathbf{h}_t$ 是同一個東西。$\mathbf{c}_t$ 是內部的長期記憶，不經過 $\tanh$ 限制，可以超出 $(-1,1)$。$\mathbf{h}_t$ 是對外公開的短期狀態，一定在 $(-1,1)$ 之內。LSTM 每一步要往下傳兩個狀態：$\mathbf{c}_t$ 與 $\mathbf{h}_t$。

Common error: thinking that $\mathbf{c}_t$ and $\mathbf{h}_t$ are the same thing. $\mathbf{c}_t$ is the internal long-term memory; it is not limited by $\tanh$ and can leave $(-1,1)$. $\mathbf{h}_t$ is the short-term state shown outside and always lies inside $(-1,1)$. An LSTM passes two states to the next step: $\mathbf{c}_t$ and $\mathbf{h}_t$.

### 4. 手算一整步 LSTM（One full LSTM step by hand）

把第 2、3 節的公式放在一起，用最小的例子走一遍 `[補充]`。取純量 $d=p=q=1$。上一步留下 $h_{t-1}=0.5$、$c_{t-1}=2.0$，這一步輸入 $x_t=1.0$。參數如下表。

Put the equations of Sections 2 and 3 together and run them on the smallest example `[Added]`. Take scalars $d=p=q=1$. The previous step left $h_{t-1}=0.5$ and $c_{t-1}=2.0$, and the input of this step is $x_t=1.0$. The parameters are in the table.

| 部件 | 看 $h_{t-1}$ 的權重 | 看 $x_t$ 的權重 | 窺視孔 | 偏置 |
|---|---|---|---|---|
| 遺忘門 $f$ | $W_f=0.5$ | $U_f=1.0$ | $p_f=0.2$ | $b_f=0.5$ |
| 輸入門 $i$ | $W_i=0.5$ | $U_i=0.5$ | $p_i=0.1$ | $b_i=-0.5$ |
| 候選記憶 $\tilde{c}$ | $W_c=0.4$ | $U_c=0.6$ | 無 | $b_c=0$ |
| 輸出門 $o$ | $W_o=0.5$ | $U_o=0.5$ | $p_o=0.1$ | $b_o=0$ |

| Part | Weight on $h_{t-1}$ | Weight on $x_t$ | Peephole | Bias |
|---|---|---|---|---|
| forget gate $f$ | $W_f=0.5$ | $U_f=1.0$ | $p_f=0.2$ | $b_f=0.5$ |
| input gate $i$ | $W_i=0.5$ | $U_i=0.5$ | $p_i=0.1$ | $b_i=-0.5$ |
| candidate $\tilde{c}$ | $W_c=0.4$ | $U_c=0.6$ | none | $b_c=0$ |
| output gate $o$ | $W_o=0.5$ | $U_o=0.5$ | $p_o=0.1$ | $b_o=0$ |

輸出層沿用 KP01 的 $V=1.2$、$b_y=0$。程式用未四捨五入的中間值相乘，所以手算時最後一位可能差 $1$。

The output layer keeps $V=1.2$ and $b_y=0$ from KP01. The program multiplies unrounded intermediate values, so a hand calculation may differ by $1$ in the last digit.

第一步，算兩個看舊記憶 $c_{t-1}$ 的門，以及候選記憶：

First, compute the two gates that look at the old memory $c_{t-1}$, and the candidate:

$$
\begin{aligned}
z_f&=0.5\times0.5+1.0\times1.0+0.2\times2.0+0.5=0.25+1.0+0.4+0.5=2.15, & f_t&=\sigma(2.15)=0.8957,\\
z_i&=0.5\times0.5+0.5\times1.0+0.1\times2.0-0.5=0.25+0.5+0.2-0.5=0.45, & i_t&=\sigma(0.45)=0.6106,\\
z_c&=0.4\times0.5+0.6\times1.0+0=0.8, & \tilde{c}_t&=\tanh(0.8)=0.6640.
\end{aligned}
$$

第二步，更新傳送帶：

Second, update the conveyor belt:

$$
c_t=f_tc_{t-1}+i_t\tilde{c}_t=0.8957\times2.0+0.6106\times0.6640=1.7913+0.4055=2.1968.
$$

第三步，輸出門看新記憶 $c_t$，再算隱藏狀態與輸出：

Third, the output gate looks at the new memory $c_t$, and then the hidden state and the output follow:

$$
\begin{aligned}
z_o&=0.5\times0.5+0.5\times1.0+0.1\times2.1968+0=0.25+0.5+0.2197=0.9697, & o_t&=\sigma(0.9697)=0.7251,\\
h_t&=o_t\tanh(c_t)=0.7251\times0.9756=0.7074, & y_t&=1.2\times0.7074=0.8488.
\end{aligned}
$$

讀這組數字：遺忘門 $0.8957$ 幾乎全開，舊記憶 $2.0$ 保留成 $1.7913$。輸入門 $0.6106$ 半開，只寫進 $0.4055$。新記憶 $c_t=2.1968$ 超出 $(-1,1)$，這對細胞狀態是允許的。對外的 $h_t=0.7074$ 則被 $\tanh$ 和輸出門壓在 $(-1,1)$ 之內。

Read these numbers: the forget gate $0.8957$ is nearly open, so the old memory $2.0$ is kept as $1.7913$. The input gate $0.6106$ is half open and writes only $0.4055$. The new memory $c_t=2.1968$ lies outside $(-1,1)$, which is allowed for the cell state. The $h_t=0.7074$ shown outside is held inside $(-1,1)$ by $\tanh$ and the output gate.

![LSTM 細胞方塊圖。上方一條黃色粗箭頭是細胞狀態傳送帶，從左邊 c_{t-1} 等於 2.0000 往右，經過一個乘號圓圈（保留 f_t c_{t-1} 等於 1.7913）和一個加號圓圈，到右邊 c_t 等於 2.1968。下方一條灰線是輸入 h_{t-1} 等於 0.5 與 x_t 等於 1.0，往上接到四個方塊：f_t（sig，0.8957）、i_t（sig，0.6106）、c̃_t（tanh，0.6640）、o_t（sig，0.7251）。f_t 往上接到傳送帶的乘號；i_t 和 c̃_t 先相乘（寫入 0.4055）再接到加號。右邊從傳送帶往下經過 tanh（0.9756），和 o_t 相乘得到藍色的 h_t 等於 0.7074。灰色虛線是窺視孔：從 c_{t-1} 接到 f_t、i_t，從 c_t 接到 o_t（LSTM cell diagram: the yellow conveyor belt carries c from 2.0000 through a multiply node and an add node to 2.1968; four gate boxes f, i, candidate and o sit on the input bus with values 0.8957, 0.6106, 0.6640, 0.7251; tanh of c_t times o_t gives h_t = 0.7074; dashed peephole lines feed c_{t-1} to f and i and c_t to o）](../assets/figures/lec08-kp06-cell.png)

**圖說 `lec08-kp06-cell`**〔功能：一目了然〕：先看上方黃色的傳送帶：$c_{t-1}=2.0$ 從左往右，只經過一個乘號和一個加號，就變成 $c_t=2.1968$。再看下方四個方塊：三個 sig 方塊是門，一個 tanh 方塊是候選記憶；遺忘門接到乘號，輸入門和候選記憶相乘後接到加號。最後看右邊：$c_t$ 經過 tanh 再乘輸出門，才變成對外的 $h_t=0.7074$；灰色虛線是窺視孔，注意輸出門看的是右邊的 $c_t$。

**Caption `lec08-kp06-cell`** [Function: seeing it at a glance]: First look at the yellow belt on top: $c_{t-1}=2.0$ moves from left to right through just one multiply node and one add node and becomes $c_t=2.1968$. Then look at the four boxes below: the three sig boxes are gates and the tanh box is the candidate; the forget gate feeds the multiply node, and the input gate times the candidate feeds the add node. Finally look at the right: $c_t$ goes through tanh and is multiplied by the output gate before it becomes the visible $h_t=0.7074$; the grey dashed lines are peepholes, and the output gate looks at $c_t$ on the right.

對照 `[補充]`：把三個窺視孔權重都設成 $0$，其他不變。門的淨輸入少了 $\mathbf{p}\odot\mathbf{c}$ 這一項，變成 $f_t=0.8520$、$i_t=0.5622$、$o_t=0.6792$，結果 $c_t=2.0772$、$h_t=0.6582$。這個例子裡舊記憶是正的，窺視孔讓三個門都開得更大一點。

Compare `[Added]`: set all three peephole weights to $0$ and keep everything else. The net inputs of the gates lose the term $\mathbf{p}\odot\mathbf{c}$, giving $f_t=0.8520$, $i_t=0.5622$ and $o_t=0.6792$, and then $c_t=2.0772$ and $h_t=0.6582$. In this example the old memory is positive, so the peepholes open all three gates a little wider.

### 5. 為什麼加法讓梯度活著：常數誤差流（Why addition keeps the gradient alive: the constant error carousel）

前向算完了，回到本週的大問題：梯度能不能從很遠的未來傳回來？KP04 說過，標準 RNN 每往回一步，梯度要乘一次 $\frac{\partial h_t}{\partial h_{t-1}}$。純量時這是 $W\tanh'(\cdot)$，因為 $\tanh'\le1$，它的大小不超過 $|W|$。所以往回 $k$ 步，最多剩 $|W|^k$。

The forward pass is done; return to the big question of this week: can the gradient travel back from the far future? KP04 showed that in a standard RNN the gradient is multiplied by $\frac{\partial h_t}{\partial h_{t-1}}$ at every step back. For scalars this is $W\tanh'(\cdot)$. Since $\tanh'\le1$, its size is at most $|W|$. After $k$ steps back, at most $|W|^k$ remains.

LSTM 的梯度可以走傳送帶。對 $\mathbf{c}_t=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t$ 只看「直接沿傳送帶」這條路 `[補充]`，也就是先把各個門當成常數，就得到：

In an LSTM the gradient can take the conveyor belt. For $\mathbf{c}_t=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t$, look only at the path "straight along the belt" `[Added]`, that is, treat the gates as constants for the moment. Then

$$
\frac{\partial c_t}{\partial c_{t-1}}=f_t,
\qquad
\frac{\partial c_t}{\partial c_{t-k}}=f_tf_{t-1}\cdots f_{t-k+1}.
$$

第二項 $\mathbf{i}_t\odot\tilde{\mathbf{c}}_t$ 是加上去的，對 $\mathbf{c}_{t-1}$ 微分時不會把梯度乘小。傳送帶上沒有 $W$，也沒有 $\tanh'$，每一步只乘遺忘門。向量時，把 $f_t$ 換成每個分量各自的遺忘門。

The second term $\mathbf{i}_t\odot\tilde{\mathbf{c}}_t$ is added on, so differentiating with respect to $\mathbf{c}_{t-1}$ does not shrink the gradient. The belt has no $W$ and no $\tanh'$; at every step it multiplies only by the forget gate. For vectors, replace $f_t$ by the forget gate of each component.

當 $f_t=1$ 時，導數正好是 $1$，梯度一步也不衰減。這就是**常數誤差流**（constant error carousel，CEC）。

When $f_t=1$ the derivative is exactly $1$, and the gradient does not decay at all. This is the **constant error carousel** (CEC).

來源的特例：$f_t=1.0$、$i_t=0$ 時，$c_t=1.0\times c_{t-1}+0=c_{t-1}$。記憶原封不動地保存，往回的導數也嚴格等於 $1$。

The source's special case: with $f_t=1.0$ and $i_t=0$, $c_t=1.0\times c_{t-1}+0=c_{t-1}$. The memory is stored unchanged, and the derivative going back is exactly $1$.

![半對數座標圖。橫軸是往回走的步數 k，從 0 到 50；縱軸是梯度乘數，對數刻度。三條實線是傳送帶：f_t 等於 1 時一直是 1.0000，f_t 等於 0.99 時 50 步後剩 0.6050，f_t 等於 0.9 時 50 步後剩 0.0052。藍色虛線是標準 RNN 在 W 等於 0.8 時的上界 0.8^k，50 步後只剩 1.43 乘以 10 的 -5 次方（Semilog plot of the gradient factor against steps back k: belt lines for f = 1, 0.99, 0.9 end at 1.0000, 0.6050, 0.0052 after 50 steps; the dashed plain-RNN bound 0.8^k ends at 1.43 × 10^-5）](../assets/figures/lec08-kp06-cec.png)

**圖說 `lec08-kp06-cec`**〔功能：精準度〕：先看最上面的線：$f_t=1$ 時，梯度乘數 $50$ 步後仍是 $1.0000$，這就是 CEC。再看另外兩條實線：$f_t=0.99$ 時剩 $0.6050$，$f_t=0.9$ 時剩 $0.0052$，遺忘門越小，衰減越快。最後看藍色虛線：標準 RNN 在 $W=0.8$ 時最多剩 $0.8^{50}=1.43\times10^{-5}$。縱軸是對數刻度，所以每條線都是直線，斜率就是每步乘數的對數。

**Caption `lec08-kp06-cec`** [Function: precision]: First look at the top line: with $f_t=1$ the gradient factor is still $1.0000$ after $50$ steps; this is the CEC. Then look at the other two solid lines: with $f_t=0.99$, $0.6050$ remains, and with $f_t=0.9$, $0.0052$; the smaller the forget gate, the faster the decay. Finally look at the blue dashed line: a standard RNN with $W=0.8$ keeps at most $0.8^{50}=1.43\times10^{-5}$. The vertical axis is logarithmic, so every curve is a straight line whose slope is the log of the factor per step.

關鍵差別 `[補充]`：RNN 的乘數由固定的 $W$ 與 $\tanh'$ 決定。網路沒辦法說「這一段要記久一點」。LSTM 的乘數是遺忘門，它由輸入算出來。需要長記憶時，網路可以學會把 $f_t$ 開到接近 $1$；換話題時，再把 $f_t$ 關小。

The key difference `[Added]`: the RNN's factor is fixed by $W$ and $\tanh'$, so the network cannot say "remember this part longer". The LSTM's factor is the forget gate, which is computed from the input. When long memory is needed, the network can learn to open $f_t$ close to $1$; when the topic changes, it closes $f_t$.

例如讀到「新話題：」這類段落開頭時（來源例子），遺忘門會接近 $0$，清掉上一段的背景。輸入門會接近 $1$，全力寫入新話題。

For example, at the start of a passage such as "New topic:" (source example), the forget gate goes close to $0$ and clears the background of the previous passage. The input gate goes close to $1$ and writes the new topic in full.

何時失效 `[補充]`：CEC 只在 $f_t$ 接近 $1$ 時成立。$f_t=0.9$ 時，$50$ 步後梯度只剩 $0.0052$，仍然會消失，只是比 RNN 慢得多。另外，上面的導數只算了傳送帶這一條路；門本身也依賴 $\mathbf{h}_{t-1}$，那些路仍會經過矩陣與 sigmoid。所以 LSTM 是大幅減輕梯度消失，不是保證完全沒有。

When it fails `[Added]`: the CEC holds only when $f_t$ is close to $1$. With $f_t=0.9$ only $0.0052$ of the gradient is left after $50$ steps, so it still vanishes, just far more slowly than in an RNN. Also, the derivative above counts only the belt path; the gates themselves depend on $\mathbf{h}_{t-1}$, and those paths still pass through matrices and sigmoids. So the LSTM greatly reduces the vanishing gradient; it does not guarantee that it never happens.

常見錯誤：以為「傳送帶是加法，所以梯度永遠是 $1$」。沿傳送帶每一步的導數是 $f_t$，不是 $1$。只有遺忘門全開時才是 $1$。

Common error: thinking that "the belt is additive, so the gradient is always $1$". Along the belt the derivative per step is $f_t$, not $1$. It is $1$ only when the forget gate is fully open.

### 本單元重點整理（KP06 summary）

**一句話重點**：LSTM 用三個 sigmoid 軟開關控制資訊流，細胞狀態以 $\mathbf{c}_t=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t$ 加法更新，沿這條傳送帶往回的梯度每步只乘遺忘門，所以 $\mathbf{f}_t$ 接近 $1$ 時梯度幾乎不衰減。

**One-sentence summary**: an LSTM controls its information flow with three sigmoid soft switches and updates its cell state additively by $\mathbf{c}_t=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t$; along this conveyor belt the gradient going back is multiplied only by the forget gate at each step, so it hardly decays when $\mathbf{f}_t$ is close to $1$.

**白話說明**：標準 RNN 每一步都把記憶整個重寫一遍。LSTM 則把記憶放在一條傳送帶上：遺忘門決定刮掉多少，輸入門決定裝上多少新內容，輸出門決定對外公開多少。帶子上只有乘比例和加法，所以記憶與梯度都能走得很遠。

**Plain explanation**: a standard RNN rewrites its whole memory at every step. An LSTM instead puts the memory on a conveyor belt: the forget gate decides how much to scrape off, the input gate how much new content to mount, and the output gate how much to show outside. The belt has only ratio multiplications and additions, so both memory and gradients can travel far.

**正式定義**：$\mathbf{f}_t=\sigma(\mathbf{W}_f\mathbf{h}_{t-1}+\mathbf{U}_f\mathbf{x}_t+\mathbf{p}_f\odot\mathbf{c}_{t-1}+\mathbf{b}_f)$，$\mathbf{i}_t$ 同理；$\tilde{\mathbf{c}}_t=\tanh(\mathbf{W}_c\mathbf{h}_{t-1}+\mathbf{U}_c\mathbf{x}_t+\mathbf{b}_c)$；$\mathbf{c}_t=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t$；$\mathbf{o}_t=\sigma(\mathbf{W}_o\mathbf{h}_{t-1}+\mathbf{U}_o\mathbf{x}_t+\mathbf{p}_o\odot\mathbf{c}_t+\mathbf{b}_o)$；$\mathbf{h}_t=\mathbf{o}_t\odot\tanh(\mathbf{c}_t)$；$\mathbf{y}_t=\mathbf{V}\mathbf{h}_t+\mathbf{b}_y$。

**Formal definition**: $\mathbf{f}_t=\sigma(\mathbf{W}_f\mathbf{h}_{t-1}+\mathbf{U}_f\mathbf{x}_t+\mathbf{p}_f\odot\mathbf{c}_{t-1}+\mathbf{b}_f)$, and $\mathbf{i}_t$ likewise; $\tilde{\mathbf{c}}_t=\tanh(\mathbf{W}_c\mathbf{h}_{t-1}+\mathbf{U}_c\mathbf{x}_t+\mathbf{b}_c)$; $\mathbf{c}_t=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t$; $\mathbf{o}_t=\sigma(\mathbf{W}_o\mathbf{h}_{t-1}+\mathbf{U}_o\mathbf{x}_t+\mathbf{p}_o\odot\mathbf{c}_t+\mathbf{b}_o)$; $\mathbf{h}_t=\mathbf{o}_t\odot\tanh(\mathbf{c}_t)$; $\mathbf{y}_t=\mathbf{V}\mathbf{h}_t+\mathbf{b}_y$.

**例子**：$h_{t-1}=0.5$、$c_{t-1}=2.0$、$x_t=1.0$：$f_t=0.8957$、$i_t=0.6106$、$\tilde{c}_t=0.6640$，$c_t=1.7913+0.4055=2.1968$，$o_t=0.7251$，$h_t=0.7074$，$y_t=0.8488$。沿傳送帶 $50$ 步：$f_t=1$ 剩 $1.0000$，$f_t=0.9$ 剩 $0.0052$；RNN 在 $W=0.8$ 時最多剩 $1.43\times10^{-5}$。

**Example**: $h_{t-1}=0.5$, $c_{t-1}=2.0$, $x_t=1.0$: $f_t=0.8957$, $i_t=0.6106$, $\tilde{c}_t=0.6640$, $c_t=1.7913+0.4055=2.1968$, $o_t=0.7251$, $h_t=0.7074$, $y_t=0.8488$. Along the belt for $50$ steps: $1.0000$ remains with $f_t=1$ and $0.0052$ with $f_t=0.9$; an RNN with $W=0.8$ keeps at most $1.43\times10^{-5}$.

**常見迷思**：

- 「門只輸出 $0$ 或 $1$。」錯，門是 $(0,1)$ 之間的連續比例，才能微分、才能訓練。
- 「$\mathbf{c}_t$ 和 $\mathbf{h}_t$ 是同一個狀態。」錯，$\mathbf{c}_t$ 是內部長期記憶，$\mathbf{h}_t=\mathbf{o}_t\odot\tanh(\mathbf{c}_t)$ 是對外的短期狀態。
- 「有了 LSTM，梯度永遠不會消失。」錯，沿傳送帶每步乘 $f_t$，只有 $f_t$ 接近 $1$ 時才幾乎不衰減。
- 「LSTM 的輸入門 $\mathbf{i}_t$ 就是 KP01 的淨輸入 $\mathbf{i}_t$。」錯，兩者同名但意思不同。

**Common misconceptions**:

- "A gate outputs only $0$ or $1$." Wrong: a gate is a continuous ratio in $(0,1)$, which is what makes it differentiable and trainable.
- "$\mathbf{c}_t$ and $\mathbf{h}_t$ are the same state." Wrong: $\mathbf{c}_t$ is the internal long-term memory, and $\mathbf{h}_t=\mathbf{o}_t\odot\tanh(\mathbf{c}_t)$ is the short-term state shown outside.
- "With an LSTM the gradient never vanishes." Wrong: along the belt it is multiplied by $f_t$ at each step and hardly decays only when $f_t$ is close to $1$.
- "The LSTM input gate $\mathbf{i}_t$ is the net input $\mathbf{i}_t$ of KP01." Wrong: they share a name but mean different things.

**圖**：本單元有三張圖。

- `lec08-kp06-sigmoid-gates`：sigmoid 把門的淨輸入變成 $(0,1)$ 的放行比例。
- `lec08-kp06-cell`：LSTM 細胞的傳送帶、三個門與一整步的數字。
- `lec08-kp06-cec`：沿傳送帶與沿標準 RNN 往回 $k$ 步的梯度乘數。

**Figures** — this unit has three figures.

- `lec08-kp06-sigmoid-gates`: the sigmoid turns a gate's net input into a pass ratio in $(0,1)$.
- `lec08-kp06-cell`: the conveyor belt, the three gates and the numbers of one full step of an LSTM cell.
- `lec08-kp06-cec`: the gradient factor $k$ steps back along the belt and along a standard RNN.

### 小練習（Practice）

1. 讀到新話題時，設 $c_{t-1}=2.0$、$f_t=0.05$、$i_t=0.95$、$\tilde{c}_t=-0.6$。算出 $c_t$，並說明舊記憶還剩多少。提示：分別算保留項 $f_tc_{t-1}$ 與寫入項 $i_t\tilde{c}_t$。
2. 若遺忘門每一步都是 $f_t=0.9$，沿傳送帶往回走幾步，梯度乘數會第一次小於 $0.5$？提示：找最小的 $k$ 使 $0.9^k<0.5$。
3. 設 $d=3$、$p=4$、$q=2$，含窺視孔的 LSTM 共有幾個參數？和 KP01 同樣大小的 RNN（$42$ 個）比較。提示：三個門加候選記憶共四組，每組有 $\mathbf{W}$、$\mathbf{U}$、偏置；窺視孔有三個 $p$ 維向量；別忘了 $\mathbf{V}$ 與 $\mathbf{b}_y$。

1. At a new topic, let $c_{t-1}=2.0$, $f_t=0.05$, $i_t=0.95$ and $\tilde{c}_t=-0.6$. Compute $c_t$ and say how much of the old memory is left. Hint: compute the kept term $f_tc_{t-1}$ and the written term $i_t\tilde{c}_t$ separately.
2. If the forget gate is $f_t=0.9$ at every step, after how many steps back along the belt does the gradient factor first drop below $0.5$? Hint: find the smallest $k$ with $0.9^k<0.5$.
3. Let $d=3$, $p=4$, $q=2$. How many parameters does an LSTM with peepholes have? Compare with the RNN of the same size from KP01 ($42$). Hint: the three gates plus the candidate make four groups, each with a $\mathbf{W}$, a $\mathbf{U}$ and a bias; the peepholes are three $p$-dimensional vectors; do not forget $\mathbf{V}$ and $\mathbf{b}_y$.

**第 1 題解答** `[補充]`：保留項 $0.05\times2.0=0.1$，寫入項 $0.95\times(-0.6)=-0.57$，$c_t=0.1-0.57=-0.47$。舊記憶只剩 $0.1$，記憶幾乎換成新話題的內容，連正負號都翻了。

**Solution to question 1** `[Added]`: the kept term is $0.05\times2.0=0.1$ and the written term is $0.95\times(-0.6)=-0.57$, so $c_t=0.1-0.57=-0.47$. Only $0.1$ of the old memory is left; the memory is almost replaced by the new topic, and even its sign flips.

**第 2 題解答** `[補充]`：$0.9^6=0.5314$ 還大於 $0.5$，$0.9^7=0.4783$ 小於 $0.5$，所以是 $k=7$ 步。遺忘門只要不是 $1$，梯度仍會指數衰減。

**Solution to question 2** `[Added]`: $0.9^6=0.5314$ is still above $0.5$ and $0.9^7=0.4783$ is below, so $k=7$ steps. As long as the forget gate is not $1$, the gradient still decays exponentially.

**第 3 題解答** `[補充]`：每組 $\mathbf{W}$ 有 $4\times4=16$ 個、$\mathbf{U}$ 有 $4\times3=12$ 個、偏置 $4$ 個，共 $32$ 個；四組是 $128$ 個。窺視孔 $3\times4=12$ 個，輸出層 $\mathbf{V}$ 與 $\mathbf{b}_y$ 有 $8+2=10$ 個。合計 $150$ 個，約是 RNN 的 $42$ 個的 $3.6$ 倍。門越多，參數越多；KP07 會看到更省的設計。

**Solution to question 3** `[Added]`: each group has $\mathbf{W}$ with $4\times4=16$, $\mathbf{U}$ with $4\times3=12$ and a bias with $4$, so $32$; four groups give $128$. The peepholes have $3\times4=12$, and the output layer $\mathbf{V}$ and $\mathbf{b}_y$ has $8+2=10$. The total is $150$, about $3.6$ times the $42$ of the RNN. More gates mean more parameters; KP07 shows cheaper designs.

### 小測驗（Quiz，來源原題）

來源的兩題自我檢測保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The two self-check questions of the source are kept with all options; the source page does not mark the answers directly, and the explanations below follow the source text.

1. LSTM 中三大門（Input, Forget, Output Gates）使用 Sigmoid 激活函數的主要原因是什麼？
   - (A) Sigmoid 輸出在 $0$ 到 $1$ 之間，正好充當開關比例（$0$ 代表完全關閉，$1$ 代表完全開啟）
   - (B) Sigmoid 計算速度比 $\tanh$ 快
   - (C) Sigmoid 導數永遠大於 $1$
   - (D) 為了讓輸出變成負數
2. LSTM 的細胞狀態更新公式 $\mathbf{c}_t=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t$ 為何能創造「常數誤差流（CEC）」？
   - (A) 因為更新公式包含加法結構，當遺忘門 $f_t$ 接近 $1$ 時，反向傳播的梯度能無衰減地沿加法路徑跨越長時間步傳遞
   - (B) 因為它使用了矩陣求逆
   - (C) 因為它不需要反向傳播
   - (D) 因為它把所有權重設為 $0$

1. What is the main reason the three gates of an LSTM (input, forget, output) use the sigmoid activation?
   - (A) The sigmoid outputs a value between $0$ and $1$, which works as a switch ratio ($0$ fully closed, $1$ fully open)
   - (B) The sigmoid is faster to compute than $\tanh$
   - (C) The derivative of the sigmoid is always greater than $1$
   - (D) To make the output negative
2. Why does the LSTM cell-state update $\mathbf{c}_t=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t$ create a "constant error carousel (CEC)"?
   - (A) Because the update has an additive structure; when the forget gate $f_t$ is close to $1$, the backpropagated gradient travels along the additive path across many time steps without decaying
   - (B) Because it uses a matrix inverse
   - (C) Because it does not need backpropagation
   - (D) Because it sets all weights to $0$

**解析** `[補充]`：第 1 題選 (A)：第 1 節說門是「放行比例」，必須落在 $(0,1)$。(B) 不是主因；(C) 錯，sigmoid 的導數最大只有 $0.25$；(D) 錯，sigmoid 永遠是正的。第 2 題選 (A)：第 5 節算出沿傳送帶每步的導數是 $f_t$，$f_t$ 接近 $1$ 時幾乎不衰減。(B)(C)(D) 都與式子無關；LSTM 仍然要用反向傳播訓練。

**Explanations** `[Added]`: question 1 is A: Section 1 says a gate is a "pass ratio" and must lie in $(0,1)$. B is not the main reason; C is wrong, since the sigmoid's derivative is at most $0.25$; D is wrong, since the sigmoid is always positive. Question 2 is A: Section 5 shows that the derivative per step along the belt is $f_t$, which hardly decays when $f_t$ is close to $1$. B, C and D have nothing to do with the equation; an LSTM is still trained by backpropagation.

### 留給下一單元的問題（The question for the next unit）

本單元的 LSTM 有遺忘門，需要時可以清掉舊記憶。1997 年最早的 LSTM 卻沒有遺忘門，相當於 $f_t$ 永遠是 $1$。在很長的序列上，記憶只加不減，會出什麼問題？後來的研究又如何用更少的門，做出更輕的門控單元（GRU、MGU）？下一單元（KP07）回答這兩個問題。

The LSTM of this unit has a forget gate and can clear old memory when needed. But the first LSTM of 1997 had no forget gate, which amounts to $f_t$ always being $1$. On very long sequences, what goes wrong when the memory only grows and never shrinks? And how did later work build lighter gated units (GRU, MGU) with fewer gates? The next unit (KP07) answers these two questions.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第8章「循環神經網絡與長短期記憶網絡」§4.1.1–4.1.7（「長短期記憶網路（LSTM）門控機制：三大門的作用與算式」與「LSTM 細胞狀態傳送帶、隱藏狀態與最終輸出」）及其教授祕笈與自我檢測（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch08-rnn-lstm/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §4.1.1–4.1.7 of Chapter 8, "Recurrent neural networks and long short-term memory networks", of CCC-LNS IDL_115 (the sections on the three LSTM gates and on the cell-state conveyor belt, hidden state and output), with their professor's notes and self-checks (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch08-rnn-lstm/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源教授祕笈寫 $h_t=0.9\times\tanh(1.90)=0.9\times0.956=0.860$。程式算得 $\tanh(1.90)=0.9562$，$0.9\times0.9562=0.8606$，到小數第三位是 $0.861$。來源先把 $\tanh$ 捨到三位再相乘，最後一位因此偏低；本講義未使用此例，只在此記錄。
- 來源開頭說 LSTM 是為了「徹底」解決梯度消失，教授祕笈也說傳送帶「求導結果為 1」「幾百步都不衰減」。正確的說法是：沿傳送帶每步的導數是 $f_t$，只有 $f_t=1$ 時才等於 $1$（來源內文的 CEC 段落正是這樣寫）。本講義第 5 節依此說明，並補上 $f_t<1$ 時仍會衰減的數字。
- 來源用 $\mathbf{i}_t$、$\mathbf{b}_i$ 表示輸入門與其偏置，和 KP01 的淨輸入、隱藏層偏置同名。本講義保留來源寫法，並在符號表與第 2 節明確區分。
- 來源用粗斜體 `\boldsymbol` 與 `\widetilde`；本課程網站不支援 `\boldsymbol`，改用 `\mathbf` 與 `\tilde`，意義相同。
- 本講義補上：與 KP05 滲漏單元的對照、逐元素乘積的小例子、窺視孔為什麼用向量以及輸出門為什麼看 $\mathbf{c}_t$、一整步的純量算例與無窺視孔對照、沿傳送帶的導數推導與 $50$ 步的數值比較、三題小練習與解答。
- 取捨：來源的兩題自我檢測保留原題；來源的兩個互動計算器無法放進講義，改用 `lec08-kp06-sigmoid-gates` 與 `lec08-kp06-cell` 呈現；來源的城堡門衛與工廠傳送帶兩個類比都保留。

- The source's professor's note writes $h_t=0.9\times\tanh(1.90)=0.9\times0.956=0.860$. The program gives $\tanh(1.90)=0.9562$ and $0.9\times0.9562=0.8606$, which is $0.861$ at three decimals. The source rounded $\tanh$ to three digits before multiplying, so its last digit is too low; this note does not use that example and only records the correction here.
- The source opens by saying the LSTM "completely" solves the vanishing gradient, and its professor's note says the belt "has derivative 1" and "does not decay over hundreds of steps". The accurate statement is: along the belt the derivative per step is $f_t$, which equals $1$ only when $f_t=1$ (exactly what the CEC paragraph of the source text says). Section 5 of this note follows that and adds the numbers showing decay when $f_t<1$.
- The source writes the input gate and its bias as $\mathbf{i}_t$ and $\mathbf{b}_i$, the same names as the net input and the hidden bias of KP01. This note keeps the source's names and separates them explicitly in the symbol table and Section 2.
- The source uses bold italic `\boldsymbol` and `\widetilde`; the course website does not support `\boldsymbol`, so this note uses `\mathbf` and `\tilde` with the same meaning.
- This note adds: the comparison with the KP05 leaky unit, a small example of the element-wise product, why the peephole is a vector and why the output gate looks at $\mathbf{c}_t$, a full scalar step with a no-peephole comparison, the derivative along the belt with a $50$-step numerical comparison, and three practice questions with solutions.
- Choices: the two self-check questions of the source are kept as asked; the source's two interactive calculators cannot go into a note, so `lec08-kp06-sigmoid-gates` and `lec08-kp06-cell` show the same ideas; both source analogies, the castle guards and the factory conveyor belt, are kept.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 長短期記憶網路 | long short-term memory (LSTM) |
| 門 | gate |
| 遺忘門 | forget gate |
| 輸入門 | input gate |
| 輸出門 | output gate |
| sigmoid 函數 | sigmoid function |
| 軟開關 | soft switch |
| 逐元素乘積 | element-wise product |
| 窺視孔連線 | peephole connection |
| 候選記憶（區塊輸入） | candidate memory (block input) |
| 細胞狀態 | cell state |
| 傳送帶 | conveyor belt |
| 隱藏狀態（區塊輸出） | hidden state (block output) |
| 常數誤差流 | constant error carousel (CEC) |
| 梯度消失 | vanishing gradient |
| 滲漏單元（KP05） | leaky unit |
| 循環神經網路 | recurrent neural network (RNN) |
| 線性整流函數 | rectified linear unit (ReLU) |
| 門控循環單元（KP07） | gated recurrent unit (GRU) |
| 極簡門控單元（KP07） | minimal gated unit (MGU) |
| 本課程的網頁教材 | CCC-LNS |
