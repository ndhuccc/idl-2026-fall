# 第11週 KP06：為什麼不用 RNN：平行、路徑長度與 $n^2$ 的代價，以及 BERT 與 GPT（Why no RNN: parallelism, path length and the cost of $n^2$, plus BERT and GPT）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第11週的第六個知識點（KP06），接在 KP05 之後。到 KP05 為止，整個 Transformer 已經組好了，裡面完全沒有 RNN。這一節算帳：拿掉 RNN 換到了什麼（可以平行、遠處的詞一步就連上），付出了什麼（計算與記憶體隨句長的平方成長），以及只取編碼器或只取解碼器，怎麼變成後來的 BERT 與 GPT。

This is the sixth knowledge point (KP06) of Week 11 and follows KP05. By KP05 the whole Transformer is assembled, with no RNN anywhere. This note does the accounting: what removing the RNN buys (parallel computation, distant words linked in one step), what it costs (computation and memory growing with the square of the sentence length), and how taking only the encoder or only the decoder led to BERT and GPT.

先備知識是第8週的 RNN、本週 KP01 的三種家族、KP03 的多頭注意力與 KP05 的因果遮罩。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第9章改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec11-kp06-figures.py` 算出，可以自己重算。

The prerequisites are the RNN of Week 8, this week's KP01 three families, KP03's multi-head attention and KP05's causal mask. Every term and symbol is defined when it first appears. This note is rewritten from Chapter 9 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec11-kp06-figures.py`, so you can recompute it yourself.

## KP06 為什麼不用 RNN：平行、路徑長度與 $n^2$ 的代價，以及 BERT 與 GPT（Why no RNN: parallelism, path length and the cost of $n^2$, plus BERT and GPT）

### 學習目標（Learning objectives）

讀完之後，你能用「循序步數」與「最長路徑」兩個量比較 RNN、CNN 與自注意力。你能把一層自注意力的計算量拆成 $4n^2d$ 與 $6nd^2$ 兩項，算出哪一項在什麼句長開始主導，並估計 $n\times n$ 權重表的記憶體。你也能說明 BERT 與 GPT 各用 Transformer 的哪一半、怎麼訓練、各擅長什麼。

After reading, you can compare RNNs, CNNs and self-attention with two quantities, "sequential steps" and "longest path". You can split the computation of a self-attention layer into the two terms $4n^2d$ and $6nd^2$, find the sentence length at which each dominates, and estimate the memory of the $n\times n$ weight table. You can also explain which half of the Transformer BERT and GPT each use, how they are trained, and what each is good at.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $n$ | 句子的詞數（序列長度） | number of words in the sentence (sequence length) |
| $d$ | 詞向量長度，本節取 $d=d_{\text{model}}=512$ | word-vector length, $d=d_{\text{model}}=512$ in this note |
| $k$ | 卷積核的寬度 | width of a convolution kernel |
| $\mathcal{O}(\cdot)$ | 大 $\mathcal{O}$ 記號：$n$ 變大時成長的速度，忽略常數 | big-O notation: how fast something grows as $n$ grows, ignoring constants |
| FLOPs | 浮點運算次數，一次乘法加一次加法算 $2$ 次 | number of floating-point operations, one multiply plus one add counting as $2$ |

### 1. 平行：RNN 要排隊，自注意力一次算完（Parallelism: the RNN queues up, self-attention computes at once）

第8週的 RNN 讀第 $t$ 個詞時，要用到第 $t-1$ 步的隱藏狀態，所以必須等上一步算完。一句 $n$ 個詞就要排隊算 $n$ 步。一層裡必須一步接一步做的步數叫**循序步數**（sequential steps），RNN 是 $\mathcal{O}(n)$。GPU 一次能做大量乘法，卻只能等。

When the RNN of Week 8 reads word $t$, it needs the hidden state of step $t-1$, so it must wait for the previous step. A sentence of $n$ words needs $n$ steps in a queue. The number of steps in a layer that must happen one after another is called the **sequential steps**; for an RNN it is $\mathcal{O}(n)$. The GPU could do a great many multiplications at once, but it has to wait.

自注意力把整句排成一個 $n\times d$ 矩陣，$\mathbf{Q}\mathbf{K}^\top$ 一次算出所有 $n\times n$ 個詞對的分數，不需要等前一個詞。不管句子多長，一層只要一次矩陣運算，循序步數是 $\mathcal{O}(1)$。這裡數的是「必須等前一步的步數」，不是時間：句子長了，這一次矩陣運算本身也會變久。例如 $n=100$ 時，RNN 一層要排 $100$ 步，自注意力一步。

Self-attention arranges the whole sentence as an $n\times d$ matrix, and $\mathbf{Q}\mathbf{K}^\top$ computes the scores of all $n\times n$ word pairs at once, without waiting for any earlier word. However long the sentence, a layer needs one matrix operation: the sequential steps are $\mathcal{O}(1)$. This counts "steps that must wait for the previous one", not time: for a longer sentence that one matrix operation itself takes longer. For example, with $n=100$, an RNN layer queues $100$ steps, self-attention one.

這也是 KP05 說「訓練時解碼器可以一次平行算」的原因。RNN 訓練時也用教師強制，正確答案也都知道，但它的隱藏狀態仍要一步接一步算；Transformer 沒有一步接一步的隱藏狀態，遮罩只負責讓每個位置看不到未來。

This is also why KP05 said "the decoder computes in parallel during training". RNNs also use teacher forcing in training, and the correct answers are also known, but their hidden states must still be computed one step after another; the Transformer has no step-to-step hidden state, and the mask only hides future words from each position.

常見錯誤：以為 Transformer 推論時也一步產生整句。GPT 這類模型寫字時仍要一個詞一個詞來（KP05），平行只發生在「已知整句」的時候，例如訓練，或讀入一段提示。

Common mistake: thinking the Transformer also produces a whole sentence in one step at inference. Models like GPT still write word by word (KP05); the parallelism happens only when the whole sentence is known, as in training or when reading in a prompt.

### 2. 路徑長度：遠處的詞要隔幾步（Path length: how many steps between distant words）

另一個量是**最長路徑**（maximum path length）：資訊從一個詞傳到另一個詞，最多要經過幾步。步數越多，訊號越容易在途中衰減，梯度也越容易消失（第3週、第8週）。

The other quantity is the **maximum path length**: the most steps information needs to travel from one word to another. The more steps, the more easily the signal fades on the way and the gradient vanishes (Weeks 3 and 8).

- RNN：第 $1$ 個詞到第 $n$ 個詞要經過 $n-1$ 步，是 $\mathcal{O}(n)$。
- 自注意力：任兩個詞在同一層直接算分數，$1$ 步，是 $\mathcal{O}(1)$。
- 卷積（CNN）`[補充]`：寬度 $k$ 的卷積，一層只看到左右相鄰的幾個詞，要疊好幾層才連得到遠處。一般卷積疊 $L$ 層，一個輸出看得到的範圍（感受野）寬 $L(k-1)+1$ 個詞；要讓頭尾兩個詞落在同一個感受野裡，需要 $L(k-1)+1\ge n$，也就是 $\lceil (n-1)/(k-1)\rceil$ 層。原論文表中的 $\mathcal{O}(\log_k n)$ 是擴張卷積（dilated convolution）的情形：每層的取樣點間隔依序是 $1,k,k^2,\dots$，範圍成倍擴大。

- RNN: from word $1$ to word $n$ takes $n-1$ steps, $\mathcal{O}(n)$.
- Self-attention: any two words compute a score directly in the same layer, $1$ step, $\mathcal{O}(1)$.
- Convolution (CNN) `[Added]`: a convolution of width $k$ sees only a few neighbouring words per layer, so several layers are needed to reach far. With $L$ layers of an ordinary convolution, one output sees a span (receptive field) of $L(k-1)+1$ words; for the first and last words to fall in one receptive field we need $L(k-1)+1\ge n$, that is, $\lceil (n-1)/(k-1)\rceil$ layers. The $\mathcal{O}(\log_k n)$ in the original paper's table is for dilated convolutions, whose taps are spaced $1,k,k^2,\dots$ apart layer by layer, so the span multiplies.

![左半邊上面是紅色的 RNN，六個圓圈 1 到 6 用箭頭串成一列，標題寫從第 1 個詞到第 6 個詞要 5 步；下面是藍色的自注意力，同樣六個圓圈，每一對都用淡藍色弧線相連，1 到 6 的弧線加粗，標題寫任兩個詞都是 1 步。右半邊是折線圖，橫軸句長 n 從 2 到 100，紅線 RNN 是 n−1 直線上升到 99，綠線 kernel 3 的 CNN 是 (n−1)/2 取上整，到 50，藍線自注意力一直是 1（Left: on top, a red RNN chain of six circles 1 to 6 joined by arrows, titled 5 steps from word 1 to word 6; below, blue self-attention with the same six circles, every pair joined by a light-blue arc and the 1-to-6 arc in bold, titled 1 step between any two words. Right: a line chart over sentence length n from 2 to 100; the red RNN line n−1 rises to 99, the green CNN line with kernel 3, the ceiling of (n−1)/2, reaches 50, and the blue self-attention line stays at 1）](../assets/figures/lec11-kp06-path-length.png)

**圖說 `lec11-kp06-path-length`**〔功能：一目了然、精準度〕：右圖縱軸的「步」對卷積來說是層數。先看左上：RNN 要沿著箭頭一格一格傳，$6$ 個詞頭尾隔 $5$ 步。再看左下：自注意力每一對詞都有一條線，頭尾那條加粗的只要 $1$ 步。最後看右邊：句子變長時，RNN 的路徑跟著變長（$n=100$ 時 $99$ 步），寬度 $3$ 的卷積約一半（$50$ 層），自注意力永遠是 $1$。

**Caption `lec11-kp06-path-length`** [Function: seeing it at a glance, precision]: on the right, a "step" for the convolution means a layer. first look at the top left: the RNN passes information arrow by arrow, so the two ends of $6$ words are $5$ steps apart. Then look at the bottom left: self-attention has a line between every pair, and the bold end-to-end line takes $1$ step. Finally look at the right: as the sentence grows, the RNN's path grows with it ($99$ steps at $n=100$), a width-$3$ convolution needs about half ($50$ layers), and self-attention always needs $1$.

常見錯誤：以為路徑 $\mathcal{O}(1)$ 就「徹底解決」了長距離依賴。路徑短讓長距離關係**容易學**，但模型仍要從資料學會把權重放在對的詞上；句子太長時，第3節的成本也會限制能讀多長。

Common mistake: thinking an $\mathcal{O}(1)$ path "completely solves" long-range dependencies. A short path makes long-range relations **easier to learn**, but the model must still learn from data to put weight on the right words; and for very long sentences the cost of Section 3 limits how far it can read.

### 3. 代價：$n^2$（The cost: $n^2$）

自注意力的好處來自「每一對詞都算一次」，代價也一樣：有 $n^2$ 對。來源祕笈把一層自注意力的計算量拆成兩項（矩陣乘法 $m\times p$ 乘 $p\times q$ 約 $2mpq$ 次運算）：

Self-attention's benefit comes from "computing every pair once", and so does its cost: there are $n^2$ pairs. The source's professor notes split the computation of one self-attention layer into two terms (multiplying an $m\times p$ matrix by a $p\times q$ matrix takes about $2mpq$ operations):

$$
\text{FLOPs}\approx\underbrace{6nd^2}_{\mathbf{Q},\mathbf{K},\mathbf{V}\ \text{的投影}}+\underbrace{4n^2d}_{\mathbf{Q}\mathbf{K}^\top\ \text{與權重乘}\ \mathbf{V}}.
$$

投影項跟 $n$ 成正比，分數項跟 $n^2$ 成正比。兩項相等時 $4n^2d=6nd^2$，也就是 $n=1.5d$；$d=512$ 時是 $n=768$。句子比這短，投影比較貴；比這長，$n^2$ 那一項很快就吃掉大部分的計算。

The projection term grows like $n$ and the score term like $n^2$. The two are equal when $4n^2d=6nd^2$, that is, $n=1.5d$; for $d=512$, $n=768$. Below that length the projections cost more; above it, the $n^2$ term quickly takes most of the computation.

| $n$ | $4n^2d$ | $6nd^2$ | $n^2$ 項的占比 |
|---|---|---|---|
| $128$ | $3.36\times10^{7}$ | $2.01\times10^{8}$ | $14.3\%$ |
| $1024$ | $2.15\times10^{9}$ | $1.61\times10^{9}$ | $57.1\%$ |
| $4096$ | $3.44\times10^{10}$ | $6.44\times10^{9}$ | $84.2\%$ |
| $32768$ | $2.20\times10^{12}$ | $5.15\times10^{10}$ | $97.7\%$ |

| $n$ | $4n^2d$ | $6nd^2$ | share of the $n^2$ term |
|---|---|---|---|
| $128$ | $3.36\times10^{7}$ | $2.01\times10^{8}$ | $14.3\%$ |
| $1024$ | $2.15\times10^{9}$ | $1.61\times10^{9}$ | $57.1\%$ |
| $4096$ | $3.44\times10^{10}$ | $6.44\times10^{9}$ | $84.2\%$ |
| $32768$ | $2.20\times10^{12}$ | $5.15\times10^{10}$ | $97.7\%$ |

記憶體也一樣。訓練時要把 $n\times n$ 的權重表存下來做倒傳遞，每個數用 $4$ 個位元組（float32；本講義 $1$ MB $=10^6$ 位元組、$1$ GB $=10^9$ 位元組）。$n=1024$ 時一張表 $4.19$ MB；句長變成 $32$ 倍、$n=32768$ 時，一張表變成 $1024$ 倍，$4.29$ GB，而這只是一層的一個頭。

Memory behaves the same way. In training, the $n\times n$ weight table must be stored for backpropagation, at $4$ bytes per number (float32; in this note $1$ MB $=10^6$ bytes and $1$ GB $=10^9$ bytes). At $n=1024$ one table takes $4.19$ MB; with a sentence $32$ times longer, $n=32768$, one table becomes $1024$ times larger, $4.29$ GB, and that is for a single head of a single layer.

![兩張對數座標圖。左圖橫軸句長 n、縱軸每層自注意力的 FLOPs：綠線投影 6nd² 斜率較緩，紅線 4n²d 斜率較陡，兩線在 n = 768 交叉，虛線是總和，n 很大時幾乎和紅線重合。右圖紫線是一張 n×n 權重表的位元組數，標出 n = 1024 時 4.19 MB、n = 32768 時 4.29 GB（Two log-log charts. Left, FLOPs per self-attention layer against sentence length n: the green projection line 6nd² rises gently, the red 4n²d line rises steeply, they cross at n = 768, and the dashed total almost coincides with the red line for large n. Right, the purple line gives the bytes of one n-by-n weight table, marking 4.19 MB at n = 1024 and 4.29 GB at n = 32768）](../assets/figures/lec11-kp06-cost-vs-n.png)

**圖說 `lec11-kp06-cost-vs-n`**〔功能：一目了然、精準度〕：先看左圖：兩軸都是對數刻度，綠線（$\propto n$）比紅線（$\propto n^2$）緩，兩線在 $n=768$ 交叉；交叉之後，總和（虛線）幾乎就是紅線。再看右圖：權重表的記憶體也是 $n^2$，句長每多 $10$ 倍，記憶體多 $100$ 倍。

**Caption `lec11-kp06-cost-vs-n`** [Function: seeing it at a glance, precision]: first look at the left chart: both axes are logarithmic, the green line ($\propto n$) is gentler than the red line ($\propto n^2$), and they cross at $n=768$; after the crossing, the total (dashed) is almost the red line. Then look at the right chart: the weight table's memory also grows like $n^2$, so every $10$ times longer sentence needs $100$ times more memory.

原論文的比較表把這些量整理在一起 `[部分補充：來源列出自注意力與 RNN 兩列，卷積與只看鄰近兩列整理自原論文表 1]`：

The original paper's comparison table puts these quantities together `[Partly added: the source lists the self-attention and RNN rows; the convolution and restricted rows come from Table 1 of the original paper]`:

| 層的種類 | 每層計算量 | 循序步數 | 最長路徑 |
|---|---|---|---|
| 自注意力 | $\mathcal{O}(n^2d)$ | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ |
| RNN | $\mathcal{O}(nd^2)$ | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ |
| 卷積（寬度 $k$） | $\mathcal{O}(knd^2)$ | $\mathcal{O}(1)$ | $\mathcal{O}(\log_k n)$（擴張卷積） |
| 只看鄰近 $r$ 個詞的自注意力 | $\mathcal{O}(rnd)$ | $\mathcal{O}(1)$ | $\mathcal{O}(n/r)$ |

| Layer type | Computation per layer | Sequential steps | Longest path |
|---|---|---|---|
| Self-attention | $\mathcal{O}(n^2d)$ | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ |
| RNN | $\mathcal{O}(nd^2)$ | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ |
| Convolution (width $k$) | $\mathcal{O}(knd^2)$ | $\mathcal{O}(1)$ | $\mathcal{O}(\log_k n)$ (dilated) |
| Self-attention restricted to $r$ neighbours | $\mathcal{O}(rnd)$ | $\mathcal{O}(1)$ | $\mathcal{O}(n/r)$ |

最後一列是下一單元的伏筆：只看鄰近的詞可以把 $n^2$ 降下來，代價是路徑又變長了。

The last row foreshadows the next unit: attending only to nearby words brings $n^2$ down, at the price of a longer path again.

常見錯誤：以為多頭會讓計算量乘上 $h$。KP03 算過，$h$ 個頭各用 $d/h$ 維，乘法總量和一個全寬的頭相同。但每個頭各有一張 $n\times n$ 權重表，所以記憶體與 softmax 的工作量是 $hn^2$。

Common mistake: thinking multiple heads multiply the computation by $h$. As KP03 showed, $h$ heads of $d/h$ dimensions each cost the same total number of multiplications as one full-width head. But each head keeps its own $n\times n$ weight table, so the memory and the softmax work are $hn^2$.

### 4. 只取一半：BERT 與 GPT（Taking one half: BERT and GPT）

KP01 提過 Transformer 的三種家族。2018 年起，兩種「只取一半」的作法成為主流，差別在訓練的題目：

KP01 mentioned the three families of Transformers. From 2018, two "one-half" designs became dominant; they differ in the task they are trained on:

- **BERT**（只用編碼器）：做**克漏字**。隨機選出約 $15\%$ 的詞元（token），其中 $80\%$ 換成 `[MASK]`、$10\%$ 換成隨機的詞、$10\%$ 保持不變，要模型猜回原來的詞；這叫遮罩語言模型（masked language model，MLM）。自注意力**不加**因果遮罩，每個詞都看得到左右兩邊。
- **GPT**（只用解碼器，拿掉交叉注意力）：做**接龍**。給前面的詞，猜下一個詞；這叫因果語言模型。自注意力加上 KP05 的因果遮罩，只看左邊。

- **BERT** (encoder only) does **cloze**: about $15\%$ of the tokens are selected at random, of which $80\%$ become `[MASK]`, $10\%$ a random word and $10\%$ stay unchanged, and the model must recover the original words; this is called a masked language model (MLM). Self-attention has **no** causal mask, so every word sees both sides.
- **GPT** (decoder only, without cross-attention) does **next-word prediction**: given the earlier words, guess the next one; this is called a causal language model. Self-attention uses KP05's causal mask and sees only the left.

來源祕笈的例子：`I [MASK] deep learning`。BERT 算 `[MASK]` 位置的向量時，同時看左邊的 `I` 和右邊的 `deep`、`learning`，猜出 `love`。GPT 則是：給 `I` 猜 `love`，給 `I love` 猜 `deep`，給 `I love deep` 猜 `learning`，每個位置都在出題。

The example from the source's professor notes: `I [MASK] deep learning`. When BERT computes the vector at the `[MASK]` position, it looks at `I` on the left and `deep`, `learning` on the right at the same time, and guesses `love`. GPT instead: given `I`, guess `love`; given `I love`, guess `deep`; given `I love deep`, guess `learning`; every position sets a question.

所以兩者各有長處 `[補充：數字示意]`。一段 $512$ 個詞元的文字，BERT 只有被選出的約 $77$ 個位置產生訓練訊號，但每個位置都看得到全文，擅長理解型工作，例如分類、找出人名地名、從文章中抽答案。GPT 幾乎每個位置（約 $512$ 個）都在出題，而且訓練方式和寫字方式一模一樣，擅長生成，例如續寫、對話、寫程式。原始 Transformer 與 T5 這類編碼器—解碼器模型，則常用在翻譯、摘要這類「輸入一段、輸出另一段」的工作。

So each has its strengths `[Added: numerical illustration]`. For a text of $512$ tokens, BERT gets a training signal only at the roughly $77$ selected positions, but every position sees the whole text, which suits understanding tasks such as classification, finding names of people and places, and extracting answers from a passage. GPT sets a question at almost every position (about $512$), and trains exactly the way it writes, which suits generation such as continuation, dialogue and code. Encoder-decoder models such as the original Transformer and T5 are often used for "one text in, another text out" tasks such as translation and summarization.

常見錯誤：以為 GPT 不能做分類，或 BERT 能像 GPT 一樣逐字寫長文。GPT 可以把分類寫成「接下一個詞」，例如在句尾加上「情感是：」；BERT 沒有為逐字生成而訓練，硬拿來一個一個填，品質通常不好。

Common mistake: thinking GPT cannot classify, or that BERT can write long texts word by word like GPT. GPT can turn classification into "predict the next word", for example by ending the text with "The sentiment is:"; BERT was not trained for word-by-word generation, and forcing it to fill in one word at a time usually gives poor quality.

### 本單元重點整理（KP06 summary）

**一句話重點**：拿掉 RNN 之後，一層的循序步數與任兩詞的最長路徑都從 $\mathcal{O}(n)$ 變成 $\mathcal{O}(1)$，訓練可以整句平行；代價是每一對詞都要算，計算量多了 $4n^2d$ 一項，權重表的記憶體是 $n^2$，句長超過約 $1.5d$ 後由它主導；只取編碼器做克漏字是 BERT，只取解碼器做接龍是 GPT。

**One-sentence summary**: without the RNN, a layer's sequential steps and the longest path between two words both drop from $\mathcal{O}(n)$ to $\mathcal{O}(1)$, so training runs in parallel over the whole sentence; the price is computing every pair of words, which adds a $4n^2d$ term and $n^2$ memory for the weight table, dominant once the sentence exceeds about $1.5d$; the encoder alone trained on cloze is BERT, the decoder alone trained on next-word prediction is GPT.

**白話說明**：RNN 像傳話遊戲，一個傳一個，又慢又容易傳錯；自注意力像開會，每個人都能直接和每個人說話，很快，但人一多，要說的話是人數的平方。

**Plain explanation**: an RNN is like a game of telephone, passing the message person to person, slow and error-prone; self-attention is like a meeting where everyone can talk directly to everyone, fast, but with more people the number of conversations grows with the square of the headcount.

**正式定義**：自注意力每層 $\text{FLOPs}\approx6nd^2+4n^2d$，記憶體 $\mathcal{O}(n^2)$，循序步數 $\mathcal{O}(1)$，最長路徑 $\mathcal{O}(1)$；RNN 每層 $\mathcal{O}(nd^2)$，循序步數與最長路徑都是 $\mathcal{O}(n)$。

**Formal definition**: self-attention per layer: $\text{FLOPs}\approx6nd^2+4n^2d$, memory $\mathcal{O}(n^2)$, sequential steps $\mathcal{O}(1)$, longest path $\mathcal{O}(1)$; RNN per layer: $\mathcal{O}(nd^2)$, with sequential steps and longest path both $\mathcal{O}(n)$.

**例子**：$d=512$ 時兩項在 $n=768$ 相等；$n=1024$ 一層約 $3.76\times10^9$ FLOPs、權重表 $4.19$ MB；$n=32768$ 時 $n^2$ 項占 $97.7\%$，權重表 $4.29$ GB。

**Example**: with $d=512$ the two terms are equal at $n=768$; at $n=1024$ a layer takes about $3.76\times10^9$ FLOPs and the weight table $4.19$ MB; at $n=32768$ the $n^2$ term is $97.7\%$ and the weight table $4.29$ GB.

**常見迷思**：

- 「Transformer 推論時也一次產生整句。」錯，生成仍逐字。
- 「路徑 $\mathcal{O}(1)$ 就徹底解決長距離依賴。」錯，只是比較容易學。
- 「多頭讓計算量變成 $h$ 倍。」錯，乘法總量不變（但權重表有 $h$ 張）。

**Common misconceptions**:

- "The Transformer produces a whole sentence at once at inference." Wrong: generation is still word by word.
- "An $\mathcal{O}(1)$ path completely solves long-range dependencies." Wrong: it only makes them easier to learn.
- "Multiple heads multiply the computation by $h$." Wrong: the total number of multiplications stays the same (though there are $h$ weight tables).

**圖**：本單元有兩張圖。

- `lec11-kp06-path-length`：RNN、CNN 與自注意力的最長路徑。
- `lec11-kp06-cost-vs-n`：計算量的兩項與權重表的記憶體隨句長的變化。

**Figures** — this unit has two figures.

- `lec11-kp06-path-length`: the longest path of RNNs, CNNs and self-attention.
- `lec11-kp06-cost-vs-n`: the two computation terms and the weight-table memory as the sentence grows.

### 小練習（Practice）

1. $d=768$。$4n^2d$ 與 $6nd^2$ 在多少 $n$ 相等？
2. 一層一個頭的權重表用 float32 存。$n=2048$ 時是多少位元組？$n$ 變成 $4$ 倍時是多少倍？
3. 一句 $n=50$ 個詞。RNN 與寬度 $k=5$ 的一般卷積，第 $1$ 個詞到第 $50$ 個詞的最長路徑各是幾步（層）？

1. With $d=768$, at what $n$ are $4n^2d$ and $6nd^2$ equal?
2. One head's weight table in one layer is stored in float32. How many bytes is it at $n=2048$? By what factor does it grow when $n$ becomes $4$ times larger?
3. A sentence has $n=50$ words. What is the longest path from word $1$ to word $50$ for an RNN and for an ordinary convolution of width $k=5$, in steps (layers)?

**第 1 題解答** `[補充]`：$n=1.5d=1152$。

**Solution to question 1** `[Added]`: $n=1.5d=1152$.

**第 2 題解答** `[補充]`：$4\times2048^2=16777216$ 位元組，約 $16.8$ MB（$1$ MB $=10^6$ 位元組）；$n$ 變 $4$ 倍，記憶體變 $16$ 倍。

**Solution to question 2** `[Added]`: $4\times2048^2=16777216$ bytes, about $16.8$ MB ($1$ MB $=10^6$ bytes); with $n$ four times larger, the memory grows $16$ times.

**第 3 題解答** `[補充]`：RNN $49$ 步；卷積 $\lceil 49/4\rceil=13$ 層。

**Solution to question 3** `[Added]`: RNN $49$ steps; convolution $\lceil 49/4\rceil=13$ layers.

### 小測驗（Quiz，來源原題）

來源這一段有四題自我檢測，保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The source has four self-check questions for this part, kept with all their options; the source page does not mark the answers, so the explanations below are judged from the source text.

1. 自注意力架構在長程依賴建模上優於 RNN 的核心原因是什麼？
   - (A) 因為自注意力的最大路徑長度為 $\mathcal{O}(1)$，任何兩詞皆可一步直接互連
   - (B) 因為 RNN 參數量永遠只有 1 個
   - (C) 因為 Transformer 不需要反向傳播
   - (D) 因為自注意力只能處理 3 個詞以內的句子
2. 處理長度為 $n$ 的序列時，RNN 與自注意力的循序運算次數（Sequential Operations）分別為何？
   - (A) RNN 為 $\mathcal{O}(1)$，自注意力為 $\mathcal{O}(n)$
   - (B) RNN 為 $\mathcal{O}(n)$，自注意力為 $\mathcal{O}(1)$
   - (C) 兩者皆為 $\mathcal{O}(n^2)$
   - (D) 兩者皆為 $\mathcal{O}(\log n)$
3. 自注意力機制中，計算複雜度隨序列長度 $n$ 的增長階數為何？
   - (A) 線性增長 $\mathcal{O}(n)$
   - (B) 二次方增長 $\mathcal{O}(n^2 p)$
   - (C) 對數增長 $\mathcal{O}(\log n)$
   - (D) 常數階 $\mathcal{O}(1)$
4. 在記憶體複雜度上，自注意力需要儲存什麼導致其隨長度 $n$ 呈二次方膨脹？
   - (A) 所有超參數的字串備份
   - (B) $n\times n$ 的完整注意力權重矩陣
   - (C) 使用者的硬碟目錄清單
   - (D) 純文字的英文詞典

1. What is the core reason self-attention beats the RNN at modelling long-range dependencies?
   - (A) Self-attention's maximum path length is $\mathcal{O}(1)$, so any two words connect directly in one step
   - (B) An RNN always has only one parameter
   - (C) The Transformer needs no backpropagation
   - (D) Self-attention handles only sentences of up to 3 words
2. For a sequence of length $n$, what are the sequential operations of an RNN and of self-attention?
   - (A) RNN $\mathcal{O}(1)$, self-attention $\mathcal{O}(n)$
   - (B) RNN $\mathcal{O}(n)$, self-attention $\mathcal{O}(1)$
   - (C) Both $\mathcal{O}(n^2)$
   - (D) Both $\mathcal{O}(\log n)$
3. In self-attention, at what order does the computation grow with the sequence length $n$?
   - (A) Linearly, $\mathcal{O}(n)$
   - (B) Quadratically, $\mathcal{O}(n^2 p)$
   - (C) Logarithmically, $\mathcal{O}(\log n)$
   - (D) Constant, $\mathcal{O}(1)$
4. For memory, what must self-attention store that makes it grow quadratically with $n$?
   - (A) String backups of all hyperparameters
   - (B) The full $n\times n$ attention weight matrix
   - (C) The user's hard-disk directory listing
   - (D) A plain-text English dictionary

**解析** `[補充]`：第 1 題 (A)、第 2 題 (B)、第 3 題 (B)、第 4 題 (B)。第 1 題見第2節，但「最長路徑短」是讓長距離關係容易學，不是保證學會；第 2 題見第1節；第 3、4 題見第3節，題中的 $p$ 是 $\mathbf{Q},\mathbf{K}$ 的維度，來源假設 $p\approx d$。

**Explanations** `[Added]`: 1 A, 2 B, 3 B, 4 B. Question 1, see Section 2, though a short longest path makes long-range relations easier to learn rather than guaranteeing them; question 2, see Section 1; questions 3 and 4, see Section 3, where the $p$ of the question is the dimension of $\mathbf{Q},\mathbf{K}$, which the source assumes is about $d$.

### 留給下一單元的問題（The question for the next unit）

$n^2$ 是自注意力最大的弱點：一本書、一段長程式、一整段影片，動輒幾萬到幾百萬個詞。第3節表格的最後一列給了一個方向：只看鄰近的詞。還有沒有別的辦法？能不能換個順序計算，讓 $n\times n$ 的表根本不用算出來？下一單元（本週 KP07）看稀疏注意力、線性注意力與 Performer。

$n^2$ is self-attention's greatest weakness: a book, a long program or a whole video easily runs to tens of thousands or millions of tokens. The last row of the table in Section 3 points in one direction: attend only to nearby words. Are there other ways? Could the computation be reordered so that the $n\times n$ table never has to be built? The next unit (this week's KP07) looks at sparse attention, linear attention and the Performer.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第9章「注意力機制與 Transformer 核心原理」§3.4.1–§3.4.2 與教授祕笈「BERT 與 GPT 架構典範對比」「自注意力計算複雜度分析」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §3.4.1–§3.4.2 of Chapter 9, "Core principles of attention mechanisms and the Transformer", of CCC-LNS IDL_115, and from the professor's notes "BERT versus GPT" and "Computational complexity of self-attention" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源把「教師強制的高效訓練：整個訓練序列一次性饋入」列為 Transformer 勝過 RNN 的差異之一。RNN 訓練時也用教師強制；真正的差別是 Transformer 不必沿時間一步步算隱藏狀態。本講義寫在第1節。
- 來源說最長路徑 $\mathcal{O}(1)$「徹底解決長程依賴難題」；本講義改說讓長距離關係容易學，並列為常見錯誤。
- 來源正文說總複雜度「$\approx\mathcal{O}(n^2p)$」；這只在 $n$ 比 $d$ 大時成立，本講義用祕笈的 $4n^2d+6nd^2$ 算出交叉點 $n=1.5d$。祕笈的 $6nd^2$ 只算 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 三個投影；再加上 KP03 的輸出矩陣 $\mathbf{W}_O$ 會變成 $8nd^2$，本講義沿用祕笈的算法並在此註明。
- 來源互動示範說 $n=16384$、$d=512$ 時標準注意力的運算量「高達線性架構的 9 倍」；照同一處給的公式 $(2n^2d+4nd^2)/(4nd^2)$ 算出來是 $17$ 倍，本講義不採用這個數字。
- 原論文表 1 卷積的最長路徑 $\mathcal{O}(\log_k n)$ 是擴張卷積的情形，一般卷積是 $\mathcal{O}(n/k)$；本講義在第2節說明。
- 本講義補上：$n=100$ 的循序步數、$6$ 個詞的路徑圖與路徑隨 $n$ 的折線、計算量兩項的表與圖、原論文表 1 的整理、BERT 與 GPT 在 $512$ 個詞上的訓練訊號數、三題小練習與解答。
- 取捨：規模定律（Kaplan、Chinchilla）、FlashAttention、KV 快取的複雜度、Prefix-LM、資訊論的類比，本講義不寫或只帶過。

- The source lists "efficient training with teacher forcing: the whole training sequence is fed at once" as one of the differences by which the Transformer beats the RNN. RNNs also use teacher forcing in training; the real difference is that the Transformer need not compute hidden states step by step in time. This note says so in Section 1.
- The source says an $\mathcal{O}(1)$ longest path "completely solves the long-range dependency problem"; this note says instead that it makes long-range relations easier to learn, and lists the claim as a common mistake.
- The source's main text gives the total complexity as "$\approx\mathcal{O}(n^2p)$", which holds only when $n$ exceeds $d$; this note uses the notes' $4n^2d+6nd^2$ to find the crossover $n=1.5d$. The notes' $6nd^2$ counts only the three projections $\mathbf{Q},\mathbf{K},\mathbf{V}$; adding KP03's output matrix $\mathbf{W}_O$ makes it $8nd^2$. This note keeps the notes' count and says so here.
- The source's interactive demo says that at $n=16384$, $d=512$ standard attention costs "as much as 9 times" a linear architecture; the formula given in the same place, $(2n^2d+4nd^2)/(4nd^2)$, gives $17$ times, so this note does not use that number.
- The $\mathcal{O}(\log_k n)$ longest path for convolutions in Table 1 of the original paper is for dilated convolutions; ordinary convolutions need $\mathcal{O}(n/k)$. This note explains this in Section 2.
- This note adds: the sequential steps for $n=100$, the path figure for $6$ words and the path-versus-$n$ chart, the table and figure of the two computation terms, a summary of Table 1 of the original paper, the number of training signals of BERT and GPT on $512$ words, and three practice questions with solutions.
- Choices: scaling laws (Kaplan, Chinchilla), FlashAttention, the complexity of the KV cache, prefix language models and the information-theory analogy are left out or only mentioned.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 循環神經網路（第8週） | recurrent neural network (RNN) |
| 卷積神經網路 | convolutional neural network (CNN) |
| 循序步數 | sequential steps (sequential operations) |
| 感受野 | receptive field |
| 詞元 | token |
| 最長路徑 | maximum path length |
| 大 $\mathcal{O}$ 記號 | big-$\mathcal{O}$ notation |
| 克漏字 | cloze |
| 接龍（預測下一個詞） | next-word prediction |
| 鍵值快取 | key-value cache (KV cache) |
| 百萬位元組、十億位元組 | megabyte (MB), gigabyte (GB) |
| 浮點運算次數 | floating-point operations (FLOPs) |
| 擴張卷積 | dilated convolution |
| 圖形處理器 | graphics processing unit (GPU) |
| 單精度浮點數 | float32 |
| 遮罩語言模型 | masked language model (MLM) |
| 因果語言模型 | causal language model |
| 雙向編碼器模型 | BERT (bidirectional encoder representations from Transformers) |
| 生成式預訓練模型 | GPT (generative pre-trained Transformer) |
| 文字到文字的編碼器—解碼器模型 | T5 (text-to-text transfer Transformer) |
| 規模定律 | scaling laws |
