# 第8週 KP08：雙向網路同時讀前文與後文，ELMo 把多層表示加權混合（Bidirectional networks read both earlier and later context, and ELMo mixes the layers with weights）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第8週「循環神經網路：RNN、LSTM 與 GRU」的第八個知識點（KP08），接在 KP07 的 GRU 與 MGU 之後。KP01 到 KP07 的網路都只能從左讀到右。這一節問：如果一個詞的意思要看後面的詞才確定，網路要怎麼把後文也用上？

This is the eighth knowledge point (KP08) of Week 8, "Recurrent neural networks: RNN, LSTM and GRU", and it follows the GRU and MGU of KP07. Every network from KP01 to KP07 reads only from left to right. This note asks: if the meaning of a word is settled only by the words after it, how can the network use that later context too?

先備知識是 KP01 的 RNN 前向公式，以及 KP06 的 LSTM 細胞狀態。第一次出現的詞與符號都會定義。所有數字都由 `assets/figures/src/lec08-kp08-figures.py` 算出，可以自己重算。

The prerequisites are the forward equations of the RNN in KP01 and the LSTM cell state of KP06. Every term and symbol is defined when it first appears. Every number is computed by `assets/figures/src/lec08-kp08-figures.py`, so you can recompute it yourself.

## KP08 雙向網路同時讀前文與後文，ELMo 把多層表示加權混合（Bidirectional networks read both earlier and later context, and ELMo mixes the layers with weights）

### 學習目標（Learning objectives）

讀完之後，你能說出雙向處理在什麼任務合理、在什麼任務不能用。你能寫出雙向 RNN 的前向鏈、後向鏈與輸出公式，並說出兩條鏈的權重為什麼各自獨立。

After reading, you can say for which tasks bidirectional processing is reasonable and for which it cannot be used. You can write the forward chain, the backward chain and the output equation of a bidirectional RNN, and explain why the two chains have their own weights.

你能用純量例子手算兩條鏈，說明後文怎麼改變前面位置的狀態。你也能寫出 ELMo 的加權和公式，算出一個詞的 ELMo 向量。你還能解釋同一個詞為什麼在不同句子得到不同向量。

You can compute both chains of a scalar example by hand and show how later context changes the state at an earlier position. You can also write the weighted-sum equation of ELMo, compute the ELMo vector of a word, and explain why the same word gets different vectors in different sentences.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $t$ | 時間步，第幾個元素（$t=1,2,\dots,T$） | time step, which element ($t=1,2,\dots,T$) |
| $T$ | 序列長度 | sequence length |
| $\mathbf{x}_t\in\mathbb{R}^{d}$ | 第 $t$ 步的輸入，$d$ 維 | input at step $t$, of dimension $d$ |
| $\overrightarrow{\mathbf{h}}_t\in\mathbb{R}^{p}$ | 前向鏈在第 $t$ 步的狀態，看過 $\mathbf{x}_1$ 到 $\mathbf{x}_t$ | forward-chain state at step $t$, has seen $\mathbf{x}_1$ to $\mathbf{x}_t$ |
| $\overleftarrow{\mathbf{h}}_t\in\mathbb{R}^{p}$ | 後向鏈在第 $t$ 步的狀態，看過 $\mathbf{x}_t$ 到 $\mathbf{x}_T$ | backward-chain state at step $t$, has seen $\mathbf{x}_t$ to $\mathbf{x}_T$ |
| $\overrightarrow{\mathbf{U}},\overleftarrow{\mathbf{U}}\in\mathbb{R}^{p\times d}$ | 兩條鏈各自的輸入權重 | input weights of each chain |
| $\overrightarrow{\mathbf{W}},\overleftarrow{\mathbf{W}}\in\mathbb{R}^{p\times p}$ | 兩條鏈各自的狀態傳遞權重 | state-transition weights of each chain |
| $\overrightarrow{\mathbf{V}},\overleftarrow{\mathbf{V}}\in\mathbb{R}^{q\times p}$ | 兩條鏈各自到輸出的權重 | output weights of each chain |
| $\overrightarrow{\mathbf{b}}_i,\ \overleftarrow{\mathbf{b}}_i,\ \mathbf{b}_y$ | 兩條鏈的偏置與輸出偏置 | biases of the two chains and of the output |
| $\mathbf{y}_t\in\mathbb{R}^{q}$ | 第 $t$ 步的輸出 | output at step $t$ |
| $L$ | ELMo 堆疊的 BiLSTM 層數 | number of stacked BiLSTM layers in ELMo |
| $\mathbf{h}_t^{(l)}$ | 第 $l$ 層在位置 $t$ 的雙向狀態（兩方向串接） | two-direction state of layer $l$ at position $t$ (concatenated) |
| $s_l$ | 第 $l$ 層的任務權重（純量） | task weight of layer $l$ (a scalar) |
| $\gamma$ | 整體縮放因子（純量） | global scaling factor (a scalar) |
| $\mathbf{y}_t^{\mathrm{ELMo}}$ | 位置 $t$ 的詞的 ELMo 向量 | ELMo vector of the word at position $t$ |

### 1. 只讀前文，會在哪裡卡住（Where reading only the earlier context gets stuck）

KP01 的 RNN 是**因果模型**（causal model）：第 $t$ 步的狀態只由 $\mathbf{x}_1$ 到 $\mathbf{x}_t$ 決定，看不到之後的輸入。在即時的任務裡，這是必要的限制，因為未來還沒發生。

The RNN of KP01 is a **causal model**. The state at step $t$ depends only on $\mathbf{x}_1$ to $\mathbf{x}_t$ and cannot see later inputs. In real-time tasks this is a necessary limit, because the future has not happened yet.

可是很多任務是**離線任務**（offline task）：做決定之前，整句話或整段資料都已經拿到了。例如分析一篇文章的語意、事後把一段錄音轉成文字、預測蛋白質的結構。這時只看前文，等於有資料卻不用。

But many tasks are offline. In an **offline task**, the whole sentence or the whole sequence is already available before any decision is made. Examples are analysing the meaning of an article, transcribing a recording afterwards and predicting the structure of a protein. Reading only the earlier context then means having data and not using it.

舉例 `[補充]`：「我買了蘋果，它的螢幕很亮。」讀到「蘋果」時，只看前文分不出是水果還是手機。要讀到後面的「螢幕」才確定。同一個「蘋果」換成「它很甜」，意思就是水果。

Example `[Added]`: "I bought an apple; its screen is bright." When you reach "apple", the earlier words cannot tell a fruit from a phone. Only the later word "screen" settles it. With "it is sweet" instead, the same "apple" is a fruit.

**後文能消除前面詞的歧義**，這就是**雙向處理**（bidirectional processing）的理由。

**Later context can remove the ambiguity of an earlier word**, and this is the reason for **bidirectional processing**.

類比：讀偵探小說時，第 5 頁有人把鑰匙丟進水池，你看不出他是好人還是壞人。讀到第 100 頁發現他是臥底，回頭才懂他在毀掉證據。

Analogy: in a detective novel, someone throws a key into a pond on page 5, and you cannot tell whether he is good or bad. On page 100 you learn he is an undercover agent, and looking back you see he was destroying evidence.

雙向處理就像派兩個讀者，一個從第 1 頁往後讀，一個從最後一頁往前讀。這個類比的侷限是：真的讀者讀完會「回頭重想」。雙向 RNN 的兩條鏈卻各讀一次，只在每個位置把筆記放在一起。

Bidirectional processing is like sending two readers, one reading from page 1 onward and one reading from the last page backward. The limit of the analogy: a real reader "rethinks" after finishing, while the two chains of a bidirectional RNN each read once and only put their notes together at every position.

常見錯誤：以為雙向網路到處都能用。即時語音打字是**串流任務**（streaming task）。還沒說出的話根本不存在，網路沒辦法倒著讀未來。雙向處理只適用於整段輸入都已到手的離線任務。

Common error: thinking that a bidirectional network can be used anywhere. Live speech typing is a **streaming task**. The words the speaker has not said yet do not exist, so the network cannot read the future backward. Bidirectional processing only suits offline tasks in which the whole input is already at hand.

### 2. 雙向 RNN：兩條方向相反的鏈（Bidirectional RNN: two chains running in opposite directions）

要把後文也用上，Schuster 與 Paliwal 在 1997 年提出**雙向 RNN**（bidirectional RNN，BiRNN）。它同時跑兩條 RNN：**前向鏈**（forward chain）從 $t=1$ 讀到 $t=T$，**後向鏈**（backward chain）從 $t=T$ 讀回 $t=1$。

To use later context as well, Schuster and Paliwal proposed the **bidirectional RNN** (BiRNN) in 1997. It runs two RNN chains at once: the **forward chain** reads from $t=1$ to $t=T$, and the **backward chain** reads from $t=T$ back to $t=1$.

兩條鏈的公式和 KP01 一樣，只差在「上一步」是哪一步。前向鏈用 $t-1$，後向鏈用 $t+1$：

The two chains use the same equation as KP01; they differ only in which step counts as the "previous" one. The forward chain uses $t-1$, the backward chain uses $t+1$:

$$
\begin{aligned}
\overrightarrow{\mathbf{h}}_t&=\tanh\left(\overrightarrow{\mathbf{W}}\,\overrightarrow{\mathbf{h}}_{t-1}+\overrightarrow{\mathbf{U}}\mathbf{x}_t+\overrightarrow{\mathbf{b}}_i\right),\\
\overleftarrow{\mathbf{h}}_t&=\tanh\left(\overleftarrow{\mathbf{W}}\,\overleftarrow{\mathbf{h}}_{t+1}+\overleftarrow{\mathbf{U}}\mathbf{x}_t+\overleftarrow{\mathbf{b}}_i\right).
\end{aligned}
$$

起點各自不同 `[補充]`：前向鏈從 $\overrightarrow{\mathbf{h}}_0=\mathbf{0}$ 開始，後向鏈從 $\overleftarrow{\mathbf{h}}_{T+1}=\mathbf{0}$ 開始。所以 $\overrightarrow{\mathbf{h}}_t$ 是 $\mathbf{x}_1$ 到 $\mathbf{x}_t$ 的摘要，$\overleftarrow{\mathbf{h}}_t$ 是 $\mathbf{x}_t$ 到 $\mathbf{x}_T$ 的摘要。兩份摘要都含有 $\mathbf{x}_t$ 本身。

Their starting points differ `[Added]`: the forward chain starts from $\overrightarrow{\mathbf{h}}_0=\mathbf{0}$, the backward chain from $\overleftarrow{\mathbf{h}}_{T+1}=\mathbf{0}$. So $\overrightarrow{\mathbf{h}}_t$ summarizes $\mathbf{x}_1$ to $\mathbf{x}_t$, and $\overleftarrow{\mathbf{h}}_t$ summarizes $\mathbf{x}_t$ to $\mathbf{x}_T$. Both summaries contain $\mathbf{x}_t$ itself.

每個位置的輸出把兩份摘要合起來：

The output at every position combines the two summaries:

$$
\mathbf{y}_t=\overrightarrow{\mathbf{V}}\,\overrightarrow{\mathbf{h}}_t+\overleftarrow{\mathbf{V}}\,\overleftarrow{\mathbf{h}}_t+\mathbf{b}_y.
$$

這個「各乘各的再相加」，等於先把兩個狀態**串接**（concatenation）成一個 $2p$ 維的長向量，再乘一個 $q\times 2p$ 的矩陣 `[補充]`：

This "multiply each, then add" is the same as first **concatenating** the two states into one $2p$-dimensional vector and then multiplying by a $q\times 2p$ matrix `[Added]`:

$$
\overrightarrow{\mathbf{V}}\,\overrightarrow{\mathbf{h}}_t+\overleftarrow{\mathbf{V}}\,\overleftarrow{\mathbf{h}}_t=\begin{bmatrix}\overrightarrow{\mathbf{V}} & \overleftarrow{\mathbf{V}}\end{bmatrix}\begin{bmatrix}\overrightarrow{\mathbf{h}}_t\\ \overleftarrow{\mathbf{h}}_t\end{bmatrix}.
$$

所以有的書說「串接兩個方向」，有的寫成兩項相加，兩者是同一件事。

So some books say "concatenate the two directions" and others write a sum of two terms; both describe the same thing.

兩條鏈的權重各自獨立：$\overrightarrow{\mathbf{W}}$ 和 $\overleftarrow{\mathbf{W}}$ 是兩個不同的矩陣，各自訓練。原因是「往後讀」和「往前讀」要找的線索不一樣。

The weights of the two chains are independent: $\overrightarrow{\mathbf{W}}$ and $\overleftarrow{\mathbf{W}}$ are two different matrices, each trained on its own. The reason is that "reading onward" and "reading backward" look for different clues.

代價是參數變多 `[補充]`：取 KP01 的 $d=3$、$p=4$、$q=2$，一條鏈的 $\mathbf{U}$、$\mathbf{W}$、$\mathbf{b}_i$ 共 $12+16+4=32$ 個。單向 RNN 再加 $\mathbf{V}$ 與 $\mathbf{b}_y$，共 $42$ 個。雙向是 $2\times32+2\times8+2=82$ 個。

The price is more parameters `[Added]`: with $d=3$, $p=4$, $q=2$ from KP01, one chain has $12+16+4=32$ numbers in $\mathbf{U}$, $\mathbf{W}$ and $\mathbf{b}_i$. A one-way RNN adds $\mathbf{V}$ and $\mathbf{b}_y$, $42$ in all. The bidirectional one has $2\times32+2\times8+2=82$.

常見錯誤：以為後向鏈就是「把輸入倒過來，再用同一個前向 RNN 跑一次」。後向鏈確實是倒著讀，但它有自己的一套 $\overleftarrow{\mathbf{U}}$、$\overleftarrow{\mathbf{W}}$、$\overleftarrow{\mathbf{b}}_i$，不和前向鏈共用權重。

Common error: thinking that the backward chain means "reverse the input and run the same forward RNN again". The backward chain does read in reverse, but it has its own $\overleftarrow{\mathbf{U}}$, $\overleftarrow{\mathbf{W}}$ and $\overleftarrow{\mathbf{b}}_i$ and does not share weights with the forward chain.

### 3. 手算：後文改變了前面位置的狀態（By hand: later context changes the state at an earlier position）

公式寫好了，用數字看「後文」到底怎麼進來 `[補充]`。取純量 $d=p=q=1$。前向鏈 $\overrightarrow{W}=0.5$、$\overrightarrow{U}=1.0$；後向鏈 $\overleftarrow{W}=0.6$、$\overleftarrow{U}=0.8$。

With the equations in place, use numbers to see how "later context" gets in `[Added]`. Take scalars $d=p=q=1$. The forward chain has $\overrightarrow{W}=0.5$, $\overrightarrow{U}=1.0$; the backward chain has $\overleftarrow{W}=0.6$, $\overleftarrow{U}=0.8$.

輸出 $\overrightarrow{V}=\overleftarrow{V}=1.0$，所有偏置為 $0$。準備兩個序列，只差最後一個輸入：序列 $A$ 是 $(1.0,\ 0.5,\ 1.0)$，序列 $B$ 是 $(1.0,\ 0.5,\ -1.0)$。第 $2$ 個位置的輸入都是 $x_2=0.5$，就像同一個詞「蘋果」。

The outputs use $\overrightarrow{V}=\overleftarrow{V}=1.0$, and all biases are $0$. Prepare two sequences that differ only in the last input: $A$ is $(1.0,\ 0.5,\ 1.0)$ and $B$ is $(1.0,\ 0.5,\ -1.0)$. The input at position $2$ is $x_2=0.5$ in both, like the same word "apple".

前向鏈在第 $2$ 步，兩個序列完全一樣：

At step $2$ the forward chain is exactly the same for both sequences:

$$
\begin{aligned}
\overrightarrow{h}_1&=\tanh(0.5\times0+1.0\times1.0)=\tanh(1.0)=0.7616,\\
\overrightarrow{h}_2&=\tanh(0.5\times0.7616+1.0\times0.5)=\tanh(0.3808+0.5)=\tanh(0.8808)=0.7068.
\end{aligned}
$$

後向鏈從最後一個位置開始算。序列 $A$ 的 $x_3=1.0$，序列 $B$ 的 $x_3=-1.0$：

The backward chain starts at the last position. $A$ has $x_3=1.0$ and $B$ has $x_3=-1.0$:

$$
\begin{aligned}
\text{A:}\quad\overleftarrow{h}_3&=\tanh(0.8\times1.0)=0.6640, & \overleftarrow{h}_2&=\tanh(0.6\times0.6640+0.8\times0.5)=\tanh(0.3984+0.4)=\tanh(0.7984)=0.6632,\\
\text{B:}\quad\overleftarrow{h}_3&=\tanh(0.8\times(-1.0))=-0.6640, & \overleftarrow{h}_2&=\tanh(0.6\times(-0.6640)+0.4)=\tanh(-0.3984+0.4)=\tanh(0.0016)=0.0016.
\end{aligned}
$$

第 $2$ 個位置的輸出因此不同：

So the outputs at position $2$ differ:

$$
\begin{aligned}
\text{A:}\quad y_2&=1.0\times0.7068+1.0\times0.6632=1.3700,\\
\text{B:}\quad y_2&=1.0\times0.7068+1.0\times0.0016=0.7084.
\end{aligned}
$$

同一個 $x_2=0.5$，前向狀態都是 $0.7068$，差別全部來自後向鏈帶回來的 $x_3$。序列 $A$ 和序列 $B$ 的 $y_2$ 相差 $1.3700-0.7084=0.6616$。如果只用前向鏈，第 $2$ 個位置根本分不出序列 $A$ 和序列 $B$。

The same $x_2=0.5$ has the forward state $0.7068$ in both cases; the whole difference comes from $x_3$, carried back by the backward chain. The $y_2$ of $A$ and $B$ differ by $1.3700-0.7084=0.6616$. With the forward chain alone, position $2$ could not tell $A$ from $B$ at all.

![三排方塊。中間偏上是藍色前向鏈，從左邊 h_0 = 0 往右：0.7616、0.7068、0.8749。中間偏下是橘色後向鏈，從右邊 h_4 = 0 往左：0.6640、0.6632、0.8330。最下面是輸入 1.0、0.5、1.0，各有綠色箭頭同時送進兩條鏈。最上面是輸出 1.5946、1.3700、1.5389，每個輸出收到一條藍色和一條橘色箭頭（Two chains for sequence A: the blue forward chain runs left to right with states 0.7616, 0.7068, 0.8749; the orange backward chain runs right to left with states 0.6640, 0.6632, 0.8330; every input feeds both chains, and every output y_t receives one blue and one orange arrow: 1.5946, 1.3700, 1.5389）](../assets/figures/lec08-kp08-birnn.png)

**圖說 `lec08-kp08-birnn`**〔功能：一目了然〕：先看兩條水平的鏈：藍色前向鏈的箭頭全部朝右，橘色後向鏈的箭頭全部朝左，各有自己的起點 $\overrightarrow{h}_0=0$ 與 $\overleftarrow{h}_4=0$。再看綠色箭頭：每個輸入 $x_t$ 同時送進兩條鏈。最後看黃色輸出：每個 $y_t$ 都收到一條藍色、一條橘色箭頭，例如 $y_2=0.7068+0.6632=1.3700$。

**Caption `lec08-kp08-birnn`** [Function: seeing it at a glance]: First look at the two horizontal chains: all arrows of the blue forward chain point right, all arrows of the orange backward chain point left, and each has its own start, $\overrightarrow{h}_0=0$ and $\overleftarrow{h}_4=0$. Then look at the green arrows: every input $x_t$ feeds both chains. Finally look at the yellow outputs: every $y_t$ receives one blue and one orange arrow, for example $y_2=0.7068+0.6632=1.3700$.

![左右兩組長條圖。左邊是前向狀態：t = 1、2 時 A 與 B 的長條一樣高（0.7616、0.7068），t = 3 時 A 是 +0.8749、B 是 -0.5694。右邊是後向狀態：三個位置 A 與 B 都不同，t = 1 是 0.8330 對 0.6646，t = 2 是 0.6632 對 0.0016，t = 3 是 +0.6640 對 -0.6640（Bar charts of forward and backward states for sequences A and B: the forward states agree at t = 1 and 2 and differ only at t = 3; the backward states differ at every position, e.g. 0.6632 versus 0.0016 at t = 2）](../assets/figures/lec08-kp08-later-context.png)

**圖說 `lec08-kp08-later-context`**〔功能：精準度〕：先看左邊的藍色前向狀態：$t=1,2$ 時實心（序列 $A$）和斜線（序列 $B$）一樣高，因為前向鏈還沒讀到不同的 $x_3$。再看右邊的橘色後向狀態：三個位置都不一樣高，在 $t=2$ 是 $0.6632$ 對 $0.0016$。只差一個最後的輸入，後向鏈就把差別帶回每一個較早的位置。

**Caption `lec08-kp08-later-context`** [Function: precision]: First look at the blue forward states on the left: at $t=1,2$ the solid (A) and hatched (B) bars are equally tall, because the forward chain has not yet read the different $x_3$. Then look at the orange backward states on the right: the bars differ at all three positions, $0.6632$ against $0.0016$ at $t=2$. With only the last input changed, the backward chain carries the difference back to every earlier position.

來源的算例也可以這樣讀：設某位置 $\overrightarrow{h}_2=0.60$、$\overleftarrow{h}_2=-0.40$，$\overrightarrow{V}=\overleftarrow{V}=1.0$，$b_y=0.1$。則 $y_2=1.0\times0.60+1.0\times(-0.40)+0.1=0.30$。過去的線索說「正」，未來的線索說「負」，輸出把兩邊加在一起。

The source's example reads the same way: suppose $\overrightarrow{h}_2=0.60$ and $\overleftarrow{h}_2=-0.40$ at some position, with $\overrightarrow{V}=\overleftarrow{V}=1.0$ and $b_y=0.1$. Then $y_2=1.0\times0.60+1.0\times(-0.40)+0.1=0.30$. The past clue says "positive", the future clue says "negative", and the output adds both sides.

### 4. BiLSTM：把兩條鏈換成 LSTM（BiLSTM: turning both chains into LSTM chains）

雙向 RNN 解決了「看不到後文」，卻沒有解決 KP04 的梯度消失。兩條鏈都還是普通 RNN，遠處的線索一樣傳不過來。合理的下一步，是把每條鏈的單元換成 KP06 的 LSTM。這就是**雙向 LSTM**（bidirectional LSTM，BiLSTM），由 Graves 與 Schmidhuber 在 2005 年使用。

The bidirectional RNN solves "cannot see later context", but not the vanishing gradient of KP04: both chains are still plain RNN chains, and distant clues still fail to get through. The natural next step is to replace the unit of each chain with the LSTM of KP06. This is the **bidirectional LSTM** (BiLSTM), used by Graves and Schmidhuber in 2005.

寫成式子，只是把 $\tanh$ 那一步換成 LSTM 的一步 `[補充]`。每條鏈各自帶著自己的**細胞狀態**（cell state）$\overrightarrow{\mathbf{c}}_t$ 或 $\overleftarrow{\mathbf{c}}_t$：

As an equation, the $\tanh$ step is simply replaced by one LSTM step `[Added]`. Each chain carries its own **cell state**, $\overrightarrow{\mathbf{c}}_t$ or $\overleftarrow{\mathbf{c}}_t$:

$$
\begin{aligned}
(\overrightarrow{\mathbf{h}}_t,\overrightarrow{\mathbf{c}}_t)&=\mathrm{LSTM}_{\rightarrow}\left(\overrightarrow{\mathbf{h}}_{t-1},\overrightarrow{\mathbf{c}}_{t-1},\mathbf{x}_t\right),\\
(\overleftarrow{\mathbf{h}}_t,\overleftarrow{\mathbf{c}}_t)&=\mathrm{LSTM}_{\leftarrow}\left(\overleftarrow{\mathbf{h}}_{t+1},\overleftarrow{\mathbf{c}}_{t+1},\mathbf{x}_t\right).
\end{aligned}
$$

這裡 $\mathrm{LSTM}_{\rightarrow}$ 和 $\mathrm{LSTM}_{\leftarrow}$ 各有一套門的權重，內部就是 KP06 的遺忘門、輸入門、輸出門與傳送帶。輸出照樣由 $\overrightarrow{\mathbf{h}}_t$ 和 $\overleftarrow{\mathbf{h}}_t$ 合成。BiLSTM 因此同時有兩種能力：LSTM 的長距離記憶，加上雙向的前後文。

Here $\mathrm{LSTM}_{\rightarrow}$ and $\mathrm{LSTM}_{\leftarrow}$ each have their own gate weights; inside are the forget, input and output gates and the conveyor belt of KP06. The output is still formed from $\overrightarrow{\mathbf{h}}_t$ and $\overleftarrow{\mathbf{h}}_t$. A BiLSTM therefore has two abilities at once: the long-range memory of the LSTM and the context from both sides.

在 Transformer 出現之前，BiLSTM 是自然語言處理（NLP）與語音辨識最好的架構之一，也就是當時的 SOTA（state of the art，最先進水準）。

Before the Transformer appeared, the BiLSTM was one of the best architectures for natural language processing (NLP) and speech recognition, that is, the state of the art (SOTA) of its time.

常見錯誤：以為 BiLSTM 的兩個方向共用一個細胞狀態。前向鏈的傳送帶由左往右送，後向鏈的由右往左送。兩條帶子互不相通，只在輸出時合在一起。

Common error: thinking that the two directions of a BiLSTM share one cell state. The conveyor belt of the forward chain runs left to right, that of the backward chain right to left; the two belts never meet, and only the outputs are combined.

### 5. ELMo：把每一層的雙向狀態加權混合（ELMo: mixing the two-direction states of every layer with weights）

有了 BiLSTM，下一個問題是：它算出的狀態能不能拿來當「詞的意思」？要先說明**詞嵌入**（word embedding）：把一個詞變成一串數字（一個向量），讓網路可以計算。舊的做法如 Word2Vec 給每個詞一個固定向量，叫**靜態詞向量**（static word vector）。「蘋果」不管出現在哪句話，向量都一樣。

With the BiLSTM in hand, the next question is: can the states it computes serve as "the meaning of a word"? First, a **word embedding** turns a word into a list of numbers (a vector) that a network can compute with. Older methods such as Word2Vec give each word one fixed vector, called a **static word vector**. "Apple" gets the same vector in every sentence.

**ELMo**（Embeddings from Language Models，Peters 等人 2018）改用 BiLSTM 的狀態當詞向量。BiLSTM 每個位置的狀態都讀過整句話，所以同一個詞在不同句子會得到不同向量。這叫**上下文詞嵌入**（contextual word embedding）。

**ELMo** (Embeddings from Language Models, Peters et al. 2018) uses the states of a BiLSTM as word vectors instead. The state at every position of a BiLSTM has read the whole sentence, so the same word gets different vectors in different sentences. This is called a **contextual word embedding**.

第 3 節其實已經看過這件事：同一個 $x_2=0.5$，在序列 $A$的雙向狀態是 $(0.7068,\ 0.6632)$，在序列 $B$是 $(0.7068,\ 0.0016)$。兩個向量不同，差別來自後文。

Section 3 has already shown this: the same $x_2=0.5$ has the two-direction state $(0.7068,\ 0.6632)$ in sentence $A$ and $(0.7068,\ 0.0016)$ in sentence $B$. The two vectors differ, and the difference comes from the later context.

ELMo 的 BiLSTM 先在大量文字上當**語言模型**（language model）訓練 `[補充]`。前向鏈學著猜下一個詞，後向鏈學著猜上一個詞。這樣不需要人工標註，任何文章都能拿來練。這一步叫**預訓練**（pre-training）。

The BiLSTM of ELMo is first pre-trained as a **language model** on a large amount of text `[Added]`: the forward chain learns to guess the next word, the backward chain learns to guess the previous word. This needs no human labels, so any text can be used. This step is called **pre-training**.

ELMo 把 $L$ 層 BiLSTM 疊起來（通常 $L=2$）。在第 $l$ 層、位置 $t$，先把兩個方向的狀態串接成一個向量：

ELMo stacks $L$ layers of BiLSTM (usually $L=2$). At layer $l$ and position $t$, it first concatenates the states of the two directions into one vector:

$$
\mathbf{h}_t^{(l)}=\begin{bmatrix}\overrightarrow{\mathbf{h}}_t^{(l)}\\ \overleftarrow{\mathbf{h}}_t^{(l)}\end{bmatrix}.
$$

ELMo 不只取最上面一層，而是把每一層都留下來，用任務權重加權相加：

ELMo does not take only the top layer; it keeps every layer and adds them with task weights:

$$
\mathbf{y}_t^{\mathrm{ELMo}}=\gamma\sum_{l=1}^{L}s_l\,\mathbf{h}_t^{(l)}.
$$

$s_l$ 是第 $l$ 層的權重，$\gamma$ 是整體的縮放因子，兩者都是純量。它們不是事先固定的數字，而是在下游任務訓練時，由梯度學出來的。來源的說法是：低層比較捕捉語法（例如詞性），高層比較捕捉語意（例如上下文主題）。不同任務學到不同的 $s_l$，等於自己挑要多少語法、多少語意。

$s_l$ is the weight of layer $l$ and $\gamma$ is the global scaling factor; both are scalars. They are not fixed in advance; they are learned by gradients while training on the downstream task. According to the source, lower layers capture more syntax (such as part of speech) and higher layers capture more meaning (such as the topic of the context). Different tasks learn different $s_l$, which amounts to choosing how much syntax and how much meaning to take.

用來源的數字算一次。設 $L=2$，某個詞在位置 $t$ 的兩層表示是 $\mathbf{h}_t^{(1)}=(1.0,\ 2.0)$、$\mathbf{h}_t^{(2)}=(3.0,\ 0.0)$。問答系統學到 $s_1=0.3$、$s_2=0.7$，$\gamma=1.0$：

Compute once with the source's numbers. Let $L=2$, and let the two layer representations of a word at position $t$ be $\mathbf{h}_t^{(1)}=(1.0,\ 2.0)$ and $\mathbf{h}_t^{(2)}=(3.0,\ 0.0)$. A question-answering system learns $s_1=0.3$, $s_2=0.7$ and $\gamma=1.0$:

$$
\mathbf{y}_t^{\mathrm{ELMo}}=1.0\times\left(0.3\begin{bmatrix}1.0\\ 2.0\end{bmatrix}+0.7\begin{bmatrix}3.0\\ 0.0\end{bmatrix}\right)=\begin{bmatrix}0.3\\ 0.6\end{bmatrix}+\begin{bmatrix}2.1\\ 0.0\end{bmatrix}=\begin{bmatrix}2.4\\ 0.6\end{bmatrix}.
$$

換成詞性標註任務，取來源互動展示的預設值 $s_1=0.6$、$s_2=0.4$，同樣的兩層表示得到 $(0.6\times1.0+0.4\times3.0,\ 0.6\times2.0+0.4\times0.0)=(1.8,\ 1.2)$。偏語法的任務靠向第 $1$ 層，偏語意的任務靠向第 $2$ 層。

For part-of-speech (POS) tagging, take the default of the source's interactive demo, $s_1=0.6$ and $s_2=0.4$; the same two layers give $(0.6\times1.0+0.4\times3.0,\ 0.6\times2.0+0.4\times0.0)=(1.8,\ 1.2)$. The syntax-oriented task leans toward layer $1$, the meaning-oriented task toward layer $2$.

![左邊是 ELMo 結構：最下面是位置 t 的詞，往上是第 1 層與第 2 層 BiLSTM，各寫著 h_t^(1)、h_t^(2) 等於前向與後向的串接；兩層各有一條箭頭標 s_1、s_2 送到最上面的加權和方塊。右邊是二維座標：藍點 h_t^(1) = (1.0, 2.0)，紫點 h_t^(2) = (3.0, 0.0)，兩點之間一條虛線；綠色菱形 (1.8, 1.2) 標 POS tagging，橘色菱形 (2.4, 0.6) 標 question answering，都在虛線上（ELMo structure with two stacked BiLSTM layers feeding a weighted sum; on the right, the two layer vectors (1.0, 2.0) and (3.0, 0.0) with the two weighted results (1.8, 1.2) and (2.4, 0.6) lying on the segment between them）](../assets/figures/lec08-kp08-elmo-mix.png)

**圖說 `lec08-kp08-elmo-mix`**〔功能：一目了然、精準度、具象化空間幾何〕：先看左邊：兩層 BiLSTM 都有一條箭頭送到最上面，分別乘 $s_1$ 與 $s_2$，所以每一層都被留下來。再看右邊：兩層的向量是藍點與紫點。因為 $\gamma=1$ 且 $s_1+s_2=1$，加權結果一定落在兩點之間的虛線上 `[補充]`。$s_2$ 越大，結果越靠近紫點（第 $2$ 層）。

**Caption `lec08-kp08-elmo-mix`** [Function: seeing it at a glance, precision, spatial geometry]: First look at the left: both BiLSTM layers send an arrow to the top, multiplied by $s_1$ and $s_2$, so every layer is kept. Then look at the right: the two layer vectors are the blue and purple dots. Because $\gamma=1$ and $s_1+s_2=1$, the weighted result always lies on the dashed segment between them `[Added]`. The larger $s_2$ is, the closer the result is to the purple dot (layer $2$).

$s_l$ 怎麼保證加起來是 $1$ 呢 `[補充]`？原論文先學一組沒有限制的數 $w_l$，再用第2週的 softmax 換成 $s_l$。例如 $w_1=0$、$w_2=0.8473$，softmax 之後正好是 $s_1=0.3$、$s_2=0.7$。

How are the $s_l$ kept summing to $1$ `[Added]`? The original paper learns unconstrained numbers $w_l$ and turns them into $s_l$ with the softmax of Week 2. For example, $w_1=0$ and $w_2=0.8473$ give exactly $s_1=0.3$ and $s_2=0.7$ after softmax.

常見錯誤：以為 ELMo 的層權重 $s_l$ 是人事先定好的。其實 $s_l$ 與 $\gamma$ 是在下游任務上用梯度學出來的，所以換一個任務就會換一組權重。

Common error: thinking that the layer weights $s_l$ of ELMo are set by hand in advance. In fact $s_l$ and $\gamma$ are learned by gradients on the downstream task, so a different task gets a different set of weights.

### 本單元重點整理（KP08 summary）

**一句話重點**：在整段輸入都已到手的離線任務中，雙向 RNN 用一條前向鏈與一條權重獨立的後向鏈，讓每個位置同時看到前文與後文；BiLSTM 把兩條鏈換成 LSTM；ELMo 把多層 BiLSTM 的雙向狀態以任務權重 $\gamma\sum_l s_l\mathbf{h}_t^{(l)}$ 混合，讓同一個詞在不同句子有不同向量。

**One-sentence summary**: in offline tasks where the whole input is at hand, a bidirectional RNN uses a forward chain and a backward chain with its own weights so that every position sees both earlier and later context; a BiLSTM turns both chains into LSTM chains; ELMo mixes the two-direction states of several BiLSTM layers with task weights, $\gamma\sum_l s_l\mathbf{h}_t^{(l)}$, so that the same word gets different vectors in different sentences.

**白話說明**：單向網路像只讀到一半就要回答；雙向網路像兩個人分別從頭、從尾讀，在每個位置交換筆記。ELMo 再多請幾層讀者，有的注意文法、有的注意意思，最後依任務決定要聽誰多一點。

**Plain explanation**: a one-way network is like having to answer after reading only half; a bidirectional network is like two people reading from the start and from the end and swapping notes at every position. ELMo adds more layers of readers, some watching grammar and some watching meaning, and the task decides whose voice counts more.

**正式定義**：$\overrightarrow{\mathbf{h}}_t=\tanh(\overrightarrow{\mathbf{W}}\,\overrightarrow{\mathbf{h}}_{t-1}+\overrightarrow{\mathbf{U}}\mathbf{x}_t+\overrightarrow{\mathbf{b}}_i)$，$\overleftarrow{\mathbf{h}}_t=\tanh(\overleftarrow{\mathbf{W}}\,\overleftarrow{\mathbf{h}}_{t+1}+\overleftarrow{\mathbf{U}}\mathbf{x}_t+\overleftarrow{\mathbf{b}}_i)$，$\mathbf{y}_t=\overrightarrow{\mathbf{V}}\,\overrightarrow{\mathbf{h}}_t+\overleftarrow{\mathbf{V}}\,\overleftarrow{\mathbf{h}}_t+\mathbf{b}_y$；ELMo：$\mathbf{h}_t^{(l)}=[\overrightarrow{\mathbf{h}}_t^{(l)};\overleftarrow{\mathbf{h}}_t^{(l)}]$，$\mathbf{y}_t^{\mathrm{ELMo}}=\gamma\sum_{l=1}^{L}s_l\mathbf{h}_t^{(l)}$。

**Formal definition**: $\overrightarrow{\mathbf{h}}_t=\tanh(\overrightarrow{\mathbf{W}}\,\overrightarrow{\mathbf{h}}_{t-1}+\overrightarrow{\mathbf{U}}\mathbf{x}_t+\overrightarrow{\mathbf{b}}_i)$, $\overleftarrow{\mathbf{h}}_t=\tanh(\overleftarrow{\mathbf{W}}\,\overleftarrow{\mathbf{h}}_{t+1}+\overleftarrow{\mathbf{U}}\mathbf{x}_t+\overleftarrow{\mathbf{b}}_i)$, $\mathbf{y}_t=\overrightarrow{\mathbf{V}}\,\overrightarrow{\mathbf{h}}_t+\overleftarrow{\mathbf{V}}\,\overleftarrow{\mathbf{h}}_t+\mathbf{b}_y$; ELMo: $\mathbf{h}_t^{(l)}=[\overrightarrow{\mathbf{h}}_t^{(l)};\overleftarrow{\mathbf{h}}_t^{(l)}]$, $\mathbf{y}_t^{\mathrm{ELMo}}=\gamma\sum_{l=1}^{L}s_l\mathbf{h}_t^{(l)}$.

**例子**：序列 $A=(1.0,0.5,1.0)$ 與 $B=(1.0,0.5,-1.0)$ 在第 $2$ 個位置的前向狀態都是 $0.7068$。後向狀態是 $0.6632$ 對 $0.0016$，$y_2$ 是 $1.3700$ 對 $0.7084$。ELMo：$(1.0,2.0)$ 與 $(3.0,0.0)$ 以 $s=(0.3,0.7)$ 混合得 $(2.4,0.6)$，以 $s=(0.6,0.4)$ 得 $(1.8,1.2)$。

**Example**: for $A=(1.0,0.5,1.0)$ and $B=(1.0,0.5,-1.0)$, the forward state at position $2$ is $0.7068$ in both, the backward state is $0.6632$ against $0.0016$, and $y_2$ is $1.3700$ against $0.7084$. ELMo: mixing $(1.0,2.0)$ and $(3.0,0.0)$ with $s=(0.3,0.7)$ gives $(2.4,0.6)$, and with $s=(0.6,0.4)$ gives $(1.8,1.2)$.

**常見迷思**：

- 「雙向網路到處都能用。」錯，串流任務看不到未來，只有離線任務能用。
- 「後向鏈就是把輸入倒過來，再跑同一個前向 RNN。」錯，後向鏈有自己的一套權重。
- 「BiLSTM 的兩個方向共用細胞狀態。」錯，兩條傳送帶各走各的，只在輸出合併。
- 「ELMo 只用最上面一層，層權重是人定的。」錯，每一層都用上，$s_l$ 與 $\gamma$ 由任務學出。

**Common misconceptions**:

- "A bidirectional network can be used anywhere." Wrong: streaming tasks cannot see the future; only offline tasks can use it.
- "The backward chain reverses the input and runs the same forward RNN." Wrong: the backward chain has its own weights.
- "The two directions of a BiLSTM share a cell state." Wrong: the two belts run separately and meet only at the output.
- "ELMo uses only the top layer, and the layer weights are set by hand." Wrong: every layer is used, and $s_l$ and $\gamma$ are learned from the task.

**圖**：本單元有三張圖。

- `lec08-kp08-birnn`：兩條方向相反、權重獨立的鏈，在每個位置合成輸出。
- `lec08-kp08-later-context`：只改最後一個輸入，前向狀態在前兩步不變，後向狀態每一步都變。
- `lec08-kp08-elmo-mix`：ELMo 保留每一層，加權結果落在兩層向量之間。

**Figures** — this unit has three figures.

- `lec08-kp08-birnn`: two chains running in opposite directions with their own weights, combined into an output at every position.
- `lec08-kp08-later-context`: when only the last input changes, the forward states stay the same for the first two steps, while the backward states change at every step.
- `lec08-kp08-elmo-mix`: ELMo keeps every layer, and the weighted result lies between the two layer vectors.

### 小練習（Practice）

1. 設 $d=5$、$p=8$、$q=3$。算出單向 RNN 與雙向 RNN 的參數總數（含偏置）。提示：一條鏈有 $\mathbf{U}$、$\mathbf{W}$、$\mathbf{b}_i$；雙向有兩條鏈、兩個 $\mathbf{V}$，但只有一個 $\mathbf{b}_y$。
2. 沿用第 5 節的兩層表示 $\mathbf{h}_t^{(1)}=(1.0,\ 2.0)$、$\mathbf{h}_t^{(2)}=(3.0,\ 0.0)$，取 $s_1=s_2=0.5$、$\gamma=2.0$。算出 $\mathbf{y}_t^{\mathrm{ELMo}}$。提示：先算括號內的加權和，再乘 $\gamma$。
3. 即時字幕（邊說邊出字）可以直接用 BiLSTM 嗎？如果可以晚一點出字，例如等說完一整句再出，又如何？提示：想想後向鏈第一步需要哪個輸入。

1. Let $d=5$, $p=8$, $q=3$. Compute the total number of parameters, biases included, of a one-way RNN and of a bidirectional RNN. Hint: one chain has $\mathbf{U}$, $\mathbf{W}$, $\mathbf{b}_i$; the bidirectional one has two chains and two $\mathbf{V}$, but only one $\mathbf{b}_y$.
2. Keep the two layer representations of Section 5, $\mathbf{h}_t^{(1)}=(1.0,\ 2.0)$ and $\mathbf{h}_t^{(2)}=(3.0,\ 0.0)$, and take $s_1=s_2=0.5$, $\gamma=2.0$. Compute $\mathbf{y}_t^{\mathrm{ELMo}}$. Hint: compute the weighted sum in the brackets first, then multiply by $\gamma$.
3. Can live captions (text appearing while someone speaks) use a BiLSTM directly? What if the captions may come a little later, for example after each whole sentence is spoken? Hint: think about which input the first step of the backward chain needs.

**第 1 題解答** `[補充]`：一條鏈是 $8\times5+8\times8+8=112$ 個。單向 RNN 再加 $\mathbf{V}$ 的 $3\times8=24$ 個與 $\mathbf{b}_y$ 的 $3$ 個，共 $139$ 個（和 KP01 第 1 題相同）。雙向是 $2\times112+2\times24+3=275$ 個，大約是單向的兩倍。

**Solution to question 1** `[Added]`: one chain has $8\times5+8\times8+8=112$. The one-way RNN adds $3\times8=24$ for $\mathbf{V}$ and $3$ for $\mathbf{b}_y$, $139$ in all (the same as question 1 of KP01). The bidirectional one has $2\times112+2\times24+3=275$, about twice as many.

**第 2 題解答** `[補充]`：括號內是 $0.5\times(1.0,\ 2.0)+0.5\times(3.0,\ 0.0)=(2.0,\ 1.0)$，乘 $\gamma=2.0$ 得 $(4.0,\ 2.0)$。$\gamma$ 只改變整體大小，不改變方向。

**Solution to question 2** `[Added]`: the brackets give $0.5\times(1.0,\ 2.0)+0.5\times(3.0,\ 0.0)=(2.0,\ 1.0)$, and multiplying by $\gamma=2.0$ gives $(4.0,\ 2.0)$. $\gamma$ changes only the overall size, not the direction.

**第 3 題解答** `[補充]`：邊說邊出字不能直接用。後向鏈要從最後一個輸入開始算，而那個輸入還沒出現。如果等一整句說完再出字，這一句就變成離線資料，可以用 BiLSTM；代價是字幕會延遲一句話的時間。

**Solution to question 3** `[Added]`: text that appears while speaking cannot use it directly, because the backward chain must start from the last input, which does not exist yet. If each caption waits until the whole sentence is spoken, that sentence becomes offline data and a BiLSTM can be used; the price is that the captions lag by one sentence.

### 小測驗（Quiz，來源原題）

來源的兩題自我檢測保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The two self-check questions of the source are kept with all options; the source page does not mark the answers directly, and the explanations below follow the source text.

1. 雙向 RNN（BiRNN）之所以能在離線序列任務中表現優異，關鍵原因是什麼？
   - (A) 它能同時利用當前時間步 $t$ 之前（過去）與之後（未來）的雙向上下文資訊
   - (B) 它的訓練速度快 $10$ 倍
   - (C) 它不需要使用激活函數
   - (D) 它消除了所有矩陣乘法
2. ELMo 模型如何利用多層 BiLSTM 的隱藏狀態產生特定單詞的上下文詞嵌入向量？
   - (A) 只取最頂層的隱藏狀態
   - (B) 將所有層前向與後向隱藏狀態拼接後，經由下游任務可學習的純量權重 $s_l$ 進行線性組合與縮放 $\gamma$
   - (C) 隨機選擇一層
   - (D) 對所有層取最大值（Max pooling）

1. What is the key reason that a bidirectional RNN (BiRNN) performs well on offline sequence tasks?
   - (A) It can use context from both before (the past) and after (the future) the current time step $t$
   - (B) It trains $10$ times faster
   - (C) It needs no activation function
   - (D) It removes all matrix multiplications
2. How does ELMo use the hidden states of a multi-layer BiLSTM to produce the contextual embedding of a particular word?
   - (A) It takes only the hidden state of the top layer
   - (B) It concatenates the forward and backward hidden states of every layer, combines them linearly with scalar weights $s_l$ learned on the downstream task, and scales by $\gamma$
   - (C) It picks one layer at random
   - (D) It takes the maximum over all layers (max pooling)

**解析** `[補充]`：第 1 題選 (A)：第 2 節的前向鏈帶來 $\mathbf{x}_1$ 到 $\mathbf{x}_t$，後向鏈帶來 $\mathbf{x}_t$ 到 $\mathbf{x}_T$；(B) 錯，兩條鏈的計算量約是單向的兩倍；(C) 錯，兩條鏈都用 $\tanh$；(D) 錯，每條鏈都要乘 $\mathbf{U}$、$\mathbf{W}$、$\mathbf{V}$。第 2 題選 (B)：第 5 節先串接兩方向，再用 $\gamma\sum_l s_l\mathbf{h}_t^{(l)}$ 混合；(A)(C)(D) 都只用到一層或不用學習的規則，丟掉了「依任務決定每層比重」這一點。

**Explanations** `[Added]`: question 1 is A: in Section 2 the forward chain brings $\mathbf{x}_1$ to $\mathbf{x}_t$ and the backward chain brings $\mathbf{x}_t$ to $\mathbf{x}_T$; B is wrong, since two chains cost about twice the computation of one; C is wrong, since both chains use $\tanh$; D is wrong, since every chain multiplies by $\mathbf{U}$, $\mathbf{W}$ and $\mathbf{V}$. Question 2 is B: Section 5 first concatenates the two directions and then mixes with $\gamma\sum_l s_l\mathbf{h}_t^{(l)}$; A, C and D use only one layer or a rule that is not learned, and lose the point of "letting the task decide the share of each layer".

### 本週回顧（Week recap）

這一週從 KP01 的 RNN 開始：用隱藏狀態與共享權重記住過去。KP02、KP03 用 BPTT 把誤差沿時間倒傳，算出共享權重的梯度。KP04 發現梯度要連乘很多次，會消失或爆炸；KP05 試了單位矩陣、長延遲、滲漏單元與 ESN 等早期解法。

This week started with the RNN of KP01: a hidden state and shared weights remember the past. KP02 and KP03 sent the error back through time with BPTT and computed the gradients of the shared weights. KP04 found that the gradient is multiplied many times and can vanish or explode; KP05 tried early remedies such as the identity matrix, long delays, leaky units and the ESN.

KP06 的 LSTM 用門與加法傳送帶保住長期記憶，KP07 再精簡成 GRU 與 MGU。最後 KP08 讓網路同時讀前文與後文。ELMo 再把多層 BiLSTM 變成會隨句子改變的詞向量。

The LSTM of KP06 kept long-term memory with gates and an additive conveyor belt, and KP07 slimmed it down to the GRU and the MGU. Finally KP08 let the network read both earlier and later context, and ELMo turned a multi-layer BiLSTM into word vectors that change with the sentence.

### 留給下一單元的問題（The question for the next unit）

BiLSTM 看得到前後文，但它仍要一步一步算：第 $t$ 步要等第 $t-1$ 步算完，沒辦法平行計算。句子很長時，第 $1$ 個詞的資訊也要經過很多步才傳到最後。能不能不用循環，讓任意兩個位置直接連線、互相查看？後面一週的注意力機制（attention）與 Transformer 回答這個問題。

A BiLSTM sees both sides, but it still computes step by step: step $t$ must wait for step $t-1$, so it cannot be parallelized. In a long sentence, the information of the first word must also pass through many steps to reach the end. Without recurrence, can any two positions be connected directly and look at each other? The attention mechanism and the Transformer in a later week answer this question.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第8章「循環神經網絡與長短期記憶網絡」§5「雙向處理的合理性與雙向 RNN」「雙向 LSTM 與 ELMo 上下文詞嵌入」兩節與其教授祕笈（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch08-rnn-lstm/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from the two sections of §5 of Chapter 8, "Recurrent neural networks and long short-term memory networks", of CCC-LNS IDL_115, "Why bidirectional processing is reasonable and the bidirectional RNN" and "The bidirectional LSTM and ELMo contextual word embeddings", with their professor's notes (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch08-rnn-lstm/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源的 ELMo 公式寫成 $\gamma\sum_{l=1}^{L}s_l\mathbf{h}_t^{(l)}$。Peters 等人 2018 的原式從 $l=0$ 加起，第 $0$ 層是不看上下文的詞本身的表示；$s_l$ 由 softmax 正規化。本講義正文保留來源的形式，並以 `[補充]` 說明 softmax；第 $0$ 層的差別只記在這裡。
- 來源用 `\boldsymbol` 表示向量；本課程網站不支援，改用 `\mathbf`，意義相同。
- 來源的輸出公式寫成兩項相加，生圖提示卻說「串接」；本講義補上兩者相等的式子，不是錯誤，只是說法不同。
- 來源用「警察正在追捕……」說明後文有用，但要猜的是後面那個詞本身，不是「後文改變前面的詞」。本講義改用「蘋果／螢幕」的例子 `[補充]`，並沿用來源的偵探小說類比。
- 本講義補上：純量 BiRNN 算例（序列 $A$、序列 $B$ 與 $y_2$ 的差 $0.6616$）、兩條鏈的起點、雙向參數個數（$42$ 對 $82$）、BiLSTM 的式子、語言模型預訓練的說明、ELMo 的幾何解讀與 softmax 數例、本週回顧、三題小練習與解答。
- 取捨：來源的兩題自我檢測保留原題；來源的 $y_2=0.30$ 算例與 ELMo 的 $(2.4,\ 0.6)$ 算例保留原數字；來源兩個互動展示無法放進講義，改用 `lec08-kp08-later-context` 與 `lec08-kp08-elmo-mix` 呈現。來源把這兩節編為 KP-15、KP-16，本課程合成 KP08。

- The source writes the ELMo formula as $\gamma\sum_{l=1}^{L}s_l\mathbf{h}_t^{(l)}$. The original formula of Peters et al. 2018 sums from $l=0$, where layer $0$ is the context-free representation of the word itself, and the $s_l$ are normalized by softmax. The main text keeps the source's form and explains the softmax with `[Added]`; the layer-$0$ difference is recorded only here.
- The source writes vectors with `\boldsymbol`; the course website does not support it, so this note uses `\mathbf` with the same meaning.
- The source writes the output as a sum of two terms, while its image prompt says "concatenation"; this note adds the equation showing they are equal. It is not an error, only two ways of saying the same thing.
- The source uses "the police are chasing ..." to show that later context helps, but there the word to guess is the later word itself, not "later context changing an earlier word". This note uses the "apple / screen" example instead `[Added]` and keeps the source's detective-novel analogy.
- This note adds: the scalar BiRNN example (sequences $A$ and $B$, and the $y_2$ difference of $0.6616$), the starting points of the two chains, the bidirectional parameter count ($42$ against $82$), the BiLSTM equation, the explanation of language-model pre-training, the geometric reading of ELMo and the softmax example, the week recap, and three practice questions with solutions.
- Choices: the two self-check questions of the source are kept as asked; the source's $y_2=0.30$ example and the ELMo example $(2.4,\ 0.6)$ keep their numbers; the two interactive demos of the source cannot go into a note, so `lec08-kp08-later-context` and `lec08-kp08-elmo-mix` show the same ideas. The source numbers these two sections KP-15 and KP-16; this course merges them into KP08.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 因果模型 | causal model |
| 離線任務 | offline task |
| 串流任務 | streaming task |
| 雙向處理 | bidirectional processing |
| 雙向循環神經網路 | bidirectional RNN (BiRNN) |
| 循環神經網路 | recurrent neural network (RNN) |
| 前向鏈 | forward chain |
| 後向鏈 | backward chain |
| 串接 | concatenation |
| 雙向長短期記憶網路 | bidirectional LSTM (BiLSTM) |
| 長短期記憶網路 | long short-term memory (LSTM) |
| 細胞狀態 | cell state |
| 自然語言處理 | natural language processing (NLP) |
| 最先進水準 | state of the art (SOTA) |
| 詞嵌入 | word embedding |
| 靜態詞向量 | static word vector |
| 上下文詞嵌入 | contextual word embedding |
| 來自語言模型的詞嵌入 | Embeddings from Language Models (ELMo) |
| 語言模型 | language model |
| 預訓練 | pre-training |
| 任務權重 | task weight |
| 縮放因子 | scaling factor |
| 詞性標註 | part-of-speech (POS) tagging |
| 問答系統 | question answering |
| 注意力機制 | attention mechanism |
| 變換器（後面一週） | Transformer |
| 隨時間反向傳播（KP02、KP03） | backpropagation through time (BPTT) |
| 迴聲狀態網路（KP05） | echo state network (ESN) |
| 門控循環單元（KP07） | gated recurrent unit (GRU) |
| 極簡門控單元（KP07） | minimal gated unit (MGU) |
| 詞向量方法（舊式靜態詞向量） | Word2Vec |
| 本課程的網頁教材 | CCC-LNS |
