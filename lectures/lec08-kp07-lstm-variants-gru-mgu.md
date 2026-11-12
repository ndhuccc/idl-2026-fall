# 第8週 KP07：LSTM 的演進與精簡，門越少參數越少（LSTM variants, GRU and MGU: fewer gates, fewer parameters）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第8週「循環神經網路：RNN、LSTM 與 GRU」的第七個知識點（KP07），接在 KP06 的 LSTM 三道門與細胞狀態之後。KP06 留下一個問題：1997 年最早的 LSTM 其實沒有遺忘門。沒有遺忘門，長序列會出什麼事？後來的人又怎麼把 LSTM 一路精簡成更輕的 GRU 與 MGU？

This is the seventh knowledge point (KP07) of Week 8, "Recurrent neural networks: RNN, LSTM and GRU". It follows KP06, the three gates and the cell state of the LSTM. KP06 left a question: the first LSTM of 1997 had no forget gate. What goes wrong on long sequences without it? And how was the LSTM later slimmed down into the lighter GRU and MGU?

先備知識是 KP01 的 RNN 前向公式與參數個數，以及 KP06 的 LSTM 公式。第一次出現的詞與符號都會定義。所有數字都由 `assets/figures/src/lec08-kp07-figures.py` 算出，可以自己重算。

The prerequisites are the forward equations and the parameter count of the RNN in KP01, and the LSTM equations of KP06. Every term and symbol is defined when it first appears. Every number is computed by `assets/figures/src/lec08-kp07-figures.py`, so you can recompute it yourself.

## KP07 LSTM 的演進與精簡，門越少參數越少（LSTM variants, GRU and MGU: fewer gates, fewer parameters）

### 學習目標（Learning objectives）

讀完之後，你能說出 LSTM 從 1997 到 2005 的四個階段，各補上了什麼，並能用數字說明沒有遺忘門時，細胞狀態為什麼會一直變大。你能寫出 GRU 的四條公式，並手算一步純量 GRU，也能說出 MGU 怎麼用一道門取代 GRU 的兩道門。你也能算出 LSTM、GRU、MGU 的參數個數，並說出「省 25%」「省 50%」在什麼條件下成立。

After reading, you can name the four stages of the LSTM from 1997 to 2005 and what each one added. You can show with numbers why the cell state keeps growing without a forget gate. You can write the four equations of the GRU and compute one step of a scalar GRU by hand. You can explain how the MGU replaces the two gates of the GRU with one gate. You can also count the parameters of the LSTM, the GRU and the MGU, and state when "saves 25%" and "saves 50%" hold.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{x}_t\in\mathbb{R}^{d}$ | 第 $t$ 步的輸入，$d$ 維 | input at step $t$, of dimension $d$ |
| $\mathbf{h}_t\in\mathbb{R}^{p}$ | 第 $t$ 步的隱藏狀態，$p$ 維 | hidden state at step $t$, of dimension $p$ |
| $\mathbf{c}_t\in\mathbb{R}^{p}$ | LSTM 的細胞狀態（KP06） | cell state of the LSTM (KP06) |
| $\tilde{\mathbf{c}}_t$ | LSTM 的候選記憶（KP06） | candidate memory of the LSTM (KP06) |
| $\mathbf{f}_t,\ \mathbf{i}_t,\ \mathbf{o}_t$ | LSTM 的遺忘門、輸入門、輸出門（KP06）；這裡的 $\mathbf{i}_t$ 不是 KP01 的淨輸入 | forget, input and output gates of the LSTM (KP06); here $\mathbf{i}_t$ is not the net input of KP01 |
| $\mathbf{r}_t,\ \mathbf{z}_t$ | GRU 的重置門、更新門 | reset gate and update gate of the GRU |
| $\tilde{\mathbf{h}}_t$ | GRU 與 MGU 的候選狀態 | candidate state of the GRU and the MGU |
| $\mathbf{W}_\ast\in\mathbb{R}^{p\times p},\ \mathbf{U}_\ast\in\mathbb{R}^{p\times d},\ \mathbf{b}_\ast\in\mathbb{R}^{p}$ | 第 $\ast$ 組的狀態權重、輸入權重、偏置（$\ast$ 是 $f,i,o,c,r,z$） | state weights, input weights and bias of group $\ast$ ($\ast$ is $f,i,o,c,r,z$) |
| $\mathbf{p}_f,\ \mathbf{p}_i,\ \mathbf{p}_o\in\mathbb{R}^{p}$ | 窺視孔權重 | peephole weights |
| $\sigma(a)=\dfrac{1}{1+e^{-a}}$ | sigmoid 函數，值在 $(0,1)$；來源寫成 $\operatorname{sig}$ | sigmoid function, with values in $(0,1)$; the source writes $\operatorname{sig}$ |
| $\odot$ | 逐元素相乘 | element-wise product |
| $\mathbf{1}$ | 每個分量都是 $1$ 的向量 | vector whose entries are all $1$ |

### 1. 1997 年的 LSTM：沒有遺忘門，記憶只進不出（The LSTM of 1997: no forget gate, so memory only grows）

先回答 KP06 留下的問題。Hochreiter 與 Schmidhuber 在 1997 年提出的**原始 LSTM**（original LSTM），只有輸入門和輸出門。它沒有遺忘門，也沒有後面會講的窺視孔。

First, answer the question left by KP06. The **original LSTM** of Hochreiter and Schmidhuber (1997) had only an input gate and an output gate. It had no forget gate, and no peepholes either, which come later in this note.

沒有遺忘門，等於 KP06 的細胞更新式裡 $\mathbf{f}_t$ 永遠是 $\mathbf{1}$：

Having no forget gate means that $\mathbf{f}_t$ in the cell update of KP06 is always $\mathbf{1}$:

$$
\mathbf{c}_t=\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t.
$$

舊記憶原封不動留下，新記憶只能往上加。這正是 KP06 說的常數誤差流：梯度沿著 $\mathbf{c}$ 不會衰減。好處是記得久；壞處是永遠清不掉。

The old memory stays untouched, and new memory can only be added on top. This is the constant error carousel of KP06: the gradient does not decay along $\mathbf{c}$. The benefit is a long memory; the cost is that it can never be cleared.

用來源的數字看。設一個純量細胞，每一步寫入的量都是 $i_t\tilde{c}_t=0.5$，序列長 $50$ 步，$c_0=0$。沒有遺忘門時，每步加 $0.5$：第 $10$ 步 $c_{10}=5.0$，第 $50$ 步 $c_{50}=25.0$。

Look at the source's numbers. Take a scalar cell that writes $i_t\tilde{c}_t=0.5$ at every step, over $50$ steps, with $c_0=0$. Without a forget gate it adds $0.5$ per step: $c_{10}=5.0$ at step $10$ and $c_{50}=25.0$ at step $50$.

這有什麼不好？輸出要經過 $h_t=o_t\tanh(c_t)$。$\tanh(25.0)$ 四捨五入是 $1.0000$，已經貼死在上限。這叫**飽和**（saturation）：$c_t$ 再怎麼變，$\tanh(c_t)$ 都不動了。

Why is that bad? The output goes through $h_t=o_t\tanh(c_t)$. $\tanh(25.0)$ rounds to $1.0000$, stuck at the upper limit. This is called **saturation**: however $c_t$ changes, $\tanh(c_t)$ no longer moves.

「動不動」可以用 $\tanh$ 的斜率量：$\tanh'(c)=1-\tanh^2(c)$ `[補充]`。在 $c=25.0$，斜率只有 $7.7\times10^{-22}$，梯度幾乎傳不過這一關。

"Whether it moves" can be measured by the slope of $\tanh$: $\tanh'(c)=1-\tanh^2(c)$ `[Added]`. At $c=25.0$ the slope is only $7.7\times10^{-22}$, so almost no gradient passes this point.

加上遺忘門，設 $f_t=0.7$，更新變成 $c_t=0.7\,c_{t-1}+0.5$。每一步先丟掉三成舊記憶，再加新的。$c_t$ 會停在一個**極限**：令 $c=0.7c+0.5$，解得

Add a forget gate with $f_t=0.7$; the update becomes $c_t=0.7\,c_{t-1}+0.5$. Every step first drops 30% of the old memory and then adds the new. $c_t$ settles at a **limit**: set $c=0.7c+0.5$ and solve,

$$
c_\infty=\frac{0.5}{1-0.7}=1.6667.
$$

程式算出第 $10$ 步已經是 $c_{10}=1.6196$，第 $50$ 步 $c_{50}=1.6667$。此時 $\tanh(c_{50})=0.9311$，斜率 $0.1330$，梯度還能流過。

The program gives $c_{10}=1.6196$ at step $10$ already, and $c_{50}=1.6667$ at step $50$. Then $\tanh(c_{50})=0.9311$ and the slope is $0.1330$, so the gradient can still flow.

![兩張圖。左圖是細胞狀態 c_t 對時間步 t：紅線沒有遺忘門，直直往上到 c_50 = 25.0；藍線有遺忘門 f_t = 0.7，很快平在 1.6667 的虛線。右圖是 tanh 在 c_t 的斜率，縱軸對數：紅線一路下降到 7.71×10^-22，藍線停在 0.1330（Left: the cell state without a forget gate climbs linearly to 25.0, while with f_t = 0.7 it levels off at 1.6667. Right: on a log scale, the slope of tanh at c_t falls to 7.71e-22 without the forget gate but stays at 0.1330 with it）](../assets/figures/lec08-kp07-cell-growth.png)

**圖說 `lec08-kp07-cell-growth`**〔功能：精準度〕：先看左圖：紅線沒有遺忘門，每步加 $0.5$，是一條直線；藍線有 $f_t=0.7$，約 $10$ 步內就貼近 $1.6667$ 的虛線。再看右圖的縱軸是對數刻度：紅線的斜率每過幾步就小一個數量級，到第 $50$ 步只剩 $7.71\times10^{-22}$；藍線停在 $0.1330$。左圖的「越積越多」，就是右圖的「梯度越來越過不去」。

**Caption `lec08-kp07-cell-growth`** [Function: precision]: First look at the left panel: the red line, without a forget gate, adds $0.5$ per step and is straight; the blue line with $f_t=0.7$ reaches the dashed line at $1.6667$ within about $10$ steps. Then note the log scale on the right: the red slope loses an order of magnitude every few steps, down to $7.71\times10^{-22}$ at step $50$; the blue slope stays at $0.1330$. "Piling up" on the left is "the gradient getting stuck" on the right.

常見錯誤：以為「沒有遺忘門，記憶最完整，所以最好」。記得太多也是問題：狀態一直變大會讓 $\tanh$ 飽和。而且句子換了話題，舊內容也該能清掉。

Common error: thinking that "no forget gate keeps the most complete memory, so it is best". Remembering too much is also a problem: a growing state saturates $\tanh$. And when the text changes topic, old content should be clearable.

### 2. 2000 到 2005 年：遺忘門、窺視孔與香草 LSTM（2000 to 2005: forget gate, peepholes and the vanilla LSTM）

第 1 節的問題，在 2000 年由 Gers 等人解決。他們加上**遺忘門**（forget gate）$\mathbf{f}_t$，讓網路能主動清空舊狀態。同一時期，他們又加上**窺視孔**（peephole connections）：讓門直接「看一眼」細胞狀態。

The problem of Section 1 was solved by Gers and colleagues in 2000. They added the **forget gate** $\mathbf{f}_t$, so the network can actively clear the old state. In the same period they also added **peephole connections**: the gates take a direct "peek" at the cell state.

沒有窺視孔時，門只看 $\mathbf{h}_{t-1}$ 與 $\mathbf{x}_t$。但 $\mathbf{h}_{t-1}=\mathbf{o}_{t-1}\odot\tanh(\mathbf{c}_{t-1})$，輸出門關著時，門就看不到細胞裡有什麼。窺視孔在門的淨輸入裡多加一項，例如遺忘門變成

Without peepholes, a gate sees only $\mathbf{h}_{t-1}$ and $\mathbf{x}_t$. But $\mathbf{h}_{t-1}=\mathbf{o}_{t-1}\odot\tanh(\mathbf{c}_{t-1})$, so when the output gate is closed the gates cannot see what is in the cell. A peephole adds one more term to the net input of a gate; the forget gate, for example, becomes

$$
\mathbf{f}_t=\sigma\big(\mathbf{W}_f\mathbf{h}_{t-1}+\mathbf{U}_f\mathbf{x}_t+\mathbf{p}_f\odot\mathbf{c}_{t-1}+\mathbf{b}_f\big).
$$

輸入門同樣看 $\mathbf{c}_{t-1}$；輸出門在 $\mathbf{c}_t$ 算好之後才算，所以看的是 $\mathbf{c}_t$。窺視孔權重 $\mathbf{p}$ 是向量，用逐元素相乘，所以三道門只多 $3p$ 個參數。

The input gate also looks at $\mathbf{c}_{t-1}$; the output gate is computed after $\mathbf{c}_t$, so it looks at $\mathbf{c}_t$. The peephole weights $\mathbf{p}$ are vectors used element-wise, so the three gates add only $3p$ parameters.

2005 年 Graves 與 Schmidhuber 把遺忘門、窺視孔和完整的 BPTT 訓練放在一起，後來被稱為**香草 LSTM**（vanilla LSTM），「香草」是「標準款」的意思。它的三道門（不含窺視孔的寫法）是

In 2005 Graves and Schmidhuber combined the forget gate, peepholes and full BPTT training; this was later called the **vanilla LSTM**, where "vanilla" means "the standard flavor". Its three gates (written without peepholes) are

$$
\begin{aligned}
\mathbf{i}_t&=\sigma(\mathbf{W}_i\mathbf{h}_{t-1}+\mathbf{U}_i\mathbf{x}_t+\mathbf{b}_i),\\
\mathbf{f}_t&=\sigma(\mathbf{W}_f\mathbf{h}_{t-1}+\mathbf{U}_f\mathbf{x}_t+\mathbf{b}_f),\\
\mathbf{o}_t&=\sigma(\mathbf{W}_o\mathbf{h}_{t-1}+\mathbf{U}_o\mathbf{x}_t+\mathbf{b}_o).
\end{aligned}
$$

再加上 KP06 的候選記憶 $\tilde{\mathbf{c}}_t=\tanh(\mathbf{W}_c\mathbf{h}_{t-1}+\mathbf{U}_c\mathbf{x}_t+\mathbf{b}_c)$，一共是四組「$\mathbf{W}$、$\mathbf{U}$、$\mathbf{b}$」。第 6 節數參數時會用到這個「四組」。

Add the candidate memory of KP06, $\tilde{\mathbf{c}}_t=\tanh(\mathbf{W}_c\mathbf{h}_{t-1}+\mathbf{U}_c\mathbf{x}_t+\mathbf{b}_c)$, and there are four groups of "$\mathbf{W}$, $\mathbf{U}$, $\mathbf{b}$". Section 6 uses this "four groups" to count parameters.

來源還提到兩個變體。**全門控循環**（full gate recurrence）讓每道門也接收上一步各道門的值。它多出很多 $p\times p$ 矩陣，計算變慢、也容易過擬合，後來少有人用。**投影 LSTM**（LSTM with projection）在輸出 $\mathbf{h}_t$ 前加一個線性投影層，把 $\mathbf{h}_t$ 變短，用來減少參數。

The source mentions two more variants. **Full gate recurrence** lets every gate also receive the previous values of all gates. It adds many $p\times p$ matrices, is slower and overfits easily, so it is rarely used. The **LSTM with projection** adds a linear projection layer before the output $\mathbf{h}_t$, making $\mathbf{h}_t$ shorter to reduce parameters.

| 年份 | 名稱 | 新增了什麼 |
|---|---|---|
| $1997$ | 原始 LSTM（Hochreiter 與 Schmidhuber） | 輸入門、輸出門、細胞狀態；沒有遺忘門 |
| $2000$ | Gers 等人 | 遺忘門 $\mathbf{f}_t$；窺視孔 |
| $2005$ | 香草 LSTM（Graves 與 Schmidhuber） | 遺忘門＋窺視孔＋完整 BPTT，成為標準款 |
| 之後 | 全門控循環、投影 LSTM | 門之間互連；輸出前加投影層 |

| Year | Name | What was added |
|---|---|---|
| $1997$ | Original LSTM (Hochreiter and Schmidhuber) | input gate, output gate, cell state; no forget gate |
| $2000$ | Gers et al. | forget gate $\mathbf{f}_t$; peepholes |
| $2005$ | Vanilla LSTM (Graves and Schmidhuber) | forget gate + peepholes + full BPTT, the standard version |
| later | Full gate recurrence, LSTM with projection | gates connected to each other; a projection layer before the output |

小知識 `[補充]`：PyTorch 與 TensorFlow 預設的 LSTM 沒有窺視孔，用的就是上面三條公式。來源的教授祕笈也提到原因：Greff 等人的大規模實驗發現，窺視孔對多數任務幫助很小。

Side note `[Added]`: the default LSTM of PyTorch and TensorFlow has no peepholes and uses exactly the three equations above. The source's professor's notes give the reason: large-scale experiments by Greff et al. found that peepholes help little on most tasks.

常見錯誤：以為 1997 年的 LSTM 就有遺忘門。遺忘門是 2000 年才補上的。今天說「LSTM」，指的通常是有遺忘門的版本。

Common error: thinking that the LSTM of 1997 already had a forget gate. The forget gate was added only in 2000. Today "LSTM" usually means the version with a forget gate.

### 3. GRU：兩道門、一個狀態（The GRU: two gates and one state）

香草 LSTM 解決了飽和，但它有四組權重、兩個狀態 $\mathbf{c}_t$ 與 $\mathbf{h}_t$。能不能更簡單？2014 年 Cho 等人提出**門控循環單元**（gated recurrent unit, GRU）。它把細胞狀態和隱藏狀態合成一個 $\mathbf{h}_t$，門也從三道減到兩道。

The vanilla LSTM solved saturation, but it has four groups of weights and two states, $\mathbf{c}_t$ and $\mathbf{h}_t$. Can it be simpler? In 2014 Cho and colleagues proposed the **gated recurrent unit (GRU)**. It merges the cell state and the hidden state into one $\mathbf{h}_t$, and cuts the gates from three to two.

GRU 每一步算四條式子：

The GRU computes four equations at every step:

$$
\begin{aligned}
\mathbf{r}_t&=\sigma(\mathbf{W}_r\mathbf{h}_{t-1}+\mathbf{U}_r\mathbf{x}_t+\mathbf{b}_r),\\
\mathbf{z}_t&=\sigma(\mathbf{W}_z\mathbf{h}_{t-1}+\mathbf{U}_z\mathbf{x}_t+\mathbf{b}_z),\\
\tilde{\mathbf{h}}_t&=\tanh\big(\mathbf{W}_c(\mathbf{r}_t\odot\mathbf{h}_{t-1})+\mathbf{U}_c\mathbf{x}_t+\mathbf{b}_c\big),\\
\mathbf{h}_t&=(\mathbf{1}-\mathbf{z}_t)\odot\mathbf{h}_{t-1}+\mathbf{z}_t\odot\tilde{\mathbf{h}}_t.
\end{aligned}
$$

這裡的 $\mathbf{W}_c,\mathbf{U}_c,\mathbf{b}_c$ 是 GRU 自己的參數，只是沿用來源的名字，和 LSTM 的不是同一組。

Here $\mathbf{W}_c,\mathbf{U}_c,\mathbf{b}_c$ are the GRU's own parameters; they reuse the source's names but are not the LSTM's.

一條一條看。**重置門**（reset gate）$\mathbf{r}_t$ 決定：算新候選時，要參考多少舊狀態。$\mathbf{r}_t$ 接近 $\mathbf{0}$，候選幾乎只看新輸入，像換了話題重新開始。

Take them one by one. The **reset gate** $\mathbf{r}_t$ decides how much of the old state to consult when computing the new candidate. With $\mathbf{r}_t$ near $\mathbf{0}$, the candidate looks almost only at the new input, as if the topic changed and everything restarts.

**候選狀態**（candidate state）$\tilde{\mathbf{h}}_t$ 是「如果要更新，新狀態長什麼樣」。它的角色和 LSTM 的 $\tilde{\mathbf{c}}_t$ 一樣。

The **candidate state** $\tilde{\mathbf{h}}_t$ is "what the new state would look like if we update". It plays the same role as $\tilde{\mathbf{c}}_t$ of the LSTM.

**更新門**（update gate）$\mathbf{z}_t$ 決定新舊比例。最後一條是**凸組合**（convex combination）：兩個權重 $1-z$ 和 $z$ 都在 $0$ 到 $1$ 之間，加起來剛好是 $1$。所以每個分量的 $h_t$ 一定落在 $h_{t-1}$ 和 $\tilde{h}_t$ 之間。

The **update gate** $\mathbf{z}_t$ decides the mix of new and old. The last equation is a **convex combination**: the two weights $1-z$ and $z$ lie between $0$ and $1$ and sum to exactly $1$. So in every component, $h_t$ lies between $h_{t-1}$ and $\tilde{h}_t$.

兩個極端最好記。$\mathbf{z}_t=\mathbf{0}$ 時 $\mathbf{h}_t=\mathbf{h}_{t-1}$，整個舊狀態原樣複製，相當於 LSTM 的 $\mathbf{f}_t=\mathbf{1}$ 而且不寫入。$\mathbf{z}_t=\mathbf{1}$ 時 $\mathbf{h}_t=\tilde{\mathbf{h}}_t$，全部換新。

The two extremes are easy to remember. With $\mathbf{z}_t=\mathbf{0}$, $\mathbf{h}_t=\mathbf{h}_{t-1}$: the old state is copied as is, like $\mathbf{f}_t=\mathbf{1}$ with no writing in the LSTM. With $\mathbf{z}_t=\mathbf{1}$, $\mathbf{h}_t=\tilde{\mathbf{h}}_t$: everything is replaced.

和 LSTM 對照：$1-\mathbf{z}_t$ 扮演遺忘門，$\mathbf{z}_t$ 扮演輸入門。LSTM 的兩道門各自獨立；GRU 用一個 $\mathbf{z}_t$ 同時管兩件事，規定「留多少」加「寫多少」等於 $1$。

Compared with the LSTM: $1-\mathbf{z}_t$ plays the forget gate and $\mathbf{z}_t$ plays the input gate. The two LSTM gates are independent; the GRU controls both with one $\mathbf{z}_t$, forcing "how much to keep" plus "how much to write" to equal $1$.

類比 `[補充]`：更新門像淋浴的混水龍頭。往一邊轉是全熱水（新候選），往另一邊是全冷水（舊狀態），中間按比例混合。這個類比的侷限是：真的龍頭只有一個旋鈕，而 $\mathbf{z}_t$ 的每個分量各自有一個旋鈕，而且每一步都會自己重新轉。

Analogy `[Added]`: the update gate is like a shower mixer tap. One end gives all hot water (the new candidate), the other all cold (the old state), and in between a mix. The limit of the analogy: a real tap has one knob, while every component of $\mathbf{z}_t$ has its own knob, and it re-turns itself at every step.

| 項目 | LSTM | GRU |
|---|---|---|
| 狀態 | $\mathbf{c}_t$ 與 $\mathbf{h}_t$ 兩個 | 只有 $\mathbf{h}_t$ |
| 門 | 遺忘、輸入、輸出三道 | 重置、更新兩道 |
| 新舊混合 | $\mathbf{f}_t$ 與 $\mathbf{i}_t$ 各自獨立 | $1-\mathbf{z}_t$ 與 $\mathbf{z}_t$，和為 $1$ |
| 權重組數 | $4$ | $3$ |

| Item | LSTM | GRU |
|---|---|---|
| States | two: $\mathbf{c}_t$ and $\mathbf{h}_t$ | only $\mathbf{h}_t$ |
| Gates | three: forget, input, output | two: reset, update |
| Mixing old and new | $\mathbf{f}_t$ and $\mathbf{i}_t$ independent | $1-\mathbf{z}_t$ and $\mathbf{z}_t$, summing to $1$ |
| Weight groups | $4$ | $3$ |

常見錯誤：以為 GRU 少一道門，效果一定比 LSTM 差。來源引用 Chung 等人 2014 年的比較：在多數語音與語言任務上，兩者表現不分上下。哪個好要看任務，沒有固定的贏家。

Common error: thinking that the GRU, having one gate fewer, must perform worse than the LSTM. The source cites the 2014 comparison by Chung et al.: on most speech and language tasks the two perform about equally. Which is better depends on the task; there is no fixed winner.

### 4. 手算一步純量 GRU（One step of a scalar GRU by hand）

公式看懂了，用最小的例子走一遍 `[補充]`。取純量 $d=p=1$，上一步狀態 $h_{t-1}=0.8$，這一步輸入 $x_t=1.0$。參數如下，偏置都是 $0$：

With the equations understood, run the smallest example `[Added]`. Take scalars $d=p=1$, previous state $h_{t-1}=0.8$ and input $x_t=1.0$. The parameters are below; all biases are $0$:

| 組 | $W$ | $U$ |
|---|---|---|
| 重置門 $r$ | $1.0$ | $-1.0$ |
| 更新門 $z$ | $-1.0$ | $1.0$ |
| 候選 $c$ | $1.0$ | $-1.5$ |

| Group | $W$ | $U$ |
|---|---|---|
| reset gate $r$ | $1.0$ | $-1.0$ |
| update gate $z$ | $-1.0$ | $1.0$ |
| candidate $c$ | $1.0$ | $-1.5$ |

第一步算兩道門：

First, compute the two gates:

$$
\begin{aligned}
r_t&=\sigma(1.0\times0.8-1.0\times1.0)=\sigma(-0.2)=0.4502,\\
z_t&=\sigma(-1.0\times0.8+1.0\times1.0)=\sigma(0.2)=0.5498.
\end{aligned}
$$

剛好 $r_t=1-z_t$，因為兩個淨輸入互為相反數，而 $\sigma(-a)=1-\sigma(a)$。這只是本例的巧合。

It happens that $r_t=1-z_t$, because the two net inputs are opposite and $\sigma(-a)=1-\sigma(a)$. This is only a coincidence of this example.

第二步算候選。重置門先把舊狀態打折：$r_th_{t-1}=0.4502\times0.8=0.3601$。

Second, compute the candidate. The reset gate first discounts the old state: $r_th_{t-1}=0.4502\times0.8=0.3601$.

$$
\tilde{h}_t=\tanh(1.0\times0.3601-1.5\times1.0)=\tanh(-1.1399)=-0.8144.
$$

第三步做凸組合。舊狀態留下 $(1-z_t)h_{t-1}=0.4502\times0.8=0.3601$，新候選寫入 $z_t\tilde{h}_t=0.5498\times(-0.8144)=-0.4478$。

Third, form the convex combination. The old state keeps $(1-z_t)h_{t-1}=0.4502\times0.8=0.3601$, and the candidate writes $z_t\tilde{h}_t=0.5498\times(-0.8144)=-0.4478$.

$$
h_t=0.3601+(-0.4478)=-0.0876.
$$

用未四捨五入的值相加，結果是 $-0.0876$；用表上的四位小數相加會差 $0.0001$，這是捨入造成的。新狀態落在 $0.8$ 與 $-0.8144$ 之間，而且略偏向新候選，因為 $z_t=0.5498$ 稍大於一半。

Adding the unrounded values gives $-0.0876$; adding the four-decimal values differs by $0.0001$ because of rounding. The new state lies between $0.8$ and $-0.8144$, slightly closer to the candidate, because $z_t=0.5498$ is a little above one half.

對照：如果沒有重置（$r_t=1$），候選變成 $\tilde{h}_t=\tanh(0.8-1.5)=\tanh(-0.7)=-0.6044$，最後 $h_t=0.3601-0.3323=0.0278$，連正負號都不同。重置門讓候選更少受舊狀態影響，所以更接近新輸入的方向。

Compare: without resetting ($r_t=1$), the candidate becomes $\tilde{h}_t=\tanh(0.8-1.5)=\tanh(-0.7)=-0.6044$, and finally $h_t=0.3601-0.3323=0.0278$; even the sign differs. The reset gate makes the candidate less influenced by the old state, so it leans more toward the new input.

來源也有一個更短的算例：直接給 $h_{t-1}=0.8$、$r_t=1.0$、$\tilde{h}_t=-0.4$、$z_t=0.7$，得 $h_t=0.3\times0.8+0.7\times(-0.4)=-0.04$。程式重算結果相同。

The source has a shorter example as well: it gives $h_{t-1}=0.8$, $r_t=1.0$, $\tilde{h}_t=-0.4$ and $z_t=0.7$ directly, and gets $h_t=0.3\times0.8+0.7\times(-0.4)=-0.04$. The program reproduces this result.

![一張折線圖，橫軸是更新門 z_t 從 0 到 1，縱軸是新狀態 h_t。綠色實線從 z=0 的 h_{t-1}=0.8 直線連到 z=1 的候選 -0.8144；紫色虛線是不重置 r=1 的情況，終點 -0.6044。橘色點線在 z=0.5498，綠線上的橘色方塊 h_t = -0.0876，紫線上的方塊 h_t = 0.0278（Line plot of h_t against z_t: the green line runs from h_{t-1} = 0.8 at z = 0 to the candidate −0.8144 at z = 1; the dashed purple line, with no reset, ends at −0.6044; at z_t = 0.5498 the green point is −0.0876 and the purple point 0.0278）](../assets/figures/lec08-kp07-gru-mix.png)

**圖說 `lec08-kp07-gru-mix`**〔功能：一目了然、精準度〕：先看綠色實線的兩端：左端 $z_t=0$ 是舊狀態 $0.8$，右端 $z_t=1$ 是新候選 $-0.8144$。凸組合就是在這條直線上取一點，不會跑出兩端之外。再看橘色點線：本例 $z_t=0.5498$，對到綠線上的 $h_t=-0.0876$。最後比較紫色虛線：不重置時右端只到 $-0.6044$，同一個 $z_t$ 得到 $h_t=0.0278$。兩條線起點相同，差別全來自重置門改變了候選。

**Caption `lec08-kp07-gru-mix`** [Function: seeing it at a glance, precision]: First look at the two ends of the solid green line: the left end $z_t=0$ is the old state $0.8$, the right end $z_t=1$ is the candidate $-0.8144$. A convex combination picks a point on this segment and never leaves it. Then look at the dotted orange line: here $z_t=0.5498$, which hits the green line at $h_t=-0.0876$. Finally compare the dashed purple line: without reset its right end is only $-0.6044$, and the same $z_t$ gives $h_t=0.0278$. Both lines start at the same point; the whole difference comes from the reset gate changing the candidate.

### 5. MGU：一道門管兩件事（The MGU: one gate for two jobs）

第 4 節的例子裡，重置門和更新門都在做「舊狀態要用多少」的判斷。既然如此，能不能只留一道門？**極簡門控單元**（minimal gated unit, MGU）就這麼做：只保留一道門 $\mathbf{f}_t$，同時當重置門和更新門用。

In the example of Section 4, both the reset gate and the update gate judge "how much of the old state to use". So can we keep only one gate? The **minimal gated unit (MGU)** does exactly that: it keeps a single gate $\mathbf{f}_t$ and uses it as both the reset gate and the update gate.

這個想法由 Zhou 等人在 2016 年提出；2017 年 Heck 與 Salem 又提出幾種更簡化的 MGU 變體。

The idea was proposed by Zhou and colleagues in 2016; in 2017 Heck and Salem proposed several further simplified MGU variants.

$$
\begin{aligned}
\mathbf{f}_t&=\sigma(\mathbf{W}_f\mathbf{h}_{t-1}+\mathbf{U}_f\mathbf{x}_t+\mathbf{b}_f),\\
\tilde{\mathbf{h}}_t&=\tanh\big(\mathbf{W}_c(\mathbf{f}_t\odot\mathbf{h}_{t-1})+\mathbf{U}_c\mathbf{x}_t+\mathbf{b}_c\big),\\
\mathbf{h}_t&=(\mathbf{1}-\mathbf{f}_t)\odot\mathbf{h}_{t-1}+\mathbf{f}_t\odot\tilde{\mathbf{h}}_t.
\end{aligned}
$$

和 GRU 比，只是把 $\mathbf{r}_t$ 與 $\mathbf{z}_t$ 都換成同一個 $\mathbf{f}_t$。少了一道門，就少一組 $\mathbf{W}$、$\mathbf{U}$、$\mathbf{b}$。

Compared with the GRU, both $\mathbf{r}_t$ and $\mathbf{z}_t$ are replaced by the same $\mathbf{f}_t$. One gate fewer means one group of $\mathbf{W}$, $\mathbf{U}$, $\mathbf{b}$ fewer.

同一步再算一次 `[補充]`：讓 MGU 的門用 GRU 更新門的參數（$W_f=-1.0$、$U_f=1.0$），候選參數不變。那麼 $f_t=\sigma(0.2)=0.5498$；$f_th_{t-1}=0.4399$；

Run the same step again `[Added]`: let the MGU gate use the parameters of the GRU update gate ($W_f=-1.0$, $U_f=1.0$), with the same candidate parameters. Then $f_t=\sigma(0.2)=0.5498$ and $f_th_{t-1}=0.4399$;

$$
\begin{aligned}
\tilde{h}_t&=\tanh(0.4399-1.5)=\tanh(-1.0601)=-0.7857,\\
h_t&=0.3601+0.5498\times(-0.7857)=0.3601-0.4320=-0.0719.
\end{aligned}
$$

結果和 GRU 的 $-0.0876$ 很接近，但不相同。差別在候選：MGU 用 $f_t=0.5498$ 打折舊狀態，GRU 用 $r_t=0.4502$。

The result is close to the GRU's $-0.0876$ but not equal. The difference is in the candidate: the MGU discounts the old state with $f_t=0.5498$, the GRU with $r_t=0.4502$.

這道門雖然叫「遺忘門」，方向卻和 LSTM 相反，要特別小心。在 LSTM，$\mathbf{f}_t=\mathbf{1}$ 是「舊記憶全部留下」。在 MGU，$\mathbf{f}_t=\mathbf{1}$ 讓 $\mathbf{h}_t=\tilde{\mathbf{h}}_t$，是「舊狀態全部換掉」。MGU 的 $\mathbf{f}_t$ 越大，忘得越多。

This gate is called a "forget gate", but it points the opposite way to the LSTM's, so be careful. In the LSTM, $\mathbf{f}_t=\mathbf{1}$ means "keep all old memory". In the MGU, $\mathbf{f}_t=\mathbf{1}$ gives $\mathbf{h}_t=\tilde{\mathbf{h}}_t$, which means "replace the whole old state". The larger the MGU's $\mathbf{f}_t$, the more it forgets.

能不能連最後一道門也拿掉？來源的教授祕笈說：如果把 $\mathbf{f}_t$ 固定成常數 $0.5$，就只剩固定比例的平滑混合。網路失去了「看情況決定寫入或清空」的能力，也就不再是門控網路。

Can the last gate be removed too? The source's professor's notes say: if $\mathbf{f}_t$ is fixed to the constant $0.5$, only a smooth mix with a fixed ratio is left. The network loses the ability to "decide by context whether to write or clear", so it is no longer a gated network.

常見錯誤：以為參數少一半，表現就差一半。來源指出，在許多中小型資料集上，MGU 的表現和 GRU 很接近。容量不足通常要到很複雜的大型任務才會顯現。

Common error: thinking that half the parameters means half the performance. The source notes that on many small and medium datasets the MGU performs close to the GRU. Its limited capacity usually shows only on very complex, large tasks.

### 6. 數一數參數：LSTM、GRU、MGU（Counting parameters: LSTM, GRU and MGU）

第 5 節說「少一道門就少一組權重」，現在把它算清楚。每一組都是一個 $p\times p$ 的 $\mathbf{W}$、一個 $p\times d$ 的 $\mathbf{U}$ 和 $p$ 個偏置，叫做一**塊**（block）：

Section 5 said "one gate fewer means one group of weights fewer"; now count it exactly. Each group is a $p\times p$ matrix $\mathbf{W}$, a $p\times d$ matrix $\mathbf{U}$ and $p$ biases; call it one **block**:

$$
\text{one block}=p^2+pd+p.
$$

然後只要數塊數。LSTM 有三道門加一個候選，共 $4$ 塊，GRU 有兩道門加一個候選，共 $3$ 塊，MGU 有一道門加一個候選，共 $2$ 塊。KP01 的 RNN 只有 $1$ 塊。

Then just count blocks. The LSTM has three gates plus one candidate, $4$ blocks; the GRU has two gates plus one candidate, $3$ blocks; the MGU has one gate plus one candidate, $2$ blocks. The RNN of KP01 has only $1$ block.

$$
\text{LSTM}=4(p^2+pd+p),\qquad\text{GRU}=3(p^2+pd+p),\qquad\text{MGU}=2(p^2+pd+p).
$$

先用 KP01 的小例子 $d=3$、$p=4$：一塊是 $16+12+4=32$ 個。再用來源的大例子 $p=d=256$：一塊是 $65{,}536+65{,}536+256=131{,}328$ 個。

First take the small example of KP01, $d=3$ and $p=4$: one block is $16+12+4=32$. Then the source's large example, $p=d=256$: one block is $65{,}536+65{,}536+256=131{,}328$.

| 模型 | 塊數 | $d=3,\ p=4$ | $p=d=256$ |
|---|---|---|---|
| RNN（KP01） | $1$ | $32$ | $131{,}328$ |
| MGU | $2$ | $64$ | $262{,}656$ |
| GRU | $3$ | $96$ | $393{,}984$ |
| LSTM | $4$ | $128$ | $525{,}312$ |

| Model | Blocks | $d=3,\ p=4$ | $p=d=256$ |
|---|---|---|---|
| RNN (KP01) | $1$ | $32$ | $131{,}328$ |
| MGU | $2$ | $64$ | $262{,}656$ |
| GRU | $3$ | $96$ | $393{,}984$ |
| LSTM | $4$ | $128$ | $525{,}312$ |

小例子的 RNN 是 $32$ 個，加上 KP01 輸出層的 $\mathbf{V}$ 與 $\mathbf{b}_y$ 共 $10$ 個，正好是 KP01 的 $42$ 個。

The small RNN has $32$; adding the $10$ of the output layer $\mathbf{V}$ and $\mathbf{b}_y$ in KP01 gives exactly the $42$ of KP01.

![橫條圖，p = d = 256。灰色 RNN 131,328（1 塊，0 道門）、綠色 MGU 262,656（2 塊，1 道門）、藍色 GRU 393,984（3 塊，2 道門）、紅色 LSTM 525,312（4 塊，3 道門）。橫軸註明一塊等於 p^2 + pd + p = 131,328（Horizontal bars for p = d = 256: RNN 131,328 with 1 block, MGU 262,656 with 2 blocks, GRU 393,984 with 3 blocks, LSTM 525,312 with 4 blocks; one block is p^2 + pd + p = 131,328）](../assets/figures/lec08-kp07-params.png)

**圖說 `lec08-kp07-params`**〔功能：一目了然、精準度〕：先看四條長條的長度：從 RNN 到 LSTM 是 $1:2:3:4$ 的等差階梯，每多一道門就多一塊。再看右邊的數字：MGU 正好是 LSTM 的一半，GRU 是 LSTM 的四分之三。最後看橫軸說明：所有長條都只算循環單元本身，沒有算輸出層。

**Caption `lec08-kp07-params`** [Function: seeing it at a glance, precision]: First look at the lengths of the four bars: from the RNN to the LSTM they form a $1:2:3:4$ staircase, one more block per extra gate. Then read the numbers on the right: the MGU is exactly half of the LSTM, the GRU three quarters. Finally read the axis label: every bar counts only the recurrent cell itself, without the output layer.

所以「GRU 比 LSTM 省 25%」是 $1-3/4$，「MGU 比 LSTM 省 50%」是 $1-2/4$，「MGU 比 GRU 省約 33%」是 $1-2/3=33.33\%$。程式驗算這三個比例都對。以 $p=d=256$ 來說，MGU 比 LSTM 少 $262{,}656$ 個參數。

So "the GRU saves 25% over the LSTM" is $1-3/4$, "the MGU saves 50% over the LSTM" is $1-2/4$, and "the MGU saves about 33% over the GRU" is $1-2/3=33.33\%$. The program confirms all three ratios. For $p=d=256$, the MGU has $262{,}656$ fewer parameters than the LSTM.

但這些比例有前提 `[補充]`。只數循環單元、LSTM 不含窺視孔時，才剛好是 25%。

But these ratios have conditions `[Added]`. They are exactly 25% only when counting the recurrent cell alone and the LSTM has no peepholes.

若 LSTM 有窺視孔，多 $3p$ 個：小例子是 $140$ 個，GRU 省 $31.43\%$；大例子是 $526{,}080$ 個，GRU 省 $25.11\%$。若連輸出層也算（小例子 $q=2$，多 $10$ 個），LSTM $138$、GRU $106$，只省 $23.19\%$。

With peepholes the LSTM gains $3p$: $140$ in the small example, where the GRU saves $31.43\%$; $526{,}080$ in the large example, where the GRU saves $25.11\%$. Counting the output layer too (small example, $q=2$, $10$ more), the LSTM has $138$ and the GRU $106$, a saving of only $23.19\%$.

PyTorch 的寫法每塊有兩個偏置向量 `[補充]`，一塊變成 $p^2+pd+2p$。小例子是 LSTM $144$、GRU $108$、MGU $72$；比例仍是 $4:3:2$，省 25% 不變。

PyTorch's version has two bias vectors per block `[Added]`, so one block becomes $p^2+pd+2p$. The small example gives LSTM $144$, GRU $108$, MGU $72$; the ratio is still $4:3:2$ and the 25% saving is unchanged.

第 2 節的兩個 LSTM 變體也能這樣數 `[補充]`。全門控循環依 Greff 等人的描述，多 $9$ 個 $p\times p$ 矩陣，$p=256$ 時多 $589{,}824$ 個，比整個 LSTM 還多。投影 LSTM 若把 $256$ 維的 $\mathbf{h}_t$ 投影成 $128$ 維，門的部分降為 $394{,}240$，加上投影矩陣 $32{,}768$，共 $427{,}008$，比 LSTM 少 $18.71\%$。

The two LSTM variants of Section 2 can be counted the same way `[Added]`. Full gate recurrence, as described by Greff et al., adds $9$ matrices of size $p\times p$: $589{,}824$ more for $p=256$, more than the whole LSTM. An LSTM with projection that maps the $256$-dimensional $\mathbf{h}_t$ to $128$ dimensions has $394{,}240$ in its gates plus $32{,}768$ in the projection matrix, $427{,}008$ in all, $18.71\%$ fewer than the LSTM.

常見錯誤：把「省 25%」當成整個模型、任何設定都成立。它只是循環單元塊數的比例。實際省多少，要看有沒有窺視孔、輸出層多大、框架怎麼放偏置。

Common error: treating "saves 25%" as true for the whole model in every setting. It is only the ratio of block counts in the recurrent cell. The real saving depends on peepholes, the size of the output layer and how the framework stores biases.

### 本單元重點整理（KP07 summary）

**一句話重點**：1997 年的 LSTM 沒有遺忘門，細胞狀態只增不減而飽和，2000 年補上遺忘門與窺視孔，2005 年成為香草 LSTM，GRU 把雙狀態合成一個 $\mathbf{h}_t$、門減為兩道，用 $\mathbf{h}_t=(\mathbf{1}-\mathbf{z}_t)\odot\mathbf{h}_{t-1}+\mathbf{z}_t\odot\tilde{\mathbf{h}}_t$ 做凸組合，MGU 再把兩道門合成一道，循環單元參數依序是 $4$、$3$、$2$ 塊 $p^2+pd+p$。

**One-sentence summary**: the LSTM of 1997 had no forget gate, so its cell state only grew and saturated; the forget gate and peepholes came in 2000, and the vanilla LSTM in 2005; the GRU merges the two states into one $\mathbf{h}_t$ with two gates and the convex combination $\mathbf{h}_t=(\mathbf{1}-\mathbf{z}_t)\odot\mathbf{h}_{t-1}+\mathbf{z}_t\odot\tilde{\mathbf{h}}_t$; the MGU merges the two gates into one; their recurrent cells have $4$, $3$ and $2$ blocks of $p^2+pd+p$ parameters.

**白話說明**：遺忘門讓記憶能清掉，所以不會塞爆。GRU 把「留多少」和「寫多少」綁在一起，用一個旋鈕調新舊比例。MGU 更進一步，連「算候選時參考多少舊狀態」也用同一個旋鈕。旋鈕越少，要學的參數越少。

**Plain explanation**: the forget gate lets memory be cleared, so it does not overflow. The GRU ties "how much to keep" to "how much to write" and sets the mix with one knob. The MGU goes further and uses the same knob for "how much old state the candidate consults". Fewer knobs means fewer parameters to learn.

**正式定義**：GRU 為 $\mathbf{r}_t=\sigma(\mathbf{W}_r\mathbf{h}_{t-1}+\mathbf{U}_r\mathbf{x}_t+\mathbf{b}_r)$、$\mathbf{z}_t=\sigma(\mathbf{W}_z\mathbf{h}_{t-1}+\mathbf{U}_z\mathbf{x}_t+\mathbf{b}_z)$、$\tilde{\mathbf{h}}_t=\tanh(\mathbf{W}_c(\mathbf{r}_t\odot\mathbf{h}_{t-1})+\mathbf{U}_c\mathbf{x}_t+\mathbf{b}_c)$、$\mathbf{h}_t=(\mathbf{1}-\mathbf{z}_t)\odot\mathbf{h}_{t-1}+\mathbf{z}_t\odot\tilde{\mathbf{h}}_t$；MGU 把 $\mathbf{r}_t$ 與 $\mathbf{z}_t$ 都換成 $\mathbf{f}_t$。參數：LSTM $4(p^2+pd+p)$、GRU $3(p^2+pd+p)$、MGU $2(p^2+pd+p)$。

**Formal definition**: the GRU is $\mathbf{r}_t=\sigma(\mathbf{W}_r\mathbf{h}_{t-1}+\mathbf{U}_r\mathbf{x}_t+\mathbf{b}_r)$, $\mathbf{z}_t=\sigma(\mathbf{W}_z\mathbf{h}_{t-1}+\mathbf{U}_z\mathbf{x}_t+\mathbf{b}_z)$, $\tilde{\mathbf{h}}_t=\tanh(\mathbf{W}_c(\mathbf{r}_t\odot\mathbf{h}_{t-1})+\mathbf{U}_c\mathbf{x}_t+\mathbf{b}_c)$, $\mathbf{h}_t=(\mathbf{1}-\mathbf{z}_t)\odot\mathbf{h}_{t-1}+\mathbf{z}_t\odot\tilde{\mathbf{h}}_t$; the MGU replaces both $\mathbf{r}_t$ and $\mathbf{z}_t$ with $\mathbf{f}_t$. Parameters: LSTM $4(p^2+pd+p)$, GRU $3(p^2+pd+p)$, MGU $2(p^2+pd+p)$.

**例子**：每步寫入 $0.5$、走 $50$ 步：沒有遺忘門 $c_{50}=25.0$，$\tanh$ 斜率 $7.7\times10^{-22}$，$f_t=0.7$ 時 $c_{50}=1.6667$，斜率 $0.1330$。純量 GRU（$h_{t-1}=0.8$、$x_t=1.0$）：$r_t=0.4502$、$z_t=0.5498$、$\tilde{h}_t=-0.8144$、$h_t=-0.0876$，同一步的 MGU 得 $h_t=-0.0719$。$p=d=256$ 時 LSTM、GRU、MGU 為 $525{,}312$、$393{,}984$、$262{,}656$ 個。

**Example**: writing $0.5$ per step for $50$ steps: without a forget gate $c_{50}=25.0$ and the $\tanh$ slope is $7.7\times10^{-22}$; with $f_t=0.7$, $c_{50}=1.6667$ and the slope is $0.1330$. Scalar GRU ($h_{t-1}=0.8$, $x_t=1.0$): $r_t=0.4502$, $z_t=0.5498$, $\tilde{h}_t=-0.8144$, $h_t=-0.0876$; the MGU on the same step gives $h_t=-0.0719$. For $p=d=256$ the LSTM, GRU and MGU have $525{,}312$, $393{,}984$ and $262{,}656$ parameters.

**常見迷思**：

- 「1997 年的 LSTM 就有遺忘門。」錯，遺忘門是 2000 年 Gers 等人才加上的。
- 「GRU 少一道門，表現一定比 LSTM 差。」錯，多數任務上兩者相當，要看任務。
- 「MGU 是 Heck 與 Salem 在 2017 年提出的。」不準確，最早是 2016 年 Zhou 等人提出。
- 「MGU 的遺忘門和 LSTM 的遺忘門一樣，$\mathbf{f}_t=\mathbf{1}$ 是全部記住。」錯，MGU 的 $\mathbf{f}_t=\mathbf{1}$ 是全部換新。
- 「GRU 在任何情況都剛好省 25% 參數。」錯，這只對不含窺視孔、只數循環單元的情況成立。

**Common misconceptions**:

- "The LSTM of 1997 already had a forget gate." Wrong: Gers et al. added it in 2000.
- "With one gate fewer, the GRU must perform worse than the LSTM." Wrong: on most tasks they are about equal; it depends on the task.
- "The MGU was proposed by Heck and Salem in 2017." Inaccurate: Zhou et al. first proposed it in 2016.
- "The MGU's forget gate is like the LSTM's: $\mathbf{f}_t=\mathbf{1}$ keeps everything." Wrong: in the MGU, $\mathbf{f}_t=\mathbf{1}$ replaces everything.
- "The GRU always saves exactly 25% of the parameters." Wrong: that holds only without peepholes and counting the recurrent cell alone.

**圖**：本單元有三張圖。

- `lec08-kp07-cell-growth`：沒有遺忘門時細胞狀態直線上升、$\tanh$ 斜率趨近 $0$；有遺忘門時停在極限。
- `lec08-kp07-gru-mix`：GRU 的新狀態是舊狀態與候選之間直線上的一點，重置門改變了候選的位置。
- `lec08-kp07-params`：RNN、MGU、GRU、LSTM 的參數是 $1:2:3:4$ 的階梯。

**Figures** — this unit has three figures.

- `lec08-kp07-cell-growth`: without a forget gate the cell state rises linearly and the $\tanh$ slope goes to $0$; with a forget gate it stops at a limit.
- `lec08-kp07-gru-mix`: the new GRU state is a point on the segment between the old state and the candidate, and the reset gate moves the candidate.
- `lec08-kp07-params`: the parameters of the RNN, MGU, GRU and LSTM form a $1:2:3:4$ staircase.

### 小練習（Practice）

1. 設 $d=5$、$p=8$。算出一塊的參數個數，以及 LSTM、GRU、MGU 循環單元的參數總數（不含窺視孔與輸出層）。提示：一塊是 $p^2+pd+p$。
2. 沿用第 4 節的 GRU 參數與 $h_{t-1}=0.8$，但輸入改成 $x_t=0$。算出 $r_t$、$z_t$、$\tilde{h}_t$、$h_t$。提示：$x_t=0$ 時，每個淨輸入只剩 $W$ 乘 $h_{t-1}$ 那一項。
3. 第 1 節的細胞若改用 $f_t=0.9$，每步仍寫入 $0.5$，$c_t$ 的極限是多少？這時 $\tanh$ 的斜率大約多少，比 $f_t=0.7$ 好還是差？提示：解 $c=0.9c+0.5$。

1. Let $d=5$ and $p=8$. Compute the number of parameters in one block, and the totals for the recurrent cells of the LSTM, GRU and MGU (without peepholes or output layer). Hint: one block is $p^2+pd+p$.
2. Keep the GRU parameters of Section 4 and $h_{t-1}=0.8$, but change the input to $x_t=0$. Compute $r_t$, $z_t$, $\tilde{h}_t$ and $h_t$. Hint: with $x_t=0$, each net input keeps only the term $W$ times $h_{t-1}$.
3. If the cell of Section 1 uses $f_t=0.9$ and still writes $0.5$ per step, what is the limit of $c_t$? About how large is the $\tanh$ slope then, and is that better or worse than with $f_t=0.7$? Hint: solve $c=0.9c+0.5$.

**第 1 題解答** `[補充]`：一塊是 $64+40+8=112$ 個。LSTM $4\times112=448$，GRU $3\times112=336$，MGU $2\times112=224$。

**Solution to question 1** `[Added]`: one block is $64+40+8=112$. LSTM $4\times112=448$, GRU $3\times112=336$, MGU $2\times112=224$.

**第 2 題解答** `[補充]`：$r_t=\sigma(0.8)=0.6900$，$z_t=\sigma(-0.8)=0.3100$。$r_th_{t-1}=0.5520$，$\tilde{h}_t=\tanh(0.5520)=0.5020$。$h_t=0.6900\times0.8+0.3100\times0.5020=0.5520+0.1556=0.7076$。沒有新輸入時，更新門偏小，狀態大多保留。

**Solution to question 2** `[Added]`: $r_t=\sigma(0.8)=0.6900$, $z_t=\sigma(-0.8)=0.3100$. $r_th_{t-1}=0.5520$ and $\tilde{h}_t=\tanh(0.5520)=0.5020$. $h_t=0.6900\times0.8+0.3100\times0.5020=0.5520+0.1556=0.7076$. With no new input the update gate is small, so most of the state is kept.

**第 3 題解答** `[補充]`：$c_\infty=0.5/(1-0.9)=5.0$。$\tanh(5.0)=0.9999$，斜率約 $1.8\times10^{-4}$，比 $f_t=0.7$ 的 $0.1330$ 小很多。遺忘門越接近 $1$，記得越久，但狀態也越大、越容易飽和。

**Solution to question 3** `[Added]`: $c_\infty=0.5/(1-0.9)=5.0$. $\tanh(5.0)=0.9999$ and the slope is about $1.8\times10^{-4}$, far smaller than the $0.1330$ of $f_t=0.7$. The closer the forget gate is to $1$, the longer the memory, but the larger the state and the easier it saturates.

### 小測驗（Quiz，來源原題）

來源的三題自我檢測保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The three self-check questions of the source are kept with all options; the source page does not mark the answers directly, and the explanations below follow the source text.

1. Gers 等人在 2000 年為原始 LSTM 引入「遺忘門（Forget Gate）」的主要歷史原因是什麼？
   - (A) 原始 LSTM 缺乏清空記憶的機制，處理長序列時細胞狀態會無限制累積導致飽和與失真
   - (B) 為了減少計算量
   - (C) 因為原始 LSTM 無法處理圖像
   - (D) 為了替換 Softmax
2. 相較於 LSTM，GRU 在結構上的主要精簡特徵是什麼？
   - (A) 完全不需要激活函數
   - (B) 將細胞狀態 $c_t$ 與隱藏狀態 $h_t$ 合併，並將門控減少為重置門（$r_t$）與更新門（$z_t$）兩個
   - (C) 取消了所有偏置項
   - (D) 只能處理單步序列
3. 極簡門控單元（MGU）相比於 GRU，參數數量進一步減少了多少比例？
   - (A) 矩陣參數從 3 組減少到 2 組（減少約 33% 的權重參數）
   - (B) 減少了 90%
   - (C) 沒有減少
   - (D) 增加了 50%

1. What is the main historical reason why Gers et al. introduced the "forget gate" into the original LSTM in 2000?
   - (A) The original LSTM had no way to clear its memory, so on long sequences the cell state accumulated without bound, causing saturation and distortion
   - (B) To reduce computation
   - (C) Because the original LSTM could not process images
   - (D) To replace softmax
2. Compared with the LSTM, what is the main structural simplification of the GRU?
   - (A) It needs no activation function at all
   - (B) It merges the cell state $c_t$ and the hidden state $h_t$, and cuts the gates to two: the reset gate ($r_t$) and the update gate ($z_t$)
   - (C) It removes all bias terms
   - (D) It can only process single-step sequences
3. Compared with the GRU, by what fraction does the minimal gated unit (MGU) further reduce the number of parameters?
   - (A) The weight groups drop from 3 to 2 (about 33% fewer weight parameters)
   - (B) By 90%
   - (C) Not at all
   - (D) It increases them by 50%

**解析** `[補充]`：第 1 題選 (A)：第 1 節的算例正是「沒有遺忘門，$c_{50}=25.0$ 而飽和」；遺忘門反而多一組權重，所以不是為了減少計算量。第 2 題選 (B)：第 3 節的對照表；GRU 仍用 $\sigma$ 與 $\tanh$，也有偏置。第 3 題選 (A)：第 6 節算出 $1-2/3=33.33\%$。

**Explanations** `[Added]`: question 1 is A: the example of Section 1 is exactly "no forget gate, $c_{50}=25.0$, saturation"; the forget gate actually adds a group of weights, so it is not about less computation. Question 2 is B: see the comparison table of Section 3; the GRU still uses $\sigma$ and $\tanh$ and has biases. Question 3 is A: Section 6 gives $1-2/3=33.33\%$.

### 留給下一單元的問題（The question for the next unit）

RNN、LSTM、GRU、MGU 都有一個共同點：它們只從左讀到右，$\mathbf{h}_t$ 只知道第 $1$ 步到第 $t$ 步。可是有些詞的意思要看後面才懂。

The RNN, LSTM, GRU and MGU share one thing: they read only from left to right, so $\mathbf{h}_t$ knows only steps $1$ to $t$. But the meaning of some words depends on what comes later.

例如「他把蘋果……」，後面接「吃了」還是「手機摔了」，「蘋果」的意思就不同。單向網路該怎麼辦？下一單元（KP08）的雙向循環網路回答這個問題。

In "he took the apple ...", whether "and ate it" or "and unlocked it" follows changes what "apple" means. What can a one-directional network do? The bidirectional recurrent network of the next unit (KP08) answers this question.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第8章「循環神經網絡與長短期記憶網絡」§4.2、§4.3.1、§4.3.2 與其教授祕笈（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch08-rnn-lstm/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §4.2, §4.3.1 and §4.3.2 of Chapter 8, "Recurrent neural networks and long short-term memory networks", of CCC-LNS IDL_115 and its professor's notes (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch08-rnn-lstm/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源寫 MGU 是「2017 年 Heck 與 Salem 提出」，祕笈的迷思欄也這樣說。MGU 最早由 Zhou 等人在 2016 年提出；Heck 與 Salem 在 2017 年提出的是更簡化的 MGU 變體。本講義第 5 節依此更正。
- 來源的參數比例（GRU 省 25%、MGU 比 LSTM 省 50%、比 GRU 省約 33%）程式驗算正確，但只在「只數循環單元、LSTM 無窺視孔」時成立。本講義補上有窺視孔（$31.43\%$／$25.11\%$）與含輸出層（$23.19\%$）的數字。
- 來源說 $\tanh(25)=1.0$「失去梯度」。$\tanh(25.0)$ 四捨五入是 $1.0000$，斜率是 $7.7\times10^{-22}$，不是剛好 $0$；結論不變。
- 來源 §4.2 互動展示說明有一處 `ilde{c}_t`，漏了反斜線，應是 $\tilde{c}_t$。
- 來源用粗斜體 `\boldsymbol` 與 $\operatorname{sig}$；本課程網站不支援 `\boldsymbol`，改用 `\mathbf`；sigmoid 寫成 $\sigma$，意義相同。
- 取捨：GRU 的更新式有兩種慣例。Cho 等人 2014 年的原文把 $\mathbf{z}_t$ 乘在舊狀態上，角色正好相反；本講義沿用來源的寫法（$\mathbf{z}_t$ 乘在候選上），讀其他教材時要先看清楚。
- 本講義補上：$\tanh$ 斜率的數字、純量 GRU 與 MGU 的算例、無重置的對照、$d=3,p=4$ 的參數表、PyTorch 雙偏置、全門控循環與投影 LSTM 的參數數字、三題小練習與解答。全門控循環「多 $9$ 個矩陣」依 Greff 等人的描述，請老師確認。
- 取捨：來源的互動滑桿無法放進講義，改用 `lec08-kp07-gru-mix` 與 `lec08-kp07-params` 兩張圖呈現；來源的時間軸圖改用第 2 節的表格。

- The source says the MGU was "proposed by Heck and Salem in 2017", and its misconception box says the same. The MGU was first proposed by Zhou et al. in 2016; what Heck and Salem proposed in 2017 were further simplified MGU variants. Section 5 of this note is corrected accordingly.
- The source's ratios (GRU saves 25%, MGU saves 50% over the LSTM and about 33% over the GRU) are confirmed by the program, but hold only when counting the recurrent cell alone with no peepholes. This note adds the figures with peepholes ($31.43\%$ / $25.11\%$) and with the output layer ($23.19\%$).
- The source says $\tanh(25)=1.0$ "loses the gradient". $\tanh(25.0)$ rounds to $1.0000$ and the slope is $7.7\times10^{-22}$, not exactly $0$; the conclusion is unchanged.
- The interactive demo text of §4.2 has `ilde{c}_t` with a missing backslash; it should be $\tilde{c}_t$.
- The source uses bold italic `\boldsymbol` and $\operatorname{sig}$; the course website does not support `\boldsymbol`, so this note uses `\mathbf`, and writes the sigmoid as $\sigma$ with the same meaning.
- Choice: there are two conventions for the GRU update. The 2014 paper by Cho et al. multiplies $\mathbf{z}_t$ with the old state, the opposite roles; this note follows the source ($\mathbf{z}_t$ multiplies the candidate), so check the convention when reading other texts.
- This note adds: the $\tanh$ slope numbers, the scalar GRU and MGU examples, the no-reset comparison, the parameter table for $d=3,p=4$, PyTorch's two biases, the counts for full gate recurrence and the projection LSTM, and three practice questions with solutions. "Nine extra matrices" for full gate recurrence follows Greff et al.; please confirm.
- Choice: the source's interactive sliders cannot go into a note, so the figures `lec08-kp07-gru-mix` and `lec08-kp07-params` show them; the source's timeline figure becomes the table in Section 2.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 長短期記憶網路 | long short-term memory (LSTM) |
| 原始 LSTM | original LSTM |
| 香草 LSTM | vanilla LSTM |
| 細胞狀態（KP06） | cell state |
| 候選記憶（KP06） | candidate memory |
| 遺忘門 | forget gate |
| 輸入門 | input gate |
| 輸出門 | output gate |
| 窺視孔 | peephole connections |
| 飽和 | saturation |
| 極限 | limit |
| 常數誤差流（KP06） | constant error carousel (CEC) |
| 全門控循環 | full gate recurrence |
| 投影 LSTM | LSTM with projection |
| 門控循環單元 | gated recurrent unit (GRU) |
| 重置門 | reset gate |
| 更新門 | update gate |
| 候選狀態 | candidate state |
| 凸組合 | convex combination |
| 極簡門控單元 | minimal gated unit (MGU) |
| 塊（一組權重與偏置） | block |
| 循環神經網路 | recurrent neural network (RNN) |
| 隨時間反向傳播 | backpropagation through time (BPTT) |
| 雙向循環網路（KP08） | bidirectional recurrent neural network (BiRNN) |
| 深度學習框架 | PyTorch |
| 深度學習框架 | TensorFlow |
| 本課程的網頁教材 | CCC-LNS |
