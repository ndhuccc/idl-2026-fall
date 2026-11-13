# 第11週 KP04：殘差、層正規化與前饋網路，讓一層層疊得又深又穩（Residual connections, layer normalization and the feed-forward network: stacking layers deep and stable）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第11週的第四個知識點（KP04），接在 KP03 之後。多頭注意力的輸出形狀和輸入一樣，看起來可以一層層往上疊。可是疊深了，第3週的梯度消失會回來，每層數值的大小也可能越來越偏。這一節看 KP01 圖上每個子層後面的「Add & Norm」做了什麼，以及注意力之後那個前饋網路負責什麼。

This is the fourth knowledge point (KP04) of Week 11 and follows KP03. Multi-head attention keeps the input shape, so it looks as if layers can simply be stacked. But in a deep stack the vanishing gradient of Week 3 comes back, and the scale of the numbers may drift layer after layer. This note shows what the "Add & Norm" after each sublayer in KP01's figure does, and what the feed-forward network after attention is for.

先備知識是第3週的梯度消失與連鎖律、第4週的批次正規化，以及本週 KP01 的 Transformer-Base 設定（$d_{\text{model}}=512$、$d_{\text{ff}}=2048$、$N=6$）。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第9章改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec11-kp04-figures.py` 算出，可以自己重算。

The prerequisites are the vanishing gradient and chain rule of Week 3, batch normalization from Week 4, and this week's KP01 Transformer-Base settings ($d_{\text{model}}=512$, $d_{\text{ff}}=2048$, $N=6$). Every term and symbol is defined when it first appears. This note is rewritten from Chapter 9 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec11-kp04-figures.py`, so you can recompute it yourself.

## KP04 殘差、層正規化與前饋網路，讓一層層疊得又深又穩（Residual connections, layer normalization and the feed-forward network: stacking layers deep and stable）

### 學習目標（Learning objectives）

讀完之後，你能用 $y=x+F(x)$ 的導數說明殘差連接為什麼讓梯度能傳回底層。你能算出一個向量的層正規化，並說明它和批次正規化差在「沿哪個方向算平均」。你能寫出逐位置前饋網路的公式與形狀，說明它在每個詞上各自做、為什麼佔一層編碼器裡約三分之二的參數。你也能說出一層編碼器裡兩道 Add & Norm 的位置，以及原版 Post-LN 與後來常用的 Pre-LN 差在哪裡。

After reading, you can use the derivative of $y=x+F(x)$ to explain why a residual connection lets the gradient reach the bottom layers. You can compute the layer normalization of a vector and explain how it differs from batch normalization in "which direction the average is taken". You can write the formula and shapes of the position-wise feed-forward network, explain that it works on each word separately, and why it holds about two thirds of an encoder layer's parameters. You can also name the places of the two Add & Norm steps in an encoder layer, and the difference between the original Post-LN and the later, common Pre-LN.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{x}\in\mathbb{R}^{d_{\text{model}}}$ | 一個詞在某一層的向量 | the vector of one word at some layer |
| $F(\cdot)$ | 一個子層（多頭注意力或前饋網路） | a sublayer (multi-head attention or the feed-forward network) |
| $\mathcal{L}$ | 損失 | loss |
| $\mu,\ \sigma^2$ | 一個向量各分量的平均與變異數 | mean and variance of the components of one vector |
| $\epsilon$ | 防止除以零的小常數，例如 $10^{-5}$ | a small constant that prevents division by zero, for example $10^{-5}$ |
| $\boldsymbol{\gamma},\ \boldsymbol{\beta}\in\mathbb{R}^{d_{\text{model}}}$ | 層正規化可學習的縮放與平移 | learnable scale and shift of layer normalization |
| $\operatorname{LN}(\cdot)$ | 層正規化 | layer normalization |
| $d_{\text{ff}}$ | 前饋網路中間層的寬度，Base 是 $2048$ | width of the feed-forward hidden layer, $2048$ in Base |
| $\mathbf{W}_1,\mathbf{b}_1,\mathbf{W}_2,\mathbf{b}_2$ | 前饋網路兩層的權重與偏差 | weights and biases of the two feed-forward layers |

### 1. 殘差連接：替梯度留一條直達的路（Residual connection: a direct road for the gradient）

殘差連接（residual connection）的作法很簡單：子層的輸出不直接往上送，而是**加回子層的輸入**，

A residual connection is simple: the output of a sublayer is not passed up directly but is **added back to the sublayer's input**,

$$
\mathbf{y}=\mathbf{x}+F(\mathbf{x}).
$$

這個想法來自影像的殘差網路 ResNet。子層只需要學「在原來的基礎上補什麼」，而不是從零重寫整個向量。相加需要 $\mathbf{x}$ 和 $F(\mathbf{x})$ 形狀相同，這就是 KP01、KP03 一再強調「形狀不變」的原因。

The idea comes from ResNet, the residual network for images. The sublayer only has to learn "what to add on top of what is already there", not rewrite the whole vector from scratch. The addition needs $\mathbf{x}$ and $F(\mathbf{x})$ to have the same shape, which is why KP01 and KP03 keep stressing that the shape does not change.

為什麼這樣有幫助？看倒傳遞。對 $\mathbf{x}$ 微分，$\partial\mathbf{y}/\partial\mathbf{x}=\mathbf{I}+\partial F/\partial\mathbf{x}$，其中 $\mathbf{I}$ 是單位矩陣。所以傳回 $\mathbf{x}$ 的梯度是兩項之和：一項直接照搬上面傳下來的梯度，一項經過子層。就算子層的導數很小，第一項仍然原封不動地傳下去。

Why does this help? Look at backpropagation. Differentiating with respect to $\mathbf{x}$ gives $\partial\mathbf{y}/\partial\mathbf{x}=\mathbf{I}+\partial F/\partial\mathbf{x}$, where $\mathbf{I}$ is the identity matrix. So the gradient sent back to $\mathbf{x}$ is a sum of two terms: one copies the gradient coming from above unchanged, and one passes through the sublayer. Even when the sublayer's derivative is tiny, the first term still goes down intact.

來源的純量例子：$x=2$，子層 $F(x)=0.1x^2=0.4$，輸出 $y=2.4$。若上面傳下來 $\partial\mathcal{L}/\partial y=1$，子層的導數是 $0.2x=0.4$。沒有殘差時，傳回 $x$ 的梯度只有 $0.4$；有殘差時是 $1\times(1+0.4)=1.4$。

The source's scalar example: $x=2$, sublayer $F(x)=0.1x^2=0.4$, output $y=2.4$. If $\partial\mathcal{L}/\partial y=1$ comes from above, the sublayer's derivative is $0.2x=0.4$. Without the residual, the gradient sent back to $x$ is only $0.4$; with it, $1\times(1+0.4)=1.4$.

一層看不出差別，疊很多層就很明顯 `[補充：示意]`。子層的輸入輸出都是向量，它的導數是一個矩陣（雅可比矩陣，Jacobian），倒傳遞時梯度每經過一個子層就乘上一次這個矩陣。假設每個子層的導數是一個 $64$ 維的隨機小矩陣 $\mathbf{J}_l$，會把梯度大約縮成 $0.3$ 倍；$0.3$ 是為了示意而選的，不是量測值。沒有殘差時，梯度是 $\mathbf{J}_L\cdots\mathbf{J}_1$ 連乘；有殘差時，是 $(\mathbf{I}+\mathbf{J}_L)\cdots(\mathbf{I}+\mathbf{J}_1)$ 連乘。這個示範不含層正規化，只看殘差的效果。原版 Transformer 的殘差路徑上還有層正規化，「直達項原封不動」只在第4節的 Pre-LN 才嚴格成立。

One layer shows little difference; many layers show a lot `[Added: illustrative]`. A sublayer maps a vector to a vector, so its derivative is a matrix (the Jacobian), and in backpropagation the gradient is multiplied by this matrix at every sublayer. Suppose each sublayer's derivative is a random small $64$-dimensional matrix $\mathbf{J}_l$ that shrinks the gradient to about $0.3$ times; the $0.3$ is chosen for illustration, not measured. Without residuals, the gradient is the product $\mathbf{J}_L\cdots\mathbf{J}_1$; with residuals, the product $(\mathbf{I}+\mathbf{J}_L)\cdots(\mathbf{I}+\mathbf{J}_1)$. This demonstration leaves out layer normalization and shows only the effect of the residual. In the original Transformer the residual path also passes through layer normalization, so "the direct term goes down intact" holds strictly only for the Pre-LN of Section 4.

![對數縱軸的折線圖，橫軸是梯度往回經過的子層數 0 到 24。藍線（有殘差）從 1 開始緩慢上升，在第 12 個子層約 1.77、第 24 個約 2.84；紅線（沒有殘差）從 1 一路直線下降，在第 12 個子層約 4.1 乘以 10 的負 7 次方，到第 24 個約 10 的負 13 次方。第 12 處有虛線，標示 12 個子層等於 6 層編碼器（Log-scale line chart over 0 to 24 sublayers passed back through: the blue line with residuals rises slowly from 1 to about 1.77 at sublayer 12 and 2.84 at 24; the red line without residuals falls straight down to about 4.1 times 10 to the minus 7 at sublayer 12 and about 10 to the minus 13 at 24; a dashed line at 12 marks 12 sublayers, which is 6 encoder layers）](../assets/figures/lec11-kp04-residual-gradient.png)

**圖說 `lec11-kp04-residual-gradient`**〔功能：一目了然、精準度〕：先看紅線：沒有殘差，梯度每過一個子層就縮小，經過 $12$ 個子層（$6$ 層編碼器，每層兩個子層）只剩約 $4.1\times10^{-7}$；在這個示意裡，底層幾乎學不到東西。再看藍線：有殘差，同樣的子層，梯度沒有消失，只緩慢變大，第 $12$ 個子層約 $1.77$，第 $24$ 個約 $2.84$。縱軸是對數刻度，每一格差 $100$ 倍。

**Caption `lec11-kp04-residual-gradient`** [Function: seeing it at a glance, precision]: first look at the red line: without residuals, the gradient shrinks at every sublayer, and after $12$ sublayers ($6$ encoder layers, two sublayers each) only about $4.1\times10^{-7}$ is left; in this illustration the bottom layers barely learn. Then look at the blue line: with residuals and the same sublayers, the gradient does not vanish but grows only slowly, about $1.77$ at sublayer $12$ and $2.84$ at $24$. The vertical axis is logarithmic, and each grid step is a factor of $100$.

常見錯誤：以為殘差路徑上可以隨意乘一個可學習的權重，寫成 $\mathbf{y}=w\mathbf{x}+F(\mathbf{x})$。$w$ 不是 $1$ 時，直達項變成 $w$ 的連乘，疊深了一樣會縮小或放大，直達的好處就沒了。

Common mistake: thinking the residual path may freely be multiplied by a learnable weight, $\mathbf{y}=w\mathbf{x}+F(\mathbf{x})$. When $w$ is not $1$, the direct term becomes a product of $w$'s, which shrinks or grows in a deep stack, and the benefit of the direct road is lost.

### 2. 層正規化：每個詞自己算平均（Layer normalization: each word computes its own average）

殘差把輸入一層層加上去，數值的大小可能越疊越偏。層正規化（layer normalization，LN）把**一個詞的向量**拉回平均 $0$、標準差 $1$，再讓模型自己學要的縮放與平移：

Residuals keep adding inputs layer after layer, so the scale of the numbers may drift as the stack grows. Layer normalization (LN) pulls **the vector of one word** back to mean $0$ and standard deviation $1$, then lets the model learn the scale and shift it wants:

$$
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i,\qquad \sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2,\qquad
\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad y_i=\gamma_i\hat{x}_i+\beta_i.
$$

這裡 $d=d_{\text{model}}$。$\boldsymbol{\gamma}$ 一開始設成全 $1$，$\boldsymbol{\beta}$ 設成全 $0$。

Here $d=d_{\text{model}}$. $\boldsymbol{\gamma}$ starts as all ones and $\boldsymbol{\beta}$ as all zeros.

來源的例子：$\mathbf{x}=[1,3,5]$，取 $\epsilon=0$。平均 $\mu=3$，變異數 $\sigma^2=(4+0+4)/3=2.6667$，標準差 $1.633$。所以 $\hat{\mathbf{x}}=[-1.2247,\ 0,\ 1.2247]$，平均 $0$、標準差 $1$。

The source's example: $\mathbf{x}=[1,3,5]$ with $\epsilon=0$. The mean is $\mu=3$, the variance $\sigma^2=(4+0+4)/3=2.6667$, the standard deviation $1.633$. So $\hat{\mathbf{x}}=[-1.2247,\ 0,\ 1.2247]$, with mean $0$ and standard deviation $1$.

第4週的批次正規化（batch normalization，BN）也減平均、除標準差，差在**沿哪個方向算**。把一批資料排成表格：每一列是一個詞，每一欄是一個特徵。LN 沿著列算，每個詞用自己的特徵算平均；BN 沿著欄算，每個特徵用一整批的詞算平均。

The batch normalization (BN) of Week 4 also subtracts a mean and divides by a standard deviation; the difference is **which direction it computes along**. Lay a batch out as a table: each row is a word, each column a feature. LN works along rows, each word averaging over its own features; BN works along columns, each feature averaging over the whole batch of words.

![三個 3 列 4 欄的格子。左邊灰階是輸入 X：第一列 2、4、6、8，第二列 1、1、3、3，第三列 10、0、10、0。中間是 LayerNorm，每一列用紫框框起來，結果第一列是 −1.34、−0.45、0.45、1.34，第二列 −1、−1、1、1，第三列 1、−1、1、−1。右邊是 BatchNorm，每一欄用藍框框起來，結果第一欄是 −0.58、−0.83、1.41，每一欄平均都是 0（Three 3-by-4 grids. Left, greyscale input X with rows 2, 4, 6, 8; 1, 1, 3, 3; and 10, 0, 10, 0. Middle, LayerNorm with each row boxed in purple: −1.34, −0.45, 0.45, 1.34; −1, −1, 1, 1; 1, −1, 1, −1. Right, BatchNorm with each column boxed in blue; the first column becomes −0.58, −0.83, 1.41, and every column has mean 0）](../assets/figures/lec11-kp04-ln-vs-bn.png)

**圖說 `lec11-kp04-ln-vs-bn`**〔功能：一目了然、精準度〕：先看中間：紫框一次框一列，每個詞自己變成平均 $0$、標準差 $1$，別的詞是什麼都不影響它。再看右邊：藍框一次框一欄，每個特徵在這一批裡變成平均 $0$；同一個詞的結果會隨同批的其他詞改變。例如第一列在 LN 之後是 $[-1.34,-0.45,0.45,1.34]$，在 BN 之後是 $[-0.58,1.37,-0.12,1.31]$。

**Caption `lec11-kp04-ln-vs-bn`** [Function: seeing it at a glance, precision]: first look at the middle: each purple box covers one row, so each word becomes mean $0$ and standard deviation $1$ on its own, whatever the other words are. Then look at the right: each blue box covers one column, so each feature becomes mean $0$ within this batch, and the result for a word changes with the other words in the batch. For example, the first row becomes $[-1.34,-0.45,0.45,1.34]$ after LN and $[-0.58,1.37,-0.12,1.31]$ after BN.

為什麼 Transformer 選 LN？BN 的統計量要靠一整批資料：批次小時估得不準，句子長短不一要補空位，補的空位也會混進統計；推論時 BN 改用訓練期間累積的平均，又可能和實際輸入不合。極端的例子 `[補充]`：若一批只有一個詞，而 BN 像訓練時那樣用這一批自己的統計量，每一欄的平均就是那個數本身，減完全是 $0$：第一列單獨送進去得到 $[0,0,0,0]$，送進 LN 仍是 $[-1.3416,-0.4472,0.4472,1.3416]$。LN 每個詞自給自足，訓練和推論的算法一樣，不受批次大小影響。

Why does the Transformer choose LN? BN's statistics depend on a whole batch: small batches estimate them poorly, sentences of different lengths need padding that also leaks into the statistics, and at inference BN switches to averages accumulated during training, which may not fit the actual input. An extreme example `[Added]`: if a batch holds a single word and BN uses this batch's own statistics, as it does in training, the mean of each column is that number itself, so everything becomes $0$: the first row alone gives $[0,0,0,0]$, but still $[-1.3416,-0.4472,0.4472,1.3416]$ under LN. LN lets each word stand on its own, computes the same way in training and inference, and is unaffected by the batch size.

常見錯誤：以為層正規化是對整個句子、或對整批資料算平均。它只對**一個詞的 $d_{\text{model}}$ 個分量**算；程式裡是沿最後一個維度（`dim=-1`）。

Common mistake: thinking layer normalization averages over the whole sentence or the whole batch. It averages only over **the $d_{\text{model}}$ components of one word**; in code it runs along the last dimension (`dim=-1`).

### 3. 逐位置前饋網路：每個詞自己再想一想（Position-wise feed-forward network: each word thinks it over on its own）

注意力負責「從別的詞收集資訊」。收集完之後，每個詞再各自經過一個兩層的小網路，叫做逐位置前饋網路（position-wise feed-forward network，FFN）：

Attention is responsible for "gathering information from other words". After gathering, each word passes on its own through a small two-layer network, called the position-wise feed-forward network (FFN):

$$
\operatorname{FFN}(\mathbf{x})=\max(0,\ \mathbf{x}\mathbf{W}_1+\mathbf{b}_1)\,\mathbf{W}_2+\mathbf{b}_2,\qquad
\mathbf{W}_1\in\mathbb{R}^{d_{\text{model}}\times d_{\text{ff}}},\ \mathbf{W}_2\in\mathbb{R}^{d_{\text{ff}}\times d_{\text{model}}}.
$$

第一層把 $512$ 維放大到 $d_{\text{ff}}=2048$ 維（$4$ 倍），經過 ReLU $\max(0,\cdot)$，第二層再壓回 $512$ 維。「逐位置」的意思是：句子裡每個位置用**同一組** $\mathbf{W}_1,\mathbf{W}_2$，而且各算各的，詞與詞之間不交換資訊。把整個 $n\times512$ 矩陣一次送進去，和一列一列分開送，結果完全相同 `[補充：程式驗證]`。

The first layer widens $512$ dimensions to $d_{\text{ff}}=2048$ ($4$ times), applies ReLU $\max(0,\cdot)$, and the second layer narrows back to $512$. "Position-wise" means every position in the sentence uses **the same** $\mathbf{W}_1,\mathbf{W}_2$, and each is computed separately, with no information exchanged between words. Feeding the whole $n\times512$ matrix at once gives exactly the same result as feeding it row by row `[Added: checked in code]`.

為什麼需要它？自注意力的輸出是值向量的加權平均。權重雖然由 softmax 依輸入而定，但對值向量來說，它做的是「混合」；FFN 則讓每個詞在混合之後做一次非線性的加工，形成新的特徵。來源祕笈引用 Geva 等人（2021）的一派研究，把 FFN 解讀成存放知識的鍵值記憶體。

Why is it needed? The output of self-attention is a weighted average of value vectors. The weights are set by softmax according to the input, but for the value vectors the operation is "mixing"; the FFN lets each word do one nonlinear round of processing after the mixing, forming new features. The source's professor notes cite a line of research (Geva et al., 2021) that interprets the FFN as a key-value memory storing knowledge.

FFN 很佔參數。用 KP01 的數字，一層編碼器有 $3152384$ 個參數：多頭注意力 $1050624$ 個（$33.3\%$），FFN $2099712$ 個（$66.6\%$），兩個層正規化共 $2048$ 個。也就是說，一層編碼器裡約三分之二的參數在 FFN；解碼器多了交叉注意力，FFN 佔 $2099712/4204032=49.9\%$。

The FFN holds many parameters. Using KP01's numbers, one encoder layer has $3152384$ parameters: multi-head attention $1050624$ ($33.3\%$), the FFN $2099712$ ($66.6\%$), and the two layer normalizations $2048$ together. So about two thirds of an encoder layer's parameters sit in the FFN; a decoder layer has cross-attention as well, so there the FFN holds $2099712/4204032=49.9\%$.

後來的模型常把 ReLU 換成 GELU（Gaussian error linear unit），$\operatorname{GELU}(z)=z\,\Phi(z)$，$\Phi$ 是標準常態分布的累積機率。它在負數區不完全截成 $0$，最低點約在 $z=-0.75$，值約 $-0.17$。來源的例子：$z=2$ 時 $\Phi(2)=0.97725$，$\operatorname{GELU}(2)=1.9545$，比 ReLU 的 $2$ 稍小。

Later models often replace ReLU with GELU (Gaussian error linear unit), $\operatorname{GELU}(z)=z\,\Phi(z)$, where $\Phi$ is the cumulative probability of the standard normal distribution. It does not cut negative numbers to exactly $0$; its lowest point is near $z=-0.75$, with value about $-0.17$. The source's example: at $z=2$, $\Phi(2)=0.97725$ and $\operatorname{GELU}(2)=1.9545$, slightly less than ReLU's $2$.

常見錯誤：以為 FFN 會在詞與詞之間做計算。詞與詞之間的交流全部在注意力裡完成；FFN 對每個位置各自做同樣的轉換。

Common mistake: thinking the FFN computes across words. All exchange between words happens in attention; the FFN applies the same transformation to each position separately.

### 4. 組起來：兩道 Add & Norm，疊 6 層（Putting it together: two Add & Norm steps, stacked 6 times）

原版 Transformer 的一層編碼器是兩個子層，各包一道 Add & Norm，也就是「先加殘差、再做層正規化」：

An encoder layer of the original Transformer has two sublayers, each wrapped in an Add & Norm, that is, "add the residual, then apply layer normalization":

$$
\mathbf{Z}'=\operatorname{LN}\big(\mathbf{X}+\operatorname{MHA}(\mathbf{X})\big),\qquad
\mathbf{Z}''=\operatorname{LN}\big(\mathbf{Z}'+\operatorname{FFN}(\mathbf{Z}')\big).
$$

其中 MHA 是 KP03 的多頭注意力（multi-head attention）。$\mathbf{Z}''$ 的形狀仍是 $n\times512$，直接當下一層的 $\mathbf{X}$。Transformer-Base 疊 $N=6$ 層，梯度往回要經過 $12$ 個子層，這正是第1節圖上虛線的位置。對 BERT 這類模型的探測研究（例如 Tenney 等人，2019）觀察到，靠下的層比較偏向詞形與文法，靠上的層比較偏向語意 `[補充]`；這是大致的傾向，不是設計時規定的。

Here MHA is KP03's multi-head attention. $\mathbf{Z}''$ is still $n\times512$ and serves directly as the next layer's $\mathbf{X}$. Transformer-Base stacks $N=6$ layers, so the gradient passes back through $12$ sublayers, exactly the dashed line in the figure of Section 1. Probing studies of models such as BERT (for example Tenney et al., 2019) observed that lower layers lean towards word forms and grammar and upper layers towards meaning `[Added]`; this is a rough tendency, not something fixed by design.

把 LN 放在殘差相加之後，叫 Post-LN，是原論文的作法。這樣一來，直達的殘差路徑每一層都要穿過一次 LN。Post-LN 需要學習率暖身（warmup，先用很小的學習率再慢慢加大），原論文的 $6$ 層模型就用了；沒有暖身時訓練容易發散，層數越多越明顯。後來 GPT-2 等模型多改用 Pre-LN：$\mathbf{x}+F(\operatorname{LN}(\mathbf{x}))$，LN 只放在子層的入口，殘差路徑從頭到尾不經過 LN，並在最後一層之後、輸出之前再補一個 LN。

Putting LN after the residual addition is called Post-LN, the original paper's choice. With it, the direct residual path passes through an LN at every layer. Post-LN needs a learning-rate warmup (start from a very small learning rate and raise it slowly), which the original $6$-layer model already used; without warmup, training easily diverges, more so as depth grows. Later models such as GPT-2 mostly switched to Pre-LN, $\mathbf{x}+F(\operatorname{LN}(\mathbf{x}))$: LN sits only at the entrance of the sublayer, the residual path never passes through LN from bottom to top, and one more LN is added after the last layer, before the output.

常見錯誤：以為一層編碼器只有一道 Add & Norm，或只在最上面做一次。原版每個子層後面都有一道，一層編碼器共兩道。

Common mistake: thinking an encoder layer has only one Add & Norm, or one at the very top. In the original, every sublayer is followed by one, so an encoder layer has two.

### 本單元重點整理（KP04 summary）

**一句話重點**：每個子層都寫成 $\mathbf{x}+F(\mathbf{x})$，讓梯度有一條直達底層的路；每個詞用層正規化把自己的向量拉回平均 $0$、標準差 $1$；注意力收集完資訊後，每個詞再各自經過 $512\to2048\to512$ 的前饋網路；這樣一層的形狀不變，可以疊 $6$ 層。

**One-sentence summary**: every sublayer is written as $\mathbf{x}+F(\mathbf{x})$, giving the gradient a direct road to the bottom; each word uses layer normalization to pull its own vector back to mean $0$ and standard deviation $1$; after attention gathers information, each word passes on its own through a $512\to2048\to512$ feed-forward network; a layer keeps its shape, so $6$ layers can be stacked.

**白話說明**：一封信要傳過很多層樓。殘差是一部直達電梯，原件不必每層重抄；層正規化像每層樓的音量控制，讓聲音不會越傳越大或越小；前饋網路是每個人收到大家的意見後，自己關起門來想一想。

**Plain explanation**: a letter has to pass through many floors. The residual is an express elevator, so the original need not be copied on every floor; layer normalization is a volume control on each floor, so the sound neither swells nor fades along the way; the feed-forward network is each person, after hearing everyone's opinion, closing the door to think it over alone.

**正式定義**：$\mathbf{Z}'=\operatorname{LN}(\mathbf{X}+\operatorname{MHA}(\mathbf{X}))$，$\mathbf{Z}''=\operatorname{LN}(\mathbf{Z}'+\operatorname{FFN}(\mathbf{Z}'))$，$\operatorname{LN}$ 沿每個詞的 $d_{\text{model}}$ 個分量正規化，$\operatorname{FFN}(\mathbf{x})=\max(0,\mathbf{x}\mathbf{W}_1+\mathbf{b}_1)\mathbf{W}_2+\mathbf{b}_2$。

**Formal definition**: $\mathbf{Z}'=\operatorname{LN}(\mathbf{X}+\operatorname{MHA}(\mathbf{X}))$, $\mathbf{Z}''=\operatorname{LN}(\mathbf{Z}'+\operatorname{FFN}(\mathbf{Z}'))$, where $\operatorname{LN}$ normalizes over the $d_{\text{model}}$ components of each word and $\operatorname{FFN}(\mathbf{x})=\max(0,\mathbf{x}\mathbf{W}_1+\mathbf{b}_1)\mathbf{W}_2+\mathbf{b}_2$.

**例子**：殘差把梯度從 $0.4$ 變成 $1.4$；隨機示意中，經過 $12$ 個子層，沒有殘差剩 $4.1\times10^{-7}$，有殘差約 $1.77$。$[1,3,5]$ 經 LN 變成 $[-1.2247,0,1.2247]$。一層編碼器 $3152384$ 個參數中，FFN 佔 $66.6\%$（一層解碼器是 $49.9\%$）。

**Example**: the residual turns a gradient of $0.4$ into $1.4$; in the random illustration, after $12$ sublayers, $4.1\times10^{-7}$ is left without residuals and about $1.77$ with them. $[1,3,5]$ becomes $[-1.2247,0,1.2247]$ after LN. Of the $3152384$ parameters in an encoder layer, the FFN holds $66.6\%$ (in a decoder layer, $49.9\%$).

**常見迷思**：

- 「LN 是對整批或整句算平均。」錯，只對一個詞的分量算。
- 「FFN 在詞與詞之間計算。」錯，詞與詞的交流只在注意力裡。
- 「殘差路徑上可以乘任意權重。」錯，不是 $1$ 就失去直達的好處。

**Common misconceptions**:

- "LN averages over the batch or the sentence." Wrong: only over the components of one word.
- "The FFN computes across words." Wrong: words exchange information only in attention.
- "The residual path may carry any weight." Wrong: a weight other than $1$ loses the direct road.

**圖**：本單元有兩張圖。

- `lec11-kp04-residual-gradient`：有無殘差時，梯度大小隨子層數的變化。
- `lec11-kp04-ln-vs-bn`：層正規化沿列、批次正規化沿欄。

**Figures** — this unit has two figures.

- `lec11-kp04-residual-gradient`: how the gradient size changes with the number of sublayers, with and without residuals.
- `lec11-kp04-ln-vs-bn`: layer normalization along rows, batch normalization along columns.

### 小練習（Practice）

1. 子層 $F(x)=x^2/4$，在 $x=1$。上面傳下來 $\partial\mathcal{L}/\partial y=1$。分別算出有殘差與沒有殘差時，傳回 $x$ 的梯度。
2. 算出 $\mathbf{x}=[2,2,6,6]$ 的層正規化 $\hat{\mathbf{x}}$（$\epsilon=0$）；若 $\boldsymbol{\gamma}$ 全是 $2$、$\boldsymbol{\beta}$ 全是 $1$，輸出 $\mathbf{y}$ 是多少？
3. $d_{\text{model}}=768$、$d_{\text{ff}}=3072$。FFN（含偏差）有幾個參數？

1. The sublayer is $F(x)=x^2/4$ at $x=1$, and $\partial\mathcal{L}/\partial y=1$ comes from above. Compute the gradient sent back to $x$ with and without the residual.
2. Compute the layer normalization $\hat{\mathbf{x}}$ of $\mathbf{x}=[2,2,6,6]$ ($\epsilon=0$); if $\boldsymbol{\gamma}$ is all $2$ and $\boldsymbol{\beta}$ is all $1$, what is the output $\mathbf{y}$?
3. With $d_{\text{model}}=768$ and $d_{\text{ff}}=3072$, how many parameters (with biases) does the FFN have?

**第 1 題解答** `[補充]`：$F'(x)=x/2=0.5$。沒有殘差：$0.5$；有殘差：$1+0.5=1.5$。

**Solution to question 1** `[Added]`: $F'(x)=x/2=0.5$. Without the residual: $0.5$; with it: $1+0.5=1.5$.

**第 2 題解答** `[補充]`：$\mu=4$，$\sigma^2=(4+4+4+4)/4=4$，$\sigma=2$，$\hat{\mathbf{x}}=[-1,-1,1,1]$；$\mathbf{y}=2\hat{\mathbf{x}}+1=[-1,-1,3,3]$。

**Solution to question 2** `[Added]`: $\mu=4$, $\sigma^2=(4+4+4+4)/4=4$, $\sigma=2$, $\hat{\mathbf{x}}=[-1,-1,1,1]$; $\mathbf{y}=2\hat{\mathbf{x}}+1=[-1,-1,3,3]$.

**第 3 題解答** `[補充]`：$768\times3072+3072+3072\times768+768=4722432$。

**Solution to question 3** `[Added]`: $768\times3072+3072+3072\times768+768=4722432$.

### 小測驗（Quiz，來源原題）

來源這一段有四題自我檢測，保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The source has four self-check questions for this part, kept with all their options; the source page does not mark the answers, so the explanations below are judged from the source text.

1. Layer Normalization（層正規化）的統計量（均值與變異數）是在哪個維度上計算的？
   - (A) 整個批次（Batch）的所有樣本間
   - (B) 單個樣本內部的所有特徵維度（Hidden Units）間
   - (C) 整個訓練資料集的所有圖片間
   - (D) 時間軸上的所有未來時間步間
2. 在 Transformer 中引入殘差連接 $\mathbf{Z}_t+\mathbf{X}$ 的核心作用是什麼？
   - (A) 增加模型的非線性能力
   - (B) 提供梯度直通捷徑，防止深層網路訓練時發生梯度消失
   - (C) 將特徵維度自動減半
   - (D) 強制使輸入特徵變為正數
3. Transformer 的 Position-wise FFN 是如何處理序列中各個位置的？
   - (A) 將所有位置的所有詞混在一起做卷積
   - (B) 對序列中每個位置的向量獨立、平行施加相同的兩層線性與激活轉換
   - (C) 依時間順序逐字遞迴處理
   - (D) 只處理句子的第一個詞與最後一個詞
4. 標準 Transformer Encoder 內部（每一層）包含幾道殘差連接與層正規化（Add & Norm）？
   - (A) 0 道
   - (B) 2 道（自注意力後一道、FFN 後一道）
   - (C) 10 道
   - (D) 僅在輸入處有 1 道

1. Along which dimension are the statistics (mean and variance) of layer normalization computed?
   - (A) Across all samples of the whole batch
   - (B) Across all feature dimensions (hidden units) within a single sample
   - (C) Across all images of the whole training set
   - (D) Across all future time steps on the time axis
2. What is the core role of the residual connection $\mathbf{Z}_t+\mathbf{X}$ in the Transformer?
   - (A) To add nonlinear capacity to the model
   - (B) To provide a direct shortcut for the gradient, preventing vanishing gradients when training deep networks
   - (C) To halve the feature dimension automatically
   - (D) To force the input features to be positive
3. How does the Transformer's position-wise FFN process each position of the sequence?
   - (A) It mixes all words of all positions together in a convolution
   - (B) It applies the same two-layer linear-and-activation transformation to the vector at each position, independently and in parallel
   - (C) It processes word by word recurrently in time order
   - (D) It processes only the first and last words of the sentence
4. How many residual-plus-layer-normalization steps (Add & Norm) does a standard Transformer encoder layer contain?
   - (A) 0
   - (B) 2 (one after self-attention, one after the FFN)
   - (C) 10
   - (D) Only 1, at the input

**解析** `[補充]`：四題都選 (B)。第 1 題見第2節，LN 沿一個詞的特徵算；第 2 題見第1節，$\mathbf{I}+\partial F/\partial\mathbf{x}$ 的 $\mathbf{I}$ 讓梯度直達；第 3 題見第3節；第 4 題見第4節的兩個式子。題中的 $\mathbf{Z}_t$ 就是多頭注意力的輸出 $\operatorname{MHA}(\mathbf{X})$。

**Explanations** `[Added]`: all four are B. Question 1, see Section 2: LN works over the features of one word; question 2, see Section 1: the $\mathbf{I}$ in $\mathbf{I}+\partial F/\partial\mathbf{x}$ gives the gradient a direct road; question 3, see Section 3; question 4, see the two formulas of Section 4. The $\mathbf{Z}_t$ in the questions is the multi-head attention output $\operatorname{MHA}(\mathbf{X})$.

### 留給下一單元的問題（The question for the next unit）

到這裡，編碼器的一層已經完整了：多頭注意力、前饋網路、兩道 Add & Norm，疊 $6$ 層。可是翻譯還要一個會「寫」的解碼器。它要一邊看編碼器讀到的原文，一邊一個字一個字寫出譯文，而且訓練時不能偷看還沒寫到的答案。解碼器的遮罩與交叉注意力怎麼做到這些？下一單元（本週 KP05）回答。

At this point an encoder layer is complete: multi-head attention, a feed-forward network and two Add & Norm steps, stacked $6$ times. But translation also needs a decoder that "writes". It must look at the source sentence the encoder has read while writing the translation word by word, and during training it must not peek at answers it has not reached yet. How do the decoder's mask and cross-attention achieve this? The next unit (this week's KP05) answers.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第9章「注意力機制與 Transformer 核心原理」§3.2.3–§3.2.5 與教授祕笈「層歸一化」「殘差連接與梯度傳播」「前饋網路與非線性活化」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §3.2.3–§3.2.5 of Chapter 9, "Core principles of attention mechanisms and the Transformer", of CCC-LNS IDL_115, and from the professor's notes "Layer normalization", "Residual connections and gradient flow" and "Feed-forward networks and nonlinear activation" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源正文的標準差寫成 $\sigma=\sqrt{\sum_i(h_i-\mu)^2}$，少了 $1/H$；照這個式子，$[1,3,5]$ 的 $\sigma$ 會是 $2.8284$ 而不是 $1.633$。本講義依祕笈寫成 $\sigma^2=\frac1d\sum_i(x_i-\mu)^2$。來源用粗斜體 `\boldsymbol`，本課程網站不支援，改用 `\mathbf`。
- 祕笈說「純自注意力運算本質上只是一連串的線性加權組合，若無非線性 FFN 堆疊，整個深層網路會退化為單一的線性映射」。softmax 的權重依輸入而定，自注意力本身並不是線性映射；本講義改說 FFN 讓每個詞在混合之後做一次非線性的加工。
- 祕笈說 FFN 佔「整個模型超過 $66\%$」的參數。照 KP01 的算法，這是**一層編碼器**裡的比例（$66.6\%$）；把解碼器的交叉注意力和 $37000\times512$ 的詞嵌入算進去，$12$ 層 FFN 共 $25196544$ 個，約佔全模型 $63082496$ 個的 $39.9\%$。
- 祕笈寫 $\operatorname{GELU}(2)\approx1.9544$，是用 $\Phi(2)\approx0.9772$ 四捨五入的結果；本講義用 $0.97725$，得 $1.9545$。
- 來源說「底層專注局部詞法與語法，高層提煉抽象語意」；本講義改寫成研究觀察到的大致傾向。
- 本講義補上：$12$ 個子層的梯度示意與圖、LN 與 BN 的同一批數字對照與圖、單一個詞時 BN 全為 $0$ 的例子、一層參數的拆分、三題小練習與解答。
- 取捨：RMSNorm、SwiGLU、FFN 作為鍵值記憶體的細節、內部協變數偏移的爭議、常微分方程與量子力學的類比，本講義不寫或只帶過。

- The source's main text writes the standard deviation as $\sigma=\sqrt{\sum_i(h_i-\mu)^2}$, missing $1/H$; by that formula, $\sigma$ of $[1,3,5]$ would be $2.8284$ rather than $1.633$. Following the professor's notes, this note writes $\sigma^2=\frac1d\sum_i(x_i-\mu)^2$. The source uses bold italic `\boldsymbol`, which the course website does not support, so this note uses `\mathbf`.
- The professor's notes say "pure self-attention is essentially just a series of linear weighted combinations; without stacked nonlinear FFNs, the whole deep network collapses into a single linear map". Since softmax weights depend on the input, self-attention itself is not a linear map; this note says instead that the FFN lets each word do one nonlinear round of processing after the mixing.
- The professor's notes say the FFN holds "more than $66\%$ of the whole model's" parameters. By KP01's count this is the share within **one encoder layer** ($66.6\%$); counting the decoder's cross-attention and the $37000\times512$ word embedding, the FFNs of all $12$ layers hold $25196544$ parameters, about $39.9\%$ of the model's $63082496$.
- The professor's notes give $\operatorname{GELU}(2)\approx1.9544$, from rounding $\Phi(2)\approx0.9772$; this note uses $0.97725$ and gets $1.9545$.
- The source says "lower layers focus on local word forms and grammar, upper layers distil abstract meaning"; this note rewrites it as a rough tendency that studies have observed.
- This note adds: the gradient illustration over $12$ sublayers with its figure, the LN-versus-BN comparison on the same batch with its figure, the example of BN giving all zeros for a single word, the split of one layer's parameters, and three practice questions with solutions.
- Choices: RMSNorm, SwiGLU, the details of the FFN as key-value memory, the debate over internal covariate shift, and the analogies with ordinary differential equations and quantum mechanics are left out or only mentioned.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 殘差連接 | residual connection |
| 殘差網路 | residual network (ResNet) |
| 單位矩陣 | identity matrix |
| 層正規化 | layer normalization (LN, LayerNorm) |
| 批次正規化（第4週） | batch normalization (BN, BatchNorm) |
| 可學習的縮放與平移 | learnable scale and shift |
| 逐位置前饋網路 | position-wise feed-forward network (FFN) |
| 線性整流函數 | rectified linear unit (ReLU) |
| 高斯誤差線性單元 | Gaussian error linear unit (GELU) |
| 累積機率 | cumulative probability |
| 先加後正規化 | Post-LN |
| 先正規化 | Pre-LN |
| 學習率暖身 | learning-rate warmup |
| 均方根正規化 | root mean square normalization (RMSNorm) |
| 內部協變數偏移 | internal covariate shift |
| 門控線性單元的一種 | SwiGLU (Swish-gated linear unit) |
| 生成式預訓練模型（本週 KP06） | GPT (generative pre-trained Transformer) |
