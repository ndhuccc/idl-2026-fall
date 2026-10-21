# 第4週 KP12：批次正規化（Batch normalization）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第4週「正則化與過擬合」的第十二個知識點（KP12），也是這一章的最後一個。承接 KP11 留下的問題：Dropout 是在訓練時隨機改變網路的結構，另一類問題出在訓練本身：每一層的輸入會隨著前面各層的權重一起改變，訓練起來很不穩定，能不能在層與層之間直接把輸入穩住？先備知識是平均數與變異數（KP02）、mini-batch（第3週 KP05）；第一次出現的符號都會定義，所有數字都可以自己重算。內容依 CCC-LNS 第5章 KP16 主文與教授祕笈（其中批次正規化與章節總結的部分）消化後重新撰寫，讀取日期 2026-10-20；來源沒有的內容標示 `[補充]`。最後一節把整章的方法放進一張表，並把問題交給下一章。

This is the twelfth knowledge point (KP12) of Week 4, "Regularization and overfitting", and the last of the chapter. It picks up the question KP11 left open: dropout changes the structure of the network at random during training, but another kind of trouble lies in training itself: the input of every layer changes together with the weights of the layers before it, and training becomes unstable. Can we steady the input directly between layers? It assumes mean and variance (KP02) and the mini-batch (Week 3, KP05); every symbol is defined when it first appears and every number can be recomputed. The text is rewritten from the KP16 main text and professor's guide of CCC-LNS Chapter 5 (the batch normalization and chapter summary parts), accessed on 2026-10-20; content that the source does not contain is marked `[Added]`. The last section puts all the methods of the chapter into one table and hands the next question to the next chapter.

[下載本單元投影片](../slides/lec04-kp12-batch-normalization.jbd.json)

## KP12 批次正規化（Batch normalization）

### 學習目標（Learning objectives）

讀完之後，你能對一個 mini-batch 的某個特徵，一步一步算出平均 $\mu_j$、變異數 $\sigma_j^2$ 與標準化後的 $\hat x_{ij}$，並說明 $\varepsilon$ 的作用；能說明為什麼標準化之後還要有可學習的 $\gamma$、$\beta$，以及測試時為什麼改用訓練時累積的移動平均；能指出批次正規化只固定平均與變異數、不改變分布的形狀，並說出它有效的主要原因（損失地形變平滑）與不能依賴的舊解釋；也能說明批次太小時它的問題；最後能把本章所有方法放進同一張表，說出它們各控制什麼。

After reading, you can compute step by step, for one feature of a mini-batch, the mean $\mu_j$, the variance $\sigma_j^2$ and the normalized $\hat x_{ij}$, and explain what $\varepsilon$ does; explain why learnable $\gamma$ and $\beta$ follow the normalization and why the moving averages gathered in training are used at test time; point out that batch normalization fixes only the mean and variance and does not change the shape of the distribution, state the main reason it works (a smoother loss landscape) and the older explanation that should not be relied on; explain what goes wrong with very small batches; and finally put all the methods of the chapter into one table and say what each of them controls.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $b$ | mini-batch 的大小（一批有幾筆資料） | size of the mini-batch (data points per batch) |
| $x_{ij}$ | 這一批第 $i$ 筆資料的第 $j$ 個特徵（或第 $j$ 個神經元的輸入） | feature $j$ (or the input of neuron $j$) of the $i$-th data point in this batch |
| $\mu_j$ | 這一批、特徵 $j$ 的平均 | mean of feature $j$ over this batch |
| $\sigma_j^2$ | 這一批、特徵 $j$ 的變異數（除以 $b$） | variance of feature $j$ over this batch (divided by $b$) |
| $\varepsilon$ | 很小的正常數，例如 $10^{-5}$，避免除以 0 | a small positive constant, e.g. $10^{-5}$, to avoid division by 0 |
| $\hat x_{ij}$ | 標準化後的值 | the normalized value |
| $\gamma_j,\ \beta_j$ | 可學習的尺度與位移（`[補充]`） | learnable scale and shift (`[Added]`) |

### 1. 對每個特徵做標準化（Standardizing every feature）

KP11 的 dropout 動的是「哪些神經元在場」。這一單元動的是「進到每一層的數值長什麼樣」。訓練時，前面各層的權重每更新一次，後面某一層看到的輸入分布就變一次，就像考試的難度每天都在變，很難穩定地學。批次正規化的做法很直接：對每個 mini-batch、每一個特徵（或每個神經元的輸入）各自標準化成平均 0、變異數 1。

The dropout of KP11 changed "which neurons are present". This unit changes "what the numbers entering each layer look like". During training, every time the weights of earlier layers are updated, the distribution of the input seen by a later layer shifts, like an exam whose difficulty changes every day, which is hard to learn from steadily. Batch normalization does the direct thing: for each mini-batch and each feature (or the input of each neuron), standardize it to mean 0 and variance 1.

設一個 mini-batch 有 $b$ 筆資料，第 $i$ 筆的第 $j$ 個特徵是 $x_{ij}$。先算這一批的平均與變異數，再標準化：

Let a mini-batch have $b$ data points and let $x_{ij}$ be feature $j$ of the $i$-th point. First compute the mean and the variance over this batch, then standardize:

$$
\mu_j=\frac1b\sum_{i=1}^{b}x_{ij},\qquad \sigma_j^2=\frac1b\sum_{i=1}^{b}\big(x_{ij}-\mu_j\big)^2
$$

$$
\hat x_{ij}=\frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\varepsilon}}
$$

每個特徵 $j$ 各算各的 $\mu_j$ 與 $\sigma_j^2$，彼此不相干。變異數除以 $b$（不是 $b-1$），這是來源的寫法，也是 KP02 的母體變異數。$\varepsilon$ 的作用只有一個：如果某個特徵在這一批裡每筆都一樣，$\sigma_j^2=0$，分母就是 0 而爆掉；加一個很小的 $\varepsilon$ 可以避免這件事，在一般情形下它對結果的影響小到幾乎看不見。

Each feature $j$ has its own $\mu_j$ and $\sigma_j^2$, independent of the others. The variance divides by $b$ (not $b-1$), as the source writes it and as in the population variance of KP02. The role of $\varepsilon$ is just one: if a feature is the same for every point of this batch, $\sigma_j^2=0$ and the denominator is 0 and blows up; a tiny $\varepsilon$ prevents that, and in ordinary cases its effect is almost invisible.

**數值例**：$b=4$，某個特徵的四個值是 $2,4,6,8$。平均 $\mu=\frac{2+4+6+8}{4}=5$。離差是 $-3,-1,1,3$，平方是 $9,1,1,9$，加起來 $20$，所以 $\sigma^2=\frac{20}{4}=5$。取 $\varepsilon=10^{-5}$：$\sqrt{5+10^{-5}}=2.236070$。標準化後：

**Numerical example**: $b=4$ and the four values of a feature are $2,4,6,8$. The mean is $\mu=\frac{2+4+6+8}{4}=5$. The deviations are $-3,-1,1,3$, their squares $9,1,1,9$, sum $20$, so $\sigma^2=\frac{20}{4}=5$. Take $\varepsilon=10^{-5}$: $\sqrt{5+10^{-5}}=2.236070$. After standardization:

$$
\hat x=\Big(\frac{-3}{2.236070},\ \frac{-1}{2.236070},\ \frac{1}{2.236070},\ \frac{3}{2.236070}\Big)=(-1.341639,\ -0.447213,\ 0.447213,\ 1.341639)
$$

若 $\varepsilon=0$，值是 $\pm1.341641,\ \pm0.447214$，最大差距只有 $1.3\times10^{-6}$，$\varepsilon$ 只動到第六位小數〔程式計算〕。新的平均是 $0$；新的變異數是 $\frac{\sigma^2}{\sigma^2+\varepsilon}=0.999998$，略小於 $1$，也來自 $\varepsilon$。

With $\varepsilon=0$ the values are $\pm1.341641,\ \pm0.447214$ and the largest gap is only $1.3\times10^{-6}$: $\varepsilon$ moves only the sixth decimal [computed by program]. The new mean is $0$; the new variance is $\frac{\sigma^2}{\sigma^2+\varepsilon}=0.999998$, slightly below $1$, again because of $\varepsilon$.

標準化還帶來一個好處：特徵的單位與大小不再重要。把第二個特徵取成 $x_2=100\times(2,4,6,8)=(200,400,600,800)$，它的平均是 $500$，變異數是 $50000$，標準化後的值和上面完全一樣，最大差距 $1.3\times10^{-6}$〔程式計算〕。尺度差一百倍的兩個特徵，進到下一層的時候一樣大。

Standardization has another benefit: the unit and size of a feature no longer matter. Take the second feature $x_2=100\times(2,4,6,8)=(200,400,600,800)$; its mean is $500$, its variance $50000$, and its standardized values are the same as above, with a largest gap of $1.3\times10^{-6}$ [computed by program]. Two features that differ in scale by a factor of 100 are equally large when they enter the next layer.

**但它只固定平均與變異數，不改變形狀。** 對一個偏斜的特徵，例如 $1,1,2,2,3,10$（$b=6$）：平均 $3.1667$、變異數 $9.8056$，標準化後是 $(-0.6919,-0.6919,-0.3726,-0.3726,-0.0532,\,2.1822)$，偏度（skewness）標準化前 $1.6043$、後也是 $1.6043$〔程式計算〕。右尾的離群值依然在，只是被搬到平均 $0$、變異數 $1$ 的位置。

**But it only fixes the mean and the variance, and does not change the shape.** Take a skewed feature such as $1,1,2,2,3,10$ ($b=6$): the mean is $3.1667$, the variance $9.8056$, and after standardization $(-0.6919,-0.6919,-0.3726,-0.3726,-0.0532,\,2.1822)$; the skewness is $1.6043$ before and $1.6043$ after [computed by program]. The outlier on the right tail is still there, only moved to mean $0$ and variance $1$.

![兩個並排直方圖：左邊是標準化前的偏斜特徵（1000 筆，平均 2.15、標準差 1.91，右側有長尾，範圍約 0 到 14）；右邊是標準化後（平均 0、標準差 1，範圍約 −1 到 6），形狀與左邊完全相同，只是橫軸被搬移與縮放。（Two histograms side by side: on the left a skewed feature before standardization (1000 points, mean 2.15, standard deviation 1.91, a long right tail, range about 0 to 14); on the right after standardization (mean 0, standard deviation 1, range about −1 to 6), with exactly the same shape and only the horizontal axis shifted and scaled.）](../assets/figures/lec04-kp12-hist-bn.png)

**圖說 `lec04-kp12-hist-bn`**：先看標題的兩組數字：左圖平均 $2.15$、標準差 $1.91$，右圖是 $0$ 與 $1$，這就是批次正規化做的全部。再比較兩個直方圖的外形：左右完全相同，一樣的高峰、一樣的長尾，偏度都是 $2.7898$（1000 筆的對數常態資料，固定種子，程式計算）。批次正規化搬動與縮放資料，不會把偏斜的分布變成對稱。〔功能：1、2〕

**Caption `lec04-kp12-hist-bn`**: Look first at the two sets of numbers in the titles: the left panel has mean $2.15$ and standard deviation $1.91$, the right panel has $0$ and $1$; that is all batch normalization does. Then compare the shapes: left and right are identical, the same peak and the same long tail, with skewness $2.7898$ in both (1000 log-normal points, fixed seed, computed by program). Batch normalization shifts and rescales the data and does not turn a skewed distribution into a symmetric one. [Function: 1, 2]

### 2. 可學習的尺度、位移，以及測試時怎麼辦（Learnable scale and shift, and what to do at test time）

上一節的標準化有一個副作用：它強迫每一層的輸入都是平均 $0$、變異數 $1$，但這未必是這一層最需要的分布。例如接在 sigmoid 前面，輸入集中在 $0$ 附近會讓函數接近線性。實務上的批次正規化在標準化之後再接一個**可學習的線性變換**（來源沒有寫這一步，`[補充]`）：

The standardization of the last section has a side effect: it forces the input of every layer to have mean $0$ and variance $1$, which is not necessarily the distribution this layer needs most. For instance, before a sigmoid, an input concentrated near $0$ keeps the function close to linear. In practice batch normalization follows the standardization with a **learnable linear transformation** (the source does not write this step, `[Added]`):

$$
y_{ij}=\gamma_j\,\hat x_{ij}+\beta_j
$$

$\gamma_j$（尺度）與 $\beta_j$（位移）跟其他權重一樣由梯度下降學出來。若 $\gamma_j=\sqrt{\sigma_j^2+\varepsilon}$ 且 $\beta_j=\mu_j$，這一步恰好還原原本的輸入，所以加上它並沒有減少網路能表達的函數，只是把「這一層輸入的平均與尺度」變成可以直接學的兩個參數。數值例：上面 $2,4,6,8$ 取 $\gamma=2$、$\beta=1$，得到 $y=(-1.6833,\ 0.1056,\ 1.8944,\ 3.6833)$，平均 $1$，標準差 $2$〔程式計算〕。

$\gamma_j$ (scale) and $\beta_j$ (shift) are learned by gradient descent like any other weight. If $\gamma_j=\sqrt{\sigma_j^2+\varepsilon}$ and $\beta_j=\mu_j$, this step restores the original input exactly, so adding it does not reduce the functions the network can express; it only turns "the mean and the scale of the input of this layer" into two parameters that can be learned directly. Numerical example: take $\gamma=2$ and $\beta=1$ on the $2,4,6,8$ above and get $y=(-1.6833,\ 0.1056,\ 1.8944,\ 3.6833)$ with mean $1$ and standard deviation $2$ [computed by program].

**測試時怎麼辦？**（`[補充]`）訓練時 $\mu_j,\sigma_j^2$ 來自當下那一批；測試時我們可能一次只預測一筆，一筆算不出有意義的批次統計，而且同一筆資料不應該因為跟誰同一批而得到不同的答案。常見做法是訓練時順便維護 $\mu_j$ 與 $\sigma_j^2$ 的**移動平均**（每一批用 $\bar\mu\leftarrow(1-m)\bar\mu+m\,\mu_{\text{batch}}$ 更新一點，$m$ 是小的動量常數），測試時改用這組固定的值。這和 dropout 一樣，訓練與測試的行為不同：在 PyTorch 中同樣由 `model.train()` 與 `model.eval()` 切換。

**What happens at test time?** (`[Added]`) In training $\mu_j,\sigma_j^2$ come from the current batch; at test time we may predict a single point at a time, and one point gives no meaningful batch statistic, and the same data point should not get a different answer depending on who else is in its batch. The usual remedy is to keep **moving averages** of $\mu_j$ and $\sigma_j^2$ during training (each batch nudges them with $\bar\mu\leftarrow(1-m)\bar\mu+m\,\mu_{\text{batch}}$, where $m$ is a small momentum constant) and to use these fixed values at test time. As with dropout the behaviour differs between training and testing, and in PyTorch it is switched by `model.train()` and `model.eval()` in the same way.

### 3. 為什麼有效，與何時出問題（Why it works, and when it goes wrong）

**為什麼有效。** 早期的解釋是「減輕 covariate shift」（各層輸入的分布隨訓練漂移），這也是第 1 節開頭那個直覺的來源。來源指出：後續的研究顯示這個解釋並不正確，批次正規化的主要好處來自它讓**損失地形更平滑**：損失對參數的變化更緩和，梯度更可預測，所以可以用較大的學習率、訓練更穩定。這個結論出自 Santurkar 等人在 NeurIPS 2018 的研究（`[補充]`）。我們不在這裡重做那個分析，只記住：不要把「穩住輸入分布」當成它有效的原因。

**Why it works.** The early explanation was "reducing covariate shift" (the distributions of layer inputs drift during training), and it is also the source of the intuition at the start of section 1. The source points out that later work showed this explanation to be incorrect, and that the main benefit of batch normalization comes from making the **loss landscape smoother**: the loss changes more gently with the parameters and the gradients are more predictable, so larger learning rates can be used and training is more stable. This conclusion comes from the study of Santurkar et al. at NeurIPS 2018 (`[Added]`). We do not redo that analysis here; just remember not to treat "steadying the input distribution" as the reason it works.

**輕微的正則化效果。** 批次的平均與變異數是從這一批的資料估出來的，每一批都不同，等於替每一層的輸入加了一點雜訊。雜訊有多大？用一個 $N(0,1)$ 的特徵，每個批次大小各抽 5000 批，算批次標準差與批次平均的變化〔程式計算〕：批次平均的標準差是 $\frac1{\sqrt b}$（$b=4$：$0.4912$；$b=16$：$0.2482$；$b=64$：$0.1270$；$b=256$：$0.0622$）；批次標準差的平均與真值 $1$ 也有落差（$b=4$：$0.7878$；$b=16$：$0.9539$；$b=256$：$0.9964$），而且批次小的時候低估。批次愈小雜訊愈大，正則化效果也愈明顯，但這個效果不強，不能當成 dropout 的替代品。經驗上，使用批次正規化之後，dropout 常常可以調小或不用（`[補充]`，常見做法，不是定理）。

**A mild regularizing effect.** The mean and variance of a batch are estimated from the data of that batch and differ from batch to batch, which adds a little noise to the input of each layer. How large is the noise? Take an $N(0,1)$ feature, draw 5000 batches for each batch size and look at how the batch standard deviation and the batch mean vary [computed by program]: the standard deviation of the batch mean is $\frac1{\sqrt b}$ ($b=4$: $0.4912$; $b=16$: $0.2482$; $b=64$: $0.1270$; $b=256$: $0.0622$); the mean of the batch standard deviation also falls short of the true $1$ ($b=4$: $0.7878$; $b=16$: $0.9539$; $b=256$: $0.9964$), underestimating when the batch is small. The smaller the batch, the larger the noise and the clearer the regularizing effect, but the effect is not strong and is no substitute for dropout. In experience, after batch normalization is used dropout can often be turned down or left out (`[Added]`, common practice and not a theorem).

![折線圖：橫軸是批次大小 b（2 到 256，對數刻度），縱軸是對 N(0,1) 特徵的估計。藍線是批次標準差的平均，從 b 等於 2 的 0.57 上升並趨近真值 1，淺藍帶是正負一個標準差，在小批次時很寬；橘線是批次平均的標準差，從 0.70 降到 0.06。（Line chart: the horizontal axis is the batch size b (2 to 256, logarithmic scale) and the vertical axis the estimates for an N(0,1) feature. The blue line is the mean of the batch standard deviation, rising from 0.57 at b equal to 2 toward the true value 1, with a pale-blue band of plus and minus one standard deviation that is wide for small batches; the orange line is the standard deviation of the batch mean, falling from 0.70 to 0.06.）](../assets/figures/lec04-kp12-batch-noise.png)

**圖說 `lec04-kp12-batch-noise`**：先看藍線與灰色虛線（真值 $1$）的距離：$b=2$ 時平均只有 $0.57$，而且藍帶很寬，同一個特徵每批算出的標準差可以差很多；再看橘線：批次平均的抖動隨 $b$ 變大而縮小，與 $\frac1{\sqrt b}$ 一致。批次愈大，兩條線都愈穩。〔功能：1、2〕

**Caption `lec04-kp12-batch-noise`**: Look first at the distance between the blue line and the grey dotted line (the true value $1$): at $b=2$ the mean is only $0.57$ and the blue band is wide, so the standard deviation computed for the same feature can differ a lot from batch to batch; then look at the orange line: the wobble of the batch mean shrinks as $b$ grows, in agreement with $\frac1{\sqrt b}$. The larger the batch, the steadier both lines. [Function: 1, 2]

**何時出問題。** 批次太小時，批次統計不可靠（上圖）。極端情形 $b=1$：只有一筆資料，平均就是它自己，$\sigma^2=0$，$\hat x=\frac{x-x}{\sqrt{0+\varepsilon}}=0$，每筆資料都被標準化成 $0$，資訊全部消失〔程式計算〕。所以批次正規化需要夠大的批次（一般至少幾十筆）；批次很小或不方便分批的情形，要改用其他不依賴批次的正規化方法，但這超出本單元。

**When it goes wrong.** With a very small batch the batch statistics are unreliable (the figure above). In the extreme case $b=1$: there is a single data point, the mean is that point itself, $\sigma^2=0$, $\hat x=\frac{x-x}{\sqrt{0+\varepsilon}}=0$, every data point is standardized to $0$ and all information disappears [computed by program]. So batch normalization needs a reasonably large batch (generally tens of points at least); for very small batches or situations where batching is awkward one should use other normalization methods that do not depend on the batch, but that lies beyond this unit.

### 4. 全章回顧：每一種方法在控制什麼（Chapter review: what each method controls）

批次正規化是這一章的最後一個方法。回頭看整章，所有東西圍繞著同一個問題：模型在訓練資料上表現好，不代表在新資料上也好。我們一路上做了這幾件事：

Batch normalization is the last method of this chapter. Looking back over the chapter, everything revolves around one question: doing well on the training data does not mean doing well on new data. Along the way we did the following:

- KP01–KP06：誠實地評估與認識過擬合。資料切分、不作弊；把誤差拆成偏差與變異，知道任何模型都有雜訊地板 $\sigma^2$；訓練誤差會高估模型的表現；過擬合、欠擬合與交叉驗證。
- KP07–KP10：直接壓低複雜度的正則化方法：稀疏與正則化的概念、$\ell_2$（含權重衰減）、$\ell_1$、早停法。
- KP11–KP12：隨機化與穩定訓練：dropout（隨機關神經元，近似許多子網路的平均）與批次正規化（穩定每層的輸入）。

- KP01–KP06: evaluate honestly and understand overfitting. Data splits, no cheating; split the error into bias and variance and know that every model has a noise floor $\sigma^2$; the training error overestimates how well the model does; overfitting, underfitting and cross-validation.
- KP07–KP10: regularization methods that lower the complexity directly: the idea of sparsity and regularization, $\ell_2$ (with weight decay), $\ell_1$, early stopping.
- KP11–KP12: randomization and steadier training: dropout (switch neurons off at random, approximating an average of many sub-networks) and batch normalization (steady the input of every layer).

下表把每一種方法放在同一個框架裡：它控制什麼、作用在哪裡。

The table below puts every method in the same frame: what it controls and where it acts.

| 方法 | 控制什麼 | 作用在哪裡 | 單元 |
|---|---|---|---|
| 資料切分、交叉驗證 | 誠實地評估，不改模型 | 資料 | KP01、KP06 |
| $\ell_2$ 與權重衰減 | 縮小權重，降低有效複雜度 | 目標函數 | KP07、KP08 |
| $\ell_1$ | 產生稀疏，降低有效複雜度 | 目標函數 | KP09 |
| 早停法 | 少訓練，約等於 $\ell_2$ | 訓練過程 | KP10 |
| Dropout | 降低變異（近似子網路平均） | 隱藏層 | KP11 |
| 批次正規化 | 穩定、平滑最佳化；輕微正則化 | 層與層之間 | KP12 |

| Method | What it controls | Where it acts | Unit |
|---|---|---|---|
| data split, cross-validation | honest evaluation, the model is unchanged | data | KP01, KP06 |
| $\ell_2$ and weight decay | shrink the weights, lower the effective complexity | objective function | KP07, KP08 |
| $\ell_1$ | produce sparsity, lower the effective complexity | objective function | KP09 |
| early stopping | train less, roughly the same as $\ell_2$ | training process | KP10 |
| dropout | lower the variance (approximate sub-network average) | hidden layers | KP11 |
| batch normalization | steady and smooth the optimization; mild regularization | between layers | KP12 |

整章的主線可以用一句話說：**真實誤差約等於訓練誤差加上複雜度**，所以要嘛降低有效複雜度，要嘛降低變異；而正則化的所有方法，都是在不增加偏差太多的前提下，用不同的方式達成這件事。這是來源章節總結的意思，我們把它放進本章已經建立的語言。

The main thread of the whole chapter in one sentence: **the true error is about the training error plus the complexity**, so either lower the effective complexity or lower the variance; and all the methods of regularization do this in different ways without adding too much bias. This is the meaning of the source's chapter summary, put into the language this chapter has built.

這一章的方法，大多不假設特定的網路結構。下一章開始，我們要看一種專門利用資料結構（影像的局部與平移不變）的網路：卷積神經網路（CNN）。它的卷積核共享權重、只看局部，也是一種把「複雜度」壓低的做法，這會讓本章的每一個想法在新的結構上再出現一次。

Most methods of this chapter do not assume a particular network structure. Starting from the next chapter we look at a network built to exploit the structure of the data (locality and translation invariance of images): the convolutional neural network (CNN). Its kernels share weights and look only locally, which is also a way of lowering the "complexity", so every idea of this chapter will appear again on a new structure.

### 本單元重點整理（KP12 summary）

**一句話重點**：批次正規化對每個 mini-batch 的每個特徵算平均與變異數，標準化成 $\hat x=\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}$（再接可學習的 $\gamma,\beta$），測試時改用訓練時累積的移動平均；它主要靠平滑損失地形而有效，只有輕微的正則化效果，也不改變分布的形狀。

**One-sentence summary**: Batch normalization computes the mean and variance of each feature over each mini-batch and standardizes it as $\hat x=\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}$ (followed by learnable $\gamma,\beta$), and uses moving averages gathered in training at test time; it works mainly by smoothing the loss landscape, has only a mild regularizing effect and does not change the shape of the distribution.

**白話說明**：想像每個班的成績單位不同（有的滿分 100、有的滿分 10），直接比不公平。把每班各自換算成「離班平均幾個標準差」，大家就在同一把尺上。批次正規化對每一層的輸入做同一件事，所以下一層不必一直適應變動的尺度。

**Plain explanation**: Imagine each class grades on a different scale (some out of 100, some out of 10), so direct comparison is unfair. Convert each class to "how many standard deviations from the class mean" and everyone is on the same ruler. Batch normalization does the same to the input of every layer, so the next layer need not keep adapting to a moving scale.

**正式定義**：$\mu_j=\frac1b\sum_ix_{ij}$、$\sigma_j^2=\frac1b\sum_i(x_{ij}-\mu_j)^2$、$\hat x_{ij}=\frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\varepsilon}}$、$y_{ij}=\gamma_j\hat x_{ij}+\beta_j$；測試時以訓練時累積的 $\bar\mu_j,\bar\sigma_j^2$ 取代批次統計。

**Formal definition**: $\mu_j=\frac1b\sum_ix_{ij}$, $\sigma_j^2=\frac1b\sum_i(x_{ij}-\mu_j)^2$, $\hat x_{ij}=\frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\varepsilon}}$, $y_{ij}=\gamma_j\hat x_{ij}+\beta_j$; at test time the statistics $\bar\mu_j,\bar\sigma_j^2$ accumulated in training replace the batch statistics.

**例子**：$b=4$、值 $2,4,6,8$：$\mu=5$、$\sigma^2=5$、$\hat x=(-1.341639,-0.447213,0.447213,1.341639)$（$\varepsilon=10^{-5}$）。偏斜例 $1,1,2,2,3,10$：偏度前後都是 $1.6043$。批次大小：$b=1$ 時 $\hat x=0$。逐步計算見第 1–3 節。

**Example**: $b=4$, values $2,4,6,8$: $\mu=5$, $\sigma^2=5$, $\hat x=(-1.341639,-0.447213,0.447213,1.341639)$ ($\varepsilon=10^{-5}$). Skewed example $1,1,2,2,3,10$: skewness $1.6043$ before and after. Batch size: $\hat x=0$ when $b=1$. The computations are in sections 1–3.

**常見迷思**：

- 「批次正規化讓分布變成標準常態」——只固定平均 $0$、變異數 $1$，形狀不變；偏斜的分布標準化後仍然偏斜。
- 「它有效是因為減輕了 covariate shift」——來源指出這個解釋不正確；主要好處來自損失地形更平滑。
- 「$\varepsilon$ 是學習率之類的超參數，要仔細調」——它只是避免除以 0 的小常數，一般取 $10^{-5}$ 左右，對結果幾乎沒影響。
- 「測試時也用當前批次的平均與變異數」——不用，改用訓練時累積的移動平均，否則同一筆資料的答案會隨批次而變。
- 「批次愈小愈好，正則化效果愈強」——批次太小，統計不可靠；$b=1$ 時全部變成 $0$。

**Common misconceptions**:

- "Batch normalization turns the distribution into a standard normal one." It fixes only mean $0$ and variance $1$ and leaves the shape alone; a skewed distribution is still skewed after standardization.
- "It works because it reduces covariate shift." The source points out that this explanation is incorrect; the main benefit comes from a smoother loss landscape.
- "$\varepsilon$ is a hyper-parameter like the learning rate and must be tuned carefully." It is only a small constant that avoids division by 0, usually about $10^{-5}$, and hardly affects the result.
- "At test time the mean and variance of the current batch are used too." No: the moving averages gathered in training are used, otherwise the answer for one data point would change with its batch.
- "The smaller the batch, the better, since the regularizing effect is stronger." With too small a batch the statistics are unreliable; at $b=1$ everything becomes $0$.

**圖**：本單元兩張圖，`lec04-kp12-hist-bn`（偏斜特徵標準化前後，形狀不變，功能 1、2）與 `lec04-kp12-batch-noise`（批次統計的雜訊隨批次大小下降，功能 1、2），以 `assets/figures/src/lec04-kp12-figures.py` 繪製。全章回顧的對照表在簡報中以原生表格呈現，講義以文字表格呈現。

**Figure**: Two figures in this unit, `lec04-kp12-hist-bn` (a skewed feature before and after standardization, shape unchanged; functions 1, 2) and `lec04-kp12-batch-noise` (the noise of the batch statistics falling with batch size; functions 1, 2), drawn by `assets/figures/src/lec04-kp12-figures.py`. The chapter review table is a native table in the deck and a text table in the lecture notes.

**想深入**（選讀）：本單元只看了一個特徵一個批次的標準化。在卷積網路裡，批次正規化通常對每個通道、把同一批所有位置的值一起算平均與變異數，原理相同，只是「這一批的資料」包含了空間位置；那是下一章會遇到的情形。另外，批次正規化有一些不依賴批次大小的變形（對每一筆資料自己的各個特徵做標準化），超出本單元。

**Going deeper** (optional): This unit looked at the standardization of one feature in one batch. In convolutional networks, batch normalization usually computes the mean and variance per channel over all positions of all data in the batch; the principle is the same, only "the data of this batch" includes spatial positions, which is the case of the next chapter. There are also variants that do not depend on the batch size (standardizing the features of each data point on its own), which lie beyond this unit.

**小練習**：

1. $b=4$，某特徵的值是 $3,5,9,11$（取 $\varepsilon=0$）：求 $\mu$、$\sigma^2$ 與標準化後的四個值。（提示：離差先算，再平方、平均；變異數除以 $b$。）
2. 承第 1 題，取 $\gamma=3$、$\beta=-1$，求 $y=\gamma\hat x+\beta$，並說明 $y$ 的平均與標準差各是多少。（提示：$\hat x$ 的平均是 $0$、標準差是 $1$。）
3. 若 $b=1$，標準化的結果是什麼？為什麼這會讓批次正規化在批次很小時出問題？（提示：一筆資料的平均就是它自己。）

**Practice**:

1. $b=4$ and a feature has values $3,5,9,11$ (take $\varepsilon=0$): find $\mu$, $\sigma^2$ and the four standardized values. (Hint: compute the deviations first, then square and average; the variance divides by $b$.)
2. Continuing exercise 1, take $\gamma=3$ and $\beta=-1$ and find $y=\gamma\hat x+\beta$, and say what the mean and the standard deviation of $y$ are. (Hint: $\hat x$ has mean $0$ and standard deviation $1$.)
3. What is the result of standardization when $b=1$? Why does this make batch normalization fail for very small batches? (Hint: the mean of a single point is the point itself.)

**解題示範**（以 $b=4$、值 $1,2,3,6$、$\varepsilon=0$ 示範第 1 題的算法）：

1. 平均：$\mu=\frac{1+2+3+6}{4}=3$。
2. 離差與平方：$(-2,-1,0,3)$、$(4,1,0,9)$，總和 $14$。
3. 變異數：$\sigma^2=\frac{14}{4}=3.5$，$\sqrt{3.5}=1.8708$。
4. 標準化：$\hat x=\big(\frac{-2}{1.8708},\frac{-1}{1.8708},0,\frac{3}{1.8708}\big)=(-1.0690,-0.5345,0,1.6036)$。
5. 檢查：$\hat x$ 的平均 $=\frac{-1.0690-0.5345+0+1.6036}{4}\approx0$，平方的平均 $=\frac{1.1428+0.2857+0+2.5714}{4}=1.0$。

**Worked solution** (the method of exercise 1, shown with $b=4$, values $1,2,3,6$ and $\varepsilon=0$):

1. Mean: $\mu=\frac{1+2+3+6}{4}=3$.
2. Deviations and squares: $(-2,-1,0,3)$ and $(4,1,0,9)$, with sum $14$.
3. Variance: $\sigma^2=\frac{14}{4}=3.5$ and $\sqrt{3.5}=1.8708$.
4. Standardize: $\hat x=\big(\frac{-2}{1.8708},\frac{-1}{1.8708},0,\frac{3}{1.8708}\big)=(-1.0690,-0.5345,0,1.6036)$.
5. Check: the mean of $\hat x$ is $\frac{-1.0690-0.5345+0+1.6036}{4}\approx0$ and the mean of the squares is $\frac{1.1428+0.2857+0+2.5714}{4}=1.0$.

### 來源更正與取捨（Corrections to the source）

- 來源只寫到 $\hat x=\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}$，沒有提到標準化之後的可學習 $\gamma$、$\beta$，也沒有說測試時怎麼辦。本講義補上兩者（`[補充]`），因為少了它們，批次正規化無法實際使用：測試時一筆資料沒有批次統計可用。
- 來源說批次正規化「穩定訓練、改善梯度流，並帶有輕微正則化效果」，並指出「減輕 covariate shift」的早期解釋不正確，主要好處來自平滑損失地形。本講義照此寫，並在第 1 節開頭的直覺處明講那只是直覺，不是它有效的原因；損失地形的結論只引用 Santurkar 等人（NeurIPS 2018）一處（`[補充]`），不展開。
- 來源的教授祕笈有一個理解題說 $\varepsilon$ 的作用是避免除以零、增加數值穩定性。本講義同意，並補一句：在一般情形下它對結果幾乎沒影響（我們算出 $\varepsilon=10^{-5}$ 只動到第六位小數）。
- 「對每個特徵各自標準化的尺度不變性例」「偏度前後相同的例」「批次大小對統計雜訊的模擬」「$b=1$ 全為 $0$」與「輕微的正則化可能讓 dropout 調小」是 `[補充]`；最後一項只是常見的經驗，不是定理。
- 來源的章節總結只用一段話列出方法。本講義補上「各方法控制什麼、作用在哪裡」的對照表與整章主線，是我們的整理（`[補充]`）。

- The source stops at $\hat x=\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}$; it does not mention the learnable $\gamma$ and $\beta$ after standardization, nor what to do at test time. This note adds both (`[Added]`), because without them batch normalization cannot be used in practice: at test time a single data point has no batch statistics.
- The source says batch normalization "steadies training, improves gradient flow and has a mild regularizing effect", and points out that the early explanation "reduces covariate shift" is incorrect and that the main benefit comes from a smoother loss landscape. This note follows that, and states plainly at the intuition at the start of section 1 that it is only an intuition and not the reason it works; the loss-landscape conclusion cites only Santurkar et al. (NeurIPS 2018) (`[Added]`) and is not developed further.
- A comprehension question in the source's guide says that $\varepsilon$ avoids division by zero and adds numerical stability. This note agrees and adds one sentence: in ordinary cases it hardly affects the result (we computed that $\varepsilon=10^{-5}$ moves only the sixth decimal).
- "The scale-invariance example for standardizing each feature on its own", "the example in which the skewness is the same before and after", "the simulation of the noise of the statistics against the batch size", "$b=1$ gives all zeros" and "the mild regularization may let dropout be turned down" are `[Added]`; the last is only common experience and not a theorem.
- The source's chapter summary lists the methods in one paragraph. This note adds the table of "what each method controls and where it acts" and the main thread of the chapter, which are our own synthesis (`[Added]`).

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 批次正規化 | batch normalization |
| 標準化 | standardization |
| 平均數 | mean |
| 變異數 | variance |
| 移動平均 | moving average |
| 偏度 | skewness |
| 損失地形 | loss landscape |
| 共變量偏移 | covariate shift |
| 卷積神經網路 | convolutional neural network (CNN) |
