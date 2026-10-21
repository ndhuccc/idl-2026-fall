# 第4週 KP11：Dropout（Dropout）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第4週「正則化與過擬合」的第十一個知識點（KP11）。前面的 KP05 說過，過擬合時模型太靈活、對訓練資料的細節太敏感；直覺上，「訓練很多個不同的模型再取平均」會比單一模型穩定，因為各模型的錯誤方向不同，平均後會互相抵消一部分。問題是深度網路訓練一個就很貴，訓練很多個根本負擔不起。有沒有辦法只訓練一個網路，卻得到「很多個模型取平均」的效果？先備知識是期望值的線性性質；第一次出現的符號都會定義，所有數字都可以自己重算。內容依 CCC-LNS 第5章 KP16 主文與教授祕笈（其中 Dropout 的部分）消化後重新撰寫，讀取日期 2026-10-20；來源沒有的內容標示 `[補充]`。

This is the eleventh knowledge point (KP11) of Week 4, "Regularization and overfitting". KP05 explained that an overfit model is too flexible and too sensitive to the details of the training data; intuitively, training many different models and averaging them is steadier than a single model, because the models err in different directions and part of the errors cancel in the average. The problem is that even one deep network is expensive to train, and many of them are out of reach. Can we train a single network and still get the effect of averaging many models? It assumes only the linearity of expectation; every symbol is defined when it first appears and every number can be recomputed. The text is rewritten from the KP16 main text and professor's guide of CCC-LNS Chapter 5 (the dropout part), accessed on 2026-10-20; content that the source does not contain is marked `[Added]`.

[下載本單元投影片](../slides/lec04-kp11-dropout.jbd.json)

## KP11 Dropout（Dropout）

### 學習目標（Learning objectives）

讀完之後，你能說明 dropout 在訓練時做了什麼（每一步隨機關掉一部分神經元），為什麼這等於同時訓練許多共享權重的子網路；能算出一個神經元在隨機遮罩下的期望輸入，並說明測試時為什麼要把輸出乘以保留機率 $p$（或在訓練時改乘 $1/p$）；能指出「測試時的縮放是子網路的幾何平均」在什麼條件下成立、在深網路只是近似；也能說明 dropout 與「各自獨立訓練多個模型再平均」有什麼不同，並在 PyTorch 中正確使用 `nn.Dropout` 與 `train()`／`eval()`。

After reading, you can explain what dropout does during training (at each step a random part of the neurons is switched off) and why that amounts to training many sub-networks that share weights; compute the expected input of a neuron under a random mask and explain why the output is multiplied by the keep probability $p$ at test time (or by $1/p$ during training); say when "the test-time scaling is the geometric mean of the sub-networks" holds and why it is only an approximation in deep networks; and say how dropout differs from training several independent models and averaging them and use `nn.Dropout` with `train()`/`eval()` correctly in PyTorch.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{h}$ | 某一層的神經元輸出（活化值）向量，共 $n$ 個 | the vector of neuron outputs (activations) of a layer, $n$ of them |
| $p$ | **保留機率**（keep probability）：每個神經元被留下的機率 | the **keep probability**: the chance that each neuron is kept |
| $q=1-p$ | 丟棄機率（drop probability） | the drop probability |
| $m_i\in\{0,1\}$ | 第 $i$ 個神經元的遮罩：$1$ 保留、$0$ 關掉，$\Pr(m_i=1)=p$ | mask of neuron $i$: $1$ kept, $0$ switched off, $\Pr(m_i=1)=p$ |
| $\mathbf{m}\odot\mathbf{h}$ | 逐元素相乘：被關掉的神經元輸出變成 0 | element-wise product: switched-off neurons output 0 |
| $\mathbf{w}$ | 下一層某個線性單元讀取 $\mathbf{h}$ 的權重 | the weights with which a linear unit of the next layer reads $\mathbf{h}$ |
| $z=\mathbf{w}^\top\mathbf{h}$ | 該單元的輸入（pre-activation） | the input (pre-activation) of that unit |

> 本單元的 $p$ 一律指**保留**機率。來源的主文寫「機率通常 $p=0.5$」，祕笈卻寫「以機率 $p$ 保留」，兩處沒有說清楚是保留還是丟棄。許多程式庫（包括 PyTorch）的參數是**丟棄**機率 $q=1-p$，用的時候要換算，見第 4 節。
>
> In this unit $p$ always means the **keep** probability. The source's main text says "probability usually $p=0.5$" while its guide says "keep with probability $p$", and neither makes clear whether $p$ is keep or drop. Many libraries (PyTorch included) take the **drop** probability $q=1-p$ as the parameter, so a conversion is needed; see section 4.

### 1. 隨機關掉神經元：訓練一群共享權重的子網路（Switching neurons off at random: training many weight-sharing sub-networks）

要讓「平均」有效，模型必須彼此不同，而且要有很多個。對深度網路，每個模型都要完整訓練，代價太高。Dropout 的做法是：在每一個訓練步驟，對指定那一層的每個神經元，獨立地以機率 $p$ 保留、以機率 $1-p$ 關掉（輸出變成 0），這一步的前向與反向傳播都只在留下的神經元上進行。

For averaging to work the models must differ, and there must be many of them. For a deep network every model has to be trained in full, which is too costly. Dropout does the following: at every training step, for each neuron of the chosen layer, independently keep it with probability $p$ and switch it off (output 0) with probability $1-p$; the forward and backward pass of this step run only through the neurons that remain.

在遮罩 $\mathbf{m}=(m_1,\dots,m_n)$ 之下，這一層的輸出變成 $\mathbf{m}\odot\mathbf{h}$。每個神經元被關掉都是獨立的，所以一層有 $n$ 個神經元，就有 $2^n$ 種可能的遮罩，也就是 $2^n$ 個不同的「子網路」。例如 $n=4$ 有 $2^4=16$ 個；$n=100$ 有 $2^{100}\approx1.27\times10^{30}$ 個；一層 1000 個神經元則約有 $10^{301}$ 個〔程式計算〕。每個訓練步驟抽到一個遮罩，等於選中一個子網路，更新它的權重一次。

Under a mask $\mathbf{m}=(m_1,\dots,m_n)$ the output of this layer becomes $\mathbf{m}\odot\mathbf{h}$. Each neuron is switched off independently, so a layer with $n$ neurons has $2^n$ possible masks, that is $2^n$ different "sub-networks". For $n=4$ there are $2^4=16$; for $n=100$ there are $2^{100}\approx1.27\times10^{30}$; a layer of 1000 neurons has about $10^{301}$ [computed by program]. Each training step draws one mask, which selects one sub-network, and updates its weights once.

![兩個網路並排：左邊是完整的網路，4 個藍色輸入、5 個橘色隱藏神經元、1 個綠色輸出；右邊是同一個網路套上一個遮罩，兩個隱藏神經元被畫上紅叉，與它們相連的線全部消失，留下三個隱藏神經元。（Two networks side by side: on the left the full network with 4 blue inputs, 5 orange hidden neurons and 1 green output; on the right the same network under one mask, two hidden neurons crossed in red and all lines attached to them removed, three hidden neurons remaining.）](../assets/figures/lec04-kp11-network.png)

**圖說 `lec04-kp11-network`**：先看左圖：這是平常的完整網路，每條線都有一個權重。再看右圖：同一組權重，其中兩個隱藏神經元被紅叉關掉，連到它們的線也都消失，剩下一個比較小的網路。下一個訓練步驟會抽新的遮罩，叉會換到別的神經元上。〔功能：1、3〕

**Caption `lec04-kp11-network`**: Look first at the left panel: the usual full network, with one weight on each line. Then look at the right panel: the same weights, but two hidden neurons are switched off by the red crosses and all lines attached to them vanish, leaving a smaller network. The next training step draws a new mask and the crosses move to other neurons. [Function: 1, 3]

這裡有兩個重要事實。第一，所有子網路**共享同一組權重**：一個神經元的權重在被留下的每一個子網路裡都是同一組數字，每次它被留下，就被更新一次。第二，$2^n$ 個子網路多到不可能每個都訓練到：一次訓練只走過其中極小的一部分，多數子網路從頭到尾沒被抽到過；之所以還能運作，正是因為權重是共享的。這就是 dropout 被稱為「便宜的模型平均」的原因，但它和「各自獨立訓練多個模型」並不相同，第 3 節會再比較。

Two facts matter here. First, all sub-networks **share one set of weights**: the weights of a neuron are the same numbers in every sub-network that keeps it, and each time it is kept it is updated once. Second, $2^n$ sub-networks are far too many to train each one: a whole training run visits only a tiny part of them and most are never drawn at all; the scheme still works precisely because the weights are shared. This is why dropout is called a "cheap model average", but it is not the same as training several independent models, as section 3 compares.

### 2. 測試時為什麼要縮放（Why we scale at test time）

訓練時每個神經元只有機率 $p$ 在場，測試時如果把所有神經元都打開，下一層看到的輸入就比訓練時「大」，大約多了 $1/p$ 倍。為了看清楚，我們計算下一層一個線性單元的輸入 $z=\mathbf{w}^\top\mathbf{h}$。在遮罩下，它變成 $z_{\mathbf{m}}=\mathbf{w}^\top(\mathbf{m}\odot\mathbf{h})=\sum_i w_im_ih_i$。因為 $\mathbb E(m_i)=p$，由期望的線性性質：

During training each neuron is present only with probability $p$; if all neurons are switched on at test time, the next layer sees inputs that are "larger" than in training, by roughly a factor $1/p$. To see this precisely, compute the input $z=\mathbf{w}^\top\mathbf{h}$ of one linear unit of the next layer. Under a mask it becomes $z_{\mathbf{m}}=\mathbf{w}^\top(\mathbf{m}\odot\mathbf{h})=\sum_i w_im_ih_i$. Since $\mathbb E(m_i)=p$, linearity of expectation gives:

$$
\mathbb E\big(z_{\mathbf{m}}\big)=\sum_i w_i\,\mathbb E(m_i)\,h_i=p\sum_i w_ih_i=p\,z
$$

所以訓練時這個單元的輸入，平均是完整輸入的 $p$ 倍。測試時要讓輸入的大小與訓練時一致，就把它乘以 $p$：把這一層的輸出乘 $p$，或等價地把下一層的權重乘 $p$。這叫**權重縮放規則**（weight scaling）。

So during training the input of this unit is, on average, $p$ times the full input. To make the size of the input at test time match training, multiply it by $p$: multiply the output of this layer by $p$, or equivalently multiply the weights of the next layer by $p$. This is the **weight scaling rule**.

**數值例**：一層有 4 個神經元，輸出 $\mathbf{h}=(2,-1,3,1)$；下一層一個線性單元的權重 $\mathbf{w}=(1,2,1,-1)$，$p=0.5$。完整輸入 $z=1\cdot2+2\cdot(-1)+1\cdot3+(-1)\cdot1=2-2+3-1=2$。

**Numerical example**: a layer has 4 neurons with outputs $\mathbf{h}=(2,-1,3,1)$; a linear unit of the next layer has weights $\mathbf{w}=(1,2,1,-1)$ and $p=0.5$. The full input is $z=1\cdot2+2\cdot(-1)+1\cdot3+(-1)\cdot1=2-2+3-1=2$.

- 抽到遮罩 $\mathbf{m}=(1,0,1,1)$（第二個被關掉）：$z_{\mathbf{m}}=2+0+3-1=4$。注意這一個遮罩的輸入是 $4$，比完整的 $2$ 還大，單一遮罩可以偏離很多。
- 全部 16 個遮罩各以機率 $\frac1{16}$ 出現。把 16 個 $z_{\mathbf{m}}$ 平均，得到 $1.0$，等於 $p\,z=0.5\times2=1$，一點不差。16 個值是 $-3,-2,-1,-1,0,0,0,1,1,2,2,2,3,3,4,5$〔程式計算〕。
- 它們的標準差是 $2.1213$，與公式 $\sqrt{p(1-p)\sum_i(w_ih_i)^2}=\sqrt{0.25\times18}$ 相符：單次遮罩有雜訊，平均才準。

- Draw the mask $\mathbf{m}=(1,0,1,1)$ (the second neuron switched off): $z_{\mathbf{m}}=2+0+3-1=4$. Note that this one mask gives $4$, larger than the full $2$; a single mask can deviate a lot.
- All 16 masks occur with probability $\frac1{16}$ each. Averaging the 16 values $z_{\mathbf{m}}$ gives $1.0$, equal to $p\,z=0.5\times2=1$ exactly. The 16 values are $-3,-2,-1,-1,0,0,0,1,1,2,2,2,3,3,4,5$ [computed by program].
- Their standard deviation is $2.1213$, matching $\sqrt{p(1-p)\sum_i(w_ih_i)^2}=\sqrt{0.25\times18}$: a single mask is noisy, the average is exact.

神經元多的時候用模擬更直觀：隨機取 20 個神經元的 $\mathbf{h}$ 與 $\mathbf{w}$（固定種子），$p=0.5$，抽 20000 個遮罩。完整輸入 $z=3.7040$，$p\,z=1.8520$；20000 個遮罩下的平均是 $1.8340$，標準差 $2.0989$〔程式計算〕。平均與 $p\,z$ 接近，差距來自抽樣誤差。

With many neurons a simulation is more convincing: take 20 neurons with random $\mathbf{h}$ and $\mathbf{w}$ (fixed seed), $p=0.5$, and draw 20000 masks. The full input is $z=3.7040$ and $p\,z=1.8520$; over the 20000 masks the mean is $1.8340$ and the standard deviation $2.0989$ [computed by program]. The mean is close to $p\,z$ and the gap is sampling error.

![直方圖：20000 個隨機遮罩下某個線性單元輸入的分布，呈鐘形、範圍約 −4 到 8；綠色實線標出 p 乘 z（約 1.85），位在分布中央；紅色虛線標出完整網路的 z（約 3.70），落在分布右半邊。（Histogram: the distribution of the input of a linear unit over 20000 random masks, bell-shaped, ranging from about −4 to 8; a solid green line marks p times z (about 1.85) at the centre of the distribution; a dashed red line marks the full network z (about 3.70) on its right half.）](../assets/figures/lec04-kp11-mask-hist.png)

**圖說 `lec04-kp11-mask-hist`**：先看藍色直方圖：20000 個遮罩給出 20000 個不同的輸入，寬度約 $\pm4$，這就是「子網路各說各話」；再看綠色實線：它落在分布的中心，就是 $p\,z$；最後看紅色虛線：完整網路的 $z$ 在中心的右邊。如果測試時直接用完整網路，輸入比訓練時的平均大了一倍，所以要乘 $p$。〔功能：1、2〕

**Caption `lec04-kp11-mask-hist`**: Look first at the blue histogram: 20000 masks give 20000 different inputs, about $\pm4$ wide; this is "the sub-networks each saying something different". Then look at the solid green line: it sits at the centre of the distribution and is $p\,z$. Finally look at the dashed red line: the full network's $z$ lies to the right of the centre. Using the full network directly at test time would feed in an input twice as large as the training average, hence the factor $p$. [Function: 1, 2]

**反向做法：inverted dropout**（`[補充]`）。現在的程式庫幾乎都不在測試時縮放，而是在**訓練時**把留下的神經元輸出除以 $p$，即用 $\frac1p\,\mathbf{m}\odot\mathbf{h}$。這樣訓練時的平均輸出剛好是 $\mathbf{h}$：$\mathbb E\big(\frac1p m_ih_i\big)=\frac1p\,p\,h_i=h_i$（上面的例子逐項驗算，平均 $=(2,-1,3,1)=\mathbf{h}$）。測試時直接用完整網路，什麼都不用改。兩種做法在期望上等價，只是把縮放挪到訓練那一邊，部署時不必記得「要乘 $p$」。

**The reverse way: inverted dropout** (`[Added]`). Nearly all libraries do not scale at test time; instead they divide the surviving outputs by $p$ **during training**, i.e. they use $\frac1p\,\mathbf{m}\odot\mathbf{h}$. The training-time mean is then exactly $\mathbf{h}$: $\mathbb E\big(\frac1p m_ih_i\big)=\frac1p\,p\,h_i=h_i$ (checked term by term on the example above: the mean is $(2,-1,3,1)=\mathbf{h}$). At test time the full network is used unchanged. The two ways are equivalent in expectation; the scaling is only moved to the training side, so at deployment there is nothing to remember.

### 3. 它真的是子網路的平均嗎？與獨立多模型平均的差別（Is it really an average of sub-networks? Differences from independent models）

來源的祕笈說：測試時乘 $p$ 的縮放，是在做這些子網路的**幾何平均**。這句話要加上限制才正確。

The source's guide says that the multiply-by-$p$ scaling at test time computes the **geometric mean** of the sub-networks. This sentence is correct only with a qualification.

先看成立的情形。若被 dropout 的那一層後面接的是 softmax 分類層（即 softmax 迴歸 $\mathrm{softmax}(W(\mathbf{m}\odot\mathbf{h}))$），可以證明：把 $2^n$ 個子網路輸出的機率向量做**正規化的幾何平均**（先取對數平均、再取指數、再除以總和），結果恰好等於權重縮放後的輸出 $\mathrm{softmax}(p\,W\mathbf{h})$。原因是 softmax 的對數是 $W(\mathbf{m}\odot\mathbf{h})$ 的線性函數，對數取平均就是取期望，由第 2 節的線性性質得到 $p\,W\mathbf{h}$。

First the case where it holds. If the layer subject to dropout feeds a softmax classification layer (softmax regression $\mathrm{softmax}(W(\mathbf{m}\odot\mathbf{h}))$), one can show that the **normalized geometric mean** of the probability vectors of the $2^n$ sub-networks (average the logarithms, exponentiate, divide by the sum) equals exactly the weight-scaled output $\mathrm{softmax}(p\,W\mathbf{h})$. The reason is that the logits are linear in $\mathbf{m}\odot\mathbf{h}$, averaging the logarithm is taking an expectation, and section 2 gives $p\,W\mathbf{h}$.

我們用固定種子的例子驗證〔程式計算〕。$4$ 個輸入、$3$ 個類別、$p=0.5$，枚舉全部 16 個遮罩：權重縮放的輸出是 $(0.941587,\,0.014320,\,0.044093)$，正規化幾何平均與它一樣，最大差距 $1.4\times10^{-17}$（浮點誤差）；而 16 個輸出的**算術平均**是 $(0.7970,\,0.0878,\,0.1152)$，差很多。換成一個 ReLU 隱藏層（6 個神經元，只在這層做 dropout，後面是 softmax 層），同樣枚舉 64 個遮罩，最大差距是 $5.6\times10^{-17}$，也相等。

We check this with a fixed-seed example [computed by program]. 4 inputs, 3 classes, $p=0.5$, enumerating all 16 masks: the weight-scaled output is $(0.941587,\,0.014320,\,0.044093)$ and the normalized geometric mean is the same, with a largest gap of $1.4\times10^{-17}$ (floating-point error); the **arithmetic mean** of the 16 outputs is $(0.7970,\,0.0878,\,0.1152)$, quite different. With one ReLU hidden layer (6 neurons, dropout only on this layer, followed by a softmax layer) and all 64 masks, the largest gap is $5.6\times10^{-17}$: equal again.

再看不成立的情形：兩個隱藏層（各 5 個神經元），兩層都做 dropout，中間夾著 ReLU，枚舉 $2^5\times2^5=1024$ 個遮罩。權重縮放的輸出是 $(0.22991,\,0.63433,\,0.13576)$，正規化幾何平均卻是 $(0.08085,\,0.89391,\,0.02524)$，最大差距 $0.2596$〔程式計算〕。也就是說：非線性（ReLU）夾在兩層 dropout 中間，縮放就不再等於幾何平均。這只是固定種子的一個例子，不是一般規律；結論是「在深度非線性網路，權重縮放只是近似」。實務上它效果不錯（Srivastava 等人 2014，`[補充]`），但這是經驗，不是定理。

Now the case where it fails: two hidden layers (5 neurons each) with dropout on both and a ReLU in between, enumerating $2^5\times2^5=1024$ masks. The weight-scaled output is $(0.22991,\,0.63433,\,0.13576)$ while the normalized geometric mean is $(0.08085,\,0.89391,\,0.02524)$, a largest gap of $0.2596$ [computed by program]. So with a nonlinearity (ReLU) between two dropout layers the scaling is no longer the geometric mean. This is one fixed-seed example and not a general law; the conclusion is "in deep nonlinear networks weight scaling is only an approximation". In practice it works well (Srivastava et al., 2014, `[Added]`), but that is experience, not a theorem.

**和「獨立訓練 $k$ 個模型再平均」比較。**

**Comparison with training $k$ independent models and averaging.**

| | 獨立訓練 $k$ 個模型 | Dropout |
|---|---|---|
| 模型從哪來 | 各自用不同資料（或不同初值）訓練 | 同一份資料，隨機遮罩選出子網路 |
| 模型之間 | 權重各自獨立 | **共享權重** |
| 訓練成本 | $k$ 倍 | 約 1 倍（略慢收斂） |
| 預測成本 | $k$ 個模型各跑一次再平均 | 1 次（乘 $p$ 或除以 $p$ 的縮放） |
| 平均的對象 | 算術平均（迴歸）或多數決 | 縮放近似子網路的（幾何）平均 |
| 常見用在 | 小模型、不穩定的學習器 | 深度網路 |

| | $k$ independent models | Dropout |
|---|---|---|
| where models come from | trained separately on different data (or initial values) | one data set, random masks pick the sub-networks |
| between models | independent weights | **shared weights** |
| training cost | $k$ times | about 1 time (converges a bit slower) |
| prediction cost | run $k$ models and average | 1 pass (scaling by $p$ or $1/p$) |
| what is averaged | arithmetic mean (regression) or majority vote | scaling approximates the (geometric) mean of the sub-networks |
| usually used for | small models and unstable learners | deep networks |

共享權重帶來兩個結果：好處是便宜；代價是子網路彼此**不獨立**（用的是同一組數字），平均掉錯誤的效果不如真正獨立的模型。一個可用的直覺是：dropout 逼每個神經元不能依賴特定夥伴一定在場，必須學出在各種組合下都有用的特徵，這就是來源比喻的「每次只讓部分球員上場練習，讓每個球員都能獨當一面」。

Shared weights have two consequences: the benefit is that it is cheap; the cost is that the sub-networks are **not independent** (they use the same numbers), so averaging cancels errors less well than truly independent models would. A useful intuition: dropout forces every neuron not to count on particular partners always being present, so it must learn features that are useful in many combinations; this is the source's metaphor, "let only part of the players practise each time so that every player can stand on their own".

### 4. 在 PyTorch 中使用，以及一些經驗值（Using it in PyTorch, and some rules of thumb）

下面是一個帶 dropout 的兩層網路。**這段程式沒有執行過**（本環境沒有安裝 PyTorch）；邏輯已用 numpy 對照過：留下的神經元除以 $p$ 之後，平均輸出等於原本的 $\mathbf{h}$。

Below is a two-layer network with dropout. **This code was not executed** (PyTorch is not installed in this environment); its logic was checked against numpy: after dividing the surviving neurons by $p$ the mean output equals the original $\mathbf{h}$.

```python
import torch
from torch import nn


class MLP(nn.Module):
    """Two-layer perceptron with dropout on the hidden layer.

    keep_prob is the probability that a hidden unit is KEPT (p in this unit).
    nn.Dropout expects the DROP probability q = 1 - p, so we convert.
    """

    def __init__(self, d_in=4, d_hidden=64, d_out=3, keep_prob=0.8):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.drop = nn.Dropout(p=1.0 - keep_prob)   # drop probability, NOT keep probability
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.drop(h)      # train mode: zero units w.p. 1-p and scale survivors by 1/p; eval mode: identity
        return self.fc2(h)


model = MLP()

model.train()                 # dropout is ACTIVE: every forward pass draws a new mask
x = torch.randn(8, 4)
logits_train = model(x)       # two calls on the same x give different outputs

model.eval()                  # dropout is OFF: the full network is used, no extra scaling needed
with torch.no_grad():
    logits_test = model(x)    # deterministic: the same x always gives the same output
```

要點有三。第一，`nn.Dropout(p=...)` 的 $p$ 是**丟棄**機率，與本單元的保留機率 $p$ 相反；寫 `nn.Dropout(p=0.5)` 時兩者剛好都是 $0.5$，容易搞混，改成 $0.8$ 就會出錯。第二，一定要在評估前呼叫 `model.eval()`，訓練前呼叫 `model.train()`；忘了 `eval()`，測試時還在隨機關神經元，結果每次都不同、也比較差。第三，PyTorch 採用 inverted dropout，所以測試時不必乘 $p$。

Three points. First, the $p$ of `nn.Dropout(p=...)` is the **drop** probability, the opposite of the keep probability $p$ of this unit; with `nn.Dropout(p=0.5)` both happen to be $0.5$ and are easy to confuse, but with $0.8$ the mistake shows. Second, always call `model.eval()` before evaluation and `model.train()` before training; if you forget `eval()`, neurons are still switched off at test time and the results differ on every call and are worse. Third, PyTorch uses inverted dropout, so there is nothing to multiply by $p$ at test time.

**經驗值**（`[補充]`，只是常見的做法，不是定理）：隱藏層的保留機率常取 $0.5$，輸入層常取較大的 $0.8$ 到 $0.9$（輸入少掉一點就少一點資訊）。dropout 讓每步更新用到的神經元變少，收斂通常比較慢，常需要較多的訓練輪數；模型本來就小、不過擬合時，不需要加。

**Rules of thumb** (`[Added]`, common practice and not theorems): the keep probability of hidden layers is often $0.5$ and that of the input layer a larger $0.8$ to $0.9$ (losing an input loses information). Dropout makes each update use fewer neurons, so convergence is usually slower and often needs more epochs; when the model is small and does not overfit, there is no need for it.

最後留一個問題給 KP12：Dropout 是在「訓練時隨機改變網路的結構」。另一類方法改變的是每一層**輸入的分布**：每一層的輸入隨著前面層的權重一起改變，訓練起來很不穩定。能不能在層與層之間直接把輸入穩住？這是 KP12（批次正規化）要回答的問題。

One last question for KP12: dropout "changes the structure of the network at random during training". Another family changes the **distribution of the input** of each layer: the input of each layer changes together with the weights of the layers before it, and training becomes unstable. Can we steady the input directly between layers? That is the question KP12 (batch normalization) answers.

### 本單元重點整理（KP11 summary）

**一句話重點**：Dropout 在每個訓練步驟以保留機率 $p$ 隨機關掉神經元，等於訓練 $2^n$ 個共享權重的子網路；測試時用完整網路並把輸出乘 $p$（或訓練時除以 $p$）來近似它們的平均，它不是對所有網路都精確的幾何平均。

**One-sentence summary**: At each training step dropout switches neurons off at random with keep probability $p$, which trains $2^n$ sub-networks that share weights; at test time the full network is used with the output multiplied by $p$ (or divided by $p$ during training) to approximate their average, which is not an exact geometric mean for every network.

**白話說明**：想像球隊練習，每次隨機叫幾位球員不上場，其他人照樣打完整場比賽。沒人能靠固定的搭檔，每個人都得練到單打也強。正式比賽時全員上場，但每個人用力的程度要調成「平常有人不上場時的平均」，所以輸出乘 $p$。

**Plain explanation**: Picture a team practising, where a few players are told at random to sit out each time and the others still play the whole game. Nobody can lean on a fixed partner, so everyone must be good alone. In the real match everyone plays, but each player's effort is set to the "average when some players sit out", which is why the output is multiplied by $p$.

**正式定義**：訓練時 $m_i\sim\mathrm{Bernoulli}(p)$ 獨立，該層輸出 $\mathbf{m}\odot\mathbf{h}$；$\mathbb E(\mathbf{w}^\top(\mathbf{m}\odot\mathbf{h}))=p\,\mathbf{w}^\top\mathbf{h}$。測試時：權重縮放（輸出乘 $p$），或 inverted dropout（訓練時輸出 $\frac1p\mathbf{m}\odot\mathbf{h}$、測試不變）。子網路個數 $2^n$。

**Formal definition**: during training $m_i\sim\mathrm{Bernoulli}(p)$ independently and the layer outputs $\mathbf{m}\odot\mathbf{h}$; $\mathbb E(\mathbf{w}^\top(\mathbf{m}\odot\mathbf{h}))=p\,\mathbf{w}^\top\mathbf{h}$. At test time: weight scaling (outputs multiplied by $p$), or inverted dropout (outputs $\frac1p\mathbf{m}\odot\mathbf{h}$ in training and unchanged at test). The number of sub-networks is $2^n$.

**例子**：$\mathbf{h}=(2,-1,3,1)$、$\mathbf{w}=(1,2,1,-1)$、$p=0.5$：完整 $z=2$；16 個遮罩的平均 $=1.0=p\,z$；標準差 $2.1213$。20 個神經元、20000 個遮罩：$p\,z=1.8520$，平均 $1.8340$。逐步計算見第 2 節。

**Example**: $\mathbf{h}=(2,-1,3,1)$, $\mathbf{w}=(1,2,1,-1)$, $p=0.5$: full $z=2$; the mean over the 16 masks is $1.0=p\,z$; standard deviation $2.1213$. 20 neurons and 20000 masks: $p\,z=1.8520$ and the mean $1.8340$. The computations are in section 2.

**常見迷思**：

- 「Dropout 讓網路變弱」——每個子網路確實較小，但目標是讓神經元不互相依賴，整體更穩健，並且約等於平均許多子網路。
- 「測試時也要隨機關神經元」——不要。測試時全部打開（再做縮放，或用 inverted dropout）；忘了 `model.eval()` 是常見錯誤。
- 「`nn.Dropout(p=0.8)` 是保留 80%」——不對，PyTorch 的 $p$ 是丟棄機率，會丟掉 80%。
- 「縮放一定等於子網路的幾何平均」——只在 softmax 迴歸（線性加 softmax）精確成立；深度非線性網路只是近似。
- 「Dropout 就是各自獨立訓練多個模型再平均」——不是，子網路共享權重、不獨立，也只訓練了其中極小的一部分。

**Common misconceptions**:

- "Dropout makes the network weaker." Each sub-network is indeed smaller, but the goal is to make neurons not depend on one another, so that the whole is more robust and approximates the average of many sub-networks.
- "At test time neurons are also switched off at random." No. At test time all are on (with scaling, or with inverted dropout); forgetting `model.eval()` is a common mistake.
- "`nn.Dropout(p=0.8)` keeps 80%." Wrong: the $p$ of PyTorch is the drop probability, so 80% are dropped.
- "The scaling is always the geometric mean of the sub-networks." It is exact only for softmax regression (linear plus softmax); for deep nonlinear networks it is an approximation.
- "Dropout is training several independent models and averaging them." It is not: the sub-networks share weights, are not independent, and only a tiny part of them is ever trained.

**圖**：本單元兩張圖，`lec04-kp11-network`（完整網路與一個遮罩下的子網路，功能 1、3）與 `lec04-kp11-mask-hist`（線性單元在 20000 個遮罩下的輸入分布，功能 1、2），以 `assets/figures/src/lec04-kp11-figures.py` 繪製。縮放的步驟在簡報中以原生節點與連線呈現。

**Figure**: Two figures in this unit, `lec04-kp11-network` (the full network and the sub-network under one mask; functions 1, 3) and `lec04-kp11-mask-hist` (the input of a linear unit over 20000 masks; functions 1, 2), drawn by `assets/figures/src/lec04-kp11-figures.py`. The scaling steps are drawn with native nodes and connectors in the deck.

**想深入**（選讀）：本單元只做了「一個線性單元」的期望。把 dropout 放在含 ReLU 的深網路，嚴格說期望的非線性函數不等於非線性函數的期望，所以第 3 節才會看到縮放與幾何平均不等。另外，dropout 也可以套在**權重**上（DropConnect）或整個通道上，這些變形超出本單元。

**Going deeper** (optional): This unit only took the expectation of one linear unit. In a deep network with ReLU, the expectation of a nonlinear function is not the nonlinear function of the expectation, which is why section 3 sees the scaling differ from the geometric mean. Dropout can also be applied to **weights** (DropConnect) or whole channels; these variants go beyond this unit.

**小練習**：

1. $\mathbf{h}=(1,3,-2)$、$\mathbf{w}=(2,1,1)$、$p=0.8$：求完整輸入 $z$，以及隨機遮罩下輸入的期望。（提示：期望是 $p\,z$。）
2. 用 $p=0.8$ 做 inverted dropout。某個神經元的輸出是 $1.5$：若它被留下，訓練時輸出變成多少？它在訓練時的期望輸出是多少？（提示：留下的輸出除以 $p$；留下的機率是 $p$。）
3. 一層有 10 個神經元，$p=0.5$：有幾種遮罩？某一種特定遮罩被抽到的機率是多少？兩層各 10 個神經元且都做 dropout，共有幾種遮罩？（提示：$2^{10}$，以及 $2^{20}$。）

**Practice**:

1. $\mathbf{h}=(1,3,-2)$, $\mathbf{w}=(2,1,1)$, $p=0.8$: find the full input $z$ and the expected input under a random mask. (Hint: the expectation is $p\,z$.)
2. Inverted dropout with $p=0.8$. A neuron has output $1.5$: if it is kept, what output does it give in training? What is its expected output in training? (Hint: surviving outputs are divided by $p$; the chance of being kept is $p$.)
3. A layer has 10 neurons and $p=0.5$: how many masks are there? What is the probability that one specific mask is drawn? How many masks if two layers of 10 neurons both use dropout? (Hint: $2^{10}$, and $2^{20}$.)

**解題示範**（以 $\mathbf{h}=(2,0,4)$、$\mathbf{w}=(1,3,-1)$、$p=0.5$ 示範第 1 題的算法）：

1. 完整輸入：$z=1\cdot2+3\cdot0+(-1)\cdot4=-2$。
2. 期望：$p\,z=0.5\times(-2)=-1$。
3. 用 $2^3=8$ 個遮罩驗算：每個神經元的貢獻 $(w_ih_i)=(2,0,-4)$；8 個遮罩的輸入分別是 $0,2,0,-4,2,-2,-4,-2$，總和 $-8$，平均 $-1$。
4. 結論：枚舉的平均 $-1$ 等於 $p\,z$。

**Worked solution** (the method of exercise 1, shown with $\mathbf{h}=(2,0,4)$, $\mathbf{w}=(1,3,-1)$ and $p=0.5$):

1. Full input: $z=1\cdot2+3\cdot0+(-1)\cdot4=-2$.
2. Expectation: $p\,z=0.5\times(-2)=-1$.
3. Check with the $2^3=8$ masks: the contribution of each neuron is $(w_ih_i)=(2,0,-4)$; the inputs of the 8 masks are $0,2,0,-4,2,-2,-4,-2$, with sum $-8$ and mean $-1$.
4. Conclusion: the enumerated mean $-1$ equals $p\,z$.

### 來源更正與取捨（Corrections to the source）

- 來源的主文寫「機率通常 $p=0.5$」關掉神經元，祕笈與測試縮放卻是「以機率 $p$ 保留」，並說「測試時乘 $p$」。若 $p$ 是丟棄機率，乘 $p$ 就錯了。本講義統一定義 $p$ 為**保留**機率，另用 $q=1-p$ 表示丟棄機率，並在第 4 節說明 PyTorch 的 `nn.Dropout(p=...)` 用的是丟棄機率。
- 祕笈說「測試時的縮放是在做這些子網路的幾何平均」。本講義限定：這在 softmax 迴歸（線性層加 softmax）精確成立，我們用枚舉驗證；在含非線性的深網路只是近似，並用一個固定種子的反例（最大差距 $0.2596$）說明。
- 來源把 dropout 稱為「線上 bagging」。本講義改用「便宜的模型平均」，並補上差別：子網路共享權重、彼此不獨立、只訓練其中極小一部分；原先與 bagging 變異數公式的對照已隨 Week 4 縮減刪除。
- 「inverted dropout」「$2^n$ 個子網路」「期望的逐項推導與枚舉驗證」「PyTorch 範例（未執行）」與「經驗值（隱藏層 $0.5$、輸入層 $0.8$–$0.9$）」是 `[補充]`；經驗值只是常見做法，沒有理論保證。

- The source's main text switches neurons off "with probability usually $p=0.5$", while its guide and the test-time scaling say "keep with probability $p$" and "multiply by $p$ at test time". If $p$ were the drop probability, multiplying by $p$ would be wrong. This note defines $p$ uniformly as the **keep** probability, uses $q=1-p$ for the drop probability, and explains in section 4 that PyTorch's `nn.Dropout(p=...)` takes the drop probability.
- The guide says "the scaling at test time computes the geometric mean of these sub-networks". This note restricts it: it holds exactly for softmax regression (a linear layer plus softmax), which we verify by enumeration; for deep networks with nonlinearities it is only an approximation, shown with a fixed-seed counterexample (largest gap $0.2596$).
- The source calls dropout "online bagging". This note says "cheap model averaging" instead and adds the differences: the sub-networks share weights, are not independent, and only a tiny part of them is trained; the comparison with the bagging variance formula was removed with the Week 4 reduction.
- "Inverted dropout", "$2^n$ sub-networks", "the term-by-term derivation of the expectation and the enumeration check", "the PyTorch example (not executed)" and "the rules of thumb (hidden $0.5$, input $0.8$–$0.9$)" are `[Added]`; the rules of thumb are common practice with no theoretical guarantee.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 隨機失活 | dropout |
| 保留機率 | keep probability |
| 丟棄機率 | drop probability |
| 遮罩 | mask |
| 子網路 | sub-network |
| 權重縮放規則 | weight scaling rule |
| 反向 dropout | inverted dropout |
| 幾何平均 | geometric mean |
| 權重共享 | weight sharing |
