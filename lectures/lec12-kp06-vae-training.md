# 第12週 KP06：訓練 VAE，用抽樣估計 ELBO、KL 用公式算、重參數化讓梯度穿過抽樣（Training a VAE: estimate the ELBO by sampling, compute the KL by formula, and let gradients pass through sampling by reparameterization）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第12週的第六個知識點（KP06），接在 KP05 之後。KP05 畫出了 VAE 的三個部分，也寫出了訓練目標 ELBO。這一節回答兩個實際問題：ELBO 裡有一個期望值，要怎麼算？中間有一步「抽樣」，梯度要怎麼穿過它？

This is the sixth knowledge point (KP06) of Week 12, following KP05. KP05 drew the three parts of a VAE and wrote down the training objective, the ELBO. This note answers two practical questions: the ELBO contains an expectation, so how is it computed? And there is a "sampling" step in the middle, so how do gradients pass through it?

先備知識是 KP03 的 ELBO 與兩種寫法、KP04 的 EM、KP05 的 VAE 架構，以及第3週的反向傳播與梯度下降。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第12章 §3.2–§3.4 與教授祕笈改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。圖與實驗的數字都由 `assets/figures/src/lec12-kp06-figures.py` 算出（它用到 KP05 的 `lec12_vae_toy.py`），可以自己重算。

The prerequisites are KP03's ELBO and its two forms, KP04's EM, KP05's VAE architecture, and backpropagation and gradient descent from Week 3. Every term and symbol is defined when it first appears. This note is rewritten from §3.2–§3.4 of Chapter 12 of the course's web material CCC-LNS and its professor's notes, which is what "the source" means below; content added by this note is marked `[Added]`. The figures and experimental numbers are computed by `assets/figures/src/lec12-kp06-figures.py` (which uses KP05's `lec12_vae_toy.py`), so you can recompute it yourself.

## KP06 訓練 VAE，用抽樣估計 ELBO、KL 用公式算、重參數化讓梯度穿過抽樣（Training a VAE: estimate the ELBO by sampling, compute the KL by formula, and let gradients pass through sampling by reparameterization）

### 學習目標（Learning objectives）

讀完之後，你能寫出 ELBO 的兩種抽樣估計（來源稱 Type 1 與 Type 2），並用幾個抽樣值算出估計值。你能用高斯 KL 的公式算出一個編碼器輸出的 KL。你能說明為什麼「抽樣」這一步擋住了梯度，以及重參數化 $\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$ 怎麼讓梯度穿過去。你也能寫出 VAE 的實際損失函數與訓練迴圈，並說出拿掉 KL 項會發生什麼事。

After reading, you can write the two sampling estimates of the ELBO (called Type 1 and Type 2 in the source) and compute an estimate from a few sampled values. You can use the Gaussian KL formula to compute the KL of an encoder's output. You can explain why the "sampling" step blocks gradients, and how the reparameterization $\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$ lets gradients through. You can also write the VAE's actual loss function and training loop, and say what happens when the KL term is removed.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathcal{L}$ | ELBO（KP03） | ELBO (KP03) |
| $\tilde{\mathcal{L}}$ | 用抽樣估計出來的 ELBO | the ELBO estimated by sampling |
| $\ell$ | 每筆資料抽幾個 $\mathbf{z}$ | number of $\mathbf{z}$ drawn per example |
| $\mathbf{z}_{j}$ | 第 $j$ 個抽樣值 | the $j$-th sampled value |
| $\theta_e,\ \theta_d$ | 編碼器與解碼器的權重 | weights of the encoder and the decoder |
| $\boldsymbol{\mu},\ \boldsymbol{\sigma}$ | 編碼器輸出的平均與標準差 | mean and standard deviation output by the encoder |
| $\boldsymbol{\epsilon}$ | 標準常態雜訊，從 $\mathcal{N}(\mathbf{0},I)$ 抽出 | standard-normal noise, drawn from $\mathcal{N}(\mathbf{0},I)$ |
| $I$ | 單位矩陣 | identity matrix |
| $\odot$ | 逐元素相乘 | element-wise multiplication |
| $\Sigma,\ \lvert\Sigma\rvert,\ \operatorname{tr}\Sigma$ | 共變異數矩陣、它的行列式、它的對角線總和 | covariance matrix, its determinant, the sum of its diagonal |
| $\beta$ | KL 項的權重 | weight of the KL term |
| $\hat{\mathbf{x}}$ | 解碼器的輸出（重建） | the decoder's output (reconstruction) |
| $\sigma_x$ | 解碼器常態概似的標準差 | standard deviation of the decoder's normal likelihood |
| $b$ | 小批次的資料筆數 | number of examples in a mini-batch |
| $\eta$ | 學習率 | learning rate |

### 1. ELBO 裡的期望值：用抽樣來估計（The expectation in the ELBO: estimate it by sampling）

KP03 的 ELBO 可以寫成一個期望值：

KP03's ELBO can be written as one expectation:

$$
\mathcal{L}=\mathbb{E}_{q(\mathbf{z}\mid\mathbf{x})}\big[\log p(\mathbf{x},\mathbf{z})-\log q(\mathbf{z}\mid\mathbf{x})\big].
$$

括號裡的兩項都好算：$p(\mathbf{x},\mathbf{z})=p(\mathbf{z})\,p(\mathbf{x}\mid\mathbf{z})$ 是先驗乘上解碼器給的概似，$q$ 是編碼器給的常態分布。難的是「期望值」：它要對所有可能的 $\mathbf{z}$ 積分，對神經網路來說算不出公式。

Both terms inside the brackets are easy to compute: $p(\mathbf{x},\mathbf{z})=p(\mathbf{z})\,p(\mathbf{x}\mid\mathbf{z})$ is the prior times the decoder's likelihood, and $q$ is the encoder's normal. The hard part is the expectation: it integrates over every possible $\mathbf{z}$, which has no formula when a neural network is involved.

來源的第一種作法（Type 1）是**蒙地卡羅估計**（Monte Carlo estimate）：從 $q$ 抽 $\ell$ 個 $\mathbf{z}_1,\dots,\mathbf{z}_\ell$，用平均代替期望值：

The source's first approach (Type 1) is a **Monte Carlo estimate**: draw $\ell$ values $\mathbf{z}_1,\dots,\mathbf{z}_\ell$ from $q$ and replace the expectation by an average:

$$
\tilde{\mathcal{L}}_{1}=\frac{1}{\ell}\sum_{j=1}^{\ell}\Big[\log p(\mathbf{x},\mathbf{z}_j)-\log q(\mathbf{z}_j\mid\mathbf{x})\Big].
$$

用 KP02、KP03 的一維例子走一次 `[補充]`：$z\sim\mathcal{N}(0,1)$，$x\mid z\sim\mathcal{N}(z,\ 0.5^2)$，觀察到 $x=1.2$，近似後驗取 $q=\mathcal{N}(0.5,\ 0.2)$（本講義 $\mathcal{N}(\cdot,\cdot)$ 的第二個數是變異數）。KP03 算過，這個 $q$ 的 ELBO 正確值是 $-2.1355$。假設抽到五個 $z$：$0.9,\ 0.2,\ 0.6,\ 0.1,\ 0.7$。括號裡的值依序是 $-1.2155,\ -2.8255,\ -1.9055,\ -3.0555,\ -1.6755$，平均 $-2.1355$。

One pass with the one-dimensional example of KP02 and KP03 `[Added]`: $z\sim\mathcal{N}(0,1)$, $x\mid z\sim\mathcal{N}(z,\ 0.5^2)$, the observation is $x=1.2$, and the approximate posterior is $q=\mathcal{N}(0.5,\ 0.2)$ (in this note the second number in $\mathcal{N}(\cdot,\cdot)$ is the variance). KP03 computed the exact ELBO of this $q$ as $-2.1355$. Suppose the five draws are $z=0.9,\ 0.2,\ 0.6,\ 0.1,\ 0.7$. The bracketed values are $-1.2155,\ -2.8255,\ -1.9055,\ -3.0555,\ -1.6755$, with average $-2.1355$.

這次剛好等於正確值，是巧合 `[補充]`：在這個例子裡，括號裡的值剛好是 $z$ 的一次式（$z^2$ 的係數 $-\tfrac12-2+2.5$ 互相抵消），而五個抽樣值的平均剛好是 $0.5$。一般的五個抽樣值不會這麼巧。

Hitting the exact value this time is a coincidence `[Added]`: in this example the bracketed value happens to be linear in $z$ (the coefficients of $z^2$, $-\tfrac12-2+2.5$, cancel), and the five draws happen to average exactly $0.5$. Five ordinary draws would not be so lucky.

估計會晃動，晃多少用**標準差**量。來源說誤差和 $1/\sqrt{\ell}$ 成正比：$\ell$ 變成 $100$ 倍，誤差只縮成 $1/10$。來源也說實務上常用 $\ell=1$：一個小批次有 $b$ 筆資料，每筆抽一個 $\mathbf{z}$，整批就已經有 $b$ 個抽樣值一起平均。

An estimate wobbles, and the wobble is measured by its **standard deviation**. The source says the error is proportional to $1/\sqrt{\ell}$: making $\ell$ $100$ times larger only shrinks the error to $1/10$. The source also says that $\ell=1$ is common in practice: a mini-batch has $b$ examples, each draws one $\mathbf{z}$, so the batch already averages $b$ sampled values.

常見錯誤：以為 $\ell$ 越大越好。來源提醒，$\ell$ 增加，計算量跟著線性增加，誤差卻只按 $1/\sqrt{\ell}$ 下降，很快就不划算。

Common mistake: thinking the larger $\ell$, the better. The source points out that the cost grows linearly with $\ell$ while the error falls only as $1/\sqrt{\ell}$, so it soon stops paying off.

### 2. 把 KL 拆出來，用公式算（Type 2：split off the KL and compute it by formula）

KP03 的第二種寫法把 ELBO 拆成兩項。來源的第二種作法（Type 2）就從這裡出發：

KP03's second form splits the ELBO into two terms. The source's second approach (Type 2) starts from there:

$$
\tilde{\mathcal{L}}_{2}=\frac{1}{\ell}\sum_{j=1}^{\ell}\log p(\mathbf{x}\mid\mathbf{z}_j)\;-\;D_{\text{KL}}\big(q(\mathbf{z}\mid\mathbf{x})\,\Vert\,p(\mathbf{z})\big).
$$

重建項還是要抽樣，但 KL 項不用：當 $q$ 和先驗都是常態分布時，KL 有公式解。來源給了兩個引理。一維時（Lemma 1）：

The reconstruction term still needs sampling, but the KL term does not: when $q$ and the prior are both normal, the KL has a closed form. The source gives two lemmas. In one dimension (Lemma 1):

$$
D_{\text{KL}}\big(\mathcal{N}(\mu_1,\sigma_1^2)\,\Vert\,\mathcal{N}(\mu_2,\sigma_2^2)\big)=\log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac12 .
$$

多維時（Lemma 2）把先驗換成 $\mathcal{N}(\mathbf{0},I)$，$p$ 維的結果是（這裡的 $p$ 是潛在維度，沿用來源與 KP05 的記號，不是機率密度）

In several dimensions (Lemma 2), with the prior $\mathcal{N}(\mathbf{0},I)$, the result in $p$ dimensions is (here $p$ is the latent dimension, following the source and KP05, not a density)

$$
D_{\text{KL}}\big(\mathcal{N}(\boldsymbol{\mu},\Sigma)\,\Vert\,\mathcal{N}(\mathbf{0},I)\big)=\frac12\Big[-\log\lvert\Sigma\rvert-p+\operatorname{tr}\Sigma+\lVert\boldsymbol{\mu}\rVert^2\Big].
$$

祕笈的例子：$\boldsymbol{\mu}=(0.5,\ -0.3)$，$\Sigma=\operatorname{diag}(0.1,\ 0.2)$。代入得 $\tfrac12[-\log0.02-2+0.3+0.34]=\tfrac12[3.912-2+0.64]=1.276$。`[補充]` 對角共變異數時，這就是每一維各用一次 Lemma 1 再相加：兩維各自的 KL 加起來也是 $1.276$。KP05 前向計算的 KL $0.8181$，就是 Lemma 1 代入 $\mu_1=1$、$\sigma_1=0.5$、先驗 $\mathcal{N}(0,1)$。

The professor's notes give an example: $\boldsymbol{\mu}=(0.5,\ -0.3)$ and $\Sigma=\operatorname{diag}(0.1,\ 0.2)$. Substituting gives $\tfrac12[-\log0.02-2+0.3+0.34]=\tfrac12[3.912-2+0.64]=1.276$. `[Added]` With a diagonal covariance this is just Lemma 1 applied to each dimension and summed: the two one-dimensional KLs also add up to $1.276$. The KL of $0.8181$ in KP05's forward pass is Lemma 1 with $\mu_1=1$, $\sigma_1=0.5$ and the prior $\mathcal{N}(0,1)$.

祕笈也拆解了三項各自的作用：$-\log\lvert\Sigma\rvert$ 希望變異數大，$\operatorname{tr}\Sigma$ 希望變異數小，兩者在變異數 $1$ 時取得平衡（`[補充]`：一維時是 $-\log v+v$，在 $v=1$ 最小）；$\lVert\boldsymbol{\mu}\rVert^2$ 把平均拉向原點。

The professor's notes also break down the three parts: $-\log\lvert\Sigma\rvert$ wants large variances, $\operatorname{tr}\Sigma$ wants small ones, and the two balance at variance $1$ (`[Added]`: in one dimension this is $-\log v+v$, smallest at $v=1$); $\lVert\boldsymbol{\mu}\rVert^2$ pulls the mean towards the origin.

來源說 Type 2 比 Type 1 穩定，因為 KL 部分不再有抽樣的晃動。這句話常常對，但不是一定 `[補充]`。下圖用第1節的例子比較兩種估計的晃動：

The source says Type 2 is more stable than Type 1, because the KL part no longer wobbles with sampling. This is often true but not guaranteed `[Added]`. The figure below compares the wobble of the two estimates in Section 1's example:

![左右兩張對數座標圖，橫軸是每次估計用幾個抽樣值，從 1 到 1000；縱軸是估計值的標準差。每張圖有兩條往右下的直線：橘色圓點是 Type 1，青色方塊是 Type 2。左圖 q 是 N(0.5, 0.2)，橘線在青線下方；右圖 q 是較窄的 N(0.96, 0.05)，青線在橘線下方。兩條線的斜率都是負二分之一（Two log-log panels; the horizontal axis is the number of samples per estimate, from 1 to 1000, and the vertical axis is the standard deviation of the estimate. Each panel has two straight lines going down to the right: orange circles for Type 1 and teal squares for Type 2. Left, q is N(0.5, 0.2) and the orange line lies below the teal one; right, q is the narrower N(0.96, 0.05) and the teal line lies below the orange one. Both lines have slope minus one half）](../assets/figures/lec12-kp06-mc-estimates.png)

**圖說 `lec12-kp06-mc-estimates`**〔功能：精準度、比較〕：先看兩條線的斜率：每多 $10$ 倍的抽樣值，標準差都縮成約 $1/\sqrt{10}$，這就是 $1/\sqrt{\ell}$。再看左圖：$q=\mathcal{N}(0.5,\ 0.2)$ 時，抽一個值的標準差 Type 1 是 $1.03$、Type 2 是 $1.38$，Type 1 反而比較穩。最後看右圖：$q$ 窄成 $\mathcal{N}(0.96,\ 0.05)$ 時，Type 1 是 $0.53$、Type 2 是 $0.26$，換 Type 2 比較穩。

**Caption `lec12-kp06-mc-estimates`** [Function: precision, comparison]: first look at the slopes: every $10$ times more samples shrinks the standard deviation to about $1/\sqrt{10}$, which is $1/\sqrt{\ell}$. Then look at the left panel: with $q=\mathcal{N}(0.5,\ 0.2)$, the standard deviation for one sample is $1.03$ for Type 1 and $1.38$ for Type 2, so Type 1 is actually steadier. Finally look at the right panel: when $q$ narrows to $\mathcal{N}(0.96,\ 0.05)$, Type 1 is $0.53$ and Type 2 is $0.26$, so now Type 2 is steadier.

為什麼會這樣 `[補充]`：Type 1 括號裡的 $\log p(\mathbf{x},\mathbf{z})$ 和 $\log q$ 會一起漲落、互相抵消；$q$ 正好等於後驗時，兩者的差是常數 $\log p(\mathbf{x})$，抽一個值就完全準確。可是 $\log q$ 的晃動大小是固定的：把 $\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$ 代進去，$\log q=$ 常數 $-\tfrac12\lVert\boldsymbol{\epsilon}\rVert^2$，每一維的標準差約 $0.71$，不管 $q$ 多窄都一樣。$q$ 很窄時，$\log p(\mathbf{x},\mathbf{z})$ 幾乎不動（本例標準差從 $1.25$ 降到 $0.18$），抵消不了這份固定的晃動，Type 1 就被它拖住；Type 2 把它換成公式就賺到了。VAE 訓練好的 $q$ 通常比先驗窄得多（KP05 小 VAE 的標準差只有 $0.045$ 到 $0.326$），所以實務上 Type 2 通常比較穩；加上 KL 公式本身沒有隨機性、梯度也乾淨，這就是來源說「Type 2 加常態分布」成為標準作法的原因。

Why this happens `[Added]`: in Type 1, $\log p(\mathbf{x},\mathbf{z})$ and $\log q$ rise and fall together and partly cancel; when $q$ equals the posterior, their difference is the constant $\log p(\mathbf{x})$, and a single sample is exact. But the wobble of $\log q$ has a fixed size: substituting $\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$ gives $\log q=$ constant $-\tfrac12\lVert\boldsymbol{\epsilon}\rVert^2$, with standard deviation about $0.71$ per dimension however narrow $q$ is. When $q$ is narrow, $\log p(\mathbf{x},\mathbf{z})$ barely moves (in this example its standard deviation drops from $1.25$ to $0.18$) and cannot cancel that fixed wobble, so Type 1 is held back by it; Type 2 replaces it by a formula and comes out ahead. A trained VAE's $q$ is usually much narrower than the prior (the standard deviations of KP05's small VAE are only $0.045$ to $0.326$), so in practice Type 2 is usually steadier; and since the KL formula has no randomness and gives clean gradients, this is why the source calls "Type 2 plus normal distributions" the standard practice.

來源在這裡還提到**先驗正規化**（prior regularization）：先驗不一定要用 $\mathcal{N}(\mathbf{0},I)$，可以換成混合常態或其他有結構的分布，把領域知識放進潛在空間、讓潛在軸更好解釋、讓生成更合理。來源說代價是 ELBO 和真正的對數概似之間的差距可能變大；另外 `[補充]`，KL 也可能不再有公式解。本講義只帶過這一點。

The source also mentions **prior regularization** here: the prior need not be $\mathcal{N}(\mathbf{0},I)$; it can be a normal mixture or another structured distribution that brings domain knowledge into the latent space, makes the latent axes easier to interpret and makes generation more sensible. The source says the price is that the gap between the ELBO and the true log-likelihood may grow; also `[Added]`, the KL may lose its closed form. This note only touches on it.

### 3. 重參數化技巧：讓梯度穿過抽樣（The reparameterization trick: letting gradients pass through sampling）

訓練要對編碼器的權重 $\theta_e$ 求梯度，路徑是：$\theta_e\to(\boldsymbol{\mu},\boldsymbol{\sigma})\to\mathbf{z}\to$ 解碼器 $\to$ 損失。問題出在中間那一步：如果 $\mathbf{z}$ 是「從 $\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\sigma}^2)$ 抽出來的」，它就只是一個亂數，不是 $\boldsymbol{\mu}$ 的函數，「$\boldsymbol{\mu}$ 變大一點，$\mathbf{z}$ 會變多少」沒有答案，反向傳播在這裡斷掉。

Training needs the gradient with respect to the encoder's weights $\theta_e$, along the path $\theta_e\to(\boldsymbol{\mu},\boldsymbol{\sigma})\to\mathbf{z}\to$ decoder $\to$ loss. The trouble is the middle step: if $\mathbf{z}$ is "drawn from $\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\sigma}^2)$", it is just a random number, not a function of $\boldsymbol{\mu}$, so "how much does $\mathbf{z}$ change when $\boldsymbol{\mu}$ grows a little" has no answer, and backpropagation breaks here.

**重參數化技巧**（reparameterization trick，Kingma 與 Welling 2014）換一種方式得到同一個分布：先從標準常態抽一個雜訊 $\boldsymbol{\epsilon}$，再算

The **reparameterization trick** (Kingma and Welling, 2014) gets the same distribution another way: first draw noise $\boldsymbol{\epsilon}$ from the standard normal, then compute

$$
\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon},\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},I).
$$

隨機性被移到 $\boldsymbol{\epsilon}$ 身上，而 $\boldsymbol{\epsilon}$ 和網路的權重無關，可以當成一個外加的輸入。現在 $\mathbf{z}$ 是 $\boldsymbol{\mu}$、$\boldsymbol{\sigma}$ 的普通函數，每一維都有

The randomness moves onto $\boldsymbol{\epsilon}$, which does not depend on the network's weights and can be treated as an extra input. Now $\mathbf{z}$ is an ordinary function of $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$, and in each dimension

$$
\frac{\partial z}{\partial\mu}=1,\qquad \frac{\partial z}{\partial\sigma}=\epsilon .
$$

祕笈的例子：$\mu=0.5$、$\sigma=0.8$，抽到 $\epsilon=-0.3$，得 $z=0.5+0.8\times(-0.3)=0.26$。`[補充]` 分布沒有變：抽 $10000$ 個 $\epsilon$ 算出的 $z$，平均 $0.510$、標準差 $0.804$，和 $\mathcal{N}(0.5,\ 0.8^2)$ 吻合。

The professor's notes give an example: $\mu=0.5$, $\sigma=0.8$, and the draw $\epsilon=-0.3$ give $z=0.5+0.8\times(-0.3)=0.26$. `[Added]` The distribution is unchanged: the $z$ computed from $10000$ draws of $\epsilon$ have mean $0.510$ and standard deviation $0.804$, matching $\mathcal{N}(0.5,\ 0.8^2)$.

![左右兩張圖。左邊是兩條流程：上方「之前」，紫色的 mu、sigma 方塊接到紅色的「抽樣 z」方塊再接到損失，箭頭下寫著 z 對 mu 的偏微分等於問號；下方「之後」，mu、sigma 接到青色的 z 等於 mu 加 sigma 乘 epsilon 方塊再接到損失，下方黃色的 epsilon 方塊從外面接進來，一條藍色虛線箭頭往回指向 mu、sigma，標著 z 對 mu 的偏微分是 1、對 sigma 是 epsilon。右邊是 10000 個 mu 加 sigma 乘 epsilon 的直方圖，和藍色的常態曲線 N(0.5, 0.8 平方) 重合，三條紅色虛線標在 -0.3、0.5、1.3，標題寫著樣本平均 0.510、標準差 0.804（Two panels. Left: two flows. Top, "before": a purple mu, sigma box feeds a red "sample z" box, then the loss, with "partial z over partial mu = ?" under the arrow. Bottom, "after": mu, sigma feed a teal box z = mu + sigma epsilon, then the loss; a yellow epsilon box feeds in from outside, and a dashed blue arrow points back to mu, sigma, labelled partial z over partial mu is 1 and over sigma is epsilon. Right: a histogram of 10000 values of mu + sigma epsilon matching the blue normal curve N(0.5, 0.8 squared), with red dashed lines at -0.3, 0.5 and 1.3; the title says sample mean 0.510, sd 0.804）](../assets/figures/lec12-kp06-reparameterization.png)

**圖說 `lec12-kp06-reparameterization`**〔功能：一目了然、精準度〕：先看左上：抽樣是一個隨機節點，梯度沒有路可以回到 $\mu$、$\sigma$。再看左下：把隨機性交給外面進來的 $\epsilon$，$z$ 變成普通的加法和乘法，藍色虛線就是梯度回去的路。最後看右圖：用這個方式算出的 $z$，直方圖和 $\mathcal{N}(0.5,\ 0.8^2)$ 的曲線重合，三條紅線是 $\mu-\sigma$、$\mu$、$\mu+\sigma$：分布沒有變。

**Caption `lec12-kp06-reparameterization`** [Function: seeing it at a glance, precision]: first look at the top left: sampling is a random node, and gradients have no path back to $\mu$ and $\sigma$. Then look at the bottom left: hand the randomness to $\epsilon$ coming in from outside, and $z$ becomes ordinary addition and multiplication; the dashed blue arrow is the gradient's way back. Finally look at the right panel: the histogram of $z$ computed this way matches the curve of $\mathcal{N}(0.5,\ 0.8^2)$, and the three red lines mark $\mu-\sigma$, $\mu$ and $\mu+\sigma$: the distribution is unchanged.

梯度真的對嗎 `[補充]`？取一個簡單的函數 $f(z)=z^2$，它的期望值有公式 $\mathbb{E}[z^2]=\mu^2+\sigma^2$，所以正確的梯度是 $2\mu=1$ 與 $2\sigma=1.6$。用重參數化，每個抽樣值的梯度是 $2z\cdot1$ 與 $2z\cdot\epsilon$；用同樣 $10000$ 個 $\epsilon$ 平均，得到 $1.021$ 與 $1.63$，和正確值只差抽樣的晃動。

Are the gradients really right `[Added]`? Take the simple function $f(z)=z^2$, whose expectation has the formula $\mathbb{E}[z^2]=\mu^2+\sigma^2$, so the correct gradients are $2\mu=1$ and $2\sigma=1.6$. With reparameterization, each sample's gradients are $2z\cdot1$ and $2z\cdot\epsilon$; averaging over the same $10000$ draws of $\epsilon$ gives $1.021$ and $1.63$, off from the correct values only by sampling wobble.

來源也說明了適用範圍：常態、邏輯斯（以及形狀參數固定的伽瑪）這類能寫成「位置加尺度乘雜訊」的分布都可以這樣做。離散的潛在變數（例如只取 $0$ 或 $1$）在兩個值之間沒有平滑的路，不能直接重參數化，要改用 Gumbel-Softmax 或 REINFORCE 之類的方法。

The source also states where it applies: normal, logistic (and gamma with a fixed shape parameter) and other distributions that can be written as "location plus scale times noise" all work this way. A discrete latent variable (taking only $0$ or $1$, say) has no smooth path between its values and cannot be reparameterized directly; methods such as Gumbel-Softmax or REINFORCE are used instead.

常見錯誤：以為重參數化改變了 $\mathbf{z}$ 的分布，或只是加快抽樣。分布完全一樣；改變的是計算圖的寫法，讓抽樣的隨機性不擋在梯度的路上。

Common mistake: thinking reparameterization changes the distribution of $\mathbf{z}$, or merely speeds up sampling. The distribution is exactly the same; what changes is how the computation graph is written, so that the randomness of sampling no longer blocks the gradient's path.

### 4. 從 EM 到一起反向傳播（From EM to joint backpropagation）

來源先用 KP04 的 EM 來看 VAE：編碼器扮演 $q$，所以「更新 $\theta_e$」對應 $\text{E}$ 步；解碼器扮演概似，「更新 $\theta_d$」對應 $\text{M}$ 步。兩步都用梯度往上爬一小步，而不是真的求最大值。

The source first views the VAE through KP04's EM: the encoder plays $q$, so "update $\theta_e$" corresponds to the E-step; the decoder plays the likelihood, so "update $\theta_d$" corresponds to the M-step. Each step climbs a small gradient step instead of truly maximising.

但來源接著說，實務上不交替，而是把編碼器、重參數化、解碼器串成一張可以微分的計算圖，一次更新全部權重 $\theta=\{\theta_e,\theta_d\}$：

But the source goes on to say that in practice one does not alternate; the encoder, the reparameterization and the decoder are chained into one differentiable computation graph, and all weights $\theta=\{\theta_e,\theta_d\}$ are updated at once:

$$
\theta\leftarrow\theta-\eta\,\frac{\partial}{\partial\theta}\sum_{i=1}^{b}\Big[-\tilde{\mathcal{L}}(\theta,\mathbf{x}_i)\Big].
$$

最小化 $-\tilde{\mathcal{L}}$ 就是最大化 ELBO。來源寫成實務上的損失：

Minimising $-\tilde{\mathcal{L}}$ is maximising the ELBO. The source writes the practical loss as

$$
\text{loss}=\text{重建誤差 reconstruction error}+\beta\cdot D_{\text{KL}}\big(q(\mathbf{z}\mid\mathbf{x})\,\Vert\,p(\mathbf{z})\big),
$$

其中 $\beta$ 是 KL 的權重，標準 VAE 取 $\beta=1$。每一步的流程是：取一小批 $b$ 筆資料；編碼器算出 $\boldsymbol{\mu},\boldsymbol{\sigma}$；抽 $\boldsymbol{\epsilon}$ 算 $\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$；解碼器算出 $\hat{\mathbf{x}}$；算損失；反向傳播更新 $\theta$。

where $\beta$ is the weight of the KL, with $\beta=1$ for a standard VAE. Each step goes: take a mini-batch of $b$ examples; the encoder computes $\boldsymbol{\mu},\boldsymbol{\sigma}$; draw $\boldsymbol{\epsilon}$ and compute $\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$; the decoder computes $\hat{\mathbf{x}}$; compute the loss; backpropagate and update $\theta$.

重建誤差前面的倍數其實很重要 `[補充]`。設解碼器的概似是平均 $\hat{\mathbf{x}}$、標準差 $\sigma_x$ 的常態分布，則 $-\log p(\mathbf{x}\mid\mathbf{z})=\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2/(2\sigma_x^2)+$ 常數（KP05 第3節只寫了「乘上一個常數」）。KP05 的小 VAE 用 $\sigma_x=0.1$，損失是 $50\times$ 平方誤差 $+$ KL，和「平方誤差 $+\,0.02\times$ KL」成正比（差 $50$ 倍），最佳解相同。所以若直接寫「平方誤差 $+$ KL」，等於偷偷選了一個 $\sigma_x$，也就選了一個 $\beta$。

The multiplier in front of the reconstruction error matters a lot `[Added]`. If the decoder's likelihood is a normal with mean $\hat{\mathbf{x}}$ and standard deviation $\sigma_x$, then $-\log p(\mathbf{x}\mid\mathbf{z})=\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2/(2\sigma_x^2)+$ constant (KP05 Section 3 only said "times a constant"). KP05's small VAE uses $\sigma_x=0.1$, so its loss is $50\times$ the squared error $+$ KL, proportional to "squared error $+\,0.02\times$ KL" (a factor of $50$), with the same minimiser. Writing plainly "squared error $+$ KL" therefore quietly picks a $\sigma_x$, and with it a $\beta$.

KP05 的小 VAE 就是這樣訓練出來的 `[補充]`：$300$ 個點一次全部當一批，每步每點抽一個 $\epsilon$（$\ell=1$），用 Adam 訓練 $20000$ 步。程式裡手寫的梯度抽查 $4$ 個參數和數值微分比對，最大相對誤差 $3\times10^{-9}$。每點的損失從第 $1$ 步的 $139.39$ 掉到第 $2000$ 步的 $2.97$，最後是 $2.84$，其中重建項 $0.60$、KL 項 $2.24$。

KP05's small VAE was trained exactly this way `[Added]`: all $300$ points form one batch, each step draws one $\epsilon$ per point ($\ell=1$), and Adam runs for $20000$ steps. The hand-written gradients in the code, spot-checked on $4$ parameters, agree with numerical differentiation to a maximum relative error of $3\times10^{-9}$. The loss per point falls from $139.39$ at step $1$ to $2.97$ at step $2000$, ending at $2.84$, of which the reconstruction term is $0.60$ and the KL term $2.24$.

![左右兩張圖。左圖是訓練曲線，橫軸是訓練步數 0 到 20000，縱軸是對數刻度：藍線重建項從一百多一下子掉到 1 以下並在 0.5 左右晃動；紅線 KL 項從 2 升到約 2.4 到 2.5 後慢慢降到 2.2；青色虛線總損失從一百多掉到 3 附近後持平。右圖是 beta 等於 0、沒有 KL 項時的生成結果：淺藍點是訓練資料，紅點是從標準常態抽 z 解碼出來的點，全部落在圓上，但連成一條從右側約 0 度一直繞到下方約 270 度的完整弧線，填滿了兩段資料之間左上方的空缺，標題寫著 100% 落在圓上、64% 落在資料的圓弧上（Two panels. Left, training curves over 0 to 20000 steps on a log scale: the blue reconstruction term drops from over a hundred to below 1 and wobbles around 0.5; the red KL term rises from 2 to about 2.4–2.5 and slowly falls to 2.2; the dashed teal total loss drops from over a hundred to about 3 and levels off. Right, generation with beta equal to 0, no KL term: light-blue training data, and red points decoded from z drawn from the standard normal, all on the circle but forming one long arc from about 0 degrees on the right round to about 270 degrees at the bottom, filling the empty upper-left gap between the two data arcs; the title says 100% on the circle, 64% on the data arcs）](../assets/figures/lec12-kp06-training-curves.png)

**圖說 `lec12-kp06-training-curves`**〔功能：一目了然、比較〕：先看左圖：重建項（藍）在前 $500$ 步內就大幅下降，KL 項（紅）先升後緩降，兩者拉扯之後總損失（青）持平，這是 KP05 第3節說的平衡。再看右圖：把 $\beta$ 設成 $0$、只留重建項重新訓練，從先驗抽的 $z$ 解碼後全都在圓上，卻有 $36\%$ 落在兩段資料之間沒有資料的地方。

**Caption `lec12-kp06-training-curves`** [Function: seeing it at a glance, comparison]: first look at the left panel: the reconstruction term (blue) drops sharply within the first $500$ steps, the KL term (red) rises and then slowly falls, and after this tug of war the total loss (teal) levels off; this is the balance described in KP05 Section 3. Then look at the right panel: retrain with $\beta=0$, keeping only the reconstruction term, and the decoded draws from the prior all land on the circle, yet $36\%$ fall in the empty stretch between the two data arcs.

拿掉 KL 項會怎樣 `[補充]`：$\beta=0$ 時重建項降到 $0.03$，比 $\beta=1$ 好得多；但每個點的標準差中位數縮到 $0.0001$，$q$ 幾乎變成一個點，編碼平均散在 $-3.3$ 到 $1.9$，KL 若照算是 $10.53$。編碼器又回到 KP01 那種「每筆一個確定編碼」的狀態，先驗的範圍裡沒有被整理好：生成時落在資料圓弧上的比例從 $92.6\%$ 掉到 $64\%$。這就是祕笈說的：只重建，潛在空間就失去結構，無法好好生成。

What happens without the KL term `[Added]`: with $\beta=0$ the reconstruction term drops to $0.03$, much better than with $\beta=1$; but the median standard deviation per point shrinks to $0.0001$, so $q$ is almost a single point, the code means spread from $-3.3$ to $1.9$, and the KL, if computed, would be $10.53$. The encoder is back to KP01's "one fixed code per example", and the range of the prior is not organised: the share of generated points on the data arcs drops from $92.6\%$ to $64\%$. This is what the professor's notes mean: with reconstruction alone, the latent space loses its structure and cannot generate well.

常見錯誤：以為 VAE 訓練一定要嚴格交替 $\text{E}$ 步與 $\text{M}$ 步。來源說 EM 只是理解的框架，實務上編碼器和解碼器一起用小批次梯度下降更新，比較穩定。

Common mistake: thinking VAE training must strictly alternate the E-step and the M-step. The source says EM is only a framework for understanding; in practice the encoder and decoder are updated together by mini-batch gradient descent, which is more stable.

### 本單元重點整理（KP06 summary）

**一句話重點**：ELBO 的期望值用抽樣平均來估計，誤差按 $1/\sqrt{\ell}$ 縮小；常態分布的 KL 有公式，拆出來算（Type 2）通常比較穩；重參數化 $\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$ 把隨機性移到外加的 $\boldsymbol{\epsilon}$，讓梯度穿過抽樣；最後把編碼器與解碼器串成一張圖，用「重建誤差 $+\ \beta\cdot$ KL」一起反向傳播訓練。

**One-sentence summary**: the expectation in the ELBO is estimated by a sample average whose error shrinks as $1/\sqrt{\ell}$; the KL between normals has a formula, and computing it separately (Type 2) is usually steadier; reparameterization $\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$ moves the randomness onto an outside $\boldsymbol{\epsilon}$ so gradients pass through sampling; finally the encoder and decoder are chained into one graph and trained together by backpropagation on "reconstruction error $+\ \beta\cdot$ KL".

**白話說明** `[補充]`（後半改寫自祕笈的工廠抽籤比喻）：老師要知道全班平均身高，不必量完每個人，隨機抽幾個人量就有個估計，抽越多越準，但準得越來越慢。重參數化則像把「抽籤」交給外面的公證人：老師只負責決定籤筒的中心和寬度（$\mu,\sigma$），抽出哪支籤由公證人決定；結果不好時，老師知道該怎麼調整中心和寬度。

**Plain explanation** `[Added]` (the second half adapts the professor's notes' factory-lottery analogy): to know the class's average height, a teacher need not measure everyone; measuring a few randomly chosen students gives an estimate, more accurate with more students, but improving ever more slowly. Reparameterization is like handing the lottery to an outside notary: the teacher only sets the centre and width of the draw ($\mu,\sigma$), and the notary decides which ticket comes out; when the result is poor, the teacher knows how to adjust the centre and width.

**正式定義**：$\tilde{\mathcal{L}}_1=\frac1\ell\sum_j[\log p(\mathbf{x},\mathbf{z}_j)-\log q(\mathbf{z}_j\mid\mathbf{x})]$；$\tilde{\mathcal{L}}_2=\frac1\ell\sum_j\log p(\mathbf{x}\mid\mathbf{z}_j)-D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}))$；$D_{\text{KL}}(\mathcal{N}(\boldsymbol{\mu},\Sigma)\Vert\mathcal{N}(\mathbf{0},I))=\tfrac12[-\log\lvert\Sigma\rvert-p+\operatorname{tr}\Sigma+\lVert\boldsymbol{\mu}\rVert^2]$；$\mathbf{z}_j=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}_j$，$\boldsymbol{\epsilon}_j\sim\mathcal{N}(\mathbf{0},I)$；$\theta\leftarrow\theta-\eta\,\partial(-\sum_i\tilde{\mathcal{L}})/\partial\theta$。

**Formal definition**: $\tilde{\mathcal{L}}_1=\frac1\ell\sum_j[\log p(\mathbf{x},\mathbf{z}_j)-\log q(\mathbf{z}_j\mid\mathbf{x})]$; $\tilde{\mathcal{L}}_2=\frac1\ell\sum_j\log p(\mathbf{x}\mid\mathbf{z}_j)-D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}))$; $D_{\text{KL}}(\mathcal{N}(\boldsymbol{\mu},\Sigma)\Vert\mathcal{N}(\mathbf{0},I))=\tfrac12[-\log\lvert\Sigma\rvert-p+\operatorname{tr}\Sigma+\lVert\boldsymbol{\mu}\rVert^2]$; $\mathbf{z}_j=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}_j$, $\boldsymbol{\epsilon}_j\sim\mathcal{N}(\mathbf{0},I)$; $\theta\leftarrow\theta-\eta\,\partial(-\sum_i\tilde{\mathcal{L}})/\partial\theta$.

**例子**：五個抽樣值的 Type 1 估計 $-2.1355$；祕笈例子的 KL $1.276$；$\mu=0.5$、$\sigma=0.8$、$\epsilon=-0.3$ 得 $z=0.26$；$\mathbb{E}[z^2]$ 的梯度正確值 $(1,\ 1.6)$，重參數化估計 $(1.021,\ 1.63)$；小 VAE 最後損失 $2.84$，拿掉 KL 後落在資料圓弧上的比例從 $92.6\%$ 掉到 $64\%$。

**Example**: the Type 1 estimate from five draws is $-2.1355$; the KL of the notes' example is $1.276$; $\mu=0.5$, $\sigma=0.8$, $\epsilon=-0.3$ give $z=0.26$; the gradients of $\mathbb{E}[z^2]$ are exactly $(1,\ 1.6)$ and the reparameterized estimates are $(1.021,\ 1.63)$; the small VAE ends at loss $2.84$, and without the KL the share on the data arcs drops from $92.6\%$ to $64\%$.

**常見迷思**：

- 「抽樣數 $\ell$ 越大越好。」錯，誤差只按 $1/\sqrt{\ell}$ 下降，常用 $\ell=1$。
- 「Type 2 一定比 Type 1 穩。」不一定，看 $q$ 的形狀；VAE 的窄 $q$ 通常是 Type 2 較穩。
- 「重參數化改變了 $\mathbf{z}$ 的分布。」錯，分布一樣，只是梯度有路可走。
- 「VAE 一定要交替 $\text{E}$ 步與 $\text{M}$ 步。」錯，實務上一起更新。
- 「KL 項只會讓重建變差，可以拿掉。」錯，拿掉之後潛在空間失去結構，生成變差。

**Common misconceptions**:

- "The larger the number of samples $\ell$, the better." Wrong: the error falls only as $1/\sqrt{\ell}$, and $\ell=1$ is common.
- "Type 2 is always steadier than Type 1." Not always; it depends on the shape of $q$, though for a VAE's narrow $q$ Type 2 is usually steadier.
- "Reparameterization changes the distribution of $\mathbf{z}$." Wrong: the distribution is the same; the gradient just gets a path.
- "A VAE must alternate the E-step and the M-step." Wrong: in practice both are updated together.
- "The KL term only hurts reconstruction and can be dropped." Wrong: without it the latent space loses structure and generation gets worse.

**圖**：本單元有三張圖。

- `lec12-kp06-mc-estimates`：兩種估計的晃動都按 $1/\sqrt{\ell}$ 縮小，哪一種比較穩要看 $q$。
- `lec12-kp06-reparameterization`：重參數化前後的計算圖，以及抽樣分布沒有改變。
- `lec12-kp06-training-curves`：小 VAE 的訓練曲線，以及拿掉 KL 項之後的生成結果。

**Figures** — this unit has three figures.

- `lec12-kp06-mc-estimates`: the wobble of both estimates shrinks as $1/\sqrt{\ell}$, and which is steadier depends on $q$.
- `lec12-kp06-reparameterization`: the computation graph before and after reparameterization, and the unchanged sampling distribution.
- `lec12-kp06-training-curves`: the small VAE's training curves, and generation after removing the KL term.

### 小練習（Practice）

1. 一個 Type 1 估計用 $\ell=1$ 時標準差是 $1.0$。要把標準差降到 $0.1$，$\ell$ 要多少？
2. 編碼器輸出一維的 $\mu=0$、$\sigma=2$。用 Lemma 1 算它和 $\mathcal{N}(0,1)$ 的 KL。
3. $\mu=1$、$\sigma=0.5$，抽到 $\epsilon=0.6$。算 $z$，以及 $z$ 對 $\mu$、對 $\sigma$ 的偏微分。

1. A Type 1 estimate has standard deviation $1.0$ with $\ell=1$. How large must $\ell$ be to bring it down to $0.1$?
2. An encoder outputs a one-dimensional $\mu=0$ and $\sigma=2$. Use Lemma 1 to compute its KL from $\mathcal{N}(0,1)$.
3. With $\mu=1$ and $\sigma=0.5$, the draw is $\epsilon=0.6$. Compute $z$ and the partial derivatives of $z$ with respect to $\mu$ and $\sigma$.

**第 1 題解答** `[補充]`：標準差和 $1/\sqrt{\ell}$ 成正比，要縮成 $1/10$，$\ell$ 要 $100$。

**Solution to question 1** `[Added]`: the standard deviation is proportional to $1/\sqrt{\ell}$, so shrinking it to $1/10$ needs $\ell=100$.

**第 2 題解答** `[補充]`：$\log\tfrac12+\tfrac{4+0}{2}-\tfrac12=-0.6931+2-0.5=0.8069$。變異數比先驗大，也要付 KL。

**Solution to question 2** `[Added]`: $\log\tfrac12+\tfrac{4+0}{2}-\tfrac12=-0.6931+2-0.5=0.8069$. A variance larger than the prior's also costs KL.

**第 3 題解答** `[補充]`：$z=1+0.5\times0.6=1.3$（就是 KP05 前向計算的 $z$）；$\partial z/\partial\mu=1$，$\partial z/\partial\sigma=\epsilon=0.6$。

**Solution to question 3** `[Added]`: $z=1+0.5\times0.6=1.3$ (the same $z$ as KP05's forward pass); $\partial z/\partial\mu=1$ and $\partial z/\partial\sigma=\epsilon=0.6$.

### 小測驗（Quiz，來源原題）

這裡收來源 §3.1.2 的一題（KP05 留到這裡），以及 §3.2–§3.4 八題中的七題，保留原題與全部選項（只把步驟名稱改成數學排版）；§3.2.3 那一題的選項有誤，見「來源更正」。來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

Kept here are the one question of the source's §3.1.2 (left over from KP05) and seven of the eight questions in §3.2–§3.4, with all their options (only the step names are typeset as math). The question in §3.2.3 has a faulty option; see the corrections. The source page does not mark the answers, so the explanations below are judged from the source text.

1. VAE 的 $\text{E}$-step 在更新誰的參數？
   - (A) decoder $\theta_d$
   - (B) encoder $\theta_e$
   - (C) prior
   - (D) loss function
2. Type 1 簡化的特點是？
   - (A) 拆 KL 計算
   - (B) 不拆 KL，直接 Monte Carlo 估計期望
   - (C) 只適用高斯
   - (D) 需要 analytical KL
3. Type 2 簡化後 ELBO 的兩項是？
   - (A) reconstruction + KL regularization
   - (B) MSE + 交叉熵
   - (C) log-likelihood + entropy
   - (D) prior + posterior
4. 實務上 VAE 訓練採用？
   - (A) 嚴格交替 $\text{E}/\text{M}$
   - (B) 聯合 backprop（encoder $+$ decoder 一起更新）
   - (C) 只更新 decoder
   - (D) 完全隨機更新
5. VAE 用 prior regularization 的目的？
   - (A) 加速訓練
   - (B) 注入領域知識、提升可解釋性、改善生成
   - (C) 取代 KL 項
   - (D) 避免 backprop
6. reparameterization trick 為何對 backprop 重要？
   - (A) 加速採樣
   - (B) 把隨機性轉到 $\epsilon$，讓 $z=\mu+\sigma\epsilon$ 對 $\mu,\sigma$ 可微
   - (C) 減少參數量
   - (D) 避免 KL
7. reparameterization trick 的核心是？
   - (A) 用更多層 NN
   - (B) 把採樣隨機性拉到 $\epsilon$ 上，讓 $z=\mu+\sigma\epsilon$ 對 $\mu,\sigma$ 可微
   - (C) 改用 discrete latent
   - (D) 放棄 backprop
8. VAE 整體 backprop 訓練時，loss 包含？
   - (A) 只有 reconstruction
   - (B) Reconstruction + $\beta\cdot$KL
   - (C) 只有 KL
   - (D) reconstruction + entropy

1. In a VAE, whose parameters does the $\text{E}$-step update?
   - (A) decoder $\theta_d$
   - (B) encoder $\theta_e$
   - (C) prior
   - (D) loss function
2. What characterises the Type 1 simplification?
   - (A) Computes the KL separately
   - (B) Does not split the KL; estimates the expectation directly by Monte Carlo
   - (C) Only works for Gaussians
   - (D) Needs an analytical KL
3. After the Type 2 simplification, what are the two terms of the ELBO?
   - (A) reconstruction + KL regularization
   - (B) MSE + cross-entropy
   - (C) log-likelihood + entropy
   - (D) prior + posterior
4. In practice, how is a VAE trained?
   - (A) Strictly alternating $\text{E}/\text{M}$
   - (B) Joint backprop (encoder and decoder updated together)
   - (C) Updating only the decoder
   - (D) Completely random updates
5. What is the purpose of prior regularization in a VAE?
   - (A) Faster training
   - (B) Injecting domain knowledge, improving interpretability, improving generation
   - (C) Replacing the KL term
   - (D) Avoiding backprop
6. Why does the reparameterization trick matter for backprop?
   - (A) Faster sampling
   - (B) It moves the randomness to $\epsilon$, making $z=\mu+\sigma\epsilon$ differentiable in $\mu,\sigma$
   - (C) Fewer parameters
   - (D) Avoids the KL
7. What is the core of the reparameterization trick?
   - (A) Using more NN layers
   - (B) Moving the sampling randomness onto $\epsilon$, so that $z=\mu+\sigma\epsilon$ is differentiable in $\mu,\sigma$
   - (C) Switching to a discrete latent
   - (D) Giving up backprop
8. In whole-network backprop training of a VAE, what does the loss contain?
   - (A) Only reconstruction
   - (B) Reconstruction + $\beta\cdot$KL
   - (C) Only KL
   - (D) reconstruction + entropy

**解析** `[補充]`：第 1 題選 (B)：第4節，編碼器扮演 $q$。第 2 題選 (B)：第1節。第 3 題選 (A)：第2節。第 4 題選 (B)：第4節。第 5 題選 (B)：第2節末段。第 6、7 題都選 (B)：第3節。第 8 題選 (B)：第4節。

**Explanations** `[Added]`: question 1 is B: see Section 4; the encoder plays $q$. Question 2 is B: see Section 1. Question 3 is A: see Section 2. Question 4 is B: see Section 4. Question 5 is B: see the end of Section 2. Questions 6 and 7 are B: see Section 3. Question 8 is B: see Section 4.

### 留給下一單元的問題（The question for the next unit）

VAE 訓練好了。接下來要怎麼用它？把一張圖送進去再畫回來（重建），和從先驗抽一個 $\mathbf{z}$ 畫出一張新圖（生成），各有什麼用途？KP05 的小例子已經看到生成不完美；VAE 生成的影像為什麼常常偏模糊，又有哪些改良版本？下一個知識點（本週 KP07）談測試階段、VAE 的變體，以及它和下週 GAN 的關係。

The VAE is trained. How is it used next? Feeding in an image and drawing it back (reconstruction) and drawing a $\mathbf{z}$ from the prior to draw a new image (generation): what is each for? KP05's small example already showed that generation is not perfect; why are VAE images often blurry, and what improved versions exist? The next knowledge point (this week's KP07) covers the test phase, VAE variants, and how the VAE relates to next week's GAN.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第12章「變分模型」§3.2–§3.4 與教授祕笈「用 EM 訓練 VAE」「ELBO Type 1」「ELBO Type 2」「高斯特例的解析 KL」「用近似 ELBO 訓練」「Prior regularization」「Reparameterization trick」「用 backpropagation 訓練 VAE」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch12-variational-models/index.html ，讀取日期 2026-11-14）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §3.2–§3.4 of Chapter 12, "Variational models", of CCC-LNS IDL_115, with the professor's notes "Training a VAE with EM", "ELBO Type 1", "ELBO Type 2", "Closed-form KL for Gaussians", "Training with the approximate ELBO", "Prior regularization", "Reparameterization trick" and "Training a VAE with backpropagation" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch12-variational-models/index.html , accessed 2026-11-14). Content not in the source is marked `[Added]` where it appears.

- 來源把 ELBO 寫成 $-D_{\text{KL}}(\mathbb{P}(z\mid\boldsymbol{x},\theta_e)\,\Vert\,\mathbb{P}(\boldsymbol{x},z\mid\theta_d))$（式 20），並說它「含 intractable 的 $\mathbb{P}(\boldsymbol{x},z\mid\theta_d)$（要對 $z$ 積分）」。$p(\mathbf{x},\mathbf{z})$ 對 $\mathbf{z}$ 不是機率分布，這只是 $\mathbb{E}_q[\log p(\mathbf{x},\mathbf{z})-\log q]$ 的簡寫；而 $p(\mathbf{x},\mathbf{z})=p(\mathbf{z})p(\mathbf{x}\mid\mathbf{z})$ 本身很好算，算不出來的是對 $\mathbf{z}$ 的期望值。本講義第1節這樣寫。
- 來源說抽樣讓「$\partial z/\partial\mu$ 不存在」。精確地說，抽樣節點的輸出不是 $\mu$ 的函數，所以無法沿著它反向傳播；期望值本身對 $\mu$ 的梯度是存在的，重參數化只是換一種寫法把它算出來。
- 來源 Lemma 1 的互動示範說 $\mu_1=2$、$\sigma_1=0.5$ 時 KL「$\approx2.443$」。代入公式是 $\log2+(0.25+4)/2-0.5=0.6931+2.125-0.5=2.318$；用 $200$ 萬個抽樣值估計也是 $2.32$。
- 祕笈說 $\ell=1$、$100$、$10000$ 時「估計誤差比例 $1:\sqrt{100}:\sqrt{10000}=1:10:100$」。誤差是縮小，比例應是 $1:\tfrac1{10}:\tfrac1{100}$；祕笈下一句「樣本增 100 倍、誤差減 10 倍」才是對的。
- 來源說 Type 2「訓練更穩定」「整體梯度變異較小」。這通常成立但不是定理：第2節的例子裡，$q=\mathcal{N}(0.5,\ 0.2)$ 時 Type 1 的標準差反而較小。
- 來源 §3.2.3 的小測驗，選項 (A) 寫成 $-\frac{1}{2}\sum_i(\log|\Sigma_{z\mid x}|+p+\text{tr}(\Sigma_{z\mid x})+\|\mu_{z\mid x}\|^2)$ 並加註「（但符號應為 + log 項）」。展開成 ELBO 的貢獻，$\log$ 項和 $p$ 項的符號其實已經對了，錯的是跡和 $\lVert\mu\rVert^2$ 兩項，加註指向的正好是本來就對的那一項。正確的 KL 是 $\frac12[-\log\lvert\Sigma\rvert-p+\operatorname{tr}\Sigma+\lVert\boldsymbol{\mu}\rVert^2]$。本講義不收這一題。
- 來源說「任何可重參數化的分佈（高斯、Logistic、Gamma via location-scale）」都能用同樣技巧。伽瑪分布只有在形狀參數固定時才是尺度族；要學形狀參數需要其他方法（例如隱式重參數化梯度）。
- 來源把實務損失寫成「MSE $+\ \beta\cdot$ KL」，沒有提到重建誤差前的倍數；本講義補上倍數和 $\sigma_x$、$\beta$ 的關係。
- 來源用 $\mathbb{P}(\cdot)$ 與粗斜體 `\boldsymbol` 表示機率與向量；本講義改用 $p(\cdot)$ 與 `\mathbf`（希臘字母向量 $\boldsymbol{\mu},\boldsymbol{\sigma},\boldsymbol{\epsilon}$ 例外）。小測驗保留來源原本的記號。
- 本講義補上：五個抽樣值的估計與一次式的巧合、Type 1 與 Type 2 的晃動比較與原因、Lemma 2 與 Lemma 1 的對應、$\mathbb{E}[z^2]$ 的梯度驗證、重建誤差倍數與 $\beta$ 的關係、小 VAE 的訓練紀錄與 $\beta=0$ 對照、三張圖、比喻、三題小練習與解答。
- 取捨：來源 Type 1、Type 2 的互動示範數字（Type 1 真值 $0.8$、$\beta$ 權衡的示範值）與附錄中的證明不寫；先驗正規化的各種先驗只列名稱。

- The source writes the ELBO as $-D_{\text{KL}}(\mathbb{P}(z\mid\boldsymbol{x},\theta_e)\,\Vert\,\mathbb{P}(\boldsymbol{x},z\mid\theta_d))$ (Eq. 20) and says it "contains the intractable $\mathbb{P}(\boldsymbol{x},z\mid\theta_d)$ (which needs an integral over $z$)". $p(\mathbf{x},\mathbf{z})$ is not a probability distribution in $\mathbf{z}$, so this is only shorthand for $\mathbb{E}_q[\log p(\mathbf{x},\mathbf{z})-\log q]$; and $p(\mathbf{x},\mathbf{z})=p(\mathbf{z})p(\mathbf{x}\mid\mathbf{z})$ itself is easy to compute; what cannot be computed is the expectation over $\mathbf{z}$. Section 1 says it this way.
- The source says sampling makes "$\partial z/\partial\mu$ not exist". Precisely, the output of a sampling node is not a function of $\mu$, so one cannot backpropagate along it; the gradient of the expectation with respect to $\mu$ does exist, and reparameterization is just another way of writing it so that it can be computed.
- The source's interactive demo of Lemma 1 says the KL at $\mu_1=2$, $\sigma_1=0.5$ is "$\approx2.443$". The formula gives $\log2+(0.25+4)/2-0.5=0.6931+2.125-0.5=2.318$; an estimate from $2$ million samples also gives $2.32$.
- The notes say the "ratio of estimation errors" at $\ell=1$, $100$, $10000$ is "$1:\sqrt{100}:\sqrt{10000}=1:10:100$". The error shrinks, so the ratio should be $1:\tfrac1{10}:\tfrac1{100}$; the notes' next sentence, "100 times more samples, 10 times less error", is the correct one.
- The source says Type 2 "trains more stably" with "smaller overall gradient variance". This usually holds but is not a theorem: in Section 2's example with $q=\mathcal{N}(0.5,\ 0.2)$, Type 1 has the smaller standard deviation.
- The quiz in §3.2.3 writes option A as $-\frac{1}{2}\sum_i(\log|\Sigma_{z\mid x}|+p+\text{tr}(\Sigma_{z\mid x})+\|\mu_{z\mid x}\|^2)$ with the note "(but the sign should be + for the log term)". As a contribution to the ELBO, the $\log$ and $p$ terms already have the right sign; the trace and $\lVert\mu\rVert^2$ terms are the wrong ones, so the note points at the one term that was already right. The correct KL is $\frac12[-\log\lvert\Sigma\rvert-p+\operatorname{tr}\Sigma+\lVert\boldsymbol{\mu}\rVert^2]$. This note leaves that question out.
- The source says "any reparameterizable distribution (Gaussian, Logistic, Gamma via location-scale)" works the same way. A gamma distribution is a scale family only when its shape parameter is fixed; learning the shape needs other methods (such as implicit reparameterization gradients).
- The source writes the practical loss as "MSE $+\ \beta\cdot$ KL" without mentioning the multiplier on the reconstruction error; this note adds the multiplier and its link to $\sigma_x$ and $\beta$.
- The source writes probabilities and vectors with $\mathbb{P}(\cdot)$ and bold italic `\boldsymbol`; this note uses $p(\cdot)$ and `\mathbf` (except for the Greek-letter vectors $\boldsymbol{\mu},\boldsymbol{\sigma},\boldsymbol{\epsilon}$). The quiz keeps the source's original notation.
- This note adds: the five-sample estimate and the linear-function coincidence, the comparison of the wobble of Type 1 and Type 2 and its reason, the link between Lemma 2 and Lemma 1, the gradient check with $\mathbb{E}[z^2]$, the link between the reconstruction multiplier and $\beta$, the small VAE's training record with the $\beta=0$ comparison, three figures, the analogy, and three practice questions with solutions.
- Choices: the numbers in the source's interactive demos of Type 1 and Type 2 (a true value of $0.8$ for Type 1, demo values of the $\beta$ trade-off) and the proof in the appendix are not included; the kinds of prior in prior regularization are only named.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 蒙地卡羅估計 | Monte Carlo estimate |
| 標準差 | standard deviation |
| 抽樣 | sampling |
| 公式解 | closed form |
| 引理 | lemma |
| 行列式 | determinant |
| 跡（對角線總和） | trace |
| 先驗正規化 | prior regularization |
| 重參數化技巧 | reparameterization trick |
| 計算圖 | computation graph |
| 小批次梯度下降 | mini-batch gradient descent |
| 學習率 | learning rate |
| KL 權重 | KL weight ($\beta$) |
| 數值微分 | numerical differentiation |
