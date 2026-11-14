# 第12週 KP05：VAE 的架構，編碼器輸出一個分布、從中抽樣、解碼器再畫回資料（The VAE architecture: the encoder outputs a distribution, a code is drawn from it, and the decoder draws the data back）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第12週的第五個知識點（KP05），接在 KP04 之後。KP04 的結尾提出一個想法：不要每筆資料都重新找一次近似後驗 $q$，而是訓練一個網路，看到 $\mathbf{x}$ 就直接吐出 $q$ 的參數。這一節把這個想法做成**變分自編碼器**（variational autoencoder，VAE），並說明它和 KP01 的自編碼器差在哪裡。

This is the fifth knowledge point (KP05) of Week 12, following KP04. KP04 ended with an idea: instead of searching for the approximate posterior $q$ anew for every example, train a network that looks at $\mathbf{x}$ and directly outputs the parameters of $q$. This note turns that idea into the **variational autoencoder** (VAE) and explains how it differs from KP01's autoencoder.

先備知識是 KP01 的自編碼器、KP02 的潛在變數與近似後驗、KP03 的 ELBO。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第12章 §3、§3.1 與教授祕笈改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec12-kp05-figures.py` 算出（它用到同資料夾的 `lec12_vae_toy.py`），可以自己重算。

The prerequisites are KP01's autoencoder, KP02's latent variables and approximate posterior, and KP03's ELBO. Every term and symbol is defined when it first appears. This note is rewritten from §3 and §3.1 of Chapter 12 of the course's web material CCC-LNS and its professor's notes, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec12-kp05-figures.py` (which uses `lec12_vae_toy.py` in the same folder), so you can recompute it yourself.

## KP05 VAE 的架構，編碼器輸出一個分布、從中抽樣、解碼器再畫回資料（The VAE architecture: the encoder outputs a distribution, a code is drawn from it, and the decoder draws the data back）

### 學習目標（Learning objectives）

讀完之後，你能畫出 VAE 的三個部分：輸出平均與對數變異數的編碼器、抽樣、解碼器，並寫出它的訓練目標。你能說明 VAE 的編碼器和 KP01 自編碼器的編碼器差在哪裡。你能用小數字走完一次前向計算：從 $\mathbf{x}$ 算出 $\boldsymbol{\mu}$、$\boldsymbol{\sigma}$、抽出 $\mathbf{z}$、解碼、算出重建項與 KL。你也能說出 VAE 怎麼生成新資料，以及在一個小例子裡它生成得有多好、哪裡還不夠好。

After reading, you can draw the three parts of a VAE, an encoder that outputs a mean and a log-variance, sampling, and a decoder, and write its training objective. You can explain how the VAE's encoder differs from that of KP01's autoencoder. You can carry out one forward pass with small numbers: from $\mathbf{x}$ compute $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$, draw $\mathbf{z}$, decode, and compute the reconstruction term and the KL. You can also say how a VAE generates new data, and, in a small example, how well it generates and where it still falls short.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{x}\in\mathbb{R}^d$ | 一筆資料 | one example |
| $\mathbf{z}\in\mathbb{R}^p$ | 潛在變數（編碼），長度 $p$ | latent variable (code), of length $p$ |
| $\theta_e,\ \theta_d$ | 編碼器與解碼器的權重 | weights of the encoder and the decoder |
| $q(\mathbf{z}\mid\mathbf{x})$ | 編碼器給出的近似後驗 | approximate posterior given by the encoder |
| $\boldsymbol{\mu}(\mathbf{x}),\ \log\boldsymbol{\sigma}^2(\mathbf{x})$ | 編碼器的兩組輸出：平均與對數變異數 | the encoder's two outputs: mean and log-variance |
| $\boldsymbol{\epsilon}$ | 從標準常態分布抽出的雜訊 | noise drawn from the standard normal |
| $p(\mathbf{z})=\mathcal{N}(\mathbf{0},I)$ | 先驗：標準常態分布 | prior: the standard normal |
| $p(\mathbf{x}\mid\mathbf{z})$ | 解碼器給出的概似 | likelihood given by the decoder |
| $\hat{\mathbf{x}}$ | 解碼器的輸出（重建） | the decoder's output (reconstruction) |
| $\operatorname{diag}(\cdot)$ | 對角矩陣：只有對角線有值 | diagonal matrix: values only on the diagonal |
| $I$ | 單位矩陣 | identity matrix |
| $\odot$ | 逐元素相乘 | element-wise multiplication |
| $\mathcal{L}$ | ELBO（KP03） | ELBO (KP03) |

### 1. 用一個網路給所有資料的 $q$：VAE 的三個部分（One network gives every example its $q$: the three parts of a VAE）

KP04 的變分 EM 每一輪都要對每筆資料單獨找一次最好的 $q$。VAE 的做法是讓一個神經網路學會這件事：輸入 $\mathbf{x}$，直接輸出 $q(\mathbf{z}\mid\mathbf{x})$ 的參數。這個網路學一次，所有資料都能用，這種作法叫**攤銷推論**（amortized inference）`[補充：名稱]`，就像一筆開銷分攤到所有資料上。

KP04's variational EM must search for the best $q$ separately for every example in every round. The VAE instead lets a neural network learn this job: it takes $\mathbf{x}$ and directly outputs the parameters of $q(\mathbf{z}\mid\mathbf{x})$. The network is learned once and serves every example; this is called **amortized inference** `[Added: name]`, like spreading one cost over all the data.

來源說 VAE 由三塊組成，每一塊都可以微分，所以整個網路能用反向傳播一起訓練（其中抽樣這一步，要用 KP06 的技巧才能微分，見第3節）：

The source says a VAE has three parts, each differentiable, so the whole network can be trained together by backpropagation (the sampling step becomes differentiable only with KP06's trick; see Section 3):

- **編碼器**（推論網路）：輸入 $\mathbf{x}$，輸出 $q(\mathbf{z}\mid\mathbf{x})$ 的參數，常用多元常態分布的平均與變異數（Encoder (inference network): takes $\mathbf{x}$ and outputs the parameters of $q(\mathbf{z}\mid\mathbf{x})$, usually the mean and variance of a multivariate normal）
- **抽樣**：從 $q(\mathbf{z}\mid\mathbf{x})$ 抽一個 $\mathbf{z}$；這份隨機性是 VAE 的關鍵（Sampling: draw a $\mathbf{z}$ from $q(\mathbf{z}\mid\mathbf{x})$; this randomness is the key to the VAE）
- **解碼器**（生成網路）：輸入 $\mathbf{z}$，輸出重建 $\hat{\mathbf{x}}$，或概似 $p(\mathbf{x}\mid\mathbf{z})$ 的參數（Decoder (generative network): takes $\mathbf{z}$ and outputs a reconstruction $\hat{\mathbf{x}}$, or the parameters of the likelihood $p(\mathbf{x}\mid\mathbf{z})$）

訓練目標就是 KP03 第4節的 ELBO；下式是一筆資料的，訓練時對所有資料加總：

The training objective is the ELBO of KP03 Section 4; the formula below is for one example, and training sums it over all examples:

$$
\mathcal{L}=\underbrace{\mathbb{E}_{q(\mathbf{z}\mid\mathbf{x})}\big[\log p(\mathbf{x}\mid\mathbf{z})\big]}_{\text{重建 reconstruction}}-\underbrace{D_{\text{KL}}\big(q(\mathbf{z}\mid\mathbf{x})\,\Vert\,p(\mathbf{z})\big)}_{\text{正規化 regularization}},\qquad p(\mathbf{z})=\mathcal{N}(\mathbf{0},I).
$$

![一張方塊流程圖。左邊的 x 進入紫色的編碼器，分成兩個紫色方塊 mu 與 log sigma 平方；兩者進入青色的抽樣方塊，寫著 z 從常態分布抽出；z 進入藍色的解碼器，輸出右邊的 x hat。上方寫著訓練時最大化 ELBO 等於重建項減 KL。下方一個黃色方塊是先驗 N(0, I)，橘色虛線箭頭直接指向解碼器，旁邊寫著生成時跳過編碼器（A block diagram. x on the left enters a purple encoder, which splits into two purple boxes, mu and log sigma squared; both enter a teal sampling box saying z is drawn from a normal; z enters a blue decoder, which outputs x hat on the right. Above: training maximises ELBO equals reconstruction minus KL. Below: a yellow prior box N(0, I) with a dashed orange arrow straight into the decoder, labelled generation skips the encoder）](../assets/figures/lec12-kp05-architecture.png)

**圖說 `lec12-kp05-architecture`**〔功能：一目了然〕：先看實線，由左到右是訓練時的路：$\mathbf{x}$ 經過編碼器，得到平均 $\boldsymbol{\mu}$ 與對數變異數 $\log\boldsymbol{\sigma}^2$；從這個常態分布抽一個 $\mathbf{z}$；解碼器把 $\mathbf{z}$ 畫回 $\hat{\mathbf{x}}$。再看上方的式子：兩個網路一起最大化 ELBO。最後看橘色虛線：生成時不需要編碼器，直接從先驗抽 $\mathbf{z}$ 交給解碼器。

**Caption `lec12-kp05-architecture`** [Function: seeing it at a glance]: first follow the solid arrows, left to right, the path during training: $\mathbf{x}$ passes through the encoder, giving a mean $\boldsymbol{\mu}$ and a log-variance $\log\boldsymbol{\sigma}^2$; a $\mathbf{z}$ is drawn from this normal; the decoder draws $\mathbf{z}$ back into $\hat{\mathbf{x}}$. Then look at the formula on top: the two networks maximise the ELBO together. Finally look at the dashed orange arrow: generation needs no encoder; it draws $\mathbf{z}$ from the prior and hands it to the decoder.

和 KP01 的自編碼器比，結構幾乎一樣，差在中間：自編碼器的編碼器輸出一個**確定的編碼**；VAE 的編碼器輸出編碼的**分布**，再從中抽樣。來源說這個差別是 VAE 能生成的關鍵；配上 KL 項，潛在空間會被整理得接近一個已知的分布（見第3節）。

Compared with KP01's autoencoder, the structure is almost the same; the difference is in the middle. An autoencoder's encoder outputs a **fixed code**; a VAE's encoder outputs a **distribution** over the code and then samples from it. The source says this difference is the key to the VAE's ability to generate; together with the KL term, the latent space is organised close to a known distribution (see Section 3).

常見錯誤：以為 VAE 就是更深的自編碼器。網路深淺不是差別；差別在編碼器輸出分布、中間有抽樣，以及訓練目標多了一項 KL。

Common mistake: thinking a VAE is just a deeper autoencoder. Depth is not the difference; the differences are that the encoder outputs a distribution, there is sampling in the middle, and the objective has an extra KL term.

### 2. 編碼器：輸出平均與對數變異數（The encoder: output a mean and a log-variance）

來源說，最常見的選擇是讓 $q(\mathbf{z}\mid\mathbf{x})$ 是**對角共變異數**的常態分布：

The source says the most common choice is to make $q(\mathbf{z}\mid\mathbf{x})$ a normal distribution with a **diagonal covariance**:

$$
q(\mathbf{z}\mid\mathbf{x})=\mathcal{N}\big(\mathbf{z};\ \boldsymbol{\mu}(\mathbf{x}),\ \operatorname{diag}(\boldsymbol{\sigma}^2(\mathbf{x}))\big).
$$

所以編碼器有兩組輸出，各 $p$ 個神經元：平均 $\boldsymbol{\mu}$ 與每一維的變異數。對角的意思是各維互不相關，只要記 $p$ 個變異數。KP02 第4節數過參數個數（連平均一起算）；這裡只數共變異數。若允許任意相關，共變異數矩陣要 $p(p+1)/2$ 個數：$p=4$ 時是 $4$ 對 $10$，$p=32$ 時是 $32$ 對 $528$，$p=100$ 時是 $100$ 對 $5050$。完整版還要做矩陣分解之類較貴的運算，所以實務上幾乎都用對角。

So the encoder has two groups of outputs, $p$ neurons each: the mean $\boldsymbol{\mu}$ and the variance of each dimension. Diagonal means the dimensions are uncorrelated, so only $p$ variances are needed. KP02 Section 4 counted parameters including the means; here we count only the covariance. Allowing any correlation requires $p(p+1)/2$ numbers for the covariance matrix: $4$ versus $10$ at $p=4$, $32$ versus $528$ at $p=32$, and $100$ versus $5050$ at $p=100$. The full version also needs costlier operations such as matrix factorisations, so in practice the diagonal is almost always used.

來源還提到一個實作細節：編碼器輸出的是 $\log\boldsymbol{\sigma}^2$，不是 $\boldsymbol{\sigma}^2$。變異數一定要是正數，而神經網路的輸出可以是任何實數。取指數 $\boldsymbol{\sigma}^2=e^{\log\boldsymbol{\sigma}^2}$ 一定是正的，梯度也比較穩定。

The source mentions an implementation detail: the encoder outputs $\log\boldsymbol{\sigma}^2$, not $\boldsymbol{\sigma}^2$. A variance must be positive, while a network's output can be any real number. Exponentiating, $\boldsymbol{\sigma}^2=e^{\log\boldsymbol{\sigma}^2}$, is always positive, and the gradients are more stable.

常見錯誤：以為潛在維度 $p$ 越大越好。祕笈提醒，$p$ 大時參數多、訓練難，也容易過擬合；$p$ 和 KP01 的瓶頸寬度一樣，是要調的超參數。

Common mistake: thinking the larger the latent dimension $p$, the better. The professor's notes point out that a large $p$ means more parameters, harder training and easier overfitting; like the bottleneck width in KP01, $p$ is a hyperparameter to tune.

### 3. 抽樣與解碼器，走一次前向計算（Sampling and the decoder: one forward pass）

有了 $\boldsymbol{\mu}$ 和 $\boldsymbol{\sigma}$，就從 $q(\mathbf{z}\mid\mathbf{x})$ 抽一個 $\mathbf{z}$。實作上的寫法是先抽一個標準常態的雜訊 $\boldsymbol{\epsilon}$，再算 $\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$，$\odot$ 是逐元素相乘。為什麼要這樣寫，是本週 KP06 的主題。

With $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$, draw a $\mathbf{z}$ from $q(\mathbf{z}\mid\mathbf{x})$. In practice you first draw standard-normal noise $\boldsymbol{\epsilon}$ and then compute $\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$, where $\odot$ is element-wise multiplication. Why it is written this way is the topic of this week's KP06.

解碼器也有兩種設計，來源都列了：一是直接輸出重建 $\hat{\mathbf{x}}$，損失用平方誤差 $\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2$，常用在連續資料；二是輸出分布的參數，例如黑白影像每個像素的伯努利機率，損失用 $-\log p(\mathbf{x}\mid\mathbf{z})$。祕笈也指出兩者其實是同一件事：若概似是平均 $\hat{\mathbf{x}}$、變異數固定的常態分布，最大化概似就等於最小化平方誤差；更精確地說 `[補充]`，$-\log p(\mathbf{x}\mid\mathbf{z})$ 是平方誤差乘上一個常數，再加一個常數。

The decoder also has two designs, both listed in the source: one outputs a reconstruction $\hat{\mathbf{x}}$ with the squared-error loss $\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2$, common for continuous data; the other outputs the parameters of a distribution, such as a Bernoulli probability for each pixel of a black-and-white image, with the loss $-\log p(\mathbf{x}\mid\mathbf{z})$. The professor's notes also point out that they are really the same thing: if the likelihood is a normal with mean $\hat{\mathbf{x}}$ and a fixed variance, maximising the likelihood is minimising the squared error; more precisely `[Added]`, $-\log p(\mathbf{x}\mid\mathbf{z})$ is the squared error times a constant, plus a constant.

走一次前向計算 `[補充]`。資料 $\mathbf{x}=[1,\ 2]$，$d=2$、$p=1$。假設編碼器算出 $\mu=0.5\times1+0.25\times2=1$，$\log\sigma^2=\log0.25=-1.3863$，所以 $\sigma=0.5$。抽到雜訊 $\epsilon=0.6$，得到

One forward pass `[Added]`. The data is $\mathbf{x}=[1,\ 2]$, with $d=2$ and $p=1$. Suppose the encoder computes $\mu=0.5\times1+0.25\times2=1$ and $\log\sigma^2=\log0.25=-1.3863$, so $\sigma=0.5$. The noise drawn is $\epsilon=0.6$, giving

$$
z=1+0.5\times0.6=1.3 .
$$

解碼器假設是 $\hat{\mathbf{x}}=[0.8,\ 1.6]\,z=[1.04,\ 2.08]$。平方誤差是 $0.04^2+0.08^2=0.008$。若概似是共變異數 $I$ 的二維常態分布，$-\log\mathcal{N}(\mathbf{x};\hat{\mathbf{x}},I)=\tfrac12\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2+\tfrac{d}{2}\log(2\pi)$，$d=2$ 時第二項是 $\log(2\pi)=1.8379$（本講義的 $\log$ 都是自然對數）。所以這一次抽樣對負重建項的估計是 $0.004+1.8379=1.8419$。正規化項用 KP03 的公式：$D_{\text{KL}}=\tfrac12(0.25+1-1-\log0.25)=0.8181$。這一次抽樣的負 ELBO 是 $1.8419+0.8181=2.66$，訓練就是要把它降下來。

Suppose the decoder is $\hat{\mathbf{x}}=[0.8,\ 1.6]\,z=[1.04,\ 2.08]$. The squared error is $0.04^2+0.08^2=0.008$. If the likelihood is a two-dimensional normal with covariance $I$, then $-\log\mathcal{N}(\mathbf{x};\hat{\mathbf{x}},I)=\tfrac12\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2+\tfrac{d}{2}\log(2\pi)$, and with $d=2$ the second term is $\log(2\pi)=1.8379$ (every $\log$ in this note is the natural log). So this one draw estimates the negative reconstruction term as $0.004+1.8379=1.8419$. The regularization term uses KP03's formula: $D_{\text{KL}}=\tfrac12(0.25+1-1-\log0.25)=0.8181$. The negative ELBO for this one draw is $1.8419+0.8181=2.66$, and training aims to bring it down.

來源指出兩項之間的拉扯：重建項希望每筆資料的編碼「專屬於自己」，彼此分得越開越好；KL 項希望每個 $q(\mathbf{z}\mid\mathbf{x})$ 都接近標準常態，擠在原點附近、寬度接近 $1$。VAE 在兩者之間找平衡，於是編碼彼此重疊、填滿先驗的範圍。

The source points out the tug of war between the two terms: the reconstruction term wants each example's code to be "its own", the further apart the better; the KL term wants every $q(\mathbf{z}\mid\mathbf{x})$ close to the standard normal, crowded near the origin with width near $1$. The VAE balances the two, so the codes overlap and fill the range of the prior.

常見錯誤：以為 KL 項只是防止過擬合的一種技巧。來源的小測驗特別考這一點：KL 項的主要作用是讓 $q(\mathbf{z}\mid\mathbf{x})$ 不偏離先驗，讓潛在空間有結構、可以從先驗抽樣生成。

Common mistake: thinking the KL term is just a trick against overfitting. The source's quiz targets exactly this: the KL term's main job is to keep $q(\mathbf{z}\mid\mathbf{x})$ from straying from the prior, giving the latent space structure so that we can generate by sampling from the prior.

### 4. 生成：從先驗抽 $\mathbf{z}$，交給解碼器（Generation: draw $\mathbf{z}$ from the prior and hand it to the decoder）

訓練完以後，生成新資料只要兩步：從 $\mathcal{N}(\mathbf{0},I)$ 抽一個 $\mathbf{z}$，送進解碼器。這正是 KP02 的生成故事，也是 KP01 的自編碼器做不到的事：VAE 事先規定了編碼該從哪裡挑。

After training, generating new data takes two steps: draw a $\mathbf{z}$ from $\mathcal{N}(\mathbf{0},I)$ and feed it to the decoder. This is exactly KP02's generative story, and exactly what KP01's autoencoder could not do: the VAE fixes in advance where codes should come from.

回到 KP01 的兩段圓弧 `[補充]`（單位圓上 $18^\circ$ 到 $108^\circ$ 與 $180^\circ$ 到 $270^\circ$ 兩段）：同樣 $300$ 個點、同樣 $k=p=1$，改訓練一個小 VAE。編碼器與解碼器各有一層 $32$ 個神經元、活化函數 $\tanh$，解碼器的概似是標準差 $0.1$ 的常態分布，用 Adam 訓練 $20000$ 步。第一段圓弧的編碼平均落在 $0.15$ 到 $2.2$，第二段落在 $-2.03$ 到 $-0.14$，每個點的標準差在 $0.045$ 到 $0.326$ 之間。兩段的編碼範圍合起來大致涵蓋標準常態的主要範圍，中間只剩很窄的縫。

Back to KP01's two arcs `[Added]` (the stretches from $18^\circ$ to $108^\circ$ and from $180^\circ$ to $270^\circ$ of the unit circle): the same $300$ points and the same $k=p=1$, but now we train a small VAE. The encoder and decoder each have one hidden layer of $32$ neurons with $\tanh$, the decoder's likelihood is a normal with standard deviation $0.1$, and Adam runs for $20000$ steps. The code means of the first arc lie from $0.15$ to $2.2$, those of the second from $-2.03$ to $-0.14$, and each point's standard deviation is between $0.045$ and $0.326$. Together the two ranges roughly span the main range of the standard normal, with only a narrow seam between them.

從先驗抽 $2000$ 個 $z$ 解碼：$96.9\%$ 落在圓上（離圓心的距離和 $1$ 差不到 $0.1$），$92.6\%$ 落在資料的兩段圓弧上（在圓上，而且角度在兩段圓弧之內）。比起 KP01 的自編碼器「不知道該從哪裡挑」，這已經能直接生成。剩下的幾個百分比，來自 $z=0$ 附近：那裡正是兩段編碼之間的縫，解碼出來的點沿一條直線穿過兩段圓弧之間的空地。

Draw $2000$ values of $z$ from the prior and decode them: $96.9\%$ land on the circle (distance from the centre within $0.1$ of $1$), and $92.6\%$ land on the data's two arcs (on the circle and with an angle inside one of the arcs). Compared with KP01's autoencoder, which "does not know where to pick from", this already generates directly. The remaining few percent come from near $z=0$: that is exactly the seam between the two groups of codes, and the decoded points run along a straight line across the empty ground between the arcs.

![左右兩張圖。左圖是編碼軸：上方每一條橫線是一個訓練點的平均加減兩個標準差，藍色是第一段圓弧、紫色是第二段，由左下排到右上；下方青色曲線是所有 q 的平均，和橘色虛線的標準常態分布大致重合，但在 z 等於 0 附近掉到幾乎為 0。右圖是輸入平面：淺藍點是訓練資料，橘色點是從先驗抽 2000 個 z 解碼出來的點，幾乎都貼著兩段圓弧，另有一小段直線連接兩段圓弧的端點，標題寫 96.9% 落在圓上（Two panels. Left, the code axis: each horizontal bar above is one training point's mean plus or minus two standard deviations, blue for the first arc and purple for the second, arranged from lower left to upper right; below, the teal average of all q roughly matches the dashed orange standard normal but drops to nearly 0 near z equals 0. Right, the input plane: light-blue points are the training data and orange points are 2000 decoded draws from the prior, almost all hugging the two arcs, plus a short straight segment joining the ends of the two arcs; the title says 96.9% land on the circle）](../assets/figures/lec12-kp05-latent-coverage.png)

**圖說 `lec12-kp05-latent-coverage`**〔功能：一目了然、精準度、空間幾何具象化〕：先看左圖上方：每條橫線是一個訓練點的 $q(z\mid\mathbf{x})$（每 $6$ 個點畫一個；平均加減兩個標準差，縱向位置只是為了排開），它們互相重疊。再看左圖下方：所有 $q$ 的平均（青線）和先驗（橘色虛線）大致貼合，但在 $z=0$ 附近有一道縫，縫的兩側各有一個比先驗高的尖峰。最後看右圖：從先驗抽的 $z$ 解碼後幾乎都落在圓弧上；那道縫對應圖上連接兩段圓弧的一小段直線。

**Caption `lec12-kp05-latent-coverage`** [Function: seeing it at a glance, precision, making space concrete]: first look at the top of the left panel: each bar is one training point's $q(z\mid\mathbf{x})$ (every $6$th point is drawn; mean plus or minus two standard deviations; the vertical position only spreads them out), and they overlap. Then look at the bottom of the left panel: the average of all $q$ (teal) roughly matches the prior (dashed orange), but has a seam near $z=0$ with a peak higher than the prior on each side of it. Finally look at the right panel: draws from the prior decode almost entirely onto the arcs; the seam corresponds to the short straight segment joining the two arcs.

這個例子也說明 VAE 的限制：KL 項把編碼拉向先驗，但所有 $q$ 的平均不一定和先驗完全一樣。先驗有機率、編碼卻很少的地方，生成出來就不像資料。本週 KP07 會談 VAE 的其他限制。

This example also shows a limitation of the VAE: the KL term pulls the codes towards the prior, but the average of all $q$ need not match the prior exactly. Wherever the prior has probability but codes are scarce, the generated points do not look like data. This week's KP07 discusses the VAE's other limitations.

常見錯誤：以為生成時要先把一筆真實資料送進編碼器。來源的小測驗也考這一點：生成時只用解碼器，從先驗 $\mathcal{N}(\mathbf{0},I)$ 抽 $\mathbf{z}$；送真實資料進編碼器、再解碼，叫做重建。

Common mistake: thinking generation starts by feeding a real example into the encoder. The source's quiz tests this too: generation uses only the decoder, with $\mathbf{z}$ drawn from the prior $\mathcal{N}(\mathbf{0},I)$; feeding a real example to the encoder and then decoding is called reconstruction.

### 本單元重點整理（KP05 summary）

**一句話重點**：VAE 用一個編碼器網路直接給每筆資料的近似後驗 $q(\mathbf{z}\mid\mathbf{x})=\mathcal{N}(\boldsymbol{\mu},\operatorname{diag}\boldsymbol{\sigma}^2)$，從中抽 $\mathbf{z}$，再用解碼器給出概似 $p(\mathbf{x}\mid\mathbf{z})$；兩個網路一起最大化「重建項減去離標準常態的 KL」；生成時跳過編碼器，從先驗抽 $\mathbf{z}$ 交給解碼器。

**One-sentence summary**: a VAE uses an encoder network to give each example its approximate posterior $q(\mathbf{z}\mid\mathbf{x})=\mathcal{N}(\boldsymbol{\mu},\operatorname{diag}\boldsymbol{\sigma}^2)$, draws $\mathbf{z}$ from it, and uses a decoder to give the likelihood $p(\mathbf{x}\mid\mathbf{z})$; the two networks together maximise "the reconstruction term minus the KL from the standard normal"; to generate, it skips the encoder, draws $\mathbf{z}$ from the prior and hands it to the decoder.

**白話說明** `[補充]`：KP01 的編碼器替每張照片寫一張精確的小紙條；VAE 的編碼器只寫「大概在這附近、誤差多少」，而且規定所有紙條都要寫在同一本標準筆記本的範圍裡。生成時，隨便翻到筆記本的一頁，解碼器都看得懂。

**Plain explanation** `[Added]`: KP01's encoder writes an exact slip for each photo; the VAE's encoder writes only "roughly here, give or take this much", and requires every slip to fall within the pages of one standard notebook. To generate, open the notebook at any page, and the decoder can read it.

**正式定義**：$q(\mathbf{z}\mid\mathbf{x})=\mathcal{N}(\boldsymbol{\mu}(\mathbf{x};\theta_e),\operatorname{diag}\boldsymbol{\sigma}^2(\mathbf{x};\theta_e))$，$\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$，$\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},I)$；目標 $\mathbb{E}_q[\log p(\mathbf{x}\mid\mathbf{z};\theta_d)]-D_{\text{KL}}(q(\mathbf{z}\mid\mathbf{x})\,\Vert\,\mathcal{N}(\mathbf{0},I))$。

**Formal definition**: $q(\mathbf{z}\mid\mathbf{x})=\mathcal{N}(\boldsymbol{\mu}(\mathbf{x};\theta_e),\operatorname{diag}\boldsymbol{\sigma}^2(\mathbf{x};\theta_e))$, $\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$, $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},I)$; objective $\mathbb{E}_q[\log p(\mathbf{x}\mid\mathbf{z};\theta_d)]-D_{\text{KL}}(q(\mathbf{z}\mid\mathbf{x})\,\Vert\,\mathcal{N}(\mathbf{0},I))$.

**例子**：$\mathbf{x}=[1,2]$、$\mu=1$、$\sigma=0.5$、$\epsilon=0.6$ 得 $z=1.3$、$\hat{\mathbf{x}}=[1.04,2.08]$，負重建項 $1.8419$、KL $0.8181$、負 ELBO $2.66$；兩段圓弧的小 VAE 從先驗生成，$96.9\%$ 落在圓上、$92.6\%$ 落在資料的圓弧上。

**Example**: $\mathbf{x}=[1,2]$, $\mu=1$, $\sigma=0.5$, $\epsilon=0.6$ give $z=1.3$ and $\hat{\mathbf{x}}=[1.04,2.08]$, with negative reconstruction term $1.8419$, KL $0.8181$ and negative ELBO $2.66$; the small two-arc VAE generating from the prior puts $96.9\%$ of points on the circle and $92.6\%$ on the data's arcs.

**常見迷思**：

- 「VAE 就是更深的自編碼器。」錯，差在編碼器輸出分布、中間抽樣、目標多一項 KL。
- 「KL 項只是防過擬合。」錯，它讓潛在空間貼近先驗，才能抽樣生成。
- 「生成時要先用編碼器。」錯，生成只用解碼器。
- 「潛在維度越大越好。」錯，它是要調的超參數。

**Common misconceptions**:

- "A VAE is a deeper autoencoder." Wrong: the encoder outputs a distribution, there is sampling, and the objective has an extra KL term.
- "The KL term only prevents overfitting." Wrong: it keeps the latent space close to the prior so that sampling can generate.
- "Generation uses the encoder first." Wrong: generation uses only the decoder.
- "The larger the latent dimension, the better." Wrong: it is a hyperparameter to tune.

**圖**：本單元有兩張圖。

- `lec12-kp05-architecture`：編碼器、抽樣、解碼器三部分，以及生成時的路徑。
- `lec12-kp05-latent-coverage`：小 VAE 的編碼互相重疊、大致填滿先驗，從先驗生成幾乎都落在資料上。

**Figures** — this unit has two figures.

- `lec12-kp05-architecture`: the encoder, sampling and decoder, and the path taken during generation.
- `lec12-kp05-latent-coverage`: the small VAE's codes overlap and roughly fill the prior, and generation from the prior lands almost entirely on the data.

### 小練習（Practice）

1. 沿用第3節的例子，但抽到 $\epsilon=-1$。算 $z$、$\hat{\mathbf{x}}$ 與平方誤差。
2. 編碼器的對數變異數輸出是 $\log\sigma^2=0$，平均是 $\mu=0$。這時 KL 項是多少？這代表什麼？
3. 潛在維度 $p=16$。對角與完整共變異數各要幾個數（不含平均）？

1. Keep the example of Section 3 but draw $\epsilon=-1$. Compute $z$, $\hat{\mathbf{x}}$ and the squared error.
2. The encoder outputs log-variance $\log\sigma^2=0$ and mean $\mu=0$. What is the KL term, and what does it mean?
3. The latent dimension is $p=16$. How many numbers do a diagonal and a full covariance need (not counting the means)?

**第 1 題解答** `[補充]`：$z=1+0.5\times(-1)=0.5$，$\hat{\mathbf{x}}=[0.4,\ 0.8]$，平方誤差 $0.6^2+1.2^2=1.8$。

**Solution to question 1** `[Added]`: $z=1+0.5\times(-1)=0.5$, $\hat{\mathbf{x}}=[0.4,\ 0.8]$, squared error $0.6^2+1.2^2=1.8$.

**第 2 題解答** `[補充]`：$\sigma^2=1$，KL $=\tfrac12(1+0-1-0)=0$。這時 $q$ 就是先驗本身；若每筆資料都這樣，編碼完全不帶資料的資訊，重建項會很差。

**Solution to question 2** `[Added]`: $\sigma^2=1$, so the KL is $\tfrac12(1+0-1-0)=0$. Then $q$ is the prior itself; if this happened for every example, the codes would carry no information about the data and the reconstruction term would be poor.

**第 3 題解答** `[補充]`：對角 $16$ 個；完整 $16\times17/2=136$ 個。

**Solution to question 3** `[Added]`: diagonal $16$; full $16\times17/2=136$.

### 小測驗（Quiz，來源原題）

來源 §3–§3.1.3 有五題自我檢測，這裡收四題，保留原題與全部選項；§3.1.2 的重參數化題收在 KP06。來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The source's §3–§3.1.3 has five self-check questions; four are kept here with all their options, and the reparameterization question of §3.1.2 is in KP06. the source page does not mark the answers, so the explanations below are judged from the source text.

1. VAE 與標準 autoencoder 的關鍵差別？
   - (A) VAE 用更深的網路
   - (B) VAE 的 encoder 輸出 latent 分佈參數（隨機性），AE 輸出確定編碼
   - (C) VAE 不用 backprop
   - (D) VAE 不需要 decoder
2. VAE 訓練時，KL 項的作用是？
   - (A) 讓重建更精確
   - (B) 讓 $q(z\mid x)$ 不要偏離 prior $p(z)$，讓 latent space 有結構
   - (C) 增加模型容量
   - (D) 避免過擬合的唯一方法
3. VAE encoder 為什麼常輸出對角共變異數？
   - (A) 完全沒理由
   - (B) 計算便宜且仍能表達多維分佈
   - (C) 對角才能用 backprop
   - (D) 非對角會爆炸
4. VAE decoder 訓練完成後的「生成」流程是？
   - (A) 用真實 $\mathbf{x}_i$ encoder 後取 mean
   - (B) 從 prior $p(z)=\mathcal{N}(0,I)$ 抽 $z$，過 decoder 輸出新 $\hat{\mathbf{x}}$
   - (C) 隨機初始化 decoder weights
   - (D) 用 $K$-means 聚類 $z$

1. What is the key difference between a VAE and a standard autoencoder?
   - (A) A VAE uses a deeper network
   - (B) A VAE's encoder outputs the parameters of a latent distribution (randomness), while an AE outputs a fixed code
   - (C) A VAE does not use backprop
   - (D) A VAE needs no decoder
2. During VAE training, what does the KL term do?
   - (A) Makes reconstruction more accurate
   - (B) Keeps $q(z\mid x)$ from straying from the prior $p(z)$, giving the latent space structure
   - (C) Increases model capacity
   - (D) Is the only way to avoid overfitting
3. Why does a VAE encoder usually output a diagonal covariance?
   - (A) No reason at all
   - (B) It is cheap to compute and can still express multidimensional distributions
   - (C) Only diagonal allows backpropagation
   - (D) Off-diagonal would blow up
4. After the VAE decoder is trained, what is the "generation" procedure?
   - (A) Feed a real $\mathbf{x}_i$ to the encoder and take the mean
   - (B) Draw $z$ from the prior $p(z)=\mathcal{N}(0,I)$ and pass it through the decoder to output a new $\hat{\mathbf{x}}$
   - (C) Randomly initialise the decoder weights
   - (D) Cluster $z$ with $K$-means

**解析** `[補充]`：第 1 題選 (B)：第1節。第 2 題選 (B)：第3節；KL 項其實會讓重建變差一點，所以 (A) 錯。第 3 題選 (B)：第2節；完整共變異數一樣能反向傳播，只是比較貴。第 4 題選 (B)：第4節；(A) 是重建。

**Explanations** `[Added]`: question 1 is B: see Section 1. Question 2 is B: see Section 3; the KL term actually makes reconstruction a little worse, so A is wrong. Question 3 is B: see Section 2; a full covariance can also be backpropagated, it is just costlier. Question 4 is B: see Section 4; A is reconstruction.

### 留給下一單元的問題（The question for the next unit）

第3節留了一個伏筆：$\mathbf{z}$ 是「抽」出來的。反向傳播要算梯度，可是「從一個分布抽樣」這一步要怎麼對 $\boldsymbol{\mu}$ 和 $\boldsymbol{\sigma}$ 微分？ELBO 裡的期望值又要怎麼算？下一個知識點（本週 KP06）介紹 VAE 的訓練：KL 的公式解、抽樣估計，以及讓抽樣可以微分的重參數化技巧。

Section 3 left a loose end: $\mathbf{z}$ is "drawn". Backpropagation needs gradients, but how do you differentiate the step "sample from a distribution" with respect to $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$? And how is the expectation in the ELBO computed? The next knowledge point (this week's KP06) covers training the VAE: the closed form of the KL, sampling estimates, and the reparameterization trick that makes sampling differentiable.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第12章「變分模型」§3、§3.1–§3.1.3 與教授祕笈「VAE：用神經網路做變分推論」「VAE 的三部件」「Encoder」「Latent 採樣」「Decoder」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch12-variational-models/index.html ，讀取日期 2026-11-14）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §3 and §3.1–§3.1.3 of Chapter 12, "Variational models", of CCC-LNS IDL_115, with the professor's notes "VAE: variational inference with neural networks", "The three parts of a VAE", "Encoder", "Latent sampling" and "Decoder" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch12-variational-models/index.html , accessed 2026-11-14). Content not in the source is marked `[Added]` where it appears.

- 來源的互動示範說 $p=4$ 時「對角需 4 個參數，完整需 16 個」，正文也寫完整要「$p\times p$ 個參數」。共變異數矩陣是對稱的，只要 $p(p+1)/2$ 個：$p=4$ 時是 $10$ 個。來源 Encoder 一節的祕笈其實已寫對 $p(p+1)/2$，前後不一致；本講義在第2節用正確數字。
- 祕笈寫 $d=100$ 時對角與完整「200 vs 5050」：$200$ 含平均（$100+100$），$5050$ 卻不含平均。同樣含平均應是 $200$ 對 $5150$，不含平均是 $100$ 對 $5050$。本講義一律不含平均。
- 來源說抽樣的隨機性「讓 latent space 是連續分佈而非離散編碼」；KP01 已說明一般自編碼器的編碼也是連續的。本講義改說：隨機性加上 KL 項，讓編碼互相重疊、貼近一個已知的分布。
- 來源用 $\mathbb{P}(\cdot)$ 與粗斜體 `\boldsymbol` 表示機率與向量；本講義改用 $p(\cdot)$ 與 `\mathbf`（希臘字母向量 $\boldsymbol{\mu},\boldsymbol{\sigma}$ 例外）。小測驗保留來源原本的記號。
- 來源說 VAE 的三個部分「都可微」，但它自己 §3.1.2 也說抽樣會「切斷梯度」；本講義在第1節註明抽樣要靠 KP06 的技巧。
- 本講義補上：攤銷推論的名稱、負對數概似的常數項、一次前向計算的數字例子、兩段圓弧的小 VAE 與兩張圖、筆記本比喻、三題小練習與解答。
- 取捨：反向傳播穿過抽樣的問題與重參數化技巧留給 KP06；正規化流、混合常態等較複雜的 $q$ 只在 KP07 帶過。

- The source's interactive demo says that at $p=4$ "the diagonal needs 4 parameters and the full needs 16", and its main text says the full needs "$p\times p$ parameters". The covariance matrix is symmetric, so it needs only $p(p+1)/2$: $10$ at $p=4$. The professor's notes on the Encoder already give $p(p+1)/2$, so the source contradicts itself; Section 2 uses the correct numbers.
- The notes write "200 vs 5050" for diagonal versus full at $d=100$: the $200$ includes the means ($100+100$), but the $5050$ does not. Including the means consistently gives $200$ versus $5150$; excluding them gives $100$ versus $5050$. This note always excludes the means.
- The source says sampling's randomness "makes the latent space a continuous distribution rather than discrete codes"; KP01 explained that an ordinary autoencoder's codes are continuous too. This note says instead that randomness plus the KL term make the codes overlap and stay close to a known distribution.
- The source writes probabilities and vectors with $\mathbb{P}(\cdot)$ and bold italic `\boldsymbol`; this note uses $p(\cdot)$ and `\mathbf` (except for the Greek-letter vectors $\boldsymbol{\mu},\boldsymbol{\sigma}$). The quiz keeps the source's original notation.
- The source says the three parts of a VAE are "all differentiable", yet its own §3.1.2 says sampling "cuts the gradient"; Section 1 notes that sampling needs KP06's trick.
- This note adds: the name amortized inference, the constant term of the negative log-likelihood, the numerical forward pass, the small two-arc VAE with both figures, the notebook analogy, and three practice questions with solutions.
- Choices: backpropagation through sampling and the reparameterization trick are left to KP06; more complex $q$ such as normalizing flows and normal mixtures are only mentioned in KP07.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 變分自編碼器 | variational autoencoder (VAE) |
| 攤銷推論 | amortized inference |
| 推論網路（編碼器） | inference network (encoder) |
| 生成網路（解碼器） | generative network (decoder) |
| 對角共變異數 | diagonal covariance |
| 對數變異數 | log-variance |
| 標準常態分布 | standard normal distribution |
| 逐元素相乘 | element-wise multiplication |
| 伯努利分布 | Bernoulli distribution |
| 抽樣 | sampling |
| 重建 | reconstruction |
| 生成 | generation |
| 重參數化技巧（KP06） | reparameterization trick |
| $K$ 平均法 | $K$-means |
