# 第14週 KP03：逆向過程與 ELBO（The backward process and the ELBO）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第14週「深度生成模型（三）：擴散模型」的第三個知識點（KP03）。KP02 把前向過程完全寫了出來。生成要走反方向：從雜訊 $\mathbf{x}_T$ 一步一步回到資料 $\mathbf{x}_0$。這一節先把逆向的每一步寫成一個要學的常態分布，再像第12週的 VAE 一樣，推出訓練它的目標 ELBO，並把 ELBO 拆成每一步各自的 KL 散度。

This is the third knowledge point (KP03) of Week 14, "Deep generative models (3): diffusion models". KP02 wrote the forward process out completely. Generation goes the other way: from the noise $\mathbf{x}_T$ step by step back to the data $\mathbf{x}_0$. This note first writes each backward step as a normal distribution to be learned, then, as for the VAE of Week 12, derives the training objective, the ELBO, and splits it into one KL divergence per step.

先備知識是本週 KP01、KP02 的前向過程與閉式 $q(\mathbf{x}_t\mid\mathbf{x}_0)$，以及第12週的 KL 散度（KP02）與 ELBO（KP03）。還會用到兩條機率規則：**貝氏定理**（Bayes' rule）$q(a\mid b)=q(b\mid a)\,q(a)/q(b)$，以及「對數把乘法變加法」：$\log(ab)=\log a+\log b$。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第15章 KP-3（2.2 節）與 KP-4（2.3 節）改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec14-kp03-figures.py` 算出，可以自己重算。

The prerequisites are the forward process and the closed form $q(\mathbf{x}_t\mid\mathbf{x}_0)$ of this week's KP01 and KP02, and the KL divergence (KP02) and the ELBO (KP03) of Week 12. Two probability rules are also used: **Bayes' rule** $q(a\mid b)=q(b\mid a)\,q(a)/q(b)$, and "the logarithm turns products into sums": $\log(ab)=\log a+\log b$. Every term and symbol is defined when it first appears. This note is rewritten from KP-3 (Section 2.2) and KP-4 (Section 2.3) of Chapter 15 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec14-kp03-figures.py`, so you can recompute it yourself.

## KP03 逆向過程與 ELBO（The backward process and the ELBO）

### 學習目標（Learning objectives）

讀完之後，你能寫出逆向一步 $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ 的常態形式，說出網路要輸出什麼、變異數為什麼常設成 $\sigma_t^2 I$。你能說明為什麼「每一步很小」時，逆向一步才適合假設成常態。你能寫出整條生成鏈的聯合分布，說明為什麼不能直接最大化 $\log p_\theta(\mathbf{x}_0)$，並寫出擴散模型的 ELBO。你能說出 ELBO 拆開後的三類項各是什麼、哪一類跟參數無關、訓練主要在縮小哪一類。你也能從一個三步的小例子讀出：ELBO 一定不大於 $\log p_\theta(\mathbf{x}_0)$，而且就算模型完美，每一步的 KL 也不會是 $0$。

After reading, you can write the normal form of one backward step $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$, say what the network outputs, and why the variance is often set to $\sigma_t^2 I$. You can explain why a backward step can be assumed normal only when each step is small. You can write the joint distribution of the whole generation chain, explain why $\log p_\theta(\mathbf{x}_0)$ cannot be maximised directly, and write the diffusion model's ELBO. You can name the three kinds of terms in the split ELBO, which kind does not depend on the parameters, and which kind training mainly shrinks. From a three-step example you can also read that the ELBO is never larger than $\log p_\theta(\mathbf{x}_0)$, and that even a perfect model does not make each step's KL equal to $0$.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{x}_0,\ \mathbf{x}_t,\ \mathbf{x}_T$ | 資料、第 $t$ 步、最後一步（同 KP02） | the data, step $t$, the last step (as in KP02) |
| $\alpha_t,\ \bar\alpha_t$ | $\alpha_t=1-\beta_t$，$\bar\alpha_t=\alpha_1\cdots\alpha_t$（KP02） | $\alpha_t=1-\beta_t$, $\bar\alpha_t=\alpha_1\cdots\alpha_t$ (KP02) |
| $\mathbf{x}_{0:T}$ | $\mathbf{x}_0,\mathbf{x}_1,\ldots,\mathbf{x}_T$ 整串 | the whole chain $\mathbf{x}_0,\mathbf{x}_1,\ldots,\mathbf{x}_T$ |
| $q(\cdot)$ | 前向過程的分布，固定 | distributions of the forward process, fixed |
| $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ | 逆向一步，由參數 $\theta$ 的網路決定 | one backward step, set by a network with parameters $\theta$ |
| $\mu_\theta(\mathbf{x}_t,t)$ | 網路輸出的逆向一步的平均 | the mean of the backward step output by the network |
| $\sigma_t^2$ | 逆向第 $t$ 步的變異數，每個維度相同 | the variance of backward step $t$, the same in every dimension |
| $p(\mathbf{x}_T)$ | 生成的起點分布，$\mathcal{N}(\mathbf{0},I)$ | the starting distribution of generation, $\mathcal{N}(\mathbf{0},I)$ |
| $\mathrm{KL}(q\,\Vert\,p)$ | KL 散度：$q$ 和 $p$ 差多遠，$\ge0$（第12週 KP02） | KL divergence: how far $q$ is from $p$, $\ge0$ (Week 12 KP02) |
| $\mathbb{E}_q[\cdot]$ | 依 $q$ 抽樣後取平均 | the average when sampling from $q$ |
| $\mathcal{L}$ | 擴散模型的 ELBO；用自然對數時單位叫奈特（nats） | the diffusion model's ELBO; with natural logarithms the unit is the nat |
| $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$ | 已知原資料時真正的去噪一步（公式在 KP04） | the true denoising step when the original data are known (formula in KP04) |

### 1. 逆向過程：一條要學的常態鏈（The backward process: a learned chain of normals）

逆向過程和前向方向相反（來源的式 (6)）：

The backward process runs opposite to the forward one (the source's equation (6)):

$$
\mathbf{x}_T\rightarrow\mathbf{x}_{T-1}\rightarrow\cdots\rightarrow\mathbf{x}_0 .
$$

我們不知道真正的逆向一步 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ 長什麼樣子，所以用一個神經網路來近似它。來源的式 (7) 把每一步設成常態分布，平均與共變異數都由網路決定：

We do not know what the true backward step $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ looks like, so we approximate it with a neural network. The source's equation (7) makes each step a normal distribution whose mean and covariance are set by the network:

$$
p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)=\mathcal{N}\!\left(\mathbf{x}_{t-1};\ \mu_\theta(\mathbf{x}_t,t),\ \Sigma_\theta(\mathbf{x}_t,t)\right).
$$

網路的輸入是目前的 $\mathbf{x}_t$ 和步數 $t$，輸出是「下一步該往哪裡」的平均 $\mu_\theta$。`[補充]` 所有步數共用**同一個**網路，靠輸入 $t$ 來分辨現在雜訊多大；不是每一步各訓練一個網路。

The network's inputs are the current $\mathbf{x}_t$ and the step number $t$, and its output is the mean $\mu_\theta$, "where to go next". `[Added]` All steps share **one** network, which tells how noisy things are from the input $t$; one network is not trained per step.

共變異數通常簡化成每個維度相同（來源的式 (8)），叫做**等向**（isotropic）共變異數：

The covariance is usually simplified to be the same in every dimension (the source's equation (8)), called an **isotropic** covariance:

$$
\Sigma_\theta(\mathbf{x}_t,t)=\sigma_t^2 I .
$$

DDPM 直接把 $\sigma_t^2$ 固定成一個跟 $\beta_t$ 有關的數，不讓網路學（本週 KP05）；讓網路也學變異數是 KP06 的改良。生成從標準常態出發（來源的式 (11)），整條生成鏈的聯合分布是起點乘上每一步（式 (10)）：

DDPM simply fixes $\sigma_t^2$ to a number related to $\beta_t$ and does not let the network learn it (this week's KP05); letting the network learn the variance too is an improvement in KP06. Generation starts from a standard normal (the source's equation (11)), and the joint distribution of the whole generation chain is the start times every step (equation (10)):

$$
p(\mathbf{x}_T)=\mathcal{N}(\mathbf{0},I),\qquad p_\theta(\mathbf{x}_{0:T})=p(\mathbf{x}_T)\prod_{t=1}^{T}p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t).
$$

![一條橫向的方塊鏈：淺藍色 x0、x1（藍紫色）、省略號、淡紫色 x T-1、黃色 xT。每兩個相鄰方塊之間，上方有一條往右的藍色彎箭頭，下方有一條往左的紫色彎箭頭。最上方的藍字寫：前向（固定），q(x_t | x_{t-1}) 等於平均根號 alpha_t 乘 x_{t-1}、共變異數 beta_t I 的常態；最下方的紫字寫：逆向（要學），p theta(x_{t-1} | x_t) 等於平均 mu theta(x_t, t)、共變異數 sigma_t 平方 I 的常態，起點 p(xT) 是標準常態（A horizontal chain of boxes: light-blue x0, lavender-blue x1, an ellipsis, light-purple x T-1 and yellow xT. Between each pair of neighbouring boxes there is a blue curved arrow pointing right above and a purple curved arrow pointing left below. The blue text at the top reads: forward (fixed), q(x_t | x_{t-1}) is normal with mean sqrt(alpha_t) x_{t-1} and covariance beta_t I; the purple text at the bottom reads: backward (learned), p theta(x_{t-1} | x_t) is normal with mean mu theta(x_t, t) and covariance sigma_t squared I, starting from p(xT) = standard normal)](../assets/figures/lec14-kp03-two-chains.png)

**圖說 `lec14-kp03-two-chains`**〔功能：一目了然〕：同一串 $\mathbf{x}_0,\ldots,\mathbf{x}_T$ 有兩個方向。上方藍色箭頭往右，是 KP02 寫好的固定前向過程；下方紫色箭頭往左，是本節要學的逆向過程。兩邊每一步都是常態分布，差別在於藍色的平均與變異數是公式，紫色的平均要由網路算出來。

**Caption `lec14-kp03-two-chains`** [Function: seeing it at a glance]: the same chain $\mathbf{x}_0,\ldots,\mathbf{x}_T$ has two directions. The blue arrows above point right: the fixed forward process written in KP02. The purple arrows below point left: the backward process to be learned in this note. Every step on both sides is a normal distribution; the difference is that the blue means and variances are formulas, while the purple means must be computed by the network.

常見錯誤：以為逆向過程每一步各有一個網路，或者網路一次直接輸出 $\mathbf{x}_0$。DDPM 用同一個網路，每次只往前走一小步，走 $T$ 次。

Common mistake: thinking each backward step has its own network, or that the network outputs $\mathbf{x}_0$ in one go. DDPM uses a single network that moves only one small step each time, $T$ times.

### 2. 為什麼可以假設逆向一步是常態（Why a backward step can be assumed normal）

前向一步是常態，是我們規定的；逆向一步也是常態，卻不是理所當然 `[補充]`。真正的逆向一步 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ 要用貝氏定理，從前向一步和 $\mathbf{x}_{t-1}$ 的整體分布算出來；資料的分布複雜，它也可能很複雜。關鍵在於步伐大小。下圖用 KP01 的兩堆資料（$0.5\,\mathcal{N}(-2,0.3^2)+0.5\,\mathcal{N}(2,0.3^2)$）走**一步**，看到 $x_1$ 之後，把真正的 $q(x_0\mid x_1)$ 算出來。

The forward step is normal because we said so; the backward step being normal too is not automatic `[Added]`. The true backward step $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ must be computed with Bayes' rule from the forward step and the overall distribution of $\mathbf{x}_{t-1}$; since the data distribution is complicated, it can be complicated too. The key is the step size. The figure below takes KP01's two piles of data ($0.5\,\mathcal{N}(-2,0.3^2)+0.5\,\mathcal{N}(2,0.3^2)$) through **one** step and, after seeing $x_1$, computes the true $q(x_0\mid x_1)$.

![兩張並排的曲線圖，橫軸 x0 從 -3.5 到 3.5。紫色實線是真正的 q(x0 | x1)，橘色虛線是最接近它的單一常態。左圖：小步，beta 等於 0.01，看到 x1 等於 1.5；紫線是在 1.56 附近一座又高又窄的山，高約 4.2，橘色虛線和它完全重合。右圖：大步，beta 等於 0.5，看到 x1 等於 0；紫線有兩座分開的山，各在約 -1.84 與 1.84，高約 0.7，中間幾乎是 0；橘色虛線是一座中心在 0、又矮又寬的山，高約 0.21，和紫線差很多（Two curve charts side by side, with x0 from -3.5 to 3.5. The solid purple line is the true q(x0 | x1), and the dashed orange line is the closest single normal. Left: small step, beta = 0.01, observed x1 = 1.5; the purple line is a tall, narrow peak near 1.56, about 4.2 high, and the dashed orange line coincides with it. Right: large step, beta = 0.5, observed x1 = 0; the purple line has two separate peaks at about -1.84 and 1.84, about 0.7 high, with almost nothing between them; the dashed orange line is a low, wide hill centred at 0, about 0.21 high, far from the purple line)](../assets/figures/lec14-kp03-small-vs-large-step.png)

**圖說 `lec14-kp03-small-vs-large-step`**〔功能：對照〕：左右兩張看的是同一種問題：「看到 $x_1$ 之後，$x_0$ 可能在哪裡」。左圖步伐小，答案只是一座窄山（標準差 $0.095$，差不多就是 $\sqrt{0.01}=0.1$），單一常態完全描述得了。右圖步伐大，$x_1=0$ 可能來自左堆也可能來自右堆，答案變成兩座山；和它平均、變異數都相同的單一常態（也是 KL 散度最小的那一個）只能把中心放在 $0$、拉得很寬（標準差 $1.86$），正好落在幾乎不可能的地方。

**Caption `lec14-kp03-small-vs-large-step`** [Function: comparison]: both charts ask the same question: "having seen $x_1$, where might $x_0$ be?". On the left the step is small, and the answer is just one narrow peak (standard deviation $0.095$, about $\sqrt{0.01}=0.1$), which a single normal describes perfectly. On the right the step is large: $x_1=0$ could have come from either pile, so the answer has two peaks; the single normal with the same mean and variance (which is also the one with the smallest KL divergence) can only centre at $0$ and spread wide (standard deviation $1.86$), landing exactly where $x_0$ is almost impossible.

所以擴散模型要走很多小步（DDPM 用 $1000$ 步、每步 $\beta_t\le0.02$）：每一步小，真正的逆向一步就很接近常態，用常態的 $p_\theta$ 去近似才合理。這也回答了 KP01 的問題：為什麼要把一件難事拆成很多小步。

That is why diffusion models take many small steps (DDPM uses $1000$ steps with each $\beta_t\le0.02$): when each step is small, the true backward step is close to normal, and approximating it with a normal $p_\theta$ makes sense. This also answers KP01's question of why the hard job is split into many small steps.

常見錯誤：以為真正的逆向一步一定是常態，因為前向一步是常態。上圖右邊就是反例；只有步伐夠小時，它才「近似」常態。

Common mistake: thinking the true backward step must be normal because the forward step is. The right-hand chart above is a counterexample; only when the step is small enough is it "approximately" normal.

### 3. 訓練目標：最大化 ELBO（The training objective: maximise the ELBO）

我們希望模型給真實資料很高的機率，也就是最大化 $\log p_\theta(\mathbf{x}_0)$。可是生成鏈中間有 $\mathbf{x}_1,\ldots,\mathbf{x}_T$ 這些看不到的潛在變數，要把它們全部積分掉（來源的式 (13)）：

We want the model to give real data high probability, that is, to maximise $\log p_\theta(\mathbf{x}_0)$. But the generation chain has the unseen latent variables $\mathbf{x}_1,\ldots,\mathbf{x}_T$ in the middle, which must all be integrated out (the source's equation (13)):

$$
p_\theta(\mathbf{x}_0)=\int p_\theta(\mathbf{x}_{0:T})\,d\mathbf{x}_{1:T}.
$$

這是對很多個和影像一樣大的變數積分，又經過神經網路，算不出來。第12週的 VAE 遇到同樣的問題，解法是改成最大化一個算得出來的下界 ELBO。擴散模型照做，只是把 VAE 的編碼器 $q(\mathbf{z}\mid\mathbf{x})$ 換成固定的前向過程 $q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)$：

This integrates over many variables as large as an image, passed through a neural network, and cannot be computed. The VAE of Week 12 met the same problem and solved it by maximising a computable lower bound, the ELBO. The diffusion model does the same, replacing the VAE's encoder $q(\mathbf{z}\mid\mathbf{x})$ with the fixed forward process $q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)$:

$$
\mathcal{L}=\mathbb{E}_{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}\!\left[\log\frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}\right]\le\log p_\theta(\mathbf{x}_0).
$$

和第12週一樣，兩者的差距正好是一個 KL 散度。其中 $p_\theta(\mathbf{x}_{1:T}\mid\mathbf{x}_0)=p_\theta(\mathbf{x}_{0:T})/p_\theta(\mathbf{x}_0)$ 是模型在已知 $\mathbf{x}_0$ 時，對中間路徑 $\mathbf{x}_{1:T}$ 的看法：$\log p_\theta(\mathbf{x}_0)-\mathcal{L}=\mathrm{KL}\big(q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)\,\Vert\,p_\theta(\mathbf{x}_{1:T}\mid\mathbf{x}_0)\big)\ge0$。所以 ELBO 永遠不會比 $\log p_\theta(\mathbf{x}_0)$ 大；把 ELBO 推高，就是在推高 $\log p_\theta(\mathbf{x}_0)$ 的保底值。和 VAE 不同的是，這裡的「編碼器」$q$ 沒有參數，訓練只會改變 $p_\theta$。

As in Week 12, the gap between the two is exactly a KL divergence. Here $p_\theta(\mathbf{x}_{1:T}\mid\mathbf{x}_0)=p_\theta(\mathbf{x}_{0:T})/p_\theta(\mathbf{x}_0)$ is the model's own view of the middle path $\mathbf{x}_{1:T}$ once $\mathbf{x}_0$ is known: $\log p_\theta(\mathbf{x}_0)-\mathcal{L}=\mathrm{KL}\big(q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)\,\Vert\,p_\theta(\mathbf{x}_{1:T}\mid\mathbf{x}_0)\big)\ge0$. So the ELBO is never larger than $\log p_\theta(\mathbf{x}_0)$; pushing the ELBO up raises a guaranteed floor under $\log p_\theta(\mathbf{x}_0)$. Unlike the VAE, the "encoder" $q$ here has no parameters, so training changes only $p_\theta$.

常見錯誤：以為最大化 ELBO 是因為「ELBO 比較大」或「比較準」。原因只有一個：$\log p_\theta(\mathbf{x}_0)$ 算不出來，ELBO 算得出來，而且它是一個保證不超過的下界。

Common mistake: thinking we maximise the ELBO because it is "larger" or "more accurate". There is only one reason: $\log p_\theta(\mathbf{x}_0)$ cannot be computed, while the ELBO can, and it is a guaranteed lower bound.

### 4. 把 ELBO 拆成每一步（Splitting the ELBO into steps）

上式的分子和分母都是很多項的連乘，取對數後變成很多項相加。分子照式 (10) 拆就好；分母 $q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)=\prod_{t=1}^{T}q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$（KP02 第1節） 要先換個方向，才能和分子的逆向一步一項一項對上。這裡用貝氏定理，並且多放一個條件 $\mathbf{x}_0$（馬可夫性質讓 $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})=q(\mathbf{x}_t\mid\mathbf{x}_{t-1},\mathbf{x}_0)$）：

Both numerator and denominator above are long products, and the logarithm turns them into long sums. The numerator splits directly by equation (10); the denominator $q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)=\prod_{t=1}^{T}q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ (KP02 Section 1) must first be turned around so its factors line up one by one with the backward steps in the numerator. This uses Bayes' rule with an extra condition $\mathbf{x}_0$ (the Markov property gives $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})=q(\mathbf{x}_t\mid\mathbf{x}_{t-1},\mathbf{x}_0)$):

$$
q(\mathbf{x}_t\mid\mathbf{x}_{t-1})=\frac{q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\;q(\mathbf{x}_t\mid\mathbf{x}_0)}{q(\mathbf{x}_{t-1}\mid\mathbf{x}_0)},\qquad t\ge2 .
$$

把 $T=3$ 寫出來看 `[補充]`。分母是 $q(\mathbf{x}_1\mid\mathbf{x}_0)\,q(\mathbf{x}_2\mid\mathbf{x}_1)\,q(\mathbf{x}_3\mid\mathbf{x}_2)$，後兩項用上式代換：

Write it out for $T=3$ `[Added]`. The denominator is $q(\mathbf{x}_1\mid\mathbf{x}_0)\,q(\mathbf{x}_2\mid\mathbf{x}_1)\,q(\mathbf{x}_3\mid\mathbf{x}_2)$; substitute the formula above into the last two factors:

$$
q(\mathbf{x}_1\mid\mathbf{x}_0)\cdot\frac{q(\mathbf{x}_1\mid\mathbf{x}_2,\mathbf{x}_0)\,q(\mathbf{x}_2\mid\mathbf{x}_0)}{q(\mathbf{x}_1\mid\mathbf{x}_0)}\cdot\frac{q(\mathbf{x}_2\mid\mathbf{x}_3,\mathbf{x}_0)\,q(\mathbf{x}_3\mid\mathbf{x}_0)}{q(\mathbf{x}_2\mid\mathbf{x}_0)}=q(\mathbf{x}_3\mid\mathbf{x}_0)\,q(\mathbf{x}_2\mid\mathbf{x}_3,\mathbf{x}_0)\,q(\mathbf{x}_1\mid\mathbf{x}_2,\mathbf{x}_0).
$$

$q(\mathbf{x}_1\mid\mathbf{x}_0)$ 和 $q(\mathbf{x}_2\mid\mathbf{x}_0)$ 各出現一次在分子、一次在分母，兩兩約掉。這種「中間項成對相消、只剩頭尾」的現象叫做**伸縮相消**（telescoping）。$T$ 更大時也一樣，只剩 $q(\mathbf{x}_T\mid\mathbf{x}_0)$ 和每一步的 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$。

$q(\mathbf{x}_1\mid\mathbf{x}_0)$ and $q(\mathbf{x}_2\mid\mathbf{x}_0)$ each appear once on top and once below, and cancel in pairs. This "middle terms cancel in pairs, only the ends survive" effect is called **telescoping**. For larger $T$ it works the same way, leaving only $q(\mathbf{x}_T\mid\mathbf{x}_0)$ and each step's $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$.

現在分子分母可以一項一項配對：$p(\mathbf{x}_T)$ 配 $q(\mathbf{x}_T\mid\mathbf{x}_0)$，得到起點項；$t\ge2$ 時 $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ 配 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$，得到每一步的去噪項；$p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)$ 在分母找不到對象，原樣留下，就是重建項。每一對 $\log(p/q)$ 依 $q$ 取平均，正好是一個負的 KL 散度。於是得到來源的式 (20)：

Now numerator and denominator pair up factor by factor: $p(\mathbf{x}_T)$ pairs with $q(\mathbf{x}_T\mid\mathbf{x}_0)$, giving the start term; for $t\ge2$, $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ pairs with $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$, giving the denoising terms; $p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)$ has no partner in the denominator and stays as it is, giving the reconstruction term. Averaging each pair's $\log(p/q)$ over $q$ gives exactly a negative KL divergence. This yields the source's equation (20):

$$
\mathcal{L}=-\underbrace{\mathrm{KL}\big(q(\mathbf{x}_T\mid\mathbf{x}_0)\,\Vert\,p(\mathbf{x}_T)\big)}_{\text{start}}-\sum_{t=2}^{T}\underbrace{\mathbb{E}_{q}\Big[\mathrm{KL}\big(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\,\Vert\,p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\big)\Big]}_{\text{each denoising step}}+\underbrace{\mathbb{E}_{q}\big[\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)\big]}_{\text{reconstruction}} .
$$

三類項的意思：

What the three kinds of terms mean:

- **起點項**：前向走到底的 $\mathbf{x}_T$ 和生成的起點 $\mathcal{N}(\mathbf{0},I)$ 差多遠。兩者都沒有 $\theta$，訓練改變不了它；而且 $\bar\alpha_T$ 接近 $0$ 時它本來就很小（DDPM 排程、$x_0=1$ 的一維情形只有約 $2\times10^{-5}$）。
- **去噪項**（$t=2,\ldots,T$）：學到的逆向一步 $p_\theta$，要像「已知原資料時」真正的去噪一步 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$。這是訓練的主體（來源的式 (21)）。
- **重建項**：最後一步從 $\mathbf{x}_1$ 回到 $\mathbf{x}_0$ 的機率，和 VAE 的重建項同一個角色。

- **Start term**: how far the end of the forward process $\mathbf{x}_T$ is from the generation start $\mathcal{N}(\mathbf{0},I)$. Neither contains $\theta$, so training cannot change it; and when $\bar\alpha_T$ is near $0$ it is tiny anyway (about $2\times10^{-5}$ for the DDPM schedule in the one-dimensional case with $x_0=1$).
- **Denoising terms** ($t=2,\ldots,T$): the learned backward step $p_\theta$ should resemble the true denoising step "with the original data known", $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$. These are the main body of training (the source's equation (21)).
- **Reconstruction term**: the probability of the last step from $\mathbf{x}_1$ back to $\mathbf{x}_0$, playing the same role as the VAE's reconstruction term.

為什麼要多放一個條件 $\mathbf{x}_0$？不放的話，真正的逆向一步 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ 要知道整個資料分布才算得出來（第2節）；放了 $\mathbf{x}_0$，它就是一個寫得出公式的常態分布，下一個知識點 KP04 會算出來。訓練時 $\mathbf{x}_0$ 是手上的訓練資料，當然知道。

Why add the extra condition $\mathbf{x}_0$? Without it, the true backward step $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ needs the whole data distribution to compute (Section 2); with $\mathbf{x}_0$, it is a normal distribution with a written formula, which the next knowledge point, KP04, works out. During training $\mathbf{x}_0$ is the training example in hand, so of course it is known.

常見錯誤：以為訓練的目標是讓學到的逆向一步 $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ 去接近前向一步 $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$。兩者方向相反，根本不是同一種東西；要接近的是同方向、多看一個 $\mathbf{x}_0$ 的 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$。

Common mistake: thinking training makes the learned backward step $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ approach the forward step $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$. They go in opposite directions and are not the same kind of thing; the target is $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$, in the same direction, with $\mathbf{x}_0$ also seen.

### 5. 一個三步的小例子（A three-step example）

用一個每項都能用公式算出來的小例子，把上面的式子變成數字 `[補充]`。三步，$\beta=(0.2,\ 0.4,\ 0.6)$，所以 $\bar\alpha=(0.8,\ 0.48,\ 0.192)$。資料分布是 $\mathcal{N}(0,1)$，看一筆資料 $x_0=1$。比較兩個模型：

Turn the formulas above into numbers with a small example in which every term has a formula `[Added]`. Three steps with $\beta=(0.2,\ 0.4,\ 0.6)$, so $\bar\alpha=(0.8,\ 0.48,\ 0.192)$. The data distribution is $\mathcal{N}(0,1)$; look at one example $x_0=1$. Compare two models:

- **模型 A**：逆向一步 $p_\theta(x_{t-1}\mid x_t)=\mathcal{N}(\sqrt{\alpha_t}\,x_t,\ \beta_t)$（常態的第二個參數是變異數）。資料是 $\mathcal{N}(0,1)$ 時，前向每一步的分布都還是 $\mathcal{N}(0,1)$，而且 $x_{t-1}$ 和 $x_t$ 的相關係數是 $\sqrt{\alpha_t}$；兩個標準常態、相關係數 $\rho$ 時，已知一個、另一個的條件分布是 $\mathcal{N}(\rho\,x_t,\ 1-\rho^2)$，這裡就是 $\mathcal{N}(\sqrt{\alpha_t}\,x_t,\ \beta_t)$。所以模型 A 正好是**真正的**逆向一步，是完美的模型。
- **模型 B**：把模型 A 的平均都乘上 $0.8$，其他不變，是一個不完美的模型。

- **Model A**: backward step $p_\theta(x_{t-1}\mid x_t)=\mathcal{N}(\sqrt{\alpha_t}\,x_t,\ \beta_t)$ (the second argument of the normal is the variance). With data $\mathcal{N}(0,1)$, the forward distribution stays $\mathcal{N}(0,1)$ at every step, and $x_{t-1}$ and $x_t$ have correlation $\sqrt{\alpha_t}$; for two standard normals with correlation $\rho$, the conditional of one given the other is $\mathcal{N}(\rho\,x_t,\ 1-\rho^2)$, which here is $\mathcal{N}(\sqrt{\alpha_t}\,x_t,\ \beta_t)$. So model A is exactly the **true** backward step, a perfect model.
- **Model B**: model A with every mean multiplied by $0.8$, otherwise unchanged; an imperfect model.

![一張分組長條圖，縱軸是數值（單位 nats），從 -1.75 到 0.25。六組長條，每組藍色是模型 A（真正的逆向）、橘色是模型 B（平均乘 0.8）。負的起點 KL：兩者都是 -0.107；負的第 2 步 KL：-0.478 與 -0.508；負的第 3 步 KL：-0.220 與 -0.234；重建項 log p(x0 | x1)：-0.614 與 -0.694；ELBO（總和）：-1.419 與 -1.542；log p(x0)：-1.419 與 -1.513。標題寫三步模型、資料 N(0,1)、x0=1，ELBO 小於等於 log p(x0)（A grouped bar chart with value (nats) on the vertical axis from -1.75 to 0.25. Six groups, each with blue for model A (exact reverse) and orange for model B (means times 0.8). Negative start KL: -0.107 for both; negative KL at step 2: -0.478 and -0.508; negative KL at step 3: -0.220 and -0.234; reconstruction log p(x0 | x1): -0.614 and -0.694; ELBO (sum): -1.419 and -1.542; log p(x0): -1.419 and -1.513. The title reads 3-step model, data N(0,1), x0 = 1, ELBO at most log p(x0))](../assets/figures/lec14-kp03-elbo-terms.png)

**圖說 `lec14-kp03-elbo-terms`**〔功能：精準度、對照〕：前四組是 ELBO 的四項，第五組是它們的和，最後一組是真正的 $\log p_\theta(x_0)$。先看藍色：四項加起來 $-0.107-0.478-0.220-0.614=-1.419$，正好等於 $\log p_\theta(x_0)$。再看橘色：每一項都比藍色差一點（起點項除外，它跟模型無關），總和 $-1.542$ 比它自己的 $\log p_\theta(x_0)=-1.513$ 低 $0.029$。

**Caption `lec14-kp03-elbo-terms`** [Function: precision, comparison]: the first four groups are the four terms of the ELBO, the fifth is their sum, and the last is the true $\log p_\theta(x_0)$. First the blue: the four terms add up to $-0.107-0.478-0.220-0.614=-1.419$, exactly $\log p_\theta(x_0)$. Then the orange: every term is a little worse than the blue (except the start term, which does not depend on the model), and the sum $-1.542$ is $0.029$ below its own $\log p_\theta(x_0)=-1.513$.

從這張圖可以讀出三件事。第一，ELBO 不會超過 $\log p_\theta(x_0)$：模型 B 差了 $0.029$；完美的模型 A 差距是 $0$，這時下界剛好「貼住」。直接估計 ELBO 的做法是：從 $q(x_{1:3}\mid x_0)$ 抽 $400000$ 條路徑，把每條的 $\log p_\theta(x_{0:3})-\log q(x_{1:3}\mid x_0)$ 平均起來。這樣做，模型 B 得到 $-1.5422$（標準誤 $0.0004$），和公式算的 $-1.5423$ 一致；模型 A 的每一次抽樣都剛好等於 $-1.4189$，因為完美模型下括號裡的值根本不隨機。

Three things can be read from this figure. First, the ELBO never exceeds $\log p_\theta(x_0)$: model B falls short by $0.029$; for the perfect model A the gap is $0$, and the bound is "tight". Estimating the ELBO directly means drawing $400000$ paths from $q(x_{1:3}\mid x_0)$ and averaging $\log p_\theta(x_{0:3})-\log q(x_{1:3}\mid x_0)$ over them. Doing so gives $-1.5422$ for model B (standard error $0.0004$), matching the formula's $-1.5423$; for model A every single sample equals $-1.4189$, because under a perfect model the quantity in the brackets is not random at all.

第二，就算模型完美，第 $2$、$3$ 步的 KL 也不是 $0$（$0.478$ 與 $0.220$）。原因是 $p_\theta(x_{t-1}\mid x_t)$ 看不到 $x_0$，它只能給「所有可能的 $x_0$ 平均起來」的答案，永遠比不上多看了 $x_0$ 的 $q(x_{t-1}\mid x_t,x_0)$。所以訓練擴散模型時，損失不會降到 $0$，這是正常的。

Second, even for the perfect model, the KL at steps $2$ and $3$ is not $0$ ($0.478$ and $0.220$). The reason is that $p_\theta(x_{t-1}\mid x_t)$ cannot see $x_0$; it can only answer "averaged over all possible $x_0$", which can never match $q(x_{t-1}\mid x_t,x_0)$, which also sees $x_0$. So when training a diffusion model, the loss does not drop to $0$, and that is normal.

第三，起點項兩個模型都是 $0.107$，改模型改不動它。這裡它不小，是因為只走三步、$\bar\alpha_3=0.192$ 離 $0$ 還遠；DDPM 走 $1000$ 步，它就小到可以忽略。

Third, the start term is $0.107$ for both models; changing the model cannot move it. It is not small here only because there are just three steps and $\bar\alpha_3=0.192$ is far from $0$; with DDPM's $1000$ steps it becomes negligible.

常見錯誤：以為訓練好的擴散模型，每一步的 KL 都應該接近 $0$。上例的完美模型就是反例；能降到的最低點由「看不到 $x_0$」決定，不是 $0$。

Common mistake: thinking a well-trained diffusion model should have a KL near $0$ at every step. The perfect model above is a counterexample; the lowest reachable value is set by "not seeing $x_0$", not by $0$.

### 本單元重點整理（KP03 summary）

**一句話重點**：逆向過程是一條從 $\mathcal{N}(\mathbf{0},I)$ 出發、每步都是常態 $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ 的鏈；$\log p_\theta(\mathbf{x}_0)$ 算不出來，所以最大化 ELBO，而 ELBO 可以拆成起點項、每一步的去噪 KL 與重建項，訓練主要在讓 $p_\theta$ 接近 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$。

**One-sentence summary**: the backward process is a chain starting from $\mathcal{N}(\mathbf{0},I)$ in which every step is a normal $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$; since $\log p_\theta(\mathbf{x}_0)$ cannot be computed, we maximise the ELBO, which splits into a start term, one denoising KL per step and a reconstruction term, and training mainly brings $p_\theta$ close to $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$.

**白話說明**：想像老師批改「倒帶」作業。學生只看得到現在這一格畫面，要猜上一格；老師手上有原始影片，知道正確的上一格大概在哪。ELBO 就是把每一格的「學生的猜測」和「老師看著原片給的答案」差多少加起來。學生永遠看不到原片，所以分數不會完美，但越接近越好。

**Plain explanation**: imagine a teacher marking a "rewind" assignment. The student sees only the current frame and must guess the previous one; the teacher holds the original film and knows roughly where the right previous frame is. The ELBO adds up, frame by frame, how far "the student's guess" is from "the answer the teacher gives while watching the original". The student never sees the original, so the score is never perfect, but the closer the better.

**正式定義**：$p_\theta(\mathbf{x}_{0:T})=p(\mathbf{x}_T)\prod_{t=1}^{T}p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$，$p(\mathbf{x}_T)=\mathcal{N}(\mathbf{0},I)$，$p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)=\mathcal{N}(\mu_\theta(\mathbf{x}_t,t),\sigma_t^2I)$。ELBO $\mathcal{L}=\mathbb{E}_q[\log p_\theta(\mathbf{x}_{0:T})-\log q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)]\le\log p_\theta(\mathbf{x}_0)$，且 $\mathcal{L}=-\mathrm{KL}(q(\mathbf{x}_T\mid\mathbf{x}_0)\Vert p(\mathbf{x}_T))-\sum_{t=2}^{T}\mathbb{E}_q[\mathrm{KL}(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\Vert p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t))]+\mathbb{E}_q[\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)]$。

**Formal definition**: $p_\theta(\mathbf{x}_{0:T})=p(\mathbf{x}_T)\prod_{t=1}^{T}p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ with $p(\mathbf{x}_T)=\mathcal{N}(\mathbf{0},I)$ and $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)=\mathcal{N}(\mu_\theta(\mathbf{x}_t,t),\sigma_t^2I)$. The ELBO is $\mathcal{L}=\mathbb{E}_q[\log p_\theta(\mathbf{x}_{0:T})-\log q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)]\le\log p_\theta(\mathbf{x}_0)$, and $\mathcal{L}=-\mathrm{KL}(q(\mathbf{x}_T\mid\mathbf{x}_0)\Vert p(\mathbf{x}_T))-\sum_{t=2}^{T}\mathbb{E}_q[\mathrm{KL}(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\Vert p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t))]+\mathbb{E}_q[\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)]$.

**例子**：三步、$\beta=(0.2,0.4,0.6)$、資料 $\mathcal{N}(0,1)$、$x_0=1$。完美的模型 A：四項 $-0.107$、$-0.478$、$-0.220$、$-0.614$，和為 $-1.419=\log p_\theta(x_0)$。平均乘 $0.8$ 的模型 B：ELBO $-1.542$，比 $\log p_\theta(x_0)=-1.513$ 低 $0.029$。

**Example**: three steps, $\beta=(0.2,0.4,0.6)$, data $\mathcal{N}(0,1)$, $x_0=1$. The perfect model A: four terms $-0.107$, $-0.478$, $-0.220$ and $-0.614$, summing to $-1.419=\log p_\theta(x_0)$. Model B, with means times $0.8$: ELBO $-1.542$, which is $0.029$ below $\log p_\theta(x_0)=-1.513$.

**常見迷思**：

- 「逆向一步每一步各用一個網路。」錯，所有步共用一個網路，靠輸入 $t$ 分辨。
- 「真正的逆向一步一定是常態。」錯，只有步伐小時才近似常態。
- 「最大化 ELBO 是因為它比較大。」錯，是因為 $\log p_\theta(\mathbf{x}_0)$ 算不出來，ELBO 是算得出來的下界。
- 「$p_\theta$ 要去接近前向一步 $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$。」錯，要接近的是 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$。
- 「完美的模型每一步 KL 都是 $0$。」錯，$p_\theta$ 看不到 $\mathbf{x}_0$，KL 有下限。

**Common misconceptions**:

- "Each backward step uses its own network." Wrong: all steps share one network, told apart by the input $t$.
- "The true backward step must be normal." Wrong: it is only approximately normal when the step is small.
- "We maximise the ELBO because it is larger." Wrong: because $\log p_\theta(\mathbf{x}_0)$ cannot be computed, and the ELBO is a computable lower bound.
- "$p_\theta$ should approach the forward step $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$." Wrong: the target is $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$.
- "A perfect model has zero KL at every step." Wrong: $p_\theta$ cannot see $\mathbf{x}_0$, so the KL has a floor.

**圖**：本單元有三張圖。

- `lec14-kp03-two-chains`：同一串變數的固定前向鏈與要學的逆向鏈。
- `lec14-kp03-small-vs-large-step`：小步時真正的逆向一步近似常態，大步時變成兩座山。
- `lec14-kp03-elbo-terms`：三步小模型的 ELBO 各項、總和與 $\log p_\theta(x_0)$。

**Figures** — this unit has three figures.

- `lec14-kp03-two-chains`: the fixed forward chain and the learned backward chain over the same variables.
- `lec14-kp03-small-vs-large-step`: with a small step the true backward step is nearly normal; with a large step it has two peaks.
- `lec14-kp03-elbo-terms`: the ELBO terms, their sum and $\log p_\theta(x_0)$ for the three-step model.

### 小練習（Practice）

1. 用第5節模型 B 的四項（$-0.107$、$-0.508$、$-0.234$、$-0.694$）算出 ELBO，再和 $\log p_\theta(x_0)=-1.513$ 比較。差距是多少？它代表哪一個 KL？
2. 照第4節 $T=3$ 的寫法，把 $T=4$ 的分母 $q(\mathbf{x}_{1:4}\mid\mathbf{x}_0)$ 寫成「起點項乘上三個去噪一步」。哪些項被約掉了？
3. 一位同學把 DDPM 改成只走 $T=5$ 步，每步 $\beta_t=0.5$。根據第2節，這樣做的逆向過程會遇到什麼問題？
4. 為什麼 ELBO 的起點項在訓練時可以不管？它在什麼情況下不可以忽略（雖然仍然跟 $\theta$ 無關）？

1. Using the four terms of model B in Section 5 ($-0.107$, $-0.508$, $-0.234$, $-0.694$), compute the ELBO and compare it with $\log p_\theta(x_0)=-1.513$. What is the gap? Which KL does it stand for?
2. Following Section 4's $T=3$ write-up, write the denominator $q(\mathbf{x}_{1:4}\mid\mathbf{x}_0)$ for $T=4$ as "the start term times three denoising steps". Which factors cancel?
3. A classmate changes DDPM to take only $T=5$ steps with $\beta_t=0.5$ each. Based on Section 2, what problem will the backward process run into?
4. Why can the ELBO's start term be ignored during training? When is it not negligible (even though it still does not depend on $\theta$)?

**第 1 題解答** `[補充]`：$-0.107-0.508-0.234-0.694=-1.543$（用未四捨五入的值是 $-1.542$）。差距約 $1.542-1.513=0.029$，就是第3節的 $\mathrm{KL}\big(q(x_{1:3}\mid x_0)\,\Vert\,p_\theta(x_{1:3}\mid x_0)\big)$：前向過程給的「中間路徑」和模型認為的路徑差多遠。

**Solution to question 1** `[Added]`: $-0.107-0.508-0.234-0.694=-1.543$ ($-1.542$ with the unrounded values). The gap is about $1.542-1.513=0.029$, which is Section 3's $\mathrm{KL}\big(q(x_{1:3}\mid x_0)\,\Vert\,p_\theta(x_{1:3}\mid x_0)\big)$: how far the "middle path" given by the forward process is from the path the model believes in.

**第 2 題解答** `[補充]`：$q(\mathbf{x}_{1:4}\mid\mathbf{x}_0)=q(\mathbf{x}_1\mid\mathbf{x}_0)\prod_{t=2}^{4}\frac{q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\,q(\mathbf{x}_t\mid\mathbf{x}_0)}{q(\mathbf{x}_{t-1}\mid\mathbf{x}_0)}$。分子的 $q(\mathbf{x}_2\mid\mathbf{x}_0)$、$q(\mathbf{x}_3\mid\mathbf{x}_0)$ 和分母的同名項約掉，最前面的 $q(\mathbf{x}_1\mid\mathbf{x}_0)$ 也和分母第一項約掉，剩下 $q(\mathbf{x}_4\mid\mathbf{x}_0)\,q(\mathbf{x}_3\mid\mathbf{x}_4,\mathbf{x}_0)\,q(\mathbf{x}_2\mid\mathbf{x}_3,\mathbf{x}_0)\,q(\mathbf{x}_1\mid\mathbf{x}_2,\mathbf{x}_0)$。

**Solution to question 2** `[Added]`: $q(\mathbf{x}_{1:4}\mid\mathbf{x}_0)=q(\mathbf{x}_1\mid\mathbf{x}_0)\prod_{t=2}^{4}\frac{q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\,q(\mathbf{x}_t\mid\mathbf{x}_0)}{q(\mathbf{x}_{t-1}\mid\mathbf{x}_0)}$. The factors $q(\mathbf{x}_2\mid\mathbf{x}_0)$ and $q(\mathbf{x}_3\mid\mathbf{x}_0)$ on top cancel with the same ones below, and the leading $q(\mathbf{x}_1\mid\mathbf{x}_0)$ cancels with the first factor below, leaving $q(\mathbf{x}_4\mid\mathbf{x}_0)\,q(\mathbf{x}_3\mid\mathbf{x}_4,\mathbf{x}_0)\,q(\mathbf{x}_2\mid\mathbf{x}_3,\mathbf{x}_0)\,q(\mathbf{x}_1\mid\mathbf{x}_2,\mathbf{x}_0)$.

**第 3 題解答** `[補充]`：每一步很大（$\beta=0.5$，和第2節右圖一樣），真正的逆向一步可能有好幾座山，用一個常態的 $p_\theta$ 去近似會很差：它只能把機率放在幾座山的中間，生成出模糊或「介於兩類之間」的結果。要嘛增加步數、縮小每步，要嘛換一個更有彈性的逆向分布。注意這時 $\bar\alpha_5=0.5^5\approx0.031$，起點項其實不大；真正的問題在步伐太大，不在終點。

**Solution to question 3** `[Added]`: each step is large ($\beta=0.5$, as in the right-hand chart of Section 2), so the true backward step can have several peaks, and a single normal $p_\theta$ approximates it badly: it can only put probability between the peaks, generating blurry or "in-between" results. Either take more, smaller steps, or use a more flexible backward distribution. Note that here $\bar\alpha_5=0.5^5\approx0.031$, so the start term is actually small; the real problem is the step size, not the end point.

**第 4 題解答** `[補充]`：起點項 $\mathrm{KL}(q(\mathbf{x}_T\mid\mathbf{x}_0)\Vert\mathcal{N}(\mathbf{0},I))$ 裡沒有 $\theta$，對 $\theta$ 的梯度是 $0$，所以訓練時不用算。但它的大小代表「前向的終點」和「生成的起點」不一致的程度：如果 $\bar\alpha_T$ 離 $0$ 還遠（例如第5節只走三步，起點項是 $0.107$），從 $\mathcal{N}(\mathbf{0},I)$ 開始生成就和訓練時看到的 $\mathbf{x}_T$ 不一樣，生成品質會受影響。這要靠選好排程解決，不是靠訓練。

**Solution to question 4** `[Added]`: the start term $\mathrm{KL}(q(\mathbf{x}_T\mid\mathbf{x}_0)\Vert\mathcal{N}(\mathbf{0},I))$ contains no $\theta$, so its gradient with respect to $\theta$ is $0$ and training need not compute it. But its size measures how much "the end of the forward process" disagrees with "the start of generation": if $\bar\alpha_T$ is still far from $0$ (for example, three steps in Section 5 give a start term of $0.107$), generation that starts from $\mathcal{N}(\mathbf{0},I)$ differs from the $\mathbf{x}_T$ seen in training, which hurts sample quality. That is fixed by choosing the schedule well, not by training.

### 小測驗（Quiz，來源原題）

來源 KP-3 與 KP-4 各有四題自我檢測，保留原題與全部選項；來源頁面附有答案與簡短解析，下面的解析由本講義依各節內容補寫。

The source's KP-3 and KP-4 each have four self-check questions, kept here with all their options; the source page gives the answers with short explanations, and the explanations below are written by this note from its own sections.

1. Backward process 從什麼分布開始取樣 $x_T$？
   - (A) 訓練資料分布
   - (B) 標準高斯 $\mathcal{N}(0,I)$
   - (C) 均勻分布
   - (D) $x_0$ 的複本
2. $p_\theta(x_{t-1}\mid x_t)$ 通常假設為什麼分布？
   - (A) 高斯分布
   - (B) 均勻分布
   - (C) 拉普拉斯分布
   - (D) 離散分布
3. Isotropic covariance 表示什麼？
   - (A) 所有維度有相同變異數
   - (B) 變異數為零
   - (C) 沒有共變異
   - (D) 均值為零
4. 生成過程的聯合分布 $p_\theta(x_{0:T})$ 如何表示？
   - (A) $p(x_T)$ 乘以所有 backward transition 的連乘
   - (B) 單一高斯
   - (C) $q(x_{1:T}\mid x_0)$
   - (D) 條件機率的加總
5. 為什麼 diffusion model 要最大化 ELBO 而不是直接最大化 $p_\theta(x_0)$？
   - (A) $p_\theta(x_0)$ 的積分無法直接計算
   - (B) ELBO 比較大
   - (C) ELBO 不需要 neural network
   - (D) 因為 forward process 不固定
6. 式 (21) 的訓練目標是什麼？
   - (A) 最大化 forward 與 backward 的 KL
   - (B) 最小化 $q(x_{t-1}\mid x_t,x_0)$ 與 $p_\theta(x_{t-1}\mid x_t)$ 的 KL
   - (C) 最小化資料與雜訊的均方誤差
   - (D) 最大化 VAE 的 ELBO
7. ELBO 推導中使用了哪個關鍵數學技巧把中間項消掉？
   - (A) 泰勒展開
   - (B) Telescoping sum
   - (C) Fourier transform
   - (D) 梯度下降
8. 式 (21) 中第一項 $\mathrm{KL}(q(x_T\mid x_0)\Vert p(x_T))$ 為什麼可以忽略？（原題如此；此項在式 (20)，見來源更正）
   - (A) 它與模型參數 $\theta$ 無關
   - (B) 它永遠為零
   - (C) 它不重要
   - (D) 它等於無限大

1. From what distribution does the backward process sample $x_T$?
   - (A) The training-data distribution
   - (B) The standard Gaussian $\mathcal{N}(0,I)$
   - (C) A uniform distribution
   - (D) A copy of $x_0$
2. What distribution is $p_\theta(x_{t-1}\mid x_t)$ usually assumed to be?
   - (A) Gaussian
   - (B) Uniform
   - (C) Laplace
   - (D) Discrete
3. What does an isotropic covariance mean?
   - (A) All dimensions have the same variance
   - (B) The variance is zero
   - (C) There is no covariance
   - (D) The mean is zero
4. How is the joint distribution of generation $p_\theta(x_{0:T})$ written?
   - (A) $p(x_T)$ times the product of all backward transitions
   - (B) A single Gaussian
   - (C) $q(x_{1:T}\mid x_0)$
   - (D) A sum of conditional probabilities
5. Why does a diffusion model maximise the ELBO rather than $p_\theta(x_0)$ directly?
   - (A) The integral for $p_\theta(x_0)$ cannot be computed directly
   - (B) The ELBO is larger
   - (C) The ELBO needs no neural network
   - (D) Because the forward process is not fixed
6. What is the training objective of equation (21)?
   - (A) Maximise the KL between forward and backward
   - (B) Minimise the KL between $q(x_{t-1}\mid x_t,x_0)$ and $p_\theta(x_{t-1}\mid x_t)$
   - (C) Minimise the mean squared error between data and noise
   - (D) Maximise the VAE's ELBO
7. Which key mathematical trick cancels the middle terms in the ELBO derivation?
   - (A) Taylor expansion
   - (B) Telescoping sum
   - (C) Fourier transform
   - (D) Gradient descent
8. Why can the first term $\mathrm{KL}(q(x_T\mid x_0)\Vert p(x_T))$ in equation (21) be ignored? (Worded as in the source; the term belongs to equation (20); see the corrections.)
   - (A) It does not depend on the model parameters $\theta$
   - (B) It is always zero
   - (C) It is unimportant
   - (D) It is infinite

**解析** `[補充]`：

- 第 1 題選 (B)：第1節，生成的起點 $p(\mathbf{x}_T)=\mathcal{N}(\mathbf{0},I)$；生成時手上沒有任何資料，所以 (A)、(D) 不可能。
- 第 2 題選 (A)：第1節的式 (7)；第2節說明這個假設在步伐小時才合理。
- 第 3 題選 (A)：等向就是每個維度的變異數都是 $\sigma_t^2$。(C)「沒有共變異」只說了維度之間無關，沒說變異數相同，所以不夠。
- 第 4 題選 (A)：第1節的式 (10)。(C) 是前向過程的分布，不是生成過程。
- 第 5 題選 (A)：第3節。(B) 剛好相反，ELBO 永遠不大於 $\log p_\theta(\mathbf{x}_0)$。
- 第 6 題選 (B)：第4節的去噪項。(C) 是 KP05 簡化之後的樣子，不是式 (21) 本身。
- 第 7 題選 (B)：第4節的伸縮相消。
- 第 8 題選 (A)：第4節與小練習第 4 題。它不是永遠為零，(B) 錯；只走幾步時它可能不小。

**Explanations** `[Added]`:

- Question 1: B: Section 1; the start of generation is $p(\mathbf{x}_T)=\mathcal{N}(\mathbf{0},I)$; no data are in hand during generation, so A and D are impossible.
- Question 2: A: equation (7) in Section 1; Section 2 explains that this assumption is reasonable only when steps are small.
- Question 3: A: isotropic means every dimension has variance $\sigma_t^2$. C, "no covariance", only says the dimensions are unrelated, not that the variances are equal, so it falls short.
- Question 4: A: equation (10) in Section 1. C is the distribution of the forward process, not of generation.
- Question 5: A: Section 3. B is backwards: the ELBO is never larger than $\log p_\theta(\mathbf{x}_0)$.
- Question 6: B: the denoising terms of Section 4. C is what it becomes after KP05's simplification, not equation (21) itself.
- Question 7: B: the telescoping of Section 4.
- Question 8: A: Section 4 and practice question 4. It is not always zero, so B is wrong; with only a few steps it can be sizeable.

### 留給下一單元的問題（The question for the next unit）

訓練的主體是讓 $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ 接近 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$。可是後者長什麼樣子？第4節說它是一個寫得出公式的常態分布，它的平均和變異數是多少？網路的輸出 $\mu_\theta$ 又該怎麼設計，才最容易對上它？下一個知識點（本週 KP04）用貝氏定理和配平方把它算出來，並推出「改成預測雜訊」的想法。

The main body of training brings $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ close to $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$. But what does the latter look like? Section 4 said it is a normal distribution with a written formula; what are its mean and variance? And how should the network's output $\mu_\theta$ be designed to match it most easily? The next knowledge point (this week's KP04) works it out with Bayes' rule and completing the square, and arrives at the idea of "predicting the noise instead".

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第15章「Diffusion Models」的 KP-3「Backward Process：$p_\theta$ 高斯與 isotropic covariance」與 KP-4「ELBO 推導：KL 拆解與 telescoping sum」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch15-diffusion-models/index.html ，讀取日期 2026-11-15）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from KP-3, "Backward process: Gaussian $p_\theta$ and isotropic covariance", and KP-4, "ELBO derivation: KL decomposition and telescoping sum", of Chapter 15, "Diffusion Models", of CCC-LNS IDL_115 (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch15-diffusion-models/index.html , accessed 2026-11-15). Content not in the source is marked `[Added]` where it appears.

- 來源的式 (14) 與式 (15) 把 ELBO 寫成「$\le p_\theta(x)$」「$\le p_\theta(x_0)$」。ELBO 是**對數**機率的下界，應為 $\le\log p_\theta(\mathbf{x}_0)$；本講義第3節照此寫，並寫出兩者的差距是一個 KL。
- 來源說式 (20-21) 的「最後一項通常被忽略」。重建項 $\mathbb{E}_q[\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)]$ 在 DDPM 裡並沒有被丟掉：原論文用一個專門的解碼器處理它，簡化後的損失則把它當成 $t=1$ 那一項一起訓練（本週 KP05）。可以不管的是起點項，因為它沒有 $\theta$。本講義第4節照此寫。
- 來源的小測驗第 8 題寫「式 (21) 中第一項」；第一項 $\mathrm{KL}(q(x_T\mid x_0)\Vert p(x_T))$ 在式 (20) 裡，式 (21) 已經把它拿掉了。小測驗照錄原題。
- 來源在逆向過程一節寫出 $q(x_{t-1}\mid x_t,x_{t+1},\ldots,x_T)=q(x_{t-1}\mid x_t)$ 作為馬可夫性質。這是前向鏈「倒過來看」也有的性質，本身沒錯；但它不表示 $q(x_{t-1}\mid x_t)$ 算得出來，本講義第2、4節補充說明它需要整個資料分布。
- 本講義補上：所有步數共用一個網路、為什麼步伐小時逆向一步近似常態（附圖與數字）、ELBO 與 $\log p_\theta$ 的差距是 KL、$T=3$ 的伸縮相消逐項寫出、三類項的意義與起點項在 DDPM 下的大小、為什麼要多放條件 $\mathbf{x}_0$、三步小模型的 ELBO 各項與蒙地卡羅核對、完美模型的 KL 也不為 $0$、四題小練習與解答、小測驗的解析。
- 取捨：來源兩個互動展示不重做，改用三步小模型的長條圖；符號加粗與否的取捨同 KP02。

- The source's equations (14) and (15) write the ELBO as "$\le p_\theta(x)$" and "$\le p_\theta(x_0)$". The ELBO is a lower bound on the **log** probability and should read $\le\log p_\theta(\mathbf{x}_0)$; Section 3 writes it this way and states that the gap is a KL.
- The source says "the last term is usually ignored" in equations (20–21). In DDPM the reconstruction term $\mathbb{E}_q[\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)]$ is not dropped: the original paper handles it with a dedicated decoder, and the simplified loss trains it together as the $t=1$ term (this week's KP05). The term that can be left alone is the start term, because it has no $\theta$. Section 4 follows this.
- The source's quiz question 8 says "the first term in equation (21)"; the first term $\mathrm{KL}(q(x_T\mid x_0)\Vert p(x_T))$ belongs to equation (20), and equation (21) has already removed it. The quiz keeps the original wording.
- In its backward-process section, the source writes $q(x_{t-1}\mid x_t,x_{t+1},\ldots,x_T)=q(x_{t-1}\mid x_t)$ as the Markov property. This is a property the forward chain also has "read backwards", and it is correct; but it does not mean $q(x_{t-1}\mid x_t)$ can be computed, and Sections 2 and 4 add that it needs the whole data distribution.
- This note adds: one network shared by all steps; why a backward step is nearly normal when steps are small (with a figure and numbers); the gap between the ELBO and $\log p_\theta$ being a KL; the telescoping written out term by term for $T=3$; the meaning of the three kinds of terms and the size of the start term under DDPM; why the extra condition $\mathbf{x}_0$ is added; the ELBO terms of a three-step model with a Monte Carlo check; the point that even a perfect model's KL is not $0$; four practice questions with solutions; and the explanations of the quiz.
- Choices: the source's two interactive demos are not rebuilt; the bar chart of the three-step model is used instead. Bold versus plain symbols follow the choice made in KP02.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 逆向過程（反向過程） | backward process (reverse process) |
| 去噪 | denoising |
| 等向共變異數 | isotropic covariance |
| 聯合分布 | joint distribution |
| 潛在變數 | latent variable |
| 概似 | likelihood |
| 證據下界 | evidence lower bound (ELBO) |
| KL 散度 | Kullback–Leibler (KL) divergence |
| 貝氏定理 | Bayes' rule |
| 馬可夫性質 | Markov property |
| 伸縮相消 | telescoping |
| 起點項 | start term |
| 去噪項 | denoising term |
| 重建項 | reconstruction term |
| 蒙地卡羅估計 | Monte Carlo estimate |
| 標準誤 | standard error |
| 奈特（自然對數的資訊單位） | nat (unit of information with natural logarithm) |
| 配平方（KP04） | completing the square |
| 去雜訊擴散機率模型 | denoising diffusion probabilistic model (DDPM) |
| 變分自編碼器（第12週） | variational autoencoder (VAE) |
| 泰勒展開 | Taylor expansion |
| 傅立葉轉換 | Fourier transform |
| 梯度下降 | gradient descent |
