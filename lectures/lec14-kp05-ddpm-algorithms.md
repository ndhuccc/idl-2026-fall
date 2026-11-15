# 第14週 KP05：損失、訓練與取樣：DDPM 的兩個演算法（Loss, training and sampling: the two algorithms of DDPM）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第14週「深度生成模型（三）：擴散模型」的第五個知識點（KP05）。前面四個知識點把零件都備齊了：前向過程與閉式（KP02）、ELBO 拆成每一步的 KL（KP03）、真正的去噪分布與「預測雜訊」的寫法（KP04）。這一節把它們組起來：先把每一步的 KL 算成「猜的雜訊和真的雜訊差多少」，再簡化成 DDPM 實際用的損失，最後寫出訓練與生成的兩個演算法，並在一維資料上真的訓練一個小模型。

This is the fifth knowledge point (KP05) of Week 14, "Deep generative models (3): diffusion models". The previous four knowledge points prepared all the parts: the forward process and its closed form (KP02), the ELBO split into per-step KLs (KP03), and the true denoising distribution with the "predict the noise" form (KP04). This note assembles them: first each step's KL is worked out as "how far the guessed noise is from the true noise", then it is simplified into the loss DDPM actually uses, and finally the two algorithms for training and generation are written out, and a small model is actually trained on one-dimensional data.

先備知識是本週 KP02 到 KP04，以及兩個一維常態分布之間的 KL 公式（第12週 KP02）：$\mathrm{KL}\big(\mathcal{N}(m_1,v_1)\,\Vert\,\mathcal{N}(m_2,v_2)\big)=\tfrac12\big[\log\tfrac{v_2}{v_1}+\tfrac{v_1+(m_1-m_2)^2}{v_2}-1\big]$。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第15章 KP-8（2.5 節）與 KP-9（2.6–2.7 節）改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec14-kp05-figures.py` 算出，可以自己重算（只需要 numpy 與 matplotlib，約兩分鐘）。

The prerequisites are this week's KP02 to KP04, and the KL formula between two one-dimensional normal distributions (Week 12 KP02): $\mathrm{KL}\big(\mathcal{N}(m_1,v_1)\,\Vert\,\mathcal{N}(m_2,v_2)\big)=\tfrac12\big[\log\tfrac{v_2}{v_1}+\tfrac{v_1+(m_1-m_2)^2}{v_2}-1\big]$. Every term and symbol is defined when it first appears. This note is rewritten from KP-8 (Section 2.5) and KP-9 (Sections 2.6–2.7) of Chapter 15 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec14-kp05-figures.py`, so you can recompute it yourself (it needs only numpy and matplotlib and takes about two minutes).

## KP05 損失、訓練與取樣：DDPM 的兩個演算法（Loss, training and sampling: the two algorithms of DDPM）

### 學習目標（Learning objectives）

讀完之後，你能從 KP03 的每一步 KL 推出「加權的雜訊平方誤差」，算出某一步的權重，並說明 DDPM 為什麼把權重全部換成 $1$。你能寫出演算法 1（訓練）與演算法 2（生成）的每一步，說出生成時最後一步為什麼不加雜訊，以及 $\sigma_t^2$ 的兩種常見選擇。你能手算演算法 2 的一步。你能讀懂一個一維小 DDPM 的訓練結果：損失為什麼停在某個不是 $0$ 的值、哪些步數最難、生成的樣本像不像資料。你也能比較 DDPM 與 GAN 在訓練與生成上的取捨。

After reading, you can derive a "weighted squared error of the noise" from KP03's per-step KL, compute the weight of a step, and explain why DDPM replaces all weights by $1$. You can write every step of Algorithm 1 (training) and Algorithm 2 (generation), say why the last generation step adds no noise, and name the two common choices of $\sigma_t^2$. You can compute one step of Algorithm 2 by hand. You can read the training results of a small one-dimensional DDPM: why the loss stops at a value other than $0$, which steps are hardest, and whether the generated samples look like the data. You can also compare the trade-offs of DDPM and GAN in training and generation.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\beta_t,\ \alpha_t,\ \bar\alpha_t$ | 雜訊排程與它的連乘（KP02） | the noise schedule and its running product (KP02) |
| $\epsilon$ | 訓練時自己抽的標準常態雜訊，用來從 $\mathbf{x}_0$ 一步跳到 $\mathbf{x}_t$ | standard-normal noise drawn in training, used to jump from $\mathbf{x}_0$ to $\mathbf{x}_t$ |
| $\epsilon_\theta(\mathbf{x}_t,t)$ | 網路對 $\epsilon$ 的猜測（KP04） | the network's guess of $\epsilon$ (KP04) |
| $\tilde\mu,\ \tilde\beta_t$ | 真正去噪分布的平均與變異數（KP04） | mean and variance of the true denoising distribution (KP04) |
| $\sigma_t^2$ | 生成時逆向一步的變異數，事先選定 | the variance of a backward step in generation, chosen in advance |
| $w_t$ | 完整 ELBO 裡第 $t$ 步的權重 | the weight of step $t$ in the full ELBO |
| $L_{\text{simple}}$ | DDPM 的簡化損失 | DDPM's simplified loss |
| $\mathbf{z}$ | 生成時每一步新抽的標準常態雜訊（和 $\epsilon$ 不同） | fresh standard-normal noise drawn at each generation step (not the same as $\epsilon$) |
| $\mathrm{Uniform}\{1,\ldots,T\}$ | 從 $1$ 到 $T$ 的整數裡，每個機會相同地抽一個 | draw one integer from $1$ to $T$, each equally likely |
| $E[\epsilon\mid\mathbf{x}_t]$ | 已知 $\mathbf{x}_t$ 時 $\epsilon$ 的條件平均：最好的猜測（KP04） | the conditional mean of $\epsilon$ given $\mathbf{x}_t$: the best guess (KP04) |
| $\Vert\mathbf{v}\Vert^2$ | 向量各元素平方的和 | the sum of the squared elements of a vector |

### 1. 從每一步的 KL 到「猜雜訊的平方誤差」（From each step's KL to "the squared error of guessing the noise"）

KP03 說訓練要縮小每一步的 KL：$\mathrm{KL}\big(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\,\Vert\,p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\big)$。兩邊都是常態：$q$ 是 $\mathcal{N}(\tilde\mu,\tilde\beta_t I)$（KP04），$p_\theta$ 是 $\mathcal{N}(\mu_\theta,\sigma_t^2 I)$（KP03）。用上面的一維 KL 公式，每個維度各算一次再相加：

KP03 says training shrinks each step's KL: $\mathrm{KL}\big(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\,\Vert\,p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\big)$. Both sides are normal: $q$ is $\mathcal{N}(\tilde\mu,\tilde\beta_t I)$ (KP04) and $p_\theta$ is $\mathcal{N}(\mu_\theta,\sigma_t^2 I)$ (KP03). Using the one-dimensional KL formula above, compute it per dimension and add up:

$$
\mathrm{KL}=\frac{1}{2\sigma_t^2}\,\Vert\tilde\mu-\mu_\theta\Vert^2+C,
$$

$C$ 只和 $\tilde\beta_t$、$\sigma_t^2$ 有關，兩者都是事先定好的數，所以 $C$ 跟網路參數 $\theta$ 無關，訓練時可以不管。真正要縮小的只有兩個平均的距離。KP04 的式 (37) 與式 (38) 只差把 $\epsilon$ 換成 $\epsilon_\theta$，相減後 $\mathbf{x}_t$ 那一項消掉：

$C$ depends only on $\tilde\beta_t$ and $\sigma_t^2$, both fixed in advance, so $C$ does not depend on the network parameters $\theta$ and can be ignored in training. What really has to shrink is only the distance between the two means. KP04's equations (37) and (38) differ only by replacing $\epsilon$ with $\epsilon_\theta$, so on subtracting, the $\mathbf{x}_t$ term cancels:

$$
\tilde\mu-\mu_\theta=\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\,\big(\epsilon_\theta(\mathbf{x}_t,t)-\epsilon\big).
$$

代回去，第 $t$ 步的 KL（去掉常數）就是一個**加權的雜訊平方誤差**：

Substituting back, the KL of step $t$ (dropping the constant) is a **weighted squared error of the noise**:

$$
\mathrm{KL}_t=w_t\,\Vert\epsilon-\epsilon_\theta(\mathbf{x}_t,t)\Vert^2,\qquad w_t=\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)} .
$$

用來源互動展示的數字算一次 `[補充]`：$\epsilon=0.5$、$\epsilon_\theta=0.2$、$\beta_t=0.2$、$\bar\alpha_t=0.5$，取 $\sigma_t^2=\beta_t$。平均差的係數是 $0.2/(\sqrt{0.8}\sqrt{0.5})\approx0.316$，平均差 $0.316\times0.3\approx0.095$，KL $=0.095^2/(2\times0.2)\approx0.0225$。另一種算法：$w_t=0.2/(2\times0.8\times0.5)=0.25$，乘上雜訊平方誤差 $0.3^2=0.09$，也是 $0.0225$。

Compute once with the numbers of the source's interactive demo `[Added]`: $\epsilon=0.5$, $\epsilon_\theta=0.2$, $\beta_t=0.2$ and $\bar\alpha_t=0.5$, taking $\sigma_t^2=\beta_t$. The coefficient of the mean difference is $0.2/(\sqrt{0.8}\sqrt{0.5})\approx0.316$, the mean difference is $0.316\times0.3\approx0.095$, and the KL is $0.095^2/(2\times0.2)\approx0.0225$. The other way: $w_t=0.2/(2\times0.8\times0.5)=0.25$, times the squared noise error $0.3^2=0.09$, also gives $0.0225$.

常見錯誤：以為要讓 KL 等於 $0$，網路的變異數也得學。DDPM 的 $\sigma_t^2$ 是事先選的，變異數那部分只是常數 $C$；網路只負責平均，也就是只負責猜雜訊。讓網路也學變異數是本週 KP06 的改良。

Common mistake: thinking the network must also learn the variance to drive the KL to $0$. DDPM's $\sigma_t^2$ is chosen in advance, and the variance part is just the constant $C$; the network is responsible only for the mean, that is, only for guessing the noise. Letting the network learn the variance too is an improvement in this week's KP06.

### 2. 把權重拿掉：DDPM 的簡化損失（Dropping the weights: DDPM's simplified loss）

完整的 ELBO（除了跟 $\theta$ 無關的起點項）是把所有步的 $\mathrm{KL}_t$ 加起來，每一步有自己的權重 $w_t$；$t=1$ 那一項其實是 KP03 的重建項，這裡把它也當成一個 $\sigma_1^2=\beta_1$ 的 KL 來比較。DDPM 的作者 Ho 等人發現，**把權重全部換成 $1$**，生成的樣本反而更好。這就是來源的式 (41)，DDPM 的簡化損失：

The full ELBO (apart from the start term, which does not involve $\theta$) adds up the $\mathrm{KL}_t$ of all steps, each with its own weight $w_t$; the $t=1$ term is really KP03's reconstruction term, treated here as a KL with $\sigma_1^2=\beta_1$ for comparison. Ho and colleagues, the authors of DDPM, found that **replacing all the weights by $1$** actually gives better samples. This is the source's equation (41), DDPM's simplified loss:

$$
L_{\text{simple}}=\mathbb{E}_{t,\ \mathbf{x}_0,\ \epsilon}\Big[\big\Vert\epsilon-\epsilon_\theta\big(\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\ t\big)\big\Vert^2\Big],\qquad t\sim\mathrm{Uniform}\{1,\ldots,T\}.
$$

括號裡把 $\mathbf{x}_t$ 直接寫成 KP02 的閉式。整條損失就是一句話：**隨便挑一步、隨便加雜訊，請網路猜出加了什麼雜訊，猜錯的平方就是損失。**

Inside the brackets $\mathbf{x}_t$ is written directly as KP02's closed form. The whole loss is one sentence: **pick a random step, add random noise, ask the network to guess the noise that was added, and the square of the miss is the loss.**

![一張雙對數座標的折線圖，橫軸是步數 t 從 1 到 1000，縱軸是權重。紫色虛線是簡化損失的權重，在 1 水平。橘色實線是完整 ELBO 的權重 w_t 等於 beta_t 除以 2 alpha_t 乘 1 減 alpha bar t：在 t=1 約 0.5，一路往右下降，在 t 約三四百時降到最低約 0.005，到 t=1000 回升到約 0.01。圖例在右上角（A log-log line chart with step t from 1 to 1000 horizontally and the weight vertically. The dashed purple line, the simplified loss's weight, is flat at 1. The solid orange line, the full ELBO weight w_t = beta_t / (2 alpha_t (1 - alpha bar t)), is about 0.5 at t = 1, falls to the right, reaches its lowest value of about 0.005 around t = 300 to 400, and rises back to about 0.01 at t = 1000. The legend is at the upper right)](../assets/figures/lec14-kp05-weights.png)

**圖說 `lec14-kp05-weights`**〔功能：對照〕：DDPM 排程下、取 $\sigma_t^2=\beta_t$ 時（權重化簡成 $\frac{\beta_t}{2\alpha_t(1-\bar\alpha_t)}$），完整 ELBO 的權重（橘線）從 $t=1$ 的 $0.50$ 掉到 $t=500$ 的 $0.0055$，$t=1000$ 是 $0.0102$，頭尾差約 $49$ 倍。簡化損失（紫色虛線）一律給 $1$。和完整 ELBO 比起來，簡化損失相對**加重了雜訊多的步**，相對**減輕了雜訊很少的前幾步**。

**Caption `lec14-kp05-weights`** [Function: comparison]: under the DDPM schedule with $\sigma_t^2=\beta_t$ (the weight simplifies to $\frac{\beta_t}{2\alpha_t(1-\bar\alpha_t)}$), the full ELBO weight (orange) drops from $0.50$ at $t=1$ to $0.0055$ at $t=500$, and is $0.0102$ at $t=1000$, about $49$ times smaller at the end than at the start. The simplified loss (dashed purple) gives $1$ everywhere. Compared with the full ELBO, the simplified loss relatively **emphasises the noisy steps** and **de-emphasises the first few, nearly clean steps**.

為什麼這樣反而好 `[補充]`？雜訊很少的前幾步，只是在修影像上肉眼幾乎看不見的細節；雜訊多的步驟才決定影像的大結構。簡化損失把力氣放在比較難、對樣本品質影響比較大的步驟上。代價是它不再是 ELBO 本身，用它訓練出來的模型，算出的概似分數稍差一些；本週 KP06 會讓模型兼顧兩者。

Why is this better `[Added]`? The first few, nearly clean steps only fix details almost invisible to the eye; the noisy steps decide the large-scale structure of the image. The simplified loss puts its effort on the harder steps that matter more for sample quality. The price is that it is no longer the ELBO itself, and a model trained with it scores slightly worse in likelihood; this week's KP06 makes the model good at both.

常見錯誤：以為簡化損失只是「少算一些步」。它仍然用到每一步（$t$ 從 $1$ 到 $T$ 都會被抽到），只是每一步的權重一樣；$t=1$ 那一項也一起訓練，對應 KP03 的重建項。

Common mistake: thinking the simplified loss just "skips some steps". It still uses every step ($t$ from $1$ to $T$ all get drawn), only with equal weights; the $t=1$ term is trained along with the rest, corresponding to KP03's reconstruction term.

### 3. 演算法 1：訓練（Algorithm 1: training）

把簡化損失直接寫成程序，就是 DDPM 的演算法 1。重複以下五步，直到損失不再下降：

Written directly as a procedure, the simplified loss is DDPM's Algorithm 1. Repeat these five steps until the loss stops falling:

1. 從訓練資料取一筆 $\mathbf{x}_0$。
2. 從 $\{1,\ldots,T\}$ 均勻抽一個步數 $t$。
3. 抽一個標準常態雜訊 $\epsilon\sim\mathcal{N}(\mathbf{0},I)$。
4. 算 $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\epsilon$。
5. 對 $\Vert\epsilon-\epsilon_\theta(\mathbf{x}_t,t)\Vert^2$ 做梯度下降，更新 $\theta$。

1. Take an example $\mathbf{x}_0$ from the training data.
2. Draw a step $t$ uniformly from $\{1,\ldots,T\}$.
3. Draw standard-normal noise $\epsilon\sim\mathcal{N}(\mathbf{0},I)$.
4. Compute $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\epsilon$.
5. Take a gradient step on $\Vert\epsilon-\epsilon_\theta(\mathbf{x}_t,t)\Vert^2$ to update $\theta$.

實際上會一次取一批（batch）資料，每筆各抽自己的 $t$ 與 $\epsilon$，損失取平均。注意這裡完全不用走前向鏈：KP02 的閉式讓每筆資料只要抽一次雜訊。

In practice a batch of examples is taken at once, each with its own $t$ and $\epsilon$, and the loss is averaged. Note that the forward chain is never walked here: KP02's closed form needs only one noise draw per example.

**真的訓練一次** `[補充]`。資料用 KP01 的兩堆數字 $0.5\,\mathcal{N}(-2,0.3^2)+0.5\,\mathcal{N}(2,0.3^2)$，DDPM 排程（$T=1000$）。$\epsilon_\theta$ 是一個小的多層感知器：輸入 $x_t$、$t/T$ 與幾個 $t$ 的正弦與餘弦特徵（讓網路容易分辨步數），兩個各 $64$ 個神經元的隱藏層，輸出一個數。用 Adam 訓練 $12000$ 次，每批 $1024$ 筆。因為這份資料簡單，可以直接算出「最好的猜測」$E[\epsilon\mid\mathbf{x}_t]$（KP04 第4節），用它當損失能到達的下限。

**Train it for real** `[Added]`. The data are KP01's two piles $0.5\,\mathcal{N}(-2,0.3^2)+0.5\,\mathcal{N}(2,0.3^2)$, with the DDPM schedule ($T=1000$). $\epsilon_\theta$ is a small multilayer perceptron: its inputs are $x_t$, $t/T$ and a few sine and cosine features of $t$ (to help the network tell the steps apart), it has two hidden layers of $64$ neurons, and it outputs one number. It is trained with Adam for $12000$ iterations with batches of $1024$. Because the data are simple, the "best guess" $E[\epsilon\mid\mathbf{x}_t]$ can be computed directly (KP04 Section 4) and used as the lowest loss reachable.

![兩張並排的折線圖。左圖：橫軸是訓練次數 0 到 12000，縱軸是平均的雜訊平方誤差 0 到 1；紫線是訓練損失（每 100 次的移動平均），從約 0.47 很快掉到約 0.25，之後貼著一條綠色虛線走，綠色虛線標著「完美猜測的下限 0.245」。右圖：橫軸是步數 t（對數座標，1 到 1000），縱軸是該步的損失；紫線是訓練好的網路，綠色虛線是完美猜測，兩條幾乎重合：從 t=1 的約 1.0 開始，在 t 約 10 之後下降，在 t 約 180 降到約 0.23，接著在 t 約 350 回升到約 0.50 的小山，最後在 t=1000 降到接近 0（Two line charts side by side. Left: training iteration from 0 to 12000 horizontally and the average squared noise error from 0 to 1 vertically; the purple training loss (moving average over 100 iterations) drops quickly from about 0.47 to about 0.25 and then runs along a dashed green line labelled "floor with perfect guess: 0.245". Right: step t (log scale, 1 to 1000) horizontally and the loss at that step vertically; the purple trained network and the dashed green perfect guess almost coincide: starting near 1.0 at t = 1, they fall after about t = 10, reach about 0.23 near t = 180, rise to a small hill of about 0.50 near t = 350, and finally fall to nearly 0 at t = 1000)](../assets/figures/lec14-kp05-training.png)

**圖說 `lec14-kp05-training`**〔功能：精準度、一目了然〕：左圖是演算法 1 的訓練過程：前 $100$ 次的平均損失是 $0.47$，最後 $500$ 次是 $0.245$，幾乎就是綠線的下限 $0.245$（在同一批評估樣本上，網路只比完美猜測高 $0.0007$）。右圖把損失按步數拆開：網路（紫）和完美猜測（綠）幾乎重合，說明網路已經學到在這份資料上能學的東西；剩下的損失不是網路不夠好，而是雜訊本來就猜不準。

**Caption `lec14-kp05-training`** [Function: precision, seeing it at a glance]: the left chart is Algorithm 1's training run: the average loss is $0.47$ over the first $100$ iterations and $0.245$ over the last $500$, essentially at the green floor of $0.245$ (on the same evaluation samples, the network is only $0.0007$ above the perfect guess). The right chart splits the loss by step: the network (purple) and the perfect guess (green) almost coincide, showing that the network has learned what can be learned on these data; the remaining loss is not the network's fault but the noise being inherently hard to guess.

為什麼損失降不到 $0$？給定 $\mathbf{x}_t$，雜訊 $\epsilon$ 有多難猜，取決於 $\mathbf{x}_0$ 還有多不確定。從閉式看，$\epsilon=(\mathbf{x}_t-\sqrt{\bar\alpha_t}\,\mathbf{x}_0)/\sqrt{1-\bar\alpha_t}$，所以猜 $\epsilon$ 的誤差就是猜 $\mathbf{x}_0$ 的誤差乘上 $\sqrt{\bar\alpha_t}/\sqrt{1-\bar\alpha_t}$。右圖的形狀這樣讀：

Why can the loss not reach $0$? Given $\mathbf{x}_t$, how hard $\epsilon$ is to guess depends on how uncertain $\mathbf{x}_0$ still is. From the closed form, $\epsilon=(\mathbf{x}_t-\sqrt{\bar\alpha_t}\,\mathbf{x}_0)/\sqrt{1-\bar\alpha_t}$, so the error in guessing $\epsilon$ is the error in guessing $\mathbf{x}_0$ times $\sqrt{\bar\alpha_t}/\sqrt{1-\bar\alpha_t}$. The shape of the right chart reads as follows:

- **$t$ 很小**（$t=1$ 時 $0.99$）：雜訊只有一點點，遠小於每堆資料本身的寬度 $0.3$，從 $x_t$ 分不出哪部分是雜訊，幾乎只能猜 $0$。
- **$t$ 中等**（$t=183$ 降到最低 $0.227$）：雜訊已經比每堆的寬度大得多，$x_t$ 離所屬那堆中心的距離大部分是雜訊；這時兩堆的中心相距約 $1.7$、各自的寬度約 $0.6$，還分得開，看 $x_t$ 就知道來自哪一堆，雜訊相當好猜。
- **兩堆開始重疊**（$t=347$ 回升到 $0.50$）：兩堆的中心只相距約 $1.0$、寬度約 $0.9$，$x_t$ 落在中間時，分不出它來自左堆還是右堆，雜訊又變難猜。
- **$t$ 很大**（$t=1000$ 只剩 $0.0002$）：$x_t$ 幾乎就是 $\epsilon$ 本身，一猜就中。

- **Small $t$** ($0.99$ at $t=1$): the noise is tiny, far smaller than each pile's own width $0.3$, so you cannot tell which part of $x_t$ is noise and can only guess about $0$.
- **Medium $t$** (lowest, $0.227$, at $t=183$): the noise is now much larger than each pile's width, so most of the distance from $x_t$ to its pile's centre is noise; the piles' centres are about $1.7$ apart with widths of about $0.6$, still separate, so $x_t$ reveals which pile it came from, and the noise is fairly easy to guess.
- **Piles start to overlap** (back up to $0.50$ at $t=347$): the piles' centres are only about $1.0$ apart with widths of about $0.9$; when $x_t$ falls in the middle, you cannot tell whether it came from the left or right pile, and the noise becomes hard to guess again.
- **Large $t$** (only $0.0002$ at $t=1000$): $x_t$ is almost $\epsilon$ itself, so it is guessed right away.

常見錯誤：以為訓練好的擴散模型，損失應該接近 $0$；或是看到損失停在 $0.245$ 就以為還沒訓練好。損失的下限由資料與排程決定，要拿它和下限比，不是和 $0$ 比。

Common mistake: thinking a well-trained diffusion model should have a loss near $0$, or concluding that training is unfinished because the loss stops at $0.245$. The loss floor is set by the data and the schedule; compare the loss with that floor, not with $0$.

### 4. 演算法 2：生成（Algorithm 2: generation）

生成從純雜訊開始，用 KP04 的式 (38) 一步一步往回走：

Generation starts from pure noise and steps back one at a time with KP04's equation (38):

1. 抽 $\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},I)$。
2. 對 $t=T,T-1,\ldots,1$：
   - 若 $t>1$，抽新的雜訊 $\mathbf{z}\sim\mathcal{N}(\mathbf{0},I)$；若 $t=1$，令 $\mathbf{z}=\mathbf{0}$。
   - 算 $\mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}\Big(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(\mathbf{x}_t,t)\Big)+\sigma_t\,\mathbf{z}$。
3. 回傳 $\mathbf{x}_0$。

1. Draw $\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},I)$.
2. For $t=T,T-1,\ldots,1$:
   - if $t>1$, draw fresh noise $\mathbf{z}\sim\mathcal{N}(\mathbf{0},I)$; if $t=1$, set $\mathbf{z}=\mathbf{0}$.
   - compute $\mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}\Big(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(\mathbf{x}_t,t)\Big)+\sigma_t\,\mathbf{z}$.
3. Return $\mathbf{x}_0$.

括號裡就是逆向一步的平均 $\mu_\theta$；後面的 $\sigma_t\mathbf{z}$ 是從 $\mathcal{N}(\mu_\theta,\sigma_t^2 I)$ 抽樣。最後一步不加雜訊，因為 $\mathbf{x}_0$ 是要交出去的結果，再加雜訊只會把它弄髒。$\sigma_t^2$ 常用兩種選擇：$\beta_t$ 或 KP04 的 $\tilde\beta_t$；DDPM 論文發現兩者的樣本品質差不多。

The bracket is the backward-step mean $\mu_\theta$; the trailing $\sigma_t\mathbf{z}$ samples from $\mathcal{N}(\mu_\theta,\sigma_t^2 I)$. The last step adds no noise because $\mathbf{x}_0$ is the result to hand over, and more noise would only spoil it. Two choices of $\sigma_t^2$ are common: $\beta_t$ or KP04's $\tilde\beta_t$; the DDPM paper found that both give about the same sample quality.

用第3節訓練好的網路跑演算法 2，生成 $20000$ 個樣本 `[補充]`：

Run Algorithm 2 with the network trained in Section 3 to generate $20000$ samples `[Added]`:

![一張直方圖，橫軸 x0 從 -3.5 到 3.5。淡紫色直方圖是用訓練好的網路跑演算法 2 生成的 20000 個樣本，黑色曲線是真正的資料密度：兩座在 -2 與 2、高約 0.66 的山，中間幾乎是 0。直方圖和黑線幾乎重合，中間沒有樣本（A histogram with x0 from -3.5 to 3.5. The light-purple histogram is 20000 samples generated by Algorithm 2 with the trained network, and the black curve is the true data density: two peaks at -2 and 2, about 0.66 high, with almost nothing in between. The histogram almost coincides with the black curve, with no samples in the middle)](../assets/figures/lec14-kp05-samples.png)

**圖說 `lec14-kp05-samples`**〔功能：對照〕：淡紫色是生成的樣本，黑線是真正的資料。兩堆的位置、寬度和高度都對上了：$50.3\%$ 的樣本在右堆，右堆平均 $1.99$、標準差 $0.30$，左堆平均 $-2.02$、標準差 $0.30$。落在兩堆中間（$|x|<1$）的樣本只有 $0.15\%$；真正的資料是 $0.04\%$，用完美猜測生成是 $0.06\%$。

**Caption `lec14-kp05-samples`** [Function: comparison]: light purple is the generated samples and the black line the true data. The location, width and height of both piles match: $50.3\%$ of the samples are in the right pile, whose mean is $1.99$ with standard deviation $0.30$, and the left pile has mean $-2.02$ with standard deviation $0.30$. Only $0.15\%$ of the samples fall between the piles ($|x|<1$); the true data have $0.04\%$, and generating with the perfect guess gives $0.06\%$.

改用 $\sigma_t^2=\tilde\beta_t$ 生成，右堆標準差是 $0.297$，結果幾乎一樣。注意代價：每生成一個樣本，網路要算 $1000$ 次。這是 DDPM 最大的缺點，本週 KP06 介紹加速的方法。

Generating with $\sigma_t^2=\tilde\beta_t$ instead gives a right-pile standard deviation of $0.297$, almost the same. Note the cost: each generated sample needs $1000$ network evaluations. This is DDPM's biggest drawback, and this week's KP06 introduces ways to speed it up.

常見錯誤：把生成時加的 $\mathbf{z}$ 和訓練時的 $\epsilon$ 當成同一個東西。$\epsilon$ 是訓練時用來造 $\mathbf{x}_t$ 的雜訊，也是網路要猜的答案；$\mathbf{z}$ 是生成時每一步新抽的隨機性，讓同一個起點能走出不同的樣本（KP01 的軌跡圖）。

Common mistake: treating the $\mathbf{z}$ added in generation as the same thing as the $\epsilon$ of training. $\epsilon$ is the noise used in training to build $\mathbf{x}_t$, and it is the answer the network must guess; $\mathbf{z}$ is fresh randomness drawn at each generation step, which lets the same starting point lead to different samples (KP01's path figure).

### 5. 和 GAN 比一比（Compared with a GAN）

來源指出，DDPM 比 GAN 訓練穩定。原因看演算法 1 就知道：只有一個網路、一個固定的回歸目標，損失一路往下降，沒有第13週那種兩個網路互相追趕、可能震盪或模式崩塌的問題。從第4節的生成結果也看得到，兩堆都生成出來了，沒有漏掉任何一堆。代價在生成：GAN 只要算一次生成器，DDPM 要算 $T$ 次網路。

The source points out that DDPM trains more stably than a GAN. Algorithm 1 shows why: one network and one fixed regression target, with a loss that goes steadily down, and none of Week 13's two networks chasing each other, possibly oscillating or collapsing to a few modes. Section 4's results show it too: both piles were generated, none was missed. The price is in generation: a GAN computes its generator once, while DDPM computes its network $T$ times.

| 比較 | GAN（第13週） | DDPM |
|---|---|---|
| 要訓練的網路 | 生成器與判別器兩個 | 一個雜訊預測網路 |
| 訓練目標 | 極小化極大的賽局 | 雜訊的平方誤差（回歸） |
| 訓練穩定度 | 常震盪、可能模式崩塌 | 穩定，損失整體穩定下降到下限附近 |
| 生成一個樣本 | 算一次生成器 | 算 $T$ 次網路（DDPM 是 $1000$ 次） |

| Comparison | GAN (Week 13) | DDPM |
|---|---|---|
| Networks to train | two: generator and discriminator | one noise-prediction network |
| Training objective | a minimax game | squared error of the noise (regression) |
| Training stability | often oscillates, may collapse to few modes | stable; the loss falls steadily to near its floor |
| Generating one sample | one generator evaluation | $T$ network evaluations ($1000$ for DDPM) |

常見錯誤：以為 DDPM 在每一方面都勝過 GAN。來源的小測驗也提醒：DDPM 的優勢是訓練穩定、簡單，不是生成快。

Common mistake: thinking DDPM beats GAN in every respect. The source's quiz is a reminder too: DDPM's advantage is stable, simple training, not fast generation.

### 本單元重點整理（KP05 summary）

**一句話重點**：每一步的 KL 化簡成加權的雜訊平方誤差 $w_t\Vert\epsilon-\epsilon_\theta\Vert^2$，DDPM 把權重都換成 $1$ 得到 $L_{\text{simple}}$；訓練（演算法 1）是隨機挑步、加雜訊、請網路猜雜訊，生成（演算法 2）是從純雜訊出發，用 $\epsilon_\theta$ 一步一步扣掉雜訊、再加一點新雜訊，最後一步不加。

**One-sentence summary**: each step's KL simplifies to a weighted squared noise error $w_t\Vert\epsilon-\epsilon_\theta\Vert^2$, and DDPM replaces every weight by $1$ to get $L_{\text{simple}}$; training (Algorithm 1) picks a random step, adds noise and asks the network to guess it, while generation (Algorithm 2) starts from pure noise and, step by step, subtracts noise using $\epsilon_\theta$ and adds a little fresh noise, except at the last step.

**白話說明**：訓練像出考卷：老師拿一張乾淨的照片，隨便灑一層濃淡不一的雜訊，問學生「我灑了什麼？」，答錯多少扣多少。生成像把這個本事反覆用一千次：從一張全是雜訊的紙開始，每次請學生說出「這裡面有什麼雜訊」，擦掉一點，再輕輕抖一下，最後就擦出一張新照片。

**Plain explanation**: training is like setting an exam: the teacher takes a clean photo, sprinkles a random amount of noise on it, and asks the student "what did I sprinkle?", deducting marks for how wrong the answer is. Generation applies this skill a thousand times over: starting from a sheet of pure noise, each time the student says "what noise is in here", a little is wiped off and the sheet is given a gentle shake, until a new photo appears.

**正式定義**：$\mathrm{KL}_t=w_t\,\mathbb{E}\Vert\epsilon-\epsilon_\theta(\mathbf{x}_t,t)\Vert^2+C$，$w_t=\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$；$L_{\text{simple}}=\mathbb{E}_{t\sim\mathrm{Uniform}\{1,\ldots,T\},\mathbf{x}_0,\epsilon}\Vert\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\epsilon,t)\Vert^2$。生成：$\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},I)$，$\mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}\big(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\mathbf{x}_t,t)\big)+\sigma_t\mathbf{z}$，$\mathbf{z}\sim\mathcal{N}(\mathbf{0},I)$（$t=1$ 時 $\mathbf{z}=\mathbf{0}$），$\sigma_t^2\in\{\beta_t,\tilde\beta_t\}$。

**Formal definition**: $\mathrm{KL}_t=w_t\,\mathbb{E}\Vert\epsilon-\epsilon_\theta(\mathbf{x}_t,t)\Vert^2+C$ with $w_t=\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$; $L_{\text{simple}}=\mathbb{E}_{t\sim\mathrm{Uniform}\{1,\ldots,T\},\mathbf{x}_0,\epsilon}\Vert\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\epsilon,t)\Vert^2$. Generation: $\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},I)$, $\mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}\big(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\mathbf{x}_t,t)\big)+\sigma_t\mathbf{z}$ with $\mathbf{z}\sim\mathcal{N}(\mathbf{0},I)$ ($\mathbf{z}=\mathbf{0}$ at $t=1$) and $\sigma_t^2\in\{\beta_t,\tilde\beta_t\}$.

**例子**：$\epsilon=0.5$、$\epsilon_\theta=0.2$、$\beta_t=0.2$、$\bar\alpha_t=0.5$、$\sigma_t^2=\beta_t$：$w_t=0.25$，KL $=0.25\times0.09=0.0225$。一維小 DDPM：損失從 $0.47$ 降到 $0.245$（下限 $0.245$），生成 $20000$ 個樣本，$50.3\%$ 在右堆，兩堆平均 $1.99$、$-2.02$，標準差都是 $0.30$。

**Example**: $\epsilon=0.5$, $\epsilon_\theta=0.2$, $\beta_t=0.2$, $\bar\alpha_t=0.5$, $\sigma_t^2=\beta_t$: $w_t=0.25$ and KL $=0.25\times0.09=0.0225$. The one-dimensional DDPM: the loss falls from $0.47$ to $0.245$ (floor $0.245$); of $20000$ generated samples, $50.3\%$ are in the right pile, the piles have means $1.99$ and $-2.02$, and both have standard deviation $0.30$.

**常見迷思**：

- 「網路也要學變異數，KL 才會變小。」DDPM 的 $\sigma_t^2$ 固定，網路只猜雜訊（學變異數是 KP06 的改良）。
- 「簡化損失少算了一些步。」錯，每一步都會被抽到，只是權重相同。
- 「損失應該降到 $0$。」錯，下限由資料與排程決定。
- 「生成的 $\mathbf{z}$ 就是訓練的 $\epsilon$。」錯，兩者是不同的雜訊。
- 「DDPM 每方面都勝過 GAN。」錯，DDPM 生成慢得多。

**Common misconceptions**:

- "The network must learn the variance too for the KL to shrink." DDPM's $\sigma_t^2$ is fixed and the network only guesses the noise (learning the variance is KP06's improvement).
- "The simplified loss skips some steps." Wrong: every step gets drawn, only with equal weights.
- "The loss should fall to $0$." Wrong: its floor is set by the data and the schedule.
- "The generation noise $\mathbf{z}$ is the training noise $\epsilon$." Wrong: they are different noises.
- "DDPM beats GAN in every respect." Wrong: DDPM generates much more slowly.

**圖**：本單元有三張圖。

- `lec14-kp05-weights`：完整 ELBO 的每步權重與簡化損失的權重 $1$。
- `lec14-kp05-training`：一維小 DDPM 的訓練曲線與各步損失，和完美猜測的下限比較。
- `lec14-kp05-samples`：演算法 2 生成的樣本與真正的資料密度。

**Figures** — this unit has three figures.

- `lec14-kp05-weights`: the full ELBO's per-step weights versus the simplified loss's weight $1$.
- `lec14-kp05-training`: the training curve and per-step loss of the one-dimensional DDPM, compared with the perfect-guess floor.
- `lec14-kp05-samples`: samples generated by Algorithm 2 against the true data density.

### 小練習（Practice）

1. 設 $\epsilon=-1$、$\epsilon_\theta=-0.4$、$\beta_t=0.1$、$\bar\alpha_t=0.6$，取 $\sigma_t^2=\beta_t$。算平均差、KL、權重 $w_t$ 與簡化損失，並驗算「KL $=w_t\times$ 簡化損失」。
2. 手算演算法 2 的一步：$x_t=0.5$、$\beta_t=0.2$、$\bar\alpha_t=0.5$、網路猜 $\epsilon_\theta=0.3$、$\sigma_t^2=\beta_t$，抽到 $z=-1$。$x_{t-1}$ 是多少？
3. 演算法 1 每次隨機抽一個 $t$。為什麼不對每筆資料把 $t=1$ 到 $1000$ 全部算一遍？
4. 第3節的小 DDPM 中，$t=1$ 的損失約 $1$，$t=1000$ 的損失約 $0$。用「$\epsilon$ 有多難從 $x_t$ 猜出來」解釋這兩端。

1. Let $\epsilon=-1$, $\epsilon_\theta=-0.4$, $\beta_t=0.1$ and $\bar\alpha_t=0.6$, with $\sigma_t^2=\beta_t$. Compute the mean difference, the KL, the weight $w_t$ and the simplified loss, and check that "KL $=w_t\times$ simplified loss".
2. Compute one step of Algorithm 2 by hand: $x_t=0.5$, $\beta_t=0.2$, $\bar\alpha_t=0.5$, the network guesses $\epsilon_\theta=0.3$, $\sigma_t^2=\beta_t$, and the draw is $z=-1$. What is $x_{t-1}$?
3. Algorithm 1 draws one random $t$ each time. Why not compute all of $t=1$ to $1000$ for every example?
4. In Section 3's small DDPM, the loss is about $1$ at $t=1$ and about $0$ at $t=1000$. Explain both ends in terms of "how hard $\epsilon$ is to guess from $x_t$".

**第 1 題解答** `[補充]`：$\alpha_t=0.9$。平均差的係數 $0.1/(\sqrt{0.9}\sqrt{0.4})\approx0.1667$，平均差 $\tilde\mu-\mu_\theta=0.1667\times(-0.4-(-1))=0.1$。KL $=0.1^2/(2\times0.1)=0.05$。$w_t=0.1/(2\times0.9\times0.4)\approx0.1389$，簡化損失 $(-1+0.4)^2=0.36$，$0.1389\times0.36=0.05$，一致。

**Solution to question 1** `[Added]`: $\alpha_t=0.9$. The coefficient of the mean difference is $0.1/(\sqrt{0.9}\sqrt{0.4})\approx0.1667$, and the mean difference is $\tilde\mu-\mu_\theta=0.1667\times(-0.4+1)=0.1$. The KL is $0.1^2/(2\times0.1)=0.05$. $w_t=0.1/(2\times0.9\times0.4)\approx0.1389$, the simplified loss is $(-1+0.4)^2=0.36$, and $0.1389\times0.36=0.05$, which agrees.

**第 2 題解答** `[補充]`：雜訊的係數 $0.2/\sqrt{0.5}\approx0.2828$，平均 $=(0.5-0.2828\times0.3)/\sqrt{0.8}=(0.5-0.0849)/0.8944\approx0.4641$。$\sigma_t=\sqrt{0.2}\approx0.4472$，所以 $x_{t-1}=0.4641+0.4472\times(-1)\approx0.0169$。

**Solution to question 2** `[Added]`: the noise coefficient is $0.2/\sqrt{0.5}\approx0.2828$, and the mean is $(0.5-0.2828\times0.3)/\sqrt{0.8}=(0.5-0.0849)/0.8944\approx0.4641$. $\sigma_t=\sqrt{0.2}\approx0.4472$, so $x_{t-1}=0.4641+0.4472\times(-1)\approx0.0169$.

**第 3 題解答** `[補充]`：簡化損失是對 $t$ 取平均；隨機抽一個 $t$ 得到的是這個平均的不偏估計，計算量只有全部算的 $1/1000$。用同樣的計算量，可以看到 $1000$ 倍多的不同資料與雜訊，對學習更有幫助。這和第3週「用小批次的隨機梯度代替全部資料的梯度」是同一個道理。

**Solution to question 3** `[Added]`: the simplified loss is an average over $t$; drawing one random $t$ gives an unbiased estimate of that average at $1/1000$ of the cost. With the same amount of computation you see $1000$ times more different data and noise, which helps learning more. It is the same idea as Week 3's "use a mini-batch stochastic gradient instead of the full-data gradient".

**第 4 題解答** `[補充]`：$t=1$ 時加的雜訊標準差只有 $0.01$，遠小於資料本身的寬度 $0.3$；從 $x_1$ 看不出哪部分是雜訊，最好的猜法幾乎是猜 $0$，誤差平方的平均就約是 $\epsilon$ 的變異數 $1$。$t=1000$ 時 $x_{1000}\approx\epsilon$（原資料只剩 $0.0064$ 倍），看 $x_{1000}$ 就幾乎知道 $\epsilon$，誤差接近 $0$。

**Solution to question 4** `[Added]`: at $t=1$ the added noise has standard deviation only $0.01$, far smaller than the data's own width $0.3$; from $x_1$ you cannot tell which part is noise, so the best guess is close to $0$, and the mean squared error is about the variance of $\epsilon$, which is $1$. At $t=1000$, $x_{1000}\approx\epsilon$ (only $0.0064$ of the original remains), so seeing $x_{1000}$ almost tells you $\epsilon$, and the error is near $0$.

### 小測驗（Quiz，來源原題）

來源 KP-8 與 KP-9 各有四題自我檢測，保留原題與全部選項；來源頁面附有答案與簡短解析，下面的解析由本講義依各節內容補寫。

The source's KP-8 and KP-9 each have four self-check questions, kept here with all their options; the source page gives the answers with short explanations, and the explanations below are written by this note from its own sections.

1. 兩個高斯分布的 KL 在 covariance 相等時簡化成什麼？
   - (A) mean 的 Mahalanobis 距離
   - (B) covariance 的差異
   - (C) entropy 的差
   - (D) 0
2. DDPM 最常用的簡化 loss 是什麼？
   - (A) $\Vert\epsilon-\epsilon_\theta\Vert^2$
   - (B) $\Vert x_0-x_\theta\Vert^2$
   - (C) $\Vert\mu-\mu_\theta\Vert^2$
   - (D) cross-entropy
3. 式 (40) 中的權重項在式 (41) 中被怎麼處理？
   - (A) 忽略不計
   - (B) 放大 100 倍
   - (C) 換成常數 1
   - (D) 改成指數
4. 在式 (41) 中，$x_t$ 是怎麼產生的？
   - (A) $x_t=x_0+\epsilon$
   - (B) $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$
   - (C) $x_t=\sqrt{\alpha_t}x_{t-1}+\epsilon$
   - (D) $x_t=\epsilon$
5. Algorithm 1 中，$t$ 是怎麼選的？
   - (A) 固定為 $T$
   - (B) 均勻抽樣自 $\{1,\ldots,T\}$
   - (C) 從資料集取
   - (D) 由網路決定
6. 推論時的起點是什麼？
   - (A) 真實資料 $x_0$
   - (B) 標準高斯雜訊 $x_T$
   - (C) 網路權重
   - (D) $\beta_t$ 排程
7. 訓練時網路學習的目標是什麼？
   - (A) 預測 $x_0$
   - (B) 預測加入的雜訊 $\epsilon$
   - (C) 預測 $\beta_t$
   - (D) 預測 $q(x_T)$
8. DDPM 相較於 GAN 的主要優勢是什麼？
   - (A) 推論速度更快
   - (B) 訓練更穩定、簡單
   - (C) 參數更少
   - (D) 不需要 noise schedule

1. What does the KL between two Gaussians reduce to when their covariances are equal?
   - (A) The Mahalanobis distance between the means
   - (B) The difference of the covariances
   - (C) The difference of the entropies
   - (D) 0
2. What is DDPM's most common simplified loss?
   - (A) $\Vert\epsilon-\epsilon_\theta\Vert^2$
   - (B) $\Vert x_0-x_\theta\Vert^2$
   - (C) $\Vert\mu-\mu_\theta\Vert^2$
   - (D) Cross-entropy
3. How is the weight in equation (40) treated in equation (41)?
   - (A) It is ignored
   - (B) It is multiplied by 100
   - (C) It is replaced by the constant 1
   - (D) It is turned into an exponential
4. In equation (41), how is $x_t$ produced?
   - (A) $x_t=x_0+\epsilon$
   - (B) $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$
   - (C) $x_t=\sqrt{\alpha_t}x_{t-1}+\epsilon$
   - (D) $x_t=\epsilon$
5. In Algorithm 1, how is $t$ chosen?
   - (A) Fixed at $T$
   - (B) Drawn uniformly from $\{1,\ldots,T\}$
   - (C) Taken from the dataset
   - (D) Decided by the network
6. What is the starting point of inference?
   - (A) The real data $x_0$
   - (B) Standard Gaussian noise $x_T$
   - (C) The network weights
   - (D) The $\beta_t$ schedule
7. What does the network learn during training?
   - (A) To predict $x_0$
   - (B) To predict the added noise $\epsilon$
   - (C) To predict $\beta_t$
   - (D) To predict $q(x_T)$
8. What is DDPM's main advantage over GANs?
   - (A) Faster inference
   - (B) More stable, simpler training
   - (C) Fewer parameters
   - (D) No noise schedule needed

**解析** `[補充]`：

- 第 1 題選 (A)：變異數相同時只剩平均距離的平方除以變異數，也就是「以變異數為尺度量的距離」$(m_1-m_2)^2/v$，叫做馬氏距離（Mahalanobis distance）的平方（再乘 $\tfrac12$）。第1節的兩個變異數其實不同（$\tilde\beta_t$ 與 $\sigma_t^2$），差別只是多了常數 $C$。
- 第 2 題選 (A)：第2節的 $L_{\text{simple}}$。(B)、(C) 是等價寫法下的其他目標（KP04），但不是 DDPM 用的那一個。
- 第 3 題來源答案是 (A)：第2節。說「忽略」和說「換成常數 $1$」意思相同，所以 (C) 其實也對；重點是每一步的權重都一樣。
- 第 4 題選 (B)：KP02 的閉式，所以不必走前向鏈。(C) 少了 $\sqrt{\beta_t}$，也只是單步。
- 第 5 題選 (B)：第3節演算法 1 的第 2 步，見小練習第 3 題。
- 第 6 題選 (B)：第4節演算法 2 的第 1 步。
- 第 7 題選 (B)：第3節。
- 第 8 題選 (B)：第5節；(A) 剛好相反，DDPM 生成要算 $T$ 次網路。

**Explanations** `[Added]`:

- Question 1: A: with equal variances only the squared distance between the means divided by the variance remains, "a distance measured in units of the variance", $(m_1-m_2)^2/v$, called (half) the squared Mahalanobis distance. In Section 1 the two variances actually differ ($\tilde\beta_t$ and $\sigma_t^2$), which only adds the constant $C$.
- Question 2: A: Section 2's $L_{\text{simple}}$. B and C are other targets in equivalent forms (KP04), but not the one DDPM uses.
- Question 3: the source's answer is A: Section 2. "Ignored" and "replaced by the constant $1$" mean the same thing, so C is also right; the point is that every step has the same weight.
- Question 4: B: KP02's closed form, so the forward chain need not be walked. C lacks $\sqrt{\beta_t}$ and is only a single step.
- Question 5: B: step 2 of Algorithm 1 in Section 3; see practice question 3.
- Question 6: B: step 1 of Algorithm 2 in Section 4.
- Question 7: B: Section 3.
- Question 8: B: Section 5; A is backwards, since DDPM generation needs $T$ network evaluations.

### 留給下一單元的問題（The question for the next unit）

DDPM 已經能用了，但有幾個明顯的地方可以改：$\sigma_t^2$ 只能在 $\beta_t$ 與 $\tilde\beta_t$ 之間挑一個，能不能讓網路自己學？線性排程讓後面很多步幾乎全是雜訊，是不是浪費？生成要算 $1000$ 次網路，能不能少走幾步？下一個知識點（本週 KP06）介紹 DDPM 的改良：學變異數、cosine 排程、PriorGrad，以及 DDIM 等加速方法。

DDPM works, but there are clear places to improve it: $\sigma_t^2$ can only be one of $\beta_t$ and $\tilde\beta_t$; could the network learn it? The linear schedule makes many late steps almost pure noise; is that wasteful? Generation needs $1000$ network evaluations; can it take fewer steps? The next knowledge point (this week's KP06) introduces DDPM's improvements: learned variance, the cosine schedule, PriorGrad, and speed-ups such as DDIM.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第15章「Diffusion Models」的 KP-8「Loss Function：從高斯 KL 到 $\epsilon$-loss」與 KP-9「DDPM 的訓練與推論：Algorithm 1 與 Algorithm 2」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch15-diffusion-models/index.html ，讀取日期 2026-11-15）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from KP-8, "Loss function: from Gaussian KL to the $\epsilon$-loss", and KP-9, "DDPM training and inference: Algorithm 1 and Algorithm 2", of Chapter 15, "Diffusion Models", of CCC-LNS IDL_115 (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch15-diffusion-models/index.html , accessed 2026-11-15). Content not in the source is marked `[Added]` where it appears.

- 來源的式 (40) 把權重寫成 $\frac{(1-\alpha_t)^2}{2\alpha_t(1-\bar\alpha_t)\Vert\Sigma_\theta\Vert^2}$。分母應該是逆向變異數 $\sigma_t^2$ 本身，不是它的平方（也不是矩陣的範數平方）：正確的權重是 $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$。用來源展示的數字（$\sigma_t^2=\beta_t=0.2$），正確權重是 $0.25$，KL $0.0225$；照來源寫法會差一個 $1/\sigma_t^2=5$ 倍。
- 來源說「若 $\Sigma_\theta=\tilde\Sigma=\sigma_t^2 I$，只剩 mean 的距離」。其實不必兩者相等：只要 $\sigma_t^2$ 是事先選定的，變異數那部分就是與 $\theta$ 無關的常數；DDPM 常取 $\sigma_t^2=\beta_t$，它並不等於 $\tilde\beta_t$。本講義第1節照此寫。
- 來源的演算法 2 在每一步都加 $\sigma_t\epsilon$，而且和網路的 $\epsilon_\theta$、訓練的 $\epsilon$ 用同一個符號。DDPM 原論文在最後一步（$t=1$）不加雜訊，生成時的新雜訊也另記為 $\mathbf{z}$；本講義第4節照原論文寫。
- 來源小測驗第 3 題以 (A)「忽略不計」為答案；(C)「換成常數 1」意思相同，解析中說明。
- 來源說簡化損失「效果更好」，沒有說好在哪裡、代價是什麼；本講義第2節補上：樣本品質較好，但它不再是 ELBO，概似分數稍差。
- 本講義補上：一維常態 KL 公式的使用、平均差的推導、權重的一般式與數值、兩種算法的手算核對、權重隨步數變化的圖、numpy 手寫的一維小 DDPM 實際訓練與生成（含完美猜測的損失下限、各步損失的形狀與解釋）、最後一步不加雜訊與 $\sigma_t^2$ 兩種選擇的比較、$\mathbf{z}$ 與 $\epsilon$ 的區別、GAN 與 DDPM 對照表、四題小練習與解答、小測驗的解析。
- 取捨：來源兩個互動展示不重做，改用實際訓練的圖；虛擬碼改寫成條列步驟；符號加粗與否的取捨同 KP02。

- The source's equation (40) writes the weight as $\frac{(1-\alpha_t)^2}{2\alpha_t(1-\bar\alpha_t)\Vert\Sigma_\theta\Vert^2}$. The denominator should contain the backward variance $\sigma_t^2$ itself, not its square (nor a squared matrix norm): the correct weight is $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$. With the source demo's numbers ($\sigma_t^2=\beta_t=0.2$), the correct weight is $0.25$ and the KL is $0.0225$; the source's form would be off by a factor $1/\sigma_t^2=5$.
- The source says "if $\Sigma_\theta=\tilde\Sigma=\sigma_t^2 I$, only the distance between the means remains". The two need not be equal: as long as $\sigma_t^2$ is chosen in advance, the variance part is a constant independent of $\theta$; DDPM often takes $\sigma_t^2=\beta_t$, which is not equal to $\tilde\beta_t$. Section 1 follows this.
- The source's Algorithm 2 adds $\sigma_t\epsilon$ at every step and uses the same symbol as the network's $\epsilon_\theta$ and the training $\epsilon$. The original DDPM paper adds no noise at the last step ($t=1$) and writes the fresh generation noise as $\mathbf{z}$; Section 4 follows the original paper.
- The source's quiz question 3 takes A, "ignored", as the answer; C, "replaced by the constant 1", means the same thing, which the explanation notes.
- The source says the simplified loss "works better" without saying in what way or at what cost; Section 2 adds that sample quality is better but it is no longer the ELBO, so the likelihood score is slightly worse.
- This note adds: the use of the one-dimensional normal KL formula; the derivation of the mean difference; the general form and values of the weight; a hand check by two routes; the figure of the weight against the step; an actual numpy-written one-dimensional DDPM trained and sampled (with the perfect-guess loss floor and the shape and explanation of the per-step loss); no noise at the last step and the comparison of the two choices of $\sigma_t^2$; the difference between $\mathbf{z}$ and $\epsilon$; the GAN–DDPM table; four practice questions with solutions; and the explanations of the quiz.
- Choices: the source's two interactive demos are not rebuilt; the actual training figures are used instead. The pseudocode is rewritten as numbered steps. Bold versus plain symbols follow the choice made in KP02.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 簡化損失 | simplified loss |
| 加權的雜訊平方誤差 | weighted squared error of the noise |
| 均方誤差 | mean squared error (MSE) |
| 馬氏距離 | Mahalanobis distance |
| 交叉熵 | cross-entropy |
| 演算法 1（訓練） | Algorithm 1 (training) |
| 演算法 2（生成、取樣） | Algorithm 2 (generation, sampling) |
| 均勻抽樣 | uniform sampling |
| 小批次 | mini-batch |
| 不偏估計 | unbiased estimate |
| 多層感知器 | multilayer perceptron (MLP) |
| 正弦與餘弦特徵 | sine and cosine features |
| 損失下限 | loss floor |
| 概似 | likelihood |
| 證據下界 | evidence lower bound (ELBO) |
| KL 散度 | Kullback–Leibler (KL) divergence |
| 去雜訊擴散機率模型 | denoising diffusion probabilistic model (DDPM) |
| 生成對抗網路（第13週） | generative adversarial network (GAN) |
| 模式崩塌 | mode collapse |
| 極小化極大賽局 | minimax game |
| 熵 | entropy |
| 去雜訊擴散隱式模型（KP06） | denoising diffusion implicit model (DDIM) |
