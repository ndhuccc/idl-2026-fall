# 第14週 KP01：擴散模型導論，先攪散再還原（Introduction to diffusion models: scramble first, then restore）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第14週「深度生成模型（三）：擴散模型」的第一個知識點（KP01）。前兩週學了兩種生成模型：第12週的 VAE 訓練穩定，但影像偏模糊；第13週的 GAN 影像清楚，但訓練是一場不容易收斂的對抗。這一節介紹第三種想法：先用一個固定的程序把資料一點一點攪成雜訊，再訓練一個網路學會把雜訊一點一點還原成資料。

This is the first knowledge point (KP01) of Week 14, "Deep generative models (3): diffusion models". The past two weeks covered two generative models: the VAE of Week 12 trains steadily but gives blurry images, and the GAN of Week 13 gives sharp images but trains through an adversarial game that is hard to bring to rest. This note introduces a third idea: use a fixed procedure to stir the data into noise little by little, then train a network to turn noise back into data little by little.

先備知識是第12週的潛在變數、編碼器與解碼器、VAE 的模糊問題（KP07），以及第13週 GAN 的生成器與訓練不穩定（KP04、KP06）。常態分布 $\mathcal{N}(\mu,s^2)$ 與「平均、標準差」要熟。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第15章 KP-1 改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec14-kp01-figures.py` 算出，可以自己重算。

The prerequisites are, from Week 12, latent variables, the encoder and the decoder, and the VAE's blur problem (KP07); from Week 13, the GAN's generator and its unstable training (KP04, KP06). You should be comfortable with the normal distribution $\mathcal{N}(\mu,s^2)$ and with "mean" and "standard deviation". Every term and symbol is defined when it first appears. This note is rewritten from KP-1 of Chapter 15 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec14-kp01-figures.py`, so you can recompute it yourself.

## KP01 擴散模型導論，先攪散再還原（Introduction to diffusion models: scramble first, then restore）

### 學習目標（Learning objectives）

讀完之後，你能用一句話說出擴散模型的核心精神：先加雜訊，再學去雜訊。你能說出前向過程與逆向過程各自的起點、終點，以及哪一個要學、哪一個不用學。你能用小數字手算三步加雜訊，並說明為什麼每一步要先把舊值縮小。你能從一個一維例子的圖讀出：前向過程把資料變成標準常態，逆向過程從標準常態變回資料的形狀。你也能用「自編碼器」的眼光比較擴散模型與 VAE，說出兩者最主要的差別。

After reading, you can state the core idea of a diffusion model in one sentence: add noise first, then learn to remove it. You can name the start and end of the forward process and of the backward process, and say which one is learned and which one is not. You can compute three noising steps by hand with small numbers and explain why each step first shrinks the old value. From the figures of a one-dimensional example you can read that the forward process turns the data into a standard normal, and that the backward process turns a standard normal back into the shape of the data. You can also compare a diffusion model with a VAE through the lens of an "autoencoder" and state the main difference between them.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{x}_0$ | 一筆真實資料，例如一張圖片 | one real example, such as an image |
| $\mathbf{x}_t$ | 加了 $t$ 步雜訊之後的資料，大小和 $\mathbf{x}_0$ 一樣 | the data after $t$ noising steps, the same size as $\mathbf{x}_0$ |
| $T$ | 總步數；$\mathbf{x}_T$ 幾乎是純雜訊 | the total number of steps; $\mathbf{x}_T$ is almost pure noise |
| $t$ | 第幾步，$t=1,2,\ldots,T$ | the step number, $t=1,2,\ldots,T$ |
| $\beta_t$ | 第 $t$ 步加入的雜訊量，介於 $0$ 與 $1$ 之間的小數 | the amount of noise added at step $t$, a small number between $0$ and $1$ |
| $\epsilon$ | 從標準常態抽出的雜訊，和資料一樣大 | noise drawn from the standard normal, the same size as the data |
| $\mathcal{N}(\mu,s^2)$ | 平均 $\mu$、標準差 $s$ 的常態分布 | normal distribution with mean $\mu$ and standard deviation $s$ |
| $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ | 前向過程的一步：固定的加雜訊規則 | one step of the forward process: the fixed noising rule |
| $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ | 逆向過程的一步：要學的去雜訊網路，參數是 $\theta$ | one step of the backward process: the denoising network to be learned, with parameters $\theta$ |
| $\bar\alpha_t$ | $\bar\alpha_t=(1-\beta_1)(1-\beta_2)\cdots(1-\beta_t)$；$\sqrt{\bar\alpha_t}$ 是原本的值走了 $t$ 步後還剩幾倍（本週 KP02 詳談） | $\bar\alpha_t=(1-\beta_1)(1-\beta_2)\cdots(1-\beta_t)$; $\sqrt{\bar\alpha_t}$ is the fraction of the original value left after $t$ steps (detailed in this week's KP02) |
| $q_\phi(\mathbf{z}\mid\mathbf{x})$ | VAE 的編碼器，參數是 $\phi$（第12週） | the VAE's encoder, with parameters $\phi$ (Week 12) |
| $\mathbf{z}$ | VAE 的潛在變數（第12週） | the VAE's latent variable (Week 12) |

### 1. 為什麼還要第三種生成模型（Why a third generative model）

**生成模型**（generative model）要學會真實資料的分布，再從中抽出新的樣本。前兩週的兩種做法各有一個麻煩。第12週的 VAE 用平方誤差評分，拿不準時偏好平均，影像偏模糊（第12週 KP07）。第13週的 GAN 請判別器當裁判，影像清楚，但訓練是在找兩個網路的平衡點，常常震盪或模式崩塌（第13週 KP04、KP06）。

A **generative model** learns the distribution of the real data and then draws new samples from it. The two approaches of the past two weeks each have a problem. The VAE of Week 12 scores with squared error, which prefers the average when unsure, so its images tend to be blurry (Week 12 KP07). The GAN of Week 13 hires a discriminator as the judge and gives sharp images, but training searches for a balance point between two networks and often oscillates or collapses to a few modes (Week 13 KP04, KP06).

**擴散模型**（diffusion model）換了一個角度。它不要求網路一口氣從雜訊畫出一張圖，而是把這件難事拆成很多個小步驟，每一步只去掉一點點雜訊。來源說這個想法來自非平衡熱力學：一滴墨水滴進水裡會慢慢擴散，擴散模型想學的是反過來，把散開的墨水收回成一個形狀。

A **diffusion model** takes another angle. It does not ask a network to paint an image from noise in one go; it splits this hard job into many small steps, each removing only a little noise. The source says the idea comes from non-equilibrium thermodynamics: a drop of ink in water slowly spreads out, and a diffusion model wants to learn the reverse, gathering the spread-out ink back into a shape.

`[補充]` 預告一下它的好處：本週 KP05 會看到，擴散模型的訓練目標只是「猜出加進去的雜訊」，是一個普通的回歸問題，沒有對手，不必找平衡點。代價是生成時要走很多步，比 GAN 慢；KP06 會介紹加速的方法。今天的影像生成系統，例如 Stable Diffusion、`DALL-E 2` 與 Imagen，都以擴散模型為核心。

`[Added]` A preview of the payoff: this week's KP05 shows that a diffusion model's training target is just "guess the noise that was added", an ordinary regression problem with no opponent and no balance point to find. The price is that generation takes many steps and is slower than a GAN; KP06 introduces ways to speed it up. Today's image generators, such as Stable Diffusion, DALL-E 2 and Imagen, are built around diffusion models.

常見錯誤：以為擴散模型是 GAN 的一種改良，裡面也有判別器。擴散模型沒有判別器，也沒有對抗；它只有一個去雜訊的網路。

Common mistake: thinking a diffusion model is an improved GAN with a discriminator inside. A diffusion model has no discriminator and no adversarial game; it has only a denoising network.

### 2. 先攪散：前向過程（Scramble first: the forward process）

**前向過程**（forward process）從一筆真實資料 $\mathbf{x}_0$ 開始，每一步加一點高斯雜訊，得到 $\mathbf{x}_1,\mathbf{x}_2,\ldots$，走 $T$ 步之後的 $\mathbf{x}_T$ 幾乎是純雜訊。每一步只看上一步的結果，不看更早的歷史，這種一步接一步的過程叫做**馬可夫鏈**（Markov chain）。前向過程的每一步寫成 $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$，它是事先規定好的，沒有要學的參數。

The **forward process** starts from a real example $\mathbf{x}_0$ and adds a little Gaussian noise at each step, giving $\mathbf{x}_1,\mathbf{x}_2,\ldots$; after $T$ steps, $\mathbf{x}_T$ is almost pure noise. Each step looks only at the previous result, not at earlier history; such a step-by-step process is called a **Markov chain**. One step of the forward process is written $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$; it is fixed in advance and has no parameters to learn.

一步具體怎麼加？本週 KP02 會正式寫出來，這裡先看它的樣子：舊的值先乘上一個略小於 $1$ 的數，再加上一點雜訊。

How exactly is one step taken? This week's KP02 writes it formally; here is a first look: the old value is multiplied by a number slightly below $1$, and then a little noise is added.

$$
\mathbf{x}_t=\sqrt{1-\beta_t}\;\mathbf{x}_{t-1}+\sqrt{\beta_t}\;\epsilon,\qquad \epsilon\sim\mathcal{N}(0,1)\ \text{（每個元素各自抽 / drawn for each element）}.
$$

$\beta_t$ 是第 $t$ 步的雜訊量。實際的模型用很小的 $\beta_t$ 走很多步；為了方便手算，先用一個很大的 $\beta=0.36$ 走三步 `[補充]`。這時 $\sqrt{1-\beta}=0.8$、$\sqrt{\beta}=0.6$，規則變成 $x_t=0.8\,x_{t-1}+0.6\,\epsilon$。這個例子只有一個數，所以寫成不加粗的 $x$。從 $x_0=2$ 出發，三次抽到的雜訊是 $0.5$、$-1.0$、$0.0$：

$\beta_t$ is the amount of noise at step $t$. Real models take many steps with very small $\beta_t$; for easy hand calculation, first take three steps with a large $\beta=0.36$ `[Added]`. Then $\sqrt{1-\beta}=0.8$ and $\sqrt{\beta}=0.6$, and the rule becomes $x_t=0.8\,x_{t-1}+0.6\,\epsilon$. This example has a single number, so it is written as a non-bold $x$. Starting from $x_0=2$, the three noise draws are $0.5$, $-1.0$ and $0.0$:

$$
x_1=0.8\times2+0.6\times0.5=1.9,\qquad x_2=0.8\times1.9+0.6\times(-1.0)=0.92,\qquad x_3=0.8\times0.92+0.6\times0=0.736 .
$$

看「原本的 $2$ 還剩多少」：每一步乘 $0.8$，三步後剩 $2\times0.8^3=1.024$，二十步後只剩 $2\times0.8^{20}\approx0.023$，幾乎看不出來了。其餘的部分都是雜訊。

Look at "how much of the original $2$ is left": each step multiplies by $0.8$, so after three steps $2\times0.8^3=1.024$ is left, and after twenty steps only $2\times0.8^{20}\approx0.023$, almost invisible. Everything else is noise.

為什麼要先把舊值縮小，而不是直接加雜訊 `[補充]`？如果只加不縮，每一步的變異數都會再加上 $\beta$，數值越走越大。先乘 $\sqrt{1-\beta}$ 的話，一個變異數是 $1$ 的值經過一步後，變異數是 $(1-\beta)\times1+\beta=1$，保持不變；在這個例子是 $0.8^2+0.6^2=1$。更一般地，若目前的變異數是 $v$，一步後是 $(1-\beta)v+\beta$，和 $1$ 的差距每步縮成 $(1-\beta)$ 倍；平均也每步乘上 $\sqrt{1-\beta}$，越來越接近 $0$。所以不管資料從哪裡出發，走很多步以後都不會發散，而是穩穩地變成標準常態 $\mathcal{N}(0,1)$。

Why shrink the old value first instead of just adding noise `[Added]`? If you only add, the variance grows by $\beta$ at every step and the values drift ever larger. If you first multiply by $\sqrt{1-\beta}$, a value with variance $1$ has variance $(1-\beta)\times1+\beta=1$ after one step, unchanged; in this example $0.8^2+0.6^2=1$. More generally, if the variance is $v$, one step gives $(1-\beta)v+\beta$, so the gap to $1$ shrinks by a factor $(1-\beta)$ each step; the mean is also multiplied by $\sqrt{1-\beta}$ each step and approaches $0$. So wherever the data start, after many steps they do not blow up but settle into the standard normal $\mathcal{N}(0,1)$.

常見錯誤：以為前向過程是一個要訓練的網路，像 VAE 的編碼器。前向過程只是一條固定的公式，$\beta_t$ 事先選好，訓練時不會改。

Common mistake: thinking the forward process is a network to be trained, like the VAE's encoder. The forward process is just a fixed formula; the $\beta_t$ are chosen in advance and are not changed by training.

### 3. 再還原：逆向過程（Then restore: the backward process）

**逆向過程**（backward process，也稱反向過程）反過來走：從一個純雜訊 $\mathbf{x}_T\sim\mathcal{N}(0,1)$ 出發，每一步去掉一點雜訊，得到 $\mathbf{x}_{T-1},\ldots,\mathbf{x}_1$，最後得到 $\mathbf{x}_0$。每一步寫成 $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$，由一個參數為 $\theta$ 的神經網路負責。這一步是要學的：同一個有雜訊的 $\mathbf{x}_t$，可能來自很多不同的 $\mathbf{x}_{t-1}$，網路要學會合理地往資料的方向走一小步。

The **backward process** (also called the reverse process) goes the other way: start from pure noise $\mathbf{x}_T\sim\mathcal{N}(0,1)$, remove a little noise at each step to get $\mathbf{x}_{T-1},\ldots,\mathbf{x}_1$, and finally reach $\mathbf{x}_0$. Each step is written $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ and is carried out by a neural network with parameters $\theta$. This step must be learned: the same noisy $\mathbf{x}_t$ could have come from many different $\mathbf{x}_{t-1}$, and the network must learn to take a sensible small step towards the data.

用一個一維例子看兩個方向 `[補充]`。資料是兩堆數字：一半在 $-2$ 附近，一半在 $+2$ 附近，每堆的標準差 $0.3$，寫成 $0.5\,\mathcal{N}(-2,0.3^2)+0.5\,\mathcal{N}(2,0.3^2)$。前向過程用去雜訊擴散機率模型（denoising diffusion probabilistic model，DDPM；Ho 等人 2020）論文的設定：$T=1000$ 步，$\beta_t$ 從 $0.0001$ 等距增加到 $0.02$。逆向過程本來要用學到的網路；因為這個資料太簡單，對任何一個 $x_t$，都能精確算出「可能產生它的 $x_0$ 平均在哪裡」，於是每一步往那裡走一小步、再加一點變異數為 $\beta_t$ 的新雜訊。這裡先用這個「完美的去雜訊器」代替網路，看逆向過程做得好時長什麼樣。真正的網路要靠訓練去逼近它，是本週 KP04、KP05 的主題。兩個方向各用 $20000$ 個樣本。

Look at both directions with a one-dimensional example `[Added]`. The data are two piles of numbers: half near $-2$ and half near $+2$, each with standard deviation $0.3$, written $0.5\,\mathcal{N}(-2,0.3^2)+0.5\,\mathcal{N}(2,0.3^2)$. The forward process uses the setting of the denoising diffusion probabilistic model (DDPM) paper by Ho and colleagues (2020): $T=1000$ steps, with $\beta_t$ rising evenly from $0.0001$ to $0.02$. The backward process should use a learned network; because these data are so simple, for any $x_t$ we can compute exactly where, on average, the $x_0$ that could have produced it lies; each step moves a little towards it and adds fresh noise of variance $\beta_t$. Here this "perfect denoiser" stands in for the network, to show what the backward process looks like when it works well. A real network must approach it through training, the topic of this week's KP04 and KP05. Each direction uses $20000$ samples.

![兩排各五張直方圖，橫軸是 x，從 -4 到 4，每張都有一條灰色點線畫出標準常態。上排藍色是前向過程，由左到右是 t 等於 0、100、300、600、1000：t=0 是在 -2 與 2 的兩座窄山，標準差 2.02；t=100 兩座山變矮變寬；t=300 兩座山連在一起，中間凹下；t=600 已經是一座山，標準差 1.04；t=1000 和灰色點線重合，標準差 1.00。下排紫色是用完美去雜訊器跑的逆向過程，從右邊 t=1000 的標準常態開始，往左讀，最後在 t=0 又變回兩座窄山，標準差 2.02。標題列出每個 t 的根號 alpha bar：0.947、0.630、0.161、0.006（Two rows of five histograms each, with x from -4 to 4 and a grey dotted standard normal in every panel. The top row, in blue, is the forward process, left to right at t = 0, 100, 300, 600 and 1000: at t=0 there are two narrow peaks at -2 and 2 with standard deviation 2.02; at t=100 the peaks are lower and wider; at t=300 they have merged with a dip in the middle; at t=600 there is a single hill with standard deviation 1.04; at t=1000 it lies on the grey dotted curve with standard deviation 1.00. The bottom row, in purple, is the backward process run with the perfect denoiser, starting from the standard normal at t=1000 on the right and read leftwards; at t=0 it is back to two narrow peaks with standard deviation 2.02. The titles give the square root of alpha bar at each t: 0.947, 0.630, 0.161, 0.006)](../assets/figures/lec14-kp01-forward-backward.png)

**圖說 `lec14-kp01-forward-backward`**〔功能：一目了然、精準度〕：先由左到右讀上排藍色：兩座分開的窄山先變寬，在 $t=300$ 左右連成一片，到 $t=1000$ 已經和灰色點線的標準常態分不開。再由右到左讀下排紫色：從標準常態出發，在同樣的位置先長出兩個肩膀，最後在 $t=0$ 回到兩座窄山。上下兩排在同一個 $t$ 幾乎一樣：逆向過程在每一步重現前向的「分布」，但不是把個別樣本的路徑倒著重播（下一張圖）。標題裡的 $\sqrt{\bar\alpha_t}$ 是「原本的值還剩幾倍」，就是第2節每步乘 $0.8$ 的連乘（那裡三步是 $0.8^3=0.512$）；這裡從 $0.947$ 一路降到 $0.006$。

**Caption `lec14-kp01-forward-backward`** [Function: seeing it at a glance, precision]: first read the blue top row from left to right: the two separate narrow peaks widen, merge into one around $t=300$, and by $t=1000$ cannot be told apart from the grey dotted standard normal. Then read the purple bottom row from right to left: starting from the standard normal, two shoulders grow in the same places, and at $t=0$ there are two narrow peaks again. The two rows are almost identical at each $t$: the backward process reproduces the forward *distribution* at every step, but it does not replay individual sample paths in reverse (next figure). The $\sqrt{\bar\alpha_t}$ in the titles is "the fraction of the original value that is left", the running product of Section 2's factor $0.8$ per step (there, three steps gave $0.8^3=0.512$); here it falls from $0.947$ to $0.006$.

數字可以這樣讀。前向的標準差從 $2.02$（兩堆分在兩邊，所以很大）降到 $t=600$ 的 $1.04$、$t=1000$ 的 $1.00$，平均一直在 $0$ 附近。$t=1000$ 時原本的值只剩 $\sqrt{\bar\alpha_{1000}}\approx0.006$ 倍，$\mathbf{x}_0$ 與 $\mathbf{x}_T$ 的相關係數理論值只有 $0.013$（這次模擬估計為 $0.02$），幾乎無關。逆向跑完以後，有 $50\%$ 的樣本落在右堆，右堆的平均 $2.00$、標準差 $0.30$，和真實資料一樣。

The numbers read as follows. The forward standard deviation falls from $2.02$ (large because the two piles sit on opposite sides) to $1.04$ at $t=600$ and $1.00$ at $t=1000$, while the mean stays near $0$. At $t=1000$ only $\sqrt{\bar\alpha_{1000}}\approx0.006$ of the original value is left, and the correlation between $\mathbf{x}_0$ and $\mathbf{x}_T$ is only $0.013$ in theory ($0.02$ as estimated in this simulation), practically none. After the backward run, $50\%$ of the samples land in the right pile, whose mean is $2.00$ and standard deviation $0.30$, matching the real data.

![兩張並排的折線圖，縱軸是 x_t，從 -4.2 到 4.2，灰色虛線標出 2 與 -2。左圖藍色是前向過程，橫軸 t 從 0 到 1000：四條線兩條從 2、兩條從 -2 出發，越走越亂，最後混在一起，散布在 -2 到 2 之間。右圖紫色是逆向過程，橫軸從 1000 往左減到 0：六條線都從同一個 x_T=0.3 出發，一開始糾纏在一起，最後有兩條停在 2 附近，三條停在 -2 附近，一條停在 -1.3 附近（Two line charts side by side, with x_t on the vertical axis from -4.2 to 4.2 and grey dashed lines at 2 and -2. The left chart, in blue, is the forward process, with t from 0 to 1000: four lines, two starting at 2 and two at -2, grow ever more erratic and end mixed together between -2 and 2. The right chart, in purple, is the backward process, with t decreasing from 1000 on the left to 0: six lines all start from the same x_T = 0.3, tangle at first, and end with two near 2, three near -2 and one near -1.3)](../assets/figures/lec14-kp01-trajectories.png)

**圖說 `lec14-kp01-trajectories`**〔功能：一目了然〕：左圖看單一樣本怎麼走：從 $\pm2$ 出發的四條線，走到後面已經分不出誰從哪裡來。右圖六條線用同一個起點 $x_T=0.3$，只因為每一步抽到的雜訊不同，就走到不同的終點：兩條到右堆附近、四條到左堆附近；其中一條停在 $-1.3$，那是左堆的尾巴，不是兩堆之間。

**Caption `lec14-kp01-trajectories`** [Function: seeing it at a glance]: in the left chart, follow single samples: the four lines starting at $\pm2$ end up so mixed that you cannot tell which came from where. In the right chart the six lines share the starting point $x_T=0.3$, and only because each step draws different noise they reach different end points: two near the right pile and four near the left pile; one of them stops at $-1.3$, which is the tail of the left pile, not the gap between the piles.

這兩張圖說明一件重要的事 `[補充]`：逆向過程產生的是「一筆新的資料」，不是把當初那一筆 $\mathbf{x}_0$ 找回來。前向走到底，$\mathbf{x}_T$ 已經幾乎不記得 $\mathbf{x}_0$；逆向的每一步又會再加一點新的雜訊，所以同一個起點也能走到不同的樣本。這正是我們要的：生成模型就是要從同一個分布裡抽出不一樣的新東西。

These two figures show something important `[Added]`: the backward process produces "a new example", not the original $\mathbf{x}_0$ recovered. At the end of the forward process $\mathbf{x}_T$ remembers almost nothing of $\mathbf{x}_0$, and each backward step adds a little fresh noise, so even the same starting point can reach different samples. This is exactly what we want: a generative model should draw new and different things from the same distribution.

常見錯誤：把墨水的比喻想成「把那一滴墨水原封不動地收回來」。擴散模型收回來的是一個「像墨滴」的形狀，每次都可能不一樣；它學的是整個資料分布，不是某一筆資料。

Common mistake: reading the ink analogy as "gathering that very drop of ink back exactly". What a diffusion model gathers back is a shape that "looks like a drop", which can differ every time; it learns the whole data distribution, not one particular example.

### 4. 把擴散模型看成自編碼器（A diffusion model as an autoencoder）

來源用第12週的自編碼器來類比。把前向過程看成**編碼器**（encoder）：它把資料 $\mathbf{x}_0$ 變成雜訊 $\mathbf{x}_T$；把逆向的去雜訊網路看成**解碼器**（decoder）：它把雜訊變回資料。跟 VAE 相比，關鍵差在編碼器：VAE 兩個網路都得訓練；擴散模型的編碼器是寫死的公式，要訓練的只有解碼器。

The source draws an analogy with the autoencoder of Week 12. Treat the forward process as the **encoder**: it turns data $\mathbf{x}_0$ into noise $\mathbf{x}_T$; treat the backward denoising network as the **decoder**: it turns noise back into data. The main difference from a VAE is that a VAE learns both its encoder and its decoder, while a diffusion model's encoder is fixed and only its decoder is learned.

![兩條左右對齊的方塊流程圖。上排是擴散模型：淺藍色的資料 x0，經過灰色虛線框的固定編碼器（加雜訊 T 步），變成黃色的雜訊 xT，再經過紫色的學習解碼器 p theta(x t-1 | x t)，得到淺藍色的新 x0；下方註明潛在變數 x1 到 xT 和 x0 一樣大。下排是 VAE：資料 x，經過藍色的學習編碼器 q phi(z | x)，變成灰色的較小潛在變數 z，再經過藍色的學習解碼器 p theta(x | z)，得到 x hat；下方註明兩個網路一起訓練（Two left-aligned block flow charts. Top, the diffusion model: light-blue data x0 passes through a grey dashed-border fixed encoder (add noise, T steps) into yellow noise xT, then through a purple learned decoder p theta(x t-1 | x t) to a light-blue new x0; a note below says the latents x1 to xT have the same size as x0. Bottom, the VAE: data x passes through a blue learned encoder q phi(z | x) into a grey, smaller latent z, then through a blue learned decoder p theta(x | z) to x hat; a note below says both networks are trained together)](../assets/figures/lec14-kp01-pipelines.png)

**圖說 `lec14-kp01-pipelines`**〔功能：對照〕：上下兩排用同樣的位置對齊。先比第二格：擴散模型是虛線框，代表固定、不用學；VAE 是藍色實線框，代表要學。再比中間那格：擴散模型的「潛在變數」就是雜訊 $\mathbf{x}_T$，和資料一樣大；VAE 的 $\mathbf{z}$ 通常小很多。最後看第四格：兩者都要學一個解碼器。

**Caption `lec14-kp01-pipelines`** [Function: comparison]: the two rows are aligned position by position. First compare the second box: in the diffusion model it has a dashed border, meaning fixed and not learned; in the VAE it is blue with a solid border, meaning learned. Then compare the middle box: the diffusion model's "latent variable" is the noise $\mathbf{x}_T$, as large as the data; the VAE's $\mathbf{z}$ is usually much smaller. Finally the fourth box: both learn a decoder.

| 比較 | VAE（第12週） | 擴散模型 |
|---|---|---|
| 編碼器 | 神經網路，要學 | 固定的加雜訊公式，不用學 |
| 潛在變數 | 一個 $\mathbf{z}$，通常比資料小 | 一串 $\mathbf{x}_1,\ldots,\mathbf{x}_T$，和資料一樣大 |
| 解碼器 | 一步從 $\mathbf{z}$ 變回資料 | 走 $T$ 小步，每步去掉一點雜訊 |
| 潛在變數記得資料嗎 | 記得，可以用來重建 | $\mathbf{x}_T$ 幾乎不記得 $\mathbf{x}_0$ |

| Comparison | VAE (Week 12) | Diffusion model |
|---|---|---|
| Encoder | a neural network, learned | a fixed noising formula, not learned |
| Latent variables | one $\mathbf{z}$, usually smaller than the data | a chain $\mathbf{x}_1,\ldots,\mathbf{x}_T$, as large as the data |
| Decoder | one step from $\mathbf{z}$ back to data | $T$ small steps, each removing a little noise |
| Do the latents remember the data | yes, they can be used to reconstruct | $\mathbf{x}_T$ remembers almost nothing of $\mathbf{x}_0$ |

這個類比也有界線 `[補充]`。一般自編碼器的編碼要保留資料的資訊，才能重建；擴散模型的編碼卻故意把資訊磨掉，最後的 $\mathbf{x}_T$ 跟 $\mathbf{x}_0$ 幾乎無關（第3節的相關係數約 $0.01$）。所以它的「解碼器」不是在重建某一筆資料，而是在生成新資料。好處是：編碼器固定，起點 $\mathbf{x}_T$ 的分布一定近似標準常態，生成時直接從 $\mathcal{N}(0,1)$ 抽就好，不用像 VAE 那樣另外要求 $\mathbf{z}$ 的分布貼近先驗。

The analogy has limits `[Added]`. An ordinary autoencoder's code must keep the data's information so it can reconstruct; a diffusion model's code deliberately wears that information away, and the final $\mathbf{x}_T$ is almost unrelated to $\mathbf{x}_0$ (correlation about $0.01$ in Section 3). So its "decoder" is not reconstructing one example but generating new data. The benefit: with a fixed encoder, the starting point $\mathbf{x}_T$ is always approximately standard normal, so generation simply draws from $\mathcal{N}(0,1)$, with no need, as in a VAE, to push the distribution of $\mathbf{z}$ towards a prior.

常見錯誤：以為擴散模型不需要潛在變數。它其實有很多個潛在變數，就是中間每一步的 $\mathbf{x}_1,\ldots,\mathbf{x}_T$；只是它們由固定的公式產生，不用學。本週 KP03 會像第12週一樣，用 ELBO 處理這些潛在變數。

Common mistake: thinking a diffusion model needs no latent variables. It actually has many: the intermediate $\mathbf{x}_1,\ldots,\mathbf{x}_T$ of every step; they are just produced by a fixed formula rather than learned. This week's KP03 handles these latent variables with the ELBO, as in Week 12.

### 本單元重點整理（KP01 summary）

**一句話重點**：擴散模型先用固定的前向過程把資料一步一步加雜訊，變成標準常態；再訓練一個網路，從標準常態一步一步去雜訊，生成新的資料。

**One-sentence summary**: a diffusion model first uses a fixed forward process to add noise to the data step by step until it becomes a standard normal; it then trains a network to remove noise step by step from a standard normal, generating new data.

**白話說明**：像把一滴墨水滴進水裡，看它慢慢散開；這段過程不用學，誰來做都一樣。難的是倒過來：只看一杯淡淡的墨水，一點一點把它聚回一滴墨的樣子。擴散模型把這件難事切成一千個小步，每一步只做一點點，所以每一步都學得會。

**Plain explanation**: it is like dropping ink into water and watching it slowly spread; that part needs no learning and goes the same for everyone. The hard part is the reverse: looking only at a glass of faintly tinted water, gather it back little by little into the shape of a drop. A diffusion model cuts this hard job into a thousand small steps, each doing only a little, so every step is learnable.

**正式定義**：前向過程是固定的馬可夫鏈 $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$，每一步 $\mathbf{x}_t=\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}+\sqrt{\beta_t}\,\epsilon$，$\epsilon\sim\mathcal{N}(0,1)$，$t=1,\ldots,T$，使 $\mathbf{x}_T$ 近似標準常態。逆向過程從 $\mathbf{x}_T\sim\mathcal{N}(0,1)$ 出發，以要學的 $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ 逐步走回 $\mathbf{x}_0$。

**Formal definition**: the forward process is a fixed Markov chain $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ with each step $\mathbf{x}_t=\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}+\sqrt{\beta_t}\,\epsilon$, $\epsilon\sim\mathcal{N}(0,1)$, $t=1,\ldots,T$, so that $\mathbf{x}_T$ is approximately standard normal. The backward process starts from $\mathbf{x}_T\sim\mathcal{N}(0,1)$ and steps back to $\mathbf{x}_0$ with the learned $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$.

**例子**：$\beta=0.36$、$x_0=2$、雜訊 $0.5,-1.0,0.0$ 時，$x_1=1.9$、$x_2=0.92$、$x_3=0.736$。一維兩堆資料用 $T=1000$、$\beta_t$ 從 $0.0001$ 到 $0.02$：前向的標準差從 $2.02$ 降到 $1.00$，$\sqrt{\bar\alpha_{1000}}\approx0.006$；用完美去雜訊器逆向，右堆回到平均 $2.00$、標準差 $0.30$。

**Example**: with $\beta=0.36$, $x_0=2$ and noise $0.5,-1.0,0.0$, we get $x_1=1.9$, $x_2=0.92$, $x_3=0.736$. For the one-dimensional two-pile data with $T=1000$ and $\beta_t$ from $0.0001$ to $0.02$, the forward standard deviation falls from $2.02$ to $1.00$ and $\sqrt{\bar\alpha_{1000}}\approx0.006$; running backward with the perfect denoiser, the right pile returns to mean $2.00$ and standard deviation $0.30$.

**常見迷思**：

- 「擴散模型裡也有判別器。」錯，它只有一個去雜訊網路，沒有對抗。
- 「前向過程是要訓練的編碼器。」錯，它是固定的公式。
- 「逆向過程會找回原本那一筆資料。」錯，它生成一筆新的資料。
- 「擴散模型沒有潛在變數。」錯，$\mathbf{x}_1,\ldots,\mathbf{x}_T$ 都是潛在變數，只是不用學。

**Common misconceptions**:

- "A diffusion model also has a discriminator." Wrong: it has only a denoising network and no adversarial game.
- "The forward process is an encoder to be trained." Wrong: it is a fixed formula.
- "The backward process recovers the original example." Wrong: it generates a new example.
- "A diffusion model has no latent variables." Wrong: $\mathbf{x}_1,\ldots,\mathbf{x}_T$ are all latent variables; they are just not learned.

**圖**：本單元有三張圖。

- `lec14-kp01-forward-backward`：一維兩堆資料的前向（上排）與逆向（下排）直方圖，兩排在同一個 $t$ 幾乎一樣。
- `lec14-kp01-trajectories`：單一樣本的前向軌跡，以及從同一個起點走到不同終點的逆向軌跡。
- `lec14-kp01-pipelines`：擴散模型與 VAE 的自編碼器對照，固定與要學的部分用顏色區分。

**Figures** — this unit has three figures.

- `lec14-kp01-forward-backward`: forward (top row) and backward (bottom row) histograms of the one-dimensional two-pile data; the rows are almost identical at each $t$.
- `lec14-kp01-trajectories`: forward paths of single samples, and backward paths that start at the same point and end at different ones.
- `lec14-kp01-pipelines`: the diffusion model and the VAE as autoencoders, with fixed and learned parts told apart by colour.

### 小練習（Practice）

1. 用 $\beta=0.36$ 的規則 $x_t=0.8\,x_{t-1}+0.6\,\epsilon$，從 $x_0=-1$ 出發，三次雜訊是 $1.0$、$0.0$、$-0.5$，算 $x_1$、$x_2$、$x_3$。原本的 $-1$ 在三步後還剩多少？
2. 如果把規則改成只加不縮，$x_t=x_{t-1}+0.6\,\epsilon$，從一個變異數是 $1$ 的值出發，走 $10$ 步以後變異數是多少？這說明了什麼？
3. 一位同學說：「擴散模型就是一個自編碼器，所以把一張照片加雜訊到 $\mathbf{x}_T$ 再跑逆向過程，就會得到原本那張照片。」請指出哪裡不對。

1. With the $\beta=0.36$ rule $x_t=0.8\,x_{t-1}+0.6\,\epsilon$, start from $x_0=-1$ with noise draws $1.0$, $0.0$ and $-0.5$; compute $x_1$, $x_2$ and $x_3$. How much of the original $-1$ is left after three steps?
2. If the rule only adds without shrinking, $x_t=x_{t-1}+0.6\,\epsilon$, what is the variance after $10$ steps, starting from a value with variance $1$? What does this show?
3. A classmate says: "A diffusion model is an autoencoder, so if I noise a photo up to $\mathbf{x}_T$ and run the backward process, I get the original photo back." Point out what is wrong.

**第 1 題解答** `[補充]`：$x_1=0.8\times(-1)+0.6\times1.0=-0.2$，$x_2=0.8\times(-0.2)+0=-0.16$，$x_3=0.8\times(-0.16)+0.6\times(-0.5)=-0.428$。原本的 $-1$ 只剩 $-1\times0.8^3=-0.512$，其餘是雜訊的貢獻。

**Solution to question 1** `[Added]`: $x_1=0.8\times(-1)+0.6\times1.0=-0.2$, $x_2=0.8\times(-0.2)+0=-0.16$, $x_3=0.8\times(-0.16)+0.6\times(-0.5)=-0.428$. Of the original $-1$, only $-1\times0.8^3=-0.512$ is left; the rest comes from the noise.

**第 2 題解答** `[補充]`：每一步變異數加 $0.6^2=0.36$，十步後是 $1+10\times0.36=4.6$，而且步數越多越大，不會停在某個固定的分布。先縮小舊值，才能讓變異數保持在 $1$，終點才會是標準常態。

**Solution to question 2** `[Added]`: each step adds $0.6^2=0.36$ to the variance, so after ten steps it is $1+10\times0.36=4.6$, and it keeps growing with more steps instead of settling into a fixed distribution. Shrinking the old value first is what keeps the variance at $1$, so that the end point is a standard normal.

**第 3 題解答** `[補充]`：加雜訊到 $\mathbf{x}_T$ 以後，原本的照片幾乎只剩 $0.006$ 倍（第3節），$\mathbf{x}_T$ 跟原圖幾乎無關；逆向過程每一步又會加新的雜訊。所以會得到一張「像資料」的新照片，通常不是原來那張。這也是類比的界線：擴散模型的編碼故意把資訊磨掉，它的目的是生成，不是重建。

**Solution to question 3** `[Added]`: after noising up to $\mathbf{x}_T$, only about $0.006$ of the original photo is left (Section 3), so $\mathbf{x}_T$ is almost unrelated to it, and each backward step adds fresh noise. The result is a new photo that "looks like data", usually not the original one. This is the limit of the analogy: a diffusion model's code deliberately wears information away; its purpose is generation, not reconstruction.

### 小測驗（Quiz，來源原題）

來源 KP-1 有四題自我檢測，保留原題與全部選項；來源頁面附有答案與簡短解析；下面的解析由本講義依各節內容補寫。

The source's KP-1 has four self-check questions, kept here with all their options; the source page gives the answers with short explanations; the explanations below are written by this note from its own sections.

1. Diffusion model 的核心精神是什麼？
   - (A) 用 GAN 的對抗訓練生成資料
   - (B) 先加噪再學習去噪
   - (C) 只學 encoder 不學 decoder
   - (D) 用自回歸一個像素一個像素生成
2. 在 diffusion model 的 autoencoder 類比中，哪一個部分是固定的？
   - (A) Decoder
   - (B) Encoder（前向加噪）
   - (C) 兩者都固定
   - (D) 兩者都學習
3. Diffusion model 與 VAE 的主要差異是什麼？
   - (A) Diffusion 使用連續 latent space，VAE 使用離散
   - (B) Diffusion 的 encoder 固定，VAE 的 encoder 與 decoder 都需學習
   - (C) VAE 只能做分類
   - (D) Diffusion 不需要 latent variable
4. 到 2025 年，以下哪個應用不是 diffusion model 驅動的代表系統？
   - (A) Stable Diffusion
   - (B) DALL-E
   - (C) Imagen
   - (D) Word2Vec

1. What is the core idea of a diffusion model?
   - (A) Generate data with a GAN's adversarial training
   - (B) Add noise first, then learn to remove it
   - (C) Learn only the encoder, not the decoder
   - (D) Generate autoregressively, one pixel at a time
2. In the autoencoder analogy for a diffusion model, which part is fixed?
   - (A) The decoder
   - (B) The encoder (forward noising)
   - (C) Both are fixed
   - (D) Both are learned
3. What is the main difference between a diffusion model and a VAE?
   - (A) Diffusion uses a continuous latent space and the VAE a discrete one
   - (B) Diffusion's encoder is fixed, while the VAE learns both its encoder and its decoder
   - (C) The VAE can only classify
   - (D) Diffusion needs no latent variable
4. As of 2025, which of the following is not a representative system driven by diffusion models?
   - (A) Stable Diffusion
   - (B) DALL-E
   - (C) Imagen
   - (D) Word2Vec

**解析** `[補充]`：

- 第 1 題選 (B)，見第2、3節。(A) 是 GAN；(C) 剛好相反，擴散模型的編碼器固定、解碼器要學；(D) 是自迴歸模型的做法。
- 第 2 題選 (B)，見第4節：前向加雜訊是事先規定的馬可夫鏈，只有逆向去雜訊網路要學。
- 第 3 題選 (B)，見第4節的表。(D) 錯在擴散模型其實有一串潛在變數 $\mathbf{x}_{1:T}$；(A) 錯在兩者的潛在變數都是連續的。
- 第 4 題選 (D)：Word2Vec 是把詞變成向量的方法，不是影像生成模型。另外注意，最早的 `DALL-E`（2021）是自迴歸的 Transformer，從 `DALL-E 2` 起才改用擴散模型，見「來源更正」。

**Explanations** `[Added]`:

- Question 1: B; see Sections 2 and 3. A is the GAN; C is the opposite of the truth, since a diffusion model's encoder is fixed and its decoder is learned; D is how autoregressive models work.
- Question 2: B; see Section 4: forward noising is a Markov chain fixed in advance, and only the backward denoising network is learned.
- Question 3: B; see the table in Section 4. D is wrong because a diffusion model actually has a chain of latent variables $\mathbf{x}_{1:T}$; A is wrong because both models use continuous latent variables.
- Question 4: D: Word2Vec turns words into vectors and is not an image generator. Note also that the first DALL-E (2021) was an autoregressive Transformer; diffusion models came in with DALL-E 2; see "Corrections to the source".

### 留給下一單元的問題（The question for the next unit）

第2節只用一個大的 $\beta$ 手算了三步。實際模型要走 $1000$ 步，$\beta_t$ 該怎麼排？如果要知道第 $500$ 步長什麼樣，難道要真的走 $500$ 次嗎？第3節圖上的 $\bar\alpha_t$ 又是什麼？下一個知識點（本週 KP02）把前向過程寫成正式的高斯分布，並推出「一步跳到任意第 $t$ 步」的公式。

Section 2 computed only three steps by hand with a large $\beta$. A real model takes $1000$ steps; how should the $\beta_t$ be scheduled? To see what step $500$ looks like, must we really take $500$ steps? And what is the $\bar\alpha_t$ in the titles of the Section 3 figure? The next knowledge point (this week's KP02) writes the forward process as a formal Gaussian distribution and derives a formula that "jumps straight to any step $t$".

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第15章「Diffusion Models」KP-1「章節定位：diffusion 動機、前向/逆向直覺、autoencoder/VAE 類比」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch15-diffusion-models/index.html ，讀取日期 2026-11-15）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from KP-1, "Chapter overview: the motivation for diffusion, forward and backward intuition, the autoencoder and VAE analogy", of Chapter 15, "Diffusion Models", of CCC-LNS IDL_115 (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch15-diffusion-models/index.html , accessed 2026-11-15). Content not in the source is marked `[Added]` where it appears.

- 來源寫逆向過程「學習一步一步去掉雜訊，最後還原出 $x_0$」。從純雜訊出發時，得到的是一筆新的樣本，不是原本那一筆 $x_0$；本講義第3節用軌跡圖說明，並把墨水比喻改寫成「收回一個像墨滴的形狀」。
- 小測驗第 4 題把 `DALL-E` 列為擴散模型的代表系統。最早的 `DALL-E`（2021）是自迴歸的 Transformer，從 `DALL-E 2`（2022）起才以擴散模型生成影像；本講義第1節寫 `DALL-E 2`，小測驗保留原題並在解析中註明。
- 來源的示意圖提示說「固定 encoder 反而讓訓練更穩定，因為 forward 沒有學習誤差」。擴散模型訓練穩定，主要是因為訓練目標是單純的回歸、沒有對抗（本週 KP05）；固定的前向過程讓起點分布確定、目標容易寫出，是原因之一，但不是全部。本講義第1、4節照這個說法寫。
- 本講義補上：一步加雜訊的規則（本週 KP02 正式推導）與三步手算、先縮小再加雜訊的理由、一維兩堆資料的前向與逆向模擬（用可直接算出的完美去雜訊器代替網路）與三張圖、同一起點走到不同終點的說明、自編碼器類比的界線、擴散模型與 VAE 的對照表、三題小練習與解答、小測驗的解析。
- 取捨：來源 KP-1 的「本章路線圖」改寫成各節結尾的預告；Fig. 4 的機率圖模型比較改用第4節的對照表與流程圖說明；網頁的互動展示不重做。

- The source says the backward process "learns to remove noise step by step and finally restores $x_0$". Starting from pure noise, the result is a new sample, not the original $x_0$; Section 3 shows this with path plots and rewrites the ink analogy as "gathering back a shape that looks like a drop".
- Quiz question 4 lists DALL-E as a representative diffusion system. The first DALL-E (2021) was an autoregressive Transformer; images have been generated with diffusion models only from DALL-E 2 (2022) on. Section 1 names DALL-E 2, and the quiz keeps the original question with a note in the explanation.
- The source's figure prompt says "a fixed encoder actually makes training more stable, because the forward pass has no learning error". Diffusion training is stable mainly because the training target is plain regression with no adversary (this week's KP05); the fixed forward process, which makes the starting distribution certain and the target easy to write down, is one reason but not the whole story. Sections 1 and 4 follow this account.
- This note adds: the one-step noising rule (derived formally in this week's KP02) with a three-step hand calculation, the reason for shrinking before adding noise, the forward and backward simulation of one-dimensional two-pile data (with a directly computable perfect denoiser standing in for the network) with three figures, the explanation that the same starting point reaches different end points, the limits of the autoencoder analogy, the diffusion–VAE comparison table, three practice questions with solutions, and the explanations of the quiz.
- Choices: the "chapter roadmap" of KP-1 is turned into previews at the ends of sections; the probabilistic-graph comparison of Fig. 4 is replaced by the table and flow chart of Section 4; the web page's interactive demo is not rebuilt.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 擴散模型 | diffusion model |
| 生成模型 | generative model |
| 前向過程（加雜訊） | forward process (noising) |
| 逆向過程（去雜訊） | backward process, reverse process (denoising) |
| 馬可夫鏈 | Markov chain |
| 高斯雜訊 | Gaussian noise |
| 標準常態分布 | standard normal distribution |
| 雜訊排程（KP02） | noise schedule |
| 非平衡熱力學 | non-equilibrium thermodynamics |
| 去雜訊擴散機率模型 | denoising diffusion probabilistic model (DDPM) |
| 自編碼器（第12週） | autoencoder |
| 編碼器 | encoder |
| 解碼器 | decoder |
| 潛在變數（第12週 KP02） | latent variable |
| 變分自編碼器（第12週） | variational autoencoder (VAE) |
| 生成對抗網路（第13週） | generative adversarial network (GAN) |
| 判別器（第13週） | discriminator |
| 模式崩塌（第13週 KP06） | mode collapse |
| 證據下界（KP03） | evidence lower bound (ELBO) |
| 自迴歸模型 | autoregressive model |
| 相關係數 | correlation coefficient |
| 詞向量 Word2Vec | Word2Vec (word embeddings) |
