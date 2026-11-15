# 第14週 KP06：改進 DDPM：學變異數、cosine 排程、PriorGrad 與加速（Improving DDPM: learned variance, the cosine schedule, PriorGrad and speed-ups）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第14週「深度生成模型（三）：擴散模型」的第六個知識點（KP06）。KP05 的 DDPM 已經能生成像樣的資料，但留下幾個明顯可以改的地方：逆向一步的變異數是寫死的、線性排程讓後段很多步幾乎只剩雜訊、起點一律是標準常態、生成一個樣本要算 $1000$ 次網路。這一節依序介紹對應的改良，最後在一維例子上實際比較 DDPM 與 DDIM 用少步數生成的效果。

This is the sixth knowledge point (KP06) of Week 14, "Deep generative models (3): diffusion models". KP05's DDPM can already generate decent data, but it leaves clear room for improvement: the backward-step variance is hard-coded, the linear schedule leaves many late steps as almost pure noise, the start is always a standard normal, and generating one sample takes $1000$ network evaluations. This note introduces the corresponding improvements in turn, and finally compares DDPM and DDIM generating with few steps on a one-dimensional example.

先備知識是本週 KP02 到 KP05：排程與閉式、ELBO、真正的去噪分布與 $\tilde\beta_t$、簡化損失與兩個演算法。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第15章 KP-10 到 KP-13（第 3 節）改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec14-kp06-figures.py` 算出，可以自己重算。

The prerequisites are this week's KP02 to KP05: the schedule and closed form, the ELBO, the true denoising distribution and $\tilde\beta_t$, and the simplified loss with the two algorithms. Every term and symbol is defined when it first appears. This note is rewritten from KP-10 to KP-13 (Section 3) of Chapter 15 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec14-kp06-figures.py`, so you can recompute it yourself.

## KP06 改進 DDPM：學變異數、cosine 排程、PriorGrad 與加速（Improving DDPM: learned variance, the cosine schedule, PriorGrad and speed-ups）

### 學習目標（Learning objectives）

讀完之後，你能說明為什麼可以讓網路學逆向變異數、怎麼在 $\beta_t$ 與 $\tilde\beta_t$ 之間做對數插值，以及混合損失 $L_{\text{simple}}+\lambda L_{\text{vlb}}$ 的用意。你能用一個小例子說明重要性抽樣怎麼降低估計的變異數。你能寫出 cosine 排程，並從圖說出它和線性排程的差別。你能說出 PriorGrad 改了起點分布的什麼。你能寫出 DDIM 的一步，說明它為什麼不用重新訓練就能少走步數，並讀懂 DDPM 與 DDIM 在少步數下的比較。你也能說出 Consistency Models、蒸餾與 D3PM 各自的想法。

After reading, you can explain why the network may learn the backward variance, how to interpolate between $\beta_t$ and $\tilde\beta_t$ in log space, and the purpose of the hybrid loss $L_{\text{simple}}+\lambda L_{\text{vlb}}$. You can use a small example to show how importance sampling lowers the variance of an estimate. You can write the cosine schedule and read off its difference from the linear one in a figure. You can say what PriorGrad changes about the starting distribution. You can write one DDIM step, explain why it takes fewer steps without retraining, and read a comparison of DDPM and DDIM with few steps. You can also state the ideas behind Consistency Models, distillation and D3PM.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\beta_t,\ \alpha_t,\ \bar\alpha_t$ | 雜訊排程與它的連乘（KP02） | the noise schedule and its running product (KP02) |
| $\tilde\beta_t$ | 真正去噪分布的變異數 $\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$（KP04） | the true denoising variance $\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ (KP04) |
| $\Sigma_\theta$ | 網路決定的逆向一步共變異數 | the backward-step covariance set by the network |
| $v$ | 網路輸出的插值係數，每個維度一個 | the interpolation coefficient output by the network, one per dimension |
| $L_{\text{simple}},\ L_{\text{vlb}}$ | 簡化損失（KP05）；完整的變分下界損失（負的 ELBO） | the simplified loss (KP05); the full variational-lower-bound loss (the negative ELBO) |
| $\lambda$ | 混合損失中 $L_{\text{vlb}}$ 的權重 | the weight of $L_{\text{vlb}}$ in the hybrid loss |
| $p_t$ | 抽到第 $t$ 步的機率 | the probability of drawing step $t$ |
| $f(t),\ s$ | cosine 排程的函數與小位移 $s=0.008$ | the cosine schedule's function and its small offset $s=0.008$ |
| $\mu,\ \Sigma$ | PriorGrad 用的資料平均與共變異數（和網路的 $\Sigma_\theta$ 不同） | the data mean and covariance used by PriorGrad (not the network's $\Sigma_\theta$) |
| $\hat{\mathbf{x}}_0$ | 由 $\mathbf{x}_t$ 和 $\epsilon_\theta$ 反推的原資料猜測 | the guess of the original data worked back from $\mathbf{x}_t$ and $\epsilon_\theta$ |
| $t'$ | DDIM 一次跳到的較早步數，$t'<t$（不要和 cosine 排程的位移 $s$ 混淆） | the earlier step that DDIM jumps to, $t'<t$ (not to be confused with the cosine offset $s$) |
| $\eta$ | DDIM 生成時加多少隨機性；$\eta=0$ 是完全確定 | how much randomness DDIM adds when generating; $\eta=0$ is fully deterministic |
| $K$ | 生成時實際走的步數 | the number of steps actually taken in generation |

### 1. 讓網路學變異數（Let the network learn the variance）

KP05 的 DDPM 把逆向變異數 $\sigma_t^2$ 固定成 $\beta_t$ 或 $\tilde\beta_t$。這兩個是合理範圍的兩端：KP04 已經算過，$\tilde\beta_t\le\beta_t$，而且只在前幾步差得多。DDPM 排程下，$\tilde\beta_t/\beta_t$ 在 $t=2$ 是 $0.45$，$t=10$ 是 $0.85$，$t=50$ 是 $0.96$，$t=100$ 以後幾乎是 $1$。

KP05's DDPM fixes the backward variance $\sigma_t^2$ at $\beta_t$ or $\tilde\beta_t$. These two are the ends of a reasonable range: KP04 showed that $\tilde\beta_t\le\beta_t$, and they differ much only in the first few steps. Under the DDPM schedule, $\tilde\beta_t/\beta_t$ is $0.45$ at $t=2$, $0.85$ at $t=10$, $0.96$ at $t=50$, and almost $1$ from $t=100$ on.

Nichol 與 Dhariwal（2021）讓網路自己在兩端之間挑。網路多輸出一個向量 $v$，變異數取兩者在**對數空間**的插值（來源的式 (44)）：

Nichol and Dhariwal (2021) let the network choose between the two ends itself. The network outputs an extra vector $v$, and the variance is the interpolation of the two **in log space** (the source's equation (44)):

$$
\Sigma_\theta(\mathbf{x}_t,t)=\exp\!\big(v\log\beta_t+(1-v)\log\tilde\beta_t\big).
$$

用來源互動展示的數字 `[補充]`：$\beta_t=0.2$、$\bar\alpha_{t-1}=0.6$，所以 $\tilde\beta_t\approx0.1538$。$v=0$ 得 $0.1538$，$v=1$ 得 $0.2$，$v=0.5$ 得 $0.1754$（就是兩者的幾何平均 $\sqrt{0.2\times0.1538}$）。為什麼在對數空間插值？因為指數一定是正數：就算 $v$ 跑出 $0$ 到 $1$ 之外（例如 $v=1.2$ 得 $0.2108$），變異數也不會變成負的；而且兩端都是很小的正數，比值可以差到兩倍以上（第一步甚至 $\tilde\beta_1=0$），在對數尺度上調整比較自然。

With the source's interactive-demo numbers `[Added]`: $\beta_t=0.2$ and $\bar\alpha_{t-1}=0.6$, so $\tilde\beta_t\approx0.1538$. $v=0$ gives $0.1538$, $v=1$ gives $0.2$, and $v=0.5$ gives $0.1754$ (exactly the geometric mean $\sqrt{0.2\times0.1538}$). Why interpolate in log space? Because an exponential is always positive: even if $v$ strays outside $0$ to $1$ (for example $v=1.2$ gives $0.2108$), the variance never turns negative; and since both ends are small positive numbers whose ratio can exceed two (at the first step even $\tilde\beta_1=0$), adjusting on a log scale is more natural.

$L_{\text{simple}}$ 只看平均，學不到變異數。所以改用**混合損失**（來源的式 (45)）：

$L_{\text{simple}}$ looks only at the mean and cannot learn the variance. So a **hybrid loss** is used instead (the source's equation (45)):

$$
L_{\text{hybrid}}=L_{\text{simple}}+\lambda\,L_{\text{vlb}},\qquad\lambda=0.001,
$$

$L_{\text{vlb}}$ 是 KP03 的完整 ELBO（取負號當損失），它含有變異數，負責教 $v$；$\lambda$ 很小，讓平均仍然主要由 $L_{\text{simple}}$ 來教。`[補充]` 實務上 $L_{\text{vlb}}$ 的梯度只傳給 $v$，不傳給平均。這樣做的好處主要不是讓 $1000$ 步的樣本更好看，而是讓模型的概似分數變好，並且在只走幾十步生成時品質掉得少。

$L_{\text{vlb}}$ is KP03's full ELBO (negated as a loss); it contains the variance and teaches $v$, while the small $\lambda$ keeps the mean taught mainly by $L_{\text{simple}}$. `[Added]` In practice the gradient of $L_{\text{vlb}}$ is passed only to $v$, not to the mean. The main benefit is not prettier samples at $1000$ steps, but a better likelihood score and less loss of quality when generating with only a few dozen steps.

常見錯誤：以為學變異數就是讓網路輸出任意的變異數。網路只輸出插值係數 $v$，變異數被限制在 $\beta_t$ 與 $\tilde\beta_t$ 附近，不會亂跑。

Common mistake: thinking that learning the variance means the network outputs any variance it likes. The network only outputs the interpolation coefficient $v$, so the variance stays near $\beta_t$ and $\tilde\beta_t$ and cannot wander off.

### 2. 重要性抽樣：讓估計不要那麼吵（Importance sampling: making the estimate less noisy）

演算法 1 每次均勻抽一個 $t$，用那一步的損失估計「所有步的總和」。如果各步的損失大小差很多，這個估計就很吵：這次抽到大的、下次抽到小的，梯度跟著大起大落。Nichol 與 Dhariwal 發現，直接最佳化 $L_{\text{vlb}}$ 時正是如此，它的前幾步比後面大得多（KP05 的權重圖）。

Algorithm 1 draws one $t$ uniformly each time and uses that step's loss to estimate "the sum over all steps". If the losses of different steps differ a lot in size, this estimate is noisy: a large one this time, a small one next time, and the gradient swings with it. Nichol and Dhariwal found that this is exactly what happens when optimising $L_{\text{vlb}}$ directly, whose first few steps are far larger than the rest (KP05's weight figure).

解法是**重要性抽樣**（importance sampling）：讓損失大的步數比較常被抽到（抽到第 $t$ 步的機率是 $p_t$），再把抽到的值除以 $p_t$，估計仍然不偏。一個兩步的小例子 `[補充]`：總損失 $L_1+L_2$，其中 $L_1=10$、$L_2=0.1$，總和 $10.1$。

The fix is **importance sampling**: draw the steps with large losses more often (step $t$ is drawn with probability $p_t$), and divide the drawn value by $p_t$, so the estimate stays unbiased. A two-step example `[Added]`: the total loss is $L_1+L_2$ with $L_1=10$ and $L_2=0.1$, a total of $10.1$.

- **均勻抽樣**：各一半機率，估計值是 $2L_t$，也就是 $20$ 或 $0.2$；平均是 $10.1$，但變異數高達 $98.01$。
- **重要性抽樣**：取 $p_t$ 正比於 $L_t$，即 $p_1\approx0.990$、$p_2\approx0.0099$；估計值 $L_t/p_t$ 不管抽到哪一步都是 $10.1$，變異數是 $0$。

- **Uniform sampling**: each with probability one half, the estimate is $2L_t$, that is $20$ or $0.2$; the mean is $10.1$, but the variance is as high as $98.01$.
- **Importance sampling**: take $p_t$ proportional to $L_t$, so $p_1\approx0.990$ and $p_2\approx0.0099$; the estimate $L_t/p_t$ is $10.1$ whichever step is drawn, with variance $0$.

實際上不知道每一步確切的損失，所以用訓練過程中記下的歷史數值來估計：Nichol 與 Dhariwal 取 $p_t$ 正比於每一步最近損失平方平均的平方根。

In practice the exact loss of each step is unknown, so it is estimated from values recorded during training: Nichol and Dhariwal take $p_t$ proportional to the square root of the recent average squared loss of each step.

常見錯誤：以為重要性抽樣會讓損失偏向某些步。除以 $p_t$ 的目的就是抵銷抽樣的偏好，估計的平均不變，只是變得比較不吵。

Common mistake: thinking importance sampling biases the loss towards some steps. Dividing by $p_t$ exists precisely to cancel the sampling preference: the mean of the estimate is unchanged; it only becomes less noisy.

### 3. Cosine 排程（The cosine schedule）

KP02 的線性排程有個問題：它把資料抹掉得太快。從下圖左邊看，線性排程（藍線）到一半時，原資料的權重只剩約 $0.28$；最後三分之一的步數裡，資料幾乎完全看不到，這些步對學習幫助不大。Nichol 與 Dhariwal 提出的 cosine 排程直接規定 $\bar\alpha_t$ 的形狀（來源的式 (46)–(48)）：

KP02's linear schedule has a problem: it erases the data too quickly. As the left of the figure below shows, under the linear schedule (blue) the weight of the original data is only about $0.28$ at the halfway point; in the last third of the steps the data are almost invisible, and those steps help learning little. The cosine schedule of Nichol and Dhariwal directly prescribes the shape of $\bar\alpha_t$ (the source's equations (46)–(48)):

$$
\bar\alpha_t=\frac{f(t)}{f(0)},\qquad f(t)=\cos^2\!\left(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\right),\qquad \beta_t=1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}} .
$$

$f$ 從 $t=0$ 的接近 $1$ 平順地降到 $t=T$ 的 $0$。`[補充]` 小位移 $s=0.008$ 讓一開始的 $\beta_t$ 不要小到幾乎是 $0$（這裡 $\beta_1\approx4.1\times10^{-5}$）；最後幾步 $\bar\alpha_t$ 掉向 $0$ 時，式 (48) 的 $\beta_t$ 會衝向 $1$，所以實作上把 $\beta_t$ 上限設為 $0.999$（$T=1000$ 時只有最後一步被截到）。

$f$ falls smoothly from nearly $1$ at $t=0$ to $0$ at $t=T$. `[Added]` The small offset $s=0.008$ keeps the first $\beta_t$ from being almost $0$ (here $\beta_1\approx4.1\times10^{-5}$); in the last few steps, as $\bar\alpha_t$ heads to $0$, equation (48) sends $\beta_t$ towards $1$, so in practice $\beta_t$ is capped at $0.999$ (with $T=1000$ only the last step is capped).

![兩張並排的折線圖，橫軸都是 t/T 從 0 到 1。左圖標題「資料被抹掉的速度」，縱軸是根號 alpha bar，也就是 x0 的權重：藍色實線是線性排程，從 1 開始，大約在 t/T=0.3 到 0.5 之間快速下降，t/T=0.6 時只剩約 0.16，之後貼近 0；紅色虛線是 cosine 排程（s=0.008），下降得慢而平均，t/T=0.5 時還有約 0.7，直到 t/T=1 才降到 0。右圖標題「每步加的雜訊」，縱軸是 beta_t 的對數座標：藍線從 0.0001 平緩升到 0.02；紅色虛線從約 0.00004 開始，大部分時間比藍線低，在 t/T 約 0.9 以後超過藍線並急升，最後在 0.999 被截斷（Two line charts side by side, both with t/T from 0 to 1 horizontally. Left, titled "how fast the data are erased", with sqrt(alpha bar), the weight of x0, vertically: the solid blue linear schedule starts at 1, falls fast between about t/T = 0.3 and 0.5, is only about 0.16 at t/T = 0.6 and then hugs 0; the dashed red cosine schedule (s = 0.008) falls slowly and evenly, still about 0.7 at t/T = 0.5, and reaches 0 only at t/T = 1. Right, titled "noise added per step", with beta_t on a log scale: the blue line rises gently from 0.0001 to 0.02; the dashed red line starts at about 0.00004, stays below blue most of the time, overtakes it after about t/T = 0.9 and shoots up, finally capped at 0.999)](../assets/figures/lec14-kp06-schedules.png)

**圖說 `lec14-kp06-schedules`**〔功能：對照〕：左圖比「原資料還剩多少」。線性排程在 $t=260$ 時 $\bar\alpha_t$ 就降到 $0.5$，cosine 要到 $t=497$；第 $500$ 步時，原資料的權重線性只剩 $0.28$，cosine 還有 $0.70$。$\bar\alpha_t<0.01$（幾乎全是雜訊）的步數，線性佔 $32.7\%$，cosine 只佔 $6.5\%$。右圖是每一步加多少：cosine 在前中段加得比較少，把雜訊留到最後再加。

**Caption `lec14-kp06-schedules`** [Function: comparison]: the left chart compares "how much of the original is left". Under the linear schedule $\bar\alpha_t$ falls to $0.5$ by $t=260$, while the cosine one takes until $t=497$; at step $500$ the original's weight is only $0.28$ for linear but still $0.70$ for cosine. Steps with $\bar\alpha_t<0.01$ (almost pure noise) make up $32.7\%$ of the linear schedule but only $6.5\%$ of the cosine one. The right chart is how much each step adds: cosine adds less in the early and middle steps and saves the noise for the end.

來源特別提到，這對小尺寸影像（例如 $32\times32$、$64\times64$）幫助最大。`[補充]` 一個直覺的解釋是：小影像的像素少，加一點雜訊就看不出內容，線性排程後段浪費的步數更多；原論文的說法是線性排程的末段太吵、對樣本品質貢獻很少。

The source notes that this helps most for small images (for example $32\times32$ and $64\times64$). `[Added]` An intuitive explanation: small images have few pixels, so a little noise already hides the content, and the linear schedule wastes even more late steps; the original paper's point is that the end of the linear schedule is too noisy and contributes little to sample quality.

常見錯誤：以為 cosine 排程讓最後的 $\mathbf{x}_T$「比較不像雜訊」。兩種排程走到最後都幾乎是純雜訊（cosine 的 $\bar\alpha_T$ 甚至更小）；差別在**中間的步數**怎麼分配，cosine 讓更多步數落在「資料還看得到」的區間。

Common mistake: thinking the cosine schedule makes the final $\mathbf{x}_T$ "less like noise". Both schedules end almost in pure noise (the cosine $\bar\alpha_T$ is even smaller); the difference is how the **middle steps** are spread, with cosine putting more steps where the data are still visible.

### 4. PriorGrad：讓起點更像資料（PriorGrad: a start that looks more like the data）

DDPM 的生成一律從 $\mathcal{N}(\mathbf{0},I)$ 開始。如果資料本身的平均不是 $0$、各維度的變異數也不同，逆向過程就得先花力氣把整體位置和尺度搬過去。PriorGrad 用資料的平均 $\mu$ 與共變異數 $\Sigma$ 當起點：訓練時先把資料減掉平均，雜訊改從 $\mathcal{N}(\mathbf{0},\Sigma)$ 抽（來源的式 (49)），損失也改用以 $\Sigma$ 為尺度的距離（式 (50)）：

DDPM always starts generation from $\mathcal{N}(\mathbf{0},I)$. If the data's own mean is not $0$ and the dimensions have different variances, the backward process must first spend effort moving the overall location and scale into place. PriorGrad uses the data mean $\mu$ and covariance $\Sigma$ as the start: in training the data are first centred, the noise is drawn from $\mathcal{N}(\mathbf{0},\Sigma)$ (the source's equation (49)), and the loss uses a distance scaled by $\Sigma$ (equation (50)):

$$
\mathbf{x}_t=\sqrt{\bar\alpha_t}\,(\mathbf{x}_0-\mu)+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad\epsilon\sim\mathcal{N}(\mathbf{0},\Sigma);\qquad L=\mathbb{E}\big[(\epsilon-\epsilon_\theta)^\top\Sigma^{-1}(\epsilon-\epsilon_\theta)\big].
$$

生成時從 $\mathcal{N}(\mathbf{0},\Sigma)$ 出發，最後再把 $\mu$ 加回去。用來源展示的數字感受一下 `[補充]`：資料是 $\mathcal{N}(0.5,0.8^2)$。它和標準常態的 KL 散度是 $0.168$；減掉平均之後，它和 PriorGrad 的起點 $\mathcal{N}(0,0.8^2)$ 一模一樣，KL 是 $0$。起點越接近資料，逆向過程要做的事就越少。

Generation starts from $\mathcal{N}(\mathbf{0},\Sigma)$ and adds $\mu$ back at the end. Get a feel for it with the source demo's numbers `[Added]`: the data are $\mathcal{N}(0.5,0.8^2)$. Its KL divergence from the standard normal is $0.168$; after centring, it is identical to PriorGrad's start $\mathcal{N}(0,0.8^2)$, with KL $0$. The closer the start is to the data, the less the backward process has to do.

`[補充]` PriorGrad 原本用在語音合成：$\mu$、$\Sigma$ 不是整份資料的一組統計量，而是**依每一段條件**（例如每一段頻譜）算出來的，所以它本質上是一種條件式的起點。下一個知識點 KP07 會介紹更一般的條件式擴散。

`[Added]` PriorGrad was originally used in speech synthesis: $\mu$ and $\Sigma$ are not one set of statistics for the whole dataset but are computed **per condition** (for example per spectrogram segment), so it is essentially a conditional start. The next knowledge point, KP07, introduces conditional diffusion more generally.

常見錯誤：以為 PriorGrad 改了前向過程的每一步規則。前向的形式不變，只是把資料先置中、把雜訊的形狀換成 $\Sigma$；終點因此變成 $\mathcal{N}(\mathbf{0},\Sigma)$。

Common mistake: thinking PriorGrad changes the rule of every forward step. The form of the forward process is unchanged; the data are centred first and the noise is shaped by $\Sigma$, so the end point becomes $\mathcal{N}(\mathbf{0},\Sigma)$.

### 5. 加速生成：DDIM 與其他方法（Faster generation: DDIM and other methods）

DDPM 生成一個樣本要算 $1000$ 次網路。最直接的想法是跳著走：只用其中 $K$ 個步數。可是 DDPM 每一步都加隨機雜訊，跳太大時，「一步」已經不是近似常態的小步（KP03 第2節），品質會掉。

DDPM needs $1000$ network evaluations per sample. The most direct idea is to skip: use only $K$ of the steps. But DDPM adds random noise at every step, and with large jumps a "step" is no longer a small, nearly normal step (KP03 Section 2), so quality drops.

**DDIM**（denoising diffusion implicit model，Song 等人 2021）換一種走法。它沿用**同一個訓練好的** $\epsilon_\theta$，不用重新訓練：每一步先用 KP02 的閉式反推原資料的猜測 $\hat{\mathbf{x}}_0$，再用同一個雜訊猜測直接「放回」較早的第 $t'$ 步：

**DDIM** (denoising diffusion implicit model, Song and colleagues 2021) walks differently. It reuses **the same trained** $\epsilon_\theta$ with no retraining: at each step it first works back a guess $\hat{\mathbf{x}}_0$ of the original data with KP02's closed form, then "puts it back" directly at an earlier step $t'$ using the same noise guess:

$$
\hat{\mathbf{x}}_0=\frac{\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{\bar\alpha_t}},\qquad \mathbf{x}_{t'}=\sqrt{\bar\alpha_{t'}}\,\hat{\mathbf{x}}_0+\sqrt{1-\bar\alpha_{t'}}\,\epsilon_\theta(\mathbf{x}_t,t).
$$

這裡不加任何新雜訊（對應 $\eta=0$），所以同一個起點 $\mathbf{x}_T$ 永遠得到同一個結果。DDIM 的理論是：存在另一個不是馬可夫鏈的前向過程，每一步的分布 $q(\mathbf{x}_t\mid\mathbf{x}_0)$ 和 DDPM 完全一樣，所以 KP05 的訓練目標也一樣，同一個網路可以直接用；而它的逆向過程可以直接定義在一部分步數上，所以能跳步。$\eta$ 可以調回去加隨機性（完整公式這裡省略）；$\eta=1$ 時就是變異數取 $\tilde\beta$ 的 DDPM，下面實驗裡「跳步的 DDPM」正是 $\eta=1$。

No fresh noise is added here (corresponding to $\eta=0$), so the same start $\mathbf{x}_T$ always gives the same result. The theory behind DDIM: there is another, non-Markov forward process whose per-step distributions $q(\mathbf{x}_t\mid\mathbf{x}_0)$ are exactly those of DDPM, so KP05's training objective is the same and the same network can be used directly; and its backward process can be defined directly on a subset of the steps, so it can skip. $\eta$ can be turned up to add randomness back (the full formula is omitted here); at $\eta=1$ it is exactly DDPM with variance $\tilde\beta$, and the "skipping DDPM" in the experiment below is precisely $\eta=1$.

在一維兩堆資料上比一比 `[補充]`。為了只看「走法」本身的誤差，網路換成完美的雜訊猜測 $E[\epsilon\mid\mathbf{x}_t]$（KP04 第4節），兩種方法都從 $1000$ 步裡平均挑 $K$ 步走。DDPM 跳步時，每一跳的變異數用該跳對應的 $\tilde\beta$。每種設定生成 $200000$ 個樣本。距離用「把生成的樣本搬成真正的分布，平均每個要搬多遠」來量（一維的 Wasserstein 距離，和真分布精確比較）；同樣多的真資料樣本，這個距離也有約 $0.005$，是抽樣本身的誤差水準。

Compare them on the one-dimensional two-pile data `[Added]`. To see only the error of the "walk" itself, the network is replaced by the perfect noise guess $E[\epsilon\mid\mathbf{x}_t]$ (KP04 Section 4), and both methods take $K$ evenly spaced steps out of $1000$. When DDPM skips, the variance of each jump is the $\tilde\beta$ belonging to that jump. Each setting generates $200000$ samples. Distance is measured as "how far, on average, each generated sample must be moved to match the true distribution" (the one-dimensional Wasserstein distance, compared exactly with the true distribution); a real-data sample of the same size also has a distance of about $0.005$, the error level of sampling itself.

![一張雙對數座標的折線圖，橫軸是生成步數 K，標 5、10、20、50、100、1000；縱軸是到真分布的距離。紫色圓點實線是 DDPM（eta=1，隨機），綠色方點虛線是 DDIM（eta=0，確定性），灰色點線是同樣多真資料樣本的距離約 0.005。K=5 時兩者約 0.39 與 0.32；K=10 時都約 0.09；之後 DDIM 一直比 DDPM 低約一半：K=20 是 0.044 對 0.071，K=50 是 0.018 對 0.036，K=100 是 0.0093 對 0.022；K=1000 時兩者都在灰線附近或以下（A log-log line chart with the number of generation steps K, marked 5, 10, 20, 50, 100 and 1000, horizontally, and the distance to the true distribution vertically. The purple solid line with circles is DDPM (eta = 1, stochastic), the green dashed line with squares is DDIM (eta = 0, deterministic), and the grey dotted line is the distance of a real-data sample of the same size, about 0.005. At K = 5 the two are about 0.39 and 0.32; at K = 10 both are about 0.09; after that DDIM stays at about half of DDPM: 0.044 versus 0.071 at K = 20, 0.018 versus 0.036 at K = 50, 0.0093 versus 0.022 at K = 100; at K = 1000 both are near or below the grey line)](../assets/figures/lec14-kp06-few-steps.png)

**圖說 `lec14-kp06-few-steps`**〔功能：對照、精準度〕：同一個完美的雜訊猜測，只換走法。$K=1000$ 時兩者都到了抽樣誤差的水準（DDPM $0.0035$、DDIM $0.0015$，灰線 $0.0047$）。步數減少時，DDIM 掉得慢，距離大約只有 DDPM 的一半：$K=100$ 時 $0.0093$ 對 $0.022$，$K=50$ 時 $0.018$ 對 $0.036$，$K=20$ 時 $0.044$ 對 $0.071$。DDPM 跳步時的毛病是兩堆被擠得太窄（$K=20$ 時右堆標準差 $0.23$，真的是 $0.30$；DDIM 是 $0.27$）。$K=5$ 時兩者都不行（約 $13\%$ 到 $15\%$ 的樣本落在兩堆之間）。

**Caption `lec14-kp06-few-steps`** [Function: comparison, precision]: the same perfect noise guess, only the walk changes. At $K=1000$ both reach the sampling-error level (DDPM $0.0035$, DDIM $0.0015$, grey line $0.0047$). As steps are cut, DDIM degrades more slowly, with roughly half of DDPM's distance: $0.0093$ against $0.022$ at $K=100$, $0.018$ against $0.036$ at $K=50$, and $0.044$ against $0.071$ at $K=20$. DDPM's problem when skipping is that the piles get squeezed too narrow (at $K=20$ the right pile's standard deviation is $0.23$, versus the true $0.30$; DDIM gives $0.27$). At $K=5$ neither works (about $13\%$ to $15\%$ of samples fall between the piles).

來源還列了三個方向：

The source lists three more directions:

- **Consistency Models**：直接學一個函數，把同一條**確定性**生成軌跡（像 DDIM 那樣不加雜訊的路徑）上的任何一點 $\mathbf{x}_t$ 都映射到它的終點 $\mathbf{x}_0$；生成時一步、或少數幾步就到。
- **漸進式蒸餾**（progressive distillation）：讓一個學生模型用一步模仿老師模型的兩步，學會後再當老師，如此反覆把步數一半一半地減少。
- **D3PM**：把擴散用在離散資料（文字、類別），前向過程不再是加常態雜訊，而是依轉移矩陣把類別隨機換掉。

- **Consistency Models**: directly learn a function that maps any point $\mathbf{x}_t$ on a **deterministic** generation trajectory (a noise-free path like DDIM's) to its end point $\mathbf{x}_0$; generation then takes one or a few steps.
- **Progressive distillation**: a student model learns to do in one step what the teacher does in two; once learned, it becomes the teacher, and repeating this halves the number of steps again and again.
- **D3PM**: applies diffusion to discrete data (text, categories); the forward process no longer adds normal noise but randomly swaps categories according to a transition matrix.

常見錯誤：以為 DDIM 要重新訓練一個新模型。DDIM 改的只是生成的走法，用的是和 DDPM 同一個訓練好的 $\epsilon_\theta$。

Common mistake: thinking DDIM must train a new model. DDIM changes only the way generation walks, using the same trained $\epsilon_\theta$ as DDPM.

### 本單元重點整理（KP06 summary）

**一句話重點**：改良 DDPM 的四條路：讓網路在 $\beta_t$ 與 $\tilde\beta_t$ 之間學變異數（配混合損失與重要性抽樣）、用 cosine 排程讓更多步數留在資料還看得到的區間、用 PriorGrad 讓起點更像資料，以及用 DDIM 等方法少走步數；DDIM 不用重新訓練，在少步數時明顯比跳步的 DDPM 好。

**One-sentence summary**: four routes to improving DDPM: let the network learn the variance between $\beta_t$ and $\tilde\beta_t$ (with a hybrid loss and importance sampling), use the cosine schedule to keep more steps where the data are still visible, use PriorGrad to make the start look more like the data, and take fewer steps with DDIM and similar methods; DDIM needs no retraining and is clearly better than skipping DDPM when few steps are taken.

**白話說明**：同一份食譜可以從四個地方改：火候不要寫死，讓廚師自己微調（學變異數）；下料的節奏改得平均一點，不要前面就把味道蓋掉（cosine 排程）；起鍋前的底料先調得接近成品（PriorGrad）；以及練熟以後，一次多做幾個步驟（DDIM）。

**Plain explanation**: the same recipe can be improved in four places: don't hard-code the heat, let the cook fine-tune it (learned variance); add ingredients at a more even pace instead of drowning the flavour early (cosine schedule); start from a base already close to the finished dish (PriorGrad); and, once practised, do several steps at once (DDIM).

**正式定義**：學變異數 $\Sigma_\theta=\exp(v\log\beta_t+(1-v)\log\tilde\beta_t)$，損失 $L_{\text{simple}}+\lambda L_{\text{vlb}}$、$\lambda=0.001$。cosine 排程 $\bar\alpha_t=f(t)/f(0)$，$f(t)=\cos^2\!\big(\frac{t/T+s}{1+s}\frac{\pi}{2}\big)$，$s=0.008$，$\beta_t=\min\!\big(1-\bar\alpha_t/\bar\alpha_{t-1},\,0.999\big)$。PriorGrad：$\mathbf{x}_t=\sqrt{\bar\alpha_t}(\mathbf{x}_0-\mu)+\sqrt{1-\bar\alpha_t}\,\epsilon$，$\epsilon\sim\mathcal{N}(\mathbf{0},\Sigma)$。DDIM（$\eta=0$）：$\hat{\mathbf{x}}_0=(\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta)/\sqrt{\bar\alpha_t}$，$\mathbf{x}_{t'}=\sqrt{\bar\alpha_{t'}}\,\hat{\mathbf{x}}_0+\sqrt{1-\bar\alpha_{t'}}\,\epsilon_\theta$。

**Formal definition**: learned variance $\Sigma_\theta=\exp(v\log\beta_t+(1-v)\log\tilde\beta_t)$, loss $L_{\text{simple}}+\lambda L_{\text{vlb}}$ with $\lambda=0.001$. Cosine schedule $\bar\alpha_t=f(t)/f(0)$ with $f(t)=\cos^2\!\big(\frac{t/T+s}{1+s}\frac{\pi}{2}\big)$, $s=0.008$, and $\beta_t=\min\!\big(1-\bar\alpha_t/\bar\alpha_{t-1},\,0.999\big)$. PriorGrad: $\mathbf{x}_t=\sqrt{\bar\alpha_t}(\mathbf{x}_0-\mu)+\sqrt{1-\bar\alpha_t}\,\epsilon$ with $\epsilon\sim\mathcal{N}(\mathbf{0},\Sigma)$. DDIM ($\eta=0$): $\hat{\mathbf{x}}_0=(\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta)/\sqrt{\bar\alpha_t}$ and $\mathbf{x}_{t'}=\sqrt{\bar\alpha_{t'}}\,\hat{\mathbf{x}}_0+\sqrt{1-\bar\alpha_{t'}}\,\epsilon_\theta$.

**例子**：$\beta_t=0.2$、$\tilde\beta_t=0.1538$，$v=0.5$ 時變異數 $0.1754$。cosine 排程在 $t=497$ 才讓 $\bar\alpha_t$ 降到 $0.5$（線性是 $t=260$）。一維兩堆資料、完美雜訊猜測：$K=50$ 步時 DDIM 的距離 $0.018$，跳步 DDPM 是 $0.036$。

**Example**: with $\beta_t=0.2$ and $\tilde\beta_t=0.1538$, $v=0.5$ gives variance $0.1754$. The cosine schedule brings $\bar\alpha_t$ down to $0.5$ only at $t=497$ (linear: $t=260$). On the one-dimensional two-pile data with the perfect noise guess, at $K=50$ steps DDIM's distance is $0.018$ and skipping DDPM's is $0.036$.

**常見迷思**：

- 「學變異數就是讓網路輸出任意變異數。」錯，只輸出插值係數 $v$。
- 「重要性抽樣會讓損失偏向某些步。」錯，除以 $p_t$ 之後估計仍然不偏。
- 「cosine 排程讓 $\mathbf{x}_T$ 不那麼像雜訊。」錯，終點一樣是雜訊，差在中間步數的分配。
- 「PriorGrad 改了前向每一步的規則。」錯，只是置中資料並把雜訊換成 $\mathcal{N}(\mathbf{0},\Sigma)$。
- 「DDIM 要重新訓練。」錯，它沿用同一個 $\epsilon_\theta$，只改生成走法。

**Common misconceptions**:

- "Learning the variance means outputting any variance." Wrong: only the interpolation coefficient $v$ is output.
- "Importance sampling biases the loss towards some steps." Wrong: after dividing by $p_t$ the estimate stays unbiased.
- "The cosine schedule makes $\mathbf{x}_T$ less noise-like." Wrong: the end is noise either way; the difference is how the middle steps are spread.
- "PriorGrad changes every forward step's rule." Wrong: it only centres the data and replaces the noise by $\mathcal{N}(\mathbf{0},\Sigma)$.
- "DDIM needs retraining." Wrong: it reuses the same $\epsilon_\theta$ and changes only the generation walk.

**圖**：本單元有兩張圖。

- `lec14-kp06-schedules`：線性與 cosine 排程的原資料權重與每步雜訊量。
- `lec14-kp06-few-steps`：同一個完美雜訊猜測下，DDPM 與 DDIM 在不同步數的樣本品質。

**Figures** — this unit has two figures.

- `lec14-kp06-schedules`: the original-data weight and per-step noise of the linear and cosine schedules.
- `lec14-kp06-few-steps`: sample quality of DDPM and DDIM at different numbers of steps, with the same perfect noise guess.

### 小練習（Practice）

1. 承 KP04 的練習：$\beta_t=0.1$、$\bar\alpha_{t-1}=0.5$，$\tilde\beta_t\approx0.0909$。若網路輸出 $v=0.25$，學到的變異數是多少？它比較靠近哪一端？
2. 用 cosine 排程（$s=0.008$），算 $t=T/2$ 時的 $\bar\alpha_t$ 與 $\sqrt{\bar\alpha_t}$。
3. 兩步的總損失 $L_1+L_2$，$L_1=4$、$L_2=1$。分別算均勻抽樣與重要性抽樣（$p_t$ 正比於 $L_t$）的估計值與變異數。
4. 手算 DDIM 的一步：$x_t=0.5$、$\bar\alpha_t=0.5$、$\bar\alpha_{t'}=0.8$、網路猜 $\epsilon_\theta=0.3$。$\hat x_0$ 與 $x_{t'}$ 各是多少？

1. Following KP04's practice: $\beta_t=0.1$ and $\bar\alpha_{t-1}=0.5$, so $\tilde\beta_t\approx0.0909$. If the network outputs $v=0.25$, what is the learned variance? Which end is it closer to?
2. With the cosine schedule ($s=0.008$), compute $\bar\alpha_t$ and $\sqrt{\bar\alpha_t}$ at $t=T/2$.
3. A two-step total loss $L_1+L_2$ with $L_1=4$ and $L_2=1$. Compute the estimates and their variances under uniform sampling and under importance sampling ($p_t$ proportional to $L_t$).
4. Compute one DDIM step by hand: $x_t=0.5$, $\bar\alpha_t=0.5$, $\bar\alpha_{t'}=0.8$, and the network guesses $\epsilon_\theta=0.3$. What are $\hat x_0$ and $x_{t'}$?

**第 1 題解答** `[補充]`：$\exp(0.25\log0.1+0.75\log0.0909)\approx0.0931$，比較靠近 $\tilde\beta_t=0.0909$，因為 $v=0.25$ 只給 $\beta_t$ 四分之一的份量。

**Solution to question 1** `[Added]`: $\exp(0.25\log0.1+0.75\log0.0909)\approx0.0931$, closer to $\tilde\beta_t=0.0909$, because $v=0.25$ gives $\beta_t$ only a quarter of the share.

**第 2 題解答** `[補充]`：$f(T/2)=\cos^2\!\big(\frac{0.5+0.008}{1.008}\cdot\frac{\pi}{2}\big)$，除以 $f(0)$ 得 $\bar\alpha_{T/2}\approx0.4938$，$\sqrt{\bar\alpha_{T/2}}\approx0.7027$。對照圖：線性排程在同一點只剩約 $0.28$。

**Solution to question 2** `[Added]`: $f(T/2)=\cos^2\!\big(\frac{0.5+0.008}{1.008}\cdot\frac{\pi}{2}\big)$; dividing by $f(0)$ gives $\bar\alpha_{T/2}\approx0.4938$ and $\sqrt{\bar\alpha_{T/2}}\approx0.7027$. Compare the figure: the linear schedule leaves only about $0.28$ at the same point.

**第 3 題解答** `[補充]`：均勻抽樣的估計值是 $2L_t$，即 $8$ 或 $2$，平均 $5$，變異數 $\frac12(8-5)^2+\frac12(2-5)^2=9$。重要性抽樣取 $p_1=0.8$、$p_2=0.2$，估計值 $4/0.8=5$ 或 $1/0.2=5$，永遠是 $5$，變異數 $0$。

**Solution to question 3** `[Added]`: under uniform sampling the estimate is $2L_t$, that is $8$ or $2$, with mean $5$ and variance $\frac12(8-5)^2+\frac12(2-5)^2=9$. Importance sampling takes $p_1=0.8$ and $p_2=0.2$; the estimate is $4/0.8=5$ or $1/0.2=5$, always $5$, with variance $0$.

**第 4 題解答** `[補充]`：$\hat x_0=(0.5-\sqrt{0.5}\times0.3)/\sqrt{0.5}=(0.5-0.2121)/0.7071\approx0.4071$。$x_{t'}=\sqrt{0.8}\times0.4071+\sqrt{0.2}\times0.3\approx0.3641+0.1342=0.4983$。沒有抽任何新雜訊，所以再算一次答案也一樣。

**Solution to question 4** `[Added]`: $\hat x_0=(0.5-\sqrt{0.5}\times0.3)/\sqrt{0.5}=(0.5-0.2121)/0.7071\approx0.4071$. $x_{t'}=\sqrt{0.8}\times0.4071+\sqrt{0.2}\times0.3\approx0.3641+0.1342=0.4983$. No new noise is drawn, so computing it again gives the same answer.

### 小測驗（Quiz，來源原題）

來源 KP-10 到 KP-13 各有四題自我檢測，保留原題與全部選項；來源頁面附有答案與簡短解析，下面的解析由本講義依各節內容補寫。

The source's KP-10 to KP-13 each have four self-check questions, kept here with all their options; the source page gives the answers with short explanations, and the explanations below are written by this note from its own sections.

1. $\tilde\beta_t$ 的定義是什麼？
   - (A) $\beta_t/\alpha_t$
   - (B) $\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$
   - (C) $\sqrt{\bar\alpha_t}$
   - (D) $1-\alpha_t$
2. 學習 covariance 時，兩個候選值在哪個空間做插值？
   - (A) 原始空間
   - (B) log 空間
   - (C) 平方根空間
   - (D) 倒數空間
3. 式 (45) 中的 $\lambda$ 作用是什麼？
   - (A) 控制 full KL 項的權重
   - (B) 學習率
   - (C) noise schedule 參數
   - (D) batch size
4. 學習 covariance 的向量 $v$ 是誰輸出的？
   - (A) 資料集
   - (B) DDPM 網路
   - (C) noise schedule
   - (D) optimizer
5. 原版 DDPM 的梯度雜訊主要來自哪裡？
   - (A) 學習率太大
   - (B) 時間步 $t$ 的均勻抽樣
   - (C) 網路太深
   - (D) 資料太少
6. 降低梯度雜訊的方法是什麼？
   - (A) 增加 batch size
   - (B) importance sampling
   - (C) 減少時間步
   - (D) 固定 covariance
7. Cosine schedule 的 $f(t)$ 形式是什麼？
   - (A) $\sin(t)$
   - (B) $\cos(\frac{t/T+s}{1+s}\frac{\pi}{2})^2$
   - (C) $1-t/T$
   - (D) $e^{-t}$
8. Cosine schedule 主要改善什麼問題？
   - (A) 訓練收斂太慢
   - (B) 小圖解析度下 $x_T$ 過度雜訊
   - (C) 網路參數太多
   - (D) 推論速度太慢
9. PriorGrad 的主要動機是什麼？
   - (A) 加速推論
   - (B) 資料分布可能不是標準高斯
   - (C) 減少時間步
   - (D) 學習 covariance
10. 式 (49) 中，$x_0$ 被怎麼處理？
    - (A) 加上均值
    - (B) 減去均值
    - (C) 除以變異數
    - (D) 不變
11. 式 (50) 中的 norm 是什麼？
    - (A) L2 norm
    - (B) Mahalanobis norm with $\Sigma^{-1}$
    - (C) L1 norm
    - (D) cross-entropy
12. PriorGrad 的雜訊 $\epsilon$ 從什麼分布取樣？
    - (A) $\mathcal{N}(0,I)$
    - (B) $\mathcal{N}(0,\Sigma)$
    - (C) 均勻分布
    - (D) 泊松分布
13. DDIM 的主要貢獻是什麼？
    - (A) 學習 covariance
    - (B) 用確定性隱式過程減少取樣步數
    - (C) 改成 GAN
    - (D) 只處理離散資料
14. Consistency Models 的核心想法是什麼？
    - (A) 預測雜訊
    - (B) 直接把任意 $x_t$ 映射到 $x_0$
    - (C) 學習 covariance
    - (D) 增加時間步
15. Distillation 如何加速 diffusion？
    - (A) 減少訓練資料
    - (B) 把教師模型的知識蒸餾到學生模型
    - (C) 改變 noise schedule
    - (D) 增加 batch size
16. D3PM 與原版 DDPM 的主要差異是什麼？
    - (A) 使用 categorical transition 處理離散資料
    - (B) 只生成連續像素
    - (C) 不需要 neural network
    - (D) 使用 VAE

1. What is the definition of $\tilde\beta_t$?
   - (A) $\beta_t/\alpha_t$
   - (B) $\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$
   - (C) $\sqrt{\bar\alpha_t}$
   - (D) $1-\alpha_t$
2. When learning the covariance, in what space are the two candidates interpolated?
   - (A) The original space
   - (B) Log space
   - (C) Square-root space
   - (D) Reciprocal space
3. What does $\lambda$ in equation (45) do?
   - (A) Controls the weight of the full KL term
   - (B) It is the learning rate
   - (C) It is a noise-schedule parameter
   - (D) It is the batch size
4. Who outputs the vector $v$ used to learn the covariance?
   - (A) The dataset
   - (B) The DDPM network
   - (C) The noise schedule
   - (D) The optimizer
5. Where does the gradient noise of the original DDPM mainly come from?
   - (A) Too large a learning rate
   - (B) Uniform sampling of the time step $t$
   - (C) Too deep a network
   - (D) Too little data
6. What is the method for reducing gradient noise?
   - (A) A larger batch size
   - (B) Importance sampling
   - (C) Fewer time steps
   - (D) A fixed covariance
7. What is the form of the cosine schedule's $f(t)$?
   - (A) $\sin(t)$
   - (B) $\cos(\frac{t/T+s}{1+s}\frac{\pi}{2})^2$
   - (C) $1-t/T$
   - (D) $e^{-t}$
8. What problem does the cosine schedule mainly improve?
   - (A) Training converges too slowly
   - (B) $x_T$ is too noisy at small image resolutions
   - (C) Too many network parameters
   - (D) Inference is too slow
9. What is PriorGrad's main motivation?
   - (A) Faster inference
   - (B) The data distribution may not be a standard Gaussian
   - (C) Fewer time steps
   - (D) Learning the covariance
10. In equation (49), how is $x_0$ treated?
    - (A) The mean is added
    - (B) The mean is subtracted
    - (C) It is divided by the variance
    - (D) It is unchanged
11. What norm is used in equation (50)?
    - (A) The L2 norm
    - (B) The Mahalanobis norm with $\Sigma^{-1}$
    - (C) The L1 norm
    - (D) Cross-entropy
12. From what distribution is PriorGrad's noise $\epsilon$ drawn?
    - (A) $\mathcal{N}(0,I)$
    - (B) $\mathcal{N}(0,\Sigma)$
    - (C) A uniform distribution
    - (D) A Poisson distribution
13. What is DDIM's main contribution?
    - (A) Learning the covariance
    - (B) Reducing sampling steps with a deterministic implicit process
    - (C) Turning it into a GAN
    - (D) Handling only discrete data
14. What is the core idea of Consistency Models?
    - (A) Predict the noise
    - (B) Map any $x_t$ directly to $x_0$
    - (C) Learn the covariance
    - (D) Add more time steps
15. How does distillation speed up diffusion?
    - (A) Less training data
    - (B) Distil a teacher model's knowledge into a student model
    - (C) Change the noise schedule
    - (D) A larger batch size
16. What is the main difference between D3PM and the original DDPM?
    - (A) It uses categorical transitions for discrete data
    - (B) It generates only continuous pixels
    - (C) It needs no neural network
    - (D) It uses a VAE

**解析** `[補充]`：

- 第 1 至 4 題依序選 (B)、(B)、(A)、(B)：見第1節。對數空間的好處是變異數永遠為正、兩端差好幾倍時插值較均勻；$\lambda$ 是混合損失裡 $L_{\text{vlb}}$ 的權重；$v$ 是網路多輸出的一個向量。
- 第 5 題來源答案是 (B)：見第2節。說得精確一點，吵的是直接最佳化 $L_{\text{vlb}}$ 時，各步損失大小差太多，均勻抽 $t$ 讓估計大起大落；用 $L_{\text{simple}}$ 訓練時這個問題小得多（見「來源更正」）。
- 第 6 題選 (B)：見第2節的兩步例子。
- 第 7、8 題選 (B)、(B)：見第3節。第 8 題說「$x_T$ 過度雜訊」，精確地說是線性排程讓後段太多步數幾乎只剩雜訊，小影像尤其明顯。
- 第 9 至 12 題都選 (B)：見第4節。
- 第 13 題選 (B)：見第5節；DDIM 用同一個訓練好的網路，不用重新訓練。
- 第 14、15 題選 (B)、(B)，第 16 題選 (A)：見第5節的列表。

**Explanations** `[Added]`:

- Questions 1 to 4: B, B, A, B in turn; see Section 1. Log space keeps the variance positive and interpolates more evenly when the ends differ several-fold; $\lambda$ is the weight of $L_{\text{vlb}}$ in the hybrid loss; $v$ is an extra vector output by the network.
- Question 5: the source's answer is B; see Section 2. More precisely, the noisy case is optimising $L_{\text{vlb}}$ directly, where the per-step losses differ so much that drawing $t$ uniformly makes the estimate swing; training with $L_{\text{simple}}$ has this problem much less (see "Corrections to the source").
- Question 6: B; see Section 2's two-step example.
- Questions 7 and 8: B and B; see Section 3. Question 8's "$x_T$ is too noisy" means, more precisely, that the linear schedule leaves too many late steps as almost pure noise, especially for small images.
- Questions 9 to 12: all B; see Section 4.
- Question 13: B; see Section 5; DDIM uses the same trained network without retraining.
- Questions 14 and 15: B and B; question 16: A; see the list in Section 5.

### 留給下一單元的問題（The question for the next unit）

到目前為止，模型只會「隨便生成一筆像資料的東西」。如果想指定要生成什麼，例如「一隻貓」、「照著這張草圖」、「依照這句文字描述」，要怎麼把條件放進擴散模型？下一個知識點（本週 KP07）介紹條件式擴散：ILVR、分類器引導、無分類器引導與 cross-attention。

So far the model can only "generate some example that looks like the data". To specify what to generate, such as "a cat", "following this sketch" or "matching this text description", how is the condition put into a diffusion model? The next knowledge point (this week's KP07) introduces conditional diffusion: ILVR, classifier guidance, classifier-free guidance and cross-attention.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第15章「Diffusion Models」的 KP-10「Learned Covariance」、KP-11「梯度雜訊降低與 Cosine Noise Schedule」、KP-12「PriorGrad」與 KP-13「加速變體」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch15-diffusion-models/index.html ，讀取日期 2026-11-15）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from KP-10, "Learned covariance", KP-11, "Gradient-noise reduction and the cosine noise schedule", KP-12, "PriorGrad", and KP-13, "Accelerated variants", of Chapter 15, "Diffusion Models", of CCC-LNS IDL_115 (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch15-diffusion-models/index.html , accessed 2026-11-15). Content not in the source is marked `[Added]` where it appears.

- 來源說「學習 covariance 可以提升品質」。Nichol 與 Dhariwal 的結果是：學變異數主要改善概似分數，並讓少步數生成時品質掉得少；走完全部步數時，樣本品質和固定變異數差不多。本講義第1節照此寫。
- 來源說原版 DDPM「均勻抽 $t$ 讓某些時間步的梯度雜訊很大，導致訓練不穩定」。梯度很吵的情形出現在直接最佳化 $L_{\text{vlb}}$ 時；DDPM 實際用的 $L_{\text{simple}}$ 訓練是穩定的（KP05）。重要性抽樣是為了讓 $L_{\text{vlb}}$ 可以直接訓練。小測驗第 5 題照錄原題，解析中說明。
- 來源說線性排程「在 $t$ 接近 $T$ 時會讓 $x_T$ 太雜訊」。兩種排程的終點都幾乎是純雜訊；線性排程的問題是資料抹得太快，後段約三分之一的步數幾乎只剩雜訊（本講義算出 $32.7\%$，cosine 是 $6.5\%$）。
- 來源把 PriorGrad 寫成用「訓練資料的均值與 covariance」。原論文用在語音合成，$\mu$、$\Sigma$ 依每段條件計算，是條件式的起點；本講義第4節補充。
- 來源說漸進式蒸餾把知識蒸餾到「小模型」。漸進式蒸餾的學生通常和老師一樣大，省下的是**步數**而不是模型大小；本講義第5節照此寫。
- 來源說 DDIM「只要改變一個參數」就能少走步數。DDIM 的關鍵是換了一個非馬可夫的前向過程，並沿用同一個訓練好的網路；$\eta$ 控制隨機性，少走步數靠的是跳步。本講義第5節補充，並用一維實驗比較 DDPM 與 DDIM。
- 本講義補上：$\tilde\beta_t/\beta_t$ 隨步數的數值、對數插值的數字與為何用對數、$L_{\text{vlb}}$ 只教變異數、重要性抽樣的兩步例子、cosine 排程中 $s$ 與截斷 $0.999$ 的用意及與線性排程的數值比較圖、PriorGrad 的 KL 例子、DDIM 的更新式與理論、DDPM 與 DDIM 在少步數下的比較圖、四題小練習與解答、小測驗的解析。
- 取捨：來源的互動展示不重做；Consistency Models、蒸餾與 D3PM 只做簡介；符號加粗與否的取捨同 KP02。

- The source says "learning the covariance improves quality". Nichol and Dhariwal's result is that learning the variance mainly improves the likelihood score and makes quality drop less when generating with few steps; with all steps, sample quality is about the same as with a fixed variance. Section 1 follows this.
- The source says the original DDPM's "uniform sampling of $t$ makes some time steps' gradients very noisy, making training unstable". Very noisy gradients arise when optimising $L_{\text{vlb}}$ directly; training with the $L_{\text{simple}}$ that DDPM actually uses is stable (KP05). Importance sampling is there to make $L_{\text{vlb}}$ trainable directly. Quiz question 5 keeps the original wording, with a note in the explanation.
- The source says the linear schedule "makes $x_T$ too noisy as $t$ approaches $T$". Both schedules end almost in pure noise; the linear schedule's problem is that it erases the data too quickly, so roughly the last third of the steps are almost pure noise (this note computes $32.7\%$, against $6.5\%$ for cosine).
- The source describes PriorGrad as using "the mean and covariance of the training data". The original paper is for speech synthesis, with $\mu$ and $\Sigma$ computed per condition, making it a conditional start; Section 4 adds this.
- The source says progressive distillation distils knowledge into "a small model". In progressive distillation the student is usually as large as the teacher; what is saved is the number of **steps**, not the model size. Section 5 follows this.
- The source says DDIM reduces steps "by changing just one parameter". DDIM's key is a different, non-Markov forward process that reuses the same trained network; $\eta$ controls randomness, and fewer steps come from skipping. Section 5 adds this and compares DDPM and DDIM in a one-dimensional experiment.
- This note adds: the values of $\tilde\beta_t/\beta_t$ across steps; the numbers of the log interpolation and why logs are used; $L_{\text{vlb}}$ teaching only the variance; the two-step importance-sampling example; the purpose of $s$ and the $0.999$ cap in the cosine schedule, with a numerical comparison figure against the linear schedule; the PriorGrad KL example; the DDIM update and its theory; the comparison figure of DDPM and DDIM with few steps; four practice questions with solutions; and the explanations of the quiz.
- Choices: the source's interactive demos are not rebuilt; Consistency Models, distillation and D3PM are only introduced briefly. Bold versus plain symbols follow the choice made in KP02.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 學習變異數 | learned variance |
| 對數空間插值 | interpolation in log space |
| 幾何平均 | geometric mean |
| 混合損失 | hybrid loss |
| 變分下界損失 | variational lower bound loss ($L_{\text{vlb}}$) |
| 重要性抽樣 | importance sampling |
| 不偏估計 | unbiased estimate |
| 梯度雜訊 | gradient noise |
| 餘弦排程 | cosine schedule |
| 線性排程 | linear schedule |
| 資料相關的先驗 | data-dependent prior (PriorGrad) |
| 馬氏範數 | Mahalanobis norm |
| 去雜訊擴散隱式模型 | denoising diffusion implicit model (DDIM) |
| 確定性取樣 | deterministic sampling |
| 非馬可夫過程 | non-Markov process |
| Wasserstein 距離 | Wasserstein distance |
| 一致性模型 | consistency model |
| 漸進式蒸餾 | progressive distillation |
| 離散去雜訊擴散機率模型 | discrete denoising diffusion probabilistic model (D3PM) |
| 轉移矩陣 | transition matrix |
| 概似 | likelihood |
| 去雜訊擴散機率模型 | denoising diffusion probabilistic model (DDPM) |
| 生成對抗網路 | generative adversarial network (GAN) |
| 變分自編碼器 | variational autoencoder (VAE) |
| 均勻抽樣 | uniform sampling |
| 證據下界 | evidence lower bound (ELBO) |
| KL 散度 | Kullback–Leibler (KL) divergence |
| 迭代潛在變數精煉（KP07） | iterative latent variable refinement (ILVR) |
