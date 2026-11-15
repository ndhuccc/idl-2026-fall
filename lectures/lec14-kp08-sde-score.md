# 第14週 KP08（選讀）：SDE、score matching 與全章總結（Optional: SDEs, score matching and a chapter summary）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第14週「深度生成模型（三）：擴散模型」的第八個知識點（KP08），屬於**選讀**。KP01 到 KP07 都用離散的步數 $t=1,\ldots,T$。這一節問：如果步數無限多、每一步無限小，擴散模型會變成什麼？答案是一組隨機微分方程，而網路猜的雜訊，其實是在估計一個叫做「分數」的量。最後把整章和前兩週的 VAE、GAN 放在一起比較。

This is the eighth knowledge point (KP08) of Week 14, "Deep generative models (3): diffusion models", and it is **optional**. KP01 to KP07 all used discrete steps $t=1,\ldots,T$. This note asks: if there are infinitely many steps, each infinitely small, what does a diffusion model become? The answer is a pair of stochastic differential equations, and the noise the network guesses is really an estimate of a quantity called the "score". Finally the whole chapter is compared with the VAE and GAN of the past two weeks.

先備知識是本週 KP02 到 KP07，尤其是 KP07 第2節的「分數 $\approx-\epsilon_\theta/\sqrt{1-\bar\alpha_t}$」與 KP06 的 DDIM；還會用到微積分的導數與「$dt$ 是很小的時間」的直覺。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第15章 KP-15（第 5 節）與第 6–7 節的總結改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec14-kp08-figures.py` 算出，可以自己重算。

The prerequisites are this week's KP02 to KP07, especially "score $\approx-\epsilon_\theta/\sqrt{1-\bar\alpha_t}$" from KP07 Section 2 and DDIM from KP06; calculus derivatives and the intuition that "$dt$ is a tiny time" are also used. Every term and symbol is defined when it first appears. This note is rewritten from KP-15 (Section 5) and the summary of Sections 6–7 of Chapter 15 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec14-kp08-figures.py`, so you can recompute it yourself.

## KP08（選讀）SDE、score matching 與全章總結（Optional: SDEs, score matching and a chapter summary）

### 學習目標（Learning objectives）

讀完之後，你能說明 DDPM 的一步在步伐很小時，怎麼變成前向隨機微分方程 $d\mathbf{x}=-\frac12\beta(t)\mathbf{x}\,dt+\sqrt{\beta(t)}\,d\mathbf{w}$，並用 Euler–Maruyama 法手算一步。你能寫出逆向隨機微分方程，說明分數在其中的角色，並分辨它和機率流常微分方程的差別。你能說明 denoising score matching 的目標為什麼等於 DDPM 的猜雜訊損失。你也能用一張表比較 VAE、GAN、GMMN 與擴散模型的優缺點。

After reading, you can explain how one DDPM step, when steps are tiny, becomes the forward stochastic differential equation $d\mathbf{x}=-\frac12\beta(t)\mathbf{x}\,dt+\sqrt{\beta(t)}\,d\mathbf{w}$, and compute one Euler–Maruyama step by hand. You can write the reverse stochastic differential equation, explain the score's role in it, and tell it apart from the probability-flow ordinary differential equation. You can explain why the target of denoising score matching equals DDPM's noise-guessing loss. You can also compare the strengths and weaknesses of VAEs, GANs, GMMNs and diffusion models in one table.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $t\in[0,1]$ | 連續的時間，$t=0$ 是資料、$t=1$ 是雜訊 | continuous time; $t=0$ is data and $t=1$ is noise |
| $dt$ | 很小的一段時間 | a tiny span of time |
| $\beta(t)$ | 連續版的雜訊速率 | the continuous noise rate |
| $\mathbf{w}$、$d\mathbf{w}$ | 維納過程（布朗運動）與它在 $dt$ 內的增量，$d\mathbf{w}\sim\mathcal{N}(\mathbf{0},dt\,I)$ | the Wiener process (Brownian motion) and its increment over $dt$, $d\mathbf{w}\sim\mathcal{N}(\mathbf{0},dt\,I)$ |
| $f(\mathbf{x},t)$ | 漂移係數：確定的推力 | the drift coefficient: the deterministic push |
| $g(t)$ | 擴散係數：隨機抖動的大小 | the diffusion coefficient: the size of the random jitter |
| $p_t(\mathbf{x})$ | 時間 $t$ 時資料的分布（KP07 的 $q_t$） | the data distribution at time $t$ ($q_t$ in KP07) |
| $\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$ | 分數（KP07） | the score (KP07) |
| $s_\theta(\mathbf{x},t)$ | 網路對分數的估計 | the network's estimate of the score |
| $\bar{\mathbf{w}}$ | 時間倒著走的維納過程 | a Wiener process running backward in time |
| $h$ | 數值方法的步長 | the step size of a numerical method |

### 1. 從一步到連續：前向 SDE（From steps to continuous time: the forward SDE）

DDPM 的一步是 $\mathbf{x}_t=\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}+\sqrt{\beta_t}\,\epsilon$。當 $\beta_t$ 很小時，$\sqrt{1-\beta_t}\approx1-\frac{\beta_t}{2}$（$\beta_t=0.02$ 時兩者只差 $5\times10^{-5}$；但 KP01 手算用的 $0.36$ 就差到 $0.02$，不適用）。代進去並移項：

A DDPM step is $\mathbf{x}_t=\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}+\sqrt{\beta_t}\,\epsilon$. When $\beta_t$ is small, $\sqrt{1-\beta_t}\approx1-\frac{\beta_t}{2}$ (at $\beta_t=0.02$ the two differ by only $5\times10^{-5}$; but at KP01's hand-calculation value $0.36$ they differ by $0.02$, so it does not apply). Substitute and rearrange:

$$
\mathbf{x}_t-\mathbf{x}_{t-1}\approx-\tfrac12\beta_t\,\mathbf{x}_{t-1}+\sqrt{\beta_t}\,\epsilon .
$$

把 $T$ 步排在時間 $[0,1]$ 上，每步長 $dt=1/T$，並把每步的雜訊量寫成 $\beta_t=\beta(t)\,dt$。這時 $\sqrt{\beta_t}\,\epsilon=\sqrt{\beta(t)}\,\sqrt{dt}\,\epsilon$，而 $\sqrt{dt}\,\epsilon$ 正是維納過程在 $dt$ 內的增量 $d\mathbf{w}$。$T\to\infty$ 時得到來源的式 (52)：

Lay the $T$ steps on time $[0,1]$, each of length $dt=1/T$, and write each step's noise amount as $\beta_t=\beta(t)\,dt$. Then $\sqrt{\beta_t}\,\epsilon=\sqrt{\beta(t)}\,\sqrt{dt}\,\epsilon$, and $\sqrt{dt}\,\epsilon$ is exactly the increment $d\mathbf{w}$ of a Wiener process over $dt$. As $T\to\infty$ this gives the source's equation (52):

$$
d\mathbf{x}=-\tfrac12\beta(t)\,\mathbf{x}\,dt+\sqrt{\beta(t)}\,d\mathbf{w}.
$$

這叫**隨機微分方程**（stochastic differential equation，SDE）。一般寫成 $d\mathbf{x}=f(\mathbf{x},t)\,dt+g(t)\,d\mathbf{w}$（式 (53)）：$f$ 是**漂移**，確定地把 $\mathbf{x}$ 往某處推（這裡是往 $0$ 拉）；$g$ 是**擴散係數**，決定隨機抖動多大。

This is called a **stochastic differential equation** (SDE). In general it is written $d\mathbf{x}=f(\mathbf{x},t)\,dt+g(t)\,d\mathbf{w}$ (equation (53)): $f$ is the **drift**, which deterministically pushes $\mathbf{x}$ somewhere (here, towards $0$); $g$ is the **diffusion coefficient**, which sets how large the random jitter is.

`[補充]` DDPM 的排程（$\beta_t$ 從 $0.0001$ 到 $0.02$、$T=1000$）對應 $\beta(t)=0.1+19.9\,t$，這時 $\bar\alpha(t)=\exp\!\big(-\int_0^t\beta(u)\,du\big)=\exp\!\big(-(0.1\,t+9.95\,t^2)\big)$，因為 $\log\prod(1-\beta_i)\approx-\sum\beta_i$。和離散版比：$t=0.3$（第 $300$ 步）是 $0.3963$ 對 $0.3964$，$t=0.6$ 是 $0.0262$ 對 $0.0259$，幾乎一樣。

`[Added]` The DDPM schedule ($\beta_t$ from $0.0001$ to $0.02$, $T=1000$) corresponds to $\beta(t)=0.1+19.9\,t$, giving $\bar\alpha(t)=\exp\!\big(-\int_0^t\beta(u)\,du\big)=\exp\!\big(-(0.1\,t+9.95\,t^2)\big)$, because $\log\prod(1-\beta_i)\approx-\sum\beta_i$. Compared with the discrete version: at $t=0.3$ (step $300$) it is $0.3963$ against $0.3964$, and at $t=0.6$ it is $0.0262$ against $0.0259$, almost identical.

SDE 通常用 **Euler–Maruyama 法**一小步一小步算：$\mathbf{x}\leftarrow\mathbf{x}+f\,h+g\sqrt{h}\,\mathbf{z}$，$\mathbf{z}\sim\mathcal{N}(\mathbf{0},I)$，$h$ 是步長。它就是把微分方程的 $dt$ 換成小步長 $h$、把 $d\mathbf{w}$ 換成 $\sqrt{h}\,\mathbf{z}$。用它把 KP01 的兩堆資料往前推 $1000$ 步 `[補充]`：$t=0.3$、$0.6$、$1$ 時的標準差是 $1.485$、$1.041$、$0.999$，理論值 $1.492$、$1.040$、$1.000$。

SDEs are usually solved step by small step with the **Euler–Maruyama method**: $\mathbf{x}\leftarrow\mathbf{x}+f\,h+g\sqrt{h}\,\mathbf{z}$ with $\mathbf{z}\sim\mathcal{N}(\mathbf{0},I)$ and step size $h$. It simply replaces the $dt$ of the differential equation with a small step $h$ and $d\mathbf{w}$ with $\sqrt{h}\,\mathbf{z}$. Push KP01's two piles forward $1000$ steps with it `[Added]`: at $t=0.3$, $0.6$ and $1$ the standard deviations are $1.485$, $1.041$ and $0.999$, against theoretical values $1.492$, $1.040$ and $1.000$.

常見錯誤：以為 $d\mathbf{w}$ 的大小和 $dt$ 成正比。維納增量的標準差是 $\sqrt{dt}$，比 $dt$ 大得多；這正是為什麼 DDPM 每步的雜訊乘的是 $\sqrt{\beta_t}$，不是 $\beta_t$（KP02 第1節）。

Common mistake: thinking the size of $d\mathbf{w}$ is proportional to $dt$. A Wiener increment has standard deviation $\sqrt{dt}$, much larger than $dt$; this is exactly why DDPM's per-step noise is multiplied by $\sqrt{\beta_t}$, not $\beta_t$ (KP02 Section 1).

### 2. 逆向 SDE：分數帶路（The reverse SDE: the score leads the way）

數學上有一個結果（Anderson 1982）：前向 SDE 有一個時間倒著走的版本，走出來的分布和前向完全對應（來源的式 (54)）：

A mathematical result (Anderson 1982) says the forward SDE has a version running backward in time whose distributions correspond exactly to the forward ones (the source's equation (54)):

$$
d\mathbf{x}=\big[f(\mathbf{x},t)-g(t)^2\,\nabla_{\mathbf{x}}\log p_t(\mathbf{x})\big]\,dt+g(t)\,d\bar{\mathbf{w}},
$$

時間從 $t=1$ 走回 $t=0$（所以 $dt$ 是負的）。和前向比，漂移多了一項 $-g^2\nabla\log p_t$：分數指向機率高的地方，倒著走時這一項就把樣本往資料拉。式子裡唯一不知道的就是分數；只要能估計分數，就能從雜訊生成資料。

Time runs from $t=1$ back to $t=0$ (so $dt$ is negative). Compared with the forward equation, the drift has an extra term $-g^2\nabla\log p_t$: the score points towards high probability, and running backward this term pulls the sample towards the data. The only unknown in the equation is the score; if the score can be estimated, data can be generated from noise.

`[補充]` 還有一個不加隨機性的版本，叫**機率流常微分方程**（probability-flow ODE）：

`[Added]` There is also a version with no randomness, called the **probability-flow ordinary differential equation** (probability-flow ODE):

$$
d\mathbf{x}=\big[f(\mathbf{x},t)-\tfrac12\,g(t)^2\,\nabla_{\mathbf{x}}\log p_t(\mathbf{x})\big]\,dt .
$$

它每個時間點的分布和逆向 SDE 一樣，但沒有 $d\bar{\mathbf{w}}$，所以同一個起點永遠走到同一個終點。KP06 的 DDIM（$\eta=0$）經過變數代換後，可以看成這條 ODE 的一種離散化（Song 等人 2021）。

Its distribution at every time is the same as the reverse SDE's, but it has no $d\bar{\mathbf{w}}$, so the same start always reaches the same end. KP06's DDIM ($\eta=0$) can be seen, after a change of variables, as one discretisation of this ODE (Song and colleagues 2021).

用兩堆資料比一比 `[補充]`：這份資料的分數算得出來（KP07 第2節），所以兩條方程都直接用精確的分數 $s$。起點從 $\mathcal{N}(0,1)$ 抽，時間從 $t=1$ 倒著走 $1000$ 步到 $0$，步長取正的 $h$：逆向 SDE 用 $\mathbf{x}\leftarrow\mathbf{x}-[f-g^2 s]\,h+g\sqrt{h}\,\mathbf{z}$（Euler–Maruyama 法）；ODE 用同一式但去掉雜訊項、把 $g^2$ 換成 $\tfrac12g^2$（就是普通的 Euler 法）。

Compare them on the two piles `[Added]`: the score of these data can be computed (KP07 Section 2), so both equations use the exact score $s$. Starts are drawn from $\mathcal{N}(0,1)$, and time runs backward from $t=1$ to $0$ in $1000$ steps with a positive step $h$: the reverse SDE uses $\mathbf{x}\leftarrow\mathbf{x}-[f-g^2 s]\,h+g\sqrt{h}\,\mathbf{z}$ (Euler–Maruyama), and the ODE uses the same update without the noise term and with $\tfrac12g^2$ in place of $g^2$ (plain Euler).

![兩張並排的折線圖，橫軸是時間 t，從 1 往右遞減到 0，灰色虛線標出 2 與 -2。左圖「逆向 SDE：同樣 6 個起點、跑兩次（深紫與淺紫）」：12 條線都抖動得很厲害，深紫的 6 條最後都停在 2 附近，淺紫的 6 條有 4 條停在 2 附近、2 條停在 -2 附近。右圖「機率流 ODE：每個起點一條平滑的路徑」：6 條綠線幾乎水平地走到 t 約 0.5，之後平滑地彎向 -2 或 2，起點較低的兩條到 -2.22 與 -1.85，其餘四條到 1.47、1.78、2.10、2.33，上下順序始終不變（Two line charts side by side, with time t decreasing from 1 on the left to 0 on the right, and grey dashed lines at 2 and -2. Left, "reverse SDE: same 6 starts, two runs (dark and light purple)": all 12 lines jitter strongly; the 6 dark ones all end near 2, while of the 6 light ones 4 end near 2 and 2 near -2. Right, "probability-flow ODE: one smooth path per start": the 6 green lines run almost flat until about t = 0.5, then bend smoothly towards -2 or 2; the two lowest starts end at -2.22 and -1.85, the other four at 1.47, 1.78, 2.10 and 2.33, and their vertical order never changes)](../assets/figures/lec14-kp08-paths.png)

**圖說 `lec14-kp08-paths`**〔功能：對照〕：兩張圖都從同樣 6 個起點 $(-1.2,-0.4,0.05,0.3,0.9,1.5)$ 出發。左邊的逆向 SDE 每一步都加新的隨機性，同一個起點跑兩次可以走到不同的堆（深紫的 6 條都到右堆，淺紫的有 2 條到左堆）。右邊的 ODE 沒有隨機性，每個起點只有一條平滑的路，而且一維時上下順序永遠不會交叉：起點越低，終點也越低。

**Caption `lec14-kp08-paths`** [Function: comparison]: both charts start from the same 6 points $(-1.2,-0.4,0.05,0.3,0.9,1.5)$. On the left, the reverse SDE adds fresh randomness at every step, so the same start run twice can reach different piles (all 6 dark lines reach the right pile, while 2 of the light ones reach the left). On the right, the ODE has no randomness: each start has a single smooth path, and in one dimension the vertical order never crosses: the lower the start, the lower the end.

![一張直方圖，橫軸 x0 從 -3.5 到 3.5。淡紫色填色是逆向 SDE 走 1000 步的樣本，綠色外框線是機率流 ODE 走 1000 步的樣本，黑色虛線是真正的資料密度：兩座在 -2 與 2、高約 0.66 的山。三者幾乎完全重合（A histogram with x0 from -3.5 to 3.5. The light-purple fill is samples from the reverse SDE in 1000 steps, the green outline samples from the probability-flow ODE in 1000 steps, and the black dashed line the true data density: two peaks at -2 and 2, about 0.66 high. The three almost coincide)](../assets/figures/lec14-kp08-samples.png)

**圖說 `lec14-kp08-samples`**〔功能：對照、精準度〕：從同一批 $20000$ 個起點出發，兩種方法生成的分布都和真資料一致。逆向 SDE：右堆佔 $49.7\%$、平均 $2.006$、標準差 $0.298$；ODE：右堆 $49.6\%$、平均 $1.996$、標準差 $0.299$。路徑完全不同，整體分布卻一樣。ODE 只走 $100$ 步時，右堆標準差仍是 $0.299$。

**Caption `lec14-kp08-samples`** [Function: comparison, precision]: starting from the same $20000$ points, both methods generate distributions that match the real data. Reverse SDE: $49.7\%$ in the right pile, mean $2.006$, standard deviation $0.298$; ODE: $49.6\%$, mean $1.996$, standard deviation $0.299$. The paths are entirely different, yet the overall distributions agree. With the ODE in only $100$ steps, the right pile's standard deviation is still $0.299$.

常見錯誤：以為逆向 SDE 是把前向的每一條路徑倒帶。它只保證每個時間點的「分布」對得上；個別路徑是新走出來的，和 KP01 第3節的結論一樣。

Common mistake: thinking the reverse SDE rewinds each forward path. It only guarantees that the "distribution" at each time matches; individual paths are walked anew, the same conclusion as KP01 Section 3.

### 3. Score matching：分數怎麼學（Score matching: how the score is learned）

分數 $\nabla\log p_t(\mathbf{x})$ 本身算不出來（要知道整個資料分布）。**score matching** 訓練一個網路 $s_\theta(\mathbf{x},t)$ 去估計它。來源的式 (56) 用的是 **denoising score matching**：目標不是算不出來的 $\nabla\log p_t$，而是「已知 $\mathbf{x}_0$ 時」的分數，它就有公式：

The score $\nabla\log p_t(\mathbf{x})$ itself cannot be computed (it needs the whole data distribution). **Score matching** trains a network $s_\theta(\mathbf{x},t)$ to estimate it. The source's equation (56) uses **denoising score matching**: the target is not the uncomputable $\nabla\log p_t$ but the score "with $\mathbf{x}_0$ known", which has a formula:

$$
\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t\mid\mathbf{x}_0)=-\frac{\mathbf{x}_t-\sqrt{\bar\alpha_t}\,\mathbf{x}_0}{1-\bar\alpha_t}=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}} .
$$

這裡的 $\bar\alpha_t$ 指連續版的 $\bar\alpha(t)$。第一個等號是常態 $\mathcal{N}(\sqrt{\bar\alpha_t}\mathbf{x}_0,(1-\bar\alpha_t)I)$ 的對數密度微分；第二個等號用 KP02 的閉式把 $\mathbf{x}_t-\sqrt{\bar\alpha_t}\mathbf{x}_0$ 換成 $\sqrt{1-\bar\alpha_t}\,\epsilon$。損失如下，其中 $\lambda(t)$ 是每個時間的權重、$t$ 在 $[0,1]$ 上均勻抽：

Here $\bar\alpha_t$ means the continuous $\bar\alpha(t)$. The first equality differentiates the log density of the normal $\mathcal{N}(\sqrt{\bar\alpha_t}\mathbf{x}_0,(1-\bar\alpha_t)I)$; the second uses KP02's closed form to replace $\mathbf{x}_t-\sqrt{\bar\alpha_t}\mathbf{x}_0$ by $\sqrt{1-\bar\alpha_t}\,\epsilon$. The loss is as follows, where $\lambda(t)$ is a per-time weight and $t$ is drawn uniformly on $[0,1]$:

$$
L=\mathbb{E}_{t}\Big[\lambda(t)\,\mathbb{E}_{\mathbf{x}_0,\epsilon}\big\Vert s_\theta(\mathbf{x}_t,t)+\tfrac{\epsilon}{\sqrt{1-\bar\alpha_t}}\big\Vert^2\Big].
$$

`[補充]` 兩件事讓它和 DDPM 接上。第一，這個目標裡有 $\mathbf{x}_0$，看起來和真正的分數不一樣；但網路看不到 $\mathbf{x}_0$，平方誤差最小時，網路學到的是目標的條件平均，正好就是真正的分數 $\nabla\log p_t(\mathbf{x}_t)$（Vincent 2011；KP07 第2節已經用數字驗證過這個等式）。第二，若取權重 $\lambda(t)=1-\bar\alpha_t$，並令 $\epsilon_\theta=-\sqrt{1-\bar\alpha_t}\,s_\theta$，損失就變成 $\Vert\epsilon-\epsilon_\theta\Vert^2$，正是 KP05 的 $L_{\text{simple}}$。所以 **DDPM 和 score-based 模型是同一件事的兩種寫法**：猜雜訊就是在估計分數。

`[Added]` Two facts connect it to DDPM. First, this target contains $\mathbf{x}_0$ and looks different from the true score; but the network cannot see $\mathbf{x}_0$, so at the least squared error it learns the conditional mean of the target, which is exactly the true score $\nabla\log p_t(\mathbf{x}_t)$ (Vincent 2011; KP07 Section 2 already checked this identity numerically). Second, with the weight $\lambda(t)=1-\bar\alpha_t$ and $\epsilon_\theta=-\sqrt{1-\bar\alpha_t}\,s_\theta$, the loss becomes $\Vert\epsilon-\epsilon_\theta\Vert^2$, exactly KP05's $L_{\text{simple}}$. So **DDPM and score-based models are two ways of writing the same thing**: guessing the noise is estimating the score.

常見錯誤：以為 score matching 要先知道真正的分數才能訓練。denoising score matching 的目標只用到 $\mathbf{x}_0$ 與加進去的雜訊，兩者訓練時都知道。

Common mistake: thinking score matching needs the true score to train. The target of denoising score matching uses only $\mathbf{x}_0$ and the added noise, both known in training.

### 4. 全章總結：擴散模型在生成模型中的位置（Chapter summary: where diffusion models stand）

這一章的主線是：**固定的前向加雜訊 → 逆向一步一步去雜訊 → 用 ELBO 拆成每一步的 KL → 已知 $\mathbf{x}_0$ 的去噪分布有公式 → 改成猜雜訊 → 簡化損失與兩個演算法 → 改良與加速 → 條件與引導 → 連續時間的 SDE 與分數。** 來源把它和其他生成模型比較如下（本課程第12週介紹 VAE、第13週介紹 GAN；GMMN 在來源第13章，本課程沒有上）：

The thread of this chapter: **fixed forward noising → step-by-step backward denoising → the ELBO split into per-step KLs → a formula for the denoising distribution given $\mathbf{x}_0$ → guess the noise instead → the simplified loss and two algorithms → improvements and speed-ups → conditions and guidance → continuous-time SDEs and the score.** The source compares it with other generative models as follows (this course covered VAEs in Week 12 and GANs in Week 13; GMMN is in the source's Chapter 13 and was not covered in this course):

| 模型 | 怎麼學 | 優點 | 缺點 |
|---|---|---|---|
| VAE | 最大化 ELBO，編碼器與解碼器一起學 | 機率框架清楚、可重建、訓練穩定 | 影像偏模糊 |
| GAN | 生成器與判別器的極小化極大賽局 | 影像銳利、生成快 | 訓練不穩、模式崩塌 |
| GMMN | 用核方法比對生成與真資料的分布 | 不用對抗、理論簡潔 | 高維時不易擴展 |
| 擴散模型 | 最大化 ELBO 的簡化版：猜雜訊的平方誤差 | 訓練穩定、樣本品質高、條件控制靈活 | 生成要走很多步，慢 |

| Model | How it learns | Strengths | Weaknesses |
|---|---|---|---|
| VAE | maximise the ELBO, learning encoder and decoder together | clear probabilistic framework, can reconstruct, stable training | blurry images |
| GAN | a minimax game between generator and discriminator | sharp images, fast generation | unstable training, mode collapse |
| GMMN | match generated and real distributions with kernel methods | no adversary, simple theory | hard to scale to high dimensions |
| Diffusion model | a simplified ELBO: squared error of guessing the noise | stable training, high sample quality, flexible conditioning | generation takes many steps, slow |

擴散模型目前的研究方向，來源整理為三條：生成慢，靠 DDIM、consistency models 與蒸餾加速（KP06）；計算量大，靠潛在擴散在壓縮後的空間裡做（KP07）；條件控制，擴展到文字生成影像、影片與聲音。來源也說，到 2025 年，擴散模型與自迴歸 Transformer（第11週）是生成式 AI 的兩大主流。

The source groups current research on diffusion models into three directions: slow generation, sped up by DDIM, consistency models and distillation (KP06); heavy computation, reduced by latent diffusion in a compressed space (KP07); and conditional control, extended to text-to-image, video and audio. The source also says that, as of 2025, diffusion models and autoregressive Transformers (Week 11) are the two mainstreams of generative AI.

常見錯誤：以為擴散模型和 VAE 的「變分下界」無關。DDPM 的訓練目標就是從 ELBO 推出來的（KP03 到 KP05），只是最後把權重簡化掉了。

Common mistake: thinking diffusion models have nothing to do with the VAE's "variational lower bound". DDPM's training objective is derived from the ELBO (KP03 to KP05); only at the end are the weights simplified away.

### 本單元重點整理（KP08 summary）

**一句話重點**：步數無限多時，DDPM 變成前向 SDE $d\mathbf{x}=-\frac12\beta\mathbf{x}\,dt+\sqrt\beta\,d\mathbf{w}$；生成就是解只差一項分數的逆向 SDE（或不加隨機性的機率流 ODE）；分數用 denoising score matching 學，而它和 DDPM 猜雜訊的損失是同一件事。

**One-sentence summary**: with infinitely many steps, DDPM becomes the forward SDE $d\mathbf{x}=-\frac12\beta\mathbf{x}\,dt+\sqrt\beta\,d\mathbf{w}$; generation solves the reverse SDE, which differs only by a score term (or the randomness-free probability-flow ODE); the score is learned by denoising score matching, which is the same thing as DDPM's noise-guessing loss.

**白話說明**：把一千級的樓梯磨成一道斜坡，就是 SDE。下坡時需要一個指南針告訴你「哪邊是資料」，那就是分數；網路猜的雜訊，指的正是指南針的反方向。走下坡可以一路晃著走（SDE），也可以穩穩地沿一條固定的路走（ODE），最後到的人群分布都一樣。

**Plain explanation**: grind a thousand-step staircase into a smooth slope, and you have an SDE. Going down you need a compass saying "the data are this way"; that is the score, and the noise the network guesses points exactly opposite the compass. You can stagger down (SDE) or walk steadily along a fixed path (ODE), and the crowd ends up distributed the same way either way.

**正式定義**：前向 SDE $d\mathbf{x}=f\,dt+g\,d\mathbf{w}$，DDPM 對應 $f=-\frac12\beta(t)\mathbf{x}$、$g=\sqrt{\beta(t)}$；逆向 SDE $d\mathbf{x}=[f-g^2\nabla\log p_t]\,dt+g\,d\bar{\mathbf{w}}$；機率流 ODE $d\mathbf{x}=[f-\frac12g^2\nabla\log p_t]\,dt$；denoising score matching $\mathbb{E}\Vert s_\theta(\mathbf{x}_t,t)+\epsilon/\sqrt{1-\bar\alpha_t}\Vert^2$，與 $\epsilon_\theta=-\sqrt{1-\bar\alpha_t}\,s_\theta$ 的 $L_{\text{simple}}$ 等價（權重 $1-\bar\alpha_t$）。

**Formal definition**: forward SDE $d\mathbf{x}=f\,dt+g\,d\mathbf{w}$, with DDPM corresponding to $f=-\frac12\beta(t)\mathbf{x}$ and $g=\sqrt{\beta(t)}$; reverse SDE $d\mathbf{x}=[f-g^2\nabla\log p_t]\,dt+g\,d\bar{\mathbf{w}}$; probability-flow ODE $d\mathbf{x}=[f-\frac12g^2\nabla\log p_t]\,dt$; denoising score matching $\mathbb{E}\Vert s_\theta(\mathbf{x}_t,t)+\epsilon/\sqrt{1-\bar\alpha_t}\Vert^2$, equivalent to $L_{\text{simple}}$ with $\epsilon_\theta=-\sqrt{1-\bar\alpha_t}\,s_\theta$ (weight $1-\bar\alpha_t$).

**例子**：$\beta(t)=0.1+19.9t$ 的連續版在 $t=0.3$ 給 $\bar\alpha=0.3963$，離散 DDPM 第 $300$ 步是 $0.3964$。兩堆資料用精確分數：逆向 SDE 與 ODE 各走 $1000$ 步，右堆標準差 $0.298$、$0.299$（真資料 $0.3$）。

**Example**: the continuous version with $\beta(t)=0.1+19.9t$ gives $\bar\alpha=0.3963$ at $t=0.3$, versus $0.3964$ at discrete DDPM step $300$. On the two piles with the exact score, the reverse SDE and the ODE in $1000$ steps each give right-pile standard deviations of $0.298$ and $0.299$ (real data: $0.3$).

**常見迷思**：

- 「$d\mathbf{w}$ 的大小和 $dt$ 成正比。」錯，標準差是 $\sqrt{dt}$。
- 「逆向 SDE 是把前向路徑倒帶。」錯，只有每個時間點的分布對得上。
- 「score matching 要先知道真正的分數。」錯，denoising score matching 只用 $\mathbf{x}_0$ 與雜訊。
- 「擴散模型和變分下界無關。」錯，DDPM 的損失就是從 ELBO 簡化來的。

**Common misconceptions**:

- "The size of $d\mathbf{w}$ is proportional to $dt$." Wrong: its standard deviation is $\sqrt{dt}$.
- "The reverse SDE rewinds forward paths." Wrong: only the distribution at each time matches.
- "Score matching needs the true score first." Wrong: denoising score matching uses only $\mathbf{x}_0$ and the noise.
- "Diffusion models have nothing to do with the variational lower bound." Wrong: DDPM's loss is simplified from the ELBO.

**圖**：本單元有兩張圖。

- `lec14-kp08-paths`：逆向 SDE 與機率流 ODE 從同樣起點出發的路徑。
- `lec14-kp08-samples`：兩種方法生成的樣本與真資料密度。

**Figures** — this unit has two figures.

- `lec14-kp08-paths`: paths of the reverse SDE and the probability-flow ODE from the same starts.
- `lec14-kp08-samples`: samples generated by both methods against the true data density.

### 小練習（Practice）

1. 用 Euler–Maruyama 法手算前向 SDE 的一步：$x=1$、$\beta=10$、$h=0.01$、抽到 $z=0.5$。
2. （來源習題）一維時，$q(x_t\mid x_0)$ 的變異數是多少？它的熵 $\tfrac12\log(2\pi e\,v)$ 在 $\bar\alpha_t=0.5$ 時是多少？
3. （來源習題改寫）若資料本身是 $\mathcal{N}(0,1)$，已知 $x_t$ 時 $x_0$ 的後驗平均與變異數是多少？代 $\bar\alpha_t=0.5$、$x_t=1$。
4. 用 denoising score matching 的目標公式，算 $x_0=2$、$x_t=1$、$\bar\alpha_t=0.5$ 時的目標分數，並用對應的 $\epsilon$ 驗算。

1. Compute one Euler–Maruyama step of the forward SDE by hand: $x=1$, $\beta=10$, $h=0.01$, with draw $z=0.5$.
2. (Source exercise) In one dimension, what is the variance of $q(x_t\mid x_0)$? What is its entropy $\tfrac12\log(2\pi e\,v)$ at $\bar\alpha_t=0.5$?
3. (Adapted from a source exercise) If the data themselves are $\mathcal{N}(0,1)$, what are the posterior mean and variance of $x_0$ given $x_t$? Use $\bar\alpha_t=0.5$ and $x_t=1$.
4. Using the denoising score matching target formula, compute the target score for $x_0=2$, $x_t=1$ and $\bar\alpha_t=0.5$, and check it with the corresponding $\epsilon$.

**第 1 題解答** `[補充]`：漂移 $f=-\frac12\times10\times1=-5$，$fh=-0.05$；抖動 $\sqrt{10}\sqrt{0.01}\times0.5\approx0.1581$。新的 $x=1-0.05+0.1581\approx1.1081$。

**Solution to question 1** `[Added]`: the drift is $f=-\frac12\times10\times1=-5$, so $fh=-0.05$; the jitter is $\sqrt{10}\sqrt{0.01}\times0.5\approx0.1581$. The new $x=1-0.05+0.1581\approx1.1081$.

**第 2 題解答** `[補充]`：變異數是 $1-\bar\alpha_t$（KP02）。$\bar\alpha_t=0.5$ 時 $v=0.5$，熵 $=\tfrac12\log(2\pi e\times0.5)\approx1.0724$（自然對數）。$d$ 維時是 $\tfrac d2\log(2\pi e(1-\bar\alpha_t))$。

**Solution to question 2** `[Added]`: the variance is $1-\bar\alpha_t$ (KP02). At $\bar\alpha_t=0.5$, $v=0.5$, and the entropy is $\tfrac12\log(2\pi e\times0.5)\approx1.0724$ (natural log). In $d$ dimensions it is $\tfrac d2\log(2\pi e(1-\bar\alpha_t))$.

**第 3 題解答** `[補充]`：資料 $\mathcal{N}(0,1)$ 時，$x_0$ 與 $x_t$ 都是標準常態、相關係數 $\sqrt{\bar\alpha_t}$；套用 KP03 第5節的規則（兩個相關係數 $\rho$ 的標準常態，已知一個時另一個是 $\mathcal{N}(\rho x,1-\rho^2)$），取 $\rho=\sqrt{\bar\alpha_t}$，所以 $x_0\mid x_t\sim\mathcal{N}(\sqrt{\bar\alpha_t}\,x_t,\ 1-\bar\alpha_t)$。代入得平均 $\sqrt{0.5}\approx0.7071$、變異數 $0.5$。注意：資料不是常態時，這個後驗要看資料分布，沒有一般公式。

**Solution to question 3** `[Added]`: with data $\mathcal{N}(0,1)$, $x_0$ and $x_t$ are both standard normal with correlation $\sqrt{\bar\alpha_t}$; applying KP03 Section 5's rule (for two standard normals with correlation $\rho$, the conditional of one given the other is $\mathcal{N}(\rho x,1-\rho^2)$) with $\rho=\sqrt{\bar\alpha_t}$ gives $x_0\mid x_t\sim\mathcal{N}(\sqrt{\bar\alpha_t}\,x_t,\ 1-\bar\alpha_t)$. Substituting gives mean $\sqrt{0.5}\approx0.7071$ and variance $0.5$. Note: when the data are not normal, this posterior depends on the data distribution and has no general formula.

**第 4 題解答** `[補充]`：目標 $=-(1-\sqrt{0.5}\times2)/(1-0.5)=-(1-1.4142)/0.5\approx0.8284$。對應的 $\epsilon=(1-1.4142)/\sqrt{0.5}\approx-0.5858$，$-\epsilon/\sqrt{0.5}\approx0.8284$，一致。

**Solution to question 4** `[Added]`: the target is $-(1-\sqrt{0.5}\times2)/(1-0.5)=-(1-1.4142)/0.5\approx0.8284$. The corresponding $\epsilon=(1-1.4142)/\sqrt{0.5}\approx-0.5858$, and $-\epsilon/\sqrt{0.5}\approx0.8284$, which agrees.

### 小測驗（Quiz，來源原題）

來源 KP-15 與總結一節各有四題自我檢測，保留原題與全部選項；來源頁面附有答案與簡短解析，下面的解析由本講義依各節內容補寫。

The source's KP-15 and summary section each have four self-check questions, kept here with all their options; the source page gives the answers with short explanations, and the explanations below are written by this note from its own sections.

1. 連續化 diffusion 中，$\nabla_x\log p_t(x)$ 被稱為什麼？
   - (A) Drift
   - (B) Score
   - (C) Diffusion coefficient
   - (D) Prior
2. Reverse-time SDE 的 drift 項包含什麼？
   - (A) $f(x,t)+g(t)^2\nabla_x\log p_t(x)$
   - (B) $f(x,t)-g(t)^2\nabla_x\log p_t(x)$
   - (C) 只有 $g(t)$
   - (D) $-\frac12\beta(t)x$
3. 式 (56) 的訓練目標是什麼？
   - (A) 預測雜訊 $\epsilon$
   - (B) 預測 score function
   - (C) 預測 $x_0$
   - (D) 預測 $\beta_t$
4. 逆向 SDE 常用什麼數值方法求解？
   - (A) Gradient descent
   - (B) Euler–Maruyama
   - (C) Newton method
   - (D) Backpropagation
5. Diffusion model 相較於 GAN 的主要優勢是什麼？
   - (A) 推論速度更快
   - (B) 訓練更穩定、樣本品質高
   - (C) 參數更少
   - (D) 不需要 latent variable
6. Diffusion model 的主要限制是什麼？
   - (A) 無法做條件生成
   - (B) 推論需要多步迭代，速度較慢
   - (C) 訓練資料需求極小
   - (D) 只能生成小圖
7. Stable Diffusion 屬於哪一類改進？
   - (A) 學習 covariance
   - (B) Latent diffusion
   - (C) PriorGrad
   - (D) D3PM
8. 本章習題 4 主要要求推導什麼？
   - (A) SDE 的連續極限
   - (B) DDPM 的 ELBO 並簡化成 noise-prediction loss
   - (C) GAN 的對抗 loss
   - (D) VAE 的 reparameterization trick

1. In continuous-time diffusion, what is $\nabla_x\log p_t(x)$ called?
   - (A) Drift
   - (B) Score
   - (C) Diffusion coefficient
   - (D) Prior
2. What does the drift of the reverse-time SDE contain?
   - (A) $f(x,t)+g(t)^2\nabla_x\log p_t(x)$
   - (B) $f(x,t)-g(t)^2\nabla_x\log p_t(x)$
   - (C) Only $g(t)$
   - (D) $-\frac12\beta(t)x$
3. What is the training objective of equation (56)?
   - (A) Predict the noise $\epsilon$
   - (B) Predict the score function
   - (C) Predict $x_0$
   - (D) Predict $\beta_t$
4. What numerical method is commonly used to solve the reverse SDE?
   - (A) Gradient descent
   - (B) Euler–Maruyama
   - (C) Newton's method
   - (D) Backpropagation
5. What is the main advantage of diffusion models over GANs?
   - (A) Faster inference
   - (B) More stable training and high sample quality
   - (C) Fewer parameters
   - (D) No latent variable needed
6. What is the main limitation of diffusion models?
   - (A) They cannot do conditional generation
   - (B) Inference needs many iterative steps and is slower
   - (C) They need extremely little training data
   - (D) They can only generate small images
7. Which kind of improvement is Stable Diffusion?
   - (A) Learned covariance
   - (B) Latent diffusion
   - (C) PriorGrad
   - (D) D3PM
8. What does the chapter's exercise 4 mainly ask you to derive?
   - (A) The continuous limit of the SDE
   - (B) DDPM's ELBO, simplified into the noise-prediction loss
   - (C) The GAN's adversarial loss
   - (D) The VAE's reparameterization trick

**解析** `[補充]`：

- 第 1 題選 (B)：第2節與 KP07 第2節。
- 第 2 題選 (B)：第2節的逆向 SDE；減號讓倒著走時往機率高的地方去。
- 第 3 題選 (B)：第3節。不過第3節也說明，換個寫法它就是 (A)：猜雜訊和估計分數是同一件事。
- 第 4 題選 (B)：第1、2節；(A)、(D) 是訓練網路用的方法，不是解方程的方法。
- 第 5 題選 (B)：第4節的表；(A) 剛好是擴散模型的弱點。
- 第 6 題選 (B)：第4節；KP07 說明條件生成正是它的強項，所以 (A) 錯。
- 第 7 題選 (B)：KP07 第5節。
- 第 8 題選 (B)：這正是本週 KP03 到 KP05 的主線。

**Explanations** `[Added]`:

- Question 1: B: Section 2 and KP07 Section 2.
- Question 2: B: the reverse SDE of Section 2; the minus sign makes the backward run head towards high probability.
- Question 3: B: Section 3. But Section 3 also shows that, written differently, it is A: guessing the noise and estimating the score are the same thing.
- Question 4: B: Sections 1 and 2; A and D are for training networks, not for solving equations.
- Question 5: B: the table in Section 4; A is exactly the weakness of diffusion models.
- Question 6: B: Section 4; KP07 shows conditional generation is one of their strengths, so A is wrong.
- Question 7: B: KP07 Section 5.
- Question 8: B: this is exactly the thread of this week's KP03 to KP05.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第15章「Diffusion Models」的 KP-15「SDE 連續化：Score Matching 與 Euler–Maruyama」與第 6–7 節「總結」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch15-diffusion-models/index.html ，讀取日期 2026-11-15）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from KP-15, "Continuous-time SDEs: score matching and Euler–Maruyama", and the "Summary" of Sections 6–7 of Chapter 15, "Diffusion Models", of CCC-LNS IDL_115 (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch15-diffusion-models/index.html , accessed 2026-11-15). Content not in the source is marked `[Added]` where it appears.

- 來源寫「當 $T\to\infty$、每步變化極小時，可以收斂到 SDE」，沒有說明 $\beta_t$ 要怎麼跟著 $T$ 縮小。要得到式 (52)，必須令 $\beta_t=\beta(t)\,dt$，並用到 $\sqrt{1-\beta_t}\approx1-\beta_t/2$；本講義第1節補上，並給出 DDPM 排程對應的 $\beta(t)=0.1+19.9t$。
- 來源總結一節的示意圖提示說擴散模型「不是用對抗或變分下界，而是用學習去噪」達成效果。DDPM 的損失正是從變分下界（ELBO）推導再簡化而來（KP03 到 KP05）；本講義第4節的表與常見迷思照此寫。
- 來源習題 1 要求「推導 $q(x_0\mid x_t)$ 的後驗均值與分布」。這個後驗取決於資料分布，沒有一般公式；本講義小練習第 3 題改成資料為 $\mathcal{N}(0,1)$ 的情形。
- 來源把 GMMN 列入比較；本課程第13週沒有介紹 GMMN，本講義在表中保留並註明。
- 本講義補上：一步近似的數值、維納增量為何是 $\sqrt{dt}$、Euler–Maruyama 的寫法與前向 SDE 的模擬核對、機率流 ODE 與 DDIM 的關係、逆向 SDE 與 ODE 的路徑與樣本比較圖、denoising score matching 目標的推導、它的最佳解是真正分數、它與 $L_{\text{simple}}$ 的等價、全章主線、四題小練習與解答、小測驗的解析。
- 取捨：來源的互動展示不重做；逆向 SDE 的證明（Anderson 1982）不在大學生程度範圍內，只引用結果；符號加粗與否的取捨同 KP02。

- The source writes "as $T\to\infty$ with tiny changes per step, it converges to an SDE" without saying how $\beta_t$ must shrink with $T$. Equation (52) requires $\beta_t=\beta(t)\,dt$ and uses $\sqrt{1-\beta_t}\approx1-\beta_t/2$; Section 1 adds this and gives $\beta(t)=0.1+19.9t$ for the DDPM schedule.
- The figure prompt in the source's summary says diffusion models succeed "not with adversaries or variational lower bounds but by learning to denoise". DDPM's loss is derived from the variational lower bound (ELBO) and then simplified (KP03 to KP05); the table and misconceptions in Section 4 follow this.
- The source's exercise 1 asks to "derive the posterior mean and distribution of $q(x_0\mid x_t)$". This posterior depends on the data distribution and has no general formula; practice question 3 uses the case of data $\mathcal{N}(0,1)$ instead.
- The source includes GMMN in the comparison; Week 13 of this course did not cover GMMN, and this note keeps it in the table with a note.
- This note adds: the numbers for the one-step approximation; why a Wiener increment is $\sqrt{dt}$; the Euler–Maruyama form and a simulation check of the forward SDE; the probability-flow ODE and its link to DDIM; the path and sample comparison figures for the reverse SDE and the ODE; the derivation of the denoising score matching target; the fact that its optimum is the true score; its equivalence to $L_{\text{simple}}$; the chapter thread; four practice questions with solutions; and the explanations of the quiz.
- Choices: the source's interactive demo is not rebuilt; the proof of the reverse SDE (Anderson 1982) is beyond the undergraduate level and only the result is cited. Bold versus plain symbols follow the choice made in KP02.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 隨機微分方程 | stochastic differential equation (SDE) |
| 常微分方程 | ordinary differential equation (ODE) |
| 機率流常微分方程 | probability-flow ODE |
| 維納過程（布朗運動） | Wiener process (Brownian motion) |
| 漂移係數 | drift coefficient |
| 擴散係數 | diffusion coefficient |
| 逆向隨機微分方程 | reverse-time SDE |
| Euler–Maruyama 法 | Euler–Maruyama method |
| 分數 | score |
| 分數匹配 | score matching |
| 去雜訊分數匹配 | denoising score matching |
| 熵 | entropy |
| 後驗平均 | posterior mean |
| 生成矩匹配網路 | generative moment matching network (GMMN) |
| 核方法 | kernel method |
| 潛在擴散模型 | latent diffusion model |
| 自迴歸 Transformer（第11週） | autoregressive Transformer |
| 變分自編碼器（第12週） | variational autoencoder (VAE) |
| 生成對抗網路（第13週） | generative adversarial network (GAN) |
| 證據下界 | evidence lower bound (ELBO) |
| 去雜訊擴散機率模型 | denoising diffusion probabilistic model (DDPM) |
| 去雜訊擴散隱式模型 | denoising diffusion implicit model (DDIM) |
| KL 散度 | Kullback–Leibler (KL) divergence |
| 人工智慧 | artificial intelligence (AI) |
