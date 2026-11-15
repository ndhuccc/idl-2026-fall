# 第14週 KP04：真正的去噪分布與「改成預測雜訊」（The true denoising distribution, and predicting the noise instead）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第14週「深度生成模型（三）：擴散模型」的第四個知識點（KP04）。KP03 把 ELBO 拆成每一步的 KL：學到的逆向一步 $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$，要接近「已知原資料」時真正的去噪一步 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$。這一節把後者算出來：它是一個常態分布，平均和變異數都有公式。接著把公式裡的 $\mathbf{x}_0$ 換成雜訊，得到 DDPM 最重要的設計：讓網路去猜加進去的雜訊。

This is the fourth knowledge point (KP04) of Week 14, "Deep generative models (3): diffusion models". KP03 split the ELBO into one KL per step: the learned backward step $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ should approach the true denoising step "with the original data known", $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$. This note works the latter out: it is a normal distribution whose mean and variance have formulas. Then the $\mathbf{x}_0$ in the formula is replaced by the noise, which leads to DDPM's most important design: have the network guess the noise that was added.

先備知識是本週 KP02 的閉式 $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\epsilon$、KP03 的 ELBO 拆解與貝氏定理，以及「把兩個常態的密度相乘」這個小技巧（第2節會說明）。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第15章 KP-6（2.4.2 節）與 KP-7（2.4.3 節）改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec14-kp04-figures.py` 算出，可以自己重算。

The prerequisites are the closed form $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ of this week's KP02, the ELBO split and Bayes' rule of KP03, and the small trick of "multiplying two normal densities" (explained in Section 2). Every term and symbol is defined when it first appears. This note is rewritten from KP-6 (Section 2.4.2) and KP-7 (Section 2.4.3) of Chapter 15 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec14-kp04-figures.py`, so you can recompute it yourself.

## KP04 真正的去噪分布與「改成預測雜訊」（The true denoising distribution, and predicting the noise instead）

### 學習目標（Learning objectives）

讀完之後，你能說明為什麼 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ 算不出來、多加一個條件 $\mathbf{x}_0$ 之後卻算得出來。你能用貝氏定理寫出 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$，並用「兩個常態相乘：精確度相加、平均依精確度加權」算出它的平均 $\tilde\mu$ 與變異數 $\tilde\beta_t$。你能用一組小數字手算 $\tilde\mu$ 與 $\tilde\beta_t$。你能把 $\tilde\mu$ 裡的 $\mathbf{x}_0$ 換成雜訊 $\epsilon$，推出 $\tilde\mu=\frac{1}{\sqrt{\alpha_t}}\big(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\big)$，並說出網路改成預測雜訊 $\epsilon_\theta$ 的寫法與理由。

After reading, you can explain why $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ cannot be computed, while adding the condition $\mathbf{x}_0$ makes it computable. You can write $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$ with Bayes' rule and compute its mean $\tilde\mu$ and variance $\tilde\beta_t$ with "multiplying two normals: precisions add, and the mean is precision-weighted". You can compute $\tilde\mu$ and $\tilde\beta_t$ by hand for a small set of numbers. You can replace the $\mathbf{x}_0$ in $\tilde\mu$ with the noise $\epsilon$ to derive $\tilde\mu=\frac{1}{\sqrt{\alpha_t}}\big(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\big)$, and state how and why the network instead predicts the noise $\epsilon_\theta$.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\beta_t,\ \mathcal{N},\ I$ | 第 $t$ 步的雜訊變異數、常態分布、單位矩陣（KP02） | noise variance of step $t$, normal distribution, identity matrix (KP02) |
| $\alpha_t,\ \bar\alpha_t$ | $\alpha_t=1-\beta_t$，$\bar\alpha_t=\alpha_1\cdots\alpha_t$（KP02）；約定 $\bar\alpha_0=1$ | $\alpha_t=1-\beta_t$, $\bar\alpha_t=\alpha_1\cdots\alpha_t$ (KP02); by convention $\bar\alpha_0=1$ |
| $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$ | 真正的去噪分布：已知 $\mathbf{x}_t$ 與原資料 $\mathbf{x}_0$ 時，上一步的分布 | the true denoising distribution: the previous step's distribution given $\mathbf{x}_t$ and the original $\mathbf{x}_0$ |
| $\tilde\mu(\mathbf{x}_t,\mathbf{x}_0)$ | 真正去噪分布的平均 | the mean of the true denoising distribution |
| $\tilde\beta_t$ | 真正去噪分布的變異數（每個維度相同） | the variance of the true denoising distribution (the same in every dimension) |
| $\epsilon$ | 從 $\mathbf{x}_0$ 一步跳到 $\mathbf{x}_t$ 時用的標準常態雜訊（KP02 閉式） | the standard-normal noise used to jump from $\mathbf{x}_0$ to $\mathbf{x}_t$ (KP02's closed form) |
| $\epsilon_\theta(\mathbf{x}_t,t)$ | 網路對 $\epsilon$ 的猜測 | the network's guess of $\epsilon$ |
| $\mu_\theta(\mathbf{x}_t,t)$ | 網路給的逆向一步平均（KP03） | the backward-step mean given by the network (KP03) |
| 精確度 | 變異數的倒數 $1/v$；越大表示越確定 | precision: the reciprocal $1/v$ of the variance; larger means more certain |
| $\propto$ | 「成正比」：只差一個不含變數的常數倍 | "proportional to": equal up to a constant factor not containing the variable |

### 1. 已知原資料，去噪就變簡單（With the original known, denoising becomes easy）

KP03 第2節看過：真正的逆向一步 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ 要知道整個資料分布才算得出來，而且可能有好幾座山。但在 ELBO 裡，要比對的是多一個條件的 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$，來源稱它為**真正的去噪分布**（ground-truth denoising distribution），DDPM 論文稱它為前向過程的**後驗分布**（posterior）：看到 $\mathbf{x}_t$ 之後，對上一步的推論。多知道 $\mathbf{x}_0$，事情就簡單了：用貝氏定理（來源的式 (33)），

KP03 Section 2 showed that the true backward step $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ needs the whole data distribution to compute and may have several peaks. But the ELBO compares against $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$, which has one more condition; the source calls it the **ground-truth denoising distribution**, and the DDPM paper calls it the forward process's **posterior**: the inference about the previous step after seeing $\mathbf{x}_t$. Knowing $\mathbf{x}_0$ as well makes things simple: by Bayes' rule (the source's equation (33)),

$$
q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)=\frac{q(\mathbf{x}_t\mid\mathbf{x}_{t-1},\mathbf{x}_0)\;q(\mathbf{x}_{t-1}\mid\mathbf{x}_0)}{q(\mathbf{x}_t\mid\mathbf{x}_0)} .
$$

右邊三項全是 KP02 寫好的常態分布：分子第一項是前向一步（馬可夫性質讓它等於 $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$），分子第二項和分母是一步跳的閉式。分母不含 $\mathbf{x}_{t-1}$，只是讓總機率等於 $1$ 的常數。所以，**作為 $\mathbf{x}_{t-1}$ 的函數**，這個分布正比於兩個常態密度的乘積。

All three terms on the right are normal distributions written out in KP02: the first factor on top is the forward step (the Markov property makes it equal $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$), and the second factor on top and the denominator are one-jump closed forms. The denominator does not contain $\mathbf{x}_{t-1}$; it is just the constant that makes the total probability $1$. So, **as a function of $\mathbf{x}_{t-1}$**, this distribution is proportional to the product of two normal densities.

常見錯誤：以為 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$ 是「假設」成常態的。來源寫「假設它也是高斯」，但其實不用假設：兩個常態密度相乘一定還是常態的形狀，下一節直接算出來。

Common mistake: thinking $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$ is "assumed" to be normal. The source writes "assume it is also Gaussian", but no assumption is needed: the product of two normal densities always has a normal shape, and the next section computes it directly.

### 2. 配平方：兩個常態相乘（Completing the square: multiplying two normals）

先看一維。分子的兩項，各自看成 $x_{t-1}$ 的函數：

Start in one dimension. Look at each of the two factors on top as a function of $x_{t-1}$:

- 前向一步 $q(x_t\mid x_{t-1})\propto\exp\!\big(-\frac{(x_t-\sqrt{\alpha_t}\,x_{t-1})^2}{2\beta_t}\big)$。把括號裡提出 $\sqrt{\alpha_t}$，它等於 $\exp\!\big(-\frac{(x_{t-1}-x_t/\sqrt{\alpha_t})^2}{2\beta_t/\alpha_t}\big)$，是一個平均 $x_t/\sqrt{\alpha_t}$、變異數 $\beta_t/\alpha_t$ 的常態形狀：「從 $x_t$ 往回推」的建議。
- 一步跳 $q(x_{t-1}\mid x_0)\propto\exp\!\big(-\frac{(x_{t-1}-\sqrt{\bar\alpha_{t-1}}\,x_0)^2}{2(1-\bar\alpha_{t-1})}\big)$，平均 $\sqrt{\bar\alpha_{t-1}}\,x_0$、變異數 $1-\bar\alpha_{t-1}$：「從 $x_0$ 往前推」的建議。

- The forward step $q(x_t\mid x_{t-1})\propto\exp\!\big(-\frac{(x_t-\sqrt{\alpha_t}\,x_{t-1})^2}{2\beta_t}\big)$. Pulling $\sqrt{\alpha_t}$ out of the bracket, it equals $\exp\!\big(-\frac{(x_{t-1}-x_t/\sqrt{\alpha_t})^2}{2\beta_t/\alpha_t}\big)$, a normal shape with mean $x_t/\sqrt{\alpha_t}$ and variance $\beta_t/\alpha_t$: the suggestion "working back from $x_t$".
- The jump $q(x_{t-1}\mid x_0)\propto\exp\!\big(-\frac{(x_{t-1}-\sqrt{\bar\alpha_{t-1}}\,x_0)^2}{2(1-\bar\alpha_{t-1})}\big)$, with mean $\sqrt{\bar\alpha_{t-1}}\,x_0$ and variance $1-\bar\alpha_{t-1}$: the suggestion "working forward from $x_0$".

兩個常態形狀相乘，指數相加，展開後是 $x_{t-1}$ 的二次式；把它整理成 $(x_{t-1}-\text{平均})^2/\text{變異數}$ 的樣子，就叫**配平方**（completing the square，來源的式 (34)）。整理的結果有一個好記的規則 `[補充]`：**精確度（變異數的倒數）相加，平均是兩個平均依精確度加權。**

Multiplying two normal shapes adds their exponents, giving a quadratic in $x_{t-1}$; rewriting it as $(x_{t-1}-\text{mean})^2/\text{variance}$ is called **completing the square** (the source's equation (34)). The result follows an easy rule `[Added]`: **the precisions (reciprocals of the variances) add, and the mean is the precision-weighted average of the two means.**

$$
\frac{1}{\tilde\beta_t}=\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}},\qquad \tilde\mu=\tilde\beta_t\left(\frac{\alpha_t}{\beta_t}\cdot\frac{x_t}{\sqrt{\alpha_t}}+\frac{1}{1-\bar\alpha_{t-1}}\cdot\sqrt{\bar\alpha_{t-1}}\,x_0\right).
$$

第一式通分：$\frac{\alpha_t(1-\bar\alpha_{t-1})+\beta_t}{\beta_t(1-\bar\alpha_{t-1})}$，分子 $\alpha_t-\alpha_t\bar\alpha_{t-1}+1-\alpha_t=1-\bar\alpha_t$。代回去，就得到來源的式 (35) 與變異數：

Put the first formula over a common denominator: $\frac{\alpha_t(1-\bar\alpha_{t-1})+\beta_t}{\beta_t(1-\bar\alpha_{t-1})}$, whose numerator is $\alpha_t-\alpha_t\bar\alpha_{t-1}+1-\alpha_t=1-\bar\alpha_t$. Substituting back gives the source's equation (35) and the variance:

$$
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t,\qquad \tilde\mu(\mathbf{x}_t,\mathbf{x}_0)=\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,\mathbf{x}_t+\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,\mathbf{x}_0 .
$$

向量的情形每個維度各自獨立，公式一樣，共變異數是 $\tilde\beta_t I$。注意變異數 $\tilde\beta_t$ 只跟排程有關，和 $\mathbf{x}_t$、$\mathbf{x}_0$ 都無關。

In the vector case every dimension is independent and the formulas are the same, with covariance $\tilde\beta_t I$. Note that the variance $\tilde\beta_t$ depends only on the schedule, not on $\mathbf{x}_t$ or $\mathbf{x}_0$.

**手算一次**（數字取自來源的互動展示）`[補充]`：$x_0=1$、$x_t=0.5$、$\beta_t=0.2$、$\bar\alpha_{t-1}=0.6$，所以 $\alpha_t=0.8$、$\bar\alpha_t=0.48$。

**Compute once by hand** (numbers taken from the source's interactive demo) `[Added]`: $x_0=1$, $x_t=0.5$, $\beta_t=0.2$ and $\bar\alpha_{t-1}=0.6$, so $\alpha_t=0.8$ and $\bar\alpha_t=0.48$.

- 從 $x_t$ 往回推：平均 $0.5/\sqrt{0.8}\approx0.559$，變異數 $0.2/0.8=0.25$，精確度 $4$。
- 從 $x_0$ 往前推：平均 $\sqrt{0.6}\approx0.775$，變異數 $0.4$，精確度 $2.5$。
- 精確度相加 $4+2.5=6.5$，所以 $\tilde\beta_t=1/6.5\approx0.1538$（公式：$0.4\times0.2/0.52\approx0.1538$）。
- 平均 $\tilde\mu=(4\times0.559+2.5\times0.775)/6.5\approx0.642$（公式：$0.688\times0.5+0.298\times1\approx0.642$）。

- Working back from $x_t$: mean $0.5/\sqrt{0.8}\approx0.559$, variance $0.2/0.8=0.25$, precision $4$.
- Working forward from $x_0$: mean $\sqrt{0.6}\approx0.775$, variance $0.4$, precision $2.5$.
- The precisions add to $4+2.5=6.5$, so $\tilde\beta_t=1/6.5\approx0.1538$ (formula: $0.4\times0.2/0.52\approx0.1538$).
- The mean is $\tilde\mu=(4\times0.559+2.5\times0.775)/6.5\approx0.642$ (formula: $0.688\times0.5+0.298\times1\approx0.642$).

![一張曲線圖，橫軸 x_{t-1} 從 -1.2 到 2.6。藍色虛線是「從 x_t 往回推」的常態，平均 0.559；綠色點虛線是「從 x0 往前推」的常態，平均 0.775，比藍線更矮更寬；紫色粗實線是兩者相乘的結果 q(x_{t-1} | x_t, x0)，平均 0.642，夾在兩者之間，而且比兩條都更高更窄。淡紫色直方圖是抽樣結果：從 x0=1 走到 x_{t-1} 再走一步，只留下 x_t 落在 0.5 附近的樣本，直方圖和紫線幾乎重合。三條細點線標出三個平均。標題寫 x0=1、x_t=0.5、beta_t=0.2、alpha bar_{t-1}=0.6（A curve chart with x_{t-1} from -1.2 to 2.6. The dashed blue curve is the normal "working back from x_t", mean 0.559; the dash-dot green curve is the normal "working forward from x0", mean 0.775, lower and wider than the blue; the thick solid purple curve is their product q(x_{t-1} | x_t, x0), mean 0.642, lying between the two and taller and narrower than both. The light-purple histogram comes from sampling: go from x0 = 1 to x_{t-1}, take one more step, and keep only samples whose x_t lands near 0.5; it almost coincides with the purple curve. Three thin dotted lines mark the three means. The title reads x0 = 1, x_t = 0.5, beta_t = 0.2, alpha bar_{t-1} = 0.6)](../assets/figures/lec14-kp04-posterior.png)

**圖說 `lec14-kp04-posterior`**〔功能：精準度、一目了然〕：藍色與綠色是兩個「建議」，紫色是把兩者合起來的答案。先看位置：紫色平均 $0.642$ 夾在 $0.559$ 與 $0.775$ 之間，比較靠近藍色，因為藍色的精確度較高（$4$ 比 $2.5$）。再看寬度：紫色比兩條都窄，兩個獨立的線索合起來，比只用一個更確定。淡紫色直方圖是真的抽樣結果（$400$ 萬次抽樣中留下 $42520$ 個 $x_t$ 落在 $0.49$ 到 $0.51$ 的樣本），平均 $0.642$、變異數 $0.153$，和公式一致。

**Caption `lec14-kp04-posterior`** [Function: precision, seeing it at a glance]: blue and green are two "suggestions", and purple is the answer that combines them. First the location: the purple mean $0.642$ lies between $0.559$ and $0.775$, closer to blue, because blue has the higher precision ($4$ versus $2.5$). Then the width: purple is narrower than both; two independent clues together are more certain than either alone. The light-purple histogram is a real sampling result (of $4$ million draws, the $42520$ whose $x_t$ fell between $0.49$ and $0.51$), with mean $0.642$ and variance $0.153$, matching the formulas.

常見錯誤：以為合起來的分布會比兩個建議都寬，因為「兩個不確定加在一起更不確定」。相乘是把兩個獨立線索**同時**成立的情況挑出來，精確度相加，所以變得更窄。

Common mistake: thinking the combined distribution is wider than both suggestions, because "two uncertainties together are more uncertain". Multiplying picks out the cases where both independent clues hold **at once**; the precisions add, so it gets narrower.

### 3. 把 $\mathbf{x}_0$ 換成雜訊（Replacing $\mathbf{x}_0$ with the noise）

$\tilde\mu$ 裡有 $\mathbf{x}_0$。生成的時候手上沒有 $\mathbf{x}_0$，所以網路得去「猜」某個東西。猜什麼最好？KP02 的閉式 $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ 說：已知 $\mathbf{x}_t$ 時，知道 $\mathbf{x}_0$ 就等於知道 $\epsilon$，反過來也一樣：

$\tilde\mu$ contains $\mathbf{x}_0$. During generation there is no $\mathbf{x}_0$ at hand, so the network must "guess" something. What is best to guess? KP02's closed form $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ says: given $\mathbf{x}_t$, knowing $\mathbf{x}_0$ is the same as knowing $\epsilon$, and vice versa:

$$
\mathbf{x}_0=\frac{1}{\sqrt{\bar\alpha_t}}\left(\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\,\epsilon\right).
$$

代進式 (35) 整理 `[補充]`。$\mathbf{x}_t$ 的係數變成 $\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}+\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{(1-\bar\alpha_t)\sqrt{\bar\alpha_t}}$；因為 $\sqrt{\bar\alpha_{t-1}}/\sqrt{\bar\alpha_t}=1/\sqrt{\alpha_t}$，兩項通分後分子是 $\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\bar\alpha_t$，整個係數就是 $1/\sqrt{\alpha_t}$。$\epsilon$ 的係數是 $-\frac{\beta_t\sqrt{1-\bar\alpha_t}}{(1-\bar\alpha_t)\sqrt{\alpha_t}}=-\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}$。合起來就是來源的式 (37)：

Substitute into equation (35) and simplify `[Added]`. The coefficient of $\mathbf{x}_t$ becomes $\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}+\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{(1-\bar\alpha_t)\sqrt{\bar\alpha_t}}$; since $\sqrt{\bar\alpha_{t-1}}/\sqrt{\bar\alpha_t}=1/\sqrt{\alpha_t}$, the common numerator of the two terms is $\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\bar\alpha_t$, and the whole coefficient is $1/\sqrt{\alpha_t}$. The coefficient of $\epsilon$ is $-\frac{\beta_t\sqrt{1-\bar\alpha_t}}{(1-\bar\alpha_t)\sqrt{\alpha_t}}=-\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}$. Together this is the source's equation (37):

$$
\tilde\mu=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\right).
$$

這條式子很好讀：從 $\mathbf{x}_t$ 減掉「一小份雜訊」，再稍微放大一點（除以 $\sqrt{\alpha_t}$；DDPM 排程下它略小於 $1$）。用第2節的數字核對：$\epsilon=(0.5-\sqrt{0.48}\times1)/\sqrt{0.52}\approx-0.267$，雜訊的係數 $0.2/\sqrt{0.52}\approx0.277$，所以 $\tilde\mu=(0.5-0.277\times(-0.267))/\sqrt{0.8}\approx0.642$，和第2節一樣。

This formula reads easily: subtract "a small portion of noise" from $\mathbf{x}_t$, then scale up slightly (divide by $\sqrt{\alpha_t}$, a little below $1$ under the DDPM schedule). Check it with Section 2's numbers: $\epsilon=(0.5-\sqrt{0.48}\times1)/\sqrt{0.52}\approx-0.267$, the noise coefficient is $0.2/\sqrt{0.52}\approx0.277$, so $\tilde\mu=(0.5-0.277\times(-0.267))/\sqrt{0.8}\approx0.642$, the same as Section 2.

常見錯誤：以為式 (37) 的 $\epsilon$ 是「這一步」加的雜訊。它是從 $\mathbf{x}_0$ 一步跳到 $\mathbf{x}_t$ 時用的那個合起來的雜訊（KP02 第3節），代表 $\mathbf{x}_t$ 裡全部的雜訊，不是第 $t$ 步單獨加的那一份。

Common mistake: thinking the $\epsilon$ in equation (37) is the noise added at "this step". It is the combined noise used to jump from $\mathbf{x}_0$ to $\mathbf{x}_t$ (KP02 Section 3), standing for all the noise in $\mathbf{x}_t$, not the portion added at step $t$ alone.

### 4. 讓網路預測雜訊（Let the network predict the noise）

既然真正的平均只差一個 $\epsilon$，DDPM 就讓網路去猜它：網路輸入 $\mathbf{x}_t$ 和 $t$，輸出 $\epsilon_\theta(\mathbf{x}_t,t)$，再用同一條公式組出逆向一步的平均（來源的式 (38)）：

Since the true mean is only one $\epsilon$ away, DDPM has the network guess it: the network takes $\mathbf{x}_t$ and $t$, outputs $\epsilon_\theta(\mathbf{x}_t,t)$, and the same formula builds the backward-step mean (the source's equation (38)):

$$
\mu_\theta(\mathbf{x}_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(\mathbf{x}_t,t)\right).
$$

和式 (37) 對照，唯一的差別是把真正的 $\epsilon$ 換成網路的猜測 $\epsilon_\theta$。`[補充]` KP01 用的「完美去雜訊器」就是這條式子，只是把 $\epsilon_\theta$ 換成在那個簡單資料上可以直接算出的「最好的猜測」，也就是已知 $\mathbf{x}_t$ 時 $\epsilon$ 的條件平均 $E[\epsilon\mid\mathbf{x}_t]$；KP01 說的是「$x_0$ 平均在哪裡」，兩者用閉式互換。

Compared with equation (37), the only difference is that the true $\epsilon$ is replaced by the network's guess $\epsilon_\theta$. `[Added]` The "perfect denoiser" used in KP01 is exactly this formula, with $\epsilon_\theta$ replaced by the "best guess" that can be computed directly for that simple data, namely the conditional mean $E[\epsilon\mid\mathbf{x}_t]$ of $\epsilon$ given $\mathbf{x}_t$; KP01 phrased it as "where $x_0$ lies on average", and the two convert by the closed form.

`[補充]` 網路其實有三種等價的選擇：直接輸出平均 $\mu_\theta$、輸出對 $\mathbf{x}_0$ 的猜測，或輸出對 $\epsilon$ 的猜測；給定 $\mathbf{x}_t$ 和 $t$，三者可以用公式互換。為什麼選雜訊？

`[Added]` The network actually has three equivalent choices: output the mean $\mu_\theta$ directly, output a guess of $\mathbf{x}_0$, or output a guess of $\epsilon$; given $\mathbf{x}_t$ and $t$, the three convert into one another by formula. Why choose the noise?

- **答案就在手上**：訓練時 $\epsilon$ 是我們自己抽的，拿來當正確答案最直接（KP05 的損失就是猜雜訊的平方誤差）。
- **大小一致**：不管哪一步，$\epsilon$ 都是標準常態，標準差 $1$。直接輸出平均的話，網路的輸出幾乎就是 $\mathbf{x}_t$ 本身，真正有用的修正量 $\tilde\mu-\mathbf{x}_t$ 非常小，網路要在一個大數上精確地調一點點（見下圖）。你可能會想：那就讓網路只輸出修正量。式 (38) 其實正是這麼做：$\mu_\theta=\mathbf{x}_t/\sqrt{\alpha_t}-c_t\,\epsilon_\theta$，$c_t=\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}$ 是一個很小的數（DDPM 排程下在 $0.006$ 到 $0.020$ 之間），等於把修正量除以 $c_t$，放大成大小約 $1$ 的 $\epsilon$ 再交給網路。
- **實驗結果**：DDPM 的作者比較過，預測雜訊配上 KP05 的簡化損失，樣本品質最好。它也和本週 KP08 的 score matching 有直接關係。

- **The answer is in hand**: during training we draw $\epsilon$ ourselves, so it is the most direct correct answer (KP05's loss is the squared error of guessing the noise).
- **Consistent size**: at every step $\epsilon$ is standard normal with standard deviation $1$. If the network outputs the mean directly, its output is almost $\mathbf{x}_t$ itself, and the useful correction $\tilde\mu-\mathbf{x}_t$ is tiny, so the network must adjust a large number by a small, precise amount (see the figure below). You might think: then let the network output only the correction. Equation (38) does exactly that: $\mu_\theta=\mathbf{x}_t/\sqrt{\alpha_t}-c_t\,\epsilon_\theta$, where $c_t=\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}$ is a small number (between $0.006$ and $0.020$ under the DDPM schedule); this divides the correction by $c_t$, scaling it up to an $\epsilon$ of size about $1$ before handing it to the network.
- **Experiments**: the DDPM authors compared the options, and predicting the noise together with KP05's simplified loss gave the best sample quality. It is also directly related to score matching in this week's KP08.

![一張雙對數座標的折線圖，橫軸是步數 t 從 1 到 1000，縱軸是樣本間的標準差，從 0.001 到 5。紫色實線「目標 epsilon」在 1 附近水平；綠色點虛線「目標 x0」在約 2 附近水平；橘色虛線「平均的修正量 mu tilde 減 x_t」在 0.006 到 0.01 之間，比紫線小約一百到一百六十倍、比綠線小約兩百到三百倍。標題寫：網路要輸出的東西有多大（A log-log line chart with step t from 1 to 1000 on the horizontal axis and the standard deviation over samples, from 0.001 to 5, on the vertical axis. The solid purple line, target epsilon, is flat near 1; the dash-dot green line, target x0, is flat near 2; the dashed orange line, mean correction mu tilde minus x_t, lies between 0.006 and 0.01, about 100 to 160 times smaller than the purple line and 200 to 300 times smaller than the green one. The title reads: how large is the thing the network must output?)](../assets/figures/lec14-kp04-target-scale.png)

**圖說 `lec14-kp04-target-scale`**〔功能：對照〕：用 KP01 的兩堆資料與 DDPM 排程，在每一步量三種「網路要輸出的東西」有多大 `[補充]`。紫線是雜訊 $\epsilon$，每一步都是 $1.00$；綠線是原資料 $x_0$，固定約 $2.02$；橘線是「平均減掉輸入」的修正量，只有 $0.006$ 到 $0.010$。如果網路直接輸出平均，它要先複製 $x_t$，再加上這個小一百倍以上的修正；改成預測雜訊，要學的東西在每一步都是標準差 $1$。

**Caption `lec14-kp04-target-scale`** [Function: comparison]: using KP01's two piles of data and the DDPM schedule, it measures at every step how large three kinds of "things the network must output" are `[Added]`. The purple line is the noise $\epsilon$, $1.00$ at every step; the green line is the original data $x_0$, about $2.02$ throughout; the orange line is the correction "mean minus input", only $0.006$ to $0.010$. If the network outputs the mean directly, it must copy $x_t$ and then add this correction, over a hundred times smaller; predicting the noise instead, what must be learned has standard deviation $1$ at every step.

常見錯誤：以為「預測雜訊」和「預測平均」是兩個不同的模型，會學到不同的東西。它們只是同一個量的不同寫法，可以用公式互換；差別在於網路學起來哪一種比較容易。

Common mistake: thinking "predicting the noise" and "predicting the mean" are two different models that learn different things. They are different ways of writing the same quantity and convert into each other by formula; the difference is which one is easier for the network to learn.

### 本單元重點整理（KP04 summary）

**一句話重點**：已知 $\mathbf{x}_0$ 時，真正的去噪一步是常態 $\mathcal{N}(\tilde\mu,\tilde\beta_t I)$，可用貝氏定理與配平方算出；把 $\tilde\mu$ 裡的 $\mathbf{x}_0$ 換成雜訊後，$\tilde\mu=\frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon)$，所以 DDPM 讓網路預測雜訊 $\epsilon_\theta$。

**One-sentence summary**: given $\mathbf{x}_0$, the true denoising step is the normal $\mathcal{N}(\tilde\mu,\tilde\beta_t I)$, computable with Bayes' rule and completing the square; replacing the $\mathbf{x}_0$ in $\tilde\mu$ with the noise gives $\tilde\mu=\frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon)$, so DDPM has the network predict the noise $\epsilon_\theta$.

**白話說明**：兩個目擊者各說了一個位置，一個從終點往回推、一個從起點往前推。把兩份證詞合起來，答案會落在兩者之間、比較相信說得比較肯定的那一位，而且比任何一位單獨說的都更確定。至於網路，與其叫它直接說「上一步在哪」，不如叫它說「照片裡混進了什麼雜訊」，再由公式把雜訊扣掉。

**Plain explanation**: two witnesses each name a location, one working back from the end and one working forward from the start. Combining their testimony puts the answer between the two, trusting more the one who spoke more confidently, and more certain than either alone. As for the network, rather than asking it directly "where was the previous step?", ask it "what noise got mixed into this picture?", and let the formula subtract it.

**正式定義**：$q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_{t-1};\tilde\mu(\mathbf{x}_t,\mathbf{x}_0),\tilde\beta_t I)$，其中 $\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$，$\tilde\mu=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t+\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\mathbf{x}_0=\frac{1}{\sqrt{\alpha_t}}\big(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\big)$；網路的平均 $\mu_\theta=\frac{1}{\sqrt{\alpha_t}}\big(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\mathbf{x}_t,t)\big)$。

**Formal definition**: $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_{t-1};\tilde\mu(\mathbf{x}_t,\mathbf{x}_0),\tilde\beta_t I)$ with $\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ and $\tilde\mu=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t+\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\mathbf{x}_0=\frac{1}{\sqrt{\alpha_t}}\big(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\big)$; the network's mean is $\mu_\theta=\frac{1}{\sqrt{\alpha_t}}\big(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\mathbf{x}_t,t)\big)$.

**例子**：$x_0=1$、$x_t=0.5$、$\beta_t=0.2$、$\bar\alpha_{t-1}=0.6$：兩個建議的平均 $0.559$（精確度 $4$）與 $0.775$（精確度 $2.5$），合起來 $\tilde\mu\approx0.642$、$\tilde\beta_t\approx0.1538$；用雜訊寫法 $\epsilon\approx-0.267$ 也得到 $0.642$。

**Example**: $x_0=1$, $x_t=0.5$, $\beta_t=0.2$, $\bar\alpha_{t-1}=0.6$: the two suggestions have means $0.559$ (precision $4$) and $0.775$ (precision $2.5$), combining to $\tilde\mu\approx0.642$ and $\tilde\beta_t\approx0.1538$; the noise form with $\epsilon\approx-0.267$ also gives $0.642$.

**常見迷思**：

- 「$q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$ 是假設成常態的。」錯，兩個常態相乘一定是常態形狀，是算出來的。
- 「合起來的分布比兩個建議都寬。」錯，精確度相加，會更窄。
- 「式 (37) 的 $\epsilon$ 是第 $t$ 步單獨加的雜訊。」錯，是從 $\mathbf{x}_0$ 跳到 $\mathbf{x}_t$ 的全部雜訊。
- 「預測雜訊和預測平均是不同的模型。」錯，兩者可以用公式互換，差在好不好學。

**Common misconceptions**:

- "$q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$ is assumed to be normal." Wrong: the product of two normals always has a normal shape; it is computed.
- "The combined distribution is wider than both suggestions." Wrong: the precisions add, so it is narrower.
- "The $\epsilon$ in equation (37) is the noise added at step $t$ alone." Wrong: it is all the noise of the jump from $\mathbf{x}_0$ to $\mathbf{x}_t$.
- "Predicting the noise and predicting the mean are different models." Wrong: they convert into each other by formula; the difference is how easy each is to learn.

**圖**：本單元有兩張圖。

- `lec14-kp04-posterior`：兩個常態建議相乘，得到更窄的真正去噪分布，並以抽樣核對。
- `lec14-kp04-target-scale`：在各步比較雜訊、原資料與平均修正量的大小。

**Figures** — this unit has two figures.

- `lec14-kp04-posterior`: two normal suggestions multiply into a narrower true denoising distribution, checked by sampling.
- `lec14-kp04-target-scale`: the sizes of the noise, the original data and the mean correction compared across steps.

### 小練習（Practice）

1. 設 $\beta_t=0.1$、$\bar\alpha_{t-1}=0.5$。算 $\alpha_t$、$\bar\alpha_t$、$\tilde\beta_t$，以及 $\tilde\mu$ 裡 $x_t$ 與 $x_0$ 的係數。
2. 承上題，若 $x_t=0.4$、$x_0=1$，算 $\tilde\mu$。再算出對應的 $\epsilon$，用式 (37) 驗算。
3. 看 $\tilde\beta_t$ 的公式：$t=1$ 時 $\bar\alpha_0=1$，$\tilde\beta_1$ 是多少？這合理嗎？
4. 一位同學說：「既然 $\tilde\mu$ 可以用 $\mathbf{x}_0$ 寫，也可以用 $\epsilon$ 寫，那讓網路預測 $\mathbf{x}_0$ 也一樣。」這句話哪裡對、哪裡不完整？

1. Let $\beta_t=0.1$ and $\bar\alpha_{t-1}=0.5$. Compute $\alpha_t$, $\bar\alpha_t$, $\tilde\beta_t$, and the coefficients of $x_t$ and $x_0$ in $\tilde\mu$.
2. Continuing, with $x_t=0.4$ and $x_0=1$, compute $\tilde\mu$. Then compute the corresponding $\epsilon$ and check with equation (37).
3. Look at the formula for $\tilde\beta_t$: at $t=1$, $\bar\alpha_0=1$; what is $\tilde\beta_1$? Does it make sense?
4. A classmate says: "Since $\tilde\mu$ can be written with $\mathbf{x}_0$ or with $\epsilon$, having the network predict $\mathbf{x}_0$ is just the same." What is right about this, and what is incomplete?

**第 1 題解答** `[補充]`：$\alpha_t=0.9$，$\bar\alpha_t=0.45$。$\tilde\beta_t=0.5\times0.1/0.55\approx0.0909$。$x_t$ 的係數 $\sqrt{0.9}\times0.5/0.55\approx0.8624$，$x_0$ 的係數 $\sqrt{0.5}\times0.1/0.55\approx0.1286$。

**Solution to question 1** `[Added]`: $\alpha_t=0.9$ and $\bar\alpha_t=0.45$. $\tilde\beta_t=0.5\times0.1/0.55\approx0.0909$. The coefficient of $x_t$ is $\sqrt{0.9}\times0.5/0.55\approx0.8624$, and that of $x_0$ is $\sqrt{0.5}\times0.1/0.55\approx0.1286$.

**第 2 題解答** `[補充]`：$\tilde\mu=0.8624\times0.4+0.1286\times1\approx0.4735$。$\epsilon=(0.4-\sqrt{0.45})/\sqrt{0.55}\approx-0.3652$；式 (37)：$(0.4-\frac{0.1}{\sqrt{0.55}}\times(-0.3652))/\sqrt{0.9}=(0.4+0.0492)/0.9487\approx0.4735$，一致。

**Solution to question 2** `[Added]`: $\tilde\mu=0.8624\times0.4+0.1286\times1\approx0.4735$. $\epsilon=(0.4-\sqrt{0.45})/\sqrt{0.55}\approx-0.3652$; equation (37): $(0.4-\frac{0.1}{\sqrt{0.55}}\times(-0.3652))/\sqrt{0.9}=(0.4+0.0492)/0.9487\approx0.4735$, which agrees.

**第 3 題解答** `[補充]`：$\tilde\beta_1=\frac{1-1}{1-\bar\alpha_1}\beta_1=0$。合理：$t=1$ 時「上一步」就是 $\mathbf{x}_0$ 本身，已知 $\mathbf{x}_0$，就完全確定，沒有任何不確定性。實際上 $t=1$ 由 KP03 的重建項處理（去噪 KL 只到 $t\ge2$），所以 $\tilde\beta_1=0$ 只是公式在邊界的值。在 DDPM 排程下，$\tilde\beta_t$ 很快就接近 $\beta_t$：$t=2$ 時是 $\beta_t$ 的 $0.45$ 倍，$t=10$ 時 $0.85$ 倍，$t=100$ 時 $0.98$ 倍。

**Solution to question 3** `[Added]`: $\tilde\beta_1=\frac{1-1}{1-\bar\alpha_1}\beta_1=0$. It makes sense: at $t=1$ "the previous step" is $\mathbf{x}_0$ itself, which is known, so there is no uncertainty at all. In practice $t=1$ is handled by KP03's reconstruction term (the denoising KLs run over $t\ge2$), so $\tilde\beta_1=0$ is just the formula's value at the boundary. Under the DDPM schedule, $\tilde\beta_t$ quickly approaches $\beta_t$: it is $0.45$ times $\beta_t$ at $t=2$, $0.85$ times at $t=10$ and $0.98$ times at $t=100$.

**第 4 題解答** `[補充]`：對的部分：給定 $\mathbf{x}_t$ 和 $t$，猜 $\mathbf{x}_0$ 和猜 $\epsilon$ 可以用閉式互換，所以兩種寫法能表達同樣的平均。不完整的部分：好不好學不一樣。網路的誤差換算成平均的誤差時，要乘上一個倍數：猜 $\mathbf{x}_0$ 時這個倍數從 $t=1$ 的 $1$ 降到 $t=1000$ 的約 $1.3\times10^{-4}$，差了四個數量級；猜 $\epsilon$ 時只在 $0.006$ 到 $0.020$ 之間。同樣的網路誤差，在不同步造成的影響差很多，損失的各步權重就得另外調。訓練時用哪一種寫法、配上什麼權重，會影響結果；DDPM 實驗發現預測雜訊配簡化損失最好（KP05）。

**Solution to question 4** `[Added]`: what is right: given $\mathbf{x}_t$ and $t$, guessing $\mathbf{x}_0$ and guessing $\epsilon$ convert into each other through the closed form, so both can express the same mean. What is incomplete: how easy they are to learn differs. When the network's error is converted into an error in the mean, it is multiplied by a factor: for guessing $\mathbf{x}_0$ the factor falls from $1$ at $t=1$ to about $1.3\times10^{-4}$ at $t=1000$, four orders of magnitude; for guessing $\epsilon$ it stays between $0.006$ and $0.020$. The same network error has very different effects at different steps, so the per-step weights of the loss must be tuned separately. Which form is trained, and with what weights, affects the result; the DDPM experiments found that predicting the noise with the simplified loss works best (KP05).

### 小測驗（Quiz，來源原題）

來源 KP-6 與 KP-7 各有四題自我檢測，保留原題與全部選項；來源頁面附有答案與簡短解析，下面的解析由本講義依各節內容補寫。

The source's KP-6 and KP-7 each have four self-check questions, kept here with all their options; the source page gives the answers with short explanations, and the explanations below are written by this note from its own sections.

1. 為什麼 $q(x_{t-1}\mid x_t,x_0)$ 是可解的？
   - (A) 因為它條件於原始資料 $x_0$
   - (B) 因為它是 uniform
   - (C) 因為它沒有雜訊
   - (D) 因為它是 prior
2. 求 $q(x_{t-1}\mid x_t,x_0)$ 時使用了什麼數學工具？
   - (A) Bayes 規則與配平方
   - (B) Fourier transform
   - (C) Monte Carlo
   - (D) 梯度下降
3. $q(x_{t-1}\mid x_t,x_0)$ 被稱為什麼？
   - (A) Forward distribution
   - (B) Ground-truth denoising distribution
   - (C) Prior distribution
   - (D) Variational posterior
4. 在式 (33) 中，哪一項對應 forward transition？
   - (A) $q(x_t\mid x_{t-1},x_0)$
   - (B) $q(x_{t-1}\mid x_0)$
   - (C) $q(x_t\mid x_0)$
   - (D) $p(x_T)$
5. 式 (37) 中 $\tilde\mu$ 被表示成什麼的函數？
   - (A) $x_0$
   - (B) $x_t$ 與雜訊 $\epsilon$
   - (C) $\beta_t$ 與 $\alpha_t$ 的比值
   - (D) $p(x_T)$
6. DDPM 的神經網路主要學習預測什麼？
   - (A) 原始資料 $x_0$
   - (B) 加入的雜訊 $\epsilon$
   - (C) $\beta_t$ 排程
   - (D) $\alpha_t$
7. 為什麼預測雜訊比預測均值更方便？
   - (A) 雜訊的目標就是抽樣時的標準高斯
   - (B) 均值總是零
   - (C) 雜訊不需要時間 $t$
   - (D) 均值無法解析計算
8. 式 (38) 中的 $\mu_\theta$ 與式 (37) 的 $\tilde\mu$ 關係是什麼？
   - (A) 完全無關
   - (B) $\mu_\theta$ 用 $\epsilon_\theta$ 取代 $\epsilon$
   - (C) $\mu_\theta$ 是 forward mean
   - (D) $\mu_\theta$ 忽略 $x_t$

1. Why is $q(x_{t-1}\mid x_t,x_0)$ solvable?
   - (A) Because it is conditioned on the original data $x_0$
   - (B) Because it is uniform
   - (C) Because it has no noise
   - (D) Because it is a prior
2. What mathematical tools are used to find $q(x_{t-1}\mid x_t,x_0)$?
   - (A) Bayes' rule and completing the square
   - (B) Fourier transform
   - (C) Monte Carlo
   - (D) Gradient descent
3. What is $q(x_{t-1}\mid x_t,x_0)$ called?
   - (A) Forward distribution
   - (B) Ground-truth denoising distribution
   - (C) Prior distribution
   - (D) Variational posterior
4. In equation (33), which term is the forward transition?
   - (A) $q(x_t\mid x_{t-1},x_0)$
   - (B) $q(x_{t-1}\mid x_0)$
   - (C) $q(x_t\mid x_0)$
   - (D) $p(x_T)$
5. In equation (37), $\tilde\mu$ is written as a function of what?
   - (A) $x_0$
   - (B) $x_t$ and the noise $\epsilon$
   - (C) The ratio of $\beta_t$ to $\alpha_t$
   - (D) $p(x_T)$
6. What does DDPM's neural network mainly learn to predict?
   - (A) The original data $x_0$
   - (B) The added noise $\epsilon$
   - (C) The $\beta_t$ schedule
   - (D) $\alpha_t$
7. Why is predicting the noise more convenient than predicting the mean?
   - (A) The noise target is exactly the standard Gaussian used in sampling
   - (B) The mean is always zero
   - (C) The noise does not need the time $t$
   - (D) The mean cannot be computed analytically
8. How is $\mu_\theta$ in equation (38) related to $\tilde\mu$ in equation (37)?
   - (A) Completely unrelated
   - (B) $\mu_\theta$ replaces $\epsilon$ with $\epsilon_\theta$
   - (C) $\mu_\theta$ is the forward mean
   - (D) $\mu_\theta$ ignores $x_t$

**解析** `[補充]`：

- 第 1 題選 (A)：第1節，條件於 $x_0$ 之後，貝氏定理右邊三項都是 KP02 寫好的常態。
- 第 2 題選 (A)：第1、2節。
- 第 3 題選 (B)：第1節的名稱。(D)「變分後驗」是 VAE 裡要學的編碼器分布；這裡的分布雖然也是一種後驗，卻是算出來的，不用學。
- 第 4 題選 (A)：它是前向一步，因為馬可夫性質，多出的條件 $x_0$ 不影響它。(B)、(C) 是一步跳的閉式。
- 第 5 題選 (B)：第3節，把 $x_0$ 換掉之後只剩 $x_t$ 和 $\epsilon$。
- 第 6 題選 (B)：第4節。
- 第 7 題選 (A)：意思是「雜訊是我們從標準常態自己抽出來的，訓練時答案已知，大小也固定」，見第4節的三個理由。(D) 錯，平均在已知 $x_0$ 時可以算，第2節就算了。
- 第 8 題選 (B)：第4節，兩式只差把 $\epsilon$ 換成 $\epsilon_\theta$。

**Explanations** `[Added]`:

- Question 1: A: Section 1; once conditioned on $x_0$, the three terms on the right of Bayes' rule are all normals written out in KP02.
- Question 2: A: Sections 1 and 2.
- Question 3: B: the name used in Section 1. D, "variational posterior", is the encoder distribution learned in a VAE; the distribution here is also a posterior, but it is computed, not learned.
- Question 4: A: it is the forward step; by the Markov property the extra condition $x_0$ does not affect it. B and C are one-jump closed forms.
- Question 5: B: Section 3; after replacing $x_0$, only $x_t$ and $\epsilon$ remain.
- Question 6: B: Section 4.
- Question 7: A: meaning "we draw the noise ourselves from a standard normal, so the answer is known in training and its size is fixed"; see the three reasons in Section 4. D is wrong: given $x_0$ the mean can be computed, as Section 2 did.
- Question 8: B: Section 4; the two formulas differ only by replacing $\epsilon$ with $\epsilon_\theta$.

### 留給下一單元的問題（The question for the next unit）

網路現在要輸出 $\epsilon_\theta$。KP03 的每一步 KL，代入兩個常態之後會變成什麼？是不是就是「猜的雜訊和真的雜訊差多少」？逆向一步的變異數 $\sigma_t^2$ 又該選 $\beta_t$ 還是 $\tilde\beta_t$？訓練和生成實際上怎麼一步一步做？下一個知識點（本週 KP05）把損失寫出來，並整理成 DDPM 的兩個演算法。

The network now outputs $\epsilon_\theta$. What does each step's KL from KP03 become once the two normals are substituted? Is it just "how far the guessed noise is from the true noise"? Should the backward-step variance $\sigma_t^2$ be $\beta_t$ or $\tilde\beta_t$? How are training and generation actually carried out step by step? The next knowledge point (this week's KP05) writes the loss out and organises it into DDPM's two algorithms.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第15章「Diffusion Models」的 KP-6「Ground-truth Denoising：Bayes 與配平方」與 KP-7「Denoising 均值閉式 $\tilde\mu$ 與 $\epsilon_\theta$ 重參數化」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch15-diffusion-models/index.html ，讀取日期 2026-11-15）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from KP-6, "Ground-truth denoising: Bayes and completing the square", and KP-7, "The closed-form denoising mean $\tilde\mu$ and the $\epsilon_\theta$ reparameterization", of Chapter 15, "Diffusion Models", of CCC-LNS IDL_115 (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch15-diffusion-models/index.html , accessed 2026-11-15). Content not in the source is marked `[Added]` where it appears.

- 來源的式 (32) 寫「假設它也是高斯」，並說「covariance 通常設為 $\tilde\sigma_t^2 I$」。兩者都不是假設或選擇：兩個常態密度相乘一定是常態，共變異數也算得出來，正是 $\tilde\beta_t I$。本講義第1、2節照此寫。
- 來源互動展示寫後驗均值為 $\frac{\beta_t x_0+\alpha_t(1-\bar\alpha_{t-1})x_t}{1-\bar\alpha_t}$，兩個係數都少了根號：正確是 $\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t x_0+\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_t}{1-\bar\alpha_t}$（來源自己的式 (35)）。用展示的預設數字，錯誤寫法給 $0.692$，正確是 $0.642$，抽樣核對也是 $0.642$。
- 來源小測驗第 7 題的選項 (A)「雜訊的目標就是抽樣時的標準高斯」說法含糊，來源解析給的理由是「訓練時已知加入的雜訊」。本講義第4節把理由寫完整：答案已知、各步大小一致、實驗效果較好，並在解析中說明 (A) 的意思。
- 來源寫「不如讓網路直接預測被加進去的雜訊」，容易讓人以為預測平均是錯的；本講義第4節說明三種寫法等價，差在好不好學。
- 本講義補上：分母為何是常數、兩個常態相乘的規則（精確度相加、依精確度加權）、配平方的通分與 $\tilde\beta_t$ 的推導、來源展示數字的手算與 $400$ 萬次抽樣核對、式 (35) 到式 (37) 的逐項化簡與數字驗算、$\epsilon$ 是合起來的雜訊、KP01 完美去雜訊器與式 (38) 的關係、三種預測目標的大小比較圖、四題小練習與解答、小測驗的解析。
- 取捨：來源兩個互動展示不重做，改用後驗圖與目標大小圖；符號加粗與否的取捨同 KP02。

- The source's equation (32) says "assume it is also Gaussian" and that "the covariance is usually set to $\tilde\sigma_t^2 I$". Neither is an assumption or a choice: the product of two normal densities is always normal, and its covariance can be computed, namely $\tilde\beta_t I$. Sections 1 and 2 follow this.
- The source's interactive demo gives the posterior mean as $\frac{\beta_t x_0+\alpha_t(1-\bar\alpha_{t-1})x_t}{1-\bar\alpha_t}$, with both coefficients missing their square roots: the correct form is $\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t x_0+\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_t}{1-\bar\alpha_t}$ (the source's own equation (35)). With the demo's default numbers, the wrong form gives $0.692$, while the correct value is $0.642$, which the sampling check also gives.
- Option A of the source's quiz question 7, "the noise target is exactly the standard Gaussian used in sampling", is vague, and the source's explanation gives the reason "the added noise is known during training". Section 4 states the reasons fully (the answer is known, the size is consistent across steps, and it works better in experiments), and the explanation says what A means.
- The source writes "better to have the network predict the added noise directly", which may suggest predicting the mean is wrong; Section 4 explains that the three forms are equivalent and differ in how easy they are to learn.
- This note adds: why the denominator is a constant; the rule for multiplying two normals (precisions add, precision-weighted mean); the common-denominator step of completing the square and the derivation of $\tilde\beta_t$; a hand calculation with the source demo's numbers and a check with $4$ million samples; the term-by-term simplification from equation (35) to (37) with a numeric check; the point that $\epsilon$ is the combined noise; the link between KP01's perfect denoiser and equation (38); the figure comparing the sizes of the three prediction targets; four practice questions with solutions; and the explanations of the quiz.
- Choices: the source's two interactive demos are not rebuilt; the posterior figure and the target-size figure are used instead. Bold versus plain symbols follow the choice made in KP02.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 真正的去噪分布 | ground-truth denoising distribution |
| 後驗分布 | posterior distribution |
| 貝氏定理 | Bayes' rule |
| 配平方 | completing the square |
| 精確度 | precision |
| 加權平均 | weighted average |
| 正比於 | proportional to |
| 預測雜訊 | noise prediction |
| 參數化 | parameterization |
| 變分後驗（第12週） | variational posterior |
| 先驗分布 | prior distribution |
| 分數匹配（KP08） | score matching |
| 去雜訊擴散機率模型 | denoising diffusion probabilistic model (DDPM) |
| 蒙地卡羅 | Monte Carlo |
| 傅立葉轉換 | Fourier transform |
| 梯度下降 | gradient descent |
| 均勻分布 | uniform distribution |
| 證據下界（KP03） | evidence lower bound (ELBO) |
| KL 散度 | Kullback–Leibler (KL) divergence |
| 變分自編碼器（第12週） | variational autoencoder (VAE) |
