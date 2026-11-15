# 第14週 KP02：前向過程與任意一步的閉式（The forward process and a closed form for any step）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第14週「深度生成模型（三）：擴散模型」的第二個知識點（KP02）。KP01 用一條簡單的規則看過前向過程：每一步先把舊值縮小一點，再加一點雜訊。這一節把它寫成正式的機率分布，說明雜訊量 $\beta_t$ 怎麼安排，並推出一條很有用的公式：不必真的走 $t$ 步，就能一步算出第 $t$ 步的樣子。

This is the second knowledge point (KP02) of Week 14, "Deep generative models (3): diffusion models". KP01 looked at the forward process through a simple rule: each step first shrinks the old value a little and then adds a little noise. This note writes it as a formal probability distribution, explains how the noise amounts $\beta_t$ are scheduled, and derives a very useful formula: without actually taking $t$ steps, you can compute what step $t$ looks like in one go.

先備知識是本週 KP01 的前向過程與手算例子，以及常態分布的兩個性質：常態變數乘上一個常數，還是常態；兩個互相獨立的常態變數相加，還是常態，而且變異數相加。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第15章 KP-2（2.1 節）與 KP-5（2.4.1 節）改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec14-kp02-figures.py` 算出，可以自己重算。

The prerequisites are this week's KP01, with its forward process and hand calculation, and two properties of the normal distribution: a normal variable times a constant is still normal; and the sum of two independent normal variables is still normal, with the variances adding up. Every term and symbol is defined when it first appears. This note is rewritten from KP-2 (Section 2.1) and KP-5 (Section 2.4.1) of Chapter 15 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec14-kp02-figures.py`, so you can recompute it yourself.

## KP02 前向過程與任意一步的閉式（The forward process and a closed form for any step）

### 學習目標（Learning objectives）

讀完之後，你能寫出前向過程一步的條件分布 $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$，並說出它的平均與變異數各代表什麼。你能說明「馬可夫鏈」的意思，並把整條鏈的聯合分布寫成連乘。你能說出雜訊排程 $\beta_t$ 的常見設定，並由它算出 $\alpha_t$ 與 $\bar\alpha_t$。你能用兩步的例子推出閉式 $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\epsilon$，並用它直接算出第 $t$ 步的平均與變異數。你也能說明這條公式為什麼讓訓練變快，以及它不能用來做什麼。

After reading, you can write the conditional distribution $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ of one forward step and say what its mean and variance stand for. You can explain what a "Markov chain" means and write the joint distribution of the whole chain as a product. You can state the usual setting of the noise schedule $\beta_t$ and compute $\alpha_t$ and $\bar\alpha_t$ from it. You can derive the closed form $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ from a two-step example and use it to compute the mean and variance of step $t$ directly. You can also explain why this formula makes training faster, and what it cannot be used for.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{x}_0$ | 一筆真實資料 | one real example |
| $\mathbf{x}_t$ | 加了 $t$ 步雜訊之後的資料 | the data after $t$ noising steps |
| $\mathbf{x}_{1:T}$ | $\mathbf{x}_1,\mathbf{x}_2,\ldots,\mathbf{x}_T$ 整串 | the whole chain $\mathbf{x}_1,\mathbf{x}_2,\ldots,\mathbf{x}_T$ |
| $T$ | 總步數 | the total number of steps |
| $\beta_t$ | 第 $t$ 步加入的雜訊**變異數** | the noise **variance** added at step $t$ |
| $\alpha_t$ | $\alpha_t=1-\beta_t$，第 $t$ 步保留的變異數比例 | $\alpha_t=1-\beta_t$, the share of variance kept at step $t$ |
| $\bar\alpha_t$ | $\bar\alpha_t=\alpha_1\alpha_2\cdots\alpha_t$ | $\bar\alpha_t=\alpha_1\alpha_2\cdots\alpha_t$ |
| $\epsilon$ | 標準常態雜訊，和資料一樣大，每個元素各自獨立 | standard-normal noise, the same size as the data, each element independent |
| $\mathcal{N}(\mathbf{x};\mu,\Sigma)$ | 「$\mathbf{x}$ 服從平均 $\mu$、共變異數 $\Sigma$ 的常態分布」 | "$\mathbf{x}$ follows a normal distribution with mean $\mu$ and covariance $\Sigma$" |
| $I$ | 單位矩陣；$\beta_t I$ 表示每個元素的變異數都是 $\beta_t$、彼此無關 | identity matrix; $\beta_t I$ means every element has variance $\beta_t$, unrelated to the others |
| $q(\cdot\mid\cdot)$ | 前向過程的條件分布，固定不用學 | conditional distribution of the forward process, fixed and not learned |
| $\prod$ | 連乘符號，$\prod_{t=1}^{3}a_t=a_1a_2a_3$ | product sign, $\prod_{t=1}^{3}a_t=a_1a_2a_3$ |

### 1. 前向過程是一條高斯馬可夫鏈（The forward process is a Gaussian Markov chain）

前向過程把一筆資料 $\mathbf{x}_0$ 一步一步變成雜訊：

The forward process turns an example $\mathbf{x}_0$ into noise step by step:

$$
\mathbf{x}_0\rightarrow\mathbf{x}_1\rightarrow\mathbf{x}_2\rightarrow\cdots\rightarrow\mathbf{x}_T .
$$

每一步的規則寫成一個條件分布。來源的式 (2)：給定上一步的 $\mathbf{x}_{t-1}$，新的 $\mathbf{x}_t$ 是常態分布，平均是 $\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}$，共變異數是 $\beta_t I$：

The rule of each step is written as a conditional distribution. The source's equation (2): given the previous $\mathbf{x}_{t-1}$, the new $\mathbf{x}_t$ is normal with mean $\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}$ and covariance $\beta_t I$:

$$
q(\mathbf{x}_t\mid\mathbf{x}_{t-1})=\mathcal{N}\!\left(\mathbf{x}_t;\ \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\ \beta_t I\right).
$$

這和 KP01 的規則 $\mathbf{x}_t=\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}+\sqrt{\beta_t}\,\epsilon$ 是同一件事，只是換個寫法：平均那一項是「舊值縮小」，共變異數那一項是「加上的雜訊」。要注意 $\beta_t$ 是雜訊的**變異數**，雜訊本身的標準差是 $\sqrt{\beta_t}$。$I$ 是單位矩陣，意思是每個像素各自加雜訊、變異數都是 $\beta_t$，像素之間不互相牽連。

This is the same thing as KP01's rule $\mathbf{x}_t=\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}+\sqrt{\beta_t}\,\epsilon$, only written differently: the mean is "shrink the old value", and the covariance is "the noise added". Note that $\beta_t$ is the **variance** of the noise; the noise's standard deviation is $\sqrt{\beta_t}$. $I$ is the identity matrix, meaning each pixel gets its own noise with variance $\beta_t$, and pixels do not influence one another.

每一步只看上一步，不看更早的歷史，這就是**馬可夫性質**（Markov property），來源的式 (4)：

Each step looks only at the previous one, not at earlier history; this is the **Markov property**, the source's equation (4):

$$
q(\mathbf{x}_t\mid\mathbf{x}_{t-1},\mathbf{x}_{t-2},\ldots,\mathbf{x}_0)=q(\mathbf{x}_t\mid\mathbf{x}_{t-1}).
$$

因為每一步只依賴上一步，整串 $\mathbf{x}_{1:T}$ 的機率可以先用機率的乘法規則拆開，再用馬可夫性質把每一項簡化成只看上一步，結果是每一步的連乘（來源的式 (5)）。這和「先擲一次骰子，再依結果決定下一次」的機率是一樣的算法：

Because each step depends only on the previous one, the probability of the whole chain $\mathbf{x}_{1:T}$ is first split with the chain (product) rule of probability, and the Markov property then reduces each factor to depend only on the previous step, giving a product over steps (the source's equation (5)). It is computed the same way as the probability of "roll a die, then decide the next roll from the result":

$$
q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)=\prod_{t=1}^{T}q(\mathbf{x}_t\mid\mathbf{x}_{t-1}).
$$

這條連乘在本週 KP03 推導 ELBO 時會用到。

This product is used in this week's KP03 when deriving the ELBO.

常見錯誤：把 $\beta_t$ 當成雜訊的標準差，於是寫成 $\mathbf{x}_t=\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}+\beta_t\,\epsilon$。$\beta_t$ 是變異數，乘在 $\epsilon$ 前面的是 $\sqrt{\beta_t}$；寫錯的話，KP01 第2節的「變異數保持 $1$」就不成立了。

Common mistake: treating $\beta_t$ as the noise's standard deviation and writing $\mathbf{x}_t=\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}+\beta_t\,\epsilon$. $\beta_t$ is a variance, and what multiplies $\epsilon$ is $\sqrt{\beta_t}$; get it wrong and KP01's "the variance stays at $1$" (Section 2) no longer holds.

### 2. 雜訊排程：$\beta_t$ 從小到大（The noise schedule: $\beta_t$ from small to large）

$\beta_1,\beta_2,\ldots,\beta_T$ 叫做**雜訊排程**（noise schedule），是事先選好的數字。來源說它隨時間遞增（式 (3)）：$\beta_1<\beta_2<\cdots<\beta_T$，越到後面加得越多。最常見的設定來自 DDPM 論文：$T=1000$，$\beta_t$ 從 $0.0001$ 等距增加到 $0.02$。為了少寫一些 $1-\beta_t$，定義

$\beta_1,\beta_2,\ldots,\beta_T$ are called the **noise schedule**, numbers chosen in advance. The source says they increase over time (equation (3)): $\beta_1<\beta_2<\cdots<\beta_T$, with more noise added later on. The most common setting comes from the DDPM paper: $T=1000$, with $\beta_t$ rising evenly from $0.0001$ to $0.02$. To write $1-\beta_t$ less often, define

$$
\alpha_t=1-\beta_t .
$$

`[補充]` 為什麼一開始加得少、後來加得多？資料剛開始還很清楚，細節（例如影像的邊緣與紋理）一點點雜訊就會被蓋掉，所以前面的步伐要小；到後面資料已經很模糊，小步伐幾乎看不出差別，就可以走大步一點。另一個理由在本週 KP03：每一步越小，逆向那一步就越接近常態分布，越容易學。

`[Added]` Why add little at first and more later? Early on the data are still sharp, and fine details such as edges and textures are wiped out by a little noise, so the early steps must be small; later the data are already blurry and small steps barely change anything, so bigger steps are fine. Another reason comes in this week's KP03: the smaller each step, the closer the backward step is to a normal distribution, and the easier it is to learn.

![兩張並排的折線圖，橫軸都是步數 t，從 0 到 1000。左圖：紅色實線是 beta_t，從 0.0001 直線升到 0.02，對應左邊刻度；藍色虛線是 alpha_t 等於 1 減 beta_t，從 1 直線降到 0.98，對應右邊刻度。右圖標題是 x_t 等於根號 alpha bar 乘 x0 加根號 1 減 alpha bar 乘 epsilon：藍色實線是根號 alpha bar，也就是 x0 的權重，從 1 慢慢降到接近 0；紫色虛線是根號 1 減 alpha bar，也就是雜訊的權重，從 0 升到接近 1；兩線在 t 等於 259 交叉，交點約 0.71，有一條灰色直線和黑點標出，底下寫著 t=259 兩者權重相等，都是根號 0.5（Two line charts side by side, both with step t from 0 to 1000 on the horizontal axis. Left: the solid red line is beta_t, rising straight from 0.0001 to 0.02 on the left scale; the dashed blue line is alpha_t = 1 - beta_t, falling straight from 1 to 0.98 on the right scale. Right, titled x_t = sqrt(alpha bar) x0 + sqrt(1 - alpha bar) epsilon: the solid blue line is sqrt(alpha bar), the weight of x0, falling slowly from 1 to near 0; the dashed purple line is sqrt(1 - alpha bar), the weight of the noise, rising from 0 to near 1; the two cross at t = 259 at about 0.71, marked with a grey vertical line and a black dot, with the note that at t = 259 both weights equal sqrt(0.5))](../assets/figures/lec14-kp02-schedule.png)

**圖說 `lec14-kp02-schedule`**〔功能：一目了然、精準度〕：左圖是「每一步」：紅線 $\beta_t$ 從 $0.0001$ 爬到 $0.02$，藍色虛線 $\alpha_t$ 只從 $1$ 掉到 $0.98$，每一步都只改一點點。右圖是「累積 $t$ 步之後」（第3節會推導）：藍線是原資料 $\mathbf{x}_0$ 的權重，紫色虛線是雜訊的權重；兩條在 $t=259$ 交叉，之後雜訊佔上風，到 $t=1000$ 原資料的權重只剩 $0.0064$。左圖兩條線用的是左右兩個不同的刻度，它們的交點沒有意義。

**Caption `lec14-kp02-schedule`** [Function: seeing it at a glance, precision]: the left chart is "per step": the red line $\beta_t$ climbs from $0.0001$ to $0.02$, while the dashed blue $\alpha_t$ only drops from $1$ to $0.98$, so each step changes very little. The right chart is "after $t$ steps in total" (derived in Section 3): the blue line is the weight of the original $\mathbf{x}_0$ and the dashed purple line the weight of the noise; they cross at $t=259$, after which the noise dominates, and by $t=1000$ the original's weight is only $0.0064$. The two lines of the left chart use different scales (left and right), so their crossing means nothing.

一些具體的數字（DDPM 排程）：

Some concrete numbers (DDPM schedule):

| $t$ | $\beta_t$ | $\bar\alpha_t$ | $\sqrt{\bar\alpha_t}$（$\mathbf{x}_0$ 的權重） | $\sqrt{1-\bar\alpha_t}$（雜訊的權重） |
|---|---|---|---|---|
| $1$ | $0.0001$ | $0.9999$ | $0.9999$ | $0.0100$ |
| $100$ | $0.0021$ | $0.8970$ | $0.9471$ | $0.3209$ |
| $250$ | $0.0051$ | $0.5241$ | $0.7239$ | $0.6899$ |
| $500$ | $0.0100$ | $0.07859$ | $0.2803$ | $0.9599$ |
| $750$ | $0.0150$ | $0.003351$ | $0.0579$ | $0.9983$ |
| $1000$ | $0.0200$ | $0.00004036$ | $0.0064$ | $1.0000$ |

| $t$ | $\beta_t$ | $\bar\alpha_t$ | $\sqrt{\bar\alpha_t}$ (weight of $\mathbf{x}_0$) | $\sqrt{1-\bar\alpha_t}$ (weight of noise) |
|---|---|---|---|---|
| $1$ | $0.0001$ | $0.9999$ | $0.9999$ | $0.0100$ |
| $100$ | $0.0021$ | $0.8970$ | $0.9471$ | $0.3209$ |
| $250$ | $0.0051$ | $0.5241$ | $0.7239$ | $0.6899$ |
| $500$ | $0.0100$ | $0.07859$ | $0.2803$ | $0.9599$ |
| $750$ | $0.0150$ | $0.003351$ | $0.0579$ | $0.9983$ |
| $1000$ | $0.0200$ | $0.00004036$ | $0.0064$ | $1.0000$ |

常見錯誤：以為 $\beta_t$ 一定要遞增才行。遞增是常見且好用的選擇，但真正要緊的是走到最後 $\bar\alpha_T$ 接近 $0$，也就是資料被雜訊淹沒。KP01 的手算就用固定的 $\beta=0.36$；本週 KP06 還會看到另一種「cosine」排程。

Common mistake: thinking $\beta_t$ must increase. Increasing is a common and useful choice, but what really matters is that $\bar\alpha_T$ ends up near $0$, that is, the data are drowned in noise. KP01's hand calculation used a constant $\beta=0.36$; this week's KP06 shows another, "cosine" schedule.

### 3. 一步跳到第 $t$ 步（Jumping to step $t$ in one go）

想知道第 $500$ 步長什麼樣，難道要真的走 $500$ 步？不用。因為每一步都是「乘一個數再加常態雜訊」，常態加常態還是常態，所以可以把好幾步合成一步。先看兩步，再推廣。

To see what step $500$ looks like, must we really take $500$ steps? No. Each step is "multiply by a number, then add normal noise", and normal plus normal is still normal, so several steps can be merged into one. First look at two steps, then generalise.

**兩步合成一步。** 第一步是 $\mathbf{x}_1=\sqrt{\alpha_1}\,\mathbf{x}_0+\sqrt{\beta_1}\,\epsilon_1$，第二步是 $\mathbf{x}_2=\sqrt{\alpha_2}\,\mathbf{x}_1+\sqrt{\beta_2}\,\epsilon_2$，$\epsilon_1$、$\epsilon_2$ 是兩次獨立抽出的標準常態雜訊。把 $\mathbf{x}_1$ 代進去：

**Two steps merged into one.** The first step is $\mathbf{x}_1=\sqrt{\alpha_1}\,\mathbf{x}_0+\sqrt{\beta_1}\,\epsilon_1$ and the second is $\mathbf{x}_2=\sqrt{\alpha_2}\,\mathbf{x}_1+\sqrt{\beta_2}\,\epsilon_2$, where $\epsilon_1$ and $\epsilon_2$ are two independent standard-normal draws. Substitute $\mathbf{x}_1$:

$$
\mathbf{x}_2=\sqrt{\alpha_1\alpha_2}\;\mathbf{x}_0+\underbrace{\sqrt{\alpha_2\beta_1}\;\epsilon_1+\sqrt{\beta_2}\;\epsilon_2}_{\text{noise}} .
$$

後面兩項是兩個獨立的常態雜訊相加，合起來還是常態，平均是 $0$，變異數相加：$\alpha_2\beta_1+\beta_2=\alpha_2(1-\alpha_1)+1-\alpha_2=1-\alpha_1\alpha_2$。所以這兩項可以換成**一個**標準常態雜訊 $\epsilon$ 乘上 $\sqrt{1-\alpha_1\alpha_2}$：

The last two terms are two independent normal noises added together; the sum is still normal, with mean $0$ and the variances added: $\alpha_2\beta_1+\beta_2=\alpha_2(1-\alpha_1)+1-\alpha_2=1-\alpha_1\alpha_2$. So these two terms can be replaced by **one** standard-normal noise $\epsilon$ times $\sqrt{1-\alpha_1\alpha_2}$:

$$
\mathbf{x}_2=\sqrt{\alpha_1\alpha_2}\;\mathbf{x}_0+\sqrt{1-\alpha_1\alpha_2}\;\epsilon .
$$

代個數字 `[補充]`：$\beta_1=0.1$、$\beta_2=0.2$，所以 $\alpha_1=0.9$、$\alpha_2=0.8$、$\alpha_1\alpha_2=0.72$。$\mathbf{x}_0$ 的係數是 $\sqrt{0.72}\approx0.8485$；雜訊的變異數是 $0.8\times0.1+0.2=0.08+0.2=0.28$，正好等於 $1-0.72$。

Plug in numbers `[Added]`: $\beta_1=0.1$ and $\beta_2=0.2$, so $\alpha_1=0.9$, $\alpha_2=0.8$ and $\alpha_1\alpha_2=0.72$. The coefficient of $\mathbf{x}_0$ is $\sqrt{0.72}\approx0.8485$; the noise variance is $0.8\times0.1+0.2=0.08+0.2=0.28$, exactly $1-0.72$.

**推廣到 $t$ 步。** 同樣的步驟一直做下去，每多一步，$\mathbf{x}_0$ 的係數就多乘一個 $\sqrt{\alpha_t}$。雜訊的變異數則永遠是「$1$ 減去 $\mathbf{x}_0$ 係數的平方」：若第 $t-1$ 步的雜訊變異數是 $1-\bar\alpha_{t-1}$，再走一步，舊雜訊被乘上 $\sqrt{\alpha_t}$、再加上變異數 $\beta_t$ 的新雜訊，變成 $\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\alpha_t\bar\alpha_{t-1}=1-\bar\alpha_t$。把累積的乘積記成

**Generalise to $t$ steps.** Keep repeating the same move: each extra step multiplies the coefficient of $\mathbf{x}_0$ by another $\sqrt{\alpha_t}$. The noise variance is always "$1$ minus the square of $\mathbf{x}_0$'s coefficient": if the noise variance at step $t-1$ is $1-\bar\alpha_{t-1}$, one more step multiplies the old noise by $\sqrt{\alpha_t}$ and adds new noise of variance $\beta_t$, giving $\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\alpha_t\bar\alpha_{t-1}=1-\bar\alpha_t$. Write the running product as

$$
\bar\alpha_t=\prod_{i=1}^{t}\alpha_i=\alpha_1\alpha_2\cdots\alpha_t ,
$$

就得到來源的式 (30) 與式 (31)：

and you get the source's equations (30) and (31):

$$
\mathbf{x}_t=\sqrt{\bar\alpha_t}\;\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\;\epsilon,\qquad \epsilon\sim\mathcal{N}(\mathbf{0},I),
$$

$$
q(\mathbf{x}_t\mid\mathbf{x}_0)=\mathcal{N}\!\left(\mathbf{x}_t;\ \sqrt{\bar\alpha_t}\,\mathbf{x}_0,\ (1-\bar\alpha_t)I\right).
$$

這叫做**閉式**（closed form）：一條直接算得出答案的公式，不必一步一步模擬。把 $\mathbf{x}_t$ 寫成「固定的部分加上 $\epsilon$ 的倍數」，也就是第12週 KP06 的**重參數化**（reparameterization）寫法；來源用這個名字稱呼這條取樣公式。要分清楚：能一步跳到第 $t$ 步，是因為常態加常態還是常態；第12週強調的好處（讓梯度穿過抽樣）是另一回事，這裡的前向過程沒有要學的參數。

This is called a **closed form**: a formula that gives the answer directly, without simulating step by step. Writing $\mathbf{x}_t$ as "a fixed part plus a multiple of $\epsilon$" is the **reparameterization** of Week 12 KP06; the source uses this name for the sampling formula. Keep two things apart: jumping to step $t$ at once is possible because normal plus normal is still normal; the benefit stressed in Week 12 (letting gradients pass through sampling) is a separate point, since the forward process here has no parameters to learn.

用 KP01 的手算驗證一次 `[補充]`。KP01 用固定的 $\beta=0.36$（$\alpha=0.64$），從 $x_0=2$ 走三步得到 $x_3=0.736$。閉式說 $\bar\alpha_3=0.64^3\approx0.2621$，$\sqrt{\bar\alpha_3}=0.512$、$\sqrt{1-\bar\alpha_3}\approx0.859$。三次雜訊 $0.5$、$-1.0$、$0.0$ 合起來的貢獻是 $0.8^2\times0.6\times0.5+0.8\times0.6\times(-1.0)+0.6\times0=-0.288$，等於 $0.859$ 乘上一個標準常態值 $\bar\epsilon\approx-0.335$。於是

Check it once against KP01's hand calculation `[Added]`. KP01 used a constant $\beta=0.36$ ($\alpha=0.64$) and reached $x_3=0.736$ from $x_0=2$ in three steps. The closed form says $\bar\alpha_3=0.64^3\approx0.2621$, $\sqrt{\bar\alpha_3}=0.512$ and $\sqrt{1-\bar\alpha_3}\approx0.859$. The three noise draws $0.5$, $-1.0$ and $0.0$ together contribute $0.8^2\times0.6\times0.5+0.8\times0.6\times(-1.0)+0.6\times0=-0.288$, which is $0.859$ times a standard-normal value $\bar\epsilon\approx-0.335$. So

$$
x_3=0.512\times2+0.859\times(-0.335)\approx1.024-0.288=0.736,
$$

和一步一步算的結果一樣。

the same as the step-by-step result.

$\sqrt{\bar\alpha_t}$ 和 $\sqrt{1-\bar\alpha_t}$ 就是上面那張圖右邊的兩條線：前者是原資料的權重，後者是雜訊的權重，兩者的平方加起來永遠是 $1$。在 DDPM 排程下，$\bar\alpha_t=0.5$ 發生在 $t=259$，這時兩個權重都是 $\sqrt{0.5}\approx0.71$。

$\sqrt{\bar\alpha_t}$ and $\sqrt{1-\bar\alpha_t}$ are the two lines on the right of the figure above: the first is the weight of the original data and the second the weight of the noise, and their squares always add up to $1$. Under the DDPM schedule, $\bar\alpha_t=0.5$ happens at $t=259$, where both weights are $\sqrt{0.5}\approx0.71$.

常見錯誤：以為 $\bar\alpha_t$ 是 $\beta$ 的連乘。$\bar\alpha_t$ 是 $\alpha_i=1-\beta_i$ 的連乘；$\beta$ 都是很小的數，連乘起來會幾乎是 $0$，意思完全不對。

Common mistake: thinking $\bar\alpha_t$ is the product of the $\beta$'s. $\bar\alpha_t$ is the product of $\alpha_i=1-\beta_i$; the $\beta$'s are all tiny, and their product would be nearly $0$, which means something else entirely.

### 4. 閉式為什麼重要，以及它不能做什麼（Why the closed form matters, and what it cannot do）

訓練擴散模型時，每次要隨機挑一個步數 $t$，拿到該步的有雜訊資料 $\mathbf{x}_t$，再請網路去猜加進去的雜訊（本週 KP05）。如果每次都要從 $\mathbf{x}_0$ 一步步走到第 $t$ 步，平均要走好幾百步才拿到一筆訓練資料；有了閉式，只要抽**一次**雜訊、算一行公式。

When training a diffusion model, each time you pick a random step $t$, get the noisy data $\mathbf{x}_t$ of that step, and ask the network to guess the noise that was added (this week's KP05). If each time you had to walk step by step from $\mathbf{x}_0$ to step $t$, it would take hundreds of steps on average to get one training example; with the closed form you draw noise **once** and compute one line.

下面用 KP01 的一維兩堆資料（$0.5\,\mathcal{N}(-2,0.3^2)+0.5\,\mathcal{N}(2,0.3^2)$）比一比 `[補充]`：同樣 $50000$ 個 $x_0$，一組一步步走到第 $t$ 步，一組用閉式一步跳到第 $t$ 步。

Below, compare the two with KP01's one-dimensional two-pile data ($0.5\,\mathcal{N}(-2,0.3^2)+0.5\,\mathcal{N}(2,0.3^2)$) `[Added]`: with the same $50000$ values of $x_0$, one set walks to step $t$ step by step, and the other jumps to step $t$ in one go with the closed form.

![三張並排的直方圖，分別是 t 等於 100、300、600，橫軸 x_t 從 -4.5 到 4.5。每張都有淺藍色填色的「一步步走（抽 t 次雜訊）」和紫色外框線的「一步跳（抽 1 次雜訊）」，兩者幾乎完全重合。t=100 是兩座分開的山，標準差 1.94 對 1.94；t=300 是兩座連在一起、中間凹下的山，標準差 1.49 對 1.49；t=600 是一座山，標準差 1.03 對 1.04（Three histograms side by side, at t = 100, 300 and 600, with x_t from -4.5 to 4.5. Each shows a light-blue filled histogram for "step by step (t noise draws)" and a purple outline for "one jump (1 noise draw)", and the two almost coincide. At t = 100 there are two separate peaks, standard deviation 1.94 vs 1.94; at t = 300 two merged peaks with a dip between them, 1.49 vs 1.49; at t = 600 a single hill, 1.03 vs 1.04)](../assets/figures/lec14-kp02-step-vs-jump.png)

**圖說 `lec14-kp02-step-vs-jump`**〔功能：對照、精準度〕：每張圖比較同一個 $t$ 的兩種做法。淺藍色填色是走了 $t$ 步、抽了 $t$ 次雜訊的結果，紫色外框是一步跳、只抽 $1$ 次雜訊的結果。兩者幾乎重合，標題的標準差也一樣（理論值 $1.942$、$1.492$、$1.039$），和第3節推導出的「兩者分布完全相同」一致。

**Caption `lec14-kp02-step-vs-jump`** [Function: comparison, precision]: each chart compares the two methods at the same $t$. The light-blue fill is the result of $t$ steps with $t$ noise draws, and the purple outline the result of one jump with a single noise draw. They almost coincide, and the standard deviations in the titles agree (theoretical values $1.942$, $1.492$ and $1.039$), consistent with Section 3's derivation that the two distributions are identical.

閉式也有它不能做的事 `[補充]`。它給的是「從 $\mathbf{x}_0$ 出發，第 $t$ 步的分布」，每次用都會抽一個新的 $\epsilon$。如果用它分別跳到第 $300$ 步和第 $310$ 步，兩個結果用的是兩個無關的雜訊，並不在同一條路徑上：同一個 $x_0$ 兩次跳的結果平均相差（均方根）$1.109$；而從第 $300$ 步沿著前向過程再走 $10$ 步，只差 $0.249$。兩種做法在第 $310$ 步的整體分布一樣（標準差都約 $1.47$），但只有後者是一條真正的路徑。訓練時每次只需要一個 $t$，所以這不是問題；要畫一條完整的加雜訊軌跡時，就得一步一步走。

The closed form also has something it cannot do `[Added]`. It gives "the distribution of step $t$, starting from $\mathbf{x}_0$", and each use draws a new $\epsilon$. If you use it to jump separately to step $300$ and to step $310$, the two results use unrelated noise and do not lie on the same path: for the same $x_0$ the two jumps differ by $1.109$ (root mean square), while walking $10$ more forward steps from step $300$ changes the value by only $0.249$. Both methods give the same overall distribution at step $310$ (standard deviation about $1.47$ each), but only the latter is a real path. Training needs only one $t$ at a time, so this is no problem; to draw a full noising trajectory, you must walk step by step.

常見錯誤：以為閉式是「從 $\mathbf{x}_{t-1}$ 算 $\mathbf{x}_t$」的公式。式子裡出現的是 $\mathbf{x}_0$ 和 $\bar\alpha_t$，它是直接從原資料跳過去；從上一步走一步，用的是第1節的 $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ 和單步的 $\beta_t$。

Common mistake: thinking the closed form computes $\mathbf{x}_t$ from $\mathbf{x}_{t-1}$. It contains $\mathbf{x}_0$ and $\bar\alpha_t$: it jumps straight from the original data; a single step from the previous state uses $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ of Section 1 and the one-step $\beta_t$.

### 本單元重點整理（KP02 summary）

**一句話重點**：前向過程是一條固定的高斯馬可夫鏈，每步 $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})=\mathcal{N}(\mathbf{x}_t;\sqrt{\alpha_t}\,\mathbf{x}_{t-1},\beta_t I)$；因為常態加常態還是常態，可以一步跳到任意第 $t$ 步：$\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\epsilon$。

**One-sentence summary**: the forward process is a fixed Gaussian Markov chain with steps $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})=\mathcal{N}(\mathbf{x}_t;\sqrt{\alpha_t}\,\mathbf{x}_{t-1},\beta_t I)$; because normal plus normal is still normal, you can jump to any step $t$ at once: $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\epsilon$.

**白話說明**：把一杯果汁每次倒掉一點、補一點水，倒了 $t$ 次以後還剩多少果汁，不必真的倒 $t$ 次，把每次「留下的比例」乘起來就知道。$\bar\alpha_t$ 就是這個連乘；剩下的部分全是後來加進去的雜訊。

**Plain explanation**: pour away a little of a glass of juice each time and top it up with water; to know how much juice is left after $t$ pours, you need not pour $t$ times, just multiply the "share kept" each time. $\bar\alpha_t$ is that running product; everything else is noise added along the way.

**正式定義**：給定排程 $\beta_1,\ldots,\beta_T$，令 $\alpha_t=1-\beta_t$、$\bar\alpha_t=\prod_{i=1}^{t}\alpha_i$。前向過程 $q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)=\prod_{t=1}^{T}q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$，其中 $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})=\mathcal{N}(\mathbf{x}_t;\sqrt{\alpha_t}\,\mathbf{x}_{t-1},\beta_t I)$；由此 $q(\mathbf{x}_t\mid\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_t;\sqrt{\bar\alpha_t}\,\mathbf{x}_0,(1-\bar\alpha_t)I)$。

**Formal definition**: given a schedule $\beta_1,\ldots,\beta_T$, let $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{i=1}^{t}\alpha_i$. The forward process is $q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)=\prod_{t=1}^{T}q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ with $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})=\mathcal{N}(\mathbf{x}_t;\sqrt{\alpha_t}\,\mathbf{x}_{t-1},\beta_t I)$; hence $q(\mathbf{x}_t\mid\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_t;\sqrt{\bar\alpha_t}\,\mathbf{x}_0,(1-\bar\alpha_t)I)$.

**例子**：$\beta_1=0.1$、$\beta_2=0.2$ 時，$\bar\alpha_2=0.72$，$\mathbf{x}_0$ 的係數 $0.8485$，雜訊變異數 $0.08+0.2=0.28$。DDPM 排程（$\beta_t$ 從 $0.0001$ 到 $0.02$、$T=1000$）在 $t=259$ 時原資料與雜訊的權重相等，到 $t=1000$ 原資料的權重只剩 $0.0064$。

**Example**: with $\beta_1=0.1$ and $\beta_2=0.2$, $\bar\alpha_2=0.72$, the coefficient of $\mathbf{x}_0$ is $0.8485$ and the noise variance is $0.08+0.2=0.28$. Under the DDPM schedule ($\beta_t$ from $0.0001$ to $0.02$, $T=1000$), the weights of the original data and of the noise are equal at $t=259$, and by $t=1000$ the original's weight is only $0.0064$.

**常見迷思**：

- 「$\beta_t$ 是雜訊的標準差。」錯，它是變異數，乘在 $\epsilon$ 前面的是 $\sqrt{\beta_t}$。
- 「$\beta_t$ 一定要遞增。」不一定；要緊的是最後 $\bar\alpha_T$ 接近 $0$。
- 「$\bar\alpha_t$ 是 $\beta$ 的連乘。」錯，是 $\alpha_i=1-\beta_i$ 的連乘。
- 「閉式是從 $\mathbf{x}_{t-1}$ 算 $\mathbf{x}_t$。」錯，它直接從 $\mathbf{x}_0$ 跳到第 $t$ 步。
- 「用閉式分別跳到不同的 $t$，就是一條加雜訊的路徑。」錯，每次跳都用新的雜訊，彼此不連貫。

**Common misconceptions**:

- "$\beta_t$ is the noise's standard deviation." Wrong: it is a variance, and what multiplies $\epsilon$ is $\sqrt{\beta_t}$.
- "$\beta_t$ must increase." Not necessarily; what matters is that $\bar\alpha_T$ ends near $0$.
- "$\bar\alpha_t$ is the product of the $\beta$'s." Wrong: it is the product of $\alpha_i=1-\beta_i$.
- "The closed form computes $\mathbf{x}_t$ from $\mathbf{x}_{t-1}$." Wrong: it jumps from $\mathbf{x}_0$ straight to step $t$.
- "Jumping to different $t$ with the closed form gives a noising path." Wrong: each jump uses fresh noise, so the results are not connected.

**圖**：本單元有兩張圖。

- `lec14-kp02-schedule`：DDPM 排程下每一步的 $\beta_t$、$\alpha_t$，以及累積後原資料與雜訊的權重。
- `lec14-kp02-step-vs-jump`：一步步走與一步跳在 $t=100$、$300$、$600$ 的分布完全一樣。

**Figures** — this unit has two figures.

- `lec14-kp02-schedule`: the per-step $\beta_t$ and $\alpha_t$ of the DDPM schedule, and the accumulated weights of the original data and of the noise.
- `lec14-kp02-step-vs-jump`: walking step by step and jumping at once give the same distribution at $t=100$, $300$ and $600$.

### 小練習（Practice）

1. （來源習題改寫）設 $\alpha_t=0.9^t$，也就是 $\alpha_1=0.9$、$\alpha_2=0.81$，$x_0=1.0$。用閉式算出 $x_1$ 與 $x_2$ 的平均與標準差。
2. 用 KP01 固定的 $\beta=0.36$（每步乘 $0.8$），至少要走幾步，原資料的權重 $\sqrt{\bar\alpha_t}$ 才會小於 $0.01$？
3. 在 DDPM 排程下，$\sqrt{\bar\alpha_{500}}=0.2803$、$\sqrt{1-\bar\alpha_{500}}=0.9599$。若 $x_0=2$，抽到的雜訊是 $\epsilon=0.5$，$x_{500}$ 是多少？
4. 一位同學想畫出一筆資料的加雜訊過程，於是用閉式分別算 $x_{300}$、$x_{301}$、…、$x_{310}$，每次都重新抽 $\epsilon$。畫出來的線會長什麼樣？哪裡不對？

1. (Adapted from the source's exercise) Let $\alpha_t=0.9^t$, that is, $\alpha_1=0.9$ and $\alpha_2=0.81$, with $x_0=1.0$. Use the closed form to compute the mean and standard deviation of $x_1$ and $x_2$.
2. With KP01's constant $\beta=0.36$ (multiply by $0.8$ each step), how many steps at least until the original's weight $\sqrt{\bar\alpha_t}$ falls below $0.01$?
3. Under the DDPM schedule, $\sqrt{\bar\alpha_{500}}=0.2803$ and $\sqrt{1-\bar\alpha_{500}}=0.9599$. If $x_0=2$ and the noise drawn is $\epsilon=0.5$, what is $x_{500}$?
4. A classmate wants to draw the noising process of one example, so they compute $x_{300}$, $x_{301}$, …, $x_{310}$ separately with the closed form, drawing a new $\epsilon$ each time. What will the line look like? What is wrong?

**第 1 題解答** `[補充]`：$\bar\alpha_1=0.9$，$\bar\alpha_2=0.9\times0.81=0.729$。$x_1$ 的平均 $\sqrt{0.9}\times1.0\approx0.9487$、變異數 $0.1$、標準差 $0.3162$；$x_2$ 的平均 $\sqrt{0.729}\approx0.8538$、變異數 $0.271$、標準差 $0.5206$。要得到具體的 $x_1$、$x_2$，再各自抽一個 $\epsilon$ 代進去即可。

**Solution to question 1** `[Added]`: $\bar\alpha_1=0.9$ and $\bar\alpha_2=0.9\times0.81=0.729$. $x_1$ has mean $\sqrt{0.9}\times1.0\approx0.9487$, variance $0.1$ and standard deviation $0.3162$; $x_2$ has mean $\sqrt{0.729}\approx0.8538$, variance $0.271$ and standard deviation $0.5206$. For concrete values of $x_1$ and $x_2$, draw an $\epsilon$ for each and substitute it.

**第 2 題解答** `[補充]`：$\sqrt{\bar\alpha_t}=0.8^t$。$0.8^{20}\approx0.0115$ 還不夠小，$0.8^{21}\approx0.0092<0.01$，所以至少要 $21$ 步。

**Solution to question 2** `[Added]`: $\sqrt{\bar\alpha_t}=0.8^t$. $0.8^{20}\approx0.0115$ is not yet small enough, while $0.8^{21}\approx0.0092<0.01$, so at least $21$ steps are needed.

**第 3 題解答** `[補充]`：$x_{500}=0.2803\times2+0.9599\times0.5=0.5606+0.4800\approx1.04$。原資料的貢獻只剩 $0.56$，雜訊已經佔了大半。

**Solution to question 3** `[Added]`: $x_{500}=0.2803\times2+0.9599\times0.5=0.5606+0.4800\approx1.04$. The original contributes only $0.56$; noise already makes up most of it.

**第 4 題解答** `[補充]`：每次都抽新的 $\epsilon$，相鄰兩點用的是無關的雜訊，所以線會上下亂跳：每一點本身離平均約 $\sqrt{1-\bar\alpha_t}\approx0.78$，相鄰兩點的差距大約是 $\sqrt{2}\times0.78\approx1.1$，不像真正的前向過程每步只動一點點（第4節：兩次獨立跳差 $1.109$，沿路徑走 $10$ 步只差 $0.249$）。每個 $x_t$ 單獨看分布是對的，但它們連不成一條路徑；要畫路徑，得從 $x_{300}$ 起用單步的 $q(x_t\mid x_{t-1})$ 一步一步走。

**Solution to question 4** `[Added]`: with a new $\epsilon$ each time, neighbouring points use unrelated noise, so the line jumps up and down: each point lies about $\sqrt{1-\bar\alpha_t}\approx0.78$ from its mean, and neighbouring points differ by about $\sqrt{2}\times0.78\approx1.1$, unlike the real forward process, which moves only a little per step (Section 4: two independent jumps differ by $1.109$, while $10$ steps along a path differ by only $0.249$). Each $x_t$ has the right distribution on its own, but they do not form a path; to draw a path, start from $x_{300}$ and walk step by step with the one-step $q(x_t\mid x_{t-1})$.

### 小測驗（Quiz，來源原題）

來源 KP-2 與 KP-5 各有四題自我檢測，保留原題與全部選項；來源頁面附有答案與簡短解析，下面的解析由本講義依各節內容補寫。

The source's KP-2 and KP-5 each have four self-check questions, kept here with all their options; the source page gives the answers with short explanations, and the explanations below are written by this note from its own sections.

1. Forward process 的條件分布 $q(x_t\mid x_{t-1})$ 是什麼分布？
   - (A) 均勻分布
   - (B) 高斯分布
   - (C) 拉普拉斯分布
   - (D) 伯努利分布
2. $\beta_t$ 隨時間怎麼變化？
   - (A) 遞減
   - (B) 遞增
   - (C) 固定不變
   - (D) 隨機取樣
3. Forward process 為什麼是 Markov chain？
   - (A) $x_t$ 只依賴 $x_{t-1}$
   - (B) $x_t$ 依賴所有過去狀態
   - (C) $x_t$ 不依賴任何狀態
   - (D) $x_t$ 只依賴 $x_0$
4. Joint distribution $q(x_{1:T}\mid x_0)$ 可以怎麼寫？
   - (A) 單一高斯
   - (B) 轉移機率的連乘
   - (C) 加總
   - (D) 指數函數
5. $\bar\alpha_t$ 的定義是什麼？
   - (A) $\prod_{i=1}^{t}\beta_i$
   - (B) $\prod_{i=1}^{t}\alpha_i$
   - (C) $\sum_{i=1}^{t}\alpha_i$
   - (D) $1-\beta_t$
6. 任意 $t$ 的 $x_t$ 如何從 $x_0$ 直接取樣？
   - (A) $x_t=x_0+\beta_t\epsilon$
   - (B) $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$
   - (C) $x_t=\sqrt{\beta_t}x_0+\epsilon$
   - (D) $x_t=x_0$
7. $q(x_t\mid x_0)$ 是什麼分布？
   - (A) 均勻分布
   - (B) 高斯分布
   - (C) 指數分布
   - (D) 泊松分布
8. 重參數化的主要好處是什麼？
   - (A) 減少訓練參數
   - (B) 可以直接跳到任意時間步 $t$，不用逐步加噪
   - (C) 讓 covariance 變大
   - (D) 讓 loss 變成零

1. What distribution is the forward process's conditional $q(x_t\mid x_{t-1})$?
   - (A) Uniform
   - (B) Gaussian
   - (C) Laplace
   - (D) Bernoulli
2. How does $\beta_t$ change over time?
   - (A) It decreases
   - (B) It increases
   - (C) It stays fixed
   - (D) It is sampled at random
3. Why is the forward process a Markov chain?
   - (A) $x_t$ depends only on $x_{t-1}$
   - (B) $x_t$ depends on all past states
   - (C) $x_t$ depends on no state
   - (D) $x_t$ depends only on $x_0$
4. How can the joint distribution $q(x_{1:T}\mid x_0)$ be written?
   - (A) A single Gaussian
   - (B) A product of transition probabilities
   - (C) A sum
   - (D) An exponential function
5. What is the definition of $\bar\alpha_t$?
   - (A) $\prod_{i=1}^{t}\beta_i$
   - (B) $\prod_{i=1}^{t}\alpha_i$
   - (C) $\sum_{i=1}^{t}\alpha_i$
   - (D) $1-\beta_t$
6. How is $x_t$ for any $t$ sampled directly from $x_0$?
   - (A) $x_t=x_0+\beta_t\epsilon$
   - (B) $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$
   - (C) $x_t=\sqrt{\beta_t}x_0+\epsilon$
   - (D) $x_t=x_0$
7. What distribution is $q(x_t\mid x_0)$?
   - (A) Uniform
   - (B) Gaussian
   - (C) Exponential
   - (D) Poisson
8. What is the main benefit of reparameterization?
   - (A) Fewer training parameters
   - (B) You can jump to any time step $t$ directly, without adding noise step by step
   - (C) It makes the covariance larger
   - (D) It makes the loss zero

**解析** `[補充]`：

- 第 1 題選 (B)：第1節的 $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ 是平均 $\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}$、共變異數 $\beta_t I$ 的常態（高斯）分布。
- 第 2 題選 (B)：來源與 DDPM 排程都讓 $\beta_t$ 遞增；第2節也說明遞增是常見選擇而非必要條件。
- 第 3 題選 (A)：馬可夫性質就是只看上一步。(D) 描述的是閉式 $q(\mathbf{x}_t\mid\mathbf{x}_0)$，那是推導出來的結果，不是鏈本身的定義。
- 第 4 題選 (B)：第1節的連乘 $\prod_{t=1}^{T}q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$。
- 第 5 題選 (B)：第3節，$\bar\alpha_t$ 是 $\alpha_i$ 的連乘；(A) 正是常見迷思。
- 第 6 題選 (B)：第3節的閉式。(A) 把 $\beta_t$ 當標準差，也沒有縮小 $x_0$；(C) 把係數放錯位置。
- 第 7 題選 (B)：常態加常態還是常態，平均 $\sqrt{\bar\alpha_t}\,x_0$、變異數 $1-\bar\alpha_t$。
- 第 8 題選 (B)：第4節，訓練時抽一次雜訊就能得到任意第 $t$ 步的資料。參數數量不變，(A) 不對；「重參數化」一詞的兩種好處見第3節的說明。

**Explanations** `[Added]`:

- Question 1: B: Section 1's $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ is a normal (Gaussian) distribution with mean $\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}$ and covariance $\beta_t I$.
- Question 2: B: both the source and the DDPM schedule make $\beta_t$ increase; Section 2 also explains that increasing is a common choice, not a requirement.
- Question 3: A: the Markov property means looking only at the previous step. D describes the closed form $q(\mathbf{x}_t\mid\mathbf{x}_0)$, a derived result, not the definition of the chain.
- Question 4: B: the product $\prod_{t=1}^{T}q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ of Section 1.
- Question 5: B: Section 3; $\bar\alpha_t$ is the product of the $\alpha_i$; A is exactly the common misconception.
- Question 6: B: the closed form of Section 3. A treats $\beta_t$ as a standard deviation and does not shrink $x_0$; C puts the coefficients in the wrong places.
- Question 7: B: normal plus normal is still normal, with mean $\sqrt{\bar\alpha_t}\,x_0$ and variance $1-\bar\alpha_t$.
- Question 8: B: Section 4; during training, one noise draw gives the data of any step $t$. The number of parameters does not change, so A is wrong; for the two different benefits behind the word "reparameterization", see the note in Section 3.

### 留給下一單元的問題（The question for the next unit）

前向過程現在完全寫得出來，連任意一步都有公式。可是生成要走的是反方向：從 $\mathbf{x}_T$ 回到 $\mathbf{x}_0$。逆向的一步 $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ 該長什麼樣子？要用什麼目標去訓練它，才會讓整條鏈生成的資料像真的？下一個知識點（本週 KP03）把逆向過程寫成常態分布，並像第12週一樣推出 ELBO。

The forward process can now be written out completely, with a formula even for any single step. But generation goes the other way: from $\mathbf{x}_T$ back to $\mathbf{x}_0$. What should one backward step $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ look like? What objective should train it so that the whole chain generates realistic data? The next knowledge point (this week's KP03) writes the backward process as normal distributions and derives the ELBO, as in Week 12.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第15章「Diffusion Models」的 KP-2「Forward Process：馬可夫鏈、高斯條件、$\beta_t$ 排程」與 KP-5「Reparameterization：任意 $t$ 的閉式 $q(x_t\mid x_0)$」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch15-diffusion-models/index.html ，讀取日期 2026-11-15）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from KP-2, "Forward process: Markov chain, Gaussian conditionals, $\beta_t$ schedule", and KP-5, "Reparameterization: the closed form $q(x_t\mid x_0)$ for any $t$", of Chapter 15, "Diffusion Models", of CCC-LNS IDL_115 (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch15-diffusion-models/index.html , accessed 2026-11-15). Content not in the source is marked `[Added]` where it appears.

- 來源說「$\bar\alpha_t$ 代表經過 $t$ 步後還保留多少原始資料的比例」。說得精確一點：乘在 $\mathbf{x}_0$ 前面的係數是 $\sqrt{\bar\alpha_t}$，$\bar\alpha_t$ 本身是原資料佔「變異數」的比例（資料變異數為 $1$ 時）。本講義第3節把 $\sqrt{\bar\alpha_t}$ 稱為原資料的權重，並在表中分開列出。
- 來源的式 (3) 寫 $\beta_1<\beta_2<\cdots<\beta_T$，並在小測驗以「遞增」為答案。遞增是 DDPM 的設定與常見做法，但不是前向過程成立的必要條件；本講義第2節註明，真正要緊的是 $\bar\alpha_T$ 接近 $0$。
- 來源說閉式「不需要真的跑 $t$ 次 transition」，但沒有提醒它給的是每一步各自的分布、不是一條路徑；本講義第4節補上這個限制與數字。
- 來源互動展示以「$\bar\alpha_T$ 小於 $0.05$」為正確基準；這個門檻太寬鬆：它的預設（$\beta_t$ 從 $0.01$ 到 $0.2$、$T=100$）實際給出 $\bar\alpha_T\approx1.3\times10^{-5}$，DDPM 排程是 $4.0\times10^{-5}$，常用排程都比它小三、四個數量級，通過它說明不了什麼。實務上要求 $\bar\alpha_T$ 非常接近 $0$，起點才真的接近標準常態。
- 來源習題「給定 $\alpha_t=0.9^t$ 與 $x_0=1.0$，逐步計算 $x_1$ 與 $x_2$」沒有給雜訊值；本講義第 1 題改成計算平均與標準差。
- 本講義補上：$\beta_t$ 是變異數而非標準差的提醒、$I$ 的意思、$\beta_t$ 為何先小後大、DDPM 排程的數值表與排程圖、兩步合成一步的推導與數字、用 KP01 手算驗證閉式、一步步走與一步跳的模擬比較、閉式不構成路徑的說明、四題小練習與解答、小測驗的解析。
- 取捨：來源的符號沒有加粗，本講義的一般向量用 $\mathbf{x}$、一維例子用 $x$；小測驗照錄來源的寫法。來源兩個互動展示不重做，改用排程圖與表格呈現。

- The source says "$\bar\alpha_t$ is the share of the original data kept after $t$ steps". To be precise: the coefficient multiplying $\mathbf{x}_0$ is $\sqrt{\bar\alpha_t}$, while $\bar\alpha_t$ itself is the original data's share of the *variance* (when the data have variance $1$). Section 3 calls $\sqrt{\bar\alpha_t}$ the weight of the original data and lists both in the table.
- The source's equation (3) writes $\beta_1<\beta_2<\cdots<\beta_T$, and its quiz takes "increasing" as the answer. Increasing is the DDPM setting and common practice, but it is not required for the forward process to work; Section 2 notes that what really matters is that $\bar\alpha_T$ ends near $0$.
- The source says the closed form "does not need to actually run $t$ transitions", but does not warn that it gives each step's distribution separately, not a path; Section 4 adds this limit with numbers.
- The source's interactive demo takes "$\bar\alpha_T$ below $0.05$" as the correct benchmark; this threshold is too loose: its default ($\beta_t$ from $0.01$ to $0.2$, $T=100$) actually gives $\bar\alpha_T\approx1.3\times10^{-5}$, and the DDPM schedule gives $4.0\times10^{-5}$; common schedules end three to four orders of magnitude below it, so passing it says little. In practice $\bar\alpha_T$ must be very close to $0$ for the starting point to be truly close to a standard normal.
- The source's exercise "given $\alpha_t=0.9^t$ and $x_0=1.0$, compute $x_1$ and $x_2$ step by step" gives no noise values; practice question 1 asks for the means and standard deviations instead.
- This note adds: the reminder that $\beta_t$ is a variance, not a standard deviation; the meaning of $I$; why $\beta_t$ starts small and grows; the DDPM schedule table and figure; the two-step merge with numbers; the check of the closed form against KP01's hand calculation; the simulation comparing step-by-step and one-jump; the explanation that the closed form does not give a path; four practice questions with solutions; and the explanations of the quiz.
- Choices: the source writes symbols without bold; this note uses $\mathbf{x}$ for general vectors and $x$ for one-dimensional examples, while the quiz keeps the source's notation. The source's two interactive demos are not rebuilt; the schedule figure and table are used instead.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 前向過程 | forward process |
| 馬可夫鏈 | Markov chain |
| 馬可夫性質 | Markov property |
| 條件分布 | conditional distribution |
| 聯合分布 | joint distribution |
| 高斯分布（常態分布） | Gaussian (normal) distribution |
| 共變異數 | covariance |
| 單位矩陣 | identity matrix |
| 雜訊排程 | noise schedule |
| 線性排程 | linear schedule |
| 餘弦排程（KP06） | cosine schedule |
| 閉式 | closed form |
| 重參數化（第12週 KP06） | reparameterization |
| 均方根 | root mean square (RMS) |
| 去雜訊擴散機率模型 | denoising diffusion probabilistic model (DDPM) |
| 證據下界（KP03） | evidence lower bound (ELBO) |
| 均勻分布 | uniform distribution |
| 拉普拉斯分布 | Laplace distribution |
| 伯努利分布 | Bernoulli distribution |
| 指數分布 | exponential distribution |
| 泊松分布 | Poisson distribution |
