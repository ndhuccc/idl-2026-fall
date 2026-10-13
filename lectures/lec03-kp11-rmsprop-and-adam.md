# 第3週 KP11：RMSProp 與 Adam（RMSProp and Adam）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第3章「反向傳播與最佳化」的第十一個知識點（KP11），承接 KP10 留下的問題：AdaGrad 的累積量只增不減，怎麼讓它學會遺忘？先備知識是期望、等比級數、KP07–KP10；第一次出現的符號都會定義，所有數字都可以自己重算。內容依 CCC-LNS 第3章 KP11 的主文與投影片消化後重新撰寫，讀取日期 2026-10-14；來源沒有的內容標示 `[補充]`。

This is the eleventh knowledge point (KP11) of Chapter 3. It picks up the question KP10 left open: the accumulator of AdaGrad only grows, so how can it learn to forget? It assumes expectation, geometric series and KP07–KP10; every symbol is defined when it first appears and every number can be recomputed. The text is rewritten from the KP11 main text and slides of CCC-LNS Chapter 3 (accessed on 2026-10-14); content that the source does not contain is marked `[Added]`.

[下載本單元投影片](../slides/lec03-kp11-rmsprop-and-adam.jbd.json)

## KP11 RMSProp 與 Adam（RMSProp and Adam）

### 學習目標（Learning objectives）

讀完之後，你能說出 AdaGrad 的步長為什麼只降不升，並寫出 RMSProp 用指數移動平均取代累加的式子；能算出舊梯度的權重 $\gamma^k$、記憶窗與半衰期；能在同一個玩具碗上比較 AdaGrad 與 RMSProp 的有效步長；能寫出 Adam 的 $m$、$v$ 與修正後的更新式，並手算前兩步；也能說明偏差修正為什麼不是可有可無，以及為什麼比較最佳化器是一場受控實驗。

After reading, you can say why the step of AdaGrad only falls and write the formula in which RMSProp replaces the sum by an exponential moving average; compute the weight $\gamma^k$ of an old gradient, the memory window and the half-life; compare the effective steps of AdaGrad and RMSProp on the same toy bowl; write the $m$, $v$ and the corrected update of Adam and compute its first two steps by hand; and explain why bias correction is not optional and why comparing optimizers is a controlled experiment.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $g_t$ | 第 $t$ 步的梯度（某一個坐標） | gradient at step $t$ (one coordinate) |
| $G_t$ | AdaGrad 的累積量，$G_t=\sum_{s\le t}g_s^2$ | the AdaGrad accumulator |
| $v_t$ | 梯度平方的指數移動平均 | exponential moving average of $g^2$ |
| $m_t$ | 梯度本身的指數移動平均（方向） | exponential moving average of $g$ (direction) |
| $\gamma,\ \beta_1,\ \beta_2$ | 遺忘因子（衰減率），介於 0 與 1 | decay rates between 0 and 1 |
| $\hat m_t,\ \hat v_t$ | 偏差修正後的 $m_t,\ v_t$ | bias-corrected $m_t,\ v_t$ |
| $\eta$ | 基本步長 | base step size |
| $\varepsilon$ | 避免除以零的小常數（$10^{-8}$） | small constant against division by zero |

### 1. 讓累積學會遺忘（Teaching the accumulator to forget）

KP10 的 AdaGrad 為每個坐標各記一個累積量 $G_t=G_{t-1}+g_t^2$，更新式是 $w_t=w_{t-1}-\eta\,g_t/\sqrt{\varepsilon+G_t}$，有效步長是 $\eta/\sqrt{\varepsilon+G_t}$。平方不會是負的，所以 $G_t$ 只增不減，有效步長只降不升：前期遇到的大梯度，會永遠留在分母裡。走到平緩的地方，步長也回不來。

The AdaGrad of KP10 keeps one accumulator per coordinate, $G_t=G_{t-1}+g_t^2$, and updates $w_t=w_{t-1}-\eta\,g_t/\sqrt{\varepsilon+G_t}$; the effective step is $\eta/\sqrt{\varepsilon+G_t}$. A square is never negative, so $G_t$ only grows and the effective step only falls: the large gradients met early stay in the denominator forever. Even on flat ground the step cannot recover.

**RMSProp** 只改一個地方：把累加換成**指數移動平均**（exponential moving average，EMA）：

**RMSProp** changes one thing: the running sum becomes an **exponential moving average** (EMA):

$$
v_t=\gamma\,v_{t-1}+(1-\gamma)\,g_t^2,\qquad w_t=w_{t-1}-\frac{\eta}{\sqrt{\varepsilon+v_t}}\,g_t
$$

$v_0=0$，$\gamma$ 常取 $0.9$。把遞推展開，$v_t=(1-\gamma)\sum_{s=1}^{t}\gamma^{\,t-s}g_s^2$：離現在 $k$ 步以前的梯度平方，權重是 $\gamma^k$，離得越遠權重越小。以 $\gamma=0.9$ 為例，$k=1,5,10,22$ 的權重是 $0.9,\ 0.5905,\ 0.3487,\ 0.0985$〔程式計算〕。舊資訊不是被丟掉，而是逐漸褪色。

Here $v_0=0$ and $\gamma$ is usually $0.9$. Unrolling the recursion gives $v_t=(1-\gamma)\sum_{s=1}^{t}\gamma^{\,t-s}g_s^2$: a squared gradient from $k$ steps ago has weight $\gamma^k$, smaller the further back it lies. For $\gamma=0.9$ the weights at $k=1,5,10,22$ are $0.9,\ 0.5905,\ 0.3487,\ 0.0985$ [computed by program]. Old information is not thrown away; it fades.

記憶有多長？兩個常用的量：**記憶窗**約 $1/(1-\gamma)$ 步（這是經驗的估計），**半衰期**是權重降到一半所需的步數 $\ln0.5/\ln\gamma$。$\gamma=0.9$：窗約 10 步，半衰期 $6.58$ 步；$\gamma=0.99$：約 100 步與 $68.97$ 步；$\gamma=0.999$：約 1000 步與 $692.80$ 步〔程式計算〕。

How long is the memory? Two common measures: the **memory window** is about $1/(1-\gamma)$ steps (an empirical estimate) and the **half-life** is the number of steps for the weight to halve, $\ln0.5/\ln\gamma$. For $\gamma=0.9$: a window of about 10 steps and a half-life of $6.58$; for $\gamma=0.99$: about 100 and $68.97$; for $\gamma=0.999$: about 1000 and $692.80$ [computed by program].

> 出處要如實：RMSProp 出自 Hinton 在 2012 年開設的網路課程講義（Coursera 課程 *Neural Networks for Machine Learning*），不是正式發表的論文。
>
> Be honest about the origin: RMSProp comes from the lecture slides of Hinton's 2012 online course (the Coursera course *Neural Networks for Machine Learning*); it was never published as a paper.

### 2. 同一個碗上，步長會回升（On the same bowl the step recovers）

用 KP10 的同一個碗 $f(x,y)=\tfrac12(x^2+25y^2)$，起點 $(10,2)$，梯度是 $(x,\,25y)$。兩個方法的設定相同：$\eta=0.5$，$\varepsilon=10^{-8}$；RMSProp 另取 $\gamma=0.9$。AdaGrad 照 KP10 的定義（$G_t=\sum g_s^2$，更新式分母 $\sqrt{\varepsilon+G_t}$），本講義用自己的程式重算，$\eta$ 在這裡明確取 $0.5$。KP10 取 $\eta=1$，這裡改取 $0.5$：RMSProp 的第 1 步是 $\eta/\sqrt{0.1}\approx3.16\,\eta$，$\eta=1$ 時 $x$ 一步就走 $3.162$，第 20 步損失還有 $6.92$，一開始就彈跳得太厲害；AdaGrad 在 $\eta=1$ 的第 1 步仍是 $(9,1)$，與 KP10 一致〔程式計算〕。下表看 $x$ 坐標的有效步長與損失〔程式計算〕：

Use the bowl of KP10, $f(x,y)=\tfrac12(x^2+25y^2)$, start $(10,2)$, gradient $(x,\,25y)$. Both methods use $\eta=0.5$ and $\varepsilon=10^{-8}$; RMSProp also takes $\gamma=0.9$. AdaGrad follows the definition of KP10 ($G_t=\sum g_s^2$, denominator $\sqrt{\varepsilon+G_t}$) and is recomputed here by our own program with $\eta=0.5$ stated explicitly. KP10 uses $\eta=1$ and this note uses $0.5$: the first step of RMSProp is $\eta/\sqrt{0.1}\approx3.16\,\eta$, so with $\eta=1$ the $x$ coordinate moves $3.162$ in one step and the loss is still $6.92$ at step 20, bouncing too hard from the start; AdaGrad with $\eta=1$ still lands at $(9,1)$ in step 1, as in KP10 [computed by program]. The table shows the effective step of the $x$ coordinate and the loss [computed by program]:

| 步數 $k$ | AdaGrad 有效步長 | RMSProp 有效步長 | AdaGrad 損失 | RMSProp 損失 |
|---|---|---|---|---|
| 1 | 0.0500 | 0.1581 | 73.25 | 37.63 |
| 5 | 0.0242 | 0.1034 | 41.39 | 13.75 |
| 10 | 0.0182 | 0.1088 | 30.10 | 4.54 |
| 20 | 0.0141 | 0.1640 | 21.25 | 0.249 |
| 40 | 0.0114 | 0.4655 | 12.86 | $2.3\times10^{-7}$ |

| Step $k$ | AdaGrad effective step | RMSProp effective step | AdaGrad loss | RMSProp loss |
|---|---|---|---|---|
| 1 | 0.0500 | 0.1581 | 73.25 | 37.63 |
| 5 | 0.0242 | 0.1034 | 41.39 | 13.75 |
| 10 | 0.0182 | 0.1088 | 30.10 | 4.54 |
| 20 | 0.0141 | 0.1640 | 21.25 | 0.249 |
| 40 | 0.0114 | 0.4655 | 12.86 | $2.3\times10^{-7}$ |

AdaGrad 的有效步長從 $0.05$ 一路降到 $0.0114$，從不回頭。RMSProp 先降到第 6 步的最低 $0.1025$，隨後梯度變小、$v$ 回落，步長升到第 40 步的 $0.4655$，是 AdaGrad 的 $40.9$ 倍。一個簡單的解釋：當梯度從 10 突然降到 1（梯度平方從 100 降到 1），AdaGrad 的累積量在 20 步大梯度之後是 $2000$，再加 20 步只變成 $2020$；RMSProp 的 $v$ 在 20 步後是 $87.84$，之後 10 步降到 $31.28$、20 步降到 $11.56$，分母在變小〔程式計算〕。

The effective step of AdaGrad falls from $0.05$ to $0.0114$ and never turns back. RMSProp first dips to its minimum $0.1025$ at step 6, then the gradient shrinks, $v$ falls and the step climbs to $0.4655$ at step 40, $40.9$ times that of AdaGrad. A simple illustration: when the gradient drops from 10 to 1 (the squared gradient from 100 to 1), the AdaGrad accumulator is $2000$ after 20 large gradients and only becomes $2020$ after 20 more; the $v$ of RMSProp is $87.84$ after 20 steps and falls to $31.28$ after 10 more steps and to $11.56$ after 20, so the denominator shrinks [computed by program].

![同一個碗上 x 坐標的有效步長（對數座標）：AdaGrad 從 0.05 單調降到 0.0114；RMSProp 先降到 0.1025（第 6 步），再回升到 0.4655（第 40 步）。（Effective step of the x coordinate on the same bowl (log scale): AdaGrad falls monotonically from 0.05 to 0.0114; RMSProp dips to 0.1025 at step 6 and recovers to 0.4655 at step 40.）](../assets/figures/lec03-kp11-effective-step.png)

**圖說 `lec03-kp11-effective-step`**：先看藍線：AdaGrad 的步長只往下掉；再看橘線：RMSProp 在第 6 步到達最低點，之後一路回升，第 40 步高出 AdaGrad 約 41 倍。縱軸是對數刻度，所以兩條線之間的距離代表倍數。〔功能：1、2〕

**Caption `lec03-kp11-effective-step`**: Look at the blue line first: the AdaGrad step only falls. Then the orange line: RMSProp reaches its minimum at step 6 and climbs from there, about 41 times above AdaGrad at step 40. The vertical axis is logarithmic, so the gap between the lines is a ratio. [Function: 1, 2]

回升也有代價。步長變大以後，固定的 $\eta$ 會讓 RMSProp 在最小點附近來回彈跳：同一個設定跑下去，第 80 步損失又升到 $1.35$，第 100、150、200 步是 $0.608,\ 0.836,\ 0.812$，沒有收斂到零；AdaGrad 較慢但穩，第 200 步是 $0.614$〔程式計算〕。步長要不要再搭配遞減排程，是 KP10 之後常見的實務問題。

The recovery has a price. Once the step is large again, a fixed $\eta$ lets RMSProp bounce around the minimum: running the same setup, the loss climbs back to $1.35$ at step 80 and is $0.608,\ 0.836,\ 0.812$ at steps 100, 150, 200, not converging to zero; AdaGrad is slower but steady, $0.614$ at step 200 [computed by program]. Whether to pair the step with a decay schedule is a common practical question after KP10.

### 3. Adam：方向與尺度（Adam: direction and scale）

RMSProp 只管各個坐標的**尺度**：誰的梯度大，誰的步子就小。它沒有**方向**的記憶：步子的方向還是只看當前這一筆梯度。KP05 的動量補的正是方向：把梯度做指數平均，來回震盪的分量會互相抵消，留下穩定的方向。**Adam**（adaptive moment estimation）把兩者放在一起，同時維護：

RMSProp handles only the **scale** of each coordinate: where the gradient is large, the step is small. It has no memory of **direction**: the direction of the step still depends only on the current gradient. The momentum of KP05 supplies the direction: an exponential average of the gradient cancels oscillating components and leaves a steady direction. **Adam** (adaptive moment estimation) puts the two together and keeps both:

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)\,g_t,\qquad v_t=\beta_2 v_{t-1}+(1-\beta_2)\,g_t^2
$$

$m_t$ 是梯度的指數平均（**方向**），$v_t$ 是梯度平方的指數平均（**尺度**，就是 RMSProp 的 $v$）。常見預設值是 $\beta_1=0.9$（約記 10 步）、$\beta_2=0.999$（約記 1000 步）、$\varepsilon=10^{-8}$。更新式把方向除以尺度，每個坐標各自縮放：

$m_t$ is the exponential average of the gradient (the **direction**) and $v_t$ the exponential average of its square (the **scale**, the $v$ of RMSProp). Common defaults are $\beta_1=0.9$ (about 10 steps of memory), $\beta_2=0.999$ (about 1000 steps) and $\varepsilon=10^{-8}$. The update divides the direction by the scale, each coordinate on its own:

$$
w_t=w_{t-1}-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}
$$

其中 $\hat m_t,\hat v_t$ 是修正後的 $m_t,v_t$，下一節定義。來源把分母寫成 $\sqrt{\varepsilon+\hat v_t}$，兩種寫法在 $\varepsilon$ 很小時幾乎沒有差別，各個函式庫的實作也不完全一樣。預設值 $0.9$、$0.999$、$10^{-8}$ 與 $\eta=10^{-3}$ 只是常見的起點，不是定理。

Here $\hat m_t,\hat v_t$ are the corrected $m_t,v_t$, defined in the next section. The source writes the denominator as $\sqrt{\varepsilon+\hat v_t}$; the two forms barely differ when $\varepsilon$ is tiny, and library implementations vary. The defaults $0.9$, $0.999$, $10^{-8}$ and $\eta=10^{-3}$ are common starting points, not theorems.

### 4. 偏差修正不是可有可無（Bias correction is not optional）

$m_0=0$、$v_0=0$ 是從零開始的，前幾步的 $m_t,v_t$ 一定偏低：第 1 步 $m_1=(1-\beta_1)g_1=0.1\,g_1$，只剩真梯度的十分之一；$v_1=0.001\,g_1^2$，只剩千分之一。原因可以用等比級數看清楚：若梯度的期望是定值 $\mu$（局部穩定的假設），則

Starting from $m_0=0$ and $v_0=0$, the first values of $m_t,v_t$ are necessarily too low: $m_1=(1-\beta_1)g_1=0.1\,g_1$, only a tenth of the true gradient, and $v_1=0.001\,g_1^2$, a thousandth. A geometric series shows why: if the expected gradient is a constant $\mu$ (a locally steady assumption), then

$$
\mathbb E[m_t]=(1-\beta_1)\sum_{s=1}^{t}\beta_1^{\,t-s}\mu=(1-\beta_1)\,\frac{1-\beta_1^{\,t}}{1-\beta_1}\,\mu=\big(1-\beta_1^{\,t}\big)\mu
$$

也就是 EMA 的權重加起來只有 $1-\beta_1^{\,t}$，不是 1。把它除掉就無偏了：

so the weights of the EMA sum to $1-\beta_1^{\,t}$, not to 1. Dividing it out removes the bias:

$$
\hat m_t=\frac{m_t}{1-\beta_1^{\,t}},\qquad \hat v_t=\frac{v_t}{1-\beta_2^{\,t}}
$$

$t$ 很小時修正倍率很大：$1/(1-0.9^t)$ 在 $t=1,2,10,50$ 是 $10,\ 5.26,\ 1.54,\ 1.005$；$1/(1-0.999^t)$ 在 $t=1,2,3000$ 是 $1000,\ 500.25,\ 1.05$〔程式計算〕。$t$ 變大後倍率趨近 1，修正自然退場。**第 1 步有一個漂亮的性質**：$\hat m_1=g_1$、$\hat v_1=g_1^2$，所以步長恰為 $\eta\,g_1/(|g_1|+\varepsilon)\approx\eta\cdot\mathrm{sign}(g_1)$，大小就是 $\eta$，與梯度的尺度無關。

For small $t$ the correction factor is large: $1/(1-0.9^t)$ at $t=1,2,10,50$ is $10,\ 5.26,\ 1.54,\ 1.005$; $1/(1-0.999^t)$ at $t=1,2,3000$ is $1000,\ 500.25,\ 1.05$ [computed by program]. As $t$ grows the factor approaches 1 and the correction fades out. **The first step has a neat property**: $\hat m_1=g_1$ and $\hat v_1=g_1^2$, so the step is exactly $\eta\,g_1/(|g_1|+\varepsilon)\approx\eta\cdot\mathrm{sign}(g_1)$, of size $\eta$ regardless of the scale of the gradient.

不修正會怎樣？來源說開頭幾步會「幾乎停擺」，用預設的 $(\beta_1,\beta_2)=(0.9,\,0.999)$ 計算，結果相反。若梯度恆為 $g$，不修正的步長是 $\eta\,(1-\beta_1^{\,t})/\sqrt{1-\beta_2^{\,t}}$：第 1 步是 $\eta\cdot0.1/\sqrt{0.001}=3.16\,\eta$，因為分子少了 10 倍，分母卻只少了 $\sqrt{1000}=31.6$ 倍，比值被放大。倍率在第 12 步升到最高的 $6.57$，第 100 步還有 $3.24$，要到約 3000 步才降到 $1.03$〔程式計算〕。（若 $\beta_2=0.9$，第 1 步是 $0.1/\sqrt{0.1}=0.316$ 倍，才會偏小：偏向哪個方向取決於兩個 $\beta$ 的搭配。）

What happens without correction? The source says the first steps "almost stall"; with the defaults $(\beta_1,\beta_2)=(0.9,\,0.999)$ the computation gives the opposite. For a constant gradient $g$ the uncorrected step is $\eta\,(1-\beta_1^{\,t})/\sqrt{1-\beta_2^{\,t}}$: at step 1 it is $\eta\cdot0.1/\sqrt{0.001}=3.16\,\eta$, because the numerator is 10 times too small while the denominator is only $\sqrt{1000}=31.6$ times too small, so the ratio is inflated. The factor peaks at $6.57$ at step 12, is still $3.24$ at step 100 and only falls to $1.03$ at about step 3000 [computed by program]. (With $\beta_2=0.9$ the first step is $0.1/\sqrt{0.1}=0.316$ times, which is too small: which way it errs depends on the pairing of the two $\beta$.)

![固定梯度時，未做偏差修正的 Adam 步長相對 η 的倍率（橫軸為對數）：第 1 步是 3.16，第 12 步最高 6.57，約 3000 步才降到 1.03；做偏差修正後恆為 1。（Step size of uncorrected Adam relative to η for a constant gradient (logarithmic horizontal axis): 3.16 at step 1, a peak of 6.57 at step 12 and 1.03 only at about step 3000; with correction it is exactly 1.）](../assets/figures/lec03-kp11-bias-step.png)

**圖說 `lec03-kp11-bias-step`**：先看綠色虛線：修正後的步長恆為 $\eta$ 的 1 倍；再看紅線：不修正時，開頭幾步被放大到 3 到 6.6 倍，之後非常緩慢地回到 1。橫軸是對數，所以「約 3000 步才回到 1」在圖上只占右邊一小段，不要低估它的長度。這張圖假設梯度為定值。〔功能：1、2〕

**Caption `lec03-kp11-bias-step`**: Look at the green dashed line first: the corrected step is always 1 times $\eta$. Then the red line: without correction the first steps are inflated to 3 to 6.6 times, and the return to 1 is very slow. The horizontal axis is logarithmic, so "about 3000 steps to return to 1" occupies only a short stretch on the right; do not underestimate its length. The figure assumes a constant gradient. [Function: 1, 2]

### 5. 比較最佳化器是一場受控實驗（Comparing optimizers is a controlled experiment）

Adam 是目前最常用的預設之一：它兼有方向的記憶、各坐標的縮放與起步的修正，對學習率不太挑。但**沒有定理保證它處處最好**，也不能因為它常見就當成排行榜的第一名。要比較兩個最佳化器，要像做實驗：

Adam is among the most common defaults today: it combines a memory of direction, per-coordinate scaling and a corrected start, and it is fairly forgiving about the learning rate. But **no theorem guarantees that it is the best everywhere**, and being common does not make it the winner of a leaderboard. Comparing two optimizers should be done like an experiment:

| 項目 | 怎麼做 | 為什麼 |
|---|---|---|
| 固定不變 | 同一份資料、模型、初始化、學習率排程與算力預算 | 差異才能歸因於最佳化器 |
| 重複多次 | 多個隨機種子；每個最佳化器各自調過學習率 | 分辨真差異與隨機波動；只調自己偏愛的那個不公平 |
| 要記錄 | 訓練損失、驗證指標、牆鐘時間、跨種子的變異 | 快、好、穩要一起比 |

| Item | What to do | Why |
|---|---|---|
| Hold fixed | The same data, model, initialization, learning-rate schedule and compute budget | so the difference can be attributed to the optimizer |
| Repeat | Several random seeds; tune the learning rate of every optimizer | tell a real difference from random fluctuation; tuning only your favorite is unfair |
| Record | Training loss, validation metric, wall-clock time, variation across seeds | compare fast, good and stable together |

結論也要附條件：「在這組資料、模型與預算下，A 比 B 好」，而不是「A 比 B 好」。下一個問題是：我們比較的訓練損失與驗證指標可能差很多，訓練損失低就夠了嗎？這是 KP12 的主題。

Conclusions need conditions too: "A beats B on this data, model and budget", not "A beats B". The next question: the training loss and the validation metric we compare can differ greatly, so is a low training loss enough? That is the topic of KP12.

### 本單元重點整理（KP11 summary）

**一句話重點**：RMSProp 用指數移動平均取代累加，讓舊梯度逐漸被遺忘（記憶約 $1/(1-\gamma)$ 步）；Adam 再加上梯度本身的指數平均（方向）並做偏差修正，是常用的預設之一，但不保證處處最好，比較要靠受控實驗。

**One-sentence summary**: RMSProp replaces the running sum by an exponential moving average so that old gradients fade (a memory of about $1/(1-\gamma)$ steps); Adam adds an exponential average of the gradient itself (the direction) and a bias correction; it is a common default but not guaranteed best everywhere, and comparisons need controlled experiments.

**白話說明**：AdaGrad 把歷史梯度全部記下，越記越多，步子越走越小。RMSProp 讓記憶有「半衰期」，久遠的事慢慢忘掉，步子能再變大。Adam 又加了一份「最近往哪個方向走」的記憶，並在剛起步、記憶還空的時候把數字放大回正確的尺度。

**Plain explanation**: AdaGrad remembers all past gradients, and the more it remembers the smaller its steps. RMSProp gives the memory a "half-life" so distant events are slowly forgotten and the step can grow again. Adam adds a memory of "which way we have been going lately" and, at the start when the memory is still empty, scales the numbers back to the right size.

**正式定義**：$v_t=\gamma v_{t-1}+(1-\gamma)g_t^2$，$w_t=w_{t-1}-\eta g_t/\sqrt{\varepsilon+v_t}$；Adam：$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$，$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$，$\hat m_t=m_t/(1-\beta_1^{\,t})$，$\hat v_t=v_t/(1-\beta_2^{\,t})$，$w_t=w_{t-1}-\eta\,\hat m_t/(\sqrt{\hat v_t}+\varepsilon)$。

**Formal definition**: $v_t=\gamma v_{t-1}+(1-\gamma)g_t^2$, $w_t=w_{t-1}-\eta g_t/\sqrt{\varepsilon+v_t}$; Adam: $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$, $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$, $\hat m_t=m_t/(1-\beta_1^{\,t})$, $\hat v_t=v_t/(1-\beta_2^{\,t})$, $w_t=w_{t-1}-\eta\,\hat m_t/(\sqrt{\hat v_t}+\varepsilon)$.

**例子**：碗 $f=\tfrac12(x^2+25y^2)$、起點 $(10,2)$、$\eta=0.5$：$x$ 坐標的有效步長，AdaGrad 從 $0.05$ 降到 $0.0114$，RMSProp 先降到 $0.1025$ 再升到 $0.4655$；Adam 第 1 步 $x:10\to9.5$（修正後），不修正是 $10\to8.419$。逐步計算見第 2、4 節與解題示範。

**Example**: bowl $f=\tfrac12(x^2+25y^2)$, start $(10,2)$, $\eta=0.5$: the effective step of the $x$ coordinate falls from $0.05$ to $0.0114$ for AdaGrad and, for RMSProp, dips to $0.1025$ and then rises to $0.4655$; Adam's first step is $x:10\to9.5$ (corrected) and $10\to8.419$ without correction. The computations are in sections 2 and 4 and in the worked solution.

**常見迷思**：

- 「Adam 一定最好」——它是常用的預設之一，沒有定理保證處處最好；比較要用受控實驗。
- 「偏差修正可有可無」——預設的 $\beta$ 下不修正，早期步長會被放大到 3 至 6.6 倍，約 3000 步才退場。
- 「不修正會讓開頭停擺」——那是來源的說法，用預設值計算，結果是放大，不是停擺。
- 「RMSProp 的步長回升就是更好」——回升讓它能繼續走，但固定 $\eta$ 也會在最小點附近彈跳。

**Common misconceptions**:

- "Adam is always the best." It is a common default and no theorem guarantees it is best everywhere; compare with controlled experiments.
- "Bias correction is optional." With the default $\beta$ and no correction the early steps are inflated 3 to 6.6 times and the effect only fades after about 3000 steps.
- "Without correction the first steps stall." That is the source's claim; computed with the defaults the result is inflation, not stalling.
- "A recovering RMSProp step is simply better." The recovery lets it keep moving, but with a fixed $\eta$ it also bounces around the minimum.

**圖**：本單元兩張圖，`lec03-kp11-effective-step`（兩種方法的有效步長，功能 1、2）與 `lec03-kp11-bias-step`（不修正的步長倍率，功能 1、2），以 `assets/figures/src/lec03-kp11-figures.py` 繪製，數字都由同一支程式印出。Adam 的資料流在簡報中以原生節點與連線呈現。

**Figure**: Two figures in this unit, `lec03-kp11-effective-step` (the effective steps of the two methods; functions 1, 2) and `lec03-kp11-bias-step` (the step factor without correction; functions 1, 2), drawn by `assets/figures/src/lec03-kp11-figures.py`, which prints every number from the same program. The data flow of Adam is drawn with native nodes and connectors in the deck.

**想深入**（選讀）：Adam 的收斂性有爭議。`[補充]` Reddi 等人（ICLR 2018）給出一個凸問題的例子，說明 Adam 的 $v$ 因為會「遺忘」，在某些情況下可能不收斂，並提出修改版本 AMSGrad；這也提醒我們，遺忘舊梯度既帶來步長回升，也帶來這類風險。另外，RMSProp 與 Adam 的分母是梯度平方的平均，它與曲率（Hessian）有關，但並不等於以對角矩陣近似 Hessian，這一點留給進階課程。

**Going deeper** (optional): The convergence of Adam is debated. `[Added]` Reddi et al. (ICLR 2018) give a convex example showing that, because its $v$ "forgets", Adam may fail to converge in some situations, and propose a modification called AMSGrad; this is a reminder that forgetting old gradients brings both the recovery of the step and such risks. Also, the denominator of RMSProp and Adam is an average of squared gradients; it is related to curvature (the Hessian) but is not the same as a diagonal approximation of the Hessian, a point left to advanced courses.

**小練習**：

1. $\gamma=0.99$：記憶窗約幾步？半衰期是幾步？離現在 100 步以前的梯度平方，權重是多少？（提示：$1/(1-\gamma)$、$\ln0.5/\ln\gamma$、$\gamma^{100}$。）
2. 目標 $f(w)=\tfrac12w^2$（梯度 $g=w$），起點 $w_0=-8$，$\eta=0.1$，$\beta_1=0.9$，$\beta_2=0.999$：算 Adam（有修正）的第 1 步，並說明它與 $\eta\cdot\mathrm{sign}(g)$ 的關係。（提示：第 1 步 $\hat m_1=g_1$、$\hat v_1=g_1^2$。）
3. 梯度恆為 $g=4$，$\beta_1=0.9$：寫出未修正的 $m_1,m_2,m_3$，並驗證它們等於 $4\,(1-0.9^t)$；再算修正後的 $\hat m_t$。（提示：$m_t=0.9\,m_{t-1}+0.4$。）

**Practice**:

1. $\gamma=0.99$: about how many steps is the memory window, what is the half-life, and what is the weight of a squared gradient from 100 steps ago? (Hint: $1/(1-\gamma)$, $\ln0.5/\ln\gamma$, $\gamma^{100}$.)
2. Objective $f(w)=\tfrac12w^2$ (gradient $g=w$), start $w_0=-8$, $\eta=0.1$, $\beta_1=0.9$, $\beta_2=0.999$: compute the first step of Adam (with correction) and relate it to $\eta\cdot\mathrm{sign}(g)$. (Hint: at step 1, $\hat m_1=g_1$ and $\hat v_1=g_1^2$.)
3. The gradient is constant $g=4$ and $\beta_1=0.9$: write the uncorrected $m_1,m_2,m_3$, verify they equal $4\,(1-0.9^t)$, and compute the corrected $\hat m_t$. (Hint: $m_t=0.9\,m_{t-1}+0.4$.)

**解題示範**（以玩具碗的 $x$ 坐標示範：$g=x$，$x_0=10$，$\eta=0.5$，$\beta_1=0.9$，$\beta_2=0.999$，手算 Adam 的前兩步，含偏差修正）：

1. 第 1 步：$g_1=x_0=10$。$m_1=0.9\cdot0+0.1\cdot10=1$；$v_1=0.999\cdot0+0.001\cdot10^2=0.1$。
2. 修正：$1-\beta_1^1=0.1$，$1-\beta_2^1=0.001$，所以 $\hat m_1=1/0.1=10$，$\hat v_1=0.1/0.001=100$。
3. 步長：$\eta\,\hat m_1/\sqrt{\hat v_1}=0.5\cdot10/10=0.5$（$\varepsilon$ 可忽略），所以 $x_1=10-0.5=9.5$。這恰為 $\eta\cdot\mathrm{sign}(g_1)$。
4. 第 2 步：$g_2=x_1=9.5$。$m_2=0.9\cdot1+0.1\cdot9.5=1.85$；$v_2=0.999\cdot0.1+0.001\cdot9.5^2=0.0999+0.09025=0.19015$。
5. 修正：$1-0.9^2=0.19$，$1-0.999^2=0.001999$，所以 $\hat m_2=1.85/0.19=9.7368$，$\hat v_2=0.19015/0.001999=95.1226$，$\sqrt{\hat v_2}=9.7531$。
6. 步長：$0.5\cdot9.7368/9.7531=0.4992$，所以 $x_2=9.5-0.4992=9.0008$。
7. 對照：不修正時第 1 步是 $0.5\cdot1/\sqrt{0.1}=1.581$，$x_1=8.419$；第 2 步 $g_2=8.419$，步長 $2.108$，$x_2=6.311$。兩步後修正版在 $9.0008$，未修正版衝到 $6.311$，第 1 步的步子是修正版的 $3.16$ 倍，第 2 步是 $2.108/0.4992=4.22$ 倍〔程式計算〕。

**Worked solution** (on the $x$ coordinate of the toy bowl: $g=x$, $x_0=10$, $\eta=0.5$, $\beta_1=0.9$, $\beta_2=0.999$; the first two Adam steps by hand, with bias correction):

1. Step 1: $g_1=x_0=10$. $m_1=0.9\cdot0+0.1\cdot10=1$; $v_1=0.999\cdot0+0.001\cdot10^2=0.1$.
2. Correction: $1-\beta_1^1=0.1$ and $1-\beta_2^1=0.001$, so $\hat m_1=1/0.1=10$ and $\hat v_1=0.1/0.001=100$.
3. Step size: $\eta\,\hat m_1/\sqrt{\hat v_1}=0.5\cdot10/10=0.5$ ($\varepsilon$ is negligible), so $x_1=10-0.5=9.5$. This is exactly $\eta\cdot\mathrm{sign}(g_1)$.
4. Step 2: $g_2=x_1=9.5$. $m_2=0.9\cdot1+0.1\cdot9.5=1.85$; $v_2=0.999\cdot0.1+0.001\cdot9.5^2=0.0999+0.09025=0.19015$.
5. Correction: $1-0.9^2=0.19$ and $1-0.999^2=0.001999$, so $\hat m_2=1.85/0.19=9.7368$, $\hat v_2=0.19015/0.001999=95.1226$ and $\sqrt{\hat v_2}=9.7531$.
6. Step size: $0.5\cdot9.7368/9.7531=0.4992$, so $x_2=9.5-0.4992=9.0008$.
7. Comparison: without correction the first step is $0.5\cdot1/\sqrt{0.1}=1.581$ and $x_1=8.419$; at step 2, $g_2=8.419$, the step is $2.108$ and $x_2=6.311$. After two steps the corrected version sits at $9.0008$ while the uncorrected one has rushed to $6.311$, the first step is $3.16$ times the corrected one and the second is $2.108/0.4992=4.22$ times [computed by program].

### 來源更正與取捨（Corrections to the source）

- 來源稱 Adam 是「現代深度學習的預設霸主」「絕對首選」「收斂最穩、試錯成本最低」，也稱 $\eta=10^{-3}$ 是「宇宙常數」。這些說法沒有定理或對照實驗支持，本講義改為「常用的預設之一，不保證處處最好」，並在第 5 節說明比較要靠受控實驗。
- 來源說不做偏差修正，開頭幾步「幾乎停擺」。用預設的 $(\beta_1,\beta_2)=(0.9,\,0.999)$ 計算，梯度固定時未修正的步長是 $\eta\,(1-\beta_1^{\,t})/\sqrt{1-\beta_2^{\,t}}$，第 1 步是 $3.16\,\eta$、第 12 步最高 $6.57\,\eta$，是放大而不是停擺；只有 $\beta_2$ 較小（例如 $0.9$）時才會偏小。
- 來源的式 (44)–(45) 對 $\boldsymbol m^{(k+1)}$ 除以 $1-\gamma^k$，下標差了一位；本講義統一用 $t$，第 $t$ 步的 $m_t$ 除以 $1-\beta_1^{\,t}$。無偏性的推導需要「梯度期望在局部穩定」的假設，本講義明寫這個假設。
- 來源說 $\varepsilon=10^{-8}$「剛好接近單精度下界」。這沒有依據，不採用；$\varepsilon$ 的作用只是避免除以零。來源把分母寫成 $\sqrt{\varepsilon+\hat v}$，本講義用 $\sqrt{\hat v}+\varepsilon$（原始 Adam 論文的寫法），兩者在 $\varepsilon$ 很小時幾乎相同。
- 來源說對角縮放「等價於以對角矩陣近似 Hessian」，這個說法不精確（它是梯度平方的平均，與曲率有關但不等同），不採用。來源說 RMSProp 是「非平穩環境王者」、Adam 是 Transformer 與大型語言模型的通用首選、SGD 加動量在視覺任務泛化較好，這些是經驗說法，不當成事實採用。
- 來源的 RMSProp 出處（Hinton 2012 年課程講義，不是正式論文）保留，並如實標註。Adam 的收斂爭議（Reddi 等人，2018）放在「想深入」一句話提及，不進簡報。
- AdaGrad 的定義與 KP10 相同（$G_t=\sum g_s^2$，分母 $\sqrt{\varepsilon+G_t}$，$\varepsilon=10^{-8}$），但本講義用自己的程式重算，並明寫 $\eta=0.5$。

- The source calls Adam "the default king of modern deep learning", "the absolute first choice", "the most stable and cheapest to try", and calls $\eta=10^{-3}$ a "cosmic constant". No theorem or controlled experiment supports these claims; this note says "among the common defaults, not guaranteed best everywhere" and explains in section 5 that comparing needs controlled experiments.
- The source says that without bias correction the first steps "almost stall". Computed with the defaults $(\beta_1,\beta_2)=(0.9,\,0.999)$, the uncorrected step for a constant gradient is $\eta\,(1-\beta_1^{\,t})/\sqrt{1-\beta_2^{\,t}}$, which is $3.16\,\eta$ at step 1 and peaks at $6.57\,\eta$ at step 12: inflation, not stalling; only a smaller $\beta_2$ (say $0.9$) errs low.
- The source's equations (44)–(45) divide $\boldsymbol m^{(k+1)}$ by $1-\gamma^k$, an index off by one; this note uses $t$ throughout and divides $m_t$ by $1-\beta_1^{\,t}$. The proof of unbiasedness needs the assumption that the expected gradient is locally steady, which this note states explicitly.
- The source says $\varepsilon=10^{-8}$ is "just near the lower bound of single precision". That has no basis and is not adopted; $\varepsilon$ only guards against division by zero. The source writes the denominator as $\sqrt{\varepsilon+\hat v}$; this note uses $\sqrt{\hat v}+\varepsilon$ (the form of the original Adam paper); the two are nearly identical when $\varepsilon$ is tiny.
- The source says the diagonal scaling is "equivalent to approximating the Hessian with a diagonal matrix"; this is imprecise (it is an average of squared gradients, related to curvature but not the same) and is not adopted. The source's claims that RMSProp is the "king of non-stationary settings", that Adam is the universal first choice for Transformers and large language models, and that SGD with momentum generalizes better in vision are empirical statements and are not adopted as facts.
- The source's origin of RMSProp (Hinton's 2012 course slides, not a paper) is kept and stated honestly. The convergence debate about Adam (Reddi et al., 2018) is mentioned in one sentence under "Going deeper" and kept out of the deck.
- AdaGrad is defined as in KP10 ($G_t=\sum g_s^2$, denominator $\sqrt{\varepsilon+G_t}$, $\varepsilon=10^{-8}$), but this note recomputes it with its own program and states $\eta=0.5$ explicitly.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 指數移動平均 | exponential moving average (EMA) |
| 遺忘因子（衰減率） | forgetting factor (decay rate) |
| 記憶窗 | memory window |
| 半衰期 | half-life |
| 有效步長 | effective step size |
| 一階矩（梯度的平均，方向） | first moment (mean of the gradient, direction) |
| 二階矩（梯度平方的平均，尺度） | second moment (mean of the squared gradient, scale) |
| 偏差修正 | bias correction |
| 受控實驗 | controlled experiment |
| 牆鐘時間 | wall-clock time |
