# 第4週 KP06：Stein 引理與 SURE（Stein's lemma and SURE）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第4週「正則化與過擬合」的第六個知識點（KP06），承接 KP05 留下的問題：KP05 的乾淨拆解只對沒參與訓練的點成立；如果點參與了訓練，預測 $\hat f_i$ 與觀測 $y_i$ 不再獨立，交叉項不是 0，我們要怎麼修正？先備知識是期望、微分與常態分布（高斯分布）、KP04（$y=f+\varepsilon$）與 KP05；第一次出現的符號都會定義，所有數字都可以自己重算。內容依 CCC-LNS 第5章 KP6 的主文與教授祕笈消化後重新撰寫，讀取日期 2026-10-20；來源沒有的內容標示 `[補充]`。

This is the sixth knowledge point (KP06) of Week 4, "Regularization and overfitting". It picks up the question KP05 left open: the clean split of KP05 holds only for points that took no part in training; if a point was used in training, the prediction $\hat f_i$ and the observation $y_i$ are no longer independent, the cross term is not 0, and we need a correction. It assumes expectation, derivatives and the normal (Gaussian) distribution, KP04 ($y=f+\varepsilon$) and KP05; every symbol is defined when it first appears and every number can be recomputed. The text is rewritten from the KP6 main text and professor's guide of CCC-LNS Chapter 5 (accessed on 2026-10-20); content that the source does not contain is marked `[Added]`.

[下載本單元投影片](../slides/lec04-kp06-stein-lemma-and-sure.jbd.json)

## KP06 Stein 引理與 SURE（Stein's lemma and SURE）

### 學習目標（Learning objectives）

讀完之後，你能陳述 Stein 引理 $\mathbb E\big((z-\mu)g(z)\big)=\sigma^2\,\mathbb E\big(g'(z)\big)$ 與它需要的條件，並用 $g(z)=cz$ 與 $g(z)=z^3$ 手算驗證；能從引理推出訓練點的式子 $\mathbb E\big((\hat f_0-y_0)^2\big)=\mathbb E\big((\hat f_0-f_0)^2\big)+\sigma^2-2\sigma^2\,\mathbb E\big(\partial\hat f_0/\partial y_0\big)$，再加總成 $\mathbf{Err}=\mathbf{err}-n_{tr}\sigma^2+2\sigma^2\sum_i\partial\hat f_i/\partial y_i$；能用這條式子由訓練誤差估出真實誤差，並說明為什麼敏感度越大，訓練誤差越會低估真實誤差；也能說出 SURE 的限制。

After reading, you can state Stein's lemma $\mathbb E\big((z-\mu)g(z)\big)=\sigma^2\,\mathbb E\big(g'(z)\big)$ with the conditions it needs and verify it by hand for $g(z)=cz$ and $g(z)=z^3$; derive from it the formula for a training point $\mathbb E\big((\hat f_0-y_0)^2\big)=\mathbb E\big((\hat f_0-f_0)^2\big)+\sigma^2-2\sigma^2\,\mathbb E\big(\partial\hat f_0/\partial y_0\big)$ and sum it into $\mathbf{Err}=\mathbf{err}-n_{tr}\sigma^2+2\sigma^2\sum_i\partial\hat f_i/\partial y_i$; use this formula to estimate the true error from the training error and explain why a larger sensitivity makes the training error underestimate the true error more; and name the limits of SURE.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $z$, $\mu$, $\sigma^2$ | 常態隨機變數 $z\sim\mathcal N(\mu,\sigma^2)$ 與它的平均、變異數 | a normal random variable $z\sim\mathcal N(\mu,\sigma^2)$ with its mean and variance |
| $g(z)$ | 一個可微函數 | a differentiable function |
| $g'(z)$ | $g$ 的導數 $\partial g/\partial z$ | the derivative $\partial g/\partial z$ of $g$ |
| $y_i=f_i+\varepsilon_i$ | 第 $i$ 個訓練點的觀測、真實輸出與雜訊（KP04） | observation, true output and noise of training point $i$ (KP04) |
| $\hat f_i$ | 用全部訓練資料訓練出的模型在第 $i$ 點的預測 | prediction at point $i$ of the model trained on all training data |
| $\partial\hat f_i/\partial y_i$ | 敏感度：只把 $y_i$ 動一點點，$\hat f_i$ 跟著動多少 | sensitivity: how much $\hat f_i$ moves when only $y_i$ is nudged |
| $n_{tr}$ | 訓練點的個數 | number of training points |
| $\mathbf{err}$ | 訓練集上的經驗誤差 $\sum_i(\hat f_i-y_i)^2$，算得出來 | empirical error $\sum_i(\hat f_i-y_i)^2$ on the training set, computable |
| $\mathbf{Err}$ | 真實誤差 $\sum_i(\hat f_i-f_i)^2$，看不到 | true error $\sum_i(\hat f_i-f_i)^2$, not observable |

### 1. 為什麼訓練點的交叉項不是 0（Why the cross term of a training point is not zero）

KP05 的關鍵是 $\hat f_0$ 與 $\varepsilon_0$ 獨立。但訓練點不同：模型是用這筆資料 $y_i=f_i+\varepsilon_i$ 訓練出來的，所以 $\hat f_i$ 是 $y_i$ 的函數，也就跟 $\varepsilon_i$ 有關。想像把 $\varepsilon_i$ 調大一點：$y_i$ 變大，擬合出來的曲線會被拉向它，$\hat f_i$ 也跟著變大。於是

$$
\mathbb E\big[(\hat f_i-f_i)\,\varepsilon_i\big]\neq 0 .
$$

In KP05 the key was that $\hat f_0$ and $\varepsilon_0$ are independent. A training point is different: the model was trained on the very data $y_i=f_i+\varepsilon_i$, so $\hat f_i$ is a function of $y_i$ and therefore depends on $\varepsilon_i$. Imagine making $\varepsilon_i$ a little larger: $y_i$ grows, the fitted curve is pulled towards it, and $\hat f_i$ grows too. So the cross term above is not zero.

這個交叉項是多少？一個很自然的猜測是：它跟「$\hat f_i$ 對 $y_i$ 有多敏感」成正比。Stein 引理把這個猜測變成一條精確的等式，條件是雜訊為高斯分布。

How large is this cross term? A natural guess is that it is proportional to how sensitive $\hat f_i$ is to $y_i$. Stein's lemma turns that guess into an exact equality, provided the noise is Gaussian.

### 2. Stein 引理（Stein's lemma）

**Stein 引理**：設 $z\sim\mathcal N(\mu,\sigma^2)$，$g$ 是可微函數，且 $g$ 與 $g'$ 的期望存在（積分得出來），則

$$
\mathbb E\big((z-\mu)\,g(z)\big)=\sigma^2\,\mathbb E\big(g'(z)\big).
$$

**Stein's lemma**: let $z\sim\mathcal N(\mu,\sigma^2)$ and let $g$ be differentiable with finite expectations of $g$ and $g'$. Then the equality above holds.

白話：「雜訊（$z-\mu$）與函數 $g(z)$ 的相關」等於「$\sigma^2$ 乘上函數的斜率的平均」。函數對 $z$ 越陡，它跟雜訊的相關就越大。來源把它寫成多維形式 $\mathbb E\big((\mathbf z-\boldsymbol\mu)^\top\mathbf g(\mathbf z)\big)=\sigma^2\sum_i\mathbb E(\partial g_i/\partial z_i)$，其中 $\mathbf z\sim\mathcal N(\boldsymbol\mu,\sigma^2\mathbf I)$；每一維各自用一次單變數版本，再相加。

In words: "the correlation of the noise ($z-\mu$) with the function $g(z)$" equals "$\sigma^2$ times the average slope of the function". The steeper the function is in $z$, the larger its correlation with the noise. The source writes the multivariate form $\mathbb E\big((\mathbf z-\boldsymbol\mu)^\top\mathbf g(\mathbf z)\big)=\sigma^2\sum_i\mathbb E(\partial g_i/\partial z_i)$ with $\mathbf z\sim\mathcal N(\boldsymbol\mu,\sigma^2\mathbf I)$; apply the one-dimensional version in each coordinate and add.

**先手算兩個例子**。取 $\mu=0$。

1. $g(z)=cz$：左邊 $\mathbb E(z\cdot cz)=c\,\mathbb E(z^2)=c\sigma^2$；右邊 $\sigma^2\,\mathbb E(c)=c\sigma^2$。兩邊相等。
2. $g(z)=z^3$：左邊 $\mathbb E(z\cdot z^3)=\mathbb E(z^4)=3\sigma^4$（常態分布的四次動差）；右邊 $\sigma^2\,\mathbb E(3z^2)=\sigma^2\cdot3\sigma^2=3\sigma^4$。兩邊相等。

**Two hand examples first.** Take $\mu=0$.

1. $g(z)=cz$: the left side is $\mathbb E(z\cdot cz)=c\,\mathbb E(z^2)=c\sigma^2$; the right side is $\sigma^2\,\mathbb E(c)=c\sigma^2$. They agree.
2. $g(z)=z^3$: the left side is $\mathbb E(z\cdot z^3)=\mathbb E(z^4)=3\sigma^4$ (the fourth moment of a normal variable); the right side is $\sigma^2\,\mathbb E(3z^2)=\sigma^2\cdot3\sigma^2=3\sigma^4$. They agree.

**`[補充]` 一維證明（分部積分）**。令 $\varphi(z)$ 是 $\mathcal N(\mu,\sigma^2)$ 的密度，它的導數是 $\varphi'(z)=-\dfrac{z-\mu}{\sigma^2}\,\varphi(z)$，也就是 $(z-\mu)\varphi(z)=-\sigma^2\varphi'(z)$。所以

$$
\mathbb E\big((z-\mu)g(z)\big)=\int g(z)\,(z-\mu)\varphi(z)\,dz=-\sigma^2\int g(z)\,\varphi'(z)\,dz .
$$

對右邊做分部積分：$\int g\,\varphi'\,dz=\big[g\varphi\big]_{-\infty}^{\infty}-\int g'\varphi\,dz$。密度 $\varphi$ 在無窮遠處衰減得比任何多項式快，邊界項 $\big[g\varphi\big]$ 為 0（這就是「$g$ 的增長不能太快」的條件），於是

$$
\mathbb E\big((z-\mu)g(z)\big)=\sigma^2\int g'(z)\,\varphi(z)\,dz=\sigma^2\,\mathbb E\big(g'(z)\big).
$$

**`[Added]` One-dimensional proof (integration by parts).** Let $\varphi(z)$ be the density of $\mathcal N(\mu,\sigma^2)$. Its derivative is $\varphi'(z)=-\dfrac{z-\mu}{\sigma^2}\,\varphi(z)$, that is $(z-\mu)\varphi(z)=-\sigma^2\varphi'(z)$, which gives the first display. Integrating the right side by parts gives $\int g\,\varphi'\,dz=\big[g\varphi\big]_{-\infty}^{\infty}-\int g'\varphi\,dz$. The density $\varphi$ decays faster than any polynomial, so the boundary term $\big[g\varphi\big]$ is 0 (this is the condition that $g$ must not grow too fast), and the second display follows.

這個證明同時告訴我們兩件事：高斯密度的特殊形狀（導數正比於自己乘上 $z-\mu$）正是引理成立的原因，所以換成別的雜訊分布，引理就不成立；而 $g$ 要可微、增長不能太快，才有分部積分與邊界項為 0。

The proof shows two things at once: the special shape of the Gaussian density (its derivative is proportional to itself times $z-\mu$) is exactly why the lemma holds, so for other noise distributions it fails; and $g$ must be differentiable and must not grow too fast, so that integration by parts is allowed and the boundary term is 0.

**用電腦再驗一次**〔程式計算〕。取 $g(z)=\tanh z$（不是多項式，手算不容易），$\mu=0$，每個 $\sigma$ 抽 400 萬個樣本：

| $\sigma$ | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|
| $\mathbb E\big(z\,g(z)\big)$ | 0.2066 | 0.6061 | 1.0341 | 1.4580 |
| $\sigma^2\,\mathbb E\big(g'(z)\big)$ | 0.2066 | 0.6055 | 1.0332 | 1.4597 |

兩列在 $0.002$ 以內一致，差距來自抽樣誤差。線性例也一樣：$c=2$、$\sigma=1.5$，理論值 $c\sigma^2=4.5$，模擬 $4.4962$。

**Check it again on a computer** [computed by program]. Take $g(z)=\tanh z$ (not a polynomial, so hand computation is not easy), $\mu=0$ and 4 million samples for each $\sigma$; the table above shows both sides. The two rows agree within $0.002$, and the gap is sampling error. The linear case agrees as well: $c=2$ and $\sigma=1.5$ give the theoretical value $c\sigma^2=4.5$ and the simulation $4.4962$.

![長條圖：橫軸是 σ 等於 0.5、1、1.5、2，每個 σ 有一藍一橘兩根柱子，分別是 E(z g(z)) 與 σ² E(g′(z))，兩根柱子的高度幾乎一樣，而且 σ 越大兩者都越高。（Bar chart: the horizontal axis lists sigma equal to 0.5, 1, 1.5 and 2; each sigma has a blue and an orange bar for E(z g(z)) and sigma squared times E(g prime of z); the two bars are almost equally tall, and both grow with sigma.）](../assets/figures/lec04-kp06-stein.png)

**圖說 `lec04-kp06-stein`**：先看每一對柱子：藍色是左邊 $\mathbb E(z\,g(z))$，橘色是右邊 $\sigma^2\,\mathbb E(g'(z))$，高度一致就代表引理成立；再看四對之間的變化：$\sigma$ 越大，雜訊越大，兩邊一起變大，但每一對始終相等。〔功能：1〕

**Caption `lec04-kp06-stein`**: Look first at each pair of bars: blue is the left side $\mathbb E(z\,g(z))$ and orange is the right side $\sigma^2\,\mathbb E(g'(z))$; equal heights mean that the lemma holds. Then compare the four pairs: the larger $\sigma$ is, the larger the noise, and both sides grow together, but every pair stays equal. [Function: 1]

### 3. 從引理推出訓練點的修正（From the lemma to the correction for a training point）

現在把引理用到 $y_0=f_0+\varepsilon_0$。訓練點 $0$ 的平方誤差展開成（和 KP05 相同，只是還沒丟掉交叉項）：

$$
(\hat f_0-y_0)^2=(\hat f_0-f_0)^2-2\,\varepsilon_0\,(\hat f_0-f_0)+\varepsilon_0^2 .
$$

Now apply the lemma to $y_0=f_0+\varepsilon_0$. The squared error of training point $0$ expands to the display above (as in KP05, but the cross term is not thrown away yet).

取期望。第一項是 $\mathbb E(\hat f_0-f_0)^2$，第三項是 $\sigma^2$。交叉項用引理：令 $z=\varepsilon_0$（所以 $\mu=0$），令 $g=\hat f_0-f_0$。固定真實輸出 $f_0$，$g$ 是 $\varepsilon_0$ 的函數（因為 $\hat f_0$ 是 $y_0=f_0+\varepsilon_0$ 的函數），並且

$$
\frac{\partial g}{\partial\varepsilon_0}=\frac{\partial\hat f_0}{\partial y_0}.
$$

引理給出 $\mathbb E\big(\varepsilon_0(\hat f_0-f_0)\big)=\sigma^2\,\mathbb E\big(\partial\hat f_0/\partial y_0\big)$。代回去：

$$
\mathbb E\big((\hat f_0-y_0)^2\big)=\mathbb E\big((\hat f_0-f_0)^2\big)+\sigma^2-2\sigma^2\,\mathbb E\!\left(\frac{\partial\hat f_0}{\partial y_0}\right).
$$

Take expectations. The first term is $\mathbb E(\hat f_0-f_0)^2$ and the third is $\sigma^2$. For the cross term use the lemma with $z=\varepsilon_0$ (so $\mu=0$) and $g=\hat f_0-f_0$. With the true output $f_0$ held fixed, $g$ is a function of $\varepsilon_0$ (because $\hat f_0$ is a function of $y_0=f_0+\varepsilon_0$), and its derivative is $\partial\hat f_0/\partial y_0$. The lemma gives $\mathbb E\big(\varepsilon_0(\hat f_0-f_0)\big)=\sigma^2\,\mathbb E\big(\partial\hat f_0/\partial y_0\big)$; substituting gives the last display above.

和 KP05 比：多了最後一項 $-2\sigma^2\,\mathbb E(\partial\hat f_0/\partial y_0)$。如果模型完全不看這筆資料（$\partial\hat f_0/\partial y_0=0$，像測試點），它就消失，回到 KP05。敏感度越大，這一項越負：訓練誤差比「真實誤差加雜訊」更小，也就是訓練誤差變得更樂觀。

Compared with KP05 there is one extra term $-2\sigma^2\,\mathbb E(\partial\hat f_0/\partial y_0)$. If the model ignores this data point completely ($\partial\hat f_0/\partial y_0=0$, as for a test point) the term vanishes and we are back at KP05. The larger the sensitivity, the more negative the term: the training error is smaller than "true error plus noise", that is, it becomes more optimistic.

對全部 $n_{tr}$ 個訓練點把式子**相加**（不是平均），移項，得到 SURE（Stein 無偏風險估計，Stein's unbiased risk estimate）：

$$
\mathbf{Err}=\mathbf{err}-n_{tr}\,\sigma^2+2\sigma^2\sum_{i=1}^{n_{tr}}\frac{\partial\hat f_i}{\partial y_i}.
$$

Add the formula **up** (not average) over all $n_{tr}$ training points and rearrange; this gives SURE (Stein's unbiased risk estimate), the display above.

三項各有各的意思：$\mathbf{err}$ 是算得出來的訓練誤差；$-n_{tr}\sigma^2$ 扣掉雜訊地板（和 KP05 一樣）；$+2\sigma^2\sum_i\partial\hat f_i/\partial y_i$ 是**複雜度懲罰**：模型越會「跟著每筆資料走」，加回去的就越多。這一項會在 KP07 具體化成「模型的有效參數個數」。

The three terms mean: $\mathbf{err}$ is the computable training error; $-n_{tr}\sigma^2$ removes the noise floor (as in KP05); $+2\sigma^2\sum_i\partial\hat f_i/\partial y_i$ is the **complexity penalty**: the more the model "follows every data point", the more is added back. KP07 will turn this term into the "effective number of parameters" of the model.

### 4. 數值例與限制（A numerical example and the limits）

**來源的例子**：$n_{tr}=30$，每點平均訓練 MSE 為 $3$（所以 $\mathbf{err}=30\times3=90$），$\sigma^2=1$，每個訓練點的平均敏感度為 $0.5$。

$$
\mathbf{Err}=90-30\cdot1+2\cdot1\cdot30\cdot0.5=90-30+30=90 ,
$$

每點平均 $90/30=3$，剛好等於訓練 MSE：懲罰項（$+30$）剛好抵銷了雜訊地板（$-30$）。若平均敏感度是 $0.8$：

$$
\mathbf{Err}=90-30+2\cdot1\cdot30\cdot0.8=90-30+48=108 ,
$$

每點平均 $108/30=3.6$。

**The source's example**: $n_{tr}=30$, the mean training MSE per point is $3$ (so $\mathbf{err}=30\times3=90$), $\sigma^2=1$, and the mean sensitivity of a training point is $0.5$. The first display gives $\mathbf{Err}=90$, a per-point mean of $3$, exactly the training MSE: the penalty ($+30$) cancels the noise floor ($-30$). With a mean sensitivity of $0.8$ the second display gives $\mathbf{Err}=108$, a per-point mean of $3.6$.

![長條圖：三根柱子，藍色是訓練誤差 err 等於 90；綠色是敏感度 0.5 時的真實誤差 Err 等於 90；紅色是敏感度 0.8 時的真實誤差 Err 等於 108；一條虛線畫在 90。（Bar chart with three bars: blue is the training error err equal to 90; green is the true error Err equal to 90 at sensitivity 0.5; red is the true error Err equal to 108 at sensitivity 0.8; a dashed line is drawn at 90.）](../assets/figures/lec04-kp06-sure.png)

**圖說 `lec04-kp06-sure`**：先看虛線：它標出訓練誤差 90；綠色柱子剛好碰到虛線，代表敏感度 0.5 時訓練誤差就是真實誤差的無偏估計；紅色柱子高出虛線 18，代表敏感度升到 0.8 時，真實誤差比訓練誤差看起來的大 $108-90=18$。〔功能：2〕

**Caption `lec04-kp06-sure`**: Look first at the dashed line: it marks the training error 90. The green bar just touches it, meaning that at sensitivity 0.5 the training error is an unbiased estimate of the true error; the red bar is 18 above it, meaning that at sensitivity 0.8 the true error is $108-90=18$ larger than the training error suggests. [Function: 2]

**三個檢查**（自己可以驗算）：

1. 敏感度為 $0$（模型完全不看訓練點）：$\mathbf{Err}=\mathbf{err}-n_{tr}\sigma^2$，正好是 KP05。
2. 臨界點：$\mathbf{Err}=\mathbf{err}$ 要求 $-n_{tr}\sigma^2+2\sigma^2\sum_i\partial\hat f_i/\partial y_i=0$，即平均敏感度 $=\tfrac12$，就是來源例子裡的 $0.5$。
3. 方向：平均敏感度大於 $\tfrac12$ 時 $\mathbf{Err}>\mathbf{err}$，訓練誤差低估真實誤差；小於 $\tfrac12$ 時反過來。（來源說 $0.8$ 時真實誤差「被高估」到 $3.6$，應是「訓練 MSE 低估了真實誤差」，見下面的更正。）

**Three checks** (you can verify them yourself): (1) with sensitivity $0$ (the model ignores the training points) $\mathbf{Err}=\mathbf{err}-n_{tr}\sigma^2$, exactly KP05; (2) the break-even point: $\mathbf{Err}=\mathbf{err}$ requires $-n_{tr}\sigma^2+2\sigma^2\sum_i\partial\hat f_i/\partial y_i=0$, i.e. a mean sensitivity of $\tfrac12$, the $0.5$ of the source's example; (3) direction: when the mean sensitivity is above $\tfrac12$, $\mathbf{Err}>\mathbf{err}$ and the training error underestimates the true error; below $\tfrac12$ it is the other way round. (The source says that at $0.8$ the true error is "overestimated" to $3.6$; it should be "the training MSE underestimates the true error", see the corrections below.)

**限制**：(1) 引理需要**高斯**雜訊；KP05 的等式只需要平均 0 與有限變異數，這裡多了一個假設。(2) 需要 $\hat f_i$ 對 $y_i$ 可微，且訓練是確定的（同樣的資料得到同樣的模型）；有隨機初始化、早停或不連續選擇的學習器，要小心。(3) $\sigma^2$ 要事先知道或另外估計（KP07 會看估計方法）。(4) 這是理論上的估計，**不能取代驗證集或交叉驗證**。(5) 期望是對雜訊 $\varepsilon$ 取，固定真實函數 $f$ 與輸入 $\mathbf x_i$；$\partial\hat f_i/\partial y_i$ 本身也是隨機的，實務上把觀察到的資料代進去當估計 `[補充]`。

**Limits**: (1) The lemma needs **Gaussian** noise; the identity of KP05 needs only mean 0 and a finite variance, so this is one more assumption. (2) $\hat f_i$ must be differentiable in $y_i$ and the training must be deterministic (the same data give the same model); learners with random initialization, early stopping or discontinuous choices need care. (3) $\sigma^2$ must be known beforehand or estimated separately (KP07 will look at estimators). (4) This is a theoretical estimate; it **does not replace a validation set or cross-validation**. (5) The expectation is over the noise $\varepsilon$, with the true function $f$ and the inputs $\mathbf x_i$ held fixed; $\partial\hat f_i/\partial y_i$ is itself random, and in practice one plugs in the observed data as an estimate `[Added]`.

這一章後面要抑制的就是這個敏感度：KP07 把它變成一個可以算的數字，KP08 起再看它怎麼和過擬合與正則化連起來。

What the rest of this chapter tries to hold down is exactly this sensitivity: KP07 turns it into a number we can compute, and from KP08 on we see how it connects to overfitting and regularization.

### 本單元重點整理（KP06 summary）

**一句話重點**：對參與訓練的點，$\hat f_i$ 與 $\varepsilon_i$ 不獨立，Stein 引理（高斯雜訊）把交叉項換成 $\sigma^2\,\mathbb E(\partial\hat f_i/\partial y_i)$，得到 $\mathbf{Err}=\mathbf{err}-n_{tr}\sigma^2+2\sigma^2\sum_i\partial\hat f_i/\partial y_i$；敏感度越大，訓練誤差越低估真實誤差。

**One-sentence summary**: For a point used in training, $\hat f_i$ and $\varepsilon_i$ are not independent; Stein's lemma (Gaussian noise) replaces the cross term by $\sigma^2\,\mathbb E(\partial\hat f_i/\partial y_i)$, which gives $\mathbf{Err}=\mathbf{err}-n_{tr}\sigma^2+2\sigma^2\sum_i\partial\hat f_i/\partial y_i$; the larger the sensitivity, the more the training error underestimates the true error.

**白話說明**：老師改自己出的考卷難免手軟，手軟的程度和「學生答案一變，分數會不會跟著變」有關。分數對答案越敏感，要打的折扣越大。模型也一樣：它對訓練資料越敏感，訓練誤差看起來就越漂亮，要加回去的懲罰就越多。

**Plain explanation**: A teacher grading the exam they wrote cannot help being lenient, and how lenient depends on whether the grade changes when the student's answer changes. The more sensitive the grade is to the answer, the bigger the discount to take off. A model is the same: the more sensitive it is to its training data, the better the training error looks and the more penalty has to be added back.

**正式定義**：$z\sim\mathcal N(\mu,\sigma^2)$：$\mathbb E\big((z-\mu)g(z)\big)=\sigma^2\,\mathbb E\big(g'(z)\big)$。訓練點：$\mathbb E\big((\hat f_0-y_0)^2\big)=\mathbb E\big((\hat f_0-f_0)^2\big)+\sigma^2-2\sigma^2\,\mathbb E(\partial\hat f_0/\partial y_0)$。加總：$\mathbf{Err}=\mathbf{err}-n_{tr}\sigma^2+2\sigma^2\sum_{i=1}^{n_{tr}}\partial\hat f_i/\partial y_i$。

**Formal definition**: for $z\sim\mathcal N(\mu,\sigma^2)$: $\mathbb E\big((z-\mu)g(z)\big)=\sigma^2\,\mathbb E\big(g'(z)\big)$. Training point: $\mathbb E\big((\hat f_0-y_0)^2\big)=\mathbb E\big((\hat f_0-f_0)^2\big)+\sigma^2-2\sigma^2\,\mathbb E(\partial\hat f_0/\partial y_0)$. Summed: $\mathbf{Err}=\mathbf{err}-n_{tr}\sigma^2+2\sigma^2\sum_{i=1}^{n_{tr}}\partial\hat f_i/\partial y_i$.

**例子**：$g(z)=cz$：兩邊都是 $c\sigma^2$；$g(z)=z^3$：兩邊都是 $3\sigma^4$。SURE：$n_{tr}=30$、$\mathbf{err}=90$、$\sigma^2=1$：敏感度 $0.5$ 得 $\mathbf{Err}=90-30+30=90$（每點 $3$）；敏感度 $0.8$ 得 $90-30+48=108$（每點 $3.6$）。逐步計算見第 2、4 節。

**Example**: $g(z)=cz$: both sides equal $c\sigma^2$; $g(z)=z^3$: both sides equal $3\sigma^4$. SURE: $n_{tr}=30$, $\mathbf{err}=90$, $\sigma^2=1$: sensitivity $0.5$ gives $\mathbf{Err}=90-30+30=90$ ($3$ per point); sensitivity $0.8$ gives $90-30+48=108$ ($3.6$ per point). The computations are in sections 2 and 4.

**常見迷思**：

- 「Stein 引理對任何雜訊都成立」——只對高斯雜訊；證明用到高斯密度的導數形式（第 2 節）。
- 「SURE 可以取代驗證集」——它是理論估計，需要高斯雜訊、可微的學習器與已知的 $\sigma^2$；實務上仍要用驗證集或交叉驗證。
- 「敏感度越大越好」——敏感度大是過擬合的徵兆：訓練誤差看起來好，是因為模型在跟著雜訊走。
- 「訓練誤差小，模型就學得好」——訓練誤差小也可能只是敏感度高；式 (SURE) 的第三項把它拉回現實。

**Common misconceptions**:

- "Stein's lemma holds for any noise." Only for Gaussian noise; the proof uses the derivative form of the Gaussian density (section 2).
- "SURE can replace a validation set." It is a theoretical estimate and needs Gaussian noise, a differentiable learner and a known $\sigma^2$; in practice one still uses a validation set or cross-validation.
- "The larger the sensitivity, the better." A large sensitivity is a sign of overfitting: the training error looks good because the model follows the noise.
- "A small training error means the model has learned well." It may only mean a high sensitivity; the third term of SURE pulls it back to reality.

**圖**：本單元兩張圖，`lec04-kp06-stein`（用 $g=\tanh$ 驗證 Stein 引理，功能 1）與 `lec04-kp06-sure`（SURE 數值例，功能 2），以 `assets/figures/src/lec04-kp06-figures.py` 繪製。推導的流程在簡報中以原生節點與連線呈現。

**Figure**: Two figures in this unit, `lec04-kp06-stein` (Stein's lemma checked with $g=\tanh$; function 1) and `lec04-kp06-sure` (the numerical SURE example; function 2), drawn by `assets/figures/src/lec04-kp06-figures.py`. The flow of the derivation is drawn with native nodes and connectors in the deck.

**想深入**（選讀）：多維版本可以一次推出加總式：令 $\mathbf z=\boldsymbol\varepsilon$、$g_i=\hat f_i-f_i$，直接得到 $\mathbb E\big(\boldsymbol\varepsilon^\top(\hat{\mathbf f}-\mathbf f)\big)=\sigma^2\,\mathbb E\big(\sum_i\partial\hat f_i/\partial y_i\big)$；這個和 $\sum_i\partial\hat f_i/\partial y_i$ 叫做 $\hat{\mathbf f}$ 的**散度**（divergence），對線性學習器它就是帽子矩陣的跡（KP07）。

**Going deeper** (optional): The multivariate version gives the summed formula in one step: with $\mathbf z=\boldsymbol\varepsilon$ and $g_i=\hat f_i-f_i$ we get directly $\mathbb E\big(\boldsymbol\varepsilon^\top(\hat{\mathbf f}-\mathbf f)\big)=\sigma^2\,\mathbb E\big(\sum_i\partial\hat f_i/\partial y_i\big)$; the sum $\sum_i\partial\hat f_i/\partial y_i$ is called the **divergence** of $\hat{\mathbf f}$, and for a linear learner it is the trace of the hat matrix (KP07).

**小練習**：

1. $\mu=0$、$\sigma=2$、$g(z)=z^3$。分別算 Stein 引理的左邊與右邊，確認相等。（提示：常態分布的四次動差是 $3\sigma^4$，$g'(z)=3z^2$。）
2. $n_{tr}=50$，每點平均訓練 MSE 為 $2$，$\sigma^2=0.5$，平均敏感度為 $0.6$。求 $\mathbf{Err}$ 與每點平均，並說明訓練 MSE 低估了還是高估了真實誤差。（提示：先算 $\mathbf{err}=50\times2$，再依序代入三項。）
3. $n_{tr}=40$、$\sigma^2=2$。總敏感度 $\sum_i\partial\hat f_i/\partial y_i$ 是多少時，$\mathbf{Err}=\mathbf{err}$？若總敏感度只有 $10$，$\mathbf{Err}$ 比 $\mathbf{err}$ 大還是小？（提示：令 $-n_{tr}\sigma^2+2\sigma^2\sum_i\partial\hat f_i/\partial y_i=0$。）

**Practice**:

1. $\mu=0$, $\sigma=2$, $g(z)=z^3$. Compute the left and right sides of Stein's lemma and confirm they are equal. (Hint: the fourth moment of a normal variable is $3\sigma^4$ and $g'(z)=3z^2$.)
2. $n_{tr}=50$, the mean training MSE per point is $2$, $\sigma^2=0.5$ and the mean sensitivity is $0.6$. Find $\mathbf{Err}$ and its per-point mean, and say whether the training MSE underestimates or overestimates the true error. (Hint: first $\mathbf{err}=50\times2$, then substitute the three terms in turn.)
3. $n_{tr}=40$ and $\sigma^2=2$. For which total sensitivity $\sum_i\partial\hat f_i/\partial y_i$ is $\mathbf{Err}=\mathbf{err}$? If the total sensitivity is only $10$, is $\mathbf{Err}$ larger or smaller than $\mathbf{err}$? (Hint: set $-n_{tr}\sigma^2+2\sigma^2\sum_i\partial\hat f_i/\partial y_i=0$.)

**解題示範**（以 $n_{tr}=20$、每點平均訓練 MSE $4$、$\sigma^2=1$、平均敏感度 $0.7$ 示範第 2 題的算法）：

1. 訓練誤差：$\mathbf{err}=20\times4=80$。
2. 雜訊地板：$n_{tr}\sigma^2=20\times1=20$。
3. 複雜度懲罰：$2\sigma^2\sum_i\partial\hat f_i/\partial y_i=2\cdot1\cdot(20\times0.7)=28$。
4. 真實誤差：$\mathbf{Err}=80-20+28=88$；每點 $88/20=4.4$，比訓練 MSE $4$ 大，所以訓練 MSE 低估了真實誤差。

**Worked solution** (the method of exercise 2, shown with $n_{tr}=20$, a mean training MSE of $4$, $\sigma^2=1$ and a mean sensitivity of $0.7$):

1. Training error: $\mathbf{err}=20\times4=80$.
2. Noise floor: $n_{tr}\sigma^2=20\times1=20$.
3. Complexity penalty: $2\sigma^2\sum_i\partial\hat f_i/\partial y_i=2\cdot1\cdot(20\times0.7)=28$.
4. True error: $\mathbf{Err}=80-20+28=88$; per point $88/20=4.4$, larger than the training MSE $4$, so the training MSE underestimates the true error.

### 來源更正與取捨（Corrections to the source）

- 來源說對 $n_{tr}$ 個訓練實例「取平均」後得到式 (28)，但式子本身是**加總**（左邊的 $\mathbf{Err}$、$\mathbf{err}$ 與 $\sum_i$ 都是總和）。本講義明確寫成相加。
- 來源在敏感度 $0.8$ 的例子說真實誤差「被高估」到 $3.6$。依式子，$\mathbf{Err}=108>\mathbf{err}=90$：真實誤差比訓練誤差大，是**訓練誤差低估**了真實誤差。本講義改寫為此，並補上臨界點（平均敏感度 $\tfrac12$）。
- 來源把引理與 (26)、(28) 的期望符號有時省略。本講義保留期望：等式是對雜訊 $\varepsilon$ 取期望（固定 $f$），單一資料集只是近似。
- 來源只列出引理的高斯前提的一句話。本講義補上一維證明、$g=z^3$ 的手算例、電腦驗證，以及可微與增長條件，都標為 `[補充]` 或〔程式計算〕。
- 來源說「SURE 可以估計真實誤差而不用驗證集」。本講義保留這個說法但加上限制：需要高斯雜訊、可微且確定的學習器、已知的 $\sigma^2$，且不取代驗證。

- The source says formula (28) is obtained by "averaging" over the $n_{tr}$ training points, but the formula itself is a **sum** ($\mathbf{Err}$, $\mathbf{err}$ and $\sum_i$ are all totals). This note writes it explicitly as adding up.
- For the example with sensitivity $0.8$ the source says the true error is "overestimated" to $3.6$. By the formula, $\mathbf{Err}=108>\mathbf{err}=90$: the true error is larger than the training error, so the **training error underestimates** the true error. This note rewrites it that way and adds the break-even point (mean sensitivity $\tfrac12$).
- The source sometimes drops the expectation symbols in the lemma and in (26) and (28). This note keeps the expectation: the identities are in expectation over the noise $\varepsilon$ (with $f$ fixed), and on a single data set they are only approximate.
- The source states the Gaussian premise of the lemma in a single sentence. This note adds the one-dimensional proof, the hand computation for $g=z^3$, the computer check, and the differentiability and growth conditions, marked `[Added]` or [computed by program].
- The source says "SURE lets us estimate the true error without a validation set". This note keeps the statement but adds the limits: Gaussian noise, a differentiable and deterministic learner, a known $\sigma^2$, and no replacement for validation.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| Stein 引理 | Stein's lemma |
| Stein 無偏風險估計 | Stein's unbiased risk estimate (SURE) |
| 敏感度 | sensitivity |
| 散度 | divergence |
| 複雜度懲罰 | complexity penalty |
| 高斯雜訊 | Gaussian noise |
| 分部積分 | integration by parts |
| 訓練誤差 | training error |
| 真實誤差 | true error |
