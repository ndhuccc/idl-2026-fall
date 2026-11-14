# 第13週 KP03：最優判別器與 JS 散度，判別器做到最好時，生成器在縮小什麼（The optimal discriminator and the JS divergence: what the generator shrinks when the discriminator is at its best）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第13週的第三個知識點（KP03），接在 KP02 之後。KP02 把 GAN 寫成 $\min_G\max_D V(D,G)$：對每個生成器，判別器先把 $V$ 推到最大。這一節回答 KP02 留下的問題：這個最大值長什麼樣子？判別器做到最好時，它的評分等於什麼？這時生成器真正在縮小的，是哪一種距離？

This is the third knowledge point (KP03) of Week 13, following KP02. KP02 wrote the GAN as $\min_G\max_D V(D,G)$: for each generator, the discriminator first pushes $V$ to its maximum. This note answers the question KP02 left: what does that maximum look like? When the discriminator does its best, what does its score equal? And what kind of distance is the generator really shrinking then?

先備知識是 KP02 的目標函數 $V$、第12週 KP02 的 KL 散度，以及微積分的「導數設為零求極值」。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第14章 KP-4 改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec13-kp03-figures.py` 算出，可以自己重算。

The prerequisites are KP02's objective $V$, the KL divergence of Week 12 KP02, and the calculus idea of setting a derivative to zero to find an extremum. Every term and symbol is defined when it first appears. This note is rewritten from KP-4 of Chapter 14 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec13-kp03-figures.py`, so you can recompute it yourself.

## KP03 最優判別器與 JS 散度（The optimal discriminator and the JS divergence）

### 學習目標（Learning objectives）

讀完之後，你能用「每一點各自求極大」推出最優判別器 $D^*(\mathbf{x})=p_{\text{data}}(\mathbf{x})/\big(p_{\text{data}}(\mathbf{x})+p_g(\mathbf{x})\big)$，並在小例子裡算出它的值。你能把 $D^*$ 代回 $V$，一步一步推出 $V(D^*,G)=-\log4+2\,\mathrm{JSD}(p_{\text{data}}\,\Vert\,p_g)$，並說明最小值 $-\log4$ 在什麼時候出現。你也能看懂 JS 散度在兩個分布離很遠時會變平，並說出這對訓練的影響，以及這個結論成立需要哪些假設。

After reading, you can derive the optimal discriminator $D^*(\mathbf{x})=p_{\text{data}}(\mathbf{x})/\big(p_{\text{data}}(\mathbf{x})+p_g(\mathbf{x})\big)$ by maximising at each point separately, and compute its value in a small example. You can substitute $D^*$ back into $V$, derive step by step that $V(D^*,G)=-\log4+2\,\mathrm{JSD}(p_{\text{data}}\,\Vert\,p_g)$, and say when the minimum $-\log4$ is reached. You can also see that the JS divergence goes flat when the two distributions are far apart, explain what that means for training, and state which assumptions the result needs.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $p_{\text{data}}(\mathbf{x}),\ p_g(\mathbf{x})$ | 真資料與生成資料在 $\mathbf{x}$ 的機率密度 | probability densities of real and generated data at $\mathbf{x}$ |
| $V(D,G)$ | KP02 的目標函數 | the objective of KP02 |
| $D^*(\mathbf{x})$ | 給定生成器時，最好的判別器 | the best discriminator for a given generator |
| $a,\ b,\ y$ | 第1節的簡寫：$a=p_{\text{data}}(\mathbf{x})$、$b=p_g(\mathbf{x})$、$y=D(\mathbf{x})$ | shorthand of Section 1: $a=p_{\text{data}}(\mathbf{x})$, $b=p_g(\mathbf{x})$, $y=D(\mathbf{x})$ |
| $m(\mathbf{x})$ | 兩個分布的平均 $\tfrac12\big(p_{\text{data}}+p_g\big)$ | the average of the two distributions $\tfrac12\big(p_{\text{data}}+p_g\big)$ |
| $D_{\text{KL}}(P\,\Vert\,Q)$ | KL 散度（第12週 KP02） | KL divergence (Week 12 KP02) |
| $\mathrm{JSD}(P\,\Vert\,Q)$ | JS 散度 | Jensen–Shannon divergence |
| $\log$ | 自然對數 | natural logarithm |

### 1. 固定生成器，判別器的最好答案（With the generator fixed, the discriminator's best answer）

先固定生成器，它的輸出有一個分布 $p_g$。KP02 的 $V$ 有兩個期望值：第一個在真資料上取平均，第二個在生成資料上取平均。把期望值寫成積分（加總每一個可能的 $\mathbf{x}$），兩項就能合在同一個積分裡：

First fix the generator; its outputs have a distribution $p_g$. KP02's $V$ has two expectations: the first averages over real data, the second over generated data. Writing each expectation as an integral (a sum over every possible $\mathbf{x}$), the two terms fit into one integral:

$$
V(D,G)=\int\Big(p_{\text{data}}(\mathbf{x})\log D(\mathbf{x})+p_g(\mathbf{x})\log\big(1-D(\mathbf{x})\big)\Big)\,d\mathbf{x}.
$$

判別器可以在每一個 $\mathbf{x}$ 各自選一個值 $D(\mathbf{x})$，彼此不影響。所以讓整個積分最大，就是在每一點讓括號裡最大。來源把這一點的問題寫成一個一元函數：記 $a=p_{\text{data}}(\mathbf{x})$、$b=p_g(\mathbf{x})$、$y=D(\mathbf{x})$，

The discriminator may choose a value $D(\mathbf{x})$ at each $\mathbf{x}$ separately, with no effect on the others. So maximising the whole integral means maximising the bracket at each point. The source writes the problem at one point as a function of one variable: with $a=p_{\text{data}}(\mathbf{x})$, $b=p_g(\mathbf{x})$ and $y=D(\mathbf{x})$,

$$
f(y)=a\log y+b\log(1-y),\qquad f'(y)=\frac{a}{y}-\frac{b}{1-y}=0\ \Longrightarrow\ y^*=\frac{a}{a+b}.
$$

$f$ 是一條往下彎的曲線（兩個 $\log$ 都是凹函數，乘上不是負數的 $a$、$b$ 再相加仍然是凹的），所以導數為零的地方就是最大值 `[補充：理由]`。若 $b=0$（這裡沒有生成資料），導數永遠大於 $0$，最大值在邊界 $y=1$，公式 $a/(a+b)$ 同樣給出 $1$；若 $a=b=0$，這一點不影響 $V$，評分給多少都可以。換回原來的符號，得到**最優判別器**（optimal discriminator）：

$f$ is a curve that bends downward (both logarithms are concave, and multiplying them by the non-negative $a$ and $b$ and adding keeps the result concave), so the point where the derivative is zero is the maximum `[Added: reason]`. If $b=0$ (no generated data here), the derivative is always positive and the maximum is at the boundary $y=1$, which the formula $a/(a+b)$ also gives; if $a=b=0$, this point does not affect $V$ and any score will do. Back in the original symbols, this gives the **optimal discriminator**:

$$
D^*(\mathbf{x})=\frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x})+p_g(\mathbf{x})}.
$$

代入小數字 `[補充]`：某一點 $a=0.6$、$b=0.2$，最好的評分是 $y^*=0.6/0.8=0.75$，此時 $f=0.6\log0.75+0.2\log0.25=-0.450$。若判別器給 $0.5$，$f=-0.555$；給 $0.9$，$f=-0.524$，都比較小。若某一點真假的密度一樣，例如 $a=b=0.3$，最好的評分就是 $0.5$。

Plug in small numbers `[Added]`: at a point with $a=0.6$ and $b=0.2$, the best score is $y^*=0.6/0.8=0.75$, giving $f=0.6\log0.75+0.2\log0.25=-0.450$. Scoring $0.5$ gives $f=-0.555$ and scoring $0.9$ gives $f=-0.524$, both smaller. If real and fake have equal density at a point, such as $a=b=0.3$, the best score is $0.5$.

![一張曲線圖，橫軸是 y=D(x)，從 0 到 1；縱軸是 a log y + b log(1-y)。藍線 a=0.6、b=0.2，在 y=0.75 有最高點，標著 y*=0.75；橘線 a=0.3、b=0.3，左右對稱，在 y=0.5 有最高點，標著 y*=0.50。兩條線在兩端都往下急降（A line chart with y=D(x) from 0 to 1 on the horizontal axis and a log y + b log(1-y) on the vertical axis. The blue line, a=0.6 and b=0.2, peaks at y=0.75, labelled y*=0.75; the orange line, a=0.3 and b=0.3, is symmetric and peaks at y=0.5, labelled y*=0.50. Both lines drop sharply at both ends）](../assets/figures/lec13-kp03-pointwise.png)

**圖說 `lec13-kp03-pointwise`**〔功能：一目了然、精準度〕：先看藍線：這一點的真資料密度是假資料的 $3$ 倍，最高點就落在 $0.75$，正好是 $a/(a+b)$。再看橘線：真假一樣多，最高點在中間的 $0.5$。兩端急降，是因為評分接近 $0$ 或 $1$ 時，$\log$ 會掉到負無窮。

**Caption `lec13-kp03-pointwise`** [Function: seeing it at a glance, precision]: first look at the blue line: here the real density is $3$ times the fake density, and the peak sits at $0.75$, exactly $a/(a+b)$. Then look at the orange line: real and fake are equally common, and the peak is in the middle at $0.5$. Both ends drop sharply because the logarithm falls to minus infinity as the score nears $0$ or $1$.

常見錯誤：以為最優判別器在真資料上一律給 $1$、在假資料上一律給 $0$。真假的分布常常重疊；在重疊的地方，同一個 $\mathbf{x}$ 可能是真的也可能是假的，最好的評分只能是「真的佔幾成」。

Common mistake: thinking the optimal discriminator gives $1$ to every real example and $0$ to every fake. The real and fake distributions often overlap; where they do, the same $\mathbf{x}$ may be real or fake, and the best score can only be "what fraction is real".

### 2. 小例子：最優判別器長什麼樣子（A small example: what the optimal discriminator looks like）

回到 KP01 的一維例子 `[補充]`：真資料 $p_{\text{data}}=\mathcal{N}(2,\ 0.5^2)$，生成器剛開始時 $p_g=\mathcal{N}(0,1)$。兩個都是常態分布，$D^*$ 的 logit 就是兩個密度相除再取 $\log$：

Back to KP01's one-dimensional example `[Added]`: the real data is $p_{\text{data}}=\mathcal{N}(2,\ 0.5^2)$, and at the start the generator has $p_g=\mathcal{N}(0,1)$. Both are normal, so the logit of $D^*$ is the logarithm of the ratio of the two densities:

$$
\log\frac{p_{\text{data}}(x)}{p_g(x)}=-2(x-2)^2+\log2+\frac{x^2}{2}=-1.5x^2+8x-7.307 .
$$

這裡 $\log2$ 來自兩個標準差的比 $1/0.5$。因為 $D^*=\dfrac{a}{a+b}=\dfrac{1}{1+b/a}=\sigma\big(\log(a/b)\big)$（$\sigma$ 與 logit 見 KP01），也就是 $D^*=\sigma\big(\log(p_{\text{data}}/p_g)\big)$，可以直接算：$D^*(0)=0.0007$，$D^*(1)=0.309$，$D^*(2)=0.937$，$D^*(3)=0.961$。

Here $\log2$ comes from the ratio $1/0.5$ of the two standard deviations. Since $D^*=\dfrac{a}{a+b}=\dfrac{1}{1+b/a}=\sigma\big(\log(a/b)\big)$ ($\sigma$ and the logit are from KP01), that is, $D^*=\sigma\big(\log(p_{\text{data}}/p_g)\big)$, we can compute directly: $D^*(0)=0.0007$, $D^*(1)=0.309$, $D^*(2)=0.937$ and $D^*(3)=0.961$.

這個 logit 是 $x$ 的二次式，正好是 KP01 小判別器 $\sigma(w_0+w_1x+w_2x^2)$ 能表示的形狀。實際用兩群各 $200000$ 個樣本（種子 $3$）訓練一個這樣的邏輯斯迴歸，學到的權重是 $(-7.308,\ 8.002,\ -1.504)$，幾乎就是 $(-7.307,\ 8,\ -1.5)$。在這個例子裡，判別器的形狀剛好夠用、樣本也夠多，訓練出來的就是 $D^*$；一般的網路要做到這點，也需要足夠的容量與資料。

This logit is quadratic in $x$, exactly the shape KP01's small discriminator $\sigma(w_0+w_1x+w_2x^2)$ can represent. Training such a logistic regression on two groups of $200000$ samples each (seed $3$) gives weights $(-7.308,\ 8.002,\ -1.504)$, almost exactly $(-7.307,\ 8,\ -1.5)$. In this example the discriminator's shape is exactly flexible enough and there are plenty of samples, so training recovers $D^*$; a general network needs enough capacity and data to do the same.

![一張圖，橫軸 x 從 -3 到 4.5。淺藍填色是真資料分布，中心在 2；橘線是生成分布，中心在 0、比較寬。紫色實線是最優判別器，對應右邊 0 到 1 的刻度：左邊接近 0，在 1 附近急升，在 2 到 3 之間接近 0.95，到 4 之後又往下掉；粉紅虛線是用樣本訓練的邏輯斯迴歸，和紫線完全重疊；灰色點線是 0.5（A chart with x from -3 to 4.5. The light-blue fill is the real distribution centred at 2; the orange line is the generated distribution centred at 0 and wider. The solid purple line is the optimal discriminator on the right-hand scale from 0 to 1: near 0 on the left, rising sharply around 1, close to 0.95 between 2 and 3, and dropping again beyond 4; the dashed pink line, a logistic regression trained on samples, lies exactly on the purple line; the grey dotted line is 0.5）](../assets/figures/lec13-kp03-optimal-d.png)

**圖說 `lec13-kp03-optimal-d`**〔功能：精準度、空間幾何具象化〕：先看兩個分布：藍色的真資料在 $2$，橘色的生成資料在 $0$。再看紫線：哪裡藍色多，$D^*$ 就高；哪裡橘色多，$D^*$ 就低；兩者交界處穿過 $0.5$。最右邊紫線又往下掉，是因為橘色比較寬，那裡的尾巴比藍色厚；兩個密度在那裡都很小，看不出來，但決定 $D^*$ 的是它們的比值。灰色點線是 $D=0.5$。粉紅虛線和紫線疊在一起，表示用樣本訓練就能學到 $D^*$。

**Caption `lec13-kp03-optimal-d`** [Function: precision, making space concrete]: first look at the two distributions: blue real data at $2$ and orange generated data at $0$. Then look at the purple line: where blue dominates, $D^*$ is high; where orange dominates, it is low; it crosses $0.5$ where they meet. At the far right the purple line drops again because the orange distribution is wider and its tail is thicker there than the blue one; both densities are too small to see there, but what decides $D^*$ is their ratio. The grey dotted line is $D=0.5$. The dashed pink line lies on the purple one, showing that training on samples learns $D^*$.

$D^*$ 還有一個用處 `[補充]`：$D^*/(1-D^*)=p_{\text{data}}/p_g$。判別器其實在估計兩個密度的比值，這個想法在 KP08（選讀）會再出現。

$D^*$ has one more use `[Added]`: $D^*/(1-D^*)=p_{\text{data}}/p_g$. The discriminator is really estimating the ratio of the two densities, an idea that returns in KP08 (optional).

常見錯誤：以為 $D^*$ 只要把生成資料所在的地方判低就好，不必理會真資料有多密。$D^*$ 同時取決於兩個密度：同一個 $\mathbf{x}$，真資料越密、生成資料越疏，評分才越高。

Common mistake: thinking $D^*$ only needs to score low wherever generated data lies, ignoring how dense the real data is. $D^*$ depends on both densities: at the same $\mathbf{x}$, the denser the real data and the sparser the generated data, the higher the score.

### 3. 把 $D^*$ 代回去：生成器在縮小 JS 散度（Substituting $D^*$ back: the generator shrinks the JS divergence）

判別器做到最好之後，剩下的 $V(D^*,G)$ 只和生成器有關。把 $D^*$ 代回 KP02 的 $V$：

Once the discriminator is at its best, the remaining $V(D^*,G)$ depends only on the generator. Substitute $D^*$ into KP02's $V$:

$$
V(D^*,G)=\mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}\Big[\log\frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x})+p_g(\mathbf{x})}\Big]+\mathbb{E}_{\mathbf{x}\sim p_g}\Big[\log\frac{p_g(\mathbf{x})}{p_{\text{data}}(\mathbf{x})+p_g(\mathbf{x})}\Big].
$$

來源接著用 KL 散度與 JS 散度整理。把每個分母寫成 $2\,m(\mathbf{x})$，其中 $m=\tfrac12(p_{\text{data}}+p_g)$ 是兩個分布的平均 `[補充：推導步驟]`。每一項的 $\log$ 就拆成「對 $m$ 的比值」減去 $\log2$：

The source then rewrites it with the KL and JS divergences. Write each denominator as $2\,m(\mathbf{x})$, where $m=\tfrac12(p_{\text{data}}+p_g)$ is the average of the two distributions `[Added: derivation steps]`. The logarithm in each term then splits into "the ratio to $m$" minus $\log2$:

$$
V(D^*,G)=\underbrace{\mathbb{E}_{p_{\text{data}}}\Big[\log\frac{p_{\text{data}}}{m}\Big]}_{D_{\text{KL}}(p_{\text{data}}\,\Vert\,m)}-\log2+\underbrace{\mathbb{E}_{p_g}\Big[\log\frac{p_g}{m}\Big]}_{D_{\text{KL}}(p_g\,\Vert\,m)}-\log2 .
$$

兩個 $-\log2$ 合起來是 $-\log4$。兩個 KL 散度（第12週 KP02：衡量一個分布離另一個多遠，永遠 $\ge0$）的平均，就是 **JS 散度**（Jensen–Shannon divergence）：

The two $-\log2$ terms add up to $-\log4$. The average of the two KL divergences (Week 12 KP02: how far one distribution is from another, always $\ge0$) is the **Jensen–Shannon divergence**:

$$
\mathrm{JSD}(p_{\text{data}}\,\Vert\,p_g)=\tfrac12D_{\text{KL}}(p_{\text{data}}\,\Vert\,m)+\tfrac12D_{\text{KL}}(p_g\,\Vert\,m),\qquad
V(D^*,G)=-\log4+2\,\mathrm{JSD}(p_{\text{data}}\,\Vert\,p_g).
$$

JS 散度有三個好性質 `[補充]`：它對兩個分布是對稱的（交換位置不變）；它永遠在 $0$ 到 $\log2$ 之間；它等於 $0$ 若且唯若兩個分布一樣。所以 $V(D^*,G)$ 介於 $-\log4\approx-1.386$ 與 $0$ 之間，而且只有 $p_g=p_{\text{data}}$ 時才達到最小值 $-\log4$；這時 $D^*=\tfrac12$ 處處成立，$V=\log\tfrac12+\log\tfrac12=-\log4$，正是 KP01 的終點。

The JS divergence has three good properties `[Added]`: it is symmetric in the two distributions (swapping them changes nothing); it always lies between $0$ and $\log2$; and it is $0$ if and only if the two distributions are equal. So $V(D^*,G)$ lies between $-\log4\approx-1.386$ and $0$, and reaches its minimum $-\log4$ only when $p_g=p_{\text{data}}$; then $D^*=\tfrac12$ everywhere and $V=\log\tfrac12+\log\tfrac12=-\log4$, exactly KP01's end point.

用第2節的例子核對 `[補充]`：$p_{\text{data}}=\mathcal{N}(2,\ 0.5^2)$、$p_g=\mathcal{N}(0,1)$ 時，數值積分得到 $\mathrm{JSD}=0.4743$，所以 $V(D^*,G)=-1.3863+2\times0.4743=-0.4377$；直接把 $D^*$ 代進 $V$ 積分，也是 $-0.4377$。

Check with Section 2's example `[Added]`: with $p_{\text{data}}=\mathcal{N}(2,\ 0.5^2)$ and $p_g=\mathcal{N}(0,1)$, numerical integration gives $\mathrm{JSD}=0.4743$, so $V(D^*,G)=-1.3863+2\times0.4743=-0.4377$; integrating $V$ directly with $D^*$ plugged in also gives $-0.4377$.

結論是：在判別器每次都做到最好的前提下，生成器最小化 $V$，就等於最小化真資料與生成資料之間的 JS 散度。

The conclusion: provided the discriminator is at its best every time, the generator minimising $V$ is the same as minimising the JS divergence between the real and generated distributions.

常見錯誤：以為 GAN 和第12週一樣在最大化概似，也就是最小化 $D_{\text{KL}}(p_{\text{data}}\,\Vert\,p_g)$。這裡出現的是 JS 散度，不是那個 KL 散度；兩者的差別會影響生成器的行為，KP08（選讀）再比較。

Common mistake: thinking a GAN maximises likelihood as in Week 12, that is, minimises $D_{\text{KL}}(p_{\text{data}}\,\Vert\,p_g)$. What appears here is the JS divergence, not that KL divergence; the difference affects how the generator behaves, compared in KP08 (optional).

### 4. 兩個分布離很遠時，JS 散度變平（When the two distributions are far apart, the JS divergence goes flat）

JS 散度有上限 $\log2$，這件事有代價 `[補充]`。讓生成器的分布 $p_g=\mathcal{N}(a,\ 0.5^2)$ 和真資料一樣寬，只移動中心 $a$，看 JS 散度怎麼變：

The JS divergence has an upper limit of $\log2$, and that comes at a price `[Added]`. Let the generator's distribution $p_g=\mathcal{N}(a,\ 0.5^2)$ be as wide as the real data, move only its centre $a$, and watch the JS divergence:

| 生成器的中心 $a$ | $2$ | $1.5$ | $1$ | $0$ | $-1$ | $-3$ |
|---|---|---|---|---|---|---|
| JS 散度 | $0$ | $0.111$ | $0.337$ | $0.633$ | $0.689$ | $0.693$ |
| $V(D^*,G)$ | $-1.386$ | $-1.164$ | $-0.713$ | $-0.121$ | $-0.008$ | $0.000$ |

| Generator centre $a$ | $2$ | $1.5$ | $1$ | $0$ | $-1$ | $-3$ |
|---|---|---|---|---|---|---|
| JS divergence | $0$ | $0.111$ | $0.337$ | $0.633$ | $0.689$ | $0.693$ |
| $V(D^*,G)$ | $-1.386$ | $-1.164$ | $-0.713$ | $-0.121$ | $-0.008$ | $0.000$ |

中心從 $2$ 移到 $0$，JS 散度一路升高；可是從 $-1$ 到 $-3$，它只從 $0.689$ 變到 $0.693$，幾乎不動。兩個分布幾乎不重疊時，不管再遠，JS 散度都只是越來越接近 $\log2$。常態分布永遠有一點重疊，所以只是「接近」；若兩個分布真的完全不重疊（例如資料落在很低維的形狀上），JS 散度就正好是 $\log2$，梯度正好是 $0$ `[補充]`。

As the centre moves from $2$ to $0$, the JS divergence keeps rising; but from $-1$ to $-3$ it changes only from $0.689$ to $0.693$, barely moving. Once the two distributions hardly overlap, the JS divergence only creeps closer to $\log2$ however far apart they are. Normal distributions always overlap a little, so it is only "close"; if two distributions truly do not overlap (for example when data lies on a very low-dimensional shape), the JS divergence is exactly $\log2$ and the gradient exactly $0$ `[Added]`.

![一張曲線圖，橫軸是生成器的中心 a，從 -4 到 8；縱軸是 JS 散度，從 0 到 0.7。青色曲線在 a=2 降到 0，兩旁升高，到 a 小於約 -0.5 或大於約 4.5 時就貼在紅色虛線 log 2=0.693 上，這兩段塗淡黃色，各寫著幾乎不重疊、近乎平、梯度接近零。曲線上標出 a=2、1、0、-1 四個點，數值 0.000、0.337、0.633、0.689（A line chart with the generator centre a from -4 to 8 on the horizontal axis and the JS divergence from 0 to 0.7 on the vertical axis. The teal curve falls to 0 at a=2 and rises on both sides; for a below about -0.5 or above about 4.5 it lies on the dashed red line log 2=0.693, and these two regions are shaded pale yellow, each labelled almost no overlap, nearly flat, near-zero gradient. Four points are marked at a=2, 1, 0 and -1, with values 0.000, 0.337, 0.633 and 0.689）](../assets/figures/lec13-kp03-jsd-shift.png)

**圖說 `lec13-kp03-jsd-shift`**〔功能：一目了然、精準度〕：先看中間的谷：生成器的中心越接近 $2$，JS 散度越小，谷底是 $0$。再看兩旁淡黃色的區域：曲線貼在 $\log2$ 上，幾乎是平的；生成器在這裡往哪邊移，JS 散度都不變，也就拿不到該往哪走的梯度。

**Caption `lec13-kp03-jsd-shift`** [Function: seeing it at a glance, precision]: first look at the valley in the middle: the closer the generator's centre is to $2$, the smaller the JS divergence, with $0$ at the bottom. Then look at the pale-yellow regions on both sides: the curve lies on $\log2$ and is almost flat; whichever way the generator moves there, the JS divergence does not change, so it gets no gradient telling it where to go.

這解釋了一個常見的訓練困難：訓練初期生成器和真資料離很遠，若判別器又做得很好，生成器幾乎沒有梯度可用。KP02 的非飽和損失是一種補救；KP07 的 Wasserstein 距離則是換一把在遠處仍有坡度的尺。

This explains a common training difficulty: early on the generator is far from the real data, and if the discriminator is very good, the generator has almost no gradient to use. KP02's non-saturating loss is one remedy; KP07's Wasserstein distance replaces the ruler with one that still has a slope far away.

這個結論還有幾個前提，讀的時候要記得 `[補充]`。第一，判別器每一步都做到最好，而且能在每一點自由選評分；實際上它只是一個形狀有限的網路，只訓練了有限步。第二，推導把生成器看成可以任意改變的分布 $p_g$；實際上生成器只能在自己的權重能表示的範圍裡移動。第三，若生成器改用 KP02 的非飽和損失，它最小化的就不再嚴格是 JS 散度，不過終點仍然是 $p_g=p_{\text{data}}$。

The result also rests on some conditions to keep in mind `[Added]`. First, the discriminator is at its best at every step and can choose its score freely at each point; in practice it is only a network of limited shape, trained for a finite number of steps. Second, the derivation treats the generator as a distribution $p_g$ that can change freely; in practice the generator moves only within what its weights can represent. Third, if the generator uses KP02's non-saturating loss, what it minimises is no longer strictly the JS divergence, though the end point is still $p_g=p_{\text{data}}$.

常見錯誤：以為 JS 散度越接近 $\log2$，生成器就越知道要大幅修改。剛好相反：在平台上 JS 散度幾乎不變，梯度接近 $0$，生成器反而不知道往哪裡改。

Common mistake: thinking that the closer the JS divergence is to $\log2$, the more surely the generator knows to change a lot. It is the opposite: on the plateau the JS divergence barely changes, the gradient is near $0$, and the generator does not know which way to change.

### 本單元重點整理（KP03 summary）

**一句話重點**：固定生成器時，最好的判別器是 $D^*=p_{\text{data}}/(p_{\text{data}}+p_g)$；代回去得到 $V(D^*,G)=-\log4+2\,\mathrm{JSD}(p_{\text{data}}\,\Vert\,p_g)$，所以在判別器最優的前提下，生成器在縮小 JS 散度，最小值 $-\log4$ 出現在 $p_g=p_{\text{data}}$、$D^*=\tfrac12$；但兩個分布離很遠時，JS 散度停在 $\log2$ 不動，生成器拿不到梯度。

**One-sentence summary**: with the generator fixed, the best discriminator is $D^*=p_{\text{data}}/(p_{\text{data}}+p_g)$; substituting back gives $V(D^*,G)=-\log4+2\,\mathrm{JSD}(p_{\text{data}}\,\Vert\,p_g)$, so with an optimal discriminator the generator shrinks the JS divergence, with minimum $-\log4$ at $p_g=p_{\text{data}}$ and $D^*=\tfrac12$; but when the two distributions are far apart the JS divergence sits at $\log2$ and the generator gets no gradient.

**白話說明**：最厲害的鑑定師，看到一幅畫時給的分數，就是「這種畫裡真跡佔幾成」。學徒要做的，是讓真跡與仿作的分布越來越像；鑑定師的分數總結起來，正好量出兩者差多少。可是學徒的仿作如果和真跡完全不像，「差多少」就停在最大值，學徒不知道先改哪裡。

**Plain explanation**: the best possible authenticator, looking at a painting, gives the score "what fraction of paintings like this are genuine". The apprentice's task is to make the copies' distribution ever closer to the genuine one, and the authenticator's scores, added up, measure exactly how different they are. But if the copies look nothing like the genuine works, "how different" is stuck at its maximum, and the apprentice does not know what to fix first.

**正式定義**：$D^*(\mathbf{x})=\dfrac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x})+p_g(\mathbf{x})}$；$\mathrm{JSD}(P\,\Vert\,Q)=\tfrac12D_{\text{KL}}(P\,\Vert\,M)+\tfrac12D_{\text{KL}}(Q\,\Vert\,M)$，$M=\tfrac12(P+Q)$；$\max_DV(D,G)=V(D^*,G)=-\log4+2\,\mathrm{JSD}(p_{\text{data}}\,\Vert\,p_g)$，$0\le\mathrm{JSD}\le\log2$。

**Formal definition**: $D^*(\mathbf{x})=\dfrac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x})+p_g(\mathbf{x})}$; $\mathrm{JSD}(P\,\Vert\,Q)=\tfrac12D_{\text{KL}}(P\,\Vert\,M)+\tfrac12D_{\text{KL}}(Q\,\Vert\,M)$ with $M=\tfrac12(P+Q)$; $\max_DV(D,G)=V(D^*,G)=-\log4+2\,\mathrm{JSD}(p_{\text{data}}\,\Vert\,p_g)$, with $0\le\mathrm{JSD}\le\log2$.

**例子**：$a=0.6$、$b=0.2$ 時 $y^*=0.75$；$p_{\text{data}}=\mathcal{N}(2,0.5^2)$、$p_g=\mathcal{N}(0,1)$ 時 $D^*$ 的 logit 是 $-1.5x^2+8x-7.307$，$D^*(2)=0.937$，邏輯斯迴歸學到 $(-7.308,\ 8.002,\ -1.504)$，$\mathrm{JSD}=0.4743$、$V(D^*,G)=-0.4377$；把生成器中心從 $2$ 移到 $-3$，JS 散度從 $0$ 升到 $0.693$ 後幾乎不再變。

**Example**: with $a=0.6$ and $b=0.2$, $y^*=0.75$; with $p_{\text{data}}=\mathcal{N}(2,0.5^2)$ and $p_g=\mathcal{N}(0,1)$ the logit of $D^*$ is $-1.5x^2+8x-7.307$, $D^*(2)=0.937$, a logistic regression learns $(-7.308,\ 8.002,\ -1.504)$, $\mathrm{JSD}=0.4743$ and $V(D^*,G)=-0.4377$; moving the generator centre from $2$ to $-3$, the JS divergence rises from $0$ to $0.693$ and then barely changes.

**常見迷思**：

- 「最優判別器把真的都判 $1$、假的都判 $0$。」錯，重疊處只能給真的佔幾成。
- 「GAN 在最大化概似。」錯，判別器最優時生成器縮小的是 JS 散度。
- 「JS 散度接近 $\log2$ 時梯度最大。」錯，那裡是平台，梯度接近 $0$。

**Common misconceptions**:

- "The optimal discriminator scores every real example $1$ and every fake $0$." Wrong: where they overlap it can only give the fraction that is real.
- "A GAN maximises likelihood." Wrong: with an optimal discriminator the generator shrinks the JS divergence.
- "The gradient is largest when the JS divergence is near $\log2$." Wrong: that is a plateau, with gradient near $0$.

**圖**：本單元有三張圖。

- `lec13-kp03-pointwise`：每一點各自求極大，最高點在 $a/(a+b)$。
- `lec13-kp03-optimal-d`：一維例子的 $D^*$，以及用樣本訓練出的判別器和它重合。
- `lec13-kp03-jsd-shift`：生成器中心移動時的 JS 散度，兩旁是貼在 $\log2$ 的平台。

**Figures** — this unit has three figures.

- `lec13-kp03-pointwise`: maximising at each point separately, with the peak at $a/(a+b)$.
- `lec13-kp03-optimal-d`: $D^*$ in the one-dimensional example, with a discriminator trained on samples lying on it.
- `lec13-kp03-jsd-shift`: the JS divergence as the generator centre moves, with plateaus at $\log2$ on both sides.

### 小練習（Practice）

1. 某一點真資料密度 $0.3$、生成資料密度 $0.1$。最優判別器在這裡給幾分？若生成資料密度是 $0$ 呢？
2. 兩個分布完全一樣時，$\mathrm{JSD}$ 與 $V(D^*,G)$ 各是多少？兩個分布完全不重疊時呢？
3. 一位同學說：「判別器越強越好，最好每一步都把它訓練到最優。」請用第4節的結果說明這樣做可能有什麼問題。

1. At some point the real density is $0.3$ and the generated density $0.1$. What score does the optimal discriminator give there? What if the generated density is $0$?
2. When the two distributions are identical, what are $\mathrm{JSD}$ and $V(D^*,G)$? What if they do not overlap at all?
3. A classmate says: "The stronger the discriminator the better; ideally train it to optimality at every step." Use the result of Section 4 to explain what could go wrong.

**第 1 題解答** `[補充]`：$0.3/(0.3+0.1)=0.75$。若生成資料密度是 $0$，$D^*=0.3/0.3=1$：這一點只可能是真的。

**Solution to question 1** `[Added]`: $0.3/(0.3+0.1)=0.75$. If the generated density is $0$, $D^*=0.3/0.3=1$: this point can only be real.

**第 2 題解答** `[補充]`：完全一樣時 $\mathrm{JSD}=0$，$V(D^*,G)=-\log4\approx-1.386$。完全不重疊時 $\mathrm{JSD}=\log2$，$V(D^*,G)=-\log4+2\log2=0$。

**Solution to question 2** `[Added]`: when identical, $\mathrm{JSD}=0$ and $V(D^*,G)=-\log4\approx-1.386$. With no overlap, $\mathrm{JSD}=\log2$ and $V(D^*,G)=-\log4+2\log2=0$.

**第 3 題解答** `[補充]`：訓練初期生成器和真資料離很遠，若判別器已經最優，生成器面對的就是 JS 散度的平台，梯度接近 $0$，學不動（也是 KP02 梯度飽和的另一面）。實務上要讓兩者勢均力敵，或改用非飽和損失、或換 KP07 的 Wasserstein 距離。

**Solution to question 3** `[Added]`: early in training the generator is far from the real data; if the discriminator is already optimal, the generator faces the plateau of the JS divergence, with gradient near $0$, and cannot learn (the other face of KP02's saturating gradient). In practice the two should be kept evenly matched, or the non-saturating loss used, or KP07's Wasserstein distance.

### 小測驗（Quiz，來源原題）

來源 KP-4 有一題自我檢測，保留原題與全部選項；來源頁面附有解析，下面依它說明。

The source's KP-4 has one self-check question, kept here with all options; the source page gives an explanation, which the note follows below.

1. 當判別器達到最優狀態 $D^*(\mathbf{x})$ 時，極小化極大博弈目標函數 $V(D^*,G)$ 的最小值為何？
   - (A) 最小值為 $0$，此時 JS 散度為無窮大
   - (B) 最小值為 $-\log4$，此時生成分佈 $p_g$ 完全等於真實分佈 $p_{\text{data}}$
   - (C) 最小值為 $+\log2$，此時生成器徹底失敗
   - (D) 最小值為 $-1$，此時判別器準確率為 $100\%$

1. When the discriminator is optimal, $D^*(\mathbf{x})$, what is the minimum of the minimax objective $V(D^*,G)$?
   - (A) The minimum is $0$, when the JS divergence is infinite
   - (B) The minimum is $-\log4$, when the generated distribution $p_g$ equals the real distribution $p_{\text{data}}$
   - (C) The minimum is $+\log2$, when the generator fails completely
   - (D) The minimum is $-1$, when the discriminator is $100\%$ accurate

**解析**：選 (B)。來源的理由：$p_g=p_{\text{data}}$ 時 $D^*=\tfrac12$，$V=\log\tfrac12+\log\tfrac12=-\log4\approx-1.386$，JS 散度是 $0$（見第3節）。`[補充]` 其他選項：(A) 錯，JS 散度最大只有 $\log2$，不會是無窮大，而 $V=0$ 是兩個分布完全不重疊時的最大值；(C) 錯，$V$ 不會超過 $0$，$+\log2$ 不可能出現；(D) 錯，判別器 $100\%$ 準確表示兩個分布不重疊，這時 $V=0$，是最大值，而且 $-1$ 不是任何一種情況的值。

**Explanation**: the answer is B. The source's reason: when $p_g=p_{\text{data}}$, $D^*=\tfrac12$, $V=\log\tfrac12+\log\tfrac12=-\log4\approx-1.386$, and the JS divergence is $0$ (see Section 3). `[Added]` The other options: A is wrong because the JS divergence is at most $\log2$, never infinite, and $V=0$ is the maximum, reached when the distributions do not overlap; C is wrong because $V$ never exceeds $0$, so $+\log2$ cannot occur; D is wrong because a $100\%$ accurate discriminator means the distributions do not overlap, giving $V=0$, the maximum, and $-1$ is not the value in any case.

### 留給下一單元的問題（The question for the next unit）

第3節說，判別器最優時，生成器的最好答案是 $p_g=p_{\text{data}}$，這時誰都沒有理由再改。這種「雙方都不想動」的狀態，在賽局理論叫什麼？實際用梯度去追它時，會不會繞圈、不收斂？下一個知識點（本週 KP04）談納許均衡，以及 GAN 訓練為什麼難。

Section 3 said that with an optimal discriminator the generator's best answer is $p_g=p_{\text{data}}$, after which neither side has any reason to change. What does game theory call this state where "neither side wants to move"? And when we chase it with gradients in practice, might we circle around instead of converging? The next knowledge point (this week's KP04) discusses the Nash equilibrium and why GAN training is hard.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第14章「產生式對抗網絡」KP-4「最優判別器 $D^*$ 與 Jensen-Shannon（JS）散度極值推導」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch14-gan/index.html ，讀取日期 2026-11-15）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from KP-4, "The optimal discriminator $D^*$ and the Jensen–Shannon (JS) divergence", of Chapter 14, "Generative adversarial networks", of CCC-LNS IDL_115 (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch14-gan/index.html , accessed 2026-11-15). Content not in the source is marked `[Added]` where it appears.

- 來源的結論寫「判別器達到最優時，生成器的訓練目標等價於極小化 JS 散度」。這在生成器用 KP02 原本的損失 $\log(1-D(G(\mathbf{z})))$ 時成立；改用非飽和損失後，生成器最小化的就不再嚴格是 JS 散度。本講義第4節註明，並列出另外兩個前提。
- 來源從 $V(D^*,G)$ 直接寫「整理可得」$-\log4+2\,\mathrm{JSD}$；本講義補上把分母寫成 $2m$、拆出兩個 $\log2$ 的步驟，以及 $f$ 是凹函數所以零點是最大值。
- 來源的互動展示用「分佈重合度」滑桿，沒有說明怎麼定義；本講義改用移動常態分布中心的例子。
- 來源用粗斜體 `\boldsymbol` 表示向量，並把 KL 散度寫成 $KLD$；本講義改用 `\mathbf` 與第12週的 $D_{\text{KL}}$。
- 本講義補上：每一點求極大的數字例子、一維例子的 $D^*$ 與 logit 推導、用樣本訓練的邏輯斯迴歸對照、密度比的說明、$V(D^*,G)$ 的數值核對、JS 散度的三個性質、移動中心的 JS 散度表與平台、結論的三個前提、三題小練習與解答。

- The source's conclusion says "when the discriminator is optimal, the generator's training objective is equivalent to minimising the JS divergence". This holds when the generator uses KP02's original loss $\log(1-D(G(\mathbf{z})))$; with the non-saturating loss the generator no longer strictly minimises the JS divergence. Section 4 notes this and lists two further conditions.
- The source goes from $V(D^*,G)$ to $-\log4+2\,\mathrm{JSD}$ with only "rearranging gives"; this note adds the steps of writing the denominator as $2m$ and splitting off two $\log2$ terms, and the concavity of $f$ that makes its zero a maximum.
- The source's interactive demo uses a "distribution overlap" slider without defining it; this note uses the example of moving the centre of a normal distribution instead.
- The source writes vectors with bold italic `\boldsymbol` and the KL divergence as $KLD$; this note uses `\mathbf` and Week 12's $D_{\text{KL}}$.
- This note adds: a numerical example of pointwise maximisation, $D^*$ and its logit in the one-dimensional example, the comparison with a logistic regression trained on samples, the density-ratio remark, the numerical check of $V(D^*,G)$, the three properties of the JS divergence, the table and plateau of the JS divergence as the centre moves, the three conditions of the result, and three practice questions with solutions.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 最優判別器 | optimal discriminator |
| 機率密度 | probability density |
| 凹函數 | concave function |
| 邏輯斯迴歸 | logistic regression |
| 密度比 | density ratio |
| KL 散度（第12週 KP02） | Kullback–Leibler (KL) divergence |
| JS 散度 | Jensen–Shannon divergence (JSD) |
| 平台（梯度接近零的區域） | plateau (a region where the gradient is near zero) |
| Wasserstein 距離（KP07） | Wasserstein distance |
| 納許均衡（KP04） | Nash equilibrium |
| 最大概似 | maximum likelihood |
| 生成對抗網路 | generative adversarial network (GAN) |
