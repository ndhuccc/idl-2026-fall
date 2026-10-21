# 第3週 KP10：AdaGrad，逐座標的自適應學習率（AdaGrad: per-coordinate adaptive learning rates）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第3章「反向傳播與最佳化」的第十個知識點（KP10）。KP09 看到隨機梯度要靠遞減的步長才會收斂，可是那是一條人工訂的排程，所有座標共用同一條。這一單元問：能不能讓步長按座標自動調？先備知識是梯度、KP02 的學習率與 KP05 的碗形例子；第一次出現的符號都會定義，所有數字都可以自己重算。內容依 CCC-LNS 第3章 KP10 的主文與投影片消化後重新撰寫，讀取日期 2026-10-14；來源沒有的內容標示 `[補充]`。

This is the tenth knowledge point (KP10) of Chapter 3. KP09 showed that stochastic gradients need a decreasing step size to converge, but that is a hand-made schedule shared by every coordinate. This unit asks: can the step size adjust itself coordinate by coordinate? It assumes gradients, the learning rate of KP02 and the bowl example of KP05; every symbol is defined when it first appears and every number can be recomputed. The text is rewritten from the KP10 main text and slides of CCC-LNS Chapter 3 (accessed on 2026-10-14); content that the source does not contain is marked `[Added]`.

[下載本單元投影片](../slides/lec03-kp10-adagrad.jbd.json)

## KP10 AdaGrad：逐座標的自適應學習率（AdaGrad: per-coordinate adaptive learning rates）

### 學習目標（Learning objectives）

讀完之後，你能說出一個全域學習率為什麼顧不了尺度差很多的座標；能寫出 AdaGrad 的累積量與更新規則，並用手算兩步的累積平方和與有效步長；能說明第一步每個座標都走 $\eta$ 的原因，以及 $\varepsilon$ 不是裝飾（它決定微小梯度的步長）；能在玩具碗上解釋為什麼陡的座標步長被壓得比平的座標小；也能指出「累積只增不減」使有效步長只降不升，並說明為什麼後期需要一種遺忘的機制。

After reading, you can say why one global learning rate cannot serve coordinates whose scales differ a lot; write the AdaGrad accumulator and update rule and compute two steps of accumulated squares and effective step sizes by hand; explain why every coordinate moves by $\eta$ in the first step and why $\varepsilon$ is not decoration (it sets the step for tiny gradients); explain on the toy bowl why the steep coordinate gets a smaller step than the flat one; and point out that "accumulation only grows" makes the effective step size non-increasing, so a mechanism of forgetting is needed later.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $w_j$ | 參數向量 $w$ 的第 $j$ 個座標 | the $j$-th coordinate of the parameter vector $w$ |
| $g_{t,j}$ | 第 $t$ 步梯度的第 $j$ 個座標 | the $j$-th coordinate of the gradient at step $t$ |
| $G_{t,j}$ | 累積量：前 $t$ 步該座標梯度的平方和 $\sum_{s\le t}g_{s,j}^2$ | accumulator: the sum of squares of that coordinate's gradients over the first $t$ steps |
| $\eta$ | 基礎學習率（所有座標共用的常數） | base learning rate (one constant shared by all coordinates) |
| $\varepsilon$ | 很小的正常數，避免除以零 | a tiny positive constant that avoids division by zero |
| $\eta/\sqrt{\varepsilon+G_{t,j}}$ | 座標 $j$ 的有效步長 | the effective step size of coordinate $j$ |

### 1. 一個學習率顧不了所有維度（One learning rate cannot serve every dimension）

KP09 的收斂條件要求步長遞減，那是一條人工訂的排程：「第 $k$ 步用 $1/(k+1)$」對所有座標一視同仁。但參數的各個座標梯度尺度常差很多。取 KP05 的玩具碗 $f(x,y)=\tfrac12(x^2+25y^2)$，梯度是 $(x,\,25y)$：$y$ 方向的曲率是 $x$ 方向的 25 倍。用固定步長 $\eta$ 做梯度下降，兩個座標各自乘上固定的係數：

The convergence condition of KP09 asks for a decreasing step size, a hand-made schedule: "use $1/(k+1)$ at step $k$" treats every coordinate alike. But the coordinates of the parameters often have very different gradient scales. Take the toy bowl of KP05, $f(x,y)=\tfrac12(x^2+25y^2)$, with gradient $(x,\,25y)$: the curvature in $y$ is 25 times that in $x$. Gradient descent with a fixed step $\eta$ multiplies each coordinate by a fixed factor:

$$
x_{k+1}=(1-\eta)\,x_k,\qquad y_{k+1}=(1-25\eta)\,y_k
$$

要兩個座標都收斂，需要 $|1-25\eta|<1$，即 $\eta<2/25=0.08$。從 $(10,2)$ 出發，三種步長的結果〔程式計算〕：

Both coordinates converge only if $|1-25\eta|<1$, that is $\eta<2/25=0.08$. From $(10,2)$, the three step sizes give [computed by program]:

| $\eta$ | $x$ 的係數 | $y$ 的係數 | 結果 |
|---|---|---|---|
| 0.01 | 0.990 | 0.750 | $y$ 很快收斂，但 $x$ 縮到 $1/10$ 要 230 步 |
| 0.07 | 0.930 | $-0.750$ | 都收斂，$x$ 縮到 $1/10$ 要 32 步，$y$ 每步變號 |
| 0.10 | 0.900 | $-1.500$ | $y$ 每步放大 1.5 倍，30 步後 $\lvert y\rvert\approx3.8\times10^{5}$，發散 |

| $\eta$ | factor of $x$ | factor of $y$ | result |
|---|---|---|---|
| 0.01 | 0.990 | 0.750 | $y$ converges quickly, but $x$ needs 230 steps to shrink to $1/10$ |
| 0.07 | 0.930 | $-0.750$ | both converge; $x$ needs 32 steps to shrink to $1/10$ and $y$ flips sign every step |
| 0.10 | 0.900 | $-1.500$ | $y$ grows 1.5 times per step; after 30 steps $\lvert y\rvert\approx3.8\times10^{5}$, it diverges |

這就是兩難：陡的座標限制了學習率的上限，平的座標卻因此走得慢。真實的網路有成千上萬個座標，每個座標的尺度又隨訓練變化，沒有辦法事先一個一個調。AdaGrad（Duchi 等人，2011）的想法是讓演算法自己根據每個座標看過的梯度，各配一個步長。

This is the dilemma: the steep coordinate limits the upper bound of the learning rate and the flat coordinate therefore moves slowly. A real network has thousands or millions of coordinates whose scales also change during training, so tuning them one by one in advance is impossible. The idea of AdaGrad (Duchi et al., 2011) is to let the algorithm assign each coordinate its own step size from the gradients that coordinate has seen.

### 2. 更新規則：以歷史梯度平方和縮放（The update rule: scaling by the history of squared gradients）

為每個座標 $j$ 各記一個累積量，把看過的梯度平方加起來：

Keep one accumulator per coordinate $j$ and add up the squared gradients seen so far:

$$
G_{t,j}=\sum_{s=1}^{t}g_{s,j}^{2},\qquad
w_{j}\leftarrow w_{j}-\frac{\eta}{\sqrt{\varepsilon+G_{t,j}}}\,g_{t,j}
$$

整個向量一起算，等於逐元素的運算：先 $G\leftarrow G+g\odot g$，再 $w\leftarrow w-\eta\,g\,/\sqrt{\varepsilon+G}$。額外的記憶體是一個和 $w$ 一樣長的向量，運算量是 $O(d)$。更新用的是**含本步梯度的** $G_{t,j}$。把 $\eta/\sqrt{\varepsilon+G_{t,j}}$ 看成座標 $j$ 的**有效步長**：累積量大的座標，有效步長小；累積量小的座標，有效步長大。

For the whole vector this is an element-wise operation: first $G\leftarrow G+g\odot g$, then $w\leftarrow w-\eta\,g\,/\sqrt{\varepsilon+G}$. The extra memory is one vector of the same length as $w$ and the cost is $O(d)$. The update uses $G_{t,j}$ **including the current gradient**. Read $\eta/\sqrt{\varepsilon+G_{t,j}}$ as the **effective step size** of coordinate $j$: a coordinate with a large accumulator gets a small effective step, one with a small accumulator gets a large one.

**第一步每個座標都走 $\eta$。** 第一步 $G_{1,j}=g_{1,j}^2$，所以移動量是 $\eta\,g/\sqrt{\varepsilon+g^2}\approx\eta\cdot\mathrm{sign}(g)$（當 $g^2\gg\varepsilon$）：不管梯度是 10 還是 50，每個座標都走 $\eta$，方向由梯度的符號決定。從 $(10,2)$、$\eta=1$：梯度 $(10,50)$，第一步到 $(9,1)$〔程式計算〕。之後各座標才因累積量不同而分道揚鑣。

**In the first step every coordinate moves by $\eta$.** In the first step $G_{1,j}=g_{1,j}^2$, so the move is $\eta\,g/\sqrt{\varepsilon+g^2}\approx\eta\cdot\mathrm{sign}(g)$ (when $g^2\gg\varepsilon$): whether the gradient is 10 or 50, each coordinate moves by $\eta$ in the direction of the gradient's sign. From $(10,2)$ with $\eta=1$: the gradient is $(10,50)$ and the first step lands at $(9,1)$ [computed by program]. Only afterwards do the coordinates part ways because their accumulators differ.

**$\varepsilon$ 不是裝飾。** 它至少做兩件事：（1）避免 $G=0$ 時 $0/0$（某座標還沒有出現過梯度，$\varepsilon=0$ 會得到 NaN；有 $\varepsilon$ 則該步移動量為 0）；（2）決定**微小梯度**的第一步有多大。第一次出現的梯度 $g=0.001$、$\eta=1$：$\varepsilon=10^{-8}$ 時移動 $0.9950$；$\varepsilon=10^{-6}$ 時 $0.7071$；$\varepsilon=10^{-4}$ 時只有 $0.0995$〔程式計算〕。梯度的平方和 $\varepsilon$ 差不多大時，$\varepsilon$ 就左右了步長。另外 $\varepsilon$ 放在根號裡或外面，不同的函式庫做法不同 `[補充]`（例如 PyTorch 的 Adagrad 是 $\sqrt{G}+\varepsilon$，預設 $10^{-10}$），數字大時差別可忽略，梯度很小時要留意。

**$\varepsilon$ is not decoration.** It does at least two things: (1) it avoids $0/0$ when $G=0$ (a coordinate that has never had a gradient gives NaN with $\varepsilon=0$; with $\varepsilon$ the move of that step is 0); (2) it decides how large the first step for a **tiny gradient** is. For a first-time gradient $g=0.001$ with $\eta=1$: with $\varepsilon=10^{-8}$ the move is $0.9950$; with $\varepsilon=10^{-6}$ it is $0.7071$; with $\varepsilon=10^{-4}$ only $0.0995$ [computed by program]. When the squared gradient and $\varepsilon$ are of similar size, $\varepsilon$ governs the step. Libraries also differ on whether $\varepsilon$ sits inside or outside the root `[Added]` (PyTorch's Adagrad uses $\sqrt{G}+\varepsilon$ with default $10^{-10}$); the difference is negligible for large numbers but matters for very small gradients.

**稀疏座標得到較大的步長。** 假設兩個座標，梯度大小在出現時都是 1：座標 A 每一步都出現，座標 B 只在第 1、50、100 步出現。B 的累積量只有 1、2、3，有效步長是 $1,\ 0.7071,\ 0.5774$；A 在第 100 步的累積量是 100，有效步長是 $0.1$〔程式計算〕。罕見的座標每次出現都能走大步，常出現的座標被壓小。這就是 AdaGrad 常用在稀疏特徵（如詞向量、推薦系統）的原因 `[補充]`。

**Sparse coordinates get larger steps.** Take two coordinates whose gradient is 1 whenever present: coordinate A is present at every step and coordinate B only at steps 1, 50 and 100. B's accumulator is only 1, 2, 3 and its effective step is $1,\ 0.7071,\ 0.5774$; A's accumulator at step 100 is 100 and its effective step is $0.1$ [computed by program]. A rare coordinate takes a large step every time it appears and a frequent coordinate is damped. This is why AdaGrad is often used for sparse features such as word vectors and recommender systems `[Added]`.

### 3. 在碗上跑幾步（Running a few steps on the bowl）

仍用 $f(x,y)=\tfrac12(x^2+25y^2)$、起點 $(10,2)$、$\eta=1$、$\varepsilon=10^{-8}$，前兩步〔程式計算〕：

Still the bowl $f(x,y)=\tfrac12(x^2+25y^2)$ from $(10,2)$ with $\eta=1$, $\varepsilon=10^{-8}$; the first two steps [computed by program]:

| 步 $k$ | 梯度 $(g_x,g_y)$ | 累積 $(G_x,G_y)$ | 有效步長 $(x,y)$ | 移動量 | 新位置 |
|---|---|---|---|---|---|
| 1 | $(10,\,50)$ | $(100,\,2500)$ | $(0.1000,\,0.0200)$ | $(1,\,1)$ | $(9,\,1)$ |
| 2 | $(9,\,25)$ | $(181,\,3125)$ | $(0.0743,\,0.0179)$ | $(0.669,\,0.447)$ | $(8.331,\,0.553)$ |

| step $k$ | gradient $(g_x,g_y)$ | accumulator $(G_x,G_y)$ | effective step $(x,y)$ | move | new position |
|---|---|---|---|---|---|
| 1 | $(10,\,50)$ | $(100,\,2500)$ | $(0.1000,\,0.0200)$ | $(1,\,1)$ | $(9,\,1)$ |
| 2 | $(9,\,25)$ | $(181,\,3125)$ | $(0.0743,\,0.0179)$ | $(0.669,\,0.447)$ | $(8.331,\,0.553)$ |

第 1 步兩個座標都走 1，但有效步長相差 5 倍（$0.1$ 對 $0.02$），因為 $y$ 的梯度是 $x$ 的 5 倍。陡的 $y$ 座標步長小，所以它不會像 $\eta=0.1$ 的梯度下降那樣越衝越遠；平的 $x$ 座標的步長較大。

In step 1 both coordinates move by 1, yet the effective steps differ by a factor of 5 ($0.1$ versus $0.02$) because the gradient of $y$ is 5 times that of $x$. The steep coordinate $y$ takes small steps, so it does not overshoot further and further as gradient descent with $\eta=0.1$ does; the flat coordinate $x$ takes larger steps.

![玩具碗上的路徑：橘線是 η=1 的 AdaGrad，青綠線是 η=0.07 的梯度下降，各 30 步，都從 (10, 2) 出發；灰色是等高線。AdaGrad 的 y 座標單調縮向 0，梯度下降的 y 座標每步變號。（Paths on the toy bowl: the orange line is AdaGrad with η=1, the teal line is gradient descent with η=0.07, 30 steps each from (10, 2); the grey curves are contours. AdaGrad's y coordinate shrinks monotonically to 0 while gradient descent's y flips sign every step.）](../assets/figures/lec03-kp10-path.png)

**圖說 `lec03-kp10-path`**：先看青綠線：梯度下降 $\eta=0.07$（接近 $0.08$ 的上限）在 $y$ 方向來回跳，每步變號、振幅乘 $0.75$ 慢慢縮小。再看橘線：AdaGrad 的 $y$ 從 $2$ 到 $1$、$0.553$、$0.313$，沒有變號，約 10 步就降到 $0.006$。30 步後 AdaGrad 在 $(2.657,\,0)$，梯度下降在 $(1.134,\,0.0004)$：**這個例子裡調好的梯度下降在 $x$ 方向反而走得比較遠**。AdaGrad 的好處是不必事先知道 25 這個數字就能穩定，而不是在任何情況下都較快。〔功能：1、3〕

**Caption `lec03-kp10-path`**: Look at the teal line first: gradient descent with $\eta=0.07$ (close to the $0.08$ limit) bounces in the $y$ direction, flipping sign every step with its amplitude shrinking by a factor of $0.75$. Then the orange line: AdaGrad's $y$ goes $2\to1\to0.553\to0.313$ with no sign change and drops to $0.006$ in about 10 steps. After 30 steps AdaGrad is at $(2.657,\,0)$ and gradient descent at $(1.134,\,0.0004)$: **in this example the well-tuned gradient descent actually goes further in the $x$ direction**. The benefit of AdaGrad is stability without knowing the number 25 in advance, not being faster in every case. [Function: 1, 3]

![兩個座標的有效步長 η/√G：藍線是 x，橘線是 y，紅色虛線是梯度下降的固定步長 0.07。y 的有效步長一開始就小（0.02）並很快固定在 0.0171；x 的有效步長從 0.1 緩慢降到 0.033。（The effective step η/√G of the two coordinates: blue is x, orange is y, the red dashed line is the fixed step 0.07 of gradient descent. The effective step of y is small from the start (0.02) and settles at 0.0171 quickly; that of x falls slowly from 0.1 to 0.033.）](../assets/figures/lec03-kp10-eff-step.png)

**圖說 `lec03-kp10-eff-step`**：縱軸是對數刻度，讀三個點：第 1 步 $x$ 是 $0.1$、$y$ 是 $0.02$；第 10 步 $x$ 是 $0.0420$、$y$ 是 $0.0171$；第 30 步 $x$ 是 $0.0335$、$y$ 是 $0.0171$。比值 $y/x$ 從 $0.20$ 升到 $0.51$。$y$ 的曲線趨平是因為 $y$ 很快降到接近 0，梯度消失，$G_y$ 幾乎不再增加；它有 $73\%$ 是第一步的 $50^2=2500$ 貢獻的。有效步長沒有變大，只是不再下降。〔功能：1、2〕

**Caption `lec03-kp10-eff-step`**: The vertical axis is logarithmic; read three points: at step 1, $x$ is $0.1$ and $y$ is $0.02$; at step 10, $x$ is $0.0420$ and $y$ is $0.0171$; at step 30, $x$ is $0.0335$ and $y$ is $0.0171$. The ratio $y/x$ rises from $0.20$ to $0.51$. The curve of $y$ flattens because $y$ soon falls near 0, its gradient vanishes and $G_y$ barely grows; $73\%$ of it comes from the first step's $50^2=2500$. The effective step never grows; it only stops falling. [Function: 1, 2]

### 4. 累積只增不減：有效步長只降不升（Accumulation only grows, so the effective step only falls）

平方是非負的，所以 $G_{t,j}$ 對 $t$ 單調不減，有效步長 $\eta/\sqrt{\varepsilon+G_{t,j}}$ 也就**只降不升**。若某座標的梯度大小一直是 $g$，則 $G_{k}=k\,g^2$，有效步長是 $\eta/(|g|\sqrt k)$：以 $1/\sqrt k$ 的速度下降。而且早期的大梯度**永遠留在累積量裡**：下面的例子中，一個座標第一步的梯度是 $50$，之後都是 $1$。

Squares are non-negative, so $G_{t,j}$ is non-decreasing in $t$ and the effective step $\eta/\sqrt{\varepsilon+G_{t,j}}$ is **non-increasing**. If a coordinate's gradient has constant size $g$, then $G_k=k\,g^2$ and the effective step is $\eta/(|g|\sqrt k)$: it falls at the rate $1/\sqrt k$. Early large gradients also **stay in the accumulator forever**: in the next example one coordinate has gradient $50$ at the first step and $1$ afterwards.

![有效步長隨步數下降（η=1，雙對數座標）：藍線是梯度一直是 1，為 1/√k；橘線是第一步梯度 50、之後都是 1。第 100 步橘線只有 0.0196，藍線有 0.1；到第 10000 步兩者才接近（0.0089 對 0.01）。（Effective step versus step count (η=1, log-log axes): the blue line has gradient 1 throughout, 1/√k; the orange line has gradient 50 at the first step and 1 afterwards. At step 100 the orange line is only 0.0196 while the blue is 0.1; they come close only by step 10000 (0.0089 versus 0.01).）](../assets/figures/lec03-kp10-shrink.png)

**圖說 `lec03-kp10-shrink`**：先看藍線的兩個點：第 100 步 $0.1$、第 10000 步 $0.01$，每多 100 倍的步數，有效步長只剩 $1/10$。再看橘線：一次早期的大梯度讓它從頭到尾比藍線小，第 100 步是 $0.0196$（藍線的 $19.6\%$），直到 $k$ 大到 $G$ 被後面的小梯度追上才接近。這是 AdaGrad 的記憶只加不減的後果。〔功能：1、2〕

**Caption `lec03-kp10-shrink`**: Read the two blue points first: $0.1$ at step 100 and $0.01$ at step 10000; every 100 times more steps leaves only $1/10$ of the effective step. Now the orange line: one early large gradient keeps it below the blue line throughout, $0.0196$ at step 100 ($19.6\%$ of the blue), until the later small gradients catch up with $G$ at large $k$. This is the consequence of an AdaGrad memory that only adds. [Function: 1, 2]

所以訓練後期有可能步長變得太小：例如梯度不小，但累積量已經很大，模型走不動。也無法適應地形變化：前面雜訊大的梯度永遠壓著之後的步長。修正的方向是讓累積量能**遺忘**舊的梯度，讓有效步長有機會回升。怎麼讓它遺忘？這是 KP11 的問題。

So late in training the step can become too small: the gradient is not small but the accumulator is already large and the model cannot move. It also cannot adapt to changing terrain: noisy early gradients press on all later steps forever. The fix is to let the accumulator **forget** old gradients so that the effective step can recover. How to make it forget? That is the question of KP11.

### 本單元重點整理（KP10 summary）

**一句話重點**：AdaGrad 用每個座標「歷史梯度平方和」的平方根縮放步長：梯度常大的座標步小、罕見的座標步大；代價是累積只增不減，有效步長只降不升。

**One-sentence summary**: AdaGrad scales each coordinate's step by the square root of its "sum of historical squared gradients": coordinates with often-large gradients take small steps and rare coordinates take large ones; the price is that the accumulation only grows, so the effective step only falls.

**白話說明**：一個學習率對尺度不同的座標不是太大就是太小。AdaGrad 替每個座標記下「到目前為止它被推得多用力」（梯度平方的總和），被推得越用力，之後的步子就越小心；很少被推的座標，步子就大。沒有人需要事先告訴它哪個方向陡。缺點是記帳本只會越寫越厚，步子只會越來越小。

**Plain explanation**: One learning rate is too big or too small for coordinates of different scales. AdaGrad records for each coordinate "how hard it has been pushed so far" (the sum of squared gradients); the harder it was pushed, the more carefully the later steps are taken, and a coordinate that is rarely pushed takes big steps. Nobody has to tell it in advance which direction is steep. The drawback is that the ledger only gets thicker and the steps only get smaller.

**正式定義**：$G_{t,j}=\sum_{s\le t}g_{s,j}^2$；$w_j\leftarrow w_j-\dfrac{\eta}{\sqrt{\varepsilon+G_{t,j}}}\,g_{t,j}$；有效步長 $\eta/\sqrt{\varepsilon+G_{t,j}}$ 對 $t$ 單調不增。

**Formal definition**: $G_{t,j}=\sum_{s\le t}g_{s,j}^2$; $w_j\leftarrow w_j-\dfrac{\eta}{\sqrt{\varepsilon+G_{t,j}}}\,g_{t,j}$; the effective step $\eta/\sqrt{\varepsilon+G_{t,j}}$ is non-increasing in $t$.

**例子**：玩具碗從 $(10,2)$、$\eta=1$：第 1 步梯度 $(10,50)$，累積 $(100,2500)$，每個座標都走 1，到 $(9,1)$；第 2 步梯度 $(9,25)$，累積 $(181,3125)$，有效步長 $(0.0743,\,0.0179)$，到 $(8.331,\,0.553)$。常數梯度 1：有效步長 $1/\sqrt k$，第 100 步 $0.1$、第 10000 步 $0.01$。

**Example**: the toy bowl from $(10,2)$ with $\eta=1$: step 1 has gradient $(10,50)$ and accumulators $(100,2500)$; each coordinate moves by 1 to $(9,1)$; step 2 has gradient $(9,25)$, accumulators $(181,3125)$, effective steps $(0.0743,\,0.0179)$ and lands at $(8.331,\,0.553)$. A constant gradient 1: the effective step is $1/\sqrt k$, $0.1$ at step 100 and $0.01$ at step 10000.

**常見迷思**：

- 「AdaGrad 一定比 SGD 好」——不一定。玩具碗上 $\eta=0.07$ 的梯度下降 30 步後 $x=1.134$，$\eta=1$ 的 AdaGrad 是 $2.657$。它的好處是不必知道各座標的尺度，而不是任何情況下都快。
- 「$\varepsilon$ 只是裝飾」——它避免 $0/0$，也決定微小梯度的步長：$g=0.001$ 時，$\varepsilon=10^{-8},10^{-6},10^{-4}$ 的第一步是 $0.9950,\,0.7071,\,0.0995$。
- 「有效步長會自己調回來」——不會。分母單調不減，有效步長只降不升；梯度消失時只是停止下降。
- 「第一步的移動量和梯度大小成正比」——不是，第一步每個座標都約等於 $\eta$（符號 $\times\eta$）。

**Common misconceptions**:

- "AdaGrad is always better than SGD." Not necessarily. On the toy bowl gradient descent with $\eta=0.07$ reaches $x=1.134$ after 30 steps while AdaGrad with $\eta=1$ reaches $2.657$. Its benefit is not needing to know the scale of each coordinate, not being fast in every case.
- "$\varepsilon$ is just decoration." It avoids $0/0$ and also sets the step for tiny gradients: with $g=0.001$ the first step is $0.9950,\,0.7071,\,0.0995$ for $\varepsilon=10^{-8},10^{-6},10^{-4}$.
- "The effective step adjusts back by itself." It does not. The denominator is non-decreasing and the effective step only falls; when the gradient vanishes it merely stops falling.
- "The first move is proportional to the gradient size." No: in the first step every coordinate moves by about $\eta$ (sign times $\eta$).

**圖**：本單元三張圖，`lec03-kp10-path`（碗上路徑，功能 1、3）、`lec03-kp10-eff-step`（兩座標的有效步長，功能 1、2）、`lec03-kp10-shrink`（累積只增不減，功能 1、2），以 `assets/figures/src/lec03-kp10-figures.py` 繪製，所有引用的數字由同一支腳本印出。

**Figures**: Three figures in this unit, `lec03-kp10-path` (paths on the bowl; functions 1, 3), `lec03-kp10-eff-step` (the effective steps of the two coordinates; functions 1, 2) and `lec03-kp10-shrink` (accumulation only grows; functions 1, 2), drawn by `assets/figures/src/lec03-kp10-figures.py`; every number quoted is printed by the same script.

**想深入**（選讀）：AdaGrad 在凸的線上學習問題有後悔界（regret bound）的理論保證，證明需要凸性，留給進階課程。把它想成「對角的曲率校正」只是一種比喻：累積的是梯度平方，不是 Hessian，兩者在一般情況下不相等，所以不要把 $G_{t,j}$ 當成曲率的估計來用。另一個細節：若學習率本身也隨步數遞減，會和 $G$ 的遞減疊加，實務上通常用固定的 $\eta$。

**Going deeper** (optional): AdaGrad has a theoretical guarantee in the form of a regret bound for convex online learning; the proof needs convexity and is left to advanced courses. Thinking of it as a "diagonal curvature correction" is only an analogy: it accumulates squared gradients, not the Hessian, and the two are not equal in general, so do not use $G_{t,j}$ as a curvature estimate. One more detail: if the learning rate itself also decreases with the step count it stacks with the decay of $G$; in practice a constant $\eta$ is usually used.

**小練習**：

1. 某座標的前四步梯度是 $3,\,4,\,0,\,12$，$\eta=1$、$\varepsilon\to0$：寫出每一步的累積量 $G$ 與有效步長。（提示：每步把梯度平方加進去再開根號。）
2. 某座標第一次出現梯度 $g=0.01$，$\eta=1$、$\varepsilon=10^{-4}$：第一步移動多少？（提示：$g/\sqrt{\varepsilon+g^2}$。）
3. 某座標的梯度大小永遠是 $g$，有效步長要降到第 1 步的 $1/10$ 需要幾步？（提示：$\eta/(|g|\sqrt k)$ 對 $k=1$。）

**Practice**:

1. The gradients of a coordinate over the first four steps are $3,\,4,\,0,\,12$ with $\eta=1$ and $\varepsilon\to0$: write the accumulator $G$ and the effective step of each step. (Hint: add the squared gradient and take the root.)
2. A coordinate has its first gradient $g=0.01$ with $\eta=1$ and $\varepsilon=10^{-4}$: how far does it move in the first step? (Hint: $g/\sqrt{\varepsilon+g^2}$.)
3. The gradient of a coordinate always has size $g$: how many steps until the effective step falls to $1/10$ of its step-1 value? (Hint: $\eta/(|g|\sqrt k)$ against $k=1$.)

**解題示範**（用玩具碗的前兩步示範第 1 題的算法）：已知 $\eta=1$、$\varepsilon\to0$，第 1 步梯度 $(10,50)$，第 2 步梯度 $(9,25)$。

1. 第 1 步累積：$G_x=10^2=100$，$G_y=50^2=2500$。
2. 第 1 步有效步長：$1/\sqrt{100}=0.1$，$1/\sqrt{2500}=0.02$；移動量 $0.1\cdot10=1$，$0.02\cdot50=1$，兩個座標都走 $\eta=1$。
3. 第 2 步累積：$G_x=100+9^2=181$，$G_y=2500+25^2=3125$。
4. 第 2 步有效步長：$1/\sqrt{181}=0.0743$，$1/\sqrt{3125}=0.0179$。
5. 移動量：$0.0743\cdot9=0.669$，$0.0179\cdot25=0.447$。
6. 檢查：第 2 步有效步長都比第 1 步小（$0.0743<0.1$、$0.0179<0.02$）；$y$ 的有效步長仍是 $x$ 的 $0.0179/0.0743=0.24$ 倍，比第 1 步的 $0.2$ 倍更接近，但 $y$ 仍較小。

**Worked solution** (the method of exercise 1, shown with the first two steps on the toy bowl): given $\eta=1$, $\varepsilon\to0$, gradient $(10,50)$ at step 1 and $(9,25)$ at step 2.

1. Accumulators after step 1: $G_x=10^2=100$, $G_y=50^2=2500$.
2. Effective steps at step 1: $1/\sqrt{100}=0.1$ and $1/\sqrt{2500}=0.02$; the moves are $0.1\cdot10=1$ and $0.02\cdot50=1$: both coordinates move by $\eta=1$.
3. Accumulators after step 2: $G_x=100+9^2=181$, $G_y=2500+25^2=3125$.
4. Effective steps at step 2: $1/\sqrt{181}=0.0743$ and $1/\sqrt{3125}=0.0179$.
5. Moves: $0.0743\cdot9=0.669$ and $0.0179\cdot25=0.447$.
6. Check: both effective steps at step 2 are smaller than at step 1 ($0.0743<0.1$, $0.0179<0.02$); the effective step of $y$ is still $0.0179/0.0743=0.24$ times that of $x$, closer to $x$ than the $0.2$ times of step 1, but still smaller.

### 來源更正與取捨（Corrections to the source）

- 來源說 AdaGrad 在「訓練中盤有效步長就衰減到 $10^{-6}$、$10^{-7}$ 以下，模型徹底凍結」。這句話沒有依據：衰減速度取決於梯度大小與步數，常數梯度時是 $1/\sqrt k$（第 10000 步是 $\eta/(100|g|)$，離 $10^{-6}$ 還很遠），本講義不採用，改用可重算的例子。
- 來源說 AdaGrad 把橢圓等高線「拉伸為近似正圓」、「無一例外」被 DistBelief 與 GloVe 選為預設最佳化器，並把它說成對角的二階曲率近似。這些是誇大或無依據的說法：本講義只說「逐座標縮放」，並提醒 $G_{t,j}$ 不是曲率估計；碗上的例子也顯示調好的梯度下降在 $x$ 方向不輸給 AdaGrad。
- 來源說 $\varepsilon$ 的主要作用是「確保分母不為零」。保留，並補上 $\varepsilon$ 決定微小梯度第一步大小的數字例（$0.9950,\,0.7071,\,0.0995$），避免讀者以為它無關緊要。
- 來源沒有說明 $\varepsilon$ 放在根號內外的差別；本講義補上並標示為 `[補充]`。後悔界只在「想深入」一句話提及。

- The source says that "by mid-training the effective step has decayed below $10^{-6}$ or $10^{-7}$ and the model freezes completely". There is no basis for this: the decay speed depends on the gradient size and the step count; for a constant gradient it is $1/\sqrt k$ (at step 10000 it is $\eta/(100|g|)$, far from $10^{-6}$). This note does not adopt it and uses recomputable examples instead.
- The source says AdaGrad stretches elliptical contours "into approximate circles", that DistBelief and GloVe chose it as the default optimizer "without exception", and presents it as a diagonal second-order curvature approximation. These are overstatements or unsupported: this note only says "per-coordinate scaling" and warns that $G_{t,j}$ is not a curvature estimate; the bowl example also shows that a well-tuned gradient descent does not lose to AdaGrad in the $x$ direction.
- The source says the main role of $\varepsilon$ is "to ensure the denominator is not zero". This is kept, with numbers added showing that $\varepsilon$ sets the first step for tiny gradients ($0.9950,\,0.7071,\,0.0995$), so that readers do not take it as unimportant.
- The source does not discuss whether $\varepsilon$ sits inside or outside the root; this note adds it and marks it `[Added]`. The regret bound is mentioned only in one sentence under "Going deeper".

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 自適應學習率 | adaptive learning rate |
| 逐座標（逐元素） | per-coordinate (element-wise) |
| 累積量 | accumulator |
| 有效步長 | effective step size |
| 梯度平方和 | sum of squared gradients |
| 平滑常數 | smoothing constant ($\varepsilon$) |
| 稀疏特徵 | sparse feature |
| 後悔界 | regret bound |
