# 第3週 KP07：隨機梯度下降（Stochastic gradient descent, SGD）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第3章「反向傳播與最佳化」的第七個知識點（KP07），承接 KP06 留下的問題：資料有上百萬筆時，連算一次完整的梯度都很貴，怎麼辦？先備知識是機率的期望與變異數、KP01–KP06；第一次出現的符號都會定義，所有數字都可以自己重算。內容依 CCC-LNS 第3章 KP07 的主文與投影片消化後重新撰寫，讀取日期 2026-10-14；來源沒有的內容標示 `[補充]`。

This is the seventh knowledge point (KP07) of Chapter 3. It picks up the question KP06 left open: with millions of examples even one full gradient is expensive, so what do we do? It assumes expectation and variance from probability and KP01–KP06; every symbol is defined when it first appears and every number can be recomputed. The text is rewritten from the KP07 main text and slides of CCC-LNS Chapter 3 (accessed on 2026-10-14); content that the source does not contain is marked `[Added]`.

[下載本單元投影片](../slides/lec03-kp07-sgd.jbd.json)

## KP07 隨機梯度下降（Stochastic gradient descent, SGD）

### 學習目標（Learning objectives）

讀完之後，你能寫出經驗風險與完整梯度並說明每步 $O(n)$ 的代價；能寫出 SGD 的單樣本更新並證明單樣本梯度是完整梯度的無偏估計；能解釋「無偏不代表每步下降」，並算出固定步長下 SGD 在最小點附近的擺動幅度；也能說明遞減步長為什麼能讓 SGD 收斂，以及 SGD 的取捨。

After reading, you can write the empirical risk and the full gradient and explain the $O(n)$ cost per step; write the single-example update of SGD and prove that the single-example gradient is an unbiased estimate of the full gradient; explain that "unbiased does not mean every step descends" and compute how widely SGD swings around the minimum with a fixed step; and explain why a decaying step size lets SGD converge, and what SGD trades off.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $n$ | 訓練樣本的筆數 | number of training examples |
| $f_i(w)$ | 第 $i$ 筆樣本的損失 | the loss of example $i$ |
| $F(w)=\tfrac1n\sum_if_i(w)$ | 全體的平均損失（經驗風險） | the average loss over all examples (empirical risk) |
| $i_k$ | 第 $k$ 步隨機抽到的樣本編號 | the index of the example drawn at step $k$ |
| $\eta_k$ | 第 $k$ 步的學習率（可以隨步數改變） | the learning rate at step $k$ (may change with $k$) |
| $\mathbb E[\cdot]$ | 對抽樣的期望值 | expectation over the sampling |
| $\sigma^2$ | 單筆梯度的變異數 | variance of a single-example gradient |

### 1. 完整梯度每一步都要掃過全部資料（The full gradient scans all the data at every step）

機器學習的目標是讓 $n$ 筆樣本損失的平均最小：

The goal of machine learning is to minimize the average loss over $n$ examples:

$$
\min_w F(w),\qquad F(w)=\frac1n\sum_{i=1}^{n}f_i(w)
$$

微分是線性的，所以完整梯度是每一筆梯度的平均：$\nabla F(w)=\tfrac1n\sum_i\nabla f_i(w)$。代價是：**每更新一步**都要把全部 $n$ 筆資料各做一次前向與反向傳播（KP06）。$n$ 是一百萬時，一步就等於掃過一百萬筆；$n$ 是數十億時，一步要花好幾個小時，訓練根本走不動。

Differentiation is linear, so the full gradient is the average of the per-example gradients: $\nabla F(w)=\tfrac1n\sum_i\nabla f_i(w)$. The cost: **every update** needs a forward and a backward pass (KP06) over all $n$ examples. With $n$ a million, one step scans a million examples; with billions, one step takes hours and training cannot move.

### 2. SGD：每步只抽一筆（SGD: draw one example per step）

想法很樸素：與其花很大的力氣算出一個精確的梯度，不如每步只**隨機抽一筆** $i_k$，用它的梯度當作完整梯度的替身：

The idea is plain: instead of spending a great effort on one exact gradient, **draw one example at random**, $i_k$, and use its gradient as a stand-in for the full gradient:

$$
w^{(k+1)}=w^{(k)}-\eta_k\,\nabla f_{i_k}\big(w^{(k)}\big)
$$

每步只對一筆資料做前向與反向，成本是 $O(1)$，和 $n$ 無關。「算一次完整的精確梯度的時間，足夠讓單筆更新走上好幾千甚至上百萬步」。

Each step does a forward and a backward pass on one example only, a cost of $O(1)$ independent of $n$. "The time to compute one exact full gradient is enough for thousands or even millions of single-example updates."

**為什麼這個替身是可以接受的？**因為它是**無偏**的。若抽樣是均勻的，$P(i=j)=1/n$，則

**Why is the stand-in acceptable?** Because it is **unbiased**. If the sampling is uniform, $P(i=j)=1/n$, then

$$
\mathbb E_i\big[\nabla f_i(w)\big]=\sum_{j=1}^{n}P(i=j)\,\nabla f_j(w)=\frac1n\sum_{j=1}^{n}\nabla f_j(w)=\nabla F(w)
$$

也就是說，單筆梯度的**期望方向**等於真正的完整梯度。用一個具體的例子驗算（`[補充]`）：四筆樣本，$f_i(w)=\tfrac12(w-a_i)^2$，$a=(2,4,6,8)$，所以 $\nabla f_i(w)=w-a_i$。完整的 $F(w)=\tfrac14\sum_if_i(w)$，$\nabla F(w)=w-\bar a=w-5$。在 $w=0$：四個單筆梯度是 $-2,-4,-6,-8$，平均 $=-5=\nabla F(0)$，相符。

In other words, the **expected direction** of the single-example gradient equals the true full gradient. A concrete check (`[Added]`): four examples with $f_i(w)=\tfrac12(w-a_i)^2$ and $a=(2,4,6,8)$, so $\nabla f_i(w)=w-a_i$. The full $F(w)=\tfrac14\sum_if_i(w)$ has $\nabla F(w)=w-\bar a=w-5$. At $w=0$ the four single-example gradients are $-2,-4,-6,-8$ with mean $-5=\nabla F(0)$, a match.

### 3. 無偏不代表每一步都往下（Unbiased does not mean every step descends）

把單筆梯度拆成「真梯度」加「零均值的雜訊」：$\nabla f_{i}(w)=\nabla F(w)+\xi_i$，$\mathbb E[\xi_i]=0$。平均而言往對的方向走，但每一步都有一份隨機的偏差。

Split the single-example gradient into "the true gradient" plus "zero-mean noise": $\nabla f_i(w)=\nabla F(w)+\xi_i$ with $\mathbb E[\xi_i]=0$. On average it moves the right way, but every step carries a random deviation.

看最小點 $w^*=5$：此時 $\nabla F(5)=0$，但四個單筆梯度是 $5-a_i=3,\,1,\,-1,\,-3$，**都不是 0**（平均才是 0）。單筆梯度的變異數 $\sigma^2=(9+1+1+9)/4=5$，$\sigma=2.236$。若恰好抽到第 1 筆（$a=2$），用 $\eta=0.2$ 更新：$w=5-0.2\cdot3=4.4$。**明明已經在最小點，一步就被推離了**。所以固定步長的 SGD 不會停在最小點，而是在附近擺動。

Look at the minimum $w^*=5$: there $\nabla F(5)=0$, but the four single-example gradients are $5-a_i=3,\,1,\,-1,\,-3$, **none of them 0** (only their mean is). The variance of a single-example gradient is $\sigma^2=(9+1+1+9)/4=5$, so $\sigma=2.236$. If example 1 ($a=2$) happens to be drawn, an update with $\eta=0.2$ gives $w=5-0.2\cdot3=4.4$: **already at the minimum, one step pushes us away**. So SGD with a fixed step does not settle at the minimum but swings around it.

擺動多大？令誤差 $e_k=w^{(k)}-5$，更新是 $e_{k+1}=(1-\eta)\,e_k+\eta\,(a_{i_k}-5)$，其中 $a_{i_k}-5$ 的平均是 0、變異數是 $\sigma^2=5$。穩態變異數 $\mathrm{Var}(e)=\eta^2\sigma^2\big/\big(1-(1-\eta)^2\big)=\eta\,\sigma^2/(2-\eta)$，所以擺動的標準差是

How wide is the swing? Let the error be $e_k=w^{(k)}-5$; the update is $e_{k+1}=(1-\eta)\,e_k+\eta\,(a_{i_k}-5)$, where $a_{i_k}-5$ has mean 0 and variance $\sigma^2=5$. The steady-state variance is $\mathrm{Var}(e)=\eta^2\sigma^2\big/\big(1-(1-\eta)^2\big)=\eta\,\sigma^2/(2-\eta)$, so the standard deviation of the swing is

$$
\mathrm{std}(e)=\sigma\sqrt{\frac{\eta}{2-\eta}}
$$

$\eta=0.2$：$2.236\cdot\sqrt{0.2/1.8}=2.236\cdot0.3333=0.745$〔程式計算〕。用程式模擬 80 步（固定亂數種子 7），後 40 步的標準差約為 $0.72$，與公式吻合。學習率越小擺動越小，但收斂也越慢（KP02 的「太小」）。

For $\eta=0.2$: $2.236\cdot\sqrt{0.2/1.8}=2.236\cdot0.3333=0.745$ [computed by program]. A program simulation of 80 steps (fixed random seed 7) gives a standard deviation of about $0.72$ over the last 40 steps, in agreement with the formula. A smaller learning rate gives a smaller swing but slower convergence (the "too small" of KP02).

![SGD 與完整梯度下降的軌跡：灰線是完整梯度（η=0.2）平滑地收斂到 w*=5；紅線是固定 η=0.2 的 SGD，在 5 附近來回擺動；青綠線是 η_k=1/(k+1) 的 SGD，擺動逐漸縮小。（Trajectories of SGD and full gradient descent: the grey line is full gradient descent (η=0.2), converging smoothly to w*=5; the red line is SGD with fixed η=0.2, swinging around 5; the teal line is SGD with η_k=1/(k+1), whose swing shrinks.）](../assets/figures/lec03-kp07-sgd-noise.png)

**圖說 `lec03-kp07-sgd-noise`**：先看灰線：完整梯度沒有雜訊，平滑地走到 $w^*=5$；再看紅線：同樣的 $\eta=0.2$，SGD 到了 5 附近還是持續擺動；最後看青綠線：把步長改成遞減的 $1/(k+1)$，擺動隨 $k$ 變小。三條線從同一個起點 $w_0=0$ 出發。〔功能：1、2〕

**Caption `lec03-kp07-sgd-noise`**: Look at the grey line first: full gradient descent has no noise and walks smoothly to $w^*=5$. Then the red line: with the same $\eta=0.2$ SGD keeps swinging around 5. Finally the teal line: with the decaying step $1/(k+1)$ the swing shrinks as $k$ grows. All three start at $w_0=0$. [Function: 1, 2]

### 4. 遞減步長與取捨（A decaying step and the trade-offs）

怎麼讓 SGD 真的停在最小點？讓步長遞減：$\eta_k\to0$，雜訊的影響就跟著縮小。上面的例子取 $\eta_k=1/(k+1)$ 有一個特別乾淨的結果（`[補充]`）：$w^{(k+1)}=w^{(k)}-\tfrac{1}{k+1}\big(w^{(k)}-a_{i_k}\big)$ 正好把 $w^{(k+1)}$ 變成到目前為止抽到的所有 $a$ 的**樣本平均**，所以誤差大約是 $\sigma/\sqrt{k}$ 的量級，隨 $k$ 慢慢縮小（80 步後 $w=5.2$，離 5 還差 $0.2$，與 $2.236/\sqrt{80}=0.25$ 相符）。更一般的步長排程與收斂條件留給 KP09。

How can SGD really settle at the minimum? Let the step decay, $\eta_k\to0$, and the effect of the noise shrinks with it. In the example above, $\eta_k=1/(k+1)$ has an especially clean result (`[Added]`): $w^{(k+1)}=w^{(k)}-\tfrac{1}{k+1}\big(w^{(k)}-a_{i_k}\big)$ turns $w^{(k+1)}$ into exactly the **sample mean** of all the $a$ drawn so far, so the error is of order $\sigma/\sqrt{k}$ and shrinks slowly with $k$ (after 80 steps $w=5.2$, still $0.2$ from 5, in line with $2.236/\sqrt{80}=0.25$). More general step schedules and the convergence conditions are left to KP09.

SGD 的取捨可以整理成一句話：**便宜但吵**。

The trade-offs of SGD fit in one phrase: **cheap but noisy**.

| 項目 | 完整梯度下降 | SGD（單筆） |
|---|---|---|
| 每步成本 | $O(n)$ | $O(1)$ |
| 梯度估計 | 精確 | 無偏、但有雜訊 |
| 每步一定下降嗎 | 步長適當時是 | 不一定 |
| 在最小點附近 | 梯度為零、停得住 | 持續擺動，除非步長遞減 |

| Item | Full gradient descent | SGD (single example) |
|---|---|---|
| Cost per step | $O(n)$ | $O(1)$ |
| Gradient estimate | exact | unbiased but noisy |
| Does every step descend? | yes with a suitable step | not necessarily |
| Near the minimum | zero gradient, it can stop | keeps swinging unless the step decays |

來源提到雜訊「是逃離鞍點與局部極小的跳板」，並說 SGD 通常泛化更好。這在某些條件下可能成立，但並非保證，本講義不下結論，留到 KP12 討論平坦與泛化。單筆梯度的雜訊還是太大，所以實務上折衷成小批次：這是 KP08 的主題。

The source says the noise "is a springboard out of saddle points and local minima" and that SGD usually generalizes better. This may hold under some conditions but is no guarantee, so this note draws no conclusion and leaves flatness and generalization to KP12. A single example is still too noisy, so in practice we compromise with mini-batches: that is the topic of KP08.

### 本單元重點整理（KP07 summary）

**一句話重點**：SGD 每步只抽一筆樣本算梯度，期望等於完整梯度（無偏），成本從 $O(n)$ 降到 $O(1)$；代價是每步有雜訊，不保證每步下降，固定步長會在最小點附近擺動。

**One-sentence summary**: SGD draws one example per step, whose gradient equals the full gradient in expectation (unbiased), cutting the cost from $O(n)$ to $O(1)$; the price is noise at every step, no guarantee of descent per step, and a swing around the minimum with a fixed step.

**白話說明**：與其花大力氣算一次完美的梯度，不如用便宜但有點偏的猜測走很多步。猜測平均來說是對的，但每次都帶一點隨機的誤差，所以腳步有些搖晃；想停在谷底，就要讓步子越走越小。

**Plain explanation**: Rather than spend a huge effort on one perfect gradient, take many steps with a cheap, slightly off guess. The guess is right on average, but each one carries a bit of random error, so the stride wobbles; to settle at the valley floor the strides must keep shrinking.

**正式定義**：$F(w)=\tfrac1n\sum_if_i(w)$；$w^{(k+1)}=w^{(k)}-\eta_k\nabla f_{i_k}(w^{(k)})$；均勻抽樣時 $\mathbb E[\nabla f_i(w)]=\nabla F(w)$；固定步長在二次例子上的擺動 $\mathrm{std}=\sigma\sqrt{\eta/(2-\eta)}$。

**Formal definition**: $F(w)=\tfrac1n\sum_if_i(w)$; $w^{(k+1)}=w^{(k)}-\eta_k\nabla f_{i_k}(w^{(k)})$; $\mathbb E[\nabla f_i(w)]=\nabla F(w)$ for uniform sampling; the swing for a fixed step on the quadratic example is $\mathrm{std}=\sigma\sqrt{\eta/(2-\eta)}$.

**例子**：$a=(2,4,6,8)$、$w^*=5$：最小點的單筆梯度 $3,1,-1,-3$（平均 0，$\sigma=2.236$）；抽到第 1 筆、$\eta=0.2$：$w=4.4$；固定 $\eta=0.2$ 的擺動 $0.745$；$\eta_k=1/(k+1)$ 的 80 步後 $w=5.2$。逐步計算見第 2–4 節。

**Example**: $a=(2,4,6,8)$ and $w^*=5$: the single-example gradients at the minimum are $3,1,-1,-3$ (mean 0, $\sigma=2.236$); drawing example 1 with $\eta=0.2$ gives $w=4.4$; the swing for fixed $\eta=0.2$ is $0.745$; with $\eta_k=1/(k+1)$, $w=5.2$ after 80 steps. The computations are in sections 2–4.

**常見迷思**：

- 「無偏就是每步都下降」——期望方向對，但單步可以走錯，在最小點也會被推離。
- 「雜訊一定幫助逃出鞍點、模型一定更好」——只在某些條件下可能，不是保證。
- 「SGD 一定比完整梯度下降快」——每步便宜，但需要更多步，總成本取決於精度要求。
- 「固定步長的 SGD 會收斂到最小點」——只會在附近擺動。

**Common misconceptions**:

- "Unbiased means every step descends." The expected direction is right, but a single step can go wrong and be pushed away even at the minimum.
- "Noise always helps escape saddles and always makes the model better." It may under some conditions; it is no guarantee.
- "SGD is always faster than full gradient descent." Each step is cheap, but more steps are needed; the total cost depends on the required accuracy.
- "SGD with a fixed step converges to the minimum." It only swings around it.

**圖**：本單元一張圖，`lec03-kp07-sgd-noise`（SGD 與完整梯度的軌跡，功能 1、2），以 `assets/figures/src/lec03-kp0608-figures.py` 繪製（亂數種子固定）。

**Figure**: One figure in this unit, `lec03-kp07-sgd-noise` (the trajectories of SGD and full gradient descent; functions 1, 2), drawn by `assets/figures/src/lec03-kp0608-figures.py` with a fixed random seed.

**想深入**（選讀）：SGD 的雜訊來自**抽樣**，不是損失本身，所以它的變異數取決於各筆梯度的差異 $\sigma^2$。在完全可以被模型擬合的問題上（每一筆的最小點相同，也叫插值情形），最小點的 $\sigma^2=0$，固定步長的 SGD 也能收斂；這是大型網路過參數化時 SGD 還能表現良好的一個理論線索，留給 KP15。另外，SGD 的歷史是 Robbins 與 Monro（1951）的隨機逼近理論，LeCun、Bottou 等人在 1990 年代末把它確立為深度學習的預設基礎。

**Going deeper** (optional): The noise of SGD comes from **sampling**, not from the loss itself, so its variance depends on how much the per-example gradients differ, $\sigma^2$. In problems the model can fit perfectly (every example has the same minimizer, the interpolation case) $\sigma^2=0$ at the minimum and SGD with a fixed step also converges; this is one theoretical clue to why SGD still performs well for overparameterized large networks, left to KP15. Historically, SGD stems from the stochastic approximation theory of Robbins and Monro (1951), and LeCun, Bottou and others established it as the default basis of deep learning in the late 1990s.

**小練習**：

1. 在 $a=(2,4,6,8)$、$f_i(w)=\tfrac12(w-a_i)^2$ 的例子上，取 $w=5$、$\eta=0.1$，若抽到第 4 筆（$a=8$），算出新的 $w$。（提示：$\nabla f_4(5)=5-8=-3$。）
2. 驗證在 $w=3$ 時單筆梯度的平均等於 $\nabla F(3)$。（提示：$\nabla f_i(3)=3-a_i$；$\nabla F(3)=3-5$。）
3. 用 $\mathrm{std}=\sigma\sqrt{\eta/(2-\eta)}$ 算 $\eta=0.1$ 時的擺動，並說明為什麼學習率減半擺動就變小。（提示：$\sigma=2.236$。）

**Practice**:

1. On the example $a=(2,4,6,8)$, $f_i(w)=\tfrac12(w-a_i)^2$, take $w=5$ and $\eta=0.1$; if example 4 ($a=8$) is drawn, compute the new $w$. (Hint: $\nabla f_4(5)=5-8=-3$.)
2. Verify that at $w=3$ the mean of the single-example gradients equals $\nabla F(3)$. (Hint: $\nabla f_i(3)=3-a_i$ and $\nabla F(3)=3-5$.)
3. Use $\mathrm{std}=\sigma\sqrt{\eta/(2-\eta)}$ to compute the swing at $\eta=0.1$ and explain why halving the learning rate shrinks it. (Hint: $\sigma=2.236$.)

**解題示範**（以 $\eta=0.5$ 示範第 3 題的算法）：

1. 單筆梯度的變異數：在最小點是 $3,1,-1,-3$，$\sigma^2=(9+1+1+9)/4=5$，$\sigma=2.236$。
2. 代入公式：$\mathrm{std}=\sigma\sqrt{\eta/(2-\eta)}=2.236\cdot\sqrt{0.5/1.5}=2.236\cdot0.5774=1.291$。
3. 比較：$\eta=0.2$ 時是 $0.745$，$\eta=0.5$ 時是 $1.291$，學習率越大擺動越大。
4. 檢查：$\eta\to0$ 時 $\sqrt{\eta/(2-\eta)}\to0$，擺動消失，與遞減步長能收斂的說法一致。

**Worked solution** (the method of exercise 3, shown with $\eta=0.5$):

1. Variance of a single-example gradient: at the minimum the gradients are $3,1,-1,-3$, so $\sigma^2=(9+1+1+9)/4=5$ and $\sigma=2.236$.
2. Substitute: $\mathrm{std}=\sigma\sqrt{\eta/(2-\eta)}=2.236\cdot\sqrt{0.5/1.5}=2.236\cdot0.5774=1.291$.
3. Compare: $0.745$ at $\eta=0.2$ and $1.291$ at $\eta=0.5$: the larger the learning rate, the larger the swing.
4. Check: as $\eta\to0$, $\sqrt{\eta/(2-\eta)}\to0$ and the swing vanishes, consistent with the claim that a decaying step converges.

### 來源更正與取捨（Corrections to the source）

- 來源說 SGD 的雜訊是「逃出鞍點與局部極小的神賜跳板」，並說「走得更快、模型更好」。這些只在某些條件下可能成立，本講義不當成結論；「每步更便宜」也不等於「達到同樣精度的總成本更低」。
- 來源把 SGD 的複雜度說成「每步 $O(1)$、與 $n$ 完全解耦」。精確的說法是每步成本與 $n$ 無關，但要達到指定精度需要的步數仍和問題有關。
- 來源的比喻與口訣（「貴的正確不如便宜的大概對」）本講義保留其意，以「便宜但吵」概括。
- 來源的 SGD 歷史（Robbins 與 Monro 1951）保留在「想深入」。

- The source says the noise of SGD is "a godsend springboard out of saddle points and local minima" and that SGD goes "faster and gives a better model". These may hold only under some conditions and are not taken as conclusions here; "cheaper per step" also does not mean "cheaper in total to reach the same accuracy".
- The source describes the complexity of SGD as "$O(1)$ per step, fully decoupled from $n$". Precisely, the cost per step is independent of $n$, but the number of steps needed for a given accuracy still depends on the problem.
- The source's slogan ("a cheap rough answer beats an expensive exact one") is kept in spirit as "cheap but noisy".
- The source's history of SGD (Robbins and Monro, 1951) is kept under "Going deeper".

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 隨機梯度下降 | stochastic gradient descent (SGD) |
| 經驗風險 | empirical risk |
| 完整梯度 | full gradient |
| 單樣本梯度 | single-example gradient |
| 無偏估計 | unbiased estimator |
| 期望值 | expectation |
| 變異數 | variance |
| 學習率排程 | step-size schedule |
| 隨機逼近 | stochastic approximation |
