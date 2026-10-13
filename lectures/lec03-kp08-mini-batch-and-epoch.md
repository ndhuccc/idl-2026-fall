# 第3週 KP08：小批次與 epoch（Mini-batches and epochs）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第3章「反向傳播與最佳化」的第八個知識點（KP08），承接 KP07 留下的問題：單筆梯度的雜訊太大，折衷怎麼做？先備知識是期望與變異數、KP01–KP07；第一次出現的符號都會定義，所有數字都可以自己重算。內容依 CCC-LNS 第3章 KP08 的主文與投影片消化後重新撰寫，讀取日期 2026-10-14；來源沒有的內容標示 `[補充]`。

This is the eighth knowledge point (KP08) of Chapter 3. It picks up the question KP07 left open: the noise of a single-example gradient is too large, so what is the compromise? It assumes expectation and variance and KP01–KP07; every symbol is defined when it first appears and every number can be recomputed. The text is rewritten from the KP08 main text and slides of CCC-LNS Chapter 3 (accessed on 2026-10-14); content that the source does not contain is marked `[Added]`.

[下載本單元投影片](../slides/lec03-kp08-mini-batch-and-epoch.jbd.json)

## KP08 小批次與 epoch（Mini-batches and epochs）

### 學習目標（Learning objectives）

讀完之後，你能說出單筆與全批各自的問題，並寫出小批次梯度與它的方差約為 $\sigma^2/b$ 的原因與限制；能把資料洗牌後切成互斥的批次，算出一個 epoch 有幾次更新；能用一組小數字算出批次估計的平均與標準差；也能說明 epoch 不是通用的算力單位，以及批次大小的取捨。

After reading, you can state the problem of the single-example and full-batch extremes and write the mini-batch gradient together with the reason and limits of its variance $\approx\sigma^2/b$; shuffle the data and cut it into disjoint batches and compute how many updates one epoch has; compute the mean and the standard deviation of a batch estimate on a small set of numbers; and explain why an epoch is not a universal unit of compute and what the batch-size trade-off is.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $n$ | 訓練樣本的筆數 | number of training examples |
| $b$ | 批次大小（每批幾筆） | batch size (examples per batch) |
| $B$ | 一個批次（樣本編號的集合） | a batch (a set of example indices) |
| $g_B(w)$ | 批次 $B$ 的平均梯度 | the average gradient of batch $B$ |
| $\sigma^2$ | 單筆梯度的變異數 | variance of a single-example gradient |
| $\lfloor n/b\rfloor$ | $n/b$ 向下取整：完整批次的個數 | $n/b$ rounded down: the number of full batches |

### 1. 兩個極端，與黃金折衷（Two extremes and the compromise）

KP07 的 SGD 每步抽 $b=1$ 筆：便宜、但每步方差大，路徑抖得像醉漢，而且一次只算一筆，GPU 上成千上萬的運算單元大部分閒著。另一個極端 $b=n$ 是完整梯度下降：沒有雜訊，但每步要掃過全部資料，記憶體也常常裝不下。

The SGD of KP07 draws $b=1$ example per step: cheap, but with a large variance per step, a path as shaky as a drunkard's walk, and a single example at a time leaves most of the thousands of compute units of a GPU idle. The other extreme, $b=n$, is full gradient descent: no noise, but each step scans all the data and the memory often cannot hold it.

**小批次**取中間：每步隨機取 $b$ 筆（$1<b<n$，常見 32 到 256），用這一批的平均梯度更新：

**Mini-batches** take the middle: each step takes $b$ examples at random ($1<b<n$, typically 32 to 256) and updates with the average gradient of the batch:

$$
g_B(w)=\frac1{|B|}\sum_{i\in B}\nabla f_i(w),\qquad w^{(k+1)}=w^{(k)}-\eta_k\,g_B\big(w^{(k)}\big)
$$

好處有兩個：（1）批次平均把方差壓低；（2）$b$ 筆的前向與反向可以打包成矩陣乘法，一次交給 GPU 並行算完。它仍然是無偏的：均勻抽樣時 $\mathbb E[g_B(w)]=\nabla F(w)$（KP07 的論證原樣成立）。

There are two benefits: (1) averaging over the batch lowers the variance; (2) the forward and backward passes of $b$ examples can be packed into matrix multiplications and run in parallel on a GPU. It stays unbiased: with uniform sampling $\mathbb E[g_B(w)]=\nabla F(w)$ (the argument of KP07 carries over unchanged).

| 批次大小 | 每步成本 | 梯度雜訊 | 硬體利用 |
|---|---|---|---|
| $b=1$（單筆） | 最低 | 最大 | 差（一次一筆） |
| $1<b<n$（小批次） | 中 | 中 | 好（矩陣運算並行） |
| $b=n$（全批） | 最高 | 零 | 常裝不下記憶體 |

| Batch size | Cost per step | Gradient noise | Hardware use |
|---|---|---|---|
| $b=1$ (single) | lowest | largest | poor (one example at a time) |
| $1<b<n$ (mini-batch) | medium | medium | good (parallel matrix operations) |
| $b=n$ (full batch) | highest | zero | often exceeds memory |

### 2. 資料怎麼分，什麼叫一個 epoch（How to cut the data and what an epoch is）

標準做法是先把整個訓練集**隨機洗牌**，再依序切成 $\lfloor n/b\rfloor$ 個大小為 $b$ 的**互斥**批次 $B_1,B_2,\dots$。互斥有兩層意思：所有批次的聯集是整個資料集（扣掉湊不滿的尾巴），而任兩個批次沒有共同的樣本：

The standard practice is to **shuffle** the training set at random and then cut it, in order, into $\lfloor n/b\rfloor$ **disjoint** batches $B_1,B_2,\dots$ of size $b$. Disjoint has two meanings: the union of the batches is the whole data set (minus a leftover tail that does not fill a batch) and any two batches share no example:

$$
\bigcup_{i}B_i=\mathcal D\ (\text{up to the leftover}),\qquad B_i\cap B_j=\varnothing\ (i\ne j)
$$

於是在一輪裡，每個樣本**恰好被看過一次**。把所有批次各用一次，叫做一個 **epoch**（訓練週期）。一個 epoch 有 $\lfloor n/b\rfloor$ 次更新。例：$n=1000$、$b=32$：$\lfloor1000/32\rfloor=31$ 個完整批次，共用掉 $31\cdot32=992$ 筆，剩下 $1000-992=8$ 筆。常見處理有兩種：把剩下的 8 筆當作一個較小的最後批次（共 32 次更新，最後一批只有 8 筆、雜訊較大），或設 `drop_last=True` 把它丟掉（31 次更新，下一個 epoch 重新洗牌，被丟的 8 筆會換一批）。每個 epoch 開始前都要**重新洗牌**，避免固定的樣本順序變成隱藏的條件。

So within one round each example is **seen exactly once**. Using every batch once is called an **epoch**. One epoch has $\lfloor n/b\rfloor$ updates. Example: $n=1000$, $b=32$: $\lfloor1000/32\rfloor=31$ full batches use $31\cdot32=992$ examples and leave $1000-992=8$. Two treatments are common: keep the 8 as a smaller last batch (32 updates; the last batch has only 8 examples and more noise), or set `drop_last=True` and discard them (31 updates; the next epoch reshuffles, so a different 8 are dropped). **Reshuffle before every epoch** so that a fixed example order does not become a hidden condition.

另一種是**有放回**的抽樣：每步從整個資料集獨立均勻抽 $b$ 筆，批次之間可以重疊，互斥的性質不再成立。抽 $n$ 次（和資料集一樣大）時，約有 $1-1/e\approx63.2\%$ 的獨特樣本被抽中，其餘沒被抽到。深度學習偏好無放回，因為每筆資料在一輪裡得到相同次數的使用，也利於連續讀取記憶體與預取資料。

The other option is sampling **with replacement**: each step draws $b$ examples independently and uniformly from the whole data set, batches may overlap and disjointness no longer holds. When $n$ draws are made (as many as the data set is large), about $1-1/e\approx63.2\%$ of the distinct examples are hit and the rest are never drawn. Deep learning prefers sampling without replacement because each example is used equally often in a round, which also suits sequential memory reads and prefetching.

### 3. 批次估計有多準（How accurate is a batch estimate）

用 KP07 的四筆資料算：在 $w=5$，單筆梯度是 $g=(3,\,1,\,-1,\,-3)$，平均 $0$，$\sigma^2=5$。

Use the four examples of KP07: at $w=5$ the single-example gradients are $g=(3,\,1,\,-1,\,-3)$ with mean $0$ and $\sigma^2=5$.

- 批次 $\{1,2\}$：$(3+1)/2=2$；批次 $\{3,4\}$：$(-1-3)/2=-2$；完整平均 $=0$。小批次的估計**可以和完整梯度的方向完全相反**（$+2$ 與 $-2$ 的方向相反），但兩個批次平均起來剛好是 0。
- 六種可能的兩筆批次（無放回）平均：$2,\,1,\,0,\,0,\,-1,\,-2$，標準差 $\sqrt{(4+1+0+0+1+4)/6}=\sqrt{10/6}=1.291$；單筆（$b=1$）標準差是 $\sqrt5=2.236$；$b=4$（全批）標準差是 $0$。

- Batch $\{1,2\}$: $(3+1)/2=2$; batch $\{3,4\}$: $(-1-3)/2=-2$; the full average is $0$. A mini-batch estimate **can point opposite to the full gradient** ($+2$ and $-2$ point in opposite directions), yet the two batches average to exactly 0.
- The six possible two-example batch means (without replacement) are $2,\,1,\,0,\,0,\,-1,\,-2$ with standard deviation $\sqrt{(4+1+0+0+1+4)/6}=\sqrt{10/6}=1.291$; a single example ($b=1$) has standard deviation $\sqrt5=2.236$; $b=4$ (full batch) has $0$.

規律是**方差隨批次大小下降**。對**獨立（有放回）**抽樣，批次平均的方差是 $\sigma^2/b$（標準差 $\sigma/\sqrt b$）。對**無放回**抽樣，還要乘上有限母體修正 $\dfrac{n-b}{n-1}$：$\mathrm{Var}=\dfrac{\sigma^2}{b}\cdot\dfrac{n-b}{n-1}$。驗算：$n=4$、$b=2$：$\tfrac52\cdot\tfrac23=1.667$，$\sqrt{1.667}=1.291$，與六種批次平均的標準差一致〔程式計算〕。$b=n$ 時修正為 0，批次就是整個資料集，沒有抽樣誤差。

The pattern is that **variance falls with batch size**. For **independent (with replacement)** sampling the variance of the batch mean is $\sigma^2/b$ (standard deviation $\sigma/\sqrt b$). For sampling **without replacement** there is also the finite-population correction $\dfrac{n-b}{n-1}$: $\mathrm{Var}=\dfrac{\sigma^2}{b}\cdot\dfrac{n-b}{n-1}$. Check: $n=4$, $b=2$: $\tfrac52\cdot\tfrac23=1.667$ and $\sqrt{1.667}=1.291$, matching the standard deviation of the six batch means [computed by program]. At $b=n$ the correction is 0: the batch is the whole data set and there is no sampling error.

![批次梯度的標準差隨批次大小下降（n=64 的例子，單筆梯度為 −31.5 到 31.5 的等差數列，σ=18.47）：藍線是有放回的 σ/√b，橘線是無放回（含有限母體修正），b=n 時降到 0。（The standard deviation of the batch gradient falls with batch size (an n=64 example whose single-example gradients are an arithmetic sequence from −31.5 to 31.5, σ=18.47): the blue line is σ/√b with replacement, the orange line is without replacement with the finite-population correction, dropping to 0 at b=n.）](../assets/figures/lec03-kp08-batch-std.png)

**圖說 `lec03-kp08-batch-std`**：先看橘線的四個點（$b=1,8,32,64$）：標準差 $18.47\to6.16\to2.33\to0$；再比較藍線：有放回的 $\sigma/\sqrt b$ 降得比較慢，$b=64$ 時仍有 $2.31$。這是 `[補充]` 的數值例，單筆梯度取 $-31.5,\dots,31.5$ 共 64 個。兩條線在 $b$ 很小時幾乎重合，差別出現在 $b$ 接近 $n$ 時。〔功能：1、2〕

**Caption `lec03-kp08-batch-std`**: Look at the four orange points ($b=1,8,32,64$) first: the standard deviation goes $18.47\to6.16\to2.33\to0$. Then compare the blue line: $\sigma/\sqrt b$ with replacement falls more slowly and is still $2.31$ at $b=64$. This is an `[Added]` numerical example with 64 single-example gradients from $-31.5$ to $31.5$. The two lines nearly coincide for small $b$ and differ when $b$ approaches $n$. [Function: 1, 2]

### 4. 批次大小的取捨，與 epoch 的限制（The batch-size trade-off and the limits of an epoch）

**批次大小控制三件事**：雜訊（越大越小）、記憶體（越大越吃）、每個 epoch 的更新次數（$\lfloor n/b\rfloor$，越大越少）。沒有放諸四海皆準的最佳值：批次大，每步精確、硬體效率高，但每個 epoch 的更新次數少；批次小，更新頻繁、帶雜訊。這也是來源說「小批次以精確度換取更新頻率」的意思。

**Batch size controls three things**: noise (larger means less), memory (larger means more) and the number of updates per epoch ($\lfloor n/b\rfloor$, larger means fewer). There is no universal best value: a large batch gives exact steps and high hardware efficiency but few updates per epoch; a small batch gives frequent, noisy updates. This is what the source means by "mini-batches trade exactness for update frequency".

**epoch 不是通用的算力單位。**「訓練 10 個 epoch」在不同的批次大小下，更新次數可以差很多：$n=1000$ 時，$b=10$ 是 1000 次更新，$b=100$ 是 100 次更新。比較兩個訓練設定時，要同時看**更新步數**與**看過的樣本總數**，不能只看 epoch 數。

**An epoch is not a universal unit of compute.** "Train for 10 epochs" can mean very different numbers of updates at different batch sizes: with $n=1000$, $b=10$ is 1000 updates and $b=100$ is 100. When comparing two training setups, look at both the **number of updates** and the **total number of examples seen**, not only the epoch count.

來源還說單筆 SGD 的 GPU 利用率「常低於 5%」、完整梯度「容易收斂到尖銳極小值」。前者沒有依據，後者在文獻上有爭議（要看批次大小與學習率怎麼搭配），本講義都不採用，尖銳與平坦的問題留到 KP12。接下來的問題是：雜訊存在時，「收斂」到底是什麼意思？這是 KP09 的主題。

The source also says that the GPU utilization of single-example SGD is "often below 5%" and that full-batch gradients "easily converge to sharp minima". The former has no basis and the latter is disputed in the literature (it depends on how batch size and learning rate are paired), so neither is adopted here; sharp versus flat is left to KP12. The next question: with noise present, what does "convergence" actually mean? That is the topic of KP09.

### 本單元重點整理（KP08 summary）

**一句話重點**：小批次每步取 $b$ 筆算平均梯度，方差約降為 $\sigma^2/b$（無放回再乘有限母體修正），又能用 GPU 矩陣運算；洗牌後切成互斥批次，每批用一次叫一個 epoch，共 $\lfloor n/b\rfloor$ 次更新。

**One-sentence summary**: A mini-batch takes $b$ examples per step and averages their gradients, lowering the variance to about $\sigma^2/b$ (times the finite-population correction without replacement) while using GPU matrix operations; after shuffling, disjoint batches used once each make one epoch with $\lfloor n/b\rfloor$ updates.

**白話說明**：單筆太吵，全批太慢也太占記憶體，小批次是折衷：多看幾筆把雜訊平均掉一些，又能一次並行算完。把資料洗牌、切成幾堆，每堆輪流用一次，就是一個 epoch。

**Plain explanation**: A single example is too noisy and the full batch is too slow and memory-hungry; a mini-batch is the compromise: look at a few examples to average out some noise, and compute them in parallel at once. Shuffle the data, cut it into piles and use each pile in turn once: that is one epoch.

**正式定義**：$g_B(w)=\tfrac1{|B|}\sum_{i\in B}\nabla f_i(w)$；批次互斥 $B_i\cap B_j=\varnothing$；epoch $=\lfloor n/b\rfloor$ 個批次各用一次；獨立抽樣 $\mathrm{Var}(g_B)=\sigma^2/b$，無放回 $\mathrm{Var}(g_B)=\tfrac{\sigma^2}{b}\tfrac{n-b}{n-1}$。

**Formal definition**: $g_B(w)=\tfrac1{|B|}\sum_{i\in B}\nabla f_i(w)$; disjoint batches $B_i\cap B_j=\varnothing$; an epoch is $\lfloor n/b\rfloor$ batches used once each; independent sampling gives $\mathrm{Var}(g_B)=\sigma^2/b$ and without replacement $\mathrm{Var}(g_B)=\tfrac{\sigma^2}{b}\tfrac{n-b}{n-1}$.

**例子**：梯度 $(3,1,-1,-3)$：批次 $\{1,2\}$ 是 $2$，$\{3,4\}$ 是 $-2$，完整是 $0$；$b=1,2,4$ 的標準差 $2.236,\,1.291,\,0$；$n=1000$、$b=32$：31 個完整批次，剩 8 筆。逐步計算見第 2–3 節。

**Example**: gradients $(3,1,-1,-3)$: batch $\{1,2\}$ gives $2$, $\{3,4\}$ gives $-2$ and the full average is $0$; the standard deviations for $b=1,2,4$ are $2.236,\,1.291,\,0$; $n=1000$, $b=32$: 31 full batches with 8 left over. The computations are in sections 2–3.

**常見迷思**：

- 「批次越大越好」——越大每步越精確，但更新次數少、記憶體吃緊，也不是越大泛化越好。
- 「方差一律是 $\sigma^2/b$」——那是獨立抽樣的結果，無放回還要乘有限母體修正。
- 「小批次的估計一定接近完整梯度」——單一批次可以方向完全相反。
- 「epoch 數相同，訓練量就相同」——更新步數與看過的樣本數才是量。

**Common misconceptions**:

- "A larger batch is better." A larger batch gives exacter steps but fewer updates and tighter memory, and larger is not always better for generalization.
- "The variance is always $\sigma^2/b$." That is the result for independent sampling; without replacement the finite-population correction applies too.
- "A mini-batch estimate is always close to the full gradient." A single batch can point in the opposite direction.
- "Equal epoch counts mean equal training effort." The number of updates and the number of examples seen are what count.

**圖**：本單元一張圖，`lec03-kp08-batch-std`（批次梯度的標準差隨批次大小下降，功能 1、2），以 `assets/figures/src/lec03-kp0608-figures.py` 繪製。分批與 epoch 的流程在簡報中以原生節點與連線呈現。

**Figure**: One figure in this unit, `lec03-kp08-batch-std` (the standard deviation of the batch gradient falling with batch size; functions 1, 2), drawn by `assets/figures/src/lec03-kp0608-figures.py`. The flow of batching and epochs is drawn with native nodes and connectors in the deck.

**想深入**（選讀）：有限母體修正 $\tfrac{n-b}{n-1}$ 可以這樣推：無放回取 $b$ 筆的平均，$b$ 個樣本互相**負相關**（總和固定時，一個偏大另一個就偏小），所以方差比獨立抽樣小；$b=n$ 時整個資料集的平均沒有隨機性，方差為 0。批次大小與學習率常常需要一起調：批次變大、雜訊變小時，常可用較大的學習率（所謂的線性縮放經驗法則），但這是經驗而非定理，細節留給進階課程。

**Going deeper** (optional): The finite-population correction $\tfrac{n-b}{n-1}$ can be seen like this: the $b$ examples drawn without replacement are **negatively correlated** (with a fixed total, one being larger makes another smaller), so the variance is below that of independent draws; at $b=n$ the mean over the whole data set has no randomness and the variance is 0. Batch size and learning rate often need to be tuned together: when the batch grows and the noise shrinks a larger learning rate often works (the so-called linear scaling rule of thumb), but that is experience, not a theorem, and the details are left to advanced courses.

**小練習**：

1. $n=2400$、$b=64$：完整批次有幾個？剩幾筆？用 `drop_last=True` 與保留小批次，一個 epoch 各有幾次更新？（提示：$\lfloor2400/64\rfloor$ 與餘數。）
2. 單筆梯度為 $(6,\,2,\,-2,\,-6)$，算 $\sigma^2$，列出所有 $b=2$ 的無放回批次平均，並用公式驗算它們的標準差。（提示：$\sigma^2=(36+4+4+36)/4$；公式 $\tfrac{\sigma^2}{2}\cdot\tfrac{n-b}{n-1}$。）
3. 兩個訓練設定：（甲）$n=1000$、$b=10$、10 個 epoch；（乙）$n=1000$、$b=100$、10 個 epoch。比較它們的更新次數與看過的樣本數。（提示：更新次數 $=\lfloor n/b\rfloor\times$ epoch 數。）

**Practice**:

1. $n=2400$, $b=64$: how many full batches are there, and how many examples are left? How many updates does one epoch have with `drop_last=True` and with a smaller last batch kept? (Hint: $\lfloor2400/64\rfloor$ and the remainder.)
2. The single-example gradients are $(6,\,2,\,-2,\,-6)$: compute $\sigma^2$, list all without-replacement batch means for $b=2$ and verify their standard deviation with the formula. (Hint: $\sigma^2=(36+4+4+36)/4$; the formula $\tfrac{\sigma^2}{2}\cdot\tfrac{n-b}{n-1}$.)
3. Two training setups: (A) $n=1000$, $b=10$, 10 epochs; (B) $n=1000$, $b=100$, 10 epochs. Compare their numbers of updates and of examples seen. (Hint: updates $=\lfloor n/b\rfloor\times$ number of epochs.)

**解題示範**（以 $n=1000$、$b=32$ 示範第 1 題的算法）：

1. 完整批次：$\lfloor1000/32\rfloor=\lfloor31.25\rfloor=31$。
2. 用掉的樣本：$31\cdot32=992$，剩 $1000-992=8$。
3. `drop_last=True`：每個 epoch 31 次更新，這 8 筆這一輪不用（下一輪重新洗牌，換一批被丟）。
4. 保留小批次：31 次完整更新再加 1 次只有 8 筆的更新，共 32 次，最後一批的雜訊較大。
5. 檢查：32 次更新用掉 $31\cdot32+8=1000$ 筆，恰好是全部資料各一次。

**Worked solution** (the method of exercise 1, shown with $n=1000$ and $b=32$):

1. Full batches: $\lfloor1000/32\rfloor=\lfloor31.25\rfloor=31$.
2. Examples used: $31\cdot32=992$, leaving $1000-992=8$.
3. `drop_last=True`: 31 updates per epoch; the 8 are not used this round (the next round reshuffles and a different 8 are dropped).
4. Keeping the small batch: 31 full updates plus 1 update with only 8 examples, 32 in all, and the last batch is noisier.
5. Check: the 32 updates use $31\cdot32+8=1000$ examples, exactly every example once.

### 來源更正與取捨（Corrections to the source）

- 來源說批次梯度的方差是 $\sigma^2/b$。這只對獨立抽樣成立；本講義補上無放回抽樣的有限母體修正 $\tfrac{n-b}{n-1}$（$b=n$ 時為 0），並用四筆資料驗算。
- 來源說單筆 SGD 的 GPU 利用率「常低於 5%」，沒有依據，刪除；來源說全批次「容易收斂到尖銳極小值、泛化欠佳」是有爭議的說法，本講義不採用，留到 KP12。
- 來源說「有放回約有 63.2% 的獨特樣本被抽中」，這是抽 $n$ 次時的結果（$1-(1-1/n)^n\to1-1/e$），本講義補上這個條件。
- 來源的 Epoch 定義（把全部 $\lfloor n/b\rfloor$ 個互斥批次各用一次）保留；補上「epoch 不是通用的算力單位」，比較時要看更新步數與看過的樣本數。

- The source says the variance of a batch gradient is $\sigma^2/b$. That holds only for independent sampling; this note adds the finite-population correction $\tfrac{n-b}{n-1}$ for sampling without replacement (0 at $b=n$) and checks it on four examples.
- The source says the GPU utilization of single-example SGD is "often below 5%", which has no basis and is dropped; the source's claim that full batches "easily converge to sharp minima with poor generalization" is disputed and not adopted here, left for KP12.
- The source says "with replacement about 63.2% of distinct examples are hit"; that is the result of $n$ draws ($1-(1-1/n)^n\to1-1/e$), a condition this note adds.
- The source's definition of an epoch (using all $\lfloor n/b\rfloor$ disjoint batches once each) is kept; this note adds that "an epoch is not a universal unit of compute" and that comparisons should look at updates and examples seen.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 小批次 | mini-batch |
| 批次大小 | batch size |
| 訓練週期 | epoch |
| 洗牌 | shuffle |
| 互斥 | disjoint |
| 有放回／無放回抽樣 | sampling with / without replacement |
| 有限母體修正 | finite-population correction |
| 矩陣運算並行 | parallel matrix operations |
