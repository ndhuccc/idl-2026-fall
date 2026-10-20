# 第4週 KP16：Bagging 與變異降低（Bagging and variance reduction）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第4週「正則化與過擬合」的第十六個知識點（KP16）。KP11 到 KP15 都在「改目標函數」或「少訓練」來壓低模型複雜度；這一單元換一條路：不改單一模型，而是訓練很多個模型再取平均，用平均來壓低變異。先備知識是 KP02（MSE 等於變異加偏差平方）與 KP03（線性組合的變異數公式，KP03 已經把本單元要用的式子留在最後）；第一次出現的符號都會定義，所有數字都可以自己重算。內容依 CCC-LNS 第5章 KP16 主文與教授祕笈（其中 Bagging 的部分）消化後重新撰寫，讀取日期 2026-10-20；來源沒有的內容標示 `[補充]`。

This is the sixteenth knowledge point (KP16) of Week 4, "Regularization and overfitting". KP11 to KP15 lowered model complexity by changing the objective or by training less; this unit takes a different road: leave a single model alone, train many models and average them, and let the average lower the variance. It assumes KP02 (MSE equals variance plus squared bias) and KP03 (the variance of a linear combination; KP03 ended by leaving the very formula used here); every symbol is defined when it first appears and every number can be recomputed. The text is rewritten from the KP16 main text and professor's guide of CCC-LNS Chapter 5 (the bagging part), accessed on 2026-10-20; content that the source does not contain is marked `[Added]`.

[下載本單元投影片](../slides/lec04-kp16-bagging-variance-reduction.jbd.json)

## KP16 Bagging 與變異降低（Bagging and variance reduction）

### 學習目標（Learning objectives）

讀完之後，你能從 KP03 的線性組合公式，一步一步推出 $k$ 個模型的平均其變異數為 $\frac{s}{k}+\frac{k-1}{k}c$；能說出 $c=0$、$c=s$ 兩個極端各給多少、為什麼平均後的變異數有一個降不掉的下限 $c$；能說明 bootstrap 抽樣為什麼讓模型彼此不同、一個 bootstrap 樣本約含 $63.2\%$ 的相異資料；也能指出 bagging 降的是變異而不是偏差，以及它在什麼情形下幫不上忙。

After reading, you can derive step by step from the linear-combination formula of KP03 that the average of $k$ models has variance $\frac{s}{k}+\frac{k-1}{k}c$; say what the two extremes $c=0$ and $c=s$ give and why the variance of the average has a floor $c$ that cannot be lowered; explain why bootstrap sampling makes the models differ and why a bootstrap sample contains about $63.2\%$ distinct points; and point out that bagging lowers variance, not bias, and when it does not help.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $h_j(\mathbf{x})$ | 第 $j$ 個模型在輸入 $\mathbf{x}$ 的預測，$j=1,\dots,k$ | prediction of the $j$-th model at input $\mathbf{x}$, $j=1,\dots,k$ |
| $k$ | 平均的模型個數 | number of models averaged |
| $\hat f(\mathbf{x})$ | $k$ 個模型的平均預測 | the average prediction of the $k$ models |
| $e_j$ | 第 $j$ 個模型的誤差（預測減去它的平均） | error of the $j$-th model (its prediction minus its mean) |
| $s$ | 每個模型誤差的變異數（假設各模型相同） | variance of each model's error (assumed equal for all models) |
| $c$ | 任兩個模型誤差的共變異數（假設任兩個都相同） | covariance of the errors of any two models (assumed the same for every pair) |
| $n$ | 訓練資料筆數 | number of training points |

### 1. 平均為什麼能降低變異（Why averaging lowers variance）

KP15 說早停法是用「少訓練」換正則化。這裡換一個想法：一個模型每次訓練出來都有點不同，是 KP02 說的變異；如果同時訓練 $k$ 個，再把預測平均，抖動會不會互相抵銷？

KP15 said that early stopping trades "less training" for regularization. Here is another idea: a model comes out a little different each time it is trained, which is the variance of KP02. If we train $k$ of them and average the predictions, will the wobbles cancel each other?

設 $k$ 個模型的預測是 $h_1(\mathbf{x}),\dots,h_k(\mathbf{x})$，平均預測是

Let the predictions of $k$ models be $h_1(\mathbf{x}),\dots,h_k(\mathbf{x})$; the average prediction is

$$
\hat f(\mathbf{x})=\frac1k\sum_{j=1}^{k}h_j(\mathbf{x})
$$

把每個模型的預測寫成「它的平均」加上誤差 $e_j$。看變異數時，常數不影響，所以只需看誤差的平均 $\frac1k\sum_j e_j$。為了能算，做兩個假設：每個 $e_j$ 的變異數都是 $s$；任兩個不同模型的誤差，共變異數都是同一個 $c$。套用 KP03 的線性組合公式，係數全部取 $a_j=\frac1k$：

Write the prediction of each model as "its mean" plus an error $e_j$. When we look at variance, constants do not matter, so we only need the average of the errors $\frac1k\sum_j e_j$. To make it computable, make two assumptions: every $e_j$ has variance $s$, and every pair of different models has the same covariance $c$ between their errors. Apply the linear-combination formula of KP03 with all coefficients $a_j=\frac1k$:

$$
\mathrm{Var}\Big(\frac1k\sum_{j=1}^{k}e_j\Big)=\sum_{j=1}^{k}\Big(\frac1k\Big)^{2}\mathrm{Var}(e_j)+\sum_{i\neq j}\frac1k\cdot\frac1k\,\mathrm{Cov}(e_i,e_j)
$$

第一項有 $k$ 個，每個是 $\frac{s}{k^2}$，加起來 $\frac{s}{k}$。第二項是「有序對」$(i,j)$，$i\neq j$，共 $k(k-1)$ 個（$(1,2)$ 與 $(2,1)$ 算兩個，與 KP03 說的一樣），每個是 $\frac{c}{k^2}$，加起來 $\frac{k(k-1)}{k^2}c=\frac{k-1}{k}c$。所以

The first sum has $k$ terms, each $\frac{s}{k^2}$, together $\frac{s}{k}$. The second sum runs over ordered pairs $(i,j)$ with $i\neq j$, which are $k(k-1)$ of them ($(1,2)$ and $(2,1)$ count as two, as KP03 said), each $\frac{c}{k^2}$, together $\frac{k(k-1)}{k^2}c=\frac{k-1}{k}c$. So

$$
\mathrm{Var}\big(\hat f(\mathbf{x})\big)=\frac{s}{k}+\frac{k-1}{k}\,c
$$

這個式子還有一個更好讀的寫法：$\frac{s}{k}+\frac{k-1}{k}c=c+\frac{s-c}{k}$。第一項 $c$ 與 $k$ 無關，是**降不掉的下限**；第二項 $\frac{s-c}{k}$ 才隨 $k$ 變大而消失。兩個極端：

There is a more readable way to write it: $\frac{s}{k}+\frac{k-1}{k}c=c+\frac{s-c}{k}$. The first term $c$ does not depend on $k$ and is the **floor that cannot be lowered**; only the second term $\frac{s-c}{k}$ vanishes as $k$ grows. The two extremes:

- 模型誤差互相獨立（$c=0$）：$\mathrm{Var}=\frac{s}{k}$，加一倍的模型，變異數減半，$k\to\infty$ 時趨近 0。
- 模型幾乎一樣（$c=s$）：$\mathrm{Var}=s$，不管取平均多少個，完全沒有改善。

- Independent errors ($c=0$): $\mathrm{Var}=\frac{s}{k}$; double the models and the variance halves, and it goes to 0 as $k\to\infty$.
- Nearly identical models ($c=s$): $\mathrm{Var}=s$; averaging any number of them improves nothing.

**數值例（來源的例子）**：$s=1$、$k=10$。若 $c=0$，$\mathrm{Var}=\frac1{10}=0.1$。若 $c=0.8$，$\mathrm{Var}=0.1+0.9\times0.8=0.82$，只從 $1$ 降到 $0.82$。用另一種寫法驗算：$c+\frac{s-c}{k}=0.8+\frac{0.2}{10}=0.82$。再多給一個 $c=0.3$ 的情形：$0.3+\frac{0.7}{10}=0.37$〔程式計算〕。三者差很多：同樣是 10 個模型，多樣性決定效果。

**Numerical example (from the source)**: $s=1$ and $k=10$. If $c=0$, $\mathrm{Var}=\frac1{10}=0.1$. If $c=0.8$, $\mathrm{Var}=0.1+0.9\times0.8=0.82$, only down from $1$ to $0.82$. Check with the other form: $c+\frac{s-c}{k}=0.8+\frac{0.2}{10}=0.82$. One more case, $c=0.3$: $0.3+\frac{0.7}{10}=0.37$ [computed by program]. The three differ a lot: with the same 10 models, diversity decides the effect.

![折線圖：橫軸是模型個數 k 從 1 到 50，縱軸是平均後的變異數（s 等於 1）；三條線分別是 c 除以 s 等於 0、0.3、0.8，在 k 等於 10 處標出 0.1 與 0.82，三條線分別趨近 0、0.3、0.8。（Line chart: the horizontal axis is the number of models k from 1 to 50 and the vertical axis is the variance of the average with s equal to 1; three lines for c over s equal to 0, 0.3 and 0.8, with 0.1 and 0.82 marked at k equal to 10; the lines approach 0, 0.3 and 0.8.）](../assets/figures/lec04-kp16-var-vs-k.png)

**圖說 `lec04-kp16-var-vs-k`**：先看三條線在 $k=1$ 的共同起點 $1$：只有一個模型時，平均就是它自己，變異數就是 $s$；再看 $k=10$ 的兩個圓點：綠線（$c=0$）掉到 $0.1$，紅線（$c=0.8$）只掉到 $0.82$。最後看每條線的尾巴：它們不會碰到 0，而是貼著各自的下限 $c$ 走平。多加模型只能消掉 $\frac{s-c}{k}$ 那一份。〔功能：1、2〕

**Caption `lec04-kp16-var-vs-k`**: Look first at the common starting point $1$ of the three lines at $k=1$: with one model the average is the model itself and the variance is $s$; then look at the two dots at $k=10$: the green line ($c=0$) falls to $0.1$ and the red line ($c=0.8$) falls only to $0.82$. Finally look at the tails: they do not reach 0 but flatten along their own floors $c$. More models can only remove the part $\frac{s-c}{k}$. [Function: 1, 2]

### 2. 怎麼讓模型彼此不同：bootstrap（How to make models differ: the bootstrap）

上一節的結論是：要讓 $c$ 小，各模型的誤差必須不要同步。可是我們手上只有一份訓練資料，用同一份資料、同一個演算法訓練 $k$ 次，結果會一模一樣。來源的作法是 **bootstrap 抽樣**：對 $n$ 筆訓練資料「有放回地」抽 $n$ 次，得到一份新資料，每個模型用自己的一份。有放回，所以有些點被抽到好幾次，有些完全沒被抽到；每個模型看到的資料不同，擬合出來的函數也就不同。

The conclusion of the last section: to make $c$ small the errors of the models must not move together. But we hold only one training set, and training $k$ times on the same data with the same algorithm gives identical models. The source's recipe is **bootstrap sampling**: draw $n$ times *with replacement* from the $n$ training points to get a new data set, and give each model its own. Because of replacement some points are drawn several times and some never; each model sees different data, so each fitted function differs.

一個 bootstrap 樣本平均含多少種不同的點？某一個點在一次抽取中**沒被**抽到的機率是 $1-\frac1n$，連續 $n$ 次都沒被抽到的機率是 $\big(1-\frac1n\big)^n$。這個數隨 $n$ 變大趨近 $e^{-1}=0.3679$，所以被抽到至少一次的機率是 $1-\big(1-\frac1n\big)^n\to1-e^{-1}=0.6321$。這個數字在第3週 KP08 談「有放回抽樣」時已經出現過。

How many different points does a bootstrap sample contain on average? The chance that a given point is **not** picked in one draw is $1-\frac1n$, and the chance that it is missed in all $n$ draws is $\big(1-\frac1n\big)^n$. As $n$ grows this tends to $e^{-1}=0.3679$, so the chance of being picked at least once is $1-\big(1-\frac1n\big)^n\to1-e^{-1}=0.6321$. This number already appeared in Week 3, KP08, when sampling with replacement was discussed.

〔程式計算〕$n=5,10,20,100,1000$ 時分別是 $0.6723$、$0.6513$、$0.6415$、$0.6340$、$0.6323$。我們也實際抽了 4000 次 $n=20$ 的 bootstrap 樣本，相異點的平均比例是 $0.641$，與理論值 $0.6415$ 相符。所以每個模型大約只看到 $63\%$ 的相異資料，彼此重疊但不相同。

[Computed by program] For $n=5,10,20,100,1000$ it is $0.6723$, $0.6513$, $0.6415$, $0.6340$, $0.6323$. We also drew 4000 bootstrap samples with $n=20$ and the mean fraction of distinct points was $0.641$, matching the theoretical $0.6415$. So each model sees about $63\%$ distinct data: overlapping, but not the same.

**這裡有一個重要的限制**：所有 bootstrap 樣本都來自同一份訓練資料，約有 $63\%$ 的點互相重疊，因此各模型的誤差通常是**正相關**的，$c>0$。上一節已經說明 $c$ 是平均後降不掉的下限，所以 bagging 的變異降低有上限，不會像獨立情形那樣降到 $\frac{s}{k}$。

**An important limit here**: all bootstrap samples come from the same training set and overlap in about $63\%$ of the points, so the errors of the models are usually **positively correlated**, $c>0$. The last section showed that $c$ is the floor of the average, so the variance reduction of bagging has a ceiling and does not fall to $\frac{s}{k}$ as in the independent case.

### 3. 用實驗看它降多少，也看它什麼時候不降（Seeing how much it lowers, and when it does not）

用一個可重算的實驗。真實函數 $f(x)=1+2x-1.5x^2$，$x$ 在 $[-1,1]$，雜訊 $\sigma=0.3$（所以 $\sigma^2=0.09$，見 KP04），每份訓練集 $n=20$ 點。學習器選 **1 近鄰迴歸**：預測就是離輸入最近的那個訓練點的標籤。它很有彈性、很不穩定，訓練資料換一點，預測就跳一格，適合拿來看變異。Bagging 取 $B=30$ 個 bootstrap 模型，平均它們的預測。

A reproducible experiment. The true function is $f(x)=1+2x-1.5x^2$ on $[-1,1]$ with noise $\sigma=0.3$ (so $\sigma^2=0.09$, see KP04) and $n=20$ points per training set. The learner is **1-nearest-neighbour regression**: the prediction is the label of the closest training point. It is flexible and unstable: change the training data a little and the prediction jumps. That makes it good for watching variance. Bagging uses $B=30$ bootstrap models and averages their predictions.

![折線圖：20 個訓練點（橘點）、真實曲線 f（黑色）、10 條淡藍色的 bootstrap 單模型階梯預測、紅色虛線的單一 1 近鄰預測，以及綠色粗線的 30 個模型平均。綠線比任何一條藍色階梯都更平滑、更貼近黑色曲線。（Plot: 20 training points (orange dots), the true curve f (black), 10 pale-blue step predictions of single bootstrap models, the dashed red prediction of one 1-NN model, and the thick green average of 30 models. The green line is smoother than any blue staircase and closer to the black curve.）](../assets/figures/lec04-kp16-bagging-demo.png)

**圖說 `lec04-kp16-bagging-demo`**：先看紅色虛線，它是用全部 20 點訓練的單一 1 近鄰：每遇到一個雜訊大的點就跳一個台階，離黑色真實曲線忽上忽下。再看淡藍色的階梯：各個 bootstrap 模型的台階位置不同，這就是「模型彼此不同」。最後看綠色粗線：把 30 條階梯平均，台階被磨成斜坡，整體靠近黑色曲線。〔功能：1、2〕

**Caption `lec04-kp16-bagging-demo`**: Look first at the dashed red line, the single 1-NN trained on all 20 points: it jumps a step at every noisy point and wanders above and below the black true curve. Then look at the pale-blue staircases: the steps of the bootstrap models sit in different places, which is what "the models differ" means. Finally look at the thick green line: averaging the 30 staircases grinds the steps into slopes and the whole curve moves toward the black one. [Function: 1, 2]

為了量化，我們重複 300 次：每次抽新的訓練集，在 $x=-0.8,-0.4,0,0.4,0.8$ 五個位置記下單一模型與 bagging 的預測，再算這五個位置上「300 次預測的變異數」〔程式計算〕。

To quantify it we repeat 300 times: each time draw a new training set, record the prediction of the single model and of bagging at the five positions $x=-0.8,-0.4,0,0.4,0.8$, then compute the variance of the 300 predictions at each position [computed by program].

| $x$ | 單一模型變異 | bagging 變異 | 比值 |
|---|---|---|---|
| $-0.8$ | $0.1735$ | $0.1117$ | $0.644$ |
| $-0.4$ | $0.1213$ | $0.0673$ | $0.555$ |
| $0.0$ | $0.1081$ | $0.0538$ | $0.498$ |
| $0.4$ | $0.0986$ | $0.0497$ | $0.504$ |
| $0.8$ | $0.0800$ | $0.0365$ | $0.456$ |
| 五點平均 | $0.1163$ | $0.0638$ | $0.549$ |

| $x$ | variance, single | variance, bagging | ratio |
|---|---|---|---|
| $-0.8$ | $0.1735$ | $0.1117$ | $0.644$ |
| $-0.4$ | $0.1213$ | $0.0673$ | $0.555$ |
| $0.0$ | $0.1081$ | $0.0538$ | $0.498$ |
| $0.4$ | $0.0986$ | $0.0497$ | $0.504$ |
| $0.8$ | $0.0800$ | $0.0365$ | $0.456$ |
| mean of the 5 | $0.1163$ | $0.0638$ | $0.549$ |

變異數大約砍掉一半，不是 $\frac1{30}$。反推：若用第 1 節的公式，$\frac1{30}+\frac{29}{30}\frac cs=0.549$ 給出 $c/s\approx0.53$，也就是這 30 個 bootstrap 模型彼此的相關約 $0.5$，與第 2 節說的「正相關、有下限」一致（這是用公式反推的粗估，因為公式假設每一對相關都相同，`[補充]`）。

The variance is cut roughly in half, not to $\frac1{30}$. Working backwards with the formula of section 1, $\frac1{30}+\frac{29}{30}\frac cs=0.549$ gives $c/s\approx0.53$: the 30 bootstrap models are correlated at about $0.5$, consistent with section 2 (positive correlation, with a floor). This is a rough back-calculation because the formula assumes every pair has the same correlation `[Added]`.

誤差呢？對真實函數 $f$ 的均方誤差（五點平均）：單一模型 $0.1164$，bagging $0.0649$。偏差平方在各點都只有 $10^{-4}$ 到 $10^{-3}$ 量級（單一模型 $0.00002$ 到 $0.00059$，bagging $0.00004$ 到 $0.00418$），比變異數小一到兩個數量級，所以這裡的改善幾乎全來自變異。若再加上雜訊地板 $\sigma^2=0.09$，對一筆新的觀測 $y$ 的期望平方誤差是 $0.2064\to0.1549$。注意地板 $0.09$ 沒有被動到：bagging 動不了雜訊。

And the error? The mean squared error against the true $f$ (averaged over the five positions) is $0.1164$ for the single model and $0.0649$ for bagging. The squared bias at each position is only of order $10^{-4}$ to $10^{-3}$ (single $0.00002$ to $0.00059$, bagging $0.00004$ to $0.00418$), one to two orders smaller than the variance, so the improvement here comes almost entirely from variance. Adding the noise floor $\sigma^2=0.09$, the expected squared error against a fresh observation $y$ goes from $0.2064$ to $0.1549$. Note that the floor $0.09$ is untouched: bagging cannot move the noise.

**什麼時候不降**（`[補充]`）：Bagging 要有「不穩定」的學習器才有東西好平均。我們在同樣的 300 個訓練集上換成**5 次多項式**，單一模型的 MSE（對 $f$）是 $0.0689$，已經比 1 近鄰好，因為它的形狀本來就接近真實的二次曲線；但 bagging 之後變成 $18.0173$〔程式計算〕。原因是 bootstrap 樣本裡相異的 $x$ 常常很少，少數幾個多項式擬合在邊緣暴衝，而一個暴衝的模型就足以拉壞整個平均（我們已經先丟掉相異 $x$ 不到 8 個的樣本，仍然如此）。結論：bagging 適合高變異、不穩定、對資料擾動敏感的學習器（如 1 近鄰、決策樹）；對本來就穩定的模型幫助很小，對受 bootstrap 破壞的模型甚至有害。

**When it does not lower it** (`[Added]`): bagging needs an *unstable* learner, something to average out. On the same 300 training sets we switched to a **degree-5 polynomial**: the single model has MSE $0.0689$ against $f$, already better than 1-NN because its shape is close to the true quadratic; after bagging it becomes $18.0173$ [computed by program]. The reason is that bootstrap samples often hold few distinct $x$ values, a few polynomial fits explode at the edges, and one exploding model is enough to wreck the average (we had already discarded samples with fewer than 8 distinct $x$ and it still happened). Conclusion: bagging suits high-variance, unstable learners (1-NN, decision trees) that are sensitive to data perturbations; it helps little for models that are already stable, and can even hurt a model that the bootstrap breaks.

### 4. 分類與代價（Classification, and the cost）

前面的平均用在迴歸。若模型是二元分類器，輸出 $h_j(\mathbf{x})\in\{-1,+1\}$，來源的作法是先平均再取正負號：

The averaging above is for regression. For binary classifiers with outputs $h_j(\mathbf{x})\in\{-1,+1\}$, the source averages first and then takes the sign:

$$
\hat f(\mathbf{x})=\mathrm{sign}\Big(\frac1k\sum_{j=1}^{k}h_j(\mathbf{x})\Big)
$$

這就是**多數決**：平均大於 0 代表 $+1$ 的票比較多。例：10 個分類器，7 個投 $+1$、3 個投 $-1$，平均是 $\frac{7-3}{10}=0.4>0$，輸出 $+1$。多數決也靠「投票的人不同調」才有用：`[補充]` 若每個分類器獨立且答對機率都是 $0.6$，11 個投票的多數決答對機率是 $0.7535$，101 個是 $0.9791$〔程式計算〕；但如果它們同調（$c$ 接近 $s$），人再多也只是 $0.6$。這又回到第 1 節的結論。

This is **majority voting**: an average above 0 means more votes for $+1$. Example: 10 classifiers, 7 vote $+1$ and 3 vote $-1$; the average is $\frac{7-3}{10}=0.4>0$ and the output is $+1$. Voting also works only if the voters do not move together: `[Added]` if each classifier is independent and correct with probability $0.6$, the majority of 11 voters is correct with probability $0.7535$ and of 101 voters $0.9791$ [computed by program]; but if they move together ($c$ close to $s$), more voters still give only $0.6$. This returns to the conclusion of section 1.

代價是什麼？Bagging 要訓練與儲存 $k$ 個模型。對小模型沒問題；但一個深度網路可能要訓練好幾天，訓練 $k$ 個就貴到做不起。那麼，能不能只訓練「一個」網路，卻得到「很多個子模型平均」的效果？這是 KP17（Dropout）要回答的問題。

What is the cost? Bagging has to train and store $k$ models. For small models that is fine; a deep network may take days to train, and training $k$ of them is too expensive. So can we train only *one* network and still get the effect of averaging many sub-models? That is the question KP17 (Dropout) answers.

### 本單元重點整理（KP16 summary）

**一句話重點**：把 $k$ 個誤差有相同變異數 $s$、兩兩共變異數 $c$ 的模型取平均，變異數是 $\frac{s}{k}+\frac{k-1}{k}c=c+\frac{s-c}{k}$；模型越多樣（$c$ 越小）降得越多，bagging 降的是變異，不是偏差。

**One-sentence summary**: Averaging $k$ models whose errors have equal variance $s$ and pairwise covariance $c$ gives variance $\frac{s}{k}+\frac{k-1}{k}c=c+\frac{s-c}{k}$; the more diverse the models (the smaller $c$) the larger the reduction, and bagging lowers variance, not bias.

**白話說明**：把錢分散在 10 支股票上，整體波動比押單一股票小得多；但如果 10 支股票全部同漲同跌，分散就沒有用。Bagging 就是投資組合：每個模型是一支股票，bootstrap 抽樣是替它們挑「不完全相同」的資料，讓它們不會同時犯錯。

**Plain explanation**: Spread money over 10 stocks and the overall swing is much smaller than with one stock; but if all 10 rise and fall together, spreading helps nothing. Bagging is a portfolio: each model is a stock, and bootstrap sampling picks "not quite the same" data for each so that they do not all err at the same time.

**正式定義**：$\hat f(\mathbf{x})=\frac1k\sum_{j=1}^{k}h_j(\mathbf{x})$（迴歸）、$\hat f(\mathbf{x})=\mathrm{sign}\big(\frac1k\sum_j h_j(\mathbf{x})\big)$（二元分類）；假設 $\mathrm{Var}(e_j)=s$、$\mathrm{Cov}(e_i,e_j)=c$（$i\neq j$），則 $\mathrm{Var}(\hat f)=\frac{s}{k}+\frac{k-1}{k}c$；bootstrap 樣本是對 $n$ 筆資料有放回抽 $n$ 次，每點被抽到的機率 $1-(1-\frac1n)^n\to1-e^{-1}\approx0.632$。

**Formal definition**: $\hat f(\mathbf{x})=\frac1k\sum_{j=1}^{k}h_j(\mathbf{x})$ (regression), $\hat f(\mathbf{x})=\mathrm{sign}\big(\frac1k\sum_j h_j(\mathbf{x})\big)$ (binary classification); assuming $\mathrm{Var}(e_j)=s$ and $\mathrm{Cov}(e_i,e_j)=c$ ($i\neq j$), $\mathrm{Var}(\hat f)=\frac{s}{k}+\frac{k-1}{k}c$; a bootstrap sample draws $n$ times with replacement from $n$ points, and each point is drawn with probability $1-(1-\frac1n)^n\to1-e^{-1}\approx0.632$.

**例子**：$s=1,k=10$：$c=0\to0.1$，$c=0.3\to0.37$，$c=0.8\to0.82$。實驗：1 近鄰、$n=20$、$B=30$：變異數五點平均 $0.1163\to0.0638$，對 $f$ 的 MSE $0.1164\to0.0649$。逐步計算見第 1–3 節。

**Example**: $s=1,k=10$: $c=0\to0.1$, $c=0.3\to0.37$, $c=0.8\to0.82$. Experiment: 1-NN, $n=20$, $B=30$: the variance averaged over five positions goes $0.1163\to0.0638$ and the MSE against $f$ goes $0.1164\to0.0649$. The computations are in sections 1–3.

**常見迷思**：

- 「Bagging 也能降低偏差」——不能。每個模型的預測平均值若都偏同一邊，平均之後還是偏同一邊；它降的是變異。
- 「模型越多越好，$k$ 夠大就趨近 0」——只有誤差互相獨立（$c=0$）才趨近 0；bootstrap 模型彼此正相關，下限是 $c$。
- 「Bagging 對任何模型都有用」——對不穩定的學習器有用，對穩定的模型幫助小，對被 bootstrap 破壞的模型甚至有害（第 3 節的 5 次多項式）。
- 「Bagging 能把雜訊地板壓低」——不能。$\sigma^2$ 是資料本身的雜訊（KP04），平均模型動不了它。

**Common misconceptions**:

- "Bagging also lowers bias." No. If the predictions of every model lean to the same side, so does their average; bagging lowers variance.
- "More models are always better, and a large $k$ drives the variance to 0." Only independent errors ($c=0$) go to 0; bootstrap models are positively correlated and the floor is $c$.
- "Bagging helps any model." It helps unstable learners, helps little for stable models, and can even hurt a model the bootstrap breaks (the degree-5 polynomial of section 3).
- "Bagging can push down the noise floor." No. $\sigma^2$ is the noise of the data itself (KP04), and averaging models cannot move it.

**圖**：本單元兩張圖，`lec04-kp16-var-vs-k`（平均後變異數隨模型個數的下降與下限，功能 1、2）與 `lec04-kp16-bagging-demo`（bootstrap 單模型與平均的比較，功能 1、2），以 `assets/figures/src/lec04-kp16-figures.py` 繪製。bootstrap 抽樣的流程在簡報中以原生節點與連線呈現。

**Figure**: Two figures in this unit, `lec04-kp16-var-vs-k` (the variance of the average against the number of models, with its floor; functions 1, 2) and `lec04-kp16-bagging-demo` (single bootstrap models against their average; functions 1, 2), drawn by `assets/figures/src/lec04-kp16-figures.py`. The flow of bootstrap sampling is drawn with native nodes and connectors in the deck.

**想深入**（選讀）：本單元把 $e_j$ 的來源寫成「各模型誤差」，其實有兩種不同的隨機性：訓練資料的抽樣，以及 bootstrap 與演算法本身的隨機。我們在 300 個訓練集上看到的變異數，同時包含兩者；第 1 節的公式把它們全部收進 $s$ 與 $c$。另外，每個 bootstrap 樣本約有 $37\%$ 的點沒被抽到，這些「袋外」（out-of-bag）點可以拿來當免費的驗證資料，但這超出本單元。

**Going deeper** (optional): This unit writes the source of $e_j$ as "the error of each model", but there are really two kinds of randomness: the sampling of the training data, and the randomness of the bootstrap and of the algorithm itself. The variance we saw over 300 training sets contains both, and the formula of section 1 packs them all into $s$ and $c$. Also, about $37\%$ of the points are left out of each bootstrap sample; these "out-of-bag" points can serve as free validation data, but that lies beyond this unit.

**小練習**：

1. $s=4$、$k=8$、$c=0.5$：求平均後的變異數，並用兩種寫法驗算。（提示：$\frac{s}{k}+\frac{k-1}{k}c$ 與 $c+\frac{s-c}{k}$。）
2. $s=2$、$c=0.5$：當 $k\to\infty$ 時平均的變異數趨近多少？要讓變異數不超過 $0.6$，最少要幾個模型？（提示：解 $c+\frac{s-c}{k}\le0.6$。）
3. 訓練資料 $n=8$ 筆，做 bootstrap 抽樣（有放回抽 8 次）。某一個特定的點被抽到至少一次的機率是多少？一個 bootstrap 樣本平均含幾個相異的點？（提示：先算沒被抽到的機率 $(1-\frac18)^8$。）

**Practice**:

1. $s=4$, $k=8$, $c=0.5$: find the variance of the average and check it in both forms. (Hint: $\frac{s}{k}+\frac{k-1}{k}c$ and $c+\frac{s-c}{k}$.)
2. $s=2$, $c=0.5$: what does the variance of the average approach as $k\to\infty$? What is the least number of models that keeps the variance at most $0.6$? (Hint: solve $c+\frac{s-c}{k}\le0.6$.)
3. $n=8$ training points, bootstrap sampling (8 draws with replacement). What is the probability that one given point is drawn at least once? How many distinct points does a bootstrap sample contain on average? (Hint: first the probability of being missed, $(1-\frac18)^8$.)

**解題示範**（以 $s=3$、$k=5$、$c=1$ 示範第 1 題的算法）：

1. 套公式：$\frac{s}{k}=\frac35=0.6$。
2. 相關項：$\frac{k-1}{k}c=\frac45\cdot1=0.8$。
3. 相加：$\mathrm{Var}=0.6+0.8=1.4$。
4. 檢查：$c+\frac{s-c}{k}=1+\frac{2}{5}=1.4$，兩種寫法相同；也確認 $1.4<s=3$，且 $1.4>c=1$。

**Worked solution** (the method of exercise 1, shown with $s=3$, $k=5$, $c=1$):

1. Apply the formula: $\frac{s}{k}=\frac35=0.6$.
2. The correlated part: $\frac{k-1}{k}c=\frac45\cdot1=0.8$.
3. Add: $\mathrm{Var}=0.6+0.8=1.4$.
4. Check: $c+\frac{s-c}{k}=1+\frac{2}{5}=1.4$, the two forms agree; also $1.4<s=3$ and $1.4>c=1$.

### 來源更正與取捨（Corrections to the source）

- 來源在說明式子時寫「每個模型的誤差 $e_j$ 獨立同分布 $\mathcal N(0,s)$，兩兩誤差的共變異數為 $c$」。獨立與 $c\neq0$ 互相矛盾（獨立就是 $c=0$）。本講義只假設每個 $e_j$ 變異數為 $s$、兩兩共變異數同為 $c$，不要求高斯分布與獨立；公式只用到變異數與共變異數。
- 來源沒有說明式子需要「$s$ 與 $c$ 對所有模型與所有成對都相同」。本講義把它寫成明確假設，並指出實際的 bootstrap 模型只是近似滿足（第 3 節的反推只是粗估）。
- 來源只寫「$c\approx s$ 時沒有幫助、$c\approx0$ 時降到 $s/k$」。本講義補上下限的寫法 $c+\frac{s-c}{k}$，並說明 bootstrap 模型彼此正相關，所以 $c>0$ 是常態。
- 來源的分類式 $\mathrm{sign}(\cdot)$ 只給式子。本講義說明它等於二元分類器的多數決，並用數字說明獨立投票與同調投票的差別（`[補充]`）。
- 來源說 bagging 降低變異，沒有提醒它不降偏差，也沒提醒它對穩定學習器幫助很小。本講義加上這兩點，並用 5 次多項式的計算結果展示「bootstrap 破壞擬合」的反例（`[補充]`）。
- 來源以「祕笈」中的 $s=1$、$k=10$ 數字為例，本講義沿用並另外算 $c=0.3$；1 近鄰實驗、$63.2\%$ 的推導與各表數字是 `[補充]`，皆已用程式重算。

- The source says "the errors $e_j$ of the models are i.i.d. $\mathcal N(0,s)$ and have pairwise covariance $c$". Independence and $c\neq0$ contradict each other (independent means $c=0$). This note only assumes that each $e_j$ has variance $s$ and every pair has the same covariance $c$; it requires neither a Gaussian distribution nor independence, and the formula uses only variances and covariances.
- The source does not say that the formula needs "$s$ and $c$ to be the same for all models and all pairs". This note states it as an explicit assumption and points out that real bootstrap models satisfy it only approximately (the back-calculation in section 3 is only a rough estimate).
- The source only says "no help when $c\approx s$, $s/k$ when $c\approx0$". This note adds the floor form $c+\frac{s-c}{k}$ and explains that bootstrap models are positively correlated, so $c>0$ is the normal case.
- The source gives the classification form $\mathrm{sign}(\cdot)$ as a formula only. This note explains that it is the majority vote of binary classifiers and uses numbers to show the difference between independent and coordinated voters (`[Added]`).
- The source says bagging lowers variance but does not warn that it does not lower bias, nor that it helps little for stable learners. This note adds both points and uses the degree-5 polynomial result as a counterexample in which the bootstrap breaks the fit (`[Added]`).
- The source's example uses the $s=1$, $k=10$ numbers of its professor's guide; this note keeps them and also computes $c=0.3$. The 1-NN experiment, the derivation of $63.2\%$ and the table numbers are `[Added]` and have all been recomputed by program.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 裝袋法 | bagging (bootstrap aggregating) |
| 自助抽樣 | bootstrap sampling |
| 有放回抽樣 | sampling with replacement |
| 共變異數 | covariance |
| 多數決 | majority vote |
| 袋外資料 | out-of-bag data |
| 不穩定的學習器 | unstable learner |
| 1 近鄰迴歸 | 1-nearest-neighbour regression |
