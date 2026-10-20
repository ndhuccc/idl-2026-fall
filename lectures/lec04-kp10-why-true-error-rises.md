# 第4週 KP10：真實誤差為何回升（Why the true error rises）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第4週「正則化與過擬合」的第十個知識點（KP10），承接 KP09 留下的問題：在 KP08 與 KP09 的圖上，訓練誤差一路下降，測試誤差與交叉驗證誤差卻先降後升，這個「升」從哪裡來？先備知識是 KP06（SURE：$\mathbf{Err}=\mathbf{err}-n\sigma^2+2\sigma^2\sum\partial\hat f_i/\partial y_i$）、KP07（敏感度就是複雜度）、KP08（過擬合）與 KP09（交叉驗證）；第一次出現的符號都會定義，所有數字都附有程式計算。內容依 CCC-LNS 第5章 KP10 的主文與教授祕笈消化後重新撰寫，讀取日期 2026-10-20；來源沒有的內容標示 `[補充]`。

This is the tenth knowledge point (KP10) of Week 4, "Regularization and overfitting". It picks up the question KP09 left open: in the figures of KP08 and KP09 the training error keeps falling while the test and cross-validation errors fall and then rise, so where does the "rise" come from? It assumes KP06 (SURE: $\mathbf{Err}=\mathbf{err}-n\sigma^2+2\sigma^2\sum\partial\hat f_i/\partial y_i$), KP07 (sensitivity is complexity), KP08 (overfitting) and KP09 (cross-validation); every symbol is defined when it first appears and every number comes from a program. The text is rewritten from the main text and the professor's guide of KP10 in Chapter 5 of CCC-LNS (accessed on 2026-10-20); content that the source does not contain is marked `[Added]`.

[下載本單元投影片](../slides/lec04-kp10-why-true-error-rises.jbd.json)

## KP10 真實誤差為何回升（Why the true error rises）

### 學習目標（Learning objectives）

讀完之後，你能把 KP06 的結果寫成「真實誤差 $\approx$ 訓練誤差 $+$ 複雜度懲罰」，並說出被丟掉的常數 $-n\sigma^2$ 為什麼不影響選模型；能用五個階段的數字驗算 $\mathbf{err}$、懲罰與它們的和，指出最低點在哪裡，並用逐階段的增減說明曲線為什麼轉向；能用多項式的實驗驗證「訓練誤差加懲罰」確實等於新資料上的誤差；也能說明驗證集和訓練集重疊時，驗證誤差為什麼看不到回升。

After reading, you can write the result of KP06 as "true error $\approx$ training error $+$ complexity penalty" and say why the dropped constant $-n\sigma^2$ does not affect model choice; verify $\mathbf{err}$, the penalty and their sum on the five-stage numbers, point out where the minimum is and use the stage-by-stage changes to explain why the curve turns; check with a polynomial experiment that "training error plus penalty" really equals the error on new data; and explain why a validation set that overlaps the training set cannot show the rise.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{err}$ | 訓練誤差：$\sum_i(\hat f_i-y_i)^2$，算得出來 | training error $\sum_i(\hat f_i-y_i)^2$, computable |
| $\mathbf{Err}$ | 真實誤差：$\sum_i(\hat f_i-f_i)^2$，看不到 | true error $\sum_i(\hat f_i-f_i)^2$, not observable |
| $\sigma^2$ | 雜訊的變異數 | noise variance |
| $n$ | 訓練點的個數（來源寫 $n_{tr}$） | number of training points (written $n_{tr}$ in the source) |
| $\partial\hat f_i/\partial y_i$ | 第 $i$ 筆標籤動一點，模型在該點的預測動多少（KP07） | how much the prediction at point $i$ moves when its label moves a little (KP07) |
| $S$ | 敏感度的總和 $\sum_i\partial\hat f_i/\partial y_i$ | the sum of the sensitivities $\sum_i\partial\hat f_i/\partial y_i$ |
| $\text{penalty}$ | 複雜度懲罰 $2\sigma^2S$ | complexity penalty $2\sigma^2S$ |
| $p$ | 參數個數，多項式 $p=d+1$ | number of parameters, $p=d+1$ for a polynomial of degree $d$ |
| $\mathcal T,\ \mathcal V$ | 訓練集、驗證集（KP01） | training set, validation set (KP01) |

### 1. 真實誤差等於訓練誤差加懲罰（True error equals training error plus a penalty）

KP06 得到：對參與訓練的點，真實誤差可以用算得出的量寫出來：

KP06 found that for the points that took part in training the true error can be written with computable quantities:

$$
\mathbf{Err}=\mathbf{err}-n\sigma^2+2\sigma^2\sum_{i=1}^{n}\frac{\partial\hat f_i}{\partial y_i}
$$

右邊三項各有各的角色：$\mathbf{err}$ 是訓練誤差，隨模型擬合得越緊而下降；$-n\sigma^2$ 是一個常數，只取決於雜訊與點數，不隨模型改變；最後一項 $2\sigma^2S$（$S=\sum\partial\hat f_i/\partial y_i$）是**複雜度懲罰**：KP07 說敏感度量的是模型對訓練標籤有多黏，越黏，$S$ 越大，懲罰越大。

The three terms on the right play different roles: $\mathbf{err}$ is the training error, which falls as the model fits more tightly; $-n\sigma^2$ is a constant that depends only on the noise and the number of points and does not change with the model; the last term $2\sigma^2S$ (with $S=\sum\partial\hat f_i/\partial y_i$) is the **complexity penalty**: KP07 said the sensitivity measures how closely the model clings to the training labels, and the stickier it is, the larger $S$ and the larger the penalty.

比較兩個模型時，$-n\sigma^2$ 對兩者一樣，不影響誰的誤差比較小。所以把它丟掉，得到來源的式 (36)：

When two models are compared, $-n\sigma^2$ is the same for both and does not change which error is smaller. So it is dropped, which gives equation (36) of the source:

$$
\mathbf{Err}\approx\mathbf{err}+2\sigma^2\sum_{i=1}^{n}\frac{\partial\hat f_i}{\partial y_i}
$$

嚴格說，右邊等於 $\mathbf{Err}+n\sigma^2$，而不是 $\mathbf{Err}$ 本身；差的就是那個常數。它有一個直接的意思：$\mathbf{Err}+n\sigma^2=\mathbb E\sum_i(\hat f_i-y_i')^2$，也就是在**同樣的 $x_i$ 上抽一組新的雜訊 $y_i'$** 時模型的期望誤差 `[補充]`。下一節的表與圖用這個量，並在需要時扣掉常數，換回 $\mathbf{Err}$。

Strictly, the right side equals $\mathbf{Err}+n\sigma^2$ rather than $\mathbf{Err}$ itself; the difference is that constant. It has a direct meaning: $\mathbf{Err}+n\sigma^2=\mathbb E\sum_i(\hat f_i-y_i')^2$, the expected error of the model when **a fresh set of noise $y_i'$ is drawn at the same $x_i$** `[Added]`. The table and figure of the next section use this quantity and subtract the constant when $\mathbf{Err}$ itself is needed.

訓練誤差下降、懲罰上升，兩者相加就是 U 形：左邊由訓練誤差的下降主導，右邊由懲罰的上升主導，轉折點就是該停的地方。這是「真實誤差回升」的全部原因，只是還沒有數字。

Training error falls and the penalty rises, and their sum is a U shape: the left side is dominated by the fall of the training error and the right side by the rise of the penalty, and the turning point is where to stop. This is the whole reason the true error rises, only without numbers yet.

### 2. 五個階段的數字（The five-stage numbers）

來源給了一組數字：$\sigma^2=1$，訓練過程分成五個階段，敏感度總和 $S$ 慢慢增加（數字是給定的示意值）：

The source gives a set of numbers: $\sigma^2=1$, training divided into five stages with the sensitivity sum $S$ slowly increasing (the numbers are given, illustrative values):

| 階段 | $\mathbf{err}$ | $S$ | 懲罰 $2\sigma^2S$ | $\mathbf{err}+$懲罰 |
|---|---|---|---|---|
| 1 | 10 | 0 | 0 | 10 |
| 2 | 7 | 0.5 | 1 | 8 |
| 3 | 5 | 1.5 | 3 | 8 |
| 4 | 4 | 3 | 6 | 10 |
| 5 | 3 | 5 | 10 | 13 |

| Stage | $\mathbf{err}$ | $S$ | Penalty $2\sigma^2S$ | $\mathbf{err}+$ penalty |
|---|---|---|---|---|
| 1 | 10 | 0 | 0 | 10 |
| 2 | 7 | 0.5 | 1 | 8 |
| 3 | 5 | 1.5 | 3 | 8 |
| 4 | 4 | 3 | 6 | 10 |
| 5 | 3 | 5 | 10 | 13 |

驗算：懲罰 $=2\cdot1\cdot S$，所以 $S=0,\,0.5,\,1.5,\,3,\,5$ 給 $0,\,1,\,3,\,6,\,10$；再逐列相加：$10+0=10$、$7+1=8$、$5+3=8$、$4+6=10$、$3+10=13$。最低是 $8$，在階段 2 與 3。階段 1 到 5 的過程中 $\mathbf{err}$ 從 $10$ 降到 $3$，但總和先降到 $8$ 再升回 $13$。

Check: the penalty is $2\cdot1\cdot S$, so $S=0,\,0.5,\,1.5,\,3,\,5$ give $0,\,1,\,3,\,6,\,10$; adding row by row: $10+0=10$, $7+1=8$, $5+3=8$, $4+6=10$, $3+10=13$. The lowest is $8$, at stages 2 and 3. From stage 1 to 5 the $\mathbf{err}$ falls from $10$ to $3$, but the sum first falls to $8$ and then rises to $13$.

為什麼在階段 3 之後轉向？看每一階段的增減：

Why does it turn after stage 3? Look at the change in each step:

| 階段變化 | $\Delta\mathbf{err}$（好處） | $\Delta$懲罰（代價） | 合計 |
|---|---|---|---|
| 1 到 2 | $-3$ | $+1$ | $-2$ |
| 2 到 3 | $-2$ | $+2$ | $0$ |
| 3 到 4 | $-1$ | $+3$ | $+2$ |
| 4 到 5 | $-1$ | $+4$ | $+3$ |

| Step | $\Delta\mathbf{err}$ (gain) | $\Delta$ penalty (cost) | Total |
|---|---|---|---|
| 1 to 2 | $-3$ | $+1$ | $-2$ |
| 2 to 3 | $-2$ | $+2$ | $0$ |
| 3 to 4 | $-1$ | $+3$ | $+2$ |
| 4 to 5 | $-1$ | $+4$ | $+3$ |

規律是：訓練誤差每一階段省下的越來越少（$3,2,1,1$），懲罰每一階段增加的越來越多（$1,2,3,4$）。當多出來的懲罰超過省下的訓練誤差，總和就開始上升，這就是階段 3 到 4。「訓練誤差還在下降，真實誤差卻已經回升」的意思就是：好處變小了，代價變大了。

The pattern: the training error saves less and less in each step ($3,2,1,1$) while the penalty adds more and more ($1,2,3,4$). When the added penalty exceeds the saved training error the sum starts to rise, which is the step from stage 3 to 4. "The training error is still falling yet the true error has already risen" means that the benefit has become small and the cost large.

![左：來源的五個階段（$\sigma^2=1$）：橘線 $\mathbf{err}$ 從 10 降到 3，紅色虛線懲罰從 0 升到 10，藍線兩者之和先降到 8（階段 2 與 3，綠色底）再升到 13。右：用 12 個點的多項式實際算出的對應曲線（每點平均，$\sigma^2=0.09$，橫軸是次數）：橘線訓練誤差，紅色虛線懲罰 $2\sigma^2p/n$，藍線兩者之和，空心圓是另抽新雜訊模擬出的同一組 $x$ 上的誤差，兩者重合。（Left: the five stages of the source ($\sigma^2=1$): the orange $\mathbf{err}$ falls from 10 to 3, the red dashed penalty rises from 0 to 10, and the blue sum first falls to 8 (stages 2 and 3, green band) and then rises to 13. Right: the corresponding curves actually computed with polynomials on 12 points (per-point means, $\sigma^2=0.09$, degree on the horizontal axis): orange training error, red dashed penalty $2\sigma^2p/n$, blue their sum, and hollow circles the simulated error at the same $x$ with fresh noise; the two coincide.）](../assets/figures/lec04-kp10-three-curves.png)

**圖說 `lec04-kp10-three-curves`**：先看左圖綠色底的範圍：藍線在階段 2 與 3 都是 8，是最低；橘線與紅色虛線在階段 3 到 4 之間交叉，交叉後懲罰的上升比訓練誤差的下降更快。再看右圖：藍線在次數 2 最低（$0.1128$），之後緩慢上升到次數 10 的 $0.1725$；每一個藍點都被空心圓蓋住，表示「訓練誤差加懲罰」與模擬的新資料誤差一致。右圖的回升比左圖緩和得多，因為懲罰是次數的線性函數，左圖的懲罰是示意用的加速增加。〔功能：1、2〕

**Caption `lec04-kp10-three-curves`**: First look at the green band of the left panel: the blue line is 8 at both stages 2 and 3, the lowest; the orange line and the red dashed line cross between stages 3 and 4, and after the crossing the penalty rises faster than the training error falls. Then the right panel: the blue line is lowest at degree 2 ($0.1128$) and rises slowly to $0.1725$ at degree 10; every blue point is covered by a hollow circle, showing that "training error plus penalty" agrees with the simulated error on new data. The rise in the right panel is much gentler than in the left panel because the penalty is a linear function of the degree while the penalty on the left is an illustrative accelerating one. [Function: 1, 2]

### 3. 用多項式實際算一次（Computing it with polynomials）

五個階段是給定的示意值，我們來驗證這個關係是不是真的。設定與 KP08 相同：真實函數 $f(x)=1+2x-1.5x^2$，$\sigma^2=0.09$，$n=12$ 個固定的 $x$（等距，$-1$ 到 $1$），配 $d=1,\dots,10$ 次的多項式。KP07 已經推出：最小平方多項式的敏感度總和 $S=\operatorname{tr}(H)=p=d+1$，所以懲罰是 $2\sigma^2p$，每點平均是 $2\sigma^2p/n$。我們用 4000 組新的雜訊重複，檢查「訓練誤差加懲罰」是否等於在同一組 $x$ 上抽新雜訊後的誤差（每點平均，程式計算）：

The five stages are given illustrative values, so let us verify that the relation really holds. The setting is the same as in KP08: true function $f(x)=1+2x-1.5x^2$, $\sigma^2=0.09$, $n=12$ fixed $x$ values (equally spaced from $-1$ to $1$), polynomials of degree $d=1,\dots,10$. KP07 derived that for a least-squares polynomial the sensitivity sum is $S=\operatorname{tr}(H)=p=d+1$, so the penalty is $2\sigma^2p$ and per point $2\sigma^2p/n$. We repeat with 4000 sets of fresh noise and check whether "training error plus penalty" equals the error after drawing fresh noise at the same $x$ (per-point means, computed by program):

| 次數 $d$ | $p$ | 訓練誤差 | 懲罰 $2\sigma^2p/n$ | 兩者之和 | 模擬的新資料誤差 | 差 |
|---|---|---|---|---|---|---|
| 1 | 2 | 0.3517 | 0.0300 | 0.3817 | 0.3779 | $-0.0038$ |
| 2 | 3 | 0.0678 | 0.0450 | **0.1128** | 0.1124 | $-0.0003$ |
| 3 | 4 | 0.0600 | 0.0600 | 0.1200 | 0.1202 | $+0.0003$ |
| 4 | 5 | 0.0524 | 0.0750 | 0.1274 | 0.1280 | $+0.0005$ |
| 5 | 6 | 0.0449 | 0.0900 | 0.1349 | 0.1357 | $+0.0007$ |
| 6 | 7 | 0.0373 | 0.1050 | 0.1423 | 0.1432 | $+0.0009$ |
| 7 | 8 | 0.0298 | 0.1200 | 0.1498 | 0.1507 | $+0.0008$ |
| 8 | 9 | 0.0223 | 0.1350 | 0.1573 | 0.1580 | $+0.0007$ |
| 9 | 10 | 0.0148 | 0.1500 | 0.1648 | 0.1654 | $+0.0006$ |
| 10 | 11 | 0.0075 | 0.1650 | 0.1725 | 0.1730 | $+0.0006$ |

| Degree $d$ | $p$ | Training error | Penalty $2\sigma^2p/n$ | Sum | Simulated error on new data | Difference |
|---|---|---|---|---|---|---|
| 1 | 2 | 0.3517 | 0.0300 | 0.3817 | 0.3779 | $-0.0038$ |
| 2 | 3 | 0.0678 | 0.0450 | **0.1128** | 0.1124 | $-0.0003$ |
| 3 | 4 | 0.0600 | 0.0600 | 0.1200 | 0.1202 | $+0.0003$ |
| 4 | 5 | 0.0524 | 0.0750 | 0.1274 | 0.1280 | $+0.0005$ |
| 5 | 6 | 0.0449 | 0.0900 | 0.1349 | 0.1357 | $+0.0007$ |
| 6 | 7 | 0.0373 | 0.1050 | 0.1423 | 0.1432 | $+0.0009$ |
| 7 | 8 | 0.0298 | 0.1200 | 0.1498 | 0.1507 | $+0.0008$ |
| 8 | 9 | 0.0223 | 0.1350 | 0.1573 | 0.1580 | $+0.0007$ |
| 9 | 10 | 0.0148 | 0.1500 | 0.1648 | 0.1654 | $+0.0006$ |
| 10 | 11 | 0.0075 | 0.1650 | 0.1725 | 0.1730 | $+0.0006$ |

差只有幾個萬分之一到千分之一（$d\ge2$ 時最多 $0.0009$；$d=1$ 的 $0.0038$ 是模擬的隨機誤差，這一列的值本身比較大），所以「訓練誤差加懲罰 $=$ 新資料上的誤差」確實成立。訓練誤差一路下降到 $0.0075$，懲罰一路上升到 $0.1650$，總和在 $d=2$ 最低。扣掉常數就得到真實誤差（每點）：$d=2$ 是 $0.1128-0.09=0.0228$，$d=10$ 是 $0.1725-0.09=0.0825$，與直接模擬 $\operatorname{mean}(\hat f-f)^2$ 的 $0.0225$ 與 $0.0829$ 一致。

The differences are a few ten-thousandths to a thousandth (at most $0.0009$ for $d\ge2$; the $0.0038$ at $d=1$ is simulation error in a row whose values are themselves larger), so "training error plus penalty $=$ the error on new data" does hold. The training error falls all the way to $0.0075$, the penalty rises all the way to $0.1650$ and the sum is lowest at $d=2$. Subtracting the constant gives the true error (per point): $0.1128-0.09=0.0228$ for $d=2$ and $0.1725-0.09=0.0825$ for $d=10$, in agreement with $0.0225$ and $0.0829$ from simulating $\operatorname{mean}(\hat f-f)^2$ directly.

有兩點要老實說。第一，這裡的橫軸是模型的次數，不是「訓練進行了多久」；來源說的「訓練越久懲罰越大」對神經網路是合理的說法，但我們沒有在網路上驗證它，KP15 的早停法會用二次近似處理這件事。第二，這個例子在 $d=1$ 到 $2$ 的下降主要來自偏差消失（直線不含真實函數），之後才是懲罰主導的緩慢回升；回升不是必然很陡，資料越少、模型越有彈性，回升越明顯。

Two things must be said honestly. First, the horizontal axis here is the degree of the model, not "how long the training has run"; the source's "the longer the training the larger the penalty" is plausible for neural networks but we did not verify it on a network, and the early stopping of KP15 treats it with a quadratic approximation. Second, the fall from $d=1$ to $2$ in this example comes mainly from the bias vanishing (a line does not contain the true function), and only afterwards does the penalty dominate with a slow rise; the rise is not necessarily steep, and the fewer the data and the more flexible the model, the more visible it is.

### 4. 為什麼驗證集不能和訓練集重疊（Why the validation set must not overlap the training set）

回升有一個前提：我們要看得見它。如果驗證集 $\mathcal V$ 和訓練集 $\mathcal T$ 重疊，驗證誤差裡就包含了模型已經「看過答案」的點，那些點的誤差和訓練誤差一樣一路下降，把回升遮住。來源整理了各種交集的意義：$\mathcal T\cap\mathcal R\ne\varnothing$ 是作弊 #1，$\mathcal V\cap\mathcal R\ne\varnothing$ 是作弊 #2（KP01），而 $\mathcal T\cap\mathcal V\ne\varnothing$ 不算作弊，卻會讓驗證誤差變成訓練誤差的同義詞，使我們看不到過擬合。

The rise has a precondition: we must be able to see it. If the validation set $\mathcal V$ overlaps the training set $\mathcal T$, the validation error contains points whose answers the model has already "seen", and the errors on those points keep falling like the training error and hide the rise. The source summarizes what the intersections mean: $\mathcal T\cap\mathcal R\ne\varnothing$ is cheating #1, $\mathcal V\cap\mathcal R\ne\varnothing$ is cheating #2 (KP01), and $\mathcal T\cap\mathcal V\ne\varnothing$ is not cheating but turns the validation error into a synonym of the training error and keeps us from seeing overfitting.

用實驗看一次（程式計算）。取 $n=15$ 個訓練點（$x$ 等距、雜訊固定種子），多項式次數 $d=1,\dots,10$，比較三種驗證方式：（甲）驗證集就是訓練集本身（完全重疊）；（乙）驗證集有 20 筆，其中 10 筆來自訓練集、10 筆是新抽的（一半重疊）；（丙）驗證集是 2000 筆新抽的點（乾淨）。

See it in an experiment (computed by program). Take $n=15$ training points (equally spaced $x$, a fixed noise seed) and polynomial degrees $d=1,\dots,10$, and compare three kinds of validation: (A) the validation set is the training set itself (full overlap); (B) the validation set has 20 points, 10 from the training set and 10 freshly drawn (half overlap); (C) the validation set is 2000 freshly drawn points (clean).

| 次數 $d$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 甲：完全重疊 | 0.3018 | 0.1137 | 0.1137 | 0.1109 | 0.0817 | 0.0563 | 0.0544 | 0.0421 | 0.0336 | 0.0145 |
| 乙：一半重疊 | 0.3975 | 0.1370 | 0.1373 | 0.1344 | 0.1223 | 0.1114 | **0.1083** | 0.1258 | 0.1425 | 0.2249 |
| 丙：乾淨 | 0.2863 | 0.0922 | 0.0921 | **0.0919** | 0.1191 | 0.1361 | 0.1365 | 0.1576 | 0.1771 | 0.2666 |

| Degree $d$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| A: full overlap | 0.3018 | 0.1137 | 0.1137 | 0.1109 | 0.0817 | 0.0563 | 0.0544 | 0.0421 | 0.0336 | 0.0145 |
| B: half overlap | 0.3975 | 0.1370 | 0.1373 | 0.1344 | 0.1223 | 0.1114 | **0.1083** | 0.1258 | 0.1425 | 0.2249 |
| C: clean | 0.2863 | 0.0922 | 0.0921 | **0.0919** | 0.1191 | 0.1361 | 0.1365 | 0.1576 | 0.1771 | 0.2666 |

甲的誤差單調下降，最低在 $d=10$（$0.0145$），完全看不到回升，選模型會選到最複雜的。丙在 $d=2$ 到 $4$ 幾乎相同（$0.0922,\,0.0921,\,0.0919$，差 $0.0003$，分不出來），之後一路上升到 $0.2666$，這才是真實的 U 形。乙夾在中間：它的最低點被拉到 $d=7$（$0.1083$），選到過擬合的模型；高次時它也開始回升，但終點 $0.2249$ 比丙的 $0.2666$ 低，因為被重疊的 10 筆點把誤差壓低了。乙的整體高度不能和丙直接比：它只有 20 筆、其中新抽的只有 10 筆，雜訊大，要比較的是**最低點出現在哪裡**。

Curve A falls monotonically and is lowest at $d=10$ ($0.0145$), showing no rise at all, so choosing by it picks the most complex model. Curve C is nearly the same from $d=2$ to $4$ ($0.0922,\,0.0921,\,0.0919$, a difference of $0.0003$ that cannot be told apart) and then rises all the way to $0.2666$, the true U shape. Curve B is in between: its lowest point is pulled to $d=7$ ($0.1083$), choosing an overfitted model; at high degree it also rises, but its endpoint $0.2249$ is below the $0.2666$ of C because the 10 overlapping points hold the error down. The overall height of B cannot be compared directly with C: it has only 20 points of which just 10 are fresh, so it is noisy, and what to compare is **where the lowest point lies**.

![三種驗證方式的誤差對多項式次數（$n=15$ 個訓練點，縱軸為對數刻度）：橘線（驗證集就是訓練集）一路下降，沒有回升；金色線（一半重疊）的最低點移到次數 7；藍線（乾淨的 2000 筆新資料）在次數 2 到 4 最低，之後上升；綠色虛線是雜訊地板 0.09。（The errors of three kinds of validation against the polynomial degree ($n=15$ training points, logarithmic vertical axis): the orange line (the validation set is the training set) falls all the way with no rise; the gold line (half overlap) has its lowest point moved to degree 7; the blue line (2000 clean fresh points) is lowest at degrees 2 to 4 and rises afterwards; the green dashed line is the noise floor 0.09.）](../assets/figures/lec04-kp10-overlap.png)

**圖說 `lec04-kp10-overlap`**：先看藍線的最低區（次數 2 到 4，貼著綠色虛線）與它之後的上升；再看橘線：它從次數 5 起掉到雜訊地板以下，一路沉到 $0.0145$，這種「低於地板」就是 KP08 說的記住雜訊。最後看金色線：最低點往右移到次數 7，說明重疊的點會把「該停的位置」往複雜的方向推。這是 `[補充]` 的實驗，重疊比例與抽樣固定，只示範方向，不代表所有資料都會得到相同的最低點。〔功能：1、2〕

**Caption `lec04-kp10-overlap`**: First look at the low region of the blue line (degrees 2 to 4, hugging the green dashed line) and its rise afterwards; then the orange line: from degree 5 on it drops below the noise floor and sinks to $0.0145$, a "below the floor" value of the kind KP08 called memorizing the noise. Finally the gold line: its lowest point shifts to the right, to degree 7, showing that overlapping points push the "place to stop" toward the complex side. This is an `[Added]` experiment with a fixed overlap ratio and fixed draws; it shows the direction only and does not mean every data set gives the same lowest point. [Function: 1, 2]

所以 $\mathcal T\cap\mathcal V=\varnothing$ 不只是為了公平，更是為了**偵測得到轉折點**：沒有它，選複雜度與早停都失去依據。但懲罰項 $2\sigma^2\sum\partial\hat f_i/\partial y_i$ 很難直接算（要對每筆標籤求導），而且我們通常不知道 $\sigma^2$。能不能不靠驗證集，直接在目標函數裡把複雜度壓住？這是 KP11 要回答的問題。

So $\mathcal T\cap\mathcal V=\varnothing$ is not only for fairness but for **being able to detect the turning point**: without it, choosing the complexity and early stopping lose their basis. But the penalty $2\sigma^2\sum\partial\hat f_i/\partial y_i$ is hard to compute directly (it needs a derivative with respect to every label) and we usually do not know $\sigma^2$. Can we press the complexity down directly inside the objective instead of relying on a validation set? That is the question KP11 answers.

### 本單元重點整理（KP10 summary）

**一句話重點**：真實誤差 $\approx$ 訓練誤差 $+$ 複雜度懲罰 $2\sigma^2\sum\partial\hat f_i/\partial y_i$（差一個常數 $n\sigma^2$）；訓練誤差下降、懲罰上升，總和先降後升，最低點就是該停的位置；驗證集不能與訓練集重疊，否則看不到回升。

**One-sentence summary**: True error $\approx$ training error $+$ the complexity penalty $2\sigma^2\sum\partial\hat f_i/\partial y_i$ (up to a constant $n\sigma^2$); the training error falls and the penalty rises, so the sum falls then rises and its lowest point is where to stop; the validation set must not overlap the training set or the rise cannot be seen.

**白話說明**：練習題越做越順（訓練誤差下降），但熬夜的副作用越來越大（複雜度懲罰上升）。過了某個時間點，副作用超過進步，真正考試就退步了。

**Plain explanation**: The practice problems get easier (the training error falls) but the side effects of staying up late keep growing (the complexity penalty rises). Past some point the side effects exceed the progress and the real exam gets worse.

**正式定義**：$\mathbf{Err}=\mathbf{err}-n\sigma^2+2\sigma^2\sum_i\partial\hat f_i/\partial y_i$；式 (36) 丟掉常數 $-n\sigma^2$；最小平方模型 $\sum_i\partial\hat f_i/\partial y_i=p$，每點懲罰 $2\sigma^2p/n$。

**Formal definition**: $\mathbf{Err}=\mathbf{err}-n\sigma^2+2\sigma^2\sum_i\partial\hat f_i/\partial y_i$; equation (36) drops the constant $-n\sigma^2$; for a least-squares model $\sum_i\partial\hat f_i/\partial y_i=p$ and the penalty per point is $2\sigma^2p/n$.

**例子**：$\sigma^2=1$ 的五階段：$\mathbf{err}=10,7,5,4,3$、懲罰 $0,1,3,6,10$、總和 $10,8,8,10,13$，最低在階段 2 與 3。多項式 $n=12$：總和最低在 $d=2$（$0.1128$），與模擬的 $0.1124$ 一致。

**Example**: the five stages with $\sigma^2=1$: $\mathbf{err}=10,7,5,4,3$, penalty $0,1,3,6,10$, sum $10,8,8,10,13$, lowest at stages 2 and 3. Polynomials with $n=12$: the sum is lowest at $d=2$ ($0.1128$), agreeing with the simulated $0.1124$.

**常見迷思**：

- 「訓練誤差還在下降，就該繼續訓練」——要看驗證或真實誤差，那裡才有回升。
- 「複雜度懲罰是固定常數」——它隨複雜度（或訓練）上升；固定的是 $-n\sigma^2$。
- 「真實誤差一定會陡峭地回升」——取決於資料量與模型彈性；$n=12$ 的例子與 KP09 的 $n=100$ 例子回升都很緩和。
- 「早停只看訓練誤差」——早停看的是驗證誤差開始上升。
- 「驗證集與訓練集重疊也沒關係」——不算作弊，但看不到回升，選模型會偏向過度複雜。

**Common misconceptions**:

- "The training error is still falling, so keep training." Look at the validation or true error, where the rise is.
- "The complexity penalty is a fixed constant." It rises with complexity (or training); the fixed one is $-n\sigma^2$.
- "The true error always rises steeply." It depends on the amount of data and the flexibility of the model; the $n=12$ example here and the $n=100$ example of KP09 rise only gently.
- "Early stopping looks only at the training error." Early stopping looks at the validation error starting to rise.
- "It does not matter if the validation set overlaps the training set." It is not cheating, but the rise cannot be seen and the choice leans toward over-complex models.

**圖**：本單元兩張圖：`lec04-kp10-three-curves`（五階段與多項式實驗的 $\mathbf{err}$、懲罰、兩者之和，功能 1、2）與 `lec04-kp10-overlap`（三種驗證方式，功能 1、2），以 `assets/figures/src/lec04-kp10-figures.py` 繪製；腳本印出本講義引用的每一個數字。

**Figure**: Two figures in this unit: `lec04-kp10-three-curves` ($\mathbf{err}$, penalty and their sum for the five stages and the polynomial experiment; functions 1, 2) and `lec04-kp10-overlap` (three kinds of validation; functions 1, 2), drawn by `assets/figures/src/lec04-kp10-figures.py`; the script prints every number quoted in this note.

**想深入**（選讀）`[補充]`：對最小平方多項式，$\mathbf{err}+2\sigma^2p$ 作為「模型選擇準則」就是統計裡的 Mallows 的 $C_p$（KP07 已提到）；在高斯雜訊且 $\sigma^2$ 已知時，它與 AIC 等價（差一個常數與尺度）。這些準則需要知道或估計 $\sigma^2$（KP07 的估計式），而且只對線性於參數的模型成立，所以在深度學習裡，我們用驗證集（KP09）與正則化（KP11 起）來代替。

**Going deeper** (optional) `[Added]`: For a least-squares polynomial, $\mathbf{err}+2\sigma^2p$ as a "model selection criterion" is Mallows' $C_p$ of statistics (already mentioned in KP07); under Gaussian noise with known $\sigma^2$ it is equivalent to AIC (up to a constant and a scale). These criteria need $\sigma^2$ to be known or estimated (the estimator of KP07) and hold only for models linear in their parameters, so in deep learning we use a validation set (KP09) and regularization (from KP11 on) instead.

**小練習**：

1. $\sigma^2=1$，某訓練過程四個階段的 $\mathbf{err}=9,6,4,3$，敏感度總和 $S=0,1,2,4$：算出每階段的懲罰與總和，找出最低點。（提示：懲罰 $=2\sigma^2S$。）
2. $n=20$、$\sigma^2=0.25$，一個含 $p=4$ 個參數的最小平方模型，訓練 MSE（每點）是 $0.30$：它每點的懲罰是多少？「訓練誤差加懲罰」是多少？再扣掉 $\sigma^2$ 得到的是什麼量？（提示：$2\sigma^2p/n$。）
3. 驗證集 $\mathcal V$ 有一半的點來自訓練集。高複雜度的模型在這些重疊的點上的誤差是偏小還是偏大？這會讓驗證誤差的最低點往複雜還是往簡單的方向移？（提示：想想訓練誤差隨複雜度怎麼變。）

**Practice**:

1. $\sigma^2=1$ and the four stages of some training process have $\mathbf{err}=9,6,4,3$ and sensitivity sums $S=0,1,2,4$: compute the penalty and the sum of each stage and find the lowest point. (Hint: penalty $=2\sigma^2S$.)
2. $n=20$, $\sigma^2=0.25$, and a least-squares model with $p=4$ parameters has a training MSE (per point) of $0.30$: what is its penalty per point? What is "training error plus penalty"? And what quantity do you get after subtracting $\sigma^2$? (Hint: $2\sigma^2p/n$.)
3. Half the points of the validation set $\mathcal V$ come from the training set. Is the error of a highly complex model on those overlapping points too small or too large? Does this move the lowest point of the validation error toward the complex or the simple side? (Hint: think how the training error changes with complexity.)

**解題示範**（以另一組數字示範第 1 題的算法）：$\sigma^2=0.5$，四個階段 $\mathbf{err}=8,5,3,2$、$S=0,0.5,2,4$。

1. 懲罰 $=2\sigma^2S=1\cdot S$：$0,\,0.5,\,2,\,4$。
2. 總和：$8+0=8$、$5+0.5=5.5$、$3+2=5$、$2+4=6$。
3. 最低是 $5$，在階段 3。
4. 檢查增減：階段 2 到 3 的 $\Delta\mathbf{err}=-2$、$\Delta$懲罰 $=+1.5$，合計 $-0.5$；階段 3 到 4 是 $-1$ 與 $+2$，合計 $+1$，所以轉向發生在階段 3。

**Worked solution** (the method of exercise 1 shown with another set of numbers): $\sigma^2=0.5$ and four stages with $\mathbf{err}=8,5,3,2$ and $S=0,0.5,2,4$.

1. Penalty $=2\sigma^2S=1\cdot S$: $0,\,0.5,\,2,\,4$.
2. Sums: $8+0=8$, $5+0.5=5.5$, $3+2=5$, $2+4=6$.
3. The lowest is $5$, at stage 3.
4. Check the changes: from stage 2 to 3, $\Delta\mathbf{err}=-2$ and $\Delta$penalty $=+1.5$, total $-0.5$; from stage 3 to 4 they are $-1$ and $+2$, total $+1$, so the turn happens at stage 3.

### 來源更正與取捨（Corrections to the source）

- 來源的式 (36) 寫成 $\mathbf{Err}\approx\mathbf{err}+2\sigma^2\sum\partial\hat f_i/\partial y_i$。嚴格說，右邊等於 $\mathbf{Err}+n\sigma^2$，丟掉的是 KP06 式 (28) 裡的常數 $-n\sigma^2$；教授祕笈五階段表裡標成「$\mathbf{Err}$」的 $10,8,8,10,13$ 其實是 $\mathbf{err}+$懲罰，要扣掉 $n\sigma^2$ 才是真實誤差。因為常數不隨階段變，最低點的位置不受影響。本講義把這一點寫明。
- 來源五階段表的敏感度總和與懲罰是給定的示意值，不是由任何模型算出；本講義補上多項式實驗，驗證「訓練誤差加懲罰 $=$ 新資料誤差」，並說明實驗的橫軸是次數，不是訓練時間。
- 來源說「隨著訓練進行，模型越來越複雜，這項不斷增加」。對多項式次數，這是 $S=p$ 隨次數遞增；對神經網路的訓練時間，本講義沒有驗證，留給 KP15 的二次近似。
- 來源的 Fig. 6 是書上的示意圖，本講義不使用，改用程式畫出的兩張圖；來源的曲線看起來回升很陡，我們的兩個實驗（$n=12$、$n=100$）回升都很緩，如實寫明。
- 來源說 $\mathcal T\cap\mathcal V\ne\varnothing$「不構成作弊，但會妨礙偵測過擬合」，本講義保留，並用重疊實驗補上數字。

- Equation (36) of the source reads $\mathbf{Err}\approx\mathbf{err}+2\sigma^2\sum\partial\hat f_i/\partial y_i$. Strictly the right side equals $\mathbf{Err}+n\sigma^2$, having dropped the constant $-n\sigma^2$ of equation (28) in KP06; the values $10,8,8,10,13$ labelled "$\mathbf{Err}$" in the five-stage table of the professor's guide are really $\mathbf{err}+$ penalty, and $n\sigma^2$ must be subtracted to get the true error. Since the constant does not change across stages, the location of the lowest point is unaffected. This note says so explicitly.
- The sensitivity sums and penalties of the source's five-stage table are given illustrative values, not computed from any model; this note adds a polynomial experiment that verifies "training error plus penalty $=$ error on new data" and says that the horizontal axis there is the degree, not the training time.
- The source says "as training proceeds the model becomes more and more complex and this term keeps increasing". For the polynomial degree that is $S=p$ increasing with the degree; for the training time of a neural network this note has not verified it and leaves it to the quadratic approximation of KP15.
- Fig. 6 of the source is a book illustration and is not used here; two figures drawn by program replace it. The curve in the source looks as if it rises steeply, whereas our two experiments ($n=12$ and $n=100$) rise only gently, and this is stated plainly.
- The source says $\mathcal T\cap\mathcal V\ne\varnothing$ "is not cheating but hinders the detection of overfitting"; this note keeps that and adds numbers with the overlap experiment.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 真實誤差 | true error |
| 訓練誤差（經驗誤差） | training (empirical) error |
| 複雜度懲罰 | complexity penalty |
| 敏感度 | sensitivity |
| 雜訊地板 | noise floor |
| 驗證集 | validation set |
| 重疊 | overlap |
| 早停法 | early stopping |
| 帽子矩陣 | hat matrix |
| 偏差 | bias |
