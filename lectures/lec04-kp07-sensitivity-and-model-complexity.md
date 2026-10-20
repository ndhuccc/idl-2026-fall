# 第4週 KP07：訓練點敏感度與模型複雜度（Training-point sensitivity and model complexity）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第4週「正則化與過擬合」的第七個知識點（KP07），承接 KP06 留下的問題：SURE 的懲罰項 $2\sigma^2\sum_i\partial\hat f_i/\partial y_i$ 裡的偏微分到底在量什麼，怎麼算，又怎麼和「模型複雜度」連起來？先備知識是 KP06、微分與基本的最小平方法（線性迴歸）；第一次出現的符號都會定義，所有數字都可以自己重算。內容依 CCC-LNS 第5章 KP7 的主文與教授祕笈消化後重新撰寫，讀取日期 2026-10-20；來源沒有的內容標示 `[補充]`。

This is the seventh knowledge point (KP07) of Week 4, "Regularization and overfitting". It picks up the question KP06 left open: what does the partial derivative in the SURE penalty $2\sigma^2\sum_i\partial\hat f_i/\partial y_i$ measure, how is it computed, and how does it connect to "model complexity"? It assumes KP06, derivatives and basic least squares (linear regression); every symbol is defined when it first appears and every number can be recomputed. The text is rewritten from the KP7 main text and professor's guide of CCC-LNS Chapter 5 (accessed on 2026-10-20); content that the source does not contain is marked `[Added]`.

[下載本單元投影片](../slides/lec04-kp07-sensitivity-and-model-complexity.jbd.json)

## KP07 訓練點敏感度與模型複雜度（Training-point sensitivity and model complexity）

### 學習目標（Learning objectives）

讀完之後，你能用一句話說出 $\partial\hat f_i/\partial y_i$ 的意思：第 $i$ 筆標籤動一點，模型在這一點的預測動多少；能用五個點算出直線擬合與四次多項式的敏感度，並比較兩者；能說明對「參數線性」的模型，敏感度的總和等於參數個數 $p$，所以 SURE 的懲罰是 $2\sigma^2p$（Mallows 的 $C_p$）；也能說出用殘差估計 $\sigma^2$ 時要用簡單模型，以及除數為什麼是 $n-p$ 而不是 $n-1$。

After reading, you can say in one sentence what $\partial\hat f_i/\partial y_i$ means: how much the prediction at point $i$ moves when the $i$-th label is nudged; compute the sensitivities of a straight-line fit and of a degree-4 polynomial on five points and compare them; explain that for a model that is linear in its parameters the total sensitivity equals the number of parameters $p$, so the SURE penalty is $2\sigma^2p$ (Mallows' $C_p$); and say why a simple model should be used to estimate $\sigma^2$ from residuals and why the divisor is $n-p$ rather than $n-1$.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $x_i,\,y_i$ | 第 $i$ 個訓練點的輸入與標籤 | input and label of training point $i$ |
| $\hat f_i$ | 訓練好的模型在第 $i$ 點的預測 | prediction of the trained model at point $i$ |
| $\partial\hat f_i/\partial y_i$ | 敏感度：只動 $y_i$，$\hat f_i$ 的變化率 | sensitivity: rate of change of $\hat f_i$ when only $y_i$ moves |
| $p$ | 模型的參數個數（直線 $p=2$，四次多項式 $p=5$） | number of parameters (a line has $p=2$, a degree-4 polynomial $p=5$) |
| $\mathbf D$ | 設計矩陣：每列是一個訓練點的特徵 $(1,x_i,x_i^2,\dots)$ | design matrix: each row holds the features $(1,x_i,x_i^2,\dots)$ of one training point |
| $\mathbf H$, $h_{ii}$ | 帽子矩陣（把 $\mathbf y$ 變成 $\hat{\mathbf f}$ 的矩陣）與它的對角元 | hat matrix (maps $\mathbf y$ to $\hat{\mathbf f}$) and its diagonal entries |
| $\sigma^2$ | 雜訊變異數（KP04） | noise variance (KP04) |

### 1. 敏感度到底在量什麼（What the sensitivity measures）

KP06 的結論是

$$
\mathbf{Err}=\mathbf{err}-n_{tr}\sigma^2+2\sigma^2\sum_{i=1}^{n_{tr}}\frac{\partial\hat f_i}{\partial y_i}.
$$

$\partial\hat f_i/\partial y_i$ 的意思很具體：**把第 $i$ 筆資料的標籤 $y_i$ 輕輕往上推一點點，模型在這一點的預測 $\hat f_i$ 會跟著動多少**。等於 $0$：模型完全不理這筆資料；等於 $1$：預測緊緊貼著這筆標籤走，$\hat f_i$ 就是 $y_i$ 本身。

KP06's result is the formula above. The meaning of $\partial\hat f_i/\partial y_i$ is concrete: **push the label $y_i$ of the $i$-th data point up by a small amount; how much does the prediction $\hat f_i$ at that point move with it?** A value of $0$ means the model ignores this data point; a value of $1$ means the prediction sticks to the label, and $\hat f_i$ is simply $y_i$.

來源用「黏土與橡皮筋」比喻：黏土可以捏成任何形狀，點一動，形狀就大幅改變；橡皮筋只能拉成直線，點動一點，影響很小。模型越像黏土，敏感度越大，也就是越「複雜」。

The source uses the picture of clay versus a rubber band: clay can be shaped into anything, so when one point moves the shape changes a lot; a rubber band can only be stretched straight, so a small move has little effect. The more a model resembles clay, the larger its sensitivity, that is, the more "complex" it is.

**`[補充]` 線性模型可以直接算出來。** 若模型對參數是線性的（直線、多項式都是），設計矩陣 $\mathbf D$ 的每一列是一個點的特徵，例如多項式的一列是 $(1,x_i,x_i^2,\dots)$。最小平方法給出參數 $\hat{\boldsymbol\beta}=(\mathbf D^\top\mathbf D)^{-1}\mathbf D^\top\mathbf y$，所以預測是

$$
\hat{\mathbf f}=\mathbf D\hat{\boldsymbol\beta}=\mathbf H\,\mathbf y,\qquad \mathbf H=\mathbf D(\mathbf D^\top\mathbf D)^{-1}\mathbf D^\top .
$$

預測是 $\mathbf y$ 的線性函數，所以 $\partial\hat f_i/\partial y_j=h_{ij}$，敏感度就是**帽子矩陣的對角元** $h_{ii}$。它們的總和是矩陣的跡：

$$
\sum_i\frac{\partial\hat f_i}{\partial y_i}=\operatorname{tr}\mathbf H=\operatorname{tr}\big((\mathbf D^\top\mathbf D)^{-1}\mathbf D^\top\mathbf D\big)=\operatorname{tr}(\mathbf I_p)=p .
$$

**`[Added]` For linear models the sensitivity can be computed exactly.** If the model is linear in its parameters (lines and polynomials are), each row of the design matrix $\mathbf D$ holds the features of one point, for a polynomial $(1,x_i,x_i^2,\dots)$. Least squares gives $\hat{\boldsymbol\beta}=(\mathbf D^\top\mathbf D)^{-1}\mathbf D^\top\mathbf y$, so the prediction is the first display: a linear function of $\mathbf y$. Hence $\partial\hat f_i/\partial y_j=h_{ij}$ and the sensitivity is the **diagonal entry $h_{ii}$ of the hat matrix**. Their sum is the trace of the matrix, the second display, using $\operatorname{tr}(\mathbf A\mathbf B)=\operatorname{tr}(\mathbf B\mathbf A)$.

這個結果很重要：總敏感度就是**參數個數** $p$。所以對這類模型，SURE 的懲罰 $2\sigma^2\sum_i\partial\hat f_i/\partial y_i=2\sigma^2p$，整條式子變成

$$
\mathbf{Err}=\mathbf{err}-n_{tr}\sigma^2+2\sigma^2p .
$$

這正是統計學裡的 **Mallows $C_p$**。複雜度的懲罰，在這裡就是「參數個數乘上 $2\sigma^2$」。

This result matters: the total sensitivity is the **number of parameters** $p$. So for this kind of model the SURE penalty is $2\sigma^2\sum_i\partial\hat f_i/\partial y_i=2\sigma^2p$ and the whole formula becomes the last display, which is the **Mallows $C_p$** of statistics. The penalty for complexity is simply "the number of parameters times $2\sigma^2$".

### 2. 五個點的例子：直線與四次多項式（Five points: a line and a degree-4 polynomial）

取五個訓練點 $x=(-2,-1,0,1,2)$（標籤隨便取，例如 $y=(0.3,\,1.1,\,1.6,\,1.0,\,0.6)$；敏感度不依賴 $y$）。

Take five training points $x=(-2,-1,0,1,2)$ (any labels will do, for example $y=(0.3,\,1.1,\,1.6,\,1.0,\,0.6)$; the sensitivities do not depend on $y$).

**直線（$p=2$）。** 設計矩陣每列是 $(1,x_i)$。因為 $\sum x_i=0$、$\sum x_i^2=10$，

$$
\mathbf D^\top\mathbf D=\begin{pmatrix}5&0\\0&10\end{pmatrix},\qquad h_{ii}=(1,x_i)\begin{pmatrix}1/5&0\\0&1/10\end{pmatrix}\begin{pmatrix}1\\x_i\end{pmatrix}=\frac15+\frac{x_i^2}{10}.
$$

五個點的敏感度是 $0.6,\ 0.3,\ 0.2,\ 0.3,\ 0.6$，總和 $2.0=p$。中間那點 $x=0$ 最小：$0.2$。把它的標籤往上推 $0.1$，直線在這一點的預測只動 $0.2\times0.1=0.02$〔程式計算〕。（來源例子裡的「$0.02$」正好和這個數字吻合。）

**Line ($p=2$).** Each row of the design matrix is $(1,x_i)$. Because $\sum x_i=0$ and $\sum x_i^2=10$, the first display follows, and the sensitivities of the five points are $0.6,\ 0.3,\ 0.2,\ 0.3,\ 0.6$ with total $2.0=p$. The middle point $x=0$ has the smallest one, $0.2$. Pushing its label up by $0.1$ moves the prediction of the line at this point by only $0.2\times0.1=0.02$ [computed by program]. (The "$0.02$" in the source's example agrees with this number.)

**四次多項式（$p=5$）。** 五個點、五個參數：設計矩陣 $\mathbf D$ 是 $5\times5$ 的可逆方陣，所以 $\mathbf H=\mathbf D(\mathbf D^\top\mathbf D)^{-1}\mathbf D^\top=\mathbf D\mathbf D^{-1}\mathbf D^{-\top}\mathbf D^\top=\mathbf I$。曲線恰好穿過每個點，$\hat f_i=y_i$，每個敏感度都是 $h_{ii}=1$，總和 $5=p$。把中間點的標籤推 $0.1$，預測就剛好動 $0.1$：比直線大 $5$ 倍。

**Degree-4 polynomial ($p=5$).** Five points and five parameters: the design matrix $\mathbf D$ is an invertible $5\times5$ matrix, so $\mathbf H=\mathbf I$ as in the display above. The curve passes through every point, $\hat f_i=y_i$, every sensitivity is $h_{ii}=1$ and the total is $5=p$. Pushing the label of the middle point by $0.1$ moves the prediction by exactly $0.1$: five times as much as for the line.

![兩張並排的圖：左圖是直線擬合，標籤上移後（藍線變成橘色虛線）幾乎沒變，預測只動 0.10；右圖是四次多項式，標籤上移後曲線整條被扭動，在該點動了 0.50。（Two panels side by side: on the left the straight-line fit barely changes when the label is moved up, the prediction moving only 0.10; on the right the degree-4 polynomial is bent as a whole and moves 0.50 at that point.）](../assets/figures/lec04-kp07-shift.png)

**圖說 `lec04-kp07-shift`**：兩張圖都把同一個點（$x=0$）的標籤往上推 $0.5$（為了看得清楚；推 $0.1$ 時動的量按比例縮小為 $0.02$ 與 $0.10$），藍線是推之前，橘色虛線是推之後。先看左圖：直線只微微傾斜，在該點預測動了 $0.10$；再看右圖：多項式穿過所有點，推一個點就帶動整條曲線扭動，在該點預測動了 $0.50$，兩端也翹得更厲害。〔功能：1〕

**Caption `lec04-kp07-shift`**: In both panels the label of the same point ($x=0$) is pushed up by $0.5$ (so that it is visible; for a push of $0.1$ the movements scale down proportionally to $0.02$ and $0.10$); the blue line is before the push and the orange dashed line after. Look at the left panel first: the line tilts only slightly and the prediction at that point moves by $0.10$. Then the right panel: the polynomial passes through all points, so pushing one point bends the whole curve, the prediction moves by $0.50$, and both ends swing out more. [Function: 1]

![長條圖：五個 x 位置各有兩根柱子，藍色是直線的敏感度 0.6、0.3、0.2、0.3、0.6，總和 2；橘色是四次多項式的敏感度，五根都等於 1，總和 5。（Bar chart with two bars at each of the five x positions: blue are the sensitivities of the line, 0.6, 0.3, 0.2, 0.3, 0.6 with total 2; orange are those of the degree-4 polynomial, all equal to 1 with total 5.）](../assets/figures/lec04-kp07-hat.png)

**圖說 `lec04-kp07-hat`**：先看橘色：四次多項式每個點的敏感度都頂到上限 $1$，五個加起來是 $5$；再看藍色：直線的敏感度比較低，而且兩端（$x=\pm2$，離中心遠的點）比中間大，加起來是 $2$。兩個總和就是各自的參數個數。〔功能：1、2〕

**Caption `lec04-kp07-hat`**: Look first at orange: the degree-4 polynomial has sensitivity at the ceiling $1$ at every point, and the five add up to $5$. Then blue: the line has lower sensitivities, larger at the two ends ($x=\pm2$, points far from the center) than in the middle, and they add up to $2$. The two totals are the numbers of parameters. [Function: 1, 2]

### 3. 代回 SURE：複雜度怎麼付代價（Back into SURE: how complexity pays a price）

用 $\sigma^2=0.09$ 與五個點：直線的懲罰是 $2\sigma^2p=2\cdot0.09\cdot2=0.36$；四次多項式是 $2\cdot0.09\cdot5=0.90$。複雜的模型多付 $0.54$。

With $\sigma^2=0.09$ and five points: the penalty of the line is $2\sigma^2p=2\cdot0.09\cdot2=0.36$; that of the degree-4 polynomial is $2\cdot0.09\cdot5=0.90$. The complex model pays $0.54$ more.

**一個你可以自己驗證的檢查**。四次多項式穿過每個點，$\mathbf{err}=0$。SURE 說

$$
\mathbf{Err}=0-5\sigma^2+2\sigma^2\cdot5=5\sigma^2=0.45 .
$$

直接算：$\hat f_i=y_i=f_i+\varepsilon_i$，所以 $\mathbf{Err}=\sum_i(\hat f_i-f_i)^2=\sum_i\varepsilon_i^2$，期望是 $5\sigma^2$。兩者一致：曲線把每個雜訊都背下來了，真實誤差就是五份雜訊。訓練誤差等於 $0$，一點都不代表模型好。

**A check you can do yourself.** The degree-4 polynomial passes through every point, so $\mathbf{err}=0$. SURE gives $\mathbf{Err}=0-5\sigma^2+2\sigma^2\cdot5=5\sigma^2=0.45$. Directly: $\hat f_i=y_i=f_i+\varepsilon_i$, so $\mathbf{Err}=\sum_i(\hat f_i-f_i)^2=\sum_i\varepsilon_i^2$, whose expectation is $5\sigma^2$. The two agree: the curve has memorized every noise term, and the true error is five shares of noise. A training error of $0$ says nothing about how good the model is.

這就是過擬合的量化語言：複雜模型的訓練誤差小，是因為敏感度大；懲罰項把這個「黏」的程度加回去。後面的 L2、L1 正則化與提前停止，都是在間接壓低這個敏感度。

This is the quantitative language of overfitting: a complex model has a small training error because its sensitivity is large, and the penalty adds this "stickiness" back. L2 and L1 regularization and early stopping later in this chapter all lower this sensitivity indirectly.

### 4. 雜訊變異數 $\sigma^2$ 從哪裡來（Where $\sigma^2$ comes from）

SURE 需要 $\sigma^2$，實務上它是未知的。來源建議用一個簡單模型（例如線性迴歸）的殘差來估計：

$$
\sigma^2\approx\frac1{n_{tr}-1}\sum_{i=1}^{n_{tr}}(y_i-\hat f_i)^2 .
$$

用簡單模型的理由是對的：複雜模型把雜訊也擬合進去，殘差變小，$\sigma^2$ 會被低估。極端例子就是上面的四次多項式：殘差全是 $0$，估出來的 $\sigma^2=0$。

SURE needs $\sigma^2$, which is unknown in practice. The source suggests estimating it from the residuals of a simple model (for example linear regression), the formula above. The reason for using a simple model is sound: a complex model fits the noise too, its residuals shrink, and $\sigma^2$ is underestimated. The extreme case is the degree-4 polynomial above: all residuals are $0$ and the estimate is $\sigma^2=0$.

**`[補充]` 除數要用 $n-p$。** 對有 $p$ 個參數的最小平方擬合，殘差平方和 $\mathrm{RSS}=\sum_i(y_i-\hat f_i)^2$ 只有 $n-p$ 個自由度（$p$ 個方向被擬合吃掉了），所以不偏的估計是 $\mathrm{RSS}/(n-p)$。用 $n-1$ 只在 $p=1$（只估一個平均）時才對。五個點、直線（$p=2$）、真實函數是直線、$\sigma^2=0.09$，重複 20 萬次〔程式計算〕：

| 除數 | $n-1=4$ | $n-p=3$ | 真值 |
|---|---|---|---|
| 估計的平均 | $0.0676$ | $0.0902$ | $0.09$ |

**`[Added]` The divisor should be $n-p$.** For a least-squares fit with $p$ parameters, the residual sum of squares $\mathrm{RSS}=\sum_i(y_i-\hat f_i)^2$ has only $n-p$ degrees of freedom ($p$ directions are eaten by the fit), so the unbiased estimate is $\mathrm{RSS}/(n-p)$. Using $n-1$ is right only for $p=1$ (estimating a single mean). With five points, a line ($p=2$), a straight-line truth and $\sigma^2=0.09$, repeated 200000 times [computed by program], the table shows the mean of the estimates: $n-1$ gives $0.0676$ (too small, exactly $\tfrac34$ of the truth) and $n-p$ gives $0.0902$ (close to the truth $0.09$).

還要補一個條件：簡單模型本身要大致正確。如果真實函數是拋物線而你用直線，殘差裡還混著「直線的偏差」，估出來的 $\sigma^2$ 會偏大 `[補充]`。實務上常用一個「夠簡單、但不離譜」的模型，或用多個模型比較。

One more condition: the simple model must itself be roughly right. If the truth is a parabola and you use a line, the residuals still contain the "bias of the line", and the estimate of $\sigma^2$ comes out too large `[Added]`. In practice one uses a model that is "simple enough but not absurd", or compares several models.

### 本單元重點整理（KP07 summary）

**一句話重點**：$\partial\hat f_i/\partial y_i$ 是「第 $i$ 筆標籤動一點，預測動多少」，量的是模型有多緊盯訓練資料；對參數線性的模型，它是帽子矩陣的對角元 $h_{ii}$，總和等於參數個數 $p$，SURE 的懲罰就是 $2\sigma^2p$（Mallows $C_p$）。

**One-sentence summary**: $\partial\hat f_i/\partial y_i$ is "how much the prediction moves when the $i$-th label is nudged" and measures how tightly the model clings to the training data; for a model linear in its parameters it is the diagonal entry $h_{ii}$ of the hat matrix, the total equals the number of parameters $p$, and the SURE penalty is $2\sigma^2p$ (Mallows' $C_p$).

**白話說明**：用黏土和橡皮筋繞過幾個釘子：橡皮筋拉直，釘子動一點，形狀幾乎不變；黏土會貼著每根釘子，釘子一動，整個形狀跟著變。模型的複雜度，就是它像黏土的程度。

**Plain explanation**: Wrap clay and a rubber band around a few nails. The rubber band is pulled straight and hardly changes when a nail moves; the clay hugs every nail and its whole shape changes when one nail moves. The complexity of a model is how much it behaves like clay.

**正式定義**：敏感度 $s_i=\partial\hat f_i/\partial y_i$。線性模型：$\hat{\mathbf f}=\mathbf H\mathbf y$、$\mathbf H=\mathbf D(\mathbf D^\top\mathbf D)^{-1}\mathbf D^\top$、$s_i=h_{ii}$、$\sum_is_i=\operatorname{tr}\mathbf H=p$、$\mathbf{Err}=\mathbf{err}-n_{tr}\sigma^2+2\sigma^2p$；$\sigma^2$ 的不偏估計為 $\mathrm{RSS}/(n-p)$。

**Formal definition**: sensitivity $s_i=\partial\hat f_i/\partial y_i$. Linear model: $\hat{\mathbf f}=\mathbf H\mathbf y$, $\mathbf H=\mathbf D(\mathbf D^\top\mathbf D)^{-1}\mathbf D^\top$, $s_i=h_{ii}$, $\sum_is_i=\operatorname{tr}\mathbf H=p$, $\mathbf{Err}=\mathbf{err}-n_{tr}\sigma^2+2\sigma^2p$; an unbiased estimate of $\sigma^2$ is $\mathrm{RSS}/(n-p)$.

**例子**：五點 $x=(-2,-1,0,1,2)$：直線 $h_{ii}=0.6,0.3,0.2,0.3,0.6$（總和 $2$），中間點推 $0.1$ 預測動 $0.02$；四次多項式每個 $h_{ii}=1$（總和 $5$），推 $0.1$ 動 $0.1$。$\sigma^2=0.09$：懲罰 $0.36$ 對 $0.90$；四次多項式 $\mathbf{Err}=0-0.45+0.90=0.45=5\sigma^2$。逐步計算見第 2、3 節。

**Example**: five points $x=(-2,-1,0,1,2)$: the line has $h_{ii}=0.6,0.3,0.2,0.3,0.6$ (total $2$) and a push of $0.1$ at the middle point moves the prediction by $0.02$; the degree-4 polynomial has every $h_{ii}=1$ (total $5$) and moves by $0.1$. With $\sigma^2=0.09$ the penalties are $0.36$ versus $0.90$; for the degree-4 polynomial $\mathbf{Err}=0-0.45+0.90=0.45=5\sigma^2$. The computations are in sections 2 and 3.

**常見迷思**：

- 「敏感度大代表模型學得好」——它代表模型對訓練資料過度記憶，訓練誤差小是因為貼著雜訊走。
- 「用複雜模型估計 $\sigma^2$ 比較準」——複雜模型把雜訊也擬合掉，$\sigma^2$ 會被低估，極端時估成 $0$。
- 「$\hat f$ 動得多，就是模型錯得多」——敏感度量的是「跟著這筆標籤動」，和模型對不對是兩回事。
- 「敏感度懲罰只是理論」——L2、L1 正則化與提前停止都是在間接壓低它。

**Common misconceptions**:

- "A large sensitivity means that the model has learned well." It means that the model over-memorizes its training data; the training error is small because it follows the noise.
- "Estimating $\sigma^2$ with a complex model is more accurate." A complex model fits the noise away, so $\sigma^2$ is underestimated and in the extreme case estimated as $0$.
- "If $\hat f$ moves a lot the model is very wrong." Sensitivity measures "moving with this label", which is a different matter from being right or wrong.
- "The sensitivity penalty is only theory." L2 and L1 regularization and early stopping all lower it indirectly.

**圖**：本單元兩張圖，`lec04-kp07-shift`（直線與四次多項式在標籤被推動後的變化，功能 1）與 `lec04-kp07-hat`（兩種模型的敏感度長條圖，功能 1、2），以 `assets/figures/src/lec04-kp07-figures.py` 繪製。

**Figure**: Two figures in this unit, `lec04-kp07-shift` (the line and the degree-4 polynomial after a label is pushed; function 1) and `lec04-kp07-hat` (bars of the sensitivities of both models; functions 1, 2), drawn by `assets/figures/src/lec04-kp07-figures.py`.

**想深入**（選讀）：帽子矩陣 $\mathbf H$ 是投影矩陣（$\mathbf H^2=\mathbf H$），所以 $0\le h_{ii}\le1$，跡等於秩 $p$。對不是參數線性的模型（神經網路），$\partial\hat f_i/\partial y_i$ 仍有定義，總和叫「有效自由度」（effective degrees of freedom），它可以不是整數，也不一定等於參數個數；這是為什麼參數很多的網路不一定過擬合得那麼厲害。

**Going deeper** (optional): The hat matrix $\mathbf H$ is a projection ($\mathbf H^2=\mathbf H$), so $0\le h_{ii}\le1$ and the trace equals the rank $p$. For models that are not linear in the parameters (neural networks) $\partial\hat f_i/\partial y_i$ is still defined; its sum is called the "effective degrees of freedom", which need not be an integer or equal the number of parameters; this is one reason why a network with many parameters need not overfit as badly as the count suggests.

**小練習**：

1. 三個點 $x=(-1,0,1)$，用直線擬合。算出三個敏感度 $h_{ii}$，確認總和是 $2$；再算：把 $x=1$ 的標籤推 $0.3$，直線在該點的預測動多少？（提示：$\mathbf D^\top\mathbf D=\operatorname{diag}(3,2)$，$h_{ii}=\tfrac13+\tfrac{x_i^2}{2}$。）
2. $n=10$、$\sigma^2=0.5$，一個有 $p=3$ 個參數的線性模型，訓練誤差 $\mathbf{err}=12$。用 $C_p$ 求 $\mathbf{Err}$ 與每點平均。（提示：先算 $n\sigma^2$ 與 $2\sigma^2p$。）
3. 在 $n=12$ 個點上擬合直線，殘差平方和 $\mathrm{RSS}=5.0$。分別用除數 $n-1$ 與 $n-p$ 估計 $\sigma^2$，說明哪一個是不偏的。（提示：直線有 $p=2$。）

**Practice**:

1. Three points $x=(-1,0,1)$ fitted by a line. Compute the three sensitivities $h_{ii}$ and confirm that they add up to $2$; then compute how much the prediction of the line at $x=1$ moves if the label of that point is pushed by $0.3$. (Hint: $\mathbf D^\top\mathbf D=\operatorname{diag}(3,2)$ and $h_{ii}=\tfrac13+\tfrac{x_i^2}{2}$.)
2. $n=10$, $\sigma^2=0.5$, a linear model with $p=3$ parameters and training error $\mathbf{err}=12$. Use $C_p$ to find $\mathbf{Err}$ and its per-point mean. (Hint: compute $n\sigma^2$ and $2\sigma^2p$ first.)
3. A line is fitted on $n=12$ points and the residual sum of squares is $\mathrm{RSS}=5.0$. Estimate $\sigma^2$ with the divisors $n-1$ and $n-p$ and say which is unbiased. (Hint: a line has $p=2$.)

**解題示範**（以 $n=8$、$\sigma^2=0.25$、$p=2$、$\mathbf{err}=3$ 示範第 2 題的算法）：

1. 雜訊地板：$n\sigma^2=8\cdot0.25=2$。
2. 複雜度懲罰：$2\sigma^2p=2\cdot0.25\cdot2=1$。
3. 真實誤差：$\mathbf{Err}=3-2+1=2$；每點平均 $2/8=0.25$。
4. 檢查：$\mathbf{Err}$ 比 $\mathbf{err}$ 小，因為 $p=2<n/2=4$：總敏感度小於 $n/2$，懲罰沒有補回整個雜訊地板（KP06 的臨界點）。

**Worked solution** (the method of exercise 2, shown with $n=8$, $\sigma^2=0.25$, $p=2$ and $\mathbf{err}=3$):

1. Noise floor: $n\sigma^2=8\cdot0.25=2$.
2. Complexity penalty: $2\sigma^2p=2\cdot0.25\cdot2=1$.
3. True error: $\mathbf{Err}=3-2+1=2$; per-point mean $2/8=0.25$.
4. Check: $\mathbf{Err}$ is smaller than $\mathbf{err}$ because $p=2<n/2=4$: the total sensitivity is below $n/2$, so the penalty does not win back the whole noise floor (the break-even point of KP06).

### 來源更正與取捨（Corrections to the source）

- 來源說高次多項式是「10 次多項式」，但例子只有 5 個點。5 個點不能決定 10 次多項式（有無限多條曲線穿過這些點）。本講義改用恰好穿過 5 點的**四次**多項式，敏感度可以精確算出。
- 來源的「$0.02$」與「$0.09$」是示意數字。本講義用精確值：直線中間點 $0.2\times0.1=0.02$（與來源吻合）；四次多項式 $1\times0.1=0.1$（來源寫「$0.09$ 甚至更多」，精確值是 $0.1$）。
- 來源的式 (29) 用 $n_{tr}-1$ 當除數。對 $p$ 個參數的最小平方擬合，不偏的除數是 $n-p$。本講義明確寫出兩者的差別，並用模擬展示（$0.0676$ 對 $0.0902$）。這是 `[補充]`。
- 來源只說「敏感度大＝複雜」。本講義補上帽子矩陣、$\sum_ih_{ii}=p$ 與 Mallows $C_p$ 的連結，以及五點例的精確檢查（$\mathbf{Err}=5\sigma^2$），都是 `[補充]`。

- The source speaks of a "degree-10 polynomial" but the example has only 5 points. Five points do not determine a degree-10 polynomial (infinitely many curves pass through them). This note uses the **degree-4** polynomial that passes through exactly the 5 points, for which the sensitivities can be computed exactly.
- The source's "$0.02$" and "$0.09$" are illustrative numbers. This note uses exact values: the middle point of the line gives $0.2\times0.1=0.02$ (agreeing with the source); the degree-4 polynomial gives $1\times0.1=0.1$ (the source says "$0.09$ or even more"; the exact value is $0.1$).
- The source's formula (29) divides by $n_{tr}-1$. For a least-squares fit with $p$ parameters the unbiased divisor is $n-p$. This note states the difference openly and shows it by simulation ($0.0676$ versus $0.0902$). This is `[Added]`.
- The source only says "large sensitivity = complex". This note adds the hat matrix, $\sum_ih_{ii}=p$ and the link to Mallows' $C_p$, as well as the exact check on the five-point example ($\mathbf{Err}=5\sigma^2$), all `[Added]`.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 敏感度 | sensitivity |
| 模型複雜度 | model complexity |
| 帽子矩陣 | hat matrix |
| 設計矩陣 | design matrix |
| 跡 | trace |
| 有效自由度 | effective degrees of freedom |
| 殘差平方和 | residual sum of squares (RSS) |
| 自由度 | degrees of freedom |
| 不偏估計 | unbiased estimate |
| 過擬合 | overfitting |
