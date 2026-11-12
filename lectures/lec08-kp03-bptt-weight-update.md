# 第8週 KP03：共享權重的梯度是每一步外積的總和，算完再一起更新（The gradient of a shared weight is a sum of per-step outer products, followed by one update）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第8週的第三個知識點（KP03），接在 KP02 之後。KP02 從最後一步倒著算回來，得到每一步的狀態梯度 $\delta_t$。這一節問：有了 $\delta_1,\dots,\delta_T$，共用的 $\mathbf{V}$、$\mathbf{W}$、$\mathbf{U}$ 和兩個偏置各該怎麼改？

This is the third knowledge point (KP03) of Week 8 and follows KP02. KP02 worked backward from the last step and obtained the state gradient $\delta_t$ of every step. This note asks: given $\delta_1,\dots,\delta_T$, how should the shared $\mathbf{V}$, $\mathbf{W}$, $\mathbf{U}$ and the two biases change?

先備知識是 KP01 的三條前向公式、KP02 的 $\delta_t$，以及第2週的梯度下降與 $\tanh$ 的導數 $1-\tanh^2$。第一次出現的詞與符號都會定義。所有數字都由 `assets/figures/src/lec08-kp03-figures.py` 算出，可以自己重算。

The prerequisites are the three forward equations of KP01, the $\delta_t$ of KP02, and gradient descent and the $\tanh$ derivative $1-\tanh^2$ from Week 2. Every term and symbol is defined when it first appears. Every number is computed by `assets/figures/src/lec08-kp03-figures.py`, so you can recompute it yourself.

## KP03 共享權重的梯度是每一步外積的總和，算完再一起更新（The gradient of a shared weight is a sum of per-step outer products, followed by one update）

### 學習目標（Learning objectives）

讀完之後，你能先把 $\delta_t$ 乘上 $\tanh$ 的導數，得到淨輸入上的誤差 $\mathbf{g}_t$。你能寫出 $\mathbf{V}$、$\mathbf{W}$、$\mathbf{U}$ 的梯度公式：每一步一個外積，再對 $t$ 加總，並檢查維度。你能說明來源的 Kronecker 積寫法和外積是同一件事。你能寫出兩個偏置的梯度與五條梯度下降更新式，並用 3 步純量例子手算一次完整的更新。

After reading, you can multiply $\delta_t$ by the $\tanh$ derivative to get the error $\mathbf{g}_t$ at the net input. You can write the gradients of $\mathbf{V}$, $\mathbf{W}$ and $\mathbf{U}$, one outer product per step summed over $t$, and check their dimensions. You can explain why the source's Kronecker-product form says the same thing as the outer product. You can write the gradients of the two biases and the five gradient-descent update rules, and carry out one full update of the 3-step scalar example by hand.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathcal{L}=\sum_{t=1}^{T}\mathcal{L}_t$ | 總損失，各步損失之和 | total loss, the sum of the step losses |
| $\mathbf{e}_t=\partial\mathcal{L}_t/\partial\mathbf{y}_t\in\mathbb{R}^{q}$ | 第 $t$ 步的輸出誤差 | output error at step $t$ |
| $\delta_t=\partial\mathcal{L}/\partial\mathbf{h}_t\in\mathbb{R}^{p}$ | 第 $t$ 步的狀態梯度（KP02） | state gradient at step $t$ (KP02) |
| $\mathbf{g}_t=\partial\mathcal{L}/\partial\mathbf{i}_t\in\mathbb{R}^{p}$ | 淨輸入上的誤差（第1節定義） | error at the net input (defined in Section 1) |
| $\odot$ | 逐元素相乘 | element-wise product |
| $\mathbf{a}\mathbf{b}^\top$ | 外積，結果是矩陣 | outer product, a matrix |
| $\otimes$ | Kronecker 積 | Kronecker product |
| $\operatorname{vec}$、$\operatorname{vec}^{-1}$ | 把矩陣攤平成向量、再摺回矩陣 | flatten a matrix into a vector, and fold it back |
| $\mathbf{I}_p$ | $p\times p$ 單位矩陣 | $p\times p$ identity matrix |
| $\eta>0$ | 學習率，每次更新的步長 | learning rate, the step size of an update |

### 1. 先把 $\delta_t$ 帶過 $\tanh$（First carry $\delta_t$ through $\tanh$）

KP02 留下的問題是：有了 $\delta_t$，權重要怎麼改？第一步是看清楚權重「住在哪裡」。$\mathbf{V}$ 和 $\mathbf{b}_y$ 住在輸出式 $\mathbf{y}_t=\mathbf{V}\mathbf{h}_t+\mathbf{b}_y$ 裡。$\mathbf{W}$、$\mathbf{U}$、$\mathbf{b}_i$ 住在淨輸入 $\mathbf{i}_t=\mathbf{W}\mathbf{h}_{t-1}+\mathbf{U}\mathbf{x}_t+\mathbf{b}_i$ 裡，和 $\mathbf{h}_t$ 之間還隔著一層 $\tanh$。

KP02 left the question: given $\delta_t$, how do the weights change? The first step is to see where each weight "lives". $\mathbf{V}$ and $\mathbf{b}_y$ live in the output equation $\mathbf{y}_t=\mathbf{V}\mathbf{h}_t+\mathbf{b}_y$. $\mathbf{W}$, $\mathbf{U}$ and $\mathbf{b}_i$ live in the net input $\mathbf{i}_t=\mathbf{W}\mathbf{h}_{t-1}+\mathbf{U}\mathbf{x}_t+\mathbf{b}_i$, which is separated from $\mathbf{h}_t$ by a $\tanh$.

所以要先把 $\delta_t$ 往回推過 $\tanh$。$\mathbf{h}_t=\tanh(\mathbf{i}_t)$ 是逐元素運算，每個分量的導數是 $1-h_{t,j}^2$。鏈鎖律給出：

So first push $\delta_t$ back through the $\tanh$. $\mathbf{h}_t=\tanh(\mathbf{i}_t)$ acts element by element, and the derivative of each component is $1-h_{t,j}^2$. The chain rule gives:

$$
\mathbf{g}_t=\frac{\partial\mathcal{L}}{\partial\mathbf{i}_t}=(1-\mathbf{h}_t^2)\odot\delta_t=\operatorname{diag}(1-\mathbf{h}_t^2)\,\delta_t.
$$

我們把 $\mathbf{g}_t$ 叫做**局部誤差**（local error）：總損失對第 $t$ 步淨輸入的梯度。它告訴你：第 $t$ 步的淨輸入每個分量若稍微變大，總損失會怎麼變。$\mathbf{W}$、$\mathbf{U}$、$\mathbf{b}_i$ 的梯度全部從 $\mathbf{g}_t$ 出發 `[補充]`。

We call $\mathbf{g}_t$ the **local error**: the gradient of the total loss with respect to the net input of step $t$. It tells you how the total loss changes if each component of the net input at step $t$ grows a little. The gradients of $\mathbf{W}$, $\mathbf{U}$ and $\mathbf{b}_i$ all start from $\mathbf{g}_t$ `[Added]`.

主例沿用 KP02 的 3 步純量例子（$d=p=q=1$）：$W=0.5$、$V=1.0$，每一步 $1-h_t^2=0.8$，每一步輸出誤差 $e_t=1.0$。KP02 由此倒算出 $\delta_3=1.00$、$\delta_2=1.00+0.5\times0.8\times1.00=1.40$、$\delta_1=1.00+0.5\times0.8\times1.40=1.56$。

The running example is the 3-step scalar example of KP02 ($d=p=q=1$): $W=0.5$, $V=1.0$, $1-h_t^2=0.8$ at every step and output error $e_t=1.0$ at every step. From these KP02 works back to $\delta_3=1.00$, $\delta_2=1.00+0.5\times0.8\times1.00=1.40$ and $\delta_1=1.00+0.5\times0.8\times1.40=1.56$.

要算 $U$ 的梯度，還需要輸入值。補齊方式 `[補充]`：取 $U=1.0$、$b_i=b_y=0$、$h_0=0$，輸入 $x_1=0.4812$、$x_2=x_3=0.2576$。這組輸入讓每一步都得到 $h_t=\sqrt{0.2}=0.4472$，正好 $1-h_t^2=0.8$。目標值取 $y^*_t=-0.5528$，損失 $\mathcal{L}_t=\tfrac12(y_t-y^*_t)^2$；因為 $y_t=0.4472$，每步 $e_t=y_t-y^*_t=1.0$，總損失 $\mathcal{L}=1.5$。

To compute the gradient of $U$ we also need the inputs. The completion `[Added]`: take $U=1.0$, $b_i=b_y=0$, $h_0=0$ and inputs $x_1=0.4812$, $x_2=x_3=0.2576$. These inputs give $h_t=\sqrt{0.2}=0.4472$ at every step, so $1-h_t^2=0.8$ exactly. The targets are $y^*_t=-0.5528$ with loss $\mathcal{L}_t=\tfrac12(y_t-y^*_t)^2$; since $y_t=0.4472$, every step has $e_t=y_t-y^*_t=1.0$, and the total loss is $\mathcal{L}=1.5$.

| $t$ | $x_t$ | $h_{t-1}$ | $h_t$ | $e_t$ | $\delta_t$ | $g_t=0.8\,\delta_t$ |
|---|---|---|---|---|---|---|
| $1$ | $0.4812$ | $0$ | $0.4472$ | $1.0$ | $1.56$ | $1.248$ |
| $2$ | $0.2576$ | $0.4472$ | $0.4472$ | $1.0$ | $1.40$ | $1.120$ |
| $3$ | $0.2576$ | $0.4472$ | $0.4472$ | $1.0$ | $1.00$ | $0.800$ |

| $t$ | $x_t$ | $h_{t-1}$ | $h_t$ | $e_t$ | $\delta_t$ | $g_t=0.8\,\delta_t$ |
|---|---|---|---|---|---|---|
| $1$ | $0.4812$ | $0$ | $0.4472$ | $1.0$ | $1.56$ | $1.248$ |
| $2$ | $0.2576$ | $0.4472$ | $0.4472$ | $1.0$ | $1.40$ | $1.120$ |
| $3$ | $0.2576$ | $0.4472$ | $0.4472$ | $1.0$ | $1.00$ | $0.800$ |

常見錯誤：直接拿 $\delta_t$ 去算 $\mathbf{W}$ 的梯度，漏掉 $\tanh$ 的導數。$\delta_t$ 是對 $\mathbf{h}_t$ 的梯度；$\mathbf{W}$ 在 $\tanh$ 裡面，必須先乘上 $1-\mathbf{h}_t^2$。

Common error: using $\delta_t$ directly for the gradient of $\mathbf{W}$ and forgetting the $\tanh$ derivative. $\delta_t$ is the gradient with respect to $\mathbf{h}_t$; $\mathbf{W}$ sits inside the $\tanh$, so you must multiply by $1-\mathbf{h}_t^2$ first.

### 2. $\mathbf{V}$ 的梯度：每一步一個外積，再加總（The gradient of $\mathbf{V}$: one outer product per step, then a sum）

有了 $\mathbf{e}_t$ 和 $\mathbf{g}_t$，先處理最簡單的 $\mathbf{V}$。看單一步、單一個元素：$y_{t,k}=\sum_j V_{kj}h_{t,j}+b_{y,k}$，所以 $\partial y_{t,k}/\partial V_{kj}=h_{t,j}$。再乘上輸出誤差 $e_{t,k}$，第 $t$ 步對 $V_{kj}$ 的貢獻就是 $e_{t,k}h_{t,j}$。

With $\mathbf{e}_t$ and $\mathbf{g}_t$ in hand, start with the simplest one, $\mathbf{V}$. Look at one step and one entry: $y_{t,k}=\sum_j V_{kj}h_{t,j}+b_{y,k}$, so $\partial y_{t,k}/\partial V_{kj}=h_{t,j}$. Multiplying by the output error $e_{t,k}$, the contribution of step $t$ to $V_{kj}$ is $e_{t,k}h_{t,j}$.

把所有 $(k,j)$ 排成矩陣，第 $k$ 列第 $j$ 行放 $e_{t,k}h_{t,j}$，就是**外積**（outer product）$\mathbf{e}_t\mathbf{h}_t^\top$：一個直的向量乘一個橫的向量，得到一個矩陣。維度是 $(q\times1)(1\times p)=q\times p$，正好和 $\mathbf{V}$ 一樣大。

Arranging all $(k,j)$ in a matrix, with $e_{t,k}h_{t,j}$ in row $k$ and column $j$, gives the **outer product** $\mathbf{e}_t\mathbf{h}_t^\top$: a column vector times a row vector gives a matrix. Its size is $(q\times1)(1\times p)=q\times p$, exactly the size of $\mathbf{V}$.

$\mathbf{V}$ 在每一步都用了一次，所以總梯度要把每一步的貢獻加起來：

$\mathbf{V}$ is used once at every step, so the total gradient adds up the contributions of all steps:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{V}}=\sum_{t=1}^{T}\mathbf{e}_t\,\mathbf{h}_t^\top.
$$

為什麼是加總？這是多變數鏈鎖律：一個量從好幾條路影響結果，總變化率等於各條路的變化率相加。展開圖上，$\mathbf{V}$ 有 $T$ 條路通往損失，每一步一條。

Why a sum? This is the multivariable chain rule: when one quantity affects the result along several paths, the total rate of change is the sum over the paths. In the unrolled graph $\mathbf{V}$ has $T$ paths to the loss, one per step.

來源的類比是連鎖店：同一份管理規範在每家分店執行，總公司要改規範時，把每家分店的修改建議全部加起來。這個類比的侷限是：分店可以同時各自營業，RNN 的第 $t$ 步卻要等第 $t-1$ 步算完。

The source's analogy is a chain store: the same rulebook is used in every branch, and when headquarters revises it, it adds up the suggestions of all branches. The limit of the analogy: branches can operate at the same time, while step $t$ of an RNN must wait for step $t-1$.

主例：$e_t=1.0$、$h_t=0.4472$，每一步貢獻 $1.0\times0.4472=0.4472$，三步加總得 $\partial\mathcal{L}/\partial V=1.3416$。

Running example: with $e_t=1.0$ and $h_t=0.4472$, every step contributes $1.0\times0.4472=0.4472$, and the three steps add up to $\partial\mathcal{L}/\partial V=1.3416$.

常見錯誤：只拿最後一步 $t=T$ 的貢獻來改 $\mathbf{V}$。前面每一步也用了同一個 $\mathbf{V}$，漏掉它們，等於忽略前面那些步的修改需求。

Common error: changing $\mathbf{V}$ using only the contribution of the last step $t=T$. Every earlier step used the same $\mathbf{V}$ too; leaving them out ignores what those steps need.

### 3. $\mathbf{W}$ 與 $\mathbf{U}$ 的梯度：同樣的外積，換成 $\mathbf{g}_t$（The gradients of $\mathbf{W}$ and $\mathbf{U}$: the same outer product with $\mathbf{g}_t$）

$\mathbf{V}$ 的規則可以照搬到淨輸入。$\mathbf{i}_t=\mathbf{W}\mathbf{h}_{t-1}+\mathbf{U}\mathbf{x}_t+\mathbf{b}_i$ 和 $\mathbf{y}_t=\mathbf{V}\mathbf{h}_t+\mathbf{b}_y$ 的形狀完全一樣：「矩陣乘向量，再加偏置」。只要把輸出誤差 $\mathbf{e}_t$ 換成局部誤差 $\mathbf{g}_t$，把乘在矩陣右邊的向量換掉：

The rule for $\mathbf{V}$ carries over to the net input. $\mathbf{i}_t=\mathbf{W}\mathbf{h}_{t-1}+\mathbf{U}\mathbf{x}_t+\mathbf{b}_i$ has exactly the shape of $\mathbf{y}_t=\mathbf{V}\mathbf{h}_t+\mathbf{b}_y$: "a matrix times a vector, plus a bias". Replace the output error $\mathbf{e}_t$ by the local error $\mathbf{g}_t$, and use the vector that multiplies the matrix on its right:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{W}}=\sum_{t=1}^{T}\mathbf{g}_t\,\mathbf{h}_{t-1}^\top,\qquad
\frac{\partial\mathcal{L}}{\partial\mathbf{U}}=\sum_{t=1}^{T}\mathbf{g}_t\,\mathbf{x}_t^\top.
$$

維度檢查：$\mathbf{g}_t\mathbf{h}_{t-1}^\top$ 是 $(p\times1)(1\times p)=p\times p$，和 $\mathbf{W}$ 一樣；$\mathbf{g}_t\mathbf{x}_t^\top$ 是 $(p\times1)(1\times d)=p\times d$，和 $\mathbf{U}$ 一樣。一個好記的規則 `[補充]`：**梯度 = 左邊的誤差 $\times$ 右邊輸入的轉置**，誤差來自矩陣的輸出端，輸入是矩陣右邊乘的那個向量。

Dimension check: $\mathbf{g}_t\mathbf{h}_{t-1}^\top$ is $(p\times1)(1\times p)=p\times p$, like $\mathbf{W}$; $\mathbf{g}_t\mathbf{x}_t^\top$ is $(p\times1)(1\times d)=p\times d$, like $\mathbf{U}$. An easy rule to remember `[Added]`: **gradient = the error on the left times the transposed input on the right**, where the error comes from the output side of the matrix and the input is the vector the matrix multiplies.

這裡有一個重要的假設：算第 $t$ 步的貢獻時，把 $\mathbf{h}_{t-1}$ 當成已知的常數。可是 $\mathbf{h}_{t-1}$ 本身也是用 $\mathbf{W}$ 算出來的，這條間接的路不就漏了嗎？

There is an important assumption here: when computing the contribution of step $t$, $\mathbf{h}_{t-1}$ is treated as a known constant. But $\mathbf{h}_{t-1}$ itself was computed with $\mathbf{W}$; is that indirect path lost?

沒有漏。它已經藏在 $\delta_{t-1}$ 裡：KP02 算 $\delta_{t-1}$ 時，把來自未來各步的影響都收進去了。所以這裡每一步只算「直接」的那一份，再對 $t$ 加總，每條路剛好算一次 `[補充]`。

It is not. It is already inside $\delta_{t-1}$: when KP02 computed $\delta_{t-1}$, it collected the influence of all future steps. So here each step counts only its "direct" share, and summing over $t$ counts every path exactly once `[Added]`.

主例 `[補充]`：$W$ 的三步貢獻是 $g_1h_0=1.248\times0=0$、$g_2h_1=1.120\times0.4472=0.5009$、$g_3h_2=0.800\times0.4472=0.3578$，總和 $\partial\mathcal{L}/\partial W=0.8587$。$U$ 的三步貢獻是 $1.248\times0.4812=0.6006$、$1.120\times0.2576=0.2885$、$0.800\times0.2576=0.2061$，總和 $\partial\mathcal{L}/\partial U=1.0952$。

Running example `[Added]`: the three contributions to $W$ are $g_1h_0=1.248\times0=0$, $g_2h_1=1.120\times0.4472=0.5009$ and $g_3h_2=0.800\times0.4472=0.3578$, which add up to $\partial\mathcal{L}/\partial W=0.8587$. The three contributions to $U$ are $1.248\times0.4812=0.6006$, $1.120\times0.2576=0.2885$ and $0.800\times0.2576=0.2061$, which add up to $\partial\mathcal{L}/\partial U=1.0952$.

注意第 $1$ 步對 $W$ 的貢獻是 $0$。原因是 $h_0=0$：第 $1$ 步沒有舊記憶可以乘，$W$ 在這一步沒有作用，也就沒有可改的。來源用同一個道理問：如果某一步輸入 $\mathbf{x}_t=\mathbf{0}$，它對 $\mathbf{U}$ 的貢獻 $\mathbf{g}_t\mathbf{x}_t^\top$ 也是 $\mathbf{0}$。

Notice that step $1$ contributes $0$ to $W$. The reason is $h_0=0$: at step $1$ there is no old memory to multiply, so $W$ does nothing there and has nothing to correct. The source asks the same thing about inputs: if some step has $\mathbf{x}_t=\mathbf{0}$, its contribution $\mathbf{g}_t\mathbf{x}_t^\top$ to $\mathbf{U}$ is $\mathbf{0}$ as well.

![五組堆疊長條，橫軸依序是 V、W、U、b_i、b_y 的梯度。每根長條由三段組成，淺藍、藍、深藍分別是第 1、2、3 步的貢獻，頂端寫總和：V 為 1.3416（三段各 0.4472），W 為 0.8587（第 1 步是 0，第 2 步 0.5009，第 3 步 0.3578），U 為 1.0952（0.6006、0.2885、0.2061），b_i 為 3.1680（1.2480、1.1200、0.8000），b_y 為 3.0000（三段各 1.0000）（Five stacked bars for the gradients of V, W, U, b_i and b_y; light, medium and dark blue segments are the contributions of steps 1, 2 and 3, with the sum on top; the step-1 segment of W is zero）](../assets/figures/lec08-kp03-step-sums.png)

**圖說 `lec08-kp03-step-sums`**〔功能：一目了然、精準度〕：先看每根長條的三種藍色：每一種是某一步的貢獻，疊起來的高度就是總梯度。再看 $W$ 那一根：沒有淺藍色那一段，因為 $h_0=0$，第 $1$ 步對 $W$ 沒有貢獻。最後比較 $b_i$ 與 $b_y$：$b_y$ 每段都是 $1.0000$（輸出誤差），$b_i$ 的三段是 $g_t$，越早的步越大。

**Caption `lec08-kp03-step-sums`** [Function: seeing it at a glance, precision]: First look at the three blues in each bar: each is the contribution of one step, and the stacked height is the total gradient. Then look at the bar of $W$: the light-blue segment is missing, because $h_0=0$ and step $1$ contributes nothing to $W$. Finally compare $b_i$ and $b_y$: every segment of $b_y$ is $1.0000$ (the output error), while the three segments of $b_i$ are the $g_t$, larger for earlier steps.

### 4. Kronecker 積：同一件事的「攤平」寫法（The Kronecker product: the "flattened" way of writing the same thing）

上一節用外積寫出梯度。來源卻寫成 Kronecker 積與 $\operatorname{vec}^{-1}$，看起來很嚇人。它們是同一件事，只是記帳方式不同。

The previous section wrote the gradients as outer products. The source instead writes Kronecker products and $\operatorname{vec}^{-1}$, which look frightening. They are the same thing with different bookkeeping.

問題出在「對矩陣微分」：$\mathbf{i}_t$ 是向量，$\mathbf{W}$ 是矩陣，「向量對矩陣的導數」不是一般的矩陣。正式的矩陣微積分（來源說的 Magnus–Neudecker 方法）先把矩陣攤平成向量，把問題變回「向量對向量」。

The trouble is "differentiating with respect to a matrix": $\mathbf{i}_t$ is a vector and $\mathbf{W}$ is a matrix, and "the derivative of a vector with respect to a matrix" is not an ordinary matrix. Formal matrix calculus (the Magnus–Neudecker method named by the source) first flattens the matrix into a vector, turning the problem back into "vector with respect to vector".

**向量化**（vectorization）$\operatorname{vec}(\mathbf{W})$：把矩陣的行一行接一行疊成一個長向量。$p=2$ 時，$\operatorname{vec}(\mathbf{W})=[w_{11},w_{21},w_{12},w_{22}]^\top$。$\operatorname{vec}^{-1}$ 是反過來，把長向量摺回矩陣。

**Vectorization** $\operatorname{vec}(\mathbf{W})$: stack the columns of the matrix one below another into one long vector. For $p=2$, $\operatorname{vec}(\mathbf{W})=[w_{11},w_{21},w_{12},w_{22}]^\top$. $\operatorname{vec}^{-1}$ does the reverse and folds the long vector back into a matrix.

**Kronecker 積**（Kronecker product）$\mathbf{A}\otimes\mathbf{B}$：把 $\mathbf{A}$ 的每個元素 $a_{ij}$ 換成一整塊 $a_{ij}\mathbf{B}$。例如 $\mathbf{h}^\top\otimes\mathbf{I}_2=[\,h_1\mathbf{I}_2\ \ h_2\mathbf{I}_2\,]$，是一個 $2\times4$ 的矩陣。用它可以把 $\mathbf{W}\mathbf{h}$ 改寫成「矩陣乘 $\operatorname{vec}(\mathbf{W})$」：

**Kronecker product** $\mathbf{A}\otimes\mathbf{B}$: replace every entry $a_{ij}$ of $\mathbf{A}$ by a whole block $a_{ij}\mathbf{B}$. For example, $\mathbf{h}^\top\otimes\mathbf{I}_2=[\,h_1\mathbf{I}_2\ \ h_2\mathbf{I}_2\,]$ is a $2\times4$ matrix. With it, $\mathbf{W}\mathbf{h}$ can be rewritten as "a matrix times $\operatorname{vec}(\mathbf{W})$":

$$
\mathbf{W}\mathbf{h}=\begin{bmatrix}h_1&0&h_2&0\\0&h_1&0&h_2\end{bmatrix}\begin{bmatrix}w_{11}\\w_{21}\\w_{12}\\w_{22}\end{bmatrix}=\begin{bmatrix}w_{11}h_1+w_{12}h_2\\w_{21}h_1+w_{22}h_2\end{bmatrix}.
$$

所以 $\mathbf{i}_t$ 對 $\operatorname{vec}(\mathbf{W})$ 的導數就是那個 $2\times4$ 矩陣，也就是來源的 $\partial\mathbf{i}_t/\partial\mathbf{W}=\mathbf{h}_{t-1}^\top\otimes\mathbf{I}_p$。鏈鎖律接著把它轉置，乘上局部誤差：$(\mathbf{h}_{t-1}^\top\otimes\mathbf{I}_2)^\top\mathbf{g}_t=[h_1g_1,\ h_1g_2,\ h_2g_1,\ h_2g_2]^\top$。用 $\operatorname{vec}^{-1}$ 摺回 $2\times2$，第一行是 $h_1\mathbf{g}_t$、第二行是 $h_2\mathbf{g}_t$，正好是外積 $\mathbf{g}_t\mathbf{h}_{t-1}^\top$。

So the derivative of $\mathbf{i}_t$ with respect to $\operatorname{vec}(\mathbf{W})$ is that $2\times4$ matrix, which is the source's $\partial\mathbf{i}_t/\partial\mathbf{W}=\mathbf{h}_{t-1}^\top\otimes\mathbf{I}_p$. The chain rule then transposes it and multiplies by the local error: $(\mathbf{h}_{t-1}^\top\otimes\mathbf{I}_2)^\top\mathbf{g}_t=[h_1g_1,\ h_1g_2,\ h_2g_1,\ h_2g_2]^\top$. Folding it back into $2\times2$ with $\operatorname{vec}^{-1}$, the first column is $h_1\mathbf{g}_t$ and the second is $h_2\mathbf{g}_t$, which is exactly the outer product $\mathbf{g}_t\mathbf{h}_{t-1}^\top$.

來源的數字例：$\mathbf{g}_t=[0.5,\ 0.2]^\top$、$\mathbf{h}_{t-1}=[1.0,\ 0.0]^\top$。攤平再摺回，得到

The source's numerical example: $\mathbf{g}_t=[0.5,\ 0.2]^\top$, $\mathbf{h}_{t-1}=[1.0,\ 0.0]^\top$. Flattening and folding back gives

$$
\mathbf{g}_t\mathbf{h}_{t-1}^\top=\begin{bmatrix}0.5\\0.2\end{bmatrix}\begin{bmatrix}1.0&0.0\end{bmatrix}=\begin{bmatrix}0.5&0.0\\0.2&0.0\end{bmatrix}.
$$

第二行全是 $0$，因為 $h_2=0$：$\mathbf{W}$ 的第二行只乘 $h_2$，這一步沒用到它。程式驗證了兩種算法的結果完全相同 `[補充]`。

The second column is all $0$, because $h_2=0$: the second column of $\mathbf{W}$ only multiplies $h_2$, so it was not used at this step. The program confirms that the two methods give identical results `[Added]`.

![三格示意。左：二乘二的權重矩陣，四個元素塗成黃、紅、綠、紫，經過向量化箭頭，一行接一行疊成一根長條。中：二乘四的 Kronecker 矩陣，由兩塊「狀態分量乘單位矩陣」並排而成，例子的上一步狀態是 1.0 與 0.0。右：經過反向量化摺回的二乘二梯度矩陣，數值是 0.5、0.0、0.2、0.0，顏色位置和左邊的權重矩陣相同（Three panels: the weight matrix flattened column by column into one bar; the two-by-four Kronecker matrix made of two blocks, each a state component times the identity; the gradient folded back into a two-by-two matrix with values 0.5, 0.0, 0.2 and 0.0, colored like the matching weight entries）](../assets/figures/lec08-kp03-vec-kron.png)

**圖說 `lec08-kp03-vec-kron`**〔功能：一目了然〕：先看左邊：$\mathbf{W}$ 的四個顏色被 $\operatorname{vec}$ 一行一行疊成一根長條。再看中間：Kronecker 形式是 $2\times4$ 的矩陣，由兩塊 $h_j\mathbf{I}_2$ 並排而成；它乘上那根長條就得到 $\mathbf{W}\mathbf{h}_{t-1}$。最後看右邊：梯度用 $\operatorname{vec}^{-1}$ 摺回後，每個顏色回到和左邊 $\mathbf{W}$ 相同的位置，大小也是 $2\times2$，數值就是外積。

**Caption `lec08-kp03-vec-kron`** [Function: seeing it at a glance]: First look on the left: $\operatorname{vec}$ stacks the four colors of $\mathbf{W}$ column by column into one long bar. Then look in the middle: the Kronecker form is a $2\times4$ matrix made of two blocks $h_j\mathbf{I}_2$ side by side; multiplied by that bar it gives $\mathbf{W}\mathbf{h}_{t-1}$. Finally look on the right: after $\operatorname{vec}^{-1}$ folds the gradient back, every color returns to the same position as in $\mathbf{W}$, the size is $2\times2$ again, and the values are the outer product.

常見錯誤：以為有了 $\otimes$，梯度就變成 $p^2\times p^2$ 那麼大。Kronecker 積只出現在「攤平」的中間步驟；摺回之後，$\partial\mathcal{L}/\partial\mathbf{W}$ 一定是 $p\times p$。寫程式時也不必真的建出 $\mathbf{I}_p$，直接算外積就好。

Common error: thinking that with $\otimes$ the gradient becomes as large as $p^2\times p^2$. The Kronecker product appears only in the intermediate "flattened" step; after folding back, $\partial\mathcal{L}/\partial\mathbf{W}$ is always $p\times p$. In code you never need to build $\mathbf{I}_p$; you just compute the outer product.

### 5. 偏置的梯度：把誤差直接加起來（Bias gradients: add up the errors directly）

三個權重矩陣處理完，還剩兩個偏置。偏置可以看成「乘在常數 $1$ 上的權重」。套用第3節的規則「誤差 $\times$ 輸入的轉置」，輸入是 $1$，所以每一步的貢獻就是誤差本身：

With the three weight matrices done, two biases remain. A bias can be seen as "a weight that multiplies the constant $1$". Applying the rule of Section 3, "error times transposed input", with input $1$, each step contributes just the error itself:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{b}_y}=\sum_{t=1}^{T}\mathbf{e}_t,\qquad
\frac{\partial\mathcal{L}}{\partial\mathbf{b}_i}=\sum_{t=1}^{T}\mathbf{g}_t=\sum_{t=1}^{T}\operatorname{diag}(1-\mathbf{h}_t^2)\,\delta_t.
$$

偏置和權重一樣，所有時間步共用同一組，所以同樣要對 $t$ 加總。$\mathbf{b}_y$ 在 $\tanh$ 外面，直接加輸出誤差；$\mathbf{b}_i$ 在 $\tanh$ 裡面，要加局部誤差 $\mathbf{g}_t$。

Like the weights, the biases are shared by all time steps, so their gradients are also summed over $t$. $\mathbf{b}_y$ is outside the $\tanh$ and adds up the output errors; $\mathbf{b}_i$ is inside the $\tanh$ and adds up the local errors $\mathbf{g}_t$.

主例：$\partial\mathcal{L}/\partial b_y=1.0+1.0+1.0=3.0$；$\partial\mathcal{L}/\partial b_i=1.248+1.120+0.800=3.168$。這兩根就是圖 `lec08-kp03-step-sums` 最右邊的兩根長條。

Running example: $\partial\mathcal{L}/\partial b_y=1.0+1.0+1.0=3.0$; $\partial\mathcal{L}/\partial b_i=1.248+1.120+0.800=3.168$. These are the two rightmost bars of the figure `lec08-kp03-step-sums`.

常見錯誤：以為每一步有自己的偏置 $\mathbf{b}_{i,t}$。偏置和權重一樣在時間上共享，全程只有一個 $\mathbf{b}_i$、一個 $\mathbf{b}_y$。

Common error: thinking that every step has its own bias $\mathbf{b}_{i,t}$. Biases are shared across time like the weights; there is only one $\mathbf{b}_i$ and one $\mathbf{b}_y$ throughout.

### 6. 梯度下降：五組參數一起往反方向走一小步（Gradient descent: all five parameters take one small step against the gradient）

五個總梯度都算好了，最後一步是更新。**梯度下降**（gradient descent）：梯度指向損失上升最快的方向，所以每個參數都往梯度的反方向走一小步。**學習率**（learning rate）$\eta>0$ 決定這一步多大：

All five total gradients are ready, and the last step is the update. This is **gradient descent**: the gradient points in the direction in which the loss rises fastest, so every parameter takes a small step against the gradient. The **learning rate** $\eta>0$ sets how large that step is:

$$
\begin{aligned}
\mathbf{V}&:=\mathbf{V}-\eta\,\frac{\partial\mathcal{L}}{\partial\mathbf{V}}, &
\mathbf{W}&:=\mathbf{W}-\eta\,\frac{\partial\mathcal{L}}{\partial\mathbf{W}}, &
\mathbf{U}&:=\mathbf{U}-\eta\,\frac{\partial\mathcal{L}}{\partial\mathbf{U}},\\
\mathbf{b}_i&:=\mathbf{b}_i-\eta\,\frac{\partial\mathcal{L}}{\partial\mathbf{b}_i}, &
\mathbf{b}_y&:=\mathbf{b}_y-\eta\,\frac{\partial\mathcal{L}}{\partial\mathbf{b}_y}.
\end{aligned}
$$

$:=$ 的意思是「用右邊算出的新值取代左邊」。減號是因為要讓損失變小：$\eta$ 是正數，沿著梯度走損失會變大，反著走才會變小。來源的類比是調音響等化器：$\eta$ 是手轉旋鈕的幅度，太大會爆音，太小轉半天沒變化。這個類比的侷限是：等化器的旋鈕各自獨立，網路的參數卻互相牽動，一個改了，其他的梯度也跟著變。

$:=$ means "replace the left side by the new value computed on the right". The minus sign is there to make the loss smaller: $\eta$ is positive, moving along the gradient raises the loss, and moving against it lowers the loss. The source's analogy is tuning an audio equalizer: $\eta$ is how far your hand turns the knob; too far and the sound blasts, too little and nothing changes for a long time. The limit of the analogy: the knobs of an equalizer are independent, while the parameters of a network pull on one another, so changing one changes the gradients of the others.

主例，取 $\eta=0.1$ `[補充]`：

Running example with $\eta=0.1$ `[Added]`:

| 參數 | 舊值 | 總梯度 | 修改量 $-\eta\times$ 梯度 | 新值 |
|---|---|---|---|---|
| $V$ | $1.0$ | $1.3416$ | $-0.1342$ | $0.8658$ |
| $W$ | $0.5$ | $0.8587$ | $-0.0859$ | $0.4141$ |
| $U$ | $1.0$ | $1.0952$ | $-0.1095$ | $0.8905$ |
| $b_i$ | $0$ | $3.168$ | $-0.3168$ | $-0.3168$ |
| $b_y$ | $0$ | $3.0$ | $-0.3$ | $-0.3$ |

| Parameter | Old value | Total gradient | Change $-\eta\times$ gradient | New value |
|---|---|---|---|---|
| $V$ | $1.0$ | $1.3416$ | $-0.1342$ | $0.8658$ |
| $W$ | $0.5$ | $0.8587$ | $-0.0859$ | $0.4141$ |
| $U$ | $1.0$ | $1.0952$ | $-0.1095$ | $0.8905$ |
| $b_i$ | $0$ | $3.168$ | $-0.3168$ | $-0.3168$ |
| $b_y$ | $0$ | $3.0$ | $-0.3$ | $-0.3$ |

所有梯度都是正的，所以每個參數都往下調。用新參數重跑一次前向，總損失從 $1.5$ 降到 $0.0977$。程式也用「把參數加減一點點、看損失變多少」的數值微分，核對了這五個梯度，到小數第四位都相同 `[補充]`。

All gradients are positive, so every parameter moves down. Running the forward pass again with the new parameters, the total loss drops from $1.5$ to $0.0977$. The program also checked the five gradients by numerical differentiation, nudging each parameter a little and watching the loss, and they agree to four decimals `[Added]`.

![折線圖，橫軸是迭代次數 0 到 11，縱軸是對數刻度的總損失。三條線都從 1.5 出發。綠線 eta 等於 0.1：一步降到 0.0977，之後平緩降到約 0.0125。藍線 eta 等於 0.3：先到 0.8461，再快速降到約 0.0006。紅線 eta 等於 0.5：損失一路上升，第 11 次超過 4000（Loss curves on a log scale for three learning rates starting from 1.5: eta = 0.1 drops to 0.0977 after one step, eta = 0.3 falls to about 0.0006, and eta = 0.5 rises above 4000 by iteration 11）](../assets/figures/lec08-kp03-learning-rate.png)

**圖說 `lec08-kp03-learning-rate`**〔功能：精準度〕：先看三條線的起點：都是 $\mathcal{L}=1.5$，每次迭代都做完整的前向、BPTT 與更新。再看綠線與藍線：$\eta=0.1$ 一步就降到 $0.0977$；$\eta=0.3$ 起步慢一點，後來降得更低。最後看紅線：$\eta=0.5$ 步子太大，每次都跨過谷底，損失越來越大。縱軸是對數刻度，每一格差 $10$ 倍。

**Caption `lec08-kp03-learning-rate`** [Function: precision]: First look at where the three lines start: all at $\mathcal{L}=1.5$, and each iteration is a full forward pass, BPTT and update. Then look at the green and blue lines: $\eta=0.1$ drops to $0.0977$ in one step; $\eta=0.3$ starts more slowly and ends lower. Finally look at the red line: with $\eta=0.5$ the steps are too large, every step jumps over the valley, and the loss keeps growing. The vertical axis is logarithmic; each grid line is a factor of $10$.

假設條件與何時失效：梯度只描述「現在這一點附近」的坡度，所以 $\eta$ 要夠小，走完一步坡度不能差太多。$\eta$ 太大時會越過谷底，甚至發散，像紅線那樣。哪個 $\eta$ 算「太大」，要看問題本身，本例 $0.5$ 就已經太大。另一個失效情形是序列很長時梯度本身爆大，固定的 $\eta$ 也會失控，這是 KP04 的主題。

Assumptions and when it fails: the gradient only describes the slope "near the current point", so $\eta$ must be small enough that the slope does not change too much within one step. When $\eta$ is too large, the step jumps over the valley and can even diverge, like the red line. Which $\eta$ counts as "too large" depends on the problem; here $0.5$ is already too large. Another failure is a very long sequence whose gradient itself blows up, so that a fixed $\eta$ also goes out of control; that is the topic of KP04.

與實作的連結 `[補充]`：程式裡通常先把五個梯度設成 $0$，在反向傳播的迴圈中每一步用 `+=` 累加外積；**全部 $T$ 步加完之後**，才一次更新參數。下面是對應的 NumPy 寫法（`e[t]`、`g[t]` 已由反向傳播算好）：

Link to code `[Added]`: a program usually sets the five gradients to $0$, adds the outer products step by step with `+=` inside the backward loop, and updates the parameters once **after all $T$ steps have been added**. Below is the matching NumPy code (`e[t]` and `g[t]` come from the backward pass):

```python
dV, dW, dU = np.zeros_like(V), np.zeros_like(W), np.zeros_like(U)
dbi, dby = np.zeros_like(b_i), np.zeros_like(b_y)
for t in range(1, T + 1):          # e[t], g[t] come from the backward pass
    dV  += np.outer(e[t], h[t])
    dby += e[t]
    dW  += np.outer(g[t], h[t - 1])
    dU  += np.outer(g[t], x[t])
    dbi += g[t]
for P, dP in ((V, dV), (W, dW), (U, dU), (b_y, dby), (b_i, dbi)):
    P -= eta * dP                  # update only after the sum is complete
```

常見錯誤：每倒推一步就馬上改一次 $\mathbf{W}$。這樣後面幾步用的 $\mathbf{W}$ 已經不是前向時那一個，算出的梯度就對不上，也破壞了「每一步同一組權重」的前提。要先把 $T$ 步全部加完，再一起更新。

Common error: changing $\mathbf{W}$ right after each backward step. Then the remaining steps use a $\mathbf{W}$ that is no longer the one of the forward pass, the gradients no longer match, and the premise "the same weights at every step" is broken. Add up all $T$ steps first, then update everything at once.

### 本單元重點整理（KP03 summary）

**一句話重點**：共享參數的梯度是「每一步的誤差 $\times$ 該步輸入的轉置」對所有時間步的總和；$\mathbf{V}$ 用輸出誤差 $\mathbf{e}_t$，$\mathbf{W}$、$\mathbf{U}$、$\mathbf{b}_i$ 用局部誤差 $\mathbf{g}_t=(1-\mathbf{h}_t^2)\odot\delta_t$；全部加完後，五組參數一起做 $\theta:=\theta-\eta\,\partial\mathcal{L}/\partial\theta$。

**One-sentence summary**: the gradient of a shared parameter is the sum over all time steps of "the error of the step times the transposed input of the step"; $\mathbf{V}$ uses the output error $\mathbf{e}_t$, while $\mathbf{W}$, $\mathbf{U}$ and $\mathbf{b}_i$ use the local error $\mathbf{g}_t=(1-\mathbf{h}_t^2)\odot\delta_t$; after everything is added, all five parameters update together by $\theta:=\theta-\eta\,\partial\mathcal{L}/\partial\theta$.

**白話說明**：同一組權重在每一步都上過場，所以每一步都會交出一份「這裡要怎麼改」的建議，就是一個外積。把所有建議加起來，才是這組權重該改的方向。最後每個參數都沿著損失下降的方向走一小步，步子大小由學習率決定。

**Plain explanation**: the same weights went on stage at every step, so every step hands in a suggestion of "how to change here", which is one outer product. Only the sum of all suggestions tells the weights which way to change. Finally every parameter takes a small step in the direction in which the loss goes down, with the step size set by the learning rate.

**正式定義**：$\mathbf{g}_t=\operatorname{diag}(1-\mathbf{h}_t^2)\,\delta_t$；$\partial\mathcal{L}/\partial\mathbf{V}=\sum_t\mathbf{e}_t\mathbf{h}_t^\top$，$\partial\mathcal{L}/\partial\mathbf{W}=\sum_t\mathbf{g}_t\mathbf{h}_{t-1}^\top$，$\partial\mathcal{L}/\partial\mathbf{U}=\sum_t\mathbf{g}_t\mathbf{x}_t^\top$，$\partial\mathcal{L}/\partial\mathbf{b}_y=\sum_t\mathbf{e}_t$，$\partial\mathcal{L}/\partial\mathbf{b}_i=\sum_t\mathbf{g}_t$；更新 $\theta:=\theta-\eta\,\partial\mathcal{L}/\partial\theta$，$\eta>0$。

**Formal definition**: $\mathbf{g}_t=\operatorname{diag}(1-\mathbf{h}_t^2)\,\delta_t$; $\partial\mathcal{L}/\partial\mathbf{V}=\sum_t\mathbf{e}_t\mathbf{h}_t^\top$, $\partial\mathcal{L}/\partial\mathbf{W}=\sum_t\mathbf{g}_t\mathbf{h}_{t-1}^\top$, $\partial\mathcal{L}/\partial\mathbf{U}=\sum_t\mathbf{g}_t\mathbf{x}_t^\top$, $\partial\mathcal{L}/\partial\mathbf{b}_y=\sum_t\mathbf{e}_t$, $\partial\mathcal{L}/\partial\mathbf{b}_i=\sum_t\mathbf{g}_t$; update $\theta:=\theta-\eta\,\partial\mathcal{L}/\partial\theta$ with $\eta>0$.

**例子**：KP02 的 3 步例子（$W=0.5$、$U=1.0$、$V=1.0$、$\delta_t=1.56,1.40,1.00$）：$g_t=1.248,1.120,0.800$；$\partial\mathcal{L}/\partial V=1.3416$、$\partial\mathcal{L}/\partial W=0.8587$、$\partial\mathcal{L}/\partial U=1.0952$、$\partial\mathcal{L}/\partial b_i=3.168$、$\partial\mathcal{L}/\partial b_y=3.0$。$\eta=0.1$ 更新後 $W=0.4141$，總損失由 $1.5$ 降到 $0.0977$。

**Example**: the 3-step example of KP02 ($W=0.5$, $U=1.0$, $V=1.0$, $\delta_t=1.56,1.40,1.00$): $g_t=1.248,1.120,0.800$; $\partial\mathcal{L}/\partial V=1.3416$, $\partial\mathcal{L}/\partial W=0.8587$, $\partial\mathcal{L}/\partial U=1.0952$, $\partial\mathcal{L}/\partial b_i=3.168$, $\partial\mathcal{L}/\partial b_y=3.0$. After the update with $\eta=0.1$, $W=0.4141$ and the total loss drops from $1.5$ to $0.0977$.

**常見迷思**：

- 「$\mathbf{W}$ 的梯度直接用 $\delta_t\mathbf{h}_{t-1}^\top$。」錯，要先乘 $\tanh$ 的導數，用 $\mathbf{g}_t$。
- 「只用最後一步的梯度更新就好。」錯，共享參數要把每一步的貢獻加總。
- 「有 Kronecker 積，梯度就變成 $p^2\times p^2$。」錯，摺回後一定和原矩陣一樣大。
- 「每倒推一步就更新一次權重。」錯，要 $T$ 步全部加完才更新。

**Common misconceptions**:

- "The gradient of $\mathbf{W}$ is simply $\delta_t\mathbf{h}_{t-1}^\top$." Wrong: multiply by the $\tanh$ derivative first and use $\mathbf{g}_t$.
- "Updating with the gradient of the last step is enough." Wrong: a shared parameter must add up the contributions of every step.
- "With a Kronecker product the gradient becomes $p^2\times p^2$." Wrong: after folding back it always has the size of the original matrix.
- "Update the weights after every backward step." Wrong: update only after all $T$ steps are added.

**圖**：本單元有三張圖。

- `lec08-kp03-step-sums`：五個總梯度各由三步的貢獻疊成，$W$ 的第 $1$ 步是 $0$。
- `lec08-kp03-vec-kron`：$\operatorname{vec}$、Kronecker 積與 $\operatorname{vec}^{-1}$ 摺回後就是外積。
- `lec08-kp03-learning-rate`：三種學習率下的損失變化，太大的 $\eta$ 會發散。

**Figures** — this unit has three figures.

- `lec08-kp03-step-sums`: each of the five total gradients is stacked from three step contributions, and step $1$ of $W$ is $0$.
- `lec08-kp03-vec-kron`: $\operatorname{vec}$, the Kronecker product and the fold-back by $\operatorname{vec}^{-1}$ give the outer product.
- `lec08-kp03-learning-rate`: the loss under three learning rates; a too-large $\eta$ diverges.

### 小練習（Practice）

1. 來源的例子：純量權重 $W=0.80$，總梯度 $\partial\mathcal{L}/\partial W=0.50$，$\eta=0.1$。更新後 $W$ 是多少？如果 $\eta=10$ 呢？提示：新值 $=$ 舊值 $-\eta\times$ 梯度。
2. 設 $p=2$、$q=1$。某一步的輸出誤差 $e_t=0.5$，狀態 $\mathbf{h}_t=[0.6,\ -0.2]^\top$。這一步對 $\partial\mathcal{L}/\partial\mathbf{V}$ 的貢獻是多大的矩陣？數值是多少？提示：外積 $\mathbf{e}_t\mathbf{h}_t^\top$ 是 $(q\times1)(1\times p)$。
3. 沿用主例，但只有最後一步有損失（KP02 的對照情形），此時 $\delta_1=0.16$、$\delta_2=0.40$、$\delta_3=1.00$，$e_1=e_2=0$、$e_3=1.0$。算出 $\partial\mathcal{L}/\partial b_i$、$\partial\mathcal{L}/\partial W$、$\partial\mathcal{L}/\partial b_y$。提示：先算 $g_t=0.8\,\delta_t$；$h_0=0$、$h_1=h_2=0.4472$。

1. The source's example: scalar weight $W=0.80$, total gradient $\partial\mathcal{L}/\partial W=0.50$, $\eta=0.1$. What is $W$ after the update? And with $\eta=10$? Hint: new value $=$ old value $-\eta\times$ gradient.
2. Let $p=2$, $q=1$. At some step the output error is $e_t=0.5$ and the state is $\mathbf{h}_t=[0.6,\ -0.2]^\top$. What size is this step's contribution to $\partial\mathcal{L}/\partial\mathbf{V}$, and what are its values? Hint: the outer product $\mathbf{e}_t\mathbf{h}_t^\top$ is $(q\times1)(1\times p)$.
3. Keep the running example, but only the last step has a loss (the comparison case of KP02), so $\delta_1=0.16$, $\delta_2=0.40$, $\delta_3=1.00$, $e_1=e_2=0$ and $e_3=1.0$. Compute $\partial\mathcal{L}/\partial b_i$, $\partial\mathcal{L}/\partial W$ and $\partial\mathcal{L}/\partial b_y$. Hint: first compute $g_t=0.8\,\delta_t$; $h_0=0$ and $h_1=h_2=0.4472$.

**第 1 題解答** `[補充]`：$\eta=0.1$ 時 $W=0.80-0.1\times0.50=0.75$。$\eta=10$ 時 $W=0.80-10\times0.50=-4.2$，一步就衝到很遠的地方，正是「學習率太大」的樣子。

**Solution to question 1** `[Added]`: with $\eta=0.1$, $W=0.80-0.1\times0.50=0.75$. With $\eta=10$, $W=0.80-10\times0.50=-4.2$; one step shoots far away, which is exactly what "a learning rate that is too large" looks like.

**第 2 題解答** `[補充]`：大小是 $1\times2$，和 $\mathbf{V}$ 一樣。數值是 $0.5\times[0.6,\ -0.2]=[0.3,\ -0.1]$。

**Solution to question 2** `[Added]`: the size is $1\times2$, the same as $\mathbf{V}$. The values are $0.5\times[0.6,\ -0.2]=[0.3,\ -0.1]$.

**第 3 題解答** `[補充]`：$g_t=0.128,\ 0.320,\ 0.800$，所以 $\partial\mathcal{L}/\partial b_i=0.128+0.320+0.800=1.248$。$\partial\mathcal{L}/\partial W=0.128\times0+0.320\times0.4472+0.800\times0.4472=0.5009$。$\partial\mathcal{L}/\partial b_y=0+0+1.0=1.0$。前面兩步雖然沒有損失，仍然對 $b_i$、$W$ 有貢獻，因為它們影響了第 $3$ 步。

**Solution to question 3** `[Added]`: $g_t=0.128,\ 0.320,\ 0.800$, so $\partial\mathcal{L}/\partial b_i=0.128+0.320+0.800=1.248$. $\partial\mathcal{L}/\partial W=0.128\times0+0.320\times0.4472+0.800\times0.4472=0.5009$. $\partial\mathcal{L}/\partial b_y=0+0+1.0=1.0$. The first two steps have no loss, yet they still contribute to $b_i$ and $W$, because they influence step $3$.

### 小測驗（Quiz，來源原題）

來源的兩題自我檢測保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The two self-check questions of the source are kept with all options; the source page does not mark the answers directly, and the explanations below follow the source text.

1. 在計算隱藏到隱藏權重 $\mathbf{W}$ 的梯度時，為什麼會用到 Kronecker 積 $(\otimes)$？
   - (A) 為了計算矩陣的逆矩陣
   - (B) 因為矩陣與向量求導時，將矩陣展開為向量處理（Vectorization），矩陣乘法求導自然產生 Kronecker 積結構
   - (C) 為了進行卷積運算
   - (D) 為了降低記憶體使用
2. 使用 BPTT 算完所有梯度後，梯度下降參數更新的符號方向為何是減號 $(-)$？
   - (A) 為了沿著梯度的反方向移動以最小化損失函數（Gradient Descent）
   - (B) 為了最大化準確率
   - (C) 這只是習慣寫法
   - (D) 因為學習率 $\eta$ 是負數

1. When computing the gradient of the hidden-to-hidden weight $\mathbf{W}$, why is the Kronecker product $(\otimes)$ used?
   - (A) To compute the inverse of a matrix
   - (B) Because when differentiating with respect to a matrix, the matrix is flattened into a vector (vectorization), and differentiating a matrix product naturally gives a Kronecker-product structure
   - (C) To perform a convolution
   - (D) To reduce memory use
2. After BPTT has computed all gradients, why does the gradient-descent update use a minus sign $(-)$?
   - (A) To move against the gradient so as to minimize the loss function (gradient descent)
   - (B) To maximize the accuracy
   - (C) It is only a habit of notation
   - (D) Because the learning rate $\eta$ is negative

**解析** `[補充]`：第 1 題選 (B)：第4節說明攤平 $\mathbf{W}$ 之後，$\mathbf{W}\mathbf{h}_{t-1}$ 變成 $(\mathbf{h}_{t-1}^\top\otimes\mathbf{I}_p)\operatorname{vec}(\mathbf{W})$；它和逆矩陣、卷積、記憶體都無關，程式實作反而不建出 Kronecker 積。第 2 題選 (A)：第6節說明梯度指向損失上升最快的方向；$\eta$ 是正數，所以 (D) 錯；減號決定方向，不是習慣，所以 (C) 錯；目標是讓損失變小，不是直接最大化準確率，所以 (B) 錯。

**Explanations** `[Added]`: question 1 is B: Section 4 shows that after flattening $\mathbf{W}$, $\mathbf{W}\mathbf{h}_{t-1}$ becomes $(\mathbf{h}_{t-1}^\top\otimes\mathbf{I}_p)\operatorname{vec}(\mathbf{W})$; this has nothing to do with inverses, convolution or memory, and code does not even build the Kronecker product. Question 2 is A: Section 6 explains that the gradient points where the loss rises fastest; $\eta$ is positive, so D is wrong; the minus sign sets the direction and is not a habit, so C is wrong; the goal is a smaller loss, not directly a higher accuracy, so B is wrong.

### 留給下一單元的問題（The question for the next unit）

到這裡，BPTT 的閉環已經完整：前向、加總損失、倒算 $\delta_t$、加總外積、更新。可是主例只有 $3$ 步。如果序列有 $50$ 步或 $100$ 步，第 $T$ 步的誤差要乘過幾十次 $W(1-h^2)$ 才能回到第 $1$ 步，它還留得住嗎？下一單元（KP04）回答這個問題。

The BPTT loop is now complete: forward pass, summed loss, backward $\delta_t$, summed outer products, update. But the running example has only $3$ steps. If the sequence has $50$ or $100$ steps, the error of step $T$ must be multiplied by $W(1-h^2)$ dozens of times to get back to step $1$. Does it survive? The next unit (KP04) answers this question.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第8章「循環神經網絡與長短期記憶網絡」§2.3.3–2.3.8 與其教授祕笈（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch08-rnn-lstm/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §2.3.3–2.3.8 of Chapter 8, "Recurrent neural networks and long short-term memory networks", of CCC-LNS IDL_115 and its professor's notes (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch08-rnn-lstm/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源教授祕笈的外積例寫 $\Delta\mathbf{W}_t=\delta_t\mathbf{h}_{t-1}^\top$，漏了 $\tanh$ 的導數。正確是 $\Delta\mathbf{W}_t=\mathbf{g}_t\mathbf{h}_{t-1}^\top$，$\mathbf{g}_t=(1-\mathbf{h}_t^2)\odot\delta_t$；來源自己的偏置公式 $\operatorname{diag}(1-\mathbf{h}_t^2)\delta_t$ 也支持這一點。本講義把該例的 $[0.5,\ 0.2]^\top$ 解讀為 $\mathbf{g}_t$。
- 來源的 $\mathbf{W}$、$\mathbf{U}$ 總梯度寫成 $\operatorname{vec}^{-1}((\partial\mathbf{h}_t/\partial\mathbf{W})^\top\delta_t)$。這裡的 $\partial\mathbf{h}_t/\partial\mathbf{W}$ 必須是把 $\mathbf{h}_{t-1}$ 當常數的「直接」導數，即 $\operatorname{diag}(1-\mathbf{h}_t^2)(\mathbf{h}_{t-1}^\top\otimes\mathbf{I}_p)$；否則間接路徑會被算兩次。本講義在第3節說明，結論不變。
- 來源例子的 $\mathbf{h}_{t-1}=[1.0,\ 0.0]^\top$ 不可能是真正的 $\tanh$ 狀態（$\tanh$ 碰不到 $1$），本講義保留原數字，當作示意用的向量。
- 來源更新公式的排版在同一列寫了兩個對齊符號 `&:=`，本講義重排成兩列。
- 來源互動展示寫「適當學習率可確保損失二次收斂」。梯度下降並不保證二次收斂；本講義只說 $\eta$ 夠小時損失會下降，並用圖 `lec08-kp03-learning-rate` 顯示 $\eta=0.5$ 會發散。來源說 $W$ 由 $0.80$ 改成 $0.75$ 後「損失成功下降」，也只在 $\eta$ 夠小時成立。
- 來源用粗斜體 `\boldsymbol` 表示向量；本講義改用 `\mathbf`，意義相同。來源把本單元標為 KP-5、KP-6，下一單元標為 KP-7；本課程對應 KP03 與 KP04。
- 本講義補上：局部誤差 $\mathbf{g}_t$ 的名稱與表格、主例的補齊（$U=1.0$、輸入 $0.4812,0.2576,0.2576$、目標 $-0.5528$）、五個梯度的完整數值與數值微分核對、「誤差 $\times$ 輸入轉置」的記法、$h_0=0$ 使第 $1$ 步對 $W$ 無貢獻的說明、$p=2$ 的 Kronecker 展開式、一次完整更新與損失變化、學習率比較圖、NumPy 程式片段、三題小練習與解答。
- 取捨：來源的兩題自我檢測保留原題；來源的互動展示無法放進講義，改用圖 `lec08-kp03-step-sums` 與 `lec08-kp03-learning-rate` 呈現；來源提到的 Adam 與動量優化器不在本單元展開。

- The source's professor's notes write the outer-product example as $\Delta\mathbf{W}_t=\delta_t\mathbf{h}_{t-1}^\top$, which misses the $\tanh$ derivative. The correct form is $\Delta\mathbf{W}_t=\mathbf{g}_t\mathbf{h}_{t-1}^\top$ with $\mathbf{g}_t=(1-\mathbf{h}_t^2)\odot\delta_t$; the source's own bias formula $\operatorname{diag}(1-\mathbf{h}_t^2)\delta_t$ supports this. This note reads the example's $[0.5,\ 0.2]^\top$ as $\mathbf{g}_t$.
- The source writes the total gradients of $\mathbf{W}$ and $\mathbf{U}$ as $\operatorname{vec}^{-1}((\partial\mathbf{h}_t/\partial\mathbf{W})^\top\delta_t)$. Here $\partial\mathbf{h}_t/\partial\mathbf{W}$ must be the "direct" derivative with $\mathbf{h}_{t-1}$ held constant, namely $\operatorname{diag}(1-\mathbf{h}_t^2)(\mathbf{h}_{t-1}^\top\otimes\mathbf{I}_p)$; otherwise the indirect paths would be counted twice. Section 3 explains this; the conclusion is unchanged.
- The source example's $\mathbf{h}_{t-1}=[1.0,\ 0.0]^\top$ cannot be a real $\tanh$ state ($\tanh$ never reaches $1$); this note keeps the numbers as an illustrative vector.
- The source's update formula puts two alignment marks `&:=` in one row; this note typesets it in two rows.
- The source's interactive demo says that "a suitable learning rate guarantees quadratic convergence of the loss". Gradient descent does not guarantee quadratic convergence; this note only says the loss decreases when $\eta$ is small enough, and the figure `lec08-kp03-learning-rate` shows that $\eta=0.5$ diverges. The source's claim that the loss "successfully decreases" after $W$ changes from $0.80$ to $0.75$ also holds only for a small enough $\eta$.
- The source writes vectors with bold italic `\boldsymbol`; this note uses `\mathbf` with the same meaning. The source labels this unit KP-5 and KP-6 and the next unit KP-7; in this course they are KP03 and KP04.
- This note adds: the name and table of the local error $\mathbf{g}_t$, the completion of the running example ($U=1.0$, inputs $0.4812,0.2576,0.2576$, targets $-0.5528$), the full values of the five gradients with a numerical-differentiation check, the "error times transposed input" rule, the explanation that $h_0=0$ makes step $1$ contribute nothing to $W$, the $p=2$ Kronecker expansion, one full update with its loss change, the learning-rate figure, the NumPy snippet, and three practice questions with solutions.
- Choices: the two self-check questions of the source are kept as asked; the source's interactive demos cannot go into a note, so the figures `lec08-kp03-step-sums` and `lec08-kp03-learning-rate` show the same ideas; the Adam and momentum optimizers mentioned by the source are not developed in this unit.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 隨時間反向傳播 | backpropagation through time (BPTT) |
| 循環神經網路 | recurrent neural network (RNN) |
| 狀態梯度 | state gradient |
| 輸出誤差 | output error |
| 局部誤差 | local error |
| 淨輸入 | net input |
| 鏈鎖律 | chain rule |
| 外積 | outer product |
| 向量化 | vectorization |
| 反向量化 | inverse vectorization |
| Kronecker 積 | Kronecker product |
| 單位矩陣 | identity matrix |
| 偏置 | bias |
| 梯度下降 | gradient descent |
| 學習率 | learning rate |
| 數值微分 | numerical differentiation |
| 發散 | divergence |
| 動量優化器（來源提及） | momentum optimizer |
| 自適應矩估計優化器（來源提及） | Adam |
| 長短期記憶網路（KP06） | long short-term memory (LSTM) |
| 本課程的網頁教材 | CCC-LNS |
