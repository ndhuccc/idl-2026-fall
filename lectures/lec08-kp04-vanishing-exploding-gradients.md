# 第8週 KP04：梯度連乘越多步，就越容易消失或爆炸（Vanishing and exploding gradients: the longer the chain, the smaller or larger the gradient）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第8週「循環神經網路：RNN、LSTM 與 GRU」的第四個知識點（KP04），接在 KP03 的權重梯度之後。KP03 已經能算出共用權重 $\mathbf{W}$ 的梯度。這一節問：序列長達 $20$ 步、$50$ 步時，最後一步的誤差倒傳回第 $1$ 步，梯度還留得住嗎？

This is the fourth knowledge point (KP04) of Week 8, "Recurrent neural networks: RNN, LSTM and GRU", and it follows the weight gradients of KP03. KP03 can already compute the gradient of the shared weight $\mathbf{W}$. This note asks: when a sequence is $20$ or $50$ steps long and the error of the last step travels back to step $1$, does the gradient survive?

先備知識是 KP01 的前向公式、KP02 的隱藏狀態梯度，以及 $\tanh$ 的導數。第一次出現的詞與符號都會定義。所有數字都由 `assets/figures/src/lec08-kp04-figures.py` 算出，可以自己重算。

The prerequisites are the forward equations of KP01, the hidden-state gradient of KP02 and the derivative of $\tanh$. Every term and symbol is defined when it first appears. Every number is computed by `assets/figures/src/lec08-kp04-figures.py`, so you can recompute it yourself.

## KP04 梯度連乘越多步，就越容易消失或爆炸（Vanishing and exploding gradients: the longer the chain, the smaller or larger the gradient）

### 學習目標（Learning objectives）

讀完之後，你能寫出「跨 $k$ 步的梯度是 $k$ 個雅可比矩陣相乘」這條式子。你能用 $\gamma^k$ 說明乘數小於 $1$ 時梯度消失、大於 $1$ 時梯度爆炸。你能用 $\tanh'\le1$ 解釋為什麼消失比爆炸常見。你也能手算一次梯度裁剪，並說明它為什麼救不了梯度消失。

After reading, you can write the equation "the gradient across $k$ steps is a product of $k$ Jacobian matrices". You can use $\gamma^k$ to explain that the gradient vanishes when the factor is below $1$ and explodes when it is above $1$. You can use $\tanh'\le1$ to explain why vanishing is more common than exploding. You can also clip a gradient by hand and explain why clipping cannot cure a vanishing gradient.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{h}_t\in\mathbb{R}^{p}$ | 第 $t$ 步的隱藏狀態 | hidden state at step $t$ |
| $\mathbf{i}_t\in\mathbb{R}^{p}$ | 第 $t$ 步進入 $\tanh$ 之前的淨輸入 | net input at step $t$, before $\tanh$ |
| $\mathbf{W}\in\mathbb{R}^{p\times p}$ | 上一步狀態到這一步狀態的權重 | hidden-to-hidden (state transition) weights |
| $\mathbf{U},\ \mathbf{V},\ \mathbf{b}_i$ | 輸入權重、輸出權重、隱藏層偏置（同 KP01） | input weights, output weights, hidden bias (as in KP01) |
| $\mathcal{L}_t,\ \mathcal{L}$ | 第 $t$ 步的損失、全部步的總損失 | loss at step $t$, total loss over all steps |
| $T$ | 序列最後一步 | last step of the sequence |
| $k=T-t$ | 梯度要倒傳的步數 | number of steps the gradient travels back |
| $\mathbf{J}_s$ | 第 $s$ 步的雅可比矩陣 $\partial\mathbf{h}_s/\partial\mathbf{h}_{s-1}$ | Jacobian matrix of step $s$, $\partial\mathbf{h}_s/\partial\mathbf{h}_{s-1}$ |
| $\operatorname{diag}(\mathbf{a})$ | 對角線放向量 $\mathbf{a}$、其餘為 $0$ 的方陣 | square matrix with $\mathbf{a}$ on the diagonal and $0$ elsewhere |
| $\gamma$ | 每倒傳一步，梯度大小被乘上的倍數 | factor that multiplies the gradient size at every backward step |
| $\mathbf{g},\ \|\mathbf{g}\|$ | 梯度向量與它的長度 | gradient vector and its length |
| $c$ | 裁剪時的長度門檻 | length threshold used for clipping |

### 1. 梯度要一站一站倒傳回去（The gradient travels back station by station）

KP03 留下的問題是：倒傳 $50$ 步，梯度還在不在？要回答它，先看梯度怎麼從第 $T$ 步走到第 $t$ 步。KP02 說過，$\mathbf{h}_t$ 只能透過 $\mathbf{h}_{t+1}$ 影響後面的步驟。所以從 $\mathbf{h}_T$ 回到 $\mathbf{h}_t$，必須一站一站走。

KP03 left the question: after $50$ backward steps, is the gradient still there? To answer it, first look at how the gradient goes from step $T$ to step $t$. KP02 showed that $\mathbf{h}_t$ can affect later steps only through $\mathbf{h}_{t+1}$. So the way back from $\mathbf{h}_T$ to $\mathbf{h}_t$ must pass every station.

微積分的連鎖律把這段路寫成連乘：

The chain rule of calculus writes this path as a product:

$$
\frac{\partial\mathbf{h}_T}{\partial\mathbf{h}_t}=\frac{\partial\mathbf{h}_T}{\partial\mathbf{h}_{T-1}}\,\frac{\partial\mathbf{h}_{T-1}}{\partial\mathbf{h}_{T-2}}\cdots\frac{\partial\mathbf{h}_{t+1}}{\partial\mathbf{h}_t}=\mathbf{J}_T\,\mathbf{J}_{T-1}\cdots\mathbf{J}_{t+1}.
$$

每一個因子 $\mathbf{J}_s=\partial\mathbf{h}_s/\partial\mathbf{h}_{s-1}$ 都是 $p\times p$ 的矩陣。它叫**雅可比矩陣**（Jacobian matrix）。第 $(a,b)$ 格的意思是：$\mathbf{h}_{s-1}$ 的第 $b$ 個分量變一點點，$\mathbf{h}_s$ 的第 $a$ 個分量跟著變多少。從 $t$ 到 $T$ 共有 $k=T-t$ 個因子。

Every factor $\mathbf{J}_s=\partial\mathbf{h}_s/\partial\mathbf{h}_{s-1}$ is a $p\times p$ matrix called a **Jacobian matrix**. Its entry $(a,b)$ says "if component $b$ of $\mathbf{h}_{s-1}$ changes a little, how much component $a$ of $\mathbf{h}_s$ changes". From $t$ to $T$ there are $k=T-t$ factors.

一個因子長什麼樣子？由 KP01 的 $\mathbf{h}_s=\tanh(\mathbf{i}_s)$、$\mathbf{i}_s=\mathbf{W}\mathbf{h}_{s-1}+\mathbf{U}\mathbf{x}_s+\mathbf{b}_i$，分兩步微分。先是 $\partial\mathbf{i}_s/\partial\mathbf{h}_{s-1}=\mathbf{W}$。再用 $\tanh'(i)=1-\tanh^2(i)$，每個分量各自乘上自己的斜率：

What does one factor look like? From KP01, $\mathbf{h}_s=\tanh(\mathbf{i}_s)$ and $\mathbf{i}_s=\mathbf{W}\mathbf{h}_{s-1}+\mathbf{U}\mathbf{x}_s+\mathbf{b}_i$, so differentiate in two steps. First, $\partial\mathbf{i}_s/\partial\mathbf{h}_{s-1}=\mathbf{W}$. Then use $\tanh'(i)=1-\tanh^2(i)$: every component is multiplied by its own slope:

$$
\mathbf{J}_s=\operatorname{diag}\!\left(1-\mathbf{h}_s^2\right)\mathbf{W}.
$$

這裡 $\mathbf{h}_s^2$ 是逐分量平方。反向傳播時，梯度向量乘的是轉置 $\mathbf{J}_s^\top=\mathbf{W}^\top\operatorname{diag}(1-\mathbf{h}_s^2)$，所以有：

Here $\mathbf{h}_s^2$ is squared component by component. In backpropagation the gradient vector is multiplied by the transpose $\mathbf{J}_s^\top=\mathbf{W}^\top\operatorname{diag}(1-\mathbf{h}_s^2)$, which gives:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{h}_t}\ \text{(via step }T\text{)}=\mathbf{J}_{t+1}^\top\cdots\mathbf{J}_T^\top\,\frac{\partial\mathcal{L}_T}{\partial\mathbf{h}_T},\qquad \mathbf{J}_s^\top=\mathbf{W}^\top\operatorname{diag}\!\left(1-\mathbf{h}_s^2\right).
$$

這就是來源寫的「$\mathbf{W}^\top$ 乘上 $\tanh$ 導數的對角矩陣」。重點只有一個：同一個 $\mathbf{W}$ 被乘了 $k$ 次，中間夾著 $k$ 個 $\tanh$ 斜率。

This is the "$\mathbf{W}^\top$ times the diagonal matrix of $\tanh$ derivatives" in the source. Only one point matters: the same $\mathbf{W}$ is multiplied $k$ times, with $k$ $\tanh$ slopes in between.

常見錯誤：以為梯度從 $\mathbf{h}_T$ 可以直接跳到 $\mathbf{h}_t$。在標準 RNN 裡沒有這種捷徑，每一步都要經過一次 $\mathbf{W}$ 和一次 $\tanh$。

Common error: thinking that the gradient can jump straight from $\mathbf{h}_T$ to $\mathbf{h}_t$. A standard RNN has no such shortcut; every step passes once through $\mathbf{W}$ and once through $\tanh$.

### 2. 連乘 $k$ 次：指數衰減或指數暴增（Multiplying $k$ times: exponential decay or exponential growth）

上一節說梯度要被乘 $k$ 次。乘 $k$ 次會怎樣？先把矩陣換成一個數。假設每倒傳一步，梯度的大小都被乘上同一個倍數 $\gamma$。走了 $k$ 步，大小就變成原來的 $\gamma^k$ 倍。

The previous section said the gradient is multiplied $k$ times. What do $k$ multiplications do? First replace the matrix by one number. Suppose every backward step multiplies the gradient size by the same factor $\gamma$. After $k$ steps the size is $\gamma^k$ times the original.

$\gamma^k$ 隨 $k$ 的變化是**指數**的，不是線性的。若 $\gamma<1$，$\gamma^k$ 很快趨近 $0$，這叫**梯度消失**（vanishing gradient）。若 $\gamma>1$，$\gamma^k$ 很快變得巨大，這叫**梯度爆炸**（exploding gradient）。只有 $\gamma=1$ 時大小不變。

$\gamma^k$ changes with $k$ **exponentially**, not linearly. If $\gamma<1$, $\gamma^k$ quickly approaches $0$; this is called a **vanishing gradient**. If $\gamma>1$, $\gamma^k$ quickly becomes huge; this is called an **exploding gradient**. Only $\gamma=1$ keeps the size unchanged.

類比：這像一排人玩傳話遊戲。若每個人都把音量降低 $20\%$（乘 $0.8$），傳到第 $20$ 個人只剩約 $1\%$ 的音量。若每個人都放大 $20\%$（乘 $1.2$），第 $20$ 個人聽到的是原來的 $38$ 倍。這個類比的侷限是：聲音只有大小，梯度還有方向，矩陣連乘也會改變方向。

Analogy: this is like a line of people playing a whisper game. If everyone lowers the volume by $20\%$ (times $0.8$), the twentieth person hears only about $1\%$ of the volume. If everyone raises it by $20\%$ (times $1.2$), the twentieth person hears $38$ times the original. The limit of the analogy: a sound has only a size, while a gradient also has a direction, and a product of matrices also changes the direction.

用數字看兩種乘數走 $20$ 步與 $50$ 步：

Look at the numbers for two pairs of factors over $20$ and $50$ steps:

| 乘數 $\gamma$ | $\gamma^{5}$ | $\gamma^{10}$ | $\gamma^{20}$ | $\gamma^{50}$ |
|---|---|---|---|---|
| $0.8$ | $0.3277$ | $0.1074$ | $0.0115$ | $0.0000143$ |
| $0.85$ | $0.4437$ | $0.1969$ | $0.0388$ | $0.000296$ |
| $1.15$ | $2.011$ | $4.046$ | $16.37$ | $1083.66$ |
| $1.2$ | $2.488$ | $6.192$ | $38.34$ | $9100.44$ |

| Factor $\gamma$ | $\gamma^{5}$ | $\gamma^{10}$ | $\gamma^{20}$ | $\gamma^{50}$ |
|---|---|---|---|---|
| $0.8$ | $0.3277$ | $0.1074$ | $0.0115$ | $0.0000143$ |
| $0.85$ | $0.4437$ | $0.1969$ | $0.0388$ | $0.000296$ |
| $1.15$ | $2.011$ | $4.046$ | $16.37$ | $1083.66$ |
| $1.2$ | $2.488$ | $6.192$ | $38.34$ | $9100.44$ |

看 $\gamma=0.85$ 這一列：走 $20$ 步剩 $0.0388$，也就是衰減了九成六以上。走 $50$ 步剩 $0.000296$，幾乎是 $0$。再看 $\gamma=1.15$：$20$ 步放大到 $16.37$ 倍，$50$ 步放大到 $1083.66$ 倍。乘數只差 $0.15$，結果卻差了上百萬倍。

Look at the row $\gamma=0.85$: after $20$ steps $0.0388$ is left, a decay of more than $96\%$. After $50$ steps $0.000296$ is left, almost $0$. Now look at $\gamma=1.15$: $20$ steps amplify the size $16.37$ times, $50$ steps $1083.66$ times. The factors differ by only $0.15$, yet the results differ by a factor of millions.

![半對數座標上的五條直線，橫軸是步數 k 從 0 到 50，縱軸是 gamma 的 k 次方。gamma 等於 1.2 與 1.15 的紅橘線往上，gamma 等於 1 的虛線水平，gamma 等於 0.85 與 0.8 的藍線往下。k 等於 20 處標出 1.2 的 20 次方等於 38.34、0.8 的 20 次方等於 0.0115（Five straight lines on a semi-log plot of gamma to the k against k from 0 to 50: red and orange lines for 1.2 and 1.15 rise, the dashed line for 1 is flat, blue lines for 0.85 and 0.8 fall; at k = 20 the values 38.34 and 0.0115 are marked）](../assets/figures/lec08-kp04-power-decay.png)

**圖說 `lec08-kp04-power-decay`**〔功能：比較〕：先看縱軸：它是對數刻度，每一格差 $100$ 倍，所以指數變化畫出來是直線。再看中間的黑色虛線 $\gamma=1$：它是分界線，上面的線都往上衝，下面的線都往下掉。最後看 $k=20$ 的兩個圓點：紅點 $38.34$ 和藍點 $0.0115$ 相差三千多倍，而兩個乘數 $1.2$ 和 $0.8$ 看起來只差一點。

**Caption `lec08-kp04-power-decay`** [Function: comparison]: First look at the vertical axis: it is logarithmic, each grid line is a factor of $100$, so exponential change becomes a straight line. Then look at the black dashed line $\gamma=1$ in the middle: it is the border; every line above it shoots up and every line below it falls. Finally look at the two dots at $k=20$: the red $38.34$ and the blue $0.0115$ differ by more than three thousand times, while the factors $1.2$ and $0.8$ look only slightly different.

梯度消失的後果是：遠方的步驟幾乎收不到梯度。KP03 的 $\mathbf{W}$ 梯度是各步貢獻相加，遠方那一份接近 $0$，加總裡只剩近處的貢獻。網路因此學不到「很久以前的輸入」和「現在的輸出」的關係。這正是 KP01 說的長距離依賴。

The consequence of a vanishing gradient: distant steps receive almost no gradient. The gradient of $\mathbf{W}$ in KP03 is a sum of contributions from all steps; the distant ones are close to $0$, so only nearby contributions remain in the sum. The network therefore cannot learn the relation between "an input long ago" and "the output now", which is the long-range dependency of KP01.

梯度爆炸的後果是：某一次更新的步伐大得離譜，參數一下子被推到很遠的地方。損失突然跳高，幾次更新後甚至變成 `NaN` 或 `Inf`。這時訓練就**發散**（diverge）了。

The consequence of an exploding gradient: one update takes an absurdly large step and pushes the parameters far away at once. The loss suddenly jumps, and after a few updates it may even become `NaN` or `Inf`; training then **diverges**.

常見錯誤：以為梯度消失是電腦的浮點數精度不夠造成的。其實這是連乘本身的數學性質；就算用無限精度計算，$0.8^{1000}$ 仍然非常接近 $0$。

Common error: thinking that a vanishing gradient is caused by the limited floating-point precision of computers. It is a mathematical property of the product itself; even with infinite precision, $0.8^{1000}$ is still extremely close to $0$.

### 3. 為什麼消失比爆炸常見（Why vanishing is more common than exploding）

上一節看到兩個方向都可能失控。實際訓練 RNN 時，哪一個比較常遇到？答案是梯度消失。原因藏在每一步的乘數 $\mathbf{W}^\top\operatorname{diag}(1-\mathbf{h}_s^2)$ 裡的第二個因子。

The previous section showed that both directions can get out of control. Which one appears more often when an RNN is trained? The answer is the vanishing gradient. The reason hides in the second factor of each step's multiplier $\mathbf{W}^\top\operatorname{diag}(1-\mathbf{h}_s^2)$.

$\tanh$ 的斜率 $\tanh'(i)=1-\tanh^2(i)$ 永遠落在 $(0,1]$。它只在 $i=0$ 時等於 $1$。程式算出：$i=1$ 時是 $0.4200$，$i=2$ 時只剩 $0.0707$。淨輸入的絕對值一大，$\tanh$ 就變得很平。這一段叫**飽和區**（saturation region），斜率接近 $0$。

The slope of $\tanh$, $\tanh'(i)=1-\tanh^2(i)$, always lies in $(0,1]$. It equals $1$ only at $i=0$. The program gives $0.4200$ at $i=1$ and only $0.0707$ at $i=2$. Once the net input is large in absolute value, $\tanh$ enters its flat **saturation region**, where the slope is close to $0$.

用純量 RNN 看得最清楚。取 $p=1$，每一步的乘數是 $\gamma_s=W(1-h_s^2)$。因為 $1-h_s^2\le1$，所以 $|\gamma_s|\le|W|$。也就是說，$\tanh$ 只會把乘數往下拉，不會往上推。

A scalar RNN shows this best. Take $p=1$; the factor of each step is $\gamma_s=W(1-h_s^2)$. Because $1-h_s^2\le1$, we get $|\gamma_s|\le|W|$. In other words, $\tanh$ can only pull the factor down, never push it up.

這帶來不對稱。要梯度消失，只要 $|W|<1$ 就一定成立；權重初始化通常偏小，很容易滿足。要梯度爆炸，不但要 $|W|>1$，狀態還要一直待在 $0$ 附近，不能飽和。兩個條件同時成立的機會小得多。

This creates an asymmetry. For the gradient to vanish, $|W|<1$ is enough; weights are usually initialized small, so this is easy to meet. For the gradient to explode, $|W|>1$ is needed and the state must also stay near $0$ without saturating. Both conditions together happen far less often.

實際跑一次 `[補充]`：讓純量 RNN 從 $h_0$ 出發，之後輸入全為 $0$，即 $h_s=\tanh(Wh_{s-1})$。程式逐步乘出 $\partial h_{20}/\partial h_0=\gamma_1\gamma_2\cdots\gamma_{20}$，結果如下。

A real run `[Added]`: start a scalar RNN from $h_0$ and then feed only zero inputs, so $h_s=\tanh(Wh_{s-1})$. The program multiplies out $\partial h_{20}/\partial h_0=\gamma_1\gamma_2\cdots\gamma_{20}$ step by step, with the results below.

| 設定 | 只看 $W$ 的 $W^{20}$ | 實際的 $\partial h_{20}/\partial h_0$ |
|---|---|---|
| $W=0.9$，$h_0=0.5$ | $0.1216$ | $0.0542$ |
| $W=1.5$，$h_0=0.5$ | $3325.26$ | $0.0000000545$ |
| $W=1.5$，$h_0=0.01$ | $3325.26$ | $0.0227$（第 $10$ 步最高到 $34.35$） |

| Setting | $W^{20}$ from $W$ alone | Actual $\partial h_{20}/\partial h_0$ |
|---|---|---|
| $W=0.9$, $h_0=0.5$ | $0.1216$ | $0.0542$ |
| $W=1.5$, $h_0=0.5$ | $3325.26$ | $0.0000000545$ |
| $W=1.5$, $h_0=0.01$ | $3325.26$ | $0.0227$ (peak $34.35$ at step $10$) |

第一列：$W=0.9$ 本來就會消失，$\tanh$ 的斜率又讓它消失得更快，實際值只有 $W^{20}$ 的一半不到。第二列最驚人：$W=1.5$ 照理該爆炸，但狀態很快飽和到 $h_{20}=0.8586$，每步乘數掉到約 $0.3943$，梯度反而消失。第三列從很小的 $h_0=0.01$ 出發，前幾步斜率接近 $1$，梯度先長到 $34.35$；等狀態長大、飽和，又掉回 $0.0227$。

Row one: $W=0.9$ vanishes anyway, and the $\tanh$ slopes make it vanish faster; the actual value is less than half of $W^{20}$. Row two is the most striking: $W=1.5$ "should" explode, but the state quickly saturates to $h_{20}=0.8586$, each factor drops to about $0.3943$, and the gradient vanishes instead. Row three starts from a tiny $h_0=0.01$; the first slopes are close to $1$, so the gradient first grows to $34.35$; once the state grows and saturates, it falls back to $0.0227$.

![左右兩張圖。左圖是 tanh 導數的鐘形曲線，最高點在 i 等於 0 時為 1.0000，i 等於 1 時 0.4200，i 等於 2 時 0.0707，上方有一條 1 的水平虛線。右圖是半對數座標上三條實線：W 等於 0.9 的藍線緩慢下降並低於點線 0.9 的 k 次方；W 等於 1.5、h_0 等於 0.5 的紅線一路掉到 10 的負 8 次方附近，遠離往上的點線 1.5 的 k 次方；W 等於 1.5、h_0 等於 0.01 的橘線先沿著點線上升到約 34，再往下掉（Left: the bell-shaped tanh derivative, 1 at i = 0, 0.42 at i = 1, 0.0707 at i = 2, under a dashed line at 1. Right: on a semi-log plot the actual chains of a scalar RNN; W = 0.9 falls below 0.9^k, W = 1.5 from h_0 = 0.5 falls to about 1e-8 while 1.5^k rises, and W = 1.5 from h_0 = 0.01 first follows 1.5^k up to about 34 and then falls）](../assets/figures/lec08-kp04-tanh-factor.png)

**圖說 `lec08-kp04-tanh-factor`**〔功能：機制〕：先看左圖：綠色曲線從來沒有超過上方的虛線 $1$，離開 $i=0$ 就很快往下掉。再看右圖的兩條點線：它們是只看 $W$ 時的預測 $0.9^k$ 和 $1.5^k$。最後比較實線與點線：每一條實線都在自己的點線下方，紅線甚至往相反方向走；這就是 $\tanh$ 斜率把連乘往下拉的結果。

**Caption `lec08-kp04-tanh-factor`** [Function: mechanism]: First look at the left panel: the green curve never rises above the dashed line at $1$ and falls quickly once it leaves $i=0$. Then look at the two dotted lines on the right: they are the predictions $0.9^k$ and $1.5^k$ from $W$ alone. Finally compare the solid lines with the dotted ones: every solid line lies below its dotted line, and the red one even goes the opposite way; this is the $\tanh$ slopes pulling the product down.

向量的情形也一樣 `[補充]`。只是「$W$ 的大小」要換成「$\mathbf{W}$ 最多能把向量拉長幾倍」。這個倍數小於 $1$ 時，梯度保證消失；大於 $1$ 只是爆炸的必要條件，不保證一定爆炸。

The vector case is the same, except that "the size of $W$" becomes "the largest factor by which the matrix $\mathbf{W}$ can stretch a vector" `[Added]`. When this factor is below $1$, the gradient is guaranteed to vanish; above $1$ is only a necessary condition for explosion, not a guarantee.

同樣的病也出現在很深的前饋網路：層數一多，連乘一樣讓前面幾層收不到梯度。第7週 KP06 的殘差連接（ResNet）就是為此設計的捷徑 `[補充]`。

The same disease appears in very deep feedforward networks: with many layers, the product again starves the first layers of gradient. The residual connections (ResNet) of Week 7 KP06 are shortcuts designed for this `[Added]`.

常見錯誤：以為 $\mathbf{W}$ 的元素大於 $1$ 就一定梯度爆炸。上表第二列說明，飽和的 $\tanh$ 會把大的 $W$ 也拉成消失。

Common error: thinking that a $\mathbf{W}$ with entries above $1$ always gives an exploding gradient. Row two of the table shows that a saturated $\tanh$ turns even a large $W$ into a vanishing gradient.

### 4. 梯度裁剪擋住爆炸，卻救不了消失（Gradient clipping stops explosion but cannot cure vanishing）

既然兩種病都會發生，各自怎麼處理？爆炸比較好處理。來源說，只要一招**梯度裁剪**（gradient clipping）：梯度太長時，等比例把它縮短到門檻 $c$。寫成式子 `[補充]`：

Since both diseases occur, how is each one handled? Explosion is the easier one. The source says one trick is enough, called **gradient clipping**. When the gradient is too long, shrink it proportionally to the threshold $c$. As an equation `[Added]`:

$$
\mathbf{g}\leftarrow
\begin{cases}
\mathbf{g}, & \|\mathbf{g}\|\le c,\\[4pt]
\dfrac{c}{\|\mathbf{g}\|}\,\mathbf{g}, & \|\mathbf{g}\|>c.
\end{cases}
$$

這裡 $\mathbf{g}$ 是所有參數的梯度排成的一個長向量，$\|\mathbf{g}\|=\sqrt{g_1^2+g_2^2+\cdots}$ 是它的長度。裁剪只改長度，不改方向，所以更新仍然往「讓損失下降」的方向走，只是步伐有上限。

Here $\mathbf{g}$ is the gradient of all parameters arranged as one long vector, and $\|\mathbf{g}\|=\sqrt{g_1^2+g_2^2+\cdots}$ is its length. Clipping changes only the length, not the direction, so the update still goes in the "loss decreases" direction, only with a capped step.

手算一次 `[補充]`：設 $\mathbf{g}=(6,8)$，門檻 $c=5$。長度 $\|\mathbf{g}\|=\sqrt{36+64}=10$，超過 $5$。縮放倍數是 $5/10=0.5$，裁剪後 $\mathbf{g}=(3,4)$，長度剛好是 $5$。另一個梯度 $(0.3,0.4)$ 的長度是 $0.5$，沒有超過門檻，保持不變。

By hand `[Added]`: let $\mathbf{g}=(6,8)$ and the threshold $c=5$. The length is $\|\mathbf{g}\|=\sqrt{36+64}=10$, above $5$. The scale is $5/10=0.5$, so after clipping $\mathbf{g}=(3,4)$, whose length is exactly $5$. Another gradient $(0.3,0.4)$ has length $0.5$, below the threshold, and stays unchanged.

![平面座標上一個半徑 5 的虛線圓。紅色長箭頭從原點指到 (6,8)，標長度 10；綠色粗箭頭沿同一方向只到圓周上的 (3,4)，標長度 5；原點附近一支藍色短箭頭 (0.3,0.4) 標 kept（A dashed circle of radius 5. A red arrow from the origin to (6,8) of length 10; a thick green arrow in the same direction ending on the circle at (3,4), length 5; a short blue arrow (0.3,0.4) near the origin marked kept）](../assets/figures/lec08-kp04-clipping.png)

**圖說 `lec08-kp04-clipping`**〔功能：一目了然〕：先看虛線圓：半徑是門檻 $c=5$，圓內的梯度不動。再看紅色箭頭 $(6,8)$：它伸出圓外，所以被截短。最後看綠色箭頭：它和紅色箭頭方向完全相同，只是停在圓周上的 $(3,4)$。

**Caption `lec08-kp04-clipping`** [Function: seeing it at a glance]: First look at the dashed circle: its radius is the threshold $c=5$, and gradients inside it are not touched. Then look at the red arrow $(6,8)$: it reaches outside the circle, so it is cut. Finally look at the green arrow: it points in exactly the same direction as the red one but stops on the circle at $(3,4)$.

為什麼同樣的招數救不了梯度消失？把太小的梯度放大，看起來像是對稱的解法。但消失的梯度裡，遠方那一份已經被乘到接近 $0$，近處的貢獻卻很大。整體放大只會把近處的貢獻一起放大，遠方的資訊仍然淹沒在裡面。來源的說法是：信號已經歸零，無法從 $0$ 裡把資訊還原。

Why can the same trick not cure a vanishing gradient? Scaling a too-small gradient up looks like the symmetric fix. But in a vanishing gradient, the distant part has already been multiplied down to nearly $0$, while the nearby contributions are large. Scaling the whole thing up scales the nearby parts too, and the distant information stays buried. In the source's words: the signal has become zero, and information cannot be recovered from $0$.

所以梯度消失要從網路結構下手：改變 $\mathbf{h}_{t-1}$ 到 $\mathbf{h}_t$ 的路，讓每個因子都接近 $1$；或乾脆開一條少乘幾次的捷徑。這正是 KP05 到 KP07 要做的事。

So the vanishing gradient must be attacked through the network structure: change the path from $\mathbf{h}_{t-1}$ to $\mathbf{h}_t$ so that every factor of the product is close to $1$, or open a shortcut with fewer multiplications. That is exactly what KP05 to KP07 do.

常見錯誤：以為裁剪會改變梯度的方向。裁剪只乘上一個正的數 $c/\|\mathbf{g}\|$，方向完全不變。

Common error: thinking that clipping changes the direction of the gradient. Clipping only multiplies by a positive number $c/\|\mathbf{g}\|$, so the direction stays exactly the same.

### 本單元重點整理（KP04 summary）

**一句話重點**：梯度從第 $T$ 步倒傳到第 $t$ 步，要連乘 $k=T-t$ 個雅可比矩陣 $\operatorname{diag}(1-\mathbf{h}_s^2)\mathbf{W}$；乘數小於 $1$ 時指數消失，大於 $1$ 時指數爆炸；因為 $\tanh'\le1$，消失比爆炸常見；爆炸可用梯度裁剪擋住，消失要改網路結構。

**One-sentence summary**: a gradient travelling from step $T$ back to step $t$ is multiplied by $k=T-t$ Jacobian matrices $\operatorname{diag}(1-\mathbf{h}_s^2)\mathbf{W}$; it vanishes exponentially when the factor is below $1$ and explodes exponentially when it is above $1$; because $\tanh'\le1$, vanishing is more common; explosion is stopped by gradient clipping, while vanishing needs a change of network structure.

**白話說明**：誤差要一站一站往回傳，每一站都打一次折扣或加一次價。站數一多，打折的會折到一毛不剩，加價的會漲到天價。$\tanh$ 這一關只會打折，所以「折到不剩」最常見。

**Plain explanation**: the error is passed back station by station, and every station applies a discount or a markup. With many stations, the discounted one shrinks to nothing and the marked-up one rises sky-high. The $\tanh$ station can only give discounts, so "shrinking to nothing" is the most common outcome.

**正式定義**：$\partial\mathbf{h}_T/\partial\mathbf{h}_t=\mathbf{J}_T\cdots\mathbf{J}_{t+1}$，$\mathbf{J}_s=\operatorname{diag}(1-\mathbf{h}_s^2)\mathbf{W}$；梯度向量乘的是 $\mathbf{J}_s^\top=\mathbf{W}^\top\operatorname{diag}(1-\mathbf{h}_s^2)$。若每步大小倍數為 $\gamma$，跨 $k$ 步為 $\gamma^k$。裁剪：$\|\mathbf{g}\|>c$ 時 $\mathbf{g}\leftarrow(c/\|\mathbf{g}\|)\mathbf{g}$。

**Formal definition**: $\partial\mathbf{h}_T/\partial\mathbf{h}_t=\mathbf{J}_T\cdots\mathbf{J}_{t+1}$, $\mathbf{J}_s=\operatorname{diag}(1-\mathbf{h}_s^2)\mathbf{W}$; the gradient vector is multiplied by $\mathbf{J}_s^\top=\mathbf{W}^\top\operatorname{diag}(1-\mathbf{h}_s^2)$. If the size factor per step is $\gamma$, it is $\gamma^k$ across $k$ steps. Clipping: if $\|\mathbf{g}\|>c$, then $\mathbf{g}\leftarrow(c/\|\mathbf{g}\|)\mathbf{g}$.

**例子**：$0.8^{20}=0.0115$，$1.2^{20}=38.34$；$0.85^{20}=0.0388$，$0.85^{50}=0.000296$；$1.15^{20}=16.37$，$1.15^{50}=1083.66$。純量 RNN 取 $W=1.5$、$h_0=0.5$ 時，$W^{20}=3325.26$，實際 $\partial h_{20}/\partial h_0=0.0000000545$。裁剪 $(6,8)$、$c=5$ 得 $(3,4)$。

**Example**: $0.8^{20}=0.0115$, $1.2^{20}=38.34$; $0.85^{20}=0.0388$, $0.85^{50}=0.000296$; $1.15^{20}=16.37$, $1.15^{50}=1083.66$. For a scalar RNN with $W=1.5$ and $h_0=0.5$, $W^{20}=3325.26$ but the actual $\partial h_{20}/\partial h_0=0.0000000545$. Clipping $(6,8)$ with $c=5$ gives $(3,4)$.

**常見迷思**：

- 「梯度消失是浮點數精度不夠。」錯，這是連乘的數學性質。
- 「$W$ 大於 $1$ 就一定爆炸。」錯，$\tanh$ 飽和會把乘數拉到 $1$ 以下。
- 「把小梯度放大就能治好消失。」錯，遠方的資訊已經被近處的貢獻淹沒。
- 「裁剪會改變梯度方向。」錯，只改長度。

**Common misconceptions**:

- "A vanishing gradient comes from limited floating-point precision." Wrong: it is a mathematical property of the product.
- "$W$ above $1$ always explodes." Wrong: a saturated $\tanh$ pulls the factor below $1$.
- "Scaling a small gradient up cures vanishing." Wrong: the distant information is already buried under nearby contributions.
- "Clipping changes the direction of the gradient." Wrong: it changes only the length.

**圖**：本單元有三張圖。

- `lec08-kp04-power-decay`：半對數座標上的 $\gamma^k$，以 $\gamma=1$ 為界一上一下。
- `lec08-kp04-tanh-factor`：$\tanh'\le1$，以及純量 RNN 的實際連乘都低於只看 $W$ 的預測。
- `lec08-kp04-clipping`：裁剪只把梯度截短到門檻，方向不變。

**Figures** — this unit has three figures.

- `lec08-kp04-power-decay`: $\gamma^k$ on a semi-log plot, rising or falling on either side of $\gamma=1$.
- `lec08-kp04-tanh-factor`: $\tanh'\le1$, and the actual chains of a scalar RNN all lie below the prediction from $W$ alone.
- `lec08-kp04-clipping`: clipping cuts the gradient to the threshold length without changing its direction.

### 小練習（Practice）

1. 純量 RNN 的 $W=0.5$，每一步的 $\tanh$ 斜率都是 $1-h_s^2=0.8$。每步乘數 $\gamma$ 是多少？倒傳 $3$ 步後剩多少？提示：$\gamma=W(1-h_s^2)$。
2. 乘數 $\gamma=0.95$ 看起來很接近 $1$。算出 $0.95^{20}$ 與 $0.95^{100}$，再算 $1.05^{20}$。這說明了什麼？提示：用計算機的次方鍵。
3. 梯度 $\mathbf{g}=(6,8)$。門檻 $c=5$ 時裁剪結果是什麼？門檻改成 $c=20$ 呢？提示：先算長度，再和 $c$ 比較。

1. A scalar RNN has $W=0.5$, and the $\tanh$ slope of every step is $1-h_s^2=0.8$. What is the factor $\gamma$ per step? How much is left after $3$ backward steps? Hint: $\gamma=W(1-h_s^2)$.
2. The factor $\gamma=0.95$ looks very close to $1$. Compute $0.95^{20}$ and $0.95^{100}$, then $1.05^{20}$. What does this show? Hint: use the power key of a calculator.
3. A gradient is $\mathbf{g}=(6,8)$. What does clipping give with the threshold $c=5$? And with $c=20$? Hint: compute the length first, then compare it with $c$.

**第 1 題解答** `[補充]`：$\gamma=0.5\times0.8=0.4$；倒傳 $3$ 步剩 $0.4^3=0.064$，也就是衰減了 $93.6\%$。

**Solution to question 1** `[Added]`: $\gamma=0.5\times0.8=0.4$; after $3$ backward steps $0.4^3=0.064$ is left, a decay of $93.6\%$.

**第 2 題解答** `[補充]`：$0.95^{20}=0.3585$，$0.95^{100}=0.00592$，$1.05^{20}=2.653$。乘數只要不是剛好 $1$，步數夠多就一定消失或爆炸；序列越長，問題越明顯。

**Solution to question 2** `[Added]`: $0.95^{20}=0.3585$, $0.95^{100}=0.00592$, $1.05^{20}=2.653$. Unless the factor is exactly $1$, enough steps always make it vanish or explode; the longer the sequence, the bigger the problem.

**第 3 題解答** `[補充]`：長度是 $10$。$c=5$ 時 $10>5$，乘上 $0.5$ 得 $(3,4)$。$c=20$ 時 $10\le20$，不裁剪，仍是 $(6,8)$。

**Solution to question 3** `[Added]`: the length is $10$. With $c=5$, $10>5$, so multiply by $0.5$ to get $(3,4)$. With $c=20$, $10\le20$, so nothing is clipped and it stays $(6,8)$.

### 小測驗（Quiz，來源原題）

來源的一題自我檢測保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The one self-check question of the source is kept with all options; the source page does not mark the answer directly, and the explanation below follows the source text.

1. 在長序列訓練中，為什麼梯度消失（Vanishing Gradient）比梯度爆炸更難用簡單技巧解決？
   - (A) 因為梯度爆炸可以用簡單的梯度裁剪（Clipping）截斷，但梯度消失意味著歷史訊息的梯度信號徹底歸零，無法重新補回
   - (B) 因為梯度消失會導致 CPU 燒毀
   - (C) 因為梯度消失無法用浮點數表示
   - (D) 因為梯度爆炸不會發生

1. In training on long sequences, why is a vanishing gradient harder to fix with a simple trick than an exploding gradient?
   - (A) Because an exploding gradient can be cut off by simple gradient clipping, while a vanishing gradient means the gradient signal of past information has gone to zero and cannot be put back
   - (B) Because a vanishing gradient burns the CPU
   - (C) Because a vanishing gradient cannot be represented with floating-point numbers
   - (D) Because an exploding gradient never happens

**解析** `[補充]`：選 (A)。第 4 節說明裁剪只要把過長的梯度截短；消失的梯度裡遠方的信號已接近 $0$，放大也救不回來。(B) 與硬體無關；(C) 錯，第 2 節說這是連乘的數學性質，不是精度問題；(D) 錯，第 2 節的 $\gamma=1.2$ 就會爆炸。

**Explanation** `[Added]`: the answer is A. Section 4 shows that clipping only shortens a gradient that is too long; in a vanishing gradient the distant signal is already near $0$ and scaling cannot bring it back. B has nothing to do with hardware; C is wrong, since Section 2 says this is a property of the product, not of precision; D is wrong, since $\gamma=1.2$ in Section 2 does explode.

### 留給下一單元的問題（The question for the next unit）

梯度裁剪擋得住爆炸，卻救不了消失。那能不能改變網路本身的結構？例如讓 $\mathbf{h}_{t-1}$ 到 $\mathbf{h}_t$ 的乘數接近 $1$，或開一條少乘幾次的捷徑，讓梯度活著傳回遠方。下一單元（KP05）介紹的單位矩陣、長延遲連線與滲漏單元回答這個問題。

Gradient clipping stops explosion but cannot cure vanishing. Can the structure of the network itself be changed so that the factor on the path from $\mathbf{h}_{t-1}$ to $\mathbf{h}_t$ is close to $1$, or so that a shortcut with fewer multiplications lets the gradient reach far back alive? The identity matrix, long-delay connections and leaky units of the next unit (KP05) answer this question.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第8章「循環神經網絡與長短期記憶網絡」§3（長距離依賴中的致命傷：梯度消失與爆炸）與其教授祕笈（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch08-rnn-lstm/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §3 of Chapter 8, "Recurrent neural networks and long short-term memory networks", of CCC-LNS IDL_115 (the fatal flaw in long-range dependencies: vanishing and exploding gradients) and its professor's notes (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch08-rnn-lstm/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源的傳話類比與互動展示寫 $0.8^{20}\approx0.0038$。程式算得 $0.8^{20}=0.0115$，本講義用 $0.0115$；結論「信號消失」不變。
- 來源寫 $1.2^{20}\approx38.33$、$1.15^{20}\approx16.36$、$0.85^{50}\approx0.000295$、$1.15^{50}\approx1083.65$。程式算得 $38.3376$、$16.3665$、$0.000295765$、$1083.657$，四捨五入後本講義寫 $38.34$、$16.37$、$0.000296$、$1083.66$。
- 來源把 $1.15^{50}\approx1083.65$ 稱為「數值溢出」。約 $1000$ 倍遠不到電腦浮點數的上限；真正的問題是更新步伐過大、訓練發散，之後才可能出現 `NaN`/`Inf`。本講義照此改寫。
- 來源把雅可比矩陣 $\partial\mathbf{h}_k/\partial\mathbf{h}_{k-1}$ 直接寫成 $\mathbf{W}^\top\operatorname{diag}(1-\mathbf{h}_k^2)$。依分子排列，雅可比矩陣是 $\operatorname{diag}(1-\mathbf{h}_k^2)\mathbf{W}$；$\mathbf{W}^\top\operatorname{diag}(1-\mathbf{h}_k^2)$ 是它的轉置，也就是反向傳播時乘在梯度向量上的那一個。兩者的放大倍數相同，結論不受影響；本講義兩個都寫出並說明關係。
- 來源說「$\mathbf{W}$ 的最大特徵值 $\lambda<1$ 就消失、$>1$ 就爆炸」。較精確的說法是看 $\mathbf{W}$ 最多能把向量拉長幾倍（最大奇異值）：小於 $1$ 保證消失，大於 $1$ 只是爆炸的必要條件。本講義第 3 節以 `[補充]` 寫出，並用純量 RNN 的實際算例說明 $W>1$ 也可能消失。
- 來源「舉一反三」一段把 ResNet 與 LSTM 的關係寫得不完整；本講義只保留「深層前饋網路也會梯度消失、殘差連接是捷徑」這一點。來源稱下一節為「KP-8」，在本課程中對應 KP05。
- 本講義補上：梯度裁剪的公式與 $(6,8)$ 算例（來源只有文字描述）、純量 RNN 的三組實際連乘、$\tanh'$ 在 $i=1,2$ 的數值、向量情形的放大倍數說明、三題小練習與解答。
- 取捨：來源的一題自我檢測保留原題；來源的 $\gamma$ 互動滑桿無法放進講義，改用 `lec08-kp04-power-decay` 圖與 $\gamma^k$ 表格呈現同樣的數字。

- The source's whisper analogy and interactive demo write $0.8^{20}\approx0.0038$. The program gives $0.8^{20}=0.0115$, which this note uses; the conclusion "the signal vanishes" is unchanged.
- The source writes $1.2^{20}\approx38.33$, $1.15^{20}\approx16.36$, $0.85^{50}\approx0.000295$ and $1.15^{50}\approx1083.65$. The program gives $38.3376$, $16.3665$, $0.000295765$ and $1083.657$, which round to $38.34$, $16.37$, $0.000296$ and $1083.66$ as written in this note.
- The source calls $1.15^{50}\approx1083.65$ a "numerical overflow". About $1000$ times is far from the floating-point limit; the real problem is an update step that is too large and training that diverges, after which `NaN`/`Inf` may appear. This note is written that way.
- The source writes the Jacobian $\partial\mathbf{h}_k/\partial\mathbf{h}_{k-1}$ directly as $\mathbf{W}^\top\operatorname{diag}(1-\mathbf{h}_k^2)$. In numerator layout the Jacobian is $\operatorname{diag}(1-\mathbf{h}_k^2)\mathbf{W}$; $\mathbf{W}^\top\operatorname{diag}(1-\mathbf{h}_k^2)$ is its transpose, the one that multiplies the gradient vector in backpropagation. Both stretch by the same amount, so the conclusion is unaffected; this note writes both and explains the relation.
- The source says "the gradient vanishes if the largest eigenvalue $\lambda$ of $\mathbf{W}$ is below $1$ and explodes if it is above $1$". More precisely, what matters is the largest factor by which $\mathbf{W}$ can stretch a vector (its largest singular value): below $1$ guarantees vanishing, above $1$ is only necessary for explosion. Section 3 states this with `[Added]` and uses a real scalar RNN to show that $W>1$ can also vanish.
- The source's "transfer" paragraph states the relation between ResNet and LSTM incompletely; this note keeps only the point that "deep feedforward networks also suffer vanishing gradients and residual connections are shortcuts". The source calls the next section "KP-8", which is KP05 in this course.
- This note adds: the clipping formula and the $(6,8)$ example (the source has only a verbal description), the three actual chains of a scalar RNN, the values of $\tanh'$ at $i=1,2$, the stretch-factor remark for the vector case, and three practice questions with solutions.
- Choices: the one self-check question of the source is kept as asked; the source's interactive $\gamma$ slider cannot go into a note, so the `lec08-kp04-power-decay` figure and the $\gamma^k$ table show the same numbers.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 長距離依賴 | long-range dependency |
| 隨時間反向傳播 | backpropagation through time (BPTT) |
| 連鎖律 | chain rule |
| 雅可比矩陣 | Jacobian matrix |
| 對角矩陣 | diagonal matrix |
| 指數衰減／指數成長 | exponential decay / exponential growth |
| 梯度消失 | vanishing gradient |
| 梯度爆炸 | exploding gradient |
| 發散 | diverge (divergence) |
| 飽和區 | saturation region |
| 奇異值 | singular value |
| 特徵值 | eigenvalue |
| 梯度裁剪 | gradient clipping |
| 門檻 | threshold |
| 殘差網路 | residual network (ResNet) |
| 循環神經網路 | recurrent neural network (RNN) |
| 長短期記憶網路（KP06） | long short-term memory (LSTM) |
| 非數值／無窮大 | not a number (NaN) / infinity (Inf) |
| 中央處理器（測驗選項） | central processing unit (CPU) |
| 本課程的網頁教材 | CCC-LNS |
