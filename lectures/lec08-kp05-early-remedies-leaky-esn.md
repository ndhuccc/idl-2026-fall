# 第8週 KP05：LSTM 之前的早期解方，替梯度開直通路（Early remedies before LSTM: opening direct paths for the gradient）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第8週「循環神經網路：RNN、LSTM 與 GRU」的第五個知識點（KP05）。KP04 找到梯度消失的病因：倒傳時，同一個係數被連乘很多次。這一節問：在 LSTM 出現之前，研究者怎麼改網路的結構，讓梯度不要一路縮到 $0$？

This is the fifth knowledge point (KP05) of Week 8, "Recurrent neural networks: RNN, LSTM and GRU". KP04 found the cause of the vanishing gradient: on the way back, the same factor is multiplied many times. This note asks: before LSTM existed, how did researchers change the structure of the network so that the gradient does not shrink all the way to $0$?

先備知識是 KP01 的 RNN 前向公式、KP04 的梯度連乘鏈，以及線性代數的特徵值與特徵向量。第一次出現的詞與符號都會定義。所有數字都由 `assets/figures/src/lec08-kp05-figures.py` 算出，可以自己重算。

The prerequisites are the RNN forward equations of KP01, the gradient product chain of KP04, and eigenvalues and eigenvectors from linear algebra. Every term and symbol is defined when it first appears. Every number is computed by `assets/figures/src/lec08-kp05-figures.py`, so you can recompute it yourself.

## KP05 LSTM 之前的早期解方，替梯度開直通路（Early remedies before LSTM: opening direct paths for the gradient）

### 學習目標（Learning objectives）

讀完之後，你能用特徵值說明連乘為什麼會消失或爆炸。你能寫出「加上單位矩陣」與「長延遲連線」兩種改法，並算出它們在 $15$ 步後留下多少梯度。你能寫出滲漏單元的更新式，說明時間常數 $\tau$ 大小的利弊。你也能說明迴聲狀態網路為什麼完全不用倒傳穿過時間。

After reading, you can use eigenvalues to explain why a repeated product vanishes or explodes. You can write the two changes "add the identity matrix" and "long-delay connections" and compute how much gradient each leaves after $15$ steps. You can write the update of a leaky unit and state the pros and cons of a large or small time constant $\tau$. You can also explain why an echo state network never has to backpropagate through time.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{h}_t\in\mathbb{R}^{p}$ | 第 $t$ 步的隱藏狀態 | hidden state at step $t$ |
| $\mathbf{W}\in\mathbb{R}^{p\times p}$ | 狀態轉移權重（同 KP01） | state transition weights (as in KP01) |
| $\lambda$ | $\mathbf{W}$ 的特徵值 | an eigenvalue of $\mathbf{W}$ |
| $\mathbf{A},\ \mathbf{\Lambda}$ | 特徵向量排成的矩陣、特徵值排成的對角矩陣 | matrix of eigenvectors, diagonal matrix of eigenvalues |
| $\mathbf{I}\in\mathbb{R}^{p\times p}$ | 單位矩陣，對角線是 $1$，其餘是 $0$ | identity matrix: ones on the diagonal, zeros elsewhere |
| $\mathbf{W}_1,\ \mathbf{W}_3\in\mathbb{R}^{p\times p}$ | 從 $1$ 步前、$3$ 步前的狀態連過來的權重 | weights from the state $1$ step and $3$ steps back |
| $k$ | 往回倒傳的步數 | number of steps back in time |
| $h_{t,j}$ | $\mathbf{h}_t$ 的第 $j$ 個分量 | component $j$ of $\mathbf{h}_t$ |
| $\tau_j\ge1$ | 第 $j$ 個分量的時間常數 | time constant of component $j$ |
| $\mathbf{W}_{j:},\ \mathbf{U}_{j:}$ | $\mathbf{W}$、$\mathbf{U}$ 的第 $j$ 列 | row $j$ of $\mathbf{W}$ and $\mathbf{U}$ |
| $\mathbf{W}_{\mathrm{res}}\in\mathbb{R}^{N\times N}$ | 迴聲狀態網路的儲水池權重，固定不訓練 | reservoir weights of an echo state network, fixed |
| $N$ | 儲水池的單元個數 | number of units in the reservoir |
| $\rho(\mathbf{W}_{\mathrm{res}})$ | 譜半徑：特徵值絕對值的最大者 | spectral radius: largest absolute value of the eigenvalues |
| $\mathbf{V},\ \mathbf{b}_y$ | 輸出層的權重與偏置（同 KP01） | output weights and bias (as in KP01) |

### 1. 從特徵值看連乘（The repeated product seen through eigenvalues）

KP04 的結論是：梯度從第 $t$ 步倒傳 $k$ 步，要經過 $k$ 個雅可比矩陣相乘。每一個都含有同一個 $\mathbf{W}$。所以要看懂消失與爆炸，就要看懂 $\mathbf{W}^k$。

The conclusion of KP04: a gradient travelling $k$ steps back from step $t$ passes through a product of $k$ Jacobian matrices. Each of them contains the same $\mathbf{W}$. So to understand vanishing and exploding, you need to understand $\mathbf{W}^k$.

線性代數給了一個好工具：**特徵值分解**（eigendecomposition）。如果 $\mathbf{W}$ 有 $p$ 個獨立的特徵向量，就能寫成

Linear algebra gives a good tool: the **eigendecomposition**. If $\mathbf{W}$ has $p$ independent eigenvectors, it can be written as

$$
\mathbf{W}=\mathbf{A}\mathbf{\Lambda}\mathbf{A}^{-1},\qquad \mathbf{W}^k=\mathbf{A}\mathbf{\Lambda}^k\mathbf{A}^{-1}.
$$

第二式的推導很短：相乘時中間的 $\mathbf{A}^{-1}\mathbf{A}$ 都變成 $\mathbf{I}$，一對一對消掉。剩下的 $\mathbf{\Lambda}^k$ 是對角矩陣，對角線上就是每個特徵值的 $k$ 次方 $\lambda^k$。若 $\mathbf{W}$ 是對稱矩陣，$\mathbf{A}^{-1}=\mathbf{A}^\top$，就是來源寫的 $\mathbf{A}\mathbf{\Lambda}\mathbf{A}^\top$。

The derivation of the second equation is short: in the product, every inner $\mathbf{A}^{-1}\mathbf{A}$ becomes $\mathbf{I}$ and cancels pair by pair. What remains, $\mathbf{\Lambda}^k$, is diagonal, with the $k$-th power $\lambda^k$ of each eigenvalue on the diagonal. If $\mathbf{W}$ is symmetric, $\mathbf{A}^{-1}=\mathbf{A}^\top$, which is the form $\mathbf{A}\mathbf{\Lambda}\mathbf{A}^\top$ written in the source.

所以每個特徵方向各自被乘上 $\lambda^k$。這就是三種命運：

So each eigen-direction is multiplied by its own $\lambda^k$. This gives three fates:

| 特徵值 | $k=15$ 時的 $\lambda^{15}$ | 結果 |
|---|---|---|
| $\lvert\lambda\rvert<1$，例如 $0.75$ | $0.0134$ | 梯度消失 |
| $\lvert\lambda\rvert=1$，例如 $1.0$ | $1.0$ | 不變 |
| $\lvert\lambda\rvert>1$，例如 $1.25$ | $28.4217$ | 梯度爆炸 |

| Eigenvalue | $\lambda^{15}$ at $k=15$ | Result |
|---|---|---|
| $\lvert\lambda\rvert<1$, e.g. $0.75$ | $0.0134$ | vanishing gradient |
| $\lvert\lambda\rvert=1$, e.g. $1.0$ | $1.0$ | unchanged |
| $\lvert\lambda\rvert>1$, e.g. $1.25$ | $28.4217$ | exploding gradient |

本節後面都沿用來源的主例子：每一步的梯度係數是 $0.75$，要倒傳 $15$ 步。標準 RNN 只剩 $0.75^{15}=0.0134$。早期的解方有兩條路：讓係數靠近 $1$，或讓要乘的次數變少。

The rest of this note keeps the source's running example: the gradient factor per step is $0.75$, and the gradient must travel $15$ steps back. A standard RNN keeps only $0.75^{15}=0.0134$. The early remedies take one of two roads: bring the factor close to $1$, or reduce the number of multiplications.

常見錯誤：以為梯度消失是電腦精度不夠造成的。不是。就算用無限精度計算，$0.75^{15}$ 還是 $0.0134$。這是連乘本身的數學性質。

Common error: thinking that the vanishing gradient is caused by limited computer precision. It is not. Even with infinite precision, $0.75^{15}$ is still $0.0134$. It is a mathematical property of the repeated product itself.

### 2. 加上單位矩陣：讓特徵值靠近 1（Adding the identity: moving eigenvalues close to 1）

第一條路是讓係數靠近 $1$。來源的改法是：把上一步的狀態「直接複製一份」加進來。寫成式子，就是在 $\mathbf{W}$ 旁邊加一個單位矩陣 $\mathbf{I}$：

The first road brings the factor close to $1$. The source's change is to "copy" the previous state directly into the new one. As an equation, an identity matrix $\mathbf{I}$ is added next to $\mathbf{W}$:

$$
\mathbf{h}_t=\tanh\big((\mathbf{W}+\mathbf{I})\mathbf{h}_{t-1}+\mathbf{U}\mathbf{x}_t+\mathbf{b}_i\big).
$$

為什麼這樣有用？若 $\mathbf{W}\mathbf{a}=\lambda\mathbf{a}$，則 $(\mathbf{W}+\mathbf{I})\mathbf{a}=\lambda\mathbf{a}+\mathbf{a}=(\lambda+1)\mathbf{a}$。所以 $\mathbf{W}+\mathbf{I}$ 的特徵向量不變，每個特徵值都加 $1$。只要 $\mathbf{W}$ 本身很小、特徵值都在 $0$ 附近，$\mathbf{W}+\mathbf{I}$ 的特徵值就都在 $1$ 附近。這種做法叫**趨近單位矩陣**（close-to-identity weight matrix）。

Why does this help? If $\mathbf{W}\mathbf{a}=\lambda\mathbf{a}$, then $(\mathbf{W}+\mathbf{I})\mathbf{a}=\lambda\mathbf{a}+\mathbf{a}=(\lambda+1)\mathbf{a}$. So $\mathbf{W}+\mathbf{I}$ has the same eigenvectors, and every eigenvalue goes up by $1$. As long as $\mathbf{W}$ itself is small, with all eigenvalues near $0$, the eigenvalues of $\mathbf{W}+\mathbf{I}$ are all near $1$. This approach is called a **close-to-identity weight matrix**.

二維例子 `[補充]`：取對稱矩陣

A two-dimensional example `[Added]`: take the symmetric matrix

$$
\mathbf{W}=\begin{bmatrix}-0.10&0.08\\0.08&-0.10\end{bmatrix}.
$$

它的特徵值是 $-0.02$ 和 $-0.18$，特徵向量分別是 $(1,1)$ 和 $(1,-1)$ 的方向。$\mathbf{W}$ 單獨連乘 $15$ 次，最大放大倍數只剩 $6.747\times10^{-12}$，等於完全消失。加上 $\mathbf{I}$ 之後，特徵值變成 $0.98$ 和 $0.82$；連乘 $15$ 次，兩個方向分別留下 $0.98^{15}=0.7386$ 和 $0.82^{15}=0.0510$。

Its eigenvalues are $-0.02$ and $-0.18$, with eigenvectors along $(1,1)$ and $(1,-1)$. Multiplying $\mathbf{W}$ by itself $15$ times leaves a largest gain of only $6.747\times10^{-12}$, which means it vanishes completely. After adding $\mathbf{I}$, the eigenvalues become $0.98$ and $0.82$; after $15$ multiplications the two directions keep $0.98^{15}=0.7386$ and $0.82^{15}=0.0510$.

類比 `[補充]`：$\mathbf{I}$ 像一條「原樣照抄」的直通管。舊狀態先沿著直通管整份送到下一步，$\mathbf{W}$ 只負責在上面做小修改。這個類比的侷限是：外面還包著 $\tanh$。它的斜率小於 $1$，梯度還是會打一點折扣，不是全部保留。

Analogy `[Added]`: $\mathbf{I}$ is like a "copy as is" pipe. The old state is sent whole to the next step along the pipe, and $\mathbf{W}$ only makes small edits on top. The limit of the analogy: $\tanh$ is still wrapped around it, and its slope is below $1$, so the gradient still gets some discount and is not kept one hundred percent.

使用條件很重要：$\mathbf{W}$ 必須小。如果 $\mathbf{W}$ 有一個特徵值是 $0.5$，加上 $\mathbf{I}$ 後變成 $1.5$，反而會爆炸。來源的「教授祕笈」也提到同一派的做法：**正交初始化**（orthogonal initialization）。正交矩陣的特徵值絕對值都剛好是 $1$，連乘時向量長度不變。

The condition matters: $\mathbf{W}$ must be small. If $\mathbf{W}$ has an eigenvalue of $0.5$, adding $\mathbf{I}$ turns it into $1.5$, which explodes instead. The source's professor's notes mention a method of the same family: **orthogonal initialization**. Every eigenvalue of an orthogonal matrix has absolute value exactly $1$, so repeated products keep the length of a vector unchanged.

常見錯誤：以為加上 $\mathbf{I}$ 會多出一倍參數。$\mathbf{I}$ 是固定的常數矩陣，不需要學。以 $p=4$ 為例，$\mathbf{W}$ 有 $16$ 個可學參數，$\mathbf{W}+\mathbf{I}$ 也還是 $16$ 個。

Common error: thinking that adding $\mathbf{I}$ doubles the parameters. $\mathbf{I}$ is a fixed constant matrix and is not learned. With $p=4$, $\mathbf{W}$ has $16$ learnable parameters, and $\mathbf{W}+\mathbf{I}$ still has $16$.

### 3. 長延遲連線：替梯度開捷徑（Long-delay connections: a shortcut for the gradient）

加上 $\mathbf{I}$ 是讓每一步的係數變大。第二條路則是讓要乘的次數變少。做法是讓狀態除了接收 $1$ 步前的 $\mathbf{h}_{t-1}$，也直接接收 $3$ 步前的 $\mathbf{h}_{t-3}$：

Adding $\mathbf{I}$ makes the factor of each step larger. The second road reduces the number of multiplications instead. The state receives not only $\mathbf{h}_{t-1}$ from $1$ step back but also $\mathbf{h}_{t-3}$ from $3$ steps back:

$$
\mathbf{h}_t=\tanh\big(\mathbf{W}_1\mathbf{h}_{t-1}+\mathbf{W}_3\mathbf{h}_{t-3}+\mathbf{U}\mathbf{x}_t+\mathbf{b}_i\big).
$$

這叫**長延遲連線**（long-delay connection），也叫**跳接**（skip connection）。倒傳時，梯度可以沿著 $\mathbf{W}_3$ 一次跳 $3$ 步。從第 $15$ 步回到第 $0$ 步，原本要經過 $15$ 次相乘；走跳接只要 $15/3=5$ 次。

This is called a **long-delay connection**, also a **skip connection**. On the way back, the gradient can jump $3$ steps at a time along $\mathbf{W}_3$. Going from step $15$ back to step $0$ used to take $15$ multiplications; along the skips it takes only $15/3=5$.

沿用主例子，並照來源假設每次跳躍的係數也是 $0.75$。標準 RNN 剩 $0.75^{15}=0.0134$，跳接剩 $0.75^{5}=0.2373$，約是 $17.76$ 倍。下圖把兩條路放在一起比較。

Keep the running example and, as the source does, assume each jump also has factor $0.75$: the standard RNN keeps $0.75^{15}=0.0134$, the skip path keeps $0.75^{5}=0.2373$, about $17.76$ times as much. The figure below compares both roads.

![對數縱軸的折線圖，橫軸是往回的步數 k 從 0 到 15。紅線是標準 RNN 的 0.75 的 k 次方，到 k 等於 15 時是 0.0134。藍線是 3 步長延遲，每 3 步一個點，到 15 時是 0.2373。綠線是趨近單位矩陣的 0.98 的 k 次方，到 15 時是 0.7386（Log-scale line chart over k = 0 to 15: standard RNN 0.75^k ends at 0.0134, the 3-step delay ends at 0.2373, the close-to-identity 0.98^k ends at 0.7386）](../assets/figures/lec08-kp05-retention.png)

**圖說 `lec08-kp05-retention`**〔功能：精準度〕：先看紅線：標準 RNN 每退一步乘 $0.75$，在對數座標上是一條往下的直線，$15$ 步後剩 $0.0134$。再看藍線：跳接每 $3$ 步才乘一次，斜率只有紅線的三分之一，最後是 $0.2373$。最後看綠線：係數 $0.98$ 幾乎平躺，留下 $0.7386$。縱軸是對數，所以紅、藍之間的距離代表倍數差。

**Caption `lec08-kp05-retention`** [Function: precision]: First look at the red line: the standard RNN multiplies by $0.75$ for every step back, a straight downward line on the log scale, leaving $0.0134$ after $15$ steps. Then look at the blue line: the skip path multiplies only once every $3$ steps, so its slope is one third of the red one, ending at $0.2373$. Finally look at the green line: the factor $0.98$ is almost flat and keeps $0.7386$. The vertical axis is logarithmic, so the gap between red and blue is a ratio.

和加上 $\mathbf{I}$ 不同，跳接需要新參數：$\mathbf{W}_3$ 和 $\mathbf{W}_1$ 一樣大，$p=4$ 時多出 $16$ 個。它還有一個假設：跳的距離是事先固定的。來源的「教授祕笈」問：能不能直接加一條 $100$ 步的連線？可以，但結構會變得很複雜，而且網路仍然不能自己決定什麼時候該跳、什麼時候該清掉。

Unlike adding $\mathbf{I}$, a skip needs new parameters: $\mathbf{W}_3$ is as large as $\mathbf{W}_1$, which adds $16$ for $p=4$. It also assumes the jump length is fixed in advance. The source's professor's notes ask: could we simply add a $100$-step connection? We could, but the structure becomes very complex, and the network still cannot decide by itself when to jump and when to clear.

常見錯誤：以為有了跳接，$\mathbf{W}_1$ 那條路就不用了。兩條路同時存在，梯度是兩條路的貢獻相加；跳接只是多開一條比較短的路。

Common error: thinking that with a skip, the $\mathbf{W}_1$ path is no longer used. Both paths exist at the same time, and the gradient is the sum of their contributions; the skip just opens an extra, shorter path.

### 4. 滲漏單元：用時間常數慢慢忘（Leaky units: forgetting slowly with a time constant）

跳接的距離是固定的整數，很僵硬。能不能讓每個狀態分量自己決定「舊的要留多少」？**滲漏單元**（leaky unit）就是這樣做。每個分量 $j$ 有一個**時間常數**（time constant）$\tau_j\ge1$：

The jump length of a skip is a fixed integer, which is rigid. Can each component of the state decide for itself "how much of the old to keep"? A **leaky unit** does exactly that. Each component $j$ has a **time constant** $\tau_j\ge1$:

$$
h_{t,j}=\Big(1-\frac{1}{\tau_j}\Big)h_{t-1,j}+\frac{1}{\tau_j}\tanh\big(\mathbf{W}_{j:}\mathbf{h}_{t-1}+\mathbf{U}_{j:}\mathbf{x}_t+b_{i,j}\big).
$$

右邊的 $\tanh(\cdots)$ 就是 KP01 的標準更新，我們叫它「候選值」。新狀態是舊狀態與候選值的加權平均，兩個權重 $1-\frac{1}{\tau_j}$ 和 $\frac{1}{\tau_j}$ 加起來剛好是 $1$。$\tau_j=1$ 時，舊狀態的權重是 $0$，就回到標準 RNN。$\tau_j$ 很大時，$h_{t,j}\approx h_{t-1,j}$，舊狀態幾乎原樣保留。

The $\tanh(\cdots)$ on the right is the standard update of KP01, which we call the "candidate". The new state is a weighted average of the old state and the candidate, and the two weights $1-\frac{1}{\tau_j}$ and $\frac{1}{\tau_j}$ add up to exactly $1$. When $\tau_j=1$, the old state has weight $0$, and we are back to the standard RNN. When $\tau_j$ is large, $h_{t,j}\approx h_{t-1,j}$, and the old state is kept almost as is.

來源的類比是有閥門的蓄水池：$\tau_j$ 越大，閥門開得越小，昨天的水流失得越慢。這個類比的侷限是：真的水只會流走，而滲漏單元每一步也會流進新的候選值。

The source's analogy is a reservoir with a valve: the larger $\tau_j$, the smaller the valve opening, and the more slowly yesterday's water drains away. The limit of the analogy: real water only drains, while a leaky unit also takes in a new candidate at every step.

來源的算例：令候選值一直是 $0$，初始 $h_0=1.0$，看記憶怎麼衰減。每一步都乘上 $1-\frac{1}{\tau}$。

The source's example: keep the candidate at $0$, start from $h_0=1.0$, and watch the memory decay. Every step multiplies by $1-\frac{1}{\tau}$.

| 時間常數 | 保留比例 $1-\frac{1}{\tau}$ | $h_1$ | $h_2$ | $h_{10}$ |
|---|---|---|---|---|
| $\tau=1$ | $0$ | $0$ | $0$ | $0$ |
| $\tau=2$ | $0.5$ | $0.5$ | $0.25$ | — |
| $\tau=10$ | $0.9$ | $0.9$ | $0.81$ | $0.3487$ |

| Time constant | Fraction kept $1-\frac{1}{\tau}$ | $h_1$ | $h_2$ | $h_{10}$ |
|---|---|---|---|---|
| $\tau=1$ | $0$ | $0$ | $0$ | $0$ |
| $\tau=2$ | $0.5$ | $0.5$ | $0.25$ | — |
| $\tau=10$ | $0.9$ | $0.9$ | $0.81$ | $0.3487$ |

記憶要多久才剩一半？解 $(1-\frac{1}{\tau})^n=0.5$ 得 $n=\ln0.5/\ln(1-\frac{1}{\tau})$。$\tau=10$ 時 $n=6.5788$ 步。來源寫的 $\tau\ln2=6.9315$ 是 $\tau$ 很大時的近似值。

How long until half the memory is left? Solving $(1-\frac{1}{\tau})^n=0.5$ gives $n=\ln0.5/\ln(1-\frac{1}{\tau})$. For $\tau=10$, $n=6.5788$ steps. The source's value $\tau\ln2=6.9315$ is an approximation that holds when $\tau$ is large.

為什麼這能幫梯度？`[補充]` 對 $h_{t-1,j}$ 微分，第一項直接給出 $1-\frac{1}{\tau_j}$，這條路不經過 $\tanh$，也不經過 $\mathbf{W}$。$\tau=10$ 時，光這條直通路每步就保留 $0.9$，倒傳 $15$ 步還剩 $0.9^{15}=0.2059$。

Why does this help the gradient? `[Added]` Differentiating with respect to $h_{t-1,j}$, the first term gives $1-\frac{1}{\tau_j}$ directly; this path passes through neither $\tanh$ nor $\mathbf{W}$. With $\tau=10$ this direct path alone keeps $0.9$ per step, and $0.9^{15}=0.2059$ is left after $15$ steps back.

但大 $\tau$ 有代價 `[補充]`：舊的忘得慢，新的也進來得慢。令候選值一直是 $1$、$h_0=0$。$\tau=1$ 時一步就到 $h_1=1$；$\tau=10$ 時 $h_1=0.1$、$h_2=0.19$，到第 $10$ 步也只有 $1-0.9^{10}=0.6513$。

But a large $\tau$ has a cost `[Added]`: it forgets the old slowly, and it also lets the new in slowly. Keep the candidate at $1$ with $h_0=0$. With $\tau=1$ the state reaches $h_1=1$ in one step; with $\tau=10$ it is $h_1=0.1$, $h_2=0.19$, and only $1-0.9^{10}=0.6513$ at step $10$.

![左右兩張折線圖，橫軸是時間步 t 從 0 到 20，四條線分別是 tau 等於 1、2、5、10。左圖從 h_0 等於 1 開始、候選值為 0：tau 等於 1 一步歸零，tau 等於 10 在第 10 步還有 0.3487，虛線標出一半的位置。右圖從 h_0 等於 0 開始、候選值為 1：tau 等於 1 一步到 1，tau 等於 10 在第 10 步只到 0.6513（Two line charts over t = 0 to 20 for tau = 1, 2, 5, 10: on the left the memory of h_0 = 1 decays, tau = 10 still has 0.3487 at step 10; on the right a constant candidate of 1 is followed, tau = 10 reaches only 0.6513 at step 10）](../assets/figures/lec08-kp05-leaky.png)

**圖說 `lec08-kp05-leaky`**〔功能：一目了然〕：先看左圖：這是「記住」的能力。紅線 $\tau=1$ 一步就掉到 $0$，綠線 $\tau=10$ 掉得最慢，第 $10$ 步還有 $0.3487$。再看右圖：這是「跟上新資訊」的能力。紅線一步就到 $1$，綠線爬得最慢。最後把兩張圖對照：同一個 $\tau$ 在左圖越好，在右圖就越差。

**Caption `lec08-kp05-leaky`** [Function: seeing it at a glance]: First look at the left panel: this is the ability to "remember". The red line $\tau=1$ drops to $0$ in one step, the green line $\tau=10$ drops most slowly and still has $0.3487$ at step $10$. Then look at the right panel: this is the ability to "follow new information". The red line reaches $1$ in one step, the green line climbs most slowly. Finally compare the two: the better a given $\tau$ does on the left, the worse it does on the right.

常見錯誤：以為 $\tau$ 越大越好。$\tau$ 大，記得久，但反應也慢；$\tau$ 小，反應快，但很快就忘。來源指出，不同分量可以用不同的 $\tau_j$，各自負責快與慢。可是每個 $\tau_j$ 一旦定下來，就不會隨著讀到的內容而改變。

Common error: thinking that a larger $\tau$ is always better. A large $\tau$ remembers long but reacts slowly; a small $\tau$ reacts fast but forgets quickly. The source notes that different components can use different $\tau_j$, some fast and some slow. But once each $\tau_j$ is set, it does not change with what is being read.

### 5. 迴聲狀態網路：循環層乾脆不訓練（Echo state networks: do not train the recurrent layer at all）

前面三種改法都在減輕倒傳穿過時間的連乘。**迴聲狀態網路**（echo state network，ESN）換一個角度。既然麻煩出在「訓練循環層」，那就不訓練它。這個想法叫**儲水池計算**（reservoir computing）。

The three changes so far all ease the repeated product of backpropagation through time. An **echo state network** (ESN) looks at it from another angle: since the trouble comes from "training the recurrent layer", just do not train it. This idea is called **reservoir computing**.

ESN 有三個部分。循環層叫**儲水池**（reservoir），它的權重 $\mathbf{W}_{\mathrm{res}}$ 隨機產生、稀疏連接，之後固定不動。輸入權重 $\mathbf{U}$ 也固定。只有最後的線性輸出層 $\mathbf{V}$ 要訓練：

An ESN has three parts. The recurrent layer is called the **reservoir**; its weights $\mathbf{W}_{\mathrm{res}}$ are random, sparsely connected, and then frozen. The input weights $\mathbf{U}$ are fixed too. Only the final linear output layer $\mathbf{V}$ is trained:

$$
\mathbf{h}_t=\tanh\big(\mathbf{W}_{\mathrm{res}}\mathbf{h}_{t-1}+\mathbf{U}\mathbf{x}_t\big),\qquad \mathbf{y}_t=\mathbf{V}\mathbf{h}_t+\mathbf{b}_y.
$$

因為 $\mathbf{y}_t$ 對 $\mathbf{V}$ 是線性的，求 $\mathbf{V}$ 就是一個線性迴歸，用最小平方法一次解出。整個訓練完全不需要倒傳穿過時間，也就沒有梯度消失。

Because $\mathbf{y}_t$ is linear in $\mathbf{V}$, finding $\mathbf{V}$ is a linear regression, solved in one go by least squares. Training never needs backpropagation through time, so there is no vanishing gradient.

隨機的儲水池要怎麼挑？`[補充]` 常見的規則是把 $\mathbf{W}_{\mathrm{res}}$ 縮放到**譜半徑**（spectral radius）$\rho(\mathbf{W}_{\mathrm{res}})<1$。由第 1 節可知，特徵值都小於 $1$ 時，舊狀態的影響會慢慢變小。所以儲水池的狀態主要由最近的輸入決定。舊輸入像回音一樣逐漸淡去，這就是「迴聲」的由來。

How should the random reservoir be chosen? `[Added]` A common rule is to scale $\mathbf{W}_{\mathrm{res}}$ so that its **spectral radius** satisfies $\rho(\mathbf{W}_{\mathrm{res}})<1$. By Section 1, when all eigenvalues are below $1$, the influence of old states shrinks over time. So the reservoir state is mainly set by recent inputs, and old inputs fade like an echo; that is where the name comes from.

一個小實驗 `[補充]`：取 $N=100$ 個單元、約 $10\%$ 的連線（$1001$ 個非零權重），縮放到 $\rho=0.9$。輸入是 $-1$ 到 $1$ 之間的隨機數。前 $200$ 步丟掉，接著 $1500$ 步訓練 $\mathbf{V}$，最後 $500$ 步測試。要記住的目標是 $k$ 步前的輸入 $y_t=x_{t-k}$。可訓練的參數只有 $100+1=101$ 個。

A small experiment `[Added]`: take $N=100$ units with about $10\%$ of the connections ($1001$ nonzero weights), scaled to $\rho=0.9$. The input is random between $-1$ and $1$. The first $200$ steps are dropped, the next $1500$ steps train $\mathbf{V}$, and the last $500$ steps test it. The target to remember is the input from $k$ steps back, $y_t=x_{t-k}$. Only $100+1=101$ parameters are trainable.

測試時用**決定係數**（coefficient of determination）$R^2$ 打分。$1$ 是完全對，$0$ 是跟猜平均值一樣。結果：$k=1$ 時 $R^2=0.9999$，$k=10$ 時 $0.9758$，$k=15$ 時 $0.7779$，$k=20$ 時只剩 $0.0547$，$k=30$ 時是 $-0.0654$。另外，從兩個不同的起始狀態餵同一串輸入。兩者的距離由第 $1$ 步的 $5.904$，縮到第 $60$ 步的 $0.0001421$，起點被忘掉了。

The test is scored by the **coefficient of determination** $R^2$: $1$ is perfect, $0$ is as good as guessing the mean. Results: $R^2=0.9999$ at $k=1$, $0.9758$ at $k=10$, $0.7779$ at $k=15$, only $0.0547$ at $k=20$, and $-0.0654$ at $k=30$. Also, feeding the same inputs from two different starting states, their distance shrinks from $5.904$ at step $1$ to $0.0001421$ at step $60$: the starting point is forgotten.

![三格圖。左格是 ESN 結構：x_t 經固定的 U 進入一團灰色、互相連線的儲水池，儲水池經一條粗橘色箭頭 V 到 y_t，標示只有 V 被訓練。中格是複數平面，100 個藍點是儲水池矩陣的特徵值，全部落在半徑 0.9 的橘色虛線圓內，外面還有一個單位圓。右格是測試 R 平方對延遲 k 的折線，k 等於 1 時 0.9999、10 時 0.9758、20 時 0.0547（Three panels: the ESN structure with a fixed grey reservoir and a thick orange trained readout V; the reservoir eigenvalues inside the circle of radius 0.9 in the complex plane; test R^2 against delay k, 0.9999 at k = 1, 0.9758 at k = 10, 0.0547 at k = 20）](../assets/figures/lec08-kp05-esn.png)

**圖說 `lec08-kp05-esn`**〔功能：具象化空間幾何〕：先看左格：灰色的儲水池和 $\mathbf{U}$ 都固定，只有粗橘色箭頭 $\mathbf{V}$ 會被訓練。再看中格：$100$ 個特徵值都在半徑 $0.9$ 的虛線圓內，沒有一個跑出單位圓，所以舊狀態會逐漸淡去。最後看右格：延遲 $k$ 越大，$R^2$ 越低，$k=20$ 時已經幾乎記不住，回音消失了。

**Caption `lec08-kp05-esn`** [Function: making space and geometry concrete]: First look at the left panel: the grey reservoir and $\mathbf{U}$ are fixed, and only the thick orange arrow $\mathbf{V}$ is trained. Then look at the middle panel: all $100$ eigenvalues lie inside the dashed circle of radius $0.9$, none outside the unit circle, so old states fade. Finally look at the right panel: the larger the delay $k$, the lower $R^2$; at $k=20$ the input is almost forgotten, and the echo is gone.

來源的「教授祕笈」也談到 ESN 的限制。儲水池的容量有限，固定的隨機權重不能為任務學出最好的特徵。所以遇到複雜的語言任務，它比不上整個網路都能訓練的模型。

The source's professor's notes also give the limit of an ESN: the reservoir has limited capacity, and fixed random weights cannot learn the best features for the task. So on complex language tasks it falls behind models whose whole network is trainable.

常見錯誤：以為 ESN 裡面沒有循環連線。儲水池裡有大量循環連線，只是它們在初始化之後就被鎖住，不再更新。

Common error: thinking that an ESN has no recurrent connections. The reservoir is full of recurrent connections; they are just locked after initialization and never updated.

### 本單元重點整理（KP05 summary）

**一句話重點**：LSTM 之前的四種解方都在對付連乘。加上 $\mathbf{I}$ 讓特徵值靠近 $1$，長延遲連線讓相乘次數變少，滲漏單元用固定的 $\tau_j$ 留下舊狀態，ESN 則乾脆不訓練循環層。

**One-sentence summary**: the four remedies before LSTM all fight the repeated product: adding $\mathbf{I}$ moves eigenvalues close to $1$, long-delay connections cut the number of multiplications, leaky units keep the old state with a fixed $\tau_j$, and an ESN does not train the recurrent layer at all.

**白話說明**：梯度像一句話被一個人一個人往回傳，每傳一次就變小聲。加 $\mathbf{I}$ 是讓每個人說話大聲一點；長延遲是跳過幾個人直接傳；滲漏單元是讓一部分的話原封不動往下傳；ESN 是根本不靠傳話來學，只調最後一個人。

**Plain explanation**: the gradient is like a sentence passed back person by person, getting quieter each time. Adding $\mathbf{I}$ makes each person speak a bit louder; a long delay skips a few people; a leaky unit passes part of the sentence on unchanged; an ESN does not learn by passing at all and only tunes the last person.

**正式定義**：$\mathbf{h}_t=\tanh((\mathbf{W}+\mathbf{I})\mathbf{h}_{t-1}+\mathbf{U}\mathbf{x}_t+\mathbf{b}_i)$；$\mathbf{h}_t=\tanh(\mathbf{W}_1\mathbf{h}_{t-1}+\mathbf{W}_3\mathbf{h}_{t-3}+\mathbf{U}\mathbf{x}_t+\mathbf{b}_i)$；$h_{t,j}=(1-\frac{1}{\tau_j})h_{t-1,j}+\frac{1}{\tau_j}\tanh(\mathbf{W}_{j:}\mathbf{h}_{t-1}+\mathbf{U}_{j:}\mathbf{x}_t+b_{i,j})$；ESN 固定 $\mathbf{W}_{\mathrm{res}}$ 與 $\mathbf{U}$，只用最小平方法求 $\mathbf{V}$。

**Formal definition**: $\mathbf{h}_t=\tanh((\mathbf{W}+\mathbf{I})\mathbf{h}_{t-1}+\mathbf{U}\mathbf{x}_t+\mathbf{b}_i)$; $\mathbf{h}_t=\tanh(\mathbf{W}_1\mathbf{h}_{t-1}+\mathbf{W}_3\mathbf{h}_{t-3}+\mathbf{U}\mathbf{x}_t+\mathbf{b}_i)$; $h_{t,j}=(1-\frac{1}{\tau_j})h_{t-1,j}+\frac{1}{\tau_j}\tanh(\mathbf{W}_{j:}\mathbf{h}_{t-1}+\mathbf{U}_{j:}\mathbf{x}_t+b_{i,j})$; an ESN fixes $\mathbf{W}_{\mathrm{res}}$ and $\mathbf{U}$ and finds only $\mathbf{V}$ by least squares.

**例子**：倒傳 $15$ 步：標準 RNN 剩 $0.75^{15}=0.0134$，$3$ 步跳接剩 $0.75^{5}=0.2373$，趨近單位矩陣剩 $0.98^{15}=0.7386$。滲漏單元 $\tau=10$：$h_{10}=0.3487$，半衰期 $6.5788$ 步。ESN（$N=100$、$\rho=0.9$）：$k=10$ 時 $R^2=0.9758$，$k=20$ 時 $0.0547$。

**Example**: $15$ steps back: the standard RNN keeps $0.75^{15}=0.0134$, the $3$-step skip $0.75^{5}=0.2373$, the close-to-identity matrix $0.98^{15}=0.7386$. Leaky unit with $\tau=10$: $h_{10}=0.3487$, half-life $6.5788$ steps. ESN ($N=100$, $\rho=0.9$): $R^2=0.9758$ at $k=10$, $0.0547$ at $k=20$.

**常見迷思**：

- 「加上 $\mathbf{I}$ 會多一倍參數。」錯，$\mathbf{I}$ 是常數，不用學。
- 「加上 $\mathbf{I}$ 一定有效。」錯，只有 $\mathbf{W}$ 本身很小時，特徵值才會落在 $1$ 附近。
- 「$\tau$ 越大越好。」錯，記得越久，對新輸入的反應也越慢。
- 「ESN 沒有循環連線。」錯，有很多，只是固定不訓練。

**Common misconceptions**:

- "Adding $\mathbf{I}$ doubles the parameters." Wrong: $\mathbf{I}$ is a constant and is not learned.
- "Adding $\mathbf{I}$ always works." Wrong: only when $\mathbf{W}$ itself is small do the eigenvalues land near $1$.
- "The larger $\tau$, the better." Wrong: the longer it remembers, the more slowly it reacts to new input.
- "An ESN has no recurrent connections." Wrong: it has many; they are just fixed and not trained.

**圖**：本單元有三張圖。

- `lec08-kp05-retention`：倒傳 $k$ 步後，標準 RNN、$3$ 步跳接與趨近單位矩陣各留下多少梯度。
- `lec08-kp05-leaky`：不同 $\tau$ 的滲漏單元，記住舊值與跟上新值的取捨。
- `lec08-kp05-esn`：ESN 的結構、儲水池特徵值與記憶隨延遲淡去。

**Figures** — this unit has three figures.

- `lec08-kp05-retention`: how much gradient the standard RNN, the $3$-step skip and the close-to-identity matrix keep after $k$ steps back.
- `lec08-kp05-leaky`: the trade-off between remembering the old value and following the new one for leaky units with different $\tau$.
- `lec08-kp05-esn`: the ESN structure, the reservoir eigenvalues, and memory fading with delay.

### 小練習（Practice）

1. 序列長度 $T=12$，每步梯度係數 $0.8$。標準 RNN 倒傳 $12$ 步留下多少？改用 $3$ 步跳接、每次跳躍係數也是 $0.8$，又留下多少？提示：跳接只要 $12/3$ 次。
2. 滲漏單元 $\tau=5$，$h_0=1.0$，候選值一直是 $0$。算出 $h_1$、$h_2$，並找出第幾步開始低於 $0.5$。提示：每步乘上 $1-\frac{1}{5}$。
3. 某個 $\mathbf{W}$ 有一個特徵值是 $0.5$。改用 $\mathbf{W}+\mathbf{I}$ 之後，這個方向的特徵值是多少？連乘 $10$ 次會怎樣？提示：用第 2 節的 $(\lambda+1)$。

1. Sequence length $T=12$, gradient factor $0.8$ per step. How much does a standard RNN keep after $12$ steps back? With a $3$-step skip whose jump factor is also $0.8$, how much is kept? Hint: the skip needs only $12/3$ jumps.
2. A leaky unit with $\tau=5$, $h_0=1.0$ and the candidate always $0$. Compute $h_1$ and $h_2$, and find at which step the state first falls below $0.5$. Hint: every step multiplies by $1-\frac{1}{5}$.
3. Some $\mathbf{W}$ has an eigenvalue $0.5$. After switching to $\mathbf{W}+\mathbf{I}$, what is the eigenvalue in this direction? What happens after $10$ multiplications? Hint: use $(\lambda+1)$ from Section 2.

**第 1 題解答** `[補充]`：標準 RNN 剩 $0.8^{12}=0.0687$；跳接只要 $4$ 次，剩 $0.8^{4}=0.4096$。

**Solution to question 1** `[Added]`: the standard RNN keeps $0.8^{12}=0.0687$; the skip needs only $4$ jumps and keeps $0.8^{4}=0.4096$.

**第 2 題解答** `[補充]`：保留比例 $0.8$，$h_1=0.8$，$h_2=0.64$，$h_3=0.512$，$h_4=0.4096$。所以第 $4$ 步開始低於 $0.5$；精確的半衰期是 $\ln0.5/\ln0.8=3.1063$ 步。

**Solution to question 2** `[Added]`: the fraction kept is $0.8$, so $h_1=0.8$, $h_2=0.64$, $h_3=0.512$, $h_4=0.4096$. The state first falls below $0.5$ at step $4$; the exact half-life is $\ln0.5/\ln0.8=3.1063$ steps.

**第 3 題解答** `[補充]`：特徵值變成 $0.5+1=1.5$，大於 $1$。連乘 $10$ 次得 $1.5^{10}=57.665$，梯度爆炸。這說明加上 $\mathbf{I}$ 只在 $\mathbf{W}$ 很小時才有用。

**Solution to question 3** `[Added]`: the eigenvalue becomes $0.5+1=1.5$, which is above $1$. Ten multiplications give $1.5^{10}=57.665$, an exploding gradient. This shows that adding $\mathbf{I}$ only helps when $\mathbf{W}$ is small.

### 小測驗（Quiz，來源原題）

來源的兩題自我檢測保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The two self-check questions of the source are kept with all options; the source page does not mark the answers directly, and the explanations below follow the source text.

1. 將 RNN 隱藏狀態更新修改為 $\mathbf{h}_t=\tanh((\mathbf{W}+\mathbf{I})\mathbf{h}_{t-1}+\mathbf{U}\mathbf{x}_t+\mathbf{b}_i)$ 的主要目的是什麼？
   - (A) 減少矩陣乘法運算量
   - (B) 引入單位矩陣 $\mathbf{I}$ 強化對角線，保持特徵值接近 $1$，防止梯度隨時間衰減歸零
   - (C) 將輸出限制在 $0$ 與 $1$ 之間
   - (D) 消除對輸入 $\mathbf{x}_t$ 的依賴
2. 迴聲狀態網絡（Echo State Network, ESN）如何徹底避免 BPTT 的梯度消失問題？
   - (A) 使用非常大的學習率
   - (B) 將循環儲水池（Reservoir）內部的權重固定不訓練，僅訓練最後的線性輸出層
   - (C) 將所有激活函數換成 ReLU
   - (D) 完全不用矩陣運算

1. What is the main purpose of changing the hidden-state update of an RNN to $\mathbf{h}_t=\tanh((\mathbf{W}+\mathbf{I})\mathbf{h}_{t-1}+\mathbf{U}\mathbf{x}_t+\mathbf{b}_i)$?
   - (A) Reducing the amount of matrix multiplication
   - (B) Adding the identity matrix $\mathbf{I}$ strengthens the diagonal, keeps the eigenvalues close to $1$, and prevents the gradient from decaying to zero over time
   - (C) Limiting the output to between $0$ and $1$
   - (D) Removing the dependence on the input $\mathbf{x}_t$
2. How does an echo state network (ESN) avoid the vanishing gradient of BPTT completely?
   - (A) Using a very large learning rate
   - (B) Keeping the weights inside the recurrent reservoir fixed and training only the final linear output layer
   - (C) Replacing every activation function with ReLU
   - (D) Not using matrix operations at all

**解析** `[補充]`：第 1 題選 (B)：第 2 節說明，加上 $\mathbf{I}$ 讓每個特徵值加 $1$。在 $\mathbf{W}$ 很小的前提下，特徵值就靠近 $1$。(A) 錯，計算量沒有變少；(C) 錯，$\tanh$ 的輸出在 $-1$ 與 $1$ 之間；(D) 錯，$\mathbf{U}\mathbf{x}_t$ 還在。第 2 題選 (B)：第 5 節說明，只訓練線性的 $\mathbf{V}$。它用最小平方法求解，不需要倒傳穿過時間；(A)(C)(D) 都沒有拿掉時間上的連乘。

**Explanations** `[Added]`: question 1 is B: Section 2 shows that adding $\mathbf{I}$ raises every eigenvalue by $1$; given that $\mathbf{W}$ is small, the eigenvalues end up near $1$. A is wrong, the computation is not reduced; C is wrong, the output of $\tanh$ lies between $-1$ and $1$; D is wrong, $\mathbf{U}\mathbf{x}_t$ is still there. Question 2 is B: Section 5 shows that only the linear $\mathbf{V}$ is trained, by least squares, with no backpropagation through time; A, C and D do not remove the product over time.

### 留給下一單元的問題（The question for the next unit）

滲漏單元的 $\tau_j$ 一旦定下來就不變。可是讀一句話時，有時該牢牢記住（例如主詞），有時該立刻清掉（例如讀到句號）。固定的 $\tau$ 自己決定不了什麼時候該記、什麼時候該忘。網路能不能學會依照讀到的內容，動態地打開和關上這道門？下一單元（KP06）的長短期記憶網路（LSTM）回答這個問題。

Once the $\tau_j$ of a leaky unit is set, it never changes. But when reading a sentence, sometimes you must hold on firmly (for example, to the subject) and sometimes clear at once (for example, at a full stop). A fixed $\tau$ cannot decide by itself when to remember and when to forget. Can the network learn to open and close this gate dynamically, according to what it reads? Long short-term memory (LSTM) in the next unit (KP06) answers this question.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第8章「循環神經網絡與長短期記憶網絡」§3.1–3.2「緩解梯度問題的早期結構創新：特徵值與長延遲」、§3.3–3.5「滲漏單元（Leaky Units）與迴聲狀態網絡（ESN）」與其教授祕笈（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch08-rnn-lstm/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §3.1–3.2, "Early structural remedies for the gradient problem: eigenvalues and long delays", and §3.3–3.5, "Leaky units and echo state networks (ESN)", of Chapter 8, "Recurrent neural networks and long short-term memory networks", of CCC-LNS IDL_115 and its professor's notes (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch08-rnn-lstm/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源網頁在「特徵值 $\lambda$ 的大小決定了狀態變化的命運」之後斷掉，只剩 $\lambda$。本講義依 KP04 的結論補回三種情形：$\lvert\lambda\rvert<1$ 消失、$=1$ 不變、$>1$ 爆炸。
- 來源寫 $\mathbf{W}=\mathbf{A}\mathbf{\Lambda}\mathbf{A}^\top$。這只對對稱矩陣成立；一般情形是 $\mathbf{A}\mathbf{\Lambda}\mathbf{A}^{-1}$。本講義寫一般式，並註明對稱時的特例。
- 來源說加上 $\mathbf{I}$「使特徵值靠近 $1$」。精確說法是每個特徵值都加 $1$，只有 $\mathbf{W}$ 的特徵值在 $0$ 附近時才會靠近 $1$。本講義補上這個條件與反例（第 3 題）。
- 來源互動展示寫「傳統 $0.8^{20}\approx0.003$」。程式算得 $0.8^{20}=0.0115$；另一個值 $0.9^{20}$ 是 $0.1216$（來源寫 $0.121$）。本講義不引用這兩個值，改用教授祕笈的 $T=15$ 主例子。
- 來源寫 $\tau=10$ 的半衰期是 $\tau\ln2\approx6.9$ 步。這是 $\tau$ 大時的近似；精確值是 $\ln0.5/\ln0.9=6.5788$ 步。
- 來源寫跳接「提升了 $17$ 倍」，程式算得 $0.2373/0.0134$ 的精確比值是 $17.76$。
- 來源用 `\boldsymbol`；本課程網站不支援，改用 `\mathbf`。來源的 $W_{res}$ 寫成 $\mathbf{W}_{\mathrm{res}}$。
- 本講義補上：二維 $\mathbf{W}+\mathbf{I}$ 的特徵值例子、參數個數比較。也補上滲漏單元的梯度直通路與「反應變慢」的代價。另補 ESN 的式子、譜半徑 $\rho<1$ 的規則、ESN 記憶實驗（隨機種子 $0$），以及三題小練習與解答。
- 取捨：來源的兩題自我檢測保留原題。來源的兩個互動滑桿無法放進講義，改用 `lec08-kp05-retention` 與 `lec08-kp05-leaky` 兩張圖呈現。來源提到滲漏單元可升級成 GRU 的更新門，留到 KP07 再講。

- The source web page breaks off after "the size of the eigenvalue $\lambda$ decides the fate of the state change", leaving only $\lambda$. This note restores the three cases from the conclusion of KP04: $\lvert\lambda\rvert<1$ vanishes, $=1$ is unchanged, $>1$ explodes.
- The source writes $\mathbf{W}=\mathbf{A}\mathbf{\Lambda}\mathbf{A}^\top$. That holds only for symmetric matrices; the general form is $\mathbf{A}\mathbf{\Lambda}\mathbf{A}^{-1}$. This note writes the general form and states the symmetric special case.
- The source says adding $\mathbf{I}$ "moves the eigenvalues close to $1$". Precisely, every eigenvalue goes up by $1$, and they end up near $1$ only if the eigenvalues of $\mathbf{W}$ are near $0$. This note adds the condition and a counterexample (question 3).
- The source's interactive demo writes "standard $0.8^{20}\approx0.003$". The program gives $0.8^{20}=0.0115$; the other value, $0.9^{20}$, is $0.1216$ (the source writes $0.121$). This note does not quote these two values and uses the $T=15$ running example of the professor's notes instead.
- The source gives the half-life for $\tau=10$ as $\tau\ln2\approx6.9$ steps. That is an approximation for large $\tau$; the exact value is $\ln0.5/\ln0.9=6.5788$ steps.
- The source says the skip "improves by $17$ times"; the program gives the exact ratio $0.2373/0.0134$ as $17.76$.
- The source uses `\boldsymbol`; the course website does not support it, so this note uses `\mathbf`. The source's $W_{res}$ is written $\mathbf{W}_{\mathrm{res}}$.
- This note adds: the two-dimensional $\mathbf{W}+\mathbf{I}$ eigenvalue example, the parameter-count comparison, the direct gradient path of a leaky unit and its "slower reaction" cost, the ESN equations, the spectral-radius rule $\rho<1$, the ESN memory experiment (random seed $0$), and three practice questions with solutions.
- Choices: the two self-check questions of the source are kept as asked; the source's two interactive sliders cannot go into a note, so the `lec08-kp05-retention` and `lec08-kp05-leaky` figures show them instead. The source's remark that a leaky unit can be upgraded into the GRU update gate is left for KP07.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 特徵值分解 | eigendecomposition |
| 特徵值、特徵向量 | eigenvalue, eigenvector |
| 單位矩陣 | identity matrix |
| 趨近單位矩陣 | close-to-identity weight matrix |
| 正交初始化 | orthogonal initialization |
| 長延遲連線 | long-delay connection |
| 跳接 | skip connection |
| 滲漏單元 | leaky unit |
| 時間常數 | time constant |
| 迴聲狀態網路 | echo state network (ESN) |
| 儲水池計算 | reservoir computing |
| 儲水池 | reservoir |
| 譜半徑 | spectral radius |
| 最小平方法 | least squares |
| 決定係數 | coefficient of determination ($R^2$) |
| 梯度消失、梯度爆炸 | vanishing gradient, exploding gradient |
| 雅可比矩陣 | Jacobian matrix |
| 隨時間反向傳播 | backpropagation through time (BPTT) |
| 循環神經網路 | recurrent neural network (RNN) |
| 長短期記憶網路（KP06） | long short-term memory (LSTM) |
| 門控循環單元（KP07） | gated recurrent unit (GRU) |
| 線性整流函數 | rectified linear unit (ReLU) |
| 本課程的網頁教材 | CCC-LNS |
