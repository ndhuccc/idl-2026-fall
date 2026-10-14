# 第2週 KP09：RBF 用固定的局部基函數，SOM 不用標籤就能自己分群（RBF uses fixed local basis functions; SOM groups data without labels）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)
>
> 週次：第2週（2026-09-17）｜主題：神經網路基礎：感知器與多層感知器
>
> Week 2 (2026-09-17) | Topic: Neural network fundamentals: perceptron and multilayer perceptron

本講義是第2章的第九個知識點（KP09），承接 KP08 留下的問題：如果第一層不學，固定成一組局部的基函數，只學第二層會怎樣？這一節介紹兩個「概念重要、但不在深度學習主線」的模型：RBF 網路（有標籤，固定基函數加最小二乘）與 SOM（沒有標籤，自組織成簇）。先備知識是 KP05（層）、KP08（最小二乘）與向量距離。內容依第2章主文重新講解，來源讀取日期為 2026-10-14；新增的釐清標為 `[補充]`。

This is the ninth knowledge point (KP09) of Chapter 2. It picks up the question KP08 left open: what if the first layer is not learned but fixed to a set of local basis functions, and only the second layer is learned? This note introduces two models that are "conceptually important but not on the main line of deep learning": the RBF network (with labels, fixed basis functions plus least squares) and the SOM (without labels, self-organizing clusters). It assumes KP05 (layers), KP08 (least squares) and vector distance. It re-explains the main text of Chapter 2, accessed on 2026-10-14; clarifications that are new are marked `[Added]`.

[下載本單元投影片](../slides/lec02-kp09-rbf-and-som.jbd.json)

## KP09 RBF 用固定的局部基函數，SOM 不用標籤就能自己分群（RBF uses fixed local basis functions; SOM groups data without labels）

**建議時間 / Time**：15 分鐘　**對應實作 / Related practice**：用兩個高斯基函數手算 XOR 的 RBF 解；手算 SOM 一次勝者與鄰域更新。

### 學習目標（Learning objectives）

讀完之後，你能說出 RBF 網路兩層各自的角色（第一層固定、第二層可訓練）；能用高斯基函數算出 XOR 四個點在基函數空間的座標，並說明為什麼在那裡一條直線就夠；能寫出最小二乘解 $\mathbf{W}=(\Phi\Phi^\top)^{-1}\Phi\mathbf{Y}^\top$ 與奇異時的偽逆；能手算 SOM 的勝者與一次鄰域更新，並說明 SOM 與 RBF 的差別。

After reading, you can say the roles of the two layers of an RBF network (the first fixed, the second trainable); compute the coordinates of the four XOR points in the basis-function space with Gaussian basis functions and explain why a line suffices there; write the least-squares solution $\mathbf{W}=(\Phi\Phi^\top)^{-1}\Phi\mathbf{Y}^\top$ and its pseudo-inverse form when singular; do one winner and neighbourhood update of a SOM by hand; and state how SOM differs from RBF.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\phi_i(\mathbf{x})$ | 第 $i$ 個基函數（共 $m$ 個） | the $i$-th basis function ($m$ in all) |
| $\boldsymbol\mu_i,\ \sigma_i^2$ | 高斯基函數的中心與變異數 | centre and variance of a Gaussian basis function |
| $\Phi\in\mathbb{R}^{m\times n}$ | 每欄是一筆資料對 $m$ 個基函數的值 | each column holds the values of the $m$ basis functions for one sample |
| $\mathbf{W}\in\mathbb{R}^{m\times k}$ | 第二層權重 | the second-layer weights |
| $\mathbf{Y}$ | 目標矩陣 | the target matrix |
| $w_{ij}$ | SOM 中輸入第 $i$ 維到神經元 $j$ 的權重 | the weight from input dimension $i$ to neuron $j$ of the SOM |
| $z$ | SOM 的勝者神經元 | the winning neuron of the SOM |
| $\mathcal{N}_z^{(\tau)}$ | 第 $\tau$ 步勝者的鄰域 | the neighbourhood of the winner at step $\tau$ |
| $\eta^{(\tau)}$ | 第 $\tau$ 步的學習率 | the learning rate at step $\tau$ |

### 1. RBF 網路：第一層固定，第二層用最小二乘（RBF network: fixed first layer, least squares for the second）

RBF（徑向基函數）網路是 Broomhead 與 Lowe 在 1988 年提出的兩層網路。第一層把輸入 $\mathbf{x}$ 送進 $m$ 個固定的基函數，最常用的是高斯

$$
\phi_i(\mathbf{x})=\exp\!\Big(-\frac{\|\mathbf{x}-\boldsymbol\mu_i\|^2}{2\sigma_i^2}\Big)
$$

它在中心 $\boldsymbol\mu_i$ 最大（值為 1），離中心越遠越接近 0，是「局部」的。第一層**不訓練**：先對訓練資料做 K-means 取 $m$ 個群心當 $\boldsymbol\mu_i$，再由各群的變異數決定 $\sigma_i$。第二層是可訓練的權重，輸出是基函數的加權和 $\hat y_j=\sum_{i=1}^{m}w_{ij}\phi_i(\mathbf{x})$。

The RBF (radial basis function) network is a two-layer network proposed by Broomhead and Lowe in 1988. The first layer sends the input $\mathbf{x}$ into $m$ fixed basis functions, most commonly Gaussians as above. A Gaussian is largest at its centre $\boldsymbol\mu_i$ (value 1) and approaches 0 farther away; it is "local". The first layer is **not trained**: run K-means on the training data to take $m$ cluster centres as the $\boldsymbol\mu_i$, then set $\sigma_i$ from the variance of each cluster. The second layer holds the trainable weights, and the output is the weighted sum of the basis functions $\hat y_j=\sum_{i=1}^{m}w_{ij}\phi_i(\mathbf{x})$.

因為第一層固定，剩下的問題對 $\mathbf{W}$ 是線性的，用最小二乘一次解出：把 $n$ 筆資料寫成 $\hat{\mathbf{Y}}=\mathbf{W}^\top\Phi$，最小化 $\|\mathbf{Y}-\mathbf{W}^\top\Phi\|_F^2$，令導數為零得 $\Phi\Phi^\top\mathbf{W}=\Phi\mathbf{Y}^\top$，所以

$$
\mathbf{W}=(\Phi\Phi^\top)^{-1}\Phi\mathbf{Y}^\top,\qquad\text{若 }\Phi\Phi^\top\text{ 奇異：}\ \mathbf{W}=(\Phi\Phi^\top)^{\dagger}\Phi\mathbf{Y}^\top
$$

Because the first layer is fixed, the remaining problem is linear in $\mathbf{W}$ and is solved at once by least squares: write the $n$ samples as $\hat{\mathbf{Y}}=\mathbf{W}^\top\Phi$, minimize $\|\mathbf{Y}-\mathbf{W}^\top\Phi\|_F^2$, and setting the derivative to zero gives $\Phi\Phi^\top\mathbf{W}=\Phi\mathbf{Y}^\top$, hence the solution above, with the pseudo-inverse $\dagger$ when $\Phi\Phi^\top$ is singular.

**用 XOR 手算〔程式計算〕。** 取兩個高斯，中心 $\boldsymbol\mu_1=(0,0)$、$\boldsymbol\mu_2=(1,1)$，$2\sigma^2=1$，所以 $\phi_i=e^{-\|\mathbf{x}-\boldsymbol\mu_i\|^2}$。四個點在基函數空間的座標 $(\phi_1,\phi_2)$：

| $(x_1,x_2)$ | $y$ | $\phi_1$ | $\phi_2$ |
|---|---|---|---|
| (0,0) | 0 | $1$ | $e^{-2}=0.135$ |
| (0,1) | 1 | $e^{-1}=0.368$ | $0.368$ |
| (1,0) | 1 | $0.368$ | $0.368$ |
| (1,1) | 0 | $0.135$ | $1$ |

$(0,1)$ 與 $(1,0)$ 落在同一點 $(0.368,0.368)$，$(0,0)$ 與 $(1,1)$ 在兩側。加上一個偏置項後做最小二乘，得到 $\mathbf{W}=(-2.503,\,-2.503)$、偏置 $2.841$，四個輸出恰好是 $0,1,1,0$。分界線是輸出 $=0.5$，即 $\phi_1+\phi_2=0.936$：類別 0 的和是 $1.135$，類別 1 的和是 $0.736$。

**A hand calculation on XOR (computed by program).** Take two Gaussians with centres $\boldsymbol\mu_1=(0,0)$ and $\boldsymbol\mu_2=(1,1)$ and $2\sigma^2=1$, so $\phi_i=e^{-\|\mathbf{x}-\boldsymbol\mu_i\|^2}$. The coordinates $(\phi_1,\phi_2)$ of the four points in the basis-function space are in the table above. $(0,1)$ and $(1,0)$ land on the same point $(0.368,0.368)$, with $(0,0)$ and $(1,1)$ on either side. Adding a bias term and doing least squares gives $\mathbf{W}=(-2.503,\,-2.503)$ with bias $2.841$, and the four outputs are exactly $0,1,1,0$. The boundary is output $=0.5$, i.e. $\phi_1+\phi_2=0.936$: the sum for class 0 is $1.135$ and for class 1 is $0.736$.

![XOR 在輸入空間與在兩個高斯基函數空間](../assets/figures/lec02-kp09-rbf-xor.png)

**圖說 `lec02-kp09-rbf-xor`**：先看左圖，兩個叉是高斯中心，虛線圈是它們的影響範圍，XOR 的四點不能用一條直線分開；再看右圖，換成基函數座標後，兩個圓點重疊在 $(0.368,0.368)$，綠色直線就把兩類分開。 / **Caption `lec02-kp09-rbf-xor`:** Look first at the left panel: the two crosses are the Gaussian centres and the dashed circles their reach; the four XOR points cannot be separated by a line. Then the right panel: in basis-function coordinates the two circles coincide at $(0.368,0.368)$, and the green line separates the classes.

這和 KP05 的隱藏層是同一個想法：先把點搬到新座標，再用一條直線分。差別是 KP05 的第一層權重要找，RBF 的第一層是事先固定的。

This is the same idea as the hidden layer of KP05: move the points to new coordinates first, then separate with a line. The difference is that the first-layer weights of KP05 must be found, whereas the first layer of the RBF is fixed beforehand.

### 2. SOM：不用標籤，讓神經元自己排成地圖（SOM: no labels, let the neurons arrange themselves into a map）

SOM（自組織映射，Kohonen 1982）是無監督的單層網路：輸入 $\mathbf{x}\in\mathbb{R}^d$ 接到 $k$ 顆神經元，每顆代表一個簇的原型（權重向量）。神經元排成預設的 1D 或 2D 格子，每顆有「鄰域」。訓練重複五步：

1. 初始化：權重取小的隨機值，鄰域設為格子的一半，學習率 $\eta^{(0)}\in(0,1]$。
2. 選一筆輸入，找勝者：$z=\arg\min_j\|\mathbf{x}-\mathbf{w}_j\|$（距離最近的神經元）。
3. 更新勝者**鄰域內**的神經元：$w_{ij}\leftarrow w_{ij}+\eta^{(\tau)}(x_i-w_{ij})$，鄰域外不動。
4. 學習率衰減 $\eta^{(\tau+1)}=\eta^{(0)}(1-\tau/t)$，鄰域逐漸縮小。
5. 回到第 2 步，重複。

測試時，勝者神經元就是這筆輸入所屬的簇。

The SOM (self-organizing map, Kohonen 1982) is an unsupervised single-layer network: the input $\mathbf{x}\in\mathbb{R}^d$ connects to $k$ neurons, each representing the prototype (weight vector) of a cluster. The neurons sit on a preset 1D or 2D grid, and each has a "neighbourhood". Training repeats five steps: (1) initialize with small random weights, a neighbourhood of half the grid and a learning rate $\eta^{(0)}\in(0,1]$; (2) pick an input and find the winner $z=\arg\min_j\|\mathbf{x}-\mathbf{w}_j\|$, the nearest neuron; (3) update the neurons **inside the winner's neighbourhood** by $w_{ij}\leftarrow w_{ij}+\eta^{(\tau)}(x_i-w_{ij})$, leaving the others alone; (4) decay the learning rate by $\eta^{(\tau+1)}=\eta^{(0)}(1-\tau/t)$ and let the neighbourhood shrink gradually; (5) go back to step 2. At test time the winning neuron is the cluster of the input.

**手算一步〔程式計算〕。** 三顆神經元排成一列，權重 $\mathbf{w}_1=(0.2,0.2)$、$\mathbf{w}_2=(0.5,0.5)$、$\mathbf{w}_3=(0.8,0.8)$。輸入 $\mathbf{x}=(0.9,0.7)$：

- 距離（歐氏）：$0.860,\ 0.447,\ 0.141$，勝者是神經元 3；
- 取 $\eta=0.5$，鄰域是勝者與它左邊一格（神經元 2、3）：
  - $\mathbf{w}_3=(0.8,0.8)+0.5\,(0.1,-0.1)=(0.85,0.75)$，
  - $\mathbf{w}_2=(0.5,0.5)+0.5\,(0.4,0.2)=(0.7,0.6)$，
  - $\mathbf{w}_1$ 在鄰域外，不變。

**One step by hand (computed by program).** Three neurons sit in a row with weights $\mathbf{w}_1=(0.2,0.2)$, $\mathbf{w}_2=(0.5,0.5)$, $\mathbf{w}_3=(0.8,0.8)$. For the input $\mathbf{x}=(0.9,0.7)$: the Euclidean distances are $0.860,\ 0.447,\ 0.141$, so neuron 3 wins; with $\eta=0.5$ and the neighbourhood being the winner and the one to its left (neurons 2 and 3), $\mathbf{w}_3=(0.8,0.8)+0.5\,(0.1,-0.1)=(0.85,0.75)$ and $\mathbf{w}_2=(0.5,0.5)+0.5\,(0.4,0.2)=(0.7,0.6)$, while $\mathbf{w}_1$ lies outside the neighbourhood and is unchanged.

重點是：不只勝者，連它的鄰居也被拉向輸入，所以格子上相鄰的神經元最後代表相近的資料，這叫拓撲保序。

The point is that not only the winner but also its neighbours are pulled toward the input, so neurons that are adjacent on the grid end up representing similar data; this is topology preservation.

![8×8 的 SOM 格子：訓練前擠在中央，訓練後攤開鋪滿單位正方形](../assets/figures/lec02-kp09-som-grid.png)

**圖說 `lec02-kp09-som-grid`**：先看左圖，64 個神經元的權重起初擠在正方形中央，連線亂成一團；再看右圖，從單位正方形隨機抽 4000 筆輸入訓練後，格子攤開鋪滿正方形，相鄰神經元仍是相鄰的。 / **Caption `lec02-kp09-som-grid`:** Look first at the left panel: the weights of the 64 neurons start crowded in the middle of the square, the links in a tangle; then at the right panel: after training on 4000 random inputs from the unit square, the grid has spread over the square and neighbouring neurons are still neighbours.

### 本單元重點整理（KP09 summary）

**一句話重點**：RBF 固定第一層的局部基函數、只用最小二乘解第二層（有標籤）；SOM 不用標籤，勝者與它的鄰居一起被拉向輸入，自己排成保持鄰近關係的地圖。

**One-sentence summary**: RBF fixes the local basis functions of the first layer and solves only the second layer by least squares (with labels); SOM uses no labels, and the winner and its neighbours are pulled toward the input together, arranging themselves into a map that preserves neighbourhood.

**白話說明**：RBF 像先在地圖上插好幾面旗子，每面旗子只管附近；一個點離哪面旗子近，就受那面旗子的影響，最後只要決定每面旗子的投票權重。SOM 像一張橡皮網，每丟一個資料點，網上最近的結點把自己和鄰居拉過去，多丟幾次，網就貼合了資料的形狀。

**Plain explanation**: RBF is like planting several flags on a map, each in charge of its surroundings; a point is influenced by the flags near it, and in the end only the voting weight of each flag must be decided. SOM is like a rubber net: whenever a data point is thrown on it, the nearest knot pulls itself and its neighbours toward it, and after many throws the net fits the shape of the data.

**正式定義**：RBF $\hat y_j=\sum_iw_{ij}\phi_i(\mathbf{x})$，$\mathbf{W}=(\Phi\Phi^\top)^{-1}\Phi\mathbf{Y}^\top$；SOM 勝者 $z=\arg\min_j\|\mathbf{x}-\mathbf{w}_j\|$，更新 $w_{ij}\leftarrow w_{ij}+\eta(x_i-w_{ij})$（$j$ 在鄰域內）。

**Formal definition**: RBF $\hat y_j=\sum_iw_{ij}\phi_i(\mathbf{x})$ with $\mathbf{W}=(\Phi\Phi^\top)^{-1}\Phi\mathbf{Y}^\top$; SOM winner $z=\arg\min_j\|\mathbf{x}-\mathbf{w}_j\|$, with the update $w_{ij}\leftarrow w_{ij}+\eta(x_i-w_{ij})$ for $j$ in the neighbourhood.

**例子**：XOR 用中心 $(0,0)$、$(1,1)$ 的兩個高斯，$(0,1)$ 與 $(1,0)$ 重疊在 $(0.368,0.368)$，$\mathbf{W}=(-2.503,-2.503)$、偏置 $2.841$；SOM 輸入 $(0.9,0.7)$ 勝者是神經元 3，$\mathbf{w}_3\to(0.85,0.75)$、$\mathbf{w}_2\to(0.7,0.6)$。

**Example**: For XOR, with two Gaussians centred at $(0,0)$ and $(1,1)$, $(0,1)$ and $(1,0)$ coincide at $(0.368,0.368)$ with $\mathbf{W}=(-2.503,-2.503)$ and bias $2.841$; for the SOM input $(0.9,0.7)$ neuron 3 wins and $\mathbf{w}_3\to(0.85,0.75)$, $\mathbf{w}_2\to(0.7,0.6)$.

**常見迷思**：

- 「RBF 網路的第一層也要用反向傳播訓練」——不用。第一層由 K-means 事先決定，只有第二層用最小二乘解。
- 「SOM 的輸出是類別預測」——不是。SOM 沒有標籤也沒有輸出層，勝者只告訴你這筆資料屬於哪一簇。

**Common misconceptions**:

- "The first layer of an RBF network is also trained by backpropagation." No. The first layer is decided beforehand by K-means, and only the second layer is solved by least squares.
- "The output of a SOM is a class prediction." No. A SOM has neither labels nor an output layer; the winner only tells which cluster the sample belongs to.

**圖**：本單元兩張圖，`lec02-kp09-rbf-xor`（輸入空間與基函數空間，功能 1、2）、`lec02-kp09-som-grid`（SOM 格子訓練前後，功能 1），以 `assets/figures/src/lec02-kp0810-figures.py` 繪製。

**Figures**: Two figures in this unit, `lec02-kp09-rbf-xor` (input space and basis-function space; functions 1, 2) and `lec02-kp09-som-grid` (the SOM grid before and after training; function 1), drawn by `assets/figures/src/lec02-kp0810-figures.py`.

**想深入**（選讀）：高斯基函數的寬度 $\sigma$ 決定一面旗子管多遠：太窄，基函數之間有空隙，預測在空隙處接近 0；太寬，基函數幾乎重疊，$\Phi\Phi^\top$ 接近奇異，這就是偽逆派上用場的情形。

**Going deeper** (optional): The width $\sigma$ of a Gaussian decides how far a flag reaches: too narrow and there are gaps between basis functions, where the prediction goes toward 0; too wide and the basis functions nearly coincide, making $\Phi\Phi^\top$ nearly singular, which is the case where the pseudo-inverse is useful.

**小練習**：

1. 兩個高斯中心 $(0,0)$、$(1,1)$，$2\sigma^2=1$，算點 $(0.5,0.5)$ 的 $(\phi_1,\phi_2)$。（提示：先算到兩中心的平方距離。）
2. SOM 兩個神經元 $\mathbf{w}_1=(0,0)$、$\mathbf{w}_2=(1,1)$，輸入 $(0.8,0.6)$，誰是勝者？若只更新勝者、$\eta=0.5$，新權重是多少？（提示：算歐氏距離。）
3. 為什麼 RBF 的第二層可以一次解出，而 MLP 的所有層不行？（提示：看對 $\mathbf{W}$ 是不是線性。）

**Practice**:

1. With two Gaussian centres $(0,0)$ and $(1,1)$ and $2\sigma^2=1$, compute $(\phi_1,\phi_2)$ for the point $(0.5,0.5)$. (Hint: compute the squared distances to the two centres first.)
2. A SOM has two neurons $\mathbf{w}_1=(0,0)$ and $\mathbf{w}_2=(1,1)$ and the input $(0.8,0.6)$. Which is the winner? If only the winner is updated with $\eta=0.5$, what is its new weight? (Hint: compute the Euclidean distances.)
3. Why can the second layer of the RBF be solved at once while all layers of an MLP cannot? (Hint: is the problem linear in $\mathbf{W}$?)

**解題示範**（第 1 題）：

1. 到 $(0,0)$ 的平方距離是 $0.5^2+0.5^2=0.5$，$\phi_1=e^{-0.5}=0.607$。
2. 到 $(1,1)$ 的平方距離同樣是 $0.5$，$\phi_2=e^{-0.5}=0.607$。
3. 所以 $(\phi_1,\phi_2)=(0.607,\,0.607)$：它在兩個中心的正中間，兩面旗子影響一樣大。
4. 檢查：$\phi_1+\phi_2=1.213$，比上面 XOR 的分界 $0.936$ 大，所以這一點會被判到類別 0 那一側。

**Worked solution** (exercise 1):

1. The squared distance to $(0,0)$ is $0.5^2+0.5^2=0.5$, so $\phi_1=e^{-0.5}=0.607$.
2. The squared distance to $(1,1)$ is likewise $0.5$, so $\phi_2=e^{-0.5}=0.607$.
3. So $(\phi_1,\phi_2)=(0.607,\,0.607)$: it lies exactly midway between the two centres, and both flags influence it equally.
4. Check: $\phi_1+\phi_2=1.213$, larger than the XOR boundary $0.936$, so this point falls on the class-0 side.

**自我檢核**（來源四題，不附答案）：

| 題 / Question | A | B | C | D | 提示 / Hint |
|---|---|---|---|---|---|
| 1. RBF 第一層基函數的參數 $\boldsymbol\mu_i,\sigma_i$ 怎麼決定？ / How are $\boldsymbol\mu_i,\sigma_i$ of the first layer decided? | 由反向傳播訓練 | 先對訓練資料跑 K-means 取群心與變異數 | 隨機初始化後凍結 | 用牛頓法迭代 | 第一層不訓練。 / The first layer is not trained. |
| 2. 若 $\Phi\Phi^\top$ 奇異，權重解應改用？ / If $\Phi\Phi^\top$ is singular, what should the solution use? | $(\Phi\Phi^\top)^{-1}$ | 偽逆 $(\Phi\Phi^\top)^{\dagger}\Phi\mathbf{Y}^\top$ | $\Phi^{-1}$ | $\mathbf{W}=\mathbf{Y}$ | 看第 1 節。 / Section 1. |
| 3. SOM 勝者神經元 $z$ 的定義？ / How is the winner $z$ of a SOM defined? | 距離總和最大者 | 距離最小者 | $\|\mathbf{w}_j\|$ 最小者 | 隨機選取 | 看第 2 節的第 2 步。 / Step 2 of section 2. |
| 4. SOM 權重更新的對象？ / Which neurons are updated in a SOM? | 所有神經元 | 只有勝者本身 | 勝者鄰域內的神經元 | 輸出層以外的神經元 | 看第 3 步。 / Step 3. |

**留給下一個知識點的問題**：到現在我們看了許多模型：M-P、Perceptron、ADALINE、MLP、RBF、SOM。它們共通的是「疊起來的神經元」。一個問題自然出現：只用一層隱藏層、夠多神經元，到底能表示多少種函數？這就是萬有逼近定理。

**Question for the next knowledge point**: We have now seen many models: M-P, Perceptron, ADALINE, MLP, RBF and SOM. What they share is "stacked neurons". A question arises naturally: with only one hidden layer and enough neurons, how many kinds of functions can be represented? That is the universal approximation theorem.

### 來源更正與取捨（Corrections to the source）

- 來源說 SOM 的鄰域「每步對半縮小」$\mathcal{N}^{(\tau+1)}=\mathcal{N}^{(\tau)}/2$。每步對半的話，鄰域在幾步內就縮到零，之後只剩勝者本身；實務上鄰域是**逐漸**縮小的（本講義的圖用高斯鄰域，半徑由約 3.4 慢慢降到 0.4）。
- 來源的勝者式寫成 $\sum_i\|x_i-w_{ij}\|_2$（對每一維取絕對值再加總，等於 L1 距離）。多數教科書用歐氏距離；本講義的手算用歐氏距離，兩種距離在這個例子選出同一個勝者（L1：$1.2,\ 0.6,\ 0.2$）。
- 來源的 RBF 輸出式沒有偏置。上面的 XOR 例子必須加偏置才能得到 $0,1,1,0$（不加偏置時，要讓 $(0,0)$ 與 $(1,1)$ 的輸出為 0，只能 $\mathbf{W}=\mathbf{0}$，$(0,1)$ 就得不到 1），所以本講義加了一個偏置項，等價於多一個恆為 1 的基函數（`[補充]`）。
- 來源沒有給 XOR 的 RBF 例子；上面的中心 $(0,0)$、$(1,1)$ 是手選的。XOR 四點是對稱的，K-means 在這四點上沒有唯一的群心，所以這個例子不是 K-means 的結果（`[補充]`）。
- 來源把 RBF 的第二種基函數寫成 logistic 形式（式 60）；常用的是高斯，本講義只用高斯。

- The source says the SOM neighbourhood is "halved at each step", $\mathcal{N}^{(\tau+1)}=\mathcal{N}^{(\tau)}/2$. Halving at every step would shrink the neighbourhood to zero within a few steps, leaving only the winner; in practice the neighbourhood shrinks **gradually** (the figure of this note uses a Gaussian neighbourhood whose radius slowly falls from about 3.4 to 0.4).
- The source writes the winner as $\sum_i\|x_i-w_{ij}\|_2$ (the absolute value per dimension summed, i.e. the L1 distance). Most textbooks use the Euclidean distance; the hand calculation of this note uses it, and both distances pick the same winner in this example (L1: $1.2,\ 0.6,\ 0.2$).
- The RBF output of the source has no bias. The XOR example above needs a bias to give $0,1,1,0$ (without a bias, making the outputs of $(0,0)$ and $(1,1)$ zero forces $\mathbf{W}=\mathbf{0}$, and then $(0,1)$ cannot give 1), so this note adds a bias term, equivalent to one more basis function that is always 1 (`[Added]`).
- The source gives no RBF example for XOR; the centres $(0,0)$, $(1,1)$ above are hand-picked. The four XOR points are symmetric, so K-means has no unique centres on them, and the example is not a K-means result (`[Added]`).
- The source writes a second kind of RBF basis function in logistic form (equation 60); the Gaussian is the common one, and this note uses only the Gaussian.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 徑向基函數 | radial basis function (RBF) |
| 基函數 | basis function |
| 自組織映射 | self-organizing map (SOM) |
| 勝者 | winner |
| 鄰域 | neighbourhood |
| 偽逆 | pseudo-inverse |
| 拓撲保序 | topology preservation |
| 無監督學習 | unsupervised learning |
