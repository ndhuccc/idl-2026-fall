# 第2週 KP05：把神經元接成層，非線性才讓多層有力量（Stacking neurons into layers: nonlinearity is what makes depth matter）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)
>
> 週次：第2週（2026-09-17）｜主題：神經網路基礎：感知器與多層感知器
>
> Week 2 (2026-09-17) | Topic: Neural network fundamentals: perceptron and multilayer perceptron

本講義是第2章的第五個知識點（KP05），承接 KP04 留下的問題：KP01 已經知道 AND、OR、NOT 都能由一個神經元做出來，而 XOR 可以寫成「至少一個為 1，而且不是兩個都為 1」，能不能把神經元接起來？答案是多層感知機（multilayer perceptron, MLP）。這一節說明它的結構，手算一個能算 XOR 的 2-2-1 網路，並證明：若每層都是線性活化，再多層也坍縮成一層。先備知識是 KP01–KP04 與矩陣乘法。內容依第2章主文重新講解，來源讀取日期為 2026-10-14；新增的釐清標為 `[補充]`。

This is the fifth knowledge point (KP05) of Chapter 2. It picks up the question KP04 left open: KP01 showed that AND, OR and NOT can each be made by one neuron, and XOR can be written as "at least one is 1, and not both are 1"; could we connect neurons? The answer is the multilayer perceptron (MLP). This note describes its structure, computes by hand a 2-2-1 network that computes XOR, and proves that if every layer has a linear activation, any number of layers collapses into one. It assumes KP01–KP04 and matrix multiplication. It re-explains the main text of Chapter 2, accessed on 2026-10-14; clarifications that are new are marked `[Added]`.

[下載本單元投影片](../slides/lec02-kp05-mlp-and-nonlinearity.jbd.json)

## KP05 把神經元接成層，非線性才讓多層有力量（Stacking neurons into layers: nonlinearity is what makes depth matter）

**建議時間 / Time**：15 分鐘　**對應實作 / Related practice**：手算 2-2-1 網路的 XOR，並驗證全線性網路的輸出恆為常數。

### 學習目標（Learning objectives）

讀完之後，你能說出輸入層、隱藏層、輸出層、全連接與前饋的意思；能寫出每顆神經元的兩步 $a_i=\sum_\ell w_{i\ell}z_\ell$、$z_i=\sigma(a_i)$；能手算 2-2-1 網路對四個輸入的輸出；能說明為什麼全線性活化的多層網路等於一層。

After reading, you can say what input, hidden and output layers, fully connected and feedforward mean; write the two steps of each neuron, $a_i=\sum_\ell w_{i\ell}z_\ell$ and $z_i=\sigma(a_i)$; compute by hand the output of a 2-2-1 network for the four inputs; and explain why a multilayer network with only linear activations equals one layer.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $z_\ell$ | 上一層第 $\ell$ 顆神經元的輸出（第一層就是輸入） | output of neuron $\ell$ in the previous layer (the input for the first layer) |
| $w_{i\ell}$ | 從神經元 $\ell$ 連到神經元 $i$ 的權重（偏置併入） | weight from neuron $\ell$ to neuron $i$ (bias folded in) |
| $a_i=\sum_\ell w_{i\ell}z_\ell$ | 活化前加權和（線性的一步） | pre-activation weighted sum (the linear step) |
| $\sigma$ | 活化函數，通常非線性 | activation function, usually nonlinear |
| $z_i=\sigma(a_i)$ | 神經元 $i$ 的輸出 | output of neuron $i$ |
| $\mathbf{W}_k$ | 第 $k$ 層的權重矩陣 | weight matrix of layer $k$ |

### 1. 結構：層、全連接、前饋（Structure: layers, fully connected, feedforward）

**輸入層**（input layer）接收原始資料；**隱藏層**（hidden layer）是中間的層，看不到答案也看不到原始資料；**輸出層**（output layer）給出結果。前一層的每一顆神經元都連到下一層的每一顆，叫**全連接**（fully connected）或**密集層**（dense layer）；資訊只向前傳、不回頭，叫**前饋**（feedforward）。每顆神經元做兩步：先線性加總 $a_i=\sum_\ell w_{i\ell}z_\ell$，再非線性活化 $z_i=\sigma(a_i)$；整層一起寫是 $\mathbf{z}^{(k)}=\sigma(\mathbf{W}_k^\top\mathbf{z}^{(k-1)})$。這就是 KP01 的神經元，只是輸入來自上一層，而且 $\sigma$ 不一定是階梯。文獻數「層」的方式不同：有的把輸入層也算進去，所以「四層網路」可能是「輸入加兩層隱藏加輸出」。

The **input layer** receives the raw data; **hidden layers** are the layers in between, which see neither the answer nor the raw data; the **output layer** gives the result. Every neuron of one layer connects to every neuron of the next, which is **fully connected** or a **dense layer**; information only moves forward, never back, which is **feedforward**. Each neuron does two steps: first the linear sum $a_i=\sum_\ell w_{i\ell}z_\ell$, then the nonlinear activation $z_i=\sigma(a_i)$; written for a whole layer, $\mathbf{z}^{(k)}=\sigma(\mathbf{W}_k^\top\mathbf{z}^{(k-1)})$. This is the neuron of KP01, except that its inputs come from the previous layer and $\sigma$ need not be the step. Papers count "layers" differently: some include the input layer, so a "four-layer network" may mean input, two hidden layers and output.

### 2. 手算：用三個閘接成 XOR（By hand: building XOR from three gates）

XOR 是「至少一個為 1（OR）而且不是兩個都為 1（NAND）」。取階梯 $f(s)=1$ 當 $s\ge0$，隱藏層第一顆是 OR：$s_1=x_1+x_2-1$；第二顆是 NAND：$s_2=-x_1-x_2+1.5$；輸出層是 AND：$s_3=h_1+h_2-2$。

XOR is "at least one is 1 (OR) and not both are 1 (NAND)". With the step $f(s)=1$ when $s\ge0$, the first hidden neuron is OR, $s_1=x_1+x_2-1$; the second is NAND, $s_2=-x_1-x_2+1.5$; the output neuron is AND, $s_3=h_1+h_2-2$.

| $(x_1,x_2)$ | $s_1$ | $h_1$ | $s_2$ | $h_2$ | $s_3=h_1+h_2-2$ | $o$ |
|---|---|---|---|---|---|---|
| (0,0) | $-1$ | 0 | $1.5$ | 1 | $-1$ | 0 |
| (0,1) | $0$ | 1 | $0.5$ | 1 | $0$ | 1 |
| (1,0) | $0$ | 1 | $0.5$ | 1 | $0$ | 1 |
| (1,1) | $1$ | 1 | $-0.5$ | 0 | $-1$ | 0 |

輸出 $0,1,1,0$ 正是 XOR〔程式計算，每格可手算驗證〕。第一層把四個點變成新座標 $(h_1,h_2)$：原本在對角線上的 $(0,1)$ 與 $(1,0)$ 都變成 $(1,1)$，$(0,0)\to(0,1)$、$(1,1)\to(1,0)$。在新座標裡，輸出為 1 的點只剩 $(1,1)$，一個 AND 就切開。

The outputs $0,1,1,0$ are exactly XOR (computed by program; every cell can be checked by hand). The first layer turns the four points into new coordinates $(h_1,h_2)$: the diagonal pair $(0,1)$ and $(1,0)$ both become $(1,1)$, with $(0,0)\to(0,1)$ and $(1,1)\to(1,0)$. In the new coordinates only $(1,1)$ has output 1, so one AND cuts it off.

![XOR 網路：隱藏層為 OR 與 NAND，輸出層為 AND](../assets/figures/lec02-kp05-xor-network.png)

**圖說 `lec02-kp05-xor-network`**：先看左邊兩個輸入圓；再看中間兩個隱藏神經元，連線上的數字是權重（$-1$ 表示抑制）；最後看右邊的 AND，它只看兩個隱藏神經元的輸出。 / **Caption `lec02-kp05-xor-network`:** Look first at the two input circles; then at the two hidden neurons, whose line labels are weights ($-1$ means inhibitory); finally at the AND, which sees only the outputs of the two hidden neurons.

[補充] 來源只描述網路結構，沒有給 XOR 的手工權重。上面的權重是用 KP01 的閘組出來的，用來回答第1章 KP08 與 KP04 留下的「多層且有非線性能不能做 XOR」。怎麼讓機器自己學出多層權重，是第3章反向傳播的問題。

[Added] The source describes the structure but gives no hand-made XOR weights. The weights above are built from the gates of KP01 to answer the question left by Chapter 1 KP08 and KP04: can a multilayer nonlinear network do XOR? How a machine learns the weights of a multilayer network by itself is the question of Chapter 3, backpropagation.

### 3. 為什麼一定要非線性：全線性會坍縮（Why nonlinearity is a must: all-linear collapses）

取同一組權重，但把階梯換成恆等活化 $\sigma(a)=a$。隱藏層輸出 $h_1=x_1+x_2-1$、$h_2=-x_1-x_2+1.5$，輸出 $y=h_1+h_2-2=-1.5$：不管輸入是什麼，輸出都是常數，連 XOR 的一半都做不到。原因是兩層合起來的矩陣 $\mathbf{W}_2\mathbf{W}_1=(1,1)\begin{pmatrix}1&1\\-1&-1\end{pmatrix}=(0,0)$。

Take the same weights but replace the step by the identity activation $\sigma(a)=a$. The hidden outputs are $h_1=x_1+x_2-1$ and $h_2=-x_1-x_2+1.5$, and the output is $y=h_1+h_2-2=-1.5$: whatever the input, the output is a constant, and not even half of XOR is achieved. The reason is that the combined matrix of the two layers is $\mathbf{W}_2\mathbf{W}_1=(1,1)\begin{pmatrix}1&1\\-1&-1\end{pmatrix}=(0,0)$.

一般地說，三層網路是 $\mathbf{y}=\sigma_3(\mathbf{W}_3^\top\sigma_2(\mathbf{W}_2^\top\sigma_1(\mathbf{W}_1^\top\mathbf{x})))$。若所有 $\sigma_i$ 都是線性的，每個 $\sigma_i\mathbf{W}_i^\top$ 仍是一個矩陣，記為 $\mathbf{V}_i^\top$，連乘還是一個矩陣：

In general a three-layer network is $\mathbf{y}=\sigma_3(\mathbf{W}_3^\top\sigma_2(\mathbf{W}_2^\top\sigma_1(\mathbf{W}_1^\top\mathbf{x})))$. If all $\sigma_i$ are linear, each $\sigma_i\mathbf{W}_i^\top$ is still a matrix, written $\mathbf{V}_i^\top$, and a product of matrices is again a matrix:

$$
\mathbf{y}=\mathbf{V}_3^\top\mathbf{V}_2^\top\mathbf{V}_1^\top\mathbf{x}=\mathbf{V}^\top\mathbf{x}
$$

疊 100 層也只是一次線性投影。加上偏置也一樣：線性加偏置是仿射函數，連乘仍是仿射函數，邊界仍是直線。

A hundred layers are still one linear projection. Biases do not help: a linear map plus a bias is affine, a product of affine maps is again affine, and the boundary stays a straight line.

![同一組隨機權重：恆等活化坍縮成一條直線邊界，ReLU 的邊界彎折](../assets/figures/lec02-kp05-collapse-vs-relu.png)

**圖說 `lec02-kp05-collapse-vs-relu`**：先看左圖，三層網路的邊界只是一條直線；再看右圖，權重完全一樣，只把活化換成 ReLU，邊界彎折成好幾段，把平面分成四塊。 / **Caption `lec02-kp05-collapse-vs-relu`:** Look first at the left panel: the boundary of the three-layer network is one straight line; then at the right panel: with exactly the same weights but ReLU activations, the boundary bends into several segments and splits the plane into four pieces.

### 本單元重點整理（KP05 summary）

**一句話重點**：把神經元接成前饋的層，每層「先線性加總、後非線性活化」，就能做出單一神經元做不到的 XOR；但若所有活化都是線性，再多層也坍縮成一層。

**One-sentence summary**: Connect neurons into feedforward layers, each doing "linear sum first, nonlinear activation second", and XOR, which one neuron cannot do, becomes possible; but if every activation is linear, any number of layers collapses into one.

**白話說明**：像接力的流水線：第一站的幾個工人各自看原料做簡單判斷，第二站只看第一站的結果。每一站就是一層。若每個工人只會「照比例放大」，接再多站也只是把比例乘起來，仍是一個比例。

**Plain explanation**: Like a relay assembly line: the workers at the first station each look at the raw material and make a simple judgment, and the second station looks only at the results of the first. Each station is a layer. If every worker can only "scale proportionally", any number of stations just multiplies the scales together into one scale.

**正式定義**：$a_i=\sum_\ell w_{i\ell}z_\ell$、$z_i=\sigma(a_i)$；$\mathbf{z}^{(k)}=\sigma(\mathbf{W}_k^\top\mathbf{z}^{(k-1)})$；全線性時 $\mathbf{y}=\mathbf{V}^\top\mathbf{x}$，$\mathbf{V}=\mathbf{V}_1\mathbf{V}_2\mathbf{V}_3$。

**Formal definition**: $a_i=\sum_\ell w_{i\ell}z_\ell$ and $z_i=\sigma(a_i)$; $\mathbf{z}^{(k)}=\sigma(\mathbf{W}_k^\top\mathbf{z}^{(k-1)})$; when all linear, $\mathbf{y}=\mathbf{V}^\top\mathbf{x}$ with $\mathbf{V}=\mathbf{V}_1\mathbf{V}_2\mathbf{V}_3$.

**例子**：2-2-1 網路（OR、NAND、AND）對 $(0,0),(0,1),(1,0),(1,1)$ 輸出 $0,1,1,0$；同權重改恆等活化，輸出恆為 $-1.5$，$\mathbf{W}_2\mathbf{W}_1=(0,0)$。

**Example**: The 2-2-1 network (OR, NAND, AND) outputs $0,1,1,0$ for $(0,0),(0,1),(1,0),(1,1)$; with the same weights and identity activations the output is the constant $-1.5$, since $\mathbf{W}_2\mathbf{W}_1=(0,0)$.

**常見迷思**：

- 「層數越多，網路一定越強」——不對。全線性時 100 層與 1 層完全等價；力量來自每層之間的非線性。
- 「有非線性就一定學得會」——不對。上面的權重是手工湊的；隱藏神經元太少或權重不對，一樣解不了，怎麼學出權重是第3章的事。

**Common misconceptions**:

- "More layers always mean a stronger network." No: with linear activations 100 layers equal 1 layer; the strength comes from the nonlinearity between layers.
- "With nonlinearity it can always be learned." No: the weights above were set by hand; with too few hidden neurons or wrong weights the problem is still unsolved, and how to learn the weights is the topic of Chapter 3.

**圖**：本單元兩張圖，`lec02-kp05-xor-network`（網路結構與權重，功能 1）、`lec02-kp05-collapse-vs-relu`（恆等與 ReLU 的邊界，功能 1、3），以 `assets/figures/src/lec02-kp0507-figures.py` 繪製。網路結構在簡報中用原生節點與連線畫出。

**Figures**: Two figures in this unit, `lec02-kp05-xor-network` (structure and weights; function 1) and `lec02-kp05-collapse-vs-relu` (boundaries with identity versus ReLU; functions 1, 3), drawn by `assets/figures/src/lec02-kp0507-figures.py`. The network structure is drawn with native nodes and connectors in the slides.

**想深入**（選讀）：「多層感知機」的名字來自每顆神經元都是 Perceptron 式的神經元。第1章 KP08 提醒過，「堆疊單元」與「能用誤差有效訓練所有層」是兩回事；本節只回答前者。

**Going deeper** (optional): The name "multilayer perceptron" comes from every neuron being Perceptron-style. Chapter 1 KP08 reminded us that "stacking units" and "training all layers effectively with an error signal" are different things; this note answers only the former.

**小練習**：

1. 用上面的 XOR 網路，算輸入 $(1,0)$ 時 $s_1,h_1,s_2,h_2,s_3,o$ 各是多少。（提示：照表格的算式一格一格代入。）
2. 兩層恆等網路 $\mathbf{W}_1=\begin{pmatrix}1&0\\1&1\end{pmatrix}$、$\mathbf{W}_2=(1,2)$，輸出 $y=\mathbf{W}_2\mathbf{W}_1\mathbf{x}$。合併後的單一矩陣是多少？（提示：先算矩陣乘法。）
3. 為什麼把網路加深到 10 層，卻都用恆等活化，不會比 1 層強？（提示：10 個矩陣相乘的結果是什麼？）

**Practice**:

1. With the XOR network above, compute $s_1,h_1,s_2,h_2,s_3,o$ for the input $(1,0)$. (Hint: substitute cell by cell with the formulas of the table.)
2. A two-layer identity network has $\mathbf{W}_1=\begin{pmatrix}1&0\\1&1\end{pmatrix}$ and $\mathbf{W}_2=(1,2)$, with output $y=\mathbf{W}_2\mathbf{W}_1\mathbf{x}$. What is the merged single matrix? (Hint: do the matrix product first.)
3. Why is a 10-layer network with all identity activations no stronger than a 1-layer one? (Hint: what is the product of 10 matrices?)

**解題示範**（第 1 題）：

1. $s_1=1+0-1=0$，$h_1=f(0)=1$。
2. $s_2=-1-0+1.5=0.5$，$h_2=f(0.5)=1$。
3. $s_3=h_1+h_2-2=1+1-2=0$，$o=f(0)=1$。
4. 檢查：XOR$(1,0)=1$，與第 2 節表格的 $(1,0)$ 列一致。

**Worked solution** (exercise 1):

1. $s_1=1+0-1=0$ and $h_1=f(0)=1$.
2. $s_2=-1-0+1.5=0.5$ and $h_2=f(0.5)=1$.
3. $s_3=h_1+h_2-2=1+1-2=0$ and $o=f(0)=1$.
4. Check: XOR$(1,0)=1$, consistent with the $(1,0)$ row of the table in section 2.

**自我檢核**（來源四題，不附答案）：

| 題 / Question | A | B | C | D | 提示 / Hint |
|---|---|---|---|---|---|
| 1. MLP 的「前饋」指的是？ / What does "feedforward" mean? | 資訊回饋到前一層 | 資訊只向前傳、不回頭 | 只有一層神經元 | 神經元可隨意連線 | 看第 1 節。 / Section 1. |
| 2. 輸入層在文獻的稱呼上常被視為？ / How is the input layer often counted? | 輸出層 | 隱藏層之一 | 無作用層 | 最後一層 | 看層數的算法。 / See how layers are counted. |
| 3. 如果所有活化函數都是線性的，整座網路會變成？ / What if all activations are linear? | 非線性模型 | 一次線性投影 | 指數級複雜模型 | 無法訓練 | 看第 3 節。 / Section 3. |
| 4. 網路能處理 XOR 這類非線性模式，表達力來自？ / Where does the power for XOR come from? | 更多隱藏層 | 非線性活化函數 | 更大的學習率 | 更小的批次 | 全線性的結果是什麼？ / What happens when all is linear? |

**留給下一個知識點的問題**：階梯函數可以當非線性，但它在 0 處不可導、其他地方導數為 0。要讓機器用梯度學習，該選哪一種非線性？

**Question for the next knowledge point**: The step function can serve as the nonlinearity, but it is not differentiable at 0 and has derivative 0 elsewhere. To let a machine learn by gradients, which nonlinearity should we choose?

### 來源更正與取捨（Corrections to the source）

- 來源說 MLP「由 Rosenblatt 在 1958 年和 Perceptron 同一篇論文提出」，並說「直到反向傳播出現才真正能學」。依第1章 KP08，本講義不採單一年份與單一來源的說法：多層結構的想法早就有，系統化的誤差反傳在 1986 年普及，兩者要分開講。
- 來源沒有給 XOR 的多層手算例，上面的 2-2-1 網路為明示補充。
- 來源的「輸入層通常也被算作隱藏層之一」只是計數習慣；本講義用「輸入、隱藏、輸出」三種層，並說明文獻數法不同。

- The source says the MLP was "proposed by Rosenblatt in 1958 in the same paper as the Perceptron" and "could truly learn only after backpropagation appeared". Following Chapter 1 KP08, this note does not adopt a single year and a single origin: the idea of multilayer structure is old, and systematic error backpropagation became popular in 1986; the two must be told apart.
- The source gives no multilayer hand-computed XOR example; the 2-2-1 network above is an explicit addition.
- The source's "the input layer is often counted as one of the hidden layers" is only a counting habit; this note uses the three kinds of layers input, hidden and output and notes that papers count differently.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 多層感知機 | multilayer perceptron (MLP) |
| 前饋 | feedforward |
| 全連接層／密集層 | fully connected (dense) layer |
| 輸入層、隱藏層、輸出層 | input, hidden, output layer |
| 活化函數 | activation function |
| 坍縮 | collapse |
| 仿射 | affine |
| 恆等函數 | identity function |
