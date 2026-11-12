# 第8週 KP02：BPTT 把各步損失相加，隱藏狀態的梯度由兩條路匯合（BPTT adds the step losses, and the hidden-state gradient merges two paths）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第8週的第二個知識點（KP02），接在 KP01 之後。KP01 留下一個問題：同一個 $\mathbf{W}$ 每一步都用了一次，預測錯了，誤差要怎麼倒回去？這一節回答前半段：先算出「損失對每一步隱藏狀態的梯度」。

This is the second knowledge point (KP02) of Week 8 and follows KP01. KP01 left a question: the same $\mathbf{W}$ is used once at every step, so when a prediction is wrong, how does the error travel back? This note answers the first half: it computes "the gradient of the loss with respect to the hidden state of every step".

先備知識是 KP01 的三條前向公式、第2週的反向傳播與鏈鎖律，以及微積分的導數。第一次出現的詞與符號都會定義。所有數字都由 `assets/figures/src/lec08-kp02-figures.py` 算出，可以自己重算。

The prerequisites are the three forward equations of KP01, backpropagation and the chain rule of Week 2, and derivatives from calculus. Every term and symbol is defined when it first appears. Every number is computed by `assets/figures/src/lec08-kp02-figures.py`, so you can recompute it yourself.

## KP02 BPTT 把各步損失相加，隱藏狀態的梯度由兩條路匯合（BPTT adds the step losses, and the hidden-state gradient merges two paths）

### 學習目標（Learning objectives）

讀完之後，你能說出 BPTT 就是「在時間展開圖上做一般的反向傳播」。你能寫出總損失 $\mathcal{L}=\sum_{t=1}^{T}\mathcal{L}_t$，並說明共用的權重為什麼要把各步梯度相加。你能寫出 $\delta_t$ 的兩條路公式與最後一步的起點 $\delta_T$。你也能用三步的純量例子，從 $t=3$ 倒算回 $t=1$。

After reading, you can say that BPTT is "ordinary backpropagation on the unrolled graph". You can write the total loss $\mathcal{L}=\sum_{t=1}^{T}\mathcal{L}_t$ and explain why a shared weight must add up its step gradients. You can write the two-path formula for $\delta_t$ and its starting point $\delta_T$ at the last step. You can also work a three-step scalar example backward from $t=3$ to $t=1$.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $t,\ T$ | 時間步；展開的步數 | time step; number of unrolled steps |
| $\mathcal{L}_t$ | 第 $t$ 步的損失（一個數） | loss of step $t$ (a number) |
| $\mathcal{L}$ | 總損失，各步損失的和 | total loss, the sum of the step losses |
| $\mathbf{h}_t\in\mathbb{R}^{p},\ \mathbf{y}_t\in\mathbb{R}^{q}$ | 第 $t$ 步的隱藏狀態與輸出（同 KP01） | hidden state and output of step $t$ (as in KP01) |
| $\mathbf{U},\ \mathbf{W},\ \mathbf{V},\ \mathbf{b}_i,\ \mathbf{b}_y$ | 所有時間步共用的參數（同 KP01） | parameters shared by all steps (as in KP01) |
| $\frac{\partial\mathcal{L}}{\partial\mathbf{y}_t}\in\mathbb{R}^{q}$ | 總損失對輸出 $\mathbf{y}_t$ 的梯度 | gradient of the total loss with respect to the output $\mathbf{y}_t$ |
| $\delta_t:=\frac{\partial\mathcal{L}}{\partial\mathbf{h}_t}\in\mathbb{R}^{p}$ | 總損失對隱藏狀態 $\mathbf{h}_t$ 的梯度 | gradient of the total loss with respect to the hidden state $\mathbf{h}_t$ |
| $\operatorname{diag}(\mathbf{a})$ | 把向量 $\mathbf{a}$ 放在對角線的方陣 | square matrix with the vector $\mathbf{a}$ on its diagonal |
| $\mathbf{V}^{\top},\ \mathbf{W}^{\top}$ | 矩陣的轉置 | transposes of the matrices |

本講義把梯度都寫成**行向量**（column vector），和 $\mathbf{h}_t$ 的形狀一樣。

In this note every gradient is written as a **column vector**, the same shape as $\mathbf{h}_t$.

### 1. 把展開圖當成一個很深的網路（Treating the unrolled graph as one deep network）

KP01 的問題是：共用的權重要怎麼知道該改多少？答案是**隨時間反向傳播**，簡稱 BPTT（backpropagation through time）。做法是先把 RNN 沿時間展開，再把展開圖當成很深的前饋網路，用第2週的反向傳播求梯度。

KP01's question was how a shared weight learns how much to change. The answer is **backpropagation through time** (BPTT). It is direct: first unroll the RNN along time, then treat the unrolled graph as a very deep feedforward network, and get the gradients with the backpropagation of Week 2.

展開圖裡，每個時間步就像一層。所以 BPTT 不是新的微分規則，只是把鏈鎖律用在「層數等於時間步數」的網路上。

In the unrolled graph every time step acts like a layer. So BPTT is not a new rule of differentiation; it is the chain rule applied to a network whose number of layers equals the number of time steps.

序列可能很長，從第一個字展開到最後一個字並不實際。來源因此假設網路有 **$T$ 階馬可夫性質**（$T$-th order Markov property）。意思是：計算時只往回看最近的 $T$ 步。於是每次只展開 $T$ 步，訓練的目標是這 $T$ 步損失的總和：

A sequence can be very long, and unrolling from the first word to the last is not practical. So the source assumes the network has the **$T$-th order Markov property**: the computation looks back only over the most recent $T$ steps. Each time we unroll only $T$ steps, and the training goal is the sum of the losses of these $T$ steps:

$$
\mathcal{L}=\sum_{t=1}^{T}\mathcal{L}_t.
$$

這裡的 $\mathcal{L}_t$ 是第 $t$ 步的損失，例如第2週學過的均方誤差或交叉熵。整段序列只看一個數字 $\mathcal{L}$，越小越好。

Here $\mathcal{L}_t$ is the loss of step $t$, for example the mean squared error or the cross-entropy of Week 2. The whole sequence is judged by one number $\mathcal{L}$; smaller is better.

類比：把 $T$ 步想成一場有 $T$ 關的遊戲。每一關扣一次分 $\mathcal{L}_t$，整場的總扣分就是各關相加。這個類比的侷限是：遊戲每關規則不同，RNN 每一步卻用同一組參數。

Analogy: think of the $T$ steps as a game with $T$ levels. Each level costs a penalty $\mathcal{L}_t$, and the total penalty is the sum over the levels. The limit of the analogy: game levels have different rules, but an RNN uses the same parameters at every step.

BPTT 要算的是 $\mathcal{L}$ 對這五組參數的梯度：

BPTT must compute the gradient of $\mathcal{L}$ with respect to these five groups of parameters:

$$
\{\mathbf{V},\ \mathbf{W},\ \mathbf{U},\ \mathbf{b}_i,\ \mathbf{b}_y\}.
$$

關鍵在「共用」。同一個 $\mathbf{W}$ 在第 $1$、$2$、$\dots$、$T$ 步都出現過。$\mathbf{W}$ 稍微改一點，每一步都會跟著變。所以依多變數鏈鎖律，總梯度是每一步貢獻的和。

The key is "shared". The same $\mathbf{W}$ appears at steps $1$, $2$, $\dots$, $T$. A small change of $\mathbf{W}$ changes every step. So by the multivariable chain rule, the total gradient is the sum of the contributions of every step.

例子（來源的數字）：三步的損失是 $\mathcal{L}_1=0.50$、$\mathcal{L}_2=0.30$、$\mathcal{L}_3=0.10$，總損失 $\mathcal{L}=0.50+0.30+0.10=0.90$。若三步對純量 $W$ 的梯度貢獻是 $0.05$、$0.15$、$0.20$，總梯度就是 $0.05+0.15+0.20=0.40$。

Example (numbers from the source): the losses of three steps are $\mathcal{L}_1=0.50$, $\mathcal{L}_2=0.30$, $\mathcal{L}_3=0.10$, so the total loss is $\mathcal{L}=0.50+0.30+0.10=0.90$. If the three steps contribute $0.05$, $0.15$ and $0.20$ to the gradient of a scalar $W$, the total gradient is $0.05+0.15+0.20=0.40$.

![左圖：三根藍色長條是 L_1=0.50、L_2=0.30、L_3=0.10，紅色折線是累加和 0.50、0.80、0.90。右圖：一根疊起來的長條，由 0.05、0.15、0.20 三段組成，總高 0.40；旁邊三根分開的長條分別是 0.05、0.15、0.20（Left: blue bars of the step losses 0.50, 0.30, 0.10 and a red running sum 0.50, 0.80, 0.90; right: one stacked bar made of 0.05, 0.15 and 0.20 that reaches 0.40, beside the three separate bars）](../assets/figures/lec08-kp02-loss-sum.png)

**圖說 `lec08-kp02-loss-sum`**〔功能：一目了然〕：先看左圖：藍色長條是每一步的損失，紅線累加到總損失 $0.90$。再看右圖：右邊三根長條是各步對 $W$ 的梯度。左邊把它們疊成一根，高度 $0.40$ 就是總梯度。兩張圖做的是同一件事：每一步的東西都要加進去。

**Caption `lec08-kp02-loss-sum`** [Function: seeing it at a glance]: First look at the left panel: the blue bars are the step losses, and the red line keeps adding them until it stops at the total loss $0.90$. Then look at the right panel: the three bars on the right are the step gradients for $W$, and the left bar stacks them; its height $0.40$ is the total gradient. Both panels do the same thing: every step must be added in.

特例 `[補充]`：有些任務只在最後一步才有答案，例如整句的情緒分類。這時前面各步的損失是 $0$，總損失只剩 $\mathcal{L}=\mathcal{L}_T$。梯度只從最後一步開始往回傳。

Special case `[Added]`: some tasks have an answer only at the last step, for example classifying the sentiment of a whole sentence. Then the losses of the earlier steps are $0$ and the total loss is just $\mathcal{L}=\mathcal{L}_T$. The gradient starts only at the last step and flows back.

假設與失效：$T$ 步的窗口要夠長，才裝得下任務需要的歷史。關鍵線索若在 $T$ 步之前，窗口外的梯度根本不會算，網路就學不到這個關聯 `[補充]`。

Assumption and failure: the window of $T$ steps must be long enough to hold the history the task needs. If the key clue lies more than $T$ steps back, the gradient outside the window is never computed, and the network cannot learn that link `[Added]`.

常見錯誤：以為每一步有自己的 $\mathbf{W}_t$，各自更新。展開圖裡只有一個 $\mathbf{W}$，所以只有一個總梯度，它是各步貢獻的和。

Common error: thinking that every step has its own $\mathbf{W}_t$ that is updated separately. The unrolled graph has only one $\mathbf{W}$, so there is only one total gradient, the sum of the step contributions.

### 2. 往回走的第一步：損失對輸出的梯度（The first step back: the gradient with respect to the output）

要把各步的貢獻算出來，得從離損失最近的地方開始。離 $\mathcal{L}_t$ 最近的是輸出 $\mathbf{y}_t$。

To compute the contributions of the steps, start where the loss is closest. The quantity closest to $\mathcal{L}_t$ is the output $\mathbf{y}_t$.

$\mathbf{y}_t$ 只出現在第 $t$ 步的損失裡，別的步的損失都不直接用到它。所以總損失對它的梯度，就等於單步損失對它的梯度：

$\mathbf{y}_t$ appears only in the loss of step $t$; no other step's loss uses it directly. So the gradient of the total loss with respect to it equals the gradient of the single-step loss:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{y}_t}=\frac{\partial\mathcal{L}_t}{\partial\mathbf{y}_t}.
$$

如果輸出後面還接 softmax，$\hat{\mathbf{y}}_t=\operatorname{softmax}(\mathbf{y}_t)$，就再多乘一段鏈鎖律：

If a softmax follows the output, $\hat{\mathbf{y}}_t=\operatorname{softmax}(\mathbf{y}_t)$, multiply by one more link of the chain rule:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{y}_t}=\frac{\partial\hat{\mathbf{y}}_t}{\partial\mathbf{y}_t}^{\top}\,\frac{\partial\mathcal{L}_t}{\partial\hat{\mathbf{y}}_t}.
$$

這裡 $\frac{\partial\hat{\mathbf{y}}_t}{\partial\mathbf{y}_t}$ 是 softmax 的 $q\times q$ 導數矩陣，轉置是為了讓結果是行向量。

Here $\frac{\partial\hat{\mathbf{y}}_t}{\partial\mathbf{y}_t}$ is the $q\times q$ derivative matrix of the softmax; the transpose keeps the result a column vector.

具體例子 `[補充]`：用均方誤差 $\mathcal{L}_t=\frac{1}{2}(y_t-r_t)^2$，$r_t$ 是目標值，則 $\frac{\partial\mathcal{L}_t}{\partial y_t}=y_t-r_t$。若 $y_t=1.5$、$r_t=0.5$，損失是 $0.5$，梯度是 $1.0$：輸出比目標大了 $1.0$。softmax 配交叉熵時，結果也很簡單（第2週）：機率減去正確答案的 one-hot 向量。

Concrete example `[Added]`: with the mean squared error $\mathcal{L}_t=\frac{1}{2}(y_t-r_t)^2$, where $r_t$ is the target, $\frac{\partial\mathcal{L}_t}{\partial y_t}=y_t-r_t$. If $y_t=1.5$ and $r_t=0.5$, the loss is $0.5$ and the gradient is $1.0$: the output is $1.0$ above the target. With softmax and cross-entropy the result is also simple: "the probabilities minus the one-hot vector of the correct answer" (Week 2).

常見錯誤：以為後面幾步的損失也會直接拉動 $\mathbf{y}_t$。後面的步只透過 $\mathbf{h}_t$ 受影響，不透過 $\mathbf{y}_t$；那條路留給下一節的 $\delta_t$。

Common error: thinking that the losses of later steps also pull on $\mathbf{y}_t$ directly. Later steps are affected only through $\mathbf{h}_t$, not through $\mathbf{y}_t$; that path belongs to $\delta_t$ in the next section.

### 3. 隱藏狀態的梯度由兩條路匯合（The hidden-state gradient merges two paths）

有了 $\frac{\partial\mathcal{L}}{\partial\mathbf{y}_t}$，再往下一層就是 $\mathbf{h}_t$。定義**隱藏狀態梯度**（hidden-state gradient）：

With $\frac{\partial\mathcal{L}}{\partial\mathbf{y}_t}$ in hand, the next layer down is $\mathbf{h}_t$. Define the **hidden-state gradient**:

$$
\delta_t:=\frac{\partial\mathcal{L}}{\partial\mathbf{h}_t}.
$$

它回答：$\mathbf{h}_t$ 稍微變一點，總損失會變多少。

It answers: if $\mathbf{h}_t$ changes a little, how much does the total loss change?

先看前向時 $\mathbf{h}_t$ 被送到哪裡。它走了兩條路。第一條往上：$\mathbf{y}_t=\mathbf{V}\mathbf{h}_t+\mathbf{b}_y$。第二條往右：$\mathbf{h}_{t+1}=\tanh(\mathbf{W}\mathbf{h}_t+\mathbf{U}\mathbf{x}_{t+1}+\mathbf{b}_i)$。

First look at where $\mathbf{h}_t$ goes in the forward pass. It takes two paths. The first goes up: $\mathbf{y}_t=\mathbf{V}\mathbf{h}_t+\mathbf{b}_y$. The second goes right: $\mathbf{h}_{t+1}=\tanh(\mathbf{W}\mathbf{h}_t+\mathbf{U}\mathbf{x}_{t+1}+\mathbf{b}_i)$.

多變數鏈鎖律說：一個變數經過幾條路影響結果，梯度就把每條路的貢獻加起來。先用純量看清楚（$p=q=1$）：

The multivariable chain rule says: when a variable affects the result through several paths, the gradient adds the contribution of every path. See it first with scalars ($p=q=1$):

$$
\delta_t=\underbrace{\frac{\partial\mathcal{L}}{\partial y_t}\cdot V}_{\text{path 1}}+\underbrace{\delta_{t+1}\cdot(1-h_{t+1}^2)\cdot W}_{\text{path 2}}.
$$

第一路：$y_t=Vh_t+b_y$，所以 $\frac{\partial y_t}{\partial h_t}=V$。第二路：$h_{t+1}=\tanh(i_{t+1})$，$i_{t+1}=Wh_t+\dots$。$\tanh$ 的導數是 $1-\tanh^2$，所以 $\frac{\partial h_{t+1}}{\partial h_t}=(1-h_{t+1}^2)\,W$。注意這裡配的是 $h_{t+1}$，因為被微分的是第 $t+1$ 步的 $\tanh$。

Path 1: $y_t=Vh_t+b_y$, so $\frac{\partial y_t}{\partial h_t}=V$. Path 2: $h_{t+1}=\tanh(i_{t+1})$ with $i_{t+1}=Wh_t+\dots$. The derivative of $\tanh$ is $1-\tanh^2$, so $\frac{\partial h_{t+1}}{\partial h_t}=(1-h_{t+1}^2)\,W$. Note that $h_{t+1}$ appears here, because the $\tanh$ being differentiated is the one of step $t+1$.

換成向量，每個乘法都換成矩陣，並用轉置讓形狀對得上：

For vectors, every product becomes a matrix, and transposes make the shapes match:

$$
\delta_t=\mathbf{V}^{\top}\frac{\partial\mathcal{L}}{\partial\mathbf{y}_t}+\mathbf{W}^{\top}\operatorname{diag}\!\left(1-\mathbf{h}_{t+1}^2\right)\delta_{t+1}.
$$

這裡 $\mathbf{h}_{t+1}^2$ 是逐個分量平方。形狀檢查：$\mathbf{V}^{\top}$ 是 $p\times q$，乘上 $q\times1$ 得 $p\times1$；$\mathbf{W}^{\top}\operatorname{diag}(\cdot)$ 是 $p\times p$，乘上 $p\times1$ 也得 $p\times1$。兩項一樣大，才能相加。

Here $\mathbf{h}_{t+1}^2$ squares each component. Shape check: $\mathbf{V}^{\top}$ is $p\times q$, times $q\times1$ gives $p\times1$; $\mathbf{W}^{\top}\operatorname{diag}(\cdot)$ is $p\times p$, times $p\times1$ also gives $p\times1$. Both terms have the same size, so they can be added.

最後一步 $t=T$ 沒有下一步，第二路不存在。這是倒算的**起點**（boundary condition）：

The last step $t=T$ has no next step, so path 2 does not exist. This is the **boundary condition** where the backward computation starts:

$$
\delta_T=\mathbf{V}^{\top}\frac{\partial\mathcal{L}_T}{\partial\mathbf{y}_T}.
$$

因為 $\delta_t$ 要用到 $\delta_{t+1}$，計算順序一定是 $t=T,\ T-1,\ \dots,\ 1$，和前向相反。

Because $\delta_t$ needs $\delta_{t+1}$, the order of computation must be $t=T,\ T-1,\ \dots,\ 1$, the reverse of the forward pass.

類比：把 $\mathbf{h}_t$ 想成一座水庫。前向時它開兩條水渠，一條供給輸出 $\mathbf{y}_t$，一條流進下一座水庫 $\mathbf{h}_{t+1}$。反向時水倒流，水庫收到的總量是兩條渠的回流相加。這個類比的侷限是：水量不會是負的，梯度乘上負的權重卻可以變號。

Analogy: think of $\mathbf{h}_t$ as a reservoir. In the forward pass it opens two channels, one feeding the output $\mathbf{y}_t$ and one flowing into the next reservoir $\mathbf{h}_{t+1}$. In the backward pass the water flows back, and the reservoir receives the sum of both return flows. The limit of the analogy: water is never negative, but a gradient multiplied by a negative weight can change sign.

常見錯誤：只寫第一路，忘了未來那一路；或第二路漏掉 $\tanh$ 的導數 $(1-\mathbf{h}_{t+1}^2)$。前向經過了 $\tanh$，反向就一定要乘它的導數。

Common error: writing only path 1 and forgetting the path from the future; or dropping the $\tanh$ derivative $(1-\mathbf{h}_{t+1}^2)$ in path 2. The forward pass went through $\tanh$, so the backward pass must multiply by its derivative.

### 4. 手算三步：從 $t=3$ 倒推回 $t=1$（Three steps by hand: from $t=3$ back to $t=1$）

公式有了，用數字走一遍。這個例子取自來源教授祕笈，下一單元（KP03）會接著用它算權重的梯度。

The formula is ready; now run it with numbers. The example comes from the source's professor's notes, and the next unit (KP03) continues it to compute the weight gradients.

本例的數字：純量 $p=q=1$，$T=3$，$V=1.0$，$W=0.5$。為了好算，假設每一步的 $\tanh$ 導數都是 $1-h_{t+1}^2=0.8$（也就是 $h_{t+1}=\pm0.4472$）。再假設每一步的輸出梯度都是 $\frac{\partial\mathcal{L}_t}{\partial y_t}=1.0$，例如輸出比目標大 $1.0$。於是第二路每往回一步，就乘上 $W(1-h_{t+1}^2)=0.5\times0.8=0.4$。

The numbers of the example: scalars $p=q=1$, $T=3$, $V=1.0$, $W=0.5$. To keep it simple, assume the $\tanh$ derivative is $1-h_{t+1}^2=0.8$ at every step (that is, $h_{t+1}=\pm0.4472$). Also assume the output gradient is $\frac{\partial\mathcal{L}_t}{\partial y_t}=1.0$ at every step, for example an output $1.0$ above its target. So path 2 multiplies by $W(1-h_{t+1}^2)=0.5\times0.8=0.4$ for every step back.

計算 $\delta_t$ 用不到 $U$ 與輸入 $x_t$；它們只在前向時決定 $h_t$ 的大小，而本例已直接給定導數 $0.8$。

Computing $\delta_t$ does not need $U$ or the inputs $x_t$; they only decide the size of $h_t$ in the forward pass, and the example gives the derivative $0.8$ directly.

$$
\begin{aligned}
\delta_3&=1.0\times1.0=1.00,\\
\delta_2&=1.0\times1.0+1.00\times0.4=1.00+0.40=1.40,\\
\delta_1&=1.0\times1.0+1.40\times0.4=1.00+0.56=1.56.
\end{aligned}
$$

第 $3$ 步只有第一路。第 $2$ 步兩路相加：自己的輸出給 $1.00$，未來給 $0.40$。第 $1$ 步的未來那一路是 $0.56$，比 $0.40$ 大。原因是 $\delta_2$ 已經把第 $3$ 步的影響也帶進來了。

Step $3$ has only path 1. Step $2$ adds two paths: its own output gives $1.00$ and the future gives $0.40$. At step $1$ the future path gives $0.56$, more than $0.40$, because $\delta_2$ already carries the influence of step $3$ as well.

![三個時間步的展開圖。上排是 y_1、y_2、y_3，下排是 h_1、h_2、h_3。橘色箭頭從每個 y_t 往下指到 h_t，標「乘 V，+1.00」；綠色箭頭從 h_3 往左到 h_2、從 h_2 往左到 h_1，標「乘 W(1-h^2)=乘 0.4」，數值 +0.40 與 +0.56。每個 h_t 下方寫 δ_1=1.56、δ_2=1.40、δ_3=1.00，最下方箭頭表示倒算順序 3→2→1（Unrolled graph of three steps: orange arrows from each y_t down to h_t carry +1.00, green arrows from h_3 to h_2 and h_2 to h_1 carry +0.40 and +0.56, giving δ_1=1.56, δ_2=1.40, δ_3=1.00 computed in the order 3, 2, 1）](../assets/figures/lec08-kp02-two-paths.png)

**圖說 `lec08-kp02-two-paths`**〔功能：一目了然〕：先看橘色箭頭：每個 $y_t$ 往下送回 $+1.00$，即第一路。再看綠色箭頭：它們從右往左走，每走一格乘 $0.4$，這是第二路。最右邊的 $h_3$ 沒有綠色箭頭進來，所以 $\delta_3$ 只有一項。最後看最下面的灰色箭頭：計算從右邊開始，往左做。

**Caption `lec08-kp02-two-paths`** [Function: seeing it at a glance]: First look at the orange arrows: every $y_t$ sends $+1.00$ back down; this is path 1. Then look at the green arrows: they go from right to left and multiply by $0.4$ at every hop; this is path 2. The rightmost $h_3$ has no green arrow coming in, so $\delta_3$ has only one term. Finally look at the grey arrow at the bottom: the computation starts on the right and goes left.

對照 `[補充]`：假設只有最後一步有損失（第 1 節的特例）。前兩步的輸出梯度是 $0$，第一路消失，只剩未來那一路。這時 $\delta_3=1.00$、$\delta_2=1.00\times0.4=0.40$、$\delta_1=0.40\times0.4=0.16$。梯度每往回一步就縮成 $0.4$ 倍。

Compare `[Added]`: if only the last step has a loss (the special case of Section 1), the output gradients of the first two steps are $0$; path 1 vanishes and only the future path is left. Then $\delta_3=1.00$, $\delta_2=1.00\times0.4=0.40$ and $\delta_1=0.40\times0.4=0.16$. The gradient shrinks to $0.4$ times its size at every step back.

![三組長條，分別是 δ_3、δ_2、δ_1。每組左邊是「每步都有輸出」：δ_3=1.00（全橘），δ_2=1.40（橘 1.00 加綠 0.40），δ_1=1.56（橘 1.00 加綠 0.56）。每組右邊是「只有最後一步有輸出」：1.00（全橘）、0.40（全綠）、0.16（全綠）（Three groups of bars for δ_3, δ_2, δ_1; with outputs at every step they are 1.00, 1.40, 1.56 made of orange 1.00 plus green; with the last step only they are 1.00, 0.40, 0.16）](../assets/figures/lec08-kp02-delta-parts.png)

**圖說 `lec08-kp02-delta-parts`**〔功能：精準度〕：先看每組左邊的長條（每步都有輸出）：橘色永遠是 $1.00$，綠色從 $0$ 長到 $0.40$、$0.56$，所以總高度往回越來越大。再看每組右邊的長條（只有最後一步有輸出）：除了 $\delta_3$，全是綠色，而且 $1.00\to0.40\to0.16$ 越來越短。兩邊的差別完全來自橘色的第一路。

**Caption `lec08-kp02-delta-parts`** [Function: precision]: First look at the left bar of each group (an output at every step): the orange part is always $1.00$, and the green part grows from $0$ to $0.40$ and $0.56$, so the total grows as we go back. Then look at the right bar of each group (an output at the last step only): apart from $\delta_3$ it is all green, and it gets shorter, $1.00\to0.40\to0.16$. The whole difference between the two comes from the orange path 1.

這個縮小現象先記下來。序列一長，乘上很多個 $0.4$，最早幾步收到的梯度會接近 $0$。KP04 會專門討論它。

Keep this shrinking in mind. With a long sequence, many factors of $0.4$ are multiplied, and the earliest steps receive a gradient close to $0$. KP04 discusses this in detail.

常見錯誤：從 $t=1$ 開始往後算 $\delta_t$。這時 $\delta_2$ 還不知道，第 $1$ 步根本算不出來。前向是 $1\to T$，反向一定是 $T\to1$。

Common error: computing $\delta_t$ starting from $t=1$. Then $\delta_2$ is not yet known, and step $1$ cannot be computed at all. The forward pass runs $1\to T$; the backward pass must run $T\to1$.

### 本單元重點整理（KP02 summary）

**一句話重點**：BPTT 把 RNN 展開成 $T$ 層的網路，以 $\mathcal{L}=\sum_t\mathcal{L}_t$ 為目標；從 $t=T$ 倒算時，每一步的 $\delta_t$ 是兩路之和：自己輸出傳回的梯度，加上 $\delta_{t+1}$ 經 $\tanh$ 導數與 $\mathbf{W}$ 傳回的梯度。

**One-sentence summary**: BPTT unrolls the RNN into a $T$-layer network with the goal $\mathcal{L}=\sum_t\mathcal{L}_t$; working back from $t=T$, the state gradient $\delta_t$ of every step equals "the gradient sent back by its own output" plus "the gradient sent back by the next step's $\delta_{t+1}$ through the $\tanh$ derivative and $\mathbf{W}$".

**白話說明**：前向時 $\mathbf{h}_t$ 同時餵給輸出和下一步。所以反向時，它從這兩處收回責任，加起來就是 $\delta_t$。最後一步沒有下一步，是倒算的起點。共用的參數每一步都出力，所以它的總梯度是各步的和。

**Plain explanation**: in the forward pass $\mathbf{h}_t$ feeds both the output and the next step, so in the backward pass it receives blame from both places, and the sum is $\delta_t$. The last step has no next step and is where the backward computation starts. A shared parameter works at every step, so its total gradient is the sum over the steps.

**正式定義**：$\mathcal{L}=\sum_{t=1}^{T}\mathcal{L}_t$；$\frac{\partial\mathcal{L}}{\partial\mathbf{y}_t}=\frac{\partial\mathcal{L}_t}{\partial\mathbf{y}_t}$；$\delta_T=\mathbf{V}^{\top}\frac{\partial\mathcal{L}_T}{\partial\mathbf{y}_T}$；當 $t<T$，$\delta_t=\mathbf{V}^{\top}\frac{\partial\mathcal{L}}{\partial\mathbf{y}_t}+\mathbf{W}^{\top}\operatorname{diag}(1-\mathbf{h}_{t+1}^2)\,\delta_{t+1}$。

**Formal definition**: $\mathcal{L}=\sum_{t=1}^{T}\mathcal{L}_t$; $\frac{\partial\mathcal{L}}{\partial\mathbf{y}_t}=\frac{\partial\mathcal{L}_t}{\partial\mathbf{y}_t}$; $\delta_T=\mathbf{V}^{\top}\frac{\partial\mathcal{L}_T}{\partial\mathbf{y}_T}$; for $t<T$, $\delta_t=\mathbf{V}^{\top}\frac{\partial\mathcal{L}}{\partial\mathbf{y}_t}+\mathbf{W}^{\top}\operatorname{diag}(1-\mathbf{h}_{t+1}^2)\,\delta_{t+1}$.

**例子**：$\mathcal{L}=0.50+0.30+0.10=0.90$；$W$ 的總梯度 $0.05+0.15+0.20=0.40$。$V=1.0$、$W=0.5$、$1-h^2=0.8$、輸出梯度皆 $1.0$：$\delta_3=1.00$、$\delta_2=1.40$、$\delta_1=1.56$；只有最後一步有損失時是 $1.00$、$0.40$、$0.16$。

**Example**: $\mathcal{L}=0.50+0.30+0.10=0.90$; the total gradient of $W$ is $0.05+0.15+0.20=0.40$. With $V=1.0$, $W=0.5$, $1-h^2=0.8$ and all output gradients $1.0$: $\delta_3=1.00$, $\delta_2=1.40$, $\delta_1=1.56$; with a loss at the last step only they are $1.00$, $0.40$, $0.16$.

**常見迷思**：

- 「BPTT 是一種新的微分方法。」錯，它就是第2週的反向傳播，用在展開圖上。
- 「每一步有自己的 $\mathbf{W}_t$。」錯，只有一個 $\mathbf{W}$，總梯度是各步相加。
- 「$\delta_t$ 從 $t=1$ 往後算。」錯，它要用到 $\delta_{t+1}$，必須從 $t=T$ 往回算。
- 「$\delta_t$ 只看自己的輸出。」錯，非最後一步還要加上未來傳回的那一路。

**Common misconceptions**:

- "BPTT is a new method of differentiation." Wrong: it is the backpropagation of Week 2, applied to the unrolled graph.
- "Every step has its own $\mathbf{W}_t$." Wrong: there is only one $\mathbf{W}$, and its total gradient adds up the steps.
- "$\delta_t$ is computed from $t=1$ onward." Wrong: it needs $\delta_{t+1}$, so it must be computed back from $t=T$.
- "$\delta_t$ depends only on its own output." Wrong: every step except the last also adds the path coming back from the future.

**圖**：本單元有三張圖。

- `lec08-kp02-loss-sum`：各步損失加成總損失，共用 $W$ 的各步梯度加成總梯度。
- `lec08-kp02-two-paths`：三步展開圖上的兩條反向路徑與 $\delta_t$ 的數值。
- `lec08-kp02-delta-parts`：$\delta_t$ 拆成兩路，對照「每步有輸出」與「只有最後一步有輸出」。

**Figures** — this unit has three figures.

- `lec08-kp02-loss-sum`: the step losses add up to the total loss, and the step gradients of the shared $W$ add up to its total gradient.
- `lec08-kp02-two-paths`: the two backward paths on the unrolled three-step graph, with the values of $\delta_t$.
- `lec08-kp02-delta-parts`: $\delta_t$ split into its two paths, comparing "an output at every step" with "an output at the last step only".

### 小練習（Practice）

1. 展開 $T=3$ 步，損失是 $\mathcal{L}_1=0.40$、$\mathcal{L}_2=0.25$、$\mathcal{L}_3=0.15$，各步對 $W$ 的梯度貢獻是 $0.10$、$0.20$、$0.30$。求總損失與 $W$ 的總梯度。提示：兩個都是相加。
2. 純量 RNN，$V=2.0$、$W=0.5$，每步 $1-h_{t+1}^2=0.6$，三步的輸出梯度都是 $1.0$。依序算 $\delta_3$、$\delta_2$、$\delta_1$。提示：第二路每步乘 $0.5\times0.6$。
3. 來源互動展示的設定：$V=1.0$、$W=0.5$、狀態都是 $h=0$，只有最後一步有輸出梯度 $1.0$。算 $\delta_3$、$\delta_2$、$\delta_1$，並說明為什麼每往回一步剛好乘 $0.5$。提示：$h=0$ 時 $\tanh$ 的導數是多少？

1. Unroll $T=3$ steps with losses $\mathcal{L}_1=0.40$, $\mathcal{L}_2=0.25$, $\mathcal{L}_3=0.15$ and step contributions $0.10$, $0.20$, $0.30$ to the gradient of $W$. Find the total loss and the total gradient of $W$. Hint: both are sums.
2. A scalar RNN has $V=2.0$, $W=0.5$, $1-h_{t+1}^2=0.6$ at every step, and output gradient $1.0$ at all three steps. Compute $\delta_3$, $\delta_2$, $\delta_1$ in order. Hint: path 2 multiplies by $0.5\times0.6$ per step.
3. The setting of the source's interactive demo: $V=1.0$, $W=0.5$, all states $h=0$, and an output gradient $1.0$ at the last step only. Compute $\delta_3$, $\delta_2$, $\delta_1$ and explain why every step back multiplies by exactly $0.5$. Hint: what is the derivative of $\tanh$ when $h=0$?

**第 1 題解答** `[補充]`：總損失 $0.40+0.25+0.15=0.80$；總梯度 $0.10+0.20+0.30=0.60$。

**Solution to question 1** `[Added]`: the total loss is $0.40+0.25+0.15=0.80$; the total gradient is $0.10+0.20+0.30=0.60$.

**第 2 題解答** `[補充]`：每步乘數 $0.5\times0.6=0.3$。$\delta_3=1.0\times2.0=2.0$；$\delta_2=2.0+2.0\times0.3=2.6$；$\delta_1=2.0+2.6\times0.3=2.0+0.78=2.78$。

**Solution to question 2** `[Added]`: the factor per step is $0.5\times0.6=0.3$. $\delta_3=1.0\times2.0=2.0$; $\delta_2=2.0+2.0\times0.3=2.6$; $\delta_1=2.0+2.6\times0.3=2.0+0.78=2.78$.

**第 3 題解答** `[補充]`：$h=0$ 時 $1-h^2=1$，所以每步乘數就是 $W=0.5$。前兩步沒有輸出梯度，只剩第二路：$\delta_3=1.0$、$\delta_2=0.5$、$\delta_1=0.25$。

**Solution to question 3** `[Added]`: when $h=0$, $1-h^2=1$, so the factor per step is just $W=0.5$. The first two steps have no output gradient, so only path 2 remains: $\delta_3=1.0$, $\delta_2=0.5$, $\delta_1=0.25$.

### 小測驗（Quiz，來源原題）

來源的兩題自我檢測保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The two self-check questions of the source are kept with all options; the source page does not mark the answers directly, and the explanations below follow the source text.

1. 在 BPTT 演算法中，為什麼總損失 $\mathcal{L}$ 對權重 $\mathbf{W}$ 的梯度需要對所有時間步 $t$ 進行累加（summation）？
   - (A) 因為 $\mathbf{W}$ 是在所有時間步共享的同一個參數矩陣，根據多元微積分鏈鎖律必須累加每個時間步的貢獻
   - (B) 因為每個時間步的 $\mathbf{W}$ 都不一樣
   - (C) 為了防止梯度爆炸
   - (D) 這是高斯分布的要求
2. 為什麼計算時間步 $t$ 的隱藏狀態梯度 $\delta_t$ 時，公式包含兩項？
   - (A) 一項來自輸入 $\mathbf{x}_t$，一項來自輸出 $\mathbf{y}_t$
   - (B) 一項來自當前輸出 $\mathbf{y}_t$ 的影響，另一項來自下一時間步隱藏狀態 $\mathbf{h}_{t+1}$ 的反向回傳影響
   - (C) 一項是偏置 $\mathbf{b}_i$，一項是偏置 $\mathbf{b}_y$
   - (D) 為了計算正則化項

1. In the BPTT algorithm, why must the gradient of the total loss $\mathcal{L}$ with respect to the weight $\mathbf{W}$ be summed over all time steps $t$?
   - (A) Because $\mathbf{W}$ is one parameter matrix shared by all time steps, and the multivariable chain rule requires adding the contribution of every step
   - (B) Because $\mathbf{W}$ is different at every time step
   - (C) To prevent exploding gradients
   - (D) It is required by the Gaussian distribution
2. Why does the formula for the hidden-state gradient $\delta_t$ at step $t$ contain two terms?
   - (A) One comes from the input $\mathbf{x}_t$, the other from the output $\mathbf{y}_t$
   - (B) One comes from the effect on the current output $\mathbf{y}_t$, the other is passed back from the next hidden state $\mathbf{h}_{t+1}$
   - (C) One is the bias $\mathbf{b}_i$, the other the bias $\mathbf{b}_y$
   - (D) To compute a regularization term

**解析** `[補充]`：第 1 題選 (A)：第 1 節說同一個 $\mathbf{W}$ 每一步都出現，所以各步貢獻要相加；(B) 正好說反了；(C) 相加不會防止梯度爆炸（見 KP04）；(D) 與題目無關。第 2 題選 (B)：第 3 節說前向時 $\mathbf{h}_t$ 走向 $\mathbf{y}_t$ 與 $\mathbf{h}_{t+1}$ 兩條路；$\mathbf{x}_t$ 和兩個偏置都不在 $\delta_t$ 的公式裡。

**Explanations** `[Added]`: question 1 is A: Section 1 says the same $\mathbf{W}$ appears at every step, so the step contributions are added; B says the opposite; C is wrong, since adding does not prevent exploding gradients (see KP04); D is irrelevant. Question 2 is B: Section 3 says that in the forward pass $\mathbf{h}_t$ goes into both $\mathbf{y}_t$ and $\mathbf{h}_{t+1}$; neither $\mathbf{x}_t$ nor the two biases appear in the formula for $\delta_t$.

### 留給下一單元的問題（The question for the next unit）

現在每一步都有了 $\delta_1,\delta_2,\dots,\delta_T$。但我們真正要改的是共用的 $\mathbf{U}$、$\mathbf{W}$、$\mathbf{V}$ 和兩個偏置。要怎麼把這些 $\delta_t$ 換成參數的梯度，再一步一步加總？下一單元（KP03）接著用本節的三步例子回答。

Now every step has its $\delta_1,\delta_2,\dots,\delta_T$. But what we really want to change are the shared $\mathbf{U}$, $\mathbf{W}$, $\mathbf{V}$ and the two biases. How do we turn these $\delta_t$ into gradients of the parameters and add them up over the steps? The next unit (KP03) answers this with the three-step example of this note.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第8章「循環神經網絡與長短期記憶網絡」§2.3、§2.3.1–2.3.2 與其教授祕笈（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch08-rnn-lstm/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §2.3 and §2.3.1–2.3.2 of Chapter 8, "Recurrent neural networks and long short-term memory networks", of CCC-LNS IDL_115 and its professor's notes (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch08-rnn-lstm/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源寫 $\frac{\partial\mathbf{h}_{t+1}}{\partial\mathbf{h}_t}=\mathbf{W}\operatorname{diag}(1-\mathbf{h}_{t+1}^2)$，並寫 $\delta_t=\frac{\partial\mathcal{L}}{\partial\mathbf{y}_t}\mathbf{V}+\delta_{t+1}\mathbf{W}\operatorname{diag}(1-\mathbf{h}_{t+1}^2)$。向量情形下矩陣順序不對。正確的**雅可比矩陣**（把每個分量對每個分量的導數排成的矩陣）是 $\operatorname{diag}(1-\mathbf{h}_{t+1}^2)\,\mathbf{W}$。本講義改用行向量寫法 $\delta_t=\mathbf{V}^{\top}\frac{\partial\mathcal{L}}{\partial\mathbf{y}_t}+\mathbf{W}^{\top}\operatorname{diag}(1-\mathbf{h}_{t+1}^2)\,\delta_{t+1}$。程式用 $p=2$ 的隨機例子（種子 $0$）以差分驗證：本講義的式子與差分結果 $(1.7749,\ 0.1196)$ 一致；照來源的順序算得 $(1.3318,\ -0.0163)$，差 $0.4431$。純量情形兩者相同，所以本講義的數字例不受影響。
- 來源說網路「具有 $T$ 階馬可夫性質」。本講義照來源陳述；補充說明這其實是訓練時只展開 $T$ 步的做法，窗口外的關聯學不到 `[補充]`。
- 來源教授祕笈的特例句有缺字（「前面時間步的 $\mathcal{L}_t=0$（$t$」後面斷掉），本講義依上下文補成「$t<T$ 的各步損失為 $0$」。
- 來源用粗斜體 `\boldsymbol` 表示向量；本課程網站不支援，改用 `\mathbf`；$\delta_t$ 不加粗，在符號表說明它是 $p$ 維向量。
- 主例子 $V=1.0$、$W=0.5$、$1-h^2=0.8$、輸出梯度 $1.0$ 取自來源教授祕笈，也是本課程舊版研究生講義 KP02 的例子；本講義補上 $h_{t+1}=\pm0.4472$ 與均方誤差的解讀 `[補充]`。
- 本講義補上：只有最後一步有損失的對照（$1.00$、$0.40$、$0.16$）、softmax 交叉熵的梯度提示、形狀檢查、三題小練習與解答。第 3 題取自來源互動展示的理論基準（$W=0.5$、$V=1$、$h=0$ 時每步乘 $0.5$）。
- 取捨：來源的兩題自我檢測保留原題；來源的互動滑桿與 AI 生圖提示無法放進講義，改用 `lec08-kp02-loss-sum`、`lec08-kp02-two-paths`、`lec08-kp02-delta-parts` 三張圖呈現。

- The source writes $\frac{\partial\mathbf{h}_{t+1}}{\partial\mathbf{h}_t}=\mathbf{W}\operatorname{diag}(1-\mathbf{h}_{t+1}^2)$ and $\delta_t=\frac{\partial\mathcal{L}}{\partial\mathbf{y}_t}\mathbf{V}+\delta_{t+1}\mathbf{W}\operatorname{diag}(1-\mathbf{h}_{t+1}^2)$. For vectors the matrix order is wrong: the correct Jacobian is $\operatorname{diag}(1-\mathbf{h}_{t+1}^2)\,\mathbf{W}$. This note uses the column-vector form $\delta_t=\mathbf{V}^{\top}\frac{\partial\mathcal{L}}{\partial\mathbf{y}_t}+\mathbf{W}^{\top}\operatorname{diag}(1-\mathbf{h}_{t+1}^2)\,\delta_{t+1}$. The program checks it with finite differences on a random example with $p=2$ (seed $0$): this note's formula matches the finite-difference result $(1.7749,\ 0.1196)$; the source's order gives $(1.3318,\ -0.0163)$, off by $0.4431$. In the scalar case both agree, so the numerical examples of this note are not affected.
- The source says the network "has the $T$-th order Markov property". This note keeps the source's statement and adds that it really means unrolling only $T$ steps in training, so links outside the window are not learned `[Added]`.
- A sentence in the source's professor's notes about the special case is cut off ("the earlier steps have $\mathcal{L}_t=0$ ($t$" breaks off); this note completes it from context as "the losses of the steps $t<T$ are $0$".
- The source writes vectors with bold italic `\boldsymbol`; the course website does not support it, so this note uses `\mathbf`; $\delta_t$ is not bold, and the symbol table states that it is a $p$-dimensional vector.
- The running example $V=1.0$, $W=0.5$, $1-h^2=0.8$ and output gradient $1.0$ comes from the source's professor's notes and is also the example of the course's earlier graduate-level KP02 note; this note adds the reading $h_{t+1}=\pm0.4472$ and the mean-squared-error interpretation `[Added]`.
- This note adds: the comparison with a loss at the last step only ($1.00$, $0.40$, $0.16$), the hint on the softmax cross-entropy gradient, the shape check, and three practice questions with solutions. Question 3 uses the theoretical baseline of the source's interactive demo (factor $0.5$ per step when $W=0.5$, $V=1$, $h=0$).
- Choices: the two self-check questions of the source are kept as asked; the source's interactive sliders and AI image prompts cannot go into a note, so the three figures `lec08-kp02-loss-sum`, `lec08-kp02-two-paths` and `lec08-kp02-delta-parts` show the same content.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 隨時間反向傳播 | backpropagation through time (BPTT) |
| 時間展開 | unrolling in time |
| $T$ 階馬可夫性質 | $T$-th order Markov property |
| 總損失 | total loss |
| 鏈鎖律 | chain rule |
| 反向傳播 | backpropagation |
| 均方誤差 | mean squared error (MSE) |
| 交叉熵 | cross-entropy |
| 隱藏狀態梯度 | hidden-state gradient |
| 雅可比矩陣 | Jacobian matrix |
| 起點（邊界條件） | boundary condition |
| 行向量 | column vector |
| 轉置 | transpose |
| 對角矩陣 | diagonal matrix |
| 參數共享 | parameter sharing |
| 循環神經網路 | recurrent neural network (RNN) |
| 前饋網路 | feedforward network |
| 梯度消失（KP04） | vanishing gradient |
| 本課程的網頁教材 | CCC-LNS |
