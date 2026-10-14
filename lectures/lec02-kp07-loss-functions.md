# 第2週 KP07：損失函數把「錯多少」變成一個數（A loss function turns "how wrong" into one number）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)
>
> 週次：第2週（2026-09-17）｜主題：神經網路基礎：感知器與多層感知器
>
> Week 2 (2026-09-17) | Topic: Neural network fundamentals: perceptron and multilayer perceptron

本講義是第2章的第七個知識點（KP07），承接 KP06 留下的問題：網路輸出了一個結果，要拿它和正確答案比，並把「差多少」變成一個數，才能知道往哪個方向調權重。這個數就是損失（loss）。這一節介紹六種常用損失：回歸的 MSE、MAE、Huber，分類的交叉熵、KL 散度與 Hinge，並用數字比較它們。先備知識是 KP06 與對數。第3章 KP01 已把損失當成「最佳化要最小化的分數」；這一節講損失本身怎麼定義。內容依第2章主文重新講解，來源讀取日期為 2026-10-14；新增的釐清標為 `[補充]`。

This is the seventh knowledge point (KP07) of Chapter 2. It picks up the question KP06 left open: the network gives a result, and we must compare it with the right answer and turn "how far off" into one number before we know which way to adjust the weights. That number is the loss. This note introduces six common losses: MSE, MAE and Huber for regression, and cross-entropy, KL divergence and hinge for classification, and compares them with numbers. It assumes KP06 and logarithms. Chapter 3 KP01 already treated the loss as "the score the optimizer minimizes"; this note covers how the loss itself is defined. It re-explains the main text of Chapter 2, accessed on 2026-10-14; clarifications that are new are marked `[Added]`.

[下載本單元投影片](../slides/lec02-kp07-loss-functions.jbd.json)

## KP07 損失函數把「錯多少」變成一個數（A loss function turns "how wrong" into one number）

**建議時間 / Time**：15 分鐘　**對應實作 / Related practice**：用同一組殘差算 MSE、MAE、Huber，並用 softmax 輸出算交叉熵。

### 學習目標（Learning objectives）

讀完之後，你能寫出六種損失的公式並說出各自適用的場合；能用一組含離群值的殘差說明 MSE、MAE、Huber 的差別；能用 softmax 輸出手算交叉熵與 KL 散度；能說明 hinge 為什麼連「分對但太近」也要罰；能為回歸與分類各選一種損失。

After reading, you can write the six losses and say where each applies; use a set of residuals with an outlier to show how MSE, MAE and Huber differ; compute the cross-entropy and KL divergence by hand from a softmax output; explain why hinge punishes even "right but too close"; and choose a loss for regression and one for classification.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $f_o(\mathbf{x}_i)$ | 網路對第 $i$ 筆輸入的輸出 | the network output for the $i$-th input |
| $y_i$ | 第 $i$ 筆的目標（回歸為實數，二分類 hinge 用 $\pm1$） | the target of the $i$-th sample (a real number for regression, $\pm1$ for binary hinge) |
| $e=f_o-y$ | 殘差 | the residual |
| $b,\ c$ | 批次大小、類別數 | batch size and number of classes |
| $\delta$ | Huber 的切換門檻 | the switching threshold of Huber |
| $m$ | Hinge 的安全邊界 | the margin of hinge |

### 1. 回歸：MSE、MAE、Huber（Regression: MSE, MAE and Huber）

| 損失 | 公式（對批次求和） | 特點 |
|---|---|---|
| MSE | $\sum_{i=1}^{b}e_i^2$ | 處處可導；平方把離群值放大 |
| MAE | $\sum_{i=1}^{b}\lvert e_i\rvert$ | 對離群值穩健；零點不可導 |
| Huber | $\lvert e\rvert\le\delta$ 時 $\tfrac12e^2$，否則 $\delta(\lvert e\rvert-\tfrac12\delta)$ | 兩者折衷，要調 $\delta$ |

四筆資料的殘差是 $(1,1,1,10)$，最後一筆是離群值。取 Huber 的 $\delta=1$〔程式計算〕：

Four data points have residuals $(1,1,1,10)$, and the last is an outlier. Take Huber with $\delta=1$ (computed by program):

| 損失 | 四筆的值 | 總和 | 離群點占比 |
|---|---|---|---|
| MSE | $1,\,1,\,1,\,100$ | $103$ | $100/103=97\%$ |
| MAE | $1,\,1,\,1,\,10$ | $13$ | $10/13=77\%$ |
| Huber | $0.5,\,0.5,\,0.5,\,9.5$ | $11$ | $9.5/11=86\%$ |

Huber 的最後一筆是 $\delta(|e|-\tfrac12\delta)=1\times(10-0.5)=9.5$。MSE 幾乎只在乎那一個離群點，MAE 最不受它左右，Huber 在中間：這就是「MSE 對離群值敏感」的具體意思。

The last Huber term is $\delta(|e|-\tfrac12\delta)=1\times(10-0.5)=9.5$. MSE cares almost only about that one outlier, MAE is least swayed by it, and Huber sits in between: this is what "MSE is sensitive to outliers" means in numbers.

![MSE、MAE 與 Huber（$\delta=1$）隨殘差變化](../assets/figures/lec02-kp07-loss-regression.png)

**圖說 `lec02-kp07-loss-regression`**：先看紅色的 MSE，它是拋物線，殘差越大升得越快；再看藍色 MAE，兩條直線；最後看綠色虛線 Huber，中間像拋物線、兩側像直線，而且比 MAE 低。 / **Caption `lec02-kp07-loss-regression`:** Look first at the MSE, a parabola that rises faster the larger the residual; then at the MAE, two straight lines; finally at the dashed Huber, parabolic in the middle and linear on both sides, and below the MAE.

### 2. 分類：交叉熵與 KL 散度（Classification: cross-entropy and KL divergence）

交叉熵是 $-\sum_i\sum_{l=1}^{c}(y_i)_l\log f_o(\mathbf{x}_i)_l$：「給正確類別的機率，取負對數」。它要求輸出像機率，所以輸出層必須接 softmax（多類）或 sigmoid（二元）。沿用 KP06 的 softmax 輸出 $\mathbf{p}=(0.665,\,0.245,\,0.090)$〔程式計算〕：

Cross-entropy is $-\sum_i\sum_{l=1}^{c}(y_i)_l\log f_o(\mathbf{x}_i)_l$: "the negative log of the probability given to the correct class". It needs the output to look like probabilities, so the output layer must be softmax (multi-class) or sigmoid (binary). Reuse the softmax output of KP06, $\mathbf{p}=(0.665,\,0.245,\,0.090)$ (computed by program):

| 正確類別 | 預測機率 $p$ | 交叉熵 $-\ln p$ |
|---|---|---|
| 第 1 類 | $0.665$ | $0.408$ |
| 第 2 類 | $0.245$ | $1.408$ |
| 第 3 類 | $0.090$ | $2.408$ |

預測給正確答案的機率越小，損失越大。KL 散度 $\sum_l(y_i)_l\log\frac{(y_i)_l}{f(\mathbf{x}_i)_l}$ 用在目標本身是機率分布（軟標籤，例如知識蒸餾）的時候。目標 $\mathbf{y}=(0.7,0.2,0.1)$ 對上面的 $\mathbf{p}$：$0.7\ln\frac{0.7}{0.665}+0.2\ln\frac{0.2}{0.245}+0.1\ln\frac{0.1}{0.090}=0.0058$〔程式計算〕，很接近 0，因為兩個分布很像。若目標是 one-hot $(1,0,0)$，KL 只剩 $\ln\frac{1}{0.665}=0.408$，與交叉熵相同。

The less probability the prediction gives to the correct answer, the larger the loss. The KL divergence $\sum_l(y_i)_l\log\frac{(y_i)_l}{f(\mathbf{x}_i)_l}$ is used when the target itself is a probability distribution (a soft label, as in knowledge distillation). For the target $\mathbf{y}=(0.7,0.2,0.1)$ against the $\mathbf{p}$ above: $0.7\ln\frac{0.7}{0.665}+0.2\ln\frac{0.2}{0.245}+0.1\ln\frac{0.1}{0.090}=0.0058$ (computed by program), very close to 0 because the two distributions are alike. With a one-hot target $(1,0,0)$ the KL reduces to $\ln\frac{1}{0.665}=0.408$, the same as the cross-entropy.

### 3. Hinge：分對了但離邊界太近，也要罰（Hinge: punishing even "right but too close"）

Hinge 用在二分類，標籤 $y_i\in\{-1,+1\}$：損失 $\max\big(0,\,m-y_if_o(\mathbf{x}_i)\big)$。取 $m=1$。$y=+1$ 時：$f_o=1.5$ 損失 $0$；$f_o=0.3$ 損失 $1-0.3=0.7$；$f_o=-0.5$ 損失 $1-(-0.5)=1.5$。第二個點分對了（$0.3>0$），卻因離邊界不到 $m$ 而被罰 $0.7$。它和 KP02 那種只問對錯的 0-1 損失不同，是連續的。

Hinge is used in binary classification with labels $y_i\in\{-1,+1\}$: the loss is $\max\big(0,\,m-y_if_o(\mathbf{x}_i)\big)$. Take $m=1$. For $y=+1$: $f_o=1.5$ gives loss $0$; $f_o=0.3$ gives $1-0.3=0.7$; $f_o=-0.5$ gives $1-(-0.5)=1.5$. The second point is classified right ($0.3>0$) yet is punished by $0.7$ because it lies within $m$ of the boundary. Unlike the 0-1 loss of KP02 that asks only right or wrong, it is continuous.

![Hinge 損失與 0-1 損失隨 $y\,f_o$ 變化](../assets/figures/lec02-kp07-loss-hinge.png)

**圖說 `lec02-kp07-loss-hinge`**：先看灰色點線，0-1 損失只在分錯（$yf<0$）時為 1；再看橘色 hinge，在 $yf<1$ 時就開始罰，三個標出的點損失分別是 0、0.7、1.5。 / **Caption `lec02-kp07-loss-hinge`:** Look first at the grey dotted 0-1 loss, which is 1 only when the point is wrong ($yf<0$); then at the orange hinge, which starts punishing already when $yf<1$, with losses 0, 0.7 and 1.5 at the three marked points.

### 本單元重點整理（KP07 summary）

**一句話重點**：損失函數把「網路錯了多少」變成一個可以最小化的數；回歸常用 MSE、MAE、Huber，分類幾乎一律用交叉熵（輸出層配 softmax 或 sigmoid）。

**One-sentence summary**: A loss function turns "how wrong the network is" into one number to minimize; regression commonly uses MSE, MAE or Huber, and classification almost always uses cross-entropy (with softmax or sigmoid in the output layer).

**白話說明**：像練射箭：損失是箭離靶心的距離。MSE 把距離平方，一支飛很遠的箭被重罰；MAE 只用距離本身，較寬容；Huber 在近處像 MSE、遠處像 MAE。分類量的不是距離，而是「預測機率」與「正確答案」差多少。

**Plain explanation**: Like archery: the loss is the distance of the arrow from the bullseye. MSE squares it, so a wild arrow is heavily punished; MAE uses the distance itself and is more tolerant; Huber acts like MSE up close and like MAE far away. Classification measures not a distance but how far the predicted probability is from the correct answer.

**正式定義**：MSE $\sum e^2$；MAE $\sum|e|$；Huber 在 $|e|\le\delta$ 用 $\tfrac12e^2$、否則 $\delta(|e|-\tfrac12\delta)$；交叉熵 $-\sum\sum y_l\log f_l$；KL $\sum\sum y_l\log(y_l/f_l)$；Hinge $\max(0,m-yf_o)$。

**Formal definition**: MSE $\sum e^2$; MAE $\sum|e|$; Huber uses $\tfrac12e^2$ when $|e|\le\delta$ and $\delta(|e|-\tfrac12\delta)$ otherwise; cross-entropy $-\sum\sum y_l\log f_l$; KL $\sum\sum y_l\log(y_l/f_l)$; hinge $\max(0,m-yf_o)$.

**例子**：殘差 $(1,1,1,10)$：MSE 總和 103（離群點占 97%）、MAE 13（77%）、Huber 11（86%）；softmax $(0.665,0.245,0.090)$ 的交叉熵為 $0.408/1.408/2.408$；hinge 在 $yf_o=1.5,0.3,-0.5$ 為 $0,0.7,1.5$。

**Example**: Residuals $(1,1,1,10)$: MSE sum 103 (outlier 97%), MAE 13 (77%), Huber 11 (86%); the cross-entropy of the softmax $(0.665,0.245,0.090)$ is $0.408/1.408/2.408$; hinge at $yf_o=1.5,0.3,-0.5$ is $0,0.7,1.5$.

**常見迷思**：

- 「損失越小，網路一定越好」——不對。損失只在訓練資料上量到，太小可能只是把訓練資料背起來（KP04 的泛化問題）。
- 「不同損失的數值可以直接比大小」——不能：MSE 的 $103$ 與 MAE 的 $13$ 沒有誰比較大的意義，它們是不同的尺。

**Common misconceptions**:

- "The smaller the loss, the better the network." No: the loss is measured on the training data only, and a very small value may just mean the training data were memorized (the generalization problem of KP04).
- "Values of different losses can be compared directly." They cannot: MSE's $103$ and MAE's $13$ say nothing meaningful about which is larger, because they are different rulers.

**圖**：本單元兩張圖，`lec02-kp07-loss-regression`（三種回歸損失，功能 1、2）、`lec02-kp07-loss-hinge`（hinge 與 0-1 損失，功能 1、2），以 `assets/figures/src/lec02-kp0507-figures.py` 繪製。

**Figures**: Two figures in this unit, `lec02-kp07-loss-regression` (three regression losses; functions 1, 2) and `lec02-kp07-loss-hinge` (hinge and 0-1 loss; functions 1, 2), drawn by `assets/figures/src/lec02-kp0507-figures.py`.

**想深入**（選讀）：Huber 在 $|e|=\delta$ 處，二次段與線性段的值與斜率都相同，所以整條曲線是平滑的：二次段 $\tfrac12\delta^2$、斜率 $\delta$；線性段 $\delta(\delta-\tfrac12\delta)=\tfrac12\delta^2$、斜率 $\delta$。另外，二元交叉熵的負值正是第 KP08 要用的對數似然。

**Going deeper** (optional): At $|e|=\delta$ the quadratic piece and the linear piece of Huber have the same value and slope, so the whole curve is smooth: the quadratic piece gives $\tfrac12\delta^2$ with slope $\delta$, and the linear piece $\delta(\delta-\tfrac12\delta)=\tfrac12\delta^2$ with slope $\delta$. Also, the negative of the binary cross-entropy is exactly the log-likelihood that KP08 will use.

**小練習**：

1. 殘差 $(2,-1,0.5,6)$，算 MSE、MAE 與 Huber（$\delta=1$）的總和。（提示：Huber 要先判斷每個殘差在二次區還是線性區。）
2. 3 類的 softmax 輸入 $(0,0,0)$，正確類別任選，交叉熵是多少？（提示：三個機率各是多少？）
3. Hinge（$m=1$），$y=-1$、$f_o=0.8$，損失是多少？它分對了嗎？（提示：先算 $yf_o$。）

**Practice**:

1. For residuals $(2,-1,0.5,6)$ compute the sums of MSE, MAE and Huber ($\delta=1$). (Hint: for Huber, first decide whether each residual lies in the quadratic or the linear zone.)
2. For a 3-class softmax with input $(0,0,0)$, what is the cross-entropy for any correct class? (Hint: what is each probability?)
3. For hinge with $m=1$, $y=-1$ and $f_o=0.8$, what is the loss, and is the point classified right? (Hint: compute $yf_o$ first.)

**解題示範**（第 1 題）：

1. MSE：$2^2+1^2+0.5^2+6^2=4+1+0.25+36=41.25$。
2. MAE：$2+1+0.5+6=9.5$。
3. Huber：$|2|>1$ 用線性 $1\times(2-0.5)=1.5$；$|-1|\le1$ 用二次 $\tfrac12\cdot1=0.5$；$0.5$ 用二次 $\tfrac12\cdot0.25=0.125$；$6$ 用線性 $1\times(6-0.5)=5.5$；總和 $1.5+0.5+0.125+5.5=7.625$。
4. 檢查：離群值 6 在 MSE 占 $36/41.25=87\%$，在 Huber 只占 $5.5/7.625=72\%$，與第 1 節的規律一致。

**Worked solution** (exercise 1):

1. MSE: $2^2+1^2+0.5^2+6^2=4+1+0.25+36=41.25$.
2. MAE: $2+1+0.5+6=9.5$.
3. Huber: $|2|>1$ uses the linear piece $1\times(2-0.5)=1.5$; $|-1|\le1$ uses the quadratic $\tfrac12\cdot1=0.5$; $0.5$ uses the quadratic $\tfrac12\cdot0.25=0.125$; $6$ uses the linear piece $1\times(6-0.5)=5.5$; the sum is $1.5+0.5+0.125+5.5=7.625$.
4. Check: the outlier 6 makes up $36/41.25=87\%$ of the MSE but only $5.5/7.625=72\%$ of the Huber, consistent with the pattern of section 1.

**自我檢核**（來源四題，不附答案）：

| 題 / Question | A | B | C | D | 提示 / Hint |
|---|---|---|---|---|---|
| 1. MSE 的主要缺點？ / What is the main drawback of MSE? | 零點不可導 | 平方把離群值放大，對離群值敏感 | 無法用於回歸 | 輸出不落在 $[0,1]$ | 看第 1 節的表。 / The table of section 1. |
| 2. Huber 的超參數 $\delta$ 作用？ / What does $\delta$ do? | 決定學習率 | 決定二次與線性區的切換門檻 | 決定類別數 | 決定 margin | 看公式。 / See the formula. |
| 3. 哪種損失在目標是軟標籤時特別有用？ / Which loss suits soft targets? | Hinge | MSE | KL divergence | MAE | 看第 2 節。 / Section 2. |
| 4. 使用交叉熵時，輸出層必須具備什麼條件？ / What must the output layer satisfy for cross-entropy? | 線性活化即可 | 接 softmax 或 sigmoid，使預測落在 $[0,1]$ 且和為 1 | 用 maxout | 必須有 bias | 對數裡的值要像機率。 / The value inside the log must look like a probability. |

**留給下一個知識點的問題**：Perceptron 的損失只數「錯幾個」，是離散的，不能用梯度學。如果改成連續的平方誤差，就能一步一步往下走。這個想法就是 ADALINE。

**Question for the next knowledge point**: The Perceptron's loss only counts "how many are wrong", which is discrete and cannot be learned by gradients. If it is replaced by a continuous squared error, we can walk downhill step by step. That idea is ADALINE.

### 來源更正與取捨（Corrections to the source）

- 來源說 hinge 在「$|f_o|\ge m$ 時損失為 0」。依公式 $\max(0,m-yf_o)$，正確條件是 $yf_o\ge m$（分對而且超過邊界）；$f_o=-1.5$、$y=+1$ 時 $|f_o|\ge m$ 但損失是 $2.5$。
- 來源把 KL 的 one-hot 情形說成「退化成交叉熵」，本講義補上算例（$0.408$ 兩者相同），數字由程式驗算（`[補充]`）。
- 來源沒有離群值的數字例；上面 $(1,1,1,10)$ 的比較為明示補充。

- The source says hinge is 0 "when $|f_o|\ge m$". By the formula $\max(0,m-yf_o)$ the correct condition is $yf_o\ge m$ (right and beyond the margin); with $f_o=-1.5$ and $y=+1$ we have $|f_o|\ge m$ but the loss is $2.5$.
- The source says KL "reduces to cross-entropy" for one-hot targets; this note adds a worked case (both $0.408$), with numbers checked by program (`[Added]`).
- The source gives no numeric outlier example; the $(1,1,1,10)$ comparison above is an explicit addition.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 損失函數 | loss function |
| 殘差 | residual |
| 離群值 | outlier |
| 均方誤差 | mean squared error (MSE) |
| 平均絕對誤差 | mean absolute error (MAE) |
| 交叉熵 | cross-entropy |
| KL 散度 | KL divergence |
| 軟標籤 | soft label |
| 安全邊界 | margin |
