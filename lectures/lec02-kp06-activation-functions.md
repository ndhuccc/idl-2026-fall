# 第2週 KP06：13 種活化函數分四族，隱藏層優先 ReLU 家族（Thirteen activation functions in four families: prefer the ReLU family in hidden layers）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)
>
> 週次：第2週（2026-09-17）｜主題：神經網路基礎：感知器與多層感知器
>
> Week 2 (2026-09-17) | Topic: Neural network fundamentals: perceptron and multilayer perceptron

本講義是第2章的第六個知識點（KP06），承接 KP05 留下的問題：階梯函數可以當非線性，但它在 0 處不可導、其他地方導數為 0，梯度學不到東西，那該選哪一種非線性？這一節把來源的 13 種活化函數分成四族，每種看三件事：公式、值域、導數。先備知識是 KP05 與導數。sigmoid 的斜率與梯度消失在第3章 KP06 有完整圖解，這裡只引用結論。內容依第2章主文重新講解，來源讀取日期為 2026-10-14；新增的釐清標為 `[補充]`。

This is the sixth knowledge point (KP06) of Chapter 2. It picks up the question KP05 left open: the step function can serve as the nonlinearity, but it is not differentiable at 0 and has derivative 0 elsewhere, so a gradient learns nothing from it; which nonlinearity should we choose? This note sorts the 13 activation functions of the source into four families and looks at three things for each: formula, range and derivative. It assumes KP05 and derivatives. The slope of the sigmoid and vanishing gradients are drawn in full in Chapter 3 KP06, so only the conclusion is cited here. It re-explains the main text of Chapter 2, accessed on 2026-10-14; clarifications that are new are marked `[Added]`.

[下載本單元投影片](../slides/lec02-kp06-activation-functions.jbd.json)

## KP06 13 種活化函數分四族，隱藏層優先 ReLU 家族（Thirteen activation functions in four families）

**建議時間 / Time**：15 分鐘　**對應實作 / Related practice**：手算 softmax，並比較 ReLU 家族在 $a=\pm2$ 的值與導數。

### 學習目標（Learning objectives）

讀完之後，你能把 13 種活化函數分成基礎、S 形、ReLU 家族與跨層四族；能說出 sigmoid、tanh、ReLU 的值域與導數；能手算 softmax；能說明 dying ReLU 以及 Leaky ReLU、PReLU、ELU 如何修補；能為隱藏層與輸出層各選一個。

After reading, you can sort the 13 activation functions into the basic, S-shaped, ReLU and cross-layer families; state the range and derivative of sigmoid, tanh and ReLU; compute a softmax by hand; explain dying ReLU and how Leaky ReLU, PReLU and ELU repair it; and pick one for the hidden layers and one for the output layer.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $a$ | 活化前的加權和（KP05 的 $a_i$） | the pre-activation sum ($a_i$ of KP05) |
| $\sigma(a)$ | 活化函數 | the activation function |
| $\sigma'(a)$ | 活化函數對 $a$ 的導數 | derivative of the activation with respect to $a$ |
| $\alpha$ | PReLU 與 ELU 的負側參數 | the negative-side parameter of PReLU and ELU |
| $\mathbf{a}=(a_1,\dots,a_m)$ | softmax 的整層輸入 | the whole-layer input of softmax |

### 1. 四族一張表（Four families in one table）

選哪一種活化，決定兩件事：輸出的範圍（能不能當機率）與導數的大小（學習時梯度能不能傳回去）。

The choice of activation decides two things: the range of the output (whether it can serve as a probability) and the size of the derivative (whether gradients can flow back during learning).

| 族 | 名稱 | 公式 $\sigma(a)$ | 值域 | 導數 |
|---|---|---|---|---|
| 基礎 | Linear | $a$ | $(-\infty,\infty)$ | $1$ |
| 基礎 | Binary step / Sign | $0$ 或 $1$／$-1$ 或 $+1$ | $\{0,1\}$／$\{-1,+1\}$ | $0$，在 $0$ 不可導 |
| 基礎 | Softmax（整層） | $e^{a_i}/\sum_je^{a_j}$ | $(0,1)$，和為 1 | $\sigma_i(\delta_{ij}-\sigma_j)$ |
| S 形 | Sigmoid | $1/(1+e^{-a})$ | $(0,1)$ | $\sigma(1-\sigma)$ |
| S 形 | tanh | $(e^a-e^{-a})/(e^a+e^{-a})$ | $(-1,1)$ | $1-\sigma^2$ |
| S 形 | Gaussian | $e^{-a^2}$ | $(0,1]$ | $-2a\,e^{-a^2}$ |
| ReLU 族 | Softplus | $\ln(1+e^a)$ | $(0,\infty)$ | $1/(1+e^{-a})$ |
| ReLU 族 | ReLU | $\max(0,a)$ | $[0,\infty)$ | $a>0$ 為 1，$a<0$ 為 0 |
| ReLU 族 | Leaky ReLU | $a$ 若 $a\ge0$，$0.01a$ 若 $a<0$ | $(-\infty,\infty)$ | $1$ 或 $0.01$ |
| ReLU 族 | PReLU | 同上，斜率 $\alpha$ 可學習 | $(-\infty,\infty)$ | $1$ 或 $\alpha$ |
| ReLU 族 | ELU | $a$ 若 $a\ge0$，$\alpha(e^a-1)$ 若 $a<0$ | $(-\alpha,\infty)$ | $1$ 或 $\alpha e^a$ |
| 跨層 | Maxout | 多個線性單元的最大值 | $(-\infty,\infty)$ | 取到最大者為 1，其餘為 0 |

Softmax 與 Maxout 要看整層（或多個單元）一起算，其餘逐點作用。Linear 放在隱藏層就是 KP05 的坍縮，只適合回歸的輸出層。

Softmax and Maxout look at a whole layer (or several units) together, while the others act point by point. Linear in a hidden layer is the collapse of KP05, so it suits only the output layer of a regression.

### 2. 基礎與 S 形：Softmax 與梯度消失（Basic and S-shaped: softmax and vanishing gradients）

**Softmax 算例。** 輸出層 3 顆神經元，加權和 $\mathbf{a}=(2,1,0)$。先取指數 $e^2=7.389$、$e^1=2.718$、$e^0=1$，總和 $11.107$；再除以總和得 $(0.665,\,0.245,\,0.090)$。三個數都在 $(0,1)$，加起來 $0.665+0.245+0.090=1$，所以輸出可以直接讀成「屬於三類的機率」〔程式計算〕。

**A softmax example.** The output layer has 3 neurons with sums $\mathbf{a}=(2,1,0)$. First take exponentials $e^2=7.389$, $e^1=2.718$, $e^0=1$, with sum $11.107$; then divide by the sum to get $(0.665,\,0.245,\,0.090)$. All three lie in $(0,1)$ and add up to $0.665+0.245+0.090=1$, so the output can be read directly as "probabilities of three classes" (computed by program).

**S 形與梯度消失。** Sigmoid 在 $a=0$ 輸出 $0.5$，導數 $0.5\times(1-0.5)=0.25$，這已是它導數的最大值；tanh 在 $a=0$ 導數是 $1$。學習時梯度要一層一層連乘導數往回傳，5 層 sigmoid 最好情況也只剩 $0.25^5\approx0.00098$，這就是梯度消失〔補充算例，數字由程式驗算；完整推導見第3章 KP06〕。

**S-shapes and vanishing gradients.** The sigmoid outputs $0.5$ at $a=0$ with derivative $0.5\times(1-0.5)=0.25$, already the maximum of its derivative; the tanh derivative at $a=0$ is $1$. During learning the gradient is multiplied by derivatives layer after layer on its way back, so five sigmoid layers in the best case leave only $0.25^5\approx0.00098$; this is the vanishing gradient (an added worked example with numbers checked by program; the full derivation is in Chapter 3 KP06).

### 3. ReLU 家族：導數不縮小，但要防 dying（The ReLU family: the derivative does not shrink, but beware dying）

ReLU 是 $\max(0,a)$：正側導數恆為 1，梯度可以原封不動傳回去，不像 S 形越傳越小；負值歸零，讓許多神經元同時「安靜」（稀疏）。但若某顆神經元的輸入總是負的，導數恆為 0，權重永遠不更新，這就是 **dying ReLU**。Leaky ReLU 在負側保留 $0.01$ 的小斜率；PReLU 把這個斜率變成可學習參數；ELU 與 softplus 把轉折處磨平。取 $a=-2$ 與 $a=2$〔程式計算〕：

ReLU is $\max(0,a)$: the derivative is exactly 1 on the positive side, so the gradient passes back unchanged, unlike S-shapes that shrink it; negative values are zeroed, so many neurons go quiet at once (sparsity). But if the input of a neuron is always negative, its derivative is always 0 and its weights never update; this is **dying ReLU**. Leaky ReLU keeps a small slope of $0.01$ on the negative side; PReLU turns that slope into a learnable parameter; ELU and softplus smooth the corner. At $a=-2$ and $a=2$ (computed by program):

| 函數 | $\sigma(-2)$ | $\sigma(2)$ | 導數於 $a=-2$ | 導數於 $a=2$ |
|---|---|---|---|---|
| ReLU | $0$ | $2$ | $0$ | $1$ |
| Leaky ReLU（0.01） | $-0.02$ | $2$ | $0.01$ | $1$ |
| ELU（$\alpha=1$） | $-0.865$ | $2$ | $0.135$ | $1$ |
| Softplus | $0.127$ | $2.127$ | $0.119$ | $0.881$ |

![ReLU、Leaky ReLU、ELU 與 softplus 的曲線](../assets/figures/lec02-kp06-act-relu-family.png)

**圖說 `lec02-kp06-act-relu-family`**：先看正的一側，四條線幾乎重疊（斜率 1）；再看負的一側，ReLU 是 0，Leaky 略低於 0，ELU 往下走到 $-1$ 附近，softplus 則維持在 0 以上。 / **Caption `lec02-kp06-act-relu-family`:** Look first at the positive side, where the four lines almost overlap (slope 1); then at the negative side: ReLU is 0, Leaky slightly below 0, ELU heads toward $-1$, and softplus stays above 0.

### 4. 怎麼選（How to choose）

隱藏層先試 ReLU 家族，怕 dying ReLU 就換 Leaky ReLU 或 PReLU；輸出層看任務：回歸用 Linear，二元分類用 Sigmoid，多類別互斥分類用 Softmax。表中的導數都是「在可導處」：ReLU、Leaky ReLU 與 PReLU 在 $a=0$ 不可導，實作上約定一個值（通常取 0 或 1）。若一個樣本可同時屬於多類，改用每個輸出各自的 sigmoid，而不是 softmax。

Try the ReLU family first in hidden layers, and switch to Leaky ReLU or PReLU if dying ReLU is a worry; choose the output layer by task: Linear for regression, Sigmoid for binary classification, Softmax for mutually exclusive multi-class classification. All derivatives in the table hold "where differentiable": ReLU, Leaky ReLU and PReLU are not differentiable at $a=0$, and in practice a value is fixed by convention (usually 0 or 1). If a sample can belong to several classes at once, use an individual sigmoid per output instead of softmax.

### 本單元重點整理（KP06 summary）

**一句話重點**：活化函數依值域與導數分四族；隱藏層優先用導數不縮小的 ReLU 家族，輸出層依任務選 Linear、Sigmoid 或 Softmax。

**One-sentence summary**: Activation functions fall into four families by range and derivative; prefer the ReLU family, whose derivative does not shrink, in hidden layers, and choose Linear, Sigmoid or Softmax for the output layer by task.

**白話說明**：每顆神經元門口有一個「閘門」把加權和轉成輸出。階梯是硬開關；S 形是慢慢開、開到底就不再增加的水龍頭；ReLU 是「負的全擋掉，正的原樣通過」。閘門的形狀決定訊號與學習用的回饋能不能順利通過。

**Plain explanation**: There is a "gate" at the door of every neuron that turns the weighted sum into an output. The step is a hard switch; S-shapes are a tap that opens slowly and stops increasing when fully open; ReLU "blocks everything negative and lets positives pass unchanged". The shape of the gate decides whether signals, and the feedback used for learning, can get through.

**正式定義**：見第 1 節的表；softmax $\sigma_i(\mathbf{a})=e^{a_i}/\sum_je^{a_j}$；ReLU $\max(0,a)$；Leaky ReLU 負側斜率 $0.01$。

**Formal definition**: See the table of section 1; softmax $\sigma_i(\mathbf{a})=e^{a_i}/\sum_je^{a_j}$; ReLU $\max(0,a)$; Leaky ReLU with negative slope $0.01$.

**例子**：softmax$(2,1,0)=(0.665,0.245,0.090)$；$a=-2$ 時 ReLU、Leaky、ELU 的導數是 $0$、$0.01$、$0.135$。

**Example**: softmax$(2,1,0)=(0.665,0.245,0.090)$; at $a=-2$ the derivatives of ReLU, Leaky and ELU are $0$, $0.01$ and $0.135$.

**常見迷思**：

- 「ReLU 處處可導，所以最安全」——不對，它在 $a=0$ 不可導，而且負側導數為 0，可能讓神經元死掉。
- 「sigmoid 與 softmax 是同一回事」——不是：sigmoid 對單一個數，輸出彼此獨立；softmax 對整層，輸出加起來為 1。

**Common misconceptions**:

- "ReLU is differentiable everywhere, so it is the safest." No: it is not differentiable at $a=0$, and its zero derivative on the negative side can kill a neuron.
- "Sigmoid and softmax are the same." No: sigmoid acts on one number and its outputs are independent; softmax acts on a whole layer and its outputs sum to 1.

**圖**：本單元一張圖，`lec02-kp06-act-relu-family`（ReLU 家族曲線，功能 1、2），以 `assets/figures/src/lec02-kp0507-figures.py` 繪製。sigmoid 的斜率圖見第3章 KP06 的 `lec03-kp06-sigmoid`。

**Figure**: One figure in this unit, `lec02-kp06-act-relu-family` (curves of the ReLU family; functions 1, 2), drawn by `assets/figures/src/lec02-kp0507-figures.py`. The slope plot of the sigmoid is `lec03-kp06-sigmoid` in Chapter 3 KP06.

**想深入**（選讀）：Sigmoid 的導數可由 $\sigma=(1+e^{-a})^{-1}$ 直接求得：$\sigma'=e^{-a}(1+e^{-a})^{-2}=\sigma\cdot\frac{e^{-a}}{1+e^{-a}}=\sigma(1-\sigma)$。Softplus 的導數正好是 sigmoid，所以 softplus 是「ReLU 的光滑版」，它的導數是「階梯的光滑版」。

**Going deeper** (optional): The derivative of the sigmoid follows directly from $\sigma=(1+e^{-a})^{-1}$: $\sigma'=e^{-a}(1+e^{-a})^{-2}=\sigma\cdot\frac{e^{-a}}{1+e^{-a}}=\sigma(1-\sigma)$. The derivative of softplus is exactly the sigmoid, so softplus is a "smooth ReLU" and its derivative a "smooth step".

**小練習**：

1. 算 sigmoid 在 $a=1$ 的輸出與導數。（提示：先算 $\sigma(1)=1/(1+e^{-1})$，再用 $\sigma(1-\sigma)$。）
2. 對 $\mathbf{a}=(1,1,1)$ 做 softmax，結果是什麼？（提示：三個指數相同。）
3. 某顆神經元的輸入對所有訓練資料都是負的，用 ReLU 會怎樣？換成 Leaky ReLU 會怎樣？（提示：看導數。）

**Practice**:

1. Compute the output and derivative of the sigmoid at $a=1$. (Hint: compute $\sigma(1)=1/(1+e^{-1})$ first, then use $\sigma(1-\sigma)$.)
2. Apply softmax to $\mathbf{a}=(1,1,1)$. What is the result? (Hint: the three exponentials are equal.)
3. A neuron's input is negative for all training data. What happens with ReLU, and what with Leaky ReLU? (Hint: look at the derivative.)

**解題示範**（第 1 題）：

1. $e^{-1}=0.3679$，所以 $\sigma(1)=1/(1+0.3679)=0.7311$。
2. 導數 $\sigma'(1)=\sigma(1-\sigma)=0.7311\times0.2689=0.1966$。
3. 檢查：導數小於最大值 $0.25$，且 $a=1$ 離 0 不遠，與第 2 節「在 0 處最大」一致。

**Worked solution** (exercise 1):

1. $e^{-1}=0.3679$, so $\sigma(1)=1/(1+0.3679)=0.7311$.
2. The derivative is $\sigma'(1)=\sigma(1-\sigma)=0.7311\times0.2689=0.1966$.
3. Check: the derivative is below the maximum $0.25$ and $a=1$ is not far from 0, consistent with "largest at 0" in section 2.

**自我檢核**（來源四題，不附答案）：

| 題 / Question | A | B | C | D | 提示 / Hint |
|---|---|---|---|---|---|
| 1. 為何 ReLU 被視為深度學習的里程碑？ / Why is ReLU a milestone? | 值域 $[-1,1]$ 且零中心 | 正值段導數恆為 1 避免梯度消失，並歸零負值帶來稀疏 | 輸出可當機率 | 負區間有非零斜率 | 看第 3 節。 / Section 3. |
| 2. 哪一種是「跨層」而非逐點作用、輸出可視為多類機率分布？ / Which acts across a layer and gives a distribution? | ReLU | maxout | softmax | tanh | 看第 2 節。 / Section 2. |
| 3. 「dying ReLU」指的是？ / What is "dying ReLU"? | 輸出太大而爆炸 | 負區間導數為 0、權重不再更新而死掉 | 無法處理正數輸入 | 計算太慢 | 看導數。 / Look at the derivative. |
| 4. PReLU 與 Leaky ReLU 的主要差別？ / How do PReLU and Leaky ReLU differ? | 值域不同 | PReLU 的負區間斜率 $\alpha$ 可學習，Leaky 固定 0.01 | PReLU 只能用在輸出層 | 完全相同 | 看符號表。 / See the symbol table. |

**留給下一個知識點的問題**：網路輸出了一個結果，我們要拿它和正確答案比，並把「差多少」變成一個數，才能知道往哪個方向調權重。這個數該怎麼定義？

**Question for the next knowledge point**: The network gives a result, and we must compare it with the right answer and turn "how far off" into one number before we know which way to adjust the weights. How should that number be defined?

### 來源更正與取捨（Corrections to the source）

- 來源把 Binary Step 的值域寫成 $[0,1]$；本講義寫成 $\{0,1\}$，因為它只取兩個值。
- 來源把 Maxout 寫成 $\sigma_i(\mathbf{a})=\max_j(\sigma_j)$，符號含糊；本講義說明為「多個線性單元的最大值」，詳細形式不展開。
- 來源說 ReLU「避免梯度消失」；本講義限定為「正側導數恆為 1，梯度不縮小」，並補上 S 形 $0.25^5$ 的數字對照（`[補充]`）。
- 來源把 ReLU 類在 $a=0$ 的導數稱為「未定義」；本講義補充實作上會約定一個值。

- The source gives the range of Binary Step as $[0,1]$; this note writes $\{0,1\}$ because it takes only two values.
- The source writes Maxout as $\sigma_i(\mathbf{a})=\max_j(\sigma_j)$, whose notation is ambiguous; this note describes it as "the maximum of several linear units" without expanding the detailed form.
- The source says ReLU "avoids vanishing gradients"; this note limits that to "the derivative is exactly 1 on the positive side, so the gradient does not shrink" and adds the $0.25^5$ number for S-shapes as a comparison (`[Added]`).
- The source calls the derivative of the ReLU-type functions at $a=0$ "undefined"; this note adds that in practice a value is fixed by convention.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 活化函數 | activation function |
| 梯度消失 | vanishing gradient |
| 稀疏性 | sparsity |
| 逐點 | pointwise |
| 值域 | range |
| 導數 | derivative |
| 跨層 | cross-layer |
