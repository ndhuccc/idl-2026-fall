# 第3週 KP14：高效 SAM 與自適應 SAM（Efficient and Adaptive SAM）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

> **自主學習 / Self-study**：本單元為自主學習內容，課堂不講授，請自行閱讀。

本講義是第3章「反向傳播與最佳化」的第十四個知識點（KP14），承接 KP13 結尾留下的代價與限制：SAM 每步要算兩次梯度，大約貴一倍；而且它量到的銳度會隨權重的尺度改變。先備知識是 KP12–KP13（平坦解、SAM 的擾動 $\epsilon^{\ast}$）、期望與變異數、向量範數；第一次出現的符號都會定義，所有數字都可以自己重算。內容依 CCC-LNS 第3章 KP14 的主文與投影片消化後重新撰寫，讀取日期 2026-10-14；來源沒有的內容標示 `[補充]`。

This is the fourteenth knowledge point (KP14) of Chapter 3. It picks up the costs and limits KP13 left open: SAM computes two gradients per step, which costs about twice as much, and the sharpness it measures changes with the scale of the weights. It assumes KP12–KP13 (flat minima and the SAM perturbation $\epsilon^{\ast}$), expectation and variance, and vector norms; every symbol is defined when it first appears and every number can be recomputed. The text is rewritten from the KP14 main text and slides of CCC-LNS Chapter 3 (accessed on 2026-10-14); content that the source does not contain is marked `[Added]`.

[下載本單元投影片](../slides/lec03-kp14-efficient-and-adaptive-sam.jbd.json)

## KP14 高效 SAM 與自適應 SAM（Efficient and Adaptive SAM）

### 學習目標（Learning objectives）

讀完之後，你能說出 SAM 每步貴在哪裡、尺度問題長什麼樣；能寫出 SWP 的遮罩擾動 $\hat\epsilon=m\odot\epsilon^{\ast}/\beta$，並推導 $\mathbb E[\hat\epsilon]=\epsilon^{\ast}$，同時說明為什麼梯度的期望不一定無偏；能寫出 SDS 的篩選條件並用成本模型算出每步的相對成本；能寫出 ASAM 的鄰域 $\|T_w^{-1}\epsilon\|\le\rho$、推導 $p=2$ 時的 $\epsilon^{\ast}$，並驗證條件式 (56)；也能分清楚三種方法改的分別是計算預算、樣本選擇還是鄰域幾何。

After reading, you can say where SAM spends its extra cost and what the scale problem looks like; write the masked perturbation $\hat\epsilon=m\odot\epsilon^{\ast}/\beta$ of SWP, derive $\mathbb E[\hat\epsilon]=\epsilon^{\ast}$ and explain why the expected gradient need not be unbiased; write the selection rule of SDS and compute the relative cost per step with a cost model; write the neighbourhood $\|T_w^{-1}\epsilon\|\le\rho$ of ASAM, derive $\epsilon^{\ast}$ for $p=2$ and verify condition (56); and tell apart what each of the three methods changes: the compute budget, the sample selection or the neighbourhood geometry.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $L(w)$ | 損失函數 | the loss function |
| $g=\nabla L(w)$ | $w$ 處的梯度 | the gradient at $w$ |
| $\rho$ | 擾動半徑 | perturbation radius |
| $\epsilon^{\ast}$ | SAM 的（一階）最壞擾動，$p=2$ 時 $\epsilon^{\ast}=\rho\,g/\|g\|$ | the (first-order) worst-case perturbation of SAM; $\epsilon^{\ast}=\rho\,g/\|g\|$ for $p=2$ |
| $m,\ \beta$ | 隨機遮罩（0 或 1 的向量）、每個分量取 1 的機率 | random mask (a 0/1 vector) and the probability that a component is 1 |
| $\hat\epsilon$ | SWP 的遮罩擾動 | the masked perturbation of SWP |
| $\ell_i$ | 第 $i$ 筆樣本的損失 | loss of sample $i$ |
| $\alpha,\ \mathcal B^{+},\ \mathcal B^{-}$ | SDS 的門檻、被選中的樣本集合、被跳過的樣本集合 | the SDS threshold, the selected samples and the skipped samples |
| $r$ | 保留比例 $\lvert\mathcal B^{+}\rvert/\lvert\mathcal B\rvert$ | kept fraction $\lvert\mathcal B^{+}\rvert/\lvert\mathcal B\rvert$ |
| $T_w,\ \gamma$ | ASAM 的正規化矩陣 $\mathrm{diag}(\lvert w\rvert+\gamma)$ 與穩定項 | the ASAM normalization matrix $\mathrm{diag}(\lvert w\rvert+\gamma)$ and its stabilizer |
| $A$ | 可逆的縮放矩陣（本講義取對角、元素為正） | an invertible scaling matrix (diagonal with positive entries here) |

### 1. SAM 的兩個痛點（Two pain points of SAM）

**成本。** SGD 每步算一次梯度。SAM 先在 $w$ 算梯度 $g$，據此求出擾動 $\epsilon^{\ast}$，再到 $w+\epsilon^{\ast}$ 算第二次梯度來更新，所以每步有兩次前向、兩次反向。要把「貴」量化需要一個成本模型 `[補充]`：設前向的成本是 1 個單位，反向是前向的兩倍（常見的粗略假設，實際依模型與硬體而變）。SGD 每步是 $1+2=3$ 個單位，SAM 是 $2\times3=6$ 個單位，約兩倍〔程式計算〕。下面所有「相對成本」都用這個模型；換成別的比例，數字會變，結論的形狀不變。

**Cost.** SGD computes one gradient per step. SAM first computes the gradient $g$ at $w$, builds the perturbation $\epsilon^{\ast}$ from it, and then computes a second gradient at $w+\epsilon^{\ast}$ for the update, so each step has two forward and two backward passes. To quantify "expensive" we need a cost model `[Added]`: let a forward pass cost 1 unit and a backward pass twice as much (a common rough assumption; it depends on the model and the hardware). An SGD step is $1+2=3$ units and a SAM step $2\times3=6$ units, about twice [computed by program]. All "relative costs" below use this model; with another ratio the numbers change but the shape of the conclusion does not.

**尺度。** 取兩層線性網路 $L(w_1,w_2)=\tfrac12(w_1w_2-1)^2$（KP12 的重新縮放例子）。只要 $w_1w_2=1$，損失就是 0：$(1,1)$ 與 $(4,\,0.25)$ 都是最小點，而且網路實現的函數相同。但固定 $\rho=0.5$ 的 SAM 銳度（鄰域內最大損失減去中心損失）不一樣：$(1,1)$ 是 $0.346$，$(4,\,0.25)$ 是 $2.069$，差 $5.98$ 倍〔程式計算，圓盤上的暴力搜尋〕。同一個函數，因為權重怎麼分配而被量成不同的銳度，這使「平坦」的比較依賴權重的單位。

**Scale.** Take the two-layer linear network $L(w_1,w_2)=\tfrac12(w_1w_2-1)^2$ (the rescaling example of KP12). Whenever $w_1w_2=1$ the loss is 0: $(1,1)$ and $(4,\,0.25)$ are both minimizers and the network realizes the same function. But the SAM sharpness with a fixed $\rho=0.5$ (the largest loss in the neighbourhood minus the loss at the centre) differs: $0.346$ at $(1,1)$ and $2.069$ at $(4,\,0.25)$, a factor of $5.98$ [computed by program, brute-force search on the disc]. The same function is measured with different sharpness depending on how the weights are split, so the comparison of "flatness" depends on the units of the weights.

兩個痛點對應兩種對症下藥：成本可以靠少算一點（SWP 少擾動權重，SDS 少處理樣本）；尺度要靠改鄰域的形狀（ASAM）。

The two pain points call for two kinds of remedies: the cost can be cut by computing less (SWP perturbs fewer weights, SDS processes fewer samples); the scale problem needs a different shape of neighbourhood (ASAM).

### 2. SWP：隨機遮罩的擾動（SWP: a randomly masked perturbation）

**做法。** SWP（stochastic weight perturbation）每步抽一個遮罩 $m=(m_1,\dots,m_n)$，各分量獨立，$m_i\sim\mathrm{Bernoulli}(\beta)$，也就是 $P(m_i=1)=\beta$、$P(m_i=0)=1-\beta$。擾動取成

**Method.** SWP (stochastic weight perturbation) draws a mask $m=(m_1,\dots,m_n)$ at every step, with independent components $m_i\sim\mathrm{Bernoulli}(\beta)$, that is $P(m_i=1)=\beta$ and $P(m_i=0)=1-\beta$. The perturbation is

$$
\hat\epsilon=\frac{1}{\beta}\,m\odot\epsilon^{\ast},\qquad \hat\epsilon_i=\frac{m_i}{\beta}\,\epsilon^{\ast}_i
$$

其中 $\odot$ 是逐元素相乘。沒被選中的權重這一步不受擾動，被選中的放大 $1/\beta$ 倍。（來源把它寫成 $\tfrac1\beta m^{\top}\epsilon^{\ast}$，那是內積，結果是純量，應該是逐元素相乘，見「來源更正與取捨」。）平均有 $\beta n$ 個權重被擾動。

where $\odot$ is element-wise multiplication. A weight that is not selected is not perturbed in this step; a selected one is amplified by $1/\beta$. (The source writes $\tfrac1\beta m^{\top}\epsilon^{\ast}$, which is an inner product and a scalar; it should be an element-wise product, see "Corrections to the source".) On average $\beta n$ weights are perturbed.

**期望等於 $\epsilon^{\ast}$。** 在給定 $w$ 與這個 mini-batch 時，$\epsilon^{\ast}$ 是固定的向量，遮罩與它獨立，所以

**The expectation equals $\epsilon^{\ast}$.** Given $w$ and the mini-batch, $\epsilon^{\ast}$ is a fixed vector and the mask is independent of it, so

$$
\mathbb E[\hat\epsilon_i]=\frac{\mathbb E[m_i]}{\beta}\,\epsilon^{\ast}_i=\frac{\beta}{\beta}\,\epsilon^{\ast}_i=\epsilon^{\ast}_i
$$

這就是除以 $\beta$ 的理由：補回被丟掉的部分。順便得到變異數：$\mathrm{Var}(\hat\epsilon_i)=(\epsilon^{\ast}_i)^2\,\tfrac{1-\beta}{\beta}$，以及 $\mathbb E\|\hat\epsilon\|^2=\|\epsilon^{\ast}\|^2/\beta$。$\beta$ 越小，省得越多，雜訊也越大。

This is the reason for dividing by $\beta$: it puts back what was dropped. The variance follows: $\mathrm{Var}(\hat\epsilon_i)=(\epsilon^{\ast}_i)^2\,\tfrac{1-\beta}{\beta}$, and $\mathbb E\|\hat\epsilon\|^2=\|\epsilon^{\ast}\|^2/\beta$. The smaller $\beta$ is, the more is saved and the larger the noise.

**玩具碗上驗證。** 碗 $f(x,y)=\tfrac12(x^2+25y^2)$，起點 $(10,2)$，梯度 $g=(10,\,50)$，$\|g\|=50.990$，$\rho=0.5$，所以 $\epsilon^{\ast}=\rho\,g/\|g\|=(0.0981,\,0.4903)$。取 $\beta=0.5$，兩個分量的遮罩有四種組合，各機率 $1/4$：

**Check on the toy bowl.** Bowl $f(x,y)=\tfrac12(x^2+25y^2)$, start $(10,2)$, gradient $g=(10,\,50)$, $\|g\|=50.990$, $\rho=0.5$, so $\epsilon^{\ast}=\rho\,g/\|g\|=(0.0981,\,0.4903)$. With $\beta=0.5$ the two-component mask has four combinations, each with probability $1/4$:

| 遮罩 $m$ | $\hat\epsilon$ | $\lVert\hat\epsilon\rVert$ |
|---|---|---|
| $(0,0)$ | $(0,\ 0)$ | $0$ |
| $(1,0)$ | $(0.1961,\ 0)$ | $0.1961$ |
| $(0,1)$ | $(0,\ 0.9806)$ | $0.9806$ |
| $(1,1)$ | $(0.1961,\ 0.9806)$ | $1.0$ |

| Mask $m$ | $\hat\epsilon$ | $\lVert\hat\epsilon\rVert$ |
|---|---|---|
| $(0,0)$ | $(0,\ 0)$ | $0$ |
| $(1,0)$ | $(0.1961,\ 0)$ | $0.1961$ |
| $(0,1)$ | $(0,\ 0.9806)$ | $0.9806$ |
| $(1,1)$ | $(0.1961,\ 0.9806)$ | $1.0$ |

四個值的平均正好是 $(0.0981,\,0.4903)=\epsilon^{\ast}$；$\mathbb E\|\hat\epsilon\|^2=0.5=\rho^2/\beta$〔程式計算〕。用隨機抽樣模擬：$10^4$ 個遮罩的平均是 $(0.0984,\,0.4804)$，$N$ 個遮罩的平均與 $\epsilon^{\ast}$ 的相對誤差（300 次重複的方均根）在 $N=10,\,10^2,\,10^3,\,10^4,\,10^5$ 時是 $0.303,\ 0.096,\ 0.032,\ 0.0094,\ 0.0031$，大約隨 $1/\sqrt N$ 下降〔程式計算〕。

The mean of the four values is exactly $(0.0981,\,0.4903)=\epsilon^{\ast}$; $\mathbb E\|\hat\epsilon\|^2=0.5=\rho^2/\beta$ [computed by program]. Random sampling agrees: the mean of $10^4$ masks is $(0.0984,\,0.4804)$, and the relative error of the mean of $N$ masks with respect to $\epsilon^{\ast}$ (root mean square over 300 repetitions) at $N=10,\,10^2,\,10^3,\,10^4,\,10^5$ is $0.303,\ 0.096,\ 0.032,\ 0.0094,\ 0.0031$, falling roughly like $1/\sqrt N$ [computed by program].

**不要把結論說過頭。** 以上只說明 $\hat\epsilon$ 本身的期望，有兩點要限縮。第一，單次的 $\hat\epsilon$ 可以比 $\rho$ 長：$(1,1)$ 那一格是 $1.0=\rho/\beta$，已經跑出半徑 $0.5$ 的球。第二，SAM 真正用到的是梯度 $\nabla L(w+\hat\epsilon)$，它是 $\hat\epsilon$ 的函數；若這個函數不是線性的，期望不能搬進去：$\mathbb E[\nabla L(w+\hat\epsilon)]\ne\nabla L(w+\epsilon^{\ast})$。碗的梯度是線性的，四種遮罩的梯度 $(10.196,\,74.51),\ (10.196,\,50),\ (10,\,74.51),\ (10,\,50)$ 平均是 $(10.098,\,62.257)$，與 $\nabla f(w+\epsilon^{\ast})$ 完全相同，所以碗剛好無偏。換成四次函數 $f_q=\tfrac14(x^4+y^4)$，在 $w=(1,1)$、同樣 $\rho=0.5$、$\beta=0.5$：$\epsilon^{\ast}=(0.3536,\,0.3536)$，$x$ 分量的梯度 $(1+\hat\epsilon_x)^3$ 在四種遮罩下是 $4.975,\ 4.975,\ 1,\ 1$，平均 $2.987$，而 $\nabla f_q(w+\epsilon^{\ast})$ 的 $x$ 分量是 $2.480$，整個向量的相對差是 $20.5\%$〔程式計算〕。二階展開可以看出原因：偏差大約是 $\tfrac12\nabla^3L$ 乘上 $\hat\epsilon$ 的變異數 `[補充]`，與 $\beta$ 越小、$\hat\epsilon$ 越吵直接相關。圖 `lec03-kp14-swp-unbiased` 把這幾件事放在一起。

**Do not overstate the result.** The above concerns only the expectation of $\hat\epsilon$ itself, and two caveats apply. First, a single $\hat\epsilon$ can be longer than $\rho$: the $(1,1)$ row has $1.0=\rho/\beta$, already outside the ball of radius $0.5$. Second, what SAM really uses is the gradient $\nabla L(w+\hat\epsilon)$, a function of $\hat\epsilon$; if it is not linear, the expectation cannot be moved inside: $\mathbb E[\nabla L(w+\hat\epsilon)]\ne\nabla L(w+\epsilon^{\ast})$. The gradient of the bowl is linear: the four gradients $(10.196,\,74.51),\ (10.196,\,50),\ (10,\,74.51),\ (10,\,50)$ average to $(10.098,\,62.257)$, identical to $\nabla f(w+\epsilon^{\ast})$, so the bowl happens to be unbiased. Replace it with the quartic $f_q=\tfrac14(x^4+y^4)$ at $w=(1,1)$ with the same $\rho=0.5$ and $\beta=0.5$: $\epsilon^{\ast}=(0.3536,\,0.3536)$; the $x$ component of the gradient, $(1+\hat\epsilon_x)^3$, is $4.975,\ 4.975,\ 1,\ 1$ under the four masks, with mean $2.987$, while the $x$ component of $\nabla f_q(w+\epsilon^{\ast})$ is $2.480$; the relative difference of the whole vector is $20.5\%$ [computed by program]. A second-order expansion shows why: the bias is about $\tfrac12\nabla^3L$ times the variance of $\hat\epsilon$ `[Added]`, tied directly to how noisy $\hat\epsilon$ is when $\beta$ is small. The figure `lec03-kp14-swp-unbiased` puts these points together.

![左圖：玩具碗在 (10,2)、β=0.5 時 ε̂ 只有四種可能，各機率 1/4，平均落在 ε*。右圖：N 個遮罩的平均，碗上 ε̂ 與梯度的相對誤差隨 N 下降，四次函數的梯度平均停在約 20%。（Left: on the toy bowl at (10,2) with β=0.5, ε̂ has four possible values, each with probability 1/4, and their mean is ε*. Right: the mean of N masks; on the bowl the relative error of ε̂ and of the gradient falls with N, while for the quartic the mean gradient stays about 20% off.）](../assets/figures/lec03-kp14-swp-unbiased.png)

**圖說 `lec03-kp14-swp-unbiased`**：先看左圖：藍點是四種遮罩的 $\hat\epsilon$，橘色星號是它們的平均，落在 $\epsilon^{\ast}$，也就是半徑 $\rho$ 的虛線圓上；$(1,1)$ 那個點在圓外。再看右圖：藍線與綠線一路下降，代表平均逼近目標；紅線停在 $0.2$ 左右，代表四次函數的期望梯度有一個去不掉的偏差。兩軸都是對數刻度。〔功能：1、2〕

**Caption `lec03-kp14-swp-unbiased`**: Look at the left panel first: the blue dots are $\hat\epsilon$ for the four masks and the orange star is their mean, which sits at $\epsilon^{\ast}$ on the dashed circle of radius $\rho$; the $(1,1)$ dot lies outside the circle. Then the right panel: the blue and green lines keep falling, meaning that the mean approaches its target; the red line stays near $0.2$, meaning that the expected gradient of the quartic has a bias that does not go away. Both axes are logarithmic. [Function: 1, 2]

**省多少？** SWP 少擾動了權重，但實際省下的時間取決於實作（要不要為被遮掉的權重算那一部分梯度、記憶體怎麼搬），不是一個由 $\beta$ 決定的固定倍數。本講義不給倍數 `[補充]`。另外，隨機遮罩帶來的雜訊是否有類似 Dropout 的正則化效果，來源沒有證據，這裡不採用。

**How much is saved?** SWP perturbs fewer weights, but the time actually saved depends on the implementation (whether the gradient part for the masked weights is skipped, how memory is moved); it is not a fixed multiple set by $\beta$. This note gives no multiple `[Added]`. Whether the noise of the random mask has a Dropout-like regularizing effect is not supported by evidence in the source and is not adopted here.

用步長 $\eta=0.05$ 在碗上各跑 60 步（起點 $(10,2)$，$\rho=0.5$）：SAM（完整 $\epsilon^{\ast}$）的終點損失是 $8.77$；SWP 用 5 個隨機種子，終點損失是 $26.3,\ 34.0,\ 1.28,\ 0.198,\ 1.29$〔程式計算〕。這不是速度或效果的比較，只是提醒：隨機遮罩讓每條軌跡的變異變大，種子之間差到 170 倍。

Running 60 steps on the bowl with step size $\eta=0.05$ (start $(10,2)$, $\rho=0.5$): the final loss of SAM (full $\epsilon^{\ast}$) is $8.77$; SWP with 5 random seeds ends at $26.3,\ 34.0,\ 1.28,\ 0.198,\ 1.29$ [computed by program]. This is not a comparison of speed or quality; it only reminds us that the random mask makes each trajectory more variable, a factor of 170 between seeds.

### 3. SDS：只讓銳度敏感的樣本做第二次反傳（SDS: only sharpness-sensitive samples take the second backward pass）

**做法。** SDS（sharpness-sensitive data selection）從資料這一維省。先用整批算出擾動 $\epsilon^{\ast}$，再對每筆樣本做一次前向，看擾動讓它的損失漲了多少，$\ell_i(w+\epsilon^{\ast})-\ell_i(w)$。漲幅大於門檻 $\alpha$ 的樣本放進 $\mathcal B^{+}$，只有它們做 SAM 的第二次反傳；其餘的 $\mathcal B^{-}$ 跳過：

**Method.** SDS (sharpness-sensitive data selection) saves along the data dimension. First compute the perturbation $\epsilon^{\ast}$ from the whole batch, then run a forward pass for every sample to see how much the perturbation raises its loss, $\ell_i(w+\epsilon^{\ast})-\ell_i(w)$. Samples whose rise exceeds the threshold $\alpha$ go into $\mathcal B^{+}$ and only they take the second SAM backward pass; the others, $\mathcal B^{-}$, are skipped:

$$
\mathcal B^{+}=\{\,i\in\mathcal B:\ \ell_i(w+\epsilon^{\ast})-\ell_i(w)>\alpha\,\},\qquad \mathcal B^{-}=\mathcal B\setminus\mathcal B^{+}
$$

$\alpha$ 是超參數（來源對恰好等於 $\alpha$ 的情形沒有規定，這裡把它歸入 $\mathcal B^{-}$）。

$\alpha$ is a hyperparameter (the source does not say what happens at exactly $\alpha$; here such a sample goes to $\mathcal B^{-}$).

**四筆樣本的例子。** 取 $f_i(w)=\tfrac12(w-a_i)^2$，$a=(2,4,6,8)$，這四筆的整批最小點是 $w^{\ast}=5$（KP07 起用的例子）。在 $w=7$：各筆梯度 $g_i=w-a_i=(5,3,1,-1)$，整批梯度 $2$，所以一維的 $\epsilon^{\ast}=+\rho=+0.5$。各筆的漲幅是 $g_i\epsilon^{\ast}+\tfrac12(\epsilon^{\ast})^2=(2.625,\ 1.625,\ 0.625,\ -0.375)$；取 $\alpha=1$，$\mathcal B^{+}=\{a=2,\,a=4\}$，$a=6$ 與 $a=8$ 被跳過。四個漲幅的平均 $1.125$ 就是整批的損失漲幅〔程式計算〕。

**A four-sample example.** Take $f_i(w)=\tfrac12(w-a_i)^2$ with $a=(2,4,6,8)$, whose batch minimizer is $w^{\ast}=5$ (the example used since KP07). At $w=7$ the sample gradients are $g_i=w-a_i=(5,3,1,-1)$, the batch gradient is $2$, so the one-dimensional $\epsilon^{\ast}=+\rho=+0.5$. The rises are $g_i\epsilon^{\ast}+\tfrac12(\epsilon^{\ast})^2=(2.625,\ 1.625,\ 0.625,\ -0.375)$; with $\alpha=1$, $\mathcal B^{+}=\{a=2,\,a=4\}$ and $a=6$, $a=8$ are skipped. The mean of the four rises, $1.125$, is the loss rise of the whole batch [computed by program].

**成本要誠實算。** SDS 並不省掉所有第二輪的工作：第一次反傳整批都要做，擾動點的前向整批都要做，只有第二次反傳按保留比例 $r=|\mathcal B^{+}|/|\mathcal B|$ 縮小。用第 1 節的成本模型（前向 1、反向 2）：每步 $(1+2)+1+2r=4+2r$ 個單位，相對 SGD 的 3 個單位是 $(4+2r)/3$ 倍。$r=1$ 是 $2.0$（等於 SAM），$r=0.75,\,0.5,\,0.25,\,0$ 是 $1.83,\ 1.67,\ 1.50,\ 1.33$〔程式計算〕。也就是在這個模型下，單用 SDS 的下限是 $4/3\approx1.33$，不可能低於它。若反向是前向的三倍，下限是 $1.25$（$r=0.5$ 時 $1.625$）〔程式計算〕。來源說合併 SWP 與 SDS 後約 $1.2$–$1.3$ 倍，那個數字依賴來源的實驗設定與實作，在我們的成本模型下單靠 SDS 做不到，所以只當作來源的說法，不當成定值。

**The cost must be counted honestly.** SDS does not remove all the second-round work: the first backward pass runs on the whole batch, the forward pass at the perturbed point runs on the whole batch, and only the second backward pass shrinks with the kept fraction $r=|\mathcal B^{+}|/|\mathcal B|$. With the cost model of section 1 (forward 1, backward 2) a step costs $(1+2)+1+2r=4+2r$ units, against 3 units for SGD, that is $(4+2r)/3$ times. At $r=1$ it is $2.0$ (equal to SAM), and at $r=0.75,\,0.5,\,0.25,\,0$ it is $1.83,\ 1.67,\ 1.50,\ 1.33$ [computed by program]. So in this model SDS alone cannot go below $4/3\approx1.33$. If the backward pass costs three times the forward, the floor is $1.25$ ($1.625$ at $r=0.5$) [computed by program]. The source reports about $1.2$–$1.3$ times for SWP and SDS together; that figure depends on the source's experimental setting and implementation and cannot be reached by SDS alone in our cost model, so it is quoted as the source's claim, not as a constant.

**還有一個提醒** `[補充]`：第二次梯度只在 $\mathcal B^{+}$ 上算，而 $\mathcal B^{+}$ 是依損失的漲幅挑的，所以它不再是完整批次 SAM 梯度的無偏估計；SDS 用的是一個偏向「銳度高的樣本」的近似。圖 `lec03-kp14-sds-select` 左圖畫出上面的例子，右圖畫出成本隨保留比例的變化。

**One more caution** `[Added]`: the second gradient is computed on $\mathcal B^{+}$ only, and $\mathcal B^{+}$ is picked by the loss rise, so it is no longer an unbiased estimate of the full-batch SAM gradient; SDS uses an approximation tilted toward high-sharpness samples. The figure `lec03-kp14-sds-select` shows the example above on the left and the cost against the kept fraction on the right.

![左圖：四筆樣本在擾動後的損失漲幅 2.625、1.625、0.625、-0.375，α=1，前兩筆進入 B+。右圖：每步成本相對 SGD 隨保留比例 r 線性上升，r=0.5 時約 1.67，r=1 時等於 SAM 的 2。（Left: the loss rises of four samples after the perturbation, 2.625, 1.625, 0.625, -0.375; with α=1 the first two enter B+. Right: the step cost relative to SGD rises linearly with the kept fraction r, about 1.67 at r=0.5 and 2 at r=1, equal to SAM.）](../assets/figures/lec03-kp14-sds-select.png)

**圖說 `lec03-kp14-sds-select`**：先看左圖：橘色長條超過黃色虛線 $\alpha=1$，是被選中的樣本；藍色長條沒超過，被跳過。注意第四筆的漲幅是負的：擾動讓它的損失下降。再看右圖：成本是保留比例 $r$ 的直線，起點 $4/3$，終點是 SAM 的 $2$，所以保留得越少越省，但省得有下限。〔功能：1、2〕

**Caption `lec03-kp14-sds-select`**: Look at the left panel first: orange bars above the yellow dashed line $\alpha=1$ are the selected samples; blue bars below it are skipped. Note that the fourth rise is negative: the perturbation lowers its loss. Then the right panel: the cost is a straight line in the kept fraction $r$, from $4/3$ up to the $2$ of SAM, so keeping less saves more, but the saving has a floor. [Function: 1, 2]

### 4. ASAM：讓鄰域跟著權重的尺度走（ASAM: letting the neighbourhood follow the scale of the weights）

**鄰域怎麼改。** SAM 的鄰域是各向同性的球 $\|\epsilon\|\le\rho$，對每個權重一視同仁。ASAM（adaptive SAM）改成

**How the neighbourhood changes.** The neighbourhood of SAM is an isotropic ball $\|\epsilon\|\le\rho$ that treats every weight alike. ASAM (adaptive SAM) changes it to

$$
\max_{\|T_w^{-1}\epsilon\|_p\le\rho}\ L(w+\epsilon)-L(w),\qquad T_w=\mathrm{diag}(|w_1|+\gamma,\ \dots,\ |w_n|+\gamma)
$$

大的權重有大的容許擾動，小的權重有小的：球變成一個沿著各坐標軸、長度正比於 $|w_i|+\gamma$ 的橢球。$\gamma>0$ 是小的穩定項，避免權重為 0 時鄰域退化成一個點。

A large weight is allowed a large perturbation and a small weight a small one: the ball becomes an ellipsoid along the coordinate axes with semi-axes proportional to $|w_i|+\gamma$. The small stabilizer $\gamma>0$ keeps the neighbourhood from collapsing to a point when a weight is 0.

**$p=2$ 時的最壞擾動（自己推一次）。** 令 $u=T_w^{-1}\epsilon$，則 $\epsilon=T_wu$，約束是 $\|u\|\le\rho$。一階近似下目標是 $g^{\top}\epsilon=g^{\top}T_wu=(T_wg)^{\top}u$（$T_w$ 是對角矩陣，對稱）。長度固定的 $u$ 使內積最大的方向是 $T_wg$ 的方向，所以 $u^{\ast}=\rho\,T_wg/\|T_wg\|$，換回去得到

**The worst-case perturbation for $p=2$ (derive it yourself).** Let $u=T_w^{-1}\epsilon$, so $\epsilon=T_wu$ and the constraint is $\|u\|\le\rho$. To first order the objective is $g^{\top}\epsilon=g^{\top}T_wu=(T_wg)^{\top}u$ ($T_w$ is diagonal, hence symmetric). The direction that maximizes the inner product for a fixed length of $u$ is that of $T_wg$, so $u^{\ast}=\rho\,T_wg/\|T_wg\|$, and going back gives

$$
\epsilon^{\ast}=T_wu^{\ast}=\rho\,\frac{T_w^{2}\,g}{\|T_w\,g\|}
$$

驗證：取 $w=(1,-2,0.5)$，$g=(0.3,0.1,-0.8)$，$\rho=0.1$，$\gamma=0.01$，公式給出 $\epsilon^{\ast}=(0.0560,\ 0.0739,\ -0.0381)$，$\|T_w^{-1}\epsilon^{\ast}\|=0.1$，$g^{\top}\epsilon^{\ast}=0.054651$，等於 $\rho\|T_wg\|$；在邊界上隨機取兩百萬個點，內積最大也只有 $0.054651$〔程式計算〕。

Check: take $w=(1,-2,0.5)$, $g=(0.3,0.1,-0.8)$, $\rho=0.1$, $\gamma=0.01$; the formula gives $\epsilon^{\ast}=(0.0560,\ 0.0739,\ -0.0381)$ with $\|T_w^{-1}\epsilon^{\ast}\|=0.1$ and $g^{\top}\epsilon^{\ast}=0.054651$, equal to $\rho\|T_wg\|$; among two million random points on the boundary the largest inner product is also $0.054651$ [computed by program].

在玩具碗的 $(10,2)$：SAM 的 $\epsilon^{\ast}=(0.098,\,0.490)$，相對權重是 $(0.98\%,\ 24.5\%)$，$y$ 方向被擾動得很大；ASAM 取 $\rho=0.1$、$\gamma=0.01$，$\epsilon^{\ast}=(0.706,\,0.142)$，相對權重是 $(7.06\%,\ 7.12\%)$，各方向的相對擾動幾乎一樣〔程式計算〕。這就是「跟著尺度走」的意思。注意 ASAM 的 $\rho$ 是相對權重大小的比例，SAM 的 $\rho$ 是絕對長度，兩者不能直接比較，上面兩個 $\rho$ 數值不同也是因為單位不同。

At $(10,2)$ on the toy bowl: the SAM perturbation $\epsilon^{\ast}=(0.098,\,0.490)$ is $(0.98\%,\ 24.5\%)$ of the weights, so the $y$ direction is perturbed very strongly; ASAM with $\rho=0.1$ and $\gamma=0.01$ gives $\epsilon^{\ast}=(0.706,\,0.142)$, which is $(7.06\%,\ 7.12\%)$ of the weights, almost the same relative perturbation in every direction [computed by program]. This is what "following the scale" means. Note that the $\rho$ of ASAM is a proportion of the weight sizes while the $\rho$ of SAM is an absolute length; they cannot be compared directly, and the two $\rho$ values above differ because their units differ.

**尺度不變性。** 重新縮放 $w\mapsto Aw$（$A=\mathrm{diag}(s_1,\dots,s_n)$，$s_i>0$），同一個函數換成新參數後的損失是 $L'(w')=L(A^{-1}w')$。ASAM 要求矩陣 $T_w$ 滿足

**Scale invariance.** Under a rescaling $w\mapsto Aw$ ($A=\mathrm{diag}(s_1,\dots,s_n)$, $s_i>0$), the same function written in the new parameters has the loss $L'(w')=L(A^{-1}w')$. ASAM requires the matrix $T_w$ to satisfy

$$
T_{Aw}^{-1}A=T_w^{-1}\tag{56}
$$

有了 (56)，把擾動也換成 $\epsilon'=A\epsilon$：約束 $\|T_{Aw}^{-1}\epsilon'\|=\|T_{Aw}^{-1}A\epsilon\|=\|T_w^{-1}\epsilon\|\le\rho$ 不變；目標 $L'(Aw+A\epsilon)-L'(Aw)=L(w+\epsilon)-L(w)$ 也不變。所以新舊參數上的 ASAM 銳度完全相同。

With (56), also write the perturbation as $\epsilon'=A\epsilon$: the constraint $\|T_{Aw}^{-1}\epsilon'\|=\|T_{Aw}^{-1}A\epsilon\|=\|T_w^{-1}\epsilon\|\le\rho$ is unchanged, and so is the objective $L'(Aw+A\epsilon)-L'(Aw)=L(w+\epsilon)-L(w)$. Hence the ASAM sharpness is exactly the same in the old and the new parameters.

**(56) 對 $\gamma>0$ 只是近似。** $T_{Aw}=\mathrm{diag}(s_i|w_i|+\gamma)$，而 $AT_w=\mathrm{diag}(s_i|w_i|+s_i\gamma)$，兩者只在 $\gamma=0$ 時相等。來源說「$T_{Aw}=AT_w$ 恆成立」，不準確。數值例：$w=(0.5,\,2)$，$A=\mathrm{diag}(4,\,0.25)$，$T_{Aw}^{-1}A$ 的對角線在 $\gamma=0$ 是 $(2,\ 0.5)$，與 $T_w^{-1}=(2,\ 0.5)$ 完全相同；$\gamma=0.01$ 是 $(1.990,\ 0.490)$ 對 $(1.961,\ 0.498)$，最大差 $0.029$〔程式計算〕。另外，對角的 $T_w$ 只對「對角、元素為正」的 $A$ 成立；對一般的可逆矩陣（例如旋轉）(56) 不成立，所以來源說的「任意可逆線性變換」要限縮成「各坐標獨立的正縮放」。

**(56) is only approximate for $\gamma>0$.** $T_{Aw}=\mathrm{diag}(s_i|w_i|+\gamma)$ while $AT_w=\mathrm{diag}(s_i|w_i|+s_i\gamma)$; they agree only at $\gamma=0$. The source says "$T_{Aw}=AT_w$ always holds", which is not accurate. Numerical example: $w=(0.5,\,2)$, $A=\mathrm{diag}(4,\,0.25)$; the diagonal of $T_{Aw}^{-1}A$ is $(2,\ 0.5)$ at $\gamma=0$, exactly equal to $T_w^{-1}=(2,\ 0.5)$; at $\gamma=0.01$ it is $(1.990,\ 0.490)$ against $(1.961,\ 0.498)$, a maximum difference of $0.029$ [computed by program]. Moreover a diagonal $T_w$ works only for $A$ diagonal with positive entries; for a general invertible matrix (a rotation, say) (56) fails, so the source's "any invertible linear transformation" must be narrowed to "positive scaling of each coordinate independently".

**兩層線性網路上的數字。** 把最小點 $(a,\,1/a)$ 的縮放 $a$ 從 $1/4$ 掃到 $8$（它們都是同一個函數），固定 $\rho=0.5$ 的 SAM 銳度是 $2.069,\ 0.614,\ 0.346,\ 0.614,\ 2.069,\ 8.049$（$a=1/4,\,1/2,\,1,\,2,\,4,\,8$），最大與最小差 $23.3$ 倍；ASAM（$\rho=0.1$、$\gamma=0.01$）是 $0.01120,\ 0.01100,\ 0.01094,\ 0.01100,\ 0.01120,\ 0.01165$，最大與最小只差 $1.065$ 倍；$\gamma=0$ 時全部是 $0.01072$，完全不變〔程式計算〕。一階估計也能看出原因：在最小點附近 $L\approx\tfrac12(w_2\epsilon_1+w_1\epsilon_2)^2$，SAM 的最大值是 $\tfrac12\rho^2(w_1^2+w_2^2)=\tfrac12\rho^2(a^2+a^{-2})$，隨 $a$ 增大（$a=1,\,4,\,8$ 時是 $0.250,\ 2.008,\ 8.002$，與暴力搜尋的 $0.346,\ 2.069,\ 8.049$ 同一個量級）；ASAM 代入 $\epsilon_i=t_iu_i$、$t_i\approx|w_i|$ 後是 $\rho^2(w_1w_2)^2=\rho^2=0.01$，與 $a$ 無關〔程式計算〕。

**Numbers on the two-layer linear network.** Sweep the scale $a$ of the minimizer $(a,\,1/a)$ from $1/4$ to $8$ (all are the same function). The SAM sharpness with $\rho=0.5$ is $2.069,\ 0.614,\ 0.346,\ 0.614,\ 2.069,\ 8.049$ for $a=1/4,\,1/2,\,1,\,2,\,4,\,8$, a factor of $23.3$ between the largest and the smallest; ASAM ($\rho=0.1$, $\gamma=0.01$) gives $0.01120,\ 0.01100,\ 0.01094,\ 0.01100,\ 0.01120,\ 0.01165$, only a factor of $1.065$; at $\gamma=0$ all are $0.01072$, exactly invariant [computed by program]. A first-order estimate shows why: near the minimizer $L\approx\tfrac12(w_2\epsilon_1+w_1\epsilon_2)^2$, and the SAM maximum is $\tfrac12\rho^2(w_1^2+w_2^2)=\tfrac12\rho^2(a^2+a^{-2})$, which grows with $a$ (at $a=1,\,4,\,8$ it is $0.250,\ 2.008,\ 8.002$, the same order as the brute-force $0.346,\ 2.069,\ 8.049$); for ASAM, substituting $\epsilon_i=t_iu_i$ with $t_i\approx|w_i|$ gives $\rho^2(w_1w_2)^2=\rho^2=0.01$, independent of $a$ [computed by program].

![左圖：在 (4, 0.25) 這個最小點，SAM 的固定圓盤跨過谷底兩側，ASAM 的橢圓沿著谷底拉長、貼著權重大小。右圖：最小點 (a, 1/a) 的縮放 a 從 1/4 到 8，SAM 銳度從 0.346 升到 8.05，ASAM 銳度幾乎不動（約 0.011）。（Left: at the minimizer (4, 0.25), the fixed disc of SAM reaches across both sides of the valley while the ellipse of ASAM is stretched along the valley and follows the weight sizes. Right: as the scale a of the minimizer (a, 1/a) goes from 1/4 to 8, the SAM sharpness rises from 0.346 to 8.05 while the ASAM sharpness hardly moves, about 0.011.）](../assets/figures/lec03-kp14-asam-scale.png)

**圖說 `lec03-kp14-asam-scale`**：先看左圖：灰線是損失的等高線，綠線是最小點構成的谷底 $w_1w_2=1$；藍色圓盤是 SAM 的鄰域，上下伸出谷底很遠，所以量到的損失很大；橘色細橢圓是 ASAM 的鄰域，沿著谷底貼得很緊。再看右圖：縱軸是對數刻度，藍線隨尺度彎成 U 形，橘線幾乎是水平線。兩種銳度的縱軸數值不能互相比大小，因為 $\rho$ 的意義不同。〔功能：1、2〕

**Caption `lec03-kp14-asam-scale`**: Look at the left panel first: the grey lines are loss contours and the green line is the valley floor of minimizers, $w_1w_2=1$; the blue disc is the neighbourhood of SAM, which reaches far across the valley and therefore measures a large loss; the thin orange ellipse is the neighbourhood of ASAM and hugs the valley. Then the right panel: the vertical axis is logarithmic, the blue line bends into a U shape with the scale and the orange line is almost horizontal. The values of the two sharpnesses cannot be compared in size because $\rho$ means different things. [Function: 1, 2]

**ASAM 不省計算。** 它仍然是兩次前向加兩次反向，約 $2$ 倍的成本；它改的是鄰域的形狀。

**ASAM saves no computation.** It still takes two forward and two backward passes, about $2$ times the cost; what it changes is the shape of the neighbourhood.

### 5. 三種方法改的是不同的東西（The three methods change different things）

| 方法 | 改的是什麼 | 相對 SAM | 不保證什麼 |
|---|---|---|---|
| SWP | 計算預算：每步只擾動隨機一部分權重 | $\mathbb E[\hat\epsilon]=\epsilon^{\ast}$，變異變大 | 梯度的期望不保證無偏；省多少時間看實作 |
| SDS | 樣本選擇：只有漲幅大於 $\alpha$ 的樣本做第二次反傳 | 成本 $(4+2r)/3$ 倍 SGD（模型下限 $4/3$） | 第二次梯度是偏向高銳度樣本的近似 |
| ASAM | 鄰域幾何：球換成 $\|T_w^{-1}\epsilon\|\le\rho$ | 成本不變（約 2 倍），銳度對正縮放近乎不變 | $\gamma>0$ 時 (56) 只是近似；$\rho$ 不能與 SAM 的比 |

| Method | What it changes | Relative to SAM | What is not guaranteed |
|---|---|---|---|
| SWP | the compute budget: only a random part of the weights is perturbed per step | $\mathbb E[\hat\epsilon]=\epsilon^{\ast}$, larger variance | the expected gradient is not guaranteed unbiased; the time saved depends on the implementation |
| SDS | the sample selection: only samples with a rise above $\alpha$ take the second backward pass | cost $(4+2r)/3$ times SGD (floor $4/3$ in the model) | the second gradient is an approximation tilted to high-sharpness samples |
| ASAM | the neighbourhood geometry: the ball becomes $\|T_w^{-1}\epsilon\|\le\rho$ | cost unchanged (about 2 times), sharpness nearly invariant under positive scaling | (56) is only approximate for $\gamma>0$; $\rho$ cannot be compared with that of SAM |

三者可以疊在一起用，但各自解決的問題不同，不要把「省了計算」與「改了鄰域」混為一談。最後留下一個問題：到目前為止我們用實驗與小例子看這些方法，從最佳化理論的角度，它們到底保證了什麼？這是 KP15 的主題。

The three can be stacked, but each solves a different problem; do not mix "saving computation" with "changing the neighbourhood". One question remains: so far we have examined these methods with experiments and small examples; what do they actually guarantee from the viewpoint of optimization theory? That is the topic of KP15.

### 本單元重點整理（KP14 summary）

**一句話重點**：SAM 貴在兩次梯度、又對權重尺度敏感；SWP 用隨機遮罩讓擾動的期望等於 $\epsilon^{\ast}$（梯度的期望不保證無偏），SDS 只讓漲幅大於 $\alpha$ 的樣本做第二次反傳，ASAM 用 $T_w$ 把鄰域正規化，使銳度對各坐標的正縮放近乎不變；三者改的分別是計算預算、樣本選擇與鄰域幾何。

**One-sentence summary**: SAM is costly because of its two gradients and is sensitive to the scale of the weights; SWP uses a random mask so that the expected perturbation equals $\epsilon^{\ast}$ (the expected gradient is not guaranteed unbiased), SDS lets only samples whose loss rise exceeds $\alpha$ take the second backward pass, and ASAM normalizes the neighbourhood with $T_w$ so that the sharpness is nearly invariant under positive scaling of the coordinates; the three change the compute budget, the sample selection and the neighbourhood geometry respectively.

**白話說明**：SAM 像每走一步都要先「試探一下鄰居」再走，所以貴一倍。SWP 的做法是每次只試探一部分方向，再把試探的幅度放大補回來，平均起來和全部試探一樣；SDS 是只對看起來「站在懸崖邊」的樣本再仔細檢查一次；ASAM 則是發現「一步多長算近」取決於用什麼單位量，所以讓每個方向的「近」跟著那個方向的權重大小走，換單位就不會改變結論。

**Plain explanation**: SAM is like probing the neighbours before every step, so it costs twice as much. SWP probes only some directions each time and scales the probe up to compensate, so that on average it equals probing all of them; SDS re-checks only the samples that seem to stand at a cliff edge; ASAM notices that "how long a step counts as near" depends on the unit of measurement, so it lets "near" in each direction follow the size of the weight in that direction, and changing the unit no longer changes the conclusion.

**正式定義**：SWP：$m_i\sim\mathrm{Bernoulli}(\beta)$，$\hat\epsilon=m\odot\epsilon^{\ast}/\beta$，$\mathbb E[\hat\epsilon]=\epsilon^{\ast}$。SDS：$\mathcal B^{+}=\{i:\ \ell_i(w+\epsilon^{\ast})-\ell_i(w)>\alpha\}$。ASAM：$\max_{\|T_w^{-1}\epsilon\|\le\rho}L(w+\epsilon)-L(w)$，$T_w=\mathrm{diag}(|w|+\gamma)$，$p=2$ 時 $\epsilon^{\ast}=\rho\,T_w^{2}g/\|T_wg\|$，$T_{Aw}^{-1}A=T_w^{-1}$（式 (56)，$\gamma=0$ 時精確）。

**Formal definition**: SWP: $m_i\sim\mathrm{Bernoulli}(\beta)$, $\hat\epsilon=m\odot\epsilon^{\ast}/\beta$, $\mathbb E[\hat\epsilon]=\epsilon^{\ast}$. SDS: $\mathcal B^{+}=\{i:\ \ell_i(w+\epsilon^{\ast})-\ell_i(w)>\alpha\}$. ASAM: $\max_{\|T_w^{-1}\epsilon\|\le\rho}L(w+\epsilon)-L(w)$ with $T_w=\mathrm{diag}(|w|+\gamma)$; for $p=2$, $\epsilon^{\ast}=\rho\,T_w^{2}g/\|T_wg\|$; and $T_{Aw}^{-1}A=T_w^{-1}$ (equation (56), exact at $\gamma=0$).

**例子**：碗 $f=\tfrac12(x^2+25y^2)$、起點 $(10,2)$、$\rho=0.5$、$\beta=0.5$：$\epsilon^{\ast}=(0.0981,\,0.4903)$，四種遮罩的 $\hat\epsilon$ 平均正好是它；四次函數 $f_q$ 的期望梯度卻差 $20.5\%$。四筆樣本、$w=7$、$\alpha=1$：漲幅 $2.625,\ 1.625,\ 0.625,\ -0.375$，選出 $a=2,\,4$。兩層線性網路：SAM 銳度在 $a=1/4$ 到 $8$ 差 $23.3$ 倍，ASAM 差 $1.065$ 倍。逐步計算見第 2–4 節與解題示範。

**Example**: bowl $f=\tfrac12(x^2+25y^2)$, start $(10,2)$, $\rho=0.5$, $\beta=0.5$: $\epsilon^{\ast}=(0.0981,\,0.4903)$ and the mean of $\hat\epsilon$ over the four masks is exactly that; the expected gradient of the quartic $f_q$ is off by $20.5\%$. Four samples, $w=7$, $\alpha=1$: rises $2.625,\ 1.625,\ 0.625,\ -0.375$, selecting $a=2,\,4$. Two-layer linear network: the SAM sharpness differs by a factor of $23.3$ from $a=1/4$ to $8$, the ASAM sharpness by $1.065$. The computations are in sections 2–4 and in the worked solution.

**常見迷思**：

- 「SWP 嚴格無偏，所以方向一定不偏」——只有 $\hat\epsilon$ 的期望等於 $\epsilon^{\ast}$；梯度 $\nabla L(w+\hat\epsilon)$ 對 $\hat\epsilon$ 是非線性時不保證無偏（四次函數例差 $20.5\%$），單次 $\hat\epsilon$ 也可以超出 $\rho$ 球。
- 「SDS 把成本砍成固定的 1.2 倍」——成本是 $(4+2r)/3$，依保留比例 $r$ 與成本模型而變，單用 SDS 的下限是 $4/3$。
- 「ASAM 比 SAM 快」——ASAM 仍是兩次反傳，約 $2$ 倍；它改的是鄰域形狀，不是計算量。
- 「ASAM 的 $\rho$ 與 SAM 的 $\rho$ 可以直接比」——一個是相對權重大小的比例，一個是絕對長度。
- 「式 (56) 對任何可逆變換、任何 $\gamma$ 都恆成立」——對角的 $T_w$ 只對正的對角縮放成立，且 $\gamma>0$ 時只是近似。

**Common misconceptions**:

- "SWP is strictly unbiased, so the direction is never biased." Only the expectation of $\hat\epsilon$ equals $\epsilon^{\ast}$; when $\nabla L(w+\hat\epsilon)$ is nonlinear in $\hat\epsilon$ it is not guaranteed unbiased (20.5% off in the quartic example), and a single $\hat\epsilon$ can leave the $\rho$-ball.
- "SDS cuts the cost to a fixed 1.2 times." The cost is $(4+2r)/3$, depending on the kept fraction $r$ and the cost model; the floor of SDS alone is $4/3$.
- "ASAM is faster than SAM." ASAM still takes two backward passes, about $2$ times; it changes the shape of the neighbourhood, not the amount of computation.
- "The $\rho$ of ASAM and the $\rho$ of SAM can be compared directly." One is a proportion of the weight sizes, the other an absolute length.
- "Equation (56) holds for every invertible transformation and every $\gamma$." A diagonal $T_w$ works only for positive diagonal scalings, and only approximately when $\gamma>0$.

**圖**：本單元三張圖，`lec03-kp14-swp-unbiased`（遮罩擾動的平均與梯度的偏差，功能 1、2）、`lec03-kp14-sds-select`（樣本篩選與成本，功能 1、2）與 `lec03-kp14-asam-scale`（鄰域形狀與銳度隨尺度的變化，功能 1、2），以 `assets/figures/src/lec03-kp14-figures.py` 繪製，數字都由同一支程式印出。

**Figures**: Three figures in this unit, `lec03-kp14-swp-unbiased` (the mean of the masked perturbation and the bias of the gradient; functions 1, 2), `lec03-kp14-sds-select` (sample selection and cost; functions 1, 2) and `lec03-kp14-asam-scale` (the shape of the neighbourhood and the sharpness against the scale; functions 1, 2), drawn by `assets/figures/src/lec03-kp14-figures.py`, which prints every number from the same program.

**想深入**（選讀）：`[補充]` SWP 與 SDS 出自 ESAM（Efficient SAM，Du 等人，約 2021–2022 年；講義作者憑記憶標註，未逐字查證出處與實驗細節），ASAM 出自 Kwon 等人（ICML 2021，同樣未逐字查證）。實作上，SDS 也有改成「取漲幅最大的前一定比例」而不是固定門檻 $\alpha$ 的寫法，這一點未查證，請以原論文為準。另一個值得想的問題：ASAM 的銳度對正縮放近乎不變，但同一組權重是否「真的比較平坦」仍取決於損失本身，不變性只是排除了單位的干擾。

**Going deeper** (optional): `[Added]` SWP and SDS come from ESAM (Efficient SAM, Du et al., around 2021–2022; the author of this note cites it from memory and has not checked the venue and the experimental details word by word), and ASAM from Kwon et al. (ICML 2021, likewise unchecked). In practice SDS is also written as "take the top fraction by loss rise" instead of a fixed threshold $\alpha$; this is unchecked, so consult the original paper. A further question: the ASAM sharpness is nearly invariant under positive scaling, but whether a set of weights is "really flatter" still depends on the loss itself; invariance only removes the interference of units.

**小練習**：

1. $\beta=0.25$，$\epsilon^{\ast}=(0.2,\,-0.4,\,0.1)$：遮罩 $m=(1,0,1)$ 時 $\hat\epsilon$ 是多少？$\mathbb E\|\hat\epsilon\|^2$ 是多少？與 $\|\epsilon^{\ast}\|^2$ 差幾倍？（提示：$\hat\epsilon_i=m_i\epsilon^{\ast}_i/\beta$；$\mathbb E\|\hat\epsilon\|^2=\|\epsilon^{\ast}\|^2/\beta$。）
2. 一個 mini-batch 擾動後的五筆損失漲幅是 $(1.4,\,0.2,\,0.9,\,-0.1,\,0.6)$，$\alpha=0.5$：$\mathcal B^{+}$ 有幾筆？保留比例 $r$ 是多少？用成本模型（前向 1、反向 2）算每步是 SGD 的幾倍？（提示：$(4+2r)/3$。）
3. $w=(2,\,0.5)$，$g=(0.4,\,1.2)$，$\rho=0.1$，取 $\gamma=0$：寫出 $T_w$，算 ASAM 的 $\epsilon^{\ast}=\rho\,T_w^{2}g/\|T_wg\|$，並驗證 $\|T_w^{-1}\epsilon^{\ast}\|=\rho$；再算同樣 $\rho$ 的 SAM 擾動 $\rho\,g/\|g\|$ 比較。（提示：先算 $T_wg$。）

**Practice**:

1. $\beta=0.25$ and $\epsilon^{\ast}=(0.2,\,-0.4,\,0.1)$: what is $\hat\epsilon$ for the mask $m=(1,0,1)$? What is $\mathbb E\|\hat\epsilon\|^2$ and how many times is it $\|\epsilon^{\ast}\|^2$? (Hint: $\hat\epsilon_i=m_i\epsilon^{\ast}_i/\beta$; $\mathbb E\|\hat\epsilon\|^2=\|\epsilon^{\ast}\|^2/\beta$.)
2. After the perturbation the loss rises of five samples in a mini-batch are $(1.4,\,0.2,\,0.9,\,-0.1,\,0.6)$ and $\alpha=0.5$: how many samples are in $\mathcal B^{+}$? What is the kept fraction $r$? Using the cost model (forward 1, backward 2), how many times SGD is the cost per step? (Hint: $(4+2r)/3$.)
3. $w=(2,\,0.5)$, $g=(0.4,\,1.2)$, $\rho=0.1$, $\gamma=0$: write $T_w$, compute the ASAM perturbation $\epsilon^{\ast}=\rho\,T_w^{2}g/\|T_wg\|$, verify $\|T_w^{-1}\epsilon^{\ast}\|=\rho$, and compare with the SAM perturbation $\rho\,g/\|g\|$ at the same $\rho$. (Hint: compute $T_wg$ first.)

**解題示範**（玩具碗：$w=(10,\,2)$，$\rho=0.5$，$\beta=0.5$；算 $\epsilon^{\ast}$、遮罩 $m=(1,0)$ 的 $\hat\epsilon$，並驗證四種遮罩的期望）：

1. 梯度：$g=(x,\,25y)=(10,\,50)$，$\|g\|=\sqrt{10^2+50^2}=\sqrt{2600}=50.990$。
2. SAM 擾動：$\epsilon^{\ast}=\rho\,g/\|g\|=0.5\cdot(10,\,50)/50.990=(0.0981,\ 0.4903)$。
3. 遮罩 $m=(1,0)$：$\hat\epsilon=m\odot\epsilon^{\ast}/\beta=(1\cdot0.0981/0.5,\ 0)=(0.1961,\ 0)$，擾動後的點是 $(10.196,\ 2)$，梯度是 $(10.196,\ 50)$；對照 $\nabla f(w+\epsilon^{\ast})=(10.098,\ 62.257)$，單次的梯度與完整 SAM 的梯度差很多（$y$ 分量 $50$ 對 $62.26$）。
4. 四種遮罩各機率 $\beta^{k}(1-\beta)^{2-k}=0.25$：$\hat\epsilon$ 是 $(0,0),\ (0.1961,0),\ (0,0.9806),\ (0.1961,0.9806)$。
5. 期望：$x$ 分量 $0.25\,(0+0.1961+0+0.1961)=0.0981$；$y$ 分量 $0.25\,(0+0+0.9806+0.9806)=0.4903$，等於 $\epsilon^{\ast}$。
6. 梯度的期望：碗的梯度是線性的，$\mathbb E[\nabla f(w+\hat\epsilon)]=\nabla f(w+\mathbb E[\hat\epsilon])=\nabla f(w+\epsilon^{\ast})=(10.098,\,62.257)$；直接平均四個梯度 $(10.196,74.51),(10.196,50),(10,74.51),(10,50)$ 也得到同一個值。
7. 長度：$\mathbb E\|\hat\epsilon\|^2=0.25\,(0+0.0385+0.9616+1.0)=0.5=\rho^2/\beta$；最長的一格 $\|\hat\epsilon\|=1.0=\rho/\beta$，超出半徑 $0.5$ 的球〔程式計算〕。
8. 對照：若函數不是二次的（四次函數例），第 6 步的等號不成立，差 $20.5\%$。

**Worked solution** (toy bowl: $w=(10,\,2)$, $\rho=0.5$, $\beta=0.5$; compute $\epsilon^{\ast}$, $\hat\epsilon$ for the mask $m=(1,0)$, and verify the expectation over the four masks):

1. Gradient: $g=(x,\,25y)=(10,\,50)$, $\|g\|=\sqrt{10^2+50^2}=\sqrt{2600}=50.990$.
2. SAM perturbation: $\epsilon^{\ast}=\rho\,g/\|g\|=0.5\cdot(10,\,50)/50.990=(0.0981,\ 0.4903)$.
3. Mask $m=(1,0)$: $\hat\epsilon=m\odot\epsilon^{\ast}/\beta=(1\cdot0.0981/0.5,\ 0)=(0.1961,\ 0)$; the perturbed point is $(10.196,\ 2)$ with gradient $(10.196,\ 50)$; compare with $\nabla f(w+\epsilon^{\ast})=(10.098,\ 62.257)$: a single draw gives a gradient far from that of full SAM ($50$ against $62.26$ in the $y$ component).
4. The four masks each have probability $\beta^{k}(1-\beta)^{2-k}=0.25$: $\hat\epsilon$ is $(0,0),\ (0.1961,0),\ (0,0.9806),\ (0.1961,0.9806)$.
5. Expectation: $x$ component $0.25\,(0+0.1961+0+0.1961)=0.0981$; $y$ component $0.25\,(0+0+0.9806+0.9806)=0.4903$, equal to $\epsilon^{\ast}$.
6. Expected gradient: the gradient of the bowl is linear, so $\mathbb E[\nabla f(w+\hat\epsilon)]=\nabla f(w+\mathbb E[\hat\epsilon])=\nabla f(w+\epsilon^{\ast})=(10.098,\,62.257)$; averaging the four gradients $(10.196,74.51),(10.196,50),(10,74.51),(10,50)$ directly gives the same value.
7. Length: $\mathbb E\|\hat\epsilon\|^2=0.25\,(0+0.0385+0.9616+1.0)=0.5=\rho^2/\beta$; the longest case has $\|\hat\epsilon\|=1.0=\rho/\beta$, outside the ball of radius $0.5$ [computed by program].
8. Contrast: for a function that is not quadratic (the quartic example) the equality in step 6 fails, with a $20.5\%$ difference.

### 來源更正與取捨（Corrections to the source）

- 來源說 SWP「期望嚴格無偏、少算權重不失方向」「在期望上與 Vanilla SAM 完全一致」。經驗證，只有 $\mathbb E[\hat\epsilon]=\epsilon^{\ast}$ 成立；梯度 $\nabla L(w+\hat\epsilon)$ 是 $\hat\epsilon$ 的非線性函數時，期望梯度不等於 $\nabla L(w+\epsilon^{\ast})$（四次函數例差 $20.5\%$，碗因梯度線性而剛好相等），單次 $\hat\epsilon$ 的長度也可達 $\rho/\beta$。本講義改為限縮後的說法。
- 來源把式 (54) 寫成 $\tfrac1\beta m^{\top}\epsilon^{\ast}$，$m^{\top}\epsilon^{\ast}$ 是內積（純量）；本講義改為逐元素相乘 $\tfrac1\beta m\odot\epsilon^{\ast}$，來源在註解裡也這樣寫。
- 來源說 SWP 引入「類似 Dropout 的隨機正則化效應」，並說 SDS「節省約 50%–70% 樣本的反向傳播」「合併後 $1.2$–$1.3$ 倍」，圖中的「縮放前銳度 0.40、縮放後 0.95」也是示意數字。這些沒有可驗證的依據，不採用；本講義的銳度數字（$0.346$、$2.069$）、成本比例（$(4+2r)/3$）都由自己的程式計算，並標明是成本模型。SDS 的閾值 $\alpha$ 與實際節省比例屬超參數與經驗，不給固定倍數。
- 來源的式 (55) 把漲幅寫成整批的 $\mathcal L(w+\epsilon^{\ast})-\mathcal L(w)$，卻用它來挑樣本；本講義改為逐樣本的 $\ell_i(w+\epsilon^{\ast})-\ell_i(w)$，並補上等於 $\alpha$ 時歸入 $\mathcal B^{-}$。第二次梯度只在 $\mathcal B^{+}$ 上算，因而不再是完整 SAM 梯度的無偏估計，這是本講義的補充說明 `[補充]`，來源沒有提到。
- 來源說 $T_{Aw}=AT_w$ 對 $T_w=\mathrm{diag}(|w|+\gamma)$ 恆成立，因而式 (56) 對「任意可逆線性縮放矩陣」成立。經驗證，只有 $\gamma=0$ 且 $A$ 是正的對角矩陣時才精確成立；$\gamma>0$ 時只是近似（例：最大差 $0.029$，ASAM 銳度跨尺度差 $1.065$ 倍，不是「誤差 0.000」）。本講義改為條件式。
- 來源比較 SAM 與 ASAM 時用同樣的 $\rho$ 符號；ASAM 的 $\rho$ 是相對權重大小的比例，與 SAM 的絕對長度 $\rho$ 不能直接比較，本講義在兩處明寫。來源在表格中說 ASAM 適用於「含 BatchNorm、通道縮放敏感的深層 CNN」、ESAM 適用於「大型 Transformer / ViT」，這些是經驗說法，不當成事實採用。
- 來源把 Vanilla SAM、ESAM、ASAM 的年份寫成 2020、2021、2021，並說 SAM「從實驗室玩具躍升為工業級標配」。年份與評價都沒有在本講義查證，不採用；ESAM 與 ASAM 的出處依 `[補充]` 標註，未逐字查證。
- 來源的成本欄「ASAM 約 $2.0\times$」與本講義的「兩次前向加兩次反向」一致，保留。

- The source says SWP is "strictly unbiased in expectation, losing no direction by computing fewer weights" and "identical to vanilla SAM in expectation". Verification shows that only $\mathbb E[\hat\epsilon]=\epsilon^{\ast}$ holds; when $\nabla L(w+\hat\epsilon)$ is nonlinear in $\hat\epsilon$ the expected gradient differs from $\nabla L(w+\epsilon^{\ast})$ (20.5% off in the quartic example; the bowl happens to agree because its gradient is linear), and a single $\hat\epsilon$ can be as long as $\rho/\beta$. This note uses the narrowed statement.
- The source writes equation (54) as $\tfrac1\beta m^{\top}\epsilon^{\ast}$, where $m^{\top}\epsilon^{\ast}$ is an inner product (a scalar); this note uses the element-wise product $\tfrac1\beta m\odot\epsilon^{\ast}$, as the source's own remark does.
- The source says SWP brings a "Dropout-like random regularization effect", that SDS "saves the backward pass of about 50%–70% of the samples" and that the combination costs "$1.2$–$1.3$ times"; the "sharpness 0.40 before and 0.95 after scaling" in its figure is also an illustrative number. None of these has a verifiable basis and they are not adopted; the sharpness values here ($0.346$, $2.069$) and the cost ratio ($(4+2r)/3$) are computed by this note's own program and labelled as a cost model. The SDS threshold $\alpha$ and the actual saving are hyperparameters and empirical, so no fixed multiple is given.
- The source's equation (55) writes the rise as the batch quantity $\mathcal L(w+\epsilon^{\ast})-\mathcal L(w)$ yet uses it to select samples; this note writes the per-sample $\ell_i(w+\epsilon^{\ast})-\ell_i(w)$ and assigns equality with $\alpha$ to $\mathcal B^{-}$. Because the second gradient is computed on $\mathcal B^{+}$ only, it is no longer an unbiased estimate of the full SAM gradient; this is this note's own addition `[Added]` and the source does not mention it.
- The source says $T_{Aw}=AT_w$ always holds for $T_w=\mathrm{diag}(|w|+\gamma)$ and so (56) holds for "any invertible linear scaling matrix". Verification shows it holds exactly only for $\gamma=0$ and a positive diagonal $A$; for $\gamma>0$ it is approximate (for example a maximum difference of $0.029$, and the ASAM sharpness varies by a factor of $1.065$ across scales, not "error 0.000"). This note states the conditions.
- The source uses the same symbol $\rho$ for SAM and ASAM; the $\rho$ of ASAM is a proportion of the weight sizes and cannot be compared directly with the absolute length $\rho$ of SAM; this note says so in two places. The source's table says ASAM suits "deep CNNs with BatchNorm and sensitivity to channel scaling" and ESAM suits "large Transformers / ViT"; these are empirical claims and are not adopted as facts.
- The source dates vanilla SAM, ESAM and ASAM as 2020, 2021 and 2021 and says SAM "leaps from a laboratory toy to an industrial-grade standard". The dates and the judgement are not verified here and not adopted; the origins of ESAM and ASAM are marked `[Added]` and not checked word by word.
- The source's cost row, "ASAM about $2.0\times$", agrees with this note's "two forward and two backward passes" and is kept.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 隨機權重擾動（SWP） | stochastic weight perturbation (SWP) |
| 銳度敏感資料篩選（SDS） | sharpness-sensitive data selection (SDS) |
| 自適應 SAM（ASAM） | adaptive SAM (ASAM) |
| 白努利分佈 | Bernoulli distribution |
| 遮罩 | mask |
| 逐元素相乘 | element-wise (Hadamard) product |
| 損失漲幅 | loss rise |
| 保留比例 | kept fraction |
| 成本模型 | cost model |
| 正規化矩陣 | normalization matrix |
| 尺度不變性 | scale invariance |
| 重新縮放 | rescaling |
| 鄰域（擾動球、橢球） | neighbourhood (perturbation ball, ellipsoid) |
