# 第4週 KP09：L1 正則化與軟閾值（L1 regularization and the soft threshold）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第4章「正則化與過擬合」的第九個知識點（KP09），承接 KP08 留下的問題：$\ell_2$ 的每個收縮因子都大於 $0$，不會產生精確的零，想要稀疏解怎麼辦？先備知識是 KP07 的 $\ell_0$、$\ell_1$、$\ell_2$ 與稀疏性，KP08 的收縮，以及單變數微分（絕對值在 $0$ 不可微，會用到次梯度）。本講義的內容取自 CCC-LNS 第5章 KP13 主文與教授祕笈，讀取日期 2026-10-20；來源沒有的補充一律標記 `[補充]`。

This is the ninth knowledge point (KP09) of Chapter 4. It picks up the question KP08 left open: every shrink factor of $\ell_2$ is above $0$ and never produces an exact zero, so what if we want a sparse solution? It assumes $\ell_0$, $\ell_1$, $\ell_2$ and sparsity from KP07, the shrinkage of KP08, and single-variable differentiation (the absolute value is not differentiable at $0$, so a subgradient is used). The content is taken from CCC-LNS Chapter 5, KP09 main text and the professor's guide, read on 2026-10-20; everything not in the source is marked `[Added]`.

[下載本單元投影片](../slides/lec04-kp09-l1-regularization-soft-threshold.jbd.json)

## KP09 L1 正則化與軟閾值（L1 regularization and the soft threshold）

### 學習目標（Learning objectives）

讀完之後，你能說出為什麼用 $\ell_1$ 取代 $\ell_0$（凸鬆弛），並寫出 lasso 目標；能用次梯度推出單一座標的軟閾值公式 $x^{\dagger}=\mathrm{sign}(x^{\ast})\max(|x^{\ast}|-\alpha/h,\,0)$ 並算出數值例；能用菱形與圓形的約束圖說明 $\ell_1$ 為什麼容易產生精確的零，也知道這只是「容易」而不是「一定」；並知道軟閾值公式的前提是 Hessian 為對角矩陣。

After reading, you can say why $\ell_1$ replaces $\ell_0$ (convex relaxation) and write the lasso objective; derive the one-coordinate soft-threshold formula $x^{\dagger}=\mathrm{sign}(x^{\ast})\max(|x^{\ast}|-\alpha/h,\,0)$ with a subgradient and compute numerical examples; use the diamond and circle constraint picture to explain why $\ell_1$ readily produces exact zeros, knowing that this is "likely", not "certain"; and know that the soft-threshold formula assumes a diagonal Hessian.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\|\mathbf{x}\|_0,\ \|\mathbf{x}\|_1$ | 非零個數；絕對值和 | number of nonzeros; sum of absolute values |
| $\alpha>0$ | $\ell_1$ 懲罰的強度 | strength of the $\ell_1$ penalty |
| $x_j^{\ast}$, $x_j^{\dagger}$ | 第 $j$ 個座標：無正則化的解；$\ell_1$ 正則化後的解 | the $j$-th coordinate: the unregularized solution; the solution after $\ell_1$ regularization |
| $h_j>0$ | $J$ 在第 $j$ 個座標的二階導數（Hessian 對角線的元素） | second derivative of $J$ along the $j$-th coordinate (an entry of the Hessian diagonal) |
| $\alpha/h_j$ | 軟閾值的門檻 | the threshold of the soft threshold |
| $\mathrm{sign}(x)$ | 正數為 $+1$、負數為 $-1$、零為 $0$ | $+1$ for positive, $-1$ for negative, $0$ at zero |
| $\partial\lvert x\rvert$ | $\lvert x\rvert$ 的次梯度集合：$x\neq0$ 時是 $\{\mathrm{sign}(x)\}$，$x=0$ 時是區間 $[-1,1]$ | the subdifferential of $\lvert x\rvert$: $\{\mathrm{sign}(x)\}$ for $x\neq0$ and the interval $[-1,1]$ at $x=0$ |

### 1. 從 $\ell_0$ 到 $\ell_1$（From $\ell_0$ to $\ell_1$）

KP08 的 $\ell_2$ 只縮小、不歸零，所以不會自動挑出少數有用的特徵。最理想的稀疏懲罰，其實是 KP07 介紹過的 $\ell_0$：數有幾個座標不是零。可是它有兩個麻煩。第一，它不是真正的範數：把 $\mathbf{x}$ 放大兩倍，$\|2\mathbf{x}\|_0=\|\mathbf{x}\|_0$，不滿足範數要求的齊次性。第二，它不連續也不是凸函數，來源指出最小化它是 NP-hard，也就是特徵很多時基本上要把所有「哪些座標非零」的組合都試一遍。

The $\ell_2$ of KP08 only shrinks and never zeroes, so it does not pick out the few useful features by itself. The ideal sparsity penalty is actually the $\ell_0$ of KP07: count how many coordinates are not zero. But it has two problems. First, it is not a true norm: scaling $\mathbf{x}$ by two gives $\|2\mathbf{x}\|_0=\|\mathbf{x}\|_0$, violating the homogeneity a norm requires. Second, it is neither continuous nor convex, and the source states that minimizing it is NP-hard, that is, with many features one essentially has to try every combination of "which coordinates are nonzero".

解法是用最接近它的**凸**函數代替，叫做**凸鬆弛**（convex relaxation）：$\ell_1$ 範數。於是得到 lasso（least absolute shrinkage and selection operator，最小絕對值收縮與選擇算子）：

The remedy is to replace it by the closest **convex** function, called a **convex relaxation**: the $\ell_1$ norm. This gives the lasso (least absolute shrinkage and selection operator):

$$
\underset{\mathbf{x}}{\operatorname{minimize}}\;\tilde J(\mathbf{x};\theta):=J(\mathbf{x};\theta)+\alpha\|\mathbf{x}\|_1
$$

$\ell_1$ 是凸函數，所以 $J$ 為凸時整個目標也是凸的，有可靠的算法；代價是它在 $x_j=0$ 有一個尖角（不可微）。這個尖角是它能產生精確零的原因，下一節把它算出來。

$\ell_1$ is convex, so when $J$ is convex the whole objective is convex and reliable algorithms exist; the price is a sharp corner (non-differentiable) at $x_j=0$. That corner is the reason it can produce exact zeros, which the next section works out.

### 2. 單一座標的軟閾值（The soft threshold for one coordinate）

承接上一節：先看最簡單的情形，只有一個座標，$J$ 的二次近似是 $\tfrac12 h(x-x^{\ast})^2$，$h>0$，加上 $\ell_1$ 懲罰：

Continuing from the previous section: take the simplest case, one coordinate, where $J$ has the quadratic form $\tfrac12 h(x-x^{\ast})^2$ with $h>0$, plus the $\ell_1$ penalty:

$$
\tilde J(x)=\tfrac12 h\,(x-x^{\ast})^2+\alpha|x|
$$

最小點要滿足「$0$ 屬於次梯度」，也就是 $h(x-x^{\ast})+\alpha s=0$，其中 $s\in\partial|x|$。分三種情況：

At the minimum, $0$ must belong to the subdifferential, that is, $h(x-x^{\ast})+\alpha s=0$ with $s\in\partial|x|$. Three cases:

- $x>0$：$s=1$，得 $x=x^{\ast}-\alpha/h$；要與 $x>0$ 一致，必須 $x^{\ast}>\alpha/h$。
- $x<0$：$s=-1$，得 $x=x^{\ast}+\alpha/h$；要與 $x<0$ 一致，必須 $x^{\ast}<-\alpha/h$。
- $x=0$：$s$ 可以是 $[-1,1]$ 裡任何數，條件變成 $-hx^{\ast}+\alpha s=0$ 有解，也就是 $|hx^{\ast}|\le\alpha$，即 $|x^{\ast}|\le\alpha/h$。

- $x>0$: $s=1$, so $x=x^{\ast}-\alpha/h$; to agree with $x>0$ we need $x^{\ast}>\alpha/h$.
- $x<0$: $s=-1$, so $x=x^{\ast}+\alpha/h$; to agree with $x<0$ we need $x^{\ast}<-\alpha/h$.
- $x=0$: $s$ may be any number in $[-1,1]$, and the condition $-hx^{\ast}+\alpha s=0$ is solvable exactly when $|hx^{\ast}|\le\alpha$, that is, $|x^{\ast}|\le\alpha/h$.

三種情況合起來就是**軟閾值**（soft threshold）：

The three cases together are the **soft threshold**:

$$
x^{\dagger}=\begin{cases}
x^{\ast}-\alpha/h, & x^{\ast}>\alpha/h,\\
0, & |x^{\ast}|\le\alpha/h,\\
x^{\ast}+\alpha/h, & x^{\ast}<-\alpha/h,
\end{cases}
\qquad\text{即}\qquad x^{\dagger}=\mathrm{sign}(x^{\ast})\max\!\big(|x^{\ast}|-\alpha/h,\ 0\big)
$$

白話說：門檻 $\alpha/h$ 以內的係數被「關掉」變成零；門檻以外的係數整個往零平移 $\alpha/h$，方向不變。用 $h=2$、$\alpha=1$（門檻 $0.5$）算：

In words: coefficients inside the threshold $\alpha/h$ are "switched off" to zero; coefficients outside are shifted toward zero by $\alpha/h$, keeping their sign. With $h=2$, $\alpha=1$ (threshold $0.5$):

| $x^{\ast}$ | 判斷 | $x^{\dagger}$ | 對照：$\ell_2$（KP08）$\dfrac{h}{h+\alpha}x^{\ast}$ |
|---|---|---|---|
| $1.2$ | $1.2>0.5$：減去 $0.5$ | $0.7$ | $0.8$ |
| $0.3$ | $0.3\le0.5$：關掉 | $0$ | $0.2$ |
| $-0.8$ | $-0.8<-0.5$：加上 $0.5$ | $-0.3$ | $-0.5333$ |

| $x^{\ast}$ | decision | $x^{\dagger}$ | for comparison: $\ell_2$ (KP08) $\dfrac{h}{h+\alpha}x^{\ast}$ |
|---|---|---|---|
| $1.2$ | $1.2>0.5$: subtract $0.5$ | $0.7$ | $0.8$ |
| $0.3$ | $0.3\le0.5$: switch off | $0$ | $0.2$ |
| $-0.8$ | $-0.8<-0.5$: add $0.5$ | $-0.3$ | $-0.5333$ |

三個值都用程式在細網格上直接最小化 $\tfrac12h(x-x^{\ast})^2+\alpha|x|$ 驗證，網格最小點分別是 $0.7000$、$0.0000$、$-0.3000$〔程式計算〕。

All three values were checked by minimizing $\tfrac12h(x-x^{\ast})^2+\alpha|x|$ directly on a fine grid; the grid minima are $0.7000$, $0.0000$ and $-0.3000$ [computed by program].

![軟閾值函數：橘色折線在 −0.5 到 0.5 之間是水平的零，之外平行於對角虛線但低 0.5；藍線是 ℓ2 的等比例縮小；三個綠點是 1.2→0.7、0.3→0、−0.8→−0.3（The soft-threshold function: the orange broken line is flat at zero between −0.5 and 0.5 and parallel to the dashed diagonal, 0.5 below it, outside; the blue line is the proportional shrinking of ℓ2; the three green points are 1.2→0.7, 0.3→0, −0.8→−0.3）](../assets/figures/lec04-kp09-softthreshold.png)

**圖說 `lec04-kp09-softthreshold`**：先看橘色折線中間那一段平台：兩條金色虛線（$\pm\alpha/h=\pm0.5$）之間，不管 $x^{\ast}$ 是多少，輸出都是 $0$，這是 $\ell_1$ 的「死區」。再看平台之外：橘線與對角虛線平行，差一個固定的 $0.5$，而藍色的 $\ell_2$ 是一條斜率小於 $1$ 的直線，沒有死區。三個綠點對應上表的三列。〔功能：1、2〕

**Caption `lec04-kp09-softthreshold`**: Look first at the flat stretch in the middle of the orange broken line: between the two golden dotted lines ($\pm\alpha/h=\pm0.5$) the output is $0$ whatever $x^{\ast}$ is, the "dead zone" of $\ell_1$. Then look outside the plateau: the orange line is parallel to the dashed diagonal at a fixed offset of $0.5$, while the blue $\ell_2$ is a straight line with slope below $1$ and has no dead zone. The three green points are the three rows of the table. [Function: 1, 2]

這也說明 $\ell_1$ 與 $\ell_2$ 的另一個差別：$\ell_2$ 對大係數按比例縮得多，$\ell_1$ 對所有門檻外的係數都減去同一個 $\alpha/h$。至於一個多座標的問題裡，零是怎麼出現的？下一節換成幾何圖來看。

This also shows another difference between $\ell_1$ and $\ell_2$: $\ell_2$ shrinks a large coefficient proportionally more, while $\ell_1$ subtracts the same $\alpha/h$ from every coefficient outside the threshold. How do zeros appear in a problem with several coordinates? The next section switches to a geometric picture.

### 3. 菱形與圓形的幾何，以及公式的前提（The diamond and the circle, and what the formula assumes）

承接上一節：多個座標時，可以把「最小化 $J+\alpha\|\mathbf{x}\|_1$」換成等價的限制形式「在 $\|\mathbf{x}\|_1\le r$ 之內最小化 $J$」（對偶關係，與 KP07 的「想深入」同一件事，$\alpha$ 越大等價的 $r$ 越小）。`[補充]` 限制區域在兩個座標時是一個**菱形**；$\ell_2$ 的對應區域是**圓**。$J$ 的等高線是橢圓，最小點就是「從 $\mathbf{x}^{\ast}$ 出發長大的橢圓第一次碰到限制區域」的那一點。

Continuing from the previous section: with several coordinates, "minimize $J+\alpha\|\mathbf{x}\|_1$" can be replaced by the equivalent constrained form "minimize $J$ within $\|\mathbf{x}\|_1\le r$" (a duality relation, the same idea as the "going deeper" paragraph of KP07; the larger $\alpha$, the smaller the equivalent $r$). `[Added]` With two coordinates the constraint region is a **diamond**; the corresponding region of $\ell_2$ is a **circle**. The contours of $J$ are ellipses, and the minimum is the point where the ellipse growing out of $\mathbf{x}^{\ast}$ first touches the constraint region.

取 $\mathbf{x}^{\ast}=(1.6,\,0.9)$、$\mathbf{H}=\begin{pmatrix}2&1.2\\1.2&1.5\end{pmatrix}$、$r=1$。程式沿邊界掃描得到：$\ell_1$ 菱形上的最小點是 $(1.0000,\,0.0000)$（$J=1.6155$），剛好是菱形落在座標軸上的頂點，所以第二個座標是**精確的零**；$\ell_2$ 圓上的最小點是 $(0.8097,\,0.5869)$（$J=0.9951$），兩個座標都不是零〔程式計算〕。

Take $\mathbf{x}^{\ast}=(1.6,\,0.9)$, $\mathbf{H}=\begin{pmatrix}2&1.2\\1.2&1.5\end{pmatrix}$ and $r=1$. A boundary scan by program gives: the minimum on the $\ell_1$ diamond is $(1.0000,\,0.0000)$ ($J=1.6155$), exactly the vertex of the diamond on a coordinate axis, so the second coordinate is an **exact zero**; the minimum on the $\ell_2$ circle is $(0.8097,\,0.5869)$ ($J=0.9951$), with neither coordinate zero [computed by program].

![兩種約束區域與同一組損失等高線：左圖圓形的接觸點是 (0.81, 0.59)，兩個座標都不為零；右圖菱形的接觸點是頂點 (1.00, 0.00)，第二個座標為零（The same loss contours with two constraint regions: on the left the circle touches at (0.81, 0.59), neither coordinate zero; on the right the diamond touches at its vertex (1.00, 0.00), second coordinate zero）](../assets/figures/lec04-kp09-diamond.png)

**圖說 `lec04-kp09-diamond`**：先看左右兩張圖共用的藍色橢圓等高線與中心 $\mathbf{x}^{\ast}$，再看橘色那一圈：它是剛好碰到限制區域的等高線，接觸點是橘色圓點。左圖的圓是平滑的，接觸點落在弧的中間；右圖菱形有尖角，等高線先碰到尖角，圓點落在座標軸上。這是空間幾何的直觀：尖角把解「吸」到軸上。〔功能：1、3〕

**Caption `lec04-kp09-diamond`**: Look first at the blue elliptical contours shared by both panels and their centre $\mathbf{x}^{\ast}$, then at the orange contour: it is the contour that just touches the constraint region, and the orange dot is the touching point. The circle on the left is smooth and the contact lies in the middle of the arc; the diamond on the right has corners, the contour hits a corner first, and the dot sits on a coordinate axis. This is the spatial intuition: the corner "pulls" the solution onto an axis. [Function: 1, 3]

但這只是「容易」，不是「一定」。同一個 $\mathbf{x}^{\ast}$、只把 $\mathbf{H}$ 換成另一個正定矩陣（$\mathbf{H}=\begin{pmatrix}1.5&-0.9\\-0.9&1\end{pmatrix}$，特徵值從約 $0.52,\ 2.98$ 變成 $0.32,\ 2.18$，橢圓的形狀與方向都改變），菱形上的最小點變成 $(0.9372,\,0.0628)$，不在軸上〔程式計算〕。再用 $2819$ 組隨機的 $\mathbf{x}^{\ast}$ 與正定 $\mathbf{H}$（固定種子，且 $\|\mathbf{x}^{\ast}\|_1>1$）統計：$\ell_1$ 的最小點剛好落在軸上的有 $62.6\%$，$\ell_2$ 的最小點落在軸上的只有 $0.1\%$（網格解析度造成的邊界情形，理論上機率為 $0$）〔程式計算〕。所以正確的說法是：**$\ell_1$ 的尖角讓精確零變得很常見，$\ell_2$ 幾乎不會**。

But this is "likely", not "certain". With the same $\mathbf{x}^{\ast}$ and only $\mathbf{H}$ changed to another positive-definite matrix ($\mathbf{H}=\begin{pmatrix}1.5&-0.9\\-0.9&1\end{pmatrix}$, whose eigenvalues change from about $0.52,\ 2.98$ to $0.32,\ 2.18$, so both the shape and the orientation of the ellipse change), the minimum on the diamond becomes $(0.9372,\,0.0628)$, not on an axis [computed by program]. Over $2819$ random pairs of $\mathbf{x}^{\ast}$ and positive-definite $\mathbf{H}$ (fixed seed, with $\|\mathbf{x}^{\ast}\|_1>1$), the $\ell_1$ minimum lies exactly on an axis in $62.6\%$ of the cases and the $\ell_2$ minimum in only $0.1\%$ (boundary cases from the grid resolution; in theory the probability is $0$) [computed by program]. So the correct statement is: **the corner of $\ell_1$ makes exact zeros very common, while $\ell_2$ almost never produces them**.

最後說軟閾值公式的前提：上一節的推導假設 Hessian 是**對角矩陣**，各座標互不牽連。一般的 $\mathbf{H}$ 不是對角，這時把每個座標各做一次軟閾值**不是**正確答案。上面的例子（$\alpha=1$）：逐座標軟閾值，門檻用對角線元素 $\alpha/h_{jj}=0.5,\ 0.667$，得 $(1.1,\,0.2333)$，目標值 $2.3167$；真正的 lasso 解是 $(1.4077,\,0.3872)$，目標值 $2.1474$，更低〔程式計算〕。也不能像 KP08 那樣轉到特徵方向：$\ell_2$ 轉了座標不變，$\ell_1$ 卻會變（$\mathbf{v}=(3,0)$ 的 $\ell_1$ 是 $3$，轉 $30^{\circ}$ 後變成 $4.0981$，$\ell_2$ 仍是 $3$），菱形的角被綁在座標軸上。`[補充]` 實務上用的是反覆做這種逐座標更新直到穩定（座標下降法）；本單元只需要知道公式在對角情形成立。

Finally, what the soft-threshold formula assumes: the derivation of the previous section assumes the Hessian is **diagonal**, with the coordinates not coupled. A general $\mathbf{H}$ is not diagonal, and then soft-thresholding each coordinate once is **not** the right answer. In the example above ($\alpha=1$), coordinate-wise soft thresholds with the diagonal entries $\alpha/h_{jj}=0.5,\ 0.667$ give $(1.1,\,0.2333)$ with objective $2.3167$, while the true lasso solution is $(1.4077,\,0.3872)$ with the lower objective $2.1474$ [computed by program]. Nor can we rotate into the eigen-directions as in KP08: $\ell_2$ is unchanged by rotating the coordinates, $\ell_1$ is not ($\ell_1$ of $\mathbf{v}=(3,0)$ is $3$ and becomes $4.0981$ after a $30^{\circ}$ rotation, while $\ell_2$ stays $3$); the corners of the diamond are tied to the coordinate axes. `[Added]` In practice one repeats this coordinate-wise update until it settles (coordinate descent); this unit only needs the fact that the formula holds in the diagonal case.

還有一個實務提醒：$\alpha$ 越大零越多，但不是越大越好，太大會把真正有用的特徵也關掉；$\alpha$ 要像 KP07 說的用驗證集挑。

One practical reminder: the larger $\alpha$, the more zeros, but larger is not better; too large an $\alpha$ switches off useful features as well, and $\alpha$ should be chosen on a validation set as KP07 said.

接下來的問題：$\ell_2$ 與 $\ell_1$ 都只是在一般的目標上加一項，都是靠改目標函數來限制模型。能不能完全不改目標，只靠少訓練幾步，就得到類似的效果？這是 KP10 的主題。

The next question: $\ell_2$ and $\ell_1$ both limit the model by changing the objective. Can we leave the objective alone and get a similar effect just by training fewer steps? That is the topic of KP10.

### 本單元重點整理（KP09 summary）

**一句話重點**：$\ell_0$ 難算，所以用凸鬆弛 $\ell_1$；在 Hessian 為對角的情形，每個座標做軟閾值 $x^{\dagger}=\mathrm{sign}(x^{\ast})\max(|x^{\ast}|-\alpha/h,\,0)$，門檻內變精確的零；幾何上菱形的尖角在座標軸上，讓零很常見。

**One-sentence summary**: $\ell_0$ is hard to compute, so use the convex relaxation $\ell_1$; with a diagonal Hessian each coordinate is soft-thresholded, $x^{\dagger}=\mathrm{sign}(x^{\ast})\max(|x^{\ast}|-\alpha/h,\,0)$, and coefficients inside the threshold become exact zeros; geometrically the corners of the diamond lie on the coordinate axes and make zeros common.

**白話說明**：想像一排開關。$\ell_0$ 是直接數有幾個開著；$\ell_1$ 像一個智慧開關：電流小於門檻的設備直接斷電（零），夠大的設備繼續用，但每個都降一點電壓（減去同樣的 $\alpha/h$）。

**Plain explanation**: Picture a row of switches. $\ell_0$ counts how many are on; $\ell_1$ is a smart switch: a device drawing less than the threshold is cut off (zero), and a device above it keeps running but with a little less voltage (the same $\alpha/h$ subtracted from each).

**正式定義**：$\tilde J(\mathbf{x};\theta)=J(\mathbf{x};\theta)+\alpha\|\mathbf{x}\|_1$；$h>0$ 的單一座標：$x^{\dagger}=\mathrm{sign}(x^{\ast})\max(|x^{\ast}|-\alpha/h,\,0)$。

**Formal definition**: $\tilde J(\mathbf{x};\theta)=J(\mathbf{x};\theta)+\alpha\|\mathbf{x}\|_1$; for one coordinate with $h>0$: $x^{\dagger}=\mathrm{sign}(x^{\ast})\max(|x^{\ast}|-\alpha/h,\,0)$.

**例子**：$h=2$、$\alpha=1$（門檻 $0.5$）：$x^{\ast}=1.2,\ 0.3,\ -0.8$ 變成 $0.7,\ 0,\ -0.3$。逐步計算見第 2 節。

**Example**: $h=2$, $\alpha=1$ (threshold $0.5$): $x^{\ast}=1.2,\ 0.3,\ -0.8$ become $0.7,\ 0,\ -0.3$. The computations are in section 2.

**常見迷思**：

- 「$\ell_1$ 只是把權重縮小」——它會產生精確的零；$\ell_2$ 才是只縮小。
- 「菱形的接觸點一定在軸上」——只是常見：換一個橢圓方向，接觸點可以離開軸（第 3 節）。
- 「軟閾值公式對任何 $\mathbf{H}$ 都成立」——只在 Hessian 為對角時成立。
- 「$\alpha$ 越大越好」——零越多不代表越好，太大會關掉有用的特徵。

**Common misconceptions**:

- "$\ell_1$ just shrinks the weights." It produces exact zeros; it is $\ell_2$ that only shrinks.
- "The diamond always touches on an axis." It is common but not certain: another orientation of the ellipse moves the touching point off the axis (section 3).
- "The soft-threshold formula holds for any $\mathbf{H}$." It holds only when the Hessian is diagonal.
- "The larger $\alpha$ the better." More zeros does not mean better; too large switches off useful features.

**圖**：本單元兩張圖，`lec04-kp09-softthreshold`（軟閾值與 $\ell_2$ 比較，功能 1、2）與 `lec04-kp09-diamond`（菱形與圓形約束下的最小點，功能 1、3），以 `assets/figures/src/lec04-kp09-figures.py` 繪製。

**Figure**: Two figures in this unit, `lec04-kp09-softthreshold` (the soft threshold compared with $\ell_2$; functions 1, 2) and `lec04-kp09-diamond` (the minima under the diamond and the circle; functions 1, 3), drawn by `assets/figures/src/lec04-kp09-figures.py`.

**想深入**（選讀）：$\ell_1$ 之所以是 $\ell_0$ 「最好的凸近似」，有一個精確的說法：在每個座標都落在 $[-1,1]$ 的範圍內，$\|\mathbf{x}\|_1$ 是 $\|\mathbf{x}\|_0$ 的凸包絡（最大的凸下界函數）。座標超出這個範圍時要先縮放才適用，這也是實務上先把特徵標準化的理由之一。`[補充]`

**Going deeper** (optional): There is a precise sense in which $\ell_1$ is the "best convex approximation" of $\ell_0$: on the range where every coordinate lies in $[-1,1]$, $\|\mathbf{x}\|_1$ is the convex envelope (the largest convex lower bound) of $\|\mathbf{x}\|_0$. Coordinates outside that range must be rescaled first for this to apply, one of the reasons why features are standardized in practice. `[Added]`

**小練習**：

1. $h=4$、$\alpha=3$：門檻是多少？$x^{\ast}=2,\ 0.5,\ -1$ 分別變成多少？（提示：門檻是 $\alpha/h$，逐一比較 $|x^{\ast}|$ 與門檻。）
2. $h=2$、$x^{\ast}=0.9$：$\alpha$ 至少多大才會讓 $x^{\dagger}=0$？同樣的 $x^{\ast}$ 用 $\ell_2$（$\dfrac{h}{h+\alpha}x^{\ast}$），$\alpha$ 再大能得到精確的零嗎？（提示：零的條件是 $|x^{\ast}|\le\alpha/h$。）
3. $\mathbf{H}=\mathbf{I}$、$\mathbf{x}^{\ast}=(3,\,0.2)$、限制 $\|\mathbf{x}\|_1\le1$：此時最小點就是菱形上離 $\mathbf{x}^{\ast}$ 最近的點，求它並說明哪個座標為零；再求 $\ell_2$ 圓 $\|\mathbf{x}\|_2\le1$ 上的最小點 $\mathbf{x}^{\ast}/\|\mathbf{x}^{\ast}\|_2$。（提示：先求 $\mathbf{x}^{\ast}$ 往直線 $x_1+x_2=1$ 的垂足，看它在不在菱形的那條邊上。）

**Practice**:

1. $h=4$, $\alpha=3$: what is the threshold? What do $x^{\ast}=2,\ 0.5,\ -1$ become? (Hint: the threshold is $\alpha/h$; compare each $|x^{\ast}|$ with it.)
2. $h=2$, $x^{\ast}=0.9$: how large must $\alpha$ be at least for $x^{\dagger}=0$? With the same $x^{\ast}$ and $\ell_2$ ($\dfrac{h}{h+\alpha}x^{\ast}$), can a larger $\alpha$ give an exact zero? (Hint: the condition for zero is $|x^{\ast}|\le\alpha/h$.)
3. $\mathbf{H}=\mathbf{I}$, $\mathbf{x}^{\ast}=(3,\,0.2)$, constraint $\|\mathbf{x}\|_1\le1$: the minimum is then the point of the diamond nearest to $\mathbf{x}^{\ast}$; find it and say which coordinate is zero; then find the minimum $\mathbf{x}^{\ast}/\|\mathbf{x}^{\ast}\|_2$ on the $\ell_2$ circle $\|\mathbf{x}\|_2\le1$. (Hint: first find the foot of the perpendicular from $\mathbf{x}^{\ast}$ to the line $x_1+x_2=1$ and see whether it lies on that edge of the diamond.)

**解題示範**（第 1 題）：

1. 門檻：$\alpha/h=3/4=0.75$。
2. $x^{\ast}=2$：$2>0.75$，$x^{\dagger}=2-0.75=1.25$。
3. $x^{\ast}=0.5$：$|0.5|\le0.75$，$x^{\dagger}=0$。
4. $x^{\ast}=-1$：$-1<-0.75$，$x^{\dagger}=-1+0.75=-0.25$。
5. 檢查：結果 $(1.25,\ 0,\ -0.25)$ 的絕對值都比原來小，符號沒變，中間那個落在死區。

**Worked solution** (exercise 1):

1. Threshold: $\alpha/h=3/4=0.75$.
2. $x^{\ast}=2$: $2>0.75$, so $x^{\dagger}=2-0.75=1.25$.
3. $x^{\ast}=0.5$: $|0.5|\le0.75$, so $x^{\dagger}=0$.
4. $x^{\ast}=-1$: $-1<-0.75$, so $x^{\dagger}=-1+0.75=-0.25$.
5. Check: the results $(1.25,\ 0,\ -0.25)$ all have smaller absolute value than before with unchanged signs, and the middle one falls in the dead zone.

### 來源更正與取捨（Corrections to the source）

- 來源把軟閾值的分段條件寫在**輸出** $x_j^{\dagger}$ 上（「$x_j^{\dagger}>0$ 時…、$x_j^{\dagger}<0$ 時…」），這是循環定義：要先知道答案才知道用哪一行。本講義改成條件寫在 $x_j^{\ast}$ 上，並把來源另外用文字交代的「$|x_j^{\ast}|\le\alpha/h_j$ 時為零」併進同一個分段公式，並用次梯度推導。
- 來源說「接觸點容易落在軸上」。本講義用同一個 $\mathbf{x}^{\ast}$、換一個橢圓方向的反例，以及 $2819$ 組隨機橢圓的統計（$62.6\%$ 對 $0.1\%$）把「容易」量化，並明說不是「一定」。
- 來源的軟閾值公式前提是 Hessian 對角，只在一句話裡帶過。本講義用數字（逐座標軟閾值 $2.3167$ 對真解 $2.1474$）說明為什麼不能直接套到非對角 $\mathbf{H}$，並指出 $\ell_1$ 不像 $\ell_2$ 具有旋轉不變性。`[補充]`
- 來源說 $\ell_0$ 最小化是 NP-hard；本講義引述為來源的說法，不另行證明。
- 來源把 $\ell_1$ 懲罰寫成 $\alpha\|\mathbf{x}\|_1$（沒有 $\tfrac12$），本單元也用這個寫法；與 KP08 的 $\tfrac{\alpha}{2}\|\mathbf{x}\|_2^2$ 的 $\alpha$ 不能直接互相比較。

- The source writes the case conditions of the soft threshold on the **output** $x_j^{\dagger}$ ("when $x_j^{\dagger}>0$ …, when $x_j^{\dagger}<0$ …"), which is circular: one has to know the answer to know which line applies. This note puts the conditions on $x_j^{\ast}$, merges the "zero when $|x_j^{\ast}|\le\alpha/h_j$" that the source states in a separate sentence into the same piecewise formula, and derives it with a subgradient.
- The source says the touching point "easily" lies on an axis. This note quantifies "easily" with a counter-example (same $\mathbf{x}^{\ast}$, another orientation of the ellipse) and with the statistics of $2819$ random ellipses ($62.6\%$ against $0.1\%$), and says plainly that it is not "always".
- The source states the diagonal-Hessian assumption of the soft threshold in a single sentence. This note uses numbers (coordinate-wise soft threshold $2.3167$ against the true solution $2.1474$) to show why it cannot be applied directly to a non-diagonal $\mathbf{H}$, and points out that $\ell_1$, unlike $\ell_2$, is not rotation invariant. `[Added]`
- The source states that minimizing $\ell_0$ is NP-hard; this note quotes it as the source's statement and does not prove it.
- The source writes the $\ell_1$ penalty as $\alpha\|\mathbf{x}\|_1$ (no $\tfrac12$) and so does this unit; its $\alpha$ cannot be compared directly with the $\alpha$ of KP08's $\tfrac{\alpha}{2}\|\mathbf{x}\|_2^2$.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| $\ell_1$ 正則化／lasso | $\ell_1$ regularization / lasso |
| 凸鬆弛 | convex relaxation |
| 軟閾值 | soft threshold(ing) |
| 門檻／死區 | threshold / dead zone |
| 次梯度 | subgradient |
| 稀疏解 | sparse solution |
| 特徵選擇 | feature selection |
| 菱形／圓形約束 | diamond / circular constraint |
| 凸包絡 | convex envelope |
| 座標下降法 | coordinate descent |
