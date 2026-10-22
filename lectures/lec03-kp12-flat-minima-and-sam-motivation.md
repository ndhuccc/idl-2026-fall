# 第3週 KP12：平坦極小值與 SAM 的動機（Flat minima and the motivation for SAM）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

> **自主學習 / Self-study**：本單元為自主學習內容，課堂不講授，請自行閱讀。

本講義是第3章「反向傳播與最佳化」的第十二個知識點（KP12），承接 KP11 留下的問題：訓練損失低就夠了嗎？先備知識是梯度、KP01 的 Hessian 與 KP05 的特徵值；第一次出現的符號都會定義，所有數字都可以自己重算。內容依 CCC-LNS 第3章 KP12 的主文與投影片消化後重新撰寫，讀取日期 2026-10-14；來源沒有的內容標示 `[補充]`。

This is the twelfth knowledge point (KP12) of Chapter 3. It picks up the question KP11 left open: is a low training loss enough? It assumes the gradient, the Hessian of KP01 and the eigenvalues of KP05; every symbol is defined when it first appears and every number can be recomputed. The text is rewritten from the KP12 main text and slides of CCC-LNS Chapter 3 (accessed on 2026-10-14); content that the source does not contain is marked `[Added]`.

[下載本單元投影片](../slides/lec03-kp12-flat-minima-and-sam-motivation.jbd.json)

## KP12 平坦極小值與 SAM 的動機（Flat minima and the motivation for SAM）

### 學習目標（Learning objectives）

讀完之後，你能說明為什麼兩個訓練損失同樣低的解，在測試時可以差很多；能把資料差異近似成權重擾動，並算出擾動後的損失上升；能由二階展開寫出 $\Delta L\approx\tfrac12\,\Delta w^{\top}H\,\Delta w\le\tfrac12\lambda_{\max}\|\Delta w\|^2$，並解釋它的條件；能用兩層線性網路說明「平坦度」與參數化有關，所以曲率只是診斷而不是證明；也能寫出 SAM 的目標 $\min_w\max_{\|\epsilon\|\le\rho}L(w+\epsilon)$，說出它的動機與留給 KP13 的問題。

After reading, you can explain why two solutions with equally low training loss may differ a lot at test time; approximate a data difference as a weight perturbation and compute the resulting rise in loss; write $\Delta L\approx\tfrac12\,\Delta w^{\top}H\,\Delta w\le\tfrac12\lambda_{\max}\|\Delta w\|^2$ from a second-order expansion and state its conditions; use a two-layer linear network to show that flatness depends on the parametrization, so curvature is a diagnostic and not a proof; and write the SAM objective $\min_w\max_{\|\epsilon\|\le\rho}L(w+\epsilon)$, its motivation and the question it leaves for KP13.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $w,\ w^{\ast}$ | 權重；訓練損失的一個極小點 | the weights; a minimizer of the training loss |
| $L(w)$ | 訓練損失 | the training loss |
| $\Delta w$ | 權重擾動（把資料差異近似成權重的位移） | weight perturbation (a data difference approximated as a shift of the weights) |
| $\Delta L$ | 損失上升 $L(w^{\ast}+\Delta w)-L(w^{\ast})$ | the rise in loss, $L(w^{\ast}+\Delta w)-L(w^{\ast})$ |
| $H$ | $w^{\ast}$ 處的 Hessian（二階偏導數矩陣，對稱） | the Hessian at $w^{\ast}$ (symmetric matrix of second derivatives) |
| $\lambda_{\max}$ | $H$ 的最大特徵值（最陡方向的曲率） | the largest eigenvalue of $H$ (curvature along the steepest direction) |
| $\mathrm{Tr}(H)$ | $H$ 的跡，等於所有特徵值的和 | the trace of $H$, the sum of all eigenvalues |
| $k$ | 一維二次函數 $\tfrac12kw^2$ 的曲率 | curvature of the one-dimensional quadratic $\tfrac12kw^2$ |
| $\rho,\ \epsilon$ | 鄰域半徑；鄰域內的擾動，$\|\epsilon\|\le\rho$ | neighborhood radius; a perturbation inside it, $\|\epsilon\|\le\rho$ |
| $\sigma$ | 隨機擾動每個坐標的標準差 | standard deviation of a random perturbation per coordinate |

### 1. 損失一樣低，命運不一樣（Equal loss, different fates）

KP11 的最後一句問：訓練損失低就夠了嗎？答案是不夠。現代網路的參數遠多於樣本，損失曲面上有很多訓練損失都接近 0 的極小點。它們在訓練集上一樣好，在沒見過的資料上卻可以差很多。

KP11 ended with the question whether a low training loss is enough. The answer is no. A modern network has far more parameters than examples, so the loss surface has many minima whose training loss is close to 0. They are equally good on the training set but may differ a lot on unseen data.

先用一個你可以完全算清楚的例子。同一條損失曲線上有兩個極小點：尖銳谷在 $w=-1$，曲率 $k=20$；平坦谷在 $w=1.5$，曲率 $k=0.6$。兩個谷底的訓練損失都是 0。曲線取兩個二次函數的較小者：$L(w)=\min\{\tfrac12\cdot20\,(w+1)^2,\ \tfrac12\cdot0.6\,(w-1.5)^2\}$。

Start with an example you can compute completely. One loss curve has two minima: a sharp valley at $w=-1$ with curvature $k=20$ and a flat valley at $w=1.5$ with curvature $k=0.6$. Both valley floors have training loss 0. The curve takes the smaller of two quadratics, $L(w)=\min\{\tfrac12\cdot20\,(w+1)^2,\ \tfrac12\cdot0.6\,(w-1.5)^2\}$.

測試資料和訓練資料總有差別。我們把這個差別粗略地當成權重被推了一小段 $\Delta w$：測試損失約等於 $L(w^{\ast}+\Delta w)$。這是一個**近似**。它在「測試損失曲線只是把訓練損失曲線平移 $\Delta w$」時精確成立，一般情況只是啟發式的想法，不是定理。損失的上升是 $\Delta L=L(w^{\ast}+\Delta w)-L(w^{\ast})$。

Test data always differ from training data. We approximate that difference roughly as the weights being pushed by a small $\Delta w$, so the test loss is about $L(w^{\ast}+\Delta w)$. This is an **approximation**. It is exact when the test loss curve is just the training loss curve shifted by $\Delta w$; in general it is a heuristic idea, not a theorem. The rise in loss is $\Delta L=L(w^{\ast}+\Delta w)-L(w^{\ast})$.

取 $\Delta w=0.3$。尖銳谷的損失升到 $\tfrac12\cdot20\cdot0.3^2=0.9$，平坦谷只升到 $\tfrac12\cdot0.6\cdot0.3^2=0.027$，差 $0.9/0.027=33.3$ 倍，正好是 $20/0.6$。反過來問：若兩個谷都只容許損失上升 $0.027$，尖銳谷只能承受 $0.052$ 的位移，平坦谷可以承受 $0.3$〔程式計算〕。

Take $\Delta w=0.3$. The sharp valley rises to $\tfrac12\cdot20\cdot0.3^2=0.9$ and the flat valley only to $\tfrac12\cdot0.6\cdot0.3^2=0.027$, a ratio of $0.9/0.027=33.3$, exactly $20/0.6$. Turn it around: if both valleys may rise only by $0.027$, the sharp one tolerates a shift of $0.052$ and the flat one $0.3$ [computed by program].

![一條損失曲線上的兩個極小點：尖銳谷（k=20，w=−1）與平坦谷（k=0.6，w=1.5），兩者訓練損失都是 0；陰影是半徑 0.3 的鄰域，方塊是 w*+0.3 處，損失分別升高 0.9 與 0.027。（One loss curve with two minima, a sharp valley (k=20 at w=−1) and a flat valley (k=0.6 at w=1.5), both with training loss 0; the shaded bands are neighborhoods of radius 0.3 and the squares mark w*+0.3, where the loss rises by 0.9 and 0.027.）](../assets/figures/lec03-kp12-two-valleys.png)

**圖說 `lec03-kp12-two-valleys`**：先看兩個圓點：它們都在損失 0，打成平手。再看右邊的方塊：往右走 0.3，尖銳谷（藍）已經爬到 0.9，平坦谷（橘）只爬到 0.027。陰影寬度相同，所以差別來自谷的形狀。〔功能：1、2、3〕

**Caption `lec03-kp12-two-valleys`**: Look at the two dots first: both sit at loss 0, a tie. Then the squares on the right: after a step of 0.3 the sharp valley (blue) has climbed to 0.9 while the flat valley (orange) has climbed only to 0.027. The shaded bands have the same width, so the difference comes from the shape of the valley. [Function: 1, 2, 3]

來源還列出幾種「權重會被擾動」的情況，例如部署時的量化、Dropout 與 BatchNorm 統計量的晃動。這些可以當作**類比**：它們都讓權重偏離訓練時的值一點點。但它們與資料分佈偏移不是同一回事，本講義不把它們當成證明。

The source lists other situations in which the weights get perturbed, such as quantization at deployment and fluctuations of Dropout and BatchNorm statistics. Treat them as **analogies**: each moves the weights slightly away from the values seen in training. But they are not the same thing as a distribution shift, and this note does not treat them as proofs.

### 2. 用 Hessian 算鄰域的損失上升（Computing the rise with the Hessian）

上一節是逐點去量。能不能由極小點的曲率直接算出來？可以，只要 $\Delta w$ 夠小。極小點的梯度是 0，所以在 $w^{\ast}$ 附近展開到二階：

The last section measured point by point. Can the curvature at the minimum give the rise directly? Yes, as long as $\Delta w$ is small enough. The gradient at a minimum is 0, so expanding to second order around $w^{\ast}$ gives

$$
L(w^{\ast}+\Delta w)=L(w^{\ast})+\underbrace{\nabla L(w^{\ast})^{\top}\Delta w}_{=0}+\tfrac12\,\Delta w^{\top}H\,\Delta w+\text{(third-order and higher terms)}
$$

忽略三次以上的項，$\Delta L\approx\tfrac12\,\Delta w^{\top}H\,\Delta w$，其中 $H=\nabla^2L(w^{\ast})$。在極小點，$H$ 是對稱的，而且所有特徵值都不小於 0。

Dropping the third-order terms gives $\Delta L\approx\tfrac12\,\Delta w^{\top}H\,\Delta w$, where $H=\nabla^2L(w^{\ast})$. At a minimum $H$ is symmetric and all its eigenvalues are at least 0.

怎麼得到上界？把 $\Delta w$ 沿 $H$ 的單位特徵向量 $v_i$ 分解：$\Delta w=\sum_ic_iv_i$，對應的特徵值是 $\lambda_i$。那麼 $\Delta w^{\top}H\Delta w=\sum_i\lambda_ic_i^2$。每個 $\lambda_i\le\lambda_{\max}$，而 $\sum_ic_i^2=\|\Delta w\|^2$，所以

How do we get the upper bound? Decompose $\Delta w$ along the unit eigenvectors $v_i$ of $H$, $\Delta w=\sum_ic_iv_i$, with eigenvalues $\lambda_i$. Then $\Delta w^{\top}H\Delta w=\sum_i\lambda_ic_i^2$. Every $\lambda_i\le\lambda_{\max}$ and $\sum_ic_i^2=\|\Delta w\|^2$, so

$$
\Delta L\;\approx\;\tfrac12\,\Delta w^{\top}H\,\Delta w\;\le\;\tfrac12\,\lambda_{\max}\,\|\Delta w\|^{2}
$$

等號在 $\Delta w$ 恰好沿最大特徵向量時成立。也就是說，$\lambda_{\max}$ 控制「鄰域內最壞的方向」。半徑 $\rho$ 的鄰域內，最壞的上升約是 $\tfrac12\lambda_{\max}\rho^2$。

Equality holds when $\Delta w$ lies exactly along the top eigenvector. So $\lambda_{\max}$ controls the worst direction in the neighborhood. In a neighborhood of radius $\rho$ the worst rise is about $\tfrac12\lambda_{\max}\rho^2$.

一維時，$H$ 就是數字 $k$，二階展開對二次函數是精確的：$\Delta L=\tfrac12k\,\Delta w^2$。這正是上一節的 $0.9$ 與 $0.027$。如果擾動不是最壞的，而是隨機的，每個坐標獨立、平均 0、標準差 $\sigma$，那麼 $\mathbb E[\Delta w^{\top}H\Delta w]=\sigma^2\sum_i\lambda_i$，所以平均上升是 $\tfrac12\sigma^2\,\mathrm{Tr}(H)$。一維、$\sigma=0.1$ 時，$k=20$ 得 $0.1$，$k=0.6$ 得 $0.003$〔程式計算〕。

In one dimension $H$ is just the number $k$ and the second-order expansion is exact for a quadratic: $\Delta L=\tfrac12k\,\Delta w^2$. These are exactly the $0.9$ and $0.027$ of the last section. If the perturbation is not the worst one but random, with independent coordinates of mean 0 and standard deviation $\sigma$, then $\mathbb E[\Delta w^{\top}H\Delta w]=\sigma^2\sum_i\lambda_i$, so the average rise is $\tfrac12\sigma^2\,\mathrm{Tr}(H)$. In one dimension with $\sigma=0.1$, $k=20$ gives $0.1$ and $k=0.6$ gives $0.003$ [computed by program].

條件要說清楚。第一，$w^{\ast}$ 要是（局部）極小點，才能丟掉一次項。第二，$\Delta w$ 要小到可以忽略三次以上的項。第三，這只是**局部**的結論：兩個谷相距很遠時，它們各自的 $H$ 沒有告訴你谷與谷之間發生什麼事。來源把平坦解寫成「$\Delta L\approx0$」，並說尖銳解的 $\Delta L$ 「呈指數級爆炸」；兩者都不對：平坦解的 $\Delta L$ 是**較小**而不是 0（例如 $0.027$），而它隨 $\|\Delta w\|$ 是**二次**成長，不是指數成長。

The conditions must be stated. First, $w^{\ast}$ must be a (local) minimum so that the first-order term can be dropped. Second, $\Delta w$ must be small enough to ignore third-order terms. Third, this is a **local** conclusion: when two valleys are far apart, their own $H$ say nothing about what happens between them. The source writes $\Delta L\approx0$ for flat solutions and says $\Delta L$ "explodes exponentially" for sharp ones; both are wrong: for a flat solution $\Delta L$ is **smaller**, not 0 (for example $0.027$), and it grows **quadratically** in $\|\Delta w\|$, not exponentially.

### 3. 平坦度不是泛化的證明（Flatness is not a proof of generalization）

前面說「曲率小，所以鄰域上升小」。反過來問：曲率小，是不是就代表泛化好？不是。原因是：曲率依賴你怎麼寫參數，而函數本身不依賴。

So far: small curvature means a small rise in the neighborhood. Conversely, does small curvature mean good generalization? No. The reason is that curvature depends on how you write the parameters, while the function itself does not.

看最小的例子：兩層線性網路 $y=w_2w_1x$，只有一筆樣本 $(x,y)=(1,1)$，損失 $L(w_1,w_2)=\tfrac12(w_2w_1-1)^2$。所有 $w_1w_2=1$ 的點都是極小點，損失都是 0。Hessian 是 $H=\begin{pmatrix}w_2^2&1\\1&w_1^2\end{pmatrix}$（在極小點）。

Take the smallest example: a two-layer linear network $y=w_2w_1x$ with a single example $(x,y)=(1,1)$ and loss $L(w_1,w_2)=\tfrac12(w_2w_1-1)^2$. Every point with $w_1w_2=1$ is a minimum with loss 0. The Hessian there is $H=\begin{pmatrix}w_2^2&1\\1&w_1^2\end{pmatrix}$.

取兩個點：$(w_1,w_2)=(1,1)$ 與 $(4,\tfrac14)$。乘積都是 1，所以對**任何** $x$，預測 $w_2w_1x$ 都完全一樣，例如 $x=-2,0.5,3$ 時都是 $-2,0.5,3$。兩個網路是同一個函數。但 $\lambda_{\max}$ 不同：$H$ 的行列式是 $w_1^2w_2^2-1=0$，所以兩個特徵值是 0 與 $\mathrm{Tr}(H)=w_1^2+w_2^2$，$\lambda_{\max}$ 分別是 $2$ 與 $16+\tfrac1{16}=16.06$，差 $8.03$ 倍〔程式計算〕。

Take two points: $(w_1,w_2)=(1,1)$ and $(4,\tfrac14)$. Both have product 1, so for **every** $x$ the prediction $w_2w_1x$ is identical, for example $-2,0.5,3$ at $x=-2,0.5,3$. The two networks are the same function. But $\lambda_{\max}$ differs: the determinant of $H$ is $w_1^2w_2^2-1=0$, so the two eigenvalues are 0 and $\mathrm{Tr}(H)=w_1^2+w_2^2$, and $\lambda_{\max}$ is $2$ and $16+\tfrac1{16}=16.06$, a ratio of $8.03$ [computed by program].

一般地說，把 $(w_1,w_2)$ 改成 $(\alpha w_1,\,w_2/\alpha)$，函數不變，但 $\lambda_{\max}=\alpha^2+\alpha^{-2}$ 可以任意大。來源引用的 Dinh 等人（2017）把這個現象推到有 ReLU 的網路：這類網路有類似的縮放對稱，所以「尖銳」與「平坦」可以靠重新縮放權重來偽造。

In general, replacing $(w_1,w_2)$ by $(\alpha w_1,\,w_2/\alpha)$ leaves the function unchanged but makes $\lambda_{\max}=\alpha^2+\alpha^{-2}$ arbitrarily large. Dinh et al. (2017), cited by the source, carry this to networks with ReLU: such networks have a similar scaling symmetry, so "sharp" and "flat" can be faked by rescaling the weights.

![兩層線性網路 L=½(w₂w₁−1)² 的等高線。綠線是極小值曲線 w₁w₂=1；藍圈與橘圈是半徑 0.3 的鄰域，圓心分別是 (1,1) 與 (4,0.25)，最大曲率 λmax 是 2.00 與 16.06，但兩點代表同一個函數。（Contours of the two-layer linear network L=½(w₂w₁−1)². The green line is the minimum curve w₁w₂=1; the blue and orange circles are neighborhoods of radius 0.3 around (1,1) and (4,0.25), with largest curvature λmax of 2.00 and 16.06, yet the two points are the same function.）](../assets/figures/lec03-kp12-rescale.png)

**圖說 `lec03-kp12-rescale`**：先看綠線：線上每一點的預測都相同，損失都是 0。再看兩個等大的圓：藍圈裡的等高線稀疏，橘圈裡擠滿了等高線。圓在半徑相同的歐氏鄰域裡切到的損失不同，但函數沒變。〔功能：2、3〕

**Caption `lec03-kp12-rescale`**: Look at the green line first: every point on it makes the same predictions and has loss 0. Then the two circles of equal size: the contours are sparse inside the blue circle and crowded inside the orange one. Equal Euclidean neighborhoods cut different losses although the function has not changed. [Function: 2, 3]

這個縮放例子也對上界做了檢驗。半徑 $0.3$ 的圓內，二次近似給出的最壞上升是 $\tfrac12\lambda_{\max}\cdot0.3^2=0.090$ 與 $0.723$，真正的損失最大值是 $0.110$ 與 $0.733$：近似的量級對，但有誤差，因為忽略了三次以上的項。隨機擾動 $\sigma=0.1$ 時，$\tfrac12\sigma^2\mathrm{Tr}(H)$ 是 $0.0100$ 與 $0.0803$，蒙地卡羅（40 萬次）的平均是 $0.0100$ 與 $0.0802$〔程式計算〕。

This rescaling example also tests the bound. Inside a circle of radius $0.3$, the quadratic approximation gives worst rises $\tfrac12\lambda_{\max}\cdot0.3^2=0.090$ and $0.723$, while the true maxima of the loss are $0.110$ and $0.733$: the order of magnitude is right but there is an error, because third-order and higher terms were dropped. For random perturbations with $\sigma=0.1$, $\tfrac12\sigma^2\mathrm{Tr}(H)$ is $0.0100$ and $0.0803$, and a Monte Carlo average (400,000 draws) gives $0.0100$ and $0.0802$ [computed by program].

那麼優化器偏好平坦解的說法呢？來源說小批量 SGD 的噪聲把參數「踢出」尖銳谷，所以偏好平坦解；說大批量與牛頓法會陷入尖銳谷。這些只是**有條件的經驗傾向**：Keskar 等人（2017）觀察到大批量訓練的泛化差距與較尖銳的解有關，但後續研究指出調整學習率與訓練步數可以縮小差距（Hoffer 等人 2017；Goyal 等人 2017），Dinh 等人也質疑用曲率當尺度的做法 `[補充]`。沒有定理說牛頓法必定走到尖銳解。

What about the claim that optimizers prefer flat solutions? The source says the noise of small-batch SGD "kicks" the parameters out of sharp valleys, so it prefers flat solutions, and that large batches and Newton's method fall into sharp valleys. These are only **conditional empirical tendencies**: Keskar et al. (2017) observed that the generalization gap of large-batch training is associated with sharper solutions, but later work showed that adjusting the learning rate and the number of steps can shrink the gap (Hoffer et al. 2017; Goyal et al. 2017), and Dinh et al. also question using curvature as the yardstick `[Added]`. No theorem says Newton's method must end in a sharp solution.

結論：曲率是**診斷**，不是證明。它告訴你這個解對一小段權重位移有多敏感，卻不能單獨保證泛化，也不能在不同參數化之間直接比大小。

Conclusion: curvature is a **diagnostic**, not a proof. It tells you how sensitive the solution is to a small shift of the weights; it cannot guarantee generalization on its own and cannot be compared directly across parametrizations.

### 4. 把鄰域最壞損失放進目標（Putting the worst neighborhood loss into the objective）

既然鄰域的損失上升和泛化有關，能不能不要期待優化器碰巧走到平坦解，而是直接把它寫進目標？這就是 **SAM**（Sharpness-Aware Minimization，銳度感知最小化）的動機（Foret 等人，2021）。它不只要求 $L(w)$ 低，還要求 $w$ 的鄰域內最壞的損失也低：

Since the rise of the loss in a neighborhood is related to generalization, can we stop hoping that the optimizer stumbles onto a flat solution and write it into the objective instead? That is the motivation of **SAM** (Sharpness-Aware Minimization; Foret et al., 2021). It asks not only for a low $L(w)$ but also for a low worst loss in the neighborhood of $w$:

$$
\min_{w}\ \max_{\|\epsilon\|\le\rho}L(w+\epsilon)\;=\;\min_{w}\Big\{L(w)+\big[\max_{\|\epsilon\|\le\rho}L(w+\epsilon)-L(w)\big]\Big\}
$$

方括號是**銳度項**：鄰域最壞損失比 $L(w)$ 高多少。第二節說，二次近似下這個最大值約是 $\tfrac12\lambda_{\max}\rho^2$，所以銳度項大致就是 $\lambda_{\max}$ 的大小。

The bracket is the **sharpness term**: how far the worst loss in the ball exceeds $L(w)$. Section 2 says that under the quadratic approximation this maximum is about $\tfrac12\lambda_{\max}\rho^2$, so the sharpness term is roughly the size of $\lambda_{\max}$.

用第一節的曲線，$\rho=0.3$：尖銳谷底的鄰域最壞損失是 $0.9$，平坦谷底是 $0.027$。原來的 $L$ 在兩個谷底都是 0，打平；加上銳度項之後，最小點是 $w=1.5$ 的平坦谷〔程式計算〕。

On the curve of Section 1 with $\rho=0.3$, the worst neighborhood loss is $0.9$ at the sharp valley floor and $0.027$ at the flat one. The original $L$ is 0 at both floors, a tie; once the sharpness term is added, the minimizer is the flat valley at $w=1.5$ [computed by program].

![同一條損失曲線 L(w)（實線）與鄰域最壞損失 max L(w+ε)（虛線，ρ=0.3）。尖銳谷底的虛線值是 0.9，平坦谷底只有 0.027，最小化虛線選到 w=1.5。（The same loss curve L(w) (solid) and the worst neighborhood loss max L(w+ε) (dashed, ρ=0.3). The dashed value is 0.9 at the sharp floor and 0.027 at the flat floor; minimizing the dashed curve selects w=1.5.）](../assets/figures/lec03-kp12-sam-objective.png)

**圖說 `lec03-kp12-sam-objective`**：先看兩個圓點：實線在兩個谷底都是 0。再看菱形：虛線在尖銳谷底是 0.9，在平坦谷底是 0.027。虛線總是在實線上方，而且在尖銳的地方被抬得更高。〔功能：1、2〕

**Caption `lec03-kp12-sam-objective`**: Look at the two dots first: the solid line is 0 at both floors. Then the diamonds: the dashed line is 0.9 at the sharp floor and 0.027 at the flat floor. The dashed line always lies above the solid one and is lifted higher where the valley is sharp. [Function: 1, 2]

這裡有兩個限制，要誠實說。第一，上面的鄰域是歐氏球，第三節的縮放例子對它同樣成立，所以 SAM 不是萬靈丹；有人改用隨縮放調整的鄰域，例如 ASAM（Kwon 等人，2021）`[補充]`。第二，「最小化鄰域最壞損失」本身是動機，不是「一定泛化更好」的保證。還剩一個實作問題：內層的最大化每一步都要解，太貴了。怎麼用一次梯度近似它？這是 KP13 的主題。

Two limits must be stated honestly. First, the neighborhood above is a Euclidean ball and the rescaling example of Section 3 applies to it as well, so SAM is not a cure-all; some work uses a neighborhood that adapts to scaling, such as ASAM (Kwon et al., 2021) `[Added]`. Second, "minimize the worst neighborhood loss" is a motivation, not a guarantee of better generalization. One practical problem remains: the inner maximization would have to be solved at every step, which is too expensive. How can one gradient approximate it? That is the topic of KP13.

### 本單元重點整理（KP12 summary）

**一句話重點**：兩個訓練損失同樣低的解，受同樣的權重擾動後，損失的上升可以差很多，二次近似下它由 Hessian 的 $\lambda_{\max}$ 決定；但曲率與參數化有關，只是診斷而不是證明；SAM 的動機是把「鄰域最壞損失」直接放進目標。

**One-sentence summary**: Two solutions with equally low training loss can rise very differently under the same weight perturbation; under a quadratic approximation this is set by the $\lambda_{\max}$ of the Hessian; but curvature depends on the parametrization, so it is a diagnostic and not a proof; SAM is motivated by putting the "worst neighborhood loss" directly into the objective.

**白話說明**：站在谷底，往旁邊走一小步，會爬多高？谷窄，爬得高；谷寬，爬得低。測試資料就像把你推了一小步。SAM 的想法是：不要只看你站的那一點有多低，要看你周圍一圈最高的地方有多高。

**Plain explanation**: Standing at the bottom of a valley, how high do you climb if you step sideways a little? A narrow valley makes you climb high; a wide one, low. Test data is like a small push. The idea of SAM is: do not only look at how low the point you stand on is, look at how high the ground is around you.

**正式定義**：$\Delta L=L(w^{\ast}+\Delta w)-L(w^{\ast})\approx\tfrac12\Delta w^{\top}H\Delta w\le\tfrac12\lambda_{\max}\|\Delta w\|^2$；隨機擾動 $\mathbb E[\Delta L]\approx\tfrac12\sigma^2\mathrm{Tr}(H)$；SAM：$\min_w\max_{\|\epsilon\|\le\rho}L(w+\epsilon)$。

**Formal definition**: $\Delta L=L(w^{\ast}+\Delta w)-L(w^{\ast})\approx\tfrac12\Delta w^{\top}H\Delta w\le\tfrac12\lambda_{\max}\|\Delta w\|^2$; for a random perturbation $\mathbb E[\Delta L]\approx\tfrac12\sigma^2\mathrm{Tr}(H)$; SAM: $\min_w\max_{\|\epsilon\|\le\rho}L(w+\epsilon)$.

**例子**：一維兩個谷 $k=20$ 與 $k=0.6$，$\Delta w=0.3$：損失上升 $0.9$ 與 $0.027$，差 $33.3$ 倍；兩層線性網路 $(1,1)$ 與 $(4,\tfrac14)$ 是同一個函數，$\lambda_{\max}=2$ 與 $16.06$；$\rho=0.3$ 時，鄰域最壞損失 $0.9$ 對 $0.027$，最小化它選到平坦谷。逐步計算見各節與解題示範。

**Example**: One-dimensional valleys $k=20$ and $k=0.6$ with $\Delta w=0.3$: the loss rises by $0.9$ and $0.027$, a ratio of $33.3$; the two-layer linear network at $(1,1)$ and $(4,\tfrac14)$ is the same function with $\lambda_{\max}=2$ and $16.06$; with $\rho=0.3$ the worst neighborhood loss is $0.9$ versus $0.027$ and minimizing it picks the flat valley. The step-by-step computations are in the sections and the worked solution.

**常見迷思**：

- 「平坦解的 $\Delta L\approx0$」——是較小，不是 0；它隨位移二次成長（$0.027$，不是 0）。
- 「平坦的解一定泛化更好」——曲率與參數化有關，同一個函數可以有 $8.03$ 倍的 $\lambda_{\max}$；它是診斷，不是證明。
- 「大批量一定收斂到尖銳解、牛頓法必陷尖銳解」——只是有條件的經驗傾向，沒有定理，後續研究也有不同結果。
- 「SAM 保證找到平坦解」——它最小化的是鄰域最壞損失；它是動機，不是保證，歐氏球也不是縮放不變的。

**Common misconceptions**:

- "A flat solution has $\Delta L\approx0$." It is smaller, not 0; it grows quadratically with the shift ($0.027$, not 0).
- "A flat solution always generalizes better." Curvature depends on the parametrization; one function can have a $\lambda_{\max}$ $8.03$ times larger; it is a diagnostic, not a proof.
- "Large batches always end in sharp minima and Newton's method must fall into one." These are conditional empirical tendencies with no theorem, and later work gives different results.
- "SAM guarantees a flat solution." It minimizes the worst neighborhood loss; that is a motivation, not a guarantee, and a Euclidean ball is not scale invariant.

**圖**：本單元三張圖，`lec03-kp12-two-valleys`（兩個谷的損失上升，功能 1、2、3）、`lec03-kp12-rescale`（同一函數的兩個點，功能 2、3）與 `lec03-kp12-sam-objective`（原損失與鄰域最壞損失，功能 1、2），以 `assets/figures/src/lec03-kp12-figures.py` 繪製，數字都由同一支程式印出。二次型的推導（第 2 節）與 SAM 的目標式純屬代數，在簡報中用原生節點與連線呈現，不另畫圖。

**Figure**: Three figures in this unit: `lec03-kp12-two-valleys` (the rise of the loss in two valleys; functions 1, 2, 3), `lec03-kp12-rescale` (two points of the same function; functions 2, 3) and `lec03-kp12-sam-objective` (the original loss and the worst neighborhood loss; functions 1, 2), drawn by `assets/figures/src/lec03-kp12-figures.py`, which prints every number from the same program. The derivation of the quadratic form (Section 2) and the SAM objective are purely algebraic and are drawn in the deck with native nodes and connectors, without a separate figure.

**想深入**（選讀）：`[補充]` PAC-Bayes 泛化界把「權重被擾動後損失變多少」放進界裡，是平坦度與泛化之間的一條理論線索，但界的鬆緊與條件要另外討論。SGD 的噪聲對解的形狀有隱式的正則化效果，在某些簡化模型裡可以分析，一般情形仍是開放問題。這兩點留給進階課程。

**Going deeper** (optional): `[Added]` PAC-Bayes generalization bounds put "how much the loss grows when the weights are perturbed" into the bound; they are one theoretical thread between flatness and generalization, but how tight the bounds are and under what conditions needs separate discussion. The noise of SGD has an implicit regularizing effect on the shape of the solution, which can be analyzed in some simplified models; the general case is still open. Both points are left to advanced courses.

**小練習**：

1. 兩個谷的曲率是 $k=5$ 與 $k=0.2$，$\Delta w=0.5$：各上升多少？若只容許上升 $0.1$，各能承受多大的位移？（提示：$\Delta L=\tfrac12k\,\Delta w^2$。）
2. 極小點的 Hessian 是 $\mathrm{diag}(10,\,0.1)$，鄰域半徑 $\rho=0.2$：二次近似下最壞的上升是多少？沿 $x$ 軸與沿 $y$ 軸各是多少？（提示：$\tfrac12\lambda_{\max}\rho^2$ 與 $\tfrac12\lambda\rho^2$。）
3. 把兩層線性網路取 $\alpha=2$：點 $(2,\tfrac12)$ 的 $H$ 是什麼？它的 $\lambda_{\max}$ 與公式 $\alpha^2+\alpha^{-2}$ 一致嗎？（提示：先算 $\mathrm{Tr}(H)$，再用行列式為 0。）

**Practice**:

1. Two valleys have curvature $k=5$ and $k=0.2$ and $\Delta w=0.5$: how much does each rise? If only a rise of $0.1$ is allowed, how large a shift can each tolerate? (Hint: $\Delta L=\tfrac12k\,\Delta w^2$.)
2. The Hessian at a minimum is $\mathrm{diag}(10,\,0.1)$ and the neighborhood radius is $\rho=0.2$: what is the worst rise under the quadratic approximation? What are the rises along the $x$ axis and along the $y$ axis? (Hint: $\tfrac12\lambda_{\max}\rho^2$ and $\tfrac12\lambda\rho^2$.)
3. Take $\alpha=2$ in the two-layer linear network: what is $H$ at $(2,\tfrac12)$? Does its $\lambda_{\max}$ agree with the formula $\alpha^2+\alpha^{-2}$? (Hint: compute $\mathrm{Tr}(H)$ first, then use that the determinant is 0.)

**解題示範**（兩層線性網路、$L=\tfrac12(w_2w_1-1)^2$，給定兩個極小點的 Hessian，比較它們的鄰域最壞損失，並說明重新縮放後結論怎麼變。$\rho=0.3$）：

1. 兩個點：$A=(1,1)$，$B=(4,\tfrac14)$。乘積都是 1，所以都是極小點，損失都是 0，而且預測相同。
2. Hessian：$H_A=\begin{pmatrix}1&1\\1&1\end{pmatrix}$，$H_B=\begin{pmatrix}\tfrac1{16}&1\\1&16\end{pmatrix}$。
3. 特徵值：$H_A$ 的跡是 2、行列式是 $1-1=0$，所以特徵值是 $2$ 與 $0$；$H_B$ 的跡是 $16.0625$、行列式是 $\tfrac1{16}\cdot16-1=0$，所以特徵值是 $16.0625$ 與 $0$。
4. 鄰域最壞損失（二次近似）：$\tfrac12\lambda_{\max}\rho^2$，$A$ 是 $\tfrac12\cdot2\cdot0.09=0.090$，$B$ 是 $\tfrac12\cdot16.0625\cdot0.09=0.723$。$B$ 是 $A$ 的 $8.03$ 倍。
5. 方向：$B$ 的最大特徵向量約是 $(0.062,\,0.998)$，幾乎沿 $w_2$ 軸：沿 $(0.3,0)$ 只上升 $0.0028$，沿 $(0,0.3)$ 上升 $0.720$。$A$ 沿兩個軸各上升 $0.045$。
6. 檢驗：真正的損失在半徑 $0.3$ 的圓內最大是 $0.110$（$A$）與 $0.733$（$B$）。二次近似的量級對，略低一些，因為少了三次以上的項。
7. 判斷：以 $\lambda_{\max}$ 看，$A$ 比 $B$ 平坦 $8.03$ 倍，鄰域最壞損失 $A$ 小。
8. 重新縮放：$B$ 是 $A$ 的 $(\alpha w_1,w_2/\alpha)$，$\alpha=4$。兩者是同一個函數，對任何 $x$ 預測相同，測試表現也相同。可是 $\lambda_{\max}$ 差 $8.03$ 倍，反方向縮放（$\alpha=\tfrac14$）還會讓 $A$ 看起來「更尖」。所以不同參數化之間不能直接用 $\lambda_{\max}$ 排名；曲率只在同一個網路、尺度相近的解之間比較才有意義，而且只是診斷。〔程式計算〕

**Worked solution** (two-layer linear network, $L=\tfrac12(w_2w_1-1)^2$; given the Hessians of two minima, compare their worst neighborhood losses and say how the conclusion changes after rescaling; $\rho=0.3$):

1. Two points: $A=(1,1)$ and $B=(4,\tfrac14)$. Both products are 1, so both are minima with loss 0 and the same predictions.
2. Hessians: $H_A=\begin{pmatrix}1&1\\1&1\end{pmatrix}$ and $H_B=\begin{pmatrix}\tfrac1{16}&1\\1&16\end{pmatrix}$.
3. Eigenvalues: $H_A$ has trace 2 and determinant $1-1=0$, so the eigenvalues are $2$ and $0$; $H_B$ has trace $16.0625$ and determinant $\tfrac1{16}\cdot16-1=0$, so the eigenvalues are $16.0625$ and $0$.
4. Worst neighborhood loss (quadratic approximation): $\tfrac12\lambda_{\max}\rho^2$ is $\tfrac12\cdot2\cdot0.09=0.090$ for $A$ and $\tfrac12\cdot16.0625\cdot0.09=0.723$ for $B$. $B$ is $8.03$ times $A$.
5. Directions: the top eigenvector of $B$ is about $(0.062,\,0.998)$, almost along the $w_2$ axis: along $(0.3,0)$ the rise is only $0.0028$ and along $(0,0.3)$ it is $0.720$. $A$ rises $0.045$ along each axis.
6. Check: the true maximum of the loss in the circle of radius $0.3$ is $0.110$ ($A$) and $0.733$ ($B$). The quadratic approximation has the right size and is slightly low, because third-order and higher terms were dropped.
7. Judgment: by $\lambda_{\max}$, $A$ is $8.03$ times flatter than $B$ and has the smaller worst neighborhood loss.
8. Rescaling: $B$ is $A$ rescaled as $(\alpha w_1,w_2/\alpha)$ with $\alpha=4$. They are the same function, with the same prediction for every $x$ and the same test behavior. Yet $\lambda_{\max}$ differs by $8.03$ times, and rescaling the other way ($\alpha=\tfrac14$) would make $A$ look "sharper". So $\lambda_{\max}$ cannot rank solutions across parametrizations; curvature is meaningful only among solutions of the same network at comparable scales, and even then it is a diagnostic [computed by program].

### 來源更正與取捨（Corrections to the source）

- 來源說尖銳解的 $\Delta L$「呈指數級爆炸」、平坦解的 $\Delta L\approx0$、平坦解的 $\lambda_{\max}\approx0$。二次近似下 $\Delta L$ 隨位移**二次**成長，平坦解是**較小**（例子中 $0.027$）而不是 0，$\lambda_{\max}$ 是較小（例子中 $0.6$）而不是 0。本講義改用這些說法。
- 來源說「Newton 法必定陷入尖銳極小值」「大批量一定收斂到尖銳解」。沒有定理支持；大批量的泛化差距有觀察（Keskar 等人 2017），但後續研究顯示調整學習率與步數可縮小差距 `[補充]`。本講義改為「有條件的經驗傾向」。
- 來源說 SGD 噪聲的協方差約是 $\tfrac{\eta}{2B}\nabla^2L$，並以此解釋 SGD 偏好平坦解。這個式子需要多項假設（二次近似、噪聲與曲率成比例等），我們沒有逐一查證它的推導，所以不採用為事實，只保留「SGD 噪聲偏好平坦解是經驗傾向」。
- 來源說量化、Dropout、BatchNorm 的晃動「等價於」權重擾動，平坦解對它們「幾乎零衰減」「天然防禦」。這些是類比，不是證明，本講義只說它們讓權重偏離一點點。資料偏移等於權重位移的近似，只在「測試損失曲線是訓練損失曲線的平移」時精確成立，一般只是啟發式想法，已在第 1 節寫明。
- 來源說 SAM「本質上沿 $\lambda_{\max}$ 的主特徵向量方向探索」。這只在二次近似下對內層最大化成立；實際的 SAM 怎麼近似內層最大化，留給 KP13。來源也說只有平坦解「才能成為全域性極小點」，這不準確：SAM 的目標是鄰域最壞損失的最小點，它不保證平坦，也不保證泛化更好。
- 來源說平坦極小值「為 PAC-Bayes 提供嚴謹的支援」，並把大量優點歸給平坦度。本講義只在「想深入」用一句話提 PAC-Bayes 與隱式正則化，不當成已證明的結論。Dinh 等人（2017）的縮放論點保留，並用兩層線性網路自己算一遍。
- 來源把 SAM 寫作 Foret 等人（2020）與（ICLR 2021）兩種年份；本講義統一為 ICLR 2021（預印本 2020）。來源的選擇題與互動實驗室（$k=20$、$k=0.6$）的數字與本講義一致，本講義用自己的程式重算。

- The source says the $\Delta L$ of sharp solutions "explodes exponentially", that $\Delta L\approx0$ for flat ones and that their $\lambda_{\max}\approx0$. Under the quadratic approximation $\Delta L$ grows **quadratically** with the shift; for a flat solution it is **smaller** (here $0.027$), not 0, and $\lambda_{\max}$ is smaller (here $0.6$), not 0. This note uses these statements instead.
- The source says "Newton's method must fall into sharp minima" and "large batches always converge to sharp minima". No theorem supports this; the generalization gap of large batches was observed (Keskar et al. 2017), but later work shows that adjusting the learning rate and the number of steps can shrink it `[Added]`. This note says "conditional empirical tendency".
- The source says the covariance of SGD noise is about $\tfrac{\eta}{2B}\nabla^2L$ and uses it to explain why SGD prefers flat solutions. This expression needs several assumptions (quadratic approximation, noise proportional to curvature and so on); we did not verify its derivation, so it is not adopted as a fact. Only "the preference of SGD noise for flat solutions is an empirical tendency" is kept.
- The source says quantization and the fluctuations of Dropout and BatchNorm are "equivalent to" weight perturbations and that flat solutions have "almost zero degradation" and a "natural defense". These are analogies, not proofs; this note only says they move the weights slightly. Treating a data shift as a weight shift is exact only when the test loss curve is the training loss curve shifted; in general it is a heuristic idea, as stated in Section 1.
- The source says SAM "essentially explores along the top eigenvector of $\lambda_{\max}$". This holds for the inner maximization only under the quadratic approximation; how the actual SAM approximates the inner maximization is left to KP13. The source also says that only flat solutions "can become the global minimum"; this is inaccurate: the SAM objective is minimized at the point with the smallest worst neighborhood loss, which guarantees neither flatness nor better generalization.
- The source says flat minima give "rigorous support for PAC-Bayes" and credits flatness with many benefits. This note mentions PAC-Bayes and implicit regularization in one sentence under "Going deeper" and does not treat them as proven conclusions. The rescaling argument of Dinh et al. (2017) is kept and recomputed with a two-layer linear network.
- The source dates SAM both as Foret et al. (2020) and as ICLR 2021; this note uses ICLR 2021 (preprint 2020). The numbers of the source's quiz and interactive lab ($k=20$, $k=0.6$) agree with this note, which recomputes them with its own program.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 平坦極小值 | flat minimum |
| 尖銳極小值 | sharp minimum |
| 泛化差距 | generalization gap |
| 權重擾動 | weight perturbation |
| 曲率 | curvature |
| Hessian（二階偏導數矩陣） | Hessian matrix |
| 最大特徵值 | largest eigenvalue |
| 跡 | trace |
| 二階展開 | second-order Taylor expansion |
| 重新縮放（縮放對稱） | rescaling (scaling symmetry) |
| 參數化 | parametrization |
| 鄰域 | neighborhood |
| 銳度項 | sharpness term |
| 銳度感知最小化 | Sharpness-Aware Minimization (SAM) |
| 隱式正則化 | implicit regularization |
