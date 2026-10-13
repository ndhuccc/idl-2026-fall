# 第3週 KP01：最佳化即參數搜尋（Optimization as parameter search）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第3章「反向傳播與最佳化」的第一個知識點（KP01）。先備知識是微積分的偏導數、向量與矩陣的基本運算；第一次出現的符號都會定義，所有數字都可以自己重算。內容依 CCC-LNS 第3章 KP01 的主文與互動算例消化後重新撰寫，讀取日期 2026-10-13；來源沒有的例子標示 `[補充]`。

This is the first knowledge point (KP01) of Chapter 3, "Backpropagation and optimization". It assumes partial derivatives and basic vector and matrix operations; every symbol is defined when it first appears and every number can be recomputed. The text is rewritten from the KP01 material of CCC-LNS Chapter 3 (accessed on 2026-10-13); examples that the source does not contain are marked `[Added]`.

[下載本單元投影片](../slides/lec03-kp01-optimization-as-search.jbd.json)

## KP01 最佳化即參數搜尋（Optimization as parameter search）

### 學習目標（Learning objectives）

讀完之後，你能說出假設類別、損失與最佳化器各負責什麼；能把「一次更新」拆成預測、算損失、求梯度、更新四個動作；能用一個二維的例子算出牛頓步與負梯度步的差別；能算出海森矩陣在不同參數數量下的記憶體與運算量，並說明深度學習為什麼選擇一階方法。

After reading, you can say what the hypothesis class, the loss and the optimizer each do; split one update into predicting, scoring the loss, differentiating and updating; compute a Newton step and a negative-gradient step on a two-dimensional example; compute the memory and operations a Hessian needs for different parameter counts; and explain why deep learning chooses first-order methods.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\theta$ | 模型的全部參數（一個向量，有 $d$ 個數） | all parameters of the model (a vector of $d$ numbers) |
| $d$ | 參數的個數 | number of parameters |
| $\mathcal{L}(\theta)$ | 損失：這組參數的錯誤分數（一個數字） | loss: the error score of these parameters (one number) |
| $\nabla f$ | 梯度：損失上升最快的方向（一個向量） | gradient: the direction in which the loss rises fastest (a vector) |
| $\eta$ | 學習率：每一步走多遠的正數 | learning rate: a positive number that sets the step length |
| $H$ | 海森矩陣：損失的二階偏導數排成的 $d\times d$ 矩陣 | Hessian: the $d\times d$ matrix of second partial derivatives |
| $\theta_t$ | 第 $t$ 步時的參數 | the parameters at step $t$ |

### 1. 訓練裡的山、坡度與走路的人（The mountain, the slope and the hiker in training）

想像你被蒙住眼睛放在一座山上，目標是走到最低的山谷。你看不到整座山，只能用腳感覺此刻哪個方向最陡。訓練一個深度網路就是這件事：山是「每一組參數對應的損失」，坡度是損失在這一點的變化方向，走路的人就是最佳化器。

Imagine being dropped blindfolded onto a mountain with the goal of reaching the lowest valley. You cannot see the whole mountain; you only feel which direction is steepest under your feet. Training a deep network is exactly this: the mountain is "the loss of every parameter setting", the slope is how the loss changes at the current point, and the hiker is the optimizer.

機器學習把這個過程拆成三個角色。**假設類別**決定模型能畫出哪些函數，也就決定山可能的形狀；**損失**把預測與答案的差距換成一個分數，每一組參數都有一個分數；**最佳化器**看著分數，決定參數下一步往哪裡調整。把目標寫成一行式子就是

Machine learning splits the process into three roles. The **hypothesis class** decides which functions the model can draw, hence the possible shapes of the mountain; the **loss** turns the gap between prediction and answer into a score, one per parameter setting; the **optimizer** reads the scores and decides how to adjust the parameters next. The goal fits in one line:

$$
\min_{\theta}\ \mathcal{L}(\theta)
$$

意思是：在所有可能的參數 $\theta$ 裡，找出讓損失 $\mathcal{L}(\theta)$ 最小的那一組。注意這一行只說「要找什麼」，沒有說「怎麼找」。損失只替參數排名次：它能告訴你這組參數比那組好，卻不會告訴你下一步該往哪個方向走。決定怎麼走，是最佳化器的工作，也是整個訓練流程真正的引擎。

It says: among all possible parameters $\theta$, find the one with the smallest loss $\mathcal{L}(\theta)$. Note that it states what to find, not how to find it. The loss only ranks parameter settings: it can tell you that one setting is better than another, but not which direction to move next. Deciding how to move is the optimizer's job, and it is the real engine of the whole training pipeline.

### 2. 最佳化器的一步（One step of the optimizer）

接下來看最佳化器在每一步究竟做了什麼。第一步，用目前的參數 $\theta_t$ 和資料做預測。第二步，把預測和標準答案比較，算出損失 $\mathcal{L}(\theta_t)$。第三步，求損失在這一點的梯度 $\nabla \mathcal{L}(\theta_t)$，也就是損失上升最快的方向。第四步，最佳化器選出下一組參數 $\theta_{t+1}$，然後回到第一步，重複到停止為止（何時停止是 KP03 的主題）。

Now look at what the optimizer does at each step. First, predict with the current parameters $\theta_t$ and the data. Second, compare the prediction with the answer to get the loss $\mathcal{L}(\theta_t)$. Third, compute the gradient $\nabla \mathcal{L}(\theta_t)$ at this point, the direction in which the loss rises fastest. Fourth, the optimizer chooses the next parameters $\theta_{t+1}$ and returns to the first step, repeating until it stops (when to stop is the topic of KP03).

下面用一個只有兩個參數的碗來看這四步。這個碗是 `[補充]` 的玩具函數，本講義後面都用它當主例子：

The following two-parameter bowl shows the four steps. It is a `[Added]` toy function and serves as the main example for the rest of this note:

$$
f(x,y)=\tfrac12\,(x^2+25y^2),\qquad \theta_0=(10,\,2)
$$

在 $\theta_0$ 的損失是 $f(10,2)=\tfrac12(100+100)=100$。偏導數是 $\partial f/\partial x=x$、$\partial f/\partial y=25y$，所以梯度 $\nabla f(10,2)=(10,\,50)$。它幾乎指向 $y$ 方向，因為碗在 $y$ 方向陡得多。負梯度 $-\nabla f$ 就是下坡的方向。取學習率 $\eta=0.06$ 走一步：

At $\theta_0$ the loss is $f(10,2)=\tfrac12(100+100)=100$. The partial derivatives are $\partial f/\partial x=x$ and $\partial f/\partial y=25y$, so the gradient is $\nabla f(10,2)=(10,\,50)$. It points almost along $y$ because the bowl is much steeper in $y$. The negative gradient $-\nabla f$ is the downhill direction. Take one step with learning rate $\eta=0.06$:

$$
\theta_1=\theta_0-0.06\,\nabla f(\theta_0)=(10-0.06\cdot10,\ 2-0.06\cdot50)=(10-0.6,\ 2-3.0)=(9.4,\,-1.0)
$$

新位置的損失是 $f(9.4,-1.0)=\tfrac12(88.36+25)=56.68$，比 100 小，確實往下走了。（學習率怎麼選、為什麼 $0.06$ 可行，是 KP02 的主題。）

The loss at the new point is $f(9.4,-1.0)=\tfrac12(88.36+25)=56.68$, smaller than 100, so we did go downhill. (How to choose the learning rate, and why $0.06$ works here, is the topic of KP02.)

![玩具碗的等高線：θ₀=(10,2) 的紅箭頭是梯度（上坡），青綠箭頭是負梯度（下坡），空心圈是走一步後的 θ₁=(9.4,−1.0)，星號是最小點。（Contours of the toy bowl: the red arrow at θ₀=(10,2) is the gradient (uphill), the teal arrow is the negative gradient (downhill), the open circle is θ₁=(9.4,−1.0) after one step, and the star is the minimum.）](../assets/figures/lec03-kp01-contours.png)

**圖說 `lec03-kp01-contours`**：先看橢圓等高線，每一圈上的損失相同，越靠近中心越低；再看 $\theta_0$ 的兩支箭頭：紅色是梯度，指向上坡；青綠色是負梯度，指向下坡，更新就沿著它走一小步。注意兩支箭頭只表示方向，不表示長度；負梯度也沒有直指星號。〔功能：1、3〕

**Caption `lec03-kp01-contours`**: Look at the elliptical contours first: the loss is the same along each ring and gets smaller toward the center. Then look at the two arrows at $\theta_0$: red is the gradient, pointing uphill; teal is the negative gradient, pointing downhill, and the update takes a small step along it. The arrows show direction only, not length, and the negative gradient does not point straight at the star. [Function: 1, 3]

### 3. 多用曲率會怎樣（What if we also use curvature）

上一節只用了梯度，也就是損失的一階導數。梯度只知道這一點最陡的方向，不知道碗有多彎。如果再多用一點資訊，也就是**曲率**，能不能走得更聰明？曲率由海森矩陣 $H$（損失的二階偏導數排成的矩陣）描述。我們的碗很簡單：$\partial^2 f/\partial x^2=1$、$\partial^2 f/\partial y^2=25$、混合項為零，所以 $H=\mathrm{diag}(1,25)$。

The last section used only the gradient, the first derivative of the loss. The gradient knows the steepest direction at this point but not how strongly the bowl curves. What if we also use **curvature**? Curvature is described by the Hessian $H$, the matrix of second partial derivatives. Our bowl is simple: $\partial^2 f/\partial x^2=1$, $\partial^2 f/\partial y^2=25$ and the mixed terms are zero, so $H=\mathrm{diag}(1,25)$.

牛頓法用 $H^{-1}$ 把梯度重新縮放：新位置是 $\theta_0-H^{-1}\nabla f(\theta_0)$。代入數字：$H^{-1}\nabla f=(10/1,\ 50/25)=(10,\,2)$，所以新位置是 $(10,2)-(10,2)=(0,\,0)$，一步就走到最小點，損失為 0。原因是這個碗剛好是二次函數，牛頓法對二次函數是精確的。

Newton's method rescales the gradient with $H^{-1}$: the new point is $\theta_0-H^{-1}\nabla f(\theta_0)$. With numbers, $H^{-1}\nabla f=(10/1,\ 50/25)=(10,\,2)$, so the new point is $(10,2)-(10,2)=(0,\,0)$: the minimum in one step, with loss 0. This works because the bowl is exactly quadratic, and Newton's method is exact on quadratics.

![同一個碗上的兩種步伐：左，牛頓步 −H⁻¹∇f 是一條直指最小點的藍色箭頭；右，負梯度的青綠箭頭幾乎垂直往下，橘色折線是學習率 0.06 的前 12 步，呈之字形。（Two steps on the same bowl: left, the Newton step −H⁻¹∇f is a blue arrow aimed straight at the minimum; right, the teal negative-gradient arrow points almost straight down and the orange polyline shows the first 12 steps with learning rate 0.06, zigzagging.）](../assets/figures/lec03-kp01-newton-gd.png)

**圖說 `lec03-kp01-newton-gd`**：先比較兩個面板的箭頭：左邊的牛頓步直指星號，右邊的負梯度幾乎垂直往下，沒有指向星號；再看右圖的橘色折線：每一步在 $y$ 方向來回跨過谷底，在 $x$ 方向只前進一小段。兩個面板的座標比例相同，所以橢圓的狹長是真的。〔功能：2、3〕

**Caption `lec03-kp01-newton-gd`**: Compare the arrows in the two panels first: the Newton step on the left aims at the star, while the negative gradient on the right points almost straight down, not at the star. Then follow the orange polyline: each step crosses the valley floor in $y$ and advances only a little in $x$. Both panels use the same axis scale, so the elongated ellipses are real. [Function: 2, 3]

負梯度為什麼不指向谷底？因為梯度只看「此刻最陡」。這個碗在 $y$ 方向（係數 25）比 $x$ 方向（係數 1）陡 25 倍，最陡的方向就被 $y$ 方向主導，幾乎垂直往下。之後 $y$ 每步乘上 $1-0.06\cdot25=-0.5$（正負交替地縮小），而 $x$ 每步只乘上 $1-0.06\cdot1=0.94$（只縮小 6%），所以路徑呈之字形且橫向前進很慢。用程式算出損失：第 10 步剩 $14.5$，第 30 步還有 $1.2$〔程式計算〕。牛頓法在第 1 步就是 0。

Why does the negative gradient not point at the valley floor? The gradient sees only "steepest right now". The bowl is 25 times steeper in $y$ (coefficient 25) than in $x$ (coefficient 1), so the steepest direction is dominated by $y$ and points almost straight down. After that, $y$ is multiplied by $1-0.06\cdot25=-0.5$ at each step (shrinking with alternating sign), while $x$ is multiplied only by $1-0.06\cdot1=0.94$ (shrinking by 6%), so the path zigzags and advances slowly along $x$. Computed by program, the loss is $14.5$ after 10 steps and still $1.2$ after 30 steps; Newton's method reaches 0 at step 1.

### 4. 曲率的帳單與一階的選擇（The curvature bill and the choice of first order）

曲率這麼有用，為什麼深度學習不用？因為要付的代價隨參數個數 $d$ 快速成長。海森矩陣有 $d\times d=d^2$ 個元素；每個數用 float32 存要 4 bytes，所以光是存 $H$ 就要 $4d^2$ bytes。牛頓步還要解 $H\,\Delta\theta=\nabla f$，也就是求逆，運算量約 $d^3$ 次。相對地，梯度只有 $d$ 個數，$4d$ bytes。

Curvature is so useful; why does deep learning not use it? Because the price grows quickly with the number of parameters $d$. The Hessian has $d\times d=d^2$ entries; at 4 bytes per float32 number, storing $H$ alone takes $4d^2$ bytes. A Newton step must also solve $H\,\Delta\theta=\nabla f$, i.e. invert $H$, which costs about $d^3$ operations. The gradient, by contrast, has only $d$ numbers, $4d$ bytes.

| 參數數 $d$ | 梯度 $g$：$4d$ bytes | 海森 $H$：$4d^2$ bytes | 求逆 $d^3$ 次運算 | 時間（$10^{13}$ 次／秒） |
|---|---|---|---|---|
| $10^2$ | 400 B | 40 KB | $10^6$ | 0.1 µs |
| $10^4$ | 40 KB | 400 MB | $10^{12}$ | 0.1 s |
| $10^6$ | 4 MB | 4 TB | $10^{18}$ | $10^5$ s ≈ 27.8 小時 |

| Parameters $d$ | Gradient $g$: $4d$ bytes | Hessian $H$: $4d^2$ bytes | Inversion: $d^3$ operations | Time ($10^{13}$ ops/s) |
|---|---|---|---|---|
| $10^2$ | 400 B | 40 KB | $10^6$ | 0.1 µs |
| $10^4$ | 40 KB | 400 MB | $10^{12}$ | 0.1 s |
| $10^6$ | 4 MB | 4 TB | $10^{18}$ | $10^5$ s ≈ 27.8 hours |

當 $d=10^6$（對今天的網路來說算小），海森矩陣有 $10^{12}$ 個元素，要 4 TB；求逆約 $10^{18}$ 次運算，以每秒 $10^{13}$ 次（約 10 TFLOPS 的硬體）計要 $10^{18}/10^{13}=10^5$ 秒，約 27.8 小時，而且這只是**一步**。表中的時間用了假設的硬體速度，只是量級估計〔程式計算〕。另外，高維非凸的損失曲面上有很多鞍點，牛頓法容易被吸過去；來源也指出這一點。

At $d=10^6$ (small for today's networks) the Hessian has $10^{12}$ entries and needs 4 TB; inversion takes about $10^{18}$ operations, which at $10^{13}$ operations per second (hardware of about 10 TFLOPS) is $10^{18}/10^{13}=10^5$ seconds, about 27.8 hours, for just **one** step. The times in the table assume a hardware speed and are order-of-magnitude estimates [computed by program]. In addition, a high-dimensional non-convex loss surface has many saddle points that attract Newton's method; the source points this out as well.

所以深度學習選擇**一階方法**：每一步只需要一個梯度向量，而反向傳播（KP06）正是能便宜地算出整個梯度向量的方法，不必儲存 $H$。這個選擇也有代價：一階方法只拿到方向，沒有曲率，每一步要走多遠、怎麼避免之字形，就成了接下來整章要處理的問題。KP02 先從最基本的規則開始：既然只有方向，這一步該走多遠？

So deep learning chooses **first-order methods**: each step needs only one gradient vector, and backpropagation (KP06) is exactly the method that computes the whole gradient vector cheaply, without storing $H$. The choice has a price: a first-order method receives only a direction and no curvature, so how far to step and how to avoid zigzagging become the questions of the rest of the chapter. KP02 starts with the most basic rule: given only a direction, how far should one step?

### 本單元重點整理（KP01 summary）

**一句話重點**：訓練就是在參數空間搜尋讓損失最小的參數；因為曲率太貴，深度學習只用反向傳播便宜算出的一階梯度。

**One-sentence summary**: Training is a search in parameter space for the parameters with the smallest loss; because curvature is too expensive, deep learning uses only the first-order gradient that backpropagation computes cheaply.

**白話說明**：損失只給分數，最佳化器負責決定怎麼走；每一步是預測、算損失、求梯度、更新，重複到停止。梯度告訴你哪邊最陡，曲率告訴你碗有多彎，但曲率的帳單隨參數數平方成長。

**Plain explanation**: The loss only gives scores and the optimizer decides how to move; each step predicts, scores the loss, differentiates and updates, repeating until it stops. The gradient tells you which side is steepest and the curvature tells you how the bowl bends, but the curvature bill grows with the square of the parameter count.

**正式定義**：目標 $\min_\theta\mathcal{L}(\theta)$；一階更新使用 $\nabla\mathcal{L}(\theta_t)$ 選出 $\theta_{t+1}$；牛頓更新 $\theta_{t+1}=\theta_t-H^{-1}\nabla\mathcal{L}(\theta_t)$ 另需 $H=\nabla^2\mathcal{L}(\theta_t)$，儲存 $O(d^2)$、求逆 $O(d^3)$。

**Formal definition**: The goal is $\min_\theta\mathcal{L}(\theta)$; a first-order update uses $\nabla\mathcal{L}(\theta_t)$ to choose $\theta_{t+1}$; the Newton update $\theta_{t+1}=\theta_t-H^{-1}\nabla\mathcal{L}(\theta_t)$ also needs $H=\nabla^2\mathcal{L}(\theta_t)$, with $O(d^2)$ storage and $O(d^3)$ inversion.

**例子**：玩具碗 $f=\tfrac12(x^2+25y^2)$、$\theta_0=(10,2)$：$\nabla f=(10,50)$；負梯度步（$\eta=0.06$）到 $(9.4,-1.0)$，損失 100 降到 56.68；牛頓步到 $(0,0)$，損失 0。逐步計算見第 2、3 節。

**Example**: For the toy bowl $f=\tfrac12(x^2+25y^2)$ with $\theta_0=(10,2)$: $\nabla f=(10,50)$; the negative-gradient step ($\eta=0.06$) reaches $(9.4,-1.0)$ and lowers the loss from 100 to 56.68; the Newton step reaches $(0,0)$ with loss 0. The step-by-step computations are in sections 2 and 3.

**常見迷思**：

- 「損失會告訴最佳化器怎麼走」——不會，損失只替參數打分。
- 「梯度指向最低點」——不一定，它只是這一點最陡的上坡方向；在狹長的碗裡負梯度幾乎不指向谷底。
- 「二階方法一定比較好」——小問題上牛頓法很強，但在 $d=10^6$ 時成本無法負擔。
- 「一階方法就是亂走」——一階方法每一步都有明確的方向與規則，後面各節會一步步改良它。

**Common misconceptions**:

- "The loss tells the optimizer how to move." It does not; the loss only scores parameters.
- "The gradient points at the minimum." Not necessarily; it is only the steepest uphill direction at this point, and in an elongated bowl the negative gradient hardly points at the valley floor.
- "Second order is always better." Newton's method is strong on small problems, but at $d=10^6$ its cost is unaffordable.
- "First order just wanders." Each first-order step has a definite direction and rule, and the following sections improve it step by step.

**圖**：本單元兩張圖，`lec03-kp01-contours`（梯度與負梯度的方向，功能 1、3）與 `lec03-kp01-newton-gd`（牛頓步與負梯度步的對照，功能 2、3），皆以 `assets/figures/src/lec03-kp01-search-figures.py` 用同一份資料繪製。

**Figure**: Two figures in this unit, `lec03-kp01-contours` (directions of the gradient and the negative gradient; functions 1, 3) and `lec03-kp01-newton-gd` (the Newton step against the negative-gradient step; functions 2, 3), both drawn from the same data by `assets/figures/src/lec03-kp01-search-figures.py`.

**想深入**（選讀）：牛頓法在二次函數上精確，是因為二次函數 $f=\tfrac12\theta^\top H\theta$ 的梯度正好是 $H\theta$，所以 $H^{-1}\nabla f=\theta$，一步到最小點。真實網路的損失不是二次函數，$H$ 隨位置改變，牛頓法只在局部近似成立，而且 $H$ 有負特徵值（鞍點附近）時牛頓步可能朝錯的方向走。另外，一階方法的「雜訊與泛化」關係留到 KP07 與 KP12。

**Going deeper** (optional): Newton's method is exact on a quadratic because the gradient of $f=\tfrac12\theta^\top H\theta$ is $H\theta$, so $H^{-1}\nabla f=\theta$ and one step reaches the minimum. A real network's loss is not quadratic, $H$ changes with position, so Newton's method holds only locally, and near saddle points where $H$ has negative eigenvalues a Newton step can head in the wrong direction. The relation between first-order noise and generalization is left for KP07 and KP12.

**小練習**：

1. 用自己的話說明：為什麼損失函數不能取代最佳化器？（提示：想想「排名次」和「指方向」的差別。）
2. 對 $d=5\times10^4$ 的網路，計算 $H$ 的記憶體（float32）、梯度的記憶體，以及求逆約需多少秒（每秒 $10^{13}$ 次）。（提示：先算 $d^2$，再乘 4 bytes；$d^3$ 除以 $10^{13}$。）
3. 在玩具碗上，$\eta=0.06$ 時 $y$ 每步乘 $-0.5$、$x$ 每步乘 $0.94$。算出 $y$ 與 $x$ 的前兩步，看看它們各自怎麼變。（提示：$y$ 的初值是 2，$x$ 的初值是 10。）

**Practice**:

1. In your own words, explain why the loss function cannot replace the optimizer. (Hint: think about the difference between ranking and pointing a direction.)
2. For a network with $d=5\times10^4$, compute the memory of $H$ (float32), the memory of the gradient, and about how many seconds inversion takes (at $10^{13}$ operations per second). (Hint: compute $d^2$ first, then multiply by 4 bytes; divide $d^3$ by $10^{13}$.)
3. On the toy bowl with $\eta=0.06$, $y$ is multiplied by $-0.5$ and $x$ by $0.94$ at each step. Compute the first two steps of $y$ and of $x$ and see how each changes. (Hint: $y$ starts at 2 and $x$ starts at 10.)

**解題示範**（以 $d=2\times10^5$ 示範第 2 題的算法）：

1. 梯度：$4d=4\times2\times10^5=8\times10^5$ bytes $=800$ KB。
2. 海森：$d^2=(2\times10^5)^2=4\times10^{10}$，乘 4 bytes 得 $1.6\times10^{11}$ bytes $=160$ GB。
3. 求逆：$d^3=(2\times10^5)^3=8\times10^{15}$ 次，除以 $10^{13}$ 次／秒得 $800$ 秒，約 13.3 分鐘。
4. 檢查：和表中 $d=10^4$ 比，$d$ 變成 20 倍，$H$ 的記憶體從 400 MB 變成 160 GB（400 倍），時間從 0.1 s 變成 800 s（8000 倍），與 $d^2$、$d^3$ 的成長一致。

**Worked solution** (the method of exercise 2, shown with $d=2\times10^5$):

1. Gradient: $4d=4\times2\times10^5=8\times10^5$ bytes $=800$ KB.
2. Hessian: $d^2=(2\times10^5)^2=4\times10^{10}$; times 4 bytes gives $1.6\times10^{11}$ bytes $=160$ GB.
3. Inversion: $d^3=(2\times10^5)^3=8\times10^{15}$ operations; divided by $10^{13}$ operations per second gives $800$ seconds, about 13.3 minutes.
4. Check: compared with $d=10^4$ in the table, $d$ is 20 times larger, the Hessian memory grows from 400 MB to 160 GB (400 times) and the time from 0.1 s to 800 s (8000 times), matching the growth of $d^2$ and $d^3$.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 假設類別 | hypothesis class |
| 損失（函數） | loss (function) |
| 最佳化器 | optimizer |
| 梯度 | gradient |
| 學習率 | learning rate |
| 海森矩陣 | Hessian matrix |
| 牛頓法 | Newton's method |
| 一階方法／二階方法 | first-order method / second-order method |
| 曲率 | curvature |
| 鞍點 | saddle point |
