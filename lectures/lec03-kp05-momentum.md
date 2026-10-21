# 第3週 KP05：動量（Momentum）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第3章「反向傳播與最佳化」的第五個知識點（KP05），承接 KP04 留下的問題：最陡下降的之字形（相鄰梯度互相垂直）能不能抑制？先備知識是 KP01–KP04 與微積分；第一次出現的符號都會定義，所有數字都可以自己重算。內容依 CCC-LNS 第3章 KP05 的主文與投影片消化後重新撰寫，讀取日期 2026-10-14；來源沒有的內容標示 `[補充]`。

This is the fifth knowledge point (KP05) of Chapter 3. It picks up the question KP04 left open: can the zigzag of steepest descent (adjacent gradients orthogonal) be damped? It assumes KP01–KP04 and calculus; every symbol is defined when it first appears and every number can be recomputed. The text is rewritten from the KP05 main text and slides of CCC-LNS Chapter 3 (accessed on 2026-10-14); content that the source does not contain is marked `[Added]`.

[下載本單元投影片](../slides/lec03-kp05-momentum.jbd.json)

## KP05 動量（Momentum）

### 學習目標（Learning objectives）

讀完之後，你能說明為什麼狹長的碗讓一個固定學習率顧不了兩個方向；能寫出動量更新式並手算兩步；能解釋動量為什麼能抵消橫向的來回擺動、累加縱向的前進，並說出穩態的有效步長 $\eta/(1-\alpha)$；也能說明動量的限制：損失不一定單調下降，學習率仍有上限。

After reading, you can explain why one fixed learning rate cannot serve two directions of an elongated bowl; write the momentum update and compute two steps by hand; explain why momentum cancels sideways oscillation and accumulates forward progress, and state the steady-state effective step $\eta/(1-\alpha)$; and describe the limits of momentum: the loss is not monotone and the learning rate still has an upper bound.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\lambda_{\max},\lambda_{\min}$ | 海森矩陣最大、最小的特徵值（各方向的曲率） | largest and smallest Hessian eigenvalues (curvature per direction) |
| $\kappa=\lambda_{\max}/\lambda_{\min}$ | 條件數：最陡與最平方向的曲率比 | condition number: ratio of the steepest to the flattest curvature |
| $\eta$ | 學習率 | learning rate |
| $\alpha\in[0,1)$ | 動量係數：保留多少上一步的位移（常取 0.9） | momentum coefficient: how much of the last displacement is kept (often 0.9) |
| $\Delta w^{(k)}$ | 第 $k$ 步的更新量（位移） | the update (displacement) of step $k$ |

### 1. 為什麼狹長的碗讓純梯度下降這麼慢（Why an elongated bowl slows plain gradient descent）

回到 KP01 的玩具碗 $f(x,y)=\tfrac12(x^2+25y^2)$。它在 $x$ 方向的曲率是 $\lambda=1$，在 $y$ 方向是 $\lambda=25$，所以條件數 $\kappa=25/1=25$。固定學習率 $\eta$ 的梯度下降，在每個方向上是把座標乘上固定倍率 $1-\eta\lambda$：$x$ 乘 $1-\eta$，$y$ 乘 $1-25\eta$（KP02 的 $1-10\eta$ 就是同一個道理）。

Back to the toy bowl $f(x,y)=\tfrac12(x^2+25y^2)$ of KP01. Its curvature is $\lambda=1$ along $x$ and $\lambda=25$ along $y$, so the condition number is $\kappa=25/1=25$. Gradient descent with a fixed learning rate $\eta$ multiplies each coordinate by a fixed factor $1-\eta\lambda$: $x$ by $1-\eta$ and $y$ by $1-25\eta$ (the $1-10\eta$ of KP02 is the same idea).

要穩定，每個方向的倍率絕對值都要小於 1，最苛刻的是最陡的方向：$|1-25\eta|<1$，即 $\eta<2/\lambda_{\max}=2/25=0.08$。這就是一個學習率顧不了兩個方向的困境：

For stability the factor in every direction must have absolute value below 1, and the steepest direction is the most demanding: $|1-25\eta|<1$, i.e. $\eta<2/\lambda_{\max}=2/25=0.08$. This is why one learning rate cannot serve two directions:

| 學習率 $\eta$ | $x$ 倍率 $1-\eta$ | $y$ 倍率 $1-25\eta$ | 40 步後的損失 | 結果 |
|---|---|---|---|---|
| 0.06 | 0.94 | −0.5 | 0.354 | $x$ 方向走得慢 |
| 0.09 | 0.91 | −1.25 | $2.83\times10^{9}$ | $y$ 方向發散 |

| Learning rate $\eta$ | $x$ factor $1-\eta$ | $y$ factor $1-25\eta$ | Loss after 40 steps | Result |
|---|---|---|---|---|
| 0.06 | 0.94 | −0.5 | 0.354 | slow along $x$ |
| 0.09 | 0.91 | −1.25 | $2.83\times10^{9}$ | diverges along $y$ |

調大 $\eta$ 想讓 $x$ 走快一點，$y$ 就發散；調小 $\eta$ 讓 $y$ 穩定，$x$ 就走不動。即使取最好的固定學習率 $\eta^*=2/(\lambda_{\max}+\lambda_{\min})=2/26\approx0.0769$，誤差每步最多只縮小 $(\kappa-1)/(\kappa+1)=24/26\approx0.923$。$\kappa=1000$ 時這個比例是 $999/1001\approx0.998$：走一千步誤差幾乎沒變（`[補充]` 數字：$0.923^{100}\approx3.3\times10^{-4}$，$0.998^{1000}\approx0.135$）。

Raising $\eta$ to speed up $x$ makes $y$ diverge; lowering $\eta$ to stabilize $y$ leaves $x$ crawling. Even the best fixed learning rate $\eta^*=2/(\lambda_{\max}+\lambda_{\min})=2/26\approx0.0769$ shrinks the error by at most a factor of $(\kappa-1)/(\kappa+1)=24/26\approx0.923$ per step. For $\kappa=1000$ the ratio is $999/1001\approx0.998$: after a thousand steps the error has hardly moved (`[Added]` numbers: $0.923^{100}\approx3.3\times10^{-4}$ and $0.998^{1000}\approx0.135$).

我們希望：不用算昂貴的二階資訊（KP01），就能壓抑高曲率方向的擺動，並放大低曲率方向的前進。這就是動量要做的事。

What we want is to damp the oscillation in the high-curvature direction and amplify the progress in the low-curvature direction without computing expensive second-order information (KP01). That is what momentum does.

### 2. 動量更新（The momentum update）

純梯度下降只看「此刻的斜率」。動量讓最佳化器「記得剛才往哪走」：每一步的更新量等於**上一步更新量的一部分**，加上這一步的負梯度：

Plain gradient descent sees only "the slope right now". Momentum lets the optimizer "remember where it was heading": each update is **a fraction of the previous update** plus this step's negative gradient:

$$
\Delta w^{(k)}=\alpha\,\Delta w^{(k-1)}-\eta\,\nabla f\big(w^{(k)}\big),\qquad w^{(k+1)}=w^{(k)}+\Delta w^{(k)}
$$

起點取 $\Delta w^{(-1)}=\mathbf 0$。$\alpha\in[0,1)$ 是動量係數，常取 $0.9$；$\alpha=0$ 時第一式變成 $\Delta w^{(k)}=-\eta\nabla f$，**精確退化成純梯度下降**。這和 PyTorch 等框架的速度形式一樣：令 $v=\Delta w$，則 $v\leftarrow\alpha v-\eta\nabla f$，$w\leftarrow w+v$。物理上，純梯度下降像沒有質量的粒子在黏稠的蜂蜜裡：外力一消失就停；動量給粒子質量，讓它有慣性。

Start with $\Delta w^{(-1)}=\mathbf 0$. The momentum coefficient $\alpha\in[0,1)$ is often $0.9$; at $\alpha=0$ the first equation becomes $\Delta w^{(k)}=-\eta\nabla f$, **reducing exactly to plain gradient descent**. This matches the velocity form of frameworks such as PyTorch: with $v=\Delta w$, $v\leftarrow\alpha v-\eta\nabla f$ and $w\leftarrow w+v$. Physically, plain gradient descent is a massless particle in thick honey that stops when the force stops; momentum gives the particle mass, hence inertia.

**手算兩步**（玩具碗，$\theta_0=(10,2)$，$\eta=0.06$，$\alpha=0.8$）：

**Two steps by hand** (toy bowl, $\theta_0=(10,2)$, $\eta=0.06$, $\alpha=0.8$):

第 1 步：$\nabla f(\theta_0)=(10,50)$，$\Delta w^{(0)}=0.8\cdot(0,0)-0.06\cdot(10,50)=(-0.6,\,-3.0)$，$\theta_1=(10,2)+(-0.6,-3.0)=(9.4,\,-1.0)$。第一步和純梯度下降一樣，因為還沒有「上一步」。

Step 1: $\nabla f(\theta_0)=(10,50)$, $\Delta w^{(0)}=0.8\cdot(0,0)-0.06\cdot(10,50)=(-0.6,\,-3.0)$ and $\theta_1=(10,2)+(-0.6,-3.0)=(9.4,\,-1.0)$. The first step equals plain gradient descent because there is no previous step yet.

第 2 步：$\nabla f(\theta_1)=(9.4,\,25\cdot(-1.0))=(9.4,\,-25)$，$\Delta w^{(1)}=0.8\cdot(-0.6,-3.0)-0.06\cdot(9.4,-25)=(-0.48-0.564,\ -2.4+1.5)=(-1.044,\,-0.9)$，$\theta_2=(9.4,-1.0)+(-1.044,-0.9)=(8.356,\,-1.9)$。純梯度下降的第 2 步是 $(8.836,\,0.5)$。看 $y$ 座標：上一步往下衝（$-3.0$）的慣性還在，這一步的梯度往上拉（$+1.5$），合起來是 $-0.9$，所以擺動已經被削弱；但慣性也讓它衝到 $y=-1.9$，比純梯度下降的 $0.5$ 離 0 更遠。

Step 2: $\nabla f(\theta_1)=(9.4,\,25\cdot(-1.0))=(9.4,\,-25)$, $\Delta w^{(1)}=0.8\cdot(-0.6,-3.0)-0.06\cdot(9.4,-25)=(-0.48-0.564,\ -2.4+1.5)=(-1.044,\,-0.9)$ and $\theta_2=(9.4,-1.0)+(-1.044,-0.9)=(8.356,\,-1.9)$. Plain gradient descent's second step is $(8.836,\,0.5)$. Look at the $y$ coordinate: the inertia of the downward rush ($-3.0$) is still there and this step's gradient pulls back up ($+1.5$), together $-0.9$, so the swing is already weakened; but the inertia also carries it to $y=-1.9$, farther from 0 than the $0.5$ of plain gradient descent.

損失也不是單調的：$f(\theta_1)=56.68$，$f(\theta_2)=\tfrac12(8.356^2+25\cdot1.9^2)=\tfrac12(69.82+90.25)=80.04$，**比上一步高**；純梯度下降的 $f(\theta_2)=42.16$。動量的損失不保證每步下降〔程式計算〕。

The loss is not monotone either: $f(\theta_1)=56.68$ and $f(\theta_2)=\tfrac12(8.356^2+25\cdot1.9^2)=\tfrac12(69.82+90.25)=80.04$, **higher than the last step**; plain gradient descent has $f(\theta_2)=42.16$. Momentum does not guarantee a decrease at every step [computed by program].

### 3. 為什麼有效：抵消與累加（Why it works: cancellation and accumulation）

把遞迴式往回展開，$\Delta w^{(k)}=-\eta\sum_{i=0}^{k}\alpha^i\,\nabla f\big(w^{(k-i)}\big)$：更新量是**歷史梯度的指數加權和**，越舊的梯度權重 $\alpha^i$ 越小（指數移動平均的樣子）。兩個結果：

Unrolling the recursion gives $\Delta w^{(k)}=-\eta\sum_{i=0}^{k}\alpha^i\,\nabla f\big(w^{(k-i)}\big)$: the update is an **exponentially weighted sum of past gradients**, older ones weighted by $\alpha^i$ and fading (an exponential moving average). Two consequences:

- **抵消**：$y$ 方向的梯度每步正負交替（碗很陡，一步就衝過谷底），相加時互相抵消，所以橫向擺動變小。
- **累加**：$x$ 方向的梯度每步都同號（碗很平），相加時越疊越大。若梯度長期是同一個值 $g$，更新量趨近穩態 $\Delta w=-\eta g/(1-\alpha)$，等於把學習率放大成**有效步長 $\eta/(1-\alpha)$**：$\alpha=0.8$ 時是 $0.06/0.2=0.3$，是純梯度下降的 5 倍；$\alpha=0.9$ 時是 10 倍。

- **Cancellation**: the gradients along $y$ alternate in sign from step to step (the bowl is steep, so each step jumps over the valley floor) and cancel when summed, so the sideways swing shrinks.
- **Accumulation**: the gradients along $x$ keep the same sign (the bowl is flat) and pile up when summed. If the gradient stays at one value $g$ for long, the update approaches the steady state $\Delta w=-\eta g/(1-\alpha)$, i.e. the learning rate is amplified into an **effective step $\eta/(1-\alpha)$**: $0.06/0.2=0.3$ for $\alpha=0.8$, 5 times plain gradient descent; 10 times for $\alpha=0.9$.

![純梯度下降（紅）與動量（青綠，α=0.8）在玩具碗上的前 40 步：兩者都從 (10,2) 出發；紅色路徑在 x 方向每步只前進一小段，青綠色路徑橫向擺動被削弱、在 x 方向前進得快得多，並在最小點附近衝過頭再回來。（The first 40 steps of plain gradient descent (red) and momentum (teal, α=0.8) on the toy bowl: both start at (10,2); the red path advances only a little along x per step, the teal path has weaker sideways swing, advances much faster along x, and overshoots and returns near the minimum.）](../assets/figures/lec03-kp05-paths.png)

**圖說 `lec03-kp05-paths`**：先看紅線：每步在 $x$ 方向只前進一點，$y$ 方向來回；再看青綠線：前幾步擺動很大（慣性），之後橫向擺動被抵消，$x$ 方向的前進明顯更快；最後看星號附近：動量衝過最小點後再繞回來。〔功能：1、3〕

**Caption `lec03-kp05-paths`**: Look at the red line first: it advances only a little along $x$ per step and swings back and forth in $y$. Then the teal line: the first few swings are large (inertia), after which the sideways swing is cancelled and the progress along $x$ is clearly faster. Finally, near the star: momentum overshoots the minimum and swings back. [Function: 1, 3]

用程式算同樣的 40 步：損失〔程式計算〕

The same 40 steps by program (losses, [computed by program]):

| 步 $k$ | 1 | 2 | 10 | 20 | 40 |
|---|---|---|---|---|---|
| 純梯度下降 $f$ | 56.68 | 42.16 | 14.51 | 4.208 | 0.3542 |
| 動量 $f$（$\alpha=0.8$） | 56.68 | 80.04 | 3.168 | 0.8270 | 0.0163 |

| Step $k$ | 1 | 2 | 10 | 20 | 40 |
|---|---|---|---|---|---|
| plain GD $f$ | 56.68 | 42.16 | 14.51 | 4.208 | 0.3542 |
| momentum $f$ ($\alpha=0.8$) | 56.68 | 80.04 | 3.168 | 0.8270 | 0.0163 |

動量在第 2 步比較差，到了第 10 步已經好 4 倍以上，第 40 步好 20 倍以上。

Momentum is worse at step 2, but by step 10 it is more than 4 times better and by step 40 more than 20 times.

### 4. 動量的限制（The limits of momentum）

**損失不是每步都降**：上面第 2 步就看到損失上升。動量是「帶著慣性衝」，會衝過頭；整體趨勢是下降的，但不保證單調。

**The loss does not fall at every step**: step 2 above already shows it rising. Momentum is "rushing with inertia" and overshoots; the overall trend is downward but not guaranteed to be monotone.

**學習率仍有上限。**對二次函數可以推出（`[補充]`）動量的穩定範圍是 $0<\eta<2(1+\alpha)/\lambda_{\max}$：$\alpha=0.8$ 時上限從純梯度下降的 $0.08$ 放大到 $2\cdot1.8/25=0.144$。所以 $\eta=0.09$ 在純梯度下降會發散（損失 $2.83\times10^9$），在動量下反而收斂（40 步後損失 $0.00304$）；但 $\eta=0.2$ 超過上限，動量也發散（$\alpha=0.8$ 或 $0.95$ 都發散）。換句話說，動量**放寬**了安全範圍，並沒有讓任意學習率變安全。

**The learning rate still has an upper bound.** For a quadratic one can show (`[Added]`) that the stable range with momentum is $0<\eta<2(1+\alpha)/\lambda_{\max}$: for $\alpha=0.8$ the bound grows from $0.08$ for plain gradient descent to $2\cdot1.8/25=0.144$. So $\eta=0.09$ diverges under plain gradient descent (loss $2.83\times10^9$) but converges under momentum (loss $0.00304$ after 40 steps); yet $\eta=0.2$ exceeds the bound and momentum diverges too (for both $\alpha=0.8$ and $0.95$). In other words, momentum **widens** the safe range but does not make every learning rate safe.

| $\eta$ | $\alpha$ | 40 步後 $\lVert\theta\rVert$ | 40 步後 $f$ | 結果 |
|---|---|---|---|---|
| 0.06 | 0 | 0.842 | 0.354 | 收斂（慢） |
| 0.06 | 0.8 | 0.116 | 0.0163 | 收斂（快） |
| 0.09 | 0 | $1.5\times10^{4}$ | $2.83\times10^{9}$ | 發散 |
| 0.09 | 0.8 | 0.0776 | 0.00304 | 收斂 |
| 0.2 | 0.8 | $1.3\times10^{19}$ | $2.0\times10^{39}$ | 發散 |
| 0.2 | 0.95 | $5.4\times10^{17}$ | $3.7\times10^{36}$ | 發散 |

| $\eta$ | $\alpha$ | $\lVert\theta\rVert$ after 40 steps | $f$ after 40 steps | Result |
|---|---|---|---|---|
| 0.06 | 0 | 0.842 | 0.354 | converges (slowly) |
| 0.06 | 0.8 | 0.116 | 0.0163 | converges (fast) |
| 0.09 | 0 | $1.5\times10^{4}$ | $2.83\times10^{9}$ | diverges |
| 0.09 | 0.8 | 0.0776 | 0.00304 | converges |
| 0.2 | 0.8 | $1.3\times10^{19}$ | $2.0\times10^{39}$ | diverges |
| 0.2 | 0.95 | $5.4\times10^{17}$ | $3.7\times10^{36}$ | diverges |

**多一個要調的超參數。**$\alpha$ 太大，慣性太重，會衝過頭、來回很久；太小，效果有限。常用 $\alpha=0.9$ 是經驗值。動量也沒有改變「只有梯度、沒有曲率」這件事：它只是把過去的梯度累積起來用。

**One more hyperparameter to tune.** With $\alpha$ too large the inertia is heavy and the iterate overshoots and swings for long; too small and the effect is limited. $\alpha=0.9$ is a common empirical value. Momentum also does not change the fact that we have only the gradient and no curvature: it merely accumulates past gradients.

整章到這裡，我們都把梯度當成「已經算好」。下一個問題是：這個梯度到底是怎麼算出來的？這是 KP06 反向傳播的主題。

So far in this chapter we have treated the gradient as "already computed". The next question is: how is this gradient actually computed? That is the topic of KP06, backpropagation.

### 本單元重點整理（KP05 summary）

**一句話重點**：動量把上一步的位移也加進本步更新，橫向正負交替的分量抵消、縱向同向的分量累加，不必二階資訊就能加速狹長的碗；但損失不單調、學習率仍有上限。

**One-sentence summary**: Momentum adds the previous displacement to the current update: sideways components that alternate in sign cancel and forward components with the same sign accumulate, speeding up an elongated bowl without second-order information; but the loss is not monotone and the learning rate still has an upper bound.

**白話說明**：純梯度下降只看腳下，在狹長的山谷兩壁來回彈跳；動量讓它記得剛才往哪走，像重鐵球滾下山谷，左右的推力被時間平均掉，前進的推力一直疊加。代價是會衝過頭，還多一個 $\alpha$ 要調。

**Plain explanation**: Plain gradient descent sees only the ground under its feet and bounces between the walls of a narrow valley; momentum remembers where it was heading, like a heavy iron ball rolling down the valley, where the left-right pushes average out over time and the forward pushes keep adding up. The price is overshooting and one more parameter $\alpha$ to tune.

**正式定義**：$\Delta w^{(k)}=\alpha\Delta w^{(k-1)}-\eta\nabla f(w^{(k)})$，$w^{(k+1)}=w^{(k)}+\Delta w^{(k)}$，$\alpha\in[0,1)$，$\alpha=0$ 即純梯度下降；穩態有效步長 $\eta/(1-\alpha)$。

**Formal definition**: $\Delta w^{(k)}=\alpha\Delta w^{(k-1)}-\eta\nabla f(w^{(k)})$, $w^{(k+1)}=w^{(k)}+\Delta w^{(k)}$ with $\alpha\in[0,1)$; $\alpha=0$ is plain gradient descent; the steady-state effective step is $\eta/(1-\alpha)$.

**例子**：玩具碗、$\eta=0.06$、$\alpha=0.8$：$\theta_1=(9.4,-1.0)$、$\theta_2=(8.356,-1.9)$（純梯度下降 $(8.836,0.5)$）；損失 40 步後 $0.0163$，純梯度下降 $0.354$；$\eta=0.09$ 純梯度下降發散、動量收斂。逐步計算見第 2–4 節。

**Example**: toy bowl, $\eta=0.06$, $\alpha=0.8$: $\theta_1=(9.4,-1.0)$ and $\theta_2=(8.356,-1.9)$ (plain gradient descent: $(8.836,0.5)$); the loss after 40 steps is $0.0163$ against $0.354$ for plain gradient descent; at $\eta=0.09$ plain gradient descent diverges and momentum converges. The computations are in sections 2–4.

**常見迷思**：

- 「動量一定讓每一步都更好」——第 2 步損失還上升。
- 「$\alpha$ 越大越好」——太大會衝過頭、來回很久。
- 「動量能讓任何學習率都安全」——它放寬了上限，但 $\eta=0.2$ 仍然發散。
- 「動量用了二階資訊」——它只累積過去的一階梯度。

**Common misconceptions**:

- "Momentum makes every step better." The loss even rises at step 2.
- "A larger $\alpha$ is better." Too large overshoots and swings for long.
- "Momentum makes every learning rate safe." It relaxes the bound, but $\eta=0.2$ still diverges.
- "Momentum uses second-order information." It only accumulates past first-order gradients.

**圖**：本單元一張圖，`lec03-kp05-paths`（純梯度下降與動量的路徑對照，功能 1、3），以 `assets/figures/src/lec03-kp0345-figures.py` 繪製。

**Figure**: One figure in this unit, `lec03-kp05-paths` (the paths of plain gradient descent and momentum; functions 1, 3), drawn by `assets/figures/src/lec03-kp0345-figures.py`.

**想深入**（選讀）：把動量用在二次函數，每個曲率方向 $\lambda$ 的座標滿足 $z_{k+1}=(1+\alpha-\eta\lambda)\,z_k-\alpha\,z_{k-1}$。特徵方程 $r^2-(1+\alpha-\eta\lambda)r+\alpha=0$ 的判別式為負時兩根是共軛複數，模長都是 $\sqrt\alpha$，所以每個方向每步的收斂比例都是 $\sqrt\alpha$：本講義的玩具碗在 $\alpha=0.8$ 時是 $\sqrt{0.8}\approx0.894$（$x$ 與 $y$ 方向相同），比純梯度下降 $x$ 方向的 $0.94$ 好。選最佳的 $\eta,\alpha$ 時，收斂比例可達 $(\sqrt\kappa-1)/(\sqrt\kappa+1)$（$\kappa=25$ 時 $=2/3$），條件數的影響從 $\kappa$ 降到 $\sqrt\kappa$。另有 Nesterov 動量：先沿著慣性方向試走一步，再在那裡算梯度，細節留給進階課程。

**Going deeper** (optional): For a quadratic with momentum, the coordinate along each curvature direction $\lambda$ satisfies $z_{k+1}=(1+\alpha-\eta\lambda)\,z_k-\alpha\,z_{k-1}$. When the discriminant of the characteristic equation $r^2-(1+\alpha-\eta\lambda)r+\alpha=0$ is negative the two roots are complex conjugates with modulus $\sqrt\alpha$, so every direction contracts by $\sqrt\alpha$ per step: for the toy bowl at $\alpha=0.8$ that is $\sqrt{0.8}\approx0.894$ (the same along $x$ and $y$), better than the $0.94$ of plain gradient descent along $x$. With the best $\eta,\alpha$ the contraction ratio reaches $(\sqrt\kappa-1)/(\sqrt\kappa+1)$ ($=2/3$ for $\kappa=25$): the dependence on the condition number drops from $\kappa$ to $\sqrt\kappa$. There is also Nesterov momentum, which first steps along the inertia direction and computes the gradient there; its details are left to advanced courses.

**小練習**：

1. 玩具碗上取 $\theta_0=(10,2)$、$\eta=0.06$、$\alpha=0.5$，算出 $\theta_1$ 與 $\theta_2$，並與純梯度下降的 $\theta_2=(8.836,\,0.5)$ 比較 $y$ 座標。（提示：$\theta_1=(9.4,-1.0)$ 不變；$\Delta w^{(1)}=0.5\,\Delta w^{(0)}-0.06\,\nabla f(\theta_1)$。）
2. 學習率 $\eta=0.01$、動量係數 $\alpha=0.9$，穩態的有效步長是多少？（提示：$\eta/(1-\alpha)$。）
3. 為什麼 $\eta=0.2$ 在 $\alpha=0.8$ 時仍然發散？用穩定範圍 $2(1+\alpha)/\lambda_{\max}$ 說明。（提示：算出上限，與 $0.2$ 比較。）

**Practice**:

1. On the toy bowl take $\theta_0=(10,2)$, $\eta=0.06$, $\alpha=0.5$; compute $\theta_1$ and $\theta_2$ and compare the $y$ coordinate with $\theta_2=(8.836,\,0.5)$ of plain gradient descent. (Hint: $\theta_1=(9.4,-1.0)$ is unchanged; $\Delta w^{(1)}=0.5\,\Delta w^{(0)}-0.06\,\nabla f(\theta_1)$.)
2. With learning rate $\eta=0.01$ and momentum coefficient $\alpha=0.9$, what is the steady-state effective step? (Hint: $\eta/(1-\alpha)$.)
3. Why does $\eta=0.2$ still diverge at $\alpha=0.8$? Explain with the stable range $2(1+\alpha)/\lambda_{\max}$. (Hint: compute the bound and compare with $0.2$.)

**解題示範**（以 $\eta=0.04$、$\alpha=0.5$ 示範第 1 題的算法）：

1. 第 1 步：$\nabla f(\theta_0)=(10,50)$，$\Delta w^{(0)}=-0.04\cdot(10,50)=(-0.4,\,-2.0)$，$\theta_1=(9.6,\,0)$。
2. 第 2 步的梯度：$\nabla f(\theta_1)=(9.6,\,25\cdot0)=(9.6,\,0)$。
3. 更新量：$\Delta w^{(1)}=0.5\cdot(-0.4,-2.0)-0.04\cdot(9.6,0)=(-0.2-0.384,\ -1.0-0)=(-0.584,\,-1.0)$。
4. 位置：$\theta_2=(9.6,0)+(-0.584,-1.0)=(9.016,\,-1.0)$。
5. 檢查：$y$ 方向這一步的梯度是 0，但慣性（上一步的 $-2.0$ 乘 $\alpha=0.5$）仍把它往下推 $-1.0$，所以衝過 $y=0$，是動量「衝過頭」的小例子。

**Worked solution** (the method of exercise 1, shown with $\eta=0.04$ and $\alpha=0.5$):

1. Step 1: $\nabla f(\theta_0)=(10,50)$, $\Delta w^{(0)}=-0.04\cdot(10,50)=(-0.4,\,-2.0)$ and $\theta_1=(9.6,\,0)$.
2. The gradient for step 2: $\nabla f(\theta_1)=(9.6,\,25\cdot0)=(9.6,\,0)$.
3. The update: $\Delta w^{(1)}=0.5\cdot(-0.4,-2.0)-0.04\cdot(9.6,0)=(-0.2-0.384,\ -1.0-0)=(-0.584,\,-1.0)$.
4. The position: $\theta_2=(9.6,0)+(-0.584,-1.0)=(9.016,\,-1.0)$.
5. Check: along $y$ the gradient of this step is 0, yet inertia (the previous $-2.0$ times $\alpha=0.5$) still pushes it down by $-1.0$, overshooting $y=0$: a small example of momentum overshooting.

### 來源更正與取捨（Corrections to the source）

- 來源說「動量不會把不安全的學習率變安全」。精確的說法是：在二次函數上，動量把穩定上限從 $2/\lambda_{\max}$ 放寬到 $2(1+\alpha)/\lambda_{\max}$，所以純梯度下降發散的 $\eta=0.09$ 在 $\alpha=0.8$ 時收斂；但超過新上限（$\eta=0.2$）仍然發散。本講義用程式驗證這兩點。
- 來源用「動量加速」概括整個效果；本講義補上損失不單調（第 2 步損失從 56.68 升到 80.04），避免誤解成每步都更好。
- 來源的山谷向量合成圖是定性示意；本講義改用玩具碗的實際路徑與數字。來源的條件數界 $\big(\tfrac{\kappa-1}{\kappa+1}\big)^k$ 是正確的，本講義用 $\kappa=25$ 與 $\kappa=1000$ 兩個數字說明它。

- The source says "momentum does not make an unsafe learning rate safe". Precisely: on a quadratic, momentum widens the stable bound from $2/\lambda_{\max}$ to $2(1+\alpha)/\lambda_{\max}$, so $\eta=0.09$, which diverges under plain gradient descent, converges at $\alpha=0.8$; yet beyond the new bound ($\eta=0.2$) it still diverges. This note verifies both points by program.
- The source summarizes the whole effect as "momentum accelerates"; this note adds that the loss is not monotone (it rises from 56.68 to 80.04 at step 2) so that it is not read as better at every step.
- The source's ravine vector-composition figure is a qualitative sketch; this note uses the actual path and numbers of the toy bowl. The source's condition-number bound $\big(\tfrac{\kappa-1}{\kappa+1}\big)^k$ is correct, and this note illustrates it with $\kappa=25$ and $\kappa=1000$.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 動量 | momentum |
| 動量係數 | momentum coefficient |
| 條件數 | condition number |
| 曲率 | curvature |
| 特徵值 | eigenvalue |
| 有效步長 | effective step |
| 指數移動平均 | exponential moving average |
| 慣性 | inertia |
| 衝過頭 | overshoot |
