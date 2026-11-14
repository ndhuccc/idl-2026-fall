# 第13週 KP04：納許均衡與訓練為什麼難，終點存在，但梯度會繞著它轉（Nash equilibrium and why training is hard: the end point exists, but gradients circle around it）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第13週的第四個知識點（KP04），接在 KP03 之後。KP03 證明：判別器最優時，生成器的最好答案是 $p_g=p_{\text{data}}$，這時判別器只能猜 $\tfrac12$。這一節先替這個「雙方都不想動」的狀態取名字，再用一個兩個變數的小賽局說明：就算終點存在，用梯度去追它也可能繞圈、甚至越繞越遠。

This is the fourth knowledge point (KP04) of Week 13, following KP03. KP03 proved that with an optimal discriminator, the generator's best answer is $p_g=p_{\text{data}}$, where the discriminator can only guess $\tfrac12$. This note first names this state where "neither side wants to move", then uses a small two-variable game to show that even when the end point exists, chasing it with gradients may circle around it, or even drift further away.

先備知識是 KP02 的交替更新、KP03 的最優判別器，以及第3週的梯度下降與學習率。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第14章 KP-5 改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。圖與實驗的數字都由 `assets/figures/src/lec13-kp04-figures.py` 算出，其餘小算例可以用計算機驗算。

The prerequisites are KP02's alternating updates, KP03's optimal discriminator, and the gradient descent and learning rate of Week 3. Every term and symbol is defined when it first appears. This note is rewritten from KP-5 of Chapter 14 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. The figures and experimental numbers are computed by `assets/figures/src/lec13-kp04-figures.py`; the other small calculations can be checked with a calculator.

## KP04 納許均衡與訓練為什麼難（Nash equilibrium and why training is hard）

### 學習目標（Learning objectives）

讀完之後，你能說出納許均衡的定義，並指出 GAN 的納許均衡是 $p_g=p_{\text{data}}$、$D=\tfrac12$。你能說明它是一個鞍點，而不是一個損失的最低點。你能在玩具賽局 $\min_x\max_y\,xy$ 裡畫出梯度的方向，說明連續時間的軌跡為什麼繞圓。你能算出同時更新時半徑每步放大 $\sqrt{1+\eta^2}$ 倍，並說出交替更新為什麼會繞圈卻不收斂。你也能從小 GAN 的訓練曲線看出這種擺盪，說出實務上該注意什麼。

After reading, you can state the definition of a Nash equilibrium and identify the GAN's Nash equilibrium as $p_g=p_{\text{data}}$, $D=\tfrac12$. You can explain that it is a saddle point, not the bottom of a loss. In the toy game $\min_x\max_y\,xy$ you can draw the gradient directions and explain why the continuous-time path circles. You can compute that with simultaneous updates the radius grows by $\sqrt{1+\eta^2}$ per step, and explain why alternating updates circle without converging. You can also spot this oscillation in a small GAN's training curve and say what to watch for in practice.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $V(D,G)$ | KP02 的目標函數 | the objective of KP02 |
| $p_{\text{data}},\ p_g$ | 真資料與生成資料的分布 | distributions of real and generated data |
| $x,\ y$ | 玩具賽局的兩個玩家：$x$ 想讓 $V$ 小，$y$ 想讓 $V$ 大 | the two players of the toy game: $x$ wants $V$ small, $y$ wants $V$ large |
| $\eta$ | 學習率 | learning rate |
| $\dot{x},\ \dot{y}$ | $x$、$y$ 隨時間的變化率 | rates of change of $x$ and $y$ over time |
| $r$ | 點 $(x,y)$ 到原點的距離 $\sqrt{x^2+y^2}$ | distance of the point $(x,y)$ from the origin, $\sqrt{x^2+y^2}$ |
| $\lvert b\rvert$ | KP01 小 GAN 生成器的寬度（標準差） | width (standard deviation) of the generator of KP01's small GAN |

### 1. 雙方都不想動：納許均衡（Neither side wants to move: the Nash equilibrium）

賽局理論裡，**納許均衡**（Nash equilibrium）是這樣的狀態：任何一個玩家單獨改變自己的做法，都不會得到更好的結果。不是大家都滿意，而是「別人不動的話，我動了也沒好處」。

In game theory, a **Nash equilibrium** is a state in which no player can do better by changing their own strategy alone. It is not that everyone is happy, but that "if the others stay put, moving gains me nothing".

來源指出，GAN 的全域納許均衡就是 KP03 的終點：

The source points out that the GAN's global Nash equilibrium is exactly KP03's end point:

$$
p_g(\mathbf{x})=p_{\text{data}}(\mathbf{x}),\qquad D^*(\mathbf{x})=\tfrac12 .
$$

兩邊都檢查一次 `[補充：理由]`。判別器這邊：生成器已經是 $p_{\text{data}}$，KP03 說最好的評分是 $p_{\text{data}}/(p_{\text{data}}+p_g)=\tfrac12$，它改成別的評分只會讓 $V$ 變小。生成器這邊：判別器處處給 $\tfrac12$，$V=\log\tfrac12+\log\tfrac12$ 和生成器長什麼樣子無關，它怎麼改都拿不到更好的 $V$。所以兩邊都沒有理由單獨改變。

Check both sides `[Added: reason]`. The discriminator: the generator is already $p_{\text{data}}$, and KP03 says the best score is $p_{\text{data}}/(p_{\text{data}}+p_g)=\tfrac12$; any other score only makes $V$ smaller. The generator: the discriminator gives $\tfrac12$ everywhere, so $V=\log\tfrac12+\log\tfrac12$ does not depend on what the generator looks like, and no change gives it a better $V$. So neither side has a reason to change alone.

這個點和第3週的「損失最低點」不一樣。一般訓練只有一個人在把一個損失往下推；GAN 是一個人往下推、一個人往上推，均衡點對生成器是最低、對判別器是最高，叫做**鞍點**（saddle point），像馬鞍中間：前後方向是低點，左右方向是高點。

This point differs from the "lowest point of a loss" of Week 3. Ordinary training has one player pushing one loss down; a GAN has one pushing down and one pushing up, and the equilibrium is lowest for the generator and highest for the discriminator. It is called a **saddle point**, like the middle of a horse saddle: low along one direction and high along the other.

常見錯誤：以為納許均衡是兩個網路「都贏了」的狀態。在 GAN 的均衡點，判別器只能瞎猜一半一半，它完全輸掉了分辨真假的任務；均衡只表示雙方都無法再單方面改善。

Common mistake: thinking a Nash equilibrium is a state where both networks "win". At the GAN equilibrium the discriminator can only guess fifty-fifty, having completely lost its task of telling real from fake; equilibrium only means neither side can improve alone any more.

### 2. 最小的對抗賽局：$V=xy$（The smallest adversarial game: $V=xy$）

GAN 太複雜，來源用一個只有兩個數的玩具賽局來看梯度怎麼走：$x$ 想讓 $V=xy$ 變小，$y$ 想讓它變大，$\min_x\max_y\,xy$。它的納許均衡是原點 $(0,0)$：若 $y=0$，不管 $x$ 是多少，$V$ 都是 $0$，$x$ 改了也沒用；若 $x=0$，$y$ 改了也沒用。

A GAN is too complicated, so the source uses a toy game with only two numbers to see how gradients move: $x$ wants $V=xy$ smaller and $y$ wants it larger, $\min_x\max_y\,xy$. Its Nash equilibrium is the origin $(0,0)$: if $y=0$, then $V=0$ whatever $x$ is, so changing $x$ is useless; if $x=0$, changing $y$ is useless.

各自往自己想要的方向走：$x$ 做梯度下降，方向是 $-\partial V/\partial x=-y$；$y$ 做梯度上升，方向是 $+\partial V/\partial y=x$。把兩個方向畫成箭頭，就是下面這張圖。

Each moves in the direction it wants: $x$ does gradient descent, with direction $-\partial V/\partial x=-y$; $y$ does gradient ascent, with direction $+\partial V/\partial y=x$. Drawing the two directions as arrows gives the figure below.

![一張正方形的圖，橫軸 x 是想讓 V 變小的一方，像生成器；縱軸 y 是想讓 V 變大的一方，像判別器。背景顏色是 V=xy：右上與左下偏紅表示正，左上與右下偏藍表示負。整片深灰色小箭頭繞著中心逆時針轉，沒有一支指向中心；中心有一顆橘色星星，標著納許均衡（A square plot. The horizontal axis x is the side that wants V smaller, like the generator; the vertical axis y is the side that wants V larger, like the discriminator. The background colour shows V=xy: reddish (positive) at the top right and bottom left, bluish (negative) at the top left and bottom right. Small dark slate arrows everywhere turn anticlockwise around the centre, and none points to it; an orange star at the centre is labelled Nash equilibrium）](../assets/figures/lec13-kp04-vector-field.png)

**圖說 `lec13-kp04-vector-field`**〔功能：一目了然、空間幾何具象化〕：先看中心的星星，那是均衡點。再看箭頭：每一支都和「指向中心的方向」垂直，整片繞著中心逆時針轉。這就是問題所在：雙方都照自己的梯度走，合起來卻是繞圈，而不是往均衡點靠近。

**Caption `lec13-kp04-vector-field`** [Function: seeing it at a glance, making space concrete]: first look at the star in the centre, the equilibrium. Then look at the arrows: each is perpendicular to the direction pointing to the centre, and together they turn anticlockwise around it. That is the problem: each side follows its own gradient, yet together they circle instead of approaching the equilibrium.

把步子想成無限小，就是來源寫的連續時間動態 $\dot{x}=-y$、$\dot{y}=x$。到原點的距離不變 `[補充：推導]`：

Making the steps infinitely small gives the source's continuous-time dynamics $\dot{x}=-y$, $\dot{y}=x$. The distance to the origin does not change `[Added: derivation]`:

$$
\frac{d}{dt}\big(x^2+y^2\big)=2x\dot{x}+2y\dot{y}=2x(-y)+2y(x)=0 .
$$

所以從 $(1,0)$ 出發，點會永遠沿著半徑 $1$ 的圓轉，既不靠近、也不遠離原點。

So starting from $(1,0)$, the point circles forever on the circle of radius $1$, neither approaching nor leaving the origin.

常見錯誤：以為「每個玩家都往自己有利的方向走」，就一定會走到均衡點。單一損失的梯度下降會一路往下，但兩個方向相反的目標合在一起，可能像這裡一樣只是繞圈。

Common mistake: thinking that if "each player moves in its own favourable direction", they must reach the equilibrium. Gradient descent on a single loss keeps going down, but two opposite objectives together may just circle, as here.

### 3. 真正一步一步走：同時更新會越繞越遠，交替更新會一直繞（Real discrete steps: simultaneous updates spiral out, alternating updates keep circling）

訓練時步子不是無限小，學習率 $\eta$ 是一個固定的數。來源說，用離散的梯度更新，軌跡不會收斂到原點，反而會螺旋往外。這裡要分兩種做法 `[補充]`。

In training, steps are not infinitely small; the learning rate $\eta$ is a fixed number. The source says that with discrete gradient updates the path does not converge to the origin but spirals outward. Two ways of updating must be distinguished here `[Added]`.

**同時更新**（simultaneous updates）：兩人都用舊的位置算方向，一起走一步，$x\leftarrow x-\eta y$、$y\leftarrow y+\eta x$。新的距離平方是 $(x-\eta y)^2+(y+\eta x)^2=(1+\eta^2)(x^2+y^2)$，所以每一步半徑都放大 $\sqrt{1+\eta^2}$ 倍。$\eta=0.15$ 時是 $1.0112$ 倍，$100$ 步後半徑從 $1$ 變成 $(1+0.15^2)^{50}=3.04$。這就是來源說的往外螺旋。

**Simultaneous updates**: both compute their direction from the old position and step together, $x\leftarrow x-\eta y$, $y\leftarrow y+\eta x$. The new squared distance is $(x-\eta y)^2+(y+\eta x)^2=(1+\eta^2)(x^2+y^2)$, so every step multiplies the radius by $\sqrt{1+\eta^2}$. With $\eta=0.15$ that is $1.0112$, and after $100$ steps the radius grows from $1$ to $(1+0.15^2)^{50}=3.04$. This is the outward spiral the source describes.

**交替更新**（alternating updates）：KP02 的做法，先動一個、另一個再看著新的位置動，$x\leftarrow x-\eta y$，然後 $y\leftarrow y+\eta x_{\text{新}}$（KP02 是判別器先動；這裡讓 $x$ 先動，換成 $y$ 先動結論一樣，$100$ 步後半徑是 $1.039$）。這一步可以寫成矩陣 $\begin{pmatrix}1&-\eta\\ \eta&1-\eta^2\end{pmatrix}$ 乘上 $(x,y)$，它的行列式是 $1$，當 $\eta<2$ 時，兩個特徵值的大小也都是 $1$。所以半徑不會隨時間越來越大，點沿著一個固定、略微傾斜的橢圓一直轉（`[補充]` 這一步保持不變的量是 $x^2+y^2-\eta xy$）：$\eta=0.15$ 時，$100$ 步裡半徑只在 $0.964$ 到 $1.040$ 之間，大約每 $41.8$ 步轉一圈。它不會飛走，但也永遠到不了原點。

**Alternating updates**: KP02's way, where one moves first and the other then moves while looking at the new position, $x\leftarrow x-\eta y$ and then $y\leftarrow y+\eta x_{\text{new}}$ (KP02 moves the discriminator first; here $x$ moves first, and letting $y$ move first gives the same conclusion, radius $1.039$ after $100$ steps). This step can be written as the matrix $\begin{pmatrix}1&-\eta\\ \eta&1-\eta^2\end{pmatrix}$ times $(x,y)$; its determinant is $1$, and when $\eta<2$ both eigenvalues have size $1$. So the radius does not grow over time, and the point keeps going round a fixed, slightly tilted ellipse (`[Added]` the quantity this step keeps unchanged is $x^2+y^2-\eta xy$): with $\eta=0.15$, over $100$ steps the radius stays between $0.964$ and $1.040$, and it turns once about every $41.8$ steps. It does not fly away, but it never reaches the origin either.

![左中右三張圖，橫軸 x、縱軸 y，範圍都是 -3 到 3，中心有一顆星星是均衡點，黑色方塊 (1,0) 是起點。左圖連續時間：一個半徑 1 的青色圓，寫著半徑一直是 1。中圖同時更新、學習率 0.15：紅色的點一圈比一圈遠，螺旋往外，100 步後半徑 3.04。右圖交替更新、學習率 0.15：藍色的點沿著半徑約 1 的圓轉，100 步後半徑 0.96（Three panels with x on the horizontal axis and y on the vertical, both from -3 to 3; a star at the centre is the equilibrium and a black square at (1,0) is the start. Left, continuous time: a teal circle of radius 1, labelled radius stays 1. Middle, simultaneous updates with learning rate 0.15: red points move further out each turn, spiralling outward, radius 3.04 after 100 steps. Right, alternating updates with learning rate 0.15: blue points go round a circle of radius about 1, radius 0.96 after 100 steps）](../assets/figures/lec13-kp04-trajectories.png)

**圖說 `lec13-kp04-trajectories`**〔功能：一目了然、精準度、空間幾何具象化〕：三張都從黑色方塊 $(1,0)$ 出發，目標都是中間的星星。先看左圖：步子無限小時沿圓轉。再看中圖：同時更新每步多走出去一點，$100$ 步後離中心 $3.04$。最後看右圖：交替更新留在圓上繞，$100$ 步後是 $0.96$。沒有一種走到星星。

**Caption `lec13-kp04-trajectories`** [Function: seeing it at a glance, precision, making space concrete]: all three start from the black square $(1,0)$ and all aim at the star in the middle. First look at the left panel: with infinitely small steps the point circles. Then the middle panel: simultaneous updates drift a little further out every step, reaching $3.04$ from the centre after $100$ steps. Finally the right panel: alternating updates stay on the circle, at $0.96$ after $100$ steps. None reaches the star.

把學習率調小並不能解決同時更新的問題，只會讓它慢一點：$100$ 步後，$\eta=0.05$ 的半徑是 $1.133$，$\eta=0.3$ 的半徑是 $74.4$。對交替更新這種一直繞圈、不會飛走的情形，有一個簡單的補救 `[補充]`：不看最後一步，而是看所有步的平均位置（對同時更新的往外螺旋，平均也救不回來）。交替更新走 $4000$ 步，所有點的平均是 $(-0.0006,\ 0.0031)$，幾乎就是原點；繞圈的點平均起來，會落在圈的中心。

Making the learning rate smaller does not fix simultaneous updates; it only slows them down: after $100$ steps the radius is $1.133$ for $\eta=0.05$ and $74.4$ for $\eta=0.3$. For the alternating case, which keeps circling without flying away, there is a simple remedy `[Added]`: look not at the last step but at the average position over all steps (for the outward spiral of simultaneous updates, averaging does not help). Alternating updates over $4000$ steps have an average point of $(-0.0006,\ 0.0031)$, almost exactly the origin; points that circle average out to the centre of the circle.

常見錯誤：以為來源說的「往外螺旋」就是 KP02 的交替更新。在這個玩具賽局裡，往外螺旋的是同時更新；交替更新（學習率不太大時）不會發散，而是一直繞圈。兩者都不收斂，但壞的方式不同。

Common mistake: thinking the source's "outward spiral" is KP02's alternating update. In this toy game it is the simultaneous update that spirals out; the alternating update (with a learning rate that is not too large) does not diverge but keeps circling. Neither converges, but they fail in different ways.

### 4. 在真正的 GAN 裡看到的擺盪（The oscillation seen in a real GAN）

玩具賽局的「繞圈」在 GAN 裡看起來是什麼樣子？回到 KP01 的一維小 GAN `[補充]`，同樣的設定（每步判別器更新 $10$ 次、學習率 $0.1$ 與 $0.005$、每批 $256$、種子 $0$），這次跑 $3000$ 步，這次追蹤生成器的寬度 $\lvert b\rvert$，目標是真資料的 $0.5$。

What does the toy game's "circling" look like in a GAN? Back to KP01's one-dimensional GAN `[Added]` with the same settings ($10$ discriminator updates per step, learning rates $0.1$ and $0.005$, batch $256$, seed $0$), this time running $3000$ steps and tracking the generator's width $\lvert b\rvert$, whose target is the real $0.5$.

![一張折線圖，橫軸是訓練步數 0 到 3000，縱軸是生成器寬度 |b|，從 0 到 1.1。藍色虛線在 0.5，標著真資料寬度。橘色曲線先短暫升到約 1.03，再急降，在第 523 步到最低 0.229，再升到第 1141 步的 0.670，又降到第 1665 步的 0.413，之後擺幅越來越小，到 3000 步約在 0.5；三個轉折點用紅點標出（A line chart with training steps 0 to 3000 on the horizontal axis and the generator width |b| from 0 to 1.1 on the vertical axis. A dashed blue line at 0.5 is labelled real width. The orange curve, after a brief rise to about 1.03, drops sharply to a low of 0.229 at step 523, rises to 0.670 at step 1141, falls to 0.413 at step 1665, then swings less and less, sitting near 0.5 by step 3000; the three turning points are marked with red dots）](../assets/figures/lec13-kp04-gan-oscillation.png)

**圖說 `lec13-kp04-gan-oscillation`**〔功能：一目了然、精準度〕：先看藍色虛線，那是目標寬度。再看橘色曲線：它沒有直接停在 $0.5$，而是先衝過頭到 $0.229$，再反彈到 $0.670$，又回到 $0.413$，一次比一次接近。三個紅點標出這些轉折。

**Caption `lec13-kp04-gan-oscillation`** [Function: seeing it at a glance, precision]: first look at the dashed blue line, the target width. Then look at the orange curve: it does not settle straight at $0.5$ but overshoots to $0.229$, bounces back to $0.670$, returns to $0.413$, getting closer each time. The three red dots mark these turns.

這是同一種「你追我、我追你」的現象：生成器太窄時，判別器學會說「兩邊才是真的」，生成器就變寬；變得太寬，判別器又改口。在這個例子裡擺盪慢慢變小，$\lvert b\rvert$ 停在 $0.5$ 附近（第 $3000$ 步是 $0.494$，最後 $500$ 步仍在 $0.469$ 到 $0.499$ 之間小幅擺動）。判別器每步先更新 $10$ 次有幫助：同樣的種子改成每步只更新 $1$ 次，第 $3000$ 步的 $\lvert b\rvert$ 反而縮到約 $0.15$，生成器變得太窄。大型網路沒有這樣的保證：擺盪可能一直持續，或越來越大。

This is the same "you chase me, I chase you" pattern: when the generator is too narrow, the discriminator learns to say "the sides are real", and the generator widens; when it becomes too wide, the discriminator changes its mind again. In this example the swings shrink and $\lvert b\rvert$ settles near $0.5$ ($0.494$ at step $3000$, still swinging slightly between $0.469$ and $0.499$ over the last $500$ steps). The $10$ discriminator updates per step help: with the same seed and only $1$ update per step, $\lvert b\rvert$ instead shrinks to about $0.15$ at step $3000$, and the generator becomes too narrow. Large networks have no such guarantee: the swings may continue, or grow.

所以訓練 GAN 時要注意幾件事 `[補充]`：損失曲線上下擺動是正常的，不能只看「損失有沒有一直下降」來判斷好壞；要讓判別器和生成器勢均力敵，學習率不要太大；看生成的樣本，或看生成器參數在多步之間的平均（例如移動平均）。更有系統的改良（模式崩塌的對策、換一把尺）在 KP06 與 KP07。

So when training a GAN, watch a few things `[Added]`: loss curves going up and down are normal, so you cannot judge progress only by "is the loss going down"; keep the discriminator and generator evenly matched and the learning rates not too large; look at generated samples, or at the average of the generator's parameters over many steps (for example a moving average). More systematic improvements (remedies for mode collapse, a different ruler) come in KP06 and KP07.

常見錯誤：以為 GAN 的損失一直下降才代表訓練正常。判別器與生成器的損失互相拉扯，本來就會上下擺動；均衡點是鞍點，不是一個損失的最低點。

Common mistake: thinking GAN training is healthy only if the loss keeps going down. The discriminator's and generator's losses pull against each other and naturally swing up and down; the equilibrium is a saddle point, not the bottom of one loss.

### 本單元重點整理（KP04 summary）

**一句話重點**：GAN 的納許均衡是 $p_g=p_{\text{data}}$、$D=\tfrac12$，它是鞍點而不是損失的最低點；在最小的對抗賽局 $\min_x\max_y\,xy$ 裡，梯度方向繞著均衡點轉，連續時間沿圓走，同時更新每步半徑放大 $\sqrt{1+\eta^2}$ 倍、螺旋往外，交替更新一直繞圈；真正的 GAN 也會擺盪，所以要看樣本與生成器參數的平均，不能只看損失。

**One-sentence summary**: the GAN's Nash equilibrium is $p_g=p_{\text{data}}$ with $D=\tfrac12$, a saddle point rather than the bottom of a loss; in the smallest adversarial game $\min_x\max_y\,xy$ the gradient directions turn around the equilibrium, the continuous-time path follows a circle, simultaneous updates multiply the radius by $\sqrt{1+\eta^2}$ each step and spiral out, and alternating updates keep circling; real GANs oscillate too, so look at samples and at averages of the generator's parameters, not only at the loss.

**白話說明**：兩個人在玩「你往左我就往右」的遊戲。理論上有一個兩人都不想再動的位置，可是每個人都只看對方現在站哪裡就行動，結果兩人一直繞著那個位置轉。步子跨大一點，甚至越繞越遠。

**Plain explanation**: two people play "if you go left, I go right". In theory there is a spot where neither wants to move any more, but each acts only on where the other stands right now, so they keep circling that spot. With bigger strides they even drift further away.

**正式定義**：$(D^\dagger,G^\dagger)$ 是納許均衡，若 $V(D,G^\dagger)\le V(D^\dagger,G^\dagger)\le V(D^\dagger,G)$ 對所有 $D$、$G$ 成立；GAN 的全域均衡 $p_{g^\dagger}=p_{\text{data}}$、$D^\dagger\equiv\tfrac12$。玩具賽局 $\min_x\max_yxy$：連續時間 $\dot{x}=-y$、$\dot{y}=x$；同時更新 $r_{t+1}=\sqrt{1+\eta^2}\,r_t$；交替更新的矩陣 $\begin{pmatrix}1&-\eta\\ \eta&1-\eta^2\end{pmatrix}$ 行列式為 $1$。

**Formal definition**: $(D^\dagger,G^\dagger)$ is a Nash equilibrium if $V(D,G^\dagger)\le V(D^\dagger,G^\dagger)\le V(D^\dagger,G)$ for all $D$ and $G$; the GAN's global equilibrium is $p_{g^\dagger}=p_{\text{data}}$, $D^\dagger\equiv\tfrac12$. Toy game $\min_x\max_yxy$: continuous time $\dot{x}=-y$, $\dot{y}=x$; simultaneous updates $r_{t+1}=\sqrt{1+\eta^2}\,r_t$; the alternating update matrix $\begin{pmatrix}1&-\eta\\ \eta&1-\eta^2\end{pmatrix}$ has determinant $1$.

**例子**：$\eta=0.15$，從 $(1,0)$ 出發 $100$ 步：同時更新半徑 $3.04$，交替更新半徑在 $0.964$ 到 $1.040$ 之間、約每 $41.8$ 步一圈；交替更新 $4000$ 步的平均點 $(-0.0006,\ 0.0031)$；KP01 小 GAN 的寬度 $1\to0.229\to0.670\to0.413\to$ 約 $0.5$。

**Example**: with $\eta=0.15$, $100$ steps from $(1,0)$: simultaneous radius $3.04$; alternating radius between $0.964$ and $1.040$, about one turn per $41.8$ steps; the average point of $4000$ alternating steps is $(-0.0006,\ 0.0031)$; the width of KP01's small GAN goes $1\to0.229\to0.670\to0.413\to$ about $0.5$.

**常見迷思**：

- 「納許均衡表示兩邊都贏。」錯，判別器在均衡點只能猜一半一半。
- 「各自往有利的方向走就會到均衡點。」錯，兩個相反的目標可能讓梯度繞圈。
- 「往外螺旋的就是交替更新。」錯，在玩具賽局裡往外螺旋的是同時更新，交替更新是一直繞。
- 「損失一直下降才是好的訓練。」錯，GAN 的損失本來就會擺盪。

**Common misconceptions**:

- "A Nash equilibrium means both sides win." Wrong: at the equilibrium the discriminator can only guess fifty-fifty.
- "Each moving in its favourable direction reaches the equilibrium." Wrong: two opposite objectives can make the gradients circle.
- "The outward spiral is the alternating update." Wrong: in the toy game the simultaneous update spirals out, while the alternating update keeps circling.
- "Good training means the loss keeps going down." Wrong: GAN losses naturally oscillate.

**圖**：本單元有三張圖。

- `lec13-kp04-vector-field`：$V=xy$ 的更新方向繞著均衡點轉。
- `lec13-kp04-trajectories`：連續時間、同時更新、交替更新三種軌跡。
- `lec13-kp04-gan-oscillation`：KP01 小 GAN 的寬度衝過頭再擺回來。

**Figures** — this unit has three figures.

- `lec13-kp04-vector-field`: the update directions of $V=xy$ turn around the equilibrium.
- `lec13-kp04-trajectories`: the three paths of continuous time, simultaneous updates and alternating updates.
- `lec13-kp04-gan-oscillation`: the width of KP01's small GAN overshoots and swings back.

### 小練習（Practice）

1. 在 $V=xy$ 裡，點在 $(2,\ 1)$，學習率 $\eta=0.1$。用同時更新走一步，新位置是多少？距離原點從多少變成多少？
2. 同時更新、$\eta=0.1$，從半徑 $1$ 出發，走 $200$ 步後半徑約是多少？
3. 一位同學看到 GAN 的生成器損失在訓練中一下高一下低，就認為程式有錯。你會怎麼回答？要確認訓練有沒有進步，可以看什麼？

1. In $V=xy$, the point is at $(2,\ 1)$ and the learning rate is $\eta=0.1$. Take one simultaneous step. Where is the new point? How does its distance from the origin change?
2. With simultaneous updates and $\eta=0.1$, starting from radius $1$, about what is the radius after $200$ steps?
3. A classmate sees a GAN's generator loss go up and down during training and concludes the code is wrong. How would you reply? What can you look at to check whether training is making progress?

**第 1 題解答** `[補充]`：$x\leftarrow2-0.1\times1=1.9$，$y\leftarrow1+0.1\times2=1.2$，新位置 $(1.9,\ 1.2)$。距離從 $\sqrt{5}\approx2.236$ 變成 $\sqrt{1.9^2+1.2^2}=\sqrt{5.05}\approx2.247$，正好放大 $\sqrt{1.01}$ 倍。

**Solution to question 1** `[Added]`: $x\leftarrow2-0.1\times1=1.9$ and $y\leftarrow1+0.1\times2=1.2$, so the new point is $(1.9,\ 1.2)$. The distance changes from $\sqrt{5}\approx2.236$ to $\sqrt{1.9^2+1.2^2}=\sqrt{5.05}\approx2.247$, exactly a factor of $\sqrt{1.01}$.

**第 2 題解答** `[補充]`：$(\sqrt{1.01})^{200}=1.01^{100}\approx2.70$。

**Solution to question 2** `[Added]`: $(\sqrt{1.01})^{200}=1.01^{100}\approx2.70$.

**第 3 題解答** `[補充]`：GAN 的兩個損失互相拉扯，均衡點是鞍點，損失本來就會擺盪，這本身不能當作程式有錯的證據。可以定期看生成的樣本，或像第4節那樣追蹤一個能和真資料比較的量（例如寬度、平均），或看生成器參數在多步之間的平均（例如移動平均）。

**Solution to question 3** `[Added]`: a GAN's two losses pull against each other and the equilibrium is a saddle point, so the losses naturally swing; this alone is not evidence of a bug. Look at generated samples regularly, track a quantity that can be compared with the real data (such as width or mean, as in Section 4), or look at the average of the generator's parameters over many steps (for example a moving average).

### 小測驗（Quiz，來源原題）

來源 KP-5 有一題自我檢測，保留原題與全部選項；來源頁面附有解析，下面依它說明。

The source's KP-5 has one self-check question, kept here with all options; the source page gives an explanation, which the note follows below.

1. GAN 達致理論全局納許均衡 (Nash Equilibrium) 時，判別器 $D^*(\mathbf{x})$ 對任何輸入樣本的輸出值應為多少？
   - (A) 輸出 $1.0$（全部判為真）
   - (B) 輸出 $0.0$（全部判為假）
   - (C) 輸出 $0.5$（完全無法區分真假，機率各半）
   - (D) 輸出與輸入樣本維度相等的矩陣

1. When a GAN reaches its theoretical global Nash equilibrium, what should the discriminator $D^*(\mathbf{x})$ output for any input sample?
   - (A) $1.0$ (everything judged real)
   - (B) $0.0$ (everything judged fake)
   - (C) $0.5$ (it cannot tell real from fake at all; fifty-fifty)
   - (D) A matrix with the same dimension as the input sample

**解析** `[補充]`：選 (C)，見第1節與 KP03：$p_g=p_{\text{data}}$ 時 $D^*=p_{\text{data}}/(p_{\text{data}}+p_g)=\tfrac12$。

**Explanation** `[Added]`: the answer is C; see Section 1 and KP03: when $p_g=p_{\text{data}}$, $D^*=p_{\text{data}}/(p_{\text{data}}+p_g)=\tfrac12$.

### 留給下一單元的問題（The question for the next unit）

前四個知識點都在談 GAN 的核心機制。實際做影像生成時，有兩個很自然的要求：能不能指定要生成哪一類（例如「生成數字 $7$」）？生成器用什麼網路架構，才能畫出清楚的影像？雜訊 $\mathbf{z}$ 本身又有什麼結構？下一個知識點（本週 KP05）介紹條件式 GAN、DCGAN 與潛在空間。

The first four knowledge points discussed the GAN's core mechanism. In real image generation there are two natural wishes: can we choose which class to generate (for example "generate the digit $7$")? What network architecture lets the generator draw sharp images? And what structure does the noise $\mathbf{z}$ itself have? The next knowledge point (this week's KP05) introduces conditional GANs, DCGAN and the latent space.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第14章「產生式對抗網絡」KP-5「納許均衡（Nash Equilibrium）與收斂幾何分析」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch14-gan/index.html ，讀取日期 2026-11-15）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from KP-5, "Nash equilibrium and the geometry of convergence", of Chapter 14, "Generative adversarial networks", of CCC-LNS IDL_115 (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch14-gan/index.html , accessed 2026-11-15). Content not in the source is marked `[Added]` where it appears.

- 來源說「若採用離散梯度更新，軌跡不會收斂到原點，反而會以外擴螺旋線發散」，並以此解釋「GAN 交替 SGD 訓練不穩定」。在 $V=xy$ 裡，往外螺旋的是同時更新（半徑每步乘 $\sqrt{1+\eta^2}$）；KP02 的交替更新在學習率不太大時（$\eta<2$）半徑不變、一直繞圈。兩者都不收斂，但方式不同。本講義第3節分開說明。
- 來源的互動展示說「學習率過大時離散軌跡呈螺旋外擴」，容易讓人以為學習率夠小就會收斂；同時更新在任何正的學習率下都會往外，只是學習率小時比較慢。
- 來源的「教科書圖 14.3」與 AI 生圖提示不重做；本講義改畫向量場、三種軌跡與小 GAN 的寬度曲線。
- 來源用粗斜體 `\boldsymbol` 表示向量；本講義改用 `\mathbf`。
- 本講義補上：均衡兩邊的檢查、鞍點的說明、距離不變的推導、同時與交替更新的計算、迭代平均、小 GAN 的寬度擺盪、實務注意事項、三題小練習與解答。
- 取捨：一般非凸賽局的收斂理論（例如雙時間尺度更新）不寫。

- The source says "with discrete gradient updates the path does not converge to the origin but diverges in an outward spiral", and uses this to explain "the instability of alternating SGD training of GANs". In $V=xy$ it is the simultaneous update that spirals out (the radius is multiplied by $\sqrt{1+\eta^2}$ each step); KP02's alternating update, with a learning rate that is not too large ($\eta<2$), keeps the radius and circles forever. Neither converges, but they fail differently. Section 3 treats them separately.
- The source's interactive demo says "with too large a learning rate the discrete path spirals outward", which suggests a small enough learning rate would converge; the simultaneous update spirals outward for every positive learning rate, only more slowly when it is small.
- The source's "textbook figure 14.3" and AI image prompt are not reproduced; this note draws the vector field, the three paths and the small GAN's width curve instead.
- The source writes vectors with bold italic `\boldsymbol`; this note uses `\mathbf`.
- This note adds: the check of both sides of the equilibrium, the explanation of a saddle point, the derivation that the distance is constant, the calculations for simultaneous and alternating updates, iterate averaging, the small GAN's width oscillation, practical advice, and three practice questions with solutions.
- Choices: convergence theory for general non-convex games (such as two-time-scale updates) is not covered.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 賽局理論 | game theory |
| 納許均衡 | Nash equilibrium |
| 鞍點 | saddle point |
| 玩具賽局 | toy game |
| 連續時間動態 | continuous-time dynamics |
| 同時更新 | simultaneous updates |
| 交替更新（KP02） | alternating updates |
| 行列式 | determinant |
| 特徵值 | eigenvalue |
| 迭代平均 | iterate averaging |
| 擺盪 | oscillation |
| 生成對抗網路 | generative adversarial network (GAN) |
