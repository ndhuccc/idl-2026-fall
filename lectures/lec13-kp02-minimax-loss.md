# 第13週 KP02：極小化極大賽局，把兩個相反的願望寫成一條式子（The minimax game: write two opposite wishes as one formula）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第13週的第二個知識點（KP02），接在 KP01 之後。KP01 只用文字說了：判別器想把真的判高、假的判低，生成器想讓假的被判高。這一節把這兩個願望寫成同一條式子，說明怎麼輪流用梯度去追它，以及實務上生成器為什麼要換一個損失。

This is the second knowledge point (KP02) of Week 13, following KP01. KP01 said only in words that the discriminator wants real high and fake low, while the generator wants its fakes scored high. This note writes the two wishes as one formula, explains how to chase it with gradients in turns, and why in practice the generator switches to a different loss.

先備知識是 KP01 的生成器與判別器、第2週的二元交叉熵與 sigmoid、第3週的梯度下降與反向傳播。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第14章 KP-2 與 KP-3 改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。圖與實驗的數字都由 `assets/figures/src/lec13-kp02-figures.py` 算出，其餘小算例可以用計算機驗算。

The prerequisites are KP01's generator and discriminator, the binary cross-entropy and sigmoid of Week 2, and gradient descent and backpropagation of Week 3. Every term and symbol is defined when it first appears. This note is rewritten from KP-2 and KP-3 of Chapter 14 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. The figures and experimental numbers are computed by `assets/figures/src/lec13-kp02-figures.py`; the other small calculations can be checked with a calculator.

## KP02 極小化極大賽局，把兩個相反的願望寫成一條式子（The minimax game: write two opposite wishes as one formula）

### 學習目標（Learning objectives）

讀完之後，你能寫出 GAN 的目標函數 $V(D,G)$，說出每一項是誰在乎、希望它變大還是變小。你能看出判別器的願望就是第2週的二元交叉熵，並用一小批評分算出 $V$。你能寫出交替更新的步驟：先對判別器做梯度上升，再對生成器做梯度下降。你也能用 sigmoid 的導數說明，為什麼訓練初期原本的生成器損失幾乎沒有梯度，以及改用 $-\log D(G(\mathbf{z}))$ 怎麼解決。

After reading, you can write the GAN objective $V(D,G)$ and say who cares about each term and whether they want it larger or smaller. You can see that the discriminator's wish is the binary cross-entropy of Week 2, and compute $V$ from a small batch of scores. You can write the alternating update: gradient ascent for the discriminator, then gradient descent for the generator. You can also explain, with the derivative of the sigmoid, why the original generator loss gives almost no gradient early in training, and how switching to $-\log D(G(\mathbf{z}))$ fixes it.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $p_{\text{data}},\ p_z$ | 真資料的分布、雜訊的分布 | distribution of the real data, distribution of the noise |
| $D(\mathbf{x})$ | 判別器認為 $\mathbf{x}$ 是真資料的機率 | the discriminator's probability that $\mathbf{x}$ is real |
| $G(\mathbf{z})$ | 生成器由雜訊 $\mathbf{z}$ 生成的資料 | the example the generator makes from noise $\mathbf{z}$ |
| $V(D,G)$ | 賽局的值（目標函數） | value of the game (objective function) |
| $\mathbb{E}_{\mathbf{x}\sim p}[\cdot]$ | 在分布 $p$ 下的期望值（平均） | expectation (average) under distribution $p$ |
| $h$ | 判別器在 sigmoid 之前的實數（logit），$D=\sigma(h)$ | the discriminator's real number before the sigmoid (logit), $D=\sigma(h)$ |
| $\theta_D,\ \theta_G$ | 判別器與生成器的權重 | weights of the discriminator and the generator |
| $\eta$ | 學習率 | learning rate |
| $m$ | 一批的樣本數 | number of samples in a batch |
| $k$ | 每更新生成器一次，先更新判別器幾次 | number of discriminator updates per generator update |

### 1. 判別器的願望就是二元交叉熵（The discriminator's wish is the binary cross-entropy）

判別器其實是一個二元分類器：真資料的標籤是 $1$，生成的資料標籤是 $0$。第2週學過，這種分類器用**二元交叉熵**（binary cross-entropy）訓練：標籤 $1$ 的樣本要讓 $\log D$ 大，標籤 $0$ 的樣本要讓 $\log(1-D)$ 大。把真的一半和假的一半加起來，判別器想讓下面這個量越大越好：

The discriminator is really a binary classifier: real data has label $1$ and generated data label $0$. As in Week 2, such a classifier is trained with the **binary cross-entropy**: samples with label $1$ want $\log D$ large, and samples with label $0$ want $\log(1-D)$ large. Adding the real half and the fake half, the discriminator wants the following quantity as large as possible:

$$
V(D,G)=\mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}\big[\log D(\mathbf{x})\big]+\mathbb{E}_{\mathbf{z}\sim p_z}\big[\log\big(1-D(G(\mathbf{z}))\big)\big].
$$

第一項看真資料：$D(\mathbf{x})$ 越接近 $1$，$\log D(\mathbf{x})$ 越接近 $0$。第二項看假資料：$D(G(\mathbf{z}))$ 越接近 $0$，$\log(1-D(G(\mathbf{z})))$ 越接近 $0$。兩項都不會超過 $0$，所以 $V$ 不會超過 $0$（sigmoid 的輸出碰不到 $0$ 或 $1$，只能越來越接近）。把真的一半與假的一半各算一次二元交叉熵再相加，正好是 $-V$；所以最大化 $V$ 就是最小化交叉熵。

The first term looks at real data: the closer $D(\mathbf{x})$ is to $1$, the closer $\log D(\mathbf{x})$ is to $0$. The second looks at fakes: the closer $D(G(\mathbf{z}))$ is to $0$, the closer $\log(1-D(G(\mathbf{z})))$ is to $0$. Neither term exceeds $0$, so $V$ never exceeds $0$ (a sigmoid output never reaches $0$ or $1$, it only gets closer). Computing the binary cross-entropy separately on the real half and on the fake half and adding the two gives exactly $-V$; so maximising $V$ means minimising the cross-entropy.

實際訓練時，期望值換成一批樣本的平均 `[補充：例子]`。假設一批有兩個真樣本，判別器給 $0.8$ 與 $0.6$；兩個假樣本，給 $0.3$ 與 $0.1$：

In practice the expectations are replaced by averages over a batch `[Added: example]`. Suppose a batch has two real samples scored $0.8$ and $0.6$, and two fakes scored $0.3$ and $0.1$:

$$
\tfrac12(\log0.8+\log0.6)=-0.367,\qquad \tfrac12(\log0.7+\log0.9)=-0.231,\qquad V=-0.367-0.231=-0.598 .
$$

這批的損失 $-V$ 就是 $0.598$，也就是真假兩半各自的二元交叉熵相加；若改成對全部 $4$ 個樣本取平均，每個樣本是 $0.299$。判別器更新後，若真的評分更高、假的更低，$V$ 就往 $0$ 靠近。

The loss $-V$ of this batch is $0.598$, the sum of the binary cross-entropies of the real half and the fake half; averaged over all $4$ samples instead, it is $0.299$ per sample. If an update makes the discriminator score real samples higher and fakes lower, $V$ moves towards $0$.

常見錯誤：以為 GAN 的判別器需要人工標註的標籤。標籤是自動來的：從資料集拿的標 $1$，生成器做的標 $0$，不需要任何人去標。

Common mistake: thinking a GAN's discriminator needs human-labelled data. The labels come for free: samples taken from the dataset are labelled $1$ and samples made by the generator are labelled $0$, with no human labelling at all.

### 2. 同一條式子，兩個人往相反方向拉（One formula, two players pulling in opposite directions）

第1節的 $V$ 也能描述生成器的願望。生成器只能改變第二項裡的 $G(\mathbf{z})$；它想讓判別器把假的判高，也就是讓 $D(G(\mathbf{z}))$ 接近 $1$，讓 $\log(1-D(G(\mathbf{z})))$ 變得很負。所以生成器要讓 $V$ 越小越好。來源把兩者合起來寫成**極小化極大**（minimax）問題：

The $V$ of Section 1 also describes the generator's wish. The generator can change only $G(\mathbf{z})$ in the second term; it wants the discriminator to score fakes high, making $D(G(\mathbf{z}))$ close to $1$ and $\log(1-D(G(\mathbf{z})))$ very negative. So the generator wants $V$ as small as possible. The source combines the two as a **minimax** problem:

$$
\min_G\ \max_D\ V(D,G).
$$

讀法是由內往外：對每一個生成器 $G$，判別器先盡力把 $V$ 推到最大；生成器再挑一個讓這個「最大值」最小的 $G$。這是兩個玩家的**零和賽局**（zero-sum game）：一方得到的就是另一方失去的。

Read it from the inside out: for each generator $G$, the discriminator first pushes $V$ as high as it can; then the generator picks the $G$ that makes this "maximum" smallest. This is a two-player **zero-sum game**: what one side gains, the other loses.

![一張方形的色階圖。橫軸是判別器對假樣本的評分 D(G(z))，從 0 到 1；縱軸是對真樣本的評分 D(x)，從 0 到 1；顏色代表 V，左上角最亮接近 0，右下角最暗。白色虛線是等高線。左上的點 (0.1, 0.9) 標 V=-0.21，中間的點 (0.5, 0.5) 標 V=-1.39。粉紅箭頭往左上，寫著 D: max V；黃色箭頭往右，寫著 G: min V（A square colour map. The horizontal axis is the discriminator's score on fakes D(G(z)), from 0 to 1; the vertical axis is its score on real samples D(x), from 0 to 1; colour shows V, brightest near 0 at the top left and darkest at the bottom right. Dashed white lines are contours. The point (0.1, 0.9) at the top left is labelled V=-0.21 and the middle point (0.5, 0.5) V=-1.39. A pink arrow points up-left, labelled D: max V; a yellow arrow points right, labelled G: min V）](../assets/figures/lec13-kp02-value-surface.png)

**圖說 `lec13-kp02-value-surface`**〔功能：一目了然、精準度〕：先看顏色：$V$ 只由兩個評分決定，左上角（真的判高、假的判低）最接近 $0$。再看兩支箭頭：判別器往左上推，生成器只能沿橫軸往右推，因為它碰不到真樣本的評分。最後看兩個點：左上的 $(0.1,\ 0.9)$ 是判別器佔上風，中間的 $(0.5,\ 0.5)$ 是 KP01 那種分不出真假的狀態。

**Caption `lec13-kp02-value-surface`** [Function: seeing it at a glance, precision]: first look at the colour: $V$ depends only on the two scores, and the top left (real high, fake low) is closest to $0$. Then look at the two arrows: the discriminator pushes towards the top left, while the generator can push only rightwards along the horizontal axis, because it cannot touch the score on real samples. Finally look at the two points: $(0.1,\ 0.9)$ at the top left is the discriminator winning, and $(0.5,\ 0.5)$ in the middle is KP01's state where real and fake cannot be told apart.

用來源互動展示的預設值算一次：真的評分 $0.9$、假的評分 $0.1$，$V=\log0.9+\log0.9=-0.211$。若兩者都是 $0.5$，$V=2\log0.5=-1.386$。若生成器騙過了判別器，假的也被評 $0.9$，$V=\log0.9+\log0.1=-2.408$，比前兩種都小，正是生成器要的方向。

Compute once with the default values of the source's interactive demo: real score $0.9$ and fake score $0.1$ give $V=\log0.9+\log0.9=-0.211$. If both are $0.5$, $V=2\log0.5=-1.386$. If the generator fools the discriminator so that fakes also get $0.9$, $V=\log0.9+\log0.1=-2.408$, smaller than both, which is exactly the generator's direction.

常見錯誤：以為生成器也要讓第一項 $\log D(\mathbf{x})$ 變小。第一項只和真資料與判別器有關，生成器的權重根本不出現在裡面，對它求梯度是 $0$；生成器只管第二項。

Common mistake: thinking the generator also wants the first term $\log D(\mathbf{x})$ to shrink. The first term involves only the real data and the discriminator; the generator's weights do not appear in it, so its gradient with respect to them is $0$; the generator cares only about the second term.

### 3. 交替更新：先上坡，再下坡（Alternating updates: climb, then descend）

$\min\max$ 沒辦法一次解出來。來源寫出實際的作法：**交替隨機梯度**（alternating stochastic gradient）。每一輪做兩個階段，期望值都用一批 $m$ 個樣本的平均代替：

$\min\max$ cannot be solved in one shot. The source writes out the practical method: **alternating stochastic gradient**. Each round has two phases, with every expectation replaced by an average over a batch of $m$ samples:

$$
\theta_D\leftarrow\theta_D+\eta\,\nabla_{\theta_D}\Big(\frac1m\sum_{i=1}^{m}\log D(\mathbf{x}^{(i)})+\frac1m\sum_{i=1}^{m}\log\big(1-D(G(\mathbf{z}^{(i)}))\big)\Big),
$$

$$
\theta_G\leftarrow\theta_G-\eta\,\nabla_{\theta_G}\Big(\frac1m\sum_{i=1}^{m}\log\big(1-D(G(\mathbf{z}^{(i)}))\big)\Big).
$$

一步一步寫出來：

Written out step by step:

1. 固定生成器，從資料集抽 $m$ 個真樣本，從 $p_z$ 抽 $m$ 個雜訊、做出 $m$ 個假樣本（Freeze the generator; draw $m$ real samples from the dataset and $m$ noise vectors from $p_z$, making $m$ fakes）
2. 對判別器的權重做一次梯度**上升**，讓 $V$ 變大；這一步常重複 $k$ 次，每次都抽新的一批（Take one gradient **ascent** step on the discriminator's weights to make $V$ larger; this is often repeated $k$ times, each time with a new batch）
3. 固定判別器，再抽 $m$ 個雜訊，對生成器的權重做一次梯度**下降**，讓第二項變小（Freeze the discriminator, draw $m$ new noise vectors, and take one gradient **descent** step on the generator's weights to make the second term smaller）
4. 回到第1步（Go back to step 1）

生成器的梯度要穿過判別器才拿得到：$\log(1-D(G(\mathbf{z})))$ 對 $\theta_G$ 求導，連鎖律會經過 $D$ 的每一層，再進入 $G$。第3步「固定判別器」的意思只是不改它的權重，梯度照樣穿過它。KP01 的小 GAN 也是這樣交替訓練，$k=10$；不過它的生成器用的是第4節的非飽和損失。

The generator gets its gradient only by passing through the discriminator: differentiating $\log(1-D(G(\mathbf{z})))$ with respect to $\theta_G$, the chain rule goes through every layer of $D$ and then into $G$. "Freezing the discriminator" in step 3 only means not changing its weights; the gradient still flows through it. KP01's small GAN alternated in the same way, with $k=10$, though its generator used the non-saturating loss of Section 4.

常見錯誤：以為交替更新可以省掉，把兩個網路接在一起，用同一個梯度方向一起更新。兩者的目標相反：判別器要 $V$ 變大，生成器要 $V$ 變小；同一個方向不可能同時滿足兩者，所以要分開、各自往自己的方向走。

Common mistake: thinking the alternation can be skipped, joining the two networks and updating them together in one gradient direction. Their goals are opposite: the discriminator wants $V$ larger and the generator wants $V$ smaller; no single direction satisfies both, so they must be updated separately, each in its own direction.

### 4. 訓練初期的梯度飽和，換一個損失（Saturating gradients early on, and a different loss）

第3節的生成器損失有一個實務上的毛病。訓練剛開始時，生成器做得很差，判別器很容易認出假樣本，$D(G(\mathbf{z}))$ 接近 $0$。來源指出，這時 $\log(1-D(G(\mathbf{z})))$ 的曲線很平，生成器幾乎拿不到梯度，叫做**梯度飽和**（saturating gradient）。「平」是對什麼而言？下面說明。

The generator loss of Section 3 has a practical flaw. Early in training the generator is poor, the discriminator spots fakes easily, and $D(G(\mathbf{z}))$ is close to $0$. The source points out that the curve of $\log(1-D(G(\mathbf{z})))$ is then flat, so the generator gets almost no gradient; this is called a **saturating gradient**. Flat with respect to what? The next paragraph explains.

精確地說，問題出在 sigmoid `[補充：推導]`。判別器的輸出是 $D=\sigma(h)$，而 $\sigma'(h)=\sigma(h)\big(1-\sigma(h)\big)$（第2週）。用連鎖律對 $h$ 求導：

Precisely, the trouble lies in the sigmoid `[Added: derivation]`. The discriminator outputs $D=\sigma(h)$, and $\sigma'(h)=\sigma(h)\big(1-\sigma(h)\big)$ (Week 2). Differentiating with respect to $h$ by the chain rule:

$$
\frac{d}{dh}\log\big(1-\sigma(h)\big)=-\frac{\sigma(h)(1-\sigma(h))}{1-\sigma(h)}=-D,\qquad \frac{d}{dh}\Big(-\log\sigma(h)\Big)=-\frac{\sigma(h)(1-\sigma(h))}{\sigma(h)}=-(1-D).
$$

生成器的梯度都要經過 $h$ 才能往回傳。用原本的損失，梯度大小是 $D$；判別器很有把握時 $D=0.01$，梯度只剩 $0.01$。來源建議的**非飽和損失**（non-saturating loss）改成讓生成器最大化 $\log D(G(\mathbf{z}))$，也就是最小化 $-\log D(G(\mathbf{z}))$；它的梯度大小是 $1-D=0.99$，正好在最需要的時候最大。

Every gradient for the generator must pass back through $h$. With the original loss its size is $D$; when the discriminator is confident, $D=0.01$ and the gradient is only $0.01$. The **non-saturating loss** recommended by the source has the generator maximise $\log D(G(\mathbf{z}))$, that is, minimise $-\log D(G(\mathbf{z}))$; its gradient has size $1-D=0.99$, largest exactly when it is needed most.

![左右兩張圖，橫軸都是 D(G(z))，從 0 到 1，左邊 0 到 0.1 塗淡黃色，標著訓練初期。左圖是生成器損失：紅線 log(1-D) 在左邊很平，接近 0，到右邊才急降；綠線 -log D 在左邊陡升到很高，往右慢慢降到 0。右圖是對 logit h 的梯度大小：紅線是 D，從 0 直線升到 1；綠線是 1-D，從 1 直線降到 0；在 D=0.01 處標出紅點 0.01 與綠點 0.99（Two panels, each with D(G(z)) from 0 to 1 on the horizontal axis and the region from 0 to 0.1 shaded pale yellow, labelled early training. Left, the generator loss: the red line log(1-D) is flat near 0 on the left and drops sharply only on the right; the green line -log D shoots up high on the left and falls slowly to 0 on the right. Right, the size of the gradient with respect to the logit h: the red line D rises straight from 0 to 1, the green line 1-D falls straight from 1 to 0, with a red point 0.01 and a green point 0.99 marked at D=0.01）](../assets/figures/lec13-kp02-saturation.png)

**圖說 `lec13-kp02-saturation`**〔功能：一目了然、精準度〕：先看左圖淡黃色的訓練初期：對 $D$ 而言，紅線變化慢（斜率約 $-1$），綠線的斜率 $-1/D$ 卻很大。再看右圖，這才是真正傳回生成器的梯度大小（對 logit）：在 $D(G(\mathbf{z}))=0.01$ 時，紅色只有 $0.01$，綠色有 $0.99$，差約 $100$ 倍。兩條線在 $0.5$ 交叉，後期兩者差不多。

**Caption `lec13-kp02-saturation`** [Function: seeing it at a glance, precision]: first look at the pale-yellow early region of the left panel: with respect to $D$, the red curve changes slowly (slope about $-1$) while the green one's slope $-1/D$ is huge. Then look at the right panel, which is the size of the gradient that actually reaches the generator (with respect to the logit): at $D(G(\mathbf{z}))=0.01$ red is only $0.01$ and green is $0.99$, about $100$ times larger. The two lines cross at $0.5$, so late in training they are similar.

用 KP01 的小 GAN 實際比一次 `[補充]`。同樣的真資料 $\mathcal{N}(2,\ 0.5^2)$、同樣的設定（每步判別器更新 $10$ 次、學習率 $0.1$ 與 $0.005$、每批 $256$、種子 $0$），只是讓生成器從很遠的 $a=-3$ 出發，分別用兩種損失訓練 $2000$ 步。

Compare the two in KP01's small GAN `[Added]`. Same real data $\mathcal{N}(2,\ 0.5^2)$ and same settings ($10$ discriminator updates per step, learning rates $0.1$ and $0.005$, batch $256$, seed $0$), but the generator starts far away at $a=-3$, and is trained for $2000$ steps with each loss.

![一張折線圖，橫軸是訓練步數 0 到 2000，縱軸是生成器的平均 a。藍色虛線在 2，標著真資料平均。綠線（非飽和）從 -3 很快上升，大約 450 步就到 2，之後在 2 附近小幅擺動。紅線（飽和）幾乎是平的，2000 步時才到約 -2.35（A line chart with training steps 0 to 2000 on the horizontal axis and the generator mean a on the vertical axis. A dashed blue line at 2 is labelled real mean. The green line (non-saturating) rises fast from -3, reaching 2 around step 450, then wobbles slightly around 2. The red line (saturating) is almost flat, reaching only about -2.35 at step 2000）](../assets/figures/lec13-kp02-far-start.png)

**圖說 `lec13-kp02-far-start`**〔功能：一目了然、比較〕：先看綠線：非飽和損失讓生成器的中心在幾百步內就從 $-3$ 走到 $2$。再看紅線：同樣的判別器、同樣的步數，飽和損失幾乎原地不動。藍色虛線是目標。

**Caption `lec13-kp02-far-start`** [Function: seeing it at a glance, comparison]: first look at the green line: with the non-saturating loss the generator's centre walks from $-3$ to $2$ within a few hundred steps. Then look at the red line: with the same discriminator settings and the same number of steps, the saturating loss barely moves. The dashed blue line is the target.

數字是這樣：非飽和損失在第 $500$ 步，生成器的中心已到 $2.053$。飽和損失在第 $500$ 步只到 $-2.948$，第 $2000$ 步也只有 $-2.354$；第 $100$ 到 $1000$ 步，判別器對假樣本的平均評分大多低於 $0.01$（例如第 $500$ 步是 $0.0022$），對 logit 的梯度也就只有這麼小；到後段它才升到約 $0.1$，生成器開始加快移動。

The numbers: with the non-saturating loss the generator's centre has reached $2.053$ by step $500$. With the saturating loss it is only at $-2.948$ at step $500$ and $-2.354$ at step $2000$; from step $100$ to $1000$ the discriminator's average score on fakes stays mostly below $0.01$ (for example $0.0022$ at step $500$), so the gradient with respect to the logit is just as small; only later does it climb to about $0.1$, and the generator starts to move faster.

換了損失，生成器就不再嚴格地在最小化 $V$；兩種損失的目標點一樣（都希望 $D(G(\mathbf{z}))$ 變大），只是爬坡的力道不同。KP03 會用最優判別器說明原本的 $V$ 在衡量什麼。

With the new loss the generator no longer strictly minimises $V$; both losses aim at the same thing (larger $D(G(\mathbf{z}))$), only with different pulling strength. KP03 uses the optimal discriminator to show what the original $V$ measures.

常見錯誤：以為非飽和損失「保證不會訓練失敗」。來源的小測驗也提醒，它只解決初期梯度太小的問題，不能保證不發生模式崩塌等其他問題（KP06）。

Common mistake: thinking the non-saturating loss "guarantees training will not fail". As the source's quiz also notes, it only fixes the too-small gradient early on; it does not prevent other problems such as mode collapse (KP06).

### 本單元重點整理（KP02 summary）

**一句話重點**：GAN 的目標是 $\min_G\max_D\ \mathbb{E}[\log D(\mathbf{x})]+\mathbb{E}[\log(1-D(G(\mathbf{z})))]$；判別器的部分就是二元交叉熵，生成器只影響第二項；訓練時交替做判別器的梯度上升與生成器的梯度下降；因為 sigmoid 的關係，初期原本的生成器損失梯度只有 $D$，所以實務上改用梯度為 $1-D$ 的 $-\log D(G(\mathbf{z}))$。

**One-sentence summary**: the GAN objective is $\min_G\max_D\ \mathbb{E}[\log D(\mathbf{x})]+\mathbb{E}[\log(1-D(G(\mathbf{z})))]$; the discriminator's part is the binary cross-entropy and the generator affects only the second term; training alternates gradient ascent on the discriminator with gradient descent on the generator; because of the sigmoid, the original generator loss has gradient only $D$ early on, so in practice it is replaced by $-\log D(G(\mathbf{z}))$, whose gradient is $1-D$.

**白話說明**：同一張計分表，鑑定師想讓分數高，仿畫的學徒想讓分數低；兩人輪流出手。學徒剛入門時畫得太差，鑑定師一眼看穿；原本的計分方式這時幾乎不告訴學徒該怎麼改，換一種計分方式，學徒才知道往哪裡努力。

**Plain explanation**: on one scoresheet, the authenticator wants the score high and the copying apprentice wants it low; they take turns. When the apprentice is a beginner and paints badly, the authenticator sees through it at once; the original scoring then tells the apprentice almost nothing about how to improve, and only a different scoring shows where to work.

**正式定義**：$V(D,G)=\mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}[\log D(\mathbf{x})]+\mathbb{E}_{\mathbf{z}\sim p_z}[\log(1-D(G(\mathbf{z})))]$，求 $\min_G\max_D V$；交替更新 $\theta_D\leftarrow\theta_D+\eta\nabla_{\theta_D}\hat V$（$\hat V$ 是第3節用一批樣本算的 $V$，重複 $k$ 次）與 $\theta_G\leftarrow\theta_G-\eta\nabla_{\theta_G}\frac1m\sum_i\log(1-D(G(\mathbf{z}^{(i)})))$，實務上改用 $\theta_G\leftarrow\theta_G-\eta\nabla_{\theta_G}\frac1m\sum_i\big(-\log D(G(\mathbf{z}^{(i)}))\big)$。

**Formal definition**: $V(D,G)=\mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}[\log D(\mathbf{x})]+\mathbb{E}_{\mathbf{z}\sim p_z}[\log(1-D(G(\mathbf{z})))]$, solve $\min_G\max_D V$; alternate $\theta_D\leftarrow\theta_D+\eta\nabla_{\theta_D}\hat V$ ($\hat V$ is the batch estimate of $V$ from Section 3; repeated $k$ times) with $\theta_G\leftarrow\theta_G-\eta\nabla_{\theta_G}\frac1m\sum_i\log(1-D(G(\mathbf{z}^{(i)})))$, replaced in practice by $\theta_G\leftarrow\theta_G-\eta\nabla_{\theta_G}\frac1m\sum_i\big(-\log D(G(\mathbf{z}^{(i)}))\big)$.

**例子**：評分 $0.9$ 與 $0.1$ 時 $V=-0.211$，都是 $0.5$ 時 $V=-1.386$；一批真 $0.8$、$0.6$ 與假 $0.3$、$0.1$ 的 $V=-0.598$；$D(G(\mathbf{z}))=0.01$ 時兩種生成器損失的梯度大小是 $0.01$ 對 $0.99$；小 GAN 從 $a=-3$ 出發，非飽和損失第 $500$ 步到 $2.053$，飽和損失第 $2000$ 步只到 $-2.354$。

**Example**: scores $0.9$ and $0.1$ give $V=-0.211$, and both $0.5$ give $V=-1.386$; a batch with real $0.8$, $0.6$ and fake $0.3$, $0.1$ gives $V=-0.598$; at $D(G(\mathbf{z}))=0.01$ the two generator losses have gradient sizes $0.01$ and $0.99$; the small GAN starting from $a=-3$ reaches $2.053$ by step $500$ with the non-saturating loss, but only $-2.354$ by step $2000$ with the saturating one.

**常見迷思**：

- 「判別器需要人工標籤。」錯，真的標 $1$、生成的標 $0$，自動就有。
- 「生成器也要讓 $\log D(\mathbf{x})$ 變小。」錯，那一項和生成器無關。
- 「兩個網路可以用同一個梯度方向一起更新。」錯，目標相反，要交替、各走各的方向。
- 「非飽和損失保證訓練成功。」錯，它只解決初期梯度太小。

**Common misconceptions**:

- "The discriminator needs human labels." Wrong: real samples are labelled $1$ and generated ones $0$, automatically.
- "The generator also wants $\log D(\mathbf{x})$ smaller." Wrong: that term does not involve the generator.
- "The two networks can be updated together in one gradient direction." Wrong: their goals are opposite, so they alternate, each in its own direction.
- "The non-saturating loss guarantees success." Wrong: it only fixes the too-small early gradient.

**圖**：本單元有三張圖。

- `lec13-kp02-value-surface`：$V$ 只由兩個評分決定，判別器往左上推、生成器往右推。
- `lec13-kp02-saturation`：兩種生成器損失與它們對 logit 的梯度大小。
- `lec13-kp02-far-start`：從遠處出發時，非飽和損失很快走到目標，飽和損失幾乎不動。

**Figures** — this unit has three figures.

- `lec13-kp02-value-surface`: $V$ depends only on the two scores; the discriminator pushes up-left and the generator pushes right.
- `lec13-kp02-saturation`: the two generator losses and the size of their gradients with respect to the logit.
- `lec13-kp02-far-start`: from a far start, the non-saturating loss reaches the target fast while the saturating loss barely moves.

### 小練習（Practice）

1. 一批有兩個真樣本，判別器評 $0.9$ 與 $0.7$；兩個假樣本，評 $0.2$ 與 $0.4$。算 $V$。
2. 判別器對某個假樣本的 logit 是 $h=-3$。分別算兩種生成器損失對 $h$ 的梯度大小，相差幾倍？
3. 某次訓練中，判別器對假樣本的平均評分一直停在 $0.001$ 附近，生成器幾乎沒有進步。你會先檢查什麼？

1. A batch has two real samples scored $0.9$ and $0.7$, and two fakes scored $0.2$ and $0.4$. Compute $V$.
2. The discriminator's logit for a fake sample is $h=-3$. Compute the size of the gradient of each generator loss with respect to $h$. How many times larger is one than the other?
3. In some training run, the discriminator's average score on fakes stays near $0.001$ and the generator barely improves. What would you check first?

**第 1 題解答** `[補充]`：$\tfrac12(\log0.9+\log0.7)=-0.231$，$\tfrac12(\log0.8+\log0.6)=-0.367$，$V=-0.598$。（和第1節的例子數字一樣，只是真假兩邊對調。）

**Solution to question 1** `[Added]`: $\tfrac12(\log0.9+\log0.7)=-0.231$, $\tfrac12(\log0.8+\log0.6)=-0.367$, so $V=-0.598$. (The same numbers as Section 1's example, with the real and fake sides swapped.)

**第 2 題解答** `[補充]`：$D=\sigma(-3)=1/(1+e^{3})\approx0.047$。飽和損失的梯度大小是 $D\approx0.047$，非飽和損失是 $1-D\approx0.953$，相差約 $20$ 倍。

**Solution to question 2** `[Added]`: $D=\sigma(-3)=1/(1+e^{3})\approx0.047$. The saturating loss has gradient size $D\approx0.047$ and the non-saturating loss $1-D\approx0.953$, about $20$ times larger.

**第 3 題解答** `[補充]`：先看生成器用的是哪一種損失；若是 $\log(1-D(G(\mathbf{z})))$，梯度大小只有約 $0.001$，應改成非飽和的 $-\log D(G(\mathbf{z}))$。也可以檢查判別器是否更新太多（$k$ 太大或學習率太高），讓它太早就完全看穿生成器。

**Solution to question 3** `[Added]`: first check which loss the generator uses; with $\log(1-D(G(\mathbf{z})))$ the gradient size is only about $0.001$, so switch to the non-saturating $-\log D(G(\mathbf{z}))$. You can also check whether the discriminator is updated too much ($k$ too large or learning rate too high), letting it see through the generator completely too early.

### 小測驗（Quiz，來源原題）

來源 KP-2 與 KP-3 各有一題自我檢測，保留原題與全部選項；來源頁面附有解析，下面依它說明，並補上更精確的理由。

The source's KP-2 and KP-3 each have one self-check question, kept here with all options; the source page gives explanations, which the note follows below with a more precise reason added.

1. 在 GAN 的 Minimax 目標函數 $\min_G\max_D V(D,G)$ 中，生成器 $G$ 的優化目標為何？
   - (A) 極大化 $\mathbb{E}[\log D(\mathbf{x})]+\mathbb{E}[\log(1-D(G(\mathbf{z})))]$
   - (B) 極小化 $\mathbb{E}[\log(1-D(G(\mathbf{z})))]$，即設法讓判別器對生成樣本輸出 $D(G(\mathbf{z}))$ 接近 $1$
   - (C) 將雜訊 $\mathbf{z}$ 直接變為 $\mathbf{0}$ 向量
   - (D) 極大化判別器的權重矩陣特徵值
2. GAN 在訓練初期生成器品質極差時，為何建議將生成器損失改為極大化 $\log D(G(\mathbf{z}))$？
   - (A) 因為原始的 $\log(1-D(G(\mathbf{z})))$ 在 $D(G(\mathbf{z}))$ 近於 $0$ 時梯度平緩飽和，改用 $\log D(G(\mathbf{z}))$ 能提供更大的初始梯度驅動力
   - (B) 因為 $\log D(G(\mathbf{z}))$ 可以省去計算二進位交叉熵
   - (C) 因為判別器在初期無法計算導數
   - (D) 因為改用 $\log D(G(\mathbf{z}))$ 能夠保證生成器永不崩塌

1. In the GAN minimax objective $\min_G\max_D V(D,G)$, what does the generator $G$ optimise?
   - (A) Maximise $\mathbb{E}[\log D(\mathbf{x})]+\mathbb{E}[\log(1-D(G(\mathbf{z})))]$
   - (B) Minimise $\mathbb{E}[\log(1-D(G(\mathbf{z})))]$, that is, try to make the discriminator output $D(G(\mathbf{z}))$ close to $1$ on generated samples
   - (C) Turn the noise $\mathbf{z}$ directly into the zero vector $\mathbf{0}$
   - (D) Maximise the eigenvalues of the discriminator's weight matrix
2. When the generator is very poor early in GAN training, why is it recommended to change the generator loss to maximising $\log D(G(\mathbf{z}))$?
   - (A) Because the original $\log(1-D(G(\mathbf{z})))$ has a flat, saturated gradient when $D(G(\mathbf{z}))$ is near $0$, while $\log D(G(\mathbf{z}))$ gives a larger initial gradient
   - (B) Because $\log D(G(\mathbf{z}))$ saves computing the binary cross-entropy
   - (C) Because the discriminator cannot compute derivatives early on
   - (D) Because switching to $\log D(G(\mathbf{z}))$ guarantees the generator never collapses

**解析** `[補充]`：第 1 題選 (B)：第2節，生成器碰不到第一項。第 2 題選 (A)：第4節。來源的解析說「$\log(1-x)$ 在 $x=0$ 附近的導數只有 $-1$，梯度過小」；其實 $-1$ 並不小，真正變小的是乘上 sigmoid 的導數之後，對 logit 的梯度只剩 $D$（見第4節）。(D) 錯，見第4節最後一段。

**Explanations** `[Added]`: question 1 is B: see Section 2; the generator cannot touch the first term. Question 2 is A: see Section 4. The source's explanation says "the derivative of $\log(1-x)$ near $x=0$ is only $-1$, so the gradient is too small"; in fact $-1$ is not small; what becomes small is the gradient with respect to the logit after multiplying by the sigmoid's derivative, which leaves only $D$ (see Section 4). D is wrong; see the last paragraph of Section 4.

### 留給下一單元的問題（The question for the next unit）

第2節說：對每一個生成器，判別器先把 $V$ 推到最大。這個「最大」長什麼樣子？如果判別器真的做到最好，它的評分 $D(\mathbf{x})$ 會等於什麼？這時剩下的 $V$ 又在量什麼？下一個知識點（本週 KP03）求出最優判別器，發現生成器其實在縮小一種分布之間的距離。

Section 2 said: for each generator, the discriminator first pushes $V$ to its maximum. What does that maximum look like? If the discriminator really does its best, what does its score $D(\mathbf{x})$ equal? And what does the remaining $V$ measure then? The next knowledge point (this week's KP03) finds the optimal discriminator and discovers that the generator is in fact shrinking a kind of distance between distributions.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第14章「產生式對抗網絡」KP-2「對抗式學習與極小化極大賽局」與 KP-3「損失函數、交叉熵與二階段梯度優化」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch14-gan/index.html ，讀取日期 2026-11-15）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from KP-2, "Adversarial learning and the minimax game", and KP-3, "Loss functions, cross-entropy and two-phase gradient optimisation", of Chapter 14, "Generative adversarial networks", of CCC-LNS IDL_115 (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch14-gan/index.html , accessed 2026-11-15). Content not in the source is marked `[Added]` where it appears.

- 來源的小測驗解析說 $\log(1-x)$ 在 $0$ 附近的導數「只有 $-1$，梯度過小」，而 $\log x$ 的導數「趨近無窮大」。對 $D$ 本身求導確實是 $-1$ 與很大的數，但 $-1$ 不算小；梯度真正傳回生成器時要乘上 sigmoid 的導數 $D(1-D)$，對 logit 的梯度才變成 $D$ 與 $1-D$。兩種看法的比例一樣（$D=0.01$ 時都是約 $99$ 倍），來源的相對比較沒錯；但只有對 logit 看，才看得出原本的梯度本身趨近 $0$。而且兩種損失後面的連鎖律（$h$ 對 $\mathbf{x}$、$\mathbf{x}$ 對 $\theta_G$）完全一樣，所以比較對 logit 的梯度是公平的。本講義第4節照這個推導寫，數字也用對 logit 的梯度。
- 來源的互動展示說飽和損失的梯度「趨近於 $-1$ 平緩飽和」；同上，那是對 $D$ 的導數，對 logit 的梯度趨近 $0$。
- 來源的「教科書圖 14.2」與 AI 生圖提示不重做；本講義改畫 $V$ 的色階圖。
- 來源用粗斜體 `\boldsymbol` 表示向量；本課程網站不支援，改用 `\mathbf`。
- 本講義補上：判別器的目標與二元交叉熵的關係、一批評分的算例、生成器梯度要穿過判別器的說明、$k$ 的意義、對 logit 求導的推導、小 GAN 從遠處出發的兩種損失比較、三題小練習與解答。
- 取捨：非飽和損失改變了賽局的嚴格形式，它和原本目標的關係只簡述，細節留給 KP03 與 KP08。

- The source's quiz explanation says the derivative of $\log(1-x)$ near $0$ is "only $-1$, so the gradient is too small", while that of $\log x$ "tends to infinity". Differentiating with respect to $D$ itself does give $-1$ and a very large number, but $-1$ is not small; when the gradient actually flows back to the generator it is multiplied by the sigmoid's derivative $D(1-D)$, so the gradients with respect to the logit become $D$ and $1-D$. The ratio is the same in both views (about $99$ times at $D=0.01$), so the source's relative comparison is right; but only the logit view shows that the original gradient itself goes to $0$. Moreover, the rest of the chain rule ($h$ with respect to $\mathbf{x}$, $\mathbf{x}$ with respect to $\theta_G$) is identical for both losses, so comparing gradients with respect to the logit is fair. Section 4 follows this derivation and uses gradients with respect to the logit in its numbers.
- The source's interactive demo says the saturating loss's gradient "flattens towards $-1$"; as above, that is the derivative with respect to $D$; the gradient with respect to the logit goes to $0$.
- The source's "textbook figure 14.2" and its AI image prompt are not reproduced; this note draws a colour map of $V$ instead.
- The source writes vectors with bold italic `\boldsymbol`; the course website does not support it, so this note uses `\mathbf`.
- This note adds: the link between the discriminator's objective and the binary cross-entropy, a batch calculation, the explanation that the generator's gradient passes through the discriminator, the meaning of $k$, the derivation with respect to the logit, the comparison of the two losses in the small GAN from a far start, and three practice questions with solutions.
- Choices: the non-saturating loss changes the strict form of the game; its relation to the original objective is only sketched, with details left to KP03 and KP08.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 二元交叉熵（第2週） | binary cross-entropy |
| 目標函數 | objective function |
| 賽局的值 | value of the game |
| 極小化極大 | minimax |
| 零和賽局 | zero-sum game |
| 交替隨機梯度 | alternating stochastic gradient |
| 梯度上升 | gradient ascent |
| 梯度下降 | gradient descent |
| 連鎖律 | chain rule |
| 梯度飽和 | saturating gradient |
| 非飽和損失 | non-saturating loss |
| logit（分對數） | logit |
| 模式崩塌（KP06） | mode collapse |
| 生成對抗網路 | generative adversarial network (GAN) |
