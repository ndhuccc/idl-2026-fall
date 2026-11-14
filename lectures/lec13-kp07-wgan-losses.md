# 第13週 KP07：換一把尺：WGAN 與其他損失，分布離很遠時也量得出「還差多遠」（Changing the ruler: WGAN and other losses, measuring "how far still to go" even when the distributions are far apart）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第13週的第七個知識點（KP07），接在 KP06 之後。KP03 說，判別器最優時生成器在縮小 JS 散度；可是兩個分布離很遠時，JS 散度停在 $\log2$ 附近幾乎不動，生成器拿不到方向。KP06 又看到，判別器的評分方式會讓生成器只守一個峰。這一節換一把尺：先介紹 Wasserstein 距離（推土機距離）與 WGAN，再簡介 LSGAN、EBGAN、D2GAN 這幾種換掉損失的做法。

This is the seventh knowledge point (KP07) of Week 13, following KP06. KP03 said that with an optimal discriminator the generator shrinks the JS divergence; but when the two distributions are far apart, the JS divergence sits near $\log2$ and barely moves, so the generator gets no direction. KP06 then saw that the way the discriminator scores can make the generator hold only one peak. This note changes the ruler: it first introduces the Wasserstein distance (the earth mover's distance) and WGAN, then briefly introduces LSGAN, EBGAN and D2GAN, which change the loss in other ways.

先備知識是 KP02 的非飽和損失、KP03 的最優判別器與 JS 散度，以及微積分的導數。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第14章的 KP-8 與 KP-13 改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec13-kp07-figures.py` 算出，可以自己重算（需要 PyTorch，約一分鐘）。

The prerequisites are KP02's non-saturating loss, KP03's optimal discriminator and JS divergence, and derivatives from calculus. Every term and symbol is defined when it first appears. This note is rewritten from KP-8 and KP-13 of Chapter 14 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec13-kp07-figures.py`, so you can recompute it yourself (it needs PyTorch and takes about a minute).

## KP07 換一把尺：WGAN 與其他損失（Changing the ruler: WGAN and other losses）

### 學習目標（Learning objectives）

讀完之後，你能用「搬土」的比喻說明 Wasserstein 距離，並在小例子裡算出它。你能比較 JS 散度與 Wasserstein 距離在兩個分布離很遠時的值與斜率，說明為什麼後者仍能給生成器方向。你能寫出 WGAN 的目標函數，說明評論家（critic）和判別器的差別，以及 1-Lipschitz 限制是什麼、為什麼需要、怎麼用權重剪裁或梯度懲罰近似地做到。你也能說出 LSGAN、EBGAN、D2GAN 各自改了什麼，並算出不同損失下生成器梯度的大小。

After reading, you can explain the Wasserstein distance with the "moving earth" picture and compute it in a small example. You can compare the values and slopes of the JS divergence and the Wasserstein distance when two distributions are far apart, and explain why the latter still gives the generator a direction. You can write the WGAN objective, explain how a critic differs from a discriminator, and say what the 1-Lipschitz constraint is, why it is needed, and how weight clipping or a gradient penalty enforces it approximately. You can also say what LSGAN, EBGAN and D2GAN each change, and compute the size of the generator's gradient under different losses.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $G,\ D,\ \mathbf{z},\ p_z$ | 生成器、判別器、雜訊與雜訊分布（同 KP02） | generator, discriminator, noise and its distribution (as in KP02) |
| $p_{\text{data}},\ p_g$ | 真資料與生成資料的分布 | the distributions of real and generated data |
| $\mathrm{JSD}$ | JS 散度（KP03） | JS divergence (KP03) |
| $W(p_{\text{data}},p_g)$ | Wasserstein 距離（推土機距離） | Wasserstein distance (earth mover's distance) |
| $\gamma$ | 搬運計畫：哪裡的土搬到哪裡、搬多少 | a transport plan: how much earth moves from where to where |
| $F_p(x)$ | 累積分布函數：小於等於 $x$ 的機率 | cumulative distribution function: the probability of being at most $x$ |
| $f(\mathbf{x})$ | 評論家：給每個樣本一個實數分數 | the critic: gives each sample a real-number score |
| $a$ | 生成分布 $\mathcal{N}(a,1)$ 的中心 | the centre of the generated distribution $\mathcal{N}(a,1)$ |
| $\lambda$ | 梯度懲罰的權重 | the weight of the gradient penalty |
| $a,\ b,\ c$（第5節） | LSGAN 的三個目標分數 | the three target scores of LSGAN |
| $s$ | 判別器對一個樣本的原始分數（還沒經過 sigmoid） | the discriminator's raw score for a sample (before any sigmoid) |

### 1. 為什麼要換一把尺（Why change the ruler）

沿用 KP03 第2節的例子：真資料 $p_{\text{data}}=\mathcal{N}(2,0.5^2)$，生成器畫的是 $\mathcal{N}(a,1)$，這次讓中心 $a$ 可以移動（$a=0$ 就是 KP03 算過的情形，$\mathrm{JSD}=0.4743$）。`[補充]` $a$ 從 $2$ 往左移到 $-3$ 時，JS 散度升到 $0.692$，貼在 $\log2\approx0.693$ 下面，這和 KP03 第4節的平台是同一種現象（那裡生成器的寬度是 $0.5$，所以數字略有不同）。再算它對 $a$ 的斜率，也就是生成器能感覺到的梯度：$a=0$ 時是 $-0.223$，$a=-1$ 時是 $-0.093$，$a=-3$ 時只剩 $-0.0031$。離得越遠，生成器越感覺不到該往哪走，可是這正是它最需要方向的時候。

Using the example of KP03 Section 2: the real data are $p_{\text{data}}=\mathcal{N}(2,0.5^2)$ and the generator draws $\mathcal{N}(a,1)$, this time with a movable centre $a$ ($a=0$ is the case KP03 computed, with $\mathrm{JSD}=0.4743$). `[Added]` As $a$ moves left from $2$ to $-3$, the JS divergence rises to $0.692$, just under $\log2\approx0.693$, the same kind of plateau as in KP03 Section 4 (there the generator's width was $0.5$, so the numbers differ slightly). We also compute its slope with respect to $a$, which is the gradient the generator can feel: $-0.223$ at $a=0$, $-0.093$ at $a=-1$, and only $-0.0031$ at $a=-3$. The farther away it is, the less the generator can feel where to go, and that is exactly when it needs a direction most.

我們想要一把尺，滿足兩件事：分布離得越遠，量出來的值就越大；而且不管離多遠，斜率都不會消失。

We want a ruler that does two things: the farther apart the distributions, the larger its value; and however far apart they are, its slope does not vanish.

### 2. 推土機距離（The earth mover's distance）

把每個分布想成一堆土：哪裡機率大，哪裡的土就高。把真資料那堆土，一鏟一鏟搬成生成資料那堆土的形狀，每搬一單位的土要付「搬的距離」這麼多力氣。最省力的搬法所付的總力氣，就是 Wasserstein 距離，也叫推土機距離（earth mover's distance，EMD）。來源寫成：

Think of each distribution as a pile of earth: where the probability is large, the pile is high. Reshape the real-data pile into the shape of the generated-data pile, spade by spade, paying "the distance moved" for each unit of earth. The total effort of the cheapest way to do it is the Wasserstein distance, also called the earth mover's distance (EMD). The source writes:

$$W(p_{\text{data}},p_g)=\inf_{\gamma\in\Pi(p_{\text{data}},p_g)}\ \mathbb{E}_{(\mathbf{x},\mathbf{y})\sim\gamma}\big[\lVert\mathbf{x}-\mathbf{y}\rVert\big]$$

$\gamma$ 是一個搬運計畫：$\gamma(\mathbf{x},\mathbf{y})$ 說有多少土從 $\mathbf{x}$ 搬到 $\mathbf{y}$。$\Pi(p_{\text{data}},p_g)$ 是所有「搬完之後形狀剛好對」的計畫。$\inf$ 是在這些計畫裡找最小（最省力）的那一個。

$\gamma$ is a transport plan: $\gamma(\mathbf{x},\mathbf{y})$ says how much earth moves from $\mathbf{x}$ to $\mathbf{y}$. $\Pi(p_{\text{data}},p_g)$ is the set of all plans that end with exactly the right shape. $\inf$ picks the smallest (cheapest) among these plans.

小例子 `[補充]`：土堆 $p$ 在 $0$ 和 $1$ 各有半單位，土堆 $q$ 在 $3$ 和 $4$ 各有半單位。最省力的搬法是 $0\to3$、$1\to4$，每半單位都搬 $3$，總力氣 $0.5\times3+0.5\times3=3$，所以 $W=3$。如果把 $q$ 再往右移 $1$，$W$ 就變成 $4$：距離跟著位移一起長大，不會停在某個上限。在一維時還有一個好用的公式 `[補充]`：$W=\int\lvert F_p(x)-F_q(x)\rvert\,dx$，也就是兩條累積分布曲線之間的面積。

A small example `[Added]`: pile $p$ has half a unit at $0$ and half at $1$, and pile $q$ has half a unit at $3$ and half at $4$. The cheapest plan is $0\to3$ and $1\to4$, moving each half-unit $3$, for a total of $0.5\times3+0.5\times3=3$, so $W=3$. Shift $q$ right by another $1$ and $W$ becomes $4$: the distance grows with the shift and does not stop at some ceiling. In one dimension there is also a handy formula `[Added]`: $W=\int\lvert F_p(x)-F_q(x)\rvert\,dx$, the area between the two cumulative distribution curves.

回到第1節的例子 `[補充]`：$a=-3$ 時 $W=5.0$，$a=-1$ 時 $3.0$，$a=0$ 時 $2.0$；離得遠時 $W\approx2-a$，斜率一直是 $-1$。$W$ 的最小值出現在 $a=2$，但只到 $0.399$ 而不是 $0$，因為生成器的寬度固定是 $1$、真資料是 $0.5$，形狀本來就不一樣。

Back to Section 1's example `[Added]`: $W=5.0$ at $a=-3$, $3.0$ at $a=-1$, and $2.0$ at $a=0$; when far apart $W\approx2-a$, with a slope of $-1$ throughout. The minimum of $W$ is at $a=2$, but it only gets down to $0.399$, not $0$, because the generator's width is fixed at $1$ while the real data's is $0.5$, so the shapes differ anyway.

![左右兩張圖，橫軸都是生成器中心 a，從 -4 到 6；藍色虛線在 a=2，標著 real centre。左圖縱軸是距離：綠線 W 是 V 字形，從 a=-4 的 6 直線降到 a=2 的約 0.4，再直線升到 a=6 的 4；紫線 JSD 很低，在 a=2 附近有一個淺谷，兩邊都貼在灰色點線 log 2 上。右圖縱軸是斜率：綠線 dW/da 在左邊一直是 -1，在 a=2 附近穿過 0 升到 +1；紫線 dJSD/da 只在中間有小小的起伏，兩邊都接近 0；左邊淡黃色區域寫著：最左邊 JSD 斜率接近 0，W 斜率仍是 -1（Two panels, both with the generator centre a from -4 to 6 horizontally; a dashed blue line at a=2 is labelled real centre. Left, distance on the vertical axis: the green W line is V-shaped, falling straight from 6 at a=-4 to about 0.4 at a=2 and rising straight to 4 at a=6; the purple JSD line is low, with a shallow valley near a=2 and lying on the grey dotted log 2 line on both sides. Right, slope on the vertical axis: the green dW/da line stays at -1 on the left, crossing 0 near a=2 and rising to +1; the purple dJSD/da line has only small bumps in the middle and is near 0 on both sides; a pale yellow region on the left reads: far left, JSD slope near 0, W slope still -1）](../assets/figures/lec13-kp07-jsd-vs-w.png)

**圖說 `lec13-kp07-jsd-vs-w`**〔功能：一目了然、精準度〕：先看左圖：綠色的 $W$ 離得越遠就越大，兩邊都是直線、中間一個谷；紫色的 JSD 一離開中間就貼平在 $\log2$。再看右圖的斜率，也就是生成器感覺到的力：淡黃色區域裡，紫線幾乎是 $0$，綠線仍然是 $-1$。生成器在最遠的地方，用 $W$ 仍知道該往右走。

**Caption `lec13-kp07-jsd-vs-w`** [Function: seeing it at a glance, precision]: first look at the left panel: the green $W$ grows the farther apart they are, straight on both sides with a valley in the middle; the purple JSD flattens onto $\log2$ as soon as it leaves the middle. Then look at the slopes on the right, the force the generator feels: in the pale yellow region the purple line is almost $0$ while the green one is still $-1$. Even at its farthest, a generator using $W$ still knows to move right.

常見錯誤：以為兩個常態分布「中心離很遠」時，JS 散度就恰好等於 $\log2$、梯度恰好是 $0$。常態分布的尾巴永遠有一點重疊，JS 散度只是**非常接近** $\log2$，斜率**非常接近** $0$（$a=-3$ 時 $-0.0031$）；實務上一樣學不動。

Common mistake: thinking that when two normal distributions have "centres far apart", the JS divergence equals $\log2$ exactly and the gradient is exactly $0$. Normal distributions' tails always overlap a little, so the JS divergence is only **very close** to $\log2$ and the slope **very close** to $0$ ($-0.0031$ at $a=-3$); in practice the generator cannot learn either way.

### 3. WGAN：用評論家量距離（WGAN: measuring the distance with a critic）

直接找最省力的搬法很難算。Arjovsky 等人（2017）的 Wasserstein GAN（WGAN）用了一個數學上的對偶定理（Kantorovich–Rubinstein 對偶），把它換成另一個問題：

Finding the cheapest plan directly is hard to compute. The Wasserstein GAN (WGAN) of Arjovsky et al. (2017) uses a duality theorem (Kantorovich–Rubinstein duality) to turn it into another problem:

$$W(p_{\text{data}},p_g)=\max_{f:\ \text{1-Lipschitz}}\ \mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}\big[f(\mathbf{x})\big]-\mathbb{E}_{\mathbf{x}\sim p_g}\big[f(\mathbf{x})\big]$$

$f$ 叫做評論家（critic）。它和 KP02 的判別器有兩個不同：輸出可以是任何實數，不是 $0$ 到 $1$ 的機率，所以最後不接 sigmoid；而且它必須是 **1-Lipschitz**，意思是對任兩點 $\lvert f(\mathbf{x})-f(\mathbf{y})\rvert\le\lVert\mathbf{x}-\mathbf{y}\rVert$。在一維就是「斜率的大小不超過 $1$」；可微分時等於 $\lVert\nabla_{\mathbf{x}}f(\mathbf{x})\rVert\le1$。評論家要讓真資料的平均分數和生成資料的平均分數差得越多越好；在斜率受限的條件下，它能拉開的最大差距就是 $W$。

$f$ is called the critic. It differs from KP02's discriminator in two ways: its output can be any real number, not a probability between $0$ and $1$, so there is no sigmoid at the end; and it must be **1-Lipschitz**, meaning $\lvert f(\mathbf{x})-f(\mathbf{y})\rvert\le\lVert\mathbf{x}-\mathbf{y}\rVert$ for any two points. In one dimension this is "the slope's size is at most $1$"; for a differentiable $f$ it is $\lVert\nabla_{\mathbf{x}}f(\mathbf{x})\rVert\le1$. The critic tries to make the average score of real data exceed that of generated data by as much as possible; with its slope limited, the largest gap it can open is exactly $W$.

為什麼一定要限制斜率？`[補充]` 若沒有限制，評論家把分數放大十倍，差距也放大十倍，最大值會是無窮大，什麼都量不出來。限制斜率之後，差距只能靠「真假之間隔得多遠」撐起來，所以量到的正是距離。

Why must the slope be limited? `[Added]` Without a limit, the critic could multiply its scores by ten and the gap would grow tenfold; the maximum would be infinite and measure nothing. With the slope limited, the gap can only come from "how far apart real and fake are", so what it measures is exactly the distance.

WGAN 的訓練就是生成器去縮小這個距離：

WGAN training has the generator shrink this distance:

$$\min_G\ \max_{f:\ \text{1-Lipschitz}}\ \mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}\big[f(\mathbf{x})\big]-\mathbb{E}_{\mathbf{z}\sim p_z}\big[f(G(\mathbf{z}))\big]$$

看離得很遠的情形 `[補充]`：$a=-3$，生成資料 $\mathcal{N}(-3,1)$ 整個在真資料的左邊。最好的評論家是 $f^*(x)=x$（加上任何常數也一樣好），斜率處處是 $+1$：越往右分數越高。代進去，$\mathbb{E}_{\text{data}}[x]-\mathbb{E}_{g}[x]=2-(-3)=5$，正好等於 $W=5.0$。生成器的損失是 $-\mathbb{E}[f(a+z)]=-a$，對 $a$ 的梯度是 $-1$，往梯度的反方向走就是把 $a$ 加大：生成器得到一個固定大小、指向右邊的推力。反觀 KP03 的最優判別器 $D^*$：在生成資料所在的地方，$D^*(-3)\approx4\times10^{-22}$、$D^*(-2)\approx4\times10^{-14}$，整片都貼在 $0$，平得沒有坡度。

Look at the far-apart case `[Added]`: $a=-3$, with the generated $\mathcal{N}(-3,1)$ entirely to the left of the real data. The best critic is $f^*(x)=x$ (adding any constant works equally well), with slope $+1$ everywhere: the farther right, the higher the score. Substituting, $\mathbb{E}_{\text{data}}[x]-\mathbb{E}_{g}[x]=2-(-3)=5$, exactly $W=5.0$. The generator's loss is $-\mathbb{E}[f(a+z)]=-a$, whose gradient with respect to $a$ is $-1$; stepping against the gradient increases $a$: the generator gets a push of fixed size pointing right. Compare KP03's optimal discriminator $D^*$: where the generated data are, $D^*(-3)\approx4\times10^{-22}$ and $D^*(-2)\approx4\times10^{-14}$, all lying on $0$, flat with no slope.

![一張圖，橫軸 x 從 -6.5 到 5.5。淡橘色是生成資料 N(-3,1)，淡藍色是真資料 N(2,0.5²)，都縮放到同樣高度。紫色實線是最優判別器 D*，在 x 小於 0 時都貼在 0，在 0 到 1 之間急升到 1。右邊另一條縱軸是評論家的值：綠色虛線是最優評論家 f*，一條斜率 1 的直線從左下到右上；紅色點線是用樣本、λ=100 訓練出的評論家，和綠線幾乎重合，只在 x 大於 4 之後往下彎。標題寫：離很遠（a=-3）時 D* 在假資料上平貼在 0，評論家仍保持斜率（One chart with x from -6.5 to 5.5. Pale orange is the generated N(-3,1) and pale blue the real N(2,0.5²), both scaled to the same height. The solid purple line is the optimal discriminator D*, lying on 0 for x below 0 and rising steeply to 1 between 0 and 1. A second vertical axis on the right shows critic values: the dashed green line is the optimal critic f*, a straight line of slope 1 from bottom left to top right; the dotted red line is a critic trained on samples with λ=100, almost on top of the green line, bending down only beyond x of about 4. The title reads: far apart (a=-3), D* is flat at 0 over the fakes; the critic keeps a slope）](../assets/figures/lec13-kp07-critic.png)

**圖說 `lec13-kp07-critic`**〔功能：一目了然、空間幾何具象化〕：先看兩堆分布：橘色的假資料在左、藍色的真資料在右，中間幾乎不重疊。再看紫線 $D^*$：在橘色那一整片都是 $0$，生成器碰它得不到坡度。最後看綠色虛線：評論家是一條斜坡，不管假資料在哪裡，都告訴它「往右分數更高」。紅色點線是用樣本訓練出來的評論家，和綠線幾乎一樣；它已上下平移，在 $x=0$ 與 $f^*$ 對齊，因為評論家加減一個常數，兩群平均的差不變。

**Caption `lec13-kp07-critic`** [Function: seeing it at a glance, making space concrete]: first look at the two distributions: orange fakes on the left and blue real data on the right, barely overlapping. Then look at the purple $D^*$: it is $0$ across the whole orange region, so the generator touching it gets no slope. Finally look at the dashed green line: the critic is a ramp that tells the fakes "the score is higher to the right" wherever they are. The dotted red line is a critic trained on samples, almost the same as the green line; it has been shifted up or down to match $f^*$ at $x=0$, because adding a constant to the critic leaves the difference of the two averages unchanged.

常見錯誤：以為評論家的輸出也是「是真的機率」。評論家的分數沒有上下限，單一個分數沒有意義；有意義的是真、假兩群平均分數的**差**，它估計的是距離。

Common mistake: thinking the critic's output is also "the probability of being real". The critic's scores have no bounds, and a single score means nothing; what means something is the **difference** between the average scores of the real and fake groups, which estimates the distance.

### 4. 怎麼讓評論家守規矩（How to keep the critic within the rule）

神經網路本身不會自動滿足 1-Lipschitz，要另外想辦法。來源提到兩種：

A neural network does not satisfy 1-Lipschitz by itself, so something extra is needed. The source mentions two methods:

- **權重剪裁**（weight clipping）：WGAN 原論文的做法。每次更新後，把評論家的每個權重都截在一個小範圍內，例如 $[-0.01,\ 0.01]$ `[補充]`。權重小，斜率就有上限。缺點是太粗糙：上限不一定剛好是 $1$，而且會讓評論家能表達的函數變得很簡單 `[補充]`。
- **梯度懲罰**（gradient penalty）：Gulrajani 等人（2017）的 WGAN-GP。在真、假樣本之間隨機取點 $\hat{\mathbf{x}}$，在損失裡加上 $\lambda\,\mathbb{E}\big[(\lVert\nabla f(\hat{\mathbf{x}})\rVert-1)^2\big]$，斜率離 $1$ 越遠罰越多；論文用 $\lambda=10$ `[補充]`。

- **Weight clipping**: the method of the original WGAN paper. After each update, every weight of the critic is clamped into a small range, such as $[-0.01,\ 0.01]$ `[Added]`. Small weights put a ceiling on the slope. The drawback is that it is crude: the ceiling is not necessarily exactly $1$, and it makes the functions the critic can express very simple `[Added]`.
- **Gradient penalty**: the WGAN-GP of Gulrajani et al. (2017). Random points $\hat{\mathbf{x}}$ are taken between real and fake samples, and $\lambda\,\mathbb{E}\big[(\lVert\nabla f(\hat{\mathbf{x}})\rVert-1)^2\big]$ is added to the loss, penalising slopes the farther they are from $1$; the paper uses $\lambda=10$ `[Added]`.

梯度懲罰是「罰」而不是「禁止」，所以只是近似 `[補充]`。用第3節的例子（$a=-3$，精確的 $W=5.0$），以一個小網路當評論家、各訓練 $3000$ 步：$\lambda=10$ 時估計值是 $6.20$，斜率最大到 $1.27$，因為斜率稍微超過 $1$ 只被罰一點點，評論家就把差距撐大一些；$\lambda=100$ 時估計值是 $5.16$，接近精確值。用一個簡單的估算可以驗證這個解釋：若評論家處處斜率都是 $k$，它要讓 $5k-\lambda(k-1)^2$ 最大，得 $k=1+5/(2\lambda)$，預測的估計值是 $6.25$（$\lambda=10$）與 $5.125$（$\lambda=100$），和學出來的 $6.20$、$5.16$ 很接近。$\lambda=100$ 只為這個示範；估計值偏大並不影響生成器拿到的方向，實務上仍常用 $\lambda=10$。這個評論家只在有樣本的地方可靠：$\lambda=100$ 的那一個，在 $x$ 大於約 $4$、幾乎沒有樣本的地方，斜率甚至變成負的（最小 $-0.66$），圖上紅色點線在右端往下彎就是這個原因。

The gradient penalty "penalises" rather than "forbids", so it is only approximate `[Added]`. Using Section 3's example ($a=-3$, exact $W=5.0$) with a small network as the critic, trained $3000$ steps each: with $\lambda=10$ the estimate is $6.20$ and slopes reach $1.27$, because a slope slightly above $1$ is penalised only a little, so the critic stretches the gap a bit; with $\lambda=100$ the estimate is $5.16$, close to the exact value. A simple estimate confirms this explanation: if the critic had slope $k$ everywhere, it would maximise $5k-\lambda(k-1)^2$, giving $k=1+5/(2\lambda)$ and predicted estimates of $6.25$ ($\lambda=10$) and $5.125$ ($\lambda=100$), close to the learned $6.20$ and $5.16$. $\lambda=100$ is only for this demonstration; an inflated estimate does not change the direction the generator receives, and $\lambda=10$ is still common in practice. The critic is reliable only where there are samples: the $\lambda=100$ one even gets a negative slope (as low as $-0.66$) where $x$ is above about $4$ and there are almost no samples, which is why the dotted red line bends down at the right end of the figure.

`[補充]` 實務上，WGAN 通常每更新一次生成器，就先更新評論家好幾次（常用 $5$ 次），讓評論家的估計夠準。評論家估計的距離通常會隨訓練下降，可以拿來看訓練進度；這是一般 GAN 的損失做不到的（KP04 說過它們會上下擺動）。許多實驗也發現 WGAN 比較少模式崩塌，但這不是保證。

`[Added]` In practice, WGAN usually updates the critic several times (often $5$) for each generator update, so the critic's estimate is accurate enough. The critic's estimated distance tends to fall as training proceeds and can be used to watch progress, which the losses of an ordinary GAN cannot do (KP04 said they swing up and down). Many experiments also find less mode collapse with WGAN, but this is not a guarantee.

常見錯誤：以為只要把判別器最後的 sigmoid 拿掉，就是 WGAN。少了斜率的限制，評論家的分數會無限放大，訓練會發散；WGAN 的關鍵是「無 sigmoid 的評論家」加上「1-Lipschitz 限制」兩者一起。

Common mistake: thinking that removing the discriminator's final sigmoid is all WGAN is. Without the slope constraint the critic's scores blow up and training diverges; the key of WGAN is "a critic without sigmoid" together with "the 1-Lipschitz constraint".

### 5. 其他換損失的做法（Other ways of changing the loss）

**最小平方 GAN（LSGAN）。** Mao 等人（2017）把交叉熵換成平方誤差。判別器輸出一個原始分數 $s=D(\mathbf{x})$（不經過 sigmoid），想讓真資料的分數接近 $b$、假資料接近 $a$；生成器想讓假資料的分數接近 $c$：

**Least-squares GAN (LSGAN).** Mao et al. (2017) replace the cross-entropy with a squared error. The discriminator outputs a raw score $s=D(\mathbf{x})$ (no sigmoid) and wants real data's scores near $b$ and fake data's near $a$; the generator wants the fakes' scores near $c$:

$$\min_D\ \tfrac12\,\mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}\big[(D(\mathbf{x})-b)^2\big]+\tfrac12\,\mathbb{E}_{\mathbf{z}\sim p_z}\big[(D(G(\mathbf{z}))-a)^2\big],\qquad\min_G\ \tfrac12\,\mathbb{E}_{\mathbf{z}\sim p_z}\big[(D(G(\mathbf{z}))-c)^2\big]$$

這裡的 $a,b,c$ 是 LSGAN 論文用的三個目標分數，和第1節的生成器中心 $a$ 無關。常用 $a=0$、$b=c=1$ `[補充]`。好處在生成器的梯度：損失 $\tfrac12(s-1)^2$ 對 $s$ 的梯度是 $s-1$，一個假樣本被判得越假（$s$ 越小），拉回它的力就越大。對照 KP02 的兩種交叉熵：原本的飽和損失 $\log(1-\sigma(s))$，梯度大小是 $\sigma(s)$，判得越假越接近 $0$；非飽和損失 $-\log\sigma(s)$，梯度大小是 $1-\sigma(s)$，最多到 $1$。例如 $s=-3$ 時，三者分別是 $0.047$、$0.953$、$4$。`[補充]` 選 $b-c=1$、$b-a=2$（例如 $a=-1,b=1,c=0$）時，論文證明 LSGAN 在縮小另一種散度（Pearson $\chi^2$ 散度）。

Here $a,b,c$ are the three target scores used in the LSGAN paper, unrelated to the generator centre $a$ of Section 1. $a=0$ and $b=c=1$ are common `[Added]`. The benefit is in the generator's gradient: the loss $\tfrac12(s-1)^2$ has gradient $s-1$ with respect to $s$, so the more clearly fake a sample is judged (the smaller $s$), the stronger the pull back. Compare KP02's two cross-entropies: the original saturating loss $\log(1-\sigma(s))$ has gradient size $\sigma(s)$, approaching $0$ as the sample looks more fake; the non-saturating loss $-\log\sigma(s)$ has size $1-\sigma(s)$, at most $1$. For example at $s=-3$ the three sizes are $0.047$, $0.953$ and $4$. `[Added]` With $b-c=1$ and $b-a=2$ (for example $a=-1,b=1,c=0$), the paper proves LSGAN shrinks another divergence (the Pearson $\chi^2$ divergence).

![一張折線圖，橫軸是判別器對一個假樣本的原始分數 s，從 -6 到 3，越左表示越明顯是假的；縱軸是生成器梯度的大小，從 0 到 7.4。紅色實線是飽和損失，大小 σ(s)，在左邊貼近 0，往右慢慢升到 1。橘色虛線是非飽和損失，大小 1-σ(s)，在左邊接近 1，往右降到 0。綠色實線是 LSGAN，大小 |s-1|，是一個 V 字，在 s=1 是 0，往左直線變大，在 s=-3 已經是 4、在 s=-6 是 7（A line chart with the discriminator's raw score s for a fake sample from -6 to 3 horizontally, further left meaning more clearly fake, and the size of the generator's gradient from 0 to 7.4 vertically. The solid red line is the saturating loss, size σ(s), near 0 on the left and rising slowly to 1 on the right. The dashed orange line is the non-saturating loss, size 1-σ(s), near 1 on the left and falling to 0 on the right. The solid green line is LSGAN, size |s-1|, a V shape that is 0 at s=1 and grows linearly to the left, already 4 at s=-3 and 7 at s=-6）](../assets/figures/lec13-kp07-score-gradient.png)

**圖說 `lec13-kp07-score-gradient`**〔功能：精準度、一目了然〕：先看左半邊，那裡的假樣本已經被判得很假。紅線（飽和損失）幾乎是 $0$：生成器拿不到力。橘色虛線（非飽和損失）大約是 $1$。綠線（LSGAN）越往左越大。不同損失的分數 $s$ 尺度不完全一樣，這張圖只比較「形狀」：判得越假時，梯度是消失、持平，還是變大。

**Caption `lec13-kp07-score-gradient`** [Function: precision, seeing it at a glance]: first look at the left half, where fakes are already judged clearly fake. The red line (saturating loss) is almost $0$: the generator gets no force. The dashed orange line (non-saturating loss) is about $1$. The green line (LSGAN) grows the farther left it goes. The score $s$ is not on exactly the same scale for different losses, so this figure compares only the "shape": as a sample is judged more fake, does the gradient vanish, stay level, or grow.

**以能量為基礎的 GAN（EBGAN）。** Zhao 等人（2017）把判別器做成一個自編碼器（第12週），用重建誤差 $E(\mathbf{x})=\lVert\text{Dec}(\text{Enc}(\mathbf{x}))-\mathbf{x}\rVert$ 當「能量」（實作常用平方誤差）：真資料要低能量（重建得好），假資料要高能量。`[補充]` 判別器的損失是 $E(\mathbf{x})+\max\big(0,\ m-E(G(\mathbf{z}))\big)$：真資料的能量越低越好，假資料的能量只要超過門檻 $m$ 就不再推；生成器則想讓 $E(G(\mathbf{z}))$ 越低越好。

**Energy-based GAN (EBGAN).** Zhao et al. (2017) make the discriminator an autoencoder (Week 12) and use the reconstruction error $E(\mathbf{x})=\lVert\text{Dec}(\text{Enc}(\mathbf{x}))-\mathbf{x}\rVert$ as the "energy" (implementations often use the squared error): real data should have low energy (reconstruct well) and fakes high energy. `[Added]` The discriminator's loss is $E(\mathbf{x})+\max\big(0,\ m-E(G(\mathbf{z}))\big)$: real data's energy as low as possible, and a fake's energy pushed up only until it passes a margin $m$; the generator wants $E(G(\mathbf{z}))$ as low as possible.

**雙判別器 GAN（D2GAN）。** Nguyen 等人（2017）用兩個判別器：$D_1$ 對真資料給高分，$D_2$ 對生成資料給高分。來源說它結合 KL 與反向 KL。`[補充]` 回想 KP06：正向 KL 逼生成器蓋住所有模式，反向 KL 讓它守住峰；兩者一起用，就是想在「蓋全」與「畫準」之間取得平衡，減少模式遺漏。

**Dual discriminator GAN (D2GAN).** Nguyen et al. (2017) use two discriminators: $D_1$ gives high scores to real data and $D_2$ to generated data. The source says it combines the KL and reverse KL divergences. `[Added]` Recall KP06: the forward KL forces the generator to cover every mode, while the reverse KL makes it hold peaks; using both aims for a balance between "covering everything" and "drawing accurately", reducing mode dropping.

常見錯誤：以為 LSGAN 的好處只是「平方比較大」。重點是梯度：損失是平方，梯度是一次方 $s-1$，所以拉回的力隨距離**線性**增加，而不是消失。

Common mistake: thinking LSGAN's benefit is just "squares are bigger". The point is the gradient: the loss is a square and the gradient is linear, $s-1$, so the pull back grows **linearly** with the distance instead of vanishing.

### 本單元重點整理（KP07 summary）

**一句話重點**：兩個分布離很遠時 JS 散度貼平在 $\log2$、斜率接近 $0$；Wasserstein 距離是「最省力的搬土力氣」，離得越遠越大、斜率不消失；WGAN 用 1-Lipschitz 的評論家估計它，評論家沒有 sigmoid、斜率受限，靠權重剪裁或梯度懲罰近似做到；LSGAN、EBGAN、D2GAN 用平方誤差、能量、雙判別器改掉原本的損失。

**One-sentence summary**: when two distributions are far apart, the JS divergence flattens onto $\log2$ with a slope near $0$; the Wasserstein distance is "the least effort to move the earth", growing the farther apart they are with a slope that does not vanish; WGAN estimates it with a 1-Lipschitz critic that has no sigmoid and a limited slope, enforced approximately by weight clipping or a gradient penalty; LSGAN, EBGAN and D2GAN replace the original loss with squared error, energy and two discriminators.

**白話說明**：舊的尺像一個只會說「像」或「不像」的評審：作品差得很遠時只會一直說「不像」，學徒不知道往哪改。新的尺像一個會說「還差五步」的教練：差得再遠也告訴你距離和方向。只是這個教練的打分要守規矩，不能隨便把分數放大，否則「五步」就沒有意義。

**Plain explanation**: the old ruler is like a judge who only says "alike" or "not alike": when the work is far off, it just keeps saying "not alike", and the apprentice does not know what to change. The new ruler is like a coach who says "five steps to go": however far off you are, it tells you the distance and the direction. But this coach must score by the rules and not inflate scores at will, or "five steps" means nothing.

**正式定義**（$\lVert f\rVert_L\le1$ 表示 $f$ 是 1-Lipschitz）：$W(p_{\text{data}},p_g)=\inf_{\gamma\in\Pi}\mathbb{E}_{\gamma}\lVert\mathbf{x}-\mathbf{y}\rVert=\max_{\lVert f\rVert_L\le1}\mathbb{E}_{p_{\text{data}}}[f]-\mathbb{E}_{p_g}[f]$；一維時 $W=\int\lvert F_p-F_q\rvert\,dx$；WGAN：$\min_G\max_{\lVert f\rVert_L\le1}\mathbb{E}[f(\mathbf{x})]-\mathbb{E}[f(G(\mathbf{z}))]$；梯度懲罰 $\lambda\,\mathbb{E}[(\lVert\nabla f(\hat{\mathbf{x}})\rVert-1)^2]$；LSGAN 生成器損失 $\tfrac12\mathbb{E}[(D(G(\mathbf{z}))-c)^2]$。

**Formal definition** ($\lVert f\rVert_L\le1$ means $f$ is 1-Lipschitz): $W(p_{\text{data}},p_g)=\inf_{\gamma\in\Pi}\mathbb{E}_{\gamma}\lVert\mathbf{x}-\mathbf{y}\rVert=\max_{\lVert f\rVert_L\le1}\mathbb{E}_{p_{\text{data}}}[f]-\mathbb{E}_{p_g}[f]$; in one dimension $W=\int\lvert F_p-F_q\rvert\,dx$; WGAN: $\min_G\max_{\lVert f\rVert_L\le1}\mathbb{E}[f(\mathbf{x})]-\mathbb{E}[f(G(\mathbf{z}))]$; gradient penalty $\lambda\,\mathbb{E}[(\lVert\nabla f(\hat{\mathbf{x}})\rVert-1)^2]$; LSGAN generator loss $\tfrac12\mathbb{E}[(D(G(\mathbf{z}))-c)^2]$.

**例子**：土堆 $\{0,1\}\to\{3,4\}$ 的 $W=3$；真資料 $\mathcal{N}(2,0.5^2)$、生成 $\mathcal{N}(-3,1)$ 時 $\mathrm{JSD}=0.692$、斜率 $-0.0031$，$W=5.0$、斜率 $-1$，最優評論家 $f^*(x)=x$；學出的評論家 $\lambda=10$ 估 $6.20$、$\lambda=100$ 估 $5.16$；$s=-3$ 時生成器梯度大小：飽和 $0.047$、非飽和 $0.953$、LSGAN $4$。

**Example**: piles $\{0,1\}\to\{3,4\}$ give $W=3$; with real $\mathcal{N}(2,0.5^2)$ and generated $\mathcal{N}(-3,1)$, $\mathrm{JSD}=0.692$ with slope $-0.0031$ and $W=5.0$ with slope $-1$, and the optimal critic is $f^*(x)=x$; learned critics estimate $6.20$ with $\lambda=10$ and $5.16$ with $\lambda=100$; at $s=-3$ the generator's gradient sizes are $0.047$ saturating, $0.953$ non-saturating, and $4$ for LSGAN.

**常見迷思**：

- 「中心離很遠時 JS 散度剛好是 $\log2$。」錯，常態分布永遠有一點重疊，只是非常接近。
- 「評論家輸出的是真的機率。」錯，它的分數沒有上下限，有意義的是兩群平均的差。
- 「拿掉 sigmoid 就是 WGAN。」錯，還需要 1-Lipschitz 限制。
- 「WGAN 保證不會模式崩塌。」錯，它通常比較穩定，但不保證。

**Common misconceptions**:

- "When the centres are far apart, the JS divergence is exactly $\log2$." Wrong: normal distributions always overlap a little; it is only very close.
- "The critic outputs the probability of being real." Wrong: its scores are unbounded, and what matters is the difference between the two groups' averages.
- "Removing the sigmoid makes it a WGAN." Wrong: the 1-Lipschitz constraint is also needed.
- "WGAN guarantees no mode collapse." Wrong: it is usually more stable, but there is no guarantee.

**圖**：本單元有三張圖。

- `lec13-kp07-jsd-vs-w`：生成器中心移動時，JS 散度與 Wasserstein 距離的值和斜率。
- `lec13-kp07-critic`：離很遠時，最優判別器平貼在 $0$，評論家保持斜率 $1$。
- `lec13-kp07-score-gradient`：三種生成器損失的梯度大小，隨假樣本的分數變化。

**Figures** — this unit has three figures.

- `lec13-kp07-jsd-vs-w`: the values and slopes of the JS divergence and the Wasserstein distance as the generator centre moves.
- `lec13-kp07-critic`: when far apart, the optimal discriminator lies flat at $0$ while the critic keeps slope $1$.
- `lec13-kp07-score-gradient`: the generator's gradient size under three losses, as a function of a fake sample's score.

### 小練習（Practice）

1. 土堆 $p$ 在 $0$ 和 $2$ 各有半單位，土堆 $q$ 在 $1$ 和 $5$ 各有半單位。有兩種一對一的搬法，各要多少力氣？$W$ 是多少？再用 $\int\lvert F_p-F_q\rvert\,dx$ 驗算。
2. 下列哪些函數是 1-Lipschitz：$f(x)=0.5x$、$f(x)=\sin x$、$f(x)=x^2$、$f(x)=3x$？若第3節的評論家改用 $f(x)=3x$，算出的「距離」會是多少？這說明了什麼？
3. 一個假樣本的原始分數 $s=-6$。分別用飽和損失、非飽和損失、LSGAN（$c=1$），生成器的梯度大小是多少？

1. Pile $p$ has half a unit at $0$ and half at $2$; pile $q$ has half a unit at $1$ and half at $5$. There are two one-to-one plans: how much effort does each take? What is $W$? Check it with $\int\lvert F_p-F_q\rvert\,dx$.
2. Which of these functions are 1-Lipschitz: $f(x)=0.5x$, $f(x)=\sin x$, $f(x)=x^2$, $f(x)=3x$? If Section 3's critic used $f(x)=3x$, what "distance" would it report? What does this show?
3. A fake sample has raw score $s=-6$. What is the size of the generator's gradient under the saturating loss, the non-saturating loss, and LSGAN ($c=1$)?

**第 1 題解答** `[補充]`：$0\to1$、$2\to5$：$0.5\times1+0.5\times3=2$；$0\to5$、$2\to1$：$0.5\times5+0.5\times1=3$。最省力是 $2$，所以 $W=2$。驗算：$x$ 在 $[0,1)$ 時 $F_p-F_q=0.5-0=0.5$，面積 $0.5$；在 $[1,2)$ 時 $0.5-0.5=0$；在 $[2,5)$ 時 $1-0.5=0.5$，面積 $1.5$；總共 $2$。

**Solution to question 1** `[Added]`: $0\to1$ and $2\to5$: $0.5\times1+0.5\times3=2$; $0\to5$ and $2\to1$: $0.5\times5+0.5\times1=3$. The cheaper is $2$, so $W=2$. Check: on $[0,1)$, $F_p-F_q=0.5-0=0.5$, area $0.5$; on $[1,2)$, $0.5-0.5=0$; on $[2,5)$, $1-0.5=0.5$, area $1.5$; total $2$.

**第 2 題解答** `[補充]`：$0.5x$ 的斜率是 $0.5$，是；$\sin x$ 的斜率 $\cos x$ 大小不超過 $1$，是；$x^2$ 的斜率 $2x$ 沒有上限，不是；$3x$ 的斜率是 $3$，不是。用 $f(x)=3x$：$3\times2-3\times(-3)=15$，是真正距離 $5$ 的三倍。不限制斜率，評論家就能任意放大差距，量出來的數字就不再是距離。

**Solution to question 2** `[Added]`: $0.5x$ has slope $0.5$, yes; $\sin x$ has slope $\cos x$ of size at most $1$, yes; $x^2$ has slope $2x$ with no bound, no; $3x$ has slope $3$, no. With $f(x)=3x$: $3\times2-3\times(-3)=15$, three times the true distance $5$. Without a slope limit the critic can inflate the gap at will, and the number it reports is no longer a distance.

**第 3 題解答** `[補充]`：飽和 $\sigma(-6)\approx0.0025$；非飽和 $1-\sigma(-6)\approx0.9975$；LSGAN $\lvert-6-1\rvert=7$。被判得很假的樣本，在飽和損失下幾乎拿不到力，LSGAN 反而拉得最用力。

**Solution to question 3** `[Added]`: saturating $\sigma(-6)\approx0.0025$; non-saturating $1-\sigma(-6)\approx0.9975$; LSGAN $\lvert-6-1\rvert=7$. A sample judged very fake gets almost no force under the saturating loss, while LSGAN pulls it hardest.

### 小測驗（Quiz，來源原題）

來源 KP-8 與 KP-13 各有一題自我檢測，保留原題與全部選項；來源頁面附有解析，下面依它說明。

The source's KP-8 and KP-13 each have one self-check question, kept here with all options; the source page gives explanations, which the note follows below.

1. Wasserstein GAN (WGAN) 相比於傳統原始 GAN，最大的理論突破為何？
   - (A) 放棄使用神經網絡，改用高斯混合模型
   - (B) 採用 Wasserstein 距離 (EMD) 取代 JS 散度，即使兩分佈無重疊也能提供平滑持續的梯度，從根本上解決訓練不穩定與模式崩塌
   - (C) 將判別器的輸出強制設為 $0$ 或 $1$ 的離散整數
   - (D) 不再需要潛在隨機雜訊 $\mathbf{z}$ 輸入

1. Compared with the original GAN, what is the biggest theoretical breakthrough of the Wasserstein GAN (WGAN)?
   - (A) Giving up neural networks for a Gaussian mixture model
   - (B) Replacing the JS divergence with the Wasserstein distance (EMD), which gives a smooth, lasting gradient even when the two distributions do not overlap, fundamentally solving training instability and mode collapse
   - (C) Forcing the discriminator's output to be the discrete integer $0$ or $1$
   - (D) No longer needing the latent random noise $\mathbf{z}$ as input

**解析**：選 (B)。來源的理由：兩個分布不重疊時，JS 散度是常數 $\log2$、梯度為零；Wasserstein 距離隨距離平滑變化，即使不重疊也能提供梯度（見第1、2節）。`[補充]` 選項裡「從根本上解決訓練不穩定與模式崩塌」說得太滿：WGAN 通常比較穩定，但評論家的限制只能近似做到（第4節），也不保證不崩塌。其他選項：(A)、(D) 都不對，WGAN 仍是神經網路、仍用雜訊 $\mathbf{z}$；(C) 正好相反，評論家的輸出是沒有上下限的實數。

**Explanation**: the answer is B. The source's reason: when the two distributions do not overlap, the JS divergence is the constant $\log2$ with zero gradient, while the Wasserstein distance changes smoothly with the distance and gives a gradient even without overlap (see Sections 1 and 2). `[Added]` The option's "fundamentally solving training instability and mode collapse" claims too much: WGAN is usually more stable, but the critic's constraint can only be enforced approximately (Section 4), and collapse is not ruled out. The other options: A and D are wrong, since WGAN is still a neural network and still uses the noise $\mathbf{z}$; C is the opposite of the truth, since the critic outputs an unbounded real number.

2. 最小二乘 GAN (LSGAN) 採用 MSE 損失取代傳統交叉熵損失的主要好處為何？
   - (A) 可以完全不需要 GPU 加速
   - (B) 對遠離決策邊界的偏離生成樣本給予二次方距離懲罰，避免了傳統 GAN 交叉熵的梯度飽和問題，使訓練更加穩定
   - (C) 使生成器參數縮減 $50\%$
   - (D) 將生成圖像直接轉換為向量圖

2. What is the main benefit of the least-squares GAN (LSGAN) replacing the traditional cross-entropy loss with an MSE loss?
   - (A) It no longer needs GPU acceleration at all
   - (B) It gives generated samples that stray far from the decision boundary a squared-distance penalty, avoiding the gradient saturation of the traditional GAN's cross-entropy and making training more stable
   - (C) It cuts the generator's parameters by $50\%$
   - (D) It turns generated images directly into vector graphics

**解析**：選 (B)。來源的理由：傳統 GAN 的交叉熵在判別成功後梯度趨近 $0$，LSGAN 的平方損失則依離邊界的距離給出成比例的拉回梯度（見第5節）。`[補充]` 「梯度為 $0$」指的是 KP02 原本的飽和損失；非飽和損失已經避開這個問題，只是梯度最多到 $1$。另外，「二次方」是損失，梯度是一次方。其他選項：(A)、(C)、(D) 都和損失函數無關。

**Explanation**: the answer is B. The source's reason: the traditional GAN's cross-entropy has a gradient approaching $0$ once the discrimination succeeds, while LSGAN's squared loss gives a pull-back gradient proportional to the distance from the boundary (see Section 5). `[Added]` "Gradient of $0$" refers to KP02's original saturating loss; the non-saturating loss already avoids this, though its gradient is at most $1$. Also, the loss is quadratic, while its gradient is linear. The other options: A, C and D have nothing to do with the loss function.

### 留給下一單元的問題（The question for the next unit）

這一週看到了好幾把尺：原始 GAN 對應 JS 散度，WGAN 對應 Wasserstein 距離，LSGAN（取 $b-c=1$、$b-a=2$ 時）對應 Pearson $\chi^2$ 散度，KP06 又談到正向與反向 KL。它們之間有沒有一個共同的框架？第12週的 VAE 用最大概似訓練，GAN 卻完全不寫出 $p_g$ 的公式，兩者差在哪裡？下一個知識點（本週 KP08，選讀）談 GAN 與最大概似、$f$-GAN，以及貝氏的觀點。

This week has met several rulers: the original GAN goes with the JS divergence, WGAN with the Wasserstein distance, LSGAN (with $b-c=1$ and $b-a=2$) with the Pearson $\chi^2$ divergence, and KP06 discussed the forward and reverse KL. Is there one framework that covers them? Week 12's VAE trains by maximum likelihood, while a GAN never writes down a formula for $p_g$: what is the difference? The next knowledge point (this week's KP08, optional) discusses GANs and maximum likelihood, $f$-GAN, and the Bayesian view.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第14章「產生式對抗網絡」的 KP-8「雙判別器 D2GAN 與 Wasserstein GAN（WGAN／Earth Mover's Distance）」與 KP-13「最小二乘 LSGAN 與基於能量 EBGAN」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch14-gan/index.html ，讀取日期 2026-11-15）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from Chapter 14, "Generative adversarial networks", of CCC-LNS IDL_115: KP-8, "The dual discriminator D2GAN and the Wasserstein GAN (WGAN / earth mover's distance)", and KP-13, "The least-squares LSGAN and the energy-based EBGAN" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch14-gan/index.html , accessed 2026-11-15). Content not in the source is marked `[Added]` where it appears.

- 來源的互動展示說兩個常態分布中心距離大於 $2$「無重疊」時，JS 散度「恆定為 $\log2$（梯度歸零）」。常態分布永遠有一點重疊，JS 散度只是非常接近 $\log2$、梯度非常接近 $0$；本講義第1、2節用數字說明（$a=-3$ 時 $0.692$、斜率 $-0.0031$）。 若兩個分布真的完全不重疊（例如第2節的土堆 $\{0,1\}$ 與 $\{3,4\}$），JS 散度確實恰好是 $\log2$，而 $W=3$；來源的說法在這種情形是對的。
- 來源 KP-8 的自我檢測選項 (B) 說 WGAN「從根本上解決訓練不穩定與模式崩塌」；本講義保留原題，在解析中說明這說得太滿，並在第4節用學出的評論家說明限制只能近似做到。
- 來源 KP-13 的 LSGAN 公式排版不完整（`align` 環境沒有正確開頭）；本講義重寫成兩條式子，並補上常用的 $a,b,c$ 值。
- 來源說傳統交叉熵在判別成功後「不會再產生梯度」。這是 KP02 的飽和損失；非飽和損失已經避開，本講義第5節把三種損失放在一起比較。來源互動展示說懲罰「呈二次方暴增」，本講義補充：損失是二次方，梯度是一次方。
- 來源已說判別器是自編碼器，但把重建結果也寫成 $D(\mathbf{x})$（能量 $\lVert D(\mathbf{x})-\mathbf{x}\rVert^2$），和他處 $D(\mathbf{x})$ 是一個分數的用法衝突；本講義把重建寫成 $\text{Dec}(\text{Enc}(\mathbf{x}))$，並補上判別器的門檻損失。
- 來源用粗斜體 `\boldsymbol` 表示向量；本講義改用 `\mathbf`。
- 本講義補上：搬土的小例子與一維的累積分布公式、JS 散度與 $W$ 的斜率比較、$W$ 最小值不是 $0$ 的原因、評論家與判別器的差別、為什麼要限制斜率、最優評論家 $f^*(x)=x$ 與 $D^*$ 的數值對照、權重剪裁的範圍與缺點、梯度懲罰的公式與兩種 $\lambda$ 的實驗、評論家更新次數、LSGAN 的梯度比較與 $\chi^2$ 散度、EBGAN 的門檻損失、D2GAN 與 KP06 的關聯、三題小練習與解答。

- The source's interactive demo says that when two normal distributions' centres are more than $2$ apart they "do not overlap" and the JS divergence is "constant at $\log2$ (zero gradient)". Normal distributions always overlap a little, so the JS divergence is only very close to $\log2$ and the gradient very close to $0$; Sections 1 and 2 show this with numbers ($0.692$ with slope $-0.0031$ at $a=-3$). If two distributions truly do not overlap at all (such as the earth piles $\{0,1\}$ and $\{3,4\}$ of Section 2), the JS divergence is indeed exactly $\log2$, while $W=3$; the source's statement is right in that case.
- Option B of the source's KP-8 self-check says WGAN "fundamentally solves training instability and mode collapse"; this note keeps the question as is, explains in the explanation that this claims too much, and uses learned critics in Section 4 to show the constraint is enforced only approximately.
- The source's KP-13 LSGAN formula is incompletely typeset (the `align` environment is not opened correctly); this note rewrites it as two formulas and adds the common values of $a,b,c$.
- The source says the traditional cross-entropy "produces no more gradient" after successful discrimination. That is KP02's saturating loss; the non-saturating loss already avoids it, and Section 5 compares the three losses together. The source's demo says the penalty "grows quadratically"; this note adds that the loss is quadratic while the gradient is linear.
- The source does say the discriminator is an autoencoder, but it also writes the reconstruction as $D(\mathbf{x})$ (energy $\lVert D(\mathbf{x})-\mathbf{x}\rVert^2$), clashing with $D(\mathbf{x})$ as a score elsewhere; this note writes the reconstruction as $\text{Dec}(\text{Enc}(\mathbf{x}))$ and adds the discriminator's margin loss.
- The source writes vectors with bold italic `\boldsymbol`; this note uses `\mathbf`.
- This note adds: the small earth-moving example and the one-dimensional cumulative-distribution formula, the comparison of the slopes of the JS divergence and $W$, why the minimum of $W$ is not $0$, how a critic differs from a discriminator, why the slope must be limited, the numerical comparison of the optimal critic $f^*(x)=x$ with $D^*$, the range and drawback of weight clipping, the gradient-penalty formula with experiments at two values of $\lambda$, the number of critic updates, LSGAN's gradient comparison and the $\chi^2$ divergence, EBGAN's margin loss, the link between D2GAN and KP06, and three practice questions with solutions.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| Wasserstein 距離 | Wasserstein distance |
| 推土機距離 | earth mover's distance (EMD) |
| 搬運計畫 | transport plan |
| 累積分布函數 | cumulative distribution function (CDF) |
| 對偶 | duality |
| 評論家 | critic |
| 1-Lipschitz 限制 | 1-Lipschitz constraint |
| 權重剪裁 | weight clipping |
| 梯度懲罰 | gradient penalty (GP) |
| 最小平方 GAN | least-squares GAN (LSGAN) |
| 原始分數 | raw score |
| Pearson $\chi^2$ 散度 | Pearson $\chi^2$ divergence |
| 以能量為基礎的 GAN | energy-based GAN (EBGAN) |
| 重建誤差 | reconstruction error |
| 門檻 | margin |
| 雙判別器 GAN | dual discriminator GAN (D2GAN) |
| 飽和損失、非飽和損失（KP02） | saturating loss, non-saturating loss |
| JS 散度（KP03） | Jensen–Shannon divergence (JSD) |
