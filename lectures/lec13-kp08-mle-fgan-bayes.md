# 第13週 KP08（選讀）：GAN 與最大概似、$f$-GAN、貝氏 GAN，從更大的框架看 GAN（Optional: GANs and maximum likelihood, $f$-GAN and Bayesian GANs, seeing the GAN in a larger framework）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第13週的第八個知識點（KP08），屬於**選讀**，接在 KP07 之後。這一週已經看過好幾把尺：原始 GAN 對應 JS 散度，WGAN 對應 Wasserstein 距離，LSGAN 對應 Pearson $\chi^2$ 散度，KP06 又比較了正向與反向 KL。這一節退後一步回答三個問題：GAN 和第12週 VAE 所近似的最大概似（它最大化的是對數概似的下界 ELBO）差在哪裡？這些散度有沒有一個共同的框架？把貝氏的想法放進 GAN 會怎樣？

This is the eighth knowledge point (KP08) of Week 13, an **optional** unit following KP07. This week has met several rulers: the original GAN goes with the JS divergence, WGAN with the Wasserstein distance, LSGAN with the Pearson $\chi^2$ divergence, and KP06 compared the forward and reverse KL. This note steps back to answer three questions: how does a GAN differ from the maximum likelihood that Week 12's VAE approximates (via the ELBO, a lower bound on the log-likelihood)? Is there a common framework for these divergences? What happens when Bayesian ideas are put into a GAN?

先備知識是 KP03 的 JS 散度與密度比、KP06 的正向與反向 KL、KP07 的 Wasserstein 距離，以及第12週的 VAE 與 KL 散度。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第14章的 KP-9、KP-10、KP-11 改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec13-kp08-figures.py` 算出，可以自己重算。

The prerequisites are KP03's JS divergence and density ratio, KP06's forward and reverse KL, KP07's Wasserstein distance, and Week 12's VAE and KL divergence. Every term and symbol is defined when it first appears. This note is rewritten from KP-9, KP-10 and KP-11 of Chapter 14 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec13-kp08-figures.py`, so you can recompute it yourself.

## KP08 GAN 與最大概似、$f$-GAN、貝氏 GAN（GANs and maximum likelihood, $f$-GAN and Bayesian GANs）

### 學習目標（Learning objectives）

讀完之後，你能說明顯式模型與隱式模型的差別，推導「最大概似等於極小化正向 KL」，並說明為什麼生成器的樣本落在低維曲面上時，最大概似無法使用。你能寫出 $f$-散度的定義，用同一個公式算出 KL、反向 KL、Pearson $\chi^2$、總變異與 JS 散度。你能算出一個凸函數的 Fenchel 共軛，說明 $f$-GAN 的下界為什麼成立、什麼時候等號成立。最後，你能用一兩句話說出 AVB 與貝氏 GAN 各自想解決什麼問題。

After reading, you can explain the difference between explicit and implicit models, derive "maximum likelihood equals minimising the forward KL", and explain why maximum likelihood cannot be used when the generator's samples lie on a low-dimensional surface. You can write the definition of an $f$-divergence and use one formula to compute the KL, reverse KL, Pearson $\chi^2$, total variation and JS divergences. You can compute the Fenchel conjugate of a convex function and explain why the $f$-GAN lower bound holds and when it is tight. Finally, you can say in a sentence or two what problem AVB and the Bayesian GAN each try to solve.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $G,\ D,\ \mathbf{z},\ p_z$ | 生成器、判別器、雜訊與雜訊分布（同 KP02） | generator, discriminator, noise and its distribution (as in KP02) |
| $p_\theta(\mathbf{x})$ | 參數為 $\theta$ 的模型寫得出的機率密度 | the probability density written down by a model with parameters $\theta$ |
| $p_{\text{data}},\ p_g$ | 真資料與生成資料的分布 | the distributions of real and generated data |
| $D_{\text{KL}}(P\,\Vert\,Q)$ | KL 散度（第12週 KP02） | KL divergence (Week 12 KP02) |
| $u$ | 密度比 $p(\mathbf{x})/q(\mathbf{x})$ | the density ratio $p(\mathbf{x})/q(\mathbf{x})$ |
| $f(u)$ | 定義 $f$-散度用的凸函數，$f(1)=0$ | the convex function defining an $f$-divergence, with $f(1)=0$ |
| $D_f(P\,\Vert\,Q)$ | $f$-散度 | $f$-divergence |
| $f^*(t)$ | $f$ 的 Fenchel 共軛 | the Fenchel conjugate of $f$ |
| $T(\mathbf{x})$ | $f$-GAN 裡的測試函數，由判別器網路扮演 | the test function in $f$-GAN, played by the discriminator network |
| $\eta$ | SGLD 的步長 | the step size of SGLD |

### 1. 顯式模型與隱式模型（Explicit and implicit models）

來源把生成模型分成兩大類。**顯式模型**（explicit model）寫得出每一點的機率密度 $p_\theta(\mathbf{x})$，例如高斯混合、自迴歸模型、流模型；它們用**最大概似估計**（maximum likelihood estimation，MLE）訓練：讓訓練資料的對數機率總和 $\sum_i\log p_\theta(\mathbf{x}_i)$ 越大越好。**隱式模型**（implicit model）不寫出密度，只提供一個抽樣的方法：抽雜訊 $\mathbf{z}$、算 $G(\mathbf{z})$。GAN 就是隱式模型：我們可以很快抽出樣本，卻算不出「這張圖的機率是多少」。

The source divides generative models into two families. An **explicit model** writes down the probability density $p_\theta(\mathbf{x})$ at every point, such as Gaussian mixtures, autoregressive models and flow models; they are trained by **maximum likelihood estimation** (MLE), making the sum of the training data's log probabilities $\sum_i\log p_\theta(\mathbf{x}_i)$ as large as possible. An **implicit model** writes down no density and only provides a way to sample: draw noise $\mathbf{z}$ and compute $G(\mathbf{z})$. A GAN is an implicit model: we can draw samples quickly but cannot compute "what is the probability of this image".

最大概似和 KL 散度的關係 `[補充]`：資料很多時，平均對數概似 $\frac1N\sum_i\log p_\theta(\mathbf{x}_i)$ 接近 $\mathbb{E}_{p_{\text{data}}}[\log p_\theta(\mathbf{x})]$。而

The link between maximum likelihood and the KL divergence `[Added]`: with plenty of data, the average log-likelihood $\frac1N\sum_i\log p_\theta(\mathbf{x}_i)$ is close to $\mathbb{E}_{p_{\text{data}}}[\log p_\theta(\mathbf{x})]$. And

$$D_{\text{KL}}(p_{\text{data}}\,\Vert\,p_\theta)=\mathbb{E}_{p_{\text{data}}}\big[\log p_{\text{data}}(\mathbf{x})\big]-\mathbb{E}_{p_{\text{data}}}\big[\log p_\theta(\mathbf{x})\big].$$

第一項和 $\theta$ 無關，所以把第二項（平均對數概似）變大，就等於把正向 KL 變小。KP06 說過，正向 KL 是「涵蓋模式」的：真資料存在的地方，模型都必須給一些機率。KP06 的雙峰例子裡，用一個常態分布做最大概似，得到的正是蓋住兩峰的 $\mathcal{N}(0,\ 2.06^2)$。常見的說法是，這就是為什麼最大概似訓練的影像模型容易產生模糊、「平均」的樣本；這是一個直觀的解釋，不是嚴格的定理。

The first term does not depend on $\theta$, so making the second term (the average log-likelihood) larger is the same as making the forward KL smaller. KP06 said the forward KL is "mode-covering": wherever real data exist, the model must give some probability. In KP06's two-peak example, maximum likelihood with one normal distribution gives exactly the wide $\mathcal{N}(0,\ 2.06^2)$ covering both peaks. A common account is that this is why image models trained by maximum likelihood tend to produce blurry, "averaged" samples; it is an intuitive explanation, not a strict theorem.

為什麼 GAN 不乾脆也用最大概似？`[補充]` 除了算不出密度以外，還有一個更根本的原因：生成器的雜訊通常比資料的維度低很多（例如 $100$ 維的 $\mathbf{z}$ 生成 $64\times64\times3$ 的影像），所以所有生成樣本都落在高維空間裡一個「薄薄的曲面」上，在曲面外的機率密度是 $0$。看一個最小的例子：雜訊 $z$ 只有 $1$ 維，生成器把它映到平面上的一條曲線；真資料是平面上的一團點。$600$ 個真資料點裡，只有 $1.67\%$ 離曲線在 $0.01$ 以內，離曲線的距離中位數是 $0.379$。即使這 $1.67\%$ 也不在曲線上：真資料點剛好落在曲線上的機率是 $0$。對每一個真資料點，生成器給的密度都是 $0$，對數概似是 $-\infty$，正向 KL 是無窮大，最大概似根本無從下手。

Why does a GAN not just use maximum likelihood too? `[Added]` Besides not being able to compute the density, there is a more fundamental reason: the generator's noise usually has far fewer dimensions than the data (for example a $100$-dimensional $\mathbf{z}$ generating a $64\times64\times3$ image), so all generated samples lie on a "thin surface" in the high-dimensional space, with probability density $0$ off the surface. Take the smallest example: the noise $z$ is $1$-dimensional and the generator maps it to a curve in the plane, while the real data are a cloud of points in the plane. Of $600$ real points, only $1.67\%$ are within $0.01$ of the curve, and the median distance to the curve is $0.379$. Even these $1.67\%$ are not on the curve: a real point lands exactly on it with probability $0$. For every real point the generator's density is $0$, the log-likelihood is $-\infty$ and the forward KL is infinite, so maximum likelihood has nothing to work with.

![一張散點圖，橫軸 x1、縱軸 x2，x1 範圍約 -3.5 到 3.5、x2 約 -3 到 3。藍點是真資料，一團斜斜的二維雲。紅點是生成器的樣本，全部落在一條像 S 形的彎曲細線上，從左下 (-2.2, -0.6) 經過中心彎到右上 (2.2, 0.6)。標題寫：所有假樣本都在一條曲線上，在二維中它們沒有密度（A scatter plot with x1 horizontal and x2 vertical, x1 roughly -3.5 to 3.5 and x2 roughly -3 to 3. Blue points are the real data, a tilted two-dimensional cloud. Red points are the generator's samples, all lying on one thin S-shaped curve running from the lower left (-2.2, -0.6) through the centre to the upper right (2.2, 0.6). The title reads: all fakes lie on one curve, in 2-D they have no density）](../assets/figures/lec13-kp08-implicit-manifold.png)

**圖說 `lec13-kp08-implicit-manifold`**〔功能：空間幾何具象化、一目了然〕：先看藍點：真資料散成一整片。再看紅點：生成器的雜訊只有一維，所以樣本都排在一條線上。藍點幾乎都不在紅線上，用「紅線給藍點多少機率」來評分，每一點都是 $0$。

**Caption `lec13-kp08-implicit-manifold`** [Function: making space concrete, seeing it at a glance]: first look at the blue points: the real data spread over a whole area. Then the red points: the generator's noise is one-dimensional, so its samples line up on a curve. Almost no blue point is on the red curve, so scoring by "how much probability the red curve gives the blue points" gives $0$ everywhere.

這也解釋了本週前面的現象：兩個分布落在不同的「薄曲面」上、幾乎或完全不重疊時，JS 散度停在或非常接近 $\log2$（KP03），而 Wasserstein 距離仍然有意義（KP07）`[補充]`。GAN 的判別器不需要知道 $p_g$ 的公式，只要能比較兩堆樣本，所以能處理這種情況。

This also explains what we saw earlier this week: when two distributions lie on different "thin surfaces" and barely or never overlap, the JS divergence sits at or very near $\log2$ (KP03), while the Wasserstein distance still makes sense (KP07) `[Added]`. A GAN's discriminator does not need a formula for $p_g$, only the ability to compare two piles of samples, so it can handle this situation.

常見錯誤：以為 GAN 就是在極小化反向 KL。來源的互動展示把「MLE 對正向 KL」和「GAN 對反向 KL」並列。KP03 說過，判別器最優、用原本的損失時，GAN 縮小的是 JS 散度；KP06 說過，非飽和損失和反向 KL 有關，但不完全相同。說「GAN 比較像尋找模式」可以，說「GAN 就是反向 KL」不對。

Common mistake: thinking a GAN minimises the reverse KL. The source's demo puts "MLE versus forward KL" next to "GAN versus reverse KL". KP03 showed that with an optimal discriminator and the original loss, a GAN shrinks the JS divergence; KP06 showed that the non-saturating loss is related to the reverse KL but not the same. Saying "a GAN is more mode-seeking" is fine; saying "a GAN is the reverse KL" is not.

### 2. $f$-散度：一個公式涵蓋很多尺（$f$-divergences: one formula covering many rulers）

KL 散度與 JS 散度都是同一家族的成員。挑一個凸函數 $f$，要求 $f(1)=0$，就可以定義 $f$-散度：

The KL and JS divergences belong to one family. Pick a convex function $f$ with $f(1)=0$, and it defines an $f$-divergence:

$$D_f(P\,\Vert\,Q)=\int q(\mathbf{x})\,f\!\left(\frac{p(\mathbf{x})}{q(\mathbf{x})}\right)d\mathbf{x}$$

想法是 `[補充]`：在每一點看密度比 $u=p/q$。兩個分布一樣時處處 $u=1$，$f(1)=0$，散度是 $0$；$u$ 離 $1$ 越遠，凸函數 $f(u)$ 越大，散度就越大。不同的 $f$，代表對「比值偏離 $1$」的懲罰方式不同：

The idea is `[Added]`: look at the density ratio $u=p/q$ at each point. When the two distributions are the same, $u=1$ everywhere and $f(1)=0$, so the divergence is $0$; the farther $u$ is from $1$, the larger the convex $f(u)$ and the larger the divergence. Different choices of $f$ penalise "the ratio straying from $1$" in different ways:

| 散度（Divergence） | $f(u)$ |
|---|---|
| KL $D_{\text{KL}}(P\,\Vert\,Q)$ | $u\log u$ |
| 反向 KL $D_{\text{KL}}(Q\,\Vert\,P)$（reverse KL） | $-\log u$ |
| Pearson $\chi^2$ | $(u-1)^2$ |
| 總變異（total variation） | $\tfrac12\lvert u-1\rvert$ |
| $2\,\mathrm{JSD}$ | $u\log u-(u+1)\log\frac{u+1}{2}$ |

小例子 `[補充]`：兩點的分布 $p=(0.5,\ 0.5)$、$q=(0.8,\ 0.2)$，兩點的比值是 $u=(0.625,\ 2.5)$。用同一個公式 $\sum_x q(x)f(u(x))$ 算出：KL $0.223$、反向 KL $0.193$、Pearson $\chi^2$ $0.5625$、總變異 $0.3$、$2\,\mathrm{JSD}$ $0.101$（所以 $\mathrm{JSD}=0.0507$）。每一個都和用各自的直接公式算的一樣。同一對分布，五把尺量出五個不同的數字。

A small example `[Added]`: two-point distributions $p=(0.5,\ 0.5)$ and $q=(0.8,\ 0.2)$, with ratios $u=(0.625,\ 2.5)$ at the two points. The one formula $\sum_x q(x)f(u(x))$ gives: KL $0.223$, reverse KL $0.193$, Pearson $\chi^2$ $0.5625$, total variation $0.3$, and $2\,\mathrm{JSD}$ $0.101$ (so $\mathrm{JSD}=0.0507$). Each matches its own direct formula. The same pair of distributions gets five different numbers from five rulers.

![左右兩張圖。左圖橫軸是密度比 u，從 0 到 4，縱軸 f(u) 從 -0.6 到 3；五條曲線都在 u=1 交於 0，標著 u=1 時全部是 0：藍線 u log u、紅線 -log u（在 u 小時衝上去）、綠線 (u-1)²、橘線 ½|u-1| 是 V 字、紫線 2JSD 最平緩；圖例放在圖下方。右圖是長條圖，同一對 p=(0.5,0.5)、q=(0.8,0.2)：KL 0.223、反向 KL 0.193、χ² 0.562、TV 0.300、2 JSD 0.101（Two panels. Left: the density ratio u from 0 to 4 horizontally and f(u) from -0.6 to 3 vertically; all five curves meet at 0 when u=1, labelled all are 0 at u=1: blue u log u, red -log u (shooting up for small u), green (u-1)², orange ½|u-1| as a V, and the flattest purple 2JSD; the legend is below the plot. Right: a bar chart for the same pair p=(0.5,0.5), q=(0.8,0.2): KL 0.223, reverse KL 0.193, chi-squared 0.562, TV 0.300, 2 JSD 0.101）](../assets/figures/lec13-kp08-f-functions.png)

**圖說 `lec13-kp08-f-functions`**〔功能：一目了然、精準度〕：先看左圖：五條曲線形狀不同，但都在 $u=1$ 交於 $0$，而且都是往上彎的凸函數。再看右圖：同一對分布，五把尺量出的值都不一樣，$\chi^2$ 最大、$2\,\mathrm{JSD}$ 最小。

**Caption `lec13-kp08-f-functions`** [Function: seeing it at a glance, precision]: first look at the left panel: the five curves have different shapes, but all meet at $0$ when $u=1$, and all bend upward as convex functions. Then the right panel: for the same pair of distributions the five rulers give different values, $\chi^2$ the largest and $2\,\mathrm{JSD}$ the smallest.

`[補充]` 這解釋了本週的幾個連結：KP03 的原始 GAN 在判別器最優時得到 $2\,\mathrm{JSD}-\log4$，對應表中最後一行；KP07 的 LSGAN 對應 Pearson $\chi^2$（比較的是 $p_{\text{data}}+p_g$ 與 $2p_g$，不是表中的 $\chi^2(p_{\text{data}}\,\Vert\,p_g)$）。但 KP07 的 Wasserstein 距離**不是** $f$-散度：$f$-散度只比較「同一點上的密度比」，兩個分布不重疊時比值沒有意義；Wasserstein 距離比較的是「土要搬多遠」，所以不重疊時仍然有用。

`[Added]` This explains several links this week: KP03's original GAN with an optimal discriminator gives $2\,\mathrm{JSD}-\log4$, matching the last row of the table; KP07's LSGAN goes with Pearson $\chi^2$ (between $p_{\text{data}}+p_g$ and $2p_g$, not the table's $\chi^2(p_{\text{data}}\,\Vert\,p_g)$). But KP07's Wasserstein distance is **not** an $f$-divergence: an $f$-divergence compares only "the density ratio at the same point", which is meaningless when the distributions do not overlap; the Wasserstein distance compares "how far the earth must move", so it stays useful without overlap.

常見錯誤：以為不同的 $f$-散度只是同一個數字換了單位。它們會給出不同的數字，而且「誰比較大」也會隨分布改變，對應的訓練行為（涵蓋模式或尋找模式）也不同。

Common mistake: thinking different $f$-divergences are the same number in different units. They give different numbers, which one is larger can change with the distributions, and the corresponding training behaviour (mode-covering or mode-seeking) differs too.

### 3. Fenchel 共軛與 $f$-GAN（The Fenchel conjugate and $f$-GAN）

$f$-散度要知道兩個密度，GAN 卻只有樣本。Nowozin 等人（2016）的 $f$-GAN 用凸函數的 **Fenchel 共軛** 把它改寫成「只要能抽樣就能估計」的形式。凸函數 $f$ 的共軛定義成

An $f$-divergence needs both densities, but a GAN has only samples. The $f$-GAN of Nowozin et al. (2016) uses the **Fenchel conjugate** of a convex function to rewrite it in a form that "can be estimated as long as we can sample". The conjugate of a convex $f$ is defined as

$$f^*(t)=\sup_u\big\{\,u\,t-f(u)\,\big\},$$

而且（對這裡用到的凸函數）反過來也成立：$f(u)=\sup_t\{\,t\,u-f^*(t)\,\}$。把它代進 $f$-散度、把每一點的最佳 $t$ 換成一個函數 $T(\mathbf{x})$，就得到下界：

and (for the convex functions used here) the reverse also holds: $f(u)=\sup_t\{\,t\,u-f^*(t)\,\}$. Substituting this into the $f$-divergence and replacing the best $t$ at each point by a function $T(\mathbf{x})$ gives the lower bound:

$$D_f(P\,\Vert\,Q)\ \ge\ \sup_{T}\Big(\mathbb{E}_{\mathbf{x}\sim P}\big[T(\mathbf{x})\big]-\mathbb{E}_{\mathbf{x}\sim Q}\big[f^*(T(\mathbf{x}))\big]\Big)$$

右邊只用到兩個期望值，用兩堆樣本的平均就能估計。為什麼是下界？`[補充]` 由共軛的定義，任何 $u,t$ 都有 $u\,t-f^*(t)\le f(u)$（Fenchel–Young 不等式）；在每一點取 $u=p/q$、$t=T(\mathbf{x})$，乘上 $q$ 再加總，就得到「任何 $T$ 都 $\le D_f$」。等號在 $T(\mathbf{x})=f'\big(p(\mathbf{x})/q(\mathbf{x})\big)$ 時成立。

The right side uses only two expectations, which can be estimated by averaging over two piles of samples. Why is it a lower bound? `[Added]` By the definition of the conjugate, $u\,t-f^*(t)\le f(u)$ for any $u,t$ (the Fenchel–Young inequality); taking $u=p/q$ and $t=T(\mathbf{x})$ at each point, multiplying by $q$ and summing gives "any $T$ is $\le D_f$". Equality holds when $T(\mathbf{x})=f'\big(p(\mathbf{x})/q(\mathbf{x})\big)$.

用 Pearson $\chi^2$ 算一次 `[補充]`：$f(u)=(u-1)^2$。在 $u\,t-(u-1)^2$ 裡對 $u$ 求最大，得 $u=1+t/2$，代回去 $f^*(t)=t+t^2/4$。第2節的例子裡最好的 $T$ 是 $f'(u)=2(u-1)$，兩點分別是 $-0.75$ 與 $3$；代進下界：$\mathbb{E}_p[T]=0.5\times(-0.75)+0.5\times3=1.125$，$\mathbb{E}_q[f^*(T)]=0.8\times(-0.609)+0.2\times5.25=0.5625$，差是 $0.5625$，正好等於 $\chi^2$。若把 $T$ 縮放成 $c$ 倍，$c=0.5$ 或 $c=1.5$ 時下界都只有 $0.4219$，$c=0$ 時是 $0$。

Working it out for Pearson $\chi^2$ `[Added]`: $f(u)=(u-1)^2$. Maximising $u\,t-(u-1)^2$ over $u$ gives $u=1+t/2$, and substituting back, $f^*(t)=t+t^2/4$. In Section 2's example the best $T$ is $f'(u)=2(u-1)$, which is $-0.75$ and $3$ at the two points; in the bound, $\mathbb{E}_p[T]=0.5\times(-0.75)+0.5\times3=1.125$ and $\mathbb{E}_q[f^*(T)]=0.8\times(-0.609)+0.2\times5.25=0.5625$, so the difference is $0.5625$, exactly $\chi^2$. Scaling $T$ by a factor $c$, the bound is only $0.4219$ at $c=0.5$ or $c=1.5$, and $0$ at $c=0$.

![一張折線圖，橫軸是測試函數的縮放倍數 c，從 0 到 2；縱軸是值，從 -0.05 到 0.7。綠色曲線是下界，形狀像一個倒過來的碗，從 c=0 的 0 升到 c=1 的最高點 0.5625，再降回 c=2 的 0。黑色虛線在 0.5625，標著真正的 χ²。紅點標在 c=1 的最高點，箭頭說明：c=1 時 T 等於 f'(p/q)，下界等於散度（A line chart with the scale c of the test function from 0 to 2 horizontally and value from -0.05 to 0.7 vertically. The green curve is the lower bound, shaped like an upside-down bowl, rising from 0 at c=0 to its peak 0.5625 at c=1 and falling back to 0 at c=2. A dashed black line at 0.5625 is labelled the true chi-squared. A red dot marks the peak at c=1, with an arrow noting: at c=1, T equals f'(p/q) and the bound equals the divergence）](../assets/figures/lec13-kp08-fenchel-bound.png)

**圖說 `lec13-kp08-fenchel-bound`**〔功能：精準度、一目了然〕：先看黑色虛線：那是真正的 $\chi^2=0.5625$。再看綠色曲線：每一個 $T$ 算出的值都在虛線下面，所以是下界。最後看紅點：只有 $T$ 剛好等於 $f'(p/q)$ 時，下界才碰到真正的散度。

**Caption `lec13-kp08-fenchel-bound`** [Function: precision, seeing it at a glance]: first look at the dashed black line: the true $\chi^2=0.5625$. Then the green curve: every $T$ gives a value below the dashed line, so it is a lower bound. Finally the red dot: only when $T$ is exactly $f'(p/q)$ does the bound reach the true divergence.

$f$-GAN 就是把 $T$ 交給判別器網路：判別器把下界推到最大，好估計 $D_f(p_{\text{data}}\,\Vert\,p_g)$；生成器再把這個估計壓小 `[補充]`。換一個 $f$，就得到一個新的 GAN。這和 KP03 是同一件事：KP03 的最優判別器 $D^*=p_{\text{data}}/(p_{\text{data}}+p_g)$，它的 logit 就是 $\log(p_{\text{data}}/p_g)$，判別器其實在學密度比。

$f$-GAN hands $T$ to the discriminator network: the discriminator pushes the bound as high as it can to estimate $D_f(p_{\text{data}}\,\Vert\,p_g)$, and the generator then pushes that estimate down `[Added]`. Change $f$ and you get a new GAN. This is the same thing as in KP03: KP03's optimal discriminator $D^*=p_{\text{data}}/(p_{\text{data}}+p_g)$ has logit $\log(p_{\text{data}}/p_g)$, so the discriminator is really learning the density ratio.

常見錯誤：以為下界算出來越大越不準。剛好相反：任何 $T$ 都只會**低估**散度，所以判別器要把下界推到**最大**，才最接近真正的散度。

Common mistake: thinking a larger computed bound is less accurate. The opposite holds: any $T$ can only **underestimate** the divergence, so the discriminator must push the bound to its **maximum** to get closest to the true divergence.

### 4. 貝氏的觀點：AVB 與貝氏 GAN（The Bayesian view: AVB and the Bayesian GAN）

**對抗變分貝氏（AVB）。** 第12週的 VAE 要求編碼器的後驗 $q_\phi(\mathbf{z}\mid\mathbf{x})$ 是簡單的高斯分布，因為 KL 散度要有公式才算得出來。Mescheder 等人（2017）的對抗變分貝氏（adversarial variational Bayes，AVB）讓後驗可以是任何能抽樣的複雜分布，再用一個判別器去估計 KL 需要的對數密度比 $\log q_\phi(\mathbf{z}\mid\mathbf{x})-\log p(\mathbf{z})$。`[補充]` 這用的正是第3節的道理：判別器在分辨兩堆樣本時，自然學到它們的密度比。

**Adversarial variational Bayes (AVB).** Week 12's VAE requires the encoder's posterior $q_\phi(\mathbf{z}\mid\mathbf{x})$ to be a simple Gaussian, because the KL divergence needs a formula to be computed. The adversarial variational Bayes (AVB) of Mescheder et al. (2017) lets the posterior be any complex distribution we can sample from, and uses a discriminator to estimate the log density ratio $\log q_\phi(\mathbf{z}\mid\mathbf{x})-\log p(\mathbf{z})$ that the KL needs. `[Added]` This uses exactly the idea of Section 3: a discriminator telling two piles of samples apart naturally learns their density ratio.

**貝氏 GAN（Bayesian GAN）。** 一般的 GAN 只找一組生成器權重 $\boldsymbol{\theta}_G$ 和一組判別器權重 $\boldsymbol{\theta}_D$。Saatci 與 Wilson（2017）的貝氏 GAN 改成考慮權重的**後驗分布** $p(\boldsymbol{\theta}_G\mid\mathcal{D})$、$p(\boldsymbol{\theta}_D\mid\mathcal{D})$，用馬可夫鏈蒙地卡羅（MCMC）從後驗抽出好幾組權重（論文用的是隨機梯度漢米頓蒙地卡羅 SGHMC；來源以較簡單的同類方法，隨機梯度朗之萬動力學（stochastic gradient Langevin dynamics，SGLD）說明）。SGLD 每一步沿對數後驗的梯度走 $\eta/2$ 倍的一步，再加一點隨機雜訊，

**The Bayesian GAN.** An ordinary GAN finds one set of generator weights $\boldsymbol{\theta}_G$ and one set of discriminator weights $\boldsymbol{\theta}_D$. The Bayesian GAN of Saatci and Wilson (2017) considers instead the **posterior distributions** of the weights, $p(\boldsymbol{\theta}_G\mid\mathcal{D})$ and $p(\boldsymbol{\theta}_D\mid\mathcal{D})$, and draws several sets of weights from the posterior by Markov chain Monte Carlo (MCMC) (the paper uses stochastic gradient Hamiltonian Monte Carlo, SGHMC; the source illustrates it with the simpler related method stochastic gradient Langevin dynamics, SGLD). Each SGLD step moves $\eta/2$ times the gradient of the log posterior and adds a little random noise,

$$\Delta\boldsymbol{\theta}=\frac{\eta}{2}\nabla_{\boldsymbol{\theta}}\log p(\boldsymbol{\theta}\mid\mathcal{D})+\boldsymbol{\epsilon},\qquad\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\ \eta I).$$

`[補充]` 每一組生成器權重可能專精不同的模式，合起來比較多樣，這是它想減輕 KP06 模式崩塌的原因。代價是要同時維持好幾組網路，計算量跟著增加；論文報告有改善，但不保證。

`[Added]` Each set of generator weights may specialise in different modes, and together they are more varied; this is how it aims to ease KP06's mode collapse. The cost is maintaining several networks at once, with computation growing accordingly; the paper reports improvements, but there is no guarantee.

常見錯誤：以為 SGLD 只是「加了雜訊的梯度下降」，雜訊越大越好。雜訊的變異數必須剛好等於步長 $\eta$，是為了讓抽出來的權重**服從後驗分布**；隨便加大雜訊，抽到的就不再是後驗。

Common mistake: thinking SGLD is just "gradient descent with noise", and the more noise the better. The noise variance must equal the step size $\eta$ so that the sampled weights **follow the posterior**; adding more noise at will means the samples no longer come from the posterior.

### 本單元重點整理（KP08 summary）

**一句話重點**：顯式模型寫得出密度、用最大概似（等於極小化正向 KL）訓練；GAN 是隱式模型，只能抽樣，而且樣本常落在低維曲面上，使最大概似無法使用；KL、反向 KL、$\chi^2$、總變異、JS 散度都是 $f$-散度；$f$-GAN 用 Fenchel 共軛把它寫成判別器可以估計的下界；AVB 用判別器估計密度比讓 VAE 的後驗更靈活，貝氏 GAN 用 MCMC（論文用 SGHMC，SGLD 是較簡單的同類方法）抽多組權重來增加多樣性。

**One-sentence summary**: explicit models write down a density and train by maximum likelihood (minimising the forward KL); a GAN is an implicit model that can only sample, and its samples often lie on a low-dimensional surface, which rules out maximum likelihood; the KL, reverse KL, $\chi^2$, total variation and JS divergences are all $f$-divergences; $f$-GAN uses the Fenchel conjugate to write them as a lower bound the discriminator can estimate; AVB uses a discriminator to estimate a density ratio so a VAE's posterior can be more flexible, and the Bayesian GAN samples several sets of weights by MCMC (SGHMC in the paper; SGLD is a simpler related method) for more variety.

**白話說明**：最大概似像是「替每一張真圖打分數：模型覺得它有多可能」；GAN 像是「不打分數，只把真圖和假圖放在一起，看能不能分出來」。$f$-散度是一整套尺，每一把都在問「這裡的真假比例偏離 1 多少」；$f$-GAN 說，只要讓判別器去猜這個比例，任何一把尺都能做成 GAN。

**Plain explanation**: maximum likelihood is like "scoring each real image by how likely the model finds it"; a GAN is like "not scoring at all, just putting real and fake images together and seeing whether they can be told apart". $f$-divergences are a whole set of rulers, each asking "how far does the real-to-fake ratio here stray from 1"; $f$-GAN says that as long as the discriminator learns to guess this ratio, any of these rulers can become a GAN.

**正式定義**：$\max_\theta\mathbb{E}_{p_{\text{data}}}[\log p_\theta]\iff\min_\theta D_{\text{KL}}(p_{\text{data}}\,\Vert\,p_\theta)$；$D_f(P\,\Vert\,Q)=\int q\,f(p/q)$，$f$ 凸、$f(1)=0$；$f^*(t)=\sup_u\{ut-f(u)\}$；$D_f(P\,\Vert\,Q)\ge\sup_T\mathbb{E}_P[T]-\mathbb{E}_Q[f^*(T)]$，等號在 $T=f'(p/q)$；SGLD $\Delta\boldsymbol{\theta}=\frac{\eta}{2}\nabla\log p(\boldsymbol{\theta}\mid\mathcal{D})+\boldsymbol{\epsilon}$，$\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\eta I)$。

**Formal definition**: $\max_\theta\mathbb{E}_{p_{\text{data}}}[\log p_\theta]\iff\min_\theta D_{\text{KL}}(p_{\text{data}}\,\Vert\,p_\theta)$; $D_f(P\,\Vert\,Q)=\int q\,f(p/q)$ with $f$ convex and $f(1)=0$; $f^*(t)=\sup_u\{ut-f(u)\}$; $D_f(P\,\Vert\,Q)\ge\sup_T\mathbb{E}_P[T]-\mathbb{E}_Q[f^*(T)]$, with equality at $T=f'(p/q)$; SGLD $\Delta\boldsymbol{\theta}=\frac{\eta}{2}\nabla\log p(\boldsymbol{\theta}\mid\mathcal{D})+\boldsymbol{\epsilon}$, $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\eta I)$.

**例子**：一維雜訊生成二維資料時，只有 $1.67\%$ 的真資料點離生成曲線在 $0.01$ 以內；$p=(0.5,0.5)$、$q=(0.8,0.2)$ 時 KL $0.223$、反向 KL $0.193$、$\chi^2$ $0.5625$、總變異 $0.3$、JSD $0.0507$；$\chi^2$ 的共軛 $f^*(t)=t+t^2/4$，最佳 $T=(-0.75,\ 3)$ 時下界等於 $0.5625$，縮放成 $0.5$ 或 $1.5$ 倍時只有 $0.4219$。

**Example**: with one-dimensional noise generating two-dimensional data, only $1.67\%$ of the real points are within $0.01$ of the generated curve; for $p=(0.5,0.5)$ and $q=(0.8,0.2)$, KL is $0.223$, reverse KL $0.193$, $\chi^2$ $0.5625$, total variation $0.3$ and JSD $0.0507$; the conjugate of $\chi^2$ is $f^*(t)=t+t^2/4$, and the bound equals $0.5625$ at the best $T=(-0.75,\ 3)$ but only $0.4219$ when scaled by $0.5$ or $1.5$.

**常見迷思**：

- 「GAN 就是在極小化反向 KL。」錯，原始 GAN 在判別器最優時縮小的是 JS 散度。
- 「Wasserstein 距離也是一種 $f$-散度。」錯，它比較位置，不比較同一點的密度比。
- 「$f$-GAN 的下界越大越不準。」錯，任何 $T$ 都只會低估，越大越接近真值。
- 「SGLD 的雜訊越大越好。」錯，雜訊的變異數必須等於步長，才是從後驗抽樣。

**Common misconceptions**:

- "A GAN minimises the reverse KL." Wrong: the original GAN with an optimal discriminator shrinks the JS divergence.
- "The Wasserstein distance is also an $f$-divergence." Wrong: it compares positions, not density ratios at the same point.
- "A larger $f$-GAN bound is less accurate." Wrong: any $T$ only underestimates, so larger is closer to the truth.
- "More noise in SGLD is better." Wrong: the noise variance must equal the step size for the samples to come from the posterior.

**圖**：本單元有三張圖。

- `lec13-kp08-implicit-manifold`：一維雜訊的生成器，樣本都在一條曲線上，真資料幾乎都不在上面。
- `lec13-kp08-f-functions`：五個 $f$ 函數，以及同一對分布的五個 $f$-散度值。
- `lec13-kp08-fenchel-bound`：$\chi^2$ 的下界隨 $T$ 的縮放變化，只有最佳 $T$ 時等於散度。

**Figures** — this unit has three figures.

- `lec13-kp08-implicit-manifold`: a generator with one-dimensional noise, whose samples all lie on one curve that almost no real point lies on.
- `lec13-kp08-f-functions`: five $f$ functions and the five $f$-divergence values for one pair of distributions.
- `lec13-kp08-fenchel-bound`: the $\chi^2$ lower bound as $T$ is scaled, equal to the divergence only at the best $T$.

### 小練習（Practice）

1. 從 $D_{\text{KL}}(p_{\text{data}}\,\Vert\,p_\theta)=\mathbb{E}_{p_{\text{data}}}[\log p_{\text{data}}]-\mathbb{E}_{p_{\text{data}}}[\log p_\theta]$ 出發，說明為什麼最大化平均對數概似等於極小化正向 KL。
2. 用 $D_f=\sum_x q(x)f\big(p(x)/q(x)\big)$ 與 $f(u)=(u-1)^2$，手算 $p=(0.5,0.5)$、$q=(0.8,0.2)$ 的 Pearson $\chi^2$。
3. 對 $f(u)=(u-1)^2$，$T=(0,\ 0)$ 時下界是多少？為什麼任何 $T$ 都不會超過 $0.5625$？

1. Starting from $D_{\text{KL}}(p_{\text{data}}\,\Vert\,p_\theta)=\mathbb{E}_{p_{\text{data}}}[\log p_{\text{data}}]-\mathbb{E}_{p_{\text{data}}}[\log p_\theta]$, explain why maximising the average log-likelihood is the same as minimising the forward KL.
2. Using $D_f=\sum_x q(x)f\big(p(x)/q(x)\big)$ with $f(u)=(u-1)^2$, compute by hand the Pearson $\chi^2$ for $p=(0.5,0.5)$ and $q=(0.8,0.2)$.
3. For $f(u)=(u-1)^2$, what is the lower bound when $T=(0,\ 0)$? Why can no $T$ exceed $0.5625$?

**第 1 題解答** `[補充]`：第一項 $\mathbb{E}_{p_{\text{data}}}[\log p_{\text{data}}]$ 只和真資料有關，不隨 $\theta$ 改變。所以 KL 越小，第二項 $\mathbb{E}_{p_{\text{data}}}[\log p_\theta]$ 就要越大；而它正是資料很多時的平均對數概似。

**Solution to question 1** `[Added]`: the first term $\mathbb{E}_{p_{\text{data}}}[\log p_{\text{data}}]$ depends only on the real data and does not change with $\theta$. So the smaller the KL, the larger the second term $\mathbb{E}_{p_{\text{data}}}[\log p_\theta]$ must be, and that is exactly the average log-likelihood when there is plenty of data.

**第 2 題解答** `[補充]`：比值 $u=(0.625,\ 2.5)$，$f(u)=(0.140625,\ 2.25)$。加權：$0.8\times0.140625+0.2\times2.25=0.1125+0.45=0.5625$。

**Solution to question 2** `[Added]`: the ratios are $u=(0.625,\ 2.5)$ and $f(u)=(0.140625,\ 2.25)$. Weighted: $0.8\times0.140625+0.2\times2.25=0.1125+0.45=0.5625$.

**第 3 題解答** `[補充]`：$f^*(0)=0$，所以下界是 $0-0=0$。任何 $u,t$ 都有 $u\,t-f^*(t)\le f(u)$（共軛的定義）；在每一點代 $u=p/q$、$t=T$，乘 $q$ 再加總，左邊就是下界、右邊就是 $\chi^2=0.5625$。

**Solution to question 3** `[Added]`: $f^*(0)=0$, so the bound is $0-0=0$. Any $u,t$ satisfy $u\,t-f^*(t)\le f(u)$ (the definition of the conjugate); substituting $u=p/q$ and $t=T$ at each point, multiplying by $q$ and summing, the left side becomes the bound and the right side $\chi^2=0.5625$.

### 小測驗（Quiz，來源原題）

來源 KP-9、KP-10、KP-11 各有一題自我檢測，保留原題與全部選項；來源頁面附有解析，下面依它說明。

The source's KP-9, KP-10 and KP-11 each have one self-check question, kept here with all options; the source page gives explanations, which the note follows below.

1. 極大似然估計 (MLE) 與 GAN 隱式生成模型在學習數據分佈上的根本差異為何？
   - (A) MLE 需要顯式定義機率密度 $p_\theta(\mathbf{x})$，而 GAN 為不需寫出機率密度的隱式採樣模型
   - (B) MLE 只能處理一維數據，而 GAN 可以處理圖像
   - (C) MLE 採用極小化極大賽局，而 GAN 採用最小二乘法
   - (D) 兩者在數學上完全沒有任何區別

1. What is the fundamental difference between maximum likelihood estimation (MLE) and the implicit GAN model in learning a data distribution?
   - (A) MLE needs an explicitly defined probability density $p_\theta(\mathbf{x})$, while a GAN is an implicit sampling model that need not write down a density
   - (B) MLE handles only one-dimensional data, while a GAN can handle images
   - (C) MLE uses a minimax game, while a GAN uses least squares
   - (D) There is no mathematical difference between them at all

**解析**：選 (A)。來源的理由：MLE 是顯式密度模型，直接最佳化 $\log p_\theta(\mathbf{x})$；GAN 是隱式模型，透過抽樣與判別器的對抗擬合分布（見第1節）。`[補充]` (B) 錯，自迴歸、流模型等顯式模型也能處理影像；(C) 剛好說反，極小化極大是 GAN 的做法；(D) 錯，第1節的例子裡兩者的行為完全不同。

**Explanation**: the answer is A. The source's reason: MLE is an explicit density model that optimises $\log p_\theta(\mathbf{x})$ directly, while a GAN is an implicit model that fits the distribution through sampling and competition with a discriminator (see Section 1). `[Added]` B is wrong, since explicit models such as autoregressive and flow models handle images too; C has it backwards, since the minimax game is the GAN's method; D is wrong, since in Section 1's example the two behave entirely differently.

2. Nowozin 等人提出的 $f$-GAN 理論證明了什麼重要結論？
   - (A) 證明了只有 JS 散度可以用於 GAN 的對抗訓練
   - (B) 利用 Fenchel 凸對偶變換，證明了所有廣義 $f$-散度 (如 Total Variation, Pearson $\chi^2$) 都能統一構建對應的 GAN 對抗目標
   - (C) 證明了生成器神經網絡不需要激活函數
   - (D) 證明了判別器權重必須保持全正數

2. What important result did the $f$-GAN theory of Nowozin et al. establish?
   - (A) That only the JS divergence can be used for adversarial GAN training
   - (B) That, using the Fenchel convex duality, every generalised $f$-divergence (such as total variation and Pearson $\chi^2$) can be turned into a corresponding adversarial GAN objective in one unified way
   - (C) That the generator network needs no activation functions
   - (D) That the discriminator's weights must all stay positive

**解析**：選 (B)。來源的理由：$f$-GAN 用凸對偶把散度的下界變成變分最佳化的形式，任何 $f$-散度都能導出對應的對抗訓練（見第3節）。`[補充]` (A) 正好相反；(C)、(D) 和 $f$-GAN 無關。也要記得：這是**下界**，判別器能力不夠時會低估散度。

**Explanation**: the answer is B. The source's reason: $f$-GAN uses convex duality to turn a lower bound on the divergence into a variational optimisation, so any $f$-divergence yields a corresponding adversarial training scheme (see Section 3). `[Added]` A is the opposite; C and D have nothing to do with $f$-GAN. Also remember that it is a **lower bound**: a discriminator that is not capable enough underestimates the divergence.

3. 對抗變分貝氏 (AVB) 如何改進傳統 VAE 變分推斷的限制？
   - (A) 徹底取消 Encoder 神經網絡
   - (B) 利用對抗判別器隱式估算對數密度比，打破變分後驗 $q(\mathbf{z}\mid\mathbf{x})$ 必須為簡單高斯分佈的限制
   - (C) 將影像強制降採樣為灰階圖
   - (D) 使用手動規則替代反向傳播

3. How does adversarial variational Bayes (AVB) improve on the limitation of variational inference in the traditional VAE?
   - (A) It removes the encoder network entirely
   - (B) It uses an adversarial discriminator to estimate the log density ratio implicitly, lifting the restriction that the variational posterior $q(\mathbf{z}\mid\mathbf{x})$ must be a simple Gaussian
   - (C) It forces images to be downsampled to greyscale
   - (D) It replaces backpropagation with hand-written rules

**解析**：選 (B)。來源的理由：AVB 用判別器的對抗計算後驗與先驗的密度比，讓模型可以用沒有封閉公式的複雜網路當後驗（見第4節）。`[補充]` (A) 錯，AVB 仍然有編碼器，而且編碼器可以更複雜；(C)、(D) 和變分推斷無關。

**Explanation**: the answer is B. The source's reason: AVB uses a discriminator in competition to compute the density ratio of the posterior and the prior, so the model can use a complex network without a closed-form formula as its posterior (see Section 4). `[Added]` A is wrong, since AVB still has an encoder, and it can be more complex; C and D have nothing to do with variational inference.

### 留給下一單元的問題（The question for the next unit）

本週到這裡結束。回顧一下：GAN 用「判別器分辨兩堆樣本」來代替寫出機率密度，這個想法讓它能畫出清楚的樣本，卻也帶來不穩定與模式崩塌；本週後半看到的都是讓這把「分辨的尺」更可靠的方法。課程接下來的週次會把這些生成模型放到應用與期末專題裡，你可以想想：你的專題資料比較適合顯式模型、VAE，還是 GAN？為什麼？

This week ends here. To recap: a GAN replaces writing down a probability density with "a discriminator telling two piles of samples apart", an idea that lets it draw sharp samples but also brings instability and mode collapse; the second half of this week was all about making this "ruler of telling apart" more reliable. The coming weeks of the course put these generative models into applications and final projects; think about it: would your project's data suit an explicit model, a VAE, or a GAN better? Why?

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第14章「產生式對抗網絡」的 KP-9「極大似然估計（MLE）與 GAN 隱式分配對比」、KP-10「$f$-GAN 廣義變分凸對偶推導」與 KP-11「對抗變分貝氏（AVB）與貝氏 GAN（BGAN）」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch14-gan/index.html ，讀取日期 2026-11-15）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from Chapter 14, "Generative adversarial networks", of CCC-LNS IDL_115: KP-9, "Maximum likelihood estimation (MLE) versus the GAN's implicit distribution"; KP-10, "The generalised variational convex-duality derivation of $f$-GAN"; and KP-11, "Adversarial variational Bayes (AVB) and the Bayesian GAN (BGAN)" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch14-gan/index.html , accessed 2026-11-15). Content not in the source is marked `[Added]` where it appears.

- 來源 KP-10 把 Fenchel 共軛寫成 $f^*(t)=\sup_u\{ut-f^*(t)\}$，右邊誤用了 $f^*$；正確是 $f^*(t)=\sup_u\{ut-f(u)\}$，反過來 $f(u)=\sup_t\{tu-f^*(t)\}$。本講義第3節改正，並用 $\chi^2$ 實際算一次。
- 來源 KP-9 的互動展示把 GAN 和反向 KL 畫上等號；原始 GAN 在判別器最優時縮小的是 JS 散度，非飽和損失只和反向 KL 有關（KP03、KP06）。本講義第1節的常見錯誤說明。
- 來源說最大概似「傾向於產生平均泛化的模糊樣本」；本講義保留這個說法，但註明它是常見的直觀解釋，不是定理。
- 來源說貝氏 GAN「有效防止了過擬合與模式崩塌」，說得太絕對；本講義改成論文報告有改善，但不保證，並註明計算代價。
- 來源寫的是「Saatci 等人」；本講義寫明作者是 Saatci 與 Wilson（2017）。
- 來源說貝氏 GAN 用 SGLD 抽樣，公式裡用的是先驗 $\nabla\log p(\boldsymbol{\theta})$；論文實際用 SGHMC（SGLD 是較簡單的同類方法），而且要沿**後驗** $p(\boldsymbol{\theta}\mid\mathcal{D})$ 的梯度走。本講義保留 SGLD 公式作說明，並改成後驗。
- 來源用粗斜體 `\boldsymbol` 表示一般向量，並把 KL 散度寫成 $KLD$；本講義對資料向量改用 `\mathbf`（權重 $\boldsymbol{\theta}$、雜訊 $\boldsymbol{\epsilon}$ 等希臘字母保留粗斜體），KL 用第12週的 $D_{\text{KL}}$。
- 本講義補上：最大概似等於正向 KL 的推導、低維曲面使最大概似無法使用的例子與數字、和 KP03／KP07 的關聯、$f$-散度的五個 $f$ 與數字例子、Wasserstein 距離不是 $f$-散度、Fenchel–Young 不等式與等號條件、$\chi^2$ 共軛的推導與數字、判別器在學密度比、AVB 與第3節的關聯、SGLD 雜訊大小的意義、三題小練習與解答。

- The source's KP-10 writes the Fenchel conjugate as $f^*(t)=\sup_u\{ut-f^*(t)\}$, wrongly using $f^*$ on the right; the correct form is $f^*(t)=\sup_u\{ut-f(u)\}$, and conversely $f(u)=\sup_t\{tu-f^*(t)\}$. Section 3 corrects this and works through $\chi^2$.
- The source's KP-9 demo equates the GAN with the reverse KL; the original GAN with an optimal discriminator shrinks the JS divergence, and only the non-saturating loss relates to the reverse KL (KP03, KP06). The common mistake in Section 1 explains this.
- The source says maximum likelihood "tends to produce averaged, blurry samples"; this note keeps the statement but notes it is a common intuitive explanation, not a theorem.
- The source says the Bayesian GAN "effectively prevents overfitting and mode collapse", which is too absolute; this note says the paper reports improvements without a guarantee, and mentions the computational cost.
- The source writes "Saatci et al."; this note names the authors, Saatci and Wilson (2017).
- The source says the Bayesian GAN samples with SGLD, with the prior $\nabla\log p(\boldsymbol{\theta})$ in its formula; the paper actually uses SGHMC (SGLD is a simpler related method), and the step must follow the gradient of the **posterior** $p(\boldsymbol{\theta}\mid\mathcal{D})$. This note keeps the SGLD formula as an illustration and uses the posterior.
- The source writes ordinary vectors with bold italic `\boldsymbol` and the KL divergence as $KLD$; this note uses `\mathbf` for data vectors (keeping bold italic for Greek letters such as the weights $\boldsymbol{\theta}$ and the noise $\boldsymbol{\epsilon}$) and Week 12's $D_{\text{KL}}$.
- This note adds: the derivation that maximum likelihood equals the forward KL, the example and numbers showing that a low-dimensional surface rules out maximum likelihood, the links with KP03 and KP07, the five $f$ functions and a numerical example, the fact that the Wasserstein distance is not an $f$-divergence, the Fenchel–Young inequality and the equality condition, the derivation and numbers for the $\chi^2$ conjugate, the discriminator learning the density ratio, the link between AVB and Section 3, the meaning of the noise size in SGLD, and three practice questions with solutions.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 顯式模型 | explicit model |
| 隱式模型 | implicit model |
| 最大概似估計 | maximum likelihood estimation (MLE) |
| 對數概似 | log-likelihood |
| 正向 KL、反向 KL（KP06） | forward KL, reverse KL |
| 低維曲面 | low-dimensional surface (manifold) |
| $f$-散度 | $f$-divergence |
| 凸函數 | convex function |
| 密度比 | density ratio |
| Pearson $\chi^2$ 散度 | Pearson $\chi^2$ divergence |
| 總變異距離 | total variation distance |
| Fenchel 共軛 | Fenchel conjugate |
| Fenchel–Young 不等式 | Fenchel–Young inequality |
| 下界 | lower bound |
| 測試函數 | test function |
| 對抗變分貝氏 | adversarial variational Bayes (AVB) |
| 後驗分布 | posterior distribution |
| 貝氏 GAN | Bayesian GAN |
| 隨機梯度朗之萬動力學 | stochastic gradient Langevin dynamics (SGLD) |
| Wasserstein 距離（KP07） | Wasserstein distance |
