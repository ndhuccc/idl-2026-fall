# 第13週 KP06：模式崩塌與對策，生成器為什麼只畫幾種、怎麼逼它畫全部（Mode collapse and its remedies: why the generator draws only a few kinds, and how to make it draw them all）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第13週的第六個知識點（KP06），接在 KP05 之後。KP05 的練習題提到：生成器可能不管輸入什麼，都畫同一種東西。這一節正面處理這個問題。它叫做模式崩塌，是 GAN 訓練最常見、也最惡名昭彰的失敗方式之一。我們先用一個小實驗看它長什麼樣子，再說明它為什麼會發生，最後介紹三種對策：小批次判別、特徵匹配、展開 GAN。

This is the sixth knowledge point (KP06) of Week 13, following KP05. A KP05 practice question noted that the generator might draw the same thing whatever its input. This note tackles that problem head on. It is called mode collapse, and it is one of the most common and notorious ways GAN training fails. We first look at it in a small experiment, then explain why it happens, and finally introduce three remedies: minibatch discrimination, feature matching and the unrolled GAN.

先備知識是 KP02 的非飽和損失、KP03 的 JS 散度、KP04 的擺盪，以及第12週 KP02 的 KL 散度。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第14章的 KP-7 與 KP-12（特徵匹配部分）改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec13-kp06-figures.py` 算出，可以自己重算（需要 PyTorch，單執行緒約三分鐘）。

The prerequisites are KP02's non-saturating loss, KP03's JS divergence, KP04's oscillation, and the KL divergence of Week 12 KP02. Every term and symbol is defined when it first appears. This note is rewritten from KP-7 and the feature-matching part of KP-12 of Chapter 14 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec13-kp06-figures.py`, so you can recompute it yourself (it needs PyTorch and takes about three minutes on one thread).

## KP06 模式崩塌與對策（Mode collapse and its remedies）

### 學習目標（Learning objectives）

讀完之後，你能說出什麼是模式，什麼是完全崩塌與部分崩塌，並在二維的小實驗裡數出生成器涵蓋了幾個模式。你能用「生成器對著固定的判別器找最好答案」說明崩塌為什麼會發生，也能用兩個方向的 KL 散度，說明「只守住一個峰」和「蓋住全部峰」的差別。你能說明小批次判別、特徵匹配、展開 GAN 各自的想法，算出一個批次裡的「近鄰數」，並說出特徵匹配的限制。

After reading, you can say what a mode is, what full and partial collapse are, and count how many modes a generator covers in a small two-dimensional experiment. You can explain why collapse happens using "the generator's best answer against a fixed discriminator", and use the two directions of the KL divergence to explain the difference between "holding one peak" and "covering all peaks". You can explain the idea of each of minibatch discrimination, feature matching and the unrolled GAN, compute the "number of close neighbours" in a batch, and state a limitation of feature matching.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $G,\ D,\ \mathbf{z},\ p_z$ | 生成器、判別器、雜訊與雜訊分布（同 KP02） | generator, discriminator, noise and its distribution (as in KP02) |
| $p_{\text{data}},\ p_g$ | 真資料與生成資料的分布 | the distributions of real and generated data |
| $\mathrm{JSD}$ | JS 散度（KP03） | JS divergence (KP03) |
| $D_{\text{KL}}(P\,\Vert\,Q)$ | KL 散度（第12週 KP02） | KL divergence (Week 12 KP02) |
| $q$ | 第2節裡用來配合目標的常態分布 | in Section 2, the normal distribution fitted to the target |
| $n$ | 一個批次的樣本數 | the number of samples in a batch |
| $r$ | 判斷「靠得很近」的距離門檻 | the distance threshold for "very close" |
| $\mathbf{f}(\mathbf{x})$ | 判別器某個中間層的特徵向量 | the feature vector of an intermediate layer of the discriminator |
| $\mathcal{L}_{\text{FM}}$ | 特徵匹配損失 | the feature-matching loss |
| $k$ | 展開 GAN 往前看的判別器更新步數 | the number of discriminator updates the unrolled GAN looks ahead |

### 1. 什麼是模式崩塌（What mode collapse is）

一個分布的模式（mode）就是它的一個峰，也就是資料特別集中的一群。手寫數字有 $0$ 到 $9$ 十種，就可以看成十個模式。模式崩塌（mode collapse）是指：真資料有很多個模式，生成器卻只畫其中一個或少數幾個，其他的全部放棄。每一張生成的圖可能都很逼真，但整體缺乏多樣性。只畫一種叫完全崩塌；漏掉一部分叫部分崩塌，也叫模式遺漏（mode dropping）`[補充]`。

A mode of a distribution is one of its peaks, a group where the data are especially concentrated. Handwritten digits come in ten kinds, $0$ to $9$, which can be seen as ten modes. Mode collapse means that the real data have many modes but the generator draws only one or a few of them and gives up on the rest. Each generated image may be very realistic, yet as a whole they lack variety. Drawing only one kind is full collapse; missing some is partial collapse, also called mode dropping `[Added]`.

我們用一個小實驗看它 `[補充]`。真資料是平面上 $8$ 個小團，平均分布在半徑 $2$ 的圓上，每團的標準差 $0.05$。生成器和判別器都是兩個隱藏層、每層 $128$ 個單元的小網路，雜訊 $\mathbf{z}$ 是 $2$ 維，用 KP02 的非飽和損失與 Adam 最佳化器，每批 $256$ 個樣本，每步各更新一次，訓練 $5000$ 步。若 $2000$ 個生成樣本裡，至少 $2\%$ 落在某團中心 $0.15$ 以內，就算這個模式被涵蓋。

Let us see it in a small experiment `[Added]`. The real data are $8$ small clusters in the plane, evenly placed on a circle of radius $2$, each with standard deviation $0.05$. The generator and discriminator are small networks with two hidden layers of $128$ units, the noise $\mathbf{z}$ is $2$-dimensional, and we use KP02's non-saturating loss with the Adam optimiser, batches of $256$, and one update each per step, for $5000$ steps. A mode counts as covered if at least $2\%$ of $2000$ generated samples lie within $0.15$ of its centre.

![左中右三張散點圖，橫軸 x1、縱軸 x2，範圍都是 -3 到 3，八個空心圓標出真資料的八個中心，排成一圈。左圖藍點是真資料，八團都很集中。中圖紅點是標準 GAN（種子 1）：只在四個斜角的中心聚成團，上下左右四個中心幾乎沒有點，團與團之間沿著圓還有一條稀疏的紅點，標題寫 4 of 8 modes。右圖綠點是特徵匹配（種子 1）：八個中心都有點，但點沿著整個圓散開，標題寫 8 of 8 modes（Three scatter plots with x1 horizontal and x2 vertical, both from -3 to 3; eight open circles mark the eight real centres on a ring. Left: blue real data, eight tight clusters. Middle: red points from the standard GAN (seed 1) cluster only at the four diagonal centres, with almost none at the top, bottom, left and right centres, and a thin trail of red points runs along the ring between clusters; the title says 4 of 8 modes. Right: green points from feature matching (seed 1) reach all eight centres but spread along the whole circle; the title says 8 of 8 modes）](../assets/figures/lec13-kp06-ring-samples.png)

**圖說 `lec13-kp06-ring-samples`**〔功能：一目了然、空間幾何具象化〕：先看左圖：真資料是八團。再看中圖：標準 GAN 只畫出四個斜角的團，上下左右四團被放棄了，這就是部分崩塌；團與團之間另有一條稀疏的點。最後看右圖：改用特徵匹配（第4節）後，八團都有點，但點比較散，很多落在團與團之間。

**Caption `lec13-kp06-ring-samples`** [Function: seeing it at a glance, making space concrete]: first look at the left panel: the real data are eight clusters. Then the middle panel: the standard GAN draws only the four diagonal clusters and abandons the top, bottom, left and right ones; this is partial collapse. A thin trail of points also runs between the clusters. Finally the right panel: with feature matching (Section 4), all eight clusters get points, but the points are more scattered, and many fall between clusters.

隨機種子決定網路的初始權重與每次抽的雜訊。用 $5$ 個不同的隨機種子各跑一次，標準 GAN 最後涵蓋的模式數是 $6,4,4,4,4$，沒有一次涵蓋全部 $8$ 個。最後停在 $4$ 個的四次，都在第 $1000$ 步以前就到了 $4$，之後一直到第 $5000$ 步都沒有再變（每 $250$ 步檢查一次）。

A random seed decides the initial weights and the noise drawn each time. Running $5$ different random seeds, the standard GAN ends up covering $6,4,4,4,4$ modes, never all $8$. The four runs that end at $4$ all reach $4$ by step $1000$ and stay there until step $5000$ (checked every $250$ steps).

常見錯誤：以為每張生成的圖都很逼真，就代表 GAN 訓練成功。模式崩塌的生成器，每一張都可能很逼真；問題在於整體缺乏多樣性。要檢查多樣性，必須一次看很多張，或像上面一樣數涵蓋了幾個模式。

Common mistake: thinking that if every generated image looks realistic, the GAN has trained well. A collapsed generator may make every image realistic; the problem is the lack of variety as a whole. To check variety you must look at many samples at once, or count the covered modes as above.

### 2. 為什麼會崩塌（Why collapse happens）

**理由一：判別器一次只看一張。** `[補充]` 判別器對每一張圖各自打分數，它不知道同一批裡其他圖長什麼樣子。想像判別器暫時固定不動，生成器要讓 $D(G(\mathbf{z}))$ 越大越好。最好的做法是什麼？找出判別器分數最高的那一點，把**所有** $\mathbf{z}$ 都送到那裡。畫得多不多樣，判別器根本看不到，所以生成器沒有理由保持多樣。判別器接著學會「那一點是假的」，生成器就跳到下一個高分處；這就是 KP04 的「你追我、我追你」，在模式之間跳來跳去，也叫模式跳躍（mode hopping）。跳躍也可能慢慢停下來，變成第1節那樣穩定的部分崩塌。

**Reason one: the discriminator looks at one sample at a time.** `[Added]` The discriminator scores each image separately and does not know what the other images in the batch look like. Imagine the discriminator held fixed for a moment while the generator makes $D(G(\mathbf{z}))$ as large as possible. What is the best move? Find the point where the discriminator's score is highest and send **every** $\mathbf{z}$ there. The discriminator cannot see variety at all, so the generator has no reason to keep it. The discriminator then learns "that point is fake", and the generator jumps to the next high-scoring place; this is KP04's "you chase me, I chase you", jumping between modes, also called mode hopping. The hopping may also die down into a stable partial collapse, as in Section 1.

**理由二：衡量差距的方向。** 來源說，極小化 $D_{\text{KL}}(p_g\,\Vert\,p_{\text{data}})$ 是「尋找模式」（mode-seeking）的。KL 散度不對稱，兩個方向的懲罰不同 `[補充]`：

**Reason two: the direction of the gap measure.** The source says minimising $D_{\text{KL}}(p_g\,\Vert\,p_{\text{data}})$ is "mode-seeking". The KL divergence is not symmetric, and its two directions penalise different things `[Added]`:

- **正向** $D_{\text{KL}}(p_{\text{data}}\,\Vert\,p_g)=\mathbb{E}_{p_{\text{data}}}\big[\log\frac{p_{\text{data}}}{p_g}\big]$：在真資料存在、生成器卻幾乎不畫的地方，$p_g$ 很小、$\log$ 很大，罰很重。所以它逼生成器**蓋住所有**真資料。
- **反向** $D_{\text{KL}}(p_g\,\Vert\,p_{\text{data}})=\mathbb{E}_{p_g}\big[\log\frac{p_g}{p_{\text{data}}}\big]$：只在生成器有畫的地方算分。漏掉一個模式不會被罰；畫到沒有真資料的地方才被重罰。所以它讓生成器**守在一個峰上**（前提是生成器沒辦法同時畫好所有峰；若能完全畫對，兩個方向的 KL 都是 $0$）。

- **Forward** $D_{\text{KL}}(p_{\text{data}}\,\Vert\,p_g)=\mathbb{E}_{p_{\text{data}}}\big[\log\frac{p_{\text{data}}}{p_g}\big]$: where real data exist but the generator almost never draws, $p_g$ is tiny, the logarithm is large, and the penalty is heavy. So it forces the generator to **cover all** the real data.
- **Reverse** $D_{\text{KL}}(p_g\,\Vert\,p_{\text{data}})=\mathbb{E}_{p_g}\big[\log\frac{p_g}{p_{\text{data}}}\big]$: it only scores where the generator draws. Missing a mode is not penalised; drawing where there are no real data is penalised heavily. So it makes the generator **stay on one peak** (this assumes the generator cannot fit every peak; if it matches the data exactly, both directions are $0$).

例子 `[補充]`：目標是兩個峰 $p=\tfrac12\mathcal{N}(-2,0.5^2)+\tfrac12\mathcal{N}(2,0.5^2)$，我們只准用一個常態分布 $q$ 去配。極小化正向 KL 的答案是 $\mathcal{N}(0,\ 2.06^2)$：中心在兩峰中間，寬到兩峰都蓋住，正向 KL 是 $0.724$。極小化反向 KL 的答案是 $\mathcal{N}(2,\ 0.5^2)$：剛好坐在右邊那個峰上，反向 KL 是 $0.693=\log2$（坐在左峰也一樣）。反過來比，寬的那個的反向 KL 是 $2.097$，因為它把很多機率放在兩峰之間沒有資料的地方；窄的那個的正向 KL 高達 $15.31$，因為整個左峰都被漏掉。

Example `[Added]`: the target has two peaks, $p=\tfrac12\mathcal{N}(-2,0.5^2)+\tfrac12\mathcal{N}(2,0.5^2)$, and we may fit only one normal distribution $q$. Minimising the forward KL gives $\mathcal{N}(0,\ 2.06^2)$: centred between the peaks and wide enough to cover both, with forward KL $0.724$. Minimising the reverse KL gives $\mathcal{N}(2,\ 0.5^2)$: sitting exactly on the right peak, with reverse KL $0.693=\log2$ (the left peak does equally well). Compared the other way, the wide one has reverse KL $2.097$, because it puts much probability in the gap between the peaks where there are no data; the narrow one has forward KL as high as $15.31$, because the whole left peak is missed.

![一張折線圖，橫軸 x 從 -5.5 到 5.5，縱軸是密度 0 到 0.9。淡藍色填色是目標分布 p，兩個峰在 -2 和 +2，高約 0.4。橘色實線是極小化正向 KL 的 N(0, 2.06²)，又矮又寬，中心在 0，旁邊寫著「兩邊都蓋住，但把機率放在中間的空隙」。紅色虛線是極小化反向 KL 的 N(2, 0.5²)，高約 0.8，只疊在右邊的峰上，旁邊寫著「坐在一個峰上，忽略另一個」（A line chart with x from -5.5 to 5.5 and density from 0 to 0.9. Pale blue fill is the target p with two peaks at -2 and +2, each about 0.4 high. The solid orange line is N(0, 2.06²), which minimises the forward KL: low and wide, centred at 0, labelled covers both, but puts mass in the gap. The dashed red line is N(2, 0.5²), which minimises the reverse KL: about 0.8 high, sitting only on the right peak, labelled sits on one bump, ignores the other）](../assets/figures/lec13-kp06-kl-direction.png)

**圖說 `lec13-kp06-kl-direction`**〔功能：一目了然、精準度〕：先看淡藍色的兩個峰，那是目標。再看橘線：正向 KL 怕漏掉資料，寧可攤平蓋住兩峰，連中間沒資料的空隙也蓋到了。最後看紅色虛線：反向 KL 怕畫到沒資料的地方，乾脆只守一個峰。

**Caption `lec13-kp06-kl-direction`** [Function: seeing it at a glance, precision]: first look at the two pale blue peaks, the target. Then the orange line: the forward KL fears missing data, so it would rather spread out and cover both peaks, even the empty gap between them. Finally the dashed red line: the reverse KL fears drawing where there are no data, so it simply holds one peak.

`[補充]` GAN 並不是直接在極小化反向 KL。KP03 說過，判別器最優、用原本的損失時，生成器縮小的是 JS 散度；改用非飽和損失、而且判別器同樣是最優時，可以證明它的梯度方向等於在縮小 $D_{\text{KL}}(p_g\,\Vert\,p_{\text{data}})-2\,\mathrm{JSD}$（Arjovsky 與 Bottou，2017），裡面正好有反向 KL；而 $-2\,\mathrm{JSD}$ 那一項是往反方向推，也讓訓練更不穩定。所以「尋找模式」的傾向是一個合理的解釋，但不是唯一原因；理由一的「判別器看不到多樣性」同樣重要。

`[Added]` A GAN does not literally minimise the reverse KL. KP03 showed that with an optimal discriminator and the original loss, the generator shrinks the JS divergence; with the non-saturating loss, and again an optimal discriminator, one can show its gradient direction is that of shrinking $D_{\text{KL}}(p_g\,\Vert\,p_{\text{data}})-2\,\mathrm{JSD}$ (Arjovsky and Bottou, 2017), which contains exactly the reverse KL; the $-2\,\mathrm{JSD}$ term pushes the other way, adding to the instability. So the "mode-seeking" tendency is a reasonable explanation, but not the only cause; reason one, "the discriminator cannot see variety", matters just as much.

常見錯誤：以為把生成器的學習率調小就能避免崩塌。來源說生成器更新過快時容易崩塌；但在上面的實驗裡，生成器的學習率（$10^{-4}$）已經是判別器（$10^{-3}$）的十分之一，仍然停在 $4$ 個模式 `[補充]`。學習率會影響，但問題的根源是判別器看不到多樣性。

Common mistake: thinking that lowering the generator's learning rate avoids collapse. The source says collapse happens easily when the generator updates too fast; but in the experiment above the generator's learning rate ($10^{-4}$) is already a tenth of the discriminator's ($10^{-3}$), and it still settles on $4$ modes `[Added]`. The learning rate has an effect, but the root of the problem is that the discriminator cannot see variety.

### 3. 對策一：小批次判別（Remedy one: minibatch discrimination）

既然問題是判別器一次只看一張，那就讓它**一次看一整批**。Salimans 等人（2016）的小批次判別（minibatch discrimination）在判別器裡加一層：對每一個樣本，算它和同一批其他樣本有多接近，把這個「接近程度」當成額外的特徵交給判別器。生成器若把很多樣本擠在一起，判別器就能發現「這一批太像了，是假的」，生成器只好把樣本分散開。

Since the problem is that the discriminator sees one sample at a time, let it **see a whole batch at once**. The minibatch discrimination of Salimans et al. (2016) adds a layer to the discriminator: for each sample it computes how close the sample is to the others in the same batch and hands this "closeness" to the discriminator as an extra feature. If the generator crowds many samples together, the discriminator can notice "this batch is too alike, it is fake", and the generator has to spread its samples out.

用數字看這個想法 `[補充]`。數一數：一批 $n=64$ 個樣本裡，每個樣本平均有幾個「近鄰」，也就是距離小於 $r=0.3$ 的其他樣本。真資料平均分在 $8$ 團，每團約 $8$ 個樣本，每個樣本約有 $63/8\approx7.88$ 個近鄰；若生成器只畫 $4$ 團，每團約 $16$ 個，近鄰約 $63/4\approx15.75$ 個，多了一倍。實際算一批真資料是 $7.38$（這一批各團人數不完全相等）；上面那個停在 $4$ 團的標準 GAN（種子 1）是 $9.12$，因為它有一部分樣本散在團與團之間，所以沒有到 $15.75$，但仍然明顯偏多。

Let us see the idea in numbers `[Added]`. Count how many "close neighbours" each sample has on average in a batch of $n=64$, that is, other samples within distance $r=0.3$. Real data split evenly into $8$ clusters of about $8$ samples, so each sample has about $63/8\approx7.88$ close neighbours; a generator drawing only $4$ clusters puts about $16$ in each, giving about $63/4\approx15.75$, twice as many. Computed on an actual real batch it is $7.38$ (the clusters in this batch are not exactly equal in size); for the standard GAN above that settled on $4$ clusters (seed 1) it is $9.12$: some of its samples are scattered between clusters, so it does not reach $15.75$, but it is still clearly higher.

`[補充]` 反過來說，一批 $64$ 個真樣本漏掉某一團的機率，最多只有 $8\times(7/8)^{64}\approx0.0016$。所以「一批裡只出現少數幾團」是假資料的強烈訊號。小批次判別就是把這種「整批的統計」交給判別器，讓它自己學會用。本講義沒有在這個小實驗上實作小批次判別的訓練，上面的近鄰數只是說明它利用的訊號。

`[Added]` Put the other way, the probability that a batch of $64$ real samples misses some cluster is at most $8\times(7/8)^{64}\approx0.0016$. So "only a few clusters appear in the batch" is strong evidence of fake data. Minibatch discrimination hands this kind of "whole-batch statistic" to the discriminator and lets it learn to use it. This note does not implement minibatch-discrimination training on the small experiment; the close-neighbour count above only illustrates the signal it exploits.

### 4. 對策二：特徵匹配（Remedy two: feature matching）

同樣來自 Salimans 等人（2016）的特徵匹配（feature matching）換了生成器的目標：不要再直接追求「騙過判別器的最後一個分數」，而是要求生成樣本在判別器**中間層**的平均特徵，和真資料的一樣：

Feature matching, also from Salimans et al. (2016), changes the generator's goal: instead of chasing "fool the discriminator's final score" directly, it asks that the average features of generated samples at an **intermediate layer** of the discriminator equal those of real data:

$$\mathcal{L}_{\text{FM}}=\Big\lVert\,\mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}\big[\mathbf{f}(\mathbf{x})\big]-\mathbb{E}_{\mathbf{z}\sim p_z}\big[\mathbf{f}(G(\mathbf{z}))\big]\Big\rVert_2^2$$

$\mathbf{f}(\mathbf{x})$ 是判別器某個中間層的輸出（一個向量），判別器照常訓練。重點在「平均」：這個目標看的是**一整批**生成樣本的統計量，而不是單一張圖。把所有樣本擠到同一點，平均特徵通常對不上，所以生成器不能再只畫一種。

$\mathbf{f}(\mathbf{x})$ is the output (a vector) of an intermediate layer of the discriminator, and the discriminator is trained as usual. The key is the "average": the goal looks at a statistic of a **whole batch** of generated samples, not at a single image. Crowding every sample onto one point usually fails to match the average features, so the generator can no longer draw just one kind.

在第1節的實驗裡，只把生成器的損失換成 $\mathcal{L}_{\text{FM}}$（$\mathbf{f}$ 取判別器最後一個隱藏層的 $128$ 維輸出），其他完全不變 `[補充]`：$5$ 個種子最後涵蓋 $4,8,8,8,6$ 個模式，有 $3$ 次涵蓋全部 $8$ 個；標準 GAN 是 $0$ 次。

In the experiment of Section 1, we only replace the generator's loss with $\mathcal{L}_{\text{FM}}$ ($\mathbf{f}$ is the $128$-dimensional output of the discriminator's last hidden layer), changing nothing else `[Added]`: the $5$ seeds end up covering $4,8,8,8,6$ modes, all $8$ in $3$ runs; the standard GAN did so in $0$.

![一張折線圖，橫軸是訓練步數 0 到 5000，縱軸是涵蓋的模式數 0 到 8。五條紅線是標準 GAN 的五個種子，大約在第 1000 步前升到 4，之後四條停在 4，一條在第 1750 步爬到 6。五條綠線是特徵匹配，起步比較慢，約第 1000 到 1250 步才開始上升，最後三條到 8、一條到 6、一條停在 4。標題說明線條上下稍微錯開以免重疊（A line chart with training steps 0 to 5000 horizontally and modes covered, 0 to 8, vertically. Five red lines are the standard GAN's five seeds: they rise to 4 before about step 1000, then four stay at 4 and one climbs to 6 by step 1750. Five green lines are feature matching: they start more slowly, beginning to rise around steps 1000 to 1250, and end with three at 8, one at 6 and one at 4. The title notes the lines are nudged apart vertically so they do not overlap）](../assets/figures/lec13-kp06-coverage.png)

**圖說 `lec13-kp06-coverage`**〔功能：精準度、一目了然〕：先看紅線：標準 GAN 四條很快就停在 $4$ 個，另一條到第 $1750$ 步停在 $6$，之後都不動。再看綠線：特徵匹配起步慢，但多數最後爬到 $8$。線條上下稍微錯開，只是為了不互相遮住。

**Caption `lec13-kp06-coverage`** [Function: precision, seeing it at a glance]: first look at the red lines: four standard-GAN runs quickly settle at $4$, and the fifth settles at $6$ by step $1750$; none moves after that. Then the green lines: feature matching starts slowly, but most climb to $8$ in the end. The lines are nudged apart vertically only so they do not hide each other.

特徵匹配也有代價 `[補充]`。回到圖 `lec13-kp06-ring-samples` 的右圖：八團都有了，但點比較散。在 $2000$ 個樣本裡，落在某團中心 $0.15$ 以內的比例，標準 GAN 的五個種子是 $0.79$ 到 $0.85$，特徵匹配只有 $0.56$ 到 $0.63$。換成影像，可能就是樣本比較多樣、但比較不清楚（Salimans 等人也提到特徵匹配的影像品質較弱）。而且它也不保證成功：五次裡仍有兩次沒涵蓋全部。

Feature matching has a cost too `[Added]`. Go back to the right panel of `lec13-kp06-ring-samples`: all eight clusters are there, but the points are more scattered. Of the $2000$ samples, the share within $0.15$ of some cluster centre is $0.79$ to $0.85$ across the standard GAN's five seeds, but only $0.56$ to $0.63$ for feature matching. For images, this may mean more varied but less sharp samples (Salimans et al. also report weaker visual quality with feature matching). Nor is success guaranteed: two of the five runs still miss some clusters.

常見錯誤：以為只要對上**任何**特徵的平均就夠了。如果特徵只是 $\mathbf{f}(\mathbf{x})=\mathbf{x}$ 本身，八團的平均是原點 $(0,0)$；生成器把所有樣本都放在原點，平均也是 $(0,0)$，$\mathcal{L}_{\text{FM}}=0$，卻是最嚴重的崩塌 `[補充]`。特徵匹配之所以有用，是因為判別器的中間層特徵會一直被訓練成「能分辨真假」的豐富特徵。

Common mistake: thinking that matching the average of **any** feature is enough. If the feature is just $\mathbf{f}(\mathbf{x})=\mathbf{x}$, the average of the eight clusters is the origin $(0,0)$; a generator putting every sample at the origin also has average $(0,0)$ and $\mathcal{L}_{\text{FM}}=0$, yet that is the worst collapse of all `[Added]`. Feature matching works because the discriminator's intermediate features keep being trained into rich features that can tell real from fake.

### 5. 對策三：展開 GAN 與其他做法（Remedy three: the unrolled GAN, and other approaches）

第2節的理由一說，生成器只看「判別器現在的樣子」，所以會一路跳到下一個高分處。Metz 等人（2017）的展開 GAN（unrolled GAN）讓生成器更新時**往前看**：先在計算中模擬判別器接下來 $k$ 步的更新，再用「$k$ 步之後的判別器」來算生成器的梯度，而且梯度會穿過這 $k$ 步一起計算 `[補充]`。生成器因此看得到「我若全擠到這一點，判別器幾步之後就會壓低這裡」，跳到單一模式就不再划算。代價是每一步要多算 $k$ 次判別器的更新，以及穿過它們的梯度，計算量至少約變成 $k$ 倍；判別器本身仍照常更新 `[補充]`。

Reason one in Section 2 said that the generator only sees "what the discriminator is like now", so it keeps jumping to the next high-scoring spot. The unrolled GAN of Metz et al. (2017) makes the generator **look ahead** when it updates: it first simulates the discriminator's next $k$ updates in the computation, then computes the generator's gradient using "the discriminator after $k$ steps", with the gradient passing back through those $k$ steps `[Added]`. The generator can thus see "if I all crowd onto this point, a few steps later the discriminator will push this spot down", so jumping to a single mode no longer pays. The cost is $k$ extra discriminator updates per step, plus the gradient through them, at least about $k$ times the computation; the discriminator itself is still updated as usual `[Added]`.

這和 KP04 的觀察一致：只看對手當下位置去追，會一直繞圈；能預料對手下一步，追逐就比較穩定。本講義沒有在第1節的實驗上重現展開 GAN，所以不放它的實驗數字。

This agrees with KP04's observation: chasing the opponent's current position keeps circling; anticipating the opponent's next move makes the chase more stable. This note does not reproduce the unrolled GAN on the Section 1 experiment, so it gives no experimental numbers for it.

來源的教科書圖目錄還提到多生成器混合網路（Mixture GAN，MGAN）：用好幾個生成器，各自負責一部分模式 `[補充]`。KP07 會介紹另一條路：換一把衡量真假差距的尺，讓訓練本身穩定，模式崩塌也因此減輕。

The source's list of textbook figures also mentions the mixture GAN (MGAN): several generators, each responsible for some of the modes `[Added]`. KP07 introduces another route: change the ruler that measures the gap between real and fake so that training itself is stable, which also reduces mode collapse.

常見錯誤：以為這些對策能保證涵蓋所有模式。它們只是讓生成器**比較有理由**保持多樣；實驗裡特徵匹配五次仍有兩次漏掉。實務上要一直檢查多樣性，不能只看損失。

Common mistake: thinking these remedies guarantee that every mode is covered. They only give the generator **more reason** to stay varied; in the experiment, feature matching still missed clusters in two of five runs. In practice you must keep checking variety, not just the loss.

### 本單元重點整理（KP06 summary）

**一句話重點**：模式崩塌是生成器只畫少數幾個模式；它來自判別器一次只看一張、看不到多樣性，加上非飽和損失帶有反向 KL「在畫不全時守住一個峰」的傾向；對策是讓生成器的目標或判別器的輸入包含整批的統計（小批次判別、特徵匹配），或讓生成器往前看判別器幾步（展開 GAN），但都不保證一定成功。

**One-sentence summary**: mode collapse is a generator drawing only a few modes; it comes from the discriminator seeing one sample at a time and so not seeing variety, together with the non-saturating loss's reverse-KL tendency to "hold one peak when it cannot fit them all"; the remedies put whole-batch statistics into the generator's goal or the discriminator's input (minibatch discrimination, feature matching), or let the generator look a few discriminator steps ahead (the unrolled GAN), but none guarantees success.

**白話說明**：一個學生發現老師的考卷某一題型最容易拿分，就只練那一題型，其他全部放棄；每次交的答案都對，但只會一種。小批次判別像老師一次看整疊考卷，發現「怎麼全是同一題型」；特徵匹配像老師要求整疊考卷的題型比例要和課本一樣；展開 GAN 像學生先想好「我若只練這一題型，老師下週就會改出題方式」。

**Plain explanation**: a student finds that one type of question on the teacher's exams scores most easily, so practises only that type and gives up on the rest; every answer handed in is right, but only of one kind. Minibatch discrimination is like the teacher looking at the whole stack at once and noticing "why is it all the same type"; feature matching is like the teacher requiring the mix of question types in the stack to match the textbook; the unrolled GAN is like the student first thinking "if I practise only this type, next week the teacher will change the questions".

**正式定義**：正向 KL $D_{\text{KL}}(p_{\text{data}}\,\Vert\,p_g)$ 涵蓋所有模式，反向 KL $D_{\text{KL}}(p_g\,\Vert\,p_{\text{data}})$ 在模型不夠表達時傾向尋找模式；特徵匹配 $\mathcal{L}_{\text{FM}}=\lVert\mathbb{E}_{p_{\text{data}}}[\mathbf{f}(\mathbf{x})]-\mathbb{E}_{p_z}[\mathbf{f}(G(\mathbf{z}))]\rVert_2^2$；展開 GAN 用 $k$ 步更新後的判別器計算生成器的梯度。

**Formal definition**: the forward KL $D_{\text{KL}}(p_{\text{data}}\,\Vert\,p_g)$ covers all modes, and the reverse KL $D_{\text{KL}}(p_g\,\Vert\,p_{\text{data}})$ tends to seek modes when the model cannot fit them all; feature matching $\mathcal{L}_{\text{FM}}=\lVert\mathbb{E}_{p_{\text{data}}}[\mathbf{f}(\mathbf{x})]-\mathbb{E}_{p_z}[\mathbf{f}(G(\mathbf{z}))]\rVert_2^2$; the unrolled GAN computes the generator's gradient with the discriminator after $k$ updates.

**例子**：八團環的實驗，標準 GAN 五個種子涵蓋 $6,4,4,4,4$ 個模式，特徵匹配 $4,8,8,8,6$ 個；落在團內的比例 $0.79$–$0.85$ 對 $0.56$–$0.63$。一批 $64$ 個的近鄰數：真資料 $7.38$，崩塌的生成器 $9.12$。雙峰配一個常態：正向 KL 得 $\mathcal{N}(0,2.06^2)$，反向 KL 得 $\mathcal{N}(2,0.5^2)$，值 $\log2=0.693$。

**Example**: in the eight-cluster ring experiment, the standard GAN's five seeds cover $6,4,4,4,4$ modes and feature matching $4,8,8,8,6$; the share of samples inside a cluster is $0.79$–$0.85$ versus $0.56$–$0.63$. Close neighbours in a batch of $64$: $7.38$ for real data, $9.12$ for the collapsed generator. Fitting one normal to two peaks: the forward KL gives $\mathcal{N}(0,2.06^2)$, the reverse KL gives $\mathcal{N}(2,0.5^2)$ with value $\log2=0.693$.

**常見迷思**：

- 「每張都逼真就是訓練成功。」錯，還要看整體的多樣性。
- 「把生成器學習率調小就不會崩塌。」錯，實驗裡生成器學習率只有判別器的十分之一，仍然崩塌。
- 「特徵匹配只要對上任何特徵的平均就好。」錯，只對上 $\mathbf{x}$ 的平均時，全擠在原點也是 $0$ 損失。
- 「用了這些對策就一定涵蓋全部模式。」錯，它們只是讓多樣性比較有利。

**Common misconceptions**:

- "If every sample is realistic, training succeeded." Wrong: the variety as a whole matters too.
- "A smaller generator learning rate prevents collapse." Wrong: in the experiment the generator's learning rate was a tenth of the discriminator's, and it still collapsed.
- "Feature matching only needs the average of any feature to match." Wrong: matching only the average of $\mathbf{x}$, crowding everything at the origin also gives zero loss.
- "With these remedies every mode is covered." Wrong: they only make variety more favourable.

**圖**：本單元有三張圖。

- `lec13-kp06-ring-samples`：真資料八團、標準 GAN 只畫四團、特徵匹配畫到八團但比較散。
- `lec13-kp06-kl-direction`：雙峰目標下，正向 KL 與反向 KL 的最佳常態分布。
- `lec13-kp06-coverage`：五個種子的涵蓋模式數隨訓練步數變化。

**Figures** — this unit has three figures.

- `lec13-kp06-ring-samples`: the real eight clusters, the standard GAN drawing only four, and feature matching reaching eight but more scattered.
- `lec13-kp06-kl-direction`: the best normal fit under the forward and the reverse KL for a two-peak target.
- `lec13-kp06-coverage`: the number of covered modes over training steps for five seeds.

### 小練習（Practice）

1. 一批 $64$ 個樣本。若真資料平均分在 $8$ 團，每個樣本大約有幾個同團的其他樣本？若生成器只畫 $2$ 團呢？
2. 八團環的每一團中心離原點都是 $2$，團的標準差 $0.05$。若特徵是 $\mathbf{f}(\mathbf{x})=\mathbf{x}$，把所有樣本放在原點的生成器，特徵匹配損失是多少？若特徵改成 $\mathbf{f}(\mathbf{x})=\big(\mathbf{x},\ \lVert\mathbf{x}\rVert^2\big)$ 呢？（提示：真資料的 $\mathbb{E}\lVert\mathbf{x}\rVert^2=4+2\times0.05^2$。）
3. 第2節的例子裡，為什麼反向 KL 的最佳解坐在**一個**峰上時，值幾乎剛好是 $\log2$？（提示：在右峰附近，$p(x)\approx\tfrac12q(x)$。）

1. A batch has $64$ samples. If the real data split evenly into $8$ clusters, about how many other samples from the same cluster does each sample have? And if the generator draws only $2$ clusters?
2. Every cluster centre of the eight-cluster ring is at distance $2$ from the origin, with cluster standard deviation $0.05$. With the feature $\mathbf{f}(\mathbf{x})=\mathbf{x}$, what is the feature-matching loss of a generator that puts every sample at the origin? What if the feature is $\mathbf{f}(\mathbf{x})=\big(\mathbf{x},\ \lVert\mathbf{x}\rVert^2\big)$? (Hint: for the real data, $\mathbb{E}\lVert\mathbf{x}\rVert^2=4+2\times0.05^2$.)
3. In the example of Section 2, why is the reverse KL almost exactly $\log2$ when its best fit sits on **one** peak? (Hint: near the right peak, $p(x)\approx\tfrac12q(x)$.)

**第 1 題解答** `[補充]`：$8$ 團時每團約 $8$ 個，同團的其他樣本約 $63/8\approx7.9$ 個；$2$ 團時每團約 $32$ 個，約 $63/2=31.5$ 個，多了四倍。批次裡「太多近鄰」就是崩塌的訊號。

**Solution to question 1** `[Added]`: with $8$ clusters there are about $8$ per cluster, so about $63/8\approx7.9$ others from the same cluster; with $2$ clusters about $32$ per cluster, so about $63/2=31.5$, four times as many. "Too many close neighbours" in a batch is a sign of collapse.

**第 2 題解答** `[補充]`：真資料的 $\mathbb{E}[\mathbf{x}]=(0,0)$，原點的生成器也是 $(0,0)$，損失 $0$，完全看不出崩塌。加上 $\lVert\mathbf{x}\rVert^2$ 這個特徵後，真資料是 $4.005$，原點的生成器是 $0$，損失 $(4.005-0)^2\approx16.04$，這時就抓得到了。特徵越豐富，越難被「投機」騙過。

**Solution to question 2** `[Added]`: for the real data $\mathbb{E}[\mathbf{x}]=(0,0)$, and the origin generator also gives $(0,0)$, so the loss is $0$ and the collapse is invisible. Adding the feature $\lVert\mathbf{x}\rVert^2$, the real data give $4.005$ and the origin generator $0$, so the loss is $(4.005-0)^2\approx16.04$, and now it is caught. The richer the features, the harder they are to fool by "gaming".

**第 3 題解答** `[補充]`：$q=\mathcal{N}(2,0.5^2)$ 恰好等於 $p$ 的右半個成分，在右峰附近左峰幾乎是 $0$，所以 $p(x)\approx\tfrac12q(x)$，$\log\frac{q(x)}{p(x)}\approx\log2$。反向 KL 只在 $q$ 有機率的地方取平均，所以整體幾乎就是 $\log2\approx0.693$（精確值 $0.69305$，因為左峰的尾巴並不完全是 $0$）。漏掉的左峰完全沒被算進去。

**Solution to question 3** `[Added]`: $q=\mathcal{N}(2,0.5^2)$ equals exactly the right component of $p$, and near the right peak the left one is almost $0$, so $p(x)\approx\tfrac12q(x)$ and $\log\frac{q(x)}{p(x)}\approx\log2$. The reverse KL averages only where $q$ has probability, so the total is almost exactly $\log2\approx0.693$ (precisely $0.69305$, because the left peak's tail is not exactly $0$). The missed left peak is not counted at all.

### 小測驗（Quiz，來源原題）

來源 KP-7 有一題自我檢測，保留原題與全部選項；來源頁面附有解析，下面依它說明。

The source's KP-7 has one self-check question, kept here with all options; the source page gives an explanation, which the note follows below.

1. 在 GAN 訓練中，模式崩塌 (Mode Collapse) 現象指的是什麼？
   - (A) 判別器的梯度直接變為零且網絡崩潰
   - (B) 生成器忽視真實數據的多樣性，只反覆生成少數幾種（甚至單一）高信心模式的樣本
   - (C) 模型記憶體空間爆滿導致程式中斷
   - (D) 生成器輸出的圖像解析度縮小為 $1\times1$ 像素

1. In GAN training, what does the phenomenon of mode collapse refer to?
   - (A) The discriminator's gradient drops straight to zero and the network breaks down
   - (B) The generator ignores the variety of the real data and repeatedly generates samples from only a few (or even one) high-confidence modes
   - (C) The model runs out of memory and the program stops
   - (D) The generator's output images shrink to $1\times1$ pixel

**解析**：選 (B)。來源的理由：生成器找到某個能騙過判別器的局部樣本後，就不再學習整個資料分布的多樣性，只重複產出一種或少數幾種模式（見第1節）。`[補充]` 其他選項：(A) 比較接近 KP02、KP03 的梯度消失（但那裡是生成器拿不到梯度），是另一種問題；(C)、(D) 是程式或架構的問題，和模式無關。

**Explanation**: the answer is B. The source's reason: once the generator finds some local sample that fools the discriminator, it stops learning the variety of the whole data distribution and only repeats one or a few modes (see Section 1). `[Added]` The other options: A is closer to the vanishing gradient of KP02 and KP03 (though there it is the generator that gets no gradient), a different problem; C and D are program or architecture problems, unrelated to modes.

### 留給下一單元的問題（The question for the next unit）

KP03 說兩個分布離很遠時，JS 散度停在 $\log2$，生成器拿不到梯度；這一節又看到，判別器的評分方式會讓生成器只守一個峰。兩個問題都和「用什麼來衡量真假的差距」有關。有沒有一把尺，在兩個分布不重疊時仍能說出「還差多遠」，而且讓訓練更穩定？下一個知識點（本週 KP07）介紹 Wasserstein 距離與 WGAN，以及其他幾種換掉損失的做法。

KP03 said that when the two distributions are far apart, the JS divergence sits at $\log2$ and the generator gets no gradient; this note saw that the way the discriminator scores can make the generator hold just one peak. Both problems concern "what we use to measure the gap between real and fake". Is there a ruler that still says "how far apart" when the distributions do not overlap, and makes training more stable? The next knowledge point (this week's KP07) introduces the Wasserstein distance and WGAN, and several other ways of changing the loss.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第14章「產生式對抗網絡」的 KP-7「模式崩塌（Mode Collapse）、Minibatch Discrimination 與 Unrolled GAN」與 KP-12「特徵匹配與 InfoGAN 互資訊解耦」的特徵匹配部分（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch14-gan/index.html ，讀取日期 2026-11-15）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from Chapter 14, "Generative adversarial networks", of CCC-LNS IDL_115: KP-7, "Mode collapse, minibatch discrimination and the unrolled GAN", and the feature-matching part of KP-12, "Feature matching and InfoGAN mutual-information disentangling" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch14-gan/index.html , accessed 2026-11-15). Content not in the source is marked `[Added]` where it appears.

- 來源說模式崩塌的數學根源是極小化反向 KL $D_{\text{KL}}(p_g\,\Vert\,p_{\text{data}})$。反向 KL 確實是尋找模式的，但 GAN 不是直接極小化它；本講義第2節說明它和非飽和損失的關係，並補上「判別器一次只看一張」這個同樣重要的原因。
- 來源說「生成器更新過快時」容易崩塌，這和第2節理由一一致（生成器對著還沒跟上的判別器找最好答案）；但本講義的實驗顯示，只把生成器學習率調小（判別器的十分之一）並不足以避免崩塌。
- 來源的互動展示是理想化的：標準 GAN「全部凝固為數字 1」、小批次判別後「均勻產出 0 到 9」。實際上常見的是部分崩塌，對策也不保證成功；本講義用實驗數字呈現（五次中三次涵蓋全部）。
- 來源說小批次判別讓判別器「立刻識別並給予極低分懲罰」，說得太絕對；它提供判別器一個可以學著使用的整批統計，本講義用近鄰數說明，並註明沒有實作它的訓練。
- 特徵匹配原在來源 KP-12，本講義移到這裡和其他對策一起講，並補上它的代價（樣本比較散）與限制（特徵太簡單時會被騙過）。
- 來源用粗斜體 `\boldsymbol` 表示向量，並把 KL 散度寫成 $KLD$；本講義改用 `\mathbf` 與第12週的 $D_{\text{KL}}$。
- 本講義補上：模式、完全與部分崩塌、模式跳躍的定義；八團環的實驗與五個種子的結果；判別器固定時生成器的最好答案；正向與反向 KL 的比較與雙峰數字例子；近鄰數與漏團機率；特徵匹配的實驗、代價與限制；展開 GAN 的計算方式與代價；MGAN 的一句說明；三題小練習與解答。

- The source says the mathematical root of mode collapse is minimising the reverse KL $D_{\text{KL}}(p_g\,\Vert\,p_{\text{data}})$. The reverse KL is indeed mode-seeking, but a GAN does not minimise it directly; Section 2 explains its relation to the non-saturating loss and adds the equally important cause "the discriminator sees one sample at a time".
- The source says collapse happens easily "when the generator updates too fast", which agrees with reason one in Section 2 (the generator finds its best answer against a discriminator that has not caught up); but this note's experiment shows that merely making the generator's learning rate smaller (a tenth of the discriminator's) is not enough to avoid collapse.
- The source's interactive demo is idealised: the standard GAN "freezes entirely into the digit 1" and minibatch discrimination "produces 0 to 9 evenly". In practice partial collapse is common and the remedies do not guarantee success; this note shows experimental numbers (all modes covered in three of five runs).
- The source says minibatch discrimination lets the discriminator "immediately recognise and give a very low score", which is too absolute; it gives the discriminator a whole-batch statistic it can learn to use. This note illustrates it with the close-neighbour count and states that its training was not implemented.
- Feature matching was in the source's KP-12; this note moves it here with the other remedies and adds its cost (more scattered samples) and limitation (it can be fooled when the features are too simple).
- The source writes vectors with bold italic `\boldsymbol` and the KL divergence as $KLD$; this note uses `\mathbf` and Week 12's $D_{\text{KL}}$.
- This note adds: definitions of mode, full and partial collapse, and mode hopping; the eight-cluster ring experiment with five seeds; the generator's best answer against a fixed discriminator; the comparison of forward and reverse KL with a two-peak numerical example; the close-neighbour count and the probability of missing a cluster; the feature-matching experiment, its cost and its limitation; how the unrolled GAN computes and what it costs; one sentence on MGAN; and three practice questions with solutions.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 模式 | mode |
| 模式崩塌 | mode collapse |
| 模式遺漏（部分崩塌） | mode dropping |
| 模式跳躍 | mode hopping |
| 尋找模式 | mode-seeking |
| 涵蓋模式 | mode-covering |
| 正向 KL 散度 | forward KL divergence |
| 反向 KL 散度 | reverse KL divergence |
| 小批次判別 | minibatch discrimination |
| 近鄰 | close neighbour |
| 特徵匹配 | feature matching |
| 中間層特徵 | intermediate-layer features |
| 展開 GAN | unrolled GAN |
| 多生成器混合網路 | mixture GAN (MGAN) |
| 非飽和損失（KP02） | non-saturating loss |
| Wasserstein 距離（KP07） | Wasserstein distance |
