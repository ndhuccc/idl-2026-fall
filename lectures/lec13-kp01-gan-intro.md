# 第13週 KP01：GAN 導論，請一個判別器來當裁判（Introduction to GANs: hire a discriminator as the judge）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第13週「深度生成模型（二）：生成對抗網路」的第一個知識點（KP01）。第12週的 VAE 已經能生成新資料，但影像偏模糊：它用平方誤差量「畫得像不像」，這把尺在拿不準時偏好平均（第12週 KP07）。這一節換一個想法：不寫出這把尺，改請另一個神經網路來判斷「這張看起來真不真」。

This is the first knowledge point (KP01) of Week 13, "Deep generative models (2): generative adversarial networks". The VAE of Week 12 can already generate new data, but its images tend to be blurry: it measures "how alike" with squared error, a ruler that prefers the average when it is unsure (Week 12 KP07). This note tries another idea: instead of writing such a ruler down, hire another neural network to judge "does this look real?".

先備知識是第2週的多層感知器與 sigmoid 函數、第3週的反向傳播與梯度下降，以及第12週 KP02 的潛在變數與 KP07 的模糊問題。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第14章 KP-1 與 KP-2 的比喻改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec13-kp01-figures.py` 算出，可以自己重算。

The prerequisites are the multilayer perceptron and the sigmoid function of Week 2, backpropagation and gradient descent of Week 3, and from Week 12 the latent variables of KP02 and the blur problem of KP07. Every term and symbol is defined when it first appears. This note is rewritten from KP-1 of Chapter 14 of the course's web material CCC-LNS, together with the analogy of its KP-2, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec13-kp01-figures.py`, so you can recompute it yourself.

## KP01 GAN 導論，請一個判別器來當裁判（Introduction to GANs: hire a discriminator as the judge）

### 學習目標（Learning objectives）

讀完之後，你能說出 GAN 想解決的問題：生成品質需要一個可以算梯度的評分者。你能畫出 GAN 的兩個網路，生成器與判別器，寫出各自的輸入與輸出，並說出各自想達成什麼。你能用小數字算出判別器對一個點的評分。你能看懂一個小 GAN 的訓練過程：生成的分布怎麼一步步移向真實資料，判別器的評分為什麼最後停在 $0.5$ 附近。你也能說出 GAN 和 VAE 在「怎麼學」與「學完能做什麼」上的差別。

After reading, you can state the problem a GAN addresses: generation quality needs a judge whose score has a gradient. You can draw the two networks of a GAN, the generator and the discriminator, write the input and output of each, and say what each wants. You can compute the discriminator's score for one point with small numbers. You can read the training of a small GAN: how the generated distribution moves step by step towards the real data, and why the discriminator's scores end up near $0.5$. You can also explain how a GAN differs from a VAE in how it learns and in what it can do after training.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{x}$ | 一筆資料（真的或生成的） | one example (real or generated) |
| $p_{\text{data}}$ | 真實資料的分布 | distribution of the real data |
| $\mathbf{z}\in\mathbb{R}^k$ | 雜訊向量，也就是潛在變數 | noise vector, that is, the latent variable |
| $p_z$ | 雜訊的分布，常用標準常態 | distribution of the noise, usually the standard normal |
| $k$ | 雜訊向量的長度 | length of the noise vector |
| $G(\cdot)$ | 生成器：把 $\mathbf{z}$ 變成一筆資料 | generator: turns $\mathbf{z}$ into an example |
| $\hat{\mathbf{x}}$ | 生成的一筆資料，$\hat{\mathbf{x}}=G(\mathbf{z})$ | one generated example, $\hat{\mathbf{x}}=G(\mathbf{z})$ |
| $\mathcal{X}$ | 所有可能資料組成的空間 | the space of all possible examples |
| $h(\cdot)$ | 判別器在 sigmoid 之前算出的實數（logit） | the real number the discriminator computes before the sigmoid (logit) |
| $p_g$ | 生成器輸出的分布 | distribution of the generator's outputs |
| $D(\cdot)$ | 判別器：輸出「這筆是真資料」的機率 | discriminator: outputs the probability "this example is real" |
| $\sigma(\cdot)$ | sigmoid 函數 $\sigma(t)=1/(1+e^{-t})$（第2週） | sigmoid function $\sigma(t)=1/(1+e^{-t})$ (Week 2) |
| $a,\ b$ | 第3節小例子裡生成器的兩個參數 | the generator's two parameters in the small example of Section 3 |
| $w_0,\ w_1,\ w_2$ | 小例子裡判別器的三個權重 | the discriminator's three weights in the small example |
| $\mathcal{N}(\mu,s^2)$ | 平均 $\mu$、標準差 $s$ 的常態分布 | normal distribution with mean $\mu$ and standard deviation $s$ |

### 1. 生成需要一個能算梯度的裁判（Generation needs a judge with a gradient）

**生成模型**（generative model）的目標，是學會真實資料的分布 $p_{\text{data}}$，再從中抽出新的樣本，例如沒見過的人臉、畫作或聲音。訓練它需要一個分數，告訴模型現在畫得好不好；反向傳播（第3週）要從這個分數算出梯度，才知道權重該往哪邊調。

A **generative model** aims to learn the distribution $p_{\text{data}}$ of the real data and then draw new samples from it, such as unseen faces, paintings or sounds. Training it needs a score that tells the model how well it is doing now; backpropagation (Week 3) must compute a gradient from that score to know which way to move the weights.

第12週的 VAE 用重建項當分數，對常態解碼器來說就是平方誤差。KP07 看過這把尺的毛病：同一個編碼可能對應好幾張清楚的影像時，平方誤差最小的答案是它們的平均，所以影像變模糊。最理想的分數其實是「人看了覺得像不像真的」。來源指出困難在這裡：人的主觀判斷沒有導數，沒辦法拿來反向傳播。

The VAE of Week 12 uses the reconstruction term as its score, which for a normal decoder is squared error. KP07 saw this ruler's flaw: when one code may correspond to several sharp images, the answer with the least squared error is their average, so the image blurs. The ideal score is really "does a person find it real?". The source points out the difficulty: human judgment has no derivative, so it cannot be used for backpropagation.

2014 年 Goodfellow 等人提出的**生成對抗網路**（generative adversarial network，GAN）想出了辦法：再訓練一個神經網路當裁判。這個裁判叫**判別器**（discriminator），它也是可微分的函數，所以「裁判覺得像不像」可以算梯度，傳回給負責生成的網路。

The **generative adversarial network** (GAN), proposed by Goodfellow and colleagues in 2014, found a way: train another neural network as the judge. This judge is called the **discriminator**; it is a differentiable function too, so "how real the judge finds it" has a gradient that can flow back to the network doing the generating.

常見錯誤：以為 GAN 的判別器是事先訓練好、固定不動的評分器。它和生成器一起從頭訓練，而且一直在變；第3節會看到，裁判的標準跟著生成器的進步一起提高。

Common mistake: thinking the GAN's discriminator is a scorer trained in advance and then frozen. It is trained from scratch together with the generator and keeps changing; Section 3 shows the judge's standard rising along with the generator's progress.

### 2. 兩個網路：生成器與判別器（Two networks: the generator and the discriminator）

GAN 由兩個網路組成。**生成器**（generator）$G$ 的輸入是一個從 $p_z$ 抽出的**雜訊向量** $\mathbf{z}$，例如標準常態；輸出是一筆生成的資料 $G(\mathbf{z})$。所有生成資料形成的分布記為 $p_g$。$\mathbf{z}$ 就是第12週 KP02 的潛在變數：看不見的原因，決定這筆資料長什麼樣子。生成器的任務是讓生成的資料騙過判別器。

A GAN consists of two networks. The **generator** $G$ takes a **noise vector** $\mathbf{z}$ drawn from $p_z$, such as the standard normal, and outputs a generated example $G(\mathbf{z})$. The distribution of all generated examples is written $p_g$. $\mathbf{z}$ is the latent variable of Week 12 KP02: the hidden cause that decides what the example looks like. The generator's job is to make its examples fool the discriminator.

**判別器** $D$ 的輸入是一筆資料 $\mathbf{x}$，可能是真的，也可能是生成的；輸出是一個 $0$ 到 $1$ 之間的數 $D(\mathbf{x})$，代表它認為這筆是真資料的機率。它的任務是把真的判高、假的判低，就像第2週的二元分類器，最後一層用 sigmoid：

The **discriminator** $D$ takes an example $\mathbf{x}$, which may be real or generated, and outputs a number $D(\mathbf{x})$ between $0$ and $1$, its probability that the example is real. Its job is to score real examples high and fakes low, just like the binary classifier of Week 2, with a sigmoid in the last layer:

$$
\hat{\mathbf{x}}=G(\mathbf{z}),\quad \mathbf{z}\sim p_z;\qquad D(\mathbf{x})=\sigma\big(h(\mathbf{x})\big)\in[0,1],
$$

其中 $h(\mathbf{x})$ 是判別器前面各層算出的一個實數，叫做 logit。$h$ 越大，$D$ 越接近 $1$。

where $h(\mathbf{x})$ is the real number computed by the discriminator's earlier layers, called the logit. The larger $h$, the closer $D$ is to $1$.

![一張方塊流程圖。下排：黃色的雜訊 z 進入藍色的生成器 G，輸出灰色的假樣本 G(z)；上排：綠色的真實資料給出灰色的真樣本 x。兩條箭頭都進入紫色的判別器 D，再輸出 D 在 0 到 1 之間、代表是真的機率。一條紅色虛線從判別器彎回生成器，旁邊寫著生成器只能透過判別器的回饋學習（A block diagram. Bottom row: yellow noise z enters the blue generator G, which outputs a grey fake sample G(z); top row: green real data gives a grey real sample x. Both arrows enter the purple discriminator D, which outputs D between 0 and 1, the probability of being real. A dashed red curve runs from the discriminator back to the generator, labelled the generator learns only through the discriminator's feedback）](../assets/figures/lec13-kp01-architecture.png)

**圖說 `lec13-kp01-architecture`**〔功能：一目了然〕：先看下排，雜訊 $\mathbf{z}$ 經過生成器變成假樣本；再看上排，真資料直接送出真樣本；兩路都進判別器，得到「是真的機率」。最後看紅色虛線：生成器從頭到尾沒有看過真資料，它只收到判別器傳回的梯度。

**Caption `lec13-kp01-architecture`** [Function: seeing it at a glance]: first look at the bottom row, where noise $\mathbf{z}$ passes through the generator and becomes a fake sample; then the top row, where real data gives real samples directly; both enter the discriminator, which gives "the probability of being real". Finally look at the dashed red curve: the generator never sees the real data at all; it only receives the gradient the discriminator sends back.

用一個一維的小 GAN 算一次 `[補充]`。真資料來自 $\mathcal{N}(2,\ 0.5^2)$。判別器用 $D(x)=\sigma(w_0+w_1x+w_2x^2)$；多一個 $x^2$ 項，它才看得出「寬度不對」，第3節會用到。生成器與訓練方式在第3節說明（每一「步」是判別器更新 $10$ 次、生成器更新 $1$ 次），訓練 $50$ 步後，三個權重約是 $w_0=-3.024$、$w_1=2.01$、$w_2=0.199$。對真資料的中心 $x=2$：

Compute once with a small one-dimensional GAN `[Added]`. The real data come from $\mathcal{N}(2,\ 0.5^2)$. The discriminator is $D(x)=\sigma(w_0+w_1x+w_2x^2)$; the extra $x^2$ term lets it notice "the width is wrong", which Section 3 uses. The generator and the training are described in Section 3 (each "step" updates the discriminator $10$ times and the generator once); after $50$ training steps its three weights are about $w_0=-3.024$, $w_1=2.01$ and $w_2=0.199$. At the centre of the real data, $x=2$:

$$
h(2)=-3.024+2.01\times2+0.199\times4\approx1.79,\qquad D(2)=\frac{1}{1+e^{-1.79}}\approx0.857 .
$$

對 $x=0$，$h(0)=w_0=-3.024$，$D(0)=1/(1+e^{3.024})\approx0.046$。這時生成器的樣本還以 $0$ 附近為中心、散得很寬，所以判別器學會了：$2$ 附近大概是真的，$0$ 附近大概是假的。

At $x=0$, $h(0)=w_0=-3.024$ and $D(0)=1/(1+e^{3.024})\approx0.046$. At this point the generator's samples are still centred near $0$ and spread wide, so the discriminator has learned: near $2$ is probably real, near $0$ is probably fake.

常見錯誤：以為生成器的輸入是一張真實的圖片，它負責「修圖」。生成器的輸入只有雜訊 $\mathbf{z}$；每抽一個新的 $\mathbf{z}$，就得到一筆新的資料。把真資料送進網路的只有判別器。

Common mistake: thinking the generator takes a real picture and "retouches" it. The generator's only input is the noise $\mathbf{z}$; each new $\mathbf{z}$ drawn gives a new example. Only the discriminator receives real data.

### 3. 對抗的循環：裁判越嚴，生成越像（The adversarial loop: the stricter the judge, the better the fakes）

兩個網路怎麼一起變強？來源用了 Goodfellow 的比喻：生成器像做假鈔的人，判別器像警察。警察學會分辨真鈔與假鈔，做假鈔的人就改進手法；假鈔變好，警察又得提升眼力。雙方輪流升級，直到假鈔和真鈔再也分不出來。這個比喻的侷限是：GAN 的判別器每次只用眼前這一批樣本更新，可能忘掉以前抓到過的假樣本，生成器也可能退回舊的手法 `[補充]`。

How do the two networks get stronger together? The source uses Goodfellow's analogy: the generator is a counterfeiter and the discriminator is the police. As the police learn to tell real notes from fakes, the counterfeiter improves the technique; as the fakes improve, the police must sharpen their eyes. The two take turns to upgrade, until fake and real notes can no longer be told apart. The analogy's limit: a GAN's discriminator is updated only on the current batch, so it can forget kinds of fakes it caught before, and the generator may slip back to them `[Added]`.

訓練就是把這兩件事輪流做：先固定生成器，用一批真樣本和一批假樣本更新判別器，讓它把真的判高、假的判低；再固定判別器，更新生成器，讓判別器把它的樣本判得更像真的。兩者的目標相反，所以叫**對抗**（adversarial）；KP02 會把它寫成一條式子。

Training alternates these two moves: first freeze the generator and update the discriminator with a batch of real and a batch of fake samples, so that it scores real ones high and fakes low; then freeze the discriminator and update the generator, so that the discriminator scores its samples as more real. The two goals are opposite, hence **adversarial**; KP02 writes this as one formula.

下圖是第2節那個小 GAN 的完整訓練 `[補充]`。生成器只有兩個參數：$G(z)=a+bz$，$z\sim\mathcal{N}(0,1)$，所以它的樣本是平均 $a$、標準差 $\lvert b\rvert$ 的常態分布；一開始 $a=0$、$b=1$。每一步先更新判別器 $10$ 次，再更新生成器 $1$ 次；學習率分別是 $0.1$ 與 $0.005$，每批 $256$ 個樣本，亂數種子 $0$，共 $6000$ 步。圖上「平均 $D$」用另外 $100000$ 個樣本估計（種子 $99$）。生成器更新時用的是「讓 $\log D(G(z))$ 變大」，KP02 第4節會說明為什麼不直接照原本的式子做。因為這裡的 $G$ 是一次函數，$p_g$ 剛好是常態分布 $\mathcal{N}(a,b^2)$，所以能畫成橘色曲線。

The figure below shows the full training of the small GAN of Section 2 `[Added]`. The generator has only two parameters: $G(z)=a+bz$ with $z\sim\mathcal{N}(0,1)$, so its samples follow a normal with mean $a$ and standard deviation $\lvert b\rvert$; it starts at $a=0$ and $b=1$. Each step updates the discriminator $10$ times and then the generator once; the learning rates are $0.1$ and $0.005$, each batch has $256$ samples, the random seed is $0$, and there are $6000$ steps. The "avg $D$" on the figure is estimated with another $100000$ samples (seed $99$). The generator update makes $\log D(G(z))$ larger; KP02 Section 4 explains why it does not follow the original formula literally. Because $G$ here is a linear function, $p_g$ is exactly the normal $\mathcal{N}(a,b^2)$, so it can be drawn as the orange curve.

![五張並排的小圖，橫軸是 x。每張都有淺藍色填色的真資料分布，中心在 2；橘色曲線是生成器的分布；紫色虛線是判別器 D(x)，對應右邊 0 到 1 的刻度；灰色點線標著 D=0.5。第 50 步：橘色曲線在 0 附近、很寬，紫線從左邊接近 0 升到右邊接近 1，平均 D 真 0.80、假 0.20。第 150 步：橘色曲線移到 1.18、變窄，平均 D 真 0.67、假 0.34。第 400 步：橘色曲線在 2.05、非常窄高，紫線是一個中間略高於 0.5、兩邊低的矮山丘，平均 D 都約 0.5。第 1000 步：橘色曲線在 1.99、寬度 0.58，紫線變成中間約 0.5、兩邊升高的山谷。第 6000 步：橘色曲線和藍色填色幾乎完全重合，紫線是一條接近 0.5 的平線（Five small panels side by side with x on the horizontal axis. Each has the real distribution as a light-blue fill centred at 2, the generator's distribution as an orange curve, the discriminator D(x) as a dashed purple curve on the right-hand scale from 0 to 1, and a grey dotted line marked D=0.5. Step 50: the orange curve is wide and near 0, the purple curve rises from near 0 on the left to near 1 on the right, avg D real 0.80 and fake 0.20. Step 150: the orange curve has moved to 1.18 and narrowed, avg D real 0.67 and fake 0.34. Step 400: the orange curve is at 2.05, very narrow and tall, the purple curve is a low hill slightly above 0.5 in the middle and low on both sides, and both avg D are about 0.5. Step 1000: the orange curve is at 1.99 with width 0.58, and the purple curve has become a valley, about 0.5 in the middle and rising on both sides. Step 6000: the orange curve almost exactly covers the blue fill, and the purple curve is a flat line near 0.5）](../assets/figures/lec13-kp01-toy-training.png)

**圖說 `lec13-kp01-toy-training`**〔功能：一目了然、精準度〕：由左到右看五個時間點。先看橘色曲線怎麼走：從 $0$ 附近移到 $2$，中途一度變得太窄、又稍微太寬，最後和藍色的真資料重合。再看紫色虛線：一開始是一道「右邊是真」的階梯，第 $400$ 步是矮山丘，第 $1000$ 步變成山谷，最後攤平在 $0.5$。每張圖左上角的兩個數，是判別器對真樣本與假樣本的平均評分。

**Caption `lec13-kp01-toy-training`** [Function: seeing it at a glance, precision]: read the five moments from left to right. First follow the orange curve: it moves from near $0$ to $2$, becomes too narrow for a while and then slightly too wide, and finally covers the blue real data. Then follow the dashed purple curve: it starts as a step saying "the right side is real", is a low hill at step $400$, becomes a valley at step $1000$, and finally flattens at $0.5$. The two numbers at the top left of each panel are the discriminator's average scores on real and fake samples.

這個過程可以逐步讀出來。第 $50$ 步，判別器對真樣本平均給 $0.80$、假樣本 $0.20$，分得很開；它的評分越往右越高，等於告訴生成器「往右走」。第 $150$ 步，生成器的中心已經移到 $1.18$，判別器的平均評分縮到 $0.67$ 與 $0.34$。第 $400$ 步，中心到了 $2.05$，可是標準差只剩 $0.26$，比真資料的 $0.5$ 窄。這時判別器跟不上，只是一座中間略高於 $0.5$ 的矮山丘，生成器還繼續變窄，大約第 $523$ 步窄到 $0.229$。

The process can be read step by step. At step $50$ the discriminator gives real samples $0.80$ on average and fakes $0.20$, far apart; its score rises to the right, which tells the generator "move right". At step $150$ the generator's centre has reached $1.18$, and the average scores have narrowed to $0.67$ and $0.34$. At step $400$ the centre is at $2.05$, but the standard deviation is only $0.26$, narrower than the real $0.5$. The discriminator lags behind here: it is only a low hill slightly above $0.5$ in the middle, and the generator keeps narrowing, down to $0.229$ around step $523$.

判別器隨後改成山谷形：第 $1000$ 步，它在中心 $D(2)\approx0.491$，在左邊遠處 $D(0)\approx0.887$。兩旁真資料還有、假資料卻很少的地方評分高，等於告訴生成器「往兩邊散開」，標準差回升到 $0.58$，在第 $1141$ 步左右多衝到 $0.67$，再慢慢收回。到第 $6000$ 步，生成器變成 $G(z)=2.00+0.50z$，正好是真資料的分布；判別器對真假樣本的平均評分都是 $0.494$。

The discriminator then turns into a valley: at step $1000$ it gives $D(2)\approx0.491$ at the centre and $D(0)\approx0.887$ far to the left. It scores high on both sides, where real data still lie but fakes are rare, which tells the generator "spread out"; the standard deviation climbs back to $0.58$, overshoots to $0.67$ around step $1141$, and then settles. By step $6000$ the generator has become $G(z)=2.00+0.50z$, exactly the real distribution, and the discriminator's average score is $0.494$ for both real and fake samples.

注意第 $400$ 步：真假樣本的平均評分都約 $0.5$，生成器的寬度卻明明不對 `[補充]`。所以「平均評分接近 $0.5$」本身不能證明已經學好，還要看判別器在各處的評分。

Note step $400$: both average scores are about $0.5$, yet the generator's width is plainly wrong `[Added]`. So "average scores near $0.5$" by itself does not prove the job is done; you must also look at the discriminator's scores everywhere.

判別器最後停在 $0.5$ 附近，不是它壞了，而是它真的分不出來：兩種樣本的分布一樣，最好的判斷就是一律猜一半。來源的互動展示也以此為理想終點：$p_g=p_{\text{data}}$，$D(\mathbf{x})=0.5$。KP03 會證明這件事。

The discriminator ending near $0.5$ does not mean it broke; it truly cannot tell: the two kinds of samples have the same distribution, so the best judgment is to guess one half every time. The source's interactive demo also takes this as the ideal end point: $p_g=p_{\text{data}}$ and $D(\mathbf{x})=0.5$. KP03 proves it.

常見錯誤：以為訓練好的 GAN，判別器應該把真樣本判成接近 $1$。那只在生成器還很差的時候成立。生成器越成功，判別器越分不出來，評分就越接近 $0.5$。

Common mistake: thinking that in a well-trained GAN the discriminator should score real samples near $1$. That holds only while the generator is still poor. The more the generator succeeds, the less the discriminator can tell, and the closer its scores get to $0.5$.

### 4. GAN 和 VAE 的不同（How a GAN differs from a VAE）

GAN 和 VAE 背後是同一個生成故事：先抽一個看不見的原因 $\mathbf{z}$，再由網路把它變成資料 $\mathbf{x}$，來源畫成「$\mathbf{z}$ 指向 $\mathbf{x}$」的機率圖。差別在怎麼學，以及學完能做什麼。

A GAN and a VAE share the same generative story: draw a hidden cause $\mathbf{z}$, then a network turns it into data $\mathbf{x}$; the source draws this as a probabilistic graph with "$\mathbf{z}$ pointing to $\mathbf{x}$". They differ in how they learn and in what they can do afterwards.

| 比較 | VAE（第12週） | GAN |
|---|---|---|
| 分數從哪裡來 | ELBO：重建項減 KL，寫得出公式 | 判別器的評分，跟著訓練一起變 |
| 網路 | 編碼器＋解碼器 | 生成器＋判別器，沒有編碼器 |
| 學完以後 | 可以重建、可以算 ELBO | 只能抽樣，給不出一筆資料的機率 |
| 影像 | 平方誤差偏好平均，偏模糊 | 模糊的平均騙不過判別器，通常較清楚 |

| Comparison | VAE (Week 12) | GAN |
|---|---|---|
| Where the score comes from | ELBO: reconstruction minus KL, a written formula | the discriminator's score, which changes during training |
| Networks | encoder + decoder | generator + discriminator, no encoder |
| After training | can reconstruct and compute the ELBO | can only sample; gives no probability for an example |
| Images | squared error prefers the average, so blurry | a blurry average cannot fool the discriminator, so usually sharper |

表裡第三列值得多想一下 `[補充]`：GAN 的生成器只會「抽樣」，沒有一條公式能算出某筆資料出現的機率 $p_g(\mathbf{x})$，這叫**隱式**（implicit）的生成模型；第3節能畫出 $p_g$，只是因為那個生成器是一次函數，深層、非線性的生成器就算不出來。VAE 則是**顯式**（explicit）的機率模型：它寫得出 $p(\mathbf{x})$ 的定義，只是算不出來，改用 ELBO 這個下界近似。KP08（選讀）會回頭比較兩者。代價也在這裡：沒有一條固定的分數，GAN 的訓練常常不穩定，KP04 會談。

The third row deserves a closer look `[Added]`: a GAN's generator can only "sample"; no formula gives the probability $p_g(\mathbf{x})$ of a given example, which makes it an **implicit** generative model; Section 3 could draw $p_g$ only because that generator is linear, and a deep nonlinear generator gives no such formula. The VAE is an **explicit** probability model: it defines $p(\mathbf{x})$, which cannot be computed exactly, so it is approximated through the ELBO, a lower bound. KP08 (optional) returns to this comparison. The price is here too: without a fixed score, GAN training is often unstable, which KP04 discusses.

常見錯誤：以為訓練好之後，判別器還要留著，生成新資料時要先經過它。生成時只用生成器：抽一個 $\mathbf{z}$，算 $G(\mathbf{z})$ 就好；判別器只在訓練時當裁判。

Common mistake: thinking the discriminator must be kept after training and that new data must pass through it. Generation uses only the generator: draw a $\mathbf{z}$ and compute $G(\mathbf{z})$; the discriminator serves as the judge only during training.

### 本單元重點整理（KP01 summary）

**一句話重點**：GAN 用一個可微分的判別器當裁判，取代寫死的平方誤差；生成器把雜訊 $\mathbf{z}$ 變成資料、想騙過判別器，判別器想分辨真假，兩者輪流訓練，直到生成的分布和真資料一樣、判別器只能猜 $0.5$。

**One-sentence summary**: a GAN replaces a fixed ruler such as squared error with a differentiable discriminator as the judge; the generator turns noise $\mathbf{z}$ into data and tries to fool the discriminator, the discriminator tries to tell real from fake, and the two train in turns until the generated distribution matches the real data and the discriminator can only guess $0.5$.

**白話說明**：一個學徒在練習畫名畫，旁邊有一個鑑定師。學徒一開始畫得很差，鑑定師一眼就看穿；學徒照鑑定師嫌棄的地方修改，鑑定師也越看越仔細。等到鑑定師對真畫和仿畫都只能說「一半一半」，學徒就出師了。

**Plain explanation**: an apprentice practises painting masterpieces next to an art authenticator. At first the apprentice paints badly and the authenticator sees through it at once; the apprentice fixes whatever the authenticator dislikes, and the authenticator looks ever more closely. When the authenticator can only say "fifty-fifty" for both real and copied paintings, the apprentice has learned the craft.

**正式定義**：生成器 $G:\mathbb{R}^k\to\mathcal{X}$，$\hat{\mathbf{x}}=G(\mathbf{z})$，$\mathbf{z}\sim p_z$，輸出分布記為 $p_g$；判別器 $D:\mathcal{X}\to[0,1]$，$D(\mathbf{x})=\sigma(h(\mathbf{x}))$ 是「$\mathbf{x}$ 來自 $p_{\text{data}}$」的機率。訓練輪流更新 $D$（真判高、假判低）與 $G$（讓 $D(G(\mathbf{z}))$ 變大）；理想終點 $p_g=p_{\text{data}}$、$D\equiv0.5$。

**Formal definition**: the generator $G:\mathbb{R}^k\to\mathcal{X}$, $\hat{\mathbf{x}}=G(\mathbf{z})$ with $\mathbf{z}\sim p_z$, has output distribution $p_g$; the discriminator $D:\mathcal{X}\to[0,1]$, $D(\mathbf{x})=\sigma(h(\mathbf{x}))$, is the probability that $\mathbf{x}$ comes from $p_{\text{data}}$. Training alternates updates of $D$ (real high, fake low) and of $G$ (make $D(G(\mathbf{z}))$ larger); the ideal end point is $p_g=p_{\text{data}}$ with $D\equiv0.5$.

**例子**：一維小 GAN，真資料 $\mathcal{N}(2,\ 0.5^2)$，生成器 $G(z)=a+bz$ 從 $a=0$、$b=1$ 出發；第 $50$ 步判別器給 $D(2)\approx0.857$、$D(0)\approx0.046$，真假平均評分 $0.80$ 與 $0.20$；第 $400$ 步中心到 $2.05$ 但標準差只有 $0.26$；第 $6000$ 步 $G(z)=2.00+0.50z$，真假平均評分都是 $0.494$。

**Example**: a one-dimensional GAN with real data $\mathcal{N}(2,\ 0.5^2)$ and generator $G(z)=a+bz$ starting from $a=0$, $b=1$; at step $50$ the discriminator gives $D(2)\approx0.857$ and $D(0)\approx0.046$, with average scores $0.80$ for real and $0.20$ for fake; at step $400$ the centre reaches $2.05$ but the standard deviation is only $0.26$; at step $6000$, $G(z)=2.00+0.50z$, and both average scores are $0.494$.

**常見迷思**：

- 「判別器是事先訓練好、固定的評分器。」錯，它和生成器一起從頭訓練，而且一直在變。
- 「生成器的輸入是一張真圖片。」錯，它的輸入只有雜訊；只有判別器看得到真資料。
- 「訓練好的判別器應該把真樣本判成接近 $1$。」錯，生成器成功時，判別器的評分接近 $0.5$。
- 「生成時要經過判別器。」錯，生成只用生成器。

**Common misconceptions**:

- "The discriminator is a scorer trained in advance and then frozen." Wrong: it is trained from scratch with the generator and keeps changing.
- "The generator's input is a real picture." Wrong: its only input is noise; only the discriminator sees real data.
- "A well-trained discriminator should score real samples near $1$." Wrong: when the generator succeeds, the discriminator's scores are near $0.5$.
- "Generation passes through the discriminator." Wrong: generation uses only the generator.

**圖**：本單元有兩張圖。

- `lec13-kp01-architecture`：生成器、判別器與兩條資料路徑，生成器只透過判別器的回饋學習。
- `lec13-kp01-toy-training`：一維小 GAN 的四個時間點，生成分布移向真資料，判別器最後攤平在 $0.5$。

**Figures** — this unit has two figures.

- `lec13-kp01-architecture`: the generator, the discriminator and the two data paths; the generator learns only through the discriminator's feedback.
- `lec13-kp01-toy-training`: four moments of a one-dimensional GAN, as the generated distribution moves towards the real data and the discriminator finally flattens at $0.5$.

### 小練習（Practice）

1. 用第2節第 $50$ 步的判別器權重 $w_0=-3.024$、$w_1=2.01$、$w_2=0.199$，算 $D(1)$。這個評分比 $D(2)$ 高還是低？為什麼合理？
2. 某時刻判別器對真樣本的平均評分是 $0.9$，對假樣本是 $0.1$。這時生成器做得好不好？判別器提供給生成器的訊號是什麼？
3. 一位同學說：「我的 GAN 訓練完，判別器對所有樣本都輸出 $0.5$，判別器一定壞了。」請說明這句話哪裡不對，以及它在什麼情況下才值得懷疑。

1. With the step-$50$ discriminator weights of Section 2, $w_0=-3.024$, $w_1=2.01$ and $w_2=0.199$, compute $D(1)$. Is this score higher or lower than $D(2)$? Why is that reasonable?
2. At some moment the discriminator's average score is $0.9$ on real samples and $0.1$ on fakes. Is the generator doing well? What signal does the discriminator give the generator?
3. A classmate says: "After training, my GAN's discriminator outputs $0.5$ for every sample, so it must be broken." Explain what is wrong with this statement, and when it would be worth suspecting.

**第 1 題解答** `[補充]`：$h(1)=-3.024+2.01+0.199=-0.815$，$D(1)=1/(1+e^{0.815})\approx0.307$，比 $D(2)\approx0.857$ 低。這時假樣本還集中在 $0$ 附近、真樣本在 $2$ 附近，$x=1$ 夾在中間，判別器自然沒那麼有把握是真的。

**Solution to question 1** `[Added]`: $h(1)=-3.024+2.01+0.199=-0.815$ and $D(1)=1/(1+e^{0.815})\approx0.307$, lower than $D(2)\approx0.857$. At this point the fakes are still around $0$ and the real samples around $2$; $x=1$ lies between them, so the discriminator is naturally less sure it is real.

**第 2 題解答** `[補充]`：做得不好，判別器能把真假分得很開。判別器的評分在哪個方向升高，就是告訴生成器樣本該往哪裡移；生成器沿著讓 $D(G(\mathbf{z}))$ 變大的方向更新。

**Solution to question 2** `[Added]`: not well; the discriminator separates real from fake by a wide margin. The direction in which the discriminator's score rises tells the generator where to move its samples; the generator updates in the direction that makes $D(G(\mathbf{z}))$ larger.

**第 3 題解答** `[補充]`：如果生成的分布已經和真資料一樣，判別器分不出來，最好的答案就是一律給 $0.5$，這正是理想的終點。值得懷疑的情況是：生成的樣本明明看起來不像真的，判別器卻還是輸出 $0.5$，這表示判別器沒學好，例如它太弱或訓練步數不夠。

**Solution to question 3** `[Added]`: if the generated distribution already matches the real data, the discriminator cannot tell them apart, and the best answer is $0.5$ for everything, which is exactly the ideal end point. It is worth suspecting when the generated samples plainly do not look real yet the discriminator still outputs $0.5$; that means the discriminator has not learned well, for example because it is too weak or has had too few training steps.

### 小測驗（Quiz，來源原題）

來源 KP-1 有一題自我檢測，保留原題與全部選項；來源頁面附有解析，下面依它說明。

The source's KP-1 has one self-check question, kept here with all its options; the source page gives an explanation, which the note follows below.

1. 傳統人眼觀察評估生成樣本品質無法直接用於神經網絡優化的根本原因為何？
   - (A) 因為人類觀察者審美觀各不相同，數據雜訊過大
   - (B) 因為人類主觀判斷無法計算數學導數 (Derivatives)，無法進行反向傳播梯度優化
   - (C) 因為人眼處理影像的速度太慢
   - (D) 因為人類只能識別 RGB 彩色照片

1. Why, fundamentally, can traditional human visual assessment of generated samples not be used directly to optimise a neural network?
   - (A) Because human observers' tastes differ and the data are too noisy
   - (B) Because human subjective judgment has no mathematical derivative, so gradient-based backpropagation cannot be done
   - (C) Because human eyes process images too slowly
   - (D) Because humans can only recognise RGB colour photos

**解析** `[補充]`：選 (B)，見第1節。(A) 與 (C) 是實務上的麻煩，但就算評審一致、速度夠快，人的判斷仍然沒有梯度；GAN 用可微分的判別器補上這一點。

**Explanation** `[Added]`: the answer is B; see Section 1. A and C are practical nuisances, but even with consistent and fast judges, human judgment still has no gradient; a GAN fills this gap with a differentiable discriminator.

### 留給下一單元的問題（The question for the next unit）

第3節只用文字說了「判別器想把真的判高、假的判低，生成器想讓假的被判高」。這兩個相反的願望，能不能寫成同一條式子？判別器該最大化什麼、生成器該最小化什麼？下一個知識點（本週 KP02）把它寫成極小化極大的賽局。

Section 3 only said in words that "the discriminator wants real high and fake low, and the generator wants its fakes scored high". Can these two opposite wishes be written as one formula? What should the discriminator maximise, and what should the generator minimise? The next knowledge point (this week's KP02) writes it as a minimax game.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第14章「產生式對抗網絡」KP-1「GAN 導論與博弈論基礎」，以及 KP-2 開頭的假鈔比喻（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch14-gan/index.html ，讀取日期 2026-11-15）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from KP-1, "Introduction to GANs and game-theoretic foundations", of Chapter 14, "Generative adversarial networks", of CCC-LNS IDL_115, together with the counterfeiting analogy at the start of KP-2 (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch14-gan/index.html , accessed 2026-11-15). Content not in the source is marked `[Added]` where it appears.

- 來源說 VAE 與高斯混合模型「需要顯式擬合機率密度函數，這導致生成的樣本常常呈現模糊」。顯式本身不會造成模糊：也有其他顯式模型（例如自迴歸的 PixelCNN、正規化流）能生成清楚的樣本。VAE 的模糊，第12週 KP07 已說明主因是常態解碼器（平方誤差）在拿不準時偏好平均；另外，最大概似會想蓋住所有資料，也可能把機率放在兩群資料之間。本講義第1節照這個原因寫。
- 來源稱兩者是「零和動態博弈」並說「兩者在訓練過程中不斷互相強化」。「互相強化」描述的是理想情況；實際訓練常常不穩定（KP04、KP06），本講義在第4節註明。
- 來源用粗斜體 `\boldsymbol` 表示向量；本課程網站不支援，改用 `\mathbf`。來源譯名「產生式對抗網絡」，本課程沿用進度表的「生成對抗網路」。
- 本講義補上：一維小 GAN 的完整訓練與兩張圖、判別器評分的手算例子、假鈔比喻的侷限、平均評分接近 $0.5$ 不代表學好、GAN 與 VAE 的對照表、隱式與顯式的說明、學徒與鑑定師的白話說明、三題小練習與解答。
- 取捨：KP-1 的互動展示不重做；KP-2 的目標函數留給本週 KP02。

- The source says VAEs and Gaussian mixture models "must fit a probability density explicitly, which makes the generated samples often blurry". Being explicit does not by itself cause blur: other explicit models, such as the autoregressive PixelCNN and normalizing flows, give sharp samples. For the VAE, Week 12 KP07 explained that the main cause is the normal (squared-error) decoder preferring the average when unsure; in addition, maximum likelihood tries to cover all the data and can put probability between groups of data. Section 1 follows this cause.
- The source calls the setting a "zero-sum dynamic game" in which "the two keep strengthening each other during training". "Strengthening each other" describes the ideal case; real training is often unstable (KP04, KP06), which Section 4 notes.
- The source writes vectors with bold italic `\boldsymbol`; the course website does not support it, so this note uses `\mathbf`. The source's Chinese name for GAN differs slightly from the one used in the course schedule, which this note follows.
- This note adds: the full training of a one-dimensional GAN with two figures, the hand calculation of a discriminator score, the limit of the counterfeiting analogy, the point that average scores near $0.5$ do not prove success, the GAN–VAE comparison table, the explanation of implicit versus explicit, the apprentice-and-authenticator explanation, and three practice questions with solutions.
- Choices: the KP-1 interactive demo is not rebuilt; the objective of KP-2 is left to this week's KP02.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 生成模型 | generative model |
| 生成對抗網路 | generative adversarial network (GAN) |
| 生成器 | generator |
| 判別器 | discriminator |
| 雜訊向量 | noise vector |
| 潛在變數（第12週 KP02） | latent variable |
| 對抗 | adversarial |
| 分對數 | logit |
| 隱式生成模型 | implicit generative model |
| 顯式生成模型 | explicit generative model |
| 極小化極大賽局（KP02） | minimax game |
| 變分自編碼器（第12週） | variational autoencoder (VAE) |
| 證據下界（第12週 KP03） | evidence lower bound (ELBO) |
| KL 散度（第12週 KP02） | Kullback–Leibler (KL) divergence |
| 紅綠藍色彩（小測驗選項） | red, green, blue (RGB) colour |
| 自迴歸影像模型 PixelCNN | PixelCNN (autoregressive image model) |
| 正規化流 | normalizing flow |
| 最大概似 | maximum likelihood |
