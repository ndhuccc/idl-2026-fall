# 第12週 KP02：潛在變數與近似後驗，看不見的原因反推不出來，就找一個簡單的分布來猜（Latent variables and the approximate posterior: when the hidden cause cannot be worked out exactly, guess it with a simple distribution）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第12週的第二個知識點（KP02），接在 KP01 之後。KP01 的結論是：自編碼器沒有規定編碼該落在哪裡，所以不能直接生成。這一節換成機率的眼光：把編碼看成每筆資料背後看不見的「原因」，問「看到資料以後，原因可能是什麼」，並說明這個問題為什麼通常算不出精確答案。

This is the second knowledge point (KP02) of Week 12, following KP01. KP01 concluded that an autoencoder does not say where codes should lie, so it cannot generate directly. This note switches to the eyes of probability: it treats the code as the invisible "cause" behind each example, asks "having seen the data, what might the cause be", and explains why this question usually has no exact answer we can compute.

先備知識是 KP01、第1週 KP06 的先驗、密度與後驗、第1週 KP07 的因子模型，以及高中的條件機率與常態分布。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第12章 §1–§2 改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec12-kp02-figures.py` 算出，可以自己重算。

The prerequisites are KP01, the prior, density and posterior of Week 1 KP06, the factor model of Week 1 KP07, and high-school conditional probability and the normal distribution. Every term and symbol is defined when it first appears. This note is rewritten from §1–§2 of Chapter 12 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec12-kp02-figures.py`, so you can recompute it yourself.

## KP02 潛在變數與近似後驗，看不見的原因反推不出來，就找一個簡單的分布來猜（Latent variables and the approximate posterior: when the hidden cause cannot be worked out exactly, guess it with a simple distribution）

### 學習目標（Learning objectives）

讀完之後，你能用一個例子說明什麼是潛在變數，並寫出「先抽原因、再產生資料」的生成故事。你能用貝氏定理寫出後驗，並在一個小例子裡算出先驗、概似、後驗的數值。你能說明後驗的分母為什麼通常算不出來。你也能寫出 KL 散度的定義、算一個小例子，並說明變分推論為什麼要用一個簡單的分布 $q$ 去近似後驗。

After reading, you can explain what a latent variable is with an example, and write the generative story "first draw a cause, then produce the data". You can write the posterior with Bayes' rule and compute the prior, likelihood and posterior in a small example. You can explain why the denominator of the posterior usually cannot be computed. You can also write the definition of the KL divergence, compute a small example, and explain why variational inference approximates the posterior with a simple distribution $q$.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{x}$ | 看得見的資料（一筆） | observed data (one example) |
| $\mathbf{z}$ | 潛在變數：看不見的原因 | latent variable: the invisible cause |
| $\theta$ | 模型的參數 | parameters of the model |
| $p(\mathbf{z})$ | 先驗：還沒看資料時，對 $\mathbf{z}$ 的看法 | prior: belief about $\mathbf{z}$ before seeing data |
| $p(\mathbf{x}\mid\mathbf{z},\theta)$ | 概似：原因是 $\mathbf{z}$ 時，產生 $\mathbf{x}$ 的機率密度 | likelihood: density of producing $\mathbf{x}$ when the cause is $\mathbf{z}$ |
| $p(\mathbf{z}\mid\mathbf{x},\theta)$ | 後驗：看到 $\mathbf{x}$ 之後，對 $\mathbf{z}$ 的看法 | posterior: belief about $\mathbf{z}$ after seeing $\mathbf{x}$ |
| $p(\mathbf{x}\mid\theta)$ | 證據（邊際概似）：資料本身的機率密度 | evidence (marginal likelihood): density of the data itself |
| $q(\mathbf{z})$ | 近似後驗：我們選來猜後驗的簡單分布 | approximate posterior: the simple distribution we pick to guess the posterior |
| $D_{\text{KL}}(q\,\Vert\,p)$ | KL 散度：$q$ 和 $p$ 差多遠 | KL divergence: how far $q$ is from $p$ |
| $\mathcal{N}(\mu,\sigma^2)$ | 平均 $\mu$、變異數 $\sigma^2$ 的常態分布 | normal distribution with mean $\mu$ and variance $\sigma^2$ |

### 1. 看不見的原因：潛在變數（The invisible cause: latent variables）

來源用一個生活例子開場。你觀察到「穿著鞋睡覺的人，第二天早上常常頭痛」。直覺會說穿鞋造成頭痛。可是再看下去，穿鞋睡覺的人大多是喝醉了。真正的原因是酒醉，穿鞋和頭痛都是它的結果，兩者只是一起出現。

The source opens with an everyday example. You notice that "people who sleep with their shoes on often wake up with a headache". Intuition says the shoes cause the headache. But look further: most people who sleep in their shoes are drunk. The real cause is drunkenness; shoes and headache are both its effects, and they merely appear together.

酒醉就是一個**潛在變數**（latent variable）：它看不見，卻決定了看得見的現象。機器學習裡也一樣：一張臉的照片由很多看不見的因素決定，例如臉型、角度、光線。我們用 $\mathbf{z}$ 代表這些原因，用 $\mathbf{x}$ 代表看得見的資料。

Drunkenness is a **latent variable**: it is invisible, yet it decides what we can see. Machine learning is the same: a photo of a face is decided by many invisible factors, such as face shape, angle and lighting. We write $\mathbf{z}$ for these causes and $\mathbf{x}$ for the visible data.

這類模型可以寫成一個**生成故事**（generative story），分兩步：先從**先驗**（prior）$p(\mathbf{z})$ 抽一個原因 $\mathbf{z}$，再從**概似**（likelihood）$p(\mathbf{x}\mid\mathbf{z},\theta)$ 產生資料 $\mathbf{x}$。$\theta$ 是模型的參數，例如神經網路的權重。第1週 KP07 的因子模型就是這種故事 `[補充：連結]`：少數幾個共同因素 $\mathbf{z}$，乘上一個矩陣，再加上雜訊，得到資料。

Such models can be written as a **generative story** in two steps: first draw a cause $\mathbf{z}$ from the **prior** $p(\mathbf{z})$, then produce the data $\mathbf{x}$ from the **likelihood** $p(\mathbf{x}\mid\mathbf{z},\theta)$. $\theta$ is the model's parameters, for example the weights of a neural network. The factor model of Week 1 KP07 is such a story `[Added: link]`: a few common factors $\mathbf{z}$, multiplied by a matrix, plus noise, give the data.

這個故事和 KP01 接得上。KP01 的解碼器「給一個編碼，畫出一筆資料」，給的是概似 $p(\mathbf{x}\mid\mathbf{z},\theta)$ 的中心 `[補充：連結]`。而 KP01 缺的那一塊，「編碼該從哪裡挑」，就是先驗 $p(\mathbf{z})$：它事先規定了原因的分布。生成新資料時，只要照故事做兩步。

This story connects to KP01. KP01's decoder, "given a code, draw an example", gives the centre of the likelihood $p(\mathbf{x}\mid\mathbf{z},\theta)$ `[Added: link]`. And the missing piece in KP01, "where to pick codes from", is the prior $p(\mathbf{z})$: it fixes the distribution of causes in advance. To generate new data, just follow the story's two steps.

常見錯誤：以為一起出現的兩件事，一定有一件造成另一件。來源的穿鞋例子就是反例：兩者都由看不見的第三件事造成。找潛在變數，就是在找那個看不見的共同原因。

Common mistake: thinking that when two things appear together, one must cause the other. The source's shoe example is a counterexample: both are caused by an invisible third thing. Looking for latent variables means looking for that invisible common cause.

### 2. 從資料反推原因：貝氏定理與後驗（Working back from data to cause: Bayes' rule and the posterior）

生成故事是「原因 → 資料」。學習與理解卻要反過來：看到一筆資料 $\mathbf{x}$，原因 $\mathbf{z}$ 可能是什麼？這個「看到資料之後對原因的看法」叫**後驗**（posterior）$p(\mathbf{z}\mid\mathbf{x},\theta)$。KP01 的編碼器只給出 $\mathbf{z}$ 的一個點估計；後驗則給出所有可能的 $\mathbf{z}$ 各有多可能。

The generative story runs "cause → data". Learning and understanding run the other way: having seen an example $\mathbf{x}$, what might its cause $\mathbf{z}$ be? This "belief about the cause after seeing the data" is the **posterior** $p(\mathbf{z}\mid\mathbf{x},\theta)$. KP01's encoder gives only one point guess of $\mathbf{z}$; the posterior says how likely every possible $\mathbf{z}$ is.

貝氏定理把後驗寫成三樣東西：

Bayes' rule writes the posterior in terms of three things:

$$
p(\mathbf{z}\mid\mathbf{x},\theta)=\frac{p(\mathbf{x}\mid\mathbf{z},\theta)\,p(\mathbf{z})}{p(\mathbf{x}\mid\theta)},\qquad
p(\mathbf{x}\mid\theta)=\int p(\mathbf{x}\mid\mathbf{z},\theta)\,p(\mathbf{z})\,d\mathbf{z}.
$$

白話：後驗正比於「先驗乘上概似」，也就是「這個原因本來多常見」乘上「這個原因產生這筆資料的可能性」。分母 $p(\mathbf{x}\mid\theta)$ 叫**證據**（evidence），它把所有可能的原因都加起來，讓後驗的總面積等於 $1$。

In plain words: the posterior is proportional to "prior times likelihood", that is, "how common this cause is to begin with" times "how likely this cause is to produce this example". The denominator $p(\mathbf{x}\mid\theta)$ is called the **evidence**; it adds up over all possible causes so that the posterior's total area is $1$.

用一個一維的小例子算一次 `[補充]`。先驗 $z\sim\mathcal{N}(0,1)$；資料 $x$ 等於 $z$ 加上標準差 $0.5$ 的雜訊，也就是 $x\mid z\sim\mathcal{N}(z,0.5^2)$。觀察到 $x=1.2$。在 $z=1$ 這一點：

A small one-dimensional example `[Added]`. The prior is $z\sim\mathcal{N}(0,1)$; the data $x$ is $z$ plus noise with standard deviation $0.5$, that is, $x\mid z\sim\mathcal{N}(z,0.5^2)$. We observe $x=1.2$. At the point $z=1$:

- 先驗密度 $p(z{=}1)=0.2420$（Prior density $p(z{=}1)=0.2420$）
- 概似 $p(x{=}1.2\mid z{=}1)=0.7365$（Likelihood $p(x{=}1.2\mid z{=}1)=0.7365$）
- 兩者相乘 $0.2420\times0.7365=0.1782$（Their product $0.2420\times0.7365=0.1782$）
- 證據 $p(x{=}1.2)=0.2006$（這個模型下 $x\sim\mathcal{N}(0,1.25)$，可以直接算）（Evidence $p(x{=}1.2)=0.2006$, computable directly since $x\sim\mathcal{N}(0,1.25)$ in this model）
- 後驗密度 $0.1782/0.2006=0.8885$（Posterior density $0.1782/0.2006=0.8885$）

對每一個 $z$ 都這樣算，就得到整條後驗曲線。這個例子剛好有公式解。兩個獨立常態的變異數可以相加，所以 $x\sim\mathcal{N}(0,\ 1+0.5^2)=\mathcal{N}(0,\ 1.25)$；後驗是 $\mathcal{N}\big(x/(1+\sigma^2),\ \sigma^2/(1+\sigma^2)\big)$，其中 $\sigma=0.5$，代入得 $\mathcal{N}(0.96,\ 0.2)$：平均 $1.2/1.25=0.96$，變異數 $0.25/1.25=0.2$（$\mathcal{N}$ 的第二個數一律是變異數）。注意密度不是機率，可以大於 $1$，例如這裡的 $0.8885$ 和下一張圖裡的 $2.5$；第1週 KP06 談過密度。

Doing this for every $z$ gives the whole posterior curve. This example happens to have a closed-form answer. Variances of independent normals add, so $x\sim\mathcal{N}(0,\ 1+0.5^2)=\mathcal{N}(0,\ 1.25)$; the posterior is $\mathcal{N}\big(x/(1+\sigma^2),\ \sigma^2/(1+\sigma^2)\big)$ with $\sigma=0.5$, giving $\mathcal{N}(0.96,\ 0.2)$: mean $1.2/1.25=0.96$ and variance $0.25/1.25=0.2$ (the second number of $\mathcal{N}$ is always the variance). Note that a density is not a probability and can exceed $1$, like $0.8885$ here and $2.5$ in the next figure; Week 1 KP06 covered densities.

![一張密度圖，橫軸是 z。灰色虛線是先驗，以 0 為中心的鐘形曲線；橘色線是概似，以 1.2 為中心；藍色粗線是後驗，以 0.96 為中心，比兩者都窄、都高。z 等於 1 的地方有三個點，分別在三條曲線上（A density plot over z. The grey dashed prior is a bell centred at 0; the orange likelihood is centred at 1.2; the thick blue posterior is centred at 0.96, narrower and taller than both. Three dots at z equals 1 sit on the three curves）](../assets/figures/lec12-kp02-bayes-posterior.png)

**圖說 `lec12-kp02-bayes-posterior`**〔功能：一目了然、精準度〕：先看灰色虛線：沒看資料前，$z$ 最可能在 $0$ 附近。再看橘線：只看資料，$z$ 最可能在 $1.2$ 附近。最後看藍線：兩者相乘再除以證據，後驗落在兩者之間的 $0.96$，而且比兩者都窄，因為兩個資訊加在一起比較確定。$z=1$ 的三個點就是上面算的 $0.2420$、$0.7365$、$0.8885$。

**Caption `lec12-kp02-bayes-posterior`** [Function: seeing it at a glance, precision]: first look at the grey dashed curve: before seeing data, $z$ is most likely near $0$. Then look at the orange curve: from the data alone, $z$ is most likely near $1.2$. Finally look at the blue curve: multiplying the two and dividing by the evidence puts the posterior between them at $0.96$, and narrower than both, because two pieces of information together are more certain. The three dots at $z=1$ are the $0.2420$, $0.7365$ and $0.8885$ computed above.

常見錯誤 `[補充]`：以為後驗就是概似。概似只問「這個原因產生這筆資料的可能性」，沒有考慮這個原因本來多常見。上圖的橘線和藍線就不一樣：後驗被先驗往 $0$ 拉了一點。

Common mistake `[Added]`: thinking that the posterior is the likelihood. The likelihood only asks how likely this cause is to produce this example; it ignores how common the cause is to begin with. In the figure above the orange and blue curves differ: the prior pulls the posterior a little towards $0$.

### 3. 分母的積分通常算不出來（The integral in the denominator usually cannot be computed）

第2節的例子能算，是因為一切都是常態分布，而且 $x$ 只是 $z$ 的線性函數。來源指出，真實模型的問題出在分母：$p(\mathbf{x}\mid\theta)=\int p(\mathbf{x}\mid\mathbf{z},\theta)p(\mathbf{z})\,d\mathbf{z}$ 要對所有可能的 $\mathbf{z}$ 積分。當概似由神經網路給出、$\mathbf{z}$ 又是高維連續的，這個積分沒有公式解 `[補充：條件]`。這叫後驗**不可解**（intractable）。注意「高維」本身不是問題：第1週的因子模型維度再高，因為是線性加常態雜訊，仍然有公式解；第4節的一維非線性例子雖然只有一維，也已經沒有公式解，只能靠數值加總。

The example of Section 2 could be computed because everything is normal and $x$ is just a linear function of $z$. The source points out that for real models the trouble is the denominator: $p(\mathbf{x}\mid\theta)=\int p(\mathbf{x}\mid\mathbf{z},\theta)p(\mathbf{z})\,d\mathbf{z}$ integrates over every possible $\mathbf{z}$. When the likelihood is given by a neural network and $\mathbf{z}$ is high-dimensional and continuous, this integral has no closed form `[Added: condition]`. The posterior is then called **intractable**. Note that high dimension alone is not the problem: the factor model of Week 1, however many dimensions it has, still has a closed form because it is linear with normal noise; and the nonlinear example of Section 4 is only one-dimensional, yet it already has no closed form and must be summed numerically.

那用電腦一格一格加總呢 `[補充：數字]`？每一維取 $100$ 個格點：一維要算 $100$ 次，二維要 $100^2=10^4$ 次，十維就要 $100^{10}=10^{20}$ 次。就算每秒算 $10^9$ 次，也要約 $3169$ 年。真實的潛在變數常有幾十到幾百維，暴力加總完全不可能。

What about adding it up on a grid by computer `[Added: numbers]`? With $100$ grid points per dimension, one dimension needs $100$ evaluations, two need $100^2=10^4$, and ten need $100^{10}=10^{20}$. Even at $10^9$ evaluations per second, that takes about $3169$ years. Real latent variables often have tens to hundreds of dimensions, so brute-force summing is hopeless.

常見錯誤：以為後驗算不出來，是因為資料太少或網路不夠深。來源的小測驗就考這一點：原因是分母那個對所有 $\mathbf{z}$ 的積分，和資料量、網路深度無關。

Common mistake: thinking the posterior cannot be computed because there is too little data or the network is not deep enough. The source's quiz tests exactly this: the reason is the integral over all $\mathbf{z}$ in the denominator, which has nothing to do with the amount of data or the depth of the network.

### 4. 找一個簡單的分布 $q$ 來近似（Approximate it with a simple distribution $q$）

既然真正的後驗算不出來，來源的想法是：**找一個簡單、算得動的分布 $q(\mathbf{z})$ 來近似它**。這叫**變分推論**（variational inference），$q$ 叫**近似後驗**。「變分」的意思是：在一群候選分布裡調整 $q$ 的參數，找出最好的那一個。

Since the true posterior cannot be computed, the source's idea is to **find a simple, computable distribution $q(\mathbf{z})$ to approximate it**. This is **variational inference**, and $q$ is the **approximate posterior**. "Variational" means adjusting the parameters of $q$ within a family of candidate distributions to find the best one.

「最好」要有一把尺。來源用的是 **KL 散度**（Kullback–Leibler divergence）：

"Best" needs a ruler. The source uses the **KL divergence** (Kullback–Leibler divergence):

$$
D_{\text{KL}}(q\,\Vert\,p)=\int q(\mathbf{z})\log\frac{q(\mathbf{z})}{p(\mathbf{z})}\,d\mathbf{z}\ \ge\ 0,
$$

只有 $q$ 和 $p$ 完全一樣時才等於 $0$。這裡和本週其他地方的 $\log$ 都是自然對數 $\ln$（底數 $e$），不是高中常用的以 $10$ 為底。結果只有有限個時，積分換成加總：$\sum_i q_i\log(q_i/p_i)$。白話：在 $q$ 認為可能的地方，比較 $q$ 和 $p$ 的對數差，再用 $q$ 加權平均。

which is $0$ only when $q$ and $p$ are identical. Here and throughout this week, $\log$ is the natural logarithm $\ln$ (base $e$), not the base-$10$ log common in high school. When there are only finitely many outcomes, the integral becomes a sum: $\sum_i q_i\log(q_i/p_i)$. In plain words: wherever $q$ thinks something is likely, compare the logs of $q$ and $p$, and average with $q$ as the weights.

兩個結果的小例子 `[補充]`：$q=[0.5,\ 0.5]$、$p=[0.9,\ 0.1]$。

A two-outcome example `[Added]`: $q=[0.5,\ 0.5]$ and $p=[0.9,\ 0.1]$.

$$
D_{\text{KL}}(q\,\Vert\,p)=0.5\log\frac{0.5}{0.9}+0.5\log\frac{0.5}{0.1}=-0.2939+0.8047=0.5108 .
$$

反過來算，$D_{\text{KL}}(p\,\Vert\,q)=0.3681$。兩個方向不一樣，所以 KL 散度不是一般的「距離」。變分推論用的是 $D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid\mathbf{x},\theta))$，$q$ 放在前面。

Computed the other way, $D_{\text{KL}}(p\,\Vert\,q)=0.3681$. The two directions differ, so the KL divergence is not an ordinary "distance". Variational inference uses $D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid\mathbf{x},\theta))$, with $q$ first.

$q$ 要選什麼樣子？來源建議用常態分布這類簡單的分布：好算、可以微分，又能表達多維的形狀。用一個例子看看近似長什麼樣 `[補充]`。把第2節的模型改成非線性：$x\mid z\sim\mathcal{N}(z^2,\ 0.3^2)$，觀察到 $x=1$。$z=1$ 和 $z=-1$ 都能產生 $x=1$，所以真正的後驗有兩個峰，分別在 $\pm0.977$，各占一半的機率。

What shape should $q$ have? The source suggests a simple family such as the normal distribution: easy to compute, differentiable, and still able to express multidimensional shapes. Let us see what an approximation looks like `[Added]`. Make the model of Section 2 nonlinear: $x\mid z\sim\mathcal{N}(z^2,\ 0.3^2)$, and observe $x=1$. Both $z=1$ and $z=-1$ can produce $x=1$, so the true posterior has two peaks, at $\pm0.977$, each holding half the probability.

![一張密度圖，橫軸是 z。淺藍色填色是真正的後驗，在 z 約負 1 和正 1 各有一個高峰。綠色粗線是最好的常態分布，又窄又高，只蓋住右邊的峰，標示 KL 等於 0.72。紅色虛線是一個以 0 為中心的寬常態分布，橫跨兩個峰，標示 KL 等於 7.36（A density plot over z. The light-blue filled true posterior has peaks near z equals minus 1 and plus 1. The thick green best normal is narrow and tall and covers only the right peak, with KL 0.72. The red dashed wide normal centred at 0 spans both peaks, with KL 7.36）](../assets/figures/lec12-kp02-best-gaussian.png)

**圖說 `lec12-kp02-best-gaussian`**〔功能：一目了然、精準度、空間幾何具象化〕：先看淺藍色：真正的後驗有兩個峰。再看綠線：在所有常態分布裡，讓 $D_{\text{KL}}(q\,\Vert\,p)$ 最小的是 $\mathcal{N}(0.94,\ 0.16^2)$，它只蓋住一個峰，KL 是 $0.72$。最後看紅色虛線：用後驗本身的平均與標準差做成的寬常態分布，雖然橫跨兩個峰，卻把機率放在兩峰之間，更放在兩峰外側的尾巴：那裡後驗幾乎是 $0$。尾巴（$|z|>1.5$）只有 $q$ 的 $11\%$ 機率，卻貢獻了 $6.07$；兩峰之間（$|z|<0.5$）貢獻 $1.27$；合計 KL 高達 $7.36$。

**Caption `lec12-kp02-best-gaussian`** [Function: seeing it at a glance, precision, making space concrete]: first look at the light blue: the true posterior has two peaks. Then look at the green curve: among all normal distributions, the one with the smallest $D_{\text{KL}}(q\,\Vert\,p)$ is $\mathcal{N}(0.94,\ 0.16^2)$; it covers only one peak, with KL $0.72$. Finally look at the red dashed curve: the wide normal built from the posterior's own mean and standard deviation spans both peaks but puts probability between them and, above all, in the tails beyond them, where the posterior is nearly $0$. The tails ($|z|>1.5$) hold only $11\%$ of $q$'s probability yet contribute $6.07$; the region between the peaks ($|z|<0.5$) contributes $1.27$; in all, the KL is as high as $7.36$.

這個例子說明兩件事 `[補充]`。第一，近似有代價：一個常態分布裝不下兩個峰，最好的 $q$ 只能選一個，所以 KL 沒辦法降到 $0$。最好的 $q$ 幾乎所有機率都在 $z>0$，而 $p$ 在那裡只有一半；光這一點就貢獻 $1\times\log(1/0.5)=\log 2=0.69$，剩下的 $0.72-0.69=0.03$ 是 $q$ 和右峰形狀的差距。第二，$D_{\text{KL}}(q\,\Vert\,p)$ 特別懲罰「$q$ 放了機率、$p$ 卻幾乎沒有」的地方，所以最好的 $q$ 寧可縮在一個峰上，也不去蓋兩峰之間和外側的空地。

This example shows two things `[Added]`. First, approximation has a price: one normal cannot hold two peaks, so the best $q$ must pick one, and the KL cannot reach $0$. The best $q$ puts almost all its probability at $z>0$, where $p$ has only half; that alone contributes $1\times\log(1/0.5)=\log 2=0.69$, and the remaining $0.72-0.69=0.03$ is the mismatch between $q$ and the shape of the right peak. Second, $D_{\text{KL}}(q\,\Vert\,p)$ especially penalises places where $q$ puts probability but $p$ has almost none, so the best $q$ prefers to shrink onto one peak rather than cover the empty ground between and beyond the peaks.

選常態分布還有一個要算的成本：參數數量。$\mathbf{z}$ 有 $d$ 維時，若各維互不相關（**對角**共變異數），只要 $d$ 個平均和 $d$ 個標準差，共 $2d$ 個；若允許任意相關（**完整**共變異數），共變異數矩陣是對稱的，要 $d(d+1)/2$ 個數，再加 $d$ 個平均。

Choosing the normal has another cost to count: the number of parameters. When $\mathbf{z}$ has $d$ dimensions and the dimensions are uncorrelated (a **diagonal** covariance), you need only $d$ means and $d$ standard deviations, $2d$ in all; allowing any correlation (a **full** covariance), the covariance matrix is symmetric and needs $d(d+1)/2$ numbers, plus $d$ means.

| $d$ | 對角共變異數 | 完整共變異數 |
|---|---|---|
| $2$ | $2\times2=4$ | $2+3=5$ |
| $10$ | $2\times10=20$ | $10+55=65$ |

| $d$ | diagonal covariance | full covariance |
|---|---|---|
| $2$ | $2\times2=4$ | $2+3=5$ |
| $10$ | $2\times10=20$ | $10+55=65$ |

$d=10$ 時完整版是對角版的 $3.25$ 倍；維度越高差距越大。所以 VAE 幾乎都用對角常態分布，本週 KP05 會再看到。

At $d=10$ the full version needs $3.25$ times as many; the gap grows with the dimension. That is why the VAE almost always uses a diagonal normal, as this week's KP05 will show.

最後還有一個卡住的地方。我們想把 $D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid\mathbf{x},\theta))$ 降到最小，可是這個式子裡就有那個算不出來的後驗。要怎麼最小化一個連算都算不出來的量？這是下一個知識點要解決的問題。

One snag remains. We want to minimise $D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid\mathbf{x},\theta))$, but this expression contains the very posterior we cannot compute. How do you minimise a quantity you cannot even evaluate? That is the problem of the next knowledge point.

常見錯誤：以為「近似」就是不準、不好用。來源提醒，後驗的形狀和所選的分布族相近時（例如大致只有一個峰），選得好的 $q$ 可以很接近真正的後驗，而且可以微分、可以用梯度最佳化。另一個錯誤是以為變分推論只用在 VAE；它是通用的貝氏推論工具，VAE 只是其中一個應用。

Common mistake: thinking that "approximate" means inaccurate and useless. The source points out that when the posterior's shape is close to the chosen family (for example, roughly one peak), a well-chosen $q$ can come very close to the true posterior, and it can be differentiated and optimised by gradients. Another mistake is thinking variational inference is used only in the VAE; it is a general tool for Bayesian inference, and the VAE is just one application.

### 本單元重點整理（KP02 summary）

**一句話重點**：把資料背後看不見的原因寫成潛在變數 $\mathbf{z}$，生成故事是「先從先驗抽 $\mathbf{z}$、再從概似產生 $\mathbf{x}$」；反推原因要用貝氏定理求後驗，但分母要對所有 $\mathbf{z}$ 積分，通常算不出來；變分推論改找一個簡單的分布 $q$，用 KL 散度衡量它和後驗差多遠。

**One-sentence summary**: write the invisible cause behind the data as a latent variable $\mathbf{z}$, with the generative story "draw $\mathbf{z}$ from the prior, then produce $\mathbf{x}$ from the likelihood"; working back to the cause needs the posterior from Bayes' rule, but its denominator integrates over all $\mathbf{z}$ and usually cannot be computed; variational inference instead looks for a simple distribution $q$, using the KL divergence to measure how far it is from the posterior.

**白話說明** `[補充]`：看到一個人早上頭痛，想猜他昨晚發生什麼事。把所有可能的經過一一算清楚太難了，所以先挑一種簡單的猜法，再一步步調整，讓它和真正的答案越來越像。

**Plain explanation** `[Added]`: you see someone with a headache in the morning and want to guess what happened last night. Working out every possible story exactly is too hard, so you pick a simple way of guessing and adjust it step by step until it looks more and more like the true answer.

**正式定義**：$p(\mathbf{z}\mid\mathbf{x},\theta)=p(\mathbf{x}\mid\mathbf{z},\theta)p(\mathbf{z})/p(\mathbf{x}\mid\theta)$，$p(\mathbf{x}\mid\theta)=\int p(\mathbf{x}\mid\mathbf{z},\theta)p(\mathbf{z})\,d\mathbf{z}$；變分推論在一群簡單分布中找 $q$，讓 $D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid\mathbf{x},\theta))=\int q\log(q/p)\,d\mathbf{z}$ 最小。

**Formal definition**: $p(\mathbf{z}\mid\mathbf{x},\theta)=p(\mathbf{x}\mid\mathbf{z},\theta)p(\mathbf{z})/p(\mathbf{x}\mid\theta)$, with $p(\mathbf{x}\mid\theta)=\int p(\mathbf{x}\mid\mathbf{z},\theta)p(\mathbf{z})\,d\mathbf{z}$; variational inference searches a family of simple distributions for the $q$ that minimises $D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid\mathbf{x},\theta))=\int q\log(q/p)\,d\mathbf{z}$.

**例子**：$z\sim\mathcal{N}(0,1)$、$x\mid z\sim\mathcal{N}(z,0.5^2)$、$x=1.2$ 時後驗是 $\mathcal{N}(0.96,0.2)$；十維格點要 $10^{20}$ 次運算；$q=[0.5,0.5]$ 對 $p=[0.9,0.1]$ 的 KL 是 $0.5108$，反方向是 $0.3681$；雙峰後驗的最好常態近似 KL 是 $0.72$、寬常態分布是 $7.36$；$d=10$ 時對角與完整共變異數各要 $20$ 與 $65$ 個參數。

**Example**: with $z\sim\mathcal{N}(0,1)$, $x\mid z\sim\mathcal{N}(z,0.5^2)$ and $x=1.2$, the posterior is $\mathcal{N}(0.96,0.2)$; a ten-dimensional grid needs $10^{20}$ evaluations; the KL of $q=[0.5,0.5]$ against $p=[0.9,0.1]$ is $0.5108$, and $0.3681$ the other way; the best normal approximation of the two-peaked posterior has KL $0.72$ and the wide normal $7.36$; at $d=10$ the diagonal and full covariances need $20$ and $65$ parameters.

**常見迷思**：

- 「一起出現就有因果。」錯，可能有共同的潛在原因。
- 「後驗就是概似。」錯，後驗還乘上了先驗。
- 「後驗算不出來是因為資料太少。」錯，是分母的積分。
- 「KL 散度是距離，兩個方向一樣。」錯，它不對稱。

**Common misconceptions**:

- "Appearing together means one causes the other." Wrong: there may be a common latent cause.
- "The posterior is the likelihood." Wrong: the posterior also multiplies in the prior.
- "The posterior cannot be computed because there is too little data." Wrong: it is the integral in the denominator.
- "The KL divergence is a distance, the same in both directions." Wrong: it is not symmetric.

**圖**：本單元有兩張圖。

- `lec12-kp02-bayes-posterior`：先驗、概似、後驗三條曲線，後驗在兩者之間且更窄。
- `lec12-kp02-best-gaussian`：雙峰的後驗，最好的常態近似只蓋住一個峰。

**Figures** — this unit has two figures.

- `lec12-kp02-bayes-posterior`: the prior, likelihood and posterior curves; the posterior lies between the other two and is narrower.
- `lec12-kp02-best-gaussian`: a two-peaked posterior whose best normal approximation covers only one peak.

### 小練習（Practice）

1. 第2節的模型，觀察到 $x=2.5$。後驗的平均和變異數是多少？（提示：平均是 $x/1.25$，變異數不變。）
2. $q=[0.8,\ 0.2]$、$p=[0.5,\ 0.5]$。算 $D_{\text{KL}}(q\,\Vert\,p)$，取到小數點後 $4$ 位。
3. 潛在變數有 $d=4$ 維。對角常態分布與完整常態分布各要幾個參數？

1. In the model of Section 2, we observe $x=2.5$. What are the mean and variance of the posterior? (Hint: the mean is $x/1.25$ and the variance does not change.)
2. With $q=[0.8,\ 0.2]$ and $p=[0.5,\ 0.5]$, compute $D_{\text{KL}}(q\,\Vert\,p)$ to $4$ decimal places.
3. The latent variable has $d=4$ dimensions. How many parameters do a diagonal normal and a full normal need?

**第 1 題解答** `[補充]`：平均 $2.5/1.25=2$，變異數 $0.2$；後驗是 $\mathcal{N}(2,\ 0.2)$。

**Solution to question 1** `[Added]`: the mean is $2.5/1.25=2$ and the variance $0.2$; the posterior is $\mathcal{N}(2,\ 0.2)$.

**第 2 題解答** `[補充]`：$0.8\log(0.8/0.5)+0.2\log(0.2/0.5)=0.3760-0.1833=0.1927$。

**Solution to question 2** `[Added]`: $0.8\log(0.8/0.5)+0.2\log(0.2/0.5)=0.3760-0.1833=0.1927$.

**第 3 題解答** `[補充]`：對角 $2\times4=8$ 個；完整 $4+4\times5/2=14$ 個。

**Solution to question 3** `[Added]`: diagonal $2\times4=8$; full $4+4\times5/2=14$.

### 小測驗（Quiz，來源原題）

來源這一段有兩題自我檢測，保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The source has two self-check questions for this part, kept here with all their options; the source page does not mark the answers, so the explanations below are judged from the source text.

1. 本章核心想問的問題是？
   - (A) 如何訓練更大的神經網路
   - (B) 如何估計「看不見的因子」對可觀察資料的影響
   - (C) 如何加速 SGD 收斂
   - (D) 如何設計 CNN 架構
2. 變分推論為什麼需要「近似」後驗？
   - (A) 因為資料量太少
   - (B) 因為真後驗的分母是 intractable 積分
   - (C) 因為神經網路不夠深
   - (D) 因為最佳化方法不夠好

1. What is the core question of this chapter?
   - (A) How to train larger neural networks
   - (B) How to estimate the influence of "invisible factors" on observable data
   - (C) How to speed up the convergence of SGD
   - (D) How to design CNN architectures
2. Why does variational inference need to "approximate" the posterior?
   - (A) Because there is too little data
   - (B) Because the denominator of the true posterior is an intractable integral
   - (C) Because the neural network is not deep enough
   - (D) Because the optimisation method is not good enough

**解析** `[補充]`：第 1 題選 (B)：第1節，潛在變數就是看不見的因子；第 2 題選 (B)：第3節，困難在分母對所有 $\mathbf{z}$ 的積分，和資料量、網路深度、最佳化方法無關。

**Explanations** `[Added]`: question 1 is B: see Section 1; latent variables are the invisible factors. Question 2 is B: see Section 3; the difficulty is the integral over all $\mathbf{z}$ in the denominator, unrelated to the amount of data, the depth of the network or the optimisation method.

### 留給下一單元的問題（The question for the next unit）

第4節卡在一個地方：要讓 $q$ 接近後驗，就要最小化 $D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid\mathbf{x},\theta))$，可是這個量本身就含有算不出來的後驗。有沒有辦法把它改寫成另一個算得出來的量，讓「最小化 KL」變成「最大化另一個東西」？下一個知識點（本週 KP03）推出這個東西：ELBO。

Section 4 got stuck at one point: to bring $q$ close to the posterior we must minimise $D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid\mathbf{x},\theta))$, yet this quantity itself contains the posterior we cannot compute. Can it be rewritten as something we can compute, turning "minimise the KL" into "maximise something else"? The next knowledge point (this week's KP03) derives that something: the ELBO.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第12章「變分模型」§1–§2 與教授祕笈「全章地圖」「變分推論：為什麼需要近似後驗」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch12-variational-models/index.html ，讀取日期 2026-11-14）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §1–§2 of Chapter 12, "Variational models", of CCC-LNS IDL_115, with the professor's notes "Chapter map" and "Variational inference: why approximate the posterior" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch12-variational-models/index.html , accessed 2026-11-14). Content not in the source is marked `[Added]` where it appears.

- 祕笈的練習說 $d=10$ 的完整常態分布要「$10+10\times10/2=60$ 個」，並說和對角版「差距 3 倍」；對稱的共變異數矩陣要 $d(d+1)/2=55$ 個數，加上 $10$ 個平均共 $65$ 個，是對角版 $20$ 個的 $3.25$ 倍。本講義在第4節用正確數字。
- 祕笈的全章練習說 $d=2$ 時對角版 $4$ 個「遠少於完整 $\Sigma$ 的 $d^2=4$ 個」；兩邊都是 $4$，並不更少。正確比較是對角 $4$ 個對完整 $5$ 個（$2$ 個平均加 $3$ 個共變異數）。
- 祕笈另有兩處寫完整共變異數要「$d^2$ 個」；共變異數矩陣對稱，只要 $d(d+1)/2$ 個，加上平均共 $d+d(d+1)/2$ 個。
- 來源用 $\mathbb{P}(\cdot)$ 與粗斜體 `\boldsymbol` 表示機率與向量；本講義改用 $p(\cdot)$ 與 `\mathbf`，意義相同。
- 本講義補上：與 KP01 解碼器和先驗的連結、「後驗就是概似」的迷思、不可解的條件說明、對數底數與離散 KL、$\log 2$ 的拆解、摘要的頭痛比喻、與第1週因子模型的連結、一維常態例子的逐點計算與圖、格點加總的成本估計、兩個結果的 KL 例子與不對稱性、雙峰後驗的最好常態近似與圖、三題小練習與解答。
- 取捨：來源祕笈的「博士生比例」與指數族的一般說法，本講義不寫；KL 翻轉成 ELBO 的推導留給 KP03。

- The notes' exercise says a full normal with $d=10$ needs "$10+10\times10/2=60$" and is "3 times" the diagonal one; a symmetric covariance matrix needs $d(d+1)/2=55$ numbers, plus $10$ means makes $65$, which is $3.25$ times the diagonal $20$. Section 4 uses the correct numbers.
- The notes' chapter exercise says that at $d=2$ the diagonal $4$ is "far fewer than the full $\Sigma$'s $d^2=4$"; both are $4$, so it is not fewer. The correct comparison is diagonal $4$ against full $5$ ($2$ means plus $3$ covariance entries).
- The notes say in two more places that a full covariance needs "$d^2$" numbers; the matrix is symmetric, so it needs only $d(d+1)/2$, or $d+d(d+1)/2$ including the means.
- The source writes probabilities and vectors with $\mathbb{P}(\cdot)$ and bold italic `\boldsymbol`; this note uses $p(\cdot)$ and `\mathbf`, with the same meaning.
- This note adds: the link to KP01's decoder and the prior, the "posterior equals likelihood" misconception, the conditions for intractability, the log base and discrete KL, the breakdown into $\log 2$, the headache analogy in the summary, the link to the factor model of Week 1, the point-by-point computation and figure of the one-dimensional normal example, the cost estimate of grid summing, the two-outcome KL example and its asymmetry, the best normal approximation of a two-peaked posterior with its figure, and three practice questions with solutions.
- Choices: the notes' "share of PhD students" analogy and the general remarks on exponential families are not covered; the derivation that flips the KL into the ELBO is left to KP03.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 潛在變數 | latent variable |
| 生成故事 | generative story |
| 先驗 | prior |
| 概似 | likelihood |
| 後驗 | posterior |
| 證據（邊際概似） | evidence (marginal likelihood) |
| 貝氏定理 | Bayes' rule |
| 不可解 | intractable |
| 變分推論 | variational inference |
| 近似後驗 | approximate posterior |
| KL 散度 | Kullback–Leibler (KL) divergence |
| 常態分布 | normal distribution |
| 共變異數 | covariance |
| 對角共變異數 | diagonal covariance |
| 完整共變異數 | full covariance |
| 因子模型（第1週） | factor model |
| 證據下界（KP03） | evidence lower bound (ELBO) |
| 變分自編碼器（KP05） | variational autoencoder (VAE) |
| 隨機梯度下降（第3週） | stochastic gradient descent (SGD) |
| 卷積神經網路（第6週） | convolutional neural network (CNN) |
