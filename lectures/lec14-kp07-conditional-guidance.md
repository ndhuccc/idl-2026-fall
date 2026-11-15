# 第14週 KP07：條件式擴散：ILVR、引導與 cross-attention（Conditional diffusion: ILVR, guidance and cross-attention）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第14週「深度生成模型（三）：擴散模型」的第七個知識點（KP07）。到 KP06 為止，擴散模型只會「隨便生成一筆像資料的東西」。實際用的系統要能聽指令：照著一張參考圖、生成指定的類別、或依照一句文字描述。這一節介紹三類做法：生成時直接改樣本的 ILVR、用分數把樣本推向條件的「引導」，以及把文字送進網路的 cross-attention。

This is the seventh knowledge point (KP07) of Week 14, "Deep generative models (3): diffusion models". Up to KP06, a diffusion model can only "generate some example that looks like the data". Real systems must follow instructions: follow a reference image, generate a given class, or match a text description. This note introduces three kinds of methods: ILVR, which edits the sample directly during generation; "guidance", which uses scores to push the sample towards the condition; and cross-attention, which feeds text into the network.

先備知識是本週 KP02 到 KP06：閉式、預測雜訊、演算法 2，以及第11週 Transformer 的注意力機制（query、key、value）。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第15章 KP-14（第 4 節）改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec14-kp07-figures.py` 算出，可以自己重算。

The prerequisites are this week's KP02 to KP06 (the closed form, noise prediction, Algorithm 2) and the attention mechanism of Week 11's Transformer (query, key, value). Every term and symbol is defined when it first appears. This note is rewritten from KP-14 (Section 4) of Chapter 15 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec14-kp07-figures.py`, so you can recompute it yourself.

## KP07 條件式擴散：ILVR、引導與 cross-attention（Conditional diffusion: ILVR, guidance and cross-attention）

### 學習目標（Learning objectives）

讀完之後，你能說明 ILVR 怎麼在生成的每一步把樣本的低頻換成參考圖的低頻，並手算一個小例子。你能說出「分數」是什麼，並說明它和網路猜的雜訊之間的關係。你能用貝氏定理寫出分類器引導，說明它需要什麼樣的分類器。你能寫出無分類器引導的公式，說明訓練時怎麼讓同一個網路同時學會有條件與無條件，並從一維例子讀出引導強度 $\gamma$ 的取捨。你也能說明 cross-attention 怎麼把文字條件送進去噪網路，並算一個小例子。

After reading, you can explain how ILVR replaces the sample's low frequencies with the reference image's at every generation step, and work a small example by hand. You can say what a "score" is and how it relates to the noise the network guesses. You can write classifier guidance with Bayes' rule and say what kind of classifier it needs. You can write the classifier-free guidance formula, explain how training makes one network learn both conditional and unconditional generation, and read the trade-off of the guidance strength $\gamma$ from a one-dimensional example. You can also explain how cross-attention feeds a text condition into the denoising network, and compute a small example.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $c$ | 條件，例如類別或一句文字 | the condition, such as a class or a sentence |
| $\varnothing$ | 「沒有條件」的空白記號 | the "no condition" null token |
| $\mathbf{y}$ | ILVR 的參考圖 | ILVR's reference image |
| $\phi_N(\cdot)$ | 低通濾波：先縮小 $N$ 倍再放大回原尺寸，只留下粗略的形狀 | low-pass filter: shrink by a factor $N$ and enlarge back, keeping only the coarse shape |
| $\nabla_{\mathbf{x}}\log p(\mathbf{x})$ | 分數（score）：對數機率對 $\mathbf{x}$ 的梯度，指向機率變大的方向 | the score: the gradient of the log probability with respect to $\mathbf{x}$, pointing towards higher probability |
| $q_t(\mathbf{x}_t)$ | 第 $t$ 步加雜訊後資料的分布 | the distribution of the data after $t$ noising steps |
| $p_\psi(c\mid\mathbf{x}_t)$ | 引導用的分類器，參數 $\psi$，輸入是有雜訊的 $\mathbf{x}_t$ | the guidance classifier with parameters $\psi$, taking noisy $\mathbf{x}_t$ as input |
| $\epsilon_\theta(\mathbf{x}_t,t,c)$ | 有條件的雜訊預測；$c=\varnothing$ 時是無條件 | conditional noise prediction; unconditional when $c=\varnothing$ |
| $\gamma$ | 引導強度 | the guidance strength |
| $Q,\ K,\ V$ | 注意力的 query、key、value（第11週） | attention's query, key and value (Week 11) |

### 1. ILVR：照著參考圖的輪廓生成（ILVR: generating along a reference image's outline）

**ILVR**（iterative latent variable refinement，迭代潛在變數精煉）不用重新訓練，只在生成時多做一件事。給一張參考圖 $\mathbf{y}$，每走一步逆向：

**ILVR** (iterative latent variable refinement) needs no retraining; it only adds one thing during generation. Given a reference image $\mathbf{y}$, at every backward step:

1. 照演算法 2 先走一步，得到暫時的 $\mathbf{x}'_{t-1}$。
2. 把參考圖也加到同樣多的雜訊：$\mathbf{y}_{t-1}=\sqrt{\bar\alpha_{t-1}}\,\mathbf{y}+\sqrt{1-\bar\alpha_{t-1}}\,\epsilon$，$\epsilon\sim\mathcal{N}(\mathbf{0},I)$ 是新抽的雜訊（KP02 的閉式）。
3. 把 $\mathbf{x}'_{t-1}$ 的低頻換成 $\mathbf{y}_{t-1}$ 的低頻：

1. Take one step of Algorithm 2 to get a temporary $\mathbf{x}'_{t-1}$.
2. Noise the reference image to the same level: $\mathbf{y}_{t-1}=\sqrt{\bar\alpha_{t-1}}\,\mathbf{y}+\sqrt{1-\bar\alpha_{t-1}}\,\epsilon$ with freshly drawn $\epsilon\sim\mathcal{N}(\mathbf{0},I)$ (KP02's closed form).
3. Replace the low frequencies of $\mathbf{x}'_{t-1}$ with those of $\mathbf{y}_{t-1}$:

$$
\mathbf{x}_{t-1}=\phi_N(\mathbf{y}_{t-1})+\mathbf{x}'_{t-1}-\phi_N(\mathbf{x}'_{t-1}).
$$

$\phi_N$ 是**低通濾波**：先把影像縮小 $N$ 倍（相鄰的像素取平均），再放大回原尺寸，只留下大塊的明暗與輪廓。$\mathbf{x}'-\phi_N(\mathbf{x}')$ 是生成樣本自己的細節（高頻）。所以結果是「參考圖的大輪廓，加上自己生成的細節」。

$\phi_N$ is a **low-pass filter**: shrink the image by a factor $N$ (averaging neighbouring pixels), then enlarge it back, keeping only the large patches of light and dark and the outline. $\mathbf{x}'-\phi_N(\mathbf{x}')$ is the generated sample's own detail (high frequency). So the result is "the reference image's large outline plus self-generated detail".

**手算一次** `[補充]`：用長度 $4$ 的向量當「影像」，$N=2$（每兩個相鄰的數取平均）。$\mathbf{x}'=(3,1,0,2)$、$\mathbf{y}=(0,2,4,4)$（為了好算，先不加雜訊）。$\phi_2(\mathbf{x}')=(2,2,1,1)$，$\phi_2(\mathbf{y})=(1,1,4,4)$，結果是 $(1,1,4,4)+(3,1,0,2)-(2,2,1,1)=(2,0,3,5)$。驗算：$\phi_2$ 結果 $=(1,1,4,4)$，正好是參考圖的低頻；而結果減去自己的低頻是 $(1,-1,-1,1)$，和 $\mathbf{x}'$ 的細節 $(3,1,0,2)-(2,2,1,1)=(1,-1,-1,1)$ 一樣。

**Compute once by hand** `[Added]`: use length-$4$ vectors as "images" with $N=2$ (average each pair of neighbours). $\mathbf{x}'=(3,1,0,2)$ and $\mathbf{y}=(0,2,4,4)$ (no noise, for easy arithmetic). $\phi_2(\mathbf{x}')=(2,2,1,1)$ and $\phi_2(\mathbf{y})=(1,1,4,4)$, so the result is $(1,1,4,4)+(3,1,0,2)-(2,2,1,1)=(2,0,3,5)$. Check: $\phi_2$ of the result $=(1,1,4,4)$, exactly the reference's low frequencies; and the result minus its own low frequencies is $(1,-1,-1,1)$, the same as the detail of $\mathbf{x}'$, $(3,1,0,2)-(2,2,1,1)=(1,-1,-1,1)$.

![三張並排的折線圖，橫軸是位置 0 到 63。每張都有一條彩色實線（訊號）和一條黑色階梯虛線（每 8 個點取平均的低通結果）。左圖紫色是生成的 x，一條彎彎曲曲、細節很多的訊號，低通階梯在 -1.3 到 2 之間跳動。中圖綠色是參考 y，一條平滑的正弦波加上小波紋，低通階梯先升後降、最後略為回升。右圖紅色是 phi_N(y) + x - phi_N(x)：它的黑色低通階梯和中圖的參考完全一樣，但紅線上的細小起伏來自左圖（Three line charts side by side, with position 0 to 63 horizontally. Each has a coloured solid line (the signal) and a black dashed staircase (the low-pass result averaging every 8 points). Left, purple, is the generated x, a wiggly signal with lots of detail, its low-pass staircase jumping between -1.3 and 2. Middle, green, is the reference y, a smooth sine wave with small ripples, its staircase rising, falling, then rising slightly at the end. Right, red, is phi_N(y) + x - phi_N(x): its black low-pass staircase is exactly the reference's, but the small wiggles on the red line come from the left chart)](../assets/figures/lec14-kp07-ilvr.png)

**圖說 `lec14-kp07-ilvr`**〔功能：一目了然〕：用長度 $64$ 的訊號、$N=8$ 示意一次替換 `[補充]`。先比中圖和右圖的黑色階梯：完全相同，表示結果的大輪廓跟著參考走。再比左圖和右圖彩色線上的小起伏：形狀一樣，表示細節保留了生成樣本自己的。在真正的 ILVR 裡，每一步都做這個替換，一路走下來，樣本就長成「輪廓像 $\mathbf{y}$、細節自己生成」的新圖。

**Caption `lec14-kp07-ilvr`** [Function: seeing it at a glance]: one replacement illustrated with a length-$64$ signal and $N=8$ `[Added]`. First compare the black staircases of the middle and right charts: identical, so the result's large outline follows the reference. Then compare the small wiggles on the coloured lines of the left and right charts: the same shape, so the detail stays the generated sample's own. Real ILVR does this replacement at every step, so the sample grows into a new image "with an outline like $\mathbf{y}$ and self-generated detail".

`[補充]` 原論文通常只在生成的前段做替換，從 $T$ 做到某個停止步就不再約束，讓最後幾步自由補細節。$N$ 是另一個控制鈕：$N$ 小，保留的低頻比較多，結果很像參考圖；$N$ 大，只有很粗的輪廓被固定，生成的自由度比較大。

`[Added]` The original paper usually applies the replacement only in the earlier part of generation, from $T$ down to some stopping step, leaving the last steps unconstrained to fill in detail. $N$ is another control knob: with a small $N$, more low frequencies are kept and the result looks much like the reference; with a large $N$, only a very coarse outline is fixed, leaving more freedom to generate.

常見錯誤：以為 ILVR 要訓練一個新的條件模型。它用的是原本無條件的 DDPM，只在生成時改樣本。

Common mistake: thinking ILVR trains a new conditional model. It uses the original unconditional DDPM and only edits the sample during generation.

### 2. 分數：雜訊預測的另一個名字（The score: another name for the noise prediction）

要講引導，先認識**分數**（score）：$\nabla_{\mathbf{x}}\log p(\mathbf{x})$，對數機率對 $\mathbf{x}$ 的梯度。它是一個箭頭，指向「往哪裡走，機率會變大」。`[補充]` 擴散模型猜的雜訊和分數幾乎是同一個東西：

To discuss guidance, first meet the **score**: $\nabla_{\mathbf{x}}\log p(\mathbf{x})$, the gradient of the log probability with respect to $\mathbf{x}$. It is an arrow pointing "which way to move to increase probability". `[Added]` The noise a diffusion model guesses is almost the same thing as the score:

$$
\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t)=-\frac{E[\epsilon\mid\mathbf{x}_t]}{\sqrt{1-\bar\alpha_t}}\approx-\frac{\epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{1-\bar\alpha_t}} .
$$

直覺：雜訊是「從資料被推出去的方向」，所以往反方向走，就是往資料（機率高的地方）走。用 KP01 的兩堆資料驗證：$t=100$、$x_t=0.5$ 時，直接微分 $q_t$ 的密度得到分數 $7.5884$，用 $-E[\epsilon\mid x_t]/\sqrt{1-\bar\alpha_t}$ 算也是 $7.5884$；在其他位置與 $t=300$ 也都一樣。這個關係在本週 KP08 會成為主角。

Intuition: the noise is "the direction the data were pushed out in", so moving the opposite way moves towards the data (where probability is high). Check it with KP01's two piles: at $t=100$ and $x_t=0.5$, differentiating the density of $q_t$ directly gives a score of $7.5884$, and $-E[\epsilon\mid x_t]/\sqrt{1-\bar\alpha_t}$ also gives $7.5884$; it agrees at other positions and at $t=300$ too. This relation becomes the main character in this week's KP08.

### 3. 分類器引導（Classifier guidance）

想生成條件 $c$ 的資料，要的是有條件的分數 $\nabla\log p(\mathbf{x}_t\mid c)$。貝氏定理 $p(\mathbf{x}\mid c)=p(c\mid\mathbf{x})\,p(\mathbf{x})/p(c)$ 取對數再對 $\mathbf{x}$ 微分，$p(c)$ 不含 $\mathbf{x}$ 就消失了：

To generate data with condition $c$, the conditional score $\nabla\log p(\mathbf{x}_t\mid c)$ is needed. Take Bayes' rule $p(\mathbf{x}\mid c)=p(c\mid\mathbf{x})\,p(\mathbf{x})/p(c)$, apply the log and differentiate with respect to $\mathbf{x}$; $p(c)$ does not contain $\mathbf{x}$ and disappears:

$$
\nabla\log p(\mathbf{x}_t\mid c)=\nabla\log p(\mathbf{x}_t)+\nabla\log p(c\mid\mathbf{x}_t).
$$

這裡的 $p(\mathbf{x}_t)$ 就是第2節的 $q_t$（加了雜訊的資料分布），$\nabla$ 都是對 $\mathbf{x}_t$ 微分。第一項是原本無條件模型的分數；第二項來自一個分類器 $p_\psi(c\mid\mathbf{x}_t)$：它的梯度告訴樣本「往哪裡走，會更像類別 $c$」。**分類器引導**（classifier guidance）把第二項乘上強度 $\gamma$，再用第2節的關係換回雜訊的寫法：

Here $p(\mathbf{x}_t)$ is Section 2's $q_t$ (the noised data distribution), and every $\nabla$ is with respect to $\mathbf{x}_t$. The first term is the score of the original unconditional model; the second comes from a classifier $p_\psi(c\mid\mathbf{x}_t)$, whose gradient tells the sample "which way to move to look more like class $c$". **Classifier guidance** multiplies the second term by a strength $\gamma$ and converts back to noise form with Section 2's relation:

$$
\hat\epsilon=\epsilon_\theta(\mathbf{x}_t,t)-\gamma\sqrt{1-\bar\alpha_t}\;\nabla_{\mathbf{x}_t}\log p_\psi(c\mid\mathbf{x}_t).
$$

再把 $\hat\epsilon$ 放進演算法 2 就行。`[補充]` 注意分類器的輸入是**有雜訊的** $\mathbf{x}_t$，所以要另外在各種雜訊程度的資料上訓練它；一般的影像分類器看到雜訊圖會亂猜，梯度沒有意義。這是分類器引導的主要麻煩。

Then put $\hat\epsilon$ into Algorithm 2. `[Added]` Note that the classifier's input is the **noisy** $\mathbf{x}_t$, so it must be trained separately on data at all noise levels; an ordinary image classifier guesses wildly on noisy images, and its gradient is meaningless. This is the main nuisance of classifier guidance.

常見錯誤：以為分類器引導可以直接拿一個現成的影像分類器來用。它要的是在有雜訊的 $\mathbf{x}_t$ 上訓練過的分類器。

Common mistake: thinking classifier guidance can use any off-the-shelf image classifier. It needs a classifier trained on noisy $\mathbf{x}_t$.

### 4. 無分類器引導（Classifier-free guidance）

**無分類器引導**（classifier-free guidance，CFG）省掉分類器。訓練時，網路多一個輸入 $c$；每一筆資料以一個小機率（常用 $10\%$ 到 $20\%$）把條件換成空白 $\varnothing$。這樣**同一個網路**同時學會有條件的 $\epsilon_\theta(\mathbf{x}_t,t,c)$ 與無條件的 $\epsilon_\theta(\mathbf{x}_t,t,\varnothing)$。生成時兩個都算，然後外插：

**Classifier-free guidance** (CFG) does away with the classifier. In training the network gets an extra input $c$; each example has its condition replaced by the null token $\varnothing$ with a small probability (commonly $10\%$ to $20\%$). Then **one network** learns both the conditional $\epsilon_\theta(\mathbf{x}_t,t,c)$ and the unconditional $\epsilon_\theta(\mathbf{x}_t,t,\varnothing)$. In generation both are computed and extrapolated:

$$
\tilde\epsilon=\epsilon_\theta(\mathbf{x}_t,t,\varnothing)+\gamma\,\big(\epsilon_\theta(\mathbf{x}_t,t,c)-\epsilon_\theta(\mathbf{x}_t,t,\varnothing)\big).
$$

$\gamma=0$ 是無條件，$\gamma=1$ 是一般的有條件生成，$\gamma>1$ 把「有條件比無條件多出的那一份」放大。`[補充]` 括號裡的差，按第2節換算，在網路猜得準時就等於 $-\sqrt{1-\bar\alpha_t}\,\nabla\log p(c\mid\mathbf{x}_t)$（因為 $\nabla\log p(c\mid\mathbf{x})=\nabla\log p(\mathbf{x}\mid c)-\nabla\log p(\mathbf{x})$），所以 CFG 等於用一個「藏在網路裡的分類器」做分類器引導。原論文把公式寫成 $(1+w)\,\epsilon_c-w\,\epsilon_\varnothing$，就是 $\gamma=1+w$。

$\gamma=0$ is unconditional, $\gamma=1$ is ordinary conditional generation, and $\gamma>1$ amplifies "the extra part that conditioning adds". `[Added]` By Section 2's conversion, the difference in the bracket equals (for ideal noise predictions) $-\sqrt{1-\bar\alpha_t}\,\nabla\log p(c\mid\mathbf{x}_t)$ (because $\nabla\log p(c\mid\mathbf{x})=\nabla\log p(\mathbf{x}\mid c)-\nabla\log p(\mathbf{x})$), so CFG equals classifier guidance with a "classifier hidden inside the network". The original paper writes the formula as $(1+w)\,\epsilon_c-w\,\epsilon_\varnothing$, which is $\gamma=1+w$.

**一維實驗** `[補充]`：兩個互相重疊的類別，「右」是 $\mathcal{N}(1,0.8^2)$，「左」是 $\mathcal{N}(-1,0.8^2)$，無條件是兩者各半。資料簡單，有條件與無條件的完美雜訊猜測都算得出來；用 DDPM 排程跑演算法 2，條件設為「右」，比較不同的 $\gamma$，各生成 $20000$ 個樣本。

**A one-dimensional experiment** `[Added]`: two overlapping classes, "right" is $\mathcal{N}(1,0.8^2)$ and "left" is $\mathcal{N}(-1,0.8^2)$, with the unconditional distribution half of each. The data are simple, so the perfect conditional and unconditional noise guesses can both be computed; run Algorithm 2 with the DDPM schedule, condition on "right", and compare different $\gamma$, with $20000$ samples each.

![四張並排的直方圖，橫軸 x0 從 -3.5 到 3.5，每張都有一條黑色虛線畫出類別 c 的密度 N(1, 0.8 平方)。第一張「無條件」：淡紫色直方圖是兩座低矮相連的山，x0>0 佔 49.8%，平均 -0.01、標準差 1.29。第二張「有條件，gamma=1」：直方圖和黑色虛線重合，x0>0 佔 89.3%，平均 0.99、標準差 0.80。第三張「引導，gamma=3」：直方圖往右移、變窄變高，x0>0 佔 99.9%，平均 1.54、標準差 0.62。第四張「引導，gamma=7」：再往右移，x0>0 佔 100%，平均 1.87、標準差 0.60（Four histograms side by side, with x0 from -3.5 to 3.5 and a black dashed line for the class-c density N(1, 0.8 squared) in each. First, "unconditional": the light-purple histogram is two low, joined hills, with 49.8% at x0 > 0, mean -0.01 and standard deviation 1.29. Second, "conditional, gamma = 1": the histogram coincides with the dashed line, 89.3% at x0 > 0, mean 0.99, standard deviation 0.80. Third, "guided, gamma = 3": the histogram moves right and becomes narrower and taller, 99.9% at x0 > 0, mean 1.54, standard deviation 0.62. Fourth, "guided, gamma = 7": further right, 100% at x0 > 0, mean 1.87, standard deviation 0.60)](../assets/figures/lec14-kp07-guidance.png)

**圖說 `lec14-kp07-guidance`**〔功能：對照、精準度〕：由左到右，$\gamma$ 從無條件加到 $7$。$\gamma=1$ 時，生成的分布正好是「右」這一類：有 $89.3\%$ 大於 $0$（理論值 $89.4\%$），因為兩類重疊，本來就有一成樣本落在左邊。$\gamma=3$ 與 $\gamma=7$ 時，樣本被推離「左」那一類：幾乎全部大於 $0$，平均移到 $1.54$、$1.87$，標準差縮到 $0.62$、$0.60$。代價是它們已經不是真正的 $\mathcal{N}(1,0.8^2)$：更「典型」，但多樣性變少。

**Caption `lec14-kp07-guidance`** [Function: comparison, precision]: from left to right, $\gamma$ grows from unconditional to $7$. At $\gamma=1$ the generated distribution is exactly the "right" class: $89.3\%$ are above $0$ (theory: $89.4\%$), because the classes overlap and a tenth of the samples naturally fall on the left. At $\gamma=3$ and $\gamma=7$ the samples are pushed away from the "left" class: almost all are above $0$, the mean moves to $1.54$ and $1.87$, and the standard deviation shrinks to $0.62$ and $0.60$. The price is that they are no longer the true $\mathcal{N}(1,0.8^2)$: more "typical", but less diverse.

這就是影像系統裡常見的取捨：$\gamma$ 調大，圖更符合文字描述、更「到位」，但每張越來越像、細節變少，太大還會過度飽和。實務上 $\gamma$ 依系統而定，例如 Stable Diffusion 的預設約是 $7.5$。

This is the common trade-off in image systems: a larger $\gamma$ makes images fit the text better and more "on point", but they become more alike with less detail, and too large a value oversaturates them. In practice $\gamma$ depends on the system; Stable Diffusion's default, for example, is about $7.5$.

常見錯誤：以為 $\gamma$ 越大，生成的就越接近「真正的那一類」。$\gamma=1$ 才是那一類的分布；$\gamma>1$ 刻意偏離它，換取更明顯的條件特徵。

Common mistake: thinking that the larger $\gamma$, the closer the samples are to "the true class". $\gamma=1$ gives the class distribution; $\gamma>1$ deliberately departs from it to gain more pronounced conditional features.

### 5. Cross-attention：把文字送進網路（Cross-attention: feeding text into the network）

文字生成影像的系統（例如 Stable Diffusion）裡，條件 $c$ 是一整句話。做法是：先用一個文字編碼器把句子變成一串向量（每個詞元一個），再在去噪網路（通常是 `U-Net`）的許多層加入**交叉注意力**（cross-attention）：**query 來自影像的特徵，key 和 value 來自文字的向量**。每個影像位置依照自己的 query，去決定要多看哪幾個詞。

In text-to-image systems (such as Stable Diffusion), the condition $c$ is a whole sentence. The method: a text encoder first turns the sentence into a sequence of vectors (one per token), and **cross-attention** layers are added at many layers of the denoising network (usually a `U-Net`): **the queries come from the image features, while the keys and values come from the text vectors**. Each image position uses its own query to decide which words to attend to.

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V .
$$

這就是第11週的注意力，只是 $Q$ 和 $K,V$ 來自不同的地方。**算一個小例子** `[補充]`：一個影像位置的 query $\mathbf{q}=(1,0)$，三個詞元的 key 是 $(1,0)$、$(0,1)$、$(0.6,0.8)$，value 是 $(1,0)$、$(0,1)$、$(0.5,0.5)$，$d=2$。分數 $\mathbf{q}\cdot\mathbf{k}/\sqrt2$ 是 $0.7071$、$0$、$0.4243$；softmax 後的權重是 $0.445$、$0.220$、$0.335$；輸出是權重乘 value 再相加，$(0.613,\ 0.387)$。這個位置最「注意」第一個詞，因為它的 key 和 query 最像。

This is Week 11's attention, except that $Q$ and $K,V$ come from different places. **A small example** `[Added]`: one image position has query $\mathbf{q}=(1,0)$; three tokens have keys $(1,0)$, $(0,1)$ and $(0.6,0.8)$, and values $(1,0)$, $(0,1)$ and $(0.5,0.5)$, with $d=2$. The scores $\mathbf{q}\cdot\mathbf{k}/\sqrt2$ are $0.7071$, $0$ and $0.4243$; after softmax the weights are $0.445$, $0.220$ and $0.335$; the output is the weighted sum of the values, $(0.613,\ 0.387)$. This position "attends" most to the first token, because its key is most like the query.

`[補充]` Stable Diffusion 還有一個重要設計：它不在原始像素上擴散，而是先用一個自編碼器（第12週）把影像壓成較小的潛在表示，在潛在空間裡做擴散，最後再解碼回影像，所以叫**潛在擴散模型**（latent diffusion）。這讓計算量小很多。文字條件通常也搭配第4節的無分類器引導。

`[Added]` Stable Diffusion has another key design: it does not diffuse on raw pixels; an autoencoder (Week 12) first compresses the image into a smaller latent representation, diffusion runs in that latent space, and the result is decoded back into an image, hence the name **latent diffusion**. This greatly reduces computation. The text condition is usually combined with Section 4's classifier-free guidance as well.

常見錯誤：以為文字條件是「直接加到像素上」。文字是透過 cross-attention 影響網路中間的特徵，網路再據此猜雜訊。

Common mistake: thinking the text condition is "added directly to the pixels". The text affects the network's intermediate features through cross-attention, and the network guesses the noise based on them.

### 本單元重點整理（KP07 summary）

**一句話重點**：條件式擴散有三類做法：ILVR 在生成時把樣本的低頻換成參考圖的；分類器引導與無分類器引導利用「分數 $\approx-\epsilon_\theta/\sqrt{1-\bar\alpha_t}$」把樣本推向條件，CFG 用同一個網路的有條件與無條件預測外插 $\tilde\epsilon=\epsilon_\varnothing+\gamma(\epsilon_c-\epsilon_\varnothing)$；文字則透過 cross-attention 送進去噪網路。

**One-sentence summary**: conditional diffusion has three kinds of methods: ILVR replaces the sample's low frequencies with the reference image's during generation; classifier guidance and classifier-free guidance use "score $\approx-\epsilon_\theta/\sqrt{1-\bar\alpha_t}$" to push the sample towards the condition, with CFG extrapolating the same network's conditional and unconditional predictions as $\tilde\epsilon=\epsilon_\varnothing+\gamma(\epsilon_c-\epsilon_\varnothing)$; and text enters the denoising network through cross-attention.

**白話說明**：ILVR 像描圖：底稿的大輪廓照描，細節自己畫。引導像一位老師在旁邊說「再像一點」：分類器引導是另外請一位鑑定師打分數，無分類器引導是讓畫家自己比較「照題目畫」和「隨便畫」差在哪，再把差別放大。cross-attention 像畫家邊畫邊回頭看題目上的每個字，決定這一塊要聽哪幾個字的。

**Plain explanation**: ILVR is like tracing: copy the draft's large outline and draw the details yourself. Guidance is like a teacher beside you saying "a bit more like it": classifier guidance hires a separate judge to score, while classifier-free guidance lets the painter compare "painting to the brief" with "painting freely" and amplifies the difference. Cross-attention is like the painter glancing back at each word of the brief while painting, deciding which words this patch should follow.

**正式定義**：ILVR：$\mathbf{x}_{t-1}=\phi_N(\mathbf{y}_{t-1})+\mathbf{x}'_{t-1}-\phi_N(\mathbf{x}'_{t-1})$，$\mathbf{y}_{t-1}=\sqrt{\bar\alpha_{t-1}}\mathbf{y}+\sqrt{1-\bar\alpha_{t-1}}\epsilon$。分數與雜訊：$\nabla\log q_t(\mathbf{x}_t)=-E[\epsilon\mid\mathbf{x}_t]/\sqrt{1-\bar\alpha_t}$。分類器引導：$\hat\epsilon=\epsilon_\theta-\gamma\sqrt{1-\bar\alpha_t}\nabla\log p_\psi(c\mid\mathbf{x}_t)$。無分類器引導：$\tilde\epsilon=\epsilon_\theta(\mathbf{x}_t,t,\varnothing)+\gamma(\epsilon_\theta(\mathbf{x}_t,t,c)-\epsilon_\theta(\mathbf{x}_t,t,\varnothing))$。cross-attention：$\mathrm{softmax}(QK^\top/\sqrt d)V$，$Q$ 來自影像特徵、$K,V$ 來自文字。

**Formal definition**: ILVR: $\mathbf{x}_{t-1}=\phi_N(\mathbf{y}_{t-1})+\mathbf{x}'_{t-1}-\phi_N(\mathbf{x}'_{t-1})$ with $\mathbf{y}_{t-1}=\sqrt{\bar\alpha_{t-1}}\mathbf{y}+\sqrt{1-\bar\alpha_{t-1}}\epsilon$. Score and noise: $\nabla\log q_t(\mathbf{x}_t)=-E[\epsilon\mid\mathbf{x}_t]/\sqrt{1-\bar\alpha_t}$. Classifier guidance: $\hat\epsilon=\epsilon_\theta-\gamma\sqrt{1-\bar\alpha_t}\nabla\log p_\psi(c\mid\mathbf{x}_t)$. Classifier-free guidance: $\tilde\epsilon=\epsilon_\theta(\mathbf{x}_t,t,\varnothing)+\gamma(\epsilon_\theta(\mathbf{x}_t,t,c)-\epsilon_\theta(\mathbf{x}_t,t,\varnothing))$. Cross-attention: $\mathrm{softmax}(QK^\top/\sqrt d)V$, with $Q$ from image features and $K,V$ from text.

**例子**：ILVR 手算 $(3,1,0,2)$ 與參考 $(0,2,4,4)$、$N=2$，得 $(2,0,3,5)$。一維 CFG：$\gamma=1$ 時 $89.3\%$ 樣本大於 $0$、平均 $0.99$、標準差 $0.80$；$\gamma=7$ 時 $100\%$、平均 $1.87$、標準差 $0.60$。cross-attention 小例子的權重 $0.445$、$0.220$、$0.335$。

**Example**: ILVR by hand with $(3,1,0,2)$, reference $(0,2,4,4)$ and $N=2$ gives $(2,0,3,5)$. One-dimensional CFG: at $\gamma=1$, $89.3\%$ of samples are above $0$, with mean $0.99$ and standard deviation $0.80$; at $\gamma=7$, $100\%$, mean $1.87$, standard deviation $0.60$. The cross-attention example has weights $0.445$, $0.220$ and $0.335$.

**常見迷思**：

- 「ILVR 要訓練新模型。」錯，它只在生成時改樣本。
- 「分類器引導可以用現成的影像分類器。」錯，分類器要在有雜訊的 $\mathbf{x}_t$ 上訓練。
- 「CFG 要兩個網路。」錯，同一個網路，靠隨機丟掉條件同時學會兩種。
- 「$\gamma$ 越大越接近真正的類別。」錯，$\gamma=1$ 才是；$\gamma>1$ 更典型但多樣性變少。
- 「文字條件直接加到像素上。」錯，透過 cross-attention 影響網路中間的特徵。

**Common misconceptions**:

- "ILVR trains a new model." Wrong: it only edits the sample during generation.
- "Classifier guidance can use an off-the-shelf image classifier." Wrong: the classifier must be trained on noisy $\mathbf{x}_t$.
- "CFG needs two networks." Wrong: one network learns both by randomly dropping the condition.
- "The larger $\gamma$, the closer to the true class." Wrong: $\gamma=1$ is; $\gamma>1$ is more typical but less diverse.
- "The text condition is added directly to the pixels." Wrong: it acts on the network's intermediate features through cross-attention.

**圖**：本單元有兩張圖。

- `lec14-kp07-ilvr`：一次低頻替換：結果的輪廓跟參考圖、細節跟生成樣本。
- `lec14-kp07-guidance`：一維條件生成在不同引導強度 $\gamma$ 下的樣本分布。

**Figures** — this unit has two figures.

- `lec14-kp07-ilvr`: one low-frequency replacement: the result's outline follows the reference and its detail follows the generated sample.
- `lec14-kp07-guidance`: the sample distribution of one-dimensional conditional generation under different guidance strengths $\gamma$.

### 小練習（Practice）

1. ILVR 手算：$\mathbf{x}'=(1,3,2,2)$、$\mathbf{y}=(4,0,1,1)$、$N=2$。算 $\phi_2(\mathbf{x}')$、$\phi_2(\mathbf{y})$ 與替換後的結果，並驗算結果的低頻等於參考的低頻。
2. 某一步網路給 $\epsilon_\theta(x_t,t,\varnothing)=0.4$、$\epsilon_\theta(x_t,t,c)=0.1$。算 $\gamma=0$、$1$、$3$ 時的 $\tilde\epsilon$。$\gamma=3$ 時往哪個方向「多推」了？
3. 為什麼分類器引導的分類器必須在有雜訊的 $\mathbf{x}_t$ 上訓練？如果只在乾淨影像上訓練，生成前段（$t$ 很大）會發生什麼事？
4. 根據第4節的實驗，把 $\gamma$ 從 $1$ 調到 $7$，「大於 $0$ 的比例」與「標準差」各怎麼變？用「條件符合度」與「多樣性」說明這個取捨。

1. ILVR by hand: $\mathbf{x}'=(1,3,2,2)$, $\mathbf{y}=(4,0,1,1)$, $N=2$. Compute $\phi_2(\mathbf{x}')$, $\phi_2(\mathbf{y})$ and the result after replacement, and check that the result's low frequencies equal the reference's.
2. At some step the network gives $\epsilon_\theta(x_t,t,\varnothing)=0.4$ and $\epsilon_\theta(x_t,t,c)=0.1$. Compute $\tilde\epsilon$ for $\gamma=0$, $1$ and $3$. In which direction does $\gamma=3$ "push extra"?
3. Why must classifier guidance's classifier be trained on noisy $\mathbf{x}_t$? If it were trained only on clean images, what would happen early in generation (large $t$)?
4. From Section 4's experiment, as $\gamma$ goes from $1$ to $7$, how do "the share above $0$" and "the standard deviation" change? Explain the trade-off in terms of "fit to the condition" and "diversity".

**第 1 題解答** `[補充]`：$\phi_2(\mathbf{x}')=(2,2,2,2)$，$\phi_2(\mathbf{y})=(2,2,1,1)$。結果 $=(2,2,1,1)+(1,3,2,2)-(2,2,2,2)=(1,3,1,1)$。驗算：$\phi_2(1,3,1,1)=(2,2,1,1)$，等於參考的低頻。

**Solution to question 1** `[Added]`: $\phi_2(\mathbf{x}')=(2,2,2,2)$ and $\phi_2(\mathbf{y})=(2,2,1,1)$. The result is $(2,2,1,1)+(1,3,2,2)-(2,2,2,2)=(1,3,1,1)$. Check: $\phi_2(1,3,1,1)=(2,2,1,1)$, equal to the reference's low frequencies.

**第 2 題解答** `[補充]`：$\tilde\epsilon=0.4+\gamma(0.1-0.4)$：$\gamma=0$ 得 $0.4$（無條件），$\gamma=1$ 得 $0.1$（有條件），$\gamma=3$ 得 $-0.5$。有條件比無條件少猜了 $0.3$ 的雜訊，$\gamma=3$ 把這個差放大三倍，所以猜的雜訊更往負方向。$\epsilon_c-\epsilon_\varnothing<0$ 表示 $\nabla\log p(c\mid x_t)>0$（第4節的換算），也就是條件 $c$ 在 $x$ 變大的方向；演算法 2 是從 $x_t$ 扣掉雜訊，扣掉一個更負的數，樣本就往 $+x$、也就是往 $c$ 移。

**Solution to question 2** `[Added]`: $\tilde\epsilon=0.4+\gamma(0.1-0.4)$: $\gamma=0$ gives $0.4$ (unconditional), $\gamma=1$ gives $0.1$ (conditional), and $\gamma=3$ gives $-0.5$. The conditional prediction guesses $0.3$ less noise than the unconditional one; $\gamma=3$ triples that difference, so the guessed noise moves further negative. $\epsilon_c-\epsilon_\varnothing<0$ means $\nabla\log p(c\mid x_t)>0$ (Section 4's conversion), so condition $c$ lies in the direction of increasing $x$; Algorithm 2 subtracts the noise from $x_t$, and subtracting a more negative number moves the sample towards $+x$, that is, towards $c$.

**第 3 題解答** `[補充]`：生成時分類器看到的是 $\mathbf{x}_t$，前段幾乎是純雜訊。只在乾淨影像上訓練的分類器從沒看過這種輸入，給的機率和梯度沒有意義，引導會把樣本推往錯誤或隨機的方向；而前段正是決定大結構（例如是哪一類）的時候，所以影響最大。

**Solution to question 3** `[Added]`: during generation the classifier sees $\mathbf{x}_t$, which is almost pure noise early on. A classifier trained only on clean images has never seen such input, so its probabilities and gradients are meaningless, and guidance pushes the sample in wrong or random directions; and the early stage is exactly when the large structure (for example, which class) is decided, so the damage is greatest.

**第 4 題解答** `[補充]`：大於 $0$ 的比例從 $89.3\%$ 升到 $100\%$，標準差從 $0.80$ 降到 $0.60$（平均從 $0.99$ 移到 $1.87$）。比例升高代表樣本更明確地屬於「右」類（條件符合度高），標準差變小代表樣本彼此更像（多樣性低）。$\gamma=1$ 才是這一類真正的分布；要更「到位」就得犧牲多樣性。

**Solution to question 4** `[Added]`: the share above $0$ rises from $89.3\%$ to $100\%$, and the standard deviation falls from $0.80$ to $0.60$ (the mean moves from $0.99$ to $1.87$). The rising share means samples belong more clearly to the "right" class (better fit to the condition); the smaller standard deviation means samples resemble each other more (less diversity). $\gamma=1$ is the class's true distribution; being more "on point" costs diversity.

### 小測驗（Quiz，來源原題）

來源 KP-14 有四題自我檢測，保留原題與全部選項；來源頁面附有答案與簡短解析，下面的解析由本講義依各節內容補寫。

The source's KP-14 has four self-check questions, kept here with all their options; the source page gives the answers with short explanations, and the explanations below are written by this note from its own sections.

1. ILVR 用什麼方式控制生成結果？
   - (A) 替換高頻資訊
   - (B) 替換低頻資訊以符合參考圖
   - (C) 修改訓練 loss
   - (D) 增加時間步
2. Classifier-Free Guidance (CFG) 不需要什麼？
   - (A) 條件資訊
   - (B) 額外分類器
   - (C) 網路
   - (D) noise schedule
3. 在 text-to-image diffusion 中，文字條件通常如何注入？
   - (A) 直接加到 pixel
   - (B) Cross-attention
   - (C) 替換 covariance
   - (D) 改變 optimizer
4. ILVR 中的 $n$ 代表什麼？
   - (A) 時間步總數
   - (B) 低通濾波器的下採樣倍率
   - (C) batch size
   - (D) noise level

1. How does ILVR control the generated result?
   - (A) By replacing high-frequency information
   - (B) By replacing low-frequency information to match the reference image
   - (C) By changing the training loss
   - (D) By adding time steps
2. What does Classifier-Free Guidance (CFG) not need?
   - (A) Condition information
   - (B) An extra classifier
   - (C) A network
   - (D) A noise schedule
3. In text-to-image diffusion, how is the text condition usually injected?
   - (A) Added directly to the pixels
   - (B) Cross-attention
   - (C) Replacing the covariance
   - (D) Changing the optimizer
4. What does $n$ stand for in ILVR?
   - (A) The total number of time steps
   - (B) The downsampling factor of the low-pass filter
   - (C) The batch size
   - (D) The noise level

**解析** `[補充]`：

- 第 1 題選 (B)：第1節。高頻（細節）留著用生成樣本自己的，所以 (A) 剛好相反。
- 第 2 題選 (B)：第4節；它仍然需要條件資訊和網路。
- 第 3 題選 (B)：第5節。
- 第 4 題選 (B)：來源寫作 $\phi_n$，本講義寫作 $\phi_N$；$N$ 越大，保留的只有越粗的輪廓。

**Explanations** `[Added]`:

- Question 1: B: Section 1. The high frequencies (detail) stay the generated sample's own, so A is backwards.
- Question 2: B: Section 4; it still needs the condition information and a network.
- Question 3: B: Section 5.
- Question 4: B: the source writes $\phi_n$ and this note $\phi_N$; the larger $N$, the coarser the outline that is kept.

### 留給下一單元的問題（The question for the next unit）

第2節發現，擴散模型猜的雜訊其實就是分數。如果把步數 $T$ 增加到無限多，每一步無限小，前向與逆向過程會變成什麼？「分數」在其中扮演什麼角色？擴散模型和 VAE、GAN 比起來，到底好在哪、缺點在哪？下一個知識點（本週 KP08，選讀）用隨機微分方程與 score matching 看擴散模型，並做全章總結。

Section 2 found that the noise a diffusion model guesses is really the score. If the number of steps $T$ grows without limit, with each step infinitely small, what do the forward and backward processes become? What role does the "score" play there? Compared with VAEs and GANs, what exactly are the strengths and weaknesses of diffusion models? The next knowledge point (this week's KP08, optional) views diffusion models through stochastic differential equations and score matching, and sums up the chapter.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第15章「Diffusion Models」的 KP-14「Conditional Diffusion：ILVR、Classifier Guidance 與 Cross-Attention」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch15-diffusion-models/index.html ，讀取日期 2026-11-15）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from KP-14, "Conditional diffusion: ILVR, classifier guidance and cross-attention", of Chapter 15, "Diffusion Models", of CCC-LNS IDL_115 (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch15-diffusion-models/index.html , accessed 2026-11-15). Content not in the source is marked `[Added]` where it appears.

- 來源的互動展示寫 $s_{\text{CFG}}=w\,\nabla\log p(x\mid c)+(1-w)\,\nabla\log p(x)$，其中 $w=1$ 是一般的有條件生成；原論文（Ho 與 Salimans）把公式寫成 $(1+w)\,\epsilon_c-w\,\epsilon_\varnothing$，那裡 $w=0$ 才是一般的有條件生成。兩者只差 $1$；本講義統一用 $\gamma$（$\gamma=1$ 為有條件），並在第4節說明對應。
- 來源說分類器引導「用一個額外訓練的分類器分數來引導」，沒有說明分類器必須在有雜訊的 $\mathbf{x}_t$ 上訓練；本講義第3節與小練習第 3 題補上。
- 來源說 CFG「把兩者分數做外插，增強條件強度」，沒有提到代價；本講義第4節以實驗說明 $\gamma>1$ 會犧牲多樣性、且不再是該類的真正分布。
- 本講義補上：ILVR 的三個步驟與手算例子、低頻替換的示意圖、$N$ 的作用、分數的定義與「分數 $=-$ 雜訊 $/\sqrt{1-\bar\alpha_t}$」的數值驗證、分類器引導的推導與雜訊寫法、CFG 的訓練方式與它等於隱藏分類器的說明、一維條件生成的引導實驗、cross-attention 的小例子、潛在擴散的簡介、四題小練習與解答、小測驗的解析。
- 取捨：來源的互動展示與 Algorithm 3 的虛擬碼不重做，改寫成第1節的條列步驟；符號加粗與否的取捨同 KP02。

- The source's interactive demo writes $s_{\text{CFG}}=w\,\nabla\log p(x\mid c)+(1-w)\,\nabla\log p(x)$, where $w=1$ is ordinary conditional generation; the original paper (Ho and Salimans) writes $(1+w)\,\epsilon_c-w\,\epsilon_\varnothing$, where $w=0$ is ordinary conditional generation. The two differ by $1$; this note uses $\gamma$ throughout ($\gamma=1$ is conditional) and explains the correspondence in Section 4.
- The source says classifier guidance "guides with the score of an extra trained classifier" without saying the classifier must be trained on noisy $\mathbf{x}_t$; Section 3 and practice question 3 add this.
- The source says CFG "extrapolates the two scores to strengthen the condition" without mentioning the cost; Section 4 shows by experiment that $\gamma>1$ sacrifices diversity and is no longer the class's true distribution.
- This note adds: the three steps of ILVR with a hand example; the low-frequency replacement figure; the role of $N$; the definition of the score and a numerical check of "score $=-$ noise $/\sqrt{1-\bar\alpha_t}$"; the derivation of classifier guidance and its noise form; how CFG is trained and why it equals a hidden classifier; the one-dimensional conditional guidance experiment; the cross-attention example; a brief introduction to latent diffusion; four practice questions with solutions; and the explanations of the quiz.
- Choices: the source's interactive demo and the pseudocode of Algorithm 3 are not rebuilt; they are rewritten as the numbered steps of Section 1. Bold versus plain symbols follow the choice made in KP02.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 條件式擴散 | conditional diffusion |
| 迭代潛在變數精煉 | iterative latent variable refinement (ILVR) |
| 低通濾波 | low-pass filter |
| 下採樣倍率 | downsampling factor |
| 低頻、高頻 | low frequency, high frequency |
| 分數（$\nabla\log p$） | score |
| 外插 | extrapolate |
| 分類器引導 | classifier guidance |
| 無分類器引導 | classifier-free guidance (CFG) |
| 引導強度 | guidance strength |
| 空白條件 | null condition (null token) |
| 多樣性 | diversity |
| 交叉注意力 | cross-attention |
| 詞元 | token |
| 文字編碼器 | text encoder |
| U 形網路 | `U-Net` |
| 潛在擴散模型 | latent diffusion model |
| 自編碼器（第12週） | autoencoder |
| 去雜訊擴散機率模型 | denoising diffusion probabilistic model (DDPM) |
| 貝氏定理 | Bayes' rule |
| 生成對抗網路（第13週） | generative adversarial network (GAN) |
| 變分自編碼器（第12週） | variational autoencoder (VAE) |
