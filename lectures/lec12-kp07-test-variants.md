# 第12週 KP07：使用 VAE，重建與生成、為什麼會模糊、變體與它在今天的位置（Using a VAE: reconstruction and generation, why it blurs, its variants and where it stands today）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第12週的第七個、也是最後一個知識點（KP07），接在 KP06 之後。KP06 把 VAE 訓練好了。這一節談訓練好以後怎麼用它，為什麼它畫出來的影像常常偏模糊，有哪些改良的版本，以及它在今天的生成模型裡站在哪裡。

This is the seventh and last knowledge point (KP07) of Week 12, following KP06. KP06 trained the VAE. This note covers how to use it once trained, why the images it draws are often blurry, which improved versions exist, and where it stands among today's generative models.

先備知識是 KP05 的 VAE 架構與 KP06 的訓練。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第12章 §3.5、§3.6、§4 與教授祕笈改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。圖與實驗的數字都由 `assets/figures/src/lec12-kp07-figures.py` 算出（它用到 KP05 的 `lec12_vae_toy.py`），可以自己重算。

The prerequisites are KP05's VAE architecture and KP06's training. Every term and symbol is defined when it first appears. This note is rewritten from §3.5, §3.6 and §4 of Chapter 12 of the course's web material CCC-LNS and its professor's notes, which is what "the source" means below; content added by this note is marked `[Added]`. The figures and experimental numbers are computed by `assets/figures/src/lec12-kp07-figures.py` (which uses KP05's `lec12_vae_toy.py`), so you can recompute them yourself.

## KP07 使用 VAE，重建與生成、為什麼會模糊、變體與它在今天的位置（Using a VAE: reconstruction and generation, why it blurs, its variants and where it stands today）

### 學習目標（Learning objectives）

讀完之後，你能分辨 VAE 的兩種用法：重建要用編碼器和解碼器，生成只用解碼器；並說出用重建誤差做異常偵測的道理。你能用一個小例子說明，以平方誤差訓練的解碼器為什麼偏向輸出「平均」，所以影像模糊。你能說出 $\beta$-VAE 等變體各想解決什麼，以及調大 $\beta$ 的代價。你也能說出 VAE 的主要優點與限制，以及它和擴散模型、下週 GAN 的關係。

After reading, you can tell apart the VAE's two uses: reconstruction uses the encoder and decoder, generation uses only the decoder; and you can explain why reconstruction error detects anomalies. Using a small example, you can explain why a decoder trained with squared error leans towards outputting an "average", which blurs images. You can say what $\beta$-VAE and the other variants each try to fix, and what raising $\beta$ costs. You can also state the VAE's main strengths and limits, and how it relates to diffusion models and next week's GAN.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{x},\ \hat{\mathbf{x}}$ | 輸入資料與解碼器的輸出 | input data and the decoder's output |
| $\mathbf{z}$ | 潛在變數（編碼） | latent variable (code) |
| $q(\mathbf{z}\mid\mathbf{x})$ | 編碼器給出的近似後驗 | approximate posterior given by the encoder |
| $p(\mathbf{z})=\mathcal{N}(\mathbf{0},I)$ | 先驗：標準常態分布 | prior: the standard normal |
| $\mu$ | 編碼器輸出的平均 | mean output by the encoder |
| $\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2$ | 平方重建誤差 | squared reconstruction error |
| $\beta$ | KL 項的權重 | weight of the KL term |
| $\mathbb{E}[\cdot]$ | 期望值（按機率加權的平均） | expectation (probability-weighted average) |
| $\sigma$ | 標準差 | standard deviation |
| $D_{\text{KL}}$ | KL 散度（KP02） | KL divergence (KP02) |
| $\arg\min_{c}f(c)$ | 讓 $f$ 最小的那個 $c$ | the $c$ that makes $f$ smallest |

### 1. 測試階段：重建與生成（The test phase: reconstruction and generation）

來源把訓練好的 VAE 分成兩種用法。**重建**（reconstruction）：把一筆資料 $\mathbf{x}$ 送進編碼器得到 $q(\mathbf{z}\mid\mathbf{x})$，取一個 $\mathbf{z}$，解碼器畫回 $\hat{\mathbf{x}}$，再比較兩者。**生成**（generation）：從先驗 $\mathcal{N}(\mathbf{0},I)$ 直接抽 $\mathbf{z}$，交給解碼器，完全不用編碼器。KP05 第4節已經做過生成。

The source divides the trained VAE into two uses. **Reconstruction**: feed an example $\mathbf{x}$ to the encoder to get $q(\mathbf{z}\mid\mathbf{x})$, take a $\mathbf{z}$, let the decoder draw $\hat{\mathbf{x}}$ back, and compare the two. **Generation**: draw $\mathbf{z}$ directly from the prior $\mathcal{N}(\mathbf{0},I)$ and hand it to the decoder, with no encoder at all. KP05 Section 4 already did generation.

祕笈提醒兩者不能互換：要重建一筆特定的資料，就必須用 $q(\mathbf{z}\mid\mathbf{x})$；只要「看起來像資料」，從先驗抽就夠了。生成之所以行得通，來源的解釋是 KL 項把每個 $q(\mathbf{z}\mid\mathbf{x})$ 拉向先驗，所以從先驗抽的 $\mathbf{z}$，看起來像編碼器會給的 $\mathbf{z}$。

The professor's notes warn that the two are not interchangeable: to reconstruct a particular example you must use $q(\mathbf{z}\mid\mathbf{x})$; if you only want something that "looks like data", drawing from the prior is enough. Generation works, the source explains, because the KL term pulls each $q(\mathbf{z}\mid\mathbf{x})$ towards the prior, so a $\mathbf{z}$ drawn from the prior looks like one the encoder would give.

重建的一個用途是**異常偵測**（anomaly detection）。來源說，重建誤差大的資料可以當成異常。用 KP05 的小 VAE 試試 `[補充]`（亂數種子與 KP05 相同：資料 $1$、初始權重 $2$），重建時取編碼器的平均當 $z$（不抽樣，誤差才固定、可重現）：

One use of reconstruction is **anomaly detection**. The source says an example with a large reconstruction error can be treated as an anomaly. Try it with KP05's small VAE `[Added]` (same random seeds as KP05: $1$ for the data, $2$ for the initial weights), taking the encoder's mean as $z$ when reconstructing (no sampling, so the error is fixed and repeatable):

| 輸入 | $z$ | $\hat{\mathbf{x}}$ | 平方誤差 |
|---|---|---|---|
| $300$ 個訓練點 | | | 中位數 $0.0026$，最大 $0.027$ |
| 第一段圓弧上的新點（$60^\circ$） | $0.834$ | $(0.563,\ 0.843)$ | $0.0044$ |
| 圓心 $(0,\ 0)$ | $0.022$ | $(-0.469,\ 0.689)$ | $0.694$ |
| 兩段圓弧之外的點（$315^\circ$，圖例 gap point） | $-0.045$ | $(-0.920,\ 0.106)$ | $3.308$ |

| input | $z$ | $\hat{\mathbf{x}}$ | squared error |
|---|---|---|---|
| $300$ training points | | | median $0.0026$, max $0.027$ |
| a new point on arc 1 ($60^\circ$) | $0.834$ | $(0.563,\ 0.843)$ | $0.0044$ |
| the centre $(0,\ 0)$ | $0.022$ | $(-0.469,\ 0.689)$ | $0.694$ |
| a point off both arcs ($315^\circ$, "gap point" in the legend) | $-0.045$ | $(-0.920,\ 0.106)$ | $3.308$ |

新點雖然沒看過，但落在資料的圓弧上，誤差和訓練點同一個等級。另外兩個點不像訓練資料，誤差是訓練點最大誤差的約 $26$ 倍與 $120$ 倍。若把門檻設在訓練點的最大誤差 $0.027$，這兩點都會被標成異常。

The new point was never seen, but it lies on a data arc, and its error is on the same scale as the training points. The other two points do not look like the training data, and their errors are about $26$ and $120$ times the training points' largest error. With the threshold set at the training points' largest error, $0.027$, both would be flagged as anomalies.

![左右兩張圖。左圖是輸入平面：淺藍點是兩段圓弧上的訓練資料，三個空心圓是測試輸入，箭頭指向打叉的重建結果：綠色的新點在第一段圓弧上，重建幾乎在原處；紅色的圓心被畫到圓內、兩段圓弧之間的那條直線上；橘色的 315 度點被畫到左邊圓弧上端附近，箭頭很長。右圖是對數橫軸的直方圖：300 個訓練點的平方誤差集中在 0.001 到 0.01 附近，綠線 0.004 在其中，紅線 0.694 和橘線 3.308 在很右邊（Two panels. Left, the input plane: light-blue training data on two arcs, three hollow circles as test inputs, and arrows to crossed reconstructions: the green new point on arc 1 is reconstructed almost in place; the red centre is drawn inside the circle, onto the straight line between the two arcs; the orange 315-degree point is drawn near the upper end of the left arc, with a long arrow. Right, a histogram on a log horizontal axis: the squared errors of the 300 training points cluster around 0.001 to 0.01, the green line at 0.004 lies among them, and the red line at 0.694 and the orange line at 3.308 lie far to the right）](../assets/figures/lec12-kp07-test-phase.png)

**圖說 `lec12-kp07-test-phase`**〔功能：一目了然、比較〕：先看左圖：解碼器只會畫出圓弧上或兩段之間那條直線上的點，所以不在圓弧上的輸入都被「拉」到這些地方：$315^\circ$ 的點被拉到左邊圓弧上端附近，圓心被拉到兩段之間的直線上（$z\approx0$ 的縫）。箭頭越長，誤差越大（誤差是長度的平方）。再看右圖：訓練點的誤差擠在左邊，新點（綠線）也在裡面；圓心（紅線）和 $315^\circ$ 的點（橘線）遠遠落在右邊，這就是異常偵測的訊號。

**Caption `lec12-kp07-test-phase`** [Function: seeing it at a glance, comparison]: first look at the left panel: the decoder can only draw points on the arcs or on the straight line between them, so every input off the arcs is "pulled" there: the $315^\circ$ point to near the upper end of the left arc, the centre onto the straight line between the arcs (the seam at $z\approx0$). The longer the arrow, the larger the error (the error is the length squared). Then look at the right panel: the training errors crowd on the left, with the new point (green) among them; the centre (red) and the $315^\circ$ point (orange) land far to the right, which is the anomaly signal.

注意這兩個異常點的 $z$ 都接近 $0$ `[補充]`：KP05 的兩段編碼分別在 $0.15$ 到 $2.2$ 與 $-2.03$ 到 $-0.14$，$0.022$ 和 $-0.045$ 正好落在中間的縫。它們不像任何一段圓弧，編碼器就把它們放在那道縫附近；誤差大是因為縫裡解碼不出它們，不是因為 $z$ 偏離先驗（$z=0$ 正是先驗機率最高的地方）。

Note that both anomalous points get $z$ close to $0$ `[Added]`: KP05's two groups of codes span $0.15$ to $2.2$ and $-2.03$ to $-0.14$, so $0.022$ and $-0.045$ fall right in the seam. They resemble neither arc, so the encoder puts them near the seam; the error is large because the seam cannot decode them, not because $z$ strays from the prior ($z=0$ is where the prior is highest).

常見錯誤：以為測試時一定要先用編碼器。祕笈提醒，「測試 $=$ 重建」只對重建任務成立；生成時直接從先驗抽 $\mathbf{z}$。

Common mistake: thinking testing always starts with the encoder. The professor's notes point out that "testing $=$ reconstruction" holds only for reconstruction tasks; generation draws $\mathbf{z}$ straight from the prior.

### 2. 為什麼 VAE 的影像偏模糊（Why VAE images tend to be blurry）

來源說，和 GAN 比，VAE 生成的影像通常模糊，原因是三件事疊加：常態分布的解碼器、ELBO 只是下界、以及抽樣的隨機性；三者讓模型傾向輸出「平均」。祕笈特別指出第一點：常態解碼器等於平方誤差損失。

The source says that VAE images are usually blurry compared with a GAN's, because of three things stacked together: the normal-distribution decoder, the ELBO being only a lower bound, and the randomness of sampling; together they make the model lean towards outputting an "average". The professor's notes single out the first: a normal decoder is the same as a squared-error loss.

為什麼平方誤差會選平均 `[補充]`？假設同一個編碼 $\mathbf{z}$ 對應兩張一樣可能的影像，例如兩個像素的 $[1,\ 0]$ 和 $[0,\ 1]$。解碼器只能給一個答案 $\hat{\mathbf{x}}$。若猜其中一張，例如 $[1,\ 0]$，一半機會全對、一半機會差 $\lVert[1,0]-[0,1]\rVert^2=2$，平均誤差 $\mathbb{E}\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2=1$。若輸出平均 $[0.5,\ 0.5]$，兩種情況誤差都是 $0.25+0.25=0.5$。平方誤差比較喜歡灰色的平均，即使它根本不是任何一張真實的影像。

Why does squared error pick the average `[Added]`? Suppose one code $\mathbf{z}$ corresponds to two equally likely images, say the two-pixel $[1,\ 0]$ and $[0,\ 1]$. The decoder can give only one answer $\hat{\mathbf{x}}$. Guessing one of them, say $[1,\ 0]$, is exactly right half the time and off by $\lVert[1,0]-[0,1]\rVert^2=2$ the other half, an average error $\mathbb{E}\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2=1$. Outputting the average $[0.5,\ 0.5]$ gives error $0.25+0.25=0.5$ either way. Squared error prefers the grey average, even though it is not any real image.

一般地說，讓 $\mathbb{E}\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2$ 最小的 $\hat{\mathbf{x}}$，正好是這些可能影像的平均 $\mathbf{m}=\mathbb{E}[\mathbf{x}]$ `[補充：理由]`：$\mathbb{E}\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2=\mathbb{E}\lVert\mathbf{x}-\mathbf{m}\rVert^2+\lVert\mathbf{m}-\hat{\mathbf{x}}\rVert^2$，第一項和 $\hat{\mathbf{x}}$ 無關，第二項在 $\hat{\mathbf{x}}=\mathbf{m}$ 時是 $0$。下圖把它放到一張有邊緣的小影像上：

In general, the $\hat{\mathbf{x}}$ that minimises $\mathbb{E}\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2$ is exactly the average of the possible images, $\mathbf{m}=\mathbb{E}[\mathbf{x}]$ `[Added: reason]`: $\mathbb{E}\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2=\mathbb{E}\lVert\mathbf{x}-\mathbf{m}\rVert^2+\lVert\mathbf{m}-\hat{\mathbf{x}}\rVert^2$; the first term does not involve $\hat{\mathbf{x}}$, and the second is $0$ at $\hat{\mathbf{x}}=\mathbf{m}$. The figure below applies this to a small image with an edge:

![三條各有 20 個像素的橫條。第一條是清楚的影像 A，像素 0 到 7 是白色的 0，8 到 19 是黑色的 1；第二條是影像 B，邊緣移到像素 12；第三條是兩者的平均，像素 8 到 11 是灰色的 0.5，其他和 A、B 相同（Three strips of 20 pixels. The first is sharp image A, with white 0 for pixels 0 to 7 and black 1 for 8 to 19; the second is image B, with the edge moved to pixel 12; the third is their average, with grey 0.5 for pixels 8 to 11 and the rest the same as A and B）](../assets/figures/lec12-kp07-blur.png)

**圖說 `lec12-kp07-blur`**〔功能：一目了然、精準度〕：先看上兩條：同一個位置的邊緣，可能在像素 $8$，也可能在像素 $12$，兩張都很清楚。再看第三條：平方誤差偏好的答案，是在像素 $8$ 到 $11$ 塗上 $0.5$ 的灰色，邊緣變成一段模糊的過渡。猜上面第一張的平均誤差是 $\tfrac12\times0+\tfrac12\times4=2$，輸出平均的誤差是 $1$。

**Caption `lec12-kp07-blur`** [Function: seeing it at a glance, precision]: first look at the top two strips: the edge may be at pixel $8$ or at pixel $12$, and both images are sharp. Then look at the third strip: the answer squared error prefers paints pixels $8$ to $11$ a grey $0.5$, so the edge becomes a blurred transition. Guessing A has average error $\tfrac12\times0+\tfrac12\times4=2$; outputting the average has error $1$.

KP05 的小 VAE 也有同樣的現象 `[補充]`：在兩段編碼之間的縫 $z\approx0$，解碼器拿不準該畫哪一段圓弧，就畫在兩段之間的直線上，也就是「平均」的位置。抽樣也會造成同樣的效果 `[補充]`：訓練時解碼器收到的是從 $q$ 抽出、帶雜訊的 $\mathbf{z}$，不同資料的編碼會互相重疊，同一個 $\mathbf{z}$ 可能來自好幾筆資料，解碼器只好取它們的平均。真實影像裡，「拿不準」到處都是（頭髮的位置、紋理的細節），所以影像偏糊。

KP05's small VAE shows the same thing `[Added]`: at the seam $z\approx0$ between the two groups of codes, the decoder is unsure which arc to draw and draws on the straight line between them, at the "average" position. Sampling has the same effect `[Added]`: during training the decoder receives noisy $\mathbf{z}$ drawn from $q$, so the codes of different examples overlap and one $\mathbf{z}$ may come from several examples, which the decoder can only average. In real images, uncertainty is everywhere (where a strand of hair goes, fine texture), so the images come out soft.

常見錯誤：以為模糊是因為訓練資料太少或網路太深。來源的小測驗也考這一點：主因是解碼器的假設、ELBO 下界與抽樣三者疊加，而不是網路深度或資料量。

Common mistake: thinking the blur comes from too little training data or too deep a network. The source's quiz tests this too: the main cause is the decoder's assumption, the ELBO lower bound and sampling stacked together, not network depth or data size.

### 3. VAE 的變體（Variants of the VAE）

來源列了幾個常見的變體：

The source lists several common variants:

- **$\beta$-VAE**：把 KL 項乘上 $\beta>1$，壓縮得更緊，希望潛在空間的各個軸分別對應資料裡不同的因素（稱為**解纏**，disentangled）（$\beta$-VAE: multiply the KL term by $\beta>1$ for tighter compression, hoping each axis of the latent space captures a different factor of the data (called **disentangled**)）
- **階層式 VAE**（hierarchical VAE）：好幾層潛在變數，捕捉不同尺度的特徵（Hierarchical VAE: several layers of latent variables, capturing features at different scales）
- **VAE-GAN**：用 GAN 的判別器取代平方誤差，讓影像更清楚（VAE-GAN: replace the squared error with a GAN discriminator for sharper images）
- **VQ-VAE**：潛在變數是離散的，從一本「碼本」裡挑，用在音訊與影像（VQ-VAE: the latent variable is discrete, chosen from a "codebook", used for audio and images）
- **條件式 VAE**（CVAE）：潛在變數和標籤一起使用，可以指定要生成哪一類（Conditional VAE: the latent variable is used together with a label, so you can choose which class to generate）

調大 $\beta$ 的代價，可以用小 VAE 直接看 `[補充]`。同樣的資料與網路，只改 $\beta$ 各訓練一次。注意這裡的 $\beta$ 乘在 KP06 第4節的損失上（重建項前面有 $1/(2\sigma_x^2)=50$ 倍），和文獻裡「平方誤差 $+\ \beta\cdot$ KL」的 $\beta$ 刻度不同。表中的平方誤差與 KL 是訓練最後一步、$z$ 從 $q$ 抽樣時的平均（含抽樣雜訊），所以比第1節用 $z=\mu$ 的重建誤差大；生成落在資料圓弧上的判準同 KP05（離圓心距離與 $1$ 差不到 $0.1$，角度在兩段圓弧之內）：

The cost of raising $\beta$ can be seen directly with the small VAE `[Added]`. Same data and network, trained once for each $\beta$. Note that this $\beta$ multiplies the loss of KP06 Section 4 (whose reconstruction term carries a factor $1/(2\sigma_x^2)=50$), so its scale differs from the $\beta$ in the literature's "squared error $+\ \beta\cdot$ KL". The squared error and KL in the table are averages at the last training step with $z$ sampled from $q$ (including sampling noise), so they are larger than Section 1's reconstruction errors with $z=\mu$; "on the data arcs" uses KP05's criterion (distance from the centre within $0.1$ of $1$, angle inside an arc):

| $\beta$ | $0$ | $0.3$ | $1$ | $3$ | $10$ | $30$ | $100$ |
|---|---|---|---|---|---|---|---|
| 平方誤差 | $0.0006$ | $0.0039$ | $0.012$ | $0.031$ | $0.107$ | $0.346$ | $0.971$ |
| KL | $10.53$ | $2.81$ | $2.24$ | $1.74$ | $1.22$ | $0.63$ | $0.001$ |
| 生成落在資料圓弧上 | $64.0\%$ | $87.2\%$ | $92.6\%$ | $94.4\%$ | $88.5\%$ | $42.3\%$ | $0\%$ |

| $\beta$ | $0$ | $0.3$ | $1$ | $3$ | $10$ | $30$ | $100$ |
|---|---|---|---|---|---|---|---|
| squared error | $0.0006$ | $0.0039$ | $0.012$ | $0.031$ | $0.107$ | $0.346$ | $0.971$ |
| KL | $10.53$ | $2.81$ | $2.24$ | $1.74$ | $1.22$ | $0.63$ | $0.001$ |
| generated on the data arcs | $64.0\%$ | $87.2\%$ | $92.6\%$ | $94.4\%$ | $88.5\%$ | $42.3\%$ | $0\%$ |

$\beta$ 越大，KL 越小、重建越差。生成品質則是兩頭都不好：$\beta$ 太小，潛在空間沒整理好（KP06 第4節）；$\beta$ 太大，編碼器乾脆放棄資訊。$\beta=100$ 時 KL 幾乎是 $0$，每個 $q(z\mid\mathbf{x})$ 都變成先驗本身（標準差中位數 $0.996$），解碼器收到的 $z$ 不帶任何資訊，只好對所有輸入都畫出資料的平均附近。這時平方誤差 $0.971$，幾乎等於資料的總變異數 $0.974$（每個點到資料平均的平方距離的平均）：就是「每次都猜平均」的誤差。這叫**後驗崩塌**（posterior collapse），祕笈也提醒「$\beta$ 過大會 collapse」。

The larger $\beta$, the smaller the KL and the worse the reconstruction. Generation suffers at both ends: with $\beta$ too small the latent space is not organised (KP06 Section 4); with $\beta$ too large the encoder gives up on information altogether. At $\beta=100$ the KL is almost $0$, every $q(z\mid\mathbf{x})$ becomes the prior itself (median standard deviation $0.996$), and the decoder receives a $z$ carrying no information, so it draws near the data's average for every input. The squared error is then $0.971$, almost the data's total variance $0.974$ (the average squared distance from each point to the data mean): exactly the error of "always guessing the average". This is called **posterior collapse**, and the professor's notes also warn that "too large a $\beta$ collapses".

![三張圖。左圖：橫軸是 beta 從 0 到 100 的七個值，縱軸是對數刻度；藍線平方誤差從 0.0006 一路升到 0.97，紅線 KL 從 10.5 一路降到 0.001。中圖：beta 等於 10 時從先驗生成的點，大致沿兩段圓弧排列，另有一條直線從上方圓弧的左端連到左邊圓弧，標題寫 88% 落在資料圓弧上（四捨五入前是 88.5%）。右圖：beta 等於 100 時，所有生成的點擠成圓心附近的一小段，不在任何圓弧上，標題寫 0%（Three panels. Left: the horizontal axis has seven values of beta from 0 to 100, the vertical axis is logarithmic; the blue squared error rises steadily from 0.0006 to 0.97 and the red KL falls steadily from 10.5 to 0.001. Middle: points generated from the prior at beta 10, roughly along the two arcs, plus a straight segment from the left end of the upper arc to the left arc; the title says 88% on the data arcs (88.5% before rounding). Right: at beta 100 all generated points crowd into a short segment near the centre, on no arc; the title says 0%）](../assets/figures/lec12-kp07-beta-sweep.png)

**圖說 `lec12-kp07-beta-sweep`**〔功能：比較、精準度〕：先看左圖：兩條線往相反方向走，$\beta$ 是在「重建好」和「KL 小」之間調的旋鈕。再看中圖：$\beta=10$ 時生成大多還在圓弧上，但兩段之間那條直線（KP05 的縫）變長了。最後看右圖：$\beta=100$ 時所有生成的點擠在圓心附近，模型只會畫平均，這就是後驗崩塌。

**Caption `lec12-kp07-beta-sweep`** [Function: comparison, precision]: first look at the left panel: the two lines move in opposite directions, so $\beta$ is a knob between "good reconstruction" and "small KL". Then look at the middle panel: at $\beta=10$ generation mostly stays on the arcs, but the straight segment between them (KP05's seam) has grown longer. Finally look at the right panel: at $\beta=100$ all generated points crowd near the centre; the model can only draw the average, which is posterior collapse.

常見錯誤：以為 $\beta$-VAE 是用來讓影像更清楚。$\beta>1$ 讓重建變差，影像只會更糊；它要換的是更緊、更好解釋的潛在空間。真正針對清晰度的是 VAE-GAN 這類改掉平方誤差的作法。

Common mistake: thinking $\beta$-VAE is for sharper images. $\beta>1$ worsens reconstruction, so images only get blurrier; what it buys is a tighter, more interpretable latent space. What actually targets sharpness is VAE-GAN and other methods that replace the squared error.

### 4. 結論：VAE 的優點、限制與後續（Conclusion: strengths, limits and what came after）

來源把 VAE 稱為「機率深度學習」的代表作，把貝氏推論、神經網路與資訊壓縮放在一起。它的優點是：

The source calls the VAE the flagship of "probabilistic deep learning", combining Bayesian inference, neural networks and information compression. Its strengths are:

- 有原則的機率基礎：ELBO 是對數概似的下界，每一項都有清楚的意義（A principled probabilistic basis: the ELBO is a lower bound on the log-likelihood, and every term has a clear meaning）
- 有結構的潛在空間：KL 項把 $q$ 拉向先驗（A structured latent space: the KL term pulls $q$ towards the prior）
- 端到端可微：重參數化加反向傳播，整個模型一起訓練（End-to-end differentiable: reparameterization plus backpropagation trains the whole model together）
- 用途廣：異常偵測、半監督學習、壓縮、科學資料的低維表示（Broadly useful: anomaly detection, semi-supervised learning, compression, low-dimensional representations of scientific data）

限制則是：

Its limits are:

- 影像偏模糊（第2節）（Images tend to be blurry (Section 2)）
- ELBO 只是下界，把它推高不保證真的學好資料的分布（The ELBO is only a lower bound; pushing it up does not guarantee the data distribution is truly learned）
- 潛在維度要事先決定（The latent dimension must be fixed in advance）
- 在影像與文字的生成品質上，已被擴散模型與自迴歸 Transformer 超越（In the quality of image and text generation, it has been overtaken by diffusion models and autoregressive Transformers）

但 VAE 的想法沒有消失。來源舉的例子是**潛在擴散模型**（latent diffusion）：先用一個 VAE 把影像壓成較小的編碼，在編碼上做擴散，最後再用解碼器畫回影像。今天許多文字生成影像的系統就是這個架構 `[補充]`。那裡的 VAE 只負責壓縮與還原，而且通常加上感知損失與對抗損失一起訓練，所以模糊不是問題 `[補充]`。

But the VAE's ideas have not disappeared. The source's example is **latent diffusion**: a VAE first compresses images into smaller codes, diffusion runs on the codes, and the decoder finally draws the images back. Many of today's text-to-image systems use this design `[Added]`. The VAE there only compresses and restores, and is usually trained with perceptual and adversarial losses as well, so blur is not a problem `[Added]`.

下週的 GAN 走另一條路 `[補充：銜接]`：它不寫出概似、也不用平方誤差，而是讓一個判別器去分辨真假，只要求生成的影像「看起來真」。第2節的灰色平均騙不過判別器，所以 GAN 的影像通常比較清楚；代價是訓練比較不穩定，也沒有像 ELBO 這樣的機率詮釋。祕笈的比喻是：VAE 像一把瑞士刀，工具齊全、原理清楚，但需要最銳利的刀時，就該換別的工具。

Next week's GAN takes another road `[Added: bridge]`: it writes no likelihood and uses no squared error; instead a discriminator tells real from fake, and the generated images need only "look real". Section 2's grey average cannot fool a discriminator, so GAN images are usually sharper; the price is less stable training and no probabilistic reading like the ELBO. The professor's notes compare the VAE to a Swiss army knife: a full set of tools and clear principles, but when you need the sharpest blade, use another tool.

常見錯誤：以為 VAE 已經過時、沒有用了。祕笈提醒，VAE 在影像生成上被超越，但在異常偵測、語音、科學建模仍很活躍，而且是潛在擴散模型的一部分。

Common mistake: thinking the VAE is outdated and useless. The professor's notes point out that the VAE has been overtaken in image generation, but remains active in anomaly detection, speech and scientific modelling, and is a building block of latent diffusion.

### 本單元重點整理（KP07 summary）

**一句話重點**：訓練好的 VAE 可以重建（編碼器加解碼器，重建誤差可用來偵測異常）或生成（只用解碼器，從先驗抽 $\mathbf{z}$）；平方誤差讓解碼器在不確定時輸出平均，所以影像偏模糊；$\beta$ 在重建與潛在空間的整齊之間取捨，太大會後驗崩塌；VAE 在影像生成上已被擴散模型超越，但仍是潛在擴散模型的一部分，下週的 GAN 則改用判別器追求清晰。

**One-sentence summary**: a trained VAE can reconstruct (encoder plus decoder; the reconstruction error detects anomalies) or generate (decoder only, with $\mathbf{z}$ drawn from the prior); squared error makes the decoder output an average when uncertain, so images are blurry; $\beta$ trades reconstruction against an orderly latent space and collapses the posterior when too large; the VAE has been overtaken by diffusion in image generation but remains part of latent diffusion, while next week's GAN uses a discriminator to pursue sharpness.

**白話說明** `[補充]`：老師要學生畫「一隻貓」，又只用「和標準答案差多少」來打分。學生不確定貓頭要朝左還是朝右，最安全的作法是畫一隻兩邊都有點像的模糊貓，扣分最少。VAE 的解碼器就是這個學生。

**Plain explanation** `[Added]`: a teacher asks students to draw "a cat" and grades only by "how far from the model answer". A student unsure whether the cat faces left or right plays safe by drawing a blurry cat that looks a bit like both, which loses the fewest points. The VAE's decoder is that student.

**正式定義**：重建 $\mathbf{x}\to q(\mathbf{z}\mid\mathbf{x})\to\mathbf{z}\to\hat{\mathbf{x}}$；生成 $\mathbf{z}\sim\mathcal{N}(\mathbf{0},I)\to\hat{\mathbf{x}}$；$\arg\min_{\hat{\mathbf{x}}}\mathbb{E}[\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2\mid\mathbf{z}]=\mathbb{E}[\mathbf{x}\mid\mathbf{z}]$（同一個 $\mathbf{z}$ 可能對應的那些 $\mathbf{x}$ 的平均）；$\beta$-VAE 損失 $=$ 重建誤差 $+\ \beta\cdot D_{\text{KL}}(q(\mathbf{z}\mid\mathbf{x})\,\Vert\,p(\mathbf{z}))$。

**Formal definition**: reconstruction $\mathbf{x}\to q(\mathbf{z}\mid\mathbf{x})\to\mathbf{z}\to\hat{\mathbf{x}}$; generation $\mathbf{z}\sim\mathcal{N}(\mathbf{0},I)\to\hat{\mathbf{x}}$; $\arg\min_{\hat{\mathbf{x}}}\mathbb{E}[\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2\mid\mathbf{z}]=\mathbb{E}[\mathbf{x}\mid\mathbf{z}]$ (the average of the $\mathbf{x}$ that the same $\mathbf{z}$ may correspond to); $\beta$-VAE loss $=$ reconstruction error $+\ \beta\cdot D_{\text{KL}}(q(\mathbf{z}\mid\mathbf{x})\,\Vert\,p(\mathbf{z}))$.

**例子**：小 VAE 的訓練點重建誤差最大 $0.027$，圓心 $0.694$、$315^\circ$ 的點 $3.308$；$[1,0]$ 與 $[0,1]$ 猜一張的平均誤差 $1$、輸出平均 $0.5$；$\beta=100$ 時 KL $0.001$、平方誤差 $0.971\approx$ 資料總變異數 $0.974$，生成全部落在資料之外。

**Example**: the small VAE's largest training reconstruction error is $0.027$, versus $0.694$ for the centre and $3.308$ for the $315^\circ$ point; for $[1,0]$ and $[0,1]$, guessing one has average error $1$ and outputting the average $0.5$; at $\beta=100$ the KL is $0.001$ and the squared error $0.971\approx$ the data's total variance $0.974$, and every generated point lands off the data.

**常見迷思**：

- 「測試時一定先用編碼器。」錯，生成只用解碼器。
- 「模糊是因為網路太小。」錯，主因是平方誤差偏好平均等三個因素。
- 「$\beta$-VAE 讓影像更清楚。」錯，$\beta>1$ 讓重建變差；太大會後驗崩塌。
- 「VAE 已經沒用了。」錯，它在異常偵測、語音、科學建模仍活躍，也是潛在擴散的一部分。

**Common misconceptions**:

- "Testing always starts with the encoder." Wrong: generation uses only the decoder.
- "The blur comes from too small a network." Wrong: the main cause is the three factors, including squared error's preference for the average.
- "$\beta$-VAE makes images sharper." Wrong: $\beta>1$ worsens reconstruction, and too large a value collapses the posterior.
- "The VAE is useless now." Wrong: it remains active in anomaly detection, speech and scientific modelling, and is part of latent diffusion.

**圖**：本單元有三張圖。

- `lec12-kp07-test-phase`：重建與用重建誤差做異常偵測。
- `lec12-kp07-blur`：平方誤差偏好兩張清楚影像的平均。
- `lec12-kp07-beta-sweep`：$\beta$ 的取捨與後驗崩塌。

**Figures** — this unit has three figures.

- `lec12-kp07-test-phase`: reconstruction, and anomaly detection by reconstruction error.
- `lec12-kp07-blur`: squared error prefers the average of two sharp images.
- `lec12-kp07-beta-sweep`: the trade-off of $\beta$ and posterior collapse.

### 小練習（Practice）

1. 同一個編碼對應三張一樣可能的單像素影像，值是 $0$、$0$、$1$。讓平方誤差的平均最小的輸出是多少？這時的平均誤差是多少？
2. 一個異常偵測系統把門檻設在訓練點的最大重建誤差。門檻設得太低或太高，各會發生什麼事？
3. 一個 VAE 訓練完，KL 項幾乎是 $0$，生成的影像幾乎都一樣。最可能的原因是什麼？可以怎麼調？

1. One code corresponds to three equally likely one-pixel images with values $0$, $0$ and $1$. Which output minimises the average squared error, and what is that average error?
2. An anomaly detector sets its threshold at the largest training reconstruction error. What happens if the threshold is set too low, or too high?
3. After training a VAE, the KL term is almost $0$ and the generated images are almost all the same. What is the most likely cause, and how could you adjust it?

**第 1 題解答** `[補充]`：平均 $\tfrac13$。誤差 $\tfrac23\times(\tfrac13)^2+\tfrac13\times(\tfrac23)^2=\tfrac{2}{27}+\tfrac{4}{27}=\tfrac29\approx0.222$；若輸出 $0$，誤差是 $\tfrac13\approx0.333$。

**Solution to question 1** `[Added]`: the average, $\tfrac13$. The error is $\tfrac23\times(\tfrac13)^2+\tfrac13\times(\tfrac23)^2=\tfrac{2}{27}+\tfrac{4}{27}=\tfrac29\approx0.222$; outputting $0$ gives $\tfrac13\approx0.333$.

**第 2 題解答** `[補充]`：太低會把正常的新資料也標成異常（誤報多）；太高會讓真正的異常溜過（漏報多）。實務上用一批沒參與訓練的正常資料來選門檻。

**Solution to question 2** `[Added]`: too low flags normal new data as anomalies (many false alarms); too high lets real anomalies slip through (many misses). In practice the threshold is chosen on a batch of normal data not used in training.

**第 3 題解答** `[補充]`：後驗崩塌：KL 的權重相對太大（$\beta$ 太大，或重建項前的倍數太小，見 KP06 第4節），編碼器放棄資訊。可以調小 $\beta$，或讓重建項的權重變大。另外 `[補充]`，解碼器太強（例如自迴歸解碼器）也會讓它不需要編碼而崩塌；常見對策還有讓 $\beta$ 從 $0$ 慢慢升上來（KL 退火）。

**Solution to question 3** `[Added]`: posterior collapse: the KL weight is relatively too large ($\beta$ too large, or the multiplier on the reconstruction term too small; see KP06 Section 4), so the encoder gives up information. Lower $\beta$, or increase the weight of the reconstruction term. Also `[Added]`, an overly powerful decoder (an autoregressive one, say) can collapse because it does not need the code; another common remedy is to raise $\beta$ slowly from $0$ (KL annealing).

### 小測驗（Quiz，來源原題）

來源 §3.5、§3.6、§4 各有一題自我檢測，保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The source's §3.5, §3.6 and §4 each have one self-check question, kept here with all their options; the source page does not mark the answers, so the explanations below are judged from the source text.

1. VAE 的「生成」階段如何取 latent $z$？
   - (A) 用 encoder 從測試 $\boldsymbol{x}$ 取
   - (B) 直接從 prior $p(z)=\mathcal{N}(0,I)$ 抽樣
   - (C) 用 $K$-means 聚類 $z$
   - (D) 用 PCA 取 $z$
2. VAE 在圖像生成上為何模糊？
   - (A) 訓練資料太少
   - (B) Gaussian decoder $+$ ELBO 下界 $+$ 隨機採樣三因素疊加
   - (C) 網路太深
   - (D) KL 權重過低
3. VAE 在 2020 以後被什麼取代了影像生成的主流地位？
   - (A) 更深的網路
   - (B) Diffusion models 與 autoregressive transformers
   - (C) 強化學習
   - (D) 支持向量機

1. In the VAE's "generation" phase, how is the latent $z$ obtained?
   - (A) From a test $\boldsymbol{x}$ via the encoder
   - (B) Sampled directly from the prior $p(z)=\mathcal{N}(0,I)$
   - (C) By clustering $z$ with $K$-means
   - (D) By taking $z$ from PCA
2. Why is the VAE blurry in image generation?
   - (A) Too little training data
   - (B) Gaussian decoder $+$ ELBO lower bound $+$ random sampling stacked together
   - (C) The network is too deep
   - (D) The KL weight is too low
3. After 2020, what replaced the VAE as the mainstream of image generation?
   - (A) Deeper networks
   - (B) Diffusion models and autoregressive transformers
   - (C) Reinforcement learning
   - (D) Support vector machines

**解析** `[補充]`：第 1 題選 (B)：第1節；(A) 是重建。第 2 題選 (B)：第2節。第 3 題選 (B)：第4節。

**Explanations** `[Added]`: question 1 is B: see Section 1; A is reconstruction. Question 2 is B: see Section 2. Question 3 is B: see Section 4.

### 留給下一單元的問題（The question for the next unit）

本週從自編碼器出發，經過潛在變數、ELBO、EM，走到 VAE。VAE 的影像偏模糊，根源在於它用平方誤差這種「和答案差多少」的尺。如果不寫出概似，改請另一個網路來判斷「這張圖看起來真不真」，生成器又該怎麼訓練？這就是下一週（第13週）的生成對抗網路 GAN。

This week started from the autoencoder and went through latent variables, the ELBO and EM to the VAE. VAE images are blurry because it uses a ruler like squared error, "how far from the answer". If we write no likelihood and instead ask another network to judge "does this image look real", how should the generator be trained? That is next week's (Week 13) generative adversarial network, the GAN.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第12章「變分模型」§3.5「測試階段：採樣 vs 重建」、§3.6「其他 VAE 變體」、§4「結論：VAE 的優缺與後續」與對應的教授祕笈（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch12-variational-models/index.html ，讀取日期 2026-11-14）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §3.5 "Test phase: sampling vs reconstruction", §3.6 "Other VAE variants" and §4 "Conclusion: strengths, weaknesses and what came after" of Chapter 12, "Variational models", of CCC-LNS IDL_115, with the corresponding professor's notes (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch12-variational-models/index.html , accessed 2026-11-14). Content not in the source is marked `[Added]` where it appears.

- 來源把 $\beta$-VAE 列在「VAE 變體（解模糊問題）」之下，說各變體的「共同動機：解決 VAE 影像模糊」。$\beta>1$ 會讓重建變差（第3節的實驗），對影像而言就是更糊，$\beta$-VAE 的動機是解纏與更緊的潛在空間；本講義把它和針對清晰度的 VAE-GAN 分開說明。
- 來源說 VAE「在 2014-2018 是影像生成主流；2018 以後被 GAN 追平」。GAN 和 VAE 都在 2013–2014 年間提出，之後幾年清晰影像的生成多半由 GAN 領先；這段時間線只能當粗略的印象，本講義不採用。
- 來源說「VQ-VAE 用於 SoundStream、MusicGen」。精確地說，這些系統用的是向量量化的神經音訊編解碼器（MusicGen 用的是 EnCodec 的離散碼），屬於 VQ-VAE 的思路，不是 VQ-VAE 本身；本講義只寫 VQ-VAE「用在音訊與影像」。
- 祕笈說異常偵測有效，是因為「異常資料的 $z$ 偏離 prior」。第1節的例子正好相反：兩個異常點的 $z$ 都接近 $0$，就在先驗最高處；誤差大來自編碼之間的縫解碼不出它們。本講義照實驗結果寫。
- 來源本章其他節用 $\mathbb{P}(\cdot)$ 與粗斜體 `\boldsymbol` 表示機率與向量；本講義改用 $p(\cdot)$ 與 `\mathbf`。小測驗保留來源原本的記號，只把加號與 $K$ 改成數學排版。
- 本講義補上：異常偵測的數字例子與圖、平方誤差偏好平均的推導與兩張圖、小 VAE 的縫與「平均」的連結、$\beta$ 掃描實驗與後驗崩塌、和下週 GAN 的銜接、畫貓的比喻、三題小練習與解答。
- 取捨：來源的相機比喻、各變體的進一步細節（如階層式 VAE 的結構）與應用清單只簡述；來源 §4 的結論圖不重畫。

- The source lists $\beta$-VAE under "VAE variants (fixing blur)", saying the variants' "common motivation is to fix VAE blur". $\beta>1$ worsens reconstruction (Section 3's experiment), which for images means more blur; $\beta$-VAE is motivated by disentanglement and a tighter latent space, so this note separates it from VAE-GAN, which targets sharpness.
- The source says the VAE "was the mainstream of image generation in 2014–2018 and was matched by GANs after 2018". GANs and VAEs were both proposed in 2013–2014, and for several years afterwards GANs mostly led in sharp image generation; this timeline is at best a rough impression, and this note does not use it.
- The source says "VQ-VAE is used in SoundStream and MusicGen". Precisely, these systems use vector-quantized neural audio codecs (MusicGen uses EnCodec's discrete codes), which follow the VQ-VAE idea but are not VQ-VAE itself; this note only says VQ-VAE is "used for audio and images".
- The professor's notes say anomaly detection works because "an anomaly's $z$ strays from the prior". Section 1's example shows the opposite: both anomalies get $z$ near $0$, where the prior is highest; the large error comes from the seam between the codes failing to decode them. This note follows the experiment.
- Elsewhere in this chapter the source writes probabilities and vectors with $\mathbb{P}(\cdot)$ and bold italic `\boldsymbol`; this note uses $p(\cdot)$ and `\mathbf`. The quiz keeps the source's original notation, with only the plus signs and $K$ typeset as math.
- This note adds: the numerical anomaly-detection example and figure, the derivation that squared error prefers the average with two figures, the link between the small VAE's seam and the "average", the $\beta$ sweep and posterior collapse, the bridge to next week's GAN, the cat-drawing analogy, and three practice questions with solutions.
- Choices: the source's camera analogy, further details of each variant (such as the structure of a hierarchical VAE) and its list of applications are only summarised; the source's §4 conclusion diagram is not redrawn.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 重建 | reconstruction |
| 生成 | generation |
| 異常偵測 | anomaly detection |
| 門檻 | threshold |
| 模糊 | blur |
| 解纏 | disentanglement |
| 階層式 VAE | hierarchical VAE |
| 判別器 | discriminator |
| 碼本 | codebook |
| VAE 加判別器 | VAE-GAN |
| 向量量化 VAE | VQ-VAE |
| 向量量化 | vector quantization |
| 條件式 VAE | conditional VAE (CVAE) |
| 後驗崩塌 | posterior collapse |
| 擴散模型 | diffusion model |
| 潛在擴散模型 | latent diffusion |
| 自迴歸 | autoregressive |
| 生成對抗網路 | generative adversarial network (GAN) |
