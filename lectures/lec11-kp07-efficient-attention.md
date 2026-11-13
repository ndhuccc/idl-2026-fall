# 第11週 KP07：長序列的替代方案：稀疏注意力、線性注意力與 Performer（Alternatives for long sequences: sparse attention, linear attention and the Performer）

> 程度：大學生（undergrad），選讀／Level: Undergraduate (undergrad), optional reading

本講義是第11週的第七個知識點（KP07），也是本週的最後一個，接在 KP06 之後。KP06 算出自注意力的弱點：每一對詞都要算，計算與記憶體跟句長的平方成正比。這一節是選讀程度，看三條降低 $n^2$ 的路：只算一部分詞對（稀疏注意力）、換個順序做矩陣乘法（線性注意力）、用隨機特徵近似 softmax（Performer），最後簡短帶過其他方向。

This is the seventh knowledge point (KP07) of Week 11, the last of the week, following KP06. KP06 found self-attention's weakness: every pair of words is computed, so computation and memory grow with the square of the sentence length. This optional-reading note looks at three routes to lowering $n^2$: compute only some pairs (sparse attention), reorder the matrix products (linear attention), and approximate softmax with random features (the Performer), then briefly mentions other directions.

先備知識是 KP06 的 $4n^2d$ 與 $n\times n$ 權重表、第10週的 softmax 注意力，以及高中的指數與矩陣乘法。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第9章改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec11-kp07-figures.py` 算出，可以自己重算。

The prerequisites are KP06's $4n^2d$ and $n\times n$ weight table, the softmax attention of Week 10, and high-school exponentials and matrix products. Every term and symbol is defined when it first appears. This note is rewritten from Chapter 9 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec11-kp07-figures.py`, so you can recompute it yourself.

## KP07 長序列的替代方案：稀疏注意力、線性注意力與 Performer（Alternatives for long sequences: sparse attention, linear attention and the Performer）

### 學習目標（Learning objectives）

讀完之後，你能畫出滑動窗口與全域詞元的稀疏注意力圖樣，數出要算幾對，並說明全域詞元為什麼能讓遠處的詞兩層就連上。你能說明 softmax 為什麼擋住了矩陣乘法的結合律，以及拿掉它之後計算量怎麼從 $n^2$ 變成 $n$。你能用一句話說出 Performer 的想法：用正的隨機特徵近似 $e^{\mathbf{q}^\top\mathbf{k}}$，特徵越多越準。你也能分辨 FlashAttention 和前面三者的不同：它不近似，只是換個方式在硬體上算。

After reading, you can draw the sparse attention patterns of a sliding window and a global token, count the pairs to compute, and explain why a global token links distant words in two layers. You can explain why softmax blocks the associativity of matrix multiplication, and how the computation drops from $n^2$ to $n$ once it is removed. You can state the Performer's idea in one sentence: approximate $e^{\mathbf{q}^\top\mathbf{k}}$ with positive random features, more accurately with more features. You can also tell FlashAttention apart from the other three: it does not approximate, it only computes differently on the hardware.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $n,\ d$ | 句長與每個頭的向量長度 | sentence length and per-head vector length |
| $w$ | 滑動窗口的半徑：每個詞看左右各 $w$ 個詞 | radius of the sliding window: each word sees $w$ words on each side |
| $\mathbf{Q},\mathbf{K}\in\mathbb{R}^{n\times d},\ \mathbf{V}\in\mathbb{R}^{n\times d}$ | 查詢、鍵、值矩陣，每列一個詞 | query, key and value matrices, one word per row |
| $\phi(\cdot)$ | 特徵映射：把一個向量變成一串特徵 | feature map: turns a vector into a list of features |
| $m$ | 隨機特徵的個數 | number of random features |
| $\mathbf{w}_1,\dots,\mathbf{w}_m$ | 從標準常態分布抽出的隨機向量 | random vectors drawn from the standard normal distribution |

### 1. 稀疏注意力：只算一部分詞對（Sparse attention: compute only some pairs）

最直接的辦法：不要每一對都算。把 $n\times n$ 的權重表想成一張圖，只保留一部分格子。

The most direct idea: do not compute every pair. Think of the $n\times n$ weight table as a grid and keep only some of its cells.

- **滑動窗口**（sliding window）：每個詞只看左右各 $w$ 個詞。一句話裡大部分的關係都在附近，這樣就保留了大部分的資訊。每個詞約算 $2w+1$ 對，總共約 $(2w+1)n$ 對，是 $\mathcal{O}(nw)$。
- **全域詞元**（global token）：挑少數幾個位置（例如句首）讓它看所有詞，也被所有詞看。它像轉運站：任兩個詞先把資訊交給它，下一層再從它那裡拿，兩層就連上。
- Longformer 結合這兩種；BigBird 再加上少量隨機連線。

- **Sliding window**: each word looks only at $w$ words on each side. Most relations in a sentence are nearby, so most of the information is kept. Each word computes about $2w+1$ pairs, about $(2w+1)n$ in total, which is $\mathcal{O}(nw)$.
- **Global token**: pick a few positions (for example the start of the sentence) that look at every word and are looked at by every word. They act like a transfer station: any two words pass information to it in one layer and take it back in the next, so they connect in two layers.
- Longformer combines the two; BigBird adds a few random links as well.

用 $n=16$、$w=2$ 數一數 `[補充]`：全注意力要算 $256$ 對。只用滑動窗口是 $74$ 對（$28.9\%$），但第 $0$ 個詞到第 $15$ 個詞要經過 $8$ 層才連得上。加上一個全域詞元是 $100$ 對（$39.1\%$），任兩個詞最多 $2$ 層。

Count them with $n=16$ and $w=2$ `[Added]`: full attention computes $256$ pairs. The sliding window alone needs $74$ ($28.9\%$), but word $0$ and word $15$ connect only after $8$ layers. Adding one global token gives $100$ pairs ($39.1\%$), and any two words are at most $2$ layers apart.

![三個 16 乘 16 的格子，藍色表示要算的詞對。左邊全部是藍色，標題寫 256 對、1 層。中間只有對角線附近一條寬 5 格的藍帶，標題寫滑動窗口 w=2、74 對、8 層。右邊是同樣的藍帶，再加上第 0 列與第 0 欄全是藍色，用橘框框起來，註明 token 0 看所有詞、也被所有詞看，標題寫 100 對、2 層（Three 16-by-16 grids; blue marks the pairs computed. Left: all blue, titled 256 pairs, 1 layer. Middle: only a blue band five cells wide around the diagonal, titled sliding window w = 2, 74 pairs, 8 layers. Right: the same band plus row 0 and column 0 all blue, outlined in orange and noted as token 0 sees all and is seen by all, titled 100 pairs, 2 layers）](../assets/figures/lec11-kp07-sparse-patterns.png)

**圖說 `lec11-kp07-sparse-patterns`**〔功能：一目了然、精準度〕：由左往右看藍色越來越少。中間只剩對角線附近的藍帶，計算量降到不到三成，但遠處的詞要傳 $8$ 層。右邊多了橘框的一列一欄，只多算 $26$ 對，遠處的詞就變成 $2$ 層可達。

**Caption `lec11-kp07-sparse-patterns`** [Function: seeing it at a glance, precision]: from left to right the blue shrinks. The middle keeps only a band around the diagonal, cutting the computation below $30\%$, but distant words need $8$ layers. The right adds one orange-outlined row and column, only $26$ more pairs, and distant words become reachable in $2$ layers.

來源祕笈的大例子：$n=1000$，BigBird 每個詞約連 $2\times16+2\times4+4=44$ 個詞，共 $44000$ 對，只有全注意力 $1000000$ 對的 $4.4\%$。

The large example from the source's professor notes: with $n=1000$, BigBird links each word to about $2\times16+2\times4+4=44$ words, $44000$ pairs in all, only $4.4\%$ of full attention's $1000000$.

常見錯誤：以為先算出完整的 $n\times n$ 表、再把不要的格子乘上 $0$，就能省時間。那樣該算的都算了；真正省下來，要用只計算保留格子的程式。

Common mistake: thinking you save time by computing the full $n\times n$ table and then multiplying the unwanted cells by $0$. That still computes everything; real savings need code that computes only the kept cells.

### 2. 結合律：換個順序乘（Associativity: multiply in a different order）

第二條路不刪詞對，而是換計算順序。先假裝沒有 softmax，注意力就是 $(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$。矩陣乘法有結合律：

The second route keeps all pairs but changes the order of computation. Pretend for a moment there is no softmax; attention is then $(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$. Matrix multiplication is associative:

$$
(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}=\mathbf{Q}(\mathbf{K}^\top\mathbf{V}).
$$

左邊先算 $\mathbf{Q}\mathbf{K}^\top$，得到 $n\times n$ 的表，再乘 $\mathbf{V}$，共約 $4n^2d$ 次運算。右邊先算 $\mathbf{K}^\top\mathbf{V}$，只是 $d\times d$ 的小矩陣，和句長無關，再乘 $\mathbf{Q}$，共約 $4nd^2$ 次。兩邊結果相同 `[補充：程式驗證]`，成本比是 $n/d$。來源的例子：$n=4096$、$d=256$ 時快 $16$ 倍；$n=10000$、$d=64$ 時快 $156.25$ 倍。

The left side first computes $\mathbf{Q}\mathbf{K}^\top$, an $n\times n$ table, then multiplies by $\mathbf{V}$, about $4n^2d$ operations in all. The right side first computes $\mathbf{K}^\top\mathbf{V}$, only a $d\times d$ small matrix independent of the sentence length, then multiplies by $\mathbf{Q}$, about $4nd^2$ in all. The results are equal `[Added: checked in code]`, and the cost ratio is $n/d$. The source's examples: $16$ times faster at $n=4096$, $d=256$; $156.25$ times at $n=10000$, $d=64$.

問題是 softmax。真正的注意力是 $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$，softmax 要對 $\mathbf{Q}\mathbf{K}^\top$ 的每個數取指數、再逐列正規化，所以一定得先把 $n\times n$ 的表算出來，結合律用不上。

The obstacle is softmax. Real attention is $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$, and softmax exponentiates every entry of $\mathbf{Q}\mathbf{K}^\top$ and normalizes row by row, so the $n\times n$ table must be computed first, and associativity cannot be used.

**線性注意力**（linear attention）的想法：如果能找到一個特徵映射 $\phi$，讓 $e^{\mathbf{q}^\top\mathbf{k}}\approx\phi(\mathbf{q})^\top\phi(\mathbf{k})$，那麼第 $i$ 個詞的輸出

The idea of **linear attention**: if we can find a feature map $\phi$ with $e^{\mathbf{q}^\top\mathbf{k}}\approx\phi(\mathbf{q})^\top\phi(\mathbf{k})$, then the output of word $i$

$$
\mathbf{z}_i=\frac{\sum_j e^{\mathbf{q}_i^\top\mathbf{k}_j}\mathbf{v}_j}{\sum_j e^{\mathbf{q}_i^\top\mathbf{k}_j}}
\approx\frac{\phi(\mathbf{q}_i)^\top\big(\sum_j\phi(\mathbf{k}_j)\mathbf{v}_j^\top\big)}{\phi(\mathbf{q}_i)^\top\big(\sum_j\phi(\mathbf{k}_j)\big)}
$$

括號裡的兩個總和和 $i$ 無關，算一次就能給所有詞共用。整個計算就和 $n$ 成正比，不再出現 $n\times n$ 的表。

The two sums in parentheses do not depend on $i$, so they are computed once and shared by all words. The whole computation then grows in proportion to $n$, and no $n\times n$ table appears.

常見錯誤：以為線性注意力和原本的注意力完全一樣。只有在 $\phi(\mathbf{q})^\top\phi(\mathbf{k})$ 剛好等於 $e^{\mathbf{q}^\top\mathbf{k}}$ 時才一樣；實際上是近似，權重通常比較平緩，要精確找出某一個詞時會比較吃虧。

Common mistake: thinking linear attention is exactly the same as the original. It is only when $\phi(\mathbf{q})^\top\phi(\mathbf{k})$ equals $e^{\mathbf{q}^\top\mathbf{k}}$ exactly; in practice it is an approximation, its weights are usually flatter, and it does worse when it must pick out one specific word precisely.

### 3. Performer：用正的隨機特徵近似 softmax（The Performer: approximating softmax with positive random features）

Performer 給了一個具體的 $\phi$。從標準常態分布抽 $m$ 個隨機向量 $\mathbf{w}_1,\dots,\mathbf{w}_m$，定義

The Performer supplies a concrete $\phi$. Draw $m$ random vectors $\mathbf{w}_1,\dots,\mathbf{w}_m$ from the standard normal distribution and define

$$
\phi(\mathbf{x})=\frac{e^{-\lVert\mathbf{x}\rVert^2/2}}{\sqrt{2m}}\Big[e^{\mathbf{w}_1^\top\mathbf{x}},\dots,e^{\mathbf{w}_m^\top\mathbf{x}},\ e^{-\mathbf{w}_1^\top\mathbf{x}},\dots,e^{-\mathbf{w}_m^\top\mathbf{x}}\Big].
$$

它的性質是：平均起來（對隨機的 $\mathbf{w}$ 取期望值），$\phi(\mathbf{q})^\top\phi(\mathbf{k})$ 剛好等於 $e^{\mathbf{q}^\top\mathbf{k}}$；而且每個特徵都是指數，一定是正數。這套作法叫 FAVOR+（Fast Attention Via positive Orthogonal Random features），「+」指的就是正數；原論文還讓 $\mathbf{w}$ 互相正交，減少誤差。要注意：期望值相等的是**未正規化的分數**；注意力權重是兩個隨機量相除，$m$ 小時就算平均很多次也會偏一些。

Its property: on average (taking the expectation over the random $\mathbf{w}$), $\phi(\mathbf{q})^\top\phi(\mathbf{k})$ equals exactly $e^{\mathbf{q}^\top\mathbf{k}}$; and every feature is an exponential, hence positive. The method is called FAVOR+ (Fast Attention Via positive Orthogonal Random features), where the "+" refers to positivity; the original paper also makes the $\mathbf{w}$ mutually orthogonal to reduce the error. Note that what equals the target on average is the **unnormalized score**; an attention weight divides one random quantity by another, so for small $m$ it is somewhat biased even when averaged over many draws.

為什麼要正的？較早的隨機傅立葉特徵用 $\sin$、$\cos$，近似的分數可能是負的，注意力權重就會出現負數，甚至分母接近 $0$。用一個 $4$ 維的例子試 `[補充：示意]`：一個查詢對 $8$ 個鍵，各抽 $50$ 次隨機特徵（$\mathbf{w}$ 獨立抽樣，沒有做正交化）。用 $\sin$、$\cos$ 特徵時，$m=4$ 有 $98\%$ 的抽樣出現負分數，$m=64$ 仍有 $62\%$；用正的特徵時，每個特徵都是指數，分數必然是正的，所有抽樣當然都沒有負分數。

Why positive? Earlier random Fourier features use $\sin$ and $\cos$, so an approximated score may be negative, giving negative attention weights or a denominator near $0$. Try a $4$-dimensional example `[Added: illustrative]`: one query against $8$ keys, with $50$ draws of random features ($\mathbf{w}$ drawn independently, without orthogonalization). With $\sin$/$\cos$ features, $98\%$ of the draws at $m=4$ contain a negative score, and still $62\%$ at $m=64$; with positive features every feature is an exponential, so the score is necessarily positive, and no draw has a negative score.

![左圖是長條圖，橫軸是 8 個鍵 k1 到 k8，黑色是精確的 softmax 權重，三種紅色是正特徵在 m=16、256、4096 時 50 次抽樣的平均，上面的誤差棒表示散布；m 越大，紅色越接近黑色，誤差棒越短，例如 k6 精確是 0.327，m=16 平均 0.247，m=4096 平均 0.322。右圖是對數座標，橫軸隨機特徵數 m 從 4 到 2048，縱軸是 8 個權重的 L1 誤差總和（50 次平均）：紫線三角特徵從約 3 降到 0.07，紅線正特徵從約 0.51 降到 0.078，兩線在 m=512 附近交叉，灰色虛線是 1/√m 的參考斜率（Left: bar chart over 8 keys k1 to k8; black is the exact softmax weight, three reds are the mean of 50 draws of positive features at m = 16, 256 and 4096, with error bars for the spread; as m grows the red approaches the black and the bars shorten, for example k6 is 0.327 exactly, 0.247 on average at m = 16 and 0.322 at m = 4096. Right: log-log chart of the L1 error summed over the 8 weights (mean of 50 draws) against m from 4 to 2048; the purple trig-feature line falls from about 3 to 0.07, the red positive-feature line from about 0.51 to 0.078, the two cross near m = 512, and a grey dotted line shows the reference slope 1/√m）](../assets/figures/lec11-kp07-random-features.png)

**圖說 `lec11-kp07-random-features`**〔功能：一目了然、精準度〕：先看左圖：每組最左邊的黑色是精確答案；三根紅色由淺到深是 $m=16$、$256$、$4096$，越深越接近黑色，誤差棒也越短；$m=16$ 時的平均仍偏離黑色（例如 $k_6$ 是 $0.247$ 對 $0.327$），就是上面說的權重偏差。再看右圖，縱軸是 $8$ 個權重誤差的總和：兩條線都隨 $m$ 下降。特徵少的時候，正特徵（紅）的誤差小得多；三角特徵的誤差可以超過 $2$，是因為出現了負權重。兩線在 $m=512$ 附近交叉，之後三角特徵略好。紅線在 $m$ 小時下降得慢，$m$ 大時接近 $1/\sqrt{m}$ 的斜率，要很多特徵才夠準。

**Caption `lec11-kp07-random-features`** [Function: seeing it at a glance, precision]: first look at the left: the black bar at the left of each group is the exact answer; the three reds, light to dark, are $m=16$, $256$ and $4096$, getting closer to black and with shorter error bars as they darken; at $m=16$ the mean is still off the black (for example $k_6$: $0.247$ against $0.327$), which is the weight bias described above. Then look at the right, whose vertical axis sums the errors of the $8$ weights: both lines fall as $m$ grows. With few features, the positive features (red) have a much smaller error; the trig error can exceed $2$ because negative weights appear. The lines cross near $m=512$, after which the trig features are slightly better. The red line falls slowly for small $m$ and approaches the $1/\sqrt{m}$ slope for large $m$, so many features are needed for good accuracy.

成本方面，$\phi(\mathbf{Q})$ 和 $\phi(\mathbf{K})$ 都是 $n\times 2m$，照第2節的順序算，計算量約和 $nmd$ 成正比，對句長是線性的。代價是近似誤差：$m$ 要夠大才準，而且在需要很尖銳注意力的工作上，表現可能不如精確的 softmax。

As for cost, $\phi(\mathbf{Q})$ and $\phi(\mathbf{K})$ are both $n\times 2m$; computed in the order of Section 2, the work is roughly proportional to $nmd$, linear in the sentence length. The price is the approximation error: $m$ must be large enough for accuracy, and on tasks that need very sharp attention it may do worse than exact softmax.

常見錯誤：以為隨機特徵的 $\mathbf{w}$ 要跟著訓練。標準作法中 $\mathbf{w}$ 是隨機抽出後固定（或定期重抽）的，不用梯度更新。

Common mistake: thinking the random $\mathbf{w}$ of the features must be trained. In the standard method, $\mathbf{w}$ is drawn at random and kept fixed (or redrawn now and then), not updated by gradients.

### 4. 其他方向（Other directions）

來源還列了幾條路，這裡只用一兩句說明：

The source lists a few more routes, each described here in a sentence or two:

- **FlashAttention**：不改公式、也不近似。它把 $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$ 切成小塊，在 GPU 晶片上的高速記憶體裡算完 softmax 與加權，不把整張 $n\times n$ 表寫到較慢的顯示記憶體。乘法次數仍和 $n^2d$ 成正比（倒傳遞時還會重算一部分），但記憶體只要和 $n$ 成正比，因為少了大量讀寫，速度快了好幾倍。
- **狀態空間模型**（state space model，例如 Mamba）：回到「一步接一步更新一個狀態」的想法，但狀態不是一個向量，而是用高維矩陣封裝記憶，並設計成訓練時也能平行算；推論時每產生一個詞的成本固定，不隨前文變長。
- **模型壓縮**：量化（把權重從 $16$ 位元存成 $8$ 或 $4$ 位元）、剪掉不重要的頭與層、把大模型的能力蒸餾到小模型。

- **FlashAttention**: changes neither the formula nor the result. It cuts $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ into blocks and finishes softmax and weighting inside the GPU chip's fast memory, never writing the full $n\times n$ table to the slower graphics memory. The multiply count is still proportional to $n^2d$ (the backward pass even recomputes some of it), but memory grows only with $n$, and with far less reading and writing it runs several times faster.
- **State space models** (for example Mamba): return to the idea of "updating a state step by step", but with memory packed into a high-dimensional matrix rather than a single vector, and designed so training can still run in parallel; at inference, each new word costs a fixed amount that does not grow with the preceding text.
- **Model compression**: quantization (storing weights in $8$ or $4$ bits instead of $16$), pruning unimportant heads and layers, and distilling a large model's ability into a small one.

常見錯誤：把 FlashAttention 和線性注意力歸為同一類。線性注意力改了公式、結果是近似；FlashAttention 結果和標準注意力相同，只改了在硬體上的算法。

Common mistake: putting FlashAttention in the same class as linear attention. Linear attention changes the formula and gives an approximation; FlashAttention gives the same result as standard attention and changes only how it is computed on the hardware.

### 本單元重點整理（KP07 summary）

**一句話重點**：要降低 $n^2$，可以只算一部分詞對（滑動窗口加全域詞元，$\mathcal{O}(nw)$），或用特徵映射 $\phi$ 近似 $e^{\mathbf{q}^\top\mathbf{k}}$ 再用結合律先算 $\sum_j\phi(\mathbf{k}_j)\mathbf{v}_j^\top$（線性注意力；Performer 用正的隨機特徵），或不改公式只改硬體上的算法（FlashAttention）。

**One-sentence summary**: to lower $n^2$, compute only some pairs (sliding window plus global tokens, $\mathcal{O}(nw)$), or approximate $e^{\mathbf{q}^\top\mathbf{k}}$ with a feature map $\phi$ and use associativity to compute $\sum_j\phi(\mathbf{k}_j)\mathbf{v}_j^\top$ first (linear attention; the Performer uses positive random features), or keep the formula and change only how it runs on the hardware (FlashAttention).

**白話說明**：一場千人大會，不必人人互相握手：可以只和鄰座握手，再請幾位主持人負責轉達（稀疏）；也可以每個人把意見投進同一個意見箱，大家再從箱子裡各取所需（線性注意力）。

**Plain explanation**: at a meeting of a thousand people, not everyone needs to shake hands with everyone: people can greet only their neighbours and let a few hosts relay messages (sparse), or everyone can drop their views into one shared box and each take what they need from it (linear attention).

**正式定義**：線性注意力 $\mathbf{z}_i\approx\phi(\mathbf{q}_i)^\top\big(\sum_j\phi(\mathbf{k}_j)\mathbf{v}_j^\top\big)\big/\phi(\mathbf{q}_i)^\top\big(\sum_j\phi(\mathbf{k}_j)\big)$；Performer 的 $\phi(\mathbf{x})=\frac{e^{-\lVert\mathbf{x}\rVert^2/2}}{\sqrt{2m}}[e^{\mathbf{w}_i^\top\mathbf{x}},e^{-\mathbf{w}_i^\top\mathbf{x}}]_{i=1}^{m}$，期望值 $\mathbb{E}[\phi(\mathbf{q})^\top\phi(\mathbf{k})]=e^{\mathbf{q}^\top\mathbf{k}}$。

**Formal definition**: linear attention $\mathbf{z}_i\approx\phi(\mathbf{q}_i)^\top\big(\sum_j\phi(\mathbf{k}_j)\mathbf{v}_j^\top\big)\big/\phi(\mathbf{q}_i)^\top\big(\sum_j\phi(\mathbf{k}_j)\big)$; the Performer's $\phi(\mathbf{x})=\frac{e^{-\lVert\mathbf{x}\rVert^2/2}}{\sqrt{2m}}[e^{\mathbf{w}_i^\top\mathbf{x}},e^{-\mathbf{w}_i^\top\mathbf{x}}]_{i=1}^{m}$, with expectation $\mathbb{E}[\phi(\mathbf{q})^\top\phi(\mathbf{k})]=e^{\mathbf{q}^\top\mathbf{k}}$.

**例子**：$n=16$、$w=2$：全注意力 $256$ 對，窗口 $74$ 對（$8$ 層可達），加一個全域詞元 $100$ 對（$2$ 層可達）。結合律在 $n=4096$、$d=256$ 時快 $16$ 倍。隨機特徵：$\sin$、$\cos$ 在 $m=64$ 時 $62\%$ 的抽樣有負分數，正特徵從不出現。

**Example**: $n=16$, $w=2$: full attention $256$ pairs, window $74$ pairs (reachable in $8$ layers), one global token added $100$ pairs (reachable in $2$ layers). Associativity is $16$ times faster at $n=4096$, $d=256$. Random features: at $m=64$, $62\%$ of the $\sin$/$\cos$ draws have a negative score; positive features never do.

**常見迷思**：

- 「算完整張表再遮掉就能省時間。」錯，要只算保留的格子。
- 「線性注意力和原本完全一樣。」錯，是近似。
- 「FlashAttention 是一種近似。」錯，結果和標準注意力相同。

**Common misconceptions**:

- "Computing the whole table and then masking saves time." Wrong: only the kept cells must be computed.
- "Linear attention is exactly the original." Wrong: it is an approximation.
- "FlashAttention is an approximation." Wrong: its result equals standard attention.

**圖**：本單元有兩張圖。

- `lec11-kp07-sparse-patterns`：全注意力、滑動窗口、窗口加全域詞元的圖樣。
- `lec11-kp07-random-features`：隨機特徵近似的權重與誤差。

**Figures** — this unit has two figures.

- `lec11-kp07-sparse-patterns`: the patterns of full attention, a sliding window, and a window plus a global token.
- `lec11-kp07-random-features`: the weights and errors of the random-feature approximation.

### 小練習（Practice）

1. $n=1000$、$w=8$ 的滑動窗口，大約要算幾對？是全注意力的幾分之幾？
2. $n=8192$、$d=64$。$(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ 與 $\mathbf{Q}(\mathbf{K}^\top\mathbf{V})$ 的運算量比是多少？
3. 為什麼 $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ 不能直接改成 $\mathbf{Q}(\mathbf{K}^\top\mathbf{V})$？用一句話回答。

1. A sliding window with $n=1000$ and $w=8$: about how many pairs are computed, and what fraction of full attention is that?
2. With $n=8192$ and $d=64$, what is the ratio of the operation counts of $(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ and $\mathbf{Q}(\mathbf{K}^\top\mathbf{V})$?
3. Why can $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ not simply be rewritten as $\mathbf{Q}(\mathbf{K}^\top\mathbf{V})$? Answer in one sentence.

**第 1 題解答** `[補充]`：約 $(2\times8+1)\times1000=17000$ 對（句首句尾略少），是 $1000000$ 對的 $1.7\%$。

**Solution to question 1** `[Added]`: about $(2\times8+1)\times1000=17000$ pairs (slightly fewer at the two ends), $1.7\%$ of the $1000000$ pairs.

**第 2 題解答** `[補充]`：$n/d=8192/64=128$ 倍。

**Solution to question 2** `[Added]`: $n/d=8192/64=128$ times.

**第 3 題解答** `[補充]`：softmax 要對 $\mathbf{Q}\mathbf{K}^\top$ 的每個數取指數並逐列正規化，不是矩陣乘法，所以結合律不適用。

**Solution to question 3** `[Added]`: softmax exponentiates every entry of $\mathbf{Q}\mathbf{K}^\top$ and normalizes row by row, which is not a matrix product, so associativity does not apply.

### 小測驗（Quiz，來源原題）

來源這一段有十題自我檢測，保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。來源用「每一直行（column）一個詞」的記號，題中的 $\mathbf{V}\mathbf{Q}^\top$ 扮演本講義 $\mathbf{K}^\top\mathbf{V}$ 的角色，$r$、$p$ 是值與查詢的維度；在本講義的記號下 $r=p=d$。

The source has ten self-check questions for this part, kept with all their options; the source page does not mark the answers, so the explanations below are judged from the source text. The source uses "one word per column" notation: its $\mathbf{V}\mathbf{Q}^\top$ plays the role of this note's $\mathbf{K}^\top\mathbf{V}$, and $r$, $p$ are the value and query dimensions; in this note's notation $r=p=d$.

1. 為什麼學界需要積極研發 Transformer 的替代方案與效率變體？
   - (A) 因為 Transformer 的 $\mathcal{O}(n^2)$ 複雜度在處理超長序列時會面臨算力與記憶體爆炸
   - (B) 因為 Transformer 無法在 Python 中執行
   - (C) 因為 Transformer 只能處理 1 個單詞
   - (D) 因為 Transformer 完全沒有任何優點
2. 新興的狀態空間模型（如 Mamba）相較於傳統 RNN 的關鍵優勢是什麼？
   - (A) 完全不使用矩陣運算
   - (B) 採用線性狀態更新與矩陣封裝記憶，兼具線性推論複雜度與強大長程記憶容量
   - (C) 只能處理二進制輸入
   - (D) 必須使用比 Transformer 多 1000 倍的記憶體
3. Longformer 是如何將自注意力複雜度從 $\mathcal{O}(n^2)$ 降低至線性相關的？
   - (A) 將所有單詞直接刪除
   - (B) 結合局部滑動窗口注意力與少數全域標記注意力
   - (C) 將模型限制在只能輸出 0 與 1
   - (D) 使用 CPU 取代所有 GPU
4. FlashAttention 達成大幅加速與節省顯存的核心關鍵是什麼？
   - (A) 修改了注意力的數學定義使其產生近似誤差
   - (B) 利用 GPU 高速 SRAM 分塊計算與線上 Softmax，避免在慢速 HBM 顯存中物化 $n\times n$ 矩陣
   - (C) 完全捨棄反向傳播演算法
   - (D) 將所有浮點數強制轉換為整數
5. 若能將 Softmax 拆解為 $\mathbf{Q}'^\top\mathbf{K}'$，為什麼能將運算量從 $\mathcal{O}(n^2)$ 降為 $\mathcal{O}(n)$？
   - (A) 因為可以利用矩陣乘法結合律，先算 $(\mathbf{V}\mathbf{Q}'^\top)$ 消除 $n\times n$ 中間矩陣
   - (B) 因為所有矩陣元素都會自動變為 0
   - (C) 因為序列長度會被強制設為 1
   - (D) 因為不再需要進行任何矩陣乘法
6. 在無 Softmax 的結合律乘法 $(\mathbf{V}\mathbf{Q}^\top)\mathbf{K}$ 中，中間矩陣 $(\mathbf{V}\mathbf{Q}^\top)$ 的維度是多少？
   - (A) $n\times n$
   - (B) $r\times p$（與序列長度 $n$ 完全無關）
   - (C) $1\times 1$
   - (D) $d\times d$
7. 隨機傅立葉特徵（Random Fourier Features）的主要數學功能是什麼？
   - (A) 將非線性核函數近似為有限維隨機特徵向量的線性內積 $\phi(\mathbf{x})^\top\phi(\mathbf{y})$
   - (B) 將所有資料點投影到單一常數 1
   - (C) 消除所有機率分佈
   - (D) 將高斯分佈轉換為均勻分佈
8. 在高斯 RBF 核的隨機特徵近似中，隨著隨機特徵數 $s$ 增加，近似誤差會如何變化？
   - (A) 誤差會無限放大
   - (B) 蒙地卡羅近似誤差逐漸縮小，逼近真實核值
   - (C) 完全沒有任何影響
   - (D) 導致矩陣無法計算
9. Performer 的 FAVOR+ 演算法如何將注意力複雜度降至 $\mathcal{O}(n)$ 線性階？
   - (A) 將 Softmax 指數項展開為正隨機特徵內積 $\phi_{\text{SM}}(\mathbf{x})^\top\phi_{\text{SM}}(\mathbf{y})$，釋放矩陣結合律
   - (B) 直接丟棄所有 Key 向量
   - (C) 強制限制句子長度不能超過 10 個詞
   - (D) 將模型轉為卷積神經網路
10. 在 Performer 中，歸一化度矩陣 $\mathbf{D}$ 的計算複雜度是多少？
    - (A) $\mathcal{O}(n^3)$
    - (B) $\mathcal{O}(ns)$ 線性複雜度（透過預先對 $\mathbf{K}'$ 沿序列求和）
    - (C) $\mathcal{O}(\exp(n))$
    - (D) 無法在多項式時間內計算

1. Why do researchers actively develop alternatives and efficient variants of the Transformer?
   - (A) Because the Transformer's $\mathcal{O}(n^2)$ complexity makes computation and memory explode on very long sequences
   - (B) Because the Transformer cannot run in Python
   - (C) Because the Transformer can handle only one word
   - (D) Because the Transformer has no advantages at all
2. What is the key advantage of emerging state space models (such as Mamba) over traditional RNNs?
   - (A) They use no matrix operations at all
   - (B) Linear state updates and matrix-packed memory, combining linear inference cost with strong long-range memory
   - (C) They handle only binary input
   - (D) They need 1000 times more memory than the Transformer
3. How does Longformer reduce self-attention's complexity from $\mathcal{O}(n^2)$ to linear?
   - (A) By deleting all the words
   - (B) By combining local sliding-window attention with attention for a few global tokens
   - (C) By restricting the model to output only 0 and 1
   - (D) By replacing all GPUs with CPUs
4. What is the key to FlashAttention's large speed-up and memory saving?
   - (A) It changes the mathematical definition of attention, introducing approximation error
   - (B) It computes in blocks in the GPU's fast SRAM with an online softmax, avoiding materializing the $n\times n$ matrix in slow HBM memory
   - (C) It drops backpropagation entirely
   - (D) It forces all floating-point numbers into integers
5. If softmax can be split into $\mathbf{Q}'^\top\mathbf{K}'$, why does the cost drop from $\mathcal{O}(n^2)$ to $\mathcal{O}(n)$?
   - (A) Associativity of matrix multiplication lets us compute $(\mathbf{V}\mathbf{Q}'^\top)$ first, eliminating the $n\times n$ intermediate matrix
   - (B) All matrix entries automatically become 0
   - (C) The sequence length is forced to 1
   - (D) No matrix multiplication is needed at all
6. In the associative product $(\mathbf{V}\mathbf{Q}^\top)\mathbf{K}$ without softmax, what is the size of the intermediate matrix $(\mathbf{V}\mathbf{Q}^\top)$?
   - (A) $n\times n$
   - (B) $r\times p$ (completely independent of the sequence length $n$)
   - (C) $1\times 1$
   - (D) $d\times d$
7. What is the main mathematical role of random Fourier features?
   - (A) To approximate a nonlinear kernel function by the linear inner product $\phi(\mathbf{x})^\top\phi(\mathbf{y})$ of finite random feature vectors
   - (B) To project every data point to the constant 1
   - (C) To eliminate all probability distributions
   - (D) To turn a Gaussian distribution into a uniform one
8. In the random-feature approximation of the Gaussian RBF kernel, how does the error change as the number of random features $s$ grows?
   - (A) The error grows without bound
   - (B) The Monte Carlo error gradually shrinks, approaching the true kernel value
   - (C) There is no effect at all
   - (D) The matrix becomes impossible to compute
9. How does the Performer's FAVOR+ bring attention down to linear $\mathcal{O}(n)$?
   - (A) By expanding the softmax exponential into an inner product of positive random features $\phi_{\text{SM}}(\mathbf{x})^\top\phi_{\text{SM}}(\mathbf{y})$, unlocking associativity
   - (B) By discarding all key vectors
   - (C) By forcing sentences to be at most 10 words long
   - (D) By turning the model into a convolutional network
10. In the Performer, what is the cost of computing the normalizing degree matrix $\mathbf{D}$?
    - (A) $\mathcal{O}(n^3)$
    - (B) Linear, $\mathcal{O}(ns)$ (by summing $\mathbf{K}'$ along the sequence first)
    - (C) $\mathcal{O}(\exp(n))$
    - (D) Not computable in polynomial time

**解析** `[補充]`：1 (A)、2 (B)、3 (B)、4 (B)、5 (A)、6 (B)、7 (A)、8 (B)、9 (A)、10 (B)。第 1、2、4 題見第4節與 KP06；第 3 題見第1節；第 5、6 題見第2節，第 6 題在本講義記號下 (B) 和 (D) 指同一件事，照來源記號選 (B)，重點是與 $n$ 無關；第 7、8、9 題見第3節（第 8 題問的是高斯核，誤差縮小的道理可類比右圖）；第 10 題就是第2節分母裡的 $\sum_j\phi(\mathbf{k}_j)$，算一次即可。

**Explanations** `[Added]`: 1 A, 2 B, 3 B, 4 B, 5 A, 6 B, 7 A, 8 B, 9 A, 10 B. Questions 1, 2 and 4, see Section 4 and KP06; question 3, see Section 1; questions 5 and 6, see Section 2, and for question 6, options B and D describe the same thing in this note's notation, but in the source's notation the answer is B, the point being independence from $n$; questions 7, 8 and 9, see Section 3 (question 8 is about the Gaussian kernel; the shrinking error is analogous to the right chart); question 10 is the $\sum_j\phi(\mathbf{k}_j)$ in the denominator of Section 2, computed only once.

### 留給下一單元的問題（The question for the next unit）

這週從自注意力組出了完整的 Transformer，也看到它的代價與改良。到目前為止，模型都在「給輸入、算輸出」：分類、翻譯、預測下一個詞。如果想讓模型自己「創造」新的資料，例如一張從沒見過的臉、一段新的聲音，要怎麼做？下一週（第12週）進入生成模型。

This week built the complete Transformer from self-attention and looked at its costs and improvements. So far, models have been "given an input, compute an output": classify, translate, predict the next word. What if we want a model to "create" new data on its own, such as a face never seen before or a new piece of sound? Next week (Week 12) turns to generative models.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第9章「注意力機制與 Transformer 核心原理」§4–§4.2.3 與教授祕笈「稀疏注意力與局部感受野」「矩陣乘法結合律與線性注意力」「隨機傅立葉特徵與 Bochner 定理」「FAVOR+ 正定特徵與 Performer」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。本單元是選讀程度，省略了較深的數學。

This note is rewritten from §4–§4.2.3 of Chapter 9, "Core principles of attention mechanisms and the Transformer", of CCC-LNS IDL_115, and from the professor's notes "Sparse attention patterns", "Matrix associativity and linear attention", "Random Fourier features and Bochner's theorem" and "FAVOR+ and the Performer" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears. This unit is optional reading and leaves out the deeper mathematics.

- 來源正文把 FlashAttention 列在「把複雜度壓低至 $\mathcal{O}(n)$ 或 $\mathcal{O}(n\log n)$」的方法裡。FlashAttention 的乘法次數仍和 $n^2d$ 成正比，省的是記憶體與讀寫；本講義在第4節分開說明。
- 來源正文的 FAVOR+ 小節，把用 $\sin$、$\cos$ 的特徵映射稱為「正隨機特徵映射」。$\sin$、$\cos$ 會有負值；真正的正特徵是來源互動示範與祕笈用的指數形式，本講義採用後者。
- 祕笈的 FAVOR+ 公式有 $2m$ 個特徵，正規化卻寫成 $1/\sqrt{m}$（應為 $1/\sqrt{2m}$，祕笈自己的程式碼也用 $\sqrt{2m}$）。$m=1$ 時等於沒有除以 $2$，所以數值例子算出 $1.8963$；照那個公式，$m$ 增大時會收斂到 $2e\approx5.44$ 而不是 $e$。改正後，這一次抽樣的估計是 $0.9482$，目標值是 $e\approx2.7183$；單次抽樣本來就可能差很遠，期望值才等於目標。
- 來源正文寫高斯核 $e^{-\lVert\mathbf{x}-\mathbf{y}\rVert^2/\gamma}$ 可用 $\mathbf{w}\sim\mathcal{N}(0,\mathbf{I})$ 近似；這只在 $\gamma=2$ 時成立。本講義不展開隨機傅立葉特徵的推導。
- 來源第4節開頭的圖把 Kaplan 與 Chinchilla 兩種規模定律的式子與指數混在一起；本講義不採用。
- 本講義補上：$n=16$ 的三種稀疏圖樣、對數與層數、結合律的程式驗證、線性注意力的逐詞公式、隨機特徵的 $50$ 次抽樣實驗與兩張圖、三題小練習與解答。
- 取捨：Reformer、Linformer、BigBird 的隨機連線細節、Bochner 定理、度矩陣的推導、規模定律、量化的具體方法，本講義不寫或只帶過。

- The source's main text lists FlashAttention among methods that "bring the complexity down to $\mathcal{O}(n)$ or $\mathcal{O}(n\log n)$". FlashAttention still performs a number of multiplications proportional to $n^2d$; what it saves is memory and reads/writes. This note explains it separately in Section 4.
- The source's FAVOR+ subsection calls the feature map built from $\sin$ and $\cos$ a "positive random feature map". $\sin$ and $\cos$ take negative values; the truly positive features are the exponential form used in the source's interactive demo and the professor's notes, which this note adopts.
- The FAVOR+ formula in the professor's notes has $2m$ features but normalizes by $1/\sqrt{m}$ (it should be $1/\sqrt{2m}$, which the notes' own code uses). At $m=1$ this omits the division by $2$, so the numerical example gets $1.8963$; with that formula, the estimate would converge to $2e\approx5.44$ rather than $e$ as $m$ grows. Corrected, this single draw estimates $0.9482$ against the target $e\approx2.7183$; a single draw can easily be far off, and only the expectation equals the target.
- The source's main text says the Gaussian kernel $e^{-\lVert\mathbf{x}-\mathbf{y}\rVert^2/\gamma}$ can be approximated with $\mathbf{w}\sim\mathcal{N}(0,\mathbf{I})$; this holds only for $\gamma=2$. This note does not develop the derivation of random Fourier features.
- The figure at the start of the source's Section 4 mixes the formulas and exponents of the Kaplan and Chinchilla scaling laws; this note does not use it.
- This note adds: the three sparse patterns for $n=16$ with their pair and layer counts, a code check of associativity, the per-word formula of linear attention, the $50$-draw random-feature experiment with two figures, and three practice questions with solutions.
- Choices: Reformer, Linformer, the details of BigBird's random links, Bochner's theorem, the derivation of the degree matrix, scaling laws and specific quantization methods are left out or only mentioned.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 稀疏注意力 | sparse attention |
| 滑動窗口 | sliding window |
| 全域詞元 | global token |
| 詞元 | token |
| 結合律 | associativity |
| 線性注意力 | linear attention |
| 特徵映射 | feature map |
| 隨機特徵 | random features |
| 隨機傅立葉特徵 | random Fourier features (RFF) |
| 正的正交隨機特徵 | `FAVOR+`, Fast Attention Via positive Orthogonal Random features |
| 模型壓縮 | model compression |
| 長文件 Transformer 模型 | Longformer, BigBird |
| 循環神經網路（第8週） | recurrent neural network (RNN) |
| 期望值 | expectation |
| 蒙地卡羅近似 | Monte Carlo approximation |
| 高斯核（徑向基函數核） | Gaussian kernel (radial basis function, RBF) |
| 分塊計算的精確注意力 | FlashAttention |
| 狀態空間模型 | state space model (SSM) |
| 量化 | quantization |
| 剪枝 | pruning |
| 蒸餾 | distillation |
| 圖形處理器 | graphics processing unit (GPU) |
| 晶片內的高速記憶體 | static random-access memory (SRAM) |
| 高頻寬顯示記憶體 | high-bandwidth memory (HBM) |
| 中央處理器 | central processing unit (CPU) |
