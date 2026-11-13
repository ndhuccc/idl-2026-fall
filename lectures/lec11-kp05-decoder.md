# 第11週 KP05：解碼器，一邊看原文、一邊不偷看答案地寫（The decoder: writing while reading the source, without peeking at the answer）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第11週的第五個知識點（KP05），接在 KP04 之後。到 KP04 為止，編碼器已經完整，能把原文讀成一疊 $n\times512$ 的向量。翻譯還需要一個會「寫」的解碼器：它要一個詞一個詞寫出譯文，寫每個詞時都能查原文，而且不能偷看還沒寫到的詞。這一節看解碼器的三個子層，以及最後怎麼把向量變成「下一個詞」。

This is the fifth knowledge point (KP05) of Week 11 and follows KP04. By KP04 the encoder is complete and reads the source sentence into a stack of $n\times512$ vectors. Translation also needs a decoder that "writes": it writes the translation word by word, can consult the source for every word, and must not peek at words it has not written yet. This note covers the decoder's three sublayers and how a vector finally becomes "the next word".

先備知識是本週 KP01 的架構圖、KP03 的多頭注意力、KP04 的 Add & Norm，以及第10週 KP03 的帶注意力 Seq2Seq。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第9章改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec11-kp05-figures.py` 算出，可以自己重算。

The prerequisites are this week's KP01 architecture figure, KP03's multi-head attention, KP04's Add & Norm, and the Seq2Seq with attention of Week 10 KP03. Every term and symbol is defined when it first appears. This note is rewritten from Chapter 9 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec11-kp05-figures.py`, so you can recompute it yourself.

## KP05 解碼器，一邊看原文、一邊不偷看答案地寫（The decoder: writing while reading the source, without peeking at the answer）

### 學習目標（Learning objectives）

讀完之後，你能說出解碼器一層的三個子層與各自的 $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$ 從哪裡來。你能寫出因果遮罩矩陣，並用 $e^{-\infty}=0$ 算出遮罩後的注意力權重。你能說明為什麼訓練時解碼器可以平行計算，推論時卻要一個詞一個詞來。你能算出交叉注意力的權重與形狀。最後，你能用溫度、貪婪解碼、Top-$k$ 與 Top-$p$ 從輸出機率選出下一個詞。

After reading, you can name the three sublayers of a decoder layer and where each one's $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ come from. You can write the causal mask matrix and use $e^{-\infty}=0$ to compute masked attention weights. You can explain why the decoder computes in parallel during training but goes word by word during inference. You can compute the weights and shapes of cross-attention. Finally, you can pick the next word from the output probabilities with temperature, greedy decoding, Top-$k$ and Top-$p$.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $T_x,\ T_y$ | 原文與譯文的詞數 | number of words in the source and in the translation |
| $\mathbf{H}\in\mathbb{R}^{T_x\times d_{\text{model}}}$ | 編碼器最上層的輸出 | output of the top encoder layer |
| $\mathbf{S}\in\mathbb{R}^{T_y\times d_{\text{model}}}$ | 同一層遮罩自注意力子層（含 Add & Norm）的輸出，每列是一個譯文位置 | output of the same layer's masked self-attention sublayer (with Add & Norm), one target position per row |
| $d_k$ | 每個頭的查詢與鍵的長度，Base 是 $d_{\text{model}}/h=64$（KP03） | length of each head's queries and keys, $d_{\text{model}}/h=64$ in Base (KP03) |
| $\mathbf{M}\in\mathbb{R}^{T_y\times T_y}$ | 因果遮罩 | causal mask |
| $\lvert\mathcal{V}\rvert$ | 詞表的大小 | size of the vocabulary |
| $\mathbf{z}\in\mathbb{R}^{\lvert\mathcal{V}\rvert}$ | 每個詞的未正規化分數（logits） | unnormalized score of each word (logits) |
| $T$ | 取樣溫度 | sampling temperature |

### 1. 解碼器一層：三個子層（A decoder layer: three sublayers）

解碼器也疊 $N=6$ 層，每層比編碼器多一個子層，共三個，每個後面都接 KP04 的 Add & Norm：

The decoder also stacks $N=6$ layers, each with one more sublayer than the encoder, three in all, each followed by KP04's Add & Norm:

1. **遮罩多頭自注意力**：譯文已寫出的詞彼此做自注意力，但每個位置只能看自己和前面。
2. **交叉注意力**：用解碼器的狀態當查詢，到編碼器的輸出裡找原文的資訊。
3. **逐位置前饋網路**：和 KP04 相同。

1. **Masked multi-head self-attention**: the words of the translation written so far attend to each other, but each position sees only itself and earlier positions.
2. **Cross-attention**: the decoder's state serves as the query and looks up the source information in the encoder's output.
3. **Position-wise feed-forward network**: the same as in KP04.

解碼器的輸入是「往右移一格」（shifted right）的譯文：最前面放一個開始記號 `<s>`，後面接正確譯文去掉最後一個詞。以法翻英 "Je suis étudiant" → "I am a student" 為例，解碼器的輸入是 `<s> I am a`，要預測的輸出是 `I am a student`。第 $i$ 個位置看到的是前 $i-1$ 個譯文詞加上 `<s>`，要預測第 $i$ 個詞。

The decoder's input is the translation "shifted right": a start token `<s>` goes first, followed by the correct translation without its last word. For the French-to-English "Je suis étudiant" → "I am a student", the decoder input is `<s> I am a` and the outputs to predict are `I am a student`. Position $i$ sees `<s>` plus the first $i-1$ translation words and must predict word $i$.

**訓練與推論不一樣。** 訓練時整句正確譯文都已知道，所以把整個右移的輸入一次送進去，所有位置同時算出預測與損失；這種「拿正確答案當輸入」的作法叫教師強制（teacher forcing）。推論時沒有正確答案，只能先寫第 $1$ 個詞，把它接到輸入後面，再寫第 $2$ 個詞，一個一個來。

**Training and inference differ.** During training the whole correct translation is known, so the entire shifted input is fed at once and all positions compute their predictions and losses together; feeding the correct answer as input this way is called teacher forcing. During inference there is no correct answer, so the model writes word $1$, appends it to the input, writes word $2$, and so on, one at a time.

常見錯誤：以為訓練時解碼器也要一個詞一個詞跑。訓練時一次平行算完；能這樣做而不作弊，靠的是第2節的遮罩。

Common mistake: thinking the decoder also runs word by word during training. Training computes everything in parallel at once; what makes this possible without cheating is the mask of Section 2.

### 2. 因果遮罩：把右上角蓋住（The causal mask: covering the upper right）

一次送進整句時，每個位置的自注意力本來看得到後面所有位置的輸入。右移之後，第 $2$ 個位置的輸入 `I` 正是第 $1$ 個位置要預測的答案，第 $3$ 個位置的輸入 `am` 是第 $2$ 個位置的答案，以此類推。為了不偷看，在分數上加一個遮罩 $\mathbf{M}$，再做 softmax：

When the whole sentence is fed at once, self-attention at every position could see the inputs of all later positions. After the shift, the input `I` at position $2$ is exactly the answer position $1$ must predict, the input `am` at position $3$ is position $2$'s answer, and so on. To prevent peeking, a mask $\mathbf{M}$ is added to the scores before softmax:

$$
\operatorname{MaskedAttn}(\mathbf{Q},\mathbf{K},\mathbf{V})=\operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}+\mathbf{M}\right)\mathbf{V},\qquad
M_{ij}=\begin{cases}0, & j\le i,\\ -\infty, & j>i.\end{cases}
$$

第 $i$ 列是第 $i$ 個位置的查詢，第 $j$ 欄是第 $j$ 個位置的鍵。$j>i$ 就是「未來」，分數變成 $-\infty$，而 $e^{-\infty}=0$，所以權重剛好是 $0$。對角線 $j=i$ 是自己，設成 $0$，看得到。

Row $i$ is the query at position $i$ and column $j$ the key at position $j$. $j>i$ means "the future": the score becomes $-\infty$, and since $e^{-\infty}=0$, the weight is exactly $0$. The diagonal $j=i$ is the position itself, set to $0$ and visible.

來源祕笈的例子：三個位置的分數（已縮放）是 $\begin{bmatrix}2&3&1\\1&4&5\\2&1&3\end{bmatrix}$。加上遮罩後逐列做 softmax：

The example from the source's professor notes: the (already scaled) scores of three positions are $\begin{bmatrix}2&3&1\\1&4&5\\2&1&3\end{bmatrix}$. After adding the mask, softmax row by row gives:

| 查詢 | 第1個鍵 | 第2個鍵 | 第3個鍵 |
|---|---|---|---|
| 第1列 $\operatorname{softmax}([2,-\infty,-\infty])$ | $1$ | $0$ | $0$ |
| 第2列 $\operatorname{softmax}([1,4,-\infty])$ | $0.0474$ | $0.9526$ | $0$ |
| 第3列 $\operatorname{softmax}([2,1,3])$ | $0.2447$ | $0.0900$ | $0.6652$ |

| Query | key 1 | key 2 | key 3 |
|---|---|---|---|
| Row 1 $\operatorname{softmax}([2,-\infty,-\infty])$ | $1$ | $0$ | $0$ |
| Row 2 $\operatorname{softmax}([1,4,-\infty])$ | $0.0474$ | $0.9526$ | $0$ |
| Row 3 $\operatorname{softmax}([2,1,3])$ | $0.2447$ | $0.0900$ | $0.6652$ |

![三個 3 乘 3 格子。左邊橘色是分數，第一列 2、3、1，第二列 1、4、5，第三列 2、1、3。中間把右上三格換成深灰色的負無限大，並用紅框圈起來。右邊藍色是 softmax 後的權重：第一列 1、0、0，第二列 0.0474、0.9526、0，第三列 0.2447、0.0900、0.6652，右上紅框內都是 0，旁邊寫 future: weight 0（Three 3-by-3 grids. Left, orange scores with rows 2, 3, 1; 1, 4, 5; 2, 1, 3. Middle, the three upper-right cells are replaced by dark grey minus infinity, outlined in red. Right, blue weights after softmax: 1, 0, 0; 0.0474, 0.9526, 0; 0.2447, 0.0900, 0.6652; the red-outlined upper right is all 0, labelled future: weight 0）](../assets/figures/lec11-kp05-causal-mask.png)

**圖說 `lec11-kp05-causal-mask`**〔功能：一目了然、精準度〕：圖中的 pos 1–3 是第 $1$ 到第 $3$ 個位置。先看中間：紅框裡的右上三格是「未來」，分數換成 $-\infty$。再看右邊：紅框裡的權重全是 $0$，其他每一列仍加總為 $1$。注意第2列：沒有遮罩時它最看重第 $3$ 個詞（分數 $5$），遮罩之後只能在前兩個詞裡分配，$0.9526$ 給了第 $2$ 個詞。

**Caption `lec11-kp05-causal-mask`** [Function: seeing it at a glance, precision]: pos 1–3 in the figure are positions $1$ to $3$. First look at the middle: the three upper-right cells inside the red outline are "the future", and their scores become $-\infty$. Then look at the right: the weights inside the red outline are all $0$, and every row still sums to $1$. Note row 2: without the mask it would favour word $3$ (score $5$); with the mask it can only share among the first two words, giving $0.9526$ to word $2$.

遮罩真的擋住了未來嗎？`[補充：程式驗證]` 把第 $3$ 個詞的向量整個換掉，再做一次遮罩自注意力：第 $1$、$2$ 個位置的輸出每一個位元都不變，只有第 $3$ 個位置改變。所以每個位置的預測只依賴它自己和前面的詞，訓練時一次送進整句也不會作弊。

Does the mask really block the future? `[Added: checked in code]` Replace the vector of word $3$ entirely and run masked self-attention again: the outputs at positions $1$ and $2$ are unchanged down to the last bit, and only position $3$ changes. So each position's prediction depends only on itself and earlier words, and feeding the whole sentence at once in training is not cheating.

常見錯誤：把對角線也設成 $-\infty$。對角線是自己，要看得到；只有 $j>i$ 的未來才遮。

Common mistake: setting the diagonal to $-\infty$ as well. The diagonal is the position itself and must stay visible; only the future, $j>i$, is masked.

### 3. 交叉注意力：拿著問題去查原文（Cross-attention: taking a question to the source）

第二個子層是交叉注意力（cross-attention，也叫編碼器—解碼器注意力）。它和自注意力的算法完全相同，差在三樣東西的來源不同：

The second sublayer is cross-attention (also called encoder-decoder attention). It computes exactly like self-attention; the difference is where the three ingredients come from:

$$
\mathbf{Q}=\mathbf{S}\mathbf{W}_Q,\qquad \mathbf{K}=\mathbf{H}\mathbf{W}_K,\qquad \mathbf{V}=\mathbf{H}\mathbf{W}_V,\qquad
\operatorname{CrossAttn}=\operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}.
$$

查詢來自同一層遮罩自注意力子層（含 Add & Norm）的輸出 $\mathbf{S}$，代表「我現在要寫的這個詞，需要原文的什麼」；鍵和值來自編碼器最上層的輸出 $\mathbf{H}$，代表原文的資訊。這就是第10週帶注意力 Seq2Seq 的「對齊」，只是改成多頭、而且一次算所有位置。

The query comes from the output $\mathbf{S}$ of the same layer's masked self-attention sublayer (with Add & Norm), meaning "what do I need from the source for the word I am writing now"; the keys and values come from the top encoder output $\mathbf{H}$, the information of the source. This is the "alignment" of Week 10's Seq2Seq with attention, now multi-headed and computed for all positions at once.

形狀 `[補充]`：原文 $T_x=5$ 個詞、譯文 $T_y=4$ 個詞，一個頭 $d_k=64$。$\mathbf{Q}$ 是 $4\times64$，$\mathbf{K}$、$\mathbf{V}$ 是 $5\times64$，權重 $\mathbf{Q}\mathbf{K}^\top$ 是 $4\times5$，每一列是一個譯文位置對 $5$ 個原文詞的關注，輸出是 $4\times64$。權重不再是方陣，輸出列數跟著譯文走，和原文多長無關。

Shapes `[Added]`: with $T_x=5$ source words, $T_y=4$ target words and one head with $d_k=64$, $\mathbf{Q}$ is $4\times64$, $\mathbf{K}$ and $\mathbf{V}$ are $5\times64$, the weights $\mathbf{Q}\mathbf{K}^\top$ are $4\times5$, each row being one target position's attention over the $5$ source words, and the output is $4\times64$. The weights are no longer square; the number of output rows follows the translation, whatever the length of the source.

來源祕笈的例子：正在寫第 $1$ 個譯文詞，查詢 $\mathbf{q}=[1,1]$；原文三個詞的鍵是 $[2,0]$、$[0,2]$、$[2,2]$，$d_k=2$。點積是 $2,2,4$，除以 $\sqrt2$ 得 $1.4142,1.4142,2.8284$，softmax 後是 $[0.1636,\ 0.1636,\ 0.6728]$：這個譯文詞把約 $67\%$ 的注意力放在原文第 $3$ 個詞。

The example from the source's professor notes: writing the first target word, the query is $\mathbf{q}=[1,1]$; the keys of three source words are $[2,0]$, $[0,2]$, $[2,2]$, with $d_k=2$. The dot products are $2,2,4$; divided by $\sqrt2$ they are $1.4142,1.4142,2.8284$; after softmax, $[0.1636,\ 0.1636,\ 0.6728]$: this target word puts about $67\%$ of its attention on source word $3$.

交叉注意力**不加因果遮罩**：原文在寫譯文之前就整句讀完了，譯文的每個詞都可以看原文的每個詞，包含原文後面的詞。另外，$6$ 層解碼器各有自己的 $\mathbf{W}_K$、$\mathbf{W}_V$，每層對編碼器輸出各算一次 $\mathbf{K}$、$\mathbf{V}$，推論時寫每個詞都重複使用。

Cross-attention **has no causal mask**: the source is read in full before the translation starts, so every target word may look at every source word, including later ones. Also, each of the $6$ decoder layers has its own $\mathbf{W}_K$ and $\mathbf{W}_V$, so each layer computes $\mathbf{K}$ and $\mathbf{V}$ from the encoder output once and reuses them for every word written during inference.

常見錯誤：把來源搞反，以為查詢來自編碼器。記法：誰在「問」，查詢就來自誰；在寫譯文的是解碼器，所以 $\mathbf{Q}$ 來自解碼器，$\mathbf{K}$、$\mathbf{V}$ 來自編碼器。

Common mistake: swapping the sources and thinking the query comes from the encoder. A way to remember: whoever is "asking" supplies the query; the decoder is writing the translation, so $\mathbf{Q}$ comes from the decoder and $\mathbf{K}$, $\mathbf{V}$ from the encoder.

### 4. 輸出：從向量到下一個詞（Output: from a vector to the next word）

解碼器最上層每個位置輸出一個 $512$ 維向量。一個線性層把它投影成長度 $\lvert\mathcal{V}\rvert$ 的分數 $\mathbf{z}$（logits），每個分量對應詞表裡的一個詞；Transformer-Base 在英德翻譯用的共用詞表約 $37000$ 個（單位是子詞，不一定是整個詞；英法翻譯則是約 $32000$ 個），這個線性層有 $512\times37000=18944000$ 個權重。原論文讓兩個詞嵌入層和這個線性層共用同一個矩陣（嵌入時再乘上 $\sqrt{d_{\text{model}}}$）`[補充]`，KP01 的參數總數就是這樣算的。再用 softmax 把分數變成機率。

At every position the top decoder layer outputs a $512$-dimensional vector. A linear layer projects it to scores $\mathbf{z}$ (logits) of length $\lvert\mathcal{V}\rvert$, one per word in the vocabulary; Transformer-Base's shared vocabulary for English-German has about $37000$ entries (subword units, not necessarily whole words; about $32000$ for English-French), so this linear layer has $512\times37000=18944000$ weights. The original paper shares one matrix among the two embedding layers and this linear layer (the embeddings are multiplied by $\sqrt{d_{\text{model}}}$) `[Added]`, which is how KP01's total parameter count was computed. Softmax then turns the scores into probabilities.

有了機率，要選哪個詞？最簡單是**貪婪解碼**（greedy）：每次選機率最大的詞。也可以先調整**溫度** $T$，再抽樣：

With probabilities in hand, which word to choose? The simplest is **greedy decoding**: always pick the most probable word. Alternatively, adjust the **temperature** $T$ first and then sample:

$$
P(w_k)=\frac{e^{z_k/T}}{\sum_{j}e^{z_j/T}},\qquad T>0.
$$

$T=1$ 就是一般的 softmax；$T$ 越小，分數差距被放大，分布越尖，越接近貪婪；$T$ 越大，分布越平，越多樣但也越容易亂寫。來源的例子：兩個詞的分數 $4,2$，$T=1$ 時機率 $0.8808,0.1192$；$T=0.5$ 時 $0.9820,0.0180$；$T=2$ 時 $0.7311,0.2689$。

$T=1$ is ordinary softmax; a smaller $T$ magnifies score differences, so the distribution gets sharper and closer to greedy; a larger $T$ flattens it, giving more variety but also more nonsense. The source's example: two words with scores $4,2$ have probabilities $0.8808,0.1192$ at $T=1$; $0.9820,0.0180$ at $T=0.5$; $0.7311,0.2689$ at $T=2$.

抽樣時，機率很小的怪詞偶爾也會被抽中。兩種截斷法：**Top-$k$** 只留機率最大的 $k$ 個詞；**Top-$p$**（nucleus）從大到小累加機率，加到第一次 $\ge p$ 為止，留下這幾個詞。留下的詞重新正規化，再抽樣。

When sampling, odd words with tiny probability are occasionally drawn. Two ways to cut them off: **Top-$k$** keeps only the $k$ most probable words; **Top-$p$** (nucleus) adds probabilities from largest to smallest until the total first reaches $\ge p$, and keeps those words. The kept words are renormalized, then sampled.

用一個例子看 `[補充：示意]`：「the cat sat on the ...」之後，五個候選詞 mat、floor、sofa、roof、car 的分數是 $3,2,1.5,0.5,-1$。

See an example `[Added: illustrative]`: after "the cat sat on the ...", five candidates mat, floor, sofa, roof, car have scores $3,2,1.5,0.5,-1$.

![三張長條圖，標題是 the cat sat on the 之後的下一個詞，五根長條是 mat、floor、sofa、roof、car。紫色 T=0.5：0.839、0.114、0.042、0.006、0.000。藍色 T=1.0：0.591、0.217、0.132、0.049、0.011，其中 roof 與 car 畫上斜線變淡，註明 top-p = 0.9 只留 mat、floor、sofa。橘色 T=2.0：0.400、0.242、0.189、0.115、0.054（Three bar charts for the next word after the cat sat on the, with bars mat, floor, sofa, roof, car. Purple T=0.5: 0.839, 0.114, 0.042, 0.006, 0.000. Blue T=1.0: 0.591, 0.217, 0.132, 0.049, 0.011, with roof and car hatched and faded, noting that top-p = 0.9 keeps mat, floor, sofa. Orange T=2.0: 0.400, 0.242, 0.189, 0.115, 0.054）](../assets/figures/lec11-kp05-temperature.png)

**圖說 `lec11-kp05-temperature`**〔功能：一目了然、精準度〕：由左往右看溫度變高：$T=0.5$ 時 mat 獨佔 $0.839$；$T=2$ 時 mat 只剩 $0.400$，car 也有 $0.054$。再看中間的斜線：$T=1$ 時從大到小累加，$0.5912\to0.8087\to0.9406$，加到 sofa 才達到 $0.9$，所以 Top-$p=0.9$ 留下三個詞，roof 和 car 被丟掉。

**Caption `lec11-kp05-temperature`** [Function: seeing it at a glance, precision]: read left to right as the temperature rises: at $T=0.5$, mat takes $0.839$ alone; at $T=2$, mat has only $0.400$ and even car has $0.054$. Then look at the hatching in the middle: at $T=1$, adding from largest to smallest gives $0.5912\to0.8087\to0.9406$, reaching $0.9$ only at sofa, so Top-$p=0.9$ keeps three words and drops roof and car.

留下的三個詞重新正規化，機率變成 $0.6285,\ 0.2312,\ 0.1402$。若用 Top-$k=2$，只留 mat 和 floor，變成 $0.7311,\ 0.2689$。選出的詞接到輸入的最後，作為下一步的右移輸入，GPT 一類的模型就是這樣一直往下寫。

The three kept words are renormalized to $0.6285,\ 0.2312,\ 0.1402$. With Top-$k=2$, only mat and floor remain, becoming $0.7311,\ 0.2689$. The chosen word is appended to the input as the next step's shifted input; models like GPT keep writing this way.

常見錯誤：以為溫度設成 $0$ 就好。公式裡要除以 $T$，程式直接設 $0$ 會除以零；要確定性的輸出，就直接用貪婪解碼（取最大值）。

Common mistake: thinking the temperature can simply be set to $0$. The formula divides by $T$, so setting it to $0$ in code divides by zero; for a deterministic output, use greedy decoding (take the maximum) directly.

### 本單元重點整理（KP05 summary）

**一句話重點**：解碼器一層有三個子層：遮罩自注意力讓每個位置只看自己和前面，訓練時才能一次平行算；交叉注意力用解碼器的查詢去查編碼器的鍵與值；前饋網路再加工；最後線性層與 softmax 給出詞表上的機率，用貪婪、溫度、Top-$k$ 或 Top-$p$ 選出下一個詞。

**One-sentence summary**: a decoder layer has three sublayers: masked self-attention lets each position see only itself and earlier positions, which allows parallel training; cross-attention uses the decoder's queries to look up the encoder's keys and values; the feed-forward network processes further; finally a linear layer and softmax give probabilities over the vocabulary, and greedy decoding, temperature, Top-$k$ or Top-$p$ picks the next word.

**白話說明**：做克漏字時，老師用紙板蓋住你右邊還沒寫的題目（遮罩）；你可以隨時翻看整篇原文（交叉注意力）；寫完一格，紙板往右移一格（推論時逐字寫）。

**Plain explanation**: in a cloze test, the teacher covers the unanswered blanks to your right with a card (the mask); you may look at the whole source text at any time (cross-attention); after each blank, the card moves one step right (writing word by word in inference).

**正式定義**：$\operatorname{MaskedAttn}=\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}+\mathbf{M})\mathbf{V}$，$M_{ij}=0$（$j\le i$）、$-\infty$（$j>i$）；$\operatorname{CrossAttn}=\operatorname{softmax}(\mathbf{S}\mathbf{W}_Q(\mathbf{H}\mathbf{W}_K)^\top/\sqrt{d_k})\,\mathbf{H}\mathbf{W}_V$；$P(w_k)=e^{z_k/T}/\sum_j e^{z_j/T}$。

**Formal definition**: $\operatorname{MaskedAttn}=\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}+\mathbf{M})\mathbf{V}$ with $M_{ij}=0$ ($j\le i$) and $-\infty$ ($j>i$); $\operatorname{CrossAttn}=\operatorname{softmax}(\mathbf{S}\mathbf{W}_Q(\mathbf{H}\mathbf{W}_K)^\top/\sqrt{d_k})\,\mathbf{H}\mathbf{W}_V$; $P(w_k)=e^{z_k/T}/\sum_j e^{z_j/T}$.

**例子**：遮罩後第2列權重 $[0.0474,0.9526,0]$；交叉注意力 $[0.1636,0.1636,0.6728]$；「the cat sat on the」在 $T=1$ 時 Top-$p=0.9$ 留下 mat、floor、sofa。

**Example**: after masking, row 2 has weights $[0.0474,0.9526,0]$; cross-attention gives $[0.1636,0.1636,0.6728]$; for "the cat sat on the" at $T=1$, Top-$p=0.9$ keeps mat, floor, sofa.

**常見迷思**：

- 「訓練時解碼器也逐字跑。」錯，有遮罩就能一次平行算。
- 「交叉注意力的查詢來自編碼器。」錯，查詢來自解碼器。
- 「交叉注意力也要加因果遮罩。」錯，原文整句都可以看。

**Common misconceptions**:

- "The decoder runs word by word in training too." Wrong: with the mask it computes everything in parallel.
- "The query of cross-attention comes from the encoder." Wrong: it comes from the decoder.
- "Cross-attention also needs a causal mask." Wrong: the whole source may be seen.

**圖**：本單元有兩張圖。

- `lec11-kp05-causal-mask`：分數、加遮罩、softmax 後的權重。
- `lec11-kp05-temperature`：三種溫度下的下一個詞機率與 Top-$p$ 截斷。

**Figures** — this unit has two figures.

- `lec11-kp05-causal-mask`: scores, scores plus mask, and weights after softmax.
- `lec11-kp05-temperature`: next-word probabilities at three temperatures, with Top-$p$ truncation.

### 小練習（Practice）

1. 四個位置的因果遮罩 $\mathbf{M}$ 寫出來是什麼？第 $2$ 列的分數若是 $[0,2,7,7]$（已縮放），遮罩後的權重是多少？
2. 原文 $T_x=12$ 個詞、譯文已寫出 $T_y=3$ 個詞。解碼器的遮罩自注意力權重和交叉注意力權重各是幾乘幾？
3. 兩個候選詞分數 $2$ 和 $0$。分別算 $T=1$ 與 $T=0.5$ 時第一個詞的機率。

1. Write the causal mask $\mathbf{M}$ for four positions. If row $2$ has (scaled) scores $[0,2,7,7]$, what are its weights after masking?
2. The source has $T_x=12$ words and $T_y=3$ target words have been written. What are the shapes of the decoder's masked self-attention weights and cross-attention weights?
3. Two candidate words have scores $2$ and $0$. Compute the first word's probability at $T=1$ and at $T=0.5$.

**第 1 題解答** `[補充]`：$\mathbf{M}=\begin{bmatrix}0&-\infty&-\infty&-\infty\\0&0&-\infty&-\infty\\0&0&0&-\infty\\0&0&0&0\end{bmatrix}$；第 $2$ 列只剩 $[0,2]$，權重 $[e^0,e^2]/(1+e^2)=[0.1192,\ 0.8808,\ 0,\ 0]$。

**Solution to question 1** `[Added]`: $\mathbf{M}=\begin{bmatrix}0&-\infty&-\infty&-\infty\\0&0&-\infty&-\infty\\0&0&0&-\infty\\0&0&0&0\end{bmatrix}$; row $2$ keeps only $[0,2]$, giving weights $[e^0,e^2]/(1+e^2)=[0.1192,\ 0.8808,\ 0,\ 0]$.

**第 2 題解答** `[補充]`：遮罩自注意力是 $3\times3$；交叉注意力是 $3\times12$。

**Solution to question 2** `[Added]`: masked self-attention is $3\times3$; cross-attention is $3\times12$.

**第 3 題解答** `[補充]`：$T=1$：$e^2/(e^2+1)=0.8808$；$T=0.5$：分數變成 $4,0$，$e^4/(e^4+1)=0.9820$。

**Solution to question 3** `[Added]`: $T=1$: $e^2/(e^2+1)=0.8808$; $T=0.5$: the scores become $4,0$, and $e^4/(e^4+1)=0.9820$.

### 小測驗（Quiz，來源原題）

來源這一段有八題自我檢測，保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The source has eight self-check questions for this part, kept with all their options; the source page does not mark the answers, so the explanations below are judged from the source text.

1. 解碼器在生成第 $i$ 個詞時，其自注意力模組接收什麼輸入？
   - (A) 所有未來的目標單詞 $\mathbf{y}_{i\dots n}$
   - (B) 先前已生成的單詞序列 $\{\mathbf{y}_1,\dots,\mathbf{y}_{i-1}\}$ 向右平移後的嵌入
   - (C) 隨機生成的雜訊文字
   - (D) 僅能接收單一標點符號
2. 在訓練階段，解碼器為何能實現完全平行運算？
   - (A) 因為目標序列完整已知，可配合因果遮罩（Causal Mask）一次性平行輸入計算損失
   - (B) 因為訓練時不需要使用損失函數
   - (C) 因為解碼器在訓練時被完全關閉
   - (D) 因為解碼器不需要反向傳播
3. 在遮罩注意力矩陣中，未來位置被加上 $-\infty$ 的根本原因為何？
   - (A) 使矩陣行列式恆等於 0
   - (B) 經過 Softmax 指數運算後 $e^{-\infty}=0$，使未來位置的注意力權重精確為零
   - (C) 防止電腦硬體過熱
   - (D) 為了將矩陣強制轉置
4. 遮罩矩陣 $\mathbf{M}$ 的幾何結構屬於哪種類型？
   - (A) 對角線為 0、右上三角為 $-\infty$ 的下三角矩陣
   - (B) 全為 1 的矩陣
   - (C) 完全對稱的方陣
   - (D) 稀疏單位矩陣
5. 在 Transformer 交叉注意力（Cross-Attention）中，$\mathbf{Q}$ 與 $\mathbf{K},\mathbf{V}$ 分別來自何處？
   - (A) $\mathbf{Q}$ 來自編碼器，$\mathbf{K},\mathbf{V}$ 來自解碼器
   - (B) $\mathbf{Q}$ 來自解碼器前一層，$\mathbf{K},\mathbf{V}$ 來自編碼器頂層輸出
   - (C) 全部來自解碼器自身
   - (D) 全部來自外部隨機常數矩陣
6. 在英翻法任務中，交叉注意力矩陣的主要功能是什麼？
   - (A) 將法文字典按字母順序排序
   - (B) 動態計算目標法文詞與來源英文詞之間的語義對齊關聯
   - (C) 隨機刪除來源句子中的名詞
   - (D) 將英文單詞直接替換為 ASCII 碼
7. 線性輸出層輸出的 Logits 向量維度通常等於什麼？
   - (A) 序列長度 $n$
   - (B) 詞彙表總詞數 $\lvert V\rvert$（Vocabulary Size）
   - (C) 模型層數 $N$
   - (D) 批次大小 Batch Size
8. 在生成採樣時，降低溫度參數 $\tau$（例如從 1.0 降至 0.2）會對輸出產生什麼影響？
   - (A) 使分佈更平坦、隨機性大幅增加
   - (B) 使高機率詞更加突出（分佈更尖銳）、輸出更確定穩定
   - (C) 導致模型無法輸出任何文字
   - (D) 自動將詞彙表大小減半

1. When the decoder generates word $i$, what does its self-attention module receive as input?
   - (A) All future target words $\mathbf{y}_{i\dots n}$
   - (B) The embeddings of the previously generated words $\{\mathbf{y}_1,\dots,\mathbf{y}_{i-1}\}$, shifted right
   - (C) Randomly generated noise text
   - (D) Only a single punctuation mark
2. Why can the decoder compute fully in parallel during training?
   - (A) Because the target sequence is fully known, so with the causal mask it can be fed at once to compute the loss in parallel
   - (B) Because training needs no loss function
   - (C) Because the decoder is switched off during training
   - (D) Because the decoder needs no backpropagation
3. In the masked attention matrix, what is the fundamental reason future positions get $-\infty$ added?
   - (A) To make the determinant always 0
   - (B) After the softmax exponential, $e^{-\infty}=0$, so the attention weight of future positions is exactly zero
   - (C) To keep the hardware from overheating
   - (D) To force the matrix to be transposed
4. What kind of structure does the mask matrix $\mathbf{M}$ have?
   - (A) A lower triangular matrix with 0 on the diagonal and $-\infty$ in the upper-right triangle
   - (B) A matrix of all ones
   - (C) A fully symmetric square matrix
   - (D) A sparse identity matrix
5. In the Transformer's cross-attention, where do $\mathbf{Q}$ and $\mathbf{K},\mathbf{V}$ come from?
   - (A) $\mathbf{Q}$ from the encoder, $\mathbf{K},\mathbf{V}$ from the decoder
   - (B) $\mathbf{Q}$ from the decoder's previous layer, $\mathbf{K},\mathbf{V}$ from the top encoder output
   - (C) All from the decoder itself
   - (D) All from external random constant matrices
6. In an English-to-French task, what is the main function of the cross-attention matrix?
   - (A) To sort the French dictionary alphabetically
   - (B) To compute, dynamically, the semantic alignment between target French words and source English words
   - (C) To delete nouns from the source sentence at random
   - (D) To replace English words with ASCII codes
7. The logits vector output by the linear output layer usually has which dimension?
   - (A) The sequence length $n$
   - (B) The vocabulary size $\lvert V\rvert$
   - (C) The number of layers $N$
   - (D) The batch size
8. When sampling, what does lowering the temperature $\tau$ (for example from 1.0 to 0.2) do to the output?
   - (A) Flattens the distribution and greatly increases randomness
   - (B) Makes high-probability words stand out more (a sharper distribution), giving more certain, stable output
   - (C) Makes the model unable to output any text
   - (D) Halves the vocabulary size automatically

**解析** `[補充]`：第 1 題 (B)、第 2 題 (A)、第 3 題 (B)、第 4 題 (A)、第 5 題 (B)、第 6 題 (B)、第 7 題 (B)、第 8 題 (B)。第 1、2 題見第1節；第 3、4 題見第2節（第 4 題的「下三角」說法見文末更正）；第 5、6 題見第3節，題中的「解碼器前一層」就是同一層遮罩自注意力子層的輸出；第 7、8 題見第4節，題中的 $\tau$ 就是本講義的 $T$。

**Explanations** `[Added]`: 1 B, 2 A, 3 B, 4 A, 5 B, 6 B, 7 B, 8 B. Questions 1 and 2, see Section 1; 3 and 4, see Section 2 (on the "lower triangular" wording of question 4, see the corrections at the end); 5 and 6, see Section 3, where the questions' "decoder's previous layer" means the output of the same layer's masked self-attention sublayer; 7 and 8, see Section 4, where the $\tau$ of the questions is this note's $T$.

### 留給下一單元的問題（The question for the next unit）

到這裡，整個 Transformer 都組好了：編碼器、解碼器、輸出層，裡面完全沒有 RNN。拿掉 RNN 到底換到了什麼？每層的計算量、能不能平行、遠處的詞要隔幾步才能互相影響，和 RNN、CNN 比起來如何？只用編碼器或只用解碼器的 BERT 和 GPT 又是怎麼來的？下一單元（本週 KP06）回答。

At this point the whole Transformer is assembled: encoder, decoder and output layer, with no RNN anywhere. What exactly do we gain by removing the RNN? How do the computation per layer, the parallelism, and the number of steps between distant words compare with RNNs and CNNs? And where do BERT and GPT, which use only the encoder or only the decoder, come from? The next unit (this week's KP06) answers.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第9章「注意力機制與 Transformer 核心原理」§3.3–§3.3.3 與教授祕笈「因果遮罩與自回歸生成」「交叉注意力與編碼解碼通信」「Softmax 溫度係數與取樣策略」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §3.3–§3.3.3 of Chapter 9, "Core principles of attention mechanisms and the Transformer", of CCC-LNS IDL_115, and from the professor's notes "Causal masking and autoregressive decoding", "Cross-attention and encoder-decoder communication" and "Softmax temperature and sampling strategies" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源把遮罩 $\mathbf{M}$ 稱為「下三角矩陣」。嚴格說，$\mathbf{M}$ 是右上角為 $-\infty$、其餘為 $0$ 的矩陣；真正呈下三角形狀的是 softmax 之後的**權重矩陣**。本講義這樣區分。祕笈又說輸出是「嚴格的下三角結構」；權重矩陣的對角線不是 $0$，所以它是下三角，但不是嚴格下三角。
- 來源正文把遮罩寫在縮放的括號裡，$\operatorname{softmax}(\frac{1}{\sqrt p}(\mathbf{Q}^\top\mathbf{K}+\mathbf{M}))$，$p$ 就是 $d_k$；因為 $-\infty/\sqrt p$ 仍是 $-\infty$，結果相同。本講義依祕笈寫成 $\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}+\mathbf{M}$，並改用「列為詞」的記號。來源用粗斜體 `\boldsymbol`，本課程網站不支援，改用 `\mathbf`。
- 祕笈的交叉注意力例子寫 $[0.165,0.165,0.670]$；精算是 $[0.1636,0.1636,0.6728]$，本講義用精算值。
- 來源的解碼器小節夾著一份前饋網路的提示詞與祕笈連結，與該小節內容不符；本講義不採用。
- 本講義補上：右移輸入的完整例子、換掉未來詞時前面輸出不變的程式驗證、交叉注意力的形狀、輸出層與詞嵌入共用權重、「the cat sat on the」的溫度與 Top-$k$／Top-$p$ 示意與兩張圖、三題小練習與解答。
- 取捨：KV 快取的記憶體公式、填充遮罩（padding mask）、束搜尋（beam search）、多模態與擴散模型的交叉注意力、物理與退火的類比，本講義不寫或只帶過。

- The source calls the mask $\mathbf{M}$ a "lower triangular matrix". Strictly, $\mathbf{M}$ has $-\infty$ in the upper right and $0$ elsewhere; what is truly lower triangular is the **weight matrix** after softmax. This note keeps the two apart. The notes also call the output "strictly lower triangular"; the diagonal weights are not $0$, so it is lower triangular but not strictly so.
- The source's main text puts the mask inside the scaling, $\operatorname{softmax}(\frac{1}{\sqrt p}(\mathbf{Q}^\top\mathbf{K}+\mathbf{M}))$, where $p$ is $d_k$; since $-\infty/\sqrt p$ is still $-\infty$, the result is the same. Following the professor's notes, this note writes $\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}+\mathbf{M}$ in the "one word per row" notation. The source uses bold italic `\boldsymbol`, which the course website does not support, so this note uses `\mathbf`.
- The professor's notes give the cross-attention example as $[0.165,0.165,0.670]$; the exact values are $[0.1636,0.1636,0.6728]$, used here.
- The source's decoder subsection contains an image prompt and a notes link about the feed-forward network, which do not match the subsection; this note does not use them.
- This note adds: a full example of the shifted-right input, a code check that earlier outputs stay unchanged when a future word is replaced, the shapes of cross-attention, the weight sharing between the output layer and the word embedding, the temperature and Top-$k$/Top-$p$ illustration for "the cat sat on the" with two figures, and three practice questions with solutions.
- Choices: the KV-cache memory formula, the padding mask, beam search, cross-attention in multimodal and diffusion models, and the analogies with physics and annealing are left out or only mentioned.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 解碼器 | decoder |
| 遮罩多頭自注意力 | masked multi-head self-attention |
| 因果遮罩 | causal mask |
| 右移輸入 | shifted-right input |
| 開始記號 | start token |
| 教師強制 | teacher forcing |
| 自回歸生成 | autoregressive generation |
| 交叉注意力（編碼器—解碼器注意力） | cross-attention (encoder-decoder attention) |
| 詞表 | vocabulary |
| 子詞 | subword |
| 未正規化分數 | logits |
| 貪婪解碼 | greedy decoding |
| 溫度 | temperature |
| Top-$k$ 取樣 | top-$k$ sampling |
| 核取樣 | top-$p$ (nucleus) sampling |
| 束搜尋 | beam search |
| 填充遮罩 | padding mask |
| 鍵值快取 | key-value cache (KV cache) |
| 生成式預訓練模型（本週 KP06） | GPT (generative pre-trained Transformer) |
| 雙向編碼器模型（本週 KP06） | BERT |
| 循環神經網路（第8週） | recurrent neural network (RNN) |
| 卷積神經網路 | convolutional neural network (CNN) |
| 美國資訊交換標準碼 | ASCII |
