# 第10週 KP04：Query、Key、Value，注意力是一種「軟性」的查表（Query, Key and Value: attention is a "soft" table lookup）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第10週的第四個知識點（KP04），接在 KP03 之後。KP03 的編碼器狀態 $\mathbf{h}_j$ 一身兼兩職：拿來和查詢比較，也拿來被加權取用。這一節把這兩個角色拆開，寫成一般的查詢機制：Query、Key 與 Value。

This is the fourth knowledge point (KP04) of Week 10 and follows KP03. In KP03 each encoder state $\mathbf{h}_j$ does two jobs: it is compared with the query, and it is the content that gets weighted and taken. This note splits the two roles and writes a general lookup mechanism: Query, Key and Value.

先備知識是 KP01 的分數、softmax 與加權平均，以及 KP03 的分數函數。本講義依課程網頁教材 CCC-LNS 第9章改寫，文中「來源」就是指它；本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec10-kp04-figures.py` 算出，可以自己重算。

The prerequisites are the scores, softmax and weighted average of KP01, and the score functions of KP03. This note is rewritten from Chapter 9 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec10-kp04-figures.py`, so you can recompute it yourself.

## KP04 Query、Key、Value，注意力是一種「軟性」的查表（Query, Key and Value: attention is a "soft" table lookup）

### 學習目標（Learning objectives）

讀完之後，你能用查表的比喻說明 Query、Key、Value 各自的角色。你能說出「硬查表」和注意力的「軟查表」差在哪裡，並算出兩者的輸出。你能寫出注意力的一般式子，並說明為什麼要把 Key 和 Value 分開。你也能比較內積、縮放內積、餘弦三種相似度給出的權重，並指出來源的「加法相似度」為什麼和查詢無關。

After reading, you can explain the roles of Query, Key and Value with the analogy of a table lookup. You can state how a "hard lookup" differs from the "soft lookup" of attention, and compute the output of both. You can write the general formula of attention and explain why Key and Value are separated. You can also compare the weights given by three similarity measures, inner product, scaled inner product and cosine, and point out why the source's "additive similarity" does not depend on the query.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{q}$ | 查詢（Query）：要找什麼 | query: what we are looking for |
| $\mathbf{k}_i$ | 第 $i$ 筆資料的鍵（Key）：拿來比對的標籤 | key of item $i$: the label used for matching |
| $\mathbf{v}_i$ | 第 $i$ 筆資料的值（Value）：真正取出的內容 | value of item $i$: the content actually taken |
| $s_i$ | 查詢和第 $i$ 個鍵的相似度分數 | similarity score of the query and key $i$ |
| $a_i$ | 第 $i$ 筆資料的權重 | weight of item $i$ |
| $n$ | 資料筆數 | number of items |
| $p$ | 查詢與鍵的維度（實作常記為 $d_k$） | dimension of the query and the keys (often written $d_k$ in code) |
| $r$ | 值的維度（實作常記為 $d_v$） | dimension of the values (often written $d_v$ in code) |
| $\mathbf{W}$ | 廣義內積的可學習 $p\times p$ 矩陣 | learnable $p\times p$ matrix of the general inner product |
| $\mathbf{w}_q,\ \mathbf{w}_k$ | 加法相似度的可學習向量 | learnable vectors of the additive similarity |
| $\lVert\mathbf{x}\rVert$ | 向量的長度 | length of a vector |

### 1. 查表：用鍵找資料，取出值（Table lookup: find items by key, take out the value）

先看一般的查表。一本字典，每一條有一個**鍵**（key），例如詞目「蘋果」，和一個**值**（value），例如它的解釋。你手上有一個**查詢**（query），例如「蘋果」。查表就是拿查詢逐條和鍵比對，完全相同的那一條，取出它的值。

First look at an ordinary table lookup. In a dictionary, every entry has a **key**, for example the headword "apple", and a **value**, for example its definition. You hold a **query**, for example "apple". Looking up means comparing the query with every key, and taking out the value of the entry whose key matches exactly.

來源的比喻是圖書館：查詢是你心中想找的主題，鍵是每本書封面上的標籤，值是書裡的內容。你拿主題去對標籤，找到對的書，再把書的內容讀進腦中。

The source's analogy is a library: the query is the topic you want, the keys are the labels on the book covers, and the values are the contents of the books. You match your topic against the labels, find the right book and read its content into your head.

注意鍵和值是兩樣東西：比對用的是封面標籤，取出來的是書的內容。KP03 的 $\mathbf{h}_j$ 同時當標籤和內容；這一節要把它們分開，記成 $\mathbf{k}_i$ 和 $\mathbf{v}_i$。

Note that the key and the value are two different things: you match on the cover label but take out the book's content. The $\mathbf{h}_j$ of KP03 served as both label and content; this note separates them and writes $\mathbf{k}_i$ and $\mathbf{v}_i$.

常見錯誤：以為 Query、Key、Value 是三種不同的資料來源。它們是三種**角色**：同一個向量經過不同的轉換，可以同時扮演三個角色；KP05 的自注意力就是這樣。

Common error: thinking that Query, Key and Value are three different sources of data. They are three **roles**: the same vector, transformed in different ways, can play all three at once; the self-attention of KP05 does exactly this.

### 2. 從硬查表到軟查表（From a hard lookup to a soft lookup）

一般的查表是**硬查表**：只取最相符的那一條，其他完全不理。問題和 KP01 的硬選擇一樣：結果對分數的變化不平滑，沒辦法用反向傳播學習。而且查詢不一定剛好等於某個鍵，「最像」的那條也不見得是唯一有用的。

An ordinary lookup is a **hard lookup**: take only the best-matching entry and ignore all others. The problem is the same as the hard selection of KP01: the result does not change smoothly with the scores, so it cannot be learned with backpropagation. Also, the query need not equal any key exactly, and the "most similar" entry is not necessarily the only useful one.

注意力把它變成**軟查表**（soft lookup）：每一條都取一點，取多少由查詢和鍵的相似度決定。寫成式子：

Attention turns it into a **soft lookup**: take a little from every entry, with the amount set by how similar the query is to the key. As formulas:

$$
s_i=\operatorname{similarity}(\mathbf{q},\mathbf{k}_i),\qquad a_i=\frac{\exp(s_i)}{\sum_{j=1}^{n}\exp(s_j)},\qquad \operatorname{attention}(\mathbf{q},\{\mathbf{k}_i\},\{\mathbf{v}_i\})=\sum_{i=1}^{n}a_i\,\mathbf{v}_i.
$$

和 KP01 比，唯一的差別是：分數用鍵 $\mathbf{k}_i$ 算，加權的卻是值 $\mathbf{v}_i$。

Compared with KP01, the only difference is that the scores are computed from the keys $\mathbf{k}_i$, while the weighted items are the values $\mathbf{v}_i$.

主例子 `[補充]`：鍵沿用 KP01 的三個向量 $\mathbf{k}_1=[1,0]^\top$、$\mathbf{k}_2=[0,1]^\top$、$\mathbf{k}_3=[1,1]^\top$；值換成另外三個向量 $\mathbf{v}_1=[10,0]^\top$、$\mathbf{v}_2=[0,10]^\top$、$\mathbf{v}_3=[5,5]^\top$；查詢 $\mathbf{q}=[2,1]^\top$，相似度用內積。

Main example `[Added]`: the keys are the three vectors of KP01, $\mathbf{k}_1=[1,0]^\top$, $\mathbf{k}_2=[0,1]^\top$, $\mathbf{k}_3=[1,1]^\top$; the values are three other vectors, $\mathbf{v}_1=[10,0]^\top$, $\mathbf{v}_2=[0,10]^\top$, $\mathbf{v}_3=[5,5]^\top$; the query is $\mathbf{q}=[2,1]^\top$, and the similarity is the inner product.

- 分數：$s_1=2$、$s_2=1$、$s_3=2+1=3$。
- 指數：$7.389$、$2.718$、$20.086$，總和 $30.193$。
- 權重：$0.2447$、$0.0900$、$0.6652$。
- 軟查表的輸出：$0.2447\,\mathbf{v}_1+0.0900\,\mathbf{v}_2+0.6652\,\mathbf{v}_3=[5.7735,\ 4.2265]^\top$。
- 硬查表只取分數最高的第 $3$ 條，輸出就是 $\mathbf{v}_3=[5,5]^\top$。

- Scores: $s_1=2$, $s_2=1$, $s_3=2+1=3$.
- Exponentials: $7.389$, $2.718$, $20.086$, total $30.193$.
- Weights: $0.2447$, $0.0900$, $0.6652$.
- Output of the soft lookup: $0.2447\,\mathbf{v}_1+0.0900\,\mathbf{v}_2+0.6652\,\mathbf{v}_3=[5.7735,\ 4.2265]^\top$.
- The hard lookup takes only entry $3$, with the highest score, so its output is $\mathbf{v}_3=[5,5]^\top$.

![兩張長條圖，查詢 q=[2,1]，三個鍵的分數 2、1、3。左圖是硬查表，只有第三個鍵的權重是 1，其他是 0，輸出 [5,5]。右圖是軟查表，權重 0.2447、0.0900、0.6652，輸出 [5.7735, 4.2265]（Hard lookup puts weight 1 on key 3 and outputs [5,5]; soft lookup spreads the weights as 0.2447, 0.0900, 0.6652 and outputs [5.7735, 4.2265]）](../assets/figures/lec10-kp04-hard-vs-soft.png)

**圖說 `lec10-kp04-hard-vs-soft`**〔功能：一目了然、精準度〕：先看左圖：硬查表把全部份量給分數最高的 $\mathbf{k}_3$，輸出就是 $\mathbf{v}_3=[5,5]^\top$。再看右圖：軟查表三條都有份量，$\mathbf{k}_3$ 仍然最多（$0.6652$），但 $\mathbf{k}_1$ 也分到 $0.2447$，所以輸出被 $\mathbf{v}_1=[10,0]^\top$ 往第一個分量拉，變成 $[5.7735,\ 4.2265]^\top$。

**Caption `lec10-kp04-hard-vs-soft`** [Function: seeing it at a glance, precision]: First look at the left panel: the hard lookup gives the whole share to $\mathbf{k}_3$, the highest score, so the output is $\mathbf{v}_3=[5,5]^\top$. Then look at the right panel: the soft lookup gives every entry a share; $\mathbf{k}_3$ still gets the most ($0.6652$), but $\mathbf{k}_1$ also gets $0.2447$, so the output is pulled toward the first component by $\mathbf{v}_1=[10,0]^\top$ and becomes $[5.7735,\ 4.2265]^\top$.

常見錯誤：以為輸出一定等於某一個值。軟查表的輸出是所有值的加權平均，通常不會剛好等於任何一個 $\mathbf{v}_i$；只有權重幾乎全給一條時，才接近那一條。

Common error: thinking that the output must equal one of the values. The output of a soft lookup is a weighted average of all values and usually equals none of the $\mathbf{v}_i$ exactly; it is close to one of them only when nearly all the weight goes there.

### 3. 為什麼要把 Key 和 Value 分開（Why separate Key from Value）

如果像 KP03 一樣讓 $\mathbf{k}_i=\mathbf{v}_i=\mathbf{h}_i$，就等於規定「拿來比對的特徵」和「拿來傳遞的特徵」必須一模一樣。很多時候這兩件事要的東西不同。

If, as in KP03, $\mathbf{k}_i=\mathbf{v}_i=\mathbf{h}_i$, that forces "the features used for matching" and "the features passed on" to be identical. Often the two jobs want different things.

舉例 `[補充]`：在一句話裡找「這個代名詞指誰」，比對時要看的是「是不是人、單數還是複數」這類線索；找到之後，要傳遞的卻是那個名詞的意思。用同一個向量同時做好兩件事，比較困難。

An example `[Added]`: to find "whom this pronoun refers to" in a sentence, the matching should look at clues such as "is it a person, singular or plural"; once found, what should be passed on is the meaning of that noun. Doing both well with a single vector is harder.

所以實際的模型會用不同的可學習矩陣，把同一份資料分別轉換成鍵與值：比對用一種轉換，傳遞用另一種。來源稱之為**解耦**（decoupling）。在 KP03 的翻譯模型裡，查詢來自解碼器、鍵與值來自編碼器；到了 KP05 的自注意力，查詢、鍵、值三者都由同一份輸入各自轉換而來。

So real models use different learnable matrices to turn the same data into keys and into values: one transformation for matching, another for passing on. The source calls this **decoupling**. In the translation model of KP03 the query comes from the decoder and the keys and values from the encoder; in the self-attention of KP05, the query, key and value are all transformed from the same input.

鍵和值的維度也不必相同。鍵要和查詢算內積，所以兩者長度一樣，都是 $p$；值可以是另一個長度，例如 $r$，輸出就跟著是 $r$ 維。

The keys and values need not have the same dimension either. A key must have an inner product with the query, so the two have the same length $p$; a value can have another length, say $r$, and the output then has $r$ dimensions too.

常見錯誤：把 $p$ 和 $r$（實作裡寫成 $d_k$ 和 $d_v$，鍵與值的維度）當成必須相等。實作裡常把兩者設成一樣，是為了方便，不是數學上的要求。

Common error: treating $p$ and $r$ (written $d_k$ and $d_v$ in code, the dimensions of keys and values) as necessarily equal. Implementations often set them equal for convenience, not because the math requires it.

### 4. 相似度怎麼量（How to measure similarity）

第2節的 $\operatorname{similarity}$ 有幾種選擇。來源列出四種，前三種在 KP03 的分數函數表裡出現過：

The $\operatorname{similarity}$ of Section 2 has several options. The source lists four; the first three appeared in the score-function table of KP03:

- 內積：$s_i=\mathbf{q}^\top\mathbf{k}_i$。
- 縮放內積（最常用）：$s_i=\mathbf{q}^\top\mathbf{k}_i/\sqrt{p}$。
- 廣義內積：$s_i=\mathbf{q}^\top\mathbf{W}\mathbf{k}_i$，$\mathbf{W}$ 是可學習的 $p\times p$ 矩陣。
- 加法相似度：$s_i=\mathbf{w}_q^\top\mathbf{q}+\mathbf{w}_k^\top\mathbf{k}_i$。

- Inner product: $s_i=\mathbf{q}^\top\mathbf{k}_i$.
- Scaled inner product (the most common): $s_i=\mathbf{q}^\top\mathbf{k}_i/\sqrt{p}$.
- General inner product: $s_i=\mathbf{q}^\top\mathbf{W}\mathbf{k}_i$, with a learnable $p\times p$ matrix $\mathbf{W}$.
- Additive similarity: $s_i=\mathbf{w}_q^\top\mathbf{q}+\mathbf{w}_k^\top\mathbf{k}_i$.

另外常見的還有**餘弦相似度**（cosine similarity）$\mathbf{q}^\top\mathbf{k}_i/(\lVert\mathbf{q}\rVert\,\lVert\mathbf{k}_i\rVert)$：先把兩個向量的長度都除掉，只看夾角 `[補充]`。用主例子比較三種；主例子的維度 $p=2$，所以縮放內積是除以 $\sqrt{2}$：

Another common choice is the **cosine similarity** $\mathbf{q}^\top\mathbf{k}_i/(\lVert\mathbf{q}\rVert\,\lVert\mathbf{k}_i\rVert)$: divide out the lengths of both vectors and look only at the angle `[Added]`. Compare three of them on the main example; its dimension is $p=2$, so the scaled inner product divides by $\sqrt{2}$:

| 相似度 | 三個分數 | 三個權重 | 輸出 |
|---|---|---|---|
| 內積 | $2,\ 1,\ 3$ | $0.2447,\ 0.0900,\ 0.6652$ | $[5.7735,\ 4.2265]^\top$ |
| 縮放內積（$\sqrt{2}$） | $1.4142,\ 0.7071,\ 2.1213$ | $0.2840,\ 0.1400,\ 0.5760$ | $[5.7198,\ 4.2802]^\top$ |
| 餘弦 | $0.8944,\ 0.4472,\ 0.9487$ | $0.3710,\ 0.2372,\ 0.3917$ | $[5.6690,\ 4.3310]^\top$ |

| Similarity | Three scores | Three weights | Output |
|---|---|---|---|
| Inner product | $2,\ 1,\ 3$ | $0.2447,\ 0.0900,\ 0.6652$ | $[5.7735,\ 4.2265]^\top$ |
| Scaled inner product ($\sqrt{2}$) | $1.4142,\ 0.7071,\ 2.1213$ | $0.2840,\ 0.1400,\ 0.5760$ | $[5.7198,\ 4.2802]^\top$ |
| Cosine | $0.8944,\ 0.4472,\ 0.9487$ | $0.3710,\ 0.2372,\ 0.3917$ | $[5.6690,\ 4.3310]^\top$ |

![同一個查詢和三個鍵，三種相似度的 softmax 權重並排長條圖。內積（紅）最集中，k_3 拿 0.665；縮放內積（藍）k_3 拿 0.576；餘弦（綠）最平均，k_1 0.371、k_3 0.392 幾乎一樣（Three similarity measures on the same query and keys: the inner product is the most peaked, the scaled inner product less so, and cosine is the flattest, giving k_1 and k_3 almost the same weight）](../assets/figures/lec10-kp04-similarity-measures.png)

**圖說 `lec10-kp04-similarity-measures`**〔功能：精準度〕：先看紅色：內積的權重最集中，$\mathbf{k}_3$ 拿 $0.665$。再看藍色：除以 $\sqrt{2}$ 以後分數差距變小，權重比較平均。最後看綠色：餘弦只看夾角，$\mathbf{k}_1$ 和 $\mathbf{k}_3$ 跟查詢的夾角分別約 $27^\circ$ 和 $18^\circ$，相差不大，權重 $0.371$ 和 $0.392$ 也很接近；長度較長的 $\mathbf{k}_3$ 在內積下的優勢（$0.665$）大幅縮小了。

**Caption `lec10-kp04-similarity-measures`** [Function: precision]: First look at the red bars: the inner product gives the most concentrated weights, $0.665$ for $\mathbf{k}_3$. Then look at the blue bars: after dividing by $\sqrt{2}$ the score gaps shrink and the weights are more even. Finally look at the green bars: cosine looks only at angles; $\mathbf{k}_1$ and $\mathbf{k}_3$ make angles of about $27^\circ$ and $18^\circ$ with the query, not far apart, and their weights $0.371$ and $0.392$ are close; the advantage that the longer $\mathbf{k}_3$ has under the inner product ($0.665$) shrinks a lot.

來源的加法相似度要特別小心 `[補充]`：$\mathbf{w}_q^\top\mathbf{q}$ 這一項對每個鍵都一樣。KP01 說過，每個分數加同一個數，softmax 的權重不變，所以這一項會被抵銷，權重**完全不受查詢影響**。例如取 $\mathbf{w}_q=[1,0]^\top$、$\mathbf{w}_k=[1,1]^\top$，查詢 $[2,1]^\top$ 的分數是 $3,3,4$，查詢 $[0,3]^\top$ 的分數是 $1,1,2$，兩者的權重都是 $0.2119,\ 0.2119,\ 0.5761$。KP03 的 Bahdanau 加性分數不會這樣：相加之後還經過非線性的 $\tanh$，查詢對分數的貢獻會隨著鍵而改變，不再是對每個鍵都相同的一個數，所以不會被抵銷。

The additive similarity of the source needs care `[Added]`: the term $\mathbf{w}_q^\top\mathbf{q}$ is the same for every key. KP01 showed that adding the same number to every score leaves the softmax weights unchanged, so this term cancels, and the weights **do not depend on the query at all**. For example, with $\mathbf{w}_q=[1,0]^\top$ and $\mathbf{w}_k=[1,1]^\top$, the query $[2,1]^\top$ gives scores $3,3,4$ and the query $[0,3]^\top$ gives $1,1,2$; both give the weights $0.2119,\ 0.2119,\ 0.5761$. The additive Bahdanau score of KP03 does not have this problem: after the addition it passes through the nonlinear $\tanh$, so the query's contribution to the score changes with the key and is no longer the same number for every key, and it does not cancel.

常見錯誤：以為除以 $\sqrt{p}$ 只是讓數字好看。它改變了權重的尖銳程度，在高維時非常重要；KP05 會算給你看。

Common error: thinking that dividing by $\sqrt{p}$ only makes the numbers look nicer. It changes how sharp the weights are and matters a lot in high dimensions; KP05 works it out.

### 本單元重點整理（KP04 summary）

**一句話重點**：注意力是軟查表：查詢 $\mathbf{q}$ 和每個鍵 $\mathbf{k}_i$ 算相似度，經 softmax 變成權重，再把對應的值 $\mathbf{v}_i$ 加權平均；把鍵和值分開，讓「拿來比對的」和「拿來取用的」可以是不同的特徵。

**One-sentence summary**: attention is a soft lookup: the query $\mathbf{q}$ is compared with every key $\mathbf{k}_i$, the similarities become weights through softmax, and the corresponding values $\mathbf{v}_i$ are averaged with these weights; separating keys from values lets "what is matched" and "what is taken" be different features.

**白話說明**：去圖書館找資料，你不只拿走書名最符合的那一本，而是每本都翻一翻，越相關的讀得越多，最後把讀到的內容依比例混在一起。

**Plain explanation**: in a library you do not take only the book whose title fits best; you leaf through every book, read more of the more relevant ones, and finally blend what you read in proportion.

**正式定義**：$s_i=\operatorname{similarity}(\mathbf{q},\mathbf{k}_i)$，$a_i=\exp(s_i)/\sum_j\exp(s_j)$，$\operatorname{attention}(\mathbf{q},\{\mathbf{k}_i\},\{\mathbf{v}_i\})=\sum_i a_i\mathbf{v}_i$；常用 $\operatorname{similarity}(\mathbf{q},\mathbf{k})=\mathbf{q}^\top\mathbf{k}/\sqrt{p}$。

**Formal definition**: $s_i=\operatorname{similarity}(\mathbf{q},\mathbf{k}_i)$, $a_i=\exp(s_i)/\sum_j\exp(s_j)$, $\operatorname{attention}(\mathbf{q},\{\mathbf{k}_i\},\{\mathbf{v}_i\})=\sum_i a_i\mathbf{v}_i$; a common choice is $\operatorname{similarity}(\mathbf{q},\mathbf{k})=\mathbf{q}^\top\mathbf{k}/\sqrt{p}$.

**例子**：$\mathbf{q}=[2,1]^\top$，鍵 $[1,0]^\top,[0,1]^\top,[1,1]^\top$，值 $[10,0]^\top,[0,10]^\top,[5,5]^\top$。內積權重 $0.2447,\ 0.0900,\ 0.6652$，輸出 $[5.7735,\ 4.2265]^\top$；硬查表輸出 $[5,5]^\top$。

**Example**: $\mathbf{q}=[2,1]^\top$, keys $[1,0]^\top,[0,1]^\top,[1,1]^\top$, values $[10,0]^\top,[0,10]^\top,[5,5]^\top$. The inner-product weights are $0.2447,\ 0.0900,\ 0.6652$ and the output is $[5.7735,\ 4.2265]^\top$; the hard lookup outputs $[5,5]^\top$.

**常見迷思**：

- 「Query、Key、Value 是三份不同的資料。」錯，它們是三種角色，可以來自同一份輸入。
- 「輸出一定是某一個值。」錯，它是所有值的加權平均。
- 「鍵和值的維度必須相同。」錯，只有查詢和鍵必須同維度。

**Common misconceptions**:

- "Query, Key and Value are three different pieces of data." Wrong: they are three roles and can come from the same input.
- "The output must be one of the values." Wrong: it is a weighted average of all values.
- "Keys and values must have the same dimension." Wrong: only the query and the keys must.

**圖**：本單元有兩張圖。

- `lec10-kp04-hard-vs-soft`：同一個查詢下，硬查表與軟查表的權重與輸出。
- `lec10-kp04-similarity-measures`：內積、縮放內積、餘弦三種相似度給出的權重。

**Figures** — this unit has two figures.

- `lec10-kp04-hard-vs-soft`: the weights and outputs of a hard and a soft lookup for the same query.
- `lec10-kp04-similarity-measures`: the weights given by the inner product, the scaled inner product and cosine.

### 小練習（Practice）

1. 三筆資料的權重是 $0.7$、$0.2$、$0.1$，值是 $\mathbf{v}_1=[1,0]^\top$、$\mathbf{v}_2=[0,1]^\top$、$\mathbf{v}_3=[1,1]^\top$。注意力的輸出是什麼？
2. 沿用主例子的鍵和值，查詢改成 $\mathbf{q}=[0,2]^\top$，相似度用內積。算出分數、權重與輸出。
3. 用一句話解釋：為什麼來源的加法相似度 $\mathbf{w}_q^\top\mathbf{q}+\mathbf{w}_k^\top\mathbf{k}_i$ 給出的權重和查詢無關？

1. Three items have weights $0.7$, $0.2$, $0.1$ and values $\mathbf{v}_1=[1,0]^\top$, $\mathbf{v}_2=[0,1]^\top$, $\mathbf{v}_3=[1,1]^\top$. What is the output of attention?
2. Keep the keys and values of the main example and change the query to $\mathbf{q}=[0,2]^\top$, with the inner product as similarity. Compute the scores, the weights and the output.
3. In one sentence: why are the weights given by the source's additive similarity $\mathbf{w}_q^\top\mathbf{q}+\mathbf{w}_k^\top\mathbf{k}_i$ independent of the query?

**第 1 題解答** `[補充]`：$0.7[1,0]^\top+0.2[0,1]^\top+0.1[1,1]^\top=[0.8,\ 0.3]^\top$。

**Solution to question 1** `[Added]`: $0.7[1,0]^\top+0.2[0,1]^\top+0.1[1,1]^\top=[0.8,\ 0.3]^\top$.

**第 2 題解答** `[補充]`：分數 $0,\ 2,\ 2$，權重 $0.0634,\ 0.4683,\ 0.4683$。輸出 $0.0634[10,0]^\top+0.4683[0,10]^\top+0.4683[5,5]^\top=[2.9753,\ 7.0247]^\top$，往第二個分量偏。

**Solution to question 2** `[Added]`: the scores are $0,\ 2,\ 2$ and the weights $0.0634,\ 0.4683,\ 0.4683$. The output is $0.0634[10,0]^\top+0.4683[0,10]^\top+0.4683[5,5]^\top=[2.9753,\ 7.0247]^\top$, leaning toward the second component.

**第 3 題解答** `[補充]`：$\mathbf{w}_q^\top\mathbf{q}$ 對每個鍵都是同一個數，而 softmax 對「每個分數加同一個數」不敏感，所以這一項被抵銷，只剩 $\mathbf{w}_k^\top\mathbf{k}_i$ 決定權重。

**Solution to question 3** `[Added]`: $\mathbf{w}_q^\top\mathbf{q}$ is the same number for every key, and softmax does not react to "adding the same number to every score", so this term cancels and only $\mathbf{w}_k^\top\mathbf{k}_i$ decides the weights.

### 小測驗（Quiz，來源原題）

來源的兩題自我檢測保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The two self-check questions of the source are kept with all options; the source page does not mark the answers directly, and the explanations below follow the source text.

1. 在縮放內積注意力中，除以 $\sqrt{p}$（維度開根號）的主要目的為何？
   - (A) 為了讓數值強制等於 $0$
   - (B) 防止高維向量內積過大推入 Softmax 飽和區導致梯度消失
   - (C) 為了加速矩陣轉置
   - (D) 為了將浮點數轉換為整數
2. 若 Query 與三個 Keys 的相似度權重分別為 $0.7$、$0.2$、$0.1$，對應 Values 為 $\mathbf{v}_1,\mathbf{v}_2,\mathbf{v}_3$，則注意力輸出為何？
   - (A) $0.7\mathbf{v}_1+0.2\mathbf{v}_2+0.1\mathbf{v}_3$
   - (B) $\mathbf{v}_1+\mathbf{v}_2+\mathbf{v}_3$
   - (C) $0.7\mathbf{v}_1\times0.2\mathbf{v}_2\times0.1\mathbf{v}_3$
   - (D) $\max(\mathbf{v}_1,\mathbf{v}_2,\mathbf{v}_3)$

1. In scaled inner-product attention, what is the main purpose of dividing by $\sqrt{p}$ (the square root of the dimension)?
   - (A) To force the values to $0$
   - (B) To keep large inner products of high-dimensional vectors from pushing softmax into its saturated region, where gradients vanish
   - (C) To speed up matrix transposition
   - (D) To turn floating-point numbers into integers
2. If the similarity weights of a query and three keys are $0.7$, $0.2$ and $0.1$, with values $\mathbf{v}_1,\mathbf{v}_2,\mathbf{v}_3$, what is the attention output?
   - (A) $0.7\mathbf{v}_1+0.2\mathbf{v}_2+0.1\mathbf{v}_3$
   - (B) $\mathbf{v}_1+\mathbf{v}_2+\mathbf{v}_3$
   - (C) $0.7\mathbf{v}_1\times0.2\mathbf{v}_2\times0.1\mathbf{v}_3$
   - (D) $\max(\mathbf{v}_1,\mathbf{v}_2,\mathbf{v}_3)$

**解析** `[補充]`：第 1 題選 (B)：KP01 第4節說過分數整體放大時權重會變尖；KP05 會算出高維時內積有多大，以及權重變尖為什麼會讓梯度消失。第 2 題選 (A)：這就是第2節的 $\sum_i a_i\mathbf{v}_i$，也是小練習第 1 題。

**Explanations** `[Added]`: question 1 is B: Section 4 of KP01 showed that scaling up all scores makes the weights sharper; KP05 computes how large inner products get in high dimensions and why sharp weights make gradients vanish. Question 2 is A: this is the $\sum_i a_i\mathbf{v}_i$ of Section 2, and also practice question 1.

### 留給下一單元的問題（The question for the next unit）

到目前為止，查詢都來自別處：KP03 的查詢是解碼器狀態，鍵和值是編碼器狀態。如果查詢、鍵、值全部來自同一句話，讓每個詞都拿自己去查同一句裡的其他詞，會得到什麼？下一單元（KP05）的自注意力回答這個問題。

So far the query has always come from somewhere else: in KP03 the query is the decoder state, and the keys and values are encoder states. What happens if the queries, keys and values all come from the same sentence, so that every word uses itself to look up the other words of that sentence? The self-attention of the next unit (KP05) answers this question.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第9章「注意力機制與 Transformer 核心原理」§2.4.2 與教授祕笈「Query-Key-Value 檢索機制」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §2.4.2 of Chapter 9, "Core principles of attention mechanisms and the Transformer", of CCC-LNS IDL_115 and its professor's notes "The Query-Key-Value retrieval mechanism" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源的「加法相似度」$\mathbf{w}_q^\top\mathbf{q}+\mathbf{w}_k^\top\mathbf{k}_i$ 經過 softmax 後與查詢無關（第4節的數字例）；本講義照錄來源的式子，並註明這一點。實務上的加性注意力是 KP03 的 Bahdanau 形式，相加後還有 $\tanh$。
- 來源用粗斜體 `\boldsymbol` 表示向量；本課程網站不支援，改用 `\mathbf`，意義相同。
- 本講義補上：硬查表與軟查表的數字對照與圖、三種相似度的比較表與圖、餘弦相似度、代名詞的例子、三題小練習與解答。
- 取捨：來源教授祕笈的矩陣形式 $\operatorname{Softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$ 與它的數字例，移到 KP05 和自注意力一起講；把三個投影合成一次矩陣乘法的實作技巧不寫。

- The source's "additive similarity" $\mathbf{w}_q^\top\mathbf{q}+\mathbf{w}_k^\top\mathbf{k}_i$ does not depend on the query after softmax (the numerical example of Section 4); this note keeps the source's formula and points this out. Additive attention in practice is the Bahdanau form of KP03, which applies $\tanh$ after the addition.
- The source writes vectors with bold italic `\boldsymbol`; the course website does not support it, so this note uses `\mathbf` with the same meaning.
- This note adds: the numerical comparison and figure of hard versus soft lookup, the comparison table and figure of three similarity measures, the cosine similarity, the pronoun example, and three practice questions with solutions.
- Choices: the matrix form $\operatorname{Softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$ of the professor's notes and its numerical example are moved to KP05, together with self-attention; the implementation trick of merging the three projections into one matrix multiplication is left out.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 查詢 | query |
| 鍵 | key |
| 值 | value |
| 硬查表 | hard lookup |
| 軟查表 | soft lookup |
| 解耦 | decoupling |
| 相似度 | similarity |
| 內積 | inner product |
| 縮放內積 | scaled inner product |
| 廣義內積 | general inner product |
| 加法相似度 | additive similarity |
| 餘弦相似度 | cosine similarity |
| 自注意力（KP05） | self-attention |
