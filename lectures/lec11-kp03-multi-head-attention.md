# 第11週 KP03：多頭注意力，好幾組查詢、鍵、值同時看不同的關係（Multi-head attention: several sets of queries, keys and values watch different relations at once）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第11週的第三個知識點（KP03），接在 KP02 之後。加上位置編碼後，自注意力分得出順序了。可是一組查詢、鍵、值只算出一組權重；一句話裡卻同時有好幾種關係。這一節看 Transformer 怎麼讓好幾組注意力平行工作，又不多花參數。

This is the third knowledge point (KP03) of Week 11 and follows KP02. With the positional encoding, self-attention can tell the order. But one set of queries, keys and values produces only one set of weights, while a sentence holds several kinds of relation at once. This note shows how the Transformer runs several attentions in parallel without spending more parameters.

先備知識是第10週 KP05 的自注意力矩陣形式 $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$，以及本週 KP01 的 $d_{\text{model}}=512$。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第9章改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec11-kp03-figures.py` 算出，可以自己重算。

The prerequisites are the matrix form of self-attention from Week 10 KP05, $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$, and the $d_{\text{model}}=512$ of this week's KP01. Every term and symbol is defined when it first appears. This note is rewritten from Chapter 9 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec11-kp03-figures.py`, so you can recompute it yourself.

## KP03 多頭注意力，好幾組查詢、鍵、值同時看不同的關係（Multi-head attention: several sets of queries, keys and values watch different relations at once）

### 學習目標（Learning objectives）

讀完之後，你能說明為什麼一個注意力頭會把不同的關係混在一起。你能寫出多頭注意力的三個步驟：各頭投影、各頭做縮放點積注意力、拼接後再投影，並標出每一步的矩陣形狀。你能算出 Transformer-Base 的多頭注意力和「一個全寬的頭」參數量與計算量相同。你也能說明頭數不是越多越好。

After reading, you can explain why one attention head mixes different relations together. You can write the three steps of multi-head attention, namely projecting for each head, running scaled dot-product attention in each head, and concatenating then projecting, and mark the matrix shape at each step. You can work out that the multi-head attention of Transformer-Base has the same parameters and computation as "one full-width head". You can also explain why more heads is not always better.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $h$ | 頭的數目 | number of heads |
| $d_{\text{model}}$ | 詞向量長度（本週 KP01） | word-vector length (this week's KP01) |
| $d_k,\ d_v$ | 每個頭的查詢與鍵的長度 $d_k$、值的長度 $d_v$，通常都是 $d_{\text{model}}/h$ | length of each head's queries and keys, and of its values, usually both $d_{\text{model}}/h$ |
| $\mathbf{X}\in\mathbb{R}^{n\times d_{\text{model}}}$ | 一層的輸入，每一列是一個詞 | input of a layer, one word per row |
| $\mathbf{W}_{Q,i},\mathbf{W}_{K,i}\in\mathbb{R}^{d_{\text{model}}\times d_k}$ | 第 $i$ 個頭的查詢與鍵矩陣 | query and key matrices of head $i$ |
| $\mathbf{W}_{V,i}\in\mathbb{R}^{d_{\text{model}}\times d_v}$ | 第 $i$ 個頭的值矩陣 | value matrix of head $i$ |
| $\operatorname{head}_i\in\mathbb{R}^{n\times d_v}$ | 第 $i$ 個頭的輸出 | output of head $i$ |
| $\mathbf{W}_O\in\mathbb{R}^{hd_v\times d_{\text{model}}}$ | 拼接之後的輸出矩陣 | output matrix applied after concatenation |
| $\operatorname{concat}(\cdot)$ | 把幾個矩陣左右接成一個 | joins several matrices side by side |

### 1. 一個頭只有一組權重，不同的關係會被攪在一起（One head has one set of weights, so different relations get stirred together）

看一句話：「那隻動物沒有過馬路，因為它很累。」讀到「它」時，模型至少要知道兩件事：「它」指的是「動物」，而說的是「它」「很累」。第10週學的自注意力，替「它」算出**一組**權重，再把各詞的值加權平均成一個向量。

Take the sentence "The animal didn't cross the road because it was tired." When reading "it", the model needs at least two things: "it" refers to "the animal", and what is said about "it" is "tired". The self-attention of Week 10 computes **one** set of weights for "it", then averages the values of the words into a single vector.

問題在第二步。如果一組權重要同時照顧兩種關係，它只能把份量分給兩邊，加權平均以後，兩種資訊就攪在同一個向量裡。來源的說法是：softmax 的加權平均會讓不同的關聯互相干擾。

The trouble lies in the second step. If one set of weights must serve two relations at once, it can only split its share between both sides, and after the weighted average the two pieces of information are stirred into the same vector. In the source's words, the weighted average of softmax makes different associations interfere with each other.

用數字看 `[補充：示意]`：查詢是「它」，三個鍵是「動物、它、很累」，每個詞的值就是它自己的代號。頭 1 專門找「它指誰」，分數 $3,0,0$；頭 2 專門找「說它什麼」，分數 $0,0,3$；兩頭都用 $d_k=2$。若只有一個頭要同時做兩件事，它的查詢就是兩個頭的查詢左右接起來（長度 $d_k=4$），點積等於兩頭分數相加，所以分數是 $3,0,3$；縮放也從 $\sqrt{2}$ 變成 $\sqrt{4}$。

See it with numbers `[Added: illustrative]`: the query is "it", the three keys are "animal, it, tired", and each word's value is simply its own label. Head 1 looks only for "who is it", with scores $3,0,0$; head 2 looks only for "what is said about it", with scores $0,0,3$; both use $d_k=2$. If a single head must do both jobs, its query is the two heads' queries joined side by side (length $d_k=4$), so its dot product is the sum of the two heads' scores, $3,0,3$; the scaling also changes from $\sqrt{2}$ to $\sqrt{4}$.

| 情況 | 動物 | 它 | 很累 |
|---|---|---|---|
| 頭 1：$\operatorname{softmax}([3,0,0]/\sqrt{2})$ | $0.8066$ | $0.0967$ | $0.0967$ |
| 頭 2：$\operatorname{softmax}([0,0,3]/\sqrt{2})$ | $0.0967$ | $0.0967$ | $0.8066$ |
| 一個頭：$\operatorname{softmax}([3,0,3]/\sqrt{4})$ | $0.4498$ | $0.1004$ | $0.4498$ |

| Case | animal | it | tired |
|---|---|---|---|
| Head 1: $\operatorname{softmax}([3,0,0]/\sqrt{2})$ | $0.8066$ | $0.0967$ | $0.0967$ |
| Head 2: $\operatorname{softmax}([0,0,3]/\sqrt{2})$ | $0.0967$ | $0.0967$ | $0.8066$ |
| One head: $\operatorname{softmax}([3,0,3]/\sqrt{4})$ | $0.4498$ | $0.1004$ | $0.4498$ |

![三張長條圖，查詢都是 it，三個鍵是 animal、it、tired。藍色頭 1 把 0.8066 給 animal；綠色頭 2 把 0.8066 給 tired；紅色的單一個頭把 0.4498 給 animal、0.4498 給 tired，中間的 it 只有 0.1004（Three bar charts for the query it over animal, it, tired: blue head 1 puts 0.8066 on animal, green head 2 puts 0.8066 on tired, and the red single head splits 0.4498 and 0.4498 between animal and tired, with 0.1004 on it）](../assets/figures/lec11-kp03-heads-vs-single.png)

**圖說 `lec11-kp03-heads-vs-single`**〔功能：一目了然、精準度〕：先看藍色與綠色：兩個頭各自把大部分份量放在自己要找的詞，一個看「動物」，一個看「很累」。再看紅色：一個頭要兼顧兩件事，只好大致對半分，每種關係只拿到約 $0.45$，而不是 $0.81$。多頭時，兩個頭的輸出並排放在不同的欄位；一個頭時，兩種資訊混在同一個加權平均裡。這裡的值是代號，還看得出兩份；真實的值向量會把兩種關係加在同一組座標上，就分不開了。

**Caption `lec11-kp03-heads-vs-single`** [Function: seeing it at a glance, precision]: first look at blue and green: each head puts most of its share on the word it is looking for, one on "animal" and one on "tired". Then look at red: one head serving both jobs has to split the share roughly in half, so each relation gets only about $0.45$ instead of $0.81$. With several heads, the two outputs sit side by side in different columns; with one head, both pieces of information sit in one weighted average. Here the values are labels, so the two parts can still be seen; real value vectors add both relations into the same coordinates, and then they cannot be separated.

常見錯誤：以為每個頭的分工是人先規定好的。上面的「指誰」「說什麼」只是為了說明而設定的；真實模型裡每個頭學到什麼由訓練決定，有的頭分工清楚，有的頭看起來沒有明顯意義。

Common mistake: thinking the role of each head is assigned by people beforehand. The "who" and "what" above were set up only for illustration; in a real model what each head learns is decided by training, and some heads have clear roles while others show no obvious meaning.

### 2. 三個步驟：各自投影、各自注意、拼起來再投影（Three steps: project separately, attend separately, concatenate and project）

多頭注意力讓 $h$ 個頭各自做一次第10週的自注意力。第一步，每個頭用自己的三個矩陣，把同一個輸入 $\mathbf{X}$ 投影成自己的查詢、鍵、值：

Multi-head attention lets each of $h$ heads run the self-attention of Week 10 once. In the first step, each head uses its own three matrices to project the same input $\mathbf{X}$ into its own queries, keys and values:

$$
\mathbf{Q}_i=\mathbf{X}\mathbf{W}_{Q,i},\qquad \mathbf{K}_i=\mathbf{X}\mathbf{W}_{K,i},\qquad \mathbf{V}_i=\mathbf{X}\mathbf{W}_{V,i},\qquad i=1,\dots,h.
$$

第二步，每個頭各自做縮放點積注意力，得到 $n\times d_v$ 的輸出：

In the second step, each head runs scaled dot-product attention on its own and gets an $n\times d_v$ output:

$$
\operatorname{head}_i=\operatorname{softmax}\!\left(\frac{\mathbf{Q}_i\mathbf{K}_i^\top}{\sqrt{d_k}}\right)\mathbf{V}_i.
$$

第三步，把 $h$ 個頭的輸出左右拼成一個 $n\times hd_v$ 的矩陣，再乘上輸出矩陣 $\mathbf{W}_O$，讓各頭的發現互相混合，回到 $n\times d_{\text{model}}$：

In the third step, the $h$ head outputs are joined side by side into an $n\times hd_v$ matrix, which is multiplied by the output matrix $\mathbf{W}_O$ so that the heads' findings are mixed, returning to $n\times d_{\text{model}}$:

$$
\operatorname{MHA}(\mathbf{X})=\operatorname{concat}(\operatorname{head}_1,\dots,\operatorname{head}_h)\,\mathbf{W}_O.
$$

來源的拼接例子：$d_{\text{model}}=4$、$h=2$、兩個詞，頭 1 輸出 $\begin{bmatrix}1&0\\0.5&0.5\end{bmatrix}$，頭 2 輸出 $\begin{bmatrix}0&2\\1&1\end{bmatrix}$，拼接後是 $\begin{bmatrix}1&0&0&2\\0.5&0.5&1&1\end{bmatrix}$。每一列仍是一個詞，前兩欄來自頭 1，後兩欄來自頭 2。

The source's concatenation example: $d_{\text{model}}=4$, $h=2$, two words; head 1 outputs $\begin{bmatrix}1&0\\0.5&0.5\end{bmatrix}$ and head 2 outputs $\begin{bmatrix}0&2\\1&1\end{bmatrix}$, so the concatenation is $\begin{bmatrix}1&0&0&2\\0.5&0.5&1&1\end{bmatrix}$. Each row is still one word; the first two columns come from head 1 and the last two from head 2.

![最左邊灰色方塊是輸入 X，形狀 n 乘 512。它分出 8 條箭頭到 8 個藍色方塊（畫出頭 1、2、3、省略號與頭 8），每個寫著在 n 乘 64 上做注意力，上方標 W_Q,i、W_K,i、W_V,i 各是 512 乘 64。8 個頭再匯進黃色的 concat，形狀 n 乘 512，接著是紫色的 W_O（512 乘 512），最後是綠色的輸出 n 乘 512。下方寫 8 乘 64 等於 512，總寬度和一個全寬的頭相同（A grey box X of shape n by 512 fans out to eight blue heads, each attending on n by 64 with W_Q,i, W_K,i, W_V,i of 512 by 64; the heads merge into a yellow concat of n by 512, then a purple W_O of 512 by 512, then a green output of n by 512; a note says 8 times 64 equals 512, the same width as one full head）](../assets/figures/lec11-kp03-mha-shapes.png)

**圖說 `lec11-kp03-mha-shapes`**〔功能：一目了然、精準度〕：先看左邊：同一個 $n\times512$ 的輸入，分給 $8$ 個頭。再看中間：每個頭只在 $64$ 維的小空間裡做注意力，矩陣是 $512\times64$。最後看右邊：$8$ 份 $n\times64$ 拼回 $n\times512$，經 $\mathbf{W}_O$ 混合後，形狀和輸入一樣，所以能接 KP01 的 Add & Norm。

**Caption `lec11-kp03-mha-shapes`** [Function: seeing it at a glance, precision]: first look at the left: the same $n\times512$ input is handed to $8$ heads. Then look at the middle: each head attends only within a small $64$-dimensional space, with $512\times64$ matrices. Finally look at the right: the $8$ pieces of $n\times64$ join back into $n\times512$ and, after mixing through $\mathbf{W}_O$, have the same shape as the input, so KP01's Add & Norm can follow.

常見錯誤：以為拼接以後就結束了，不需要 $\mathbf{W}_O$。拼接只是把各頭的結果並排；$\mathbf{W}_O$ 在這一個子層裡就把各頭的發現混合起來，並把它們對應回 $d_{\text{model}}$ 的空間，才能和輸入做殘差相加（本週 KP04）。

Common mistake: thinking the job ends with concatenation and $\mathbf{W}_O$ is unnecessary. Concatenation only places the heads' results side by side; $\mathbf{W}_O$ mixes the heads' findings within this sublayer and maps them back into the $d_{\text{model}}$ space, so they can be added to the input in the residual step (this week's KP04).

### 3. 切成 8 份，總成本不變（Split into 8 pieces, the total cost stays the same）

多了 $8$ 個頭，參數會變成 $8$ 倍嗎？不會。關鍵是每個頭的寬度縮成 $d_k=d_v=d_{\text{model}}/h=512/8=64$。

With $8$ heads, do the parameters become $8$ times as many? No. The key is that each head's width shrinks to $d_k=d_v=d_{\text{model}}/h=512/8=64$.

算一算 `[補充]`，用的是 KP01 的 Transformer-Base（$d_{\text{model}}=512$、$h=8$），而且不計偏差（bias，線性層在矩陣乘法之後再加上的常數向量）。$8$ 個頭各有三個 $512\times64$ 的矩陣，共 $3\times8\times512\times64=786432$ 個，再加上 $\mathbf{W}_O$ 的 $512\times512=262144$ 個，合計 $1048576$ 個。一個全寬的頭用三個 $512\times512$ 的矩陣加上同樣的 $\mathbf{W}_O$，也是 $4\times512\times512=1048576$ 個。

Count them `[Added]` for KP01's Transformer-Base ($d_{\text{model}}=512$, $h=8$), ignoring biases (a bias is the constant vector a linear layer adds after the matrix product). The $8$ heads each have three $512\times64$ matrices, $3\times8\times512\times64=786432$ in total, plus $512\times512=262144$ for $\mathbf{W}_O$, giving $1048576$. One full-width head uses three $512\times512$ matrices plus the same $\mathbf{W}_O$, also $4\times512\times512=1048576$.

計算量也一樣。一句 $n=10$ 個詞的話，算分數 $\mathbf{Q}_i\mathbf{K}_i^\top$ 時，$8$ 個頭各要 $10\times10\times64$ 次乘法，共 $51200$ 次；一個全寬的頭要 $10\times10\times512=51200$ 次。投影與「權重乘以值」這兩步的乘法次數也相同。差別只在多頭會產生 $8$ 張 $10\times10$ 的權重表，共 $800$ 個數，而不是一張 $100$ 個數，softmax 要算的指數也從 $100$ 個變成 $800$ 個，這部分很小。

The computation is the same too. For a sentence of $n=10$ words, computing the scores $\mathbf{Q}_i\mathbf{K}_i^\top$ takes $10\times10\times64$ multiplications per head, $51200$ for all $8$ heads; one full-width head takes $10\times10\times512=51200$. The projections and the "weights times values" step also take the same number of multiplications. The only difference is that multiple heads produce $8$ weight tables of $10\times10$, $800$ numbers, instead of one table of $100$, so softmax computes $800$ exponentials instead of $100$, a small part of the work.

實作上，$8$ 個頭不是用迴圈一個一個算。程式把 $\mathbf{X}\mathbf{W}_Q$ 一次算成 $n\times512$，再把這 $512$ 欄重新排成 $8$ 份 $64$ 欄，所有頭在同一個矩陣運算裡平行完成。

In practice the $8$ heads are not computed one by one in a loop. The code computes $\mathbf{X}\mathbf{W}_Q$ once as $n\times512$, rearranges those $512$ columns into $8$ groups of $64$, and all heads finish in parallel inside one matrix operation.

常見錯誤：以為多頭能把參數減少到 $1/h$。每個頭變窄了，但頭變多了，兩者相乘剛好抵銷，總參數和一個全寬的頭一樣。

Common mistake: thinking multiple heads cut the parameters to $1/h$. Each head becomes narrower, but there are more heads; the two effects cancel exactly, and the total equals that of one full-width head.

### 4. 頭數不是越多越好（More heads is not always better）

既然總成本不變，把頭數加到 $512$ 個，每頭 $d_k=1$，不是能看 $512$ 種關係嗎？問題是每個頭的空間太小了。$d_k=1$ 時，查詢和鍵都只剩一個數 $q$ 和 $k$，分數就是 $qk$。$q$ 是正數時，永遠最看重 $k$ 最大的那個詞；$q$ 是負數時，永遠最看重 $k$ 最小的那個詞。這樣的頭只能表達「看最極端的那個詞」，很難表達有意義的關係。原論文在總計算量不變下比較頭數（英翻德開發集，BLEU 分數越高越好）：$h=1$ 是 $24.9$，$h=8$ 與 $h=16$ 都是 $25.8$，$h=32$ 降到 $25.4$ `[補充]`。單頭明顯較差，頭太多也略為下降。

Since the total cost does not change, why not raise the number of heads to $512$, with $d_k=1$ each, and watch $512$ relations? The trouble is that each head's space becomes too small. With $d_k=1$, a query and a key are single numbers $q$ and $k$, and the score is $qk$. When $q$ is positive, the head always prefers the word with the largest $k$; when $q$ is negative, the word with the smallest $k$. Such a head can only say "look at the most extreme word", and meaningful relations are hard to express. The original paper compared head counts at constant total computation (English-to-German development set, higher BLEU is better): $h=1$ scored $24.9$, $h=8$ and $h=16$ both $25.8$, and $h=32$ dropped to $25.4$ `[Added]`. One head is clearly worse, and too many heads also drop slightly.

所以頭數是一個要調的設定。Transformer-Base 選 $h=8$、$d_k=64$；更大的模型通常同時加大 $d_{\text{model}}$ 和 $h$，讓每個頭維持幾十維的寬度。

So the number of heads is a setting to tune. Transformer-Base chooses $h=8$ and $d_k=64$; larger models usually raise both $d_{\text{model}}$ and $h$, keeping each head a few dozen dimensions wide.

常見錯誤：以為頭越多，模型一定越好。頭變多、每頭變窄，到了某個程度以後，每個頭的表達力不足，效果反而下降。

Common mistake: thinking more heads always means a better model. More heads means narrower heads, and beyond some point each head lacks expressive power and results get worse.

### 本單元重點整理（KP03 summary）

**一句話重點**：多頭注意力把 $d_{\text{model}}$ 切成 $h$ 份，每個頭用自己的 $\mathbf{W}_{Q,i},\mathbf{W}_{K,i},\mathbf{W}_{V,i}$ 在 $d_k=d_{\text{model}}/h$ 維裡做一次縮放點積注意力，再把各頭輸出拼接後乘上 $\mathbf{W}_O$；不同的關係可以放在不同的頭，總參數與計算量和一個全寬的頭相同。

**One-sentence summary**: multi-head attention splits $d_{\text{model}}$ into $h$ pieces; each head uses its own $\mathbf{W}_{Q,i},\mathbf{W}_{K,i},\mathbf{W}_{V,i}$ to run scaled dot-product attention in $d_k=d_{\text{model}}/h$ dimensions, and the head outputs are concatenated and multiplied by $\mathbf{W}_O$; different relations can live in different heads, with the same total parameters and computation as one full-width head.

**白話說明**：與其請一位讀者同時注意文法、指代和修飾，不如請八位讀者各看一個面向，最後把八份筆記並排交給一位編輯整合。

**Plain explanation**: instead of asking one reader to watch grammar, references and modifiers all at once, ask eight readers to watch one aspect each, then hand the eight sets of notes side by side to an editor who combines them.

**正式定義**：$\operatorname{head}_i=\operatorname{softmax}(\mathbf{X}\mathbf{W}_{Q,i}(\mathbf{X}\mathbf{W}_{K,i})^\top/\sqrt{d_k})\,\mathbf{X}\mathbf{W}_{V,i}$，$\operatorname{MHA}(\mathbf{X})=\operatorname{concat}(\operatorname{head}_1,\dots,\operatorname{head}_h)\mathbf{W}_O$，$d_k=d_v=d_{\text{model}}/h$。

**Formal definition**: $\operatorname{head}_i=\operatorname{softmax}(\mathbf{X}\mathbf{W}_{Q,i}(\mathbf{X}\mathbf{W}_{K,i})^\top/\sqrt{d_k})\,\mathbf{X}\mathbf{W}_{V,i}$, $\operatorname{MHA}(\mathbf{X})=\operatorname{concat}(\operatorname{head}_1,\dots,\operatorname{head}_h)\mathbf{W}_O$, $d_k=d_v=d_{\text{model}}/h$.

**例子**：示意例子中，兩個頭各把 $0.8066$ 放在「動物」與「很累」，一個頭只能各給 $0.4498$。Transformer-Base：$8$ 個頭各 $64$ 維，參數 $1048576$ 個、$n=10$ 時分數乘法 $51200$ 次，都和一個全寬的頭相同。

**Example**: in the illustrative example, the two heads put $0.8066$ on "animal" and on "tired" respectively, while one head can give each only $0.4498$. Transformer-Base: $8$ heads of $64$ dimensions, $1048576$ parameters and, for $n=10$, $51200$ score multiplications, both equal to one full-width head.

**常見迷思**：

- 「多頭把參數減少到 $1/h$。」錯，總參數不變。
- 「每個頭的分工是人規定的。」錯，分工由訓練決定，也不一定看得懂。
- 「頭越多越好。」錯，每頭太窄時表達力不足。

**Common misconceptions**:

- "Multiple heads cut the parameters to $1/h$." Wrong: the total stays the same.
- "Each head's role is assigned by people." Wrong: the roles come from training and are not always interpretable.
- "More heads are always better." Wrong: heads that are too narrow lack expressive power.

**圖**：本單元有兩張圖。

- `lec11-kp03-heads-vs-single`：查詢「它」在兩個頭與一個頭之下的權重。
- `lec11-kp03-mha-shapes`：多頭注意力每一步的矩陣形狀。

**Figures** — this unit has two figures.

- `lec11-kp03-heads-vs-single`: the weights of the query "it" under two heads and under one head.
- `lec11-kp03-mha-shapes`: the matrix shape at each step of multi-head attention.

### 小練習（Practice）

1. $d_{\text{model}}=768$、$h=12$。每個頭的 $d_k$ 是多少？$\mathbf{W}_{Q,i}$ 的形狀是什麼？拼接後是幾欄？
2. 承第 1 題，不計偏差，多頭注意力共有幾個參數？和一個全寬的頭比較。
3. 兩個頭的輸出分別是 $\begin{bmatrix}2&1\\0&3\end{bmatrix}$ 和 $\begin{bmatrix}1&1\\4&0\end{bmatrix}$。寫出拼接的結果，並說明每一列代表什麼。

1. $d_{\text{model}}=768$ and $h=12$. What is $d_k$ per head? What is the shape of $\mathbf{W}_{Q,i}$? How many columns does the concatenation have?
2. Continuing question 1 and ignoring biases, how many parameters does multi-head attention have? Compare with one full-width head.
3. Two heads output $\begin{bmatrix}2&1\\0&3\end{bmatrix}$ and $\begin{bmatrix}1&1\\4&0\end{bmatrix}$. Write the concatenation and say what each row represents.

**第 1 題解答** `[補充]`：$d_k=768/12=64$；$\mathbf{W}_{Q,i}$ 是 $768\times64$；拼接後是 $12\times64=768$ 欄。

**Solution to question 1** `[Added]`: $d_k=768/12=64$; $\mathbf{W}_{Q,i}$ is $768\times64$; the concatenation has $12\times64=768$ columns.

**第 2 題解答** `[補充]`：$3\times12\times768\times64+768\times768=1769472+589824=2359296$；一個全寬的頭是 $4\times768\times768=2359296$，相同。

**Solution to question 2** `[Added]`: $3\times12\times768\times64+768\times768=1769472+589824=2359296$; one full-width head has $4\times768\times768=2359296$, the same.

**第 3 題解答** `[補充]`：$\begin{bmatrix}2&1&1&1\\0&3&4&0\end{bmatrix}$；每一列仍是一個詞，前兩欄是頭 1 對這個詞的結果，後兩欄是頭 2 的結果。

**Solution to question 3** `[Added]`: $\begin{bmatrix}2&1&1&1\\0&3&4&0\end{bmatrix}$; each row is still one word, with the first two columns from head 1 and the last two from head 2.

### 小測驗（Quiz，來源原題）

來源的兩題自我檢測保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。題中的 $\mathbf{Z}_i$ 就是本講義的 $\operatorname{head}_i$，$\mathbf{Z}_t$ 就是 $\operatorname{MHA}(\mathbf{X})$。

The source's two self-check questions are kept with all their options; the source page does not mark the answers, so the explanations below are judged from the source text. In the questions, $\mathbf{Z}_i$ is this note's $\operatorname{head}_i$ and $\mathbf{Z}_t$ is $\operatorname{MHA}(\mathbf{X})$.

1. 多頭注意力機制（Multihead Attention）的主要優勢是什麼？
   - (A) 減少參數量至原來的 $1/h$
   - (B) 允許模型在不同的子空間中同時關注不同位置與不同類型的特徵關係
   - (C) 強制每個頭輸出完全相同的結果
   - (D) 將注意力權重強制限制為 $0$ 或 $1$
2. 各個頭的注意力輸出 $\mathbf{Z}_1,\dots,\mathbf{Z}_h$ 計算完成後，如何整合成最終輸出 $\mathbf{Z}_t$？
   - (A) 取所有頭輸出的最大值
   - (B) 將各頭輸出拼接（concat）後通過線性輸出矩陣 $\mathbf{W}_O$ 投影
   - (C) 隨機丟棄其中 $h-1$ 個頭
   - (D) 將各頭輸出元素直接相除

1. What is the main advantage of multi-head attention?
   - (A) It reduces the parameters to $1/h$ of the original
   - (B) It lets the model attend at once to different positions and different kinds of feature relations in different subspaces
   - (C) It forces every head to output exactly the same result
   - (D) It forces the attention weights to be $0$ or $1$
2. After the head outputs $\mathbf{Z}_1,\dots,\mathbf{Z}_h$ are computed, how are they combined into the final output $\mathbf{Z}_t$?
   - (A) Take the maximum of all head outputs
   - (B) Concatenate the head outputs and project them with the linear output matrix $\mathbf{W}_O$
   - (C) Randomly drop $h-1$ of the heads
   - (D) Divide the head outputs element by element

**解析** `[補充]`：第 1 題選 (B)：第2、3節把 $d_{\text{model}}$ 切成 $h$ 個子空間，第1節的例子顯示不同的頭看不同的關係；(A) 錯，第3節算出總參數不變。第 2 題選 (B)：就是第2節的第三步。

**Explanations** `[Added]`: question 1 is B: Sections 2 and 3 split $d_{\text{model}}$ into $h$ subspaces, and the example of Section 1 shows different heads watching different relations; A is wrong, as Section 3 shows the total parameters are unchanged. Question 2 is B: it is the third step of Section 2.

### 留給下一單元的問題（The question for the next unit）

多頭注意力的輸出形狀和輸入一樣，可以疊很多層。可是疊得深，第3週的梯度消失就會回來；每層輸出的數值大小也可能越來越偏。KP01 圖上每個子層後面那個「Add & Norm」到底做了什麼？子層之後的前饋網路又負責什麼？下一單元（本週 KP04）回答這些問題。

Multi-head attention keeps the input shape, so many layers can be stacked. But with deep stacks, the vanishing gradient of Week 3 comes back, and the scale of each layer's output may drift further and further. What exactly does the "Add & Norm" after every sublayer in KP01's figure do? And what is the feed-forward network after it for? The next unit (this week's KP04) answers these questions.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第9章「注意力機制與 Transformer 核心原理」§3.2.2 與教授祕笈「多頭注意力與子空間投影」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §3.2.2 of Chapter 9, "Core principles of attention mechanisms and the Transformer", of CCC-LNS IDL_115, and from the professor's notes "Multi-head attention and subspace projection" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源正文把詞向量排成行，寫 $\mathbf{Q}_i=\mathbf{W}_{Q,i}^\top\mathbf{X}$、$\mathbf{Z}_i=\mathbf{V}_i\operatorname{softmax}(\cdots)$；本講義沿用「列為詞」的記號，寫成 $\mathbf{X}\mathbf{W}_{Q,i}$ 與 $\operatorname{softmax}(\cdots)\mathbf{V}_i$，意義相同。
- 來源正文把頭的維度記為 $p$、輸出矩陣寫成 $\mathbf{W}_O\in\mathbb{R}^{(hr)\times d}$；本講義依祕笈統一寫成 $d_k$、$d_v$ 與 $hd_v\times d_{\text{model}}$。來源用粗斜體 `\boldsymbol`，本課程網站不支援，改用 `\mathbf`。
- 來源祕笈說多頭與「單頭全維度」的總參數量與總計算量一致；本講義在第3節不計偏差算出具體數字，並補充多頭會多存 $h$ 張權重表。
- 來源正文說單頭「容易將所有注意力權重集中在單一語義關聯上」，祕笈則說單頭會把多種關聯「攪拌」在一起；兩種說法不同，本講義採祕笈的說法，並用第1節的數字示範。
- 來源正文把多頭和最大平均差異（MMD）類比；這需要本週沒有的先備，本講義不寫。
- 本講義補上：「它」的兩頭與一頭數字示意與圖、參數與乘法次數的計算、頭太多的問題、三題小練習與解答。
- 取捨：MQA、GQA 與 KV 快取、PyTorch 的 `contiguous()` 細節、量子力學的類比，本講義不寫。

- The source's main text arranges word vectors as columns, writing $\mathbf{Q}_i=\mathbf{W}_{Q,i}^\top\mathbf{X}$ and $\mathbf{Z}_i=\mathbf{V}_i\operatorname{softmax}(\cdots)$; this note keeps the "one word per row" notation, $\mathbf{X}\mathbf{W}_{Q,i}$ and $\operatorname{softmax}(\cdots)\mathbf{V}_i$, with the same meaning.
- The source's main text calls the head dimension $p$ and writes the output matrix as $\mathbf{W}_O\in\mathbb{R}^{(hr)\times d}$; following the professor's notes, this note writes $d_k$, $d_v$ and $hd_v\times d_{\text{model}}$ throughout. The source uses bold italic `\boldsymbol`, which the course website does not support, so this note uses `\mathbf`.
- The professor's notes say multi-head attention has the same total parameters and computation as "one full-dimension head"; Section 3 works out the numbers without biases and adds that multiple heads store $h$ weight tables.
- The source's main text says a single head "tends to concentrate all its weight on one semantic relation", while the professor's notes say a single head "stirs" several relations together; the two statements differ, and this note follows the notes, illustrated with the numbers of Section 1.
- The source's main text compares multi-head attention with maximum mean discrepancy (MMD); that needs background this week does not have, so this note leaves it out.
- This note adds: the two-heads-versus-one numerical illustration of "it" with its figure, the counts of parameters and multiplications, the problem of too many heads, and three practice questions with solutions.
- Choices: MQA, GQA and the KV cache, the PyTorch `contiguous()` detail and the quantum-mechanics analogy are not covered.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 多頭注意力 | multi-head attention |
| 注意力頭 | attention head |
| 縮放點積注意力（第10週） | scaled dot-product attention |
| 投影 | projection |
| 拼接 | concatenation |
| 輸出矩陣 | output matrix |
| 子空間 | subspace |
| 全寬的頭 | full-width head |
| 偏差 | bias |
| BLEU 分數（翻譯品質指標） | BLEU score |
| 最大平均差異 | maximum mean discrepancy (MMD) |
| 多查詢注意力 | multi-query attention (MQA) |
| 分組查詢注意力 | grouped-query attention (GQA) |
| 鍵值快取 | key-value cache (KV cache) |
| PyTorch（深度學習程式庫） | PyTorch |
