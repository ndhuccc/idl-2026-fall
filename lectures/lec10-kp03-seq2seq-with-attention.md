# 第10週 KP03：帶注意力的 Seq2Seq，每寫一個詞就重新看一次整句（Seq2Seq with attention: look at the whole sentence again for every word you write）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第10週的第三個知識點（KP03），接在 KP02 之後。KP02 的 Seq2Seq 把整句來源壓成一個固定的上下文向量 $\mathbf{c}$，句子一長，前面的詞就被蓋掉。這一節把 KP01 的注意力放進解碼器：每寫一個詞，都依相關程度重新加權編碼器的所有隱藏狀態。

This is the third knowledge point (KP03) of Week 10 and follows KP02. The Seq2Seq of KP02 compresses the whole source sentence into one fixed context vector $\mathbf{c}$, and in a long sentence the early words are overwritten. This note puts the attention of KP01 into the decoder: for every word it writes, the decoder reweights all hidden states of the encoder by relevance.

先備知識是 KP01 的分數、softmax 與加權平均，以及 KP02 的編碼器、解碼器與 $\mathbf{l}_i$。本講義依課程網頁教材 CCC-LNS 第9章改寫，文中「來源」就是指它；本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec10-kp03-figures.py` 算出，可以自己重算。

The prerequisites are the scores, softmax and weighted average of KP01, and the encoder, decoder and $\mathbf{l}_i$ of KP02. This note is rewritten from Chapter 9 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec10-kp03-figures.py`, so you can recompute it yourself.

## KP03 帶注意力的 Seq2Seq，每寫一個詞就重新看一次整句（Seq2Seq with attention: look at the whole sentence again for every word you write）

### 學習目標（Learning objectives）

讀完之後，你能說明帶注意力的 Seq2Seq 和 KP02 的差別：每個輸出詞都有自己的上下文向量 $\mathbf{c}_i$。你能寫出算 $\mathbf{c}_i$ 的三個步驟：對齊分數、softmax、加權和，並指出查詢是哪一個向量。你能讀懂對齊矩陣，說出翻譯每個詞時模型在看哪裡。你也能比較加性、點積、廣義與縮放點積四種分數函數，並說出注意力要付出的計算代價。

After reading, you can explain how Seq2Seq with attention differs from KP02: every output word has its own context vector $\mathbf{c}_i$. You can write the three steps that compute $\mathbf{c}_i$: alignment scores, softmax and weighted sum, and point out which vector is the query. You can read an alignment matrix and say where the model is looking while translating each word. You can also compare four score functions, additive, dot product, general and scaled dot product, and state the computational price of attention.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{h}_j$ | 編碼器第 $j$ 步的隱藏狀態（$j=1,\dots,n$，$n=T_x$） | hidden state of the encoder at step $j$ ($j=1,\dots,n$, $n=T_x$) |
| $\mathbf{l}_{i-1}$ | 解碼器寫第 $i$ 個詞之前的狀態，當作查詢 | decoder state before writing the $i$-th word, used as the query |
| $s_{ij}$ | 寫第 $i$ 個詞時，對第 $j$ 個來源詞的對齊分數 | alignment score of source word $j$ when writing word $i$ |
| $a_{ij}$ | 對應的注意力權重，$\sum_j a_{ij}=1$ | the corresponding attention weight, $\sum_j a_{ij}=1$ |
| $\mathbf{c}_i$ | 寫第 $i$ 個詞專用的上下文向量 | context vector used only for word $i$ |
| $T_x,\ T_y$ | 來源句與目標句的長度 | lengths of the source and target sentences |
| $\mathbf{q},\ \mathbf{k}$ | 第4節分數函數裡的查詢與被比較的向量 | the query and the compared vector in the score functions of Section 4 |
| $d_k$ | $\mathbf{q}$、$\mathbf{k}$ 的維度 | dimension of $\mathbf{q}$ and $\mathbf{k}$ |
| $\mathbf{W}_a,\ \mathbf{U}_a,\ \mathbf{v}_a$ | 分數函數的可學習參數（廣義形式與加性形式各有自己的 $\mathbf{W}_a$） | learnable parameters of the score functions (the general and the additive forms each have their own $\mathbf{W}_a$) |

### 1. 每個輸出詞都有自己的上下文向量（Every output word gets its own context vector）

KP02 的問題是「整句只剩一個 $\mathbf{c}$，每一步都用它」。Bahdanau 等人在 2014 年提出的改法很直接：不要事先壓成一份摘要；改成每寫一個詞，就重新算一個專用的上下文向量 $\mathbf{c}_i$。

The problem of KP02 is "the whole sentence leaves one $\mathbf{c}$, used at every step". The fix proposed by Bahdanau and colleagues in 2014 is direct: do not compress everything into one summary beforehand; instead, compute a dedicated context vector $\mathbf{c}_i$ for every word you write.

$\mathbf{c}_i$ 是編碼器**所有**隱藏狀態的加權平均，正是 KP01 的那個式子：

$\mathbf{c}_i$ is a weighted average of **all** hidden states of the encoder, exactly the formula of KP01:

$$
\mathbf{c}_i=\sum_{j=1}^{n}a_{ij}\,\mathbf{h}_j,\qquad a_{ij}\ge 0,\quad \sum_{j=1}^{n}a_{ij}=1.
$$

這樣一來，句首的詞不必穿過一長串更新才能被看到；解碼器寫哪個詞，就直接取用哪幾個來源詞的狀態。權重 $a_{ij}$ 表示寫第 $i$ 個詞時，模型分給第 $j$ 個來源詞多少注意力。

Now the first word of the sentence no longer has to pass through a long chain of updates to be seen; whichever word the decoder writes, it takes the states of the relevant source words directly. The weight $a_{ij}$ says how much attention the model gives to source word $j$ while writing word $i$.

來源的比喻：做翻譯時，你不會先把整篇原文背下來再默寫；你是翻到哪裡，眼睛就回頭掃一下原文的相關字。這個比喻的侷限是：人的眼睛一次只盯一個地方，而注意力同時給每個來源詞一點份量，只是多寡不同。

The source's analogy: when translating, you do not memorize the whole original and then write from memory; as you go, your eyes glance back at the relevant words of the original. The limit of the analogy: human eyes fix on one place at a time, while attention gives every source word some share at once, just in different amounts.

常見錯誤：以為帶注意力以後就不需要編碼器的 RNN 了。這一版的 Seq2Seq 仍然用 RNN 產生 $\mathbf{h}_j$，只是解碼器不再只拿最後一個；要到第11週的 Transformer，才把 RNN 整個拿掉。

Common error: thinking that the encoder's RNN is no longer needed once attention is added. This version of Seq2Seq still uses an RNN to produce the $\mathbf{h}_j$; the decoder just no longer takes only the last one. Only the Transformer of Week 11 removes the RNN altogether.

### 2. 三個步驟：對齊分數、softmax、加權和（Three steps: alignment score, softmax, weighted sum）

$\mathbf{c}_i$ 怎麼算？套用 KP01 的做法，只要決定「查詢是誰」。寫第 $i$ 個詞之前，解碼器最新的狀態是 $\mathbf{l}_{i-1}$，它代表「我已經寫到哪裡、接下來要寫什麼」，所以拿它當查詢。

How is $\mathbf{c}_i$ computed? Apply the method of KP01; you only need to decide "who is the query". Before writing the $i$-th word, the latest decoder state is $\mathbf{l}_{i-1}$; it stands for "where I am in the translation and what comes next", so it serves as the query.

第一步是**對齊分數**（alignment score）：比一比查詢和每個來源詞的狀態有多相關。

Step one is the **alignment score**: measure how related the query is to the state of each source word.

$$
s_{ij}=\operatorname{similarity}(\mathbf{l}_{i-1},\mathbf{h}_j),\qquad j=1,\dots,n.
$$

第二步是 softmax：把同一個 $i$ 的分數變成權重。

Step two is softmax: turn the scores of the same $i$ into weights.

$$
a_{ij}=\frac{\exp(s_{ij})}{\sum_{k=1}^{n}\exp(s_{ik})}.
$$

第三步是加權和：算出 $\mathbf{c}_i=\sum_j a_{ij}\mathbf{h}_j$，交給解碼器。和 KP02 比，解碼器式子只改了一個地方，$\mathbf{c}$ 換成 $\mathbf{c}_i$：

Step three is the weighted sum: compute $\mathbf{c}_i=\sum_j a_{ij}\mathbf{h}_j$ and hand it to the decoder. Compared with KP02, the decoder equation changes in one place only, $\mathbf{c}$ becomes $\mathbf{c}_i$:

$$
\mathbb{P}(\mathbf{y}_i\mid\mathbf{y}_1,\dots,\mathbf{y}_{i-1})=g(\mathbf{y}_{i-1},\mathbf{l}_i,\mathbf{c}_i).
$$

這裡的 $\mathbf{l}_i$ 是解碼器拿到 $\mathbf{c}_i$ 之後更新出的新狀態：$\mathbf{l}_i$ 由 $\mathbf{l}_{i-1}$、$\mathbf{y}_{i-1}$ 與 $\mathbf{c}_i$ 算出 `[補充]`。所以順序是：用舊狀態 $\mathbf{l}_{i-1}$ 去查，得到 $\mathbf{c}_i$，再更新成 $\mathbf{l}_i$，寫出第 $i$ 個詞。

Here $\mathbf{l}_i$ is the new decoder state updated after receiving $\mathbf{c}_i$: $\mathbf{l}_i$ is computed from $\mathbf{l}_{i-1}$, $\mathbf{y}_{i-1}$ and $\mathbf{c}_i$ `[Added]`. So the order is: query with the old state $\mathbf{l}_{i-1}$, get $\mathbf{c}_i$, update to $\mathbf{l}_i$, and write the $i$-th word.

來源的小例子：兩個來源詞，$\mathbf{h}_1=[1,0]^\top$、$\mathbf{h}_2=[0,1]^\top$，某一步算出的對齊分數是 $s_{i1}=1.0$、$s_{i2}=2.0$。取指數得 $2.718$、$7.389$，總和 $10.107$，所以

The source's small example: two source words, $\mathbf{h}_1=[1,0]^\top$ and $\mathbf{h}_2=[0,1]^\top$, and at some step the alignment scores are $s_{i1}=1.0$ and $s_{i2}=2.0$. The exponentials are $2.718$ and $7.389$, with total $10.107$, so

$$
a_{i1}=\frac{2.718}{10.107}=0.2689,\qquad a_{i2}=\frac{7.389}{10.107}=0.7311,\qquad \mathbf{c}_i=0.2689\,\mathbf{h}_1+0.7311\,\mathbf{h}_2=\begin{bmatrix}0.2689\\0.7311\end{bmatrix}.
$$

第二個詞的分數只多 $1$，卻分到大約四分之三的注意力：softmax 會把分數的差距放大成份量的差距。

The second word's score is higher by only $1$, yet it receives about three quarters of the attention: softmax turns a gap in scores into a larger gap in shares.

![左邊三個藍色方塊是來源詞 Je、suis、étudiant 的隱藏狀態 h。右上紫色方塊是查詢 l_3，有三條藍色曲線連到三個 h，線的粗細代表權重：連到 étudiant 的最粗，標 a=0.7870，連到 Je 和 suis 的細線各標 a=0.1065。右側黃色方塊寫 c_4 等於 a_4j 乘 h_j 的總和，上方標 weighted sum of the three h，往下箭頭指向 write student（The query l_3 is linked to the three source states by lines whose widths are the weights 0.1065, 0.1065 and 0.7870; their weighted sum c_4 is used to write "student"）](../assets/figures/lec10-kp03-one-step.png)

**圖說 `lec10-kp03-one-step`**〔功能：一目了然〕：先看右上的紫色查詢 $\mathbf{l}_3$，它是解碼器寫第 $4$ 個詞之前的狀態。再看三條藍線：線越粗，權重越大，連到 étudiant 的那條最粗（$0.7870$），另外兩條各 $0.1065$。最後看右側的黃色方塊：三個 $\mathbf{h}$ 依這些權重相加得到 $\mathbf{c}_4$，解碼器用它寫出 `student`。

**Caption `lec10-kp03-one-step`** [Function: seeing it at a glance]: First look at the purple query $\mathbf{l}_3$ at the top right; it is the decoder state before writing the $4$-th word. Then look at the three blue lines: the thicker the line, the larger the weight, and the line to étudiant is the thickest ($0.7870$) while the other two are $0.1065$ each. Finally look at the yellow box on the right: the three $\mathbf{h}$ are added with these weights into $\mathbf{c}_4$, which the decoder uses to write "student".

常見錯誤：把 softmax 做在錯的方向。對同一個要寫的詞 $i$，是把它對所有來源詞 $j$ 的分數一起正規化；不是對同一個來源詞，把所有輸出詞加總成 $1$。

Common error: applying softmax in the wrong direction. For one word $i$ being written, you normalize its scores over all source words $j$ together; you do not make the weights of one source word add up to $1$ over all output words.

### 3. 對齊矩陣：翻譯時的視線軌跡（The alignment matrix: where the eyes go during translation）

把每一步的權重排成一張表：第 $i$ 列是寫第 $i$ 個目標詞時的權重，第 $j$ 行是第 $j$ 個來源詞。這張表叫**對齊矩陣**（alignment matrix），每一列加起來都是 $1$。

Arrange the weights of every step into a table: row $i$ holds the weights while writing target word $i$, column $j$ is source word $j$. This table is called the **alignment matrix**, and every row adds up to $1$.

用一個簡化的例子看 `[補充]`：翻譯 `Je suis étudiant` → `I am a student`。為了好算，令三個來源狀態是 $\mathbf{h}_1=[1,0,0]^\top$、$\mathbf{h}_2=[0,1,0]^\top$、$\mathbf{h}_3=[0,0,1]^\top$，四個查詢（也就是寫每個詞之前的 $\mathbf{l}_{i-1}$）是為了示範而直接指定的：$[2,0,0]^\top$、$[0,2,0]^\top$、$[0,0.8,1.2]^\top$、$[0,0,2]^\top$，分數用內積。寫 `I` 時分數是 $2,0,0$，權重是 $0.7870,\ 0.1065,\ 0.1065$。

Look at a simplified example `[Added]`: translate "Je suis étudiant" → "I am a student". To keep the arithmetic easy, let the three source states be $\mathbf{h}_1=[1,0,0]^\top$, $\mathbf{h}_2=[0,1,0]^\top$, $\mathbf{h}_3=[0,0,1]^\top$, the four queries (the $\mathbf{l}_{i-1}$ before each word) are simply set for the demonstration as $[2,0,0]^\top$, $[0,2,0]^\top$, $[0,0.8,1.2]^\top$, $[0,0,2]^\top$, and the scores are inner products. When writing "I" the scores are $2,0,0$ and the weights $0.7870,\ 0.1065,\ 0.1065$.

![四列三行的藍色熱圖，列是目標詞 I、am、a、student，行是來源詞 Je、suis、étudiant，每格寫出權重。對角線三格 I–Je、am–suis、student–étudiant 都是深藍的 0.7870；a 那一列分散為 0.1528、0.3400、0.5072（A 4-by-3 heat map of attention weights: I, am and student each put 0.7870 on Je, suis and étudiant respectively, while a spreads its weight as 0.1528, 0.3400, 0.5072）](../assets/figures/lec10-kp03-alignment-matrix.png)

**圖說 `lec10-kp03-alignment-matrix`**〔功能：一目了然、精準度〕：先看三個深藍色格子：寫 `I` 時盯著 Je，寫 `am` 時盯著 suis，寫 `student` 時盯著 étudiant，各占 $0.7870$。再看 `a` 那一列：示範的查詢 $[0,0.8,1.2]^\top$ 同時對準 suis 和 étudiant，權重就分散開來，étudiant 拿 $0.5072$ 最多。真實模型裡常見類似的情形：法文沒有單獨對應 `a` 的詞，冠詞要看後面的名詞。最後確認每一列加起來都是 $1$。

**Caption `lec10-kp03-alignment-matrix`** [Function: seeing it at a glance, precision]: First look at the three dark blue cells: writing "I" looks at Je, writing "am" at suis, writing "student" at étudiant, with $0.7870$ each. Then look at the row of "a": the demonstration query $[0,0.8,1.2]^\top$ points at both suis and étudiant, so the weight spreads out, with étudiant getting the most, $0.5072$. Real models often show something similar: French has no word that matches "a" alone, and the article depends on the noun that follows. Finally check that every row adds up to $1$.

真實模型的對齊矩陣是訓練出來的，不是人寫的；可是訓練完畫出來，常常就像這張圖一樣大致沿著對角線，碰到語序不同的語言時會出現斜向或跳躍的格子。這讓注意力多了一個好處：可以畫出來檢查模型在看哪裡。

In a real model the alignment matrix is learned, not written by hand; yet after training it often looks like this figure, roughly along the diagonal, with slanted or jumping cells for languages with different word orders. This gives attention one more benefit: you can draw it to check where the model looks.

常見錯誤：以為權重大就代表那個詞「造成」了輸出。權重只說明 $\mathbf{c}_i$ 是怎麼混出來的；輸出還經過解碼器其他運算，權重高不一定等於影響大。

Common error: thinking that a large weight means that word "caused" the output. The weights only say how $\mathbf{c}_i$ was mixed; the output still passes through other parts of the decoder, so a high weight does not necessarily mean a large influence.

### 4. 分數怎麼算：加性與乘性（How to score: additive versus multiplicative）

第2節的 $\operatorname{similarity}$ 還沒說是什麼。來源列出幾種常見的**分數函數**（score function），用 $\mathbf{q}$ 代表查詢、$\mathbf{k}$ 代表被比較的狀態：

The $\operatorname{similarity}$ of Section 2 has not been specified yet. The source lists several common **score functions**, with $\mathbf{q}$ for the query and $\mathbf{k}$ for the state being compared:

| 名稱 | 分數 | 特點 |
|---|---|---|
| 點積 | $\mathbf{q}^\top\mathbf{k}$ | 最簡單，兩個向量的維度要相同 |
| 廣義（雙線性） | $\mathbf{q}^\top\mathbf{W}_a\mathbf{k}$ | 多一個可學習矩陣，維度可以不同 |
| 縮放點積 | $\mathbf{q}^\top\mathbf{k}/\sqrt{d_k}$ | 除以維度的平方根，KP05 說明原因 |
| 加性（Bahdanau） | $\mathbf{v}_a^\top\tanh(\mathbf{W}_a\mathbf{q}+\mathbf{U}_a\mathbf{k})$ | 像一層小型 MLP，表達力強 |

| Name | Score | Feature |
|---|---|---|
| Dot product | $\mathbf{q}^\top\mathbf{k}$ | simplest; both vectors must have the same dimension |
| General (bilinear) | $\mathbf{q}^\top\mathbf{W}_a\mathbf{k}$ | one more learnable matrix; dimensions may differ |
| Scaled dot product | $\mathbf{q}^\top\mathbf{k}/\sqrt{d_k}$ | divides by the square root of the dimension; KP05 explains why |
| Additive (Bahdanau) | $\mathbf{v}_a^\top\tanh(\mathbf{W}_a\mathbf{q}+\mathbf{U}_a\mathbf{k})$ | like a small one-layer MLP; very expressive |

來源的數字例：$\mathbf{q}=[1,2]^\top$、$\mathbf{k}=[2,1]^\top$、$\mathbf{W}_a=\begin{bmatrix}1&0\\0&2\end{bmatrix}$，維度 $d_k=2$。

The source's numerical example: $\mathbf{q}=[1,2]^\top$, $\mathbf{k}=[2,1]^\top$, $\mathbf{W}_a=\begin{bmatrix}1&0\\0&2\end{bmatrix}$ and dimension $d_k=2$.

- 點積：$1\times2+2\times1=4$。
- 廣義：$\mathbf{W}_a\mathbf{k}=[2,2]^\top$，$\mathbf{q}^\top\mathbf{W}_a\mathbf{k}=1\times2+2\times2=6$。
- 縮放點積：$4/\sqrt{2}=2.8284$。
- 加性 `[補充]`：取 $\mathbf{W}_a=\mathbf{U}_a=\mathbf{I}$、$\mathbf{v}_a=[1,1]^\top$。先算 $\mathbf{W}_a\mathbf{q}+\mathbf{U}_a\mathbf{k}=[1,2]^\top+[2,1]^\top=[3,3]^\top$，$\tanh(3)=0.99505$，分數是 $2\times0.99505=1.9901$。

- Dot product: $1\times2+2\times1=4$.
- General: $\mathbf{W}_a\mathbf{k}=[2,2]^\top$, so $\mathbf{q}^\top\mathbf{W}_a\mathbf{k}=1\times2+2\times2=6$.
- Scaled dot product: $4/\sqrt{2}=2.8284$.
- Additive `[Added]`: take $\mathbf{W}_a=\mathbf{U}_a=\mathbf{I}$ and $\mathbf{v}_a=[1,1]^\top$. First $\mathbf{W}_a\mathbf{q}+\mathbf{U}_a\mathbf{k}=[1,2]^\top+[2,1]^\top=[3,3]^\top$, $\tanh(3)=0.99505$, and the score is $2\times0.99505=1.9901$.

為什麼 Bahdanau 用加性？因為解碼器狀態和編碼器狀態常常長度不同、意義也不同，$\mathbf{W}_a$、$\mathbf{U}_a$ 先把兩者投影到同一個空間再比較。為什麼後來大家改用點積？因為點積只要矩陣乘法，在 GPU 上非常快，一次就能算完整張分數表；加性要多做 $\tanh$ 和兩次投影，難以這樣加速。

Why did Bahdanau use the additive form? Because the decoder and encoder states often have different lengths and meanings, and $\mathbf{W}_a$, $\mathbf{U}_a$ first project both into a common space before comparing. Why did people later switch to the dot product? Because the dot product needs only matrix multiplication, which is very fast on a GPU and computes the whole score table at once; the additive form needs an extra $\tanh$ and two projections, which are hard to speed up that way.

注意力也有代價。寫每一個目標詞，都要和每一個來源詞算一次分數，所以一共要算 $T_x\times T_y$ 個分數。第3節的例子是 $3\times4=12$ 個；來源 $20$ 詞、目標 $25$ 詞，就是 $500$ 個 `[補充]`。換來的是：任何一個來源詞，都只隔「一次加權」就能被解碼器看到。

Attention has a price too. For every target word you compute a score with every source word, so there are $T_x\times T_y$ scores in all. The example of Section 3 has $3\times4=12$; a $20$-word source and a $25$-word target give $500$ `[Added]`. In return, every source word is only "one weighting" away from the decoder.

常見錯誤：以為分數函數的參數越多越好。廣義形式的 $\mathbf{W}_a$ 有 $d_k^2$ 個參數，維度一高就容易過擬合。提出 Transformer 的論文觀察到：維度小時，點積和加性表現差不多；維度大時，沒有縮放的點積反而比較差，原因是內積太大、softmax 太尖，KP05 會算給你看。除以 $\sqrt{d_k}$ 以後，點積就能追上，而且快得多。

Common error: thinking that more parameters in a score function always help. The $\mathbf{W}_a$ of the general form has $d_k^2$ parameters, which easily overfit when the dimension is high. The paper that introduced the Transformer observed that with a small dimension the dot product and the additive form perform about the same, while with a large dimension the unscaled dot product does worse, because inner products get too large and softmax too sharp; KP05 works this out. After dividing by $\sqrt{d_k}$, the dot product catches up and is much faster.

### 本單元重點整理（KP03 summary）

**一句話重點**：帶注意力的 Seq2Seq 在寫每個詞之前，拿解碼器狀態 $\mathbf{l}_{i-1}$ 當查詢，和編碼器的每個 $\mathbf{h}_j$ 算對齊分數，經 softmax 得到權重 $a_{ij}$，再加權得到專用的上下文向量 $\mathbf{c}_i$；這讓任何來源詞都能被直接取用，代價是要算 $T_x\times T_y$ 個分數。

**One-sentence summary**: before writing each word, Seq2Seq with attention uses the decoder state $\mathbf{l}_{i-1}$ as a query, scores it against every encoder state $\mathbf{h}_j$, turns the scores into weights $a_{ij}$ with softmax, and forms a dedicated context vector $\mathbf{c}_i$ by weighting; every source word can then be reached directly, at the price of $T_x\times T_y$ scores.

**白話說明**：KP02 是先把整篇原文記成一張紙條再翻；這一節是把原文攤在桌上，每寫一個詞就回頭看一眼，看最相關的地方看得最仔細。

**Plain explanation**: KP02 writes the whole original on one note and then translates; this note keeps the original open on the desk, glances back at it before every word, and looks most closely at the most relevant spots.

**正式定義**：$s_{ij}=\operatorname{similarity}(\mathbf{l}_{i-1},\mathbf{h}_j)$，$a_{ij}=\exp(s_{ij})/\sum_{k}\exp(s_{ik})$，$\mathbf{c}_i=\sum_j a_{ij}\mathbf{h}_j$，$\mathbb{P}(\mathbf{y}_i\mid\mathbf{y}_1,\dots,\mathbf{y}_{i-1})=g(\mathbf{y}_{i-1},\mathbf{l}_i,\mathbf{c}_i)$。

**Formal definition**: $s_{ij}=\operatorname{similarity}(\mathbf{l}_{i-1},\mathbf{h}_j)$, $a_{ij}=\exp(s_{ij})/\sum_{k}\exp(s_{ik})$, $\mathbf{c}_i=\sum_j a_{ij}\mathbf{h}_j$, $\mathbb{P}(\mathbf{y}_i\mid\mathbf{y}_1,\dots,\mathbf{y}_{i-1})=g(\mathbf{y}_{i-1},\mathbf{l}_i,\mathbf{c}_i)$.

**例子**：分數 $1.0$、$2.0$ 得權重 $0.2689$、$0.7311$，$\mathbf{c}_i=[0.2689,\ 0.7311]^\top$。翻譯例中，`I`、`am`、`student` 各把 $0.7870$ 放在對應的法文詞，`a` 分散為 $0.1528,\ 0.3400,\ 0.5072$。$\mathbf{q}=[1,2]^\top$、$\mathbf{k}=[2,1]^\top$ 的點積 $4$、廣義 $6$、縮放 $2.8284$。

**Example**: scores $1.0$ and $2.0$ give weights $0.2689$ and $0.7311$ and $\mathbf{c}_i=[0.2689,\ 0.7311]^\top$. In the translation example, "I", "am" and "student" each put $0.7870$ on the matching French word, while "a" spreads as $0.1528,\ 0.3400,\ 0.5072$. For $\mathbf{q}=[1,2]^\top$ and $\mathbf{k}=[2,1]^\top$, the dot product is $4$, the general score $6$ and the scaled score $2.8284$.

**常見迷思**：

- 「加了注意力就不需要 RNN。」錯，這一版仍用 RNN 產生 $\mathbf{h}_j$；第11週的 Transformer 才拿掉 RNN。
- 「softmax 是對同一個來源詞跨所有輸出詞做。」錯，是對同一個輸出詞跨所有來源詞做，每一列加總為 $1$。
- 「權重最大的詞就是造成輸出的原因。」錯，權重只描述 $\mathbf{c}_i$ 怎麼混合，不等於因果貢獻。

**Common misconceptions**:

- "With attention there is no need for an RNN." Wrong: this version still uses an RNN to produce the $\mathbf{h}_j$; the Transformer of Week 11 removes it.
- "Softmax runs over all output words for one source word." Wrong: it runs over all source words for one output word, and every row adds up to $1$.
- "The word with the largest weight caused the output." Wrong: the weights only describe how $\mathbf{c}_i$ is mixed, not a causal contribution.

**圖**：本單元有兩張圖。

- `lec10-kp03-one-step`：解碼的一步，查詢連到每個來源狀態，線粗代表權重，加權和就是 $\mathbf{c}_i$。
- `lec10-kp03-alignment-matrix`：翻譯例的對齊矩陣，大致沿對角線，`a` 的權重分散。

**Figures** — this unit has two figures.

- `lec10-kp03-one-step`: one decoding step; the query is linked to every source state, line width shows the weight, and the weighted sum is $\mathbf{c}_i$.
- `lec10-kp03-alignment-matrix`: the alignment matrix of the translation example, roughly diagonal, with the weight of "a" spread out.

### 小練習（Practice）

1. 三個來源狀態 $\mathbf{h}_1=[2,0]^\top$、$\mathbf{h}_2=[0,2]^\top$、$\mathbf{h}_3=[1,1]^\top$，某一步的對齊分數是 $0,\ 1,\ 3$。算出三個權重與 $\mathbf{c}_i$。提示：$\exp(1)=2.718$、$\exp(3)=20.086$。
2. 來源句 $20$ 個詞、譯文 $25$ 個詞。帶注意力的 Seq2Seq 一共要算幾個對齊分數？KP02 的 Seq2Seq 要算幾個？
3. 第3節翻譯例中，寫 `a` 時權重為什麼沒有集中在某一個法文詞？如果查詢改成 $[0,0,2]^\top$，權重會變成什麼？

1. Three source states $\mathbf{h}_1=[2,0]^\top$, $\mathbf{h}_2=[0,2]^\top$, $\mathbf{h}_3=[1,1]^\top$ have alignment scores $0,\ 1,\ 3$ at some step. Compute the three weights and $\mathbf{c}_i$. Hint: $\exp(1)=2.718$ and $\exp(3)=20.086$.
2. The source sentence has $20$ words and the translation $25$. How many alignment scores does Seq2Seq with attention compute in total? How many does the Seq2Seq of KP02 compute?
3. In the translation example of Section 3, why is the weight not concentrated on one French word when writing "a"? If the query were $[0,0,2]^\top$, what would the weights become?

**第 1 題解答** `[補充]`：分母 $1+2.718+20.086=23.804$，權重 $0.0420,\ 0.1142,\ 0.8438$。$\mathbf{c}_i=0.0420[2,0]^\top+0.1142[0,2]^\top+0.8438[1,1]^\top=[0.9278,\ 1.0722]^\top$。

**Solution to question 1** `[Added]`: the denominator is $1+2.718+20.086=23.804$ and the weights are $0.0420,\ 0.1142,\ 0.8438$. $\mathbf{c}_i=0.0420[2,0]^\top+0.1142[0,2]^\top+0.8438[1,1]^\top=[0.9278,\ 1.0722]^\top$.

**第 2 題解答** `[補充]`：$20\times25=500$ 個。KP02 的 Seq2Seq 不算對齊分數，是 $0$ 個；它每一步都用同一個 $\mathbf{c}=\mathbf{h}_{20}$。

**Solution to question 2** `[Added]`: $20\times25=500$. The Seq2Seq of KP02 computes no alignment scores at all, $0$; it uses the same $\mathbf{c}=\mathbf{h}_{20}$ at every step.

**第 3 題解答** `[補充]`：數學上的原因是查詢 $[0,0.8,1.2]^\top$ 和 $\mathbf{h}_2$、$\mathbf{h}_3$ 的內積都不是 $0$（分別是 $0.8$、$1.2$），所以權重分散；語言上的解讀是法文沒有單獨對應 `a` 的詞。查詢改成 $[0,0,2]^\top$ 時，分數是 $0,0,2$，權重變成 $0.1065,\ 0.1065,\ 0.7870$，和寫 `student` 時一樣。

**Solution to question 3** `[Added]`: the mathematical reason is that the query $[0,0.8,1.2]^\top$ has nonzero inner products with both $\mathbf{h}_2$ and $\mathbf{h}_3$ ($0.8$ and $1.2$), so the weight spreads out; the linguistic reading is that French has no word that matches "a" alone. With the query $[0,0,2]^\top$, the scores are $0,0,2$ and the weights become $0.1065,\ 0.1065,\ 0.7870$, the same as when writing "student".

### 小測驗（Quiz，來源原題）

來源的兩題自我檢測保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The two self-check questions of the source are kept with all options; the source page does not mark the answers directly, and the explanations below follow the source text.

1. 具注意力機制的 Seq2Seq 如何克服傳統固定向量的瓶頸？
   - (A) 將序列截斷為固定長度 $5$
   - (B) 為每個輸出詞動態計算所有編碼器狀態的加權和 $\mathbf{c}_i=\sum_j a_{ij}\mathbf{h}_j$
   - (C) 捨棄所有編碼器狀態，只使用常數向量
   - (D) 使用雙向 RNN 代替所有神經元
2. 注意力權重向量 $[a_{i1},a_{i2},\dots,a_{in}]^\top$ 滿足何種數學性質？
   - (A) 所有元素平方和為 $1$
   - (B) 元素均大於等於 $0$ 且總和恆等於 $1$（離散機率分佈）
   - (C) 所有元素均小於 $0$
   - (D) 元素均為整數

1. How does Seq2Seq with attention overcome the bottleneck of the traditional fixed vector?
   - (A) By truncating sequences to a fixed length of $5$
   - (B) By computing, for every output word, a dynamic weighted sum of all encoder states $\mathbf{c}_i=\sum_j a_{ij}\mathbf{h}_j$
   - (C) By dropping all encoder states and using a constant vector
   - (D) By replacing all neurons with a bidirectional RNN
2. What mathematical property does the attention weight vector $[a_{i1},a_{i2},\dots,a_{in}]^\top$ satisfy?
   - (A) The sum of the squares of its entries is $1$
   - (B) Its entries are at least $0$ and always add up to $1$ (a discrete probability distribution)
   - (C) All entries are negative
   - (D) All entries are integers

**解析** `[補充]`：第 1 題選 (B)：這就是第1節的 $\mathbf{c}_i$；(D) 的雙向 RNN 在 KP02 已說明只能緩解。第 2 題選 (B)：softmax 的輸出都大於 $0$、加總為 $1$；(A) 描述的是長度為 $1$ 的單位向量，不是權重。

**Explanations** `[Added]`: question 1 is B: this is the $\mathbf{c}_i$ of Section 1; KP02 showed that the bidirectional RNN of D only eases the problem. Question 2 is B: the outputs of softmax are positive and add up to $1$; A describes a unit vector of length $1$, not weights.

### 留給下一單元的問題（The question for the next unit）

這一節的 $\mathbf{h}_j$ 同時做了兩件事：拿來和查詢比相關程度，也拿來被加權取用。如果把「拿來比的」和「拿來取用的」分開成兩個不同的向量，會不會更有彈性？下一單元（KP04）把注意力寫成一般的查詢機制：Query、Key 與 Value。

In this note each $\mathbf{h}_j$ does two jobs: it is compared with the query, and it is the content that gets weighted and taken. If "what is compared" and "what is taken" were split into two different vectors, would that be more flexible? The next unit (KP04) writes attention as a general lookup mechanism: Query, Key and Value.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第9章「注意力機制與 Transformer 核心原理」§2.3 與教授祕笈「Bahdanau 動態對齊機制」「注意力評分函數演進」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §2.3 of Chapter 9, "Core principles of attention mechanisms and the Transformer", of CCC-LNS IDL_115 and its professor's notes "Bahdanau dynamic alignment" and "The evolution of attention score functions" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源寫 $a_{i1}\approx0.269$、$a_{i2}\approx0.731$；本講義用小數第四位 $0.2689$、$0.7311$。
- 來源的正文把對齊分數寫成 $s_{ij}$、教授祕笈寫成 $e_{ij}$，解碼器狀態在正文是 $\mathbf{l}$、在祕笈是 $s$；本講義統一用 $s_{ij}$ 表示分數、$\mathbf{l}$ 表示解碼器狀態，與 KP01、KP02 一致。
- 來源用粗斜體 `\boldsymbol` 表示向量；本課程網站不支援，改用 `\mathbf`，意義相同。
- 本講義補上：翻譯例的對齊矩陣與單步示意圖（來源只有互動示範）、加性分數的數字例、分數個數 $T_x\times T_y$ 的數字例、三題小練習與解答。
- 來源教授祕笈說「維度充足時，加性與點積理論性能差異極小」；提出 Transformer 的論文（Vaswani 等人，2017）的觀察是：維度大時，沒有縮放的點積不如加性，縮放後才追上。本講義依後者改寫第4節的常見錯誤。
- 取捨：來源教授祕笈提到 Luong 注意力使用當前步的狀態 $\mathbf{l}_i$、Bahdanau 使用前一步的 $\mathbf{l}_{i-1}$，本講義只採 Bahdanau 的寫法，不展開比較；「把 $\mathbf{U}_a\mathbf{h}_j$ 預先算好」的實作技巧也不寫。

- The source writes $a_{i1}\approx0.269$ and $a_{i2}\approx0.731$; this note uses four decimals, $0.2689$ and $0.7311$.
- The source's main text writes the alignment score as $s_{ij}$ and the professor's notes as $e_{ij}$, and the decoder state is $\mathbf{l}$ in the main text and $s$ in the notes; this note uses $s_{ij}$ for scores and $\mathbf{l}$ for the decoder state throughout, consistent with KP01 and KP02.
- The source writes vectors with bold italic `\boldsymbol`; the course website does not support it, so this note uses `\mathbf` with the same meaning.
- This note adds: the alignment matrix and the one-step diagram of the translation example (the source has only an interactive demo), a numerical example of the additive score, a numerical example of the number of scores $T_x\times T_y$, and three practice questions with solutions.
- The professor's notes say that "with enough dimensions the additive and dot-product forms differ very little in theory"; the paper that introduced the Transformer (Vaswani and colleagues, 2017) observed that with a large dimension the unscaled dot product does worse than the additive form and catches up only after scaling. This note rewrites the common error of Section 4 accordingly.
- Choices: the professor's notes mention that Luong attention uses the current state $\mathbf{l}_i$ while Bahdanau uses the previous $\mathbf{l}_{i-1}$; this note follows Bahdanau only and does not compare them, and leaves out the implementation trick of precomputing $\mathbf{U}_a\mathbf{h}_j$.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 帶注意力的 Seq2Seq | Seq2Seq with attention |
| 對齊分數 | alignment score |
| 對齊矩陣 | alignment matrix |
| 注意力權重 | attention weight |
| 分數函數 | score function |
| 點積 | dot product |
| 廣義（雙線性）分數 | general (bilinear) score |
| 縮放點積 | scaled dot product |
| 加性注意力 | additive attention |
| 多層感知器 | multilayer perceptron (MLP) |
| 循環神經網路 | recurrent neural network (RNN) |
| 圖形處理器 | graphics processing unit (GPU) |
| 過擬合 | overfitting |
| 鍵、值（KP04） | key, value |
