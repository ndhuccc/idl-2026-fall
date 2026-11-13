# 第10週 KP05：自注意力，每個詞拿整句重新描述自己（Self-attention: every word redescribes itself using the whole sentence）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第10週的第五個知識點（KP05），接在 KP04 之後。KP04 的查詢來自解碼器，鍵和值來自編碼器。這一節讓三者都來自同一句話：每個詞同時當查詢者、被查者與內容，用整句重新描述自己。這就是第11週 Transformer 的核心零件。

This is the fifth knowledge point (KP05) of Week 10 and follows KP04. In KP04 the query came from the decoder and the keys and values from the encoder. In this note all three come from the same sentence: every word is at once the seeker, the one being sought and the content, and redescribes itself using the whole sentence. This is the core part of the Transformer of Week 11.

先備知識是 KP04 的 Query、Key、Value 與縮放內積，以及矩陣乘法的維度規則。本講義依課程網頁教材 CCC-LNS 第9章改寫，文中「來源」就是指它；本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec10-kp05-figures.py` 算出，可以自己重算。

The prerequisites are the Query, Key, Value and scaled inner product of KP04, and the dimension rule of matrix multiplication. This note is rewritten from Chapter 9 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec10-kp05-figures.py`, so you can recompute it yourself.

## KP05 自注意力，每個詞拿整句重新描述自己（Self-attention: every word redescribes itself using the whole sentence）

### 學習目標（Learning objectives）

讀完之後，你能用例子說明為什麼一個詞的意思要看上下文，固定的詞向量不夠用。你能寫出自注意力怎麼用三個矩陣 $\mathbf{W}_Q$、$\mathbf{W}_K$、$\mathbf{W}_V$ 從同一句話產生查詢、鍵與值。你能寫出矩陣形式 $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$，標出每個矩陣的形狀，並手算一個小例子。你也能說明為什麼要除以 $\sqrt{d_k}$，而不是 $d_k$ 或什麼都不除。

After reading, you can explain with an example why the meaning of a word depends on its context and fixed word vectors are not enough. You can write how self-attention uses three matrices $\mathbf{W}_Q$, $\mathbf{W}_K$, $\mathbf{W}_V$ to produce queries, keys and values from the same sentence. You can write the matrix form $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$, mark the shape of every matrix, and compute a small example by hand. You can also explain why we divide by $\sqrt{d_k}$, rather than by $d_k$ or by nothing.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $n$ | 句子的詞數 | number of words in the sentence |
| $\mathbf{x}_i\in\mathbb{R}^{d}$ | 第 $i$ 個詞的輸入向量，$d$ 維 | input vector of word $i$, of dimension $d$ |
| $d_k$ | 查詢與鍵的維度，就是 KP04 的 $p$ | dimension of the queries and keys, the $p$ of KP04 |
| $d_v$ | 值的維度，就是 KP04 的 $r$ | dimension of the values, the $r$ of KP04 |
| $\mathbf{X}\in\mathbb{R}^{n\times d}$ | 把 $n$ 個詞向量排成列的矩陣（第 $i$ 列是 $\mathbf{x}_i^\top$） | matrix whose rows are the $n$ word vectors (row $i$ is $\mathbf{x}_i^\top$) |
| $\mathbf{W}_Q,\mathbf{W}_K\in\mathbb{R}^{d\times d_k}$ | 產生查詢與鍵的可學習矩陣 | learnable matrices that produce queries and keys |
| $\mathbf{W}_V\in\mathbb{R}^{d\times d_v}$ | 產生值的可學習矩陣 | learnable matrix that produces values |
| $\mathbf{Q},\mathbf{K}\in\mathbb{R}^{n\times d_k}$、$\mathbf{V}\in\mathbb{R}^{n\times d_v}$ | 所有詞的查詢、鍵、值，每列一個詞 | queries, keys and values of all words, one word per row |
| $\mathbf{A}\in\mathbb{R}^{n\times n}$ | 注意力權重矩陣，$a_{ij}$ 是詞 $i$ 分給詞 $j$ 的權重 | attention weight matrix; $a_{ij}$ is the weight word $i$ gives to word $j$ |
| $\mathbf{Z}\in\mathbb{R}^{n\times d_v}$ | 自注意力的輸出，第 $i$ 列是詞 $i$ 的新表示 $\mathbf{z}_i^\top$ | output of self-attention; row $i$ is the new representation $\mathbf{z}_i^\top$ of word $i$ |
| $\operatorname{Var}$ | 變異數，數值散開的程度 | variance, how spread out a number is |

### 1. 一個詞的意思要看上下文（The meaning of a word depends on its context）

早期的**詞嵌入**（word embedding），例如 Word2Vec、GloVe，給詞彙表裡每個詞一個固定的向量。不論「bank」出現在「river bank」還是「bank account」，它的向量都一樣。

Early **word embeddings**, such as Word2Vec and GloVe, give every word of the vocabulary one fixed vector. Whether "bank" appears in "river bank" or "bank account", its vector is the same.

來源的例句：`The FBI is chasing a criminal on the run.` 這裡的 run 不是「跑步」，而是和前面的 on the 合起來，意思是「在逃」。讀到 chasing 時，要同時想到主詞 FBI 和受詞 criminal；讀到 run 時，又要連回 chasing 和 criminal。

The source's example: "The FBI is chasing a criminal on the run." Here "run" does not mean "running"; together with the "on the" before it, it means "fleeing". When reading "chasing", you must think of both the subject FBI and the object criminal; when reading "run", you must link back to "chasing" and "criminal".

所以一個詞好的表示，不該只看它自己，而要由句子裡相關的詞組合出來。來源稱之為**複合嵌入**（composite embedding）。換成 KP04 的語言：每個詞都該拿自己當查詢，到同一句話裡去查，把查到的內容混進自己的新表示。

So a good representation of a word should not look only at the word itself; it should be composed from the related words of the sentence. The source calls this a **composite embedding**. In the language of KP04: every word should use itself as a query, look up the same sentence, and mix what it finds into its new representation.

常見錯誤：以為「自注意力」是每個詞只注意自己。「自」的意思是查詢、鍵、值都來自同一句話；每個詞注意的是整句，包括自己。

Common error: thinking that "self-attention" means every word attends only to itself. "Self" means that the queries, keys and values all come from the same sentence; every word attends to the whole sentence, itself included.

### 2. 從同一句話產生查詢、鍵與值（Queries, keys and values from the same sentence）

KP04 說過，查詢、鍵、值是三種角色，可以由同一份資料轉換出來。自注意力就是這樣做：用三個可學習的矩陣，把每個詞向量 $\mathbf{x}_i$ 分別轉換成它的查詢、鍵與值。

KP04 said that query, key and value are three roles that can be produced from the same data. Self-attention does exactly that: three learnable matrices turn each word vector $\mathbf{x}_i$ into its query, its key and its value.

$$
\mathbf{q}_i=\mathbf{W}_Q^\top\mathbf{x}_i,\qquad \mathbf{k}_i=\mathbf{W}_K^\top\mathbf{x}_i,\qquad \mathbf{v}_i=\mathbf{W}_V^\top\mathbf{x}_i.
$$

然後每個詞 $i$ 拿自己的查詢 $\mathbf{q}_i$，和全句每個詞的鍵 $\mathbf{k}_j$ 算縮放內積，經 softmax 得到權重，再把全句的值加權平均：

Then every word $i$ takes its own query $\mathbf{q}_i$, computes the scaled inner product with the key $\mathbf{k}_j$ of every word of the sentence, turns the results into weights with softmax, and averages the values of the whole sentence with these weights:

$$
a_{ij}=\frac{\exp\!\left(\mathbf{q}_i^\top\mathbf{k}_j/\sqrt{d_k}\right)}{\sum_{m=1}^{n}\exp\!\left(\mathbf{q}_i^\top\mathbf{k}_m/\sqrt{d_k}\right)},\qquad \mathbf{z}_i=\sum_{j=1}^{n}a_{ij}\,\mathbf{v}_j.
$$

$\mathbf{z}_i$ 就是第 $i$ 個詞的新表示：它混入了句子裡和它相關的詞。三個矩陣 $\mathbf{W}_Q$、$\mathbf{W}_K$、$\mathbf{W}_V$ 是訓練學出來的；學到什麼樣的「相關」，由任務決定。

$\mathbf{z}_i$ is the new representation of word $i$: it has the related words of the sentence mixed in. The three matrices $\mathbf{W}_Q$, $\mathbf{W}_K$, $\mathbf{W}_V$ are learned in training; which kind of "relatedness" they learn depends on the task.

常見錯誤：以為自注意力沒有參數。分數和加權本身沒有參數，但 $\mathbf{W}_Q$、$\mathbf{W}_K$、$\mathbf{W}_V$ 是可學習的矩陣，自注意力的本事全在它們身上。

Common error: thinking that self-attention has no parameters. The scoring and the weighting have none, but $\mathbf{W}_Q$, $\mathbf{W}_K$, $\mathbf{W}_V$ are learnable matrices, and all the power of self-attention lies in them.

### 3. 整句一次算：矩陣形式（The whole sentence at once: the matrix form）

把每個詞向量排成矩陣 $\mathbf{X}$ 的一列，查詢、鍵、值也就都變成矩陣，每一列是一個詞。$\mathbf{X}\mathbf{W}_Q$ 的第 $i$ 列是 $\mathbf{x}_i^\top\mathbf{W}_Q=(\mathbf{W}_Q^\top\mathbf{x}_i)^\top=\mathbf{q}_i^\top$，正是第2節的查詢：

Put every word vector as one row of a matrix $\mathbf{X}$; then the queries, keys and values also become matrices with one word per row. Row $i$ of $\mathbf{X}\mathbf{W}_Q$ is $\mathbf{x}_i^\top\mathbf{W}_Q=(\mathbf{W}_Q^\top\mathbf{x}_i)^\top=\mathbf{q}_i^\top$, exactly the query of Section 2:

$$
\mathbf{Q}=\mathbf{X}\mathbf{W}_Q,\qquad \mathbf{K}=\mathbf{X}\mathbf{W}_K,\qquad \mathbf{V}=\mathbf{X}\mathbf{W}_V.
$$

整句的自注意力可以寫成一行：

Self-attention over the whole sentence fits in one line:

$$
\mathbf{Z}=\operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}.
$$

這裡的 softmax 是逐列做：每一列各自正規化，加總為 $1$。看形狀：$\mathbf{Q}\mathbf{K}^\top$ 是 $(n\times d_k)(d_k\times n)=n\times n$，第 $i$ 列第 $j$ 行就是 $\mathbf{q}_i^\top\mathbf{k}_j$；softmax 後得到 $n\times n$ 的權重矩陣 $\mathbf{A}$；再乘 $\mathbf{V}$，得到 $(n\times n)(n\times d_v)=n\times d_v$，每一列是一個詞的新表示。

The softmax here works row by row: each row is normalized on its own and adds up to $1$. Look at the shapes: $\mathbf{Q}\mathbf{K}^\top$ is $(n\times d_k)(d_k\times n)=n\times n$, and its row $i$, column $j$ entry is $\mathbf{q}_i^\top\mathbf{k}_j$; after softmax this is the $n\times n$ weight matrix $\mathbf{A}$; multiplying by $\mathbf{V}$ gives $(n\times n)(n\times d_v)=n\times d_v$, one new representation per row.

例子一 `[補充]`：三個詞，詞向量就是 KP01 的三個向量 $\mathbf{x}_1=[1,0]^\top$、$\mathbf{x}_2=[0,1]^\top$、$\mathbf{x}_3=[1,1]^\top$。為了好算，令 $\mathbf{W}_Q=\mathbf{W}_K=\mathbf{W}_V=\mathbf{I}$，所以 $\mathbf{Q}=\mathbf{K}=\mathbf{V}=\mathbf{X}$，$d_k=2$。

Example one `[Added]`: three words whose vectors are the three vectors of KP01, $\mathbf{x}_1=[1,0]^\top$, $\mathbf{x}_2=[0,1]^\top$, $\mathbf{x}_3=[1,1]^\top$. To keep it simple, let $\mathbf{W}_Q=\mathbf{W}_K=\mathbf{W}_V=\mathbf{I}$, so $\mathbf{Q}=\mathbf{K}=\mathbf{V}=\mathbf{X}$ and $d_k=2$.

$$
\mathbf{Q}\mathbf{K}^\top=\begin{bmatrix}1&0&1\\0&1&1\\1&1&2\end{bmatrix},\qquad \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{2}}=\begin{bmatrix}0.7071&0&0.7071\\0&0.7071&0.7071\\0.7071&0.7071&1.4142\end{bmatrix}.
$$

逐列做 softmax，再乘 $\mathbf{V}$：

Apply softmax row by row, then multiply by $\mathbf{V}$:

$$
\mathbf{A}=\begin{bmatrix}0.4011&0.1978&0.4011\\0.1978&0.4011&0.4011\\0.2483&0.2483&0.5035\end{bmatrix},\qquad \mathbf{Z}=\mathbf{A}\mathbf{V}=\begin{bmatrix}0.8022&0.5989\\0.5989&0.8022\\0.7517&0.7517\end{bmatrix}.
$$

![左圖是 3 乘 3 的紫色熱圖，列是三個詞、行是三個鍵，每格寫出權重：第一列 0.4011、0.1978、0.4011，第二列 0.1978、0.4011、0.4011，第三列 0.2483、0.2483、0.5035。右圖是平面上三個黑點 x_1=(1,0)、x_2=(0,1)、x_3=(1,1)，各有一條紫色箭頭指向自己的新表示 z_1、z_2、z_3，三顆紫色星都往三角形中間靠攏（A 3-by-3 heat map of the self-attention weights, and a plane showing each word vector moving to its new representation, all three moving toward the middle of the triangle）](../assets/figures/lec10-kp05-three-word-attention.png)

**圖說 `lec10-kp05-three-word-attention`**〔功能：一目了然、空間幾何具象化〕：先看左圖的每一列：每個詞把最多的權重給自己和 $\mathbf{x}_3$，因為 $\mathbf{x}_3$ 和誰都有共同的分量；三列各自加總為 $1$。再看右圖：每個詞從黑點沿紫色箭頭移到紫色星，新表示 $\mathbf{z}_i$ 都往其他詞靠了一些，這就是「用整句重新描述自己」。

**Caption `lec10-kp05-three-word-attention`** [Function: seeing it at a glance, picturing the geometry]: First look at each row of the left panel: every word gives the most weight to itself and to $\mathbf{x}_3$, because $\mathbf{x}_3$ shares a component with everyone; each row adds up to $1$. Then look at the right panel: each word moves along a purple arrow from its black dot to a purple star, and every new representation $\mathbf{z}_i$ has moved somewhat toward the other words; this is "redescribing itself using the whole sentence".

例子二（來源的數字）：$n=2$，直接給定 $\mathbf{Q}=\begin{bmatrix}1&0\\0&1\end{bmatrix}$、$\mathbf{K}=\begin{bmatrix}1&1\\0&2\end{bmatrix}$、$\mathbf{V}=\begin{bmatrix}10&0\\0&20\end{bmatrix}$、$d_k=2$。則 $\mathbf{Q}\mathbf{K}^\top=\begin{bmatrix}1&0\\1&2\end{bmatrix}$，除以 $\sqrt{2}$ 得 $\begin{bmatrix}0.7071&0\\0.7071&1.4142\end{bmatrix}$，逐列 softmax 得 $\mathbf{A}=\begin{bmatrix}0.6698&0.3302\\0.3302&0.6698\end{bmatrix}$，輸出 $\mathbf{A}\mathbf{V}=\begin{bmatrix}6.6976&6.6048\\3.3024&13.3952\end{bmatrix}$。

Example two (the source's numbers): $n=2$, with $\mathbf{Q}=\begin{bmatrix}1&0\\0&1\end{bmatrix}$, $\mathbf{K}=\begin{bmatrix}1&1\\0&2\end{bmatrix}$, $\mathbf{V}=\begin{bmatrix}10&0\\0&20\end{bmatrix}$ and $d_k=2$ given directly. Then $\mathbf{Q}\mathbf{K}^\top=\begin{bmatrix}1&0\\1&2\end{bmatrix}$; dividing by $\sqrt{2}$ gives $\begin{bmatrix}0.7071&0\\0.7071&1.4142\end{bmatrix}$; row-wise softmax gives $\mathbf{A}=\begin{bmatrix}0.6698&0.3302\\0.3302&0.6698\end{bmatrix}$, and the output is $\mathbf{A}\mathbf{V}=\begin{bmatrix}6.6976&6.6048\\3.3024&13.3952\end{bmatrix}$.

常見錯誤：softmax 做錯方向。每一列對應一個查詢，要對「這個查詢對所有鍵的分數」正規化，也就是沿著列、在最後一個維度上做；寫程式時是 `dim=-1`。

Common error: applying softmax in the wrong direction. Each row belongs to one query, and you normalize "this query's scores against all keys", that is, along the row, over the last dimension; in code this is `dim=-1`.

### 4. 為什麼要除以 $\sqrt{d_k}$（Why divide by $\sqrt{d_k}$）

KP01 第4節說過：分數整體放大，softmax 會變尖。內積的大小正好會隨維度自動變大。

Section 4 of KP01 showed that scaling all scores up makes softmax sharper. The size of an inner product grows with the dimension by itself.

為什麼？假設查詢和鍵的每個分量都是平均 $0$、變異數 $1$，而且彼此獨立。內積 $\mathbf{q}^\top\mathbf{k}=\sum_{m=1}^{d_k}q_mk_m$ 是 $d_k$ 個獨立小項的和。每一項 $q_mk_m$ 的平均是 $0\times0=0$；因為平均是 $0$，它的變異數就是平方的平均，$\operatorname{Var}(q_mk_m)=E[q_m^2]\,E[k_m^2]=1\times1=1$。獨立的項相加，變異數也相加；而一個數除以 $c$，變異數會除以 $c^2$：

Why? Suppose every component of the query and the key has mean $0$ and variance $1$, all independent. The inner product $\mathbf{q}^\top\mathbf{k}=\sum_{m=1}^{d_k}q_mk_m$ is a sum of $d_k$ independent small terms. Each $q_mk_m$ has mean $0\times0=0$; since its mean is $0$, its variance is the mean of its square, $\operatorname{Var}(q_mk_m)=E[q_m^2]\,E[k_m^2]=1\times1=1$. When independent terms are added, their variances add; and dividing a number by $c$ divides its variance by $c^2$:

$$
\operatorname{Var}\!\left(\mathbf{q}^\top\mathbf{k}\right)=d_k,\qquad \operatorname{Var}\!\left(\frac{\mathbf{q}^\top\mathbf{k}}{\sqrt{d_k}}\right)=\frac{d_k}{d_k}=1.
$$

所以內積的典型大小（標準差）是 $\sqrt{d_k}$。$d_k=64$ 時，單一分數常常偏離 $0$ 到 $\pm8$，甚至 $\pm16$；除以 $\sqrt{d_k}$ 後，標準差拉回 $1$，不論維度多大都一樣。

So the typical size (the standard deviation) of an inner product is $\sqrt{d_k}$. With $d_k=64$, a single score often lands $\pm8$ or even $\pm16$ away from $0$; after dividing by $\sqrt{d_k}$, the standard deviation is back to $1$, whatever the dimension.

分數太大有什麼壞處？來源的數字：$d_k=64$，兩個分數是 $16$ 和 $0$。不縮放時，softmax 是 $[0.9999999,\ 0.0000001]$，幾乎只選一個。softmax 對自己輸入的變化率是 $a(1-a)$，這時只剩約 $1.1\times10^{-7}$，梯度幾乎傳不回去。除以 $\sqrt{64}=8$ 後，分數變成 $2$ 和 $0$，softmax 是 $[0.8808,\ 0.1192]$，變化率 $0.8808\times0.1192=0.105$，梯度正常流動。

What harm do large scores do? The source's numbers: $d_k=64$ and two scores $16$ and $0$. Without scaling, softmax gives $[0.9999999,\ 0.0000001]$, nearly picking one. The rate of change of softmax with respect to its input is $a(1-a)$, which is only about $1.1\times10^{-7}$ here, so the gradient barely flows back. After dividing by $\sqrt{64}=8$, the scores become $2$ and $0$, softmax gives $[0.8808,\ 0.1192]$, the rate of change is $0.8808\times0.1192=0.105$, and the gradient flows normally.

模擬 `[補充]`：每次試驗取 $1$ 個隨機查詢和 $8$ 個隨機鍵（每個分量獨立、平均 $0$、變異數 $1$），每種維度試 $2000$ 次；隨機種子只在開頭設一次（$0$），維度依表中順序執行。量所有分數合起來的標準差，以及每次試驗 softmax 最大權重的平均值。

Simulation `[Added]`: each trial draws $1$ random query and $8$ random keys (every component independent, mean $0$, variance $1$), with $2000$ trials per dimension; the random seed is set once at the start ($0$), and the dimensions run in the order of the table. Measure the standard deviation of all scores pooled together, and the average over trials of the largest softmax weight.

| $d_k$ | 內積標準差 | $\sqrt{d_k}$ | 最大權重（不縮放） | 最大權重（除以 $\sqrt{d_k}$） | 最大權重（除以 $d_k$） |
|---|---|---|---|---|---|
| $4$ | $2.004$ | $2.000$ | $0.5235$ | $0.3420$ | $0.2268$ |
| $16$ | $4.032$ | $4.000$ | $0.7499$ | $0.3577$ | $0.1743$ |
| $64$ | $8.009$ | $8.000$ | $0.8734$ | $0.3559$ | $0.1483$ |
| $256$ | $15.868$ | $16.000$ | $0.9373$ | $0.3595$ | $0.1364$ |
| $512$ | $22.775$ | $22.627$ | $0.9487$ | $0.3584$ | $0.1330$ |

| $d_k$ | Std of inner product | $\sqrt{d_k}$ | Largest weight (unscaled) | Largest weight (divided by $\sqrt{d_k}$) | Largest weight (divided by $d_k$) |
|---|---|---|---|---|---|
| $4$ | $2.004$ | $2.000$ | $0.5235$ | $0.3420$ | $0.2268$ |
| $16$ | $4.032$ | $4.000$ | $0.7499$ | $0.3577$ | $0.1743$ |
| $64$ | $8.009$ | $8.000$ | $0.8734$ | $0.3559$ | $0.1483$ |
| $256$ | $15.868$ | $16.000$ | $0.9373$ | $0.3595$ | $0.1364$ |
| $512$ | $22.775$ | $22.627$ | $0.9487$ | $0.3584$ | $0.1330$ |

![兩張圖，橫軸都是維度 d_k（4、16、64、256、512，對數刻度）。左圖紅線是模擬的內積標準差，和灰色虛線根號 d_k 幾乎重疊，從 2 升到約 22.8。右圖紅線是不縮放時 8 個鍵的最大權重平均，從 0.52 升到 0.95；藍線是除以根號 d_k 後，一直維持在約 0.34 到 0.36；灰色點線是平均分配 1/8（Left: the simulated standard deviation of inner products follows sqrt(d_k). Right: without scaling, the largest of 8 softmax weights climbs from 0.52 to 0.95 as d_k grows; with scaling it stays near 0.35）](../assets/figures/lec10-kp05-scaling.png)

**圖說 `lec10-kp05-scaling`**〔功能：精準度〕：先看左圖：紅色的模擬標準差和灰色虛線 $\sqrt{d_k}$ 幾乎重合，證實內積會隨維度長成 $\sqrt{d_k}$ 那麼大。再看右圖：不縮放時（紅），維度越高，最大權重越接近 $1$，softmax 幾乎只選一個；除以 $\sqrt{d_k}$ 後（藍），不論維度多大，最大權重都穩定在約 $0.35$。

**Caption `lec10-kp05-scaling`** [Function: precision]: First look at the left panel: the red simulated standard deviation almost coincides with the grey dashed line $\sqrt{d_k}$, confirming that inner products grow as large as $\sqrt{d_k}$. Then look at the right panel: without scaling (red), the higher the dimension, the closer the largest weight gets to $1$, and softmax nearly picks one; after dividing by $\sqrt{d_k}$ (blue), the largest weight stays near $0.35$ whatever the dimension.

常見錯誤：以為除以 $d_k$ 也行，甚至更穩。除以 $d_k$ 會把變異數壓成 $1/d_k$，維度一高，分數全部擠在 $0$ 附近，權重接近平均分配，模型反而分不出誰重要；上表最右一欄的最大權重從 $0.2268$ 一路降到 $0.1330$，逼近平均分配的 $1/8=0.125$。除以 $\sqrt{d_k}$ 剛好讓變異數回到 $1$。

Common error: thinking that dividing by $d_k$ works too, or is even safer. Dividing by $d_k$ shrinks the variance to $1/d_k$; in high dimensions all scores crowd near $0$, the weights become nearly uniform, and the model can no longer tell what matters; in the last column of the table above the largest weight falls from $0.2268$ to $0.1330$, close to the uniform $1/8=0.125$. Dividing by $\sqrt{d_k}$ brings the variance back to exactly $1$.

**想深入**（選讀）：把內積展開，$\mathbf{q}_i^\top\mathbf{k}_j=\mathbf{x}_i^\top\mathbf{W}_Q\mathbf{W}_K^\top\mathbf{x}_j$。如果 $\mathbf{W}_Q=\mathbf{W}_K=\mathbf{W}$，就變成 $(\mathbf{W}^\top\mathbf{x}_i)^\top(\mathbf{W}^\top\mathbf{x}_j)$：先把兩個詞都轉換到新空間，再算內積。這和支持向量機裡的「核函數」是同一種想法，第11週以後介紹的線性注意力就從這個角度出發。

**Going deeper** (optional): expanding the inner product gives $\mathbf{q}_i^\top\mathbf{k}_j=\mathbf{x}_i^\top\mathbf{W}_Q\mathbf{W}_K^\top\mathbf{x}_j$. If $\mathbf{W}_Q=\mathbf{W}_K=\mathbf{W}$, it becomes $(\mathbf{W}^\top\mathbf{x}_i)^\top(\mathbf{W}^\top\mathbf{x}_j)$: first map both words into a new space, then take the inner product. This is the same idea as the "kernel function" of support vector machines, and the linear attention introduced after Week 11 starts from this viewpoint.

### 本單元重點整理（KP05 summary）

**一句話重點**：自注意力用三個可學習矩陣把同一句話的每個詞轉成查詢、鍵與值，算 $\mathbf{Z}=\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$，讓每個詞用整句的相關詞重新描述自己；除以 $\sqrt{d_k}$ 是因為內積的大小約是 $\sqrt{d_k}$，不除的話 softmax 太尖、梯度消失。

**One-sentence summary**: self-attention uses three learnable matrices to turn every word of the same sentence into a query, a key and a value, computes $\mathbf{Z}=\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$, and lets every word redescribe itself with the related words of the whole sentence; it divides by $\sqrt{d_k}$ because inner products are about $\sqrt{d_k}$ in size, and without it softmax becomes too sharp and gradients vanish.

**白話說明**：開會時，每個人輪流問一句「誰說的跟我有關？」，再把相關的人的發言依關聯程度整理進自己的筆記。人人都這樣做一遍，每個人的筆記都帶有整場會議的脈絡。

**Plain explanation**: in a meeting, everyone in turn asks "whose words relate to mine?", then folds the related remarks into their own notes in proportion to how related they are. After everyone has done this, each person's notes carry the context of the whole meeting.

**正式定義**：$\mathbf{Q}=\mathbf{X}\mathbf{W}_Q$，$\mathbf{K}=\mathbf{X}\mathbf{W}_K$，$\mathbf{V}=\mathbf{X}\mathbf{W}_V$，$\mathbf{Z}=\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$，softmax 逐列做；$\mathbf{Q}\mathbf{K}^\top$ 是 $n\times n$，$\mathbf{Z}$ 是 $n\times d_v$。

**Formal definition**: $\mathbf{Q}=\mathbf{X}\mathbf{W}_Q$, $\mathbf{K}=\mathbf{X}\mathbf{W}_K$, $\mathbf{V}=\mathbf{X}\mathbf{W}_V$, $\mathbf{Z}=\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$, with softmax applied row by row; $\mathbf{Q}\mathbf{K}^\top$ is $n\times n$ and $\mathbf{Z}$ is $n\times d_v$.

**例子**：三個詞 $[1,0]^\top,[0,1]^\top,[1,1]^\top$、三個矩陣都取 $\mathbf{I}$：$\mathbf{A}$ 的三列是 $[0.4011,0.1978,0.4011]$、$[0.1978,0.4011,0.4011]$、$[0.2483,0.2483,0.5035]$，$\mathbf{Z}$ 的三列是 $[0.8022,0.5989]$、$[0.5989,0.8022]$、$[0.7517,0.7517]$。$d_k=64$、分數 $16,0$：不縮放得 $[0.9999999,0.0000001]$，縮放得 $[0.8808,0.1192]$。

**Example**: three words $[1,0]^\top,[0,1]^\top,[1,1]^\top$ with all three matrices equal to $\mathbf{I}$: the rows of $\mathbf{A}$ are $[0.4011,0.1978,0.4011]$, $[0.1978,0.4011,0.4011]$, $[0.2483,0.2483,0.5035]$, and the rows of $\mathbf{Z}$ are $[0.8022,0.5989]$, $[0.5989,0.8022]$, $[0.7517,0.7517]$. With $d_k=64$ and scores $16,0$: unscaled gives $[0.9999999,0.0000001]$, scaled gives $[0.8808,0.1192]$.

**常見迷思**：

- 「自注意力是每個詞只注意自己。」錯，每個詞注意整句，「自」指查詢、鍵、值都來自同一句。
- 「自注意力沒有參數。」錯，$\mathbf{W}_Q$、$\mathbf{W}_K$、$\mathbf{W}_V$ 都要學。
- 「除以 $d_k$ 比除以 $\sqrt{d_k}$ 更穩。」錯，除以 $d_k$ 會讓權重趨近平均，分不出重點。

**Common misconceptions**:

- "In self-attention every word attends only to itself." Wrong: every word attends to the whole sentence; "self" means the queries, keys and values all come from the same sentence.
- "Self-attention has no parameters." Wrong: $\mathbf{W}_Q$, $\mathbf{W}_K$, $\mathbf{W}_V$ must be learned.
- "Dividing by $d_k$ is safer than dividing by $\sqrt{d_k}$." Wrong: dividing by $d_k$ makes the weights nearly uniform, and nothing stands out.

**圖**：本單元有兩張圖。

- `lec10-kp05-three-word-attention`：三個詞的權重矩陣，以及每個詞移到它的新表示。
- `lec10-kp05-scaling`：內積隨維度長成 $\sqrt{d_k}$，不縮放時 softmax 越來越尖。

**Figures** — this unit has two figures.

- `lec10-kp05-three-word-attention`: the weight matrix of three words, and each word moving to its new representation.
- `lec10-kp05-scaling`: inner products grow like $\sqrt{d_k}$, and without scaling softmax gets ever sharper.

### 小練習（Practice）

1. 一句話有 $n=10$ 個詞，$d_k=64$、$d_v=32$。寫出 $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$、$\mathbf{Q}\mathbf{K}^\top$、$\mathbf{A}$、$\mathbf{Z}$ 的形狀。
2. 兩個詞，$\mathbf{Q}=\mathbf{K}=\begin{bmatrix}2&0\\0&2\end{bmatrix}$，$d_k=2$。算出 $\mathbf{Q}\mathbf{K}^\top$ 與權重矩陣 $\mathbf{A}$。提示：$4/\sqrt{2}=2.8284$，$\exp(2.8284)=16.919$。
3. $d_k=256$ 時，內積的典型大小約是多少？應該除以多少？

1. A sentence has $n=10$ words, $d_k=64$ and $d_v=32$. Write the shapes of $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$, $\mathbf{Q}\mathbf{K}^\top$, $\mathbf{A}$ and $\mathbf{Z}$.
2. Two words with $\mathbf{Q}=\mathbf{K}=\begin{bmatrix}2&0\\0&2\end{bmatrix}$ and $d_k=2$. Compute $\mathbf{Q}\mathbf{K}^\top$ and the weight matrix $\mathbf{A}$. Hint: $4/\sqrt{2}=2.8284$ and $\exp(2.8284)=16.919$.
3. With $d_k=256$, about how large is a typical inner product? What should you divide by?

**第 1 題解答** `[補充]`：$\mathbf{Q}$、$\mathbf{K}$ 是 $10\times64$，$\mathbf{V}$ 是 $10\times32$，$\mathbf{Q}\mathbf{K}^\top$ 和 $\mathbf{A}$ 是 $10\times10$，$\mathbf{Z}$ 是 $10\times32$。

**Solution to question 1** `[Added]`: $\mathbf{Q}$ and $\mathbf{K}$ are $10\times64$, $\mathbf{V}$ is $10\times32$, $\mathbf{Q}\mathbf{K}^\top$ and $\mathbf{A}$ are $10\times10$, and $\mathbf{Z}$ is $10\times32$.

**第 2 題解答** `[補充]`：$\mathbf{Q}\mathbf{K}^\top=\begin{bmatrix}4&0\\0&4\end{bmatrix}$，除以 $\sqrt{2}$ 得 $\begin{bmatrix}2.8284&0\\0&2.8284\end{bmatrix}$。每列 softmax：$16.919/(16.919+1)=0.9442$，所以 $\mathbf{A}=\begin{bmatrix}0.9442&0.0558\\0.0558&0.9442\end{bmatrix}$，兩個詞都主要注意自己。

**Solution to question 2** `[Added]`: $\mathbf{Q}\mathbf{K}^\top=\begin{bmatrix}4&0\\0&4\end{bmatrix}$; dividing by $\sqrt{2}$ gives $\begin{bmatrix}2.8284&0\\0&2.8284\end{bmatrix}$. Softmax of each row: $16.919/(16.919+1)=0.9442$, so $\mathbf{A}=\begin{bmatrix}0.9442&0.0558\\0.0558&0.9442\end{bmatrix}$, and both words attend mainly to themselves.

**第 3 題解答** `[補充]`：標準差約 $\sqrt{256}=16$（第4節的模擬是 $15.868$），所以分數要除以 $16$。

**Solution to question 3** `[Added]`: the standard deviation is about $\sqrt{256}=16$ (the simulation of Section 4 gives $15.868$), so the scores should be divided by $16$.

### 小測驗（Quiz，來源原題）

來源的兩題自我檢測保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。來源把詞向量排成**行**，所以它寫 $\mathbf{Q}^\top\mathbf{K}$；本講義把詞向量排成**列**，同一個矩陣寫成 $\mathbf{Q}\mathbf{K}^\top$。

The two self-check questions of the source are kept with all options; the source page does not mark the answers directly, and the explanations below follow the source text. The source arranges word vectors as **columns** and so writes $\mathbf{Q}^\top\mathbf{K}$; this note arranges them as **rows**, and the same matrix is written $\mathbf{Q}\mathbf{K}^\top$.

1. 自注意力矩陣運算式中，$\mathbf{Q}^\top\mathbf{K}$ 矩陣的維度是多少（設序列長度為 $n$）？
   - (A) $n\times n$
   - (B) $p\times p$
   - (C) $d\times r$
   - (D) $1\times n$
2. 為什麼稱此機制為「自」注意力（Self-Attention）？
   - (A) 因為模型完全不需要任何訓練
   - (B) 因為 $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$ 皆來自同一條輸入序列 $\mathbf{X}$ 的不同線性投影
   - (C) 因為它只能處理單一數字
   - (D) 因為只能關注自身無法關注其他詞

1. In the matrix form of self-attention, what is the size of the matrix $\mathbf{Q}^\top\mathbf{K}$ (for a sequence of length $n$)?
   - (A) $n\times n$
   - (B) $p\times p$
   - (C) $d\times r$
   - (D) $1\times n$
2. Why is this mechanism called "self"-attention?
   - (A) Because the model needs no training at all
   - (B) Because $\mathbf{Q}$, $\mathbf{K}$ and $\mathbf{V}$ all come from different linear projections of the same input sequence $\mathbf{X}$
   - (C) Because it can only handle a single number
   - (D) Because it can only attend to itself and not to other words

**解析** `[補充]`：第 1 題選 (A)：每個詞對每個詞各有一個分數，所以是 $n\times n$；第3節例子一的 $3\times3$ 就是這樣。第 2 題選 (B)：這是第1節常見錯誤的重點；(D) 正是那個錯誤說法。

**Explanations** `[Added]`: question 1 is A: there is one score for every pair of words, hence $n\times n$, as in the $3\times3$ of example one in Section 3. Question 2 is B: this is the point of the common error in Section 1, and D is exactly that wrong statement.

### 留給下一單元的問題（The question for the next unit）

到這裡，注意力都在處理文字。查詢、鍵、值只要是向量就行，那影像的每個區塊、語音的每一小段，能不能也用同樣的方法互相注意？下一單元（KP06）看注意力在影像與語音上的用法。

So far attention has been handling text. Queries, keys and values only need to be vectors; can the regions of an image or the short pieces of speech attend to each other in the same way? The next unit (KP06) looks at attention in images and speech.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第9章「注意力機制與 Transformer 核心原理」§2.4.1、§2.4.3 與教授祕笈「縮放點積注意力與方差穩定」「Query-Key-Value 檢索機制」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §2.4.1 and §2.4.3 of Chapter 9, "Core principles of attention mechanisms and the Transformer", of CCC-LNS IDL_115 and its professor's notes "Scaled dot-product attention and variance stabilization" and "The Query-Key-Value retrieval mechanism" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 方向更正：來源 §2.4.3 把詞向量排成行（$\mathbf{X}\in\mathbb{R}^{d\times n}$），寫 $\mathbf{Z}=\mathbf{V}\operatorname{softmax}(\mathbf{Q}^\top\mathbf{K}/\sqrt{p})$，並說 softmax「沿每一列」做。在這個記號下，$\mathbf{Z}$ 的第 $j$ 行是 $\sum_i\mathbf{v}_i\,[\cdot]_{ij}$，要讓它成為值的加權平均，必須是每一**行**加總為 $1$。本講義改用來源教授祕笈的「列為詞」記號 $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$，softmax 逐列做，意思明確。
- 來源教授祕笈的例子寫輸出 $[[6.70,6.60],[3.30,13.40]]$；本講義用小數第四位 $[[6.6976,6.6048],[3.3024,13.3952]]$。來源的維度記號 $p$ 與 $r$，本講義依祕笈改用 $d_k$ 與 $d_v$。
- 來源用粗斜體 `\boldsymbol` 表示向量；本課程網站不支援，改用 `\mathbf`，意義相同。小測驗第 1 題保留來源的 $\mathbf{Q}^\top\mathbf{K}$ 與選項中的 $p$、$d$、$r$。
- 本講義補上：三個詞的例子與圖、維度模擬的表與圖、三題小練習與解答。
- 取捨：來源提到核函數與 Performer 的連結，只放在「想深入」；統計物理的溫度比喻、混合精度下遮罩數值的細節不寫。

- Direction correction: source §2.4.3 arranges word vectors as columns ($\mathbf{X}\in\mathbb{R}^{d\times n}$), writes $\mathbf{Z}=\mathbf{V}\operatorname{softmax}(\mathbf{Q}^\top\mathbf{K}/\sqrt{p})$, and says that softmax runs "along each row". With this notation, column $j$ of $\mathbf{Z}$ is $\sum_i\mathbf{v}_i\,[\cdot]_{ij}$, and for it to be a weighted average of the values, each **column** must add up to $1$. This note uses the "words as rows" notation of the professor's notes, $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$, with softmax row by row, which is unambiguous.
- The professor's notes give the output $[[6.70,6.60],[3.30,13.40]]$; this note uses four decimals, $[[6.6976,6.6048],[3.3024,13.3952]]$. Following the professor's notes, the source's dimension symbols $p$ and $r$ are written $d_k$ and $d_v$ here.
- The source writes vectors with bold italic `\boldsymbol`; the course website does not support it, so this note uses `\mathbf` with the same meaning. Quiz question 1 keeps the source's $\mathbf{Q}^\top\mathbf{K}$ and the $p$, $d$, $r$ in its options.
- This note adds: the three-word example and its figure, the table and figure of the dimension simulation, and three practice questions with solutions.
- Choices: the source's link to kernels and the Performer appears only under "Going deeper"; the temperature analogy from statistical physics and the details of mask values in mixed precision are left out.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 自注意力 | self-attention |
| 詞嵌入 | word embedding |
| 複合嵌入 | composite embedding |
| 注意力權重矩陣 | attention weight matrix |
| 縮放點積注意力 | scaled dot-product attention |
| 變異數 | variance |
| 標準差 | standard deviation |
| 核函數 | kernel function |
| 支持向量機 | support vector machine (SVM) |
| 線性注意力 | linear attention |
| 聯邦調查局（例句） | Federal Bureau of Investigation (FBI) |
| 兩種早期的詞嵌入方法 | Word2Vec, GloVe (Global Vectors) |
