# 第10週 KP06：注意力不只用在文字，影像與語音也能「看重點」（Attention is not only for text: images and speech can also "focus")

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第10週的最後一個知識點（KP06），接在 KP05 之後。前五個知識點都用文字當例子；可是注意力只要求查詢、鍵、值是向量，並不在乎向量從哪裡來。這一節看影像與語音怎麼變成一組可以被注意的向量，以及影像特有的兩種注意力：空間注意力與通道注意力。

This is the last knowledge point (KP06) of Week 10 and follows KP05. The first five knowledge points used text as the example; yet attention only needs the queries, keys and values to be vectors and does not care where the vectors come from. This note shows how images and speech become a set of vectors that can be attended to, and two kinds of attention specific to images: spatial attention and channel attention.

先備知識是第6、7週的卷積神經網路（特徵圖、通道），KP03 的帶注意力 Seq2Seq，以及第2週的 sigmoid 函數。本講義依課程網頁教材 CCC-LNS 第9章改寫，文中「來源」就是指它；本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec10-kp06-figures.py` 算出，可以自己重算。

The prerequisites are the convolutional networks of Weeks 6 and 7 (feature maps, channels), the Seq2Seq with attention of KP03, and the sigmoid function of Week 2. This note is rewritten from Chapter 9 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec10-kp06-figures.py`, so you can recompute it yourself.

## KP06 注意力不只用在文字，影像與語音也能「看重點」（Attention is not only for text: images and speech can also "focus"）

### 學習目標（Learning objectives）

讀完之後，你能說明圖片描述模型怎麼把 CNN 特徵圖的每個位置當成 KP03 的 $\mathbf{h}_j$，讓解碼器寫每個詞時注意圖上不同的區域。你能說明語音辨識怎麼把聲音的每一小段當成 $\mathbf{h}_j$。你能區分空間注意力（看「哪裡」）和通道注意力（看「什麼」），並手算一次 SENet 的壓縮、激發、縮放三步。你也能說明為什麼通道注意力用 sigmoid，權重不必加總為 $1$。

After reading, you can explain how an image-captioning model treats every position of a CNN feature map as the $\mathbf{h}_j$ of KP03, so that the decoder attends to different regions of the image for every word. You can explain how speech recognition treats every short piece of sound as an $\mathbf{h}_j$. You can tell spatial attention ("where") from channel attention ("what"), and compute the squeeze, excitation and scale steps of SENet by hand. You can also explain why channel attention uses sigmoid and why its weights need not add up to $1$.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{X}\in\mathbb{R}^{C\times H\times W}$ | CNN 的特徵圖：$C$ 個通道，每個 $H\times W$ | feature map of a CNN: $C$ channels of size $H\times W$ |
| $\mathbf{h}_j$ | 特徵圖上第 $j$ 個位置（或第 $j$ 段聲音）的向量 | vector at position $j$ of the feature map (or of sound segment $j$) |
| $n'$ | 被注意的位置個數，例如 $H\times W$ | number of attended positions, for example $H\times W$ |
| $a_{ij},\ \mathbf{c}_i$ | KP03 的注意力權重與上下文向量：寫第 $i$ 個詞時給位置 $j$ 的權重，以及加權後的向量 | the attention weights and context vector of KP03: the weight given to position $j$ when writing word $i$, and the weighted vector |
| $X_c(i,j)$ | 第 $c$ 個通道在位置 $(i,j)$ 的值 | value of channel $c$ at position $(i,j)$ |
| $z_c$ | 第 $c$ 個通道的平均值（壓縮的結果） | mean of channel $c$ (result of the squeeze) |
| $s_c$ | 第 $c$ 個通道的閘門值，介於 $0$ 與 $1$ | gate of channel $c$, between $0$ and $1$ |
| $\sigma(\cdot)$ | sigmoid 函數 $1/(1+e^{-x})$ | sigmoid function $1/(1+e^{-x})$ |
| $\mathbf{W}_1,\ \mathbf{W}_2,\ r$ | SENet 兩層全連接的權重與縮減比例 | weights and reduction ratio of SENet's two fully connected layers |

### 1. 看圖說話：讓解碼器注意圖上的區域（Image captioning: let the decoder attend to regions of the image）

**圖片描述**（image captioning）的任務是看一張圖、寫一句話描述它，例如「大象在水裡」。來源介紹的 Show, Attend and Tell 模型，結構就是 KP03 的帶注意力 Seq2Seq，只是編碼器換成 CNN。

**Image captioning** means looking at an image and writing a sentence about it, for example "an elephant is in the water". The Show, Attend and Tell model described by the source has the structure of the Seq2Seq with attention of KP03, except that the encoder is a CNN.

CNN 的最後一層特徵圖有 $H\times W$ 個位置，每個位置有 $C$ 個通道的數值，合起來就是一個 $C$ 維向量，描述圖上那一小塊。把這 $n'=H\times W$ 個向量當成 KP03 的 $\mathbf{h}_1,\dots,\mathbf{h}_{n'}$，解碼器（LSTM）寫每個詞之前，拿自己的狀態當查詢，對這些位置做注意力。

The last feature map of a CNN has $H\times W$ positions, and each position has values in $C$ channels, which together form a $C$-dimensional vector describing that small patch of the image. Treat these $n'=H\times W$ vectors as the $\mathbf{h}_1,\dots,\mathbf{h}_{n'}$ of KP03; before writing each word, the decoder (an LSTM) uses its state as the query and attends over these positions.

寫 `elephant` 時，權重集中在大象所在的區塊；寫 `water` 時，權重移到下方的水面。來源的模型用 $14\times14$ 的特徵圖時，每寫一個詞要對 $196$ 個位置算分數，寫一句 $10$ 個詞就是 $1960$ 個分數 `[補充]`。

When writing "elephant", the weight concentrates on the patch where the elephant is; when writing "water", it moves to the water at the bottom. With the $14\times14$ feature map used by the source's model, every word needs scores over $196$ positions, and a $10$-word sentence needs $1960$ scores `[Added]`.

![兩張 7 乘 7 的熱圖，暗色是權重小，亮色是權重大。座標從 0 起算。左圖寫 elephant 時，最亮的格子在第 2 列第 3 行，權重 0.3331，上下左右四格次亮、斜角四格更暗；右圖寫 water 時，亮處移到下方，第 5、6 列第 3 行兩格並列最亮，最大權重 0.2218（Two 7-by-7 attention maps with coordinates counted from 0: writing "elephant" concentrates the weight in the cell at row 2, column 3, max 0.3331, with its four side neighbours next brightest; writing "water" moves it to two equally bright cells at rows 5 and 6, column 3, max 0.2218）](../assets/figures/lec10-kp06-spatial-attention.png)

**圖說 `lec10-kp06-spatial-attention`**〔功能：一目了然、空間幾何具象化〕：這是示意例子 `[補充]`，用 $7\times7=49$ 個位置代替真實的 $14\times14$。先看左圖：寫 `elephant` 時，最亮的一格拿到 $0.3331$，加上周圍幾格，最亮的 $9$ 格合計 $0.7155$；平均分配的話每格只有 $0.0204$。再看右圖：寫 `water` 時，亮處移到下方，第 $5$、$6$ 列中間兩格並列最亮。兩張圖的 $49$ 個權重各自加總為 $1$。

**Caption `lec10-kp06-spatial-attention`** [Function: seeing it at a glance, picturing the geometry]: this is an illustrative example `[Added]` that uses $7\times7=49$ positions instead of a real $14\times14$. First look at the left panel: when writing "elephant", the brightest cell gets $0.3331$, and together with its neighbours the $9$ brightest cells add up to $0.7155$; a uniform split would give each cell only $0.0204$. Then look at the right panel: when writing "water", the bright spot moves to the bottom, with the two middle cells of rows $5$ and $6$ equally bright. In each panel the $49$ weights add up to $1$.

來源也提到**視覺問答**（visual question answering）：問題本身當查詢，去注意圖上和問題最相關的區域，再回答。結構相同，只是查詢從「寫到哪個詞」換成「問的是什麼」。

The source also mentions **visual question answering**: the question itself serves as the query, attends to the regions of the image most related to it, and then the model answers. The structure is the same; only the query changes from "which word is being written" to "what is being asked".

常見錯誤：以為注意力圖上最亮的地方，就是模型「認出」物體的證據。它只說明 $\mathbf{c}_i$ 是從哪些位置混出來的；KP03 說過，權重高不等於因果貢獻大。

Common error: thinking that the brightest spot of an attention map is proof that the model "recognized" the object. It only says from which positions $\mathbf{c}_i$ was mixed; as KP03 noted, a high weight is not the same as a large causal contribution.

### 2. 聽寫：讓字對齊到聲音的時間（Speech recognition: align letters to moments in the sound）

語音辨識的 Listen, Attend and Spell 模型做的是同一件事。編碼器把聲音切成很多短片段（例如每 $10$ 毫秒一段），每段變成一個向量 $\mathbf{h}_j$；解碼器一個字母一個字母拼出文字，拼每個字母前，對所有片段做注意力。

The Listen, Attend and Spell model of speech recognition does the same thing. The encoder cuts the sound into many short segments (for example one every $10$ milliseconds) and turns each into a vector $\mathbf{h}_j$; the decoder spells the text letter by letter, attending over all segments before each letter.

一段 $3$ 秒的錄音，以每段 $10$ 毫秒計是 $300$ 個片段；拼出 $20$ 個字母，就要算 $300\times20=6000$ 個分數 `[補充]`。訓練好以後，權重會沿著時間往前移：拼前面的字母時看前面的聲音，拼後面的字母時看後面的聲音，就像 KP03 對齊矩陣的對角線。

A $3$-second recording with one segment per $10$ milliseconds has $300$ segments; spelling $20$ letters needs $300\times20=6000$ scores `[Added]`. After training, the weights move forward in time: early letters look at early sound and later letters at later sound, like the diagonal of the alignment matrix in KP03.

常見錯誤：以為語音的注意力要事先告訴模型每個字母在哪一段聲音。不需要；對齊是模型在訓練中自己學出來的，正如翻譯模型自己學出詞的對應。

Common error: thinking that speech attention needs to be told in advance which sound segment each letter belongs to. It does not; the alignment is learned by the model during training, just as the translation model learns the word correspondence itself.

### 3. 空間注意力與通道注意力（Spatial attention and channel attention）

在 CNN 裡，注意力可以沿兩個方向加權。**空間注意力**（spatial attention）對 $H\times W$ 個位置各給一個權重，回答「在哪裡」。第1節的權重也是對位置加權，但那裡的查詢來自解碼器；CNN 內部的空間注意力則由特徵圖本身算出每個位置的權重 `[補充]`。**通道注意力**（channel attention）對 $C$ 個通道各給一個權重，回答「是什麼」：例如加強負責毛皮紋理的通道，壓低只管背景邊緣的通道。

In a CNN, attention can weight along two directions. **Spatial attention** gives each of the $H\times W$ positions a weight and answers "where". The weights of Section 1 also weight positions, but there the query comes from the decoder; spatial attention inside a CNN computes the weight of each position from the feature map itself `[Added]`. **Channel attention** gives each of the $C$ channels a weight and answers "what": for example, it strengthens the channels that detect fur texture and weakens those that only detect background edges.

通道注意力的代表是 **SENet**（Squeeze-and-Excitation Network），分三步：

The representative of channel attention is **SENet** (Squeeze-and-Excitation Network), in three steps:

1. **壓縮**（squeeze）：把每個通道的整張圖平均成一個數，$z_c=\frac{1}{H\times W}\sum_{i=1}^{H}\sum_{j=1}^{W}X_c(i,j)$。
2. **激發**（excitation）：用兩層全連接網路和 sigmoid 算出每個通道的閘門，$\mathbf{s}=\sigma(\mathbf{W}_2\operatorname{ReLU}(\mathbf{W}_1\mathbf{z}))$。
3. **縮放**（scale）：每個通道乘上自己的閘門，$\tilde{X}_c=s_c\,X_c$。

1. **Squeeze**: average each channel's whole map into one number, $z_c=\frac{1}{H\times W}\sum_{i=1}^{H}\sum_{j=1}^{W}X_c(i,j)$.
2. **Excitation**: compute a gate for each channel with two fully connected layers and sigmoid, $\mathbf{s}=\sigma(\mathbf{W}_2\operatorname{ReLU}(\mathbf{W}_1\mathbf{z}))$.
3. **Scale**: multiply each channel by its own gate, $\tilde{X}_c=s_c\,X_c$.

$\mathbf{W}_1$ 把 $C$ 個數縮成 $C/r$ 個，$\mathbf{W}_2$ 再放回 $C$ 個；縮減比例 $r$ 讓參數很少。例如 $C=256$、$r=16$ 時，兩層共 $2\times256\times16=8192$ 個權重，而同一層一個 $3\times3$ 卷積有 $256\times256\times9=589824$ 個 `[補充]`。來源提到，SENet 只多了不到 $1\%$ 的參數，就拿下 ImageNet 冠軍。

$\mathbf{W}_1$ shrinks the $C$ numbers to $C/r$ and $\mathbf{W}_2$ expands them back to $C$; the reduction ratio $r$ keeps the parameters few. For example, with $C=256$ and $r=16$ the two layers have $2\times256\times16=8192$ weights, while one $3\times3$ convolution of the same layer has $256\times256\times9=589824$ `[Added]`. The source notes that SENet won the ImageNet competition with less than $1\%$ extra parameters.

來源的數字例：$C=2$，第 $1$ 個通道的 $2\times2$ 特徵圖全是 $2$，第 $2$ 個全是 $0$，所以 $z_1=2.0$、$z_2=0.0$。假設激發網路算出、進 sigmoid 之前的分數 $\mathbf{W}_2\operatorname{ReLU}(\mathbf{W}_1\mathbf{z})$ 是 $2.0$ 和 $-2.0$（這裡省略 $\mathbf{W}_1$、$\mathbf{W}_2$ 的計算，直接假設），sigmoid 後閘門是 $s_1=\sigma(2)=0.8808$、$s_2=\sigma(-2)=0.1192$。縮放後，第 $1$ 個通道每格變成 $0.8808\times2=1.7616$；第 $2$ 個通道原本是 $0$，乘上 $0.1192$ 仍是 $0$，它的閘門很小，表示若有數值也會被壓低。

The source's numerical example: $C=2$; the $2\times2$ map of channel $1$ is all $2$ and that of channel $2$ all $0$, so $z_1=2.0$ and $z_2=0.0$. Suppose the excitation network's scores before sigmoid, $\mathbf{W}_2\operatorname{ReLU}(\mathbf{W}_1\mathbf{z})$, are $2.0$ and $-2.0$ (the computation with $\mathbf{W}_1$ and $\mathbf{W}_2$ is skipped and simply assumed); after sigmoid the gates are $s_1=\sigma(2)=0.8808$ and $s_2=\sigma(-2)=0.1192$. After scaling, every cell of channel $1$ becomes $0.8808\times2=1.7616$; channel $2$ was $0$ and stays $0$ after multiplying by $0.1192$, and its small gate means that any values it had would be pushed down.

四個通道的例子 `[補充]`：平均值 $z=[2.0,\ 0.5,\ 1.0,\ 0.0]$，進 sigmoid 之前的激發分數同樣直接假設為 $[2.0,\ -1.0,\ 0.5,\ -2.0]$，閘門是 $[0.8808,\ 0.2689,\ 0.6225,\ 0.1192]$，四個閘門加起來是 $1.8914$，不是 $1$。因為 $\tilde{X}_c=s_cX_c$ 是整張圖乘同一個數，縮放後的通道平均就是 $s_cz_c$：$[1.7616,\ 0.1345,\ 0.6225,\ 0]$。

A four-channel example `[Added]`: the means are $z=[2.0,\ 0.5,\ 1.0,\ 0.0]$, the excitation scores before sigmoid are again simply assumed to be $[2.0,\ -1.0,\ 0.5,\ -2.0]$, and the gates are $[0.8808,\ 0.2689,\ 0.6225,\ 0.1192]$; the four gates add up to $1.8914$, not $1$. Since $\tilde{X}_c=s_cX_c$ multiplies the whole map by one number, the mean of each scaled channel is $s_cz_c$: $[1.7616,\ 0.1345,\ 0.6225,\ 0]$.

![三張長條圖，橫軸都是四個通道。左圖灰色是每個通道的平均值 2.00、0.50、1.00、0.00；中圖橘色是 sigmoid 閘門 0.8808、0.2689、0.6225、0.1192，上方有一條 1 的點線；右圖藍色是閘門乘平均值 1.7616、0.1345、0.6225、0.0000（Squeeze gives channel means 2, 0.5, 1, 0; excitation gives sigmoid gates 0.8808, 0.2689, 0.6225, 0.1192, each below 1 and not summing to 1; scaling gives 1.7616, 0.1345, 0.6225, 0）](../assets/figures/lec10-kp06-channel-attention.png)

**圖說 `lec10-kp06-channel-attention`**〔功能：一目了然、精準度〕：先看左圖：壓縮把每個通道變成一個平均值。再看中圖：每個通道各自通過 sigmoid，閘門都在 $0$ 與 $1$ 之間，但彼此獨立，加總是 $1.8914$。最後看右圖：乘上閘門以後，第 $1$、$3$ 個通道保留大部分，第 $2$、$4$ 個通道被壓低。

**Caption `lec10-kp06-channel-attention`** [Function: seeing it at a glance, precision]: First look at the left panel: the squeeze turns each channel into one mean. Then look at the middle panel: each channel passes through sigmoid on its own, every gate lies between $0$ and $1$, but they are independent and add up to $1.8914$. Finally look at the right panel: after multiplying by the gates, channels $1$ and $3$ keep most of their size, while channels $2$ and $4$ are pushed down.

為什麼通道注意力用 sigmoid，不用 softmax？softmax 讓各位置互相競爭，一個變大，其他就得變小，適合「從幾個候選裡挑重點」。通道則可能同時都有用：一張貓的照片，毛皮紋理和耳朵形狀兩個通道可能都該保留。sigmoid 讓每個通道各自決定開多少，不必搶同一份總量 `[補充]`。

Why does channel attention use sigmoid instead of softmax? Softmax makes the positions compete: when one grows, the others must shrink, which suits "picking the important ones among candidates". Channels, however, may all be useful at once: for a photo of a cat, both the fur-texture channel and the ear-shape channel may deserve to be kept. Sigmoid lets each channel decide on its own how far to open, without fighting over one shared total `[Added]`.

常見錯誤：把兩種注意力的形狀搞混。通道注意力的輸出是 $C\times1\times1$，每個通道一個數；空間注意力的輸出是 $1\times H\times W$，每個位置一個數。

Common error: mixing up the shapes of the two kinds. The output of channel attention is $C\times1\times1$, one number per channel; the output of spatial attention is $1\times H\times W$, one number per position.

**想深入**（選讀）：來源提到，**Vision Transformer**（ViT）更進一步，把圖片切成一塊一塊，每塊當成一個「詞」，直接用 KP05 的自注意力，讓每塊圖去注意其他塊。這讓影像和文字可以用同一種架構處理；第11週學完 Transformer 後再回頭看會更清楚。

**Going deeper** (optional): the source notes that the **Vision Transformer** (ViT) goes further: it cuts an image into patches, treats each patch as a "word", and applies the self-attention of KP05 directly, so that every patch attends to the others. This lets images and text be handled by the same architecture; it will be clearer after the Transformer of Week 11.

### 本單元重點整理（KP06 summary）

**一句話重點**：注意力只要求一組向量，所以 CNN 特徵圖的每個位置、聲音的每一小段都能當成 KP03 的 $\mathbf{h}_j$，讓解碼器寫字時注意圖上的區域或聲音的時間；在 CNN 內部，空間注意力對位置加權回答「哪裡」，通道注意力（SENet）以壓縮、激發、縮放三步對通道加權回答「什麼」，閘門用 sigmoid、不必加總為 $1$。

**One-sentence summary**: attention needs only a set of vectors, so every position of a CNN feature map and every short piece of sound can serve as the $\mathbf{h}_j$ of KP03, letting the decoder attend to regions of an image or moments of a sound while writing; inside a CNN, spatial attention weights the positions to answer "where", and channel attention (SENet) weights the channels in three steps, squeeze, excitation and scale, to answer "what", with sigmoid gates that need not add up to $1$.

**白話說明**：看圖說話時，說到「大象」就看大象，說到「水」就看水面；聽寫時，拼到哪個字母，就往後聽對應那一段時間的聲音。CNN 內部也可以調音量：重要的位置、重要的特徵通道開大聲一點。

**Plain explanation**: when describing a picture, you look at the elephant when you say "elephant" and at the water when you say "water"; when taking dictation, you move forward and listen to the stretch of sound that matches each letter. A CNN can also adjust its own volume inside: turn up the important positions and the important feature channels.

**正式定義**：圖片描述與語音辨識：$\mathbf{c}_i=\sum_{j=1}^{n'}a_{ij}\mathbf{h}_j$，$\mathbf{h}_j$ 是特徵圖位置或聲音片段。SENet：$z_c=\frac{1}{HW}\sum_{i,j}X_c(i,j)$，$\mathbf{s}=\sigma(\mathbf{W}_2\operatorname{ReLU}(\mathbf{W}_1\mathbf{z}))$，$\tilde{X}_c=s_cX_c$。

**Formal definition**: image captioning and speech recognition: $\mathbf{c}_i=\sum_{j=1}^{n'}a_{ij}\mathbf{h}_j$, where $\mathbf{h}_j$ is a feature-map position or a sound segment. SENet: $z_c=\frac{1}{HW}\sum_{i,j}X_c(i,j)$, $\mathbf{s}=\sigma(\mathbf{W}_2\operatorname{ReLU}(\mathbf{W}_1\mathbf{z}))$, $\tilde{X}_c=s_cX_c$.

**例子**：$14\times14$ 特徵圖每個詞 $196$ 個分數；$3$ 秒錄音、$20$ 個字母共 $6000$ 個分數。SENet：$z=[2,0]$、分數 $[2,-2]$，閘門 $0.8808$、$0.1192$，第 $1$ 個通道變成 $1.7616$。

**Example**: a $14\times14$ feature map gives $196$ scores per word; $3$ seconds of audio and $20$ letters give $6000$ scores. SENet: $z=[2,0]$ and scores $[2,-2]$ give gates $0.8808$ and $0.1192$, and channel $1$ becomes $1.7616$.

**常見迷思**：

- 「注意力圖最亮的地方就是模型認出物體的證據。」錯，權重只說明怎麼混合，不等於因果。
- 「語音注意力要事先標好每個字母對應哪段聲音。」錯，對齊是訓練中學出來的。
- 「通道注意力的權重要加總為 $1$。」錯，它用 sigmoid，各通道獨立。

**Common misconceptions**:

- "The brightest spot of an attention map proves the model recognized the object." Wrong: the weights only describe the mixing, not causation.
- "Speech attention needs each letter labelled with its sound segment in advance." Wrong: the alignment is learned in training.
- "Channel attention weights must add up to $1$." Wrong: it uses sigmoid, and the channels are independent.

**圖**：本單元有兩張圖。

- `lec10-kp06-spatial-attention`：寫兩個不同的詞時，$7\times7$ 位置上的注意力分布（示意）。
- `lec10-kp06-channel-attention`：SENet 的壓縮、激發、縮放三步。

**Figures** — this unit has two figures.

- `lec10-kp06-spatial-attention`: the attention over $7\times7$ positions while writing two different words (illustrative).
- `lec10-kp06-channel-attention`: the squeeze, excitation and scale steps of SENet.

### 小練習（Practice）

1. 圖片描述模型用 $7\times7$ 的特徵圖，寫一句 $8$ 個詞的描述。一共要算幾個注意力分數？每個詞的 $49$ 個權重加起來是多少？
2. SENet 有三個通道，激發網路的分數是 $0$、$3$、$-1$。算出三個閘門。提示：$e^{-3}=0.0498$，$e^{1}=2.718$。
3. 一張街景照片，汽車和行人兩種物體都很重要。用通道注意力時，為什麼 sigmoid 比 softmax 合適？

1. An image-captioning model uses a $7\times7$ feature map and writes an $8$-word description. How many attention scores does it compute in total? What do the $49$ weights of each word add up to?
2. A SENet has three channels, and the excitation network gives scores $0$, $3$ and $-1$. Compute the three gates. Hint: $e^{-3}=0.0498$ and $e^{1}=2.718$.
3. A street photo contains cars and pedestrians, both important. For channel attention, why is sigmoid more suitable than softmax?

**第 1 題解答** `[補充]`：$49\times8=392$ 個分數；每個詞的 $49$ 個權重是 softmax 的結果，加起來是 $1$。

**Solution to question 1** `[Added]`: $49\times8=392$ scores; the $49$ weights of each word come from softmax and add up to $1$.

**第 2 題解答** `[補充]`：$\sigma(0)=0.5$，$\sigma(3)=1/(1+0.0498)=0.9526$，$\sigma(-1)=1/(1+2.718)=0.2689$。

**Solution to question 2** `[Added]`: $\sigma(0)=0.5$, $\sigma(3)=1/(1+0.0498)=0.9526$, $\sigma(-1)=1/(1+2.718)=0.2689$.

**第 3 題解答** `[補充]`：softmax 會讓通道互相搶同一份總量，加強汽車的通道就得壓低行人的通道；sigmoid 讓每個通道各自開關，兩種通道可以同時保留。

**Solution to question 3** `[Added]`: softmax makes the channels fight over one shared total, so strengthening the car channels would weaken the pedestrian channels; sigmoid lets each channel open on its own, so both kinds can be kept at once.

### 小測驗（Quiz，來源原題）

來源的兩題自我檢測保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The two self-check questions of the source are kept with all options; the source page does not mark the answers directly, and the explanations below follow the source text.

1. 在「Show, Attend and Tell」圖像描述模型中，扮演編碼器提取視覺特徵的是哪種架構？
   - (A) 支援向量機（SVM）
   - (B) 卷積神經網路（CNN）
   - (C) 主成分分析（PCA）
   - (D) $K$-means 分群
2. 在多模態注意力中，CNN 提取的空間特徵對應到注意力機制中的什麼角色？
   - (A) 解碼器輸出單詞
   - (B) 可供查詢對齊的隱藏特徵向量集合 $\{\mathbf{h}_j\}$
   - (C) 全域固定偏置常數
   - (D) 損失函數正則項

1. In the "Show, Attend and Tell" image-captioning model, which architecture serves as the encoder that extracts visual features?
   - (A) Support vector machine (SVM)
   - (B) Convolutional neural network (CNN)
   - (C) Principal component analysis (PCA)
   - (D) $K$-means clustering
2. In multimodal attention, what role do the spatial features extracted by the CNN play in the attention mechanism?
   - (A) The words output by the decoder
   - (B) The set of hidden feature vectors $\{\mathbf{h}_j\}$ that the query aligns against
   - (C) A global fixed bias constant
   - (D) A regularization term of the loss

**解析** `[補充]`：第 1 題選 (B)：第1節說明編碼器是 CNN，最後一層特徵圖的每個位置提供一個向量。第 2 題選 (B)：這些位置向量就是 KP03 的 $\mathbf{h}_j$，被解碼器的查詢拿來比對與加權；(A) 是解碼器的輸出，不是被注意的對象。

**Explanations** `[Added]`: question 1 is B: Section 1 explains that the encoder is a CNN, and every position of its last feature map provides a vector. Question 2 is B: these position vectors are the $\mathbf{h}_j$ of KP03, compared and weighted by the decoder's query; A is the decoder's output, not what is attended to.

### 留給下一單元的問題（The question for the next unit）

本週從「平均會稀釋重點」走到自注意力，也看到它能用在影像與語音。但目前的模型多半還靠 RNN 或 CNN 先產生向量。如果整個模型只用自注意力，完全不用 RNN，詞的順序要怎麼表示？多個注意力可以同時看不同的關係嗎？第11週的 Transformer 回答這些問題。

This week went from "averaging dilutes what matters" to self-attention, and showed that it works for images and speech too. But most of these models still rely on an RNN or a CNN to produce the vectors first. If a whole model used only self-attention, with no RNN at all, how would it represent word order? Could several attentions look at different relations at once? The Transformer of Week 11 answers these questions.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第9章「注意力機制與 Transformer 核心原理」§2.5 與教授祕笈「視覺注意力機制與跨模態應用」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §2.5 of Chapter 9, "Core principles of attention mechanisms and the Transformer", of CCC-LNS IDL_115 and its professor's notes "Visual attention and cross-modal applications" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源寫 $\sigma(2)\approx0.88$、$\sigma(-2)\approx0.12$、縮放後 $1.76$；本講義用小數第四位 $0.8808$、$0.1192$、$1.7616$。
- 來源用粗斜體 `\boldsymbol` 表示向量；本課程網站不支援，改用 `\mathbf`，意義相同。小測驗第 1 題選項 (D) 的 `K-means` 改寫成 $K$-means，意思不變。
- 本講義補上：$7\times7$ 空間注意力的示意圖（權重由人為設定的分數算出，不是真實模型的輸出）、四個通道的 SENet 例子與圖、分數個數的算例、sigmoid 與 softmax 的比較、三題小練習與解答。
- 取捨：來源的 Non-Local 網路、降維比例 $r$ 的取捨、ViT 的歸納偏置討論與神經生理學連結，本講義只在「想深入」提到 ViT，其餘不寫。

- The source writes $\sigma(2)\approx0.88$, $\sigma(-2)\approx0.12$ and $1.76$ after scaling; this note uses four decimals, $0.8808$, $0.1192$ and $1.7616$.
- The source writes vectors with bold italic `\boldsymbol`; the course website does not support it, so this note uses `\mathbf` with the same meaning. Option D of quiz question 1, `K-means`, is written $K$-means with the same meaning.
- This note adds: the $7\times7$ spatial-attention illustration (the weights come from hand-set scores, not from a real model), the four-channel SENet example and its figure, the score-count examples, the comparison of sigmoid and softmax, and three practice questions with solutions.
- Choices: the source's Non-Local networks, the trade-off of the reduction ratio $r$, the discussion of ViT's inductive bias and the link to neurophysiology are left out, except for a mention of ViT under "Going deeper".

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 圖片描述 | image captioning |
| 視覺問答 | visual question answering |
| 語音辨識 | speech recognition |
| 空間注意力 | spatial attention |
| 通道注意力 | channel attention |
| 壓縮與激發網路 | Squeeze-and-Excitation Network (SENet) |
| 壓縮 | squeeze |
| 激發 | excitation |
| 縮放 | scale |
| 閘門 | gate |
| 特徵圖 | feature map |
| 卷積神經網路 | convolutional neural network (CNN) |
| 長短期記憶網路 | long short-term memory (LSTM) |
| 循環神經網路 | recurrent neural network (RNN) |
| 視覺 Transformer | Vision Transformer (ViT) |
| 支持向量機 | support vector machine (SVM) |
| 主成分分析 | principal component analysis (PCA) |
| 整流線性單元 | rectified linear unit (ReLU) |
| ImageNet 影像資料集（及其辨識競賽 ILSVRC） | ImageNet (and its recognition challenge ILSVRC) |
