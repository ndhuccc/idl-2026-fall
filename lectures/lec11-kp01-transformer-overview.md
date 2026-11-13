# 第11週 KP01：Transformer 全景，編碼器一次讀完整句，解碼器一邊參考一邊寫（The Transformer at a glance: the encoder reads the whole sentence at once, the decoder writes while consulting it）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第11週「注意力機制與 Transformer（二）：架構與應用」的第一個知識點（KP01）。第10週把自注意力拆成一個個零件：查詢、鍵、值、縮放與 softmax。這一節把零件組裝起來，看一座完全不用 RNN 的翻譯模型長什麼樣子。

This is the first knowledge point (KP01) of Week 11, "Attention mechanism and Transformer (2): architecture and applications". Week 10 took self-attention apart into its pieces: query, key, value, scaling and softmax. This note puts the pieces together and shows what a translation model without any RNN looks like.

先備知識是第10週的帶注意力 Seq2Seq（KP03）與自注意力的矩陣形式（KP05）。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第9章改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec11-kp01-figures.py` 算出，可以自己重算。

The prerequisites are the Seq2Seq with attention of Week 10 (KP03) and the matrix form of self-attention (KP05). Every term and symbol is defined when it first appears. This note is rewritten from Chapter 9 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec11-kp01-figures.py`, so you can recompute it yourself.

## KP01 Transformer 全景，編碼器一次讀完整句，解碼器一邊參考一邊寫（The Transformer at a glance: the encoder reads the whole sentence at once, the decoder writes while consulting it）

### 學習目標（Learning objectives）

讀完之後，你能說出 Transformer 的編碼器與解碼器各由哪些子層組成，以及它們的順序。你能追蹤一句話在編碼器裡的矩陣形狀，並說明為什麼形狀不變才能一層層堆疊。你能用一個小例子說明自注意力不管詞的順序，所以模型需要位置編碼。你也能區分只用編碼器、只用解碼器，以及兩者都用的三種模型家族。

After reading, you can name the sublayers that make up the encoder and the decoder of the Transformer, and their order. You can follow the matrix shape of a sentence through the encoder and explain why an unchanged shape is what allows layers to be stacked. You can use a small example to show that self-attention ignores word order, which is why the model needs positional encoding. You can also tell apart the three model families: encoder only, decoder only, and encoder plus decoder.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $n$ | 句子的詞數 | number of words in the sentence |
| $d_{\text{model}}$ | 每個詞向量的長度，整個模型都用這個長度 | length of every word vector, used throughout the model |
| $\mathbf{Z}_{l}\in\mathbb{R}^{n\times d_{\text{model}}}$ | 第 $l$ 層編碼器的輸出，每一列是一個詞 | output of encoder layer $l$, one word per row |
| $N$ | 編碼器（或解碼器）疊幾層 | number of stacked encoder (or decoder) layers |
| $h$ | 注意力頭的數目（本週 KP03 詳談） | number of attention heads (detailed in this week's KP03) |
| $d_k$ | 每個頭的查詢與鍵的長度 | length of the queries and keys of each head |
| $d_{\text{ff}}$ | 前饋網路中間層的寬度 | width of the middle layer of the feed-forward network |
| $\operatorname{MHA}(\cdot)$ | 多頭注意力 | multi-head attention |
| $\operatorname{FFN}(\cdot)$ | 逐位置前饋網路 | position-wise feed-forward network |
| $\operatorname{LN}(\cdot)$ | 層正規化（本週 KP04 詳談） | layer normalization (detailed in this week's KP04) |

### 1. 保留編碼器—解碼器的骨架，拿掉 RNN（Keep the encoder–decoder skeleton, remove the RNN）

第10週 KP03 的帶注意力 Seq2Seq 已經很強：解碼器寫每個詞時，都能回頭加權編碼器的所有狀態 $\mathbf{h}_j$。可是那些 $\mathbf{h}_j$ 仍然由 RNN 一步一步算出來。第 $t$ 步要等第 $t-1$ 步算完，整句話無法同時處理。

The Seq2Seq with attention of Week 10 KP03 was already strong: before writing each word, the decoder could reweight all encoder states $\mathbf{h}_j$. Yet those $\mathbf{h}_j$ were still computed by an RNN, one step at a time. Step $t$ had to wait for step $t-1$, so a sentence could not be processed all at once.

2017 年，Vaswani 等人在論文《Attention Is All You Need》提出 **Transformer**。它保留「編碼器讀原文、解碼器寫譯文」的骨架，但把 RNN 全部拿掉，也不用卷積。兩邊都只用注意力和**前饋網路**（feed-forward network，一般的全連接網路）一層層堆起來。

In 2017, Vaswani and colleagues introduced the **Transformer** in the paper "Attention Is All You Need". It keeps the skeleton "the encoder reads the source, the decoder writes the translation", but removes every RNN and uses no convolution either. Both sides are built only from attention and **feed-forward networks** (ordinary fully connected networks), stacked layer upon layer.

![左右兩座塔。左塔是編碼器：最下面是輸入嵌入加位置編碼，藍色虛線框裡依序是多頭自注意力、add & norm、前饋網路、add & norm，框旁標乘 N；上方輸出 n 乘 d_model。右塔是解碼器：輸出嵌入加位置編碼，紫色虛線框裡依序是遮罩多頭自注意力、add & norm、交叉注意力、add & norm、前饋網路、add & norm，框旁標乘 N；最上方是線性層加 softmax，輸出下一個詞的機率。一條藍色箭頭從編碼器輸出指向交叉注意力，標 K、V（Two towers: the encoder stacks multi-head self-attention and feed-forward, each followed by add & norm, N times; the decoder stacks masked self-attention, cross-attention and feed-forward, each followed by add & norm, N times, then linear and softmax; a blue arrow labelled K, V goes from the encoder output to the cross-attention）](../assets/figures/lec11-kp01-architecture.png)

**圖說 `lec11-kp01-architecture`**〔功能：一目了然〕：先看左塔，由下往上：詞先變成向量並加上位置資訊，接著經過「注意力、整理」和「前饋、整理」兩組子層，虛線框表示這一組要疊 $N$ 層。再看右塔：它多了一個交叉注意力，藍色箭頭表示它的鍵 $K$ 與值 $V$ 來自編碼器的輸出，查詢 $Q$ 來自解碼器自己。最後看右塔頂端：線性層加 softmax 給出下一個詞的機率。

**Caption `lec11-kp01-architecture`** [Function: seeing it at a glance]: First read the left tower from bottom to top: words become vectors with position information added, then pass two groups of sublayers, "attention, tidy up" and "feed-forward, tidy up"; the dashed frame means this group is stacked $N$ times. Then look at the right tower: it has an extra cross-attention, and the blue arrow shows that its keys $K$ and values $V$ come from the encoder output while its query $Q$ comes from the decoder itself. Finally look at the top of the right tower: a linear layer plus softmax gives the probabilities of the next word.

來源的比喻是兩棟摩天大樓。左棟是編碼器，所有詞同時進來，互相交流，整理出一份帶上下文的語意地圖。右棟是解碼器，一邊看左棟送來的地圖，一邊按順序寫出下一個詞。這個比喻的侷限是：真正的兩座塔不是各蓋一次就好，解碼器每一層都會向編碼器的最後輸出取資料，而且兩座塔是一起訓練的。

The source's analogy is two skyscrapers. The left building is the encoder: all words come in together, talk to one another, and produce a semantic map with context. The right building is the decoder: it consults the map sent over from the left building while writing the next word in order. The limitation of this analogy is that the towers are not built separately once: every decoder layer fetches from the encoder's final output, and the two towers are trained together.

常見錯誤：以為 Transformer 是「自編碼器」。來源用了這個詞，但自編碼器要把輸入還原成自己；Transformer 的輸入是法文、輸出是英文，是編碼器—解碼器模型。

Common mistake: thinking the Transformer is an "autoencoder". The source uses that word, but an autoencoder reconstructs its own input; the Transformer takes French in and puts English out, so it is an encoder–decoder model.

### 2. 編碼器的一層：注意力、前饋，各自包一層整理（One encoder layer: attention and feed-forward, each wrapped in a tidy-up step）

圖的左塔先看懂一層就夠了，因為 $N$ 層長得一模一樣。進入第一層之前，每個詞先查表變成一個 $d_{\text{model}}$ 維的**詞嵌入**（embedding），再加上**位置編碼**（positional encoding，本週 KP02 詳談）。一句 $n$ 個詞的話，就是一個 $n\times d_{\text{model}}$ 的矩陣 $\mathbf{Z}_0$，每一列是一個詞。

For the left tower, it is enough to understand one layer, because the $N$ layers look exactly alike. Before the first layer, each word is looked up in a table and becomes a $d_{\text{model}}$-dimensional **embedding**, and then a **positional encoding** is added (detailed in this week's KP02). A sentence of $n$ words is thus an $n\times d_{\text{model}}$ matrix $\mathbf{Z}_0$, one word per row.

每一層做兩件事。第一件是**多頭自注意力**：第10週 KP05 的自注意力，同時做好幾組，讓每個詞依整句重新描述自己（本週 KP03 詳談）。第二件是前饋網路：對每個詞**各自**做同一個兩層全連接網路，詞和詞之間不交流。每件事做完都接一個「Add & Norm」：把輸入加回輸出（**殘差連接**，residual connection），再做**層正規化**（layer normalization）。本週 KP04 會說明這兩步為什麼重要。

Each layer does two things. The first is **multi-head self-attention**: the self-attention of Week 10 KP05, done in several groups at once, so that every word redescribes itself using the whole sentence (detailed in this week's KP03). The second is the feed-forward network: the same two-layer fully connected network applied to **each word separately**, with no exchange between words. Each step is followed by "Add & Norm": add the input back to the output (the **residual connection**), then apply **layer normalization**. This week's KP04 explains why these two steps matter.

寫成式子，第 $l$ 層從 $\mathbf{Z}_{l-1}$ 算出 $\mathbf{Z}_l$：

As formulas, layer $l$ computes $\mathbf{Z}_l$ from $\mathbf{Z}_{l-1}$:

$$
\tilde{\mathbf{Z}}_l=\operatorname{LN}\!\left(\mathbf{Z}_{l-1}+\operatorname{MHA}(\mathbf{Z}_{l-1})\right),\qquad \mathbf{Z}_l=\operatorname{LN}\!\left(\tilde{\mathbf{Z}}_l+\operatorname{FFN}(\tilde{\mathbf{Z}}_l)\right).
$$

關鍵是形狀。多頭注意力和前饋網路的輸出，形狀都和輸入一樣，是 $n\times d_{\text{model}}$。所以第 $l$ 層的輸出可以直接當第 $l+1$ 層的輸入，同樣的層可以一直往上疊。原論文疊 $N=6$ 層。

The key is the shape. Multi-head attention and the feed-forward network both output the same shape as their input, $n\times d_{\text{model}}$. So the output of layer $l$ can be fed straight into layer $l+1$, and identical layers can be stacked indefinitely. The original paper stacks $N=6$ layers.

例子（來源的 Transformer-Base 設定）：$d_{\text{model}}=512$、$h=8$ 個頭、每頭 $d_k=512/8=64$、$d_{\text{ff}}=4\times512=2048$、$N=6$，詞表是英德翻譯共用的 $37000$ 個**子詞**（subword，常見的詞或詞的片段），總參數約 $6500$ 萬。拿一句 $4$ 個詞的話追蹤形狀：

Example (the source's Transformer-Base settings): $d_{\text{model}}=512$, $h=8$ heads, $d_k=512/8=64$ per head, $d_{\text{ff}}=4\times512=2048$, $N=6$, a shared English–German vocabulary of $37000$ **subwords** (common words or pieces of words), and about $65$ million parameters in total. Follow the shape of a $4$-word sentence:

| 位置 | 形狀 |
|---|---|
| 嵌入加位置編碼 $\mathbf{Z}_0$ | $4\times512$ |
| 多頭注意力之後 | $4\times512$ |
| 前饋網路的中間層 | $4\times2048$ |
| 一層的輸出 $\mathbf{Z}_1$ | $4\times512$ |

| Where | Shape |
|---|---|
| Embedding plus positional encoding $\mathbf{Z}_0$ | $4\times512$ |
| After multi-head attention | $4\times512$ |
| Middle layer of the feed-forward network | $4\times2048$ |
| Output of one layer $\mathbf{Z}_1$ | $4\times512$ |

一層有多少參數？`[補充]` 多頭注意力有 $4$ 個 $512\times512$ 的矩陣（查詢、鍵、值與最後的輸出投影）加上偏差，共 $4\times(512\times512+512)=1050624$ 個；$8$ 個頭各用其中 $64$ 行，合起來剛好是一個 $512\times512$ 矩陣。前饋網路是 $512\to2048\to512$，共 $(512\times2048+2048)+(2048\times512+512)=2099712$ 個。兩個層正規化各有 $512$ 個縮放與 $512$ 個平移參數，共 $1024$ 個。一層合計 $3152384$ 個，大約 $315$ 萬，其中前饋網路占三分之二（$0.6661$）。

How many parameters does one layer have? `[Added]` Multi-head attention has $4$ matrices of size $512\times512$ (query, key, value and the final output projection) plus biases, $4\times(512\times512+512)=1050624$ in total; the $8$ heads each use $64$ columns of them, which together make exactly one $512\times512$ matrix. The feed-forward network goes $512\to2048\to512$, giving $(512\times2048+2048)+(2048\times512+512)=2099712$. The two layer normalizations each have $512$ scale and $512$ shift parameters, $1024$ in all. One layer totals $3152384$, about $3.15$ million, and the feed-forward network is two thirds of it ($0.6661$).

常見錯誤：以為 $N$ 層共用同一組參數。它們只是**結構**相同；每一層都有自己的一組權重，$6$ 層編碼器就有 $6\times3152384=18914304$ 個參數。再加上 $6$ 層解碼器約 $2522$ 萬、共用的詞嵌入 $37000\times512$ 約 $1894$ 萬，合計約 $6308$ 萬，和論文報告的約 $6500$ 萬同一量級 `[補充]`。

Common mistake: thinking the $N$ layers share one set of parameters. They only share the same **structure**; every layer has its own weights, so the $6$ encoder layers hold $6\times3152384=18914304$ parameters. Adding about $25.22$ million for the $6$ decoder layers and about $18.94$ million for the shared $37000\times512$ embedding gives about $63.08$ million, in line with the paper's reported $65$ million `[Added]`.

### 3. 自注意力不管詞的順序，所以要加位置編碼（Self-attention ignores word order, so positional encoding is needed）

上一節說，進編碼器前要先「加上位置編碼」。為什麼需要？因為 RNN 拿掉以後，模型裡已經沒有任何東西在一步一步走。第10週 KP05 的自注意力只看「誰和誰相關」，完全不看誰排在前面。

The previous section said that a positional encoding is added before the encoder. Why is it needed? Because once the RNN is gone, nothing in the model walks through the sentence step by step any more. The self-attention of Week 10 KP05 only looks at "who is related to whom", not at who comes first.

用數字看清楚 `[補充]`：沿用第10週 KP05 的設定，三個矩陣都取單位矩陣，$d_k=2$，所以 $\mathbf{Z}=\operatorname{softmax}(\mathbf{X}\mathbf{X}^\top/\sqrt{2})\,\mathbf{X}$。令「我」$=[1,0]^\top$、「打」$=[0,1]^\top$、「你」$=[1,1]^\top$，沒有加位置編碼。句子「我打你」算完自注意力，三個詞的新表示是：「我」$[0.8022,\ 0.5989]^\top$，「打」$[0.5989,\ 0.8022]^\top$，「你」$[0.7517,\ 0.7517]^\top$。把句子換成「你打我」再算一次，每個詞得到的新表示**一模一樣**，只是排列的位置跟著換了。

See it with numbers `[Added]`: keep the setting of Week 10 KP05, with all three matrices equal to the identity and $d_k=2$, so $\mathbf{Z}=\operatorname{softmax}(\mathbf{X}\mathbf{X}^\top/\sqrt{2})\,\mathbf{X}$. Let "wo" (I) $=[1,0]^\top$, "da" (hit) $=[0,1]^\top$ and "ni" (you) $=[1,1]^\top$, with no positional encoding. For the sentence "wo da ni" (I hit you), self-attention gives these new representations: "wo" $[0.8022,\ 0.5989]^\top$, "da" $[0.5989,\ 0.8022]^\top$, "ni" $[0.7517,\ 0.7517]^\top$. Swap the sentence to "ni da wo" (you hit me) and compute again: every word gets **exactly the same** new representation; only the rows are reordered.

![兩張長條圖並排，每個詞兩根長條代表新表示的兩個分量。左圖順序是 wo、da、ni，右圖順序是 ni、da、wo。藍色的 wo 在兩圖都是 0.8022 與 0.5989，綠色的 da 都是 0.5989 與 0.8022，紅色的 ni 都是 0.7517 與 0.7517，只有所在位置不同（Two bar charts side by side; each word has two bars for the two components of its new representation. The left uses the order wo, da, ni and the right ni, da, wo. Blue wo is 0.8022 and 0.5989 in both, green da is 0.5989 and 0.8022, and red ni is 0.7517 and 0.7517; only their positions differ）](../assets/figures/lec11-kp01-permutation.png)

**圖說 `lec11-kp01-permutation`**〔功能：精準度〕：先看同一個顏色在左右兩圖的長條：藍色的「我」不論排第一還是第三，兩個分量都是 $0.8022$ 和 $0.5989$；紅色的「你」也都是 $0.7517$。再看位置標籤：唯一改變的是誰站在哪個位置。所以只看每個詞的新表示，模型分不出是誰打了誰。

**Caption `lec11-kp01-permutation`** [Function: precision]: First compare the bars of the same colour across the two panels: the blue "wo" (I) has components $0.8022$ and $0.5989$ whether it stands first or third, and the red "ni" (you) is $0.7517$ in both. Then look at the position labels: the only thing that changes is who stands where. So, from each word's new representation alone, the model cannot tell who hit whom.

這個性質叫**置換等變**（permutation equivariance）：輸入的列換了順序，輸出的列就照同樣方式換順序，內容不變。前饋網路與 Add & Norm 都對每個詞各自處理，也不看位置。所以整個編碼器都分不出「我打你」和「你打我」。解法是在輸入時，就把「第幾個位置」寫進每個詞的向量，這就是本週 KP02 的位置編碼。

This property is called **permutation equivariance**: when the input rows are reordered, the output rows are reordered the same way and their contents stay the same. The feed-forward network and Add & Norm treat each word separately and ignore position too. So the whole encoder cannot tell "I hit you" from "you hit me". The fix is to write "which position" into each word's vector at the input, and that is the positional encoding of this week's KP02.

常見錯誤：以為來源說的「拿掉位置編碼，兩句的輸出完全一致」是指整個輸出矩陣相同。精確的說法是：每個詞的新表示相同，輸出矩陣的列只是換了順序。這已經足以讓模型分不出語序。

Common mistake: reading the source's "without positional encoding the two sentences give exactly the same output" as "the whole output matrix is identical". The precise statement is that every word gets the same new representation and the rows of the output matrix are merely reordered. That is already enough for the model to lose the word order.

### 4. 解碼器：遮罩、交叉注意力，再加三種模型家族（The decoder: masking and cross-attention, plus three model families）

編碼器把原文整理好以後，輪到右塔的解碼器寫譯文。解碼器的一層有三個子層，每個也都包著 Add & Norm。

Once the encoder has processed the source, the decoder in the right tower writes the translation. A decoder layer has three sublayers, each also wrapped in Add & Norm.

- **遮罩自注意力**（masked self-attention）：對已經寫出的詞做自注意力，但遮住還沒寫的詞，寫第 $i$ 個詞時只能看開始符號與前 $i-1$ 個詞（**Masked self-attention**: self-attention over the words already written, with the unwritten words masked out, so writing word $i$ can only look at the start symbol and the first $i-1$ words）
- **交叉注意力**（cross-attention）：查詢來自解碼器，鍵和值來自編碼器的輸出，和第10週 KP03 的帶注意力 Seq2Seq 同一個想法（**Cross-attention**: the queries come from the decoder and the keys and values from the encoder output, the same idea as the Seq2Seq with attention of Week 10 KP03）
- **前饋網路**：和編碼器的一樣，對每個位置各自處理（**Feed-forward network**: the same as in the encoder, applied to each position separately）

疊完 $N$ 層後，最上面的線性層把每個位置的 $512$ 維向量變成 $37000$ 個分數，一個子詞一個分數，再經 softmax 得到下一個詞的機率。解碼器的輸入是「已經寫出的詞，往右移一格」：第一個位置放一個表示開始的符號，所以寫第 $i$ 個詞時，輸入是前 $i-1$ 個詞。

After the $N$ layers, the top linear layer turns the $512$-dimensional vector at each position into $37000$ scores, one per subword, and softmax turns them into the probabilities of the next subword. The decoder's input is "the words already written, shifted right by one": the first position holds a start symbol, so when writing word $i$ the input is the first $i-1$ words.

訓練時，解碼器的輸入用正確譯文的前面幾個詞，而不是模型自己猜的詞，這叫**教師強制**（teacher forcing）。因為正確譯文整句已知，就可以把整句一次放進去，再靠遮罩保證每個位置看不到自己右邊的答案，所有位置平行計算。真正使用時沒有答案，只能一次寫一個詞，再把寫出的詞接回輸入。本週 KP05 會把遮罩和交叉注意力算給你看。

During training, the decoder's input is the earlier words of the correct translation rather than the model's own guesses; this is called **teacher forcing**. Because the correct translation is known in full, the whole sentence can go in at once, and the mask guarantees that no position sees the answers to its right, so all positions are computed in parallel. In actual use there is no answer, so the model writes one word at a time and feeds each written word back into its input. This week's KP05 will compute the mask and the cross-attention for you.

這張藍圖後來被拆成三種用法。只用左塔的是**只用編碼器**（encoder-only）的模型，例如 BERT，擅長理解與分類。只用右塔、不要交叉注意力的是**只用解碼器**（decoder-only）的模型，例如 GPT，擅長一個詞一個詞地生成文字。兩座塔都用的是原始的 Transformer 與 T5，擅長翻譯這類「讀一段、寫另一段」的任務。本週 KP06 會比較 BERT 與 GPT 的訓練方式。

This blueprint was later split into three uses. Using only the left tower gives **encoder-only** models such as BERT, good at understanding and classification. Using only the right tower, without cross-attention, gives **decoder-only** models such as GPT, good at generating text word by word. Using both towers gives the original Transformer and T5, good at "read one passage, write another" tasks such as translation. This week's KP06 compares how BERT and GPT are trained.

常見錯誤：以為解碼器在訓練時也必須一個詞一個詞地算。訓練時用教師強制，譯文已知，再靠遮罩就能整句平行計算；只有真正生成時才必須一步一步來。

Common mistake: thinking the decoder must compute word by word during training too. In training, teacher forcing supplies the known translation and the mask allows the whole sentence to be computed in parallel; only real generation has to proceed step by step.

### 本單元重點整理（KP01 summary）

**一句話重點**：Transformer 保留編碼器—解碼器的骨架，但只用注意力與前饋網路堆疊；每個子層包著殘差與層正規化、輸入輸出同為 $n\times d_{\text{model}}$，所以能疊 $N$ 層；自注意力不管語序，所以輸入要加位置編碼；解碼器用遮罩只看已寫出的詞，用交叉注意力向編碼器取資料。

**One-sentence summary**: the Transformer keeps the encoder–decoder skeleton but builds it only from attention and feed-forward networks; every sublayer is wrapped in a residual connection and layer normalization and keeps the shape $n\times d_{\text{model}}$, so $N$ layers can be stacked; self-attention ignores word order, so the input gets a positional encoding; the decoder uses a mask to see only the words written so far and cross-attention to fetch from the encoder.

**白話說明**：左棟的人一起讀完整篇原文，互相討論，整理成一份筆記；右棟的人照順序寫譯文，每寫一個詞，都翻一下左棟的筆記，而且不准偷看自己還沒寫的部分。

**Plain explanation**: the people in the left building read the whole source together, discuss it and condense it into notes; the people in the right building write the translation in order, glance at the left building's notes before every word, and are not allowed to peek at the part they have not written yet.

**正式定義**：$\tilde{\mathbf{Z}}_l=\operatorname{LN}(\mathbf{Z}_{l-1}+\operatorname{MHA}(\mathbf{Z}_{l-1}))$，$\mathbf{Z}_l=\operatorname{LN}(\tilde{\mathbf{Z}}_l+\operatorname{FFN}(\tilde{\mathbf{Z}}_l))$，$l=1,\dots,N$；解碼器每層依序是遮罩自注意力、交叉注意力、前饋網路，各包 Add & Norm，最後是線性層加 softmax。

**Formal definition**: $\tilde{\mathbf{Z}}_l=\operatorname{LN}(\mathbf{Z}_{l-1}+\operatorname{MHA}(\mathbf{Z}_{l-1}))$, $\mathbf{Z}_l=\operatorname{LN}(\tilde{\mathbf{Z}}_l+\operatorname{FFN}(\tilde{\mathbf{Z}}_l))$, $l=1,\dots,N$; each decoder layer is masked self-attention, cross-attention and a feed-forward network in that order, each wrapped in Add & Norm, followed at the top by a linear layer and softmax.

**例子**：Transformer-Base 的 $d_{\text{model}}=512$、$h=8$、$d_k=64$、$d_{\text{ff}}=2048$、$N=6$；$4$ 個詞的句子在編碼器裡一直是 $4\times512$；一層編碼器有 $3152384$ 個參數（約 $315$ 萬），前饋網路占 $0.6661$。

**Example**: Transformer-Base has $d_{\text{model}}=512$, $h=8$, $d_k=64$, $d_{\text{ff}}=2048$, $N=6$; a $4$-word sentence stays $4\times512$ throughout the encoder; one encoder layer has $3152384$ parameters (about $3.15$ million), $0.6661$ of them in the feed-forward network.

**常見迷思**：

- 「Transformer 是自編碼器。」錯，它是編碼器—解碼器，輸入與輸出是不同的序列。
- 「$N$ 層共用同一組參數。」錯，結構相同、權重各自獨立。
- 「解碼器訓練時也要一步一步算。」錯，教師強制加遮罩可以整句平行。

**Common misconceptions**:

- "The Transformer is an autoencoder." Wrong: it is an encoder–decoder, and the input and output are different sequences.
- "The $N$ layers share one set of parameters." Wrong: the structure is the same, but each layer has its own weights.
- "The decoder must compute step by step during training." Wrong: teacher forcing with a mask lets the whole sentence run in parallel.

**圖**：本單元有兩張圖。

- `lec11-kp01-architecture`：編碼器與解碼器兩座塔的子層與資料流。
- `lec11-kp01-permutation`：沒有位置編碼時，換了詞序，每個詞的新表示不變。

**Figures** — this unit has two figures.

- `lec11-kp01-architecture`: the sublayers and data flow of the encoder and decoder towers.
- `lec11-kp01-permutation`: without positional encoding, changing the word order leaves every word's new representation unchanged.

### 小練習（Practice）

1. 一句話有 $n=10$ 個詞，用 Transformer-Base 的設定。寫出編碼器第一層的輸入、多頭注意力之後、前饋網路中間層、這一層輸出的形狀。
2. 把設定縮小成 $d_{\text{model}}=256$、$d_{\text{ff}}=1024$。一層編碼器的前饋網路（含偏差）有幾個參數？多頭注意力的四個矩陣（含偏差）呢？
3. 解碼器為什麼同時需要遮罩自注意力和交叉注意力？各用一句話說明它們分別看什麼。

1. A sentence has $n=10$ words and uses the Transformer-Base settings. Write the shapes of the input to the first encoder layer, the output of multi-head attention, the middle layer of the feed-forward network, and the output of the layer.
2. Shrink the settings to $d_{\text{model}}=256$ and $d_{\text{ff}}=1024$. How many parameters (with biases) does the feed-forward network of one encoder layer have? And the four matrices of multi-head attention (with biases)?
3. Why does the decoder need both masked self-attention and cross-attention? Say in one sentence each what they look at.

**第 1 題解答** `[補充]`：輸入 $10\times512$，多頭注意力之後 $10\times512$，前饋網路中間層 $10\times2048$，這一層輸出 $10\times512$。

**Solution to question 1** `[Added]`: the input is $10\times512$, after multi-head attention $10\times512$, the middle layer of the feed-forward network $10\times2048$, and the layer output $10\times512$.

**第 2 題解答** `[補充]`：前饋網路 $(256\times1024+1024)+(1024\times256+256)=525568$ 個；多頭注意力 $4\times(256\times256+256)=263168$ 個。

**Solution to question 2** `[Added]`: the feed-forward network has $(256\times1024+1024)+(1024\times256+256)=525568$; multi-head attention has $4\times(256\times256+256)=263168$.

**第 3 題解答** `[補充]`：遮罩自注意力看的是自己已經寫出的詞，讓前後文連貫，又不偷看答案；交叉注意力看的是編碼器整理好的原文，讓譯文忠於原文。

**Solution to question 3** `[Added]`: masked self-attention looks at the words the decoder has already written, keeping the output coherent without peeking at the answer; cross-attention looks at the source as processed by the encoder, keeping the translation faithful to the source.

### 小測驗（Quiz，來源原題）

來源的兩題自我檢測保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The source's two self-check questions are kept with all their options; the source page does not mark the answers, so the explanations below are judged from the source text.

1. Transformer 架構相較於傳統 RNN 最根本的突破是什麼？
   - (A) 大幅增加遞迴時間步長
   - (B) 徹底捨棄遞迴結構，純粹依賴注意力機制實現全並行序列建模
   - (C) 完全不使用任何線性轉換矩陣
   - (D) 只能處理單詞無法處理句子
2. 標準 Transformer 編碼器中通常堆疊幾層相同的 Encoder Block？
   - (A) $1$ 層
   - (B) $6$ 層（$N=6$）
   - (C) $100$ 層
   - (D) $0$ 層

1. What is the most fundamental breakthrough of the Transformer architecture over a traditional RNN?
   - (A) It greatly increases the number of recurrent time steps
   - (B) It drops recurrence completely and relies purely on attention to model sequences fully in parallel
   - (C) It uses no linear transformation matrices at all
   - (D) It can only process single words, not sentences
2. How many identical encoder blocks does a standard Transformer encoder usually stack?
   - (A) $1$
   - (B) $6$ ($N=6$)
   - (C) $100$
   - (D) $0$

**解析** `[補充]`：第 1 題選 (B)：第1節說明它拿掉 RNN、只用注意力與前饋網路，選項中的「純粹」指的是不用遞迴；(C) 錯，查詢、鍵、值與前饋網路都是線性轉換。第 2 題選 (B)：第2節的 Transformer-Base 設定是 $N=6$。

**Explanations** `[Added]`: question 1 is B: Section 1 explains that it removes the RNN and uses only attention and feed-forward networks, so "purely" in the option means "without recurrence"; C is wrong because the queries, keys, values and feed-forward networks are all linear transformations. Question 2 is B: the Transformer-Base setting in Section 2 is $N=6$.

### 留給下一單元的問題（The question for the next unit）

第3節說，自注意力分不出「我打你」和「你打我」，所以要在輸入時把位置寫進詞向量。可是要寫什麼？直接寫 $1,2,3,\dots$ 好嗎？句子很長時數字會變得很大；換成別的寫法，模型又要怎麼從中讀出「兩個詞相隔多遠」？下一單元（KP02）的正弦位置編碼回答這個問題。

Section 3 showed that self-attention cannot tell "I hit you" from "you hit me", so position must be written into the word vectors at the input. But what should be written? Would $1,2,3,\dots$ do? For a long sentence the numbers grow large; with another scheme, how would the model read off "how far apart two words are"? The sinusoidal positional encoding of the next unit (KP02) answers this.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第9章「注意力機制與 Transformer 核心原理」§3.1–§3.2、§3.3（概觀）與教授祕笈「Transformer 完整架構全景」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §3.1–§3.2 and the §3.3 overview of Chapter 9, "Core principles of attention mechanisms and the Transformer", of CCC-LNS IDL_115, and from the professor's notes "The Transformer blueprint" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源把 Transformer 稱為「革命性自編碼器架構」；自編碼器是把輸入還原成自己的模型，Transformer 是編碼器—解碼器，本講義改用後者並在第1節說明。
- 來源說拿掉位置編碼時「我打你」與「你打我」的輸出「完全一致」；精確說法是每個詞的新表示相同、輸出的列換了順序，本講義在第3節以數字示範。
- 來源把詞向量排成行（$\mathbf{X}\in\mathbb{R}^{d\times n}$）；本講義沿用第10週的「列為詞」記號（$n\times d_{\text{model}}$），意義相同。來源用粗斜體 `\boldsymbol` 表示向量；本課程網站不支援，改用 `\mathbf`。
- 來源寫「詞表大小 $37000$」；原論文這是英德翻譯共用的子詞詞表，本講義寫明是子詞。
- 本講義補上：一層編碼器的參數拆解、四個詞的形狀追蹤表、語序的數字示範與圖、三題小練習與解答。總參數「約 $6500$ 萬」沿用來源與原論文。
- 取捨：殘差與層正規化的順序（Post-LN 與 Pre-LN）留給 KP04；BERT 與 GPT 的訓練方式留給 KP06；來源提到的圖神經網路連結與 KV 快取，本講義不寫。

- The source calls the Transformer a "revolutionary autoencoder architecture"; an autoencoder reconstructs its own input, while the Transformer is an encoder–decoder, so this note uses the latter and explains it in Section 1.
- The source says that without positional encoding the outputs for "I hit you" and "you hit me" are "exactly the same"; precisely, every word gets the same new representation and the output rows are reordered, which Section 3 shows with numbers.
- The source arranges word vectors as columns ($\mathbf{X}\in\mathbb{R}^{d\times n}$); this note keeps Week 10's "one word per row" notation ($n\times d_{\text{model}}$) with the same meaning. The source writes vectors with bold italic `\boldsymbol`; the course website does not support it, so this note uses `\mathbf`.
- The source writes "vocabulary size $37000$"; in the original paper this is a shared English–German subword vocabulary, and this note says so.
- This note adds: the parameter breakdown of one encoder layer, the shape table for a four-word sentence, the numerical demonstration of word order with its figure, and three practice questions with solutions. The total of "about $65$ million" parameters follows the source and the original paper.
- Choices: the order of the residual connection and layer normalization (Post-LN versus Pre-LN) is left to KP04; how BERT and GPT are trained is left to KP06; the source's links to graph neural networks and the KV cache are not covered.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 編碼器 | encoder |
| 解碼器 | decoder |
| 前饋網路 | feed-forward network |
| 詞嵌入 | embedding |
| 位置編碼（KP02） | positional encoding |
| 多頭自注意力（KP03） | multi-head self-attention |
| 殘差連接（KP04） | residual connection |
| 層正規化（KP04） | layer normalization (LN) |
| 置換等變 | permutation equivariance |
| 遮罩自注意力（KP05） | masked self-attention |
| 交叉注意力（KP05） | cross-attention |
| 教師強制 | teacher forcing |
| 只用編碼器的模型 | encoder-only model |
| 只用解碼器的模型 | decoder-only model |
| 自編碼器 | autoencoder |
| 子詞 | subword |
| 循環神經網路 | recurrent neural network (RNN) |
| 雙向編碼器表示模型（KP06） | Bidirectional Encoder Representations from Transformers (BERT) |
| 生成式預訓練模型（KP06） | Generative Pre-trained Transformer (GPT) |
