# 第10週 KP02：Seq2Seq 把整句壓進一個上下文向量，句子一長就裝不下（Seq2Seq squeezes a whole sentence into one context vector, which cannot hold a long sentence）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第10週的第二個知識點（KP02），接在 KP01 之後。KP01 說明了注意力是「依查詢給每個位置不同的份量」，但向量和查詢都是直接給的。這一節先看注意力出現之前的翻譯模型：它怎麼讀一句話、再寫出另一句話，又卡在哪裡。

This is the second knowledge point (KP02) of Week 10 and follows KP01. KP01 explained that attention "gives each position a different share according to a query", but the vectors and the query were simply given. This note first looks at the translation model from before attention: how it reads one sentence and writes another, and where it gets stuck.

先備知識是第8週的 RNN 與隱藏狀態，以及 KP01 的加權平均。本講義依課程網頁教材 CCC-LNS 第9章改寫，文中「來源」就是指它；本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec10-kp02-figures.py` 算出，可以自己重算。

The prerequisites are the RNN and hidden state of Week 8 and the weighted average of KP01. This note is rewritten from Chapter 9 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec10-kp02-figures.py`, so you can recompute it yourself.

## KP02 Seq2Seq 把整句壓進一個上下文向量，句子一長就裝不下（Seq2Seq squeezes a whole sentence into one context vector, which cannot hold a long sentence）

### 學習目標（Learning objectives）

讀完之後，你能說出編碼器與解碼器各自的工作，以及兩者之間的上下文向量扮演什麼角色。你能寫出 Seq2Seq 解碼器每一步產生下一個詞所依靠的三樣東西。你能用一個簡單的線性編碼器算出「第一個詞在上下文向量裡還剩多少」，並說明句子變長時為什麼會遺忘。你也能說明為什麼只把隱藏狀態加大，解決不了這個問題。

After reading, you can state the jobs of the encoder and the decoder, and the role of the context vector between them. You can write the three things the Seq2Seq decoder relies on to produce the next word at each step. You can use a simple linear encoder to compute "how much of the first word is left in the context vector", and explain why a longer sentence makes the model forget. You can also explain why merely enlarging the hidden state does not solve the problem.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{x}_1,\dots,\mathbf{x}_n$ | 來源句的 $n$ 個詞（向量） | the $n$ words (vectors) of the source sentence |
| $\mathbf{y}_1,\mathbf{y}_2,\dots$ | 目標句的詞 | the words of the target sentence |
| $\mathbf{h}_t\in\mathbb{R}^{p}$ | 編碼器第 $t$ 步的隱藏狀態 | hidden state of the encoder at step $t$ |
| $p$ | 隱藏狀態的維度（向量有幾個數） | dimension of the hidden state (how many numbers it has) |
| $\mathbf{c}\in\mathbb{R}^{p}$ | 上下文向量，交給解碼器的整句摘要 | context vector, the summary handed to the decoder |
| $\mathbf{l}_i$ | 解碼器第 $i$ 步的內部狀態 | internal state of the decoder at step $i$ |
| $g(\cdot)$ | 解碼器算出下一個詞機率的函數 | the decoder's function that gives the next-word probabilities |
| $\mathbb{P}(\cdot\mid\cdot)$ | 條件機率 | conditional probability |

### 1. 編碼器、解碼器與上下文向量（Encoder, decoder and the context vector）

很多任務是「讀進一種東西、寫出另一種東西」：法文翻成英文、語音轉成文字、卡通風格的大象圖換成真實照片。來源把這類模型拆成兩半。**編碼器**（encoder）把輸入壓成一個向量；**解碼器**（decoder）只看這個向量，把它轉成輸出。

Many tasks "read one thing and write another": French into English, speech into text, a cartoon elephant into a real photo. The source splits such models into two halves. The **encoder** compresses the input into one vector; the **decoder** looks only at this vector and turns it into the output.

兩半之間的那個向量叫**上下文向量**（context vector），記作 $\mathbf{c}$。它就像一份濃縮摘要：讀到法文的 `éléphant`，編碼器產生代表「大象」這個概念的 $\mathbf{c}$，解碼器再由 $\mathbf{c}$ 寫出英文 `elephant`。

The vector between the two halves is called the **context vector**, written $\mathbf{c}$. It is like a condensed summary: reading the French "éléphant", the encoder produces a $\mathbf{c}$ that stands for the concept "elephant", and the decoder writes the English "elephant" from $\mathbf{c}$.

這一週的主角是翻譯；其他例子只是說明這個骨架用途很廣。輸入和輸出是同一種東西、目標是把輸入還原時，這種模型叫**自編碼器**（autoencoder）。第12週會專門講它；這一節只借用「編碼器—上下文向量—解碼器」這個骨架。

This week's main case is translation; the other examples only show that the skeleton is widely useful. When the input and the output are the same kind of thing and the goal is to rebuild the input, the model is called an **autoencoder**. Week 12 covers it in detail; this note only borrows the skeleton "encoder – context vector – decoder".

常見錯誤：以為上下文向量就是輸入本身。它是編碼器算出來的新向量，長度固定，通常比輸入短得多；解碼器看不到原始輸入，只看得到它。

Common error: thinking that the context vector is the input itself. It is a new vector computed by the encoder, with a fixed length that is usually much shorter than the input; the decoder cannot see the raw input, only this vector.

### 2. Seq2Seq：用兩個 RNN 做翻譯（Seq2Seq: translation with two RNNs）

當輸入和輸出都是序列，而且兩者長度可以不同，例如三個法文詞 `Je suis étudiant` 翻成四個英文詞 `I am a student`，就需要**序列到序列模型**（sequence-to-sequence model，Seq2Seq）。

When both the input and the output are sequences whose lengths may differ, for example three French words "Je suis étudiant" translated into four English words "I am a student", you need a **sequence-to-sequence model** (Seq2Seq).

編碼器是第8週的 RNN。它一個詞一個詞讀，每讀一個詞就更新隱藏狀態：$\mathbf{h}_t$ 由 $\mathbf{h}_{t-1}$ 和 $\mathbf{x}_t$ 算出。讀完最後一個詞，最後的隱藏狀態就當作上下文向量：

The encoder is the RNN of Week 8. It reads one word at a time and updates its hidden state after each word: $\mathbf{h}_t$ is computed from $\mathbf{h}_{t-1}$ and $\mathbf{x}_t$. After the last word, the last hidden state serves as the context vector:

$$
\mathbf{c}=\mathbf{h}_n.
$$

解碼器是另一個 RNN，一次寫一個詞。寫第 $i$ 個詞時，它依靠三樣東西：剛寫出的前一個詞 $\mathbf{y}_{i-1}$、自己的內部狀態 $\mathbf{l}_i$，以及上下文向量 $\mathbf{c}$。用式子寫：

The decoder is another RNN that writes one word at a time. To write the $i$-th word it relies on three things: the word it has just written, $\mathbf{y}_{i-1}$; its own internal state $\mathbf{l}_i$; and the context vector $\mathbf{c}$. As an equation:

$$
\mathbb{P}(\mathbf{y}_i\mid\mathbf{y}_1,\dots,\mathbf{y}_{i-1})=g(\mathbf{y}_{i-1},\mathbf{l}_i,\mathbf{c}).
$$

左邊是「已經寫出前面幾個詞時，下一個詞是什麼的機率」。條件裡沒有寫出來源句，因為來源句的資訊全部經由 $\mathbf{c}$ 進來。解碼器自己的狀態也一步一步更新：$\mathbf{l}_i$ 由 $\mathbf{l}_{i-1}$、前一個詞 $\mathbf{y}_{i-1}$ 與 $\mathbf{c}$ 算出，第一步的狀態通常就從 $\mathbf{c}$ 開始 `[補充]`。像這樣用自己前面的輸出去決定下一個輸出，叫**自回歸**（autoregressive）生成。

The left side is "the probability of the next word, given the words already written". The source sentence does not appear in the condition, because all of its information enters through $\mathbf{c}$. The decoder's own state is also updated step by step: $\mathbf{l}_i$ is computed from $\mathbf{l}_{i-1}$, the previous word $\mathbf{y}_{i-1}$ and $\mathbf{c}$, and the first state usually starts from $\mathbf{c}$ `[Added]`. Using one's own earlier outputs to decide the next output like this is called **autoregressive** generation.

![左邊三個綠色方塊是法文詞 Je、suis、étudiant，往上各進入一個藍色隱藏狀態 h_1、h_2、h_3，藍色箭頭由左往右串起。h_3 以粗橘箭頭接到中間的黃色方塊 c = h_3，上方標 fixed size。右邊是四個紫色解碼器狀態 l_1 到 l_4，紫色箭頭由左往右串起，每個往上寫出一個英文詞 I、am、a、student，綠色斜箭頭把寫出的詞送回下一個解碼器狀態；黃色方塊有四條橘色虛線分別接到四個解碼器狀態（Three French words feed the encoder states h_1 to h_3; h_3 becomes the single fixed-size context vector c, and dashed lines carry the same c to every one of the four decoder states that write I, am, a, student）](../assets/figures/lec10-kp02-seq2seq-bottleneck.png)

**圖說 `lec10-kp02-seq2seq-bottleneck`**〔功能：一目了然〕：先看左邊，三個法文詞由下往上進入編碼器，藍色箭頭把狀態一路往右傳。再看中間的黃色方塊，整句話只留下這一個向量 $\mathbf{c}=\mathbf{h}_3$，大小固定。最後看右邊：綠色斜箭頭把剛寫出的詞送回下一步，四條橘色虛線則把同一個 $\mathbf{c}$ 送給每一步解碼，寫 `I` 和寫 `student` 時看到的是同一份摘要。

**Caption `lec10-kp02-seq2seq-bottleneck`** [Function: seeing it at a glance]: First look at the left: the three French words enter the encoder from below, and the blue arrows pass the state to the right. Then look at the yellow box in the middle: the whole sentence leaves behind only this one vector, $\mathbf{c}=\mathbf{h}_3$, of fixed size. Finally look at the right: the green slanted arrows feed each written word back into the next step, and four orange dashed lines deliver the same $\mathbf{c}$ to every decoding step, so writing "I" and writing "student" see the same summary.

Seq2Seq 的好處是**端到端**（end to end）：從原文到譯文，整條流程由一個網路一起學會。在它之前，翻譯系統要人工整理「哪個片語對應哪個片語」的大表，再拼起來 `[補充]`。這在當時是很大的進步。

The benefit of Seq2Seq is that it is **end to end**: the whole path from original to translation is learned by one network. Before it, translation systems needed large hand-built tables of "which phrase matches which phrase", stitched together `[Added]`. This was a big step forward at the time.

常見錯誤：以為解碼器每一步都能重新看一次法文句子。它看不到；它手上只有固定的 $\mathbf{c}$，加上自己寫過的詞。

Common error: thinking that the decoder can look at the French sentence again at every step. It cannot; all it has is the fixed $\mathbf{c}$ plus the words it has already written.

### 3. 瓶頸：前面的詞被後面的詞蓋掉（The bottleneck: early words are overwritten by later ones）

問題就出在「整句只剩一個 $\mathbf{c}$」。不論句子有 $5$ 個詞還是 $50$ 個詞，$\mathbf{c}$ 的長度都是 $p$。而且 $\mathbf{c}$ 是最後一個隱藏狀態，越早讀進來的詞，被後面的更新覆蓋得越多。這叫**資訊瓶頸**（information bottleneck）。

The trouble is exactly that "the whole sentence leaves only one $\mathbf{c}$". Whether the sentence has $5$ words or $50$, the length of $\mathbf{c}$ is $p$. And since $\mathbf{c}$ is the last hidden state, the earlier a word is read, the more it is overwritten by later updates. This is called the **information bottleneck**.

用最簡單的編碼器看清楚（來源的例子）：取純量、去掉 $\tanh$，令

See it with the simplest encoder (the source's example): use scalars, drop the $\tanh$, and let

$$
h_t=0.5\,h_{t-1}+x_t,\qquad h_0=0.
$$

讀兩個詞時，$h_1=x_1$，$h_2=0.5\,x_1+x_2$。所以 $c=h_2$ 裡面，第一個詞只剩 $0.5$ 倍。每多讀一個詞，前面所有的詞都再乘一次 $0.5$。讀 $T$ 個詞以後，第 $t$ 個詞在 $c$ 裡的份量是 $0.5^{T-t}$，第一個詞只剩 $0.5^{T-1}$：

With two words, $h_1=x_1$ and $h_2=0.5\,x_1+x_2$. So inside $c=h_2$, the first word is left with only a factor $0.5$. Each extra word multiplies all the earlier ones by $0.5$ again. After $T$ words, the $t$-th word has weight $0.5^{T-t}$ in $c$, and the first word only $0.5^{T-1}$:

| 句長 $T$ | 第一個詞的份量 $0.5^{T-1}$ |
|---|---|
| $2$ | $0.5$ |
| $5$ | $0.0625$ |
| $10$ | $0.001953$ |
| $20$ | $0.0000019$ |

| Sentence length $T$ | Weight of the first word $0.5^{T-1}$ |
|---|---|
| $2$ | $0.5$ |
| $5$ | $0.0625$ |
| $10$ | $0.001953$ |
| $20$ | $0.0000019$ |

![兩張對數縱軸的長條圖，橫軸是來源詞的位置。左圖 T=5，第一個詞（紅色）的份量 0.0625，往右每個加倍，最後一個是 1。右圖 T=20，第一個詞只剩約 1.9 乘 10 的負 6 次方，長條從左到右一路變高（Two log-scale bar charts of the weight of each source word inside c: with T=5 the first word keeps 0.0625; with T=20 it keeps about 1.9e-6, and the bars rise steadily from left to right）](../assets/figures/lec10-kp02-early-words-fade.png)

**圖說 `lec10-kp02-early-words-fade`**〔功能：精準度〕：先看左圖：五個詞的份量從右往左每一格減半，紅色的第一個詞是 $0.0625$。再看右圖：二十個詞時，紅色長條幾乎貼在底部，只剩約 $1.9\times10^{-6}$。縱軸是對數刻度，每一格差 $10$ 倍，所以長條排成一條斜直線，表示份量是等比例地縮小。

**Caption `lec10-kp02-early-words-fade`** [Function: precision]: First look at the left panel: the weights of the five words halve with every step from right to left, and the red first word is $0.0625$. Then look at the right panel: with twenty words, the red bar almost touches the bottom at about $1.9\times10^{-6}$. The vertical axis is logarithmic, each grid line a factor of $10$, so the bars form a straight slope, which means the weights shrink by the same ratio at every step.

真實的編碼器有 $\tanh$ 和矩陣，不一定剛好是 $0.5$，但「後讀的詞蓋掉先讀的詞」這個趨勢一樣。這也就是第8週 KP04 的梯度消失換個角度看：往前看越遠，連乘的因子越多。

A real encoder has $\tanh$ and matrices, so the factor is not exactly $0.5$, but the trend "later words overwrite earlier ones" is the same. It is the vanishing gradient of Week 8 KP04 seen from another angle: the further back you look, the more factors are multiplied together.

來源的比喻：編碼器像一位口譯員，聽完整篇演講後，把精華裝進一個固定大小的「記憶膠囊」交給另一位口譯員念出英文。演講只有幾句時沒問題；演講一長，小小的膠囊就裝不下。這個比喻的侷限是：真的膠囊是「滿了就裝不進」，而 $\mathbf{c}$ 是「新的蓋掉舊的」，早期內容不是被擋在外面，而是被稀釋掉。

The source's analogy: the encoder is like an interpreter who listens to a whole speech and packs its essence into a fixed-size "memory capsule" for another interpreter to read out in English. A speech of a few sentences is fine; a long speech no longer fits the small capsule. The limit of the analogy: a real capsule "refuses new content when full", while $\mathbf{c}$ "lets new content overwrite old", so early content is not kept out but diluted.

### 4. 加大狀態或倒著讀，都只是緩解（A bigger state or reading backward only eases the problem）

既然 $\mathbf{c}$ 太小，把隱藏狀態的維度 $p$ 加大可以嗎？效果有限。參數會變多、更容易過擬合；而且只要 $\mathbf{c}$ 還是「最後一步的狀態」，越早的詞被連乘得越多這件事沒有改變。

Since $\mathbf{c}$ is too small, can you enlarge the dimension $p$ of the hidden state? Only a little. There are more parameters and more risk of overfitting; and as long as $\mathbf{c}$ is still "the state of the last step", the earlier a word is, the more factors it is multiplied by.

換成第8週的 LSTM 或 GRU 呢？門控能讓早期的詞衰減得慢一些，但 $\mathbf{c}$ 仍然只有一份、長度仍然固定，句子再長一點就又裝不下 `[補充]`。改用第8週 KP08 的雙向 RNN 呢？它讓句首的詞也離某一個方向的終點很近，但兩個方向最後仍拼成一個固定長度的向量，容量的上限還在。

What about the LSTM or GRU of Week 8? Their gates make early words fade more slowly, but $\mathbf{c}$ is still one vector of fixed length, and a somewhat longer sentence again does not fit `[Added]`. What about the bidirectional RNN of Week 8 KP08? It brings the first words close to the end of one direction, but the two directions are still joined into one fixed-length vector, so the cap on capacity remains.

來源也提到一個當年的小技巧：把來源句子倒過來輸入。這樣來源的第一個詞最後才讀，離解碼器寫第一個詞的時間最近，翻譯品質確實提升。可是句子中間的詞照樣離得很遠，它只是把問題移了位置。

The source also mentions a trick of the time: feed the source sentence backward. Then the first source word is read last, closest in time to when the decoder writes its first word, and translation quality does improve. Yet the words in the middle of the sentence are still far away; the trick only moves the problem.

這幾種做法都沒有碰到根本：解碼器每一步都只能看同一個固定的 $\mathbf{c}$。KP01 已經告訴我們另一條路：與其事先壓成一份摘要，不如每次需要時，依相關程度回頭取用每個位置。

None of these touches the root: at every step the decoder can only look at the same fixed $\mathbf{c}$. KP01 has already shown another way: instead of compressing everything into one summary beforehand, look back at every position whenever needed, weighted by relevance.

常見錯誤：以為只要模型夠大，長句子就沒問題。來源提到，沒有注意力的 Seq2Seq 在句子超過約 $20$ 個詞以後，翻譯品質就明顯下滑；瓶頸來自架構，不是單純的大小問題。

Common error: thinking that long sentences are fine as long as the model is big enough. The source notes that Seq2Seq without attention degrades clearly once a sentence goes beyond about $20$ words; the bottleneck comes from the architecture, not simply from the size.

### 本單元重點整理（KP02 summary）

**一句話重點**：Seq2Seq 用編碼器 RNN 把整句來源壓成最後一個隱藏狀態 $\mathbf{c}=\mathbf{h}_n$，再由解碼器 RNN 依 $\mathbf{y}_{i-1}$、$\mathbf{l}_i$、$\mathbf{c}$ 一個詞一個詞寫出譯文；因為整句只剩一個固定長度的 $\mathbf{c}$，越早的詞被蓋掉越多，句子一長就翻不好。

**One-sentence summary**: Seq2Seq uses an encoder RNN to compress the whole source sentence into its last hidden state $\mathbf{c}=\mathbf{h}_n$, and a decoder RNN writes the translation word by word from $\mathbf{y}_{i-1}$, $\mathbf{l}_i$ and $\mathbf{c}$; because the whole sentence leaves only one fixed-length $\mathbf{c}$, earlier words are overwritten more, and long sentences are translated badly.

**白話說明**：先把整篇聽完、記成一張小紙條，再只看紙條翻譯。短的演講沒問題；長的演講，紙條寫不下，開頭講什麼早就被後面的內容擠掉了。

**Plain explanation**: listen to the whole speech, write it down on one small note, and translate from the note alone. A short speech is fine; for a long one the note runs out of room, and what was said at the start has long been pushed out by what came later.

**正式定義**：編碼器 $\mathbf{h}_t=f(\mathbf{h}_{t-1},\mathbf{x}_t)$，$\mathbf{c}=\mathbf{h}_n$；解碼器 $\mathbb{P}(\mathbf{y}_i\mid\mathbf{y}_1,\dots,\mathbf{y}_{i-1})=g(\mathbf{y}_{i-1},\mathbf{l}_i,\mathbf{c})$，每一步用同一個 $\mathbf{c}$。

**Formal definition**: encoder $\mathbf{h}_t=f(\mathbf{h}_{t-1},\mathbf{x}_t)$, $\mathbf{c}=\mathbf{h}_n$; decoder $\mathbb{P}(\mathbf{y}_i\mid\mathbf{y}_1,\dots,\mathbf{y}_{i-1})=g(\mathbf{y}_{i-1},\mathbf{l}_i,\mathbf{c})$, with the same $\mathbf{c}$ at every step.

**例子**：線性編碼器 $h_t=0.5\,h_{t-1}+x_t$、$h_0=0$。$T=2$ 時 $c=0.5\,x_1+x_2$；第一個詞的份量在 $T=5$ 時是 $0.0625$，$T=10$ 時是 $0.001953$，$T=20$ 時約 $1.9\times10^{-6}$。

**Example**: linear encoder $h_t=0.5\,h_{t-1}+x_t$, $h_0=0$. For $T=2$, $c=0.5\,x_1+x_2$; the weight of the first word is $0.0625$ for $T=5$, $0.001953$ for $T=10$ and about $1.9\times10^{-6}$ for $T=20$.

**常見迷思**：

- 「解碼器可以隨時回頭看原句。」錯，它只看得到 $\mathbf{c}$ 和自己寫過的詞。
- 「把隱藏狀態加大就能解決長句問題。」錯，只要 $\mathbf{c}$ 還是最後一步的狀態，早期的詞照樣被連乘稀釋。
- 「雙向 RNN 徹底解決了瓶頸。」錯，最後仍是一個固定長度的向量。

**Common misconceptions**:

- "The decoder can look back at the original sentence any time." Wrong: it sees only $\mathbf{c}$ and the words it has written.
- "Enlarging the hidden state fixes long sentences." Wrong: as long as $\mathbf{c}$ is the last state, early words are still diluted by repeated multiplication.
- "A bidirectional RNN removes the bottleneck." Wrong: in the end it is still one fixed-length vector.

**圖**：本單元有兩張圖。

- `lec10-kp02-seq2seq-bottleneck`：編碼器、唯一的上下文向量、解碼器，每一步都拿同一個 $\mathbf{c}$。
- `lec10-kp02-early-words-fade`：線性編碼器裡，每個來源詞在 $\mathbf{c}$ 中的份量，越早越小。

**Figures** — this unit has two figures.

- `lec10-kp02-seq2seq-bottleneck`: encoder, the single context vector and decoder, with the same $\mathbf{c}$ at every step.
- `lec10-kp02-early-words-fade`: in the linear encoder, the weight of each source word inside $\mathbf{c}$, smaller for earlier words.

### 小練習（Practice）

下面第 1 題要分清楚兩件事：「份量」是某個詞前面乘的係數，例如 $0.25$；「占比」是它的貢獻占 $c$ 總和的比例，例如 $0.25\div1.75$。

Question 1 below separates two things: the "weight" is the coefficient in front of a word, for example $0.25$; the "share" is the fraction of the total $c$ that its contribution makes up, for example $0.25\div1.75$.

1. 用 $h_t=0.5\,h_{t-1}+x_t$、$h_0=0$，輸入三個詞 $x_1=x_2=x_3=1$。算出 $h_1$、$h_2$、$h_3$。三個詞在 $c=h_3$ 裡各占多少份量？第一個詞的貢獻占 $c$ 的幾成？
2. 同一個編碼器，句子至少要多長，第一個詞的份量 $0.5^{T-1}$ 才會小於 $0.01$？
3. 翻譯 `I am a student` 寫到 `student` 時，解碼器依靠哪三樣東西？其中哪一樣和寫 `I` 時完全相同？

1. Use $h_t=0.5\,h_{t-1}+x_t$, $h_0=0$, and three input words $x_1=x_2=x_3=1$. Compute $h_1$, $h_2$, $h_3$. What weight does each word have inside $c=h_3$? What fraction of $c$ comes from the first word?
2. With the same encoder, how long must a sentence be before the weight of the first word, $0.5^{T-1}$, drops below $0.01$?
3. When the decoder writes "student" while translating "I am a student", which three things does it rely on? Which of them is exactly the same as when it wrote "I"?

**第 1 題解答** `[補充]`：$h_1=1$，$h_2=0.5+1=1.5$，$h_3=0.75+1=1.75$。三個詞的份量是 $0.25$、$0.5$、$1$。第一個詞貢獻 $0.25$，占 $0.25/1.75=0.1429$，大約一成四。

**Solution to question 1** `[Added]`: $h_1=1$, $h_2=0.5+1=1.5$, $h_3=0.75+1=1.75$. The three weights are $0.25$, $0.5$ and $1$. The first word contributes $0.25$, which is $0.25/1.75=0.1429$, about 14 percent.

**第 2 題解答** `[補充]`：$0.5^{6}=0.015625$ 還大於 $0.01$，$0.5^{7}=0.0078125$ 才小於 $0.01$，所以 $T-1=7$，句子至少要 $8$ 個詞。

**Solution to question 2** `[Added]`: $0.5^{6}=0.015625$ is still above $0.01$, and $0.5^{7}=0.0078125$ is below it, so $T-1=7$ and the sentence must have at least $8$ words.

**第 3 題解答** `[補充]`：依靠前一個詞 `a`、自己的狀態 $\mathbf{l}_4$，以及上下文向量 $\mathbf{c}$。其中 $\mathbf{c}$ 和寫 `I` 時完全相同，這正是瓶頸所在。

**Solution to question 3** `[Added]`: it relies on the previous word "a", its own state $\mathbf{l}_4$ and the context vector $\mathbf{c}$. The vector $\mathbf{c}$ is exactly the same as when it wrote "I", which is precisely the bottleneck.

### 小測驗（Quiz，來源原題）

來源的兩題自我檢測保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The two self-check questions of the source are kept with all options; the source page does not mark the answers directly, and the explanations below follow the source text.

1. 傳統 Seq2Seq 模型（無注意力）的核心缺陷是什麼？
   - (A) 計算速度太快
   - (B) 強制將整條長序列資訊壓縮進單一固定維度的上下文向量 $\mathbf{c}=\mathbf{h}_n$，導致長距遺忘
   - (C) 無法使用 GPU 運算
   - (D) 輸出只能是單一數字
2. 傳統 Seq2Seq 解碼器在預測第 $i$ 個詞時，依賴哪三項輸入？
   - (A) 前一詞 $\mathbf{y}_{i-1}$、當前解碼器狀態 $\mathbf{l}_i$ 與上下文向量 $\mathbf{c}$
   - (B) 隨機初始向量、超參數 $\alpha$ 與學習率
   - (C) 所有未來單詞 $\mathbf{y}_{i+1},\dots,\mathbf{y}_{n}$
   - (D) 僅依賴第一時間步輸入 $\mathbf{x}_1$

1. What is the core weakness of the traditional Seq2Seq model (without attention)?
   - (A) It computes too fast
   - (B) It forces the whole long sequence into a single fixed-dimension context vector $\mathbf{c}=\mathbf{h}_n$, which causes long-range forgetting
   - (C) It cannot run on a GPU
   - (D) Its output can only be a single number
2. When the traditional Seq2Seq decoder predicts the $i$-th word, which three inputs does it rely on?
   - (A) The previous word $\mathbf{y}_{i-1}$, the current decoder state $\mathbf{l}_i$ and the context vector $\mathbf{c}$
   - (B) A random initial vector, a hyperparameter $\alpha$ and the learning rate
   - (C) All future words $\mathbf{y}_{i+1},\dots,\mathbf{y}_{n}$
   - (D) Only the first input $\mathbf{x}_1$

**解析** `[補充]`：第 1 題選 (B)：第 3 節的計算顯示，整句只剩一個 $\mathbf{c}$，早期的詞被一再稀釋。第 2 題選 (A)：就是第 2 節解碼器式子 $g(\mathbf{y}_{i-1},\mathbf{l}_i,\mathbf{c})$ 的三個輸入；(C) 不可能，因為未來的詞還沒寫出來。

**Explanations** `[Added]`: question 1 is B: the calculation of Section 3 shows that the whole sentence leaves only one $\mathbf{c}$, and early words are diluted again and again. Question 2 is A: these are the three inputs of the decoder equation $g(\mathbf{y}_{i-1},\mathbf{l}_i,\mathbf{c})$ of Section 2; C is impossible because future words have not been written yet.

### 留給下一單元的問題（The question for the next unit）

如果解碼器寫每一個詞時，都能用 KP01 的方法，依相關程度重新加權編碼器的所有隱藏狀態 $\mathbf{h}_1,\dots,\mathbf{h}_n$，而不是只拿 $\mathbf{h}_n$，瓶頸是不是就消失了？這時查詢是誰、權重怎麼算？下一單元（KP03）的「帶注意力的 Seq2Seq」回答這個問題。

If, for every word it writes, the decoder could use the method of KP01 to reweight all encoder hidden states $\mathbf{h}_1,\dots,\mathbf{h}_n$ by relevance, instead of taking only $\mathbf{h}_n$, would the bottleneck disappear? Who is the query then, and how are the weights computed? The next unit (KP03), "Seq2Seq with attention", answers this question.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第9章「注意力機制與 Transformer 核心原理」§2.1、§2.2 與教授祕笈「Auto-Encoder 架構與表徵學習」「Seq2Seq 架構與上下文向量瓶頸」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §2.1 and §2.2 of Chapter 9, "Core principles of attention mechanisms and the Transformer", of CCC-LNS IDL_115 and its professor's notes "Autoencoder architecture and representation learning" and "Seq2Seq architecture and the context-vector bottleneck" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源教授祕笈中，解碼器的機率式在網頁上斷掉了；本講義依 §2.2 正文的式子 $g(\mathbf{y}_{i-1},\mathbf{l}_i,\mathbf{c})$ 寫出。來源祕笈把解碼器狀態寫成 $s_i$、正文寫成 $\mathbf{l}_i$；本講義用 $\mathbf{l}_i$，因為 $s$ 在 KP01 已經用來表示分數。
- 來源用粗斜體 `\boldsymbol` 表示向量；本課程網站不支援，改用 `\mathbf`，意義相同。小測驗第 2 題的選項 (C) 原寫 $\mathbf{y}_{i+1\dots n}$，改寫成 $\mathbf{y}_{i+1},\dots,\mathbf{y}_{n}$，意思不變。
- 本講義補上：句長 $5$、$10$、$20$ 的份量表與長條圖、與第8週梯度消失的連結、三題小練習與解答。
- 取捨：來源 §2.1 自編碼器的三種跨領域應用（領域轉換、領域適應、領域泛化）留到第12週「自編碼器與 VAE」；教授祕笈提到的「暴露偏差」與控制理論連結，需要本週還沒有的先備知識，本講義不寫。

- In the source's professor's notes, the decoder's probability formula is cut off on the web page; this note writes it from the main-text formula $g(\mathbf{y}_{i-1},\mathbf{l}_i,\mathbf{c})$ of §2.2. The professor's notes write the decoder state as $s_i$ and the main text as $\mathbf{l}_i$; this note uses $\mathbf{l}_i$, because $s$ already stands for the scores in KP01.
- The source writes vectors with bold italic `\boldsymbol`; the course website does not support it, so this note uses `\mathbf` with the same meaning. Option C of quiz question 2 was written $\mathbf{y}_{i+1\dots n}$ and is rewritten as $\mathbf{y}_{i+1},\dots,\mathbf{y}_{n}$ with the same meaning.
- This note adds: the weight table and bar chart for lengths $5$, $10$ and $20$, the link to the vanishing gradient of Week 8, and three practice questions with solutions.
- Choices: the three cross-domain uses of the autoencoder in source §2.1 (domain transformation, adaptation and generalization) are left to Week 12, "Autoencoders and VAE"; the "exposure bias" and the link to control theory in the professor's notes need background this week does not have, so this note leaves them out.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 編碼器 | encoder |
| 解碼器 | decoder |
| 上下文向量 | context vector |
| 自編碼器（第12週） | autoencoder |
| 序列到序列模型 | sequence-to-sequence model (Seq2Seq) |
| 自回歸 | autoregressive |
| 條件機率 | conditional probability |
| 資訊瓶頸 | information bottleneck |
| 端到端 | end to end |
| 雙向循環神經網路 | bidirectional RNN |
| 過擬合 | overfitting |
| 梯度消失 | vanishing gradient |
| 帶注意力的 Seq2Seq（KP03） | Seq2Seq with attention |
| 變分自編碼器（第12週） | variational autoencoder (VAE) |
| 圖形處理器 | graphics processing unit (GPU) |
