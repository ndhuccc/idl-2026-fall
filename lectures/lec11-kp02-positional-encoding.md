# 第11週 KP02：正弦位置編碼，用快慢不同的波替每個位置蓋上指紋（Sinusoidal positional encoding: waves of different speeds stamp a fingerprint on every position）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第11週的第二個知識點（KP02），接在 KP01 之後。KP01 用數字示範：沒有位置資訊時，自注意力分不出「我打你」和「你打我」。這一節回答下一個問題：位置要怎麼寫進詞向量，模型才讀得出順序，甚至讀得出兩個詞相隔多遠？

This is the second knowledge point (KP02) of Week 11 and follows KP01. KP01 showed with numbers that without position information, self-attention cannot tell "I hit you" from "you hit me". This note answers the next question: how should position be written into the word vectors so that the model can read the order, and even how far apart two words are?

先備知識是 KP01 的編碼器流程，以及高中的正弦、餘弦與和角公式。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第9章改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec11-kp02-figures.py` 算出，可以自己重算。

The prerequisites are the encoder flow of KP01 and the high-school sine, cosine and angle-addition formulas. Every term and symbol is defined when it first appears. This note is rewritten from Chapter 9 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec11-kp02-figures.py`, so you can recompute it yourself.

## KP02 正弦位置編碼，用快慢不同的波替每個位置蓋上指紋（Sinusoidal positional encoding: waves of different speeds stamp a fingerprint on every position）

### 學習目標（Learning objectives）

讀完之後，你能說明為什麼不能直接把 $1,2,3,\dots$ 當成位置資訊。你能寫出正弦位置編碼的公式，並算出一個小維度的編碼向量。你能說明低維度變化快、高維度變化慢，以及這讓每個位置都有獨特的編碼。你也能說明兩個位置的編碼內積只和相隔距離有關，以及位置編碼是加到詞嵌入上、不是接在後面。

After reading, you can explain why $1,2,3,\dots$ cannot simply be used as position information. You can write the formula of the sinusoidal positional encoding and compute the encoding vector for a small dimension. You can explain that low dimensions change fast and high dimensions slowly, and that this gives every position a unique code. You can also explain that the inner product of two positions' codes depends only on how far apart they are, and that the positional encoding is added to the word embedding, not appended to it.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $pos$ | 詞在句子裡的位置，從 $0$ 算起 | position of a word in the sentence, counted from $0$ |
| $d$ | 位置編碼（也是詞向量）的長度，就是 KP01 的 $d_{\text{model}}$ | length of the positional encoding (and of the word vector), the $d_{\text{model}}$ of KP01 |
| $i$ | 第幾對維度，$i=0,1,\dots,d/2-1$ | index of a pair of dimensions, $i=0,1,\dots,d/2-1$ |
| $\mathbf{p}_{pos}\in\mathbb{R}^{d}$ | 位置 $pos$ 的編碼向量 | encoding vector of position $pos$ |
| $\omega_i$ | 第 $i$ 對維度的角頻率：位置每加 $1$，角度轉多少弧度 | angular frequency of pair $i$: how many radians the angle turns per position |
| $\lambda_i$ | 第 $i$ 對維度的波長，$\lambda_i=2\pi/\omega_i$ | wavelength of pair $i$, $\lambda_i=2\pi/\omega_i$ |
| $k$ | 兩個位置相隔的距離 | offset between two positions |
| $\mathbf{x}_{pos}$ | 位置 $pos$ 的詞嵌入 | word embedding at position $pos$ |

### 1. 位置要寫進向量，但不能直接寫 1、2、3（Position must go into the vector, but not as 1, 2, 3）

KP01 說，進入編碼器前要把「第幾個位置」寫進每個詞的向量。最直覺的做法是多放一個數字，第一個詞寫 $1$、第二個寫 $2$，依此類推。可是一篇文章可能有幾千個詞，這個數字會長到幾千。其他分量大多在 $-1$ 到 $1$ 之間，這個超大的數會蓋過詞本身的意思；它乘上權重後，輸出和梯度也跟著變得很大，訓練不穩。

KP01 said that "which position" must be written into each word's vector before the encoder. The most obvious way is to add one number: $1$ for the first word, $2$ for the second, and so on. But a document can have thousands of words, so this number grows into the thousands. Most other components lie between $-1$ and $1$, so such a huge number drowns out the word's meaning; multiplied by the weights, it also makes the outputs and gradients very large, so training becomes unstable.

那除以句子長度，壓到 $0$ 到 $1$ 之間呢？這又帶來另一個問題：同樣是「下一個詞」，在 $5$ 個詞的句子裡差 $1/5=0.2$，在 $50$ 個詞的句子裡只差 $1/50=0.02$。同一段距離在不同句子裡代表不同的數字，模型很難學。

What about dividing by the sentence length, squeezing it into $0$ to $1$? That brings another problem: "the next word" differs by $1/5=0.2$ in a $5$-word sentence but by only $1/50=0.02$ in a $50$-word sentence. The same distance means different numbers in different sentences, which is hard for the model to learn.

所以好的位置編碼要同時做到三件事。第一，數值有上下界，不會隨句長爆炸。第二，每個位置的編碼都不一樣。第三，相隔同樣距離的兩個位置，不論在句子哪裡，關係都一樣。下一節的正弦位置編碼三件都做到。

So a good positional encoding must do three things at once. First, its values are bounded and do not explode with the sentence length. Second, every position gets a different code. Third, two positions the same distance apart relate in the same way wherever they are in the sentence. The sinusoidal positional encoding of the next section achieves all three.

常見錯誤：以為位置編碼是讓模型「學」順序的參數。原論文的正弦位置編碼是用公式直接算出來的，沒有要學的參數。也有模型改用可學習的位置向量；原論文試過兩種，結果幾乎一樣 `[補充]`。

Common mistake: thinking the positional encoding is a parameter the model "learns" to capture order. The sinusoidal encoding of the original paper is computed directly by a formula and has no parameters to learn. Some models use learnable position vectors instead; the original paper tried both and found almost the same results `[Added]`.

### 2. 正弦位置編碼：每一對維度是一個轉速不同的指針（The sinusoidal encoding: each pair of dimensions is a hand turning at its own speed）

原論文的做法是：把 $d$ 個維度兩兩分成一對，第 $i$ 對用同一個頻率 $\omega_i$，偶數維放正弦、奇數維放餘弦：

The original paper's recipe: split the $d$ dimensions into pairs, give pair $i$ one frequency $\omega_i$, and put a sine in the even dimension and a cosine in the odd one:

$$
\mathbf{p}_{pos}(2i)=\sin(\omega_i\,pos),\qquad \mathbf{p}_{pos}(2i+1)=\cos(\omega_i\,pos),\qquad \omega_i=\frac{1}{10000^{2i/d}}.
$$

第 $0$ 對的頻率是 $1$，越後面的對頻率越小，變化越慢。波長是 $\lambda_i=2\pi\times10000^{2i/d}$：第 $0$ 對只有 $2\pi\approx6.28$ 個位置就轉一圈；$d$ 很大時（例如 $d=512$），最後一對要約六萬個位置才轉一圈。維度從 $0$ 開始編號，第 $2i$ 與第 $2i+1$ 維屬於第 $i$ 對。

Pair $0$ has frequency $1$; later pairs have smaller frequencies and change more slowly. The wavelength is $\lambda_i=2\pi\times10000^{2i/d}$: pair $0$ completes a cycle every $2\pi\approx6.28$ positions, while for large $d$ (for example $d=512$) the last pair needs about sixty thousand positions. Dimensions are numbered from $0$, and dimensions $2i$ and $2i+1$ form pair $i$.

例子（來源的數字）：$d=4$，所以有兩對，$\omega_0=1$、$\omega_1=1/10000^{2/4}=1/100=0.01$。位置 $pos=1$ 的編碼是：

Example (the source's numbers): $d=4$, so there are two pairs, $\omega_0=1$ and $\omega_1=1/10000^{2/4}=1/100=0.01$. The code of position $pos=1$ is:

$$
\mathbf{p}_1=\left[\sin 1,\ \cos 1,\ \sin 0.01,\ \cos 0.01\right]^\top=[0.8415,\ 0.5403,\ 0.0100,\ 0.99995]^\top.
$$

位置 $0$ 的編碼是 $[0,\ 1,\ 0,\ 1]^\top$ `[補充]`。前兩維變化很大，後兩維幾乎沒動，這就是「快慢不同」。所有值都在 $-1$ 到 $1$ 之間，第一個要求達成。

The code of position $0$ is $[0,\ 1,\ 0,\ 1]^\top$ `[Added]`. The first two dimensions change a lot while the last two barely move: that is "different speeds". Every value lies between $-1$ and $1$, so the first requirement is met.

![左圖是 50 列、64 行的紅藍色熱圖，每一列是一個位置的編碼，左邊幾行紅藍快速交錯，越往右顏色變化越慢，右半部幾乎整行同色。右圖畫三條正弦曲線：紅色第 0 維波長 6.3，快速上下震盪；藍色第 10 維波長 26.5，慢慢起伏；綠色第 40 維波長 1986.9，在 50 個位置內幾乎是一條緩升的直線（Left: a 50-by-64 red-blue heat map, one row per position; the left columns flip colour quickly and the right half barely changes. Right: three sine curves; red dimension 0 with wavelength 6.3 oscillates fast, blue dimension 10 with wavelength 26.5 rises and falls slowly, and green dimension 40 with wavelength 1986.9 is almost a slowly rising line over 50 positions）](../assets/figures/lec11-kp02-pe-waves.png)

**圖說 `lec11-kp02-pe-waves`**〔功能：一目了然、精準度〕：這是 $d=64$ 的例子 `[補充]`。先看左圖的一列：它就是某個位置的編碼，從左到右由快到慢。再沿著一行往下看：左邊的行紅藍快速交錯，右邊的行幾乎不變。最後看右圖：三條曲線是三個維度隨位置變化的樣子，第 $0$、$10$、$40$ 維分別屬於第 $0$、$5$、$20$ 對，波長從 $6.3$、$26.5$ 一路到 $1986.9$。

**Caption `lec11-kp02-pe-waves`** [Function: seeing it at a glance, precision]: this is a $d=64$ example `[Added]`. First read one row of the left panel: it is one position's code, fast on the left and slow on the right. Then read down one column: the left columns flip red and blue quickly while the right columns hardly change. Finally look at the right panel: the three curves show how three dimensions change with position, dimensions $0$, $10$ and $40$ belong to pairs $0$, $5$ and $20$, with wavelengths from $6.3$ and $26.5$ up to $1986.9$.

來源的比喻是時鐘：秒針轉得最快，分辨相鄰的位置；時針轉得最慢，分辨很遠的位置。好幾根轉速不同的指針合起來，每個時刻的組合都不一樣，第二個要求也達成。這個比喻的侷限是：時鐘每 $12$ 小時會整個重複一次，而位置編碼的頻率不是整數倍關係，在實際用得到的長度內不會重複。

The source's analogy is a clock: the second hand turns fastest and tells neighbouring positions apart; the hour hand turns slowest and tells distant positions apart. Several hands at different speeds together give a different combination at every moment, so the second requirement is met too. The limitation of this analogy is that a clock repeats exactly every $12$ hours, while the frequencies of the positional encoding are not integer multiples of one another, so the code does not repeat within any length used in practice.

常見錯誤：以為第 $i$ 維就是第 $i$ 個位置。$i$ 是維度（向量裡的第幾個數），$pos$ 才是位置（句子裡的第幾個詞）；每個位置都有完整的 $d$ 個數。

Common mistake: thinking dimension $i$ is position $i$. $i$ indexes a dimension (which number in the vector), while $pos$ is the position (which word in the sentence); every position has all $d$ numbers.

### 3. 相隔的距離藏在內積裡（The distance is hidden in the inner product）

第三個要求是：相隔 $k$ 的兩個位置，不論在哪裡，關係都一樣。看同一對維度。位置 $pos$ 這一對是 $(\sin\omega pos,\ \cos\omega pos)$，位置 $pos+k$ 是 $(\sin\omega(pos+k),\ \cos\omega(pos+k))$。兩者相乘再相加，用和角公式的差角形式 $\cos(a-b)=\cos a\cos b+\sin a\sin b$：

The third requirement is that two positions $k$ apart relate in the same way wherever they are. Look at one pair of dimensions. At position $pos$ the pair is $(\sin\omega pos,\ \cos\omega pos)$, and at position $pos+k$ it is $(\sin\omega(pos+k),\ \cos\omega(pos+k))$. Multiply and add them, using the difference form of the angle-addition formula, $\cos(a-b)=\cos a\cos b+\sin a\sin b$:

$$
\sin(\omega pos)\sin(\omega(pos+k))+\cos(\omega pos)\cos(\omega(pos+k))=\cos(\omega k).
$$

右邊只有 $k$，沒有 $pos$。把每一對加起來，整個內積就是 $\mathbf{p}_{pos}^\top\mathbf{p}_{pos+k}=\sum_{i}\cos(\omega_i k)$，也只和距離 $k$ 有關。特別地，$k=0$ 時每一項都是 $1$，所以 $\lVert\mathbf{p}_{pos}\rVert^2=d/2$，每個位置的編碼長度都一樣。

The right side contains only $k$, not $pos$. Summing over all pairs, the whole inner product is $\mathbf{p}_{pos}^\top\mathbf{p}_{pos+k}=\sum_{i}\cos(\omega_i k)$, which also depends only on the distance $k$. In particular, at $k=0$ every term is $1$, so $\lVert\mathbf{p}_{pos}\rVert^2=d/2$: every position's code has the same length.

用數字核對 `[補充]`：$d=4$、$k=4$ 時，從位置 $0$、$10$、$37$ 出發，內積都是 $0.3456$。$d=64$ 時都是 $23.9344$。**餘弦相似度**（cosine similarity）是內積除以兩個長度的乘積；因為每個編碼的長度平方都是 $d/2$，這裡就是內積除以 $d/2$。來源的互動示範取 $d=64$、位置 $10$ 與 $14$，餘弦相似度是 $23.9344/32=0.7479$。

Check it with numbers `[Added]`: with $d=4$ and $k=4$, starting from position $0$, $10$ or $37$, the inner product is always $0.3456$. With $d=64$ it is always $23.9344$. **Cosine similarity** is the inner product divided by the product of the two lengths; since every code has squared length $d/2$, here it is the inner product divided by $d/2$. The source's interactive demo takes $d=64$ and positions $10$ and $14$; their cosine similarity is $23.9344/32=0.7479$.

![紫色曲線是 d=64 時，位置 0 與位置 k 的編碼餘弦相似度，從 k=0 的 1 開始下降，到 k=60 左右約 0.5，中間帶著小波紋。紅色空心圓是從位置 10 出發、綠色空心方塊是從位置 37 出發的同一種相似度，每隔 5 取一點，都剛好落在紫色曲線上（Purple curve: cosine similarity between the codes of position 0 and position k for d = 64, starting at 1 for k = 0 and falling to about 0.5 near k = 60 with small ripples. Red open circles start from position 10 and green open squares from position 37, sampled every 5; they all sit exactly on the purple curve）](../assets/figures/lec11-kp02-relative-offset.png)

**圖說 `lec11-kp02-relative-offset`**〔功能：精準度〕：先看紫色曲線：兩個位置離得越遠，相似度大致越低，但帶著小波紋，不是平滑下降。再看紅圈與綠方塊：它們從別的位置出發，卻剛好落在紫色曲線上，紅圈就套在綠方塊外面。這說明相似度只看距離 $k$，不看從哪裡開始。

**Caption `lec11-kp02-relative-offset`** [Function: precision]: first look at the purple curve: the farther apart two positions are, the lower their similarity tends to be, but with small ripples rather than a smooth fall. Then look at the red circles and green squares: they start from other positions yet sit exactly on the purple curve, with each red circle drawn around a green square. This shows that the similarity depends only on the distance $k$, not on where you start.

還有一個更強的性質。對每一對維度，從位置 $pos$ 走到 $pos+k$，等於把 $(\sin,\cos)$ 這個點轉一個固定角度 $\omega_i k$。對第 $i$ 對來說，就是

$$
\begin{bmatrix}\sin\omega_i(pos+k)\\ \cos\omega_i(pos+k)\end{bmatrix}=\begin{bmatrix}\cos\omega_i k&\sin\omega_i k\\ -\sin\omega_i k&\cos\omega_i k\end{bmatrix}\begin{bmatrix}\sin\omega_i pos\\ \cos\omega_i pos\end{bmatrix}.
$$

把每一對的這個 $2\times2$ 矩陣排在對角線上，就得到只和 $k$ 有關的旋轉矩陣 $\mathbf{R}_k$，使 $\mathbf{p}_{pos+k}=\mathbf{R}_k\mathbf{p}_{pos}$。原論文推測，這讓模型容易學會「往後 $k$ 個位置」這種相對關係。注意這些性質只描述 $\mathbf{p}$ 本身；加上詞嵌入、乘上查詢與鍵的矩陣以後，注意力分數就不再嚴格只看 $k$。

There is an even stronger property. For each pair of dimensions, moving from position $pos$ to $pos+k$ rotates the point $(\sin,\cos)$ by a fixed angle $\omega_i k$. For pair $i$ this reads as the $2\times2$ equation above. Placing each pair's $2\times2$ matrix on the diagonal gives a rotation matrix $\mathbf{R}_k$ that depends only on $k$, with $\mathbf{p}_{pos+k}=\mathbf{R}_k\mathbf{p}_{pos}$. The original paper hypothesised that this makes relative relations such as "$k$ positions later" easy for the model to learn. Note that these properties describe $\mathbf{p}$ alone; once the word embedding is added and the query and key matrices are applied, the attention score no longer depends on $k$ alone.

常見錯誤：以為相似度會隨距離嚴格遞減。圖上的曲線有波紋，例如 $k=25$ 比 $k=20$ 還高一點；大方向是越遠越不像，但不是一路平滑下降。

Common mistake: thinking the similarity decreases strictly with distance. The curve in the figure has ripples; for example, $k=25$ is a little higher than $k=20$. The general trend is "farther means less alike", but it does not fall smoothly all the way.

### 4. 加上去，不是接上去；也不能無限延長（Add it, do not append it; and it cannot be stretched forever）

位置編碼怎麼和詞嵌入合在一起？原論文是直接**相加**：$\mathbf{x}_{pos}\leftarrow\mathbf{x}_{pos}+\mathbf{p}_{pos}$，兩者長度都是 $d$。相加不增加維度，後面每一層的形狀都不用改，這正是 KP01 說的「形狀不變才能疊」。

How is the positional encoding combined with the word embedding? The original paper simply **adds** them: $\mathbf{x}_{pos}\leftarrow\mathbf{x}_{pos}+\mathbf{p}_{pos}$, both of length $d$. Adding does not increase the dimension, so the shape of every later layer stays the same; this is exactly KP01's "an unchanged shape is what allows stacking".

詞的意思和位置會不會互相干擾？會有一點，但詞嵌入是訓練出來的，模型可以學著把詞的資訊放在不太受位置影響的方向上。實際上相加的效果很好，所以原論文採用它。

Do the word's meaning and its position interfere with each other? A little, but the embeddings are learned, and the model can learn to put word information in directions that the position barely affects. In practice adding works well, which is why the original paper uses it.

另一個限制是長度。公式可以算任何 $pos$，但模型只在訓練時見過的長度範圍內學會讀這些編碼。遇到比訓練時長很多的句子，表現仍會變差。後來的大型語言模型常改用**旋轉位置編碼**（rotary position embedding，RoPE），把旋轉直接作用在查詢與鍵上，讓相對位置直接進入查詢與鍵的分數；再加上一些延伸技巧，才能撐到更長的文字 `[補充]`。

Another limitation is length. The formula works for any $pos$, but the model only learns to read these codes within the lengths seen in training. On sentences much longer than in training, performance still drops. Later large language models often switch to **rotary position embedding** (RoPE), which applies the rotation directly to the queries and keys so that relative position enters the query–key score itself; with further extensions it can be stretched to longer text `[Added]`.

常見錯誤：以為位置編碼是接在詞嵌入後面（拼接）。原論文是逐元素相加；拼接會讓向量變長，後面所有矩陣都得跟著改。

Common mistake: thinking the positional encoding is appended to the embedding (concatenation). The original paper adds them element by element; concatenation would lengthen the vector and every later matrix would have to change.

### 本單元重點整理（KP02 summary）

**一句話重點**：正弦位置編碼把 $d$ 個維度分成 $d/2$ 對，第 $i$ 對用頻率 $\omega_i=1/10000^{2i/d}$ 的正弦與餘弦，數值有界、每個位置不同，而且兩個位置的內積只和相隔距離 $k$ 有關；它直接加到詞嵌入上。

**One-sentence summary**: the sinusoidal positional encoding splits the $d$ dimensions into $d/2$ pairs and gives pair $i$ a sine and a cosine with frequency $\omega_i=1/10000^{2i/d}$; its values are bounded, every position differs, and the inner product of two positions depends only on their distance $k$; it is added directly to the word embedding.

**白話說明**：每個位置配一個好幾根指針的時鐘，快的指針分辨鄰居，慢的指針分辨遠方；兩個時刻差多少，看指針轉了多少角度就知道，跟現在幾點無關。

**Plain explanation**: every position gets a clock with several hands; the fast hands tell neighbours apart and the slow hands tell distant places apart; how far apart two moments are can be read from how much the hands have turned, no matter what time it is now.

**正式定義**：$\mathbf{p}_{pos}(2i)=\sin(\omega_i pos)$，$\mathbf{p}_{pos}(2i+1)=\cos(\omega_i pos)$，$\omega_i=1/10000^{2i/d}$；$\mathbf{p}_{pos}^\top\mathbf{p}_{pos+k}=\sum_i\cos(\omega_i k)$；輸入為 $\mathbf{x}_{pos}+\mathbf{p}_{pos}$。

**Formal definition**: $\mathbf{p}_{pos}(2i)=\sin(\omega_i pos)$, $\mathbf{p}_{pos}(2i+1)=\cos(\omega_i pos)$, $\omega_i=1/10000^{2i/d}$; $\mathbf{p}_{pos}^\top\mathbf{p}_{pos+k}=\sum_i\cos(\omega_i k)$; the input is $\mathbf{x}_{pos}+\mathbf{p}_{pos}$.

**例子**：$d=4$：$\mathbf{p}_0=[0,1,0,1]^\top$，$\mathbf{p}_1=[0.8415,\ 0.5403,\ 0.0100,\ 0.99995]^\top$；相隔 $4$ 的兩位置內積恆為 $0.3456$。$d=64$：位置 $10$ 與 $14$ 的餘弦相似度 $0.7479$。

**Example**: $d=4$: $\mathbf{p}_0=[0,1,0,1]^\top$, $\mathbf{p}_1=[0.8415,\ 0.5403,\ 0.0100,\ 0.99995]^\top$; the inner product of two positions $4$ apart is always $0.3456$. $d=64$: the cosine similarity of positions $10$ and $14$ is $0.7479$.

**常見迷思**：

- 「位置編碼是接在詞嵌入後面。」錯，原論文是相加。
- 「正弦位置編碼可以無限延長，長度多少都沒問題。」錯，超出訓練長度時表現仍會變差。
- 「相似度隨距離嚴格遞減。」錯，大致遞減，但帶著波紋。

**Common misconceptions**:

- "The positional encoding is appended to the word embedding." Wrong: the original paper adds them.
- "The sinusoidal encoding can be stretched to any length without trouble." Wrong: performance still drops beyond the training length.
- "The similarity decreases strictly with distance." Wrong: it decreases roughly, with ripples.

**圖**：本單元有兩張圖。

- `lec11-kp02-pe-waves`：$d=64$ 的位置編碼矩陣，與三個維度隨位置變化的曲線。
- `lec11-kp02-relative-offset`：兩個位置的相似度只看相隔距離，不看從哪裡出發。

**Figures** — this unit has two figures.

- `lec11-kp02-pe-waves`: the $d=64$ positional-encoding matrix and the curves of three dimensions against position.
- `lec11-kp02-relative-offset`: the similarity of two positions depends only on their distance, not on where they start.

### 小練習（Practice）

1. 取 $d=4$。算出位置 $pos=2$ 的編碼 $\mathbf{p}_2$，保留四位小數。
2. 取 $d=512$。第 $0$ 對與最後一對（$i=255$）的波長各是多少？
3. 不用計算機，說明為什麼 $\mathbf{p}_{10}^\top\mathbf{p}_{14}$ 會等於 $\mathbf{p}_{0}^\top\mathbf{p}_{4}$。

1. Take $d=4$. Compute the code $\mathbf{p}_2$ of position $pos=2$ to four decimals.
2. Take $d=512$. What are the wavelengths of pair $0$ and of the last pair ($i=255$)?
3. Without a calculator, explain why $\mathbf{p}_{10}^\top\mathbf{p}_{14}$ equals $\mathbf{p}_{0}^\top\mathbf{p}_{4}$.

**第 1 題解答** `[補充]`：$\mathbf{p}_2=[\sin 2,\ \cos 2,\ \sin 0.02,\ \cos 0.02]^\top=[0.9093,\ -0.4161,\ 0.0200,\ 0.9998]^\top$。

**Solution to question 1** `[Added]`: $\mathbf{p}_2=[\sin 2,\ \cos 2,\ \sin 0.02,\ \cos 0.02]^\top=[0.9093,\ -0.4161,\ 0.0200,\ 0.9998]^\top$.

**第 2 題解答** `[補充]`：第 $0$ 對 $\lambda_0=2\pi=6.2832$；最後一對 $\lambda_{255}=2\pi\times10000^{510/512}=60611.5$，接近上限 $2\pi\times10000=62831.9$。

**Solution to question 2** `[Added]`: pair $0$ has $\lambda_0=2\pi=6.2832$; the last pair has $\lambda_{255}=2\pi\times10000^{510/512}=60611.5$, close to the limit $2\pi\times10000=62831.9$.

**第 3 題解答** `[補充]`：每一對維度的乘積和都等於 $\cos(\omega_i k)$，只和距離 $k$ 有關；兩組位置都相隔 $k=4$，所以內積相同。

**Solution to question 3** `[Added]`: for each pair of dimensions the sum of products equals $\cos(\omega_i k)$, which depends only on the distance $k$; both pairs of positions are $k=4$ apart, so the inner products are equal.

### 小測驗（Quiz，來源原題）

來源的兩題自我檢測保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The source's two self-check questions are kept with all their options; the source page does not mark the answers, so the explanations below are judged from the source text.

1. 為什麼 Transformer 必須引入位置編碼（Positional Encoding）？
   - (A) 因為自注意力運算先天具有置換不變性，無法主動感知詞語前後順序
   - (B) 為了增加矩陣乘法的運算量
   - (C) 為了將英文字母轉換為大寫
   - (D) 為了替換所有的 Softmax 函數
2. 正弦位置編碼中，偶數維度與奇數維度分別採用什麼三角函數？
   - (A) 偶數用正切 $\tan$，奇數用餘切 $\cot$
   - (B) 偶數用正弦 $\sin$，奇數用餘弦 $\cos$
   - (C) 兩者均使用指數函數 $\exp$
   - (D) 偶數用常數 $0$，奇數用常數 $1$

1. Why must the Transformer introduce a positional encoding?
   - (A) Because self-attention is inherently permutation invariant and cannot sense word order by itself
   - (B) To increase the amount of matrix multiplication
   - (C) To convert English letters to upper case
   - (D) To replace all softmax functions
2. In the sinusoidal positional encoding, which trigonometric functions are used for the even and odd dimensions?
   - (A) Tangent $\tan$ for even, cotangent $\cot$ for odd
   - (B) Sine $\sin$ for even, cosine $\cos$ for odd
   - (C) The exponential $\exp$ for both
   - (D) The constant $0$ for even, the constant $1$ for odd

**解析** `[補充]`：第 1 題選 (A)：KP01 第3節的數字示範就是這件事；精確地說，自注意力是「置換等變」，每個詞的新表示不受位置影響。第 2 題選 (B)：就是第2節的公式。

**Explanations** `[Added]`: question 1 is A: this is the numerical demonstration in Section 3 of KP01; precisely, self-attention is "permutation equivariant", so each word's new representation is unaffected by its position. Question 2 is B: it is the formula of Section 2.

### 留給下一單元的問題（The question for the next unit）

有了位置編碼，自注意力終於分得出順序。可是一組查詢、鍵、值只能算出一組權重，只能看一種「相關」。一句話裡，主詞和動詞的關係、代名詞指誰、形容詞修飾誰，是好幾種不同的關係。能不能讓模型同時用好幾組查詢、鍵、值，各看一種關係？下一單元（本週 KP03）的多頭注意力回答這個問題。

With the positional encoding, self-attention can finally tell the order. But one set of queries, keys and values computes only one set of weights, so it sees only one kind of "relatedness". In a sentence, subject–verb links, what a pronoun refers to and what an adjective modifies are several different relations. Could the model use several sets of queries, keys and values at once, each watching one relation? The multi-head attention of the next unit (this week's KP03) answers this.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第9章「注意力機制與 Transformer 核心原理」§3.2.1 與教授祕笈「週期性三角函數位置編碼」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html ，讀取日期 2026-11-13）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §3.2.1 of Chapter 9, "Core principles of attention mechanisms and the Transformer", of CCC-LNS IDL_115, and from the professor's notes "Periodic trigonometric positional encoding" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html , accessed 2026-11-13). Content that the source does not contain is marked `[Added]` where it appears.

- 來源正文的公式寫 $10000^{2j/p}$，其中 $p$ 就是詞向量長度；本講義依祕笈統一寫成 $d$，並把位置記為 $pos$、維度對記為 $i$。
- 來源說自注意力有「置換不變性」；精確說法是置換等變（見 KP01 第3節），本講義在小測驗解析註明。
- 來源的互動示範說相似度「單調衰減伴隨微幅波紋」；本講義的計算顯示它是大致下降、但不單調（例如 $k=25$ 高於 $k=20$），第3節改寫。
- 來源祕笈說相加「相當於在正交子空間上疊加坐標」；這需要額外假設，本講義改說「模型可以學著把詞的資訊放在不太受位置影響的方向」。
- 來源用粗斜體 `\boldsymbol` 表示向量；本課程網站不支援，改用 `\mathbf`。
- 本講義補上：位置 $0$ 的編碼、$d=64$ 的熱圖與曲線、內積只和 $k$ 有關的推導與數字核對、三題小練習與解答。
- 取捨：旋轉位置編碼（RoPE）只點名，不展開；傅立葉分析與全像攝影的連結不寫。

- The source's main-text formula writes $10000^{2j/p}$, where $p$ is the word-vector length; following the professor's notes, this note writes $d$ throughout, with $pos$ for the position and $i$ for the pair of dimensions.
- The source says self-attention is "permutation invariant"; precisely it is permutation equivariant (see Section 3 of KP01), which this note points out in the quiz explanation.
- The source's interactive demo says the similarity "decays monotonically with small ripples"; this note's calculation shows a rough but not monotonic decline (for example, $k=25$ is higher than $k=20$), and Section 3 is written accordingly.
- The source's notes say adding is "equivalent to stacking coordinates in an orthogonal subspace"; this needs extra assumptions, so this note says instead that "the model can learn to put word information in directions that the position barely affects".
- The source writes vectors with bold italic `\boldsymbol`; the course website does not support it, so this note uses `\mathbf`.
- This note adds: the code of position $0$, the $d=64$ heat map and curves, the derivation and numerical check that the inner product depends only on $k$, and three practice questions with solutions.
- Choices: rotary position embedding (RoPE) is only named, not developed; the links to Fourier analysis and holography are not covered.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 位置編碼 | positional encoding |
| 正弦位置編碼 | sinusoidal positional encoding |
| 角頻率 | angular frequency |
| 波長 | wavelength |
| 詞嵌入 | word embedding |
| 和角公式 | angle-addition formula |
| 旋轉矩陣 | rotation matrix |
| 餘弦相似度 | cosine similarity |
| 可學習的位置向量 | learned position embedding |
| 置換等變（KP01） | permutation equivariance |
| 拼接 | concatenation |
| 旋轉位置編碼 | rotary position embedding (RoPE) |
