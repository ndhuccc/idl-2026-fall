# 第12週 KP01：自編碼器，把資料壓進瓶頸再還原（Autoencoders: squeeze the data through a bottleneck and rebuild it）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第12週「深度生成模型（一）：自編碼器與 VAE」的第一個知識點（KP01）。前十一週的模型幾乎都需要標籤：分類要知道類別，翻譯要有對照的譯文。這一節換一個問題：沒有標籤時，網路能不能自己學出有用的特徵？自編碼器的答案是「出一道還原自己的題目」。

This is the first knowledge point (KP01) of Week 12, "Deep generative models (1): autoencoders and VAE". Almost every model of the first eleven weeks needed labels: classification needs the classes, translation needs paired sentences. This note asks a different question: without labels, can a network learn useful features on its own? The autoencoder's answer is to set itself the task of rebuilding its own input.

先備知識是第2週的多層感知器與損失函數、第3週的反向傳播，以及第1週 KP07 的主成分分析（PCA）。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第9章 §2.1 與第12章 §3 改寫，文中「來源」就是指它們；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec12-kp01-figures.py` 算出，可以自己重算。

The prerequisites are the multilayer perceptron and loss functions of Week 2, backpropagation of Week 3, and principal component analysis (PCA) from Week 1 KP07. Every term and symbol is defined when it first appears. This note is rewritten from §2.1 of Chapter 9 and §3 of Chapter 12 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec12-kp01-figures.py`, so you can recompute it yourself.

## KP01 自編碼器，把資料壓進瓶頸再還原（Autoencoders: squeeze the data through a bottleneck and rebuild it）

### 學習目標（Learning objectives）

讀完之後，你能畫出自編碼器的三個部分：編碼器、瓶頸裡的編碼、解碼器，並寫出它的重建損失。你能用小數字算出一次編碼與重建，以及它的重建誤差。你能說明瓶頸為什麼不能太大也不能太小，以及線性自編碼器和主成分分析的關係。你也能說出一般自編碼器為什麼不能直接拿來生成新資料，這是接下來幾個知識點與 VAE 要解決的問題。

After reading, you can draw the three parts of an autoencoder, the encoder, the code in the bottleneck and the decoder, and write its reconstruction loss. You can compute one encoding and reconstruction with small numbers, together with its reconstruction error. You can explain why the bottleneck must be neither too wide nor too narrow, and how a linear autoencoder relates to principal component analysis. You can also explain why an ordinary autoencoder cannot be used directly to generate new data, which is the problem the next knowledge points and the VAE solve.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{x}\in\mathbb{R}^d$ | 一筆輸入資料，長度 $d$ | one input example, of length $d$ |
| $\mathbf{z}\in\mathbb{R}^k$ | 編碼：資料被壓縮後的短向量，長度 $k$ | code: the short vector the data is compressed into, of length $k$ |
| $k$ | 瓶頸的寬度，也就是編碼的長度 | width of the bottleneck, that is, the length of the code |
| $f(\cdot)$ | 編碼器：把 $\mathbf{x}$ 變成 $\mathbf{z}$ | encoder: turns $\mathbf{x}$ into $\mathbf{z}$ |
| $g(\cdot)$ | 解碼器：把 $\mathbf{z}$ 變回 $\hat{\mathbf{x}}$ | decoder: turns $\mathbf{z}$ back into $\hat{\mathbf{x}}$ |
| $\hat{\mathbf{x}}\in\mathbb{R}^d$ | 重建：解碼器的輸出 | reconstruction: the output of the decoder |
| $\mathbf{W}_e,\ \mathbf{b}_e$ | 編碼器的權重與偏差 | weights and bias of the encoder |
| $\mathbf{W}_d,\ \mathbf{b}_d$ | 解碼器的權重與偏差 | weights and bias of the decoder |
| $\sigma(\cdot)$ | 活化函數（第2週） | activation function (Week 2) |
| $\mathcal{L}$ | 重建損失 | reconstruction loss |

### 1. 出一道「還原自己」的題目（Set the task of rebuilding the input）

標註資料很貴。一百萬張照片要請人一張張標類別，花費驚人 `[補充：例子]`；可是沒有標籤的照片，網路上要多少有多少。所以我們想問：只用沒有標籤的資料，能不能讓網路學出好的特徵？

Labelled data is expensive. Having people label a million photos one by one costs a fortune `[Added: example]`, yet unlabelled photos are everywhere online. So we ask: using only unlabelled data, can a network learn good features?

**自編碼器**（autoencoder）的作法是自己出題：把輸入當成答案，要網路先把輸入壓縮，再還原回來。它有三個部分。**編碼器**（encoder）$f$ 把長度 $d$ 的輸入 $\mathbf{x}$ 變成長度 $k$ 的短向量 $\mathbf{z}$，叫做**編碼**（code）。中間這條窄路叫**瓶頸**（bottleneck），通常 $k$ 比 $d$ 小很多。**解碼器**（decoder）$g$ 再把 $\mathbf{z}$ 變回長度 $d$ 的 $\hat{\mathbf{x}}$。

The **autoencoder** sets its own task: it takes the input as the answer and asks the network to compress the input and then rebuild it. It has three parts. The **encoder** $f$ turns the input $\mathbf{x}$ of length $d$ into a short vector $\mathbf{z}$ of length $k$, called the **code**. This narrow passage is the **bottleneck**, and usually $k$ is much smaller than $d$. The **decoder** $g$ then turns $\mathbf{z}$ back into $\hat{\mathbf{x}}$ of length $d$.

來源寫出一層編碼器與一層解碼器的式子，損失是**重建誤差**（reconstruction error），也就是輸入與重建之差的平方和：

The source writes one-layer formulas for the encoder and the decoder, and the loss is the **reconstruction error**, the sum of squared differences between the input and its reconstruction:

$$
\mathbf{z}=f(\mathbf{x})=\sigma(\mathbf{W}_e\mathbf{x}+\mathbf{b}_e),\qquad
\hat{\mathbf{x}}=g(\mathbf{z})=\sigma(\mathbf{W}_d\mathbf{z}+\mathbf{b}_d),\qquad
\mathcal{L}=\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2 .
$$

訓練時不需要任何標籤，因為目標就是輸入自己。反向傳播（第3週）照常使用：從 $\mathcal{L}$ 對解碼器和編碼器的權重求梯度，一起更新。實際的網路常常每邊有好幾層；輸入是 $0$ 到 $1$ 的像素時，損失也可以換成第2週的二元交叉熵。

Training needs no labels, because the target is the input itself. Backpropagation (Week 3) works as usual: take the gradients of $\mathcal{L}$ with respect to the weights of the decoder and the encoder, and update them together. Real networks often have several layers on each side; when the inputs are pixels between $0$ and $1$, the loss can also be the binary cross-entropy of Week 2.

來源的白話比喻：給你一張高解析度的大頭照，只准用一張寫了 $10$ 個數字的小紙條記下它，再讓朋友只看紙條把臉畫回來。為了畫得像，這 $10$ 個數字必須記下最重要的特徵，例如臉型與膚色，而丟掉背景雜訊。這個比喻的侷限是 `[補充]`：網路學到的編碼不一定對應「臉型」這種人看得懂的特徵，它只保證重建得好。

The source's plain analogy: you get a high-resolution portrait and may record it only on a slip of paper holding $10$ numbers, then a friend must draw the face back from the slip alone. To draw it well, the $10$ numbers must capture the most important features, such as face shape and skin tone, and drop the background noise. The limitation of this analogy `[Added]` is that the code the network learns need not match features a person can name, like "face shape"; it only guarantees a good reconstruction.

來源的數字例子：輸入 $\mathbf{x}=[1,2,1]^\top$，$d=3$、$k=1$，不用活化函數也不用偏差。編碼權重 $\mathbf{W}_e=[0.5,\ 0.5,\ 0.5]$，所以

The source's numerical example: input $\mathbf{x}=[1,2,1]^\top$, $d=3$, $k=1$, with no activation and no bias. The encoder weights are $\mathbf{W}_e=[0.5,\ 0.5,\ 0.5]$, so

$$
z=0.5\times1+0.5\times2+0.5\times1=2 .
$$

解碼權重 $\mathbf{W}_d=[0.5,\ 1,\ 0.5]^\top$，所以 $\hat{\mathbf{x}}=2\times[0.5,\ 1,\ 0.5]^\top=[1,2,1]^\top$，重建誤差 $\mathcal{L}=0$。

The decoder weights are $\mathbf{W}_d=[0.5,\ 1,\ 0.5]^\top$, so $\hat{\mathbf{x}}=2\times[0.5,\ 1,\ 0.5]^\top=[1,2,1]^\top$, and the reconstruction error is $\mathcal{L}=0$.

同一組權重換一筆輸入試試 `[補充]`：$\mathbf{x}=[2,1,1]^\top$ 一樣得到 $z=2$，重建仍是 $[1,2,1]^\top$，誤差是 $(2-1)^2+(1-2)^2+(1-1)^2=2$。這組權重只「認得」形狀像 $[0.5,1,0.5]$ 的輸入。所以權重不是手設的，而是要在整批資料上把平均重建誤差降到最低。

Try the same weights on another input `[Added]`: $\mathbf{x}=[2,1,1]^\top$ also gives $z=2$, the reconstruction is again $[1,2,1]^\top$, and the error is $(2-1)^2+(1-2)^2+(1-1)^2=2$. These weights only "recognise" inputs shaped like $[0.5,1,0.5]$. That is why the weights are not set by hand: they are learned by minimising the average reconstruction error over the whole data set.

常見錯誤：以為自編碼器的目的是得到一模一樣的輸出。輸出本來就有了，複製它沒有用；真正要的是中間那個短短的編碼 $\mathbf{z}$，以及學會怎麼壓縮的編碼器。

Common mistake: thinking the goal of an autoencoder is an identical output. We already have the output, and copying it is useless; what we really want is the short code $\mathbf{z}$ in the middle and the encoder that has learned how to compress.

### 2. 瓶頸要剛好：太寬學不到東西，太窄丟太多（The bottleneck must be just right: too wide learns nothing, too narrow loses too much）

第1節的題目有一個漏洞。如果瓶頸和輸入一樣寬（$k\ge d$），網路可以讓 $f$ 和 $g$ 都是**恆等映射**（identity mapping），也就是原封不動地傳過去。重建誤差是 $0$，但什麼特徵也沒學到。

The task of Section 1 has a loophole. If the bottleneck is as wide as the input ($k\ge d$), the network can make both $f$ and $g$ the **identity mapping**, passing the input through unchanged. The reconstruction error is $0$, yet no feature has been learned.

所以瓶頸要窄，逼網路取捨：只能留下最能幫助重建的資訊。可是太窄也不行。來源提醒，$k$ 過小時重要的特徵會被丟掉，重建變差，叫做欠擬合（第4週）。$k$ 是要調的超參數，就像第4週調模型複雜度。另一條路是讓瓶頸不窄、改加正規化約束，來源有提到，本講義不談。

So the bottleneck must be narrow, forcing the network to choose and keep only the information most useful for rebuilding. But too narrow is also bad. The source warns that when $k$ is too small, important features are thrown away and the reconstruction gets worse; this is underfitting (Week 4). $k$ is a hyperparameter to tune, just like model complexity in Week 4. Another route, mentioned by the source but not covered here, keeps the bottleneck wide and adds a regularising constraint instead.

最簡單的情形看得最清楚 `[補充]`：拿掉活化函數，編碼器和解碼器都只做線性運算（含偏差），用平方誤差訓練。這時把平均平方誤差訓練到最小，$k$ 維瓶頸能做到的最好結果，就是把資料投影到變異最大的 $k$ 個方向上；這正是第1週 KP07 的主成分分析。也就是說，**線性自編碼器訓練到誤差最小時，學到的子空間就是 PCA 的子空間**，只是權重不一定是單位長度、互相垂直的向量。偏差很重要：它讓直線可以不經過原點，而是經過資料的平均點。

The simplest case shows this best `[Added]`: remove the activation function so that the encoder and the decoder are only linear (with biases), and train with squared error. Then, once the mean squared error is trained down to its minimum, the best a $k$-dimensional bottleneck can do is project the data onto the $k$ directions of largest variance; this is exactly the principal component analysis of Week 1 KP07. In other words, **at the minimum of its error, the subspace a linear autoencoder learns is the PCA subspace**, though its weights need not be unit-length, mutually perpendicular vectors. The biases matter: they let the line pass through the mean of the data instead of the origin.

用 $10$ 個二維點試算 `[補充]`（亂數種子 $0$，點散在一條斜線附近）。$d=2$、$k=1$，所以編碼器只能留下一個數，重建點一定落在一條直線上。最好的直線方向是 $[0.8828,\ 0.4698]$，平均重建誤差 $0.0153$。如果硬把直線定成通過平均點的水平線，平均誤差是 $0.8506$，大了 $50$ 倍以上。這組資料的**總變異**是每個點到平均點距離平方的平均（除以 $n=10$），等於 $3.8147$；最好的一維編碼丟掉 $0.0153$，所以留下了 $1-0.0153/3.8147\approx99.6\%$。水平線的 $0.8506$ 剛好是第二個座標 $x_2$ 的變異，因為重建把 $x_2$ 全換成了平均值。用直接算 PCA 得到的 $0.0153$，和用梯度下降訓練一個含偏差的線性自編碼器得到的值相同。

A trial with $10$ two-dimensional points `[Added]` (random seed $0$, points scattered near a slanted line). With $d=2$ and $k=1$ the encoder may keep only one number, so every reconstruction lies on one straight line. The best line has direction $[0.8828,\ 0.4698]$ and mean reconstruction error $0.0153$. Forcing the line to be horizontal through the mean point gives a mean error of $0.8506$, more than $50$ times larger. The **total variance** of this data is the mean squared distance of the points from their mean (dividing by $n=10$), which is $3.8147$; the best one-number code loses $0.0153$, so it keeps $1-0.0153/3.8147\approx99.6\%$. The horizontal line's $0.8506$ is exactly the variance of the second coordinate $x_2$, because that reconstruction replaces every $x_2$ by its mean. The $0.0153$ computed directly by PCA is the same value a linear autoencoder with biases reaches when trained by gradient descent.

![左右兩張散佈圖，各有 10 個藍色輸入點。左圖畫一條綠色斜線，白底綠框的重建點都落在斜線上，灰色虛線連接每個輸入點和它的重建點，虛線都很短，標題寫平均誤差 0.0153。右圖畫一條紅色水平線，重建點都落在水平線上，虛線長得多，標題寫平均誤差 0.8506（Two scatter plots, each with 10 blue input points. Left: a green slanted line, the white-and-green reconstructions all lie on it, short grey dashes link each input to its reconstruction, mean error 0.0153. Right: a red horizontal line, the reconstructions lie on it, the dashes are much longer, mean error 0.8506）](../assets/figures/lec12-kp01-linear-bottleneck.png)

**圖說 `lec12-kp01-linear-bottleneck`**〔功能：一目了然、精準度、空間幾何具象化〕：先看左圖：瓶頸只有 $k=1$，所以每個重建點（空心圓）都被迫落在一條直線上，灰色虛線就是每筆資料被丟掉的部分。再看右圖：同樣是一條直線，方向選錯了，虛線就長得多。最後比較兩張圖的標題：線性自編碼器要找的，就是讓虛線平方總和最小的那條線，也就是 PCA 的第一主成分方向。

**Caption `lec12-kp01-linear-bottleneck`** [Function: seeing it at a glance, precision, making space concrete]: first look at the left panel: the bottleneck has only $k=1$, so every reconstruction (hollow circle) is forced onto one straight line, and the grey dashes are what each example loses. Then look at the right panel: still a straight line, but in the wrong direction, so the dashes are much longer. Finally compare the two titles: a linear autoencoder looks for the line with the smallest sum of squared dashes, which is the first principal direction of PCA.

既然線性自編碼器就是 PCA，為什麼還要神經網路？來源的回答是：PCA 只能找平的子空間，真實資料常常躺在彎曲的低維形狀上，叫做**流形**（manifold）。把活化函數放回去，編碼器與解碼器就能把彎曲的形狀攤平、再捲回去 `[補充：說法]`。第4節的例子就是一個彎成圓弧的一維流形。

If a linear autoencoder is just PCA, why use a neural network? The source's answer: PCA can only find flat subspaces, but real data often lies on a curved low-dimensional shape called a **manifold**. Put the activation functions back, and the encoder and decoder can flatten the curved shape and roll it back up `[Added: wording]`. The example in Section 4 is a one-dimensional manifold bent into circular arcs.

常見錯誤：以為重建誤差越小，學到的特徵就越好。這裡有兩種不同的陷阱。第一，來源提醒，網路容量太大時，它可能把訓練資料連雜訊一起記下來（過擬合，第4週）；這要在沒看過的資料上量重建誤差才看得出來 `[補充]`。第二，$k\ge d$ 的恆等映射在沒看過的資料上一樣誤差為 $0$，測試也抓不到它；只有窄瓶頸（或正規化）能擋住它。

Common mistake: thinking that the smaller the reconstruction error, the better the features. There are two different traps here. First, the source warns that a network with too much capacity may memorise the training data, noise included (overfitting, Week 4); measuring the reconstruction error on unseen data reveals this `[Added]`. Second, the identity mapping with $k\ge d$ also has error $0$ on unseen data, so testing cannot catch it; only a narrow bottleneck (or regularisation) prevents it.

### 3. 編碼器—解碼器的骨架用途很廣（The encoder–decoder skeleton has many uses）

瓶頸裡的編碼是一份濃縮的摘要。來源把它稱為隱藏空間裡的**上下文向量**（context vector），和第10週 Seq2Seq 的上下文向量是同一個想法 `[補充：連結]`。只要把「還原自己」換成「轉換成另一種東西」，同一副骨架就能做很多事。來源列了三類跨領域的用途：

The code in the bottleneck is a condensed summary. The source calls it the **context vector** in the hidden space, the same idea as the context vector of the Seq2Seq model in Week 10 `[Added: link]`. Replace "rebuild yourself" with "turn into something else", and the same skeleton can do many jobs. The source lists three kinds of cross-domain use:

- **領域轉換**（domain transformation）：輸入是一種形式，輸出換成另一種，例如英文 `elephant` 變成法文 `l'éléphant`，或卡通畫的大象變成照片裡的大象（Domain transformation: the input comes in one form and the output in another, such as the English `elephant` becoming the French `l'éléphant`, or a cartoon elephant becoming an elephant in a photo）
- **領域適應**（domain adaptation）：在甲領域學好的特徵，搬到資料長得不太一樣的乙領域也要能用，例如晴天拍的照片學完，陰天的照片也要認得出來 `[補充：例子]`（Domain adaptation: features learned well in domain A should still work in domain B, whose data looks somewhat different, for example a model trained on sunny photos should still recognise cloudy ones `[Added: example]`）
- **領域泛化**（domain generalization）：同時看好幾個領域，只留下它們共有的特徵，讓模型遇到從沒見過的新領域也能用（Domain generalization: look at several domains at once and keep only the features they share, so the model still works on a domain it has never seen）

嚴格說，自編碼器的目標是還原輸入本身；第一類的翻譯與換風格，輸出和輸入不同，是一般的編碼器—解碼器模型（第11週 KP01 也這樣分）。它們共用「壓成編碼、再展開」的骨架，所以來源放在一起談。

Strictly speaking, an autoencoder's target is the input itself; translation and style change in the first kind have outputs different from their inputs, so they are general encoder–decoder models (Week 11 KP01 drew the same line). They share the skeleton "compress into a code, then expand", which is why the source discusses them together.

自編碼器的想法後來還有一個重要的延伸。不靠窄瓶頸，而是把輸入的一部分遮住，要模型從剩下的部分猜回被遮住的內容。影像上的**遮罩自編碼器**（masked autoencoder，MAE）遮掉約 $75\%$ 的小方塊；第11週 KP06 的 BERT 也是同一類想法，它選出約 $15\%$ 的詞元來猜。這類「自己出題、自己對答案」的訓練叫**自監督學習**（self-supervised learning），是今天大型模型預訓練的基礎。

The autoencoder idea later grew an important extension. Instead of a narrow bottleneck, hide part of the input and ask the model to guess the hidden part from the rest. The **masked autoencoder** (MAE) for images hides about $75\%$ of the patches; BERT from Week 11 KP06 follows the same idea, picking about $15\%$ of the tokens to guess. Training where the model sets its own questions and checks its own answers is called **self-supervised learning**, and it underlies the pretraining of today's large models.

### 4. 一般自編碼器不能直接拿來生成（An ordinary autoencoder cannot generate directly）

到這裡，解碼器 $g$ 已經學會「給一個編碼，畫出一筆資料」。很自然會想：隨便給一個編碼 $z$，不就能生成一筆新資料？這一節用例子說明，這樣做行不通。

By now the decoder $g$ has learned "given a code, draw an example". A natural thought is: feed it any code $z$, and out comes a new example. This section shows with an example why that does not work.

例子 `[補充]`：$300$ 個二維點落在單位圓的兩段圓弧上，一段在 $18^\circ$ 到 $108^\circ$，另一段在 $180^\circ$ 到 $270^\circ$，中間空著。我們訓練一個小自編碼器：編碼器與解碼器各有一層 $32$ 個神經元的隱藏層，活化函數用 $\tanh$，瓶頸 $k=1$，亂數種子 $1$，用 Adam（第3週 KP11）以學習率 $0.003$ 訓練 $12000$ 步。因為 $k=1$，編碼只是一個數，下面寫成 $z$ 而不加粗。訓練後平均重建誤差只有 $0.0005$，重建得很好。

Example `[Added]`: $300$ two-dimensional points lie on two arcs of the unit circle, one from $18^\circ$ to $108^\circ$ and the other from $180^\circ$ to $270^\circ$, with nothing in between. We train a small autoencoder: the encoder and the decoder each have one hidden layer of $32$ neurons with $\tanh$ activation, the bottleneck is $k=1$, the random seed is $1$, and Adam (Week 3 KP11) runs for $12000$ steps with learning rate $0.003$. Because $k=1$, the code is a single number, written $z$ below without bold. After training the mean reconstruction error is only $0.0005$, so it rebuilds well.

看編碼：第一段圓弧的點，編碼落在 $0.331$ 到 $1.38$；第二段落在 $-1.451$ 到 $-0.408$。兩群之間的 $-0.408$ 到 $0.331$ 是一個**空洞**：沒有任何訓練資料的編碼在那裡。

Look at the codes: points of the first arc get codes from $0.331$ to $1.38$; those of the second arc, from $-1.451$ to $-0.408$. Between the two groups, from $-0.408$ to $0.331$, is a **hole**: no training example has its code there.

![左圖是輸入平面：淺藍點是兩段圓弧上的訓練資料，藍色粗線是訓練用過的編碼解碼出來的曲線，緊貼兩段圓弧；紅色粗線是空洞裡的編碼解碼出來的曲線，畫在兩段資料之間那段沒有資料的圓弧上，上面有一個紅點；橘色虛線是超出範圍的編碼解碼出來的曲線，剛出範圍時還在圓上，越往外越偏離圓。右圖是編碼軸：藍點與紫點兩群分別在右上與左下，中間淡紅色帶標示空洞，紅點在空洞中間，兩側淡黃色帶標示超出範圍（Left, the input plane: light-blue points are training data on two arcs; thick blue curves decoded from the codes used in training hug the two arcs; a thick red curve decoded from codes in the hole runs along the empty arc between the two groups, with a red dot on it; dashed orange curves decoded from codes beyond the used range stay on the circle at first, then wander off it further out. Right, the code axis: blue and purple groups of codes, a pale red band marking the hole with a red dot in its middle, and pale yellow bands on both sides marking codes beyond the range）](../assets/figures/lec12-kp01-latent-holes.png)

**圖說 `lec12-kp01-latent-holes`**〔功能：一目了然、空間幾何具象化〕：先看右圖：兩群編碼中間有一段空洞（淡紅），兩側超出範圍的地方（淡黃）也沒有訓練編碼。再看左圖的紅線：把空洞裡的編碼送進解碼器，畫出的點落在 $110^\circ$ 到 $180^\circ$ 那段本來沒有資料的圓弧上。紅點是空洞正中間的編碼 $-0.039$，解碼到約 $145^\circ$。最後看橘色虛線：剛超出範圍一點時，解碼的點還在圓上；編碼走到 $-3.05$ 或 $2.98$ 時，解碼出的點離圓心 $1.48$ 或 $1.32$，跑出了圓。右圖兩群編碼上下錯開只是為了好看，縱向沒有意義。

**Caption `lec12-kp01-latent-holes`** [Function: seeing it at a glance, making space concrete]: first look at the right panel: there is a hole (pale red) between the two groups of codes, and the regions beyond the range on both sides (pale yellow) hold no training codes either. Then look at the red curve on the left: codes from the hole are decoded onto the arc from $110^\circ$ to $180^\circ$, where the data has no points at all. The red dot is the code in the middle of the hole, $-0.039$, decoded to about $145^\circ$. Finally look at the dashed orange curves: just beyond the range the decoded points are still on the circle; at codes $-3.05$ or $2.98$ they are at distance $1.48$ or $1.32$ from the centre, off the circle. In the right panel the two groups of codes are offset vertically only for display; the vertical position means nothing.

這兩個現象都說明同一件事：**一般自編碼器不告訴你該從哪裡挑編碼**。空洞裡的編碼解碼到 $110^\circ$ 到 $180^\circ$，資料在那裡一個點也沒有，可是沒有任何規則告訴你多久該抽到那裡一次。在這個簡單的圓上，空洞剛好補得很順；在影像這種高維資料上，空洞裡的編碼常常畫出不像任何資料的東西。範圍外的編碼離得越遠，偏得越多。訓練只要求「用過的編碼要還原得好」，從沒要求編碼要照一個已知的分布排好。

Both effects say the same thing: **an ordinary autoencoder does not tell you where to pick codes from**. Codes in the hole decode to $110^\circ$–$180^\circ$, where the data has no points at all, yet no rule tells you how often to sample there. On this simple circle the hole happens to be filled smoothly; on high-dimensional data such as images, codes in a hole often draw things that look like no example at all. Codes beyond the range drift further off the further out they go. Training only asks that codes in use be rebuilt well; it never asks the codes to follow a known distribution.

來源的說法是：一般自編碼器的編碼器輸出一個確定的編碼，潛在空間不連續，不能直接隨機採樣來生成。精確地說，編碼是連續的實數，解碼器也是連續的；問題在於編碼**沒有已知的分布**，訓練資料的編碼散成幾座小島，中間有空洞，所以你不知道該從哪裡、照什麼比例抽樣。VAE 的改法是讓編碼器輸出編碼的**分布參數**（平均與標準差），再要求這些分布貼近一個事先選好的分布，例如標準常態分布。這樣，生成時只要從標準常態分布抽一個編碼，它大致會落在解碼器學過的地方。本週 KP05 會詳細介紹。

The source puts it this way: an ordinary autoencoder's encoder outputs a fixed code, the latent space is not continuous, and you cannot sample from it at random to generate. Precisely, the codes are continuous real numbers and the decoder is continuous too; the problem is that the codes **have no known distribution**: the training codes form scattered islands with holes between them, so you do not know where, or in what proportions, to sample. The VAE's fix is to have the encoder output the **parameters of a distribution** over the code (a mean and a standard deviation), and to require these distributions to stay close to a distribution chosen in advance, such as the standard normal. Then, to generate, you just draw a code from the standard normal, and it will mostly land where the decoder has learned. This week's KP05 covers it in detail.

常見錯誤：以為重建得好，就能拿解碼器來生成。上面的自編碼器重建誤差只有 $0.0005$，可是如果在 $-3$ 到 $3$ 之間平均地挑編碼，畫出的點有些落在資料沒有的地方，有些跑出圓外。重建好與能生成是兩件事。

Common mistake: thinking that good reconstruction means the decoder can generate. The autoencoder above has a reconstruction error of only $0.0005$, yet codes picked uniformly between $-3$ and $3$ give some points where the data has none and some off the circle. Rebuilding well and being able to generate are two different things.

### 本單元重點整理（KP01 summary）

**一句話重點**：自編碼器用「把輸入壓進窄瓶頸再還原」當訓練題目，不需要標籤就能學出編碼；瓶頸太寬會學成恆等映射、太窄會丟掉重要資訊，線性時學到的就是 PCA 的子空間；但它沒有規定編碼該落在哪裡，所以不能直接隨機挑編碼來生成新資料。

**One-sentence summary**: an autoencoder trains on the task "squeeze the input through a narrow bottleneck and rebuild it", learning a code without labels; a bottleneck that is too wide learns the identity mapping and one that is too narrow loses important information, and in the linear case it learns the PCA subspace; but it does not say where codes should lie, so you cannot pick codes at random to generate new data.

**白話說明**：用一張只能寫幾個數字的小紙條記下一張照片，再讓朋友照紙條畫回來。紙條太大就直接抄、什麼都沒學；太小就畫不像。而且朋友只會畫你真的寫過的那種紙條，亂寫一張給他，他畫出來的未必像任何照片。

**Plain explanation**: record a photo on a slip that holds only a few numbers, then have a friend draw it back from the slip. If the slip is too big you just copy and learn nothing; if it is too small the drawing is poor. And the friend can only draw from the kind of slips you actually wrote; hand over a made-up slip, and the drawing may not look like any photo.

**正式定義**：$\mathbf{z}=f(\mathbf{x})=\sigma(\mathbf{W}_e\mathbf{x}+\mathbf{b}_e)\in\mathbb{R}^k$，$\hat{\mathbf{x}}=g(\mathbf{z})=\sigma(\mathbf{W}_d\mathbf{z}+\mathbf{b}_d)\in\mathbb{R}^d$，在整批資料上最小化平均的 $\mathcal{L}=\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2$，通常 $k<d$。

**Formal definition**: $\mathbf{z}=f(\mathbf{x})=\sigma(\mathbf{W}_e\mathbf{x}+\mathbf{b}_e)\in\mathbb{R}^k$, $\hat{\mathbf{x}}=g(\mathbf{z})=\sigma(\mathbf{W}_d\mathbf{z}+\mathbf{b}_d)\in\mathbb{R}^d$, minimising the average of $\mathcal{L}=\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2$ over the data set, usually with $k<d$.

**例子**：來源的 $\mathbf{x}=[1,2,1]^\top$ 得到 $z=2$、重建誤差 $0$，同一組權重遇到 $[2,1,1]^\top$ 誤差是 $2$；$10$ 個二維點用 $k=1$，最好的直線平均誤差 $0.0153$，水平線 $0.8506$；兩段圓弧的自編碼器重建誤差 $0.0005$，但空洞與範圍外的編碼都畫不出資料裡的點。

**Example**: the source's $\mathbf{x}=[1,2,1]^\top$ gives $z=2$ and error $0$, while the same weights give error $2$ on $[2,1,1]^\top$; for $10$ two-dimensional points with $k=1$, the best line has mean error $0.0153$ and the horizontal line $0.8506$; the two-arc autoencoder has reconstruction error $0.0005$, yet codes in the hole and beyond the range do not give points from the data.

**常見迷思**：

- 「自編碼器要的是一模一樣的輸出。」錯，要的是中間的編碼與會壓縮的編碼器。
- 「瓶頸越小越好。」錯，太小會欠擬合，太大會學成恆等映射。
- 「重建得好，就能拿解碼器生成。」錯，一般自編碼器沒有規定編碼該從哪裡挑。

**Common misconceptions**:

- "An autoencoder wants an identical output." Wrong: it wants the code in the middle and an encoder that compresses.
- "The smaller the bottleneck, the better." Wrong: too small underfits, too large learns the identity mapping.
- "Good reconstruction means the decoder can generate." Wrong: an ordinary autoencoder does not say where to pick codes from.

**圖**：本單元有兩張圖。

- `lec12-kp01-linear-bottleneck`：$k=1$ 的線性自編碼器把重建點壓到一條直線上，方向對錯差很多。
- `lec12-kp01-latent-holes`：編碼之間的空洞與範圍外，解碼出來都不是資料裡的點。

**Figures** — this unit has two figures.

- `lec12-kp01-linear-bottleneck`: a linear autoencoder with $k=1$ forces the reconstructions onto one line, and the right direction matters a lot.
- `lec12-kp01-latent-holes`: codes in the hole and beyond the range decode to points that are not in the data.

### 小練習（Practice）

1. 用第1節的權重 $\mathbf{W}_e=[0.5,\ 0.5,\ 0.5]$、$\mathbf{W}_d=[0.5,\ 1,\ 0.5]^\top$（無活化、無偏差），算 $\mathbf{x}=[2,4,2]^\top$ 的編碼、重建與重建誤差。
2. 一張 $28\times28$ 的灰階圖片攤平成向量，瓶頸 $k=32$。輸入長度 $d$ 是多少？編碼器把資料壓縮成原來的幾分之幾？
3. 一個自編碼器在訓練資料上的重建誤差是 $0$，瓶頸 $k=d$。請說明為什麼這不代表它學到了好特徵。

1. With the weights of Section 1, $\mathbf{W}_e=[0.5,\ 0.5,\ 0.5]$ and $\mathbf{W}_d=[0.5,\ 1,\ 0.5]^\top$ (no activation, no bias), compute the code, the reconstruction and the reconstruction error for $\mathbf{x}=[2,4,2]^\top$.
2. A $28\times28$ greyscale image is flattened into a vector, with a bottleneck of $k=32$. What is the input length $d$? To what fraction of its size does the encoder compress the data?
3. An autoencoder has reconstruction error $0$ on the training data, and its bottleneck is $k=d$. Explain why this does not mean it has learned good features.

**第 1 題解答** `[補充]`：$z=0.5\times(2+4+2)=4$，$\hat{\mathbf{x}}=4\times[0.5,\ 1,\ 0.5]^\top=[2,4,2]^\top$，重建誤差 $0$（這筆輸入的形狀和 $[0.5,1,0.5]$ 相同）。

**Solution to question 1** `[Added]`: $z=0.5\times(2+4+2)=4$, $\hat{\mathbf{x}}=4\times[0.5,\ 1,\ 0.5]^\top=[2,4,2]^\top$, reconstruction error $0$ (this input has the same shape as $[0.5,1,0.5]$).

**第 2 題解答** `[補充]`：$d=28\times28=784$；$32/784=2/49\approx0.041$，約原來的 $4.1\%$。

**Solution to question 2** `[Added]`: $d=28\times28=784$; $32/784=2/49\approx0.041$, about $4.1\%$ of the original.

**第 3 題解答** `[補充]`：瓶頸和輸入一樣寬，網路可以學成恆等映射，原封不動地傳過去，誤差自然是 $0$，但沒有被迫取捨，也就沒有學到哪些資訊比較重要。

**Solution to question 3** `[Added]`: the bottleneck is as wide as the input, so the network can learn the identity mapping and pass everything through unchanged; the error is naturally $0$, but nothing forced it to choose, so it has not learned which information matters more.

### 小測驗（Quiz，來源原題）

來源這一段有三題自我檢測（第9章兩題、第12章一題），保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The source has three self-check questions for this part (two in Chapter 9, one in Chapter 12), kept here with all their options; the source page does not mark the answers, so the explanations below are judged from the source text.

1. 在自編碼器架構中，連接編碼器與解碼器的橋樑是什麼？
   - (A) 隨機雜訊向量
   - (B) 隱藏空間中的上下文向量 $\mathbf{c}\in\mathbb{R}^p$
   - (C) 剛性分類標籤
   - (D) 直接硬連線的原始輸入像素
2. 將卡通大象圖片轉換為真實大象照片屬於何種任務？
   - (A) 特徵降維
   - (B) 領域轉換（Domain Transformation）
   - (C) 離群點剔除
   - (D) 詞彙表遮罩
3. VAE 與標準 autoencoder 的關鍵差別？
   - (A) VAE 用更深的網路
   - (B) VAE 的 encoder 輸出 latent 分佈參數（隨機性），AE 輸出確定編碼
   - (C) VAE 不用 backprop
   - (D) VAE 不需要 decoder

1. In the autoencoder architecture, what is the bridge between the encoder and the decoder?
   - (A) A random noise vector
   - (B) The context vector $\mathbf{c}\in\mathbb{R}^p$ in the hidden space
   - (C) A rigid class label
   - (D) The raw input pixels, wired straight through
2. Turning a cartoon elephant picture into a photo of a real elephant is which kind of task?
   - (A) Feature dimensionality reduction
   - (B) Domain transformation
   - (C) Outlier removal
   - (D) Vocabulary masking
3. What is the key difference between a VAE and a standard autoencoder?
   - (A) The VAE uses a deeper network
   - (B) The VAE's encoder outputs the parameters of a latent distribution (randomness), while the AE outputs a fixed code
   - (C) The VAE does not use backpropagation
   - (D) The VAE does not need a decoder

**解析** `[補充]`：第 1 題選 (B)：第3節，來源的 $\mathbf{c}$ 就是本講義的編碼 $\mathbf{z}$，$p$ 就是 $k$；第 2 題選 (B)：第3節的第一類用途；第 3 題選 (B)：第4節；(A) 錯，網路深淺不是差別，兩者可以一樣深；兩者都用反向傳播，也都有解碼器。

**Explanations** `[Added]`: question 1 is B: see Section 3; the source's $\mathbf{c}$ is this note's code $\mathbf{z}$, and $p$ is $k$. Question 2 is B: the first kind of use in Section 3. Question 3 is B: see Section 4; A is wrong because depth is not the difference, and both can be equally deep; both use backpropagation and both have a decoder.

### 留給下一單元的問題（The question for the next unit）

第4節的結論是：要能生成，編碼不能隨便放，得落在一個我們知道怎麼抽樣的範圍。換個角度看：每筆資料背後，好像都有一個看不見的「原因」決定它長什麼樣子，編碼就是在猜這個原因。這個看不見的原因，在機率裡叫潛在變數。從一筆資料反推它的潛在變數，為什麼這麼難？下一個知識點（本週 KP02）談潛在變數與近似後驗。

The conclusion of Section 4: to generate, codes cannot sit just anywhere; they must lie in a region we know how to sample from. Seen another way, every example seems to have an invisible "cause" that decides what it looks like, and the code is a guess at that cause. In probability, this invisible cause is called a latent variable. Why is it so hard to work back from an example to its latent variable? The next knowledge point (this week's KP02) covers latent variables and the approximate posterior.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第9章「注意力機制與 Transformer 核心原理」§2.1 與教授祕笈「Auto-Encoder 架構與表徵學習」，以及第12章「變分模型」§3 中自編碼器與 VAE 的比較（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html 與 https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch12-variational-models/index.html ，讀取日期 2026-11-14）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §2.1 of Chapter 9, "Core principles of attention mechanisms and the Transformer", with the professor's notes "Auto-Encoder architecture and representation learning", and from the comparison of autoencoders and the VAE in §3 of Chapter 12, "Variational models", of CCC-LNS IDL_115 (web lectures, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch09-attention-transformers/index.html and https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch12-variational-models/index.html , accessed 2026-11-14). Content not in the source is marked `[Added]` where it appears.

- 來源把翻譯與卡通轉真實照片列為自編碼器的應用；自編碼器的目標是還原輸入本身，這兩例輸出與輸入不同，屬於一般的編碼器—解碼器模型。本講義在第3節說明兩者共用骨架。
- 來源說自編碼器的想法「演化為 Masked Autoencoders（MAE／BERT）」，並寫遮蓋 $75\%$ 輸入；$75\%$ 是影像 MAE 的比例，BERT 選出的是約 $15\%$ 的詞元（第11週 KP06）。本講義分開寫。
- 來源寫 $k\ge d$ 時網路「會退化成」恆等映射；精確說法是它「可以」學成恆等映射，所以學不到有用的取捨。
- 來源說一般自編碼器的潛在空間「不連續」、輸出「離散的編碼」；編碼其實是連續的實數，問題是沒有已知的分布、中間有空洞。本講義在第4節改寫。
- 來源用粗斜體 `\boldsymbol` 表示向量；本課程網站不支援，改用 `\mathbf`。來源的上下文向量記號 $\mathbf{c}\in\mathbb{R}^p$，本講義寫成編碼 $\mathbf{z}\in\mathbb{R}^k$。
- 本講義補上：標註成本的例子、比喻的侷限、與第10週上下文向量的連結、領域適應的晴天陰天例子、過擬合與恆等映射的區分、同一組權重遇到另一筆輸入的誤差、線性自編碼器與 PCA 的關係與十點例子、兩段圓弧的自編碼器例子與兩張圖、三題小練習與解答。
- 取捨：來源祕笈提到的率失真理論，以及去雜訊自編碼器、稀疏自編碼器等正規化版本，本講義不寫。

- The source lists translation and cartoon-to-photo conversion as uses of the autoencoder; an autoencoder's target is the input itself, and these two have outputs different from their inputs, so they are general encoder–decoder models. Section 3 explains that the two share a skeleton.
- The source says the autoencoder idea "evolved into Masked Autoencoders (MAE / BERT)", hiding $75\%$ of the input; $75\%$ is the ratio of the image MAE, while BERT picks about $15\%$ of the tokens (Week 11 KP06). This note treats them separately.
- The source says that with $k\ge d$ the network "degenerates into" the identity mapping; precisely, it "can" learn the identity mapping, so it learns no useful trade-off.
- The source says an ordinary autoencoder's latent space is "not continuous" and that it outputs "discrete codes"; the codes are in fact continuous real numbers, and the problem is that they have no known distribution and leave holes. Section 4 rewrites this.
- The source writes vectors with bold italic `\boldsymbol`; the course website does not support it, so this note uses `\mathbf`. The source's context vector $\mathbf{c}\in\mathbb{R}^p$ is written here as the code $\mathbf{z}\in\mathbb{R}^k$.
- This note adds: the labelling-cost example, the limitation of the analogy, the link to the Week 10 context vector, the sunny-versus-cloudy example of domain adaptation, the distinction between overfitting and the identity mapping, the error of the same weights on another input, the relation of the linear autoencoder to PCA with the ten-point example, the two-arc autoencoder example with both figures, and three practice questions with solutions.
- Choices: rate–distortion theory, mentioned in the source's notes, and regularised versions such as the denoising and sparse autoencoders are not covered.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 自編碼器 | autoencoder |
| 編碼器 | encoder |
| 解碼器 | decoder |
| 編碼 | code |
| 瓶頸 | bottleneck |
| 重建 | reconstruction |
| 重建誤差 | reconstruction error |
| 恆等映射 | identity mapping |
| 主成分分析 | principal component analysis (PCA) |
| 流形 | manifold |
| 上下文向量 | context vector |
| 領域轉換 | domain transformation |
| 領域適應 | domain adaptation |
| 領域泛化 | domain generalization |
| 遮罩自編碼器 | masked autoencoder (MAE) |
| 雙向編碼器表示模型（第11週 KP06） | Bidirectional Encoder Representations from Transformers (BERT) |
| 自監督學習 | self-supervised learning |
| 空洞（潛在空間中沒有資料的區域） | hole (a region of the latent space with no data) |
| 潛在變數（KP02） | latent variable |
| 變分自編碼器（KP05） | variational autoencoder (VAE) |
| 二元交叉熵 | binary cross-entropy |
| 欠擬合 | underfitting |
| 過擬合 | overfitting |
