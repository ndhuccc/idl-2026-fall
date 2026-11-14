# 第13週 KP05：條件式 GAN、DCGAN 與潛在空間，怎麼指定類別、畫出影像、在雜訊裡走路（Conditional GANs, DCGAN and the latent space: choosing the class, drawing images, and walking through the noise）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第13週的第五個知識點（KP05），接在 KP04 之後。KP01 到 KP04 談的是 GAN 的核心機制：兩個網路怎麼對抗、在縮小什麼距離、為什麼難訓練。這一節轉向實際做影像生成時的三個問題：能不能指定要生成哪一類？生成器要用什麼網路才畫得出影像？雜訊 $\mathbf{z}$ 所在的空間有什麼結構、怎麼在裡面平滑地走？最後簡短介紹幾個由 GAN 延伸出去的模型。

This is the fifth knowledge point (KP05) of Week 13, following KP04. KP01 to KP04 covered the GAN's core mechanism: how the two networks compete, what distance they shrink, and why training is hard. This note turns to three questions that come up in real image generation: can we choose which class to generate? What network lets the generator draw images? What structure does the space of the noise $\mathbf{z}$ have, and how do we walk through it smoothly? At the end it briefly introduces several models that grew out of the GAN.

先備知識是 KP02 的目標函數 $V(D,G)$、第6週的卷積（kernel、stride、padding）、第5週的批次正規化，以及第12週的自編碼器與 VAE。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第14章的 KP-6、KP-12（InfoGAN 部分）、KP-14、KP-15、KP-16 改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec13-kp05-figures.py` 算出，可以自己重算。

The prerequisites are KP02's objective $V(D,G)$, convolution from Week 6 (kernel, stride, padding), batch normalisation from Week 5, and the autoencoder and VAE of Week 12. Every term and symbol is defined when it first appears. This note is rewritten from KP-6, KP-12 (the InfoGAN part), KP-14, KP-15 and KP-16 of Chapter 14 of the course's web material CCC-LNS, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec13-kp05-figures.py`, so you can recompute it yourself.

## KP05 條件式 GAN、DCGAN 與潛在空間（Conditional GANs, DCGAN and the latent space）

### 學習目標（Learning objectives）

讀完之後，你能寫出條件式 GAN 的目標函數，說明標籤 $\mathbf{y}$ 為什麼要同時送進生成器與判別器。你能列出 DCGAN 的架構守則，並用轉置卷積的尺寸公式算出每一層的大小。你能說明潛在空間的向量算術與 InfoGAN 的想法，算出線性插值與球面插值的向量長度，並說明高維高斯分布的「薄殼」為什麼讓球面插值比較好。最後，你能用一兩句話說出 $K+1$ 類判別器、ProGAN、BiGAN、CycleGAN、AAE 各自解決什麼問題，並算出一個循環一致性損失。

After reading, you can write the objective of a conditional GAN and explain why the label $\mathbf{y}$ must go into both the generator and the discriminator. You can list DCGAN's architecture rules and use the size formula of the transposed convolution to compute every layer's size. You can explain latent-space vector arithmetic and the idea of InfoGAN, compute the vector lengths along linear and spherical interpolation, and explain why the "thin shell" of a high-dimensional Gaussian makes spherical interpolation the better choice. Finally, you can say in a sentence or two what problem each of the $K+1$-class discriminator, ProGAN, BiGAN, CycleGAN and the AAE solves, and compute a cycle-consistency loss.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathbf{y}$ | 條件，例如類別標籤的 one-hot 向量 | the condition, for example the one-hot vector of a class label |
| $G(\mathbf{z}\mid\mathbf{y}),\ D(\mathbf{x}\mid\mathbf{y})$ | 看得到條件的生成器與判別器 | generator and discriminator that also see the condition |
| $p$ | 潛在向量 $\mathbf{z}$ 的維度 | the dimension of the latent vector $\mathbf{z}$ |
| $\lVert\mathbf{z}\rVert$ | 向量的長度（歐氏範數） | the length (Euclidean norm) of a vector |
| $k,\ s,\ q$ | 卷積核大小、步長、補零（padding） | kernel size, stride, padding |
| $\mathbf{c}$ | InfoGAN 的可解釋代碼 | InfoGAN's interpretable code |
| $I(\mathbf{c};\,G(\mathbf{z},\mathbf{c}))$ | 代碼與生成樣本之間的互資訊 | mutual information between the code and the generated sample |
| $\alpha$ | 插值比例，從 $0$ 到 $1$ | the interpolation weight, from $0$ to $1$ |
| $\theta$ | 兩個潛在向量之間的夾角 | the angle between two latent vectors |
| $F$ | CycleGAN 中把 $Y$ 轉回 $X$ 的生成器 | in CycleGAN, the generator that maps $Y$ back to $X$ |
| $E(\mathbf{x})$ | 編碼器，把資料轉成潛在向量 | an encoder that maps data to a latent vector |

### 1. 條件式 GAN：指定要生成哪一類（Conditional GANs: choosing what to generate）

KP01 的生成器只吃雜訊 $\mathbf{z}$，每次生成什麼類別是隨機的，我們沒辦法說「給我一隻貓」或「給我數字 $7$」。Mirza 與 Osindero（2014）提出的條件式 GAN（conditional GAN，cGAN）做法很直接：多給一個條件 $\mathbf{y}$，例如類別的 one-hot 向量，而且生成器與判別器**兩邊都看得到它**。目標函數就是把 KP02 的 $V$ 裡每個 $D$ 和 $G$ 都加上「給定 $\mathbf{y}$」：

KP01's generator takes only the noise $\mathbf{z}$, so which class it produces each time is random; we cannot ask for "a cat" or "the digit $7$". The conditional GAN (cGAN) of Mirza and Osindero (2014) does something direct: it adds a condition $\mathbf{y}$, such as a class's one-hot vector, and **both** the generator and the discriminator see it. The objective is KP02's $V$ with "given $\mathbf{y}$" added to every $D$ and $G$:

$$\min_G\max_D\;V(D,G)=\mathbb{E}_{(\mathbf{x},\mathbf{y})\sim p_{\text{data}}}\big[\log D(\mathbf{x}\mid\mathbf{y})\big]+\mathbb{E}_{\mathbf{y}\sim p(\mathbf{y}),\ \mathbf{z}\sim p_z}\big[\log\big(1-D(G(\mathbf{z}\mid\mathbf{y})\mid\mathbf{y})\big)\big]$$

真資料一項是成對抽出的 $(\mathbf{x},\mathbf{y})$：一張圖和它真正的標籤。假資料一項先抽一個標籤 $\mathbf{y}$，再用它生成。實作上最簡單的「給定」就是把 $\mathbf{y}$ 和輸入接在一起：生成器的輸入是 $[\mathbf{z},\mathbf{y}]$，判別器的輸入是 $[\mathbf{x},\mathbf{y}]$；對影像，就把 $\mathbf{y}$ 複製成和影像一樣大的額外通道，或先轉成嵌入向量再接上 `[補充]`。

The real-data term draws pairs $(\mathbf{x},\mathbf{y})$: an image with its true label. The fake-data term first draws a label $\mathbf{y}$ and then generates with it. In practice the simplest form of "given" is concatenating $\mathbf{y}$ to the input: the generator's input is $[\mathbf{z},\mathbf{y}]$ and the discriminator's input is $[\mathbf{x},\mathbf{y}]$; for an image, $\mathbf{y}$ is copied into extra channels of the image's size, or first turned into an embedding vector `[Added]`.

![一張方塊流程圖。左邊兩個輸入：粉紅色的標籤 y 與黃色的雜訊 z，一起接到藍色的生成器 G(z|y)，輸出假的 x̂。真的 x 與假的 x̂ 都接到紫色的判別器 D(x|y)，最後輸出「是真的而且符合 y」的機率。一條粉紅虛線從標籤 y 往上繞過頂端，接到判別器，標著「同一個標籤 y 也送進 D」（A block diagram. Two inputs on the left, a pink label y and a yellow noise z, both feed the blue generator G(z|y), which outputs a fake x-hat. The real x and the fake x-hat both feed the purple discriminator D(x|y), which outputs the probability "real and matches y". A dashed pink line runs from the label y up over the top to the discriminator, labelled the same label y also goes to D）](../assets/figures/lec13-kp05-cgan.png)

**圖說 `lec13-kp05-cgan`**〔功能：一目了然、空間幾何具象化〕：先看左邊：標籤 $\mathbf{y}$ 和雜訊 $\mathbf{z}$ 一起進生成器。再看上方的粉紅虛線：同一個 $\mathbf{y}$ 也送到判別器。最後看右邊的輸出：判別器判斷的不只是「真不真」，而是「真不真、而且是不是 $\mathbf{y}$ 這一類」。

**Caption `lec13-kp05-cgan`** [Function: seeing it at a glance, making space concrete]: first look at the left: the label $\mathbf{y}$ and the noise $\mathbf{z}$ enter the generator together. Then look at the dashed pink line across the top: the same $\mathbf{y}$ also goes to the discriminator. Finally look at the output on the right: the discriminator judges not just "real or not" but "real, and of class $\mathbf{y}$, or not".

為什麼判別器也要看 $\mathbf{y}$？如果判別器看不到標籤，生成器收到「$7$」卻畫出一個很逼真的「$3$」，判別器也只會說「像真的」，生成器就沒有理由聽標籤的話。判別器看得到 $\mathbf{y}$ 之後，「逼真但類別不對」會被判成假，生成器才被逼著畫出符合條件的樣本 `[補充]`。

Why must the discriminator see $\mathbf{y}$ too? If it cannot see the label, a generator told "$7$" that draws a very realistic "$3$" is still judged "looks real", so the generator has no reason to obey the label. Once the discriminator sees $\mathbf{y}$, "realistic but the wrong class" is judged fake, and only then is the generator forced to draw samples that match the condition `[Added]`.

常見錯誤：只把標籤送進生成器。這樣訓練出來的生成器仍然會畫出逼真的影像，但類別常常和指定的不一樣，因為沒有人檢查「類別對不對」。

Common mistake: feeding the label only to the generator. The trained generator still draws realistic images, but their class often differs from the one asked for, because no one checks whether the class is right.

### 2. DCGAN：用卷積畫影像的架構守則（DCGAN: architecture rules for drawing images with convolutions）

KP01 的生成器只是一條直線 $a+bz$。要生成影像，2015 年 Radford 等人提出的深度卷積 GAN（deep convolutional GAN，DCGAN）訂下了一組後來被廣泛沿用的設計守則。來源列出三條：

KP01's generator was just a line $a+bz$. To generate images, the deep convolutional GAN (DCGAN) of Radford et al., 2015, set out design rules that were widely reused afterwards. The source lists three:

1. **不用池化層**（pooling）。生成器用轉置卷積（transposed convolution）把影像放大；判別器用步長為 $2$ 的卷積（strided convolution）把影像縮小。縮放的方式由網路自己學，而不是固定取最大值或平均。
2. **生成器與判別器都用批次正規化**（batch normalization），但生成器的輸出層和判別器的輸入層不用。
3. **激活函數**：生成器內部用 ReLU，輸出層用 $\tanh$，讓像素落在 $[-1,1]$；判別器內部全部用 LeakyReLU，負半邊的斜率是 $0.2$。

1. **No pooling layers.** The generator enlarges images with transposed convolutions; the discriminator shrinks them with stride-$2$ convolutions. The network learns how to resize, instead of taking a fixed maximum or average.
2. **Batch normalisation in both the generator and the discriminator**, except at the generator's output layer and the discriminator's input layer.
3. **Activations**: ReLU inside the generator and $\tanh$ at its output, so pixels lie in $[-1,1]$; LeakyReLU throughout the discriminator, with slope $0.2$ on the negative side.

轉置卷積怎麼把影像放大？`[補充]` 一層轉置卷積的輸出邊長是

How does a transposed convolution enlarge an image? `[Added]` The output side length of one transposed-convolution layer is

$$\text{out}=(\text{in}-1)\times s-2q+k.$$

常見的 DCGAN 實作用 $k=4$、$s=2$、$q=1$（原論文的圖畫的是 $5\times5$ 的卷積核），代進去得到 $\text{out}=2\times\text{in}$：每一層邊長剛好放大一倍。例如 $4\to(4-1)\times2-2+4=8$。生成器先用一層全連接把 $p=100$ 維的 $\mathbf{z}$ 投影並重排成 $4\times4\times1024$ 的特徵圖，再經過四層轉置卷積：$4\times4\times1024\to8\times8\times512\to16\times16\times256\to32\times32\times128\to64\times64\times3$。邊長每次加倍，通道數每次減半，最後一層輸出 $3$ 個顏色通道。

Common DCGAN implementations use $k=4$, $s=2$, $q=1$ (the paper's own figure shows $5\times5$ kernels), which gives $\text{out}=2\times\text{in}$: every layer exactly doubles the side length. For example $4\to(4-1)\times2-2+4=8$. The generator first projects the $p=100$-dimensional $\mathbf{z}$ with a fully connected layer and reshapes it into a $4\times4\times1024$ feature map, then passes it through four transposed convolutions: $4\times4\times1024\to8\times8\times512\to16\times16\times256\to32\times32\times128\to64\times64\times3$. The side length doubles each time and the number of channels halves, and the last layer outputs $3$ colour channels.

![一張由左到右的方塊圖。最左邊細長的橘色方塊標著 z: 100；接著四個淡藍色方塊依序標著 4×4×1024、8×8×512、16×16×256、32×32×128，方塊越來越高、越來越窄；最右邊是一個淡綠色的高方塊，標著 64×64×3，旁邊寫 image (tanh)。上方標題寫著：投影 z 之後，轉置卷積 kernel 4、stride 2、padding 1，每層尺寸加倍（A left-to-right block diagram. On the far left a thin orange block labelled z: 100; then four pale blue blocks labelled 4×4×1024, 8×8×512, 16×16×256 and 32×32×128, each taller and narrower than the last; on the far right a tall pale green block labelled 64×64×3, with image (tanh) beside it. The title reads: after projecting z, transposed conv, kernel 4, stride 2, padding 1; size doubles each layer）](../assets/figures/lec13-kp05-dcgan-shapes.png)

**圖說 `lec13-kp05-dcgan-shapes`**〔功能：一目了然、精準度〕：方塊的高度代表邊長，寬度代表通道數。先看最左邊的 $\mathbf{z}$；再從左往右看，每一步方塊變高（邊長加倍）、變窄（通道減半）；最後看綠色方塊：$64\times64$ 的彩色影像，經過 $\tanh$ 輸出。

**Caption `lec13-kp05-dcgan-shapes`** [Function: seeing it at a glance, precision]: a block's height stands for the side length and its width for the number of channels. First look at $\mathbf{z}$ on the far left; then read left to right: each step the block gets taller (side doubles) and narrower (channels halve); finally look at the green block: a $64\times64$ colour image, output through $\tanh$.

判別器大致是倒過來 `[補充]`：步長卷積的輸出邊長是 $\lfloor(\text{in}+2q-k)/s\rfloor+1$，同樣用 $k=4$、$s=2$、$q=1$，$64\to(64+2-4)/2+1=32$，每層減半到 $4\times4$；最後一層用 $k=4$、$s=1$、$q=0$ 把 $4\times4$ 變成 $1\times1$，輸出一個「是真的」的機率。

The discriminator is roughly the reverse `[Added]`: the output side of a strided convolution is $\lfloor(\text{in}+2q-k)/s\rfloor+1$; with the same $k=4$, $s=2$, $q=1$, $64\to(64+2-4)/2+1=32$, halving each layer down to $4\times4$; a last layer with $k=4$, $s=1$, $q=0$ turns $4\times4$ into $1\times1$, a single probability of "real".

常見錯誤：以為判別器也用轉置卷積。轉置卷積是用來**放大**的，屬於生成器；判別器要把影像縮小，用的是步長卷積。

Common mistake: thinking the discriminator also uses transposed convolutions. Transposed convolutions are for **enlarging** and belong to the generator; the discriminator shrinks images and uses strided convolutions.

### 3. 潛在空間的向量算術與 InfoGAN（Vector arithmetic in the latent space, and InfoGAN）

生成器把每一個 $\mathbf{z}$ 變成一張圖，所以 $\mathbf{z}$ 所在的空間叫做潛在空間（latent space）。DCGAN 的論文發現，這個空間裡的方向常常對應到有意義的變化。就像詞向量（word2vec）的「國王 $-$ 男人 $+$ 女人 $\approx$ 女王」，來源舉的例子是：

The generator turns every $\mathbf{z}$ into an image, so the space where $\mathbf{z}$ lives is called the latent space. The DCGAN paper found that directions in this space often correspond to meaningful changes. Like word vectors (word2vec), "king $-$ man $+$ woman $\approx$ queen", the source's example is:

$$\mathbf{z}_{\text{戴眼鏡男}}-\mathbf{z}_{\text{不戴眼鏡男}}+\mathbf{z}_{\text{不戴眼鏡女}}\approx\mathbf{z}_{\text{戴眼鏡女}}$$

（英文：$\mathbf{z}_{\text{man with glasses}}-\mathbf{z}_{\text{man without glasses}}+\mathbf{z}_{\text{woman without glasses}}\approx\mathbf{z}_{\text{woman with glasses}}$。）

(In English: $\mathbf{z}_{\text{man with glasses}}-\mathbf{z}_{\text{man without glasses}}+\mathbf{z}_{\text{woman without glasses}}\approx\mathbf{z}_{\text{woman with glasses}}$.)

前兩項相減，留下「戴眼鏡」這個方向；加到「不戴眼鏡的女性」上，再送進 $G$，就得到戴眼鏡的女性。`[補充]` 論文裡每個概念不是只用一個 $\mathbf{z}$，而是用三個代表樣本的 $\mathbf{z}$ 平均；單一樣本做算術，結果不穩定。

Subtracting the first two leaves the "wearing glasses" direction; adding it to "a woman without glasses" and feeding the result to $G$ gives a woman with glasses. `[Added]` In the paper each concept is not a single $\mathbf{z}$ but the average of the $\mathbf{z}$ of three exemplar samples; arithmetic on single samples is unstable.

這些有意義的方向是訓練後「剛好」出現的，我們不知道哪一個方向管什麼。InfoGAN（Chen 等人，2016）想讓它們變得可以指定：把輸入拆成兩部分，一部分是一般的雜訊 $\mathbf{z}$，另一部分是可解釋的代碼 $\mathbf{c}$，生成器寫成 $G(\mathbf{z},\mathbf{c})$。然後在 GAN 的目標之外，再要求**互資訊** $I(\mathbf{c};\,G(\mathbf{z},\mathbf{c}))$ 越大越好。互資訊大，表示看到生成的圖就能猜出 $\mathbf{c}$ 是多少，所以 $\mathbf{c}$ 一定在圖上留下了清楚、可辨認的影響。

These meaningful directions appear "by chance" after training, and we do not know which direction controls what. InfoGAN (Chen et al., 2016) aims to make them assignable: it splits the input into ordinary noise $\mathbf{z}$ and an interpretable code $\mathbf{c}$, writing the generator as $G(\mathbf{z},\mathbf{c})$. Then, on top of the GAN objective, it asks for the **mutual information** $I(\mathbf{c};\,G(\mathbf{z},\mathbf{c}))$ to be as large as possible. Large mutual information means that from the generated image you can guess the value of $\mathbf{c}$, so $\mathbf{c}$ must leave a clear, recognisable effect on the image.

`[補充]` 互資訊很難直接算，InfoGAN 加了一個輔助網路 $Q(\mathbf{c}\mid\mathbf{x})$，和判別器共用大部分的層，負責從圖片猜回 $\mathbf{c}$；猜得越準，互資訊的一個下界越大。在手寫數字上，一個有 $10$ 個值的類別代碼自動對應到「哪一個數字」，兩個連續代碼分別對應到「傾斜角度」和「筆畫粗細」，而訓練時完全沒有用到這些標籤。這就是「無監督的解耦」（disentangling）：不同的代碼各自控制一種性質，互不干擾。

`[Added]` Mutual information is hard to compute directly, so InfoGAN adds an auxiliary network $Q(\mathbf{c}\mid\mathbf{x})$, sharing most of its layers with the discriminator, that guesses $\mathbf{c}$ back from the image; the better it guesses, the larger a lower bound on the mutual information. On handwritten digits, a categorical code with $10$ values came to stand for "which digit", and two continuous codes for "slant angle" and "stroke width", with none of these labels used in training. This is unsupervised disentangling: each code controls one property without disturbing the others.

常見錯誤：以為 InfoGAN 和條件式 GAN 一樣需要標籤。條件式 GAN 的 $\mathbf{y}$ 來自資料的標籤；InfoGAN 的 $\mathbf{c}$ 是隨機抽的，它的意義是訓練中靠互資訊自己長出來的。

Common mistake: thinking InfoGAN needs labels as a conditional GAN does. The cGAN's $\mathbf{y}$ comes from the data's labels; InfoGAN's $\mathbf{c}$ is drawn at random, and its meaning emerges during training through the mutual information.

### 4. 在潛在空間裡走路：線性插值與球面插值（Walking in the latent space: linear and spherical interpolation）

要檢查一個 GAN 是學到了平滑的結構、還是只背下訓練圖片，常見做法是在潛在空間挑兩個點 $\mathbf{z}_1$、$\mathbf{z}_2$，在兩點之間取一連串中間點，把每一點都送進 $G$，看生成的圖是不是平順地漸變。這叫插值（interpolation）。最直接的是線性插值（linear interpolation，Lerp），沿直線走：

To check whether a GAN has learned a smooth structure or has only memorised the training images, a common method is to pick two points $\mathbf{z}_1$ and $\mathbf{z}_2$ in the latent space, take a series of intermediate points between them, feed each to $G$, and see whether the generated images change smoothly. This is called interpolation. The most direct is linear interpolation (Lerp), walking along a straight line:

$$\mathbf{z}(\alpha)=(1-\alpha)\,\mathbf{z}_1+\alpha\,\mathbf{z}_2,\qquad\alpha\in[0,1].$$

問題出在高維的高斯分布。$\mathbf{z}\sim\mathcal{N}(\mathbf{0},I)$ 的密度最高點在原點，可是在高維時，抽出來的點幾乎都落在離原點約 $\sqrt{p}$ 的一層薄殼上。`[補充]` 原因是 $\lVert\mathbf{z}\rVert^2$ 是 $p$ 個獨立平方的和，平均是 $p$，而長度 $\lVert\mathbf{z}\rVert$ 的標準差只有大約 $0.7$（趨近 $1/\sqrt2$），不會隨 $p$ 變大。在 $p=100$ 抽 $100000$ 個點：長度的平均是 $9.97$，標準差 $0.708$，中間 $98\%$ 落在 $8.36$ 到 $11.66$ 之間，短於 $8$ 的只有 $0.23\%$。

The problem lies in the high-dimensional Gaussian. The density of $\mathbf{z}\sim\mathcal{N}(\mathbf{0},I)$ is highest at the origin, yet in high dimensions almost all sampled points lie on a thin shell at distance about $\sqrt{p}$ from the origin. `[Added]` The reason is that $\lVert\mathbf{z}\rVert^2$ is a sum of $p$ independent squares with mean $p$, while the standard deviation of the length $\lVert\mathbf{z}\rVert$ is only about $0.7$ (tending to $1/\sqrt2$) and does not grow with $p$. Drawing $100000$ points with $p=100$: the mean length is $9.97$ with standard deviation $0.708$, the middle $98\%$ lies between $8.36$ and $11.66$, and only $0.23\%$ are shorter than $8$.

兩個隨機向量在高維幾乎互相垂直。從殼上抽兩點，直線的中點會往內陷，落到殼的內側：生成器訓練時幾乎沒見過這麼短的 $\mathbf{z}$。`[補充]` 兩個長約 $\sqrt{p}$ 且互相垂直的向量，中點長度約 $\sqrt{p/2}\approx7.07$。來源說這會讓中間的圖變模糊。改用球面線性插值（spherical linear interpolation，Slerp），沿大圓弧轉過去。大圓弧（great-circle arc）是球面上、圓心在原點的圓的一段，就像地球上沿赤道走：方向一直在轉，離球心的距離不變。

Two random vectors in high dimensions are almost perpendicular. Take two points on the shell, and the midpoint of the straight line sinks inward, inside the shell: lengths the generator almost never saw during training. `[Added]` For two perpendicular vectors of length about $\sqrt{p}$, the midpoint has length about $\sqrt{p/2}\approx7.07$. The source says this makes the middle images blurry. Spherical linear interpolation (Slerp) instead turns along a great-circle arc. A great-circle arc is part of a circle on the sphere whose centre is the origin, like walking along the equator: the direction keeps turning while the distance from the centre stays fixed.

$$\text{Slerp}(\mathbf{z}_1,\mathbf{z}_2;\alpha)=\frac{\sin\big((1-\alpha)\theta\big)}{\sin\theta}\,\mathbf{z}_1+\frac{\sin(\alpha\theta)}{\sin\theta}\,\mathbf{z}_2,\qquad\theta=\arccos\frac{\mathbf{z}_1^{\top}\mathbf{z}_2}{\lVert\mathbf{z}_1\rVert\,\lVert\mathbf{z}_2\rVert}.$$

`[補充]` 實際抽兩個 $p=100$ 的向量：$\lVert\mathbf{z}_1\rVert=10.921$，$\lVert\mathbf{z}_2\rVert=9.256$，夾角 $\theta=91.77^\circ$（餘弦 $-0.0309$）。沿兩種路徑走，長度如下表。

`[Added]` Drawing two actual vectors with $p=100$: $\lVert\mathbf{z}_1\rVert=10.921$, $\lVert\mathbf{z}_2\rVert=9.256$, angle $\theta=91.77^\circ$ (cosine $-0.0309$). Along the two paths the lengths are:

| $\alpha$ | $0$ | $0.25$ | $0.5$ | $0.75$ | $1$ |
|---|---|---|---|---|---|
| Lerp 長度（length） | $10.921$ | $8.442$ | $7.048$ | $7.381$ | $9.256$ |
| Slerp 長度（length） | $10.921$ | $10.704$ | $10.125$ | $9.510$ | $9.256$ |

Lerp 的中點長度 $7.048$，正好接近上面估計的 $7.07$，比剛才 $100000$ 個樣本裡的每一個都短（最短的樣本是 $7.287$）；Lerp 最低點其實在 $\alpha\approx0.58$，長度 $6.953$，因為兩端不一樣長；Slerp 一路都在 $9.256$ 到 $10.921$ 之間，留在殼裡。

Lerp's midpoint has length $7.048$, close to the estimate $7.07$ above and shorter than every one of the $100000$ samples (the shortest is $7.287$); Lerp's lowest point is actually at $\alpha\approx0.58$, length $6.953$, because the two ends differ in length; Slerp stays between $9.256$ and $10.921$ all the way, inside the shell.

![左右兩張圖。左圖是 p=100 時 10 萬個樣本長度的直方圖，橫軸 6 到 13，峰在 10 附近，大約從 8 到 12；一條紅色虛線在 7.05，標著 Lerp 的中點，落在直方圖左邊、沒有樣本的地方。右圖橫軸是插值比例 α 從 0 到 1，縱軸是長度 6 到 12.5；淡藍色帶（prior: middle 98%）是先驗長度的中間 98%，從 8.36 到 11.66；綠線 Slerp 從 10.92 平滑降到 9.26，全程在藍帶裡；紅線 Lerp 從 10.92 往下彎到約 6.95（在 α≈0.58，比正中間偏右）、再回到 9.26，中段掉到藍帶下面（Two panels. Left: a histogram of the lengths of 100000 samples with p=100, horizontal axis 6 to 13, peaking near 10 and spanning roughly 8 to 12; a dashed red line at 7.05 marks the Lerp midpoint, to the left of the histogram where there are no samples. Right: interpolation weight alpha from 0 to 1 on the horizontal axis and length from 6 to 12.5 on the vertical; a pale blue band (prior: middle 98%) is the middle 98% of prior lengths, from 8.36 to 11.66; the green Slerp line falls smoothly from 10.92 to 9.26 and stays inside the band; the red Lerp line sags from 10.92 down to about 6.95 (at alpha≈0.58, right of centre) and back to 9.26, dropping below the band in the middle）](../assets/figures/lec13-kp05-lerp-slerp.png)

**圖說 `lec13-kp05-lerp-slerp`**〔功能：一目了然、精準度、空間幾何具象化〕：先看左圖：$100$ 維的高斯樣本，長度擠在 $10$ 附近，小於約 $8$ 的地方幾乎沒有樣本；紅色虛線（Lerp 的中點）落在沒有樣本的地方。再看右圖的淡藍帶，那是常見的長度範圍。最後比較兩條線：綠色的 Slerp 一直在帶子裡，紅色的 Lerp 中段掉出帶子。

**Caption `lec13-kp05-lerp-slerp`** [Function: seeing it at a glance, precision, making space concrete]: first look at the left panel: $100$-dimensional Gaussian samples have lengths crowded near $10$, and almost none are shorter than about $8$; the dashed red line (the Lerp midpoint) falls where there are no samples. Then look at the pale blue band on the right, the range of typical lengths. Finally compare the two lines: the green Slerp stays inside the band, while the red Lerp drops out of it in the middle.

來源說 Slerp「保持向量模長恆定」。`[補充]` 這只在兩個端點一樣長時成立：兩個長度都是 $r$ 時，Slerp 一路都是 $r$。上表的兩端不一樣長（$10.921$ 與 $9.256$），所以 Slerp 的長度也跟著變（嚴格說來這時也不完全在一個大圓上），只是一直留在殼的範圍內。來源互動展示的 $0.707$ 與 $1.0$，是兩個互相垂直的**單位**向量：直線中點的長度是 $\sqrt{2}/2=0.7071$，Slerp 中點仍是 $1$。

The source says Slerp "keeps the vector length constant". `[Added]` This holds only when the two endpoints have the same length: if both have length $r$, Slerp stays at $r$ throughout. The two ends in the table differ ($10.921$ and $9.256$), so Slerp's length changes too (strictly, it is then not exactly on a great circle), but it stays within the shell. The $0.707$ and $1.0$ in the source's demo are for two perpendicular **unit** vectors: the straight-line midpoint has length $\sqrt{2}/2=0.7071$, while the Slerp midpoint is still $1$.

常見錯誤：以為高斯分布的密度最高點（原點附近）就是「最常抽到的地方」。在高維，原點附近的體積太小，幾乎抽不到；樣本集中在半徑約 $\sqrt{p}$ 的殼上。

Common mistake: thinking that where a Gaussian's density is highest (near the origin) is "where samples usually land". In high dimensions the volume near the origin is tiny and almost never sampled; the samples crowd onto a shell of radius about $\sqrt{p}$.

### 5. 由 GAN 延伸出去的幾個模型（A few models that grew out of the GAN）

這一節簡短介紹幾個重要的延伸，每個只說它解決什麼問題、核心想法是什麼。細節不在本課範圍。

This section briefly introduces several important extensions, saying only what problem each solves and its core idea. The details are beyond this course.

**半監督學習：$K+1$ 類判別器與 CatGAN。** 有很多圖、但只有少數有標籤時，可以把判別器從「真／假」兩類改成 $K+1$ 類：前 $K$ 類是真資料的類別，第 $K+1$ 類專門代表「生成器做的假圖」。有標籤的圖用來學分類，沒標籤的真圖只要求「不是第 $K+1$ 類」，假圖則要判成第 $K+1$ 類 `[補充]`。判別器在分辨真假的過程中學到的特徵，也幫助了分類。CatGAN（categorical GAN）是另一種做法，用類別預測的熵（不確定性）來定目標：判別器對真圖要有把握地分到某一類（熵小），對假圖則拿不定主意（熵大），而且整體來看各類別要用得平均；生成器則希望自己的樣本被有把握地分類 `[補充]`。

**Semi-supervised learning: the $K+1$-class discriminator and CatGAN.** When there are many images but only a few labelled ones, the discriminator can be changed from two classes (real/fake) to $K+1$: the first $K$ are the real data's classes and class $K+1$ stands for "fake made by the generator". Labelled images train the classification, unlabelled real images only need to be "not class $K+1$", and fakes should be put in class $K+1$ `[Added]`. The features the discriminator learns while telling real from fake also help the classification. CatGAN (categorical GAN) is another approach, setting its goals with the entropy (uncertainty) of the class prediction: the discriminator should confidently put real images into one class (low entropy) and be undecided about fakes (high entropy), while using all classes evenly overall; the generator wants its samples classified confidently `[Added]`.

**高解析度：ProGAN。** 直接訓練 $1024\times1024$ 的 GAN 很不穩定。Karras 等人（2018）的 Progressive GAN 從 $4\times4$ 開始訓練，穩定後同時在生成器與判別器各加入一組新的層，解析度加倍：$4\to8\to16\to\cdots\to1024$，一共加倍 $8$ 次。新加的層是**慢慢淡入**的：輸出是「舊層的結果用最近鄰放大」和「新層的結果」的加權平均（判別器那一側也一樣淡入），新層的權重從 $0$ 慢慢升到 $1$ `[補充]`。低解析度先學好整體的輪廓和顏色，高解析度再補細節。

**High resolution: ProGAN.** Training a $1024\times1024$ GAN directly is very unstable. The Progressive GAN of Karras et al. (2018) starts training at $4\times4$ and, once stable, adds a new block of layers to both the generator and the discriminator, doubling the resolution: $4\to8\to16\to\cdots\to1024$, eight doublings in all. Each new layer is **faded in**: the output is a weighted average of "the old layer's result, enlarged by nearest neighbour" and "the new layer's result" (the discriminator side fades in the same way), with the new layer's weight rising slowly from $0$ to $1$ `[Added]`. The low resolutions learn the overall outline and colour first, and the high resolutions then add detail.

**反方向：BiGAN／ALI。** 一般的 GAN 只會從 $\mathbf{z}$ 到 $\mathbf{x}$，給一張真圖，它說不出對應的 $\mathbf{z}$ 是什麼。BiGAN（bidirectional GAN）與 ALI（adversarially learned inference）多加一個編碼器 $E(\mathbf{x})$，判別器改成看**成對**的資料 $D(\mathbf{x},\mathbf{z})$：真的一對是 $(\mathbf{x},E(\mathbf{x}))$，假的一對是 $(G(\mathbf{z}),\mathbf{z})$。判別器分不出來時，理論上 $E$ 與 $G$ 互為反函數 `[補充]`。

**The other direction: BiGAN / ALI.** An ordinary GAN only goes from $\mathbf{z}$ to $\mathbf{x}$; given a real image it cannot say what the matching $\mathbf{z}$ is. BiGAN (bidirectional GAN) and ALI (adversarially learned inference) add an encoder $E(\mathbf{x})$ and change the discriminator to look at **pairs**, $D(\mathbf{x},\mathbf{z})$: a real pair is $(\mathbf{x},E(\mathbf{x}))$ and a fake pair is $(G(\mathbf{z}),\mathbf{z})$. When the discriminator cannot tell them apart, in theory $E$ and $G$ are inverses of each other `[Added]`.

**沒有成對資料的風格轉換：CycleGAN。** 想把馬的照片轉成斑馬、或把照片轉成油畫，通常找不到「同一張照片的馬版和斑馬版」這種成對資料。Zhu 等人（2017）的 CycleGAN 用兩個生成器：$G$ 把 $X$（馬）轉成 $Y$（斑馬），$F$ 把 $Y$ 轉回 $X$，各配一個判別器。除了 GAN 的損失，再加上循環一致性損失（cycle-consistency loss）：

**Style translation without paired data: CycleGAN.** To turn horse photos into zebras, or photos into oil paintings, we usually cannot find paired data such as "the horse version and the zebra version of the same photo". The CycleGAN of Zhu et al. (2017) uses two generators, $G$ mapping $X$ (horses) to $Y$ (zebras) and $F$ mapping $Y$ back to $X$, each with its own discriminator. On top of the GAN losses it adds the cycle-consistency loss:

$$\mathcal{L}_{\text{cyc}}(G,F)=\mathbb{E}_{\mathbf{x}\sim p_X}\big[\lVert F(G(\mathbf{x}))-\mathbf{x}\rVert_1\big]+\mathbb{E}_{\mathbf{y}\sim p_Y}\big[\lVert G(F(\mathbf{y}))-\mathbf{y}\rVert_1\big].$$

意思是：轉過去再轉回來，要回到原來的樣子。$\lVert\cdot\rVert_1$ 是各分量差的絕對值加總。`[補充]` 小例子：一張只有三個像素的圖 $\mathbf{x}=(0.2,\ 0.6,\ 0.9)$，轉過去再轉回來得到 $F(G(\mathbf{x}))=(0.25,\ 0.5,\ 0.9)$，這張圖的循環損失是 $\lvert0.05\rvert+\lvert-0.1\rvert+\lvert0\rvert=0.15$。沒有這一項時，$G$ 只要畫出逼真的斑馬就能騙過判別器，不必理會輸入的是哪一匹馬；循環一致性逼它保留原圖的內容（姿勢、位置、背景）。

It means: translate across and back, and you should get the original. $\lVert\cdot\rVert_1$ adds up the absolute values of the component differences. `[Added]` A small example: an image of only three pixels $\mathbf{x}=(0.2,\ 0.6,\ 0.9)$ comes back as $F(G(\mathbf{x}))=(0.25,\ 0.5,\ 0.9)$, so its cycle loss is $\lvert0.05\rvert+\lvert-0.1\rvert+\lvert0\rvert=0.15$. Without this term, $G$ fools the discriminator just by drawing a realistic zebra, ignoring which horse it was given; cycle consistency forces it to keep the original image's content (pose, position, background).

**對抗式自編碼器（AAE）。** 第12週的 VAE 用 KL 散度把編碼 $\mathbf{z}$ 的分布拉向先驗 $p(\mathbf{z})$（先驗就是我們事先指定、希望編碼服從的分布，例如標準常態）。Makhzani 等人（2015）的對抗式自編碼器（adversarial autoencoder，AAE）改用對抗：編碼器 $E(\mathbf{x})$ 當生成器，產生編碼 $\mathbf{z}$；另一個判別器 $D_z$ 分辨「編碼器產生的 $\mathbf{z}$」和「從先驗抽的 $\mathbf{z}$」。再加上解碼器的重建損失 `[補充]`。好處是先驗只要能抽樣就行，例如混合高斯，不必寫得出 KL 散度的公式 `[補充]`。

**The adversarial autoencoder (AAE).** Week 12's VAE uses the KL divergence to pull the distribution of the codes $\mathbf{z}$ towards a prior $p(\mathbf{z})$ (the prior is the distribution we choose in advance for the codes to follow, such as the standard normal). The adversarial autoencoder (AAE) of Makhzani et al. (2015) uses competition instead: the encoder $E(\mathbf{x})$ plays the generator, producing codes $\mathbf{z}$, and another discriminator $D_z$ tells "codes from the encoder" from "codes drawn from the prior". A decoder's reconstruction loss is added as well `[Added]`. The benefit is that the prior only needs to be something we can sample from, such as a mixture of Gaussians, without a formula for the KL divergence `[Added]`.

常見錯誤：以為 CycleGAN 的循環損失可以單獨訓練出風格轉換。只有循環損失時，$G$ 和 $F$ 都當成「什麼都不改」就能讓損失是 $0$；要靠 GAN 的損失逼 $G(\mathbf{x})$ 看起來像斑馬，兩者缺一不可 `[補充]`。

Common mistake: thinking the cycle loss alone can train a style translator. With only the cycle loss, $G$ and $F$ can both "change nothing" and make the loss $0$; the GAN loss is what forces $G(\mathbf{x})$ to look like a zebra, and both are needed `[Added]`.

### 本單元重點整理（KP05 summary）

**一句話重點**：條件式 GAN 把標籤 $\mathbf{y}$ 同時送進生成器和判別器，才能指定類別；DCGAN 用轉置卷積（$k=4,s=2,q=1$ 每層加倍）、批次正規化與 ReLU／LeakyReLU 畫影像；潛在空間的方向有語意，InfoGAN 用互資訊讓代碼 $\mathbf{c}$ 各自控制一種性質；高維高斯樣本集中在半徑約 $\sqrt{p}$ 的殼上，所以球面插值比直線插值好；$K+1$ 類判別器、ProGAN、BiGAN、CycleGAN、AAE 分別把 GAN 用在半監督學習、高解析度、反向編碼、無配對轉換與自編碼器。

**One-sentence summary**: a conditional GAN feeds the label $\mathbf{y}$ to both the generator and the discriminator so the class can be chosen; DCGAN draws images with transposed convolutions ($k=4,s=2,q=1$ doubles each layer), batch normalisation and ReLU/LeakyReLU; directions in the latent space carry meaning, and InfoGAN uses mutual information so each code in $\mathbf{c}$ controls one property; high-dimensional Gaussian samples crowd onto a shell of radius about $\sqrt{p}$, so spherical interpolation beats straight-line interpolation; the $K+1$-class discriminator, ProGAN, BiGAN, CycleGAN and the AAE apply the GAN to semi-supervised learning, high resolution, reverse encoding, unpaired translation and autoencoders.

**白話說明**：條件式 GAN 像是點菜：廚師（生成器）知道你點什麼，評審（判別器）也拿著同一張點菜單，菜做得好但不是你點的，一樣不及格。DCGAN 是一套「怎麼一層一層把小圖放大」的做菜規則。潛在空間像一張地圖，每個方向代表一種變化；在地圖上從一點走到另一點，要沿著大家常住的那一圈走，不要穿過中間沒人住的空地。

**Plain explanation**: a conditional GAN is like ordering food: the cook (generator) knows what you ordered, and the judge (discriminator) holds the same order slip, so a good dish that is not what you ordered still fails. DCGAN is a set of recipes for "enlarging a small picture layer by layer". The latent space is like a map where each direction stands for one kind of change; to walk from one point to another, follow the ring where everyone lives, rather than cutting across the empty land in the middle.

**正式定義**：cGAN：$\min_G\max_D\ \mathbb{E}_{(\mathbf{x},\mathbf{y})}[\log D(\mathbf{x}\mid\mathbf{y})]+\mathbb{E}_{\mathbf{y},\mathbf{z}}[\log(1-D(G(\mathbf{z}\mid\mathbf{y})\mid\mathbf{y}))]$；轉置卷積 $\text{out}=(\text{in}-1)s-2q+k$；Lerp $\mathbf{z}(\alpha)=(1-\alpha)\mathbf{z}_1+\alpha\mathbf{z}_2$；Slerp $=\frac{\sin((1-\alpha)\theta)}{\sin\theta}\mathbf{z}_1+\frac{\sin(\alpha\theta)}{\sin\theta}\mathbf{z}_2$；$\mathcal{L}_{\text{cyc}}=\mathbb{E}\lVert F(G(\mathbf{x}))-\mathbf{x}\rVert_1+\mathbb{E}\lVert G(F(\mathbf{y}))-\mathbf{y}\rVert_1$。

**Formal definition**: cGAN: $\min_G\max_D\ \mathbb{E}_{(\mathbf{x},\mathbf{y})}[\log D(\mathbf{x}\mid\mathbf{y})]+\mathbb{E}_{\mathbf{y},\mathbf{z}}[\log(1-D(G(\mathbf{z}\mid\mathbf{y})\mid\mathbf{y}))]$; transposed convolution $\text{out}=(\text{in}-1)s-2q+k$; Lerp $\mathbf{z}(\alpha)=(1-\alpha)\mathbf{z}_1+\alpha\mathbf{z}_2$; Slerp $=\frac{\sin((1-\alpha)\theta)}{\sin\theta}\mathbf{z}_1+\frac{\sin(\alpha\theta)}{\sin\theta}\mathbf{z}_2$; $\mathcal{L}_{\text{cyc}}=\mathbb{E}\lVert F(G(\mathbf{x}))-\mathbf{x}\rVert_1+\mathbb{E}\lVert G(F(\mathbf{y}))-\mathbf{y}\rVert_1$.

**例子**：DCGAN 生成器 $4\times4\times1024\to8\times8\times512\to16\times16\times256\to32\times32\times128\to64\times64\times3$；$p=100$ 時樣本長度平均 $9.97$、中間 $98\%$ 在 $8.36$ 到 $11.66$；兩個樣本 $\lVert\mathbf{z}_1\rVert=10.921$、$\lVert\mathbf{z}_2\rVert=9.256$，Lerp 中點長度 $7.048$，Slerp 中點 $10.125$；兩個垂直單位向量的直線中點長度 $0.7071$；循環損失例子 $0.15$。

**Example**: DCGAN generator $4\times4\times1024\to8\times8\times512\to16\times16\times256\to32\times32\times128\to64\times64\times3$; with $p=100$ the sample lengths have mean $9.97$ and the middle $98\%$ lies between $8.36$ and $11.66$; for two samples with $\lVert\mathbf{z}_1\rVert=10.921$ and $\lVert\mathbf{z}_2\rVert=9.256$, the Lerp midpoint has length $7.048$ and the Slerp midpoint $10.125$; the straight-line midpoint of two perpendicular unit vectors has length $0.7071$; the cycle-loss example gives $0.15$.

**常見迷思**：

- 「條件式 GAN 只要把標籤給生成器。」錯，判別器也要看到標籤，否則沒人檢查類別。
- 「DCGAN 的判別器用轉置卷積。」錯，判別器用步長卷積縮小，轉置卷積是生成器放大用的。
- 「Slerp 一定保持長度不變。」錯，只有兩端一樣長時才不變。
- 「高斯樣本最常落在原點附近。」錯，高維時集中在半徑約 $\sqrt{p}$ 的殼上。

**Common misconceptions**:

- "A conditional GAN only needs to give the label to the generator." Wrong: the discriminator must see it too, or no one checks the class.
- "DCGAN's discriminator uses transposed convolutions." Wrong: the discriminator shrinks with strided convolutions; transposed convolutions are the generator's way to enlarge.
- "Slerp always keeps the length constant." Wrong: only when both ends have the same length.
- "Gaussian samples usually land near the origin." Wrong: in high dimensions they crowd onto a shell of radius about $\sqrt{p}$.

**圖**：本單元有三張圖。

- `lec13-kp05-cgan`：條件式 GAN 的架構，標籤同時送進生成器與判別器。
- `lec13-kp05-dcgan-shapes`：DCGAN 生成器每一層的尺寸。
- `lec13-kp05-lerp-slerp`：高維高斯樣本的長度分布，以及兩種插值路徑上的長度。

**Figures** — this unit has three figures.

- `lec13-kp05-cgan`: the conditional GAN architecture, with the label going to both the generator and the discriminator.
- `lec13-kp05-dcgan-shapes`: the size of every layer of the DCGAN generator.
- `lec13-kp05-lerp-slerp`: the length distribution of high-dimensional Gaussian samples, and the lengths along the two interpolation paths.

### 小練習（Practice）

1. 一層轉置卷積用 $k=4$、$s=2$、$q=1$，輸入邊長 $16$，輸出邊長是多少？若生成器從 $4\times4$ 開始、想輸出 $128\times128$，需要幾層這樣的轉置卷積？
2. 兩個長度都是 $1$ 的向量，夾角 $60^\circ$。直線中點（$\alpha=0.5$ 的 Lerp）長度是多少？Slerp 的中點呢？（提示：$\lVert\mathbf{a}+\mathbf{b}\rVert^2=\lVert\mathbf{a}\rVert^2+\lVert\mathbf{b}\rVert^2+2\,\mathbf{a}^{\top}\mathbf{b}$。）
3. 兩個像素的圖 $\mathbf{x}=(0.1,\ 0.4)$，轉過去再轉回來得到 $(0.3,\ 0.4)$。這張圖的循環損失是多少？如果 CycleGAN 拿掉循環損失，會出什麼問題？

1. A transposed-convolution layer uses $k=4$, $s=2$, $q=1$ with input side $16$. What is the output side? If a generator starts from $4\times4$ and should output $128\times128$, how many such layers does it need?
2. Two vectors of length $1$ make an angle of $60^\circ$. What is the length of the straight-line midpoint (Lerp with $\alpha=0.5$)? And of the Slerp midpoint? (Hint: $\lVert\mathbf{a}+\mathbf{b}\rVert^2=\lVert\mathbf{a}\rVert^2+\lVert\mathbf{b}\rVert^2+2\,\mathbf{a}^{\top}\mathbf{b}$.)
3. A two-pixel image $\mathbf{x}=(0.1,\ 0.4)$ comes back as $(0.3,\ 0.4)$ after translating across and back. What is its cycle loss? If CycleGAN dropped the cycle loss, what would go wrong?

**第 1 題解答** `[補充]`：$(16-1)\times2-2+4=32$。$4\to8\to16\to32\to64\to128$，加倍 $5$ 次，需要 $5$ 層。

**Solution to question 1** `[Added]`: $(16-1)\times2-2+4=32$. $4\to8\to16\to32\to64\to128$ is $5$ doublings, so $5$ layers.

**第 2 題解答** `[補充]`：$\mathbf{a}^{\top}\mathbf{b}=\cos60^\circ=0.5$，$\lVert\mathbf{a}+\mathbf{b}\rVert^2=1+1+1=3$，中點長度 $\sqrt3/2\approx0.866$。兩端一樣長，Slerp 一路長度都是 $1$。夾角越大，直線中點陷得越深：$90^\circ$ 時是 $0.7071$。

**Solution to question 2** `[Added]`: $\mathbf{a}^{\top}\mathbf{b}=\cos60^\circ=0.5$, so $\lVert\mathbf{a}+\mathbf{b}\rVert^2=1+1+1=3$ and the midpoint has length $\sqrt3/2\approx0.866$. The two ends have the same length, so Slerp stays at length $1$ throughout. The wider the angle, the deeper the straight-line midpoint sinks: $0.7071$ at $90^\circ$.

**第 3 題解答** `[補充]`：$\lvert0.3-0.1\rvert+\lvert0.4-0.4\rvert=0.2$。拿掉循環損失，$G$ 只要畫出任何一張逼真的斑馬就能騙過判別器，可能不管輸入是哪匹馬都畫成同一種斑馬，原圖的姿勢與背景都丟了。「只畫少數幾種」這個問題，下一個知識點會再遇到。

**Solution to question 3** `[Added]`: $\lvert0.3-0.1\rvert+\lvert0.4-0.4\rvert=0.2$. Without the cycle loss, $G$ fools the discriminator by drawing any realistic zebra, and might draw the same kind of zebra whatever horse it is given, losing the original pose and background. This problem of "drawing only a few kinds" comes up again in the next knowledge point.

### 小測驗（Quiz，來源原題）

來源的 KP-6、KP-12、KP-14、KP-15、KP-16 各有一題自我檢測，五題全部保留在這裡（KP-12 那題問的是 InfoGAN，所以放在本單元），保留原題與全部選項；來源頁面附有解析，下面依它說明。

The source's KP-6, KP-12, KP-14, KP-15 and KP-16 each have one self-check question; this unit keeps all five (KP-12's question is here because it asks about InfoGAN), with all options; the source page gives explanations, which the note follows below.

1. 下列關於 DCGAN 架構設計守則的敘述，何者正確？
   - (A) 判別器應大量採用 Max Pooling 來降低特徵解析度
   - (B) 生成器內部使用 ReLU 激活函數，判別器內部使用 LeakyReLU，並以轉置卷積取代 Pooling
   - (C) 生成器與判別器的每一層都必須施加 Softmax 歸一化
   - (D) DCGAN 的潛在空間向量無法進行加減算術運算

1. Which statement about DCGAN's architecture rules is correct?
   - (A) The discriminator should use a lot of max pooling to lower feature resolution
   - (B) The generator uses ReLU inside and the discriminator uses LeakyReLU inside, with transposed convolutions replacing pooling
   - (C) Every layer of the generator and discriminator must apply softmax normalisation
   - (D) DCGAN's latent vectors cannot be added or subtracted

**解析**：選 (B)。來源的理由：DCGAN 取消池化，改用轉置卷積／步長卷積，生成器內部用 ReLU、判別器內部用 LeakyReLU（見第2節）。`[補充]` 選項 (B) 的「以轉置卷積取代 Pooling」要細分：生成器用轉置卷積放大，判別器用步長卷積縮小。(A) 錯，DCGAN 不用池化；(C) 錯，DCGAN 用的是批次正規化，不是 softmax；(D) 錯，第3節正是潛在向量的加減。

**Explanation**: the answer is B. The source's reason: DCGAN drops pooling in favour of transposed / strided convolutions, with ReLU inside the generator and LeakyReLU inside the discriminator (see Section 2). `[Added]` Option B's "transposed convolutions replacing pooling" needs refining: the generator enlarges with transposed convolutions and the discriminator shrinks with strided convolutions. A is wrong because DCGAN uses no pooling; C is wrong because DCGAN uses batch normalisation, not softmax; D is wrong because Section 3 is exactly adding and subtracting latent vectors.

2. InfoGAN 能夠在無監督條件下實現語意解耦 (Disentangled Representation) 的核心機制為何？
   - (A) 極大化可解釋語意代碼 $\mathbf{c}$ 與生成樣本 $G(\mathbf{z},\mathbf{c})$ 之間的互資訊 (Mutual Information) $I(\mathbf{c};G(\mathbf{z},\mathbf{c}))$
   - (B) 將潛在向量 $\mathbf{z}$ 的維度設定為 $1$
   - (C) 在判別器中加入 L1 正則化
   - (D) 強制將所有圖像轉為二值化圖像

2. What is the core mechanism that lets InfoGAN disentangle meaning without supervision?
   - (A) Maximising the mutual information $I(\mathbf{c};G(\mathbf{z},\mathbf{c}))$ between the interpretable code $\mathbf{c}$ and the generated sample $G(\mathbf{z},\mathbf{c})$
   - (B) Setting the dimension of the latent vector $\mathbf{z}$ to $1$
   - (C) Adding L1 regularisation to the discriminator
   - (D) Forcing all images to be binary

**解析**：選 (A)。來源的理由：InfoGAN 加入極大化互資訊下界的目標，強迫生成樣本保留關於代碼 $\mathbf{c}$ 的明顯資訊（見第3節）。`[補充]` 其他選項：(B) 只把 $\mathbf{z}$ 變小，不會讓某一維自動代表某種性質；(C)、(D) 和解耦無關。

**Explanation**: the answer is A. The source's reason: InfoGAN adds the goal of maximising a lower bound on the mutual information, forcing generated samples to keep clear information about the code $\mathbf{c}$ (see Section 3). `[Added]` The other options: B only shrinks $\mathbf{z}$ and does not make any dimension stand for a property by itself; C and D have nothing to do with disentangling.

3. Progressive GAN (ProGAN) 成功生成 $1024\times1024$ 超高解析度逼真圖像的核心訓練策略為何？
   - (A) 一次性直接訓練 $1024\times1024$ 網絡
   - (B) 採用金字塔漸進生長策略，從 $4\times4$ 低解析度開始訓練，逐步平滑疊加高解析度網絡層
   - (C) 將圖像拆成 $1\times1$ 像素點獨立生成
   - (D) 取消所有卷積層，改用全連接層

3. What is the core training strategy that lets Progressive GAN (ProGAN) generate realistic $1024\times1024$ images?
   - (A) Training a $1024\times1024$ network directly in one go
   - (B) Growing progressively like a pyramid: start training at a low $4\times4$ resolution and smoothly add higher-resolution layers step by step
   - (C) Splitting the image into $1\times1$ pixels and generating each independently
   - (D) Dropping all convolutional layers in favour of fully connected ones

**解析**：選 (B)。來源的理由：ProGAN 先在 $4\times4$ 學基礎結構與顏色，穩定後逐步加入高解析度的層，大幅提高訓練穩定性（見第5節）。`[補充]` (A) 正是 ProGAN 要避開的不穩定做法；(C) 各像素獨立生成，像素之間沒有關聯，畫不出結構；(D) 與卷積的優勢背道而馳。

**Explanation**: the answer is B. The source's reason: ProGAN first learns basic structure and colour at $4\times4$ and, once stable, gradually adds higher-resolution layers, greatly improving training stability (see Section 5). `[Added]` A is exactly the unstable approach ProGAN avoids; C generates pixels independently with no link between them, so it cannot draw structure; D goes against the strengths of convolution.

4. 在高維潛在空間 $\mathbf{z}\sim\mathcal{N}(\mathbf{0},I)$ 中進行向量插值時，為何球面線性插值 (Slerp) 比普通線性插值 (Lerp) 更為優越？
   - (A) 因為 Lerp 直線穿過高維球面內部低機率密度區域導致畫質下降；Slerp 沿超球面弧線插值保持模長恆定與品質平滑
   - (B) 因為 Slerp 的計算速度比 Lerp 快 $1000$ 倍
   - (C) 因為 Lerp 無法處理複數向量
   - (D) 因為 Slerp 能將圖像自動轉為 3D 模型

4. When interpolating in a high-dimensional latent space $\mathbf{z}\sim\mathcal{N}(\mathbf{0},I)$, why is spherical linear interpolation (Slerp) better than ordinary linear interpolation (Lerp)?
   - (A) Because Lerp's straight line passes through the low-probability interior of the high-dimensional sphere and lowers image quality, while Slerp follows an arc on the hypersphere, keeping the length constant and the quality smooth
   - (B) Because Slerp computes $1000$ times faster than Lerp
   - (C) Because Lerp cannot handle complex vectors
   - (D) Because Slerp automatically turns images into 3D models

**解析**：選 (A)。來源的理由：高維高斯的點集中在球殼上，Lerp 的直線往球心偏離殼層，Slerp 沿大圓弧保持半徑（見第4節）。`[補充]` 「內部低機率密度」要說清楚：原點附近的**密度**其實最高，低的是那裡的**機率**（體積太小，幾乎抽不到）。「模長恆定」也只在兩端一樣長時成立。(B)、(C)、(D) 與插值的品質無關；Slerp 還比 Lerp 多算三角函數，並不比較快。

**Explanation**: the answer is A. The source's reason: high-dimensional Gaussian points crowd onto a spherical shell; Lerp's straight line drifts towards the centre, off the shell, while Slerp keeps the radius along a great-circle arc (see Section 4). `[Added]` "Low-probability-density interior" needs care: the **density** near the origin is in fact the highest; what is low there is the **probability** (the volume is tiny, so it is almost never sampled). "Keeping the length constant" also holds only when both ends have the same length. B, C and D have nothing to do with interpolation quality; Slerp even computes extra trigonometric functions and is not faster.

5. CycleGAN 能在缺乏成對 (Unpaired) 影像訓練資料下實現高品質跨風格轉換 (如馬 ↔ 斑馬) 的核心損失函數為何？
   - (A) 交叉熵損失 (Cross Entropy Loss)
   - (B) 循環一致性損失 (Cycle Consistency Loss: $\lVert F(G(\mathbf{x}))-\mathbf{x}\rVert_1$)
   - (C) 均方誤差損失 (MSE Loss)
   - (D) Hinge 損失

5. What is the core loss that lets CycleGAN translate styles well (such as horse ↔ zebra) without paired training images?
   - (A) Cross-entropy loss
   - (B) Cycle-consistency loss ($\lVert F(G(\mathbf{x}))-\mathbf{x}\rVert_1$)
   - (C) Mean squared error (MSE) loss
   - (D) Hinge loss

**解析**：選 (B)。來源的理由：循環一致性損失要求 $\mathbf{x}\to G(\mathbf{x})\to F(G(\mathbf{x}))$ 能還原成 $\mathbf{x}$，逼模型在沒有配對資料時保留原圖的結構（見第5節）。`[補充]` 它是 CycleGAN 和一般 GAN 不同的關鍵，但不是唯一的損失：GAN 的對抗損失仍然需要（見第5節的常見錯誤）。(A)、(C)、(D) 只是損失的形式：拿生成圖去比對「正確答案圖」時需要配對資料，而沒有配對時正好沒有；它們也可以當作對抗損失（CycleGAN 的對抗損失其實就用最小平方的形式），但那不是讓模型在沒有配對資料時保留原圖結構的關鍵。

**Explanation**: the answer is B. The source's reason: the cycle-consistency loss requires $\mathbf{x}\to G(\mathbf{x})\to F(G(\mathbf{x}))$ to restore $\mathbf{x}$, forcing the model to keep the original structure without paired data (see Section 5). `[Added]` It is the key difference between CycleGAN and an ordinary GAN, but not its only loss: the adversarial GAN loss is still needed (see the common mistake in Section 5). A, C and D are only loss forms: used to compare the output with a "correct answer" image they need paired data, which is exactly what is missing; they can also serve as the adversarial loss (CycleGAN's adversarial loss actually uses the least-squares form), but that is not what lets the model keep the original structure without pairs.

### 留給下一單元的問題（The question for the next unit）

第3題提到：沒有循環損失時，生成器可能不管輸入是什麼都畫同一種東西。其實一般的 GAN 也常出這個毛病：生成器發現某幾種樣本特別容易騙過判別器，就只畫那幾種，資料裡的其他類型全被忽略。這叫做什麼？判別器為什麼抓不到？下一個知識點（本週 KP06）談模式崩塌與它的對策。

Practice question 3 noted that without the cycle loss the generator might draw the same thing whatever its input. Ordinary GANs often have this problem too: the generator finds that a few kinds of samples fool the discriminator especially well and draws only those, ignoring every other type in the data. What is this called? Why does the discriminator not catch it? The next knowledge point (this week's KP06) discusses mode collapse and its remedies.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第14章「產生式對抗網絡」的 KP-6「條件式 GAN（cGAN）與 DCGAN 潛在空間向量算術」、KP-12「特徵匹配與 InfoGAN 互資訊解耦」的 InfoGAN 部分、KP-14「半監督 CatGAN、ProGAN 金字塔與雙向推斷 BiGAN/ALI」、KP-15「潛在空間球面線性插值（Slerp）與幾何流形」、KP-16「GAN 影像風格轉換（CycleGAN）與對抗自編碼器（AAE）」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch14-gan/index.html ，讀取日期 2026-11-15）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from Chapter 14, "Generative adversarial networks", of CCC-LNS IDL_115: KP-6, "Conditional GANs (cGAN) and DCGAN latent vector arithmetic"; the InfoGAN part of KP-12, "Feature matching and InfoGAN mutual-information disentangling"; KP-14, "Semi-supervised CatGAN, the ProGAN pyramid and bidirectional inference BiGAN/ALI"; KP-15, "Spherical linear interpolation (Slerp) in the latent space and the geometric manifold"; and KP-16, "GAN image style translation (CycleGAN) and the adversarial autoencoder (AAE)" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch14-gan/index.html , accessed 2026-11-15). Content not in the source is marked `[Added]` where it appears.

- 來源 KP-6 的 cGAN 公式中，真資料一項寫成 $\mathbf{x}\sim p_{\text{data}}$，但式子裡用到 $\mathbf{y}$；本講義改成成對抽樣 $(\mathbf{x},\mathbf{y})\sim p_{\text{data}}$，假資料一項也寫明 $\mathbf{y}\sim p(\mathbf{y})$。
- 來源 KP-15 說 Slerp「保持向量模長恆定」，互動展示寫中點長度 $0.707$ 對 $1.0$。長度恆定只在兩端一樣長時成立；$0.707$ 與 $1.0$ 是兩個互相垂直的單位向量的情形，不是 $\sqrt{p}$ 薄殼上的數字。本講義第4節用實際的 $p=100$ 樣本重算，並註明條件。
- 來源 KP-15 寫 Lerp 穿過「低機率密度區域」；原點附近的密度其實最高，低的是落在那裡的機率。本講義第4節與第4題解析說明。
- 來源 KP-6 的自我檢測選項 (B) 寫「以轉置卷積取代 Pooling」，沒有區分兩個網路；本講義說明生成器用轉置卷積、判別器用步長卷積。
- 來源 KP-6 說向量算術能生成「高清晰度」的影像；DCGAN 的輸出是 $64\times64$，本講義不採用這個形容。來源 KP-16 的「斑馬 ↔ 白馬」本講義依 CycleGAN 原論文寫成馬 ↔ 斑馬。
- 來源 KP-12 的特徵匹配（feature matching）屬於訓練技巧，移到本週 KP06 與模式崩塌的對策一起講；本單元只用 KP-12 的 InfoGAN 部分。
- 來源用粗斜體 `\boldsymbol` 表示向量；本講義改用 `\mathbf`。
- 本講義補上：標籤為什麼要送進判別器、轉置卷積與步長卷積的尺寸公式與逐層尺寸、向量算術用三個樣本平均、InfoGAN 的輔助網路 $Q$ 與手寫數字的例子、薄殼的原因與 $p=100$ 的數字、Lerp 與 Slerp 的長度表、$K+1$ 類判別器的訓練方式、ProGAN 的淡入、BiGAN 的成對判別、CycleGAN 的數字例子與為什麼兩種損失都要、AAE 的好處、三題小練習與解答。

- In the source's KP-6 cGAN formula, the real-data term is written $\mathbf{x}\sim p_{\text{data}}$ although $\mathbf{y}$ appears in it; this note samples pairs $(\mathbf{x},\mathbf{y})\sim p_{\text{data}}$ and states $\mathbf{y}\sim p(\mathbf{y})$ in the fake-data term.
- The source's KP-15 says Slerp "keeps the vector length constant", and its demo gives midpoint lengths of $0.707$ versus $1.0$. Constant length holds only when both ends have the same length; $0.707$ and $1.0$ are the case of two perpendicular unit vectors, not numbers on the $\sqrt{p}$ shell. Section 4 recomputes with actual $p=100$ samples and states the condition.
- The source's KP-15 says Lerp passes through a "low probability density region"; the density near the origin is in fact the highest, and what is low is the probability of landing there. Section 4 and the explanation of quiz question 4 make this clear.
- Option B of the source's KP-6 self-check says "transposed convolutions replacing pooling" without distinguishing the two networks; this note explains that the generator uses transposed convolutions and the discriminator strided convolutions.
- The source's KP-6 says vector arithmetic generates "high-definition" images; DCGAN outputs $64\times64$, so this note does not use that description. The source's KP-16 "zebra ↔ white horse" is written horse ↔ zebra, following the CycleGAN paper.
- Feature matching from the source's KP-12 is a training technique and moves to this week's KP06, together with the remedies for mode collapse; this unit uses only the InfoGAN part of KP-12.
- The source writes vectors with bold italic `\boldsymbol`; this note uses `\mathbf`.
- This note adds: why the label must go to the discriminator, the size formulas of transposed and strided convolutions with the layer-by-layer sizes, averaging three samples for vector arithmetic, InfoGAN's auxiliary network $Q$ and the handwritten-digit example, the reason for the thin shell with numbers for $p=100$, the length table for Lerp and Slerp, how the $K+1$-class discriminator is trained, ProGAN's fade-in, BiGAN's pair discrimination, CycleGAN's numerical example and why both losses are needed, the AAE's benefit, and three practice questions with solutions.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 條件式 GAN | conditional GAN (cGAN) |
| 深度卷積 GAN | deep convolutional GAN (DCGAN) |
| 轉置卷積 | transposed convolution |
| 步長卷積 | strided convolution |
| 池化 | pooling |
| 批次正規化 | batch normalization |
| 潛在空間 | latent space |
| 互資訊 | mutual information |
| 解耦 | disentangling |
| 插值 | interpolation |
| 線性插值 | linear interpolation (Lerp) |
| 球面線性插值 | spherical linear interpolation (Slerp) |
| 半監督學習 | semi-supervised learning |
| 熵 | entropy |
| 漸進式 GAN | Progressive GAN (ProGAN) |
| 雙向 GAN | bidirectional GAN (BiGAN) |
| 對抗學習推論 | adversarially learned inference (ALI) |
| 循環一致性損失 | cycle-consistency loss |
| 對抗式自編碼器 | adversarial autoencoder (AAE) |
| 模式崩塌（KP06） | mode collapse |
