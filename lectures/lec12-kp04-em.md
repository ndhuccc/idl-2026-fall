# 第12週 KP04：EM 演算法，輪流「猜原因」與「改模型」，每一輪都不會變差（The EM algorithm: take turns guessing the causes and improving the model, and no round ever makes things worse）

> 程度：大學生（undergrad），選讀／Level: Undergraduate (undergrad), optional reading

本講義是第12週的第四個知識點（KP04），接在 KP03 之後，屬於選讀。KP03 說 ELBO 有兩個旋鈕：近似後驗 $q$ 和模型參數 $\theta$。這一節介紹一個經典的作法：輪流轉兩個旋鈕，叫 EM 演算法。它比神經網路早很多，但想法和本週後面的 VAE 一脈相承。

This is the fourth knowledge point (KP04) of Week 12, following KP03, and it is optional reading. KP03 showed that the ELBO has two knobs: the approximate posterior $q$ and the model parameters $\theta$. This note introduces a classic method that turns the two knobs in turns, the EM algorithm. It is far older than neural networks, but its idea leads directly to this week's VAE.

先備知識是 KP02 的後驗、KP03 的 ELBO 與 $\log p\ge\text{ELBO}$，以及第1週的最大概似估計（MLE）。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第12章 §2.2 與教授祕笈改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec12-kp04-figures.py` 算出，可以自己重算。

The prerequisites are the posterior of KP02, the ELBO and $\log p\ge\text{ELBO}$ of KP03, and maximum likelihood estimation (MLE) from Week 1. Every term and symbol is defined when it first appears. This note is rewritten from §2.2 of Chapter 12 of the course's web material CCC-LNS and its professor's notes, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec12-kp04-figures.py`, so you can recompute it yourself.

## KP04 EM 演算法，輪流「猜原因」與「改模型」，每一輪都不會變差（The EM algorithm: take turns guessing the causes and improving the model, and no round ever makes things worse）

### 學習目標（Learning objectives）

讀完之後，你能說出 EM 的兩個步驟各做什麼，以及它們分別對應 ELBO 的哪一個旋鈕。你能在兩群常態混合的小例子裡，親手算出一個點的「軟標籤」和一輪更新後的兩群中心。你能用 KP03 的 $\log p\ge\text{ELBO}$ 說明 EM 為什麼每一輪都不會讓對數概似變差。你也能說明後驗算不出來時，變分 EM 改了哪一步、失去了什麼保證。

After reading, you can say what each of EM's two steps does and which knob of the ELBO each one turns. In a small example of a two-group normal mixture, you can compute by hand the "soft label" of one point and the two group centres after one round of updates. You can use KP03's $\log p\ge\text{ELBO}$ to explain why no round of EM makes the log-likelihood worse. You can also explain what variational EM changes when the posterior cannot be computed, and which guarantee it loses.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $X=\{x_1,\dots,x_n\}$ | 全部資料 | all the data |
| $z_i$ | 第 $i$ 筆資料的潛在變數（這裡是它屬於哪一群） | latent variable of example $i$ (here, which group it belongs to) |
| $\theta^{(t)}$ | 第 $t$ 輪的模型參數 | model parameters after round $t$ |
| $q^{(t)}$ | 第 $t$ 輪的近似後驗 | approximate posterior in round $t$ |
| $\mathcal{L}(q,\theta)$ | ELBO（KP03） | ELBO (KP03) |
| $\mu_1,\ \mu_2$ | 兩群常態分布的中心 | centres of the two normal groups |
| $\mathbf{z}=(z_1,\dots,z_n)$ | 全部資料的潛在變數 | latent variables of all the data |
| $\mathcal{N}(x;\ \mu,\ 1)$ | 平均 $\mu$、標準差 $1$ 的常態密度：$e^{-(x-\mu)^2/2}/\sqrt{2\pi}$ | normal density with mean $\mu$ and standard deviation $1$: $e^{-(x-\mu)^2/2}/\sqrt{2\pi}$ |
| $\mathbb{E}_q[\cdot]$ | 按 $q$ 的機率加權的平均（期望值） | average weighted by the probabilities of $q$ (expectation) |
| $r_i$ | 軟標籤：第 $i$ 點屬於第 1 群的機率 | soft label: probability that point $i$ belongs to group 1 |
| $\arg\max_\theta f(\theta)$ | 讓 $f$ 最大的那個 $\theta$ | the $\theta$ that makes $f$ largest |

### 1. 資料缺了一塊：用期望值補起來（Part of the data is missing: fill it with an expectation）

**最大概似估計**（maximum likelihood estimation，MLE）是挑出讓資料最可能出現的參數。資料完整時它很好用。來源說，EM 原本是為了「資料不完整」設計的：有些值沒有量到，MLE 就卡住了。

**Maximum likelihood estimation** (MLE) picks the parameters under which the data is most likely. It works well when the data is complete. The source explains that EM was designed for incomplete data: when some values were never measured, MLE gets stuck.

來源寫了一段簡化的對話：「資料有缺怎麼辦？」「用缺的部分的期望值代替。」「這樣 MLE 就能做了。」直接丟掉有缺的資料，會浪費資訊；隨便填一個數，又不準。EM 的作法是：用目前的參數，算出缺的部分「可能是什麼、各有多可能」，再拿這份補好的資料做 MLE，然後重複。

The source writes a simplified dialogue: "What if data is missing?" "Replace the missing part by its expectation." "Then MLE works." Dropping the incomplete data wastes information, and filling in an arbitrary number is inaccurate. EM's approach: with the current parameters, work out what the missing part could be and how likely each possibility is, use the filled-in data for MLE, and repeat.

潛在變數就是一種「永遠缺的資料」：它本來就看不見。所以 EM 也能用在 KP02 的潛在變數模型上，這是它和本週的關聯。

A latent variable is "data that is always missing": it is invisible by nature. That is why EM also applies to the latent-variable models of KP02, which is how it connects to this week.

用一個例子 `[補充]`。資料是 $8$ 個數：$-2.3,\ -1.7,\ -1.2,\ -0.4,\ 0.6,\ 1.3,\ 1.9,\ 2.6$。我們假設它們來自兩群常態分布，標準差都是 $1$，兩群各占一半，未知的只有兩群的中心 $\mu_1,\mu_2$。缺的資料是「每個點屬於哪一群」，也就是潛在變數 $z_i$。

An example `[Added]`. The data are $8$ numbers: $-2.3,\ -1.7,\ -1.2,\ -0.4,\ 0.6,\ 1.3,\ 1.9,\ 2.6$. We assume they come from two normal groups, both with standard deviation $1$, each holding half the data, and only the centres $\mu_1,\mu_2$ are unknown. The missing data is "which group each point belongs to", that is, the latent variable $z_i$.

如果知道每個點屬於哪一群（例如左邊四個是第 1 群、右邊四個是第 2 群），MLE 很簡單：每群取平均，得到 $-1.4$ 和 $1.6$。困難在於不知道。EM 的作法是先猜一組中心，從 $\mu_1=-0.2$、$\mu_2=0.3$ 開始，然後輪流做兩步。

If we knew which group each point belonged to (say the left four in group 1 and the right four in group 2), MLE would be easy: average each group, getting $-1.4$ and $1.6$. The difficulty is that we do not know. EM starts from a guess of the centres, $\mu_1=-0.2$ and $\mu_2=0.3$, and then takes two steps in turns.

### 2. $\text{E}$ 步與 $\text{M}$ 步（The E-step and the M-step）

**$\text{E}$ 步**（expectation step）：固定目前的參數，對每個點算它屬於各群的機率，也就是後驗。這個機率叫**軟標籤**（soft label）$r_i$：不是硬說「它屬於第 1 群」，而是說「它有多少機率屬於第 1 群」。

**E-step** (expectation step): fix the current parameters and compute, for each point, the probability that it belongs to each group, that is, the posterior. This probability is called a **soft label** $r_i$: instead of insisting "it is in group 1", it says "how likely it is to be in group 1".

以 $x_4=-0.4$ 為例，用一開始的中心 $-0.2$ 和 $0.3$。第 1 群給它的分數是 $0.5\times\mathcal{N}(-0.4;\ -0.2,\ 1)=0.1955$，第 2 群是 $0.5\times\mathcal{N}(-0.4;\ 0.3,\ 1)=0.1561$，兩群加總是 $0.3516$。這就是 KP02 的貝氏定理：$0.5$ 是先驗，$\mathcal{N}(\cdot)$ 是概似，加總是證據，相除得到後驗 $p(z_4=1\mid x_4,\theta)$：

Take $x_4=-0.4$ with the initial centres $-0.2$ and $0.3$. Group 1 gives it a score of $0.5\times\mathcal{N}(-0.4;\ -0.2,\ 1)=0.1955$, group 2 gives $0.5\times\mathcal{N}(-0.4;\ 0.3,\ 1)=0.1561$, and together they give $0.3516$. This is KP02's Bayes rule: $0.5$ is the prior, $\mathcal{N}(\cdot)$ is the likelihood, the sum is the evidence, and the ratio is the posterior $p(z_4=1\mid x_4,\theta)$:

$$
r_4=\frac{0.1955}{0.3516}=0.5560 .
$$

這個點有 $55.6\%$ 屬於第 1 群、$44.4\%$ 屬於第 2 群。八個點的 $r_i$ 依序是 $0.764,\ 0.706,\ 0.651,\ 0.556,\ 0.432,\ 0.349,\ 0.284,\ 0.218$：越左邊越像第 1 群。

This point is $55.6\%$ group 1 and $44.4\%$ group 2. The $r_i$ of the eight points are $0.764,\ 0.706,\ 0.651,\ 0.556,\ 0.432,\ 0.349,\ 0.284,\ 0.218$ in order: the further left, the more like group 1.

**$\text{M}$ 步**（maximization step）：固定軟標籤，重新估參數。每群的新中心是**加權平均**，權重就是軟標籤：

**M-step** (maximization step): fix the soft labels and re-estimate the parameters. Each group's new centre is a **weighted average**, with the soft labels as the weights:

$$
\mu_1=\frac{\sum_i r_i\,x_i}{\sum_i r_i},\qquad \mu_2=\frac{\sum_i (1-r_i)\,x_i}{\sum_i (1-r_i)} .
$$

為什麼是加權平均 `[補充]`：本例的兩群比例 $0.5$ 和標準差 $1$ 固定不調，期望對數概似裡和 $\mu_1$ 有關的部分是 $-\tfrac12\sum_i r_i(x_i-\mu_1)^2$。對 $\mu_1$ 微分並設為 $0$，得到 $\sum_i r_i(x_i-\mu_1)=0$，移項就是上式；$\mu_2$ 同理，權重換成 $1-r_i$。

Why a weighted average `[Added]`: in this example the group shares $0.5$ and the standard deviation $1$ are fixed, and the part of the expected log-likelihood that depends on $\mu_1$ is $-\tfrac12\sum_i r_i(x_i-\mu_1)^2$. Setting its derivative with respect to $\mu_1$ to $0$ gives $\sum_i r_i(x_i-\mu_1)=0$, which rearranges to the formula above; $\mu_2$ is the same with weights $1-r_i$.

第一輪算出來，中心從 $(-0.2,\ 0.3)$ 移到 $(-0.5408,\ 0.7281)$，兩群拉開了。這就是「把缺的部分用期望值補起來再做 MLE」：如果軟標籤都是 $0$ 或 $1$，這個式子就退回「每群取平均」。

After the first round the centres move from $(-0.2,\ 0.3)$ to $(-0.5408,\ 0.7281)$, and the groups spread apart. This is exactly "fill the missing part with its expectation, then do MLE": if every soft label were $0$ or $1$, the formula would reduce to "average each group".

重複幾輪，對數概似 $\log p(X\mid\theta)$ 一路上升：

Repeat for a few rounds, and the log-likelihood $\log p(X\mid\theta)$ keeps rising:

| 輪 $t$ | $\mu_1$ | $\mu_2$ | $\log p(X\mid\theta^{(t)})$ |
|---|---|---|---|
| $0$ | $-0.2000$ | $0.3000$ | $-18.0093$ |
| $1$ | $-0.5408$ | $0.7281$ | $-16.4481$ |
| $2$ | $-1.0703$ | $1.2731$ | $-14.7076$ |
| $3$ | $-1.2879$ | $1.4963$ | $-14.4825$ |
| $4$ | $-1.3198$ | $1.5286$ | $-14.4780$ |
| $20$ | $-1.3238$ | $1.5324$ | $-14.4779$ |

| round $t$ | $\mu_1$ | $\mu_2$ | $\log p(X\mid\theta^{(t)})$ |
|---|---|---|---|
| $0$ | $-0.2000$ | $0.3000$ | $-18.0093$ |
| $1$ | $-0.5408$ | $0.7281$ | $-16.4481$ |
| $2$ | $-1.0703$ | $1.2731$ | $-14.7076$ |
| $3$ | $-1.2879$ | $1.4963$ | $-14.4825$ |
| $4$ | $-1.3198$ | $1.5286$ | $-14.4780$ |
| $20$ | $-1.3238$ | $1.5324$ | $-14.4779$ |

最後的中心 $-1.32$ 和 $1.53$，和「知道標籤時」的 $-1.4$、$1.6$ 很接近，但不完全一樣：中間的點在兩群都有份量，收斂時 $-0.4$ 的軟標籤是 $0.81$、$0.6$ 是 $0.20$。

The final centres $-1.32$ and $1.53$ are close to the "labels known" answer of $-1.4$ and $1.6$, but not equal: middle points count towards both groups; at convergence the soft label of $-0.4$ is $0.81$ and that of $0.6$ is $0.20$.

![三張並排的圖，標題是第 0、1、20 輪。每張都有一條藍色鐘形曲線（第 1 群）和一條橘色鐘形曲線（第 2 群），下方是八個圓點，顏色從藍到紅表示屬於第 1 群的機率由高到低。第 0 輪兩條曲線幾乎重疊、圓點顏色很淡；第 1 輪兩條曲線拉開；第 20 輪兩條曲線分別在 -1.32 和 1.53，兩邊外側三點顏色很深，$-0.4$ 和 $0.6$ 兩點較淡（Three panels titled rounds 0, 1 and 20. Each has a blue bell curve (group 1) and an orange bell curve (group 2), with eight dots below coloured from blue to red as the probability of group 1 goes from high to low. At round 0 the curves nearly overlap and the dots are pale; at round 1 they spread apart; at round 20 they sit at -1.32 and 1.53, the outer three dots on each side dark, and the dots at -0.4 and 0.6 lighter）](../assets/figures/lec12-kp04-em-fit.png)

**圖說 `lec12-kp04-em-fit`**〔功能：一目了然、精準度〕：先看左圖：一開始兩群的中心靠得很近，每個點的顏色都很淡，表示 $\text{E}$ 步算出的軟標籤接近一半一半。再看中圖：一輪 $\text{M}$ 步之後，兩條曲線各自往自己那邊的點移動。最後看右圖：二十輪之後，曲線停在 $-1.32$ 和 $1.53$，兩邊外側三點顏色很深；$-0.4$（$r\approx0.81$）和 $0.6$（$r\approx0.20$）較淡，因為它們仍有一部分算進另一群；標題的 $\log p(X)$ 從 $-18.0093$ 升到 $-14.4779$。

**Caption `lec12-kp04-em-fit`** [Function: seeing it at a glance, precision]: first look at the left panel: at the start the two centres are close, and every dot is pale, meaning the E-step's soft labels are near half and half. Then look at the middle panel: after one M-step, each curve moves towards its own points. Finally look at the right panel: after twenty rounds the curves stop at $-1.32$ and $1.53$, the outer three dots on each side are dark, while $-0.4$ ($r\approx0.81$) and $0.6$ ($r\approx0.20$) are lighter because they still count partly towards the other group; the $\log p(X)$ in the titles rises from $-18.0093$ to $-14.4779$.

常見錯誤：以為 $\text{E}$ 步只是「把缺的值填一個數」。來源提醒，$\text{E}$ 步算的是完整的「期望對數概似」$\mathbb{E}_q[\log p(X,\mathbf{z}\mid\theta)]$：每個可能的值都按它的機率算進去。本例的軟標籤就是這樣，不是硬把點分給某一群。

Common mistake: thinking that the E-step just "fills in one number for the missing value". The source points out that the E-step computes the whole "expected log-likelihood" $\mathbb{E}_q[\log p(X,\mathbf{z}\mid\theta)]$: every possible value counts in proportion to its probability. The soft labels in this example do exactly that rather than forcing each point into one group.

### 3. 為什麼每一輪都不會變差：從 ELBO 看 EM（Why no round gets worse: EM through the ELBO）

來源把 EM 寫成輪流最大化 KP03 的 ELBO：

The source writes EM as maximising KP03's ELBO in turns:

$$
\text{E 步／E-step: } q^{(t)}=\arg\max_q\mathcal{L}(q,\theta^{(t-1)}),\qquad
\text{M 步／M-step: } \theta^{(t)}=\arg\max_\theta\mathcal{L}(q^{(t)},\theta).
$$

把 ELBO 拆開 `[補充]`：$\mathcal{L}(q,\theta)=\mathbb{E}_q[\log p(X,\mathbf{z}\mid\theta)]-\mathbb{E}_q[\log q(\mathbf{z})]$，第二項不含 $\theta$，所以 $\text{M}$ 步就是最大化期望對數概似（祕笈也這樣說）。

Split the ELBO `[Added]`: $\mathcal{L}(q,\theta)=\mathbb{E}_q[\log p(X,\mathbf{z}\mid\theta)]-\mathbb{E}_q[\log q(\mathbf{z})]$; the second term does not involve $\theta$, so the M-step simply maximises the expected log-likelihood (as the professor's notes also say).

用 KP03 的那一行 $\log p(X\mid\theta)=\mathcal{L}(q,\theta)+D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid X,\theta))$ 來看兩步：

Look at the two steps through KP03's line $\log p(X\mid\theta)=\mathcal{L}(q,\theta)+D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid X,\theta))$:

- $\text{E}$ 步固定 $\theta$，所以對數概似不動；把 ELBO 推到最大，就是把 KL 降到 $0$，也就是讓 $q$ 等於真正的後驗。這時 ELBO 剛好碰到對數概似，差距是 $0$（The E-step fixes $\theta$, so the log-likelihood does not move; pushing the ELBO to its maximum drives the KL to $0$, that is, makes $q$ the true posterior. The ELBO then touches the log-likelihood, with gap $0$）
- $\text{M}$ 步固定 $q$，調 $\theta$ 讓 ELBO 變大。KP03 說過 $\log p\ge\text{ELBO}$ 永遠成立，而 $\text{M}$ 步之前差距是 $0$，所以新的對數概似至少升了 ELBO 升的那麼多（The M-step fixes $q$ and adjusts $\theta$ to raise the ELBO. KP03 showed that $\log p\ge\text{ELBO}$ always holds, and the gap was $0$ before the M-step, so the new log-likelihood rises at least as much as the ELBO did）

寫成一串 `[補充]`：

As one chain `[Added]`:

$$
\log p(X\mid\theta^{(t)})\ \ge\ \mathcal{L}(q^{(t)},\theta^{(t)})\ \ge\ \mathcal{L}(q^{(t)},\theta^{(t-1)})\ =\ \log p(X\mid\theta^{(t-1)}).
$$

第一個 $\ge$ 是 KP03 的下界，第二個是 $\text{M}$ 步讓 ELBO 變大，最後的等號是 $\text{E}$ 步讓差距變成 $0$。

The first $\ge$ is KP03's lower bound, the second is the M-step raising the ELBO, and the final equality is the E-step closing the gap.

所以對數概似每一輪都不會下降。來源寫成 $\log p(X\mid\theta^{(t)})\ge\log p(X\mid\theta^{(t-1)})$。在上面的例子 `[補充]`，第一輪 $\text{E}$ 步之後 ELBO 等於 $-18.0093$；$\text{M}$ 步把 ELBO 推到 $-17.4091$，而新的對數概似是 $-16.4481$，比 ELBO 還高出 $0.9609$。下一輪的 $\text{E}$ 步再把 ELBO 拉上來，碰到 $-16.4481$。

So the log-likelihood never decreases from one round to the next. The source writes $\log p(X\mid\theta^{(t)})\ge\log p(X\mid\theta^{(t-1)})$. In the example above `[Added]`, after the first E-step the ELBO equals $-18.0093$; the M-step pushes the ELBO to $-17.4091$, while the new log-likelihood is $-16.4481$, $0.9609$ above the ELBO. The next round's E-step pulls the ELBO up again to touch $-16.4481$.

![一張折線圖，橫軸是第 t 輪。綠色實線是對數概似，從第 0 輪的 -18.0 一路升到第 3 輪約 -14.48 之後持平。青色虛線是 ELBO，每一輪從綠線上的點出發，往右上走到一個比綠線低的方塊點，再在下一輪被拉回綠線；方塊點和綠線之間有紅色雙箭頭，第 1 輪的標示差距 0.9609（A line chart over round t. The solid green log-likelihood rises from -18.0 at round 0 to about -14.48 at round 3 and then levels off. The dashed teal ELBO starts each round on the green line, climbs to a square below the green line, and is pulled back to the green line in the next round; red double arrows, the round-1 one between the square and the green line is labelled gap 0.9609）](../assets/figures/lec12-kp04-em-stairs.png)

**圖說 `lec12-kp04-em-stairs`**〔功能：一目了然、精準度〕：先看綠線：對數概似每一輪都上升，第 3 輪以後幾乎不動。再看青色虛線：每一段的左端是 $\text{E}$ 步之後的 ELBO，剛好在綠線上；右端是 $\text{M}$ 步之後的 ELBO，比綠線低。最後看紅色雙箭頭：$\text{M}$ 步之後 ELBO 和新的對數概似之間有差距，第 1 輪是 $0.9609$，下一輪的 $\text{E}$ 步再把它補成 $0$：圖上虛線從方塊垂直跳回綠點，就是這一步。橫軸只畫到第 6 輪，之後幾乎不動。

**Caption `lec12-kp04-em-stairs`** [Function: seeing it at a glance, precision]: first look at the green line: the log-likelihood rises every round and barely moves after round 3. Then look at the dashed teal line: the left end of each segment is the ELBO after the E-step, exactly on the green line; the right end is the ELBO after the M-step, below the green line. Finally look at the red double arrows: after the M-step there is a gap between the ELBO and the new log-likelihood, $0.9609$ in round 1, and the next E-step closes it again: the vertical jump of the dashed line from a square back to the green dot is that step. The horizontal axis stops at round 6, after which almost nothing moves.

來源的比喻：你在找山頂。$\text{E}$ 步看清楚自己站在哪條等高線上，$\text{M}$ 步沿著坡往上爬一段。兩步交替，直到爬不動為止。這個比喻的侷限 `[補充]` 是：真正的 EM 每次爬的不是對數概似本身，而是 ELBO 這個「貼著它的下界」。

The source's analogy: you are looking for a mountain top. The E-step works out which contour line you stand on, and the M-step climbs some way up the slope. The two steps alternate until you cannot climb further. The limitation of the analogy `[Added]` is that real EM does not climb the log-likelihood itself, but the ELBO, a lower bound pressed against it.

常見錯誤：以為 EM 一定找到最好的答案。來源提醒，EM 只保證不往下走；它通常停在一個再也改進不了的點（駐點），多半是局部最大值，偶爾是鞍點，不一定是全域最大值。實務上會從幾個不同的起點各跑一次，取最好的。

Common mistake: thinking EM always finds the best answer. The source points out that EM only guarantees never going downhill; it usually settles at a point where it cannot improve further (a stationary point), mostly a local maximum, occasionally a saddle point, and not necessarily the global maximum. In practice it is run from several starting points, and the best result is kept.

### 4. 後驗算不出來時：變分 EM（When the posterior cannot be computed: variational EM）

第3節的保證有一個前提：$\text{E}$ 步要真的能算出後驗。本節的混合模型可以，因為每個點只有兩種可能。可是 KP02 說過，像神經網路這種模型，後驗通常算不出來。

The guarantee of Section 3 has a premise: the E-step must actually be able to compute the posterior. The mixture model here can, since each point has only two possibilities. But as KP02 explained, for models like neural networks the posterior usually cannot be computed.

來源的處理是**變分 EM**（variational EM）：$\text{E}$ 步改成在一群簡單分布（例如常態分布）裡，找讓 ELBO 最大的 $q$。這樣 KL 不一定降到 $0$，ELBO 碰不到對數概似。代價 `[補充]` 是：只要每一步都不讓 ELBO 變小，保證只剩「ELBO 每一輪不下降」，對數概似本身不再保證每輪上升；不過 ELBO 仍是下界，對數概似至少一直待在一個上升的地板之上。

The source's remedy is **variational EM**: the E-step instead searches a family of simple distributions (such as normals) for the $q$ that maximises the ELBO. The KL then need not reach $0$, and the ELBO cannot touch the log-likelihood. The price `[Added]` is that, as long as no step lowers the ELBO, the only guarantee left is "the ELBO never decreases from one round to the next"; the log-likelihood itself is no longer guaranteed to rise every round, although, since the ELBO is still a lower bound, it at least stays above a rising floor.

`[補充]` 變分 EM 每一輪都要對每一筆資料單獨找一次最好的 $q$。資料有幾百萬筆時，這太慢了。能不能訓練一個網路，看到一筆資料 $\mathbf{x}$ 就直接吐出它的 $q$？這就是 VAE 的想法。

`[Added]` Variational EM must search for the best $q$ separately for every example in every round. With millions of examples, this is too slow. Could we train a network that looks at one example $\mathbf{x}$ and directly outputs its $q$? That is the idea of the VAE.

常見錯誤：以為 EM 只能用在常態混合。來源提醒，EM 是處理任何潛在變數模型的通用方法，常態混合只是最常見的例子。

Common mistake: thinking EM works only for normal mixtures. The source points out that EM is a general method for any latent-variable model; the normal mixture is just the most common example.

### 本單元重點整理（KP04 summary）

**一句話重點**：EM 輪流做兩步：$\text{E}$ 步固定參數、算出潛在變數的後驗（讓 ELBO 碰到對數概似），$\text{M}$ 步固定後驗、調參數讓 ELBO 變大；因為 $\log p\ge\text{ELBO}$，對數概似每一輪都不會下降，但只保證走到局部最大值；後驗算不出來時改用變分 EM，保證只剩 ELBO 不下降。

**One-sentence summary**: EM alternates two steps: the E-step fixes the parameters and computes the posterior of the latent variables (making the ELBO touch the log-likelihood), and the M-step fixes the posterior and adjusts the parameters to raise the ELBO; because $\log p\ge\text{ELBO}$, the log-likelihood never decreases from round to round, though only a local maximum is guaranteed; when the posterior cannot be computed, variational EM is used, and only the ELBO is guaranteed not to decrease.

**白話說明** `[補充]`：老師拿到一疊沒寫班級的考卷，想算兩班的平均。先隨便猜兩班平均，照猜的平均判斷每張考卷「比較像哪一班、各有幾成像」，再用這些比例重算兩班平均，一直重複，直到平均不再變。

**Plain explanation** `[Added]`: a teacher gets a pile of exam papers with no class written on them and wants the average of each of two classes. Guess the two averages, judge from them how much each paper looks like each class, recompute the two averages using those proportions, and repeat until the averages stop changing.

**正式定義**：$\text{E}$ 步 $q^{(t)}=\arg\max_q\mathcal{L}(q,\theta^{(t-1)})=p(\mathbf{z}\mid X,\theta^{(t-1)})$；$\text{M}$ 步 $\theta^{(t)}=\arg\max_\theta\mathbb{E}_{q^{(t)}}[\log p(X,\mathbf{z}\mid\theta)]$；保證 $\log p(X\mid\theta^{(t)})\ge\log p(X\mid\theta^{(t-1)})$。

**Formal definition**: E-step $q^{(t)}=\arg\max_q\mathcal{L}(q,\theta^{(t-1)})=p(\mathbf{z}\mid X,\theta^{(t-1)})$; M-step $\theta^{(t)}=\arg\max_\theta\mathbb{E}_{q^{(t)}}[\log p(X,\mathbf{z}\mid\theta)]$; guarantee $\log p(X\mid\theta^{(t)})\ge\log p(X\mid\theta^{(t-1)})$.

**例子**：$8$ 個數、兩群常態混合、從中心 $(-0.2,\ 0.3)$ 開始；$x=-0.4$ 的軟標籤是 $0.5560$；一輪後中心 $(-0.5408,\ 0.7281)$；對數概似從 $-18.0093$ 升到 $-14.4779$，最後中心 $(-1.3238,\ 1.5324)$。

**Example**: $8$ numbers, a two-group normal mixture, starting from centres $(-0.2,\ 0.3)$; the soft label of $x=-0.4$ is $0.5560$; after one round the centres are $(-0.5408,\ 0.7281)$; the log-likelihood rises from $-18.0093$ to $-14.4779$, and the final centres are $(-1.3238,\ 1.5324)$.

**常見迷思**：

- 「$\text{E}$ 步只是把缺的值填一個數。」錯，它按機率把每個可能都算進去。
- 「EM 一定找到最好的答案。」錯，只保證局部最大值。
- 「EM 只能用在常態混合。」錯，它適用於任何潛在變數模型。

**Common misconceptions**:

- "The E-step just fills in one number for the missing value." Wrong: it counts every possibility by its probability.
- "EM always finds the best answer." Wrong: only a local maximum is guaranteed.
- "EM works only for normal mixtures." Wrong: it applies to any latent-variable model.

**圖**：本單元有兩張圖。

- `lec12-kp04-em-fit`：第 0、1、20 輪的兩群曲線與軟標籤顏色。
- `lec12-kp04-em-stairs`：$\text{E}$ 步讓 ELBO 碰到對數概似，$\text{M}$ 步把兩者一起往上推。

**Figures** — this unit has two figures.

- `lec12-kp04-em-fit`: the two group curves and the soft-label colours at rounds 0, 1 and 20.
- `lec12-kp04-em-stairs`: the E-step makes the ELBO touch the log-likelihood, and the M-step pushes both up.

### 小練習（Practice）

1. 用一開始的中心 $\mu_1=-0.2$、$\mu_2=0.3$，算 $x=2.6$ 屬於第 1 群的軟標籤，取到小數點後 $3$ 位。
2. 假設三個點 $0,\ 2,\ 4$ 屬於第 1 群的軟標籤是 $1,\ 0.5,\ 0$。$\text{M}$ 步算出的 $\mu_1$ 與 $\mu_2$ 各是多少？
3. 說明為什麼 $\text{M}$ 步之後，新的對數概似一定不低於 $\text{M}$ 步之後的 ELBO。

1. With the initial centres $\mu_1=-0.2$ and $\mu_2=0.3$, compute the soft label of $x=2.6$ for group 1, to $3$ decimal places.
2. Suppose three points $0,\ 2,\ 4$ have soft labels $1,\ 0.5,\ 0$ for group 1. What are the $\mu_1$ and $\mu_2$ from the M-step?
3. Explain why, after the M-step, the new log-likelihood is never below the ELBO after the M-step.

**第 1 題解答** `[補充]`：兩群標準差相同、各占一半，所以 $r=1/\big(1+e^{(x-\mu_1)^2/2-(x-\mu_2)^2/2}\big)$；$(2.6+0.2)^2/2-(2.6-0.3)^2/2=3.92-2.645=1.275$，$r=1/(1+e^{1.275})=0.218$，和第2節的表一致。

**Solution to question 1** `[Added]`: the groups have the same standard deviation and equal shares, so $r=1/\big(1+e^{(x-\mu_1)^2/2-(x-\mu_2)^2/2}\big)$; $(2.6+0.2)^2/2-(2.6-0.3)^2/2=3.92-2.645=1.275$, and $r=1/(1+e^{1.275})=0.218$, matching Section 2.

**第 2 題解答** `[補充]`：$\mu_1=(1\times0+0.5\times2+0\times4)/(1+0.5+0)=1/1.5=0.667$；$\mu_2=(0\times0+0.5\times2+1\times4)/(0+0.5+1)=5/1.5=3.333$。

**Solution to question 2** `[Added]`: $\mu_1=(1\times0+0.5\times2+0\times4)/(1+0.5+0)=1/1.5=0.667$; $\mu_2=(0\times0+0.5\times2+1\times4)/(0+0.5+1)=5/1.5=3.333$.

**第 3 題解答** `[補充]`：KP03 的 $\log p(X\mid\theta)=\mathcal{L}(q,\theta)+D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid X,\theta))$ 對任何 $q$ 和 $\theta$ 都成立，而 KL $\ge0$，所以對數概似永遠不低於 ELBO。

**Solution to question 3** `[Added]`: KP03's $\log p(X\mid\theta)=\mathcal{L}(q,\theta)+D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid X,\theta))$ holds for any $q$ and $\theta$, and the KL is $\ge0$, so the log-likelihood is never below the ELBO.

### 小測驗（Quiz，來源原題）

來源這一段有三題自我檢測，保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The source has three self-check questions for this part, kept here with all their options; the source page does not mark the answers, so the explanations below are judged from the source text.

1. EM 的 $\text{E}$-step 做什麼？
   - (A) 更新參數 $\theta$
   - (B) 對 missing／latent 變數取期望
   - (C) 直接計算 MLE
   - (D) 做梯度下降
2. EM 演算法對 missing data 的核心想法是？
   - (A) 直接丟掉
   - (B) 用 $0$ 填補
   - (C) 用 missing 的期望填補再 MLE
   - (D) 用線性回歸估計
3. 變分推論的 $\text{E}$-step 給的 $q^{(t)}$ 是？
   - (A) 隨機初始化
   - (B) $\arg\max_q\mathcal{L}(q,\theta^{(t-1)})$，通常 $=\mathbb{P}(z\mid\mathbf{x},\theta^{(t-1)})$
   - (C) 直接等於先驗
   - (D) 固定不更新

1. What does EM's E-step do?
   - (A) Update the parameters $\theta$
   - (B) Take the expectation over the missing / latent variables
   - (C) Compute the MLE directly
   - (D) Do gradient descent
2. What is the core idea of the EM algorithm for missing data?
   - (A) Drop it
   - (B) Fill it with $0$
   - (C) Fill it with the expectation of the missing part, then do MLE
   - (D) Estimate it with linear regression
3. What is the $q^{(t)}$ given by the E-step of variational inference?
   - (A) A random initialisation
   - (B) $\arg\max_q\mathcal{L}(q,\theta^{(t-1)})$, usually $=\mathbb{P}(z\mid\mathbf{x},\theta^{(t-1)})$
   - (C) Exactly the prior
   - (D) Fixed and never updated

**解析** `[補充]`：第 1 題選 (B)：第2節，更新參數是 $\text{M}$ 步。第 2 題選 (C)：第1節。第 3 題選 (B)：第3節第一點（KL 降到 $0$ 時 $q$ 就是後驗）與重點整理的正式定義；後驗算不出來時，就是在簡單分布裡讓 ELBO 最大的那一個（第4節）。題中的 $\mathbb{P}$ 就是本講義的 $p$。

**Explanations** `[Added]`: question 1 is B: see Section 2; updating the parameters is the M-step. Question 2 is C: see Section 1. Question 3 is B: see the first bullet of Section 3 (when the KL reaches $0$, $q$ is the posterior) and the formal definition in the summary; when the posterior cannot be computed, it is the member of the simple family that maximises the ELBO (Section 4). In the questions, $\mathbb{P}$ is this note's $p$.

### 留給下一單元的問題（The question for the next unit）

第4節的結尾提出一個想法：不要每筆資料都重新找一次 $q$，而是訓練一個網路，看到 $\mathbf{x}$ 就直接吐出 $q$ 的參數；再用另一個網路當概似，從 $\mathbf{z}$ 畫回 $\mathbf{x}$。這兩個網路長什麼樣子？和 KP01 的編碼器、解碼器有什麼不同？下一個知識點（本週 KP05）介紹 VAE 的架構。

Section 4 ended with an idea: instead of searching for $q$ anew for every example, train a network that looks at $\mathbf{x}$ and directly outputs the parameters of $q$; and use another network as the likelihood, drawing $\mathbf{x}$ back from $\mathbf{z}$. What do these two networks look like, and how do they differ from KP01's encoder and decoder? The next knowledge point (this week's KP05) introduces the architecture of the VAE.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第12章「變分模型」§2.2–§2.2.2 與教授祕笈「EM：missing data 的迭代解法」「EM 背景：把缺失值換成期望」「變分推論的 EM」（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch12-variational-models/index.html ，讀取日期 2026-11-14）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §2.2–§2.2.2 of Chapter 12, "Variational models", of CCC-LNS IDL_115, with the professor's notes "EM: an iterative remedy for missing data", "EM background: replace missing values by expectations" and "EM for variational inference" (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch12-variational-models/index.html , accessed 2026-11-14). Content not in the source is marked `[Added]` where it appears.

- 來源正文說 EM「每一步都單調提升 log-likelihood」（祕笈已加上「或持平」）；精確說法是「不會下降」（可能持平），本講義這樣寫。
- 來源說 EM「一定收斂到 local maximum」；精確地說，它通常停在一個再也改進不了的點（駐點），多半是局部最大值，少數情況是鞍點。本講義寫「通常是局部最大值」。
- 來源把變分 EM 的 $\text{E}$ 步寫成「$q^{(t)}\leftarrow\mathbb{P}(z\mid\mathbf{x},\theta^{(t-1)})$」，並在括號裡說後驗不可解時只能在 $q$ 的家族裡找最好的；本講義把後者獨立成第4節，並說明這時只剩 ELBO 不下降的保證。
- 來源用 $\mathbb{P}(\cdot)$ 與粗斜體 `\boldsymbol`；本講義改用 $p(\cdot)$ 與 `\mathbf`。小測驗保留來源原本的記號。
- 本講義補上：$8$ 個數的兩群混合例子、軟標籤與加權平均的計算、迭代表、兩張圖、$\text{M}$ 步之後差距的數字、變分 EM 的保證、考卷的比喻、三題小練習與解答。
- 取捨：來源的血壓缺值例子與互動示範本講義不寫。

- The source's main text says EM "monotonically increases the log-likelihood at every step" (the professor's notes already add "or stays the same"); precisely, it "never decreases" (it may stay the same), and this note says so.
- The source says EM "is sure to converge to a local maximum"; precisely, it usually settles at a point where it cannot improve further (a stationary point), mostly a local maximum and occasionally a saddle point. This note writes "usually a local maximum".
- The source writes variational EM's E-step as "$q^{(t)}\leftarrow\mathbb{P}(z\mid\mathbf{x},\theta^{(t-1)})$" and adds in parentheses that when the posterior is intractable one can only search the family of $q$; this note makes the latter its own Section 4 and explains that only the guarantee "the ELBO does not decrease" remains.
- The source uses $\mathbb{P}(\cdot)$ and bold italic `\boldsymbol`; this note uses $p(\cdot)$ and `\mathbf`. The quiz keeps the source's original notation.
- This note adds: the two-group mixture of $8$ numbers, the computation of soft labels and weighted averages, the table of rounds, both figures, the numbers for the gap after the M-step, the guarantee of variational EM, the exam-paper analogy, and three practice questions with solutions.
- Choices: the source's blood-pressure example of missing data and its interactive demos are not covered.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| EM 演算法 | expectation–maximization (EM) algorithm |
| $\text{E}$ 步 | expectation step (E-step) |
| $\text{M}$ 步 | maximization step (M-step) |
| 最大概似估計 | maximum likelihood estimation (MLE) |
| 缺失資料 | missing data |
| 軟標籤 | soft label |
| 加權平均 | weighted average |
| 常態混合模型 | normal (Gaussian) mixture model |
| 局部最大值 | local maximum |
| 全域最大值 | global maximum |
| 鞍點 | saddle point |
| 變分 EM | variational EM |
| 證據下界（KP03） | evidence lower bound (ELBO) |
| 變分自編碼器（KP05） | variational autoencoder (VAE) |
