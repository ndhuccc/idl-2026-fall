# 第12週 KP03：ELBO，算不出來的對數概似有一個算得出來的下界，兩者只差一個 KL（The ELBO: the log-likelihood we cannot compute has a lower bound we can, and the two differ by exactly one KL）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第12週的第三個知識點（KP03），接在 KP02 之後。KP02 卡在一個地方：要讓近似後驗 $q$ 接近真正的後驗，就要最小化 $D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid\mathbf{x},\theta))$，可是這個量含有算不出來的後驗。這一節把它改寫，得到一個算得出來的量 ELBO，並說明「最大化 ELBO」為什麼同時做到兩件事。

This is the third knowledge point (KP03) of Week 12, following KP02. KP02 got stuck at one point: to bring the approximate posterior $q$ close to the true posterior we must minimise $D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid\mathbf{x},\theta))$, but this quantity contains the posterior we cannot compute. This note rewrites it into a computable quantity, the ELBO, and explains why "maximising the ELBO" achieves two things at once.

先備知識是 KP02 的先驗、概似、後驗、證據與 KL 散度，以及高中的對數運算 $\log(ab)=\log a+\log b$。第一次出現的詞與符號都會定義。本講義依課程網頁教材 CCC-LNS 第12章 §2.1、§3 開頭的 VAE 目標與教授祕笈「ELBO」改寫，文中「來源」就是指它；來源沒有、由本講義補上的內容標 `[補充]`。所有數字都由 `assets/figures/src/lec12-kp03-figures.py` 算出，可以自己重算。

The prerequisites are the prior, likelihood, posterior, evidence and KL divergence of KP02, and the high-school log rule $\log(ab)=\log a+\log b$. Every term and symbol is defined when it first appears. This note is rewritten from §2.1 of Chapter 12 of the course's web material CCC-LNS, the VAE objective at the start of §3, and the professor's note on the ELBO, which is what "the source" means below; content added by this note is marked `[Added]`. Every number is computed by `assets/figures/src/lec12-kp03-figures.py`, so you can recompute it yourself.

## KP03 ELBO，算不出來的對數概似有一個算得出來的下界，兩者只差一個 KL（The ELBO: the log-likelihood we cannot compute has a lower bound we can, and the two differ by exactly one KL）

### 學習目標（Learning objectives）

讀完之後，你能一步步推出 $\log p(\mathbf{x}\mid\theta)=\text{ELBO}+D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid\mathbf{x},\theta))$，並說明為什麼 ELBO 是對數概似的下界。你能在 KP02 的小例子裡算出三種 $q$ 的 ELBO 與差距，並核對兩者相加不變。你能把 ELBO 拆成「重建項減去離先驗的 KL」，說出兩項各在拉什麼。你也能說明最大化 ELBO 對 $q$ 和對 $\theta$ 各有什麼作用。

After reading, you can derive $\log p(\mathbf{x}\mid\theta)=\text{ELBO}+D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid\mathbf{x},\theta))$ step by step and explain why the ELBO is a lower bound on the log-likelihood. You can compute the ELBO and the gap for three choices of $q$ in KP02's small example, and check that their sum stays the same. You can split the ELBO into "a reconstruction term minus the KL from the prior" and say what each term pulls towards. You can also explain what maximising the ELBO does for $q$ and for $\theta$.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\log p(\mathbf{x}\mid\theta)$ | 對數概似（對數證據）：資料在模型下的對數機率密度 | log-likelihood (log-evidence): log density of the data under the model |
| $p(\mathbf{x},\mathbf{z}\mid\theta)$ | 聯合分布：先驗乘概似 $p(\mathbf{x}\mid\mathbf{z},\theta)p(\mathbf{z})$ | joint distribution: prior times likelihood $p(\mathbf{x}\mid\mathbf{z},\theta)p(\mathbf{z})$ |
| $q(\mathbf{z})$ | 近似後驗（KP02） | approximate posterior (KP02) |
| $\mathbb{E}_{q}[f(\mathbf{z})]$ | 期望值：$\mathbf{z}$ 照 $q$ 抽樣時 $f$ 的平均，等於 $\int q(\mathbf{z})f(\mathbf{z})\,d\mathbf{z}$ | expectation: the average of $f$ when $\mathbf{z}$ is drawn from $q$, equal to $\int q(\mathbf{z})f(\mathbf{z})\,d\mathbf{z}$ |
| $\mathcal{L}(q,\theta)$ | ELBO，證據下界 | ELBO, the evidence lower bound |
| $\log$ | 自然對數（以 $e$ 為底） | natural logarithm (base $e$) |

### 1. 把 KL 拆開，算不出來的部分剛好是一個常數（Open up the KL: the part we cannot compute is just a constant）

先寫出 KP02 的 KL。KP02 的定義是 $\int q\log(q/p)\,d\mathbf{z}$，用符號表的期望值寫法，就是 $\mathbb{E}_q[\log q-\log p]$：

Start from KP02's KL. KP02 defined it as $\int q\log(q/p)\,d\mathbf{z}$, which in the expectation notation of the symbol table is $\mathbb{E}_q[\log q-\log p]$:

$$
D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid\mathbf{x},\theta))=\mathbb{E}_{q}\big[\log q(\mathbf{z})-\log p(\mathbf{z}\mid\mathbf{x},\theta)\big].
$$

麻煩的是後驗。用貝氏定理把它換掉：$p(\mathbf{z}\mid\mathbf{x},\theta)=p(\mathbf{x},\mathbf{z}\mid\theta)/p(\mathbf{x}\mid\theta)$，取對數得到 $\log p(\mathbf{z}\mid\mathbf{x},\theta)=\log p(\mathbf{x},\mathbf{z}\mid\theta)-\log p(\mathbf{x}\mid\theta)$。代回去：

The troublemaker is the posterior. Replace it with Bayes' rule: $p(\mathbf{z}\mid\mathbf{x},\theta)=p(\mathbf{x},\mathbf{z}\mid\theta)/p(\mathbf{x}\mid\theta)$, and taking logs gives $\log p(\mathbf{z}\mid\mathbf{x},\theta)=\log p(\mathbf{x},\mathbf{z}\mid\theta)-\log p(\mathbf{x}\mid\theta)$. Substitute back:

$$
D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid\mathbf{x},\theta))=\mathbb{E}_{q}\big[\log q(\mathbf{z})-\log p(\mathbf{x},\mathbf{z}\mid\theta)\big]+\log p(\mathbf{x}\mid\theta).
$$

中間一步是 $\mathbb{E}_q[\log q-\log p(\mathbf{x},\mathbf{z}\mid\theta)+\log p(\mathbf{x}\mid\theta)]$。最後一項 $\log p(\mathbf{x}\mid\theta)$ 和 $\mathbf{z}$ 無關，是一個常數；$q$ 的總面積是 $1$，所以常數的期望值就是它自己，可以拿到期望值外面。再把 KL 移到等號另一邊、兩邊同乘 $-1$，括號裡的 $\log q-\log p(\mathbf{x},\mathbf{z}\mid\theta)$ 就變成 $\log p(\mathbf{x},\mathbf{z}\mid\theta)-\log q$。這樣就得到本單元最重要的一行：

The middle step is $\mathbb{E}_q[\log q-\log p(\mathbf{x},\mathbf{z}\mid\theta)+\log p(\mathbf{x}\mid\theta)]$. The last term $\log p(\mathbf{x}\mid\theta)$ does not depend on $\mathbf{z}$, so it is a constant; $q$ has total area $1$, so the expectation of a constant is the constant itself, and it comes out of the expectation. Then move the KL to the other side and multiply both sides by $-1$, which turns $\log q-\log p(\mathbf{x},\mathbf{z}\mid\theta)$ into $\log p(\mathbf{x},\mathbf{z}\mid\theta)-\log q$. This gives the most important line of this unit:

$$
\log p(\mathbf{x}\mid\theta)=\underbrace{\mathbb{E}_{q}\big[\log p(\mathbf{x},\mathbf{z}\mid\theta)-\log q(\mathbf{z})\big]}_{\mathcal{L}(q,\theta)\ =\ \text{ELBO}}+D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid\mathbf{x},\theta)).
$$

左邊的對數概似（又叫對數證據）是我們最想要、卻算不出來的。右邊第一項叫 **ELBO**（evidence lower bound，證據下界），只用到聯合分布 $p(\mathbf{x},\mathbf{z}\mid\theta)$ 和 $q$，兩者在任何一個 $\mathbf{z}$ 都算得出值：聯合分布就是「先驗乘概似」，不需要對所有 $\mathbf{z}$ 積分的證據。外面那個對 $q$ 的平均，可以從 $q$ 抽樣來估計（第2節）。

The log-likelihood on the left (also called the log-evidence) is what we want most but cannot compute. The first term on the right is the **ELBO** (evidence lower bound); it uses only the joint distribution $p(\mathbf{x},\mathbf{z}\mid\theta)$ and $q$, both of which can be evaluated at any $\mathbf{z}$: the joint is just "prior times likelihood", without the evidence's integral over all $\mathbf{z}$. The outer average over $q$ can be estimated by sampling from $q$ (Section 2).

右邊第二項是 KL 散度，KP02 說過它一定 $\ge0$。所以：

The second term on the right is the KL divergence, which KP02 stated is always $\ge0$. Therefore:

$$
\mathcal{L}(q,\theta)\le\log p(\mathbf{x}\mid\theta),
$$

而且只有在 $q$ 剛好等於真正的後驗時才相等。這就是「下界」的意思：ELBO 永遠不會超過對數概似，兩者的差距正好是 $q$ 和後驗之間的 KL。

with equality only when $q$ is exactly the true posterior. That is what "lower bound" means: the ELBO never exceeds the log-likelihood, and the gap between them is exactly the KL between $q$ and the posterior.

來源祕笈的小例子：若 $\log p(\mathbf{x})=0$、KL 是 $0.7$，那 ELBO 就是 $0-0.7=-0.7$。

The source notes' small example: if $\log p(\mathbf{x})=0$ and the KL is $0.7$, then the ELBO is $0-0.7=-0.7$.

常見錯誤：以為 ELBO 是對數概似的「近似值」。它是下界，不是估計：它一定比較小，差距就是 KL；只有 KL 為 $0$ 時兩者才一樣。

Common mistake: thinking the ELBO is an "approximation" of the log-likelihood. It is a lower bound, not an estimate: it is always smaller, by exactly the KL; the two agree only when the KL is $0$.

### 2. 用 KP02 的例子算一遍（Compute it with KP02's example）

沿用 KP02 的線性模型（本節的例子、表格、抽樣估計與圖都是 `[補充]`）：$z\sim\mathcal{N}(0,1)$，$x\mid z\sim\mathcal{N}(z,0.5^2)$，觀察到 $x=1.2$。KP02 算過證據 $p(x)=0.2006$，所以 $\log p(x)=-1.6065$。真正的後驗是 $\mathcal{N}(0.96,\ 0.2)$。這個例子什麼都有公式解，正好拿來核對第1節那一行。

Keep KP02's linear model (this section's example, table, sampling estimate and figure are all `[Added]`): $z\sim\mathcal{N}(0,1)$, $x\mid z\sim\mathcal{N}(z,0.5^2)$, and we observe $x=1.2$. KP02 computed the evidence $p(x)=0.2006$, so $\log p(x)=-1.6065$. The true posterior is $\mathcal{N}(0.96,\ 0.2)$. Everything in this example has a closed form, which makes it perfect for checking the line of Section 1.

挑三個 $q$ 來算：

Try three choices of $q$:

| $q$ | ELBO | 差距 $D_{\text{KL}}(q\,\Vert\,p(z\mid x))$ | 兩者相加 |
|---|---|---|---|
| 真正的後驗 $\mathcal{N}(0.96,\ 0.2)$ | $-1.6065$ | $0$ | $-1.6065$ |
| 平均偏掉的 $\mathcal{N}(0.5,\ 0.2)$ | $-2.1355$ | $0.5290$ | $-1.6065$ |
| 先驗本身 $\mathcal{N}(0,\ 1)$ | $-5.1058$ | $3.4993$ | $-1.6065$ |

| $q$ | ELBO | gap $D_{\text{KL}}(q\,\Vert\,p(z\mid x))$ | sum |
|---|---|---|---|
| the true posterior $\mathcal{N}(0.96,\ 0.2)$ | $-1.6065$ | $0$ | $-1.6065$ |
| shifted mean $\mathcal{N}(0.5,\ 0.2)$ | $-2.1355$ | $0.5290$ | $-1.6065$ |
| the prior itself $\mathcal{N}(0,\ 1)$ | $-5.1058$ | $3.4993$ | $-1.6065$ |

表裡的數字照下面三個公式算（$\sigma^2=0.25$、$x=1.2$、$q=\mathcal{N}(m,s^2)$，真正後驗是 $\mathcal{N}(0.96,0.2)$）：

$$
\mathbb{E}_q[\log p(x\mid z)]=-\tfrac12\log(2\pi\sigma^2)-\frac{(x-m)^2+s^2}{2\sigma^2},\qquad
D_{\text{KL}}(q\,\Vert\,\mathcal{N}(0,1))=\tfrac12\big(s^2+m^2-1-\log s^2\big),
$$

$$
D_{\text{KL}}(q\,\Vert\,\mathcal{N}(\mu,v))=\tfrac12\Big(\frac{s^2}{v}+\frac{(m-\mu)^2}{v}-1+\log\frac{v}{s^2}\Big).
$$

以 $m=0.5$、$s^2=0.2$ 為例：第一項 $-\tfrac12\log(0.5\pi)-0.69/0.5=-0.2258-1.38=-1.6058$；離先驗的 KL $\tfrac12(0.2+0.25-1+1.6094)=0.5297$；ELBO 是兩者相減（第4節會說明為什麼）$-1.6058-0.5297=-2.1355$；離後驗的 KL $\tfrac12(1+0.2116/0.2-1+0)=0.5290$。

The numbers in the table come from three formulas ($\sigma^2=0.25$, $x=1.2$, $q=\mathcal{N}(m,s^2)$, true posterior $\mathcal{N}(0.96,0.2)$), shown above. Take $m=0.5$, $s^2=0.2$: the first term is $-\tfrac12\log(0.5\pi)-0.69/0.5=-0.2258-1.38=-1.6058$; the KL from the prior is $\tfrac12(0.2+0.25-1+1.6094)=0.5297$; the ELBO is the first minus the second (Section 4 explains why), $-1.6058-0.5297=-2.1355$; and the KL from the posterior is $\tfrac12(1+0.2116/0.2-1+0)=0.5290$.

三列的最後一欄都一樣：不管 $q$ 怎麼選，ELBO 加差距永遠等於 $\log p(x)$。$q$ 越像後驗，ELBO 越高、差距越小。

The last column is the same in all three rows: whatever $q$ is, the ELBO plus the gap always equals $\log p(x)$. The more $q$ looks like the posterior, the higher the ELBO and the smaller the gap.

ELBO 是一個期望值，所以也可以抽樣估計 `[補充]`：從 $q=\mathcal{N}(0.5,\ 0.2)$ 抽 $200000$ 個 $z$，對每個 $z$ 算 $\log p(x,z)-\log q(z)$ 再平均，得到 $-2.1312$，和公式值 $-2.1355$ 很接近。神經網路的模型沒有公式解時，就靠這種抽樣來估 ELBO，本週 KP06 會用到。

The ELBO is an expectation, so it can also be estimated by sampling `[Added]`: draw $200000$ values of $z$ from $q=\mathcal{N}(0.5,\ 0.2)$, compute $\log p(x,z)-\log q(z)$ for each, and average; the result is $-2.1312$, close to the formula's $-2.1355$. When a neural-network model has no closed form, this sampling is how the ELBO is estimated, as this week's KP06 will show.

![一張圖，橫軸是 q 的平均 m，q 的變異數固定為 0.2。綠色水平線是 log p(x) 等於負 1.6065，不隨 q 改變。深青色曲線是 ELBO，像倒過來的碗，最高點在 m 等於 0.96，剛好碰到綠線，標示差距 0。兩線之間塗成淺青色，代表差距等於 KL。m 等於 0.5 的紫點標示差距 0.5290（A plot over the mean m of q, with q's variance fixed at 0.2. A green horizontal line is log p(x) equals minus 1.6065, independent of q. A dark-teal ELBO curve shaped like an upside-down bowl peaks at m equals 0.96, just touching the green line, labelled gap 0. The area between the lines is shaded light teal, meaning the gap equals the KL. A purple dot at m equals 0.5 is labelled gap 0.5290）](../assets/figures/lec12-kp03-elbo-gap.png)

**圖說 `lec12-kp03-elbo-gap`**〔功能：一目了然、精準度〕：先看綠線：$\log p(x)$ 是一條水平線，因為它和 $q$ 無關。再看青色曲線：$q$ 的變異數固定為 $0.2$，ELBO 隨 $q$ 的平均 $m$ 改變，從不高於綠線。中間的淺青色就是差距，也就是 KL。最後看兩個點：紅點在 $m=0.96$，$q$ 等於真正的後驗，曲線碰到綠線、差距為 $0$；紫點在 $m=0.5$，差距是 $0.5290$。

**Caption `lec12-kp03-elbo-gap`** [Function: seeing it at a glance, precision]: first look at the green line: $\log p(x)$ is horizontal because it does not depend on $q$. Then look at the teal curve: with $q$'s variance fixed at $0.2$, the ELBO changes with the mean $m$ of $q$ and is never above the green line. The light teal between them is the gap, that is, the KL. Finally look at the two dots: the red dot at $m=0.96$ is where $q$ equals the true posterior, the curve touches the green line and the gap is $0$; the purple dot at $m=0.5$ has gap $0.5290$.

### 3. 最大化 ELBO 一次做兩件事（Maximising the ELBO does two jobs at once）

第1節那一行告訴我們怎麼用 ELBO。分兩種情況看。

The line of Section 1 tells us how to use the ELBO. Look at two cases.

**固定 $\theta$，調 $q$。** 這時左邊的 $\log p(\mathbf{x}\mid\theta)$ 是固定的數。ELBO 加 KL 等於一個常數，所以 ELBO 變大，KL 就一定變小。來源說「最大化 ELBO 等價於最小化 KL」，指的就是這種情況。KP02 卡住的問題解開了：我們不必算 KL，只要把 ELBO 往上推。

**Fix $\theta$, adjust $q$.** Then $\log p(\mathbf{x}\mid\theta)$ on the left is a fixed number. ELBO plus KL equals a constant, so when the ELBO goes up, the KL must go down. The source's "maximising the ELBO is equivalent to minimising the KL" refers to exactly this case. KP02's problem is solved: we need not compute the KL, only push the ELBO up.

**調 $\theta$。** 這時 $\log p(\mathbf{x}\mid\theta)$ 也會變。ELBO 是它的下界，把下界往上推，通常也把模型的對數概似往上推，也就是讓模型更能解釋資料。這就是在訓練模型。`[補充]` 能確定的是 $\log p\ge$ ELBO 永遠成立：如果調整前差距是 $0$（例如 $q$ 剛好是後驗），調整後 ELBO 又比原來的 $\log p$ 高，那新的對數概似一定也升高了，至少升高同樣的量。KP04 的 EM 就靠這一點保證不會變差。

**Adjust $\theta$.** Now $\log p(\mathbf{x}\mid\theta)$ changes too. The ELBO is its lower bound, and pushing the bound up usually pushes the model's log-likelihood up as well, making the model explain the data better. This is training the model. `[Added]` What is certain is that $\log p\ge$ ELBO always holds: if the gap was $0$ before the adjustment (for example, $q$ is exactly the posterior) and afterwards the ELBO is higher than the old $\log p$, then the new log-likelihood must have risen too, by at least as much. KP04's EM relies on exactly this to never get worse.

所以只要一個目標「把 ELBO 推到最大」，就同時在做兩件事：讓 $q$ 接近後驗，以及讓模型解釋資料。本週 KP04 的 EM 演算法把這兩件事輪流做，KP05、KP06 的 VAE 則用梯度一起做。

So a single goal, "push the ELBO as high as possible", does two jobs at once: it brings $q$ close to the posterior and makes the model explain the data. This week's KP04 (the EM algorithm) does the two jobs in turns, while the VAE of KP05 and KP06 does them together with gradients.

常見錯誤 `[補充]`：以為 ELBO 變大，對數概似就一定變大。不一定：$\theta$ 改變時，差距也會變，ELBO 升高可能只是差距變小了。固定 $\theta$ 時，對數概似根本不動，動的是 KL。只有像上一段那樣，ELBO 升到比原來的 $\log p$ 還高，才能確定對數概似升高了。

Common mistake `[Added]`: thinking that a larger ELBO always means a larger log-likelihood. Not necessarily: when $\theta$ changes the gap changes too, and a rising ELBO may only mean a shrinking gap. With $\theta$ fixed, the log-likelihood does not move at all; what moves is the KL. Only when the ELBO rises above the old $\log p$, as in the previous paragraph, is the log-likelihood certain to have risen.

### 4. ELBO 的另一種寫法：重建減去離先驗的距離（Another way to write the ELBO: reconstruction minus distance from the prior）

把聯合分布拆成概似乘先驗 `[補充：推導步驟]`，$\log p(\mathbf{x},\mathbf{z}\mid\theta)=\log p(\mathbf{x}\mid\mathbf{z},\theta)+\log p(\mathbf{z})$，代進 ELBO，後兩項 $\log p(\mathbf{z})-\log q(\mathbf{z})$ 的期望值剛好是負的 KL：

Split the joint into likelihood times prior `[Added: derivation step]`, $\log p(\mathbf{x},\mathbf{z}\mid\theta)=\log p(\mathbf{x}\mid\mathbf{z},\theta)+\log p(\mathbf{z})$, and substitute into the ELBO; the expectation of the last two terms, $\log p(\mathbf{z})-\log q(\mathbf{z})$, is exactly the negative of a KL:

$$
\mathcal{L}(q,\theta)=\underbrace{\mathbb{E}_{q}\big[\log p(\mathbf{x}\mid\mathbf{z},\theta)\big]}_{\text{重建 reconstruction}}-\underbrace{D_{\text{KL}}(q(\mathbf{z})\,\Vert\,p(\mathbf{z}))}_{\text{離先驗多遠 distance from the prior}}.
$$

注意這個 KL 和第1節的不同：它比的是 $q$ 和**先驗**，不是和後驗，所以算得出來。第一項叫**重建項**：照 $q$ 抽一個原因 $\mathbf{z}$，它能把資料 $\mathbf{x}$ 解釋得多好。第二項是**正規化項**：不准 $q$ 離先驗太遠。來源在 §3 介紹 VAE 時寫的就是這個式子。

Note that this KL differs from the one in Section 1: it compares $q$ with the **prior**, not the posterior, so it can be computed. The first term is the **reconstruction term**: draw a cause $\mathbf{z}$ from $q$, and see how well it explains the data $\mathbf{x}$. The second is a **regularisation term**: it does not let $q$ stray too far from the prior. This is the formula the source writes when it introduces the VAE in §3.

兩項在拉不同的方向 `[補充：例子]`。在 KP02 的例子裡，固定 $q$ 的變異數為 $0.2$，只改平均 $m$：重建項希望 $m$ 靠近資料 $x=1.2$，正規化項希望 $m$ 靠近先驗的中心 $0$。兩者相加後，最好的 $m$ 是 $0.96$，正好是真正後驗的平均。以真正的後驗為例，重建項是 $-0.7410$，正規化項是 $0.8655$，ELBO 是 $-0.7410-0.8655=-1.6065$。

The two terms pull in different directions `[Added: example]`. In KP02's example, fix the variance of $q$ at $0.2$ and change only the mean $m$: the reconstruction term wants $m$ near the data $x=1.2$, and the regularisation term wants $m$ near the prior's centre $0$. Adding the two, the best $m$ is $0.96$, exactly the mean of the true posterior. For the true posterior itself, the reconstruction term is $-0.7410$, the regularisation term is $0.8655$, and the ELBO is $-0.7410-0.8655=-1.6065$.

![一張圖，橫軸是 q 的平均 m。藍線是重建項，最高點在 m 等於 1.2；橘線是負的 KL，最高點在 m 等於 0；深青色粗線是兩者相加的 ELBO，最高點在 m 等於 0.96。三條垂直虛線分別標示 m 等於 0、0.96、1.2（A plot over the mean m of q. The blue reconstruction term peaks at m equals 1.2; the orange negative KL peaks at m equals 0; the thick dark-teal ELBO, their sum, peaks at m equals 0.96. Three dotted vertical lines mark m equals 0, 0.96 and 1.2）](../assets/figures/lec12-kp03-elbo-terms.png)

**圖說 `lec12-kp03-elbo-terms`**〔功能：一目了然、精準度〕：先看藍線：只看重建，$q$ 的平均最好放在資料 $x=1.2$。再看橘線：只看正規化，最好放在先驗的中心 $0$。最後看青色粗線：兩者相加，最高點落在中間的 $0.96$，這就是真正後驗的平均。ELBO 在「解釋資料」和「別離先驗太遠」之間找平衡。

**Caption `lec12-kp03-elbo-terms`** [Function: seeing it at a glance, precision]: first look at the blue curve: for reconstruction alone, the mean of $q$ is best at the data $x=1.2$. Then look at the orange curve: for regularisation alone, it is best at the prior's centre $0$. Finally look at the thick teal curve: their sum peaks in between, at $0.96$, which is the mean of the true posterior. The ELBO balances "explain the data" against "do not stray far from the prior".

常見錯誤 `[補充]`：把這一節的 KL（$q$ 對先驗）和第1節的 KL（$q$ 對後驗）搞混。前者算得出來，是 ELBO 的一部分；後者算不出來，是 ELBO 和對數概似之間的差距。

Common mistake `[Added]`: confusing this section's KL ($q$ against the prior) with Section 1's KL ($q$ against the posterior). The former can be computed and is part of the ELBO; the latter cannot be computed and is the gap between the ELBO and the log-likelihood.

### 本單元重點整理（KP03 summary）

**一句話重點**：$\log p(\mathbf{x}\mid\theta)=\mathcal{L}(q,\theta)+D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid\mathbf{x},\theta))$；ELBO 只用到聯合分布和 $q$，算得出來，又因為 KL $\ge0$ 而成為對數概似的下界；固定 $\theta$ 時最大化 ELBO 就是最小化那個算不出來的 KL，調 $\theta$ 時則在訓練模型；ELBO 也等於「重建項減去 $q$ 離先驗的 KL」。

**One-sentence summary**: $\log p(\mathbf{x}\mid\theta)=\mathcal{L}(q,\theta)+D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}\mid\mathbf{x},\theta))$; the ELBO uses only the joint distribution and $q$, so it can be computed, and because the KL is $\ge0$ it is a lower bound on the log-likelihood; with $\theta$ fixed, maximising the ELBO minimises the uncomputable KL, and adjusting $\theta$ trains the model; the ELBO also equals "the reconstruction term minus the KL of $q$ from the prior".

**白話說明** `[補充]`：你想知道一座山有多高，卻量不到山頂。可是你能量到一個一定比山頂低的平台，而且平台離山頂的距離，剛好就是「你的猜測錯了多少」。把平台往上推，要嘛是猜得更準，要嘛是整座山變高了。

**Plain explanation** `[Added]`: you want to know how high a mountain is but cannot measure the summit. You can, however, measure a platform that is always lower than the summit, and the distance from the platform to the summit is exactly "how wrong your guess is". Pushing the platform up means either guessing better or making the whole mountain higher.

**正式定義**：$\mathcal{L}(q,\theta)=\mathbb{E}_{q}[\log p(\mathbf{x},\mathbf{z}\mid\theta)-\log q(\mathbf{z})]=\mathbb{E}_{q}[\log p(\mathbf{x}\mid\mathbf{z},\theta)]-D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}))\le\log p(\mathbf{x}\mid\theta)$，等號在 $q=p(\mathbf{z}\mid\mathbf{x},\theta)$ 時成立。

**Formal definition**: $\mathcal{L}(q,\theta)=\mathbb{E}_{q}[\log p(\mathbf{x},\mathbf{z}\mid\theta)-\log q(\mathbf{z})]=\mathbb{E}_{q}[\log p(\mathbf{x}\mid\mathbf{z},\theta)]-D_{\text{KL}}(q\,\Vert\,p(\mathbf{z}))\le\log p(\mathbf{x}\mid\theta)$, with equality when $q=p(\mathbf{z}\mid\mathbf{x},\theta)$.

**例子**：KP02 的模型、$x=1.2$，$\log p(x)=-1.6065$；$q$ 是後驗時 ELBO $-1.6065$、差距 $0$；$q=\mathcal{N}(0.5,0.2)$ 時 ELBO $-2.1355$、差距 $0.5290$；$q=\mathcal{N}(0,1)$ 時 ELBO $-5.1058$、差距 $3.4993$；三者相加都是 $-1.6065$。

**Example**: KP02's model with $x=1.2$ has $\log p(x)=-1.6065$; with $q$ the posterior, the ELBO is $-1.6065$ and the gap $0$; with $q=\mathcal{N}(0.5,0.2)$, the ELBO is $-2.1355$ and the gap $0.5290$; with $q=\mathcal{N}(0,1)$, the ELBO is $-5.1058$ and the gap $3.4993$; each sum is $-1.6065$.

**常見迷思**：

- 「ELBO 是對數概似的近似值。」錯，它是下界，差距是 KL。
- 「ELBO 變大，對數概似一定變大。」錯，差距也會變；只有 ELBO 升到比原來的 $\log p$ 還高時才確定。
- 「ELBO 裡的 KL 就是 $q$ 和後驗的 KL。」錯，ELBO 裡的是 $q$ 和先驗的 KL。

**Common misconceptions**:

- "The ELBO is an approximation of the log-likelihood." Wrong: it is a lower bound, and the gap is the KL.
- "A larger ELBO always means a larger log-likelihood." Wrong: the gap changes too; it is certain only when the ELBO rises above the old $\log p$.
- "The KL inside the ELBO is the KL between $q$ and the posterior." Wrong: the one inside the ELBO is between $q$ and the prior.

**圖**：本單元有兩張圖。

- `lec12-kp03-elbo-gap`：ELBO 永遠在 $\log p(x)$ 下面，差距就是 KL。
- `lec12-kp03-elbo-terms`：重建項與離先驗的 KL 互相拉扯，ELBO 的最高點落在真正後驗的平均。

**Figures** — this unit has two figures.

- `lec12-kp03-elbo-gap`: the ELBO always stays below $\log p(x)$, and the gap is the KL.
- `lec12-kp03-elbo-terms`: the reconstruction term and the KL from the prior pull against each other, and the ELBO peaks at the mean of the true posterior.

### 小練習（Practice）

1. 某筆資料的 $\log p(\mathbf{x})=-3$。一個 $q$ 的 ELBO 是 $-3.4$。這個 $q$ 和真正後驗的 KL 是多少？
2. 在第2節的表裡，$q$ 從 $\mathcal{N}(0.5,0.2)$ 換成真正的後驗，ELBO 增加了多少？KL 減少了多少？兩者為什麼一樣？
3. 第4節的寫法裡，如果把正規化項拿掉、只最大化重建項，在 KP02 的例子中 $q$ 的平均會跑到哪裡？這像 KP01 的哪一種模型？

1. An example has $\log p(\mathbf{x})=-3$. Some $q$ has ELBO $-3.4$. What is the KL between this $q$ and the true posterior?
2. In the table of Section 2, when $q$ changes from $\mathcal{N}(0.5,0.2)$ to the true posterior, how much does the ELBO increase, and how much does the KL decrease? Why are they the same?
3. In the form of Section 4, if the regularisation term is removed and only the reconstruction term is maximised, where does the mean of $q$ go in KP02's example? Which model of KP01 does this resemble?

**第 1 題解答** `[補充]`：$-3-(-3.4)=0.4$。

**Solution to question 1** `[Added]`: $-3-(-3.4)=0.4$.

**第 2 題解答** `[補充]`：ELBO 從 $-2.1355$ 升到 $-1.6065$，增加 $0.5290$；KL 從 $0.5290$ 降到 $0$，也減少 $0.5290$。因為兩者相加永遠等於固定的 $\log p(x)=-1.6065$。

**Solution to question 2** `[Added]`: the ELBO rises from $-2.1355$ to $-1.6065$, an increase of $0.5290$; the KL falls from $0.5290$ to $0$, also a decrease of $0.5290$. They match because their sum always equals the fixed $\log p(x)=-1.6065$.

**第 3 題解答** `[補充]`：重建項在 $m=x=1.2$ 最大，所以平均會跑到 $1.2$，只管把資料解釋好、不管先驗。如果 $q$ 的變異數也可以調，重建項還會把 $s^2$ 縮到 $0$，$q$ 變成落在 $1.2$ 的單一個點，也就是一個確定的編碼。這像 KP01 的一般自編碼器：只求重建，不規定編碼該落在哪裡。

**Solution to question 3** `[Added]`: the reconstruction term is largest at $m=x=1.2$, so the mean moves to $1.2$, caring only about explaining the data and ignoring the prior. If the variance of $q$ may change too, the reconstruction term also shrinks $s^2$ to $0$, and $q$ collapses to a single point at $1.2$, a fixed code. This resembles KP01's ordinary autoencoder: it only seeks reconstruction and does not say where codes should lie.

### 小測驗（Quiz，來源原題）

來源這一段有一題自我檢測，保留原題與全部選項；來源頁面沒有直接標出答案，下面的解析依來源內文判斷。

The source has one self-check question for this part, kept here with all its options; the source page does not mark the answer, so the explanation below is judged from the source text.

1. ELBO 與 log-likelihood 的關係是？
   - (A) ELBO $=$ log-likelihood
   - (B) ELBO $\le$ log-likelihood，且差距 $=$ $D_{\text{KL}}(q\Vert\mathbb{P}(z_i\mid\mathbf{x}_i,\theta))$
   - (C) ELBO $\ge$ log-likelihood
   - (D) ELBO 與 log-likelihood 無關

1. What is the relationship between the ELBO and the log-likelihood?
   - (A) ELBO $=$ log-likelihood
   - (B) ELBO $\le$ log-likelihood, and the gap $=$ $D_{\text{KL}}(q\Vert\mathbb{P}(z_i\mid\mathbf{x}_i,\theta))$
   - (C) ELBO $\ge$ log-likelihood
   - (D) The ELBO is unrelated to the log-likelihood

**解析** `[補充]`：選 (B)：第1節。(A) 只在 $q$ 等於真正後驗時成立；(C) 方向反了，因為 KL $\ge0$；(D) 錯，兩者正好差一個 KL。題中的 $\mathbb{P}$ 就是本講義的 $p$，下標 $i$ 指第 $i$ 筆資料。

**Explanation** `[Added]`: B; see Section 1. A holds only when $q$ is the true posterior; C has the direction reversed, because the KL is $\ge0$; D is wrong, as the two differ by exactly the KL. In the question, $\mathbb{P}$ is this note's $p$, and the subscript $i$ means the $i$-th data point.

### 留給下一單元的問題（The question for the next unit）

第3節說，ELBO 有兩個旋鈕：$q$ 和 $\theta$。一個自然的作法是輪流轉：先固定 $\theta$，把 $q$ 調到最好；再固定 $q$，把 $\theta$ 調到最好；一直重複。這就是統計學裡有名的 EM 演算法。它為什麼每一輪都不會變差？它和「把缺失的資料補起來」有什麼關係？下一個知識點（本週 KP04，選讀）介紹 EM。

Section 3 showed that the ELBO has two knobs: $q$ and $\theta$. A natural approach is to turn them in turns: fix $\theta$ and set $q$ as well as possible; then fix $q$ and set $\theta$ as well as possible; repeat. This is the famous EM algorithm of statistics. Why does each round never make things worse? And what does it have to do with "filling in missing data"? The next knowledge point (this week's KP04, optional reading) introduces EM.

### 來源更正與取捨（Corrections to the source）

本講義依 CCC-LNS IDL_115 第12章「變分模型」§2.1 與教授祕笈「ELBO：log-likelihood 的下界」，以及 §3 開頭 VAE 的 ELBO 目標（網頁講義，https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch12-variational-models/index.html ，讀取日期 2026-11-14）重新撰寫。來源沒有的內容，都在出現的地方標示 `[補充]`。

This note is rewritten from §2.1 of Chapter 12, "Variational models", of CCC-LNS IDL_115, with the professor's notes "ELBO: a lower bound on the log-likelihood" and the VAE's ELBO objective at the start of §3 (web lecture, https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch12-variational-models/index.html , accessed 2026-11-14). Content not in the source is marked `[Added]` where it appears.

- 來源把推導寫成「$D_{\text{KL}}(q\Vert\mathbb{P}(z\mid x))=\log\mathbb{P}(x)+D_{\text{KL}}(q\Vert\mathbb{P}(x,z))$」，第二項寫成對聯合分布的 KL。$\mathbb{P}(x,z)$ 當成 $z$ 的函數時面積不是 $1$，不是一個機率分布，所以這一項不是真正的 KL，可能是負的；本講義改寫成期望值 $\mathbb{E}_q[\log q-\log p(\mathbf{x},\mathbf{z})]$，結果相同。
- 來源說最大化 ELBO 與最小化 KL「兩者等價，因為只差一個常數」；這只在固定 $\theta$、只調 $q$ 時成立。調 $\theta$ 時對數概似也會變，本講義在第3節分開說明。
- 祕笈的比喻說「最大化 ELBO 等價於最大化 log p（因為差距是固定的 $D_{\text{KL}}$）」；差距會隨 $q$ 與 $\theta$ 改變，並不固定。本講義不採用這個說法。
- 來源用 $\mathbb{P}(\cdot)$ 與粗斜體 `\boldsymbol`；本講義改用 $p(\cdot)$ 與 `\mathbf`。小測驗保留來源原本的記號。
- 本講義補上：推導的中間步驟、三種 $q$ 的公式、數字表與抽樣估計、兩張圖、第3節對 $q$ 與 $\theta$ 分開的分析與 $\log p\ge$ ELBO 的保證、第4節的拆解步驟與數字例子、兩個 KL 的迷思、山與平台的比喻、三題小練習與解答。
- 取捨：來源的「月收入 80%」比喻本講義不寫。

- The source writes the derivation as "$D_{\text{KL}}(q\Vert\mathbb{P}(z\mid x))=\log\mathbb{P}(x)+D_{\text{KL}}(q\Vert\mathbb{P}(x,z))$", with the second term written as a KL against the joint distribution. As a function of $z$, $\mathbb{P}(x,z)$ does not have area $1$ and is not a probability distribution, so this term is not a true KL and can be negative; this note writes it as the expectation $\mathbb{E}_q[\log q-\log p(\mathbf{x},\mathbf{z})]$, with the same result.
- The source says maximising the ELBO and minimising the KL "are equivalent, since they differ only by a constant"; this holds only with $\theta$ fixed and only $q$ adjusted. When $\theta$ is adjusted the log-likelihood changes too, and Section 3 treats the two cases separately.
- The notes' analogy says "maximising the ELBO is equivalent to maximising log p (because the gap is a fixed $D_{\text{KL}}$)"; the gap changes with $q$ and $\theta$ and is not fixed, so this note does not use that statement.
- The source uses $\mathbb{P}(\cdot)$ and bold italic `\boldsymbol`; this note uses $p(\cdot)$ and `\mathbf`. The quiz keeps the source's original notation.
- This note adds: the middle step of the derivation, the formulas for three choices of $q$ with their table and sampling estimate, both figures, Section 3's separate analysis of $q$ and $\theta$ and the $\log p\ge$ ELBO guarantee, Section 4's splitting step and numerical example, the misconception about the two KLs, the mountain-and-platform analogy, and three practice questions with solutions.
- Choices: the source's "80% of average income" analogy is not used.

## 術語表（Glossary）

| 中文 | English |
|---|---|
| 對數概似 | log-likelihood |
| 對數證據 | log-evidence |
| 證據下界 | evidence lower bound (ELBO) |
| KL 散度（KP02） | Kullback–Leibler (KL) divergence |
| 聯合分布 | joint distribution |
| 期望值 | expectation |
| 下界 | lower bound |
| 重建項 | reconstruction term |
| 正規化項 | regularisation term |
| 自然對數 | natural logarithm |
| 抽樣估計（蒙地卡羅，KP06） | sampling estimate (Monte Carlo) |
| EM 演算法（KP04） | expectation–maximisation (EM) algorithm |
| 變分自編碼器（KP05） | variational autoencoder (VAE) |
