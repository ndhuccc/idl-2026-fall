# 第2週 KP10：萬有逼近定理：一層隱藏層夠多就能逼近任何連續函數，但沒說怎麼學（Universal approximation: one hidden layer can approximate any continuous function, but the theorem does not say how to learn it）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)
>
> 週次：第2週（2026-09-17）｜主題：神經網路基礎：感知器與多層感知器
>
> Week 2 (2026-09-17) | Topic: Neural network fundamentals: perceptron and multilayer perceptron

本講義是第2章的第十個知識點（KP10），也是本章的最後一個。承接 KP09 留下的問題：只用一層隱藏層、夠多神經元，到底能表示多少種函數？答案是萬有逼近定理（universal approximation theorem, UAT）。這一節用「很陡的 sigmoid 疊成階梯」親手建出一個逼近，量出誤差如何隨神經元數下降，再把定理的證明縮成四步，並說清楚它保證什麼、不保證什麼。先備知識是 KP05、KP06。內容依第2章主文重新講解，來源讀取日期為 2026-10-14；新增的釐清標為 `[補充]`。

This is the tenth and last knowledge point (KP10) of Chapter 2. It picks up the question left by KP09: with only one hidden layer and enough neurons, how many kinds of functions can be represented? The answer is the universal approximation theorem (UAT). This note builds an approximation by hand by stacking very steep sigmoids into a staircase, measures how the error falls with the number of neurons, condenses the proof into four steps, and states clearly what the theorem guarantees and what it does not. It assumes KP05 and KP06. It re-explains the main text of Chapter 2, accessed on 2026-10-14; clarifications that are new are marked `[Added]`.

[下載本單元投影片](../slides/lec02-kp10-universal-approximation.jbd.json)

## KP10 萬有逼近定理：一層隱藏層夠多就能逼近任何連續函數，但沒說怎麼學（Universal approximation: one hidden layer can approximate any continuous function, but the theorem does not say how to learn it）

**建議時間 / Time**：15 分鐘　**對應實作 / Related practice**：用階梯法逼近 $\sin2\pi x$，算 $N=10,20,40$ 時的最大誤差；回顧本章六個模型。

### 學習目標（Learning objectives）

讀完之後，你能寫出定理的敘述與它對活化函數的條件；能用很陡的 sigmoid 疊出階梯逼近，並算出誤差大約隨 $1/N$ 下降；能把四步證明用自己的話說出來；能說明定理保證「存在」而不保證「學得到」；能把本章的六個模型放回同一條時間線。

After reading, you can state the theorem and its condition on the activation function; build a staircase approximation from very steep sigmoids and compute that the error falls roughly like $1/N$; retell the four-step proof in your own words; explain that the theorem guarantees "existence" but not "learnability"; and place the six models of this chapter on one timeline.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\sigma$ | 活化函數（連續、有界、非常數） | the activation function (continuous, bounded, non-constant) |
| $f$ | 要逼近的連續函數，定義在 $[0,1]^n$ | the continuous function to approximate, defined on $[0,1]^n$ |
| $F(\mathbf{x})=\sum_{i=1}^{N}\alpha_i\sigma(\mathbf{w}_i^\top\mathbf{x}+\theta_i)$ | 單隱藏層、$N$ 顆神經元的網路 | a network with one hidden layer of $N$ neurons |
| $\alpha_i,\ \mathbf{w}_i,\ \theta_i$ | 輸出權重、輸入權重、偏置 | output weights, input weights and biases |
| $\varepsilon$ | 容許的誤差 | the allowed error |
| $\sup$ | 最大值（上確界） | the maximum (supremum) |

### 1. 定理說了什麼（What the theorem says）

**定理。** 設 $\sigma:\mathbb{R}\to\mathbb{R}$ 是連續、有界、而且不是常數的函數。則對任意連續函數 $f:[0,1]^n\to\mathbb{R}$ 與任意 $\varepsilon>0$，存在 $N$ 與參數 $\alpha_i,\mathbf{w}_i,\theta_i$，使

$$
F(\mathbf{x})=\sum_{i=1}^{N}\alpha_i\,\sigma(\mathbf{w}_i^\top\mathbf{x}+\theta_i)\quad\text{滿足}\quad \sup_{\mathbf{x}\in[0,1]^n}\big|f(\mathbf{x})-F(\mathbf{x})\big|<\varepsilon
$$

也就是說，單隱藏層、$N$ 顆神經元的網路，能在有限範圍（緊緻域）上把任何連續函數逼近到任意精度。它屬於「表示定理」：像 Fourier 級數用三角函數、小波用小波基底去逼近函數，神經網路的 $\sigma(\mathbf{w}^\top\mathbf{x}+\theta)$ 也是一組基底。

**The theorem.** Let $\sigma:\mathbb{R}\to\mathbb{R}$ be continuous, bounded and not constant. Then for every continuous function $f:[0,1]^n\to\mathbb{R}$ and every $\varepsilon>0$ there exist $N$ and parameters $\alpha_i,\mathbf{w}_i,\theta_i$ such that $F$ above satisfies the sup-bound. That is, a network with one hidden layer of $N$ neurons can approximate any continuous function to any precision on a bounded (compact) domain. It is a "representation theorem": just as a Fourier series approximates functions with trigonometric functions and wavelets with wavelet bases, the units $\sigma(\mathbf{w}^\top\mathbf{x}+\theta)$ of a neural network also form a family of basis functions.

### 2. 親手建一個逼近：階梯法（Build one by hand: the staircase method）

取一維的 $f(x)=\sin2\pi x$，$x\in[0,1]$。把 $[0,1]$ 切成 $N$ 格，格子的分界點是 $t_0=0<t_1<\dots<t_N=1$，每格取中點的函數值 $v_i=f\big(\tfrac{t_{i-1}+t_i}{2}\big)$。把一個很陡的 sigmoid $\sigma(k(x-t_i))$（$k$ 很大）看成一個「在 $t_i$ 處從 0 跳到 1 的階梯」，則

$$
F(x)=v_1+\sum_{i=1}^{N-1}(v_{i+1}-v_i)\,\sigma\big(k(x-t_i)\big)
$$

在每一格內等於該格的 $v_i$，整體就是一條階梯；它確實是 $\sum\alpha_i\sigma(w_ix+\theta_i)$ 的形式，只是 $w_i=k$、$\theta_i=-kt_i$、$\alpha_i=v_{i+1}-v_i$ 都是手算出來的，不是學出來的（再加上一個常數項 $v_1$，等於再多一個偏置）。

Take the one-dimensional $f(x)=\sin2\pi x$ on $[0,1]$. Cut $[0,1]$ into $N$ cells with boundaries $t_0=0<t_1<\dots<t_N=1$ and take the value at the midpoint of each cell, $v_i=f\big(\tfrac{t_{i-1}+t_i}{2}\big)$. Regard a very steep sigmoid $\sigma(k(x-t_i))$ (large $k$) as a "step that jumps from 0 to 1 at $t_i$"; then $F$ above equals $v_i$ inside each cell, so the whole is a staircase. It really has the form $\sum\alpha_i\sigma(w_ix+\theta_i)$, with $w_i=k$, $\theta_i=-kt_i$ and $\alpha_i=v_{i+1}-v_i$ computed by hand rather than learned (plus the constant $v_1$, which is one more bias).

**誤差〔程式計算，$k=2000$〕。** 最大誤差 $\sup|f-F|$：

| $N$ | 3 | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|
| 最大誤差 | $0.866$ | $0.588$ | $0.309$ | $0.156$ | $0.0785$ | $0.0393$ |

神經元數加倍，誤差大約減半。原因是：每格寬 $1/N$，取中點時格內最多偏離半格寬乘上函數的最大斜率 $2\pi$，即 $\tfrac{1}{2N}\cdot2\pi=\pi/N$，與表中 $0.309\approx\pi/10$ 相符。要讓誤差小於 $0.01$，約需 $N\approx320$ 顆。這說明「夠多」可能真的很多。

**The error (computed by program, $k=2000$).** The maximum error $\sup|f-F|$ is in the table above. Doubling the number of neurons roughly halves the error. The reason is that each cell has width $1/N$, so with the midpoint the function deviates inside a cell by at most half the width times the maximum slope $2\pi$, i.e. $\tfrac{1}{2N}\cdot2\pi=\pi/N$, matching $0.309\approx\pi/10$ in the table. To push the error below $0.01$ takes about $N\approx320$ neurons. "Enough" can indeed be a lot.

![用階梯法逼近 $\sin2\pi x$，以及最大誤差隨神經元數 $N$ 下降](../assets/figures/lec02-kp10-uat-staircase.png)

**圖說 `lec02-kp10-uat-staircase`**：先看左圖，黑線是目標，藍色（$N=5$）是粗的階梯，橘色（$N=20$）貼近很多；再看右圖，雙對數座標下誤差是一條斜率約 $-1$ 的直線，標出 $N=10,20,40$ 的 $0.309,0.156,0.078$。 / **Caption `lec02-kp10-uat-staircase`:** Look first at the left panel: the black line is the target, the blue ($N=5$) is a coarse staircase and the orange ($N=20$) follows it much more closely; then the right panel: on log-log axes the error is a straight line of slope about $-1$, with $0.309,0.156,0.078$ marked at $N=10,20,40$.

### 3. 四步證明的骨架（The skeleton of the four-step proof）

階梯法只對一維有用。一般的定理用下面的反證法，這裡只看骨架（細節是泛函分析）：

1. **假設空間**：令 $\mathcal{H}_\sigma=\mathrm{span}\{\sigma(\mathbf{w}^\top\mathbf{x}+\theta)\}$。要證明它在連續函數空間 $C([0,1]^n)$（用最大值距離）中**稠密**，意思是任何連續函數都能被它任意逼近。
2. **反證**：假設不稠密。由 Hahn-Banach 定理，存在一個非零的連續線性泛函 $L$，它對 $\mathcal{H}_\sigma$ 中每一個函數都輸出 0。
3. **Riesz 表示**：這個 $L$ 可寫成對某個帶符號測度 $\mu$ 的積分 $L(f)=\int f\,d\mu$，所以 $\int\sigma(\mathbf{w}^\top\mathbf{x}+\theta)\,d\mu=0$ 對所有 $\mathbf{w},\theta$ 成立。
4. **矛盾**：$\sigma$ 的性質（連續、有界、非常數）使這些函數足以把不同的測度分開，迫使 $\mu=0$，與 $L\ne0$ 矛盾。所以 $\mathcal{H}_\sigma$ 稠密。

The staircase method works only in one dimension. The general theorem uses the following proof by contradiction; only the skeleton is shown here (the details are functional analysis). (1) **Hypothesis space**: let $\mathcal{H}_\sigma=\mathrm{span}\{\sigma(\mathbf{w}^\top\mathbf{x}+\theta)\}$; we must show it is **dense** in the space $C([0,1]^n)$ of continuous functions (with the max-distance), meaning every continuous function can be approximated arbitrarily well. (2) **Contradiction**: suppose it is not dense; by the Hahn-Banach theorem there is a nonzero continuous linear functional $L$ that outputs 0 on every function of $\mathcal{H}_\sigma$. (3) **Riesz representation**: this $L$ can be written as an integral $L(f)=\int f\,d\mu$ against a signed measure $\mu$, so $\int\sigma(\mathbf{w}^\top\mathbf{x}+\theta)\,d\mu=0$ for all $\mathbf{w},\theta$. (4) **Contradiction**: the properties of $\sigma$ (continuous, bounded, non-constant) make these functions rich enough to tell different measures apart, forcing $\mu=0$, which contradicts $L\ne0$. Hence $\mathcal{H}_\sigma$ is dense.

### 4. 定理保證什麼、不保證什麼（What it guarantees and what it does not）

**保證**：只要 $N$ 夠大，單隱藏層網路**存在**一組參數，能把任何連續函數逼近到任意精度。

**不保證**：(1) 不告訴你要多大的 $N$（階梯法的例子需要幾百顆，高維時只會更多）；(2) 不告訴你怎麼找到參數，上面的 $\alpha_i,\mathbf{w}_i,\theta_i$ 是我們為了一維的例子手算出來的，一般情形沒有這種公式，要靠學習演算法去找，這是第3章的主題；(3) 不保證學到的參數在沒看過的資料上也好（KP04 的泛化問題）；(4) 不保證訓練會收斂到全域最佳。

**Guarantees**: if $N$ is large enough, a one-hidden-layer network **has** a set of parameters that approximates any continuous function to any precision.

**Does not guarantee**: (1) it does not say how large $N$ must be (the staircase example needs hundreds, and in higher dimensions many more); (2) it does not say how to find the parameters: the $\alpha_i,\mathbf{w}_i,\theta_i$ above were computed by hand for the one-dimensional example, there is no such formula in general, and a learning algorithm must find them, which is the topic of Chapter 3; (3) it does not guarantee the learned parameters are also good on unseen data (the generalization problem of KP04); (4) it does not guarantee that training converges to the global optimum.

### 本單元重點整理（KP10 summary）

**一句話重點**：萬有逼近定理說單隱藏層、夠多神經元的網路能任意逼近連續函數，但只保證「有解」，不保證「找得到」，也不保證「泛化好」。

**One-sentence summary**: The universal approximation theorem says a one-hidden-layer network with enough neurons can approximate any continuous function arbitrarily well, but it guarantees only that a solution exists, not that it can be found or generalizes well.

**白話說明**：像說「只要積木夠多，任何形狀都蓋得出來」。這句話是真的，但沒告訴你要幾塊、也沒給你說明書。階梯法就是我們為一個簡單形狀自己寫的說明書：一格一格往上疊。

**Plain explanation**: Like saying "with enough building blocks any shape can be built". It is true, but it does not tell you how many blocks, and it gives no instructions. The staircase method is the instruction sheet we wrote ourselves for one simple shape: stack up cell by cell.

**正式定義**：$\sigma$ 連續、有界、非常數時，$\forall f\in C([0,1]^n)$、$\forall\varepsilon>0$，$\exists N,\alpha_i,\mathbf{w}_i,\theta_i$：$\sup|f-\sum\alpha_i\sigma(\mathbf{w}_i^\top\mathbf{x}+\theta_i)|<\varepsilon$。

**Formal definition**: For $\sigma$ continuous, bounded and non-constant, $\forall f\in C([0,1]^n)$, $\forall\varepsilon>0$, $\exists N,\alpha_i,\mathbf{w}_i,\theta_i$ such that $\sup|f-\sum\alpha_i\sigma(\mathbf{w}_i^\top\mathbf{x}+\theta_i)|<\varepsilon$.

**例子**：$\sin2\pi x$ 的階梯逼近，$N=10,20,40$ 的最大誤差是 $0.309,0.156,0.0785$，約為 $\pi/N$；要小於 $0.01$ 約需 $320$ 顆。

**Example**: For the staircase approximation of $\sin2\pi x$, the maximum errors at $N=10,20,40$ are $0.309,0.156,0.0785$, about $\pi/N$; going below $0.01$ takes about $320$ neurons.

**常見迷思**：

- 「定理保證神經網路一定學得會任何函數」——不是。它只說參數存在；怎麼找到、需要多少神經元、泛化好不好，都不在定理內。
- 「活化函數必須是 sigmoid 或有界的才行」——定理的這個版本以有界的 $\sigma$ 敘述；之後的研究證明，只要不是多項式，連 ReLU 這類無界的活化函數也成立（`[補充]`）。
- 「需要很多層才有表達力」——就逼近能力而言，一層隱藏層已足夠；多層的好處是通常用少很多的神經元就能做到同樣的事（第3章以後與第1章 KP09 的參數量討論會再回到這點）。

**Common misconceptions**:

- "The theorem guarantees a neural network can always learn any function." No. It says only that the parameters exist; how to find them, how many neurons are needed and whether it generalizes are all outside the theorem.
- "The activation function must be a sigmoid or bounded." This version of the theorem is stated for a bounded $\sigma$; later work showed that it holds for any non-polynomial activation, including unbounded ones such as ReLU (`[Added]`).
- "Many layers are needed for expressive power." For approximation power, one hidden layer already suffices; the benefit of several layers is that they can often do the same with far fewer neurons (Chapter 3 onward and the parameter-count discussion of Chapter 1 KP09 return to this).

**圖**：本單元一張圖，`lec02-kp10-uat-staircase`（階梯逼近與誤差，功能 1、2），以 `assets/figures/src/lec02-kp0810-figures.py` 繪製。

**Figures**: One figure in this unit, `lec02-kp10-uat-staircase` (the staircase approximation and its error; functions 1, 2), drawn by `assets/figures/src/lec02-kp0810-figures.py`.

**想深入**（選讀）：本章的六個模型按年份是 M-P（1943）、Perceptron（1958）、ADALINE（1960）、SOM（1982）、MLP（1986）、RBF（1988）。它們有共同的零件：加權和、非線性活化、用誤差調權重；UAT 為這個零件組出的網路提供了「能表示」的理論後盾。

**Going deeper** (optional): By year, the six models of this chapter are M-P (1943), Perceptron (1958), ADALINE (1960), SOM (1982), MLP (1986) and RBF (1988). They share the same parts: a weighted sum, a nonlinear activation, and adjusting weights with an error; the UAT gives the networks built from these parts a theoretical backing for "can represent".

**小練習**：

1. 用階梯法逼近 $f(x)=x$ 於 $[0,1]$，$N=4$，取各格中點的值。寫出 $v_1,\dots,v_4$ 與 $\alpha_i=v_{i+1}-v_i$。（提示：格子是 $[0,0.25],[0.25,0.5],\dots$。）
2. 估計：要把 $\sin2\pi x$ 的階梯逼近誤差降到 $0.05$，約需多少神經元？（提示：用 $\pi/N$。）
3. UAT 為什麼不能直接拿來當「訓練會成功」的保證？（提示：想想定理裡有沒有「怎麼找參數」。）

**Practice**:

1. Approximate $f(x)=x$ on $[0,1]$ by the staircase method with $N=4$, using the value at each cell midpoint. Write $v_1,\dots,v_4$ and $\alpha_i=v_{i+1}-v_i$. (Hint: the cells are $[0,0.25],[0.25,0.5],\dots$.)
2. Estimate how many neurons the staircase approximation of $\sin2\pi x$ needs for an error of $0.05$. (Hint: use $\pi/N$.)
3. Why can the UAT not be used directly as a guarantee that "training will succeed"? (Hint: does the theorem say how to find the parameters?)

**解題示範**（第 1 題）：

1. 四格中點是 $0.125,\ 0.375,\ 0.625,\ 0.875$，所以 $v=(0.125,\,0.375,\,0.625,\,0.875)$。
2. 相鄰差是 $\alpha_i=v_{i+1}-v_i=0.25,\,0.25,\,0.25$，$t_1,t_2,t_3=0.25,\,0.5,\,0.75$。
3. 於是 $F(x)=0.125+0.25\,[\sigma(k(x-0.25))+\sigma(k(x-0.5))+\sigma(k(x-0.75))]$。
4. 檢查：在 $x=0.6$ 時前兩個 sigmoid 都已跳到 1、第三個還是 0，$F=0.125+0.5=0.625$，正是第三格的 $v_3$；最大誤差是每格的半寬 $0.125$。

**Worked solution** (exercise 1):

1. The four midpoints are $0.125,\ 0.375,\ 0.625,\ 0.875$, so $v=(0.125,\,0.375,\,0.625,\,0.875)$.
2. The consecutive differences are $\alpha_i=v_{i+1}-v_i=0.25,\,0.25,\,0.25$, with $t_1,t_2,t_3=0.25,\,0.5,\,0.75$.
3. Hence $F(x)=0.125+0.25\,[\sigma(k(x-0.25))+\sigma(k(x-0.5))+\sigma(k(x-0.75))]$.
4. Check: at $x=0.6$ the first two sigmoids have already jumped to 1 and the third is still 0, so $F=0.125+0.5=0.625$, exactly $v_3$ of the third cell; the maximum error is half a cell width, $0.125$.

**自我檢核**（來源四題，不附答案）：

| 題 / Question | A | B | C | D | 提示 / Hint |
|---|---|---|---|---|---|
| 1. UAT 對活化函數 $\sigma$ 的核心假設？ / The core assumption of the UAT on $\sigma$? | 必須是 ReLU | 連續、有界、且非恆為常數 | 必須可微 | 必須有界且可導 | 看第 1 節的定理。 / The theorem in section 1. |
| 2. 反證步驟中，假設空間不稠密時會得到？ / In the contradiction step, if the space is not dense, we get? | 存在非零線性泛函 $L$ 湮滅整個 $\mathcal{H}_\sigma$ | 存在零泛函 | $\sigma$ 必為常數 | 找不到 $\varepsilon$ | 看第 3 節第 2 步。 / Step 2 of section 3. |
| 3. UAT 保證什麼、不提供什麼？ / What does the UAT guarantee and what not? | 保證能逼近任意連續函數，但不提供找參數的構造性方法 | 保證收斂到全域最佳 | 保證不 overfit | 保證計算複雜度很低 | 看第 4 節。 / Section 4. |
| 4. 把神經網路對比 Fourier 級數，UAT 指出神經網路是什麼？ / Comparing a network with a Fourier series, what does the UAT say it is? | 與三角函數無關 | 構成逼近函數的一組基底，類比 Fourier 的三角函數 | 只能逼近線性函數 | 只能逼近常數函數 | 看第 1 節最後一句。 / The last sentence of section 1. |

**留給下一個知識點的問題**：第2章看完了。我們知道網路「能表示」什麼，也看過手工湊出來的 XOR 權重與階梯法的參數。真正的問題是：這些權重該怎麼自己學出來，特別是多層網路中間那些看不到答案的隱藏層？這就是第3章的反向傳播與梯度下降。

**Question for the next chapter**: Chapter 2 is done. We know what a network "can represent", and we have seen hand-made XOR weights and the staircase parameters. The real question is how to learn these weights by themselves, especially in the hidden layers in the middle of a multilayer network that never see the answer. That is Chapter 3: backpropagation and gradient descent.

### 來源更正與取捨（Corrections to the source）

- 來源把 UAT 的條件寫成「連續、有界、且非恆為常數」，這是定理常見版本（Cybenko 1989、Hornik 1991）的條件；它不涵蓋 ReLU 這類無界的活化函數。之後 Leshno 等人（1993）證明，只要活化函數連續且不是多項式，定理就成立，所以 ReLU 也適用（`[補充]`，結論引自文獻，本講義沒有另行驗證）。
- 來源的第 4 步寫「$\sigma$ 連續、有界、非常數，足以分離測度」。這一步是整個證明最難的部分（需要一個「判別性」的性質，Cybenko 針對 sigmoid 形的函數證明），本講義只保留骨架，不重現細節。
- 來源的章末圖把 MLP 標為 1986 年；這個年份對應反向傳播的普及，而不是「疊多層」這個想法的出現（第1章 KP08 有更完整的討論）。
- 階梯法與誤差表（$N=3,\dots,80$）是本講義新增的例子，數字由程式驗算（`[補充]`）。

- The source states the condition of the UAT as "continuous, bounded and not constant", the condition of the commonly cited versions (Cybenko 1989, Hornik 1991); it does not cover unbounded activations such as ReLU. Leshno et al. (1993) later proved that the theorem holds whenever the activation is continuous and not a polynomial, so ReLU also applies (`[Added]`; the conclusion is cited from the literature and has not been separately verified in this note).
- In step 4 the source says "$\sigma$ continuous, bounded, non-constant is enough to separate measures". This step is the hardest part of the whole proof (it needs a "discriminatory" property, which Cybenko proved for sigmoidal functions); this note keeps only the skeleton and does not reproduce the details.
- The closing figure of the source labels the MLP as 1986; this year corresponds to the popularization of backpropagation, not to the appearance of the idea of "stacking layers" (Chapter 1 KP08 discusses it more fully).
- The staircase method and the error table ($N=3,\dots,80$) are new examples of this note, with numbers checked by program (`[Added]`).

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 萬有逼近定理 | universal approximation theorem (UAT) |
| 表示定理 | representation theorem |
| 稠密 | dense |
| 緊緻域 | compact domain |
| 上確界（最大誤差） | supremum (maximum error) |
| 階梯逼近 | staircase approximation |
| 基底 | basis |
| 泛函 | functional |
