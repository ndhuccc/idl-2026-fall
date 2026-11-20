# 第4週 KP05：過擬合、欠擬合與泛化（Overfitting, underfitting and generalization）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第4週「正則化與過擬合」的第五個知識點（KP05），承接 KP04 留下的問題：模型越緊貼訓練點，訓練誤差就比新資料上的誤差低得越多，那麼複雜度太低與太高的模型，實際上各長什麼樣子？先備知識是 KP02（偏差與變異）、KP03（雜訊地板 $\sigma^2$）與 KP04（測試誤差的拆解，含訓練誤差為何偏低）；第一次出現的符號都會定義，所有數字都附有程式計算。內容依 CCC-LNS 第5章 KP8 的主文與教授祕笈消化後重新撰寫，讀取日期 2026-10-20；來源沒有的內容標示 `[補充]`。

This is the fifth knowledge point (KP05) of Week 4, "Regularization and overfitting". It picks up the question KP04 left open: the more tightly a model hugs the training points, the further its training error falls below its error on new data, so what do models of too low and too high complexity actually look like? It assumes KP02 (bias and variance), KP03 (the noise floor $\sigma^2$) and KP04 (decomposing the test error, including why the training error is too low); every symbol is defined when it first appears and every number comes from a program. The text is rewritten from the main text and the professor's guide of KP8 in Chapter 5 of CCC-LNS (accessed on 2026-10-20); content that the source does not contain is marked `[Added]`.

[下載本單元投影片](../slides/lec04-kp05-overfitting-underfitting-generalization.jbd.json)

## KP05 過擬合、欠擬合與泛化（Overfitting, underfitting and generalization）

### 學習目標（Learning objectives）

讀完之後，你能說出泛化的意思，並用「訓練誤差與測試誤差各是高或低」分辨欠擬合、良好擬合與過擬合；能把測試誤差寫成 $\text{bias}^2+\text{var}+\sigma^2$，指出欠擬合是偏差大、過擬合是變異大；能看出訓練誤差低於雜訊地板 $\sigma^2$ 是記住雜訊的警訊；也能說明為什麼要用驗證或測試誤差而不是訓練誤差來選模型。

After reading, you can say what generalization means and tell underfitting, good fit and overfitting apart by whether the training and the test error are high or low; write the test error as $\text{bias}^2+\text{var}+\sigma^2$ and point out that underfitting is large bias and overfitting is large variance; recognize a training error below the noise floor $\sigma^2$ as a warning that noise has been memorized; and explain why the model is chosen by the validation or test error and not by the training error.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $f(x)$ | 真實函數，這一單元取 $1+2x-1.5x^2$ | the true function, here $1+2x-1.5x^2$ |
| $\sigma^2$ | 雜訊的變異數，這一單元取 $0.3^2=0.09$ | noise variance, here $0.3^2=0.09$ |
| $n$ | 訓練點的個數，這一單元 $n=12$ | number of training points, here $n=12$ |
| $d$ | 多項式的次數，也就是複雜度的刻度 | degree of the polynomial, the scale of complexity |
| $p$ | 參數個數，多項式 $p=d+1$ | number of parameters, $p=d+1$ for a polynomial |
| $\hat f_d$ | 在訓練點上用最小平方法配出的 $d$ 次多項式 | the degree-$d$ polynomial fitted to the training points by least squares |
| $\mathrm{train}_d$ | 訓練 MSE：訓練點上的平均平方誤差 | training MSE: mean squared error on the training points |
| $\mathrm{test}_d$ | 測試 MSE：對新資料的平均平方誤差 | test MSE: mean squared error on fresh data |

### 1. 三種狀態與泛化（Three states and generalization）

KP04 的四點例子已經看到：模型越緊貼訓練點（越複雜），訓練誤差就比新資料上的誤差低得越多。現在要看它的後果。訓練的目的不是把手上的點背起來，而是對**沒見過的資料**也預測得準，這個能力叫**泛化**（generalization）。依複雜度，模型會落在三種狀態之一：**欠擬合**是模型太簡單，連訓練資料都解釋不好；**過擬合**是模型太複雜，把訓練資料連同雜訊都記住，對新資料反而差；**良好擬合**介於兩者之間。

The four-point example of KP04 showed that the more tightly a model hugs the training points (the more complex it is), the further its training error falls below its error on new data. Now we look at the consequences. The goal of training is not to memorize the points at hand but to predict well on **data it has not seen**; this ability is called **generalization**. By complexity a model falls into one of three states: **underfitting** is a model too simple to explain even the training data; **overfitting** is a model so complex that it memorizes the training data together with its noise and does worse on new data; **good fit** lies between the two.

來源給的是一張表，假設真實函數是二次曲線，列出 1 次、3 次、10 次多項式的訓練與測試誤差。那張表是示意用的數字，無法重現，而且它說真實函數是二次、又把 3 次列為最接近，前後不一致（見「來源更正與取捨」）。所以我們自己做一次實驗，每個數字都可以重算：

The source gives a table that assumes a quadratic true function and lists the training and test errors of degrees 1, 3 and 10. Those numbers are illustrative, cannot be reproduced, and are inconsistent: the truth is said to be quadratic yet degree 3 is called the closest (see "Corrections to the source"). So we run our own experiment, in which every number can be recomputed:

- 真實函數 $f(x)=1+2x-1.5x^2$，是二次；雜訊標準差 $0.3$，所以雜訊地板 $\sigma^2=0.09$（KP03）。
- 訓練資料：$n=12$ 個點，$x$ 在 $-1$ 到 $1$ 之間等距，$y=f(x)+\varepsilon$，雜訊用固定的亂數種子 11 抽出。
- 模型：次數 $d=1,2,\dots,10$ 的多項式，用最小平方法配出。
- 測試 MSE：對新的帶雜訊資料的期望平方誤差。依 KP04 的恆等式，它等於 $\operatorname{mean}(\hat f-f)^2+\sigma^2$（對 $x\in[-1,1]$ 平均）；我們也抽 200000 個新的帶雜訊點直接平均來交叉驗算，兩者最多差 0.001。

- The true function is $f(x)=1+2x-1.5x^2$, a quadratic; the noise standard deviation is $0.3$, so the noise floor is $\sigma^2=0.09$ (KP03).
- Training data: $n=12$ points with $x$ equally spaced from $-1$ to $1$ and $y=f(x)+\varepsilon$, the noise drawn with the fixed random seed 11.
- Models: polynomials of degree $d=1,2,\dots,10$ fitted by least squares.
- Test MSE: the expected squared error on fresh noisy data. By the identity of KP04 it equals $\operatorname{mean}(\hat f-f)^2+\sigma^2$ (averaged over $x\in[-1,1]$); we also draw 200000 fresh noisy points and average directly as a cross-check, and the two differ by at most 0.001.

![同一組 12 個訓練點（實心）與 8 個沒參與訓練的測試點（空心），配出 1 次、2 次、9 次多項式（虛線是真實函數）：1 次是直線，彎不起來（欠擬合）；2 次貼近真實曲線（良好擬合）；9 次穿過幾乎每一個訓練點，卻在點與點之間劇烈擺動（過擬合）。每格標題是訓練 MSE 與測試 MSE。（The same 12 training points (filled) and 8 test points that took no part in training (hollow), fitted with polynomials of degree 1, 2 and 9 (the dashed line is the true function): degree 1 is a straight line that cannot bend (underfit); degree 2 follows the true curve (good fit); degree 9 passes through almost every training point and swings wildly between them (overfit). Each panel title gives the training and the test MSE.）](../assets/figures/lec04-kp05-fits.png)

**圖說 `lec04-kp05-fits`**：先看左格：直線從頭到尾都在曲線的同一側偏離，訓練 MSE 與測試 MSE 都高（$0.288$ 與 $0.294$），兩者幾乎相等，這是欠擬合的特徵。再看右格：9 次曲線穿過訓練點，訓練 MSE 只有 $0.009$，比雜訊地板 $0.09$ 小一個數量級，但在兩端與點與點之間擺盪，測試 MSE 升到 $0.289$。中間格的 2 次曲線和虛線幾乎重疊，訓練 $0.072$、測試 $0.096$。〔功能：1、2〕

**Caption `lec04-kp05-fits`**: Start with the left panel: the straight line stays on the same side of the curve over long stretches, and both the training and the test MSE are high ($0.288$ and $0.294$) and nearly equal, the signature of underfitting. Then the right panel: the degree-9 curve passes through the training points, with a training MSE of only $0.009$, an order of magnitude below the noise floor $0.09$, but it swings at both ends and between the points, and the test MSE rises to $0.289$. The degree-2 curve in the middle almost coincides with the dashed line, with training $0.072$ and test $0.096$. [Function: 1, 2]

### 2. 訓練誤差一路下降，測試誤差先降後升（Training error keeps falling; test error falls then rises）

上一節看了三個次數；把 $d=1$ 到 $10$ 全部配一遍，把兩種誤差並排（單位是 MSE，程式計算）：

The previous section looked at three degrees; fit all of $d=1$ to $10$ and put the two errors side by side (MSE, computed by program):

| 次數 $d$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 訓練 MSE | 0.2878 | 0.0720 | 0.0716 | 0.0675 | 0.0610 | 0.0460 | 0.0287 | 0.0221 | 0.0091 | 0.0069 |
| 測試 MSE | 0.2936 | 0.0963 | 0.0971 | 0.1041 | 0.1113 | 0.1315 | 0.1543 | 0.1827 | 0.2891 | 0.4398 |

| Degree $d$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Training MSE | 0.2878 | 0.0720 | 0.0716 | 0.0675 | 0.0610 | 0.0460 | 0.0287 | 0.0221 | 0.0091 | 0.0069 |
| Test MSE | 0.2936 | 0.0963 | 0.0971 | 0.1041 | 0.1113 | 0.1315 | 0.1543 | 0.1827 | 0.2891 | 0.4398 |

從表和下圖讀出四件事。第一，**訓練 MSE 隨次數單調下降**：高次多項式包含低次多項式（令高次的係數為 0 就得到低次），所以最小平方法在更大的模型族裡能找到的訓練誤差不會更大 `[補充]`。第二，**測試 MSE 在 $d=2$ 最低**（$0.0963$），之後一路上升；$d=9$ 時 $0.289$，已經和直線（$d=1$，$0.294$）一樣糟，$d=10$ 時 $0.440$，是雜訊地板的約 $4.9$ 倍。第三，**測試 MSE 永遠高於雜訊地板 $0.09$**：這是 KP03 的結論，再好的模型也越不過 $\sigma^2$；最好的 $0.0963$ 只比地板高 $0.0063$。第四，**訓練 MSE 可以低於地板**：$d=9$ 的 $0.0091$ 只有 $\sigma^2$ 的約十分之一。誤差比雜訊本身還小，代表模型不只學到訊號，連雜訊都背下來了。這是過擬合最直接的警訊。

Four things can be read from the table and the figure below. First, **the training MSE falls monotonically with the degree**: a high-degree polynomial contains the lower-degree ones (set the extra coefficients to 0), so least squares over the larger family cannot find a larger training error `[Added]`. Second, **the test MSE is lowest at $d=2$** ($0.0963$) and rises from then on; at $d=9$ it is $0.289$, already as bad as the straight line ($d=1$, $0.294$), and at $d=10$ it is $0.440$, about $4.9$ times the noise floor. Third, **the test MSE always stays above the noise floor $0.09$**: this is the conclusion of KP03, no model gets below $\sigma^2$; the best value $0.0963$ is only $0.0063$ above the floor. Fourth, **the training MSE can fall below the floor**: $0.0091$ at $d=9$ is only about a tenth of $\sigma^2$. An error smaller than the noise itself means the model has learned not just the signal but also memorized the noise, the most direct warning sign of overfitting.

![訓練 MSE（橘）與測試 MSE（藍）對多項式次數作圖（縱軸為對數刻度），綠色虛線是雜訊地板 $\sigma^2=0.09$：橘線一路下降並穿過地板；藍線在 $d=2$ 最低，之後上升，$d=9$ 之後急升。（Training MSE (orange) and test MSE (blue) against the polynomial degree, with a logarithmic vertical axis; the green dashed line is the noise floor $\sigma^2=0.09$: the orange line keeps falling and crosses the floor; the blue line is lowest at $d=2$ and then rises, steeply after $d=9$.）](../assets/figures/lec04-kp05-train-test.png)

**圖說 `lec04-kp05-train-test`**：先找綠色虛線（$0.09$），再看藍線的最低點（$d=2$，$0.0963$，剛好貼著虛線上方）。從 $d=2$ 往右，橘線繼續下降而藍線上升，兩線之間的縫隙越開越大，這個縫隙就是「訓練誤差不能代表測試誤差」的程度。橘線從 $d=2$ 起就已經在虛線下方。縱軸是對數刻度，所以右端藍線看起來緩，實際上 $d=10$ 比 $d=2$ 高約 $4.6$ 倍。〔功能：1、2〕

**Caption `lec04-kp05-train-test`**: Find the green dashed line ($0.09$) first, then the lowest point of the blue line ($d=2$, $0.0963$, just above the dashed line). To the right of $d=2$ the orange line keeps falling while the blue line rises, and the gap between them widens; that gap is how far the training error fails to represent the test error. The orange line is already below the dashed line from $d=2$ on. The vertical axis is logarithmic, so the right end of the blue line looks gentler than it is: $d=10$ is about $4.6$ times higher than $d=2$. [Function: 1, 2]

這張表只是一組訓練資料的結果，所以訓練誤差會有一點隨機的起伏。換一組雜訊，這些數字會跟著小幅變動，但趨勢不變。

The table is the result of one training set, so the training errors carry a little random scatter. With a different draw of noise these numbers shift a little, but the trend stays.

### 3. 用偏差與變異讀這張圖（Reading it with bias and variance）

上一節看到「次數太低與太高都不好」，現在問為什麼。KP02 說 $\mathrm{MSE}=\mathrm{Var}+\mathrm{Bias}^2$，KP03 與 KP04 說新資料的誤差還要再加上無法消除的 $\sigma^2$。在固定的 $x$ 上合起來：

The last section found that both too low and too high a degree are bad; now we ask why. KP02 said $\mathrm{MSE}=\mathrm{Var}+\mathrm{Bias}^2$, and KP03 and KP04 said that the error on new data also carries the irreducible $\sigma^2$. At a fixed $x$ the three combine:

$$
\mathbb E\big((\hat f(x)-y)^2\big)=\underbrace{\big(\mathbb E\hat f(x)-f(x)\big)^2}_{\text{bias}^2}+\underbrace{\mathrm{Var}\big(\hat f(x)\big)}_{\text{var}}+\sigma^2
$$

期望是對「重抽訓練資料」取的：想像用同樣的 12 個 $x$、每次抽新的雜訊，重複配 2000 次，就得到 2000 條曲線；它們的平均與真實函數的差給偏差，它們彼此的散布給變異。再對 $x\in[-1,1]$ 平均，得到下表（程式計算）：

The expectation is taken over redrawing the training data: imagine using the same 12 values of $x$ and drawing fresh noise each time, repeating the fit 2000 times to obtain 2000 curves; the gap between their mean and the true function gives the bias and their scatter gives the variance. Averaging also over $x\in[-1,1]$ gives the table below (computed by program):

| 次數 $d$ | bias² | var | $\sigma^2$ | 三項之和（期望測試 MSE） | 狀態 |
|---|---|---|---|---|---|
| 1 | 0.2084 | 0.0134 | 0.0900 | 0.3118 | 欠擬合：偏差主導 |
| 2 | 0.0000 | 0.0192 | 0.0900 | 0.1092 | 良好擬合 |
| 3 | 0.0000 | 0.0250 | 0.0900 | 0.1150 | 多出的一個參數換來較大變異 |
| 9 | 0.0000 | 0.1166 | 0.0900 | 0.2066 | 過擬合：變異主導 |

| Degree $d$ | bias² | var | $\sigma^2$ | Sum (expected test MSE) | State |
|---|---|---|---|---|---|
| 1 | 0.2084 | 0.0134 | 0.0900 | 0.3118 | underfit: bias dominates |
| 2 | 0.0000 | 0.0192 | 0.0900 | 0.1092 | good fit |
| 3 | 0.0000 | 0.0250 | 0.0900 | 0.1150 | one more parameter buys more variance |
| 9 | 0.0000 | 0.1166 | 0.0900 | 0.2066 | overfit: variance dominates |

怎麼讀這張表？直線（$d=1$）的 bias² 是 $0.208$，占總和的三分之二，變異只有 $0.013$：無論換哪一組訓練資料，它都在同一個地方錯，這是**高偏差、低變異**，也就是欠擬合。9 次多項式的偏差是 0（它含有真實函數，平均下來完全對），但變異是 $0.117$，約是 $d=2$ 的 $6$ 倍：換一組雜訊，曲線就換一個樣子，這是**低偏差、高變異**，也就是過擬合。$d=2$ 與 $d=3$ 的偏差都是 0（兩者都含有二次的真實函數，表中的 $0.0000$ 是模擬誤差以內的 0），差別只在變異，$d=3$ 多一個參數，變異從 $0.019$ 增加到 $0.025$：包含正確答案之後，再多的彈性只會增加變異。

How to read the table? The straight line ($d=1$) has bias² of $0.208$, two thirds of the sum, and a variance of only $0.013$: whichever training set we use, it is wrong in the same place, **high bias and low variance**, which is underfitting. The degree-9 polynomial has zero bias (it contains the true function, so on average it is exactly right) but a variance of $0.117$, about $6$ times that of $d=2$: change the noise and the curve changes shape, **low bias and high variance**, which is overfitting. Degrees 2 and 3 both have zero bias (both contain the quadratic truth; the $0.0000$ in the table is zero up to simulation error) and differ only in variance: $d=3$ has one more parameter and the variance rises from $0.019$ to $0.025$. Once the right answer is contained, extra flexibility only adds variance.

注意兩處對不上的地方不是錯誤。第一，上一節的測試 MSE（例如 $d=9$ 是 $0.289$）是**手上這一組**訓練資料的結果，本節的 $0.207$ 是對**所有可能的訓練資料**取平均，所以不同；一組資料抽到的雜訊偏大或偏小，都會讓單次結果偏離平均。第二，與 KP02 的整份拆解一樣，三項只是把同一個總誤差分開說明，沒有哪一項可以單獨最小化而不影響別項：次數升高時偏差降而變異升，這就是偏差與變異的取捨。

Two apparent mismatches are not errors. First, the test MSE of the previous section (for example $0.289$ at $d=9$) is the result for **this one** training set, while $0.207$ here is the average over **all possible** training sets; a draw with larger or smaller noise moves a single result away from the average. Second, as in KP02, the three terms only split one total error for explanation, and none can be minimized alone without affecting the others: raising the degree lowers the bias and raises the variance, which is the bias-variance trade-off.

### 4. 該怎麼選模型（How to choose the model）

前三節的結論是：訓練誤差只會隨次數下降，所以**用訓練誤差選模型，永遠會選到最複雜的那個**（這裡就是 $d=10$，測試 MSE $0.440$ 是所有候選裡最差的之一）。來源的反轉也是這句話：訓練誤差最低的模型往往不是最好的，我們真正要選的是驗證或測試誤差低的模型，那個模型的訓練誤差可以比較高（$d=2$ 的訓練 MSE $0.0720$ 就高於 $d=9$ 的 $0.0091$）。

The conclusion of the first three sections: the training error only falls with the degree, so **choosing the model by the training error always picks the most complex one** (here $d=10$, whose test MSE $0.440$ is among the worst of all candidates). The source's reversal says the same: the model with the lowest training error is often not the best one, what we really want is low validation or test error, and that model may have a higher training error (the training MSE $0.0720$ at $d=2$ is above $0.0091$ at $d=9$).

比較實用的判斷方式是同時看兩個數字：訓練誤差與它和驗證誤差的差距。兩者都高且接近，是欠擬合，要增加複雜度；訓練誤差很低但驗證誤差高很多，是過擬合，要降低複雜度或抑制它（KP07 起的各種正則化）；兩者都接近雜訊地板，就是良好擬合。複雜的模型也可能欠擬合：訓練時間不夠、或正則化壓得太強，都會讓複雜的模型擬合不好，所以「模型夠大就不會欠擬合」並不成立。

A practical way to judge is to look at two numbers: the training error and its gap to the validation error. Both high and close means underfitting, so add complexity; a very low training error with a much higher validation error means overfitting, so reduce the complexity or restrain it (the regularization methods from KP07 on); both near the noise floor is a good fit. A complex model can also underfit: too little training or too strong a regularization keeps even a complex model from fitting, so "a big enough model never underfits" is not true.

$d=2$ 與 $d=3$ 的測試 MSE 是 $0.0963$ 與 $0.0971$，差 $0.0008$，這個差距遠小於一組資料的隨機起伏，實際上分不出來。這時合理的選擇是比較簡單的那個：誤差差不多，參數少、變異小 `[補充]`（KP07 會把這個想法變成可以計算的規則）。

The test MSEs of $d=2$ and $d=3$ are $0.0963$ and $0.0971$, a difference of $0.0008$, far below the random scatter of one data set, so in practice they cannot be told apart. The reasonable choice is then the simpler one: similar error, fewer parameters, smaller variance `[Added]` (KP07 turns this idea into a computable rule).

最後一個問題。挑次數要靠**沒參與訓練**的資料；如果直接用測試集挑，就是 KP01 的作弊 #2，所以需要一份獨立的驗證集。可是我們這裡總共只有 12 個點，拿出幾個當驗證集，訓練就更少，估計也更不穩。資料這麼少時，怎麼不浪費任何一筆又能誠實地選模型？這就是 KP06 要回答的問題。

One last question. Choosing the degree needs data that **took no part in training**; choosing it with the test set is cheating #2 of KP01, so a separate validation set is needed. But here we have only 12 points in all, and setting some aside as a validation set leaves even less for training and makes the estimate less stable. With so little data, how can we avoid wasting any point and still choose the model honestly? That is the question KP06 answers.

### 本單元重點整理（KP05 summary）

**一句話重點**：欠擬合是高偏差低變異，訓練與測試誤差都高；過擬合是低偏差高變異，訓練誤差極低、測試誤差高；訓練誤差隨複雜度一路下降，所以要靠驗證或測試誤差選模型。

**One-sentence summary**: Underfitting is high bias and low variance with both training and test error high; overfitting is low bias and high variance with a very low training error and a high test error; the training error falls all the way with complexity, so the model must be chosen by the validation or test error.

**白話說明**：想像畫人像：簡筆畫連五官都沒有，誰也認不出（欠擬合）；寫實肖像抓住特徵（良好擬合）；連毛孔都畫出來，只像那一張照片，換個角度就不像了（過擬合）。多項式的次數就是「畫得多細」的刻度。

**Plain explanation**: Think of drawing a portrait: a stick figure lacks even the facial features, so nobody can recognize it (underfitting); a realistic portrait captures the features (good fit); drawing every pore makes it resemble only that one photo and not the person from another angle (overfitting). The degree of the polynomial is the scale of "how fine the drawing is".

**正式定義**：泛化 $=$ 對沒見過的資料的表現；$\mathbb E\big((\hat f-y)^2\big)=\text{bias}^2+\text{var}+\sigma^2$；欠擬合：bias 大、var 小、訓練誤差也大；過擬合：bias 小、var 大、訓練誤差遠小於測試誤差。

**Formal definition**: generalization $=$ performance on unseen data; $\mathbb E\big((\hat f-y)^2\big)=\text{bias}^2+\text{var}+\sigma^2$; underfitting: large bias, small var, and a large training error too; overfitting: small bias, large var, and a training error far below the test error.

**例子**：真實函數二次、$n=12$、$\sigma^2=0.09$：訓練 MSE 由 $0.288$（$d=1$）降到 $0.007$（$d=10$）；測試 MSE 在 $d=2$ 最低（$0.0963$），$d=9$ 升到 $0.289$。期望測試 MSE 拆解：$d=1$ 是 $0.208+0.013+0.09$，$d=9$ 是 $0+0.117+0.09$。

**Example**: quadratic truth, $n=12$, $\sigma^2=0.09$: the training MSE falls from $0.288$ ($d=1$) to $0.007$ ($d=10$); the test MSE is lowest at $d=2$ ($0.0963$) and rises to $0.289$ at $d=9$. The expected test MSE splits as $0.208+0.013+0.09$ for $d=1$ and $0+0.117+0.09$ for $d=9$.

**常見迷思**：

- 「訓練誤差低就是模型好」——訓練誤差隨複雜度只降不升，要看驗證或測試誤差。
- 「過擬合是因為資料有雜訊」`[補充]`——沒有雜訊也會因模型彈性過大而在資料點之間亂擺，雜訊只是讓問題更明顯。
- 「只要模型夠複雜就不會欠擬合」——訓練不足或正則化太強，複雜的模型也會欠擬合。
- 「測試誤差可以降到 0」——有雜訊時它不會低於 $\sigma^2$；訓練誤差低於 $\sigma^2$ 反而是警訊。

**Common misconceptions**:

- "A low training error means a good model." The training error only falls with complexity; look at the validation or test error.
- "Overfitting happens because the data has noise." `[Added]` Even without noise a too flexible model swings between the data points; noise only makes the problem more visible.
- "A complex enough model never underfits." Too little training or too strong a regularization makes even a complex model underfit.
- "The test error can reach 0." With noise it does not go below $\sigma^2$; a training error below $\sigma^2$ is a warning instead.

**圖**：本單元兩張圖：`lec04-kp05-fits`（同一組點的 1、2、9 次擬合，功能 1、2）與 `lec04-kp05-train-test`（訓練與測試 MSE 對次數，功能 1、2），以 `assets/figures/src/lec04-kp05-figures.py` 繪製；腳本印出本講義引用的每一個數字。

**Figure**: Two figures in this unit: `lec04-kp05-fits` (degree 1, 2 and 9 fits of the same points; functions 1, 2) and `lec04-kp05-train-test` (training and test MSE against the degree; functions 1, 2), drawn by `assets/figures/src/lec04-kp05-figures.py`; the script prints every number quoted in this note.

**想深入**（選讀）`[補充]`：模型訓練到 $d=11$（$p=12=n$）就能剛好穿過全部 12 個點，訓練 MSE 為 0，這叫插值；此時殘差沒有自由度，沒有東西可以估計 $\sigma^2$。這就是 KP04 的「死背模型」：模型完全跟著每一筆標籤走。另一個觀察是，即使在實務上的大型網路，參數比資料還多，也常常沒有出現這裡看到的測試誤差暴增，這是現代深度學習仍在研究的現象（所謂雙下降），本課程不處理。

**Going deeper** (optional) `[Added]`: A model of degree $d=11$ ($p=12=n$) passes exactly through all 12 points and has a training MSE of 0; this is interpolation. The residuals then have no degrees of freedom left and nothing is available to estimate $\sigma^2$. This is the "memorizing model" of KP04: the model follows every label exactly. Another observation: large networks in practice, with more parameters than data, often do not show the blow-up of the test error seen here, a phenomenon modern deep learning still studies (so-called double descent) and this course does not treat.

**小練習**：

1. 三個模型的（訓練 MSE，測試 MSE）是 A $(0.29,\,0.30)$、B $(0.07,\,0.10)$、C $(0.01,\,0.29)$，雜訊地板 $\sigma^2=0.09$。把它們分別歸為欠擬合、良好擬合或過擬合，說出理由。（提示：看訓練誤差是否遠高於或低於地板，以及兩個誤差的差距。）
2. $n=12$、$\sigma^2=0.09$，一個含有真實函數的最小平方模型有 $p=6$ 個參數，它的期望訓練 MSE 是多少？$p=12$ 時呢？（提示：$\sigma^2(n-p)/n$。）
3. 某模型在 $x$ 上的 $\text{bias}^2=0.05$、$\text{var}=0.03$、$\sigma^2=0.09$；另一個較簡單的模型 $\text{bias}^2=0.20$、$\text{var}=0.01$。算兩者的期望測試 MSE，並說明哪一個較好、較簡單的那個屬於哪一種狀態。（提示：三項相加。）

**Practice**:

1. Three models have (training MSE, test MSE) of A $(0.29,\,0.30)$, B $(0.07,\,0.10)$ and C $(0.01,\,0.29)$, and the noise floor is $\sigma^2=0.09$. Classify each as underfit, good fit or overfit and give the reason. (Hint: look at whether the training error is far above or below the floor and at the gap between the two errors.)
2. $n=12$, $\sigma^2=0.09$, and a least-squares model that contains the truth has $p=6$ parameters: what is its expected training MSE? And for $p=12$? (Hint: $\sigma^2(n-p)/n$.)
3. A model has $\text{bias}^2=0.05$, $\text{var}=0.03$ and $\sigma^2=0.09$ at $x$; a simpler model has $\text{bias}^2=0.20$ and $\text{var}=0.01$. Compute the expected test MSE of both, say which is better, and which state the simpler one is in. (Hint: add the three terms.)

**解題示範**（用另一組數字示範第 1 題的判斷方法）：模型 D 的（訓練 MSE，測試 MSE）是 $(0.05,\,0.30)$，$\sigma^2=0.09$。

1. 訓練 MSE $0.05$ 低於地板 $0.09$：模型把雜訊的一部分也背下來了，這是過擬合的警訊。
2. 差距 $0.30-0.05=0.25$，約是地板的 $2.8$ 倍：新資料上的表現遠比訓練時差。
3. 測試 MSE $0.30$ 遠高於地板：泛化不好。
4. 結論：過擬合。處理方向是降低複雜度，或用正則化抑制它（KP07 起）。

**Worked solution** (the judgment of exercise 1 shown with another set of numbers): model D has (training MSE, test MSE) of $(0.05,\,0.30)$ and $\sigma^2=0.09$.

1. The training MSE $0.05$ is below the floor $0.09$: the model has memorized part of the noise, a warning sign of overfitting.
2. The gap is $0.30-0.05=0.25$, about $2.8$ times the floor: the performance on new data is far worse than during training.
3. The test MSE $0.30$ is far above the floor: generalization is poor.
4. Conclusion: overfitting. The remedy is to reduce the complexity or restrain it with regularization (from KP07 on).

### 來源更正與取捨（Corrections to the source）

- 來源的過擬合表（1 次 $5.2/5.0$、3 次 $1.1/1.2$、10 次 $0.05/3.8$）是示意數字，無法重現；而且它說真實函數是二次，卻說 3 次「最接近真實函數」，前後不一致。本講義不引用這些數字，改用自己的實驗（二次真實函數、$n=12$、$\sigma^2=0.09$、固定亂數種子），結果是 2 次的測試 MSE 最低，3 次幾乎持平。
- 來源說良好擬合是「bias 與 variance 都夠低」。這裡補上量化的基準：測試誤差不可能低於雜訊地板 $\sigma^2$（KP03），所以良好擬合的意思是測試誤差接近 $\sigma^2$，不是接近 0。
- 來源的「訓練誤差低 $=$ 模型好」被正確地列為誤解；本講義再加一條可操作的警訊：訓練誤差低於 $\sigma^2$ 通常表示模型記住了雜訊。
- 來源的 $\mathbf{err}$、$\mathbf{Err}$ 是對點求和；本講義這一單元的訓練 MSE 與測試 MSE 是平均（每點），數字才能直接和 $\sigma^2$ 比較。
- 來源的圖（Fig. 5）是書上的示意圖，本講義不使用，全部圖都由程式重繪。

- The source's overfitting table (degree 1 $5.2/5.0$, degree 3 $1.1/1.2$, degree 10 $0.05/3.8$) is illustrative and cannot be reproduced; it also says the truth is quadratic yet calls degree 3 "closest to the true function", which is inconsistent. This note does not quote those numbers and uses its own experiment (quadratic truth, $n=12$, $\sigma^2=0.09$, fixed random seed), which gives the lowest test MSE at degree 2 with degree 3 almost level.
- The source says a good fit has "both bias and variance low enough". This note adds a quantitative yardstick: the test error cannot go below the noise floor $\sigma^2$ (KP03), so a good fit means a test error near $\sigma^2$, not near 0.
- The source correctly lists "low training error $=$ good model" as a misconception; this note adds an operational warning: a training error below $\sigma^2$ usually means the model has memorized the noise.
- The source's $\mathbf{err}$ and $\mathbf{Err}$ are sums over points; in this unit the training and test MSE are means (per point) so that they compare directly with $\sigma^2$.
- The source's figure (Fig. 5) is a book illustration and is not used here; all figures are redrawn by program.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 泛化 | generalization |
| 欠擬合 | underfitting |
| 過擬合 | overfitting |
| 良好擬合 | good fit |
| 偏差 | bias |
| 變異 | variance |
| 雜訊地板 | noise floor |
| 訓練誤差 | training error |
| 測試誤差 | test error |
| 驗證誤差 | validation error |
| 多項式次數 | polynomial degree |
| 插值 | interpolation |
