# 第4週 KP01：資料切分與兩種作弊（Data splits and two kinds of cheating）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第4章「正則化與過擬合」的第一個知識點（KP01）。第3章教的是怎麼把損失壓低；這一章要先問一個更基本的問題：壓低的那個數字，可不可信？先備知識只有第3章的訓練流程與「準確率」這種簡單分數；第一次出現的符號都會定義，所有數字都可以自己重算。內容依 CCC-LNS 第5章 KP01 的主文與教授祕笈消化後重新撰寫，讀取日期 2026-10-20；來源沒有的內容標示 `[補充]`。

This is the first knowledge point (KP01) of Chapter 4, "Regularization and overfitting". Chapter 3 taught how to push the loss down; this chapter starts with a more basic question: can the number we pushed down be trusted? The only prerequisites are the training loop of Chapter 3 and simple scores such as accuracy; every symbol is defined when it first appears and every number can be recomputed. The text is rewritten from the KP01 main text and the professor's guide of CCC-LNS Chapter 5 (accessed on 2026-10-20); content that the source does not contain is marked `[Added]`.

[下載本單元投影片](../slides/lec04-kp01-data-splits-and-cheating.jbd.json)

## KP01 資料切分與兩種作弊（Data splits and two kinds of cheating）

### 學習目標（Learning objectives）

讀完之後，你能說出為什麼不能用同一份資料既訓練又評分；能把資料切成互不相交的訓練、驗證、測試三份，並說出各自的用途；能舉出兩種作弊（訓練與測試重疊、用測試集選模型）與一種不算作弊卻會讓驗證失效的重疊，並用小數字算出作弊讓分數虛高多少。

After reading, you can say why the same data must not be used both for training and for scoring; cut the data into three disjoint parts (training, validation, test) and state what each is for; name two kinds of cheating (training/test overlap, choosing a model on the test set) and one overlap that is not cheating but breaks validation; and compute with small numbers how much cheating inflates a score.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $\mathcal{D}$ | 手上全部的資料（資料集） | all the data at hand (the data set) |
| $\mathcal{T}$ | 訓練集：用來調整模型的權重 | training set: used to fit the weights |
| $\mathcal{V}$ | 驗證集：用來挑超參數、決定何時停止 | validation set: used to pick hyper-parameters and decide when to stop |
| $\mathcal{R}$ | 測試集：只在最後報告時使用 | test set: used only for the final report |
| $\cap,\ \cup$ | 交集、聯集 | intersection, union |
| $\varnothing$ | 空集合（沒有任何共同元素） | the empty set (no element in common) |

### 1. 為什麼不能又教又考（Why the same data cannot teach and examine）

想像一位老師把一疊題目發給學生當作業，期末考又原封不動拿同一疊題目。學生只要把答案背起來就能拿滿分，但這個滿分說明不了他懂不懂這門課。機器學習的模型也一樣：它的目標是對**沒見過**的資料做出好預測，所以評分用的資料必須是它沒見過的資料。

Imagine a teacher who hands out a pile of problems as homework and then uses the very same pile as the final exam. A student can memorise the answers and score full marks, but that score says nothing about whether he understands the course. A machine-learning model is in the same position: its goal is to predict well on data it has **not seen**, so the data used for scoring must be data it has not seen.

用一個小實驗看這件事有多嚴重 `[補充]`：在單位正方形裡隨機放 60 個點，每個點的標籤隨機取 $+1$ 或 $-1$（標籤和位置毫無關係，所以任何模型都只能靠猜）。拿一個「最近鄰」分類器（1-nearest-neighbour，1-NN：對一個新點，直接抄最靠近它的那個訓練點的標籤）。它等於把 60 個答案背了下來。

A small experiment shows how serious this is `[Added]`: put 60 points at random in the unit square and give each a label $+1$ or $-1$ at random (the label has nothing to do with the position, so any model can only guess). Take a nearest-neighbour classifier (1-NN: for a new point, copy the label of the closest training point). It has memorised the 60 answers.

- 拿這 60 個**訓練點本身**來打分：每個點最近的訓練點就是它自己（距離 0），所以答對率是 $100\%$。
- 另外隨機抽 1000 個**新點**（也是隨機標籤）打分：答對率是 $49.4\%$，和丟硬幣（$50\%$）沒有差別。

- Scored on the **60 training points themselves**: the nearest training point of each is itself (distance 0), so the accuracy is $100\%$.
- Scored on 1000 **fresh points** (also with random labels): the accuracy is $49.4\%$, no different from a coin toss ($50\%$).

同一個模型，同一天，兩個分數差了 $50.6$ 個百分點〔程式計算〕。哪一個才是它真正的本事？當然是第二個。這就是資料切分的全部動機：**評分只能用沒參與過學習的資料。**

The same model on the same day gets two scores that differ by $50.6$ percentage points [computed by program]. Which one is its real ability? The second, of course. This is the whole motivation for splitting data: **score only on data that did not take part in learning.**

### 2. 三疊資料，各有各的用途（Three piles, three jobs）

最簡單的切分是兩份：訓練集 $\mathcal{T}$ 與測試集 $\mathcal{R}$，兩者沒有共同元素：

The simplest split has two parts, a training set $\mathcal{T}$ and a test set $\mathcal{R}$, with no element in common:

$$
\mathcal{D}=\mathcal{T}\cup\mathcal{R},\qquad \mathcal{T}\cap\mathcal{R}=\varnothing
$$

但實際訓練時還有很多要決定的事：學習率設多少？網路要幾層？訓練到第幾輪停？這些**超參數**（hyper-parameters，不由梯度下降學出來、要人或程式去挑的設定）也得靠分數來挑。如果用測試集來挑，測試集就不再是「沒見過」的了（第 3 節會算給你看）。所以再切出一份**驗證集**（validation set）$\mathcal{V}$，專門負責挑選：

But training involves many more decisions: what learning rate, how many layers, after which round to stop? These **hyper-parameters** (settings that gradient descent does not learn and that a person or a program must pick) also have to be picked by a score. If the test set is used for that, it is no longer "unseen" (Section 3 computes this for you). So a third part is cut out, the **validation set** $\mathcal{V}$, whose only job is selection:

$$
\mathcal{D}=\mathcal{T}\cup\mathcal{V}\cup\mathcal{R},\qquad \mathcal{T}\cap\mathcal{V}=\varnothing,\quad \mathcal{T}\cap\mathcal{R}=\varnothing,\quad \mathcal{V}\cap\mathcal{R}=\varnothing
$$

三份兩兩互不相交，各有分工：

The three parts are pairwise disjoint and each has its own job:

| 集合 | 用途 | 比喻 |
|---|---|---|
| $\mathcal{T}$ 訓練集 | 調整權重（梯度下降只看它） | 平常的練習題 |
| $\mathcal{V}$ 驗證集 | 挑超參數、決定何時停止、在幾個模型中選一個 | 模擬考 |
| $\mathcal{R}$ 測試集 | 全部選完之後，只報告一次最終分數 | 期末考 |

| Set | Job | Analogy |
|---|---|---|
| $\mathcal{T}$ training | fit the weights (gradient descent sees only this) | everyday practice problems |
| $\mathcal{V}$ validation | pick hyper-parameters, decide when to stop, choose among models | mock exam |
| $\mathcal{R}$ test | after every choice is made, report the final score once | final exam |

例：$\mathcal{D}$ 有 100 筆資料，常見比例是 $60/20/20$：$\mathcal{T}$ 有 60 筆、$\mathcal{V}$ 有 20 筆、$\mathcal{R}$ 有 20 筆，$60+20+20=100$。比例不是重點，重點是**測試集從頭到尾沒被用來做任何決定**。

Example: $\mathcal{D}$ has 100 examples and a common ratio is $60/20/20$: $\mathcal{T}$ has 60, $\mathcal{V}$ has 20 and $\mathcal{R}$ has 20, and $60+20+20=100$. The ratio is not the point; the point is that **the test set is never used to make any decision**.

![資料集 D 切成三份互不相交的集合：訓練集 T（60 筆）調整權重，驗證集 V（20 筆）挑超參數與停止時間，測試集 R（20 筆）只做最後報告。（The data set D is cut into three disjoint sets: the training set T (60 examples) fits the weights, the validation set V (20) picks hyper-parameters and the stopping time, and the test set R (20) is used only for the final report.）](../assets/figures/lec04-kp01-split-roles.png)

**圖說 `lec04-kp01-split-roles`**：先看長條，三個方塊的寬度和是整條 100 筆，方塊之間有空隙，表示沒有重疊；再看每個方塊下方的箭頭，它指向那份資料的唯一用途。測試集的用途只有一個「報告一次」，這是它和驗證集最大的差別。〔功能：1〕

**Caption `lec04-kp01-split-roles`**: Look at the bar first: the widths of the three boxes add up to the whole 100, and the gaps between them mean no overlap. Then look at the arrow under each box, which points to the single job of that part. The test set has exactly one job, "report once", and that is its main difference from the validation set. [Function: 1]

### 3. 兩種作弊，和一種不算作弊的重疊（Two kinds of cheating and one harmless-looking overlap）

切分有三條規則（三個交集都是空集合），破壞任何一條都會讓分數失真。來源把前兩種叫做作弊：

The split has three rules (all three intersections are empty), and breaking any of them distorts the scores. The source names the first two as cheating:

**作弊一（Cheating #1）：$\mathcal{T}\cap\mathcal{R}\neq\varnothing$，訓練集與測試集重疊。** 模型訓練時已經看過考題，測試分數虛高。第 1 節的實驗就是這種作弊：用訓練點打分得到 $100\%$，換成新點只剩 $49.4\%$。

**Cheating #1: $\mathcal{T}\cap\mathcal{R}\neq\varnothing$, the training and test sets overlap.** The model has already seen the exam questions during training, so the test score is inflated. The experiment of Section 1 is exactly this: $100\%$ on the training points, only $49.4\%$ on fresh points.

**作弊二（Cheating #2）：$\mathcal{V}\cap\mathcal{R}\neq\varnothing$，拿測試集當驗證集用。** 它更隱蔽，因為模型沒有直接「訓練」在測試集上，只是你**根據測試分數**去挑東西。用一個隨機實驗 `[補充]` 說明：隨機做 200 個線性分類器（權重是亂數），拿 20 個隨機標籤的測試點打分，然後留下得分最高的那一個。

**Cheating #2: $\mathcal{V}\cap\mathcal{R}\neq\varnothing$, the test set is used as a validation set.** It is subtler, because the model is never trained on the test set; you only **choose by the test score**. A random experiment `[Added]` shows it: make 200 linear classifiers with random weights, score them on 20 test points with random labels, and keep the one with the highest score.

- 200 個分類器的平均得分是 $51.5\%$，和猜一樣。
- 得分最高的那一個在 20 個測試點上答對 $75.0\%$（15 題）。
- 把這個「冠軍」拿到 1000 個新點上考：$49.1\%$，又回到猜的水準。

- The average score of the 200 classifiers is $51.5\%$, the same as guessing.
- The best one gets $75.0\%$ (15 of 20) on the test points.
- That "champion" scored on 1000 fresh points gets $49.1\%$, back at the level of guessing.

為什麼會這樣？一個只會猜的分類器，在 20 題裡答對 15 題以上的機率是 $\sum_{k=15}^{20}\binom{20}{k}/2^{20}=21700/1048576\approx2.07\%$。如果 200 個分類器彼此獨立，至少有一個達到這個分數的機率是 $1-(1-0.0207)^{200}\approx98.5\%$〔程式計算〕。也就是說，「挑最高分」幾乎一定會挑到一個靠運氣好的分數。挑得越多，最高分越漂亮、也越不可信。

Why? A guesser gets at least 15 of 20 right with probability $\sum_{k=15}^{20}\binom{20}{k}/2^{20}=21700/1048576\approx2.07\%$. If the 200 classifiers were independent, the probability that at least one reaches this score is $1-(1-0.0207)^{200}\approx98.5\%$ [computed by program]. In other words, "pick the highest score" almost surely picks a score that is good only by luck. The more you pick from, the prettier and the less trustworthy the best score is.

![兩種作弊的效果：左半是作弊一，1-NN 在自己的訓練點上得 100%，在 1000 個新點上只有 49.4%；右半是作弊二，在 20 個測試點上挑出的最佳分類器得 75.0%，在 1000 個新點上只有 49.1%。虛線是猜的水準 50%。（The effect of the two kinds of cheating: the left half is cheating #1, a 1-NN gets 100% on its own training points but only 49.4% on 1000 fresh points; the right half is cheating #2, the best classifier chosen on 20 test points gets 75.0% but only 49.1% on 1000 fresh points. The dashed line is the chance level, 50%.）](../assets/figures/lec04-kp01-cheating-scores.png)

**圖說 `lec04-kp01-cheating-scores`**：先比較每一對紅綠長條：紅色是作弊時看到的分數，綠色是同一個模型在新點上的真實分數。兩對的紅色都明顯高於虛線，綠色都貼著虛線，表示這兩個模型其實什麼都沒學到。這是 `[補充]` 的數值例，標籤全是隨機的，所以真實水準必然是 $50\%$。〔功能：1、2〕

**Caption `lec04-kp01-cheating-scores`**: Compare each red-green pair first: red is the score seen while cheating, green is the true score of the same model on fresh points. In both pairs the red bar is well above the dashed line and the green bar sits on it, so neither model has learned anything. This is an `[Added]` numerical example whose labels are all random, so the true level must be $50\%$. [Function: 1, 2]

**第三種重疊：$\mathcal{T}\cap\mathcal{V}\neq\varnothing$。** 它不算作弊，因為測試集沒被碰到；但它讓驗證分數變成訓練分數的同義詞，你再也看不出模型有沒有過擬合。用一個小回歸例子 `[補充]` 看：真實曲線 $f(x)=1+2x-1.5x^2$，觀測 $y=f(x)+\varepsilon$，雜訊標準差 $\sigma=0.3$（所以不可避免的誤差是 $\sigma^2=0.09$，第 4 個 KP 才會細講）。取 20 個訓練點，用 1、2、3、5、9 次多項式去配，比較三種「分數」（誤差是均方誤差，越小越好）：

**The third overlap: $\mathcal{T}\cap\mathcal{V}\neq\varnothing$.** It is not cheating, since the test set is untouched, but it makes the validation score a synonym of the training score, and you can no longer see whether the model overfits. A small regression example `[Added]`: the true curve is $f(x)=1+2x-1.5x^2$, observations are $y=f(x)+\varepsilon$ with noise standard deviation $\sigma=0.3$ (so the unavoidable error is $\sigma^2=0.09$, detailed in KP04). Take 20 training points and fit polynomials of degree 1, 2, 3, 5 and 9, comparing three kinds of "score" (the error is the mean squared error, smaller is better):

| 多項式次數 | 訓練誤差（＝重疊時的「驗證」誤差） | 乾淨驗證誤差（20 個新點） | 測試誤差（2000 個新點） |
|---|---|---|---|
| 1 | 0.3212 | 0.2401 | 0.2989 |
| 2 | 0.1251 | 0.0892 | 0.0940 |
| 3 | 0.1227 | 0.0847 | 0.0971 |
| 5 | 0.0865 | 0.1130 | 0.1652 |
| 9 | 0.0614 | 1.9277 | 9.1979 |

| Polynomial degree | Training error (= the "validation" error under overlap) | Clean validation error (20 fresh points) | Test error (2000 fresh points) |
|---|---|---|---|
| 1 | 0.3212 | 0.2401 | 0.2989 |
| 2 | 0.1251 | 0.0892 | 0.0940 |
| 3 | 0.1227 | 0.0847 | 0.0971 |
| 5 | 0.0865 | 0.1130 | 0.1652 |
| 9 | 0.0614 | 1.9277 | 9.1979 |

用重疊的驗證集選模型，會選誤差最小的 9 次（0.0614），結果測試誤差高達 9.1979，是最好的 2 次（0.0940）的約 98 倍。用乾淨的驗證集選，選到 3 次，測試誤差 0.0971，已經很接近最好。注意乾淨驗證集只有 20 個點，所以它選 3 次而不是測試誤差最小的 2 次（0.0847 與 0.0892 只差一點點）：驗證集小，分數會有雜訊，這是後面的知識點要處理的問題；但它不會被訓練分數騙去選 9 次。

Choosing with the overlapping validation set picks the smallest error, degree 9 (0.0614), and its test error is as large as 9.1979, about 98 times that of the best, degree 2 (0.0940). Choosing with the clean validation set picks degree 3, whose test error 0.0971 is already close to the best. Note that the clean validation set has only 20 points, so it picks degree 3 rather than degree 2, which has the smallest test error (0.0847 and 0.0892 differ only slightly): a small validation set gives noisy scores, a problem later KPs deal with; but it is not fooled by the training score into picking degree 9.

### 4. 紀律的代價，與留給 KP02 的問題（What the discipline costs and the question left for KP02）

切分的規則可以濃縮成一句話：**越靠近最後的決定，用到的資料越要保持沒見過。** 訓練集可以反覆用；驗證集每挑一次就多洩漏一點；測試集只能用一次，用完不能回頭再改模型。如果你看完測試分數後又去改超參數，等於把測試集變成了驗證集，也就是作弊二。

The rules of splitting can be condensed into one sentence: **the closer to the final decision, the more the data must stay unseen.** The training set may be used again and again; the validation set leaks a little more each time you choose by it; the test set may be used once, and the model must not be changed afterwards. If you look at the test score and then tweak a hyper-parameter, the test set has become a validation set, which is cheating #2.

這也有代價：每一份留作驗證或測試的資料，就少了一份拿來訓練。資料很少的時候該怎麼辦？KP09 的交叉驗證會給出辦法。

There is a price: every example kept for validation or testing is one fewer for training. What to do when data are scarce? The cross-validation of KP09 gives an answer.

現在我們確定了「要用沒見過的資料評分」。但評分用的「誤差」本身又是怎麼量的？一個估計又準又穩，要怎麼用數字說清楚？這是 KP02 的主題：偏差、變異與均方誤差。

Now we know to "score on unseen data". But how is the "error" used for scoring itself measured? How do we say in numbers that an estimate is both accurate and stable? That is the topic of KP02: bias, variance and mean squared error.

### 本單元重點整理（KP01 summary）

**一句話重點**：資料切成互不相交的訓練、驗證、測試三份，分別用來調權重、挑超參數、最後報告一次；測試集只要被用來做過任何決定（不論是訓練在上面，還是依它的分數去挑），分數就虛高。

**One-sentence summary**: Cut the data into three disjoint parts, used to fit the weights, to pick hyper-parameters and to report once at the end; as soon as the test set is used to make any decision (training on it or choosing by its score), the score is inflated.

**白話說明**：練習題、模擬考、期末考不能混用。期末考題如果提前見過，或是根據期末考的成績去調整讀書方法，成績都不能代表真正的實力。

**Plain explanation**: Practice problems, mock exams and the final exam must not be mixed. If the final-exam questions were seen in advance, or the way of studying is adjusted according to the final-exam score, the grade no longer shows real ability.

**正式定義**：$\mathcal{D}=\mathcal{T}\cup\mathcal{V}\cup\mathcal{R}$，且 $\mathcal{T}\cap\mathcal{V}=\mathcal{T}\cap\mathcal{R}=\mathcal{V}\cap\mathcal{R}=\varnothing$。作弊一：$\mathcal{T}\cap\mathcal{R}\neq\varnothing$；作弊二：$\mathcal{V}\cap\mathcal{R}\neq\varnothing$；$\mathcal{T}\cap\mathcal{V}\neq\varnothing$ 不涉及測試集，但使驗證誤差失去偵測過擬合的能力。

**Formal definition**: $\mathcal{D}=\mathcal{T}\cup\mathcal{V}\cup\mathcal{R}$ with $\mathcal{T}\cap\mathcal{V}=\mathcal{T}\cap\mathcal{R}=\mathcal{V}\cap\mathcal{R}=\varnothing$. Cheating #1: $\mathcal{T}\cap\mathcal{R}\neq\varnothing$; cheating #2: $\mathcal{V}\cap\mathcal{R}\neq\varnothing$; $\mathcal{T}\cap\mathcal{V}\neq\varnothing$ does not involve the test set but leaves the validation error unable to detect overfitting.

**例子**：100 筆資料切成 $60/20/20$；隨機標籤下，1-NN 在自己的訓練點得 $100\%$、在新點得 $49.4\%$；200 個隨機分類器在 20 個測試點上挑出的最佳者得 $75.0\%$、在新點得 $49.1\%$。逐步計算見第 1–3 節。

**Example**: 100 examples cut as $60/20/20$; with random labels a 1-NN gets $100\%$ on its own training points and $49.4\%$ on fresh points; the best of 200 random classifiers chosen on 20 test points gets $75.0\%$ and on fresh points $49.1\%$. The computations are in Sections 1–3.

**常見迷思**：

- 「測試集偶爾拿來調一下參數沒關係」——只要依測試分數做過一次選擇，它就不再是沒見過的資料（作弊二）。
- 「驗證集要和測試集一樣大」——大小不是重點，互不相交才是。
- 「訓練集和驗證集重疊沒關係」——它不算作弊，但驗證誤差會變得和訓練誤差一樣漂亮，過擬合就看不出來。
- 「比例一定要 $60/20/20$」——比例只是常見選擇，要依資料量調整（資料少時用 KP09 的交叉驗證）。

**Common misconceptions**:

- "Using the test set to tweak a parameter now and then is harmless." Once a single choice has been made by the test score, it is no longer unseen data (cheating #2).
- "The validation set must be as large as the test set." Size is not the point; being disjoint is.
- "Training and validation sets may overlap." It is not cheating, but the validation error becomes as pretty as the training error and overfitting goes unseen.
- "The ratio must be $60/20/20$." It is only a common choice and depends on the amount of data (with little data use the cross-validation of KP09).

**圖**：本單元兩張圖：`lec04-kp01-split-roles`（三份資料的分工，功能 1）、`lec04-kp01-cheating-scores`（兩種作弊讓分數虛高，功能 1、2），以 `assets/figures/src/lec04-kp01-figures.py` 繪製；多項式表格的數字由同一支程式印出。

**Figure**: Two figures in this unit: `lec04-kp01-split-roles` (the jobs of the three parts; function 1) and `lec04-kp01-cheating-scores` (how the two kinds of cheating inflate the score; functions 1, 2), drawn by `assets/figures/src/lec04-kp01-figures.py`; the numbers of the polynomial table are printed by the same program.

**想深入**（選讀）：作弊二的陷阱可以量化：從 $m$ 個互相獨立、都只會猜的模型裡挑測試分數最高者，最高分的期望會隨 $m$ 變大而升高，而它在新資料上的真實水準始終是猜的水準。所以「驗證集上選了很多組超參數」也會漸漸把驗證分數用成偏樂觀，這也是為什麼最後還要保留一份從沒被選擇碰過的測試集。

**想深入**之二：資料切分還有一個常見的陷阱叫資料洩漏（data leakage）：例如在切分之前就用全部資料算平均與標準差來做標準化，測試集的資訊就偷偷進了訓練。標準化的統計量應只用 $\mathcal{T}$ 算，再套用到 $\mathcal{V}$ 與 $\mathcal{R}$（這個建議 `[補充]`，來源沒有談）。

**Going deeper** (optional): The trap of cheating #2 can be quantified: pick the model with the highest test score out of $m$ independent models that can only guess; the expected best score rises as $m$ grows, while its true level on new data stays at the guessing level. So trying many hyper-parameter settings on the validation set also makes the validation score gradually optimistic, which is why a test set that no selection ever touched is still kept at the end.

**Going deeper**, part two: another common pitfall of splitting is data leakage: for example, standardising with the mean and standard deviation of all the data before the split lets information of the test set slip into training. The statistics for standardising should be computed on $\mathcal{T}$ only and then applied to $\mathcal{V}$ and $\mathcal{R}$ (this advice is `[Added]`; the source does not discuss it).

**小練習**：

1. 一個資料集有 2400 筆，切成 $70/15/15$：三份各有幾筆？它們的和是多少？（提示：$2400\times0.7$ 等。）
2. 一個只會猜的分類器在 10 個測試題中至少答對 8 題的機率是多少？如果有 50 個互相獨立的這種分類器，至少有一個達到的機率是多少？（提示：$\sum_{k=8}^{10}\binom{10}{k}/2^{10}$，再用 $1-(1-p)^{50}$。）
3. 說明下面三種做法各屬於哪一種重疊或洩漏，以及為什麼會讓分數失真：（甲）切分前先把 6 筆重複的資料各複製一份，之後隨機切分；（乙）看完測試分數，把學習率從 0.1 改成 0.01 再訓練；（丙）用 $\mathcal{T}$ 當驗證集選早停的輪數。（提示：先問「這份資料有沒有被用來做決定」。）

**Practice**:

1. A data set has 2400 examples and is cut $70/15/15$: how many are in each of the three parts, and what is their sum? (Hint: $2400\times0.7$ and so on.)
2. A classifier that can only guess gets at least 8 of 10 test questions right with what probability? With 50 independent such classifiers, what is the probability that at least one reaches it? (Hint: $\sum_{k=8}^{10}\binom{10}{k}/2^{10}$, then $1-(1-p)^{50}$.)
3. State which kind of overlap or leakage each of these three practices is and why it distorts the score: (A) duplicate 6 examples before the split and then split at random; (B) after seeing the test score, change the learning rate from 0.1 to 0.01 and retrain; (C) use $\mathcal{T}$ as the validation set to choose the early-stopping round. (Hint: first ask "has this data been used to make a decision?")

**解題示範**（以 $n=1000$、$70/15/15$ 示範第 1 題的算法）：

1. 訓練集：$1000\times0.70=700$ 筆。
2. 驗證集：$1000\times0.15=150$ 筆；測試集也是 $150$ 筆。
3. 檢查和：$700+150+150=1000$，恰好是全部資料，且每一筆只出現在一份裡。
4. 兩兩不相交：因為是從打亂的編號依序切出 $[0,700)$、$[700,850)$、$[850,1000)$，三段沒有共同編號。

**Worked solution** (the method of exercise 1, shown with $n=1000$ and $70/15/15$):

1. Training set: $1000\times0.70=700$ examples.
2. Validation set: $1000\times0.15=150$ examples; the test set is also $150$.
3. Check the sum: $700+150+150=1000$, exactly all the data, with each example in one part only.
4. Pairwise disjoint: the shuffled indices are cut in order into $[0,700)$, $[700,850)$ and $[850,1000)$, three ranges with no index in common.

### 來源更正與取捨（Corrections to the source）

- 來源說 $\mathcal{T}\cap\mathcal{V}\neq\varnothing$「不算作弊」。就測試集是否被洩漏而言這是對的；本講義補上它的後果（驗證誤差等於訓練誤差、偵測不到過擬合），並用多項式的表格實際算出來。
- 來源只用比喻說明兩種作弊「分數會虛高」。本講義補上兩個可重算的隨機實驗，並補上作弊二的機率解釋（從很多個分類器裡挑最高分）。
- 來源把分數虛高當成結論，沒有量化；本講義用的數字（$100\%$ 對 $49.4\%$、$75.0\%$ 對 $49.1\%$）來自固定亂數種子的小實驗，換一個種子數字會變，但方向（作弊分數高、真實水準是猜）不變。
- 來源的式 (1)–(4) 以編號出現；本講義不沿用來源的式號。

- The source says $\mathcal{T}\cap\mathcal{V}\neq\varnothing$ "is not cheating". That is right as far as the test set not being leaked is concerned; this note adds its consequence (the validation error equals the training error and overfitting cannot be detected) and computes it in the polynomial table.
- The source explains with analogies only that the two kinds of cheating make the score "too high". This note adds two recomputable random experiments and the probabilistic explanation of cheating #2 (picking the best from many classifiers).
- The source states that inflation as a conclusion without quantifying it; the numbers used here ($100\%$ versus $49.4\%$, $75.0\%$ versus $49.1\%$) come from small experiments with a fixed random seed, and another seed changes the numbers but not the direction (the cheating score is high, the true level is guessing).
- The source's Eqs. (1)–(4) carry numbers; this note does not reuse the source's equation numbers.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 資料集 | data set |
| 訓練集 | training set |
| 驗證集 | validation set |
| 測試集 | test set |
| 互不相交 | pairwise disjoint |
| 超參數 | hyper-parameter |
| 最近鄰分類器 | nearest-neighbour classifier |
| 作弊（資料洩漏） | cheating (data leakage) |
| 過擬合 | overfitting |
| 均方誤差 | mean squared error |
