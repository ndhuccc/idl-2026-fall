# 第4週 KP06：交叉驗證（Cross-validation）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)

本講義是第4週「正則化與過擬合」的第六個知識點（KP06），承接 KP05 留下的問題：挑模型要靠沒參與訓練的資料，可是資料不多時，切出一份驗證集會讓訓練變少、估計也不穩，怎麼辦？先備知識是 KP01（資料切分與兩種作弊）與 KP05（過擬合與欠擬合）。第一次出現的符號都會定義，所有數字都附有程式計算。內容依 CCC-LNS 第5章 KP9 的主文與教授祕笈消化後重新撰寫，讀取日期 2026-10-20；來源沒有的內容標示 `[補充]`。

This is the sixth knowledge point (KP06) of Week 4, "Regularization and overfitting". It picks up the question KP05 left open: choosing a model needs data that took no part in training, but when the data is scarce, setting aside a validation set leaves less for training and makes the estimate unstable, so what now? It assumes KP01 (data splits and the two kinds of cheating) and KP05 (overfitting and underfitting). Every symbol is defined when it first appears and every number comes from a program. The text is rewritten from the main text and the professor's guide of KP9 in Chapter 5 of CCC-LNS (accessed on 2026-10-20); content that the source does not contain is marked `[Added]`.

[下載本單元投影片](../slides/lec04-kp06-cross-validation.jbd.json)

## KP06 交叉驗證（Cross-validation）

### 學習目標（Learning objectives）

讀完之後，你能說明只切一次驗證集為什麼靠運氣；能把資料切成 $K$ 個互不相交的 fold，寫出 $K$-fold 的流程與 $\mathbf{Err}=\frac1K\sum_k\mathbf{Err}_k$，並用五個 fold 的誤差算出一個交叉驗證誤差；能用交叉驗證選多項式次數，並說出選完之後要用全部資料重新訓練；能比較 K-fold 與 LOOCV 的成本與估計品質，說明為什麼 $K$ 不是越大越好；也能說出交叉驗證不能取代最後的測試集。

After reading, you can explain why a single validation split depends on luck; cut the data into $K$ disjoint folds, write the $K$-fold procedure and $\mathbf{Err}=\frac1K\sum_k\mathbf{Err}_k$, and compute a cross-validation error from five fold errors; choose the polynomial degree by cross-validation and say that after the choice the model is retrained on all the data; compare K-fold with LOOCV in cost and in the quality of the estimate and explain why a larger $K$ is not always better; and say why cross-validation cannot replace the final test set.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $n$ | 資料的總筆數，這一單元 $n=100$ | total number of data points, here $n=100$ |
| $K$ | fold 的個數 | number of folds |
| $\mathcal D_k$ | 第 $k$ 個 fold（資料的一個子集） | the $k$-th fold (a subset of the data) |
| $\mathbf{Err}_k$ | 用其餘 $K-1$ 個 fold 訓練、在 $\mathcal D_k$ 上量到的誤差；本單元以平均平方誤差 $E_k$ 表示 | the error measured on $\mathcal D_k$ by a model trained on the other $K-1$ folds; in this unit written as the mean squared error $E_k$ |
| $\mathrm{CV}$ | 交叉驗證誤差：$\frac1K\sum_kE_k$ | cross-validation error $\frac1K\sum_kE_k$ |
| $d$ | 多項式的次數（複雜度的刻度） | polynomial degree (the scale of complexity) |

### 1. 只切一次，結果靠運氣（A single split depends on luck）

KP05 的最後一個問題是：挑次數要靠沒參與訓練的資料，可是資料很少。把 12 個點再切成訓練與驗證，各剩更少，估計更不穩。為了看清楚「切一次」有多靠運氣，這一單元改用比較大的資料：真實函數仍是 $f(x)=1+2x-1.5x^2$，雜訊標準差 $0.3$（地板 $\sigma^2=0.09$），$n=100$ 個點，$x$ 在 $-1$ 到 $1$ 之間隨機取，亂數種子固定。

The last question of KP05 was that choosing the degree needs data that took no part in training, yet the data is scarce. Splitting 12 points further into training and validation leaves even less and makes the estimate less stable. To see how much a single split depends on luck, this unit uses a larger data set: the true function is still $f(x)=1+2x-1.5x^2$, the noise standard deviation is $0.3$ (floor $\sigma^2=0.09$), and there are $n=100$ points with $x$ drawn at random in $[-1,1]$ and a fixed random seed.

先做一個實驗：把這 100 筆資料隨機切成 80 筆訓練、20 筆驗證，配一個 2 次多項式，量驗證 MSE。重複 300 次，每次用不同的隨機切法（資料與模型都不變），結果是：平均 $0.1006$，標準差 $0.0279$，最小 $0.0375$，最大 $0.1787$（程式計算）。同一個模型、同一批資料，只因為哪 20 筆被分到驗證集，驗證誤差就可以相差 $4.8$ 倍。再讓 1 到 8 次的多項式互相比較、每次選驗證 MSE 最低的次數：300 次裡，只有 158 次選到正確的 2 次，68 次選到 7 次，34 次選到 5 次，其餘分散在 1、3、4、6、8 次。

Run an experiment first: split the 100 points at random into 80 for training and 20 for validation, fit a degree-2 polynomial and measure the validation MSE. Repeat 300 times with a different random split each time (the data and the model stay the same). The result is a mean of $0.1006$, a standard deviation of $0.0279$, a minimum of $0.0375$ and a maximum of $0.1787$ (computed by program). The same model and the same data differ by a factor of $4.8$ in validation error only because of which 20 points happen to fall in the validation set. Now let the polynomials of degree 1 to 8 compete and each time pick the degree with the lowest validation MSE: out of 300 splits only 158 pick the correct degree 2, 68 pick degree 7, 34 pick degree 5 and the rest are spread over degrees 1, 3, 4, 6 and 8.

這就是問題所在。一次切分的驗證誤差混著「這 20 筆剛好難不難」的運氣，用它挑模型，選到什麼要看切法。直覺的解法是：既然一次靠運氣，就多切幾次、平均起來，而且讓每一筆資料都有機會當考題。

This is the problem. The validation error of one split mixes in the luck of "how hard these 20 points happen to be", and choosing a model with it makes the choice depend on the split. The intuitive remedy: if one split depends on luck, split several times and average, and let every point get a turn as an exam question.

### 2. K-fold：輪流當考題（K-fold: every fold takes a turn）

$K$-fold 先把資料隨機洗牌，切成 $K$ 個大小相近、**互不相交**的 fold $\mathcal D_1,\dots,\mathcal D_K$，聯集是全部資料：

$K$-fold first shuffles the data at random and cuts it into $K$ **disjoint** folds $\mathcal D_1,\dots,\mathcal D_K$ of nearly equal size whose union is all the data:

$$
|\mathcal D_1|\approx\cdots\approx|\mathcal D_K|,\qquad \bigcup_{k=1}^K\mathcal D_k=\mathcal D,\qquad \mathcal D_i\cap\mathcal D_j=\varnothing\ (i\ne j)
$$

接著對 $k=1,\dots,K$ 各做一輪：把 $\mathcal D_k$ 留出來當驗證，用其餘 $K-1$ 個 fold 訓練，在 $\mathcal D_k$ 上量誤差 $\mathbf{Err}_k$。最後把 $K$ 個誤差平均：

Then for each $k=1,\dots,K$ do one round: hold out $\mathcal D_k$ for validation, train on the other $K-1$ folds and measure the error $\mathbf{Err}_k$ on $\mathcal D_k$. Finally average the $K$ errors:

$$
\mathbf{Err}=\frac1K\sum_{k=1}^{K}\mathbf{Err}_k
$$

每一輪裡，被留出的 fold 沒參與那一輪的訓練，所以 KP04 的獨立性成立，每個 $\mathbf{Err}_k$ 都是誠實的誤差估計。來源把留出的 fold 叫做測試集 $\mathcal R$，本講義改叫「留出的 fold」，避免和 KP01 那份全程不能碰的最後測試集混在一起（見「來源更正與取捨」）。

In each round the held-out fold took no part in that round's training, so the independence of KP04 holds and every $\mathbf{Err}_k$ is an honest error estimate. The source calls the held-out fold the test set $\mathcal R$; this note calls it the "held-out fold" so that it is not confused with the final test set of KP01 that must never be touched (see "Corrections to the source").

用 100 筆資料、$K=5$ 算一次。洗牌後切成 5 個 fold，每個 20 筆；固定的洗牌種子下，2 次多項式的五個 fold 誤差是（程式計算）：

Take 100 points and $K=5$. Shuffle and cut into 5 folds of 20 points; with a fixed shuffling seed the five fold errors of the degree-2 polynomial are (computed by program):

| fold $k$ | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| $E_k$ | 0.1153 | 0.1008 | 0.0642 | 0.1090 | 0.1218 |

| fold $k$ | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| $E_k$ | 0.1153 | 0.1008 | 0.0642 | 0.1090 | 0.1218 |

五個誤差加起來是 $0.5111$，平均 $\mathrm{CV}=0.5111/5=0.1022$。注意單一個 fold 的誤差很不穩：最小的 $0.0642$ 低於雜訊地板 $0.09$，最大的 $0.1218$ 是最小的約 $1.9$ 倍，這正是第 1 節的運氣；平均之後的 $0.1022$ 才是比較可靠的估計。

The five errors sum to $0.5111$ and the mean is $\mathrm{CV}=0.5111/5=0.1022$. Notice that a single fold's error is unstable: the smallest, $0.0642$, is below the noise floor $0.09$ and the largest, $0.1218$, is $1.9$ times the smallest; that is the luck of Section 1. The average $0.1022$ is the more reliable estimate.

![5-fold 交叉驗證的輪替（上，$K=5$，每個 fold 20 筆）：第 $k$ 輪把第 $k$ 個 fold（紅）留出來當驗證，其餘四個 fold（藍）訓練，最後得到 $E_1,\dots,E_5$；留一法 LOOCV（下，$K=n=100$，用 20 個格子示意）：每輪只留 1 筆，要訓練 100 次，每次用 99 筆。（The rotation of 5-fold cross-validation (top, $K=5$, 20 points per fold): round $k$ holds out the $k$-th fold (red) for validation and trains on the other four folds (blue), ending with $E_1,\dots,E_5$; leave-one-out LOOCV (bottom, $K=n=100$, drawn with 20 blocks): each round holds out one point, so 100 trainings of 99 points each are needed.）](../assets/figures/lec04-kp06-folds.png)

**圖說 `lec04-kp06-folds`**：先看上半部紅色方塊排成一條對角線：每一輪只有一個 fold 是紅的（留出），其餘都是藍的（訓練），五輪下來每個 fold 恰好被留出一次，每筆資料恰好被驗證一次、被訓練四次。再看下半部：LOOCV 的紅色格子只有一小格，代表每一輪只留 1 筆，訓練集幾乎是全部資料（99 筆），但要做 100 輪。這是 `[補充]` 的示意圖，格子數是示意，不等於實際筆數。〔功能：1、3〕

**Caption `lec04-kp06-folds`**: First look at the red blocks of the upper part forming a diagonal: in each round only one fold is red (held out) and the others are blue (training); over the five rounds every fold is held out exactly once, so every point is validated exactly once and trained on four times. Then the lower part: the red block of LOOCV is a single small cell, one point held out per round, with a training set of almost all the data (99 points) but 100 rounds to run. This is an `[Added]` schematic and the number of blocks is symbolic, not the actual number of points. [Function: 1, 3]

### 3. 用交叉驗證選次數（Choosing the degree with cross-validation）

把 K-fold 用在第 1 節的問題：對 1 到 8 次的每一個次數，各算一次 5-fold 的 CV 誤差，選最小的（程式計算，右欄是留一法 LOOCV，見下一節）：

Apply K-fold to the problem of Section 1: for every degree from 1 to 8 compute the 5-fold CV error and pick the smallest (computed by program; the right column is leave-one-out LOOCV, see the next section):

| 次數 $d$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 5-fold CV | 0.3136 | **0.1022** | 0.1038 | 0.1063 | 0.1076 | 0.1101 | 0.1068 | 0.1084 |
| LOOCV | 0.3049 | **0.1026** | 0.1052 | 0.1077 | 0.1094 | 0.1124 | 0.1112 | 0.1136 |

| Degree $d$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 5-fold CV | 0.3136 | **0.1022** | 0.1038 | 0.1063 | 0.1076 | 0.1101 | 0.1068 | 0.1084 |
| LOOCV | 0.3049 | **0.1026** | 0.1052 | 0.1077 | 0.1094 | 0.1124 | 0.1112 | 0.1136 |

5-fold CV 與 LOOCV 都選 2 次。對同一批資料重新洗牌 300 次、每次都重做 5-fold CV，有 299 次選 2 次，只有 1 次選 7 次；對照第 1 節的單次 80/20 切分只有 158 次選對，交叉驗證把「切法的運氣」平均掉了。同一個模型（2 次）的 CV 誤差，在 300 次不同洗牌下的標準差只有 $0.0030$，而單次切分的驗證誤差標準差是 $0.0279$，約小了 $9$ 倍（程式計算）。

Both 5-fold CV and LOOCV pick degree 2. Reshuffling the same data 300 times and redoing the 5-fold CV each time, 299 runs pick degree 2 and only 1 picks degree 7; compare the 158 correct choices of the single 80/20 splits in Section 1: cross-validation averages out the luck of the split. For the same model (degree 2) the standard deviation of the CV error over the 300 shuffles is only $0.0030$, against $0.0279$ for the validation error of a single split, about $9$ times smaller (computed by program).

![5-fold 的 CV 誤差對多項式次數作圖（縱軸為對數刻度）：灰點是每個次數的五個 fold 誤差 $E_k$，藍線是它們的平均（CV 誤差），紅圈標出最小值在 2 次，綠色虛線是雜訊地板 $\sigma^2=0.09$。（The 5-fold CV error against the polynomial degree (logarithmic vertical axis): the grey dots are the five fold errors $E_k$ of each degree, the blue line is their mean (the CV error), the red circle marks the minimum at degree 2 and the green dashed line is the noise floor $\sigma^2=0.09$.）](../assets/figures/lec04-kp06-cv-curve.png)

**圖說 `lec04-kp06-cv-curve`**：先看每個次數上的灰點：五個點的高低就是第 2 節那種 fold 之間的差異（2 次時是 $0.064$ 到 $0.122$），散得比相鄰次數的平均差距還開。再看藍線：1 次最高（$0.314$），2 次最低（$0.102$），之後只緩慢上升到約 $0.11$。回升很溫和，是因為這裡有 100 筆資料，高次多項式的變異被壓住了，不像 KP05 只有 12 點時 9 次暴增到 $0.289$；資料越少，回升越陡。〔功能：1、2〕

**Caption `lec04-kp06-cv-curve`**: First look at the grey dots at each degree: the spread of the five dots is the fold-to-fold difference of Section 2 (at degree 2 from $0.064$ to $0.122$), wider than the gap between the means of neighbouring degrees. Then the blue line: it is highest at degree 1 ($0.314$), lowest at degree 2 ($0.102$) and then rises only slowly to about $0.11$. The rise is gentle because there are 100 points and the variance of the high-degree polynomials is held down, unlike the 12 points of KP05 where degree 9 jumped to $0.289$; the fewer the data, the steeper the rise. [Function: 1, 2]

選定次數之後還有一步：**用全部 100 筆資料重新訓練**一次這個次數的模型，當作最後的模型。交叉驗證的 $K$ 個模型只是用來「量」，每個都只看到 $(K-1)/K$ 的資料，不是要交付的模型。

One step remains after the degree is chosen: **retrain the model of that degree on all 100 points** and use it as the final model. The $K$ models of cross-validation are only for "measuring", each sees only $(K-1)/K$ of the data, and none of them is the model to deliver.

### 4. LOOCV、$K$ 的取捨，與交叉驗證的界線（LOOCV, the choice of $K$ and the limits）

當資料很少時，可以讓每一筆輪流當唯一的驗證點，其餘 $n-1$ 筆訓練，這是**留一法**（leave-one-out cross-validation, LOOCV），就是 $K=n$：

When the data is very scarce, every point can take a turn as the only validation point with the other $n-1$ points for training; this is **leave-one-out cross-validation (LOOCV)**, the case $K=n$:

$$
\mathbf{Err}=\frac1n\sum_{i=1}^{n}\mathbf{Err}_i
$$

它的優點是幾乎不浪費資料（每次訓練 $n-1=99$ 筆，5-fold 是 80 筆），缺點是要訓練 $n$ 次。以 2 次多項式驗算：實際訓練 100 次、每次留一點，LOOCV 誤差是 $0.1026$（程式計算）。對最小平方這類簡單模型另有只配一次就能算出 LOOCV 的捷徑，但神經網路沒有，這裡不介紹。

Its advantage is that it hardly wastes data (each training uses $n-1=99$ points, against 80 for 5-fold) and its disadvantage is $n$ trainings. Check with the degree-2 polynomial: 100 actual trainings, each leaving one point out, give an LOOCV error of $0.1026$ (computed by program). For simple models such as least squares there is a shortcut that computes the LOOCV from a single fit, but neural networks have none, so it is not covered here.

那 $K$ 越大越好嗎？不一定。用 300 組各自抽出的 100 筆資料，對 2 次多項式比較不同的 $K$（程式計算）。目標是「用全部 100 筆配出來的模型的真正測試誤差」（平均 $0.0926$）：

Is a larger $K$ always better? Not necessarily. Compare different $K$ for the degree-2 polynomial on 300 independently drawn data sets of 100 points (computed by program). The target is the true test error of the model fitted on all 100 points (mean $0.0926$):

| $K$ | 訓練次數 | 估計值的平均 | 平均偏高（估計 $-$ 目標） | 估計值的標準差 |
|---|---|---|---|---|
| 2 | 2 | 0.0970 | +0.0044 | 0.0140 |
| 5 | 5 | 0.0952 | +0.0026 | 0.0128 |
| 10 | 10 | 0.0947 | +0.0022 | 0.0125 |
| 20 | 20 | 0.0946 | +0.0021 | 0.0125 |
| 100（LOOCV） | 100 | 0.0946 | +0.0020 | 0.0125 |

| $K$ | Trainings | Mean estimate | Mean excess (estimate $-$ target) | Std of the estimate |
|---|---|---|---|---|
| 2 | 2 | 0.0970 | +0.0044 | 0.0140 |
| 5 | 5 | 0.0952 | +0.0026 | 0.0128 |
| 10 | 10 | 0.0947 | +0.0022 | 0.0125 |
| 20 | 20 | 0.0946 | +0.0021 | 0.0125 |
| 100 (LOOCV) | 100 | 0.0946 | +0.0020 | 0.0125 |

兩件事。第一，$K$ 越大估計**偏高越少**：每輪的訓練集越接近全部資料，量到的就越接近全資料模型的誤差；$K=2$ 時每輪只訓練 50 筆，偏高 $0.0044$，是 LOOCV 的 $2.2$ 倍。第二，標準差在 $K=10$ 以後就不再下降（$0.0125$），LOOCV 並沒有比 10-fold 穩，卻要訓練 10 倍的次數（在沒有捷徑的模型上）。這個結果只代表這個設定（平滑的線性模型、$n=100$）；來源的教授祕笈也說 LOOCV 的訓練集彼此幾乎一樣、模型高度相關，估計的變異不一定比較小。所以實務上常用 $K=5$ 或 $10$，不必追求 LOOCV。

Two things. First, a larger $K$ makes the estimate **less too high**: the closer the training set of each round is to all the data, the closer the measured error is to that of the full-data model; at $K=2$ each round trains on only 50 points and the excess is $0.0044$, $2.2$ times that of LOOCV. Second, the standard deviation stops falling after $K=10$ ($0.0125$); LOOCV is no steadier than 10-fold yet needs 10 times as many trainings (for models without the shortcut). This result is for this setting only (a smooth linear model, $n=100$); the professor's guide of the source also says that the training sets of LOOCV are almost identical, the models are highly correlated and the variance of the estimate is not necessarily smaller. So in practice $K=5$ or $10$ is common and there is no need to chase LOOCV.

最後說交叉驗證的界線。交叉驗證用來做兩件事：（I）決定模型的複雜度，（II）調整超參數（例如後面 KP07 起的 $\alpha$）。但它**不能取代最後的測試集**：用交叉驗證挑了次數、調了超參數，這些資料已經參與了選擇，所以最後的誤差報告還是要靠一份從頭到尾沒碰過的測試集，否則就是 KP01 的作弊 #2。另外，任何用資料算出來的前處理（例如標準化的平均與標準差）必須在每一輪裡只用訓練的 $K-1$ 個 fold 重新算，不能先用全部資料算好再切 `[補充]`，否則留出的 fold 的資訊已經漏進訓練。

Finally the limits of cross-validation. It does two things: (I) decide the model complexity and (II) tune hyperparameters (for example the $\alpha$ from KP07 on). But it **cannot replace the final test set**: once cross-validation has chosen the degree and tuned the hyperparameters, those data have taken part in the selection, so the final error report still needs a test set that was never touched from start to finish, or it is cheating #2 of KP01. Moreover, any preprocessing computed from the data (for example the mean and standard deviation of standardization) must be recomputed in every round from the $K-1$ training folds only, and not computed on all the data before the split `[Added]`, otherwise information of the held-out fold has already leaked into training.

下一個問題：交叉驗證幫我們從幾個候選模型中挑出複雜度合適的那一個；可是能不能在訓練的時候，就直接把模型的複雜度壓住？這是 KP07 的主題。

The next question: cross-validation helps us pick, among a few candidates, the model of suitable complexity; but can we press the complexity of the model down directly while training? That is the topic of KP07.

### 本單元重點整理（KP06 summary）

**一句話重點**：把資料切成 $K$ 個互不相交的 fold，輪流留出一個驗證、用其餘訓練，再把 $K$ 個誤差平均；$K=n$ 是 LOOCV；它平均掉單次切分的運氣，用來選複雜度與超參數，但不能取代最後的測試集。

**One-sentence summary**: Cut the data into $K$ disjoint folds, hold one out for validation in turn and train on the rest, then average the $K$ errors; $K=n$ is LOOCV; it averages out the luck of a single split and is used to choose the complexity and the hyperparameters, but it cannot replace the final test set.

**白話說明**：像循環賽：每支隊伍輪流當一次主場（被驗證），其他隊伍當客場（訓練），排名看所有場次的平均，不看單一場。這樣最公平，也最不浪費資料。

**Plain explanation**: Like a round-robin tournament: every team takes a turn at home (being validated) while the others play away (training), and the ranking looks at the average over all games, not a single one. This is the fairest and wastes the least data.

**正式定義**：$\mathcal D_i\cap\mathcal D_j=\varnothing$，$\bigcup_k\mathcal D_k=\mathcal D$；$\mathbf{Err}_k$ 是用 $\mathcal D\setminus\mathcal D_k$ 訓練、在 $\mathcal D_k$ 上量到的誤差；$\mathbf{Err}=\frac1K\sum_{k=1}^K\mathbf{Err}_k$；LOOCV 取 $K=n$。

**Formal definition**: $\mathcal D_i\cap\mathcal D_j=\varnothing$ and $\bigcup_k\mathcal D_k=\mathcal D$; $\mathbf{Err}_k$ is the error on $\mathcal D_k$ of a model trained on $\mathcal D\setminus\mathcal D_k$; $\mathbf{Err}=\frac1K\sum_{k=1}^K\mathbf{Err}_k$; LOOCV takes $K=n$.

**例子**：$n=100$、$K=5$：每個 fold 20 筆，2 次多項式的五個誤差 $0.1153,\,0.1008,\,0.0642,\,0.1090,\,0.1218$，平均 $0.1022$；LOOCV 要訓練 100 次、每次 99 筆，得 $0.1026$。逐步計算見第 2–4 節。

**Example**: $n=100$, $K=5$: 20 points per fold and the five errors of the degree-2 polynomial $0.1153,\,0.1008,\,0.0642,\,0.1090,\,0.1218$ average to $0.1022$; LOOCV needs 100 trainings of 99 points each and gives $0.1026$. The computations are in Sections 2–4.

**常見迷思**：

- 「交叉驗證可以取代最後的測試集」——不行，它已經參與了選擇，最後還要一份沒碰過的測試集。
- 「LOOCV 一定比 5-fold 準」——資料夠多時兩者接近，LOOCV 還貴得多。
- 「$K$ 越大越好」——偏高的幅度隨 $K$ 減小，但標準差不再下降，成本線性增加。
- 「fold 之間可以重疊」——互斥是基本要求，否則同一筆資料同時在訓練與驗證。
- 「選完次數就用交叉驗證裡的模型」——要用全部資料重新訓練。

**Common misconceptions**:

- "Cross-validation can replace the final test set." No: it has taken part in the selection, and a never-touched test set is still needed at the end.
- "LOOCV is always more accurate than 5-fold." With enough data the two are close, and LOOCV is far more expensive.
- "A larger $K$ is always better." The excess shrinks with $K$ but the standard deviation stops falling and the cost grows linearly.
- "The folds may overlap." Disjointness is a basic requirement, or the same point is in training and in validation at once.
- "After choosing the degree, use a model from the cross-validation." Retrain on all the data.

**圖**：本單元兩張圖：`lec04-kp06-folds`（5-fold 輪替與 LOOCV，功能 1、3）與 `lec04-kp06-cv-curve`（CV 誤差對次數，功能 1、2），以 `assets/figures/src/lec04-kp06-figures.py` 繪製；腳本印出本講義引用的每一個數字。

**Figure**: Two figures in this unit: `lec04-kp06-folds` (the 5-fold rotation and LOOCV; functions 1, 3) and `lec04-kp06-cv-curve` (CV error against the degree; functions 1, 2), drawn by `assets/figures/src/lec04-kp06-figures.py`; the script prints every number quoted in this note.

**想深入**（選讀）：K-fold 的 $K$ 個訓練集彼此重疊很多（每兩個共用 $K-2$ 個 fold），所以 $K$ 個誤差 $E_k$ **不是獨立的**，平均的標準誤差不能直接用 $\sigma/\sqrt K$ 估 `[補充]`，這也是為什麼不能把五個 fold 誤差當五次獨立重複來做統計檢定。另外，交叉驗證估計的是「用 $(K-1)/K$ 的資料訓練的模型」的誤差，不是全資料模型的誤差，這就是上表偏高的來源；資料很少、學習曲線還很陡的時候，這個偏高會比較明顯。

**Going deeper** (optional): The $K$ training sets of K-fold overlap a lot (any two share $K-2$ folds), so the $K$ errors $E_k$ are **not independent** and the standard error of their mean cannot be estimated directly by $\sigma/\sqrt K$ `[Added]`; this is also why the five fold errors must not be treated as five independent repetitions in a significance test. Moreover, cross-validation estimates the error of "a model trained on $(K-1)/K$ of the data", not that of the full-data model, which is the source of the excess in the table above; when the data is very scarce and the learning curve is still steep, the excess is more visible.

**小練習**：

1. $n=1000$、$K=10$：每個 fold 幾筆？每一輪用幾筆訓練？總共要訓練幾次？LOOCV 呢？（提示：$n/K$、$n-n/K$。）
2. 4-fold 的 fold 誤差是 $0.20,\,0.16,\,0.24,\,0.20$：CV 誤差是多少？有沒有哪一個 fold 的誤差離平均特別遠？（提示：平均與每個誤差的差。）
3. 同樣 $n=100$：比較 5-fold 與 10-fold 每一輪的訓練筆數與總訓練次數，說明哪一個的估計偏高比較少、為什麼。（提示：訓練集越接近全部資料，量到的越接近全資料模型的誤差。）

**Practice**:

1. $n=1000$, $K=10$: how many points per fold? How many points does each round train on? How many trainings in all? And for LOOCV? (Hint: $n/K$ and $n-n/K$.)
2. The fold errors of a 4-fold CV are $0.20,\,0.16,\,0.24,\,0.20$: what is the CV error? Is any fold's error unusually far from the mean? (Hint: the difference between the mean and each error.)
3. With the same $n=100$: compare the training size per round and the total number of trainings for 5-fold and 10-fold, and say which one has less excess and why. (Hint: the closer the training set is to all the data, the closer the measured error is to that of the full-data model.)

**解題示範**（以 $n=500$、$K=5$ 示範第 1 題的算法）：

1. 每個 fold：$500/5=100$ 筆。
2. 每輪訓練筆數：$500-100=400$ 筆，驗證 100 筆。
3. 總訓練次數：5 次（每個 fold 當一次驗證）。
4. LOOCV：$K=500$，要訓練 500 次，每次 499 筆。
5. 檢查：5 個 fold 共 $5\cdot100=500$，剛好是全部資料各驗證一次、訓練四次。

**Worked solution** (the method of exercise 1, shown with $n=500$ and $K=5$):

1. Points per fold: $500/5=100$.
2. Training points per round: $500-100=400$, with 100 for validation.
3. Total trainings: 5 (each fold validates once).
4. LOOCV: $K=500$, so 500 trainings of 499 points each.
5. Check: the 5 folds hold $5\cdot100=500$ points, exactly all the data, each validated once and trained on four times.

### 來源更正與取捨（Corrections to the source）

- 來源把每一輪留出的 fold 叫做測試集 $\mathcal R$，同時又說「驗證集與測試集必須分開，否則落入作弊 #2」。這兩句放在一起容易混淆：留出的 fold 在這一輪扮演驗證，真正的測試集是從頭到尾不碰的那一份。本講義改稱「留出的 fold」，並說明交叉驗證不能取代最後的測試集。
- 來源說交叉驗證「降低切分造成的變異」，這是對的，本講義補上數字：同一個 2 次模型，單次 80/20 切分的驗證誤差標準差 $0.0279$，5-fold CV 誤差在不同洗牌下的標準差 $0.0030$（程式計算）。
- 來源與教授祕笈關於 $K$ 與變異的說法不一致：祕笈說 LOOCV 的估計變異「不一定小」，來源的互動示範卻寫「$K$ 越小或 LOOCV 時，每次估計的變異較大」。本講義以程式計算的結果為準：在這個設定下 $K=2$ 的標準差最大（$0.0140$），LOOCV 與 10-fold 相同（$0.0125$），沒有證據說 LOOCV 比較穩；結論僅限這個設定，不推廣。
- 來源說 $K=10$「最常用」，祕笈說常用 5 或 10；本講義不指定唯一值，兩者都常見。
- 來源的式 (32)–(34) 編號不沿用；fold 大小「約相等」是因為 $n$ 不一定被 $K$ 整除，差最多 1 筆。

- The source calls the fold held out in each round the test set $\mathcal R$ and also says that "the validation set and the test set must be kept apart or it falls into cheating #2". Together these are easy to confuse: the held-out fold plays the validation role in this round, and the real test set is the one never touched from start to finish. This note calls it the "held-out fold" and says that cross-validation cannot replace the final test set.
- The source says cross-validation "lowers the variation caused by the split"; that is correct and this note adds numbers: for the same degree-2 model the validation error of a single 80/20 split has a standard deviation of $0.0279$, and the 5-fold CV error has $0.0030$ over different shuffles (computed by program).
- The statements of the source and of its professor's guide about $K$ and variance disagree: the guide says the variance of the LOOCV estimate is "not necessarily smaller", while the interactive demo writes "a smaller $K$ or LOOCV gives a larger variation per estimate". This note goes by the computed result: in this setting $K=2$ has the largest standard deviation ($0.0140$) and LOOCV equals 10-fold ($0.0125$), so there is no evidence that LOOCV is steadier; the conclusion is limited to this setting and not generalized.
- The source says $K=10$ is "the most common" and the guide says 5 or 10 are common; this note names no single value, both are common.
- The equation numbers (32)–(34) of the source are not reused; the folds have "about equal" size because $n$ need not be divisible by $K$, and the sizes differ by at most one point.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 交叉驗證 | cross-validation |
| $K$ 折交叉驗證 | $K$-fold cross-validation |
| 留一法 | leave-one-out cross-validation (LOOCV) |
| fold（折） | fold |
| 留出的 fold | held-out fold |
| 互不相交 | disjoint |
| 超參數 | hyperparameter |
| 洗牌 | shuffle |
| 資料洩漏 | data leakage |
