# 第2週 KP03：標籤改成 ±1，更新規則縮成「猜錯時加 2ηyx」（With ±1 labels the update shrinks to "add 2ηyx when wrong"）

> 程度：大學生（undergrad）／Level: Undergraduate (undergrad)
>
> 週次：第2週（2026-09-17）｜主題：神經網路基礎：感知器與多層感知器
>
> Week 2 (2026-09-17) | Topic: Neural network fundamentals: perceptron and multilayer perceptron

本講義是第2章的第三個知識點（KP03），承接 KP02 留下的問題：用 0/1 標籤時誤差有三種情形，改用 $+1$ 與 $-1$ 後，規則能不能更短，還看得出幾何意義？第1章 KP08 直接寫出了 $\pm1$ 形式的更新；這一節**推導**它，用代數與幾何兩條路。先備知識是 KP02、向量內積與一點點偏導數。內容依第2章主文重新講解，來源讀取日期為 2026-10-14；新增的釐清標為 `[補充]`。

This is the third knowledge point (KP03) of Chapter 2. It picks up the question KP02 left open: with 0/1 labels the error has three cases, so after switching to $+1$ and $-1$, can the rule be shorter and reveal a geometric meaning? Chapter 1 KP08 simply wrote the update in $\pm1$ form; this note **derives** it, by an algebraic route and a geometric route. It assumes KP02, vector inner products and a little partial differentiation. It re-explains the main text of Chapter 2, accessed on 2026-10-14; clarifications that are new are marked `[Added]`.

[下載本單元投影片](../slides/lec02-kp03-signum-update-derivations.jbd.json)

## KP03 標籤改成 ±1，更新規則縮成「猜錯時加 2ηyx」（With ±1 labels the update shrinks to "add 2ηyx when wrong"）

**建議時間 / Time**：15 分鐘　**對應實作 / Related practice**：用代數與幾何兩種方式推出同一條更新式，並驗算 OR 的軌跡。

### 學習目標（Learning objectives）

讀完之後，你能列出 $y,o\in\{\pm1\}$ 的四種組合並化簡出 $2\eta y_i\mathbf{x}_i$；能說明 $\mathbf{w}$ 是決策邊界的法向量、$d=y(\mathbf{w}^\top\mathbf{x}+w_0)/\|\mathbf{w}\|$ 的正負代表對錯；能由「只看錯分點」的損失做梯度下降得到同一條規則；能驗算 $\eta=\tfrac12$ 時軌跡與 KP02 相同。

After reading, you can list the four combinations of $y,o\in\{\pm1\}$ and reduce to $2\eta y_i\mathbf{x}_i$; explain that $\mathbf{w}$ is the normal vector of the decision boundary and that the sign of $d=y(\mathbf{w}^\top\mathbf{x}+w_0)/\|\mathbf{w}\|$ says right or wrong; get the same rule by gradient descent on a loss over the wrongly classified points; and check that with $\eta=\tfrac12$ the trace equals that of KP02.

### 符號表（Symbols）

| 符號 | 中文意義 | English meaning |
|---|---|---|
| $y_i,\ o_i\in\{+1,-1\}$ | 正確標籤與符號函數的輸出 | correct label and output of the sign function |
| $\operatorname{sign}(s)$ | $s\ge0$ 取 $+1$，否則 $-1$ | $+1$ when $s\ge0$, otherwise $-1$ |
| $\mathbf{w}$ | 權重向量（幾何段不含 $w_0$） | weight vector (excluding $w_0$ in the geometric part) |
| $d$ | 帶符號距離 $y(\mathbf{w}^\top\mathbf{x}+w_0)/\|\mathbf{w}\|$ | signed distance $y(\mathbf{w}^\top\mathbf{x}+w_0)/\|\mathbf{w}\|$ |
| $M$ | 被猜錯的點的集合 | the set of wrongly classified points |
| $E$ | 只看錯分點的誤差 $-\sum_{i\in M}y_i(\mathbf{w}^\top\mathbf{x}_i+w_0)$ | the error over wrong points only |

### 1. 代數推導：四種組合，猜錯的兩種化簡成同一式（Algebraic route）

激活換成符號函數、標籤也寫成 $\pm1$，更新式仍是 $\mathbf{w}\leftarrow\mathbf{w}+\eta(y-o)\mathbf{x}$，但 $y,o$ 只有四種組合：

With a sign activation and labels written as $\pm1$, the update is still $\mathbf{w}\leftarrow\mathbf{w}+\eta(y-o)\mathbf{x}$, but $y,o$ have only four combinations:

| $y$ | $o$ | $y-o$ | $\eta(y-o)\mathbf{x}$ | 結果 |
|---|---|---|---|---|
| $+1$ | $+1$ | $0$ | $\mathbf{0}$ | 不更新 |
| $+1$ | $-1$ | $+2$ | $2\eta\mathbf{x}=2\eta y\mathbf{x}$ | 更新 |
| $-1$ | $+1$ | $-2$ | $-2\eta\mathbf{x}=2\eta y\mathbf{x}$ | 更新 |
| $-1$ | $-1$ | $0$ | $\mathbf{0}$ | 不更新 |

兩個「猜錯」的列都等於 $2\eta y\mathbf{x}$，所以

Both "wrong" rows equal $2\eta y\mathbf{x}$, so

$$
\mathbf{w}\leftarrow\mathbf{w}+2\eta\,y_i\,\mathbf{x}_i\quad\text{只在 }o_i\ne y_i\text{ 時（only when }o_i\neq y_i\text{）}
$$

係數 2 沒有什麼神祕，只是 $y-o$ 在兩者反號時必然是 $\pm2$。偏置已併入 $\mathbf{w}$，所以 $w_0\leftarrow w_0+2\eta y_i$ 同時發生。第1章 KP08 寫的 $\mathbf{w}\leftarrow\mathbf{w}+\eta y\mathbf{x}$ 就是把這個 2 併進學習率。

The factor 2 is no magic; it is simply what $y-o$ equals when $\pm1$ values disagree. The bias is already inside $\mathbf{w}$, so $w_0\leftarrow w_0+2\eta y_i$ happens at the same time. The $\mathbf{w}\leftarrow\mathbf{w}+\eta y\mathbf{x}$ of Chapter 1 KP08 is this rule with the 2 absorbed into the learning rate.

### 2. 幾何推導：法向量、距離、只看錯分點的誤差（Geometric route）

決策邊界是 $\mathbf{w}^\top\mathbf{x}+w_0=0$。邊界上兩點 $\mathbf{x}_1,\mathbf{x}_2$ 都滿足它，相減得 $\mathbf{w}^\top(\mathbf{x}_1-\mathbf{x}_2)=0$；$\mathbf{x}_1-\mathbf{x}_2$ 沿著邊界，所以 $\mathbf{w}$ 與邊界垂直，是**法向量**。任一點到邊界的帶符號距離是 $(\mathbf{w}^\top\mathbf{x}+w_0)/\|\mathbf{w}\|$，乘上標籤 $y$ 得 $d$：猜對時 $d\ge0$，猜錯時 $d<0$。用 $y$ 乘上去，就不必用不可微的絕對值。

The decision boundary is $\mathbf{w}^\top\mathbf{x}+w_0=0$. Two points $\mathbf{x}_1,\mathbf{x}_2$ on it both satisfy this, and subtracting gives $\mathbf{w}^\top(\mathbf{x}_1-\mathbf{x}_2)=0$; the vector $\mathbf{x}_1-\mathbf{x}_2$ runs along the boundary, so $\mathbf{w}$ is perpendicular to it: it is the **normal vector**. The signed distance of any point is $(\mathbf{w}^\top\mathbf{x}+w_0)/\|\mathbf{w}\|$, and multiplying by the label $y$ gives $d$: $d\ge0$ when right and $d<0$ when wrong. Multiplying by $y$ avoids the non-differentiable absolute value.

要讓錯分點變對，就是把它們的 $d$ 變大。取被猜錯的點集合 $M$，定義誤差 $E=-\sum_{i\in M}y_i(\mathbf{w}^\top\mathbf{x}_i+w_0)$，有錯時它為正、越小越好。梯度是 $\partial E/\partial\mathbf{w}=-\sum_{i\in M}y_i\mathbf{x}_i$，梯度下降得

To turn wrong points right is to push their $d$ up. Take the set $M$ of wrongly classified points and define $E=-\sum_{i\in M}y_i(\mathbf{w}^\top\mathbf{x}_i+w_0)$, positive when anything is wrong and best when small. The gradient is $\partial E/\partial\mathbf{w}=-\sum_{i\in M}y_i\mathbf{x}_i$, and gradient descent gives

$$
\mathbf{w}\leftarrow\mathbf{w}+\eta'\sum_{i\in M}y_i\mathbf{x}_i
$$

一次只看一個錯點，就是 $\mathbf{w}\leftarrow\mathbf{w}+\eta'y_i\mathbf{x}_i$，與代數結果只差一個係數（$\eta'=2\eta$），併進學習率即可。兩條路殊途同歸。

Looking at one wrong point at a time gives $\mathbf{w}\leftarrow\mathbf{w}+\eta'y_i\mathbf{x}_i$, which differs from the algebraic result only by a constant ($\eta'=2\eta$) absorbed into the learning rate. The two routes arrive at the same place.

![法向量 w、邊界與四個點的帶符號距離 d](../assets/figures/lec02-kp03-signum-geometry.png)

**圖說 `lec02-kp03-signum-geometry`**：先看斜線 $x_1+x_2=1.5$；再看從線上伸出的箭頭，它垂直於線（直角符號），方向是 $\mathbf{w}=(1,1)$；最後看四個點旁的 $d$，兩個紅圈的點 $d=-0.35$，在錯的一側。 / **Caption `lec02-kp03-signum-geometry`:** Look first at the line $x_1+x_2=1.5$; then at the arrow leaving the line, perpendicular to it (right-angle mark) and pointing along $\mathbf{w}=(1,1)$; finally at the $d$ beside the four points, where the two red-ringed points have $d=-0.35$ and lie on the wrong side.

### 3. 數字驗算（Numeric checks）

**距離。** 取 $\mathbf{w}=(1,1)$、$w_0=-1.5$，$\|\mathbf{w}\|=\sqrt2\approx1.414$，OR 的標籤 $(0,0)\to-1$、其餘 $+1$。$(0,0)$：$s=-1.5$，$d=(-1)(-1.5)/1.414=1.06$；$(0,1)$：$s=-0.5$，$d=-0.5/1.414=-0.35$，錯；$(1,0)$ 同樣 $-0.35$，錯；$(1,1)$：$s=0.5$，$d=0.35$，對。

**Distances.** Take $\mathbf{w}=(1,1)$ and $w_0=-1.5$, so $\|\mathbf{w}\|=\sqrt2\approx1.414$, with OR labels $(0,0)\to-1$ and $+1$ otherwise. $(0,0)$: $s=-1.5$ and $d=(-1)(-1.5)/1.414=1.06$; $(0,1)$: $s=-0.5$ and $d=-0.5/1.414=-0.35$, wrong; $(1,0)$ the same $-0.35$, wrong; $(1,1)$: $s=0.5$ and $d=0.35$, right.

**一次更新。** 拿錯點 $(0,1)$（$y=+1$），$\eta=\tfrac12$，更新得 $\mathbf{w}=(1,2)$、$w_0=-0.5$，新的 $s=0\cdot1+1\cdot2-0.5=1.5>0$，這一點對了。$s$ 增加 $1.5-(-0.5)=2$，正好等於 $2\eta(\|\mathbf{x}\|^2+1)=2\cdot\tfrac12\cdot(1+1)$。

**One update.** Take the wrong point $(0,1)$ ($y=+1$) with $\eta=\tfrac12$; the update gives $\mathbf{w}=(1,2)$ and $w_0=-0.5$, and the new $s=0\cdot1+1\cdot2-0.5=1.5>0$, so this point is right. The sum rose by $1.5-(-0.5)=2$, exactly $2\eta(\|\mathbf{x}\|^2+1)=2\cdot\tfrac12\cdot(1+1)$.

**軌跡與 KP02 相同。** 用 $\eta=\tfrac12$（$2\eta=1$，更新就是 $\mathbf{w}\leftarrow\mathbf{w}+y\mathbf{x}$）、從全 0 學 OR，5 次更新依序得 $(0,0,-1)$、$(0,1,0)$、$(0,1,-1)$、$(1,1,0)$、$(1,1,-1)$，第 4 輪全對〔程式計算〕。與 KP02 的二元版**逐項相同**，證明兩種寫法只是記號不同。

**The trace equals KP02.** With $\eta=\tfrac12$ ($2\eta=1$, so the update is $\mathbf{w}\leftarrow\mathbf{w}+y\mathbf{x}$) and learning OR from all zeros, the 5 updates give $(0,0,-1)$, $(0,1,0)$, $(0,1,-1)$, $(1,1,0)$ and $(1,1,-1)$ in turn, and pass 4 is all right (computed by program). They are **identical term by term** to the binary version of KP02, showing that the two forms differ only in notation.

### 本單元重點整理（KP03 summary）

**一句話重點**：用 $\pm1$ 標籤，代數（四種組合）與幾何（法向量與距離的梯度下降）兩條路都得到「只在猜錯時 $\mathbf{w}\leftarrow\mathbf{w}+2\eta y_i\mathbf{x}_i$」。

**One-sentence summary**: With $\pm1$ labels, both the algebraic route (four combinations) and the geometric route (gradient descent on normal vector and distance) give "only when wrong, $\mathbf{w}\leftarrow\mathbf{w}+2\eta y_i\mathbf{x}_i$".

**白話說明**：猜對與猜錯變成「同號或異號」。猜錯的點被往它該在的那一側拉：標籤是 $+1$ 就把 $\mathbf{x}$ 加進權重，是 $-1$ 就減掉。

**Plain explanation**: Right and wrong become "same sign or opposite sign". A wrong point is pulled toward the side it belongs to: add $\mathbf{x}$ to the weights if the label is $+1$ and subtract it if $-1$.

**正式定義**：$\mathbf{w}\leftarrow\mathbf{w}+2\eta y_i\mathbf{x}_i$（只在 $o_i\ne y_i$）；$d=y(\mathbf{w}^\top\mathbf{x}+w_0)/\|\mathbf{w}\|$；$E=-\sum_{i\in M}y_i(\mathbf{w}^\top\mathbf{x}_i+w_0)$，$\partial E/\partial\mathbf{w}=-\sum_{i\in M}y_i\mathbf{x}_i$。

**Formal definition**: $\mathbf{w}\leftarrow\mathbf{w}+2\eta y_i\mathbf{x}_i$ (only when $o_i\ne y_i$); $d=y(\mathbf{w}^\top\mathbf{x}+w_0)/\|\mathbf{w}\|$; $E=-\sum_{i\in M}y_i(\mathbf{w}^\top\mathbf{x}_i+w_0)$ and $\partial E/\partial\mathbf{w}=-\sum_{i\in M}y_i\mathbf{x}_i$.

**例子**：$\mathbf{w}=(1,1)$、$w_0=-1.5$：$d=(1.06,-0.35,-0.35,0.35)$；對錯點 $(0,1)$ 以 $\eta=\tfrac12$ 更新後 $s$ 由 $-0.5$ 升到 $1.5$。

**Example**: With $\mathbf{w}=(1,1)$ and $w_0=-1.5$: $d=(1.06,-0.35,-0.35,0.35)$; after updating the wrong point $(0,1)$ with $\eta=\tfrac12$, $s$ rises from $-0.5$ to $1.5$.

**常見迷思**：

- 「更新時把所有點的 $y\mathbf{x}$ 都加進去」——不對，只有猜錯的點才加；猜對的點也加，權重會一直長大，還可能把分對的點推錯。
- 「$\operatorname{sign}(0)$ 取什麼都一樣」——不一樣：剛好落在邊界上的點會被分到另一邊，軌跡也會不同。本講義取 $+1$，與 KP01 的 $s\ge0$ 放電一致。

**Common misconceptions**:

- "At each update add $y\mathbf{x}$ for every point." No: only wrongly classified points are added; if right points were added too, the weights would keep growing and could push already-correct points to the wrong side.
- "It does not matter what $\operatorname{sign}(0)$ is." It does: a point exactly on the boundary goes to the other side and the trace changes. This note takes $+1$, matching KP01's "fire when $s\ge0$".

**圖**：本單元一張圖，`lec02-kp03-signum-geometry`（法向量與帶符號距離，功能 2、3），以 `assets/figures/src/lec02-kp0204-figures.py` 繪製。「兩條路得到同一規則」在簡報中用原生節點與連線表示。

**Figure**: One figure in this unit, `lec02-kp03-signum-geometry` (normal vector and signed distance; functions 2, 3), drawn by `assets/figures/src/lec02-kp0204-figures.py`. "Two routes, one rule" is shown with native nodes and connectors in the slides.

**想深入**（選讀）：來源把幾何路徑寫成「最大化正確點的距離」與「最小化錯誤點的距離」兩種說法。嚴格來說，只看錯分點的 $E$ 才是 Perceptron 的損失，它在所有點都對時恰為 0。這個離散的損失之後會被 ADALINE（KP08）換成連續、處處可微的平方誤差。

**Going deeper** (optional): The source states the geometric route as both "maximize the distance of correct points" and "minimize the distance of wrong points". Strictly, only the $E$ over wrong points is the Perceptron's loss, which is exactly 0 when every point is right. ADALINE (KP08) will replace this loss by a squared error that is continuous and differentiable everywhere.

**小練習**：

1. $\mathbf{w}=(1,1)$、$w_0=-1.5$，點 $(1,0)$ 的標籤 $y=+1$，$\eta=\tfrac12$ 更新一次後，新的 $s$ 是多少？（提示：先算新權重，再代回 $(1,0)$。）
2. 邊界 $\mathbf{w}=(3,4)$、$w_0=-5$，點 $(0,0)$ 到邊界的距離是多少？（提示：$|w_0|/\|\mathbf{w}\|$，先算 $\|\mathbf{w}\|$。）
3. 為什麼用 $\pm1$ 標籤時，誤差只可能是 $0$ 或 $\pm2$，不會是 $\pm1$？（提示：列出四種 $(y,o)$ 組合。）

**Practice**:

1. With $\mathbf{w}=(1,1)$ and $w_0=-1.5$, the point $(1,0)$ has label $y=+1$. After one update with $\eta=\tfrac12$, what is the new $s$? (Hint: compute the new weights, then substitute $(1,0)$.)
2. For the boundary $\mathbf{w}=(3,4)$, $w_0=-5$, what is the distance from $(0,0)$ to the boundary? (Hint: $|w_0|/\|\mathbf{w}\|$; compute $\|\mathbf{w}\|$ first.)
3. Why can the error be only $0$ or $\pm2$ and never $\pm1$ with $\pm1$ labels? (Hint: list the four $(y,o)$ combinations.)

**解題示範**（第 1 題）：

1. 目前 $s=1\cdot1+1\cdot0-1.5=-0.5$，$o=-1$，$y=+1$，猜錯。
2. 更新：$\mathbf{x}=(1,0)$（偏置另算），$2\eta=1$，$\mathbf{w}\leftarrow(1,1)+1\cdot(+1)\cdot(1,0)=(2,1)$，$w_0\leftarrow-1.5+1=-0.5$。
3. 新的 $s=2\cdot1+1\cdot0-0.5=1.5>0$，這一點對了。
4. 檢查：$s$ 增加 $1.5-(-0.5)=2=2\eta(\|\mathbf{x}\|^2+1)=1\cdot(1+1)$。

**Worked solution** (exercise 1):

1. Currently $s=1\cdot1+1\cdot0-1.5=-0.5$, $o=-1$ and $y=+1$, so it is wrong.
2. Update: $\mathbf{x}=(1,0)$ (the bias separately), $2\eta=1$, so $\mathbf{w}\leftarrow(1,1)+1\cdot(+1)\cdot(1,0)=(2,1)$ and $w_0\leftarrow-1.5+1=-0.5$.
3. The new $s=2\cdot1+1\cdot0-0.5=1.5>0$, so this point is right.
4. Check: $s$ rose by $1.5-(-0.5)=2=2\eta(\|\mathbf{x}\|^2+1)=1\cdot(1+1)$.

**自我檢核**（來源四題，不附答案）：

| 題 / Question | A | B | C | D | 提示 / Hint |
|---|---|---|---|---|---|
| 1. Signum 激活的 Perceptron，輸出 $o_i$ 取值？ / Which values can $o_i$ take? | 只有 0 | 只有 $+1$ | $+1$ 或 $-1$ | 任何實數 | 看符號表。 / See the symbol table. |
| 2. $y_i,o_i\in\{-1,1\}$ 且 $o_i\ne y_i$ 時，$\eta(y_i-o_i)\mathbf{x}_i$ 化簡為？ / What does it reduce to? | $\eta y_i\mathbf{x}_i$ | $2\eta y_i\mathbf{x}_i$ | $-2\eta\mathbf{x}_i$ | 0 | 看第 1 節表格。 / See the table in section 1. |
| 3. 為何把 $d=\lvert\mathbf{w}^\top\mathbf{x}+w_0\rvert$ 改寫成 $y(\mathbf{w}^\top\mathbf{x}+w_0)$？ / Why rewrite the distance? | 讓距離變大 / Make it larger | 絕對值不可微，改寫後可微 / The absolute value is not differentiable | 讓 $\mathbf{w}$ 變單位向量 / Make $\mathbf{w}$ a unit vector | 讓誤差變小 / Reduce the error | 想想梯度。 / Think of the gradient. |
| 4. 最小化「錯分距離」與最大化「正確距離」最後得到的更新規則？ / What do the two routes give? | 完全相同 / The same | 方向相反 / Opposite | 差一個負號 / Differ by a sign | 無法比較 / Not comparable | 看第 2 節。 / See section 2. |

**留給下一個知識點的問題**：Perceptron 這麼會學，是不是什麼分類問題都學得會？如果兩類資料沒辦法用一條直線分開，會發生什麼事？

**Question for the next knowledge point**: The Perceptron learns well, but can it learn any classification problem? What happens if the two classes cannot be separated by one line?

### 來源更正與取捨（Corrections to the source）

- 來源在幾何路徑裡寫「把正確點的距離乘 $-1$ 得 error」。嚴格來說，Perceptron 的損失只對**錯分點**求和，本講義改寫成 $E=-\sum_{i\in M}y_i(\mathbf{w}^\top\mathbf{x}_i+w_0)$。
- 來源說兩條路「殊途同歸」，但係數不同（代數是 $2\eta$，幾何是 $\eta'$）。本講義說明差別只是學習率的重新命名。
- 來源的批次式 (8) 用指示函數 $\mathbb{I}(o_i\ne y_i)$ 當開關；本講義用集合 $M$ 表示，兩者等價。

- In the geometric route the source "multiplies the distance of correct points by $-1$ to get the error". Strictly, the Perceptron's loss sums over the **wrongly classified points** only, so this note writes $E=-\sum_{i\in M}y_i(\mathbf{w}^\top\mathbf{x}_i+w_0)$.
- The source says the two routes "reach the same place", but the constants differ ($2\eta$ in the algebraic route and $\eta'$ in the geometric one). This note says the difference is only a renaming of the learning rate.
- The source's batch formula (8) uses the indicator $\mathbb{I}(o_i\ne y_i)$ as a switch; this note uses the set $M$, which is equivalent.

### 術語對照（Terminology）

| 中文 | English |
|---|---|
| 符號函數 | sign (signum) function |
| 法向量 | normal vector |
| 帶符號距離 | signed distance |
| 決策邊界 | decision boundary |
| 梯度下降 | gradient descent |
| 錯分點 | misclassified point |
| 指示函數 | indicator function |
