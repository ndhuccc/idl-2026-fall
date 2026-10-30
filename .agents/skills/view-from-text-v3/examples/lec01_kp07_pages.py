import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *
from lec01_kp07_nums import *

KP = "KP07"; N = 10
TEAL = "#0f766e"

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    # ============ 1 ============
    g = new(1, T("KP07 回歸與降維保留不同資訊", "KP07 Regression vs reduction: what each keeps"))
    g.beat = 1
    card(g, 50, 130, 570, 470, BLUE); head(g, 70, 136, 530, T("回歸：預測一個數值", "Regression: predict a number"), BLUE, 24)
    g.line([(110, 520), (570, 520)], MUTED, sw=3, z=3); g.line([(110, 520), (110, 220)], MUTED, sw=3, z=3)
    for x, y in [(160, 440), (220, 420), (270, 380), (330, 350), (390, 330), (450, 290), (510, 270)]: g.ellipse(x, y, 10, fill=BLUE, stroke="#ffffff", sw=2, z=5)
    g.line([(130, 460), (550, 260)], ORG, sw=5, z=4)
    g.text(70, 545, 530, 40, T("輸入特徵 → 預測直徑", "input features → predicted diameter"), size=20, color=INK, bold=True, align="center", z=8)
    g.beat = 2
    card(g, 660, 130, 570, 470, GRN); head(g, 680, 136, 530, T("降維：改變資料的表示", "Reduction: change the representation"), GRN, 24)
    for (x, y) in [(760, 260), (830, 330), (900, 300), (980, 380), (1060, 340), (1130, 420), (800, 440), (1000, 470)]: g.ellipse(x, y, 10, fill="#64748b", stroke="#ffffff", sw=2, z=5)
    arrow(g, (950, 500), (950, 548), GRN, sw=5, head=18); g.line([(780, 560), (1130, 560)], GRN, sw=5, z=4)
    for x in (800, 850, 910, 990, 1060, 1110): g.ellipse(x, 560, 7, fill=GRN, z=5)
    g.text(680, 570, 530, 36, T("$d$ 個座標 → $p$ 個座標", "$d$ coordinates → $p$ coordinates"), size=20, color=INK, bold=True, align="center", z=8) if False else None
    g.beat = 3
    banner(g, T("先問要預測什麼，再問要保留什麼", "Ask what to predict, then what to preserve"))
    pages.append((g, T("預測與表示", "Prediction vs representation"), {
        0: T("KP05 問「保留什麼」。這個單元分清楚兩件事：預測連續值的回歸，和改變資料表示的降維。（圖是示意，不是講義的資料。）", "KP05 asked what to preserve. This unit separates two things: regression, which predicts a number, and reduction, which changes the representation. (The plots are sketches, not the note's data.)"),
        1: T("回歸：從特徵預測數值，例如水果的直徑。", "Regression: predict a number from features, such as a fruit's diameter."),
        2: T("降維：把 d 個座標變成 p 個座標的表示。正則化控制預測模型的係數，降維改變資料的表示；不是每次都必須先降維再預測。", "Reduction: turn d coordinates into a p-coordinate representation. Regularization controls the model's coefficients, reduction changes the representation; there is no mandatory reduce-then-predict pipeline."),
        3: T("兩者各有各的任務。", "Each has its own job.")}))

    # ============ 2:殘差 ============
    g = new(2, T("回歸：把預測的誤差（殘差）壓低", "Regression: make the prediction errors (residuals) small"))
    pill(g, 960, 96, 270, T("示意：不是講義的資料", "sketch: not the note's data"), GRAY, size=16)
    g.beat = 1
    card(g, 50, 140, 700, 460, BLUE); head(g, 70, 146, 660, T("殘差 = 實際值 − 預測值", "residual = actual − predicted"), BLUE, 22)
    g.line([(110, 560), (720, 560)], MUTED, sw=3, z=3); g.line([(110, 560), (110, 200)], MUTED, sw=3, z=3)
    pts = [(170, 470), (240, 420), (300, 410), (370, 350), (440, 340), (500, 280), (560, 300), (640, 240)]
    line = lambda x: 500 - (x - 130) * 0.62
    g.beat = 2
    g.line([(130, line(130)), (700, line(700))], ORG, sw=6, z=5)
    for x, y in pts:
        g.line([(x, y), (x, line(x))], RED, sw=3, z=4)
    g.beat = 1
    for x, y in pts: g.ellipse(x, y, 10, fill=BLUE, stroke="#ffffff", sw=2, z=6)
    g.beat = 3
    card(g, 780, 140, 450, 460, GRAY); head(g, 800, 146, 410, T("未懲罰最小平方法", "Unpenalized least squares"), MUTED, 22)
    g.text(800, 210, 410, 130, r"$\min_{\boldsymbol\beta,b}\ \frac{1}{2n}\|\mathbf y-D\boldsymbol\beta-b\mathbf 1\|_2^2$", size=26, color=INK, align="center", vcenter=True, z=8)
    g.text(800, 360, 410, 220, T("$D$ 每一列是一個樣本\n（$D=X^{\\top}$，$X$ 是 KP02 的直欄約定）\n\n係數太大、不穩定時，\n只壓低訓練殘差並不夠", "each row of $D$ is a sample\n($D=X^{\\top}$, with KP02's column convention for $X$)\n\ncoefficients can grow unstable\nwhen only training residuals are minimized"), size=19, color=INK, bold=True, align="center", z=8)
    banner(g, T("只壓低訓練殘差，可能得到不穩定的大係數", "Minimizing training residuals alone can give unstable, large coefficients"))
    pages.append((g, T("回歸與殘差", "Regression and residuals"), {
        0: T("線性回歸從特徵預測數值。這張圖是示意：點是資料。", "Linear regression predicts a number from features. This sketch shows the data as points."),
        1: T("橘色線是預測；紅色的短線是殘差，也就是實際值減預測值。", "The orange line is the prediction; the short red segments are residuals, actual minus predicted."),
        2: T("未懲罰的最小平方法，把殘差平方的平均壓到最小。D 每一列是一個樣本，是 KP02 以直欄放樣本的 X 的轉置。最小平方也可以理解成把 y 投影到設計矩陣的欄空間；唯一解需要滿秩條件，並不是每個特徵統計獨立。", "Unpenalized least squares minimizes the mean squared residual. Each row of D is a sample, the transpose of KP02's column-sample X. Least squares can also be seen as projecting y onto the design matrix's column space; uniqueness needs full rank, not statistical independence of features."),
        3: T("只壓低訓練殘差，可能給出不穩定的大係數，所以可以在損失外加懲罰。", "Minimizing training residuals only can give unstable large coefficients, so a penalty can be added to the loss.")}))

    # ============ 3:Ridge vs Lasso ============
    g = new(3, T("Ridge 與 Lasso：在損失外加不同的懲罰", "Ridge and Lasso: different penalties added to the loss"))
    for i, (col, light, nm, f_, sub) in enumerate(((BLUE, BLUEL, "Ridge", r"$\text{loss}+\frac{\lambda}{2}\|\boldsymbol\beta\|_2^2$", T("平方 $L_2$ 懲罰：係數整體收縮", "squared $L_2$ penalty: shrinks all coefficients")), (ORG, ORGL, "Lasso", r"$\text{loss}+\lambda\|\boldsymbol\beta\|_1$", T("$L_1$ 懲罰：可能把係數壓成零", "$L_1$ penalty: may set coefficients to zero")))):
        g.beat = i + 1
        x = 50 + i * 600
        card(g, x, 130, 580, 380, col); head(g, x + 20, 136, 540, nm, col, 32)
        g.rect(x + 40, 210, 500, 120, fill=light, stroke=col, sw=3, radius=14, z=3); g.text(x + 50, 210, 480, 120, f_, size=36, color=INK, align="center", vcenter=True, z=8)
        g.text(x + 40, 440, 500, 60, sub, size=22, color=INK, bold=True, align="center", z=8)
        if i == 0: g.ellipse(x + 290, 395, 38, stroke=col, sw=5, z=4, fill="rgba(29,78,216,0.10)")
        else: g.poly([(x + 290, 355), (x + 335, 395), (x + 290, 435), (x + 245, 395)], fill="rgba(194,65,12,0.10)", stroke=col, sw=5, z=4)
    g.beat = 3
    g.text(50, 530, 1180, 90, T("$\\lambda\\ge0$ 是懲罰強度；通常不懲罰截距 $b$。\n不同軟體的 `alpha` 與這裡的 $\\lambda$ 數值不能直接對照", "$\\lambda\\ge0$ is the penalty strength; the intercept $b$ is usually not penalized.\nAnother package's `alpha` cannot be compared with this $\\lambda$ directly"), size=20, color=MUTED, bold=True, align="center", z=8)
    banner(g, T("這是改變預測模型的係數，不是把輸入投影成較低維", "This changes the model's coefficients; it does not project inputs to fewer dimensions"))
    pages.append((g, T("Ridge 與 Lasso", "Ridge and Lasso"), {
        0: T("Ridge 在損失後加 lambda 乘係數的平方 L2 範數，再除以 2。", "Ridge adds lambda times the squared L2 norm of the coefficients, divided by 2, to the loss."),
        1: T("Lasso 加 lambda 乘係數的 L1 範數。", "Lasso adds lambda times the L1 norm of the coefficients."),
        2: T("lambda 不小於 0，是懲罰強度。通常不懲罰截距。兩種規範化的數值，不能直接拿另一套軟體的 alpha 對照。它們改變預測模型的係數，不是把輸入維度直接投影成較低維。", "Lambda is non-negative and sets the penalty strength; the intercept is usually not penalized. The numbers cannot be compared directly with another package's alpha. They change the model's coefficients rather than projecting inputs to fewer dimensions."),
        3: T("選 lambda 時，要看沒有用來擬合的資料。", "Choose lambda by looking at data not used for fitting.")}))

    # ============ 4:控制例 ============
    g = new(4, T("固定數字：Ridge 收縮，Lasso 把一個係數壓成 0", "Fixed numbers: Ridge shrinks, Lasso zeroes"))
    pill(g, 890, 62, 140, T("講義補充", "Added note"), GRAY, size=16)
    g.beat = 1
    card(g, 50, 120, 1180, 100, GRAY); g.text(70, 124, 1140, 92, T(r"指定 $\mathbf z=(3,\,0.5)$、$\lambda=1$、$L(\boldsymbol\beta)=\frac12\|\boldsymbol\beta-\mathbf z\|_2^2$（兩個獨立係數）", r"Prescribed $\mathbf z=(3,\,0.5)$, $\lambda=1$, $L(\boldsymbol\beta)=\frac12\|\boldsymbol\beta-\mathbf z\|_2^2$ (two independent coefficients)"), size=24, color=INK, align="center", vcenter=True, z=8)
    ox, oy, U = 100, 560, 62
    bars = [(T("未加懲罰", "no penalty"), z, GRAY, "#e2e8f0", 140), (T("Ridge", "Ridge"), ridge, BLUE, BLUEL, 510), (T("Lasso", "Lasso"), lasso, ORG, ORGL, 880)]
    def group(x0, nm, vals, col, light):
        g.text(x0 - 10, 250, 330, 36, nm, size=24, color=col if col != GRAY else MUTED, bold=True, align="center", vcenter=True, z=8)
        for j, v in enumerate(vals):
            h = max(v * U, 3); g.rect(x0 + j * 120, oy - h, 90, h, fill=light, stroke=col, sw=3, radius=4, z=4)
            g.text(x0 + j * 120 - 10, oy - h - 40, 110, 34, rf"$\beta_{j + 1}={v:g}$", size=20, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.line([(ox, oy), (1180, oy)], MUTED, sw=3, z=3)
    group(140, *bars[0][:1], bars[0][1], bars[0][2], bars[0][3])
    g.beat = 2
    group(510, bars[1][0], bars[1][1], bars[1][2], bars[1][3])
    g.text(480, 600 - 6, 400, 34, r"$\boldsymbol\beta_R=\frac{\mathbf z}{1+\lambda}=(1.5,\,0.25)$", size=20, color=BLUE, align="center", vcenter=True, z=8) if False else None
    g.beat = 3
    group(880, bars[2][0], bars[2][1], bars[2][2], bars[2][3])
    banner(g, T(r"Ridge $\boldsymbol\beta_R=(1.5,\,0.25)$；Lasso $\boldsymbol\beta_L=(2,\,0)$", r"Ridge $\boldsymbol\beta_R=(1.5,\,0.25)$; Lasso $\boldsymbol\beta_L=(2,\,0)$"))
    pages.append((g, T("Ridge 與 Lasso 的算例", "A worked Ridge and Lasso example"), {
        0: T("這是講義的補充，一個可以完整手算的控制例：指定 z 等於 (3, 0.5)、lambda 等於 1，損失是 1/2 乘 beta 減 z 的平方範數。這是兩個獨立係數的教學控制，不是任意相關設計矩陣的 Ridge 或 Lasso 通式，也不是擬合的資料結果。未加懲罰的解就是 z。", "A supplement: a control example that can be computed by hand. Prescribe z equal to (3, 0.5), lambda equal to 1, and loss one half of the squared norm of beta minus z. It is a teaching control with two independent coefficients, not the general Ridge or Lasso formula for correlated designs, and not a fitted result. The unpenalized solution is z itself."),
        1: T("Ridge：一階條件 beta 減 z 加 lambda 乘 beta 等於 0，所以 beta 等於 z 除以 1 加 lambda，得到 1.5 和 0.25。兩個係數都被等比例收縮。", "Ridge: the first-order condition beta minus z plus lambda times beta equals 0, so beta equals z over 1 plus lambda, giving 1.5 and 0.25. Both coefficients shrink proportionally."),
        2: T("Lasso：對每個座標做軟閾值，絕對值減 lambda，小於 0 就取 0。第一個是 3 減 1 等於 2；第二個是 0.5 減 1 小於 0，所以是 0。結果 (2, 0)。", "Lasso: soft-threshold each coordinate, absolute value minus lambda, floored at 0. The first is 3 minus 1, which is 2; the second is 0.5 minus 1, below 0, so 0. The result is (2, 0)."),
        3: T("所有數值由程式與獨立算式核對。Lasso 可能得到零係數，卻不保證選出真正有用的特徵；係數大小也受特徵尺度影響。", "All values were checked by program and independent formulas. Lasso can zero coefficients without guaranteeing it picks the truly useful features; coefficient size also depends on feature scale.")}))

    # ============ 5:幾何 ============
    g = new(5, T("圓滑的 $L_2$ 球，與帶角的 $L_1$ 球", "The smooth $L_2$ ball and the cornered $L_1$ ball"))
    pill(g, 870, 96, 360, T("示意：形狀不代表同一預算", "sketch: shapes are not a shared budget"), GRAY, size=16)
    g.beat = 1
    card(g, 50, 140, 720, 460, BLUE); head(g, 70, 146, 680, T("係數平面 $(\\beta_1,\\beta_2)$", "Coefficient plane $(\\beta_1,\\beta_2)$"), BLUE, 22)
    cx, cy, S = 200, 400, 80
    g.line([(100, cy), (740, cy)], MUTED, sw=3, z=2, arrow="end", head=14); g.line([(cx, 580), (cx, 190)], MUTED, sw=3, z=2, arrow="end", head=14)
    g.text(710, cy + 6, 50, 30, r"$\beta_1$", size=22, color=MUTED, z=8); g.text(cx + 10, 186, 50, 30, r"$\beta_2$", size=22, color=MUTED, z=8)
    g.ellipse(cx + z[0] * S, cy - z[1] * S, 12, fill=INK, stroke="#ffffff", sw=3, z=7); g.text(cx + z[0] * S - 20, cy - z[1] * S - 46, 160, 34, r"$\mathbf z=(3,\,0.5)$", size=20, color=INK, bold=True, z=8)
    g.beat = 2
    r2 = math.hypot(*ridge) * S
    g.ellipse(cx, cy, r2, stroke=BLUE, sw=5, z=4, fill="rgba(29,78,216,0.07)"); g.ellipse(cx + ridge[0] * S, cy - ridge[1] * S, 10, fill=BLUE, stroke="#ffffff", sw=3, z=7)
    g.text(cx + ridge[0] * S - 100, cy - ridge[1] * S - 62, 230, 30, r"Ridge $(1.5,\,0.25)$", size=18, color=BLUE, bold=True, z=8)
    g.beat = 3
    r1 = lasso[0] * S
    g.poly([(cx + r1, cy), (cx, cy - r1), (cx - r1, cy), (cx, cy + r1)], fill="rgba(194,65,12,0.08)", stroke=ORG, sw=5, z=4); g.ellipse(cx + lasso[0] * S, cy, 10, fill=ORG, stroke="#ffffff", sw=3, z=7)
    g.text(cx + lasso[0] * S + 14, cy + 22, 220, 30, r"Lasso $(2,\,0)$", size=18, color=ORG, bold=True, z=8)
    card(g, 810, 140, 420, 460, GRAY); head(g, 830, 146, 380, T("看點", "What to notice"), MUTED, 22)
    g.text(830, 200, 380, 380, T("$L_2$ 球是圓：\n沒有角，係數整體收縮\n\n$L_1$ 球是菱形：\n在座標軸上有角，\n解容易落在軸上，\n所以可能出現零係數\n\n零係數不保證\n選出真實有用的特徵", "the $L_2$ ball is a disk:\nno corners, coefficients shrink together\n\nthe $L_1$ ball is a diamond:\ncorners on the axes,\nsolutions often land on an axis,\nso zero coefficients can occur\n\nzeros do not guarantee\nthe truly useful features"), size=20, color=INK, bold=True, z=8)
    banner(g, T("形狀幫助理解「為什麼」，不是一般資料的解", "The shapes explain why, not a solution for general data"))
    pages.append((g, T("正則化的幾何", "The geometry of regularization"), {
        0: T("把兩個係數畫在平面上：黑點是指定的 z，也是沒有懲罰時的解。", "Plot the two coefficients on a plane: the black dot is the prescribed z, also the unpenalized solution."),
        1: T("Ridge 的解 (1.5, 0.25)，落在一個圓上。這個圓畫在 Ridge 解的位置，只是幾何示意。", "The Ridge solution (1.5, 0.25) lies on a circle drawn through it; a geometric sketch only."),
        2: T("Lasso 的解 (2, 0)，落在菱形的角上，剛好在座標軸上，第二個係數是 0。圓與菱形只是幾何示意，沒有宣稱同一個約束預算，也不是一般資料的解。", "The Lasso solution (2, 0) sits at a corner of a diamond, on the axis, with the second coefficient 0. The disk and diamond are sketches that do not claim a shared constraint budget or a general-data solution."),
        3: T("這幫助理解為什麼 Lasso 可能得到零係數，但稀疏不等於選出了真正有用的特徵。", "This helps explain why Lasso may give zero coefficients, but sparsity alone does not establish correct feature selection.")}))

    # ============ 6:回歸家族 ============
    g = new(6, T("回歸家族：各回答不同的問題，不是升級階梯", "Regression family: different questions"))
    pill(g, 960, 96, 270, T("示意：不是講義的資料", "sketch: not the note's data"), GRAY, size=16)
    fam = [(BLUE, BLUEL, T("線性回歸", "Linear regression"), T("條件平均能用特徵的\n線性組合預測嗎？", "can a linear combination\npredict the mean?")), (ORG, ORGL, "GLM", T("機率、計數也能用\n連結函數處理嗎？", "can a link function handle\nprobabilities and counts?")),
           (GRN, GRNL, T("分位數回歸", "Quantile regression"), T("想預測第 $\\tau$ 分位，\n而不是平均？", "predict the $\\tau$-quantile\ninstead of the mean?")), (PUR, PURL, T("非線性回歸", "Nonlinear regression"), T("輸入與回應的\n關係彎曲怎麼辦？", "what if the relation\nbends?"))]
    for i, (col, light, nm, sub) in enumerate(fam):
        g.beat = min(i + 1, 3)
        x = 50 + i * 300
        g.rect(x, 140, 280, 440, fill=CARD, stroke=col, sw=3, radius=16, z=1)
        g.rect(x, 140, 280, 70, fill=light, radius=16, z=2); g.rect(x, 180, 280, 30, fill=light, z=2); g.text(x + 10, 140, 260, 70, nm, size=22 if zh else 20, color=col, bold=True, align="center", vcenter=True, z=8)
        if i == 0: g.line([(x + 40, 400), (x + 240, 270)], col, sw=6, z=5)
        if i == 1: g.line([(x + 40 + k * 20, 400 - 120 / (1 + math.exp(-(k - 5) * 0.9))) for k in range(11)] and [(x + 40 + k * 20, 400 - 130 / (1 + math.exp(-(k - 5) * 0.9))) for k in range(11)], col, sw=6, smooth=True, z=5)
        if i == 2:
            g.poly([(x + 40, 330), (x + 140, 300), (x + 240, 250), (x + 240, 360), (x + 140, 380), (x + 40, 420)], fill="rgba(21,128,61,0.15)", stroke=col, sw=3, z=3); g.line([(x + 40, 375), (x + 140, 340), (x + 240, 305)], col, sw=5, dash=True, z=5)
        if i == 3: g.line([(x + 40, 400), (x + 100, 330), (x + 160, 290), (x + 210, 330), (x + 240, 270)], col, sw=6, smooth=True, z=5)
        g.text(x + 12, 450, 256, 120, sub, size=18, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    banner(g, T("選哪個，由目標與假設決定：回歸與分類也可共享樹、KNN、神經網路", "Target and assumptions decide; trees, KNN, nets serve both tasks"))
    pages.append((g, T("回歸家族", "The regression family"), {
        0: T("這四種是不同的問題，不是前一個不夠就一定改用下一個的歷史階梯。圖是示意。", "These are different questions, not a ladder where each replaces the last. The plots are sketches."),
        1: T("廣義線性模型用連結函數處理機率、計數等；條件分布與連結函數要相配，羅吉斯迴歸是例子。", "Generalized linear models use a link function for probabilities and counts; the conditional distribution and link must match, with logistic regression as one example."),
        2: T("分位數回歸預測某個分位數，最小化 pinball 損失；非線性回歸處理彎曲的關係，多項式特徵對係數仍可以是線性的。", "Quantile regression predicts a quantile by minimizing pinball loss; nonlinear regression handles bent relations, and polynomial features can remain linear in the coefficients."),
        3: T("回歸與分類可以共享工具，差在目標與輸出。", "Regression and classification can share tools; the difference is the target and output.")}))

    # ============ 7:降維家族 ============
    g = new(7, T("降維：每個方法保留的東西不同", "Reduction: each method preserves something different"))
    items = [(BLUE, BLUEL, "PCA", T("最大的線性變異", "greatest linear variance")), (ORG, ORGL, "FDA", T("類別的分離（用標籤）", "class separation (uses labels)")), (GRN, GRNL, "MDS / Sammon", T("成對的差異；Sammon 重視小距離", "pairwise dissimilarities; Sammon stresses small distances")),
             (PUR, PURL, T("核 PCA", "Kernel PCA"), T("在核隱含空間做 PCA", "PCA in an implicit feature space")), (RED, REDL, "Isomap / LLE", T("測地距離 / 局部重建權重", "geodesic distances / local reconstruction weights")), (TEAL, "#ccfbf1", "HSIC", T("與標籤的核相依性", "kernel dependence on labels"))]
    for i, (col, light, nm, sub) in enumerate(items):
        g.beat = 1 if i < 2 else (2 if i < 4 else 3)
        x = 50 + (i % 3) * 400; y = 130 + (i // 3) * 200
        g.rect(x, y, 380, 180, fill=light, stroke=col, sw=3, radius=16, z=3); g.text(x + 10, y + 12, 360, 50, nm, size=26, color=col, bold=True, align="center", vcenter=True, z=8); g.text(x + 20, y + 72, 340, 100, sub, size=20, color=INK, bold=True, align="center", z=8)
        g.ellipse(x + 40, y + 36, 14, fill=col, z=5)
    g.beat = 3
    g.text(50, 540, 1180, 90, T("來源的年代順序只是記憶線索：PCA（20 世紀初）→ Sammon（1969）→ 核 PCA（1990 年代後期）→ Isomap、LLE（2000）\n不表示後者取代前者", "The source's chronology is only a memory aid: PCA (early 1900s) → Sammon (1969) → kernel PCA (late 1990s) → Isomap, LLE (2000)\nit does not mean each replaces the previous one"), size=19, color=MUTED, bold=True, align="center", z=8)
    banner(g, T("降維要說清楚「保留什麼」：維度變低本身不保證保留任何東西", "Say what is preserved: lower dimension alone guarantees nothing"))
    pages.append((g, T("降維家族", "The reduction family"), {
        0: T("降維建立 z 等於 f(x)，p 小於 d。z 可以服務視覺化、雜訊處理或後續預測，但必須說清楚保留的是哪種資訊。PCA 保留最大線性變異；FDA 用類別側資訊分開類別。", "Reduction builds z equal to f of x with p less than d. It can serve visualization, noise handling or prediction, but must say what it preserves. PCA keeps the greatest linear variance; FDA uses class side information to separate classes."),
        1: T("MDS 依差異矩陣配置點；Sammon 特別重視原空間的小距離。核 PCA 在核隱含空間做 PCA，核的選擇會影響結果。", "MDS arranges points from dissimilarities; Sammon especially weighs small original distances. Kernel PCA is PCA in an implicit feature space, and the kernel choice matters."),
        2: T("Isomap 用近鄰圖的最短路徑近似測地距離；LLE 先用近鄰重建權重，再在低維保留。近鄰選錯或取樣稀疏可能扭曲結果。HSIC 是核相依性統計量，不等於普通線性相關。", "Isomap approximates geodesic distance by shortest paths on a neighbor graph; LLE first finds local reconstruction weights then keeps them in low dimension. Bad neighborhoods or sparse sampling can distort results. HSIC is a kernel dependence statistic, not ordinary linear correlation."),
        3: T("年代順序只能當記憶線索，不表示某種唯一首創，也不是單一必然的演化路徑。", "The chronology is a memory aid, not a claim of a unique first or an inevitable evolution.")}))

    # ============ 8:因子模型 ============
    g = new(8, T("因子模型：少數潛在因子，加上各座標的雜訊", "Factor models: a few latent factors plus coordinate noise"))
    g.beat = 1
    steps = [(BLUE, BLUEL, r"$\mathbf z$", T("潛在因子", "latent factor")), (ORG, ORGL, r"$W$", T("載荷矩陣", "loading matrix")), (GRN, GRNL, r"$+\boldsymbol\mu+\boldsymbol\varepsilon$", T("加均值與雜訊", "add mean and noise")), (PUR, PURL, r"$\mathbf x$", T("觀測", "observation"))]
    for i, (col, light, sym, nm) in enumerate(steps):
        g.beat = 1 if i < 2 else 2
        x = 50 + i * 305
        g.rect(x, 150, 250, 200, fill=light, stroke=col, sw=3, radius=16, z=3); g.text(x, 160, 250, 110, sym, size=40, color=INK, bold=True, align="center", vcenter=True, z=8); g.text(x, 280, 250, 50, nm, size=22, color=col, bold=True, align="center", vcenter=True, z=8)
        if i < 3: arrow(g, (x + 256, 250), (x + 300, 250), GRAY, sw=5, head=16)
    g.beat = 2
    g.text(50, 370, 1180, 70, r"$\mathbf x=W\mathbf z+\boldsymbol\mu+\boldsymbol\varepsilon,\quad \operatorname{Cov}(\mathbf x)=WW^{\top}+\Psi$", size=32, color=INK, align="center", vcenter=True, z=8)
    g.beat = 3
    for i, (col, light, nm, f_, sub) in enumerate(((BLUE, BLUEL, T("因子分析 FA", "Factor analysis (FA)"), r"$\Psi=\operatorname{diag}(\psi_1,\ldots,\psi_d)$", T("各座標雜訊變異可以不同", "noise variance may differ by coordinate")), (ORG, ORGL, T("機率式 PCA（pPCA）", "Probabilistic PCA (pPCA)"), r"$\Psi=\sigma^2 I_d$", T("各方向雜訊變異相同（等向）", "the same noise variance in every direction")))):
        x = 50 + i * 600
        g.rect(x, 460, 580, 150, fill=light, stroke=col, sw=3, radius=14, z=3); g.text(x + 10, 466, 560, 40, nm, size=22, color=col, bold=True, align="center", vcenter=True, z=8); g.text(x + 10, 508, 560, 50, f_, size=26, color=INK, align="center", vcenter=True, z=8); g.text(x + 10, 560, 560, 40, sub, size=18, color=INK, bold=True, align="center", z=8)
    banner(g, T("FA 與 pPCA 的差別，在雜訊的假設", "FA and pPCA differ in their noise assumption"))
    pages.append((g, T("因子模型", "Factor models"), {
        0: T("另一種問題是：觀察到的散布，是不是由少數共同因素加上各座標的雜訊形成的？潛在因子 z 經過載荷矩陣 W 映到觀察空間。", "Another question: is the observed spread produced by a few shared factors plus coordinate noise? The latent factor z is mapped into observation space by the loading matrix W."),
        1: T("加上均值與雜訊以後，就成為觀測 x；總共變異數是 W 乘 W 轉置，再加 Psi。", "Adding the mean and noise gives the observation x; the total covariance is W times W transpose plus Psi."),
        2: T("因子分析允許各座標有不同的獨立雜訊變異；機率式 PCA 把它限制成各方向相同。等向須同時有對角形式與各座標相同的變異。對完整觀測、高斯模型與標準極大概似條件，pPCA 的最優載荷子空間和 PCA 的主方向一致，有閉式解；EM 不等於變分推論。", "Factor analysis allows different independent noise variances; probabilistic PCA restricts them to be equal. Isotropic requires both diagonal form and equal variances. For complete data, a Gaussian model and standard maximum likelihood, the optimal pPCA loading subspace matches PCA's principal directions with a closed form; EM is not variational inference."),
        3: T("講義的 4 題自我檢核留在講義。", "The note's 4 self-check questions stay in the note.")}))

    # ============ 9:兩種解釋 ============
    g = new(9, T("同一個總散布，可以有兩種雜訊解釋", "One total spread, two noise explanations"))
    pill(g, 890, 62, 140, T("講義補充", "Added note"), GRAY, size=16)
    g.beat = 1
    card(g, 50, 120, 1180, 100, GRAY); g.text(70, 124, 1140, 92, T(r"沿用 KP05 的 8 點：$C=\operatorname{diag}(9,\,1.04)$，$d=2,\ q=1$", r"Reusing KP05's 8 points: $C=\operatorname{diag}(9,\,1.04)$, $d=2,\ q=1$"), size=26, color=INK, align="center", vcenter=True, z=8)
    S9 = 38
    def panel(x0, col, light, ttl, Wv, psi, line1, line2, beat):
        g.beat = beat
        card(g, x0, 240, 580, 360, col); head(g, x0 + 20, 246, 540, ttl, col, 22)
        cx, cy = x0 + 160, 420
        g.ellipse(cx, cy, math.sqrt(C[0]) * S9 * 0.6, math.sqrt(C[1]) * S9 * 0.6, stroke=TEAL, sw=4, z=3, fill="rgba(15,118,110,0.06)")
        g.ellipse(cx, cy, math.sqrt(psi[0]) * S9 * 0.6, math.sqrt(psi[1]) * S9 * 0.6, stroke=col, sw=4, z=5, fill="rgba(29,78,216,0.12)" if col == BLUE else "rgba(194,65,12,0.12)")
        g.text(x0 + 290, 280, 280, 40, line1, size=20, color=INK, bold=True, z=8)
        a_, b_ = line2.split("\n")
        g.text(x0 + 290, 330, 280, 44, a_, size=20, color=INK, z=8); g.text(x0 + 290, 380, 285, 44, b_, size=18, color=INK, z=8)
        g.text(x0 + 300, 470, 270, 110, T("青綠：總散布 $C$\n有色：雜訊 $\\Psi$", "teal: total spread $C$\ncolor: noise $\\Psi$"), size=16, color=MUTED, z=8)
    panel(50, BLUE, BLUEL, T("FA：雜訊各座標不同", "FA: noise differs by coordinate"), fa_W, fa_psi, r"$W\approx(2.9155,\,0)^{\top}$", r"$\Psi=\operatorname{diag}(0.5,\,1.04)$" + "\n" + r"$WW^{\top}+\Psi=\operatorname{diag}(9,\,1.04)$", 2)
    panel(650, ORG, ORGL, T("pPCA：雜訊是圓", "pPCA: noise is a circle"), pp_W, pp_psi, r"$W\approx(2.8213,\,0)^{\top}$", r"$\Psi=1.04\,I$" + "\n" + r"$WW^{\top}+\Psi=\operatorname{diag}(9,\,1.04)$", 3)
    banner(g, T("兩組都是指定且可核算的分解，不是由 8 點識別出的唯一真相", "Both are specified, checkable decompositions, not a unique truth identified from 8 points"))
    pages.append((g, T("同一散布兩種解釋", "Two readings of one spread"), {
        0: T("這是講義的補充：沿用 KP05 那 8 個固定點的總共變異數 C，等於對角矩陣 9 與 1.04，設 d 等於 2、q 等於 1。", "A supplement: reuse the total covariance C of KP05's 8 fixed points, the diagonal matrix 9 and 1.04, with d equal to 2 and q equal to 1."),
        1: T("因子分析：W 是根號 8.5，約 2.9155，Psi 是對角 0.5 與 1.04。W W 轉置加 Psi 等於對角 9 與 1.04。青綠色橢圓是總散布，藍色橢圓是雜訊，長短軸不同。", "Factor analysis: W is the square root of 8.5, about 2.9155, and Psi is diagonal 0.5 and 1.04. W W transpose plus Psi equals diagonal 9 and 1.04. The teal ellipse is the total spread and the blue ellipse the noise, with unequal axes."),
        2: T("機率式 PCA：W 是根號 7.96，約 2.8213，Psi 是 1.04 乘單位矩陣。同樣加起來是對角 9 與 1.04。雜訊是圓。", "Probabilistic PCA: W is the square root of 7.96, about 2.8213, and Psi is 1.04 times the identity. It also sums to diagonal 9 and 1.04. The noise is a circle."),
        3: T("兩組是指定且可核算的模型分解，不是由 8 點識別出的唯一生成真相。數值由程式核對。", "The two are specified, checkable model decompositions, not a unique generative truth identified from the 8 points. Values were checked by program.")}))

    # ============ 10 ============
    g = new(10, T("不要把「先降維再回歸」當成通用流程", "Don't treat “reduce, then regress” as universal"))
    g.beat = 1
    card(g, 50, 130, 570, 400, RED); head(g, 70, 136, 530, T("低變異的方向，可能對預測最重要", "A low-variance direction can matter most"), RED, 22)
    g.line([(120, 340), (560, 340)], MUTED, sw=3, z=3, arrow="end", head=14); g.line([(340, 460), (340, 190)], MUTED, sw=3, z=3, arrow="end", head=14)
    for (x, y) in [(190, 300), (190, 285), (190, 390), (190, 375), (490, 300), (490, 285), (490, 390), (490, 375)]: g.ellipse(x, y, 10, fill=BLUE if y < 340 else ORG, stroke="#ffffff", sw=2, z=5)
    g.text(70, 464, 530, 62, T("KP05 的 8 點：$x_1$ 變異最大，\n分開類別的卻是 $x_2$", "KP05's 8 points: $x_1$ has the most variance,\nbut $x_2$ separates the classes"), size=18, color=INK, bold=True, align="center", z=8)
    g.beat = 2
    card(g, 660, 130, 570, 400, GRN); head(g, 680, 136, 530, T("其他定位", "Other pieces"), GRN, 22)
    g.text(680, 190, 530, 330, T("• JL 引理：有限點集的隨機線性投影可近似保距，\n  維度量級約 $O(\\log n/\\epsilon^2)$；\n  不是神經網路初始化的證明\n• SDR / KDR：問是否存在 $B$ 使\n  $P(Y\\mid\\mathbf X)=P(Y\\mid B^{\\top}\\mathbf X)$", "• JL lemma: random linear projections approximately\n  preserve distances among finite points,\n  dimension about $O(\\log n/\\epsilon^2)$;\n  not a proof about network initialization\n• SDR / KDR: is there $B$ with\n  $P(Y\\mid\\mathbf X)=P(Y\\mid B^{\\top}\\mathbf X)$?"), size=19, color=INK, bold=True, z=8)
    g.beat = 3
    g.text(50, 548, 1180, 70, T("降維、直接建模與其他表示，要一起在沒用來擬合的資料上比較", "Compare reduction, direct modeling and other representations on data not used for fitting"), size=22, color=INK, bold=True, align="center", z=8)
    banner(g, T(r"下一站 KP08：神經網路如何把權重變成可學的參數 $\to$", r"Next, KP08: how a network turns weights into learnable parameters $\to$"))
    pages.append((g, T("不是通用流程", "Not a universal recipe"), {
        0: T("常見迷思：把先降維再回歸寫成通用流程。降維可能丟掉對標籤最有用、但總變異很小的方向，回看 KP05 的 8 點。", "Common misconception: writing reduce-then-regress as a universal recipe. Reduction can drop a direction that is most useful for the label but has little total variance; recall KP05's 8 points."),
        1: T("另外兩個定位：JL 引理對有限點集的隨機線性投影給出近似保距保證，不是神經網路初始化或訓練有效性的直接證明。充分維度縮減問是否存在投影矩陣 B，使條件分布只依賴投影後的資料，這是保留預測資訊的目標，不是自動找到最小維度。", "Two more pieces: the JL lemma gives approximate distance preservation for random linear projections of finite point sets, not a direct proof about network initialization or training. Sufficient dimension reduction asks whether a projection B exists so the conditional distribution depends only on the projected data, a goal of preserving predictive information, not automatically the smallest dimension."),
        2: T("應與直接建模及其他表示一起，在未見資料上比較。", "Compare with direct modeling and other representations on unseen data."),
        3: T("口訣：先問要預測什麼，再問要保留什麼；係數懲罰與表示準則各有任務。下一個單元談神經網路。", "Rule of thumb: ask what to predict and what to preserve; penalties and representation criteria have different jobs. The next unit covers neural networks.")}))
    return pages
