import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *
from lec01_kp06_nums import *

KP = "KP06"; N = 10

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)
    ell = lambda g, cx, cy, rx, ry, col, z=3, fill="transparent", sw=4, dash=False: g.ellipse(cx, cy, rx, ry, fill=fill, stroke=col, sw=sw, z=z, **({"borderStyle": "dashed"} if dash else {}))

    # ============ 1:六個家族 ============
    g = new(1, T("KP06 分類工具各有不同假設", "KP06 Classifiers make different assumptions"))
    g.beat = 1
    g.rect(470, 130, 340, 90, fill=BAN, radius=18, z=3, shadowColor="rgba(30,58,138,0.35)", shadowBlur=12, shadowOffsetY=4); g.text(470, 130, 340, 90, T("監督式分類", "supervised classification"), size=26, color="#ffffff", bold=True, align="center", vcenter=True, z=8)
    fams = [(BLUE, BLUEL, T("貝氏 / LDA / QDA", "Bayes / LDA / QDA"), T("為每類描述資料密度", "model each class's density")), (ORG, ORGL, T("羅吉斯迴歸", "Logistic regression"), T("直接建類別機率", "model class probability directly")),
            (GRN, GRNL, T("KNN / 度量", "KNN / metrics"), T("比較鄰居", "compare neighbors")), (PUR, PURL, T("樹 / 隨機森林", "Trees / random forest"), T("逐步切分空間", "split space step by step")),
            (RED, REDL, "SVM / " + T("核方法", "kernels"), T("尋找間隔邊界", "find a margin boundary")), (GOLD, "#fef3c7", T("集成 / Boosting", "Ensembles / boosting"), T("組合多個模型", "combine several models"))]
    for i, (col, light, nm, sub) in enumerate(fams):
        g.beat = 2 if i < 3 else 3
        x = 50 + i * 200
        g.line([(640, 220), (640, 260), (x + 90, 260), (x + 90, 300)], GRAY, sw=4, z=2)
        g.rect(x, 300, 180, 220, fill=light, stroke=col, sw=3, radius=14, z=3); g.text(x + 6, 316, 168, 90, nm, size=22 if zh else 17, color=col if col != GOLD else "#92400e", bold=True, align="center", z=8); g.text(x + 8, 410, 164, 100, sub, size=18, color=INK, align="center", z=8)
    g.beat = 3
    g.text(50, 540, 1180, 80, T("六個家族互相重疊：集成方法本身常以「樹」為基底", "The six families overlap: ensembles are often built from trees"), size=22, color=MUTED, bold=True, align="center", z=8)
    banner(g, T("沒有一個家族，對所有資料集都最好", "No family is best on every dataset"))
    pages.append((g, T("六個工具家族", "Six families of tools"), {
        0: T("KP03 定義了輸出類別這項任務，但沒有指定演算法。這個單元比較分類工具的假設。", "KP03 defined the task of outputting a class but not the algorithm. This unit compares the assumptions of classifiers."),
        1: T("前三個：貝氏類的方法為每一類描述資料密度；羅吉斯迴歸直接描述類別機率；KNN 比較鄰居。", "The first three: Bayes-type methods model each class's density, logistic regression models class probability directly, KNN compares neighbors."),
        2: T("後三個：樹逐步切分空間；SVM 尋找間隔邊界；集成方法組合多個模型。連線只是概念分組，不是歷史衍生關係。", "The other three: trees split space step by step, SVMs find a margin boundary, ensembles combine models. The lines are conceptual groups, not historical lineage."),
        3: T("這六個家族互相重疊，沒有任何家族對所有資料集都最好。", "These families overlap, and none is best on every dataset.")}))

    # ============ 2:先驗、密度、後驗 ============
    g = new(2, T("貝氏：先驗 × 密度，再歸一化成後驗", "Bayes: prior × density, then normalize to a posterior"))
    g.beat = 1
    card(g, 90, 110, 1100, 120, GRAY); g.text(110, 108, 1060, 130, r"$P(Y=j\mid X=x)=\frac{\pi_j f_j(x)}{\sum_{k=1}^{K}\pi_k f_k(x)}$", size=44, color=INK, align="center", vcenter=True, z=8)
    items = [(BLUE, BLUEL, r"$\pi_j$", T("先驗", "prior"), T("看特徵之前的類別比例", "class share before seeing features")), (ORG, ORGL, r"$f_j(x)$", T("類條件密度", "class-conditional density"), T("第 $j$ 類中，$x$ 附近有多集中", "how concentrated near $x$ within class $j$")), (GRN, GRNL, r"$P(Y=j\mid X=x)$", T("後驗", "posterior"), T("看過特徵之後的類別機率", "class probability after seeing $x$"))]
    for i, (col, light, sym, nm, sub) in enumerate(items):
        g.beat = i + 1 if i < 2 else 3
        x = 60 + i * 400
        g.rect(x, 270, 360, 300, fill=light, stroke=col, sw=3, radius=16, z=3); g.text(x, 280, 360, 80, sym, size=32 if i < 2 else 26, color=col, bold=True, align="center", vcenter=True, z=8); g.text(x, 370, 360, 50, nm, size=24, color=INK, bold=True, align="center", vcenter=True, z=8); g.text(x + 20, 430, 320, 120, sub, size=20, color=INK, align="center", z=8)
    arrow(g, (425, 420), (455, 420), GRAY, sw=4, head=14); arrow(g, (825, 420), (855, 420), GRAY, sw=4, head=14)
    g.beat = 3
    banner(g, T("密度不是點機率：連續的 $P(X=x)=0$", "A density is not a point probability: continuous $P(X=x)=0$"))
    pages.append((g, T("先驗、密度、後驗", "Prior, density, posterior"), {
        0: T("對連續的 X 與有限個類別：後驗等於先驗乘密度，再除以所有類別加總。", "For continuous X and finite classes: the posterior is the prior times the density, divided by the sum over all classes."),
        1: T("先驗 pi j：看特徵之前的類別比例。", "The prior pi j: the class share before seeing the features."),
        2: T("類條件密度 f j：在第 j 類中，某個位置附近有多集中。密度可以大於 1。", "The class-conditional density f j: how concentrated the class is near a position. A density can exceed 1."),
        3: T("後驗：看過特徵之後的類別機率。連續分布的 P(X=x) 等於 0，來源把密度寫成 P(X=x) 會混淆點機率與密度。若後驗正確、各類誤判代價相同，取最大後驗有最小的預期誤判率，但不保證每筆都判對。", "The posterior: class probability after seeing the features. For a continuous distribution P(X=x) is 0; writing a density as P(X=x) mixes up point probability and density. With correct posteriors and equal costs, picking the maximum has the smallest expected error, though not every case is right.")}))

    # ============ 3:圓度例的密度 ============
    g = new(3, T("圓度 $x=0.7$：兩個類的密度差很多", "Roundness $x=0.7$: the two class densities differ hugely"))
    g.beat = 1
    card(g, 50, 120, 760, 480, BLUE); head(g, 70, 126, 700, T("類條件密度 $f_1,f_2$（縱軸是密度）", "Class densities $f_1,f_2$ (the vertical axis is density)"), BLUE, 22)
    ox, oy, W, H = 110, 560, 660, 340
    X = lambda v: ox + v * W; Y = lambda d: oy - d / 4.2 * H
    g.line([(ox, oy), (ox + W, oy)], MUTED, sw=3, z=3)
    for t in (0, 0.2, 0.4, 0.6, 0.8, 1.0): g.line([(X(t), oy), (X(t), oy + 6)], MUTED, sw=2, z=3); g.text(X(t) - 24, oy + 8, 48, 24, f"{t:.1f}", size=16, color=MUTED, align="center", z=8)
    xs_ = [i / 100 for i in range(0, 101, 2)]
    g.line([(X(v), Y(pdf(v, 0.2, 0.1))) for v in xs_], ORG, sw=5, smooth=True, z=5); g.line([(X(v), Y(pdf(v, 0.8, 0.1))) for v in xs_], BLUE, sw=5, smooth=True, z=5)
    g.text(X(0.2) - 100, Y(pdf(0.2, 0.2, 0.1)) - 40, 200, 30, T("黃瓜 $\\mu_2=0.2$", "cucumber $\\mu_2=0.2$"), size=18, color=ORG, bold=True, align="center", z=8); g.text(X(0.8) - 100, Y(pdf(0.8, 0.8, 0.1)) - 40, 200, 30, T("蘋果 $\\mu_1=0.8$", "apple $\\mu_1=0.8$"), size=18, color=BLUE, bold=True, align="center", z=8)
    g.beat = 2
    g.line([(X(0.7), oy), (X(0.7), Y(5.0) if False else oy - 380)], RED, sw=4, dash=True, z=4); g.text(X(0.7) + 6, oy - 392, 120, 28, r"$x=0.7$", size=20, color=RED, bold=True, z=8)
    g.ellipse(X(0.7), Y(f1), 9, fill=BLUE, stroke="#ffffff", sw=2, z=7); g.ellipse(X(0.7), Y(f2), 9, fill=ORG, stroke="#ffffff", sw=2, z=7)
    g.beat = 3
    card(g, 850, 120, 380, 480, GRAY); head(g, 870, 126, 340, T("代入密度公式", "Plug into the density"), MUTED, 22)
    g.rect(870, 190, 340, 120, fill=BLUEL, stroke=BLUE, sw=3, radius=12, z=3); g.text(880, 192, 320, 116, rf"$f_1(0.7)={f1:.9f}$".replace("2.419707245", "2.419707245") + "\n" + T("標準化距離 $-1$", "standardized distance $-1$"), size=20, color=INK, bold=True, align="center", z=8)
    g.rect(870, 330, 340, 120, fill=ORGL, stroke=ORG, sw=3, radius=12, z=3); g.text(880, 332, 320, 116, rf"$f_2(0.7)={f2:.9f}$" + "\n" + T("標準化距離 $5$", "standardized distance $5$"), size=20, color=INK, bold=True, align="center", z=8)
    g.text(870, 470, 340, 110, T("共同前因子\n$1/(0.1\\sqrt{2\\pi})=3.989422804$", "common factor\n$1/(0.1\\sqrt{2\\pi})=3.989422804$"), size=18, color=MUTED, align="center", z=8)
    banner(g, T("來源備課表的 0.24、0.0001 與高斯公式不合，這裡已重算", "The source's 0.24 and 0.0001 disagree with the Gaussian formula; recomputed here"))
    pages.append((g, T("圓度的密度", "Densities at x = 0.7"), {
        0: T("來源的例子：蘋果是第 1 類、黃瓜是第 2 類，量一個圓度 x。蘋果的平均是 0.8，黃瓜是 0.2，標準差都是 0.1，先驗各 0.5。縱軸是密度，不是單點機率。", "The source's example: apple is class 1, cucumber class 2, one roundness measurement x. Means 0.8 and 0.2, both standard deviations 0.1, priors 0.5 each. The vertical axis is density, not point probability."),
        1: T("圓度是 0.7：離蘋果的平均近，離黃瓜的平均遠。", "At roundness 0.7: near the apple mean, far from the cucumber mean."),
        2: T("蘋果的標準化距離是負 1，黃瓜的是 5。代入共同前因子 3.989422804，得到蘋果類的密度約 2.4197，黃瓜類約 0.0000149。", "The apple's standardized distance is minus 1, the cucumber's is 5. With the common factor 3.989422804, the apple density is about 2.4197 and the cucumber's about 0.0000149."),
        3: T("來源備課表把這組參數的密度寫成約 0.24 和 0.0001，後驗寫 0.9996 和 0.0004；與高斯密度公式不合，這裡保留參數、重新計算。", "The source's guide lists about 0.24 and 0.0001 and posteriors 0.9996 and 0.0004 for the same parameters, which disagree with the Gaussian formula; we keep the parameters and recompute.")}))

    # ============ 4:先驗加權、歸一化 ============
    g = new(4, T("先乘先驗，再除以證據：得到後驗", "Multiply by the prior, normalize: the posterior"))
    g.beat = 1
    for i, (col, light, nm, q, f_) in enumerate(((BLUE, BLUEL, T("蘋果", "apple"), q1, f1), (ORG, ORGL, T("黃瓜", "cucumber"), q2, f2))):
        y = 130 + i * 130
        g.rect(50, y, 1180, 110, fill=light, stroke=col, sw=3, radius=14, z=2)
        g.text(70, y + 6, 160, 98, nm, size=26, color=col, bold=True, vcenter=True, z=8)
        g.text(240, y + 6, 940, 98, rf"$q_{i + 1}=0.5\times{f_:.9f}={q:.12f}$".replace("0.000007433598", "0.000007433598") if i == 0 else rf"$q_2=0.5\times{f_:.12f}={q:.12f}$", size=26, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    arrow(g, (640, 366), (640, 396), GRAY, sw=4, head=14); star(g, 1160, 440, 16, GOLD, z=6)
    g.rect(50, 400, 1180, 80, fill=CARD, stroke=GRAY, sw=2, radius=14, z=2); g.text(70, 402, 1140, 76, rf"$f_X(0.7)=q_1+q_2={fx:.9f}$", size=30, color=INK, align="center", vcenter=True, z=8)
    g.beat = 3
    g.rect(50, 500, 1180, 40, fill="#e2e8f0", radius=8, z=2); g.rect(50, 500, 1180 * post1, 40, fill=BLUE, radius=8, z=3)
    g.text(50, 548, 1180, 40, rf"$P(Y=1\mid X=0.7)={post1:.9f},\quad P(Y=2\mid X=0.7)={post2:.9f}$", size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
    arrow(g, (1100, 600), (1190, 520), RED, sw=3, head=12); g.text(930, 596, 200, 30, T("黃瓜只佔極小一點", "cucumber: a sliver"), size=16, color=RED, bold=True, z=8)
    banner(g, T("比較類別可略去共同分母，但不可略去不同的先驗", "Common denominators may be dropped when comparing; different priors may not"))
    pages.append((g, T("後驗", "The posterior"), {
        0: T("先乘先驗：蘋果 q1 等於 0.5 乘 2.4197，約 1.2099；黃瓜 q2 約 0.0000074。", "Multiply by the prior: apple q1 is 0.5 times 2.4197, about 1.2099; cucumber q2 about 0.0000074."),
        1: T("加起來是證據密度，約 1.2099。", "The sum is the evidence density, about 1.2099."),
        2: T("歸一化：蘋果的後驗約 0.999994，黃瓜約 0.000006，兩個和為 1。比較類別時可以略去共同分母，但不可略去不同的先驗。", "Normalize: the apple posterior is about 0.999994 and the cucumber about 0.000006, summing to 1. When comparing classes you may drop the common denominator but not different priors."),
        3: T("所有數值由程式重算，並和講義的值逐位核對。", "All values were recomputed by program and checked digit by digit against the note.")}))

    # ============ 5:平手 ============
    g = new(5, T("兩類一樣可能：後驗各 0.5，還要定平手規則", "Equally likely: posteriors 0.5, need a tie rule"))
    g.beat = 1
    card(g, 50, 120, 760, 480, BLUE); head(g, 70, 126, 700, T(r"互動預設：$\mu_1=3,\ \mu_2=7,\ \sigma=1$", r"Interactive default: $\mu_1=3,\ \mu_2=7,\ \sigma=1$"), BLUE, 22)
    ox, oy, W, H = 110, 560, 660, 330
    X = lambda v: ox + v / 10 * W; Y = lambda d: oy - d / 0.45 * H
    g.line([(ox, oy), (ox + W, oy)], MUTED, sw=3, z=3)
    for t in (0, 2, 4, 6, 8, 10): g.line([(X(t), oy), (X(t), oy + 6)], MUTED, sw=2, z=3); g.text(X(t) - 20, oy + 8, 40, 24, str(t), size=16, color=MUTED, align="center", z=8)
    xs_ = [i / 10 for i in range(0, 101, 2)]
    g.line([(X(v), Y(pdf(v, 3, 1))) for v in xs_], BLUE, sw=5, smooth=True, z=5); g.line([(X(v), Y(pdf(v, 7, 1))) for v in xs_], ORG, sw=5, smooth=True, z=5)
    g.beat = 2
    g.line([(X(5), oy), (X(5), oy - 380)], RED, sw=4, dash=True, z=4); g.text(X(5) + 6, oy - 392, 100, 28, r"$x=5$", size=20, color=RED, bold=True, z=8)
    g.ellipse(X(5), Y(t1), 10, fill=GOLD, stroke="#b45309", sw=3, z=7)
    g.beat = 3
    card(g, 850, 120, 380, 480, GRAY); head(g, 870, 126, 340, T("兩邊相同", "Both equal"), MUTED, 22)
    g.text(870, 190, 340, 140, rf"$f_1(5)=f_2(5)={t1:.9f}$", size=22, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.text(870, 340, 340, 80, rf"$q_j={tq:.9f}$", size=24, color=INK, align="center", vcenter=True, z=8)
    g.text(870, 430, 340, 70, T(r"後驗各 $0.5$", r"posteriors $0.5$ each"), size=26, color=GRN, bold=True, align="center", vcenter=True, z=8)
    g.text(870, 510, 340, 80, T("要輸出一個類別，\n得明定平手規則", "to output one class,\na tie rule is needed"), size=18, color=MUTED, align="center", z=8)
    banner(g, T("兩類離 $x=5$ 一樣遠：資料沒有偏向哪一類", "Both classes are equally far from $x=5$: the data favor neither"))
    pages.append((g, T("平手", "A tie"), {
        0: T("來源互動元件另有一組不同的預設：平均 3 與 7，標準差 1，先驗各 0.5，x 等於 5。", "The interactive component has another default: means 3 and 7, standard deviation 1, priors 0.5 each, x equal to 5."),
        1: T("x 等於 5：兩類距各自的平均都是 2。", "At x equal to 5, both classes are 2 away from their means."),
        2: T("所以兩個密度都是 0.053990967，各自加權後 0.026995483，證據 0.053990967，後驗各 0.5。這是平手；若要輸出一個類別，仍需明定平手規則。", "So both densities equal 0.053990967, each weighted density is 0.026995483, the evidence is 0.053990967, and each posterior is 0.5. It is a tie; outputting one class still needs a tie rule."),
        3: T("這組值同樣由程式重算。", "These values were also recomputed by program.")}))

    # ============ 6:LDA vs QDA ============
    g = new(6, T("共變異數的假設，決定邊界是直線還是曲線", "Covariance sets a straight or curved boundary"))
    pill(g, 890, 62, 140, T("講義補充", "Added note"), GRAY, size=16)
    S6 = 70
    def panel(x0, col, light, ttl, sig1, sig2, boundary, beat):
        g.beat = beat
        card(g, x0, 120, 580, 480, col); head(g, x0 + 20, 126, 540, ttl, col, 22)
        cx, cy = x0 + 290, 390
        g.line([(x0 + 40, cy), (x0 + 540, cy)], MUTED, sw=3, z=2); g.line([(cx, cy + 190), (cx, cy - 190)], MUTED, sw=3, z=2)
        for (mx, sg, c) in ((-1, sig1, BLUE), (1, sig2, ORG)):
            ell(g, cx + mx * S6, cy, math.sqrt(sg[0]) * S6, math.sqrt(sg[1]) * S6, c, z=4, fill="rgba(29,78,216,0.08)" if c == BLUE else "rgba(194,65,12,0.08)")
            g.ellipse(cx + mx * S6, cy, 7, fill=c, stroke="#ffffff", sw=2, z=6)
        boundary(cx, cy)
    def lda_b(cx, cy): g.line([(cx, cy - 180), (cx, cy + 180)], RED, sw=6, z=6)
    def qda_b(cx, cy):
        ok = lambda p: abs(p[0] * S6) < 255 and abs(p[1] * S6) < 185
        L_ = sorted([(x, y) for x, y in QPTS if x < -1.0 and ok((x, y))], key=lambda p: p[1]); R_ = sorted([(x, y) for x, y in QPTS if x >= -1.0 and ok((x, y))], key=lambda p: p[1])
        for br in (L_, R_):
            if len(br) >= 2: g.line([(cx + x * S6, cy - y * S6) for x, y in br], RED, sw=6, smooth=True, z=6)
    panel(50, BLUE, BLUEL, T(r"LDA：共用 $\Sigma$，邊界是直線 $x_1=0$", r"LDA: shared $\Sigma$, straight boundary $x_1=0$"), LDA_S, LDA_S, lda_b, 1)
    panel(660, ORG, ORGL, T(r"QDA：各自 $\Sigma$，邊界是曲線", r"QDA: separate $\Sigma$, curved boundary"), Q1, Q2, qda_b, 2)
    g.beat = 3
    g.text(50, 606, 1180, 30, "", size=16, z=1)
    banner(g, T("兩個圓點只標示平均，不是訓練樣本：這裡只看共變異數假設的影響", "The dots mark means, not training samples: only the covariance assumption varies"))
    pages.append((g, T("LDA 與 QDA", "LDA and QDA"), {
        0: T("這是講義的補充：固定兩類平均 (−1, 0) 和 (1, 0)、相等先驗，只改共變異數。圖上的圓點只標示平均，不是訓練樣本。", "A supplement: fix the two class means (−1, 0) and (1, 0) and equal priors, and vary only the covariance. The dots mark means, not training samples."),
        1: T("LDA 讓兩類共用同一個共變異數 diag(1, 0.5)：二次項抵消，等後驗邊界是直線 x1 等於 0。", "LDA lets both classes share the covariance diag(1, 0.5): the quadratic terms cancel and the equal-posterior boundary is the straight line x1 equal to 0."),
        2: T("QDA 讓每類有自己的共變異數：這裡是 diag(0.4, 0.8) 和 diag(1.8, 0.3)。邊界變成曲線。邊界由兩個對數分數相等解出，沒有擬合資料。", "QDA gives each class its own covariance, here diag(0.4, 0.8) and diag(1.8, 0.3). The boundary becomes curved. It is solved from equal log-scores, with no data fitted."),
        3: T("FDA 是選投影方向的判別準則；二類共用共變異數的高斯模型下，它的方向可與 LDA 對應，但兩者不是所有情境下完全相同的程序。", "FDA is a criterion for choosing a projection direction; under a two-class shared-covariance Gaussian model its direction corresponds to LDA's, but they are not identical procedures in every setting.")}))

    # ============ 7:Logistic ============
    g = new(7, T("羅吉斯迴歸：不經密度，直接建類別機率", "Logistic regression: model the class probability directly"))
    pill(g, 900, 96, 330, T("示意（參數 $b=0,\\ \\beta=1$）", "sketch ($b=0,\\ \\beta=1$)"), GRAY, size=16)
    g.beat = 1
    card(g, 50, 140, 1180, 90, GRAY); g.text(70, 134, 1140, 104, r"$P(Y=1\mid\mathbf x)=\frac{1}{1+e^{-(b+\boldsymbol\beta^{\top}\mathbf x)}}$", size=40, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    ox, oy, W, H = 200, 580, 880, 290
    X = lambda v: ox + (v + 6) / 12 * W; Y = lambda p: oy - p * H
    g.line([(ox, oy), (ox + W, oy)], MUTED, sw=3, z=3); g.line([(ox, oy), (ox, oy - H - 10)], MUTED, sw=3, z=3)
    for t in (0, 1): g.text(ox - 50, oy - t * H - 14, 40, 28, str(t), size=18, color=MUTED, align="right", z=8)
    g.line([(ox, Y(0.5)), (ox + W, Y(0.5))], GRAY, sw=2, dash=True, z=2)
    g.line([(X(v / 2), Y(1 / (1 + math.exp(-v / 2)))) for v in range(-12, 13)], ORG, sw=6, smooth=True, z=5)
    g.beat = 3
    g.ellipse(X(0), Y(0.5), 12, fill=GOLD, stroke="#b45309", sw=3, z=7)
    g.text(X(0) + 30, Y(0.5) + 14, 280, 34, T("機率 $0.5$ 的分界", "the $0.5$ boundary"), size=20, color=INK, bold=True, z=8)
    g.text(ox + 30, 270, 300, 40, T("基本式是二類", "the basic form is binary"), size=20, color=MUTED, bold=True, z=8)
    banner(g, T("好處：直接問「$P(Y=1\\mid\\mathbf x)$」；基本式是二類線性 log-odds", "It models $P(Y=1\\mid\\mathbf x)$ directly; the basic form is binary with linear log-odds"))
    pages.append((g, T("羅吉斯迴歸", "Logistic regression"), {
        0: T("羅吉斯迴歸不先建每類的密度，直接用 sigmoid 建二類的後驗機率。基本式是二類，可另外擴充成多類。", "Logistic regression skips the class densities and models the binary posterior with a sigmoid. The basic form is binary; multi-class is an extension."),
        1: T("輸出是 0 到 1 之間的機率。這條曲線是示意，參數取 b 等於 0、beta 等於 1，不是訓練結果。", "The output is a probability between 0 and 1. This curve is a sketch with b equal to 0 and beta equal to 1, not a training result."),
        2: T("機率 0.5 的位置就是分界。", "The point where the probability is 0.5 is the boundary."),
        3: T("基本式是二類線性的 log-odds。", "The basic form has linear log-odds in the binary case.")}))

    # ============ 8:KNN 與 樹 ============
    g = new(8, T("KNN 看鄰居投票，樹逐層問是非", "KNN votes among neighbors, trees ask yes-no questions"))
    pill(g, 960, 96, 270, T("示意：不是講義的資料", "sketch: not the note's data"), GRAY, size=16)
    g.beat = 1
    card(g, 50, 140, 590, 460, GRN); head(g, 70, 146, 550, T("KNN：$k=3$ 個最近鄰居投票", "KNN: $k=3$ nearest neighbors vote"), GRN, 22)
    pts_ = [(130, 280, BLUE), (200, 340, BLUE), (150, 420, BLUE), (260, 460, BLUE), (460, 270, ORG), (530, 350, ORG), (480, 450, ORG), (370, 390, ORG), (330, 300, BLUE)]
    q = (340, 360)
    for x, y, c in pts_: g.ellipse(x, y, 13, fill=c, stroke="#ffffff", sw=2, z=5)
    g.beat = 2
    near3 = sorted(pts_, key=lambda p: math.hypot(p[0] - q[0], p[1] - q[1]))[:3]
    r = max(math.hypot(p[0] - q[0], p[1] - q[1]) for p in near3) + 8
    g.ellipse(q[0], q[1], r, stroke=GRN, sw=4, z=4, fill="rgba(21,128,61,0.08)"); star(g, q[0], q[1], 17, GOLD, z=7, stroke="#b45309", sw=2)
    cnt_b = sum(1 for p in near3 if p[2] == BLUE); cnt_o = 3 - cnt_b
    g.text(70, 530, 550, 50, T(f"投票：藍 {cnt_b}、橘 {cnt_o}", f"vote: blue {cnt_b}, orange {cnt_o}"), size=22, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    card(g, 680, 140, 550, 460, PUR); head(g, 700, 146, 510, T("決策樹：逐層用閾值分割", "Decision tree: split by thresholds"), PUR, 22)
    g.poly([(955, 215), (1035, 255), (955, 295), (875, 255)], fill=PURL, stroke=PUR, sw=3, z=3); g.text(895, 235, 120, 40, r"$x_1\le t$?", size=20, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.line([(915, 275), (815, 370)], GRAY, sw=4, arrow="end", head=14, z=4); g.line([(995, 275), (1095, 370)], GRAY, sw=4, arrow="end", head=14, z=4)
    g.poly([(815, 400), (860, 440), (815, 480), (770, 440)], fill=PURL, stroke=PUR, sw=3, z=3); g.text(775, 420, 80, 40, r"$x_2\le s$?", size=16, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.rect(1020, 380, 150, 90, fill=ORGL, stroke=ORG, sw=3, radius=12, z=3); g.text(1020, 380, 150, 90, T("橘", "orange"), size=24, color=ORG, bold=True, align="center", vcenter=True, z=8)
    g.rect(700, 520, 120, 60, fill=BLUEL, stroke=BLUE, sw=3, radius=10, z=3); g.text(700, 520, 120, 60, T("藍", "blue"), size=22, color=BLUE, bold=True, align="center", vcenter=True, z=8)
    g.rect(840, 520, 120, 60, fill=ORGL, stroke=ORG, sw=3, radius=10, z=3); g.text(840, 520, 120, 60, T("橘", "orange"), size=22, color=ORG, bold=True, align="center", vcenter=True, z=8)
    g.line([(790, 475), (760, 520)], GRAY, sw=3, z=4); g.line([(835, 475), (890, 520)], GRAY, sw=3, z=4)
    banner(g, T("KNN 受距離、特徵尺度與 $k$ 影響；單棵樹易受資料擾動", "KNN depends on distance, scaling and $k$; a single tree is sensitive to data changes"))
    pages.append((g, T("KNN 與決策樹", "KNN and decision trees"), {
        0: T("這兩頁的圖是示意，不是講義的資料。KNN 依距離找 k 個最近鄰居，再依鄰居標籤投票。", "These sketches are not the note's data. KNN finds the k nearest neighbors by distance and votes on their labels."),
        1: T("星號是要分類的點，圈出 3 個最近鄰居，多數決定類別。「不擬合參數」不等於沒有模型假設：距離、特徵尺度和 k 都會影響結果。", "The star is the point to classify; the circle holds its 3 nearest neighbors and the majority decides. Not fitting parameters does not mean no assumptions: distance, feature scaling and k all matter."),
        2: T("決策樹逐層用特徵的閾值把空間分開。單棵樹容易受資料擾動，深度可以控制複雜度。", "A decision tree splits the space level by level with feature thresholds. A single tree is sensitive to data changes, and depth controls complexity."),
        3: T("CART、ID3、C4.5 等是決策樹脈絡的歷史名稱。", "CART, ID3 and C4.5 are historical names within the decision-tree family.")}))

    # ============ 9:SVM 與集成 ============
    g = new(9, T("SVM 找間隔最寬的邊界，集成方法合併多個模型", "SVMs seek a wide margin; ensembles combine models"))
    pill(g, 960, 96, 270, T("示意：不是講義的資料", "sketch: not the note's data"), GRAY, size=16)
    g.beat = 1
    card(g, 50, 140, 590, 460, RED); head(g, 70, 146, 550, T("SVM：在間隔與錯誤之間取捨", "SVM: trade margin against errors"), RED, 22)
    g.line([(190, 540), (500, 220)], RED, sw=6, z=5); g.line([(130, 520), (440, 200)], RED, sw=3, dash=True, z=4); g.line([(250, 560), (560, 240)], RED, sw=3, dash=True, z=4)
    for (x, y) in [(130, 300), (200, 260), (170, 380), (280, 330), (140, 440)]: g.ellipse(x + 10, y, 13, fill=BLUE, stroke="#ffffff", sw=2, z=6)
    for (x, y) in [(400, 480), (480, 420), (520, 520), (350, 540), (540, 330)]: g.ellipse(x, y, 13, fill=ORG, stroke="#ffffff", sw=2, z=6)
    for (x, y) in [(170, 380), (350, 540)]: g.ellipse(x + (10 if x < 300 else 0), y, 22, stroke=GOLD, sw=5, z=7)
    g.text(300, 156, 330, 34, T("圈出的是支持向量", "circled: support vectors"), size=18, color=MUTED, bold=True, align="right", z=8)
    g.beat = 2
    g.text(70, 555, 550, 1, "", size=16, z=1)
    g.beat = 3
    card(g, 680, 140, 550, 460, GOLD); head(g, 700, 146, 510, T("集成：組合多個模型", "Ensembles: combine models"), "#92400e", 22)
    for i in range(3):
        x = 720 + i * 150
        g.poly([(x + 40, 230), (x + 80, 280), (x, 280)], fill=PURL, stroke=PUR, sw=3, z=3); g.line([(x + 25, 280), (x + 10, 330)], PUR, sw=3, z=3); g.line([(x + 55, 280), (x + 70, 330)], PUR, sw=3, z=3)
        g.ellipse(x + 8, 345, 14, fill=ORG, z=4); g.ellipse(x + 72, 345, 14, fill=BLUE, z=4)
        g.line([(x + 40, 370), (x + 40, 420)], GRAY, sw=3, arrow="end", head=12, z=4)
    g.rect(740, 430, 440, 70, fill="#fef3c7", stroke="#b45309", sw=3, radius=12, z=3); g.text(740, 430, 440, 70, T("聚合（投票／加權）", "aggregate (vote / weights)"), size=22, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.text(700, 520, 510, 70, T("隨機森林常以重抽樣建多棵樹；Boosting 依序修正前一個的不足", "forests resample to grow many trees; boosting fixes the previous learner's gaps in turn"), size=18, color=INK, align="center", z=8)
    banner(g, T("核函數不保證可分，也不保證普遍更準", "Kernels guarantee neither separability nor universally better accuracy"))
    pages.append((g, T("SVM 與集成", "SVM and ensembles"), {
        0: T("SVM 在間隔與錯誤之間取捨。實線是分界，虛線是間隔邊緣，圈出的點是支持向量。這是硬間隔、線性可分的簡化圖；軟間隔的支持向量也可能在間隔內甚至被分錯，所以「只在乎最靠近邊界的點」需要加條件。", "An SVM trades margin against errors. The solid line is the boundary, dashed lines the margin edges, circled points the support vectors. This is the hard-margin, separable simplification; soft-margin support vectors can lie inside the margin or even be misclassified, so only caring about points nearest the boundary needs a condition."),
        1: T("核函數用隱含特徵空間的內積，不必顯式算高維座標，但需要合適的核；核不保證可分，也不保證普遍更準。", "Kernels use inner products in an implicit feature space without computing high-dimensional coordinates, but a suitable kernel is needed; they guarantee neither separability nor universally better accuracy."),
        2: T("集成方法組合多個模型。隨機森林常以重抽樣與特徵隨機化建多棵樹再聚合；Boosting 依序修正前面的不足，兩者不同。", "Ensemble methods combine models. Random forests often grow many trees by resampling and feature randomization then aggregate; boosting fixes earlier shortfalls in turn. They differ."),
        3: T("樹與集成是重疊的概念。", "Trees and ensembles are overlapping concepts.")}))

    # ============ 10 ============
    g = new(10, T("先問「怎樣產生分數」，再問「哪些假設可被資料挑戰」", "How are scores produced? Which assumptions fail?"))
    g.beat = 1
    for i, (col, light, ttl, sub) in enumerate(((BLUE, BLUEL, T("怎樣產生分數？", "How are scores produced?"), T("密度、直接機率、鄰居、切分、間隔、組合", "densities, direct probability, neighbors, splits, margins, ensembles")), (ORG, ORGL, T("哪些假設可被挑戰？", "Which assumptions can be challenged?"), T("共變異數、獨立、距離與尺度、$k$、核", "covariance, independence, distance and scale, $k$, kernels")))):
        g.beat = i + 1
        x = 60 + i * 600
        g.rect(x, 150, 560, 300, fill=light, stroke=col, sw=3, radius=18, z=3); g.text(x + 20, 170, 520, 90, ttl, size=30 if zh else 25, color=col, bold=True, align="center", z=8); g.text(x + 30, 270, 500, 150, sub, size=22, color=INK, bold=True, align="center", z=8)
        star(g, x + 280, 130, 16, GOLD, z=6)
    g.beat = 3
    g.text(60, 480, 1160, 100, T("評估選擇時，要看未參與擬合的資料：沒有任何家族因名稱、稀疏性或理論最優而在所有資料上必勝", "Judge choices on data not used for fitting: no family wins everywhere because of its name, sparsity or theoretical optimality"), size=24, color=INK, bold=True, align="center", z=8)
    mascot(g, 1100, 440, 0.0 + 0.45, "proud") if False else None
    banner(g, T(r"下一站 KP07：連續值預測與降維 $\to$", r"Next, KP07: numerical prediction and dimensionality reduction $\to$"))
    pages.append((g, T("口訣與銜接", "Rule of thumb"), {
        0: T("口訣：先問怎樣產生分數，再問哪些假設可以被資料挑戰。", "Rule of thumb: ask how scores are produced, then which assumptions the data can challenge."),
        1: T("每個家族產生分數的方式不同，也各有可以被檢驗的假設。", "Each family produces scores differently and has its own checkable assumptions."),
        2: T("常見迷思：真實的類條件密度難估，不代表 Bayes 算不出來；玩具例可以直接算，高維估計困難才是實務問題。", "Misconception: a hard-to-estimate class density does not mean Bayes cannot be computed; toy examples can be computed directly, and high-dimensional estimation is the practical problem."),
        3: T("講義的 4 題自我檢核留在講義。下一個單元轉向連續值預測與降維。", "The note's 4 self-check questions stay in the note. The next unit turns to numerical prediction and dimensionality reduction.")}))
    return pages
