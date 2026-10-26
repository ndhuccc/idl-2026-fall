"""KP22 (robust regression and ridge) -- eighth sample of view-from-text-v2. Plan: kp22_plan.json. Every number is recomputed here with kp22_nums.py
(same settings as the lecture; verified against it)."""
import os, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools")); sys.path.insert(0, HERE)
from figkit.parts import T, NFig, header_chip, panel, chip, outline_box, takeaway, tint, solid, INK
import kp22_nums as N


def tr(zh, en): return {"zh": zh, "en": en}


def hu(x, nd):   # half-up rounding to nd decimals
    return str(Decimal(repr(round(float(x), 9))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


XC, YC = N.trend(out=False); XO, YO = N.trend()
OC, OO = N.ols(XC, YC), N.ols(XO, YO); HB = N.huber_irls(XO, YO, 1.0)
OUT = (5.0, 21.0)


def outlier():
    ss = [N.ols(*N.trend(seed=s, out=False))[0] for s in range(3)]; so = [N.ols(*N.trend(seed=s))[0] for s in range(3)]
    spec = {
        "id": "lec05-kp22-outlier",
        "alt": tr(f"20 個點沿著 y=2x+1（雜訊標準差 0.3），再加一個離群點 (5,21)，比趨勢線的 11 高出 10。最小平方：乾淨資料斜率 {hu(OC[0],3)}、截距 {hu(OC[1],3)}；加了離群點斜率 {hu(OO[0],3)}、截距 {hu(OO[1],3)}。換三組雜訊（種子 0、1、2）斜率 " + "、".join(hu(v, 2) for v in so) + "，乾淨資料 " + "、".join(hu(v, 3) for v in ss) + "。",
                  f"20 points along y=2x+1 (noise standard deviation 0.3) plus an outlier at (5,21), 10 above the trend line's 11. Least squares: clean slope {hu(OC[0],3)}, intercept {hu(OC[1],3)}; with the outlier slope {hu(OO[0],3)}, intercept {hu(OO[1],3)}. Over three noise draws (seeds 0, 1, 2) the slope is " + ", ".join(hu(v, 2) for v in so) + " against " + ", ".join(hu(v, 3) for v in ss) + " clean."),
        "plot": {"x": [-0.2, 6.2], "y": [-1.5, 24], "x_ticks": [0, 1, 2, 3, 4, 5], "y_ticks": [0, 5, 10, 15, 20], "x_label": "x", "y_label": "y", "w": 430, "h": 300},
        "series": [{"kind": "seg", "a": (0, OC[1]), "b": (5.4, OC[0] * 5.4 + OC[1]), "role": "blue", "width": 4}, {"kind": "seg", "a": (0, OO[1]), "b": (5.4, OO[0] * 5.4 + OO[1]), "role": "red", "width": 4},
                   {"kind": "dots", "points": [(float(a), float(b)) for a, b in zip(XC, YC)], "role": "blue", "d": 10}, {"kind": "dots", "points": [OUT], "role": "red", "d": 18, "shape": "square"}],
        "notes": [{"x": 5.0, "y": 23.2, "text": tr("離群點", "outlier"), "ha": "center", "w": 90}, {"x": 5.5, "y": OC[0] * 5.4 + OC[1] - 3.2, "text": tr("乾淨", "clean"), "ha": "center", "w": 80},
                  {"x": 3.6, "y": OO[0] * 3.6 + OO[1] + 3.6, "text": tr("有離群點", "with outlier"), "ha": "center", "w": 110}],
        "side": [{"title": tr("離群點", "Outlier"), "role": "red", "name": tr("term:離群點", "term:Outlier"), "h": 1.1, "text": tr("遠離其他點所屬趨勢的點（極端值或標錯的標籤）；\n位在 $x$ 範圍邊緣時影響更大。", "a point far from the trend of the others (an extreme value\nor a wrong label); it matters more at the edge of the $x$ range.")},
                 {"title": tr("最小平方的結果", "Least-squares result"), "role": "blue", "h": 1.0, "text": tr(f"斜率 {hu(OC[0],3)} → {hu(OO[0],3)}，\n截距 {hu(OC[1],3)} → {hu(OO[1],3)}（真值 2 與 1）。", f"slope {hu(OC[0],3)} → {hu(OO[0],3)},\nintercept {hu(OC[1],3)} → {hu(OO[1],3)} (truth 2 and 1).")},
                 {"title": tr("為什麼", "Why"), "role": "orange", "h": 1.0, "text": tr("平方讓誤差 10 的罰分是誤差 1 的 100 倍，\n解被這一個點主導。", "squaring makes an error of 10 cost 100 times an\nerror of 1, so this one point dominates the fit.")}],
        "takeaway": tr(f"一個離群點把最小平方斜率從 {hu(OC[0],2)} 拉到 {hu(OO[0],2)}（真值 2）。", f"One outlier drags the least-squares slope from {hu(OC[0],2)} to {hu(OO[0],2)} (truth 2)."),
    }
    return "plot_xy", spec


def losses():
    es = [i * 0.1 for i in range(-40, 41)]
    sq = lambda e: 0.5 * e * e
    h = lambda e: float(N.huber(np.array([e]))[0]); ei = lambda e: float(N.eins(np.array([e]))[0])
    spec = {
        "id": "lec05-kp22-losses",
        "alt": tr("三條損失曲線（殘差 e）：平方 ½e²、Huber（c=1）、ε-不敏感（ε=0.5）。小誤差附近接近重合（ε-不敏感在管內為 0），大誤差處平方飆升，Huber 與 ε-不敏感只是線性上升。e=3：4.5、2.5、2.5；e=10：50、9.5、9.5。",
                  "Three loss curves (residual e): squared ½e², Huber (c=1), ε-insensitive (ε=0.5). They nearly coincide for small errors (the ε-insensitive loss is 0 inside the tube) while for large errors the square shoots up and the other two only grow linearly. e=3: 4.5, 2.5, 2.5; e=10: 50, 9.5, 9.5."),
        "plot": {"x": [-4.4, 4.4], "y": [0, 9], "x_ticks": [-4, -2, 0, 2, 4], "y_ticks": [0, 2, 4, 6, 8], "x_label": tr("殘差 e", "residual e"), "y_label": tr("損失", "loss"), "w": 420, "h": 300},
        "series": [{"kind": "line", "points": [(e, sq(e)) for e in es if sq(e) <= 8.6], "role": "red", "width": 4}, {"kind": "line", "points": [(e, h(e)) for e in es], "role": "green", "width": 4}, {"kind": "line", "points": [(e, ei(e)) for e in es], "role": "blue", "width": 4}],
        "notes": [{"x": 3.55, "y": 8.35, "text": tr("平方", "squared"), "ha": "center", "w": 80}, {"x": -2.4, "y": 5.8, "text": tr("Huber 與 ε-不敏感\n大誤差處重合", "Huber and ε-insens.\ncoincide for large errors"), "ha": "center", "w": 190}],
        "side": [{"title": tr("穩健損失", "Robust loss"), "role": "green", "name": tr("term:穩健損失", "term:Robust loss"), "h": 0.8, "text": tr("對大誤差手下留情的損失函數，\n讓少數壞點不再主導解。", "a loss that goes easy on large errors, so a few\nbad points no longer dominate the fit.")},
                 {"title": tr("Huber 損失", "Huber loss"), "role": "green", "name": tr("term:Huber 損失", "term:Huber loss"), "h": 1.3, "text": tr("$|e|\\leq c$：$\\frac{1}{2}e^2$；$|e|>c$：$c|e|-\\frac{c^2}{2}$。\n小誤差用平方，大誤差改成線性。", "$|e|\\leq c$: $\\frac{1}{2}e^2$; $|e|>c$: $c|e|-\\frac{c^2}{2}$.\nsquare for small errors, linear for large ones.")},
                 {"title": tr("ε-不敏感損失", "ε-insensitive loss"), "role": "blue", "name": tr("term:ε-不敏感損失", "term:ε-insensitive loss"), "h": 1.1, "text": tr("$\\max(0,|e|-\\epsilon)$：不超過 $\\epsilon$ 的\n誤差不罰，超出的部分線性增長。", "$\\max(0,|e|-\\epsilon)$: errors up to $\\epsilon$ cost\nnothing, the excess grows linearly.")},
                 {"title": tr("數值", "Numbers"), "role": "orange", "h": 0.9, "text": tr("$e=3$：平方 4.5、Huber 2.5、ε-不敏感 2.5；\n$e=10$：50、9.5、9.5。", "$e=3$: square 4.5, Huber 2.5, ε-insensitive 2.5;\n$e=10$: 50, 9.5, 9.5.")}],
        "takeaway": tr("大誤差處平方飆升，Huber 與 ε-不敏感只線性上升：離群點的影響降到線性級。", "For large errors the square shoots up while Huber and ε-insensitive grow linearly: an outlier's influence drops to linear."),
    }
    return "plot_xy", spec


def hfit():
    sh = [N.huber_irls(*N.trend(seed=s))[0] for s in range(3)]; so = [N.ols(*N.trend(seed=s))[0] for s in range(3)]
    spec = {
        "id": "lec05-kp22-hfit",
        "alt": tr(f"同樣 21 個點：Huber 線（c=1）斜率 {hu(HB[0],3)}、截距 {hu(HB[1],3)}，貼近乾淨資料的趨勢；最小平方線斜率 {hu(OO[0],3)} 被離群點拉高。三組雜訊（種子 0、1、2）Huber 斜率 " + "、".join(hu(v, 2) for v in sh) + "，最小平方 " + "、".join(hu(v, 2) for v in so) + "。IRLS：先最小平方，算殘差給權重（|e|≤c 為 1，其餘 c/|e|），解加權最小平方，重複 100 次。",
                  f"The same 21 points: the Huber line (c=1) has slope {hu(HB[0],3)} and intercept {hu(HB[1],3)}, close to the clean trend; the least-squares line has slope {hu(OO[0],3)}, pulled up by the outlier. Over three noise draws (seeds 0, 1, 2) the Huber slope is " + ", ".join(hu(v, 2) for v in sh) + " and least squares " + ", ".join(hu(v, 2) for v in so) + ". IRLS: start from least squares, weight by the residuals (1 for |e|≤c, else c/|e|), solve a weighted least squares, repeat 100 times."),
        "plot": {"x": [-0.2, 6.2], "y": [-1.5, 24], "x_ticks": [0, 1, 2, 3, 4, 5], "y_ticks": [0, 5, 10, 15, 20], "x_label": "x", "y_label": "y", "w": 430, "h": 300},
        "series": [{"kind": "seg", "a": (0, HB[1]), "b": (5.4, HB[0] * 5.4 + HB[1]), "role": "green", "width": 4}, {"kind": "seg", "a": (0, OO[1]), "b": (5.4, OO[0] * 5.4 + OO[1]), "role": "red", "width": 4},
                   {"kind": "dots", "points": [(float(a), float(b)) for a, b in zip(XC, YC)], "role": "blue", "d": 10}, {"kind": "dots", "points": [OUT], "role": "red", "d": 18, "shape": "square"}],
        "notes": [{"x": 5.5, "y": HB[0] * 5.4 + HB[1] - 3.2, "text": "Huber", "ha": "center", "w": 80}, {"x": 5.55, "y": OO[0] * 5.4 + OO[1] + 3, "text": tr("最小平方", "least squares"), "ha": "center", "w": 110}],
        "side": [{"title": tr("IRLS（反覆加權最小平方）", "IRLS (iteratively reweighted least squares)"), "role": "blue", "name": tr("term:IRLS", "term:IRLS"), "h": 2.0, "text": tr("1. 先用一般最小平方得一條線。\n2. 算殘差 $e$：$|e|\\leq c$ 權重 1，\n   其餘權重 $c/|e|$。\n3. 解一次加權最小平方。\n4. 回到 2，重複 100 次。", "1. start with an ordinary least-squares line.\n2. residuals $e$: weight 1 if $|e|\\leq c$,\n   else $c/|e|$.\n3. solve one weighted least squares.\n4. back to 2, repeat 100 times.")},
                 {"title": tr("結果", "Result"), "role": "green", "h": 1.0, "text": tr(f"Huber 斜率 {hu(HB[0],3)}（截距 {hu(HB[1],3)}），\n最小平方 {hu(OO[0],3)}，乾淨資料 {hu(OC[0],3)}。", f"Huber slope {hu(HB[0],3)} (intercept {hu(HB[1],3)}),\nleast squares {hu(OO[0],3)}, clean data {hu(OC[0],3)}.")},
                 {"title": tr("仍略偏", "Still a little off"), "role": "orange", "h": 0.9, "text": tr("真值是 2：離群點的權重被壓到\n$c/|e|$，偏得少很多但沒有消失。", "the truth is 2: the outlier's weight is pressed\nto $c/|e|$, far less bias but not gone.")}],
        "takeaway": tr(f"Huber 線斜率 {hu(HB[0],2)}，貼近趨勢；最小平方線被離群點拉到 {hu(OO[0],2)}。", f"The Huber slope is {hu(HB[0],2)}, close to the trend; least squares is pulled to {hu(OO[0],2)} by the outlier."),
    }
    return "plot_xy", spec


def _sin():
    from sklearn.svm import SVR
    rng = np.random.default_rng(1); xs = np.sort(rng.uniform(0, 2 * np.pi, 60)); ys = np.sin(xs) + rng.normal(0, 0.15, 60); return xs, ys


def tube():
    from sklearn.svm import SVR
    xs, ys = _sin(); eps = 0.2; m = SVR(kernel="rbf", gamma=0.5, C=10, epsilon=eps).fit(xs[:, None], ys); xg = np.linspace(0, 2 * np.pi, 120); pg = m.predict(xg[:, None]); sv = set(int(i) for i in m.support_)
    f = NFig("lec05-kp22-tube", 1160, 440, T(tr(f"在預測線上下各畫寬 ε 的管子（這裡 ε={eps}，正弦資料 60 點、RBF 核 SVR）。管子裡的點誤差在容忍範圍，權重為 0；只有落在管邊或管外的 {len(sv)} 個點有權重，它們就是支撐向量；管外的點權重碰到上限 C。SVR 最小化 ½‖w‖²+CΣ(ξ+ξ*)；ε 是稀疏度的旋鈕：管子越寬，支撐向量越少。",
                                              f"A tube of width ε above and below the prediction (here ε={eps}, 60 sine points, RBF SVR). Points inside the tube are within the tolerance and get weight 0; only the {len(sv)} points on the tube's edge or outside have weights: they are the support vectors, and those outside reach the cap C. SVR minimises ½‖w‖²+CΣ(ξ+ξ*); ε is the sparsity knob: the wider the tube, the fewer the support vectors.")))
    header_chip(f, 0, 0, 640, 34, "blue", T(tr(f"管子與支撐向量（ε = {eps}）", f"The tube and the support vectors (ε = {eps})")), size=17); panel(f, 0, 38, 640, 340, "blue")
    ox, oy, pw, ph = 50, 70, 570, 280; sx = lambda x: ox + x / (2 * np.pi) * pw; sy = lambda y: oy + ph / 2 - y / 1.5 * ph / 2
    f.line(ox, oy + ph / 2, ox + pw, oy + ph / 2, "MUT", 1)
    for a in range(len(xg) - 1):
        f.line(sx(xg[a]), sy(pg[a] + eps), sx(xg[a + 1]), sy(pg[a + 1] + eps), "#7a8f86", 2); f.line(sx(xg[a]), sy(pg[a] - eps), sx(xg[a + 1]), sy(pg[a + 1] - eps), "#7a8f86", 2)
        f.line(sx(xg[a]), sy(pg[a]), sx(xg[a + 1]), sy(pg[a + 1]), solid("green"), 4)
    for i, (a, b) in enumerate(zip(xs, ys)):
        if i in sv: f.ellipse(sx(a), sy(b), 13, solid("red"))
        else: f.ellipse(sx(a), sy(b), 8, solid("blue"))
    f.text(ox + 150, oy + 6, T(tr("● 支撐向量（管邊或管外）", "● support vector (edge or outside)")), "#e07a70", 16, True, "left", w=340); f.text(ox + 400, oy + 36, T(tr("● 管內的點", "● inside the tube")), "#6a9fd8", 16, True, "left", w=170)
    f.text(ox + 300, oy + ph + 12, T(tr(f"{len(sv)} 個支撐向量，其餘 {60-len(sv)} 個點在管內、權重 0", f"{len(sv)} support vectors, the other {60-len(sv)} points are inside the tube with weight 0")), "TXT", 16, True, w=560)
    f.rect(670, 0, 490, 150, fill=tint("green", 0.12), stroke=tint("green", 0.55), sw=1.5, radius=12)
    f.text(686, 75, T(tr("SVR（支持向量迴歸）：最小化 $\\frac{1}{2}\\|w\\|^2+C\\sum_i(\\xi_i+\\xi_i^\\star)$，\n點的誤差不超過管寬加鬆弛量；$\\xi_i,\\xi_i^\\star$ 是點在管子\n上方、下方超出的量。", "SVR (support vector regression): minimise $\\frac{1}{2}\\|w\\|^2+C\\sum_i(\\xi_i+\\xi_i^\\star)$,\nerror ≤ tube width + slack\n(ξ, ξ* = excess above or below the tube).")), "TXT", 16, False, "left", w=462, math=True, name=T(tr("term:SVR", "term:SVR")))
    f.rect(670, 164, 490, 144, fill=tint("orange", 0.12), stroke=tint("orange", 0.55), sw=1.5, radius=12)
    f.text(686, 236, T(tr("KKT 條件（最優解必須滿足的條件）的讀法：\n• 誤差 $<\\epsilon$（管內）：$\\lambda_i=\\lambda_i^\\star=0$，不是支撐向量。\n• 誤差 $\\geq\\epsilon$（管邊或管外）：是支撐向量。\n• 誤差 $>\\epsilon$（管外）：權重碰到上限，$\\lambda=C$。", "KKT conditions (what an optimum must satisfy), read as:\n• error $<\\epsilon$ (inside): $\\lambda_i=\\lambda_i^\\star=0$, not a support vector.\n• error $\\geq\\epsilon$ (edge or outside): a support vector.\n• error $>\\epsilon$ (outside): the weight hits the cap, $\\lambda=C$.")), "TXT", 16, False, "left", w=462, math=True, name=T(tr("term:KKT 條件", "term:KKT conditions")))
    chip(f, 690, 324, 450, 54, "yellow", T(tr("ε 是稀疏度的旋鈕：管子越寬，管內的點越多，支撐向量越少", "ε is the sparsity knob: wider tube, fewer support vectors")), size=16, bold=True, dark=True)
    takeaway(f, 0, 388, 1160, 44, T(tr("忽略小誤差，換來稀疏：只有管邊與管外的點有權重。", "Ignoring small errors buys sparsity: only the points on the tube's edge or outside carry weights.")))
    return f


def eps_effect():
    from sklearn.svm import SVR
    xs, ys = _sin(); xg = np.linspace(0, 2 * np.pi, 300); rows = []
    for eps in (0.01, 0.1, 0.2, 0.3, 0.5, 1.0):
        m = SVR(kernel="rbf", gamma=0.5, C=10, epsilon=eps).fit(xs[:, None], ys); mse = float(np.mean((m.predict(xg[:, None]) - np.sin(xg)) ** 2)); inside = float(np.mean(np.abs(m.predict(xs[:, None]) - ys) <= eps + 1e-9)); rows.append((eps, len(m.support_), mse, inside))
    f = NFig("lec05-kp22-eps", 1160, 450, T(tr("掃 ε（正弦資料 60 點，RBF 核 SVR，γ=0.5、C=10；MSE 是預測曲線與真正 sin 在 300 個等距點上的均方誤差）：ε=0.01、0.1、0.2、0.3、0.5、1.0 的支撐向量數 " + "、".join(str(r[1]) for r in rows) + "；MSE " + "、".join(f"{r[2]:.4f}" for r in rows) + "。ε 越大支撐向量越少，但解越粗；ε 大到管子比 sin 的振幅還寬時，曲線幾乎被拉平。",
                                              "Sweeping ε (60 sine points, RBF SVR, γ=0.5, C=10; MSE is the mean squared error between the predicted curve and the true sin at 300 equally spaced points): for ε=0.01, 0.1, 0.2, 0.3, 0.5, 1.0 the number of support vectors is " + ", ".join(str(r[1]) for r in rows) + "; MSE " + ", ".join(f"{r[2]:.4f}" for r in rows) + ". The larger ε, the fewer support vectors but the coarser the fit; once the tube is wider than the sine's amplitude the curve is nearly flat.")))
    for k, (hd, role) in enumerate(((tr("支撐向量數（越少越稀疏）", "support vectors (fewer = sparser)"), "blue"), (tr("對真 sin 的 MSE（越小越準，對數刻度）", "MSE against the true sin (smaller = better, log scale)"), "orange"))):
        x0 = k * 590
        header_chip(f, x0, 0, 570, 34, role, T(hd), size=16); panel(f, x0, 38, 570, 300, role)
        base = 280; bw = 56; gap = (570 - 6 * bw) / 7
        f.line(x0 + 20, base, x0 + 550, base, "MUT", 2)
        for i, (eps, nsv, mse, inside) in enumerate(rows):
            x = x0 + gap * (i + 1) + bw * i
            if k == 0: h = nsv / 60 * 190; lab = str(nsv)
            else: h = (np.log10(mse) + 3.3) / 3.9 * 190; lab = f"{mse:.4f}"
            f.rect(x, base - h, bw, h, fill=solid(role), radius=4)
            f.text(x + bw / 2, base - h - 14, lab, "TXT", 16, True, w=80)
            f.text(x + bw / 2, base + 18, f"{eps:g}", "TXT", 17, True, w=70)
        f.text(x0 + 285, base + 46, T(tr("管寬 ε", "tube width ε")), "MUT", 16, w=200)
    takeaway(f, 0, 360, 1160, 76, T(tr("ε 越大，支撐向量越少（58 → 2），但解越粗：MSE 從 0.0008 升到 0.3455；管內的訓練點比例從 13% 升到 100%。", "Larger ε: fewer support vectors (58 → 2), coarser fit: MSE 0.0008 → 0.3455, inside the tube 13% → 100%.")))
    return f


def ridge():
    f = NFig("lec05-kp22-ridge", 1160, 450, T(tr("脊迴歸：從 SVR 令 ε=0、w₀=0；不等式限制改成等式（誤差剛好等於鬆弛量 ξ_i）；目標改成最小化 C′‖w‖²+Σξ²，C′=1/(2C)。閉式解 g(x)=yᵀ(𝒦+C′I)⁻¹p，其中 y 是目標向量、p=(K(x,x₁),…,K(x,x_N))ᵀ，與 KP21 的核最小平方只差一項 C′I。代價：ε=0 沒有「小誤差不貢獻」的點，所有點都有權重，失去稀疏。",
                                              "Ridge regression: from SVR set ε=0 and w₀=0; turn the inequality constraints into equalities (the error is exactly the slack ξ_i); the objective becomes minimising C′‖w‖²+Σξ², C′=1/(2C). Closed form g(x)=yᵀ(𝒦+C′I)⁻¹p, with y the target vector and p=(K(x,x₁),…,K(x,x_N))ᵀ, differing from KP21's kernel least squares only by C′I. The price: with ε=0 no point contributes nothing, every point has a weight, and sparsity is lost.")))
    f.rect(0, 0, 570, 80, fill=tint("blue", 0.12), stroke=tint("blue", 0.55), sw=1.5, radius=12)
    f.text(14, 40, T(tr("脊迴歸：加了範數懲罰的最小平方——除了預測誤差的平方，\n再多罰解的長度，避免係數過大。", "ridge regression: least squares with a norm penalty: besides the squared\nprediction error it also penalises the length of the solution.")), "TXT", 16, False, "left", w=545, name=T(tr("term:脊迴歸", "term:Ridge regression")))
    f.rect(590, 0, 570, 80, fill=tint("green", 0.12), stroke=tint("green", 0.55), sw=1.5, radius=12)
    f.text(604, 40, T(tr("閉式解：能直接用公式算出答案，不需要疊代。", "closed-form solution: the answer comes straight from a formula, no iteration needed.")), "TXT", 16, False, "left", w=545, name=T(tr("term:閉式解", "term:Closed-form solution")))
    steps = [("blue", tr("① 從 SVR 出發", "① Start from SVR"), tr("令 $\\epsilon=0$、$w_0=0$", "set $\\epsilon=0$, $w_0=0$")),
             ("orange", tr("② 不等式改等式", "② Equalities"), tr("誤差剛好等於鬆弛量 $\\xi_i$；\n最小化 $\\mathcal{C}\\|w\\|^2+\\sum_i\\xi_i^2$，$\\mathcal{C}=\\frac{1}{2C}$", "the error is exactly the slack $\\xi_i$;\nminimise $\\mathcal{C}\\|w\\|^2+\\sum_i\\xi_i^2$, $\\mathcal{C}=\\frac{1}{2C}$")),
             ("green", tr("③ 閉式解", "③ Closed form"), "$g(x)=y^T(\\mathcal{K}+\\mathcal{C}I)^{-1}p$")]
    w = (1160 - 2 * 36) / 3
    for k, (role, hd, tx) in enumerate(steps):
        x = k * (w + 36)
        header_chip(f, x, 100, w, 34, role, T(hd), size=18); panel(f, x, 138, w, 110, role)
        f.text(x + w / 2, 193, T(tx), "TXT", 18, w=w - 16, math=True)
        if k < 2: f.text(x + w + 18, 193, "→", "TXT", 26, True, w=36)
    chip(f, 0, 268, 560, 44, "yellow", T(tr("與 KP21 的核最小平方只差一項 $\\mathcal{C}I$", "differs from KP21's kernel least squares only by $\\mathcal{C}I$")), size=18, bold=True, dark=True, math=True)
    chip(f, 600, 268, 560, 44, "red", T(tr("代價：所有點都有權重，失去稀疏", "price: every point has a weight, sparsity is lost")), size=18, bold=True, dark=True)
    takeaway(f, 0, 336, 1160, 56, T(tr("ε=0 換來閉式解（不用疊代），代價是失去稀疏：天下沒有白吃的午餐。", "ε=0 buys a closed form (no iteration) at the price of sparsity: there is no free lunch.")))
    return f


def collinear():
    rng = np.random.default_rng(2); X = rng.normal(size=(15, 4)); y = rng.normal(size=15); C = 0.8
    wp = np.linalg.solve(X.T @ X + C * np.eye(4), X.T @ y); wd = X.T @ np.linalg.solve(X @ X.T + C * np.eye(15), y); diff = float(np.abs(wp - wd).max())
    rng = np.random.default_rng(3); t = rng.normal(size=30); X2 = np.c_[t, t + 1e-4 * rng.normal(size=30)]; y2 = t + 0.1 * rng.normal(size=30)
    w0 = np.linalg.lstsq(X2, y2, rcond=None)[0]; c0 = np.linalg.cond(X2.T @ X2); w1 = np.linalg.solve(X2.T @ X2 + 0.1 * np.eye(2), X2.T @ y2); c1 = np.linalg.cond(X2.T @ X2 + 0.1 * np.eye(2))
    sc = lambda v: f"{v:.0e}".replace("e+0", "e").replace("e+", "e")
    f = NFig("lec05-kp22-collinear", 1160, 450, T(tr(f"線性核的兩種寫法 w=(XᵀX+C′I)⁻¹Xᵀy=Xᵀ(XXᵀ+C′I)⁻¹y 算出的 w 最大相差 {diff:.1e}（15 點、4 個特徵，捨入誤差）。兩個欄位幾乎相同時（30 點）：最小平方係數 ({hu(w0[0],1)}, {hu(w0[1],1)})，XᵀX 的條件數約 {c0:.0e}；加上 C′=0.1：係數 ({hu(w1[0],2)}, {hu(w1[1],2)})，條件數 {c1:.0f}。代價是係數被壓向 0。",
                                              f"The two forms of the linear kernel, w=(XᵀX+C′I)⁻¹Xᵀy=Xᵀ(XXᵀ+C′I)⁻¹y, give w that differ by at most {diff:.1e} (15 points, 4 features, rounding error). With two nearly identical columns (30 points): least-squares coefficients ({hu(w0[0],1)}, {hu(w0[1],1)}), condition number of XᵀX about {c0:.0e}; with C′=0.1: coefficients ({hu(w1[0],2)}, {hu(w1[1],2)}), condition number {c1:.0f}. The price is that the coefficients are pulled toward 0.")))
    f.rect(0, 0, 570, 78, fill=tint("blue", 0.12), stroke=tint("blue", 0.55), sw=1.5, radius=12)
    f.text(14, 39, T(tr("共線：兩個欄位幾乎相同，其中一個\n幾乎能由另一個算出來。", "collinearity: two columns nearly identical, so one\nis almost computable from the other.")), "TXT", 16, False, "left", w=545, name=T(tr("term:共線", "term:Collinearity")))
    f.rect(590, 0, 570, 78, fill=tint("green", 0.12), stroke=tint("green", 0.55), sw=1.5, radius=12)
    f.text(604, 39, T(tr("條件數：矩陣最大與最小特徵值的比；越大，資料的\n一點點擾動就讓解大幅跳動。", "condition number: ratio of the largest to the smallest eigenvalue;\nthe larger, the more a small data change makes the solution jump.")), "TXT", 16, False, "left", w=545, name=T(tr("term:條件數", "term:Condition number")))
    header_chip(f, 0, 98, 560, 34, "grey", T(tr("兩種寫法是同一個解", "the two forms are the same solution")), size=17); panel(f, 0, 136, 560, 210, "grey")
    f.text(280, 178, "$w=(X^TX+\\mathcal{C}I)^{-1}X^Ty$", "TXT", 20, w=520, math=True)
    f.text(280, 222, "$w=X^T(XX^T+\\mathcal{C}I)^{-1}y$", "TXT", 20, w=520, math=True)
    f.text(280, 268, T(tr("左邊解特徵數大小的方程，右邊解點數大小的方程；\n特徵多、點少時右邊划算。", "the left solves a system the size of the features, the right\none the size of the points; with many features and few points the right is cheaper.")), "MUT", 16, w=530)
    chip(f, 60, 306, 440, 32, "yellow", T(tr(f"最大相差 {diff:.1e}（捨入誤差）", f"largest difference {diff:.1e} (rounding error)")), size=17, bold=True, dark=True)
    for k, (hd, role, w, cn) in enumerate(((tr("一般最小平方（C′=0）", "ordinary least squares (C′=0)"), "red", (w0, 1), c0), (tr("脊迴歸（C′=0.1）", "ridge (C′=0.1)"), "green", (w1, 2), c1))):
        x = 590 + k * 295
        header_chip(f, x, 98, 275, 34, role, T(hd), size=16); panel(f, x, 136, 275, 210, role)
        f.text(x + 137, 178, f"({hu(w[0][0], w[1])}, {hu(w[0][1], w[1])})", "TXT", 26, True, w=260)
        f.text(x + 137, 214, T(tr("係數 $w$", "coefficients $w$")), "MUT", 16, w=200, math=True)
        f.text(x + 137, 262, T(tr("條件數", "condition number")), "MUT", 16, w=200)
        f.text(x + 137, 300, (f"5×10⁸" if k == 0 else f"{cn:.0f}"), "TXT", 28, True, w=250)
    takeaway(f, 0, 366, 1160, 56, T(tr("沿對角線加一個小正數 C′，條件數從 5×10⁸ 降到 776，係數從 (−170, 171) 回到 (0.49, 0.49)；代價是係數被壓向 0。", "A small C′ on the diagonal: condition number 5×10⁸ → 776, coefficients back to (0.49, 0.49); price: shrunk toward 0.")))
    return f


def compare():
    sv = N.svr_lin(XO, YO, 0.5, 10.0); rd = N.ridge_lin(XO, YO, 0.1)
    items = [(tr("最小平方", "least squares"), OO[0], 21, "grey"), (tr("Huber（c=1）", "Huber (c=1)"), HB[0], 21, "green"), (tr("ε-不敏感（ε=0.5）", "ε-insensitive (ε=0.5)"), sv[0][0], len(sv[1].support_), "blue"), (tr("脊迴歸（C′=0.1）", "ridge (C′=0.1)"), rd[0], 21, "red")]
    spec = {
        "id": "lec05-kp22-compare",
        "alt": tr("同一份有離群點的 21 點，斜率（真值 2）：最小平方 " + hu(items[0][1], 3) + "（21 個非零係數）、Huber " + hu(items[1][1], 3) + "（21）、ε-不敏感線性 SVR（ε=0.5、C=10）" + hu(items[2][1], 3) + f"（{items[2][2]} 個支撐向量）、脊迴歸（C′=0.1）" + hu(items[3][1], 3) + "（21）。脊迴歸抗的是共線與過擬合，不是離群點。",
                  "On the same 21 points with the outlier, the slope (truth 2): least squares " + hu(items[0][1], 3) + " (21 non-zero coefficients), Huber " + hu(items[1][1], 3) + " (21), ε-insensitive linear SVR (ε=0.5, C=10) " + hu(items[2][1], 3) + f" ({items[2][2]} support vectors), ridge (C′=0.1) " + hu(items[3][1], 3) + " (21). Ridge resists collinearity and overfitting, not outliers."),
        "unit": tr("斜率（真值 2）", "slope (truth 2)"),
        "bars": [{"label": n, "value": round(v + 1e-9, 3), "role": r, "note": T(tr(f"非零係數 {k}" if k == 21 else f"支撐向量 {k}", f"non-zero coefficients {k}" if k == 21 else f"support vectors {k}"))} for n, v, k, r in items],
        "max": 3, "ticks": [0, 1, 2, 3],
        "takeaway": tr("Huber 與 ε-不敏感抗離群點；脊迴歸抗的是共線與過擬合，不是離群點（2.679 對 2.686）。", "Huber and ε-insensitive resist outliers; ridge resists collinearity and overfitting, not outliers (2.679 against 2.686)."),
    }
    return "bars_compare", spec


def csweep():
    cs = [0.2, 0.5, 1, 2, 5, 20]; sl = [N.huber_irls(XO, YO, c)[0] for c in cs]
    spec = {
        "id": "lec05-kp22-csweep",
        "alt": tr("Huber 的門檻 c（同一份 21 點）：c=0.2、0.5、1、2、5、20 的斜率 " + "、".join(hu(v, 3) for v in sl) + "。c 大時所有誤差落在平方區，等於最小平方；c 小離群點幾乎被忽略；c 要配合雜訊大小（標準差 0.3）選。",
                  "The Huber threshold c (the same 21 points): the slope for c=0.2, 0.5, 1, 2, 5, 20 is " + ", ".join(hu(v, 3) for v in sl) + ". For large c every error falls in the squared zone, which is least squares; for small c the outlier is nearly ignored; c should be chosen with the noise size (standard deviation 0.3)."),
        "plot": {"x": [0.5, 6.5], "y": [1.9, 2.8], "x_ticks": [1, 2, 3, 4, 5, 6], "x_names": {str(i + 1): str(c) for i, c in enumerate(cs)}, "y_ticks": [2.0, 2.2, 2.4, 2.6, 2.8], "x_label": tr("Huber 的門檻 c", "Huber threshold c"), "y_label": tr("斜率", "slope"), "w": 430, "h": 300},
        "series": [{"kind": "hline", "y": 2.0, "role": "grey", "width": 2, "label": tr("真值 2", "truth 2")}, {"kind": "line", "points": [(i + 1, v) for i, v in enumerate(sl)], "role": "green", "width": 4, "markers": True, "labels": [hu(v, 3) for v in sl], "label_dy": -20}],
        "side": [{"title": tr("c 大", "Large c"), "role": "red", "h": 1.0, "text": tr("所有誤差都落在平方區，\n等於最小平方（斜率 2.686）。", "every error falls in the squared zone,\nwhich equals least squares (slope 2.686).")},
                 {"title": tr("c 小", "Small c"), "role": "green", "h": 1.0, "text": tr("離群點幾乎被忽略，\n斜率貼近真值 2。", "the outlier is nearly ignored,\nthe slope is close to the truth 2.")},
                 {"title": tr("怎麼選", "How to choose"), "role": "orange", "h": 1.0, "text": tr("c 要配合雜訊大小選\n（這裡標準差 0.3）。", "choose c with the noise size\n(here the standard deviation 0.3).")}],
        "takeaway": tr("c 越大越像最小平方：斜率由 2.045 升到 2.686。", "The larger c, the more like least squares: the slope rises from 2.045 to 2.686."),
    }
    return "plot_xy", spec


def solution():
    spec = {
        "id": "lec05-kp22-solution",
        "alt": tr("解答：Huber（c=1）：e=0.5 ≤ 1：½·0.25=0.125；e=3 > 1：1·3−½=2.5（平方 ½e² 會給 0.125 與 4.5）。ε-不敏感（ε=0.5）：e=0.3：max(0,0.3−0.5)=0；e=2：max(0,2−0.5)=1.5。小誤差在 Huber 與平方一樣、在 ε-不敏感變成 0；大誤差兩者都線性增長。",
                  "Solution: Huber (c=1): e=0.5 ≤ 1: ½·0.25=0.125; e=3 > 1: 1·3−½=2.5 (the square ½e² would give 0.125 and 4.5). ε-insensitive (ε=0.5): e=0.3: max(0,0.3−0.5)=0; e=2: max(0,2−0.5)=1.5. Small errors are the same as the square for Huber and become 0 for ε-insensitive; large errors grow linearly for both."),
        "steps": [{"role": "blue", "title": tr("① 題 1：Huber", "① Q1: Huber"), "lines": [tr("$e=0.5\\leq1$：$\\frac{1}{2}\\times0.25$", "$e=0.5\\leq1$: $\\frac{1}{2}\\times0.25$"), tr("$e=3>1$：$1\\times3-\\frac{1}{2}$", "$e=3>1$: $1\\times3-\\frac{1}{2}$"), tr("平方會給 0.125 與 4.5", "the square would give 0.125 and 4.5")], "result": "0.125；2.5"},
                  {"role": "green", "title": tr("② 題 2：ε-不敏感", "② Q2: ε-insensitive"), "lines": ["$e=0.3$：$\\max(0,\\ 0.3-0.5)$", "$e=2$：$\\max(0,\\ 2-0.5)$"], "result": "0；1.5"},
                  {"role": "orange", "title": tr("③ 核對", "③ Check"), "lines": [tr("小誤差：Huber 與平方一樣，", "small errors: Huber equals the square,"), tr("ε-不敏感變成 0；", "ε-insensitive becomes 0;"), tr("大誤差：兩者都線性增長", "large errors: both grow linearly")], "result": tr("兩者都不怕大誤差", "neither fears large errors")}],
        "takeaway": tr("小誤差 Huber 像平方、ε-不敏感為 0；大誤差兩者都只線性增長。", "For small errors Huber acts like the square and ε-insensitive is 0; for large errors both grow only linearly."),
    }
    return "steps_flow", spec
