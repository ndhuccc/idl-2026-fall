"""KP24 (combining classifiers) -- fifth sample of view-from-text-v2. Plan: kp24_plan.json. Numbers come from kp24_nums.py (verified against the lecture);
the slow experiments (sections 6 and 7) are cached in kp24_cache.json, written by kp24_nums.py's numbers (4 decimals)."""
import os, sys, json
from math import comb
from decimal import Decimal, ROUND_HALF_UP
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from figkit.parts import T, NFig, header_chip, panel, chip, outline_box, takeaway, tint, solid, INK


def tr(zh, en): return {"zh": zh, "en": en}


def r1(x):   # one decimal, half up (85.15 -> 85.2), the way the lecture rounds
    return str(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def Pc(L, p): return sum(comb(L, m) * p ** m * (1 - p) ** (L - m) for m in range((L + 1) // 2, L + 1))


C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kp24_cache.json")))


def complement():
    n = 2000; ks = [("logistic", 1703, "blue", tr("logistic 迴歸", "logistic regression")), ("tree", 1652, "orange", tr("決策樹（深度 4）", "decision tree (depth 4)")),
                    ("knn", 1756, "green", "3-NN"), ("vote", 1723, "grey", tr("三者投票", "vote of the three"))]
    best = 1756; err = n - best; ok_other = 146
    f = NFig("lec05-kp24-complement", 1160, 420, T(tr(f"新點 2000 個：logistic 迴歸判對 1703（{r1(1703/20)}%）、決策樹 1652（{r1(1652/20)}%）、3-NN 1756（{r1(1756/20)}%）、三者投票 1723（{r1(1723/20)}%）。最好的 3-NN 錯 {err} 點，其中 {ok_other} 點（約 {ok_other/err*100:.0f}%）至少有一個其他分類器答對，這是互補資訊；但三者投票仍低於 3-NN。",
                                              f"Of 2000 new points: logistic regression gets 1703 right ({r1(1703/20)}%), the tree 1652 ({r1(1652/20)}%), 3-NN 1756 ({r1(1756/20)}%), the vote of the three 1723 ({r1(1723/20)}%). The best, 3-NN, errs on {err} points; {ok_other} of them (about {ok_other/err*100:.0f}%) are right in at least one other classifier, which is complementary information; yet the vote of the three stays below 3-NN.")))
    header_chip(f, 0, 0, 700, 34, "blue", T(tr("新點準確率（2000 個新點）", "accuracy on 2000 new points")), size=18); panel(f, 0, 38, 700, 290, "blue")
    sc = lambda v: v / 100 * 380
    for i, (_, c, role, nm) in enumerate(ks):
        y = 60 + i * 66
        f.text(14, y + 18, T(nm), "TXT", 17, True, "left", w=190)
        f.rect(210, y, sc(c / 20), 36, fill=solid(role), radius=5)
        f.text(210 + sc(c / 20) + 10, y + 12, f"{r1(c / 20)}%", "TXT", 20, True, "left", w=90)
        f.text(210 + sc(c / 20) + 10, y + 32, f"{c}/2000", "MUT", 16, False, "left", w=100)
    header_chip(f, 740, 0, 420, 34, "green", T(tr(f"最好的 3-NN 錯了 {err} 點", f"The best, 3-NN, errs on {err} points")), size=18); panel(f, 740, 38, 420, 290, "green")
    w1 = 380 * ok_other / err
    f.rect(760, 80, w1, 60, fill=solid("green"), radius=6, text=str(ok_other), tcolor="#1c1712", size=26, bold=True)
    f.rect(760 + w1, 80, 380 - w1, 60, fill=solid("red"), radius=6, text=str(err - ok_other), tcolor="#1c1712", size=26, bold=True)
    f.text(760, 170, T(tr(f"{ok_other} 點（約 {ok_other/err*100:.0f}%）至少有另一個分類器答對", f"{ok_other} points (about {ok_other/err*100:.0f}%) are right in at least one other classifier")), "TXT", 16, False, "left", w=380, name=None)
    f.text(760, 200, T(tr(f"{err-ok_other} 點三個分類器都錯", f"{err-ok_other} points all three get wrong")), "TXT", 16, False, "left", w=380)
    chip(f, 780, 238, 340, 40, "yellow", T(tr("這就是互補資訊", "this is complementary information")), size=20, bold=True, dark=True)
    takeaway(f, 0, 346, 1160, 54, T(tr("最好的分類器也會在別人答對的點上出錯；但三者投票只有 86.2%，低於 3-NN 的 87.8%，互補資訊要挖得出來才有價值。", "Even the best errs where others are right; but voting gets 86.2%, below 3-NN's 87.8%.")))
    return f


def product_rule():
    f = NFig("lec05-kp24-product", 1160, 470, T(tr("乘積規則：後驗機率是分類器看到 x 後對各類的把握；KL 距離衡量兩個分布差多遠。找結合後的分布 P，使它與 L 個分類器的平均 KL 距離最小：展開平均；令 Q_i=∏P_ji^(1/L)/C；代回得 KL(P‖Q)−ln C；KL≥0，P=Q 時最小。例：三個分類器 (0.8,0.2)、(0.6,0.4)、(0.7,0.3)，ω1 乘積 0.336、ω2 為 0.024，判 ω1。",
                                              "Product rule: the posterior probability is a classifier's confidence in each class after seeing x; the KL distance measures how far two distributions are. Find the combined distribution P minimising the mean KL distance to the L classifiers: expand the mean; let Q_i=∏P_ji^(1/L)/C; substitute to get KL(P‖Q)−ln C; KL≥0, minimal at P=Q. Example: three classifiers (0.8,0.2), (0.6,0.4), (0.7,0.3): ω1 has product 0.336, ω2 has 0.024, so choose ω1.")))
    f.rect(0, 0, 570, 78, fill=tint("blue", 0.12), stroke=tint("blue", 0.55), sw=1.5, radius=12)
    f.text(14, 39, T(tr("後驗機率 $P_j(\\omega_i|x)$：第 $j$ 個分類器看到 $x$ 後，\n認為它屬於第 $i$ 類的把握；各類加起來為 1。", "posterior $P_j(\\omega_i|x)$: how sure classifier $j$ is, after seeing $x$,\nthat it belongs to class $i$; the classes sum to 1.")), "TXT", 16, False, "left", w=545, math=True, name=T(tr("term:後驗機率", "term:Posterior probability")))
    f.rect(590, 0, 570, 78, fill=tint("green", 0.12), stroke=tint("green", 0.55), sw=1.5, radius=12)
    f.text(604, 39, T(tr("KL 距離 $D(P\\|P_j)=\\sum_iP_i\\ln(P_i/P_{ji})$：兩個分布差多遠；\n不對稱，嚴格說不是距離。", "KL distance $D(P\\|P_j)=\\sum_iP_i\\ln(P_i/P_{ji})$: how far two distributions\nare; asymmetric, strictly not a distance.")), "TXT", 16, False, "left", w=545, math=True, name=T(tr("term:KL 距離", "term:KL distance")))
    steps = [("blue", "$\\frac{1}{L}\\sum_jD_j=\\sum_iP_i\\ln P_i-\\sum_iP_i\\cdot\\frac{1}{L}\\sum_j\\ln P_{ji}$", tr("展開平均", "expand the mean")),
             ("green", "$Q_i=\\frac{1}{C}\\prod_jP_{ji}^{1/L}$", tr("令 $Q$（$C$ 讓它加起來為 1）", "let $Q$ ($C$: sums to 1)")),
             ("orange", "$\\frac{1}{L}\\sum_jD_j=\\sum_iP_i\\ln\\frac{P_i}{Q_i}-\\ln C$", tr("代回：第一項是 $P$ 與 $Q$ 的 KL", "substitute: first term = KL")),
             ("red", "$P=Q$", tr("KL $\\geq0$；$P=Q$ 時為 0", "KL $\\geq0$; 0 at $P=Q$"))]
    for k, (role, tex, why) in enumerate(steps):
        y = 98 + k * 64
        chip(f, 0, y + 6, 36, 36, role, str(k + 1), size=20, bold=True, dark=True)
        f.rect(46, y, 430, 50, fill=tint(role, 0.2), stroke=solid(role), sw=2, radius=12, text=tex, tcolor="TXT", size=19, math=True)
        f.text(488, y + 25, T(why), "TXT", 16, False, "left", w=225, math="$" in T(why))
    f.rect(0, 360, 700, 48, fill=tint("yellow", 0.25), radius=12)
    f.text(350, 384, T(tr("乘積規則：各分類器機率相乘再開 $L$ 次方，取最大的類別　$P(\\omega_i|x)=\\frac{1}{C}\\prod_{j}P_j(\\omega_i|x)^{1/L}$", "product rule: multiply, take the $L$-th root, pick the largest: $P(\\omega_i|x)=\\frac{1}{C}\\prod_{j}P_j^{1/L}$")), "TXT", 16, True, w=680, math=True, name=T(tr("term:乘積規則", "term:Product rule")))
    header_chip(f, 730, 98, 430, 34, "orange", T(tr("例：三個分類器、兩類", "Example: three classifiers, two classes")), size=18); panel(f, 730, 136, 430, 272, "orange")
    for i, (a, b) in enumerate(((0.8, 0.2), (0.6, 0.4), (0.7, 0.3))):
        f.text(748, 170 + i * 30, T(tr(f"分類器 {i+1}：$\\omega_1$ 給 {a}，$\\omega_2$ 給 {b}", f"classifier {i+1}: {a} for $\\omega_1$, {b} for $\\omega_2$")), "TXT", 16, False, "left", w=400, math=True)
    p1 = 0.8 * 0.6 * 0.7; p2 = 0.2 * 0.4 * 0.3
    for i, (nm, v, role, tex) in enumerate((("$\\omega_1$", p1, "blue", "$0.8\\times0.6\\times0.7=0.336$"), ("$\\omega_2$", p2, "orange", "$0.2\\times0.4\\times0.3=0.024$"))):
        y = 268 + i * 52
        f.text(748, y + 14, nm, "TXT", 20, True, "left", w=40, math=True)
        f.rect(790, y, max(v / 0.336 * 120, 8), 28, fill=solid(role), radius=5)
        f.text(790 + max(v / 0.336 * 120, 8) + 10, y + 14, tex, "TXT", 16, False, "left", w=260, math=True)
    chip(f, 770, 366, 350, 32, "green", T(tr("乘積最大的是 $\\omega_1$，判 $\\omega_1$", "the largest product is $\\omega_1$: choose $\\omega_1$")), size=17, bold=True, dark=True, math=True)
    takeaway(f, 0, 420, 1160, 44, T(tr("乘積規則 = 各分類器機率的幾何平均：讓結合分布與所有分類器的平均 KL 距離最小。", "The product rule: the geometric mean, which minimises the mean KL distance.")))
    return f


def sum_rule():
    f = NFig("lec05-kp24-sum", 1160, 456, T(tr("加總規則：把 KL 距離的兩個引數對調（P_j 放前面），平均後只剩 −ΣP̄_i ln P_i，P=P̄ 時最小，所以最佳解是算術平均 P=(1/L)ΣP_j。同樣三個分類器 ω1 平均 0.70、ω2 為 0.30，也判 ω1。兩個規則各有資訊論出身，理論上沒有偏好依據；乘積輸出接近零時很脆弱，實務上加總常較穩（來源）。",
                                              "Sum rule: swap the two arguments of the KL distance (P_j first); after averaging only −ΣP̄_i ln P_i depends on P, minimal at P=P̄, so the best solution is the arithmetic mean P=(1/L)ΣP_j. The same three classifiers give an average of 0.70 for ω1 and 0.30 for ω2: choose ω1 again. Each rule has a rigorous information-theoretic origin, so theory has no preference; the product is fragile when an output is near zero and in practice the sum is often steadier (source).")))
    cols = [("blue", tr("乘積規則", "Product rule"), "$D_j=\\sum_iP_i\\ln\\frac{P_i}{P_{ji}}$", tr("$P$ 放前面", "$P$ first"), "$P=\\frac{1}{C}\\prod_jP_j^{1/L}$", tr("幾何平均", "geo. mean"), "$0.336\\ \\text{對}\\ 0.024$", tr("0.336 對 0.024", "0.336 vs 0.024")),
            ("green", tr("加總規則", "Sum rule"), "$D_j'=\\sum_iP_{ji}\\ln\\frac{P_{ji}}{P_i}$", tr("$P_j$ 放前面", "$P_j$ first"), "$P=\\frac{1}{L}\\sum_jP_j$", tr("算術平均", "arith. mean"), "$0.70\\ \\text{對}\\ 0.30$", tr("0.70 對 0.30", "0.70 vs 0.30"))]
    for k, (role, hd, kl, kl_note, rule, rule_nm, _, ex) in enumerate(cols):
        x = k * 590
        header_chip(f, x, 0, 570, 36, role, T(hd), size=20); panel(f, x, 40, 570, 292, role)
        f.text(x + 20, 72, T(tr("KL 距離的引數順序", "argument order of the KL distance")), "MUT", 16, False, "left", w=300)
        f.text(x + 262, 108, kl, "TXT", 22, w=300, math=True); chip(f, x + 440, 92, 112, 32, role, T(kl_note), size=16, bold=True, alpha=0.4, math=True)
        f.text(x + 20, 156, T(tr("平均距離最小的結合分布", "combined distribution with the smallest mean distance")), "MUT", 16, False, "left", w=400)
        f.text(x + 262, 196, rule, "TXT", 24, True, w=300, math=True); chip(f, x + 440, 180, 112, 32, role, T(rule_nm), size=16, bold=True, alpha=0.4)
        f.text(x + 20, 244, T(tr("同樣三個分類器 $(0.8,0.2)$、$(0.6,0.4)$、$(0.7,0.3)$", "the same three classifiers $(0.8,0.2)$, $(0.6,0.4)$, $(0.7,0.3)$")), "MUT", 16, False, "left", w=540, math=True)
        sc = (0.336, 0.024) if k == 0 else (0.70, 0.30); mx = 0.336 if k == 0 else 0.70
        for i, (nm, v, r2) in enumerate((("$\\omega_1$", sc[0], "blue"), ("$\\omega_2$", sc[1], "orange"))):
            f.text(x + 30, 274 + i * 28, nm, "TXT", 18, True, "left", w=40, math=True)
            f.rect(x + 72, 262 + i * 28, max(v / mx * 200, 6), 20, fill=solid(r2), radius=4)
            f.text(x + 72 + max(v / mx * 200, 6) + 10, 272 + i * 28, (f"{v:.2f}" if k == 1 else f"{v:g}"), "TXT", 18, True, "left", w=70)
        chip(f, x + 400, 268, 152, 34, "yellow", T(tr("判 $\\omega_1$", "choose $\\omega_1$")), size=18, bold=True, dark=True, math=True)
    f.text(580, 352, T(tr("加總規則：把各分類器的機率取算術平均，取最大的類別。", "sum rule: average the classifiers' probabilities arithmetically and pick the largest class.")), "TXT", 16, True, w=1120, name=T(tr("term:加總規則", "term:Sum rule")))
    f.text(580, 380, T(tr("理論上沒有偏好依據；乘積輸出接近零時很脆弱，實務上加總常較穩（來源）。", "No theoretical preference; the product is fragile near zero, the sum often steadier in practice (source).")), "MUT", 16, w=1120)
    takeaway(f, 0, 408, 1160, 44, T(tr("兩個引數對調：乘積規則得到幾何平均，加總規則得到算術平均。", "Swap the two arguments: the product rule gives the geometric mean, the sum rule the arithmetic mean.")))
    return f


def _rules_page(fid, alt, outs, hd_out, rules, take, vmax, defn=None):
    f = NFig(fid, 1160, 470, alt)
    header_chip(f, 0, 0, 1160, 34, "blue", T(hd_out), size=18); panel(f, 0, 38, 1160, 190, "blue")
    n = len(outs); w = 1100 / n
    for i, v in enumerate(outs):
        cx = 30 + w * (i + 0.5); h = max(v / vmax * 110, 4)
        f.rect(cx - 26, 200 - h, 52, h, fill=solid("red" if v < 0.1 else "blue"), radius=4)
        f.text(cx, 200 - h - 14, f"{v:g}", "TXT", 18, True, w=90)
        f.text(cx, 214, T(tr(f"分類器 {i+1}", f"classifier {i+1}")), "MUT", 16, w=110)
    if defn: f.text(580, 60, T(defn[0]), "TXT", 16, True, w=1100, name=T(defn[1]))
    rw = (1160 - 2 * 20) / 3
    for k, (hd, l1, l2, v1, v2, res, good) in enumerate(rules):
        x = k * (rw + 20); role = "green" if good else "red"
        header_chip(f, x, 244, rw, 34, "grey", T(hd), size=19); panel(f, x, 282, rw, 126, role)
        f.text(x + 16, 306, T(l1), "TXT", 17, False, "left", w=rw - 30, math=True); f.text(x + 16, 340, T(l2), "TXT", 17, False, "left", w=rw - 30, math=True)
        chip(f, x + 16, 362, rw - 32, 36, role, T(res), size=19, bold=True, dark=True)
    takeaway(f, 0, 424, 1160, 40, T(take))
    return f


def veto():
    outs = [0.7, 0.7, 0.7, 0.7, 0.001]
    pr1 = 0.7 ** 4 * 0.001; pr2 = 0.3 ** 4 * 0.999; s1 = (4 * 0.7 + 0.001) / 5
    alt = tr(f"五個分類器對正確類 ω1 給 0.7、0.7、0.7、0.7、0.001：乘積 ω1 為 {pr1:.5f}、ω2 為 {pr2:.5f}，選 ω2（錯）；加總 ω1 為 {s1:.4f}、ω2 為 {1-s1:.4f}，選 ω1（對）；投票 4 票對 1 票，選 ω1（對）。",
             f"Five classifiers give the correct class ω1 0.7, 0.7, 0.7, 0.7, 0.001: product ω1 {pr1:.5f}, ω2 {pr2:.5f}, choose ω2 (wrong); sum ω1 {s1:.4f}, ω2 {1-s1:.4f}, choose ω1 (right); vote 4 to 1, choose ω1 (right).")
    rules = [(tr("乘積", "Product"), f"$\\omega_1:\\ 0.7^4\\times0.001={pr1:.5f}$".replace("0.00024", "0.00024"), f"$\\omega_2:\\ 0.3^4\\times0.999={pr2:.5f}$", pr1, pr2, tr("選 ω2（錯）", "chooses ω2 (wrong)"), False),
             (tr("加總", "Sum"), f"$\\omega_1:\\ (4\\times0.7+0.001)/5={s1:.4f}$", f"$\\omega_2:\\ {1-s1:.4f}$", s1, 1 - s1, tr("選 ω1（對）", "chooses ω1 (right)"), True),
             (tr("投票", "Vote"), tr("$\\omega_1$：4 票", "$\\omega_1$: 4 votes"), tr("$\\omega_2$：1 票", "$\\omega_2$: 1 vote"), 4, 1, tr("選 ω1（對）", "chooses ω1 (right)"), True)]
    return _rules_page("lec05-kp24-veto", alt, outs, tr("五個分類器給正確類 ω₁ 的機率", "probability each of five classifiers gives the correct class ω₁"), rules,
                       tr("乘積被一個接近零的因子整個拉到零，等於一票否決；加總與投票不受影響。", "One near-zero factor drags the product to zero, a veto; the sum and the vote are unaffected."), 0.7)


def vote_curves():
    Ls = list(range(1, 52, 2))
    rows = [(0.6, "green"), (0.55, "blue"), (0.5, "grey"), (0.45, "red")]
    spec = {
        "id": "lec05-kp24-vote",
        "alt": tr("獨立假設下的多數投票正確率 P_c(L)：p=0.6 時 L=3、11、51 為 " + "、".join(f"{Pc(L, 0.6):.3f}" for L in (3, 11, 51)) + "；p=0.55 緩慢上升；p=0.5 恆為 0.5；p=0.45 下降到 " + f"{Pc(51, 0.45):.3f}" + "。例：p=0.6、L=3：0.432+0.216=0.648。",
                  "Majority-vote accuracy P_c(L) under independence: with p=0.6 it is " + ", ".join(f"{Pc(L, 0.6):.3f}" for L in (3, 11, 51)) + " at L=3, 11, 51; p=0.55 rises slowly; p=0.5 stays 0.5; p=0.45 falls to " + f"{Pc(51, 0.45):.3f}" + ". Example p=0.6, L=3: 0.432+0.216=0.648."),
        "plot": {"x": [0, 62], "y": [0.2, 1.0], "x_ticks": [1, 11, 21, 31, 41, 51], "y_ticks": [0.2, 0.4, 0.5, 0.6, 0.8, 1.0], "x_label": tr("分類器個數 L（奇數）", "number of classifiers L (odd)"), "y_label": tr("多數決正確率 P_c(L)", "majority-vote accuracy P_c(L)"), "w": 420, "h": 300},
        "series": [{"kind": "line", "points": [(L, Pc(L, p)) for L in Ls], "role": r, "width": 4} for p, r in rows] +
                  [{"kind": "dots", "points": [(L, Pc(L, 0.6)) for L in (3, 11, 51)], "role": "green", "d": 14}],
        "notes": [{"x": 52, "y": Pc(51, 0.6), "text": "p = 0.6", "ha": "left", "w": 90}, {"x": 52, "y": Pc(51, 0.55), "text": "p = 0.55", "ha": "left", "w": 90}, {"x": 52, "y": 0.5, "text": "p = 0.5", "ha": "left", "w": 90}, {"x": 52, "y": Pc(51, 0.45), "text": "p = 0.45", "ha": "left", "w": 90}, {"x": 3, "y": Pc(3, 0.6) + 0.06, "text": f"{Pc(3, 0.6):.3f}", "ha": "left"}, {"x": 11, "y": Pc(11, 0.6) + 0.06, "text": f"{Pc(11, 0.6):.3f}"}, {"x": 47, "y": Pc(51, 0.6) + 0.06, "text": f"{Pc(51, 0.6):.3f}"}],
        "side": [{"title": tr("多數投票", "Majority vote"), "role": "green", "name": tr("term:多數投票", "term:Majority vote"), "h": 1.0, "text": tr("至少 $l_c$ 個分類器同意才決定；\n$L$ 為奇數時 $l_c=(L+1)/2$。", "decide only if at least $l_c$ classifiers agree;\nfor odd $L$, $l_c=(L+1)/2$.")},
                 {"title": tr("二項機率", "Binomial probability"), "role": "blue", "name": tr("term:二項機率", "term:Binomial probability"), "h": 1.35, "text": tr("$L$ 次獨立試驗、每次成功機率 $p$，\n恰好成功 $m$ 次：$\\binom{L}{m}p^m(1-p)^{L-m}$。\n$P_c(L)=\\sum_{m=l_c}^{L}$ 這個機率。", "$L$ independent trials, success chance $p$,\nexactly $m$ successes: $\\binom{L}{m}p^m(1-p)^{L-m}$.\n$P_c(L)=\\sum_{m=l_c}^{L}$ of it.")},
                 {"title": tr("例：p=0.6、L=3", "Example: p=0.6, L=3"), "role": "orange", "h": 1.0, "text": tr("$l_c=2$：$3\\times0.6^2\\times0.4+0.6^3$\n$=0.432+0.216=0.648$", "$l_c=2$: $3\\times0.6^2\\times0.4+0.6^3$\n$=0.432+0.216=0.648$")}],
        "takeaway": tr("獨立假設下：p>0.5 時 L 越大越好，p<0.5 越大越糟，p=0.5 恆為 0.5。", "Under independence: for p>0.5 more is better, for p<0.5 more is worse, and p=0.5 stays 0.5."),
    }
    return "plot_xy", spec


def illusion():
    p = C["single_mean"]; th = Pc(25, p)
    spec = {
        "id": "lec05-kp24-illusion",
        "alt": tr(f"20 份訓練集、每份 25 棵 bagging 樹：單棵樹平均準確率 {r1(p*100)}%；代入獨立公式 P_c(25,p) 預測 99.99%；實際多數投票只有 {r1(C['vote']*100)}%；成員與多數決不同的比例 V={C['V']:.3f}。這些樹用同一份資料的重抽樣訓練，錯誤高度共享。",
                  f"20 training sets, 25 bagged trees each: the single-tree mean accuracy is {r1(p*100)}%; the independence formula P_c(25,p) predicts 99.99%; the actual majority vote gets only {r1(C['vote']*100)}%; members differ from the majority in a fraction V={C['V']:.3f}. The trees are trained on resamples of the same data, so their errors are highly shared."),
        "unit": tr("新資料準確率（%）", "accuracy on new data (%)"),
        "bars": [{"label": tr("單棵樹的平均", "single tree mean"), "value": round(p * 100, 1), "role": "orange", "suffix": "%", "note": "p"},
                 {"label": tr("獨立公式的預測", "independence formula"), "value": round(C["theory"] * 100, 2), "role": "red", "suffix": "%", "note": "P_c(25,p)"},
                 {"label": tr("實際多數投票", "actual majority vote"), "value": round(C["vote"] * 100, 1), "role": "green", "suffix": "%", "note": f"V = {C['V']:.3f}"}],
        "max": 100, "ticks": [0, 20, 40, 60, 80, 100],
        "takeaway": tr("公式預測 99.99%，實際只有 87.7%：這些樹用同一份資料重抽樣訓練，錯誤高度共享。", "Formula 99.99%, actual 87.7%: the trees resample one data set, so their errors are shared."),
    }
    return "bars_compare", spec


def diversity():
    s = C["single7"]; B = C["boot"]; R = C["sub"]; D = C["types"]
    pc = lambda a: f"{a[0]*100:.1f} ± {a[1]*100:.1f}%"
    spec = {
        "id": "lec05-kp24-diversity",
        "alt": tr(f"10 維資料、訓練 150 點共 15 次、新資料 2000 點：單棵樹（深度 3）{s*100:.1f}%；bootstrap 樹 11 棵 {pc(B)}，V={B[2]:.3f}；隨機特徵子集 11 棵 {pc(R)}，V={R[2]:.3f}；5 種分類器 {pc(D)}，V={D[2]:.3f}。V 最大的不是最準的。",
                  f"10-dimensional data, 150 training points 15 times, 2000 new points: single tree (depth 3) {s*100:.1f}%; 11 bootstrap trees {pc(B)}, V={B[2]:.3f}; 11 random-feature trees {pc(R)}, V={R[2]:.3f}; 5 classifier types {pc(D)}, V={D[2]:.3f}. The largest V is not the most accurate."),
        "columns": [
            {"role": "blue", "header": tr("多樣性 V", "Diversity V"), "items": [{"label": tr("定義", "meaning"), "lines": [tr("成員決策彼此不同的程度；", "how different members' decisions are;"), tr("量法 $V$：與多數決不同的比例", "$V$: share differing from majority")], "name": tr("term:多樣性", "term:Diversity")}, {"label": tr("單棵樹", "baseline"), "lines": [tr(f"深度 3 的單棵樹 {s*100:.1f}%", f"one depth-3 tree {s*100:.1f}%")]}]},
            {"role": "green", "header": "bootstrap", "items": [{"label": tr("做法", "how"), "lines": [tr("有放回重抽樣，同特徵空間、", "resample with replacement,"), tr("不同資料點（11 棵樹）", "same features, other points")], "name": tr("term:bootstrap", "term:Bootstrap")}, {"label": tr("結果", "result"), "lines": [pc(B), f"$V={B[2]:.3f}$"]}]},
            {"role": "orange", "header": tr("不同特徵子空間", "Other feature subspaces"), "items": [{"label": tr("做法", "how"), "lines": [tr("每棵樹從 10 個座標", "each tree gets 4 of the"), tr("隨機挑 4 個（11 棵）", "10 coordinates (11 trees)")]}, {"label": tr("結果", "result"), "lines": [pc(R), f"$V={R[2]:.3f}$"]}]},
            {"role": "red", "header": tr("不同分類器類型", "Other classifier types"), "items": [{"label": tr("做法", "how"), "lines": [tr("logistic、樹、5-NN、", "logistic, trees,"), tr("15-NN、深度 5 的樹", "5-NN, 15-NN, depth 5")]}, {"label": tr("結果", "result"), "lines": [pc(D), f"$V={D[2]:.3f}$"]}]}],
        "takeaway": tr("V 最大的不是最準的；5 種分類器含 logistic，剛好適合這份資料，不能據此說不同分類器一定較好。", "The largest V is not the most accurate; the 5 types include a logistic fit that suits this data."),
    }
    return "columns_text", spec


def bayes():
    spec = {
        "id": "lec05-kp24-bayes",
        "alt": tr("貝氏觀點：假設各分類器看到的 x_j 彼此獨立，貝氏規則得到 max_k P(ω_k)^(1−L) ∏_j P(ω_k|x_j)，又是乘積規則，但對後驗估計的誤差非常敏感；加總較耐誤差，實務上常勝過乘積（來源）。來源的手寫數字實驗：沒有一種規則永遠最好；有的案例結合比最佳單一分類器還差；同一分類器放在不同特徵集上結合效果最好。本講義也看到投票 86.2% 低於 3-NN 87.8%，隨機特徵子集最差。加總與多數投票最常用。",
                  "Bayesian view: assuming the x_j seen by the classifiers are independent, the Bayes rule gives max_k P(ω_k)^(1−L) ∏_j P(ω_k|x_j), again a product rule, very sensitive to errors in the posterior estimates; the sum is more error-tolerant and often beats the product in practice (source). The source's handwritten-digit experiment: no rule is always best; in some cases the combination is worse than the best single classifier; combining the same classifier on different feature sets works best. This lecture also saw the vote at 86.2% below 3-NN at 87.8% and random feature subsets worst. The sum and the majority vote are the most used."),
        "term": tr("貝氏觀點：結合規則沒有萬靈丹", "The Bayesian view: no combination rule is a panacea"), "name": False,
        "definition": tr("假設各分類器看到的 $x_j$ 彼此獨立，貝氏規則得到乘積規則，但對後驗估計的誤差非常敏感；加總較耐誤差。", "If the $x_j$ are independent, Bayes gives a product rule, sensitive to posterior errors; the sum is steadier."),
        "parts": [{"label": tr("貝氏規則", "Bayes rule"), "role": "blue", "text": "$\\max_kP(\\omega_k)^{1-L}\\prod_jP(\\omega_k|x_j)$"},
                  {"label": tr("來源實驗", "Source"), "role": "green", "text": tr("沒有一種規則永遠最好；\n有的案例結合比最佳單一\n分類器還差；同一分類器放\n在不同特徵集上效果最好。", "no rule is always best; sometimes\nthe combination is worse than the\nbest single classifier; the same\nclassifier on different feature\nsets combines best.")},
                  {"label": tr("本講義", "This lecture"), "role": "orange", "text": tr("三者投票 86.2% <\n3-NN 87.8%；隨機特徵\n子集的投票最差。", "vote of three 86.2% <\n3-NN 87.8%; random feature\nsubsets voted worst.")}],
        "takeaway": tr("加總與多數投票最常用；沒有萬靈丹定理。", "The sum and the majority vote are the most used; there is no panacea theorem."),
    }
    return "concept_card", spec


def median():
    outs = [0.80, 0.85, 0.75, 0.01, 0.01]
    sm1 = sum(outs) / 5; sm2 = 1 - sm1
    pr1 = 1.0
    for o in outs: pr1 *= o
    pr2 = 1.0
    for o in outs: pr2 *= 1 - o
    md = sorted(outs)[2]
    alt = tr(f"五個分類器給 ω1：0.80、0.85、0.75、0.01、0.01：加總 ω1 {sm1:.3f}、ω2 {sm2:.3f}，選 ω2（錯）；乘積 ω1 {pr1:.1e}、ω2 {pr2:.1e}，選 ω2（錯）；中位數 ω1 {md:g}、ω2 {1-md:g}，選 ω1（對）。",
             f"Five classifiers give ω1 0.80, 0.85, 0.75, 0.01, 0.01: sum ω1 {sm1:.3f}, ω2 {sm2:.3f}, choose ω2 (wrong); product ω1 {pr1:.1e}, ω2 {pr2:.1e}, choose ω2 (wrong); median ω1 {md:g}, ω2 {1-md:g}, choose ω1 (right).")
    rules = [(tr("加總", "Sum"), f"$\\omega_1:\\ {sm1:.3f}$", f"$\\omega_2:\\ {sm2:.3f}$", sm1, sm2, tr("選 ω2（錯）", "chooses ω2 (wrong)"), False),
             (tr("乘積", "Product"), "$\\omega_1:\\ 5.1\\times10^{-5}$", "$\\omega_2:\\ 7.4\\times10^{-3}$", pr1, pr2, tr("選 ω2（錯）", "chooses ω2 (wrong)"), False),
             (tr("中位數", "Median"), f"$\\omega_1:\\ {md:g}$", f"$\\omega_2:\\ {1-md:g}$", md, 1 - md, tr("選 ω1（對）", "chooses ω1 (right)"), True)]
    f = _rules_page("lec05-kp24-median", alt, outs, tr("五個分類器給 ω₁ 的機率（兩個離群的 0.01）", "probability each of five classifiers gives ω₁ (two outliers at 0.01)"), rules,
                    tr("離群輸出出現時，加總被極端值拉走，中位數較穩健。", "With outliers the sum is pulled away by the extremes, the median stays steady."), 1.1,
                    defn=(tr("中位數規則：取各分類器機率的中位數，再取最大的類別。", "median rule: take the median of the classifiers' probabilities, then pick the largest class."), tr("term:中位數規則", "term:Median rule")))
    return f


def solution():
    l3 = 3 * 0.7 ** 2 * 0.3 + 0.7 ** 3; l5 = 10 * 0.7 ** 3 * 0.3 ** 2 + 5 * 0.7 ** 4 * 0.3 + 0.7 ** 5
    spec = {
        "id": "lec05-kp24-solution",
        "alt": tr(f"p=0.7、獨立：L=3，l_c=2：3·0.7²·0.3+0.7³=0.441+0.343={l3:.3f}；L=5，l_c=3：10·0.7³·0.3²+5·0.7⁴·0.3+0.7⁵=0.3087+0.36015+0.16807={l5:.3f}，比 L=3 多 {l5-l3:.3f}；0.7→{l3:.3f}→{l5:.3f} 單調上升（獨立假設下），真實相依的分類器不會升這麼多。",
                  f"p=0.7, independent: L=3, l_c=2: 3·0.7²·0.3+0.7³=0.441+0.343={l3:.3f}; L=5, l_c=3: 10·0.7³·0.3²+5·0.7⁴·0.3+0.7⁵=0.3087+0.36015+0.16807={l5:.3f}, {l5-l3:.3f} more than L=3; 0.7→{l3:.3f}→{l5:.3f} rises monotonically (under independence); dependent real classifiers do not rise this much."),
        "steps": [{"role": "blue", "title": "① L=3", "lines": ["$l_c=2$：$3\\times0.7^2\\times0.3=0.441$", "$0.7^3=0.343$", "$0.441+0.343$"], "result": f"$P_c(3)={l3:.3f}$"},
                  {"role": "green", "title": "② L=5", "lines": ["$l_c=3$：$10\\times0.7^3\\times0.3^2=0.3087$", "$5\\times0.7^4\\times0.3=0.36015$", "$0.7^5=0.16807$"], "result": f"$P_c(5)={l5:.3f}$"},
                  {"role": "orange", "title": tr("③ 差多少", "③ The gain"), "lines": [f"${l5:.4f}-{l3:.3f}$", tr("$0.7\\to0.784\\to0.837$ 上升", "$0.7\\to0.784\\to0.837$ rises")], "result": f"$+{l5-l3:.3f}$"}],
        "takeaway": tr("獨立假設下單調上升；真實的相依分類器不會升這麼多。", "Monotone under independence; real dependent classifiers do not rise this much."),
    }
    return "steps_flow", spec
