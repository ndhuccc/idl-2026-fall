"""KP10 pages designed in HTML (view-from-text-v3, route A). pageN(lang) -> body HTML of a 1160x480 figure."""
import os, sys, math
import numpy as np
SK = os.environ.get("COURSE_SKILLS") or (os.path.join(os.path.abspath(os.environ["COURSE_REPO"]), ".agents", "skills") if os.environ.get("COURSE_REPO") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, SK + "/view-from-text-v3/examples"); sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, HERE)
from kp17_pages import Axes, tr, take, legend, fmt, sci, txt, named, card, head, chip, formula, svg, BL, RD, VI, GR, YE, OR, TE, INK, TXT, MUT, hexc, build, native, _sup, rgba, rgb
from kp15_pages import box, cardH
import kp10_nums as K
W, H = 1160, 480
def clip(poly, a, b, c):   # keep a*x+b*y+c >= 0
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]; fp, fq = a * p[0] + b * p[1] + c, a * q[0] + b * q[1] + c
        if fp >= 0: out.append(p)
        if (fp >= 0) != (fq >= 0): u = fp / (fp - fq); out.append((p[0] + u * (q[0] - p[0]), p[1] + u * (q[1] - p[1])))
    return out
def dline(ax, x0, y0, x1, y1, col, w=2, dash="7 6"): ax.raw(f'<line x1="{ax.X(x0):.1f}" y1="{ax.Y(y0):.1f}" x2="{ax.X(x1):.1f}" y2="{ax.Y(y1):.1f}" stroke="{col}" stroke-width="{w}" stroke-dasharray="{dash}"/>'); return ax

# ------------------------------------------------ 1: memorizing vs learning
def page1(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 760, 404, "blue", t("同一批帶雜訊的樣本：平滑規律與背熟的折線（補充示意）", "The same noisy samples: a smooth trend versus a memorizing polyline (illustrative)"), 17)
    ax = Axes(70, 62, 670, 280, (0, 1), (-1.6, 1.6)); ax.frame([0, .25, .5, .75, 1], [-1, 0, 1], xnames={0: "0", .25: "0.25", .5: "0.5", .75: "0.75", 1: "1"}, yfmt=lambda v: f"{v:g}".replace("-", "−"))
    xs = np.linspace(0, 1, 60); ax.line([(float(a), float(math.sin(2 * math.pi * a))) for a in xs], GR, 5)
    x, y, c = K.sine_demo(); ax.line([(float(a), float(b)) for a, b in zip(x, y)], RD, 3, "9 6")
    for a, b in zip(x, y): ax.dot(float(a), float(b), "#e0c85f", 7)
    o += ax.html() + legend([(GR, t("平滑規律 sin(2πx)", "smooth trend sin(2πx)")), (RD, t("折線：穿過每個點", "polyline through every point"))], 470, 76, 270)
    o += txt(12, 372, 736, t("黃點是帶雜訊的 10 個樣本；折線把雜訊也背熟了。〔補充示意，不是網路訓練結果〕", "Yellow: 10 noisy samples; the polyline memorized the noise too. (illustration only)"), 16, MUT, False, "center", 24)
    o += cardH(776, 0, 384, 404, "violet", t("名詞", "Terms"))
    o += named(790, 50, 356, t("泛化：對沒參與訓練的新資料仍能正確預測；不是把原來的答案背熟。", "Generalization: still predicting new data that was not used in training correctly; not memorizing the original answers."), "term:" + t("泛化", "Generalization"), 17, TXT, 100)
    o += card(790, 170, 356, 220, "yellow", .12) + txt(798, 180, 340, t("控制網路大小，就是避免只背熟訓練資料。", "Controlling size means avoiding memorizing the training data."), 17, TXT, True, "left", 24)
    o += txt(798, 262, 340, t("分類也有同樣取捨：先找主要規律，再看樣本的小波動；不必把每個波動都當成規律。", "Classification has the same trade-off: find the main pattern first; not every wiggle is a rule."), 17, TXT, False, "left", 24)
    o += take(t, "訓練分數很高，不代表新資料也準：要控制大小，先分清「背熟」與「學會」。", "A high training score does not mean new data is predicted well: tell memorizing from learning first.", 418)
    return o

# ------------------------------------------------ 2: counting free parameters
def page2(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 470, 404, "blue", t("2–3–1 網路：全部相連、每個計算單元有偏置", "A 2–3–1 network: fully connected, every unit has a bias"), 16)
    ins = [(80, 140), (80, 280)]; hid = [(235, 100), (235, 210), (235, 320)]; out = [(390, 210)]
    s = ""
    for a in ins:
        for b in hid: s += f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#6a9fd6" stroke-width="2" opacity="0.7"/>'
    for b in hid: s += f'<line x1="{b[0]}" y1="{b[1]}" x2="{out[0][0]}" y2="{out[0][1]}" stroke="#d9822b" stroke-width="2" opacity="0.7"/>'
    for (x, y) in ins: s += f'<circle cx="{x}" cy="{y}" r="22" fill="#47688f" stroke="#ffffff" stroke-width="2"/>'
    for (x, y) in hid: s += f'<circle cx="{x}" cy="{y}" r="22" fill="#4f9a68" stroke="#ffffff" stroke-width="2"/>'
    for (x, y) in out: s += f'<circle cx="{x}" cy="{y}" r="22" fill="#a8452b" stroke="#ffffff" stroke-width="2"/>'
    o += svg(0, 0, W, H, s)
    o += txt(30, 128, 100, "x₁", 18, TXT, True, "center", 26) if False else ""
    o += txt(20, 128, 40, "x₁", 18, TXT, True, "center", 26) + txt(20, 268, 40, "x₂", 18, TXT, True, "center", 26) + txt(360, 160, 60, "ŷ", 18, TXT, True, "center", 26)
    o += chip(30, 352, 130, "blue", t("輸入 d = 2", "inputs d = 2"), False, 30, 16) + chip(170, 352, 130, "green", t("隱藏 3 個", "3 hidden"), False, 30, 16) + chip(310, 352, 140, "orange", t("輸出 1 個", "1 output"), False, 30, 16)
    o += cardH(486, 0, 320, 404, "green", t("自由參數的計數", "Counting free parameters"), 17)
    o += card(498, 50, 296, 90, "green", .16) + txt(498, 56, 296, t("隱藏層：每個單元 2 個權重 + 1 個偏置", "Hidden: 2 weights + 1 bias each"), 16, TXT, True, "center", 24) + formula(498, 92, 296, r"(2+1)\times3=9", 24, 40)
    o += card(498, 152, 296, 90, "orange", .16) + txt(498, 158, 296, t("輸出層：3 個權重 + 1 個偏置", "Output: 3 weights + 1 bias"), 16, TXT, True, "center", 24) + formula(498, 194, 296, r"(3+1)\times1=4", 24, 40)
    o += card(498, 254, 296, 100, "yellow", .2) + formula(498, 270, 296, r"9+4=13", 32, 50) + txt(498, 326, 296, t("不是只有 3 個隱藏單元", "not just the 3 hidden units"), 16, TXT, True, "center", 24)
    o += cardH(822, 0, 338, 404, "violet", t("名詞", "Terms"))
    o += named(834, 50, 316, t("自由參數：訓練時可以獨立改變的數字；權重與偏置都算。", "Free parameter: a number that training can change independently; weights and biases both count."), "term:" + t("自由參數", "Free parameter"), 16, TXT, 76)
    o += named(834, 150, 316, t("欠擬合：連主要規律都表達不好，訓練與新資料的表現都可能差。", "Underfitting: the model cannot even express the main pattern, so performance can be poor on both training and new data."), "term:" + t("欠擬合", "Underfitting"), 16, TXT, 100)
    o += named(834, 272, 316, t("過擬合：配合訓練資料的偶然細節，反而損害新資料的預測。", "Overfitting: fitting chance details of the training data, which hurts predictions on new data."), "term:" + t("過擬合", "Overfitting"), 16, TXT, 100)
    o += take(t, "「較大」不是過擬合的充分條件：資料量、權重大小、限制方式與訓練程序都會影響結果。", "\"Larger\" is not sufficient for overfitting: data size, weight size, constraints and training all matter.", 418)
    return o

# ------------------------------------------------ 3: validation curve / early stopping
def page3(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 700, 404, "orange", t("每個檢查點的訓練損失與驗證損失（手編教學數字）", "Training and validation loss at each checkpoint (hand-made teaching numbers)"), 17)
    ax = Axes(90, 66, 580, 250, (.6, 5.4), (0, .62)); ax.frame([1, 2, 3, 4, 5], [0, .2, .4, .6], yfmt=lambda v: f"{v:.1f}")
    ep = [1, 2, 3, 4, 5]; ax.line(list(zip(ep, K.TRAIN)), BL, 4).line(list(zip(ep, K.VAL)), OR, 4)
    dline(ax, 3, 0, 3, .62, "#e0c85f", 2)
    for e, a, b in zip(ep, K.TRAIN, K.VAL): ax.dot(e, a, BL, 6).dot(e, b, OR, 6)
    ax.text(3, .25, "0.25", "#e0c85f", "start", 18, True, dx=16, dy=-16); ax.text(5, .4, "0.40", OR, "end", 17, True, dx=-12, dy=-12); ax.text(5, .10, "0.10", BL, "end", 17, True, dx=-12, dy=26)
    ax.raw(f'<text x="{ax.x+ax.w/2}" y="{ax.y+ax.h+50}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">{t("訓練輪數", "epoch")}</text>')
    o += ax.html() + legend([(BL, t("訓練損失", "training loss")), (OR, t("驗證損失", "validation loss"))], 110, 262, 170)
    o += txt(12, 372, 676, t("虛線是要保存的第 3 輪；兩條曲線不必相交，差距也不必從零開始。", "The dashed line marks epoch 3, the one to keep; the curves need not cross and the gap need not start at zero."), 16, MUT, False, "center", 24)
    o += cardH(716, 0, 444, 404, "violet", t("名詞與判讀", "Terms and reading"))
    o += named(728, 50, 420, t("驗證集：不拿來更新權重，用來選模型或訓練檢查點的資料。", "Validation set: data not used to update weights, used to pick a model or a checkpoint."), "term:" + t("驗證集", "Validation set"), 16, TXT, 76)
    o += named(728, 130, 420, t("過度訓練：架構固定，後續訓練開始追逐訓練集的細節，驗證表現反而惡化。", "Overtraining: with a fixed architecture, later training chases training-set details and validation performance worsens."), "term:" + t("過度訓練", "Overtraining"), 16, TXT, 100)
    o += named(728, 236, 420, t("早停：依驗證表現停止訓練並還原最佳檢查點；實作可容許幾次沒改善。", "Early stopping: stop training by validation performance and restore the best checkpoint; implementations may allow a few non-improving checks."), "term:" + t("早停", "Early stopping"), 16, TXT, 100)
    o += chip(728, 346, 420, "yellow", t("驗證最小 0.25 在第 3 輪 → 保留第 3 輪", "minimum 0.25 at epoch 3 → keep epoch 3"), True, 40, 17)
    o += take(t, "第 5 輪訓練損失最低（0.10），驗證損失卻從 0.25 升到 0.40，增加 0.40 − 0.25 = 0.15。", "Epoch 5 has the lowest training loss (0.10) but validation rose from 0.25 to 0.40, an increase of 0.40 − 0.25 = 0.15.", 418)
    return o

# ------------------------------------------------ 4: roles of the three data sets
def page4(lang):
    t = tr(lang); o = ""
    cols = [("blue", t("訓練集", "Training set"), t("用它更新權重，學到資料中的規律。", "Used to update the weights and learn the pattern."), t("不能拿來判斷新資料準不準：它已經見過了。", "Cannot judge accuracy on new data: it has been seen already.")),
            ("orange", t("驗證集", "Validation set"), t("用它選大小、懲罰強度與停止點。", "Used to choose the size, the penalty strength and the stopping point."), t("不直接更新權重。", "Never updates the weights directly.")),
            ("green", t("測試集", "Test set"), t("模型選完之後才做最後評估。", "Used for the final evaluation only after the model is chosen."), t("反覆看它來挑模型，就不再是獨立檢查；也要避免重複個體或時間資訊洩漏。", "Reusing it to pick models ends its independence; also avoid duplicate or time-information leakage."))]
    for i, (role, name, a, b) in enumerate(cols):
        x0 = i * 392; o += cardH(x0, 0, 376, 262, role, name, 19)
        o += chip(x0 + 12, 50, 352, role, t("用來做什麼", "What it is for"), False, 28, 16) + txt(x0 + 12, 86, 352, a, 17, TXT, False, "left", 24)
        o += chip(x0 + 12, 146, 352, "red", t("不能拿來做什麼", "What it must not do"), False, 28, 16) + txt(x0 + 12, 182, 352, b, 17, TXT, False, "left", 24)
    o += cardH(0, 274, 1160, 130, "yellow", t("小樣本的數字很粗", "Small-sample numbers are coarse"), 17)
    o += card(14, 318, 360, 76, "yellow", .16) + formula(14, 322, 360, r"\frac{1}{10}=0.1", 24, 40) + txt(14, 362, 360, t("十筆中錯一筆：錯誤率 0.1", "one error in ten: error rate 0.1"), 16, TXT, False, "center", 24)
    o += card(386, 318, 360, 76, "yellow", .16) + formula(386, 322, 360, r"\frac{2}{10}=0.2", 24, 40) + txt(386, 362, 360, t("錯兩筆：錯誤率 0.2", "two errors: error rate 0.2"), 16, TXT, False, "center", 24)
    o += txt(758, 318, 390, t("只差一個案例，估計就差十個百分點。重複切分或交叉驗證可輔助，但每次仍須把訓練與評估分開。", "One more case moves the estimate ten points. Repeated splits or cross-validation help; keep training and evaluation apart."), 16, TXT, False, "left", 24)
    o += take(t, "常見迷思：參數更少、或訓練更久，並不各自保證新資料更準；也沒有脫離資料量與任務的「最佳神經元數」。", "Myth: fewer parameters or longer training do not each guarantee better accuracy; no universal best neuron count.", 418)
    return o

# ------------------------------------------------ 5: region upper bound
def page5(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 640, 404, "teal", t("K 個超平面在 d 維切出的區域數上限", "Maximum regions cut by K hyperplanes in d dimensions"), 17)
    ax = Axes(76, 62, 530, 280, (0.5, 6.5), (0, 46)); ax.frame([1, 2, 3, 4, 5, 6], [0, 10, 20, 30, 40])
    for d, col in ((1, TE), (2, OR), (3, RD)):
        pts = [(Kk, K.M(Kk, d)) for Kk in range(1, 7)]; ax.line(pts, col, 4)
        for a, b in pts: ax.dot(a, b, col, 6)
        ax.text(6, K.M(6, d), f"d = {d}: {K.M(6, d)}", col, "end", 17, True, dx=-14, dy=(-14 if d != 2 else 42))
    ax.raw(f'<text x="{ax.x+ax.w/2}" y="{ax.y+ax.h+50}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">{t("超平面數 K", "number of hyperplanes K")}</text>')
    o += ax.html()
    o += cardH(656, 0, 504, 404, "violet", t("公式與數值表", "Formula and value table"), 17)
    o += card(668, 48, 480, 72, "violet", .2) + formula(668, 54, 480, r"M_{\max}(K,d)=\sum_{m=0}^{d}\binom{K}{m}", 26, 66)
    o += txt(668, 128, 480, t("C(K, m)：從 K 個挑 m 個的方法數，m > K 時為 0。一般位置（沒有重合、平行、多線共點）才達到上限。", "C(K, m): ways to choose m of K, 0 when m > K. The bound is reached only in general position (no coincident, parallel or concurrent hyperplanes)."), 16, TXT, False, "left", 24)
    xs = [775, 855, 935, 1015]
    o += txt(668, 214, 80, "d \\ K", 16, MUT, True, "center", 24)
    for j, Kk in enumerate((1, 2, 3, 4)): o += txt(xs[j], 214, 70, str(Kk), 16, MUT, True, "center", 24)
    for i, d in enumerate((1, 2, 3)):
        y0 = 244 + i * 38; o += card(668, y0, 480, 34, "teal", .1) + txt(668, y0 + 4, 80, str(d), 17, TXT, True, "center", 26)
        for j, Kk in enumerate((1, 2, 3, 4)): o += txt(xs[j], y0 + 4, 70, str(K.M(Kk, d)), 17, TXT, d == 2 and Kk == 3, "center", 26)
    o += chip(668, 364, 480, "yellow", t("d = 2、K = 2：1 + 2 + 1 = 4（選 0、1、2 條線）", "d = 2, K = 2: 1 + 2 + 1 = 4"), True, 30, 16)
    o += take(t, "幾何計數只回答「架構最多能切出幾塊」，不回答「預測好不好」；兩者是不同的問題。", "Geometric counting says how many regions at most, not whether predictions are good: different questions.", 418)
    return o

# ------------------------------------------------ 6: XOR construction
def page6(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 600, 404, "blue", t("XOR：兩條平行線切出三條帶", "XOR: two parallel lines cut three bands"), 17)
    ax = Axes(70, 56, 380, 300, (-.4, 1.4), (-.4, 1.4))
    s = ""
    for c0, c1, col in ((-3, .5, OR), (.5, 1.5, BL), (1.5, 3.5, OR)):
        P_ = clip(clip([(-.4, -.4), (1.4, -.4), (1.4, 1.4), (-.4, 1.4)], 1, 1, -c0), -1, -1, c1)
        if len(P_) >= 3: s += f'<polygon points="{" ".join(f"{ax.X(x):.1f},{ax.Y(y):.1f}" for x, y in P_)}" fill="{col}" fill-opacity="0.32"/>'
    ax.raw(s); ax.frame([0, 1], [0, 1])
    for c in (.5, 1.5): ax.line([(max(-.4, c - 1.4), c - max(-.4, c - 1.4)), (min(1.4, c + .4), c - min(1.4, c + .4))], "#e0c85f", 4)
    for (a, b) in ((0, 1), (1, 0)): ax.dot(a, b, "#ffffff", 14).dot(a, b, "#6a9fd6", 10)
    for (a, b) in ((0, 0), (1, 1)): ax.dot(a, b, "#ffffff", 14).dot(a, b, "#d97757", 10)
    o += ax.html()
    o += txt(ax.X(0) + 14, ax.Y(1) - 34, 90, "A (0,1)", 16, TXT, True, "left", 24) + txt(ax.X(1) - 30, ax.Y(0) + 14, 90, "A (1,0)", 16, TXT, True, "left", 24)
    o += txt(ax.X(0) + 14, ax.Y(0) + 14, 90, "B (0,0)", 16, TXT, True, "left", 24) + txt(ax.X(1) - 50, ax.Y(1) - 36, 90, "B (1,1)", 16, TXT, True, "left", 24)
    o += txt(ax.X(1.4) + 6, ax.Y(.1) - 28, 130, "x₁ + x₂ = 1.5", 16, "#e0c85f", True, "left", 24) + txt(ax.X(.9) + 18, ax.Y(-.4) - 30, 130, "x₁ + x₂ = 0.5", 16, "#e0c85f", True, "left", 24)
    o += cardH(616, 0, 544, 404, "orange", t("三條帶與四個點的和 x₁ + x₂", "Three bands and the sum x₁ + x₂ at the four points"), 17)
    rows = [("(0, 0)", 0, "B", t("x₁ + x₂ < 0.5：下方帶", "below 0.5: lower band"), "orange"), ("(0, 1)", 1, "A", t("0.5 < x₁ + x₂ < 1.5：中間帶", "between 0.5 and 1.5: middle band"), "blue"), ("(1, 0)", 1, "A", t("0.5 < x₁ + x₂ < 1.5：中間帶", "between 0.5 and 1.5: middle band"), "blue"), ("(1, 1)", 2, "B", t("x₁ + x₂ > 1.5：上方帶", "above 1.5: upper band"), "orange")]
    for i, (p, sm, cl, desc, role) in enumerate(rows):
        y0 = 52 + i * 52; o += card(628, y0, 520, 44, role, .16) + txt(634, y0 + 8, 84, p, 17, TXT, True, "center", 28) + txt(722, y0 + 8, 50, f"{sm}", 17, TXT, False, "center", 28) + txt(776, y0 + 8, 36, cl, 17, TXT, True, "center", 28) + txt(816, y0 + 8, 326, desc, 16, TXT, False, "left", 28)
    o += txt(628, 264, 520, t("KP01 的 x₁ + x₂ = 0.5 與 1.5 平行，只切成三條帶，兩個 A 點住在中間那條。", "The KP01 lines x₁ + x₂ = 0.5 and 1.5 are parallel and cut only three bands; both A points live in the middle one."), 17, TXT, False, "left", 24)
    o += card(628, 330, 520, 64, "yellow", .16) + txt(634, 336, 508, t("「四塊上界容得下三塊」只是容量檢查；兩個單元夠不夠，要靠這個明確構造來證明。", "\"The 4-region bound holds 3 regions\" is only a capacity check; the explicit construction proves two units suffice."), 16, TXT, True, "center", 48)
    o += take(t, "上界沒有告訴我們怎麼找到適合的大小；兩個隱藏單元能解 XOR，靠的是明確的構造。", "The bound does not tell us how to find a suitable size; two hidden units solve XOR by an explicit construction.", 418)
    return o

# ------------------------------------------------ 7: three strategies
def page7(lang):
    t = tr(lang); o = ""
    cs = [("blue", t("解析法", "Analytic methods"), t("用代數或統計分析估計網路的能力。", "Estimate the network's capability with algebra or statistics."), t("限制：通常忽略具體的最佳化過程。", "Limit: usually ignores the actual optimization process.")),
          ("orange", t("剪枝", "Pruning"), t("先訓練較大的網路，再依規則刪掉較不重要的連線（下一個 KP）。", "Train a larger network first, then delete the less important connections by a rule (next KP)."), t("要點：刪完仍須檢查表現。", "Point: still check performance after deleting.")),
          ("green", t("構造式方法", "Constructive methods"), t("從小架構出發，逐步增加單元（KP05）。", "Start from a small architecture and add units step by step (KP05)."), t("要點：每加一步都要檢查。", "Point: check after every addition."))]
    for i, (role, name, a, b) in enumerate(cs):
        x0 = i * 392; o += cardH(x0, 0, 376, 340, role, name, 19)
        o += chip(x0 + 12, 52, 352, role, t("做法", "Approach"), False, 28, 16) + txt(x0 + 12, 90, 352, a, 18, TXT, False, "left", 24)
        o += chip(x0 + 12, 200, 352, "yellow", t("提醒", "Reminder"), False, 28, 16) + txt(x0 + 12, 238, 352, b, 18, TXT, False, "left", 24)
    o += card(0, 352, 1160, 56, "violet", .18) + txt(14, 362, 1132, t("三條路都需要用驗證資料檢查：沒有一種方法能跳過這一步。", "All three routes need checking on validation data: none can skip that step."), 19, TXT, True, "center", 34)
    return o

# ------------------------------------------------ 8: the source's sine demo
def page8(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 580, 404, "blue", t("來源正弦示範的設定", "Setup of the source's sine demo"), 18)
    o += formula(10, 46, 560, r"y=\sin(2\pi x)+\epsilon", 26, 46) + txt(14, 96, 552, t("ε 是加到觀測上的雜訊", "ε is noise added to the observations"), 16, MUT, False, "center", 24)
    o += card(14, 130, 262, 100, "teal", .16) + txt(14, 138, 262, t("訓練點", "training points"), 17, TXT, True, "center", 26) + txt(14, 168, 262, "15", 40, TXT, True, "center", 52)
    o += card(292, 130, 274, 100, "orange", .16) + txt(292, 138, 274, t("獨立測試點", "independent test points"), 17, TXT, True, "center", 26) + txt(292, 168, 274, "10", 40, TXT, True, "center", 52)
    o += formula(10, 244, 560, r"\hat{y}=\sum_{j=1}^{k}v_j\tanh(a_jx+b_j)+c", 24, 54)
    o += txt(14, 306, 552, t("k 是隱藏單元數；每個單元的 a、b、v 與輸出偏置 c 都可調。", "k is the number of hidden units; each unit's a, b, v and the bias c are adjustable."), 16, TXT, False, "center", 24)
    o += txt(14, 346, 552, t("每個單元 3 個參數，再加輸出偏置 c：參數數 3k + 1。", "3 per unit + output bias c: 3k + 1 parameters."), 17, TXT, True, "center", 28)
    o += cardH(596, 0, 564, 404, "orange", t("不同 k 的參數數與來源的描述", "Parameter counts for several k and the source's description"), 17)
    o += txt(608, 52, 90, "k", 16, MUT, True, "center", 24) + txt(700, 52, 200, t("參數數 3k + 1", "parameters 3k + 1"), 16, MUT, True, "center", 24) + txt(912, 52, 236, t("來源描述", "the source says"), 16, MUT, True, "center", 24)
    for i, (k, say, role) in enumerate(((3, t("約 3 到 5 較合適", "about 3 to 5 suits"), "green"), (5, t("約 3 到 5 較合適", "about 3 to 5 suits"), "green"), (15, t("可能過擬合", "may overfit"), "red"))):
        y0 = 84 + i * 64; o += card(608, y0, 540, 54, role, .18) + txt(608, y0 + 12, 90, str(k), 20, TXT, True, "center", 30) + txt(700, y0 + 12, 200, f"3 × {k} + 1 = {K.params(2, k) if False else 3 * k + 1}", 18, TXT, False, "center", 30) + txt(912, y0 + 12, 236, say, 17, TXT, True, "center", 28)
    o += card(608, 286, 540, 100, "yellow", .16) + txt(616, 294, 524, t("這不是任何資料的通用最佳值：只說明來源的設定，不把一次實驗當成通用最佳設定；本教材沒有重跑這個訓練。", "Not a universal best value: this only describes the source's setup; the training was not rerun."), 17, TXT, True, "center", 84)
    o += take(t, "來源的正弦示範是連續預測；k 只有 3 到 5 與 15 兩種說法，其餘都要用驗證資料自己檢查。", "Only k of 3 to 5 and k = 15 are described; check any other k on validation data.", 418)
    return o

# ------------------------------------------------ 9: extension: recursion + MSE + code
def page9(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 640, 404, "teal", t("為什麼區域數是這個形式：遞迴", "Why the region count has this form: a recursion"), 17)
    o += txt(12, 48, 616, t("在一般位置加入第 K 個超平面，它把某些區域切成兩半；新切出的部分，對應其他 K − 1 個超平面在新平面內形成的低一維區域。", "Adding the K-th hyperplane (general position) splits some regions in two; the new pieces match the lower-dimensional regions the other K − 1 hyperplanes form in the new plane."), 16, TXT, False, "left", 24)
    o += card(14, 176, 612, 56, "teal", .2) + formula(14, 178, 612, r"R(K,d)=R(K-1,d)+R(K-1,d-1)", 24, 54)
    o += txt(14, 238, 612, t("起點 R(0, d) = 1、R(K, 0) = 1；用組合數的 Pascal 關係得 R(K, d) = Σ C(K, m)。", "Start: R(0, d) = 1, R(K, 0) = 1; Pascal's rule for binomials gives R(K, d) = Σ C(K, m)."), 16, MUT, False, "center", 48)
    o += card(14, 292, 612, 54, "green", .16) + formula(14, 294, 612, r"R(3,2)=R(2,2)+R(2,1)=4+3=7", 24, 56)
    o += txt(14, 352, 612, t("與 1 + 3 + 3 = 7 一致；只計空間分割，不含輸出層能否把任意區域標籤分對。〔程式計算〕", "Agrees with 1 + 3 + 3 = 7; counts space partitions only, not whether the output layer can label any region correctly. (computed)"), 16, TXT, False, "center", 48)
    o += cardH(656, 0, 504, 190, "orange", t("連續預測的平均平方誤差", "Mean squared error for continuous prediction"), 17)
    o += formula(664, 46, 488, r"L=\frac{1}{N}\sum_{n=1}^{N}(\hat{y}_n-t_n)^2", 22, 56)
    o += formula(664, 104, 488, r"\frac{1^2+(-1)^2}{2}=1", 22, 52)
    o += txt(664, 156, 488, t("分類要另外指定門檻與錯誤率，兩者不是同一個量。", "Classification needs a threshold and an error rate instead."), 16, TXT, False, "center", 24)
    o += cardH(656, 202, 504, 202, "violet", t("自己可執行的檢查點選擇", "A checkpoint selection you can run"), 17)
    o += card(668, 246, 480, 100, "grey", .12) + txt(676, 250, 464, "validation = [0.55, 0.35, 0.25, 0.28, 0.40]<br>best = validation.index(min(validation)) + 1<br>assert best == 3", 15 if False else 16, TXT, False, "left", 24)
    o += txt(668, 354, 480, t("只選已提供的檢查點，沒有訓練神經網路。", "It only picks among given checkpoints; no network is trained."), 16, MUT, False, "center", 24)
    o += take(t, "真實訓練還要保存與還原該輪的權重、固定資料切分，並記錄評估指標。", "Real training also saves and restores that epoch's weights, fixes the split and records the metric.", 418)
    return o

# ------------------------------------------------ 10: exercises
def page10(lang):
    t = tr(lang); o = ""
    qs = [("blue", t("題 1", "Q1"), t("訓練損失 0.40、0.20、0.10，驗證損失 0.45、0.25、0.30。選哪一輪？", "Training 0.40, 0.20, 0.10;<br>validation 0.45, 0.25, 0.30.<br>Which epoch?"), [r"\min\{0.45,\ 0.25,\ 0.30\}=0.25", r"0.30-0.25=0.05"], t("保留第 2 輪。", "Keep epoch 2."), t("核對：最後一輪的訓練值較低也不改選擇。", "Check: the last epoch's lower training value changes nothing.")),
          ("red", t("題 2", "Q2"), t("二維三條一般位置直線的最大區域數是多少？三線平行還是這個數嗎？", "Most regions of three lines in<br>general position in 2-D?<br>Still so if parallel?"), [r"M_{\max}=\binom{3}{0}+\binom{3}{1}+\binom{3}{2}", r"=1+3+3=7"], t("最多 7 塊；三線平行只有 4 條帶。", "At most 7; three parallel lines give only 4 bands."), t("核對：選檢查點時，幾何區域數沒有參與比較。", "Check: region count plays no part in the choice."))]
    for i, (role, name, q, tex, ans, chk) in enumerate(qs):
        x = i * 590; o += card(x, 0, 570, 470, role) + head(x, 0, 570, role, name, 36, 18)
        o += txt(x + 16, 52, 538, q, 19, TXT, True, "left", 84)
        o += chip(x + 16, 150, 130, "grey", t("解題示範", "Solution"), False, 30, 16)
        o += card(x + 16, 194, 538, 150, "grey", .1) + formula(x + 16, 200, 538, tex[0], 22, 66) + formula(x + 16, 274, 538, tex[1], 22, 66)
        o += chip(x + 16, 364, 538, role, ans, True, 42, 20)
        o += txt(x + 16, 420, 538, chk, 17, MUT, False, "left", 48)
    return o

PAGES_DESIGN = [(f"p{i}", globals()[f"page{i}"]) for i in range(1, 11)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:]
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp10_" + fid, fn, W, H, out)
