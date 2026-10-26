"""KP25 (Boosting and AdaBoost) -- sixth sample of view-from-text-v2. Plan: kp25_plan.json. Hand-computed numbers are calculated here;
the experiment curves come from kp25_cache.json (built by kp25_cache_build.py with kp25_nums.py, the same settings as the lecture)."""
import os, sys, json
from math import log, log10, exp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from figkit.parts import T, NFig, header_chip, panel, chip, outline_box, takeaway, tint, solid, INK


def tr(zh, en): return {"zh": zh, "en": en}


def alpha(P): return 0.5 * log((1 - P) / P)


C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kp25_cache.json")))
# exact ties such as 5.75% must round half up (5.8), like the lecture: nudge every cached value by 1e-9 before it is formatted
C = {k: ([x + 1e-9 for x in v] if isinstance(v, list) else {kk: [x + 1e-9 for x in vv] for kk, vv in v.items()}) for k, v in C.items()}
LOGR = sorted(set([max(1, round(10 ** (i / 20))) for i in range(0, 67)] + [17, 30, 36, 2000]))
XT = {"0": "1", "1": "10", "2": "100", "3": "1000"}


def weak():
    st = [e * 100 for e in C["stump"]]; mean = sum(st) / 20
    spec = {
        "id": "lec05-kp25-weak",
        "alt": tr(f"20 個「x_j>0？」決策樁的測試錯誤：{min(st):.0f}% 到 {max(st):.0f}%，平均 {mean:.1f}%，都低於 50%（亂猜）但只好一點。AdaBoost 的機制：每輪換一個樣本權重分布，給最難的點更多權重，最後加權投票。",
                  f"Test errors of 20 stumps 'x_j>0?': {min(st):.0f}% to {max(st):.0f}%, mean {mean:.1f}%, all below 50% (guessing) but only a little. The AdaBoost mechanism: each round changes the sample weights, giving the hardest points more weight, and the final vote is weighted."),
        "plot": {"x": [0, 25], "y": [0, 55], "x_ticks": [1, 5, 10, 15, 20], "y_ticks": [0, 10, 20, 30, 40, 50], "x_label": tr("第 j 個決策樁", "stump j"), "y_label": tr("測試錯誤（%）", "test error (%)"), "w": 430, "h": 300},
        "series": [{"kind": "hline", "y": 50, "role": "red", "width": 2, "label": tr("亂猜", "guess")}, {"kind": "hline", "y": mean, "role": "blue", "width": 2, "label": tr("平均", "mean")},
                   {"kind": "dots", "points": [(j + 1, e) for j, e in enumerate(st)], "role": "orange", "d": 13}],
        "side": [{"title": tr("弱分類器", "Weak classifier"), "role": "orange", "name": tr("term:弱分類器", "term:Weak classifier"), "h": 1.0, "text": tr("錯誤率只比亂猜（0.5）\n好一點點的分類器。", "a classifier whose error is only\na little below guessing (0.5).")},
                 {"title": tr("決策樁", "Stump"), "role": "blue", "name": tr("term:決策樁", "term:Stump"), "h": 1.0, "text": tr("只問一個特徵的一個問題，\n例如「$x_j>0$？」的樹。", "a tree with one question on one\nfeature, such as \"$x_j>0$?\".")},
                 {"title": tr("提升的機制", "How boosting works"), "role": "green", "h": 1.15, "text": tr("每輪換一個樣本權重分布，\n給最難（被分錯）的點更多權重，\n最後加權投票。", "each round changes the sample weights,\nmore weight on the hardest points,\nthen a weighted vote.")}],
        "takeaway": tr(f"20 個決策樁的測試錯誤都在 {min(st):.0f}% 到 {max(st):.0f}%，平均 {mean:.1f}%：每個都只比亂猜好一點。", f"The 20 stumps all err {min(st):.0f}% to {max(st):.0f}% (mean {mean:.1f}%): each is only a little better than guessing."),
    }
    return "plot_xy", spec


def born():
    f = NFig("lec05-kp25-born", 1160, 440, T(tr("樣本權重的來源：逐級最佳化，每次只最佳化新加入的分類器：F_m=F_{m−1}+αφ。指數損失（分對罰小、分錯罰重）拆開：exp(−yF_m)=exp(−yF_{m−1})·exp(−yαφ)；令 w_i=exp(−yF_{m−1})，J=Σw_i·exp(−yαφ)；w_i 不含 α 與 φ，它就是樣本權重，所以 φ_m 要最小化加權錯誤率 P_m。",
                                              "Where the sample weights come from: stage-wise optimisation optimises only the newly added classifier, F_m=F_{m−1}+αφ. Split the exponential loss (small penalty when right, heavy when wrong): exp(−yF_m)=exp(−yF_{m−1})·exp(−yαφ); let w_i=exp(−yF_{m−1}), J=Σw_i·exp(−yαφ); w_i contains neither α nor φ: it is the sample weight, so φ_m must minimise the weighted error P_m.")))
    f.rect(0, 0, 570, 78, fill=tint("blue", 0.12), stroke=tint("blue", 0.55), sw=1.5, radius=12)
    f.text(14, 39, T(tr("逐級最佳化：每次只最佳化新加入的分類器，\n先前的凍結：$F_m=F_{m-1}+\\alpha_m\\phi_m$。", "stage-wise optimisation: optimise only the newly added\nclassifier, the earlier ones frozen: $F_m=F_{m-1}+\\alpha_m\\phi_m$.")), "TXT", 17, False, "left", w=545, math=True, name=T(tr("term:逐級最佳化", "term:Stage-wise optimization")))
    f.rect(590, 0, 570, 78, fill=tint("green", 0.12), stroke=tint("green", 0.55), sw=1.5, radius=12)
    f.text(604, 39, T(tr("指數損失 $e^{-z}$，$z=y\\,F(x)$：分對（$z>0$）罰小，\n分錯（$z<0$）罰重。", "exponential loss $e^{-z}$, $z=y\\,F(x)$: a small penalty when\nright ($z>0$), a heavy one when wrong ($z<0$).")), "TXT", 17, False, "left", w=545, math=True)
    steps = [("blue", tr("① 目標", "① Goal"), "$J=\\sum_i\\exp(-y_iF_m(x_i))$"),
             ("green", tr("② 拆開", "② Split"), "$\\exp(-y_iF_{m-1})\\cdot\\exp(-y_i\\alpha\\phi)$"),
             ("orange", tr("③ 命名", "③ Name"), "$w_i=\\exp(-y_iF_{m-1}(x_i))$"),
             ("red", tr("④ 結果", "④ Result"), "$J=\\sum_iw_i\\exp(-y_i\\alpha\\phi)$")]
    w = (1160 - 3 * 36) / 4
    for k, (role, hd, tex) in enumerate(steps):
        x = k * (w + 36)
        header_chip(f, x, 100, w, 34, role, T(hd), size=18); panel(f, x, 138, w, 96, role)
        f.text(x + w / 2, 186, tex, "TXT", 16, w=w - 12, math=True)
        if k < 3: f.text(x + w + 18, 186, "→", "TXT", 26, True, w=36)
    f.rect(0, 258, 1160, 64, fill=tint("yellow", 0.22), radius=12)
    f.text(580, 290, T(tr("$w_i$ 不含 $\\alpha$ 也不含 $\\phi$：它就是樣本權重（越大，這個點越難、越要被照顧）。", "$w_i$ contains neither $\\alpha$ nor $\\phi$: it is the sample weight (the larger, the harder the point and the more it needs care).")), "TXT", 19, True, w=1120, math=True)
    chip(f, 280, 342, 600, 40, "green", T(tr("於是 $\\phi_m$ 要最小化加權錯誤率 $P_m$（權重正規化為總和 1）", "so $\\phi_m$ must minimise the weighted error $P_m$")), size=18, bold=True, dark=True, math=True)
    takeaway(f, 0, 396, 1160, 40, T(tr("樣本權重不是額外加的：它是指數損失逐級拆開自然長出來的。", "The sample weights are not added by hand: they grow out of splitting the exponential loss stage by stage.")))
    return f


def alpha_curve():
    Ps = [0.02 + i * 0.01 for i in range(97)]; pts = [0.1, 0.3, 0.5, 0.7]
    spec = {
        "id": "lec05-kp25-alpha",
        "alt": tr("話語權 α：分對的權重和 1−P 乘 e^{−α}，分錯的 P 乘 e^{α}：J(α)=e^{−α}(1−P)+e^{α}P，微分為零得 α=½ln((1−P)/P)。P=0.1、0.3、0.5、0.7 時 α=" + "、".join(f"{alpha(p):.3f}" for p in pts) + "：很準的話語權大，亂猜為 0，比亂猜還差（P>0.5）為負，把答案反過來用。",
                  "The voice α: the weights of the right points sum to 1−P and are multiplied by e^{−α}, the wrong ones sum to P and are multiplied by e^{α}: J(α)=e^{−α}(1−P)+e^{α}P; setting the derivative to zero gives α=½ln((1−P)/P). For P=0.1, 0.3, 0.5, 0.7, α=" + ", ".join(f"{alpha(p):.3f}" for p in pts) + ": a very accurate one gets a large voice, guessing gets 0, and worse than guessing (P>0.5) is negative, using the answer reversed."),
        "plot": {"x": [0, 1], "y": [-1.6, 2.2], "x_ticks": [0, 0.25, 0.5, 0.75, 1], "y_ticks": [-1, 0, 1, 2], "x_label": tr("加權錯誤率 P", "weighted error P"), "y_label": tr("話語權 α", "voice α"), "w": 430, "h": 298},
        "series": [{"kind": "hline", "y": 0, "role": "grey", "width": 2}, {"kind": "line", "points": [(p, alpha(p)) for p in Ps], "role": "green", "width": 4},
                   {"kind": "dots", "points": [(p, alpha(p)) for p in pts], "role": "orange", "d": 15}],
        "notes": [{"x": 0.1, "y": alpha(0.1) + 0.28, "text": f"{alpha(0.1):.3f}", "ha": "left"}, {"x": 0.3, "y": alpha(0.3) + 0.28, "text": f"{alpha(0.3):.3f}"}, {"x": 0.5, "y": 0.28, "text": "0"}, {"x": 0.74, "y": alpha(0.7) - 0.28, "text": f"{alpha(0.7):.3f}"}],
        "side": [{"title": tr("推導", "Derivation"), "role": "blue", "h": 1.3, "text": tr("分對的權重和 $1-P$ 乘 $e^{-\\alpha}$，\n分錯的 $P$ 乘 $e^{\\alpha}$：\n$J(\\alpha)=e^{-\\alpha}(1-P)+e^{\\alpha}P$。", "right weights sum to $1-P$, times $e^{-\\alpha}$;\nwrong ones $P$, times $e^{\\alpha}$:\n$J(\\alpha)=e^{-\\alpha}(1-P)+e^{\\alpha}P$.")},
                 {"title": tr("微分為零", "Set the derivative to 0"), "role": "green", "h": 1.2, "text": tr("$-e^{-\\alpha}(1-P)+e^{\\alpha}P=0$，\n$\\alpha=\\frac{1}{2}\\ln\\frac{1-P}{P}$。", "$-e^{-\\alpha}(1-P)+e^{\\alpha}P=0$,\n$\\alpha=\\frac{1}{2}\\ln\\frac{1-P}{P}$.")},
                 {"title": tr("怎麼讀", "How to read"), "role": "orange", "h": 1.35, "text": tr("很準 → 話語權大；亂猜 → 0；\n比亂猜差（$P>0.5$）→ 負，\n把答案反過來用。", "accurate → large voice; guessing → 0;\nworse than guessing ($P>0.5$) → negative,\nthe answer is used reversed.")}],
        "takeaway": tr("話語權由加權錯誤率決定：錯得越少，投票的分量越大。", "The voice is set by the weighted error: the fewer the errors, the heavier the vote."),
    }
    return "plot_xy", spec


def update():
    P = 0.3; a = alpha(P); wm = 0.1 * exp(a); wc = 0.1 * exp(-a); Z = 3 * wm + 7 * wc; nm, nc = wm / Z, wc / Z
    f = NFig("lec05-kp25-update", 1160, 450, T(tr(f"一輪權重更新：10 個點各 0.1，φ 錯 3 個：P=0.3，α=½ln(0.7/0.3)={a:.4f}。錯的乘 e^α={exp(a):.4f}：0.1→{wm:.4f}；對的乘 e^(−α)={exp(-a):.4f}：0.1→{wc:.4f}；Z=3×{wm:.4f}+7×{wc:.4f}={Z:.4f}；正規化後錯的各 {nm:.4f}、對的各 {nc:.4f}；錯的三個合起來 {3*nm:.1f}，下一輪的 φ 必須先照顧它們。",
                                              f"One round of weight update: 10 points of 0.1 each, φ wrong on 3: P=0.3, α=½ln(0.7/0.3)={a:.4f}. Wrong ones are multiplied by e^α={exp(a):.4f}: 0.1→{wm:.4f}; right ones by e^(−α)={exp(-a):.4f}: 0.1→{wc:.4f}; Z=3×{wm:.4f}+7×{wc:.4f}={Z:.4f}; after normalising each wrong one is {nm:.4f} and each right one {nc:.4f}; the three wrong ones together are {3*nm:.1f}, so the next φ must look after them first.")))
    header_chip(f, 0, 0, 640, 34, "blue", T(tr("10 個樣本的權重：更新前（灰）與更新後（彩色）", "weights of the 10 samples: before (grey) and after (coloured)")), size=17); panel(f, 0, 38, 640, 300, "blue")
    base = 300; sc = 1200
    f.line(20, base, 620, base, "MUT", 2)
    for i in range(10):
        x = 40 + i * 58; wrong = i < 3
        f.rect(x, base - 0.1 * sc, 22, 0.1 * sc, fill="#9d9488", radius=3)
        v = nm if wrong else nc
        f.rect(x + 24, base - v * sc, 22, v * sc, fill=solid("red" if wrong else "green"), radius=3)
        f.text(x + 23, base + 16, str(i + 1), "MUT", 16, w=40)
    f.line(20, base - 0.1 * sc, 620, base - 0.1 * sc, "MUT", 1)
    f.text(610, base - 0.1 * sc - 12, "0.1", "MUT", 16, False, "right", w=60)
    f.text(100, 78, T(tr(f"錯的 3 個：{nm:.4f}", f"the 3 wrong: {nm:.4f}")), "TXT", 17, True, "left", w=220)
    f.text(330, 130, T(tr(f"對的 7 個：{nc:.4f}", f"the 7 right: {nc:.4f}")), "TXT", 17, True, "left", w=240)
    steps = [("blue", tr("① 加權錯誤率與話語權", "① weighted error and voice"), f"$P=0.3,\\ \\alpha=\\frac{{1}}{{2}}\\ln\\frac{{0.7}}{{0.3}}={a:.4f}$"),
             ("orange", tr("② 錯的放大、對的縮小", "② wrong up, right down"), f"$e^{{\\alpha}}={exp(a):.4f}:\\ 0.1\\to{wm:.4f}$\n$e^{{-\\alpha}}={exp(-a):.4f}:\\ 0.1\\to{wc:.4f}$"),
             ("green", tr("③ 正規化", "③ normalise"), f"$Z=3\\times{wm:.4f}+7\\times{wc:.4f}={Z:.4f}$")]
    for k, (role, hd, tex) in enumerate(steps):
        y = k * 108
        header_chip(f, 670, y, 490, 30, role, T(hd), size=16); panel(f, 670, y + 34, 490, 66, role)
        f.text(915, y + 68, tex, "TXT", 17, w=470, math=True)
    chip(f, 670, 330, 490, 44, "yellow", T(tr(f"錯的 3 個合起來 3×{nm:.4f}={3*nm:.1f}", f"the 3 wrong ones together: 3×{nm:.4f}={3*nm:.1f}")), size=19, bold=True, dark=True)
    takeaway(f, 0, 392, 1160, 44, T(tr("做錯的點變大、做對的變小；下一輪的分類器必須先照顧那三個錯點。", "Wrong points grow and right ones shrink; the next classifier must look after those three first.")))
    return f


def _curve(vals, role, name):
    pts = [(log10(r), vals[r - 1] * 100) for r in LOGR]
    return {"kind": "line", "points": pts, "role": role, "width": 4, "name": name}


def curve():
    tr_, te = C["train"], C["test"]; z = next(i + 1 for i, v in enumerate(tr_) if v < 1e-6)
    spec = {
        "id": "lec05-kp25-curve",
        "alt": tr(f"AdaBoost 的訓練與測試錯誤對輪數（對數軸，20 維兩類高斯、訓練 200、測試 400 點）：第 1 輪訓練 {tr_[0]*100:.1f}%、測試 {te[0]*100:.2f}%；第 {z} 輪起訓練錯誤為 0；測試錯誤第 10、50、200、2000 輪為 {te[9]*100:.2f}%、{te[49]*100:.2f}%、{te[199]*100:.2f}%、{te[1999]*100:.2f}%，約 4% 持平；400 個測試點上 1% 只是 4 點。",
                  f"AdaBoost training and test error against rounds (log axis; 20-dimensional two-class Gaussians, 200 training and 400 test points): round 1 has training {tr_[0]*100:.1f}% and test {te[0]*100:.2f}%; from round {z} the training error is 0; the test error at rounds 10, 50, 200, 2000 is {te[9]*100:.2f}%, {te[49]*100:.2f}%, {te[199]*100:.2f}%, {te[1999]*100:.2f}%, levelling near 4%; with 400 test points 1% is only 4 points."),
        "plot": {"x": [-0.1, 3.4], "y": [0, 40], "x_ticks": [0, 1, 2, 3], "x_names": XT, "y_ticks": [0, 10, 20, 30, 40], "x_label": tr("輪數（對數軸）", "rounds (log axis)"), "y_label": tr("錯誤（%）", "error (%)"), "w": 430, "h": 300},
        "series": [_curve(tr_, "blue", tr("訓練錯誤", "training error")), _curve(te, "red", tr("測試錯誤", "test error")),
                   {"kind": "vline", "x": log10(z), "role": "green", "width": 2, "label": tr(f"第 {z} 輪", f"round {z}")}],
        "side": [{"title": tr("AdaBoost 三步", "AdaBoost in three steps"), "role": "blue", "h": 1.7, "text": tr("1. 起始 $w_i=1/N$。\n2. 重複：找最小化 $P_m$ 的 $\\phi_m$，\n算 $\\alpha_m$，更新 $w_i$。\n3. 最後 $f(x)=\\mathrm{sign}\\sum_k\\alpha_k\\phi_k(x)$。", "1. start with $w_i=1/N$.\n2. repeat: find $\\phi_m$ minimising $P_m$,\ncompute $\\alpha_m$, update $w_i$.\n3. finally $f(x)=\\mathrm{sign}\\sum_k\\alpha_k\\phi_k(x)$.")},
                 {"title": tr("第 1 輪", "Round 1"), "role": "green", "h": 1.0, "text": tr(f"訓練 {tr_[0]*100:.1f}%、測試 {te[0]*100:.2f}%，\n比隨便一個決策樁（平均 31.6%）好。", f"training {tr_[0]*100:.1f}%, test {te[0]*100:.2f}%,\nbetter than a random stump (mean 31.6%).")},
                 {"title": tr(f"第 {z} 輪之後", f"After round {z}"), "role": "orange", "h": 1.35, "text": tr(f"訓練錯誤歸零，測試錯誤繼續降：\n{te[9]*100:.2f}% → {te[49]*100:.2f}% → {te[199]*100:.2f}%（200 輪），\n約 4% 持平；400 點上 1% 只是 4 點。", f"training error is 0, test error keeps falling:\n{te[9]*100:.2f}% → {te[49]*100:.2f}% → {te[199]*100:.2f}% (200 rounds),\nflat near 4%; with 400 points 1% is 4 points.")}],
        "takeaway": tr("訓練錯誤歸零之後，測試錯誤還繼續降，約 200 輪後持平在約 4%。", "After the training error reaches 0 the test error keeps falling and levels near 4% after about 200 rounds."),
    }
    return "plot_xy", spec


def margin():
    mm = C["minm"]; marks = [1, 10, 50, 100, 200, 2000]
    spec = {
        "id": "lec05-kp25-margin",
        "alt": tr("最小正規化邊距隨輪數升高：第 1、10、50、100、200、2000 輪為 " + "、".join(f"{mm[r-1]:+.3f}" for r in marks) + "；邊距小於 0.1 的比例由 25.5% 降到 0%。正規化邊距 yF(x)/Σα 在 [−1,1]，正的才分對，越大信心越高。",
                  "The minimum normalised margin rises with rounds: at rounds 1, 10, 50, 100, 200, 2000 it is " + ", ".join(f"{mm[r-1]:+.3f}" for r in marks) + "; the share of margins below 0.1 falls from 25.5% to 0%. The normalised margin yF(x)/Σα lies in [−1,1]: positive means right, larger means more confident."),
        "plot": {"x": [-0.1, 3.4], "y": [-1.1, 0.4], "x_ticks": [0, 1, 2, 3], "x_names": XT, "y_ticks": [-1, -0.5, 0, 0.25], "x_label": tr("輪數（對數軸）", "rounds (log axis)"), "y_label": tr("最小邊距", "minimum margin"), "w": 430, "h": 300},
        "series": [{"kind": "hline", "y": 0, "role": "grey", "width": 2}, {"kind": "line", "points": [(log10(r), mm[r - 1]) for r in LOGR], "role": "green", "width": 4},
                   {"kind": "dots", "points": [(log10(r), mm[r - 1]) for r in marks], "role": "orange", "d": 14}],
        "notes": [{"x": log10(1) + 0.2, "y": mm[0] + 0.1, "text": f"{mm[0]:+.3f}", "ha": "left"}, {"x": log10(10), "y": mm[9] - 0.12, "text": f"{mm[9]:+.3f}"}, {"x": log10(50), "y": mm[49] - 0.14, "text": f"{mm[49]:+.3f}"}, {"x": log10(2000), "y": mm[1999] + 0.12, "text": f"{mm[1999]:+.3f}"}],
        "side": [{"title": tr("正規化邊距", "Normalised margin"), "role": "blue", "name": tr("term:正規化邊距", "term:Normalized margin"), "h": 1.15, "text": tr("$yF(x)/\\sum_k\\alpha_k\\in[-1,1]$；\n正的才分對，越大信心越高。", "$yF(x)/\\sum_k\\alpha_k\\in[-1,1]$;\npositive means right, larger more confident.")},
                 {"title": tr("為什麼測試不上升", "Why the test error does not rise"), "role": "green", "h": 1.25, "text": tr(f"最小邊距被不斷抬高：{mm[0]:+.3f} → {mm[49]:+.3f} → {mm[1999]:+.3f}；\n邊距小於 0.1 的比例 25.5% → 0%。\n上界與輪數 $K$ 無關。", f"the minimum margin keeps rising: {mm[0]:+.3f} → {mm[49]:+.3f} → {mm[1999]:+.3f};\nshare of margins below 0.1: 25.5% → 0%.\nthe bound does not depend on the rounds $K$.")},
                 {"title": tr("限制", "Limit"), "role": "orange", "h": 1.0, "text": tr("這個上界很鬆（除非 $N$ 到數萬），\n只能當定性解釋。", "the bound is very loose (unless $N$ is in the\ntens of thousands): a qualitative explanation only.")}],
        "takeaway": tr("最小邊距不斷升高，是「訓練錯誤歸零後測試錯誤仍不上升」的定性解釋。", "The rising minimum margin is a qualitative reason why the test error does not rise after the training error reaches 0."),
    }
    return "plot_xy", spec


def noise():
    cl = C["noise"]["0.0"]; fl = C["noise"]["0.15"]; mn = min(range(2000), key=lambda i: fl[i]) + 1
    spec = {
        "id": "lec05-kp25-noise",
        "alt": tr(f"標籤乾淨與訓練標籤翻 15% 的測試錯誤（各 5 次平均，測試標籤乾淨）：乾淨標籤一路降到 {cl[1999]*100:.1f}%；翻 15% 時約 30 輪最低（第 10、30、100、500、2000 輪：{fl[9]*100:.1f}%、{fl[29]*100:.1f}%、{fl[99]*100:.1f}%、{fl[499]*100:.1f}%、{fl[1999]*100:.1f}%），之後上升到 {fl[1999]*100:.1f}%；2000 輪後 15.5% 的翻標籤樣本占 36% 的權重（其中一次實驗）。",
                  f"Test error with clean labels and with training labels flipped at 15% (means of 5 runs, test labels clean): with clean labels it falls steadily to {cl[1999]*100:.1f}%; with 15% flipped it is lowest near round 30 (rounds 10, 30, 100, 500, 2000: {fl[9]*100:.1f}%, {fl[29]*100:.1f}%, {fl[99]*100:.1f}%, {fl[499]*100:.1f}%, {fl[1999]*100:.1f}%) and then rises to {fl[1999]*100:.1f}%; after 2000 rounds the 15.5% flipped samples hold 36% of the weight (one run)."),
        "plot": {"x": [-0.1, 3.4], "y": [0, 40], "x_ticks": [0, 1, 2, 3], "x_names": XT, "y_ticks": [0, 10, 20, 30, 40], "x_label": tr("輪數（對數軸）", "rounds (log axis)"), "y_label": tr("測試錯誤（%）", "test error (%)"), "w": 430, "h": 300, "legend_pos": "left"},
        "series": [_curve(cl, "green", tr("標籤乾淨", "clean labels")), _curve(fl, "red", tr("翻 15%", "15% flipped")),
                   {"kind": "dots", "points": [(log10(30), fl[29] * 100), (log10(2000), fl[1999] * 100), (log10(2000), cl[1999] * 100)], "role": "orange", "d": 13}],
        "notes": [{"x": log10(30), "y": fl[29] * 100 - 3.2, "text": f"{fl[29]*100:.1f}", "ha": "center"}, {"x": log10(2000), "y": fl[1999] * 100 + 2.6, "text": f"{fl[1999]*100:.1f}"}, {"x": log10(2000), "y": cl[1999] * 100 + 2.6, "text": f"{cl[1999]*100:.1f}"}],
        "side": [{"title": tr("設定", "Setup"), "role": "blue", "h": 1.0, "text": tr("訓練標籤隨機翻 15%，\n測試標籤乾淨；各做 5 次取平均。", "training labels flipped at random (15%),\ntest labels clean; mean of 5 runs.")},
                 {"title": tr("可能原因", "Possible cause"), "role": "orange", "h": 1.2, "text": tr("指數損失對大負邊距的壞樣本懲罰過重；\n2000 輪後 15.5% 的翻標籤樣本\n占了 36% 的權重（其中一次實驗）。", "the exponential loss punishes badly\nplaced samples too heavily; after 2000 rounds the\n15.5% flipped samples hold 36% of the weight (one run).")},
                 {"title": tr("常見迷思", "Common myth"), "role": "red", "h": 1.15, "text": tr("「AdaBoost 永遠不會過擬合」。\n在貝氏誤差高、標籤有雜訊的困難任務上\n它可能急劇退化（來源）。", "\"AdaBoost never overfits\". On hard tasks with a high\nBayes error and noisy labels it can\ndegrade sharply (source).")}],
        "takeaway": tr(f"標籤乾淨時一路降到 {cl[1999]*100:.1f}%；翻 15% 時約 30 輪最低（{fl[29]*100:.1f}%），之後上升到 {fl[1999]*100:.1f}%。", f"With clean labels it falls to {cl[1999]*100:.1f}%; with 15% flipped it is lowest near round 30 ({fl[29]*100:.1f}%) and then rises to {fl[1999]*100:.1f}%."),
    }
    return "plot_xy", spec


def negative():
    P = 0.7; a = alpha(P)
    spec = {
        "id": "lec05-kp25-negative",
        "alt": tr(f"負的 α：P=0.7：α=½ln(0.3/0.7)={a:.4f}；{a:.4f}φ=+{-a:.4f}(−φ)，而 −φ 的加權錯誤率是 0.3；φ 做錯的點乘 e^({a:.4f})={exp(a):.4f}（變小），做對的乘 {exp(-a):.4f}（變大），與反過來的分類器完全一樣。",
                  f"Negative α: P=0.7: α=½ln(0.3/0.7)={a:.4f}; {a:.4f}φ=+{-a:.4f}(−φ), and −φ has weighted error 0.3; points φ gets wrong are multiplied by e^({a:.4f})={exp(a):.4f} (smaller), those it gets right by {exp(-a):.4f} (larger), exactly as for the reversed classifier."),
        "steps": [{"role": "blue", "title": tr("① 話語權", "① The voice"), "lines": ["$P=0.7$", f"$\\alpha=\\frac{{1}}{{2}}\\ln\\frac{{0.3}}{{0.7}}={a:.4f}$"], "result": tr("α 是負的", "α is negative")},
                  {"role": "green", "title": tr("② 等於反過來用", "② Same as reversing"), "lines": [f"${a:.4f}\\,\\phi=+{-a:.4f}\\,(-\\phi)$", tr("而 $-\\phi$ 的加權錯誤率是 0.3", "and $-\\phi$ has weighted error 0.3")], "result": tr("$-\\phi$ 是好分類器", "$-\\phi$ is a good one")},
                  {"role": "orange", "title": tr("③ 權重更新", "③ Weight update"), "lines": [tr(f"$\\phi$ 錯的點乘 {exp(a):.4f}（變小）", f"points $\\phi$ gets wrong: $\\times{exp(a):.4f}$ (down)"), tr(f"做對的乘 {exp(-a):.4f}（變大）", f"points it gets right: $\\times{exp(-a):.4f}$ (up)")], "result": tr("與反過來的分類器一樣", "same as the reversed one")}],
        "takeaway": tr("比亂猜還差的分類器，負的 α 等於把它的答案反過來用。", "For a classifier worse than guessing, a negative α simply uses its answer reversed."),
    }
    return "steps_flow", spec


def solution():
    P = 0.2; a = alpha(P); e1, e2 = exp(a), exp(-a); wm = 0.2 * e1; wc = 0.2 * e2; Z = wm + 4 * wc
    spec = {
        "id": "lec05-kp25-solution",
        "alt": tr(f"解答：題 1：α=½ln(0.8/0.2)=½ln4={a:.4f}，e^α={e1:.0f}、e^(−α)={e2:g}。題 2：錯的 0.2×2={wm:.1f}，對的 4 個各 0.2×0.5={wc:.1f}；Z={wm:.1f}+4×{wc:.1f}={Z:.1f}；正規化後錯的 {wm/Z:.1f}、對的各 {wc/Z:.3f}，總和 {wm/Z+4*wc/Z:g}；更新後被舊 φ 分錯的權重總和恆為 0.5。",
                  f"Solution: Q1: α=½ln(0.8/0.2)=½ln4={a:.4f}, e^α={e1:.0f}, e^(−α)={e2:g}. Q2: the wrong one 0.2×2={wm:.1f}, the 4 right ones 0.2×0.5={wc:.1f} each; Z={wm:.1f}+4×{wc:.1f}={Z:.1f}; normalised the wrong one is {wm/Z:.1f} and each right one {wc/Z:.3f}, summing to {wm/Z+4*wc/Z:g}; after the update the weights the old φ gets wrong always sum to 0.5."),
        "steps": [{"role": "blue", "title": tr("① 題 1：α", "① Q1: α"), "lines": ["$\\alpha=\\frac{1}{2}\\ln\\frac{0.8}{0.2}=\\frac{1}{2}\\ln4$", f"$e^{{\\alpha}}=2,\\ e^{{-\\alpha}}=0.5$"], "result": f"$\\alpha={a:.4f}$"},
                  {"role": "green", "title": tr("② 題 2：更新", "② Q2: update"), "lines": [f"錯的 $0.2\\times2={wm:.1f}$" if False else tr(f"錯的 $0.2\\times2={wm:.1f}$", f"the wrong one $0.2\\times2={wm:.1f}$"), tr(f"對的 4 個各 $0.2\\times0.5={wc:.1f}$", f"4 right ones $0.2\\times0.5={wc:.1f}$ each"), f"$Z={wm:.1f}+4\\times{wc:.1f}={Z:.1f}$"], "result": tr("Z = 0.8", "Z = 0.8")},
                  {"role": "orange", "title": tr("③ 正規化", "③ Normalise"), "lines": [tr(f"錯的 ${wm:.1f}/{Z:.1f}={wm/Z:.1f}$", f"wrong: ${wm:.1f}/{Z:.1f}={wm/Z:.1f}$"), tr(f"對的各 ${wc:.1f}/{Z:.1f}={wc/Z:.3f}$", f"each right: ${wc:.1f}/{Z:.1f}={wc/Z:.3f}$"), f"$0.5+4\\times{wc/Z:.3f}=1$"], "result": tr("錯的剛好占一半", "the wrong one holds half")}],
        "takeaway": tr("更新後，被舊 φ 分錯的權重總和恆為 0.5，所以必須換新的 φ。", "After the update the weights the old φ gets wrong always sum to 0.5, so a new φ is needed."),
    }
    return "steps_flow", spec
