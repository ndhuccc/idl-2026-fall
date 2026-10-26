"""KP26 (class imbalance) -- seventh sample of view-from-text-v2. Plan: kp26_plan.json. Every average count comes from kp26_cache.json
(built by kp26_calc.py with kp26_nums.py; reproduces the lecture's 136 numbers exactly); metrics are derived here from the unrounded counts."""
import os, sys, json
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from figkit.parts import T, NFig, header_chip, panel, chip, outline_box, takeaway, tint, solid, INK
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)


def tr(zh, en): return {"zh": zh, "en": en}


def h1(x):   # one decimal, half up, of a percentage
    return str(Decimal(repr(round(x, 9))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


C = json.load(open(os.path.join(HERE, "kp26_cache.json")))


def M(c, nmin=100, nmaj=5000):
    tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
    rec = tp / (tp + fn); spec = tn / (tn + fp)
    return dict(tp=tp, fp=fp, rec=rec * 100, prec=(tp / (tp + fp) * 100 if tp + fp > 0 else None), acc=(tp + tn) / (tp + fn + fp + tn) * 100, spec=spec, bal=(rec + spec) / 2)


def setup():
    from kp26_nums import make
    X, y = make(1000, 20, 2.0, 0)
    pa = [(float(a), float(b)) for (a, b), c in zip(X, y) if c == 0]; pb = [(float(a), float(b)) for (a, b), c in zip(X, y) if c == 1]
    spec = {
        "id": "lec05-kp26-setup",
        "alt": tr("一組訓練資料：1000 個多數點取自以 (0,0) 為中心的標準常態，20 個稀有點取自以 (2,0) 為中心的標準常態，兩團明顯重疊，所以不可能完全分開。訓練 1000＋20 個點，測試固定 5000＋100 個點（稀有點占 2%）；RBF 核 SVM，C=1；訓練集用亂數種子 0 到 19 各抽一次，共 20 次取平均。",
                  "One training set: 1000 majority points drawn from a standard normal centred at (0,0) and 20 rare points from a standard normal centred at (2,0); the two clouds clearly overlap, so they cannot be separated completely. Training 1000+20 points, test fixed at 5000+100 points (2% rare); RBF SVM with C=1; the training set is drawn with seeds 0 to 19, 20 times, and averaged."),
        "plot": {"x": [-4, 6], "y": [-4, 4], "x_ticks": [-4, -2, 0, 2, 4, 6], "y_ticks": [-4, -2, 0, 2, 4], "x_label": "x₁", "y_label": "x₂", "w": 420, "h": 300, "legend_pos": "left"},
        "series": [{"kind": "dots", "points": pa, "role": "blue", "d": 6, "name": tr("多數類（1000 點）", "majority (1000 points)")},
                   {"kind": "dots", "points": pb, "role": "orange", "d": 12, "shape": "square", "name": tr("稀有類（20 點）", "rare class (20 points)")}],
        "side": [{"title": tr("類別不平衡", "Class imbalance"), "role": "orange", "name": tr("term:類別不平衡", "term:Class imbalance"), "h": 1.1, "text": tr("某一類樣本極多、另一類極少；\n常見於文字分類、罕見疾病、浮油偵測。", "one class has very many samples and\nanother very few; common in text\nclassification, rare diseases, oil-slick detection.")},
                 {"title": tr("資料", "Data"), "role": "blue", "h": 1.0, "text": tr("訓練 1000＋20 點；測試 5000＋100 點，\n稀有點只占 2%。兩類標準差都是 1。", "training 1000+20, test 5000+100: only 2%\nof the test points are rare; both classes\nhave standard deviation 1.")},
                 {"title": tr("模型與重複", "Model and repeats"), "role": "green", "h": 1.0, "text": tr("RBF 核 SVM，$C=1$；\n訓練集抽 20 次（種子 0 到 19）取平均。", "RBF-kernel SVM, $C=1$;\n20 training sets (seeds 0 to 19), averaged.")}],
        "takeaway": tr("兩團明顯重疊，不可能完全分開；稀有點只占測試的 2%。", "The two clouds overlap clearly, so they cannot be separated completely; rare points are only 2% of the test."),
    }
    return "plot_xy", spec


def metrics():
    s = M(C["plain_svm"]); allm = dict(rec=0.0, prec=None, acc=5000 / 5100 * 100)
    f = NFig("lec05-kp26-metrics", 1160, 460, T(tr(f"測試 100 個稀有點、5000 個多數點。SVM（20 次平均）：TP {s['tp']:g}、FN {C['plain_svm']['fn']:g}、FP {s['fp']:g}、TN {C['plain_svm']['tn']:g}。召回 {h1(s['rec'])}%、精確率 {h1(s['prec'])}%、準確率 {h1(s['acc'])}%；全判多數：召回 0、準確率 {h1(allm['acc'])}%。準確率從 {h1(allm['acc'])}% 到 {h1(s['acc'])}% 幾乎沒變，召回卻從 0 到 {h1(s['rec'])}%：準確率看不見稀有類。",
                                              f"Test: 100 rare points and 5000 majority points. SVM (mean of 20 runs): TP {s['tp']:g}, FN {C['plain_svm']['fn']:g}, FP {s['fp']:g}, TN {C['plain_svm']['tn']:g}. Recall {h1(s['rec'])}%, precision {h1(s['prec'])}%, accuracy {h1(s['acc'])}%; the all-majority classifier has recall 0 and accuracy {h1(allm['acc'])}%. Accuracy hardly moves from {h1(allm['acc'])}% to {h1(s['acc'])}% while recall goes from 0 to {h1(s['rec'])}%: accuracy cannot see the rare class.")))
    header_chip(f, 0, 0, 520, 34, "blue", T(tr("SVM 的四個計數（20 次平均）", "SVM's four counts (mean of 20 runs)")), size=17); panel(f, 0, 38, 520, 300, "blue")
    cells = [("TP", s["tp"], "green", 0, 0, tr("抓到的稀有點", "caught")), ("FN", C["plain_svm"]["fn"], "red", 1, 0, tr("漏掉的稀有點", "missed")),
             ("FP", s["fp"], "orange", 0, 1, tr("誤報數", "false alarms")), ("TN", C["plain_svm"]["tn"], "grey", 1, 1, tr("判對的多數點", "majority right"))]
    f.text(190, 66, T(tr("判成稀有", "called rare")), "MUT", 16, True, w=140); f.text(350, 66, T(tr("判成多數", "called majority")), "MUT", 16, True, w=150)
    f.text(62, 150, T(tr("實際\n稀有", "truly\nrare")), "MUT", 16, True, w=100); f.text(62, 260, T(tr("實際\n多數", "truly\nmajority")), "MUT", 16, True, w=100)
    for nm, v, role, cx, cy, cap in cells:
        x = 120 + cx * 160; y = 84 + cy * 118
        f.rect(x, y, 150, 108, fill=tint(role, 0.28), stroke=solid(role), sw=2, radius=10)
        f.text(x + 75, y + 26, nm, "TXT", 20, True, w=120); f.text(x + 75, y + 62, f"{v:g}", "TXT", 26, True, w=140); f.text(x + 75, y + 90, T(cap), "MUT", 16, w=146)
    defs = [("term:召回率", "term:Recall", tr("召回率：稀有點被抓到的比例", "recall: share of rare points caught"), "$\\mathrm{TP}/(\\mathrm{TP}+\\mathrm{FN})$"),
            ("term:精確率", "term:Precision", tr("精確率：判為稀有的點裡真的稀有的比例", "precision: share of called-rare points truly rare"), "$\\mathrm{TP}/(\\mathrm{TP}+\\mathrm{FP})$"),
            ("", "", tr("準確率：全部判對的比例", "accuracy: share of all points right"), "$(\\mathrm{TP}+\\mathrm{TN})/5100$")]
    for i, (zn, en_, tx, fm) in enumerate(defs):
        y = 38 + i * 64
        f.text(560, y + 10, T(tx), "TXT", 16, True, "left", w=600, name=T(tr(zn, en_)) if zn else None)
        f.text(560, y + 40, fm, "TXT", 18, False, "left", w=300, math=True)
    # comparison table
    header_chip(f, 560, 228, 600, 34, "grey", T(tr("兩個分類器並排", "two classifiers side by side")), size=17); panel(f, 560, 266, 600, 72, "grey")
    cols = [(tr("全判多數", "all-majority"), allm), ("SVM", dict(rec=s["rec"], prec=s["prec"], acc=s["acc"]))]
    for k, (nm, m) in enumerate(cols):
        x = 580 + k * 290
        f.text(x, 286, T(nm), "TXT", 17, True, "left", w=124)
        f.text(x + 130, 286, T(tr(f"召回 {h1(m['rec'])}%", f"recall {h1(m['rec'])}%")), "TXT", 17, False, "left", w=150)
        f.text(x, 316, T(tr(f"準確率 {h1(m['acc'])}%", f"accuracy {h1(m['acc'])}%")), "TXT", 17, True, "left", w=170)
        if m["prec"] is not None: f.text(x + 170, 316, T(tr(f"精確率 {h1(m['prec'])}%", f"precision {h1(m['prec'])}%")), "TXT", 16, False, "left", w=140)
    takeaway(f, 0, 360, 1160, 50, T(tr(f"準確率從 {h1(allm['acc'])}% 到 {h1(s['acc'])}% 幾乎沒變，召回卻從 0 到 {h1(s['rec'])}%：準確率看不見稀有類。", f"Accuracy barely moves, {h1(allm['acc'])}% to {h1(s['acc'])}%, while recall goes from 0 to {h1(s['rec'])}%: accuracy cannot see the rare class.")))
    return f


def simplicity():
    Cs = [("0.01", C["C"]["0.01"]), ("0.1", C["C"]["0.1"]), ("1", C["C"]["1"]), ("10", C["C"]["10"]), ("100", C["C"]["100"])]
    Ls = [("50", C["leaf"]["50"]), ("20", C["leaf"]["20"]), ("5", C["leaf"]["5"]), ("1", C["leaf"]["1"])]
    f = NFig("lec05-kp26-simplicity", 1160, 440, T(tr("越簡單，抓到的稀有點越少，而準確率始終約 98%。SVM：C 由 0.01 到 100，召回 " + "、".join(h1(M(c)["rec"]) + "%" for _, c in Cs) + "，準確率 " + "、".join(h1(M(c)["acc"]) + "%" for _, c in Cs) + "。決策樹（每片葉至少 k 點）：k 由 50 到 1，召回 " + "、".join(h1(M(c)["rec"]) + "%" for _, c in Ls) + "，準確率 " + "、".join(h1(M(c)["acc"]) + "%" for _, c in Ls) + "。",
                                                 "The simpler, the fewer rare points are caught, while accuracy stays near 98%. SVM: C from 0.01 to 100, recall " + ", ".join(h1(M(c)["rec"]) + "%" for _, c in Cs) + ", accuracy " + ", ".join(h1(M(c)["acc"]) + "%" for _, c in Cs) + ". Decision tree (at least k points per leaf): k from 50 to 1, recall " + ", ".join(h1(M(c)["rec"]) + "%" for _, c in Ls) + ", accuracy " + ", ".join(h1(M(c)["acc"]) + "%" for _, c in Ls) + ".")))
    for k, (hd, role, items, xlab, note) in enumerate(((tr("SVM：$C$ 越小越簡單（越不容忍分錯的罰分）", "SVM: smaller $C$ is simpler (a smaller penalty for errors)"), "blue", Cs, "C", tr("$C$", "C")),
                                                      (tr("決策樹：每片葉至少 $k$ 點，$k$ 越大越簡單", "Decision tree: at least $k$ points per leaf, larger $k$ is simpler"), "green", Ls, "k", tr("$k$", "k")))):
        x0 = k * 590; n = len(items); bw = 80; gap = (570 - n * bw) / (n + 1)
        header_chip(f, x0, 0, 570, 34, role, T(hd), size=16, ) if False else f.rect(x0, 0, 570, 34, fill=solid(role), radius=10, text=T(hd), tcolor="#1c1712", size=16, bold=True, math=True)
        panel(f, x0, 38, 570, 300, role)
        base = 230; sc = 4.6
        f.line(x0 + 20, base, x0 + 550, base, "MUT", 2)
        for i, (lab, c) in enumerate(items):
            m = M(c); x = x0 + gap * (i + 1) + bw * i
            h = max(m["rec"] * sc, 2)
            f.rect(x, base - h, bw, h, fill=solid(role), radius=4)
            f.text(x + bw / 2, base - h - 14, f"{h1(m['rec'])}%", "TXT", 17, True, w=90)
            f.text(x + bw / 2, base + 18, f"{lab}", "TXT", 18, True, w=80)
            chip(f, x - 2, 274, bw + 4, 30, "grey", f"{h1(m['acc'])}%", size=16, bold=True, alpha=0.2)
        f.text(x0 + 14, 86, T(tr("召回（抓到的稀有點比例）", "recall (share of rare points caught)")), "TXT", 16, True, "left", w=300)
        f.text(x0 + 556, 86, T(tr("下方灰框：準確率", "grey boxes: accuracy")), "MUT", 16, False, "right", w=200)
    takeaway(f, 0, 356, 1160, 50, T(tr("越簡單，TP 越少，直到全判多數；而準確率始終約 98%。簡單的假設不會花力氣照顧稀有案例（來源）。", "Simpler means fewer TP, down to all-majority; accuracy stays near 98% (source: simple hypotheses ignore rare cases).")))
    return f


def difficulty():
    rows = ["100:100", "1000:100", "1000:10", "10000:100", "10000:10"]
    f = NFig("lec05-kp26-difficulty", 1160, 440, T(tr("同一個 RBF SVM，稀有類召回（測試 2000＋2000）：簡單任務（中心差 4）100:100、1000:100、1000:10、10000:100、10000:10 為 " + "、".join(h1(M(C['grid'][f'easy-{r}'])['rec']) + "%" for r in rows) + "；困難任務（中心差 1.5）為 " + "、".join(h1(M(C['grid'][f'hard-{r}'])['rec']) + "%" for r in rows) + "。稀有點只有 10 個時簡單任務也差；困難任務上 1000 比 100 就把召回壓到 14.9%。",
                                                 "The same RBF SVM, recall of the rare class (test 2000+2000): easy task (centres 4 apart) at 100:100, 1000:100, 1000:10, 10000:100, 10000:10 is " + ", ".join(h1(M(C['grid'][f'easy-{r}'])['rec']) + "%" for r in rows) + "; hard task (centres 1.5 apart) is " + ", ".join(h1(M(C['grid'][f'hard-{r}'])['rec']) + "%" for r in rows) + ". With only 10 rare points even the easy task is poor; on the hard task 1000 against 100 already presses the recall down to 14.9%.")))
    header_chip(f, 0, 0, 700, 34, "blue", T(tr("稀有類召回（％）：多數:稀有 對 任務難度", "recall of the rare class (%): majority:rare against task difficulty")), size=16); panel(f, 0, 38, 700, 330, "blue")
    for j, hd in enumerate((tr("簡單（差 4）", "easy (4 apart)"), tr("困難（差 1.5）", "hard (1.5 apart)"))):
        f.text(330 + j * 180, 66, T(hd), "TXT", 16, True, w=170)
    for i, r in enumerate(rows):
        y = 84 + i * 56
        f.text(150, y + 24, r, "TXT", 20, True, "right", w=140)
        for j, kind in enumerate(("easy", "hard")):
            m = M(C["grid"][f"{kind}-{r}"]); v = m["rec"]
            f.rect(246 + j * 180, y, 170, 48, fill=tint("green", 0.12 + 0.55 * v / 100), stroke=solid("green"), sw=1.5, radius=8)
            f.text(246 + j * 180 + 85, y + 16, f"{h1(v)}%", "TXT", 22, True, w=160)
            f.text(246 + j * 180 + 85, y + 38, T(tr(f"平均 TP {m['tp']:g}", f"mean TP {m['tp']:g}")), "MUT", 16, w=160)
    f.rect(740, 0, 420, 368, fill=tint("orange", 0.12), stroke=tint("orange", 0.6), sw=1.5, radius=12)
    f.text(760, 40, T(tr("大不平衡未必有害", "A large imbalance is not necessarily harmful")), "TXT", 18, True, "left", w=380)
    f.text(760, 120, T(tr("要與任務難度、訓練點數一起看（來源，Japkowicz）。\n簡單任務或資料量大時可能沒問題；\n困難（類別重疊）或點數不足時，\n小不平衡也可能致命。", "look at it together with the task difficulty and the number of training points (source, Japkowicz).\nSimple tasks or lots of data may be fine;\nhard (overlapping) tasks or too few points\ncan be fatal even at a small imbalance.")), "TXT", 16, False, "left", w=380)
    f.text(760, 280, T(tr("稀有點只有 10 個時，簡單任務也差。\n困難任務上 1000 比 100 就把召回壓到 14.9%。", "With only 10 rare points even the easy task is poor.\nOn the hard task 1000 against 100 already presses recall to 14.9%.")), "TXT", 16, True, "left", w=380)
    takeaway(f, 0, 388, 1160, 44, T(tr("類別不平衡傷害多大，要看任務難度與訓練點數，不能只看比例。", "How much imbalance hurts depends on the task difficulty and the number of points, not on the ratio alone.")))
    return f


def resample():
    plain, over, under = M(C["plain_svm"]), M(C["over_svm"]), M(C["under_svm"])
    f = NFig("lec05-kp26-resample", 1160, 460, T(tr(f"同一個 SVM（C=1）。一般（1000＋20）：召回 {h1(plain['rec'])}%、精確率 {h1(plain['prec'])}%、準確率 {h1(plain['acc'])}%；過採樣（稀有類再抽到 1000＋1000）：{h1(over['rec'])}%、{h1(over['prec'])}%、{h1(over['acc'])}%；欠採樣（多數類只留 20，20＋20）：{h1(under['rec'])}%、{h1(under['prec'])}%、{h1(under['acc'])}%。稀有類找回來了，但精確率掉到約 10%，準確率掉到 85% 到 88%。",
                                              f"The same SVM (C=1). Plain (1000+20): recall {h1(plain['rec'])}%, precision {h1(plain['prec'])}%, accuracy {h1(plain['acc'])}%; oversampling (rare class redrawn to 1000+1000): {h1(over['rec'])}%, {h1(over['prec'])}%, {h1(over['acc'])}%; undersampling (only 20 majority points kept, 20+20): {h1(under['rec'])}%, {h1(under['prec'])}%, {h1(under['acc'])}%. The rare class is found again, but precision drops to about 10% and accuracy to 85-88%.")))
    rows = [(tr("一般", "plain"), "1000＋20", plain, "grey", None, None), (tr("過採樣", "oversampling"), "1000＋1000", over, "green", ("term:過採樣", "term:Oversampling"), tr("過採樣：從 20 個稀有點有放回地再抽 980 個", "oversampling: redraw 980 more from the 20 rare points, with replacement")),
            (tr("欠採樣", "undersampling"), "20＋20", under, "orange", ("term:欠採樣", "term:Undersampling"), tr("欠採樣：從 1000 個多數點不放回地只留 20 個", "undersampling: keep only 20 of the 1000 majority points, without replacement"))]
    heads = (tr("召回", "recall"), tr("精確率", "precision"), tr("準確率", "accuracy"))
    for j, hd in enumerate(heads): f.text(720 + j * 150, 16, T(hd), "TXT", 17, True, w=130)
    sc = 1.15
    for i, (nm, comp, m, role, tname, defn) in enumerate(rows):
        y = 40 + i * 116
        panel(f, 0, y, 1160, 104, role)
        chip(f, 14, y + 10, 140, 32, role, T(nm), size=17, bold=True, alpha=0.4)
        f.text(170, y + 26, T(tr(f"訓練集 {comp}", f"training set {comp}")), "TXT", 17, True, "left", w=240)
        if tname: f.text(14, y + 74, T(defn), "TXT", 16, False, "left", w=560, name=T(tr(*tname)))
        else: f.text(14, y + 74, T(tr("不處理，直接訓練", "no treatment, train as is")), "MUT", 16, False, "left", w=560)
        for j, v in enumerate((m["rec"], m["prec"], m["acc"])):
            x = 660 + j * 150
            f.rect(x, y + 18, 20, 70 * 0 + 10, fill="transparent") if False else None
            h = max(v * 0.72, 3)
            f.rect(x + 30, y + 90 - h, 60, h, fill=solid(role), radius=4)
            f.text(x + 60, y + 90 - h - 14, f"{h1(v)}%", "TXT", 17, True, w=100)
    takeaway(f, 0, 398, 1160, 50, T(tr("稀有類找回來了，但精確率掉到約 10%，準確率掉到 85% 到 88%：召回換來了誤報。", "The rare class is found, but precision falls to about 10% and accuracy to 85-88%: recall bought with false alarms.")))
    return f


def focus():
    ks = [("1", 20), ("3", 60), ("5", 100), ("10", 200)]
    pts_rec = [(i + 1, M(C["focus"][k])["rec"]) for i, (k, m) in enumerate(ks)] + [(5, M(C["plain_svm"])["rec"])]
    pts_acc = [(i + 1, M(C["focus"][k])["acc"]) for i, (k, m) in enumerate(ks)] + [(5, M(C["plain_svm"])["acc"])]
    fp20 = C["focus"]["1"]["fp"]; fpu = C["under_svm"]["fp"]
    spec = {
        "id": "lec05-kp26-focus",
        "alt": tr(f"聚焦欠採樣：先訓練一個 SVM，對 1000 個多數點算決策值，留下最像稀有類的 m 個，再與 20 個稀有點重新訓練。m=20、60、100、200、1000（一般）：召回 " + "、".join(h1(v) + "%" for _, v in pts_rec) + "，準確率 " + "、".join(h1(v) + "%" for _, v in pts_acc) + f"。留 20 個時誤報 {fp20:g} 個，比隨機欠採樣的 {fpu:g} 個還多。目標分布調到多少才對是未解問題（來源）。",
                  f"Focused undersampling: train an SVM first, compute decision values for the 1000 majority points, keep the m that look most like the rare class and retrain with the 20 rare points. For m=20, 60, 100, 200, 1000 (plain): recall " + ", ".join(h1(v) + "%" for _, v in pts_rec) + ", accuracy " + ", ".join(h1(v) + "%" for _, v in pts_acc) + f". Keeping 20 gives {fp20:g} false alarms, more than the {fpu:g} of random undersampling. How far the target distribution should be shifted is an open problem (source)."),
        "plot": {"x": [0.5, 6.4], "y": [0, 112], "x_ticks": [1, 2, 3, 4, 5], "x_names": {"1": "20", "2": "60", "3": "100", "4": "200", "5": "1000"}, "y_ticks": [0, 25, 50, 75, 100], "x_label": tr("留下的多數點數 m", "majority points kept, m"), "y_label": "%", "w": 430, "h": 300},
        "series": [{"kind": "line", "points": pts_rec, "role": "green", "width": 4, "markers": True, "labels": [h1(v) for _, v in pts_rec], "label_dy": -18},
                   {"kind": "line", "points": pts_acc, "role": "blue", "width": 4, "markers": True, "labels": [h1(v) for _, v in pts_acc], "label_dy": -18}],
        "notes": [{"x": 5, "y": 40, "text": tr("一般 SVM", "plain SVM"), "ha": "center", "w": 110}, {"x": 5.15, "y": pts_rec[-1][1] + 12, "text": tr("召回", "recall"), "ha": "left", "w": 90}, {"x": 5.15, "y": pts_acc[-1][1] - 9, "text": tr("準確率", "accuracy"), "ha": "left", "w": 90}],
        "side": [{"title": tr("做法", "Method"), "role": "blue", "h": 1.35, "text": tr("1. 先訓練一個 SVM（$C=1$）。\n2. 對 1000 個多數點算決策值，\n留下最像稀有類的 $m$ 個。\n3. 與 20 個稀有點重新訓練。", "1. train an SVM first ($C=1$).\n2. compute decision values of the 1000\nmajority points, keep the $m$ most rare-like.\n3. retrain with the 20 rare points.")},
                 {"title": tr(f"留 20 個時", "Keeping 20"), "role": "red", "h": 1.0, "text": tr(f"誤報 {fp20:g} 個，比隨機欠採樣的\n{fpu:g} 個還多；留 20 與留 60 差很多。", f"{fp20:g} false alarms, more than the {fpu:g}\nof random undersampling; 20 and 60 differ a lot.")},
                 {"title": tr("未解問題", "Open problem"), "role": "orange", "h": 1.0, "text": tr("目標分布調到多少才對，\n沒有現成答案（來源）。", "how far to shift the target distribution\nhas no ready answer (source).")}],
        "takeaway": tr("留越少，召回越高，但誤報暴增、準確率掉到 58%；留多少很難定。", "Fewer kept, higher recall, but false alarms explode and accuracy drops to 58%; m is hard to set."),
    }
    return "plot_xy", spec


def costs():
    items = [(tr("SVM，一般", "SVM, plain"), M(C["plain_svm"]), "grey"), (tr("SVM，少數類 C 為 50 倍", "SVM, rare-class C ×50"), M(C["cost_svm"]), "green"),
             (tr("AdaBoost，w=1", "AdaBoost, w=1"), M(C["ada"]["1"]), "blue"), (tr("AdaBoost，w=10", "AdaBoost, w=10"), M(C["ada"]["10"]), "blue"), (tr("AdaBoost，w=50", "AdaBoost, w=50"), M(C["ada"]["50"]), "blue")]
    spec = {
        "id": "lec05-kp26-costs",
        "alt": tr("改代價：SVM 少數類的 C 設為多數類的 50 倍（權重 25.5 對 0.51）；AdaBoost 100 個決策樁，少數類初始權重乘 w=1、10、50。召回（精確率）：" + "；".join(f"{T(n, 'zh')} {h1(m['rec'])}%（{h1(m['prec'])}%）" for n, m, _ in items) + "。改代價和改資料一樣，TP 與 FP 一起升、精確率下降。",
                  "Changing the costs: the SVM's C for the rare class is 50 times that of the majority (weights 25.5 against 0.51); AdaBoost with 100 stumps multiplies the initial weight of the rare class by w=1, 10, 50. Recall (precision): " + "; ".join(f"{T(n, 'en')} {h1(m['rec'])}% ({h1(m['prec'])}%)" for n, m, _ in items) + ". As with changing the data, TP and FP rise together and precision falls."),
        "unit": tr("稀有類召回（%）", "recall of the rare class (%)"),
        "bars": [{"label": n, "value": round(m["rec"] + 1e-9, 1), "role": r, "suffix": "%", "note": T(tr(f"精確率 {h1(m['prec'])}%", f"precision {h1(m['prec'])}%"))} for n, m, r in items],
        "max": 100, "ticks": [0, 20, 40, 60, 80, 100],
        "takeaway": tr("改代價和改資料一樣：召回上升，精確率下降。", "Like changing the data: recall rises and precision falls."),
    }
    return "bars_compare", spec


def remember():
    spec = {
        "id": "lec05-kp26-remember",
        "alt": tr("不平衡的傷害不是來自「數字不一樣」，而是來自「簡單性偏好 × 稀有類」的交互作用。對策兩條：改資料（重抽樣：過採樣、欠採樣、聚焦欠採樣）或改代價（SVM 類別權重、AdaBoost 初始權重）。常見迷思：準確率高就夠了；全判多數也有 98.0%，評估要看召回與精確率。類別不平衡在實務上非常重要，設計者必須意識到它並警覺各種對策（來源）。",
                  "The harm of imbalance does not come from the numbers being different but from the interaction of the preference for simplicity with the rare class. Two remedies: change the data (resampling: over-, under- and focused undersampling) or change the costs (SVM class weights, AdaBoost initial weights). Common myth: high accuracy is enough; all-majority also has 98.0%, so look at recall and precision. Class imbalance matters a lot in practice, and designers must be aware of it and of the remedies (source)."),
        "term": tr("類別不平衡：傷害從哪裡來，怎麼對付", "Class imbalance: where the harm comes from and what to do"), "name": False,
        "definition": tr("傷害不是來自「數字不一樣」，而是來自「簡單性偏好 × 稀有類」的交互作用。", "The harm comes not from different numbers but from the preference for simplicity meeting a rare class."),
        "parts": [{"label": tr("改資料", "Data"), "role": "green", "text": tr("重抽樣：過採樣、欠採樣、\n聚焦欠採樣。召回升，精確率降。", "resampling: over-, under- and\nfocused undersampling. Recall up, precision down.")},
                  {"label": tr("改代價", "Costs"), "role": "blue", "text": tr("加權：SVM 的類別權重、\nAdaBoost 的初始權重。\n召回升，精確率降。", "weights: the SVM's class weights,\nAdaBoost's initial weights.\nRecall up, precision down.")},
                  {"label": tr("常見迷思", "Myth"), "role": "red", "text": tr("「準確率高就夠了」。\n全判多數也有 98.0%；\n要看召回與精確率。", "\"High accuracy is enough\".\nAll-majority also has 98.0%;\nlook at recall and precision.")}],
        "takeaway": tr("類別不平衡在實務上非常重要：要意識到它，並警覺各種對策（來源）。", "Class imbalance matters a lot in practice: be aware of it and of the remedies (source)."),
    }
    return "concept_card", spec


def compare():
    rows = [("svm", tr("一般", "plain"), M(C["plain_svm"])), ("svm", tr("過採樣", "over"), M(C["over_svm"])), ("svm", tr("欠採樣", "under"), M(C["under_svm"])), ("svm", tr("代價敏感", "cost"), M(C["cost_svm"])),
            ("tree", tr("一般", "plain"), M(C["plain_tree"])), ("tree", tr("過採樣", "over"), M(C["over_tree"])), ("tree", tr("欠採樣", "under"), M(C["under_tree"])), ("tree", tr("代價敏感", "cost"), M(C["cost_tree"]))]
    svm = [r for r in rows if r[0] == "svm"]; tre = [r for r in rows if r[0] == "tree"]
    fm = lambda rs: "、".join(f"{h1(r[2]['bal']*100)[:-2] if False else format(r[2]['bal'], '.2f')}" for r in rs)
    spec = {
        "id": "lec05-kp26-compare",
        "alt": tr("八種處理的召回（橫軸）與精確率（縱軸）：SVM 一般、過採樣、欠採樣、代價敏感，決策樹（每片葉至少 5 點）同樣四種。召回越高精確率越低；平衡準確率 SVM " + fm(svm) + "，決策樹 " + fm(tre) + "。沒有誰明顯贏，差別是取捨。",
                  "Recall (horizontal) against precision (vertical) for eight treatments: SVM plain, oversampled, undersampled, cost-sensitive, and the same four for a decision tree (at least 5 points per leaf). The higher the recall, the lower the precision; balanced accuracy for the SVM is " + fm(svm) + " and for the tree " + fm(tre) + ". No one clearly wins: the difference is a trade-off."),
        "plot": {"x": [0, 95], "y": [0, 85], "x_ticks": [0, 20, 40, 60, 80], "y_ticks": [0, 20, 40, 60, 80], "x_label": tr("召回（%）", "recall (%)"), "y_label": tr("精確率（%）", "precision (%)"), "w": 430, "h": 300},
        "series": [{"kind": "dots", "points": [(r[2]["rec"], r[2]["prec"]) for r in svm], "role": "blue", "d": 16, "name": "SVM"},
                   {"kind": "dots", "points": [(r[2]["rec"], r[2]["prec"]) for r in tre], "role": "orange", "d": 16, "shape": "square", "name": tr("決策樹", "tree")}],
        "notes": [{"x": svm[0][2]["rec"] + 4, "y": svm[0][2]["prec"] + 4, "text": tr("SVM 一般", "SVM plain"), "ha": "left", "w": 110}, {"x": tre[0][2]["rec"] + 4, "y": tre[0][2]["prec"] + 4, "text": tr("樹 一般", "tree plain"), "ha": "left", "w": 110},
                  {"x": tre[1][2]["rec"] + 4, "y": tre[1][2]["prec"] + 4, "text": tr("樹 過採樣", "tree over"), "ha": "left", "w": 110}, {"x": tre[3][2]["rec"] + 4, "y": tre[3][2]["prec"] + 4, "text": tr("樹 代價敏感", "tree cost"), "ha": "left", "w": 120},
                  {"x": 66, "y": 60, "text": tr("SVM 的過採樣、欠採樣、代價敏感\n與樹的欠採樣擠在右下", "SVM over, under, cost-sensitive and\nthe tree's undersampling crowd bottom right"), "ha": "center", "w": 300}],
        "side": [{"title": tr("特異度與平衡準確率", "Specificity and balanced accuracy"), "role": "blue", "name": tr("term:平衡準確率", "term:Balanced accuracy"), "h": 1.35, "text": tr("特異度＝多數點被判對的比例，TN/5000。\n平衡準確率＝（召回＋特異度）/ 2；\n全判多數 $(0+1)/2=0.50$。", "specificity = share of majority points right, TN/5000.\nbalanced accuracy = (recall + specificity)/2;\nall-majority gives $(0+1)/2=0.50$.")},
                 {"title": tr("平衡準確率", "Balanced accuracy"), "role": "green", "h": 1.15, "text": tr("SVM：" + fm(svm) + "\n決策樹：" + fm(tre) + "\n（依序：一般、過採樣、欠採樣、代價敏感）", "SVM: " + fm(svm) + "\ntree: " + fm(tre) + "\n(in order: plain, over, under, cost-sensitive)")},
                 {"title": tr("代價敏感", "Cost-sensitive"), "role": "orange", "name": tr("term:代價敏感", "term:Cost-sensitive"), "h": 1.0, "text": tr("不動資料，改分類器的代價\n（類別權重、初始權重）。", "leave the data alone and change the\nclassifier's costs (class or initial weights).")}],
        "takeaway": tr("沒有誰明顯贏，差別是召回與精確率的取捨。", "No one clearly wins: the difference is the trade-off between recall and precision."),
    }
    return "plot_xy", spec


def solution():
    a_acc = 1000 / 1010; b_acc = 966 / 1010; b_prec = 6 / 46; b_spec = 960 / 1000; b_bal = (0.6 + b_spec) / 2
    spec = {
        "id": "lec05-kp26-solution",
        "alt": tr(f"1000 個負例、10 個正例。A（全判負）：TP 0、FN 10、FP 0、TN 1000，準確率 1000/1010={a_acc:.3f}、召回 0、平衡準確率 (0+1)/2=0.5。B（抓到 6 個、誤報 40 個）：TP 6、FN 4、FP 40、TN 960，準確率 966/1010={b_acc:.3f}、召回 6/10=0.6、特異度 960/1000={b_spec:.2f}、平衡準確率 (0.6+0.96)/2={b_bal:.2f}；B 的精確率只有 6/46={b_prec:.2f}，這就是取捨。",
                  f"1000 negatives and 10 positives. A (all negative): TP 0, FN 10, FP 0, TN 1000, accuracy 1000/1010={a_acc:.3f}, recall 0, balanced accuracy (0+1)/2=0.5. B (6 caught, 40 false alarms): TP 6, FN 4, FP 40, TN 960, accuracy 966/1010={b_acc:.3f}, recall 6/10=0.6, specificity 960/1000={b_spec:.2f}, balanced accuracy (0.6+0.96)/2={b_bal:.2f}; B's precision is only 6/46={b_prec:.2f}, which is the trade-off."),
        "steps": [{"role": "blue", "title": tr("① A：全判負", "① A: all negative"), "lines": ["TP 0、FN 10、FP 0、TN 1000", f"$1000/1010={a_acc:.3f}$", tr("召回 0", "recall 0")], "result": tr("平衡準確率 0.5", "balanced acc. 0.5")},
                  {"role": "green", "title": tr("② B：抓 6、誤報 40", "② B: 6 caught, 40 false"), "lines": ["TP 6、FN 4、FP 40、TN 960", f"$966/1010={b_acc:.3f}$", tr("召回 $6/10=0.6$", "recall $6/10=0.6$")], "result": tr(f"平衡準確率 {b_bal:.2f}", f"balanced acc. {b_bal:.2f}")},
                  {"role": "orange", "title": tr("③ 核對", "③ Check"), "lines": [tr("A 的準確率比 B 高，", "A's accuracy beats B's,"), tr("卻一個稀有點也沒抓到", "yet catches no rare point"), tr(f"B 的精確率 $6/46={b_prec:.2f}$", f"B's precision $6/46={b_prec:.2f}$")], "result": tr("這就是取捨", "this is the trade-off")}],
        "takeaway": tr("準確率高的 A 一個稀有點也沒抓到；B 抓到 6 成，代價是只有 13% 的警報是真的。", "A, with the higher accuracy, catches no rare point; B catches 60% at the cost that only 13% of its alarms are real."),
    }
    return "steps_flow", spec
