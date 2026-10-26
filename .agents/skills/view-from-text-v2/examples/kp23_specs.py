"""KP23 (decision trees) pages expressed as layout specs -- the acceptance sample of view-from-text-v2.
Each function returns (layout_name, spec). Numbers are computed here, never typed; strings are {"zh","en"} pairs.
Use:  from figkit import layouts;  f = layouts.build(*impurity_examples());  f.to_nodes(page, 60, y, "dark");  f.to_png(path)"""
import numpy as np


def H(c):
    c = np.asarray(c, float); p = c[c > 0] / c.sum(); return float(-(p * np.log2(p)).sum())


def tr(zh, en): return {"zh": zh, "en": en}


LEG2 = [("circle", tr("圓：第一類", "circle: class 1")), ("square", tr("方：第二類", "square: class 2"))]


def impurity_examples():
    cnts = ((4, 4, 2), (1, 1, 1), (5, 0, 0)); Is = [H(c) for c in cnts]
    t = [-(c / 10) * np.log2(c / 10) for c in cnts[0]]
    spec = {
        "id": "lec05-kp23-impfull",
        "alt": tr(f"三個節點的不純度：(4,4,2) 為 {t[0]:.3f}+{t[1]:.3f}+{t[2]:.3f}={Is[0]:.3f}；(1,1,1) 為 log₂3={Is[1]:.3f}（三類時的最大值）；(5,0,0) 為 0（完全純）；公式 I(t)=−Σ pᵢ log₂ pᵢ，pᵢ＝該類點數÷節點總點數；越平均不純度越高。",
                  f"Impurity of three nodes: (4,4,2) is {t[0]:.3f}+{t[1]:.3f}+{t[2]:.3f}={Is[0]:.3f}; (1,1,1) is log₂3={Is[1]:.3f} (the maximum for three classes); (5,0,0) is 0 (pure); formula I(t)=−Σ pᵢ log₂ pᵢ with pᵢ = class points ÷ node points; the more even, the larger."),
        "legend": [("circle", tr("第一類", "class 1")), ("square", tr("第二類", "class 2")), ("tri", tr("第三類", "class 3"))],
        "columns": [
            {"role": "red", "header": tr("三類混合：(4,4,2)", "Mixed: (4,4,2)"), "icons": [("circle", 4), ("square", 4), ("tri", 2)],
             "lines": [tr("共 10 點", "10 points"), tr("比例 0.4、0.4、0.2", "p = 0.4, 0.4, 0.2")],
             "eq": ["$I=-(2\\times0.4\\log_2 0.4+0.2\\log_2 0.2)$", f"$\\approx {t[0]:.3f}+{t[1]:.3f}+{t[2]:.3f}$"], "result": f"$I\\approx {Is[0]:.3f}$",
             "comment": tr("有混合，但三類比例不完全相同", "mixed, but the shares differ")},
            {"role": "blue", "header": tr("三類一樣多：(1,1,1)", "Equal: (1,1,1)"), "icons": [("circle", 1), ("square", 1), ("tri", 1)], "icons_mode": "line",
             "lines": [tr("共 3 點", "3 points"), tr("比例 1/3、1/3、1/3", "p = 1/3, 1/3, 1/3")],
             "eq": ["$I=-3\\times\\frac{1}{3}\\log_2\\frac{1}{3}=\\log_2 3$"], "result": f"$I\\approx {Is[1]:.3f}$",
             "comment": tr("每類機會相同，最難猜下一點", "equal shares: hardest to guess")},
            {"role": "green", "header": tr("全是同一類：(5,0,0)", "One class: (5,0,0)"), "icons": [("circle", 5)], "icons_mode": "line",
             "lines": [tr("共 5 點", "5 points"), tr("比例 1、0、0", "p = 1, 0, 0")],
             "eq": ["$I=-1\\times\\log_2 1=0$"], "result": "$I=0$",
             "comment": tr("每一點都屬同一類，沒有混雜", "every point is one class: no mixing"), "note": tr("約定：$0\\log_2 0=0$", "convention: $0\\log_2 0=0$")}],
        "formula": "$I(t)=-\\sum_i p_i\\log_2 p_i$",
        "formula_caption": tr("$p_i$＝該類點數 ÷ 節點總點數；$t$ 是節點", "$p_i$ = points of class $i$ ÷ points in node $t$"),
        "takeaway": tr("看的是各類的比例，不是總點數：三類越平均，不純度越高。", "It is the class shares, not the total: the more even the classes, the larger the impurity."),
    }
    return "columns_compare", spec


def weighting():
    I = lambda c: H(c)
    IA, IB = I((4, 5)), I((4, 1))
    dwA = 1 - (0.1 * 0 + 0.9 * IA); dwB = 1 - (0.5 * IB + 0.5 * IB); duA = 1 - (0 + IA) / 2; duB = 1 - (IB + IB) / 2
    spec = {
        "id": "lec05-kp23-wfull",
        "alt": tr(f"為什麼要按點數加權：父節點 10 點（5 圓 5 方）I=1。分裂 A 切出 1 個純點（I=0）與 9 點（I={IA:.3f}）；分裂 B 切成兩堆各 5 點（I={IB:.3f}）。按點數加權：A 的 ΔI={dwA:.3f}，B 的 ΔI={dwB:.3f}，選 B；直接平均：A 的 ΔI={duA:.3f}，B 的 ΔI={duB:.3f}，誤選 A。",
                  f"Why weight by points: the parent has 10 points (5 circles, 5 squares), I=1. Split A isolates 1 pure point (I=0) and 9 points (I={IA:.3f}); split B makes two piles of 5 points (I={IB:.3f}). Weighted: ΔI of A={dwA:.3f}, of B={dwB:.3f}, pick B; plain average: A={duA:.3f}, B={duB:.3f}, wrongly picks A."),
        "parent": {"rows": [("circle", 5), ("square", 5)], "lines": [tr("原本：10 個點", "Before: 10 points"), tr("不純度 $I=1$", "impurity $I=1$")]},
        "legend": LEG2,
        "options": [
            {"role": "red", "header": tr("分裂 A：只分出 1 個純點", "Split A: isolates 1 pure point"),
             "children": [{"rows": [("circle", 1)], "label": tr("1 點：I = 0", "1 pt: I = 0")}, {"rows": [("circle", 4), ("square", 5)], "label": tr(f"9 點：I = {IA:.3f}", f"9 pts: I = {IA:.3f}")}],
             "comment": tr("只有 1 個點變純，其餘 9 個仍很混雜", "only 1 point became pure; 9 are still mixed")},
            {"role": "blue", "header": tr("分裂 B：兩堆都比較純", "Split B: both piles purer"),
             "children": [{"rows": [("circle", 4), ("square", 1)], "label": tr(f"5 點：I = {IB:.3f}", f"5 pts: I = {IB:.3f}")}, {"rows": [("circle", 1), ("square", 4)], "label": tr(f"5 點：I = {IB:.3f}", f"5 pts: I = {IB:.3f}")}],
             "comment": tr("改善涵蓋全部 10 個點", "the gain covers all 10 points")}],
        "rows": [
            {"role": "green", "label": [tr("正確：", "Correct:"), tr("按點數加權", "weight by points")],
             "formulas": [f"$\\Delta I=1-(\\frac{{1}}{{10}}\\times0+\\frac{{9}}{{10}}\\times{IA:.3f})\\approx{dwA:.3f}$", f"$\\Delta I=1-(\\frac{{5}}{{10}}\\times{IB:.3f}+\\frac{{5}}{{10}}\\times{IB:.3f})\\approx{dwB:.3f}$"],
             "chip": {"text": tr("選 B ✓", "pick B ✓"), "role": "green"}},
            {"role": "blue", "label": [tr("直接平均：", "Plain average:"), tr("兩堆一樣重要", "both piles equal")],
             "formulas": [f"$\\Delta I=1-\\frac{{0+{IA:.3f}}}{{2}}\\approx{duA:.3f}$", f"$\\Delta I=1-\\frac{{{IB:.3f}+{IB:.3f}}}{{2}}\\approx{duB:.3f}$"],
             "chip": {"text": tr("誤選 A", "picks A ✗"), "role": "red"}}],
        "takeaway": tr("1 個點的小堆，不能和 9 個點的大堆各占一半！　ΔI 越大，代表分裂後整體改善越多。", "A pile of 1 point cannot count as much as a pile of 9!　The larger ΔI, the more the split helps overall."),
    }
    return "two_options", spec


def tree_size():
    import os, sys
    from sklearn.tree import DecisionTreeClassifier
    sys.path.insert(0, os.environ.get("KP23_NUMS", "."))
    import nums23 as N
    Xtr, ytr = N.moons2(100, 1); Xte, yte = N.moons2(1000, 2)
    R = []
    for d in (1, None):
        t = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=0).fit(Xtr, ytr)
        R.append((t.get_depth(), t.get_n_leaves(), int(round(t.score(Xtr, ytr) * 200)), int(round(t.score(Xte, yte) * 2000))))
    pc = lambda a, b: "100%" if a == b else f"{a / b * 100:.1f}%"
    (d1, n1, tr1, te1), (d2, n2, tr2, te2) = R
    g1, g2 = tr1 / 200 * 100 - te1 / 20, tr2 / 200 * 100 - te2 / 20
    stats = lambda a, b: [{"label": tr("訓練：", "training: "), "value": pc(a, 200), "sub": tr(f"{a} / 200 判對", f"{a} / 200 right"), "role": "blue"},
                           {"label": tr("新資料：", "new data: "), "value": pc(b, 2000), "sub": tr(f"{b} / 2000 判對", f"{b} / 2000 right"), "role": "orange"}]
    spec = {
        "id": "lec05-kp23-dfull",
        "alt": tr(f"樹越大越好嗎：淺樹（深度 {d1}，{n1} 片葉）訓練 {tr1}/200＝{pc(tr1,200)}、新資料 {te1}/2000＝{pc(te1,2000)}，差距 {g1:.1f} 個百分點；深樹（深度 {d2}，{n2} 片葉）訓練 {tr2}/200＝{pc(tr2,200)}、新資料 {te2}/2000＝{pc(te2,2000)}，差距 {g2:.1f} 個百分點。過擬合：訓練很高、新資料不再進步。",
                  f"Is a bigger tree better: shallow tree (depth {d1}, {n1} leaves) training {tr1}/200 = {pc(tr1,200)}, new data {te1}/2000 = {pc(te1,2000)}, gap {g1:.1f} points; deep tree (depth {d2}, {n2} leaves) training {tr2}/200 = {pc(tr2,200)}, new data {te2}/2000 = {pc(te2,2000)}, gap {g2:.1f} points. Overfitting: training high, new data no better."),
        "panels": [
            {"role": "blue", "header": tr("淺樹：問題少、規則粗", "Shallow tree: few questions, coarse rules"),
             "sketch": [("tree_shallow", {"cx": 280, "y": 58, "root_text": tr("問題", "question"), "leaf_text": tr("分類", "class")})],
             "caption": tr(f"深度 {d1}｜{n1} 個葉節點", f"depth {d1} | {n1} leaves"), "stats": stats(tr1, te1), "gap": tr(f"兩者差距：{g1:.1f} 個百分點", f"gap: {g1:.1f} percentage points")},
            {"role": "red", "header": tr("深樹：問題多、細節也記住", "Deep tree: many questions, details memorised"),
             "sketch": [("tree_deep", {"cx": 760, "y": 50}), ("magnifier_note", {"cx": 1000, "cy": 86, "text_lines": [tr("連偶然的雜訊，", "even chance noise"), tr("也可能被記成規則", "can be memorised")]})],
             "caption": tr(f"深度 {d2}｜{n2} 個葉節點", f"depth {d2} | {n2} leaves"), "stats": stats(tr2, te2), "gap": tr(f"兩者差距：{g2:.1f} 個百分點", f"gap: {g2:.1f} percentage points")}],
        "boxes": [
            {"kind": "text", "x": 0, "w": 440, "title": tr("什麼是過擬合？", "What is overfitting?"), "title_w": 190,
             "lines": [tr("模型把訓練資料的細節也學進去，", "It memorises training details:"), tr("訓練分數很高，新資料卻不再進步。", "training score high, new data no better.")], "name": tr("term:過擬合", "term:Overfitting")},
            {"kind": "pair", "x": 456, "w": 284, "left": tr("熟題 ✓✓✓", "seen ✓✓✓"), "right": tr("新題 ???", "new ???"), "caption": tr("熟題全對，不代表新題也全對", "all seen right ≠ all new right")},
            {"kind": "text", "x": 756, "w": 404, "title": tr("名詞小教室", "Glossary"), "title_w": 150,
             "lines": [tr("深度：沿一條路徑，最多連問幾次", "depth: most questions along one path"), tr("葉節點：最後給出分類結果的位置", "leaf: where a class is finally given")]}],
        "takeaway": tr("樹繼續長大，訓練準確率升到 100%；新資料表現卻停在約 85%。", "As the tree grows, training accuracy reaches 100%, but new data stalls near 85%."),
        "footnote": tr("單次實驗：含雜訊的雙月牙資料，訓練 200 點、新資料 2000 點；不代表所有資料。", "One experiment: noisy two-moon data, 200 training and 2000 new points; not all data."),
    }
    return "before_after", spec
