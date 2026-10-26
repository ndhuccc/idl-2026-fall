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
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import kp23_nums as N
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


def _nums():
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import kp23_nums as N
    return N


def entropy_curve():
    ps = [i / 40 for i in range(1, 40)]
    pts = [(0, 0)] + [(p, H([p, 1 - p])) for p in ps] + [(1, 0)]
    spec = {
        "id": "lec05-kp23-entropy",
        "alt": tr("兩類時的不純度（熵）曲線：橫軸是第一類的比例 p，縱軸是 I(t)；兩類各占一半（p=0.5）時最大，I=1；全是同一類（p=0 或 1）時 I=0。公式 I(t)=−Σ P(ωᵢ|t) log₂ P(ωᵢ|t)。",
                  "Two-class impurity (entropy) curve: the horizontal axis is the share p of class 1, the vertical axis is I(t); it is largest at p=0.5 (equal classes) with I=1 and 0 when all points are one class (p=0 or 1). Formula I(t)=−Σ P(ωᵢ|t) log₂ P(ωᵢ|t)."),
        "plot": {"x": [0, 1], "y": [0, 1.14], "x_ticks": [0, 0.25, 0.5, 0.75, 1], "y_ticks": [0, 0.5, 1], "x_label": tr("第一類的比例 p", "share p of class 1"), "y_label": tr("不純度 I", "impurity I"), "w": 470, "h": 300},
        "series": [{"kind": "vline", "x": 0.5, "role": "grey", "width": 1.5}, {"kind": "line", "points": pts, "role": "orange", "width": 4}],
        "notes": [{"x": 0.5, "y": 1.08, "text": tr("各占一半：I = 1（最大）", "equal halves: I = 1 (largest)"), "ha": "center"}, {"x": 0.2, "y": 0.1, "text": tr("全是同一類：I = 0", "one class: I = 0"), "ha": "center"}],
        "side": [
            {"title": tr("不純度", "Impurity"), "role": "blue", "name": tr("term:不純度", "term:Impurity"), "text": tr("各類的點都有叫不純，\n全是同一類叫純。\n$I(t)=-\\sum_{i=1}^{M}P(\\omega_i|t)\\log_2P(\\omega_i|t)$", "several classes: impure;\none class only: pure.\n$I(t)=-\\sum_{i=1}^{M}P(\\omega_i|t)\\log_2P(\\omega_i|t)$")},
            {"title": tr("符號", "Symbols"), "role": "green", "text": tr("$t$ 節點，$M$ 類別數，$N_t$ 節點內點數，\n$N_t^i$ 第 $i$ 類點數，$P(\\omega_i|t)=N_t^i/N_t$", "$t$ node, $M$ classes, $N_t$ points in node,\n$N_t^i$ points of class $i$, $P(\\omega_i|t)=N_t^i/N_t$")},
            {"title": tr("性質", "Properties"), "role": "orange", "text": tr("各類一樣多最大（$\\log_2M$）；\n全是同一類是 0（約定 $0\\log0=0$）", "largest when classes are equal ($\\log_2M$);\n0 when all one class ($0\\log0=0$)")}],
        "takeaway": tr("越平均越混亂：看的是各類的比例，不是總點數。", "The more even the classes, the more mixed: it is the shares that count, not the total."),
    }
    return "plot_xy", spec


def prune_chart():
    from sklearn.tree import DecisionTreeClassifier
    N = _nums(); Xtr, ytr = N.moons2(100, 1); Xte, yte = N.moons2(1000, 2)
    als = (0.0, 0.01, 0.02, 0.05); rows = []
    for a in als:
        t = DecisionTreeClassifier(criterion="entropy", ccp_alpha=a, random_state=0).fit(Xtr, ytr)
        rows.append((a, int(t.get_n_leaves()), int(round(t.score(Xtr, ytr) * 200)), int(round(t.score(Xte, yte) * 2000))))
    best = max(range(4), key=lambda i: rows[i][3])
    names = {str(i): tr(f"α={a:g}\n{l} 片葉", f"α={a:g}\n{l} leaves") for i, (a, l, _, _) in enumerate(rows)}
    spec = {
        "id": "lec05-kp23-prune",
        "alt": tr("剪枝強度 α 與準確率：" + "；".join(f"α={a:g}：{l} 片葉，訓練 {t}/200={t/2:.1f}%，新點 {n}/2000={n/20:.1f}%" for a, l, t, n in rows) + "。訓練分數下降、新資料反而進步。",
                  "Pruning strength α and accuracy: " + "; ".join(f"α={a:g}: {l} leaves, training {t}/200={t/2:.1f}%, new points {n}/2000={n/20:.1f}%" for a, l, t, n in rows) + ". Training falls while new data improves."),
        "plot": {"x": [-0.4, 3.4], "y": [80, 102], "x_ticks": [0, 1, 2, 3], "x_names": names, "y_ticks": [80, 85, 90, 95, 100], "x_label": tr("剪枝強度 α（剩下的葉數）", "pruning strength α (leaves left)"), "y_label": tr("準確率（%）", "accuracy (%)"), "w": 560, "h": 300},
        "series": [{"kind": "line", "points": [(i, r[2] / 2) for i, r in enumerate(rows)], "role": "blue", "markers": True, "labels": [("100%" if r[2] == 200 else f"{r[2] / 2:.1f}%") for r in rows], "label_dy": -22, "name": tr("訓練", "training")},
                   {"kind": "line", "points": [(i, r[3] / 20) for i, r in enumerate(rows)], "role": "red", "markers": True, "labels": [f"{r[3] / 20:.1f}%" for r in rows], "label_dy": 24, "name": tr("新資料", "new data")}],
        "side": [{"title": tr("剪枝", "Pruning"), "role": "orange", "name": tr("term:剪枝", "term:Pruning"), "text": tr("先長成大樹，再剪掉\n不值得的分支；α 越大剪得越兇。", "grow a big tree, then cut the\nbranches not worth keeping;\na larger α prunes harder.")},
                 {"title": tr("怎麼讀", "How to read"), "role": "blue", "text": tr("同一份 200 個訓練點、\n2000 個新點，只改 α。", "Same 200 training and 2000\nnew points, only α changes.")},
                 {"title": tr("判對數", "Right counts"), "role": "green", "text": "\n".join(tr(f"α={a:g}：訓練 {t}/200，新點 {n}/2000", f"α={a:g}: training {t}/200, new {n}/2000")["zh" if True else "en"] for a, l, t, n in rows) if False else tr("\n".join(f"α={a:g}：訓練 {t}/200，新點 {n}/2000" for a, l, t, n in rows), "\n".join(f"α={a:g}: training {t}/200, new {n}/2000" for a, l, t, n in rows))}],
        "takeaway": tr(f"訓練分數下降，新資料反而進步：{rows[0][3]/20:.1f}% → {rows[best][3]/20:.1f}%（{rows[best][1]} 片葉）。", f"Training falls, new data improves: {rows[0][3]/20:.1f}% → {rows[best][3]/20:.1f}% ({rows[best][1]} leaves)."),
    }
    return "plot_xy", spec


def oblique_chart():
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    rng = np.random.default_rng(5); X = rng.uniform(-1, 1, (300, 2)); y = (X[:, 0] + X[:, 1] > 0).astype(int)
    Xt = rng.uniform(-1, 1, (3000, 2)); yt = (Xt[:, 0] + Xt[:, 1] > 0).astype(int)
    rows = []
    for d in (1, 3, 6, None):
        t = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=0).fit(X, y)
        rows.append((d, int(t.get_n_leaves()), int(round(t.score(Xt, yt) * 3000))))
    pts0 = [(float(a), float(b)) for (a, b), c in zip(X, y) if c == 0]; pts1 = [(float(a), float(b)) for (a, b), c in zip(X, y) if c == 1]
    lab = lambda d: tr("不限深度" if d is None else f"深度 {d}", "full tree" if d is None else f"depth {d}")
    zh = "\n".join(f"{lab(d)['zh']}：{l} 片葉，新點 {c}/3000＝{c/30:.1f}%" for d, l, c in rows)
    en = "\n".join(f"{lab(d)['en']}: {l} leaves, new {c}/3000 = {c/30:.1f}%" for d, l, c in rows)
    spec = {
        "id": "lec05-kp23-oblique",
        "alt": tr("類別由斜線 x₁+x₂=0 決定（300 個訓練點）：一個斜向問題就完全分對；軸平行的樹要用階梯逼近：" + "；".join(f"{lab(d)['zh']} {l} 片葉、新點 {c}/3000={c/30:.1f}%" for d, l, c in rows) + "。",
                  "The class is set by the oblique line x₁+x₂=0 (300 training points): one oblique question separates them exactly; axis-parallel trees need a staircase: " + "; ".join(f"{lab(d)['en']} {l} leaves, new {c}/3000={c/30:.1f}%" for d, l, c in rows) + "."),
        "plot": {"x": [-1, 1], "y": [-1, 1], "x_ticks": [-1, 0, 1], "y_ticks": [-1, 0, 1], "x_label": "x₁", "y_label": "x₂", "w": 300, "h": 300},
        "series": [{"kind": "dots", "points": pts0, "role": "blue", "d": 9, "shape": "circle"}, {"kind": "dots", "points": pts1, "role": "orange", "d": 8, "shape": "square"},
                   {"kind": "seg", "a": (-1, 1), "b": (1, -1), "role": "green", "width": 4}],
        "notes": [{"x": 0.62, "y": -0.78, "text": tr("x₁ + x₂ = 0", "x₁ + x₂ = 0"), "color": "TXT"}],
        "side": [{"title": tr("斜向問題一刀就夠", "One oblique question is enough"), "role": "green", "text": tr("$x_1+x_2\\leq 0$？\n切得完全正確", "$x_1+x_2\\leq 0$ ?\nexact split")},
                 {"title": tr("軸平行的樹要階梯逼近", "Axis-parallel trees need a staircase"), "role": "red", "text": tr(zh, en), "math": False}],
        "takeaway": tr("要 13 到 14 片葉才到約 95%；斜向問題切得好，但訓練更複雜。", "13 to 14 leaves are needed for about 95%; oblique questions cut well but are harder to train."),
    }
    return "plot_xy", spec


def variance_chart():
    from sklearn.tree import DecisionTreeClassifier
    N = _nums(); X, yy = N.moons2(100, 40)
    thr = []; leaves = []
    for s in range(8):
        Xs, ys = N.moons2(100, 40 + s); t = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(Xs, ys); thr.append(float(t.tree_.threshold[0])); leaves.append(int(t.get_n_leaves()))
    pts0 = [(float(a), float(b)) for (a, b), c in zip(X, yy) if c == 0]; pts1 = [(float(a), float(b)) for (a, b), c in zip(X, yy) if c == 1]
    srt = sorted(zip(thr, range(40, 48)))
    spec = {
        "id": "lec05-kp23-var",
        "alt": tr("8 份訓練資料（種子 40 到 47）各長一棵不限深度的樹，第一刀都切第二個特徵，門檻：" + "、".join(f"{t:.2f}" for t in thr) + f"，在 {min(thr):.2f} 到 {max(thr):.2f} 之間；葉數 {min(leaves)} 到 {max(leaves)}。圖上的點是種子 40 那份資料，水平線是各棵樹的第一刀。",
                  "Trees without depth limit grown on 8 training sets (seeds 40 to 47) all cut the second feature first, with thresholds " + ", ".join(f"{t:.2f}" for t in thr) + f", between {min(thr):.2f} and {max(thr):.2f}; leaves {min(leaves)} to {max(leaves)}. The points are training set 40, the horizontal lines the first cuts of the trees."),
        "plot": {"x": [-1.5, 2.5], "y": [-0.4, 0.8], "x_ticks": [-1, 0, 1, 2], "y_ticks": [-0.25, 0, 0.25, 0.5, 0.75], "x_label": tr("特徵 1", "feature 1"), "y_label": tr("特徵 2", "feature 2"), "w": 560, "h": 300},
        "series": [{"kind": "dots", "points": pts0, "role": "blue", "d": 8, "shape": "circle"}, {"kind": "dots", "points": pts1, "role": "orange", "d": 7, "shape": "square"}] +
                  [{"kind": "hline", "y": t, "role": "red", "width": 2} for t in thr],
        "side": [{"title": tr("高變異", "High variance"), "role": "red", "name": tr("term:高變異", "term:High variance"), "text": tr("訓練資料稍微不同，\n長出來的樹就很不同。", "slightly different training data\ngrow very different trees.")},
                 {"title": tr("8 棵樹的第一刀", "First cut of 8 trees"), "role": "blue", "text": tr("都切第二個特徵，門檻在\n" + f"{min(thr):.2f} 到 {max(thr):.2f} 之間\n（紅線；種子 40 到 47）", "all on feature 2, thresholds between\n" + f"{min(thr):.2f} and {max(thr):.2f}\n(red lines; seeds 40 to 47)")},
                 {"title": tr("葉數", "Leaves"), "role": "orange", "text": tr(f"{min(leaves)} 到 {max(leaves)} 片葉；\n樹是階層式的，高處的錯\n會傳到下面所有葉。", f"{min(leaves)} to {max(leaves)} leaves;\na tree is hierarchical, so an\nerror high up spreads below.")}],
        "takeaway": tr("門檻在 " + f"{min(thr):.2f} 到 {max(thr):.2f}" + " 之間跳動：換一份資料，樹就長得不一樣。", f"The threshold jumps between {min(thr):.2f} and {max(thr):.2f}: a new training set grows a different tree."),
    }
    return "plot_xy", spec
